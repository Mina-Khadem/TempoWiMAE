"""Masked-reconstruction pretraining of TempoWiMAE.

Example::

    python train.py --data_root /path/to/wifo            # paper configuration (config.PRETRAIN)
    python train.py --data_root /path/to/wifo --epochs 25
    python train.py --data_root /path/to/wifo --set model.attn_mode=full
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader

from input_preprocess import CSIDataset
from masking import MaskRatioCurriculum, build_batch_mask, pick_pattern
from tempowimae_model import TempoWiMAEConfig, build_model, save_checkpoint
from utils import Tee, TensorBoardLogger, apply_overrides, dump_config, join_path, resolve_device


# ==============================================================================
# pretraining: Masked-reconstruction pretraining of TempoWiMAE.
# ==============================================================================

# Patterns whose ratio may be drawn per batch when ``masking.temporal_ratio_random`` is on.
TEMPORAL_PATTERNS = ("temporal_frame", "temporal", "temporal_tail")


def build_patch_targets(videos: torch.Tensor, bool_masked_pos: torch.Tensor, patch_size, tubelet_size: int,
                        normalize_target: bool = True) -> torch.Tensor:
    """Reconstruction targets of the masked tokens, ``[B, N_masked, p_t*p_h*p_w*C]``.

    With ``normalize_target`` every tubelet is standardised over its ``p_t*p_h*p_w``
    positions (separately for the real and imaginary channel) before the loss.
    """
    ph, pw = int(patch_size[0]), int(patch_size[1])
    if normalize_target:
        sq = rearrange(videos, "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
                       p0=tubelet_size, p1=ph, p2=pw)
        sq = (sq - sq.mean(dim=-2, keepdim=True)) / (sq.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        patch = rearrange(sq, "b n p c -> b n (p c)")
    else:
        patch = rearrange(videos, "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
                          p0=tubelet_size, p1=ph, p2=pw)
    batch_size, _, channels = patch.shape
    return patch[bool_masked_pos].reshape(batch_size, -1, channels)


def reconstruction_metrics(outputs: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8):
    """Return ``(mse, nmse, nmse_db)``; NMSE is the per-sample ratio averaged over the batch."""
    err = outputs - labels
    se = err.pow(2)
    mse = se.mean()
    num = se.flatten(1).sum(dim=1)
    den = labels.pow(2).flatten(1).sum(dim=1).clamp_min(eps)
    nmse = (num / den).mean()
    nmse_db = 10.0 * torch.log10(nmse.clamp_min(eps))
    return mse, nmse, nmse_db


def learning_rate_at(step: int, steps_per_epoch: int, opt_cfg: Dict[str, Any]) -> float:
    """Linear warm-up from ``warmup_lr`` to ``lr``, then cosine decay to ``min_lr`` (per step)."""
    warmup_steps = int(opt_cfg["warmup_epochs"]) * steps_per_epoch
    total_steps = int(opt_cfg["epochs"]) * steps_per_epoch
    lr, warmup_lr, min_lr = float(opt_cfg["lr"]), float(opt_cfg["warmup_lr"]), float(opt_cfg["min_lr"])
    if step < warmup_steps and warmup_steps > 0:
        return warmup_lr + (lr - warmup_lr) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def sample_mask_ratio(pattern: str, base_ratio: float, mask_scale: float, rng,
                      temporal_random: bool = False, temporal_range=None) -> float:
    """Mask ratio of one batch: the pattern's base ratio (optionally drawn per batch for
    temporal patterns) times the curriculum scale, clipped to ``[0, 0.95]``."""
    base = float(base_ratio)
    if temporal_random and pattern in TEMPORAL_PATTERNS and temporal_range:
        lo, hi = float(temporal_range[0]), float(temporal_range[1])
        base = float(rng.uniform(lo, hi))
    return float(np.clip(base * mask_scale, 0.0, 0.95))


def evaluate(model, loader, device, patch_size, tubelet_size, normalize_target, amp_dtype, use_amp,
             token_grid, patterns, weights, ratios, mask_scale=1.0, max_batches=0,
             temporal_ratio_random=False, temporal_ratio_range=None) -> Dict[str, float]:
    """Validation reconstruction error under the training masking distribution.

    Patterns and ratios are drawn with a fixed ``RandomState(0)`` so the estimate is
    comparable across epochs.
    """
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "nmse": 0.0, "nmse_db": 0.0, "n": 0}
    rng_val = np.random.RandomState(0)
    with torch.no_grad():
        for batch_index, videos in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            videos = videos.to(device, non_blocking=True)
            batch_size = videos.shape[0]
            pattern = pick_pattern(patterns, weights, rng_val)
            ratio = sample_mask_ratio(pattern, ratios[pattern], mask_scale, rng_val,
                                      temporal_ratio_random, temporal_ratio_range)
            mask = torch.from_numpy(build_batch_mask(token_grid, pattern, ratio, batch_size)).to(device).bool()
            labels = build_patch_targets(videos, mask, patch_size, tubelet_size, normalize_target)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(videos, mask)
                mse, nmse, nmse_db = reconstruction_metrics(outputs, labels)
            totals["loss"] += mse.item()
            totals["mse"] += mse.item()
            totals["nmse"] += nmse.item()
            totals["nmse_db"] += nmse_db.item()
            totals["n"] += 1
    model.train()
    n = max(totals["n"], 1)
    return {"loss": totals["loss"] / n, "mse": totals["mse"] / n,
            "nmse": totals["nmse"] / n, "nmse_db": totals["nmse_db"] / n, "num_batches": totals["n"]}


def _write_history(history: Dict[str, list], run_dir: Path, interrupted: bool = False) -> None:
    payload = dict(history)
    payload["interrupted"] = interrupted
    with open(run_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_pretraining(cfg: Dict[str, Any], device_spec: Optional[str] = None) -> Path:
    """Pretrain TempoWiMAE with the resolved configuration ``cfg`` and return the run directory.

    The run directory ``<logging.output_dir>/<experiment_name>/<timestamp>/`` receives
    ``config.json`` (resolved configuration), ``run_config.json`` (geometry, parameter
    counts, sample counts), ``log.txt``, ``training_history.json``, TensorBoard events
    and ``checkpoint-<epoch>.pth`` files.
    """
    data_cfg, geom = cfg["data"], cfg["geometry"]
    model_cfg_dict, mask_cfg = cfg["model"], cfg["masking"]
    opt_cfg, log_cfg = cfg["optimization"], cfg["logging"]

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_spec or cfg.get("device"))
    rng = np.random.RandomState(seed)

    height, width, num_frames = int(geom["subcarriers"]), int(geom["antennas"]), int(geom["num_frames"])
    patch_h, patch_w = (int(v) for v in geom["patch_size"])
    tubelet_size = int(geom["tubelet_size"])
    in_chans = int(geom.get("in_chans", 2))
    if num_frames % tubelet_size or height % patch_h or width % patch_w:
        raise ValueError("geometry: num_frames must be divisible by tubelet_size and the grid by patch_size.")
    token_grid = (num_frames // tubelet_size, height // patch_h, width // patch_w)

    model_config = TempoWiMAEConfig.from_dict(model_cfg_dict).validate()
    size = str(model_cfg_dict.get("size", "nano"))

    patterns = list(mask_cfg["patterns"])
    weights = np.asarray(mask_cfg["weights"], dtype=float)
    if len(weights) != len(patterns):
        raise ValueError("masking.patterns and masking.weights must have the same length.")
    weights = weights / weights.sum()
    ratios = {str(k): float(v) for k, v in mask_cfg["ratios"].items()}
    missing = [p for p in patterns if p not in ratios]
    if missing:
        raise ValueError(f"masking.ratios has no entry for {missing}.")
    normalize_target = bool(mask_cfg.get("normalize_target", True))
    temporal_ratio_random = bool(mask_cfg.get("temporal_ratio_random", False))
    temporal_ratio_range = mask_cfg.get("temporal_ratio_range")

    run_dir = Path(log_cfg["output_dir"]) / str(cfg["experiment_name"]) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(run_dir / "log.txt", "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    try:
        model = build_model(size, input_size=(height, width), patch_size=(patch_h, patch_w),
                            tubelet_size=tubelet_size, num_frames=num_frames, in_chans=in_chans,
                            model_config=model_config).to(device)
        n_total = sum(p.numel() for p in model.parameters())
        n_encoder = sum(p.numel() for p in model.encoder.parameters())

        root = data_cfg["root"]
        train_paths = [join_path(root, f) for f in data_cfg["train_files"]]
        train_dataset = CSIDataset(train_paths, int(data_cfg.get("max_samples_per_file", 0) or 0))
        num_workers = int(data_cfg.get("num_workers", 4))
        batch_size = int(opt_cfg["batch_size"])
        if len(train_dataset) < batch_size:
            raise ValueError(f"Only {len(train_dataset)} training samples for batch_size={batch_size}.")
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True, pin_memory=True)
        steps_per_epoch = len(loader)

        val_paths = [join_path(root, f) for f in data_cfg.get("val_files", []) or []]
        val_paths = [p for p in val_paths if os.path.exists(p)]
        val_loader = None
        val_dataset = None
        if val_paths:
            val_dataset = CSIDataset(val_paths, int(data_cfg.get("max_val_samples_per_file", 0) or 0))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, drop_last=False, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]), betas=tuple(opt_cfg["betas"]),
                                      eps=float(opt_cfg["eps"]), weight_decay=float(opt_cfg["weight_decay"]))

        curriculum = None
        curr_cfg = mask_cfg.get("curriculum") or {}
        if curr_cfg.get("enable", False):
            curriculum = MaskRatioCurriculum(start=curr_cfg["start_scale"], end=1.0,
                                             warmup_epochs=curr_cfg["warmup_epochs"], shape=curr_cfg.get("shape", "cosine"))

        use_amp = device.type == "cuda" and bool(opt_cfg.get("amp", True))
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        tb = None
        if log_cfg.get("tensorboard", True):
            tb = TensorBoardLogger(log_dir=str(run_dir / "tensorboard"))
            if not tb.enabled:
                tb = None

        dump_config(cfg, run_dir / "config.json")
        run_info = {
            "model_size": size, "model_config": model_config.to_dict(), "token_grid": list(token_grid),
            "parameters_total": n_total, "parameters_encoder": n_encoder, "device": str(device),
            "num_train_samples": len(train_dataset), "num_val_samples": len(val_dataset) if val_dataset else 0,
            "steps_per_epoch": steps_per_epoch, "amp_dtype": str(amp_dtype) if use_amp else "float32",
        }
        with open(run_dir / "run_config.json", "w", encoding="utf-8") as handle:
            json.dump(run_info, handle, indent=2)

        print("=" * 60)
        print(f"  TempoWiMAE pretraining  |  {cfg['experiment_name']}")
        print("=" * 60)
        print(f"[run] {run_dir}")
        print(f"[run] model={size} params={n_total/1e6:.3f}M (encoder {n_encoder/1e6:.3f}M) token_grid={token_grid}")
        print(f"[run] samples={len(train_dataset)} val={len(val_dataset) if val_dataset else 0} "
              f"steps/epoch={steps_per_epoch} device={device}")
        print(f"[run] architecture={model_config.to_dict()}")
        print(f"[run] masking patterns={patterns} weights={np.round(weights, 3).tolist()} ratios={ratios}")

        epochs = int(opt_cfg["epochs"])
        monitor_every = int(log_cfg.get("monitor_every", 1))
        monitor_batches = int(log_cfg.get("monitor_batches", 0))
        log_interval = int(log_cfg.get("log_interval", 50))
        save_ckpt_freq = int(log_cfg.get("save_ckpt_freq", 25))
        clip_grad = float(opt_cfg.get("clip_grad", 0) or 0)

        history: Dict[str, list] = {}
        run_start = time.time()
        interrupted = False

        try:
            for epoch in range(epochs):
                model.train()
                mask_scale = curriculum.value_at(epoch) if curriculum else 1.0
                agg = {"loss": 0.0, "mse": 0.0, "nmse": 0.0, "nmse_db": 0.0, "grad_norm": 0.0, "lr": 0.0, "n": 0}
                epoch_start = time.time()

                for step, videos in enumerate(loader):
                    global_step = epoch * steps_per_epoch + step
                    cur_lr = learning_rate_at(global_step, steps_per_epoch, opt_cfg)
                    for group in optimizer.param_groups:
                        group["lr"] = cur_lr

                    videos = videos.to(device, non_blocking=True)
                    current_batch = videos.shape[0]

                    # Hybrid masking: one pattern per batch, applied to every sample of the batch.
                    pattern = pick_pattern(patterns, weights, rng)
                    ratio = sample_mask_ratio(pattern, ratios[pattern], mask_scale, rng,
                                              temporal_ratio_random, temporal_ratio_range)
                    mask = torch.from_numpy(build_batch_mask(token_grid, pattern, ratio, current_batch)).to(device).bool()

                    with torch.no_grad():
                        labels = build_patch_targets(videos, mask, (patch_h, patch_w), tubelet_size, normalize_target)
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        outputs = model(videos, mask)
                        mse, nmse, nmse_db = reconstruction_metrics(outputs, labels)
                        loss = mse

                    if not math.isfinite(loss.item()):
                        print(f"[stop] non-finite loss at epoch {epoch} step {step}")
                        interrupted = True
                        break

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = 0.0
                    if clip_grad:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad))
                    optimizer.step()

                    agg["loss"] += loss.item()
                    agg["mse"] += mse.item()
                    agg["nmse"] += nmse.item()
                    agg["nmse_db"] += nmse_db.item()
                    agg["grad_norm"] += grad_norm
                    agg["lr"] += cur_lr
                    agg["n"] += 1

                    if step % log_interval == 0:
                        print(f"  Epoch: [{epoch}] [{step:4d}/{steps_per_epoch}]  lr={cur_lr:.2e}  "
                              f"pattern={pattern}  ratio={ratio:.2f}  mse={loss.item():.5f}  "
                              f"nmse={nmse.item():.5f} ({nmse_db.item():.2f} dB)  grad_norm={grad_norm:.3f}")

                    if tb is not None:
                        tb.set_step(global_step)
                        tb.update(head="train", loss=loss.item(), nmse=nmse.item(),
                                  nmse_db=nmse_db.item(), lr=cur_lr, grad_norm=grad_norm)

                if interrupted:
                    break

                n = max(agg["n"], 1)
                train_stats = {"loss": agg["loss"] / n, "mse": agg["mse"] / n, "nmse": agg["nmse"] / n,
                               "nmse_db": agg["nmse_db"] / n, "lr": agg["lr"] / n, "grad_norm": agg["grad_norm"] / n}
                epoch_time = time.time() - epoch_start

                val_stats = None
                if val_loader and (epoch % monitor_every == 0 or epoch + 1 == epochs):
                    val_stats = evaluate(model, val_loader, device, (patch_h, patch_w), tubelet_size,
                                         normalize_target, amp_dtype, use_amp, token_grid,
                                         patterns, weights, ratios, mask_scale=mask_scale,
                                         max_batches=monitor_batches,
                                         temporal_ratio_random=temporal_ratio_random,
                                         temporal_ratio_range=temporal_ratio_range)
                    print(f"  [val]  nmse={val_stats['nmse']:.5f} ({val_stats['nmse_db']:.2f} dB)")

                history.setdefault("epoch", []).append(epoch)
                for key in ("loss", "mse", "nmse", "nmse_db", "lr", "grad_norm"):
                    history.setdefault(f"train_{key}", []).append(float(train_stats[key]))
                for key in ("loss", "nmse", "nmse_db"):
                    history.setdefault(f"validation_{key}", []).append(None if val_stats is None else float(val_stats[key]))
                history.setdefault("mask_scale", []).append(float(mask_scale))
                history.setdefault("epoch_time_seconds", []).append(float(epoch_time))
                _write_history(history, run_dir)

                val_text = (f"  val_nmse={val_stats['nmse']:.5f} ({val_stats['nmse_db']:.2f} dB)" if val_stats else "")
                print(f"[epoch {epoch}]  mse={train_stats['loss']:.5f}  nmse={train_stats['nmse']:.5f} "
                      f"({train_stats['nmse_db']:.2f} dB){val_text}  lr={train_stats['lr']:.2e}  "
                      f"grad={train_stats['grad_norm']:.3f}  scale={mask_scale:.3f}  ({epoch_time:.1f}s)")

                if tb is not None:
                    tb.set_step(epoch)
                    tb.update(head="epoch/train", loss=train_stats["loss"], nmse=train_stats["nmse"],
                              nmse_db=train_stats["nmse_db"], lr=train_stats["lr"], grad_norm=train_stats["grad_norm"])
                    if val_stats:
                        tb.update(head="epoch/val", nmse=val_stats["nmse"], nmse_db=val_stats["nmse_db"])

                is_final = (epoch + 1 == epochs)
                if is_final or (save_ckpt_freq > 0 and (epoch + 1) % save_ckpt_freq == 0):
                    ckpt_path = run_dir / f"checkpoint-{epoch}.pth"
                    save_checkpoint(ckpt_path, model, epoch, model_config, cfg)
                    print(f"[ckpt] wrote {ckpt_path}")
        except KeyboardInterrupt:
            print("\n[interrupt] saving partial state ...")
            interrupted = True
            last_epoch = history["epoch"][-1] if history.get("epoch") else -1
            ckpt_path = run_dir / "checkpoint-last.pth"
            save_checkpoint(ckpt_path, model, last_epoch, model_config, cfg)
            print(f"[ckpt] wrote {ckpt_path} (epoch {last_epoch})")

        _write_history(history, run_dir, interrupted=interrupted)
        if tb is not None:
            tb.flush()
        total_time = time.time() - run_start
        print(f"\n[done] {run_dir}  (total {total_time/60:.1f} min)")
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_handle.close()
    return run_dir


# ==============================================================================
# command line
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain TempoWiMAE by masked reconstruction of CSI sequences. "
                    "Defaults are the paper configuration in config.PRETRAIN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default=None,
                        help="Folder containing D4/, D7/, D12/, D16/ (overrides config and $TEMPOWIMAE_DATA_ROOT).")
    parser.add_argument("--output_dir", default=None, help="Overrides logging.output_dir.")
    parser.add_argument("--experiment_name", default=None, help="Overrides experiment_name.")
    parser.add_argument("--epochs", type=int, default=None, help="Overrides optimization.epochs (paper: 200; matched budget: 25).")
    parser.add_argument("--batch_size", type=int, default=None, help="Overrides optimization.batch_size.")
    parser.add_argument("--seed", type=int, default=None, help="Overrides seed.")
    parser.add_argument("--max_samples_per_file", type=int, default=None,
                        help="Use only the first n samples of every file (quick check); 0 = all.")
    parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if available).")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config.PRETRAIN entry, e.g. --set model.size=tiny "
                             "--set model.attn_mode=full --set masking.patterns=[tube] --set masking.weights=[1.0].")
    return parser


def main(argv=None) -> None:
    import config as paper_config

    args = build_parser().parse_args(argv)
    overrides = list(args.overrides)
    if args.output_dir is not None:
        overrides.append(f"logging.output_dir={args.output_dir}")
    if args.experiment_name is not None:
        overrides.append(f"experiment_name={args.experiment_name}")
    if args.epochs is not None:
        overrides.append(f"optimization.epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"optimization.batch_size={args.batch_size}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.max_samples_per_file is not None:
        overrides.append(f"data.max_samples_per_file={args.max_samples_per_file}")
    cfg = apply_overrides(paper_config.PRETRAIN, overrides)
    root = args.data_root or os.environ.get("TEMPOWIMAE_DATA_ROOT") or cfg["data"].get("root")
    if not root:
        raise SystemExit("No data root given: pass --data_root, set $TEMPOWIMAE_DATA_ROOT or fill config.PRETRAIN['data']['root'].")
    cfg["data"]["root"] = root
    run_pretraining(cfg, device_spec=args.device)


if __name__ == "__main__":
    main()
