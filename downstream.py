"""Frozen-encoder downstream evaluation: channel estimation and channel prediction.

Example::

    python downstream.py --task channel_estimation --data_dir data/channel_estimation
    python downstream.py --task channel_prediction --setting 3_to_1 --data_dir data/prediction_3_to_1
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from input_preprocess import build_downstream_datasets
from tempowimae_model import TempoWiMAEConfig, build_model, checkpoint_model_config, load_pretrained_encoder, read_checkpoint
from utils import apply_overrides, normalize_csi_input, resolve_device, resolve_input_normalization, set_seed, unpatchify_csi


# ==============================================================================
# tasks: Downstream task registry: inputs, targets, losses and metrics.
# ==============================================================================

def _per_sample_nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of ``||pred - target||^2 / ||target||^2`` (all non-batch dims pooled)."""
    diff = torch.abs(prediction - target) ** 2
    power = torch.abs(target) ** 2
    nmse = (
        torch.sum(diff, dim=tuple(range(1, diff.ndim)))
        / torch.sum(power, dim=tuple(range(1, power.ndim))).clamp_min(1e-8)
    )
    return nmse.mean()


class ChannelEstimationLoss(nn.Module):
    """MSE between the estimated and the full CSI tensor."""

    def forward(self, prediction: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.mse_loss(prediction, batch["h_full"])


class ChannelPredictionLoss(nn.Module):
    """Per-sample NMSE over the predicted snapshots."""

    def forward(self, prediction: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return _per_sample_nmse(prediction, batch["target_future"])


def channel_estimation_metric(prediction: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return _per_sample_nmse(prediction, batch["h_full"])


def channel_prediction_metric(prediction: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return _per_sample_nmse(prediction, batch["target_future"])


DOWNSTREAM_TASKS = {
    "channel_estimation": {
        "display_name": "Channel estimation",
        "input_key": "h_inter",          # pilot observations interpolated to the full grid
        "target_key": "h_full",
        "loss_fn": ChannelEstimationLoss,
        "metric_fn": channel_estimation_metric,
        "metric_name": "NMSE",
        "metric_mode": "min",
    },
    "channel_prediction": {
        "display_name": "Channel prediction",
        "input_key": "h_full",           # observed snapshots followed by zeroed future slots
        "target_key": "target_future",
        "loss_fn": ChannelPredictionLoss,
        "metric_fn": channel_prediction_metric,
        "metric_name": "NMSE",
        "metric_mode": "min",
    },
}


# ==============================================================================
# heads: Lightweight downstream heads applied per token on top of the (frozen) encoder.
# ==============================================================================

READOUT_MODES = ("end_pad", "causal_pad", "none")


def prediction_window(observed_snapshots: int, horizon: int, tubelet_size: int,
                      readout: str = "end_pad") -> Tuple[int, int, int]:
    """Window geometry of a prediction task: ``(num_frames, pad_front, horizon_pad)``.

    * ``end_pad`` (paper): the observed snapshots are followed by the horizon, zero-extended
      at the window end to the next tubelet multiple, so the target snapshots may share a
      token with observed ones.
    * ``causal_pad``: zeros are prepended so that the first target snapshot starts a fresh
      token (the target never shares a token with an observation).
    * ``none``: window = observed + horizon, which must itself be a tubelet multiple.
    """
    t = int(tubelet_size)
    tpast, hor = int(observed_snapshots), int(horizon)
    if readout == "end_pad":
        pad_front = 0
        hor_pad = ((tpast + hor + t - 1) // t) * t - tpast
    elif readout == "causal_pad":
        pad_front = (t - (tpast % t)) % t
        hor_pad = ((hor + t - 1) // t) * t
    elif readout == "none":
        pad_front = 0
        hor_pad = hor
        if (tpast + hor) % t != 0:
            raise ValueError(
                f"readout 'none' needs observed_snapshots + horizon ({tpast + hor}) divisible by "
                f"tubelet_size ({t}); use readout 'end_pad' or 'causal_pad'."
            )
    else:
        raise ValueError(f"Unknown readout '{readout}'. Choose from {READOUT_MODES}.")
    return pad_front + tpast + hor_pad, pad_front, hor_pad


def build_channel_pred_mlp(feature_dim: int, patch_dim: int, mlp_layers: int = 2) -> nn.Sequential:
    """Per-token MLP mapping an encoder token to its reconstructed tubelet values.

    ``mlp_layers``: 1 = linear probe, 2 = ``Linear(d, 2d) -> ReLU -> Linear(2d, patch_dim)``
    (paper), 3 = two hidden layers of width ``2d``.
    """
    if int(mlp_layers) == 1:
        return nn.Sequential(nn.Linear(feature_dim, patch_dim))
    if int(mlp_layers) == 3:
        return nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, patch_dim),
        )
    return nn.Sequential(
        nn.Linear(feature_dim, 2 * feature_dim),
        nn.ReLU(),
        nn.Linear(2 * feature_dim, patch_dim),
    )


class ChannelEstimationHead(nn.Module):
    """Two-layer per-token MLP that maps every encoder token to its full-CSI tubelet."""

    def __init__(self, feature_dim: int, channels: int, num_frames: int, height: int, width: int,
                 patch_size, tubelet_size: int, num_tokens: int):
        super().__init__()
        self.channels = int(channels)
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.tubelet_size = int(tubelet_size)
        self.num_tokens = int(num_tokens)
        self.patch_dim = self.channels * self.tubelet_size * self.patch_size[0] * self.patch_size[1]
        self.fine_tune_layer = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, self.patch_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """``[B, N, D]`` tokens -> estimated CSI ``[B, C, T, H, W]``."""
        if embeddings.ndim != 3 or embeddings.shape[1] != self.num_tokens:
            raise ValueError(f"Expected token embeddings [B, {self.num_tokens}, D], got {tuple(embeddings.shape)}")
        patch_tokens = self.fine_tune_layer(embeddings)
        return unpatchify_csi(
            patch_tokens, channels=self.channels, num_frames=self.num_frames, height=self.height,
            width=self.width, patch_size=self.patch_size, tubelet_size=self.tubelet_size,
        )


class ChannelPredictionHead(nn.Module):
    """Per-token MLP that reconstructs the full window and returns the target snapshots.

    The observed snapshots are placed in a zero-padded window (see :func:`prediction_window`),
    encoded with all tokens visible, mapped back to CSI values and sliced at the target
    positions. The head is the only trainable part when the encoder is frozen.
    """

    def __init__(self, feature_dim: int, channels: int, window_frames: int, height: int, width: int,
                 patch_size, tubelet_size: int, observed_snapshots: int, horizon: int,
                 mlp_layers: int = 2, readout: str = "end_pad"):
        super().__init__()
        self.channels = int(channels)
        self.window_frames = int(window_frames)
        self.height = int(height)
        self.width = int(width)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.tubelet_size = int(tubelet_size)
        self.observed_snapshots = int(observed_snapshots)
        self.horizon = int(horizon)
        self.readout = str(readout)
        expected, self._pad_front, self._hor_pad = prediction_window(
            self.observed_snapshots, self.horizon, self.tubelet_size, self.readout)
        if self.window_frames != expected:
            raise ValueError(
                f"readout '{self.readout}': window_frames ({self.window_frames}) must equal {expected} "
                f"for observed_snapshots={self.observed_snapshots}, horizon={self.horizon}, "
                f"tubelet_size={self.tubelet_size}."
            )
        self.patch_dim = self.channels * self.tubelet_size * self.patch_size[0] * self.patch_size[1]
        self.head = build_channel_pred_mlp(feature_dim, self.patch_dim, int(mlp_layers))

    def forward(self, backbone, sequence: torch.Tensor) -> torch.Tensor:
        """``sequence`` ``[B, C, >=T_p, H, W]`` -> predicted snapshots ``[B, C, h, H, W]``."""
        if sequence.ndim != 5:
            raise ValueError(f"Expected a 5D input (batch, channels, frames, height, width), got {tuple(sequence.shape)}.")
        if sequence.shape[2] < self.observed_snapshots:
            raise ValueError(f"Need at least {self.observed_snapshots} observed snapshots, got {sequence.shape[2]}.")
        observed = sequence[:, :, :self.observed_snapshots, :, :]
        b, c, _, h, w = observed.shape
        pad_front = observed.new_zeros(b, c, self._pad_front, h, w)
        future = observed.new_zeros(b, c, self._hor_pad, h, w)
        window = torch.cat([pad_front, observed, future], dim=2)
        predicted_tokens = self.head(backbone.extract_token_embeddings(window))
        reconstructed = unpatchify_csi(
            predicted_tokens, channels=self.channels, num_frames=self.window_frames,
            height=self.height, width=self.width, patch_size=self.patch_size, tubelet_size=self.tubelet_size,
        )
        start = self._pad_front + self.observed_snapshots
        return reconstructed[:, :, start:start + self.horizon, :, :]


# ==============================================================================
# backbone: Pretrained TempoWiMAE encoder used as a (frozen) feature extractor downstream.
# ==============================================================================

def _validate_geometry(input_size, patch_size, num_frames, tubelet_size) -> None:
    height, width = int(input_size[0]), int(input_size[1])
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    num_frames, tubelet_size = int(num_frames), int(tubelet_size)
    errors = []
    if height < patch_h or width < patch_w:
        errors.append(f"input size {height}x{width} is smaller than the patch {patch_h}x{patch_w}")
    if num_frames < tubelet_size:
        errors.append(f"num_frames {num_frames} is smaller than tubelet_size {tubelet_size}")
    if height % patch_h or width % patch_w:
        errors.append(f"input size {height}x{width} is not divisible by the patch {patch_h}x{patch_w}")
    if num_frames % tubelet_size:
        errors.append(f"num_frames {num_frames} is not divisible by tubelet_size {tubelet_size}")
    if errors:
        raise ValueError("Invalid downstream geometry: " + "; ".join(errors) + ".")


def _normalize_finetune_patterns(patterns: Iterable[str]) -> list:
    """Map user-facing ``layer9`` / ``block9`` to the parameter prefix ``blocks.9``."""
    normalized = []
    for pattern in patterns:
        text = str(pattern).strip()
        match = re.fullmatch(r"(?:layer|block)s?\.?(\d+)", text.lower().replace(" ", ""))
        normalized.append(f"blocks.{int(match.group(1))}" if match else text)
    return normalized


class EncoderBackbone(nn.Module):
    """TempoWiMAE encoder rebuilt from a checkpoint for a downstream geometry.

    Args:
        checkpoint_path: pretraining checkpoint (``None`` keeps random initialisation,
            e.g. for a supervised-from-scratch control with ``finetune_mode="full"``).
        size: model size, must match the checkpoint.
        input_size, patch_size, num_frames, tubelet_size, in_chans: downstream geometry.
        finetune_mode: ``"none"`` (frozen, paper), ``"partial"`` (unfreeze
            ``finetune_layers``) or ``"full"``.
        input_normalization / input_normalization_scope: per-sample normalisation applied
            before the encoder (``"rms"`` global for channel estimation, ``"none"`` for prediction).
        model_config: ``"auto"`` reads the architecture options stored in the checkpoint;
            a mapping or :class:`TempoWiMAEConfig` overrides them.
    """

    def __init__(self, checkpoint_path: Optional[str], size: str = "nano", input_size=(32, 32),
                 patch_size=(4, 4), num_frames: int = 4, tubelet_size: int = 4, in_chans: int = 2,
                 finetune_mode: str = "none", finetune_layers: Optional[Iterable[str]] = None,
                 input_normalization: str = "none", input_normalization_scope: str = "global",
                 model_config: Any = "auto"):
        super().__init__()
        self.finetune_mode = finetune_mode
        self.finetune_layers = _normalize_finetune_patterns(finetune_layers or [])
        self.input_size = tuple(int(v) for v in input_size)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.in_chans = int(in_chans)
        self.input_normalization, self.input_normalization_scope = resolve_input_normalization(
            input_normalization, input_normalization_scope)
        _validate_geometry(self.input_size, self.patch_size, self.num_frames, self.tubelet_size)

        checkpoint = read_checkpoint(checkpoint_path) if checkpoint_path else None
        if isinstance(model_config, (dict, TempoWiMAEConfig)):
            resolved = TempoWiMAEConfig.from_dict(model_config) if isinstance(model_config, dict) else model_config
        elif checkpoint is not None:
            resolved = checkpoint_model_config(checkpoint)
            if resolved is None:
                raise ValueError(
                    "The checkpoint stores no architecture options; pass model_config explicitly "
                    "(attn_mode, pos_embed_mode, ...)."
                )
        else:
            resolved = TempoWiMAEConfig()
        self.model_config = resolved.validate()
        print(f"[tempowimae] building encoder with architecture {self.model_config.to_dict()}")

        pretrain_model = build_model(
            size, input_size=self.input_size, patch_size=self.patch_size, tubelet_size=self.tubelet_size,
            num_frames=self.num_frames, in_chans=self.in_chans, model_config=self.model_config,
        )
        self.load_info = load_pretrained_encoder(pretrain_model, checkpoint) if checkpoint is not None else None
        if checkpoint is None:
            print("[tempowimae] no checkpoint given: the encoder keeps its random initialisation")
        self.encoder = pretrain_model.encoder
        self.embed_dim = self.encoder.embed_dim
        self.num_tokens = self.encoder.patch_embed.num_patches
        self._configure_finetuning()

    def _configure_finetuning(self) -> None:
        mode = self.finetune_mode
        if mode not in {"none", "partial", "full"}:
            raise ValueError(f"Unsupported finetune_mode: {mode}. Use 'none', 'partial' or 'full'.")
        if mode == "full":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
            self.encoder_trainable = True
            return
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if mode == "partial":
            matched = []
            for name, parameter in self.encoder.named_parameters():
                if any(pattern in name for pattern in self.finetune_layers):
                    parameter.requires_grad = True
                    matched.append(name)
            if not matched:
                print(f"Partial fine-tuning requested, but no encoder parameters matched {self.finetune_layers}")
            else:
                print(f"Partially fine-tuning {len(matched)} encoder parameters, e.g. {', '.join(matched[:5])}")
        self.encoder_trainable = any(p.requires_grad for p in self.encoder.parameters())

    def _apply_input_normalization(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalization == "none":
            return x
        if self.input_normalization_scope == "global":
            reduce_dims = (1, 2, 3, 4)
            if self.input_normalization == "standardized":
                mean = x.mean(dim=reduce_dims, keepdim=True)
                variance = x.var(dim=reduce_dims, unbiased=False, keepdim=True)
                return (x - mean) / torch.sqrt(variance + 1e-6)
            if self.input_normalization == "rms":
                rms = torch.sqrt(x.pow(2).mean(dim=reduce_dims, keepdim=True) + 1e-6)
                return x / rms
        normalized_samples = [
            normalize_csi_input(sample, normalization_type=self.input_normalization,
                                normalization_scope=self.input_normalization_scope,
                                patch_size=self.patch_size, tubelet_size=self.tubelet_size)
            for sample in x
        ]
        return torch.stack(normalized_samples, dim=0)

    def extract_token_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a CSI tensor ``[B, C, T, H, W]`` with every token visible -> ``[B, N, D]``."""
        if x.ndim != 5:
            raise ValueError(f"Expected a 5D CSI input (batch, channels, frames, height, width), got {tuple(x.shape)}.")
        actual = (int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
        expected = (self.num_frames, self.input_size[0], self.input_size[1])
        if actual != expected:
            raise ValueError(
                f"CSI shape does not match the backbone geometry: expected (frames, height, width)={expected}, "
                f"got {actual}. Check geometry / prediction settings against the dataset."
            )
        x = self._apply_input_normalization(x)
        no_mask = torch.zeros((x.shape[0], self.num_tokens), device=x.device, dtype=torch.bool)
        if not self.encoder_trainable:
            with torch.no_grad():
                return self.encoder.forward_features(x, no_mask)
        return self.encoder.forward_features(x, no_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_token_embeddings(x)


# ==============================================================================
# model: Backbone + task head wrapper and its builder.
# ==============================================================================

class DownstreamModel(nn.Module):
    """Route a batch through the encoder backbone and the task head."""

    def __init__(self, backbone: EncoderBackbone, head: nn.Module, task: str):
        super().__init__()
        if task not in DOWNSTREAM_TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {list(DOWNSTREAM_TASKS)}.")
        self.backbone = backbone
        self.head = head
        self.task = task
        self.input_key = DOWNSTREAM_TASKS[task]["input_key"]

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        csi_input = batch[self.input_key]
        if self.task == "channel_prediction":
            return self.head(self.backbone, csi_input)
        return self.head(self.backbone.extract_token_embeddings(csi_input))


def build_downstream_model(task: str, checkpoint_path: Optional[str], size: str, input_size, patch_size,
                           num_frames: int, tubelet_size: int, in_chans: int = 2, finetune_mode: str = "none",
                           finetune_layers=None, input_normalization: str = "none",
                           input_normalization_scope: str = "global", model_config: Any = "auto",
                           prediction: Optional[Dict[str, Any]] = None) -> DownstreamModel:
    """Build the backbone (from the checkpoint) and the head of ``task``.

    ``prediction`` holds ``observed_snapshots``, ``horizon``, ``readout`` and
    ``head_mlp_layers`` for channel prediction; ``num_frames`` must equal the window
    length returned by ``prediction_window``.
    """
    backbone = EncoderBackbone(
        checkpoint_path, size=size, input_size=input_size, patch_size=patch_size, num_frames=num_frames,
        tubelet_size=tubelet_size, in_chans=in_chans, finetune_mode=finetune_mode,
        finetune_layers=finetune_layers, input_normalization=input_normalization,
        input_normalization_scope=input_normalization_scope, model_config=model_config,
    )
    if task == "channel_estimation":
        head = ChannelEstimationHead(
            feature_dim=backbone.embed_dim, channels=backbone.in_chans, num_frames=backbone.num_frames,
            height=backbone.input_size[0], width=backbone.input_size[1], patch_size=backbone.patch_size,
            tubelet_size=backbone.tubelet_size, num_tokens=backbone.num_tokens,
        )
    elif task == "channel_prediction":
        prediction = prediction or {}
        head = ChannelPredictionHead(
            feature_dim=backbone.embed_dim, channels=backbone.in_chans, window_frames=backbone.num_frames,
            height=backbone.input_size[0], width=backbone.input_size[1], patch_size=backbone.patch_size,
            tubelet_size=backbone.tubelet_size, observed_snapshots=int(prediction["observed_snapshots"]),
            horizon=int(prediction.get("horizon", 1)), mlp_layers=int(prediction.get("head_mlp_layers", 2)),
            readout=str(prediction.get("readout", "end_pad")),
        )
    else:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(DOWNSTREAM_TASKS)}.")
    return DownstreamModel(backbone, head, task)


# ==============================================================================
# engine: Epoch loops and prediction collection for downstream training.
# ==============================================================================

def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()}


def _batch_size(batch: Dict[str, torch.Tensor]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return 1


def train_one_epoch(model, data_loader, optimizer, criterion, metric_fn, device: torch.device) -> Dict[str, float]:
    """One pass over ``data_loader``; returns sample-weighted mean loss and metric."""
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    num_samples = 0
    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = criterion(prediction, batch)
        metric = metric_fn(prediction.detach(), batch)
        loss.backward()
        optimizer.step()
        batch_size = _batch_size(batch)
        total_loss += float(loss.item()) * batch_size
        total_metric += float(metric.item()) * batch_size
        num_samples += batch_size
    return {"loss": total_loss / max(num_samples, 1), "metric": total_metric / max(num_samples, 1),
            "num_samples": num_samples}


@torch.no_grad()
def evaluate(model, data_loader, criterion, metric_fn, device: torch.device) -> Dict[str, float]:
    """Sample-weighted mean loss and metric over ``data_loader`` (the metric is the per-sample NMSE mean)."""
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    num_samples = 0
    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)
        prediction = model(batch)
        loss = criterion(prediction, batch)
        batch_size = _batch_size(batch)
        total_loss += float(loss.item()) * batch_size
        total_metric += float(metric_fn(prediction, batch).item()) * batch_size
        num_samples += batch_size
    return {"loss": total_loss / max(num_samples, 1), "metric": total_metric / max(num_samples, 1),
            "num_samples": num_samples}


@torch.no_grad()
def collect_predictions(model, data_loader, device: torch.device, target_key: str):
    """Return ``(predictions, targets)`` over the loader as CPU tensors (targets ``None`` if absent)."""
    model.eval()
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)
        predictions.append(model(batch).cpu())
        if target_key in batch:
            targets.append(batch[target_key].cpu())
    predictions_tensor = torch.cat(predictions, dim=0)
    targets_tensor = torch.cat(targets, dim=0) if targets else None
    return predictions_tensor, targets_tensor


def per_snapshot_nmse(predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
    """Per-sample NMSE of each predicted snapshot separately (``[B, C, h, H, W]`` -> ``h`` values)."""
    values = []
    for k in range(predictions.shape[2]):
        pred_k, target_k = predictions.select(2, k), targets.select(2, k)
        dims = tuple(range(1, pred_k.ndim))
        values.append(float(((pred_k - target_k).pow(2).sum(dims) / target_k.pow(2).sum(dims).clamp_min(1e-8)).mean()))
    return values


# ==============================================================================
# downstream: Downstream head training on a frozen (or fine-tuned) TempoWiMAE encoder.
# ==============================================================================

def _to_db(value: float) -> Optional[float]:
    return 10.0 * math.log10(value) if value and value > 0 else None


def _write_history(exp_dir: Path, train_losses, val_metrics, val_epochs, best_metric, interrupted=False, extra=None):
    history = {
        "train_losses": train_losses,
        "val_metrics": val_metrics,
        "val_epochs": val_epochs,
        "best_metric": best_metric,
        "epochs_completed": len(train_losses),
        "interrupted": interrupted,
    }
    if extra:
        history.update(extra)
    with open(exp_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def run_downstream(cfg: Dict[str, Any], checkpoint: Optional[str] = None, data_dir: Optional[str] = None,
                   output_dir: Optional[str] = None, device_spec: Optional[str] = None) -> Dict[str, Any]:
    """Train the task head of ``cfg['task']`` and evaluate it on the reporting split.

    Model selection uses the validation split (best validation NMSE, early stopping);
    the reported number is the test split whenever it is labelled (always for prediction,
    for channel estimation when ``X_test.mat`` exists), otherwise validation.

    Outputs in ``<output.dir>/<experiment_name>/<timestamp>_<checkpoint>_<size>_<finetune_mode>/``:
    ``run_config.json``, ``training_history.json``, ``results.json``,
    ``model_save/head_best.pth`` and optionally ``predictions_<split>.npz``.
    """
    task = cfg["task"]
    if task not in DOWNSTREAM_TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(DOWNSTREAM_TASKS)}.")
    task_spec = DOWNSTREAM_TASKS[task]
    geom, model_cfg, train_cfg = cfg["geometry"], cfg["model"], cfg["training"]
    data_cfg, out_cfg = cfg.get("data", {}), cfg.get("output", {})

    data_dir = data_dir or data_cfg.get("dir")
    if not data_dir:
        raise ValueError("No data directory given: pass --data-dir or set data.dir in the configuration.")
    checkpoint = checkpoint if checkpoint is not None else model_cfg.get("checkpoint")
    size = str(model_cfg.get("size", "nano"))
    finetune_mode = str(model_cfg.get("finetune_mode", "none"))
    input_size = tuple(int(v) for v in geom["input_size"])
    patch_size = tuple(int(v) for v in geom["patch_size"])
    tubelet_size = int(geom["tubelet_size"])
    in_chans = int(geom.get("in_chans", 2))

    prediction = None
    if task == "channel_prediction":
        prediction = dict(cfg["prediction"])
        num_frames, pad_front, hor_pad = prediction_window(
            prediction["observed_snapshots"], prediction.get("horizon", 1), tubelet_size,
            prediction.get("readout", "end_pad"))
        print(f"[window] readout={prediction.get('readout', 'end_pad')} observed={prediction['observed_snapshots']} "
              f"horizon={prediction.get('horizon', 1)} tubelet={tubelet_size} -> num_frames={num_frames} "
              f"(pad_front={pad_front}, horizon_pad={hor_pad})")
    else:
        num_frames = int(geom["num_frames"])

    seed = int(train_cfg.get("seed", 100))
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["lr"])
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    batch_size = int(train_cfg.get("batch_size", 64))
    early_stop = int(train_cfg.get("early_stop", 1000))
    num_workers = int(train_cfg.get("num_workers", 8))

    set_seed(seed, deterministic_cudnn=True)
    device = resolve_device(device_spec or train_cfg.get("device"))

    datasets = build_downstream_datasets(task, data_dir, seed=seed, train_limit=data_cfg.get("train_limit"))
    if task == "channel_prediction":
        train_set = datasets["train"]
        if train_set.observed_snapshots != int(prediction["observed_snapshots"]) or \
                train_set.horizon != int(prediction.get("horizon", 1)):
            raise ValueError(
                f"Data in {data_dir} holds {train_set.observed_snapshots} observed snapshots and a horizon of "
                f"{train_set.horizon}, but the configuration asks for {prediction['observed_snapshots']} -> "
                f"{prediction.get('horizon', 1)}."
            )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    has_test_labels = task == "channel_prediction" or getattr(datasets["test"], "h_full", None) is not None
    reporting_split = "test" if has_test_labels else "val"
    reporting_loader = test_loader if reporting_split == "test" else val_loader

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_stem = Path(checkpoint).stem if checkpoint else "random_init"
    run_name = f"{date_str}_{ckpt_stem}_{size}_{finetune_mode}"
    exp_dir = Path(output_dir or out_cfg.get("dir", "outputs") ) / str(cfg.get("experiment_name", task)) / run_name
    (exp_dir / "model_save").mkdir(parents=True, exist_ok=True)

    run_config = {
        "date": date_str, "task": task, "checkpoint": checkpoint, "size": size, "finetune_mode": finetune_mode,
        "finetune_layers": list(model_cfg.get("finetune_layers", []) or []), "data_dir": str(data_dir),
        "train_limit": data_cfg.get("train_limit"), "epochs": epochs, "lr": lr, "min_lr": min_lr,
        "batch_size": batch_size, "early_stop": early_stop, "num_workers": num_workers, "seed": seed,
        "input_size": list(input_size), "patch_size": list(patch_size), "num_frames": num_frames,
        "tubelet_size": tubelet_size, "input_normalization": model_cfg.get("input_normalization", "none"),
        "input_normalization_scope": model_cfg.get("input_normalization_scope", "global"),
        "prediction": prediction, "selection_split": "val", "reporting_split": reporting_split,
        "metric": "NMSE (mean per-sample linear ratio; dB = 10 log10)",
        "num_train_samples": len(datasets["train"]), "num_val_samples": len(datasets["val"]),
        "num_test_samples": len(datasets["test"]),
    }
    with open(exp_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    print("=" * 60)
    print(f"  TempoWiMAE downstream  |  {task_spec['display_name']}")
    print("=" * 60)
    print(f"Checkpoint      : {checkpoint}")
    print(f"Size            : {size}   finetune_mode: {finetune_mode}")
    print(f"Data            : {data_dir}  (train {len(datasets['train'])}, val {len(datasets['val'])}, "
          f"test {len(datasets['test'])}, train_limit={data_cfg.get('train_limit')})")
    print(f"Epochs          : {epochs}   lr: {lr}   early_stop: {early_stop}   seed: {seed}")
    print(f"Reporting split : {reporting_split}")
    print(f"Output dir      : {exp_dir}")

    model = build_downstream_model(
        task, checkpoint, size=size, input_size=input_size, patch_size=patch_size, num_frames=num_frames,
        tubelet_size=tubelet_size, in_chans=in_chans, finetune_mode=finetune_mode,
        finetune_layers=model_cfg.get("finetune_layers", []), input_normalization=model_cfg.get("input_normalization", "none"),
        input_normalization_scope=model_cfg.get("input_normalization_scope", "global"),
        model_config=model_cfg.get("architecture", "auto"), prediction=prediction,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params / 1e6:.3f}M")
    print(f"Trainable parameters : {trainable_params / 1e6:.5f}M\n")

    criterion = task_spec["loss_fn"]().to(device)
    metric_fn = task_spec["metric_fn"]
    metric_name = task_spec["metric_name"]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)   # constant learning rate

    train_losses, val_metrics, val_epochs = [], [], []
    best_metric = float("inf")
    patience = 0
    best_path = exp_dir / "model_save" / "head_best.pth"
    interrupted = False
    try:
        for epoch in range(epochs):
            train_stats = train_one_epoch(model, train_loader, optimizer, criterion, metric_fn, device)
            train_losses.append(train_stats["loss"])
            val_metric = evaluate(model, val_loader, criterion, metric_fn, device)["metric"]
            val_metrics.append(val_metric)
            val_epochs.append(epoch + 1)
            if val_metric < best_metric:
                best_metric = val_metric
                patience = 0
                head_state = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
                torch.save(head_state, best_path)
            else:
                patience += 1
            print(f"Epoch [{epoch + 1}/{epochs}]  train loss {train_stats['loss']:.6f}  "
                  f"val {metric_name} {val_metric:.6f}  best {best_metric:.6f}")
            _write_history(exp_dir, train_losses, val_metrics, val_epochs, best_metric)
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"], min_lr)
            if patience >= early_stop:
                print(f"Early stopping after {epoch + 1} epochs (patience={early_stop}).")
                break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nTraining interrupted after {len(train_losses)} epochs; evaluating the best head so far.")

    results: Dict[str, Any] = {
        "task": task, "checkpoint": checkpoint, "selection_split": "val", "reporting_split": reporting_split,
        "best_val_nmse": best_metric, "best_val_nmse_db": _to_db(best_metric),
        "epochs_completed": len(train_losses), "interrupted": interrupted,
    }
    if best_path.exists():
        head_state = torch.load(str(best_path), map_location=device, weights_only=True)
        model.load_state_dict(head_state, strict=False)
        report_stats = evaluate(model, reporting_loader, criterion, metric_fn, device)
        reporting_metric = report_stats["metric"]
        results.update({
            f"{reporting_split}_nmse": reporting_metric,
            f"{reporting_split}_nmse_db": _to_db(reporting_metric),
            f"{reporting_split}_loss": report_stats["loss"],
            f"{reporting_split}_num_samples": report_stats["num_samples"],
            "reporting_nmse": reporting_metric,
            "reporting_nmse_db": _to_db(reporting_metric),
        })
        predictions, targets = collect_predictions(model, reporting_loader, device, task_spec["target_key"])
        if task == "channel_prediction" and targets is not None and predictions.shape[2] > 1:
            snapshot_values = per_snapshot_nmse(predictions, targets)
            results[f"{reporting_split}_nmse_per_snapshot"] = snapshot_values
            results[f"{reporting_split}_nmse_per_snapshot_db"] = [_to_db(v) for v in snapshot_values]
        if out_cfg.get("save_predictions", False):
            payload = {"prediction": predictions.numpy()}
            if targets is not None:
                payload["target"] = targets.numpy()
            np.savez_compressed(exp_dir / f"predictions_{reporting_split}.npz", **payload)
        _write_history(exp_dir, train_losses, val_metrics, val_epochs, best_metric, interrupted=interrupted,
                       extra={"selection_split": "val", "reporting_split": reporting_split,
                              "reporting_metric": reporting_metric})
        print(f"\n{reporting_split.title()} {metric_name}: {reporting_metric:.6f} ({_to_db(reporting_metric):.2f} dB)")
    else:
        print("\nNo best head was saved (no completed epoch); skipping the final evaluation.")

    with open(exp_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Best val {metric_name}: {best_metric:.6f}")
    print(f"Artifacts saved to {exp_dir}")
    results["output_dir"] = str(exp_dir)
    return results


# ==============================================================================
# command line
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    import config as paper_config

    parser = argparse.ArgumentParser(
        description="Train a lightweight task head on a pretrained TempoWiMAE encoder and report the test NMSE. "
                    "Defaults are the paper settings in config.CHANNEL_ESTIMATION / config.CHANNEL_PREDICTION.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default="channel_prediction", choices=["channel_estimation", "channel_prediction"])
    parser.add_argument("--setting", default="3_to_1", choices=list(paper_config.PREDICTION_SETTINGS),
                        help="Prediction setting T_p -> h (channel_prediction only).")
    parser.add_argument("--observed_snapshots", type=int, default=None, help="T_p; overrides --setting.")
    parser.add_argument("--horizon", type=int, default=None, help="h; overrides --setting (depth sweep: s = 1 .. 11).")
    parser.add_argument("--checkpoint", default="models/tempowimae_nano_paper_200ep.pth",
                        help="Pretraining checkpoint (models/tempowimae_nano_matched_25ep.pth for the matched budget).")
    parser.add_argument("--data_dir", default=None,
                        help="Folder with the task's .mat files (default: data/channel_estimation or data/prediction_<Tp>_to_<h>).")
    parser.add_argument("--output_dir", default=None, help="Overrides output.dir.")
    parser.add_argument("--experiment_name", default=None, help="Overrides experiment_name.")
    parser.add_argument("--epochs", type=int, default=None, help="Overrides training.epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Overrides training.seed.")
    parser.add_argument("--train_limit", type=int, default=None,
                        help="Train on a seeded subset of this many labelled samples (label-fraction study: 40, 80, 200, 400).")
    parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if available).")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config entry, e.g. --set model.finetune_mode=full "
                             "--set prediction.head_mlp_layers=1 --set model.checkpoint=null.")
    return parser


def main(argv=None) -> None:
    import config as paper_config

    args = build_parser().parse_args(argv)
    overrides = list(args.overrides)
    if args.task == "channel_estimation":
        cfg = copy.deepcopy(paper_config.CHANNEL_ESTIMATION)
        data_dir = args.data_dir or "data/channel_estimation"
    else:
        cfg = copy.deepcopy(paper_config.CHANNEL_PREDICTION)
        observed, horizon = paper_config.PREDICTION_SETTINGS[args.setting]
        if args.observed_snapshots is not None:
            observed = args.observed_snapshots
        if args.horizon is not None:
            horizon = args.horizon
        cfg["prediction"]["observed_snapshots"] = int(observed)
        cfg["prediction"]["horizon"] = int(horizon)
        if args.setting == "depth_sweep":
            cfg["experiment_name"] = f"prediction_observe{observed}_depth{horizon}"
            data_dir = args.data_dir or f"data/prediction_observe{observed}_depth{horizon}"
        else:
            cfg["experiment_name"] = f"prediction_{observed}_to_{horizon}"
            data_dir = args.data_dir or f"data/prediction_{observed}_to_{horizon}"
    if args.experiment_name is not None:
        overrides.append(f"experiment_name={args.experiment_name}")
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.seed is not None:
        overrides.append(f"training.seed={args.seed}")
    if args.train_limit is not None:
        overrides.append(f"data.train_limit={args.train_limit}")
    cfg = apply_overrides(cfg, overrides)
    checkpoint = args.checkpoint if "model.checkpoint" not in "".join(overrides) else cfg["model"].get("checkpoint")
    run_downstream(cfg, checkpoint=checkpoint, data_dir=data_dir, output_dir=args.output_dir, device_spec=args.device)


if __name__ == "__main__":
    main()
