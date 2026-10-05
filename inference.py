"""Load a TempoWiMAE checkpoint, encode CSI samples and report the masked-reconstruction error.

Example::

    python inference.py                                            # released checkpoint, random input
    python inference.py --input data/wifo/D4/X_val.mat --num_samples 8 --output embeddings.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from input_preprocess import CSIDataset
from masking import build_batch_mask
from tempowimae_model import build_model, checkpoint_model_config, load_full_model, read_checkpoint
from train import build_patch_targets, reconstruction_metrics
from utils import resolve_device


def geometry_from_checkpoint(checkpoint: dict) -> dict:
    """Read the pretraining geometry stored in the checkpoint (paper defaults if absent)."""
    cfg = checkpoint.get("config") or {}
    geom = cfg.get("geometry", cfg)   # nested (this code) or flat (original research runs)
    return {
        "num_frames": int(geom.get("num_frames", 16)),
        "subcarriers": int(geom.get("subcarriers", 32)),
        "antennas": int(geom.get("antennas", 32)),
        "patch_size": tuple(geom.get("patch_size", (4, 4))),
        "tubelet_size": int(geom.get("tubelet_size", 4)),
        "in_chans": int(geom.get("in_chans", 2)),
        "size": str(cfg.get("model", {}).get("size", "nano")) if isinstance(cfg.get("model"), dict) else "nano",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Encode CSI samples with a pretrained TempoWiMAE checkpoint.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default="models/tempowimae_nano_paper_200ep.pth")
    parser.add_argument("--input", default=None, help="MATLAB v7.3 CSI file in the pretraining format (random input if omitted).")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--mask_pattern", default="temporal_frame",
                        help="Pattern for the reconstruction check: random | tube | frequency_tube | antenna_tube | temporal_tail | temporal_frame")
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--output", default=None, help="Optional .npz with the encoder tokens of every sample.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    checkpoint = read_checkpoint(args.checkpoint)
    geom = geometry_from_checkpoint(checkpoint)
    device = resolve_device(args.device)
    model = build_model(geom["size"], input_size=(geom["subcarriers"], geom["antennas"]), patch_size=geom["patch_size"],
                        tubelet_size=geom["tubelet_size"], num_frames=geom["num_frames"], in_chans=geom["in_chans"],
                        model_config=checkpoint_model_config(checkpoint))
    load_full_model(model, checkpoint, strict=True)
    model = model.to(device).eval()
    print(f"[inference] {args.checkpoint}: epoch {checkpoint.get('epoch')}, architecture {model.model_config.to_dict()}")

    if args.input:
        dataset = CSIDataset([args.input], max_samples_per_file=args.num_samples)
        x = torch.stack([dataset[i] for i in range(len(dataset))])
    else:
        torch.manual_seed(0)
        x = torch.randn(args.num_samples, geom["in_chans"], geom["num_frames"], geom["subcarriers"], geom["antennas"])
    x = x.to(device)
    num_tokens = model.encoder.patch_embed.num_patches
    token_grid = (geom["num_frames"] // geom["tubelet_size"], geom["subcarriers"] // geom["patch_size"][0],
                  geom["antennas"] // geom["patch_size"][1])

    with torch.no_grad():
        # Frozen-encoder features: every token visible (this is what the downstream heads consume).
        tokens = model.encoder(x, torch.zeros(x.shape[0], num_tokens, dtype=torch.bool, device=device))
        # Masked reconstruction of the pretraining objective.
        np.random.seed(0)
        mask = torch.from_numpy(build_batch_mask(token_grid, args.mask_pattern, args.mask_ratio, x.shape[0])).to(device)
        targets = build_patch_targets(x, mask, geom["patch_size"], geom["tubelet_size"], normalize_target=True)
        outputs = model(x, mask)
        mse, nmse, nmse_db = reconstruction_metrics(outputs, targets)
    print(f"[inference] input {tuple(x.shape)} -> encoder tokens {tuple(tokens.shape)} (token grid {token_grid})")
    print(f"[inference] {args.mask_pattern} masking at ratio {args.mask_ratio}: reconstruction NMSE on masked tokens "
          f"{nmse.item():.4f} ({nmse_db.item():.2f} dB), MSE {mse.item():.4f}")
    if args.output:
        np.savez_compressed(args.output, tokens=tokens.cpu().numpy(), mask=mask.cpu().numpy())
        print(f"[inference] tokens saved to {args.output}")


if __name__ == "__main__":
    main()
