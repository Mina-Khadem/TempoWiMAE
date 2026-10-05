"""Shared helpers: CSI tensor utilities, seeding, logging, device selection and configuration overrides."""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import random
from typing import Optional, Tuple

import numpy as np
import torch
from einops import rearrange


# ==============================================================================
# csi: CSI tensor helpers shared by pretraining and downstream evaluation.
#
# CSI samples are real tensors of shape ``[C=2, T, H, W]``: channel 0 is the real
# part and channel 1 the imaginary part of the complex channel, ``T`` indexes
# snapshots in time and ``H x W`` is the subcarrier-antenna grid of one snapshot.
# ==============================================================================

VALID_INPUT_NORMALIZATION_TYPES = ("none", "standardized", "rms")
VALID_INPUT_NORMALIZATION_SCOPES = ("global", "patch")


def to_2tuple(value) -> Tuple[int, int]:
    """Return ``(a, b)`` for a scalar ``a`` or a pair ``(a, b)``."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a pair, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def complex_to_channels(array: np.ndarray) -> torch.Tensor:
    """Stack real and imaginary parts as two channels: ``[N, ...] complex -> [N, 2, ...] float32``."""
    stacked = np.stack([array.real, array.imag], axis=1)
    return torch.from_numpy(stacked.astype(np.float32, copy=False))


def channels_to_complex(tensor: torch.Tensor) -> np.ndarray:
    """Inverse of :func:`complex_to_channels`."""
    array = tensor.detach().cpu().numpy()
    return array[:, 0] + 1j * array[:, 1]


def unpatchify_csi(tokens: torch.Tensor, channels: int, num_frames: int, height: int, width: int,
                   patch_size=(4, 4), tubelet_size: int = 4) -> torch.Tensor:
    """Fold per-token patch predictions ``[B, N, p0*p1*p2*C]`` back into a CSI tensor ``[B, C, T, H, W]``.

    Token order is the Conv3d flatten order of the patch embedding (time outer, then
    height, then width), and the within-token layout is ``(p0, p1, p2, c)``.
    """
    patch_h, patch_w = to_2tuple(patch_size)
    return rearrange(
        tokens,
        "b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)",
        t=num_frames // tubelet_size,
        h=height // patch_h,
        w=width // patch_w,
        p0=tubelet_size,
        p1=patch_h,
        p2=patch_w,
        c=channels,
    )


def resolve_input_normalization(normalization_type: str, normalization_scope: str) -> Tuple[str, str]:
    """Validate the downstream input-normalization choice and return ``(type, scope)``."""
    normalization_type = str(normalization_type).lower()
    normalization_scope = str(normalization_scope).lower()
    if normalization_type not in VALID_INPUT_NORMALIZATION_TYPES:
        raise ValueError(
            f"Unsupported input_normalization '{normalization_type}'. "
            f"Choose from {VALID_INPUT_NORMALIZATION_TYPES}."
        )
    if normalization_type == "none":
        return "none", "global"
    if normalization_scope not in VALID_INPUT_NORMALIZATION_SCOPES:
        raise ValueError(
            f"Unsupported input_normalization_scope '{normalization_scope}'. "
            f"Choose from {VALID_INPUT_NORMALIZATION_SCOPES}."
        )
    return normalization_type, normalization_scope


def _normalize_patchwise(csi_tensor: torch.Tensor, normalization_type: str,
                         patch_size: Tuple[int, int], tubelet_size: int, eps: float) -> torch.Tensor:
    csi_tensor = csi_tensor.contiguous()
    channels, frames, height, width = csi_tensor.shape
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    tubelet_size = int(tubelet_size)
    if tubelet_size <= 0 or patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"patch_size={patch_size} and tubelet_size={tubelet_size} must be positive.")
    if frames % tubelet_size != 0 or height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            "Per-patch input normalization requires the CSI tensor shape to align with "
            f"tubelet_size={tubelet_size} and patch_size={patch_size}, got {tuple(csi_tensor.shape)}."
        )

    patch_grid = csi_tensor.view(
        channels, frames // tubelet_size, tubelet_size,
        height // patch_h, patch_h, width // patch_w, patch_w,
    ).permute(1, 3, 5, 0, 2, 4, 6).contiguous()
    patch_values = patch_grid.view(-1, channels * tubelet_size * patch_h * patch_w)

    if normalization_type == "standardized":
        mean = patch_values.mean(dim=1, keepdim=True)
        variance = patch_values.var(dim=1, unbiased=False, keepdim=True)
        normalized = (patch_values - mean) / torch.sqrt(variance + eps)
    elif normalization_type == "rms":
        rms = torch.sqrt(patch_values.pow(2).mean(dim=1, keepdim=True) + eps)
        normalized = patch_values / rms
    else:
        raise ValueError(f"Unsupported patchwise input normalization type: {normalization_type}")

    patch_grid = normalized.view(
        frames // tubelet_size, height // patch_h, width // patch_w,
        channels, tubelet_size, patch_h, patch_w,
    )
    return patch_grid.permute(3, 0, 4, 1, 5, 2, 6).contiguous().view(channels, frames, height, width)


def normalize_csi_input(csi_tensor: torch.Tensor, normalization_type: str, normalization_scope: str,
                        patch_size: Optional[Tuple[int, int]] = None, tubelet_size: int = 1,
                        eps: float = 1e-6) -> torch.Tensor:
    """Normalize one CSI sample ``[C, T, H, W]`` before it enters the encoder."""
    normalization_type, normalization_scope = resolve_input_normalization(normalization_type, normalization_scope)
    if normalization_type == "none":
        return csi_tensor
    if normalization_scope == "global" and normalization_type == "standardized":
        mean = csi_tensor.mean()
        variance = csi_tensor.var(unbiased=False)
        return (csi_tensor - mean) / torch.sqrt(variance + eps)
    if normalization_scope == "global" and normalization_type == "rms":
        rms = torch.sqrt(csi_tensor.pow(2).mean() + eps)
        return csi_tensor / rms
    if patch_size is None:
        raise ValueError("Per-patch input normalization requires patch_size to be set.")
    return _normalize_patchwise(csi_tensor, normalization_type=normalization_type,
                                patch_size=patch_size, tubelet_size=tubelet_size, eps=eps)


# ==============================================================================
# seed: Random-seed control.
# ==============================================================================

def set_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (all CUDA devices).

    Args:
        seed: the seed exposed in every configuration file.
        deterministic_cudnn: disable cuDNN autotuning and request deterministic
            kernels (used by the downstream runs).
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ==============================================================================
# logging_utils: Console mirroring, TensorBoard scalars and device selection.
# ==============================================================================

class Tee:
    """Write the same text to several streams (console and ``log.txt``)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


class TensorBoardLogger:
    """Minimal scalar logger; silently disabled when no TensorBoard writer is installed."""

    def __init__(self, log_dir: str):
        self.writer = None
        self.step = 0
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except Exception:
            try:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(logdir=log_dir)
            except Exception:
                self.writer = None
        if self.writer is None:
            print("[tensorboard] no SummaryWriter available (install tensorboard or tensorboardX); scalars disabled")

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def set_step(self, step: Optional[int] = None) -> None:
        self.step = self.step + 1 if step is None else int(step)

    def update(self, head: str = "scalar", step: Optional[int] = None, **scalars) -> None:
        if self.writer is None:
            return
        for key, value in scalars.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f"{head}/{key}", float(value), self.step if step is None else step)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()


def resolve_device(spec: Optional[str] = None) -> torch.device:
    """Return the requested device (``cuda``, ``cuda:1``, ``cpu``); default to CUDA when available."""
    if spec:
        device = torch.device(spec)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Device '{spec}' requested but CUDA is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# configuration helpers (the paper settings live in config.py)
# ==============================================================================

def join_path(root, path) -> str:
    """Join ``path`` onto ``root`` unless ``path`` is already absolute."""
    path = Path(path)
    return str(path) if path.is_absolute() else str(Path(root) / path)


def dump_config(cfg: dict, path) -> None:
    """Write the resolved configuration next to the run outputs (JSON)."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, default=str)


def parse_value(text: str):
    """Parse a command-line value: ``true`` / ``false`` / ``null``, numbers, lists, dicts, else a string."""
    lowered = text.strip().lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none"):
        return None
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return text.strip()


def apply_overrides(cfg: dict, assignments) -> dict:
    """Return a copy of ``cfg`` with ``section.key=value`` assignments applied.

    Examples: ``optimization.lr=5e-4``, ``model.size=tiny``, ``masking.patterns=[tube]``,
    ``masking.weights=[1.0]``, ``model.finetune_mode=full``.
    """
    cfg = copy.deepcopy(cfg)
    for item in assignments or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must have the form section.key=value")
        dotted, raw = item.split("=", 1)
        node = cfg
        keys = dotted.strip().split(".")
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = parse_value(raw)
    return cfg
