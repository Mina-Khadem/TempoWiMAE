"""Masking patterns over the token grid and the hybrid per-batch selection used for pretraining."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


# ==============================================================================
# patterns: The six masking patterns over the token grid and the hybrid per-batch selection.
#
# Every generator returns a flat ``(T*H*W,)`` array over the token grid
# ``(T/p_t, H/p_h, W/p_w)`` with ``1`` for masked and ``0`` for visible tokens, and
# masks a constant number of tokens for a fixed ratio. That constant count is what
# lets the encoder gather the visible tokens with one reshape, so one pattern and
# one ratio are applied to every sample of a batch (:func:`build_batch_mask`).
#
# The generators draw from NumPy's global random state (``np.random``), which the
# training entry point seeds; :func:`pick_pattern` draws from the ``RandomState``
# it is given.
# ==============================================================================

class RandomMaskingGenerator:
    """Mask individual tokens uniformly at random."""

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.total_patches = self.frames * self.height * self.width
        self.total_masks = int(mask_ratio * self.total_patches)

    def set_mask_ratio(self, mask_ratio):
        self.total_masks = int(mask_ratio * self.total_patches)

    def __repr__(self):
        return f"Random mask: total {self.total_patches}, masked {self.total_masks}"

    def __call__(self):
        mask = np.hstack([np.zeros(self.total_patches - self.total_masks), np.ones(self.total_masks)])
        np.random.shuffle(mask)
        return mask


class TubeMaskingGenerator:
    """Mask the same subcarrier-antenna cells in every token time step."""

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def set_mask_ratio(self, mask_ratio):
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        return f"Tube mask: total {self.total_patches}, masked {self.total_masks}"

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        return np.tile(mask_per_frame, (self.frames, 1)).flatten()


class FrequencyTubeMaskingGenerator:
    """Mask whole rows of the grid (subcarrier axis) across all antennas and time steps."""

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_masks_along_height = int(mask_ratio * self.height)
        self.total_patches = self.frames * self.height * self.width
        self.total_masks = self.frames * self.num_masks_along_height * self.width

    def set_mask_ratio(self, mask_ratio):
        self.num_masks_along_height = int(mask_ratio * self.height)
        self.total_masks = self.frames * self.num_masks_along_height * self.width

    def __repr__(self):
        return f"Frequency tube mask: total {self.total_patches}, masked {self.total_masks}"

    def __call__(self):
        mask_along_height = np.hstack([
            np.zeros(self.height - self.num_masks_along_height),
            np.ones(self.num_masks_along_height),
        ])
        np.random.shuffle(mask_along_height)
        mask_per_frame = np.repeat(mask_along_height[:, None], self.width, axis=1)
        return np.tile(mask_per_frame[None, :, :], (self.frames, 1, 1)).flatten()


class AntennaTubeMaskingGenerator:
    """Mask whole columns of the grid (antenna axis) across all subcarriers and time steps."""

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_masks_along_width = int(mask_ratio * self.width)
        self.total_patches = self.frames * self.height * self.width
        self.total_masks = self.frames * self.height * self.num_masks_along_width

    def set_mask_ratio(self, mask_ratio):
        self.num_masks_along_width = int(mask_ratio * self.width)
        self.total_masks = self.frames * self.height * self.num_masks_along_width

    def __repr__(self):
        return f"Antenna tube mask: total {self.total_patches}, masked {self.total_masks}"

    def __call__(self):
        mask_along_width = np.hstack([
            np.zeros(self.width - self.num_masks_along_width),
            np.ones(self.num_masks_along_width),
        ])
        np.random.shuffle(mask_along_width)
        mask_per_frame = np.repeat(mask_along_width[None, :], self.height, axis=0)
        return np.tile(mask_per_frame[None, :, :], (self.frames, 1, 1)).flatten()


class TemporalTailMaskingGenerator:
    """Mask a contiguous block of token time steps at the end of the sequence (forecasting)."""

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self._set(mask_ratio)
        self.total_patches = self.frames * self.height * self.width

    def _set(self, mask_ratio):
        if mask_ratio <= 0:
            self.num_masked_frames = 0
        else:
            self.num_masked_frames = max(1, int(mask_ratio * self.frames))
        self.num_masked_frames = min(self.frames, self.num_masked_frames)
        self.total_masks = self.num_masked_frames * self.height * self.width

    def set_mask_ratio(self, mask_ratio):
        self._set(mask_ratio)

    def __repr__(self):
        return (f"Temporal tail mask: total {self.frames*self.height*self.width}, "
                f"masked {self.total_masks}, tail frames {self.num_masked_frames}")

    def __call__(self):
        visible = np.zeros((self.frames - self.num_masked_frames, self.height, self.width))
        masked = np.ones((self.num_masked_frames, self.height, self.width))
        return np.concatenate([visible, masked], axis=0).flatten()


class TemporalFrameMaskingGenerator:
    """Mask a random subset of complete token time steps (interpolation and extrapolation)."""

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.total_patches = self.frames * self.height * self.width
        self._set(mask_ratio)

    def _set(self, mask_ratio):
        if mask_ratio <= 0:
            self.num_masked_frames = 0
        else:
            self.num_masked_frames = max(1, int(mask_ratio * self.frames))
        self.num_masked_frames = min(self.frames, self.num_masked_frames)
        self.total_masks = self.num_masked_frames * self.height * self.width

    def set_mask_ratio(self, mask_ratio):
        self._set(mask_ratio)

    def __repr__(self):
        return (f"Temporal frame mask: total {self.total_patches}, "
                f"masked {self.total_masks}, masked frames {self.num_masked_frames}")

    def __call__(self):
        frame_flags = np.hstack([
            np.zeros(self.frames - self.num_masked_frames),
            np.ones(self.num_masked_frames),
        ])
        np.random.shuffle(frame_flags)
        mask = np.repeat(frame_flags[:, None], self.height * self.width, axis=1)
        return mask.flatten()


MASK_PATTERNS = {
    "random": RandomMaskingGenerator,
    "tube": TubeMaskingGenerator,
    "frequency_tube": FrequencyTubeMaskingGenerator,
    "antenna_tube": AntennaTubeMaskingGenerator,
    "temporal_tail": TemporalTailMaskingGenerator,
    "temporal_frame": TemporalFrameMaskingGenerator,
}

# ``temporal`` is accepted as an alias of ``temporal_frame`` in older configurations.
_PATTERN_ALIASES = {"temporal": "temporal_frame"}


def canonical_pattern(name: str) -> str:
    return _PATTERN_ALIASES.get(str(name), str(name))


def build_mask_generator(pattern: str, input_size, mask_ratio: float):
    """Instantiate the generator of ``pattern`` for the token grid ``input_size`` ``(T, H, W)``."""
    pattern = canonical_pattern(pattern)
    if pattern not in MASK_PATTERNS:
        raise ValueError(f"Unknown masking pattern '{pattern}'. Choices: {sorted(MASK_PATTERNS)}")
    return MASK_PATTERNS[pattern](input_size, mask_ratio)


def build_batch_mask(input_size, pattern: str, mask_ratio: float, batch_size: int) -> np.ndarray:
    """One pattern and ratio applied to all samples of a batch; returns a bool array ``[B, T*H*W]``."""
    generator = build_mask_generator(pattern, input_size, mask_ratio)
    masks = [generator().astype(bool) for _ in range(int(batch_size))]
    return np.stack(masks, axis=0)


def pick_pattern(patterns: Sequence[str], weights: Sequence[float], rng=None) -> str:
    """Draw one pattern name with the given selection weights (normalised internally)."""
    w = np.asarray([float(x) for x in weights], dtype=np.float64)
    w = w / w.sum()
    r = rng if rng is not None else np.random
    return str(patterns[int(r.choice(len(patterns), p=w))])


# ==============================================================================
# curriculum: Optional mask-ratio curriculum (disabled in the paper configuration).
# ==============================================================================

class MaskRatioCurriculum:
    """Per-epoch multiplier applied to every pattern's base mask ratio.

    The value ramps from ``start`` to ``end`` over ``warmup_epochs`` (linear or cosine)
    and is then held at ``end``. The ratio is constant within an epoch so every batch
    keeps a constant visible-token count.
    """

    def __init__(self, start: float, end: float, warmup_epochs: int, shape: str = "linear",
                 floor: float = 0.0, ceil: float = 1.0):
        if shape not in ("linear", "cosine"):
            raise ValueError(f"shape must be 'linear' or 'cosine', got {shape}")
        self.start = float(start)
        self.end = float(end)
        self.warmup_epochs = int(warmup_epochs)
        self.shape = shape
        self.floor = float(floor)
        self.ceil = float(ceil)

    @property
    def enabled(self) -> bool:
        return self.warmup_epochs > 0 and abs(self.start - self.end) > 1e-8

    def value_at(self, epoch: int) -> float:
        if not self.enabled:
            return self.end
        if epoch >= self.warmup_epochs:
            frac = 1.0
        else:
            frac = epoch / float(self.warmup_epochs)
        if self.shape == "cosine":
            frac = 0.5 * (1.0 - math.cos(math.pi * frac))
        val = self.start + (self.end - self.start) * frac
        return max(self.floor, min(self.ceil, val))
