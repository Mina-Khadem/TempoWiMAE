"""Data loading for pretraining (MATLAB v7.3 CSI files) and for the downstream tasks, plus a
command-line tool that builds the downstream split files from raw CSI sequences (see README).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import complex_to_channels


# ==============================================================================
# pretraining: Pretraining dataset: lazy reader of MATLAB v7.3 complex CSI files.
# ==============================================================================

class CSIDataset(Dataset):
    """Complex CSI stored as ``(S, A, T, N)`` in MATLAB v7.3 files -> float tensors ``[2, T, S, A]``.

    Each file holds one variable (the first key not starting with ``#``) whose last
    axis indexes samples. Complex values may be stored either as a compound
    ``real``/``imag`` dtype or as a native complex dtype.

    Args:
        paths: list of ``.mat`` files; all samples of all files are concatenated.
        max_samples_per_file: use only the first ``n`` samples of every file (0 = all).
    """

    def __init__(self, paths: Sequence[str], max_samples_per_file: int = 0):
        self.index = []
        self._handles = {}
        self._varname = {}
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Pretraining file not found: {path}. See README.md (Data).")
            with h5py.File(path, "r") as handle:
                var = next(k for k in handle.keys() if not k.startswith("#"))
                n = handle[var].shape[-1]
            if max_samples_per_file and max_samples_per_file > 0:
                n = min(n, int(max_samples_per_file))
            self._varname[path] = var
            self.index.extend((path, i) for i in range(n))

    def __len__(self):
        return len(self.index)

    def _handle(self, path):
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def __getitem__(self, idx):
        path, sample = self.index[idx]
        dataset = self._handle(path)[self._varname[path]]
        raw = dataset[..., sample]
        if raw.dtype.names and "real" in raw.dtype.names:
            real = np.asarray(raw["real"], dtype=np.float32)
            imag = np.asarray(raw["imag"], dtype=np.float32)
        else:
            array = np.asarray(raw)
            real = array.real.astype(np.float32)
            imag = array.imag.astype(np.float32)
        # (S, A, T) -> (T, S, A)
        real = np.transpose(real, (2, 0, 1))
        imag = np.transpose(imag, (2, 0, 1))
        return torch.from_numpy(np.stack([real, imag], axis=0))


# ==============================================================================
# downstream: Downstream datasets: channel estimation from pilots and channel prediction.
# ==============================================================================

def _resolve_variable_name(handle: h5py.File, requested: Optional[str]) -> str:
    if requested:
        if requested not in handle:
            raise KeyError(f"Variable '{requested}' not found in {handle.filename}. Available keys: {list(handle.keys())}")
        return requested
    for key in handle.keys():
        if not key.startswith("#"):
            return key
    raise ValueError(f"No dataset variable found in {handle.filename}.")


def load_complex_mat(path: str, variable_name: Optional[str] = None) -> np.ndarray:
    """Load a complex array from a MATLAB v7.3 file and restore MATLAB axis order.

    MATLAB stores arrays column-major, so h5py exposes the reversed axes; the result
    has the shape the array had in MATLAB, e.g. ``(N, T, H, W)``.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Downstream data file not found: {path}. See README.md (Data).")
    with h5py.File(path, "r") as handle:
        key = _resolve_variable_name(handle, variable_name)
        raw = handle[key][()]
    if raw.dtype.fields and "real" in raw.dtype.fields and "imag" in raw.dtype.fields:
        array = raw["real"] + 1j * raw["imag"]
    else:
        array = raw
    axes = tuple(range(array.ndim - 1, -1, -1))
    return np.transpose(array, axes=axes)


class ChannelEstimationDataset(Dataset):
    """Pilot observations interpolated to the full grid, paired with the full CSI target.

    Files hold complex arrays of shape ``(N, T, H, W_pilot)`` (pilots) and ``(N, T, H, W)``
    (full CSI). The pilot tensor is interpolated (trilinear) to the full-CSI size, which
    is the model input ``h_inter``; the target is ``h_full``.

    Args:
        pilot_path / full_path: MATLAB v7.3 files; ``full_path`` may be ``None`` for an
            unlabelled split, in which case ``target_size`` ``(T, H, W)`` is required.
        limit: keep only the first ``limit`` samples (after the optional shuffle).
        shuffle_seed: shuffle the sample order once with ``np.random.RandomState(seed)``.
    """

    def __init__(self, pilot_path: str, full_path: Optional[str] = None, pilot_variable: Optional[str] = None,
                 full_variable: Optional[str] = None, target_size: Optional[Tuple[int, int, int]] = None,
                 limit: Optional[int] = None, shuffle_seed: Optional[int] = None):
        super().__init__()
        self.pilot_path = str(pilot_path)
        self.full_path = None if full_path is None else str(full_path)
        pilot_complex = load_complex_mat(self.pilot_path, pilot_variable)
        if pilot_complex.ndim != 4:
            raise ValueError(f"Expected pilot CSI to be 4D after MATLAB transpose, got shape {pilot_complex.shape}")
        if shuffle_seed is not None:
            indices = np.arange(pilot_complex.shape[0])
            np.random.RandomState(shuffle_seed).shuffle(indices)
            pilot_complex = pilot_complex[indices]
        if limit is not None:
            pilot_complex = pilot_complex[:limit]
        self.pilot_complex = pilot_complex
        self.h_pilot = complex_to_channels(pilot_complex)

        if full_path is not None:
            full_complex = load_complex_mat(self.full_path, full_variable)
            if full_complex.ndim != 4:
                raise ValueError(f"Expected full CSI to be 4D after MATLAB transpose, got shape {full_complex.shape}")
            if shuffle_seed is not None:
                indices = np.arange(full_complex.shape[0])
                np.random.RandomState(shuffle_seed).shuffle(indices)
                full_complex = full_complex[indices]
            if limit is not None:
                full_complex = full_complex[:limit]
            if full_complex.shape[0] != self.h_pilot.shape[0]:
                raise ValueError(
                    f"Pilot sample count {self.h_pilot.shape[0]} does not match full sample count {full_complex.shape[0]}"
                )
            self.full_complex = full_complex
            self.h_full = complex_to_channels(full_complex)
            self.target_size = tuple(int(dim) for dim in self.h_full.shape[-3:])
        else:
            self.full_complex = None
            self.h_full = None
            if target_size is None:
                raise ValueError("target_size is required when full_path is not provided.")
            self.target_size = tuple(int(dim) for dim in target_size)

        self.h_inter = F.interpolate(self.h_pilot, size=self.target_size, mode="trilinear", align_corners=False)

    def __len__(self):
        return int(self.h_pilot.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = {
            "h_pilot": self.h_pilot[index].float(),
            "h_inter": self.h_inter[index].float(),
        }
        if self.h_full is not None:
            sample["h_full"] = self.h_full[index].float()
        return sample


class ChannelPredictionDataset(Dataset):
    """Observed snapshots and future targets from pre-saved split files.

    ``csi_path`` holds ``X`` of shape ``[N, 2, T_p, H, W]`` (observed snapshots) and
    ``label_path`` holds ``Y`` of shape ``[N, 2, h, H, W]`` (the ``h`` following
    snapshots), both as real/imaginary float arrays in MATLAB (scipy) files. The model
    input ``h_full`` is ``X`` extended with ``h`` zero snapshots, so the backbone always
    sees ``T_p + h`` frames with the future slots zeroed; ``target_future`` is ``Y``.

    Args:
        limit: train on a seeded random subset of ``limit`` samples (label-fraction study);
            the subset keeps the original order and is a no-op when ``limit >= N``.
        shuffle_seed: seed of that subset (defaults to 100).
    """

    def __init__(self, csi_path: str, label_path: str, limit: Optional[int] = None,
                 shuffle_seed: Optional[int] = None):
        super().__init__()
        for path in (csi_path, label_path):
            if not Path(path).exists():
                raise FileNotFoundError(f"Downstream data file not found: {path}. See README.md (Data).")
        X = sio.loadmat(str(csi_path))["X"].astype(np.float32)
        Y = sio.loadmat(str(label_path))["Y"].astype(np.float32)

        if limit is not None and int(limit) < X.shape[0]:
            generator = np.random.RandomState(100 if shuffle_seed is None else int(shuffle_seed))
            selected = np.sort(generator.permutation(X.shape[0])[:int(limit)])
            X, Y = X[selected], Y[selected]
        horizon = Y.shape[2]
        zeros = np.zeros((X.shape[0], X.shape[1], horizon, X.shape[3], X.shape[4]), np.float32)
        self.h_full = torch.from_numpy(np.concatenate([X, zeros], axis=2)).float()
        self.horizon = horizon
        self.observed_snapshots = int(X.shape[2])
        self.target_future = torch.from_numpy(Y).float()

    def __len__(self):
        return int(self.h_full.shape[0])

    def __getitem__(self, index):
        return {
            "h_full": self.h_full[index],
            "target_future": self.target_future[index],
        }


def build_downstream_datasets(task: str, data_dir: str, seed: int = 100,
                              train_limit: Optional[int] = None) -> Dict[str, Dataset]:
    """Build the train / val / test datasets of a downstream task from ``data_dir``.

    ``channel_estimation`` expects ``X_pilot_{train,val,test}.mat`` and
    ``X_{train,val}.mat`` (plus ``X_test.mat`` when the test split is labelled);
    ``channel_prediction`` expects ``{train,val,test}.mat`` and ``{train,val,test}_label.mat``.
    """
    data_dir = Path(data_dir)
    if task == "channel_estimation":
        train_dataset = ChannelEstimationDataset(
            data_dir / "X_pilot_train.mat", data_dir / "X_train.mat", limit=train_limit, shuffle_seed=seed)
        val_dataset = ChannelEstimationDataset(
            data_dir / "X_pilot_val.mat", data_dir / "X_val.mat", shuffle_seed=seed)
        test_full = data_dir / "X_test.mat"
        test_dataset = ChannelEstimationDataset(
            data_dir / "X_pilot_test.mat", test_full if test_full.exists() else None,
            target_size=train_dataset.target_size)
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    if task == "channel_prediction":
        train_dataset = ChannelPredictionDataset(
            data_dir / "train.mat", data_dir / "train_label.mat", limit=train_limit, shuffle_seed=seed)
        val_dataset = ChannelPredictionDataset(data_dir / "val.mat", data_dir / "val_label.mat")
        test_dataset = ChannelPredictionDataset(data_dir / "test.mat", data_dir / "test_label.mat")
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    raise ValueError(f"Unknown downstream task '{task}'. Choose 'channel_estimation' or 'channel_prediction'.")


# ==============================================================================
# command-line tool: build the downstream split files from raw CSI sequences
# ==============================================================================
# Input: a .npy / .npz (key "H") or MATLAB .mat array of complex CSI sequences of shape
# (N, T, H, W) = (sequences, snapshots, antennas, subcarriers). The paper recipe is applied:
#   * sequences are split in order into train / val / test (800 / 100 / 100 in the paper);
#   * prediction T_p -> h: the first --window snapshots (16) are kept, every sequence is divided
#     by its complex RMS over those snapshots, {split}.mat holds X [n, 2, T_p, H, W] and
#     {split}_label.mat holds Y [n, 2, h, H, W] (real / imaginary channels);
#   * channel estimation: the first --ce-frames snapshots (4), RMS-normalised over those
#     snapshots; X_{split}.mat (full CSI) and X_pilot_{split}.mat (every --pilot-stride-th
#     subcarrier along the last axis) as complex MATLAB v7.3 arrays.
# The files match the paper's split files up to float32 rounding (< 4e-7 on unit-RMS data).
#
#   python input_preprocess.py sequences.npz --out data --prediction 3:1 2:2 4:1 11:1 \
#       --depth-sweep 5:1-11 --channel-estimation

def load_sequences(path: str, key: str = "H") -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix == ".npz":
        return np.load(path, allow_pickle=True)[key]
    if path.suffix == ".mat":
        try:
            return sio.loadmat(str(path))[key]
        except NotImplementedError:
            with h5py.File(path, "r") as handle:
                raw = handle[key][()]
            if raw.dtype.fields and "real" in raw.dtype.fields:
                raw = raw["real"] + 1j * raw["imag"]
            return np.transpose(raw, tuple(range(raw.ndim - 1, -1, -1)))
    raise ValueError(f"Unsupported input file: {path}")


def real_imag(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real, x.imag], 1).astype(np.float32)


def write_complex_v73(path: Path, name: str, array: np.ndarray) -> None:
    """Write a complex array as MATLAB v7.3 (HDF5) so that ``load_complex_mat`` restores its shape."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset(name, data=np.ascontiguousarray(np.transpose(array)).astype(np.complex64))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the downstream split files (channel estimation and prediction) from raw CSI sequences; see README.md (Data).")
    parser.add_argument("sequences", help="complex CSI array (N, T, H, W): .npy, .npz or .mat")
    parser.add_argument("--key", default="H", help="array name inside .npz / .mat files")
    parser.add_argument("--out", required=True, help="output root (data/downstream)")
    parser.add_argument("--window", type=int, default=16, help="snapshots kept per sequence")
    parser.add_argument("--splits", default="800,100,100", help="train,val,test sequence counts (in order)")
    parser.add_argument("--prediction", nargs="*", default=[], metavar="TP:H",
                        help="prediction settings, e.g. 3:1 2:2 4:1 11:1")
    parser.add_argument("--depth-sweep", default=None, metavar="TP:S1-S2",
                        help="observe TP snapshots and predict s = S1..S2 snapshots, e.g. 5:1-11")
    parser.add_argument("--channel-estimation", action="store_true", help="also write the channel-estimation files")
    parser.add_argument("--ce-frames", type=int, default=4, help="snapshots per channel-estimation sample")
    parser.add_argument("--pilot-stride", type=int, default=4, help="pilot every k-th subcarrier (last axis)")
    parser.add_argument("--no-normalize", action="store_true", help="skip the per-sequence RMS normalisation")
    args = parser.parse_args()

    sequences = np.asarray(load_sequences(args.sequences, args.key))
    if sequences.ndim != 4:
        raise ValueError(f"expected (N, T, H, W), got {sequences.shape}")
    sequences = sequences.astype(np.complex64)
    counts = [int(c) for c in args.splits.split(",")]
    if sum(counts) > sequences.shape[0]:
        raise ValueError(f"splits need {sum(counts)} sequences, only {sequences.shape[0]} available")
    bounds = np.cumsum([0] + counts)
    out = Path(args.out)

    def windowed(num_frames: int):
        """First ``num_frames`` snapshots of every split, RMS-normalised per sequence over those snapshots."""
        if num_frames > sequences.shape[1]:
            raise ValueError(f"{num_frames} snapshots requested, only {sequences.shape[1]} available")
        window = sequences[:, :num_frames]
        if not args.no_normalize:
            power = np.mean(np.abs(window) ** 2, axis=(1, 2, 3), keepdims=True)
            window = window / np.sqrt(power).clip(1e-12)
        return {name: window[bounds[i]:bounds[i + 1]] for i, name in enumerate(("train", "val", "test"))}

    splits = windowed(args.window)

    settings = [tuple(int(v) for v in s.split(":")) for s in args.prediction]
    folders = {f"prediction_{tp}_to_{h}": (tp, h) for tp, h in settings}
    if args.depth_sweep:
        tp, span = args.depth_sweep.split(":")
        lo, hi = (int(v) for v in span.split("-"))
        for s in range(lo, hi + 1):
            folders[f"prediction_observe{tp}_depth{s}"] = (int(tp), s)
    for folder, (tp, h) in folders.items():
        if tp + h > args.window:
            raise ValueError(f"{folder}: {tp} + {h} snapshots exceed the {args.window}-snapshot window")
        target = out / folder
        target.mkdir(parents=True, exist_ok=True)
        for split, seq in splits.items():
            sio.savemat(target / f"{split}.mat", {"X": real_imag(seq[:, :tp])})
            sio.savemat(target / f"{split}_label.mat", {"Y": real_imag(seq[:, tp:tp + h])})
        print(f"{folder}: observed {tp}, horizon {h} -> {target}")

    if args.channel_estimation:
        target = out / "channel_estimation"
        target.mkdir(parents=True, exist_ok=True)
        for split, full in windowed(args.ce_frames).items():
            write_complex_v73(target / f"X_{split}.mat", f"X_{split}", full)
            write_complex_v73(target / f"X_pilot_{split}.mat", f"X_pilot_{split}", full[..., ::args.pilot_stride])
        print(f"channel_estimation: {args.ce_frames} snapshots, pilots every {args.pilot_stride}-th subcarrier -> {target}")


if __name__ == "__main__":
    main()
