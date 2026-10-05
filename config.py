"""Paper configuration of TempoWiMAE.

Every entry holds the value used in the paper; the alternatives are listed in the comment
next to it. To use another option, replace the value (or pass ``--set section.key=value``
to ``train.py`` / ``downstream.py``).
"""

# ==============================================================================
# Pretraining (Table I). This recipe produced the off-the-shelf checkpoint
# models/tempowimae_nano_paper_200ep.pth (200 epochs); the matched-budget checkpoint
# models/tempowimae_nano_matched_25ep.pth differs only in optimization.epochs = 25.
# ==============================================================================
PRETRAIN = {
    "experiment_name": "tempowimae_nano_paper",
    "seed": 100,                          # pretraining seed of the released checkpoints

    "data": {
        "root": "data/wifo",              # folder with D4/ D7/ D12/ D16/ (or --data_root / $TEMPOWIMAE_DATA_ROOT)
        "train_files": ["D4/X_train.mat", "D7/X_train.mat", "D12/X_train.mat", "D16/X_train.mat"],
        "val_files": ["D4/X_val.mat", "D7/X_val.mat", "D12/X_val.mat", "D16/X_val.mat"],
        "max_samples_per_file": 0,        # 0 = all samples per file; e.g. 64 for a quick check
        "max_val_samples_per_file": 0,
        "num_workers": 4,
    },

    "geometry": {                         # token grid = (T / tubelet) x (S / patch) x (A / patch) = 4 x 8 x 8 = 256 tokens
        "num_frames": 16,                 # T, snapshots per sample
        "subcarriers": 32,                # S
        "antennas": 32,                   # A
        "patch_size": [4, 4],             # (subcarrier, antenna) patch
        "tubelet_size": 4,                # options: 4 (paper) | 1 | 2 | 8 | 16   (must divide num_frames)
        "in_chans": 2,                    # real and imaginary parts
    },

    "model": {
        "size": "nano",                   # options: "nano" (paper) | "tiny" | "little" | "small" | "base"
        "attn_mode": "factored",          # options: "factored" (paper) | "full"   (joint attention, same parameters)
        "factored_scope": "both",         # options: "both" (paper) | "encoder" | "decoder"   (where the factored bias applies)
        "pos_embed_mode": "factored",     # options: "factored" (paper) | "flat" | "none"   (none requires use_rope True)
        "split_freq_antenna": False,      # options: False (paper) | True   (temporal / frequency / antenna three-layer cycle)
        "temporal_local_bias": False,     # options: False (paper) | True   (then set temporal_bias_window > 0)
        "temporal_bias_window": 0,
        "use_rope": False,                # options: False (paper) | True   (rotary encoding inside attention)
        "rope_axes": "temporal",          # options: "temporal" | "factored_thw"
        "rope_base": 10000.0,
    },

    "masking": {
        # Hybrid strategy: one pattern is drawn per batch with the weights below and applied to every sample.
        # Single-pattern pretraining: "patterns": ["tube"], "weights": [1.0]   (any of the six names).
        "patterns": ["random", "tube", "temporal_frame", "antenna_tube", "temporal_tail", "frequency_tube"],
        "weights":  [0.20,     0.05,   0.50,             0.05,           0.15,            0.05],
        "ratios": {                       # mask ratio per pattern (paper: 0.60 for all six)
            "random": 0.60, "tube": 0.60, "temporal_frame": 0.60,
            "antenna_tube": 0.60, "temporal_tail": 0.60, "frequency_tube": 0.60,
        },
        "temporal_ratio_random": False,   # options: False (paper) | True   (draw temporal ratios per batch from temporal_ratio_range)
        "temporal_ratio_range": [0.10, 0.55],
        "normalize_target": True,         # standardise each target tubelet before the reconstruction loss
        "curriculum": {
            "enable": False,              # options: False (paper) | True   (scale all ratios from start_scale to 1 over warmup_epochs)
            "start_scale": 0.6,
            "warmup_epochs": 5,
            "shape": "cosine",            # options: "cosine" | "linear"
        },
    },

    "optimization": {
        "epochs": 200,                    # options: 200 (off-the-shelf checkpoint) | 25 (matched-budget comparison)
        "batch_size": 128,
        "lr": 1.0e-3,
        "warmup_lr": 1.0e-4,
        "min_lr": 1.0e-5,
        "warmup_epochs": 5,               # linear warm-up, then cosine decay to min_lr
        "weight_decay": 0.05,
        "betas": [0.9, 0.95],
        "eps": 1.0e-8,
        "clip_grad": 0.05,                # gradient-norm clipping (0 disables)
        "amp": True,                      # bf16 autocast on CUDA (fp16 if bf16 is unsupported); ignored on CPU
    },

    "logging": {
        "output_dir": "outputs/pretraining",
        "monitor_every": 1,               # validate every N epochs
        "monitor_batches": 0,             # 0 = full validation set
        "save_ckpt_freq": 25,             # checkpoint every N epochs (the last epoch is always saved)
        "log_interval": 50,               # steps between console lines
        "tensorboard": True,
    },
}

# ==============================================================================
# Channel estimation (Section III-C-1): frozen encoder, two-layer per-token MLP head,
# T = 4 snapshots, pilots on 8 of the 32 subcarriers, 800 / 100 / 100 samples.
# ==============================================================================
CHANNEL_ESTIMATION = {
    "task": "channel_estimation",
    "experiment_name": "channel_estimation",

    "data": {
        "dir": "data/channel_estimation", # X_pilot_{train,val,test}.mat + X_{train,val,test}.mat (README, Data); or --data_dir
        "train_limit": None,              # options: None (all 800 labels, paper) | 400 | 200 | 80 | 40   (label-fraction study)
    },

    "geometry": {
        "num_frames": 4,                  # T, snapshots per sample
        "input_size": [32, 32],           # (height, width) of one snapshot: the 32 x 32 antenna-subcarrier grid
        "patch_size": [4, 4],
        "tubelet_size": 4,                # options: 4 (paper, matches pretraining) | 1 | 2   (must divide num_frames)
        "in_chans": 2,
    },

    "model": {
        "size": "nano",                   # options: "nano" (paper) | "tiny" | "little" | "small" | "base"   (must match the checkpoint)
        "checkpoint": "models/tempowimae_nano_paper_200ep.pth",   # options: "models/tempowimae_nano_matched_25ep.pth" (matched budget) | None (random init); or --checkpoint
        "architecture": "auto",           # "auto" = use the attention / positional-encoding options stored in the checkpoint
        "finetune_mode": "none",          # options: "none" (frozen encoder, paper) | "partial" | "full"
        "finetune_layers": [],            # with finetune_mode "partial", e.g. ["blocks.5", "norm"]
        "input_normalization": "rms",     # options: "rms" (paper, channel estimation) | "none" | "standardized"
        "input_normalization_scope": "global",   # options: "global" | "patch"
    },

    "training": {
        "epochs": 200,
        "lr": 1.0e-4,                     # constant learning rate (Adam)
        "min_lr": 1.0e-6,
        "batch_size": 64,
        "early_stop": 50,                 # patience (epochs without validation improvement)
        "num_workers": 8,
        "seed": 100,
    },

    "output": {
        "dir": "outputs/channel_estimation",
        "save_predictions": False,        # True also writes predictions_test.npz
    },
}

# ==============================================================================
# Channel prediction (Section III-C-2/3): frozen encoder, two-layer per-token MLP head,
# 800 / 100 / 100 sequences. The observation length and horizon are chosen with
# --setting (PREDICTION_SETTINGS) or --observed_snapshots / --horizon.
# ==============================================================================
PREDICTION_SETTINGS = {                   # name: (observed snapshots T_p, horizon h)
    "3_to_1": (3, 1),
    "2_to_2": (2, 2),
    "4_to_1": (4, 1),
    "11_to_1": (11, 1),
    "depth_sweep": (5, 1),                # 5 observed snapshots, --horizon s = 1 .. 11; score = NMSE of snapshot 5 + s
}

CHANNEL_PREDICTION = {
    "task": "channel_prediction",
    "experiment_name": "prediction_3_to_1",   # set from --setting

    "data": {
        "dir": "data/prediction_3_to_1",  # {train,val,test}.mat + {train,val,test}_label.mat (README, Data); or --data_dir
        "train_limit": None,              # options: None (all 800 labels, paper) | 400 | 200 | 80 | 40   (label-fraction study)
    },

    "geometry": {
        "input_size": [32, 32],           # (height, width) of one snapshot: the 32 x 32 antenna-subcarrier grid
        "patch_size": [4, 4],
        "tubelet_size": 4,                # options: 4 (paper, matches pretraining) | 1 | 2
        "in_chans": 2,
    },

    "prediction": {
        "observed_snapshots": 3,          # T_p (set from --setting / --observed_snapshots)
        "horizon": 1,                     # h   (set from --setting / --horizon)
        "readout": "end_pad",             # options: "end_pad" (paper) | "causal_pad" | "none"   (see README, Channel Prediction)
        "head_mlp_layers": 2,             # options: 2 (paper) | 1 (linear probe) | 3
    },

    "model": {
        "size": "nano",                   # options: "nano" (paper) | "tiny" | "little" | "small" | "base"   (must match the checkpoint)
        "checkpoint": "models/tempowimae_nano_paper_200ep.pth",   # options: "models/tempowimae_nano_matched_25ep.pth" (matched budget) | None (random init); or --checkpoint
        "architecture": "auto",           # "auto" = use the attention / positional-encoding options stored in the checkpoint
        "finetune_mode": "none",          # options: "none" (frozen encoder, paper) | "partial" | "full"
        "finetune_layers": [],            # with finetune_mode "partial", e.g. ["blocks.5", "norm"]
        "input_normalization": "none",    # options: "none" (paper, prediction) | "rms" | "standardized"
        "input_normalization_scope": "global",   # options: "global" | "patch"
    },

    "training": {
        "epochs": 200,
        "lr": 1.0e-3,                     # constant learning rate (Adam)
        "min_lr": 1.0e-6,
        "batch_size": 64,
        "early_stop": 1000,               # patience larger than the epoch count: no early stopping (paper)
        "num_workers": 8,
        "seed": 100,
    },

    "output": {
        "dir": "outputs/channel_prediction",
        "save_predictions": False,        # True also writes predictions_test.npz
    },
}
