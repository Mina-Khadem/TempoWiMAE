# TempoWiMAE

Masked-autoencoder pretraining for dynamic channel state information (CSI) with
parameter-free factored space-time attention. The pretrained encoder is frozen and
evaluated with lightweight heads on channel estimation and channel prediction.

# [Paper](PAPER_URL) · [arXiv](ARXIV_URL) · [Checkpoints](models/) · [Hugging Face](HUGGINGFACE_CHECKPOINT_URL)

## Overview

TempoWiMAE represents a sequence of `T` CSI snapshots over an `S x A` subcarrier-antenna
grid as a `2 x T x S x A` tensor (real and imaginary parts), partitions it into
time-subcarrier-antenna tubelets, encodes only the visible tubelets and reconstructs the
masked ones with a lightweight decoder. After pretraining the decoder is discarded.

* **Factored space-time attention through a fixed bias.** Temporal layers connect tokens
  that share the same subcarrier-antenna position across time, spatial-frequency layers
  connect tokens within the same snapshot. The restriction is a fixed pre-softmax bias
  (0 / `-inf`) added to ordinary multi-head self-attention, so it adds no parameters;
  setting the bias to zero (`attn_mode = "full"`) recovers joint attention with the same
  weights.
* **Any visible-token subset.** The bias is built from the `(t, s, a)` grid index of every
  token, so it works with the irregular token sets left by random masking.
* **Factored positional encoding** `p(t, s, a) = p_T(t) + p_S(s, a)` (fixed sinusoidal).
* **Hybrid masking** over six patterns (random, tube, frequency-tube, antenna-tube,
  temporal-frame, temporal-tail); one pattern is drawn per batch with configurable weights.
* **Frozen-encoder evaluation** with a two-layer per-token MLP head for channel estimation
  from sparse pilots and for channel prediction with several observation lengths and
  horizons (3 -> 1, 2 -> 2, 4 -> 1, 11 -> 1, prediction-depth sweep).

The paper configuration is the nano model: 6 encoder blocks, `D = 64`, 4 heads, MLP
ratio 2, 4 decoder blocks (0.210M encoder / 0.356M total parameters), pretrained on the
WiFo datasets D4, D7, D12, D16 (`16 x 32 x 32` samples, patch `4 x 4`, tubelet 4) with
the six-pattern hybrid masking at ratio 0.60 for 200 epochs.

## Files

| File | Content |
|---|---|
| `config.py` | Paper settings for pretraining, channel estimation and channel prediction; every entry lists its alternatives as a comment (model size, attention mode and scope, positional encoding, masking patterns / weights / ratios, tubelet, schedule, readout, head depth, fine-tuning, label fraction). |
| `tempowimae_model.py` | `TempoWiMAEConfig` (parameter-free options), positional encodings, Transformer blocks with the factored attention bias (`build_factored_biases`), `TempoWiMAEEncoder` / `TempoWiMAEDecoder` / `TempoWiMAE`, `MODEL_SIZES`, `build_model`, checkpoint reading / writing. |
| `masking.py` | The six masking patterns, `build_batch_mask`, `pick_pattern`, optional mask-ratio curriculum. |
| `input_preprocess.py` | `CSIDataset` (MATLAB v7.3 pretraining files), `ChannelEstimationDataset`, `ChannelPredictionDataset`, and a command-line tool that builds the downstream split files from raw CSI sequences. |
| `train.py` | Pretraining: targets, loss / NMSE, learning-rate schedule, hybrid masking loop, validation, checkpoints. |
| `downstream.py` | Frozen-encoder backbone, task heads, losses / metrics, head training with validation-based selection and test reporting. |
| `inference.py` | Load a checkpoint, encode CSI samples, report the masked-reconstruction NMSE. |
| `utils.py` | CSI tensor helpers (unpatchify, input normalisation, complex <-> channels), seeding, logging, configuration overrides. |
| `models/` | Released checkpoints (see below). |

## Installation

Tested with Python 3.10, PyTorch 2.5.1 (CUDA 12.4) and the versions in `requirements.txt`
on Ubuntu 22.04 with an NVIDIA RTX 6000 Ada GPU. A GPU is used when available.

```bash
git clone REPOSITORY_URL
cd TempoWiMAE
conda create -n tempowimae python=3.10 -y
conda activate tempowimae
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124   # or the CPU wheels
pip install -r requirements.txt
```

## Data

No dataset is redistributed. All CSI samples are handled as real tensors `[2, T, H, W]`
(channel 0 real part, channel 1 imaginary part, `T` snapshots, `H x W` grid).

**Pretraining: WiFo datasets D4, D7, D12, D16** (Liu et al., "WiFo: Wireless foundation
model for channel prediction", *Science China Information Sciences*, 2025; generated with
QuaDRiGa). Arrange the files as

```text
data/wifo/D4/X_train.mat  data/wifo/D4/X_val.mat   (likewise D7, D12, D16)
```

MATLAB v7.3 files with one variable each, shape `(32, 32, 16, N)` through h5py
(subcarriers, antennas, snapshots, samples); complex values stored as a compound
`real` / `imag` dtype or a native complex dtype. The loader produces `[2, 16, 32, 32]`
tensors; no normalisation is applied. The paper uses 9 000 training samples per dataset.
Point the code at the folder with `--data_root` or `TEMPOWIMAE_DATA_ROOT`.

**Downstream: ray-traced urban CSI sequences** (Sionna RT, Soho, London, 5.9 GHz,
`32 x 32` antenna-subcarrier grid, 0.5 ms snapshot spacing, 1 000 sequences split in order
into 800 / 100 / 100 train / val / test). Given any complex array of CSI sequences of shape
`(N, T, H, W)` (`.npy`, `.npz` with key `H`, or `.mat`), the paper's split files are built with

```bash
python input_preprocess.py sequences.npz --out data \
    --prediction 3:1 2:2 4:1 11:1 --depth-sweep 5:1-11 --channel-estimation
```

which writes

```text
data/prediction_3_to_1/              train.mat train_label.mat val.mat val_label.mat test.mat test_label.mat
data/prediction_2_to_2/  data/prediction_4_to_1/  data/prediction_11_to_1/
data/prediction_observe5_depth<s>/   s = 1 .. 11
data/channel_estimation/             X_pilot_{train,val,test}.mat  X_{train,val,test}.mat
```

* Prediction `T_p -> h`: the first 16 snapshots of every sequence are kept and the sequence
  is divided by its complex RMS over them; `X [n, 2, T_p, H, W]` holds snapshots `1 .. T_p`
  and `Y [n, 2, h, H, W]` snapshots `T_p+1 .. T_p+h` (float32, real / imaginary channels,
  `scipy.io` format). At run time `h` zero snapshots are appended to `X`.
* Channel estimation: the first 4 snapshots, RMS-normalised over those 4; `X_*` is the full
  complex CSI `(n, 4, 32, 32)` and `X_pilot_*` every fourth subcarrier `(n, 4, 32, 8)`
  (MATLAB v7.3). The pilot tensor is interpolated to the full grid (trilinear) and is the
  model input; the full CSI is the target.
* Label-fraction study: `--train_limit 40 | 80 | 200 | 400` trains the head on a seeded
  subset of the 800 training samples.

## Pretrained Checkpoints

| File | Recipe | Epochs | Role in the paper |
|---|---|---|---|
| `models/tempowimae_nano_paper_200ep.pth` | `config.PRETRAIN` | 200 | off-the-shelf checkpoint |
| `models/tempowimae_nano_matched_25ep.pth` | `config.PRETRAIN` with `epochs = 25` | 25 | matched-budget comparison with the baselines |

Each file is a `torch.save` dictionary with the full pretraining model under `"model"`
(encoder, decoder, encoder-to-decoder projection, mask token; 356 352 parameters), the
architecture options under `"model_config"` and the training configuration under
`"config"`. No downstream heads are included; they are trained by `downstream.py`.
The same files are mirrored at [Hugging Face](HUGGINGFACE_CHECKPOINT_URL).

```python
from tempowimae_model import build_model, read_checkpoint, checkpoint_model_config, load_pretrained_encoder

ckpt = read_checkpoint("models/tempowimae_nano_paper_200ep.pth")
model = build_model("nano", input_size=(32, 32), patch_size=(4, 4), tubelet_size=4, num_frames=16,
                    model_config=checkpoint_model_config(ckpt))
load_pretrained_encoder(model, ckpt)     # encoder weights (load_full_model loads the decoder too)
encoder = model.encoder                  # tokens = encoder(x, mask); x: [B, 2, T, 32, 32], mask: [B, N] bool (True = masked)
```

`python inference.py --input data/wifo/D4/X_val.mat --num_samples 8` encodes a few samples
and prints the masked-reconstruction NMSE.

## Pretraining

```bash
python train.py --data_root data/wifo                 # paper configuration (config.PRETRAIN), 200 epochs
python train.py --data_root data/wifo --epochs 25     # matched budget
python train.py --data_root data/wifo --set model.attn_mode=full                                   # joint-attention control
python train.py --data_root data/wifo --set masking.patterns=[temporal_frame] --set masking.weights=[1.0]   # single pattern
python train.py --data_root data/wifo --max_samples_per_file 64 --epochs 2 --batch_size 16        # quick check
```

Every run writes `outputs/pretraining/<experiment_name>/<timestamp>/` with the resolved
`config.json`, `run_config.json`, `log.txt`, `training_history.json` (train / validation
reconstruction NMSE per epoch), TensorBoard events and `checkpoint-<epoch>.pth` every 25
epochs. The 200-epoch run takes about 50 minutes on the verification GPU.

## Channel Estimation

Frozen encoder, `T = 4`, pilots on 8 of 32 subcarriers, two-layer per-token MLP head,
200 epochs of Adam (lr 1e-4) with early stopping on the validation NMSE; the test NMSE is reported.

```bash
python downstream.py --task channel_estimation --data_dir data/channel_estimation
```

Output: `outputs/channel_estimation/channel_estimation/<run>/results.json`
(`best_val_nmse`, `test_nmse`, `test_nmse_db`), `training_history.json`, `model_save/head_best.pth`.

## Channel Prediction

The `T_p` observed snapshots are followed by zeroed future slots, the frozen encoder
embeds the window, the two-layer per-token head reconstructs it and the predicted snapshots
are scored by NMSE (Adam lr 1e-3, 200 epochs).

```bash
python downstream.py --task channel_prediction --setting 3_to_1  --data_dir data/prediction_3_to_1
python downstream.py --task channel_prediction --setting 2_to_2  --data_dir data/prediction_2_to_2
python downstream.py --task channel_prediction --setting 4_to_1  --data_dir data/prediction_4_to_1
python downstream.py --task channel_prediction --setting 11_to_1 --data_dir data/prediction_11_to_1
# prediction-depth sweep, s = 1 .. 11; the score of snapshot 5+s is test_nmse_per_snapshot[-1] in results.json
python downstream.py --task channel_prediction --setting depth_sweep --horizon 3 --data_dir data/prediction_observe5_depth3
# matched-budget checkpoint, label fraction, supervised-from-scratch control
python downstream.py --task channel_prediction --setting 3_to_1 --checkpoint models/tempowimae_nano_matched_25ep.pth --data_dir data/prediction_3_to_1
python downstream.py --task channel_prediction --setting 3_to_1 --train_limit 40 --data_dir data/prediction_3_to_1
python downstream.py --task channel_prediction --setting 3_to_1 --set model.checkpoint=null --set model.finetune_mode=full --data_dir data/prediction_3_to_1
```

Readout (`config.CHANNEL_PREDICTION["prediction"]["readout"]`): `end_pad` (paper) zero-extends
the horizon at the window end to the next tubelet multiple (3 -> 1 and 2 -> 2 use 4 frames,
4 -> 1 uses 8, 11 -> 1 uses 12 at tubelet 4); `causal_pad` prepends zeros so the target starts a
new tubelet; `none` requires `T_p + h` to be a tubelet multiple.

## Reproducing Paper Results

Test NMSE obtained with this code and the released checkpoints (frozen encoder, 800 training
labels, seed 100). The values are identical to those behind the paper's figures and were
reproduced bit-for-bit on the verification machine; on other hardware the last digits may differ.
One run takes two to three minutes on the verification GPU.

| Task | Command | Off-the-shelf (200 ep) NMSE (dB) | Matched (25 ep) NMSE (dB) |
|---|---|---|---|
| Channel estimation | `--task channel_estimation` | 0.02681 (-15.72) | 0.02373 (-16.25) |
| Prediction 3 -> 1 | `--setting 3_to_1` | 0.00800 (-20.97) | 0.00885 (-20.53) |
| Prediction 2 -> 2 | `--setting 2_to_2` | 0.02158 (-16.66) | 0.02170 (-16.64) |
| Prediction 4 -> 1 | `--setting 4_to_1` | 0.01778 (-17.50) | 0.02308 (-16.37) |
| Prediction 11 -> 1 | `--setting 11_to_1` | 0.00768 (-21.15) | 0.00779 (-21.08) |

Ablations of the paper: attention topology and positional encoding (Table II) are pretrained with
`--set model.attn_mode=full`, `--set model.factored_scope=encoder` or `--set model.pos_embed_mode=flat`;
masking patterns (Fig. 2) with a single pattern and three seeds (`--seed 100 | 101 | 102`);
label fractions (Fig. 3) with `--train_limit`; the depth sweep (Fig. 4) with `--setting depth_sweep`.
The WiFo and LWM-temporal baselines are evaluated with their authors' code and are not included.

## Paper-to-Code Map

| Paper component | Implementation |
|---|---|
| Complex CSI input, real / imaginary channels (Sec. II-A) | `input_preprocess.py::CSIDataset`, `utils.py::complex_to_channels` |
| Tubelet tokenization (Sec. II-B) | `tempowimae_model.py::PatchEmbed` |
| Factored positional encoding (Sec. II-C) | `tempowimae_model.py::get_factored_sincos_pos_embed` |
| Factored space-time attention bias, alternating layers (Sec. II-D) | `tempowimae_model.py::build_factored_biases`, `Attention.forward`, `TempoWiMAEEncoder.forward_features`, `TempoWiMAEDecoder.forward_features` |
| Masking patterns and hybrid selection (Sec. II-E, Table I) | `masking.py`, `config.py::PRETRAIN["masking"]` |
| Encoder / decoder / reconstruction loss (Eq. 8-11) | `tempowimae_model.py::TempoWiMAE.forward`, `train.py::build_patch_targets`, `reconstruction_metrics` |
| Pretraining schedule (Table I) | `train.py::run_pretraining`, `learning_rate_at`, `config.py::PRETRAIN` |
| Frozen encoder for downstream tasks (Sec. III-C) | `downstream.py::EncoderBackbone` |
| Channel estimation (Eq. 12) | `downstream.py::ChannelEstimationHead`, `input_preprocess.py::ChannelEstimationDataset` |
| Channel prediction and depth sweep (Eq. 13-14) | `downstream.py::ChannelPredictionHead`, `prediction_window`, `input_preprocess.py::ChannelPredictionDataset` |
| NMSE losses / metrics, selection on val, reporting on test | `downstream.py::DOWNSTREAM_TASKS`, `run_downstream` |

## Citation

```bibtex
@article{khadem2026tempowimae,
  title   = {PAPER_TITLE},
  author  = {Khadem, Mina and ...},
  journal = {VENUE},
  year    = {2026}
}
```

## Contact

**Mina Khadem**

- Email: [YOUR_EMAIL](mailto:YOUR_EMAIL)
- LinkedIn: [YOUR_LINKEDIN_URL](YOUR_LINKEDIN_URL)
- Hugging Face: [YOUR_HUGGINGFACE_URL](YOUR_HUGGINGFACE_URL)

## License

This project is licensed under the Apache License 2.0 (see `LICENSE`). Parts of the
Transformer implementation are adapted from third-party code whose notices are kept in
`NOTICE` and `THIRD_PARTY_LICENSES.md`.
