"""TempoWiMAE model: architecture options, positional encodings, Transformer blocks with the
factored attention bias, encoder / decoder / masked autoencoder, model sizes and checkpoint utilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import drop_path, to_2tuple as timm_to_2tuple
from timm.layers import trunc_normal_ as _timm_trunc_normal_

from utils import to_2tuple


# ==============================================================================
# config: Parameter-free architecture options of TempoWiMAE.
# ==============================================================================

@dataclass
class TempoWiMAEConfig:
    """Attention connectivity and positional-encoding options.

    None of these options adds trainable parameters, so one set of weights can be
    rebuilt under any combination. Checkpoints therefore store this configuration,
    and the loader rebuilds the model from it (``read_checkpoint`` / ``checkpoint_model_config`` below).

    Attributes:
        attn_mode: ``"factored"`` alternates temporal and spatial-frequency attention
            across Transformer layers through a fixed pre-softmax bias (paper);
            ``"full"`` removes the bias and recovers joint attention.
        factored_scope: where the factored bias applies: ``"both"`` (paper),
            ``"encoder"`` or ``"decoder"`` (the other stage runs joint attention).
        split_freq_antenna: replace the spatial-frequency layer by separate frequency
            and antenna layers (temporal / frequency / antenna three-layer cycle).
        temporal_local_bias: restrict temporal attention to
            ``|t_i - t_j| <= temporal_bias_window``.
        temporal_bias_window: half-width of the temporal band (``> 0`` when enabled).
        pos_embed_mode: additive sinusoidal positional encoding: ``"factored"``
            (temporal + subcarrier-antenna, paper), ``"flat"`` (1-D over the token
            index) or ``"none"`` (no additive encoding; requires ``use_rope``).
        use_rope: rotary positional encoding applied to queries and keys.
        rope_axes: ``"temporal"`` rotates by the time index only, ``"factored_thw"``
            rotates separate sub-blocks of each head by the (t, h, w) grid index.
        rope_base: rotary frequency base.
    """

    attn_mode: str = "factored"        # options: factored (paper) | full
    factored_scope: str = "both"       # options: both (paper) | encoder | decoder
    split_freq_antenna: bool = False
    temporal_local_bias: bool = False
    temporal_bias_window: int = 0
    pos_embed_mode: str = "factored"   # options: factored (paper) | flat | none
    use_rope: bool = False
    rope_axes: str = "temporal"        # options: temporal | factored_thw
    rope_base: float = 10000.0

    def validate(self) -> "TempoWiMAEConfig":
        """Reject inconsistent option combinations."""
        if self.use_rope and self.rope_axes not in ("temporal", "factored_thw"):
            raise ValueError(f"rope_axes must be 'temporal' or 'factored_thw', got {self.rope_axes}")
        if self.attn_mode not in ("full", "factored"):
            raise ValueError(f"attn_mode must be 'full' or 'factored', got {self.attn_mode}")
        if self.factored_scope not in ("both", "encoder", "decoder"):
            raise ValueError(f"factored_scope must be 'both', 'encoder' or 'decoder', got {self.factored_scope}")
        if (self.split_freq_antenna or self.temporal_local_bias) and self.attn_mode != "factored":
            raise ValueError(
                "split_freq_antenna and temporal_local_bias refine factored attention; set attn_mode='factored'."
            )
        if self.temporal_local_bias and self.temporal_bias_window <= 0:
            raise ValueError("temporal_local_bias=True requires temporal_bias_window > 0.")
        if self.pos_embed_mode not in ("factored", "flat", "none"):
            raise ValueError(f"pos_embed_mode must be 'factored', 'flat' or 'none', got {self.pos_embed_mode}")
        if self.pos_embed_mode == "none" and not self.use_rope:
            raise ValueError("pos_embed_mode='none' removes all position information unless use_rope=True.")
        return self

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TempoWiMAEConfig":
        """Build a configuration from a mapping, ignoring keys that are not options."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def coerce_config(cfg: Optional[Any]) -> TempoWiMAEConfig:
    """Accept ``None`` (defaults), a mapping or a :class:`TempoWiMAEConfig`."""
    if cfg is None:
        return TempoWiMAEConfig()
    if isinstance(cfg, TempoWiMAEConfig):
        return cfg
    if isinstance(cfg, dict):
        return TempoWiMAEConfig.from_dict(cfg)
    raise TypeError(f"model_config must be None, a dict or TempoWiMAEConfig; got {type(cfg)}")


# ==============================================================================
# positional: Positional encodings: factored / flat sinusoidal embeddings and rotary encoding.
# ==============================================================================

def _sincos_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """1-D sinusoidal embedding of ``positions`` ``[M]`` into ``[M, embed_dim]``."""
    assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = np.outer(positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_factored_sincos_pos_embed(embed_dim: int, num_t: int, num_h: int, num_w: int) -> torch.Tensor:
    """Factored positional encoding ``p(t, h, w) = p_T(t) + p_S(h, w)`` of shape ``[1, T*H*W, D]``.

    ``p_S`` concatenates a sinusoidal encoding of the subcarrier index (first half of
    the dimensions) and of the antenna index (second half). Token order matches the
    Conv3d flatten order (time outer, height middle, width inner).
    """
    t_embed = _sincos_1d(embed_dim, np.arange(num_t, dtype=np.float32))

    d_h = embed_dim // 2
    d_w = embed_dim - d_h
    h_embed = _sincos_1d(d_h, np.arange(num_h, dtype=np.float32))
    w_embed = _sincos_1d(d_w, np.arange(num_w, dtype=np.float32))

    hw_embed = np.concatenate(
        [np.tile(h_embed[:, None, :], (1, num_w, 1)),
         np.tile(w_embed[None, :, :], (num_h, 1, 1))],
        axis=-1,
    ).reshape(num_h * num_w, embed_dim)

    spatial = np.tile(hw_embed[None, :, :], (num_t, 1, 1))
    temporal = np.tile(t_embed[:, None, :], (1, num_h * num_w, 1))

    pos = (spatial + temporal).reshape(num_t * num_h * num_w, embed_dim)
    return torch.tensor(pos, dtype=torch.float, requires_grad=False).unsqueeze(0)


def get_flat_sincos_pos_embed(embed_dim: int, num_positions: int) -> torch.Tensor:
    """1-D sinusoidal encoding over the flattened token index, ignoring the (t, h, w) grid."""
    pos = _sincos_1d(embed_dim, np.arange(num_positions, dtype=np.float32))
    return torch.tensor(pos, dtype=torch.float, requires_grad=False).unsqueeze(0)


def build_pos_embed(mode: str, embed_dim: int, num_t: int, num_h: int, num_w: int) -> torch.Tensor:
    """Select the additive positional encoding by ``TempoWiMAEConfig.pos_embed_mode``."""
    if mode == "factored":
        return get_factored_sincos_pos_embed(embed_dim, num_t, num_h, num_w)
    if mode == "flat":
        return get_flat_sincos_pos_embed(embed_dim, num_t * num_h * num_w)
    if mode == "none":
        return torch.zeros(1, num_t * num_h * num_w, embed_dim, requires_grad=False)
    raise ValueError(f"unknown pos_embed_mode {mode!r}")


def build_grid_positions(num_t: int, num_h: int, num_w: int) -> torch.Tensor:
    """``[N, 3]`` grid index ``(t, h, w)`` per token in Conv3d flatten order."""
    t = torch.arange(num_t).view(num_t, 1, 1).expand(num_t, num_h, num_w)
    h = torch.arange(num_h).view(1, num_h, 1).expand(num_t, num_h, num_w)
    w = torch.arange(num_w).view(1, 1, num_w).expand(num_t, num_h, num_w)
    return torch.stack([t, h, w], dim=-1).reshape(num_t * num_h * num_w, 3).long()


# --- rotary positional encoding -------------------------------------------------
# After masking the encoder sees tokens in a gathered order, so rotation uses each
# token's true grid index rather than its position in the packed sequence.

def _rope_cos_sin(pos: torch.Tensor, dim: int, base: float, dtype: torch.dtype):
    """``pos`` ``[B, N]`` grid index along one axis -> cos, sin of shape ``[B, 1, N, dim]``."""
    half = dim // 2
    device = pos.device
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = pos.float()[..., None] * inv_freq[None, None, :]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos()[:, None].to(dtype), emb.sin()[:, None].to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def _split_dims_3(head_dim: int):
    """Split an even head_dim into three even parts (t, h, w); time takes the remainder."""
    base = head_dim // 3
    base -= base % 2
    d_h = d_w = max(2, base)
    d_t = head_dim - d_h - d_w
    if d_t < 2 or d_t % 2 != 0:
        d_h = d_w = 2
        d_t = head_dim - 4
    return d_t, d_h, d_w


def apply_rope(q: torch.Tensor, k: torch.Tensor, rope_pos: torch.Tensor, cfg):
    """Rotate ``q`` and ``k`` ``[B, heads, N, head_dim]`` by the grid positions ``rope_pos`` ``[B, N, 3]``."""
    head_dim = q.shape[-1]
    dtype = q.dtype
    if cfg.rope_axes == "temporal":
        if head_dim % 2 != 0:
            raise ValueError(f"temporal RoPE needs even head_dim, got {head_dim}")
        cos, sin = _rope_cos_sin(rope_pos[..., 0], head_dim, cfg.rope_base, dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
    dims = _split_dims_3(head_dim)
    q_out, k_out, offset = [], [], 0
    for axis, block_dim in enumerate(dims):
        qb, kb = q[..., offset:offset + block_dim], k[..., offset:offset + block_dim]
        cos, sin = _rope_cos_sin(rope_pos[..., axis], block_dim, cfg.rope_base, dtype)
        q_out.append(qb * cos + _rotate_half(qb) * sin)
        k_out.append(kb * cos + _rotate_half(kb) * sin)
        offset += block_dim
    return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)


# ==============================================================================
# attention: Transformer building blocks and the factored attention bias.
#
# The block structure (pre-norm attention and MLP with layer-scale) is adapted from
# third-party masked-autoencoder code; see THIRD_PARTY_LICENSES.md for attribution.
# ==============================================================================

def _allow_to_bias(allow: torch.Tensor) -> torch.Tensor:
    """Boolean connectivity ``[B, N, N]`` -> additive bias ``[B, 1, N, N]`` (0 allowed, -inf otherwise)."""
    return torch.zeros_like(allow, dtype=torch.float32).masked_fill(~allow, float("-inf"))[:, None]


def build_factored_biases(token_positions: torch.Tensor, cfg) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Build the fixed pre-softmax biases that restrict attention per layer.

    Args:
        token_positions: ``[B, N, 3]`` grid index ``(t, h, w)`` of every token in the
            sequence actually fed to the Transformer (any visible subset).
        cfg: :class:`TempoWiMAEConfig`.

    Returns:
        ``(biases, cycle)``: a bias per attention scope and the scope order applied to
        successive layers. The default cycle is ``["temporal", "spatial"]``: temporal
        layers connect tokens sharing the same subcarrier-antenna position, spatial
        layers connect tokens within the same snapshot. The diagonal is always allowed,
        so no row is ever fully masked.
    """
    t, h, w = token_positions[..., 0], token_positions[..., 1], token_positions[..., 2]
    same_t = (t[:, :, None] == t[:, None, :])
    same_h = (h[:, :, None] == h[:, None, :])
    same_w = (w[:, :, None] == w[:, None, :])

    allow_temporal = same_h & same_w
    if cfg.temporal_local_bias and cfg.temporal_bias_window > 0:
        dt = (t[:, :, None] - t[:, None, :]).abs()
        allow_temporal = allow_temporal & (dt <= cfg.temporal_bias_window)

    biases = {"temporal": _allow_to_bias(allow_temporal)}
    if cfg.split_freq_antenna:
        biases["frequency"] = _allow_to_bias(same_t & same_w)   # vary subcarrier only
        biases["antenna"] = _allow_to_bias(same_t & same_h)     # vary antenna only
        cycle = ["temporal", "frequency", "antenna"]
    else:
        biases["spatial"] = _allow_to_bias(same_t)              # vary subcarrier and antenna within a snapshot
        cycle = ["temporal", "spatial"]
    return biases, cycle


class DropPath(nn.Module):
    """Stochastic depth per sample (inactive at the paper setting ``drop_path_rate=0``)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with an optional additive connectivity bias and rotary encoding."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., model_config=None):
        super().__init__()
        self.model_config = coerce_config(model_config)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            # Query and value biases only; the key bias is a constant zero.
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope_pos=None, attention_bias=None):
        """Args:
            x: ``[B, N, D]`` tokens.
            rope_pos: ``[B, N, 3]`` grid positions, used only with rotary encoding.
            attention_bias: ``[B, 1, N, N]`` additive pre-softmax bias (0 / -inf) or ``None``.
        """
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.model_config.use_rope and rope_pos is not None:
            q, k = apply_rope(q, k, rope_pos, self.model_config)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Factored attention: disallowed token pairs receive -inf and hence zero weight.
        if attention_bias is not None:
            attn = attn + attention_bias.to(attn.dtype)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Pre-norm Transformer block with layer-scale parameters ``gamma_1`` / ``gamma_2``."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 model_config=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, model_config=model_config)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rope_pos=None, attention_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rope_pos, attention_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rope_pos, attention_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Tubelet embedding: a 3-D convolution over ``(tubelet, patch_h, patch_w)`` with matching stride.

    Input ``[B, C, T, H, W]`` -> tokens ``[B, N, D]`` with ``N = (T/p_t) * (H/p_h) * (W/p_w)``.
    """

    def __init__(self, img_size=(32, 32), patch_size=(4, 4), in_chans=2, embed_dim=64, num_frames=16, tubelet_size=4):
        super().__init__()
        img_size = timm_to_2tuple(img_size)
        patch_size = timm_to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ==============================================================================
# tempowimae: TempoWiMAE: asymmetric masked autoencoder for complex CSI sequences.
#
# The encoder processes only the visible tubelets, a lightweight decoder reconstructs
# the masked ones, and both stages can run factored (temporal / spatial-frequency)
# or joint attention under the same parameters (see :class:`TempoWiMAEConfig`).
# ==============================================================================

def trunc_normal_(tensor, mean=0., std=1.):
    _timm_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _init_linear_and_norm(module: nn.Module) -> None:
    """Xavier-uniform linear weights with zero bias; identity LayerNorm."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


class TempoWiMAEEncoder(nn.Module):
    """Encoder over the visible tubelets.

    Args:
        img_size: ``(H, W)`` subcarrier-antenna grid of one snapshot.
        patch_size: ``(p_h, p_w)`` within-snapshot patch.
        in_chans: input channels (2: real and imaginary parts).
        embed_dim, depth, num_heads, mlp_ratio: Transformer size.
        tubelet_size: number of snapshots per token.
        num_frames: snapshots per sample ``T``.
        model_config: :class:`TempoWiMAEConfig` or mapping.
    """

    def __init__(self, img_size=(32, 32), patch_size=(4, 4), in_chans=2, embed_dim=64, depth=6, num_heads=4,
                 mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, tubelet_size=4, num_frames=16, model_config=None):
        super().__init__()
        self.model_config = coerce_config(model_config)
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, num_frames=num_frames, tubelet_size=tubelet_size,
        )

        patch_h, patch_w = self.patch_embed.patch_size
        height, width = self.patch_embed.img_size
        self.num_t_patches = num_frames // int(tubelet_size)
        self.num_h_patches = height // patch_h
        self.num_w_patches = width // patch_w

        # Fixed sinusoidal encoding; a plain tensor, so it is not part of the state dict.
        self.pos_embed = build_pos_embed(
            self.model_config.pos_embed_mode,
            embed_dim, self.num_t_patches, self.num_h_patches, self.num_w_patches,
        )

        # (t, h, w) grid index per token, needed to build the attention bias for any visible subset.
        self.register_buffer(
            "grid_pos",
            build_grid_positions(self.num_t_patches, self.num_h_patches, self.num_w_patches),
            persistent=False,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values, model_config=self.model_config)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.apply(_init_linear_and_norm)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the visible tubelets.

        Args:
            x: CSI tensor ``[B, C, T, H, W]``.
            mask: boolean ``[B, N]``; ``True`` marks masked tokens. Every sample of a
                batch must have the same number of visible tokens.

        Returns:
            Encoded visible tokens ``[B, N_visible, D]`` in token order.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        batch_size, _, channels = x.shape
        x_vis = x[~mask].reshape(batch_size, -1, channels)

        cfg = self.model_config
        enc_factored = cfg.attn_mode == "factored" and cfg.factored_scope in ("both", "encoder")
        need_pos = cfg.use_rope or enc_factored
        rope_pos, biases, cycle = None, None, None
        if need_pos:
            grid = self.grid_pos.to(x.device).unsqueeze(0).expand(batch_size, -1, -1)
            token_positions = grid[~mask].reshape(batch_size, -1, 3)
            if cfg.use_rope:
                rope_pos = token_positions
            if enc_factored:
                biases, cycle = build_factored_biases(token_positions, cfg)

        for i, blk in enumerate(self.blocks):
            # Alternate the attention scope (temporal, spatial, temporal, ...) across layers.
            attention_bias = biases[cycle[i % len(cycle)]] if enc_factored else None
            x_vis = blk(x_vis, rope_pos, attention_bias)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        return self.forward_features(x, mask)


class TempoWiMAEDecoder(nn.Module):
    """Decoder that reconstructs the per-token patch values from visible and mask tokens."""

    def __init__(self, patch_size=(4, 4), num_classes=128, embed_dim=64, depth=4, num_heads=4,
                 mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 tubelet_size=4, in_chans=2, model_config=None):
        super().__init__()
        self.model_config = coerce_config(model_config)
        patch_h, patch_w = to_2tuple(patch_size)
        expected_num_classes = in_chans * tubelet_size * patch_h * patch_w
        if num_classes != expected_num_classes:
            raise ValueError(
                f"decoder output dimension must be {expected_num_classes} for patch_size={patch_size}, "
                f"tubelet_size={tubelet_size}, in_chans={in_chans}; got {num_classes}."
            )
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = (patch_h, patch_w)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values, model_config=self.model_config)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(_init_linear_and_norm)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x: torch.Tensor, return_token_num: int, token_positions=None) -> torch.Tensor:
        """Run the decoder blocks.

        Args:
            x: ``[B, N, D_dec]`` decoder tokens ordered ``[visible ; masked]``.
            return_token_num: number of trailing tokens to return (0 returns all).
            token_positions: ``[B, N, 3]`` grid index in the same order, required for
                factored attention or rotary encoding.
        """
        cfg = self.model_config
        dec_factored = cfg.attn_mode == "factored" and cfg.factored_scope in ("both", "decoder")
        rope_pos, biases, cycle = None, None, None
        if token_positions is not None:
            if cfg.use_rope:
                rope_pos = token_positions
            if dec_factored:
                biases, cycle = build_factored_biases(token_positions, cfg)

        for i, blk in enumerate(self.blocks):
            attention_bias = biases[cycle[i % len(cycle)]] if dec_factored else None
            x = blk(x, rope_pos, attention_bias)

        hidden = self.norm(x)
        if return_token_num > 0:
            return hidden[:, -return_token_num:]
        return hidden

    def forward(self, x, return_token_num, token_positions=None):
        hidden = self.forward_features(x, return_token_num, token_positions)
        return self.head(hidden)


class TempoWiMAE(nn.Module):
    """Masked autoencoder: encoder over visible tubelets, decoder over the full token set.

    ``forward(x, mask)`` returns the reconstructed patch values of the masked tokens,
    ``[B, N_masked, p_t * p_h * p_w * C]``, in masked-token order.
    """

    def __init__(self,
                 img_size=(32, 32),
                 patch_size=(4, 4),
                 encoder_in_chans=2,
                 encoder_embed_dim=64,
                 encoder_depth=6,
                 encoder_num_heads=4,
                 decoder_num_classes=None,
                 decoder_embed_dim=64,
                 decoder_depth=4,
                 decoder_num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 tubelet_size=4,
                 num_frames=16,
                 model_config=None):
        super().__init__()
        self.model_config = coerce_config(model_config)
        self.model_config.validate()

        patch_h, patch_w = to_2tuple(patch_size)
        if decoder_num_classes is None:
            decoder_num_classes = encoder_in_chans * tubelet_size * patch_h * patch_w

        self.encoder = TempoWiMAEEncoder(
            img_size=img_size, patch_size=(patch_h, patch_w), in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
            tubelet_size=tubelet_size, num_frames=num_frames, model_config=self.model_config,
        )

        self.decoder = TempoWiMAEDecoder(
            patch_size=(patch_h, patch_w), num_classes=decoder_num_classes, embed_dim=decoder_embed_dim,
            depth=decoder_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size,
            in_chans=encoder_in_chans, model_config=self.model_config,
        )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = build_pos_embed(
            self.model_config.pos_embed_mode,
            decoder_embed_dim,
            self.encoder.num_t_patches,
            self.encoder.num_h_patches,
            self.encoder.num_w_patches,
        )

        trunc_normal_(self.mask_token, std=.02)
        self.apply(_init_linear_and_norm)

    def get_num_layers(self):
        return self.encoder.get_num_layers() + self.decoder.get_num_layers()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_vis = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)
        batch_size, _, channels = x_vis.shape
        expand_pos_embed = self.pos_embed.expand(batch_size, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(batch_size, -1, channels)
        pos_emd_mask = expand_pos_embed[mask].reshape(batch_size, -1, channels)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        # Decoder token order is [visible ; masked]; the grid positions follow the same order.
        dec_token_positions = None
        if self.model_config.use_rope or self.model_config.attn_mode == "factored":
            grid = self.encoder.grid_pos.to(x.device).unsqueeze(0).expand(batch_size, -1, -1)
            dec_token_positions = torch.cat(
                [grid[~mask].reshape(batch_size, -1, 3), grid[mask].reshape(batch_size, -1, 3)], dim=1
            )

        x = self.decoder(x_full, pos_emd_mask.shape[1], dec_token_positions)
        return x


# Size ladder. All sizes use 6 encoder blocks and 4 decoder blocks; only ``nano`` uses
# mlp_ratio 2. Parameter counts are for the paper geometry (16 x 32 x 32, patch 4x4, tubelet 4).
MODEL_SIZES: Dict[str, Dict[str, int]] = {
    "nano":   dict(encoder_embed_dim=64,  encoder_depth=6, encoder_num_heads=4,   # paper: 0.210M encoder / 0.356M total
                   decoder_embed_dim=64,  decoder_depth=4, decoder_num_heads=4, mlp_ratio=2),
    "tiny":   dict(encoder_embed_dim=96,  encoder_depth=6, encoder_num_heads=4,   # 0.687M encoder / 0.743M total
                   decoder_embed_dim=32,  decoder_depth=4, decoder_num_heads=4, mlp_ratio=4),
    "little": dict(encoder_embed_dim=128, encoder_depth=6, encoder_num_heads=4,   # 1.215M encoder / 1.424M total
                   decoder_embed_dim=64,  decoder_depth=4, decoder_num_heads=4, mlp_ratio=4),
    "small":  dict(encoder_embed_dim=256, encoder_depth=6, encoder_num_heads=4,   # 4.806M encoder / 5.617M total
                   decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4, mlp_ratio=4),
    "base":   dict(encoder_embed_dim=512, encoder_depth=6, encoder_num_heads=8,   # 19.12M encoder / 22.31M total
                   decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4, mlp_ratio=4),
}


def build_model(size: str = "nano", *, input_size=(32, 32), patch_size=(4, 4), tubelet_size: int = 4,
                num_frames: int = 16, in_chans: int = 2, model_config: Optional[object] = None) -> TempoWiMAE:
    """Instantiate TempoWiMAE at one of the registered sizes.

    Args:
        size: ``"nano"`` (paper) | ``"tiny"`` | ``"little"`` | ``"small"`` | ``"base"``.
        input_size: ``(H, W)`` subcarrier-antenna grid of one snapshot.
        patch_size: ``(p_h, p_w)`` patch; ``tubelet_size`` snapshots per token.
        num_frames: snapshots per sample.
        in_chans: input channels (2 for real / imaginary).
        model_config: :class:`TempoWiMAEConfig`, mapping or ``None`` (paper defaults).
    """
    if size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size '{size}'. Choose from {list(MODEL_SIZES)}.")
    kwargs = dict(MODEL_SIZES[size])
    return TempoWiMAE(
        img_size=to_2tuple(input_size), patch_size=to_2tuple(patch_size), encoder_in_chans=int(in_chans),
        tubelet_size=int(tubelet_size), num_frames=int(num_frames), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.1, model_config=model_config, **kwargs,
    )


# ==============================================================================
# checkpoint: Checkpoint reading, writing and encoder loading.
#
# Checkpoint format (a ``torch.save`` dictionary):
#
# * ``"model"``: full state dict of :class:`TempoWiMAE` (encoder, decoder,
#   ``encoder_to_decoder`` and ``mask_token``); the sinusoidal positional encodings
#   are recomputed and are not stored.
# * ``"model_config"``: the :class:`TempoWiMAEConfig` options the weights were trained
#   with. Checkpoints produced by the original research code store the same mapping
#   under the key ``"csi2_cfg"``; both keys are read, and new checkpoints write both.
# * ``"epoch"`` and ``"config"``: training epoch and the resolved run configuration.
# ==============================================================================

CONFIG_KEY = "model_config"
LEGACY_CONFIG_KEY = "csi2_cfg"   # key used by the original research code


def read_checkpoint(path) -> Dict[str, Any]:
    """Load a checkpoint dictionary onto the CPU."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. See README.md (Pretrained Checkpoints) for download instructions."
        )
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint content in {path}: expected a dict, got {type(checkpoint)}")
    return checkpoint


def checkpoint_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Return the model state dict from a checkpoint dict or a bare state dict."""
    return checkpoint["model"] if "model" in checkpoint else checkpoint


def checkpoint_model_config(checkpoint: Dict[str, Any]) -> Optional[TempoWiMAEConfig]:
    """Return the architecture options stored in the checkpoint, or ``None`` if absent."""
    for key in (CONFIG_KEY, LEGACY_CONFIG_KEY):
        if isinstance(checkpoint.get(key), dict):
            return TempoWiMAEConfig.from_dict(checkpoint[key])
    return None


def load_compatible_state_dict(module: nn.Module, state_dict: Dict[str, torch.Tensor], label: str) -> Dict[str, Any]:
    """Load every tensor whose key and shape match ``module``; report the rest."""
    module_state = module.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in module_state:
            skipped.append((key, "missing_in_target"))
            continue
        if module_state[key].shape != value.shape:
            skipped.append((key, f"shape {tuple(value.shape)} -> {tuple(module_state[key].shape)}"))
            continue
        compatible_state[key] = value

    missing, unexpected = module.load_state_dict(compatible_state, strict=False)
    if skipped:
        preview = ", ".join(f"{name} ({reason})" for name, reason in skipped[:5])
        print(f"Loaded compatible {label} weights and skipped {len(skipped)} mismatched keys: {preview}")
    if missing:
        print(f"{label} keys left at initialization because they were absent or incompatible: {len(missing)}")
    if unexpected:
        print(f"Unexpected {label} keys ignored during loading: {len(unexpected)}")
    return {"missing": list(missing), "unexpected": list(unexpected), "skipped": skipped, "loaded": len(compatible_state)}


def load_pretrained_encoder(model: nn.Module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Load the ``encoder.*`` tensors of a full checkpoint into ``model.encoder``."""
    state_dict = checkpoint_state_dict(checkpoint)
    encoder_state = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith("encoder.")}
    if not encoder_state:
        raise ValueError("The checkpoint contains no 'encoder.*' tensors.")
    return load_compatible_state_dict(model.encoder, encoder_state, "pretrained encoder")


def load_full_model(model: nn.Module, checkpoint: Dict[str, Any], strict: bool = True):
    """Load encoder, decoder, projection and mask token (for continued pretraining or regression tests)."""
    return model.load_state_dict(checkpoint_state_dict(checkpoint), strict=strict)


def save_checkpoint(path, model: nn.Module, epoch: int, model_config: TempoWiMAEConfig,
                    run_config: Optional[Dict[str, Any]] = None) -> None:
    """Write a checkpoint in the format documented at the top of this module."""
    payload = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        CONFIG_KEY: model_config.to_dict(),
        LEGACY_CONFIG_KEY: model_config.to_dict(),
        "config": run_config,
    }
    torch.save(payload, str(path))
