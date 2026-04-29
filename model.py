# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Grouped surface/volume Transolver model for DrivAerML."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import SURFACE_X_DIM, SURFACE_Y_DIM, VOLUME_X_DIM, VOLUME_Y_DIM


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _init_linear(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _apply_token_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x
    return x * mask.unsqueeze(-1).to(device=x.device, dtype=x.dtype)


class LinearProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.project = nn.Linear(input_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_linear(self.project)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


class RFFEncoding(nn.Module):
    """Gaussian random Fourier feature coordinate encoding (Tancik et al. 2020).

    Lifts ``in_dim`` raw coordinates into a ``2 * num_features`` basis via a fixed
    random Gaussian projection followed by sin/cos. ``B`` is registered as a
    buffer so it is non-trainable but follows the model across devices and is
    broadcast across DDP ranks (DDP default ``broadcast_buffers=True``).
    """

    def __init__(self, in_dim: int, num_features: int = 32, sigma: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_features = num_features
        self.sigma = sigma
        self.register_buffer("B", torch.randn(in_dim, num_features) * sigma)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x @ self.B.to(dtype=x.dtype))
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class ContinuousSincosEmbed(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, max_wavelength: int = 10_000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        padding = hidden_dim % input_dim
        dim_per_axis = (hidden_dim - padding) // input_dim
        sincos_padding = dim_per_axis % 2
        self.padding = padding + sincos_padding * input_dim
        effective_dim_per_axis = (hidden_dim - self.padding) // input_dim
        if effective_dim_per_axis <= 0:
            raise ValueError("hidden_dim must be large enough for the requested input dimension")
        arange = torch.arange(0, effective_dim_per_axis, 2, dtype=torch.float32)
        self.register_buffer("omega", 1.0 / max_wavelength ** (arange / effective_dim_per_axis))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.float()
        out = coords.unsqueeze(-1) * self.omega
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        emb = emb.flatten(start_dim=-2)
        if self.padding > 0:
            padding = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, padding], dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.net.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpActDownMlp(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_dim)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransolverAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_head = hidden_dim // num_heads
        self.num_slices = num_slices
        self.dropout = dropout

        self.temperature = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.5))
        self.in_project_x = LinearProjection(hidden_dim, hidden_dim)
        self.in_project_fx = LinearProjection(hidden_dim, hidden_dim)
        self.in_project_slice = LinearProjection(self.dim_head, num_slices)
        self.qkv = LinearProjection(self.dim_head, self.dim_head * 3, bias=False)
        self.proj = LinearProjection(hidden_dim, hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def create_slices(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = x.shape
        fx_mid = self.in_project_fx(x).view(batch_size, num_tokens, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        x_mid = self.in_project_x(x).view(batch_size, num_tokens, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        slice_logits = self.in_project_slice(x_mid) / self.temperature
        slice_weights = F.softmax(slice_logits, dim=-1)
        if attn_mask is not None:
            slice_weights = slice_weights * attn_mask[:, None, :, None].to(
                device=slice_weights.device,
                dtype=slice_weights.dtype,
            )
        slice_norm = slice_weights.sum(dim=2, keepdim=False).unsqueeze(-1)
        slice_tokens = torch.einsum("bhnc,bhns->bhsc", fx_mid, slice_weights) / (slice_norm + 1e-5)
        return slice_tokens, slice_weights

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_slice_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        slice_tokens, slice_weights = self.create_slices(x, attn_mask=attn_mask)
        qkv = self.qkv(slice_tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        out_slice = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out_x = torch.einsum("bhsc,bhns->bhnc", out_slice, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], self.hidden_dim)
        out_x = _apply_token_mask(out_x, attn_mask)
        out_x = self.proj_dropout(self.proj(out_x))
        out_x = _apply_token_mask(out_x, attn_mask)
        if return_slice_tokens:
            slice_tokens_flat = out_slice.permute(0, 2, 1, 3).contiguous().view(
                out_slice.shape[0], out_slice.shape[2], self.hidden_dim
            )
            return out_x, slice_tokens_flat
        return out_x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_expansion_factor: int | float,
        num_slices: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        mlp_hidden_dim = int(math.ceil(hidden_dim * mlp_expansion_factor))
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attention = TransolverAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_slices=num_slices,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = UpActDownMlp(hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_slice_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = _apply_token_mask(x, attn_mask)
        if return_slice_tokens:
            attn_out, slice_tokens = self.attention(
                self.norm1(x), attn_mask=attn_mask, return_slice_tokens=True
            )
            x = x + attn_out
        else:
            x = x + self.attention(self.norm1(x), attn_mask=attn_mask)
        x = _apply_token_mask(x, attn_mask)
        x = x + self.mlp(self.norm2(x))
        x = _apply_token_mask(x, attn_mask)
        if return_slice_tokens:
            return x, slice_tokens
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        depth: int,
        hidden_dim: int,
        num_heads: int,
        mlp_expansion_factor: int | float,
        num_slices: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_expansion_factor=mlp_expansion_factor,
                    num_slices=num_slices,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_slice_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        last_idx = len(self.blocks) - 1
        slice_tokens_last: torch.Tensor | None = None
        for i, block in enumerate(self.blocks):
            if return_slice_tokens and i == last_idx:
                x, slice_tokens_last = block(x, attn_mask=attn_mask, return_slice_tokens=True)
            else:
                x = block(x, attn_mask=attn_mask)
        if return_slice_tokens:
            return x, slice_tokens_last
        return x


class ANPCrossAttentionLayer(nn.Module):
    """Pre-norm cross-attention layer for the ANP-style surface decoder.

    Queries are per-point surface embeddings; keys/values are slice-token
    anchors emitted by the backbone's last Transolver block. A learnable
    scalar gate (initialised to 1.0, i.e. full pass-through) scales the
    cross-attn residual: it lets the optimiser dial the head's contribution
    up or down without the gradient-starvation pathology of zero-init or
    near-zero-init `out_proj` schemes. Symptoms of that pathology — softmax
    saturating to uniform, q/k gradients ~1e-12, attention pinned at
    log(S) — are documented in PR #35.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.kv_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            _init_linear(module)
        # Learnable scalar gate, init=1.0 (full pass-through). Trainable so
        # the model can dampen the head if it hurts; init=1.0 (vs 0 / 1e-2)
        # ensures Q/K/V receive normal gradient from step 0, which is needed
        # for attention to escape uniform under bf16.
        self.gate = nn.Parameter(torch.ones(1))

    def forward(
        self, q_in: torch.Tensor, kv_in: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q_x = self.q_norm(q_in)
        kv_x = self.kv_norm(kv_in)
        b, l, _ = q_x.shape
        s = kv_x.shape[1]
        h = self.num_heads
        d = self.head_dim
        q = self.q_proj(q_x).view(b, l, h, d).transpose(1, 2)
        k = self.k_proj(kv_x).view(b, s, h, d).transpose(1, 2)
        v = self.v_proj(kv_x).view(b, s, h, d).transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn_logits, dim=-1)
        with torch.no_grad():
            attn_d = attn.detach().float()
            log_attn = (attn_d + 1e-9).log()
            entropy_per_head = -(attn_d * log_attn).sum(dim=-1).mean(dim=(0, 2))
            mean_entropy = entropy_per_head.mean()
            head_entropy_std = entropy_per_head.std(unbiased=False) if h > 1 else torch.zeros_like(mean_entropy)
            top_k = min(4, s)
            top_mass = attn_d.topk(top_k, dim=-1).values.sum(dim=-1).mean()
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, l, self.hidden_dim)
        out = self.proj_dropout(self.out_proj(out)) * self.gate
        return out, {
            "entropy": mean_entropy,
            "head_entropy_std": head_entropy_std,
            "top4_mass": top_mass,
            "gate": self.gate.detach().squeeze(),
        }


class ANPSurfaceDecoder(nn.Module):
    """ANP-style cross-attention decoder for surface predictions.

    Stacks `num_layers` cross-attention layers (Q = per-point surface
    embeddings, K/V = backbone slice-token anchors), each with a residual
    connection. A final LayerNorm + 2-layer MLP project the residual stream
    to the per-point output.

    A learnable per-anchor positional embedding `slice_pos_embed` is added to
    the slice tokens before the first cross-attention layer. Without it,
    backbone slice tokens degenerate to ≈ `mean(fx_mid)` at init (the
    softmax slice weights are nearly uniform, so all 128 anchors carry the
    same content), which makes V uniform across k and zeroes the
    `dL/d(attn_logits)` gradient onto Q/K. Documented in PR #35.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        output_dim: int,
        num_slices: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.num_slices = num_slices
        self.layers = nn.ModuleList(
            [
                ANPCrossAttentionLayer(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.slice_pos_embed = nn.Parameter(torch.randn(1, num_slices, hidden_dim) * 0.02)
        self.norm_final = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.out.apply(_init_linear)

    def forward(
        self,
        surface_q: torch.Tensor,
        slice_kv: torch.Tensor,
        surface_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = surface_q
        slice_kv = slice_kv + self.slice_pos_embed.to(dtype=slice_kv.dtype)
        diagnostics: dict[str, torch.Tensor] = {}
        for i, layer in enumerate(self.layers):
            attn_out, diag = layer(x, slice_kv)
            x = x + attn_out
            diagnostics[f"anp_attn_entropy_layer{i}"] = diag["entropy"]
            diagnostics[f"anp_head_entropy_std_layer{i}"] = diag["head_entropy_std"]
            diagnostics[f"anp_top4_mass_layer{i}"] = diag["top4_mass"]
            diagnostics[f"anp_gate_layer{i}"] = diag["gate"]
        x = self.norm_final(x)
        out = self.out(x)
        out = out * surface_mask.unsqueeze(-1).to(dtype=out.dtype, device=out.device)
        return out, diagnostics


class SurfaceTransolver(nn.Module):
    """Grouped Transolver for surface pressure, wall shear, and volume pressure."""

    def __init__(
        self,
        *,
        space_dim: int = 3,
        surface_input_dim: int = SURFACE_X_DIM,
        surface_output_dim: int = SURFACE_Y_DIM,
        volume_input_dim: int = VOLUME_X_DIM,
        volume_output_dim: int = VOLUME_Y_DIM,
        n_layers: int = 3,
        n_hidden: int = 192,
        dropout: float = 0.0,
        n_head: int = 3,
        mlp_ratio: int = 4,
        slice_num: int = 96,
        rff_num_features: int = 0,
        rff_sigma: float = 1.0,
        surface_decoder: str = "mlp",
        surface_decoder_layers: int = 2,
        surface_decoder_heads: int = 8,
        surface_decoder_dropout: float = 0.0,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.surface_input_dim = surface_input_dim
        self.surface_output_dim = surface_output_dim
        self.volume_input_dim = volume_input_dim
        self.volume_output_dim = volume_output_dim
        self.rff_num_features = rff_num_features
        self.rff_sigma = rff_sigma
        surface_extra_dim = max(0, self.surface_input_dim - space_dim)
        volume_extra_dim = max(0, self.volume_input_dim - space_dim)

        self.pos_embed = ContinuousSincosEmbed(hidden_dim=n_hidden, input_dim=space_dim)
        self.surface_bias = MLP(input_dim=n_hidden, hidden_dim=n_hidden, output_dim=n_hidden)
        self.volume_bias = MLP(input_dim=n_hidden, hidden_dim=n_hidden, output_dim=n_hidden)

        if rff_num_features > 0:
            self.surface_rff = RFFEncoding(
                in_dim=space_dim, num_features=rff_num_features, sigma=rff_sigma
            )
            self.volume_rff = RFFEncoding(
                in_dim=space_dim, num_features=rff_num_features, sigma=rff_sigma
            )
            rff_out_dim = 2 * rff_num_features
        else:
            self.surface_rff = None
            self.volume_rff = None
            rff_out_dim = 0

        surface_proj_in = surface_extra_dim + rff_out_dim
        volume_proj_in = volume_extra_dim + rff_out_dim
        self.project_surface_features = (
            LinearProjection(surface_proj_in, n_hidden) if surface_proj_in > 0 else None
        )
        self.project_volume_features = (
            LinearProjection(volume_proj_in, n_hidden) if volume_proj_in > 0 else None
        )
        self.surface_placeholder = nn.Parameter(torch.rand(1, 1, n_hidden) / n_hidden)
        self.volume_placeholder = nn.Parameter(torch.rand(1, 1, n_hidden) / n_hidden)
        self.backbone = Transformer(
            depth=n_layers,
            hidden_dim=n_hidden,
            num_heads=n_head,
            mlp_expansion_factor=mlp_ratio,
            num_slices=slice_num,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(n_hidden, eps=1e-6)
        decoder_kind = surface_decoder.lower()
        if decoder_kind not in {"mlp", "anp"}:
            raise ValueError(
                f"surface_decoder must be 'mlp' or 'anp', got {surface_decoder!r}"
            )
        self.surface_decoder_kind = decoder_kind
        if decoder_kind == "anp":
            self.surface_out = None
            self.anp_surface_decoder = ANPSurfaceDecoder(
                hidden_dim=n_hidden,
                output_dim=self.surface_output_dim,
                num_slices=slice_num,
                num_layers=surface_decoder_layers,
                num_heads=surface_decoder_heads,
                dropout=surface_decoder_dropout,
            )
        else:
            self.surface_out = LinearProjection(n_hidden, self.surface_output_dim)
            self.anp_surface_decoder = None
        self.volume_out = LinearProjection(n_hidden, self.volume_output_dim)

    def _encode_group(
        self,
        x: torch.Tensor,
        *,
        rff: nn.Module | None,
        project_features: LinearProjection | None,
        bias: MLP,
        placeholder: torch.Tensor,
    ) -> torch.Tensor:
        pos = x[:, :, : self.space_dim]
        hidden = self.pos_embed(pos)
        feature_parts: list[torch.Tensor] = []
        if x.shape[-1] > self.space_dim:
            feature_parts.append(x[:, :, self.space_dim :])
        if rff is not None:
            feature_parts.append(rff(pos))
        if project_features is not None and feature_parts:
            features = (
                feature_parts[0] if len(feature_parts) == 1 else torch.cat(feature_parts, dim=-1)
            )
            hidden = hidden + project_features(features)
        return bias(hidden) + placeholder

    def forward(
        self,
        *,
        surface_x: torch.Tensor | None = None,
        surface_mask: torch.Tensor | None = None,
        volume_x: torch.Tensor | None = None,
        volume_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if surface_x is None and volume_x is None:
            raise ValueError("SurfaceTransolver requires surface_x or volume_x")
        if surface_x is not None and surface_mask is None:
            raise ValueError("SurfaceTransolver requires surface_mask when surface_x is provided")
        if volume_x is not None and volume_mask is None:
            raise ValueError("SurfaceTransolver requires volume_mask when volume_x is provided")

        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        surface_tokens = 0
        volume_tokens = 0

        if surface_x is not None:
            surface_tokens = surface_x.shape[1]
            tokens.append(
                self._encode_group(
                    surface_x,
                    rff=self.surface_rff,
                    project_features=self.project_surface_features,
                    bias=self.surface_bias,
                    placeholder=self.surface_placeholder,
                )
            )
            masks.append(surface_mask)

        if volume_x is not None:
            volume_tokens = volume_x.shape[1]
            tokens.append(
                self._encode_group(
                    volume_x,
                    rff=self.volume_rff,
                    project_features=self.project_volume_features,
                    bias=self.volume_bias,
                    placeholder=self.volume_placeholder,
                )
            )
            masks.append(volume_mask)

        attn_mask = torch.cat(masks, dim=1)
        hidden = _apply_token_mask(torch.cat(tokens, dim=1), attn_mask)
        use_anp = self.surface_decoder_kind == "anp" and surface_x is not None
        if use_anp:
            hidden, slice_tokens_kv = self.backbone(
                hidden, attn_mask=attn_mask, return_slice_tokens=True
            )
        else:
            hidden = self.backbone(hidden, attn_mask=attn_mask)
            slice_tokens_kv = None
        hidden = _apply_token_mask(hidden, attn_mask)
        hidden_norm = _apply_token_mask(self.norm(hidden), attn_mask)

        cursor = 0
        surface_hidden = hidden_norm[:, cursor : cursor + surface_tokens]
        cursor += surface_tokens
        volume_hidden = hidden_norm[:, cursor : cursor + volume_tokens]

        anp_diagnostics: dict[str, torch.Tensor] = {}
        if surface_x is not None:
            if use_anp:
                surface_preds, anp_diagnostics = self.anp_surface_decoder(
                    surface_hidden, slice_tokens_kv, surface_mask
                )
            else:
                surface_preds = self.surface_out(surface_hidden) * surface_mask.unsqueeze(-1)
        else:
            batch_size = volume_x.shape[0]
            surface_preds = volume_hidden.new_zeros(batch_size, 0, self.surface_output_dim)

        if volume_x is not None:
            volume_preds = self.volume_out(volume_hidden) * volume_mask.unsqueeze(-1)
        else:
            batch_size = surface_x.shape[0]
            volume_preds = surface_hidden.new_zeros(batch_size, 0, self.volume_output_dim)

        output = {
            "surface_preds": surface_preds,
            "volume_preds": volume_preds,
            "hidden": hidden,
            "surface_hidden": surface_hidden,
            "volume_hidden": volume_hidden,
        }
        for key, value in anp_diagnostics.items():
            output[f"aux/{key}"] = value
        return output
