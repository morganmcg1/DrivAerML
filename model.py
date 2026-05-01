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


class FourierEmbed(nn.Module):
    """Fourier positional encoding with geometric frequency progression."""

    def __init__(self, hidden_dim: int, input_dim: int = 3, num_freqs: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)
        raw_dim = input_dim * num_freqs * 2
        self.proj = nn.Linear(raw_dim, hidden_dim) if raw_dim != hidden_dim else nn.Identity()
        if isinstance(self.proj, nn.Linear):
            _init_linear(self.proj)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.float()
        angles = coords.unsqueeze(-1) * self.freqs * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        emb = emb.flatten(start_dim=-2)
        return self.proj(emb)


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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
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
        return _apply_token_mask(out_x, attn_mask)


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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = _apply_token_mask(x, attn_mask)
        x = x + self.attention(self.norm1(x), attn_mask=attn_mask)
        x = _apply_token_mask(x, attn_mask)
        x = x + self.mlp(self.norm2(x))
        x = _apply_token_mask(x, attn_mask)
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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return x


class KNNLocalAttention(nn.Module):
    """k-NN local self-attention over surface tokens conditioned on 3D xyz neighbourhood.

    For each surface query point, gathers its k nearest neighbours in 3D xyz space,
    runs multi-head attention with a learned relative-position bias (relative xyz +
    query surface normal) added to keys and values, and returns the residual+LayerNorm
    output. Padded tokens are excluded from the neighbour set via a large-distance mask.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        k: int = 16,
        knn_chunk_size: int = 4096,
        pos_input_dim: int = 6,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_head = hidden_dim // num_heads
        self.k = k
        self.knn_chunk_size = knn_chunk_size
        self.pos_input_dim = pos_input_dim
        self.scale = 1.0 / math.sqrt(self.dim_head)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)

        for module in (self.q_proj, self.k_proj, self.v_proj):
            _init_linear(module)
        for module in self.pos_mlp:
            _init_linear(module)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @torch.no_grad()
    def _knn_indices(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_points, _ = coords.shape
        device = coords.device
        coords_f = coords.float()
        invalid = None
        if mask is not None:
            invalid = mask <= 0
        indices = torch.empty(batch_size, num_points, self.k, dtype=torch.long, device=device)
        chunk = max(1, min(self.knn_chunk_size, num_points))
        for start in range(0, num_points, chunk):
            end = min(start + chunk, num_points)
            q_chunk = coords_f[:, start:end, :]
            dist = torch.cdist(q_chunk, coords_f, p=2)
            if invalid is not None:
                dist = dist.masked_fill(invalid.unsqueeze(1), float("inf"))
            _, idx = torch.topk(dist, k=self.k, dim=-1, largest=False)
            indices[:, start:end] = idx
        return indices

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        normals: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = features
        batch_size, num_points, hidden_dim = features.shape

        indices = self._knn_indices(coords, mask)

        flat_idx_3 = indices.reshape(batch_size, num_points * self.k, 1).expand(-1, -1, 3)
        coords_f = coords.float()
        neighbour_coords = torch.gather(coords_f, 1, flat_idx_3).view(batch_size, num_points, self.k, 3)
        rel_pos = coords_f.unsqueeze(2) - neighbour_coords

        if normals is not None:
            normals_f = normals.float().unsqueeze(2).expand(-1, -1, self.k, -1)
            pos_input = torch.cat([rel_pos, normals_f], dim=-1)
        else:
            pos_input = rel_pos
        pos_input = pos_input.to(features.dtype)
        pos_emb = self.pos_mlp(pos_input)

        q = self.q_proj(features)
        k_proj = self.k_proj(features)
        v_proj = self.v_proj(features)

        flat_idx_d = indices.reshape(batch_size, num_points * self.k, 1).expand(-1, -1, hidden_dim)
        k_gather = torch.gather(k_proj, 1, flat_idx_d).view(batch_size, num_points, self.k, hidden_dim)
        v_gather = torch.gather(v_proj, 1, flat_idx_d).view(batch_size, num_points, self.k, hidden_dim)

        k_combined = (k_gather + pos_emb).view(
            batch_size, num_points, self.k, self.num_heads, self.dim_head
        )
        v_combined = (v_gather + pos_emb).view(
            batch_size, num_points, self.k, self.num_heads, self.dim_head
        )
        q = q.view(batch_size, num_points, self.num_heads, self.dim_head)

        attn_logits = (q.unsqueeze(2) * k_combined).sum(dim=-1) * self.scale
        if mask is not None:
            neighbour_valid = torch.gather(
                mask.to(dtype=attn_logits.dtype),
                1,
                indices.reshape(batch_size, num_points * self.k),
            ).view(batch_size, num_points, self.k)
            attn_logits = attn_logits.masked_fill(
                (neighbour_valid <= 0).unsqueeze(-1), float("-inf")
            )
        attn_weights = F.softmax(attn_logits, dim=2)
        # Padded query rows whose neighbours are all invalid produce a row of -inf
        # in attn_logits, yielding NaN after softmax. Replace with 0 — these rows are
        # masked downstream by `_apply_token_mask`, so the value does not affect outputs.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = (attn_weights.unsqueeze(-1) * v_combined).sum(dim=2)
        out = out.reshape(batch_size, num_points, hidden_dim)
        out = self.out_proj(out)
        out = _apply_token_mask(out, mask)

        h = self.norm_out(residual + out)
        h = _apply_token_mask(h, mask)
        return h


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
        fourier_pe: bool = False,
        fourier_pe_num_freqs: int = 8,
        knn_k: int = 0,
        knn_heads: int = 4,
        knn_chunk_size: int = 4096,
        knn_use_normals: bool = True,
        surface_normal_start: int = 3,
        surface_normal_dim: int = 3,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.surface_input_dim = surface_input_dim
        self.surface_output_dim = surface_output_dim
        self.volume_input_dim = volume_input_dim
        self.volume_output_dim = volume_output_dim
        surface_extra_dim = max(0, self.surface_input_dim - space_dim)
        volume_extra_dim = max(0, self.volume_input_dim - space_dim)

        if fourier_pe:
            self.pos_embed = FourierEmbed(
                hidden_dim=n_hidden, input_dim=space_dim, num_freqs=fourier_pe_num_freqs
            )
        else:
            self.pos_embed = ContinuousSincosEmbed(hidden_dim=n_hidden, input_dim=space_dim)
        self.surface_bias = MLP(input_dim=n_hidden, hidden_dim=n_hidden, output_dim=n_hidden)
        self.volume_bias = MLP(input_dim=n_hidden, hidden_dim=n_hidden, output_dim=n_hidden)
        self.project_surface_features = (
            LinearProjection(surface_extra_dim, n_hidden) if surface_extra_dim > 0 else None
        )
        self.project_volume_features = (
            LinearProjection(volume_extra_dim, n_hidden) if volume_extra_dim > 0 else None
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
        self.knn_k = knn_k
        self.knn_use_normals = knn_use_normals and surface_extra_dim >= surface_normal_dim
        self.surface_normal_start = surface_normal_start
        self.surface_normal_dim = surface_normal_dim
        if knn_k > 0:
            pos_input_dim = space_dim + (surface_normal_dim if self.knn_use_normals else 0)
            self.knn_attn = KNNLocalAttention(
                hidden_dim=n_hidden,
                num_heads=knn_heads,
                k=knn_k,
                knn_chunk_size=knn_chunk_size,
                pos_input_dim=pos_input_dim,
            )
        else:
            self.knn_attn = None
        self.surface_out = LinearProjection(n_hidden, self.surface_output_dim)
        self.volume_out = LinearProjection(n_hidden, self.volume_output_dim)

    def _encode_group(
        self,
        x: torch.Tensor,
        *,
        project_features: LinearProjection | None,
        bias: MLP,
        placeholder: torch.Tensor,
    ) -> torch.Tensor:
        pos = x[:, :, : self.space_dim]
        hidden = self.pos_embed(pos)
        if project_features is not None and x.shape[-1] > self.space_dim:
            hidden = hidden + project_features(x[:, :, self.space_dim :])
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
                    project_features=self.project_volume_features,
                    bias=self.volume_bias,
                    placeholder=self.volume_placeholder,
                )
            )
            masks.append(volume_mask)

        attn_mask = torch.cat(masks, dim=1)
        hidden = _apply_token_mask(torch.cat(tokens, dim=1), attn_mask)
        hidden = self.backbone(hidden, attn_mask=attn_mask)
        hidden = _apply_token_mask(hidden, attn_mask)
        hidden_norm = _apply_token_mask(self.norm(hidden), attn_mask)

        cursor = 0
        surface_hidden = hidden_norm[:, cursor : cursor + surface_tokens]
        cursor += surface_tokens
        volume_hidden = hidden_norm[:, cursor : cursor + volume_tokens]

        if self.knn_attn is not None and surface_x is not None:
            # NOTE: must run unconditionally on every forward (including surface_tokens==0)
            # so DDP sees consistent parameter usage across ranks; otherwise the all-reduce
            # bucket schedule diverges and NCCL deadlocks at the next allreduce. The
            # KNNLocalAttention forward handles num_points==0 by passing empty tensors
            # through every learnable parameter.
            surface_coords = surface_x[:, :, : self.space_dim]
            if self.knn_use_normals and surface_x.shape[-1] >= self.surface_normal_start + self.surface_normal_dim:
                surface_normals = surface_x[
                    :,
                    :,
                    self.surface_normal_start : self.surface_normal_start + self.surface_normal_dim,
                ]
            else:
                surface_normals = None
            surface_hidden = self.knn_attn(
                surface_hidden,
                surface_coords,
                surface_normals,
                surface_mask,
            )

        if surface_x is not None:
            surface_preds = self.surface_out(surface_hidden) * surface_mask.unsqueeze(-1)
        else:
            batch_size = volume_x.shape[0]
            surface_preds = volume_hidden.new_zeros(batch_size, 0, self.surface_output_dim)

        if volume_x is not None:
            volume_preds = self.volume_out(volume_hidden) * volume_mask.unsqueeze(-1)
        else:
            batch_size = surface_x.shape[0]
            volume_preds = surface_hidden.new_zeros(batch_size, 0, self.volume_output_dim)

        return {
            "surface_preds": surface_preds,
            "volume_preds": volume_preds,
            "hidden": hidden,
            "surface_hidden": surface_hidden,
            "volume_hidden": volume_hidden,
        }
