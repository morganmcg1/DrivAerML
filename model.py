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


class StringSeparableEncoding(nn.Module):
    """STRING-separable positional encoding with multi-sigma log-frequency init.

    Per-axis learnable spectral basis:
        phi_d(x_d) = sin(exp(log_freq_d) * 2pi * x_d + phase_d)
        psi_d(x_d) = cos(exp(log_freq_d) * 2pi * x_d + phase_d)

    With ``init_sigmas`` of length > 1, ``log_freq`` is initialised round-robin per
    feature so the encoding starts with broad spectral coverage across frequency
    octaves; per-axis specialisation is acquired through gradient descent.

    Source: copied verbatim from origin/tay@d97fb09:model.py (PR #511 reapply of
    PR #488 multi-sigma init that powered W&B run ki2q9ko9). Identical class
    body and constructor semantics; only the in-line comments were trimmed.
    """

    def __init__(
        self,
        in_dim: int,
        num_features: int = 32,
        sigma: float = 1.0,
        init_sigmas: list[float] | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_features = num_features
        if init_sigmas is not None and len(init_sigmas) > 1:
            log_sigmas = torch.tensor(
                [math.log(init_sigmas[f % len(init_sigmas)]) for f in range(num_features)],
                dtype=torch.float32,
            )
            log_freq_init = log_sigmas.unsqueeze(0).expand(in_dim, num_features).clone()
            self.log_freq = nn.Parameter(log_freq_init)
        else:
            self.log_freq = nn.Parameter(
                torch.full((in_dim, num_features), math.log(sigma))
            )
        self.phase = nn.Parameter(torch.zeros(in_dim, num_features))

    @property
    def output_dim(self) -> int:
        return 2 * self.in_dim * self.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = torch.exp(self.log_freq.to(dtype=x.dtype))
        phase = self.phase.to(dtype=x.dtype)
        proj = 2.0 * math.pi * x.unsqueeze(-1) * freq + phase
        enc = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return enc.flatten(start_dim=-2)


class MultiSigmaStringPosEmbed(nn.Module):
    """Drop-in replacement for ContinuousSincosEmbed using multi-sigma STRING.

    Wraps StringSeparableEncoding (per-axis learnable log-frequency basis with
    multi-sigma init) plus a linear projection to ``hidden_dim`` so the
    surrounding model sees the same [B, N, hidden_dim] interface.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        num_features: int = 16,
        init_sigmas: list[float] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_features = num_features
        self.init_sigmas = list(init_sigmas) if init_sigmas else None
        self.string = StringSeparableEncoding(
            in_dim=input_dim,
            num_features=num_features,
            init_sigmas=self.init_sigmas,
        )
        self.project = nn.Linear(self.string.output_dim, hidden_dim)
        nn.init.trunc_normal_(self.project.weight, std=0.02)
        nn.init.zeros_(self.project.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_f = coords.float()
        enc = self.string(coords_f)
        return self.project(enc)


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


class CurvatureAttentionBias(nn.Module):
    """Zero-init projection of curvature features to an additive token bias.

    WSS H5 (PR #1132). Surface input channels stay at 7; curvature is injected
    AFTER the input projection so the GradNorm task-share budget per token is
    unchanged. The final linear is zero-initialised, so at step 0 the bias
    contributes exactly 0 and the model is functionally identical to the
    no-curvature SOTA baseline.
    """

    def __init__(self, hidden_dim: int, curvature_dim: int = 3):
        super().__init__()
        mid = max(hidden_dim // 8, 16)
        self.net = nn.Sequential(
            nn.Linear(curvature_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, curvature: torch.Tensor) -> torch.Tensor:
        return self.net(curvature)


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


class GeometryContextBank(nn.Module):
    """Multi-scale geometry context bank for GALE-Transolver (WSS H134).

    Performs ball-query neighbourhood aggregation at multiple radii using a
    chunked pure-PyTorch implementation (no torch_cluster dependency).  For
    each radius r and each surface point, it finds the k nearest neighbours
    within radius r and computes mean, std, and max statistics over their
    positions (and optional curvature features), giving 9 features per radius.
    The resulting (6*9 = 54)-dimensional descriptor is projected to hidden_dim.

    The bank is computed ONCE per forward pass and shared across all Transformer
    blocks to avoid redundant computation.

    Args:
        hidden_dim: Output dimension (matches transformer hidden size).
        radii: Sequence of ball-query radii.
        k_neighbors: Maximum neighbours per query point per radius.
        curvature_dim: Dimension of optional curvature features (0 to disable).
        chunk_size: Number of query points processed per chunk to bound memory.
    """

    def __init__(
        self,
        hidden_dim: int,
        radii: list[float],
        k_neighbors: int = 32,
        curvature_dim: int = 0,
        chunk_size: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.radii = radii
        self.k_neighbors = k_neighbors
        self.curvature_dim = curvature_dim
        self.chunk_size = chunk_size

        # Per-point aggregation: mean + std + max of 3D positions = 9 features/radius
        # If curvature is provided, we also aggregate mean of curvature features
        geo_feat_per_radius = 9  # mean(3) + std(3) + max(3) of xyz
        curv_feat_per_radius = curvature_dim  # mean(curvature_dim) of curvature
        feat_per_radius = geo_feat_per_radius + curv_feat_per_radius
        in_dim = len(radii) * feat_per_radius
        self.feat_per_radius = feat_per_radius
        self.in_dim = in_dim

        self.project = nn.Linear(in_dim, hidden_dim)
        nn.init.trunc_normal_(self.project.weight, std=0.02)
        nn.init.zeros_(self.project.bias)

    @staticmethod
    def _chunked_knn(
        pos: torch.Tensor,
        k: int,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single kNN pass shared across all radii.

        Returns ``(knn_idx [N, k], knn_dist2 [N, k])`` — the k nearest neighbours
        per query in squared-distance order. Self-distance is masked to +inf so
        the closest neighbour is the true nearest other point. Multi-scale ball
        statistics are computed downstream by masking ``knn_dist2 > r^2``.
        """
        n = pos.shape[0]
        device = pos.device
        dtype = pos.dtype
        k_actual = min(k, n - 1)
        idx_out = torch.empty(n, k_actual, device=device, dtype=torch.long)
        dist2_out = torch.empty(n, k_actual, device=device, dtype=dtype)
        INF = torch.finfo(dtype).max * 0.5
        p_norm = (pos * pos).sum(-1)  # [N]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            c = end - start
            query = pos[start:end]
            q_norm = (query * query).sum(-1, keepdim=True)  # [C, 1]
            dist2 = (q_norm + p_norm.unsqueeze(0) - 2.0 * (query @ pos.t())).clamp(min=0.0)
            # mask self
            ar = torch.arange(c, device=device)
            dist2[ar, torch.arange(start, end, device=device)] = INF
            td, ti = dist2.topk(k_actual, dim=1, largest=False, sorted=False)
            dist2_out[start:end] = td
            idx_out[start:end] = ti
        return idx_out, dist2_out

    def forward(
        self,
        surface_pos: torch.Tensor,
        surface_curvature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-scale geometry context for all surface points.

        Args:
            surface_pos: [B, N, 3] surface point positions.
            surface_curvature: Optional [B, N, curvature_dim] curvature features.

        Returns:
            [B, N, hidden_dim] geometry context tokens (or [B, 1, hidden_dim]
            zeros when N=0, e.g. volume-only views; this lets downstream
            cross-attention always run with consistent param usage across DDP
            ranks so all_reduce on geo_cross_attn/geo_gate doesn't hang).
        """
        b_size, n, _ = surface_pos.shape
        device = surface_pos.device
        dtype = surface_pos.dtype

        # Empty-surface (volume-only) batch slots: return a single zero token
        # but still touch self.project so its weight/bias get a gradient. This
        # keeps DDP parameter usage symmetric across ranks with mixed
        # surface/volume-only samples.
        if n == 0:
            placeholder = torch.zeros(
                b_size, 1, self.in_dim, device=device, dtype=dtype
            )
            return self.project(placeholder)

        # Run aggregation in fp32 regardless of autocast — bf16's ~3-digit
        # precision is not safe for the q^2 + p^2 - 2qp cancellation in cdist.
        if surface_curvature is None and self.curvature_dim > 0:
            surface_curvature = torch.zeros(
                b_size, n, self.curvature_dim, device=device, dtype=dtype
            )

        all_outputs = []
        INF_F32 = torch.finfo(torch.float32).max * 0.5
        for b in range(b_size):
            pos_b = surface_pos[b].float()
            curv_b = surface_curvature[b].float() if surface_curvature is not None else None
            # ONE kNN pass across all radii — was previously 6x.
            knn_idx, knn_dist2 = self._chunked_knn(pos_b, self.k_neighbors, self.chunk_size)
            # Gather neighbour features once.
            nbr_pos = pos_b[knn_idx]                    # [N, k, 3]
            nbr_curv = curv_b[knn_idx] if curv_b is not None else None  # [N, k, C] or None

            per_radius = []
            for radius in self.radii:
                r2 = float(radius) * float(radius)
                in_ball = knn_dist2 <= r2              # [N, k]
                # No neighbour in ball → fall back to nearest neighbour (col 0).
                # Setting in_ball[:,0]=True here is safe for tiny radii.
                in_ball = in_ball.clone()
                in_ball[:, 0] = True
                w = in_ball.to(nbr_pos.dtype)          # [N, k]
                w_sum = w.sum(dim=1, keepdim=True).clamp_min(1.0)
                # Mean
                wpos = nbr_pos * w.unsqueeze(-1)
                mean_pos = wpos.sum(dim=1) / w_sum
                # Std (population)
                diff = (nbr_pos - mean_pos.unsqueeze(1)) * w.unsqueeze(-1)
                var = (diff * diff).sum(dim=1) / w_sum
                std_pos = (var + 1e-12).sqrt()
                # Max
                neg_inf = nbr_pos.new_full((), -INF_F32)
                masked = torch.where(w.unsqueeze(-1) > 0, nbr_pos, neg_inf)
                max_pos = masked.max(dim=1).values
                max_pos = torch.where(torch.isfinite(max_pos), max_pos, mean_pos)
                feats = [mean_pos, std_pos, max_pos]
                # Optional curvature mean
                if nbr_curv is not None:
                    wcurv = nbr_curv * w.unsqueeze(-1)
                    mean_curv = wcurv.sum(dim=1) / w_sum
                    feats.append(mean_curv)
                per_radius.append(torch.cat(feats, dim=-1))
            geo_desc_b = torch.cat(per_radius, dim=-1)   # [N, in_dim]
            geo_desc_b = torch.nan_to_num(
                geo_desc_b, nan=0.0, posinf=0.0, neginf=0.0
            )
            all_outputs.append(geo_desc_b.to(dtype))

        geo_desc = torch.stack(all_outputs, dim=0)
        return self.project(geo_desc)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_expansion_factor: int | float,
        num_slices: int,
        dropout: float = 0.0,
        use_gale: bool = False,
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
        self.use_gale = use_gale
        self.num_slices = num_slices
        if use_gale:
            self.geo_cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=4, batch_first=True, dropout=dropout
            )
            self.geo_gate = nn.Linear(2 * hidden_dim, 1)
            # Zero-init gate so at step 0 the model is identical to baseline
            nn.init.zeros_(self.geo_gate.weight)
            nn.init.constant_(self.geo_gate.bias, -10.0)  # alpha≈0 at step 0

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        geo_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = _apply_token_mask(x, attn_mask)
        H_sa = self.attention(self.norm1(x), attn_mask=attn_mask)
        x = x + H_sa
        x = _apply_token_mask(x, attn_mask)

        # GALE cross-attention: inject geometry context after physics attention
        if self.use_gale and geo_ctx is not None:
            # geo_ctx: [B, N_surf, hidden_dim]; x: [B, N_total, hidden_dim]
            # Pool geo_ctx from [B, N_surf, D] → [B, num_slices, D] to avoid OOM.
            # Full N_surf=65536 keys would require [B*4, N_total, 65536] attn matrix ~160GiB.
            # Pooling to num_slices=128 gives [B*4, N_total, 128] ≈ 336MB — feasible.
            # Use reshape+mean instead of adaptive_avg_pool1d to avoid CUDA kernel limits.
            B_g, N_g, D_g = geo_ctx.shape
            # Fix C+: guard against N_g == 0 (empty tensor) before any modulo/division.
            # Also guard N_g < num_slices (very small test batches) using actual_slices.
            if N_g > 0:
                actual_slices = min(self.num_slices, N_g)
                if actual_slices == 0 or N_g < actual_slices:
                    geo_ctx_pooled = geo_ctx
                elif N_g % actual_slices == 0:
                    group = N_g // actual_slices
                    geo_ctx_pooled = geo_ctx.reshape(B_g, actual_slices, group, D_g).mean(dim=2)
                else:
                    # Truncate to nearest multiple of actual_slices then pool.
                    trunc = (N_g // actual_slices) * actual_slices
                    group = trunc // actual_slices
                    if group == 0:
                        # N_g < actual_slices: use all tokens directly (no pooling needed)
                        geo_ctx_pooled = geo_ctx
                    else:
                        geo_ctx_pooled = geo_ctx[:, :trunc, :].reshape(B_g, actual_slices, group, D_g).mean(dim=2)
                # geo_ctx_pooled: [B, actual_slices, D]
                # Gate alpha: scalar per sample [B, 1, 1] — matches PR spec exactly.
                CA_m, _ = self.geo_cross_attn(query=x, key=geo_ctx_pooled, value=geo_ctx_pooled)
                alpha = torch.sigmoid(
                    self.geo_gate(
                        torch.cat(
                            [x.mean(dim=1, keepdim=True),
                             CA_m.mean(dim=1, keepdim=True)],
                            dim=-1,
                        )
                    )
                )  # [B, 1, 1]
                x = (1.0 - alpha) * x + alpha * CA_m
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
        use_gale: bool = False,
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
                    use_gale=use_gale,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        geo_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, geo_ctx=geo_ctx)
        return x


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
        pe_kind: str = "sincos",
        pe_num_features: int = 16,
        pe_init_sigmas: list[float] | None = None,
        use_curvature_attention_bias: bool = False,
        curvature_dim: int = 3,
        surface_out_width_factor: float = 1.0,
        use_gale_geometry_bank: bool = False,
        gale_radii: list[float] | None = None,
        gale_k_neighbors: int = 32,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.surface_input_dim = surface_input_dim
        self.surface_output_dim = surface_output_dim
        self.volume_input_dim = volume_input_dim
        self.volume_output_dim = volume_output_dim
        self.pe_kind = pe_kind
        self.pe_num_features = pe_num_features
        self.pe_init_sigmas = list(pe_init_sigmas) if pe_init_sigmas else None
        self.use_curvature_attention_bias = use_curvature_attention_bias
        self.curvature_dim = curvature_dim
        self.surface_out_width_factor = float(surface_out_width_factor)
        surface_extra_dim = max(0, self.surface_input_dim - space_dim)
        volume_extra_dim = max(0, self.volume_input_dim - space_dim)

        if pe_kind == "sincos":
            self.pos_embed: nn.Module = ContinuousSincosEmbed(
                hidden_dim=n_hidden, input_dim=space_dim
            )
        elif pe_kind == "string_multisigma":
            self.pos_embed = MultiSigmaStringPosEmbed(
                hidden_dim=n_hidden,
                input_dim=space_dim,
                num_features=pe_num_features,
                init_sigmas=self.pe_init_sigmas,
            )
        else:
            raise ValueError(
                f"Unknown pe_kind '{pe_kind}'. Supported: sincos, string_multisigma."
            )
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
        self.curvature_attn_bias: CurvatureAttentionBias | None = None
        if use_curvature_attention_bias:
            self.curvature_attn_bias = CurvatureAttentionBias(
                hidden_dim=n_hidden,
                curvature_dim=curvature_dim,
            )
        # GALE geometry context bank (H134: WSS GALE-Transolver)
        self.use_gale_geometry_bank = use_gale_geometry_bank
        self.geo_bank: GeometryContextBank | None = None
        if use_gale_geometry_bank:
            _radii = gale_radii if gale_radii is not None else [0.01, 0.05, 0.25, 1.0, 2.5, 5.0]
            # Feed curvature into the bank only when curvature attention bias is also active
            _curv_dim = curvature_dim if use_curvature_attention_bias else 0
            self.geo_bank = GeometryContextBank(
                hidden_dim=n_hidden,
                radii=_radii,
                k_neighbors=gale_k_neighbors,
                curvature_dim=_curv_dim,
            )
        self.backbone = Transformer(
            depth=n_layers,
            hidden_dim=n_hidden,
            num_heads=n_head,
            mlp_expansion_factor=mlp_ratio,
            num_slices=slice_num,
            dropout=dropout,
            use_gale=use_gale_geometry_bank,
        )
        self.norm = nn.LayerNorm(n_hidden, eps=1e-6)
        # H39 (PR #1284): widened 2-layer surface_out head (Linear-GELU-Linear)
        # with hidden width = int(n_hidden * surface_out_width_factor). At
        # factor=2.0 this matches tay H102's wider-head mechanism on top of
        # H21 base. Note: at factor=1.0 this is NOT byte-identical to the
        # pre-H39 single-Linear canonical (it adds an extra Linear+GELU);
        # this is documented in the PR and is acceptable as the H39 test
        # explicitly uses factor=2.0.
        surf_hidden_width = max(1, int(n_hidden * self.surface_out_width_factor))
        self.surface_out_hidden_width = surf_hidden_width
        self.surface_out = nn.Sequential(
            nn.Linear(n_hidden, surf_hidden_width),
            nn.GELU(),
            nn.Linear(surf_hidden_width, self.surface_output_dim),
        )
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
        surface_curvature: torch.Tensor | None = None,
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
            surface_encoded = self._encode_group(
                surface_x,
                project_features=self.project_surface_features,
                bias=self.surface_bias,
                placeholder=self.surface_placeholder,
            )
            if self.curvature_attn_bias is not None and surface_curvature is not None:
                curv = surface_curvature.to(
                    device=surface_encoded.device,
                    dtype=surface_encoded.dtype,
                )
                bias = self.curvature_attn_bias(curv)
                if surface_mask is not None:
                    bias = bias * surface_mask.unsqueeze(-1).to(dtype=bias.dtype)
                surface_encoded = surface_encoded + bias
            tokens.append(surface_encoded)
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

        # GALE geometry context bank: compute once, shared across all blocks
        geo_ctx: torch.Tensor | None = None
        if self.geo_bank is not None and surface_x is not None:
            surf_pos = surface_x[:, :, : self.space_dim]
            curv_for_bank = surface_curvature if self.use_curvature_attention_bias else None
            geo_ctx = self.geo_bank(surf_pos, curv_for_bank)

        hidden = self.backbone(hidden, attn_mask=attn_mask, geo_ctx=geo_ctx)
        hidden = _apply_token_mask(hidden, attn_mask)
        hidden_norm = _apply_token_mask(self.norm(hidden), attn_mask)

        cursor = 0
        surface_hidden = hidden_norm[:, cursor : cursor + surface_tokens]
        cursor += surface_tokens
        volume_hidden = hidden_norm[:, cursor : cursor + volume_tokens]

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
