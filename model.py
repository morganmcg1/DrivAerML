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


class NearWallCrossAttention(nn.Module):
    """Cross-attention: surface tokens query near-wall volume tokens (|SDF|<thr).

    H3 hypothesis: τ_z (vertical wall-shear) is driven by boundary-layer
    velocity gradients which are a volume quantity. By letting surface tokens
    cross-attend to volume tokens with |SDF| < ``sdf_threshold_m`` after the
    shared backbone, the surface head can incorporate near-wall flow info.

    Inputs are expected to already be normalized (in this codebase that is
    enforced by ``SurfaceTransolver.norm`` before the surface/volume split).
    Implementation uses ``F.scaled_dot_product_attention`` (matching the
    existing ``TransolverAttention`` pattern).

    The output projection uses a small (std=0.01) init rather than exact
    zero-init: a pure zero-init would zero-out gradients into Q/K/V on the
    first iter (dL/d(act) = upstream_grad @ W_out.T == 0). A small-but-non-
    zero init keeps the cross-attention near-identity at start while letting
    Q/K/V receive non-trivial gradients from step 1.

    Note on DDP: with this module enabled, the data loader produces view-
    asymmetric batches (empty surface or empty volume tensors on trailing
    views of cases where the two modalities have different point counts).
    On those steps, Q/K/V/out projections receive no gradient, so DDP must
    be constructed with ``find_unused_parameters=True`` to handle the per-
    step variation in the used-parameter set across ranks.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        sdf_threshold_m: float = 0.05,
        max_keys: int = 1024,
        out_proj_init_std: float = 0.01,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sdf_threshold_m = sdf_threshold_m
        self.max_keys = max_keys

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.trunc_normal_(self.k_proj.weight, std=0.02)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.trunc_normal_(self.v_proj.weight, std=0.02)
        nn.init.zeros_(self.v_proj.bias)
        # Small-init for out_proj keeps the cross-attention near-identity at
        # init (residual change is O(0.01) * normal_activations) while keeping
        # gradients to Q/K/V populated for DDP.
        nn.init.normal_(self.out_proj.weight, std=out_proj_init_std)
        nn.init.zeros_(self.out_proj.bias)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(
        self,
        surface_feats: torch.Tensor,
        volume_feats: torch.Tensor,
        volume_sdf: torch.Tensor,
        surface_mask: torch.Tensor | None = None,
        volume_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Fixed top-K: pick the K volume tokens with smallest |SDF| per sample.
        # This gives a fixed key-tensor shape (B, K, D) across all batches and
        # ranks — no CUDA syncs, no per-sample Python loop, no kernel
        # recompilation. Tokens beyond ``sdf_threshold_m`` are masked out via
        # an additive attention mask so they don't influence attention.
        B, n_surf, D = surface_feats.shape
        _, n_vol, _ = volume_feats.shape
        k = min(self.max_keys, n_vol)
        H = self.num_heads
        d = self.head_dim

        sdf_abs = volume_sdf.abs()
        if volume_mask is not None:
            valid_volume = volume_mask.to(dtype=torch.bool)
            sdf_abs = torch.where(
                valid_volume, sdf_abs, torch.full_like(sdf_abs, float("inf"))
            )

        topk_values, topk_indices = torch.topk(sdf_abs, k, dim=1, largest=False)
        idx_expand = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        nw_feats = torch.gather(volume_feats, dim=1, index=idx_expand)
        # True = mask out; unconditionally allow the closest key per sample to
        # guard against the (rare) case where every nearest volume point is
        # still beyond ``sdf_threshold_m`` (would otherwise softmax a fully
        # -inf row and emit NaN).
        pad_mask = topk_values >= self.sdf_threshold_m
        pad_mask[:, 0] = False

        # Build additive attention mask: 0 for valid keys, -inf for masked.
        # Shape (B, 1, 1, k) broadcasts across (H, N_surf). Construct in fp32 so
        # the -inf mask value survives bf16 autocast precision; SDPA upcasts the
        # mask internally as needed.
        attn_mask = torch.zeros(
            B, 1, 1, k, device=surface_feats.device, dtype=torch.float32
        )
        attn_mask = attn_mask.masked_fill(
            pad_mask.unsqueeze(1).unsqueeze(1), float("-inf")
        )

        q = self.q_proj(surface_feats).view(B, n_surf, H, d).transpose(1, 2)
        k_t = self.k_proj(nw_feats).view(B, k, H, d).transpose(1, 2)
        v_t = self.v_proj(nw_feats).view(B, k, H, d).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k_t, v_t, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, n_surf, D)
        out = self.out_proj(out)

        result = self.norm(surface_feats + out)
        return _apply_token_mask(result, surface_mask)


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
        use_near_wall_cross_attn: bool = False,
        near_wall_num_heads: int = 4,
        near_wall_sdf_threshold_m: float = 0.05,
        near_wall_max_keys: int = 1024,
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
        self.backbone = Transformer(
            depth=n_layers,
            hidden_dim=n_hidden,
            num_heads=n_head,
            mlp_expansion_factor=mlp_ratio,
            num_slices=slice_num,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(n_hidden, eps=1e-6)
        self.use_near_wall_cross_attn = use_near_wall_cross_attn
        self.near_wall_num_heads = near_wall_num_heads
        self.near_wall_sdf_threshold_m = near_wall_sdf_threshold_m
        self.near_wall_max_keys = near_wall_max_keys
        if use_near_wall_cross_attn:
            self.near_wall_cross_attn = NearWallCrossAttention(
                hidden_dim=n_hidden,
                num_heads=near_wall_num_heads,
                sdf_threshold_m=near_wall_sdf_threshold_m,
                max_keys=near_wall_max_keys,
            )
        else:
            self.near_wall_cross_attn = None
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

        if (
            self.near_wall_cross_attn is not None
            and surface_tokens > 0
            and volume_tokens > 0
        ):
            # volume_x channel layout is [x, y, z, sdf]; SDF is channel index 3
            volume_sdf = volume_x[:, :, 3]
            surface_hidden = self.near_wall_cross_attn(
                surface_feats=surface_hidden,
                volume_feats=volume_hidden,
                volume_sdf=volume_sdf,
                surface_mask=surface_mask,
                volume_mask=volume_mask,
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
