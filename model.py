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


class DropPath(nn.Module):
    """Per-sample Stochastic Depth (Huang et al. 2016).

    Drops entire residual branches with probability ``drop_prob`` during
    training, with 1/keep_prob rescaling so expected magnitude is preserved.
    Identity at eval, so adds zero inference cost. Broadcasting over the
    sample dimension means each sample in a batch is independently kept/dropped.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


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


class StringSeparableEncoding(nn.Module):
    """STRING-separable positional encoding.

    Replaces the fixed isotropic Gaussian RFF (Tancik et al. 2020) with a
    learnable per-axis spectral basis.  Each spatial axis ``d`` gets its own
    ``num_features`` independent frequency/phase parameters:

        phi_d(x_d) = sin(exp(log_freq_d) * 2pi * x_d + phase_d)   [num_features]
        psi_d(x_d) = cos(exp(log_freq_d) * 2pi * x_d + phase_d)   [num_features]

    All axes are concatenated to produce a ``2 * in_dim * num_features``-dim
    output, matching the axis-separable factorisation of automotive aerodynamics
    where x (stream), y (span), and z (vertical) have distinct spectral content.

    Parameters are initialised so that ``exp(log_freq)`` starts near the fixed
    RFF sigma (i.e. ``log_freq = log(sigma)``), providing a warm-start from
    the isotropic baseline.
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
        # log_freq[d, f]: learnable log-frequency per axis per feature
        if init_sigmas is not None and len(init_sigmas) > 1:
            # Multi-sigma init (PR #488): round-robin per-feature sigma so the
            # encoding starts with broad spectral coverage across frequency
            # octaves. Each axis shares the same per-feature sigma pattern;
            # per-axis specialisation is acquired through gradient descent.
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
        # phase[d, f]: learnable phase per axis per feature
        self.phase = nn.Parameter(torch.zeros(in_dim, num_features))

    @property
    def output_dim(self) -> int:
        return 2 * self.in_dim * self.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        # freq: [in_dim, num_features]
        freq = torch.exp(self.log_freq.to(dtype=x.dtype))
        phase = self.phase.to(dtype=x.dtype)
        # proj: [..., in_dim, num_features]
        proj = 2.0 * math.pi * x.unsqueeze(-1) * freq + phase
        # sin and cos each: [..., in_dim, num_features]
        enc = torch.cat([proj.sin(), proj.cos()], dim=-1)
        # flatten last two dims: [..., 2 * in_dim * num_features]
        return enc.flatten(start_dim=-2)


class MultiSigmaStringPosEmbed(nn.Module):
    """Drop-in replacement for ContinuousSincosEmbed using multi-sigma STRING.

    Wraps StringSeparableEncoding (per-axis learnable log-frequency basis with
    multi-sigma init) plus a linear projection to ``hidden_dim`` so the
    surrounding model sees the same [B, N, hidden_dim] interface.

    Used only by ensemble_eval.py to load PR #968 / #972 checkpoints
    (model_pe=string_multisigma); not exercised by the current trainer.
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
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_slices: int,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_head = hidden_dim // num_heads
        self.num_slices = num_slices
        self.dropout = dropout
        self.use_qk_norm = use_qk_norm

        self.temperature = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.5))
        self.in_project_x = LinearProjection(hidden_dim, hidden_dim)
        self.in_project_fx = LinearProjection(hidden_dim, hidden_dim)
        self.in_project_slice = LinearProjection(self.dim_head, num_slices)
        self.qkv = LinearProjection(self.dim_head, self.dim_head * 3, bias=False)
        self.proj = LinearProjection(hidden_dim, hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.dim_head, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.dim_head, elementwise_affine=True)
        else:
            self.q_norm = None
            self.k_norm = None

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
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
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
        use_qk_norm: bool = False,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        mlp_hidden_dim = int(math.ceil(hidden_dim * mlp_expansion_factor))
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attention = TransolverAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_slices=num_slices,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = UpActDownMlp(hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim)
        self.drop_path_attn = DropPath(drop_path_prob)
        self.drop_path_mlp = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = _apply_token_mask(x, attn_mask)
        x = x + self.drop_path_attn(self.attention(self.norm1(x), attn_mask=attn_mask))
        x = _apply_token_mask(x, attn_mask)
        x = x + self.drop_path_mlp(self.mlp(self.norm2(x)))
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
        use_qk_norm: bool = False,
        drop_path_max: float = 0.0,
    ):
        super().__init__()
        if depth > 1:
            drop_path_probs = [drop_path_max * i / (depth - 1) for i in range(depth)]
        else:
            drop_path_probs = [0.0]
        self.drop_path_probs = drop_path_probs
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_expansion_factor=mlp_expansion_factor,
                    num_slices=num_slices,
                    dropout=dropout,
                    use_qk_norm=use_qk_norm,
                    drop_path_prob=drop_path_probs[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return x


class BLDerivativeHead(nn.Module):
    """H355: Boundary-layer derivative decoder for wall shear stress.

    Predicts off-wall velocity via cross-attention from ghost probe positions
    into the post-backbone volume hidden states, then derives tau_w via
    Richardson finite-difference extrapolation to the wall:

        u(eta) = a * eta + b * eta^2 + O(eta^3)
        tau_w = mu * du/dn |_{wall} = mu * a

    With two off-wall samples u1 = u(eta1), u2 = u(eta2), the linear
    coefficient is recovered exactly as:

        tau_fd = mu * (eta2^2 * u1 - eta1^2 * u2) / (eta1 * eta2 * (eta2 - eta1))

    The cross-attention output projection is zero-initialised and the
    velocity decoder's final linear is zero-initialised so the head is
    identity-zero at warm-start (tau_fd == 0 at step 0); gradients then
    awaken naturally via the auxiliary loss.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        eta_distances: list[float],
        num_volume_slices: int = 256,
        dropout: float = 0.0,
        mu: float = 1.8e-5,
    ):
        super().__init__()
        if len(eta_distances) < 2:
            raise ValueError(
                f"BLDerivativeHead requires at least 2 eta distances; got {eta_distances}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_eta = len(eta_distances)
        self.num_volume_slices = num_volume_slices
        self.register_buffer("eta_grid", torch.tensor(eta_distances, dtype=torch.float32))
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        # Slice-pool volume_hidden (B, N_v, H) -> (B, S, H) via learnable
        # queries. Without this compression the ghost xattn cost is
        # O(N_s*K * N_v) which explodes at full Phase-1 scale
        # (N_s*K=131k, N_v=65k -> ~9G attention-matrix elements per sample).
        # Pooling to S=256 slice tokens caps attention memory at O(N_s*K * S).
        self.volume_slice_queries = nn.Parameter(
            torch.randn(1, num_volume_slices, hidden_dim) * 0.02
        )
        self.volume_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        nn.init.zeros_(self.volume_pool.out_proj.weight)
        nn.init.zeros_(self.volume_pool.out_proj.bias)
        self.volume_pool_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        # Cross-attention: ghost queries attend the pooled volume slice tokens.
        self.ghost_xattn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        nn.init.zeros_(self.ghost_xattn.out_proj.weight)
        nn.init.zeros_(self.ghost_xattn.out_proj.bias)
        self.ghost_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        # Velocity decoder: hidden -> hidden//2 -> 3 (halved bottleneck to
        # keep step-time overhead bounded at full Phase-1 scale; the
        # bottleneck still has SiLU non-linearity needed to learn the
        # off-wall velocity distribution). Final linear is zero-init so
        # u_probe == 0 at warm-start, killing all gradient flow from the
        # aux loss back into the backbone until the head wakes up.
        self.velocity_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        nn.init.trunc_normal_(self.velocity_decoder[0].weight, std=0.02)
        nn.init.zeros_(self.velocity_decoder[0].bias)
        nn.init.zeros_(self.velocity_decoder[2].weight)
        nn.init.zeros_(self.velocity_decoder[2].bias)

    def forward(
        self,
        *,
        ghost_query: torch.Tensor,
        volume_hidden: torch.Tensor,
        volume_mask: torch.Tensor | None,
        surface_normals: torch.Tensor,
        surface_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute tau_fd from ghost queries attending volume hidden states.

        Args:
            ghost_query: (B, N_s, K, hidden_dim) pre-encoded ghost positions.
            volume_hidden: (B, N_v, hidden_dim) post-backbone volume tokens.
            volume_mask: (B, N_v) bool/float mask; True = real, False = pad.
            surface_normals: (B, N_s, 3) outward unit normals.
            surface_mask: (B, N_s) bool/float mask for the surface tokens.

        Returns:
            dict with key 'tau_fd' = (B, N_s, 3) wall-tangential WSS in physical
            units (Pa), and 'u_probe' = (B, N_s, K, 3) for diagnostics.
        """
        B, N_s, K, H = ghost_query.shape
        if K != self.num_eta:
            raise ValueError(f"Expected K={self.num_eta} ghost probes; got {K}")
        # Flatten ghost queries for MHA: (B, N_s*K, H)
        q = ghost_query.reshape(B, N_s * K, H)
        if volume_mask is not None:
            # MHA key_padding_mask: True = ignore. volume_mask has 1 for real.
            key_padding_mask = ~volume_mask.bool()
        else:
            key_padding_mask = None
        # Step 1: pool volume_hidden to S slice tokens with learnable queries.
        slice_q = self.volume_slice_queries.expand(B, -1, -1).to(volume_hidden.dtype)
        slice_pool_out, _ = self.volume_pool(
            query=slice_q,
            key=volume_hidden,
            value=volume_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        slice_tokens = self.volume_pool_norm(slice_q + slice_pool_out)
        # Step 2: ghost queries attend the compressed slice tokens.
        xattn_out, _ = self.ghost_xattn(
            query=q,
            key=slice_tokens,
            value=slice_tokens,
            need_weights=False,
        )
        # Residual-norm: ghost_query + xattn_out (xattn_out == 0 at init).
        ghost_feat = self.ghost_norm(q + xattn_out)
        # Predict velocity at each probe (in physical units).
        u_flat = self.velocity_decoder(ghost_feat)  # (B, N_s*K, 3)
        u_probe = u_flat.view(B, N_s, K, 3)
        # Richardson 2-point extrapolation using the first two eta values.
        eta1 = self.eta_grid[0]
        eta2 = self.eta_grid[1]
        u1 = u_probe[:, :, 0, :]
        u2 = u_probe[:, :, 1, :]
        denom = eta1 * eta2 * (eta2 - eta1)
        a = (eta2 * eta2 * u1 - eta1 * eta1 * u2) / denom
        tau_fd_vec = self.mu * a  # (B, N_s, 3) in physical units
        # Project onto wall-tangent plane: subtract (tau . n) * n.
        normal_component = (tau_fd_vec * surface_normals).sum(dim=-1, keepdim=True)
        tau_fd = tau_fd_vec - normal_component * surface_normals
        if surface_mask is not None:
            tau_fd = tau_fd * surface_mask.unsqueeze(-1).to(tau_fd.dtype)
        return {"tau_fd": tau_fd, "u_probe": u_probe}


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
        rff_init_sigmas: list[float] | None = None,
        pos_encoding_mode: str = "sincos",
        use_qk_norm: bool = False,
        use_surf_to_vol_xattn: bool = False,
        use_aux_decoder_heads: bool = True,
        drop_path_max: float = 0.0,
        bl_derivative_decoder: bool = False,
        bl_eta_distances: list[float] | None = None,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.surface_input_dim = surface_input_dim
        self.surface_output_dim = surface_output_dim
        self.volume_input_dim = volume_input_dim
        self.volume_output_dim = volume_output_dim
        self.rff_num_features = rff_num_features
        self.rff_sigma = rff_sigma
        self.rff_init_sigmas = list(rff_init_sigmas) if rff_init_sigmas else None
        self.pos_encoding_mode = pos_encoding_mode
        self.use_qk_norm = use_qk_norm
        self.use_surf_to_vol_xattn = use_surf_to_vol_xattn
        self.use_aux_decoder_heads = use_aux_decoder_heads
        self.drop_path_max = drop_path_max
        surface_extra_dim = max(0, self.surface_input_dim - space_dim)
        volume_extra_dim = max(0, self.volume_input_dim - space_dim)

        if pos_encoding_mode == "string_multisigma":
            string_sep_features = rff_num_features if rff_num_features > 0 else 16
            self.pos_embed = MultiSigmaStringPosEmbed(
                hidden_dim=n_hidden,
                input_dim=space_dim,
                num_features=string_sep_features,
                init_sigmas=self.rff_init_sigmas,
            )
            self.surface_string_sep = None
            self.volume_string_sep = None
            self.surface_rff = None
            self.volume_rff = None
            rff_out_dim = 0
        elif pos_encoding_mode == "string_separable":
            # STRING-separable: learnable per-axis log_freq + phase,
            # replaces fixed isotropic Gaussian RFF.
            # num_features defaults to rff_num_features if provided, else 32.
            string_sep_features = rff_num_features if rff_num_features > 0 else 32
            self.surface_string_sep = StringSeparableEncoding(
                in_dim=space_dim,
                num_features=string_sep_features,
                sigma=rff_sigma,
                init_sigmas=self.rff_init_sigmas,
            )
            self.volume_string_sep = StringSeparableEncoding(
                in_dim=space_dim,
                num_features=string_sep_features,
                sigma=rff_sigma,
                init_sigmas=self.rff_init_sigmas,
            )
            string_sep_out_dim = self.surface_string_sep.output_dim  # 2 * space_dim * num_features
            self.pos_embed = ContinuousSincosEmbed(hidden_dim=n_hidden, input_dim=space_dim)
            self.surface_rff = None
            self.volume_rff = None
            rff_out_dim = string_sep_out_dim
        else:
            self.surface_string_sep = None
            self.volume_string_sep = None
            self.pos_embed = ContinuousSincosEmbed(hidden_dim=n_hidden, input_dim=space_dim)
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

        self.surface_bias = MLP(input_dim=n_hidden, hidden_dim=n_hidden, output_dim=n_hidden)
        self.volume_bias = MLP(input_dim=n_hidden, hidden_dim=n_hidden, output_dim=n_hidden)

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
            use_qk_norm=use_qk_norm,
            drop_path_max=drop_path_max,
        )
        self.norm = nn.LayerNorm(n_hidden, eps=1e-6)
        if use_aux_decoder_heads:
            # PR #958: dedicated vol_p auxiliary decoder head.
            # Surface head is a 2-layer MLP (cp, tau_x, tau_y, tau_z).
            # Volume head is a deeper 3-layer MLP for the harder vol_p task —
            # n_hidden -> n_hidden//2 -> n_hidden//4 -> volume_output_dim with SiLU.
            # Default for current training; older checkpoints (PR #823 era,
            # e.g. ghh0s4ne) set this False to load LinearProjection heads.
            self.surface_out = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.SiLU(),
                nn.Linear(n_hidden, self.surface_output_dim),
            )
            self.surface_out.apply(_init_linear)
            self.volume_out = nn.Sequential(
                nn.Linear(n_hidden, n_hidden // 2),
                nn.SiLU(),
                nn.Linear(n_hidden // 2, n_hidden // 4),
                nn.SiLU(),
                nn.Linear(n_hidden // 4, self.volume_output_dim),
            )
            self.volume_out.apply(_init_linear)
        else:
            self.surface_out = LinearProjection(n_hidden, self.surface_output_dim)
            self.volume_out = LinearProjection(n_hidden, self.volume_output_dim)

        # Surface->volume cross-attention (PR #823): single MHA sublayer where
        # volume hidden states (Q) attend to surface hidden states (K/V).
        # Bridges geometry-aware surface features into the volume decoder
        # to address the OOD volume_pressure gap. Zero-init out_proj weight
        # AND bias so the sublayer is identity at init (preserves baseline
        # behavior at epoch 0). See PR #823 hypothesis.
        if use_surf_to_vol_xattn:
            self.surf_to_vol_xattn = nn.MultiheadAttention(
                embed_dim=n_hidden,
                num_heads=n_head,
                batch_first=True,
                dropout=dropout,
            )
            self.surf_to_vol_xattn_norm = nn.LayerNorm(n_hidden, eps=1e-6)
            nn.init.zeros_(self.surf_to_vol_xattn.out_proj.weight)
            nn.init.zeros_(self.surf_to_vol_xattn.out_proj.bias)
        else:
            self.surf_to_vol_xattn = None
            self.surf_to_vol_xattn_norm = None

        # H355: boundary-layer derivative decoder. Ghost probes at K wall-normal
        # offsets attend volume_hidden to produce u(probe); a Richardson 2-point
        # FD then derives tau_w = mu * du/dn. Aux head is parallel to the main
        # surface_out, so warm-start safety is achieved by zero-init out_proj of
        # ghost_xattn AND zero-init final linear of velocity_decoder.
        self.bl_derivative_decoder = bool(bl_derivative_decoder)
        if self.bl_derivative_decoder:
            etas = bl_eta_distances if bl_eta_distances else [1e-5, 1e-4]
            if len(etas) < 2:
                raise ValueError(
                    f"bl-derivative-decoder requires >=2 eta distances; got {etas}"
                )
            self.bl_eta_distances = list(etas)
            self.bl_head = BLDerivativeHead(
                hidden_dim=n_hidden,
                num_heads=n_head,
                eta_distances=self.bl_eta_distances,
                dropout=dropout,
            )
        else:
            self.bl_eta_distances = None
            self.bl_head = None

    def _encode_group(
        self,
        x: torch.Tensor,
        *,
        rff: nn.Module | None,
        string_sep: nn.Module | None,
        project_features: LinearProjection | None,
        bias: MLP,
        placeholder: torch.Tensor,
    ) -> torch.Tensor:
        pos = x[:, :, : self.space_dim]
        hidden = self.pos_embed(pos)
        feature_parts: list[torch.Tensor] = []
        if x.shape[-1] > self.space_dim:
            feature_parts.append(x[:, :, self.space_dim :])
        if string_sep is not None:
            feature_parts.append(string_sep(pos))
        elif rff is not None:
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
                    string_sep=self.surface_string_sep,
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
                    string_sep=self.volume_string_sep,
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
            self.surf_to_vol_xattn is not None
            and surface_x is not None
            and volume_x is not None
            and surface_tokens > 0
            and volume_tokens > 0
        ):
            xattn_out, _ = self.surf_to_vol_xattn(
                query=volume_hidden,
                key=surface_hidden,
                value=surface_hidden,
                need_weights=False,
            )
            xattn_out = _apply_token_mask(xattn_out, volume_mask)
            volume_hidden = self.surf_to_vol_xattn_norm(volume_hidden + xattn_out)
            volume_hidden = _apply_token_mask(volume_hidden, volume_mask)

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

        outputs: dict[str, torch.Tensor] = {
            "surface_preds": surface_preds,
            "volume_preds": volume_preds,
            "hidden": hidden,
            "surface_hidden": surface_hidden,
            "volume_hidden": volume_hidden,
        }

        if (
            self.bl_head is not None
            and surface_x is not None
            and volume_x is not None
            and surface_tokens > 0
            and volume_tokens > 0
        ):
            surface_coords = surface_x[:, :, : self.space_dim]
            surface_normals = surface_x[:, :, self.space_dim : self.space_dim + 3]
            eta_t = surface_coords.new_tensor(self.bl_eta_distances).view(
                1, 1, len(self.bl_eta_distances), 1
            )
            # ghost_coords = x_surf + eta * n_outward (probe sits in fluid).
            ghost_coords = surface_coords.unsqueeze(2) + eta_t * surface_normals.unsqueeze(2)
            B, N_s, K, _ = ghost_coords.shape
            ghost_query = self.pos_embed(ghost_coords.reshape(B, N_s * K, self.space_dim))
            ghost_query = ghost_query.view(B, N_s, K, -1)
            bl_out = self.bl_head(
                ghost_query=ghost_query,
                volume_hidden=volume_hidden,
                volume_mask=volume_mask,
                surface_normals=surface_normals,
                surface_mask=surface_mask,
            )
            outputs["tau_fd"] = bl_out["tau_fd"]
            outputs["bl_u_probe"] = bl_out["u_probe"]

        return outputs
