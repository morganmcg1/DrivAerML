# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Grouped surface/volume Transolver model for DrivAerML."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

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


def normalize_to_unit_cube(
    xyz: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Normalize vol coords to [-1, 1]^3 per batch using per-sample bbox.

    Padded tokens (mask=0) are excluded from the min/max computation by
    replacing them with +/-inf sentinels before the reduction.
    """
    if mask is not None:
        m = mask.unsqueeze(-1).bool().to(device=xyz.device)
        inf_pos = torch.full_like(xyz, float("inf"))
        inf_neg = torch.full_like(xyz, float("-inf"))
        xyz_for_min = torch.where(m, xyz, inf_pos)
        xyz_for_max = torch.where(m, xyz, inf_neg)
        mins = xyz_for_min.amin(dim=1, keepdim=True)
        maxs = xyz_for_max.amax(dim=1, keepdim=True)
    else:
        mins = xyz.amin(dim=1, keepdim=True)
        maxs = xyz.amax(dim=1, keepdim=True)
    span = (maxs - mins).clamp(min=1e-6)
    return 2.0 * (xyz - mins) / span - 1.0


class SpectralConv3d(nn.Module):
    """3D Fourier spectral convolution layer (FNO building block).

    Weights stored as float32 with trailing dim=2 (real, imag) for Lion
    optimizer compatibility (Lion errors on torch.cfloat parameters).
    Uses torch.view_as_complex() at forward time. norm="ortho" is used in
    both rfftn and irfftn for spectral energy stability.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, modes, 2)
        )

    def _get_complex_weights(self) -> torch.Tensor:
        return torch.view_as_complex(self.weights.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]; FFT runs in float32 regardless of autocast dtype
        B, _, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x.float(), dim=[-3, -2, -1], norm="ortho")
        w = self._get_complex_weights()  # [in_C, out_C, m, m, m] complex64
        m = self.modes
        out_ft = torch.zeros(
            B, self.out_channels, D, H, W // 2 + 1,
            dtype=x_ft.dtype, device=x_ft.device,
        )
        out_ft[:, :, :m, :m, :m] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :m, :m, :m],
            w,
        )
        x_out = torch.fft.irfftn(out_ft, s=[D, H, W], dim=[-3, -2, -1], norm="ortho")
        return x_out.to(dtype=x.dtype)


class FNOVolDecoder(nn.Module):
    """FNO-based volume pressure decoder (residual correction head).

    Scatter point features to a regular 3D grid (trilinear splatting), run
    a stack of spectral convolutions, query back at point positions
    (trilinear via F.grid_sample), and project to a scalar correction.

    Designed as an additive residual on top of the existing MLP vol head.
    The final Linear is zero-initialised so the FNO output is exactly zero
    at init -- the model starts as the plain MLP and learns the spectral
    correction.

    Grid layout: [B, C, Z, Y, X] (z-outermost) so F.grid_sample receives
    coords in the natural (x, y, z) order. See implementation notes inline.
    """

    def __init__(
        self,
        hidden_dim: int,
        grid_res: int = 32,
        modes: int = 12,
        fno_width: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.grid_res = grid_res
        self.modes = modes
        self.fno_width = fno_width
        if modes * 2 > grid_res:
            raise ValueError(
                f"FNO modes ({modes}) must be <= grid_res//2 ({grid_res // 2})"
            )

        self.lift = nn.Linear(hidden_dim, fno_width)

        self.spectral_convs = nn.ModuleList(
            [SpectralConv3d(fno_width, fno_width, modes) for _ in range(num_layers)]
        )
        self.w_convs = nn.ModuleList(
            [nn.Conv3d(fno_width, fno_width, kernel_size=1) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.GroupNorm(min(8, fno_width), fno_width) for _ in range(num_layers)]
        )

        self.proj = nn.Sequential(
            nn.Linear(fno_width, fno_width),
            nn.GELU(),
            nn.Linear(fno_width, 1),
        )
        # Identity init: zero the final Linear so the FNO correction is 0 at start.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def _scatter_to_grid(
        self, h: torch.Tensor, xyz_norm: torch.Tensor
    ) -> torch.Tensor:
        """Trilinear splat point features to a [B, C, Z, Y, X] grid.

        flat_idx = iz*R*R + iy*R + ix so the resulting tensor has Z as the
        outer-most spatial dim. When that grid is later passed to
        F.grid_sample (which treats the 5D input as [B, C, D, H, W] and
        expects the query last-dim as (x, y, z) -> (W, H, D)), the natural
        (x_norm, y_norm, z_norm) query maps correctly: x -> W -> ix axis,
        y -> H -> iy axis, z -> D -> iz axis.
        """
        B, N, C = h.shape
        R = self.grid_res
        device = h.device
        dtype = h.dtype

        # Map [-1, 1] -> [0, R-1] voxel coords
        xyz_grid = (xyz_norm + 1.0) * 0.5 * (R - 1)  # [B, N, 3]
        idx = xyz_grid.long().clamp(0, R - 2)  # [B, N, 3]
        frac = xyz_grid - idx.to(dtype=xyz_grid.dtype)  # [B, N, 3]

        grid_flat = torch.zeros(B, C, R * R * R, device=device, dtype=dtype)

        for dx in range(2):
            wx = (1 - frac[..., 0]) if dx == 0 else frac[..., 0]
            for dy in range(2):
                wy = (1 - frac[..., 1]) if dy == 0 else frac[..., 1]
                for dz in range(2):
                    wz = (1 - frac[..., 2]) if dz == 0 else frac[..., 2]
                    w = (wx * wy * wz).unsqueeze(-1)  # [B, N, 1]
                    ix = (idx[..., 0] + dx).clamp(0, R - 1)
                    iy = (idx[..., 1] + dy).clamp(0, R - 1)
                    iz = (idx[..., 2] + dz).clamp(0, R - 1)
                    flat_idx = iz * R * R + iy * R + ix  # [B, N]
                    grid_flat.scatter_add_(
                        2,
                        flat_idx.unsqueeze(1).expand(B, C, N),
                        (h * w.to(dtype=dtype)).permute(0, 2, 1),
                    )

        return grid_flat.view(B, C, R, R, R)

    def _query_grid(
        self, grid: torch.Tensor, xyz_norm: torch.Tensor
    ) -> torch.Tensor:
        """Trilinear sample [B, C, Z, Y, X] grid at xyz_norm in [-1, 1]^3.

        F.grid_sample 5D last-dim is (x, y, z) -> (W, H, D); our grid has
        Z=D, Y=H, X=W, so passing xyz_norm in (x, y, z) order is correct.
        """
        query = xyz_norm[:, None, None, :, :]  # [B, 1, 1, N, 3]
        sampled = F.grid_sample(
            grid, query,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B, C, 1, 1, N]
        return sampled[:, :, 0, 0, :].permute(0, 2, 1)  # [B, N, C]

    def forward(
        self,
        h: torch.Tensor,
        vol_xyz: torch.Tensor,
        vol_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, N, hidden_dim] vol token features from the backbone.
            vol_xyz: [B, N, 3] vol coords in physical units.
            vol_mask: [B, N] mask (1=real, 0=padding) or None.
        Returns:
            correction: [B, N, 1] additive correction for the MLP head.
        """
        xyz_norm = normalize_to_unit_cube(vol_xyz, vol_mask)

        h_fno = self.lift(h)  # [B, N, fno_width]
        if vol_mask is not None:
            h_fno = h_fno * vol_mask.unsqueeze(-1).to(dtype=h_fno.dtype)

        grid = self._scatter_to_grid(h_fno, xyz_norm)  # [B, fno_width, R, R, R]

        for spec, w_conv, norm in zip(self.spectral_convs, self.w_convs, self.norms):
            grid = F.gelu(norm(spec(grid) + w_conv(grid)))

        h_out = self._query_grid(grid, xyz_norm)  # [B, N, fno_width]
        return self.proj(h_out)  # [B, N, 1]


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
        use_qk_norm: bool = False,
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
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
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
        rff_num_features: int = 0,
        rff_sigma: float = 1.0,
        rff_init_sigmas: list[float] | None = None,
        pos_encoding_mode: str = "sincos",
        use_qk_norm: bool = False,
        use_surf_to_vol_xattn: bool = False,
        use_fno_vol_decoder: bool = False,
        fno_grid_resolution: int = 32,
        fno_modes: int = 12,
        fno_width: int = 32,
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
        self.use_fno_vol_decoder = use_fno_vol_decoder
        self.fno_grid_resolution = fno_grid_resolution
        self.fno_modes = fno_modes
        self.fno_width = fno_width
        surface_extra_dim = max(0, self.surface_input_dim - space_dim)
        volume_extra_dim = max(0, self.volume_input_dim - space_dim)

        if pos_encoding_mode == "string_separable":
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
        )
        self.norm = nn.LayerNorm(n_hidden, eps=1e-6)
        # PR #958: dedicated vol_p auxiliary decoder head.
        # Surface head is a 2-layer MLP (cp, tau_x, tau_y, tau_z).
        # Volume head is a deeper 3-layer MLP for the harder vol_p task —
        # n_hidden -> n_hidden//2 -> n_hidden//4 -> volume_output_dim with SiLU.
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

        # PR #1040: FNO spectral residual decoder for volume pressure.
        # Scatter vol token features to a regular 3D grid, run spectral
        # convolutions, query back at point positions, and add to the MLP
        # vol head output. Final Linear is zero-initialised inside
        # FNOVolDecoder so the model starts identical to the plain MLP.
        if use_fno_vol_decoder:
            self.fno_vol_decoder = FNOVolDecoder(
                hidden_dim=n_hidden,
                grid_res=fno_grid_resolution,
                modes=fno_modes,
                fno_width=fno_width,
            )
        else:
            self.fno_vol_decoder = None

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
            volume_logits = self.volume_out(volume_hidden)
            if self.fno_vol_decoder is not None:
                vol_xyz = volume_x[:, :, : self.space_dim]
                fno_correction = self.fno_vol_decoder(
                    volume_hidden, vol_xyz, volume_mask
                )
                volume_logits = volume_logits + fno_correction
            volume_preds = volume_logits * volume_mask.unsqueeze(-1)
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
