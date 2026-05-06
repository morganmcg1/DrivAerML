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


class StringRoPE(nn.Module):
    """STRING-style RoPE: learnable per-axis frequencies and phases for 3D RoPE.

    Applies rotary position embeddings to Q/K tensors given 3D coordinates.
    Each head dimension pair (2f, 2f+1) is rotated by an angle derived from
    the coordinate along axis d:

        theta_{d,f}(x_d) = exp(log_freq[d, f]) * 2pi * x_d + phase[d, f]

    The rotation is applied as a complex-number multiply: (q_r + i*q_i) *
    exp(i*theta) = (q_r*cos - q_i*sin) + i*(q_r*sin + q_i*cos).

    Axes are combined additively: theta = sum_d theta_{d,f}.  This allows
    smooth factorisation across stream (x), span (y) and vertical (z).

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension per head.  Must be even.
    num_axes : int
        Number of spatial axes (3 for 3-D point clouds).
    init_sigmas : list[float] | None
        Scale sigmas for log-spaced init: log_freq[d, f] = log(2*pi / sigma_f).
        Round-robin across features.  If None, defaults to sigma=1.0 for all.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_axes: int = 3,
        init_sigmas: list[float] | None = None,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE; got {head_dim}")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_axes = num_axes
        num_freq_pairs = head_dim // 2

        # log_freq[num_axes, num_freq_pairs]: learnable log-frequency
        if init_sigmas and len(init_sigmas) >= 1:
            log_freq_vals = torch.tensor(
                [math.log(2.0 * math.pi / init_sigmas[f % len(init_sigmas)]) for f in range(num_freq_pairs)],
                dtype=torch.float32,
            )
            log_freq_init = log_freq_vals.unsqueeze(0).expand(num_axes, num_freq_pairs).clone()
        else:
            log_freq_init = torch.zeros(num_axes, num_freq_pairs)
        self.log_freq = nn.Parameter(log_freq_init)

        # phase[num_axes, num_freq_pairs]: learnable phase offset
        self.phase = nn.Parameter(torch.zeros(num_axes, num_freq_pairs))

    def _compute_angles(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute rotation angles from coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``[B, N, num_axes]``.

        Returns
        -------
        angles : torch.Tensor
            Shape ``[B, N, num_freq_pairs]``.
        """
        freq = torch.exp(self.log_freq.to(dtype=coords.dtype))   # [num_axes, F]
        phase = self.phase.to(dtype=coords.dtype)                  # [num_axes, F]
        # coords: [B, N, num_axes] -> [B, N, num_axes, 1] * [num_axes, F] -> [B, N, num_axes, F]
        angles = coords.unsqueeze(-1) * freq + phase               # [B, N, num_axes, F]
        # Sum across axes (additive factorisation) -> [B, N, F]
        return angles.sum(dim=-2)

    @staticmethod
    def _rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, H, N, head_dim]``.
        angles : torch.Tensor
            Shape ``[B, N, F]`` where F = head_dim // 2.

        Returns
        -------
        rotated : torch.Tensor
            Same shape as x.
        """
        B, H, N, D = x.shape
        F = D // 2
        # angles: [B, N, F] -> [B, 1, N, F] broadcast over heads
        cos_a = torch.cos(angles).unsqueeze(1)  # [B, 1, N, F]
        sin_a = torch.sin(angles).unsqueeze(1)  # [B, 1, N, F]
        # Split x into pairs
        x_r = x[..., :F]  # real part  [B, H, N, F]
        x_i = x[..., F:]  # imaginary part  [B, H, N, F]
        out_r = x_r * cos_a - x_i * sin_a
        out_i = x_r * sin_a + x_i * cos_a
        return torch.cat([out_r, out_i], dim=-1)

    def apply_self(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE for self-attention (both Q and K use the same coords).

        Parameters
        ----------
        q, k : torch.Tensor
            Shape ``[B, H, N, head_dim]``.
        coords : torch.Tensor
            Shape ``[B, N, num_axes]``.
        """
        angles = self._compute_angles(coords)
        return self._rotate(q, angles), self._rotate(k, angles)

    def apply_cross(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_coords: torch.Tensor,
        k_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE for cross-attention (Q and K use different coords).

        Parameters
        ----------
        q : torch.Tensor
            Shape ``[B, H, Nq, head_dim]``.
        k : torch.Tensor
            Shape ``[B, H, Nk, head_dim]``.
        q_coords : torch.Tensor
            Shape ``[B, Nq, num_axes]``.
        k_coords : torch.Tensor
            Shape ``[B, Nk, num_axes]``.
        """
        q_angles = self._compute_angles(q_coords)
        k_angles = self._compute_angles(k_coords)
        return self._rotate(q, q_angles), self._rotate(k, k_angles)


class AnchorStringAttention(nn.Module):
    """Anchor-STRING attention backbone.

    Replaces the Transolver backbone with a two-stage attention mechanism:

    1. Stride-select K anchor points from the N input tokens.
    2. Run ``depth`` layers of anchor self-attention with STRING-RoPE on the
       anchor tokens (K x K attention — cheap).
    3. Run a single round of point-to-anchor cross-attention (N x K) so every
       point token attends to the refined anchor representation.

    This gives O(K^2 * D) anchor self-attention + O(N*K*D) cross-attend cost
    vs. O(N^2*D) full attention.  For N=65k, K=1024, this is ~4000x cheaper
    for the self-attention stage.

    Parameters
    ----------
    depth : int
        Number of anchor self-attention layers.
    hidden_dim : int
        Token / hidden dimension.
    num_heads : int
        Attention heads.
    mlp_expansion_factor : int | float
        MLP hidden = hidden_dim * mlp_expansion_factor.
    anchor_tokens : int
        K, number of anchors.
    init_sigmas : list[float] | None
        Init sigmas for STRING-RoPE log_freq.
    dropout : float
        Dropout rate.
    use_qk_norm : bool
        If True, apply RMSNorm to Q and K in both anchor self-attn and
        point→anchor cross-attn (matches the SOTA TransolverAttention stack).
    coord_scale : list[float] | None
        Per-axis scales used to normalise coordinates before STRING-RoPE.
        Coords are divided by these values so the resulting normalised
        coordinates fit roughly in ``[-2, 2]`` per axis. This prevents the
        STRING-RoPE rotation angles from aliasing chaotically across the
        wide DrivAerML volume domain (raw x reaches +71 m, y reaches -22 m).
        Defaults to ``[35.0, 18.0, 12.0]`` — half-extents observed in the
        diagnostic run; passing ``None`` disables normalisation.
    """

    def __init__(
        self,
        *,
        depth: int,
        hidden_dim: int,
        num_heads: int,
        mlp_expansion_factor: int | float,
        anchor_tokens: int = 1024,
        init_sigmas: list[float] | None = None,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        coord_scale: list[float] | None = None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.anchor_tokens = anchor_tokens
        self.dropout = dropout
        self.use_qk_norm = use_qk_norm

        # Per-axis coordinate scale buffer for STRING-RoPE input. Raw DrivAerML
        # volume coords reach 70 m+, which makes any sigma <= 10 m alias many
        # rotations between Q and K — diagnosed root cause of the EP1
        # divergence in PR #647 / first attempt of #742. Dividing by half-extent
        # brings coords into roughly [-2, 2] per axis where the PR-spec sigmas
        # [0.1, 0.3, 1.0, 3.0, 10.0] are appropriate.
        if coord_scale is None:
            coord_scale_vec = torch.tensor([35.0, 18.0, 12.0], dtype=torch.float32)
        else:
            if len(coord_scale) != 3:
                raise ValueError(
                    f"coord_scale must have 3 entries, got {len(coord_scale)}"
                )
            coord_scale_vec = torch.tensor(coord_scale, dtype=torch.float32)
        if (coord_scale_vec <= 0).any():
            raise ValueError(f"coord_scale entries must be positive: {coord_scale}")
        self.register_buffer("coord_scale", coord_scale_vec)

        # STRING-RoPE: shared between anchor self-attn and point→anchor cross-attn
        self.string_rope = StringRoPE(
            num_heads=num_heads,
            head_dim=self.head_dim,
            num_axes=3,
            init_sigmas=init_sigmas,
        )

        # Anchor self-attention layers (reuse TransformerBlock for pre-LN + MLP)
        # TransformerBlock uses TransolverAttention internally (slice-based),
        # but we want standard multi-head attention for anchors.
        # Use explicit QKV + sdpa layers instead.
        self.anchor_norms1 = nn.ModuleList(
            [nn.LayerNorm(hidden_dim, eps=1e-6) for _ in range(depth)]
        )
        self.anchor_qkv = nn.ModuleList(
            [nn.Linear(hidden_dim, 3 * hidden_dim, bias=False) for _ in range(depth)]
        )
        self.anchor_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)]
        )
        self.anchor_norms2 = nn.ModuleList(
            [nn.LayerNorm(hidden_dim, eps=1e-6) for _ in range(depth)]
        )
        if use_qk_norm:
            self.anchor_q_norms = nn.ModuleList(
                [nn.RMSNorm(self.head_dim, elementwise_affine=True) for _ in range(depth)]
            )
            self.anchor_k_norms = nn.ModuleList(
                [nn.RMSNorm(self.head_dim, elementwise_affine=True) for _ in range(depth)]
            )
        else:
            self.anchor_q_norms = None
            self.anchor_k_norms = None
        mlp_hidden_dim = int(math.ceil(hidden_dim * mlp_expansion_factor))
        self.anchor_mlps = nn.ModuleList(
            [UpActDownMlp(hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim) for _ in range(depth)]
        )
        self.anchor_dropout = nn.Dropout(dropout)

        # Point-to-anchor cross-attention (single layer)
        self.cross_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_kv = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.cross_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_dropout = nn.Dropout(dropout)
        if use_qk_norm:
            self.cross_q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
            self.cross_k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=True)
        else:
            self.cross_q_norm = None
            self.cross_k_norm = None

        # Diagnostic: last selected anchor coords (eval-time only). Captured by
        # forward() so collect_anchor_rope_metrics can compute coord spread.
        self._last_anchor_coords: torch.Tensor | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if name.startswith("string_rope."):
                continue
            if p.dim() >= 2:
                nn.init.trunc_normal_(p, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _select_anchors(
        self,
        hidden: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stride-select K anchor tokens, mask-aware when ``attn_mask`` is given.

        With ``attn_mask`` the K anchors are sampled at evenly-spaced positions
        among the valid (non-padding) tokens of each batch row, so anchors
        never land on a zero-coord, zero-hidden padding token. Without a mask
        we fall back to a single shared stride along ``[0, N)``.

        Parameters
        ----------
        hidden : torch.Tensor
            Shape ``[B, N, D]``.
        coords : torch.Tensor
            Shape ``[B, N, 3]``.
        attn_mask : torch.Tensor | None
            Optional ``[B, N]`` boolean / 0-1 validity mask.

        Returns
        -------
        anchor_hidden : torch.Tensor
            Shape ``[B, K, D]``.
        anchor_coords : torch.Tensor
            Shape ``[B, K, 3]``.
        anchor_idx : torch.Tensor
            Shape ``[B, K]`` (mask-aware path) or ``[K]`` (no-mask path).
        """
        B, N, D = hidden.shape
        K = min(self.anchor_tokens, N)
        if attn_mask is None:
            stride = max(1, N // K)
            anchor_idx = torch.arange(0, N, stride, device=hidden.device)[:K]
            return hidden[:, anchor_idx], coords[:, anchor_idx], anchor_idx
        # Mask-aware vectorised stride selection.
        valid_cumsum = attn_mask.long().cumsum(dim=-1)  # [B, N]
        n_valid = valid_cumsum[:, -1]  # [B]
        k_range = torch.arange(K, device=hidden.device, dtype=torch.float32).unsqueeze(0)
        n_valid_f = n_valid.float().unsqueeze(1).clamp(min=1.0)
        target = (k_range * n_valid_f / float(K)).long() + 1  # [B, K], 1-based
        target = torch.minimum(target, n_valid.long().unsqueeze(1).clamp(min=1))
        anchor_idx = torch.searchsorted(valid_cumsum, target).clamp(max=N - 1)  # [B, K]
        idx_h = anchor_idx.unsqueeze(-1).expand(B, K, D)
        anchor_hidden = torch.gather(hidden, dim=1, index=idx_h)
        idx_c = anchor_idx.unsqueeze(-1).expand(B, K, coords.shape[-1])
        anchor_coords = torch.gather(coords, dim=1, index=idx_c)
        return anchor_hidden, anchor_coords, anchor_idx

    def forward(
        self,
        hidden: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        hidden : torch.Tensor
            Shape ``[B, N, D]``.
        coords : torch.Tensor
            Shape ``[B, N, 3]``.
        attn_mask : torch.Tensor | None
            Optional token validity mask ``[B, N]``.

        Returns
        -------
        out : torch.Tensor
            Shape ``[B, N, D]``.
        """
        B, N, D = hidden.shape
        H = self.num_heads
        Hd = self.head_dim

        # Normalise raw coords by the registered per-axis scale so STRING-RoPE
        # operates on coords in roughly [-2, 2] per axis (avoids many-rotation
        # aliasing for far-field volume points).
        coord_scale = self.coord_scale.to(dtype=coords.dtype, device=coords.device)
        coords_rope = coords / coord_scale

        # --- Select anchors (mask-aware so we skip padded tokens) ---
        anchor_h, anchor_coords_raw, _ = self._select_anchors(hidden, coords, attn_mask)
        anchor_coords_rope = anchor_coords_raw / coord_scale
        K = anchor_h.shape[1]
        if not self.training:
            # Diagnostic snapshot for collect_anchor_rope_metrics — keep the raw
            # coords here so the wandb metrics keep their original physical units.
            self._last_anchor_coords = anchor_coords_raw.detach()

        # --- Anchor self-attention layers with STRING-RoPE ---
        for i in range(self.depth):
            # Pre-LN
            a_normed = self.anchor_norms1[i](anchor_h)
            # QKV projection
            qkv = self.anchor_qkv[i](a_normed)
            q, k, v = qkv.chunk(3, dim=-1)
            # Reshape to [B, H, K, Hd]
            q = q.view(B, K, H, Hd).permute(0, 2, 1, 3)
            k = k.view(B, K, H, Hd).permute(0, 2, 1, 3)
            v = v.view(B, K, H, Hd).permute(0, 2, 1, 3)
            if self.anchor_q_norms is not None:
                q = self.anchor_q_norms[i](q)
                k = self.anchor_k_norms[i](k)
            # Apply STRING-RoPE (self-attention: same coords for Q and K)
            q, k = self.string_rope.apply_self(q, k, anchor_coords_rope)
            # Scaled dot-product attention
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
            )
            # [B, H, K, Hd] -> [B, K, D]
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, K, D)
            attn_out = self.anchor_dropout(self.anchor_proj[i](attn_out))
            anchor_h = anchor_h + attn_out

            # MLP block
            anchor_h = anchor_h + self.anchor_mlps[i](self.anchor_norms2[i](anchor_h))

        # --- Point-to-anchor cross-attention ---
        # Q from points, K/V from anchors
        h_normed = self.cross_norm(hidden)
        q = self.cross_q(h_normed).view(B, N, H, Hd).permute(0, 2, 1, 3)  # [B, H, N, Hd]
        kv = self.cross_kv(anchor_h)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, K, H, Hd).permute(0, 2, 1, 3)  # [B, H, K, Hd]
        v = v.view(B, K, H, Hd).permute(0, 2, 1, 3)  # [B, H, K, Hd]
        if self.cross_q_norm is not None:
            q = self.cross_q_norm(q)
            k = self.cross_k_norm(k)
        # Apply STRING-RoPE cross-attention
        q, k = self.string_rope.apply_cross(q, k, coords_rope, anchor_coords_rope)
        # Compute cross-attention
        cross_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # [B, H, N, Hd] -> [B, N, D]
        cross_out = cross_out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        cross_out = self.cross_dropout(self.cross_proj(cross_out))
        hidden = hidden + cross_out

        if attn_mask is not None:
            hidden = _apply_token_mask(hidden, attn_mask)

        return hidden


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
        backbone_kind: str = "transolver",
        anchor_tokens: int = 1024,
        anchor_selection: str = "stride",
        anchor_string_init_sigmas: list[float] | None = None,
        anchor_string_coord_scale: list[float] | None = None,
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
        self.backbone_kind = backbone_kind
        self.anchor_tokens = anchor_tokens
        self.anchor_selection = anchor_selection
        self.anchor_string_init_sigmas = list(anchor_string_init_sigmas) if anchor_string_init_sigmas else None
        self.anchor_string_coord_scale = (
            list(anchor_string_coord_scale) if anchor_string_coord_scale else None
        )
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
        if backbone_kind == "anchor_string":
            self.backbone = AnchorStringAttention(
                depth=n_layers,
                hidden_dim=n_hidden,
                num_heads=n_head,
                mlp_expansion_factor=mlp_ratio,
                anchor_tokens=anchor_tokens,
                init_sigmas=anchor_string_init_sigmas,
                dropout=dropout,
                use_qk_norm=use_qk_norm,
                coord_scale=anchor_string_coord_scale,
            )
        else:
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
        self.surface_out = LinearProjection(n_hidden, self.surface_output_dim)
        self.volume_out = LinearProjection(n_hidden, self.volume_output_dim)

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
        if self.backbone_kind == "anchor_string":
            # Build per-token xyz coords for STRING-RoPE.  Coords are the first
            # space_dim columns of the raw input tensors (before any feature projection).
            coord_parts: list[torch.Tensor] = []
            if surface_x is not None:
                coord_parts.append(surface_x[:, :, : self.space_dim])
            if volume_x is not None:
                coord_parts.append(volume_x[:, :, : self.space_dim])
            coords = torch.cat(coord_parts, dim=1)  # [B, N_total, 3]
            hidden = self.backbone(hidden, coords, attn_mask=attn_mask)
        else:
            hidden = self.backbone(hidden, attn_mask=attn_mask)
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
