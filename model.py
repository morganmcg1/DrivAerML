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
        use_wss_cp_xattn: bool = False,
        wss_cp_xattn_heads: int = 4,
        wss_cp_xattn_entropy_n_sample: int = 256,
        wss_cp_xattn_telemetry_every: int = 50,
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
        self.backbone = Transformer(
            depth=n_layers,
            hidden_dim=n_hidden,
            num_heads=n_head,
            mlp_expansion_factor=mlp_ratio,
            num_slices=slice_num,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(n_hidden, eps=1e-6)
        self.volume_out = LinearProjection(n_hidden, self.volume_output_dim)

        # H35 Wave 33 Idea C — bidirectional WSS<->surf_p decoder cross-attention.
        # Replaces the single shared surface_out projection with two independent
        # heads coupled by a single bidirectional cross-attention step.
        self.use_wss_cp_xattn = use_wss_cp_xattn
        self.wss_cp_xattn_heads = wss_cp_xattn_heads
        self.wss_cp_xattn_entropy_n_sample = wss_cp_xattn_entropy_n_sample
        self.wss_cp_xattn_telemetry_every = wss_cp_xattn_telemetry_every
        self.register_buffer(
            "_xattn_step_count", torch.zeros((), dtype=torch.long), persistent=False
        )
        if use_wss_cp_xattn:
            self.wss_norm = nn.LayerNorm(n_hidden, eps=1e-6)
            self.cp_norm = nn.LayerNorm(n_hidden, eps=1e-6)
            self.wss_to_cp_xattn = nn.MultiheadAttention(
                embed_dim=n_hidden,
                num_heads=wss_cp_xattn_heads,
                batch_first=True,
                bias=True,
            )
            self.cp_to_wss_xattn = nn.MultiheadAttention(
                embed_dim=n_hidden,
                num_heads=wss_cp_xattn_heads,
                batch_first=True,
                bias=True,
            )
            # Zero-init the output projection so the cross-attention starts as a
            # no-op residual passthrough — matches the CurvatureAttentionBias
            # convention and the LLaMA-Adapter zero-init pattern. Both heads
            # therefore equal the shared backbone output at step 0; any
            # divergence is acquired through gradient descent.
            nn.init.zeros_(self.wss_to_cp_xattn.out_proj.weight)
            nn.init.zeros_(self.wss_to_cp_xattn.out_proj.bias)
            nn.init.zeros_(self.cp_to_wss_xattn.out_proj.weight)
            nn.init.zeros_(self.cp_to_wss_xattn.out_proj.bias)
            self.wss_head = LinearProjection(n_hidden, 3)
            self.cp_head = LinearProjection(n_hidden, 1)
            self.surface_out: nn.Module | None = None
        else:
            self.surface_out = LinearProjection(n_hidden, self.surface_output_dim)

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
        hidden = self.backbone(hidden, attn_mask=attn_mask)
        hidden = _apply_token_mask(hidden, attn_mask)
        hidden_norm = _apply_token_mask(self.norm(hidden), attn_mask)

        cursor = 0
        surface_hidden = hidden_norm[:, cursor : cursor + surface_tokens]
        cursor += surface_tokens
        volume_hidden = hidden_norm[:, cursor : cursor + volume_tokens]

        xattn_telemetry: dict[str, torch.Tensor] = {}
        if surface_x is not None:
            if self.use_wss_cp_xattn:
                surface_preds, xattn_telemetry = self._wss_cp_xattn_decode(
                    surface_hidden=surface_hidden,
                    surface_mask=surface_mask,
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

        out: dict[str, torch.Tensor] = {
            "surface_preds": surface_preds,
            "volume_preds": volume_preds,
            "hidden": hidden,
            "surface_hidden": surface_hidden,
            "volume_hidden": volume_hidden,
        }
        if xattn_telemetry:
            out.update(xattn_telemetry)
        return out

    def _wss_cp_xattn_decode(
        self,
        *,
        surface_hidden: torch.Tensor,
        surface_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Bidirectional WSS<->surf_p decoder cross-attention (Wave 33 Idea C).

        ``surface_hidden`` is the post-backbone, post-norm surface stream
        ``[B, N_s, D]``. Two independent LayerNorms project it into a WSS
        stream and a cp stream; a single bidirectional cross-attention step
        couples them via residual additions; independent linear heads then
        produce ``cp_pred [B, N_s, 1]`` and ``wss_pred [B, N_s, 3]``.

        The PR's literal design uses ``nn.MultiheadAttention(num_heads=4,
        batch_first=True)`` per direction. We call them with
        ``need_weights=False`` so PyTorch can dispatch to flash-attention and
        avoid materialising an ``N_s x N_s`` attention matrix (catastrophic at
        ``N_s = 65k``). Attention entropy telemetry is computed manually on a
        downsampled subset of queries via the MHA's projection weights.

        Returns ``(surface_preds, xattn_telemetry)`` where ``surface_preds``
        preserves the model contract ``surface_preds[..., 0]=cp,
        surface_preds[..., 1:4]=wss``.
        """
        h_wss_pre = self.wss_norm(surface_hidden)
        h_cp_pre = self.cp_norm(surface_hidden)

        kp_mask = ~surface_mask.bool() if surface_mask is not None else None
        kp_mask_for_mha = None
        if kp_mask is not None and kp_mask.any():
            kp_mask_for_mha = kp_mask

        # wss queries from cp (residual)
        h_wss_aug, _ = self.wss_to_cp_xattn(
            query=h_wss_pre,
            key=h_cp_pre,
            value=h_cp_pre,
            key_padding_mask=kp_mask_for_mha,
            need_weights=False,
        )
        h_wss = h_wss_pre + h_wss_aug

        # cp queries from updated h_wss (re-normalised via wss_norm)
        h_wss_renorm = self.wss_norm(h_wss)
        h_cp_aug, _ = self.cp_to_wss_xattn(
            query=h_cp_pre,
            key=h_wss_renorm,
            value=h_wss_renorm,
            key_padding_mask=kp_mask_for_mha,
            need_weights=False,
        )
        h_cp = h_cp_pre + h_cp_aug

        wss_pred = self.wss_head(h_wss)
        cp_pred = self.cp_head(h_cp)
        surface_preds = torch.cat([cp_pred, wss_pred], dim=-1)
        surface_preds = surface_preds * surface_mask.unsqueeze(-1).to(surface_preds.dtype)

        xattn_telemetry: dict[str, torch.Tensor] = {}
        if self.training:
            with torch.no_grad():
                step = int(self._xattn_step_count.item())
                self._xattn_step_count += 1
                # head_cos_sim is cheap — log every step on rank 0
                mask_f = surface_mask.unsqueeze(-1).to(dtype=h_wss.dtype)
                h_wss_masked = h_wss * mask_f
                h_cp_masked = h_cp * mask_f
                cos_per_token = F.cosine_similarity(h_wss_masked, h_cp_masked, dim=-1)
                cos_per_token = cos_per_token * surface_mask.to(dtype=cos_per_token.dtype)
                denom = surface_mask.to(dtype=cos_per_token.dtype).sum().clamp_min(1.0)
                xattn_telemetry["xattn_head_cos_sim"] = (
                    cos_per_token.sum() / denom
                ).to(dtype=torch.float32)

                if self.wss_cp_xattn_telemetry_every > 0 and (
                    step % self.wss_cp_xattn_telemetry_every == 0
                ):
                    wss_to_cp_entropy, log_n_keys = self._xattn_entropy_subset(
                        q_pre=h_wss_pre,
                        kv_pre=h_cp_pre,
                        mha=self.wss_to_cp_xattn,
                        key_padding_mask=kp_mask,
                        surface_mask=surface_mask,
                    )
                    cp_to_wss_entropy, _ = self._xattn_entropy_subset(
                        q_pre=h_cp_pre,
                        kv_pre=h_wss_renorm,
                        mha=self.cp_to_wss_xattn,
                        key_padding_mask=kp_mask,
                        surface_mask=surface_mask,
                    )
                    xattn_telemetry["xattn_wss_to_cp_entropy"] = wss_to_cp_entropy.to(
                        dtype=torch.float32
                    )
                    xattn_telemetry["xattn_cp_to_wss_entropy"] = cp_to_wss_entropy.to(
                        dtype=torch.float32
                    )
                    xattn_telemetry["xattn_log_n_keys"] = log_n_keys.to(
                        dtype=torch.float32
                    )

        return surface_preds, xattn_telemetry

    def _xattn_entropy_subset(
        self,
        *,
        q_pre: torch.Tensor,
        kv_pre: torch.Tensor,
        mha: nn.MultiheadAttention,
        key_padding_mask: torch.Tensor | None,
        surface_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-step attention entropy on a downsampled query subset.

        Returns ``(mean_entropy, log_n_keys)`` where ``log_n_keys`` is the log
        of the per-batch-element effective key count (i.e. the uniform-attention
        entropy upper bound), so the gate ``entropy < 0.85 * log_n_keys`` can
        be evaluated downstream from the logged values.

        Avoids materialising the full ``N_s x N_s`` attention matrix by
        sampling ``n_sample`` query positions (default 256) and projecting only
        the sampled queries through ``mha.in_proj_weight``.
        """
        B, N, D = q_pre.shape
        H = mha.num_heads
        head_dim = D // H
        n_sample = min(self.wss_cp_xattn_entropy_n_sample, N)

        if n_sample <= 0 or N <= 0:
            zero = q_pre.new_zeros(())
            return zero, zero

        idx = torch.randperm(N, device=q_pre.device)[:n_sample]
        q_sub = q_pre[:, idx, :]  # [B, n_sample, D]

        in_w = mha.in_proj_weight  # [3D, D]
        in_b = mha.in_proj_bias  # [3D]
        w_q, w_k, _ = in_w.split(D, dim=0)
        b_q, b_k, _ = in_b.split(D, dim=0)

        q_proj = F.linear(q_sub, w_q, b_q).to(dtype=torch.float32)
        k_proj = F.linear(kv_pre, w_k, b_k).to(dtype=torch.float32)
        q_proj = q_proj.view(B, n_sample, H, head_dim).transpose(1, 2)
        k_proj = k_proj.view(B, N, H, head_dim).transpose(1, 2)

        scores = (q_proj @ k_proj.transpose(-1, -2)) / math.sqrt(head_dim)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        entropy_per_query = -(attn * (attn + 1e-9).log()).sum(dim=-1)
        mean_entropy = entropy_per_query.mean()

        n_keys_per_batch = surface_mask.to(dtype=torch.float32).sum(dim=-1).clamp_min(1.0)
        log_n_keys = n_keys_per_batch.log().mean()

        return mean_entropy, log_n_keys
