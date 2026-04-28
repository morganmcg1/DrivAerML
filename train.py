# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Train a grouped surface/volume Transolver on DrivAerML.

Primary metric:
    val_primary/abupt_axis_mean_rel_l2_pct

Usage:
    python train.py --epochs 50 --agent <name> --wandb_name "<name>/<experiment>"
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    SURFACE_TARGET_NAMES,
    SURFACE_X_DIM,
    SURFACE_Y_DIM,
    VOLUME_TARGET_NAMES,
    VOLUME_X_DIM,
    VOLUME_Y_DIM,
    SurfaceBatch,
    load_data,
    pad_collate,
)


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
    """Stochastic depth: drop entire residual branch with probability `drop_prob`."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # Sample in fp32 so the gate is not quantized to ~32 bf16 levels.
        random_tensor = torch.rand(shape, dtype=torch.float32, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob).to(x.dtype)
        return x / keep_prob * random_tensor


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
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = UpActDownMlp(hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim)
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = _apply_token_mask(x, attn_mask)
        x = x + self.drop_path(self.attention(self.norm1(x), attn_mask=attn_mask))
        x = _apply_token_mask(x, attn_mask)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()
        if depth <= 1:
            drop_path_rates = [stochastic_depth_prob] * depth
        else:
            drop_path_rates = [
                stochastic_depth_prob * i / (depth - 1) for i in range(depth)
            ]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_expansion_factor=mlp_expansion_factor,
                    num_slices=num_slices,
                    dropout=dropout,
                    drop_path_prob=drop_path_rates[i],
                )
                for i in range(depth)
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
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.surface_input_dim = surface_input_dim
        self.surface_output_dim = surface_output_dim
        self.volume_input_dim = volume_input_dim
        self.volume_output_dim = volume_output_dim
        surface_extra_dim = max(0, self.surface_input_dim - space_dim)
        volume_extra_dim = max(0, self.volume_input_dim - space_dim)

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
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.norm = nn.LayerNorm(n_hidden, eps=1e-6)
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


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, start_step: int = 0):
        self.decay = decay
        self.start_step = start_step
        self.step_counter = 0
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step_counter += 1
        if self.step_counter < self.start_step:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad and name in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = None


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


@dataclass
class Config:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 50
    train_surface_points: int = 40_000
    eval_surface_points: int = 40_000
    train_volume_points: int = 40_000
    eval_volume_points: int = 40_000
    validation_every: int = 10
    surface_loss_weight: float = 1.0
    volume_loss_weight: float = 1.0
    manifest: str = "data/split_manifest.json"
    data_root: str = ""
    output_dir: str = "outputs/drivaerml"
    wandb_group: str = ""
    wandb_name: str = ""
    agent: str = ""
    model_layers: int = 3
    model_hidden_dim: int = 192
    model_heads: int = 3
    model_mlp_ratio: int = 4
    model_slices: int = 96
    model_dropout: float = 0.0
    stochastic_depth_prob: float = 0.0
    amp_mode: str = "bf16"
    num_workers: int = -1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_step: int = 50
    gradient_log_every: int = 1
    log_gradient_histograms: bool = True
    weight_log_every: int = 1
    log_weight_histograms: bool = False
    slope_log_fraction: float = 0.05
    kill_thresholds: str = ""
    compile_model: bool = True
    debug: bool = False


class TargetTransform:
    def __init__(
        self,
        *,
        surface_y_mean: torch.Tensor | None = None,
        surface_y_std: torch.Tensor | None = None,
        volume_y_mean: torch.Tensor | None = None,
        volume_y_std: torch.Tensor | None = None,
        y_mean: torch.Tensor | None = None,
        y_std: torch.Tensor | None = None,
    ):
        if surface_y_mean is None:
            if y_mean is None:
                raise ValueError("TargetTransform requires surface_y_mean or y_mean")
            surface_y_mean = y_mean
        if surface_y_std is None:
            if y_std is None:
                raise ValueError("TargetTransform requires surface_y_std or y_std")
            surface_y_std = y_std
        if volume_y_mean is None:
            volume_y_mean = torch.zeros(VOLUME_Y_DIM, dtype=torch.float32)
        if volume_y_std is None:
            volume_y_std = torch.ones(VOLUME_Y_DIM, dtype=torch.float32)
        self.surface_y_mean = surface_y_mean
        self.surface_y_std = surface_y_std.clamp(min=1e-6)
        self.volume_y_mean = volume_y_mean
        self.volume_y_std = volume_y_std.clamp(min=1e-6)

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        return self.apply_surface(y)

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        return self.invert_surface(y)

    def apply_surface(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.surface_y_mean.to(y.device)) / self.surface_y_std.to(y.device)

    def invert_surface(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.surface_y_std.to(y.device) + self.surface_y_mean.to(y.device)

    def apply_volume(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.volume_y_mean.to(y.device)) / self.volume_y_std.to(y.device)

    def invert_volume(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.volume_y_std.to(y.device) + self.volume_y_mean.to(y.device)


def parse_args(argv: Iterable[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="DrivAerML surface/volume trainer")
    defaults = Config()
    help_text = {
        "kill_thresholds": (
            "Optional early-stop checks. Format: "
            "'STEP:METRIC<NUMBER[,STEP:METRIC>=NUMBER...]'; commas or semicolons "
            "separate checks. STEP is a global optimizer step and METRIC must match "
            "a logged W&B key exactly, for example "
            "'500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25'."
        ),
    }
    for field in fields(Config):
        value = getattr(defaults, field.name)
        arg_name = f"--{field.name.replace('_', '-')}"
        help_value = help_text.get(field.name)
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value, help=help_value)
            parser.add_argument(f"--no-{field.name.replace('_', '-')}", action="store_false", dest=field.name)
        else:
            parser.add_argument(arg_name, type=type(value), default=value, help=help_value)
    namespace = parser.parse_args(argv)
    return Config(**vars(namespace))


def autocast_context(device: torch.device, amp_mode: str):
    if amp_mode != "bf16" or device.type != "cuda":
        return nullcontext()
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: True)
    if not supports_bf16():
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def resolve_num_workers(config: Config) -> int:
    if config.num_workers >= 0:
        return config.num_workers
    if config.debug or not torch.cuda.is_available():
        return 0
    return min(4, os.cpu_count() or 4)


def make_loaders(
    config: Config,
) -> tuple[DataLoader, dict[str, DataLoader], dict[str, DataLoader], dict[str, torch.Tensor]]:
    train_ds, val_splits, test_splits, stats = load_data(
        manifest_path=config.manifest,
        root=config.data_root or None,
        train_surface_points=config.train_surface_points,
        eval_surface_points=config.eval_surface_points,
        train_volume_points=config.train_volume_points,
        eval_volume_points=config.eval_volume_points,
        debug=config.debug,
    )
    num_workers = resolve_num_workers(config)
    loader_kwargs = {
        "collate_fn": pad_collate,
        "num_workers": num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loaders = {
        name: DataLoader(ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }
    test_loaders = {
        name: DataLoader(ds, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_splits.items()
    }
    return train_loader, val_loaders, test_loaders, stats


def build_model(config: Config) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=config.model_layers,
        n_hidden=config.model_hidden_dim,
        dropout=config.model_dropout,
        n_head=config.model_heads,
        mlp_ratio=config.model_mlp_ratio,
        slice_num=config.model_slices,
        stochastic_depth_prob=config.stochastic_depth_prob,
    )


def _metric_path(name: str) -> str:
    cleaned = name.removeprefix("_orig_mod.")
    return cleaned.replace(".", "/") if cleaned else "root"


def _empty_grad_accumulator() -> dict[str, float]:
    return {
        "sum": 0.0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "max_abs": 0.0,
        "zero_count": 0.0,
        "element_count": 0.0,
        "nonfinite_count": 0.0,
        "param_sum_sq": 0.0,
        "param_element_count": 0.0,
        "tensor_count": 0.0,
    }


def _add_tensor_stats(
    accumulator: dict[str, float],
    *,
    grad: torch.Tensor,
    param: torch.Tensor,
) -> None:
    grad_flat = grad.detach().float().reshape(-1)
    param_flat = param.detach().float().reshape(-1)
    finite_grad = grad_flat[torch.isfinite(grad_flat)]
    finite_param = param_flat[torch.isfinite(param_flat)]

    accumulator["tensor_count"] += 1.0
    accumulator["element_count"] += float(finite_grad.numel())
    accumulator["nonfinite_count"] += float(grad_flat.numel() - finite_grad.numel())
    accumulator["param_element_count"] += float(finite_param.numel())
    if finite_grad.numel() > 0:
        abs_grad = finite_grad.abs()
        accumulator["sum"] += float(finite_grad.sum().item())
        accumulator["sum_abs"] += float(abs_grad.sum().item())
        accumulator["sum_sq"] += float(finite_grad.square().sum().item())
        accumulator["max_abs"] = max(accumulator["max_abs"], float(abs_grad.max().item()))
        accumulator["zero_count"] += float((finite_grad == 0).sum().item())
    if finite_param.numel() > 0:
        accumulator["param_sum_sq"] += float(finite_param.square().sum().item())


def _finalize_grad_stats(accumulator: dict[str, float]) -> dict[str, float]:
    element_count = max(accumulator["element_count"], 1.0)
    grad_norm = math.sqrt(accumulator["sum_sq"])
    param_norm = math.sqrt(accumulator["param_sum_sq"])
    return {
        "global_norm": grad_norm,
        "mean": accumulator["sum"] / element_count,
        "mean_abs": accumulator["sum_abs"] / element_count,
        "rms": math.sqrt(accumulator["sum_sq"] / element_count),
        "max_abs": accumulator["max_abs"],
        "zero_fraction": accumulator["zero_count"] / element_count,
        "nonfinite_count": accumulator["nonfinite_count"],
        "element_count": accumulator["element_count"],
        "tensor_count": accumulator["tensor_count"],
        "param_norm": param_norm,
        "grad_to_param_norm": grad_norm / (param_norm + 1e-12),
    }


def _empty_weight_accumulator() -> dict[str, float]:
    return {
        "sum": 0.0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "max_abs": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "zero_count": 0.0,
        "element_count": 0.0,
        "nonfinite_count": 0.0,
        "tensor_count": 0.0,
    }


def _add_weight_tensor_stats(accumulator: dict[str, float], param: torch.Tensor) -> torch.Tensor:
    param_flat = param.detach().float().reshape(-1)
    finite_param = param_flat[torch.isfinite(param_flat)]

    accumulator["tensor_count"] += 1.0
    accumulator["element_count"] += float(finite_param.numel())
    accumulator["nonfinite_count"] += float(param_flat.numel() - finite_param.numel())
    if finite_param.numel() > 0:
        abs_param = finite_param.abs()
        accumulator["sum"] += float(finite_param.sum().item())
        accumulator["sum_abs"] += float(abs_param.sum().item())
        accumulator["sum_sq"] += float(finite_param.square().sum().item())
        accumulator["max_abs"] = max(accumulator["max_abs"], float(abs_param.max().item()))
        accumulator["min"] = min(accumulator["min"], float(finite_param.min().item()))
        accumulator["max"] = max(accumulator["max"], float(finite_param.max().item()))
        accumulator["zero_count"] += float((finite_param == 0).sum().item())
    return finite_param


def _finalize_weight_stats(accumulator: dict[str, float]) -> dict[str, float]:
    element_count = max(accumulator["element_count"], 1.0)
    mean = accumulator["sum"] / element_count
    mean_square = accumulator["sum_sq"] / element_count
    variance = max(0.0, mean_square - mean * mean)
    return {
        "global_norm": math.sqrt(accumulator["sum_sq"]),
        "mean": mean,
        "mean_abs": accumulator["sum_abs"] / element_count,
        "rms": math.sqrt(mean_square),
        "std": math.sqrt(variance),
        "max_abs": accumulator["max_abs"],
        "min": accumulator["min"] if math.isfinite(accumulator["min"]) else 0.0,
        "max": accumulator["max"] if math.isfinite(accumulator["max"]) else 0.0,
        "zero_fraction": accumulator["zero_count"] / element_count,
        "nonfinite_count": accumulator["nonfinite_count"],
        "element_count": accumulator["element_count"],
        "tensor_count": accumulator["tensor_count"],
    }


def _gradient_module_paths(
    *,
    base_model: nn.Module,
    modules: dict[str, nn.Module],
    module_name: str,
) -> list[tuple[str, str]]:
    if not module_name:
        return [(type(base_model).__name__, "root")]

    paths: list[tuple[str, str]] = []
    parts = module_name.split(".")
    for end in range(1, len(parts) + 1):
        path = ".".join(parts[:end])
        module = modules.get(path)
        if module is None or isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue
        paths.append((type(module).__name__, _metric_path(path)))
    return paths


def _parameter_display_type(
    *,
    base_model: nn.Module,
    modules: dict[str, nn.Module],
    module_name: str,
) -> str:
    module = modules.get(module_name)
    if module is None:
        return type(base_model).__name__
    if not isinstance(module, nn.Linear):
        return type(module).__name__

    parts = module_name.split(".")
    for end in range(len(parts) - 1, 0, -1):
        parent = modules.get(".".join(parts[:end]))
        if parent is None or isinstance(parent, (nn.Linear, nn.ModuleList, nn.Sequential)):
            continue
        return type(parent).__name__
    return type(module).__name__


def collect_gradient_metrics(
    model: nn.Module,
    *,
    log_histograms: bool,
) -> dict[str, float | wandb.Histogram]:
    """Collect high-fidelity gradient telemetry immediately after backprop."""

    base_model = getattr(model, "_orig_mod", model)
    modules = dict(base_model.named_modules())
    global_acc = _empty_grad_accumulator()
    by_module: dict[tuple[str, str], dict[str, float]] = {}
    by_type: dict[str, dict[str, float]] = {}
    metrics: dict[str, float | wandb.Histogram] = {}
    finite_grad_chunks: list[torch.Tensor] = []
    params_with_grad = 0
    params_without_grad = 0

    for raw_name, param in base_model.named_parameters():
        module_name, _, _leaf_name = raw_name.rpartition(".")
        parameter_type = _parameter_display_type(
            base_model=base_model,
            modules=modules,
            module_name=module_name,
        )
        safe_param_name = _metric_path(raw_name)
        grad = param.grad
        if grad is None:
            params_without_grad += 1
            metrics[f"train/grad_param/{parameter_type}/{safe_param_name}/has_grad"] = 0.0
            continue

        params_with_grad += 1
        module_paths = _gradient_module_paths(
            base_model=base_model,
            modules=modules,
            module_name=module_name,
        )

        _add_tensor_stats(global_acc, grad=grad, param=param)
        for ancestor_type, ancestor_path in module_paths:
            module_acc = by_module.setdefault((ancestor_type, ancestor_path), _empty_grad_accumulator())
            type_acc = by_type.setdefault(ancestor_type, _empty_grad_accumulator())
            _add_tensor_stats(module_acc, grad=grad, param=param)
            _add_tensor_stats(type_acc, grad=grad, param=param)

        param_acc = _empty_grad_accumulator()
        _add_tensor_stats(param_acc, grad=grad, param=param)
        param_stats = _finalize_grad_stats(param_acc)
        param_prefix = f"train/grad_param/{parameter_type}/{safe_param_name}"
        for key, value in param_stats.items():
            metrics[f"{param_prefix}/{key}"] = value
        metrics[f"{param_prefix}/has_grad"] = 1.0

        finite_grad = grad.detach().float().reshape(-1)
        finite_grad = finite_grad[torch.isfinite(finite_grad)]
        if finite_grad.numel() > 0:
            finite_grad_chunks.append(finite_grad.detach().cpu())
            if log_histograms:
                metrics[f"train/grad_hist_param/{parameter_type}/{safe_param_name}"] = wandb.Histogram(
                    finite_grad.detach().cpu().numpy()
                )

    global_stats = _finalize_grad_stats(global_acc)
    for key, value in global_stats.items():
        metrics[f"train/grad/{key}"] = value
    metrics["train/grad/params_with_grad"] = float(params_with_grad)
    metrics["train/grad/params_without_grad"] = float(params_without_grad)

    for (module_type, safe_module_name), accumulator in by_module.items():
        module_prefix = f"train/grad_module/{module_type}/{safe_module_name}"
        for key, value in _finalize_grad_stats(accumulator).items():
            metrics[f"{module_prefix}/{key}"] = value

    for module_type, accumulator in by_type.items():
        type_prefix = f"train/grad_type/{module_type}"
        for key, value in _finalize_grad_stats(accumulator).items():
            metrics[f"{type_prefix}/{key}"] = value

    if log_histograms and finite_grad_chunks:
        metrics["train/grad_hist/all"] = wandb.Histogram(torch.cat(finite_grad_chunks).numpy())

    return metrics


def collect_weight_metrics(
    model: nn.Module,
    *,
    log_histograms: bool,
) -> dict[str, float | wandb.Histogram]:
    """Collect parameter telemetry after the optimizer update."""

    base_model = getattr(model, "_orig_mod", model)
    modules = dict(base_model.named_modules())
    global_acc = _empty_weight_accumulator()
    by_module: dict[tuple[str, str], dict[str, float]] = {}
    by_type: dict[str, dict[str, float]] = {}
    metrics: dict[str, float | wandb.Histogram] = {}
    finite_weight_chunks: list[torch.Tensor] = []
    trainable_tensors = 0
    frozen_tensors = 0

    for raw_name, param in base_model.named_parameters():
        if param.requires_grad:
            trainable_tensors += 1
        else:
            frozen_tensors += 1

        module_name, _, _leaf_name = raw_name.rpartition(".")
        parameter_type = _parameter_display_type(
            base_model=base_model,
            modules=modules,
            module_name=module_name,
        )
        safe_param_name = _metric_path(raw_name)
        module_paths = _gradient_module_paths(
            base_model=base_model,
            modules=modules,
            module_name=module_name,
        )

        finite_param = _add_weight_tensor_stats(global_acc, param)
        for ancestor_type, ancestor_path in module_paths:
            module_acc = by_module.setdefault((ancestor_type, ancestor_path), _empty_weight_accumulator())
            type_acc = by_type.setdefault(ancestor_type, _empty_weight_accumulator())
            _add_weight_tensor_stats(module_acc, param)
            _add_weight_tensor_stats(type_acc, param)

        param_acc = _empty_weight_accumulator()
        _add_weight_tensor_stats(param_acc, param)
        param_prefix = f"train/weight_param/{parameter_type}/{safe_param_name}"
        for key, value in _finalize_weight_stats(param_acc).items():
            metrics[f"{param_prefix}/{key}"] = value
        metrics[f"{param_prefix}/requires_grad"] = float(param.requires_grad)

        if finite_param.numel() > 0:
            finite_weight_chunks.append(finite_param.detach().cpu())
            if log_histograms:
                metrics[f"train/weight_hist_param/{parameter_type}/{safe_param_name}"] = wandb.Histogram(
                    finite_param.detach().cpu().numpy()
                )

    for key, value in _finalize_weight_stats(global_acc).items():
        metrics[f"train/weight/{key}"] = value
    metrics["train/weight/trainable_tensors"] = float(trainable_tensors)
    metrics["train/weight/frozen_tensors"] = float(frozen_tensors)

    for (module_type, safe_module_name), accumulator in by_module.items():
        module_prefix = f"train/weight_module/{module_type}/{safe_module_name}"
        for key, value in _finalize_weight_stats(accumulator).items():
            metrics[f"{module_prefix}/{key}"] = value

    for module_type, accumulator in by_type.items():
        type_prefix = f"train/weight_type/{module_type}"
        for key, value in _finalize_weight_stats(accumulator).items():
            metrics[f"{type_prefix}/{key}"] = value

    if log_histograms and finite_weight_chunks:
        metrics["train/weight_hist/all"] = wandb.Histogram(torch.cat(finite_weight_chunks).numpy())

    return metrics


def _numeric_metric_items(metrics: dict[str, object]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            number = float(value.detach().cpu().item())
        else:
            continue
        if math.isfinite(number):
            numeric[key] = number
    return numeric


def slope_source_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """Keep slope curves focused on optimization and target-quality signals."""

    keywords = (
        "loss",
        "mae",
        "rel_l2",
        "global_norm",
        "grad_to_param_norm",
        "rms",
        "max_abs",
    )
    numeric = _numeric_metric_items(metrics)
    return {key: value for key, value in numeric.items() if any(word in key for word in keywords)}


class MetricSlopeTracker:
    """Logs metric slopes over fixed fractions of the estimated update budget."""

    def __init__(self, total_steps: int, fraction: float = 0.05):
        fraction = max(float(fraction), 1e-6)
        self.interval = max(1, int(math.ceil(max(total_steps, 1) * fraction)))
        self.next_step = self.interval
        self.anchors: dict[str, tuple[int, float]] = {}

    @staticmethod
    def _curve_name(metric_name: str, namespace: str) -> str:
        prefix = f"{namespace}/"
        if metric_name.startswith(prefix):
            metric_name = metric_name.removeprefix(prefix)
        return metric_name.replace("/", "_")

    def update(
        self,
        *,
        global_step: int,
        metrics: dict[str, object],
        namespace: str,
        force: bool = False,
    ) -> dict[str, float]:
        numeric = slope_source_metrics(metrics)
        for key, value in numeric.items():
            self.anchors.setdefault(key, (global_step, value))

        if not force and global_step < self.next_step:
            return {}

        slopes: dict[str, float] = {}
        for key, value in numeric.items():
            previous_step, previous_value = self.anchors[key]
            step_delta = global_step - previous_step
            if step_delta > 0:
                slope = (value - previous_value) / step_delta
                curve = self._curve_name(key, namespace)
                slopes[f"{namespace}/slope/{curve}/per_step"] = slope
                slopes[f"{namespace}/slope/{curve}/per_1k_steps"] = slope * 1000.0
            self.anchors[key] = (global_step, value)

        while not force and global_step >= self.next_step:
            self.next_step += self.interval
        return slopes


@dataclass(frozen=True)
class KillThreshold:
    step: int
    metric: str
    operator: str
    value: float

    def passes(self, observed: float) -> bool:
        if self.operator == "<":
            return observed < self.value
        if self.operator == "<=":
            return observed <= self.value
        if self.operator == ">":
            return observed > self.value
        if self.operator == ">=":
            return observed >= self.value
        raise ValueError(f"Unsupported kill-threshold operator: {self.operator}")

    def describe(self) -> str:
        return f"step>={self.step} {self.metric}{self.operator}{self.value:g}"


def parse_kill_thresholds(raw: str) -> list[KillThreshold]:
    """Parse CLI kill-threshold specs into executable checks.

    Accepted forms are:

    - `STEP:METRIC<NUMBER`
    - `STEP:METRIC<=NUMBER`
    - `STEP:METRIC>NUMBER`
    - `STEP:METRIC>=NUMBER`

    Multiple checks may be separated by commas or semicolons. `STEP` is the
    global optimizer step where the check becomes active, and `METRIC` must
    match a logged metric key exactly, for example:

    `500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25`
    """

    if not raw.strip():
        return []
    thresholds: list[KillThreshold] = []
    for chunk in re.split(r"[;,]\s*", raw.strip()):
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                "Kill threshold is missing ':' between STEP and metric condition: "
                f"{chunk!r}. Expected STEP:METRIC<NUMBER."
            )
        step_text, condition = chunk.split(":", 1)
        try:
            step = int(step_text.strip())
        except ValueError as exc:
            raise ValueError(f"Kill threshold step must be an integer in {chunk!r}") from exc
        if step < 1:
            raise ValueError(f"Kill threshold step must be >= 1: {chunk}")
        match = re.fullmatch(r"(.+?)(<=|>=|<|>)([-+0-9.eE]+)", condition.strip())
        if match is None:
            raise ValueError(
                "Kill threshold condition must look like METRIC<NUMBER, "
                f"METRIC<=NUMBER, METRIC>NUMBER, or METRIC>=NUMBER; got {chunk!r}"
            )
        metric, operator, value_text = match.groups()
        metric = metric.strip()
        if not metric:
            raise ValueError(f"Kill threshold metric is empty in {chunk!r}")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"Kill threshold value must be numeric in {chunk!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Kill threshold value must be finite in {chunk!r}")
        thresholds.append(
            KillThreshold(
                step=step,
                metric=metric,
                operator=operator,
                value=value,
            )
        )
    return thresholds


def check_kill_thresholds(
    *,
    global_step: int,
    metrics: dict[str, object],
    thresholds: list[KillThreshold],
) -> str | None:
    numeric = _numeric_metric_items(metrics)
    for threshold in thresholds:
        if global_step < threshold.step or threshold.metric not in numeric:
            continue
        observed = numeric[threshold.metric]
        if not math.isfinite(observed) or not threshold.passes(observed):
            return (
                f"kill threshold failed at step {global_step}: "
                f"{threshold.metric}={observed:.6g} did not satisfy {threshold.describe()}"
            )
    return None


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return F.mse_loss(pred[mask], target[mask])
    return pred.sum() * 0.0


def train_loss(
    model: nn.Module,
    batch: SurfaceBatch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    *,
    surface_loss_weight: float = 1.0,
    volume_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = batch.to(device)
    surface_target = transform.apply_surface(batch.surface_y)
    volume_target = transform.apply_volume(batch.volume_y)
    with autocast_context(device, amp_mode):
        out = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
        surface_loss = masked_mse(out["surface_preds"], surface_target, batch.surface_mask)
        volume_loss = masked_mse(out["volume_preds"], volume_target, batch.volume_mask)
        loss = surface_loss_weight * surface_loss + volume_loss_weight * volume_loss
    return loss, {
        "surface_loss": float(surface_loss.detach().cpu().item()),
        "volume_loss": float(volume_loss.detach().cpu().item()),
    }


def _masked_sse_count(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    if not bool(mask.any()):
        return 0.0, 0
    diff = pred[mask].float() - target[mask].float()
    return float(diff.square().sum().detach().cpu().item()), int(diff.numel())


def _accumulate_case_rel_l2(
    store: dict[str, list[float]],
    *,
    case_id: str,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if pred.numel() == 0:
        return
    target_sq = float(target.float().square().sum().detach().cpu().item())
    if target_sq <= 0.0:
        return
    error_sq = float((pred.float() - target.float()).square().sum().detach().cpu().item())
    state = store.setdefault(case_id, [0.0, 0.0])
    state[0] += error_sq
    state[1] += target_sq


def _rel_l2(store: dict[str, list[float]]) -> tuple[float, float]:
    values = [
        math.sqrt(error_sq / target_sq)
        for error_sq, target_sq in store.values()
        if target_sq > 0.0
    ]
    value = sum(values) / max(len(values), 1)
    return value, float(len(values))


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / max(len(finite), 1)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    *,
    amp_mode: str = "none",
) -> dict[str, float]:
    model.eval()
    surface_loss_sse = 0.0
    surface_loss_count = 0
    volume_loss_sse = 0.0
    volume_loss_count = 0
    abs_sums = {
        "surface_pressure": 0.0,
        "wall_shear": 0.0,
        "wall_shear_x": 0.0,
        "wall_shear_y": 0.0,
        "wall_shear_z": 0.0,
        "volume_pressure": 0.0,
    }
    abs_counts = {key: 0 for key in abs_sums}
    wall_shear_vector_abs_sum = 0.0
    wall_shear_vector_count = 0
    case_sums = {
        "surface_pressure": {},
        "wall_shear": {},
        "wall_shear_x": {},
        "wall_shear_y": {},
        "wall_shear_z": {},
        "volume_pressure": {},
    }

    for batch in loader:
        batch = batch.to(device)
        surface_target_norm = transform.apply_surface(batch.surface_y)
        volume_target_norm = transform.apply_volume(batch.volume_y)
        with autocast_context(device, amp_mode):
            out = model(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
        surface_pred_norm = out["surface_preds"].float()
        volume_pred_norm = out["volume_preds"].float()
        surface_sse, surface_count = _masked_sse_count(surface_pred_norm, surface_target_norm, batch.surface_mask)
        volume_sse, volume_count = _masked_sse_count(volume_pred_norm, volume_target_norm, batch.volume_mask)
        surface_loss_sse += surface_sse
        surface_loss_count += surface_count
        volume_loss_sse += volume_sse
        volume_loss_count += volume_count
        surface_pred = transform.invert_surface(surface_pred_norm)
        volume_pred = transform.invert_volume(volume_pred_norm)

        if bool(batch.surface_mask.any()):
            surface_abs = (surface_pred - batch.surface_y).abs()
            valid_surface_abs = surface_abs[batch.surface_mask]
            abs_sums["surface_pressure"] += float(valid_surface_abs[:, 0].sum().detach().cpu().item())
            abs_counts["surface_pressure"] += int(valid_surface_abs[:, 0].numel())
            wall_abs = valid_surface_abs[:, 1:4]
            abs_sums["wall_shear"] += float(wall_abs.sum().detach().cpu().item())
            abs_counts["wall_shear"] += int(wall_abs.numel())
            for offset, axis in enumerate(("x", "y", "z")):
                channel = wall_abs[:, offset]
                abs_sums[f"wall_shear_{axis}"] += float(channel.sum().detach().cpu().item())
                abs_counts[f"wall_shear_{axis}"] += int(channel.numel())
            wall_vector_error = torch.linalg.vector_norm(
                surface_pred[batch.surface_mask][:, 1:4] - batch.surface_y[batch.surface_mask][:, 1:4],
                dim=-1,
            )
            wall_shear_vector_abs_sum += float(wall_vector_error.sum().detach().cpu().item())
            wall_shear_vector_count += int(wall_vector_error.numel())

        if bool(batch.volume_mask.any()):
            volume_abs = (volume_pred - batch.volume_y).abs()[batch.volume_mask]
            abs_sums["volume_pressure"] += float(volume_abs[:, 0].sum().detach().cpu().item())
            abs_counts["volume_pressure"] += int(volume_abs[:, 0].numel())

        for case_idx, case_id in enumerate(batch.case_ids):
            surface_valid = batch.surface_mask[case_idx].bool()
            if bool(surface_valid.any()):
                surface_pred_valid = surface_pred[case_idx][surface_valid]
                surface_target_valid = batch.surface_y[case_idx][surface_valid]
                _accumulate_case_rel_l2(
                    case_sums["surface_pressure"],
                    case_id=case_id,
                    pred=surface_pred_valid[:, 0:1],
                    target=surface_target_valid[:, 0:1],
                )
                _accumulate_case_rel_l2(
                    case_sums["wall_shear"],
                    case_id=case_id,
                    pred=surface_pred_valid[:, 1:4],
                    target=surface_target_valid[:, 1:4],
                )
                for channel, axis in enumerate(("x", "y", "z"), start=1):
                    _accumulate_case_rel_l2(
                        case_sums[f"wall_shear_{axis}"],
                        case_id=case_id,
                        pred=surface_pred_valid[:, channel : channel + 1],
                        target=surface_target_valid[:, channel : channel + 1],
                    )
            volume_valid = batch.volume_mask[case_idx].bool()
            if bool(volume_valid.any()):
                _accumulate_case_rel_l2(
                    case_sums["volume_pressure"],
                    case_id=case_id,
                    pred=volume_pred[case_idx][volume_valid],
                    target=batch.volume_y[case_idx][volume_valid],
                )

    surface_pressure_rel_l2, surface_cases = _rel_l2(case_sums["surface_pressure"])
    wall_shear_rel_l2, wall_shear_cases = _rel_l2(case_sums["wall_shear"])
    wall_shear_x_rel_l2, _ = _rel_l2(case_sums["wall_shear_x"])
    wall_shear_y_rel_l2, _ = _rel_l2(case_sums["wall_shear_y"])
    wall_shear_z_rel_l2, _ = _rel_l2(case_sums["wall_shear_z"])
    volume_pressure_rel_l2, volume_cases = _rel_l2(case_sums["volume_pressure"])
    abupt_axis_mean_rel_l2 = _finite_mean(
        [
            surface_pressure_rel_l2,
            wall_shear_x_rel_l2,
            wall_shear_y_rel_l2,
            wall_shear_z_rel_l2,
            volume_pressure_rel_l2,
        ]
    )
    mae_values = {
        key: abs_sums[key] / max(abs_counts[key], 1)
        for key in abs_sums
    }
    wall_shear_vector_mae = wall_shear_vector_abs_sum / max(wall_shear_vector_count, 1)
    loss = (surface_loss_sse + volume_loss_sse) / max(surface_loss_count + volume_loss_count, 1)
    return {
        "loss": loss,
        "surface_loss": surface_loss_sse / max(surface_loss_count, 1),
        "volume_loss": volume_loss_sse / max(volume_loss_count, 1),
        "surface_pressure_mae": mae_values["surface_pressure"],
        "wall_shear_mae": mae_values["wall_shear"],
        "wall_shear_vector_mae": wall_shear_vector_mae,
        "wall_shear_x_mae": mae_values["wall_shear_x"],
        "wall_shear_y_mae": mae_values["wall_shear_y"],
        "wall_shear_z_mae": mae_values["wall_shear_z"],
        "volume_pressure_mae": mae_values["volume_pressure"],
        "surface_pressure_rel_l2": surface_pressure_rel_l2,
        "surface_pressure_rel_l2_pct": surface_pressure_rel_l2 * 100.0,
        "wall_shear_rel_l2": wall_shear_rel_l2,
        "wall_shear_rel_l2_pct": wall_shear_rel_l2 * 100.0,
        "wall_shear_x_rel_l2": wall_shear_x_rel_l2,
        "wall_shear_x_rel_l2_pct": wall_shear_x_rel_l2 * 100.0,
        "wall_shear_y_rel_l2": wall_shear_y_rel_l2,
        "wall_shear_y_rel_l2_pct": wall_shear_y_rel_l2 * 100.0,
        "wall_shear_z_rel_l2": wall_shear_z_rel_l2,
        "wall_shear_z_rel_l2_pct": wall_shear_z_rel_l2 * 100.0,
        "volume_pressure_rel_l2": volume_pressure_rel_l2,
        "volume_pressure_rel_l2_pct": volume_pressure_rel_l2 * 100.0,
        "abupt_axis_mean_rel_l2": abupt_axis_mean_rel_l2,
        "abupt_axis_mean_rel_l2_pct": abupt_axis_mean_rel_l2 * 100.0,
        "cases": max(surface_cases, wall_shear_cases, volume_cases),
        "surface_cases": surface_cases,
        "volume_cases": volume_cases,
    }


def _sanitize_artifact_token(value: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "-" for c in value)
    return out.strip("-_.") or "run"


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_model_artifact(
    run,
    model_path: Path,
    config_path: Path,
    config: Config,
    best_metrics: dict[str, float],
    test_metrics: dict[str, dict[str, float]],
    n_params: int,
) -> None:
    base = config.wandb_name or config.agent or "drivaerml"
    artifact_name = f"model-{_sanitize_artifact_token(base)}-{run.id}"
    description = (
        "DrivAerML surface/volume Transolver checkpoint; "
        f"best val abupt_axis_mean_rel_l2_pct = {best_metrics['abupt_axis_mean_rel_l2_pct']:.4f}"
    )
    metadata = {
        "run_id": run.id,
        "run_name": run.name,
        "agent": config.agent,
        "wandb_name": config.wandb_name,
        "git_commit": _git_commit_short(),
        "n_params": n_params,
        "best_epoch": int(best_metrics["epoch"]),
        "best_val_primary/abupt_axis_mean_rel_l2_pct": best_metrics["abupt_axis_mean_rel_l2_pct"],
        "best_val/surface_pressure_mae": best_metrics["surface_pressure_mae"],
        "best_val/wall_shear_mae": best_metrics["wall_shear_mae"],
        "best_val/volume_pressure_mae": best_metrics["volume_pressure_mae"],
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "train_surface_points": config.train_surface_points,
        "eval_surface_points": config.eval_surface_points,
        "train_volume_points": config.train_volume_points,
        "eval_volume_points": config.eval_volume_points,
    }
    for split_name, metrics in test_metrics.items():
        metadata[f"{split_name}/abupt_axis_mean_rel_l2_pct"] = metrics["abupt_axis_mean_rel_l2_pct"]
        metadata[f"{split_name}/surface_pressure_mae"] = metrics["surface_pressure_mae"]
        metadata[f"{split_name}/wall_shear_mae"] = metrics["wall_shear_mae"]
        metadata[f"{split_name}/volume_pressure_mae"] = metrics["volume_pressure_mae"]
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=description,
        metadata=metadata,
    )
    artifact.add_file(str(model_path), name="checkpoint.pt")
    artifact.add_file(str(config_path), name="config.yaml")
    run.log_artifact(artifact, aliases=["best", f"epoch-{int(best_metrics['epoch'])}"])
    print(f"Logged model artifact '{artifact_name}'")


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix:<14s} "
        f"loss={metrics['loss']:.5f} "
        f"abupt_axis_rel_l2_pct={metrics['abupt_axis_mean_rel_l2_pct']:.4f} "
        f"surface_p_mae={metrics['surface_pressure_mae']:.5f} "
        f"volume_p_mae={metrics['volume_pressure_mae']:.5f} "
        f"wall_shear_mae={metrics['wall_shear_mae']:.5f} "
        f"cases={int(metrics['cases'])}"
    )


def main(argv: Iterable[str] | None = None) -> None:
    config = parse_args(argv)
    kill_thresholds = parse_kill_thresholds(config.kill_thresholds)
    max_epochs = min(config.epochs, 3) if config.debug else config.epochs
    timeout_minutes = float(os.environ.get("SENPAI_TIMEOUT_MINUTES", "30"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if config.debug else ""))

    train_loader, val_loaders, test_loaders, stats = make_loaders(config)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    model = build_model(config).to(device)
    if config.compile_model:
        model = torch.compile(model)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"Model: SurfaceTransolver grouped surface+volume ({n_params / 1e6:.2f}M params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    ema = EMA(model, decay=config.ema_decay, start_step=config.ema_start_step) if config.use_ema else None
    total_estimated_steps = max(1, max_epochs * max(len(train_loader), 1))
    if kill_thresholds:
        print("Kill thresholds:", "; ".join(threshold.describe() for threshold in kill_thresholds))
    train_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)
    val_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)
    full_val_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)
    test_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        group=config.wandb_group or None,
        name=config.wandb_name or None,
        tags=[config.agent] if config.agent else [],
        config={
            **asdict(config),
            "n_params": n_params,
            "train_views": len(train_loader.dataset),
            "val_views": {k: len(v.dataset) for k, v in val_loaders.items()},
            "test_views": {k: len(v.dataset) for k, v in test_loaders.items()},
            "surface_targets": SURFACE_TARGET_NAMES,
            "volume_targets": VOLUME_TARGET_NAMES,
            "total_estimated_steps": total_estimated_steps,
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    wandb.define_metric("val_primary/*", step_metric="global_step")
    wandb.define_metric("full_val/*", step_metric="global_step")
    wandb.define_metric("full_val_primary/*", step_metric="global_step")
    wandb.define_metric("test/*", step_metric="global_step")
    wandb.define_metric("test_primary/*", step_metric="global_step")
    wandb.define_metric("train/slope/*", step_metric="global_step")
    wandb.define_metric("val/slope/*", step_metric="global_step")
    wandb.define_metric("full_val/slope/*", step_metric="global_step")
    wandb.define_metric("test/slope/*", step_metric="global_step")
    wandb.define_metric("train/grad/*", step_metric="global_step")
    wandb.define_metric("train/grad_module/*", step_metric="global_step")
    wandb.define_metric("train/grad_param/*", step_metric="global_step")
    wandb.define_metric("train/grad_type/*", step_metric="global_step")
    wandb.define_metric("train/grad_hist/*", step_metric="global_step")
    wandb.define_metric("train/grad_hist_param/*", step_metric="global_step")
    wandb.define_metric("train/weight/*", step_metric="global_step")
    wandb.define_metric("train/weight_module/*", step_metric="global_step")
    wandb.define_metric("train/weight_param/*", step_metric="global_step")
    wandb.define_metric("train/weight_type/*", step_metric="global_step")
    wandb.define_metric("train/weight_hist/*", step_metric="global_step")
    wandb.define_metric("train/weight_hist_param/*", step_metric="global_step")
    wandb.define_metric("lr", step_metric="global_step")

    output_dir = Path(config.output_dir) / f"run-{run.id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "checkpoint.pt"
    config_path = output_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(asdict(config), f)

    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    global_step = 0
    early_stop_reason: str | None = None
    timeout_hit = False
    train_start = time.time()
    val_budget_minutes = float(os.environ.get("SENPAI_VAL_BUDGET_MINUTES", "90"))
    train_timeout_minutes = max(1.0, timeout_minutes - val_budget_minutes)

    for epoch in range(max_epochs):
        if (time.time() - train_start) / 60.0 >= timeout_minutes:
            print(f"Timeout ({timeout_minutes:.1f} min). Stopping.")
            break

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}", leave=False):
            loss, batch_loss_metrics = train_loss(
                model,
                batch,
                transform,
                device,
                config.amp_mode,
                surface_loss_weight=config.surface_loss_weight,
                volume_loss_weight=config.volume_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            should_log_gradients = (
                config.gradient_log_every > 0
                and (global_step + 1) % config.gradient_log_every == 0
            )
            should_log_weights = (
                config.weight_log_every > 0
                and (global_step + 1) % config.weight_log_every == 0
            )
            gradient_metrics = (
                collect_gradient_metrics(
                    model,
                    log_histograms=config.log_gradient_histograms,
                )
                if should_log_gradients
                else {}
            )
            optimizer.step()
            if ema is not None:
                ema.update(model)
            weight_metrics = (
                collect_weight_metrics(
                    model,
                    log_histograms=config.log_weight_histograms,
                )
                if should_log_weights
                else {}
            )
            train_loss_sum += float(loss.detach().cpu().item())
            n_batches += 1
            global_step += 1
            train_log: dict[str, object] = {
                "train/loss": float(loss.detach().cpu().item()),
                "train/surface_loss": batch_loss_metrics["surface_loss"],
                "train/volume_loss": batch_loss_metrics["volume_loss"],
                "global_step": global_step,
                **gradient_metrics,
                **weight_metrics,
            }
            train_log.update(
                train_slope_tracker.update(
                    global_step=global_step,
                    metrics=train_log,
                    namespace="train",
                )
            )
            early_stop_reason = check_kill_thresholds(
                global_step=global_step,
                metrics=train_log,
                thresholds=kill_thresholds,
            )
            if early_stop_reason is not None:
                train_log["early_stop/triggered"] = 1.0
            wandb.log(train_log)
            if early_stop_reason is not None:
                print(early_stop_reason)
                break
            if (time.time() - train_start) / 60.0 >= train_timeout_minutes:
                print(
                    f"Train timeout ({train_timeout_minutes:.1f} min) mid-epoch "
                    f"at step {global_step}. Forcing validation and stopping."
                )
                timeout_hit = True
                break

        scheduler.step()
        epoch_train_loss = train_loss_sum / max(n_batches, 1)
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        should_validate = (
            epoch == 0
            or (epoch + 1) % max(config.validation_every, 1) == 0
            or epoch + 1 == max_epochs
            or (timeout_hit and n_batches > 0)
        )

        log_metrics = {
            "train/epoch_loss": epoch_train_loss,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": dt,
            "global_step": global_step,
        }
        if early_stop_reason is not None:
            log_metrics["early_stop/triggered"] = 1.0
            wandb.log(log_metrics)
            break

        if not should_validate:
            early_stop_reason = early_stop_reason or check_kill_thresholds(
                global_step=global_step,
                metrics=log_metrics,
                thresholds=kill_thresholds,
            )
            if early_stop_reason is not None:
                log_metrics["early_stop/triggered"] = 1.0
            wandb.log(log_metrics)
            print(
                f"Epoch {epoch + 1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB] "
                f"train_loss={epoch_train_loss:.5f}"
            )
            if early_stop_reason is not None:
                print(early_stop_reason)
                break
            continue

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        val_metrics = {
            name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
            for name, loader in val_loaders.items()
        }
        if ema is not None:
            ema.restore(model)

        primary_val = val_metrics["val_surface"]["abupt_axis_mean_rel_l2_pct"]
        log_metrics.update(
            {
                "val_primary/abupt_axis_mean_rel_l2_pct": primary_val,
                "val_primary/abupt_axis_mean_rel_l2": val_metrics["val_surface"]["abupt_axis_mean_rel_l2"],
                "val_primary/surface_pressure_mae": val_metrics["val_surface"]["surface_pressure_mae"],
                "val_primary/wall_shear_mae": val_metrics["val_surface"]["wall_shear_mae"],
                "val_primary/volume_pressure_mae": val_metrics["val_surface"]["volume_pressure_mae"],
                "val_primary/surface_pressure_rel_l2_pct": val_metrics["val_surface"]["surface_pressure_rel_l2_pct"],
                "val_primary/wall_shear_rel_l2_pct": val_metrics["val_surface"]["wall_shear_rel_l2_pct"],
                "val_primary/wall_shear_x_rel_l2_pct": val_metrics["val_surface"]["wall_shear_x_rel_l2_pct"],
                "val_primary/wall_shear_y_rel_l2_pct": val_metrics["val_surface"]["wall_shear_y_rel_l2_pct"],
                "val_primary/wall_shear_z_rel_l2_pct": val_metrics["val_surface"]["wall_shear_z_rel_l2_pct"],
                "val_primary/volume_pressure_rel_l2_pct": val_metrics["val_surface"]["volume_pressure_rel_l2_pct"],
            }
        )
        for split_name, metrics in val_metrics.items():
            for key, value in metrics.items():
                log_metrics[f"val/{split_name}/{key}"] = value
        log_metrics.update(
            val_slope_tracker.update(
                global_step=global_step,
                metrics=log_metrics,
                namespace="val",
                force=True,
            )
        )
        early_stop_reason = early_stop_reason or check_kill_thresholds(
            global_step=global_step,
            metrics=log_metrics,
            thresholds=kill_thresholds,
        )
        if early_stop_reason is not None:
            log_metrics["early_stop/triggered"] = 1.0
        wandb.log(log_metrics)

        improved = primary_val < best_val
        if improved:
            best_val = primary_val
            best_metrics = {"epoch": float(epoch + 1), **val_metrics["val_surface"]}
            save_model = model
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
                save_model = model
            torch.save(
                {
                    "model": save_model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                },
                model_path,
            )
            if ema is not None:
                ema.restore(model)

        tag = " *" if improved else ""
        print(
            f"Epoch {epoch + 1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB] "
            f"train_loss={epoch_train_loss:.5f} "
            f"val_abupt_axis_rel_l2_pct={primary_val:.4f}{tag}"
        )
        print_metrics("val_surface", val_metrics["val_surface"])
        if early_stop_reason is not None:
            print(early_stop_reason)
            break
        if timeout_hit:
            break

    total_minutes = (time.time() - train_start) / 60.0
    print(f"\nTraining done in {total_minutes:.1f} min")

    if early_stop_reason is not None:
        wandb.summary.update(
            {
                "early_stop/triggered": 1.0,
                "early_stop/reason": early_stop_reason,
                "early_stop/global_step": global_step,
                "total_train_minutes": total_minutes,
            }
        )
        wandb.finish()
        return

    if not best_metrics:
        print("No validation checkpoint was saved.")
        wandb.finish()
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    print(
        f"Best val: epoch {int(best_metrics['epoch'])}, "
        f"abupt_axis_mean_rel_l2_pct={best_metrics['abupt_axis_mean_rel_l2_pct']:.4f}"
    )
    wandb.summary.update(
        {
            "best_epoch": int(best_metrics["epoch"]),
            "best_val_primary/abupt_axis_mean_rel_l2_pct": best_metrics["abupt_axis_mean_rel_l2_pct"],
            "best_val/surface_pressure_mae": best_metrics["surface_pressure_mae"],
            "best_val/wall_shear_mae": best_metrics["wall_shear_mae"],
            "best_val/volume_pressure_mae": best_metrics["volume_pressure_mae"],
            "total_train_minutes": total_minutes,
        }
    )

    full_val_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
        for name, loader in val_loaders.items()
    }
    full_val_primary = full_val_metrics["val_surface"]["abupt_axis_mean_rel_l2_pct"]
    full_val_log: dict[str, object] = {
        "full_val_primary/abupt_axis_mean_rel_l2_pct": full_val_primary,
        "full_val_primary/abupt_axis_mean_rel_l2": full_val_metrics["val_surface"]["abupt_axis_mean_rel_l2"],
        "full_val_primary/surface_pressure_mae": full_val_metrics["val_surface"]["surface_pressure_mae"],
        "full_val_primary/wall_shear_mae": full_val_metrics["val_surface"]["wall_shear_mae"],
        "full_val_primary/volume_pressure_mae": full_val_metrics["val_surface"]["volume_pressure_mae"],
        "full_val_primary/surface_pressure_rel_l2_pct": full_val_metrics["val_surface"]["surface_pressure_rel_l2_pct"],
        "full_val_primary/wall_shear_rel_l2_pct": full_val_metrics["val_surface"]["wall_shear_rel_l2_pct"],
        "full_val_primary/wall_shear_x_rel_l2_pct": full_val_metrics["val_surface"]["wall_shear_x_rel_l2_pct"],
        "full_val_primary/wall_shear_y_rel_l2_pct": full_val_metrics["val_surface"]["wall_shear_y_rel_l2_pct"],
        "full_val_primary/wall_shear_z_rel_l2_pct": full_val_metrics["val_surface"]["wall_shear_z_rel_l2_pct"],
        "full_val_primary/volume_pressure_rel_l2_pct": full_val_metrics["val_surface"]["volume_pressure_rel_l2_pct"],
        "global_step": global_step,
    }
    for split_name, metrics in full_val_metrics.items():
        for key, value in metrics.items():
            full_val_log[f"full_val/{split_name}/{key}"] = value
    full_val_log.update(
        full_val_slope_tracker.update(
            global_step=global_step,
            metrics=full_val_log,
            namespace="full_val",
            force=True,
        )
    )
    wandb.log(full_val_log)
    wandb.summary.update(_numeric_metric_items(full_val_log))
    print_metrics("full_val", full_val_metrics["val_surface"])

    test_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
        for name, loader in test_loaders.items()
    }
    test_primary = test_metrics["test_surface"]["abupt_axis_mean_rel_l2_pct"]
    test_log: dict[str, object] = {
        "test_primary/abupt_axis_mean_rel_l2_pct": test_primary,
        "test_primary/abupt_axis_mean_rel_l2": test_metrics["test_surface"]["abupt_axis_mean_rel_l2"],
        "test_primary/surface_pressure_mae": test_metrics["test_surface"]["surface_pressure_mae"],
        "test_primary/wall_shear_mae": test_metrics["test_surface"]["wall_shear_mae"],
        "test_primary/volume_pressure_mae": test_metrics["test_surface"]["volume_pressure_mae"],
        "test_primary/surface_pressure_rel_l2_pct": test_metrics["test_surface"]["surface_pressure_rel_l2_pct"],
        "test_primary/wall_shear_rel_l2_pct": test_metrics["test_surface"]["wall_shear_rel_l2_pct"],
        "test_primary/wall_shear_x_rel_l2_pct": test_metrics["test_surface"]["wall_shear_x_rel_l2_pct"],
        "test_primary/wall_shear_y_rel_l2_pct": test_metrics["test_surface"]["wall_shear_y_rel_l2_pct"],
        "test_primary/wall_shear_z_rel_l2_pct": test_metrics["test_surface"]["wall_shear_z_rel_l2_pct"],
        "test_primary/volume_pressure_rel_l2_pct": test_metrics["test_surface"]["volume_pressure_rel_l2_pct"],
        "global_step": global_step,
    }
    for split_name, metrics in test_metrics.items():
        for key, value in metrics.items():
            test_log[f"test/{split_name}/{key}"] = value
    test_log.update(
        test_slope_tracker.update(
            global_step=global_step,
            metrics=test_log,
            namespace="test",
            force=True,
        )
    )
    wandb.log(test_log)
    wandb.summary.update(_numeric_metric_items(test_log))
    print_metrics("test_surface", test_metrics["test_surface"])

    log_model_artifact(
        run=run,
        model_path=model_path,
        config_path=config_path,
        config=config,
        best_metrics=best_metrics,
        test_metrics=test_metrics,
        n_params=n_params,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
