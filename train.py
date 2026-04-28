# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Train a surface-pressure Transolver on DrivAerML.

Primary metric:
    val_primary/surface_rel_l2_pct

Usage:
    python train.py --epochs 50 --agent <name> --wandb_name "<name>/<experiment>"
"""

from __future__ import annotations

import argparse
import math
import os
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

from data import SURFACE_X_DIM, SURFACE_Y_DIM, SurfaceBatch, load_data, pad_collate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _init_linear(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


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
            slice_weights = slice_weights * attn_mask[:, None, :, None].float()
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
        return self.proj_dropout(self.proj(out_x))


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
        x = x + self.attention(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
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
    """Surface-only Transolver for DrivAerML pressure coefficient prediction."""

    def __init__(
        self,
        *,
        space_dim: int = 3,
        input_dim: int = SURFACE_X_DIM,
        output_dim: int = SURFACE_Y_DIM,
        n_layers: int = 3,
        n_hidden: int = 192,
        dropout: float = 0.0,
        n_head: int = 3,
        mlp_ratio: int = 4,
        slice_num: int = 96,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        extra_dim = max(0, input_dim - space_dim)

        self.pos_embed = ContinuousSincosEmbed(hidden_dim=n_hidden, input_dim=space_dim)
        self.surface_bias = MLP(input_dim=n_hidden, hidden_dim=n_hidden, output_dim=n_hidden)
        self.project_features = LinearProjection(extra_dim, n_hidden) if extra_dim > 0 else None
        self.placeholder = nn.Parameter(torch.rand(1, 1, n_hidden) / n_hidden)
        self.backbone = Transformer(
            depth=n_layers,
            hidden_dim=n_hidden,
            num_heads=n_head,
            mlp_expansion_factor=mlp_ratio,
            num_slices=slice_num,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(n_hidden, eps=1e-6)
        self.out = LinearProjection(n_hidden, output_dim)

    def forward(self, *, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pos = x[:, :, : self.space_dim]
        hidden = self.pos_embed(pos)
        if self.project_features is not None and x.shape[-1] > self.space_dim:
            hidden = hidden + self.project_features(x[:, :, self.space_dim :])
        hidden = self.surface_bias(hidden) + self.placeholder
        hidden = self.backbone(hidden, attn_mask=mask)
        preds = self.out(self.norm(hidden)) * mask.unsqueeze(-1)
        return {"preds": preds, "hidden": hidden}


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
    batch_size: int = 1
    epochs: int = 50
    train_surface_points: int = 40_000
    eval_surface_points: int = 40_000
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
    compile_model: bool = False
    debug: bool = False


class TargetTransform:
    def __init__(self, *, y_mean: torch.Tensor, y_std: torch.Tensor):
        self.y_mean = y_mean
        self.y_std = y_std.clamp(min=1e-6)

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean.to(y.device)) / self.y_std.to(y.device)

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)


def parse_args(argv: Iterable[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="DrivAerML surface-pressure trainer")
    defaults = Config()
    for field in fields(Config):
        value = getattr(defaults, field.name)
        arg_name = f"--{field.name.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value)
            parser.add_argument(f"--no-{field.name.replace('_', '-')}", action="store_false", dest=field.name)
        else:
            parser.add_argument(arg_name, type=type(value), default=value)
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


def train_loss(
    model: nn.Module,
    batch: SurfaceBatch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
) -> torch.Tensor:
    batch = batch.to(device)
    target = transform.apply(batch.y)
    with autocast_context(device, amp_mode):
        pred = model(x=batch.x, mask=batch.mask)["preds"]
        return F.mse_loss(pred[batch.mask], target[batch.mask])


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
    loss_sum = 0.0
    loss_count = 0
    case_sums: dict[str, list[float]] = {}

    for batch in loader:
        batch = batch.to(device)
        target_norm = transform.apply(batch.y)
        with autocast_context(device, amp_mode):
            pred_norm = model(x=batch.x, mask=batch.mask)["preds"]
        pred_norm = pred_norm.float()
        loss_sum += float(F.mse_loss(pred_norm[batch.mask], target_norm[batch.mask]).detach().cpu().item())
        loss_count += 1
        pred = transform.invert(pred_norm)

        for case_idx, case_id in enumerate(batch.case_ids):
            valid = batch.mask[case_idx].bool()
            if not valid.any():
                continue
            pred_valid = pred[case_idx][valid]
            target_valid = batch.y[case_idx][valid]
            target_sq = float(target_valid.square().sum().detach().cpu().item())
            if target_sq <= 0.0:
                continue
            state = case_sums.setdefault(case_id, [0.0, 0.0])
            state[0] += float((pred_valid - target_valid).square().sum().detach().cpu().item())
            state[1] += target_sq

    rel_values = [
        math.sqrt(error_sq / target_sq)
        for error_sq, target_sq in case_sums.values()
        if target_sq > 0.0
    ]
    rel_l2 = sum(rel_values) / max(len(rel_values), 1)
    return {
        "loss": loss_sum / max(loss_count, 1),
        "surface_rel_l2": rel_l2,
        "surface_rel_l2_pct": rel_l2 * 100.0,
        "cases": float(len(rel_values)),
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
        "DrivAerML surface Transolver checkpoint; "
        f"best val surface_rel_l2_pct = {best_metrics['surface_rel_l2_pct']:.4f}"
    )
    metadata = {
        "run_id": run.id,
        "run_name": run.name,
        "agent": config.agent,
        "wandb_name": config.wandb_name,
        "git_commit": _git_commit_short(),
        "n_params": n_params,
        "best_epoch": int(best_metrics["epoch"]),
        "best_val_primary/surface_rel_l2_pct": best_metrics["surface_rel_l2_pct"],
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "train_surface_points": config.train_surface_points,
        "eval_surface_points": config.eval_surface_points,
    }
    for split_name, metrics in test_metrics.items():
        metadata[f"{split_name}/surface_rel_l2_pct"] = metrics["surface_rel_l2_pct"]
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
        f"surface_rel_l2_pct={metrics['surface_rel_l2_pct']:.4f} "
        f"cases={int(metrics['cases'])}"
    )


def main(argv: Iterable[str] | None = None) -> None:
    config = parse_args(argv)
    max_epochs = min(config.epochs, 3) if config.debug else config.epochs
    timeout_minutes = float(os.environ.get("SENPAI_TIMEOUT_MINUTES", "30"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if config.debug else ""))

    train_loader, val_loaders, test_loaders, stats = make_loaders(config)
    transform = TargetTransform(
        y_mean=stats["y_mean"].to(device),
        y_std=stats["y_std"].to(device),
    )

    model = build_model(config).to(device)
    if config.compile_model:
        model = torch.compile(model)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"Model: SurfaceTransolver ({n_params / 1e6:.2f}M params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    ema = EMA(model, decay=config.ema_decay, start_step=config.ema_start_step) if config.use_ema else None

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
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    wandb.define_metric("val_primary/*", step_metric="global_step")
    wandb.define_metric("test/*", step_metric="global_step")
    wandb.define_metric("test_primary/*", step_metric="global_step")
    wandb.define_metric("train/grad/*", step_metric="global_step")
    wandb.define_metric("train/grad_module/*", step_metric="global_step")
    wandb.define_metric("train/grad_param/*", step_metric="global_step")
    wandb.define_metric("train/grad_type/*", step_metric="global_step")
    wandb.define_metric("train/grad_hist/*", step_metric="global_step")
    wandb.define_metric("train/grad_hist_param/*", step_metric="global_step")
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
    train_start = time.time()

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
            loss = train_loss(model, batch, transform, device, config.amp_mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            should_log_gradients = (
                config.gradient_log_every > 0
                and (global_step + 1) % config.gradient_log_every == 0
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
            train_loss_sum += float(loss.detach().cpu().item())
            n_batches += 1
            global_step += 1
            wandb.log(
                {
                    "train/loss": float(loss.detach().cpu().item()),
                    "global_step": global_step,
                    **gradient_metrics,
                }
            )

        scheduler.step()
        epoch_train_loss = train_loss_sum / max(n_batches, 1)

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        val_metrics = {
            name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
            for name, loader in val_loaders.items()
        }
        if ema is not None:
            ema.restore(model)

        primary_val = val_metrics["val_surface"]["surface_rel_l2_pct"]
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        log_metrics = {
            "train/epoch_loss": epoch_train_loss,
            "val_primary/surface_rel_l2_pct": primary_val,
            "val_primary/surface_rel_l2": val_metrics["val_surface"]["surface_rel_l2"],
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": dt,
            "global_step": global_step,
        }
        for split_name, metrics in val_metrics.items():
            for key, value in metrics.items():
                log_metrics[f"val/{split_name}/{key}"] = value
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
            f"val_surface_rel_l2_pct={primary_val:.4f}{tag}"
        )
        print_metrics("val_surface", val_metrics["val_surface"])

    total_minutes = (time.time() - train_start) / 60.0
    print(f"\nTraining done in {total_minutes:.1f} min")

    if not best_metrics:
        print("No validation checkpoint was saved.")
        wandb.finish()
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    print(
        f"Best val: epoch {int(best_metrics['epoch'])}, "
        f"surface_rel_l2_pct={best_metrics['surface_rel_l2_pct']:.4f}"
    )
    wandb.summary.update(
        {
            "best_epoch": int(best_metrics["epoch"]),
            "best_val_primary/surface_rel_l2_pct": best_metrics["surface_rel_l2_pct"],
            "total_train_minutes": total_minutes,
        }
    )

    test_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
        for name, loader in test_loaders.items()
    }
    test_primary = test_metrics["test_surface"]["surface_rel_l2_pct"]
    test_log: dict[str, float] = {
        "test_primary/surface_rel_l2_pct": test_primary,
        "test_primary/surface_rel_l2": test_metrics["test_surface"]["surface_rel_l2"],
        "global_step": global_step,
    }
    for split_name, metrics in test_metrics.items():
        for key, value in metrics.items():
            test_log[f"test/{split_name}/{key}"] = value
    wandb.log(test_log)
    wandb.summary.update(test_log)
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
