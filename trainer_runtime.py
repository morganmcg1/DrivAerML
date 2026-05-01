# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Runtime helpers for the DrivAerML trainer."""

from __future__ import annotations

import math
import os
import re
import subprocess
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from data import (
    SURFACE_TARGET_NAMES,
    SurfaceBatch,
    VOLUME_TARGET_NAMES,
    VOLUME_Y_DIM,
    load_data,
    pad_collate,
)


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


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if torch.cuda.is_available():
        if enabled:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if enabled:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available, but WORLD_SIZE > 1")
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return DistributedState(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_any(state: DistributedState, value: bool, device: torch.device) -> bool:
    if not state.enabled:
        return bool(value)
    flag = torch.tensor(1 if value else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def distributed_barrier(state: DistributedState) -> None:
    if state.enabled:
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    while isinstance(current, DistributedDataParallel):
        current = current.module
    return getattr(current, "_orig_mod", current)


class StridedDistributedSampler(Sampler[int]):
    """Shard eval indices without padding or duplication."""

    def __init__(self, dataset, *, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if len(self.dataset) <= self.rank:
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


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


def autocast_context(device: torch.device, amp_mode: str):
    if amp_mode != "bf16" or device.type != "cuda":
        return nullcontext()
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: True)
    if not supports_bf16():
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def resolve_num_workers(config) -> int:
    if config.num_workers >= 0:
        return config.num_workers
    if config.debug or not torch.cuda.is_available():
        return 0
    return min(4, os.cpu_count() or 4)


def loader_kwargs(config) -> dict[str, object]:
    num_workers = resolve_num_workers(config)
    kwargs: dict[str, object] = {
        "collate_fn": pad_collate,
        "num_workers": num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


def eval_loader_for_dataset(
    dataset,
    config,
    *,
    distributed_state: DistributedState | None = None,
) -> DataLoader:
    sampler = None
    if distributed_state is not None and distributed_state.enabled:
        sampler = StridedDistributedSampler(
            dataset,
            num_replicas=distributed_state.world_size,
            rank=distributed_state.rank,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler,
        **loader_kwargs(config),
    )


def full_eval_loaders_from(
    loaders: dict[str, DataLoader],
    config,
) -> dict[str, DataLoader]:
    return {
        name: eval_loader_for_dataset(loader.dataset, config, distributed_state=None)
        for name, loader in loaders.items()
    }


def make_loaders(
    config,
    distributed_state: DistributedState | None = None,
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
    train_sampler = None
    train_shuffle = True
    if distributed_state is not None and distributed_state.enabled:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=distributed_state.world_size,
            rank=distributed_state.rank,
            shuffle=True,
            drop_last=False,
        )
        train_shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        **loader_kwargs(config),
    )
    val_loaders = {
        name: eval_loader_for_dataset(ds, config, distributed_state=distributed_state)
        for name, ds in val_splits.items()
    }
    test_loaders = {
        name: eval_loader_for_dataset(ds, config, distributed_state=distributed_state)
        for name, ds in test_splits.items()
    }
    return train_loader, val_loaders, test_loaders, stats


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

    base_model = unwrap_model(model)
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

    base_model = unwrap_model(model)
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


def numeric_metric_items(metrics: dict[str, object]) -> dict[str, float]:
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
    numeric = numeric_metric_items(metrics)
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
    numeric = numeric_metric_items(metrics)
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


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.to(device=values.device, dtype=values.dtype)
    denominator = expanded_mask.expand_as(values).sum()
    if bool(denominator.detach().cpu().item() > 0):
        return (values * expanded_mask).sum() / denominator.clamp_min(1.0)
    return values.sum() * 0.0


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target).square(), mask)


def squared_relative_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.sum() * 0.0
    mask_float = mask.to(device=pred.device, dtype=pred.dtype)
    diff_sq = (pred.float() - target.float()).square().sum(dim=-1) * mask_float
    target_sq = target.float().square().sum(dim=-1) * mask_float
    denominator = target_sq.sum(dim=1)
    valid = denominator > 0
    if bool(valid.any()):
        return (diff_sq.sum(dim=1)[valid] / denominator[valid].clamp_min(1e-12)).mean()
    return pred.sum() * 0.0


EVAL_KEYS = (
    "surface_pressure",
    "wall_shear",
    "wall_shear_x",
    "wall_shear_y",
    "wall_shear_z",
    "volume_pressure",
)


@dataclass
class EvalAccumulator:
    surface_loss_sse: float = 0.0
    surface_loss_count: int = 0
    volume_loss_sse: float = 0.0
    volume_loss_count: int = 0
    abs_sums: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in EVAL_KEYS})
    abs_counts: dict[str, int] = field(default_factory=lambda: {key: 0 for key in EVAL_KEYS})
    wall_shear_vector_abs_sum: float = 0.0
    wall_shear_vector_count: int = 0
    case_sums: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {key: {} for key in EVAL_KEYS}
    )


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


def _mirror_y_inputs(
    surface_x: torch.Tensor | None,
    volume_x: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Reflect inputs across y=0: flips surface y and ny, volume y."""
    if surface_x is not None:
        surface_x = surface_x.clone()
        surface_x[..., 1] = -surface_x[..., 1]
        surface_x[..., 4] = -surface_x[..., 4]
    if volume_x is not None:
        volume_x = volume_x.clone()
        volume_x[..., 1] = -volume_x[..., 1]
    return surface_x, volume_x


def model_forward_norm(
    model: nn.Module,
    *,
    surface_x: torch.Tensor,
    surface_mask: torch.Tensor,
    volume_x: torch.Tensor,
    volume_mask: torch.Tensor,
    device: torch.device,
    amp_mode: str,
    mirror_tta: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass returning (surface_pred_norm, volume_pred_norm) in float32.

    When ``mirror_tta`` is true, also runs a y-mirrored forward pass and averages
    the predictions. The wsy channel of the mirrored surface output is sign-flipped
    before averaging because under y-reflection the wall-shear vector transforms as
    (wsx, wsy, wsz) -> (wsx, -wsy, wsz).
    """
    with autocast_context(device, amp_mode):
        out = model(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
    surface_pred = out["surface_preds"].float()
    volume_pred = out["volume_preds"].float()
    if not mirror_tta:
        return surface_pred, volume_pred

    surface_x_mir, volume_x_mir = _mirror_y_inputs(surface_x, volume_x)
    with autocast_context(device, amp_mode):
        out_mir = model(
            surface_x=surface_x_mir,
            surface_mask=surface_mask,
            volume_x=volume_x_mir,
            volume_mask=volume_mask,
        )
    surface_pred_mir = out_mir["surface_preds"].float()
    volume_pred_mir = out_mir["volume_preds"].float()
    if surface_pred_mir.shape[-1] >= 3:
        surface_pred_mir = surface_pred_mir.clone()
        surface_pred_mir[..., 2] = -surface_pred_mir[..., 2]
    return 0.5 * (surface_pred + surface_pred_mir), 0.5 * (volume_pred + volume_pred_mir)


def accumulate_eval_batch(
    accumulator: EvalAccumulator,
    *,
    model: nn.Module,
    batch: SurfaceBatch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    mirror_tta: bool = False,
) -> None:
    batch = batch.to(device)
    surface_target_norm = transform.apply_surface(batch.surface_y)
    volume_target_norm = transform.apply_volume(batch.volume_y)
    surface_pred_norm, volume_pred_norm = model_forward_norm(
        model,
        surface_x=batch.surface_x,
        surface_mask=batch.surface_mask,
        volume_x=batch.volume_x,
        volume_mask=batch.volume_mask,
        device=device,
        amp_mode=amp_mode,
        mirror_tta=mirror_tta,
    )
    surface_sse, surface_count = _masked_sse_count(surface_pred_norm, surface_target_norm, batch.surface_mask)
    volume_sse, volume_count = _masked_sse_count(volume_pred_norm, volume_target_norm, batch.volume_mask)
    accumulator.surface_loss_sse += surface_sse
    accumulator.surface_loss_count += surface_count
    accumulator.volume_loss_sse += volume_sse
    accumulator.volume_loss_count += volume_count
    surface_pred = transform.invert_surface(surface_pred_norm)
    volume_pred = transform.invert_volume(volume_pred_norm)

    if bool(batch.surface_mask.any()):
        surface_abs = (surface_pred - batch.surface_y).abs()
        valid_surface_abs = surface_abs[batch.surface_mask]
        accumulator.abs_sums["surface_pressure"] += float(valid_surface_abs[:, 0].sum().detach().cpu().item())
        accumulator.abs_counts["surface_pressure"] += int(valid_surface_abs[:, 0].numel())
        wall_abs = valid_surface_abs[:, 1:4]
        accumulator.abs_sums["wall_shear"] += float(wall_abs.sum().detach().cpu().item())
        accumulator.abs_counts["wall_shear"] += int(wall_abs.numel())
        for offset, axis in enumerate(("x", "y", "z")):
            channel = wall_abs[:, offset]
            accumulator.abs_sums[f"wall_shear_{axis}"] += float(channel.sum().detach().cpu().item())
            accumulator.abs_counts[f"wall_shear_{axis}"] += int(channel.numel())
        wall_vector_error = torch.linalg.vector_norm(
            surface_pred[batch.surface_mask][:, 1:4] - batch.surface_y[batch.surface_mask][:, 1:4],
            dim=-1,
        )
        accumulator.wall_shear_vector_abs_sum += float(wall_vector_error.sum().detach().cpu().item())
        accumulator.wall_shear_vector_count += int(wall_vector_error.numel())

    if bool(batch.volume_mask.any()):
        volume_abs = (volume_pred - batch.volume_y).abs()[batch.volume_mask]
        accumulator.abs_sums["volume_pressure"] += float(volume_abs[:, 0].sum().detach().cpu().item())
        accumulator.abs_counts["volume_pressure"] += int(volume_abs[:, 0].numel())

    for case_idx, case_id in enumerate(batch.case_ids):
        surface_valid = batch.surface_mask[case_idx].bool()
        if bool(surface_valid.any()):
            surface_pred_valid = surface_pred[case_idx][surface_valid]
            surface_target_valid = batch.surface_y[case_idx][surface_valid]
            _accumulate_case_rel_l2(
                accumulator.case_sums["surface_pressure"],
                case_id=case_id,
                pred=surface_pred_valid[:, 0:1],
                target=surface_target_valid[:, 0:1],
            )
            _accumulate_case_rel_l2(
                accumulator.case_sums["wall_shear"],
                case_id=case_id,
                pred=surface_pred_valid[:, 1:4],
                target=surface_target_valid[:, 1:4],
            )
            for channel, axis in enumerate(("x", "y", "z"), start=1):
                _accumulate_case_rel_l2(
                    accumulator.case_sums[f"wall_shear_{axis}"],
                    case_id=case_id,
                    pred=surface_pred_valid[:, channel : channel + 1],
                    target=surface_target_valid[:, channel : channel + 1],
                )
        volume_valid = batch.volume_mask[case_idx].bool()
        if bool(volume_valid.any()):
            _accumulate_case_rel_l2(
                accumulator.case_sums["volume_pressure"],
                case_id=case_id,
                pred=volume_pred[case_idx][volume_valid],
                target=batch.volume_y[case_idx][volume_valid],
            )


def merge_eval_accumulators(accumulators: Iterable[EvalAccumulator]) -> EvalAccumulator:
    merged = EvalAccumulator()
    for accumulator in accumulators:
        merged.surface_loss_sse += accumulator.surface_loss_sse
        merged.surface_loss_count += accumulator.surface_loss_count
        merged.volume_loss_sse += accumulator.volume_loss_sse
        merged.volume_loss_count += accumulator.volume_loss_count
        merged.wall_shear_vector_abs_sum += accumulator.wall_shear_vector_abs_sum
        merged.wall_shear_vector_count += accumulator.wall_shear_vector_count
        for key in EVAL_KEYS:
            merged.abs_sums[key] += accumulator.abs_sums[key]
            merged.abs_counts[key] += accumulator.abs_counts[key]
            for case_id, values in accumulator.case_sums[key].items():
                state = merged.case_sums[key].setdefault(case_id, [0.0, 0.0])
                state[0] += values[0]
                state[1] += values[1]
    return merged


def finalize_eval_accumulator(accumulator: EvalAccumulator) -> dict[str, float]:
    surface_pressure_rel_l2, surface_cases = _rel_l2(accumulator.case_sums["surface_pressure"])
    wall_shear_rel_l2, wall_shear_cases = _rel_l2(accumulator.case_sums["wall_shear"])
    wall_shear_x_rel_l2, _ = _rel_l2(accumulator.case_sums["wall_shear_x"])
    wall_shear_y_rel_l2, _ = _rel_l2(accumulator.case_sums["wall_shear_y"])
    wall_shear_z_rel_l2, _ = _rel_l2(accumulator.case_sums["wall_shear_z"])
    volume_pressure_rel_l2, volume_cases = _rel_l2(accumulator.case_sums["volume_pressure"])
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
        key: accumulator.abs_sums[key] / max(accumulator.abs_counts[key], 1)
        for key in EVAL_KEYS
    }
    wall_shear_vector_mae = accumulator.wall_shear_vector_abs_sum / max(
        accumulator.wall_shear_vector_count, 1
    )
    loss = (accumulator.surface_loss_sse + accumulator.volume_loss_sse) / max(
        accumulator.surface_loss_count + accumulator.volume_loss_count, 1
    )
    return {
        "loss": loss,
        "surface_loss": accumulator.surface_loss_sse / max(accumulator.surface_loss_count, 1),
        "volume_loss": accumulator.volume_loss_sse / max(accumulator.volume_loss_count, 1),
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


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    *,
    amp_mode: str = "none",
    distributed_state: DistributedState | None = None,
    mirror_tta: bool = False,
) -> dict[str, float]:
    model.eval()
    accumulator = EvalAccumulator()
    for batch in loader:
        accumulate_eval_batch(
            accumulator,
            model=model,
            batch=batch,
            transform=transform,
            device=device,
            amp_mode=amp_mode,
            mirror_tta=mirror_tta,
        )
    if distributed_state is not None and distributed_state.enabled:
        gathered: list[EvalAccumulator | None] = [None for _ in range(distributed_state.world_size)]
        dist.all_gather_object(gathered, accumulator)
        if not distributed_state.is_main:
            return {}
        accumulator = merge_eval_accumulators(acc for acc in gathered if acc is not None)
    return finalize_eval_accumulator(accumulator)


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
    config,
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


PRIMARY_METRIC_KEYS = (
    "abupt_axis_mean_rel_l2_pct",
    "abupt_axis_mean_rel_l2",
    "surface_pressure_mae",
    "wall_shear_mae",
    "volume_pressure_mae",
    "surface_pressure_rel_l2_pct",
    "wall_shear_rel_l2_pct",
    "wall_shear_x_rel_l2_pct",
    "wall_shear_y_rel_l2_pct",
    "wall_shear_z_rel_l2_pct",
    "volume_pressure_rel_l2_pct",
)


def primary_metric_log(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}/{key}": metrics[key]
        for key in PRIMARY_METRIC_KEYS
    }


def assert_required_finite_metrics(log: dict[str, object], prefix: str) -> None:
    required = [f"{prefix}/{key}" for key in PRIMARY_METRIC_KEYS]
    missing = [key for key in required if key not in log]
    nonfinite = []
    for key in required:
        value = log.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            nonfinite.append(key)
            continue
        if not math.isfinite(float(value)):
            nonfinite.append(key)
    if missing or nonfinite:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if nonfinite:
            parts.append(f"nonfinite={nonfinite}")
        raise RuntimeError(f"Invalid final metric contract for {prefix}: " + ", ".join(parts))


def is_valid_primary_metric(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) > 0.0


def should_update_best_checkpoint(primary_val: float, best_val: float) -> bool:
    return is_valid_primary_metric(primary_val) and primary_val < best_val


def timeout_budget_minutes(env: Mapping[str, str] = os.environ) -> tuple[float, float, float]:
    timeout_minutes = float(env.get("SENPAI_TIMEOUT_MINUTES", "30"))
    default_val_budget = min(90.0, max(1.0, timeout_minutes * 0.25))
    val_budget_minutes = float(env.get("SENPAI_VAL_BUDGET_MINUTES", str(default_val_budget)))
    train_timeout_minutes = max(0.0, timeout_minutes - max(0.0, val_budget_minutes))
    return timeout_minutes, val_budget_minutes, train_timeout_minutes


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config,
    max_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    t_max = config.lr_cosine_t_max if config.lr_cosine_t_max > 0 else max_epochs
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, t_max),
        eta_min=config.lr_min,
    )
    if config.lr_warmup_epochs <= 0:
        return cosine_scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=max(1, config.lr_warmup_epochs),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.lr_warmup_epochs],
    )


def global_grad_norm(parameters: Iterable[torch.nn.Parameter], device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total = total + grad.square().sum()
    return total.sqrt()


def run_name_for_rank(config, state: DistributedState) -> str | None:
    if not state.enabled:
        return config.wandb_name or None
    base = config.wandb_name or config.agent or "drivaerml"
    return f"{base}-rank{state.rank}"


def wandb_group_for_rank(config, state: DistributedState) -> str | None:
    if config.wandb_group:
        return config.wandb_group
    if state.enabled:
        return config.wandb_name or config.agent or "drivaerml-ddp"
    return None


def metric_namespace(prefix: str, split_name: str, metrics: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}/{split_name}/{key}": value
        for key, value in metrics.items()
    }


def define_wandb_metrics() -> None:
    wandb.define_metric("global_step")
    for metric_prefix in (
        "train/*",
        "val/*",
        "val_raw/*",
        "val_primary/*",
        "val_raw_primary/*",
        "full_val/*",
        "full_val_primary/*",
        "no_tta_full_val/*",
        "no_tta_full_val_primary/*",
        "mirror_tta_full_val/*",
        "mirror_tta_full_val_primary/*",
        "no_tta_test/*",
        "no_tta_test_primary/*",
        "mirror_tta_test/*",
        "mirror_tta_test_primary/*",
        "test/*",
        "test_primary/*",
        "train/slope/*",
        "val/slope/*",
        "full_val/slope/*",
        "test/slope/*",
        "train/grad/*",
        "train/grad_module/*",
        "train/grad_param/*",
        "train/grad_type/*",
        "train/grad_hist/*",
        "train/grad_hist_param/*",
        "train/weight/*",
        "train/weight_module/*",
        "train/weight_param/*",
        "train/weight_type/*",
        "train/weight_hist/*",
        "train/weight_hist_param/*",
        "lr",
    ):
        wandb.define_metric(metric_prefix, step_metric="global_step")


def init_wandb_run(
    *,
    config,
    state: DistributedState,
    n_params: int,
    train_loader: DataLoader,
    val_loaders: dict[str, DataLoader],
    test_loaders: dict[str, DataLoader],
    total_estimated_steps: int,
    max_epochs: int,
    train_timeout_minutes: float,
    val_budget_minutes: float,
):
    tags = [config.agent] if config.agent else []
    if state.enabled:
        tags.extend(["ddp", f"rank:{state.rank}"])
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        group=wandb_group_for_rank(config, state),
        name=run_name_for_rank(config, state),
        tags=tags,
        config={
            **asdict(config),
            "n_params": n_params,
            "train_views": len(train_loader.dataset),
            "val_views": {key: len(loader.dataset) for key, loader in val_loaders.items()},
            "test_views": {key: len(loader.dataset) for key, loader in test_loaders.items()},
            "surface_targets": SURFACE_TARGET_NAMES,
            "volume_targets": VOLUME_TARGET_NAMES,
            "total_estimated_steps": total_estimated_steps,
            "max_epochs_effective": max_epochs,
            "ddp_enabled": state.enabled,
            "ddp_rank": state.rank,
            "ddp_world_size": state.world_size,
            "train_timeout_minutes": train_timeout_minutes,
            "val_budget_minutes": val_budget_minutes,
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    define_wandb_metrics()
    return run


class _AlphonseFourierEmbed(nn.Module):
    """Reconstructed FourierEmbed used for the alphonse Wave 1 m9775k1v run.

    The exact code was a local edit and was never committed. Reconstructed from
    the state_dict's parameter names (``pos_embed.center``, ``pos_embed.half_range``,
    ``pos_embed.freqs``, ``pos_embed.proj.project.{weight,bias}``) and the buffer
    values: center=[1, 0, 0], half_range=[13, 5, 3], freqs=2^k * pi for k=0..7.
    Used only by run_eval_only to load that specific checkpoint.
    """

    def __init__(self, hidden_dim: int, input_dim: int = 3, num_freqs: int = 8):
        super().__init__()
        from model import LinearProjection  # local import to avoid cycles
        self.register_buffer("center", torch.zeros(input_dim))
        self.register_buffer("half_range", torch.ones(input_dim))
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(num_freqs).float() * math.pi
        )
        raw_dim = input_dim * num_freqs * 2
        self.proj = LinearProjection(raw_dim, hidden_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.float()
        normalized = (coords - self.center) / self.half_range.clamp(min=1e-6)
        angles = normalized.unsqueeze(-1) * self.freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        emb = emb.flatten(start_dim=-2)
        return self.proj(emb)


def resolve_checkpoint_path(spec: str) -> Path:
    """Resolve a checkpoint spec to a local file.

    Accepts a local filesystem path or a W&B artifact reference of the form
    ``wandb://entity/project/artifact:alias`` or just
    ``entity/project/artifact:alias``.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("eval_checkpoint must be set when --eval-only is used")
    local = Path(spec)
    if local.exists():
        return local
    if spec.startswith("wandb://"):
        spec = spec[len("wandb://"):]
    if spec.count("/") >= 2 and ":" in spec.rsplit("/", 1)[-1]:
        artifact = wandb.use_artifact(spec, type="model")
        artifact_dir = Path(artifact.download())
        for candidate in (artifact_dir / "checkpoint.pt", artifact_dir / "model.pt"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No checkpoint.pt or model.pt in artifact dir {artifact_dir}")
    raise FileNotFoundError(f"Cannot resolve eval_checkpoint spec: {spec!r}")


def run_eval_only(
    *,
    run,
    model: nn.Module,
    config,
    transform: TargetTransform,
    device: torch.device,
    final_val_loaders: dict[str, DataLoader],
    final_test_loaders: dict[str, DataLoader],
    n_params: int,
) -> None:
    """Load a checkpoint and log no-TTA vs mirror-TTA metrics on val + test.

    Intended for validating the TTA implementation against an existing baseline
    checkpoint. Logs all four prefixes:
      no_tta_full_val_primary/*, no_tta_test_primary/*,
      mirror_tta_full_val_primary/*, mirror_tta_test_primary/*.
    Also logs full_val_primary/* and test_primary/* mirroring config.mirror_tta
    so the standard contract is met.
    """
    checkpoint_path = resolve_checkpoint_path(config.eval_checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned_key = key.removeprefix("module.").removeprefix("_orig_mod.")
        cleaned[cleaned_key] = value

    if "pos_embed.center" in cleaned and "pos_embed.proj.project.weight" in cleaned:
        hidden_dim = int(cleaned["pos_embed.proj.project.weight"].shape[0])
        raw_dim = int(cleaned["pos_embed.proj.project.weight"].shape[1])
        num_freqs = int(cleaned["pos_embed.freqs"].numel())
        input_dim = raw_dim // (num_freqs * 2)
        legacy_pe = _AlphonseFourierEmbed(
            hidden_dim=hidden_dim, input_dim=input_dim, num_freqs=num_freqs
        )
        model.pos_embed = legacy_pe.to(device)
        print(
            "Detected legacy alphonse FourierEmbed in checkpoint; swapped pos_embed "
            f"(hidden={hidden_dim}, input={input_dim}, num_freqs={num_freqs})."
        )

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  missing example: {missing[:3]}")
        if unexpected:
            print(f"  unexpected example: {unexpected[:3]}")
    saved_epoch = checkpoint.get("epoch")
    saved_source = checkpoint.get("checkpoint_source", "raw")
    print(f"Checkpoint epoch={saved_epoch} source={saved_source}")
    wandb.summary.update(
        {
            "eval_only/checkpoint_path": str(checkpoint_path),
            "eval_only/checkpoint_epoch": int(saved_epoch) if saved_epoch is not None else -1,
            "eval_only/checkpoint_source": saved_source,
            "n_params": n_params,
        }
    )

    no_tta_val = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode, mirror_tta=False)
        for name, loader in final_val_loaders.items()
    }
    mirror_tta_val = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode, mirror_tta=True)
        for name, loader in final_val_loaders.items()
    }
    no_tta_test = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode, mirror_tta=False)
        for name, loader in final_test_loaders.items()
    }
    mirror_tta_test = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode, mirror_tta=True)
        for name, loader in final_test_loaders.items()
    }

    log: dict[str, object] = {"global_step": 0}
    log.update(primary_metric_log("no_tta_full_val_primary", no_tta_val["val_surface"]))
    log.update(primary_metric_log("mirror_tta_full_val_primary", mirror_tta_val["val_surface"]))
    log.update(primary_metric_log("no_tta_test_primary", no_tta_test["test_surface"]))
    log.update(primary_metric_log("mirror_tta_test_primary", mirror_tta_test["test_surface"]))
    for split_name, metrics in no_tta_val.items():
        log.update(metric_namespace("no_tta_full_val", split_name, metrics))
    for split_name, metrics in mirror_tta_val.items():
        log.update(metric_namespace("mirror_tta_full_val", split_name, metrics))
    for split_name, metrics in no_tta_test.items():
        log.update(metric_namespace("no_tta_test", split_name, metrics))
    for split_name, metrics in mirror_tta_test.items():
        log.update(metric_namespace("mirror_tta_test", split_name, metrics))

    chosen_val = mirror_tta_val if config.mirror_tta else no_tta_val
    chosen_test = mirror_tta_test if config.mirror_tta else no_tta_test
    log.update(primary_metric_log("full_val_primary", chosen_val["val_surface"]))
    log.update(primary_metric_log("test_primary", chosen_test["test_surface"]))
    for split_name, metrics in chosen_val.items():
        log.update(metric_namespace("full_val", split_name, metrics))
    for split_name, metrics in chosen_test.items():
        log.update(metric_namespace("test", split_name, metrics))

    print_metrics("no_tta_full_val", no_tta_val["val_surface"])
    print_metrics("mirror_tta_full_val", mirror_tta_val["val_surface"])
    print_metrics("no_tta_test", no_tta_test["test_surface"])
    print_metrics("mirror_tta_test", mirror_tta_test["test_surface"])

    delta_val = (
        no_tta_val["val_surface"]["wall_shear_y_rel_l2_pct"]
        - mirror_tta_val["val_surface"]["wall_shear_y_rel_l2_pct"]
    )
    delta_val_abupt = (
        no_tta_val["val_surface"]["abupt_axis_mean_rel_l2_pct"]
        - mirror_tta_val["val_surface"]["abupt_axis_mean_rel_l2_pct"]
    )
    print(f"\nMirror-TTA val deltas (no_tta - mirror_tta, lower is better):")
    print(f"  wall_shear_y_rel_l2_pct: {delta_val:.4f}")
    print(f"  abupt_axis_mean_rel_l2_pct: {delta_val_abupt:.4f}")

    try:
        assert_required_finite_metrics(log, "full_val_primary")
        assert_required_finite_metrics(log, "test_primary")
    except RuntimeError as exc:
        wandb.summary.update({"run_invalid": 1.0, "run_invalid/reason": str(exc)})
        wandb.finish()
        raise
    wandb.log(log)
    wandb.summary.update(numeric_metric_items(log))


def run_final_evaluation(
    *,
    run,
    model: nn.Module,
    model_path: Path,
    config_path: Path,
    config,
    transform: TargetTransform,
    device: torch.device,
    final_val_loaders: dict[str, DataLoader],
    final_test_loaders: dict[str, DataLoader],
    best_metrics: dict[str, float],
    best_checkpoint_source: str,
    n_params: int,
    global_step: int,
    total_minutes: float,
) -> None:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    print(
        f"Best val: epoch {int(best_metrics['epoch'])}, "
        f"abupt_axis_mean_rel_l2_pct={best_metrics['abupt_axis_mean_rel_l2_pct']:.4f}"
    )
    wandb.summary.update(
        {
            "best_epoch": int(best_metrics["epoch"]),
            "best_checkpoint/source": best_checkpoint_source,
            "best_checkpoint/selection_metric": "val_primary/abupt_axis_mean_rel_l2_pct",
            "best_val_primary/abupt_axis_mean_rel_l2_pct": best_metrics["abupt_axis_mean_rel_l2_pct"],
            "best_val/surface_pressure_mae": best_metrics["surface_pressure_mae"],
            "best_val/wall_shear_mae": best_metrics["wall_shear_mae"],
            "best_val/volume_pressure_mae": best_metrics["volume_pressure_mae"],
            "total_train_minutes": total_minutes,
        }
    )

    mirror_tta = bool(getattr(config, "mirror_tta", False))
    full_val_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode, mirror_tta=mirror_tta)
        for name, loader in final_val_loaders.items()
    }
    full_val_log: dict[str, object] = {
        "global_step": global_step,
        **primary_metric_log("full_val_primary", full_val_metrics["val_surface"]),
    }
    for split_name, metrics in full_val_metrics.items():
        full_val_log.update(metric_namespace("full_val", split_name, metrics))
    if mirror_tta:
        no_tta_full_val_metrics = {
            name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode, mirror_tta=False)
            for name, loader in final_val_loaders.items()
        }
        full_val_log.update(
            primary_metric_log("no_tta_full_val_primary", no_tta_full_val_metrics["val_surface"])
        )
        for split_name, metrics in no_tta_full_val_metrics.items():
            full_val_log.update(metric_namespace("no_tta_full_val", split_name, metrics))
    try:
        assert_required_finite_metrics(full_val_log, "full_val_primary")
    except RuntimeError as exc:
        wandb.summary.update({"run_invalid": 1.0, "run_invalid/reason": str(exc)})
        wandb.finish()
        raise
    wandb.log(full_val_log)
    wandb.summary.update(numeric_metric_items(full_val_log))
    print_metrics("full_val", full_val_metrics["val_surface"])
    if mirror_tta:
        print_metrics("no_tta_full_val", no_tta_full_val_metrics["val_surface"])

    test_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode, mirror_tta=mirror_tta)
        for name, loader in final_test_loaders.items()
    }
    test_log: dict[str, object] = {
        "global_step": global_step,
        **primary_metric_log("test_primary", test_metrics["test_surface"]),
    }
    for split_name, metrics in test_metrics.items():
        test_log.update(metric_namespace("test", split_name, metrics))
    if mirror_tta:
        no_tta_test_metrics = {
            name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode, mirror_tta=False)
            for name, loader in final_test_loaders.items()
        }
        test_log.update(
            primary_metric_log("no_tta_test_primary", no_tta_test_metrics["test_surface"])
        )
        for split_name, metrics in no_tta_test_metrics.items():
            test_log.update(metric_namespace("no_tta_test", split_name, metrics))
    try:
        assert_required_finite_metrics(test_log, "test_primary")
    except RuntimeError as exc:
        wandb.summary.update({"run_invalid": 1.0, "run_invalid/reason": str(exc)})
        wandb.finish()
        raise
    wandb.log(test_log)
    wandb.summary.update(numeric_metric_items(test_log))
    print_metrics("test_surface", test_metrics["test_surface"])
    if mirror_tta:
        print_metrics("no_tta_test", no_tta_test_metrics["test_surface"])

    log_model_artifact(
        run=run,
        model_path=model_path,
        config_path=config_path,
        config=config,
        best_metrics=best_metrics,
        test_metrics=test_metrics,
        n_params=n_params,
    )
