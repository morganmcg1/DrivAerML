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
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import SurfaceTransolver
from trainer_runtime import (
    EMA,
    DistributedState,
    MetricSlopeTracker,
    TargetTransform,
    autocast_context,
    build_lr_scheduler,
    check_kill_thresholds,
    cleanup_distributed,
    collect_gradient_metrics,
    collect_weight_metrics,
    define_wandb_metrics,
    distributed_any,
    distributed_barrier,
    evaluate_split,
    full_eval_loaders_from,
    global_grad_norm,
    init_distributed,
    init_wandb_run,
    is_valid_primary_metric,
    make_loaders,
    masked_mse,
    metric_namespace,
    parse_kill_thresholds,
    primary_metric_log,
    print_metrics,
    run_final_evaluation,
    should_update_best_checkpoint,
    timeout_budget_minutes,
    unwrap_model,
)


TASK_NAMES = (
    "surface_pressure",
    "wall_shear_x",
    "wall_shear_y",
    "wall_shear_z",
    "volume_pressure",
)


class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic uncertainty multi-task weighting (Kendall & Gal 2018).

    Holds a learnable log_var per task and combines per-task losses as
        L = sum_i [ 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i ].
    log_vars are clamped to [log_var_min, log_var_max] for numerical stability.
    """

    def __init__(
        self,
        n_tasks: int = 5,
        init_log_var: float = -0.5,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
    ):
        super().__init__()
        self.log_vars = nn.Parameter(torch.full((n_tasks,), float(init_log_var)))
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

    def clamped_log_vars(self) -> torch.Tensor:
        return self.log_vars.clamp(self.log_var_min, self.log_var_max)

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        clamped = self.clamped_log_vars()
        precisions = torch.exp(-clamped)
        stacked = torch.stack(losses)
        return (0.5 * precisions * stacked + 0.5 * clamped).sum()


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
    validation_every: int = 1
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
    amp_mode: str = "bf16"
    num_workers: int = -1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_step: int = 50
    eval_raw_vs_ema: bool = False
    lr_warmup_epochs: int = 0
    lr_cosine_t_max: int = 0
    lr_min: float = 1e-6
    grad_clip_norm: float = 1.0
    gradient_log_every: int = 250
    log_gradient_histograms: bool = False
    weight_log_every: int = 250
    log_weight_histograms: bool = False
    ddp_log_model_telemetry_all_ranks: bool = False
    slope_log_fraction: float = 0.05
    kill_thresholds: str = ""
    compile_model: bool = True
    debug: bool = False
    uncertainty_weighting: bool = False
    uncertainty_init_log_var: float = 0.0
    uncertainty_log_var_min: float = -5.0
    uncertainty_log_var_max: float = 5.0
    uncertainty_lr_scale: float = 1.0
    eval_only: bool = False
    eval_only_run_id: str = ""


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


def build_model(config: Config) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=config.model_layers,
        n_hidden=config.model_hidden_dim,
        dropout=config.model_dropout,
        n_head=config.model_heads,
        mlp_ratio=config.model_mlp_ratio,
        slice_num=config.model_slices,
    )


def train_loss(
    model: nn.Module,
    batch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    *,
    surface_loss_weight: float = 1.0,
    volume_loss_weight: float = 1.0,
    uncertainty_loss: UncertaintyWeightedLoss | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
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
        surface_pred = out["surface_preds"]
        volume_pred = out["volume_preds"]
        per_task_losses = [
            masked_mse(surface_pred[..., 0:1], surface_target[..., 0:1], batch.surface_mask),
            masked_mse(surface_pred[..., 1:2], surface_target[..., 1:2], batch.surface_mask),
            masked_mse(surface_pred[..., 2:3], surface_target[..., 2:3], batch.surface_mask),
            masked_mse(surface_pred[..., 3:4], surface_target[..., 3:4], batch.surface_mask),
            masked_mse(volume_pred, volume_target, batch.volume_mask),
        ]
        surface_loss = sum(per_task_losses[:4]) / 4.0
        volume_loss = per_task_losses[4]
        if uncertainty_loss is not None:
            loss = uncertainty_loss(per_task_losses)
            weighted_surface_loss = loss.detach() * 0.0
            weighted_volume_loss = loss.detach() * 0.0
        else:
            weighted_surface_loss = surface_loss_weight * surface_loss
            weighted_volume_loss = volume_loss_weight * volume_loss
            loss = weighted_surface_loss + weighted_volume_loss
        base_mse_loss = surface_loss + volume_loss
    metrics: dict[str, object] = {
        "base_mse_loss": float(base_mse_loss.detach().cpu().item()),
        "surface_loss": float(surface_loss.detach().cpu().item()),
        "volume_loss": float(volume_loss.detach().cpu().item()),
        "surface_loss_weighted": float(weighted_surface_loss.detach().cpu().item()),
        "volume_loss_weighted": float(weighted_volume_loss.detach().cpu().item()),
        "per_task_losses": [float(l.detach().cpu().item()) for l in per_task_losses],
    }
    if uncertainty_loss is not None:
        clamped = uncertainty_loss.clamped_log_vars().detach()
        precisions = torch.exp(-clamped)
        effective_weights = 0.5 * precisions
        metrics["log_vars"] = clamped.cpu().tolist()
        metrics["effective_weights"] = effective_weights.cpu().tolist()
    return loss, metrics


def run_eval_only(config: Config, state: DistributedState) -> None:
    """Load a saved best checkpoint and emit full_val/test_primary metrics.

    Resumes the original W&B run so post-hoc test metrics land on the same run
    as the training that produced the checkpoint.
    """
    if state.enabled:
        raise RuntimeError(
            "--eval-only must be launched as a single process (not via torchrun)."
        )
    if not config.eval_only_run_id:
        raise ValueError("--eval-only requires --eval-only-run-id <wandb_run_id>")

    device = state.device
    output_dir = Path(config.output_dir) / f"run-{config.eval_only_run_id}"
    model_path = output_dir / "checkpoint.pt"
    config_path = output_dir / "config.yaml"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    train_loader, val_loaders, test_loaders, stats = make_loaders(
        config, distributed_state=None
    )
    final_val_loaders = full_eval_loaders_from(val_loaders, config)
    final_test_loaders = full_eval_loaders_from(test_loaders, config)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Eval-only: built SurfaceTransolver ({n_params / 1e6:.2f}M params) on {device}")

    tags = [config.agent, "eval-only"] if config.agent else ["eval-only"]
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        id=config.eval_only_run_id,
        resume="allow",
        group=config.wandb_group or None,
        name=config.wandb_name or None,
        tags=tags,
        config={
            **asdict(config),
            "n_params": n_params,
            "eval_only": True,
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    define_wandb_metrics()

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    best_val_surface = ckpt["val_metrics"]["val_surface"]
    best_metrics = {"epoch": float(ckpt["epoch"]), **best_val_surface}
    best_checkpoint_source = ckpt.get("checkpoint_source", "raw")
    print(
        f"Loaded checkpoint epoch={ckpt['epoch']}, "
        f"val_primary/abupt_axis_mean_rel_l2_pct={best_val_surface['abupt_axis_mean_rel_l2_pct']:.4f}"
    )

    run_final_evaluation(
        run=run,
        model=model,
        model_path=model_path,
        config_path=config_path,
        config=config,
        transform=transform,
        device=device,
        final_val_loaders=final_val_loaders,
        final_test_loaders=final_test_loaders,
        best_metrics=best_metrics,
        best_checkpoint_source=best_checkpoint_source,
        n_params=n_params,
        global_step=0,
        total_minutes=0.0,
    )
    wandb.finish()


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    run = None
    try:
        config = parse_args(argv)
        if config.eval_only:
            run_eval_only(config, state)
            return
        kill_thresholds = parse_kill_thresholds(config.kill_thresholds)
        requested_epochs = config.epochs
        if os.environ.get("SENPAI_MAX_EPOCHS"):
            requested_epochs = min(requested_epochs, int(os.environ["SENPAI_MAX_EPOCHS"]))
        max_epochs = min(requested_epochs, 3) if config.debug else requested_epochs
        timeout_minutes, val_budget_minutes, train_timeout_minutes = timeout_budget_minutes()
        device = state.device
        if state.is_main:
            ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
            print(f"Device: {device}{ddp_suffix}" + (" [DEBUG]" if config.debug else ""))

        train_loader, val_loaders, test_loaders, stats = make_loaders(config, distributed_state=state)
        final_val_loaders = full_eval_loaders_from(val_loaders, config) if state.is_main else {}
        final_test_loaders = full_eval_loaders_from(test_loaders, config) if state.is_main else {}
        transform = TargetTransform(
            surface_y_mean=stats["surface_y_mean"].to(device),
            surface_y_std=stats["surface_y_std"].to(device),
            volume_y_mean=stats["volume_y_mean"].to(device),
            volume_y_std=stats["volume_y_std"].to(device),
        )

        model: nn.Module = build_model(config).to(device)
        n_params = sum(param.numel() for param in model.parameters())
        if config.compile_model:
            model = torch.compile(model)
        if state.enabled:
            ddp_kwargs = {}
            if device.type == "cuda":
                ddp_kwargs = {"device_ids": [state.local_rank], "output_device": state.local_rank}
            model = DistributedDataParallel(model, **ddp_kwargs)
        base_model = unwrap_model(model)
        if state.is_main:
            print(f"Model: SurfaceTransolver grouped surface+volume ({n_params / 1e6:.2f}M params)")

        uncertainty_loss: UncertaintyWeightedLoss | None = None
        if config.uncertainty_weighting:
            uncertainty_loss = UncertaintyWeightedLoss(
                n_tasks=len(TASK_NAMES),
                init_log_var=config.uncertainty_init_log_var,
                log_var_min=config.uncertainty_log_var_min,
                log_var_max=config.uncertainty_log_var_max,
            ).to(device)
            if state.is_main:
                print(
                    f"UncertaintyWeightedLoss enabled: {len(TASK_NAMES)} tasks, "
                    f"init log_var={config.uncertainty_init_log_var}"
                )

        optimizer_param_groups: list[dict[str, object]] = [
            {"params": list(base_model.parameters()), "weight_decay": config.weight_decay},
        ]
        if uncertainty_loss is not None:
            optimizer_param_groups.append(
                {
                    "params": list(uncertainty_loss.parameters()),
                    "weight_decay": 0.0,
                    "lr": config.lr * config.uncertainty_lr_scale,
                }
            )
        optimizer = torch.optim.AdamW(optimizer_param_groups, lr=config.lr, weight_decay=config.weight_decay)
        scheduler = build_lr_scheduler(optimizer, config, max_epochs)
        ema = EMA(base_model, decay=config.ema_decay, start_step=config.ema_start_step) if config.use_ema else None
        total_estimated_steps = max(1, max_epochs * max(len(train_loader), 1))
        if kill_thresholds and state.is_main:
            print("Kill thresholds:", "; ".join(threshold.describe() for threshold in kill_thresholds))
        train_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)
        val_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)

        run = init_wandb_run(
            config=config,
            state=state,
            n_params=n_params,
            train_loader=train_loader,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            total_estimated_steps=total_estimated_steps,
            max_epochs=max_epochs,
            train_timeout_minutes=train_timeout_minutes,
            val_budget_minutes=val_budget_minutes,
        )

        output_dir = Path(config.output_dir) / f"run-{run.id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "checkpoint.pt"
        config_path = output_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(asdict(config), f)

        best_val = float("inf")
        best_metrics: dict[str, float] = {}
        best_checkpoint_source = "ema" if ema is not None else "raw"
        global_step = 0
        early_stop_reason: str | None = None
        timeout_hit = False
        train_start = time.time()

        for epoch in range(max_epochs):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            timeout_hit = distributed_any(
                state,
                (time.time() - train_start) / 60.0 >= timeout_minutes,
                device,
            )
            if timeout_hit:
                if state.is_main:
                    print(f"Timeout ({timeout_minutes:.1f} min). Stopping.")
                break

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            model.train()
            train_loss_sum = 0.0
            n_batches = 0

            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{max_epochs}",
                leave=False,
                disable=not state.is_main,
            ):
                loss, batch_loss_metrics = train_loss(
                    model,
                    batch,
                    transform,
                    device,
                    config.amp_mode,
                    surface_loss_weight=config.surface_loss_weight,
                    volume_loss_weight=config.volume_loss_weight,
                    uncertainty_loss=uncertainty_loss,
                )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                loss_is_nonfinite = not bool(torch.isfinite(loss.detach()).item())
                skip_step = distributed_any(state, loss_is_nonfinite, device)
                should_log_model_telemetry = (
                    state.is_main or config.ddp_log_model_telemetry_all_ranks
                )
                should_log_gradients = (
                    should_log_model_telemetry
                    and config.gradient_log_every > 0
                    and global_step % config.gradient_log_every == 0
                )
                should_log_weights = (
                    should_log_model_telemetry
                    and config.weight_log_every > 0
                    and global_step % config.weight_log_every == 0
                )
                train_log: dict[str, object] = {
                    "global_step": global_step,
                    "train/lr": current_lr,
                    "lr": current_lr,
                    "train/step_skipped": 1.0 if skip_step else 0.0,
                    "train/nonfinite_loss": 1.0 if loss_is_nonfinite else 0.0,
                }
                if not loss_is_nonfinite:
                    train_log.update(
                        {
                            "train/loss": float(loss.detach().cpu().item()),
                            "train/base_mse_loss": batch_loss_metrics["base_mse_loss"],
                            "train/surface_loss": batch_loss_metrics["surface_loss"],
                            "train/volume_loss": batch_loss_metrics["volume_loss"],
                            "train/surface_loss_weighted": batch_loss_metrics["surface_loss_weighted"],
                            "train/volume_loss_weighted": batch_loss_metrics["volume_loss_weighted"],
                        }
                    )
                    per_task = batch_loss_metrics.get("per_task_losses")
                    if per_task is not None:
                        for name, value in zip(TASK_NAMES, per_task):
                            train_log[f"train/per_task_loss/{name}"] = value
                    log_var_values = batch_loss_metrics.get("log_vars")
                    if log_var_values is not None:
                        effective_weights = batch_loss_metrics.get("effective_weights") or []
                        for name, value in zip(TASK_NAMES, log_var_values):
                            train_log[f"train/log_var/{name}"] = value
                            train_log[f"train/sigma_sq/{name}"] = math.exp(float(value))
                        for name, value in zip(TASK_NAMES, effective_weights):
                            train_log[f"train/effective_weight/{name}"] = value

                if skip_step:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if (
                        uncertainty_loss is not None
                        and state.enabled
                        and uncertainty_loss.log_vars.grad is not None
                    ):
                        # DDP only all-reduces grads on the wrapped model; log_vars
                        # live outside DDP so we average them manually to keep ranks
                        # in lockstep.
                        torch.distributed.all_reduce(
                            uncertainty_loss.log_vars.grad,
                            op=torch.distributed.ReduceOp.AVG,
                        )
                    if config.grad_clip_norm > 0.0:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            base_model.parameters(),
                            max_norm=config.grad_clip_norm,
                        )
                    else:
                        grad_norm_tensor = global_grad_norm(base_model.parameters(), device)
                    grad_norm_pre_clip = float(grad_norm_tensor.detach().cpu().item())
                    grad_is_nonfinite = not math.isfinite(grad_norm_pre_clip)
                    skip_step = distributed_any(state, grad_is_nonfinite, device)
                    clipped = (
                        config.grad_clip_norm > 0.0
                        and math.isfinite(grad_norm_pre_clip)
                        and grad_norm_pre_clip > config.grad_clip_norm
                    )
                    train_log.update(
                        {
                            "train/grad/global_norm_pre_clip": grad_norm_pre_clip,
                            "train/grad/clipped": 1.0 if clipped else 0.0,
                            "train/nonfinite_grad": 1.0 if grad_is_nonfinite else 0.0,
                            "train/step_skipped": 1.0 if skip_step else 0.0,
                        }
                    )
                    gradient_metrics = (
                        collect_gradient_metrics(
                            model,
                            log_histograms=config.log_gradient_histograms,
                        )
                        if should_log_gradients and not skip_step
                        else {}
                    )
                    if skip_step:
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.step()
                        if ema is not None:
                            ema.update(base_model)
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
                        train_log.update(gradient_metrics)
                        train_log.update(weight_metrics)

                train_log.update(
                    train_slope_tracker.update(
                        global_step=global_step,
                        metrics=train_log,
                        namespace="train",
                    )
                )
                local_stop_reason = check_kill_thresholds(
                    global_step=global_step,
                    metrics=train_log,
                    thresholds=kill_thresholds,
                )
                if local_stop_reason is not None:
                    early_stop_reason = local_stop_reason
                stop_requested = distributed_any(state, early_stop_reason is not None, device)
                if stop_requested and early_stop_reason is None:
                    early_stop_reason = "distributed peer requested early stop"
                if early_stop_reason is not None:
                    train_log["early_stop/triggered"] = 1.0
                wandb.log(train_log)
                if early_stop_reason is not None:
                    if state.is_main:
                        print(early_stop_reason)
                    break
                timeout_hit = distributed_any(
                    state,
                    (time.time() - train_start) / 60.0 >= train_timeout_minutes,
                    device,
                )
                if timeout_hit:
                    if state.is_main:
                        print(
                            f"Train timeout ({train_timeout_minutes:.1f} min) mid-epoch "
                            f"at step {global_step}. Forcing validation and stopping."
                        )
                    break

            scheduler.step()
            epoch_train_loss = train_loss_sum / max(n_batches, 1)
            dt = time.time() - t0
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
            should_validate = (
                epoch == 0
                or (epoch + 1) % max(config.validation_every, 1) == 0
                or epoch + 1 == max_epochs
                or (timeout_hit and n_batches > 0)
            )

            log_metrics: dict[str, object] = {
                "train/epoch_loss": epoch_train_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "lr": scheduler.get_last_lr()[0],
                "epoch_time_s": dt,
                "global_step": global_step,
            }
            if early_stop_reason is not None:
                log_metrics["early_stop/triggered"] = 1.0
                wandb.log(log_metrics)
                break

            if not should_validate:
                local_stop_reason = check_kill_thresholds(
                    global_step=global_step,
                    metrics=log_metrics,
                    thresholds=kill_thresholds,
                )
                if local_stop_reason is not None:
                    early_stop_reason = local_stop_reason
                stop_requested = distributed_any(state, early_stop_reason is not None, device)
                if stop_requested and early_stop_reason is None:
                    early_stop_reason = "distributed peer requested early stop"
                if early_stop_reason is not None:
                    log_metrics["early_stop/triggered"] = 1.0
                wandb.log(log_metrics)
                if state.is_main:
                    print(
                        f"Epoch {epoch + 1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB] "
                        f"train_loss={epoch_train_loss:.5f}"
                    )
                if early_stop_reason is not None:
                    if state.is_main:
                        print(early_stop_reason)
                    break
                continue

            raw_val_metrics = None
            if config.eval_raw_vs_ema and ema is not None:
                raw_val_metrics = {
                    name: evaluate_split(
                        model,
                        loader,
                        transform,
                        device,
                        amp_mode=config.amp_mode,
                        distributed_state=state,
                    )
                    for name, loader in val_loaders.items()
                }
            if ema is not None:
                ema.store(base_model)
                ema.copy_to(base_model)
            val_metrics = {
                name: evaluate_split(
                    model,
                    loader,
                    transform,
                    device,
                    amp_mode=config.amp_mode,
                    distributed_state=state,
                )
                for name, loader in val_loaders.items()
            }

            if state.is_main:
                if raw_val_metrics is not None:
                    raw_surface = raw_val_metrics["val_surface"]
                    log_metrics.update(primary_metric_log("val_raw_primary", raw_surface))
                    for split_name, metrics in raw_val_metrics.items():
                        log_metrics.update(metric_namespace("val_raw", split_name, metrics))
                primary_val = val_metrics["val_surface"]["abupt_axis_mean_rel_l2_pct"]
                log_metrics.update(primary_metric_log("val_primary", val_metrics["val_surface"]))
                for split_name, metrics in val_metrics.items():
                    log_metrics.update(metric_namespace("val", split_name, metrics))
                log_metrics.update(
                    val_slope_tracker.update(
                        global_step=global_step,
                        metrics=log_metrics,
                        namespace="val",
                        force=True,
                    )
                )
                local_stop_reason = check_kill_thresholds(
                    global_step=global_step,
                    metrics=log_metrics,
                    thresholds=kill_thresholds,
                )
                if local_stop_reason is not None:
                    early_stop_reason = local_stop_reason
                if early_stop_reason is not None:
                    log_metrics["early_stop/triggered"] = 1.0
                improved = should_update_best_checkpoint(primary_val, best_val)
                if improved:
                    best_val = primary_val
                    best_metrics = {"epoch": float(epoch + 1), **val_metrics["val_surface"]}
                    torch.save(
                        {
                            "model": base_model.state_dict(),
                            "config": asdict(config),
                            "epoch": epoch + 1,
                            "val_metrics": val_metrics,
                            "checkpoint_source": best_checkpoint_source,
                            "selection_metric": "val_primary/abupt_axis_mean_rel_l2_pct",
                        },
                        model_path,
                    )
                log_metrics["best_checkpoint/updated"] = 1.0 if improved else 0.0
                log_metrics["best_checkpoint/valid_primary"] = 1.0 if is_valid_primary_metric(primary_val) else 0.0
                wandb.log(log_metrics)
                tag = " *" if improved else ""
                print(
                    f"Epoch {epoch + 1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB] "
                    f"train_loss={epoch_train_loss:.5f} "
                    f"val_abupt_axis_rel_l2_pct={primary_val:.4f}{tag}"
                )
                print_metrics("val_surface", val_metrics["val_surface"])
            else:
                wandb.log(log_metrics)
            if ema is not None:
                ema.restore(base_model)
            stop_requested = distributed_any(state, early_stop_reason is not None, device)
            if stop_requested and early_stop_reason is None:
                early_stop_reason = "rank 0 requested early stop"
            if early_stop_reason is not None:
                if state.is_main:
                    print(early_stop_reason)
                break
            if timeout_hit:
                break

        total_minutes = (time.time() - train_start) / 60.0
        if state.is_main:
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

        distributed_barrier(state)
        if not state.is_main:
            wandb.summary.update({"total_train_minutes": total_minutes})
            wandb.finish()
            return

        if not best_metrics:
            wandb.summary.update(
                {
                    "run_invalid": 1.0,
                    "run_invalid/reason": "no finite positive validation checkpoint was saved",
                    "total_train_minutes": total_minutes,
                }
            )
            print("No finite positive validation checkpoint was saved.")
            wandb.finish()
            return

        run_final_evaluation(
            run=run,
            model=base_model,
            model_path=model_path,
            config_path=config_path,
            config=config,
            transform=transform,
            device=device,
            final_val_loaders=final_val_loaders,
            final_test_loaders=final_test_loaders,
            best_metrics=best_metrics,
            best_checkpoint_source=best_checkpoint_source,
            n_params=n_params,
            global_step=global_step,
            total_minutes=total_minutes,
        )
        wandb.finish()
    finally:
        cleanup_distributed(state)


if __name__ == "__main__":
    main()
