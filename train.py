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
import gc
import hashlib
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import (
    SURFACE_Y_DIM,
    VOLUME_Y_DIM,
    DrivAerMLCase,
    DrivAerMLCaseStore,
    DrivAerMLSurfaceDataset,
    SurfaceBatch,
    pad_collate,
)
from model import LinearProjection, SurfaceTransolver
from trainer_runtime import (
    EMA,
    MetricSlopeTracker,
    TargetTransform,
    autocast_context,
    build_lr_scheduler,
    check_kill_thresholds,
    cleanup_distributed,
    collect_gradient_metrics,
    collect_weight_metrics,
    distributed_any,
    distributed_barrier,
    evaluate_split,
    full_eval_loaders_from,
    global_grad_norm,
    init_distributed,
    init_wandb_run,
    is_valid_primary_metric,
    loader_kwargs,
    make_loaders,
    masked_mse,
    metric_namespace,
    weighted_channel_mse,
    parse_kill_thresholds,
    primary_metric_log,
    print_metrics,
    run_final_evaluation,
    should_update_best_checkpoint,
    timeout_budget_minutes,
    unwrap_model,
)


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
    rff_num_features: int = 0
    rff_sigma: float = 1.0
    rff_init_sigmas: str = ""
    pos_encoding_mode: str = "sincos"
    use_qk_norm: bool = False
    use_surf_to_vol_xattn: bool = False
    tau_y_loss_weight: float = 1.0
    tau_z_loss_weight: float = 1.0
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
    optimizer: str = "adamw"
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    vol_points_schedule: str = ""
    use_gradnorm: bool = False
    gradnorm_mode: str = "ema_proxy"
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 1e-3
    gradnorm_init_warmup_steps: int = 10
    gradnorm_log_clip: float = 4.0
    gradnorm_ema_beta: float = 0.9
    gradnorm_min_weight: float = 0.0
    distill_teacher_path: str = ""
    distill_lambda_kd: float = 0.5
    distill_teacher_runs: str = ""
    distill_build_cache_only: bool = False
    distill_cache_eval_surface_points: int = 65_536
    distill_cache_eval_volume_points: int = 65_536
    distill_cache_chunk_batch_size: int = 4
    debug: bool = False


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
        "vol_points_schedule": (
            "Optional epoch-based curriculum for the train-volume-points view "
            "size. Format: 'EPOCH:POINTS:EPOCH:POINTS:...' (colon-separated). "
            "The train DataLoader is rebuilt at any epoch where the value "
            "changes. Must start at epoch 0; the value applies from that "
            "epoch onwards (inclusive) until the next breakpoint. Empty "
            "string disables the curriculum and `--train-volume-points` is "
            "used unchanged. Example: '0:16384:3:32768:6:49152:9:65536'."
        ),
        "use_gradnorm": (
            "Enable GradNorm dynamic per-task loss balancing (Chen et al., "
            "NeurIPS 2018, https://arxiv.org/abs/1711.02132). Splits the "
            "surface MSE into 4 per-channel losses (sp, tau_x, tau_y, tau_z) "
            "and treats volume_pressure as the 5th task; per-task weights "
            "are updated each step (algorithm controlled by --gradnorm-mode). "
            "When disabled, the legacy --surface-loss-weight / "
            "--volume-loss-weight scalars apply."
        ),
        "gradnorm_mode": (
            "GradNorm balancing algorithm. 'full' is the paper-faithful "
            "variant: per-task gradient norms at the last shared encoder "
            "block via autograd.grad with retain_graph (5x backward overhead "
            "for 5 tasks). 'ema_proxy' is the lightweight loss-magnitude "
            "approximation: maintains an EMA of per-task losses, computes "
            "relative training rate r_i = ema_i / initial_i, and assigns "
            "weights w_i proportional to r_i**alpha (renormalised so that "
            "mean weight = 1). Trades the formal gradient-alignment "
            "guarantee for ~1x overhead. Default: ema_proxy."
        ),
        "gradnorm_alpha": (
            "Restoring-force exponent alpha. Higher values pull "
            "slow-converging tasks back harder. Paper-recommended range "
            "[0.5, 2.0]; default 1.5 for tasks with different rates. "
            "Applied to relative training rate r_i in both modes."
        ),
        "gradnorm_lr": (
            "Learning rate for the Adam optimizer that updates the per-task "
            "log-weights in mode='full'. Unused in mode='ema_proxy' (weights "
            "are computed in closed form). The brianlan reference uses "
            "~2.5e-2; we default to 1e-3 for slower / stabler movement."
        ),
        "gradnorm_init_warmup_steps": (
            "Number of training steps over which to average the per-task "
            "losses to define L_i(0). Skips step 0 to avoid random-init "
            "spikes; default 10 steps. Applied in both modes."
        ),
        "gradnorm_log_clip": (
            "Symmetric clamp applied to the GradNorm log-weights each step "
            "(weight = exp(log_w) so |log_w|<=clip => weight in "
            "[exp(-clip), exp(clip)]). Prevents runaway divergence and "
            "task collapse. Default 4.0 => weights in [0.018, 54.6]. "
            "Applied in both modes."
        ),
        "gradnorm_ema_beta": (
            "EMA decay for the per-task loss tracker used in "
            "mode='ema_proxy' (ema = beta * ema + (1-beta) * L_i.detach()). "
            "Default 0.9 matches the advisor spec. Unused in mode='full'."
        ),
        "gradnorm_min_weight": (
            "Asymmetric per-task weight floor applied in mode='ema_proxy'. "
            "Each pre-normalisation w_raw_i is clamped to >= min_weight "
            "before the closed-form sum-T renormalisation; the up-weighting "
            "ceiling is left free. Prevents over-aggressive down-weighting "
            "of fast-converging tasks (sp/vp) while still allowing slow "
            "tasks (tau_y/tau_z) to climb. Default 0.0 disables the floor "
            "(matches Run 1 behavior). Recommended 0.7 for soft "
            "redistribution. Unused in mode='full'."
        ),
        "use_surf_to_vol_xattn": (
            "Enable surface->volume cross-attention (PR #823). After the "
            "shared Transolver backbone, a single nn.MultiheadAttention "
            "block lets volume hidden states (Q) attend to surface hidden "
            "states (K/V) before the volume output head. The out_proj "
            "weight and bias are zero-initialised so the layer is identity "
            "at init (preserves baseline at epoch 0). embed_dim follows "
            "--model-hidden-dim and num_heads follows --model-heads."
        ),
        "distill_teacher_path": (
            "Directory holding per-case averaged teacher predictions "
            "(PR #1001). When set together with --distill-teacher-runs the "
            "cache is built/refreshed under "
            "<path>/k<K>_<sha>/cases/<case_id>.pt before training; when "
            "set alone (after a prior --distill-build-cache-only run) the "
            "existing cache is loaded read-only and KD loss is enabled."
        ),
        "distill_lambda_kd": (
            "KD mixing weight in L_total = (1 - lambda) * L_hard + lambda * "
            "L_kd. L_hard is the standard masked MSE against ground truth; "
            "L_kd is the masked MSE between student and averaged teacher "
            "predictions evaluated at the SAME sampled point indices. "
            "Range [0, 1]; ignored when --distill-teacher-path is empty."
        ),
        "distill_teacher_runs": (
            "Comma- or semicolon-separated list of W&B run IDs (entity "
            "wandb-applied-ai-team, project senpai-v1-drivaerml-ddp8) whose "
            "'best' model artifact is loaded as a teacher. Predictions are "
            "averaged uniformly across all teachers. Required to build the "
            "cache; ignored once --distill-teacher-path already contains a "
            "cache with the matching K-teacher digest."
        ),
        "distill_build_cache_only": (
            "If True, build the teacher cache under --distill-teacher-path "
            "and exit (no student training). Use this to amortise teacher "
            "inference into a one-time DDP job; subsequent training runs "
            "then reuse the cache without re-running the teachers."
        ),
        "distill_cache_eval_surface_points": (
            "Surface points per teacher-inference chunk during cache build. "
            "Same eval_chunk semantics as --eval-surface-points: each case "
            "is sliced into ceil(N_surface / value) deterministic strided "
            "views which together cover every surface point exactly once."
        ),
        "distill_cache_eval_volume_points": (
            "Volume points per teacher-inference chunk during cache build. "
            "Same eval_chunk semantics as --eval-volume-points."
        ),
        "distill_cache_chunk_batch_size": (
            "Number of eval_chunk views packed into one teacher forward "
            "pass during cache build. Larger values are faster but use "
            "more GPU memory; default 4 fits ~6-7 teachers + chunk batch "
            "of 4 in 80 GB."
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


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "lion":
        from lion_pytorch import Lion

        return Lion(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.lion_beta1, config.lion_beta2),
            use_triton=False,
        )
    raise ValueError(f"Unknown optimizer '{config.optimizer}'. Supported: adamw, lion.")


def parse_rff_init_sigmas(spec: str) -> list[float] | None:
    spec = (spec or "").strip()
    if not spec:
        return None
    sigmas = [float(token.strip()) for token in spec.split(",") if token.strip()]
    if not sigmas:
        return None
    if any(s <= 0.0 for s in sigmas):
        raise ValueError(f"--rff-init-sigmas must be positive: {spec!r}")
    return sigmas


def collect_string_sep_metrics(model: nn.Module) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for attr in ("surface_string_sep", "volume_string_sep"):
        module = getattr(model, attr, None)
        if module is None:
            continue
        log_freq = module.log_freq.detach().float()
        prefix = f"string_sep/{attr}"
        metrics[f"{prefix}/log_freq_mean"] = float(log_freq.mean().item())
        metrics[f"{prefix}/log_freq_std"] = float(log_freq.std().item())
        metrics[f"{prefix}/log_freq_min"] = float(log_freq.min().item())
        metrics[f"{prefix}/log_freq_max"] = float(log_freq.max().item())
        for axis in range(log_freq.shape[0]):
            metrics[f"{prefix}/freq_axis_{axis}_mean"] = float(log_freq[axis].mean().item())
            metrics[f"{prefix}/freq_axis_{axis}_std"] = float(log_freq[axis].std().item())
    return metrics


def build_model(config: Config) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=config.model_layers,
        n_hidden=config.model_hidden_dim,
        dropout=config.model_dropout,
        n_head=config.model_heads,
        mlp_ratio=config.model_mlp_ratio,
        slice_num=config.model_slices,
        rff_num_features=config.rff_num_features,
        rff_sigma=config.rff_sigma,
        rff_init_sigmas=parse_rff_init_sigmas(config.rff_init_sigmas),
        pos_encoding_mode=config.pos_encoding_mode,
        use_qk_norm=config.use_qk_norm,
        use_surf_to_vol_xattn=config.use_surf_to_vol_xattn,
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
    surface_channel_weights: torch.Tensor | None = None,
    surface_teacher: torch.Tensor | None = None,
    volume_teacher: torch.Tensor | None = None,
    lambda_kd: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = batch.to(device)
    surface_target = transform.apply_surface(batch.surface_y)
    volume_target = transform.apply_volume(batch.volume_y)
    use_kd = (
        lambda_kd > 0.0 and surface_teacher is not None and volume_teacher is not None
    )
    if use_kd:
        # Teacher predictions live in the *normalised* output space (matching
        # ``out["surface_preds"]``), so they are compared directly against the
        # student's normalised predictions without applying ``transform``.
        surface_teacher = surface_teacher.to(device, non_blocking=True)
        volume_teacher = volume_teacher.to(device, non_blocking=True)
    with autocast_context(device, amp_mode):
        out = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
        if surface_channel_weights is not None:
            surface_loss = weighted_channel_mse(
                out["surface_preds"],
                surface_target,
                batch.surface_mask,
                surface_channel_weights,
            )
        else:
            surface_loss = masked_mse(out["surface_preds"], surface_target, batch.surface_mask)
        volume_loss = masked_mse(out["volume_preds"], volume_target, batch.volume_mask)
        weighted_surface_loss = surface_loss_weight * surface_loss
        weighted_volume_loss = volume_loss_weight * volume_loss
        label_loss = weighted_surface_loss + weighted_volume_loss
        base_mse_loss = surface_loss + volume_loss
        if use_kd:
            if surface_channel_weights is not None:
                kd_surface_loss = weighted_channel_mse(
                    out["surface_preds"],
                    surface_teacher,
                    batch.surface_mask,
                    surface_channel_weights,
                )
            else:
                kd_surface_loss = masked_mse(
                    out["surface_preds"], surface_teacher, batch.surface_mask
                )
            kd_volume_loss = masked_mse(
                out["volume_preds"], volume_teacher, batch.volume_mask
            )
            kd_weighted_surface = surface_loss_weight * kd_surface_loss
            kd_weighted_volume = volume_loss_weight * kd_volume_loss
            kd_loss = kd_weighted_surface + kd_weighted_volume
            loss = (1.0 - lambda_kd) * label_loss + lambda_kd * kd_loss
        else:
            loss = label_loss
    metrics = {
        "base_mse_loss": float(base_mse_loss.detach().cpu().item()),
        "surface_loss": float(surface_loss.detach().cpu().item()),
        "volume_loss": float(volume_loss.detach().cpu().item()),
        "surface_loss_weighted": float(weighted_surface_loss.detach().cpu().item()),
        "volume_loss_weighted": float(weighted_volume_loss.detach().cpu().item()),
    }
    if use_kd:
        metrics.update(
            {
                "kd_loss": float(kd_loss.detach().cpu().item()),
                "kd_surface_loss": float(kd_surface_loss.detach().cpu().item()),
                "kd_volume_loss": float(kd_volume_loss.detach().cpu().item()),
                "kd_surface_loss_weighted": float(kd_weighted_surface.detach().cpu().item()),
                "kd_volume_loss_weighted": float(kd_weighted_volume.detach().cpu().item()),
                "label_loss_weighted": float(label_loss.detach().cpu().item()),
                "lambda_kd": float(lambda_kd),
            }
        )
    return loss, metrics


GRADNORM_TASK_NAMES = ("sp", "tau_x", "tau_y", "tau_z", "vp")


def per_task_train_losses(
    model: nn.Module,
    batch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
) -> list[torch.Tensor]:
    """Compute the 5 per-axis task losses used by GradNorm.

    Returns scalar tensors in order: [sp, tau_x, tau_y, tau_z, vp].
    All five share the same forward graph so the GradNorm balancer can
    extract per-task gradient norms via ``torch.autograd.grad`` with
    ``retain_graph=True``.
    """

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
        sp_loss = masked_mse(surface_pred[..., 0:1], surface_target[..., 0:1], batch.surface_mask)
        taux_loss = masked_mse(surface_pred[..., 1:2], surface_target[..., 1:2], batch.surface_mask)
        tauy_loss = masked_mse(surface_pred[..., 2:3], surface_target[..., 2:3], batch.surface_mask)
        tauz_loss = masked_mse(surface_pred[..., 3:4], surface_target[..., 3:4], batch.surface_mask)
        vp_loss = masked_mse(out["volume_preds"], volume_target, batch.volume_mask)
    return [sp_loss, taux_loss, tauy_loss, tauz_loss, vp_loss]


GRADNORM_MODES = ("full", "ema_proxy")


class GradNormBalancer:
    """GradNorm dynamic multi-task loss balancing (Chen et al., NeurIPS 2018).

    Two operating modes share the same per-task log-weight state and the
    same warmup mechanism for L_i(0); they differ in how the weights are
    updated each step.

    mode='full' (paper-faithful):
        Maintains a learnable log-weight per task and updates it via Adam
        on the GradNorm L1 loss using gradient norms measured at the last
        shared encoder block. Cost: 5x backward (one autograd.grad per
        task with retain_graph=True) on top of the main backward.

        Notes vs the paper:
          * ``log_weights`` parameterisation; ``task_weights = exp(log_w) *
            T / sum(exp(log_w))`` keeps the mean-1 invariant by construction.
          * Avoids ``create_graph=True``: since ``||w * g||_2 = w * ||g||_2``
            for ``w >= 0``, the GradNorm loss is computed as
            ``W * grad_norm_detached`` and is differentiable in
            ``log_weights`` directly. Equivalent and ~2x cheaper than the
            second-order graph variant.

    mode='ema_proxy' (lightweight loss-magnitude approximation):
        Maintains an exponential moving average of per-task losses and
        sets weights in closed form each step:

            ema_i      = beta * ema_i + (1 - beta) * L_i.detach()
            r_i        = ema_i / initial_loss_i              # relative rate
            r_norm_i   = r_i / mean(r)                       # mean-1 rate
            w_raw_i    = max(r_norm_i ** alpha, min_weight)  # asym floor
            w_i        = w_raw_i * T / sum(w_raw_j)          # sum=T => mean=1

        No autograd over the balancer => ~1x overhead. Tasks whose losses
        decrease slower than average (high r_norm_i) get up-weighted; the
        alpha exponent controls how aggressively. ``log_weights.data`` is
        set from log(w_i) each step for unified logging and DDP broadcast.

        ``min_weight`` (default 0.0) is an asymmetric pre-normalisation
        floor applied to ``w_raw_i``: it caps how aggressively a fast
        task can be down-weighted while leaving the up-weighting ceiling
        free. Recommended 0.7 for soft redistribution.

        Note on the advisor spec:
            The original spec wrote ``w_i = (1/ema_i) / sum(1/ema_j) * T``
            but described the intent as "tasks with larger current loss
            get more weight" (which matches the GradNorm goal of
            up-weighting slow-converging tasks like tau_y/tau_z). The
            literal inverse-loss formula has the opposite sign (it
            DOWN-weights high-loss tasks). We implement the intent: weights
            proportional to r_i**alpha, which is the GradNorm relative
            training rate dressed up as a closed-form computation.

    DDP synchronisation (both modes): per-task losses (and per-task
    gradient norms in 'full' mode) are all-reduced (AVG) so every rank
    runs the same balancer step; ``log_weights`` are broadcast from rank 0
    each step as a belt-and-suspenders guard against numerical drift.

    L_i(0) is captured as the running average of the synced losses
    (mode='full') or the EMA snapshot at warmup completion
    (mode='ema_proxy') over the first ``init_warmup_steps`` steps.
    """

    def __init__(
        self,
        num_tasks: int,
        *,
        mode: str,
        alpha: float,
        lr: float,
        device: torch.device,
        init_warmup_steps: int,
        log_clip: float,
        ema_beta: float,
        min_weight: float = 0.0,
    ):
        if mode not in GRADNORM_MODES:
            raise ValueError(
                f"Unknown gradnorm_mode '{mode}'. Supported: {GRADNORM_MODES}."
            )
        self.mode = mode
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device
        self.init_warmup_steps = max(1, init_warmup_steps)
        self.log_clip = float(log_clip)
        self.ema_beta = float(ema_beta)
        self.min_weight = float(min_weight)
        self.log_weights = nn.Parameter(torch.zeros(num_tasks, device=device))
        if mode == "full":
            self.optimizer: torch.optim.Optimizer | None = torch.optim.Adam(
                [self.log_weights], lr=lr
            )
        else:
            self.optimizer = None
        self._init_loss_sum = torch.zeros(num_tasks, device=device)
        self._init_loss_count = 0
        self.initial_loss: torch.Tensor | None = None
        self.ema_loss: torch.Tensor | None = None
        self.last_floor_active: int = 0

    @property
    def ready(self) -> bool:
        return self.initial_loss is not None

    def task_weights(self) -> torch.Tensor:
        w = self.log_weights.exp()
        return w * (self.num_tasks / w.sum().clamp(min=1e-12))

    def task_weights_detached(self) -> torch.Tensor:
        return self.task_weights().detach()

    def update_initial_loss(self, per_task_losses_synced: torch.Tensor) -> None:
        if self.initial_loss is not None:
            return
        L = per_task_losses_synced.detach().to(self.device)
        if self.mode == "ema_proxy":
            if self.ema_loss is None:
                self.ema_loss = L.clone()
            else:
                self.ema_loss = self.ema_beta * self.ema_loss + (1.0 - self.ema_beta) * L
        self._init_loss_sum = self._init_loss_sum + L
        self._init_loss_count += 1
        if self._init_loss_count >= self.init_warmup_steps:
            if self.mode == "ema_proxy" and self.ema_loss is not None:
                self.initial_loss = self.ema_loss.detach().clamp(min=1e-12).clone()
            else:
                self.initial_loss = (
                    self._init_loss_sum / self._init_loss_count
                ).clamp(min=1e-12)

    def step(
        self,
        per_task_losses_synced: torch.Tensor,
        per_task_grad_norms_synced: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Run one GradNorm update on ``log_weights``.

        Inputs are expected to already be cross-rank synced (mean over
        ranks). In mode='ema_proxy' ``per_task_grad_norms_synced`` is
        ignored. Returns a scalar diagnostic tensor (the L1 GradNorm loss
        in 'full' mode, the per-task r_i sum-deviation in 'ema_proxy'
        mode), or ``None`` if the initial-loss warmup has not finished
        yet.
        """

        if self.initial_loss is None:
            return None
        L = per_task_losses_synced.detach().to(self.device)
        if self.mode == "ema_proxy":
            assert self.ema_loss is not None
            self.ema_loss = (
                self.ema_beta * self.ema_loss + (1.0 - self.ema_beta) * L
            )
            r = (self.ema_loss / self.initial_loss).clamp(min=1e-12)
            r = r / r.mean().clamp(min=1e-12)
            w_raw_pre = r.pow(self.alpha).clamp(min=1e-12)
            if self.min_weight > 0.0:
                self.last_floor_active = int(
                    (w_raw_pre < self.min_weight).sum().item()
                )
                w_raw = w_raw_pre.clamp(min=self.min_weight)
            else:
                self.last_floor_active = 0
                w_raw = w_raw_pre
            w = w_raw * (self.num_tasks / w_raw.sum().clamp(min=1e-12))
            with torch.no_grad():
                self.log_weights.data = w.log().clamp(
                    -self.log_clip, self.log_clip
                )
            return (r - 1.0).abs().sum().detach()

        if per_task_grad_norms_synced is None:
            raise ValueError(
                "GradNormBalancer.step: gradient norms are required in mode='full'."
            )
        G_detached = per_task_grad_norms_synced.detach().to(self.device)
        r = (L / self.initial_loss).clamp(min=1e-12)
        r = r / r.mean().clamp(min=1e-12)
        target = (G_detached.mean() * (r ** self.alpha)).detach()

        W = self.task_weights()
        G = W * G_detached
        gradnorm_loss = (G - target).abs().sum()

        self.optimizer.zero_grad()
        gradnorm_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.log_weights.clamp_(-self.log_clip, self.log_clip)
        return gradnorm_loss.detach()


def synced_per_task_tensor(
    values: list[torch.Tensor] | torch.Tensor,
    *,
    state,
    device: torch.device,
) -> torch.Tensor:
    """Stack per-task scalar tensors and all-reduce(AVG) across DDP ranks."""

    if isinstance(values, torch.Tensor):
        stacked = values.to(device=device, dtype=torch.float32)
    else:
        stacked = torch.stack([v.to(device=device, dtype=torch.float32) for v in values])
    if state.enabled:
        dist.all_reduce(stacked, op=dist.ReduceOp.AVG)
    return stacked


def parse_vol_points_schedule(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    parts = [p.strip() for p in text.split(":") if p.strip()]
    if len(parts) % 2 != 0 or len(parts) == 0:
        raise ValueError(
            f"--vol-points-schedule must be 'EPOCH:POINTS' pairs joined by ':'; got '{text}'"
        )
    schedule: list[tuple[int, int]] = []
    for i in range(0, len(parts), 2):
        try:
            epoch = int(parts[i])
            points = int(parts[i + 1])
        except ValueError as exc:
            raise ValueError(
                f"--vol-points-schedule values must be integers; got '{text}'"
            ) from exc
        if epoch < 0 or points <= 0:
            raise ValueError(
                f"--vol-points-schedule entries must satisfy epoch>=0 and points>0; "
                f"got {epoch}:{points}"
            )
        schedule.append((epoch, points))
    schedule.sort(key=lambda x: x[0])
    seen: set[int] = set()
    for start_epoch, _ in schedule:
        if start_epoch in seen:
            raise ValueError(
                f"--vol-points-schedule has duplicate epoch breakpoint {start_epoch}"
            )
        seen.add(start_epoch)
    if schedule[0][0] != 0:
        raise ValueError(
            f"--vol-points-schedule must start at epoch 0; got first entry {schedule[0]}"
        )
    return schedule


def vol_points_for_epoch(
    schedule: list[tuple[int, int]],
    epoch: int,
    fallback: int,
) -> int:
    if not schedule:
        return fallback
    current = schedule[0][1]
    for start_epoch, points in schedule:
        if epoch >= start_epoch:
            current = points
        else:
            break
    return current


def rebuild_train_loader_with_vol_points(
    config: Config,
    old_train_loader: DataLoader,
    n_points: int,
    distributed_state,
) -> DataLoader:
    """Rebuild the training DataLoader with a new max_volume_points value.

    Reuses the existing ``DrivAerMLCaseStore`` so cached point counts and
    artifact-path resolutions survive the swap. The view list is recomputed
    because ``max_volume_points`` changes the per-case view count, which in
    turn changes the dataset length that the distributed sampler must see.
    When the previous loader is a ``DistillationDataset`` the teacher cache
    and matching collate function are preserved across the swap so the KD
    loss term keeps firing at every volume-points breakpoint.
    """

    old_ds = old_train_loader.dataset
    sampling_mode = (
        "train_random" if (config.train_surface_points > 0 or n_points > 0) else "full"
    )
    teacher_cache = getattr(old_ds, "teacher_cache", None)
    if isinstance(old_ds, DistillationDataset) and teacher_cache is not None:
        train_ds: DrivAerMLSurfaceDataset = DistillationDataset(
            old_ds.case_ids,
            store=old_ds.store,
            max_surface_points=config.train_surface_points,
            max_volume_points=n_points,
            sampling_mode=sampling_mode,
            teacher_cache=teacher_cache,
        )
        collate_fn = pad_collate_with_teacher
    else:
        train_ds = DrivAerMLSurfaceDataset(
            old_ds.case_ids,
            store=old_ds.store,
            max_surface_points=config.train_surface_points,
            max_volume_points=n_points,
            sampling_mode=sampling_mode,
        )
        collate_fn = pad_collate
    train_sampler = None
    train_shuffle = True
    if distributed_state is not None and distributed_state.enabled:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=distributed_state.world_size,
            rank=distributed_state.rank,
            shuffle=True,
            drop_last=True,
        )
        train_shuffle = False
    kwargs = loader_kwargs(config)
    kwargs["collate_fn"] = collate_fn
    return DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Knowledge distillation (PR #1001): ensemble teacher cache + KD loss
# ---------------------------------------------------------------------------

KD_TEACHER_SURFACE_KEY = "kd_teacher_surface_pred"
KD_TEACHER_VOLUME_KEY = "kd_teacher_volume_pred"
KD_CACHE_FILE_VERSION = 1


def parse_distill_teacher_runs(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    raw = text.replace(";", ",").split(",")
    ids = [token.strip() for token in raw if token.strip()]
    seen: set[str] = set()
    deduped: list[str] = []
    for run_id in ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        deduped.append(run_id)
    return deduped


def kd_cache_dir_for(base: Path, run_ids: list[str]) -> Path:
    """Subdirectory under the teacher path keyed by the sorted teacher set."""

    sorted_ids = sorted(run_ids)
    digest = hashlib.sha1(",".join(sorted_ids).encode("utf-8")).hexdigest()[:10]
    return Path(base) / f"k{len(sorted_ids)}_{digest}"


def kd_case_cache_path(cache_dir: Path, case_id: str) -> Path:
    return cache_dir / "cases" / f"{case_id}.pt"


def kd_artifact_dir(cache_root: Path) -> Path:
    return Path(cache_root) / "artifacts"


def _kd_chunk_indices(total: int, view_index: int, view_count: int) -> torch.Tensor:
    """Mirror of ``DrivAerMLSurfaceDataset._indices`` for the eval_chunk branch."""

    if total <= 0:
        return torch.empty(0, dtype=torch.long)
    if view_count <= 1:
        return torch.arange(total, dtype=torch.long)
    if view_index >= view_count:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(view_index, total, view_count, dtype=torch.long)


def _kd_view_count(total: int, points_per_view: int) -> int:
    if points_per_view <= 0 or total <= points_per_view:
        return 1
    return max(1, math.ceil(total / points_per_view))


def _build_teacher_from_config(config: dict) -> SurfaceTransolver:
    """Build a SurfaceTransolver matching a logged-run config and downgrade its
    output heads to the legacy ``LinearProjection`` layout used by the PR #880
    ensemble checkpoints (pre PR #958 dedicated decoder heads).
    """

    model = SurfaceTransolver(
        n_layers=int(config.get("model_layers", 3)),
        n_hidden=int(config.get("model_hidden_dim", 192)),
        dropout=float(config.get("model_dropout", 0.0)),
        n_head=int(config.get("model_heads", 3)),
        mlp_ratio=int(config.get("model_mlp_ratio", 4)),
        slice_num=int(config.get("model_slices", 96)),
        rff_num_features=int(config.get("rff_num_features", 0)),
        rff_sigma=float(config.get("rff_sigma", 1.0)),
        rff_init_sigmas=parse_rff_init_sigmas(str(config.get("rff_init_sigmas", "") or "")),
        pos_encoding_mode=str(config.get("pos_encoding_mode", "sincos")),
        use_qk_norm=bool(config.get("use_qk_norm", False)),
        use_surf_to_vol_xattn=bool(config.get("use_surf_to_vol_xattn", False)),
    )
    # Replace the PR #958 MLP heads with the legacy single-linear heads so the
    # PR #880 ensemble state dict loads via ``strict=True``.
    n_hidden = int(config.get("model_hidden_dim", 192))
    model.surface_out = LinearProjection(n_hidden, SURFACE_Y_DIM)
    model.volume_out = LinearProjection(n_hidden, VOLUME_Y_DIM)
    return model


def _kd_load_teacher(
    run_id: str,
    artifact_root: Path,
    device: torch.device,
    api,
    entity: str,
    project: str,
    state=None,
) -> SurfaceTransolver:
    """Download (rank 0) and load a teacher checkpoint with legacy heads.

    All ranks read the same on-disk artifact after rank 0 downloads; the
    download itself is serialised behind a distributed barrier to avoid
    concurrent writes to the same artifact directory.
    """

    from ensemble_eval import download_checkpoint

    if state is None or not state.enabled or state.is_main:
        download_checkpoint(api, entity, project, run_id, artifact_root)
    if state is not None:
        distributed_barrier(state)
    artifact_dir = artifact_root / run_id
    config_path = artifact_dir / "config.yaml"
    checkpoint_path = artifact_dir / "checkpoint.pt"
    with config_path.open("r") as fh:
        teacher_config = yaml.safe_load(fh)
    teacher = _build_teacher_from_config(teacher_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    missing, unexpected = teacher.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch for teacher {run_id}: missing={missing}, "
            f"unexpected={unexpected}"
        )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def _kd_teacher_chunk_forward(
    teacher: nn.Module,
    cases: list[DrivAerMLCase],
    device: torch.device,
    amp_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the teacher on a micro-batch of (case, view) chunks.

    Returns padded ``(surface_pred, volume_pred)`` in the model's normalised
    output space; masks are not returned because the caller iterates the
    chunk index lists separately to scatter predictions back to the case
    arrays.
    """

    surface_lengths = [c.surface_x.shape[0] for c in cases]
    volume_lengths = [c.volume_x.shape[0] for c in cases]
    max_surface = max(surface_lengths) if surface_lengths else 0
    max_volume = max(volume_lengths) if volume_lengths else 0
    surface_dim = cases[0].surface_x.shape[1] if cases else 0
    volume_dim = cases[0].volume_x.shape[1] if cases else 0
    bsz = len(cases)
    surface_x = torch.zeros((bsz, max(max_surface, 1), surface_dim), dtype=torch.float32)
    volume_x = torch.zeros((bsz, max(max_volume, 1), volume_dim), dtype=torch.float32)
    surface_mask = torch.zeros((bsz, max(max_surface, 1)), dtype=torch.bool)
    volume_mask = torch.zeros((bsz, max(max_volume, 1)), dtype=torch.bool)
    for i, case in enumerate(cases):
        sl = surface_lengths[i]
        vl = volume_lengths[i]
        if sl > 0:
            surface_x[i, :sl] = case.surface_x
            surface_mask[i, :sl] = True
        if vl > 0:
            volume_x[i, :vl] = case.volume_x
            volume_mask[i, :vl] = True
    surface_x = surface_x.to(device, non_blocking=True)
    volume_x = volume_x.to(device, non_blocking=True)
    surface_mask = surface_mask.to(device, non_blocking=True)
    volume_mask = volume_mask.to(device, non_blocking=True)
    with autocast_context(device, amp_mode):
        out = teacher(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
    return out["surface_preds"].float(), out["volume_preds"].float()


def _kd_iter_case_chunks(
    case_id: str,
    store: DrivAerMLCaseStore,
    eval_surface_points: int,
    eval_volume_points: int,
    chunk_batch_size: int,
) -> Iterable[tuple[list[DrivAerMLCase], list[torch.Tensor], list[torch.Tensor]]]:
    """Yield micro-batches of eval_chunk views for a single case.

    The full case is loaded from disk once and sliced in memory for each
    chunk, so all K teachers reuse the same in-memory tensors.
    """

    counts = store.case_point_counts(case_id)
    n_surface = int(counts["n_surface"])
    n_volume = int(counts["n_volume"])
    surface_views = _kd_view_count(n_surface, eval_surface_points)
    volume_views = _kd_view_count(n_volume, eval_volume_points)
    view_count = max(surface_views, volume_views)
    full_case = store.load_case(case_id)

    def _slice(case: DrivAerMLCase, surface_idx: torch.Tensor, volume_idx: torch.Tensor) -> DrivAerMLCase:
        if surface_idx.numel() == 0:
            sx = torch.zeros((0, case.surface_x.shape[1]), dtype=case.surface_x.dtype)
            sy = torch.zeros((0, case.surface_y.shape[1]), dtype=case.surface_y.dtype)
        else:
            sx = case.surface_x.index_select(0, surface_idx)
            sy = case.surface_y.index_select(0, surface_idx)
        if volume_idx.numel() == 0:
            vx = torch.zeros((0, case.volume_x.shape[1]), dtype=case.volume_x.dtype)
            vy = torch.zeros((0, case.volume_y.shape[1]), dtype=case.volume_y.dtype)
        else:
            vx = case.volume_x.index_select(0, volume_idx)
            vy = case.volume_y.index_select(0, volume_idx)
        return DrivAerMLCase(
            case_id=case.case_id,
            surface_x=sx,
            surface_y=sy,
            volume_x=vx,
            volume_y=vy,
            metadata=dict(case.metadata),
        )

    pending_cases: list[DrivAerMLCase] = []
    pending_surface_idx: list[torch.Tensor] = []
    pending_volume_idx: list[torch.Tensor] = []
    for view_idx in range(view_count):
        if view_idx < surface_views and surface_views > 1:
            surface_idx = _kd_chunk_indices(n_surface, view_idx, surface_views)
        elif surface_views == 1 and view_idx == 0 and n_surface > 0:
            surface_idx = torch.arange(n_surface, dtype=torch.long)
        else:
            surface_idx = torch.empty(0, dtype=torch.long)
        if view_idx < volume_views and volume_views > 1:
            volume_idx = _kd_chunk_indices(n_volume, view_idx, volume_views)
        elif volume_views == 1 and view_idx == 0 and n_volume > 0:
            volume_idx = torch.arange(n_volume, dtype=torch.long)
        else:
            volume_idx = torch.empty(0, dtype=torch.long)
        chunk_case = _slice(full_case, surface_idx, volume_idx)
        pending_cases.append(chunk_case)
        pending_surface_idx.append(surface_idx)
        pending_volume_idx.append(volume_idx)
        if len(pending_cases) >= chunk_batch_size:
            yield pending_cases, pending_surface_idx, pending_volume_idx
            pending_cases = []
            pending_surface_idx = []
            pending_volume_idx = []
    if pending_cases:
        yield pending_cases, pending_surface_idx, pending_volume_idx


def _kd_save_case_cache(
    cache_dir: Path,
    case_id: str,
    surface_full: torch.Tensor,
    volume_full: torch.Tensor,
    teacher_run_ids: list[str],
) -> None:
    payload = {
        "version": KD_CACHE_FILE_VERSION,
        "case_id": case_id,
        "teacher_run_ids": list(teacher_run_ids),
        "surface_pred": surface_full.to(dtype=torch.bfloat16, copy=True).contiguous(),
        "volume_pred": volume_full.to(dtype=torch.bfloat16, copy=True).contiguous(),
    }
    target = kd_case_cache_path(cache_dir, case_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(target)


def build_teacher_cache(
    teacher_run_ids: list[str],
    config: Config,
    state,
    device: torch.device,
) -> Path:
    """Build the K-teacher averaged prediction cache for the train split.

    All teachers are loaded onto the device once. Cases are case-sharded
    across DDP ranks; for each owned case we read the case once, split it
    into eval_chunk views, run every teacher on the same view stack, and
    save the bf16 averaged predictions to ``cases/<case_id>.pt``. The build
    is idempotent: already-cached cases are skipped, so a partial build can
    be resumed by re-running the cache builder.
    """

    cache_dir = kd_cache_dir_for(Path(config.distill_teacher_path), teacher_run_ids)
    cases_dir = cache_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    store = DrivAerMLCaseStore(
        manifest_path=config.manifest, root=config.data_root or None
    )
    all_train_ids = store.case_ids("train")
    if config.debug:
        all_train_ids = all_train_ids[:4]

    todo_ids = [cid for cid in all_train_ids if not kd_case_cache_path(cache_dir, cid).exists()]
    if state.is_main:
        n_have = len(all_train_ids) - len(todo_ids)
        print(
            f"[KD] cache_dir={cache_dir} teachers={len(teacher_run_ids)} "
            f"cached={n_have}/{len(all_train_ids)} todo={len(todo_ids)}"
        )

    if not todo_ids:
        distributed_barrier(state)
        if state.is_main:
            print("[KD] cache already complete; skipping build.")
        return cache_dir

    rank = state.rank if state.enabled else 0
    world = state.world_size if state.enabled else 1
    shard_ids = [cid for i, cid in enumerate(todo_ids) if i % world == rank]
    print(
        f"[KD][rank {rank}] shard_size={len(shard_ids)} of {len(todo_ids)} "
        f"todo cases (world={world})"
    )

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8")
    artifact_root = kd_artifact_dir(Path(config.distill_teacher_path))
    artifact_root.mkdir(parents=True, exist_ok=True)

    n_teachers = len(teacher_run_ids)

    teachers: list[tuple[str, nn.Module]] = []
    for teacher_idx, run_id in enumerate(teacher_run_ids):
        t_load_start = time.time()
        if state.is_main:
            print(f"[KD] loading teacher {teacher_idx + 1}/{n_teachers} ({run_id})")
        teacher = _kd_load_teacher(
            run_id, artifact_root, device, api, entity, project, state=state
        )
        if state.is_main:
            print(f"[KD] teacher {run_id} loaded in {time.time() - t_load_start:.1f}s")
        teachers.append((run_id, teacher))

    build_start = time.time()
    for case_idx, case_id in enumerate(shard_ids):
        case_start = time.time()
        counts = store.case_point_counts(case_id)
        n_surface = int(counts["n_surface"])
        n_volume = int(counts["n_volume"])
        surface_sum = torch.zeros(
            (max(n_surface, 1), SURFACE_Y_DIM), dtype=torch.float32
        )
        volume_sum = torch.zeros(
            (max(n_volume, 1), VOLUME_Y_DIM), dtype=torch.float32
        )

        chunks = list(
            _kd_iter_case_chunks(
                case_id,
                store,
                config.distill_cache_eval_surface_points,
                config.distill_cache_eval_volume_points,
                config.distill_cache_chunk_batch_size,
            )
        )

        for _, teacher in teachers:
            for cases, surface_idx_list, volume_idx_list in chunks:
                surface_pred, volume_pred = _kd_teacher_chunk_forward(
                    teacher, cases, device, config.amp_mode
                )
                surface_pred_cpu = surface_pred.detach().cpu()
                volume_pred_cpu = volume_pred.detach().cpu()
                for j, (s_idx, v_idx) in enumerate(
                    zip(surface_idx_list, volume_idx_list)
                ):
                    s_n = s_idx.numel()
                    v_n = v_idx.numel()
                    if s_n > 0:
                        surface_sum[s_idx] += surface_pred_cpu[j, :s_n]
                    if v_n > 0:
                        volume_sum[v_idx] += volume_pred_cpu[j, :v_n]

        del chunks
        surface_avg = surface_sum / max(n_teachers, 1)
        volume_avg = volume_sum / max(n_teachers, 1)
        _kd_save_case_cache(
            cache_dir, case_id, surface_avg, volume_avg, teacher_run_ids
        )
        del surface_sum, volume_sum, surface_avg, volume_avg

        case_dt = time.time() - case_start
        total_dt = time.time() - build_start
        rate = (case_idx + 1) / max(total_dt, 1e-6)
        remaining = len(shard_ids) - (case_idx + 1)
        eta_min = (remaining / max(rate, 1e-6)) / 60.0
        print(
            f"[KD][rank {rank}] case {case_idx + 1}/{len(shard_ids)} "
            f"({case_id}) {case_dt:.1f}s; rate={rate:.2f} case/s; "
            f"eta={eta_min:.1f} min"
        )
        if (case_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    distributed_barrier(state)
    if state.is_main:
        all_present = sum(
            1 for cid in all_train_ids if kd_case_cache_path(cache_dir, cid).exists()
        )
        print(
            f"[KD] cache build complete: {all_present}/{len(all_train_ids)} per-case files at "
            f"{cache_dir}"
        )
    for _, teacher in teachers:
        del teacher
    teachers.clear()
    torch.cuda.empty_cache()
    gc.collect()
    return cache_dir


class DistillationDataset(DrivAerMLSurfaceDataset):
    """Training dataset that pairs each sampled view with teacher predictions.

    Subclasses the standard surface dataset and overrides ``__getitem__`` so
    the same per-view ``train_random`` / ``eval_chunk`` indices that select
    the case rows also select the corresponding rows from the cached teacher
    predictions. The teacher tensors are stashed in ``DrivAerMLCase.metadata``
    under ``KD_TEACHER_*`` keys so the custom collator can pad them alongside
    the targets without modifying ``data/loader.py`` or the case schema.
    """

    def __init__(
        self,
        case_ids: list[str],
        *,
        store: DrivAerMLCaseStore,
        max_surface_points: int,
        max_volume_points: int,
        sampling_mode: str,
        teacher_cache: dict[str, dict[str, torch.Tensor]],
    ):
        super().__init__(
            case_ids,
            store=store,
            max_surface_points=max_surface_points,
            max_volume_points=max_volume_points,
            sampling_mode=sampling_mode,
        )
        self.teacher_cache = teacher_cache

    def __getitem__(self, idx: int) -> DrivAerMLCase:
        view = self.views[idx]
        counts = self.store.case_point_counts(view.case_id)
        surface_idx = self._indices(
            counts["n_surface"],
            self.max_surface_points,
            view,
            group_view_count=view.surface_view_count,
        )
        volume_idx = self._indices(
            counts["n_volume"],
            self.max_volume_points,
            view,
            group_view_count=view.volume_view_count,
        )
        case = self.store.load_case(
            view.case_id,
            surface_rows=None if surface_idx is None else surface_idx.numpy(),
            volume_rows=None if volume_idx is None else volume_idx.numpy(),
        )
        metadata = dict(case.metadata)
        metadata["n_surface_full"] = int(counts["n_surface"])
        metadata["n_surface_loaded"] = int(case.surface_x.shape[0])
        metadata["surface_view_index"] = int(view.view_index)
        metadata["surface_view_count"] = int(view.surface_view_count)
        metadata["surface_sampling_mode"] = view.sampling_mode
        metadata["n_volume_full"] = int(counts["n_volume"])
        metadata["n_volume_loaded"] = int(case.volume_x.shape[0])
        metadata["volume_view_index"] = int(view.view_index)
        metadata["volume_view_count"] = int(view.volume_view_count)
        metadata["volume_sampling_mode"] = view.sampling_mode
        metadata["joint_view_count"] = int(view.view_count)

        teacher_entry = self.teacher_cache.get(view.case_id)
        if teacher_entry is not None:
            t_surface_full = teacher_entry["surface_pred"]
            t_volume_full = teacher_entry["volume_pred"]
            if surface_idx is None:
                surface_teacher = t_surface_full[: case.surface_x.shape[0]]
            elif surface_idx.numel() == 0:
                surface_teacher = torch.zeros(
                    (0, t_surface_full.shape[-1]), dtype=t_surface_full.dtype
                )
            else:
                surface_teacher = t_surface_full[surface_idx]
            if volume_idx is None:
                volume_teacher = t_volume_full[: case.volume_x.shape[0]]
            elif volume_idx.numel() == 0:
                volume_teacher = torch.zeros(
                    (0, t_volume_full.shape[-1]), dtype=t_volume_full.dtype
                )
            else:
                volume_teacher = t_volume_full[volume_idx]
            metadata[KD_TEACHER_SURFACE_KEY] = surface_teacher.contiguous()
            metadata[KD_TEACHER_VOLUME_KEY] = volume_teacher.contiguous()

        return DrivAerMLCase(
            case_id=case.case_id,
            surface_x=case.surface_x,
            surface_y=case.surface_y,
            volume_x=case.volume_x,
            volume_y=case.volume_y,
            metadata=metadata,
        )


def pad_collate_with_teacher(samples: list[DrivAerMLCase]) -> SurfaceBatch:
    """Pad a batch of samples and attach padded teacher predictions.

    Surface/volume teacher tensors are popped out of each sample's metadata so
    they don't pollute the printed metadata, then padded to the same
    ``[B, N_max, C]`` layout as the targets and attached to the returned
    ``SurfaceBatch`` as ``surface_teacher`` / ``volume_teacher`` attributes.
    Padding rows are zero; the existing masks correctly disable padded rows
    in the loss.
    """

    teacher_surface_list: list[torch.Tensor] = []
    teacher_volume_list: list[torch.Tensor] = []
    has_teacher = True
    for sample in samples:
        ts = sample.metadata.pop(KD_TEACHER_SURFACE_KEY, None)
        tv = sample.metadata.pop(KD_TEACHER_VOLUME_KEY, None)
        if ts is None or tv is None:
            has_teacher = False
        teacher_surface_list.append(ts)
        teacher_volume_list.append(tv)
    batch = pad_collate(samples)
    if not has_teacher:
        return batch
    bsz = batch.surface_y.shape[0]
    max_surface = batch.surface_y.shape[1]
    max_volume = batch.volume_y.shape[1]
    surface_dim = batch.surface_y.shape[2]
    volume_dim = batch.volume_y.shape[2]
    teacher_surface = torch.zeros((bsz, max_surface, surface_dim), dtype=torch.float32)
    teacher_volume = torch.zeros((bsz, max_volume, volume_dim), dtype=torch.float32)
    for i, (ts, tv) in enumerate(zip(teacher_surface_list, teacher_volume_list)):
        if ts is not None and ts.numel() > 0:
            teacher_surface[i, : ts.shape[0]] = ts.float()
        if tv is not None and tv.numel() > 0:
            teacher_volume[i, : tv.shape[0]] = tv.float()
    setattr(batch, "surface_teacher", teacher_surface)
    setattr(batch, "volume_teacher", teacher_volume)
    return batch


def _kd_load_case_cache(cache_dir: Path, case_id: str) -> dict[str, torch.Tensor] | None:
    path = kd_case_cache_path(cache_dir, case_id)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "surface_pred": payload["surface_pred"],
        "volume_pred": payload["volume_pred"],
    }


def preload_teacher_cache(
    cache_dir: Path,
    case_ids: list[str],
    *,
    log: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load the per-case teacher prediction cache into RAM.

    Returned dict is read-only by convention; DataLoader workers forked
    from this process inherit the same memory pages on Linux.
    """

    cache: dict[str, dict[str, torch.Tensor]] = {}
    missing: list[str] = []
    t0 = time.time()
    n_surface_bytes = 0
    n_volume_bytes = 0
    for case_id in case_ids:
        entry = _kd_load_case_cache(cache_dir, case_id)
        if entry is None:
            missing.append(case_id)
            continue
        cache[case_id] = entry
        n_surface_bytes += entry["surface_pred"].element_size() * entry["surface_pred"].numel()
        n_volume_bytes += entry["volume_pred"].element_size() * entry["volume_pred"].numel()
    if missing:
        raise FileNotFoundError(
            f"Missing teacher cache for {len(missing)} case(s) at {cache_dir}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if log:
        gb = (n_surface_bytes + n_volume_bytes) / (1024 ** 3)
        print(
            f"[KD] preloaded teacher cache: {len(cache)} cases, "
            f"{gb:.2f} GiB resident (surface={n_surface_bytes/1e9:.2f}GB, "
            f"volume={n_volume_bytes/1e9:.2f}GB) in {time.time()-t0:.1f}s"
        )
    return cache


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    run = None
    try:
        config = parse_args(argv)
        kill_thresholds = parse_kill_thresholds(config.kill_thresholds)
        vol_points_schedule = parse_vol_points_schedule(config.vol_points_schedule)
        teacher_run_ids = parse_distill_teacher_runs(config.distill_teacher_runs)
        if not 0.0 <= config.distill_lambda_kd <= 1.0:
            raise ValueError(
                f"--distill-lambda-kd must be in [0, 1]; got {config.distill_lambda_kd}"
            )
        if not config.distill_teacher_path and (
            config.distill_build_cache_only or teacher_run_ids
        ):
            raise ValueError(
                "--distill-build-cache-only and --distill-teacher-runs require "
                "--distill-teacher-path to be set."
            )
        if config.distill_build_cache_only and not teacher_run_ids:
            raise ValueError(
                "--distill-build-cache-only requires --distill-teacher-runs."
            )
        kd_enabled = bool(config.distill_teacher_path) and config.distill_lambda_kd > 0.0
        requested_epochs = config.epochs
        if os.environ.get("SENPAI_MAX_EPOCHS"):
            requested_epochs = min(requested_epochs, int(os.environ["SENPAI_MAX_EPOCHS"]))
        max_epochs = min(requested_epochs, 3) if config.debug else requested_epochs
        timeout_minutes, val_budget_minutes, train_timeout_minutes = timeout_budget_minutes()
        device = state.device
        if state.is_main:
            ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
            print(f"Device: {device}{ddp_suffix}" + (" [DEBUG]" if config.debug else ""))
            if teacher_run_ids or config.distill_teacher_path:
                print(
                    f"[KD] teachers={len(teacher_run_ids)} "
                    f"lambda_kd={config.distill_lambda_kd} "
                    f"path={config.distill_teacher_path or '<unset>'} "
                    f"build_cache_only={config.distill_build_cache_only} "
                    f"enabled={kd_enabled}"
                )

        kd_cache_dir: Path | None = None
        if teacher_run_ids:
            kd_cache_dir = build_teacher_cache(teacher_run_ids, config, state, device)
            distributed_barrier(state)
            if config.distill_build_cache_only:
                if state.is_main:
                    print(f"[KD] --distill-build-cache-only set; cache at {kd_cache_dir}; exiting.")
                return
        elif kd_enabled:
            # No --distill-teacher-runs given but --distill-teacher-path is set:
            # the user is pointing at a pre-built cache directory directly.
            kd_cache_dir = Path(config.distill_teacher_path)
            if not (kd_cache_dir / "cases").is_dir():
                # Helpful fallback: maybe the path is the cache *root* and the
                # build_teacher_cache layout placed cases under k<K>_<sha>/.
                candidates = [p for p in kd_cache_dir.iterdir() if (p / "cases").is_dir()] if kd_cache_dir.is_dir() else []
                if len(candidates) == 1:
                    kd_cache_dir = candidates[0]
                else:
                    raise FileNotFoundError(
                        f"--distill-teacher-path={config.distill_teacher_path} does not "
                        f"contain a `cases/` subdirectory and no unique k*_* child does either."
                    )

        if vol_points_schedule:
            initial_vol_points = vol_points_for_epoch(
                vol_points_schedule, 0, config.train_volume_points
            )
            if config.debug:
                initial_vol_points = min(initial_vol_points, 8_192)
            config.train_volume_points = initial_vol_points
            if state.is_main:
                print(
                    "Volume-points curriculum: "
                    + ", ".join(f"ep{e}->{p}" for e, p in vol_points_schedule)
                )
        current_train_vol_points = config.train_volume_points

        train_loader, val_loaders, test_loaders, stats = make_loaders(config, distributed_state=state)
        if kd_enabled:
            assert kd_cache_dir is not None
            old_train_ds = train_loader.dataset
            teacher_cache = preload_teacher_cache(
                kd_cache_dir,
                list(old_train_ds.case_ids),
                log=state.is_main,
            )
            train_ds = DistillationDataset(
                old_train_ds.case_ids,
                store=old_train_ds.store,
                max_surface_points=config.train_surface_points,
                max_volume_points=config.train_volume_points,
                sampling_mode=old_train_ds.sampling_mode,
                teacher_cache=teacher_cache,
            )
            train_sampler = None
            train_shuffle = True
            if state.enabled:
                train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=state.world_size,
                    rank=state.rank,
                    shuffle=True,
                    drop_last=True,
                )
                train_shuffle = False
            kwargs = loader_kwargs(config)
            kwargs["collate_fn"] = pad_collate_with_teacher
            train_loader = DataLoader(
                train_ds,
                batch_size=config.batch_size,
                shuffle=train_shuffle,
                sampler=train_sampler,
                drop_last=True,
                **kwargs,
            )
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
        # Surface targets are [cp, tau_x, tau_y, tau_z] — channels 2 and 3 carry
        # the largest gap to AB-UPT, so we expose per-channel weights here.
        surface_channel_weights = torch.tensor(
            [1.0, 1.0, config.tau_y_loss_weight, config.tau_z_loss_weight],
            device=device,
            dtype=torch.float32,
        )
        use_channel_weights = bool(
            (config.tau_y_loss_weight != 1.0) or (config.tau_z_loss_weight != 1.0)
        )
        if config.compile_model:
            model = torch.compile(model)
        if state.enabled:
            ddp_kwargs = {}
            if device.type == "cuda":
                ddp_kwargs = {"device_ids": [state.local_rank], "output_device": state.local_rank}
            # The surface->volume cross-attention sublayer (PR #823) gates on
            # `surface_tokens > 0`, but DrivAerML's loader returns
            # max(surface_views, volume_views) views per case, so views with
            # view_index >= surface_views have N_S=0 (and the gate skips the
            # layer). Different ranks hit different sub-graphs in a single
            # batch, so DDP's default find_unused_parameters=False deadlocks
            # at the gradient bucket allreduce. Enabling find_unused_parameters
            # whenever the cross-attention is on lets DDP synchronize unused
            # params across ranks, at a small per-step overhead.
            if config.use_surf_to_vol_xattn:
                ddp_kwargs["find_unused_parameters"] = True
            model = DistributedDataParallel(model, **ddp_kwargs)
        base_model = unwrap_model(model)
        if state.is_main:
            print(f"Model: SurfaceTransolver grouped surface+volume ({n_params / 1e6:.2f}M params)")

        optimizer = build_optimizer(base_model, config)
        scheduler = build_lr_scheduler(optimizer, config, max_epochs)
        ema = EMA(base_model, decay=config.ema_decay, start_step=config.ema_start_step) if config.use_ema else None

        balancer: GradNormBalancer | None = None
        gradnorm_shared_params: list[nn.Parameter] = []
        if config.use_gradnorm:
            if config.gradnorm_mode == "full" and config.compile_model:
                raise ValueError(
                    "--use-gradnorm --gradnorm-mode full requires --no-compile-model "
                    "(autograd.grad with retain_graph is not supported through "
                    "torch.compile)."
                )
            balancer = GradNormBalancer(
                num_tasks=len(GRADNORM_TASK_NAMES),
                mode=config.gradnorm_mode,
                alpha=config.gradnorm_alpha,
                lr=config.gradnorm_lr,
                device=device,
                init_warmup_steps=config.gradnorm_init_warmup_steps,
                log_clip=config.gradnorm_log_clip,
                ema_beta=config.gradnorm_ema_beta,
                min_weight=config.gradnorm_min_weight,
            )
            if config.gradnorm_mode == "full":
                gradnorm_shared_params = [
                    p for p in base_model.backbone.blocks[-1].parameters() if p.requires_grad
                ]
                if not gradnorm_shared_params:
                    raise RuntimeError(
                        "GradNorm: no trainable parameters in the last shared encoder block."
                    )
            if state.is_main:
                if config.gradnorm_mode == "full":
                    shared_param_count = sum(p.numel() for p in gradnorm_shared_params)
                    print(
                        f"GradNorm enabled (mode=full): {len(GRADNORM_TASK_NAMES)} tasks "
                        f"({', '.join(GRADNORM_TASK_NAMES)}); alpha={config.gradnorm_alpha}; "
                        f"lr={config.gradnorm_lr}; warmup={config.gradnorm_init_warmup_steps} "
                        f"steps; log_clip={config.gradnorm_log_clip}; "
                        f"shared_params={shared_param_count:,d} (last backbone block)"
                    )
                else:
                    print(
                        f"GradNorm enabled (mode=ema_proxy): {len(GRADNORM_TASK_NAMES)} tasks "
                        f"({', '.join(GRADNORM_TASK_NAMES)}); alpha={config.gradnorm_alpha}; "
                        f"ema_beta={config.gradnorm_ema_beta}; "
                        f"warmup={config.gradnorm_init_warmup_steps} steps; "
                        f"log_clip={config.gradnorm_log_clip}; "
                        f"min_weight={config.gradnorm_min_weight}; "
                        f"closed-form weights from r_i = ema_loss_i / initial_loss_i"
                    )
        total_estimated_steps = max(1, max_epochs * max(len(train_loader), 1))
        if kill_thresholds and state.is_main:
            print("Kill thresholds:", "; ".join(threshold.describe() for threshold in kill_thresholds))
        train_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)
        val_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)

        # Log multi-sigma RFF init layout for verification.
        if state.is_main:
            ss = getattr(base_model, "surface_string_sep", None)
            if ss is not None and getattr(ss, "init_sigmas", None) is not None:
                print(
                    f"StringSep multi-sigma init: sigmas={ss.init_sigmas} "
                    f"num_features={ss.num_features} layout=round-robin per axis "
                    f"(log_freq[d, f] = log(sigmas[f % len(sigmas)]))"
                )
            if use_channel_weights:
                print(
                    f"Surface channel weights: cp=1.0 tau_x=1.0 "
                    f"tau_y={config.tau_y_loss_weight} tau_z={config.tau_z_loss_weight}"
                )

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

        if state.is_main:
            ss = getattr(base_model, "surface_string_sep", None)
            init_sigmas_for_log = (
                list(ss.init_sigmas)
                if ss is not None and getattr(ss, "init_sigmas", None) is not None
                else []
            )
            wandb.summary["model/string_sep_init_sigmas"] = init_sigmas_for_log
            wandb.summary["loss/tau_y_loss_weight"] = config.tau_y_loss_weight
            wandb.summary["loss/tau_z_loss_weight"] = config.tau_z_loss_weight
            wandb.summary["kd/enabled"] = 1.0 if kd_enabled else 0.0
            wandb.summary["kd/teacher_count"] = len(teacher_run_ids)
            wandb.summary["kd/teacher_run_ids"] = list(teacher_run_ids)
            wandb.summary["kd/lambda_kd"] = float(config.distill_lambda_kd)
            if kd_cache_dir is not None:
                wandb.summary["kd/cache_dir"] = str(kd_cache_dir)

        best_val = float("inf")
        best_metrics: dict[str, float] = {}
        best_checkpoint_source = "ema" if ema is not None else "raw"
        global_step = 0
        early_stop_reason: str | None = None
        timeout_hit = False
        train_start = time.time()

        for epoch in range(max_epochs):
            if vol_points_schedule:
                desired_vol_points = vol_points_for_epoch(
                    vol_points_schedule, epoch, config.train_volume_points
                )
                if config.debug:
                    desired_vol_points = min(desired_vol_points, 8_192)
                if desired_vol_points != current_train_vol_points:
                    if state.is_main:
                        print(
                            f"Volume-points curriculum: epoch {epoch} -> "
                            f"train_volume_points={desired_vol_points} "
                            f"(was {current_train_vol_points})"
                        )
                    config.train_volume_points = desired_vol_points
                    train_loader = rebuild_train_loader_with_vol_points(
                        config, train_loader, desired_vol_points, state
                    )
                    current_train_vol_points = desired_vol_points
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
                gradnorm_metrics: dict[str, float] = {}
                if balancer is not None:
                    per_task_losses = per_task_train_losses(
                        model, batch, transform, device, config.amp_mode
                    )
                    synced_losses = synced_per_task_tensor(
                        [t.detach() for t in per_task_losses], state=state, device=device
                    )
                    balancer.update_initial_loss(synced_losses)

                    synced_norms: torch.Tensor | None = None
                    if balancer.mode == "full":
                        per_task_grad_norms: list[torch.Tensor] = []
                        for L_i in per_task_losses:
                            grads = torch.autograd.grad(
                                L_i,
                                gradnorm_shared_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            norm_sq = torch.zeros((), device=device, dtype=torch.float32)
                            for g in grads:
                                if g is not None:
                                    norm_sq = norm_sq + g.float().square().sum()
                            per_task_grad_norms.append(norm_sq.sqrt())
                        grad_norms_tensor = torch.stack(per_task_grad_norms)
                        synced_norms = synced_per_task_tensor(
                            grad_norms_tensor, state=state, device=device
                        )
                    gradnorm_loss_tensor = balancer.step(synced_losses, synced_norms)
                    if state.enabled:
                        dist.broadcast(balancer.log_weights.data, src=0)

                    W_detached = balancer.task_weights_detached()
                    weighted_terms = [W_detached[i] * per_task_losses[i] for i in range(len(per_task_losses))]
                    loss = weighted_terms[0]
                    for term in weighted_terms[1:]:
                        loss = loss + term
                    surface_unweighted = per_task_losses[0]
                    for term in per_task_losses[1:4]:
                        surface_unweighted = surface_unweighted + term
                    base_mse_total = surface_unweighted + per_task_losses[4]
                    weighted_surface = (
                        W_detached[:4] * torch.stack([t.detach() for t in per_task_losses[:4]])
                    ).sum()
                    weighted_volume = W_detached[4] * per_task_losses[4].detach()
                    batch_loss_metrics = {
                        "base_mse_loss": float(base_mse_total.detach().cpu().item()),
                        "surface_loss": float(surface_unweighted.detach().cpu().item()),
                        "volume_loss": float(per_task_losses[4].detach().cpu().item()),
                        "surface_loss_weighted": float(weighted_surface.cpu().item()),
                        "volume_loss_weighted": float(weighted_volume.cpu().item()),
                    }
                    weights_cpu = W_detached.cpu().tolist()
                    losses_cpu = synced_losses.detach().cpu().tolist()
                    for i, name in enumerate(GRADNORM_TASK_NAMES):
                        gradnorm_metrics[f"gradnorm/weight_{name}"] = weights_cpu[i]
                        gradnorm_metrics[f"train/loss_{name}"] = losses_cpu[i]
                    if synced_norms is not None:
                        norms_cpu = synced_norms.detach().cpu().tolist()
                        for i, name in enumerate(GRADNORM_TASK_NAMES):
                            gradnorm_metrics[f"gradnorm/gnorm_{name}"] = norms_cpu[i]
                    if balancer.mode == "ema_proxy" and balancer.ema_loss is not None:
                        ema_cpu = balancer.ema_loss.detach().cpu().tolist()
                        for i, name in enumerate(GRADNORM_TASK_NAMES):
                            gradnorm_metrics[f"gradnorm/ema_loss_{name}"] = ema_cpu[i]
                        if balancer.initial_loss is not None:
                            r_vec = (
                                balancer.ema_loss / balancer.initial_loss.clamp(min=1e-12)
                            ).detach().cpu().tolist()
                            for i, name in enumerate(GRADNORM_TASK_NAMES):
                                gradnorm_metrics[f"gradnorm/r_{name}"] = r_vec[i]
                    gradnorm_metrics["gradnorm/ready"] = 1.0 if balancer.ready else 0.0
                    if balancer.initial_loss is not None:
                        for i, name in enumerate(GRADNORM_TASK_NAMES):
                            gradnorm_metrics[f"gradnorm/initial_loss_{name}"] = float(
                                balancer.initial_loss[i].cpu().item()
                            )
                    if balancer.mode == "ema_proxy" and balancer.min_weight > 0.0:
                        gradnorm_metrics["gradnorm/min_weight_floor_count"] = float(
                            balancer.last_floor_active
                        )
                    if gradnorm_loss_tensor is not None:
                        gradnorm_metrics["gradnorm/loss"] = float(gradnorm_loss_tensor.cpu().item())
                else:
                    surface_teacher_batch = getattr(batch, "surface_teacher", None)
                    volume_teacher_batch = getattr(batch, "volume_teacher", None)
                    loss, batch_loss_metrics = train_loss(
                        model,
                        batch,
                        transform,
                        device,
                        config.amp_mode,
                        surface_loss_weight=config.surface_loss_weight,
                        volume_loss_weight=config.volume_loss_weight,
                        surface_channel_weights=surface_channel_weights if use_channel_weights else None,
                        surface_teacher=surface_teacher_batch if kd_enabled else None,
                        volume_teacher=volume_teacher_batch if kd_enabled else None,
                        lambda_kd=config.distill_lambda_kd if kd_enabled else 0.0,
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
                    "train/vol_points": float(current_train_vol_points),
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
                            "train/tau_y_loss_weight": config.tau_y_loss_weight,
                            "train/tau_z_loss_weight": config.tau_z_loss_weight,
                        }
                    )
                    if "kd_loss" in batch_loss_metrics:
                        train_log.update(
                            {
                                "train/kd/loss": batch_loss_metrics["kd_loss"],
                                "train/kd/surface_loss": batch_loss_metrics["kd_surface_loss"],
                                "train/kd/volume_loss": batch_loss_metrics["kd_volume_loss"],
                                "train/kd/surface_loss_weighted": batch_loss_metrics[
                                    "kd_surface_loss_weighted"
                                ],
                                "train/kd/volume_loss_weighted": batch_loss_metrics[
                                    "kd_volume_loss_weighted"
                                ],
                                "train/kd/label_loss_weighted": batch_loss_metrics[
                                    "label_loss_weighted"
                                ],
                                "train/kd/lambda": batch_loss_metrics["lambda_kd"],
                            }
                        )
                if gradnorm_metrics:
                    train_log.update(gradnorm_metrics)

                if skip_step:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
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

            # Ease VRAM pressure before val to reduce OOM risk on the SOTA stack
            # (65k surface + 65k volume points / DDP=8). Logs peak memory pre/post-clear.
            if torch.cuda.is_available():
                pre_clear_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                torch.cuda.empty_cache()
                gc.collect()
                post_clear_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                if state.is_main:
                    log_metrics["val/pre_clear_reserved_gb"] = pre_clear_gb
                    log_metrics["val/post_clear_reserved_gb"] = post_clear_gb

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
                log_metrics.update(collect_string_sep_metrics(base_model))
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
