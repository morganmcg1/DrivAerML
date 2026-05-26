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
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
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
    load_curvature_stats,
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


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


@dataclass
class Config:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 50
    train_surface_points: int = 65_536
    eval_surface_points: int = 40_000
    train_volume_points: int = 16_384
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
    model_pe: str = "sincos"
    pe_num_features: int = 16
    pe_init_sigmas: str = "0.25,0.5,1.0,2.0,4.0"
    optimizer: str = "adamw"
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
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
    lr_warmup_steps: int = 0
    lr_warmup_start_lr: float = 1e-5
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
    seed: int = -1
    nonfinite_skip_abort: int = 200
    compile_model: bool = False
    debug: bool = False
    use_y_symmetry_aug: bool = False
    y_symmetry_aug_prob: float = 0.5
    eval_only: bool = False
    eval_checkpoint: str = ""
    use_gradnorm: bool = False
    gradnorm_alpha: float = 1.0
    gradnorm_lr: float = 1e-3
    gradnorm_min_w_vol_p: float = 0.0
    use_curvature_attention_bias: bool = False
    wss_charbonnier_weight: float = 0.0
    wss_charbonnier_eps: float = 1e-3
    wss_charbonnier_axes: str = "all"
    vol_p_charbonnier_weight: float = 0.0
    vol_p_charbonnier_eps: float = 1e-3
    tau_z_loss_weight: float = 1.0
    surface_out_width_factor: float = 1.0
    use_curvature_weighted_charb: bool = False
    curvature_weight_alpha: float = 1.0
    use_z_coord_wss_weight: bool = False
    z_coord_weight_alpha: float = 1.0


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
        "wss_charbonnier_axes": (
            "Which WSS channels the supplementary Charbonnier loss applies to. "
            "'all' = tau_x, tau_y, tau_z (H10 default). 'z' = tau_z only (H10b)."
        ),
        "surface_out_width_factor": (
            "Width multiplier for surface_out hidden layer (H39 PR #1284). "
            "1.0 = matched-width 2-layer head; 2.0 = wider head (Linear(h, 2h)->GELU->Linear(2h, 4))."
        ),
        "use_curvature_weighted_charb": (
            "H138: weight the WSS Charbonnier loss by per-point curvature. "
            "Requires --use-curvature-attention-bias (to load curvature data). "
            "w_i = 1 + alpha * kappa_mag_norm where kappa_mag_norm is the per-point "
            "curvature magnitude scaled to [0, 1] using raw_max from curvature_proxy_stats."
        ),
        "curvature_weight_alpha": (
            "H138: curvature weighting strength. w_i = 1 + alpha * kappa_mag_norm. "
            "alpha=0 reduces to uniform Charbonnier; alpha=1.0 gives high-curvature "
            "points up to 2x weight."
        ),
        "use_z_coord_wss_weight": (
            "H140: weight the WSS Charbonnier loss by per-point |z| coordinate. "
            "w_i = 1 + alpha * |z_i / z_max| where z_max is the per-batch max |z| "
            "over non-padded surface points. Compounds multiplicatively with "
            "--use-curvature-weighted-charb when both are enabled (H148)."
        ),
        "z_coord_weight_alpha": (
            "H140: z-coord weighting strength. w_i = 1 + alpha * |z_i/z_max|. "
            "alpha=0 reduces to uniform Charbonnier; alpha=1.0 gives high-|z| "
            "points up to 2x weight."
        ),
    }
    choices_text = {
        "wss_charbonnier_axes": ["all", "z"],
    }
    for field in fields(Config):
        value = getattr(defaults, field.name)
        arg_name = f"--{field.name.replace('_', '-')}"
        help_value = help_text.get(field.name)
        choices = choices_text.get(field.name)
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value, help=help_value)
            parser.add_argument(f"--no-{field.name.replace('_', '-')}", action="store_false", dest=field.name)
        else:
            parser.add_argument(
                arg_name,
                type=type(value),
                default=value,
                help=help_value,
                choices=choices,
            )
    namespace = parser.parse_args(argv)
    return Config(**vars(namespace))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def step_warmup_lr(config: Config, scheduled_lr: float, global_step: int) -> float:
    if config.lr_warmup_steps <= 0:
        return scheduled_lr
    step_index = max(global_step - 1, 0)
    if step_index >= config.lr_warmup_steps:
        return scheduled_lr
    progress = step_index / max(config.lr_warmup_steps, 1)
    return config.lr_warmup_start_lr + progress * (
        scheduled_lr - config.lr_warmup_start_lr
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def parse_pe_init_sigmas(spec: str) -> list[float] | None:
    spec = (spec or "").strip()
    if not spec:
        return None
    sigmas = [float(s) for s in spec.split(",") if s.strip()]
    return sigmas or None


class GradNormWeights(nn.Module):
    """Learnable per-task GradNorm weights (Chen et al. 2018, arXiv:1711.02257).

    Stores log-weights so the multiplicative weight ``w = exp(log_w)`` is
    always positive. Weights are renormalised externally to satisfy
    ``sum(w) = n_tasks``.
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.n_tasks = n_tasks
        self.log_weights = nn.Parameter(torch.zeros(n_tasks))

    @property
    def weights(self) -> torch.Tensor:
        return torch.exp(self.log_weights)


def per_channel_masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-channel masked MSE. Returns a [C] tensor (one scalar per output channel).

    The denominator is the spatial mask count, so summing the per-channel MSEs
    and dividing by ``C`` recovers the full ``masked_mse`` value.
    """
    mask_f = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)  # [B, N, 1]
    diff_sq = (pred - target).square()  # [B, N, C]
    weighted = diff_sq * mask_f  # broadcast [B, N, C]
    denom = mask_f.sum().clamp_min(1.0)
    return weighted.sum(dim=(0, 1)) / denom  # [C]


def masked_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Masked Charbonnier (pseudo-Huber) loss: ``sqrt(eps^2 + (pred-target)^2) - eps``.

    Averaged over masked spatial points and channels, matching the convention
    used by :func:`masked_mse`. Differentiable everywhere, gradient bounded by 1
    in magnitude — robust to outliers compared with squared error.
    """
    mask_f = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)  # [B, N, 1]
    delta = pred - target
    values = torch.sqrt(eps * eps + delta * delta) - eps  # [B, N, C]
    weighted = values * mask_f
    n_pts = mask_f.sum().clamp_min(1.0)
    n_ch = float(pred.shape[-1])
    return weighted.sum() / (n_pts * n_ch)


def compute_curvature_weights(
    surface_curvature_normed: torch.Tensor,
    kappa_mag_mean: float,
    kappa_mag_std: float,
    kappa_mag_raw_max: float,
    alpha: float,
) -> torch.Tensor:
    """H138: per-point Charbonnier-loss weights derived from kappa_mag.

    ``surface_curvature_normed`` is the z-scored curvature tensor of shape
    ``[B, N, 3]`` already attached to the batch (channels ``kappa_H``,
    ``kappa_G``, ``kappa_mag``). Undo the z-score on the third channel,
    min-max normalise into ``[0, 1]`` using ``raw_max`` (``raw_min`` is 0 for
    ``kappa_mag``), then form ``w_i = 1 + alpha * kappa_norm``.
    """
    z_kappa_mag = surface_curvature_normed[..., 2]  # [B, N]
    kappa_raw = z_kappa_mag * kappa_mag_std + kappa_mag_mean
    denom = max(float(kappa_mag_raw_max), 1e-6)
    kappa_norm = (kappa_raw / denom).clamp(min=0.0, max=1.0)
    return 1.0 + alpha * kappa_norm  # [B, N]


def compute_z_coord_weights(
    surface_x: torch.Tensor,
    surface_mask: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """H140: per-point WSS-loss weights derived from |z| coordinate.

    Returns ``(w_i, z_max_batch)`` where ``w_i`` has shape ``[B, N]`` with
    ``w_i = 1 + alpha * |z_i / z_max|`` and ``z_max_batch`` is the
    non-padded max |z| over the batch (scalar tensor, for diagnostics).

    Padded points (mask=False) contribute 0 to z_max and receive weight
    based on their |z|/z_max ratio — but they're zeroed out by the mask
    in the downstream loss, so their weight value is harmless.

    DrivAerML joint-view sampling can yield 0-surface views (when a case's
    volume_view_count > surface_view_count): in that case return a
    shape-preserving unit-weight tensor and a sentinel z_max so the
    downstream Charbonnier loss (denominator clamp_min(1.0)) still
    evaluates to 0 without crashing on an empty reduction.
    """
    z = surface_x[..., 2].abs()  # [B, N]
    if z.numel() == 0:
        return torch.ones_like(z), z.new_tensor(1e-6)
    mask_bool = surface_mask.to(dtype=torch.bool)
    masked_z = torch.where(mask_bool, z, torch.zeros_like(z))
    z_max = masked_z.max().clamp_min(1e-6)
    w = 1.0 + alpha * (z / z_max)  # [B, N]
    return w, z_max


def masked_weighted_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Per-point weighted Charbonnier loss (used by H138 + H140 + compound H148).

    ``weights`` has shape ``[B, N]`` (one positive scalar per surface point).
    The loss is a weighted mean over masked points / channels:

        loss = sum_{b,n,c} w[b,n] * mask[b,n] * charb[b,n,c]
               / ( sum_{b,n} w[b,n] * mask[b,n] * C )

    Reduces to :func:`masked_charbonnier` when all weights are equal.
    """
    mask_f = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)  # [B, N, 1]
    delta = pred - target
    values = torch.sqrt(eps * eps + delta * delta) - eps  # [B, N, C]
    w = weights.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)  # [B, N, 1]
    w_mask = w * mask_f
    weighted = values * w_mask
    n_ch = float(pred.shape[-1])
    denom = w_mask.sum().clamp_min(1.0) * n_ch
    return weighted.sum() / denom


def build_model(config: Config) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=config.model_layers,
        n_hidden=config.model_hidden_dim,
        dropout=config.model_dropout,
        n_head=config.model_heads,
        mlp_ratio=config.model_mlp_ratio,
        slice_num=config.model_slices,
        pe_kind=config.model_pe,
        pe_num_features=config.pe_num_features,
        pe_init_sigmas=parse_pe_init_sigmas(config.pe_init_sigmas),
        use_curvature_attention_bias=config.use_curvature_attention_bias,
        surface_out_width_factor=config.surface_out_width_factor,
    )


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
    raise ValueError(
        f"Unknown optimizer '{config.optimizer}'. Supported: adamw, lion."
    )


def apply_y_symmetry_aug(batch, prob: float) -> torch.Tensor:
    """Reflect a random subset of batch items through the y=0 plane.

    DrivAerML car geometry is bilaterally symmetric about y=0. Under y-reflection
    (x, y, z) -> (x, -y, z) the surface point coordinate y, the surface normal
    component n_y, the wall-shear y-component tau_y, and the volume point
    coordinate y all negate. cp, surface area, sdf, tau_x, tau_z, and volume
    pressure are invariant. This in-place transform is applied per-sample with
    probability `prob`. Returns the boolean mask used (for diagnostic logging).

    Channel layout (verified):
    - surface_x: [x, y, z, nx, ny, nz, area]  -> negate idx 1 and idx 4
    - surface_y: [cp, tau_x, tau_y, tau_z]    -> negate idx 2
    - volume_x:  [x, y, z, sdf]               -> negate idx 1
    - volume_y:  [pressure]                   -> invariant
    """
    B = batch.surface_x.shape[0]
    flip_mask = torch.rand(B, device=batch.surface_x.device) < prob
    if flip_mask.any():
        idx = flip_mask.nonzero(as_tuple=True)[0]
        batch.surface_x[idx, :, 1] = -batch.surface_x[idx, :, 1]
        batch.surface_x[idx, :, 4] = -batch.surface_x[idx, :, 4]
        batch.surface_y[idx, :, 2] = -batch.surface_y[idx, :, 2]
        batch.volume_x[idx, :, 1] = -batch.volume_x[idx, :, 1]
    return flip_mask


def train_loss(
    model: nn.Module,
    batch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    *,
    surface_loss_weight: float = 1.0,
    volume_loss_weight: float = 1.0,
    use_y_symmetry_aug: bool = False,
    y_symmetry_aug_prob: float = 0.5,
    aug_log: dict | None = None,
    gradnorm_weights: GradNormWeights | None = None,
    wss_charbonnier_weight: float = 0.0,
    wss_charbonnier_eps: float = 1e-3,
    wss_charbonnier_axes: str = "all",
    vol_p_charbonnier_weight: float = 0.0,
    vol_p_charbonnier_eps: float = 1e-3,
    tau_z_loss_weight: float = 1.0,
    use_curvature_weighted_charb: bool = False,
    curvature_weight_alpha: float = 1.0,
    curvature_kappa_mag_mean: float = 0.0,
    curvature_kappa_mag_std: float = 1.0,
    curvature_kappa_mag_raw_max: float = 1.0,
    use_z_coord_wss_weight: bool = False,
    z_coord_weight_alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
    surface_curvature = getattr(batch, "surface_curvature", None)
    batch = batch.to(device)
    if surface_curvature is not None:
        surface_curvature = surface_curvature.to(device)
        setattr(batch, "surface_curvature", surface_curvature)
    if use_y_symmetry_aug:
        flip_mask = apply_y_symmetry_aug(batch, y_symmetry_aug_prob)
        if aug_log is not None:
            aug_log["n_flipped"] = int(flip_mask.sum().item())
            aug_log["batch_size"] = int(flip_mask.shape[0])
    surface_target = transform.apply_surface(batch.surface_y)
    volume_target = transform.apply_volume(batch.volume_y)
    forward_kwargs: dict[str, torch.Tensor] = dict(
        surface_x=batch.surface_x,
        surface_mask=batch.surface_mask,
        volume_x=batch.volume_x,
        volume_mask=batch.volume_mask,
    )
    if surface_curvature is not None:
        forward_kwargs["surface_curvature"] = surface_curvature
    with autocast_context(device, amp_mode):
        out = model(**forward_kwargs)
        surface_loss = masked_mse(out["surface_preds"], surface_target, batch.surface_mask)
        volume_loss = masked_mse(out["volume_preds"], volume_target, batch.volume_mask)
        # H19 advisor fix: pre-compute Charbonnier on vol_p so it can be
        # passed to GradNorm as the vol_p task signal (the L0 anchor and
        # per-task gradient norms need to reflect the reshaped Charb
        # landscape, otherwise the H19 mechanism is invisible to the
        # dynamic-weight balancing — root cause flagged in run i4zt1zrl).
        if vol_p_charbonnier_weight > 0.0:
            pred_vol_p = out["volume_preds"]  # [B, N_vol, 1]
            target_vol_p = volume_target
            loss_vol_p_charb = masked_charbonnier(
                pred_vol_p,
                target_vol_p,
                batch.volume_mask,
                eps=vol_p_charbonnier_eps,
            )
            loss_vol_p_mse_diag = masked_mse(
                pred_vol_p, target_vol_p, batch.volume_mask
            )
        else:
            loss_vol_p_charb = None
            loss_vol_p_mse_diag = None
        if gradnorm_weights is not None:
            # H26: scale per-task losses by surface_loss_weight / volume_loss_weight
            # BEFORE concatenation. L0 (captured from task_losses on the first
            # step) absorbs the scaling so r-ratios cancel; the lever then acts
            # through gradient magnitudes (c_per_task / G_bar) and the total
            # backward signal, not through r-renormalisation.
            surface_per_ch = per_channel_masked_mse(
                out["surface_preds"], surface_target, batch.surface_mask
            ) * surface_loss_weight  # [4]: cp, tau_x, tau_y, tau_z
            # H36: Asymmetric tau_z boost. Multiply ONLY index 3 (tau_z) by
            # tau_z_loss_weight. Same pre-GradNorm injection point as H26 —
            # L0 absorbs the scaling so the lever acts through gradient
            # magnitudes (c_tau_z / G_bar) and the total backward signal,
            # without disturbing cp/tau_x/tau_y budgets.
            tau_z_scale = surface_per_ch.new_tensor(
                [1.0, 1.0, 1.0, tau_z_loss_weight]
            )
            surface_per_ch = surface_per_ch * tau_z_scale
            if loss_vol_p_charb is not None:
                # H19 advisor fix: vol_p slot carries the Charbonnier tensor
                # so GradNorm's L0 anchor and per-task gradient norms reflect
                # the reshape. The Charb contribution to the total loss is
                # now w_vol_p * Charb_vol_p via the weighted sum below —
                # adding a separate `vol_p_charb_weight * Charb` term would
                # double-count.
                volume_per_ch = loss_vol_p_charb.unsqueeze(0) * volume_loss_weight  # [1]
            else:
                volume_per_ch = per_channel_masked_mse(
                    out["volume_preds"], volume_target, batch.volume_mask
                ) * volume_loss_weight  # [1]: vol_p MSE
            task_losses = torch.cat([surface_per_ch, volume_per_ch])  # [5]
            w_detached = gradnorm_weights.weights.detach()
            loss = (w_detached * task_losses).sum()
            weighted_surface_loss = (w_detached[:4] * surface_per_ch).sum()
            weighted_volume_loss = w_detached[4] * volume_per_ch[0]
        else:
            task_losses = None
            weighted_surface_loss = surface_loss_weight * surface_loss
            weighted_volume_loss = volume_loss_weight * volume_loss
            loss = weighted_surface_loss + weighted_volume_loss
        base_mse_loss = surface_loss + volume_loss
        # H10b: Supplementary Charbonnier (pseudo-Huber) on WSS channels.
        # Channels 1-3 of surface_preds/target are tau_x, tau_y, tau_z (idx 0 is cp).
        # With axes="z", restrict to channel 3 (tau_z) — the axis with the
        # largest val→test gap in H10, while letting τ_x/τ_y keep H9's MSE.
        if wss_charbonnier_weight > 0.0:
            if wss_charbonnier_axes == "z":
                pred_wss = out["surface_preds"][..., 3:4]
                target_wss = surface_target[..., 3:4]
            elif wss_charbonnier_axes == "all":
                pred_wss = out["surface_preds"][..., 1:4]
                target_wss = surface_target[..., 1:4]
            else:
                raise ValueError(
                    f"Unknown wss_charbonnier_axes={wss_charbonnier_axes}"
                )
            # H138: per-point curvature weights from kappa_mag. H140: per-point
            # |z|-coord weights. H148: compound (multiplicative product) when
            # both flags are enabled. Each weight independently rescales the
            # per-point Charbonnier loss; multiplicative composition stacks
            # emphasis on regions that are high under BOTH scalars.
            curv_weights: torch.Tensor | None = None
            z_weights: torch.Tensor | None = None
            z_max_batch: torch.Tensor | None = None
            if use_curvature_weighted_charb and surface_curvature is not None:
                curv_weights = compute_curvature_weights(
                    surface_curvature,
                    kappa_mag_mean=curvature_kappa_mag_mean,
                    kappa_mag_std=curvature_kappa_mag_std,
                    kappa_mag_raw_max=curvature_kappa_mag_raw_max,
                    alpha=curvature_weight_alpha,
                )
            if use_z_coord_wss_weight:
                z_weights, z_max_batch = compute_z_coord_weights(
                    batch.surface_x,
                    batch.surface_mask,
                    alpha=z_coord_weight_alpha,
                )
            if curv_weights is not None and z_weights is not None:
                compound_weights = curv_weights * z_weights
                loss_wss_charb = masked_weighted_charbonnier(
                    pred_wss, target_wss, batch.surface_mask, compound_weights,
                    eps=wss_charbonnier_eps,
                )
            elif curv_weights is not None:
                loss_wss_charb = masked_weighted_charbonnier(
                    pred_wss, target_wss, batch.surface_mask, curv_weights,
                    eps=wss_charbonnier_eps,
                )
            elif z_weights is not None:
                loss_wss_charb = masked_weighted_charbonnier(
                    pred_wss, target_wss, batch.surface_mask, z_weights,
                    eps=wss_charbonnier_eps,
                )
            else:
                loss_wss_charb = masked_charbonnier(
                    pred_wss, target_wss, batch.surface_mask,
                    eps=wss_charbonnier_eps,
                )
            loss_wss_mse_diag = masked_mse(pred_wss, target_wss, batch.surface_mask)
            loss = loss + wss_charbonnier_weight * loss_wss_charb
            charb_metrics = {
                "loss_wss_charb_unweighted": float(
                    loss_wss_charb.detach().cpu().item()
                ),
                "loss_wss_charb_weighted": float(
                    (wss_charbonnier_weight * loss_wss_charb).detach().cpu().item()
                ),
                "loss_wss_charb_target_mse": float(
                    loss_wss_mse_diag.detach().cpu().item()
                ),
            }
            mask_bool_diag = batch.surface_mask.to(dtype=torch.bool)
            if curv_weights is not None:
                w_masked = curv_weights.detach()[mask_bool_diag]
                if w_masked.numel() > 0:
                    charb_metrics["curv_w_mean"] = float(w_masked.mean().cpu().item())
                    charb_metrics["curv_w_max"] = float(w_masked.max().cpu().item())
                    charb_metrics["curv_w_min"] = float(w_masked.min().cpu().item())
                    charb_metrics["curv_w_p50"] = float(
                        torch.median(w_masked).cpu().item()
                    )
                    charb_metrics["curv_w_p99"] = float(
                        torch.quantile(w_masked.float(), 0.99).cpu().item()
                    )
                    charb_metrics["curv_w_above_1p5_frac"] = float(
                        (w_masked > 1.5).float().mean().cpu().item()
                    )
            if z_weights is not None:
                w_masked = z_weights.detach()[mask_bool_diag]
                if w_masked.numel() > 0:
                    charb_metrics["z_w_mean"] = float(w_masked.mean().cpu().item())
                    charb_metrics["z_w_max"] = float(w_masked.max().cpu().item())
                    charb_metrics["z_w_min"] = float(w_masked.min().cpu().item())
                    charb_metrics["z_w_p50"] = float(
                        torch.median(w_masked).cpu().item()
                    )
                    charb_metrics["z_w_p99"] = float(
                        torch.quantile(w_masked.float(), 0.99).cpu().item()
                    )
                    charb_metrics["z_w_above_1p5_frac"] = float(
                        (w_masked > 1.5).float().mean().cpu().item()
                    )
                if z_max_batch is not None:
                    charb_metrics["z_max_batch"] = float(z_max_batch.detach().cpu().item())
            if curv_weights is not None and z_weights is not None:
                w_masked = (curv_weights.detach() * z_weights.detach())[mask_bool_diag]
                if w_masked.numel() > 0:
                    charb_metrics["compound_w_mean"] = float(w_masked.mean().cpu().item())
                    charb_metrics["compound_w_max"] = float(w_masked.max().cpu().item())
                    charb_metrics["compound_w_p99"] = float(
                        torch.quantile(w_masked.float(), 0.99).cpu().item()
                    )
        else:
            charb_metrics = {}
        # H19: vol_p Charbonnier — under GradNorm, the Charb tensor is already
        # in task_losses[4] above, contributing w_vol_p * Charb_vol_p to the
        # weighted sum. Adding an extra additive term would double-count.
        # Without GradNorm we fall back to the additive form (loss += w * Charb)
        # so the mechanism still fires.
        if loss_vol_p_charb is not None:
            if gradnorm_weights is None:
                loss = loss + vol_p_charbonnier_weight * loss_vol_p_charb
            vol_p_charb_metrics = {
                "loss_vol_p_charb_unweighted": float(
                    loss_vol_p_charb.detach().cpu().item()
                ),
                "loss_vol_p_charb_weighted": float(
                    (vol_p_charbonnier_weight * loss_vol_p_charb).detach().cpu().item()
                ),
                "loss_vol_p_charb_target_mse": float(
                    loss_vol_p_mse_diag.detach().cpu().item()
                ),
            }
        else:
            vol_p_charb_metrics = {}
    metrics = {
        "base_mse_loss": float(base_mse_loss.detach().cpu().item()),
        "surface_loss": float(surface_loss.detach().cpu().item()),
        "volume_loss": float(volume_loss.detach().cpu().item()),
        "surface_loss_weighted": float(weighted_surface_loss.detach().cpu().item()),
        "volume_loss_weighted": float(weighted_volume_loss.detach().cpu().item()),
    }
    metrics.update(charb_metrics)
    metrics.update(vol_p_charb_metrics)
    return loss, metrics, task_losses


def run_eval_only(config: Config, state) -> None:
    """Load a saved checkpoint and run full val + test evaluation."""
    if config.seed >= 0:
        seed_everything(config.seed)
    device = state.device
    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"[eval-only] Device: {device}{ddp_suffix}")
        print(f"[eval-only] Checkpoint: {config.eval_checkpoint}")

    train_loader, val_loaders, test_loaders, stats = make_loaders(
        config, distributed_state=state
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
    if state.enabled:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs = {"device_ids": [state.local_rank], "output_device": state.local_rank}
        model = DistributedDataParallel(model, **ddp_kwargs)
    base_model = unwrap_model(model)

    run = init_wandb_run(
        config=config,
        state=state,
        n_params=n_params,
        train_loader=train_loader,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        total_estimated_steps=1,
        max_epochs=0,
        train_timeout_minutes=0.0,
        val_budget_minutes=0.0,
    )

    distributed_barrier(state)
    if not state.is_main:
        wandb.finish()
        return

    ckpt_path = Path(config.eval_checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    val_metrics = checkpoint["val_metrics"]["val_surface"]
    best_metrics = {
        "epoch": float(checkpoint["epoch"]),
        "abupt_axis_mean_rel_l2_pct": val_metrics["abupt_axis_mean_rel_l2_pct"],
        "surface_pressure_mae": val_metrics["surface_pressure_mae"],
        "wall_shear_mae": val_metrics["wall_shear_mae"],
        "volume_pressure_mae": val_metrics["volume_pressure_mae"],
    }
    config_path = ckpt_path.parent / "config.yaml"
    if not config_path.exists():
        with config_path.open("w") as f:
            yaml.safe_dump(asdict(config), f)

    run_final_evaluation(
        run=run,
        model=base_model,
        model_path=ckpt_path,
        config_path=config_path,
        config=config,
        transform=transform,
        device=device,
        final_val_loaders=final_val_loaders,
        final_test_loaders=final_test_loaders,
        best_metrics=best_metrics,
        best_checkpoint_source=checkpoint.get("checkpoint_source", "eval-only"),
        n_params=n_params,
        global_step=int(checkpoint["epoch"]),
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
        if config.seed >= 0:
            seed_everything(config.seed)
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

        # H138/H148: load kappa_mag stats for the per-point curvature-weighted
        # Charbonnier loss. The CurvatureAugmentedCaseStore z-scores the
        # curvature on-the-fly; we need the raw scale (mean, std, raw_max) to
        # undo the z-score and min-max normalise into [0, 1] for the weight.
        curv_kappa_mag_mean = 0.0
        curv_kappa_mag_std = 1.0
        curv_kappa_mag_raw_max = 1.0
        if config.use_curvature_weighted_charb:
            if not config.use_curvature_attention_bias:
                raise ValueError(
                    "--use-curvature-weighted-charb requires "
                    "--use-curvature-attention-bias (curvature data is loaded "
                    "via CurvatureAugmentedCaseStore which is gated by that flag)."
                )
            curv_stats_full = load_curvature_stats()
            # Channel ordering: ["kappa_H", "kappa_G", "kappa_mag"] — kappa_mag is idx 2.
            curv_kappa_mag_mean = float(curv_stats_full["mean"][2])
            curv_kappa_mag_std = float(curv_stats_full["std"][2])
            curv_kappa_mag_raw_max = float(curv_stats_full["raw_max"][2])
            if state.is_main:
                print(
                    f"H138 curvature-weighted Charb: alpha={config.curvature_weight_alpha}, "
                    f"kappa_mag mean={curv_kappa_mag_mean:.4f}, "
                    f"std={curv_kappa_mag_std:.4f}, "
                    f"raw_max={curv_kappa_mag_raw_max:.4f}"
                )
        if config.use_z_coord_wss_weight and state.is_main:
            print(
                f"H140 z-coord WSS weight: alpha={config.z_coord_weight_alpha} "
                f"(applied to WSS Charbonnier axes={config.wss_charbonnier_axes})"
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

        optimizer = build_optimizer(base_model, config)
        scheduler = build_lr_scheduler(optimizer, config, max_epochs)
        ema = EMA(base_model, decay=config.ema_decay, start_step=config.ema_start_step) if config.use_ema else None

        gradnorm_weights: GradNormWeights | None = None
        gradnorm_opt: torch.optim.Optimizer | None = None
        gradnorm_L0: torch.Tensor | None = None
        gradnorm_n_tasks = 5  # cp, tau_x, tau_y, tau_z, vol_p
        gradnorm_task_names = ("cp", "tau_x", "tau_y", "tau_z", "vol_p")
        if config.use_gradnorm:
            gradnorm_weights = GradNormWeights(n_tasks=gradnorm_n_tasks).to(device)
            gradnorm_opt = torch.optim.Adam(
                gradnorm_weights.parameters(), lr=config.gradnorm_lr
            )
            if state.is_main:
                print(
                    f"GradNorm enabled: alpha={config.gradnorm_alpha}, "
                    f"n_tasks={gradnorm_n_tasks}, lr={config.gradnorm_lr}"
                )
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
        if state.is_main:
            pe_class = type(base_model.pos_embed).__name__
            run.config.update(
                {
                    "pe_class": pe_class,
                    "pe_source_branch": (
                        "origin/tay" if config.model_pe == "string_multisigma" else "n/a"
                    ),
                    "pe_source_sha": (
                        "d97fb09" if config.model_pe == "string_multisigma" else "n/a"
                    ),
                    "pe_source_run_ki2q9ko9": (
                        config.model_pe == "string_multisigma"
                    ),
                },
                allow_val_change=True,
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
        nonfinite_skip_count = 0
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
                aug_log: dict[str, int] = {}
                loss, batch_loss_metrics, task_losses = train_loss(
                    model,
                    batch,
                    transform,
                    device,
                    config.amp_mode,
                    surface_loss_weight=config.surface_loss_weight,
                    volume_loss_weight=config.volume_loss_weight,
                    use_y_symmetry_aug=config.use_y_symmetry_aug,
                    y_symmetry_aug_prob=config.y_symmetry_aug_prob,
                    aug_log=aug_log,
                    gradnorm_weights=gradnorm_weights,
                    wss_charbonnier_weight=config.wss_charbonnier_weight,
                    wss_charbonnier_eps=config.wss_charbonnier_eps,
                    wss_charbonnier_axes=config.wss_charbonnier_axes,
                    vol_p_charbonnier_weight=config.vol_p_charbonnier_weight,
                    vol_p_charbonnier_eps=config.vol_p_charbonnier_eps,
                    tau_z_loss_weight=config.tau_z_loss_weight,
                    use_curvature_weighted_charb=config.use_curvature_weighted_charb,
                    curvature_weight_alpha=config.curvature_weight_alpha,
                    curvature_kappa_mag_mean=curv_kappa_mag_mean,
                    curvature_kappa_mag_std=curv_kappa_mag_std,
                    curvature_kappa_mag_raw_max=curv_kappa_mag_raw_max,
                    use_z_coord_wss_weight=config.use_z_coord_wss_weight,
                    z_coord_weight_alpha=config.z_coord_weight_alpha,
                )
                if (
                    config.use_y_symmetry_aug
                    and state.is_main
                    and epoch == 0
                    and global_step == 0
                ):
                    print(
                        f"[aug debug] EP0 step0 (rank0): "
                        f"{aug_log.get('n_flipped', 0)}/{aug_log.get('batch_size', 0)} "
                        f"samples y-flipped"
                    )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                scheduled_lr = scheduler.get_last_lr()[0]
                current_lr = step_warmup_lr(config, scheduled_lr, global_step)
                loss_is_nonfinite = not bool(torch.isfinite(loss.detach()).item())
                loss_skip_step = distributed_any(state, loss_is_nonfinite, device)
                skip_step = loss_skip_step
                nonfinite_skip_kind = 1 if loss_skip_step else 0
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
                    "train/nonfinite_loss": 1.0 if loss_skip_step else 0.0,
                }
                if not loss_skip_step:
                    train_log.update(
                        {
                            "train/loss": float(loss.detach().cpu().item()),
                            "train/base_mse_loss": batch_loss_metrics["base_mse_loss"],
                            "train/surface_loss": batch_loss_metrics["surface_loss"],
                            "train/volume_loss": batch_loss_metrics["volume_loss"],
                            "train/surface_loss_weighted": batch_loss_metrics["surface_loss_weighted"],
                            "train/volume_loss_weighted": batch_loss_metrics["volume_loss_weighted"],
                            "train/tau_z_loss_weight": config.tau_z_loss_weight,
                        }
                    )
                    if "loss_wss_charb_unweighted" in batch_loss_metrics:
                        # H10b diagnostics: key suffix follows --wss-charbonnier-axes
                        # so dashboards can compare "wss" (all-axes) vs "tau_z" (z-only).
                        wss_axes_label = (
                            "tau_z"
                            if config.wss_charbonnier_axes == "z"
                            else "wss"
                        )
                        target_mse = batch_loss_metrics["loss_wss_charb_target_mse"]
                        charb_unw = batch_loss_metrics["loss_wss_charb_unweighted"]
                        charb_w = batch_loss_metrics["loss_wss_charb_weighted"]
                        ratio_to_mse = (
                            charb_unw / target_mse
                            if target_mse > 1e-12
                            else float("nan")
                        )
                        weighted_ratio = (
                            charb_w / target_mse
                            if target_mse > 1e-12
                            else float("nan")
                        )
                        train_log.update(
                            {
                                f"train/loss_{wss_axes_label}_mse": target_mse,
                                f"train/loss_{wss_axes_label}_charb_unweighted": charb_unw,
                                f"train/loss_{wss_axes_label}_charb_weighted": charb_w,
                                f"train/loss_{wss_axes_label}_charb_to_mse_ratio": ratio_to_mse,
                                f"train/loss_{wss_axes_label}_charb_weighted_to_mse_ratio": weighted_ratio,
                            }
                        )
                        # H138 curvature-weight diagnostics
                        if "curv_w_mean" in batch_loss_metrics:
                            train_log.update(
                                {
                                    "train/curv_charb/w_mean": batch_loss_metrics["curv_w_mean"],
                                    "train/curv_charb/w_max": batch_loss_metrics["curv_w_max"],
                                    "train/curv_charb/w_min": batch_loss_metrics["curv_w_min"],
                                    "train/curv_charb/w_p50": batch_loss_metrics["curv_w_p50"],
                                    "train/curv_charb/w_p99": batch_loss_metrics["curv_w_p99"],
                                    "train/curv_charb/w_above_1p5_frac": batch_loss_metrics[
                                        "curv_w_above_1p5_frac"
                                    ],
                                    "train/curv_charb/alpha": config.curvature_weight_alpha,
                                }
                            )
                        # H140 z-coord-weight diagnostics
                        if "z_w_mean" in batch_loss_metrics:
                            train_log.update(
                                {
                                    "train/z_weight/w_mean": batch_loss_metrics["z_w_mean"],
                                    "train/z_weight/w_max": batch_loss_metrics["z_w_max"],
                                    "train/z_weight/w_min": batch_loss_metrics["z_w_min"],
                                    "train/z_weight/w_p50": batch_loss_metrics["z_w_p50"],
                                    "train/z_weight/w_p99": batch_loss_metrics["z_w_p99"],
                                    "train/z_weight/w_above_1p5_frac": batch_loss_metrics[
                                        "z_w_above_1p5_frac"
                                    ],
                                    "train/z_weight/alpha": config.z_coord_weight_alpha,
                                }
                            )
                            if "z_max_batch" in batch_loss_metrics:
                                train_log["train/z_weight/z_max_batch"] = batch_loss_metrics[
                                    "z_max_batch"
                                ]
                        # H148 compound-weight diagnostics (when both enabled)
                        if "compound_w_mean" in batch_loss_metrics:
                            train_log.update(
                                {
                                    "train/compound_w/w_mean": batch_loss_metrics["compound_w_mean"],
                                    "train/compound_w/w_max": batch_loss_metrics["compound_w_max"],
                                    "train/compound_w/w_p99": batch_loss_metrics["compound_w_p99"],
                                }
                            )
                    if "loss_vol_p_charb_unweighted" in batch_loss_metrics:
                        vol_p_target_mse = batch_loss_metrics["loss_vol_p_charb_target_mse"]
                        vol_p_charb_unw = batch_loss_metrics["loss_vol_p_charb_unweighted"]
                        vol_p_charb_w = batch_loss_metrics["loss_vol_p_charb_weighted"]
                        vol_p_ratio_to_mse = (
                            vol_p_charb_unw / vol_p_target_mse
                            if vol_p_target_mse > 1e-12
                            else float("nan")
                        )
                        vol_p_weighted_ratio = (
                            vol_p_charb_w / vol_p_target_mse
                            if vol_p_target_mse > 1e-12
                            else float("nan")
                        )
                        train_log.update(
                            {
                                "train/loss_vol_p_mse": vol_p_target_mse,
                                "train/loss_vol_p_charb_unweighted": vol_p_charb_unw,
                                "train/loss_vol_p_charb_weighted": vol_p_charb_w,
                                "train/loss_vol_p_charb_to_mse_ratio": vol_p_ratio_to_mse,
                                "train/loss_vol_p_charb_weighted_to_mse_ratio": vol_p_weighted_ratio,
                            }
                        )
                if config.use_y_symmetry_aug and aug_log:
                    bs = max(1, aug_log.get("batch_size", 0))
                    train_log["train/aug/y_symmetry_flip_rate"] = (
                        aug_log.get("n_flipped", 0) / bs
                    )
                    train_log["train/aug/y_symmetry_n_flipped"] = aug_log.get(
                        "n_flipped", 0
                    )
                    train_log["train/aug/y_symmetry_batch_size"] = aug_log.get(
                        "batch_size", 0
                    )

                if skip_step:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    gradnorm_log: dict[str, float] = {}
                    if (
                        gradnorm_weights is not None
                        and gradnorm_opt is not None
                        and task_losses is not None
                    ):
                        task_losses_finite = bool(
                            torch.isfinite(task_losses.detach()).all().item()
                        )
                        if task_losses_finite:
                            if gradnorm_L0 is None:
                                gradnorm_L0 = task_losses.detach().clone()
                            shared_param = base_model.norm.weight
                            c_per_task: list[torch.Tensor] = []
                            for i in range(gradnorm_n_tasks):
                                g_i = torch.autograd.grad(
                                    task_losses[i],
                                    shared_param,
                                    retain_graph=True,
                                    allow_unused=True,
                                )[0]
                                if g_i is None:
                                    c_per_task.append(
                                        torch.zeros((), device=device, dtype=torch.float32)
                                    )
                                else:
                                    c_per_task.append(g_i.detach().float().norm(2))
                            c_tensor = torch.stack(c_per_task)  # [n_tasks]
                            with torch.no_grad():
                                w_now = gradnorm_weights.weights
                                G_per_task = w_now * c_tensor  # [n_tasks]
                                G_bar = G_per_task.mean()
                                L_ratio = (
                                    task_losses.detach().float()
                                    / gradnorm_L0.float().clamp_min(1e-12)
                                )
                                r = L_ratio / L_ratio.mean().clamp_min(1e-12)
                                target = G_bar * r.pow(config.gradnorm_alpha)
                                diff = G_per_task - target
                                # d L_grad / d log_w_i = sign(diff_i) * G_per_task_i
                                log_w_grad = torch.sign(diff) * G_per_task
                                # safety: zero out non-finite components
                                log_w_grad = torch.where(
                                    torch.isfinite(log_w_grad),
                                    log_w_grad,
                                    torch.zeros_like(log_w_grad),
                                )
                            gradnorm_opt.zero_grad(set_to_none=True)
                            gradnorm_weights.log_weights.grad = log_w_grad
                            gradnorm_opt.step()
                            with torch.no_grad():
                                # renormalise so weights sum to n_tasks
                                lw = gradnorm_weights.log_weights.data
                                lw_norm = lw - torch.logsumexp(lw, dim=0) + math.log(
                                    float(gradnorm_n_tasks)
                                )
                                gradnorm_weights.log_weights.data.copy_(lw_norm)
                                # H9: hard floor on vol_p task weight to prevent
                                # GradNorm starvation (root cause of vol_p breach
                                # in H1/H2/H3/H5). Indices: cp, tau_x, tau_y,
                                # tau_z, vol_p.
                                w_vol_p_clamp_active = False
                                if config.gradnorm_min_w_vol_p > 0.0:
                                    w_curr = gradnorm_weights.weights.detach()
                                    vol_p_idx = 4
                                    min_w = config.gradnorm_min_w_vol_p
                                    if w_curr[vol_p_idx].item() < min_w:
                                        w_vol_p_clamp_active = True
                                        other_idx = torch.tensor(
                                            [0, 1, 2, 3], device=w_curr.device
                                        )
                                        deficit = min_w - w_curr[vol_p_idx].item()
                                        other_sum = w_curr[other_idx].sum().item()
                                        scale = max(
                                            (other_sum - deficit) / max(other_sum, 1e-12),
                                            0.01,
                                        )
                                        new_w = w_curr.clone()
                                        new_w[vol_p_idx] = min_w
                                        new_w[other_idx] = w_curr[other_idx] * scale
                                        gradnorm_weights.log_weights.data.copy_(
                                            new_w.clamp_min(1e-12).log()
                                        )
                                # DDP: average log_weights across ranks for sync
                                if state.enabled:
                                    torch.distributed.all_reduce(
                                        gradnorm_weights.log_weights.data,
                                        op=torch.distributed.ReduceOp.SUM,
                                    )
                                    gradnorm_weights.log_weights.data.div_(
                                        state.world_size
                                    )
                                w_post = gradnorm_weights.weights.detach().cpu()
                                c_cpu = c_tensor.detach().cpu()
                                G_cpu = G_per_task.detach().cpu()
                                tl_cpu = task_losses.detach().float().cpu()
                                r_cpu = r.detach().cpu()
                                for ti, name in enumerate(gradnorm_task_names):
                                    gradnorm_log[f"gradnorm/w_{name}"] = float(
                                        w_post[ti].item()
                                    )
                                    gradnorm_log[f"gradnorm/grad_norm_{name}"] = float(
                                        c_cpu[ti].item()
                                    )
                                    gradnorm_log[f"gradnorm/G_{name}"] = float(
                                        G_cpu[ti].item()
                                    )
                                    gradnorm_log[f"gradnorm/task_loss_{name}"] = float(
                                        tl_cpu[ti].item()
                                    )
                                    gradnorm_log[f"gradnorm/r_{name}"] = float(
                                        r_cpu[ti].item()
                                    )
                                gradnorm_log["gradnorm/G_bar"] = float(
                                    G_bar.detach().cpu().item()
                                )
                                if config.gradnorm_min_w_vol_p > 0.0:
                                    gradnorm_log["gradnorm/w_vol_p_clamp_active"] = float(
                                        w_vol_p_clamp_active
                                    )
                                    gradnorm_log["gradnorm/min_w_vol_p"] = float(
                                        config.gradnorm_min_w_vol_p
                                    )
                    if gradnorm_log:
                        train_log.update(gradnorm_log)
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
                    grad_skip_step = distributed_any(state, grad_is_nonfinite, device)
                    skip_step = grad_skip_step
                    if grad_skip_step:
                        nonfinite_skip_kind = 2
                    clipped = (
                        config.grad_clip_norm > 0.0
                        and math.isfinite(grad_norm_pre_clip)
                        and grad_norm_pre_clip > config.grad_clip_norm
                    )
                    train_log.update(
                        {
                            "train/grad/global_norm_pre_clip": grad_norm_pre_clip,
                            "train/grad/clipped": 1.0 if clipped else 0.0,
                            "train/nonfinite_grad": 1.0 if grad_skip_step else 0.0,
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
                        set_optimizer_lr(optimizer, current_lr)
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

                if skip_step:
                    nonfinite_skip_count += 1
                train_log["train/nonfinite_skip_count"] = nonfinite_skip_count
                train_log["train/nonfinite_skip_kind"] = nonfinite_skip_kind
                abort_for_nonfinite = (
                    config.nonfinite_skip_abort >= 0
                    and nonfinite_skip_count > config.nonfinite_skip_abort
                )
                if abort_for_nonfinite:
                    train_log["train/nonfinite_abort"] = 1.0

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
                if abort_for_nonfinite:
                    raise RuntimeError(
                        f"Aborting after {nonfinite_skip_count} non-finite loss/grad skips; "
                        "training is structurally broken."
                    )
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
