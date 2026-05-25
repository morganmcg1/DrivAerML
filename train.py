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
    use_imtl_g: bool = False
    imtl_g_eps: float = 1e-4
    imtl_g_clamp_negative: bool = True
    use_curvature_attention_bias: bool = False
    wss_charbonnier_weight: float = 0.0
    wss_charbonnier_eps: float = 1e-3
    wss_charbonnier_axes: str = "all"
    vol_p_charbonnier_weight: float = 0.0
    vol_p_charbonnier_eps: float = 1e-3
    tau_z_loss_weight: float = 1.0
    surface_out_width_factor: float = 1.0


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


def imtl_g_weights(
    grads: list[torch.Tensor],
    *,
    eps: float = 1e-4,
    clamp_negative: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Closed-form IMTL-G weights (Liu et al. 2021, arXiv:2108.01547).

    The "impartial" gradient direction makes the combined gradient project
    equally onto each unit-normalised task gradient. Uses the reference-task
    offset system (eq. 8 of the paper) which is the canonical formulation.

    Args:
        grads: list of T flattened gradient tensors (each shape ``[P]``) of the
            per-task losses w.r.t. a shared backbone parameter.
        eps: Tikhonov regularisation added to the diagonal of the (T-1)x(T-1)
            offset Gram matrix before solving.
        clamp_negative: clamp negative weights to 0 and renormalise. The paper
            does not clamp, but clamping is widely used in practice to avoid
            anti-correlated task subtraction destabilising early training.

    Returns:
        (alpha, diag) where alpha is a [T,] tensor summing to 1, non-negative
        if ``clamp_negative``; diag is a metrics dict (cond, alpha_min,
        alpha_max, neg_count, solve_failed).
    """
    T = len(grads)
    device = grads[0].device
    dtype = torch.float32
    diag: dict[str, float] = {
        "imtl_g/solve_failed": 0.0,
        "imtl_g/neg_count": 0.0,
    }
    if T < 2:
        alpha = torch.ones(T, device=device, dtype=dtype) / max(T, 1)
        diag["imtl_g/cond_number"] = 1.0
        return alpha, diag

    G = torch.stack([g.detach().to(dtype) for g in grads])  # [T, P]
    g_norms = G.norm(dim=1, keepdim=True).clamp_min(1e-8)
    U = G / g_norms                                          # unit gradients [T, P]

    # Reference-task offset system (task 0 is reference)
    D = G[0:1] - G[1:]                                       # [T-1, P]
    U_diff = U[0:1] - U[1:]                                  # [T-1, P]

    M = D @ U_diff.T                                          # [T-1, T-1]
    M_reg = M + eps * torch.eye(T - 1, device=device, dtype=dtype)

    # Condition diagnostic (used for fallback)
    try:
        cond = float(torch.linalg.cond(M_reg).item())
    except Exception:
        cond = float("inf")
    diag["imtl_g/cond_number"] = cond

    proj = (G[0] @ U_diff.T).unsqueeze(-1)                    # [T-1, 1]
    try:
        alpha_rest = torch.linalg.solve(M_reg, proj).squeeze(-1)  # [T-1]
        if not torch.isfinite(alpha_rest).all():
            raise RuntimeError("non-finite alpha")
    except Exception:
        diag["imtl_g/solve_failed"] = 1.0
        alpha = torch.ones(T, device=device, dtype=dtype) / T
        diag["imtl_g/alpha_min"] = float(alpha.min().item())
        diag["imtl_g/alpha_max"] = float(alpha.max().item())
        return alpha, diag

    alpha_0 = 1.0 - alpha_rest.sum()
    alpha = torch.cat([alpha_0.unsqueeze(0), alpha_rest])     # [T]

    diag["imtl_g/neg_count"] = float((alpha < 0).sum().item())
    if clamp_negative:
        alpha = alpha.clamp(min=0.0)
    s = alpha.sum().clamp_min(1e-8)
    alpha = alpha / s
    diag["imtl_g/alpha_min"] = float(alpha.min().item())
    diag["imtl_g/alpha_max"] = float(alpha.max().item())
    return alpha, diag


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
    use_imtl_g: bool = False,
    wss_charbonnier_weight: float = 0.0,
    wss_charbonnier_eps: float = 1e-3,
    wss_charbonnier_axes: str = "all",
    vol_p_charbonnier_weight: float = 0.0,
    vol_p_charbonnier_eps: float = 1e-3,
    tau_z_loss_weight: float = 1.0,
) -> tuple[
    torch.Tensor | None,
    dict[str, float],
    torch.Tensor | None,
    torch.Tensor | None,
]:
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
        if use_imtl_g:
            # H136: IMTL-G uses task_losses as input to the closed-form
            # weighting (computed outside train_loss). We preserve the same
            # 5-task structure used by H39's GradNorm (cp, tau_x, tau_y, tau_z,
            # vol_p) so the gradient-balance failure mode is visible per axis.
            surface_per_ch = per_channel_masked_mse(
                out["surface_preds"], surface_target, batch.surface_mask
            ) * surface_loss_weight  # [4]: cp, tau_x, tau_y, tau_z
            # H36: Asymmetric tau_z boost (same injection point as H26/GradNorm
            # variant — IMTL-G then balances tasks impartially around this base
            # scaling).
            tau_z_scale = surface_per_ch.new_tensor(
                [1.0, 1.0, 1.0, tau_z_loss_weight]
            )
            surface_per_ch = surface_per_ch * tau_z_scale
            if loss_vol_p_charb is not None:
                # H19 advisor fix: vol_p slot carries the Charbonnier tensor
                # so IMTL-G's per-task gradient reflects the reshaped landscape.
                # The Charb contribution to the total loss is now
                # alpha_vol_p * Charb_vol_p via the weighted sum the caller
                # forms — no separate additive term to avoid double-count.
                volume_per_ch = loss_vol_p_charb.unsqueeze(0) * volume_loss_weight  # [1]
            else:
                volume_per_ch = per_channel_masked_mse(
                    out["volume_preds"], volume_target, batch.volume_mask
                ) * volume_loss_weight  # [1]: vol_p MSE
            task_losses = torch.cat([surface_per_ch, volume_per_ch])  # [5]
            # The caller (training loop) will compute alpha from per-task
            # gradients and form weighted_loss = sum(alpha_detached * task_losses).
            # We return loss=None to signal that.
            loss = None
            # Metrics: surface/volume aggregates using uniform task weight
            # (alpha not yet known; logged values reflect raw per-task levels).
            weighted_surface_loss = surface_per_ch.sum()
            weighted_volume_loss = volume_per_ch.sum()
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
        # Supplementary additive WSS Charbonnier (H10b). Kept as an additive
        # term (separate from IMTL-G task list) to preserve H39 baseline
        # behaviour: the supplementary acts as a fixed extra gradient pull on
        # tau_z on top of whatever IMTL-G balances across the 5 base tasks.
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
            loss_wss_charb = masked_charbonnier(
                pred_wss, target_wss, batch.surface_mask, eps=wss_charbonnier_eps
            )
            loss_wss_mse_diag = masked_mse(pred_wss, target_wss, batch.surface_mask)
            supplementary_loss = wss_charbonnier_weight * loss_wss_charb
            charb_metrics = {
                "loss_wss_charb_unweighted": float(
                    loss_wss_charb.detach().cpu().item()
                ),
                "loss_wss_charb_weighted": float(
                    supplementary_loss.detach().cpu().item()
                ),
                "loss_wss_charb_target_mse": float(
                    loss_wss_mse_diag.detach().cpu().item()
                ),
            }
        else:
            supplementary_loss = None
            charb_metrics = {}
        # H19: vol_p Charbonnier behaviour.
        # - With IMTL-G: the Charb tensor is already task_losses[4], contributing
        #   alpha_vol_p * Charb_vol_p to the weighted sum the caller forms.
        #   Adding an extra additive term would double-count.
        # - Without IMTL-G: fall back to additive form so the mechanism fires.
        if loss_vol_p_charb is not None:
            if not use_imtl_g:
                add = vol_p_charbonnier_weight * loss_vol_p_charb
                if supplementary_loss is None:
                    supplementary_loss = add
                else:
                    supplementary_loss = supplementary_loss + add
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
        if loss is not None and supplementary_loss is not None:
            loss = loss + supplementary_loss
    metrics = {
        "base_mse_loss": float(base_mse_loss.detach().cpu().item()),
        "surface_loss": float(surface_loss.detach().cpu().item()),
        "volume_loss": float(volume_loss.detach().cpu().item()),
        "surface_loss_weighted": float(weighted_surface_loss.detach().cpu().item()),
        "volume_loss_weighted": float(weighted_volume_loss.detach().cpu().item()),
    }
    metrics.update(charb_metrics)
    metrics.update(vol_p_charb_metrics)
    return loss, metrics, task_losses, supplementary_loss


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

        imtl_g_n_tasks = 5  # cp, tau_x, tau_y, tau_z, vol_p
        imtl_g_task_names = ("cp", "tau_x", "tau_y", "tau_z", "vol_p")
        if config.use_imtl_g and state.is_main:
            print(
                "IMTL-G enabled: closed-form impartial gradient surgery "
                f"(Liu et al. 2021), n_tasks={imtl_g_n_tasks}, "
                f"eps={config.imtl_g_eps}, clamp_neg={config.imtl_g_clamp_negative}"
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
                loss, batch_loss_metrics, task_losses, supplementary_loss = train_loss(
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
                    use_imtl_g=config.use_imtl_g,
                    wss_charbonnier_weight=config.wss_charbonnier_weight,
                    wss_charbonnier_eps=config.wss_charbonnier_eps,
                    wss_charbonnier_axes=config.wss_charbonnier_axes,
                    vol_p_charbonnier_weight=config.vol_p_charbonnier_weight,
                    vol_p_charbonnier_eps=config.vol_p_charbonnier_eps,
                    tau_z_loss_weight=config.tau_z_loss_weight,
                )
                # === H136 IMTL-G: closed-form impartial gradient surgery ===
                # When use_imtl_g, train_loss returns loss=None with task_losses
                # tensor still attached to the autograd graph; compute alpha
                # from per-task gradients on a shared backbone parameter, then
                # form the weighted total loss for the backward pass.
                imtl_g_log: dict[str, float] = {}
                if config.use_imtl_g and task_losses is not None:
                    task_losses_finite = bool(
                        torch.isfinite(task_losses.detach()).all().item()
                    )
                    if task_losses_finite:
                        shared_param = base_model.norm.weight
                        per_task_grads: list[torch.Tensor] = []
                        for i in range(imtl_g_n_tasks):
                            g_i = torch.autograd.grad(
                                task_losses[i],
                                shared_param,
                                retain_graph=True,
                                allow_unused=True,
                            )[0]
                            if g_i is None:
                                per_task_grads.append(
                                    torch.zeros_like(shared_param).flatten().float()
                                )
                            else:
                                per_task_grads.append(g_i.detach().flatten().float())
                        alpha, imtl_diag = imtl_g_weights(
                            per_task_grads,
                            eps=config.imtl_g_eps,
                            clamp_negative=config.imtl_g_clamp_negative,
                        )
                        # DDP: average alpha across ranks for global consistency
                        if state.enabled:
                            torch.distributed.all_reduce(
                                alpha, op=torch.distributed.ReduceOp.SUM
                            )
                            alpha.div_(state.world_size)
                        weighted_task_loss = (alpha.detach() * task_losses).sum()
                        if supplementary_loss is not None:
                            loss = weighted_task_loss + supplementary_loss
                        else:
                            loss = weighted_task_loss
                        # Logging
                        alpha_cpu = alpha.detach().cpu()
                        tl_cpu = task_losses.detach().float().cpu()
                        for ti, name in enumerate(imtl_g_task_names):
                            imtl_g_log[f"imtl_g/w_{name}"] = float(
                                alpha_cpu[ti].item()
                            )
                            imtl_g_log[f"imtl_g/task_loss_{name}"] = float(
                                tl_cpu[ti].item()
                            )
                            imtl_g_log[f"imtl_g/grad_norm_{name}"] = float(
                                per_task_grads[ti].norm().item()
                            )
                        # Aggregates for advisor EP5 diagnostic (PR uses
                        # 4-task naming w_wss/w_vol_p/w_surf_p/w_abupt; we
                        # preserve H39's 5-task structure and derive these).
                        imtl_g_log["imtl_g/w_wss"] = float(
                            alpha_cpu[1:4].sum().item()
                        )
                        imtl_g_log["imtl_g/w_surf_p"] = float(
                            alpha_cpu[0].item()
                        )
                        imtl_g_log["imtl_g/w_vol_p"] = float(
                            alpha_cpu[4].item()
                        )
                        imtl_g_log["imtl_g/w_abupt"] = float(
                            alpha_cpu.mean().item()
                        )
                        imtl_g_log.update(imtl_diag)
                    else:
                        # Non-finite task losses: propagate to skip_step
                        # by setting loss to a non-finite scalar.
                        loss = task_losses.sum()
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
                    if imtl_g_log:
                        train_log.update(imtl_g_log)
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
