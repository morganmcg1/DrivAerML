# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H272: Hutchinson curvature-scaled per-parameter weight-noise TTA on H185 EP13.

Extends H252/H253 stacked TTA by replacing uniform sigma with per-parameter
sigma scaled by the diagonal Hessian estimate:

    sigma_i = sigma_base / sqrt(max(1 + beta * |diag_H_i|, 1.0))

The diagonal Hessian is estimated via the Hutchinson trace estimator using M
Rademacher random vectors and Hessian-vector products (HVP via double-backward).

Key constraint: DDP + create_graph=True are incompatible (PyTorch #63812).
Fix: run Hutchinson on unwrapped model.module on rank-0 only (bypasses allreduce
hooks), then broadcast diag_H tensors to all ranks.

Loop structure (same as H252 except perturb uses per-param sigma):
    [pre-eval] estimate diag_H on one val batch on rank-0, broadcast
    for split in (val, test):
        for mode in modes:
            for case in my_shard:
                for k in 0..K-1:            # weight perturbation
                    perturb: p_i <- clean_i + randn(seed_k) * sigma_i * |clean_i|
                    for R in resolutions:
                        for mirror in (False, True):
                            forward; un-mirror; accumulate per-point

Mirror convention unchanged from H148/H183/H209:
    surface_x [x, y, z, nx, ny, nz, area] -> negate y(1) and ny(4)
    volume_x  [x, y, z, sdf]              -> negate y(1)
    surface_y [cp, tau_x, tau_y, tau_z]   -> un-mirror tau_y(2)
    volume_y  [volume_pressure]            -> invariant

Modes:
    mirror_res_weight_noise_avg: K x R x 2 = full stacked TTA (H272 primary mode)
    weight_noise_only:           K passes at single resolution, no mirror (sanity)
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb

from data import SurfaceBatch, pad_collate
from data.loader import DrivAerMLCaseStore, DrivAerMLSurfaceDataset
from model import SurfaceTransolver
from trainer_runtime import (
    EvalAccumulator,
    TargetTransform,
    _accumulate_case_rel_l2,
    _masked_sse_count,
    autocast_context,
    cleanup_distributed,
    finalize_eval_accumulator,
    init_distributed,
    merge_eval_accumulators,
    primary_metric_log,
    print_metrics,
    unwrap_model,
)


@dataclass
class EvalConfig:
    """H272 eval config — extends H252 with Hutchinson curvature-scaled per-param sigma."""

    # Checkpoint: local path first (path exists), else W&B fetch via run_id+alias.
    checkpoint: str = "runs/h210/artifacts/h185/checkpoint.pt"
    run_id: str = "yw2a5dyl"
    checkpoint_alias: str = "epoch-13"
    cache_root: str = "outputs/h272_eval/_artifacts"

    # Multi-resolution TTA configuration
    resolutions: str = "49152,65536,81920"
    eval_modes: str = "mirror_res_weight_noise_avg"
    eval_surface_points: int = 65536

    # Weight-noise stacking knobs
    weight_noise_sigma: float = 5e-4
    weight_noise_passes: int = 5
    weight_noise_seed_base: int = 42

    # Hutchinson curvature estimation knobs
    hutchinson_m: int = 8          # number of Rademacher HVP samples
    hutchinson_beta: float = 1e4   # scaling factor: sigma_i = sigma_base / sqrt(1 + beta * |dH_i|)
    hutchinson_cache: str = "outputs/h272_diag_hessian.pt"  # cache diag_H for beta sweeps
    hutchinson_eval_surface_points: int = 1024  # small resolution for Hutchinson pre-pass (OOM-safe)

    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h272_eval"
    wandb_group: str = "h272-nezuko-hutchinson-curvature"
    wandb_name: str = ""
    wandb_project: str = "senpai-v1-drivaerml-ddp8"
    wandb_entity: str = "wandb-applied-ai-team"
    agent: str = "nezuko"

    batch_size: int = 2
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2

    # Model arch (must match H185 yw2a5dyl exactly)
    model_layers: int = 5
    model_hidden_dim: int = 512
    model_heads: int = 4
    model_mlp_ratio: int = 4
    model_slices: int = 128
    model_dropout: float = 0.0
    rff_num_features: int = 16
    rff_sigma: float = 1.0
    rff_init_sigmas: str = "0.25,0.5,1.0,2.0,4.0"
    pos_encoding_mode: str = "string_separable"
    use_qk_norm: bool = True
    use_surf_to_vol_xattn: bool = True
    drop_path_max: float = 0.1

    amp_mode: str = "bf16"
    debug: bool = False  # 2 val + 2 test cases


def parse_args(argv: Iterable[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="H272 Hutchinson curvature-scaled per-param weight-noise TTA")
    defaults = EvalConfig()
    for f in fields(EvalConfig):
        v = getattr(defaults, f.name)
        dashed = f"--{f.name.replace('_', '-')}"
        underscored = f"--{f.name}"
        cli_args = [dashed] if dashed == underscored else [dashed, underscored]
        if isinstance(v, bool):
            parser.add_argument(*cli_args, action="store_true", default=v, dest=f.name)
            neg_dashed = f"--no-{f.name.replace('_', '-')}"
            neg_underscored = f"--no_{f.name}"
            neg_args = [neg_dashed] if neg_dashed == neg_underscored else [neg_dashed, neg_underscored]
            parser.add_argument(*neg_args, action="store_false", dest=f.name)
        else:
            parser.add_argument(*cli_args, type=type(v), default=v, dest=f.name)
    ns = parser.parse_args(argv)
    cfg = EvalConfig(**{f.name: getattr(ns, f.name) for f in fields(EvalConfig)})
    return cfg


def parse_rff_init_sigmas(spec: str) -> list[float] | None:
    if not spec:
        return None
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_resolutions(spec: str) -> list[int]:
    return [int(p) for p in spec.split(",") if p.strip()]


def parse_modes(spec: str) -> list[str]:
    return [m.strip() for m in spec.split(",") if m.strip()]


def build_model(cfg: EvalConfig) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=cfg.model_layers,
        n_hidden=cfg.model_hidden_dim,
        dropout=cfg.model_dropout,
        n_head=cfg.model_heads,
        mlp_ratio=cfg.model_mlp_ratio,
        slice_num=cfg.model_slices,
        rff_num_features=cfg.rff_num_features,
        rff_sigma=cfg.rff_sigma,
        rff_init_sigmas=parse_rff_init_sigmas(cfg.rff_init_sigmas),
        pos_encoding_mode=cfg.pos_encoding_mode,
        use_qk_norm=cfg.use_qk_norm,
        use_surf_to_vol_xattn=cfg.use_surf_to_vol_xattn,
        drop_path_max=cfg.drop_path_max,
    )


def download_checkpoint(
    entity: str,
    project: str,
    run_id: str,
    alias: str,
    cache_root: Path,
) -> Path:
    cache_dir = cache_root / run_id / alias
    ckpt_path = cache_dir / "checkpoint.pt"
    if ckpt_path.exists():
        return ckpt_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    matched = None
    for art in run.logged_artifacts():
        if art.type != "model":
            continue
        if alias in art.aliases:
            matched = art
            break
    if matched is None:
        raise RuntimeError(f"No model artifact with alias '{alias}' for run {run_id}")
    matched.download(root=str(cache_dir))
    return ckpt_path


# --- mirror helpers ---


def mirror_inputs(batch: SurfaceBatch) -> SurfaceBatch:
    surface_x = batch.surface_x.clone()
    surface_x[..., 1].neg_()  # y
    surface_x[..., 4].neg_()  # normal_y
    volume_x = batch.volume_x.clone()
    volume_x[..., 1].neg_()  # y
    return SurfaceBatch(
        case_ids=batch.case_ids,
        surface_x=surface_x,
        surface_y=batch.surface_y,
        surface_mask=batch.surface_mask,
        volume_x=volume_x,
        volume_y=batch.volume_y,
        volume_mask=batch.volume_mask,
        metadata=batch.metadata,
    )


def unmirror_surface_real(pred: torch.Tensor) -> torch.Tensor:
    out = pred.clone()
    out[..., 2].neg_()  # tau_y un-mirror
    return out


def chunk_global_indices(view_index: int, view_count: int, n_full: int) -> torch.Tensor:
    if view_index >= view_count:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(view_index, n_full, view_count, dtype=torch.long)


# --- weight-noise helpers ---


@torch.no_grad()
def snapshot_clean_params(module: nn.Module) -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        if p.dtype.is_floating_point:
            snap[name] = p.detach().clone()
    return snap


@torch.no_grad()
def restore_clean_params(module: nn.Module, clean: dict[str, torch.Tensor]) -> None:
    for name, p in module.named_parameters():
        if name in clean:
            p.data.copy_(clean[name])


@torch.no_grad()
def perturb_relative_per_param_(
    module: nn.Module,
    clean: dict[str, torch.Tensor],
    sigma_per_param: dict[str, torch.Tensor],
    seed: int,
    device: torch.device,
) -> None:
    """p_i <- clean_i + randn(seed_i) * sigma_i * |clean_i|, identical across DDP ranks.

    sigma_per_param maps parameter name to a per-element sigma tensor (same shape
    as the parameter). This is the H272 replacement for the scalar sigma used in H252.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for name, p in module.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        clean_p = clean[name]
        noise = torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype)
        if name in sigma_per_param:
            sigma_i = sigma_per_param[name].to(device=device, dtype=p.dtype)
            noise.mul_(sigma_i).mul_(clean_p.abs())
        else:
            # Fallback: should not happen if sigma_per_param is built correctly.
            noise.mul_(0.0)
        p.data.copy_(clean_p).add_(noise)


# --- Hutchinson diagonal Hessian estimation ---


def _math_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Pure-tensor SDPA implementation that supports backward-of-backward.

    The fused/efficient/flash CUDA SDPA backends register custom backward ops
    whose *own backward* is not implemented, so they error on the HVP pass
    `autograd.grad(gz, params)` used here. The composable MATH backend is
    differentiable through but PyTorch's kernel selection (and especially
    nn.MultiheadAttention's fast path) sometimes bypasses ``sdpa_kernel``.
    Monkey-patching ``F.scaled_dot_product_attention`` with this math impl
    inside :func:`force_math_sdpa` is bulletproof.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    attn = torch.matmul(query, key.transpose(-2, -1)) * scale
    if is_causal:
        L, S = query.size(-2), key.size(-2)
        causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        attn = attn.masked_fill(~causal_mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return torch.matmul(attn, value)


@contextlib.contextmanager
def force_math_sdpa(model: nn.Module):
    """Force composable math SDPA inside the model for double-backward Hutchinson HVPs.

    Two simultaneous interventions are needed:
    1. Monkey-patch ``F.scaled_dot_product_attention`` to the math
       implementation above. This catches Transolver's slice-attention call.
    2. Flip every ``nn.MultiheadAttention`` to ``training=True`` so its
       forward routes through the slow Python path (which calls
       ``F.scaled_dot_product_attention`` we just patched) instead of the
       ``_native_multi_head_attention`` fast path that ignores
       ``sdpa_kernel`` and uses fused kernels with no double-backward.

    On exit, both ``F.scaled_dot_product_attention`` and the per-module
    training flags are restored.
    """
    orig_sdpa = F.scaled_dot_product_attention
    mha_modes: dict[int, bool] = {}
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            mha_modes[id(m)] = m.training
            m.train(True)
    F.scaled_dot_product_attention = _math_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig_sdpa
        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                if id(m) in mha_modes:
                    m.train(mha_modes[id(m)])


def compute_loss_for_hutchinson(
    model: nn.Module,
    batch: SurfaceBatch,
    transform: TargetTransform,
) -> torch.Tensor:
    """Forward pass for Hutchinson HVP — combined normalized MSE, no no_grad decorator.

    Returns a scalar loss suitable for double-backward. Uses normalized targets
    so surface and volume contribute comparably. AMP is NOT used here because
    create_graph=True is incompatible with bf16 autocast on many CUDA versions.
    """
    out = model(
        surface_x=batch.surface_x,
        surface_mask=batch.surface_mask,
        volume_x=batch.volume_x,
        volume_mask=batch.volume_mask,
    )
    surface_pred = out["surface_preds"].float()
    volume_pred = out["volume_preds"].float()

    surface_target_norm = transform.apply_surface(batch.surface_y.float())
    volume_target_norm = transform.apply_volume(batch.volume_y.float())

    # Masked MSE — only over valid (unpadded) points.
    surf_mask = batch.surface_mask.bool()
    vol_mask = batch.volume_mask.bool()

    surf_diff = (surface_pred - surface_target_norm)[surf_mask]
    vol_diff = (volume_pred - volume_target_norm)[vol_mask]

    surf_loss = surf_diff.pow(2).mean() if surf_diff.numel() > 0 else surface_pred.sum() * 0.0
    vol_loss = vol_diff.pow(2).mean() if vol_diff.numel() > 0 else volume_pred.sum() * 0.0

    return surf_loss + vol_loss


def hutchinson_diag_hessian(
    model: nn.Module,
    batch: SurfaceBatch,
    transform: TargetTransform,
    M: int = 8,
) -> list[torch.Tensor]:
    """Estimate diagonal Hessian via Hutchinson trace estimator.

    IMPORTANT: Must be called on the UNWRAPPED model (model.module or model._orig_mod)
    on rank-0 only. DDP allreduce hooks are incompatible with create_graph=True
    (PyTorch issue #63812).

    Args:
        model: Unwrapped (non-DDP) model, already in eval() mode, on rank-0 device.
        batch: A single SurfaceBatch already moved to rank-0 device.
        transform: TargetTransform for computing normalized loss.
        M: Number of Rademacher HVP samples (default 8).

    Returns:
        List of |diag_H| tensors, one per requires_grad parameter (same order as
        model.parameters() filtered to requires_grad=True).
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    diag_H = [torch.zeros_like(p, dtype=torch.float32) for p in params]

    with force_math_sdpa(model):
        for m_idx in range(M):
            # Rademacher probe vector z_i in {-1, +1}^d.
            # Use a fixed seed per sample for reproducibility.
            torch.manual_seed(m_idx * 7919 + 13)
            z = [
                (torch.randint(0, 2, p.shape, device=p.device, dtype=torch.float32) * 2.0 - 1.0)
                for p in params
            ]

            # First-order grad with create_graph=True so we can differentiate again.
            loss = compute_loss_for_hutchinson(model, batch, transform)
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

            # Replace None grads (unused params) with zero.
            grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)
            ]

            # g·z = scalar: sum of element-wise products across all param tensors.
            gz = sum((g * zv).sum() for g, zv in zip(grads, z))

            # HVP via second-order backward: Hv = d(g·z)/d(params).
            hvp = torch.autograd.grad(gz, params, retain_graph=False, allow_unused=True)

            for i, (h, zv) in enumerate(zip(hvp, z)):
                if h is not None:
                    # Hutchinson estimator: diag_H_i += z_i * (H*z)_i / M
                    diag_H[i] += (zv * h.detach()).float() / M

            del grads, gz, hvp, z, loss
            torch.cuda.empty_cache()

    # Return absolute values — we only care about curvature magnitude.
    return [dh.abs() for dh in diag_H]


def get_sigma_per_param(
    diag_H: list[torch.Tensor],
    param_names: list[str],
    sigma_base: float = 5e-4,
    beta: float = 1e4,
) -> dict[str, torch.Tensor]:
    """Compute per-parameter sigma tensors from diagonal Hessian estimate.

    Formula: sigma_i = sigma_base / sqrt(max(1 + beta * |diag_H_i|, 1.0))

    This ensures sigma_i <= sigma_base always (denominator >= 1.0 from clamp).
    High curvature params get lower sigma (more conservative perturbation).

    Args:
        diag_H: List of |diag_H| tensors (one per param, same order as param_names).
        param_names: List of parameter names aligned with diag_H.
        sigma_base: Base sigma (upper bound on all per-param sigmas).
        beta: Curvature scaling factor.

    Returns:
        Dict mapping parameter name -> per-element sigma tensor.
    """
    sigma_dict: dict[str, torch.Tensor] = {}
    for name, dh in zip(param_names, diag_H):
        # clamp(min=1.0) ensures denominator >= 1.0 -> sigma_i <= sigma_base
        denominator = (1.0 + beta * dh).clamp(min=1.0)
        sigma_i = sigma_base / denominator.sqrt()
        sigma_dict[name] = sigma_i
    return sigma_dict


def log_sigma_diagnostics(
    sigma_per_param: dict[str, torch.Tensor],
    sigma_base: float,
    diag_H_flat: torch.Tensor,
    run: wandb.sdk.wandb_run.Run | None,
) -> None:
    """Log per-param sigma ratio statistics and diag_H percentiles to W&B."""
    all_sigmas = torch.cat([s.flatten().float().cpu() for s in sigma_per_param.values()])
    ratios = all_sigmas / sigma_base  # should be in (0, 1]

    ratio_min = float(ratios.min().item())
    ratio_mean = float(ratios.mean().item())
    ratio_p1 = float(ratios.kthvalue(max(1, int(0.01 * ratios.numel()))).values.item())
    ratio_p10 = float(ratios.kthvalue(max(1, int(0.10 * ratios.numel()))).values.item())
    ratio_p50 = float(ratios.kthvalue(max(1, int(0.50 * ratios.numel()))).values.item())
    frac_below_half = float((ratios < 0.5).float().mean().item())

    dh_p1 = float(diag_H_flat.kthvalue(max(1, int(0.01 * diag_H_flat.numel()))).values.item())
    dh_p10 = float(diag_H_flat.kthvalue(max(1, int(0.10 * diag_H_flat.numel()))).values.item())
    dh_p50 = float(diag_H_flat.kthvalue(max(1, int(0.50 * diag_H_flat.numel()))).values.item())
    dh_p90 = float(diag_H_flat.kthvalue(max(1, int(0.90 * diag_H_flat.numel()))).values.item())
    dh_p99 = float(diag_H_flat.kthvalue(max(1, int(0.99 * diag_H_flat.numel()))).values.item())

    print(f"  sigma_ratio: min={ratio_min:.4f} mean={ratio_mean:.4f} p1={ratio_p1:.4f} "
          f"p10={ratio_p10:.4f} p50={ratio_p50:.4f} frac_below_half={frac_below_half:.4f}")
    print(f"  diag_H (abs): p1={dh_p1:.3e} p10={dh_p10:.3e} p50={dh_p50:.3e} "
          f"p90={dh_p90:.3e} p99={dh_p99:.3e}")

    if run is not None:
        wandb.log({
            "hutchinson/sigma_ratio_min": ratio_min,
            "hutchinson/sigma_ratio_mean": ratio_mean,
            "hutchinson/sigma_ratio_p1": ratio_p1,
            "hutchinson/sigma_ratio_p10": ratio_p10,
            "hutchinson/sigma_ratio_p50": ratio_p50,
            "hutchinson/sigma_ratio_frac_below_half": frac_below_half,
            "hutchinson/diag_H_p1": dh_p1,
            "hutchinson/diag_H_p10": dh_p10,
            "hutchinson/diag_H_p50": dh_p50,
            "hutchinson/diag_H_p90": dh_p90,
            "hutchinson/diag_H_p99": dh_p99,
        })


# --- per-case stacked TTA ---


@torch.no_grad()
def process_case_stacked(
    *,
    case_id: str,
    model: nn.Module,
    transform: TargetTransform,
    device: torch.device,
    store: DrivAerMLCaseStore,
    cfg: EvalConfig,
    resolutions: list[int],
    use_mirror: bool,
    clean: dict[str, torch.Tensor],
    K_passes: int,
    sigma_per_param: dict[str, torch.Tensor],
    seed_base: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, dict]:
    """Run K x R x M (mirror) TTA passes for one case and return per-point averaged
    predictions in real units.

    Unlike H252, each weight perturbation uses per-parameter sigma tensors derived
    from the Hutchinson diagonal Hessian estimate.
    """
    counts = store.case_point_counts(case_id)
    n_surf = counts["n_surface"]
    n_vol = counts["n_volume"]

    surf_sum = torch.zeros(n_surf, 4, dtype=torch.float32)
    surf_cnt = torch.zeros(n_surf, dtype=torch.int32)
    vol_sum = torch.zeros(n_vol, 1, dtype=torch.float32)
    vol_cnt = torch.zeros(n_vol, dtype=torch.int32)

    eval_module = unwrap_model(model)
    timing: dict[str, float] = {
        "forward_seconds": 0.0,
        "io_seconds": 0.0,
        "perturb_seconds": 0.0,
        "n_forwards": 0,
    }

    mirror_flags = (False, True) if use_mirror else (False,)

    for k in range(K_passes):
        # Perturb once per k; reuse across (res, mirror) inner passes.
        t_p = time.time()
        seed = (seed_base + k) * 100003
        perturb_relative_per_param_(
            eval_module,
            clean,
            sigma_per_param=sigma_per_param,
            seed=seed,
            device=device,
        )
        timing["perturb_seconds"] += time.time() - t_p

        for R in resolutions:
            dataset = DrivAerMLSurfaceDataset(
                case_ids=[case_id],
                store=store,
                max_surface_points=cfg.eval_surface_points,
                max_volume_points=R,
                sampling_mode="eval_chunk",
            )
            loader_iter = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                collate_fn=pad_collate,
                persistent_workers=False,
                prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            )
            for batch in loader_iter:
                batch = batch.to(device)
                metas_cached = list(batch.metadata)
                for mirror in mirror_flags:
                    t0 = time.time()
                    model_input = mirror_inputs(batch) if mirror else batch
                    with autocast_context(device, cfg.amp_mode):
                        out = eval_module(
                            surface_x=model_input.surface_x,
                            surface_mask=model_input.surface_mask,
                            volume_x=model_input.volume_x,
                            volume_mask=model_input.volume_mask,
                        )
                    surface_pred_real = transform.invert_surface(out["surface_preds"].float())
                    volume_pred_real = transform.invert_volume(out["volume_preds"].float())
                    if mirror:
                        surface_pred_real = unmirror_surface_real(surface_pred_real)
                    timing["forward_seconds"] += time.time() - t0
                    timing["n_forwards"] += int(batch.surface_x.shape[0])

                    t1 = time.time()
                    surface_pred_real_cpu = surface_pred_real.detach().cpu()
                    volume_pred_real_cpu = volume_pred_real.detach().cpu()
                    surface_mask_cpu = batch.surface_mask.detach().cpu()
                    volume_mask_cpu = batch.volume_mask.detach().cpu()
                    for i in range(len(batch.case_ids)):
                        meta = metas_cached[i]
                        s_view_idx = int(meta["surface_view_index"])
                        s_view_count = int(meta["surface_view_count"])
                        v_view_idx = int(meta["volume_view_index"])
                        v_view_count = int(meta["volume_view_count"])

                        if s_view_idx < s_view_count:
                            s_global = chunk_global_indices(s_view_idx, s_view_count, n_surf)
                            s_mask_i = surface_mask_cpu[i].bool()
                            s_chunk = surface_pred_real_cpu[i][s_mask_i]
                            if s_global.shape[0] != s_chunk.shape[0]:
                                raise RuntimeError(
                                    f"surface global-index mismatch case={case_id} R={R} "
                                    f"view_idx={s_view_idx}/{s_view_count}: "
                                    f"global={s_global.shape[0]} vs chunk={s_chunk.shape[0]}"
                                )
                            if s_global.numel() > 0:
                                surf_sum.index_add_(0, s_global, s_chunk)
                                surf_cnt.index_add_(
                                    0, s_global, torch.ones_like(s_global, dtype=torch.int32)
                                )

                        if v_view_idx < v_view_count:
                            v_global = chunk_global_indices(v_view_idx, v_view_count, n_vol)
                            v_mask_i = volume_mask_cpu[i].bool()
                            v_chunk = volume_pred_real_cpu[i][v_mask_i]
                            if v_global.shape[0] != v_chunk.shape[0]:
                                raise RuntimeError(
                                    f"volume global-index mismatch case={case_id} R={R} "
                                    f"view_idx={v_view_idx}/{v_view_count}: "
                                    f"global={v_global.shape[0]} vs chunk={v_chunk.shape[0]}"
                                )
                            if v_global.numel() > 0:
                                vol_sum.index_add_(0, v_global, v_chunk)
                                vol_cnt.index_add_(
                                    0, v_global, torch.ones_like(v_global, dtype=torch.int32)
                                )
                    timing["io_seconds"] += time.time() - t1

    if int(surf_cnt.min()) == 0:
        raise RuntimeError(
            f"surface coverage incomplete for case {case_id}: "
            f"{int((surf_cnt == 0).sum())} points have zero contributing chunks"
        )
    if int(vol_cnt.min()) == 0:
        raise RuntimeError(
            f"volume coverage incomplete for case {case_id}: "
            f"{int((vol_cnt == 0).sum())} points have zero contributing chunks"
        )

    surf_avg = surf_sum / surf_cnt.unsqueeze(-1).to(torch.float32)
    vol_avg = vol_sum / vol_cnt.unsqueeze(-1).to(torch.float32)

    case = store.load_case(case_id)
    surf_y = case.surface_y.float()
    vol_y = case.volume_y.float()

    n_passes = K_passes * len(resolutions) * (2 if use_mirror else 1)
    return surf_avg, vol_avg, surf_y, vol_y, n_passes, timing


def add_case_to_accumulator(
    acc: EvalAccumulator,
    *,
    case_id: str,
    surface_pred_real: torch.Tensor,
    volume_pred_real: torch.Tensor,
    surface_y: torch.Tensor,
    volume_y: torch.Tensor,
    transform: TargetTransform,
) -> None:
    surface_pred_norm = transform.apply_surface(surface_pred_real)
    volume_pred_norm = transform.apply_volume(volume_pred_real)
    surface_target_norm = transform.apply_surface(surface_y)
    volume_target_norm = transform.apply_volume(volume_y)

    mask_surface = torch.ones(surface_pred_norm.shape[0], dtype=torch.bool)
    mask_volume = torch.ones(volume_pred_norm.shape[0], dtype=torch.bool)
    surface_sse, surface_count = _masked_sse_count(
        surface_pred_norm, surface_target_norm, mask_surface
    )
    volume_sse, volume_count = _masked_sse_count(
        volume_pred_norm, volume_target_norm, mask_volume
    )
    acc.surface_loss_sse += surface_sse
    acc.surface_loss_count += surface_count
    acc.volume_loss_sse += volume_sse
    acc.volume_loss_count += volume_count

    surface_abs = (surface_pred_real - surface_y).abs()
    acc.abs_sums["surface_pressure"] += float(surface_abs[:, 0].sum().item())
    acc.abs_counts["surface_pressure"] += int(surface_abs[:, 0].numel())
    wall_abs = surface_abs[:, 1:4]
    acc.abs_sums["wall_shear"] += float(wall_abs.sum().item())
    acc.abs_counts["wall_shear"] += int(wall_abs.numel())
    for offset, axis in enumerate(("x", "y", "z")):
        channel = wall_abs[:, offset]
        acc.abs_sums[f"wall_shear_{axis}"] += float(channel.sum().item())
        acc.abs_counts[f"wall_shear_{axis}"] += int(channel.numel())
    wall_vector_error = torch.linalg.vector_norm(
        surface_pred_real[:, 1:4] - surface_y[:, 1:4],
        dim=-1,
    )
    acc.wall_shear_vector_abs_sum += float(wall_vector_error.sum().item())
    acc.wall_shear_vector_count += int(wall_vector_error.numel())

    volume_abs = (volume_pred_real - volume_y).abs()
    acc.abs_sums["volume_pressure"] += float(volume_abs[:, 0].sum().item())
    acc.abs_counts["volume_pressure"] += int(volume_abs[:, 0].numel())

    _accumulate_case_rel_l2(
        acc.case_sums["surface_pressure"],
        case_id=case_id,
        pred=surface_pred_real[:, 0:1],
        target=surface_y[:, 0:1],
    )
    _accumulate_case_rel_l2(
        acc.case_sums["wall_shear"],
        case_id=case_id,
        pred=surface_pred_real[:, 1:4],
        target=surface_y[:, 1:4],
    )
    for channel, axis in enumerate(("x", "y", "z"), start=1):
        _accumulate_case_rel_l2(
            acc.case_sums[f"wall_shear_{axis}"],
            case_id=case_id,
            pred=surface_pred_real[:, channel : channel + 1],
            target=surface_y[:, channel : channel + 1],
        )
    _accumulate_case_rel_l2(
        acc.case_sums["volume_pressure"],
        case_id=case_id,
        pred=volume_pred_real,
        target=volume_y,
    )


def evaluate_split_stacked(
    *,
    split_name: str,
    case_ids: list[str],
    model: nn.Module,
    transform: TargetTransform,
    device: torch.device,
    store: DrivAerMLCaseStore,
    cfg: EvalConfig,
    resolutions: list[int],
    use_mirror: bool,
    clean: dict[str, torch.Tensor],
    K_passes: int,
    sigma_per_param: dict[str, torch.Tensor],
    seed_base: int,
    distributed_state,
) -> dict[str, float]:
    rank = distributed_state.rank if distributed_state and distributed_state.enabled else 0
    world_size = (
        distributed_state.world_size
        if distributed_state and distributed_state.enabled
        else 1
    )
    my_cases = case_ids[rank::world_size]

    acc = EvalAccumulator()
    total_t0 = time.time()
    total_forward = 0.0
    total_io = 0.0
    total_perturb = 0.0
    total_forwards = 0
    for case_idx, case_id in enumerate(my_cases):
        t0 = time.time()
        surf_avg, vol_avg, surf_y, vol_y, n_passes, timing = process_case_stacked(
            case_id=case_id,
            model=model,
            transform=transform,
            device=device,
            store=store,
            cfg=cfg,
            resolutions=resolutions,
            use_mirror=use_mirror,
            clean=clean,
            K_passes=K_passes,
            sigma_per_param=sigma_per_param,
            seed_base=seed_base,
        )
        add_case_to_accumulator(
            acc,
            case_id=case_id,
            surface_pred_real=surf_avg,
            volume_pred_real=vol_avg,
            surface_y=surf_y,
            volume_y=vol_y,
            transform=transform,
        )
        total_forward += timing["forward_seconds"]
        total_io += timing["io_seconds"]
        total_perturb += timing["perturb_seconds"]
        total_forwards += timing["n_forwards"]
        dt = time.time() - t0
        print(
            f"  [rank {rank}] {split_name} {case_id} ({case_idx + 1}/{len(my_cases)}): "
            f"n_passes={n_passes} forwards={timing['n_forwards']} "
            f"forward={timing['forward_seconds']:.1f}s "
            f"perturb={timing['perturb_seconds']:.1f}s "
            f"io={timing['io_seconds']:.1f}s total={dt:.1f}s",
            flush=True,
        )

    print(
        f"[rank {rank}] {split_name} complete: {len(my_cases)} cases in "
        f"{time.time() - total_t0:.1f}s "
        f"(forward={total_forward:.1f}s perturb={total_perturb:.1f}s "
        f"io={total_io:.1f}s n_forwards={total_forwards})",
        flush=True,
    )

    if distributed_state is not None and distributed_state.enabled:
        gathered: list[EvalAccumulator | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, acc)
        if not distributed_state.is_main:
            return {}
        merged = merge_eval_accumulators(g for g in gathered if g is not None)
    else:
        merged = acc

    return finalize_eval_accumulator(merged)


def resolve_checkpoint_path(cfg: EvalConfig, state) -> tuple[Path, str]:
    """Use local checkpoint path if it exists; else download from W&B."""
    local = Path(cfg.checkpoint)
    if local.exists():
        return local, "local"

    cache_root = Path(cfg.cache_root)
    if state.is_main:
        ckpt_path = download_checkpoint(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            run_id=cfg.run_id,
            alias=cfg.checkpoint_alias,
            cache_root=cache_root,
        )
        ckpt_path_str = str(ckpt_path)
    else:
        ckpt_path_str = ""
    if state.enabled:
        obj = [ckpt_path_str]
        dist.broadcast_object_list(obj, src=0)
        ckpt_path_str = obj[0]
    return Path(ckpt_path_str), "wandb"


def get_one_val_batch(
    store: DrivAerMLCaseStore,
    cfg: EvalConfig,
    device: torch.device,
    val_case_ids: list[str],
    n_surface_points: int | None = None,
    batch_size: int | None = None,
) -> SurfaceBatch:
    """Load one batch from the first val case.

    When called for the Hutchinson pre-pass, pass n_surface_points and batch_size
    as small values (e.g. 1024, 1) to avoid OOM from create_graph=True HVPs.
    Both surface and volume are set to surf_pts — both scale create_graph memory.
    The curvature structure is resolution-independent so a small batch is valid.
    """
    # Use the first val case only; we need a single batch with both surface and volume data.
    first_case = val_case_ids[0]
    surf_pts = n_surface_points if n_surface_points is not None else cfg.eval_surface_points
    bs = batch_size if batch_size is not None else cfg.batch_size
    dataset = DrivAerMLSurfaceDataset(
        case_ids=[first_case],
        store=store,
        max_surface_points=surf_pts,
        max_volume_points=surf_pts,  # keep volume small too — both scale create_graph memory
        sampling_mode="eval_chunk",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=0,  # no multiprocessing for single batch
        pin_memory=False,
        collate_fn=pad_collate,
        persistent_workers=False,
    )
    batch = next(iter(loader))
    return batch.to(device)


def compute_or_load_diag_hessian(
    model: nn.Module,
    transform: TargetTransform,
    val_case_ids: list[str],
    store: DrivAerMLCaseStore,
    cfg: EvalConfig,
    device: torch.device,
    state,
    run,
) -> list[torch.Tensor]:
    """Run Hutchinson on rank-0 (or load cached result) and broadcast to all ranks.

    Returns list of |diag_H| tensors (one per requires_grad param, float32, on `device`).
    """
    eval_module = unwrap_model(model)
    param_names = [name for name, p in eval_module.named_parameters() if p.requires_grad]
    cache_path = Path(cfg.hutchinson_cache)

    if state.is_main:
        if cache_path.exists():
            print(f"Loading cached diag_H from {cache_path}", flush=True)
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            diag_H_cpu = cached["diag_H"]
            # Move to device
            diag_H = [dh.to(device) for dh in diag_H_cpu]
            print(f"  Loaded {len(diag_H)} diag_H tensors from cache", flush=True)
        else:
            print(
                f"Running Hutchinson diagonal Hessian estimation: M={cfg.hutchinson_m}",
                flush=True,
            )
            t_hutch = time.time()
            # Use small resolution (hutchinson_eval_surface_points) and batch_size=1
            # to avoid OOM from create_graph=True keeping the full attention graph.
            # Curvature structure is resolution-independent so this is valid.
            val_batch = get_one_val_batch(
                store, cfg, device, val_case_ids,
                n_surface_points=cfg.hutchinson_eval_surface_points,
                batch_size=1,
            )
            diag_H = hutchinson_diag_hessian(
                model=eval_module,
                batch=val_batch,
                transform=transform,
                M=cfg.hutchinson_m,
            )
            dt_hutch = time.time() - t_hutch
            print(f"  Hutchinson done in {dt_hutch:.1f}s", flush=True)

            # Cache to disk so beta sweeps can reuse without re-running M HVPs.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"diag_H": [dh.cpu() for dh in diag_H]}, cache_path)
            print(f"  Cached diag_H to {cache_path}", flush=True)

            if run is not None:
                wandb.log({"hutchinson/estimation_seconds": dt_hutch})
    else:
        # Other ranks allocate empty tensors to receive broadcast.
        diag_H = [
            torch.empty_like(p, dtype=torch.float32)
            for p in eval_module.parameters()
            if p.requires_grad
        ]

    # Broadcast diag_H from rank-0 to all ranks.
    if state.enabled:
        for tensor in diag_H:
            dist.broadcast(tensor, src=0)
        # Barrier to ensure all ranks have diag_H before proceeding.
        dist.barrier()

    return diag_H


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device

    resolutions = parse_resolutions(cfg.resolutions)
    modes = parse_modes(cfg.eval_modes)
    valid_modes = ("weight_noise_only", "mirror_res_weight_noise_avg")
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Unknown eval mode: {mode!r} (valid: {valid_modes})")

    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        print(f"Resolutions: {resolutions}")
        print(f"Modes: {modes}")
        print(
            f"Weight-noise TTA: sigma_base={cfg.weight_noise_sigma} "
            f"K_passes={cfg.weight_noise_passes} seed_base={cfg.weight_noise_seed_base}"
        )
        print(
            f"Hutchinson: M={cfg.hutchinson_m} beta={cfg.hutchinson_beta} "
            f"cache={cfg.hutchinson_cache}"
        )

    ckpt_path, ckpt_source = resolve_checkpoint_path(cfg, state)
    if state.is_main:
        print(f"Checkpoint: {ckpt_path} (source={ckpt_source})")

    store = DrivAerMLCaseStore(
        manifest_path=cfg.manifest, root=cfg.data_root or None
    )
    from data.loader import target_stats_from_normalizers  # noqa: PLC0415 — late import

    stats = target_stats_from_normalizers(store)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    model = build_model(cfg).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ck["model"]
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if state.is_main:
        print(
            f"Loaded checkpoint epoch={ck.get('epoch')} source={ck.get('checkpoint_source')}"
        )
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  missing[:5]={missing[:5]}")
        if unexpected:
            print(f"  unexpected[:5]={unexpected[:5]}")
    model.eval()

    eval_module = unwrap_model(model)
    clean = snapshot_clean_params(eval_module)
    if state.is_main:
        n_perturbed = sum(t.numel() for t in clean.values())
        print(f"Snapshotted {len(clean)} fp tensors ({n_perturbed:,} elements)")

    val_case_ids = store.case_ids("val")
    test_case_ids = store.case_ids("test")
    if cfg.debug:
        val_case_ids = val_case_ids[:2]
        test_case_ids = test_case_ids[:2]
        if state.is_main:
            print(f"DEBUG: {len(val_case_ids)} val + {len(test_case_ids)} test cases only")

    run = None
    if state.is_main:
        run_name = (
            cfg.wandb_name
            or f"{cfg.agent}/h272-hutchinson-sigma-{cfg.weight_noise_sigma}-beta-{cfg.hutchinson_beta:.0e}-K{cfg.weight_noise_passes}"
        )
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", cfg.wandb_project),
            entity=os.environ.get("WANDB_ENTITY", cfg.wandb_entity),
            group=cfg.wandb_group,
            name=run_name,
            config={
                **asdict(cfg),
                "checkpoint_epoch": ck.get("epoch"),
                "checkpoint_source": ck.get("checkpoint_source"),
                "resolutions_list": resolutions,
                "modes_list": modes,
                "n_val_cases": len(val_case_ids),
                "n_test_cases": len(test_case_ids),
                "ckpt_source": ckpt_source,
            },
            tags=[
                "h272",
                "hutchinson",
                "curvature-noise",
                "tta",
                "stacked",
                "weight-noise",
                "multi-res",
                "mirror",
                "eval-only",
                cfg.agent,
            ],
            reinit="finish_previous",
        )

    # --- Hutchinson pre-pass: estimate diag_H, compute per-param sigma ---
    # Run on unwrapped model on rank-0 only (DDP+create_graph incompatibility),
    # then broadcast to all ranks. This is a one-time cost before the eval loop.
    if state.is_main:
        print("\n=== Hutchinson diagonal Hessian pre-pass ===", flush=True)

    diag_H = compute_or_load_diag_hessian(
        model=model,
        transform=transform,
        val_case_ids=val_case_ids,
        store=store,
        cfg=cfg,
        device=device,
        state=state,
        run=run,
    )

    # Restore clean weights after Hutchinson pass (in-place backward modifies nothing,
    # but be defensive in case any compute_loss_for_hutchinson side-effects occurred).
    restore_clean_params(eval_module, clean)

    # Build per-param sigma dict (aligned to named_parameters order, requires_grad only).
    param_names = [name for name, p in eval_module.named_parameters() if p.requires_grad]
    sigma_per_param = get_sigma_per_param(
        diag_H=diag_H,
        param_names=param_names,
        sigma_base=cfg.weight_noise_sigma,
        beta=cfg.hutchinson_beta,
    )

    if state.is_main:
        diag_H_flat = torch.cat([dh.flatten().float().cpu() for dh in diag_H])
        log_sigma_diagnostics(sigma_per_param, cfg.weight_noise_sigma, diag_H_flat, run)

    # --- Main evaluation loop ---
    splits = [
        ("val_surface", val_case_ids, "full_val"),
        ("test_surface", test_case_ids, "test"),
    ]

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, case_ids, log_prefix in splits:
        summary[split_name] = {}
        for mode in modes:
            if mode == "weight_noise_only":
                # K-pass weight-noise average at a single resolution, no mirror.
                mode_resolutions = [cfg.eval_surface_points]
                use_mirror = False
            elif mode == "mirror_res_weight_noise_avg":
                mode_resolutions = list(resolutions)
                use_mirror = True
            else:
                raise ValueError(f"Unknown mode {mode!r}")

            if state.is_main:
                print(
                    f"\n=== {split_name} mode={mode} resolutions={mode_resolutions} "
                    f"mirror={use_mirror} K={cfg.weight_noise_passes} "
                    f"sigma_base={cfg.weight_noise_sigma} beta={cfg.hutchinson_beta} ===",
                    flush=True,
                )
            t0 = time.time()
            metrics = evaluate_split_stacked(
                split_name=f"{split_name}/{mode}",
                case_ids=case_ids,
                model=model,
                transform=transform,
                device=device,
                store=store,
                cfg=cfg,
                resolutions=mode_resolutions,
                use_mirror=use_mirror,
                clean=clean,
                K_passes=cfg.weight_noise_passes,
                sigma_per_param=sigma_per_param,
                seed_base=cfg.weight_noise_seed_base,
                distributed_state=state,
            )
            dt = time.time() - t0
            if state.is_main:
                print(f"  total {split_name}/{mode}: {dt:.1f}s")
                print_metrics(f"{split_name}/{mode}", metrics)
                summary[split_name][mode] = metrics

                log_obj: dict[str, float] = {}
                log_obj.update(primary_metric_log(f"{log_prefix}_primary/{mode}", metrics))
                log_obj.update({f"{log_prefix}_extra/{mode}/loss": metrics["loss"]})
                log_obj[f"{log_prefix}_extra/{mode}/seconds"] = dt
                if run is not None:
                    wandb.log(log_obj)

    # Always restore clean weights at the end.
    restore_clean_params(eval_module, clean)

    if state.is_main and summary:
        print("\n=== Summary (rel_l2_pct lower-is-better) ===")
        for split, modes_dict in summary.items():
            print(f"\n[{split}]")
            keys_to_show = (
                "abupt_axis_mean_rel_l2_pct",
                "surface_pressure_rel_l2_pct",
                "wall_shear_rel_l2_pct",
                "wall_shear_x_rel_l2_pct",
                "wall_shear_y_rel_l2_pct",
                "wall_shear_z_rel_l2_pct",
                "volume_pressure_rel_l2_pct",
            )
            mode_keys = list(modes_dict.keys())
            header = f"  {'metric':<36s} " + " ".join(f"{m:>36s}" for m in mode_keys)
            print(header)
            for k in keys_to_show:
                row = f"  {k:<36s} " + " ".join(
                    f"{modes_dict[m][k]:>36.4f}" for m in mode_keys
                )
                print(row)

        if run is not None:
            for split, modes_dict in summary.items():
                for mode, metrics in modes_dict.items():
                    for k, v in metrics.items():
                        try:
                            run.summary[f"{split}/{mode}/{k}"] = float(v)
                        except Exception:
                            pass
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
