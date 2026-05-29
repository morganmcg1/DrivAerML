# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H273: Taylor 2nd-order correction — subtract quadratic bias from anti-thetic mean.

Builds on H268 (anti-thetic standalone) and H253 (multi-res + weight-noise stack).
Anti-thetic pair averaging cancels the linear-gradient Taylor term ∇f(w)·ε; this
script uses the same paired samples to estimate the **quadratic Taylor residual**
and subtract a λ-weighted version of it from the anti-thetic mean.

For K anti-thetic pairs (w±ε_k), per (resolution, batch, mirror):
    f_base                    = forward(clean weights)
    f_anti_k = (f_+_k + f_-_k) / 2    ≈ f_base + ½ε_kᵀHε_k + O(ε⁴)
    f_anti   = mean_k(f_anti_k)       ≈ f_base + ½σ²·tr(H) + O(ε⁴)
    r_k      = f_anti_k - f_base      ≈ ½ε_kᵀHε_k
    r_mean   = mean_k(r_k)            = f_anti - f_base
    f_out(λ) = f_anti - λ · r_mean    = (1-λ)·f_anti + λ·f_base

λ=0 recovers the plain anti-thetic mean (H268 / H253 baseline at K pairs).
λ=1 is the clean prediction (no anti-thetic TTA).
0<λ<1 interpolates: a non-zero best λ implies the curvature residual carries
useful signal (the convex bowl pulls the anti-thetic mean upward; subtracting a
fraction of the residual moves predictions back toward the unbiased target).

Multi-resolution / mirror TTA is layered on top: f_base and f_anti are each
averaged across {resolutions} × {orig, mirror?} per-point in real units before
the λ sweep, so the sweep is **completely free post-hoc** — only the per-case
f_anti and f_base buffers are computed during inference.

Modes:
    weight_noise_res_avg          : 6-res, no mirror, K anti-thetic pairs (Arm A)
    weight_noise_mirror_res_avg   : 6-res + mirror, K pairs (Arm B)

Pair restoration uses an explicit "set from clean snapshot ± δ" pattern instead of
in-place add/subtract — model parameters are stored in FP32 here (not BF16), so
the +δ / -2δ / +δ in-place pattern accumulates ULP-scale drift (≈1e-7 per pair).
Restoring from the clean snapshot before each forward keeps every (clean ± δ_k)
evaluation exact, with no drift across pairs or resolutions.

Relative noise (H242/H253/H268 convention): δ = randn(seed) * sigma * |clean_p|.
Deltas are pre-materialised once per case (K tensors on GPU), so the same noise
patterns are reused for every (resolution, batch, mirror) tuple within a case.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
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
    """H273 Taylor 2nd-order correction eval config.

    Defaults target the H185 EP13 EMA checkpoint and the H253 6-resolution stack.
    """

    checkpoint: str = "outputs/h236_eval/_artifacts/yw2a5dyl/epoch-13/checkpoint.pt"
    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h273_eval"
    wandb_group: str = "h273-askeladd-taylor2"
    wandb_name: str = ""
    wandb_project: str = "senpai-v1-drivaerml-ddp8"
    wandb_entity: str = "wandb-applied-ai-team"
    agent: str = "askeladd"

    # Multi-resolution TTA configuration
    resolutions: str = "32768,49152,65536,81920,98304,131072"
    eval_modes: str = "weight_noise_res_avg"
    eval_surface_points: int = 65536

    # H273 anti-thetic + Taylor 2 knobs
    weight_noise_sigma: float = 5e-4
    n_antithetic_pairs: int = 3
    weight_noise_seed_base: int = 42
    taylor_lambda: float = 0.1
    taylor_lambda_sweep: str = "0.0,0.05,0.1,0.15,0.2,0.3"
    verify_restore: bool = True  # assert in-place restore once per case for the first case

    # Eval-only loader params
    batch_size: int = 2
    num_workers: int = 4
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
    debug: bool = False  # debug: 2 val + 2 test cases only


def parse_args(argv: Iterable[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="H273 Taylor 2nd-order anti-thetic TTA eval")
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
    return [int(p.strip()) for p in spec.split(",") if p.strip()]


def parse_modes(spec: str) -> list[str]:
    return [m.strip() for m in spec.split(",") if m.strip()]


def parse_lambdas(spec: str, fallback: float) -> list[float]:
    if not spec or not spec.strip():
        return [float(fallback)]
    out: list[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        out = [float(fallback)]
    return out


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


# --- mirror helpers (H148/H183/H209 convention) ---


def mirror_inputs(batch: SurfaceBatch) -> SurfaceBatch:
    """Negate y/normal_y in surface_x, y in volume_x. Keep masks and targets unchanged."""
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
    """surface real-units [cp, tau_x, tau_y, tau_z]: negate tau_y to un-mirror."""
    out = pred.clone()
    out[..., 2].neg_()
    return out


def chunk_global_indices(view_index: int, view_count: int, n_full: int) -> torch.Tensor:
    """Reconstruct the global stride indices for a chunk emitted by ``eval_chunk`` sampling."""
    if view_index >= view_count:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(view_index, n_full, view_count, dtype=torch.long)


# --- Weight-noise helpers ---


@torch.no_grad()
def snapshot_clean_params(module: nn.Module) -> dict[str, torch.Tensor]:
    """Pre-copy floating-point parameter tensors for fast restore/verification."""
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
def assert_clean(module: nn.Module, clean: dict[str, torch.Tensor], where: str) -> None:
    """Verify ``copy_(clean) -> add(delta) -> copy_(clean)`` ends exactly at the snapshot.

    Restoration is performed by overwriting from the snapshot, so equality must
    be bit-exact (no FP32 rounding involved beyond the copy_ itself).
    """
    for name, p in module.named_parameters():
        if name not in clean:
            continue
        if not torch.equal(p.data, clean[name]):
            diff = (p.data.float() - clean[name].float()).abs()
            max_diff = float(diff.max().item())
            raise RuntimeError(
                f"Parameter restoration check failed at {where}: param={name} "
                f"max|p - clean|={max_diff:.3e} (expected exact equality after copy_)"
            )


@torch.no_grad()
def materialize_delta(
    module: nn.Module,
    clean: dict[str, torch.Tensor],
    sigma: float,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build per-parameter relative-noise delta tensors deterministically.

    delta_p = randn(seed) * sigma * |clean_p|
    Uses a device-local Generator so the draw is identical across DDP ranks
    (same param order and shapes => same delta on every rank).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    delta: dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        clean_p = clean[name]
        noise = torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype)
        noise.mul_(sigma).mul_(clean_p.abs())
        delta[name] = noise
    return delta


@torch.no_grad()
def set_perturbed_(
    module: nn.Module,
    clean: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    scale: float,
) -> None:
    """p <- clean[name] + scale * delta[name] exactly (overwrite from snapshot).

    Exact in FP32: copy_ writes the snapshot bit-for-bit; the subsequent add_
    is the only rounding step. This eliminates the cross-pair drift the
    naïve +1/-2/+1 in-place pattern would accumulate.
    """
    for name, p in module.named_parameters():
        if name not in delta:
            continue
        p.data.copy_(clean[name]).add_(delta[name], alpha=scale)


# --- Per-case multi-resolution Taylor-2 anti-thetic accumulation ---


@torch.no_grad()
def _forward_real(
    model: nn.Module,
    batch_input: SurfaceBatch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    mirror: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_context(device, amp_mode):
        out = model(
            surface_x=batch_input.surface_x,
            surface_mask=batch_input.surface_mask,
            volume_x=batch_input.volume_x,
            volume_mask=batch_input.volume_mask,
        )
    surface_pred_real = transform.invert_surface(out["surface_preds"].float())
    volume_pred_real = transform.invert_volume(out["volume_preds"].float())
    if mirror:
        surface_pred_real = unmirror_surface_real(surface_pred_real)
        # volume_pressure is scalar and invariant under y-mirror.
    return surface_pred_real, volume_pred_real


@torch.no_grad()
def _scatter_into_buffers(
    *,
    case_id: str,
    K_res: int,
    batch: SurfaceBatch,
    surface_pred_real: torch.Tensor,
    volume_pred_real: torch.Tensor,
    n_surf: int,
    n_vol: int,
    surf_sum: torch.Tensor,
    surf_cnt: torch.Tensor,
    vol_sum: torch.Tensor,
    vol_cnt: torch.Tensor,
    metas_cached: list,
    timing: dict,
) -> None:
    """Scatter per-point predictions into CPU buffers using stride metadata."""
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
                    f"surface global-index mismatch case={case_id} K={K_res} "
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
                    f"volume global-index mismatch case={case_id} K={K_res} "
                    f"view_idx={v_view_idx}/{v_view_count}: "
                    f"global={v_global.shape[0]} vs chunk={v_chunk.shape[0]}"
                )
            if v_global.numel() > 0:
                vol_sum.index_add_(0, v_global, v_chunk)
                vol_cnt.index_add_(
                    0, v_global, torch.ones_like(v_global, dtype=torch.int32)
                )
    timing["io_seconds"] += time.time() - t1


@torch.no_grad()
def process_case_taylor2(
    *,
    case_id: str,
    case_index: int,
    model: nn.Module,
    transform: TargetTransform,
    device: torch.device,
    store: DrivAerMLCaseStore,
    cfg: EvalConfig,
    resolutions: list[int],
    use_mirror: bool,
    sigma: float,
    K_pairs: int,
    seed_base: int,
    clean: dict[str, torch.Tensor],
    verify_restore_this_case: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, dict,
]:
    """Run multi-res anti-thetic + base passes for one case.

    Returns (surf_avg_base, surf_avg_anti, vol_avg_base, vol_avg_anti,
             surf_y, vol_y, n_passes, timing). All averaged predictions are
    in **real units** (denormalized) and per-point.
    """
    counts = store.case_point_counts(case_id)
    n_surf = counts["n_surface"]
    n_vol = counts["n_volume"]

    # Base (clean weights) accumulators: 1 contribution per (res, mirror) per point.
    surf_base_sum = torch.zeros(n_surf, 4, dtype=torch.float32)
    surf_base_cnt = torch.zeros(n_surf, dtype=torch.int32)
    vol_base_sum = torch.zeros(n_vol, 1, dtype=torch.float32)
    vol_base_cnt = torch.zeros(n_vol, dtype=torch.int32)
    # Anti-thetic mid-point accumulators: K contributions per (res, mirror) per point.
    surf_anti_sum = torch.zeros(n_surf, 4, dtype=torch.float32)
    surf_anti_cnt = torch.zeros(n_surf, dtype=torch.int32)
    vol_anti_sum = torch.zeros(n_vol, 1, dtype=torch.float32)
    vol_anti_cnt = torch.zeros(n_vol, dtype=torch.int32)

    timing: dict[str, float] = {
        "forward_seconds": 0.0,
        "io_seconds": 0.0,
        "perturb_seconds": 0.0,
        "n_forwards": 0,
    }

    eval_module = unwrap_model(model)

    # Pre-materialize K deltas once per case (same noise reused across res/mirror).
    t_perturb = time.time()
    deltas: list[dict[str, torch.Tensor]] = []
    for k in range(K_pairs):
        seed = (seed_base + k) * 100003
        deltas.append(materialize_delta(eval_module, clean, sigma, seed, device))
    timing["perturb_seconds"] += time.time() - t_perturb

    mirror_flags = (False, True) if use_mirror else (False,)
    n_passes_per_batch = (1 + 2 * K_pairs) * len(mirror_flags)

    try:
        for K in resolutions:
            dataset = DrivAerMLSurfaceDataset(
                case_ids=[case_id],
                store=store,
                max_surface_points=cfg.eval_surface_points,
                max_volume_points=K,
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
                    model_input = mirror_inputs(batch) if mirror else batch

                    # --- Base prediction (clean weights) ---
                    t0 = time.time()
                    surf_base, vol_base = _forward_real(
                        eval_module, model_input, transform, device, cfg.amp_mode, mirror
                    )
                    timing["forward_seconds"] += time.time() - t0
                    timing["n_forwards"] += int(batch.surface_x.shape[0])

                    _scatter_into_buffers(
                        case_id=case_id, K_res=K, batch=batch,
                        surface_pred_real=surf_base, volume_pred_real=vol_base,
                        n_surf=n_surf, n_vol=n_vol,
                        surf_sum=surf_base_sum, surf_cnt=surf_base_cnt,
                        vol_sum=vol_base_sum, vol_cnt=vol_base_cnt,
                        metas_cached=metas_cached, timing=timing,
                    )

                    # --- K anti-thetic pairs (set from clean ± δ explicitly, exact) ---
                    for k in range(K_pairs):
                        delta = deltas[k]

                        # p = clean + δ_k
                        set_perturbed_(eval_module, clean, delta, +1.0)
                        t0 = time.time()
                        surf_plus, vol_plus = _forward_real(
                            eval_module, model_input, transform, device, cfg.amp_mode, mirror
                        )
                        timing["forward_seconds"] += time.time() - t0
                        timing["n_forwards"] += int(batch.surface_x.shape[0])

                        # p = clean − δ_k
                        set_perturbed_(eval_module, clean, delta, -1.0)
                        t0 = time.time()
                        surf_minus, vol_minus = _forward_real(
                            eval_module, model_input, transform, device, cfg.amp_mode, mirror
                        )
                        timing["forward_seconds"] += time.time() - t0
                        timing["n_forwards"] += int(batch.surface_x.shape[0])

                        # Restore clean exactly (snapshot copy) before any further work.
                        restore_clean_params(eval_module, clean)

                        if verify_restore_this_case and k == 0:
                            assert_clean(
                                eval_module, clean,
                                where=f"case={case_id} pair=0 res={K} mirror={mirror}",
                            )

                        surf_mid = (surf_plus + surf_minus) * 0.5
                        vol_mid = (vol_plus + vol_minus) * 0.5

                        _scatter_into_buffers(
                            case_id=case_id, K_res=K, batch=batch,
                            surface_pred_real=surf_mid, volume_pred_real=vol_mid,
                            n_surf=n_surf, n_vol=n_vol,
                            surf_sum=surf_anti_sum, surf_cnt=surf_anti_cnt,
                            vol_sum=vol_anti_sum, vol_cnt=vol_anti_cnt,
                            metas_cached=metas_cached, timing=timing,
                        )
    finally:
        # Always restore clean weights so subsequent cases start unperturbed.
        restore_clean_params(eval_module, clean)
        del deltas

    if int(surf_base_cnt.min()) == 0 or int(surf_anti_cnt.min()) == 0:
        raise RuntimeError(
            f"surface coverage incomplete for case {case_id}: "
            f"base_min={int(surf_base_cnt.min())} anti_min={int(surf_anti_cnt.min())}"
        )
    if int(vol_base_cnt.min()) == 0 or int(vol_anti_cnt.min()) == 0:
        raise RuntimeError(
            f"volume coverage incomplete for case {case_id}: "
            f"base_min={int(vol_base_cnt.min())} anti_min={int(vol_anti_cnt.min())}"
        )

    surf_avg_base = surf_base_sum / surf_base_cnt.unsqueeze(-1).to(torch.float32)
    vol_avg_base = vol_base_sum / vol_base_cnt.unsqueeze(-1).to(torch.float32)
    surf_avg_anti = surf_anti_sum / surf_anti_cnt.unsqueeze(-1).to(torch.float32)
    vol_avg_anti = vol_anti_sum / vol_anti_cnt.unsqueeze(-1).to(torch.float32)

    case = store.load_case(case_id)
    surf_y = case.surface_y.float()
    vol_y = case.volume_y.float()

    # n_passes_total (informational)
    # Approximated as n_passes_per_batch × number of batches across all resolutions.
    # Not strictly needed for math; reported in timing log.
    return (
        surf_avg_base, surf_avg_anti,
        vol_avg_base, vol_avg_anti,
        surf_y, vol_y,
        n_passes_per_batch,  # passes per (case-batch) — printed for diagnostic
        timing,
    )


# --- Metric accumulation per (lambda, split) ---


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
    """Accumulate metrics for one fully-predicted case into ``acc`` (full-case, no chunk masks)."""
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


def evaluate_split_taylor2(
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
    sigma: float,
    K_pairs: int,
    seed_base: int,
    lambdas: list[float],
    clean: dict[str, torch.Tensor],
    distributed_state,
) -> dict[float, dict[str, float]]:
    """Distribute cases across ranks, compute base+anti per case, λ-sweep post-hoc."""
    rank = distributed_state.rank if distributed_state and distributed_state.enabled else 0
    world_size = (
        distributed_state.world_size
        if distributed_state and distributed_state.enabled
        else 1
    )
    my_cases = case_ids[rank::world_size]

    # One accumulator per lambda value.
    accs: dict[float, EvalAccumulator] = {lam: EvalAccumulator() for lam in lambdas}

    total_t0 = time.time()
    total_forward = 0.0
    total_io = 0.0
    total_perturb = 0.0
    total_forwards = 0
    for case_idx, case_id in enumerate(my_cases):
        verify_this_case = bool(cfg.verify_restore) and case_idx == 0
        t0 = time.time()
        (
            surf_base, surf_anti,
            vol_base, vol_anti,
            surf_y, vol_y,
            n_passes_per_batch, timing,
        ) = process_case_taylor2(
            case_id=case_id,
            case_index=case_idx,
            model=model,
            transform=transform,
            device=device,
            store=store,
            cfg=cfg,
            resolutions=resolutions,
            use_mirror=use_mirror,
            sigma=sigma,
            K_pairs=K_pairs,
            seed_base=seed_base,
            clean=clean,
            verify_restore_this_case=verify_this_case,
        )

        # Sweep lambda post-hoc.
        for lam in lambdas:
            surf_lam = (1.0 - lam) * surf_anti + lam * surf_base
            vol_lam = (1.0 - lam) * vol_anti + lam * vol_base
            add_case_to_accumulator(
                accs[lam],
                case_id=case_id,
                surface_pred_real=surf_lam,
                volume_pred_real=vol_lam,
                surface_y=surf_y,
                volume_y=vol_y,
                transform=transform,
            )

        total_forward += timing["forward_seconds"]
        total_io += timing["io_seconds"]
        total_perturb += timing.get("perturb_seconds", 0.0)
        total_forwards += timing["n_forwards"]
        dt = time.time() - t0
        print(
            f"  [rank {rank}] {split_name} {case_id} ({case_idx + 1}/{len(my_cases)}): "
            f"forwards={timing['n_forwards']} passes_per_batch={n_passes_per_batch} "
            f"perturb={timing.get('perturb_seconds', 0.0):.1f}s "
            f"forward={timing['forward_seconds']:.1f}s io={timing['io_seconds']:.1f}s "
            f"total={dt:.1f}s",
            flush=True,
        )

    print(
        f"[rank {rank}] {split_name} complete: {len(my_cases)} cases in "
        f"{time.time() - total_t0:.1f}s "
        f"(forward={total_forward:.1f}s io={total_io:.1f}s perturb={total_perturb:.1f}s "
        f"n_forwards={total_forwards})",
        flush=True,
    )

    # Gather each lambda's accumulator across ranks.
    out: dict[float, dict[str, float]] = {}
    for lam in lambdas:
        acc = accs[lam]
        if distributed_state is not None and distributed_state.enabled:
            gathered: list[EvalAccumulator | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, acc)
            if not distributed_state.is_main:
                out[lam] = {}
                continue
            merged = merge_eval_accumulators(g for g in gathered if g is not None)
        else:
            merged = acc
        out[lam] = finalize_eval_accumulator(merged)

    return out


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device

    resolutions = parse_resolutions(cfg.resolutions)
    modes = parse_modes(cfg.eval_modes)
    valid_modes = ("weight_noise_res_avg", "weight_noise_mirror_res_avg")
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Unknown eval mode: {mode!r} (valid: {valid_modes})")
    lambdas = parse_lambdas(cfg.taylor_lambda_sweep, cfg.taylor_lambda)
    # Always include λ=0 as the anti-thetic floor for direct comparison.
    if 0.0 not in lambdas:
        lambdas = [0.0] + lambdas

    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        print(f"Checkpoint: {cfg.checkpoint}")
        print(f"Resolutions: {resolutions}")
        print(f"Modes: {modes}")
        print(
            f"H273 knobs: sigma_rel={cfg.weight_noise_sigma} K_pairs={cfg.n_antithetic_pairs} "
            f"seed_base={cfg.weight_noise_seed_base} lambdas={lambdas}"
        )

    ckpt_path = Path(cfg.checkpoint)
    if state.is_main and not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {ckpt_path}")

    # Build store and normalization stats.
    store = DrivAerMLCaseStore(manifest_path=cfg.manifest, root=cfg.data_root or None)
    from data.loader import target_stats_from_normalizers  # noqa: PLC0415 — late import
    stats = target_stats_from_normalizers(store)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    # Build and load model.
    model = build_model(cfg).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ck["model"]
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if state.is_main:
        print(f"Loaded checkpoint epoch={ck.get('epoch')} source={ck.get('checkpoint_source')}")
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
        print(f"Snapshotted {len(clean)} floating-point tensors ({n_perturbed:,} elements)")

    val_case_ids = store.case_ids("val")
    test_case_ids = store.case_ids("test")
    if cfg.debug:
        val_case_ids = val_case_ids[:2]
        test_case_ids = test_case_ids[:2]
        if state.is_main:
            print(
                f"DEBUG: limiting to {len(val_case_ids)} val + {len(test_case_ids)} test cases"
            )

    # W&B init on rank 0.
    run = None
    if state.is_main:
        run_name = cfg.wandb_name or (
            f"{cfg.agent}/h273-taylor2-K{cfg.n_antithetic_pairs}-"
            f"sigma{cfg.weight_noise_sigma}-modes{','.join(modes)}"
        )
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", cfg.wandb_project),
            entity=os.environ.get("WANDB_ENTITY", cfg.wandb_entity),
            group=cfg.wandb_group,
            name=run_name,
            config={
                **asdict(cfg),
                "checkpoint_run_id": "yw2a5dyl",
                "checkpoint_epoch": ck.get("epoch"),
                "checkpoint_source": ck.get("checkpoint_source"),
                "resolutions_list": resolutions,
                "modes_list": modes,
                "lambdas_list": lambdas,
                "n_val_cases": len(val_case_ids),
                "n_test_cases": len(test_case_ids),
            },
            tags=["h273", "tta", "anti-thetic", "taylor2", "eval-only", cfg.agent],
            reinit="finish_previous",
        )

    splits = [
        ("val_surface", val_case_ids, "full_val"),
        ("test_surface", test_case_ids, "test"),
    ]

    # summary[split][mode][lambda] -> metrics dict
    summary: dict[str, dict[str, dict[float, dict[str, float]]]] = {}
    for split_name, case_ids, log_prefix in splits:
        summary[split_name] = {}
        for mode in modes:
            use_mirror = mode == "weight_noise_mirror_res_avg"
            if state.is_main:
                print(
                    f"\n=== Evaluating split={split_name} mode={mode} "
                    f"resolutions={resolutions} mirror={use_mirror} ===",
                    flush=True,
                )
            t0 = time.time()
            metrics_by_lambda = evaluate_split_taylor2(
                split_name=f"{split_name}/{mode}",
                case_ids=case_ids,
                model=model,
                transform=transform,
                device=device,
                store=store,
                cfg=cfg,
                resolutions=resolutions,
                use_mirror=use_mirror,
                sigma=cfg.weight_noise_sigma,
                K_pairs=cfg.n_antithetic_pairs,
                seed_base=cfg.weight_noise_seed_base,
                lambdas=lambdas,
                clean=clean,
                distributed_state=state,
            )
            dt = time.time() - t0
            if state.is_main:
                print(f"  total {split_name}/{mode}: {dt:.1f}s")
                summary[split_name][mode] = metrics_by_lambda
                log_obj: dict[str, float] = {}
                for lam, metrics in metrics_by_lambda.items():
                    lam_tag = f"lambda{lam:.3f}".replace(".", "p")
                    print(f"  -- {mode} λ={lam:.3f} --")
                    print_metrics(f"{split_name}/{mode}/{lam_tag}", metrics)
                    log_obj.update(
                        primary_metric_log(
                            f"{log_prefix}_primary/{mode}/{lam_tag}", metrics
                        )
                    )
                    log_obj[f"{log_prefix}_extra/{mode}/{lam_tag}/loss"] = metrics["loss"]
                log_obj[f"{log_prefix}_extra/{mode}/seconds"] = dt
                if run is not None:
                    wandb.log(log_obj)

    # Restore clean weights once globally before exiting.
    restore_clean_params(eval_module, clean)

    if state.is_main and summary:
        print("\n=== Summary (rel_l2_pct lower-is-better) ===")
        for split, modes_dict in summary.items():
            for mode, lambda_dict in modes_dict.items():
                print(f"\n[{split} / {mode}]")
                keys_to_show = (
                    "abupt_axis_mean_rel_l2_pct",
                    "surface_pressure_rel_l2_pct",
                    "wall_shear_rel_l2_pct",
                    "wall_shear_x_rel_l2_pct",
                    "wall_shear_y_rel_l2_pct",
                    "wall_shear_z_rel_l2_pct",
                    "volume_pressure_rel_l2_pct",
                )
                lam_keys = sorted(lambda_dict.keys())
                header = (
                    f"  {'metric':<36s} "
                    + " ".join(f"λ={l:>6.3f}" for l in lam_keys)
                )
                print(header)
                for k in keys_to_show:
                    row = f"  {k:<36s} " + " ".join(
                        f"{lambda_dict[l][k]:>9.4f}" for l in lam_keys
                    )
                    print(row)

        if run is not None:
            for split, modes_dict in summary.items():
                for mode, lambda_dict in modes_dict.items():
                    for lam, metrics in lambda_dict.items():
                        lam_tag = f"lambda{lam:.3f}".replace(".", "p")
                        for k, v in metrics.items():
                            try:
                                run.summary[f"{split}/{mode}/{lam_tag}/{k}"] = float(v)
                            except Exception:
                                pass
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
