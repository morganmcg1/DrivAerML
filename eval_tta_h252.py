# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H252: Stacked weight-noise × multi-res × mirror TTA on H185 EP13.

Builds on H236 (multi-res × mirror) by adding an outer K-pass weight-perturbation
loop. For each case the total prediction is averaged over K × R × M passes:

    for k in 0..K-1:                # weight perturbation index
        perturb model: p <- clean_p + randn(seed_k) * sigma * |clean_p|
        for R in resolutions:       # eval_volume_points value
            for mirror in (False, True):
                forward; un-mirror if needed; accumulate per-point in real units

Perturbed weights are deterministic across DDP ranks (Generator seeded by
`(seed_base + k) * 100003`), and reused across (res, mirror) inner passes.
The same per-point chunk-accumulation logic as eval_multi_res.py covers every
surface and volume point across the (R, mirror) sub-grid; the K outer loop just
adds K independent perturbed-weight samples to the average.

Mirror convention (H148/H183/H209), unchanged:
    surface_x [x, y, z, nx, ny, nz, area] -> negate y(1) and ny(4)
    volume_x  [x, y, z, sdf]              -> negate y(1)
    surface_y predictions [cp, tau_x, tau_y, tau_z] -> un-mirror by negating tau_y(2)
    volume_y predictions [volume_pressure] -> invariant

Modes:
    weight_noise_only:           K passes at a single eval_volume_points
                                 (no mirror). Sanity-reproduces H242 noise_only.
    weight_noise_mirror_res_avg: K x R x 2 = full stacked TTA.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
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
    """H252 eval config — extends H236 with weight-noise stacking."""

    # Checkpoint: local path first (path exists), else W&B fetch via run_id+alias.
    checkpoint: str = "runs/h210/artifacts/h185/checkpoint.pt"
    run_id: str = "yw2a5dyl"
    checkpoint_alias: str = "epoch-13"
    cache_root: str = "outputs/h252_eval/_artifacts"

    # Multi-resolution TTA configuration
    resolutions: str = "49152,65536,81920"
    eval_modes: str = "weight_noise_mirror_res_avg"
    eval_surface_points: int = 65536

    # Weight-noise stacking knobs
    weight_noise_sigma: float = 5e-4
    weight_noise_passes: int = 5
    weight_noise_seed_base: int = 42
    # H274: anti-thetic noise pairs. When True, each of weight_noise_passes
    # samples ε once and evaluates BOTH f(θ+ε) and f(θ−ε), so the effective
    # number of forward passes per (res, mirror) is 2 * weight_noise_passes.
    # Anti-thetic pairs cancel the linear Taylor term, reducing variance.
    antithetic_noise: bool = False

    # H300: per-channel affine calibration. Fits alpha_c, beta_c via OLS on val
    # predictions vs val targets (after TTA aggregation), then applies the SAME
    # (alpha, beta) to test predictions before metric computation. Logs both raw
    # and calibrated rel_l2-family metrics; MAE is not analytically expressible
    # under affine + sufficient stats and is reported only on the raw output.
    test_time_calibration: bool = False
    # H313: regional/zonal per-channel calibration on top of H300. Accepts a
    # comma-separated list of zone counts, e.g. "6" or "4,6"; the TTA outputs
    # are aggregated into per-(N_zone, zone, channel) sufficient stats once,
    # and an OLS (alpha, beta) is fit per zone x surface channel for each N
    # in the list. Volume pressure keeps the H300 global affine (no volume
    # zoning, per researcher recommendation given tight test_VP headroom).
    # Supported partitions: 4 (nose/rear/underbody/body) and 6
    # (nose/roof/sides/underbody/rear/body).
    regional_num_zones: str = ""
    # H328: ridge-regularized regional calibration. CSV of ridge_lambda values
    # to sweep on the SAME per-zone val sufficient stats. Each (zone, channel)
    # (alpha, beta) is shrunk toward the H300 global per-channel fit:
    #   min_{α,β} Σ (α p + β - t)^2 + λ [(α-α_global_c)^2 + (β-β_global_c)^2]
    # Solving the normal equations gives a closed-form shrinkage per zone.
    # λ=0 reproduces H313 unregularised; λ→∞ collapses to H300 global per
    # channel. The token "inf" is accepted to request the global-collapse arm.
    regional_ridge_lambda_sweep: str = "0.0"

    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h252_eval"
    wandb_group: str = "h242-tanjiro-stacked-tta"
    wandb_name: str = ""
    wandb_project: str = "senpai-v1-drivaerml-ddp8"
    wandb_entity: str = "wandb-applied-ai-team"
    agent: str = "tanjiro"

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
    parser = argparse.ArgumentParser(description="H252 stacked weight-noise x multi-res x mirror eval")
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


def parse_zone_counts(spec: str) -> list[int]:
    """Parse a CSV like "4,6" into [4, 6]. Empty string -> []."""
    out: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        n = int(piece)
        if n not in (4, 6):
            raise ValueError(
                f"regional_num_zones values must be 4 or 6 (got {n!r}); supported "
                "partitions are 4 (nose/rear/underbody/body) and 6 "
                "(nose/roof/sides/underbody/rear/body)."
            )
        out.append(n)
    return out


def parse_ridge_lambdas(spec: str) -> list[float]:
    """Parse a CSV like "0.0,0.01,1.0,inf" into [0.0, 0.01, 1.0, float('inf')].

    Empty string -> [0.0] (default to OLS / H313 reproduction).
    Negative values are rejected. The literal token "inf" (case-insensitive)
    expands to `float('inf')` and means "shrink fully to the global per-channel
    fit", which reproduces H300.
    """
    out: list[float] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.lower() in ("inf", "+inf", "infinity"):
            out.append(float("inf"))
            continue
        v = float(piece)
        if v < 0.0:
            raise ValueError(f"regional_ridge_lambda must be >= 0 (got {v!r})")
        out.append(v)
    if not out:
        out.append(0.0)
    return out


def _fmt_lambda(lam: float) -> str:
    """Human-friendly tag for a ridge lambda value (used in W&B keys/logs)."""
    if lam == float("inf"):
        return "inf"
    if lam == 0.0:
        return "0p0"
    s = f"{lam:g}"
    return s.replace(".", "p").replace("+", "").replace("-", "neg")


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
def perturb_relative_(
    module: nn.Module,
    clean: dict[str, torch.Tensor],
    sigma: float,
    seed: int,
    device: torch.device,
    sign: float = 1.0,
) -> None:
    """p <- clean_p + sign * randn(seed) * sigma * |clean_p|, identical across DDP ranks.

    With matched ``seed`` and ``sign=±1.0`` calls, produces an anti-thetic pair
    whose linear Taylor term cancels in the average (Finding NN / H274).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for name, p in module.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        clean_p = clean[name]
        noise = torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype)
        noise.mul_(sigma * sign).mul_(clean_p.abs())
        p.data.copy_(clean_p).add_(noise)


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
    sigma: float,
    seed_base: int,
    antithetic: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    dict,
]:
    """Run K_eff x R x M (mirror) TTA passes for one case and return per-point
    averaged predictions in real units. The outer K loop perturbs the model
    weights and reuses the perturbed copy across all (res, mirror) inner passes.

    With ``antithetic=False`` K_eff = ``K_passes`` (i.i.d. samples). With
    ``antithetic=True`` K_eff = ``2 * K_passes``: for each k we sample ε once
    (same seed) and evaluate both f(θ+ε) and f(θ−ε), so the linear Taylor term
    cancels in the per-point average (Finding NN / H274).
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

    # Build the (noise_idx, sign) schedule. Anti-thetic pairs share a seed so
    # +ε / −ε use the SAME draw of ε (otherwise it would just be 2K i.i.d.).
    if antithetic and sigma > 0.0:
        schedule = [(k, sign) for k in range(K_passes) for sign in (+1.0, -1.0)]
    else:
        schedule = [(k, +1.0) for k in range(K_passes)]

    for k, sign in schedule:
        # Perturb once per (k, sign); reuse across (res, mirror) inner passes.
        t_p = time.time()
        if sigma > 0.0:
            seed = (seed_base + k) * 100003
            perturb_relative_(
                eval_module, clean, sigma=sigma, seed=seed, device=device, sign=sign
            )
        else:
            restore_clean_params(eval_module, clean)
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
                                    f"surface global-index mismatch case={case_id} K={R} "
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
                                    f"volume global-index mismatch case={case_id} K={R} "
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
    surf_xyz = case.surface_x[:, 0:3].float()

    k_eff = K_passes * (2 if (antithetic and sigma > 0.0) else 1)
    n_passes = k_eff * len(resolutions) * (2 if use_mirror else 1)
    return surf_avg, vol_avg, surf_y, vol_y, surf_xyz, n_passes, timing


# --- H300 affine calibration helpers ---


# Surface preds are [cp, tau_x, tau_y, tau_z]; volume preds are [volume_pressure].
N_SURF_CHANNELS = 4
N_VOL_CHANNELS = 1
# Sufficient-stats column layout: [N, sum_p, sum_t, sum_pp, sum_tt, sum_pt].
N_STAT_COLS = 6


@dataclass
class CalibrationStats:
    """Per-case per-channel sufficient stats for fitting an affine OLS
    calibration y_hat = alpha * y + beta and recomputing per-case rel_l2
    metrics under that affine map. Surface has 4 channels (cp, tau_x, tau_y,
    tau_z), volume has 1 (volume_pressure).

    Optionally also tracks H313 per-zone surface stats when zone assignments
    are provided. ``surf_zonal_global`` is a dict keyed by partition size
    (n_zones) with values of shape (n_zones, N_SURF_CHANNELS, N_STAT_COLS).
    ``surf_zonal_per_case`` is a dict[n_zones -> dict[case_id -> tensor of
    same shape]].
    """

    surf_global: torch.Tensor = field(
        default_factory=lambda: torch.zeros(N_SURF_CHANNELS, N_STAT_COLS, dtype=torch.float64)
    )
    vol_global: torch.Tensor = field(
        default_factory=lambda: torch.zeros(N_VOL_CHANNELS, N_STAT_COLS, dtype=torch.float64)
    )
    surf_per_case: dict[str, torch.Tensor] = field(default_factory=dict)
    vol_per_case: dict[str, torch.Tensor] = field(default_factory=dict)
    surf_zonal_global: dict[int, torch.Tensor] = field(default_factory=dict)
    surf_zonal_per_case: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)


def _channel_stats(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return (n_channels, 6) tensor with [N, sum_p, sum_t, sum_pp, sum_tt, sum_pt].

    pred, target: (N_points, n_channels) tensors in real units. Computed in float64
    so accumulation across cases / DDP ranks does not lose precision.
    """
    p = pred.double()
    t = target.double()
    n_points = p.shape[0]
    n_ch = p.shape[1]
    stats = torch.empty(n_ch, N_STAT_COLS, dtype=torch.float64)
    stats[:, 0] = float(n_points)
    stats[:, 1] = p.sum(0)
    stats[:, 2] = t.sum(0)
    stats[:, 3] = (p * p).sum(0)
    stats[:, 4] = (t * t).sum(0)
    stats[:, 5] = (p * t).sum(0)
    return stats


def _channel_stats_zonal(
    pred: torch.Tensor, target: torch.Tensor, zone: torch.Tensor, n_zones: int
) -> torch.Tensor:
    """Per-zone sufficient stats. Returns (n_zones, n_channels, 6).

    Each zone slot holds the same six columns as `_channel_stats` summed only
    over points assigned to that zone. Empty zones produce all-zero rows.
    """
    p = pred.double()
    t = target.double()
    n_ch = p.shape[1]
    stats = torch.zeros(n_zones, n_ch, N_STAT_COLS, dtype=torch.float64)
    for z_idx in range(n_zones):
        mask = zone == z_idx
        if not mask.any():
            continue
        p_z = p[mask]
        t_z = t[mask]
        stats[z_idx, :, 0] = float(p_z.shape[0])
        stats[z_idx, :, 1] = p_z.sum(0)
        stats[z_idx, :, 2] = t_z.sum(0)
        stats[z_idx, :, 3] = (p_z * p_z).sum(0)
        stats[z_idx, :, 4] = (t_z * t_z).sum(0)
        stats[z_idx, :, 5] = (p_z * t_z).sum(0)
    return stats


def compute_zone_thresholds(
    store: DrivAerMLCaseStore, case_ids: Iterable[str]
) -> dict[str, float]:
    """Pool surface xyz across ``case_ids`` and return percentile thresholds
    used by `assign_zones`. Keys: x_20, x_80, z_20, z_70, abs_y_60.
    """
    import numpy as np  # noqa: PLC0415 — small local import keeps top-of-file clean

    xyz_chunks: list[np.ndarray] = []
    for case_id in case_ids:
        case = store.load_case(case_id)
        xyz_chunks.append(case.surface_x[:, 0:3].numpy())
    xyz = np.concatenate(xyz_chunks, axis=0)
    return {
        "x_20": float(np.percentile(xyz[:, 0], 20)),
        "x_80": float(np.percentile(xyz[:, 0], 80)),
        "z_20": float(np.percentile(xyz[:, 2], 20)),
        "z_70": float(np.percentile(xyz[:, 2], 70)),
        "abs_y_60": float(np.percentile(np.abs(xyz[:, 1]), 60)),
    }


def assign_zones(
    xyz: torch.Tensor, thresholds: dict[str, float], n_zones: int
) -> torch.Tensor:
    """Assign each surface point to a spatial zone (returned as int64 indices).

    n_zones=6:
        0 = nose      (x < x_20)
        1 = roof      (mid-x band, z > z_70)
        2 = sides     (mid-x band, |y| > abs_y_60, z in [z_20, z_70])
        3 = underbody (mid-x band, z < z_20)
        4 = rear/wake (x > x_80)
        5 = body remainder (default)

    n_zones=4 (collapses roof/sides into body):
        0 = nose      (x < x_20)
        1 = rear/wake (x > x_80)
        2 = underbody (mid-x band, z < z_20)
        3 = body remainder (default)

    The mid-x and underbody/roof rules are applied first (lower priority); the
    boundary x rules overwrite afterwards (higher priority). Points falling
    only inside the body remainder go to the default index.
    """
    if n_zones not in (4, 6):
        raise ValueError(f"regional_num_zones must be 4 or 6, got {n_zones}")
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    abs_y = y.abs()
    x_20 = thresholds["x_20"]
    x_80 = thresholds["x_80"]
    z_20 = thresholds["z_20"]
    z_70 = thresholds["z_70"]
    abs_y_60 = thresholds["abs_y_60"]
    mid_x = (x >= x_20) & (x <= x_80)
    if n_zones == 6:
        zone = torch.full((xyz.shape[0],), 5, dtype=torch.long)
        zone[mid_x & (abs_y > abs_y_60)] = 2
        zone[mid_x & (z < z_20)] = 3
        zone[mid_x & (z > z_70)] = 1
        zone[x < x_20] = 0
        zone[x > x_80] = 4
    else:
        zone = torch.full((xyz.shape[0],), 3, dtype=torch.long)
        zone[mid_x & (z < z_20)] = 2
        zone[x < x_20] = 0
        zone[x > x_80] = 1
    return zone


def add_case_to_calibration(
    cal: CalibrationStats,
    *,
    case_id: str,
    surface_pred_real: torch.Tensor,
    volume_pred_real: torch.Tensor,
    surface_y: torch.Tensor,
    volume_y: torch.Tensor,
    surface_zones_by_n: dict[int, torch.Tensor] | None = None,
) -> None:
    surf_stats = _channel_stats(surface_pred_real, surface_y)
    vol_stats = _channel_stats(volume_pred_real, volume_y)
    cal.surf_per_case[case_id] = surf_stats
    cal.vol_per_case[case_id] = vol_stats
    cal.surf_global += surf_stats
    cal.vol_global += vol_stats
    if surface_zones_by_n:
        for n_zones, zone in surface_zones_by_n.items():
            zonal_stats = _channel_stats_zonal(
                surface_pred_real, surface_y, zone, n_zones
            )
            cal.surf_zonal_per_case.setdefault(n_zones, {})[case_id] = zonal_stats
            if n_zones not in cal.surf_zonal_global:
                cal.surf_zonal_global[n_zones] = torch.zeros_like(zonal_stats)
            cal.surf_zonal_global[n_zones] = (
                cal.surf_zonal_global[n_zones] + zonal_stats
            )


def merge_calibration_stats(parts: Iterable[CalibrationStats]) -> CalibrationStats:
    merged = CalibrationStats()
    for part in parts:
        merged.surf_global += part.surf_global
        merged.vol_global += part.vol_global
        merged.surf_per_case.update(part.surf_per_case)
        merged.vol_per_case.update(part.vol_per_case)
        for n_zones, gstats in part.surf_zonal_global.items():
            if n_zones not in merged.surf_zonal_global:
                merged.surf_zonal_global[n_zones] = torch.zeros_like(gstats)
            merged.surf_zonal_global[n_zones] = (
                merged.surf_zonal_global[n_zones] + gstats
            )
        for n_zones, by_case in part.surf_zonal_per_case.items():
            merged.surf_zonal_per_case.setdefault(n_zones, {}).update(by_case)
    return merged


def fit_affine_per_channel(
    global_stats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OLS alpha, beta per channel from the global sufficient stats.

    alpha = cov(p, t) / var(p);  beta = mu_t - alpha * mu_p, all per channel.
    Falls back to identity (alpha=1, beta=0) on any degenerate channel where
    var(p) is non-positive (constant predictions).
    """
    N = global_stats[:, 0]
    sum_p = global_stats[:, 1]
    sum_t = global_stats[:, 2]
    sum_pp = global_stats[:, 3]
    sum_pt = global_stats[:, 5]
    N_safe = N.clamp(min=1.0)
    mu_p = sum_p / N_safe
    mu_t = sum_t / N_safe
    cov_pt = sum_pt - N * mu_p * mu_t
    var_p = sum_pp - N * mu_p * mu_p
    degenerate = var_p <= 1e-12
    alpha_raw = cov_pt / var_p.clamp(min=1e-12)
    alpha = torch.where(degenerate, torch.ones_like(alpha_raw), alpha_raw)
    beta = torch.where(degenerate, torch.zeros_like(alpha_raw), mu_t - alpha * mu_p)
    return alpha, beta


def _affine_error_sq(
    stats: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Sum_p (alpha * p + beta - t)^2 per channel given sufficient stats.

    stats: (..., 6) with [N, sum_p, sum_t, sum_pp, sum_tt, sum_pt].
    alpha, beta: broadcastable to stats[..., 0] shape.
    """
    N = stats[..., 0]
    sum_p = stats[..., 1]
    sum_t = stats[..., 2]
    sum_pp = stats[..., 3]
    sum_tt = stats[..., 4]
    sum_pt = stats[..., 5]
    return (
        alpha * alpha * sum_pp
        + 2.0 * alpha * beta * sum_p
        - 2.0 * alpha * sum_pt
        + beta * beta * N
        - 2.0 * beta * sum_t
        + sum_tt
    )


def fit_affine_per_zone_per_channel(
    zonal_global_stats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised OLS over (n_zones, n_channels) sufficient stats.

    zonal_global_stats: (n_zones, n_channels, 6) with the same column layout
    as `_channel_stats`. Returns alpha, beta of shape (n_zones, n_channels).
    Empty or near-constant zones (var_p <= 1e-12) fall back to identity.
    """
    N = zonal_global_stats[..., 0]
    sum_p = zonal_global_stats[..., 1]
    sum_t = zonal_global_stats[..., 2]
    sum_pp = zonal_global_stats[..., 3]
    sum_pt = zonal_global_stats[..., 5]
    N_safe = N.clamp(min=1.0)
    mu_p = sum_p / N_safe
    mu_t = sum_t / N_safe
    cov_pt = sum_pt - N * mu_p * mu_t
    var_p = sum_pp - N * mu_p * mu_p
    degenerate = var_p <= 1e-12
    alpha_raw = cov_pt / var_p.clamp(min=1e-12)
    alpha = torch.where(degenerate, torch.ones_like(alpha_raw), alpha_raw)
    beta = torch.where(degenerate, torch.zeros_like(alpha_raw), mu_t - alpha * mu_p)
    return alpha, beta


def fit_affine_per_zone_per_channel_ridge(
    zonal_global_stats: torch.Tensor,
    alpha_global_c: torch.Tensor,
    beta_global_c: torch.Tensor,
    ridge_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised RIDGE fit per (zone, channel) shrunk toward the H300 global
    per-channel (alpha_global_c, beta_global_c).

    For each (zone z, channel c), solves the closed-form normal equations:

        [[sum_pp + λ, sum_p],      [α_zc]     [sum_pt + λ·α_global_c]
         [sum_p,      n + λ ]]  @  [β_zc]  =  [sum_t  + λ·β_global_c ]

    At λ=0 this reduces to OLS exactly (matches H313). At λ→∞ the 2x2 system
    collapses to (α_zc, β_zc) = (α_global_c, β_global_c) — i.e. every zone
    folds back to the H300 global per-channel fit, recovering H300 metrics.

    Args:
        zonal_global_stats: (n_zones, n_channels, 6) sufficient stats with
            columns [N, sum_p, sum_t, sum_pp, sum_tt, sum_pt].
        alpha_global_c: (n_channels,) — H300 per-channel global α fit.
        beta_global_c:  (n_channels,) — H300 per-channel global β fit.
        ridge_lambda: shrinkage strength (>= 0; float('inf') triggers the
            analytic collapse to the global fit).

    Returns:
        (alpha, beta), each of shape (n_zones, n_channels).
    """
    if ridge_lambda < 0.0:
        raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda!r}")

    n_zones, n_ch = zonal_global_stats.shape[0], zonal_global_stats.shape[1]
    a_global = alpha_global_c.to(zonal_global_stats.dtype).reshape(1, n_ch).expand(n_zones, n_ch)
    b_global = beta_global_c.to(zonal_global_stats.dtype).reshape(1, n_ch).expand(n_zones, n_ch)

    if ridge_lambda == float("inf"):
        # At λ=∞ every (zone, channel) collapses to the global per-channel fit.
        return a_global.clone(), b_global.clone()

    N = zonal_global_stats[..., 0]
    sum_p = zonal_global_stats[..., 1]
    sum_t = zonal_global_stats[..., 2]
    sum_pp = zonal_global_stats[..., 3]
    sum_pt = zonal_global_stats[..., 5]

    # A = [[A11, A12],[A21, A22]], b = [b1, b2] per (zone, channel)
    A11 = sum_pp + ridge_lambda
    A12 = sum_p
    A21 = sum_p
    A22 = N + ridge_lambda
    b1 = sum_pt + ridge_lambda * a_global
    b2 = sum_t + ridge_lambda * b_global

    det = A11 * A22 - A12 * A21
    # det = (sum_pp + λ)(N + λ) - sum_p^2. For λ>0 and any non-empty zone this
    # is strictly positive (Cauchy–Schwarz: sum_pp · N >= sum_p^2). For λ=0
    # with an empty/constant zone det can be ~0; fall back to the global fit
    # in that case (consistent with the H313 degenerate-zone fallback).
    degenerate = det.abs() <= 1e-18
    det_safe = torch.where(degenerate, torch.ones_like(det), det)
    alpha_raw = (A22 * b1 - A12 * b2) / det_safe
    beta_raw = (A11 * b2 - A21 * b1) / det_safe
    alpha = torch.where(degenerate, a_global, alpha_raw)
    beta = torch.where(degenerate, b_global, beta_raw)
    return alpha, beta


def compute_calibrated_metrics_regional(
    cal: CalibrationStats,
    n_zones: int,
    alpha_surf_zonal: torch.Tensor,
    beta_surf_zonal: torch.Tensor,
    alpha_vol: torch.Tensor,
    beta_vol: torch.Tensor,
) -> dict[str, float]:
    """Per-case rel_l2 under H313 regional surface calibration + H300 global
    volume calibration. Per-zone (alpha, beta) maps are summed in error space
    inside each case before computing the per-channel rel_l2 ratio. Mirrors
    `compute_calibrated_metrics` keys for downstream parsing.
    """
    surf_zonal_per_case = cal.surf_zonal_per_case.get(n_zones)
    if not surf_zonal_per_case:
        raise RuntimeError(
            f"CalibrationStats has no zonal stats for n_zones={n_zones}; cannot compute regional metrics"
        )
    surf_ids = sorted(surf_zonal_per_case.keys())

    surf_per_channel_rel_l2: list[torch.Tensor] = []
    vec_rel_l2_list: list[torch.Tensor] = []
    for case_id in surf_ids:
        s = surf_zonal_per_case[case_id]  # (n_zones, 4, 6)
        e_sq_z = _affine_error_sq(s, alpha_surf_zonal, beta_surf_zonal)  # (n_zones, 4)
        e_sq_total = e_sq_z.sum(dim=0)  # (4,)
        t_sq_total = s[..., 4].sum(dim=0)  # (4,)
        rel = torch.sqrt(e_sq_total.clamp(min=0.0) / t_sq_total.clamp(min=1e-12))
        surf_per_channel_rel_l2.append(rel)

        s_vec = s[:, 1:4]  # (n_zones, 3, 6)
        e_sq_vec_z = _affine_error_sq(
            s_vec, alpha_surf_zonal[:, 1:4], beta_surf_zonal[:, 1:4]
        )  # (n_zones, 3)
        e_sq_vec_total = e_sq_vec_z.sum()
        t_sq_vec_total = s_vec[..., 4].sum()
        rel_vec = torch.sqrt(
            e_sq_vec_total.clamp(min=0.0) / t_sq_vec_total.clamp(min=1e-12)
        )
        vec_rel_l2_list.append(rel_vec)
    surf_rel_l2 = torch.stack(surf_per_channel_rel_l2, dim=0)
    vec_rel_l2 = torch.stack(vec_rel_l2_list, dim=0)

    vol_ids = sorted(cal.vol_per_case.keys())
    vol_per_channel_rel_l2: list[torch.Tensor] = []
    for case_id in vol_ids:
        v = cal.vol_per_case[case_id]  # (1, 6)
        e_sq = _affine_error_sq(v, alpha_vol, beta_vol)
        t_sq = v[..., 4]
        rel = torch.sqrt(e_sq.clamp(min=0.0) / t_sq.clamp(min=1e-12))
        vol_per_channel_rel_l2.append(rel)
    vol_rel_l2 = torch.stack(vol_per_channel_rel_l2, dim=0)

    sp = float(surf_rel_l2[:, 0].mean().item())
    tx = float(surf_rel_l2[:, 1].mean().item())
    ty = float(surf_rel_l2[:, 2].mean().item())
    tz = float(surf_rel_l2[:, 3].mean().item())
    ws_vec = float(vec_rel_l2.mean().item())
    vp = float(vol_rel_l2[:, 0].mean().item())
    abupt = (sp + tx + ty + tz + vp) / 5.0
    return {
        "surface_pressure_rel_l2": sp,
        "surface_pressure_rel_l2_pct": sp * 100.0,
        "wall_shear_rel_l2": ws_vec,
        "wall_shear_rel_l2_pct": ws_vec * 100.0,
        "wall_shear_x_rel_l2": tx,
        "wall_shear_x_rel_l2_pct": tx * 100.0,
        "wall_shear_y_rel_l2": ty,
        "wall_shear_y_rel_l2_pct": ty * 100.0,
        "wall_shear_z_rel_l2": tz,
        "wall_shear_z_rel_l2_pct": tz * 100.0,
        "volume_pressure_rel_l2": vp,
        "volume_pressure_rel_l2_pct": vp * 100.0,
        "abupt_axis_mean_rel_l2": abupt,
        "abupt_axis_mean_rel_l2_pct": abupt * 100.0,
        "cases": float(len(surf_ids)),
    }


def compute_calibrated_metrics(
    cal: CalibrationStats,
    alpha_surf: torch.Tensor,
    beta_surf: torch.Tensor,
    alpha_vol: torch.Tensor,
    beta_vol: torch.Tensor,
) -> dict[str, float]:
    """Per-case averaged rel_l2 metrics on calibrated predictions.

    Mirrors the keys produced by ``finalize_eval_accumulator`` for the rel_l2
    family. MAE-style metrics are not analytically computable from these
    sufficient stats under an affine map and are omitted.
    """
    surf_ids = sorted(cal.surf_per_case.keys())
    if not surf_ids:
        raise RuntimeError("CalibrationStats is empty; cannot compute calibrated metrics")

    surf_per_channel_rel_l2: list[torch.Tensor] = []
    vec_rel_l2_list: list[torch.Tensor] = []
    for case_id in surf_ids:
        s = cal.surf_per_case[case_id]  # (4, 6)
        e_sq = _affine_error_sq(s, alpha_surf, beta_surf)  # (4,)
        t_sq = s[..., 4]  # sum of squared targets per channel
        rel = torch.sqrt(e_sq.clamp(min=0.0) / t_sq.clamp(min=1e-12))
        surf_per_channel_rel_l2.append(rel)

        s_vec = s[1:4]  # (3, 6): tau_x, tau_y, tau_z
        e_sq_vec = _affine_error_sq(s_vec, alpha_surf[1:4], beta_surf[1:4])  # (3,)
        t_sq_vec = s_vec[..., 4]
        rel_vec = torch.sqrt(e_sq_vec.sum().clamp(min=0.0) / t_sq_vec.sum().clamp(min=1e-12))
        vec_rel_l2_list.append(rel_vec)
    surf_rel_l2 = torch.stack(surf_per_channel_rel_l2, dim=0)  # (n_cases, 4)
    vec_rel_l2 = torch.stack(vec_rel_l2_list, dim=0)  # (n_cases,)

    vol_ids = sorted(cal.vol_per_case.keys())
    vol_per_channel_rel_l2: list[torch.Tensor] = []
    for case_id in vol_ids:
        v = cal.vol_per_case[case_id]  # (1, 6)
        e_sq = _affine_error_sq(v, alpha_vol, beta_vol)  # (1,)
        t_sq = v[..., 4]
        rel = torch.sqrt(e_sq.clamp(min=0.0) / t_sq.clamp(min=1e-12))
        vol_per_channel_rel_l2.append(rel)
    vol_rel_l2 = torch.stack(vol_per_channel_rel_l2, dim=0)  # (n_cases, 1)

    sp = float(surf_rel_l2[:, 0].mean().item())
    tx = float(surf_rel_l2[:, 1].mean().item())
    ty = float(surf_rel_l2[:, 2].mean().item())
    tz = float(surf_rel_l2[:, 3].mean().item())
    ws_vec = float(vec_rel_l2.mean().item())
    vp = float(vol_rel_l2[:, 0].mean().item())
    abupt = (sp + tx + ty + tz + vp) / 5.0
    return {
        "surface_pressure_rel_l2": sp,
        "surface_pressure_rel_l2_pct": sp * 100.0,
        "wall_shear_rel_l2": ws_vec,
        "wall_shear_rel_l2_pct": ws_vec * 100.0,
        "wall_shear_x_rel_l2": tx,
        "wall_shear_x_rel_l2_pct": tx * 100.0,
        "wall_shear_y_rel_l2": ty,
        "wall_shear_y_rel_l2_pct": ty * 100.0,
        "wall_shear_z_rel_l2": tz,
        "wall_shear_z_rel_l2_pct": tz * 100.0,
        "volume_pressure_rel_l2": vp,
        "volume_pressure_rel_l2_pct": vp * 100.0,
        "abupt_axis_mean_rel_l2": abupt,
        "abupt_axis_mean_rel_l2_pct": abupt * 100.0,
        "cases": float(len(surf_ids)),
    }


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
    sigma: float,
    seed_base: int,
    distributed_state,
    antithetic: bool = False,
    collect_calibration: bool = False,
    zone_thresholds: dict[str, float] | None = None,
    zone_counts: list[int] | None = None,
) -> tuple[dict[str, float], CalibrationStats | None]:
    rank = distributed_state.rank if distributed_state and distributed_state.enabled else 0
    world_size = (
        distributed_state.world_size
        if distributed_state and distributed_state.enabled
        else 1
    )
    my_cases = case_ids[rank::world_size]

    acc = EvalAccumulator()
    cal = CalibrationStats() if collect_calibration else None
    total_t0 = time.time()
    total_forward = 0.0
    total_io = 0.0
    total_perturb = 0.0
    total_forwards = 0
    for case_idx, case_id in enumerate(my_cases):
        t0 = time.time()
        surf_avg, vol_avg, surf_y, vol_y, surf_xyz, n_passes, timing = process_case_stacked(
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
            sigma=sigma,
            seed_base=seed_base,
            antithetic=antithetic,
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
        if cal is not None:
            surface_zones_by_n: dict[int, torch.Tensor] | None = None
            if zone_thresholds is not None and zone_counts:
                surface_zones_by_n = {
                    nz: assign_zones(surf_xyz, zone_thresholds, nz)
                    for nz in zone_counts
                }
            add_case_to_calibration(
                cal,
                case_id=case_id,
                surface_pred_real=surf_avg,
                volume_pred_real=vol_avg,
                surface_y=surf_y,
                volume_y=vol_y,
                surface_zones_by_n=surface_zones_by_n,
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
        gathered_cal: list[CalibrationStats | None] | None = None
        if cal is not None:
            gathered_cal = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_cal, cal)
        if not distributed_state.is_main:
            return {}, None
        merged = merge_eval_accumulators(g for g in gathered if g is not None)
        merged_cal = (
            merge_calibration_stats(c for c in gathered_cal if c is not None)
            if gathered_cal is not None
            else None
        )
    else:
        merged = acc
        merged_cal = cal

    return finalize_eval_accumulator(merged), merged_cal


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


def main(argv: Iterable[str] | None = None) -> None:
    # NCCL TCPStore default is 600s; full-stack 60-pass eval can have val_surface
    # straggler waits >600s before the next collective, triggering a wait-timeout
    # on the lazy NCCL communicator init for the first post-val all_gather_object.
    # Bump to 120 min (same shape as thorfinn #1433 / nezuko #1432 plumbing).
    state = init_distributed(timeout=timedelta(minutes=120))
    cfg = parse_args(argv)
    device = state.device

    resolutions = parse_resolutions(cfg.resolutions)
    modes = parse_modes(cfg.eval_modes)
    zone_counts = parse_zone_counts(cfg.regional_num_zones)
    ridge_lambdas = parse_ridge_lambdas(cfg.regional_ridge_lambda_sweep)
    # PR-convention alias used by H275 instructions; semantically identical.
    mode_aliases = {"mirror_res_weight_noise_avg": "weight_noise_mirror_res_avg"}
    modes = [mode_aliases.get(m, m) for m in modes]
    valid_modes = ("weight_noise_only", "weight_noise_mirror_res_avg")
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Unknown eval mode: {mode!r} (valid: {valid_modes})")

    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        print(f"Resolutions: {resolutions}")
        print(f"Modes: {modes}")
        k_eff_print = cfg.weight_noise_passes * (2 if cfg.antithetic_noise else 1)
        print(
            f"Weight-noise TTA: sigma_rel={cfg.weight_noise_sigma} "
            f"K_passes={cfg.weight_noise_passes} "
            f"antithetic={cfg.antithetic_noise} K_eff={k_eff_print} "
            f"seed_base={cfg.weight_noise_seed_base}"
        )
        print(
            f"Calibration: H300 global={cfg.test_time_calibration} "
            f"| H313 regional zone_counts={zone_counts} "
            f"| H328 ridge_lambdas={ridge_lambdas}"
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

    # H313: Pre-compute zone thresholds from val surface xyz (rank 0), then
    # broadcast to every rank so each case's zone assignment is identical.
    # Thresholds are computed once and reused for every n_zones in zone_counts
    # (the partition rules pick the same percentile breakpoints).
    zone_thresholds: dict[str, float] | None = None
    if zone_counts and cfg.test_time_calibration:
        if state.is_main:
            t_z = time.time()
            zone_thresholds = compute_zone_thresholds(store, val_case_ids)
            print(
                f"\n=== H313 zone thresholds (zone_counts={zone_counts}, "
                f"pool=val_surface_xyz) ===",
                flush=True,
            )
            for k, v in zone_thresholds.items():
                print(f"  {k:>10s} = {v:+.6f}", flush=True)
            print(f"  threshold compute: {time.time() - t_z:.1f}s", flush=True)
        if state.enabled:
            obj: list[dict[str, float] | None] = [zone_thresholds]
            dist.broadcast_object_list(obj, src=0)
            zone_thresholds = obj[0]
    elif zone_counts and not cfg.test_time_calibration and state.is_main:
        print(
            "[H313 WARN] regional_num_zones set but --test-time-calibration "
            "not enabled; ignoring zone settings",
            flush=True,
        )

    run = None
    if state.is_main:
        run_name = (
            cfg.wandb_name
            or f"{cfg.agent}/h252-stacked-sigma-{cfg.weight_noise_sigma}-K{cfg.weight_noise_passes}"
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
                "h252",
                "tta",
                "stacked",
                "weight-noise",
                "multi-res",
                "mirror",
                "eval-only",
                cfg.agent,
                *(
                    ["h313", *(f"regional-{n}-zone" for n in zone_counts)]
                    if zone_counts
                    else []
                ),
            ],
            reinit="finish_previous",
        )

    splits = [
        ("val_surface", val_case_ids, "full_val"),
        ("test_surface", test_case_ids, "test"),
    ]

    summary: dict[str, dict[str, dict[str, float]]] = {}
    cal_summary: dict[str, dict[str, CalibrationStats | None]] = {}
    log_prefix_by_split: dict[str, str] = {}
    for split_name, case_ids, log_prefix in splits:
        summary[split_name] = {}
        cal_summary[split_name] = {}
        log_prefix_by_split[split_name] = log_prefix
        for mode in modes:
            if mode == "weight_noise_only":
                # K-pass weight-noise average at a single resolution, no mirror.
                # Used to reproduce the H242 noise_only result.
                mode_resolutions = [cfg.eval_surface_points]  # single resolution
                use_mirror = False
            elif mode == "weight_noise_mirror_res_avg":
                mode_resolutions = list(resolutions)
                use_mirror = True
            else:
                raise ValueError(f"Unknown mode {mode!r}")

            if state.is_main:
                k_eff_print = cfg.weight_noise_passes * (
                    2 if cfg.antithetic_noise else 1
                )
                print(
                    f"\n=== {split_name} mode={mode} resolutions={mode_resolutions} "
                    f"mirror={use_mirror} K={cfg.weight_noise_passes} "
                    f"antithetic={cfg.antithetic_noise} K_eff={k_eff_print} "
                    f"sigma={cfg.weight_noise_sigma} "
                    f"calibration={cfg.test_time_calibration} ===",
                    flush=True,
                )
            t0 = time.time()
            metrics, cal_stats = evaluate_split_stacked(
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
                sigma=cfg.weight_noise_sigma,
                seed_base=cfg.weight_noise_seed_base,
                distributed_state=state,
                antithetic=cfg.antithetic_noise,
                collect_calibration=cfg.test_time_calibration,
                zone_thresholds=zone_thresholds,
                zone_counts=zone_counts if zone_thresholds is not None else None,
            )
            dt = time.time() - t0
            if state.is_main:
                print(f"  total {split_name}/{mode}: {dt:.1f}s")
                print_metrics(f"{split_name}/{mode}", metrics)
                summary[split_name][mode] = metrics
                cal_summary[split_name][mode] = cal_stats

                log_obj: dict[str, float] = {}
                log_obj.update(primary_metric_log(f"{log_prefix}_primary/{mode}", metrics))
                log_obj.update({f"{log_prefix}_extra/{mode}/loss": metrics["loss"]})
                log_obj[f"{log_prefix}_extra/{mode}/seconds"] = dt
                if run is not None:
                    wandb.log(log_obj)

    # Always restore clean weights at the end.
    restore_clean_params(eval_module, clean)

    # H300: per-channel affine calibration applied on val, transferred to test.
    cal_metrics_by_split_mode: dict[str, dict[str, dict[str, float]]] = {}
    if cfg.test_time_calibration and state.is_main:
        for mode in modes:
            val_cal = cal_summary.get("val_surface", {}).get(mode)
            test_cal = cal_summary.get("test_surface", {}).get(mode)
            if val_cal is None or test_cal is None:
                print(
                    f"[calibration] mode={mode}: missing cal stats "
                    f"(val={val_cal is not None}, test={test_cal is not None}); skipping",
                    flush=True,
                )
                continue
            alpha_surf, beta_surf = fit_affine_per_channel(val_cal.surf_global)
            alpha_vol, beta_vol = fit_affine_per_channel(val_cal.vol_global)

            channel_names = ["cp", "tau_x", "tau_y", "tau_z"]
            print(f"\n=== H300 calibration (fit on val, mode={mode}) ===", flush=True)
            for c, name in enumerate(channel_names):
                print(
                    f"  surface[{name:6s}]  alpha={alpha_surf[c].item():+.6f} "
                    f"beta={beta_surf[c].item():+.6f}",
                    flush=True,
                )
            print(
                f"  volume[volume_pressure]  alpha={alpha_vol[0].item():+.6f} "
                f"beta={beta_vol[0].item():+.6f}",
                flush=True,
            )

            val_cal_metrics = compute_calibrated_metrics(
                val_cal, alpha_surf, beta_surf, alpha_vol, beta_vol
            )
            test_cal_metrics = compute_calibrated_metrics(
                test_cal, alpha_surf, beta_surf, alpha_vol, beta_vol
            )
            cal_metrics_by_split_mode.setdefault("val_surface", {})[mode] = val_cal_metrics
            cal_metrics_by_split_mode.setdefault("test_surface", {})[mode] = test_cal_metrics

            print(
                f"  val  (raw -> cal)  abupt {summary['val_surface'][mode]['abupt_axis_mean_rel_l2_pct']:.4f} "
                f"-> {val_cal_metrics['abupt_axis_mean_rel_l2_pct']:.4f}",
                flush=True,
            )
            print(
                f"  test (raw -> cal)  abupt {summary['test_surface'][mode]['abupt_axis_mean_rel_l2_pct']:.4f} "
                f"-> {test_cal_metrics['abupt_axis_mean_rel_l2_pct']:.4f}",
                flush=True,
            )

            if run is not None:
                # Log alpha/beta to W&B summary so they appear on the run page.
                for c, name in enumerate(channel_names):
                    run.summary[f"calibration/{mode}/alpha_{name}"] = float(alpha_surf[c].item())
                    run.summary[f"calibration/{mode}/beta_{name}"] = float(beta_surf[c].item())
                run.summary[f"calibration/{mode}/alpha_volume_pressure"] = float(alpha_vol[0].item())
                run.summary[f"calibration/{mode}/beta_volume_pressure"] = float(beta_vol[0].item())

                # Log calibrated primary metrics to W&B history (and as
                # split-level summary keys for downstream parsing).
                val_log = {
                    f"full_val_primary_calibrated/{mode}/{k}": v
                    for k, v in val_cal_metrics.items()
                }
                test_log = {
                    f"test_primary_calibrated/{mode}/{k}": v
                    for k, v in test_cal_metrics.items()
                }
                wandb.log({**val_log, **test_log})

                # Single-line paper-facing primary metrics (no mode suffix).
                run.summary["val_abupt_h300_calibrated"] = float(
                    val_cal_metrics["abupt_axis_mean_rel_l2_pct"]
                )
                run.summary["test_abupt_h300_calibrated"] = float(
                    test_cal_metrics["abupt_axis_mean_rel_l2_pct"]
                )
                run.summary["val_abupt_h300_raw"] = float(
                    summary["val_surface"][mode]["abupt_axis_mean_rel_l2_pct"]
                )
                run.summary["test_abupt_h300_raw"] = float(
                    summary["test_surface"][mode]["abupt_axis_mean_rel_l2_pct"]
                )

    # H313/H328: regional/zonal surface calibration on top of H300, optionally
    # ridge-regularised toward the H300 per-channel global fit. Surface channels
    # (cp, tau_x, tau_y, tau_z) get one (alpha, beta) per (zone, channel) at
    # each ridge lambda; the volume pressure channel keeps the H300 global
    # affine (no volume zoning, per researcher recommendation given tight
    # test_VP headroom). The shape is [split][mode][n_zones][lam] so multiple
    # arms (e.g., 4 and 6 zones × 7 lambdas) coexist.
    regional_metrics_by_split_mode_zones_lambda: dict[
        str, dict[str, dict[int, dict[float, dict[str, float]]]]
    ] = {}
    # (mode, n_zones) -> best lambda by val_abupt (used for paper-facing summary)
    best_lambda_by_mode_zones: dict[tuple[str, int], float] = {}
    if (
        cfg.test_time_calibration
        and zone_counts
        and zone_thresholds is not None
        and state.is_main
    ):
        zone_name_table = {
            6: ["nose", "roof", "sides", "underbody", "rear", "body"],
            4: ["nose", "rear", "underbody", "body"],
        }
        for mode in modes:
            val_cal = cal_summary.get("val_surface", {}).get(mode)
            test_cal = cal_summary.get("test_surface", {}).get(mode)
            if val_cal is None or test_cal is None:
                continue
            if not val_cal.surf_zonal_global or not val_cal.surf_zonal_per_case:
                print(
                    f"[H313] mode={mode}: missing zonal stats; skipping regional",
                    flush=True,
                )
                continue

            alpha_vol, beta_vol = fit_affine_per_channel(val_cal.vol_global)
            # H300 per-channel global (alpha_global_c, beta_global_c) is the
            # ridge shrinkage target. Fit ONCE on the val surface global stats.
            alpha_global_surf, beta_global_surf = fit_affine_per_channel(
                val_cal.surf_global
            )

            for n_zones in zone_counts:
                if n_zones not in val_cal.surf_zonal_global:
                    print(
                        f"[H313] mode={mode}: no zonal stats for n_zones={n_zones}; "
                        "skipping this arm",
                        flush=True,
                    )
                    continue
                zone_names = zone_name_table[n_zones]
                zonal_global = val_cal.surf_zonal_global[n_zones]
                channel_names = ["cp", "tau_x", "tau_y", "tau_z"]
                N_per_zone = zonal_global[..., 0, 0]  # (n_zones,) same N across channels

                print(
                    f"\n=== H328 ridge-regional calibration (n_zones={n_zones}, "
                    f"fit on val, mode={mode}) ===",
                    flush=True,
                )
                print(
                    "  zone counts (val surface points pooled):  "
                    + "  ".join(
                        f"{zone_names[z]}={int(N_per_zone[z].item()):,}"
                        for z in range(n_zones)
                    ),
                    flush=True,
                )
                print(
                    "  H300 per-channel global shrinkage target (from val_cal.surf_global):",
                    flush=True,
                )
                for c, cname in enumerate(channel_names):
                    print(
                        f"    [{cname:6s}]  α_global={alpha_global_surf[c].item():+.6f}  "
                        f"β_global={beta_global_surf[c].item():+.6f}",
                        flush=True,
                    )
                print(
                    f"  volume[volume_pressure]  alpha={alpha_vol[0].item():+.6f} "
                    f"beta={beta_vol[0].item():+.6f}  (H300 global, no zoning)",
                    flush=True,
                )

                # Sweep over ridge lambdas. Each fit is closed-form (seconds
                # total across the whole sweep) so we report all of them and
                # pick the best by val_abupt.
                lam_metrics: dict[float, tuple[
                    dict[str, float], dict[str, float], torch.Tensor, torch.Tensor
                ]] = {}
                for lam in ridge_lambdas:
                    alpha_surf_zonal, beta_surf_zonal = (
                        fit_affine_per_zone_per_channel_ridge(
                            zonal_global,
                            alpha_global_surf,
                            beta_global_surf,
                            lam,
                        )
                    )
                    val_reg_metrics = compute_calibrated_metrics_regional(
                        val_cal,
                        n_zones,
                        alpha_surf_zonal,
                        beta_surf_zonal,
                        alpha_vol,
                        beta_vol,
                    )
                    test_reg_metrics = compute_calibrated_metrics_regional(
                        test_cal,
                        n_zones,
                        alpha_surf_zonal,
                        beta_surf_zonal,
                        alpha_vol,
                        beta_vol,
                    )
                    lam_metrics[lam] = (
                        val_reg_metrics,
                        test_reg_metrics,
                        alpha_surf_zonal,
                        beta_surf_zonal,
                    )
                    regional_metrics_by_split_mode_zones_lambda.setdefault(
                        "val_surface", {}
                    ).setdefault(mode, {}).setdefault(n_zones, {})[lam] = val_reg_metrics
                    regional_metrics_by_split_mode_zones_lambda.setdefault(
                        "test_surface", {}
                    ).setdefault(mode, {}).setdefault(n_zones, {})[lam] = test_reg_metrics

                # λ-sweep summary table.
                test_sp_h300 = (
                    cal_metrics_by_split_mode.get("test_surface", {})
                    .get(mode, {})
                    .get("surface_pressure_rel_l2_pct")
                )
                h300_val_abupt = (
                    cal_metrics_by_split_mode.get("val_surface", {})
                    .get(mode, {})
                    .get("abupt_axis_mean_rel_l2_pct")
                )
                h300_test_abupt = (
                    cal_metrics_by_split_mode.get("test_surface", {})
                    .get(mode, {})
                    .get("abupt_axis_mean_rel_l2_pct")
                )
                print(
                    f"\n  --- λ-sweep (n_zones={n_zones}, mode={mode}) ---",
                    flush=True,
                )
                print(
                    f"  {'lambda':>10s}  {'val_abupt':>10s}  {'test_abupt':>11s}  "
                    f"{'val-test':>10s}  {'val_SP':>8s}  {'test_SP':>8s}  "
                    f"{'val_WSS':>8s}  {'test_WSS':>8s}  {'val_VP':>8s}  {'test_VP':>8s}",
                    flush=True,
                )
                for lam in ridge_lambdas:
                    v_m, t_m, _a, _b = lam_metrics[lam]
                    gap = v_m["abupt_axis_mean_rel_l2_pct"] - t_m["abupt_axis_mean_rel_l2_pct"]
                    print(
                        f"  {('inf' if lam == float('inf') else f'{lam:.4g}'):>10s}  "
                        f"{v_m['abupt_axis_mean_rel_l2_pct']:>10.4f}  "
                        f"{t_m['abupt_axis_mean_rel_l2_pct']:>11.4f}  "
                        f"{gap:>+10.4f}  "
                        f"{v_m['surface_pressure_rel_l2_pct']:>8.4f}  "
                        f"{t_m['surface_pressure_rel_l2_pct']:>8.4f}  "
                        f"{v_m['wall_shear_rel_l2_pct']:>8.4f}  "
                        f"{t_m['wall_shear_rel_l2_pct']:>8.4f}  "
                        f"{v_m['volume_pressure_rel_l2_pct']:>8.4f}  "
                        f"{t_m['volume_pressure_rel_l2_pct']:>8.4f}",
                        flush=True,
                    )

                # Pick best lambda by val_abupt (lower is better). Ties broken
                # toward larger lambda (more regularisation, simpler model).
                best_lam = min(
                    ridge_lambdas,
                    key=lambda L: (lam_metrics[L][0]["abupt_axis_mean_rel_l2_pct"], -L),
                )
                best_lambda_by_mode_zones[(mode, n_zones)] = best_lam
                best_v, best_t, best_a, best_b = lam_metrics[best_lam]
                print(
                    f"\n  Best λ by val_abupt: {('inf' if best_lam == float('inf') else f'{best_lam:.4g}')}  "
                    f"val_abupt={best_v['abupt_axis_mean_rel_l2_pct']:.4f}%  "
                    f"test_abupt={best_t['abupt_axis_mean_rel_l2_pct']:.4f}%",
                    flush=True,
                )
                # Per-zone α / β matrix at best λ.
                print(
                    f"  --- Per-zone (alpha, beta) at best λ = "
                    f"{('inf' if best_lam == float('inf') else f'{best_lam:.4g}')} ---",
                    flush=True,
                )
                for z in range(n_zones):
                    for c, cname in enumerate(channel_names):
                        a = best_a[z, c].item()
                        b = best_b[z, c].item()
                        print(
                            f"    zone[{zone_names[z]:9s}][{cname:6s}]  "
                            f"alpha={a:+.6f}  beta={b:+.6f}",
                            flush=True,
                        )

                # Validation invariants from the PR: λ=0 ≈ H313, λ=∞ ≈ H300.
                if 0.0 in lam_metrics:
                    v_m0, t_m0, _, _ = lam_metrics[0.0]
                    print(
                        f"  [invariant λ=0]   val_abupt={v_m0['abupt_axis_mean_rel_l2_pct']:.4f}  "
                        f"test_abupt={t_m0['abupt_axis_mean_rel_l2_pct']:.4f}  "
                        f"(should match H313 unregularised; H313 baseline was 5.8938/5.7410 at n_zones=6)",
                        flush=True,
                    )
                if float("inf") in lam_metrics:
                    v_mi, t_mi, _, _ = lam_metrics[float("inf")]
                    h300_v_str = (
                        f"{h300_val_abupt:.4f}" if h300_val_abupt is not None else "n/a"
                    )
                    h300_t_str = (
                        f"{h300_test_abupt:.4f}" if h300_test_abupt is not None else "n/a"
                    )
                    print(
                        f"  [invariant λ=∞]   val_abupt={v_mi['abupt_axis_mean_rel_l2_pct']:.4f}  "
                        f"test_abupt={t_mi['abupt_axis_mean_rel_l2_pct']:.4f}  "
                        f"(should match H300 global: val={h300_v_str}, test={h300_t_str})",
                        flush=True,
                    )

                # SP watch at best λ — H300 already exceeded the 3.577% hard floor.
                best_test_sp = best_t["surface_pressure_rel_l2_pct"]
                if test_sp_h300 is not None:
                    delta_sp = best_test_sp - test_sp_h300
                    marker = "WORSENED" if delta_sp > 0.0 else "improved"
                    print(
                        f"  [H328 SP watch n={n_zones}z best λ]  test_SP H300={test_sp_h300:.4f}% -> "
                        f"H328={best_test_sp:.4f}%  ({marker} {delta_sp:+.4f}pp; "
                        f"constraint 3.577%)",
                        flush=True,
                    )

                # W&B logging per-lambda + best-lambda summary.
                if run is not None:
                    for lam in ridge_lambdas:
                        v_m, t_m, a_zc, b_zc = lam_metrics[lam]
                        lam_tag = _fmt_lambda(lam)
                        for z in range(n_zones):
                            for c, cname in enumerate(channel_names):
                                run.summary[
                                    f"h328_{n_zones}z_lam{lam_tag}/{mode}/"
                                    f"zone_{zone_names[z]}/alpha_{cname}"
                                ] = float(a_zc[z, c].item())
                                run.summary[
                                    f"h328_{n_zones}z_lam{lam_tag}/{mode}/"
                                    f"zone_{zone_names[z]}/beta_{cname}"
                                ] = float(b_zc[z, c].item())
                            run.summary[
                                f"h328_{n_zones}z_lam{lam_tag}/{mode}/"
                                f"zone_{zone_names[z]}/n_val_points"
                            ] = float(N_per_zone[z].item())

                        val_log = {
                            f"full_val_primary_regional_{n_zones}z_lam{lam_tag}/{mode}/{k}": v
                            for k, v in v_m.items()
                        }
                        test_log = {
                            f"test_primary_regional_{n_zones}z_lam{lam_tag}/{mode}/{k}": v
                            for k, v in t_m.items()
                        }
                        wandb.log({**val_log, **test_log})

                        run.summary[
                            f"val_abupt_h328_{n_zones}z_lam{lam_tag}"
                        ] = float(v_m["abupt_axis_mean_rel_l2_pct"])
                        run.summary[
                            f"test_abupt_h328_{n_zones}z_lam{lam_tag}"
                        ] = float(t_m["abupt_axis_mean_rel_l2_pct"])
                        run.summary[
                            f"test_SP_h328_{n_zones}z_lam{lam_tag}"
                        ] = float(t_m["surface_pressure_rel_l2_pct"])
                        run.summary[
                            f"test_WSS_h328_{n_zones}z_lam{lam_tag}"
                        ] = float(t_m["wall_shear_rel_l2_pct"])

                    # Best-λ paper-facing summary keys.
                    best_lam_tag = _fmt_lambda(best_lam)
                    run.summary[f"best_lambda_h328_{n_zones}z/{mode}/value"] = (
                        float("inf") if best_lam == float("inf") else float(best_lam)
                    )
                    run.summary[f"best_lambda_h328_{n_zones}z/{mode}/tag"] = best_lam_tag
                    run.summary[f"val_abupt_h328_best_{n_zones}z"] = float(
                        best_v["abupt_axis_mean_rel_l2_pct"]
                    )
                    run.summary[f"test_abupt_h328_best_{n_zones}z"] = float(
                        best_t["abupt_axis_mean_rel_l2_pct"]
                    )
                    run.summary[f"test_SP_h328_best_{n_zones}z"] = float(
                        best_t["surface_pressure_rel_l2_pct"]
                    )
                    run.summary[f"test_WSS_h328_best_{n_zones}z"] = float(
                        best_t["wall_shear_rel_l2_pct"]
                    )

            if run is not None:
                for k, v in zone_thresholds.items():
                    run.summary[f"h313/zone_threshold/{k}"] = float(v)

        # Single-line paper-facing summary across all arms — choose the (mode,
        # n_zones, λ) triple with the best val_abupt. SENPAI-RESULT references
        # these keys.
        if run is not None and regional_metrics_by_split_mode_zones_lambda:
            best_overall: tuple[str, int, float, float] | None = None  # (mode, n_zones, lam, val_abupt)
            for (mode, n_zones), lam in best_lambda_by_mode_zones.items():
                v_abupt = regional_metrics_by_split_mode_zones_lambda[
                    "val_surface"
                ][mode][n_zones][lam]["abupt_axis_mean_rel_l2_pct"]
                if best_overall is None or v_abupt < best_overall[3]:
                    best_overall = (mode, n_zones, lam, v_abupt)
            if best_overall is not None:
                bmode, bz, blam, _ = best_overall
                bv = regional_metrics_by_split_mode_zones_lambda["val_surface"][bmode][bz][blam]
                bt = regional_metrics_by_split_mode_zones_lambda["test_surface"][bmode][bz][blam]
                run.summary["best_arm_h328/mode"] = bmode
                run.summary["best_arm_h328/n_zones"] = float(bz)
                run.summary["best_arm_h328/lambda"] = (
                    float("inf") if blam == float("inf") else float(blam)
                )
                run.summary["best_arm_h328/lambda_tag"] = _fmt_lambda(blam)
                run.summary["val_abupt_h328_best_lambda"] = float(
                    bv["abupt_axis_mean_rel_l2_pct"]
                )
                run.summary["test_abupt_h328_best_lambda"] = float(
                    bt["abupt_axis_mean_rel_l2_pct"]
                )
                run.summary["test_SP_h328_best_lambda"] = float(
                    bt["surface_pressure_rel_l2_pct"]
                )
                run.summary["test_WSS_h328_best_lambda"] = float(
                    bt["wall_shear_rel_l2_pct"]
                )
                run.summary["test_VP_h328_best_lambda"] = float(
                    bt["volume_pressure_rel_l2_pct"]
                )

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
            header = f"  {'metric':<36s} " + " ".join(f"{m:>22s}" for m in mode_keys)
            print(header)
            for k in keys_to_show:
                row = f"  {k:<36s} " + " ".join(
                    f"{modes_dict[m][k]:>22.4f}" for m in mode_keys
                )
                print(row)
            cal_modes = cal_metrics_by_split_mode.get(split, {})
            if cal_modes:
                print(f"  --- H300 global calibrated (alpha,beta fit on val) ---")
                for k in keys_to_show:
                    row = f"  {k:<36s} " + " ".join(
                        f"{cal_modes[m][k]:>22.4f}" if m in cal_modes else f"{'n/a':>22s}"
                        for m in mode_keys
                    )
                    print(row)
            reg_modes_zones_lambda = (
                regional_metrics_by_split_mode_zones_lambda.get(split, {})
            )
            if reg_modes_zones_lambda:
                for n_zones in zone_counts:
                    has_any = any(
                        n_zones in reg_modes_zones_lambda.get(m, {})
                        for m in mode_keys
                    )
                    if not has_any:
                        continue
                    for lam in ridge_lambdas:
                        has_lam = any(
                            lam in reg_modes_zones_lambda.get(m, {}).get(n_zones, {})
                            for m in mode_keys
                        )
                        if not has_lam:
                            continue
                        lam_str = (
                            "inf" if lam == float("inf") else f"{lam:g}"
                        )
                        print(
                            f"  --- H328 ridge-regional ({n_zones}-zone, "
                            f"λ={lam_str}, global VP) ---"
                        )
                        for k in keys_to_show:
                            row = f"  {k:<36s} " + " ".join(
                                f"{reg_modes_zones_lambda[m][n_zones][lam][k]:>22.4f}"
                                if m in reg_modes_zones_lambda
                                and n_zones in reg_modes_zones_lambda[m]
                                and lam in reg_modes_zones_lambda[m][n_zones]
                                else f"{'n/a':>22s}"
                                for m in mode_keys
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
            for split, modes_dict in cal_metrics_by_split_mode.items():
                for mode, metrics in modes_dict.items():
                    for k, v in metrics.items():
                        try:
                            run.summary[f"{split}_calibrated/{mode}/{k}"] = float(v)
                        except Exception:
                            pass
            for split, modes_dict in regional_metrics_by_split_mode_zones_lambda.items():
                for mode, zones_dict in modes_dict.items():
                    for n_zones, lam_dict in zones_dict.items():
                        for lam, metrics in lam_dict.items():
                            lam_tag = _fmt_lambda(lam)
                            for k, v in metrics.items():
                                try:
                                    run.summary[
                                        f"{split}_regional_{n_zones}z_lam{lam_tag}/{mode}/{k}"
                                    ] = float(v)
                                except Exception:
                                    pass
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
