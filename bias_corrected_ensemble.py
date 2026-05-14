# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Bias-corrected ensemble search (PR #1108).

Builds on the SLSQP weight search from PR #1103 by extending the predictor with
a per-channel additive bias `b_c`:

    pred_ensemble = sum_i w_i * pred_i + b_c   (b broadcast over points in a case)

The bias breaks out of the convex-combination constraint and can correct the
systematic per-channel offset of the 4-member pool. Per-channel error
cross-correlations and sum-of-errors are precomputed once per split, so each
SLSQP evaluation is O(1) and the entire LOOCV sweep takes seconds.

Math notes:
    err = (sum_i w_i pred_i + b) - target = (sum_i w_i err_i) + b   when sum_i w_i = 1
    SSE = sum_n err^2
        = sum_n (sum_i w_i err_i)^2 + 2 b sum_n (sum_i w_i err_i) + N * b^2
        = sum_{ij} w_i w_j cross_ij + 2 b sum_i w_i sum_err_i + N * b^2

so a single channel's per-case SSE is quadratic in (w, b) given the precomputed
[cross_ij, sum_err_i, N] triple.

PR text formulates the regularised objective as
``val_WSS(...) + λ * sum(b_c**2)`` but explicitly expects a non-zero bias on
SP and vol_p ("A negative bias on SP can push val_SP below 3.72%"). val_WSS
alone is insensitive to b_SP and b_vol_p, so with L2 regularisation they would
collapse to zero. To honour the spirit of the PR we use val_abupt (the mean
of the 5 per-axis rel-L2 metrics) as the primary multi-target objective: this
makes every bias contribute, and it remains separable per-channel because each
val_axis_rel_l2 only sees its own bias. We also report the val_WSS-only
objective for completeness (where b_SP / b_vol_p are pinned to zero).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.optimize as sopt
import torch
import wandb

from data import load_data
from trainer_runtime import TargetTransform

from ensemble_eval import (
    load_meta_cache_from_disk,
    load_pred_cache_from_disk,
    meta_cache_path,
    pred_cache_path,
)


# Per-axis channels (each treated as a scalar with its own bias).
# Index in the precomputed tensors matches:
PER_AXIS_CHANNELS = [
    "surface_pressure",   # surface_y[:, 0:1] — bias b_SP
    "wall_shear_x",       # surface_y[:, 1:2] — bias b_tau_x
    "wall_shear_y",       # surface_y[:, 2:3] — bias b_tau_y
    "wall_shear_z",       # surface_y[:, 3:4] — bias b_tau_z
    "volume_pressure",    # volume_y[:, 0:1]  — bias b_vol_p
]
SURFACE_SLICES = {
    "surface_pressure": (0, 1),
    "wall_shear_x": (1, 2),
    "wall_shear_y": (2, 3),
    "wall_shear_z": (3, 4),
}
WSS_AXES = ("wall_shear_x", "wall_shear_y", "wall_shear_z")


@dataclass
class PrecomputedStats:
    """Per-case, per-channel statistics needed for the bias-corrected SSE.

    All tensors are float64 CPU and live in DENORMALIZED (physical) units.
    Bias terms are also in denormalized units.

    For each channel c (one of PER_AXIS_CHANNELS), per case k:
        cross[c][k, i, j] = sum_n err_i_n_c_k * err_j_n_c_k         [M, M]
        sum_err[c][k, i]  = sum_n err_i_n_c_k                         [M]
        N[c][k]           = number of valid points in case k for channel c
        ss_target[c][k]   = sum_n target_n_c_k ** 2
        valid[c][k]       = True if ss_target[c][k] > 0
    """

    cross: dict[str, torch.Tensor]
    sum_err: dict[str, torch.Tensor]
    N: dict[str, torch.Tensor]
    ss_target: dict[str, torch.Tensor]
    valid: dict[str, torch.Tensor]
    case_ids: list[str]
    M: int

    def num_cases(self) -> int:
        return len(self.case_ids)


def precompute_stats_for_split(
    run_ids: list[str],
    batch_meta: list[dict],
    pred_cache: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    transform: TargetTransform,
    device: torch.device,
) -> PrecomputedStats:
    """One pass over the cached predictions to fill per-case sums."""

    M = len(run_ids)
    cross_by_case: dict[str, dict[str, torch.Tensor]] = {ch: {} for ch in PER_AXIS_CHANNELS}
    sum_err_by_case: dict[str, dict[str, torch.Tensor]] = {ch: {} for ch in PER_AXIS_CHANNELS}
    N_by_case: dict[str, dict[str, int]] = {ch: {} for ch in PER_AXIS_CHANNELS}
    ss_target_by_case: dict[str, dict[str, float]] = {ch: {} for ch in PER_AXIS_CHANNELS}

    for batch_idx, meta in enumerate(batch_meta):
        surface_y = meta["surface_y"].to(device=device, dtype=torch.float32)
        volume_y = meta["volume_y"].to(device=device, dtype=torch.float32)
        surface_mask = meta["surface_mask"].to(device=device, dtype=torch.bool)
        volume_mask = meta["volume_mask"].to(device=device, dtype=torch.bool)
        case_ids = list(meta["case_ids"])

        # Stack denorm errors per member: [M, B, N, 4] surface, [M, B, N, 1] volume.
        surface_errs: list[torch.Tensor] = []
        volume_errs: list[torch.Tensor] = []
        for run_id in run_ids:
            sp_norm, vp_norm = pred_cache[run_id][batch_idx]
            sp_norm = sp_norm.to(device=device, dtype=torch.float32, non_blocking=True)
            vp_norm = vp_norm.to(device=device, dtype=torch.float32, non_blocking=True)
            sp_denorm = transform.invert_surface(sp_norm)
            vp_denorm = transform.invert_volume(vp_norm)
            surface_errs.append(sp_denorm - surface_y)
            volume_errs.append(vp_denorm - volume_y)

        for case_idx, case_id in enumerate(case_ids):
            s_valid = surface_mask[case_idx]
            if bool(s_valid.any()):
                s_target_valid = surface_y[case_idx][s_valid]  # [N, 4]
                surface_errs_valid = [
                    se[case_idx][s_valid] for se in surface_errs
                ]  # list of [N, 4]
                for ch_name in ("surface_pressure", "wall_shear_x", "wall_shear_y", "wall_shear_z"):
                    lo, hi = SURFACE_SLICES[ch_name]
                    target_ch = s_target_valid[:, lo:hi]
                    ss_target_ch = float(target_ch.pow(2).sum().item())
                    N_ch = int(target_ch.numel())
                    if N_ch == 0:
                        continue
                    err_stack = torch.stack(
                        [se[:, lo:hi] for se in surface_errs_valid], dim=0
                    )  # [M, N, 1]
                    err_flat = err_stack.reshape(M, -1)  # [M, N]
                    cross = (err_flat @ err_flat.T).double().cpu()  # [M, M]
                    sum_err = err_flat.sum(dim=1).double().cpu()  # [M]
                    if case_id in cross_by_case[ch_name]:
                        cross_by_case[ch_name][case_id] += cross
                        sum_err_by_case[ch_name][case_id] += sum_err
                        N_by_case[ch_name][case_id] += N_ch
                        ss_target_by_case[ch_name][case_id] += ss_target_ch
                    else:
                        cross_by_case[ch_name][case_id] = cross
                        sum_err_by_case[ch_name][case_id] = sum_err
                        N_by_case[ch_name][case_id] = N_ch
                        ss_target_by_case[ch_name][case_id] = ss_target_ch
            v_valid = volume_mask[case_idx]
            if bool(v_valid.any()):
                v_target_valid = volume_y[case_idx][v_valid]
                ss_target_ch = float(v_target_valid.pow(2).sum().item())
                N_ch = int(v_target_valid.numel())
                if N_ch == 0:
                    continue
                volume_errs_valid = [
                    ve[case_idx][v_valid] for ve in volume_errs
                ]  # list of [N, 1]
                err_stack = torch.stack(volume_errs_valid, dim=0)  # [M, N, 1]
                err_flat = err_stack.reshape(M, -1)
                cross = (err_flat @ err_flat.T).double().cpu()
                sum_err = err_flat.sum(dim=1).double().cpu()
                ch_name = "volume_pressure"
                if case_id in cross_by_case[ch_name]:
                    cross_by_case[ch_name][case_id] += cross
                    sum_err_by_case[ch_name][case_id] += sum_err
                    N_by_case[ch_name][case_id] += N_ch
                    ss_target_by_case[ch_name][case_id] += ss_target_ch
                else:
                    cross_by_case[ch_name][case_id] = cross
                    sum_err_by_case[ch_name][case_id] = sum_err
                    N_by_case[ch_name][case_id] = N_ch
                    ss_target_by_case[ch_name][case_id] = ss_target_ch

    case_ids_sorted = sorted(
        {cid for ch in PER_AXIS_CHANNELS for cid in cross_by_case[ch].keys()}
    )
    cross: dict[str, torch.Tensor] = {}
    sum_err: dict[str, torch.Tensor] = {}
    N: dict[str, torch.Tensor] = {}
    ss_target: dict[str, torch.Tensor] = {}
    valid: dict[str, torch.Tensor] = {}
    zero_cross = torch.zeros((M, M), dtype=torch.float64)
    zero_sum_err = torch.zeros((M,), dtype=torch.float64)
    for ch_name in PER_AXIS_CHANNELS:
        cross_list = [cross_by_case[ch_name].get(cid, zero_cross) for cid in case_ids_sorted]
        sum_err_list = [sum_err_by_case[ch_name].get(cid, zero_sum_err) for cid in case_ids_sorted]
        N_list = [N_by_case[ch_name].get(cid, 0) for cid in case_ids_sorted]
        ss_list = [ss_target_by_case[ch_name].get(cid, 0.0) for cid in case_ids_sorted]
        cross[ch_name] = torch.stack(cross_list, dim=0)
        sum_err[ch_name] = torch.stack(sum_err_list, dim=0)
        N[ch_name] = torch.tensor(N_list, dtype=torch.float64)
        ss_target[ch_name] = torch.tensor(ss_list, dtype=torch.float64)
        valid[ch_name] = ss_target[ch_name] > 0
    return PrecomputedStats(
        cross=cross,
        sum_err=sum_err,
        N=N,
        ss_target=ss_target,
        valid=valid,
        case_ids=case_ids_sorted,
        M=M,
    )


def _channel_sse_per_case(
    weights: torch.Tensor,
    bias: float,
    cross: torch.Tensor,
    sum_err: torch.Tensor,
    N: torch.Tensor,
) -> torch.Tensor:
    """Compute per-case SSE for one channel under (weights, bias).

    cross:   [num_cases, M, M]
    sum_err: [num_cases, M]
    N:       [num_cases]
    """

    quad = torch.einsum("cij,i,j->c", cross, weights, weights)
    lin = 2.0 * bias * torch.einsum("ci,i->c", sum_err, weights)
    const = bias * bias * N
    return quad + lin + const


def compute_metrics(
    weights: np.ndarray | torch.Tensor,
    biases: dict[str, float],
    precomp: PrecomputedStats,
    *,
    case_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute per-axis rel-L2 metrics, WSS vector, and abupt mean.

    ``case_mask`` (bool tensor over precomp.case_ids) restricts evaluation to a
    subset of cases (used for LOOCV held-out fitting).
    """

    w = torch.as_tensor(weights, dtype=torch.float64)
    metrics: dict[str, float] = {}

    sse_per_axis: dict[str, torch.Tensor] = {}
    for ch in PER_AXIS_CHANNELS:
        b = float(biases.get(ch, 0.0))
        sse = _channel_sse_per_case(
            w, b, precomp.cross[ch], precomp.sum_err[ch], precomp.N[ch]
        )
        sse_per_axis[ch] = sse
        ss = precomp.ss_target[ch]
        valid = precomp.valid[ch]
        if case_mask is not None:
            valid = valid & case_mask
        # rel L2 per case, then mean over cases
        rel_sq = (sse[valid] / ss[valid]).clamp(min=0.0)
        rel_l2_per_case = rel_sq.sqrt() * 100.0
        metrics[f"{ch}_rel_l2_pct"] = float(rel_l2_per_case.mean().item())

    # WSS vector: combine the 3 wall-shear axes.
    sse_wss = sum(sse_per_axis[ax] for ax in WSS_AXES)
    ss_wss = sum(precomp.ss_target[ax] for ax in WSS_AXES)
    valid_wss = precomp.valid[WSS_AXES[0]]
    for ax in WSS_AXES[1:]:
        valid_wss = valid_wss & precomp.valid[ax]
    if case_mask is not None:
        valid_wss = valid_wss & case_mask
    rel_sq_wss = (sse_wss[valid_wss] / ss_wss[valid_wss]).clamp(min=0.0)
    rel_l2_per_case_wss = rel_sq_wss.sqrt() * 100.0
    metrics["wall_shear_rel_l2_pct"] = float(rel_l2_per_case_wss.mean().item())

    # AB-UPT mean: mean of the 5 per-axis rel-L2 columns.
    metrics["abupt_axis_mean_rel_l2_pct"] = float(
        np.mean(
            [
                metrics["surface_pressure_rel_l2_pct"],
                metrics["wall_shear_x_rel_l2_pct"],
                metrics["wall_shear_y_rel_l2_pct"],
                metrics["wall_shear_z_rel_l2_pct"],
                metrics["volume_pressure_rel_l2_pct"],
            ]
        )
    )
    return metrics


def biases_array_to_dict(b: np.ndarray | list[float]) -> dict[str, float]:
    return {ch: float(b[i]) for i, ch in enumerate(PER_AXIS_CHANNELS)}


def channel_std_array(stats: dict[str, torch.Tensor]) -> np.ndarray:
    """Return per-channel target std in PER_AXIS_CHANNELS order."""

    surface_y_std = stats["surface_y_std"].cpu().numpy()  # [4] = [cp, tx, ty, tz]
    volume_y_std = stats["volume_y_std"].cpu().numpy()    # [1]
    return np.array(
        [
            float(surface_y_std[0]),
            float(surface_y_std[1]),
            float(surface_y_std[2]),
            float(surface_y_std[3]),
            float(volume_y_std[0]),
        ],
        dtype=np.float64,
    )


def fit_biases_slsqp(
    precomp: PrecomputedStats,
    *,
    weights: np.ndarray,
    lambda_reg: float,
    sigma: np.ndarray,
    objective_key: str = "abupt_axis_mean_rel_l2_pct",
    case_mask: torch.Tensor | None = None,
    b_init: np.ndarray | None = None,
    bound_in_sigmas: float = 1.5,
) -> tuple[np.ndarray, float, sopt.OptimizeResult]:
    """Solve for the optimal bias vector under fixed weights.

    Optimisation variable is the normalized bias ``b_norm = b / sigma_c``; the
    physical bias returned is ``b_norm * sigma_c``. The L2 penalty is on
    ``b_norm`` so different channels with very different physical scales receive
    comparable shrinkage.

    Returns (b_phys_star, fun_at_b_star, full_result).
    """

    C = len(PER_AXIS_CHANNELS)
    if b_init is None:
        b_init = np.zeros(C, dtype=np.float64)

    def obj(b_norm: np.ndarray) -> float:
        b_phys = b_norm * sigma
        m = compute_metrics(weights, biases_array_to_dict(b_phys), precomp, case_mask=case_mask)
        return m[objective_key] + lambda_reg * float(np.sum(b_norm * b_norm))

    b_bounds = [(-bound_in_sigmas, bound_in_sigmas)] * C

    res = sopt.minimize(
        obj,
        b_init,
        method="SLSQP",
        bounds=b_bounds,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )
    b_norm_star = np.asarray(res.x, dtype=np.float64)
    b_phys_star = b_norm_star * sigma
    return b_phys_star, float(res.fun), res


def loocv_arm_a(
    precomp: PrecomputedStats,
    *,
    weights: np.ndarray,
    lambda_reg: float,
    sigma: np.ndarray,
    objective_key: str = "abupt_axis_mean_rel_l2_pct",
) -> dict[str, float]:
    """Leave-one-out cross-validation for Arm A.

    For each held-out case k_h:
        - Fit b* on the remaining 33 cases minimising ``objective_key + λ ||b_norm||^2``
        - Predict on k_h and accumulate its per-axis SSE / WSS rel-L2

    Returns LOOCV-mean metrics (averaged over the 34 held-out per-case scores).
    """

    num_cases = precomp.num_cases()
    held_metrics: list[dict[str, float]] = []
    for k_h in range(num_cases):
        mask = torch.ones(num_cases, dtype=torch.bool)
        mask[k_h] = False
        b_star, _, _ = fit_biases_slsqp(
            precomp,
            weights=weights,
            lambda_reg=lambda_reg,
            sigma=sigma,
            objective_key=objective_key,
            case_mask=mask,
        )
        held_mask = torch.zeros(num_cases, dtype=torch.bool)
        held_mask[k_h] = True
        held = compute_metrics(weights, biases_array_to_dict(b_star), precomp, case_mask=held_mask)
        held_metrics.append(held)
    keys = held_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in held_metrics])) for k in keys}


def fit_joint_w_b_slsqp(
    precomp: PrecomputedStats,
    *,
    lambda_reg: float,
    sigma: np.ndarray,
    objective_key: str = "abupt_axis_mean_rel_l2_pct",
    w_init: np.ndarray | None = None,
    b_init: np.ndarray | None = None,
    bound_in_sigmas: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, float, sopt.OptimizeResult]:
    """Joint optimisation over weights (4-simplex) and biases (5D normalized).

    The biases are optimised in normalized (σ-units) space, matching Arm A.
    """

    M = precomp.M
    C = len(PER_AXIS_CHANNELS)

    if w_init is None:
        w_init = np.array([0.375, 0.250, 0.250, 0.125], dtype=np.float64)
    if b_init is None:
        b_init = np.zeros(C, dtype=np.float64)

    x0 = np.concatenate([w_init, b_init])
    bounds = (
        [(0.0, 1.0)] * M
        + [(-bound_in_sigmas, bound_in_sigmas)] * C
    )
    constraints = [
        {"type": "eq", "fun": lambda x: float(np.sum(x[:M]) - 1.0)},
    ]

    def obj(x: np.ndarray) -> float:
        w = x[:M]
        b_norm = x[M:]
        b_phys = b_norm * sigma
        m = compute_metrics(w, biases_array_to_dict(b_phys), precomp)
        return m[objective_key] + lambda_reg * float(np.sum(b_norm * b_norm))

    res = sopt.minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )
    x_star = np.asarray(res.x, dtype=np.float64)
    w_star = x_star[:M]
    w_star = np.clip(w_star, 0.0, None)
    s = float(w_star.sum())
    w_star = w_star / s if s > 0 else np.ones(M) / M
    b_norm_star = x_star[M:]
    b_phys_star = b_norm_star * sigma
    return w_star, b_phys_star, float(res.fun), res


def normalised_bias(b_phys: np.ndarray, stats: dict[str, torch.Tensor]) -> dict[str, float]:
    """Return biases expressed in normalized (std-units) space for sanity check."""

    surface_y_std = stats["surface_y_std"].cpu().numpy()  # [4]
    volume_y_std = stats["volume_y_std"].cpu().numpy()    # [1]
    return {
        "surface_pressure": float(b_phys[0] / surface_y_std[0]),
        "wall_shear_x": float(b_phys[1] / surface_y_std[1]),
        "wall_shear_y": float(b_phys[2] / surface_y_std[2]),
        "wall_shear_z": float(b_phys[3] / surface_y_std[3]),
        "volume_pressure": float(b_phys[4] / volume_y_std[0]),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bias-corrected ensemble search (PR #1108)"
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=["56bcqp3m", "29nohj67", "a0yoxy85", "ghh0s4ne"],
    )
    parser.add_argument("--pred-cache-dir", type=str, required=True)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.375, 0.250, 0.250, 0.125],
        help="K=8 Caruana weights for Arm A.",
    )
    parser.add_argument(
        "--lambda-sweep",
        nargs="+",
        type=float,
        default=[0.0, 1e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1],
    )
    parser.add_argument(
        "--arm-b-lambda-sweep",
        nargs="+",
        type=float,
        default=[1e-3, 1e-2, 0.1],
    )
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--eval-surface-points", type=int, default=65536)
    parser.add_argument("--eval-volume-points", type=int, default=65536)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-loocv", action="store_true", help="Skip LOOCV sweep.")
    parser.add_argument("--no-arm-b", action="store_true", help="Skip Arm B.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/bias_corrected_results.json",
    )
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-group", default="edward-bias-corrected-ensemble")
    parser.add_argument("--wandb-name", default="edward/bias-corrected-arm-a")
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=["ensemble", "bias-corrected", "edward"],
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--objective",
        default="abupt_axis_mean_rel_l2_pct",
        choices=[
            "abupt_axis_mean_rel_l2_pct",
            "wall_shear_rel_l2_pct",
        ],
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_ids = list(args.run_ids)
    M = len(run_ids)
    w_k8 = np.asarray(args.weights, dtype=np.float64)
    print(f"K={M} pool: {run_ids}")
    print(f"K=8 weights: {w_k8.tolist()}")
    print(f"Objective: {args.objective}")

    pred_cache_dir = Path(args.pred_cache_dir)
    missing: list[str] = []
    for split in ("val", "test"):
        mp = meta_cache_path(pred_cache_dir, split)
        if not mp.exists():
            missing.append(str(mp))
        for run_id in run_ids:
            pp = pred_cache_path(pred_cache_dir, run_id, split)
            if not pp.exists():
                missing.append(str(pp))
    if missing:
        raise SystemExit(
            "Missing cache files (run ensemble_eval.py --cache-only first):\n  "
            + "\n  ".join(missing)
        )

    print("Loading data stats for denormalization...")
    _, _, _, stats = load_data(
        manifest_path=args.manifest,
        root=args.data_root or None,
        train_surface_points=args.eval_surface_points,
        eval_surface_points=args.eval_surface_points,
        train_volume_points=args.eval_volume_points,
        eval_volume_points=args.eval_volume_points,
        debug=False,
    )
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )
    sigma = channel_std_array(stats)
    print(f"Per-channel std (physical units): {dict(zip(PER_AXIS_CHANNELS, [round(float(s), 4) for s in sigma]))}")

    precomp_val: PrecomputedStats | None = None
    precomp_test: PrecomputedStats | None = None
    for split, store in (("val", "val"), ("test", "test")):
        print(f"Loading cached predictions for {split}...")
        batch_meta = load_meta_cache_from_disk(meta_cache_path(pred_cache_dir, split))
        pred_cache_split: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for run_id in run_ids:
            pred_cache_split[run_id] = load_pred_cache_from_disk(
                pred_cache_path(pred_cache_dir, run_id, split)
            )
        print(
            f"  {split}: {len(batch_meta)} batches, "
            f"{sum(len(pred_cache_split[r]) for r in run_ids)} cached pred lists"
        )
        t0 = time.time()
        ps = precompute_stats_for_split(
            run_ids, batch_meta, pred_cache_split, transform, device
        )
        print(
            f"  precomputed in {time.time() - t0:.1f}s, {ps.num_cases()} cases"
        )
        if split == "val":
            precomp_val = ps
        else:
            precomp_test = ps
        # Free large cached tensors before loading the next split.
        del batch_meta, pred_cache_split
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert precomp_val is not None and precomp_test is not None

    # K=8 sanity baseline (no bias).
    print("\n=== K=8 Caruana baseline (no bias) ===")
    zero_b = biases_array_to_dict(np.zeros(len(PER_AXIS_CHANNELS)))
    k8_val = compute_metrics(w_k8, zero_b, precomp_val)
    k8_test = compute_metrics(w_k8, zero_b, precomp_test)
    print("  Val:  ", {k: round(v, 4) for k, v in k8_val.items()})
    print("  Test: ", {k: round(v, 4) for k, v in k8_test.items()})

    # Arm A: fixed weights, optimize biases per λ.
    print("\n=== Arm A: fixed K=8 weights, optimise biases per λ ===")
    arm_a_results: list[dict] = []
    for lam in args.lambda_sweep:
        b_star, fun, res = fit_biases_slsqp(
            precomp_val,
            weights=w_k8,
            lambda_reg=lam,
            sigma=sigma,
            objective_key=args.objective,
        )
        val_m = compute_metrics(w_k8, biases_array_to_dict(b_star), precomp_val)
        test_m = compute_metrics(w_k8, biases_array_to_dict(b_star), precomp_test)
        b_norm = normalised_bias(b_star, stats)
        row = {
            "lambda": float(lam),
            "b_star_denorm": b_star.tolist(),
            "b_star_norm": b_norm,
            "fun": float(fun),
            "val": val_m,
            "test": test_m,
            "success": bool(res.success),
            "nit": int(getattr(res, "nit", 0)),
            "nfev": int(getattr(res, "nfev", 0)),
            "message": str(res.message),
        }
        arm_a_results.append(row)
        print(
            f"  λ={lam:.0e}  b={[round(x,4) for x in b_star]}  "
            f"|b|_norm={ {k: round(v,3) for k,v in b_norm.items()} }  "
            f"val_abupt={val_m['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"val_WSS={val_m['wall_shear_rel_l2_pct']:.4f}  "
            f"val_SP={val_m['surface_pressure_rel_l2_pct']:.4f}  "
            f"val_vol_p={val_m['volume_pressure_rel_l2_pct']:.4f}  "
            f"test_WSS={test_m['wall_shear_rel_l2_pct']:.4f}  "
            f"test_abupt={test_m['abupt_axis_mean_rel_l2_pct']:.4f}"
        )

    # LOOCV for Arm A.
    loocv_rows: list[dict] = []
    if not args.no_loocv:
        print(
            "\n=== Arm A LOOCV (34 holdouts × len(λ-sweep) solves) ==="
        )
        for lam in args.lambda_sweep:
            t0 = time.time()
            loocv_m = loocv_arm_a(
                precomp_val,
                weights=w_k8,
                lambda_reg=lam,
                sigma=sigma,
                objective_key=args.objective,
            )
            elapsed = time.time() - t0
            loocv_rows.append({"lambda": float(lam), "loocv_val": loocv_m, "elapsed_sec": elapsed})
            print(
                f"  λ={lam:.0e}  loocv_abupt={loocv_m['abupt_axis_mean_rel_l2_pct']:.4f}  "
                f"loocv_WSS={loocv_m['wall_shear_rel_l2_pct']:.4f}  "
                f"loocv_SP={loocv_m['surface_pressure_rel_l2_pct']:.4f}  "
                f"loocv_vol_p={loocv_m['volume_pressure_rel_l2_pct']:.4f}  "
                f"({elapsed:.1f}s)"
            )
    # Pick λ* by best LOOCV val_abupt (or fall back to fit val_abupt if no LOOCV).
    if loocv_rows:
        lstar_row = min(loocv_rows, key=lambda r: r["loocv_val"]["abupt_axis_mean_rel_l2_pct"])
        lambda_star = lstar_row["lambda"]
        print(
            f"\nλ* by LOOCV: {lambda_star:.0e}  "
            f"(loocv_abupt={lstar_row['loocv_val']['abupt_axis_mean_rel_l2_pct']:.4f}, "
            f"K=8 baseline loocv_abupt N/A in this run)"
        )
    else:
        lstar_row = min(arm_a_results, key=lambda r: r["val"]["abupt_axis_mean_rel_l2_pct"])
        lambda_star = lstar_row["lambda"]
        print(f"\nλ* by val fit: {lambda_star:.0e}")

    star_arm_a = next(r for r in arm_a_results if r["lambda"] == lambda_star)

    # Arm B: joint optimisation.
    arm_b_results: list[dict] = []
    if not args.no_arm_b:
        print(
            "\n=== Arm B: joint (w, b) optimisation with simplex constraint ==="
        )
        for lam in args.arm_b_lambda_sweep:
            w_star, b_star, fun, res = fit_joint_w_b_slsqp(
                precomp_val,
                lambda_reg=lam,
                sigma=sigma,
                objective_key=args.objective,
            )
            val_m = compute_metrics(w_star, biases_array_to_dict(b_star), precomp_val)
            test_m = compute_metrics(w_star, biases_array_to_dict(b_star), precomp_test)
            b_norm = normalised_bias(b_star, stats)
            row = {
                "lambda": float(lam),
                "w_star": w_star.tolist(),
                "b_star_denorm": b_star.tolist(),
                "b_star_norm": b_norm,
                "weight_shift_from_k8": float(np.abs(w_star - w_k8).sum()),
                "fun": float(fun),
                "val": val_m,
                "test": test_m,
                "success": bool(res.success),
                "nit": int(getattr(res, "nit", 0)),
                "nfev": int(getattr(res, "nfev", 0)),
                "message": str(res.message),
            }
            arm_b_results.append(row)
            print(
                f"  λ={lam:.0e}  w={[round(x,4) for x in w_star]}  "
                f"L1(w-w_K8)={row['weight_shift_from_k8']:.4f}  "
                f"b={[round(x,4) for x in b_star]}  "
                f"val_abupt={val_m['abupt_axis_mean_rel_l2_pct']:.4f}  "
                f"val_WSS={val_m['wall_shear_rel_l2_pct']:.4f}  "
                f"test_abupt={test_m['abupt_axis_mean_rel_l2_pct']:.4f}  "
                f"test_WSS={test_m['wall_shear_rel_l2_pct']:.4f}"
            )

    # Final analysis.
    print("\n=== Summary ===")
    print(
        f"K=8 baseline:   val_abupt={k8_val['abupt_axis_mean_rel_l2_pct']:.4f}  "
        f"val_WSS={k8_val['wall_shear_rel_l2_pct']:.4f}  "
        f"test_abupt={k8_test['abupt_axis_mean_rel_l2_pct']:.4f}  "
        f"test_WSS={k8_test['wall_shear_rel_l2_pct']:.4f}"
    )
    saa = star_arm_a
    print(
        f"Arm A λ*={lambda_star:.0e}:  val_abupt={saa['val']['abupt_axis_mean_rel_l2_pct']:.4f}  "
        f"val_WSS={saa['val']['wall_shear_rel_l2_pct']:.4f}  "
        f"test_abupt={saa['test']['abupt_axis_mean_rel_l2_pct']:.4f}  "
        f"test_WSS={saa['test']['wall_shear_rel_l2_pct']:.4f}"
    )
    if arm_b_results:
        best_b = min(arm_b_results, key=lambda r: r["val"]["abupt_axis_mean_rel_l2_pct"])
        print(
            f"Arm B best λ:   val_abupt={best_b['val']['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"val_WSS={best_b['val']['wall_shear_rel_l2_pct']:.4f}  "
            f"test_abupt={best_b['test']['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"test_WSS={best_b['test']['wall_shear_rel_l2_pct']:.4f}"
        )

    # Win-gate check: val_abupt < 5.7452 AND test_vol_p <= 3.643 AND test_WSS < 6.3263.
    def win_gate(val_m: dict, test_m: dict) -> bool:
        return (
            val_m["abupt_axis_mean_rel_l2_pct"] < 5.7452
            and test_m["volume_pressure_rel_l2_pct"] <= 3.643
            and test_m["wall_shear_rel_l2_pct"] < 6.3263
        )

    arm_a_win = win_gate(saa["val"], saa["test"])
    arm_b_win = bool(arm_b_results) and win_gate(best_b["val"], best_b["test"])
    print(f"Arm A win gate: {arm_a_win}  (val_abupt={saa['val']['abupt_axis_mean_rel_l2_pct']:.4f}<5.7452? {saa['val']['abupt_axis_mean_rel_l2_pct']<5.7452}, "
          f"test_vol_p={saa['test']['volume_pressure_rel_l2_pct']:.4f}<=3.643? {saa['test']['volume_pressure_rel_l2_pct']<=3.643}, "
          f"test_WSS={saa['test']['wall_shear_rel_l2_pct']:.4f}<6.3263? {saa['test']['wall_shear_rel_l2_pct']<6.3263})")
    if arm_b_results:
        print(f"Arm B win gate: {arm_b_win}")

    # Dump everything.
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_ids": run_ids,
        "k8_weights": w_k8.tolist(),
        "k8_val": k8_val,
        "k8_test": k8_test,
        "objective": args.objective,
        "lambda_sweep": list(args.lambda_sweep),
        "arm_a": arm_a_results,
        "loocv": loocv_rows,
        "lambda_star": float(lambda_star),
        "arm_a_at_star": star_arm_a,
        "arm_b_lambda_sweep": list(args.arm_b_lambda_sweep),
        "arm_b": arm_b_results,
        "arm_a_win_gate": arm_a_win,
        "arm_b_win_gate": arm_b_win if arm_b_results else None,
    }
    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {output_path}")

    # W&B logging.
    if not args.no_wandb:
        entity = args.wandb_entity or os.environ.get(
            "WANDB_ENTITY", "wandb-applied-ai-team"
        )
        project = args.wandb_project or os.environ.get(
            "WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"
        )
        run = wandb.init(
            entity=entity,
            project=project,
            group=args.wandb_group,
            name=args.wandb_name,
            tags=list(args.wandb_tags),
            config={
                "mode": "bias_corrected_ensemble",
                "run_ids": run_ids,
                "K_weights": w_k8.tolist(),
                "objective": args.objective,
                "lambda_sweep": list(args.lambda_sweep),
                "arm_b_lambda_sweep": list(args.arm_b_lambda_sweep),
            },
            mode=os.environ.get("WANDB_MODE", "online"),
        )

        summary: dict = {}
        for k, v in k8_val.items():
            summary[f"baseline_k8_val/{k}"] = float(v)
        for k, v in k8_test.items():
            summary[f"baseline_k8_test/{k}"] = float(v)

        for row in arm_a_results:
            lam = row["lambda"]
            tag = f"arm_a/lambda_{lam:.0e}".replace("+", "p")
            for k, v in row["val"].items():
                summary[f"{tag}/val/{k}"] = float(v)
            for k, v in row["test"].items():
                summary[f"{tag}/test/{k}"] = float(v)
            for i, ch in enumerate(PER_AXIS_CHANNELS):
                summary[f"{tag}/b_denorm/{ch}"] = float(row["b_star_denorm"][i])
                summary[f"{tag}/b_norm/{ch}"] = float(row["b_star_norm"][ch])

        for row in loocv_rows:
            lam = row["lambda"]
            tag = f"loocv/lambda_{lam:.0e}".replace("+", "p")
            for k, v in row["loocv_val"].items():
                summary[f"{tag}/{k}"] = float(v)

        # Star metrics for Arm A
        for k, v in star_arm_a["val"].items():
            summary[f"star_arm_a_val/{k}"] = float(v)
        for k, v in star_arm_a["test"].items():
            summary[f"star_arm_a_test/{k}"] = float(v)
            # Mirror to test_primary/* so the result surfaces in standard dashboards
            summary[f"test_primary/{k}"] = float(v)
        summary["lambda_star"] = float(lambda_star)
        summary["arm_a_win_gate"] = bool(arm_a_win)

        for row in arm_b_results:
            lam = row["lambda"]
            tag = f"arm_b/lambda_{lam:.0e}".replace("+", "p")
            for k, v in row["val"].items():
                summary[f"{tag}/val/{k}"] = float(v)
            for k, v in row["test"].items():
                summary[f"{tag}/test/{k}"] = float(v)
            for i, ch in enumerate(PER_AXIS_CHANNELS):
                summary[f"{tag}/b_denorm/{ch}"] = float(row["b_star_denorm"][i])
                summary[f"{tag}/b_norm/{ch}"] = float(row["b_star_norm"][ch])
            for i, rid in enumerate(run_ids):
                summary[f"{tag}/w/{rid}"] = float(row["w_star"][i])
            summary[f"{tag}/weight_shift_from_k8"] = float(row["weight_shift_from_k8"])

        wandb.log(summary)
        wandb.summary.update(summary)
        run_id = run.id
        wandb.finish()
        print(f"\nW&B run ID: {run_id}")


if __name__ == "__main__":
    main()
