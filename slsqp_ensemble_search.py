# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""SLSQP continuous weight search over the 4-simplex for the K=4 ensemble pool.

Loads predictions cached on disk by ``ensemble_eval.py --cache-only`` and runs
``scipy.optimize.minimize(method='SLSQP')`` on the 4-simplex to find the
weights that minimise val_WSS subject to optional ceiling constraints on
val_vol_p / val_SP.

Per-case rel-L2 metrics are linear functions of error cross-correlations
(when sum(w)=1), so each SLSQP evaluation is essentially free once
cross-correlations are precomputed. We exploit:

    (sum_i w_i * (pred_i - target))^2 = sum_ij w_i*w_j*(pred_i-t)(pred_j-t)

so per-case SSE = sum_ij w_i*w_j * cross_ij[case], where the cross matrix
is precomputed once per (channel, case) over all points in the case.

Use:

    python slsqp_ensemble_search.py \\
        --pred-cache-dir /tmp/ensemble_pred_cache \\
        --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \\
        --run-ids 56bcqp3m 29nohj67 a0yoxy85 ghh0s4ne \\
        --wandb-group edward-slsqp-ensemble \\
        --wandb-name edward/slsqp-continuous
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


# Channels we care about for the rel-L2 metrics.
SURFACE_CHANNELS = {
    "surface_pressure": (0, 1),  # slice columns 0:1
    "wall_shear": (1, 4),
    "wall_shear_x": (1, 2),
    "wall_shear_y": (2, 3),
    "wall_shear_z": (3, 4),
}

ALL_CHANNELS = list(SURFACE_CHANNELS.keys()) + ["volume_pressure"]


@dataclass
class PrecomputedCorrelations:
    """Per-case error cross-correlations and target SS for a split."""

    # cross_per_channel[channel]: tensor [num_cases, M, M] (float64, CPU)
    cross_per_channel: dict[str, torch.Tensor]
    # ss_target_per_channel[channel]: tensor [num_cases] (float64, CPU)
    ss_target_per_channel: dict[str, torch.Tensor]
    # case ordering used in the tensors above
    case_ids: list[str]
    # mask marking which cases have data per channel (some volumes may be empty in a view)
    valid_mask_per_channel: dict[str, torch.Tensor]


def precompute_correlations_for_split(
    run_ids: list[str],
    batch_meta: list[dict],
    pred_cache: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    transform: TargetTransform,
    device: torch.device,
) -> PrecomputedCorrelations:
    """One pass over the cached predictions to fill per-case sums.

    Surface targets are 4-channel ``[surface_pressure, wall_shear_x/y/z]``;
    volume targets are 1-channel ``[volume_pressure]``. We accumulate per-case:
    cross[case, i, j] = sum over points (and within-channel sub-axes) of
    ``(pred_i - target) * (pred_j - target)`` and ss_target[case] = sum
    target^2 over the same points.
    """

    M = len(run_ids)

    # Per-channel sparse accumulators keyed by case_id.
    cross_by_case: dict[str, dict[str, torch.Tensor]] = {ch: {} for ch in ALL_CHANNELS}
    ss_target_by_case: dict[str, dict[str, float]] = {ch: {} for ch in ALL_CHANNELS}

    for batch_idx, meta in enumerate(batch_meta):
        surface_y = meta["surface_y"].to(device=device, dtype=torch.float32)
        volume_y = meta["volume_y"].to(device=device, dtype=torch.float32)
        surface_mask = meta["surface_mask"].to(device=device, dtype=torch.bool)
        volume_mask = meta["volume_mask"].to(device=device, dtype=torch.bool)
        case_ids = list(meta["case_ids"])

        # Stack member predictions into [M, B, N, C] (denormalized errors).
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

        # Process each case in this batch.
        for case_idx, case_id in enumerate(case_ids):
            s_valid = surface_mask[case_idx]
            if bool(s_valid.any()):
                s_target_valid = surface_y[case_idx][s_valid]  # [N, 4]
                surface_errs_valid = [
                    se[case_idx][s_valid] for se in surface_errs
                ]  # list of [N, 4]
                for ch_name, (lo, hi) in SURFACE_CHANNELS.items():
                    target_ch = s_target_valid[:, lo:hi]
                    ss_target_ch = float(target_ch.pow(2).sum().item())
                    if ss_target_ch <= 0.0:
                        continue
                    err_stack = torch.stack(
                        [se[:, lo:hi] for se in surface_errs_valid], dim=0
                    )  # [M, N, k]
                    err_flat = err_stack.reshape(M, -1)  # [M, N*k]
                    cross = err_flat @ err_flat.T  # [M, M]
                    if case_id in cross_by_case[ch_name]:
                        cross_by_case[ch_name][case_id] += cross.double().cpu()
                        ss_target_by_case[ch_name][case_id] += ss_target_ch
                    else:
                        cross_by_case[ch_name][case_id] = cross.double().cpu()
                        ss_target_by_case[ch_name][case_id] = ss_target_ch
            v_valid = volume_mask[case_idx]
            if bool(v_valid.any()):
                v_target_valid = volume_y[case_idx][v_valid]
                ss_target_ch = float(v_target_valid.pow(2).sum().item())
                if ss_target_ch <= 0.0:
                    continue
                volume_errs_valid = [
                    ve[case_idx][v_valid] for ve in volume_errs
                ]  # list of [N, 1]
                err_stack = torch.stack(volume_errs_valid, dim=0)  # [M, N, 1]
                err_flat = err_stack.reshape(M, -1)
                cross = err_flat @ err_flat.T
                ch_name = "volume_pressure"
                if case_id in cross_by_case[ch_name]:
                    cross_by_case[ch_name][case_id] += cross.double().cpu()
                    ss_target_by_case[ch_name][case_id] += ss_target_ch
                else:
                    cross_by_case[ch_name][case_id] = cross.double().cpu()
                    ss_target_by_case[ch_name][case_id] = ss_target_ch

    # Gather every case ID and stack tensors into [num_cases, M, M].
    case_ids_sorted = sorted(
        {cid for ch in ALL_CHANNELS for cid in cross_by_case[ch].keys()}
    )
    cross_per_channel: dict[str, torch.Tensor] = {}
    ss_target_per_channel: dict[str, torch.Tensor] = {}
    valid_mask_per_channel: dict[str, torch.Tensor] = {}
    zero_cross = torch.zeros((M, M), dtype=torch.float64)
    for ch_name in ALL_CHANNELS:
        per_case = cross_by_case[ch_name]
        ss = ss_target_by_case[ch_name]
        cross_stack = torch.stack(
            [per_case.get(cid, zero_cross) for cid in case_ids_sorted], dim=0
        )  # [num_cases, M, M]
        ss_vec = torch.tensor(
            [ss.get(cid, 0.0) for cid in case_ids_sorted], dtype=torch.float64
        )
        mask = ss_vec > 0
        cross_per_channel[ch_name] = cross_stack
        ss_target_per_channel[ch_name] = ss_vec
        valid_mask_per_channel[ch_name] = mask
    return PrecomputedCorrelations(
        cross_per_channel=cross_per_channel,
        ss_target_per_channel=ss_target_per_channel,
        case_ids=case_ids_sorted,
        valid_mask_per_channel=valid_mask_per_channel,
    )


def compute_metrics_from_correlations(
    weights: np.ndarray | torch.Tensor,
    precomp: PrecomputedCorrelations,
) -> dict[str, float]:
    """Compute rel-L2 metrics in percent for an arbitrary weight vector."""

    w = torch.as_tensor(weights, dtype=torch.float64)
    rel_l2: dict[str, float] = {}
    for ch_name in ALL_CHANNELS:
        cross = precomp.cross_per_channel[ch_name]  # [num_cases, M, M]
        ss_t = precomp.ss_target_per_channel[ch_name]  # [num_cases]
        mask = precomp.valid_mask_per_channel[ch_name]
        # sse_per_case[c] = sum_{i,j} w_i * w_j * cross[c,i,j]
        sse_per_case = torch.einsum("cij,i,j->c", cross, w, w)
        # rel L2 per case, then mean over cases
        per_case = sse_per_case[mask] / ss_t[mask]
        per_case = per_case.clamp(min=0.0)
        rel_l2_per_case = per_case.sqrt() * 100.0
        rel_l2[f"{ch_name}_rel_l2_pct"] = float(rel_l2_per_case.mean().item())
    rel_l2["abupt_axis_mean_rel_l2_pct"] = float(
        np.mean(
            [
                rel_l2["surface_pressure_rel_l2_pct"],
                rel_l2["wall_shear_x_rel_l2_pct"],
                rel_l2["wall_shear_y_rel_l2_pct"],
                rel_l2["wall_shear_z_rel_l2_pct"],
                rel_l2["volume_pressure_rel_l2_pct"],
            ]
        )
    )
    return rel_l2


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 0.0, None)
    s = float(weights.sum())
    if s <= 0:
        return np.ones_like(weights) / weights.size
    return weights / s


def run_slsqp(
    precomp_val: PrecomputedCorrelations,
    *,
    starts: list[np.ndarray],
    objective_key: str = "wall_shear_rel_l2_pct",
    vol_p_ceiling: float | None,
    sp_ceiling: float | None,
    label: str,
) -> list[dict]:
    """Run SLSQP from each starting point and return per-start results."""

    M = next(iter(precomp_val.cross_per_channel.values())).shape[1]
    results: list[dict] = []

    def metric(w_np: np.ndarray, key: str) -> float:
        return compute_metrics_from_correlations(w_np, precomp_val)[key]

    def objective(w_np: np.ndarray) -> float:
        return metric(w_np, objective_key)

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
    ]
    if vol_p_ceiling is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: float(
                    vol_p_ceiling - metric(w, "volume_pressure_rel_l2_pct")
                ),
            }
        )
    if sp_ceiling is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: float(
                    sp_ceiling - metric(w, "surface_pressure_rel_l2_pct")
                ),
            }
        )

    bounds = [(0.0, 1.0)] * M

    for idx, w0 in enumerate(starts):
        w0_n = normalize_weights(np.asarray(w0, dtype=np.float64))
        t0 = time.time()
        res = sopt.minimize(
            objective,
            w0_n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8, "disp": False},
        )
        elapsed = time.time() - t0
        w_final = normalize_weights(res.x)
        metrics_final = compute_metrics_from_correlations(w_final, precomp_val)
        info = {
            "label": label,
            "start_index": idx,
            "start_weights": w0_n.tolist(),
            "final_weights": w_final.tolist(),
            "fun": float(res.fun),
            "objective": objective_key,
            "metrics": metrics_final,
            "success": bool(res.success),
            "status": int(res.status),
            "nit": int(getattr(res, "nit", 0)),
            "nfev": int(getattr(res, "nfev", 0)),
            "message": str(res.message),
            "elapsed_sec": float(elapsed),
            "feasible_vol_p": (
                None
                if vol_p_ceiling is None
                else bool(
                    metrics_final["volume_pressure_rel_l2_pct"] <= vol_p_ceiling + 1e-6
                )
            ),
            "feasible_sp": (
                None
                if sp_ceiling is None
                else bool(
                    metrics_final["surface_pressure_rel_l2_pct"] <= sp_ceiling + 1e-6
                )
            ),
        }
        results.append(info)
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SLSQP continuous weight search for K=4 ensemble"
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=["56bcqp3m", "29nohj67", "a0yoxy85", "ghh0s4ne"],
        help="W&B run IDs for the K=4 candidate pool (order matters).",
    )
    parser.add_argument(
        "--pred-cache-dir",
        type=str,
        required=True,
        help="Path to cached predictions (written by ensemble_eval.py --cache-only).",
    )
    parser.add_argument(
        "--manifest", default="data/split_manifest.json",
    )
    parser.add_argument(
        "--data-root", default="", help="DrivAerML processed data root."
    )
    parser.add_argument("--eval-surface-points", type=int, default=65536)
    parser.add_argument("--eval-volume-points", type=int, default=65536)
    parser.add_argument(
        "--vol-p-ceiling",
        type=float,
        default=3.643,
        help="val_vol_p ceiling (pct) for the constrained search.",
    )
    parser.add_argument(
        "--sp-ceiling",
        type=float,
        default=3.577,
        help="val_SP ceiling (pct) for the constrained search.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/slsqp_results.json",
        help="Path to write the full results dump (JSON).",
    )
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-group", default="edward-slsqp-ensemble")
    parser.add_argument("--wandb-name", default="edward/slsqp-continuous")
    parser.add_argument("--wandb-tags", nargs="*", default=["ensemble", "slsqp", "edward"])
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_ids = list(args.run_ids)
    M = len(run_ids)

    print(f"K={M} pool: {run_ids}")

    # Validate every cache file is present.
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

    # Load splits stats (for the target normalizer / denorm).
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

    # Load cached predictions.
    print("Loading cached predictions...")
    pred_cache: dict[str, dict[str, list[tuple[torch.Tensor, torch.Tensor]]]] = {
        "val": {},
        "test": {},
    }
    batch_meta: dict[str, list[dict]] = {}
    for split in ("val", "test"):
        batch_meta[split] = load_meta_cache_from_disk(meta_cache_path(pred_cache_dir, split))
        for run_id in run_ids:
            pred_cache[split][run_id] = load_pred_cache_from_disk(
                pred_cache_path(pred_cache_dir, run_id, split)
            )
        print(
            f"  {split}: {len(batch_meta[split])} batches, "
            f"{sum(len(pred_cache[split][r]) for r in run_ids)} cached pred lists"
        )

    # Precompute per-case correlations.
    print("Precomputing per-case correlations (val)...")
    t0 = time.time()
    precomp_val = precompute_correlations_for_split(
        run_ids, batch_meta["val"], pred_cache["val"], transform, device
    )
    print(f"  done in {time.time() - t0:.1f}s, {len(precomp_val.case_ids)} cases")
    print("Precomputing per-case correlations (test)...")
    t0 = time.time()
    precomp_test = precompute_correlations_for_split(
        run_ids, batch_meta["test"], pred_cache["test"], transform, device
    )
    print(f"  done in {time.time() - t0:.1f}s, {len(precomp_test.case_ids)} cases")

    # Sanity: K=8 Caruana baseline reproduction.
    print("\n=== Sanity: K=8 Caruana baseline weights ===")
    k8_weights = np.array([0.375, 0.250, 0.250, 0.125])
    k8_val = compute_metrics_from_correlations(k8_weights, precomp_val)
    k8_test = compute_metrics_from_correlations(k8_weights, precomp_test)
    print("  Val:  ", {k: round(v, 4) for k, v in k8_val.items()})
    print("  Test: ", {k: round(v, 4) for k, v in k8_test.items()})

    # Per-member single-model metrics (sanity).
    print("\n=== Per-member single-model val metrics ===")
    per_member_val: dict[str, dict[str, float]] = {}
    per_member_test: dict[str, dict[str, float]] = {}
    for i, run_id in enumerate(run_ids):
        w_single = np.zeros(M)
        w_single[i] = 1.0
        per_member_val[run_id] = compute_metrics_from_correlations(w_single, precomp_val)
        per_member_test[run_id] = compute_metrics_from_correlations(w_single, precomp_test)
        m = per_member_val[run_id]
        print(
            f"  {run_id}:  abupt={m['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"WSS={m['wall_shear_rel_l2_pct']:.4f}  "
            f"vol_p={m['volume_pressure_rel_l2_pct']:.4f}  "
            f"SP={m['surface_pressure_rel_l2_pct']:.4f}"
        )

    # Starting points for SLSQP (must lie on the simplex).
    starts = [
        np.array([0.375, 0.250, 0.250, 0.125]),  # K=8 Caruana
        np.array([0.25, 0.25, 0.25, 0.25]),  # uniform
        np.array([0.40, 0.30, 0.20, 0.10]),  # WSS-focused
        np.array([0.35, 0.25, 0.25, 0.15]),  # ghh0s4ne-boosted
        np.array([0.45, 0.25, 0.20, 0.10]),  # vol_p-relaxed
    ]

    print("\n=== Unconstrained SLSQP (val_WSS floor, only simplex) ===")
    unconstrained = run_slsqp(
        precomp_val,
        starts=starts,
        objective_key="wall_shear_rel_l2_pct",
        vol_p_ceiling=None,
        sp_ceiling=None,
        label="unconstrained",
    )
    for r in unconstrained:
        w = r["final_weights"]
        m = r["metrics"]
        print(
            f"  start={r['start_index']}  w={[round(x,4) for x in w]}  "
            f"val_WSS={m['wall_shear_rel_l2_pct']:.4f}  "
            f"abupt={m['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"vol_p={m['volume_pressure_rel_l2_pct']:.4f}  "
            f"SP={m['surface_pressure_rel_l2_pct']:.4f}  "
            f"nit={r['nit']} success={r['success']}"
        )

    print(
        f"\n=== Constrained SLSQP (val_vol_p<={args.vol_p_ceiling:.3f}, val_SP<={args.sp_ceiling:.3f}) ==="
    )
    constrained = run_slsqp(
        precomp_val,
        starts=starts,
        objective_key="wall_shear_rel_l2_pct",
        vol_p_ceiling=args.vol_p_ceiling,
        sp_ceiling=args.sp_ceiling,
        label="constrained",
    )
    for r in constrained:
        w = r["final_weights"]
        m = r["metrics"]
        print(
            f"  start={r['start_index']}  w={[round(x,4) for x in w]}  "
            f"val_WSS={m['wall_shear_rel_l2_pct']:.4f}  "
            f"abupt={m['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"vol_p={m['volume_pressure_rel_l2_pct']:.4f}  "
            f"SP={m['surface_pressure_rel_l2_pct']:.4f}  "
            f"feas_vp={r['feasible_vol_p']} feas_sp={r['feasible_sp']} "
            f"success={r['success']}"
        )

    # Supplementary variants explored after seeing val_SP<=3.577 is infeasible
    # on this 4-member pool (the lowest reachable val_SP is ~3.72%).
    k8_val_sp = float(k8_val["surface_pressure_rel_l2_pct"])
    k8_val_vol_p = float(k8_val["volume_pressure_rel_l2_pct"])

    print(
        f"\n=== Constrained SLSQP — no-regression vs K=8 "
        f"(val_vol_p<={k8_val_vol_p:.4f}, val_SP<={k8_val_sp:.4f}) ==="
    )
    no_regression = run_slsqp(
        precomp_val,
        starts=starts,
        objective_key="wall_shear_rel_l2_pct",
        vol_p_ceiling=k8_val_vol_p,
        sp_ceiling=k8_val_sp,
        label="no_regression",
    )
    for r in no_regression:
        w = r["final_weights"]
        m = r["metrics"]
        print(
            f"  start={r['start_index']}  w={[round(x,4) for x in w]}  "
            f"val_WSS={m['wall_shear_rel_l2_pct']:.4f}  "
            f"abupt={m['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"vol_p={m['volume_pressure_rel_l2_pct']:.4f}  "
            f"SP={m['surface_pressure_rel_l2_pct']:.4f}  "
            f"feas_vp={r['feasible_vol_p']} feas_sp={r['feasible_sp']} "
            f"success={r['success']}"
        )

    print(
        f"\n=== Constrained SLSQP — vol_p-only (val_vol_p<={args.vol_p_ceiling:.3f}, no SP cap) ==="
    )
    volp_only = run_slsqp(
        precomp_val,
        starts=starts,
        objective_key="wall_shear_rel_l2_pct",
        vol_p_ceiling=args.vol_p_ceiling,
        sp_ceiling=None,
        label="volp_only",
    )
    for r in volp_only:
        w = r["final_weights"]
        m = r["metrics"]
        print(
            f"  start={r['start_index']}  w={[round(x,4) for x in w]}  "
            f"val_WSS={m['wall_shear_rel_l2_pct']:.4f}  "
            f"abupt={m['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"vol_p={m['volume_pressure_rel_l2_pct']:.4f}  "
            f"SP={m['surface_pressure_rel_l2_pct']:.4f}  "
            f"feas_vp={r['feasible_vol_p']} success={r['success']}"
        )

    # Pick the best from each batch.
    def best(rs: list[dict], feasible_only: bool) -> dict | None:
        candidates = [
            r
            for r in rs
            if (not feasible_only)
            or (
                (r["feasible_vol_p"] in (None, True))
                and (r["feasible_sp"] in (None, True))
                and r["success"]
            )
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r["fun"])

    best_unconstrained = best(unconstrained, feasible_only=False)
    best_constrained = best(constrained, feasible_only=True)
    best_no_regression = best(no_regression, feasible_only=True)
    best_volp_only = best(volp_only, feasible_only=True)
    print("\n=== Best so far ===")
    if best_unconstrained is not None:
        print(
            "  Unconstrained: ",
            {
                "w": [round(x, 4) for x in best_unconstrained["final_weights"]],
                "val_WSS": round(best_unconstrained["fun"], 4),
            },
        )
    else:
        print("  Unconstrained: NONE")
    if best_constrained is not None:
        print(
            "  Constrained:   ",
            {
                "w": [round(x, 4) for x in best_constrained["final_weights"]],
                "val_WSS": round(best_constrained["fun"], 4),
            },
        )
    else:
        print(
            "  Constrained:   INFEASIBLE under val_vol_p<="
            f"{args.vol_p_ceiling:.3f} AND val_SP<={args.sp_ceiling:.3f}"
        )
    if best_no_regression is not None:
        print(
            "  No-regression: ",
            {
                "w": [round(x, 4) for x in best_no_regression["final_weights"]],
                "val_WSS": round(best_no_regression["fun"], 4),
            },
        )
    else:
        print("  No-regression: NONE")
    if best_volp_only is not None:
        print(
            "  vol_p-only:    ",
            {
                "w": [round(x, 4) for x in best_volp_only["final_weights"]],
                "val_WSS": round(best_volp_only["fun"], 4),
            },
        )
    else:
        print("  vol_p-only:    NONE")

    # Evaluate the most interesting candidates on the test split.
    def evaluate_on_test(weights: list[float]) -> dict[str, float]:
        return compute_metrics_from_correlations(np.array(weights), precomp_test)

    eval_candidates: list[tuple[str, list[float]]] = [
        ("K=8 Caruana baseline", [0.375, 0.250, 0.250, 0.125]),
    ]
    if best_unconstrained is not None:
        eval_candidates.append(
            ("SLSQP unconstrained best", best_unconstrained["final_weights"])
        )
    if best_constrained is not None:
        eval_candidates.append(
            ("SLSQP constrained best", best_constrained["final_weights"])
        )
    if best_no_regression is not None:
        eval_candidates.append(
            ("SLSQP no-regression best", best_no_regression["final_weights"])
        )
    if best_volp_only is not None:
        eval_candidates.append(
            ("SLSQP vol_p-only best", best_volp_only["final_weights"])
        )
    # Also evaluate ALL SLSQP results to map the basin.
    for r in unconstrained:
        eval_candidates.append(
            (f"unconstrained start_{r['start_index']}", r["final_weights"])
        )
    for r in constrained:
        if r["feasible_vol_p"] is False or r["feasible_sp"] is False:
            continue
        eval_candidates.append(
            (f"constrained start_{r['start_index']}", r["final_weights"])
        )
    for r in no_regression:
        if r["feasible_vol_p"] is False or r["feasible_sp"] is False:
            continue
        eval_candidates.append(
            (f"no_regression start_{r['start_index']}", r["final_weights"])
        )
    for r in volp_only:
        if r["feasible_vol_p"] is False:
            continue
        eval_candidates.append(
            (f"volp_only start_{r['start_index']}", r["final_weights"])
        )

    print("\n=== Test-set evaluation of candidate weights ===")
    test_eval_rows: list[dict] = []
    for label, w in eval_candidates:
        val_m = compute_metrics_from_correlations(np.array(w), precomp_val)
        test_m = evaluate_on_test(w)
        test_eval_rows.append(
            {
                "label": label,
                "weights": w,
                "val_metrics": val_m,
                "test_metrics": test_m,
            }
        )
        print(
            f"  {label:35s} w={[round(x,4) for x in w]}  "
            f"val_abupt={val_m['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"val_WSS={val_m['wall_shear_rel_l2_pct']:.4f}  "
            f"test_abupt={test_m['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"test_WSS={test_m['wall_shear_rel_l2_pct']:.4f}  "
            f"test_vol_p={test_m['volume_pressure_rel_l2_pct']:.4f}  "
            f"test_SP={test_m['surface_pressure_rel_l2_pct']:.4f}"
        )

    # Dump everything as JSON for later inspection.
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_ids": run_ids,
        "k8_weights": k8_weights.tolist(),
        "k8_val": k8_val,
        "k8_test": k8_test,
        "per_member_val": per_member_val,
        "per_member_test": per_member_test,
        "unconstrained": unconstrained,
        "constrained": constrained,
        "no_regression": no_regression,
        "volp_only": volp_only,
        "vol_p_ceiling": args.vol_p_ceiling,
        "sp_ceiling": args.sp_ceiling,
        "k8_val_sp": k8_val_sp,
        "k8_val_vol_p": k8_val_vol_p,
        "test_evaluations": test_eval_rows,
    }
    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {output_path}")

    # Log to W&B.
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
                "mode": "slsqp_continuous",
                "run_ids": run_ids,
                "K": M,
                "vol_p_ceiling": args.vol_p_ceiling,
                "sp_ceiling": args.sp_ceiling,
                "starts": [s.tolist() for s in starts],
                "objective": "wall_shear_rel_l2_pct",
            },
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        summary: dict[str, float | str | bool | list] = {}
        # K=8 baseline reproduction
        for k, v in k8_val.items():
            summary[f"baseline_k8_val/{k}"] = float(v)
        for k, v in k8_test.items():
            summary[f"baseline_k8_test/{k}"] = float(v)
        # Per-start results
        for r in unconstrained:
            for k, v in r["metrics"].items():
                summary[
                    f"unconstrained/start_{r['start_index']}/val/{k}"
                ] = float(v)
            summary[
                f"unconstrained/start_{r['start_index']}/weights"
            ] = json.dumps(r["final_weights"])
            summary[
                f"unconstrained/start_{r['start_index']}/success"
            ] = bool(r["success"])
        for r in constrained:
            for k, v in r["metrics"].items():
                summary[f"constrained/start_{r['start_index']}/val/{k}"] = float(v)
            summary[
                f"constrained/start_{r['start_index']}/weights"
            ] = json.dumps(r["final_weights"])
            summary[f"constrained/start_{r['start_index']}/success"] = bool(
                r["success"]
            )
            summary[
                f"constrained/start_{r['start_index']}/feasible_vol_p"
            ] = bool(r["feasible_vol_p"])
            summary[
                f"constrained/start_{r['start_index']}/feasible_sp"
            ] = bool(r["feasible_sp"])
        # Best results
        if best_unconstrained is not None:
            for k, v in best_unconstrained["metrics"].items():
                summary[f"best_unconstrained_val/{k}"] = float(v)
            summary["best_unconstrained_weights"] = json.dumps(
                best_unconstrained["final_weights"]
            )
            test_m = compute_metrics_from_correlations(
                np.array(best_unconstrained["final_weights"]), precomp_test
            )
            for k, v in test_m.items():
                summary[f"best_unconstrained_test/{k}"] = float(v)
                summary[f"test_primary/{k}"] = float(v)
        if best_constrained is not None:
            for k, v in best_constrained["metrics"].items():
                summary[f"best_constrained_val/{k}"] = float(v)
            summary["best_constrained_weights"] = json.dumps(
                best_constrained["final_weights"]
            )
            test_m = compute_metrics_from_correlations(
                np.array(best_constrained["final_weights"]), precomp_test
            )
            for k, v in test_m.items():
                summary[f"best_constrained_test/{k}"] = float(v)
                # Mirror to test_primary/* for downstream baseline harvesting (winner only).
        # Supplementary variants
        for variant_name, variant_rs, variant_best in [
            ("no_regression", no_regression, best_no_regression),
            ("volp_only", volp_only, best_volp_only),
        ]:
            for r in variant_rs:
                for k, v in r["metrics"].items():
                    summary[f"{variant_name}/start_{r['start_index']}/val/{k}"] = float(v)
                summary[f"{variant_name}/start_{r['start_index']}/weights"] = json.dumps(
                    r["final_weights"]
                )
                summary[f"{variant_name}/start_{r['start_index']}/success"] = bool(
                    r["success"]
                )
                summary[
                    f"{variant_name}/start_{r['start_index']}/feasible_vol_p"
                ] = bool(r["feasible_vol_p"])
                if r["feasible_sp"] is not None:
                    summary[
                        f"{variant_name}/start_{r['start_index']}/feasible_sp"
                    ] = bool(r["feasible_sp"])
            if variant_best is not None:
                for k, v in variant_best["metrics"].items():
                    summary[f"best_{variant_name}_val/{k}"] = float(v)
                summary[f"best_{variant_name}_weights"] = json.dumps(
                    variant_best["final_weights"]
                )
                test_m = compute_metrics_from_correlations(
                    np.array(variant_best["final_weights"]), precomp_test
                )
                for k, v in test_m.items():
                    summary[f"best_{variant_name}_test/{k}"] = float(v)
        # Per-member metrics
        for run_id, m in per_member_val.items():
            for k, v in m.items():
                summary[f"per_member_val/{run_id}/{k}"] = float(v)
        for run_id, m in per_member_test.items():
            for k, v in m.items():
                summary[f"per_member_test/{run_id}/{k}"] = float(v)
        # All test evaluations
        for row in test_eval_rows:
            label = row["label"].replace(" ", "_").replace(",", "")
            for k, v in row["test_metrics"].items():
                summary[f"test_eval/{label}/{k}"] = float(v)
            for k, v in row["val_metrics"].items():
                summary[f"test_eval/{label}/val_{k}"] = float(v)
        wandb.log(summary)
        wandb.summary.update(summary)
        run_id = run.id
        wandb.finish()
        print(f"\nW&B run ID: {run_id}")


if __name__ == "__main__":
    main()
