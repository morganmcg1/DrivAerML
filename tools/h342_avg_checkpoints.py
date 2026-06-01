# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H342: Post-hoc multi-checkpoint output-averaging script.

Loads N per-case prediction .npz files (one per checkpoint) produced by
``eval_tta_h252.py --save-raw-predictions`` and averages the per-case
real-unit predictions point-wise across selected checkpoints, then computes
both H300 fresh-fit calibrated metrics (α/β fit on the averaged val output,
applied to the averaged test output) AND H312-hardcoded calibrated metrics
(fixed α/β recipe-invariant, applied identically across all arms).

Per-checkpoint and per-(k, R, mirror) pass averaging are linear with uniform
weights, so the cross-checkpoint mean of per-checkpoint per-case averages
equals the mean over all (ckpt, k, R, mirror) tuples — sufficient for
H342's output-averaging hypothesis.

Usage:
    python tools/h342_avg_checkpoints.py \\
        --inputs outputs/h342/ep13_raw.npz outputs/h342/ep14_raw.npz \\
                 outputs/h342/ep15_raw.npz \\
        --arms "A:2 B:1,2 C:0,2 D:0,1,2"

Each arm is "name:i,j,k" where i,j,k index into --inputs (0-based).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


SURF_CHANNELS = ["cp", "tau_x", "tau_y", "tau_z"]
N_SURF = 4
N_VOL = 1
N_STAT_COLS = 6  # [N, sum_p, sum_t, sum_pp, sum_tt, sum_pt]

# H312 hardcoded affine calibration (recipe-invariant — H332 finding).
# Surface betas are omitted in the PR text and observed ~0 across H314 fits,
# so we set them to exact zero. Volume beta is explicit.
H312_CAL = {
    "alpha_surf": torch.tensor(
        [0.994888, 0.994397, 0.994083, 0.991722], dtype=torch.float64
    ),
    "beta_surf": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    "alpha_vol": torch.tensor([0.999687], dtype=torch.float64),
    "beta_vol": torch.tensor([-0.832811], dtype=torch.float64),
}


def _channel_stats(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def _fit_affine(global_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _affine_err_sq(
    stats: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
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


def _compute_metrics_from_per_case_stats(
    per_case_surf: dict[str, torch.Tensor],
    per_case_vol: dict[str, torch.Tensor],
    alpha_surf: torch.Tensor | None = None,
    beta_surf: torch.Tensor | None = None,
    alpha_vol: torch.Tensor | None = None,
    beta_vol: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute per-case averaged rel_l2 metrics. If alpha/beta are None,
    metrics are computed on raw predictions (identity affine)."""
    if alpha_surf is None:
        alpha_surf = torch.ones(N_SURF, dtype=torch.float64)
        beta_surf = torch.zeros(N_SURF, dtype=torch.float64)
        alpha_vol = torch.ones(N_VOL, dtype=torch.float64)
        beta_vol = torch.zeros(N_VOL, dtype=torch.float64)

    surf_ids = sorted(per_case_surf.keys())
    if not surf_ids:
        raise RuntimeError("Empty per-case stats")
    surf_per_ch: list[torch.Tensor] = []
    vec_l2: list[torch.Tensor] = []
    for cid in surf_ids:
        s = per_case_surf[cid]
        e_sq = _affine_err_sq(s, alpha_surf, beta_surf)
        t_sq = s[..., 4]
        rel = torch.sqrt(e_sq.clamp(min=0.0) / t_sq.clamp(min=1e-12))
        surf_per_ch.append(rel)
        s_vec = s[1:4]
        e_sq_vec = _affine_err_sq(s_vec, alpha_surf[1:4], beta_surf[1:4])
        t_sq_vec = s_vec[..., 4]
        rel_vec = torch.sqrt(e_sq_vec.sum().clamp(min=0.0) / t_sq_vec.sum().clamp(min=1e-12))
        vec_l2.append(rel_vec)
    surf_rel = torch.stack(surf_per_ch, dim=0)
    vec_rel = torch.stack(vec_l2, dim=0)

    vol_ids = sorted(per_case_vol.keys())
    vol_per_ch: list[torch.Tensor] = []
    for cid in vol_ids:
        v = per_case_vol[cid]
        e_sq = _affine_err_sq(v, alpha_vol, beta_vol)
        t_sq = v[..., 4]
        rel = torch.sqrt(e_sq.clamp(min=0.0) / t_sq.clamp(min=1e-12))
        vol_per_ch.append(rel)
    vol_rel = torch.stack(vol_per_ch, dim=0)

    sp = float(surf_rel[:, 0].mean().item())
    tx = float(surf_rel[:, 1].mean().item())
    ty = float(surf_rel[:, 2].mean().item())
    tz = float(surf_rel[:, 3].mean().item())
    ws_vec = float(vec_rel.mean().item())
    vp = float(vol_rel[:, 0].mean().item())
    abupt = (sp + tx + ty + tz + vp) / 5.0
    return {
        "surface_pressure_rel_l2_pct": sp * 100.0,
        "wall_shear_rel_l2_pct": ws_vec * 100.0,
        "wall_shear_x_rel_l2_pct": tx * 100.0,
        "wall_shear_y_rel_l2_pct": ty * 100.0,
        "wall_shear_z_rel_l2_pct": tz * 100.0,
        "volume_pressure_rel_l2_pct": vp * 100.0,
        "abupt_axis_mean_rel_l2_pct": abupt * 100.0,
        "cases": float(len(surf_ids)),
    }


def _load_one(path: Path) -> dict:
    """Load one .npz; return {'val': {'surf':{cid:array},...}, 'test': {...},
    'metadata': dict}."""
    data = np.load(path, allow_pickle=False)
    keys = [k for k in data.keys() if not k.startswith("__")]
    parsed: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for k in keys:
        # surf_pred__val_surface__weight_noise_mirror_res_avg__run_4
        parts = k.split("__")
        kind = parts[0]  # surf_pred|vol_pred|surf_y|vol_y
        split = parts[1]
        cid = parts[3]
        parsed.setdefault(split, {}).setdefault(kind, {})[cid] = data[k]
    metadata = json.loads(str(data["__metadata__"]))
    return {**parsed, "metadata": metadata}


def _avg_predictions(
    loaded: list[dict], indices: list[int], split: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """For an arm = list of checkpoint indices, average predictions across
    them per case. Returns dicts of cid -> tensor for surf_pred, vol_pred,
    and the (shared) surf_y, vol_y targets."""
    if not indices:
        raise ValueError("Empty arm composition")

    case_id_sets = []
    for i in indices:
        ckpt_split = loaded[i].get(split, {})
        sp_keys = set(ckpt_split.get("surf_pred", {}).keys())
        vp_keys = set(ckpt_split.get("vol_pred", {}).keys())
        if sp_keys != vp_keys:
            raise RuntimeError(
                f"Checkpoint {i} split={split} has mismatched surf/vol case sets"
            )
        case_id_sets.append(sp_keys)
    cids = case_id_sets[0]
    for s in case_id_sets[1:]:
        if s != cids:
            extra = (s - cids) | (cids - s)
            raise RuntimeError(
                f"Checkpoints {indices} disagree on case IDs for split={split}: "
                f"diff={sorted(extra)}"
            )

    surf_pred: dict[str, torch.Tensor] = {}
    vol_pred: dict[str, torch.Tensor] = {}
    surf_y: dict[str, torch.Tensor] = {}
    vol_y: dict[str, torch.Tensor] = {}
    for cid in sorted(cids):
        # Average surf preds across checkpoints (float32 -> float64 for stability).
        sp_stack = np.stack(
            [loaded[i][split]["surf_pred"][cid].astype(np.float64) for i in indices], axis=0
        )
        sp_avg = sp_stack.mean(axis=0)
        vp_stack = np.stack(
            [loaded[i][split]["vol_pred"][cid].astype(np.float64) for i in indices], axis=0
        )
        vp_avg = vp_stack.mean(axis=0)
        surf_pred[cid] = torch.from_numpy(sp_avg.astype(np.float32))
        vol_pred[cid] = torch.from_numpy(vp_avg.astype(np.float32))
        # Targets should be identical across files; take from the first.
        surf_y[cid] = torch.from_numpy(loaded[indices[0]][split]["surf_y"][cid])
        vol_y[cid] = torch.from_numpy(loaded[indices[0]][split]["vol_y"][cid])
    return surf_pred, vol_pred, surf_y, vol_y


def _per_case_stats(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for cid in sorted(preds.keys()):
        out[cid] = _channel_stats(preds[cid], targets[cid])
    return out


def _global_stats(per_case: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(per_case.values()), dim=0).sum(dim=0)


def evaluate_arm(
    arm_name: str,
    indices: list[int],
    loaded: list[dict],
) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {split: {recipe: metrics_dict}} where recipe in {raw, cal_fitted, cal_hardcoded}."""
    out: dict[str, dict[str, dict[str, float]]] = {}

    # Aggregate per-case stats for both splits first; fit α/β on val.
    per_case_by_split: dict[str, tuple[dict, dict]] = {}
    for split in ("val_surface", "test_surface"):
        sp, vp, sy, vy = _avg_predictions(loaded, indices, split)
        surf_stats = _per_case_stats(sp, sy)
        vol_stats = _per_case_stats(vp, vy)
        per_case_by_split[split] = (surf_stats, vol_stats)

    val_surf_global = _global_stats(per_case_by_split["val_surface"][0])
    val_vol_global = _global_stats(per_case_by_split["val_surface"][1])
    alpha_surf_fit, beta_surf_fit = _fit_affine(val_surf_global)
    alpha_vol_fit, beta_vol_fit = _fit_affine(val_vol_global)

    print(f"\n=== Arm {arm_name} (ckpts={indices}) calibration coefs (fit on val) ===")
    for c, name in enumerate(SURF_CHANNELS):
        print(
            f"  surface[{name:6s}]  alpha_fit={alpha_surf_fit[c].item():+.6f} "
            f"beta_fit={beta_surf_fit[c].item():+.6f}"
        )
    print(
        f"  volume[volume_pressure]  alpha_fit={alpha_vol_fit[0].item():+.6f} "
        f"beta_fit={beta_vol_fit[0].item():+.6f}"
    )

    for split, (surf_per_case, vol_per_case) in per_case_by_split.items():
        out.setdefault(split, {})
        out[split]["raw"] = _compute_metrics_from_per_case_stats(
            surf_per_case, vol_per_case
        )
        out[split]["cal_fitted"] = _compute_metrics_from_per_case_stats(
            surf_per_case, vol_per_case,
            alpha_surf=alpha_surf_fit, beta_surf=beta_surf_fit,
            alpha_vol=alpha_vol_fit, beta_vol=beta_vol_fit,
        )
        out[split]["cal_hardcoded"] = _compute_metrics_from_per_case_stats(
            surf_per_case, vol_per_case,
            alpha_surf=H312_CAL["alpha_surf"], beta_surf=H312_CAL["beta_surf"],
            alpha_vol=H312_CAL["alpha_vol"], beta_vol=H312_CAL["beta_vol"],
        )
    return out


def parse_arms(spec: str) -> list[tuple[str, list[int]]]:
    """'A:2 B:1,2 C:0,2 D:0,1,2' -> [('A',[2]), ('B',[1,2]), ...]."""
    arms: list[tuple[str, list[int]]] = []
    for tok in spec.split():
        name, idx_spec = tok.split(":")
        indices = [int(x) for x in idx_spec.split(",") if x.strip()]
        arms.append((name.strip(), indices))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs", nargs="+", required=True,
        help="Per-checkpoint .npz files (order matters; refer to by index in --arms)",
    )
    ap.add_argument(
        "--arms", default="",
        help='Space-separated "name:idx1,idx2,..." (defaults: all single + all pairwise + all-N)',
    )
    ap.add_argument(
        "--out", default="",
        help="Optional .json path to write the full results dict.",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    print(f"Loading {len(paths)} checkpoint files:")
    loaded = []
    for i, p in enumerate(paths):
        d = _load_one(p)
        meta = d["metadata"]
        ckpt = meta.get("checkpoint", "?")
        ep = meta.get("checkpoint_epoch", "?")
        print(f"  [{i}] {p.name}  checkpoint={ckpt} epoch={ep}")
        loaded.append(d)

    if args.arms:
        arms = parse_arms(args.arms)
    else:
        arms = []
        N = len(paths)
        # Single-checkpoint controls
        for i in range(N):
            arms.append((f"single_{i}", [i]))
        # All adjacent / 2-cp pairs
        for i in range(N):
            for j in range(i + 1, N):
                arms.append((f"pair_{i}_{j}", [i, j]))
        # All-N
        if N >= 3:
            arms.append((f"all_{N}", list(range(N))))

    results: dict[str, dict] = {}
    for name, indices in arms:
        if max(indices) >= len(paths):
            raise ValueError(f"Arm {name} indices out of range: {indices}")
        print(f"\n{'=' * 60}\nArm {name}: ckpts={indices}\n{'=' * 60}")
        arm_metrics = evaluate_arm(name, indices, loaded)
        results[name] = {
            "indices": indices,
            "metrics": arm_metrics,
        }

    # Pretty-print summary table
    print(f"\n\n{'=' * 100}")
    print("H342 multi-checkpoint output-averaging summary")
    print(f"{'=' * 100}")
    keys_to_show = (
        "abupt_axis_mean_rel_l2_pct",
        "surface_pressure_rel_l2_pct",
        "wall_shear_rel_l2_pct",
        "wall_shear_x_rel_l2_pct",
        "wall_shear_y_rel_l2_pct",
        "wall_shear_z_rel_l2_pct",
        "volume_pressure_rel_l2_pct",
    )
    for split in ("val_surface", "test_surface"):
        print(f"\n[{split}]")
        for recipe in ("raw", "cal_fitted", "cal_hardcoded"):
            print(f"\n  -- {recipe} --")
            header = f"  {'metric':<36s} " + " ".join(
                f"{name:>15s}" for name, _ in arms
            )
            print(header)
            for k in keys_to_show:
                row = f"  {k:<36s} " + " ".join(
                    f"{results[name]['metrics'][split][recipe][k]:>15.4f}"
                    for name, _ in arms
                )
                print(row)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()
