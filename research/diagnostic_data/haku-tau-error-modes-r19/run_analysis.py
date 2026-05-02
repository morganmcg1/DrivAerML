"""Analysis pipeline for the haku tau_y/z error-mode diagnostic (PR #363).

Consumes the .npz prediction dumps written by run_inference.py and produces
the five-section deliverable described in the PR body. Saves both a CSV-style
summary table and matplotlib PNGs.

Sections:
1. Per-axis error magnitude by surface region (signed-bbox-relative bins).
2. Spatial autocorrelation of error (Moran's I with kNN-16 graph).
3. Error vs surface curvature (PCA-eigenvalue based curvature deciles).
4. Error vs flow alignment (angle to global-x axis on tau_target deciles).
5. Per-case ranking of tau_y_err.

Outputs are written into the same directory as the predictions, plus a
stats.json with the key numbers used to build the report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scipy.spatial import cKDTree
except ImportError as e:  # pragma: no cover - scipy is in pyproject.toml
    raise SystemExit(f"scipy is required: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--moran-points", type=int, default=5000)
    p.add_argument("--curvature-knn", type=int, default=32)
    p.add_argument("--moran-knn", type=int, default=16)
    p.add_argument("--moran-permutations", type=int, default=199)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_predictions(pred_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    cases: dict[str, dict[str, np.ndarray]] = {}
    for npz_path in sorted(pred_dir.glob("*.npz")):
        case_id = npz_path.stem
        data = np.load(npz_path)
        cases[case_id] = {k: data[k] for k in data.files}
    return cases


def signed_bbox(xyz: np.ndarray) -> np.ndarray:
    """Return per-point coords in [-0.5, +0.5] using the case bbox."""
    lo = xyz.min(axis=0, keepdims=True)
    hi = xyz.max(axis=0, keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    return (xyz - lo) / span - 0.5


def assign_region(rel_xyz: np.ndarray) -> np.ndarray:
    """Tag each point with a (x_bin, y_bin, z_bin) string.

    x: front (-0.5..-0.166), mid (-0.166..0.166), rear (0.166..0.5)
    y: center (|y|<0.25), side (|y|>=0.25)
    z: underbody (z<-0.166), body (-0.166<=z<0.166), roof (z>=0.166)
    """
    x = rel_xyz[:, 0]
    y_abs = np.abs(rel_xyz[:, 1])
    z = rel_xyz[:, 2]

    x_bin = np.where(x < -1.0 / 6, "front", np.where(x < 1.0 / 6, "mid", "rear"))
    y_bin = np.where(y_abs < 0.25, "center", "side")
    z_bin = np.where(z < -1.0 / 6, "underbody", np.where(z < 1.0 / 6, "body", "roof"))
    region = np.char.add(np.char.add(x_bin.astype("U10"), "-"), y_bin.astype("U6"))
    region = np.char.add(np.char.add(region, "-"), z_bin.astype("U10"))
    return region


def per_axis_abs_error(tau_pred: np.ndarray, tau_target: np.ndarray) -> np.ndarray:
    return np.abs(tau_pred - tau_target)


def compute_curvature(xyz: np.ndarray, k: int) -> np.ndarray:
    """Approximate scalar curvature: smallest PCA eigenvalue divided by eigenvalue sum."""
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    idx = idx[:, 1:]  # drop self
    neigh = xyz[idx]  # [N, k, 3]
    centred = neigh - neigh.mean(axis=1, keepdims=True)
    cov = centred.transpose(0, 2, 1) @ centred / float(k)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum(axis=1) + 1e-12
    curv = eigvals[:, 0] / total
    return curv  # in [0, 1/3]


def morans_i(values: np.ndarray, knn_idx: np.ndarray) -> float:
    """Row-standardized Moran's I for kNN graph weights."""
    n = values.shape[0]
    mean = values.mean()
    dev = values - mean
    dev2_sum = float(np.sum(dev**2))
    if dev2_sum < 1e-12:
        return 0.0

    # Row-normalized weights: each point gets 1/k weight to each neighbour
    k = knn_idx.shape[1]
    neigh_dev = dev[knn_idx]  # [N, k]
    cross_sum = float(np.sum(dev[:, None] * neigh_dev) / k)
    # For row-standardized weights, sum_W = N
    return (n / float(n)) * (cross_sum / dev2_sum)


def morans_i_permutation_p(values: np.ndarray, knn_idx: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    observed = morans_i(values, knn_idx)
    # Permutation null: shuffle values among locations
    null_dist = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        permuted = rng.permutation(values)
        null_dist[i] = morans_i(permuted, knn_idx)
    # Two-sided p-value
    extreme = np.sum(np.abs(null_dist) >= np.abs(observed))
    p_value = (extreme + 1.0) / (n_perm + 1.0)
    return observed, p_value


def angle_to_x(target: np.ndarray) -> np.ndarray:
    """Angle (rad) between target tau-vector and global +x.
    Returns [0, pi] where 0 = streamwise, pi/2 = pure cross-flow."""
    norms = np.linalg.norm(target, axis=1)
    safe = np.maximum(norms, 1e-12)
    cos = target[:, 0] / safe
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(np.abs(cos))  # use abs(cos) to fold the back-flow into 0 too


# ---------------------------------------------------------------------------
# Section 1 — per-axis error by region
# ---------------------------------------------------------------------------


def section1_region(cases: dict, out_dir: Path) -> dict:
    region_sums = {axis: {} for axis in ("x", "y", "z")}
    region_counts = {}
    region_coords_sum = {}
    n_global = {axis: 0 for axis in ("x", "y", "z")}
    sum_global = {axis: 0.0 for axis in ("x", "y", "z")}

    for case_id, c in cases.items():
        rel = signed_bbox(c["surface_xyz"])
        region = assign_region(rel)
        err = per_axis_abs_error(c["tau_pred"], c["tau_target"])
        for ax, axis_idx in enumerate(("x", "y", "z")):
            n_global[axis_idx] += err.shape[0]
            sum_global[axis_idx] += float(err[:, ax].sum())
        for r in np.unique(region):
            mask = region == r
            n = int(mask.sum())
            region_counts[r] = region_counts.get(r, 0) + n
            for ax_idx, axis_name in enumerate(("x", "y", "z")):
                region_sums[axis_name][r] = region_sums[axis_name].get(r, 0.0) + float(err[mask, ax_idx].sum())

    # Build a sorted summary table
    region_keys = sorted(region_counts.keys())
    rows = []
    global_means = {ax: sum_global[ax] / max(n_global[ax], 1) for ax in ("x", "y", "z")}
    for r in region_keys:
        n = region_counts[r]
        if n == 0:
            continue
        means = {ax: region_sums[ax][r] / n for ax in ("x", "y", "z")}
        ratios = {ax: means[ax] / max(global_means[ax], 1e-12) for ax in ("x", "y", "z")}
        rows.append({
            "region": r,
            "n_points": n,
            "mean_abs_err_x": means["x"],
            "mean_abs_err_y": means["y"],
            "mean_abs_err_z": means["z"],
            "ratio_x_to_global": ratios["x"],
            "ratio_y_to_global": ratios["y"],
            "ratio_z_to_global": ratios["z"],
        })

    rows.sort(key=lambda r: r["mean_abs_err_y"], reverse=True)

    # Find top-3 regions by tau_y or tau_z >= 2x global mean
    hot_y = [r for r in rows if r["ratio_y_to_global"] >= 2.0]
    hot_z = [r for r in rows if r["ratio_z_to_global"] >= 2.0]

    # Save table
    csv_path = out_dir / "section1_region_table.csv"
    with csv_path.open("w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    # Heatmap-ish bar chart: top 10 regions by tau_y err
    top = rows[:10]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(top))
    width = 0.27
    ax.bar(x - width, [r["mean_abs_err_x"] for r in top], width, label="tau_x")
    ax.bar(x, [r["mean_abs_err_y"] for r in top], width, label="tau_y")
    ax.bar(x + width, [r["mean_abs_err_z"] for r in top], width, label="tau_z")
    ax.set_xticks(x)
    ax.set_xticklabels([r["region"] for r in top], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean abs error (Pa)")
    ax.set_title("Section 1: top-10 regions sorted by tau_y mean abs error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "section1_region_top10.png", dpi=120)
    plt.close(fig)

    return {
        "global_mean_abs_err": global_means,
        "n_regions": len(rows),
        "top3_y_2x_global": [r["region"] for r in hot_y[:3]],
        "top3_z_2x_global": [r["region"] for r in hot_z[:3]],
        "top10_by_tau_y_err": rows[:10],
    }


# ---------------------------------------------------------------------------
# Section 2 — spatial autocorrelation
# ---------------------------------------------------------------------------


def section2_morans_i(cases: dict, out_dir: Path, n_subsample: int, knn: int, n_perm: int, rng: np.random.Generator) -> dict:
    per_case = {}
    for case_id, c in sorted(cases.items()):
        n = c["surface_xyz"].shape[0]
        idx = rng.choice(n, size=min(n_subsample, n), replace=False)
        xyz = c["surface_xyz"][idx]
        err = np.abs(c["tau_pred"] - c["tau_target"])  # [N, 3]
        err = err[idx]

        tree = cKDTree(xyz)
        _, knn_idx = tree.query(xyz, k=knn + 1)
        knn_idx = knn_idx[:, 1:]

        result = {}
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            i_obs, p_val = morans_i_permutation_p(err[:, axis_idx], knn_idx, n_perm=n_perm, rng=rng)
            result[axis_name] = {"I": float(i_obs), "p_perm": float(p_val)}
        per_case[case_id] = result

    # Aggregate: median I across cases per axis
    summary = {}
    for axis_name in ("x", "y", "z"):
        all_i = np.array([per_case[c][axis_name]["I"] for c in per_case])
        all_p = np.array([per_case[c][axis_name]["p_perm"] for c in per_case])
        summary[axis_name] = {
            "median_I": float(np.median(all_i)),
            "mean_I": float(np.mean(all_i)),
            "min_I": float(np.min(all_i)),
            "max_I": float(np.max(all_i)),
            "frac_p_lt_0.05": float(np.mean(all_p < 0.05)),
            "n_cases": int(len(all_i)),
        }

    # Save bar chart of median I
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    axes = ["x", "y", "z"]
    medians = [summary[a]["median_I"] for a in axes]
    ax.bar(axes, medians, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("median Moran's I (kNN=16)")
    ax.set_title(f"Section 2: spatial autocorrelation of |tau_err|\nover {len(per_case)} cases ({n_subsample} pts each)")
    ax.axhline(0.0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "section2_morans_i.png", dpi=120)
    plt.close(fig)

    return {"per_axis_summary": summary, "per_case_full": per_case}


# ---------------------------------------------------------------------------
# Section 3 — curvature
# ---------------------------------------------------------------------------


def section3_curvature(cases: dict, out_dir: Path, knn: int) -> dict:
    all_curv = []
    all_err = []
    for case_id, c in sorted(cases.items()):
        xyz = c["surface_xyz"]
        # Cap to 30k points per case for speed; cKDTree at 65k * 32 is also fine but conservative.
        if xyz.shape[0] > 30_000:
            rng = np.random.default_rng(0)
            idx = rng.choice(xyz.shape[0], size=30_000, replace=False)
            xyz_sub = xyz[idx]
            err = np.abs(c["tau_pred"][idx] - c["tau_target"][idx])
        else:
            xyz_sub = xyz
            err = np.abs(c["tau_pred"] - c["tau_target"])
        curv = compute_curvature(xyz_sub, k=knn)
        all_curv.append(curv)
        all_err.append(err)
    curv = np.concatenate(all_curv, axis=0)
    err = np.concatenate(all_err, axis=0)

    deciles = np.quantile(curv, np.linspace(0.0, 1.0, 11))
    deciles = deciles.astype(np.float64)
    bins = np.searchsorted(deciles, curv, side="right") - 1
    bins = np.clip(bins, 0, 9)

    rows = []
    for d in range(10):
        mask = bins == d
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "decile": d,
            "n_points": n,
            "curvature_p50": float(np.median(curv[mask])),
            "mean_abs_err_x": float(err[mask, 0].mean()),
            "mean_abs_err_y": float(err[mask, 1].mean()),
            "mean_abs_err_z": float(err[mask, 2].mean()),
        })

    csv_path = out_dir / "section3_curvature_table.csv"
    with csv_path.open("w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    deciles_x = [r["decile"] for r in rows]
    ax.plot(deciles_x, [r["mean_abs_err_x"] for r in rows], "-o", label="tau_x")
    ax.plot(deciles_x, [r["mean_abs_err_y"] for r in rows], "-o", label="tau_y")
    ax.plot(deciles_x, [r["mean_abs_err_z"] for r in rows], "-o", label="tau_z")
    ax.set_xlabel("curvature decile (0=flat, 9=high curvature)")
    ax.set_ylabel("mean abs error (Pa)")
    ax.set_title("Section 3: error vs surface curvature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "section3_curvature.png", dpi=120)
    plt.close(fig)

    # Compare deciles 0-2 (low) vs 7-9 (high) for tau_y
    low = [r for r in rows if r["decile"] <= 2]
    high = [r for r in rows if r["decile"] >= 7]
    low_mean_y = np.mean([r["mean_abs_err_y"] for r in low])
    high_mean_y = np.mean([r["mean_abs_err_y"] for r in high])
    low_mean_z = np.mean([r["mean_abs_err_z"] for r in low])
    high_mean_z = np.mean([r["mean_abs_err_z"] for r in high])
    return {
        "table": rows,
        "tau_y_low_mean": float(low_mean_y),
        "tau_y_high_mean": float(high_mean_y),
        "tau_y_high_to_low_ratio": float(high_mean_y / max(low_mean_y, 1e-12)),
        "tau_z_low_mean": float(low_mean_z),
        "tau_z_high_mean": float(high_mean_z),
        "tau_z_high_to_low_ratio": float(high_mean_z / max(low_mean_z, 1e-12)),
    }


# ---------------------------------------------------------------------------
# Section 4 — flow alignment
# ---------------------------------------------------------------------------


def section4_alignment(cases: dict, out_dir: Path) -> dict:
    all_theta = []
    all_err = []
    all_target_norm = []
    for case_id, c in sorted(cases.items()):
        target = c["tau_target"]
        theta = angle_to_x(target)
        err = np.abs(c["tau_pred"] - target)
        all_theta.append(theta)
        all_err.append(err)
        all_target_norm.append(np.linalg.norm(target, axis=1))

    theta = np.concatenate(all_theta, axis=0)
    err = np.concatenate(all_err, axis=0)
    target_norm = np.concatenate(all_target_norm, axis=0)

    # Filter near-zero target points (those add noise)
    valid = target_norm > 0.05  # Pa-ish — see normalizer scale, conservative
    theta = theta[valid]
    err = err[valid]
    target_norm = target_norm[valid]

    deciles = np.quantile(theta, np.linspace(0.0, 1.0, 11))
    bins = np.searchsorted(deciles, theta, side="right") - 1
    bins = np.clip(bins, 0, 9)

    rows = []
    for d in range(10):
        mask = bins == d
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "decile": d,
            "n_points": n,
            "theta_p50_rad": float(np.median(theta[mask])),
            "theta_p50_deg": float(np.degrees(np.median(theta[mask]))),
            "target_norm_p50": float(np.median(target_norm[mask])),
            "mean_abs_err_x": float(err[mask, 0].mean()),
            "mean_abs_err_y": float(err[mask, 1].mean()),
            "mean_abs_err_z": float(err[mask, 2].mean()),
        })

    csv_path = out_dir / "section4_alignment_table.csv"
    with csv_path.open("w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    deciles_x = [r["theta_p50_deg"] for r in rows]
    ax.plot(deciles_x, [r["mean_abs_err_x"] for r in rows], "-o", label="tau_x")
    ax.plot(deciles_x, [r["mean_abs_err_y"] for r in rows], "-o", label="tau_y")
    ax.plot(deciles_x, [r["mean_abs_err_z"] for r in rows], "-o", label="tau_z")
    ax.set_xlabel("median theta (deg) — angle between |tau_target| and global x")
    ax.set_ylabel("mean abs error (Pa)")
    ax.set_title("Section 4: error vs flow alignment angle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "section4_alignment.png", dpi=120)
    plt.close(fig)

    # Compare alignment ratio: stream-aligned (theta < 30 deg) vs cross-flow (theta > 60 deg)
    sa_mask = theta < np.radians(30)
    cf_mask = theta > np.radians(60)
    summary_block = {
        "stream_aligned_mean_y_err": float(err[sa_mask, 1].mean()) if sa_mask.any() else None,
        "cross_flow_mean_y_err": float(err[cf_mask, 1].mean()) if cf_mask.any() else None,
        "stream_aligned_mean_z_err": float(err[sa_mask, 2].mean()) if sa_mask.any() else None,
        "cross_flow_mean_z_err": float(err[cf_mask, 2].mean()) if cf_mask.any() else None,
    }
    return {"table": rows, "summary": summary_block}


# ---------------------------------------------------------------------------
# Section 5 — per-case ranking
# ---------------------------------------------------------------------------


def section5_per_case(cases: dict, out_dir: Path) -> dict:
    rows = []
    for case_id, c in sorted(cases.items()):
        target = c["tau_target"]
        pred = c["tau_pred"]
        diff = pred - target

        def axis_rel_l2(axis_idx: int) -> float:
            num = float(np.sum(diff[:, axis_idx] ** 2))
            den = float(np.sum(target[:, axis_idx] ** 2))
            return 100.0 * np.sqrt(num / max(den, 1e-12))

        rows.append({
            "case_id": case_id,
            "n_points": int(target.shape[0]),
            "tau_x_rel_l2_pct": axis_rel_l2(0),
            "tau_y_rel_l2_pct": axis_rel_l2(1),
            "tau_z_rel_l2_pct": axis_rel_l2(2),
            "tau_y_mean_abs_err": float(np.abs(diff[:, 1]).mean()),
            "tau_z_mean_abs_err": float(np.abs(diff[:, 2]).mean()),
        })

    rows_y_sorted = sorted(rows, key=lambda r: r["tau_y_rel_l2_pct"], reverse=True)
    top3_y = rows_y_sorted[:3]
    bot3_y = rows_y_sorted[-3:]
    rows_z_sorted = sorted(rows, key=lambda r: r["tau_z_rel_l2_pct"], reverse=True)
    top3_z = rows_z_sorted[:3]
    bot3_z = rows_z_sorted[-3:]

    csv_path = out_dir / "section5_per_case_table.csv"
    with csv_path.open("w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    case_ids = [r["case_id"] for r in rows_y_sorted]
    y_vals = [r["tau_y_rel_l2_pct"] for r in rows_y_sorted]
    z_vals = [r["tau_z_rel_l2_pct"] for r in rows_y_sorted]
    x_idx = np.arange(len(rows_y_sorted))
    ax.bar(x_idx - 0.18, y_vals, 0.36, label="tau_y rel-L2 %")
    ax.bar(x_idx + 0.18, z_vals, 0.36, label="tau_z rel-L2 %")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(case_ids, rotation=90, fontsize=6)
    ax.set_ylabel("per-case rel-L2 (%)")
    ax.set_title("Section 5: per-case tau_y/z rel-L2 (sorted by tau_y desc)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "section5_per_case.png", dpi=120)
    plt.close(fig)

    return {
        "top3_tau_y": [{"case_id": r["case_id"], "tau_y_rel_l2_pct": r["tau_y_rel_l2_pct"]} for r in top3_y],
        "bot3_tau_y": [{"case_id": r["case_id"], "tau_y_rel_l2_pct": r["tau_y_rel_l2_pct"]} for r in bot3_y],
        "top3_tau_z": [{"case_id": r["case_id"], "tau_z_rel_l2_pct": r["tau_z_rel_l2_pct"]} for r in top3_z],
        "bot3_tau_z": [{"case_id": r["case_id"], "tau_z_rel_l2_pct": r["tau_z_rel_l2_pct"]} for r in bot3_z],
        "all_rows": rows,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading predictions...")
    cases = load_predictions(args.predictions_dir)
    print(f"  {len(cases)} cases, total {sum(c['surface_xyz'].shape[0] for c in cases.values())} points")

    print("Section 1: per-axis error by region...")
    s1 = section1_region(cases, args.out_dir)
    print("Section 2: spatial autocorrelation (Moran's I)...")
    s2 = section2_morans_i(
        cases,
        args.out_dir,
        n_subsample=args.moran_points,
        knn=args.moran_knn,
        n_perm=args.moran_permutations,
        rng=rng,
    )
    print("Section 3: error vs curvature...")
    s3 = section3_curvature(cases, args.out_dir, knn=args.curvature_knn)
    print("Section 4: error vs flow alignment...")
    s4 = section4_alignment(cases, args.out_dir)
    print("Section 5: per-case ranking...")
    s5 = section5_per_case(cases, args.out_dir)

    summary = {
        "n_cases": len(cases),
        "section1": s1,
        "section2": s2["per_axis_summary"],
        "section3": s3,
        "section4": s4["summary"],
        "section5": {k: v for k, v in s5.items() if k != "all_rows"},
    }
    with (args.out_dir / "stats.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    # Also save per-case stats for downstream use
    with (args.out_dir / "per_case_stats.json").open("w") as f:
        json.dump({"section5_rows": s5["all_rows"], "section2_per_case": s2["per_case_full"]}, f, indent=2, default=str)

    print("Done. Wrote stats.json and figures to", args.out_dir)


if __name__ == "__main__":
    main()
