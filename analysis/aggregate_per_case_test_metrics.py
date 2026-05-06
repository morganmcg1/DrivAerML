# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Aggregate per-case + per-region test diagnostics into Tables A-E.

Reads the .npz files produced by ``run_per_case_test_eval.py`` and prints
markdown tables that match the format requested in PR #767.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import numpy as np


def rel_l2_pct(err_sq: float, tgt_sq: float) -> float:
    if not math.isfinite(err_sq) or not math.isfinite(tgt_sq) or tgt_sq <= 0:
        return float("nan")
    return 100.0 * math.sqrt(err_sq / tgt_sq)


def _aggregate_rel_l2_pct(error_sq: np.ndarray, target_sq: np.ndarray) -> float:
    """Per-case rel_l2 then mean over cases (matches trainer_runtime.finalize)."""
    values = []
    for e, t in zip(error_sq, target_sq):
        if math.isfinite(e) and math.isfinite(t) and t > 0:
            values.append(100.0 * math.sqrt(e / t))
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _per_case_rel_l2_pct(case_table: np.ndarray, metric_idx: int) -> np.ndarray:
    err_sq = case_table[:, metric_idx, 0]
    tgt_sq = case_table[:, metric_idx, 1]
    return np.array([rel_l2_pct(e, t) for e, t in zip(err_sq, tgt_sq)])


def render_checkpoint_tables(npz_path: Path) -> str:
    z = np.load(npz_path, allow_pickle=False)
    run_id = str(z["run_id"])
    project = str(z["project"])
    case_ids: list[str] = list(z["case_ids"].tolist())
    metric_names: list[str] = list(z["metric_names"].tolist())
    case_table = z["case_table"]  # [case, metric, (err_sq, tgt_sq, count)]
    region_names: list[str] = list(z["region_names"].tolist())
    region_table = z["region_table"]
    sdf_band_names: list[str] = list(z["sdf_band_names"].tolist())
    sdf_table = z["sdf_table"]
    region_total_points = z["region_total_points"]
    sdf_total_points = z["sdf_total_points"]
    total_volume_points = int(z["total_volume_points"].item())

    midx = {m: i for i, m in enumerate(metric_names)}
    vp_per_case = _per_case_rel_l2_pct(case_table, midx["volume_pressure"])
    sp_per_case = _per_case_rel_l2_pct(case_table, midx["surface_pressure"])
    ws_per_case = _per_case_rel_l2_pct(case_table, midx["wall_shear"])
    tx_per_case = _per_case_rel_l2_pct(case_table, midx["wall_shear_x"])
    ty_per_case = _per_case_rel_l2_pct(case_table, midx["wall_shear_y"])
    tz_per_case = _per_case_rel_l2_pct(case_table, midx["wall_shear_z"])

    sort_idx = np.argsort(-np.where(np.isnan(vp_per_case), -np.inf, vp_per_case))

    out: list[str] = []
    out.append(f"## Diagnostic — `{run_id}` (project `{project}`)\n")

    # Aggregate (paper-facing test_primary equivalents).
    out.append("### Aggregate test metrics (per-case rel_L2 then case-mean)\n")
    out.append("| Metric | This-repo key | Mean test rel_L2 (%) |")
    out.append("|---|---|---:|")
    for label, key in (
        ("Surface pressure p_s", "surface_pressure"),
        ("Vector wall-shear", "wall_shear"),
        ("Volume pressure p_v", "volume_pressure"),
        ("tau_x", "wall_shear_x"),
        ("tau_y", "wall_shear_y"),
        ("tau_z", "wall_shear_z"),
    ):
        col = _per_case_rel_l2_pct(case_table, midx[key])
        finite = col[np.isfinite(col)]
        mean = float(finite.mean()) if finite.size else float("nan")
        out.append(f"| {label} | test_primary/{key}_rel_l2_pct | {mean:.4f} |")
    abupt_axis = (
        _per_case_rel_l2_pct(case_table, midx["surface_pressure"]),
        _per_case_rel_l2_pct(case_table, midx["wall_shear_x"]),
        _per_case_rel_l2_pct(case_table, midx["wall_shear_y"]),
        _per_case_rel_l2_pct(case_table, midx["wall_shear_z"]),
        _per_case_rel_l2_pct(case_table, midx["volume_pressure"]),
    )
    abupt_means = []
    for col in abupt_axis:
        finite = col[np.isfinite(col)]
        abupt_means.append(float(finite.mean()) if finite.size else float("nan"))
    finite_means = [m for m in abupt_means if math.isfinite(m)]
    abupt = sum(finite_means) / len(finite_means) if finite_means else float("nan")
    out.append(f"| **abupt axis-mean** | test_primary/abupt_axis_mean_rel_l2_pct | **{abupt:.4f}** |")
    out.append("")

    # ---- Table A — per-case test errors, ranked by test_vol_p worst→best ----
    out.append(
        "### Table A — Per-case test errors (ranked by test_vol_p worst → best)\n"
    )
    out.append(
        "| case_id | vol_pressure_rel_l2 | surface_p | wall_shear | tau_x | tau_y | tau_z |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for ci in sort_idx:
        case = case_ids[ci]
        out.append(
            f"| {case} | {vp_per_case[ci]:.3f} | {sp_per_case[ci]:.3f} | {ws_per_case[ci]:.3f} | "
            f"{tx_per_case[ci]:.3f} | {ty_per_case[ci]:.3f} | {tz_per_case[ci]:.3f} |"
        )
    out.append("")

    # ---- Table B — Aggregate stats with/without top outliers ----
    def _stats_excluding_top(k: int) -> tuple[float, float, float]:
        # Per-case rel_l2 already computed; drop the k worst, then compute mean/median/std.
        kept = vp_per_case.copy()
        finite = kept[np.isfinite(kept)]
        if k > 0:
            order = np.argsort(-np.where(np.isnan(kept), -np.inf, kept))
            drop = set(order[:k].tolist())
            mask = np.array([i not in drop for i in range(len(kept))])
            finite = kept[mask]
            finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            float(finite.mean()),
            float(np.median(finite)),
            float(finite.std(ddof=0)),
        )

    out.append("### Table B — Aggregate volume_pressure rel_L2 stats with/without worst-cases dropped\n")
    out.append("| Stat | All cases | Excluding top-4 worst | Excluding top-8 worst |")
    out.append("|---|---:|---:|---:|")
    a_mean, a_med, a_std = _stats_excluding_top(0)
    b4_mean, b4_med, b4_std = _stats_excluding_top(4)
    b8_mean, b8_med, b8_std = _stats_excluding_top(8)
    out.append(f"| mean test_vol_p (%) | {a_mean:.3f} | {b4_mean:.3f} | {b8_mean:.3f} |")
    out.append(f"| median test_vol_p (%) | {a_med:.3f} | {b4_med:.3f} | {b8_med:.3f} |")
    out.append(f"| std test_vol_p (%) | {a_std:.3f} | {b4_std:.3f} | {b8_std:.3f} |")
    out.append("")

    # ---- Table C — Per-region test_vol_p (point-weighted means) ----
    region_label_lookup = {
        "upstream": "upstream (x_rel ≤ 0.5)",
        "near_wake": "near-wake (0.5 < x_rel < 3.0, |z_rel| < 1.5)",
        "far_wake": "far-wake (x_rel ≥ 3.0)",
        "under_body": "under-body (z_rel < 0)",
        "roof": "roof (z_rel ≥ 1.0)",
    }
    out.append("### Table C — Per-region test_vol_p (point-weighted L2 over all valid points)\n")
    out.append("| Region | Point share | Mean test_vol_p (%) |")
    out.append("|---|---:|---:|")
    for ri, region in enumerate(region_names):
        err_sq = float(np.nansum(region_table[ri, :, 0]))
        tgt_sq = float(np.nansum(region_table[ri, :, 1]))
        pts = int(region_total_points[ri])
        share = pts / max(total_volume_points, 1)
        rl2 = rel_l2_pct(err_sq, tgt_sq)
        label = region_label_lookup.get(region, region)
        out.append(f"| {label} | {share:.2%} | {rl2:.4f} |")
    out.append("")

    # ---- Table D — Per-region SDF-band volume_pressure rel_L2 ----
    sdf_label_lookup = {
        "near_surface": "near-surface (|sdf| ≤ 0.05m)",
        "boundary_layer": "boundary layer (0.05 < |sdf| ≤ 0.5m)",
        "far_field": "far-field (|sdf| > 0.5m)",
    }
    out.append("### Table D — Per-SDF-band test_vol_p (point-weighted L2 over all valid points)\n")
    out.append("| SDF band | Point share | Mean test_vol_p (%) |")
    out.append("|---|---:|---:|")
    for bi, band in enumerate(sdf_band_names):
        err_sq = float(np.nansum(sdf_table[bi, :, 0]))
        tgt_sq = float(np.nansum(sdf_table[bi, :, 1]))
        pts = int(sdf_total_points[bi])
        share = pts / max(total_volume_points, 1)
        rl2 = rel_l2_pct(err_sq, tgt_sq)
        label = sdf_label_lookup.get(band, band)
        out.append(f"| {label} | {share:.2%} | {rl2:.4f} |")
    out.append("")

    # ---- Table E — val→test ratio for the worst-10 cases ----
    # We do not have val matched cases (val/test are disjoint); we report test-only
    # values for the worst-10 and the per-region rel_L2 within those cases.
    out.append("### Table E — Worst-10 cases — composition vs. dataset means\n")
    out.append(
        "| case_id | test_vol_p (%) | upstream rel_L2 | near rel_L2 | far rel_L2 | under_body rel_L2 | roof rel_L2 | n_volume_points |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ci in sort_idx[:10]:
        case = case_ids[ci]
        vp = vp_per_case[ci]
        # Total volume points for this case (sum across all regions; some pts in multiple regions).
        n_pts = int(case_table[ci, midx["volume_pressure"], 2])
        cells = [f"| {case} | {vp:.3f}"]
        for ri, region in enumerate(region_names):
            err_sq = region_table[ri, ci, 0]
            tgt_sq = region_table[ri, ci, 1]
            rl2 = rel_l2_pct(err_sq, tgt_sq)
            if math.isfinite(rl2):
                cells.append(f"{rl2:.3f}")
            else:
                cells.append("—")
        cells.append(str(n_pts))
        out.append(" | ".join(cells) + " |")
    out.append("")

    # Variance contribution analysis.
    out.append("### Variance / contribution diagnostics for the worst-cases\n")
    finite_vp = vp_per_case[np.isfinite(vp_per_case)]
    overall_mean = float(finite_vp.mean()) if finite_vp.size else float("nan")
    overall_var = float(finite_vp.var(ddof=0)) if finite_vp.size else float("nan")
    if finite_vp.size:
        order = np.argsort(-np.where(np.isnan(vp_per_case), -np.inf, vp_per_case))
        squared_devs = (vp_per_case - overall_mean) ** 2
        for k in (4, 8):
            top = order[:k]
            contrib = float(np.nansum(squared_devs[top]))
            denom = float(np.nansum(squared_devs))
            pct = 100.0 * contrib / denom if denom > 0 else float("nan")
            out.append(
                f"- Top-{k} worst cases account for **{pct:.1f}%** of the squared-deviation "
                f"sum around the overall test_vol_p mean ({overall_mean:.3f}%, σ²={overall_var:.3f})."
            )
    return "\n".join(out)


def render_combined(npz_paths: list[Path]) -> str:
    sections = [render_checkpoint_tables(p) for p in npz_paths]
    return "\n\n---\n\n".join(sections)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Aggregate per-case + per-region diagnostics")
    p.add_argument("npz_paths", nargs="+", type=Path, help="Output .npz files from run_per_case_test_eval.py")
    p.add_argument("--out", type=Path, default=None, help="Optional markdown output path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    md = render_combined(args.npz_paths)
    print(md)
    if args.out is not None:
        args.out.write_text(md)
        print(f"\nWrote markdown to {args.out}")


if __name__ == "__main__":
    main()
