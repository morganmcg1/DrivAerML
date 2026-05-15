# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Pre-training diagnostic for the spatial-prior surface sampler (PR #1120).

For each case in the 7-case sample used for the PR #1113 diagnostic, this
script computes:

- ``pearson(spatial_score, |WSS|)`` where ``spatial_score`` is the
  pre-alpha combined bias ``(front_bias + ground_bias) / 2`` so the
  ranking is independent of alpha. This is the green-light check the
  advisor asked for (ρ ≥ +0.20).
- ``pearson(spatial_score, |Cp|)`` for context.
- Per-decile bin-occupancy of the spatial-prior weights with the chosen
  alpha (the mechanism check the advisor asked for).

Outputs are written to ``student_logs/spatial_prior_diagnostic.json``
and a summary table to ``student_logs/spatial_prior_diagnostic.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.loader import _resolve_artifact_path  # noqa: PLC2701
from data.spatial_prior_sampler import compute_spatial_prior_weights_np


def _load_case_arrays(root: Path, case_id: str) -> dict[str, np.ndarray]:
    case_dir = root / case_id
    xyz = np.asarray(
        np.load(_resolve_artifact_path(case_dir / "surface_xyz.npy"), mmap_mode="r"),
        dtype=np.float32,
    )
    wss = np.asarray(
        np.load(_resolve_artifact_path(case_dir / "surface_wallshearstress.npy"), mmap_mode="r"),
        dtype=np.float32,
    )
    cp = np.asarray(
        np.load(_resolve_artifact_path(case_dir / "surface_cp.npy"), mmap_mode="r"),
        dtype=np.float32,
    ).reshape(-1)
    wss_mag = np.linalg.norm(wss, axis=1)
    return {"xyz": xyz, "wss_mag": wss_mag, "cp_abs": np.abs(cp)}


def _spatial_score(xyz: np.ndarray) -> np.ndarray:
    """Pre-alpha combined bias used purely for correlation reporting."""
    x = xyz[:, 0]
    z = xyz[:, 2]
    x_min = float(x.min())
    x_max = float(x.max())
    front_bias = (x_max - x) / max(x_max - x_min, 1e-8)
    z_abs = np.abs(z)
    z_abs_min = float(z_abs.min())
    z_abs_max = float(z_abs.max())
    ground_bias = (z_abs - z_abs_min) / max(z_abs_max - z_abs_min, 1e-8)
    return 0.5 * (front_bias + ground_bias)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    am = a - a.mean()
    bm = b - b.mean()
    denom = float(np.sqrt((am * am).sum()) * np.sqrt((bm * bm).sum()))
    if denom <= 0.0:
        return 0.0
    return float((am * bm).sum() / denom)


def _bin_occupancy(weights: np.ndarray) -> dict[str, float]:
    """Per-decile sampling-mass share under the given weights."""
    score_sorted = np.argsort(weights)[::-1]  # high-weight first
    w = weights[score_sorted]
    p = w / w.sum()
    n = w.size
    deciles = {}
    for d in range(10):
        lo = int(np.floor(d * n / 10))
        hi = int(np.floor((d + 1) * n / 10))
        deciles[f"d{d+1}_top{(d+1)*10}pct"] = float(p[:hi].sum())
    deciles["bottom_50pct_share"] = float(p[int(0.5 * n) :].sum())
    return deciles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511",
    )
    parser.add_argument(
        "--cases",
        default="run_1,run_50,run_100,run_200,run_300,run_400,run_485",
    )
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--out-json", default="student_logs/spatial_prior_diagnostic.json")
    parser.add_argument("--out-md", default="student_logs/spatial_prior_diagnostic.md")
    args = parser.parse_args()

    root = Path(args.data_root)
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    per_case = []
    for case_id in case_ids:
        arrays = _load_case_arrays(root, case_id)
        score = _spatial_score(arrays["xyz"])
        weights = compute_spatial_prior_weights_np(arrays["xyz"], alpha=args.alpha)
        bins = _bin_occupancy(weights)
        record = {
            "case_id": case_id,
            "n_surface": int(arrays["xyz"].shape[0]),
            "pearson_spatial_vs_WSS_mag": _pearson(score, arrays["wss_mag"]),
            "pearson_spatial_vs_Cp_abs": _pearson(score, arrays["cp_abs"]),
            "weights_min": float(weights.min()),
            "weights_max": float(weights.max()),
            "weights_mean": float(weights.mean()),
            "bin_occupancy": bins,
        }
        per_case.append(record)
        print(
            f"{case_id:>10} | N={record['n_surface']:>8} | "
            f"rho(spatial,|WSS|)={record['pearson_spatial_vs_WSS_mag']:+0.4f} | "
            f"rho(spatial,|Cp|)={record['pearson_spatial_vs_Cp_abs']:+0.4f} | "
            f"top10pct_mass={bins['d1_top10pct']:0.4f}"
        )

    mean_rho_wss = float(np.mean([r["pearson_spatial_vs_WSS_mag"] for r in per_case]))
    mean_rho_cp = float(np.mean([r["pearson_spatial_vs_Cp_abs"] for r in per_case]))
    mean_top10 = float(np.mean([r["bin_occupancy"]["d1_top10pct"] for r in per_case]))
    mean_bottom50 = float(np.mean([r["bin_occupancy"]["bottom_50pct_share"] for r in per_case]))

    summary = {
        "alpha": args.alpha,
        "n_cases": len(per_case),
        "mean_pearson_spatial_vs_WSS_mag": mean_rho_wss,
        "mean_pearson_spatial_vs_Cp_abs": mean_rho_cp,
        "mean_top10pct_mass": mean_top10,
        "mean_top10pct_oversample": mean_top10 / 0.10,
        "mean_bottom50pct_mass": mean_bottom50,
        "mean_bottom50pct_undersample": mean_bottom50 / 0.50,
        "green_light_threshold_rho_ge_0_20": mean_rho_wss >= 0.20,
        "per_case": per_case,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"# Spatial-prior diagnostic (alpha={args.alpha})",
        "",
        f"Cases: {', '.join(case_ids)}",
        "",
        "## Pearson correlation (mean across cases)",
        "",
        "| Quantity | Mean |",
        "|---|---:|",
        f"| pearson(spatial_score, \\|WSS\\|) | {mean_rho_wss:+0.4f} |",
        f"| pearson(spatial_score, \\|Cp\\|)  | {mean_rho_cp:+0.4f} |",
        "",
        f"Green light (ρ ≥ +0.20): **{'PASS' if mean_rho_wss >= 0.20 else 'FAIL'}**",
        "",
        "## Bin-occupancy (mean across cases)",
        "",
        "| Bin | Uniform | Weighted | Oversample |",
        "|---|---:|---:|---:|",
        f"| Top 10% | 10.00% | {100*mean_top10:0.2f}% | x{mean_top10/0.10:0.2f} |",
        f"| Bottom 50% | 50.00% | {100*mean_bottom50:0.2f}% | x{mean_bottom50/0.50:0.2f} |",
        "",
        "## Per-case",
        "",
        "| Case | N | rho(spatial,\\|WSS\\|) | rho(spatial,\\|Cp\\|) | top10%_mass | bot50%_mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in per_case:
        lines.append(
            f"| {r['case_id']} | {r['n_surface']} | "
            f"{r['pearson_spatial_vs_WSS_mag']:+0.4f} | "
            f"{r['pearson_spatial_vs_Cp_abs']:+0.4f} | "
            f"{100*r['bin_occupancy']['d1_top10pct']:0.2f}% | "
            f"{100*r['bin_occupancy']['bottom_50pct_share']:0.2f}% |"
        )
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nMean rho(spatial,|WSS|) = {mean_rho_wss:+0.4f}")
    print(f"Mean top10% mass = {mean_top10:0.4f} (oversample x{mean_top10/0.10:0.2f})")
    print(f"Mean bot50% mass = {mean_bottom50:0.4f} (undersample x{mean_bottom50/0.50:0.2f})")
    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
