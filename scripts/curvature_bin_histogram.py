"""Diagnostic: curvature bin-occupancy histogram for the curvature-weighted surface sampler.

Loads precomputed `surface_kappa_v2.npy` for a sample of cases, bins curvature
into quantile bands, and reports both the natural occupancy and the expected
sampling occupancy under `w = 1 + alpha * (kappa / mean(kappa))`. The ratio
shows the effective oversampling factor per band, which is the "band balance"
the advisor wants to inspect.

Also reports correlation of curvature with WSS magnitude (target leakage check
+ proxy validation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def quantile_bins(kappa: np.ndarray, n_bins: int = 10) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(kappa, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def fixed_bins(kappa: np.ndarray, n_bins: int = 10) -> np.ndarray:
    lo, hi = float(np.min(kappa)), float(np.max(kappa))
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def occupancy_and_weight_per_bin(
    kappa: np.ndarray,
    alpha: float,
    floor: float = 0.1,
    n_bins: int = 10,
) -> dict:
    mean_kappa = float(kappa.mean())
    kappa_normed = kappa / mean_kappa if mean_kappa > 0 else kappa
    w = 1.0 + alpha * kappa_normed
    if floor > 0.0:
        w = np.maximum(w, floor)
    probs = w / w.sum()
    edges = quantile_bins(kappa, n_bins)
    bin_idx = np.digitize(kappa, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    n_per_bin = np.bincount(bin_idx, minlength=n_bins).astype(np.int64)
    prob_per_bin = np.bincount(bin_idx, weights=probs, minlength=n_bins)
    uniform_per_bin = n_per_bin / n_per_bin.sum()
    kappa_mean_per_bin = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx == b
        kappa_mean_per_bin[b] = float(kappa[mask].mean()) if mask.any() else 0.0
    bin_lo = np.array([
        float(np.min(kappa[bin_idx == b])) if (bin_idx == b).any() else 0.0
        for b in range(n_bins)
    ])
    bin_hi = np.array([
        float(np.max(kappa[bin_idx == b])) if (bin_idx == b).any() else 0.0
        for b in range(n_bins)
    ])
    return dict(
        n_per_bin=n_per_bin.tolist(),
        prob_per_bin=prob_per_bin.tolist(),
        uniform_per_bin=uniform_per_bin.tolist(),
        oversampling_ratio=(prob_per_bin / np.maximum(uniform_per_bin, 1e-12)).tolist(),
        kappa_mean_per_bin=kappa_mean_per_bin.tolist(),
        kappa_lo_per_bin=bin_lo.tolist(),
        kappa_hi_per_bin=bin_hi.tolist(),
        kappa_overall_mean=mean_kappa,
        kappa_overall_max=float(kappa.max()),
        kappa_overall_p99=float(np.quantile(kappa, 0.99)),
        weight_overall_max=float(w.max()),
    )


def wss_correlation(kappa: np.ndarray, wss: np.ndarray) -> dict:
    wss_mag = np.linalg.norm(wss, axis=-1)
    finite = np.isfinite(kappa) & np.isfinite(wss_mag)
    a = kappa[finite]
    b = wss_mag[finite]
    pearson = float(np.corrcoef(a, b)[0, 1])
    # spearman approx via rank
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    # top-decile capture: among the natural top-10% WSS points, what fraction is in the
    # top-10% curvature points?
    top10_wss = b >= np.quantile(b, 0.90)
    top10_kappa = a >= np.quantile(a, 0.90)
    top10_intersection = float((top10_wss & top10_kappa).sum() / max(top10_wss.sum(), 1))
    return dict(
        pearson_corr=pearson,
        spearman_corr=spearman,
        top10_wss_in_top10_kappa=top10_intersection,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511",
    )
    parser.add_argument(
        "--cases", nargs="*",
        default=["run_1", "run_50", "run_100", "run_200", "run_300", "run_400"],
    )
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--floor", type=float, default=0.1)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--out-json", default="student_logs/curv_bin_hist.json")
    args = parser.parse_args()

    root = Path(args.data_root)
    out = {"args": vars(args), "per_case": {}}
    bin_probs_acc = np.zeros(args.n_bins, dtype=np.float64)
    bin_uniform_acc = np.zeros(args.n_bins, dtype=np.float64)
    ratios_acc = []
    pearsons = []
    spearmans = []
    top10_caps = []

    for case in args.cases:
        kappa_path = root / case / "surface_kappa_v2.npy"
        wss_path = root / case / "surface_wallshearstress.npy"
        if not kappa_path.exists():
            print(f"[skip] no kappa for {case}")
            continue
        kappa = np.load(kappa_path).astype(np.float32)
        info = occupancy_and_weight_per_bin(
            kappa, alpha=args.alpha, floor=args.floor, n_bins=args.n_bins,
        )
        try:
            wss = np.load(wss_path).astype(np.float32)
            wss_info = wss_correlation(kappa, wss)
        except Exception as e:  # noqa: BLE001
            wss_info = {"error": str(e)}
        out["per_case"][case] = {"bins": info, "wss": wss_info}
        bin_probs_acc += np.array(info["prob_per_bin"])
        bin_uniform_acc += np.array(info["uniform_per_bin"])
        if "pearson_corr" in wss_info:
            pearsons.append(wss_info["pearson_corr"])
            spearmans.append(wss_info["spearman_corr"])
            top10_caps.append(wss_info["top10_wss_in_top10_kappa"])
        print(
            f"[{case}] N={kappa.size:>8d} "
            f"kappa: mean={info['kappa_overall_mean']:.2f} max={info['kappa_overall_max']:.1f} "
            f"p99={info['kappa_overall_p99']:.1f} | "
            f"w_max={info['weight_overall_max']:.1f}"
        )
        for b in range(args.n_bins):
            print(
                f"   bin {b}: kappa [{info['kappa_lo_per_bin'][b]:.2f},"
                f" {info['kappa_hi_per_bin'][b]:.2f}]  "
                f"uniform={info['uniform_per_bin'][b]*100:5.2f}% "
                f"weighted={info['prob_per_bin'][b]*100:5.2f}% "
                f"oversample x{info['oversampling_ratio'][b]:.2f}"
            )
        if "pearson_corr" in wss_info:
            print(
                f"   wss corr: pearson={wss_info['pearson_corr']:.3f} "
                f"spearman={wss_info['spearman_corr']:.3f} "
                f"top10WSSinTop10Kappa={wss_info['top10_wss_in_top10_kappa']*100:.1f}%"
            )
        ratios_acc.append(np.array(info["oversampling_ratio"]))

    n = len(out["per_case"])
    if n:
        bin_probs_avg = (bin_probs_acc / n).tolist()
        bin_uniform_avg = (bin_uniform_acc / n).tolist()
        oversample_avg = (np.array(bin_probs_avg) / np.maximum(np.array(bin_uniform_avg), 1e-12)).tolist()
        ratio_per_bin_mean = np.mean(np.stack(ratios_acc, axis=0), axis=0).tolist()
        out["averaged"] = dict(
            uniform=bin_uniform_avg,
            weighted=bin_probs_avg,
            oversampling_ratio=oversample_avg,
            mean_ratio_across_cases=ratio_per_bin_mean,
            pearson_mean=float(np.mean(pearsons)) if pearsons else None,
            spearman_mean=float(np.mean(spearmans)) if spearmans else None,
            top10_capture_mean=float(np.mean(top10_caps)) if top10_caps else None,
        )
        print("\n=== AVERAGED ACROSS CASES ===")
        for b in range(args.n_bins):
            print(
                f"bin {b}: uniform={bin_uniform_avg[b]*100:5.2f}% "
                f"weighted={bin_probs_avg[b]*100:5.2f}% "
                f"oversample x{oversample_avg[b]:.2f}"
            )
        if pearsons:
            print(
                f"\nWSS-vs-kappa pearson  mean={np.mean(pearsons):.3f} "
                f"(per-case range [{min(pearsons):.3f}, {max(pearsons):.3f}])"
            )
            print(
                f"WSS-vs-kappa spearman mean={np.mean(spearmans):.3f} "
                f"(per-case range [{min(spearmans):.3f}, {max(spearmans):.3f}])"
            )
            print(
                f"Top-10% WSS coverage by top-10% kappa: "
                f"mean={np.mean(top10_caps)*100:.1f}% "
                f"(per-case range [{min(top10_caps)*100:.1f}%, {max(top10_caps)*100:.1f}%])"
            )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
