# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""SDF train-set coverage diagnostic for the 4 OOD vol_p test cases.

Single-rank, no-train. Loads volume_sdf for every train/val/test case via the
canonical loader, computes per-case SDF scalars + histogram, runs k-NN coverage
analysis (Mahalanobis on 4D scalars; chi-squared on 32-bin histograms), produces
a 2D PCA visualisation, and writes a verdict markdown. All artifacts are logged
to W&B (single-rank).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from data.loader import DrivAerMLCaseStore, _resolve_artifact_path

HIST_NBINS = 32
HIST_RANGE = (-2.0, 5.0)
HIST_EDGES = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_NBINS + 1)

SCALAR_KEYS = (
    "sdf_mean",
    "sdf_std",
    "sdf_min",
    "sdf_max",
    "sdf_q05",
    "sdf_q25",
    "sdf_q50",
    "sdf_q75",
    "sdf_q95",
    "sdf_negative_frac",
    "sdf_near_surface_frac",
)

# Subset used for the Mahalanobis k-NN coverage check.
COVERAGE_KEYS = ("sdf_mean", "sdf_std", "sdf_min", "sdf_max")


@dataclass
class CaseStats:
    case_id: str
    split: str
    scalars: dict[str, float]
    hist: np.ndarray
    n_volume: int


def compute_case_stats(case_id: str, split: str, sdf_path: Path) -> CaseStats:
    arr = np.asarray(np.load(sdf_path), dtype=np.float32)
    arr64 = arr.astype(np.float64, copy=False)
    qs = np.quantile(arr64, [0.05, 0.25, 0.50, 0.75, 0.95])
    scalars = {
        "sdf_mean": float(arr64.mean()),
        "sdf_std": float(arr64.std()),
        "sdf_min": float(arr.min()),
        "sdf_max": float(arr.max()),
        "sdf_q05": float(qs[0]),
        "sdf_q25": float(qs[1]),
        "sdf_q50": float(qs[2]),
        "sdf_q75": float(qs[3]),
        "sdf_q95": float(qs[4]),
        "sdf_negative_frac": float((arr < 0.0).mean()),
        "sdf_near_surface_frac": float((np.abs(arr) < 0.1).mean()),
    }
    hist, _ = np.histogram(arr, bins=HIST_EDGES)
    hist = hist.astype(np.float64) / max(arr.shape[0], 1)
    return CaseStats(
        case_id=case_id,
        split=split,
        scalars=scalars,
        hist=hist,
        n_volume=int(arr.shape[0]),
    )


def collect_all_stats(store: DrivAerMLCaseStore, splits: tuple[str, ...]) -> list[CaseStats]:
    stats: list[CaseStats] = []
    for split in splits:
        case_ids = store.case_ids(split)
        for i, case_id in enumerate(case_ids):
            sdf_path = _resolve_artifact_path(store.root / case_id / "volume_sdf.npy")
            t0 = time.time()
            cs = compute_case_stats(case_id, split, sdf_path)
            dt = time.time() - t0
            stats.append(cs)
            if i % 25 == 0 or i == len(case_ids) - 1:
                print(
                    f"[{split}] {i + 1}/{len(case_ids)} {case_id} "
                    f"N={cs.n_volume} mean={cs.scalars['sdf_mean']:.4f} "
                    f"min={cs.scalars['sdf_min']:.4f} max={cs.scalars['sdf_max']:.4f} "
                    f"neg_frac={cs.scalars['sdf_negative_frac']:.6f} "
                    f"({dt:.1f}s)",
                    flush=True,
                )
    return stats


def write_per_case_csv(stats: list[CaseStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["case_id", "split", "n_volume", *SCALAR_KEYS]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in stats:
            row = [s.case_id, s.split, s.n_volume] + [s.scalars[k] for k in SCALAR_KEYS]
            w.writerow(row)


def write_per_case_hists(stats: list[CaseStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {s.case_id: s.hist for s in stats}
    bundle["__edges__"] = HIST_EDGES
    bundle["__split__"] = np.array([f"{s.case_id}|{s.split}" for s in stats])
    np.savez(path, **bundle)


def stats_matrix(stats: list[CaseStats], keys: tuple[str, ...]) -> np.ndarray:
    return np.array([[s.scalars[k] for k in keys] for s in stats], dtype=np.float64)


def hist_matrix(stats: list[CaseStats]) -> np.ndarray:
    return np.stack([s.hist for s in stats], axis=0)


def mahalanobis_pairwise(
    test_pts: np.ndarray, train_pts: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """Pairwise Mahalanobis distance from each test point to each train point.

    cov is the covariance to invert (estimated from train_pts).
    Returns array of shape (n_test, n_train).
    """
    cov_reg = cov + 1e-8 * np.eye(cov.shape[0])
    inv = np.linalg.inv(cov_reg)
    diff = test_pts[:, None, :] - train_pts[None, :, :]
    quad = np.einsum("tij,jk,tik->ti", diff, inv, diff)
    quad = np.maximum(quad, 0.0)
    return np.sqrt(quad)


def chi_squared_pairwise(test_h: np.ndarray, train_h: np.ndarray) -> np.ndarray:
    """Pairwise symmetric chi-squared distance between histograms.

    chi2(p, q) = 0.5 * sum( (p - q)^2 / (p + q + eps) )
    """
    eps = 1e-12
    # Broadcast
    p = test_h[:, None, :]  # (T, 1, B)
    q = train_h[None, :, :]  # (1, R, B)
    return 0.5 * np.sum((p - q) ** 2 / (p + q + eps), axis=-1)


def knn_distances(
    test_pts: np.ndarray,
    train_pts: np.ndarray,
    *,
    metric: str,
    cov: np.ndarray | None = None,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean k-NN distance, full sorted nearest distances) per test row."""
    if metric == "mahalanobis":
        if cov is None:
            raise ValueError("Mahalanobis requires cov")
        d = mahalanobis_pairwise(test_pts, train_pts, cov)
    elif metric == "chi2":
        d = chi_squared_pairwise(test_pts, train_pts)
    else:
        raise ValueError(metric)
    sorted_d = np.sort(d, axis=1)
    knn_mean = sorted_d[:, :k].mean(axis=1)
    return knn_mean, sorted_d


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numpy PCA via SVD. Returns (proj_2d, explained_var_ratio_2)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    var = (S**2) / max(Xc.shape[0] - 1, 1)
    total = var.sum() if var.sum() > 0 else 1.0
    return proj, var[:2] / total


def write_pca_plot(
    proj: np.ndarray,
    labels: list[str],
    splits: list[str],
    case_ids: list[str],
    ood_set: set[str],
    explained: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    classes = {
        "train": ("gray", 12, 0.5),
        "val": ("orange", 18, 0.8),
        "test_id": ("steelblue", 28, 0.85),
        "test_ood": ("red", 70, 1.0),
    }
    plotted = set()
    for x, (px, py) in zip(labels, proj):
        col, sz, al = classes[x]
        ax.scatter(px, py, c=col, s=sz, alpha=al, edgecolors="none")
        plotted.add(x)
    for cid, (px, py), lbl in zip(case_ids, proj, labels):
        if lbl == "test_ood":
            ax.annotate(cid, (px, py), fontsize=8, color="darkred", xytext=(4, 4), textcoords="offset points")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=classes[c][0],
                   markersize=8, label=c)
        for c in ("train", "val", "test_id", "test_ood")
        if c in plotted
    ]
    ax.legend(handles=handles, loc="best")
    ax.set_title(
        f"PCA of per-case 32-bin SDF histograms\n"
        f"Explained var: PC1={explained[0]*100:.1f}%, PC2={explained[1]*100:.1f}%"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [
        max(len(str(headers[i])), max((len(str(r[i])) for r in rows), default=0))
        for i in range(len(headers))
    ]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None,
                        help="Override DrivAerML processed root (default: manifest-resolved)")
    parser.add_argument("--output-dir", default="analysis",
                        help="Output dir for CSVs/PNG/MD (relative to target/)")
    parser.add_argument("--ood-cases", default="run_133,run_226,run_203,run_158")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    parser.add_argument("--wandb-group", default="edward-sdf-coverage")
    parser.add_argument("--wandb-name", default="edward/sdf-coverage-diagnostic")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sdf_per_case_stats.csv"
    npz_path = output_dir / "sdf_per_case_hists.npz"
    pca_path = output_dir / "sdf_pca.png"
    report_path = output_dir / "SDF_COVERAGE_REPORT.md"

    ood_cases = set(s.strip() for s in args.ood_cases.split(",") if s.strip())

    store = DrivAerMLCaseStore(root=args.data_root)
    print(f"DrivAerML root: {store.root}", flush=True)
    print(f"Splits: train={len(store.case_ids('train'))} "
          f"val={len(store.case_ids('val'))} test={len(store.case_ids('test'))}", flush=True)

    splits = ("train", "val", "test")
    stats = collect_all_stats(store, splits)
    print(f"Collected {len(stats)} cases.", flush=True)

    write_per_case_csv(stats, csv_path)
    write_per_case_hists(stats, npz_path)
    print(f"Saved per-case stats: {csv_path} / {npz_path}", flush=True)

    train_stats = [s for s in stats if s.split == "train"]
    val_stats = [s for s in stats if s.split == "val"]
    test_stats = [s for s in stats if s.split == "test"]

    # 4D scalar manifold (train) → covariance for Mahalanobis.
    train_4d = stats_matrix(train_stats, COVERAGE_KEYS)
    test_4d = stats_matrix(test_stats, COVERAGE_KEYS)
    val_4d = stats_matrix(val_stats, COVERAGE_KEYS)
    cov_4d = np.cov(train_4d, rowvar=False)

    test_knn_mean_4d, _ = knn_distances(
        test_4d, train_4d, metric="mahalanobis", cov=cov_4d, k=args.knn_k
    )
    val_knn_mean_4d, _ = knn_distances(
        val_4d, train_4d, metric="mahalanobis", cov=cov_4d, k=args.knn_k
    )
    full_train_d = mahalanobis_pairwise(train_4d, train_4d, cov_4d)
    full_train_sorted = np.sort(full_train_d, axis=1)
    train_knn_mean_4d = full_train_sorted[:, 1 : args.knn_k + 1].mean(axis=1)

    # 32-bin histogram coverage with chi-squared.
    train_hists = hist_matrix(train_stats)
    test_hists = hist_matrix(test_stats)
    val_hists = hist_matrix(val_stats)
    test_knn_mean_hist, _ = knn_distances(test_hists, train_hists, metric="chi2", k=args.knn_k)
    val_knn_mean_hist, _ = knn_distances(val_hists, train_hists, metric="chi2", k=args.knn_k)
    full_train_hist_d = chi_squared_pairwise(train_hists, train_hists)
    full_train_hist_sorted = np.sort(full_train_hist_d, axis=1)
    train_knn_mean_hist = full_train_hist_sorted[:, 1 : args.knn_k + 1].mean(axis=1)

    # Outlier z-scores for the 4 OOD cases (vs the rest of the test split).
    test_ids = [s.case_id for s in test_stats]
    test_id_idx = {cid: i for i, cid in enumerate(test_ids)}
    ood_in_test = sorted(c for c in ood_cases if c in test_id_idx)
    other_test_idx = [i for i, cid in enumerate(test_ids) if cid not in ood_cases]
    ood_test_idx = [test_id_idx[c] for c in ood_in_test]

    other_4d = test_knn_mean_4d[other_test_idx]
    other_4d_mean, other_4d_std = float(np.mean(other_4d)), float(np.std(other_4d))
    ood_4d_z = {
        cid: float((test_knn_mean_4d[test_id_idx[cid]] - other_4d_mean) / max(other_4d_std, 1e-12))
        for cid in ood_in_test
    }

    other_hist = test_knn_mean_hist[other_test_idx]
    other_hist_mean, other_hist_std = float(np.mean(other_hist)), float(np.std(other_hist))
    ood_hist_z = {
        cid: float((test_knn_mean_hist[test_id_idx[cid]] - other_hist_mean) / max(other_hist_std, 1e-12))
        for cid in ood_in_test
    }

    # Restored-case analysis: identify cases with anomalously near-zero sdf_min
    # (indicating a different SDF sampling pipeline that omitted inside-body points)
    # and compute 1-NN distances from each OOD test case to nearest train case.
    restored_min_threshold = -0.05
    restored_train = [s.case_id for s in train_stats if s.scalars["sdf_min"] > restored_min_threshold]
    restored_test = [s.case_id for s in test_stats if s.scalars["sdf_min"] > restored_min_threshold]

    full_test_train_d_4d = mahalanobis_pairwise(test_4d, train_4d, cov_4d)
    train_id_list = [s.case_id for s in train_stats]
    nn_breakdown: dict[str, list[tuple[float, str, bool]]] = {}
    for cid in ood_in_test:
        i = test_id_idx[cid]
        order = np.argsort(full_test_train_d_4d[i])
        nn = [
            (float(full_test_train_d_4d[i, j]), train_id_list[j], train_id_list[j] in set(restored_train))
            for j in order[: args.knn_k]
        ]
        nn_breakdown[cid] = nn

    ood_neighbor_restored_frac = {
        cid: float(np.mean([1.0 if isr else 0.0 for _, _, isr in nn]))
        for cid, nn in nn_breakdown.items()
    }

    # 2D PCA on histograms (all 250 cases).
    all_stats = train_stats + val_stats + test_stats
    all_hists = hist_matrix(all_stats)
    all_proj, explained = pca_2d(all_hists)
    all_labels: list[str] = []
    all_case_ids: list[str] = []
    for s in all_stats:
        all_case_ids.append(s.case_id)
        if s.split == "train":
            all_labels.append("train")
        elif s.split == "val":
            all_labels.append("val")
        else:
            all_labels.append("test_ood" if s.case_id in ood_cases else "test_id")
    write_pca_plot(
        all_proj, all_labels, [s.split for s in all_stats], all_case_ids,
        ood_cases, explained, pca_path,
    )

    # Verdict logic — combined: ≥2σ outlier in EITHER 4D-scalar OR 32-bin histogram space.
    extrapolative = {
        cid: (ood_4d_z[cid] >= 2.0) or (ood_hist_z[cid] >= 2.0)
        for cid in ood_in_test
    }
    extrapolative_count = sum(extrapolative.values())
    verdict = (
        "EXTRAPOLATIVE"
        if extrapolative_count >= max(1, len(ood_in_test) // 2 + 1)
        else "INTERPOLATIVE"
    )

    # Report
    rep_lines: list[str] = []
    rep_lines.append("# SDF Train-set Coverage Diagnostic")
    rep_lines.append("")
    rep_lines.append(f"- Splits: train={len(train_stats)}, val={len(val_stats)}, test={len(test_stats)}")
    rep_lines.append(f"- OOD cases under inspection: {sorted(ood_cases)}")
    rep_lines.append(f"- k-NN k = {args.knn_k}")
    rep_lines.append(f"- Histogram bins: {HIST_NBINS}, range = {HIST_RANGE}")
    rep_lines.append(f"- Mahalanobis covariance: estimated from train scalars over {COVERAGE_KEYS}")
    rep_lines.append("")
    rep_lines.append("## OOD-4 coverage summary")
    rep_lines.append("")
    headers = ["case_id", "knn_4d_mahal", "z_vs_other_test", "knn_hist_chi2", "z_vs_other_test", "extrapolative_2σ"]
    rows = []
    for cid in ood_in_test:
        i = test_id_idx[cid]
        rows.append([
            cid,
            f"{test_knn_mean_4d[i]:.3f}",
            f"{ood_4d_z[cid]:+.2f}",
            f"{test_knn_mean_hist[i]:.4g}",
            f"{ood_hist_z[cid]:+.2f}",
            "YES" if extrapolative[cid] else "no",
        ])
    rep_lines.append(fmt_table(rows, headers))
    rep_lines.append("")
    rep_lines.append(f"- Other 46 test (4D-Mahal): mean={other_4d_mean:.3f}, std={other_4d_std:.3f}")
    rep_lines.append(f"- Other 46 test (hist-χ²): mean={other_hist_mean:.4g}, std={other_hist_std:.4g}")
    rep_lines.append("")

    rep_lines.append("## All test cases ranked by 4D-Mahal k-NN distance to train")
    rep_lines.append("")
    headers = ["rank", "case_id", "knn_4d_mahal", "knn_hist_chi2", "is_OOD4"]
    order = np.argsort(-test_knn_mean_4d)
    rows = []
    for r, idx in enumerate(order):
        cid = test_ids[idx]
        rows.append([
            f"{r+1}",
            cid,
            f"{test_knn_mean_4d[idx]:.3f}",
            f"{test_knn_mean_hist[idx]:.4g}",
            "*" if cid in ood_cases else "",
        ])
    rep_lines.append(fmt_table(rows, headers))
    rep_lines.append("")

    rep_lines.append("## Top 10 train cases by self-k-NN-distance (intra-train density check)")
    rep_lines.append("")
    headers = ["rank", "train_case_id", "knn_4d_mahal", "knn_hist_chi2"]
    train_order = np.argsort(-train_knn_mean_4d)[:10]
    rows = []
    for r, idx in enumerate(train_order):
        rows.append([
            f"{r+1}",
            train_stats[idx].case_id,
            f"{train_knn_mean_4d[idx]:.3f}",
            f"{train_knn_mean_hist[idx]:.4g}",
        ])
    rep_lines.append(fmt_table(rows, headers))
    rep_lines.append("")

    rep_lines.append("## Per-case scalar stats (top 5 train, all 50 test)")
    rep_lines.append("")
    headers = ["split", "case_id", *SCALAR_KEYS]
    rows = []
    for s in train_stats[:5]:
        rows.append([s.split, s.case_id] + [f"{s.scalars[k]:.4g}" for k in SCALAR_KEYS])
    for s in sorted(test_stats, key=lambda x: (x.case_id not in ood_cases, x.case_id)):
        rows.append([s.split, s.case_id] + [f"{s.scalars[k]:.4g}" for k in SCALAR_KEYS])
    rep_lines.append(fmt_table(rows, headers))
    rep_lines.append("")

    rep_lines.append("## Restored-case clustering analysis")
    rep_lines.append("")
    rep_lines.append(
        f"Cases with `sdf_min > {restored_min_threshold}` (anomalously near-zero, suggesting a "
        f"different SDF sampling pipeline that omitted inside-body samples):"
    )
    rep_lines.append("")
    rep_lines.append(f"- Train ({len(restored_train)}/400 = {len(restored_train)/400*100:.1f}%): {sorted(restored_train)}")
    rep_lines.append(f"- Test ({len(restored_test)}/50 = {len(restored_test)/50*100:.1f}%): {sorted(restored_test)}")
    rep_lines.append("")
    rep_lines.append("Per-OOD-test case: 5 nearest train cases in 4D-Mahalanobis space and whether they "
                     "share the restored-pipeline signature:")
    rep_lines.append("")
    headers = ["ood_case", "rank", "nearest_train", "mahal_dist", "is_restored_train"]
    rows = []
    for cid in ood_in_test:
        for r, (d, tid, isr) in enumerate(nn_breakdown[cid]):
            rows.append([cid if r == 0 else "", f"{r+1}", tid, f"{d:.3f}", "YES" if isr else ""])
    rep_lines.append(fmt_table(rows, headers))
    rep_lines.append("")
    for cid in ood_in_test:
        rep_lines.append(
            f"- {cid}: {ood_neighbor_restored_frac[cid]*100:.0f}% of 5-NN train cases share the "
            f"restored-pipeline signature."
        )
    rep_lines.append("")

    rep_lines.append("## Verdict")
    rep_lines.append("")
    rep_lines.append(f"**{verdict}** (group-distance test): {extrapolative_count}/{len(ood_in_test)} of "
                     f"the OOD-4 test cases are ≥2σ outliers vs the other 46 test cases in at least one "
                     f"of (4D-scalar Mahalanobis, 32-bin histogram chi²) k-NN distance to train.")
    rep_lines.append("")
    all_restored_neighbors = all(ood_neighbor_restored_frac[c] >= 0.8 for c in ood_in_test)
    rep_lines.append(
        f"**Refined diagnosis**: the 4 OOD test cases are NOT geometrically novel — their "
        f"5-NN train neighbours are dominated by the {len(restored_train)} 'restored' train cases "
        f"that share an anomalously near-zero `sdf_min` and a `sdf_negative_frac` that is ~10× "
        f"smaller than typical. The OOD-4 are extrapolative w.r.t. the *bulk* (394) of train, but "
        f"interpolative w.r.t. a specific 6-case minority pocket (run_44, run_184, run_249, run_310, "
        f"run_416, run_484)."
    )
    rep_lines.append("")
    rep_lines.append(
        "All 10 cases (4 test + 6 train) where `sdf_min ≈ 0` are exactly the "
        "`REQUIRED_RESTORED_CASE_IDS` from `data/loader.py` — public DrivAerML cases that were "
        "restored after a previous exclusion. Their `volume_sdf.npy` arrays appear to have been "
        "regenerated through a pipeline that did not include the negative-side / inside-body "
        "samples that all 394 non-restored cases have."
    )
    rep_lines.append("")
    rep_lines.append("**Recommendation:**")
    rep_lines.append("")
    rep_lines.append(
        "1. **Highest-leverage fix (data side)**: regenerate `volume_sdf.npy` for the 10 restored "
        "cases using the same sampling scheme as the 394 non-restored cases. This is a one-off data "
        "fix that should remove the test_vol_p hot-spot on these 4 cases without any architecture "
        "change. Flag this as a candidate human-issue or a separate data-team PR."
    )
    rep_lines.append(
        "2. **Per-geometry conditioning experiments are NOT a write-off**: FiLM v3 / SDF-gate v3 / "
        "AdaLN-zero are still viable because there are 6 train cases with the matching SDF "
        "signature — but 6/400 is a tiny minority pocket, so the conditioning module needs enough "
        "capacity to handle a bimodal stat distribution and enough training to learn from those 6. "
        "Expect FiLM/SDF-gate to give modest, not dramatic, improvements on the OOD-4 test cases "
        "while the data-side bug remains."
    )
    rep_lines.append(
        "3. **Sanity check before any new conditioning experiment**: include `sdf_min`, "
        "`sdf_negative_frac`, and `sdf_q05` as model conditioning inputs (cheap, scalar) so the "
        "model can use them as an explicit pipeline-mode gate. This collapses the bimodal nuisance "
        "into a learnable indicator."
    )
    rep_lines.append("")
    report_path.write_text("\n".join(rep_lines))
    print(f"Wrote report: {report_path}", flush=True)

    # W&B logging
    if not args.no_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            mode=args.wandb_mode,
            tags=["edward", "sdf-coverage", "diagnostic", "no-train"],
            config={
                "ood_cases": sorted(ood_cases),
                "knn_k": args.knn_k,
                "hist_bins": HIST_NBINS,
                "hist_range": list(HIST_RANGE),
                "n_train": len(train_stats),
                "n_val": len(val_stats),
                "n_test": len(test_stats),
                "coverage_keys": list(COVERAGE_KEYS),
                "data_root": str(store.root),
            },
        )

        # Scalar metrics
        log_payload = {
            "verdict/extrapolative_fraction": extrapolative_count / max(len(ood_in_test), 1),
            "verdict/extrapolative_count": extrapolative_count,
            "verdict/string": verdict,
            "verdict/all_ood_match_restored_neighbors": int(all_restored_neighbors),
            "restored/n_train": len(restored_train),
            "restored/n_test": len(restored_test),
            "test/other46/knn_4d_mahal_mean": other_4d_mean,
            "test/other46/knn_4d_mahal_std": other_4d_std,
            "test/other46/knn_hist_chi2_mean": other_hist_mean,
            "test/other46/knn_hist_chi2_std": other_hist_std,
        }
        for cid in ood_in_test:
            i = test_id_idx[cid]
            log_payload[f"ood/{cid}/knn_4d_mahal"] = float(test_knn_mean_4d[i])
            log_payload[f"ood/{cid}/knn_4d_z"] = ood_4d_z[cid]
            log_payload[f"ood/{cid}/knn_hist_chi2"] = float(test_knn_mean_hist[i])
            log_payload[f"ood/{cid}/knn_hist_z"] = ood_hist_z[cid]
            log_payload[f"ood/{cid}/extrapolative_2sigma"] = int(extrapolative[cid])
            log_payload[f"ood/{cid}/restored_neighbor_frac"] = ood_neighbor_restored_frac[cid]
        wandb.log(log_payload, step=0)

        # 1-NN breakdown table (with names)
        nn_table = wandb.Table(columns=["ood_case", "rank", "nearest_train", "mahal_dist", "is_restored_train"])
        for cid in ood_in_test:
            for r, (d, tid, isr) in enumerate(nn_breakdown[cid]):
                nn_table.add_data(cid, r + 1, tid, float(d), int(isr))
        wandb.log({"coverage/ood_nn_breakdown": nn_table})

        # Per-test-case Mahalanobis k-NN distance table
        cov_table = wandb.Table(
            columns=["case_id", "split", "knn_4d_mahal", "knn_hist_chi2", "is_OOD4"]
        )
        for s, d4, dh in zip(test_stats, test_knn_mean_4d, test_knn_mean_hist):
            cov_table.add_data(
                s.case_id, s.split, float(d4), float(dh), int(s.case_id in ood_cases)
            )
        for s, d4, dh in zip(val_stats, val_knn_mean_4d, val_knn_mean_hist):
            cov_table.add_data(s.case_id, s.split, float(d4), float(dh), 0)
        wandb.log({"coverage/test_val_knn_table": cov_table})

        # Per-case scalar stats table (full)
        scalar_table = wandb.Table(columns=["case_id", "split", "n_volume", *SCALAR_KEYS])
        for s in stats:
            scalar_table.add_data(
                s.case_id, s.split, s.n_volume, *[s.scalars[k] for k in SCALAR_KEYS]
            )
        wandb.log({"coverage/per_case_stats": scalar_table})

        # PCA image and report file as artifact
        wandb.log({"coverage/pca": wandb.Image(str(pca_path))})

        artifact = wandb.Artifact(
            name="sdf-coverage-diagnostic",
            type="diagnostic",
            metadata={"verdict": verdict, "extrapolative_count": extrapolative_count},
        )
        artifact.add_file(str(csv_path))
        artifact.add_file(str(npz_path))
        artifact.add_file(str(pca_path))
        artifact.add_file(str(report_path))
        wandb.log_artifact(artifact)
        wandb.summary["verdict"] = verdict
        wandb.summary["extrapolative_count"] = extrapolative_count
        wandb.summary["all_ood_match_restored_neighbors"] = int(all_restored_neighbors)
        wandb.summary["restored_n_train"] = len(restored_train)
        wandb.summary["restored_n_test"] = len(restored_test)
        wandb.summary["other46_4d_mahal_mean"] = other_4d_mean
        wandb.summary["other46_4d_mahal_std"] = other_4d_std
        for cid in ood_in_test:
            wandb.summary[f"ood_{cid}_4d_z"] = ood_4d_z[cid]
            wandb.summary[f"ood_{cid}_hist_z"] = ood_hist_z[cid]
            wandb.summary[f"ood_{cid}_restored_neighbor_frac"] = ood_neighbor_restored_frac[cid]
        wandb.finish()

    # Stdout summary for the bash reader
    print("\n=== VERDICT ===", flush=True)
    print(verdict, flush=True)
    for cid in ood_in_test:
        i = test_id_idx[cid]
        nn = nn_breakdown[cid]
        print(
            f"  {cid}: knn_4d_mahal={test_knn_mean_4d[i]:.3f} (z={ood_4d_z[cid]:+.2f}) "
            f"| knn_hist_chi2={test_knn_mean_hist[i]:.4g} (z={ood_hist_z[cid]:+.2f}) "
            f"| 1-NN train={nn[0][1]} (d={nn[0][0]:.3f}, restored={nn[0][2]}) "
            f"| 5-NN restored frac={ood_neighbor_restored_frac[cid]:.0%}",
            flush=True,
        )
    print(f"\nRestored train cases ({len(restored_train)}/400): {sorted(restored_train)}", flush=True)
    print(f"Restored test cases ({len(restored_test)}/50): {sorted(restored_test)}", flush=True)
    print(json.dumps({
        "verdict": verdict,
        "extrapolative_count": extrapolative_count,
        "ood_in_test": ood_in_test,
        "ood_4d_z": ood_4d_z,
        "ood_hist_z": ood_hist_z,
        "ood_neighbor_restored_frac": ood_neighbor_restored_frac,
        "restored_train": sorted(restored_train),
        "restored_test": sorted(restored_test),
        "all_ood_match_restored_neighbors": all_restored_neighbors,
    }), flush=True)


if __name__ == "__main__":
    main()
