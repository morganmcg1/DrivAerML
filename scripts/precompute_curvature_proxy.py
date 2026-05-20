# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Precompute k-NN normal-variation curvature proxy per case (WSS H5 PR #1132).

For each DrivAerML case:
  - Build a KDTree over surface_xyz, query k=16 nearest neighbors per point.
  - kappa_H = 1 - mean(dot(n_i, n_j)) over the k-neighborhood (mean-curvature proxy).
  - kappa_G = var(dot(n_i, n_j)) (Gaussian-curvature proxy / saddle indicator).
  - kappa_mag = sqrt(kappa_H^2 + kappa_G^2).

Writes per-case ``surface_curvature_proxy_k16_v1.npy`` of shape (V, 3) alongside
the existing surface_xyz.npy arrays. Aggregates train-split mean/std into
``curvature_proxy_stats_k16_v1.json`` placed under target/ so the case store
can z-score the channels at load time.

Idempotent: skips any case whose cache file already matches the surface row
count, so reruns are cheap.

Usage::

  python -m scripts.precompute_curvature_proxy \\
      --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \\
      --k 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DrivAerMLCaseStore  # noqa: E402

CACHE_FILENAME = "surface_curvature_proxy_k16_v1.npy"
STATS_FILENAME = "curvature_proxy_stats_k16_v1.json"


def compute_surface_curvature_proxy(
    normals: np.ndarray,
    positions: np.ndarray,
    *,
    k: int = 16,
) -> np.ndarray:
    """Return (V, 3) [kappa_H, kappa_G, kappa_mag] curvature proxy.

    The proxy follows the PR #1132 spec:
      - kappa_H = 1 - mean(dot(n_i, n_j))  over the k-NN neighborhood
      - kappa_G = var(dot(n_i, n_j))
      - kappa_mag = sqrt(kappa_H^2 + kappa_G^2)
    """
    if positions.shape[0] < k:
        raise ValueError(
            f"Need at least {k} surface points to query k-NN, got {positions.shape[0]}"
        )
    tree = cKDTree(positions, balanced_tree=True, compact_nodes=True)
    _, idxs = tree.query(positions, k=k, workers=-1)
    # idxs: (V, k); normals: (V, 3); normals[idxs]: (V, k, 3)
    # dot_products[v, j] = sum_c normals[v, c] * normals[idxs[v, j], c]
    dot_products = np.einsum("vc,vkc->vk", normals, normals[idxs])
    dot_products = np.clip(dot_products, -1.0, 1.0)
    kappa_H = (1.0 - dot_products.mean(axis=1)).astype(np.float32)
    kappa_G = dot_products.var(axis=1).astype(np.float32)
    kappa_mag = np.sqrt(kappa_H * kappa_H + kappa_G * kappa_G).astype(np.float32)
    out = np.stack([kappa_H, kappa_G, kappa_mag], axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_or_load_case(case_dir: Path, *, k: int, force: bool) -> np.ndarray:
    cache_path = case_dir / CACHE_FILENAME
    xyz_path = case_dir / "surface_xyz.npy"
    normals_path = case_dir / "surface_normals.npy"
    n_surface = int(np.load(xyz_path, mmap_mode="r").shape[0])
    if cache_path.exists() and not force:
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == (n_surface, 3) and cached.dtype == np.float32:
            return np.asarray(cached, dtype=np.float32)
    xyz = np.asarray(np.load(xyz_path), dtype=np.float32)
    normals = np.asarray(np.load(normals_path), dtype=np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    normals = normals / norms
    curv = compute_surface_curvature_proxy(normals, xyz, k=k)
    tmp = cache_path.parent / (cache_path.stem + ".tmp.npy")
    np.save(tmp, curv)
    tmp.replace(cache_path)
    return curv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Override DrivAerML PVC root")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "split_manifest.json"))
    parser.add_argument("--k", type=int, default=16, help="k-NN neighborhood size")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most N cases (0 = all). For quick debugging.",
    )
    parser.add_argument(
        "--stats-out", default=str(REPO_ROOT / STATS_FILENAME),
        help="Where to write the train-split mean/std JSON",
    )
    args = parser.parse_args()

    store = DrivAerMLCaseStore(manifest_path=args.manifest, root=args.data_root)
    splits = {split: store.case_ids(split) for split in ("train", "val", "test")}
    all_cases = splits["train"] + splits["val"] + splits["test"]
    if args.limit > 0:
        all_cases = all_cases[: args.limit]
        print(f"Processing first {len(all_cases)} cases only (--limit)")

    sum_x = np.zeros(3, dtype=np.float64)
    sum_x2 = np.zeros(3, dtype=np.float64)
    n_total = np.zeros((), dtype=np.float64)
    raw_min = np.full(3, np.inf, dtype=np.float64)
    raw_max = np.full(3, -np.inf, dtype=np.float64)
    train_set = set(splits["train"])

    t0 = time.time()
    for idx, case_id in enumerate(all_cases, 1):
        case_dir = store.root / case_id
        cached_existed = (case_dir / CACHE_FILENAME).exists()
        curv = compute_or_load_case(case_dir, k=args.k, force=args.force)
        if case_id in train_set:
            sum_x += curv.sum(axis=0, dtype=np.float64)
            sum_x2 += (curv.astype(np.float64) ** 2).sum(axis=0)
            n_total += curv.shape[0]
            raw_min = np.minimum(raw_min, curv.min(axis=0))
            raw_max = np.maximum(raw_max, curv.max(axis=0))
        if idx % 20 == 0 or idx == len(all_cases):
            dt = time.time() - t0
            print(
                f"[{idx}/{len(all_cases)}] {case_id} V={curv.shape[0]} "
                f"{'cached' if cached_existed and not args.force else 'computed'} "
                f"elapsed={dt:.1f}s"
            )

    if n_total <= 0:
        raise RuntimeError("No training cases processed; cannot derive stats")
    mean = (sum_x / n_total).astype(np.float64)
    var = (sum_x2 / n_total) - mean * mean
    std = np.sqrt(np.clip(var, 0.0, None)).astype(np.float64)
    std = np.maximum(std, 1e-6)
    stats = {
        "version": "k16_v1",
        "k": int(args.k),
        "channels": ["kappa_H", "kappa_G", "kappa_mag"],
        "n_train_points": int(n_total),
        "n_train_cases": int(len(splits["train"])),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "raw_min": raw_min.tolist(),
        "raw_max": raw_max.tolist(),
    }
    stats_path = Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nWrote stats to {stats_path}")
    print(f"  mean = {mean.tolist()}")
    print(f"  std  = {std.tolist()}")
    print(f"  raw range kappa_H = [{raw_min[0]:.4g}, {raw_max[0]:.4g}]")
    print(f"  raw range kappa_G = [{raw_min[1]:.4g}, {raw_max[1]:.4g}]")
    print(f"  raw range kappa_M = [{raw_min[2]:.4g}, {raw_max[2]:.4g}]")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
