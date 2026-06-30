#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Precompute k-NN normal-variation curvature proxy files for DrivAerML cases."""

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
    k: int,
) -> np.ndarray:
    if positions.shape[0] < k:
        raise ValueError(f"Need at least {k} surface points, got {positions.shape[0]}")
    tree = cKDTree(positions, balanced_tree=True, compact_nodes=True)
    _, idxs = tree.query(positions, k=k, workers=-1)
    dot_products = np.einsum("vc,vkc->vk", normals, normals[idxs])
    dot_products = np.clip(dot_products, -1.0, 1.0)
    kappa_h = (1.0 - dot_products.mean(axis=1)).astype(np.float32)
    kappa_g = dot_products.var(axis=1).astype(np.float32)
    kappa_mag = np.sqrt(kappa_h * kappa_h + kappa_g * kappa_g).astype(np.float32)
    curv = np.stack([kappa_h, kappa_g, kappa_mag], axis=1).astype(np.float32)
    return np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)


def valid_cache(cache_path: Path, n_surface: int) -> bool:
    if not cache_path.exists():
        return False
    cached = np.load(cache_path, mmap_mode="r")
    return cached.shape == (n_surface, 3) and cached.dtype == np.float32


def compute_or_load_case(case_dir: Path, *, k: int, force: bool, need_values: bool) -> np.ndarray | None:
    cache_path = case_dir / CACHE_FILENAME
    xyz_path = case_dir / "surface_xyz.npy"
    normals_path = case_dir / "surface_normals.npy"
    n_surface = int(np.load(xyz_path, mmap_mode="r").shape[0])
    if not force and valid_cache(cache_path, n_surface):
        if need_values:
            return np.asarray(np.load(cache_path, mmap_mode="r"), dtype=np.float32)
        return None

    xyz = np.asarray(np.load(xyz_path), dtype=np.float32)
    normals = np.asarray(np.load(normals_path), dtype=np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    curv = compute_surface_curvature_proxy(normals, xyz, k=k)
    tmp = cache_path.parent / f"{cache_path.stem}.tmp.npy"
    np.save(tmp, curv)
    tmp.replace(cache_path)
    return curv if need_values else None


def selected_case_ids(store: DrivAerMLCaseStore, args: argparse.Namespace) -> list[str]:
    if args.case_id:
        cases = list(args.case_id)
    else:
        splits = args.split or ["train", "val", "test"]
        cases = []
        for split in splits:
            cases.extend(store.case_ids(split))
    if args.limit > 0:
        cases = cases[: args.limit]
    if args.shard_count > 1:
        cases = cases[args.shard_index :: args.shard_count]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Override DrivAerML processed root")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "split_manifest.json"))
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--stats-out", default="")
    args = parser.parse_args()

    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < count")

    store = DrivAerMLCaseStore(manifest_path=args.manifest, root=args.data_root)
    case_ids = selected_case_ids(store, args)
    train_set = set(store.case_ids("train"))
    need_stats = bool(args.stats_out)
    print(
        f"Processing {len(case_ids)} cases from {store.root} "
        f"(shard {args.shard_index}/{args.shard_count})",
        flush=True,
    )

    sum_x = np.zeros(3, dtype=np.float64)
    sum_x2 = np.zeros(3, dtype=np.float64)
    n_total = 0
    raw_min = np.full(3, np.inf, dtype=np.float64)
    raw_max = np.full(3, -np.inf, dtype=np.float64)
    t0 = time.time()

    for idx, case_id in enumerate(case_ids, 1):
        case_dir = store.root / case_id
        cached_existed = valid_cache(
            case_dir / CACHE_FILENAME,
            int(np.load(case_dir / "surface_xyz.npy", mmap_mode="r").shape[0]),
        )
        curv = compute_or_load_case(
            case_dir,
            k=args.k,
            force=args.force,
            need_values=need_stats and case_id in train_set,
        )
        if curv is not None:
            sum_x += curv.sum(axis=0, dtype=np.float64)
            sum_x2 += (curv.astype(np.float64) ** 2).sum(axis=0)
            n_total += int(curv.shape[0])
            raw_min = np.minimum(raw_min, curv.min(axis=0))
            raw_max = np.maximum(raw_max, curv.max(axis=0))
        print(
            f"[{idx}/{len(case_ids)}] {case_id} "
            f"{'cached' if cached_existed and not args.force else 'computed'} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    if need_stats:
        if n_total <= 0:
            raise RuntimeError("No train cases processed; cannot derive curvature stats")
        mean = sum_x / n_total
        var = (sum_x2 / n_total) - mean * mean
        std = np.maximum(np.sqrt(np.clip(var, 0.0, None)), 1e-6)
        stats = {
            "version": "k16_v1",
            "k": int(args.k),
            "channels": ["kappa_H", "kappa_G", "kappa_mag"],
            "n_train_points": int(n_total),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "raw_min": raw_min.tolist(),
            "raw_max": raw_max.tolist(),
        }
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"Wrote stats to {stats_path}", flush=True)

    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
