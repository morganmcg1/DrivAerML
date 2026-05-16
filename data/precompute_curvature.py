# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Pre-build the per-case curvature cache for the curvature feature (PR #1146).

Run once before launching the training job to amortise the kNN compute
outside the DataLoader hot path. Safe to re-run: cases that already have
the cache file are skipped.

Usage:
    uv run python -m data.precompute_curvature \
        --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
        --knn 16 --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

from . import DEFAULT_MANIFEST
from .loader import (
    DrivAerMLCaseStore,
    _case_dir,
    _load_or_compute_curvature,
)


def _build_one(case_root: str, case_id: str, knn: int) -> tuple[str, float, bool]:
    case_dir = _case_dir(Path(case_root), case_id)
    cache = case_dir / f"surface_curvature_k{knn}.npy"
    if cache.exists():
        return case_id, 0.0, False
    t0 = time.time()
    _load_or_compute_curvature(case_dir, knn)
    return case_id, time.time() - t0, True


def _worker(args: tuple[str, str, int]) -> tuple[str, float, bool]:
    return _build_one(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-build curvature cache.")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="DrivAerML processed root (PVC path).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="Split manifest JSON path.",
    )
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    store = DrivAerMLCaseStore(manifest_path=args.manifest, root=args.data_root)
    all_cases = []
    for split in ("train", "val", "test"):
        all_cases.extend(store.case_ids(split))
    print(f"Pre-building curvature cache for {len(all_cases)} cases (K={args.knn})")
    case_root = str(store.root)
    work = [(case_root, cid, args.knn) for cid in all_cases]

    t0 = time.time()
    built = 0
    skipped = 0
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        for i, (case_id, dt, did_build) in enumerate(pool.imap_unordered(_worker, work), 1):
            if did_build:
                built += 1
            else:
                skipped += 1
            if i % 20 == 0 or i == len(work):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(work) - i)
                print(
                    f"  [{i:4d}/{len(work):4d}] {case_id}: "
                    f"{'BUILT' if did_build else 'skip'} ({dt:.2f}s) | "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )
    total = time.time() - t0
    print(f"Done: built={built} skipped={skipped} in {total:.1f}s")


if __name__ == "__main__":
    main()
