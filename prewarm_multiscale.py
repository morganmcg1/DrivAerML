#!/usr/bin/env python
"""Pre-warm the multi-scale kNN feature cache for all train+val+test cases.

Computing per-case features lazily inside DataLoader workers (32 concurrent
processes on an 8-rank DDP run) over-subscribes the CPUs because both
``cKDTree(..., workers=-1)`` and BLAS-backed numpy operations grab every
available core inside each worker. This script runs the same feature compute
sequentially within each subprocess and parallelises across cases via
``concurrent.futures.ProcessPoolExecutor``, with strict per-worker thread
caps so the wall-clock cost is bounded.

Usage::

    uv run python prewarm_multiscale.py \
        --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
        --multiscale-k-values 4,16,64 \
        --processes 16

The cache is co-located in each case directory and shared across all training
runs, so this script is a one-time cost per ``k_values`` tuple.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _limit_threads(n: int) -> None:
    """Cap per-process numpy/BLAS/MKL threads to avoid CPU over-subscription."""

    n = max(1, int(n))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(n)


def _worker(args: tuple[str, str, tuple[int, ...], int]) -> tuple[str, float, str]:
    case_id, root, k_values, threads = args
    _limit_threads(threads)
    # Imports happen inside the subprocess so the thread caps above take effect
    # before BLAS is initialised.
    from data.loader import (  # noqa: E402  (intentional late import)
        _load_or_compute_multiscale_features,
        _npy_row_count,
        _case_dir,
    )

    case_dir = _case_dir(Path(root), case_id)
    full_rows = _npy_row_count(case_dir / "surface_xyz.npy")
    t0 = time.time()
    arr = _load_or_compute_multiscale_features(case_dir, k_values, full_rows=full_rows)
    elapsed = time.time() - t0
    return case_id, elapsed, f"shape={arr.shape}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="DrivAerML processed root, e.g. /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511",
    )
    parser.add_argument(
        "--manifest",
        default="data/split_manifest.json",
        help="Path to split manifest JSON",
    )
    parser.add_argument(
        "--multiscale-k-values",
        default="4,16,64",
        help="Comma-separated kNN scales (default: 4,16,64)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="Number of subprocesses to use for feature compute (default: 8)",
    )
    parser.add_argument(
        "--threads-per-process",
        type=int,
        default=0,
        help="Per-process thread cap for numpy/cKDTree. 0 -> auto-pick (cpu_count // processes).",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Splits to pre-warm (default: all three)",
    )
    args = parser.parse_args()

    from data.loader import (  # noqa: E402  (relies on data module being importable)
        DrivAerMLCaseStore,
        parse_multiscale_k_values,
    )

    k_values = parse_multiscale_k_values(args.multiscale_k_values)
    if not k_values:
        raise SystemExit("--multiscale-k-values must be non-empty (e.g. '4,16,64')")

    store = DrivAerMLCaseStore(manifest_path=args.manifest, root=args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    case_ids: list[str] = []
    seen: set[str] = set()
    for split in splits:
        for cid in store.case_ids(split):
            if cid not in seen:
                seen.add(cid)
                case_ids.append(cid)

    n_cpus = os.cpu_count() or 8
    threads = args.threads_per_process or max(1, n_cpus // max(1, args.processes))
    print(
        f"Pre-warming multi-scale cache for {len(case_ids)} cases | "
        f"k_values={list(k_values)} | processes={args.processes} | "
        f"threads/process={threads} | data_root={store.root}"
    )

    futures_args = [(cid, str(store.root), tuple(k_values), threads) for cid in case_ids]
    successes: list[tuple[str, float, str]] = []
    failures: list[tuple[str, str]] = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        future_map = {pool.submit(_worker, fa): fa[0] for fa in futures_args}
        for i, fut in enumerate(as_completed(future_map), start=1):
            case_id = future_map[fut]
            try:
                result = fut.result()
                successes.append(result)
                elapsed = result[1]
                wall = time.time() - t_start
                print(
                    f"  [{i}/{len(case_ids)}] {case_id}: {elapsed:.1f}s ({result[2]}) "
                    f"wall={wall/60:.1f}min"
                )
            except Exception as exc:  # pragma: no cover - prewarm diagnostic
                failures.append((case_id, repr(exc)))
                print(f"  [{i}/{len(case_ids)}] {case_id}: FAILED -> {exc!r}")

    total = time.time() - t_start
    print(
        f"\nDone in {total/60:.1f}min. "
        f"{len(successes)} succeeded, {len(failures)} failed."
    )
    if failures:
        for cid, err in failures:
            print(f"  FAIL {cid}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
