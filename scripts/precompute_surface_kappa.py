"""Precompute approximate mean curvature kappa for every DrivAerML case.

Saves a per-case ``surface_kappa_v2.npy`` (float32 [N]) into each case
directory in the processed data root. Speeds up
``CurvatureWeightedSurfaceDataset`` from ~20 s per worker first-touch to a
<100 ms load.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.curvature_sampler import approx_mean_curvature_np  # noqa: E402


def case_kappa_path(case_dir: Path) -> Path:
    return case_dir / "surface_kappa_v2.npy"


def _compute_one(args: tuple[str, str, int, bool]) -> tuple[str, float, str]:
    case_id, root_str, k, force = args
    root = Path(root_str)
    case_dir = root / case_id
    out_path = case_kappa_path(case_dir)
    if out_path.exists() and not force:
        return (case_id, 0.0, "skip")
    t0 = time.time()
    try:
        xyz = np.load(case_dir / "surface_xyz.npy")
        normals = np.load(case_dir / "surface_normals.npy")
        kappa = approx_mean_curvature_np(xyz, normals, k=k)
        np.save(out_path, kappa.astype(np.float32))
        return (case_id, time.time() - t0, "ok")
    except Exception as exc:  # noqa: BLE001
        return (case_id, time.time() - t0, f"err: {exc}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    root = Path(args.data_root)
    manifest = root / "manifest.csv"
    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    case_ids = [r["case_id"] for r in rows]
    print(f"Cases in manifest: {len(case_ids)}")

    todo = [
        (cid, str(root), args.k, args.force)
        for cid in case_ids
        if args.force or not case_kappa_path(root / cid).exists()
    ]
    print(
        f"Already computed: {len(case_ids) - len(todo)}; "
        f"to compute: {len(todo)} with workers={args.workers}"
    )
    if not todo:
        print("All done.")
        return

    t_start = time.time()
    n_done = 0
    n_err = 0
    sum_dt = 0.0
    with mp.Pool(args.workers) as pool:
        for case_id, dt, status in pool.imap_unordered(_compute_one, todo, chunksize=1):
            n_done += 1
            if not status.startswith("ok") and status != "skip":
                n_err += 1
                print(f"[err] {case_id}: {status}")
            sum_dt += dt
            if n_done % 20 == 0 or n_done == len(todo):
                wall = time.time() - t_start
                rate = n_done / wall if wall > 0 else 0
                eta = (len(todo) - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done}/{len(todo)}] wall={wall:.1f}s "
                    f"avg_cpu_per_case={sum_dt/max(n_done,1):.2f}s rate={rate:.2f}/s eta={eta:.0f}s"
                )
    wall = time.time() - t_start
    print(f"Done in {wall:.1f}s; errors={n_err}")


if __name__ == "__main__":
    main()
