# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Regenerate volume_sdf.npy for the 10 REQUIRED_RESTORED cases.

The 10 restored cases (per `data/loader.REQUIRED_RESTORED_CASE_IDS`) have a
volume_sdf.npy that is missing the negative (inside-body) signed-distance branch.
Their `sdf_negative_frac` is ~10x lower than the bulk train cases. This script
recomputes volume_sdf.npy from the existing volume_xyz.npy and the raw STL surface
mesh, using the canonical pipeline (`pyvista.compute_implicit_distance` against
the triangulated STL), and replaces the symlinked volume_sdf.npy with a regular
.npy file. The backup directory is NEVER touched.

Verified: re-running this pipeline on a known-good case (run_1) reproduces its
existing volume_sdf.npy bit-exactly (zero abs diff over 100k random samples).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from data.loader import REQUIRED_RESTORED_CASE_IDS, _resolve_artifact_path

DATA_ROOT = Path("/mnt/new-pvc/Processed/drivaerml_processed")
RAW_DIR = Path("/mnt/new-pvc/Datasets/2_Drivearml")


def regenerate_one(case_id: str, dry_run: bool) -> dict:
    run_id = int(case_id.split("_")[1])
    case_dir = DATA_ROOT / case_id
    stl_path = RAW_DIR / case_id / f"drivaer_{run_id}.stl"
    sdf_link = case_dir / "volume_sdf.npy"

    t0 = time.time()
    vol_xyz_path = _resolve_artifact_path(case_dir / "volume_xyz.npy")
    vol_xyz = np.load(vol_xyz_path)
    n_pts = vol_xyz.shape[0]

    stl = pv.read(str(stl_path))
    if not isinstance(stl, pv.PolyData):
        stl = stl.extract_surface()
    stl = stl.triangulate()

    query = pv.PolyData(vol_xyz.astype(np.float32))
    query = query.compute_implicit_distance(stl, inplace=False)
    new_sdf = np.asarray(query.point_data["implicit_distance"], dtype=np.float32)

    sdf_min = float(new_sdf.min())
    sdf_max = float(new_sdf.max())
    sdf_neg_frac = float((new_sdf < 0.0).mean())
    elapsed = time.time() - t0

    # Capture old (corrupted) stats for the report
    old_sdf_path = _resolve_artifact_path(sdf_link)
    old_sdf = np.load(old_sdf_path)
    old_sdf_min = float(old_sdf.min())
    old_sdf_neg_frac = float((old_sdf < 0.0).mean())

    if dry_run:
        print(
            f"[DRY] {case_id}: N={n_pts} new_sdf_min={sdf_min:.4f} new_sdf_max={sdf_max:.4f} "
            f"new_sdf_neg_frac={sdf_neg_frac:.2e} (was min={old_sdf_min:.4f}, "
            f"neg_frac={old_sdf_neg_frac:.2e}) elapsed={elapsed:.1f}s",
            flush=True,
        )
    else:
        # Replace symlink with a real file. Write via a temp file + atomic rename.
        if sdf_link.is_symlink() or sdf_link.exists():
            os.remove(sdf_link)
        tmp_path = sdf_link.with_suffix(".npy.tmp")
        np.save(tmp_path, new_sdf)
        os.replace(tmp_path, sdf_link)
        # Verify written
        readback = np.load(sdf_link)
        assert readback.shape == (n_pts,), f"shape mismatch: {readback.shape} vs ({n_pts},)"
        assert np.allclose(readback, new_sdf, atol=1e-6, rtol=0), "value mismatch on readback"
        print(
            f"[OK] {case_id}: N={n_pts} sdf_min={sdf_min:.4f} sdf_max={sdf_max:.4f} "
            f"sdf_neg_frac={sdf_neg_frac:.2e} (was min={old_sdf_min:.4f}, "
            f"neg_frac={old_sdf_neg_frac:.2e}) elapsed={elapsed:.1f}s",
            flush=True,
        )

    return {
        "case_id": case_id,
        "n_pts": n_pts,
        "old_sdf_min": old_sdf_min,
        "old_sdf_neg_frac": old_sdf_neg_frac,
        "new_sdf_min": sdf_min,
        "new_sdf_max": sdf_max,
        "new_sdf_neg_frac": sdf_neg_frac,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write")
    parser.add_argument(
        "--cases",
        default=None,
        help="Optional comma-separated case_id subset (default: all 10 restored)",
    )
    args = parser.parse_args()

    if args.cases is not None:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(REQUIRED_RESTORED_CASE_IDS)

    print(f"Regenerating volume_sdf.npy for {len(cases)} cases: {cases}")
    print(f"  DATA_ROOT = {DATA_ROOT}")
    print(f"  RAW_DIR   = {RAW_DIR}")
    print(f"  dry_run   = {args.dry_run}")

    results = []
    t_start = time.time()
    for i, case_id in enumerate(cases):
        print(f"\n--- {i+1}/{len(cases)}: {case_id} ---", flush=True)
        results.append(regenerate_one(case_id, args.dry_run))
    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")

    print("\n=== SUMMARY ===")
    print(f"{'case_id':<12} {'old_neg_frac':>14} {'new_neg_frac':>14} {'old_sdf_min':>12} {'new_sdf_min':>12}")
    for r in results:
        print(
            f"{r['case_id']:<12} {r['old_sdf_neg_frac']:>14.6e} {r['new_sdf_neg_frac']:>14.6e} "
            f"{r['old_sdf_min']:>12.4f} {r['new_sdf_min']:>12.4f}"
        )


if __name__ == "__main__":
    main()
