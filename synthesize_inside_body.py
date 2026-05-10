# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Synthesize inside-body volume samples for the 10 REQUIRED_RESTORED cases.

Root cause (verified empirically): the 10 REQUIRED_RESTORED cases were processed
through an older pipeline that did NOT apply the inside-body augmentation step
that the 394 good cases have. Re-deriving from the VTU is impossible because the
original synthetic inside-body points are not VTU cell centres (they have per-axis
xyz_stored vs vtu_centers differences of 0.3-0.7 m).

Approved fix (Option A in PR #941, advisor: morganmcg1):
  1. STL rejection sampling: generate ~1500-2000 uniformly-random points inside
     the closed STL body geometry per case
  2. SDF: pyvista compute_implicit_distance against the triangulated STL
     (verified bit-exact match to canonical pipeline on run_1)
  3. Field interpolation (pressure / velocity / totalpcoeff): mean of the 32
     nearest existing OUTSIDE-body cells via scipy.spatial.cKDTree
  4. Volume indices: sentinel value -1 for synthetic rows (not loaded by the
     dataloader on this branch)
  5. Append synthetic rows to existing volume_xyz / volume_sdf / volume_pressure /
     volume_velocity / volume_totalpcoeff / volume_indices .npy files. Atomic
     write via .tmp + os.replace. Backup directory NEVER touched.

Acceptance criterion (sdf_coverage_diagnostic.py): post-fix sdf_negative_frac
in [1.0e-4, 2.5e-4] for all 10 cases.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from data.loader import REQUIRED_RESTORED_CASE_IDS, _resolve_artifact_path

DATA_ROOT = Path("/mnt/new-pvc/Processed/drivaerml_processed")
RAW_DIR = Path("/mnt/new-pvc/Datasets/2_Drivearml")

# Target ~1.5e-4 sdf_negative_frac (mid-range of [1.0e-4, 2.5e-4] acceptance).
# Run_1 (good reference): N=14_744_958, num_neg=1952, neg_frac=1.32e-4.
TARGET_NEG_FRAC = 1.5e-4
# Inside-body threshold: deeper than -INSIDE_THRESH counts as "deeply inside".
# Using a small margin so we avoid surface-noise points and produce a meaningful
# negative-SDF distribution.
INSIDE_THRESH = 0.005

# k for nearest-neighbour field interpolation
KNN_K = 32

# Initial number of bbox candidates (before STL inside test). The car bbox is
# ~14 m³ and the body interior is roughly 50% of bbox, so ~50% acceptance.
INITIAL_BATCH = 8192


def stl_rejection_sample(
    stl: pv.PolyData,
    n_target: int,
    rng: np.random.Generator,
    bbox_inflate: float = 0.0,
    max_iters: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n_target uniformly-random points strictly inside the closed STL.

    Returns (xyz, sdf) where xyz.shape=(n_target, 3) f32 and sdf.shape=(n_target,)
    f32 with all values < -INSIDE_THRESH.

    Strategy:
      - Loop over batches: generate uniform-in-bbox candidates, compute SDF via
        pyvista compute_implicit_distance, keep ones with sdf < -INSIDE_THRESH
      - Stop when we have >= n_target accepted points; truncate to n_target

    The closed STL means inside ≡ negative SDF (no ambiguity).
    """
    xmin, xmax, ymin, ymax, zmin, zmax = stl.bounds
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    if bbox_inflate:
        xmin, xmax = xmin - bbox_inflate * dx, xmax + bbox_inflate * dx
        ymin, ymax = ymin - bbox_inflate * dy, ymax + bbox_inflate * dy
        zmin, zmax = zmin - bbox_inflate * dz, zmax + bbox_inflate * dz

    accepted_xyz_chunks: list[np.ndarray] = []
    accepted_sdf_chunks: list[np.ndarray] = []
    n_accepted = 0
    n_attempted = 0

    for it in range(max_iters):
        if n_accepted >= n_target:
            break
        # Adaptive batch size: ramp up if acceptance is low
        deficit = n_target - n_accepted
        # Estimate acceptance rate so far; default to 0.4 if no history
        rate = max(0.05, n_accepted / max(1, n_attempted)) if n_attempted else 0.4
        batch = max(INITIAL_BATCH, int(deficit / rate * 1.2))
        batch = min(batch, 200_000)  # cap to avoid huge memory

        cand = rng.uniform(
            low=[xmin, ymin, zmin],
            high=[xmax, ymax, zmax],
            size=(batch, 3),
        ).astype(np.float32)
        query = pv.PolyData(cand)
        query = query.compute_implicit_distance(stl, inplace=False)
        sdf = np.asarray(query.point_data["implicit_distance"], dtype=np.float32)

        keep = sdf < -INSIDE_THRESH
        if keep.any():
            accepted_xyz_chunks.append(cand[keep])
            accepted_sdf_chunks.append(sdf[keep])
            n_accepted += int(keep.sum())
        n_attempted += batch
        logging.info(
            "  iter %d: tried %d, accepted %d / total %d (rate=%.3f)",
            it, batch, int(keep.sum()), n_accepted, n_accepted / n_attempted,
        )

    if n_accepted < n_target:
        raise RuntimeError(
            f"rejection sampling did not converge: got {n_accepted} / {n_target} after {max_iters} iters"
        )

    xyz = np.concatenate(accepted_xyz_chunks, axis=0)[:n_target]
    sdf = np.concatenate(accepted_sdf_chunks, axis=0)[:n_target]
    return xyz, sdf


def knn_interpolate(
    existing_xyz: np.ndarray,
    existing_sdf: np.ndarray,
    synthetic_xyz: np.ndarray,
    field_arrays: dict[str, np.ndarray],
    k: int = KNN_K,
) -> dict[str, np.ndarray]:
    """For each synthetic point, return the mean of the field over its k-NN
    in the OUTSIDE-body subset of existing points.

    `existing_sdf >= 0` defines outside-body (the bulk of the data). We use only
    outside-body neighbours so the synthetic inside-body fields are the
    extrapolation of the outside flow into the body, which is the physically
    sensible choice (CFD pressure varies smoothly across the boundary layer).
    """
    outside_mask = existing_sdf >= 0
    n_outside = int(outside_mask.sum())
    if n_outside < k:
        raise RuntimeError(f"too few outside-body samples ({n_outside}) for k={k}")
    out_xyz = existing_xyz[outside_mask]

    logging.info("  building cKDTree on %d outside-body points...", n_outside)
    t0 = time.time()
    tree = cKDTree(out_xyz)
    logging.info("  cKDTree built in %.1fs; querying %d × k=%d ...", time.time() - t0, len(synthetic_xyz), k)

    t0 = time.time()
    _, nn_idx = tree.query(synthetic_xyz, k=k)  # (M, k) indices into out_xyz
    logging.info("  cKDTree query in %.1fs", time.time() - t0)

    interp: dict[str, np.ndarray] = {}
    for fname, full_arr in field_arrays.items():
        out_field = full_arr[outside_mask]
        # mean over k neighbours
        if out_field.ndim == 1:
            interp[fname] = out_field[nn_idx].mean(axis=1).astype(np.float32)
        else:
            interp[fname] = out_field[nn_idx].mean(axis=1).astype(np.float32)
    return interp


def synthesize_one(case_id: str, dry_run: bool, n_target: int | None = None) -> dict:
    run_id = int(case_id.split("_")[1])
    case_dir = DATA_ROOT / case_id
    stl_path = RAW_DIR / case_id / f"drivaer_{run_id}.stl"

    print(f"\n=== {case_id} ===", flush=True)
    t_total = time.time()

    # Load existing arrays (resolve through symlinks if any)
    paths = {
        "xyz": _resolve_artifact_path(case_dir / "volume_xyz.npy"),
        "sdf": _resolve_artifact_path(case_dir / "volume_sdf.npy"),
        "pres": _resolve_artifact_path(case_dir / "volume_pressure.npy"),
        "vel": _resolve_artifact_path(case_dir / "volume_velocity.npy"),
        "totalp": _resolve_artifact_path(case_dir / "volume_totalpcoeff.npy"),
        "ind": _resolve_artifact_path(case_dir / "volume_indices.npy"),
    }
    print("  loading existing arrays...", flush=True)
    xyz = np.load(paths["xyz"])
    sdf = np.load(paths["sdf"])
    pres = np.load(paths["pres"])
    vel = np.load(paths["vel"])
    totalp = np.load(paths["totalp"])
    ind = np.load(paths["ind"])
    n_orig = xyz.shape[0]
    print(
        f"  existing: N={n_orig}, sdf min={sdf.min():.4f} neg_frac={(sdf<0).mean():.4e} "
        f"deep<-0.01_count={int((sdf<-0.01).sum())}",
        flush=True,
    )

    # Determine target inside-body count to reach TARGET_NEG_FRAC
    # final_neg_frac = (existing_neg_count + n_new) / (n_orig + n_new) = TARGET_NEG_FRAC
    # solve: n_new = (TARGET_NEG_FRAC * n_orig - existing_neg_count) / (1 - TARGET_NEG_FRAC)
    existing_neg = int((sdf < 0).sum())
    if n_target is None:
        n_target = int(round((TARGET_NEG_FRAC * n_orig - existing_neg) / (1 - TARGET_NEG_FRAC)))
        n_target = max(n_target, 1500)  # floor: at least the run_1 count
    print(f"  computed n_target={n_target} (existing_neg={existing_neg}, target_neg_frac={TARGET_NEG_FRAC:.2e})", flush=True)

    # Load STL and triangulate
    print("  loading STL and triangulating...", flush=True)
    stl = pv.read(str(stl_path))
    if not isinstance(stl, pv.PolyData):
        stl = stl.extract_surface()
    stl = stl.triangulate()

    # STL rejection sampling
    print(f"  STL rejection sampling, target={n_target}...", flush=True)
    rng = np.random.default_rng(0xDEAD_C0DE + run_id)  # deterministic per case
    syn_xyz, syn_sdf = stl_rejection_sample(stl, n_target, rng)
    print(
        f"  synthetic: N={syn_xyz.shape[0]}, sdf range [{syn_sdf.min():.4f}, {syn_sdf.max():.4f}], "
        f"mean={syn_sdf.mean():.4f}",
        flush=True,
    )

    # k-NN interpolation of pressure, velocity, totalpcoeff from outside-body samples
    print(f"  k-NN field interpolation (k={KNN_K})...", flush=True)
    syn_fields = knn_interpolate(
        existing_xyz=xyz,
        existing_sdf=sdf,
        synthetic_xyz=syn_xyz,
        field_arrays={"pres": pres, "vel": vel, "totalp": totalp},
        k=KNN_K,
    )
    print(
        f"  synth pres range [{syn_fields['pres'].min():.4f}, {syn_fields['pres'].max():.4f}], "
        f"mean={syn_fields['pres'].mean():.4f}",
        flush=True,
    )

    # Concatenate: place synthetic rows at the END (so existing index alignment is preserved)
    new_xyz = np.concatenate([xyz, syn_xyz.astype(np.float32)], axis=0)
    new_sdf = np.concatenate([sdf, syn_sdf.astype(np.float32)], axis=0)
    # Pressure / vel / totalp: keep ndim consistent with existing arrays
    syn_pres = syn_fields["pres"]
    if pres.ndim == 2 and syn_pres.ndim == 1:
        syn_pres = syn_pres[:, None]
    new_pres = np.concatenate([pres, syn_pres.astype(np.float32)], axis=0)
    syn_vel = syn_fields["vel"]
    if vel.ndim == 2 and syn_vel.ndim == 1:
        syn_vel = syn_vel[:, None]
    new_vel = np.concatenate([vel, syn_vel.astype(np.float32)], axis=0)
    syn_totalp = syn_fields["totalp"]
    if totalp.ndim == 2 and syn_totalp.ndim == 1:
        syn_totalp = syn_totalp[:, None]
    new_totalp = np.concatenate([totalp, syn_totalp.astype(np.float32)], axis=0)
    # Sentinel index = -1 for synthetic points (not a valid VTU cell index)
    syn_ind = np.full(syn_xyz.shape[0], -1, dtype=ind.dtype)
    new_ind = np.concatenate([ind, syn_ind], axis=0)

    n_new = new_xyz.shape[0]
    final_neg_frac = float((new_sdf < 0).mean())
    print(
        f"  new totals: N={n_new} (was {n_orig}), final_neg_frac={final_neg_frac:.4e}, "
        f"final_sdf_min={new_sdf.min():.4f}",
        flush=True,
    )

    if dry_run:
        print(f"  [DRY] would write 6 .npy files to {case_dir}", flush=True)
    else:
        # Atomic writes
        targets = {
            "volume_xyz.npy": new_xyz,
            "volume_sdf.npy": new_sdf,
            "volume_pressure.npy": new_pres,
            "volume_velocity.npy": new_vel,
            "volume_totalpcoeff.npy": new_totalp,
            "volume_indices.npy": new_ind,
        }
        for fname, arr in targets.items():
            out_path = case_dir / fname
            tmp_path = case_dir / (fname.removesuffix(".npy") + ".tmp.npy")
            np.save(tmp_path, arr)
            assert tmp_path.exists()
            if out_path.is_symlink() or out_path.exists():
                os.remove(out_path)
            os.replace(tmp_path, out_path)
        print(f"  wrote {len(targets)} npy files to {case_dir}", flush=True)

    elapsed = time.time() - t_total
    print(f"  done in {elapsed:.1f}s", flush=True)

    return {
        "case_id": case_id,
        "n_orig": int(n_orig),
        "n_new": int(n_new),
        "n_synth": int(syn_xyz.shape[0]),
        "orig_sdf_min": float(sdf.min()),
        "new_sdf_min": float(new_sdf.min()),
        "orig_neg_frac": float((sdf < 0).mean()),
        "new_neg_frac": final_neg_frac,
        "synth_sdf_min": float(syn_sdf.min()),
        "synth_sdf_max": float(syn_sdf.max()),
        "synth_pres_min": float(syn_pres.min()),
        "synth_pres_max": float(syn_pres.max()),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case_id subset (default: all 10 restored)",
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=None,
        help="Override the auto-computed inside-body sample count",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cases is not None:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(REQUIRED_RESTORED_CASE_IDS)

    print(f"Synthesizing inside-body samples for {len(cases)} cases: {cases}")
    print(f"  DATA_ROOT     = {DATA_ROOT}")
    print(f"  RAW_DIR       = {RAW_DIR}")
    print(f"  TARGET_NEG_FR = {TARGET_NEG_FRAC:.2e}")
    print(f"  INSIDE_THRESH = {INSIDE_THRESH}")
    print(f"  KNN_K         = {KNN_K}")
    print(f"  dry_run       = {args.dry_run}")

    results = []
    t_start = time.time()
    for i, case_id in enumerate(cases):
        try:
            print(f"\n--- {i+1}/{len(cases)}: {case_id} ---", flush=True)
            r = synthesize_one(case_id, args.dry_run, n_target=args.n_target)
            results.append(r)
        except Exception as exc:
            logging.exception("%s FAILED", case_id)
            results.append({"case_id": case_id, "error": str(exc)})
    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")

    print("\n=== SUMMARY ===")
    print(
        f"{'case_id':<12} {'n_orig':>10} {'n_new':>10} {'orig_neg_frac':>14} "
        f"{'new_neg_frac':>14} {'sdf_min':>10} {'elapsed':>10}"
    )
    for r in results:
        if "error" in r:
            print(f"  {r['case_id']}: ERROR {r['error']}")
        else:
            print(
                f"  {r['case_id']:<10} {r['n_orig']:>10} {r['n_new']:>10} "
                f"{r['orig_neg_frac']:>14.4e} {r['new_neg_frac']:>14.4e} "
                f"{r['new_sdf_min']:>10.4f} {r['elapsed_sec']:>10.1f}s"
            )


if __name__ == "__main__":
    main()
