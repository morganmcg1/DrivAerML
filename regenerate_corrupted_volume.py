# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Regenerate the volume arrays for the 10 REQUIRED_RESTORED cases.

Root cause: `merge_vtu_parts` in tcapelle-cfd/scripts/process_drivaerml.py only
concatenates VTU parts 00 + 01. The 10 restored cases each have 3 parts because
their raw VTU exceeds 50 GB; the third part was silently dropped, so the merged
VTU was truncated at exactly 50 GiB. Cells from the missing tail (which include
inside-body cells in the CFD mesh) never reached `volume_*.npy`, hence the SDF
has effectively no negative branch.

This script:
  1. Re-merges all 3 VTU parts for each corrupted case (appending part 02 to the
     existing parts-00+01 file) — the previous merge is verifiable as
     parts-00+01 by file size, so appending part 02 is a one-shot fix.
  2. Re-runs the canonical volume pipeline (cell-centres + 10% deterministic
     sample + signed implicit distance to the STL) to regenerate
     volume_xyz, volume_pressure, volume_velocity, volume_totalpcoeff,
     volume_indices, volume_sdf.
  3. Replaces the existing symlinks in the active processed dir with regular
     .npy files, leaving the backup directory untouched.

The canonical custom VTU reader is copied from tcapelle-cfd/scripts/process_drivaerml.py
because the standard pyvista reader chokes on > ~4GB inline base64 blobs.
"""

from __future__ import annotations

import argparse
import base64
import logging
import mmap
import os
import re
import struct
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from data.loader import REQUIRED_RESTORED_CASE_IDS

DATA_ROOT = Path("/mnt/new-pvc/Processed/drivaerml_processed")
RAW_DIR = Path("/mnt/new-pvc/Datasets/2_Drivearml")

VTU_PRESSURE_FIELD = "pMeanTrim"
VTU_VELOCITY_FIELD = "UMeanTrim"
VTU_TOTALP_FIELD = "CptMeanTrim"
SAMPLE_RATIO = 0.1  # canonical sample ratio


# --- Custom VTU reader (extracted from tcapelle-cfd/scripts/process_drivaerml.py) ---

_VTK_DTYPE_MAP: dict[str, type] = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int32": np.int32,
    "Int64": np.int64,
    "UInt32": np.uint32,
    "UInt64": np.uint64,
    "UInt8": np.uint8,
    "Int8": np.int8,
}


def _locate_vtu_dataarrays(mm: mmap.mmap) -> dict:
    """Locate all binary DataArray tags by name.

    NOTE: VTU files for OpenFOAM CFD output may include BOTH CellData and PointData
    with the same field names (e.g. pMeanTrim cell-centered AND point-interpolated).
    For CFD volume processing, we want CellData (one value per cell). Cell data is
    written before point data in the OpenFOAM output for these files, so we keep
    the FIRST occurrence of each name. The structural arrays (Points, connectivity,
    offsets) are unique and never duplicated.
    """
    tag_re = re.compile(rb"<DataArray\b([^>]*)>")
    end_tag = b"</DataArray>"
    result: dict[str, dict] = {}
    for m in tag_re.finditer(mm):
        attrs = m.group(1)

        def _attr(key: bytes) -> bytes:
            am = re.search(rb"(?:^|\s)" + key + rb"=['\"]([^'\"]*)['\"]", attrs)
            return am.group(1) if am else b""

        fmt = _attr(b"format")
        if fmt != b"binary":
            continue
        name = _attr(b"Name").decode("utf-8", errors="replace")
        if not name:
            continue
        if name in result:
            # First occurrence wins — see docstring.
            continue

        tag_end = m.end()
        data_start = tag_end
        while data_start < len(mm) and mm[data_start : data_start + 1] in (b"\n", b"\r", b" ", b"\t"):
            data_start += 1
        data_end = mm.find(end_tag, data_start)
        if data_end == -1:
            continue
        b64_end = data_end
        while b64_end > data_start and mm[b64_end - 1 : b64_end] in (b"\n", b"\r", b" ", b"\t"):
            b64_end -= 1

        result[name] = {
            "dtype": _attr(b"type").decode() or "Float32",
            "ncomp": int(_attr(b"NumberOfComponents") or b"1"),
            "b64_start": data_start,
            "b64_end": b64_end,
        }
    return result


def _decode_vtu_array(mm: mmap.mmap, info: dict) -> np.ndarray:
    CHUNK = 3 * 1024 * 1024
    dtype = _VTK_DTYPE_MAP.get(info["dtype"], np.float32)
    ncomp = info["ncomp"]
    b64_start, b64_end = info["b64_start"], info["b64_end"]

    chunks: list[bytes] = []
    pos = b64_start
    while pos < b64_end:
        n = min((CHUNK * 4 + 2) // 3, b64_end - pos)
        n = max(4, (n // 4) * 4)
        chunks.append(base64.b64decode(bytes(mm[pos : pos + n])))
        pos += n

    raw = b"".join(chunks)
    byte_count = struct.unpack_from("<Q", raw)[0]
    arr = np.frombuffer(raw[8 : 8 + byte_count], dtype=dtype).copy()
    return arr.reshape(-1, ncomp) if ncomp > 1 else arr


def read_vtu_volume(vtu_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read a large VTU file and return (cell_centres, field_arrays).

    field_arrays keys: VTU_PRESSURE_FIELD, VTU_VELOCITY_FIELD, VTU_TOTALP_FIELD.
    """
    needed_fields = [VTU_PRESSURE_FIELD, VTU_VELOCITY_FIELD, VTU_TOTALP_FIELD]
    print(f"  reading VTU: {vtu_path} ({vtu_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    with open(vtu_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        arrays = _locate_vtu_dataarrays(mm)
        missing = [f for f in needed_fields + ["Points", "connectivity", "offsets"] if f not in arrays]
        if missing:
            raise RuntimeError(f"VTU {vtu_path} is missing required arrays: {missing}; have {list(arrays)}")

        points = _decode_vtu_array(mm, arrays["Points"])
        connectivity = _decode_vtu_array(mm, arrays["connectivity"]).astype(np.int64)
        offsets = _decode_vtu_array(mm, arrays["offsets"]).astype(np.int64)
        starts = np.empty_like(offsets)
        starts[0] = 0
        starts[1:] = offsets[:-1]
        lengths = offsets - starts

        n_cells = len(offsets)
        centers = np.empty((n_cells, 3), dtype=np.float32)
        for dim in range(3):
            col = points[:, dim][connectivity]
            centers[:, dim] = np.add.reduceat(col.astype(np.float64), starts) / lengths

        field_data: dict[str, np.ndarray] = {}
        for fname in needed_fields:
            field_data[fname] = _decode_vtu_array(mm, arrays[fname])
    finally:
        mm.close()

    return centers, field_data


# --- Re-merge & process ---

def remerge_vtu_parts(case_id: str, dry_run: bool) -> Path:
    """Append part 02 to the existing parts-00+01 merged file. Idempotent."""
    run_id = int(case_id.split("_")[1])
    case_dir = RAW_DIR / case_id
    merged = case_dir / f"volume_{run_id}.vtu"
    parts = sorted(case_dir.glob(f"volume_{run_id}.vtu.*.part"))
    if len(parts) < 2:
        raise RuntimeError(f"Expected at least 2 parts for {case_id}, got {parts}")

    parts_size = sum(p.stat().st_size for p in parts)
    merged_size = merged.stat().st_size if merged.exists() else 0
    print(
        f"  parts: {[p.name for p in parts]}, sizes={[p.stat().st_size for p in parts]}, "
        f"sum={parts_size}, merged_size={merged_size}",
        flush=True,
    )
    if merged.exists() and merged_size == parts_size:
        print(f"  merged VTU is already complete (parts_size = merged_size = {parts_size})", flush=True)
        return merged

    if not merged.exists():
        # Build from scratch (parts 00 + 01 + ... + N)
        print(f"  no merged file, building from {len(parts)} parts", flush=True)
        if dry_run:
            return merged
        tmp = merged.with_suffix(".vtu.tmp")
        with tmp.open("wb") as out:
            for p in parts:
                with p.open("rb") as src:
                    while True:
                        chunk = src.read(64 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
        os.replace(tmp, merged)
        return merged

    # Merged exists but is short — verify it matches parts 00+01 exactly
    parts_00_01 = sum(p.stat().st_size for p in parts[:2])
    if merged_size != parts_00_01:
        raise RuntimeError(
            f"{case_id}: merged file size {merged_size} does not match parts[0:2] sum {parts_00_01}; "
            f"refusing to append (rebuild from scratch instead)"
        )
    # Append parts 02, 03, ...
    print(f"  merged file matches parts 00+01 ({parts_00_01} bytes); appending parts {[p.name for p in parts[2:]]}", flush=True)
    if dry_run:
        return merged
    with merged.open("ab") as out:
        for p in parts[2:]:
            with p.open("rb") as src:
                while True:
                    chunk = src.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    new_size = merged.stat().st_size
    if new_size != parts_size:
        raise RuntimeError(
            f"{case_id}: appended merge size {new_size} != parts_size {parts_size}"
        )
    return merged


def process_one_case(case_id: str, dry_run: bool) -> dict:
    run_id = int(case_id.split("_")[1])
    out_dir = DATA_ROOT / case_id
    raw_case_dir = RAW_DIR / case_id

    print(f"\n=== {case_id} ===", flush=True)
    t0 = time.time()
    # Step 1: re-merge VTU parts
    vtu_path = remerge_vtu_parts(case_id, dry_run)

    # Step 2: read VTU (cell centers + fields)
    centers, fields = read_vtu_volume(vtu_path)
    pressure = np.asarray(fields[VTU_PRESSURE_FIELD], dtype=np.float32)
    velocity = np.asarray(fields[VTU_VELOCITY_FIELD], dtype=np.float32)
    totalp = np.asarray(fields[VTU_TOTALP_FIELD], dtype=np.float32)
    if pressure.ndim == 1:
        pressure = pressure[:, None]
    if velocity.ndim == 1:
        velocity = velocity[:, None]
    if totalp.ndim == 1:
        totalp = totalp[:, None]
    n_cells = centers.shape[0]
    print(
        f"  cells={n_cells}, pressure.shape={pressure.shape}, "
        f"velocity.shape={velocity.shape}, totalp.shape={totalp.shape}",
        flush=True,
    )

    # Step 3: deterministic 10% sample (rng=run_id)
    rng = np.random.default_rng(run_id)
    sample_count = max(1, int(n_cells * SAMPLE_RATIO))
    indices = rng.permutation(n_cells)[:sample_count]
    indices.sort()
    print(f"  sampled {sample_count}/{n_cells} cells with rng=run_id", flush=True)

    sampled_xyz = centers[indices]
    sampled_pressure = pressure[indices]
    sampled_velocity = velocity[indices]
    sampled_totalp = totalp[indices]

    # Step 4: SDF on sampled cell centres (unchanged from canonical pipeline)
    print("  reading STL and computing implicit distance...", flush=True)
    stl_path = raw_case_dir / f"drivaer_{run_id}.stl"
    stl = pv.read(str(stl_path))
    if not isinstance(stl, pv.PolyData):
        stl = stl.extract_surface()
    stl = stl.triangulate()

    query = pv.PolyData(sampled_xyz.astype(np.float32))
    query = query.compute_implicit_distance(stl, inplace=False)
    sdf = np.asarray(query.point_data["implicit_distance"], dtype=np.float32)

    sdf_min, sdf_max = float(sdf.min()), float(sdf.max())
    sdf_neg_frac = float((sdf < 0.0).mean())
    sdf_neg_005_frac = float((sdf < -0.05).mean())
    print(
        f"  NEW sdf: n={sample_count}, min={sdf_min:.4f}, max={sdf_max:.4f}, "
        f"neg_frac={sdf_neg_frac:.4e}, neg<-0.05_frac={sdf_neg_005_frac:.4e}",
        flush=True,
    )

    if dry_run:
        print(f"  [DRY] would save volume_{{xyz,pressure,velocity,totalpcoeff,sdf,indices}}.npy to {out_dir}", flush=True)
    else:
        # Step 5: write all volume arrays atomically
        targets = {
            "volume_xyz.npy": sampled_xyz.astype(np.float32),
            "volume_pressure.npy": sampled_pressure.astype(np.float32),
            "volume_velocity.npy": sampled_velocity.astype(np.float32),
            "volume_totalpcoeff.npy": sampled_totalp.astype(np.float32),
            "volume_sdf.npy": sdf.astype(np.float32),
            "volume_indices.npy": indices.astype(np.int64),
        }
        for fname, arr in targets.items():
            out_path = out_dir / fname
            # np.save appends .npy if missing, so use a path that already ends in .npy
            tmp_path = out_dir / (fname.removesuffix(".npy") + ".tmp.npy")
            np.save(tmp_path, arr)
            assert tmp_path.exists(), f"np.save did not produce {tmp_path}"
            if out_path.is_symlink() or out_path.exists():
                os.remove(out_path)
            os.replace(tmp_path, out_path)
        print(f"  wrote {len(targets)} npy files to {out_dir}", flush=True)

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s", flush=True)

    return {
        "case_id": case_id,
        "n_cells_total": int(n_cells),
        "n_cells_sampled": int(sample_count),
        "sdf_min": sdf_min,
        "sdf_max": sdf_max,
        "sdf_neg_frac": sdf_neg_frac,
        "sdf_neg_005_frac": sdf_neg_005_frac,
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
    args = parser.parse_args()

    if args.cases is not None:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(REQUIRED_RESTORED_CASE_IDS)

    print(f"Regenerating volume_*.npy for {len(cases)} cases: {cases}")
    print(f"  DATA_ROOT = {DATA_ROOT}")
    print(f"  RAW_DIR   = {RAW_DIR}")
    print(f"  dry_run   = {args.dry_run}")

    results = []
    t_start = time.time()
    for i, case_id in enumerate(cases):
        try:
            r = process_one_case(case_id, args.dry_run)
            results.append(r)
        except Exception as exc:
            logging.exception(f"{case_id} FAILED")
            results.append({"case_id": case_id, "error": str(exc)})
    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")

    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"  {r['case_id']}: ERROR {r['error']}")
        else:
            print(
                f"  {r['case_id']}: n={r['n_cells_sampled']}/{r['n_cells_total']} "
                f"sdf_min={r['sdf_min']:.4f} sdf_neg_frac={r['sdf_neg_frac']:.2e} "
                f"sdf<-0.05_frac={r['sdf_neg_005_frac']:.2e} ({r['elapsed_sec']:.0f}s)"
            )


if __name__ == "__main__":
    main()
