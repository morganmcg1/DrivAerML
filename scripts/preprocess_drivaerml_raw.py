# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Build a canonical raw-only DrivAerML processed root.

This script intentionally does not synthesize inside-body samples. It derives
volume arrays from complete raw `volume_i.vtu` files and computes nonnegative
surface distance at the sampled VTU cell centers against the corresponding STL
surface.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gc
import json
import mmap
import os
import re
import shutil
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SOURCE_ROOT = Path("/mnt/new-pvc/Processed/drivaerml_processed")
DEFAULT_RAW_ROOT = Path("/mnt/new-pvc/Datasets/2_Drivearml_fixed_20260511")
DEFAULT_OUTPUT_ROOT = Path("/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511")

PRESSURE_FIELD = "pMeanTrim"
VELOCITY_FIELD = "UMeanTrim"
TOTAL_PRESSURE_FIELD = "CptMeanTrim"
SAMPLE_RATIO = 0.1
PROCESSOR_VERSION = "rawcanon-v2"
SDF_NEAR_SURFACE_ABS = 0.1
VOLUME_OUTPUTS = {
    "volume_xyz.npy",
    "volume_pressure.npy",
    "volume_velocity.npy",
    "volume_totalpcoeff.npy",
    "volume_sdf.npy",
    "volume_indices.npy",
}
VTK_DTYPE_MAP: dict[str, type[np.generic]] = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int32": np.int32,
    "Int64": np.int64,
    "UInt32": np.uint32,
    "UInt64": np.uint64,
    "UInt8": np.uint8,
    "Int8": np.int8,
}


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    split: str

    @property
    def run_id(self) -> int:
        return int(self.case_id.split("_")[1])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_artifact_path(path: Path) -> Path:
    candidates = [path]
    if path.is_symlink():
        raw_target = os.readlink(path)
        target = Path(raw_target)
        if not target.is_absolute():
            target = path.parent / target
        candidates.append(target)
        text = str(target)
        if text.startswith("/rsyncd-munged/"):
            stripped = text.removeprefix("/rsyncd-munged/").lstrip("/")
            candidates.append(Path("/" + stripped))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve artifact path: {path}")


def link_or_copy(src: Path, dst: Path, overwrite: bool) -> None:
    resolved = resolve_artifact_path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    try:
        os.link(resolved, dst)
    except FileExistsError:
        if not overwrite:
            return
        raise
    except OSError:
        shutil.copy2(resolved, dst)


def copy_root_files(source_root: Path, output_root: Path, overwrite: bool) -> None:
    skip_names = {
        "normalizers.json",
        "volume_manifest.csv",
        "sdf_qa.csv",
        "SDF_QA_REPORT.md",
        "preprocessing_provenance.json",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for entry in source_root.iterdir():
        if entry.is_dir() or entry.name in skip_names:
            continue
        link_or_copy(entry, output_root / entry.name, overwrite=overwrite)


def prepare_case_directory(source_root: Path, output_root: Path, case_id: str, overwrite: bool) -> None:
    source_dir = source_root / case_id
    output_dir = output_root / case_id
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source case directory: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in source_dir.iterdir():
        if entry.is_dir() or entry.name.startswith("volume_"):
            continue
        link_or_copy(entry, output_dir / entry.name, overwrite=overwrite)


def load_cases(source_root: Path, cases_arg: str) -> list[CaseInput]:
    rows = read_csv_rows(source_root / "manifest.csv")
    cases = [CaseInput(row["case_id"], row["split"]) for row in rows]
    if cases_arg:
        wanted = {case.strip() for case in cases_arg.split(",") if case.strip()}
        cases = [case for case in cases if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in cases})
        if missing:
            raise ValueError(f"Requested cases are not in manifest.csv: {missing}")
    return cases


def select_shard(cases: list[CaseInput], shard_index: int, shard_count: int) -> list[CaseInput]:
    if shard_count <= 1:
        return cases
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard-index must be in [0, {shard_count}), got {shard_index}")
    return [case for index, case in enumerate(cases) if index % shard_count == shard_index]


def locate_vtu_dataarrays(mm: mmap.mmap) -> dict[str, dict[str, Any]]:
    tag_re = re.compile(rb"<DataArray\b([^>]*)>")
    end_tag = b"</DataArray>"
    result: dict[str, dict[str, Any]] = {}
    for match in tag_re.finditer(mm):
        attrs = match.group(1)

        def attr(key: bytes) -> bytes:
            attr_match = re.search(rb"(?:^|\s)" + key + rb"=['\"]([^'\"]*)['\"]", attrs)
            return attr_match.group(1) if attr_match else b""

        if attr(b"format") != b"binary":
            continue
        name = attr(b"Name").decode("utf-8", errors="replace")
        if not name or name in result:
            continue

        data_start = match.end()
        while data_start < len(mm) and mm[data_start : data_start + 1] in (b"\n", b"\r", b" ", b"\t"):
            data_start += 1
        data_end = mm.find(end_tag, data_start)
        if data_end == -1:
            continue
        b64_end = data_end
        while b64_end > data_start and mm[b64_end - 1 : b64_end] in (b"\n", b"\r", b" ", b"\t"):
            b64_end -= 1

        result[name] = {
            "dtype": attr(b"type").decode() or "Float32",
            "ncomp": int(attr(b"NumberOfComponents") or b"1"),
            "b64_start": data_start,
            "b64_end": b64_end,
        }
    return result


def decode_vtu_array(mm: mmap.mmap, info: dict[str, Any]) -> np.ndarray:
    chunk_size = 3 * 1024 * 1024
    dtype = VTK_DTYPE_MAP.get(str(info["dtype"]), np.float32)
    ncomp = int(info["ncomp"])
    b64_start = int(info["b64_start"])
    b64_end = int(info["b64_end"])

    chunks: list[bytes] = []
    pos = b64_start
    while pos < b64_end:
        n = min((chunk_size * 4 + 2) // 3, b64_end - pos)
        n = max(4, (n // 4) * 4)
        chunks.append(base64.b64decode(bytes(mm[pos : pos + n])))
        pos += n

    raw = b"".join(chunks)
    byte_count = struct.unpack_from("<Q", raw)[0]
    array = np.frombuffer(raw[8 : 8 + byte_count], dtype=dtype).copy()
    del raw
    if ncomp > 1:
        return array.reshape(-1, ncomp)
    return array


def read_vtu_volume(vtu_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    required = ["Points", "connectivity", "offsets", PRESSURE_FIELD, VELOCITY_FIELD, TOTAL_PRESSURE_FIELD]
    with vtu_path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        arrays = locate_vtu_dataarrays(mm)
        missing = [name for name in required if name not in arrays]
        if missing:
            raise RuntimeError(f"VTU {vtu_path} is missing arrays: {missing}; found {sorted(arrays)}")

        points = decode_vtu_array(mm, arrays["Points"])
        connectivity = decode_vtu_array(mm, arrays["connectivity"]).astype(np.int64, copy=False)
        offsets = decode_vtu_array(mm, arrays["offsets"]).astype(np.int64, copy=False)
        starts = np.empty_like(offsets)
        starts[0] = 0
        starts[1:] = offsets[:-1]
        lengths = offsets - starts

        n_cells = int(len(offsets))
        centers = np.empty((n_cells, 3), dtype=np.float32)
        for dim in range(3):
            column = points[:, dim][connectivity]
            centers[:, dim] = np.add.reduceat(column.astype(np.float64), starts) / lengths
            del column
        del points, connectivity, offsets, starts, lengths
        gc.collect()

        fields = {
            PRESSURE_FIELD: decode_vtu_array(mm, arrays[PRESSURE_FIELD]),
            VELOCITY_FIELD: decode_vtu_array(mm, arrays[VELOCITY_FIELD]),
            TOTAL_PRESSURE_FIELD: decode_vtu_array(mm, arrays[TOTAL_PRESSURE_FIELD]),
        }
        for name, array in fields.items():
            if array.shape[0] != n_cells:
                raise RuntimeError(
                    f"VTU {vtu_path} field {name!r} has {array.shape[0]} rows, "
                    f"expected {n_cells} cell rows"
                )
        velocity = fields[VELOCITY_FIELD]
        if velocity.ndim != 2 or velocity.shape[1] != 3:
            raise RuntimeError(
                f"VTU {vtu_path} field {VELOCITY_FIELD!r} must have shape [N, 3], got {velocity.shape}"
            )
    finally:
        mm.close()
    return centers, fields, n_cells


def compute_sdf(stl_path: Path, xyz: np.ndarray, chunk_points: int) -> np.ndarray:
    import pyvista as pv

    stl = pv.read(str(stl_path))
    if not isinstance(stl, pv.PolyData):
        stl = stl.extract_surface()
    stl = stl.triangulate()

    sdf = np.empty(xyz.shape[0], dtype=np.float32)
    for start in range(0, xyz.shape[0], chunk_points):
        stop = min(start + chunk_points, xyz.shape[0])
        query = pv.PolyData(np.asarray(xyz[start:stop], dtype=np.float32))
        query = query.compute_implicit_distance(stl, inplace=False)
        sdf[start:stop] = np.asarray(query.point_data["implicit_distance"], dtype=np.float32)
        print(f"    SDF {stop}/{xyz.shape[0]}", flush=True)
    return sdf


def array_stats(array: np.ndarray) -> dict[str, float | int]:
    flat = array.reshape(-1)
    return {
        "count": int(flat.size),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat, dtype=np.float64)),
        "std": float(np.std(flat, dtype=np.float64)),
    }


def sdf_stats(sdf: np.ndarray) -> dict[str, float | int]:
    stats = array_stats(sdf)
    negative_count = int(np.count_nonzero(sdf < 0.0))
    near_count = int(np.count_nonzero(np.abs(sdf) < SDF_NEAR_SURFACE_ABS))
    stats.update(
        {
            "negative_count": negative_count,
            "negative_frac": float(negative_count / sdf.size),
            "near_surface_abs_0p1_count": near_count,
            "near_surface_abs_0p1_frac": float(near_count / sdf.size),
        }
    )
    return stats


def clamp_sdf(sdf: np.ndarray) -> np.ndarray:
    np.maximum(sdf, 0.0, out=sdf)
    return sdf


def atomic_save(path: Path, array: np.ndarray, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        path.unlink()
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def raw_parts_info(raw_case_dir: Path, run_id: int, merged_vtu: Path) -> dict[str, Any]:
    parts = sorted(raw_case_dir.glob(f"volume_{run_id}.vtu.*.part"))
    parts_size = sum(part.stat().st_size for part in parts)
    merged_size = merged_vtu.stat().st_size
    parts_match_merged = not parts or parts_size == merged_size
    if not parts_match_merged:
        print(
            f"  warning: raw VTU part size mismatch for {merged_vtu}: "
            f"merged={merged_size}, parts={parts_size}, n_parts={len(parts)}; "
            "using merged VTU",
            flush=True,
        )
    return {
        "raw_volume_part_count": len(parts),
        "raw_volume_parts_size": parts_size,
        "raw_volume_parts_match_merged": parts_match_merged,
        "raw_volume_parts_size_delta": parts_size - merged_size,
        "raw_volume_part_names": [part.name for part in parts],
    }


def process_case(
    case: CaseInput,
    source_root: Path,
    raw_root: Path,
    output_root: Path,
    *,
    sample_ratio: float,
    sdf_chunk_points: int,
    overwrite: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    run_id = case.run_id
    output_dir = output_root / case.case_id
    provenance_path = output_dir / "volume_provenance.json"
    if skip_existing and provenance_path.exists():
        return json.loads(provenance_path.read_text())

    prepare_case_directory(source_root, output_root, case.case_id, overwrite=overwrite)

    raw_case_dir = raw_root / case.case_id
    vtu_path = raw_case_dir / f"volume_{run_id}.vtu"
    stl_path = raw_case_dir / f"drivaer_{run_id}.stl"
    if not vtu_path.exists():
        raise FileNotFoundError(f"Missing raw VTU: {vtu_path}")
    if not stl_path.exists():
        raise FileNotFoundError(f"Missing raw STL: {stl_path}")

    print(f"\n=== {case.case_id} ({case.split}) ===", flush=True)
    t0 = time.time()
    parts = raw_parts_info(raw_case_dir, run_id, vtu_path)
    print(f"  reading {vtu_path} ({vtu_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    centers, fields, n_cells = read_vtu_volume(vtu_path)

    sample_count = max(1, int(n_cells * sample_ratio))
    if sample_count == n_cells:
        indices = np.arange(n_cells, dtype=np.int64)
    else:
        rng = np.random.default_rng(run_id)
        indices = rng.permutation(n_cells)[:sample_count]
        indices.sort()
    xyz = np.asarray(centers[indices], dtype=np.float32)
    del centers
    gc.collect()

    pressure = np.asarray(fields[PRESSURE_FIELD][indices], dtype=np.float32)
    velocity = np.asarray(fields[VELOCITY_FIELD][indices], dtype=np.float32)
    total_pressure = np.asarray(fields[TOTAL_PRESSURE_FIELD][indices], dtype=np.float32)
    del fields
    gc.collect()

    if pressure.ndim == 1:
        pressure = pressure[:, None]
    if velocity.ndim == 1:
        velocity = velocity[:, None]
    if total_pressure.ndim == 1:
        total_pressure = total_pressure[:, None]

    print(f"  sampled {sample_count}/{n_cells} cells; computing SDF", flush=True)
    sdf = compute_sdf(stl_path, xyz, sdf_chunk_points)
    sdf_pre_clamp_summary = sdf_stats(sdf)
    sdf = clamp_sdf(sdf)
    sdf_summary = sdf_stats(sdf)
    print(
        "  SDF "
        f"min={sdf_summary['min']:.6g} max={sdf_summary['max']:.6g} "
        f"neg_frac={sdf_summary['negative_frac']:.6g} "
        f"pre_clamp_neg_frac={sdf_pre_clamp_summary['negative_frac']:.6g}",
        flush=True,
    )

    targets = {
        "volume_xyz.npy": xyz,
        "volume_pressure.npy": pressure,
        "volume_velocity.npy": velocity,
        "volume_totalpcoeff.npy": total_pressure,
        "volume_sdf.npy": sdf,
        "volume_indices.npy": indices.astype(np.int64),
    }
    for name, array in targets.items():
        atomic_save(output_dir / name, array, overwrite=overwrite)

    elapsed = time.time() - t0
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "case_path": case.case_id,
        "split": case.split,
        "processor_version": PROCESSOR_VERSION,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "raw_case_dir": str(raw_case_dir),
        "raw_volume_vtu": str(vtu_path),
        "raw_volume_vtu_size": vtu_path.stat().st_size,
        "raw_stl": str(stl_path),
        "raw_stl_size": stl_path.stat().st_size,
        "raw_volume_points": int(n_cells),
        "sampled_volume_points": int(sample_count),
        "sample_ratio": sample_ratio,
        "sample_seed": run_id,
        "surface_distance_method": "pyvista.compute_implicit_distance",
        "surface_distance_postprocess": "clamp_min_0",
        "pressure_field": PRESSURE_FIELD,
        "velocity_field": VELOCITY_FIELD,
        "total_pressure_field": TOTAL_PRESSURE_FIELD,
        "sdf_pre_clamp_min": sdf_pre_clamp_summary["min"],
        "sdf_pre_clamp_max": sdf_pre_clamp_summary["max"],
        "sdf_pre_clamp_mean": sdf_pre_clamp_summary["mean"],
        "sdf_pre_clamp_std": sdf_pre_clamp_summary["std"],
        "sdf_pre_clamp_negative_count": sdf_pre_clamp_summary["negative_count"],
        "sdf_pre_clamp_negative_frac": sdf_pre_clamp_summary["negative_frac"],
        "sdf_min": sdf_summary["min"],
        "sdf_max": sdf_summary["max"],
        "sdf_mean": sdf_summary["mean"],
        "sdf_std": sdf_summary["std"],
        "sdf_negative_count": sdf_summary["negative_count"],
        "sdf_negative_frac": sdf_summary["negative_frac"],
        "sdf_near_surface_abs_0p1_count": sdf_summary["near_surface_abs_0p1_count"],
        "sdf_near_surface_abs_0p1_frac": sdf_summary["near_surface_abs_0p1_frac"],
        "elapsed_sec": elapsed,
        **parts,
    }
    write_json(provenance_path, row)
    print(f"  wrote {output_dir} in {elapsed:.1f}s", flush=True)
    return row


def write_dataset_provenance(
    output_root: Path,
    source_root: Path,
    raw_root: Path,
    cases: list[CaseInput],
    rows: list[dict[str, Any]],
    sample_ratio: float,
) -> None:
    payload = {
        "dataset": "DrivAerML",
        "processor_version": PROCESSOR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_processed_root": str(source_root),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "sample_ratio": sample_ratio,
        "case_count": len(cases),
        "processed_case_count": len(rows),
        "volume_fields": {
            "pressure": PRESSURE_FIELD,
            "velocity": VELOCITY_FIELD,
            "total_pressure": TOTAL_PRESSURE_FIELD,
        },
        "notes": [
            "Surface arrays are resolved from the source processed root and hardlinked or copied.",
            "Volume arrays are regenerated for every case from complete raw VTU files.",
            "No synthetic inside-body samples are added.",
            "SDF values are clamped with max(sdf, 0) after signed-distance computation.",
        ],
    }
    write_json(output_root / "preprocessing_provenance.json", payload)


def collect_existing_provenance(output_root: Path, cases: list[CaseInput]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case in cases:
        provenance_path = output_root / case.case_id / "volume_provenance.json"
        if not provenance_path.exists():
            missing.append(case.case_id)
            continue
        rows.append(json.loads(provenance_path.read_text()))
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases", default="", help="Optional comma-separated case IDs to process.")
    parser.add_argument("--sample-ratio", type=float, default=SAMPLE_RATIO)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--volume-manifest-name", default="volume_manifest.csv")
    parser.add_argument("--sdf-chunk-points", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-root-copy", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError(f"sample-ratio must be in (0, 1], got {args.sample_ratio}")

    all_cases = load_cases(args.source_root, args.cases)
    cases = select_shard(all_cases, args.shard_index, args.shard_count)
    if not cases:
        raise ValueError("No cases selected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_root_copy:
        copy_root_files(args.source_root, args.output_root, overwrite=args.overwrite)

    rows: list[dict[str, Any]] = []
    volume_manifest_name = Path(args.volume_manifest_name)
    if args.shard_count > 1:
        volume_manifest_path = args.output_root / (
            f"{volume_manifest_name.stem}_shard_{args.shard_index}{volume_manifest_name.suffix}"
        )
    else:
        volume_manifest_path = args.output_root / args.volume_manifest_name
    for index, case in enumerate(cases, start=1):
        print(f"\n--- {index}/{len(cases)} ---", flush=True)
        rows.append(
            process_case(
                case,
                args.source_root,
                args.raw_root,
                args.output_root,
                sample_ratio=args.sample_ratio,
                sdf_chunk_points=args.sdf_chunk_points,
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
            )
        )
        write_csv_rows(volume_manifest_path, rows)

    write_csv_rows(volume_manifest_path, rows)
    all_rows, missing = collect_existing_provenance(args.output_root, all_cases)
    if missing:
        print(f"Skipping full root manifest; missing provenance for {len(missing)} cases", flush=True)
    else:
        write_csv_rows(args.output_root / args.volume_manifest_name, all_rows)
        write_dataset_provenance(
            args.output_root,
            args.source_root,
            args.raw_root,
            all_cases,
            all_rows,
            args.sample_ratio,
        )
    print(f"Wrote canonical raw-only processed root: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
