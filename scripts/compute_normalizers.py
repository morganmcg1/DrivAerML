# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Compute DrivAerML normalizers from the manifest train case split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from data.loader import DEFAULT_MANIFEST, DrivAerMLCaseStore, _resolve_artifact_path

DEFAULT_FIELDS = (
    "surface_cp",
    "surface_wallshearstress",
    "volume_pressure",
)


def _as_rows(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array[:, None]
    return array.reshape(array.shape[0], -1)


def _accumulate_array(path: Path, chunk_rows: int) -> tuple[int, np.ndarray, np.ndarray]:
    array = np.load(_resolve_artifact_path(path), mmap_mode="r")
    rows = _as_rows(array)
    total = 0
    sums = np.zeros(rows.shape[1], dtype=np.float64)
    sum_squares = np.zeros(rows.shape[1], dtype=np.float64)
    for start in range(0, rows.shape[0], chunk_rows):
        chunk = np.asarray(rows[start : start + chunk_rows], dtype=np.float64)
        total += int(chunk.shape[0])
        sums += chunk.sum(axis=0)
        sum_squares += np.square(chunk).sum(axis=0)
    return total, sums, sum_squares


def _serialize(value: np.ndarray) -> float | list[float]:
    flat = value.reshape(-1)
    if flat.size == 1:
        return float(flat[0])
    return [float(item) for item in flat]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--root", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--chunk-rows", type=int, default=2_000_000)
    parser.add_argument("--field", action="append", default=[])
    args = parser.parse_args()

    store = DrivAerMLCaseStore(manifest_path=args.manifest, root=args.root or None)
    fields = tuple(args.field or DEFAULT_FIELDS)
    totals: dict[str, int] = {}
    sums: dict[str, np.ndarray] = {}
    sum_squares: dict[str, np.ndarray] = {}

    train_ids = store.case_ids("train")
    for index, case_id in enumerate(train_ids, start=1):
        case_dir = store.root / case_id
        for field in fields:
            path = case_dir / f"{field}.npy"
            count, field_sum, field_sum_squares = _accumulate_array(path, args.chunk_rows)
            if field not in totals:
                totals[field] = 0
                sums[field] = np.zeros_like(field_sum)
                sum_squares[field] = np.zeros_like(field_sum_squares)
            totals[field] += count
            sums[field] += field_sum
            sum_squares[field] += field_sum_squares
        if index % 10 == 0 or index == len(train_ids):
            print(f"processed {index}/{len(train_ids)} train cases")

    normalizers = {}
    for field in fields:
        count = totals[field]
        mean = sums[field] / count
        variance = sum_squares[field] / count - np.square(mean)
        std = np.sqrt(np.maximum(variance, 1e-12))
        normalizers[field] = {
            "mean": _serialize(mean),
            "std": _serialize(std),
            "count": int(count),
        }

    out = Path(args.out) if args.out else store.root / "normalizers.json"
    out.write_text(json.dumps(normalizers, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
