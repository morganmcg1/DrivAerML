#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Compute train-split volume_velocity target normalizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    train_ids = manifest["surface_splits"]["train"]
    root = Path(args.data_root)
    total = 0
    sums = np.zeros(3, dtype=np.float64)
    sum_squares = np.zeros(3, dtype=np.float64)

    for index, case_id in enumerate(train_ids, start=1):
        path = root / case_id / "volume_velocity.npy"
        values = np.load(path, mmap_mode="r")
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{path} must have shape [N, 3], got {values.shape}")
        for start in range(0, values.shape[0], args.chunk_rows):
            chunk = np.asarray(values[start : start + args.chunk_rows], dtype=np.float64)
            total += int(chunk.shape[0])
            sums += chunk.sum(axis=0)
            sum_squares += np.square(chunk).sum(axis=0)
        print(f"{index:03d}/{len(train_ids)} {case_id} rows={values.shape[0]}", flush=True)

    mean = sums / total
    variance = np.maximum(sum_squares / total - np.square(mean), 0.0)
    std = np.sqrt(variance)
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": int(total),
    }
    print(json.dumps({"volume_velocity": stats}, indent=2))

    if args.write:
        normalizers_path = root / "normalizers.json"
        raw = json.loads(normalizers_path.read_text())
        raw["volume_velocity"] = stats
        normalizers_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        print(f"wrote {normalizers_path}")


if __name__ == "__main__":
    main()
