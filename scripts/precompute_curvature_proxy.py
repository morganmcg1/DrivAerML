# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Compute global per-channel mean/std for the surface curvature proxy.

The curvature attention bias (introduced in H5 and used through the H39/H147
SOTA stack) requires train-split mean/std of the per-vertex 3-channel
``surface_curvature_proxy_k16_v1.npy`` arrays that live alongside each
processed DrivAerML case. ``trainer_runtime.load_curvature_stats`` reads the
resulting JSON at training time to z-score curvature features.

The script was previously assumed-present but was never committed to the
target repo; ``trainer_runtime`` simply raises ``FileNotFoundError`` with a
hint to run this module. This commit adds the script so curvature-bias runs
can launch from a freshly checked-out clone.

Usage::

    python -m scripts.precompute_curvature_proxy \\
        --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511

The stats are written to ``<repo>/curvature_proxy_stats_k16_v1.json`` next to
``trainer_runtime.py``. Re-running on the same train split is idempotent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = DEFAULT_REPO_ROOT / "data" / "split_manifest.json"
DEFAULT_STATS_PATH = DEFAULT_REPO_ROOT / "curvature_proxy_stats_k16_v1.json"
CURVATURE_CACHE_FILENAME = "surface_curvature_proxy_k16_v1.npy"
CURVATURE_DIM = 3


def resolve_data_root(manifest: dict, explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    candidates = manifest.get("case_root_candidates") or []
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    if "case_root" in manifest and Path(manifest["case_root"]).exists():
        return Path(manifest["case_root"])
    raise FileNotFoundError(
        f"No case root from manifest exists; tried: {candidates}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to data/split_manifest.json",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override case root (defaults to first existing manifest candidate).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_STATS_PATH),
        help="Where to write the curvature stats JSON.",
    )
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    data_root = resolve_data_root(manifest, args.data_root)
    train_cases = list(manifest["surface_splits"]["train"])
    print(f"data_root = {data_root}")
    print(f"train cases = {len(train_cases)}")

    total_count = 0
    sum_x = np.zeros(CURVATURE_DIM, dtype=np.float64)
    sum_x2 = np.zeros(CURVATURE_DIM, dtype=np.float64)
    missing = []
    for i, case_id in enumerate(train_cases):
        path = data_root / case_id / CURVATURE_CACHE_FILENAME
        if not path.exists():
            missing.append(case_id)
            continue
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[-1] != CURVATURE_DIM:
            raise RuntimeError(
                f"{case_id}: expected shape (N, {CURVATURE_DIM}), got {arr.shape}"
            )
        arr64 = np.asarray(arr, dtype=np.float64)
        sum_x += arr64.sum(axis=0)
        sum_x2 += (arr64 * arr64).sum(axis=0)
        total_count += arr64.shape[0]
        if (i + 1) % 50 == 0 or (i + 1) == len(train_cases):
            print(f"  processed {i + 1}/{len(train_cases)} (total points so far: {total_count})")

    if missing:
        raise FileNotFoundError(
            f"Missing curvature cache for {len(missing)} cases; first: {missing[:5]}"
        )
    if total_count == 0:
        raise RuntimeError("No train points accumulated; nothing to write.")

    mean = sum_x / total_count
    var = np.clip(sum_x2 / total_count - mean * mean, 0.0, None)
    std = np.sqrt(var)
    stats = {
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "channels": CURVATURE_DIM,
        "train_case_count": len(train_cases),
        "point_count": int(total_count),
        "data_root": str(data_root),
    }
    out_path = Path(args.out)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
