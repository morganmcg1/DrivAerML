"""Compute train-split mean/std stats for surface_curvature_proxy_k16_v1.npy.

Writes curvature_proxy_stats_k16_v1.json at the repo root with channel-wise
mean and std, used by load_curvature_stats() to z-score the proxy at training
time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CURVATURE_CACHE_FILENAME = "surface_curvature_proxy_k16_v1.npy"
CURVATURE_STATS_FILENAME = "curvature_proxy_stats_k16_v1.json"
CURVATURE_DIM = 3
DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "data" / "split_manifest.json"


def case_root_from_manifest(manifest: dict) -> Path:
    for candidate in manifest.get("case_root_candidates", []):
        path = Path(candidate)
        if path.exists():
            return path
    primary = manifest.get("case_root")
    if primary:
        return Path(primary)
    raise RuntimeError("No valid case_root found in manifest.")


def compute_stats(case_root: Path, train_cases: list[str]) -> dict[str, list[float]]:
    n = 0
    sum_ = np.zeros(CURVATURE_DIM, dtype=np.float64)
    sum_sq = np.zeros(CURVATURE_DIM, dtype=np.float64)
    for idx, case_id in enumerate(train_cases):
        cache_path = case_root / case_id / CURVATURE_CACHE_FILENAME
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing curvature cache: {cache_path}")
        arr = np.load(cache_path).astype(np.float64)
        if arr.ndim != 2 or arr.shape[1] != CURVATURE_DIM:
            raise RuntimeError(f"Unexpected curvature shape for {case_id}: {arr.shape}")
        n += arr.shape[0]
        sum_ += arr.sum(axis=0)
        sum_sq += (arr * arr).sum(axis=0)
        if (idx + 1) % 25 == 0:
            print(f"  processed {idx + 1}/{len(train_cases)} cases, n={n}")
    mean = sum_ / max(n, 1)
    variance = sum_sq / max(n, 1) - mean * mean
    std = np.sqrt(np.maximum(variance, 0.0))
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "count": int(n),
        "train_cases": len(train_cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--case-root", default=None, help="Override case_root path")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / CURVATURE_STATS_FILENAME))
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    case_root = Path(args.case_root) if args.case_root else case_root_from_manifest(manifest)
    print(f"case_root: {case_root}")
    train_cases = manifest["surface_splits"]["train"]
    print(f"train cases: {len(train_cases)}")

    stats = compute_stats(case_root, train_cases)
    print(f"mean: {stats['mean']}")
    print(f"std: {stats['std']}")
    print(f"count: {stats['count']}")

    out = Path(args.out)
    with out.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
