"""Compute train-split mean/std of the per-vertex curvature proxy.

Reads ``surface_curvature_proxy_k16_v1.npy`` from each train case in the
pinned split manifest and writes ``curvature_proxy_stats_k16_v1.json`` next
to ``trainer_runtime.py`` so ``load_curvature_stats`` can find it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "split_manifest.json"
DEFAULT_STATS_PATH = REPO_ROOT / "curvature_proxy_stats_k16_v1.json"
CURVATURE_FILENAME = "surface_curvature_proxy_k16_v1.npy"
CURVATURE_DIM = 3


def _resolve_root(manifest: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    candidates = [manifest.get("case_root", "")] + list(manifest.get("case_root_candidates", []))
    for cand in candidates:
        if cand and Path(cand).is_dir():
            return Path(cand)
    raise FileNotFoundError(f"No case_root candidate exists. Tried: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override case_root in the manifest (e.g. /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511)")
    parser.add_argument("--output", type=Path, default=DEFAULT_STATS_PATH)
    args = parser.parse_args()

    with args.manifest.open() as f:
        manifest = json.load(f)
    root = _resolve_root(manifest, args.data_root)
    train_cases = manifest["surface_splits"]["train"]
    print(f"Computing curvature stats over {len(train_cases)} train cases under {root}")

    count = 0
    sum_x = np.zeros(CURVATURE_DIM, dtype=np.float64)
    sum_x2 = np.zeros(CURVATURE_DIM, dtype=np.float64)

    for i, case_id in enumerate(train_cases):
        path = root / case_id / CURVATURE_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Missing curvature file for {case_id}: {path}")
        arr = np.load(path).astype(np.float64, copy=False)
        if arr.ndim != 2 or arr.shape[1] != CURVATURE_DIM:
            raise ValueError(f"{path}: expected (V, {CURVATURE_DIM}); got {arr.shape}")
        count += arr.shape[0]
        sum_x += arr.sum(axis=0)
        sum_x2 += (arr * arr).sum(axis=0)
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(train_cases)} cases; running count={count:,}")

    mean = sum_x / count
    var = sum_x2 / count - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    stats = {
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "count": int(count),
        "num_cases": int(len(train_cases)),
        "filename_npy": CURVATURE_FILENAME,
        "case_root": str(root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {args.output}")
    print(f"  mean = {stats['mean']}")
    print(f"  std  = {stats['std']}")
    print(f"  total surface points = {count:,}")


if __name__ == "__main__":
    main()
