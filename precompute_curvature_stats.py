# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Precompute global p5/p95 robust-normalization stats for surface curvature.

H348: per-vertex mean (H), Gaussian (K), and principal (k1, k2) curvatures
on the canonical DrivAer surface mesh. Two precomputed roots are supported:

- ``<root>/<case>.npy`` shape (N_surf, 2): "HK" layout (col 0 = H, col 1 = K)
- ``<root>/<case>.npy`` shape (N_surf, 2): "k1k2" layout (col 0 = k1, col 1 = k2)

Defaults point at the haku precompute (covers all 484 cases):

- ``curvatures_haku_v1``        — (H, K)
- ``curvatures_haku_v1_k1k2``   — (k1, k2)

The script samples a fixed number of points per train case to keep memory
bounded, then computes per-channel p5/p95 / median / mean / std, and persists
``curvature_normalization.pt`` next to ``train.py`` along with a JSON mirror.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.loader import DrivAerMLCaseStore

DEFAULT_OUT = "curvature_normalization.pt"
DEFAULT_HK_ROOT = "/mnt/new-pvc/Processed/curvatures_haku_v1"
DEFAULT_K1K2_ROOT = "/mnt/new-pvc/Processed/curvatures_haku_v1_k1k2"
DEFAULT_POINTS_PER_CASE = 50_000


def _percentile_stats(x: np.ndarray) -> dict:
    p5, p95 = np.percentile(x, [5, 95])
    return {
        "p5": float(p5),
        "p95": float(p95),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "n_samples": int(x.shape[0]),
    }


def _sample_root(
    root: Path,
    case_ids: list[str],
    points_per_case: int,
    rng: np.random.Generator,
    label: str,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    missing = 0
    for i, cid in enumerate(case_ids):
        path = root / f"{cid}.npy"
        if not path.exists():
            missing += 1
            continue
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0]
        k = min(points_per_case, n)
        idx = rng.choice(n, size=k, replace=False)
        idx.sort()
        parts.append(np.asarray(arr[idx], dtype=np.float32))
        if i % 50 == 0:
            print(f"  [{label}] case {i}/{len(case_ids)} {cid}: n={n}", flush=True)
    if missing:
        raise RuntimeError(f"[{label}] missing {missing}/{len(case_ids)} cases at {root}")
    return np.concatenate(parts, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/split_manifest.json")
    p.add_argument("--data-root", default="")
    p.add_argument("--hk-root", default=DEFAULT_HK_ROOT)
    p.add_argument("--k1k2-root", default=DEFAULT_K1K2_ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--points-per-case", type=int, default=DEFAULT_POINTS_PER_CASE)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    store = DrivAerMLCaseStore(manifest_path=args.manifest, root=args.data_root or None)
    train_ids = store.case_ids("train")
    print(f"Sampling curvature from {len(train_ids)} train cases (~{args.points_per_case} pts/case)")

    rng_hk = np.random.default_rng(args.seed)
    rng_k1k2 = np.random.default_rng(args.seed + 1)

    hk = _sample_root(Path(args.hk_root), train_ids, args.points_per_case, rng_hk, "HK")
    k1k2 = _sample_root(Path(args.k1k2_root), train_ids, args.points_per_case, rng_k1k2, "k1k2")

    stats = {
        "H":  _percentile_stats(hk[:, 0]),
        "K":  _percentile_stats(hk[:, 1]),
        "k1": _percentile_stats(k1k2[:, 0]),
        "k2": _percentile_stats(k1k2[:, 1]),
    }
    for name, s in stats.items():
        print(
            f"  {name}: p5={s['p5']:.5g} p95={s['p95']:.5g} "
            f"median={s['median']:.5g} mean={s['mean']:.5g} std={s['std']:.5g}"
        )

    payload = {
        "version": 2,
        "hk_root": args.hk_root,
        "k1k2_root": args.k1k2_root,
        "stats": stats,
    }
    out_path = Path(args.out)
    torch.save(payload, out_path)
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}  ({out_path.with_suffix('.json')} mirror)")


if __name__ == "__main__":
    main()
