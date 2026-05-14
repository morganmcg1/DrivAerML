# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Per-case mean-error diagnostic for the bias-corrected ensemble (PR #1108).

Computes for each channel c:
    e_k_mean = sum_i w_i * sum_err_i_c_k / N_c_k   (per-case ensemble mean error)
    spread:  std and IQR over cases
    target stat: would a single shared bias help?

Run after bias_corrected_ensemble.py has loaded the cache once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load_data
from trainer_runtime import TargetTransform

from ensemble_eval import (
    load_meta_cache_from_disk,
    load_pred_cache_from_disk,
    meta_cache_path,
    pred_cache_path,
)
from bias_corrected_ensemble import (
    PER_AXIS_CHANNELS,
    channel_std_array,
    precompute_stats_for_split,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-cache-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--manifest", default="data/split_manifest.json")
    p.add_argument(
        "--run-ids",
        nargs="+",
        default=["56bcqp3m", "29nohj67", "a0yoxy85", "ghh0s4ne"],
    )
    p.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.375, 0.250, 0.250, 0.125],
    )
    p.add_argument("--split", default="val", choices=["val", "test"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_ids = list(args.run_ids)
    w = np.asarray(args.weights, dtype=np.float64)

    _, _, _, stats = load_data(
        manifest_path=args.manifest, root=args.data_root,
        train_surface_points=65536, eval_surface_points=65536,
        train_volume_points=65536, eval_volume_points=65536,
    )
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )
    sigma = channel_std_array(stats)

    cache_dir = Path(args.pred_cache_dir)
    batch_meta = load_meta_cache_from_disk(meta_cache_path(cache_dir, args.split))
    pred_cache = {
        rid: load_pred_cache_from_disk(pred_cache_path(cache_dir, rid, args.split))
        for rid in run_ids
    }
    precomp = precompute_stats_for_split(run_ids, batch_meta, pred_cache, transform, device)

    print(f"\n=== Per-case ensemble mean-error spread ({args.split}, {precomp.num_cases()} cases) ===")
    print(f"{'channel':<20} {'mean_e':>10} {'std_e':>10} {'min_e':>10} {'max_e':>10} {'mean_e/sigma':>14} {'std_e/sigma':>12}")
    for i, ch in enumerate(PER_AXIS_CHANNELS):
        sum_err = precomp.sum_err[ch].numpy()   # [num_cases, M]
        N = precomp.N[ch].numpy()                # [num_cases]
        valid = precomp.valid[ch].numpy()
        # Ensemble mean error per case (in physical units)
        e_per_case = (sum_err @ w) / N           # [num_cases]
        e_v = e_per_case[valid]
        sig = sigma[i]
        print(
            f"{ch:<20} {e_v.mean():>10.4f} {e_v.std():>10.4f} {e_v.min():>10.4f} {e_v.max():>10.4f} "
            f"{e_v.mean()/sig:>14.5f} {e_v.std()/sig:>12.5f}"
        )

    # The "best" shared bias for each channel (analytically minimising squared-error sum,
    # not the rel-L2 mean, but still a useful diagnostic):
    print(f"\n=== Per-channel best shared bias (point-weighted) ===")
    print(f"{'channel':<20} {'b_phys':>12} {'b_norm':>10} {'sum_pts':>12}")
    for i, ch in enumerate(PER_AXIS_CHANNELS):
        sum_err = precomp.sum_err[ch].numpy()
        N = precomp.N[ch].numpy()
        # Point-weighted optimal shared bias = -sum_k sum_n e_n / sum_k N_k
        total_e = float((sum_err @ w).sum())
        total_N = float(N.sum())
        b_opt = -total_e / total_N if total_N > 0 else 0.0
        print(
            f"{ch:<20} {b_opt:>12.6f} {b_opt/sigma[i]:>10.6f} {int(total_N):>12d}"
        )


if __name__ == "__main__":
    main()
