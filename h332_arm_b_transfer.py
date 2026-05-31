# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H332 Arm B: apply seed-1 H312 alpha/beta to seed-2 calibration sufficient stats.

Loads the CalibrationStats dumped by --save-calibration-stats, applies the
seed-1 H312 (alpha, beta) per channel and recomputes the rel_l2 family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eval_tta_h252 import CalibrationStats, compute_calibrated_metrics, fit_affine_per_channel


# Seed-1 H312 calibration coefficients, pulled from W&B run enf61qrr summary
# (calibration/weight_noise_mirror_res_avg/{alpha,beta}_*).
H312_SEED1 = {
    "alpha_surf": torch.tensor(
        [
            0.9948879107893270,  # cp
            0.9943967724085607,  # tau_x
            0.9940827941471600,  # tau_y
            0.9917222517819623,  # tau_z
        ],
        dtype=torch.float64,
    ),
    "beta_surf": torch.tensor(
        [
            -0.0023220681890909756,  # cp
            -0.0070268912492144064,  # tau_x
            -0.0002359419212976590,  # tau_y
            -0.0001458997813937601,  # tau_z
        ],
        dtype=torch.float64,
    ),
    "alpha_vol": torch.tensor([0.9996868405490413], dtype=torch.float64),
    "beta_vol": torch.tensor([-0.8328109532412498], dtype=torch.float64),
}


def load_calibration_stats(path: Path) -> CalibrationStats:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cal = CalibrationStats()
    cal.surf_global = blob["surf_global"]
    cal.vol_global = blob["vol_global"]
    cal.surf_per_case = blob["surf_per_case"]
    cal.vol_per_case = blob["vol_per_case"]
    return cal


def report(name: str, metrics: dict[str, float]) -> None:
    keys = (
        "abupt_axis_mean_rel_l2_pct",
        "surface_pressure_rel_l2_pct",
        "wall_shear_rel_l2_pct",
        "wall_shear_x_rel_l2_pct",
        "wall_shear_y_rel_l2_pct",
        "wall_shear_z_rel_l2_pct",
        "volume_pressure_rel_l2_pct",
    )
    print(f"\n[{name}]")
    for k in keys:
        print(f"  {k:<36s}  {metrics[k]:>8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cal-stats-dir",
        default="outputs/h332_primary/calibration_stats",
        help="Directory containing val_surface__*.pt and test_surface__*.pt",
    )
    parser.add_argument(
        "--mode",
        default="weight_noise_mirror_res_avg",
        help="Eval mode suffix matching the saved filenames",
    )
    args = parser.parse_args()

    cal_dir = Path(args.cal_stats_dir)
    val_path = cal_dir / f"val_surface__{args.mode}.pt"
    test_path = cal_dir / f"test_surface__{args.mode}.pt"

    val_cal = load_calibration_stats(val_path)
    test_cal = load_calibration_stats(test_path)

    print(f"Loaded val_cal: {len(val_cal.surf_per_case)} surface cases, "
          f"{len(val_cal.vol_per_case)} volume cases")
    print(f"Loaded test_cal: {len(test_cal.surf_per_case)} surface cases, "
          f"{len(test_cal.vol_per_case)} volume cases")

    # Refit seed-2's own alpha/beta from the val sufficient stats and report —
    # cross-check vs the in-run values logged to W&B.
    alpha_surf_s2, beta_surf_s2 = fit_affine_per_channel(val_cal.surf_global)
    alpha_vol_s2, beta_vol_s2 = fit_affine_per_channel(val_cal.vol_global)

    print("\n=== Seed-2 own-fit calibration (from saved sufficient stats) ===")
    surf_names = ["cp", "tau_x", "tau_y", "tau_z"]
    for c, name in enumerate(surf_names):
        print(
            f"  surface[{name:6s}]  alpha={alpha_surf_s2[c].item():+.6f} "
            f"beta={beta_surf_s2[c].item():+.6f}"
        )
    print(
        f"  volume[volume_pressure]  alpha={alpha_vol_s2[0].item():+.6f} "
        f"beta={beta_vol_s2[0].item():+.6f}"
    )

    print("\n=== Seed-1 H312 alpha/beta (transfer source) ===")
    for c, name in enumerate(surf_names):
        print(
            f"  surface[{name:6s}]  alpha={H312_SEED1['alpha_surf'][c].item():+.6f} "
            f"beta={H312_SEED1['beta_surf'][c].item():+.6f}"
        )
    print(
        f"  volume[volume_pressure]  alpha={H312_SEED1['alpha_vol'][0].item():+.6f} "
        f"beta={H312_SEED1['beta_vol'][0].item():+.6f}"
    )

    print("\n=== Per-channel alpha delta (seed-2 minus seed-1) ===")
    for c, name in enumerate(surf_names):
        d = alpha_surf_s2[c].item() - H312_SEED1["alpha_surf"][c].item()
        d_rel = d / H312_SEED1["alpha_surf"][c].item()
        print(f"  alpha[{name:6s}]  delta={d:+.6f}  rel={d_rel*100:+.4f}%")
    d = alpha_vol_s2[0].item() - H312_SEED1["alpha_vol"][0].item()
    d_rel = d / H312_SEED1["alpha_vol"][0].item()
    print(f"  alpha[volume_pressure]  delta={d:+.6f}  rel={d_rel*100:+.4f}%")

    # ---- Arm B: apply seed-1 H312 cal to seed-2 sufficient stats. ----
    arm_b_val = compute_calibrated_metrics(
        val_cal,
        H312_SEED1["alpha_surf"],
        H312_SEED1["beta_surf"],
        H312_SEED1["alpha_vol"],
        H312_SEED1["beta_vol"],
    )
    arm_b_test = compute_calibrated_metrics(
        test_cal,
        H312_SEED1["alpha_surf"],
        H312_SEED1["beta_surf"],
        H312_SEED1["alpha_vol"],
        H312_SEED1["beta_vol"],
    )

    report("Arm B (seed-2 + seed-1 H312 cal) val", arm_b_val)
    report("Arm B (seed-2 + seed-1 H312 cal) test", arm_b_test)

    # ---- Arm C: seed-2 + seed-2 own cal (refit), for cross-check. ----
    arm_c_val = compute_calibrated_metrics(
        val_cal, alpha_surf_s2, beta_surf_s2, alpha_vol_s2, beta_vol_s2
    )
    arm_c_test = compute_calibrated_metrics(
        test_cal, alpha_surf_s2, beta_surf_s2, alpha_vol_s2, beta_vol_s2
    )
    report("Arm C (seed-2 + own cal, refit cross-check) val", arm_c_val)
    report("Arm C (seed-2 + own cal, refit cross-check) test", arm_c_test)

    # ---- Arm A (raw): identity affine, prints for completeness. ----
    identity_alpha_surf = torch.ones(4, dtype=torch.float64)
    identity_beta_surf = torch.zeros(4, dtype=torch.float64)
    identity_alpha_vol = torch.ones(1, dtype=torch.float64)
    identity_beta_vol = torch.zeros(1, dtype=torch.float64)
    arm_a_val = compute_calibrated_metrics(
        val_cal,
        identity_alpha_surf,
        identity_beta_surf,
        identity_alpha_vol,
        identity_beta_vol,
    )
    arm_a_test = compute_calibrated_metrics(
        test_cal,
        identity_alpha_surf,
        identity_beta_surf,
        identity_alpha_vol,
        identity_beta_vol,
    )
    report("Arm A (seed-2 raw / no cal) val", arm_a_val)
    report("Arm A (seed-2 raw / no cal) test", arm_a_test)


if __name__ == "__main__":
    main()
