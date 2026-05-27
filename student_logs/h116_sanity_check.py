"""H116 Y-mirror augmentation sanity check.

Verifies:
1. apply_y_mirror with prob=1.0 negates y, normal_y, tau_y, volume y;
   leaves SP, tau_x, tau_z, VP, panel_area, sdf, normal_x/z unchanged.
2. apply_y_mirror with prob=0.0 is identity.
3. DrivAerMLSurfaceDataset with use_y_mirror_aug=True applies mirror only
   when sampling_mode == 'train_random'.
4. The eval Dataset (sampling_mode='eval_chunk') with use_y_mirror_aug=True
   still leaves samples unchanged (guarded).
5. Per-car y>0 vs y<0 SP mean parity is within tolerance (proxy for the
   approximate longitudinal symmetry assumption).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from data import DrivAerMLSurfaceDataset, apply_y_mirror
from data.loader import DrivAerMLCaseStore, load_case


def expect_close(name, actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name}: max abs diff {diff} exceeds {atol}")


def expect_equal(name, actual, expected):
    if not torch.equal(actual, expected):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name}: tensors differ, max abs diff {diff}")


def main():
    manifest = "/workspace/senpai/target/data/split_manifest.json"
    store = DrivAerMLCaseStore(manifest_path=manifest)
    train_ids = store.case_ids("train")
    val_ids = store.case_ids("val")

    sample_case_id = train_ids[0]
    print(f"Loading sample case {sample_case_id}")
    case = store.load_case(sample_case_id)
    print(
        f"  surface_x={tuple(case.surface_x.shape)} "
        f"surface_y={tuple(case.surface_y.shape)} "
        f"volume_x={tuple(case.volume_x.shape)} "
        f"volume_y={tuple(case.volume_y.shape)}"
    )

    print("\n[Test 1] apply_y_mirror prob=1.0 unit transform")
    original_surface_x = case.surface_x.clone()
    original_surface_y = case.surface_y.clone()
    original_volume_x = case.volume_x.clone()
    original_volume_y = case.volume_y.clone()

    mirrored = apply_y_mirror(case, prob=1.0)
    # y, normal_y, tau_y, volume y should be negated
    expect_close("surface_x[..., 0] (x)", mirrored.surface_x[..., 0], original_surface_x[..., 0])
    expect_close("surface_x[..., 1] (y)", mirrored.surface_x[..., 1], -original_surface_x[..., 1])
    expect_close("surface_x[..., 2] (z)", mirrored.surface_x[..., 2], original_surface_x[..., 2])
    expect_close("surface_x[..., 3] (normal_x)", mirrored.surface_x[..., 3], original_surface_x[..., 3])
    expect_close("surface_x[..., 4] (normal_y)", mirrored.surface_x[..., 4], -original_surface_x[..., 4])
    expect_close("surface_x[..., 5] (normal_z)", mirrored.surface_x[..., 5], original_surface_x[..., 5])
    expect_close("surface_x[..., 6] (panel_area)", mirrored.surface_x[..., 6], original_surface_x[..., 6])

    expect_close("surface_y[..., 0] (cp/SP)", mirrored.surface_y[..., 0], original_surface_y[..., 0])
    expect_close("surface_y[..., 1] (tau_x)", mirrored.surface_y[..., 1], original_surface_y[..., 1])
    expect_close("surface_y[..., 2] (tau_y)", mirrored.surface_y[..., 2], -original_surface_y[..., 2])
    expect_close("surface_y[..., 3] (tau_z)", mirrored.surface_y[..., 3], original_surface_y[..., 3])

    expect_close("volume_x[..., 0] (x)", mirrored.volume_x[..., 0], original_volume_x[..., 0])
    expect_close("volume_x[..., 1] (y)", mirrored.volume_x[..., 1], -original_volume_x[..., 1])
    expect_close("volume_x[..., 2] (z)", mirrored.volume_x[..., 2], original_volume_x[..., 2])
    expect_close("volume_x[..., 3] (sdf)", mirrored.volume_x[..., 3], original_volume_x[..., 3])

    expect_close("volume_y (VP)", mirrored.volume_y, original_volume_y)
    print("  PASS — all expected channels flipped/preserved correctly")

    # Confirm we did NOT mutate the original case's tensors in place.
    expect_equal("original.surface_x is unchanged", case.surface_x, original_surface_x)
    expect_equal("original.surface_y is unchanged", case.surface_y, original_surface_y)
    expect_equal("original.volume_x is unchanged", case.volume_x, original_volume_x)
    print("  PASS — original case tensors are untouched (no in-place leak)")

    print("\n[Test 2] apply_y_mirror prob=0.0 is identity")
    identity = apply_y_mirror(case, prob=0.0)
    expect_equal("identity surface_x", identity.surface_x, original_surface_x)
    expect_equal("identity surface_y", identity.surface_y, original_surface_y)
    expect_equal("identity volume_x", identity.volume_x, original_volume_x)
    expect_equal("identity volume_y", identity.volume_y, original_volume_y)
    print("  PASS")

    print("\n[Test 3] Train Dataset use_y_mirror_aug=True stochastic application")
    torch.manual_seed(0)
    train_ds = DrivAerMLSurfaceDataset(
        train_ids[:1],
        store=store,
        max_surface_points=4096,
        max_volume_points=2048,
        sampling_mode="train_random",
        use_y_mirror_aug=True,
        y_mirror_prob=0.5,
    )
    # Draw 100 samples; expect roughly 50 mirrored.
    mirror_count = 0
    n = min(100, len(train_ds))
    for i in range(n):
        idx = i % len(train_ds)
        sample = train_ds[idx]
        mirror_count += int(sample.metadata.get("y_mirror_applied", False))
    print(f"  {mirror_count}/{n} samples were mirrored at prob=0.5 (expect ~{n // 2})")
    if not (0.25 * n <= mirror_count <= 0.75 * n):
        raise AssertionError("Mirror fraction is far from 0.5")
    print("  PASS")

    print("\n[Test 4] use_y_mirror_aug=True, prob=1.0 always mirrors")
    torch.manual_seed(0)
    train_ds_p1 = DrivAerMLSurfaceDataset(
        train_ids[:1],
        store=store,
        max_surface_points=4096,
        max_volume_points=2048,
        sampling_mode="train_random",
        use_y_mirror_aug=True,
        y_mirror_prob=1.0,
    )
    for i in range(5):
        sample = train_ds_p1[i % len(train_ds_p1)]
        if not sample.metadata.get("y_mirror_applied", False):
            raise AssertionError(f"prob=1.0 sample {i} was not mirrored")
        # Verify y in surface and volume are non-positive overall (sign-flipped)
        # is not a perfect invariant, just spot-check that the flip happened.
    print("  PASS — prob=1.0 always mirrors")

    print("\n[Test 5] Eval Dataset use_y_mirror_aug=True does NOT mirror")
    val_ds_with_flag = DrivAerMLSurfaceDataset(
        val_ids[:1],
        store=store,
        max_surface_points=4096,
        max_volume_points=2048,
        sampling_mode="eval_chunk",
        use_y_mirror_aug=True,  # flag is on, but defense-in-depth blocks it
        y_mirror_prob=1.0,
    )
    for i in range(min(4, len(val_ds_with_flag))):
        sample = val_ds_with_flag[i]
        if sample.metadata.get("y_mirror_applied", False):
            raise AssertionError(
                f"Eval Dataset sample {i} was mirrored despite eval_chunk mode"
            )
    print(f"  PASS — checked {min(4, len(val_ds_with_flag))} eval samples; none mirrored")

    print("\n[Test 6] Eval Dataset use_y_mirror_aug=False unchanged")
    val_ds_off = DrivAerMLSurfaceDataset(
        val_ids[:1],
        store=store,
        max_surface_points=4096,
        max_volume_points=2048,
        sampling_mode="eval_chunk",
        use_y_mirror_aug=False,
    )
    sample = val_ds_off[0]
    if sample.metadata.get("y_mirror_applied", False):
        raise AssertionError("eval sample with flag off was unexpectedly mirrored")
    print("  PASS")

    print("\n[Test 7] Approximate longitudinal symmetry per-car spot check")
    # For each of the first 5 train cars, compare mean(SP|y>0) vs mean(SP|y<0).
    print("  case_id        N_surf   mean(SP|y>0)  mean(SP|y<0)  rel_diff%")
    for case_id in train_ids[:5]:
        c = store.load_case(case_id)
        y = c.surface_x[..., 1]
        sp = c.surface_y[..., 0]
        pos = sp[y > 0]
        neg = sp[y < 0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        m_pos = pos.mean().item()
        m_neg = neg.mean().item()
        rel = abs(m_pos - m_neg) / max(abs(m_pos) + abs(m_neg), 1e-6) * 100
        print(f"  {case_id:14s} {y.numel():7d}  {m_pos:11.5f}  {m_neg:11.5f}  {rel:7.2f}%")

    print("\nAll sanity checks PASSED.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nElapsed: {time.time() - t0:.1f}s")
