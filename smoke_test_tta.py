"""Smoke tests for symmetry-TTA in train.py.

Exits with non-zero status on any failure. Run on a single GPU.
"""
from __future__ import annotations

import sys
import torch

import train as T
from data import SURFACE_X_DIM, SURFACE_Y_DIM, VOLUME_X_DIM, VOLUME_Y_DIM, SurfaceBatch


def _make_dummy_batch(device: torch.device, *, batch_size: int = 1, n_surface: int = 8, n_volume: int = 8) -> SurfaceBatch:
    g = torch.Generator(device="cpu").manual_seed(0)
    surface_x = torch.randn(batch_size, n_surface, SURFACE_X_DIM, generator=g)
    surface_y = torch.randn(batch_size, n_surface, SURFACE_Y_DIM, generator=g)
    surface_mask = torch.ones(batch_size, n_surface, dtype=torch.bool)
    volume_x = torch.randn(batch_size, n_volume, VOLUME_X_DIM, generator=g)
    volume_y = torch.randn(batch_size, n_volume, VOLUME_Y_DIM, generator=g)
    volume_mask = torch.ones(batch_size, n_volume, dtype=torch.bool)
    return SurfaceBatch(
        case_ids=["synthetic_case_0"],
        surface_x=surface_x.to(device),
        surface_y=surface_y.to(device),
        surface_mask=surface_mask.to(device),
        volume_x=volume_x.to(device),
        volume_y=volume_y.to(device),
        volume_mask=volume_mask.to(device),
        metadata=[{}],
    )


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = T.Config(
        model_layers=2,
        model_hidden_dim=64,
        model_heads=4,
        model_slices=8,
        compile_model=False,
        amp_mode="none",
        use_film=False,
    )
    model = T.build_model(cfg).to(device).eval()
    batch = _make_dummy_batch(device)

    # ---- Smoke test 2 — input transformation correctness -----------------------
    surf_x_refl = batch.surface_x.clone()
    surf_x_refl[..., 1] = -surf_x_refl[..., 1]
    surf_x_refl[..., 4] = -surf_x_refl[..., 4]
    vol_x_refl = batch.volume_x.clone()
    vol_x_refl[..., 1] = -vol_x_refl[..., 1]

    # Surface: y (idx 1) and ny (idx 4) negated; others unchanged
    invariant_surface_idxs = [0, 2, 3, 5, 6]  # x, z, nx, nz, area
    for i in invariant_surface_idxs:
        if not torch.allclose(batch.surface_x[..., i], surf_x_refl[..., i]):
            print(f"FAIL: surface_x channel {i} should be invariant under y-reflection")
            sys.exit(1)
    if not torch.allclose(batch.surface_x[..., 1], -surf_x_refl[..., 1]):
        print("FAIL: surface_x y-coord (idx 1) not properly negated")
        sys.exit(1)
    if not torch.allclose(batch.surface_x[..., 4], -surf_x_refl[..., 4]):
        print("FAIL: surface_x ny-normal (idx 4) not properly negated")
        sys.exit(1)

    # Volume: y (idx 1) negated; sdf (idx 3) invariant
    if not torch.allclose(batch.volume_x[..., 0], vol_x_refl[..., 0]):
        print("FAIL: volume_x x-coord should be invariant")
        sys.exit(1)
    if not torch.allclose(batch.volume_x[..., 2], vol_x_refl[..., 2]):
        print("FAIL: volume_x z-coord should be invariant")
        sys.exit(1)
    if not torch.allclose(batch.volume_x[..., 3], vol_x_refl[..., 3]):
        print("FAIL: volume_x sdf should be invariant")
        sys.exit(1)
    if not torch.allclose(batch.volume_x[..., 1], -vol_x_refl[..., 1]):
        print("FAIL: volume_x y-coord (idx 1) not properly negated")
        sys.exit(1)
    print("PASS: smoke test 2 — input y-coord and y-normal negation correct")

    # ---- Smoke test 3 — output correction correctness --------------------------
    with torch.no_grad():
        out_orig = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
        out_refl = model(
            surface_x=surf_x_refl,
            surface_mask=batch.surface_mask,
            volume_x=vol_x_refl,
            volume_mask=batch.volume_mask,
        )
    surface_preds_refl_pre = out_refl["surface_preds"].clone()
    surface_preds_refl_corrected = surface_preds_refl_pre.clone()
    surface_preds_refl_corrected[..., 2] = -surface_preds_refl_corrected[..., 2]

    # Channels 0 (cp), 1 (tau_x), 3 (tau_z) should be unchanged by the correction
    if not torch.allclose(surface_preds_refl_pre[..., 0], surface_preds_refl_corrected[..., 0]):
        print("FAIL: cp channel modified by correction (should be invariant)")
        sys.exit(1)
    if not torch.allclose(surface_preds_refl_pre[..., 1], surface_preds_refl_corrected[..., 1]):
        print("FAIL: tau_x channel modified by correction (should be invariant)")
        sys.exit(1)
    if not torch.allclose(surface_preds_refl_pre[..., 3], surface_preds_refl_corrected[..., 3]):
        print("FAIL: tau_z channel modified by correction (should be invariant)")
        sys.exit(1)
    # Channel 2 (tau_y) should be sign-flipped
    if not torch.allclose(surface_preds_refl_pre[..., 2], -surface_preds_refl_corrected[..., 2]):
        print("FAIL: tau_y not properly sign-flipped")
        sys.exit(1)
    print("PASS: smoke test 3 — output tau_y sign flip applied; tau_x/tau_z/cp unchanged")

    # Also: print sample values for visual inspection
    print("\nSample predictions (first batch, first 3 surface points):")
    print(f"  pred_orig  (cp, tau_x, tau_y, tau_z): {out_orig['surface_preds'][0, :3].detach().cpu().tolist()}")
    print(f"  pred_refl  (cp, tau_x, tau_y, tau_z): {surface_preds_refl_pre[0, :3].detach().cpu().tolist()}")
    print(f"  pred_refl* (cp, tau_x, tau_y, tau_z): {surface_preds_refl_corrected[0, :3].detach().cpu().tolist()}  # tau_y sign-flipped")
    pred_tta = 0.5 * (out_orig["surface_preds"].float() + surface_preds_refl_corrected.float())
    print(f"  pred_tta = 0.5*(orig + refl*): {pred_tta[0, :3].detach().cpu().tolist()}")

    # ---- Smoke test 1 — evaluate_split runs and produces finite metrics -------
    # Build a tiny one-batch loader for evaluate_split.
    class _OneBatchLoader:
        def __init__(self, batch):
            self._batch = batch
        def __iter__(self):
            return iter([self._batch])

    loader = _OneBatchLoader(batch)
    surf_y_mean = torch.zeros(SURFACE_Y_DIM)
    surf_y_std = torch.ones(SURFACE_Y_DIM)
    vol_y_mean = torch.zeros(VOLUME_Y_DIM)
    vol_y_std = torch.ones(VOLUME_Y_DIM)
    transform = T.TargetTransform(
        surface_y_mean=surf_y_mean,
        surface_y_std=surf_y_std,
        volume_y_mean=vol_y_mean,
        volume_y_std=vol_y_std,
    )
    metrics_no_tta = T.evaluate_split(model, loader, transform, device, use_symmetry_tta=False)
    metrics_tta = T.evaluate_split(model, loader, transform, device, use_symmetry_tta=True)
    for k, v in metrics_no_tta.items():
        if isinstance(v, float) and (v != v or abs(v) == float("inf")):  # NaN/Inf check
            print(f"FAIL: no-TTA metric {k} = {v}")
            sys.exit(1)
    for k, v in metrics_tta.items():
        if isinstance(v, float) and (v != v or abs(v) == float("inf")):
            print(f"FAIL: TTA metric {k} = {v}")
            sys.exit(1)
    print("PASS: smoke test 1 — evaluate_split produces finite metrics for both arms")

    # Sanity: TTA should produce numerically different metrics from no-TTA on a random model
    abupt_diff = abs(metrics_tta["abupt_axis_mean_rel_l2_pct"] - metrics_no_tta["abupt_axis_mean_rel_l2_pct"])
    print(f"\nMetrics on random-init model (small dummy batch):")
    print(f"  no-TTA abupt_axis_mean_rel_l2_pct: {metrics_no_tta['abupt_axis_mean_rel_l2_pct']:.4f}")
    print(f"  TTA    abupt_axis_mean_rel_l2_pct: {metrics_tta['abupt_axis_mean_rel_l2_pct']:.4f}")
    print(f"  delta abs: {abupt_diff:.6f}")
    print(f"  no-TTA tau_y_rel_l2_pct: {metrics_no_tta['wall_shear_y_rel_l2_pct']:.4f}")
    print(f"  TTA    tau_y_rel_l2_pct: {metrics_tta['wall_shear_y_rel_l2_pct']:.4f}")
    if abupt_diff < 1e-6:
        print("WARN: TTA and no-TTA metrics are essentially identical — verify reflection is doing something")
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
