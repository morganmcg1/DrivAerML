"""Unit tests for y_flip_batch — verifies the bilateral-symmetry transform."""
import sys
import torch

sys.path.insert(0, "/workspace/senpai/target")

from train import y_flip_batch
from data.loader import SurfaceBatch


def make_batch(B=2, N_surf=8, N_vol=6, surf_x_dim=7):
    torch.manual_seed(0)
    surface_x = torch.randn(B, N_surf, surf_x_dim)
    surface_y = torch.randn(B, N_surf, 4)  # cp, ws_x, ws_y, ws_z
    surface_mask = torch.ones(B, N_surf, dtype=torch.bool)
    volume_x = torch.randn(B, N_vol, 4)  # x, y, z, sdf
    volume_y = torch.randn(B, N_vol, 1)
    volume_mask = torch.ones(B, N_vol, dtype=torch.bool)
    return SurfaceBatch(
        case_ids=["c0", "c1"],
        surface_x=surface_x,
        surface_y=surface_y,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_y=volume_y,
        volume_mask=volume_mask,
        metadata=[{}, {}],
    )


def test_double_flip_identity():
    batch = make_batch()
    flipped = y_flip_batch(batch)
    twice = y_flip_batch(flipped)
    assert torch.allclose(twice.surface_x, batch.surface_x, atol=1e-6), "surface_x roundtrip"
    assert torch.allclose(twice.surface_y, batch.surface_y, atol=1e-6), "surface_y roundtrip"
    assert torch.allclose(twice.volume_x, batch.volume_x, atol=1e-6), "volume_x roundtrip"
    assert torch.allclose(twice.volume_y, batch.volume_y, atol=1e-6), "volume_y roundtrip"
    print("PASS: double-flip identity")


def test_no_mutation():
    batch = make_batch()
    surface_x_before = batch.surface_x.clone()
    surface_y_before = batch.surface_y.clone()
    volume_x_before = batch.volume_x.clone()
    volume_y_before = batch.volume_y.clone()
    _ = y_flip_batch(batch)
    assert torch.equal(batch.surface_x, surface_x_before), "surface_x mutated"
    assert torch.equal(batch.surface_y, surface_y_before), "surface_y mutated"
    assert torch.equal(batch.volume_x, volume_x_before), "volume_x mutated"
    assert torch.equal(batch.volume_y, volume_y_before), "volume_y mutated"
    print("PASS: no in-place mutation of input batch")


def test_sign_flip():
    """Single-point sanity: flip flips y/n_y/ws_y, preserves x/z/n_x/n_z/area/cp/ws_x/ws_z."""
    surface_x = torch.tensor([[[1.0, 2.0, 3.0, 0.4, 0.5, 0.6, 0.7]]])  # B=1, N=1, 7 chs
    surface_y = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])  # cp, ws_x, ws_y, ws_z
    volume_x = torch.tensor([[[5.0, 6.0, 7.0, 0.1]]])
    volume_y = torch.tensor([[[100.0]]])
    batch = SurfaceBatch(
        case_ids=["c0"],
        surface_x=surface_x,
        surface_y=surface_y,
        surface_mask=torch.ones(1, 1, dtype=torch.bool),
        volume_x=volume_x,
        volume_y=volume_y,
        volume_mask=torch.ones(1, 1, dtype=torch.bool),
        metadata=[{}],
    )
    flipped = y_flip_batch(batch)
    expected_sx = torch.tensor([[[1.0, -2.0, 3.0, 0.4, -0.5, 0.6, 0.7]]])
    expected_sy = torch.tensor([[[10.0, 20.0, -30.0, 40.0]]])
    expected_vx = torch.tensor([[[5.0, -6.0, 7.0, 0.1]]])
    assert torch.equal(flipped.surface_x, expected_sx), f"surface_x got {flipped.surface_x}"
    assert torch.equal(flipped.surface_y, expected_sy), f"surface_y got {flipped.surface_y}"
    assert torch.equal(flipped.volume_x, expected_vx), f"volume_x got {flipped.volume_x}"
    assert torch.equal(flipped.volume_y, volume_y), f"volume_y must be EVEN"
    print("PASS: single-point sign-flip semantics")


def test_curvature_channels_preserved():
    """With curvature features (k1_k2 — channels 7,8), they should remain EVEN under flip."""
    surface_x = torch.randn(1, 4, 9)  # 9 channels — base 7 + 2 curvature
    base = SurfaceBatch(
        case_ids=["c0"],
        surface_x=surface_x,
        surface_y=torch.randn(1, 4, 4),
        surface_mask=torch.ones(1, 4, dtype=torch.bool),
        volume_x=torch.randn(1, 3, 4),
        volume_y=torch.randn(1, 3, 1),
        volume_mask=torch.ones(1, 3, dtype=torch.bool),
        metadata=[{}],
    )
    flipped = y_flip_batch(base)
    # Curvature columns 7,8 must be unchanged
    assert torch.equal(flipped.surface_x[..., 7:], base.surface_x[..., 7:]), "curvature changed"
    # Area column 6 must be unchanged
    assert torch.equal(flipped.surface_x[..., 6], base.surface_x[..., 6]), "area changed"
    # x,z position cols (0,2) and n_x,n_z normal cols (3,5) unchanged
    for c in (0, 2, 3, 5):
        assert torch.equal(flipped.surface_x[..., c], base.surface_x[..., c]), f"col {c} changed"
    # y position (1) and n_y (4) flipped
    assert torch.equal(flipped.surface_x[..., 1], -base.surface_x[..., 1]), "y not flipped"
    assert torch.equal(flipped.surface_x[..., 4], -base.surface_x[..., 4]), "n_y not flipped"
    print("PASS: curvature channels preserved, signs flip on correct cols")


if __name__ == "__main__":
    test_double_flip_identity()
    test_no_mutation()
    test_sign_flip()
    test_curvature_channels_preserved()
    print("\nAll y_flip_batch tests passed.")
