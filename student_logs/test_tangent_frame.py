"""Smoke validation for H17 local tangent-frame WSS reparameterization.

Verifies:
1. Forward pass with use_tangent_frame_output=True returns finite [B, N, 4] surface_preds.
2. Tangent basis is orthonormal (|t1.t2|, |n.t1| < 1e-3).
3. Reconstructed tau is perpendicular to n by construction (|tau.n| ~ 0).
4. Empty-surface batches (N_S=0) do NOT crash anymore — the .amax() bug fix.
5. Baseline path (use_tangent_frame_output=False) still works.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from model import SurfaceTransolver  # noqa: E402


def _make_inputs(B, N_s, N_v, surface_dim, volume_dim, device, dtype=torch.float32):
    surface_x = torch.randn(B, N_s, surface_dim, device=device, dtype=dtype)
    # Channels 3:6 are normals; make them unit-ish by scaling random vectors.
    if N_s > 0:
        normals = torch.randn(B, N_s, 3, device=device, dtype=dtype)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        surface_x[..., 3:6] = normals
    surface_mask = torch.ones(B, N_s, device=device, dtype=torch.bool)
    volume_x = torch.randn(B, N_v, volume_dim, device=device, dtype=dtype)
    volume_mask = torch.ones(B, N_v, device=device, dtype=torch.bool)
    return surface_x, surface_mask, volume_x, volume_mask


def test_basic_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(0)

    model = SurfaceTransolver(
        space_dim=3,
        surface_input_dim=7,
        surface_output_dim=4,
        volume_input_dim=4,
        volume_output_dim=1,
        n_layers=2,
        n_hidden=64,
        n_head=2,
        slice_num=16,
        use_tangent_frame_output=True,
    ).to(device)

    B, N_s, N_v = 2, 128, 256
    surface_x, surface_mask, volume_x, volume_mask = _make_inputs(B, N_s, N_v, 7, 4, device)

    out = model(
        surface_x=surface_x,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_mask=volume_mask,
    )
    surface_preds = out["surface_preds"]
    assert surface_preds.shape == (B, N_s, 4), f"got {surface_preds.shape}"
    assert torch.isfinite(surface_preds).all(), "non-finite surface_preds"
    print(f"OK surface_preds shape={tuple(surface_preds.shape)}, finite={torch.isfinite(surface_preds).all().item()}")

    # Verify tau.n = 0 by construction
    n = torch.nn.functional.normalize(surface_x[..., 3:6], dim=-1)
    tau = surface_preds[..., 1:4]
    tau_dot_n = (tau * n).sum(dim=-1).abs().max().item()
    print(f"|tau.n|_max = {tau_dot_n:.6e}  (expected ~ 0)")
    assert tau_dot_n < 1e-3, f"tau is not perpendicular to n: |tau.n|_max={tau_dot_n}"

    # Diagnostics keys
    assert "tangent_frame/t1_t2_orthogonality_residual" in out
    assert "tangent_frame/n_t1_orthogonality_residual" in out
    t1t2 = out["tangent_frame/t1_t2_orthogonality_residual"].item()
    nt1 = out["tangent_frame/n_t1_orthogonality_residual"].item()
    print(f"|t1.t2|_mean = {t1t2:.6e}  |n.t1|_mean = {nt1:.6e}")
    assert t1t2 < 1e-3
    assert nt1 < 1e-3
    print("OK basic_forward passes")


def test_empty_surface_batch():
    """The .amax() crash that killed the previous smoke run."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)

    model = SurfaceTransolver(
        space_dim=3,
        surface_input_dim=7,
        surface_output_dim=4,
        volume_input_dim=4,
        volume_output_dim=1,
        n_layers=2,
        n_hidden=64,
        n_head=2,
        slice_num=16,
        use_tangent_frame_output=True,
    ).to(device)

    B, N_s, N_v = 2, 0, 256
    surface_x, surface_mask, volume_x, volume_mask = _make_inputs(B, N_s, N_v, 7, 4, device)
    out = model(
        surface_x=surface_x,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_mask=volume_mask,
    )
    surface_preds = out["surface_preds"]
    assert surface_preds.shape == (B, 0, 4), f"got {surface_preds.shape}"
    print(f"OK empty surface batch: surface_preds.shape={tuple(surface_preds.shape)}")
    # No tangent_frame/* keys expected since surface_hidden was empty
    tangent_keys = [k for k in out if k.startswith("tangent_frame/")]
    print(f"tangent_frame diagnostics keys (expect empty): {tangent_keys}")
    assert tangent_keys == [], f"unexpected diagnostics for empty surface: {tangent_keys}"
    print("OK empty_surface_batch passes")


def test_baseline_path():
    """use_tangent_frame_output=False should produce 4-channel head and no diagnostics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2)

    model = SurfaceTransolver(
        space_dim=3,
        surface_input_dim=7,
        surface_output_dim=4,
        volume_input_dim=4,
        volume_output_dim=1,
        n_layers=2,
        n_hidden=64,
        n_head=2,
        slice_num=16,
        use_tangent_frame_output=False,
    ).to(device)

    B, N_s, N_v = 2, 64, 128
    surface_x, surface_mask, volume_x, volume_mask = _make_inputs(B, N_s, N_v, 7, 4, device)
    out = model(
        surface_x=surface_x,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_mask=volume_mask,
    )
    assert out["surface_preds"].shape == (B, N_s, 4)
    tangent_keys = [k for k in out if k.startswith("tangent_frame/")]
    assert tangent_keys == [], f"baseline path should not emit tangent diagnostics: {tangent_keys}"
    print("OK baseline_path passes")


def test_backward_pass():
    """Gradients should flow through the 2-channel tangent head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(3)

    model = SurfaceTransolver(
        space_dim=3,
        surface_input_dim=7,
        surface_output_dim=4,
        volume_input_dim=4,
        volume_output_dim=1,
        n_layers=2,
        n_hidden=64,
        n_head=2,
        slice_num=16,
        use_tangent_frame_output=True,
    ).to(device)

    B, N_s, N_v = 2, 128, 256
    surface_x, surface_mask, volume_x, volume_mask = _make_inputs(B, N_s, N_v, 7, 4, device)
    out = model(
        surface_x=surface_x,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_mask=volume_mask,
    )
    loss = out["surface_preds"].pow(2).mean() + out["volume_preds"].pow(2).mean()
    loss.backward()
    head_grad_norm = 0.0
    head_param_count = 0
    for name, p in model.surface_out.named_parameters():
        if p.grad is None:
            print(f"WARN: no grad for surface_out.{name}")
        else:
            head_grad_norm += p.grad.float().norm().item() ** 2
            head_param_count += 1
    head_grad_norm = head_grad_norm ** 0.5
    print(f"surface_out param count={head_param_count}, total grad norm={head_grad_norm:.4e}")
    assert head_grad_norm > 0, "surface head got no gradient"
    print("OK backward_pass passes")


if __name__ == "__main__":
    test_basic_forward()
    test_empty_surface_batch()
    test_baseline_path()
    test_backward_pass()
    print("\nAll tangent-frame smoke tests passed.")
