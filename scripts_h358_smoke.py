"""H358 sanity checks: tangent basis math + zero-init equivalence.

Runs CPU-only, fast. Verifies:
  1. Tangent basis vectors are unit, orthogonal, and t1 x t2 == n.
  2. With use_tangent_resid_head=True and the head zero-init, the model
     output is exactly equal to the same model with the head disabled
     (i.e. the head is identity at step 0).
  3. After perturbing the head's final-layer weights, the predicted tau
     residual is strictly tangential: (resid . n) is ~0.
"""

from __future__ import annotations

import math

import torch

from model import (
    SurfaceTransolver,
    TangentResidualHead,
    _compute_tangent_basis,
)


def test_tangent_basis() -> None:
    torch.manual_seed(0)
    # 1000 random unit normals
    raw = torch.randn(1, 1000, 3)
    n = raw / raw.norm(dim=-1, keepdim=True)
    # Add edge cases: exactly axis-aligned normals
    edge = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.999, 0.0, 0.044],  # near-equatorial
            [0.044, 0.0, 0.999],  # near-vertical
        ]
    ).unsqueeze(0)
    n = torch.cat([n, edge], dim=1)

    t1, t2 = _compute_tangent_basis(n)
    nrm1 = t1.norm(dim=-1)
    nrm2 = t2.norm(dim=-1)
    assert torch.allclose(nrm1, torch.ones_like(nrm1), atol=1e-4), \
        f"|t1| not unit: max abs err {(nrm1 - 1).abs().max():.2e}"
    assert torch.allclose(nrm2, torch.ones_like(nrm2), atol=1e-4), \
        f"|t2| not unit: max abs err {(nrm2 - 1).abs().max():.2e}"
    d_n_t1 = (n * t1).sum(dim=-1).abs().max()
    d_n_t2 = (n * t2).sum(dim=-1).abs().max()
    d_t1_t2 = (t1 * t2).sum(dim=-1).abs().max()
    assert d_n_t1 < 1e-4, f"n.t1 not zero: {d_n_t1:.2e}"
    assert d_n_t2 < 1e-4, f"n.t2 not zero: {d_n_t2:.2e}"
    assert d_t1_t2 < 1e-4, f"t1.t2 not zero: {d_t1_t2:.2e}"
    # t1 x t2 should equal n (right-handed frame)
    cross = torch.linalg.cross(t1, t2, dim=-1)
    err = (cross - n).norm(dim=-1).max()
    assert err < 1e-4, f"t1 x t2 != n: max err {err:.2e}"
    print("PASS: tangent basis is orthonormal and right-handed")


def test_zero_init_identity() -> None:
    torch.manual_seed(0)
    model_kw = dict(
        n_layers=2,
        n_hidden=64,
        n_head=2,
        slice_num=8,
        mlp_ratio=2,
        rff_num_features=4,
        pos_encoding_mode="string_separable",
        rff_init_sigmas=[0.5, 1.0],
        use_qk_norm=True,
        use_surf_to_vol_xattn=True,
    )
    base = SurfaceTransolver(**model_kw)
    base.eval()

    torch.manual_seed(0)
    head_model = SurfaceTransolver(**model_kw, use_tangent_resid_head=True)
    head_model.eval()

    # Confirm the head exists and its output layer is zero
    assert head_model.tangent_resid_head is not None
    w = head_model.tangent_resid_head.net[-1].weight
    b = head_model.tangent_resid_head.net[-1].bias
    assert torch.all(w == 0)
    assert torch.all(b == 0)

    # Build a small batch
    B, N, M = 2, 32, 16
    surface_x = torch.randn(B, N, 7)
    raw_n = torch.randn(B, N, 3)
    surface_x[..., 3:6] = raw_n / raw_n.norm(dim=-1, keepdim=True)
    surface_mask = torch.ones(B, N, dtype=torch.bool)
    volume_x = torch.randn(B, M, 4)
    volume_mask = torch.ones(B, M, dtype=torch.bool)

    with torch.no_grad():
        out_base = base(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
        out_head = head_model(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )

    diff = (out_base["surface_preds"] - out_head["surface_preds"]).abs().max().item()
    assert diff < 1e-6, f"zero-init head changed output: max abs diff {diff:.2e}"
    print(f"PASS: zero-init head preserves baseline output (max abs diff {diff:.2e})")


def test_residual_is_tangential() -> None:
    torch.manual_seed(0)
    head = TangentResidualHead(d_model=64, hidden=32)
    # Perturb output layer so it produces non-zero tau_local
    with torch.no_grad():
        head.net[-1].weight.normal_(std=0.5)
        head.net[-1].bias.normal_(std=0.5)
    B, N = 2, 64
    feats = torch.randn(B, N, 64)
    raw_n = torch.randn(B, N, 3)
    n = raw_n / raw_n.norm(dim=-1, keepdim=True)
    resid = head(feats, n)
    proj = (resid * n).sum(dim=-1).abs().max().item()
    assert proj < 1e-4, f"residual not strictly tangent: max |resid.n| = {proj:.2e}"
    print(f"PASS: residual is tangent to surface (max |resid.n| = {proj:.2e})")


def test_cuda_bf16_forward_backward() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return
    device = torch.device("cuda")
    torch.manual_seed(0)
    model_kw = dict(
        n_layers=2,
        n_hidden=128,
        n_head=4,
        slice_num=16,
        mlp_ratio=4,
        rff_num_features=8,
        pos_encoding_mode="string_separable",
        rff_init_sigmas=[0.25, 0.5, 1.0, 2.0, 4.0],
        use_qk_norm=True,
        use_surf_to_vol_xattn=True,
        use_tangent_resid_head=True,
    )
    model = SurfaceTransolver(**model_kw).to(device)
    B, N, M = 2, 1024, 1024
    surface_x = torch.randn(B, N, 7, device=device)
    raw_n = torch.randn(B, N, 3, device=device)
    surface_x[..., 3:6] = raw_n / raw_n.norm(dim=-1, keepdim=True)
    surface_y = torch.randn(B, N, 4, device=device)
    surface_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    volume_x = torch.randn(B, M, 4, device=device)
    volume_y = torch.randn(B, M, 1, device=device)
    volume_mask = torch.ones(B, M, dtype=torch.bool, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        out = model(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
        s_pred = out["surface_preds"]
        v_pred = out["volume_preds"]
        loss = (s_pred - surface_y).pow(2).mean() + (v_pred - volume_y).pow(2).mean()
    loss.backward()

    head_grad = model.tangent_resid_head.net[0].weight.grad
    assert head_grad is not None, "tangent_resid_head has no gradient"
    assert torch.isfinite(head_grad).all(), "tangent_resid_head grad has NaN/Inf"
    # Final layer is zero-init so its weights see gradient from chain rule
    final_grad = model.tangent_resid_head.net[-1].weight.grad
    assert final_grad is not None and torch.isfinite(final_grad).all()
    print(
        f"PASS: CUDA bf16 forward+backward OK "
        f"(loss={loss.item():.4f}, head input-layer grad norm={head_grad.norm().item():.3e}, "
        f"head output-layer grad norm={final_grad.norm().item():.3e})"
    )


if __name__ == "__main__":
    test_tangent_basis()
    test_zero_init_identity()
    test_residual_is_tangential()
    test_cuda_bf16_forward_backward()
    print("\nAll H358 sanity checks passed.")
