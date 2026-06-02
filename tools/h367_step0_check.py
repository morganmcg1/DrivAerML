"""H367 step-0 invariance check.

Verifies that --use-anisotropic-frame-attention is approximately identity at
step 0 when the per-layer gate logit gamma_aniso is initialized very negative.
With aniso_init_gamma=-20 the sigmoid gate is ~2e-9 and the model should be
architecturally bit-identical to the baseline (max abs Δ < 1e-5). Also runs a
diagnostic at the training default gamma_aniso=-10 so we know what numerical
deviation to expect at the start of fine-tuning.

Mirrors the pattern from tools/h342_avg_checkpoints.py / scripts_h358_smoke.py.
"""

from __future__ import annotations

import math
import sys

import torch

from model import SurfaceTransolver, _build_local_frame, _rotate_3blocks


MODEL_KW = dict(
    n_layers=2,
    n_hidden=64,
    n_head=4,
    slice_num=8,
    mlp_ratio=2,
    rff_num_features=4,
    pos_encoding_mode="string_separable",
    rff_init_sigmas=[0.5, 1.0],
    use_qk_norm=True,
    use_surf_to_vol_xattn=True,
)


def _make_batch(device: torch.device) -> dict:
    torch.manual_seed(0)
    B, N, M = 2, 64, 48
    surface_x = torch.randn(B, N, 7, device=device)
    raw_n = torch.randn(B, N, 3, device=device)
    surface_x[..., 3:6] = raw_n / raw_n.norm(dim=-1, keepdim=True)
    surface_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    volume_x = torch.randn(B, M, 4, device=device)
    volume_mask = torch.ones(B, M, dtype=torch.bool, device=device)
    return dict(
        surface_x=surface_x,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_mask=volume_mask,
    )


def _run(model: SurfaceTransolver, batch: dict) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        return model(**batch)


def test_rotation_helpers() -> None:
    torch.manual_seed(0)
    n = torch.randn(2, 50, 3)
    n = n / n.norm(dim=-1, keepdim=True)
    R = _build_local_frame(n)
    # Rows of R must be orthonormal and t1 x t2 == n.
    t1, t2, n_row = R[..., 0, :], R[..., 1, :], R[..., 2, :]
    assert torch.allclose(t1.norm(dim=-1), torch.ones_like(t1[..., 0]), atol=1e-4)
    assert torch.allclose(t2.norm(dim=-1), torch.ones_like(t2[..., 0]), atol=1e-4)
    assert torch.allclose(n_row.norm(dim=-1), torch.ones_like(n_row[..., 0]), atol=1e-4)
    assert (t1 * t2).sum(-1).abs().max() < 1e-4
    assert (t1 * n_row).sum(-1).abs().max() < 1e-4
    assert (t2 * n_row).sum(-1).abs().max() < 1e-4
    cross = torch.linalg.cross(t1, t2, dim=-1)
    assert (cross - n_row).norm(dim=-1).max() < 1e-4
    # Edge: axis-aligned normals should still build a valid frame.
    edge = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    ).unsqueeze(0)
    R_edge = _build_local_frame(edge)
    e_t1, e_t2, e_n = R_edge[..., 0, :], R_edge[..., 1, :], R_edge[..., 2, :]
    assert torch.allclose(e_t1.norm(dim=-1), torch.ones_like(e_t1[..., 0]), atol=1e-4)
    assert torch.allclose(e_t2.norm(dim=-1), torch.ones_like(e_t2[..., 0]), atol=1e-4)
    assert (e_t1 * e_n).sum(-1).abs().max() < 1e-4
    print("PASS: _build_local_frame is orthonormal and right-handed (incl. axis-aligned)")
    # Identity rotation: R = I leaves features unchanged.
    feat = torch.randn(2, 4, 8, 16)
    R_id = torch.eye(3).expand(2, 4, 8, 3, 3)
    out = _rotate_3blocks(feat, R_id)
    assert (out - feat).abs().max() < 1e-6
    # Inverse rotation: R @ R^T = I means rotating then unrotating recovers.
    R_rand = _build_local_frame(torch.randn(2, 8, 3))
    R_rand_expand = R_rand.unsqueeze(1).expand(2, 4, 8, 3, 3)
    feat = torch.randn(2, 4, 8, 15)  # dim divisible by 3 to keep test exact
    rotated = _rotate_3blocks(feat, R_rand_expand)
    # _rotate_3blocks computes R @ block; unrotate is R^T @ rotated = block.
    R_T = R_rand_expand.transpose(-2, -1)
    unrotated = _rotate_3blocks(rotated, R_T)
    err = (unrotated - feat).abs().max().item()
    assert err < 1e-5, f"rotate then unrotate failed: max err {err:.2e}"
    print(f"PASS: _rotate_3blocks is invertible (max err {err:.2e})")


def test_step0_invariance(gamma: float = -20.0, tol: float = 1e-5) -> float:
    """Build OFF and ON models with same RNG state; assert |Δ| < tol."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    base = SurfaceTransolver(**MODEL_KW).to(device)
    torch.manual_seed(0)
    aniso = SurfaceTransolver(
        **MODEL_KW,
        use_anisotropic_frame_attention=True,
        aniso_init_gamma=gamma,
    ).to(device)

    # Copy baseline weights so both models share initial parameters apart from
    # the new gamma_aniso scalars (which exist only in `aniso`).
    base_sd = base.state_dict()
    aniso_sd = aniso.state_dict()
    for k, v in base_sd.items():
        if k in aniso_sd and aniso_sd[k].shape == v.shape:
            aniso_sd[k] = v.clone()
    aniso.load_state_dict(aniso_sd, strict=True)

    batch = _make_batch(device)
    out_base = _run(base, batch)
    out_aniso = _run(aniso, batch)

    d_surface = (out_base["surface_preds"] - out_aniso["surface_preds"]).abs().max().item()
    d_volume = (out_base["volume_preds"] - out_aniso["volume_preds"]).abs().max().item()
    sigma_gamma = 1.0 / (1.0 + math.exp(-gamma))
    print(
        f"gamma_aniso={gamma:>6.2f}  sigmoid(g)={sigma_gamma:.3e}  "
        f"max|Δsurface|={d_surface:.3e}  max|Δvolume|={d_volume:.3e}"
    )
    if tol is not None:
        assert d_surface < tol, (
            f"step-0 invariance FAILED at gamma={gamma}: "
            f"max|Δsurface|={d_surface:.3e} >= tol={tol:.3e}"
        )
    return d_surface


def main() -> None:
    print(">> H367 step-0 invariance check")
    test_rotation_helpers()
    # Architectural invariance check: very negative gate => bit-identical to baseline.
    d_strict = test_step0_invariance(gamma=-20.0, tol=1e-5)
    print(f"PASS: architectural invariance at gamma=-20 (max|Δ|={d_strict:.3e} < 1e-5)")
    # Diagnostic: report deviation at the training-default gamma=-10.
    d_train = test_step0_invariance(gamma=-10.0, tol=None)
    print(
        f"INFO: training-init gamma=-10 deviation max|Δsurface|={d_train:.3e} "
        f"(expected ~4.5e-5 from sigmoid(-10)≈4.54e-5)"
    )
    print(">> H367 step-0 invariance: OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as err:
        print(f"FAIL: {err}")
        sys.exit(1)
