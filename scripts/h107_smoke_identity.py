"""H107 identity-at-init smoke test.

Builds two SurfaceTransolver instances with identical seeds — one with the
flag off, one with the flag on — and verifies that `surface_preds`,
`volume_preds`, and `surface_hidden` are bit-identical at step 0. Also
checks: (a) the projection params exist when on / absent when off,
(b) param-count delta matches the predicted +n_hidden*(n_hidden+1).
"""

from __future__ import annotations

import sys

import torch

from model import SurfaceTransolver


def build(seed: int, *, flag: bool, n_hidden: int = 64) -> SurfaceTransolver:
    torch.manual_seed(seed)
    return SurfaceTransolver(
        n_layers=2,
        n_hidden=n_hidden,
        n_head=2,
        mlp_ratio=4,
        slice_num=32,
        use_aux_decoder_heads=True,
        use_surface_global_context_residual=flag,
    )


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> int:
    n_hidden = 64
    n_surf = 17
    n_vol = 23
    batch = 2

    model_off = build(0, flag=False, n_hidden=n_hidden).eval()
    model_on = build(0, flag=True, n_hidden=n_hidden).eval()

    # 1. Flag-off projection must not exist.
    assert model_off.surface_global_context_proj is None
    assert model_on.surface_global_context_proj is not None

    # 2. Zero-init: both weight and bias must be exactly zero.
    proj = model_on.surface_global_context_proj
    assert torch.equal(proj.weight, torch.zeros_like(proj.weight))
    assert torch.equal(proj.bias, torch.zeros_like(proj.bias))

    # 3. Param-count delta = n_hidden * n_hidden + n_hidden.
    expected_delta = n_hidden * n_hidden + n_hidden
    actual_delta = count_params(model_on) - count_params(model_off)
    assert actual_delta == expected_delta, (
        f"param delta mismatch: expected {expected_delta}, got {actual_delta}"
    )

    # 4. Inputs.
    torch.manual_seed(123)
    surface_x = torch.randn(batch, n_surf, model_on.surface_input_dim)
    surface_mask = torch.ones(batch, n_surf, dtype=torch.bool)
    surface_mask[0, -3:] = False  # ragged mask on rank 0
    volume_x = torch.randn(batch, n_vol, model_on.volume_input_dim)
    volume_mask = torch.ones(batch, n_vol, dtype=torch.bool)
    volume_mask[1, -5:] = False

    with torch.no_grad():
        out_off = model_off(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
        out_on = model_on(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )

    # 5. surface_preds, volume_preds, surface_hidden must match exactly.
    for k in ("surface_preds", "volume_preds", "surface_hidden", "volume_hidden"):
        a, b = out_off[k], out_on[k]
        assert a.shape == b.shape, f"{k}: shape mismatch {a.shape} vs {b.shape}"
        max_abs = (a - b).abs().max().item()
        assert max_abs == 0.0, f"{k}: not bit-identical at init (max_abs={max_abs})"

    # 6. Volume-only path: surface_x=None must still work (gate avoids proj).
    with torch.no_grad():
        out_vol_only_on = model_on(volume_x=volume_x, volume_mask=volume_mask)
        out_vol_only_off = model_off(volume_x=volume_x, volume_mask=volume_mask)
    for k in ("surface_preds", "volume_preds"):
        a, b = out_vol_only_off[k], out_vol_only_on[k]
        assert a.shape == b.shape, f"volume-only {k}: shape mismatch {a.shape} vs {b.shape}"
        if a.numel() == 0:
            continue
        max_abs = (a - b).abs().max().item()
        assert max_abs == 0.0, f"volume-only {k}: not bit-identical (max_abs={max_abs})"

    # 7. Non-identity post-training (perturb weights): outputs should differ.
    with torch.no_grad():
        model_on.surface_global_context_proj.weight.add_(
            torch.randn_like(model_on.surface_global_context_proj.weight) * 0.01
        )
        out_on_perturbed = model_on(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
    diff = (out_on_perturbed["surface_preds"] - out_off["surface_preds"]).abs().max().item()
    assert diff > 0.0, "after perturbing proj weights, outputs should differ"

    print("H107 smoke test PASSED")
    print(f"  param_delta = {actual_delta} (expected {expected_delta})")
    print(f"  surface_preds shapes: off={out_off['surface_preds'].shape} on={out_on['surface_preds'].shape}")
    print(f"  identity-at-init: bit-identical across surface_preds, volume_preds, surface_hidden, volume_hidden")
    print(f"  post-perturbation diff: {diff:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
