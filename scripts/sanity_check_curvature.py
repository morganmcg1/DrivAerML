"""Sanity-check curvature additive bias.

Verifies that at step 0 (with zero-init final layer of CurvatureAttentionBias),
the forward pass produces output bit-identical to the same model without
curvature, regardless of the curvature input. This is the property the PR spec
relies on so the run starts from the SOTA baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import CurvatureAttentionBias, SurfaceTransolver  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    hidden = 64
    space_dim = 3
    surface_input_dim = 7
    volume_input_dim = 4
    B = 2
    N_surf = 17
    N_vol = 11

    # Build two models with shared weights except the curvature bias.
    common_kwargs = dict(
        space_dim=space_dim,
        surface_input_dim=surface_input_dim,
        volume_input_dim=volume_input_dim,
        n_hidden=hidden,
        n_layers=2,
        n_head=4,
        slice_num=32,
        surface_output_dim=4,
        volume_output_dim=1,
    )
    model_no_curv = SurfaceTransolver(**common_kwargs, use_curvature_attention_bias=False).to(device)
    model_with_curv = SurfaceTransolver(**common_kwargs, use_curvature_attention_bias=True).to(device)

    # Copy weights from no_curv to with_curv so they match except for the bias module.
    no_state = model_no_curv.state_dict()
    with_state = model_with_curv.state_dict()
    shared_keys = set(no_state) & set(with_state)
    for k in shared_keys:
        with_state[k] = no_state[k].clone()
    model_with_curv.load_state_dict(with_state, strict=True)

    # Confirm net.2 of curvature bias module is zero-init.
    bias_mod = model_with_curv.curvature_attn_bias
    assert bias_mod is not None
    final = bias_mod.net[-1]
    fc2_w = final.weight.abs().max().item()
    fc2_b = final.bias.abs().max().item()
    print(f"net[-1].weight max abs = {fc2_w}")
    print(f"net[-1].bias   max abs = {fc2_b}")
    assert fc2_w == 0.0 and fc2_b == 0.0, "Zero-init violated"

    # Fabricate inputs.
    surface_x = torch.randn(B, N_surf, surface_input_dim, device=device)
    volume_x = torch.randn(B, N_vol, volume_input_dim, device=device)
    surface_mask = torch.ones(B, N_surf, dtype=torch.bool, device=device)
    surface_mask[1, -3:] = False
    volume_mask = torch.ones(B, N_vol, dtype=torch.bool, device=device)
    volume_mask[0, -2:] = False
    # Non-trivial curvature input.
    surface_curvature = torch.randn(B, N_surf, 3, device=device) * 10.0

    model_no_curv.eval()
    model_with_curv.eval()

    with torch.no_grad():
        out_a = model_no_curv(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
        out_b = model_with_curv(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
            surface_curvature=surface_curvature,
        )

    surf_diff = (out_a["surface_preds"] - out_b["surface_preds"]).abs().max().item()
    vol_diff = (out_a["volume_preds"] - out_b["volume_preds"]).abs().max().item()
    print(f"surface_preds max abs diff: {surf_diff:.3e}")
    print(f"volume_preds  max abs diff: {vol_diff:.3e}")

    # Bit-identical because the bias contribution is exactly zero.
    assert surf_diff == 0.0, f"Surface preds diverged at step 0! diff={surf_diff}"
    assert vol_diff == 0.0, f"Volume preds diverged at step 0! diff={vol_diff}"

    # Verify after perturbing the final weight, outputs change (so we know the bias IS active).
    with torch.no_grad():
        final.weight.add_(0.01 * torch.randn_like(final.weight))
        out_c = model_with_curv(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
            surface_curvature=surface_curvature,
        )
    surf_after = (out_a["surface_preds"] - out_c["surface_preds"]).abs().max().item()
    print(f"after perturbing fc2: surface_preds max abs diff = {surf_after:.3e}")
    assert surf_after > 0.0, "Bias module appears disconnected: perturbation did not change output"

    # Also confirm no-curvature path still works with use_curvature_attention_bias=True (when curvature=None).
    with torch.no_grad():
        out_d = model_with_curv(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
            surface_curvature=None,
        )
    print(f"with bias enabled but curvature=None: surface_preds shape {out_d['surface_preds'].shape}")

    # Count the bias parameters.
    bias_params = sum(p.numel() for p in bias_mod.parameters())
    total_params = sum(p.numel() for p in model_with_curv.parameters())
    print(f"CurvatureAttentionBias params: {bias_params} (~{100 * bias_params / total_params:.3f}% of model)")
    print("OK: zero-init forward identity verified.")


if __name__ == "__main__":
    main()
