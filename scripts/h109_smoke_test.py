"""H109 smoke test: identity at init + parameter count delta."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from model import SurfaceTransolver


def build(use_h109: bool) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=5,
        n_hidden=512,
        n_head=4,
        mlp_ratio=4,
        slice_num=128,
        rff_num_features=16,
        rff_sigma=1.0,
        rff_init_sigmas=[0.25, 0.5, 1.0, 2.0, 4.0],
        pos_encoding_mode="string_separable",
        use_qk_norm=True,
        use_surf_to_vol_xattn=True,
        use_aux_decoder_heads=True,
        use_backbone_skip_residual_decoder=use_h109,
    )


def main() -> None:
    torch.manual_seed(0)
    model_off = build(use_h109=False).eval()
    torch.manual_seed(0)
    model_on = build(use_h109=True).eval()

    n_off = sum(p.numel() for p in model_off.parameters())
    n_on = sum(p.numel() for p in model_on.parameters())
    delta = n_on - n_off
    expected = 512 * 512 + 512
    print(f"Canonical params (H109 off): {n_off:,}")
    print(f"H109 on params:               {n_on:,}")
    print(f"Delta:                        {delta:,} (expected {expected:,})")
    assert delta == expected, f"unexpected param delta {delta}"

    # Identity-at-init forward parity check.
    bsz = 2
    n_surf = 1024
    n_vol = 512
    surface_x = torch.randn(bsz, n_surf, 7)
    surface_mask = torch.ones(bsz, n_surf, dtype=torch.bool)
    volume_x = torch.randn(bsz, n_vol, 4)
    volume_mask = torch.ones(bsz, n_vol, dtype=torch.bool)

    # The two models were built from the same seed; copy state to be doubly safe
    # except for the new backbone_skip_proj on the H109 model which must remain
    # zero-initialised.
    state_off = model_off.state_dict()
    state_on = model_on.state_dict()
    extra = set(state_on) - set(state_off)
    assert extra == {
        "backbone_skip_proj.weight",
        "backbone_skip_proj.bias",
    }, f"unexpected extra keys: {extra}"
    for k in state_off:
        state_on[k] = state_off[k].clone()
    model_on.load_state_dict(state_on, strict=True)

    # Verify zero-init guarded.
    assert torch.all(model_on.backbone_skip_proj.weight == 0.0)
    assert torch.all(model_on.backbone_skip_proj.bias == 0.0)

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

    for key in ("surface_preds", "volume_preds"):
        diff = (out_on[key] - out_off[key]).abs().max().item()
        print(f"max |H109 - canonical| on {key}: {diff:.3e}")
        assert diff == 0.0, f"identity-at-init violated on {key}: {diff}"

    # Sanity: forward also works in surface-only and volume-only modes.
    out_surf_only = model_on(surface_x=surface_x, surface_mask=surface_mask)
    out_vol_only = model_on(volume_x=volume_x, volume_mask=volume_mask)
    assert out_surf_only["surface_preds"].shape == (bsz, n_surf, 4)
    assert out_vol_only["volume_preds"].shape == (bsz, n_vol, 1)

    print("OK: identity at init AND param delta match expected.")


if __name__ == "__main__":
    main()
