"""H363 step-0 invariant smoke check.

Builds two SurfaceTransolver instances with identical seeds — one with the
MoE residual decoder enabled (zero-init experts), one without — and verifies
that their surface predictions are bit-identical on a random forward pass.
This validates the smoke-gate claim that zero-init expert output Linears
preserve the EP13 base-head prediction at step 0.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from model import SurfaceTransolver  # noqa: E402


def build_model(moe: bool, num_experts: int = 2) -> SurfaceTransolver:
    torch.manual_seed(0)
    return SurfaceTransolver(
        space_dim=3,
        surface_input_dim=11,
        volume_input_dim=8,
        surface_output_dim=4,
        volume_output_dim=1,
        n_layers=2,
        n_hidden=64,
        n_head=2,
        slice_num=8,
        mlp_ratio=2,
        dropout=0.0,
        rff_num_features=4,
        rff_init_sigmas=(0.5, 1.0, 2.0, 4.0),
        pos_encoding_mode="string_separable",
        use_qk_norm=True,
        use_surf_to_vol_xattn=False,
        drop_path_max=0.0,
        moe_surface_decoder=moe,
        moe_num_experts=num_experts,
    )


def main() -> int:
    base = build_model(moe=False).eval()
    moe = build_model(moe=True, num_experts=2).eval()

    # Copy shared weights from base into moe so only the MoE-extra parameters
    # differ. Both models were built with the same seed but slight RNG drift
    # from the extra Module construction can offset later inits; we restore
    # bit-identity for the shared backbone.
    moe_state = moe.state_dict()
    base_state = base.state_dict()
    for k, v in base_state.items():
        if k in moe_state:
            moe_state[k].copy_(v)
    moe.load_state_dict(moe_state, strict=False)

    # Sanity: expert output weights/biases must be zero (batched einsum form).
    assert torch.all(moe.moe_decoder.W2 == 0), "MoE W2 has nonzero entries"
    assert torch.all(moe.moe_decoder.b2 == 0), "MoE b2 has nonzero entries"
    print("expert output Linears: ZERO (OK)")

    # Build a tiny random batch.
    torch.manual_seed(123)
    B = 2
    N_s = 32
    N_v = 16
    surface_x = torch.randn(B, N_s, 11)
    surface_mask = torch.ones(B, N_s)
    volume_x = torch.randn(B, N_v, 8)
    volume_mask = torch.ones(B, N_v)

    with torch.no_grad():
        out_base = base(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
        out_moe = moe(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )

    diff_surf = (out_base["surface_preds"] - out_moe["surface_preds"]).abs().max().item()
    diff_vol = (out_base["volume_preds"] - out_moe["volume_preds"]).abs().max().item()
    print(f"max |Δ surface_preds| = {diff_surf:.3e}")
    print(f"max |Δ volume_preds|  = {diff_vol:.3e}")
    assert "moe_probs" in out_moe and "moe_probs" not in out_base
    print(f"moe_probs shape       = {tuple(out_moe['moe_probs'].shape)}")
    # Router probs must be softmax over experts.
    p_sum = out_moe["moe_probs"].sum(dim=-1)
    print(f"router probs sum→1?   max|sum−1| = {(p_sum - 1.0).abs().max().item():.3e}")

    ok = diff_surf < 1e-6 and diff_vol < 1e-6
    print("STEP-0 INVARIANT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
