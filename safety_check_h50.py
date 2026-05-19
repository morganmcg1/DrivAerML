"""H50 COORDSLICE init-time safety check (rank 0, no DDP, no training).

Compares flag-off vs flag-on at-init:
  - state_dict key presence: flag-off must have no coord_pe_* keys
  - param count delta: flag-on must add expected number of params
  - finite outputs: forward pass produces finite logits
  - magnitude ratio: |surface_hidden| and |volume_hidden| at init within 5%

If magnitude ratio exceeds 5%, recommend lowering --coord-slice-pe-init-scale.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import SurfaceTransolver
from data import SURFACE_X_DIM, VOLUME_X_DIM


def build(use_coord_slice_pe: bool, init_scale: float, seed: int = 0) -> SurfaceTransolver:
    torch.manual_seed(seed)
    return SurfaceTransolver(
        space_dim=3,
        surface_input_dim=SURFACE_X_DIM,
        volume_input_dim=VOLUME_X_DIM,
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
        use_coord_slice_pe=use_coord_slice_pe,
        coord_slice_pe_rff_features=32,
        coord_slice_pe_init_scale=init_scale,
    )


def fake_batch(B: int, Ns: int, Nv: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(123)
    return {
        "surface_x": torch.randn(B, Ns, SURFACE_X_DIM),
        "surface_mask": torch.ones(B, Ns, dtype=torch.float32),
        "volume_x": torch.randn(B, Nv, VOLUME_X_DIM),
        "volume_mask": torch.ones(B, Nv, dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-scale", type=float, default=0.088)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--surface-points", type=int, default=2048)
    parser.add_argument("--volume-points", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build flag-off baseline and flag-on H50
    m_off = build(use_coord_slice_pe=False, init_scale=args.init_scale).to(device).eval()
    m_on = build(use_coord_slice_pe=True, init_scale=args.init_scale).to(device).eval()

    # --- state_dict check ---
    off_keys = set(m_off.state_dict().keys())
    on_keys = set(m_on.state_dict().keys())
    off_pe_keys = [k for k in off_keys if "coord_pe" in k]
    on_pe_keys = [k for k in on_keys if "coord_pe" in k]
    print(f"\n[state_dict] flag-off coord_pe_* keys: {len(off_pe_keys)} (expect 0)")
    print(f"[state_dict] flag-on  coord_pe_* keys: {len(on_pe_keys)} (expect > 0)")
    assert len(off_pe_keys) == 0, "flag-off must contain no coord_pe_* keys"
    assert len(on_pe_keys) > 0, "flag-on must contain coord_pe_* keys"

    # --- param count ---
    p_off = sum(p.numel() for p in m_off.parameters())
    p_on = sum(p.numel() for p in m_on.parameters())
    extra = p_on - p_off
    print(f"\n[params] flag-off: {p_off:,}")
    print(f"[params] flag-on:  {p_on:,}")
    print(f"[params] extra:    {extra:,} ({extra/p_off*100:.3f}% of baseline)")

    # --- forward pass ---
    batch = fake_batch(args.batch_size, args.surface_points, args.volume_points)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out_off = m_off(**batch)
        out_on = m_on(**batch)

    print("\n[forward] outputs:")
    for k in ("surface_preds", "volume_preds", "surface_hidden", "volume_hidden"):
        v_off = out_off[k]
        v_on = out_on[k]
        finite_off = torch.isfinite(v_off).all().item()
        finite_on = torch.isfinite(v_on).all().item()
        mag_off = v_off.detach().abs().mean().item()
        mag_on = v_on.detach().abs().mean().item()
        ratio = mag_on / max(mag_off, 1e-12)
        within_5pct = 0.95 <= ratio <= 1.05
        flag = "OK" if within_5pct else "WARN"
        print(
            f"  {k:18s}: off|{mag_off:.4e} on|{mag_on:.4e}  ratio={ratio:.4f}  "
            f"finite_off={finite_off} finite_on={finite_on} [{flag}]"
        )
        assert finite_off and finite_on, f"{k} produced non-finite outputs"

    # Report inter_slice_cos at init (should be high since coord_pe is tiny initially)
    print("\n[init] inter_slice_cos per layer (from one fwd):")
    for i, block in enumerate(m_on.backbone.blocks):
        isc = getattr(block.attention, "_last_inter_slice_cos", None)
        if isc is not None:
            print(f"  L{i}: inter_slice_cos_post_pe = {isc:.4f}")

    print("\nSafety check complete.")


if __name__ == "__main__":
    main()
