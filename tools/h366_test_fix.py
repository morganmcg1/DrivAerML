"""H366 quick fix verification: test the model with empty-surface and
asymmetric mask batches to confirm the defensive fix in model.py."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import SurfaceTransolver
from data import SURFACE_X_DIM, VOLUME_X_DIM


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(0)

    model = SurfaceTransolver(
        n_layers=5,
        n_hidden=512,
        dropout=0.0,
        n_head=4,
        mlp_ratio=4,
        slice_num=128,
        rff_num_features=16,
        rff_sigma=1.0,
        rff_init_sigmas=[0.25, 0.5, 1.0, 2.0, 4.0],
        pos_encoding_mode="string_separable",
        use_qk_norm=True,
        use_surf_to_vol_xattn=True,
        drop_path_max=0.10,
        use_knn_attention_bias=True,
        knn_attention_k=32,
    ).to(device).eval()

    # Case 1: empty surface (the smoking gun from the val scan)
    print("\n--- Case 1: empty surface, batch_size=2, volume=39949 ---")
    sx = torch.zeros(2, 0, SURFACE_X_DIM, device=device)
    sm = torch.zeros(2, 0, dtype=torch.bool, device=device)
    vx = torch.randn(2, 39949, VOLUME_X_DIM, device=device)
    vx[:, :, :3] = torch.rand(2, 39949, 3, device=device) * 10 - 5
    vm = torch.ones(2, 39949, dtype=torch.bool, device=device)
    try:
        with torch.no_grad():
            out = model(surface_x=sx, surface_mask=sm, volume_x=vx, volume_mask=vm)
        print(f"  OK: surface_preds={tuple(out['surface_preds'].shape)}, volume_preds={tuple(out['volume_preds'].shape)}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    # Case 2: asymmetric mask (one sample 0 mask, other normal)
    print("\n--- Case 2: surface (2, 39979, 7), mask sums=(0, 39979) ---")
    sx = torch.zeros(2, 39979, SURFACE_X_DIM, device=device)
    sx[1, :, :3] = torch.rand(39979, 3, device=device) * 10 - 5
    sm = torch.zeros(2, 39979, dtype=torch.bool, device=device)
    sm[1, :] = True
    vx = torch.randn(2, 39970, VOLUME_X_DIM, device=device)
    vx[:, :, :3] = torch.rand(2, 39970, 3, device=device) * 10 - 5
    vm = torch.ones(2, 39970, dtype=torch.bool, device=device)
    try:
        with torch.no_grad():
            out = model(surface_x=sx, surface_mask=sm, volume_x=vx, volume_mask=vm)
        print(f"  OK: surface_preds={tuple(out['surface_preds'].shape)}, volume_preds={tuple(out['volume_preds'].shape)}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    # Case 3: pathological (k > n_surf) — defensive skip
    print("\n--- Case 3: surface (1, 10, 7) (n_surf < k+1=33) ---")
    sx = torch.randn(1, 10, SURFACE_X_DIM, device=device)
    sx[:, :, :3] = torch.rand(1, 10, 3, device=device) * 10 - 5
    sm = torch.ones(1, 10, dtype=torch.bool, device=device)
    vx = torch.randn(1, 100, VOLUME_X_DIM, device=device)
    vx[:, :, :3] = torch.rand(1, 100, 3, device=device) * 10 - 5
    vm = torch.ones(1, 100, dtype=torch.bool, device=device)
    try:
        with torch.no_grad():
            out = model(surface_x=sx, surface_mask=sm, volume_x=vx, volume_mask=vm)
        print(f"  OK: surface_preds={tuple(out['surface_preds'].shape)}, volume_preds={tuple(out['volume_preds'].shape)}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    # Case 1b: empty surface under bf16 autocast (matches actual val path)
    print("\n--- Case 1b: empty surface under bf16 autocast ---")
    sx = torch.zeros(2, 0, SURFACE_X_DIM, device=device)
    sm = torch.zeros(2, 0, dtype=torch.bool, device=device)
    vx = torch.randn(2, 39949, VOLUME_X_DIM, device=device)
    vx[:, :, :3] = torch.rand(2, 39949, 3, device=device) * 10 - 5
    vm = torch.ones(2, 39949, dtype=torch.bool, device=device)
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(surface_x=sx, surface_mask=sm, volume_x=vx, volume_mask=vm)
        print(f"  OK: surface_preds={tuple(out['surface_preds'].shape)}, volume_preds={tuple(out['volume_preds'].shape)}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    # Case 4: normal case (sanity check)
    print("\n--- Case 4: normal surface/volume ---")
    sx = torch.randn(2, 40000, SURFACE_X_DIM, device=device)
    sx[:, :, :3] = torch.rand(2, 40000, 3, device=device) * 10 - 5
    sm = torch.ones(2, 40000, dtype=torch.bool, device=device)
    vx = torch.randn(2, 40000, VOLUME_X_DIM, device=device)
    vx[:, :, :3] = torch.rand(2, 40000, 3, device=device) * 10 - 5
    vm = torch.ones(2, 40000, dtype=torch.bool, device=device)
    try:
        with torch.no_grad():
            out = model(surface_x=sx, surface_mask=sm, volume_x=vx, volume_mask=vm)
        print(f"  OK: surface_preds={tuple(out['surface_preds'].shape)}, volume_preds={tuple(out['volume_preds'].shape)}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
