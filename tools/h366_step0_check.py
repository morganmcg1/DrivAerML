"""
H366 step-0 invariance check.

Verifies that enabling --use-knn-attention-bias produces outputs that are
bit-identical to the baseline at initialisation (step 0), because the
knn_bias_alpha parameter is zero-initialised (ReZero pattern).

Usage:
    python tools/h366_step0_check.py
"""

import sys
import os

# Allow running from the target repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import SurfaceTransolver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- model config matching the baseline run (H342 SOTA) ----
MODEL_KWARGS = dict(
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
)


def build_baseline() -> SurfaceTransolver:
    return SurfaceTransolver(**MODEL_KWARGS, use_knn_attention_bias=False).to(DEVICE)


def build_knn() -> SurfaceTransolver:
    return SurfaceTransolver(**MODEL_KWARGS, use_knn_attention_bias=True, knn_attention_k=32).to(DEVICE)


def copy_shared_weights(src: SurfaceTransolver, dst: SurfaceTransolver) -> None:
    """Copy every parameter that exists in both models by name."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for name in dst_sd:
        if name in src_sd:
            dst_sd[name].copy_(src_sd[name])
        # knn_bias_alpha parameters are not in src; they stay at their init value (0).
    dst.load_state_dict(dst_sd)


def make_batch(n_surf: int = 128, n_vol: int = 256, batch_size: int = 1):
    """Create a minimal random batch with fixed mesh topology."""
    from data import SURFACE_X_DIM, VOLUME_X_DIM
    torch.manual_seed(42)
    # surface_x: (B, N_surf, SURFACE_X_DIM).  [:3] holds 3D coordinates.
    surface_x = torch.randn(batch_size, n_surf, SURFACE_X_DIM, device=DEVICE)
    # Give the first 3 dims plausible coordinate magnitudes for kNN.
    surface_x[:, :, :3] = torch.rand(batch_size, n_surf, 3, device=DEVICE) * 2.0 - 1.0
    surface_mask = torch.ones(batch_size, n_surf, dtype=torch.bool, device=DEVICE)

    volume_x = torch.randn(batch_size, n_vol, VOLUME_X_DIM, device=DEVICE)
    volume_x[:, :, :3] = torch.rand(batch_size, n_vol, 3, device=DEVICE) * 2.0 - 1.0
    volume_mask = torch.ones(batch_size, n_vol, dtype=torch.bool, device=DEVICE)

    return surface_x, surface_mask, volume_x, volume_mask


def run_forward(model: SurfaceTransolver, batch):
    surface_x, surface_mask, volume_x, volume_mask = batch
    with torch.no_grad():
        out = model(
            surface_x=surface_x,
            surface_mask=surface_mask,
            volume_x=volume_x,
            volume_mask=volume_mask,
        )
    return out


def main():
    print(f"Device: {DEVICE}")
    print("Building baseline model (use_knn_attention_bias=False)...")
    baseline = build_baseline()
    baseline.eval()

    print("Building kNN model (use_knn_attention_bias=True)...")
    knn_model = build_knn()
    knn_model.eval()

    print("Copying shared weights from baseline to kNN model...")
    copy_shared_weights(baseline, knn_model)

    # Sanity: all knn_bias_alpha params must be exactly 0.
    alpha_params = [
        (name, p)
        for name, p in knn_model.named_parameters()
        if "knn_bias_alpha" in name
    ]
    assert len(alpha_params) > 0, "No knn_bias_alpha parameters found — wiring may be broken."
    for name, p in alpha_params:
        assert p.item() == 0.0, f"{name} is not zero at init: {p.item()}"
    print(f"  knn_bias_alpha params found: {len(alpha_params)}, all zero. OK")

    # Forward pass.
    batch = make_batch(n_surf=128, n_vol=256, batch_size=1)
    print("Running baseline forward pass...")
    out_base = run_forward(baseline, batch)
    print("Running kNN forward pass...")
    out_knn = run_forward(knn_model, batch)

    # Compare surface and volume predictions.
    surf_diff = (out_base["surface_preds"] - out_knn["surface_preds"]).abs().max().item()
    vol_diff = (out_base["volume_preds"] - out_knn["volume_preds"]).abs().max().item()

    print(f"  max |surface_preds baseline - knn|: {surf_diff:.2e}")
    print(f"  max |volume_preds  baseline - knn|: {vol_diff:.2e}")

    threshold = 1e-5
    ok = surf_diff < threshold and vol_diff < threshold
    if ok:
        print(f"PASS: step-0 invariance holds (threshold {threshold:.0e}).")
    else:
        print(f"FAIL: outputs differ beyond threshold {threshold:.0e}.")
        sys.exit(1)

    # Also check parameter count overhead.
    base_params = sum(p.numel() for p in baseline.parameters())
    knn_params = sum(p.numel() for p in knn_model.parameters())
    extra = knn_params - base_params
    overhead_pct = 100.0 * extra / base_params
    print(
        f"  Param overhead: +{extra} params ({overhead_pct:.4f}%) "
        f"[{base_params} -> {knn_params}]"
    )
    assert extra <= 5, (
        f"Parameter overhead {extra} exceeds the <=5 new params constraint. "
        f"(one scalar per attention layer = n_layers={MODEL_KWARGS['n_layers']})"
    )
    print("PASS: parameter overhead within constraint.")


if __name__ == "__main__":
    main()
