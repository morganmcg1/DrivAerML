"""H123 PR #1299 smoke test: physical-space tangent projection.

Verifies:
1. Buffer registration succeeds with use_wss_tangent_projection=True.
2. Forward pass yields tau_pred such that (tau_pred_phys . n_hat) ~ 0
   after projection, where tau_pred_phys = tau_pred_norm * std + mean.
3. Diagnostic keys are emitted with the _phys suffix.
4. Parameter count matches the off variant (buffers don't count).
"""
from __future__ import annotations

import torch

from train import Config, build_model


_REAL_STATS = {
    "surface_y_mean": torch.tensor(
        [-0.3038, -1.2007, 0.0015, -0.0721], dtype=torch.float32
    ),
    "surface_y_std": torch.tensor(
        [0.3563, 2.0769, 1.3564, 1.1143], dtype=torch.float32
    ),
    "volume_y_mean": torch.tensor([0.0], dtype=torch.float32),
    "volume_y_std": torch.tensor([1.0], dtype=torch.float32),
}


def _baseline_config() -> Config:
    cfg = Config()
    cfg.model_layers = 2
    cfg.model_hidden_dim = 64
    cfg.model_heads = 2
    cfg.model_mlp_ratio = 2
    cfg.model_slices = 8
    cfg.rff_num_features = 0
    cfg.pos_encoding_mode = "sincos"
    cfg.amp_mode = "fp32"
    return cfg


def _fake_batch(device, batch_size=2, n_surface=128, n_volume=128):
    # surface_x layout: [xyz(3), normals(3), area(1)] per loader.py:288
    xyz = torch.randn(batch_size, n_surface, 3, device=device)
    # Random unit normals
    raw_normals = torch.randn(batch_size, n_surface, 3, device=device)
    normals = raw_normals / raw_normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    area = torch.rand(batch_size, n_surface, 1, device=device)
    surface_x = torch.cat([xyz, normals, area], dim=-1)
    # volume_x layout: [xyz(3), sdf(1)]
    volume_x = torch.randn(batch_size, n_volume, 4, device=device)
    surface_mask = torch.ones(batch_size, n_surface, dtype=torch.bool, device=device)
    volume_mask = torch.ones(batch_size, n_volume, dtype=torch.bool, device=device)
    return surface_x, volume_x, surface_mask, volume_mask, normals


def main():
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_off = _baseline_config()
    cfg_on = _baseline_config()
    cfg_on.use_wss_tangent_projection = True

    torch.manual_seed(7)
    model_off = build_model(cfg_off, stats=_REAL_STATS).to(device)
    torch.manual_seed(7)
    model_on = build_model(cfg_on, stats=_REAL_STATS).to(device)

    n_off = sum(p.numel() for p in model_off.parameters())
    n_on = sum(p.numel() for p in model_on.parameters())
    print(f"params off={n_off}  on={n_on}  delta={n_on - n_off}")
    assert n_off == n_on, "tangent projection must add zero params (buffers only)"

    # Verify buffers
    assert hasattr(model_on, "tau_phys_mean")
    assert hasattr(model_on, "tau_phys_std")
    assert torch.allclose(
        model_on.tau_phys_mean.cpu(), _REAL_STATS["surface_y_mean"][1:4]
    )
    assert torch.allclose(
        model_on.tau_phys_std.cpu(), _REAL_STATS["surface_y_std"][1:4]
    )
    print(
        f"buffers loaded: tau_phys_mean={model_on.tau_phys_mean.tolist()} "
        f"tau_phys_std={model_on.tau_phys_std.tolist()}"
    )
    assert not hasattr(model_off, "tau_phys_mean")

    # Forward pass on the projection-on model
    model_on.eval()
    surface_x, volume_x, surface_mask, volume_mask, normals = _fake_batch(device)
    with torch.no_grad():
        out = model_on(
            surface_x=surface_x,
            volume_x=volume_x,
            surface_mask=surface_mask,
            volume_mask=volume_mask,
        )
    assert "wss_tangent_diag" in out, "missing wss_tangent_diag in forward output"
    diag = out["wss_tangent_diag"]
    for k in (
        "pre_proj_normal_component_abs_mean_phys",
        "pre_proj_normal_component_rel_mean_phys",
        "post_proj_normal_component_abs_mean_phys",
    ):
        assert k in diag, f"missing diag key {k}"
        print(f"  diag[{k}] = {diag[k].item():.6e}")

    # The MUST-HOLD sanity assertion: post-projection normal residual is zero
    post = diag["post_proj_normal_component_abs_mean_phys"].item()
    assert post < 1e-5, f"post_proj normal residual {post:.6e} should be < 1e-5"

    # Independently verify by re-computing physical projection on the output
    tau_norm = out["surface_preds"][..., 1:4]
    mean = model_on.tau_phys_mean.to(dtype=tau_norm.dtype)
    std = model_on.tau_phys_std.to(dtype=tau_norm.dtype)
    tau_phys = tau_norm * std + mean
    tau_dot_n = (tau_phys * normals).sum(dim=-1)
    abs_mean = tau_dot_n.abs().mean().item()
    print(f"independent (tau_phys . n).abs().mean() = {abs_mean:.6e}")
    assert abs_mean < 1e-4, f"physical-space tangency violated: {abs_mean:.6e}"

    # Now compare against off-variant: same inputs, different outputs (projection kills DoF)
    model_off.load_state_dict(
        {k: v for k, v in model_on.state_dict().items() if k in model_off.state_dict()},
        strict=False,
    )
    model_off.eval()
    with torch.no_grad():
        out_off = model_off(
            surface_x=surface_x,
            volume_x=volume_x,
            surface_mask=surface_mask,
            volume_mask=volume_mask,
        )
    diff = (out["surface_preds"][..., 1:4] - out_off["surface_preds"][..., 1:4]).abs().mean().item()
    print(f"|tau_pred_on - tau_pred_off|.mean() = {diff:.6e}")
    assert diff > 0.0, "projection should produce different tau output"

    # cp channel must be untouched by projection
    cp_diff = (out["surface_preds"][..., 0:1] - out_off["surface_preds"][..., 0:1]).abs().mean().item()
    print(f"|cp_on - cp_off|.mean() = {cp_diff:.6e}")
    assert cp_diff < 1e-6, "cp channel should be untouched by projection"

    # bf16 autocast + backward sanity (the actual training path)
    if device.type == "cuda":
        model_on.train()
        for p in model_on.parameters():
            p.grad = None
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_amp = model_on(
                surface_x=surface_x,
                volume_x=volume_x,
                surface_mask=surface_mask,
                volume_mask=volume_mask,
            )
            tau_pred = out_amp["surface_preds"][..., 1:4]
            loss = (tau_pred ** 2).mean()
        loss.backward()
        post_amp = out_amp["wss_tangent_diag"][
            "post_proj_normal_component_abs_mean_phys"
        ].item()
        # bf16 has ~8-bit mantissa so post-projection residual is larger than fp32.
        # tau_phys_mean magnitudes are O(1), normals are unit, so post_proj is
        # at the bf16 epsilon scale (~few e-2 absolute). Sanity: < 0.1.
        print(f"bf16 post_proj_abs_mean = {post_amp:.6e}")
        assert post_amp < 0.1, f"bf16 post_proj should be small, got {post_amp:.6e}"
        n_grad = sum(1 for p in model_on.parameters() if p.grad is not None)
        n_tot = sum(1 for _ in model_on.parameters())
        # Surface-only loss leaves the volume head with no grad — expected.
        # Just check the projection path itself didn't break gradient flow.
        print(f"grads populated: {n_grad}/{n_tot} params (surface-only loss)")
        assert n_grad >= n_tot - 8, f"projection path broke grad flow ({n_grad}/{n_tot})"
        # Verify surface_out specifically gets grads — it's the layer directly
        # before the projection, so a broken projection backward would zero its grads.
        surf_out_grad_norms = [
            p.grad.norm().item() for p in model_on.surface_out.parameters()
            if p.grad is not None
        ]
        print(f"surface_out grad norms: {surf_out_grad_norms}")
        assert all(g > 0.0 for g in surf_out_grad_norms), "surface_out grad flow is broken"

    print("\nH123 PHYSICAL-SPACE TANGENT PROJECTION smoke test PASSED.")


if __name__ == "__main__":
    main()
