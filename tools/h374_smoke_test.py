"""H374 smoke test: verify teacher-loading and consistency-loss compute path.

Confirms:
- Teacher (EP13 EMA) loads with zero missing/unexpected keys against the
  H336/H342 model recipe.
- Teacher is in eval(), requires_grad=False on all params.
- A single forward+backward with --self-consistency-weight 0.1 produces a
  finite consistency loss and nonzero student gradient.
- consistency_loss is identically zero when student == teacher (sanity).
- consistency_loss is nonzero after a perturbation of the student weights.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    Config,
    SURFACE_CHANNEL_INDEX,
    build_model,
    parse_self_consistency_channels,
    train_loss,
)
from trainer_runtime import TargetTransform  # noqa: E402


def make_config() -> Config:
    cfg = Config(
        model_layers=5,
        model_hidden_dim=512,
        model_heads=4,
        model_mlp_ratio=4,
        model_slices=128,
        use_qk_norm=True,
        use_surf_to_vol_xattn=True,
        rff_num_features=16,
        rff_init_sigmas="0.25,0.5,1.0,2.0,4.0",
        pos_encoding_mode="string_separable",
        drop_path_max=0.1,
        teacher_ckpt="outputs/drivaerml/resume_cache/yw2a5dyl/epoch-13/checkpoint.pt",
        self_consistency_weight=0.1,
        self_consistency_channels="wss_z",
        self_consistency_loss="mse",
        tau_y_loss_weight=1.3,
        tau_z_loss_weight=1.67,
        surface_loss_weight=2.0,
        volume_loss_weight=0.5,
    )
    return cfg


def make_fake_batch(device, B=2, N_surface=512, N_volume=256):
    from data.loader import SurfaceBatch

    surface_x = torch.randn(B, N_surface, 7, device=device)
    surface_y = torch.randn(B, N_surface, 4, device=device)
    surface_mask = torch.ones(B, N_surface, device=device, dtype=torch.bool)
    volume_x = torch.randn(B, N_volume, 4, device=device)
    volume_y = torch.randn(B, N_volume, 1, device=device)
    volume_mask = torch.ones(B, N_volume, device=device, dtype=torch.bool)
    return SurfaceBatch(
        case_ids=[f"case_{i}" for i in range(B)],
        surface_x=surface_x,
        surface_y=surface_y,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_y=volume_y,
        volume_mask=volume_mask,
        metadata=[{} for _ in range(B)],
    )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = make_config()

    print(f"Device: {device}")

    teacher_path = Path(cfg.teacher_ckpt)
    assert teacher_path.is_file(), f"teacher ckpt not found: {teacher_path}"
    print(f"Teacher checkpoint: {teacher_path}")

    # Build student and teacher with the same recipe; load teacher weights.
    torch.manual_seed(0)
    student = build_model(cfg).to(device)
    teacher = build_model(cfg).to(device)
    ck = torch.load(str(teacher_path), map_location="cpu", weights_only=False)
    sd = ck["model"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    print(f"Teacher load: missing={len(missing)} unexpected={len(unexpected)}")
    assert len(missing) == 0, f"unexpected missing keys: {missing[:10]}"
    assert len(unexpected) == 0, f"unexpected extra keys: {unexpected[:10]}"

    # Also load student from same checkpoint so consistency_loss starts at 0.
    student.load_state_dict(sd, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    n_grad = sum(int(p.requires_grad) for p in teacher.parameters())
    assert n_grad == 0, f"teacher has {n_grad} trainable params"

    indices = parse_self_consistency_channels(cfg.self_consistency_channels)
    assert indices == [SURFACE_CHANNEL_INDEX["wss_z"]], f"got {indices}"
    print(f"Consistency channel indices: {indices}")

    batch = make_fake_batch(device)
    transform = TargetTransform(
        surface_y_mean=torch.zeros(4, device=device),
        surface_y_std=torch.ones(4, device=device),
        volume_y_mean=torch.zeros(1, device=device),
        volume_y_std=torch.ones(1, device=device),
    )
    surface_weights = torch.tensor(
        [1.0, 1.0, cfg.tau_y_loss_weight, cfg.tau_z_loss_weight], device=device
    )

    # Case 1a: student.eval() == teacher.eval() → no drop_path/dropout noise,
    # consistency loss should be effectively 0 modulo bf16 rounding.
    student.eval()
    loss_eq, metrics_eq = train_loss(
        student,
        batch,
        transform,
        device,
        amp_mode="bf16",
        surface_loss_weight=cfg.surface_loss_weight,
        volume_loss_weight=cfg.volume_loss_weight,
        surface_channel_weights=surface_weights,
        teacher_model=teacher,
        self_consistency_weight=cfg.self_consistency_weight,
        self_consistency_channel_indices=indices,
        self_consistency_loss_type=cfg.self_consistency_loss,
    )
    print(f"Case 1a (student.eval()==teacher) metrics: {metrics_eq}")
    cl_eq = metrics_eq.get("consistency_loss", float("nan"))
    assert cl_eq is not None and cl_eq < 1e-5, (
        f"Expected consistency_loss ~0 when student.eval()==teacher, got {cl_eq}"
    )
    assert torch.isfinite(loss_eq), f"loss non-finite: {loss_eq}"

    # Case 1b: student.train() (drop_path active) vs teacher.eval() — should be
    # small but nonzero due to drop_path noise (this is the real training case).
    student.train()
    _, metrics_train = train_loss(
        student,
        batch,
        transform,
        device,
        amp_mode="bf16",
        surface_loss_weight=cfg.surface_loss_weight,
        volume_loss_weight=cfg.volume_loss_weight,
        surface_channel_weights=surface_weights,
        teacher_model=teacher,
        self_consistency_weight=cfg.self_consistency_weight,
        self_consistency_channel_indices=indices,
        self_consistency_loss_type=cfg.self_consistency_loss,
    )
    cl_train = metrics_train["consistency_loss"]
    print(f"Case 1b (student.train()==teacher, drop_path noise) cl={cl_train:.6f}")
    assert 0.0 <= cl_train < 1.0, f"unexpected drop_path-induced cl: {cl_train}"

    # Case 2: perturb student → consistency_loss must be > 0.
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p) * 1e-2)
    student.zero_grad()
    loss_diff, metrics_diff = train_loss(
        student,
        batch,
        transform,
        device,
        amp_mode="bf16",
        surface_loss_weight=cfg.surface_loss_weight,
        volume_loss_weight=cfg.volume_loss_weight,
        surface_channel_weights=surface_weights,
        teacher_model=teacher,
        self_consistency_weight=cfg.self_consistency_weight,
        self_consistency_channel_indices=indices,
        self_consistency_loss_type=cfg.self_consistency_loss,
    )
    print(f"Case 2 (perturbed student) metrics: {metrics_diff}")
    cl_diff = metrics_diff["consistency_loss"]
    cl_w = metrics_diff["consistency_loss_weighted"]
    assert cl_diff > 1e-4, f"Expected consistency_loss >0 after perturbation, got {cl_diff}"
    assert abs(cl_w - cl_diff * cfg.self_consistency_weight) < 1e-9
    assert torch.isfinite(loss_diff)

    # Backward — student must receive nonzero gradient through the consistency
    # path.  Verify gradient flows through the surface output head AND the
    # shared backbone (encoder), not just the head.
    loss_diff.backward()
    backbone_grad_norm = 0.0
    surface_out_grad_norm = 0.0
    volume_out_grad_norm = 0.0
    for n, p in student.named_parameters():
        if p.grad is None:
            continue
        gn = float(p.grad.detach().norm().item())
        if n.startswith("surface_out") or n.startswith("surface_bias"):
            surface_out_grad_norm += gn
        elif n.startswith("volume_out") or n.startswith("volume_bias"):
            volume_out_grad_norm += gn
        elif n.startswith("backbone"):
            backbone_grad_norm += gn
    print(f"Backbone gradient norm sum: {backbone_grad_norm:.4f}")
    print(f"Surface-out gradient norm sum: {surface_out_grad_norm:.4f}")
    print(f"Volume-out gradient norm sum: {volume_out_grad_norm:.4f}")
    assert backbone_grad_norm > 0.0, "no gradient flowed into backbone"
    assert surface_out_grad_norm > 0.0, "no gradient flowed into surface head"
    assert volume_out_grad_norm > 0.0, "volume head should still receive base-loss gradient"

    # Case 3: teacher param.grad must remain None.
    teacher_has_grad = any(p.grad is not None for p in teacher.parameters())
    assert not teacher_has_grad, "teacher params received grad!"

    print("\nH374 smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
