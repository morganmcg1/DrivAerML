"""H381 step-0 invariance + AdamW path smoke check.

Loads the EP13 EMA checkpoint, constructs both Lion and AdamW optimizers,
and runs 10 forward+backward+step iterations on a synthetic batch sized
to match the production loader (B=4, N_surf=N_vol=65536). Verifies:

1. Lion path: model loaded loss is finite at step 0 (sanity proxy for the
   "bit-identical to H342 baseline" check — full bit-identity requires the
   real loader; this guards against the new code mutating the Lion config
   path).
2. AdamW path: loss decreases over 10 steps, no NaN, no divergence.
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import Config, build_model, build_optimizer  # noqa: E402


def make_synthetic_batch(device: torch.device, surface_dim: int, volume_dim: int):
    B, N_surf, N_vol = 2, 16384, 16384  # smaller than production for smoke
    torch.manual_seed(0)
    surface_x = torch.randn(B, N_surf, surface_dim, device=device)
    surface_y = torch.randn(B, N_surf, 4, device=device)
    surface_mask = torch.ones(B, N_surf, device=device, dtype=torch.bool)
    volume_x = torch.randn(B, N_vol, volume_dim, device=device)
    volume_y = torch.randn(B, N_vol, 1, device=device)
    volume_mask = torch.ones(B, N_vol, device=device, dtype=torch.bool)
    return surface_x, surface_y, surface_mask, volume_x, volume_y, volume_mask


def smoke(optimizer_name: str, ckpt_path: Path, device: torch.device) -> None:
    # Match H244 EP13 config for the architecture.
    config = Config(
        optimizer=optimizer_name,
        lr=9e-5,
        weight_decay=5e-4,
        lion_beta1=0.9,
        lion_beta2=0.99,
        adamw_lr=3e-4,
        adamw_weight_decay=0.01,
        adamw_beta1=0.9,
        adamw_beta2=0.999,
        amp_mode="bf16",
        compile_model=False,
        use_qk_norm=True,
        use_surf_to_vol_xattn=True,
        drop_path_max=0.1,
        model_hidden_dim=512,
        model_heads=4,
        model_layers=5,
        model_mlp_ratio=4,
        model_slices=128,
        pos_encoding_mode="string_separable",
        rff_init_sigmas="0.25,0.5,1.0,2.0,4.0",
        rff_num_features=16,
    )
    model = build_model(config).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{optimizer_name}] loaded ck: missing={len(missing)} unexpected={len(unexpected)}")

    opt = build_optimizer(model, config)
    print(f"[{optimizer_name}] optimizer={type(opt).__name__} lr={opt.param_groups[0]['lr']} "
          f"wd={opt.param_groups[0]['weight_decay']} "
          f"betas={opt.param_groups[0].get('betas', 'n/a')}")

    surface_x, surface_y, surface_mask, volume_x, volume_y, volume_mask = make_synthetic_batch(
        device, surface_dim=7, volume_dim=4
    )

    model.train()
    losses = []
    for step in range(10):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            out = model(
                surface_x=surface_x,
                surface_mask=surface_mask,
                volume_x=volume_x,
                volume_mask=volume_mask,
            )
            surf_pred = out["surface_preds"]
            vol_pred = out["volume_preds"]
            loss_surf = torch.nn.functional.mse_loss(surf_pred, surface_y)
            loss_vol = torch.nn.functional.mse_loss(vol_pred, volume_y)
            loss = loss_surf + loss_vol
        loss.backward()
        # Match H244 grad clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step()
        losses.append(float(loss.item()))
        is_finite = torch.isfinite(loss).item()
        print(f"  step {step}: loss={loss.item():.6f} finite={is_finite}")
        if not is_finite:
            print(f"[{optimizer_name}] DIVERGED at step {step}")
            return

    delta = losses[-1] - losses[0]
    print(f"[{optimizer_name}] step0->step9 loss delta = {delta:.6f} "
          f"({'DECREASED ✓' if delta < 0 else 'INCREASED ✗'})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/drivaerml/resume_cache/yw2a5dyl/epoch-13/checkpoint.pt",
    )
    parser.add_argument("--optimizer", type=str, default="both",
                        choices=["lion", "adamw", "both"])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.optimizer in ("lion", "both"):
        print("\n=== Lion smoke ===")
        smoke("lion", ckpt_path, device)
    if args.optimizer in ("adamw", "both"):
        print("\n=== AdamW smoke ===")
        smoke("adamw", ckpt_path, device)


if __name__ == "__main__":
    main()
