"""Standalone test-eval-only script for PR #935.

Loads a saved checkpoint (best-val-by-abupt EMA snapshot) and runs
evaluate_split on the test loader, mirroring run_final_evaluation's
test_primary/* metric layout.

Single-GPU eval is used (test split is ~50 cases, ~few minutes).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import wandb

from train import Config, build_model
from trainer_runtime import (
    TargetTransform,
    evaluate_split,
    full_eval_loaders_from,
    make_loaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PR #935 test-eval-only")
    parser.add_argument(
        "--checkpoint",
        default="outputs/drivaerml/run-ixrg3mg1/checkpoint.pt",
        help="Path to checkpoint.pt to evaluate.",
    )
    parser.add_argument(
        "--source-run-id",
        default="ixrg3mg1",
        help="W&B run id of the training run that produced the checkpoint.",
    )
    parser.add_argument("--wandb-name", default="tanjiro/sdf-vol-pe-test-eval-ixrg3mg1")
    parser.add_argument("--wandb-group", default="tanjiro-sdf-vol-pe-identity-init")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _print_split(label: str, m: dict[str, float]) -> None:
    print(f"\n=== {label} ===")
    for k in [
        "abupt_axis_mean_rel_l2_pct",
        "surface_pressure_rel_l2_pct",
        "volume_pressure_rel_l2_pct",
        "wall_shear_rel_l2_pct",
        "wall_shear_x_rel_l2_pct",
        "wall_shear_y_rel_l2_pct",
        "wall_shear_z_rel_l2_pct",
    ]:
        if k in m:
            print(f"  {k}: {m[k]:.4f}")
    cases_keys = [k for k in ("cases", "surface_cases", "volume_cases") if k in m]
    if cases_keys:
        print("  " + ", ".join(f"{k}={int(m[k])}" for k in cases_keys))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint).resolve()
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["config"])

    cfg_dict["num_workers"] = args.num_workers
    cfg_dict["compile_model"] = False
    cfg = Config(**cfg_dict)
    print(f"  source_epoch={ckpt.get('epoch')} checkpoint_source={ckpt.get('checkpoint_source')}")
    print(f"  selection_metric={ckpt.get('selection_metric')}")
    print(f"  use_sdf_vol_pe={cfg.use_sdf_vol_pe} use_surf_to_vol_xattn={cfg.use_surf_to_vol_xattn}")

    print("\nBuilding loaders…")
    _train_loader, val_loaders, test_loaders, stats = make_loaders(cfg, distributed_state=None)
    final_val_loaders = full_eval_loaders_from(val_loaders, cfg)
    final_test_loaders = full_eval_loaders_from(test_loaders, cfg)
    print(f"  val_loaders: {list(final_val_loaders.keys())} (val sizes: "
          + ", ".join(f"{k}={len(v.dataset)}" for k, v in final_val_loaders.items()) + ")")
    print(f"  test_loaders: {list(final_test_loaders.keys())} (test sizes: "
          + ", ".join(f"{k}={len(v.dataset)}" for k, v in final_test_loaders.items()) + ")")

    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    model = build_model(cfg).to(device)
    state_dict = ckpt["model"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    for k in missing:
        print(f"    missing: {k}")
    for k in unexpected:
        print(f"    unexpected: {k}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: SurfaceTransolver ({n_params / 1e6:.2f}M params)")

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project="senpai-v1-drivaerml-ddp8",
            entity="wandb-applied-ai-team",
            name=args.wandb_name,
            group=args.wandb_group,
            tags=["test-eval-only", "pr-935", "ixrg3mg1"],
            config={
                **cfg_dict,
                "_eval_only": True,
                "_source_checkpoint": str(ckpt_path),
                "_source_run_id": args.source_run_id,
                "_source_epoch": int(ckpt.get("epoch", -1)),
                "_source_checkpoint_source": ckpt.get("checkpoint_source"),
            },
            reinit=True,
        )

    with torch.no_grad():
        print("\nRunning full_val evaluation (sanity check vs recorded best val)…")
        full_val_metrics = {
            name: evaluate_split(model, loader, transform, device, amp_mode=cfg.amp_mode)
            for name, loader in final_val_loaders.items()
        }
        for split_name, m in full_val_metrics.items():
            _print_split(f"full_val/{split_name}", m)

        print("\nRunning test evaluation…")
        test_metrics = {
            name: evaluate_split(model, loader, transform, device, amp_mode=cfg.amp_mode)
            for name, loader in final_test_loaders.items()
        }
        for split_name, m in test_metrics.items():
            _print_split(f"test/{split_name}", m)

    if run is not None:
        log_payload: dict[str, float] = {}

        def _emit(prefix: str, mdict: dict[str, dict[str, float]]) -> None:
            for split_name, metrics in mdict.items():
                for k, v in metrics.items():
                    log_payload[f"{prefix}/{split_name}/{k}"] = float(v)

        _emit("full_val", full_val_metrics)
        _emit("test", test_metrics)

        primary_keys = [
            "abupt_axis_mean_rel_l2_pct",
            "surface_pressure_rel_l2_pct",
            "volume_pressure_rel_l2_pct",
            "wall_shear_rel_l2_pct",
            "wall_shear_x_rel_l2_pct",
            "wall_shear_y_rel_l2_pct",
            "wall_shear_z_rel_l2_pct",
        ]
        for primary_prefix, mdict_split_key in [
            ("full_val_primary", ("val_surface", full_val_metrics)),
            ("test_primary", ("test_surface", test_metrics)),
        ]:
            split_key, mdict = mdict_split_key
            if split_key in mdict:
                source = mdict[split_key]
                for k in primary_keys:
                    if k in source:
                        log_payload[f"{primary_prefix}/{k}"] = float(source[k])

        wandb.log(log_payload)
        wandb.summary.update(log_payload)
        wandb.summary.update({
            "best_epoch": int(ckpt.get("epoch", -1)),
            "best_checkpoint/source": ckpt.get("checkpoint_source"),
        })
        wandb.finish()

    val_surf = full_val_metrics["val_surface"]
    tsurf = test_metrics["test_surface"]

    print("\nSENPAI-RESULT")
    print(f"source_run_id={args.source_run_id}")
    print(f"source_epoch={int(ckpt.get('epoch', -1))}")
    print(f"source_checkpoint_source={ckpt.get('checkpoint_source')}")
    print(f"full_val_abupt={val_surf['abupt_axis_mean_rel_l2_pct']:.4f}")
    print(f"full_val_surface_pressure={val_surf['surface_pressure_rel_l2_pct']:.4f}")
    print(f"full_val_volume_pressure={val_surf['volume_pressure_rel_l2_pct']:.4f}")
    print(f"full_val_wall_shear={val_surf['wall_shear_rel_l2_pct']:.4f}")
    print(f"test_abupt={tsurf['abupt_axis_mean_rel_l2_pct']:.4f}")
    print(f"test_surface_pressure={tsurf['surface_pressure_rel_l2_pct']:.4f}")
    print(f"test_volume_pressure={tsurf['volume_pressure_rel_l2_pct']:.4f}")
    print(f"test_wall_shear={tsurf['wall_shear_rel_l2_pct']:.4f}")
    print(f"test_wall_shear_x={tsurf.get('wall_shear_x_rel_l2_pct', float('nan')):.4f}")
    print(f"test_wall_shear_y={tsurf.get('wall_shear_y_rel_l2_pct', float('nan')):.4f}")
    print(f"test_wall_shear_z={tsurf.get('wall_shear_z_rel_l2_pct', float('nan')):.4f}")


if __name__ == "__main__":
    os.environ.setdefault("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8")
    os.environ.setdefault("WANDB_ENTITY", "wandb-applied-ai-team")
    main()
