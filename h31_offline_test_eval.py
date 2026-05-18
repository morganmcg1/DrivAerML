"""Offline test-eval for the H31 WALLDIST log-SDF run.

Loads the local best checkpoint saved by rank-0 of the DDP run, runs full val +
test evaluation on a single GPU, and prints metrics in the same shape as
`run_final_evaluation`. Threads `use_log_sdf_feature` through to evaluate_split
so the 5th channel is appended to volume_x at the eval forward sites too.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import torch
import yaml

from data import load_data
from model import SurfaceTransolver, VOLUME_X_DIM
from train import Config, parse_rff_init_sigmas
from trainer_runtime import (
    TargetTransform,
    evaluate_split,
    eval_loader_for_dataset,
)


def _config_from_yaml(path: Path) -> Config:
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    valid = {f.name for f in fields(Config)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    return Config(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="outputs/drivaerml/run-<id> dir containing checkpoint.pt + config.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = _config_from_yaml(args.run_dir / "config.yaml")
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    train_ds, val_splits, test_splits, stats = load_data(
        manifest_path=config.manifest,
        root=config.data_root or None,
        train_surface_points=config.train_surface_points,
        eval_surface_points=config.eval_surface_points,
        train_volume_points=config.train_volume_points,
        eval_volume_points=config.eval_volume_points,
        debug=config.debug,
    )
    val_loaders = {
        name: eval_loader_for_dataset(ds, config, distributed_state=None)
        for name, ds in val_splits.items()
    }
    test_loaders = {
        name: eval_loader_for_dataset(ds, config, distributed_state=None)
        for name, ds in test_splits.items()
    }
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    volume_input_dim = VOLUME_X_DIM + (1 if config.use_log_sdf_feature else 0)
    model = SurfaceTransolver(
        n_layers=config.model_layers,
        n_hidden=config.model_hidden_dim,
        dropout=config.model_dropout,
        n_head=config.model_heads,
        mlp_ratio=config.model_mlp_ratio,
        slice_num=config.model_slices,
        rff_num_features=config.rff_num_features,
        rff_sigma=config.rff_sigma,
        rff_init_sigmas=parse_rff_init_sigmas(config.rff_init_sigmas),
        pos_encoding_mode=config.pos_encoding_mode,
        use_qk_norm=config.use_qk_norm,
        use_surf_to_vol_xattn=config.use_surf_to_vol_xattn,
        volume_input_dim=volume_input_dim,
    ).to(device)

    ckpt_path = args.run_dir / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch')}, "
        f"source={ckpt.get('checkpoint_source')}, "
        f"selection_metric={ckpt.get('selection_metric')}, "
        f"use_log_sdf_feature={config.use_log_sdf_feature}, "
        f"volume_input_dim={volume_input_dim}"
    )

    t0 = time.time()
    val_metrics = {
        name: evaluate_split(
            model,
            loader,
            transform,
            device,
            amp_mode=config.amp_mode,
            use_log_sdf_feature=config.use_log_sdf_feature,
        )
        for name, loader in val_loaders.items()
    }
    print(f"[val] elapsed={time.time() - t0:.1f}s")
    for name, m in val_metrics.items():
        print(f"\n=== full-{name} ===")
        for k, v in sorted(m.items()):
            print(f"  {k}: {v}")

    t1 = time.time()
    test_metrics = {
        name: evaluate_split(
            model,
            loader,
            transform,
            device,
            amp_mode=config.amp_mode,
            use_log_sdf_feature=config.use_log_sdf_feature,
        )
        for name, loader in test_loaders.items()
    }
    print(f"\n[test] elapsed={time.time() - t1:.1f}s")
    for name, m in test_metrics.items():
        print(f"\n=== {name} ===")
        for k, v in sorted(m.items()):
            print(f"  {k}: {v}")

    val_s = val_metrics["val_surface"]
    test_s = test_metrics["test_surface"]
    print("\n=== H31-OFFLINE-SUMMARY ===")
    print(
        "VAL:",
        {
            "abupt": val_s["abupt_axis_mean_rel_l2_pct"],
            "SP": val_s["surface_pressure_rel_l2_pct"],
            "vol_p": val_s["volume_pressure_rel_l2_pct"],
            "WSS": val_s["wall_shear_rel_l2_pct"],
            "WSS_x": val_s["wall_shear_x_rel_l2_pct"],
            "WSS_y": val_s["wall_shear_y_rel_l2_pct"],
            "WSS_z": val_s["wall_shear_z_rel_l2_pct"],
            "tauz_over_taux": val_s["wall_shear_z_rel_l2_pct"] / val_s["wall_shear_x_rel_l2_pct"],
        },
    )
    print(
        "TEST:",
        {
            "abupt": test_s["abupt_axis_mean_rel_l2_pct"],
            "SP": test_s["surface_pressure_rel_l2_pct"],
            "vol_p": test_s["volume_pressure_rel_l2_pct"],
            "WSS": test_s["wall_shear_rel_l2_pct"],
            "WSS_x": test_s["wall_shear_x_rel_l2_pct"],
            "WSS_y": test_s["wall_shear_y_rel_l2_pct"],
            "WSS_z": test_s["wall_shear_z_rel_l2_pct"],
            "tauz_over_taux": test_s["wall_shear_z_rel_l2_pct"] / test_s["wall_shear_x_rel_l2_pct"],
        },
    )


if __name__ == "__main__":
    main()
