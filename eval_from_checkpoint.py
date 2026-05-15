"""Truncation-safety eval: load local best-EMA checkpoint, run full val + test, log to existing W&B run.

Used when training is truncated mid-budget (e.g. EP10 hard truncate) and the in-process
run_final_evaluation has not executed. Single-GPU evaluation only (the full-eval loaders
already run on rank 0 in the normal path).

Usage:
    python eval_from_checkpoint.py \
        --run-dir outputs/drivaerml/run-vvv84p32 \
        --run-id vvv84p32 \
        --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields
from pathlib import Path

import torch
import wandb
import yaml

from data import load_data
from torch.utils.data import DataLoader

from train import Config, build_model, parse_rff_init_sigmas  # noqa: F401
from trainer_runtime import (
    TargetTransform,
    evaluate_split,
    eval_loader_for_dataset,
    log_model_artifact,
    metric_namespace,
    numeric_metric_items,
    primary_metric_log,
    print_metrics,
    assert_required_finite_metrics,
)


def _config_from_yaml(config_path: Path) -> Config:
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    valid = {f.name for f in fields(Config)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    return Config(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="outputs/drivaerml/run-<id>")
    parser.add_argument("--run-id", required=True, help="W&B run id to resume")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    )
    parser.add_argument(
        "--project", default=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8")
    )
    parser.add_argument(
        "--no-artifact",
        action="store_true",
        help="Skip artifact upload (faster; logs metrics only).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    model_path = run_dir / "checkpoint.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {model_path}")

    config = _config_from_yaml(config_path)
    if args.data_root:
        config.data_root = args.data_root
    config.num_workers = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading data (eval_surface_points={config.eval_surface_points}, "
          f"eval_volume_points={config.eval_volume_points}) ...")
    _, val_splits, test_splits, stats = load_data(
        manifest_path=config.manifest,
        root=config.data_root or None,
        train_surface_points=config.train_surface_points,
        eval_surface_points=config.eval_surface_points,
        train_volume_points=config.train_volume_points,
        eval_volume_points=config.eval_volume_points,
        debug=False,
    )
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    val_loaders = {
        name: eval_loader_for_dataset(ds, config, distributed_state=None)
        for name, ds in val_splits.items()
    }
    test_loaders = {
        name: eval_loader_for_dataset(ds, config, distributed_state=None)
        for name, ds in test_splits.items()
    }
    print(f"Val splits: {list(val_loaders)}  Test splits: {list(test_loaders)}")

    print(f"Loading checkpoint from {model_path} ...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError("checkpoint missing 'model' state dict")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    best_epoch = int(checkpoint.get("epoch", -1))
    src = checkpoint.get("checkpoint_source", "?")
    print(f"Loaded checkpoint: epoch={best_epoch}, source={src}")
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Resuming W&B run {args.run_id} ...")
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="must",
    )

    # Full val
    print("\n=== Full val ===")
    t0 = time.time()
    val_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
        for name, loader in val_loaders.items()
    }
    full_val_log: dict[str, object] = {
        **primary_metric_log("full_val_primary", val_metrics["val_surface"]),
    }
    for split_name, metrics in val_metrics.items():
        full_val_log.update(metric_namespace("full_val", split_name, metrics))
    assert_required_finite_metrics(full_val_log, "full_val_primary")
    wandb.log(full_val_log)
    wandb.summary.update(numeric_metric_items(full_val_log))
    print_metrics("full_val", val_metrics["val_surface"])
    print(f"full val seconds: {time.time() - t0:.1f}")

    # Test
    print("\n=== Full test ===")
    t0 = time.time()
    test_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
        for name, loader in test_loaders.items()
    }
    test_log: dict[str, object] = {
        **primary_metric_log("test_primary", test_metrics["test_surface"]),
    }
    for split_name, metrics in test_metrics.items():
        test_log.update(metric_namespace("test", split_name, metrics))
    assert_required_finite_metrics(test_log, "test_primary")
    wandb.log(test_log)
    wandb.summary.update(numeric_metric_items(test_log))
    print_metrics("test_surface", test_metrics["test_surface"])
    print(f"test seconds: {time.time() - t0:.1f}")

    # Summary annotations
    wandb.summary.update(
        {
            "truncated_post_hoc_eval": 1.0,
            "truncated_best_epoch": best_epoch,
            "truncated_best_checkpoint_source": src,
        }
    )

    # Artifact upload
    if not args.no_artifact:
        best_metrics_for_artifact = {
            "epoch": best_epoch,
            "abupt_axis_mean_rel_l2_pct": val_metrics["val_surface"][
                "abupt_axis_mean_rel_l2_pct"
            ],
            "surface_pressure_mae": val_metrics["val_surface"]["surface_pressure_mae"],
            "wall_shear_mae": val_metrics["val_surface"]["wall_shear_mae"],
            "volume_pressure_mae": val_metrics["val_surface"]["volume_pressure_mae"],
        }
        print("\nUploading model artifact ...")
        log_model_artifact(
            run=run,
            model_path=model_path,
            config_path=config_path,
            config=config,
            best_metrics=best_metrics_for_artifact,
            test_metrics=test_metrics,
            n_params=n_params,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
