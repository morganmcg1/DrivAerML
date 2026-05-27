"""Fallback test-eval for runs where the terminal kill threshold skipped
``run_final_evaluation``. Loads the saved best checkpoint and rebuilds the
model, then logs ``full_val/*``, ``full_val_primary/*``, ``test/*`` and
``test_primary/*`` against the resumed W&B run so the val→test slope can
be computed.

Usage (single GPU on rank-0):
    python run_test_eval_fallback.py \
        --run-dir outputs/drivaerml/run-<wandb-id> \
        --run-id <wandb-id>
"""

from __future__ import annotations

import argparse
import os
from dataclasses import fields
from pathlib import Path

import torch
import wandb
import yaml

from train import Config, build_model
from trainer_runtime import (
    TargetTransform,
    assert_required_finite_metrics,
    evaluate_split,
    full_eval_loaders_from,
    make_loaders,
    metric_namespace,
    numeric_metric_items,
    primary_metric_log,
    print_metrics,
)


def config_from_dict(d: dict) -> Config:
    field_names = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in d.items() if k in field_names}
    return Config(**filtered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"),
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    config_path = run_dir / "config.yaml"
    model_path = run_dir / "checkpoint.pt"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    with config_path.open("r") as fh:
        cfg_dict = yaml.safe_load(fh)
    config = config_from_dict(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Single-process loaders (no DistributedSampler / DDP). full_eval_loaders_from
    # rewraps to use the deterministic strided eval views.
    _train_loader, val_loaders, test_loaders, stats = make_loaders(
        config, distributed_state=None
    )
    final_val_loaders = full_eval_loaders_from(val_loaders, config)
    final_test_loaders = full_eval_loaders_from(test_loaders, config)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    best_epoch = int(checkpoint.get("epoch", -1))
    best_val_metrics = checkpoint.get("val_metrics", {}).get("val_surface", {})
    best_abupt = float(best_val_metrics.get("abupt_axis_mean_rel_l2_pct", float("nan")))
    print(
        f"Loaded checkpoint: epoch {best_epoch}, "
        f"source={checkpoint.get('checkpoint_source')}, "
        f"best val abupt={best_abupt:.4f}"
    )

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="must",
    )

    global_step = int(run.summary.get("_step") or 0)

    full_val_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
        for name, loader in final_val_loaders.items()
    }
    full_val_log: dict[str, object] = {
        "global_step": global_step,
        **primary_metric_log("full_val_primary", full_val_metrics["val_surface"]),
    }
    for split_name, metrics in full_val_metrics.items():
        full_val_log.update(metric_namespace("full_val", split_name, metrics))
    try:
        assert_required_finite_metrics(full_val_log, "full_val_primary")
    except RuntimeError as exc:
        wandb.summary.update({"run_invalid": 1.0, "run_invalid/reason": str(exc)})
        wandb.finish()
        raise
    wandb.log(full_val_log)
    wandb.summary.update(numeric_metric_items(full_val_log))
    print_metrics("full_val", full_val_metrics["val_surface"])

    test_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
        for name, loader in final_test_loaders.items()
    }
    test_log: dict[str, object] = {
        "global_step": global_step,
        **primary_metric_log("test_primary", test_metrics["test_surface"]),
    }
    for split_name, metrics in test_metrics.items():
        test_log.update(metric_namespace("test", split_name, metrics))
    try:
        assert_required_finite_metrics(test_log, "test_primary")
    except RuntimeError as exc:
        wandb.summary.update({"run_invalid": 1.0, "run_invalid/reason": str(exc)})
        wandb.finish()
        raise
    wandb.log(test_log)
    wandb.summary.update(numeric_metric_items(test_log))
    print_metrics("test_surface", test_metrics["test_surface"])

    wandb.summary.update(
        {
            "test_eval_fallback/triggered": 1.0,
            "test_eval_fallback/best_epoch": best_epoch,
            "test_eval_fallback/best_val_abupt": best_abupt,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
