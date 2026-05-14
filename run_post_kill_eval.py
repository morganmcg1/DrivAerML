"""Run final val + test evaluation on the EP3 best checkpoint after the run was killed."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import wandb
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from train import Config, build_model
from trainer_runtime import (
    TargetTransform,
    evaluate_split,
    full_eval_loaders_from,
    make_loaders,
    primary_metric_log,
    metric_namespace,
)


def main() -> None:
    ckpt_dir = Path("/mnt/new-pvc/checkpoints/nezuko-gradnorm-ema-proxy-slw1-v3/run-eihfy5od")
    with (ckpt_dir / "config.yaml").open() as fh:
        cfg_dict = yaml.safe_load(fh)
    # Drop fields that the Config dataclass doesn't know about.
    valid = {f for f in Config.__dataclass_fields__}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
    config = Config(**cfg_dict)

    device = torch.device("cuda:0")

    # Re-load data (using same config the run was launched with).
    train_loader, val_loaders, test_loaders, stats = make_loaders(config, distributed_state=None)
    final_val_loaders = full_eval_loaders_from(val_loaders, config)
    final_test_loaders = full_eval_loaders_from(test_loaders, config)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_dir / "checkpoint.pt", map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"
    print(f"Loaded checkpoint epoch={ckpt['epoch']} source={ckpt['checkpoint_source']}")

    wandb.init(
        entity="wandb-applied-ai-team",
        project="senpai-v1-drivaerml-ddp8",
        group="nezuko-gradnorm-ema-proxy-slw1",
        name="nezuko/gradnorm-ema-proxy-slw1-v3-post-kill-eval",
        config={**cfg_dict, "post_kill_eval_for_run": "eihfy5od", "post_kill_eval_epoch": ckpt["epoch"]},
    )

    val_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
        for name, loader in final_val_loaders.items()
    }
    val_log: dict = {**primary_metric_log("full_val_primary", val_metrics["val_surface"])}
    for split_name, metrics in val_metrics.items():
        val_log.update(metric_namespace("full_val", split_name, metrics))

    test_metrics = {
        name: evaluate_split(model, loader, transform, device, amp_mode=config.amp_mode)
        for name, loader in final_test_loaders.items()
    }
    test_log: dict = {**primary_metric_log("test_primary", test_metrics["test_surface"])}
    for split_name, metrics in test_metrics.items():
        test_log.update(metric_namespace("test", split_name, metrics))

    summary = {**val_log, **test_log}
    print("\n=== FULL VAL METRICS (val_surface) ===")
    for k, v in sorted(val_log.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.6f}")
    print("\n=== TEST METRICS (test_surface) ===")
    for k, v in sorted(test_log.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.6f}")

    wandb.log(summary)
    wandb.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float))})
    wandb.finish()


if __name__ == "__main__":
    main()
