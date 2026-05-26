"""H144 dual-checkpoint terminal evaluation.

Mimics the in-train dual-terminal-eval block in train.py:1421-1541. Loads both
the live-best and EMA-best checkpoints for an interrupted H144 run, evaluates
both on full validation and held-out test, picks the winner by test_WSS, and
logs results to the original W&B run via WANDB_RUN_ID/WANDB_RESUME env vars.

Usage:
    WANDB_RUN_ID=<run_id> WANDB_RESUME=must \\
    python eval_h144_dual.py --data-root <...> --output-dir outputs/h144_ema_weights \\
        [all the same hyperparameter flags as the original training run]
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import torch
import wandb
import yaml

from train import Config, build_model, parse_args
from trainer_runtime import (
    TargetTransform,
    cleanup_distributed,
    define_wandb_metrics,
    evaluate_split,
    full_eval_loaders_from,
    init_distributed,
    make_loaders,
    metric_namespace,
    numeric_metric_items,
    primary_metric_log,
)


def main() -> None:
    state = init_distributed()
    try:
        config = parse_args()
        device = state.device
        print(f"[eval_h144_dual] device={device}, output_dir={config.output_dir}")

        # Build loaders + transform exactly like train.py.
        train_loader, val_loaders, test_loaders, stats = make_loaders(
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

        base_model = build_model(config).to(device)
        n_params = sum(p.numel() for p in base_model.parameters())
        print(f"[eval_h144_dual] model params: {n_params/1e6:.2f}M")

        # Resume the original wandb run. Caller sets WANDB_RUN_ID + WANDB_RESUME.
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT"),
            resume=os.environ.get("WANDB_RESUME", "must"),
            id=os.environ.get("WANDB_RUN_ID"),
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        define_wandb_metrics()
        print(f"[eval_h144_dual] resumed wandb run id={run.id}")

        # Locate both checkpoints. Naming convention from train.py.
        output_dir = Path(config.output_dir) / f"run-{run.id}"
        path_live = output_dir / "checkpoint_live.pt"
        path_ema = output_dir / "checkpoint_ema.pt"
        if not path_live.exists() or not path_ema.exists():
            raise FileNotFoundError(
                f"missing checkpoints under {output_dir}: "
                f"live={path_live.exists()} ema={path_ema.exists()}"
            )

        config_path = output_dir / "config.yaml"
        if not config_path.exists():
            with config_path.open("w") as f:
                yaml.safe_dump(asdict(config), f)

        def _eval_ckpt(path: Path):
            ckpt = torch.load(path, map_location=device, weights_only=True)
            base_model.load_state_dict(ckpt["model"])
            full_val = {
                name: evaluate_split(
                    base_model, loader, transform, device, amp_mode=config.amp_mode
                )
                for name, loader in final_val_loaders.items()
            }
            test = {
                name: evaluate_split(
                    base_model, loader, transform, device, amp_mode=config.amp_mode
                )
                for name, loader in final_test_loaders.items()
            }
            return ckpt, full_val, test

        print("[eval_h144_dual] evaluating live-best checkpoint...")
        live_ckpt, live_full_val, live_test = _eval_ckpt(path_live)
        print(
            f"  live epoch={int(live_ckpt['epoch'])} "
            f"full_val_WSS={live_full_val['val_surface']['wall_shear_rel_l2_pct']:.4f} "
            f"test_WSS={live_test['test_surface']['wall_shear_rel_l2_pct']:.4f}"
        )

        print("[eval_h144_dual] evaluating ema-best checkpoint...")
        ema_ckpt, ema_full_val, ema_test = _eval_ckpt(path_ema)
        print(
            f"  ema  epoch={int(ema_ckpt['epoch'])} "
            f"full_val_WSS={ema_full_val['val_surface']['wall_shear_rel_l2_pct']:.4f} "
            f"test_WSS={ema_test['test_surface']['wall_shear_rel_l2_pct']:.4f}"
        )

        wss_live = live_test["test_surface"]["wall_shear_rel_l2_pct"]
        wss_ema = ema_test["test_surface"]["wall_shear_rel_l2_pct"]
        winner = "live" if wss_live <= wss_ema else "ema"
        print(f"\n[eval_h144_dual] WINNER by test_WSS: {winner}")
        print(f"  live test_WSS={wss_live:.4f}, ema test_WSS={wss_ema:.4f}, "
              f"delta={(wss_ema - wss_live):+.4f}pp")

        global_step = int(
            (live_ckpt if winner == "live" else ema_ckpt).get("epoch", 0)
        )

        # Log both arms' full_val + test as ema_dual_*/* keys.
        dual_log: dict[str, float] = {"global_step": global_step}
        for k, v in primary_metric_log(
            "ema_dual_live_full_val_primary", live_full_val["val_surface"]
        ).items():
            dual_log[k] = v
        for k, v in primary_metric_log(
            "ema_dual_live_test_primary", live_test["test_surface"]
        ).items():
            dual_log[k] = v
        for k, v in primary_metric_log(
            "ema_dual_ema_full_val_primary", ema_full_val["val_surface"]
        ).items():
            dual_log[k] = v
        for k, v in primary_metric_log(
            "ema_dual_ema_test_primary", ema_test["test_surface"]
        ).items():
            dual_log[k] = v
        wandb.log(dual_log)
        wandb.summary.update(
            {
                "ema_dual/winner": winner,
                "ema_dual/live_test_wss": wss_live,
                "ema_dual/ema_test_wss": wss_ema,
                "ema_dual/live_best_epoch": int(live_ckpt["epoch"]),
                "ema_dual/ema_best_epoch": int(ema_ckpt["epoch"]),
            }
        )

        # Promote the winner to the canonical test_primary/*, full_val_primary/*
        # and best_* keys so the standard senpai status check reads correctly.
        if winner == "live":
            win_ckpt, win_full_val, win_test = live_ckpt, live_full_val, live_test
            win_path = path_live
        else:
            win_ckpt, win_full_val, win_test = ema_ckpt, ema_full_val, ema_test
            win_path = path_ema
        win_val_metrics = win_ckpt["val_metrics"]["val_surface"]
        best_metrics = {
            "epoch": float(win_ckpt["epoch"]),
            "abupt_axis_mean_rel_l2_pct": win_val_metrics["abupt_axis_mean_rel_l2_pct"],
            "surface_pressure_mae": win_val_metrics["surface_pressure_mae"],
            "wall_shear_mae": win_val_metrics["wall_shear_mae"],
            "volume_pressure_mae": win_val_metrics["volume_pressure_mae"],
        }
        wandb.summary.update(
            {
                "best_epoch": int(best_metrics["epoch"]),
                "best_checkpoint/source": winner,
                "best_checkpoint/selection_metric": (
                    "val_primary/abupt_axis_mean_rel_l2_pct"
                    if winner == "live"
                    else "val_ema/abupt_axis_mean_rel_l2_pct"
                ),
                "best_val_primary/abupt_axis_mean_rel_l2_pct": best_metrics[
                    "abupt_axis_mean_rel_l2_pct"
                ],
                "best_val/surface_pressure_mae": best_metrics["surface_pressure_mae"],
                "best_val/wall_shear_mae": best_metrics["wall_shear_mae"],
                "best_val/volume_pressure_mae": best_metrics["volume_pressure_mae"],
                "h144_dual/winner_checkpoint_path": str(win_path),
                "h144_dual/killed_at_epoch": int(win_ckpt["epoch"]),
            }
        )

        full_val_log: dict[str, object] = {
            "global_step": global_step,
            **primary_metric_log("full_val_primary", win_full_val["val_surface"]),
        }
        for split_name, metrics in win_full_val.items():
            full_val_log.update(metric_namespace("full_val", split_name, metrics))
        wandb.log(full_val_log)
        wandb.summary.update(numeric_metric_items(full_val_log))

        test_log: dict[str, object] = {
            "global_step": global_step,
            **primary_metric_log("test_primary", win_test["test_surface"]),
        }
        for split_name, metrics in win_test.items():
            test_log.update(metric_namespace("test", split_name, metrics))
        wandb.log(test_log)
        wandb.summary.update(numeric_metric_items(test_log))

        print("\n[eval_h144_dual] final test_primary metrics (winner):")
        for k in [
            "wall_shear_rel_l2_pct",
            "volume_pressure_rel_l2_pct",
            "surface_pressure_rel_l2_pct",
            "abupt_axis_mean_rel_l2_pct",
            "wall_shear_x_rel_l2_pct",
            "wall_shear_y_rel_l2_pct",
            "wall_shear_z_rel_l2_pct",
        ]:
            print(f"  test_primary/{k} = {win_test['test_surface'][k]:.4f}")

        wandb.finish()
    finally:
        cleanup_distributed(state)


if __name__ == "__main__":
    main()
