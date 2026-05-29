"""Evaluate a single block-spliced checkpoint on val + test (single-GPU).

Usage:
    CUDA_VISIBLE_DEVICES=0 python runs/h213/splice_eval.py \
        --checkpoint runs/h213/splices/splice_k3.pt \
        --wandb-name h213-splice-k3 \
        --wandb-group h213-tanjiro-splice-h112-h183

Loads the splice state-dict, rebuilds the model from H112's config (both
parents use the same architecture), and reports full val + test metrics
plus the val→test slope for each metric. Logs to W&B for easy aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import wandb

# Add target root to sys.path so we can import top-level modules when invoked
# from runs/h213/ directly.
TARGET_ROOT = Path(__file__).resolve().parents[2]
if str(TARGET_ROOT) not in sys.path:
    sys.path.insert(0, str(TARGET_ROOT))

from data import load_data, pad_collate  # noqa: E402
from ensemble_eval import build_model_from_config  # noqa: E402
from trainer_runtime import TargetTransform, evaluate_split  # noqa: E402


CORRECTED_DATA_ROOT = (
    "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
)
ALT_DATA_ROOT = (
    "/mnt/pvc/Processed/drivaerml_processed_rawcanon_20260511"
)


def resolve_data_root(explicit: str) -> str:
    if explicit:
        return explicit
    if Path(CORRECTED_DATA_ROOT).exists():
        return CORRECTED_DATA_ROOT
    if Path(ALT_DATA_ROOT).exists():
        return ALT_DATA_ROOT
    raise FileNotFoundError(
        f"Neither {CORRECTED_DATA_ROOT} nor {ALT_DATA_ROOT} exists; "
        "pass --data-root explicitly."
    )


def make_eval_loader(dataset, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        collate_fn=pad_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


METRIC_KEYS = [
    "abupt_axis_mean_rel_l2_pct",
    "surface_pressure_rel_l2_pct",
    "wall_shear_rel_l2_pct",
    "wall_shear_x_rel_l2_pct",
    "wall_shear_y_rel_l2_pct",
    "wall_shear_z_rel_l2_pct",
    "volume_pressure_rel_l2_pct",
    "surface_pressure_mae",
    "wall_shear_mae",
    "wall_shear_vector_mae",
    "volume_pressure_mae",
]


def slope_log(prefix: str, val: dict[str, float], test: dict[str, float]) -> dict[str, float]:
    """val→test slope = test - val for each rel_L2_pct metric."""

    out: dict[str, float] = {}
    for k in METRIC_KEYS:
        if k in val and k in test:
            out[f"{prefix}/{k}"] = test[k] - val[k]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-group", default="h213-tanjiro-splice-h112-h183")
    parser.add_argument("--wandb-tags", nargs="*", default=["splice", "h213", "tanjiro"])
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--eval-surface-points", type=int, default=65536)
    parser.add_argument("--eval-volume-points", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-mode", default="bf16", choices=["bf16", "none"])
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state_dict = ckpt["model"]
    splice_k = ckpt.get("splice_k")
    print(
        f"Loaded splice ckpt {checkpoint_path.name}: "
        f"splice_k={splice_k}, epoch={ckpt.get('epoch')}, "
        f"source={ckpt.get('checkpoint_source')}, n_keys={len(state_dict)}"
    )

    data_root = resolve_data_root(args.data_root)
    print(f"Loading data from {data_root}")
    _, val_splits, test_splits, stats = load_data(
        manifest_path=args.manifest,
        root=data_root,
        train_surface_points=args.eval_surface_points,
        eval_surface_points=args.eval_surface_points,
        train_volume_points=args.eval_volume_points,
        eval_volume_points=args.eval_volume_points,
        debug=False,
    )
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    # Build model from H112 config (both parents identical) — detect aux heads.
    use_aux_decoder_heads = "surface_out.0.weight" in state_dict and (
        "surface_out.2.weight" in state_dict or "surface_out.0.bias" in state_dict
    )
    model = build_model_from_config(
        config, use_aux_decoder_heads=use_aux_decoder_heads
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built ({n_params/1e6:.2f}M params)")

    val_loader = make_eval_loader(
        val_splits["val_surface"], args.batch_size, args.num_workers
    )
    test_loader = make_eval_loader(
        test_splits["test_surface"], args.batch_size, args.num_workers
    )

    run = None
    if not args.no_wandb:
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            project=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"),
            name=args.wandb_name,
            group=args.wandb_group,
            tags=list(args.wandb_tags),
            config={
                "experiment": "h213-splice-eval",
                "splice_k": splice_k,
                "source_checkpoint": str(checkpoint_path),
                "h112_run_id": "u9ue2ryb",
                "h183_run_id": "5k58uzqc",
                "eval_surface_points": args.eval_surface_points,
                "eval_volume_points": args.eval_volume_points,
                "batch_size": args.batch_size,
                "amp_mode": args.amp_mode,
                "n_params": n_params,
            },
        )

    print("\n=== Running val eval ===")
    t0 = time.time()
    val_metrics = evaluate_split(
        model, val_loader, transform, device, amp_mode=args.amp_mode
    )
    val_secs = time.time() - t0
    print(
        f"val_abupt={val_metrics['abupt_axis_mean_rel_l2_pct']:.4f} "
        f"val_SP={val_metrics['surface_pressure_rel_l2_pct']:.4f} "
        f"val_WSS={val_metrics['wall_shear_rel_l2_pct']:.4f} "
        f"val_WSS_x={val_metrics['wall_shear_x_rel_l2_pct']:.4f} "
        f"val_VP={val_metrics['volume_pressure_rel_l2_pct']:.4f} "
        f"({val_secs:.1f}s)"
    )

    print("\n=== Running test eval ===")
    t0 = time.time()
    test_metrics = evaluate_split(
        model, test_loader, transform, device, amp_mode=args.amp_mode
    )
    test_secs = time.time() - t0
    print(
        f"test_abupt={test_metrics['abupt_axis_mean_rel_l2_pct']:.4f} "
        f"test_SP={test_metrics['surface_pressure_rel_l2_pct']:.4f} "
        f"test_WSS={test_metrics['wall_shear_rel_l2_pct']:.4f} "
        f"test_WSS_x={test_metrics['wall_shear_x_rel_l2_pct']:.4f} "
        f"test_VP={test_metrics['volume_pressure_rel_l2_pct']:.4f} "
        f"({test_secs:.1f}s)"
    )

    slopes = slope_log("slope", val_metrics, test_metrics)
    print(
        f"slope WSS_x={slopes['slope/wall_shear_x_rel_l2_pct']:+.4f}pp "
        f"abupt={slopes['slope/abupt_axis_mean_rel_l2_pct']:+.4f}pp"
    )

    if run is not None:
        full_val_log: dict[str, float] = {
            f"full_val_primary/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))
        }
        full_val_log.update(
            {f"full_val/val_surface/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}
        )
        test_log: dict[str, float] = {
            f"test_primary/{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))
        }
        test_log.update(
            {f"test/test_surface/{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
        )
        wandb.log({**full_val_log, **test_log, **slopes})
        wandb.summary.update(
            {
                **{f"summary/full_val_primary/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
                **{f"summary/test_primary/{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))},
                **{f"summary/{k}": v for k, v in slopes.items()},
                "summary/splice_k": splice_k,
                "summary/val_eval_seconds": val_secs,
                "summary/test_eval_seconds": test_secs,
            }
        )
        wandb.finish()

    # Also persist locally for cheap post-hoc inspection.
    out_dir = checkpoint_path.parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.wandb_name.replace('/', '_')}.json"
    with out_path.open("w") as fh:
        json.dump(
            {
                "wandb_name": args.wandb_name,
                "splice_k": splice_k,
                "val": val_metrics,
                "test": test_metrics,
                "slopes": slopes,
                "val_seconds": val_secs,
                "test_seconds": test_secs,
            },
            fh,
            indent=2,
            default=float,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
