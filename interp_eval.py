# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""H208 — Linear weight interpolation eval (H112 EP13 <-> H190 EP13).

For each alpha in --alphas, build
    W_alpha = (1 - alpha) * W_H112 + alpha * W_H190
load it into a freshly built SurfaceTransolver, and evaluate on the
DrivAerML val and test splits at full fidelity (same eval primitives as
``ensemble_eval.py``). Each alpha is logged as its own W&B run.

This is a pure inference script — no training. The two source
checkpoints must have identical state_dict keys; the script asserts
that and aborts on mismatch.

Example:

    python interp_eval.py \\
      --run-id-a u9ue2ryb --run-id-b 9f2jtrg2 \\
      --alphas 0.0 0.25 0.5 0.75 1.0 \\
      --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \\
      --wandb-group h208-fern-interp-h112-h190
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Iterable

import torch
import wandb

from data import load_data
from ensemble_eval import (
    build_model_from_config,
    download_checkpoint,
    evaluate_single_member,
    fetch_run_meta,
    make_eval_loader,
    primary_log_payload,
)
from trainer_runtime import TargetTransform


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear interpolation of two DrivAerML checkpoints, then eval."
    )
    parser.add_argument(
        "--run-id-a",
        required=True,
        help="W&B run id for endpoint A (alpha=0); state_dict is loaded "
        "with weight (1 - alpha).",
    )
    parser.add_argument(
        "--run-id-b",
        required=True,
        help="W&B run id for endpoint B (alpha=1); state_dict is loaded "
        "with weight alpha.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Interpolation coefficients to evaluate.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["val", "test"],
        default=["val", "test"],
        help="Which splits to evaluate each interpolated checkpoint on.",
    )
    parser.add_argument("--eval-surface-points", type=int, default=65536)
    parser.add_argument("--eval-volume-points", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-mode", default="bf16", choices=["bf16", "none"])
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument(
        "--cache-root",
        default="outputs/h208_interp_cache",
        help="Local cache root for the downloaded W&B model artifacts.",
    )
    parser.add_argument("--wandb-group", default="h208-fern-interp-h112-h190")
    parser.add_argument(
        "--wandb-name-prefix",
        default="h208-alpha",
        help="Each alpha logs its own run named '<prefix>-<alpha>'.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=["h208", "interp", "fern", "eval-only"],
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip W&B run creation (useful for debug smoke tests).",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="Optional cap on the number of batches per split (debug).",
    )
    return parser.parse_args(argv)


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load only the ``model`` state dict, on the requested device."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint {checkpoint_path} is missing 'model' state dict")
    return {k: v for k, v in checkpoint["model"].items()}


def interpolated_state_dict(
    sd_a: dict[str, torch.Tensor],
    sd_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Compute W = (1 - alpha) * W_A + alpha * W_B in fp32, return on the input device."""

    if set(sd_a) != set(sd_b):
        only_a = sorted(set(sd_a) - set(sd_b))[:5]
        only_b = sorted(set(sd_b) - set(sd_a))[:5]
        raise RuntimeError(
            f"State-dict keys mismatch: only-in-A={only_a}, only-in-B={only_b}"
        )
    out: dict[str, torch.Tensor] = {}
    for k, va in sd_a.items():
        vb = sd_b[k]
        if va.shape != vb.shape:
            raise RuntimeError(f"Shape mismatch on {k}: {tuple(va.shape)} vs {tuple(vb.shape)}")
        # Promote to fp32 for numerically-clean averaging, cast back to A's dtype.
        avg = (1.0 - alpha) * va.to(dtype=torch.float32) + alpha * vb.to(dtype=torch.float32)
        out[k] = avg.to(dtype=va.dtype)
    return out


def parameter_norm(state_dict: dict[str, torch.Tensor]) -> float:
    total_sq = 0.0
    for v in state_dict.values():
        total_sq += float(v.detach().to(dtype=torch.float64).pow(2).sum().item())
    return math.sqrt(total_sq)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    print(f"Resolving artifacts for run-id-a={args.run_id_a}, run-id-b={args.run_id_b}...")
    dir_a = download_checkpoint(api, entity, project, args.run_id_a, cache_root)
    dir_b = download_checkpoint(api, entity, project, args.run_id_b, cache_root)
    meta_a = fetch_run_meta(api, entity, project, args.run_id_a)
    meta_b = fetch_run_meta(api, entity, project, args.run_id_b)
    print(f"  A: {meta_a}")
    print(f"  B: {meta_b}")

    # Load both state dicts on CPU once; interpolation runs there too.
    import yaml

    with (dir_a / "config.yaml").open("r") as fh:
        config_a = yaml.safe_load(fh)
    with (dir_b / "config.yaml").open("r") as fh:
        config_b = yaml.safe_load(fh)

    print("Loading state dicts on CPU...")
    sd_a_cpu = load_state_dict(dir_a / "checkpoint.pt", device=torch.device("cpu"))
    sd_b_cpu = load_state_dict(dir_b / "checkpoint.pt", device=torch.device("cpu"))
    if set(sd_a_cpu) != set(sd_b_cpu):
        raise RuntimeError("State-dict key mismatch between A and B.")
    norm_a = parameter_norm(sd_a_cpu)
    norm_b = parameter_norm(sd_b_cpu)
    diff = {k: (sd_a_cpu[k].to(dtype=torch.float64) - sd_b_cpu[k].to(dtype=torch.float64))
            for k in sd_a_cpu}
    diff_norm = math.sqrt(sum(float(v.pow(2).sum().item()) for v in diff.values()))
    print(
        f"  |W_A| = {norm_a:.4f}, |W_B| = {norm_b:.4f}, |W_A - W_B| = {diff_norm:.4f} "
        f"({len(sd_a_cpu)} tensors)"
    )
    del diff

    print("Loading data once (val + test)...")
    _, val_splits, test_splits, stats = load_data(
        manifest_path=args.manifest,
        root=args.data_root or None,
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
    splits_to_run: dict[str, dict[str, torch.utils.data.Dataset]] = {}
    if "val" in args.split:
        splits_to_run["val"] = val_splits
    if "test" in args.split:
        splits_to_run["test"] = test_splits

    # PR #958 used a Sequential head; that's a checkpoint structure flag.
    use_aux_decoder_heads = "surface_out.0.weight" in sd_a_cpu

    summary_rows: list[dict[str, float]] = []

    for alpha in args.alphas:
        run_name = f"{args.wandb_name_prefix}-{alpha}"
        print(f"\n=== Alpha = {alpha} (run name: {run_name}) ===")
        sd_alpha_cpu = interpolated_state_dict(sd_a_cpu, sd_b_cpu, alpha)
        norm_alpha = parameter_norm(sd_alpha_cpu)
        # Build a fresh model from A's config (same architecture across A and B).
        model = build_model_from_config(
            config_a, use_aux_decoder_heads=use_aux_decoder_heads
        ).to(device)
        sd_alpha_on_device = {k: v.to(device=device) for k, v in sd_alpha_cpu.items()}
        missing, unexpected = model.load_state_dict(sd_alpha_on_device, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"State-dict mismatch at alpha={alpha}: missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        del sd_alpha_on_device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        run = None
        log_payload: dict[str, float] = {
            "interp/alpha": float(alpha),
            "interp/source_a_run_id_index": 0.0,
            "interp/source_b_run_id_index": 1.0,
            "interp/parameter_norm": norm_alpha,
        }
        summary_row: dict[str, float] = {"alpha": float(alpha)}
        if not args.no_wandb:
            wandb_config = {
                "alpha": alpha,
                "interp/parameter_norm": norm_alpha,
                "interp/parameter_norm_a": norm_a,
                "interp/parameter_norm_b": norm_b,
                "interp/parameter_diff_norm": diff_norm,
                "source_a": {
                    "run_id": args.run_id_a,
                    **{k: meta_a.get(k) for k in ("agent", "group", "wandb_name", "best_epoch")},
                    "config": config_a,
                },
                "source_b": {
                    "run_id": args.run_id_b,
                    **{k: meta_b.get(k) for k in ("agent", "group", "wandb_name", "best_epoch")},
                    "config": config_b,
                },
                "eval_surface_points": args.eval_surface_points,
                "eval_volume_points": args.eval_volume_points,
                "batch_size": args.batch_size,
                "amp_mode": args.amp_mode,
                "splits_evaluated": args.split,
            }
            run = wandb.init(
                entity=entity,
                project=project,
                group=args.wandb_group,
                name=run_name,
                tags=args.wandb_tags,
                config=wandb_config,
                reinit=True,
                mode=os.environ.get("WANDB_MODE", "online"),
            )
            for stem in ("val_primary", "test_primary", "full_val_primary"):
                wandb.define_metric(f"{stem}/*")
            wandb.define_metric("interp/*")

        for split_label, datasets in splits_to_run.items():
            for split_name, dataset in datasets.items():
                loader = make_eval_loader(dataset, args.batch_size, args.num_workers)
                print(
                    f"--- Eval alpha={alpha} on {split_label} ({split_name}, "
                    f"{len(dataset)} views, {len(loader)} batches) ---"
                )
                t0 = time.time()
                metrics = evaluate_single_member(
                    model=model,
                    loader=loader,
                    transform=transform,
                    device=device,
                    amp_mode=args.amp_mode,
                )
                elapsed = metrics.pop("_eval_seconds", time.time() - t0)
                n_batches = metrics.pop("_n_batches", float("nan"))
                print(
                    f"    abupt={metrics['abupt_axis_mean_rel_l2_pct']:.4f} "
                    f"WSS={metrics['wall_shear_rel_l2_pct']:.4f} "
                    f"WSS_x={metrics['wall_shear_x_rel_l2_pct']:.4f} "
                    f"WSS_y={metrics['wall_shear_y_rel_l2_pct']:.4f} "
                    f"WSS_z={metrics['wall_shear_z_rel_l2_pct']:.4f} "
                    f"VP={metrics['volume_pressure_rel_l2_pct']:.4f} "
                    f"SP={metrics['surface_pressure_rel_l2_pct']:.4f} "
                    f"({elapsed:.1f}s, {int(n_batches)} batches)"
                )
                # Mirror to the standard *_primary/* keys so dashboards and
                # the program's metric harvester can read them.
                mirror_prefix = "full_val_primary" if split_label == "val" else "test_primary"
                payload = primary_log_payload(mirror_prefix, metrics)
                for k, v in metrics.items():
                    payload[f"{mirror_prefix}/{k}"] = float(v)
                payload[f"{mirror_prefix}/eval_seconds"] = float(elapsed)
                payload[f"{mirror_prefix}/n_batches"] = float(n_batches)
                log_payload.update(payload)
                for k, v in metrics.items():
                    summary_row[f"{split_label}/{k}"] = float(v)
                summary_row[f"{split_label}/eval_seconds"] = float(elapsed)

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            log_payload["peak_memory_gb"] = peak_gb
            summary_row["peak_memory_gb"] = peak_gb
            print(f"  Peak GPU memory after alpha={alpha}: {peak_gb:.2f} GB")

        if run is not None:
            wandb.log(log_payload)
            wandb.summary.update(log_payload)
            wandb.finish()
        summary_rows.append(summary_row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final stdout summary across alphas.
    print("\n=== Summary across alphas ===")
    print(
        f"{'alpha':>5s}  {'val_abupt':>10s} {'val_WSS':>10s} {'val_WSS_x':>10s} "
        f"{'val_WSS_y':>10s} {'val_WSS_z':>10s} {'val_VP':>10s} {'val_SP':>10s} "
        f"{'test_WSS':>10s} {'test_VP':>10s} {'test_SP':>10s}"
    )
    for row in summary_rows:
        print(
            f"{row['alpha']:>5.2f}  "
            f"{row.get('val/abupt_axis_mean_rel_l2_pct', float('nan')):>10.4f} "
            f"{row.get('val/wall_shear_rel_l2_pct', float('nan')):>10.4f} "
            f"{row.get('val/wall_shear_x_rel_l2_pct', float('nan')):>10.4f} "
            f"{row.get('val/wall_shear_y_rel_l2_pct', float('nan')):>10.4f} "
            f"{row.get('val/wall_shear_z_rel_l2_pct', float('nan')):>10.4f} "
            f"{row.get('val/volume_pressure_rel_l2_pct', float('nan')):>10.4f} "
            f"{row.get('val/surface_pressure_rel_l2_pct', float('nan')):>10.4f} "
            f"{row.get('test/wall_shear_rel_l2_pct', float('nan')):>10.4f} "
            f"{row.get('test/volume_pressure_rel_l2_pct', float('nan')):>10.4f} "
            f"{row.get('test/surface_pressure_rel_l2_pct', float('nan')):>10.4f}"
        )


if __name__ == "__main__":
    main()
