# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Inference-time Y-mirror Test-Time Augmentation (TTA) for DrivAerML.

Loads a single trained checkpoint from W&B, runs each eval batch twice
(once on the original input, once on a Y-mirrored copy), averages the
predictions in physical (denormalized) space with the correct
polar-vector sign flips, and reports MAE / relative-L2 metrics.

DrivAerML cars have approximate bilateral (Y-axis) symmetry. Wall shear
stress is a polar vector under reflection, so the Y-mirror image of the
field at a mirrored point has tau_y negated and tau_x, tau_z, cp, vp
unchanged. Surface normals follow the same rule (only ny flips). SDF
and area are scalars and are invariant.

Mirroring is applied to the model inputs:
- surface_x channel 1 (y coord) and channel 4 (ny normal) flip sign.
- volume_x channel 1 (y coord) flips sign.

Predictions are averaged in PHYSICAL space because target normalization
``y_norm = (y - mean) / std`` injects a non-zero mean offset for tau_y
(mean ~= 0.0015) that breaks normalized-space sign flipping.

Both baseline (single forward pass) and TTA metrics are computed and
logged side by side so the gain (or regression) is directly visible.

Example::

    python target/tta_eval.py --run-id nh96x7m4 --split val test \
      --wandb-group fern-tta-mirror-y --wandb-name fern/tta-y-mirror-sota
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
import yaml

from data import load_data, pad_collate
from data.loader import SurfaceBatch
from ensemble_eval import (
    accumulate_ensemble_batch,
    build_model_from_config,
    download_checkpoint,
    load_member,
    make_eval_loader,
    primary_log_payload,
)
from model import SurfaceTransolver
from trainer_runtime import (
    EvalAccumulator,
    TargetTransform,
    autocast_context,
    finalize_eval_accumulator,
)


# Surface input channel layout: [x, y, z, nx, ny, nz, area]
SURFACE_INPUT_Y_IDX = 1
SURFACE_INPUT_NY_IDX = 4
# Volume input channel layout: [x, y, z, sdf]
VOLUME_INPUT_Y_IDX = 1
# Surface output channel layout: [cp, tau_x, tau_y, tau_z]
SURFACE_OUTPUT_TAU_Y_IDX = 2


def y_mirror_surface_x(surface_x: torch.Tensor) -> torch.Tensor:
    out = surface_x.clone()
    out[..., SURFACE_INPUT_Y_IDX] = -out[..., SURFACE_INPUT_Y_IDX]
    out[..., SURFACE_INPUT_NY_IDX] = -out[..., SURFACE_INPUT_NY_IDX]
    return out


def y_mirror_volume_x(volume_x: torch.Tensor) -> torch.Tensor:
    out = volume_x.clone()
    out[..., VOLUME_INPUT_Y_IDX] = -out[..., VOLUME_INPUT_Y_IDX]
    return out


@torch.no_grad()
def tta_predict_batch(
    model: SurfaceTransolver,
    batch: SurfaceBatch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (surface_norm_baseline, volume_norm_baseline,
    surface_norm_tta, volume_norm_tta).

    Both baseline and TTA predictions are returned in NORMALIZED space so
    they can be plugged directly into ``accumulate_ensemble_batch``. The
    TTA average itself is computed in physical space (so the tau_y sign
    flip is exact under the mean/std normalization), then mapped back to
    normalized space via ``transform.apply_*`` (linear, exact inverse of
    ``invert_*``).
    """

    with autocast_context(device, amp_mode):
        out_orig = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
    surface_norm_orig = out_orig["surface_preds"].float()
    volume_norm_orig = out_orig["volume_preds"].float()

    surface_x_m = y_mirror_surface_x(batch.surface_x)
    volume_x_m = y_mirror_volume_x(batch.volume_x)
    with autocast_context(device, amp_mode):
        out_mirror = model(
            surface_x=surface_x_m,
            surface_mask=batch.surface_mask,
            volume_x=volume_x_m,
            volume_mask=batch.volume_mask,
        )
    surface_norm_mirror = out_mirror["surface_preds"].float()
    volume_norm_mirror = out_mirror["volume_preds"].float()

    surface_phys_orig = transform.invert_surface(surface_norm_orig)
    surface_phys_mirror = transform.invert_surface(surface_norm_mirror)
    surface_phys_mirror[..., SURFACE_OUTPUT_TAU_Y_IDX] = -surface_phys_mirror[
        ..., SURFACE_OUTPUT_TAU_Y_IDX
    ]
    surface_phys_tta = 0.5 * (surface_phys_orig + surface_phys_mirror)

    volume_phys_orig = transform.invert_volume(volume_norm_orig)
    volume_phys_mirror = transform.invert_volume(volume_norm_mirror)
    volume_phys_tta = 0.5 * (volume_phys_orig + volume_phys_mirror)

    surface_norm_tta = transform.apply_surface(surface_phys_tta)
    volume_norm_tta = transform.apply_volume(volume_phys_tta)

    surface_mask_unsq = batch.surface_mask.unsqueeze(-1).to(dtype=surface_norm_tta.dtype)
    volume_mask_unsq = batch.volume_mask.unsqueeze(-1).to(dtype=volume_norm_tta.dtype)
    surface_norm_tta = surface_norm_tta * surface_mask_unsq
    volume_norm_tta = volume_norm_tta * volume_mask_unsq

    return surface_norm_orig, volume_norm_orig, surface_norm_tta, volume_norm_tta


@torch.no_grad()
def evaluate_tta_split(
    model: SurfaceTransolver,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    *,
    limit_batches: int = 0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run the TTA loop over a split and return (baseline_metrics, tta_metrics)."""

    model.eval()
    baseline_acc = EvalAccumulator()
    tta_acc = EvalAccumulator()
    n_batches = 0
    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        if limit_batches and batch_idx >= limit_batches:
            break
        batch = batch.to(device)
        s_norm_b, v_norm_b, s_norm_tta, v_norm_tta = tta_predict_batch(
            model=model,
            batch=batch,
            transform=transform,
            device=device,
            amp_mode=amp_mode,
        )
        accumulate_ensemble_batch(
            baseline_acc,
            batch=batch,
            surface_pred_norm=s_norm_b,
            volume_pred_norm=v_norm_b,
            transform=transform,
            device=device,
        )
        accumulate_ensemble_batch(
            tta_acc,
            batch=batch,
            surface_pred_norm=s_norm_tta,
            volume_pred_norm=v_norm_tta,
            transform=transform,
            device=device,
        )
        n_batches += 1

    baseline_metrics = finalize_eval_accumulator(baseline_acc)
    tta_metrics = finalize_eval_accumulator(tta_acc)
    elapsed = time.time() - t0
    baseline_metrics["_eval_seconds"] = elapsed
    tta_metrics["_eval_seconds"] = elapsed
    baseline_metrics["_n_batches"] = float(n_batches)
    tta_metrics["_n_batches"] = float(n_batches)
    return baseline_metrics, tta_metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Y-mirror Test-Time Augmentation eval for DrivAerML"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="W&B run ID whose 'best' model artifact will be loaded.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["val", "test"],
        default=["val", "test"],
        help="Which splits to evaluate.",
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
        default="outputs/tta_cache",
        help="Local directory to cache the downloaded W&B model artifact.",
    )
    parser.add_argument("--wandb-group", default="fern-tta-mirror-y")
    parser.add_argument("--wandb-name", default="fern/tta-y-mirror")
    parser.add_argument("--wandb-tags", nargs="*", default=["tta", "fern", "mirror_y"])
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
    parser.add_argument(
        "--debug-data",
        action="store_true",
        help="Pass debug=True to load_data (4 train / 2 val / 2 test cases).",
    )
    return parser.parse_args(argv)


def fetch_run_meta(api: wandb.Api, entity: str, project: str, run_id: str) -> dict:
    run_obj = api.run(f"{entity}/{project}/{run_id}")
    return {
        "run_id": run_id,
        "agent": run_obj.config.get("agent"),
        "group": run_obj.group,
        "wandb_name": run_obj.config.get("wandb_name"),
        "best_epoch": run_obj.summary_metrics.get("best_epoch"),
        "val_abupt": run_obj.summary_metrics.get(
            "val_primary/abupt_axis_mean_rel_l2_pct"
        ),
        "full_val_abupt": run_obj.summary_metrics.get(
            "full_val_primary/abupt_axis_mean_rel_l2_pct"
        ),
        "test_abupt": run_obj.summary_metrics.get(
            "test_primary/abupt_axis_mean_rel_l2_pct"
        ),
    }


def primary_metric_keys() -> tuple[str, ...]:
    return (
        "abupt_axis_mean_rel_l2_pct",
        "surface_pressure_rel_l2_pct",
        "wall_shear_rel_l2_pct",
        "wall_shear_x_rel_l2_pct",
        "wall_shear_y_rel_l2_pct",
        "wall_shear_z_rel_l2_pct",
        "volume_pressure_rel_l2_pct",
    )


def print_split_compare(split_label: str, baseline: dict[str, float], tta: dict[str, float]) -> None:
    keys = primary_metric_keys()
    print(f"\n=== {split_label.upper()} TTA comparison ===")
    print(f"{'metric':<32s} {'baseline':>12s} {'tta':>12s} {'delta_pp':>12s} {'delta_rel%':>12s}")
    for key in keys:
        b = baseline[key]
        t = tta[key]
        delta = t - b
        rel = (delta / b * 100.0) if b != 0 else float("nan")
        print(f"  {key:<30s} {b:>12.4f} {t:>12.4f} {delta:>+12.4f} {rel:>+12.4f}")


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
    print(f"Fetching artifact for run {args.run_id}...")
    member_dir = download_checkpoint(api, entity, project, args.run_id, cache_root)
    member_meta = fetch_run_meta(api, entity, project, args.run_id)
    print(
        f"Source run: {args.run_id} agent={member_meta['agent']} "
        f"val={member_meta['val_abupt']} full_val={member_meta['full_val_abupt']} "
        f"test={member_meta['test_abupt']} best_epoch={member_meta['best_epoch']}"
    )

    model, member_config = load_member(args.run_id, member_dir, device)

    print("\nLoading data...")
    _, val_splits, test_splits, stats = load_data(
        manifest_path=args.manifest,
        root=args.data_root or None,
        train_surface_points=args.eval_surface_points,
        eval_surface_points=args.eval_surface_points,
        train_volume_points=args.eval_volume_points,
        eval_volume_points=args.eval_volume_points,
        debug=args.debug_data,
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

    run = None
    if not args.no_wandb:
        wandb_config = {
            "run_id": args.run_id,
            "source_run": member_meta,
            "tta_mode": "y_mirror",
            "eval_surface_points": args.eval_surface_points,
            "eval_volume_points": args.eval_volume_points,
            "batch_size": args.batch_size,
            "amp_mode": args.amp_mode,
            "splits_evaluated": args.split,
            "model_config": member_config,
        }
        run = wandb.init(
            entity=entity,
            project=project,
            group=args.wandb_group,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=wandb_config,
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        for stem in (
            "baseline_val",
            "baseline_test",
            "tta_val",
            "tta_test",
            "delta_val",
            "delta_test",
            "full_val_primary",
            "test_primary",
        ):
            wandb.define_metric(f"{stem}/*")

    summary: dict[str, float | str] = {
        "tta_mode": "y_mirror",
        "source_run_id": args.run_id,
    }
    log_payload: dict[str, float] = {}

    for split_label, datasets in splits_to_run.items():
        for split_name, dataset in datasets.items():
            loader = make_eval_loader(dataset, args.batch_size, args.num_workers)
            print(
                f"\n=== {split_label} ({split_name}, {len(dataset)} views, "
                f"{len(loader)} batches) ==="
            )
            baseline_metrics, tta_metrics = evaluate_tta_split(
                model=model,
                loader=loader,
                transform=transform,
                device=device,
                amp_mode=args.amp_mode,
                limit_batches=args.limit_batches,
            )
            elapsed = baseline_metrics.pop("_eval_seconds", float("nan"))
            n_batches = baseline_metrics.pop("_n_batches", float("nan"))
            tta_metrics.pop("_eval_seconds", None)
            tta_metrics.pop("_n_batches", None)
            print(
                f"  baseline: abupt={baseline_metrics['abupt_axis_mean_rel_l2_pct']:.4f}  "
                f"  tta: abupt={tta_metrics['abupt_axis_mean_rel_l2_pct']:.4f}  "
                f"  ({elapsed:.1f}s, {int(n_batches)} batches)"
            )
            print_split_compare(split_label, baseline_metrics, tta_metrics)

            base_prefix = f"baseline_{split_label}"
            tta_prefix = f"tta_{split_label}"
            delta_prefix = f"delta_{split_label}"
            mirror_prefix = "full_val_primary" if split_label == "val" else "test_primary"

            payload = primary_log_payload(base_prefix, baseline_metrics)
            payload.update(primary_log_payload(tta_prefix, tta_metrics))
            for k, v in baseline_metrics.items():
                payload[f"{base_prefix}/{k}"] = float(v)
            for k, v in tta_metrics.items():
                payload[f"{tta_prefix}/{k}"] = float(v)
            for key in primary_metric_keys():
                delta_pp = float(tta_metrics[key]) - float(baseline_metrics[key])
                rel = (
                    delta_pp / float(baseline_metrics[key]) * 100.0
                    if float(baseline_metrics[key]) != 0
                    else float("nan")
                )
                payload[f"{delta_prefix}/{key}_pp"] = float(delta_pp)
                payload[f"{delta_prefix}/{key}_rel_pct"] = float(rel)
            payload.update(primary_log_payload(mirror_prefix, tta_metrics))
            payload[f"{tta_prefix}/eval_seconds"] = float(elapsed)
            payload[f"{tta_prefix}/n_batches"] = float(n_batches)
            log_payload.update(payload)
            summary.update(payload)
            summary[f"{base_prefix}/abupt_axis_mean_rel_l2_pct"] = float(
                baseline_metrics["abupt_axis_mean_rel_l2_pct"]
            )
            summary[f"{tta_prefix}/abupt_axis_mean_rel_l2_pct"] = float(
                tta_metrics["abupt_axis_mean_rel_l2_pct"]
            )
            summary[f"{delta_prefix}/abupt_axis_mean_rel_l2_pct_pp"] = float(
                tta_metrics["abupt_axis_mean_rel_l2_pct"]
                - baseline_metrics["abupt_axis_mean_rel_l2_pct"]
            )

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        log_payload["peak_memory_gb"] = peak_gb
        summary["peak_memory_gb"] = peak_gb
        print(f"\nPeak GPU memory: {peak_gb:.2f} GB")

    if run is not None:
        wandb.log(log_payload)
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
