# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Inference-time prediction ensemble for DrivAerML.

Loads K trained checkpoints from W&B, runs inference once per model on
the val and/or test split, averages the per-batch normalized
predictions across models, and computes the same MAE / relative-L2
metrics produced by ``train.py``'s final eval. All metrics are logged
to a new W&B run for direct comparison against single-model SOTA
baselines.

The K models are loaded as the ``best`` artifact alias from each W&B
run; the saved state dict already has EMA weights applied
(``checkpoint_source: ema``), so loading it is sufficient for matched
inference.

Example:

    python target/ensemble_eval.py \
      --run-ids 9mm3sz7x 49aimdiz \
      --split val test \
      --wandb-group ensemble-inference-v1 \
      --wandb-name ensemble-k2
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import wandb
import yaml

from data import load_data, pad_collate
from model import SurfaceTransolver
from trainer_runtime import (
    EvalAccumulator,
    TargetTransform,
    autocast_context,
    finalize_eval_accumulator,
    _accumulate_case_rel_l2,
    _masked_sse_count,
)


def parse_run_artifact_name(api: wandb.Api, entity: str, project: str, run_id: str) -> str:
    """Find the model artifact name for a finished training run."""

    run = api.run(f"{entity}/{project}/{run_id}")
    for art in run.logged_artifacts():
        if art.type == "model" and "best" in art.aliases:
            return f"{entity}/{project}/{art.name.split(':')[0]}:best"
    raise RuntimeError(
        f"No model artifact tagged 'best' found for run {run_id}; "
        f"check the run's logged_artifacts()."
    )


def download_checkpoint(
    api: wandb.Api,
    entity: str,
    project: str,
    run_id: str,
    cache_root: Path,
) -> Path:
    """Download the ``best`` model artifact for a run, cached on disk."""

    cache_dir = cache_root / run_id
    if (cache_dir / "checkpoint.pt").exists() and (cache_dir / "config.yaml").exists():
        return cache_dir
    artifact_ref = parse_run_artifact_name(api, entity, project, run_id)
    art = api.artifact(artifact_ref)
    art.download(root=str(cache_dir))
    return cache_dir


def parse_rff_init_sigmas(raw: object) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return [float(v.strip()) for v in text.split(",") if v.strip()] or None
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw] or None
    raise ValueError(f"Unsupported rff_init_sigmas value: {raw!r}")


def build_model_from_config(config: dict) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=int(config.get("model_layers", 3)),
        n_hidden=int(config.get("model_hidden_dim", 192)),
        dropout=float(config.get("model_dropout", 0.0)),
        n_head=int(config.get("model_heads", 3)),
        mlp_ratio=int(config.get("model_mlp_ratio", 4)),
        slice_num=int(config.get("model_slices", 96)),
        rff_num_features=int(config.get("rff_num_features", 0)),
        rff_sigma=float(config.get("rff_sigma", 1.0)),
        rff_init_sigmas=parse_rff_init_sigmas(config.get("rff_init_sigmas", None)),
        pos_encoding_mode=str(config.get("pos_encoding_mode", "sincos")),
        use_qk_norm=bool(config.get("use_qk_norm", False)),
    )


def load_member(
    run_id: str,
    artifact_dir: Path,
    device: torch.device,
) -> tuple[SurfaceTransolver, dict]:
    """Build a model from the run config and load its EMA state dict."""

    config_path = artifact_dir / "config.yaml"
    checkpoint_path = artifact_dir / "checkpoint.pt"
    with config_path.open("r") as fh:
        config = yaml.safe_load(fh)
    model = build_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint for run {run_id} is missing 'model' state dict")
    state_dict = {k: v for k, v in checkpoint["model"].items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch for run {run_id}: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    if checkpoint.get("checkpoint_source") != "ema":
        print(
            f"WARNING: run {run_id} checkpoint_source = "
            f"{checkpoint.get('checkpoint_source')!r}; expected 'ema'."
        )
    print(
        f"Loaded run {run_id}: epoch {checkpoint.get('epoch')}, "
        f"src={checkpoint.get('checkpoint_source')}, "
        f"layers={config.get('model_layers')}/{config.get('model_hidden_dim')}d/"
        f"{config.get('model_heads')}h, "
        f"pos={config.get('pos_encoding_mode')}, qk_norm={config.get('use_qk_norm')}, "
        f"rff_feat={config.get('rff_num_features')}"
    )
    return model, config


def make_eval_loader(dataset, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
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


@torch.no_grad()
def ensemble_predict_batch(
    models: list[SurfaceTransolver],
    batch,
    device: torch.device,
    amp_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average normalized predictions across the K members for one batch."""

    surface_pred_sum: torch.Tensor | None = None
    volume_pred_sum: torch.Tensor | None = None
    for model in models:
        with autocast_context(device, amp_mode):
            out = model(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
        surface_pred = out["surface_preds"].float()
        volume_pred = out["volume_preds"].float()
        if surface_pred_sum is None:
            surface_pred_sum = surface_pred
            volume_pred_sum = volume_pred
        else:
            surface_pred_sum = surface_pred_sum + surface_pred
            volume_pred_sum = volume_pred_sum + volume_pred
    k = float(len(models))
    return surface_pred_sum / k, volume_pred_sum / k


def accumulate_ensemble_batch(
    accumulator: EvalAccumulator,
    *,
    batch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
    device: torch.device,
) -> None:
    """Mirror of accumulate_eval_batch but uses pre-computed averaged preds."""

    batch = batch.to(device)
    surface_target_norm = transform.apply_surface(batch.surface_y)
    volume_target_norm = transform.apply_volume(batch.volume_y)
    surface_sse, surface_count = _masked_sse_count(
        surface_pred_norm, surface_target_norm, batch.surface_mask
    )
    volume_sse, volume_count = _masked_sse_count(
        volume_pred_norm, volume_target_norm, batch.volume_mask
    )
    accumulator.surface_loss_sse += surface_sse
    accumulator.surface_loss_count += surface_count
    accumulator.volume_loss_sse += volume_sse
    accumulator.volume_loss_count += volume_count
    surface_pred = transform.invert_surface(surface_pred_norm)
    volume_pred = transform.invert_volume(volume_pred_norm)

    if bool(batch.surface_mask.any()):
        surface_abs = (surface_pred - batch.surface_y).abs()
        valid_surface_abs = surface_abs[batch.surface_mask]
        accumulator.abs_sums["surface_pressure"] += float(
            valid_surface_abs[:, 0].sum().detach().cpu().item()
        )
        accumulator.abs_counts["surface_pressure"] += int(valid_surface_abs[:, 0].numel())
        wall_abs = valid_surface_abs[:, 1:4]
        accumulator.abs_sums["wall_shear"] += float(wall_abs.sum().detach().cpu().item())
        accumulator.abs_counts["wall_shear"] += int(wall_abs.numel())
        for offset, axis in enumerate(("x", "y", "z")):
            channel = wall_abs[:, offset]
            accumulator.abs_sums[f"wall_shear_{axis}"] += float(
                channel.sum().detach().cpu().item()
            )
            accumulator.abs_counts[f"wall_shear_{axis}"] += int(channel.numel())
        wall_vector_error = torch.linalg.vector_norm(
            surface_pred[batch.surface_mask][:, 1:4]
            - batch.surface_y[batch.surface_mask][:, 1:4],
            dim=-1,
        )
        accumulator.wall_shear_vector_abs_sum += float(
            wall_vector_error.sum().detach().cpu().item()
        )
        accumulator.wall_shear_vector_count += int(wall_vector_error.numel())

    if bool(batch.volume_mask.any()):
        volume_abs = (volume_pred - batch.volume_y).abs()[batch.volume_mask]
        accumulator.abs_sums["volume_pressure"] += float(
            volume_abs[:, 0].sum().detach().cpu().item()
        )
        accumulator.abs_counts["volume_pressure"] += int(volume_abs[:, 0].numel())

    for case_idx, case_id in enumerate(batch.case_ids):
        surface_valid = batch.surface_mask[case_idx].bool()
        if bool(surface_valid.any()):
            surface_pred_valid = surface_pred[case_idx][surface_valid]
            surface_target_valid = batch.surface_y[case_idx][surface_valid]
            _accumulate_case_rel_l2(
                accumulator.case_sums["surface_pressure"],
                case_id=case_id,
                pred=surface_pred_valid[:, 0:1],
                target=surface_target_valid[:, 0:1],
            )
            _accumulate_case_rel_l2(
                accumulator.case_sums["wall_shear"],
                case_id=case_id,
                pred=surface_pred_valid[:, 1:4],
                target=surface_target_valid[:, 1:4],
            )
            for channel, axis in enumerate(("x", "y", "z"), start=1):
                _accumulate_case_rel_l2(
                    accumulator.case_sums[f"wall_shear_{axis}"],
                    case_id=case_id,
                    pred=surface_pred_valid[:, channel : channel + 1],
                    target=surface_target_valid[:, channel : channel + 1],
                )
        volume_valid = batch.volume_mask[case_idx].bool()
        if bool(volume_valid.any()):
            _accumulate_case_rel_l2(
                accumulator.case_sums["volume_pressure"],
                case_id=case_id,
                pred=volume_pred[case_idx][volume_valid],
                target=batch.volume_y[case_idx][volume_valid],
            )


@torch.no_grad()
def evaluate_ensemble_split(
    models: list[SurfaceTransolver],
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
) -> dict[str, float]:
    """Run the K-model ensemble over a split and return finalized metrics."""

    for model in models:
        model.eval()
    accumulator = EvalAccumulator()
    n_batches = 0
    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        surface_pred, volume_pred = ensemble_predict_batch(
            models=models,
            batch=batch,
            device=device,
            amp_mode=amp_mode,
        )
        accumulate_ensemble_batch(
            accumulator,
            batch=batch,
            surface_pred_norm=surface_pred,
            volume_pred_norm=volume_pred,
            transform=transform,
            device=device,
        )
        n_batches += 1
    metrics = finalize_eval_accumulator(accumulator)
    metrics["_eval_seconds"] = time.time() - t0
    metrics["_n_batches"] = float(n_batches)
    return metrics


def evaluate_single_member(
    model: SurfaceTransolver,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
) -> dict[str, float]:
    return evaluate_ensemble_split([model], loader, transform, device, amp_mode)


def primary_log_payload(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    keys = (
        "abupt_axis_mean_rel_l2_pct",
        "surface_pressure_mae",
        "wall_shear_mae",
        "wall_shear_x_mae",
        "wall_shear_y_mae",
        "wall_shear_z_mae",
        "volume_pressure_mae",
        "surface_pressure_rel_l2_pct",
        "wall_shear_rel_l2_pct",
        "wall_shear_x_rel_l2_pct",
        "wall_shear_y_rel_l2_pct",
        "wall_shear_z_rel_l2_pct",
        "volume_pressure_rel_l2_pct",
    )
    return {f"{prefix}/{key}": float(metrics[key]) for key in keys if key in metrics}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-model prediction ensemble for DrivAerML")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
        help="W&B run IDs to ensemble. Each run must have a 'best' model artifact.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["val", "test"],
        default=["val", "test"],
        help="Which splits to evaluate the ensemble on.",
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
        default="outputs/ensemble_cache",
        help="Local directory to cache downloaded W&B model artifacts.",
    )
    parser.add_argument("--wandb-group", default="ensemble-inference-v1")
    parser.add_argument("--wandb-name", default="ensemble")
    parser.add_argument("--wandb-tags", nargs="*", default=["ensemble", "nezuko"])
    parser.add_argument(
        "--include-per-member",
        action="store_true",
        help="Also evaluate each member individually for diagnostic purposes.",
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


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    member_dirs: list[tuple[str, Path]] = []
    for run_id in args.run_ids:
        print(f"Fetching artifact for run {run_id}...")
        member_dir = download_checkpoint(api, entity, project, run_id, cache_root)
        member_dirs.append((run_id, member_dir))

    members: list[SurfaceTransolver] = []
    member_configs: list[dict] = []
    member_run_meta: list[dict] = []
    for run_id, dirpath in member_dirs:
        model, cfg = load_member(run_id, dirpath, device)
        members.append(model)
        member_configs.append(cfg)
        run_obj = api.run(f"{entity}/{project}/{run_id}")
        member_run_meta.append(
            {
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
        )

    print(f"\nEnsemble size K = {len(members)}")
    for meta in member_run_meta:
        print(
            f"  {meta['run_id']:10s} agent={meta['agent']:>10s} "
            f"val={meta['val_abupt']:.4f} full_val={meta['full_val_abupt']:.4f} "
            f"test={meta['test_abupt']:.4f}"
        )

    print("\nLoading data...")
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

    run = None
    if not args.no_wandb:
        wandb_config = {
            "ensemble_size_k": len(members),
            "run_ids": list(args.run_ids),
            "members": member_run_meta,
            "eval_surface_points": args.eval_surface_points,
            "eval_volume_points": args.eval_volume_points,
            "batch_size": args.batch_size,
            "amp_mode": args.amp_mode,
            "splits_evaluated": args.split,
            "include_per_member": args.include_per_member,
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
        wandb.define_metric("ensemble_size_k")
        for stem in ("val_primary", "test_primary", "full_val_primary"):
            wandb.define_metric(f"{stem}/*")
        wandb.define_metric("ensemble_full_val/*")
        wandb.define_metric("ensemble_test/*")
        wandb.define_metric("member_val/*")
        wandb.define_metric("member_test/*")

    summary: dict[str, float] = {"ensemble_size_k": float(len(members))}
    log_payload: dict[str, float] = {"ensemble_size_k": float(len(members))}

    for split_label, datasets in splits_to_run.items():
        for split_name, dataset in datasets.items():
            loader = make_eval_loader(dataset, args.batch_size, args.num_workers)
            print(
                f"\n=== Ensemble eval on {split_label} ({split_name}, "
                f"{len(dataset)} views, {len(loader)} batches) ==="
            )
            metrics = evaluate_ensemble_split(
                models=members,
                loader=loader,
                transform=transform,
                device=device,
                amp_mode=args.amp_mode,
            )
            elapsed = metrics.pop("_eval_seconds", float("nan"))
            n_batches = metrics.pop("_n_batches", float("nan"))
            print(
                f"  abupt_axis_mean_rel_l2_pct={metrics['abupt_axis_mean_rel_l2_pct']:.4f}"
                f"  surface_p={metrics['surface_pressure_rel_l2_pct']:.4f}"
                f"  wall_shear={metrics['wall_shear_rel_l2_pct']:.4f}"
                f"  vp={metrics['volume_pressure_rel_l2_pct']:.4f}"
                f"  tau_x={metrics['wall_shear_x_rel_l2_pct']:.4f}"
                f"  tau_y={metrics['wall_shear_y_rel_l2_pct']:.4f}"
                f"  tau_z={metrics['wall_shear_z_rel_l2_pct']:.4f}"
                f"  ({elapsed:.1f}s, {int(n_batches)} batches)"
            )
            ensemble_prefix = f"ensemble_{split_label}"
            mirror_prefix = "full_val_primary" if split_label == "val" else "test_primary"
            payload = primary_log_payload(ensemble_prefix, metrics)
            payload.update(primary_log_payload(mirror_prefix, metrics))
            for k, v in metrics.items():
                payload[f"{ensemble_prefix}/{k}"] = float(v)
            payload[f"{ensemble_prefix}/eval_seconds"] = float(elapsed)
            payload[f"{ensemble_prefix}/n_batches"] = float(n_batches)
            log_payload.update(payload)
            summary.update(payload)

    if args.include_per_member:
        for run_id, model in zip(args.run_ids, members):
            for split_label, datasets in splits_to_run.items():
                for split_name, dataset in datasets.items():
                    loader = make_eval_loader(dataset, args.batch_size, args.num_workers)
                    print(f"\n--- Member {run_id} on {split_label} ({split_name}) ---")
                    metrics = evaluate_single_member(
                        model=model,
                        loader=loader,
                        transform=transform,
                        device=device,
                        amp_mode=args.amp_mode,
                    )
                    metrics.pop("_eval_seconds", None)
                    metrics.pop("_n_batches", None)
                    print(
                        f"  abupt={metrics['abupt_axis_mean_rel_l2_pct']:.4f} "
                        f"surface_p={metrics['surface_pressure_rel_l2_pct']:.4f} "
                        f"vp={metrics['volume_pressure_rel_l2_pct']:.4f}"
                    )
                    member_prefix = f"member_{split_label}/{run_id}"
                    for k, v in metrics.items():
                        log_payload[f"{member_prefix}/{k}"] = float(v)
                    summary[f"{member_prefix}/abupt_axis_mean_rel_l2_pct"] = float(
                        metrics["abupt_axis_mean_rel_l2_pct"]
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
