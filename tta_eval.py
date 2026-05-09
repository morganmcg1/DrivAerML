# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Test-time augmentation (TTA) evaluation for DrivAerML.

Loads one or more trained checkpoints from W&B, runs inference with
y-axis-mirror TTA (and/or no-TTA baseline), reports the same MAE / relative-L2
metrics produced by ``train.py``'s final eval, and logs everything to a fresh
W&B run for direct comparison against single-model SOTA baselines and the
greedy ensemble.

Y-mirror TTA recipe (DrivAerML's approximate left-right symmetry about y=0):
    1. Negate column 1 (y) of surface_x and volume_x; negate column 4 (n_y)
       of surface_x.  Other columns are invariant.
    2. Run model forward on the mirrored geometry.
    3. Un-mirror predictions:
         - cp (surface_y[:, :, 0]):    invariant -> keep
         - tau_x (surface_y[:, :, 1]): invariant -> keep
         - tau_y (surface_y[:, :, 2]): anti-symmetric -> negate
         - tau_z (surface_y[:, :, 3]): invariant -> keep
         - vol_p (volume_y[:, :, 0]):  invariant -> keep
    4. Average original and un-mirrored predictions before metric computation.

References:
    - SFA / Frame Averaging (arxiv 2305.05577, 2112.01741): orbit-averaging
      over a discrete symmetry group is variance-reducing for an unbiased
      model and Bayes-optimal at uniform 0.5/0.5 weighting.
    - LPSDA (Brandstetter ICML 2022): Navier-Stokes admits bilateral
      symmetry as a Lie point symmetry on symmetric domains, providing
      PDE-level justification for y-flip TTA.

Example:

    cd target/
    python tta_eval.py \
      --checkpoint-run-ids ghh0s4ne \
      --tta-mode y_mirror \
      --splits val test \
      --wandb-group nezuko-tta-eval \
      --wandb-name nezuko/tta-y-mirror-sota
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import torch
import wandb

from data import load_data, pad_collate
from data.loader import SurfaceBatch
from ensemble_eval import (
    accumulate_ensemble_batch,
    download_checkpoint,
    fetch_run_meta,
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


TTA_MODES = ("none", "y_mirror")


# ---------------------------------------------------------------------------
# TTA transforms
# ---------------------------------------------------------------------------


def apply_y_mirror_to_batch(batch: SurfaceBatch) -> SurfaceBatch:
    """Return a new ``SurfaceBatch`` whose geometry is reflected across y=0.

    Negates column 1 (y) of surface_x and volume_x, and column 4 (n_y) of
    surface_x.  Targets, masks, case_ids, and the area / sdf columns are
    untouched -- the mirrored batch is only used for forward inference, so
    targets stay in the original frame for direct loss / metric comparison
    against un-mirrored predictions.

    Other surface_x columns:
        col 0: x         (invariant)
        col 1: y         -> -y
        col 2: z         (invariant)
        col 3: n_x       (invariant)
        col 4: n_y       -> -n_y
        col 5: n_z       (invariant)
        col 6: area      (invariant; scalar)

    Volume_x columns:
        col 0: x         (invariant)
        col 1: y         -> -y
        col 2: z         (invariant)
        col 3: sdf       (invariant; scalar field)
    """

    surface_x = batch.surface_x.clone()
    surface_x[..., 1] = -surface_x[..., 1]
    surface_x[..., 4] = -surface_x[..., 4]
    volume_x = batch.volume_x.clone()
    volume_x[..., 1] = -volume_x[..., 1]
    return SurfaceBatch(
        case_ids=list(batch.case_ids),
        surface_x=surface_x,
        surface_y=batch.surface_y,
        surface_mask=batch.surface_mask,
        volume_x=volume_x,
        volume_y=batch.volume_y,
        volume_mask=batch.volume_mask,
        metadata=list(batch.metadata),
    )


def unmirror_predictions(
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert predictions made on the y-mirrored input back to the canonical frame.

    surface_pred_norm: [B, N_s, 4] = [cp, tau_x, tau_y, tau_z]
        - cp, tau_x, tau_z are invariant under y-flip
        - tau_y is anti-symmetric: tau_y_canonical = -tau_y_mirrored

    volume_pred_norm: [B, N_v, 1] = [vol_pressure]
        - invariant scalar field

    The un-mirror happens in physical space rather than normalized space:
    if normalizer mean for tau_y is non-zero, naive negation of the
    normalized tensor introduces a 2 * mean / std bias.  In DrivAerML the
    train-set tau_y mean is 0.0015 (vs std 1.358), so the bias is ~0.001
    in normalized units — small but non-zero.  Doing the flip in physical
    space and renormalizing eliminates it cleanly.
    """

    surface_pred_phys = transform.invert_surface(surface_pred_norm)
    surface_pred_phys = surface_pred_phys.clone()
    surface_pred_phys[..., 2] = -surface_pred_phys[..., 2]
    surface_un_norm = transform.apply_surface(surface_pred_phys)
    return surface_un_norm, volume_pred_norm


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def tta_predict_batch(
    models: list[SurfaceTransolver],
    batch: SurfaceBatch,
    *,
    tta_mode: str,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TTA inference on one batch.

    With multiple models, predictions are averaged across both the model
    dimension AND the TTA augmentation dimension. This is what the greedy
    ensemble + TTA pathway needs.
    """

    if tta_mode not in TTA_MODES:
        raise ValueError(f"Unknown tta_mode {tta_mode!r}; choose from {TTA_MODES}")

    surface_acc: torch.Tensor | None = None
    volume_acc: torch.Tensor | None = None

    def _add(surf: torch.Tensor, vol: torch.Tensor) -> None:
        nonlocal surface_acc, volume_acc
        if surface_acc is None:
            surface_acc = surf
            volume_acc = vol
        else:
            surface_acc = surface_acc + surf
            volume_acc = volume_acc + vol

    def _forward(model: SurfaceTransolver, b: SurfaceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        with autocast_context(device, amp_mode):
            out = model(
                surface_x=b.surface_x,
                surface_mask=b.surface_mask,
                volume_x=b.volume_x,
                volume_mask=b.volume_mask,
            )
        return out["surface_preds"].float(), out["volume_preds"].float()

    for model in models:
        surf_orig, vol_orig = _forward(model, batch)
        _add(surf_orig, vol_orig)
        if tta_mode == "y_mirror":
            mirrored = apply_y_mirror_to_batch(batch)
            surf_aug, vol_aug = _forward(model, mirrored)
            surf_aug_un, vol_aug_un = unmirror_predictions(surf_aug, vol_aug, transform)
            _add(surf_aug_un, vol_aug_un)

    n_views = float(len(models))
    if tta_mode == "y_mirror":
        n_views *= 2.0
    return surface_acc / n_views, volume_acc / n_views


@torch.no_grad()
def evaluate_tta_split(
    models: list[SurfaceTransolver],
    loader,
    transform: TargetTransform,
    device: torch.device,
    *,
    tta_mode: str,
    amp_mode: str,
) -> dict[str, float]:
    """Evaluate TTA-augmented predictions on a full split."""

    for model in models:
        model.eval()
    accumulator = EvalAccumulator()
    n_batches = 0
    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        surface_pred, volume_pred = tta_predict_batch(
            models=models,
            batch=batch,
            tta_mode=tta_mode,
            transform=transform,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test-time augmentation evaluation for DrivAerML."
    )
    parser.add_argument(
        "--checkpoint-run-ids",
        type=str,
        required=True,
        help=(
            "Comma-separated W&B run IDs of trained checkpoints. With one "
            "ID this evaluates a single model with TTA. With many IDs this "
            "evaluates the multi-model ensemble with TTA averaging."
        ),
    )
    parser.add_argument(
        "--tta-mode",
        choices=list(TTA_MODES),
        default="y_mirror",
        help="TTA augmentation strategy. 'none' = no augmentation (baseline).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["val", "test"],
        default=["val", "test"],
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
        help="Local directory for cached W&B model artifacts.",
    )
    parser.add_argument("--wandb-group", default="nezuko-tta-eval")
    parser.add_argument("--wandb-name", default="nezuko/tta-y-mirror")
    parser.add_argument("--wandb-tags", nargs="*", default=["tta", "nezuko"])
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip W&B run creation (useful for debug smoke tests).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    run_ids = [r.strip() for r in args.checkpoint_run_ids.split(",") if r.strip()]
    if not run_ids:
        raise ValueError("--checkpoint-run-ids parsed to an empty list")

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    members: list[SurfaceTransolver] = []
    member_meta: list[dict] = []
    for run_id in run_ids:
        print(f"Fetching artifact for run {run_id}...")
        member_dir = download_checkpoint(api, entity, project, run_id, cache_root)
        model, _ = load_member(run_id, member_dir, device)
        members.append(model)
        member_meta.append(fetch_run_meta(api, entity, project, run_id))
    print(f"\nLoaded K={len(members)} member(s); tta_mode={args.tta_mode}")
    for m in member_meta:
        val_a = m.get("val_abupt")
        test_a = m.get("test_abupt")
        val_str = "—" if val_a is None else f"{val_a:.4f}"
        test_str = "—" if test_a is None else f"{test_a:.4f}"
        print(
            f"  {m['run_id']:>10s} agent={m.get('agent') or '—':>10s} "
            f"val={val_str} test={test_str}"
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
    if "val" in args.splits:
        splits_to_run["val"] = val_splits
    if "test" in args.splits:
        splits_to_run["test"] = test_splits

    run = None
    if not args.no_wandb:
        wandb_config = {
            "tta_mode": args.tta_mode,
            "checkpoint_run_ids": run_ids,
            "ensemble_size_k": len(members),
            "members": member_meta,
            "eval_surface_points": args.eval_surface_points,
            "eval_volume_points": args.eval_volume_points,
            "batch_size": args.batch_size,
            "amp_mode": args.amp_mode,
            "splits_evaluated": list(args.splits),
        }
        run = wandb.init(
            entity=entity,
            project=project,
            group=args.wandb_group,
            name=args.wandb_name,
            tags=list(args.wandb_tags) + [f"tta:{args.tta_mode}"],
            config=wandb_config,
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        for stem in (
            "tta_val",
            "tta_test",
            "full_val_primary",
            "test_primary",
        ):
            wandb.define_metric(f"{stem}/*")

    summary: dict[str, float | str | int] = {
        "tta_mode": args.tta_mode,
        "ensemble_size_k": int(len(members)),
        "checkpoint_run_ids": ",".join(run_ids),
    }
    log_payload: dict[str, float] = {}
    final_metrics: dict[str, dict[str, float]] = {}

    for split_label, datasets in splits_to_run.items():
        for split_name, dataset in datasets.items():
            loader = make_eval_loader(dataset, args.batch_size, args.num_workers)
            print(
                f"\n=== TTA eval on {split_label} ({split_name}, "
                f"{len(dataset)} views, {len(loader)} batches; tta={args.tta_mode}) ==="
            )
            metrics = evaluate_tta_split(
                models=members,
                loader=loader,
                transform=transform,
                device=device,
                tta_mode=args.tta_mode,
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
            final_metrics[split_label] = metrics
            tta_prefix = f"tta_{split_label}"
            mirror_prefix = "full_val_primary" if split_label == "val" else "test_primary"
            payload = primary_log_payload(tta_prefix, metrics)
            payload.update(primary_log_payload(mirror_prefix, metrics))
            for k, v in metrics.items():
                payload[f"{tta_prefix}/{k}"] = float(v)
            payload[f"{tta_prefix}/eval_seconds"] = float(elapsed)
            payload[f"{tta_prefix}/n_batches"] = float(n_batches)
            log_payload.update(payload)
            summary.update(payload)

    # Val->test ratio diagnostics: keys the PR body asked for.
    if "val" in final_metrics and "test" in final_metrics:
        for key in (
            "abupt_axis_mean_rel_l2_pct",
            "surface_pressure_rel_l2_pct",
            "wall_shear_rel_l2_pct",
            "wall_shear_x_rel_l2_pct",
            "wall_shear_y_rel_l2_pct",
            "wall_shear_z_rel_l2_pct",
            "volume_pressure_rel_l2_pct",
        ):
            v = final_metrics["val"].get(key)
            t = final_metrics["test"].get(key)
            if v and t and v > 0:
                ratio = float(t) / float(v)
                log_payload[f"tta_ratio/{key}"] = ratio
                summary[f"tta_ratio/{key}"] = ratio

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
