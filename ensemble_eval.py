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
from data.loader import SurfaceBatch
from model import SurfaceTransolver
from trainer_runtime import (
    EvalAccumulator,
    TargetTransform,
    accumulate_eval_batch,
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


def resolve_pos_encoding_mode(config: dict) -> str:
    """Normalize the PE mode across config-key variants used by historical runs.

    Newer configs use ``pos_encoding_mode`` (``sincos`` | ``string_separable``).
    Earlier multi-sigma runs (PR #968/#972) used ``model_pe`` with the value
    ``string_multisigma``. Map both to the canonical key the model accepts.
    """

    mode = config.get("pos_encoding_mode")
    if mode is not None:
        return str(mode)
    legacy = config.get("model_pe")
    if legacy is not None:
        return str(legacy)
    return "sincos"


def build_model_from_config(
    config: dict,
    *,
    use_aux_decoder_heads: bool = False,
) -> SurfaceTransolver:
    rff_num_features = int(config.get("rff_num_features", 0))
    # Historical configs that used MultiSigmaStringPosEmbed parked the feature
    # count under ``pe_num_features``; fall back to it when rff_num_features=0.
    if rff_num_features == 0:
        rff_num_features = int(config.get("pe_num_features", 0))
    init_sigmas = parse_rff_init_sigmas(
        config.get("rff_init_sigmas") or config.get("pe_init_sigmas")
    )
    return SurfaceTransolver(
        n_layers=int(config.get("model_layers", 3)),
        n_hidden=int(config.get("model_hidden_dim", 192)),
        dropout=float(config.get("model_dropout", 0.0)),
        n_head=int(config.get("model_heads", 3)),
        mlp_ratio=int(config.get("model_mlp_ratio", 4)),
        slice_num=int(config.get("model_slices", 96)),
        rff_num_features=rff_num_features,
        rff_sigma=float(config.get("rff_sigma", 1.0)),
        rff_init_sigmas=init_sigmas,
        pos_encoding_mode=resolve_pos_encoding_mode(config),
        use_qk_norm=bool(config.get("use_qk_norm", False)),
        use_surf_to_vol_xattn=bool(config.get("use_surf_to_vol_xattn", False)),
        use_aux_decoder_heads=use_aux_decoder_heads,
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint for run {run_id} is missing 'model' state dict")
    state_dict = {k: v for k, v in checkpoint["model"].items()}
    # PR #958 used Sequential heads (e.g. ``surface_out.0.weight``) — not
    # recorded in config, so detect it from the state dict directly.
    use_aux_decoder_heads = "surface_out.0.weight" in state_dict
    model = build_model_from_config(
        config, use_aux_decoder_heads=use_aux_decoder_heads
    ).to(device)
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


@torch.no_grad()
def cache_member_predictions_split(
    model: SurfaceTransolver,
    loader,
    device: torch.device,
    amp_mode: str,
    *,
    capture_meta: bool,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> tuple[list[dict[str, torch.Tensor | list[str]]], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Run one forward pass over a split, return per-batch predictions on CPU.

    When ``capture_meta`` is True we additionally cache batch metadata
    (case_ids, masks, targets) so subsequent candidates can skip storing
    duplicates.

    Predictions are stored as ``storage_dtype`` (default bf16) on CPU pinned
    memory for fast greedy ensemble averaging across many candidates without
    re-running the model.
    """

    model.eval()
    batch_meta: list[dict[str, torch.Tensor | list[str]]] = []
    pred_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch in loader:
        batch = batch.to(device)
        with autocast_context(device, amp_mode):
            out = model(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
        surface_pred = out["surface_preds"].float().detach()
        volume_pred = out["volume_preds"].float().detach()
        pred_list.append(
            (
                surface_pred.to(dtype=storage_dtype, device="cpu", copy=True),
                volume_pred.to(dtype=storage_dtype, device="cpu", copy=True),
            )
        )
        if capture_meta:
            batch_meta.append(
                {
                    "case_ids": list(batch.case_ids),
                    "surface_y": batch.surface_y.detach().cpu(),
                    "volume_y": batch.volume_y.detach().cpu(),
                    "surface_mask": batch.surface_mask.detach().cpu(),
                    "volume_mask": batch.volume_mask.detach().cpu(),
                }
            )
    return batch_meta, pred_list


def pred_cache_path(cache_root: Path, run_id: str, split_label: str) -> Path:
    return cache_root / "predictions" / run_id / f"{split_label}.pt"


def meta_cache_path(cache_root: Path, split_label: str) -> Path:
    return cache_root / "predictions" / "_meta" / f"{split_label}.pt"


def _atomic_torch_save(obj, path: Path) -> None:
    """Save with a pid-stamped tmp filename so concurrent writers don't collide."""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    try:
        tmp.replace(path)
    except FileNotFoundError:
        # Another worker may have completed the rename concurrently.
        pass


def save_pred_cache_to_disk(
    pred_list: list[tuple[torch.Tensor, torch.Tensor]],
    path: Path,
) -> None:
    _atomic_torch_save(pred_list, path)


def save_meta_cache_to_disk(
    batch_meta: list[dict[str, torch.Tensor | list[str]]],
    path: Path,
) -> None:
    _atomic_torch_save(batch_meta, path)


def load_pred_cache_from_disk(path: Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_meta_cache_from_disk(path: Path) -> list[dict[str, torch.Tensor | list[str]]]:
    return torch.load(path, map_location="cpu", weights_only=False)


def reconstruct_batch_from_meta(
    meta: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> SurfaceBatch:
    """Build a minimal ``SurfaceBatch`` from cached metadata.

    The ``surface_x`` / ``volume_x`` slots are filled with empty placeholders
    because the accumulator only consumes targets, masks, and case_ids — but
    SurfaceBatch.to() still needs valid tensors.
    """

    surface_y = meta["surface_y"]
    volume_y = meta["volume_y"]
    return SurfaceBatch(
        case_ids=list(meta["case_ids"]),
        surface_x=torch.empty(
            (surface_y.shape[0], surface_y.shape[1], 0),
            dtype=torch.float32,
        ),
        surface_y=surface_y,
        surface_mask=meta["surface_mask"],
        volume_x=torch.empty(
            (volume_y.shape[0], volume_y.shape[1], 0),
            dtype=torch.float32,
        ),
        volume_y=volume_y,
        volume_mask=meta["volume_mask"],
        metadata=[],
    ).to(device)


def metrics_for_member_set_from_cache(
    member_run_ids: list[str],
    batch_meta: list[dict[str, torch.Tensor | list[str]]],
    pred_cache_split: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    transform: TargetTransform,
    device: torch.device,
) -> dict[str, float]:
    """Compute split metrics for an ensemble subset using cached predictions.

    Predictions are averaged across ``member_run_ids`` (duplicates allowed —
    used by Caruana's with-replacement variant). No model forward passes are
    triggered.
    """

    if not member_run_ids:
        raise ValueError("member_run_ids must be non-empty")
    accumulator = EvalAccumulator()
    n = float(len(member_run_ids))
    for batch_idx, meta in enumerate(batch_meta):
        sum_surf: torch.Tensor | None = None
        sum_vol: torch.Tensor | None = None
        for run_id in member_run_ids:
            sp, vp = pred_cache_split[run_id][batch_idx]
            sp = sp.to(device=device, dtype=torch.float32, non_blocking=True)
            vp = vp.to(device=device, dtype=torch.float32, non_blocking=True)
            sum_surf = sp if sum_surf is None else sum_surf + sp
            sum_vol = vp if sum_vol is None else sum_vol + vp
        avg_surf = sum_surf / n
        avg_vol = sum_vol / n
        batch = reconstruct_batch_from_meta(meta, device)
        accumulate_ensemble_batch(
            accumulator,
            batch=batch,
            surface_pred_norm=avg_surf,
            volume_pred_norm=avg_vol,
            transform=transform,
            device=device,
        )
    return finalize_eval_accumulator(accumulator)


def greedy_forward_select(
    candidate_run_ids: list[str],
    batch_meta_val: list[dict[str, torch.Tensor | list[str]]],
    pred_cache_val: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    transform: TargetTransform,
    device: torch.device,
    *,
    max_k: int,
    min_improvement: float,
    allow_replacement: bool,
    selection_metric_key: str = "abupt_axis_mean_rel_l2_pct",
) -> tuple[list[str], list[dict[str, float | str]], dict[str, dict[str, float]]]:
    """Greedy forward ensemble selection on the val split (Caruana 2004).

    Returns ``(selected_run_ids, trajectory, per_candidate_metrics)``:

    * ``selected_run_ids`` — the chosen ensemble in selection order.
    * ``trajectory`` — one row per step with selected_run_id, ensemble_size,
      val metric and delta vs the previous step.
    * ``per_candidate_metrics`` — full single-member val metrics for every
      candidate (handy for diagnostics and the W&B summary).

    The selection metric is minimised. Early stopping triggers when the best
    next candidate improves the metric by less than ``min_improvement``.
    """

    if max_k < 1:
        raise ValueError("max_k must be >= 1")

    print("\n=== Greedy step 0: per-candidate val metrics ===")
    per_candidate_metrics: dict[str, dict[str, float]] = {}
    for run_id in candidate_run_ids:
        m = metrics_for_member_set_from_cache(
            [run_id],
            batch_meta_val,
            pred_cache_val,
            transform=transform,
            device=device,
        )
        per_candidate_metrics[run_id] = m
        print(
            f"  {run_id}: val_{selection_metric_key}={m[selection_metric_key]:.4f}  "
            f"surface_p={m['surface_pressure_rel_l2_pct']:.4f}  "
            f"wall_shear={m['wall_shear_rel_l2_pct']:.4f}  "
            f"vp={m['volume_pressure_rel_l2_pct']:.4f}"
        )

    seed_run_id = min(
        candidate_run_ids,
        key=lambda r: per_candidate_metrics[r][selection_metric_key],
    )
    seed_metric = per_candidate_metrics[seed_run_id][selection_metric_key]
    selected: list[str] = [seed_run_id]
    trajectory: list[dict[str, float | str]] = [
        {
            "step": 1,
            "selected_run_id": seed_run_id,
            "ensemble_size": 1,
            "val_metric": float(seed_metric),
            "delta_val_metric": float("nan"),
        }
    ]
    print(
        f"\nGreedy step 1: seed = {seed_run_id} "
        f"(val_{selection_metric_key}={seed_metric:.4f})"
    )

    while len(selected) < max_k:
        if allow_replacement:
            candidates_to_try = list(candidate_run_ids)
        else:
            candidates_to_try = [c for c in candidate_run_ids if c not in selected]
        if not candidates_to_try:
            print("Greedy: candidate pool exhausted (no-replacement mode); stopping.")
            break
        prev_metric = trajectory[-1]["val_metric"]
        best_c = None
        best_metric = float("inf")
        best_full_metrics: dict[str, float] | None = None
        for c in candidates_to_try:
            trial_set = selected + [c]
            m = metrics_for_member_set_from_cache(
                trial_set,
                batch_meta_val,
                pred_cache_val,
                transform=transform,
                device=device,
            )
            metric = m[selection_metric_key]
            if metric < best_metric:
                best_metric = metric
                best_c = c
                best_full_metrics = m
        delta = float(prev_metric) - float(best_metric)
        next_step = len(selected) + 1
        print(
            f"Greedy step {next_step}: best candidate = {best_c} "
            f"(val_{selection_metric_key}={best_metric:.4f}, delta={delta:+.4f}pp)"
        )
        if delta < min_improvement:
            print(
                f"  -> delta {delta:+.4f}pp < min_improvement {min_improvement:.4f}pp; stopping."
            )
            break
        selected.append(best_c)
        trajectory.append(
            {
                "step": next_step,
                "selected_run_id": best_c,
                "ensemble_size": len(selected),
                "val_metric": float(best_metric),
                "delta_val_metric": float(delta),
            }
        )

    return selected, trajectory, per_candidate_metrics


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
        required=False,
        default=None,
        help=(
            "W&B run IDs to ensemble (fixed-set mode). Each run must have a "
            "'best' model artifact. Either --run-ids or (--greedy with "
            "--candidate-run-ids) is required."
        ),
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
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Enable greedy forward ensemble selection (Caruana 2004) on the val split.",
    )
    parser.add_argument(
        "--candidate-run-ids",
        type=str,
        default="",
        help=(
            "Comma-separated W&B run IDs forming the candidate pool for "
            "--greedy. Each run must have a 'best' model artifact."
        ),
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=8,
        help="Greedy: maximum ensemble size (default 8).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        help=(
            "Greedy: early-stop threshold (val_abupt percentage points). "
            "Stop if best next candidate improves the metric by less than "
            "this value (default 0.005pp)."
        ),
    )
    parser.add_argument(
        "--allow-replacement",
        action="store_true",
        help=(
            "Greedy: allow the same member to be picked multiple times "
            "(Caruana's original formulation; off by default)."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        default="abupt_axis_mean_rel_l2_pct",
        help="Greedy: validation metric to minimise (default abupt_axis_mean_rel_l2_pct).",
    )
    parser.add_argument(
        "--pred-cache-dir",
        type=str,
        default="",
        help=(
            "Directory for caching per-candidate predictions to disk so multiple "
            "GPU workers can share inference cost. When set, predictions are "
            "loaded from disk if present (skipping inference) and saved after "
            "running inference. Must outlive a single process."
        ),
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "Greedy: run inference and save predictions to --pred-cache-dir, "
            "then exit. Used for distributing inference across GPUs."
        ),
    )
    args = parser.parse_args(argv)
    if args.greedy:
        if not args.candidate_run_ids.strip():
            parser.error("--greedy requires --candidate-run-ids")
    else:
        if not args.run_ids:
            parser.error("--run-ids is required when --greedy is not set")
    return args


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


def main_greedy(args: argparse.Namespace) -> None:
    """Greedy forward ensemble selection (Caruana 2004) over a candidate pool.

    Inference cost is one forward pass per candidate per requested split.
    Selection itself is O(K * |pool|) tensor averages on cached predictions.
    """

    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    candidate_run_ids = [
        rid.strip() for rid in args.candidate_run_ids.split(",") if rid.strip()
    ]
    if not candidate_run_ids:
        raise ValueError("--candidate-run-ids parsed to an empty list")
    print(f"Greedy candidate pool size: {len(candidate_run_ids)}")
    splits_requested = list(args.split)
    if "val" not in splits_requested:
        raise ValueError("--greedy requires 'val' in --split (selection runs on val)")

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    # Resolve candidates: download artifacts; drop ones missing a 'best' artifact.
    resolved: list[tuple[str, Path, dict]] = []
    skipped: list[tuple[str, str]] = []
    for run_id in candidate_run_ids:
        try:
            print(f"Fetching artifact for candidate {run_id}...")
            member_dir = download_checkpoint(api, entity, project, run_id, cache_root)
            run_meta = fetch_run_meta(api, entity, project, run_id)
            resolved.append((run_id, member_dir, run_meta))
        except Exception as exc:  # noqa: BLE001 — best-effort skip
            print(f"  WARNING: skipping {run_id} ({exc})")
            skipped.append((run_id, str(exc)))
    if not resolved:
        raise RuntimeError("No candidates resolved successfully")
    if skipped:
        print(f"Skipped {len(skipped)} candidate(s) due to missing artifacts.")

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
    if "val" in splits_requested:
        splits_to_run["val"] = val_splits
    if "test" in splits_requested:
        splits_to_run["test"] = test_splits

    # Phase 1: inference per candidate. Predictions on CPU bf16; metadata once
    # per split (assumes deterministic eval_chunk loader yields the same order).
    batch_meta_per_split: dict[str, list[dict]] = {label: [] for label in splits_to_run}
    pred_cache: dict[str, dict[str, list[tuple[torch.Tensor, torch.Tensor]]]] = {
        label: {} for label in splits_to_run
    }

    candidate_run_ids_resolved = [r for r, _, _ in resolved]
    member_meta_by_id: dict[str, dict] = {meta["run_id"]: meta for _, _, meta in resolved}

    pred_cache_dir = Path(args.pred_cache_dir) if args.pred_cache_dir else None
    if pred_cache_dir is not None:
        pred_cache_dir.mkdir(parents=True, exist_ok=True)

    # Try to load batch metadata from disk first (any worker that ran inference
    # may have written it). Otherwise capture in this process from the first
    # candidate that runs inference.
    if pred_cache_dir is not None:
        for split_label in splits_to_run:
            mp = meta_cache_path(pred_cache_dir, split_label)
            if mp.exists():
                batch_meta_per_split[split_label] = load_meta_cache_from_disk(mp)
                print(f"Loaded batch metadata for {split_label} from disk ({len(batch_meta_per_split[split_label])} batches)")

    print("\n=== Phase 1: caching per-candidate predictions ===")
    inference_done_for: set[str] = set()
    load_failures: list[tuple[str, str]] = []
    for cand_idx, (run_id, dirpath, _) in enumerate(resolved):
        # Try to load all per-split predictions from disk first.
        loaded_all_splits = pred_cache_dir is not None
        if pred_cache_dir is not None:
            for split_label in splits_to_run:
                pp = pred_cache_path(pred_cache_dir, run_id, split_label)
                if pp.exists():
                    pred_cache[split_label][run_id] = load_pred_cache_from_disk(pp)
                else:
                    loaded_all_splits = False
            if loaded_all_splits:
                print(
                    f"  [{cand_idx + 1}/{len(resolved)}] {run_id}: "
                    f"loaded all splits from disk cache"
                )
                continue

        # Fall back to inference for the splits that aren't cached yet. We
        # tolerate model-architecture mismatches (e.g. multi-band variants that
        # cannot be reconstructed from the current build_model_from_config) by
        # skipping the candidate rather than aborting the whole run.
        try:
            model, _ = load_member(run_id, dirpath, device)
        except Exception as exc:  # noqa: BLE001 — best-effort skip
            print(f"  WARNING: skipping {run_id} during load_member ({exc})")
            load_failures.append((run_id, str(exc)))
            for split_label in list(splits_to_run.keys()):
                pred_cache[split_label].pop(run_id, None)
            continue
        for split_label, datasets in splits_to_run.items():
            if run_id in pred_cache[split_label]:
                continue  # already loaded from disk above
            for _, dataset in datasets.items():
                loader = make_eval_loader(dataset, args.batch_size, args.num_workers)
                t0 = time.time()
                capture_meta = not bool(batch_meta_per_split[split_label])
                meta_list, pred_list = cache_member_predictions_split(
                    model=model,
                    loader=loader,
                    device=device,
                    amp_mode=args.amp_mode,
                    capture_meta=capture_meta,
                )
                if capture_meta:
                    batch_meta_per_split[split_label] = meta_list
                    if pred_cache_dir is not None:
                        save_meta_cache_to_disk(
                            meta_list, meta_cache_path(pred_cache_dir, split_label)
                        )
                pred_cache[split_label][run_id] = pred_list
                if pred_cache_dir is not None:
                    save_pred_cache_to_disk(
                        pred_list, pred_cache_path(pred_cache_dir, run_id, split_label)
                    )
                print(
                    f"  [{cand_idx + 1}/{len(resolved)}] {run_id} {split_label}: "
                    f"{len(pred_list)} batches cached ({time.time() - t0:.1f}s)"
                )
        inference_done_for.add(run_id)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if load_failures:
        # Drop failed candidates from the working pool so greedy doesn't try
        # to reference them. Retain the metadata for diagnostics.
        failed_ids = {rid for rid, _ in load_failures}
        candidate_run_ids_resolved = [
            r for r in candidate_run_ids_resolved if r not in failed_ids
        ]
        skipped.extend(load_failures)
        print(
            f"Dropped {len(failed_ids)} candidate(s) due to model-load failures: "
            f"{sorted(failed_ids)}"
        )

    if args.cache_only:
        print(
            f"\n--cache-only: ran inference for {len(inference_done_for)} candidate(s); "
            "exiting before greedy selection."
        )
        return

    for split_label in splits_to_run:
        if not batch_meta_per_split[split_label]:
            raise RuntimeError(
                f"No batch metadata for split '{split_label}'. "
                "Run inference once with --pred-cache-dir to populate it, "
                "or include at least one candidate not in the disk cache."
            )

    # Phase 2: greedy selection on val.
    print("\n=== Phase 2: greedy forward selection on val ===")
    selected, trajectory, per_candidate_metrics = greedy_forward_select(
        candidate_run_ids=candidate_run_ids_resolved,
        batch_meta_val=batch_meta_per_split["val"],
        pred_cache_val=pred_cache["val"],
        transform=transform,
        device=device,
        max_k=args.max_k,
        min_improvement=args.min_improvement,
        allow_replacement=args.allow_replacement,
        selection_metric_key=args.selection_metric,
    )

    print("\nSelected ensemble:")
    for step in trajectory:
        delta = step["delta_val_metric"]
        delta_str = "—" if (isinstance(delta, float) and math.isnan(delta)) else f"{delta:+.4f}pp"
        print(
            f"  step={step['step']:>2} K={step['ensemble_size']:>2}  "
            f"+{step['selected_run_id']}  "
            f"val_{args.selection_metric}={step['val_metric']:.4f}  "
            f"delta={delta_str}"
        )

    # Phase 3: final ensemble metrics on val and test.
    final_metrics: dict[str, dict[str, float]] = {}
    for split_label in splits_to_run:
        final_metrics[split_label] = metrics_for_member_set_from_cache(
            selected,
            batch_meta_per_split[split_label],
            pred_cache[split_label],
            transform=transform,
            device=device,
        )
    print("\n=== Final ensemble metrics ===")
    for split_label, m in final_metrics.items():
        print(
            f"  [{split_label}] abupt={m['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"surface_p={m['surface_pressure_rel_l2_pct']:.4f}  "
            f"wall_shear={m['wall_shear_rel_l2_pct']:.4f}  "
            f"vp={m['volume_pressure_rel_l2_pct']:.4f}  "
            f"tau_x={m['wall_shear_x_rel_l2_pct']:.4f}  "
            f"tau_y={m['wall_shear_y_rel_l2_pct']:.4f}  "
            f"tau_z={m['wall_shear_z_rel_l2_pct']:.4f}"
        )

    # Phase 4: W&B logging.
    run = None
    if not args.no_wandb:
        wandb_config = {
            "mode": "greedy_forward_selection",
            "candidate_pool_size": len(candidate_run_ids_resolved),
            "candidate_run_ids": candidate_run_ids_resolved,
            "skipped_candidate_run_ids": skipped,
            "max_k": args.max_k,
            "min_improvement": args.min_improvement,
            "allow_replacement": args.allow_replacement,
            "selection_metric": args.selection_metric,
            "eval_surface_points": args.eval_surface_points,
            "eval_volume_points": args.eval_volume_points,
            "batch_size": args.batch_size,
            "amp_mode": args.amp_mode,
            "splits_evaluated": splits_requested,
            "members": [member_meta_by_id[r] for r in candidate_run_ids_resolved],
        }
        run = wandb.init(
            entity=entity,
            project=project,
            group=args.wandb_group,
            name=args.wandb_name,
            tags=list(args.wandb_tags) + ["greedy"],
            config=wandb_config,
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        wandb.define_metric("greedy_step")
        wandb.define_metric("greedy/*", step_metric="greedy_step")
        wandb.define_metric("ensemble_full_val/*")
        wandb.define_metric("ensemble_test/*")
        wandb.define_metric("full_val_primary/*")
        wandb.define_metric("test_primary/*")
        wandb.define_metric("candidate_val/*")
        wandb.define_metric("member_val/*")
        wandb.define_metric("member_test/*")

    # Per-step trajectory rows
    if run is not None:
        for step in trajectory:
            delta = step["delta_val_metric"]
            wandb.log(
                {
                    "greedy_step": int(step["step"]),
                    "greedy/ensemble_size": int(step["ensemble_size"]),
                    "greedy/selected_run_id": str(step["selected_run_id"]),
                    f"greedy/val_{args.selection_metric}": float(step["val_metric"]),
                    f"greedy/delta_val_{args.selection_metric}": (
                        0.0 if isinstance(delta, float) and math.isnan(delta) else float(delta)
                    ),
                }
            )

    summary: dict[str, float | str | int] = {
        "ensemble_size_k": int(len(selected)),
        "selected_run_ids": ",".join(selected),
        "candidate_pool_size": int(len(candidate_run_ids_resolved)),
        "max_k": int(args.max_k),
        "allow_replacement": bool(args.allow_replacement),
    }
    log_payload: dict[str, float] = {}

    for split_label, m in final_metrics.items():
        ensemble_prefix = f"ensemble_{split_label}"
        mirror_prefix = "full_val_primary" if split_label == "val" else "test_primary"
        payload = primary_log_payload(ensemble_prefix, m)
        payload.update(primary_log_payload(mirror_prefix, m))
        for k, v in m.items():
            payload[f"{ensemble_prefix}/{k}"] = float(v)
        log_payload.update(payload)
        summary.update(payload)
        summary[f"{ensemble_prefix}/abupt_axis_mean_rel_l2_pct"] = float(
            m["abupt_axis_mean_rel_l2_pct"]
        )

    # Per-candidate single-member val metrics (for diagnostics).
    for run_id, m in per_candidate_metrics.items():
        for k, v in m.items():
            log_payload[f"candidate_val/{run_id}/{k}"] = float(v)
        summary[f"candidate_val/{run_id}/abupt_axis_mean_rel_l2_pct"] = float(
            m["abupt_axis_mean_rel_l2_pct"]
        )

    if args.include_per_member:
        # Per-member metrics across both splits using the cache (zero extra inference).
        for run_id in candidate_run_ids_resolved:
            for split_label in splits_to_run:
                member_metrics = metrics_for_member_set_from_cache(
                    [run_id],
                    batch_meta_per_split[split_label],
                    pred_cache[split_label],
                    transform=transform,
                    device=device,
                )
                for k, v in member_metrics.items():
                    log_payload[f"member_{split_label}/{run_id}/{k}"] = float(v)
                summary[f"member_{split_label}/{run_id}/abupt_axis_mean_rel_l2_pct"] = float(
                    member_metrics["abupt_axis_mean_rel_l2_pct"]
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


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.greedy:
        main_greedy(args)
        return

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
