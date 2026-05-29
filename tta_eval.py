# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""H211: Test-Time Augmentation (TTA) eval for H190 EP13 checkpoint.

Loads a trained DrivAerML checkpoint and computes val + test metrics under
three eval modes in a single pass:

* original  — forward on raw geometry (Pass A).
* mirrored  — forward on y=0 mirrored geometry, un-mirrored back (Pass B).
* tta       — average of (Pass A + un-mirrored Pass B) per point.

The mirror transform mirrors only the model INPUT — targets stay unchanged so
metrics are directly comparable across modes. The mirror input flip negates
``surface_x[..., 1]`` (y position), ``surface_x[..., 4]`` (normal_y) and
``volume_x[..., 1]`` (volume y position); sdf, normal_x, normal_z, x, z, area
are invariant. The mirrored-output un-mirror negates ``surface_preds[..., 2]``
(tau_y); cp, tau_x, tau_z, volume_pressure are invariant.

Run with::

    torchrun --standalone --nproc-per-node=8 tta_eval.py \\
        --checkpoint outputs/h211_tta/h190_ep13_best/checkpoint.pt \\
        --wandb-group h211-nezuko-tta-h190 \\
        --wandb-name nezuko/h211-tta-h190-ep13
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel

from data import SurfaceBatch
from model import SurfaceTransolver
from train import Config, build_model
from trainer_runtime import (
    DistributedState,
    EvalAccumulator,
    TargetTransform,
    autocast_context,
    cleanup_distributed,
    define_wandb_metrics,
    distributed_barrier,
    finalize_eval_accumulator,
    init_distributed,
    make_loaders,
    merge_eval_accumulators,
    metric_namespace,
    primary_metric_log,
    print_metrics,
    unwrap_model,
)


def parse_tta_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H211 TTA eval for DrivAerML")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint.pt produced by train.py (contains 'model' and 'config').",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511",
        help="Override the data_root recorded in the checkpoint config.",
    )
    parser.add_argument("--manifest", type=str, default="data/split_manifest.json")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override batch size (0 = use checkpoint config).",
    )
    parser.add_argument(
        "--eval-surface-points",
        type=int,
        default=0,
        help="Override eval_surface_points (0 = use checkpoint config).",
    )
    parser.add_argument(
        "--eval-volume-points",
        type=int,
        default=0,
        help="Override eval_volume_points (0 = use checkpoint config).",
    )
    parser.add_argument("--wandb-group", type=str, default="h211-nezuko-tta-h190")
    parser.add_argument("--wandb-name", type=str, default="nezuko/h211-tta-h190-ep13")
    parser.add_argument("--agent", type=str, default="nezuko")
    return parser.parse_args(argv)


def load_checkpoint_config(checkpoint: dict, overrides: argparse.Namespace) -> Config:
    raw_cfg: dict = dict(checkpoint.get("config", {}))
    valid_fields = {field.name for field in fields(Config)}
    cfg_kwargs = {key: value for key, value in raw_cfg.items() if key in valid_fields}
    if overrides.batch_size > 0:
        cfg_kwargs["batch_size"] = overrides.batch_size
    if overrides.eval_surface_points > 0:
        cfg_kwargs["eval_surface_points"] = overrides.eval_surface_points
    if overrides.eval_volume_points > 0:
        cfg_kwargs["eval_volume_points"] = overrides.eval_volume_points
    cfg_kwargs["manifest"] = overrides.manifest
    cfg_kwargs["data_root"] = overrides.data_root or cfg_kwargs.get("data_root", "")
    cfg_kwargs["agent"] = overrides.agent or cfg_kwargs.get("agent", "")
    cfg_kwargs["wandb_group"] = overrides.wandb_group
    cfg_kwargs["wandb_name"] = overrides.wandb_name
    # Force-off any train-time-only switches that exist on the current Config.
    cfg_kwargs["compile_model"] = False
    cfg_kwargs["debug"] = False
    return Config(**cfg_kwargs)


def mirror_batch_inputs(batch: SurfaceBatch) -> SurfaceBatch:
    """Return a copy of ``batch`` with input geometry mirrored about y=0.

    Negates the y components of surface positions, surface normals, and volume
    positions. Targets and masks are unchanged.
    """

    surface_x = batch.surface_x.clone()
    surface_x[..., 1] *= -1.0
    surface_x[..., 4] *= -1.0
    volume_x = batch.volume_x.clone()
    volume_x[..., 1] *= -1.0
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


def unmirror_surface_preds(preds: torch.Tensor) -> torch.Tensor:
    """Flip the tau_y channel sign so a mirrored forward becomes comparable.

    surface_preds layout is ``[cp, tau_x, tau_y, tau_z]``; only tau_y flips
    sign under y=0 reflection. Volume pressure is scalar invariant.
    """

    out = preds.clone()
    out[..., 2] *= -1.0
    return out


def _accumulate_with_preds(
    accumulator: EvalAccumulator,
    *,
    batch: SurfaceBatch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> None:
    """Accumulate one mode's predictions into ``accumulator``.

    This is the bookkeeping half of ``accumulate_eval_batch`` from
    ``trainer_runtime``, factored out so we can run two forwards (original,
    mirrored) and feed three accumulators (A, B, TTA) without re-forwarding.
    """

    from trainer_runtime import _accumulate_case_rel_l2, _masked_sse_count

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
            accumulator.abs_sums[f"wall_shear_{axis}"] += float(channel.sum().detach().cpu().item())
            accumulator.abs_counts[f"wall_shear_{axis}"] += int(channel.numel())
        wall_vector_error = torch.linalg.vector_norm(
            surface_pred[batch.surface_mask][:, 1:4] - batch.surface_y[batch.surface_mask][:, 1:4],
            dim=-1,
        )
        accumulator.wall_shear_vector_abs_sum += float(wall_vector_error.sum().detach().cpu().item())
        accumulator.wall_shear_vector_count += int(wall_vector_error.numel())

    if bool(batch.volume_mask.any()):
        volume_abs = (volume_pred - batch.volume_y).abs()[batch.volume_mask]
        accumulator.abs_sums["volume_pressure"] += float(volume_abs[:, 0].sum().detach().cpu().item())
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
def evaluate_split_tta(
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    *,
    amp_mode: str,
    distributed_state: DistributedState | None,
) -> dict[str, dict[str, float]]:
    """Run two forwards (original + mirrored) per batch and finalize 3 modes.

    Returns a dict with keys ``original``, ``mirrored``, ``tta`` mapping to the
    same metric dictionary that ``finalize_eval_accumulator`` produces.
    """

    model.eval()
    acc_original = EvalAccumulator()
    acc_mirrored = EvalAccumulator()
    acc_tta = EvalAccumulator()
    eval_module = unwrap_model(model)
    for batch in loader:
        batch = batch.to(device)
        with autocast_context(device, amp_mode):
            out_a = eval_module(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
        surface_pred_a = out_a["surface_preds"].float()
        volume_pred_a = out_a["volume_preds"].float()

        mirrored_batch = mirror_batch_inputs(batch)
        with autocast_context(device, amp_mode):
            out_b = eval_module(
                surface_x=mirrored_batch.surface_x,
                surface_mask=mirrored_batch.surface_mask,
                volume_x=mirrored_batch.volume_x,
                volume_mask=mirrored_batch.volume_mask,
            )
        # Un-mirror the model's mirrored-input prediction so its outputs are in
        # the same frame as the original-input predictions and the targets.
        surface_pred_b = unmirror_surface_preds(out_b["surface_preds"].float())
        volume_pred_b = out_b["volume_preds"].float()

        surface_pred_tta = 0.5 * (surface_pred_a + surface_pred_b)
        volume_pred_tta = 0.5 * (volume_pred_a + volume_pred_b)

        _accumulate_with_preds(
            acc_original,
            batch=batch,
            surface_pred_norm=surface_pred_a,
            volume_pred_norm=volume_pred_a,
            transform=transform,
        )
        _accumulate_with_preds(
            acc_mirrored,
            batch=batch,
            surface_pred_norm=surface_pred_b,
            volume_pred_norm=volume_pred_b,
            transform=transform,
        )
        _accumulate_with_preds(
            acc_tta,
            batch=batch,
            surface_pred_norm=surface_pred_tta,
            volume_pred_norm=volume_pred_tta,
            transform=transform,
        )

    if distributed_state is not None and distributed_state.enabled:
        gathered_original: list[EvalAccumulator | None] = [None] * distributed_state.world_size
        gathered_mirrored: list[EvalAccumulator | None] = [None] * distributed_state.world_size
        gathered_tta: list[EvalAccumulator | None] = [None] * distributed_state.world_size
        dist.all_gather_object(gathered_original, acc_original)
        dist.all_gather_object(gathered_mirrored, acc_mirrored)
        dist.all_gather_object(gathered_tta, acc_tta)
        if not distributed_state.is_main:
            return {}
        acc_original = merge_eval_accumulators(a for a in gathered_original if a is not None)
        acc_mirrored = merge_eval_accumulators(a for a in gathered_mirrored if a is not None)
        acc_tta = merge_eval_accumulators(a for a in gathered_tta if a is not None)

    return {
        "original": finalize_eval_accumulator(acc_original),
        "mirrored": finalize_eval_accumulator(acc_mirrored),
        "tta": finalize_eval_accumulator(acc_tta),
    }


def _init_wandb_for_tta(*, config: Config, state: DistributedState, n_params: int) -> object | None:
    if not state.is_main:
        return None
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        group=config.wandb_group or None,
        name=config.wandb_name or None,
        tags=[config.agent or "nezuko", "h211", "tta", "eval-only"],
        config={
            **asdict(config),
            "n_params": n_params,
            "tta/modes": ["original", "mirrored", "tta"],
            "tta/source_run_id": "9f2jtrg2",
            "tta/source_artifact": "model-nezuko-h190-mirror-aug-p025-9f2jtrg2:best",
            "tta/source_epoch": 13,
            "ddp_enabled": state.enabled,
            "ddp_world_size": state.world_size,
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    define_wandb_metrics()
    return run


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    run = None
    try:
        args = parse_tta_args(argv)
        device = state.device
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = load_checkpoint_config(checkpoint, args)
        if state.is_main:
            print(f"Device: {device} world_size={state.world_size}")
            print(f"Checkpoint: {args.checkpoint}")
            print(f"  source epoch={checkpoint.get('epoch')} source={checkpoint.get('checkpoint_source')}")
            print(f"  model_hidden_dim={config.model_hidden_dim} layers={config.model_layers}")
            print(f"  eval_surface_points={config.eval_surface_points} eval_volume_points={config.eval_volume_points}")
            print(f"  batch_size={config.batch_size}")

        # Force eval-sized volume points across the train loader so make_loaders'
        # asserts are happy without rebuilding any train state we won't use.
        config.train_volume_points = config.eval_volume_points
        train_loader, val_loaders, test_loaders, stats = make_loaders(config, distributed_state=state)
        # Free the unused train loader's workers ASAP.
        del train_loader

        transform = TargetTransform(
            surface_y_mean=stats["surface_y_mean"].to(device),
            surface_y_std=stats["surface_y_std"].to(device),
            volume_y_mean=stats["volume_y_mean"].to(device),
            volume_y_std=stats["volume_y_std"].to(device),
        )

        model: nn.Module = build_model(config).to(device)
        n_params = sum(param.numel() for param in model.parameters())
        # State dict was saved from ``base_model.state_dict()`` so keys are clean.
        model_state = checkpoint["model"]
        model.load_state_dict(model_state)
        if state.is_main:
            print(f"Loaded checkpoint state_dict ({n_params / 1e6:.2f}M params)")
        if state.enabled:
            ddp_kwargs = {}
            if device.type == "cuda":
                ddp_kwargs = {"device_ids": [state.local_rank], "output_device": state.local_rank}
            if config.use_surf_to_vol_xattn:
                ddp_kwargs["find_unused_parameters"] = True
            model = DistributedDataParallel(model, **ddp_kwargs)

        run = _init_wandb_for_tta(config=config, state=state, n_params=n_params)

        # DDP-sharded full evaluation: StridedDistributedSampler covers each
        # view exactly once across ranks, so the gathered accumulator matches
        # a single-rank full sweep up to float-associativity noise.
        t_val = time.time()
        val_modes = evaluate_split_tta(
            model,
            val_loaders["val_surface"],
            transform,
            device,
            amp_mode=config.amp_mode,
            distributed_state=state,
        )
        if state.is_main:
            val_seconds = time.time() - t_val
            print(f"Full val (DDP-sharded) done in {val_seconds:.1f}s")

        t_test = time.time()
        test_modes = evaluate_split_tta(
            model,
            test_loaders["test_surface"],
            transform,
            device,
            amp_mode=config.amp_mode,
            distributed_state=state,
        )
        if state.is_main:
            test_seconds = time.time() - t_test
            print(f"Full test (DDP-sharded) done in {test_seconds:.1f}s")

        if state.is_main:
            log_payload: dict[str, object] = {
                "global_step": 0,
                "tta/val_seconds": val_seconds,
                "tta/test_seconds": test_seconds,
            }
            for mode_name, val_metrics in val_modes.items():
                # Mirror tay's full-val keys so this run can be compared
                # directly against H190's terminal eval in W&B.
                log_payload.update(
                    primary_metric_log(f"full_val_primary/{mode_name}", val_metrics)
                )
                log_payload.update(
                    metric_namespace(f"full_val/{mode_name}", "val_surface", val_metrics)
                )
                print(f"\n== full_val :: mode={mode_name} ==")
                print_metrics(f"full_val_{mode_name}", val_metrics)
            for mode_name, t_metrics in test_modes.items():
                log_payload.update(primary_metric_log(f"test_primary/{mode_name}", t_metrics))
                log_payload.update(metric_namespace(f"test/{mode_name}", "test_surface", t_metrics))
                print(f"\n== test :: mode={mode_name} ==")
                print_metrics(f"test_{mode_name}", t_metrics)

            wandb.log(log_payload)
            wandb.summary.update({k: v for k, v in log_payload.items() if isinstance(v, (int, float))})

            # Mode deltas — handy for the PR result comment.
            def _delta(mode_metrics: dict[str, float], base_metrics: dict[str, float], key: str) -> float:
                return float(mode_metrics[key]) - float(base_metrics[key])

            base = test_modes["original"]
            for mode in ("mirrored", "tta"):
                diffs = {
                    f"tta/delta_test_{mode}/abupt_axis_mean_rel_l2_pct": _delta(
                        test_modes[mode], base, "abupt_axis_mean_rel_l2_pct"
                    ),
                    f"tta/delta_test_{mode}/wall_shear_rel_l2_pct": _delta(
                        test_modes[mode], base, "wall_shear_rel_l2_pct"
                    ),
                    f"tta/delta_test_{mode}/wall_shear_x_rel_l2_pct": _delta(
                        test_modes[mode], base, "wall_shear_x_rel_l2_pct"
                    ),
                    f"tta/delta_test_{mode}/wall_shear_y_rel_l2_pct": _delta(
                        test_modes[mode], base, "wall_shear_y_rel_l2_pct"
                    ),
                    f"tta/delta_test_{mode}/wall_shear_z_rel_l2_pct": _delta(
                        test_modes[mode], base, "wall_shear_z_rel_l2_pct"
                    ),
                    f"tta/delta_test_{mode}/volume_pressure_rel_l2_pct": _delta(
                        test_modes[mode], base, "volume_pressure_rel_l2_pct"
                    ),
                    f"tta/delta_test_{mode}/surface_pressure_rel_l2_pct": _delta(
                        test_modes[mode], base, "surface_pressure_rel_l2_pct"
                    ),
                }
                wandb.log(diffs)
                wandb.summary.update(diffs)
                print(f"\n== delta test {mode} vs original ==")
                for key, value in diffs.items():
                    print(f"  {key} = {value:+.4f}")

        distributed_barrier(state)
    finally:
        if run is not None:
            wandb.finish()
        cleanup_distributed(state)


if __name__ == "__main__":
    main()
