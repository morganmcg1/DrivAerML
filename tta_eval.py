# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H206: Test-Time mirror Augmentation (TTA) eval for H183 EP13.

Runs three evaluations per split (val_surface, test_surface):
  A: original geometry
  B: mirrored geometry, predictions un-mirrored back to original frame
  C: TTA averaged predictions ((A + B) / 2 in denormalised space)

Mirror convention (H148 / H183 / fern):
  surface_x cols [x, y, z, nx, ny, nz, area] -> negate y (col 1), ny (col 4)
  surface_y cols [cp, tau_x, tau_y, tau_z]   -> tau_y (col 2) sign-flips
  volume_x  cols [x, y, z, sdf]              -> negate y (col 1); sdf invariant
  volume_y  (volume_pressure)                -> invariant

Therefore at inference, for mode B we un-mirror by negating surface_pred col 2
(tau_y) in DENORMALISED space; volume_pred is invariant.

Single forward pass A + forward pass B per batch; both are accumulated into
three EvalAccumulators (A, B, C). DDP-friendly via the existing StridedSampler
and all_gather_object reduce path used by trainer_runtime.evaluate_split.
"""
from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml

from data import SurfaceBatch
from train import Config, build_model
from trainer_runtime import (
    DistributedState,
    EvalAccumulator,
    TargetTransform,
    _accumulate_case_rel_l2,
    _masked_sse_count,
    autocast_context,
    cleanup_distributed,
    finalize_eval_accumulator,
    init_distributed,
    make_loaders,
    merge_eval_accumulators,
    primary_metric_log,
    unwrap_model,
)


def parse_tta_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTA mirror eval for H183 EP13")
    parser.add_argument("--config-path", required=True, type=Path,
                        help="Path to H183 config.yaml (used to build the model).")
    parser.add_argument("--checkpoint-path", required=True, type=Path,
                        help="Path to H183 checkpoint.pt (EMA, epoch 13).")
    parser.add_argument("--wandb-name", default="alphonse/h206-tta-h183-ep13", type=str)
    parser.add_argument("--wandb-group", default="h206-alphonse-tta-h183", type=str)
    parser.add_argument("--wandb-project", default=os.environ.get(
        "WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"), type=str)
    parser.add_argument("--wandb-entity", default=os.environ.get(
        "WANDB_ENTITY", "wandb-applied-ai-team"), type=str)
    parser.add_argument("--data-root", default=None, type=str,
                        help="Optional data root override (defaults to config).")
    parser.add_argument("--debug", action="store_true",
                        help="Use the debug dataset slice (very small).")
    return parser.parse_args(list(argv) if argv is not None else None)


def load_config_from_yaml(path: Path) -> Config:
    """Build a Config dataclass from a saved W&B config.yaml.

    Ignores W&B-only keys and any fields no longer present on Config.
    """
    with path.open() as f:
        raw = yaml.safe_load(f)
    # W&B's config.yaml sometimes wraps values in {value: ..., desc: ...};
    # the fern artifact stores flat scalars. Handle both.
    flat: dict[str, object] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "value" in v:
            flat[k] = v["value"]
        else:
            flat[k] = v
    known = {f.name for f in fields(Config)}
    kwargs = {k: v for k, v in flat.items() if k in known}
    # Coerce ints where Config declares int but yaml gave float (e.g. rff_sigma).
    return Config(**kwargs)


def mirror_inputs(
    surface_x: torch.Tensor,
    volume_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply left-right mirror to inputs only (targets are kept untouched)."""
    surface_mirror = surface_x.clone()
    surface_mirror[..., 1] = -surface_mirror[..., 1]  # y
    surface_mirror[..., 4] = -surface_mirror[..., 4]  # ny
    volume_mirror = volume_x.clone()
    volume_mirror[..., 1] = -volume_mirror[..., 1]  # y
    return surface_mirror, volume_mirror


def unmirror_surface_pred(surface_pred: torch.Tensor) -> torch.Tensor:
    """Negate tau_y (channel 2) to map mirror-space prediction back to original frame.

    Must operate in DENORMALISED space; in normalised space a non-zero
    surface_y_mean for tau_y would break the simple sign flip.
    """
    out = surface_pred.clone()
    out[..., 2] = -out[..., 2]
    return out


def _accumulate_pred(
    accumulator: EvalAccumulator,
    *,
    batch: SurfaceBatch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    surface_pred: torch.Tensor,
    volume_pred: torch.Tensor,
    surface_target_norm: torch.Tensor,
    volume_target_norm: torch.Tensor,
) -> None:
    """Replicate the accumulation logic of trainer_runtime.accumulate_eval_batch.

    Caller has already produced normalised and denormalised predictions for the
    desired TTA mode (original, mirror, or averaged).
    """
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
def evaluate_tta_split(
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    *,
    amp_mode: str = "none",
    distributed_state: DistributedState | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate one split with 3 TTA modes in a single pass."""
    model.eval()
    acc_a = EvalAccumulator()
    acc_b = EvalAccumulator()
    acc_c = EvalAccumulator()
    eval_module = unwrap_model(model)

    for batch in loader:
        batch = batch.to(device)
        surface_target_norm = transform.apply_surface(batch.surface_y)
        volume_target_norm = transform.apply_volume(batch.volume_y)

        # Pass A: original
        with autocast_context(device, amp_mode):
            out_a = eval_module(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
        surface_a_norm = out_a["surface_preds"].float()
        volume_a_norm = out_a["volume_preds"].float()

        # Pass B: mirrored inputs
        surface_x_mir, volume_x_mir = mirror_inputs(batch.surface_x, batch.volume_x)
        with autocast_context(device, amp_mode):
            out_b = eval_module(
                surface_x=surface_x_mir,
                surface_mask=batch.surface_mask,
                volume_x=volume_x_mir,
                volume_mask=batch.volume_mask,
            )
        surface_b_mir_norm = out_b["surface_preds"].float()
        volume_b_norm = out_b["volume_preds"].float()

        # Denormalise for un-mirroring + averaging.
        surface_a = transform.invert_surface(surface_a_norm)
        surface_b_mir = transform.invert_surface(surface_b_mir_norm)
        volume_a = transform.invert_volume(volume_a_norm)
        volume_b = transform.invert_volume(volume_b_norm)

        # Un-mirror B: negate tau_y; volume_pressure is invariant.
        surface_b = unmirror_surface_pred(surface_b_mir)
        # volume_b is already in the original frame.

        # TTA average in denormalised space.
        surface_c = 0.5 * (surface_a + surface_b)
        volume_c = 0.5 * (volume_a + volume_b)

        # Re-normalise the averaged + un-mirrored variants so the SSE loss
        # accumulator (which lives in normalised space) is consistent with
        # the standard pipeline.
        surface_b_norm = transform.apply_surface(surface_b)
        surface_c_norm = transform.apply_surface(surface_c)
        volume_c_norm = transform.apply_volume(volume_c)

        _accumulate_pred(
            acc_a,
            batch=batch,
            surface_pred_norm=surface_a_norm,
            volume_pred_norm=volume_a_norm,
            surface_pred=surface_a,
            volume_pred=volume_a,
            surface_target_norm=surface_target_norm,
            volume_target_norm=volume_target_norm,
        )
        _accumulate_pred(
            acc_b,
            batch=batch,
            surface_pred_norm=surface_b_norm,
            volume_pred_norm=volume_b_norm,  # invariant
            surface_pred=surface_b,
            volume_pred=volume_b,
            surface_target_norm=surface_target_norm,
            volume_target_norm=volume_target_norm,
        )
        _accumulate_pred(
            acc_c,
            batch=batch,
            surface_pred_norm=surface_c_norm,
            volume_pred_norm=volume_c_norm,
            surface_pred=surface_c,
            volume_pred=volume_c,
            surface_target_norm=surface_target_norm,
            volume_target_norm=volume_target_norm,
        )

    if distributed_state is not None and distributed_state.enabled:
        out: dict[str, dict[str, float]] = {}
        for label, acc in (("A_original", acc_a), ("B_mirror", acc_b), ("C_tta", acc_c)):
            gathered: list[EvalAccumulator | None] = [None] * distributed_state.world_size
            dist.all_gather_object(gathered, acc)
            if not distributed_state.is_main:
                continue
            merged = merge_eval_accumulators(a for a in gathered if a is not None)
            out[label] = finalize_eval_accumulator(merged)
        return out

    return {
        "A_original": finalize_eval_accumulator(acc_a),
        "B_mirror": finalize_eval_accumulator(acc_b),
        "C_tta": finalize_eval_accumulator(acc_c),
    }


def metric_namespace(top: str, split_name: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{top}/{split_name}/{key}": value for key, value in metrics.items()}


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    run = None
    try:
        args = parse_tta_args(argv)
        config = load_config_from_yaml(args.config_path)
        if args.data_root:
            config.data_root = args.data_root
        if args.debug:
            config.debug = True
        # Eval-only run: do not torch.compile and ignore any train-time
        # augmentation flags (mirror_augmentation is not on the current
        # Config dataclass — load_config_from_yaml already filtered it).
        config.compile_model = False

        device = state.device
        if state.is_main:
            print(f"Device: {device}, world_size={state.world_size}")
            print(f"Checkpoint: {args.checkpoint_path}")
            print(f"Mirror-aug TTA — H183 EP13 (config from {args.config_path})")

        if state.is_main:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                group=args.wandb_group,
                config={
                    **{f.name: getattr(config, f.name) for f in fields(Config)},
                    "h206_mode": "tta_mirror",
                    "h206_source_run": "5k58uzqc",
                    "h206_source_epoch": 13,
                    "h206_checkpoint": str(args.checkpoint_path),
                },
            )

        # Build loaders. The make_loaders helper already returns DDP-strided
        # eval loaders (every surface and volume point is scored exactly once
        # across ranks). We do not need full_eval_loaders_from — that variant
        # is only used on rank 0 after training for single-rank summary eval.
        _, val_loaders, test_loaders, stats = make_loaders(config, distributed_state=state)

        transform = TargetTransform(
            surface_y_mean=stats["surface_y_mean"].to(device),
            surface_y_std=stats["surface_y_std"].to(device),
            volume_y_mean=stats["volume_y_mean"].to(device),
            volume_y_std=stats["volume_y_std"].to(device),
        )

        model: nn.Module = build_model(config).to(device)
        # The DDP wrap is intentionally skipped — eval is single-model; we
        # only need DDP rank-sharding through the loader sampler.
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        if "model" not in ckpt:
            raise RuntimeError(f"Checkpoint {args.checkpoint_path} missing 'model' key")
        # The checkpoint's EMA params were saved under 'model' (checkpoint_source='ema').
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
        if state.is_main:
            print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, "
                  f"source={ckpt.get('checkpoint_source')}, "
                  f"selection_metric={ckpt.get('selection_metric')}")

        # If we're DDP, we still need to use eval loaders that shard across ranks.
        # The val_loaders/test_loaders from make_loaders already use the strided sampler.
        # But full_eval_loaders_from removes the sampler for single-rank full eval.
        # For DDP rank>0 we should use the strided loaders directly.

        all_results: dict[str, dict[str, dict[str, float]]] = {}
        for split_label, loader_map in (
            ("val_surface", val_loaders),
            ("test_surface", test_loaders),
        ):
            loader = loader_map[split_label]
            if state.is_main:
                print(f"\nEvaluating split: {split_label} (DDP-strided, world_size={state.world_size})")
            metrics_by_mode = evaluate_tta_split(
                model,
                loader,
                transform,
                device,
                amp_mode=config.amp_mode,
                distributed_state=state,
            )
            if state.is_main:
                all_results[split_label] = metrics_by_mode
                for mode_label, m in metrics_by_mode.items():
                    print(f"  [{split_label}|{mode_label}] "
                          f"abupt={m['abupt_axis_mean_rel_l2_pct']:.4f} "
                          f"SP={m['surface_pressure_rel_l2_pct']:.4f} "
                          f"WSS={m['wall_shear_rel_l2_pct']:.4f} "
                          f"WSS_x={m['wall_shear_x_rel_l2_pct']:.4f} "
                          f"WSS_y={m['wall_shear_y_rel_l2_pct']:.4f} "
                          f"WSS_z={m['wall_shear_z_rel_l2_pct']:.4f} "
                          f"VP={m['volume_pressure_rel_l2_pct']:.4f} "
                          f"cases={int(m['cases'])}")

        if state.is_main:
            # Log full metric grid to W&B.
            log_payload: dict[str, float] = {}
            # Convenience namespaces mirroring the trainer's terminal log.
            #   tta_A_original/<split>/<metric>, tta_B_mirror/..., tta_C_tta/...
            for split_label, modes in all_results.items():
                for mode_label, metrics in modes.items():
                    for key, value in metrics.items():
                        log_payload[f"tta_{mode_label}/{split_label}/{key}"] = value
            # Also push primary-metric shaped keys for mode C (the TTA result),
            # so existing W&B dashboards that key on full_val_primary / test_primary
            # surface the TTA numbers directly.
            log_payload.update(primary_metric_log(
                "full_val_primary", all_results["val_surface"]["C_tta"]))
            log_payload.update(primary_metric_log(
                "test_primary", all_results["test_surface"]["C_tta"]))
            log_payload.update(primary_metric_log(
                "full_val_original", all_results["val_surface"]["A_original"]))
            log_payload.update(primary_metric_log(
                "test_original", all_results["test_surface"]["A_original"]))
            log_payload.update(primary_metric_log(
                "full_val_mirror", all_results["val_surface"]["B_mirror"]))
            log_payload.update(primary_metric_log(
                "test_mirror", all_results["test_surface"]["B_mirror"]))
            wandb.log(log_payload)
            wandb.summary.update({k: v for k, v in log_payload.items()
                                  if isinstance(v, (int, float))})

            # Print a compact comparison table.
            print("\n=== TTA Results — H183 EP13 ===")
            for split in ("val_surface", "test_surface"):
                modes = all_results[split]
                a = modes["A_original"]
                b = modes["B_mirror"]
                c = modes["C_tta"]
                print(f"\n[{split}]")
                print(f"  metric          A=orig      B=mir       C=TTA       (C-A)pp")
                for label, key in (
                    ("abupt          ", "abupt_axis_mean_rel_l2_pct"),
                    ("SP             ", "surface_pressure_rel_l2_pct"),
                    ("WSS (vec)      ", "wall_shear_rel_l2_pct"),
                    ("WSS_x          ", "wall_shear_x_rel_l2_pct"),
                    ("WSS_y          ", "wall_shear_y_rel_l2_pct"),
                    ("WSS_z          ", "wall_shear_z_rel_l2_pct"),
                    ("VP             ", "volume_pressure_rel_l2_pct"),
                ):
                    delta = c[key] - a[key]
                    print(f"  {label} {a[key]:8.4f}  {b[key]:8.4f}  "
                          f"{c[key]:8.4f}  {delta:+7.4f}")

            # SOTA gate evaluation (against PR #1283 H112 baseline).
            tt = all_results["test_surface"]["C_tta"]
            vv = all_results["val_surface"]["C_tta"]
            gate = {
                "test_WSS<=6.752": tt["wall_shear_rel_l2_pct"] <= 6.752,
                "test_VP<=3.421": tt["volume_pressure_rel_l2_pct"] <= 3.421,
                "test_SP<=3.577": tt["surface_pressure_rel_l2_pct"] <= 3.577,
                "val_abupt<6.1358": vv["abupt_axis_mean_rel_l2_pct"] < 6.1358,
            }
            print("\n--- SOTA gate check (mode C / TTA) ---")
            for name, passed in gate.items():
                print(f"  {'PASS' if passed else 'FAIL'}  {name}")
            print(f"  WINNER" if all(gate.values()) else "  NOT a SOTA candidate")
            wandb.summary.update({f"gate/{k}": int(v) for k, v in gate.items()})
            wandb.summary["gate/all_pass"] = int(all(gate.values()))
    finally:
        if run is not None:
            wandb.finish()
        cleanup_distributed(state)


if __name__ == "__main__":
    main()
