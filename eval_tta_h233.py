# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H233: Point-order permutation TTA eval on H185 EP13 checkpoint (eval-only, no training).

Tests whether Transolver's slice-attention has residual point-order dependency.
Runs three eval modes on val + test splits for a single checkpoint:
    * original      : forward pass on the original geometry (sanity check / baseline)
    * permute_only  : 4 random point-order permutations, average in normalized space
    * mirror_permute: orig + mirror(y) + 2 random permutations, average in normalized space

Permutation convention:
    surface: generate random permutation pi over N_surf dimension;
             apply same pi to surface_x AND surface_mask (so padding alignment is preserved).
             Outputs come back in permuted spatial order — un-permute via argsort(pi).
    volume:  generate independent random permutation sigma over N_vol dimension;
             apply sigma to volume_x AND volume_mask.
             Outputs come back in permuted order — un-permute via argsort(sigma).
    surface_y targets are NOT permuted (ground-truth is fixed).

If the model is truly permutation-invariant (as slice-attention theoretically is),
all permuted passes are numerically identical → 0bp gain.
If there is residual order dependency (e.g., from bf16 parallel-sum order,
positional encoding interaction, or GPU non-determinism), averaging multiple
independent permutations reduces variance → small positive gain.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml

from data import SurfaceBatch
from model import SurfaceTransolver
from trainer_runtime import (
    EVAL_KEYS,
    EvalAccumulator,
    TargetTransform,
    _accumulate_case_rel_l2,
    _masked_sse_count,
    autocast_context,
    cleanup_distributed,
    eval_loader_for_dataset,
    finalize_eval_accumulator,
    init_distributed,
    loader_kwargs,
    make_loaders,
    merge_eval_accumulators,
    primary_metric_log,
    print_metrics,
    unwrap_model,
)


@dataclass
class EvalConfig:
    """Minimal config mirroring train.Config — only fields needed for eval.

    Defaults match the H185 training-time config so the reconstructed model
    matches the checkpoint exactly.
    """

    checkpoint: str = "outputs/h185/checkpoint.pt"
    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h233_eval"
    wandb_group: str = "h233-edward-point-permute-tta"
    wandb_name: str = ""
    agent: str = "edward"

    # Modes to evaluate (comma-separated string parsed at runtime)
    eval_modes: str = "original,permute_only,mirror_permute"
    num_permute_passes: int = 4

    # Eval-only loader params (no train sampling)
    batch_size: int = 2
    eval_surface_points: int = 65536
    eval_volume_points: int = 65536
    train_surface_points: int = 65536  # only used by make_loaders; unused by us
    train_volume_points: int = 65536
    num_workers: int = -1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Model arch (must match H185 yw2a5dyl exactly)
    model_layers: int = 5
    model_hidden_dim: int = 512
    model_heads: int = 4
    model_mlp_ratio: int = 4
    model_slices: int = 128
    model_dropout: float = 0.0
    rff_num_features: int = 16
    rff_sigma: float = 1.0
    rff_init_sigmas: str = "0.25,0.5,1.0,2.0,4.0"
    pos_encoding_mode: str = "string_separable"
    use_qk_norm: bool = True
    use_surf_to_vol_xattn: bool = True
    drop_path_max: float = 0.1

    amp_mode: str = "bf16"
    debug: bool = False


def parse_args(argv: Iterable[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="H233 point-order permutation TTA eval")
    defaults = EvalConfig()
    for f in fields(EvalConfig):
        v = getattr(defaults, f.name)
        cli = f"--{f.name.replace('_', '-')}"
        if isinstance(v, bool):
            parser.add_argument(cli, action="store_true", default=v, dest=f.name)
            parser.add_argument(
                f"--no-{f.name.replace('_', '-')}",
                action="store_false",
                dest=f.name,
            )
        else:
            parser.add_argument(cli, type=type(v), default=v)
    ns = parser.parse_args(argv)
    cfg = EvalConfig(**{f.name: getattr(ns, f.name) for f in fields(EvalConfig)})
    return cfg


def parse_rff_init_sigmas(spec: str) -> list[float] | None:
    if not spec:
        return None
    return [float(x) for x in spec.split(",") if x.strip()]


def build_model(cfg: EvalConfig) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=cfg.model_layers,
        n_hidden=cfg.model_hidden_dim,
        dropout=cfg.model_dropout,
        n_head=cfg.model_heads,
        mlp_ratio=cfg.model_mlp_ratio,
        slice_num=cfg.model_slices,
        rff_num_features=cfg.rff_num_features,
        rff_sigma=cfg.rff_sigma,
        rff_init_sigmas=parse_rff_init_sigmas(cfg.rff_init_sigmas),
        pos_encoding_mode=cfg.pos_encoding_mode,
        use_qk_norm=cfg.use_qk_norm,
        use_surf_to_vol_xattn=cfg.use_surf_to_vol_xattn,
        drop_path_max=cfg.drop_path_max,
    )


# ---------------------------------------------------------------------------
# Permutation helpers
# ---------------------------------------------------------------------------


def permute_inputs(
    batch: SurfaceBatch,
    surf_perm: torch.Tensor,
    vol_perm: torch.Tensor,
) -> SurfaceBatch:
    """Reorder point clouds along the spatial (N) dimension.

    surf_perm: LongTensor of shape [N_surf] — random permutation of surface indices.
    vol_perm:  LongTensor of shape [N_vol]  — random permutation of volume indices.

    Applies the same permutation to both inputs (surface_x) and padding masks
    (surface_mask) so that padded positions remain correctly labelled after
    reordering. Targets (surface_y, volume_y) are NOT permuted — they stay in
    the original spatial order; outputs are un-permuted back after inference.
    """
    surface_x = batch.surface_x[:, surf_perm, :]
    surface_mask = batch.surface_mask[:, surf_perm]
    volume_x = batch.volume_x[:, vol_perm, :]
    volume_mask = batch.volume_mask[:, vol_perm]
    return SurfaceBatch(
        case_ids=batch.case_ids,
        surface_x=surface_x,
        surface_y=batch.surface_y,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_y=batch.volume_y,
        volume_mask=volume_mask,
        metadata=batch.metadata,
    )


def unpermute_surface_pred(pred: torch.Tensor, surf_perm: torch.Tensor) -> torch.Tensor:
    """Invert the surface permutation on model outputs.

    Model outputs are in permuted spatial order (matching the permuted surface_x
    that was fed in). We restore the original order so predictions align with the
    fixed ground-truth targets in batch.surface_y.

    inv_perm = argsort(surf_perm): position i in the output should go to
    inv_perm[i] in the restored tensor.
    """
    inv_perm = torch.argsort(surf_perm)
    return pred[:, inv_perm, :]


def unpermute_volume_pred(pred: torch.Tensor, vol_perm: torch.Tensor) -> torch.Tensor:
    """Invert the volume permutation on model outputs."""
    inv_perm = torch.argsort(vol_perm)
    return pred[:, inv_perm, :]


# ---------------------------------------------------------------------------
# Mirror helpers (H148/H183 convention — reused from H209)
# ---------------------------------------------------------------------------


def mirror_inputs(batch: SurfaceBatch) -> SurfaceBatch:
    """Negate y/normal_y in surface_x, y in volume_x. Masks and targets unchanged."""
    surface_x = batch.surface_x.clone()
    surface_x[..., 1].neg_()  # y
    surface_x[..., 4].neg_()  # normal_y
    volume_x = batch.volume_x.clone()
    volume_x[..., 1].neg_()  # y
    return SurfaceBatch(
        case_ids=batch.case_ids,
        surface_x=surface_x,
        surface_y=batch.surface_y,
        surface_mask=batch.surface_mask,
        volume_x=volume_x,
        volume_y=batch.volume_y,
        volume_mask=batch.volume_mask,
        metadata=batch.metadata,
    )


def unmirror_surface_pred(pred: torch.Tensor) -> torch.Tensor:
    """surface_pred channels [cp, tau_x, tau_y, tau_z]: un-mirror = negate tau_y."""
    out = pred.clone()
    out[..., 2].neg_()
    return out


def unmirror_volume_pred(pred: torch.Tensor) -> torch.Tensor:
    """volume_pressure is invariant under y-mirror — return as-is."""
    return pred


# ---------------------------------------------------------------------------
# Standard eval accumulation (copied verbatim from H209 — do not modify)
# ---------------------------------------------------------------------------


def _accumulate_outputs(
    acc: EvalAccumulator,
    batch: SurfaceBatch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> None:
    """Same logic as accumulate_eval_batch but takes precomputed predictions.

    Inputs are normalized predictions; we denormalize them here for the MAE / relL2
    accumulators exactly like `accumulate_eval_batch` does.
    """
    surface_target_norm = transform.apply_surface(batch.surface_y)
    volume_target_norm = transform.apply_volume(batch.volume_y)

    surface_sse, surface_count = _masked_sse_count(
        surface_pred_norm, surface_target_norm, batch.surface_mask
    )
    volume_sse, volume_count = _masked_sse_count(
        volume_pred_norm, volume_target_norm, batch.volume_mask
    )
    acc.surface_loss_sse += surface_sse
    acc.surface_loss_count += surface_count
    acc.volume_loss_sse += volume_sse
    acc.volume_loss_count += volume_count

    surface_pred = transform.invert_surface(surface_pred_norm)
    volume_pred = transform.invert_volume(volume_pred_norm)

    if bool(batch.surface_mask.any()):
        surface_abs = (surface_pred - batch.surface_y).abs()
        valid_surface_abs = surface_abs[batch.surface_mask]
        acc.abs_sums["surface_pressure"] += float(
            valid_surface_abs[:, 0].sum().detach().cpu().item()
        )
        acc.abs_counts["surface_pressure"] += int(valid_surface_abs[:, 0].numel())
        wall_abs = valid_surface_abs[:, 1:4]
        acc.abs_sums["wall_shear"] += float(wall_abs.sum().detach().cpu().item())
        acc.abs_counts["wall_shear"] += int(wall_abs.numel())
        for offset, axis in enumerate(("x", "y", "z")):
            channel = wall_abs[:, offset]
            acc.abs_sums[f"wall_shear_{axis}"] += float(channel.sum().detach().cpu().item())
            acc.abs_counts[f"wall_shear_{axis}"] += int(channel.numel())
        wall_vector_error = torch.linalg.vector_norm(
            surface_pred[batch.surface_mask][:, 1:4]
            - batch.surface_y[batch.surface_mask][:, 1:4],
            dim=-1,
        )
        acc.wall_shear_vector_abs_sum += float(
            wall_vector_error.sum().detach().cpu().item()
        )
        acc.wall_shear_vector_count += int(wall_vector_error.numel())

    if bool(batch.volume_mask.any()):
        volume_abs = (volume_pred - batch.volume_y).abs()[batch.volume_mask]
        acc.abs_sums["volume_pressure"] += float(
            volume_abs[:, 0].sum().detach().cpu().item()
        )
        acc.abs_counts["volume_pressure"] += int(volume_abs[:, 0].numel())

    for case_idx, case_id in enumerate(batch.case_ids):
        surface_valid = batch.surface_mask[case_idx].bool()
        if bool(surface_valid.any()):
            surface_pred_valid = surface_pred[case_idx][surface_valid]
            surface_target_valid = batch.surface_y[case_idx][surface_valid]
            _accumulate_case_rel_l2(
                acc.case_sums["surface_pressure"],
                case_id=case_id,
                pred=surface_pred_valid[:, 0:1],
                target=surface_target_valid[:, 0:1],
            )
            _accumulate_case_rel_l2(
                acc.case_sums["wall_shear"],
                case_id=case_id,
                pred=surface_pred_valid[:, 1:4],
                target=surface_target_valid[:, 1:4],
            )
            for channel, axis in enumerate(("x", "y", "z"), start=1):
                _accumulate_case_rel_l2(
                    acc.case_sums[f"wall_shear_{axis}"],
                    case_id=case_id,
                    pred=surface_pred_valid[:, channel : channel + 1],
                    target=surface_target_valid[:, channel : channel + 1],
                )
        volume_valid = batch.volume_mask[case_idx].bool()
        if bool(volume_valid.any()):
            _accumulate_case_rel_l2(
                acc.case_sums["volume_pressure"],
                case_id=case_id,
                pred=volume_pred[case_idx][volume_valid],
                target=batch.volume_y[case_idx][volume_valid],
            )


# ---------------------------------------------------------------------------
# DDP gather helper
# ---------------------------------------------------------------------------


def _finalize_acc(acc: EvalAccumulator, distributed_state) -> dict[str, float]:
    """Gather and finalize an accumulator across DDP ranks."""
    if distributed_state is not None and distributed_state.enabled:
        gathered = [None for _ in range(distributed_state.world_size)]
        dist.all_gather_object(gathered, acc)
        if distributed_state.is_main:
            merged = merge_eval_accumulators(g for g in gathered if g is not None)
            return finalize_eval_accumulator(merged)
        else:
            return {}
    else:
        return finalize_eval_accumulator(acc)


# ---------------------------------------------------------------------------
# Per-mode evaluation functions
# ---------------------------------------------------------------------------


def evaluate_original(
    *,
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    distributed_state,
) -> dict[str, float]:
    """Single forward pass on the original (un-augmented) geometry."""
    acc = EvalAccumulator()
    eval_module = unwrap_model(model)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with autocast_context(device, amp_mode):
                out = eval_module(
                    surface_x=batch.surface_x,
                    surface_mask=batch.surface_mask,
                    volume_x=batch.volume_x,
                    volume_mask=batch.volume_mask,
                )
            surface_pred = out["surface_preds"].float()
            volume_pred = out["volume_preds"].float()
            _accumulate_outputs(acc, batch, surface_pred, volume_pred, transform)

    return _finalize_acc(acc, distributed_state)


def evaluate_permute_only(
    *,
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    distributed_state,
    num_passes: int,
) -> dict[str, float]:
    """Average of `num_passes` random point-order permutations (equal weights 1/num_passes).

    For each batch:
      1. Draw independent random pi (surface) and sigma (volume) for each pass.
      2. Permute inputs, run forward, un-permute outputs.
      3. Average all num_passes predictions in normalized space.
      4. Accumulate the average against the original (un-permuted) targets.

    Permutations are drawn fresh per batch on the local rank. All ranks use
    different random seeds (torch default), which is fine — each rank processes
    different batches and the accumulator is gathered at the end.
    """
    acc = EvalAccumulator()
    eval_module = unwrap_model(model)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            N_surf = batch.surface_x.shape[1]
            N_vol = batch.volume_x.shape[1]

            surface_pred_sum = None
            volume_pred_sum = None

            for _ in range(num_passes):
                surf_perm = torch.randperm(N_surf, device=device)
                vol_perm = torch.randperm(N_vol, device=device)

                permuted = permute_inputs(batch, surf_perm, vol_perm)

                with autocast_context(device, amp_mode):
                    out = eval_module(
                        surface_x=permuted.surface_x,
                        surface_mask=permuted.surface_mask,
                        volume_x=permuted.volume_x,
                        volume_mask=permuted.volume_mask,
                    )

                sp = out["surface_preds"].float()
                vp = out["volume_preds"].float()

                # Un-permute outputs back to original spatial order
                sp = unpermute_surface_pred(sp, surf_perm)
                vp = unpermute_volume_pred(vp, vol_perm)

                if surface_pred_sum is None:
                    surface_pred_sum = sp
                    volume_pred_sum = vp
                else:
                    surface_pred_sum = surface_pred_sum + sp
                    volume_pred_sum = volume_pred_sum + vp

            weight = 1.0 / num_passes
            surface_pred_avg = surface_pred_sum * weight
            volume_pred_avg = volume_pred_sum * weight

            _accumulate_outputs(acc, batch, surface_pred_avg, volume_pred_avg, transform)

    return _finalize_acc(acc, distributed_state)


def evaluate_mirror_permute(
    *,
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    distributed_state,
    num_permute_passes: int,
) -> dict[str, float]:
    """Combine mirror-y with random permutations: orig + mirror + (num_permute_passes - 2) permutes.

    Total passes = 2 + (num_permute_passes - 2) = num_permute_passes (= 4 by default).
    All passes are equally weighted at 1/num_permute_passes.

    With num_permute_passes=4:
      Pass 0: original geometry
      Pass 1: mirror-y geometry (un-mirrored output)
      Pass 2: random permutation
      Pass 3: random permutation

    This stacks the validated mirror-y symmetry (Finding H209) with permutation noise
    averaging. If the model is perm-invariant, the two permute passes contribute the
    same prediction as the original, and the result degrades toward mirror-only TTA
    (which is already the SOTA baseline). If the model has residual order dependency,
    the permute passes add independent draws → improved averaging.
    """
    acc = EvalAccumulator()
    eval_module = unwrap_model(model)
    model.eval()

    # Number of extra permutation passes beyond orig + mirror
    n_extra_permute = max(0, num_permute_passes - 2)
    total_passes = 2 + n_extra_permute  # orig + mirror + n_extra_permute

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            N_surf = batch.surface_x.shape[1]
            N_vol = batch.volume_x.shape[1]

            # Pass 0: original
            with autocast_context(device, amp_mode):
                out_orig = eval_module(
                    surface_x=batch.surface_x,
                    surface_mask=batch.surface_mask,
                    volume_x=batch.volume_x,
                    volume_mask=batch.volume_mask,
                )
            sp_sum = out_orig["surface_preds"].float()
            vp_sum = out_orig["volume_preds"].float()

            # Pass 1: mirror-y
            mirrored = mirror_inputs(batch)
            with autocast_context(device, amp_mode):
                out_mirror = eval_module(
                    surface_x=mirrored.surface_x,
                    surface_mask=mirrored.surface_mask,
                    volume_x=mirrored.volume_x,
                    volume_mask=mirrored.volume_mask,
                )
            sp_mirror = unmirror_surface_pred(out_mirror["surface_preds"].float())
            vp_mirror = unmirror_volume_pred(out_mirror["volume_preds"].float())
            sp_sum = sp_sum + sp_mirror
            vp_sum = vp_sum + vp_mirror

            # Passes 2..(total_passes-1): random permutations
            for _ in range(n_extra_permute):
                surf_perm = torch.randperm(N_surf, device=device)
                vol_perm = torch.randperm(N_vol, device=device)
                permuted = permute_inputs(batch, surf_perm, vol_perm)

                with autocast_context(device, amp_mode):
                    out_perm = eval_module(
                        surface_x=permuted.surface_x,
                        surface_mask=permuted.surface_mask,
                        volume_x=permuted.volume_x,
                        volume_mask=permuted.volume_mask,
                    )
                sp_perm = unpermute_surface_pred(out_perm["surface_preds"].float(), surf_perm)
                vp_perm = unpermute_volume_pred(out_perm["volume_preds"].float(), vol_perm)
                sp_sum = sp_sum + sp_perm
                vp_sum = vp_sum + vp_perm

            weight = 1.0 / total_passes
            _accumulate_outputs(acc, batch, sp_sum * weight, vp_sum * weight, transform)

    return _finalize_acc(acc, distributed_state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device

    # Parse requested modes
    requested_modes = [m.strip() for m in cfg.eval_modes.split(",") if m.strip()]
    valid_modes = {"original", "permute_only", "mirror_permute"}
    for m in requested_modes:
        if m not in valid_modes:
            raise ValueError(f"Unknown eval mode '{m}'. Valid: {sorted(valid_modes)}")

    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        print(f"Checkpoint: {cfg.checkpoint}")
        print(f"Modes: {requested_modes}")
        print(f"num_permute_passes: {cfg.num_permute_passes}")

    train_loader, val_loaders, test_loaders, stats = make_loaders(cfg, distributed_state=state)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    model = build_model(cfg).to(device)
    ck = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ck["model"]
    # Strip any DDP prefix if present
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if state.is_main:
        print(f"Loaded checkpoint epoch={ck.get('epoch')} source={ck.get('checkpoint_source')}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  missing[:5]={missing[:5]}")
        if unexpected:
            print(f"  unexpected[:5]={unexpected[:5]}")
    model.eval()

    # WandB init (rank 0 only)
    run = None
    if state.is_main:
        run_name = cfg.wandb_name or f"{cfg.agent}/h233-point-permute-tta"
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            group=cfg.wandb_group,
            name=run_name,
            config={
                **asdict(cfg),
                "checkpoint_run_id": "yw2a5dyl",
                "checkpoint_epoch": ck.get("epoch"),
                "eval_modes_list": requested_modes,
            },
            tags=["h233", "tta", "permute", "eval-only", cfg.agent],
            reinit="finish_previous",
        )

    splits = [
        ("val_surface", val_loaders["val_surface"], "full_val"),
        ("test_surface", test_loaders["test_surface"], "test"),
    ]

    # summary[split_name][mode_name] = metrics_dict
    summary: dict[str, dict[str, dict[str, float]]] = {}

    for split_name, loader, log_prefix in splits:
        if state.is_main:
            print(f"\n=== Evaluating split={split_name} ===")
        summary[split_name] = {}

        for mode in requested_modes:
            if state.is_main:
                print(f"  -- mode={mode} --")
            t0 = time.time()

            if mode == "original":
                metrics = evaluate_original(
                    model=model,
                    loader=loader,
                    transform=transform,
                    device=device,
                    amp_mode=cfg.amp_mode,
                    distributed_state=state,
                )
            elif mode == "permute_only":
                metrics = evaluate_permute_only(
                    model=model,
                    loader=loader,
                    transform=transform,
                    device=device,
                    amp_mode=cfg.amp_mode,
                    distributed_state=state,
                    num_passes=cfg.num_permute_passes,
                )
            elif mode == "mirror_permute":
                metrics = evaluate_mirror_permute(
                    model=model,
                    loader=loader,
                    transform=transform,
                    device=device,
                    amp_mode=cfg.amp_mode,
                    distributed_state=state,
                    num_permute_passes=cfg.num_permute_passes,
                )
            else:
                raise RuntimeError(f"Unhandled mode: {mode}")

            dt = time.time() - t0
            if state.is_main:
                print(f"    done in {dt:.1f}s")
                print_metrics(split_name, metrics)
                summary[split_name][mode] = metrics

                log_obj: dict[str, float] = {}
                log_obj.update(primary_metric_log(f"{log_prefix}_primary/{mode}", metrics))
                log_obj.update({f"{log_prefix}_extra/{mode}/loss": metrics.get("loss", float("nan"))})
                if run is not None:
                    wandb.log(log_obj)

    if state.is_main and summary:
        # Compact comparison table
        keys = (
            "abupt_axis_mean_rel_l2_pct",
            "surface_pressure_rel_l2_pct",
            "wall_shear_rel_l2_pct",
            "wall_shear_x_rel_l2_pct",
            "wall_shear_y_rel_l2_pct",
            "wall_shear_z_rel_l2_pct",
            "volume_pressure_rel_l2_pct",
        )

        print("\n=== Summary (rel_l2_pct, lower-is-better) ===")
        for split_name, modes in summary.items():
            print(f"\n[{split_name}]")
            mode_names = list(modes.keys())
            header_cols = "".join(f"{m:>18s}" for m in mode_names)
            # If original is present, compute delta vs original
            has_orig = "original" in modes
            if has_orig:
                header_cols += f"  {'vs_original':>14s}"
            print(f"  {'metric':<36s}{header_cols}")
            for k in keys:
                row = f"  {k:<36s}"
                orig_val = modes.get("original", {}).get(k, float("nan")) if has_orig else float("nan")
                for m in mode_names:
                    v = modes[m].get(k, float("nan"))
                    row += f"{v:>18.4f}"
                if has_orig:
                    for m in mode_names:
                        if m != "original":
                            v = modes[m].get(k, float("nan"))
                            d = v - orig_val
                            row += f"  {d:>+14.4f}"
                            break  # only show first non-original delta per row for readability
                print(row)

        # Save to W&B summary
        if run is not None:
            for split_name, modes in summary.items():
                for mode, metrics in modes.items():
                    for k, v in metrics.items():
                        try:
                            run.summary[f"{split_name}/{mode}/{k}"] = float(v)
                        except Exception:
                            pass
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
