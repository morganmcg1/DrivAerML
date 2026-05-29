# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H240: Mesh-subsample TTA on H183 EP13 checkpoint (eval-only, no training).

Evaluates three TTA stacking modes for the H183 EP13 checkpoint in a single
pass through the loader (each batch runs 1 orig + 1 mirror + K subsample forward
passes; modes are different reductions over the same per-pass tensors).

Modes reported (per pass per split):
    * orig                          : single forward pass, no augmentation
    * mirror                        : single y-mirror pass (un-mirrored output)
    * mirror_only                   : 0.5 * (orig + mirror)
    * subsample_only_naive          : 0.5 * orig + 0.5 * (1/K) sum(sub_i)
    * subsample_only_mc             : 0.5 * orig + 0.5 * MC-mean(sub_i)
    * mirror_x_subsample_naive      : 1/(2+K) * (orig + mirror + sum(sub_i))
    * mirror_x_subsample_mc         : MC-mean over {orig, mirror, sub_1..sub_K}

MC-mean addresses a known bias of the masked-subsample strategy:
    The model applies `surface_preds = surface_out(hidden) * surface_mask`, so
    predictions at subsampled-out points are exactly 0. A naive mean of K passes
    therefore underestimates magnitudes by a factor ~keep_rate. MC averaging
    divides each output point by the number of passes in which it was active,
    removing this bias.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass, fields
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb

from data import SurfaceBatch
from model import SurfaceTransolver
from trainer_runtime import (
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
    print_metrics,
    unwrap_model,
)


@dataclass
class EvalConfig:
    """Eval-only config; defaults match the H183 training-time arch."""

    checkpoint: str = ""
    run_id: str = "5k58uzqc"
    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h240_eval"
    wandb_group: str = "h240-fern-mesh-subsample-h183"
    wandb_name: str = ""
    agent: str = "fern"

    # Eval-only loader params
    batch_size: int = 2
    eval_surface_points: int = 65536
    eval_volume_points: int = 65536
    train_surface_points: int = 65536
    train_volume_points: int = 65536
    num_workers: int = -1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Model arch (must match H183 5k58uzqc — same as H185 yw2a5dyl)
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

    # H240 subsample TTA params
    subsample_frac: float = 0.8
    subsample_passes: int = 4
    subsample_seed: int = 0


def parse_args(argv: Iterable[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="H240 mesh-subsample TTA eval on H183 EP13")
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


# --- Mirror TTA helpers (H148/H183 convention; same as eval_tta_h209.py) ---


def mirror_inputs(batch: SurfaceBatch) -> SurfaceBatch:
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
    out = pred.clone()
    out[..., 2].neg_()  # tau_y
    return out


def unmirror_volume_pred(pred: torch.Tensor) -> torch.Tensor:
    return pred


# --- Subsample TTA helpers ---


def make_subsample_mask(
    base_mask: torch.Tensor,
    keep_rate: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Random Bernoulli mask AND'd with the existing pad mask.

    base_mask: [B, N] bool — True for valid (non-pad) tokens.
    Returns:   [B, N] bool — sub-sampled mask; padding stays masked out.
    """
    rand = torch.rand(base_mask.shape, generator=generator, device=base_mask.device)
    keep = rand < keep_rate
    return base_mask & keep


# --- Eval accumulation ---


def _accumulate_outputs(
    acc: EvalAccumulator,
    batch: SurfaceBatch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> None:
    """Same logic as eval_tta_h209._accumulate_outputs: SSE/MAE/relL2."""
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


MODES = (
    "orig",
    "mirror",
    "mirror_only",
    "subsample_only_naive",
    "subsample_only_mc",
    "mirror_x_subsample_naive",
    "mirror_x_subsample_mc",
)


def evaluate_tta_split(
    *,
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    distributed_state,
    subsample_frac: float,
    subsample_passes: int,
    subsample_seed: int,
) -> dict[str, dict[str, float]]:
    """Run all 7 evaluation reductions in one pass through the loader."""
    accs = {m: EvalAccumulator() for m in MODES}

    model.eval()
    eval_module = unwrap_model(model)

    # Per-rank deterministic generator for subsample mask so each rank/seed is reproducible.
    # Mask shape depends on batch, so use a single Generator on the same device as batches.
    sub_gen = torch.Generator(device=device)
    rank = (
        distributed_state.rank if distributed_state is not None and distributed_state.enabled else 0
    )
    sub_gen.manual_seed(int(subsample_seed) * 1000 + rank)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mirrored = mirror_inputs(batch)

            with autocast_context(device, amp_mode):
                out_a = eval_module(
                    surface_x=batch.surface_x,
                    surface_mask=batch.surface_mask,
                    volume_x=batch.volume_x,
                    volume_mask=batch.volume_mask,
                )
                out_b = eval_module(
                    surface_x=mirrored.surface_x,
                    surface_mask=mirrored.surface_mask,
                    volume_x=mirrored.volume_x,
                    volume_mask=mirrored.volume_mask,
                )

            surface_pred_a = out_a["surface_preds"].float()
            volume_pred_a = out_a["volume_preds"].float()

            surface_pred_b = unmirror_surface_pred(out_b["surface_preds"].float())
            volume_pred_b = unmirror_volume_pred(out_b["volume_preds"].float())

            # K subsample passes
            sub_surface_sum = torch.zeros_like(surface_pred_a)
            sub_volume_sum = torch.zeros_like(volume_pred_a)
            sub_surface_count = torch.zeros(
                (*surface_pred_a.shape[:2], 1),
                dtype=surface_pred_a.dtype,
                device=device,
            )
            sub_volume_count = torch.zeros(
                (*volume_pred_a.shape[:2], 1),
                dtype=volume_pred_a.dtype,
                device=device,
            )
            for _ in range(subsample_passes):
                sub_surface_mask = make_subsample_mask(
                    batch.surface_mask, subsample_frac, sub_gen
                )
                sub_volume_mask = make_subsample_mask(
                    batch.volume_mask, subsample_frac, sub_gen
                )
                with autocast_context(device, amp_mode):
                    out_s = eval_module(
                        surface_x=batch.surface_x,
                        surface_mask=sub_surface_mask,
                        volume_x=batch.volume_x,
                        volume_mask=sub_volume_mask,
                    )
                sub_surface_sum = sub_surface_sum + out_s["surface_preds"].float()
                sub_volume_sum = sub_volume_sum + out_s["volume_preds"].float()
                sub_surface_count = sub_surface_count + sub_surface_mask.unsqueeze(-1).float()
                sub_volume_count = sub_volume_count + sub_volume_mask.unsqueeze(-1).float()

            # Avoid /0 at points that never appeared (rare for K=4, p=0.8 → (0.2)^4 = 0.16%).
            sub_surface_count_safe = sub_surface_count.clamp(min=1.0)
            sub_volume_count_safe = sub_volume_count.clamp(min=1.0)

            # Mode predictions ---------------------------------------------
            surface_mirror_only = 0.5 * (surface_pred_a + surface_pred_b)
            volume_mirror_only = 0.5 * (volume_pred_a + volume_pred_b)

            sub_surface_avg_naive = sub_surface_sum / float(subsample_passes)
            sub_volume_avg_naive = sub_volume_sum / float(subsample_passes)
            surface_sub_only_naive = 0.5 * (surface_pred_a + sub_surface_avg_naive)
            volume_sub_only_naive = 0.5 * (volume_pred_a + sub_volume_avg_naive)

            sub_surface_avg_mc = sub_surface_sum / sub_surface_count_safe
            sub_volume_avg_mc = sub_volume_sum / sub_volume_count_safe
            surface_sub_only_mc = 0.5 * (surface_pred_a + sub_surface_avg_mc)
            volume_sub_only_mc = 0.5 * (volume_pred_a + sub_volume_avg_mc)

            denom_stack_naive = 2.0 + float(subsample_passes)
            surface_mxsub_naive = (
                surface_pred_a + surface_pred_b + sub_surface_sum
            ) / denom_stack_naive
            volume_mxsub_naive = (
                volume_pred_a + volume_pred_b + sub_volume_sum
            ) / denom_stack_naive

            # MC stack: count orig+mirror+active_sub at each point
            mxsub_surface_count = sub_surface_count + 2.0
            mxsub_volume_count = sub_volume_count + 2.0
            surface_mxsub_mc = (
                surface_pred_a + surface_pred_b + sub_surface_sum
            ) / mxsub_surface_count
            volume_mxsub_mc = (
                volume_pred_a + volume_pred_b + sub_volume_sum
            ) / mxsub_volume_count

            preds = {
                "orig": (surface_pred_a, volume_pred_a),
                "mirror": (surface_pred_b, volume_pred_b),
                "mirror_only": (surface_mirror_only, volume_mirror_only),
                "subsample_only_naive": (surface_sub_only_naive, volume_sub_only_naive),
                "subsample_only_mc": (surface_sub_only_mc, volume_sub_only_mc),
                "mirror_x_subsample_naive": (surface_mxsub_naive, volume_mxsub_naive),
                "mirror_x_subsample_mc": (surface_mxsub_mc, volume_mxsub_mc),
            }

            for mode, (s, v) in preds.items():
                _accumulate_outputs(accs[mode], batch, s, v, transform)

    finalized: dict[str, dict[str, float]] = {}
    for mode in MODES:
        acc = accs[mode]
        if distributed_state is not None and distributed_state.enabled:
            gathered: list[EvalAccumulator | None] = [None for _ in range(distributed_state.world_size)]
            dist.all_gather_object(gathered, acc)
            if distributed_state.is_main:
                merged = merge_eval_accumulators(g for g in gathered if g is not None)
                finalized[mode] = finalize_eval_accumulator(merged)
            else:
                finalized[mode] = {}
        else:
            finalized[mode] = finalize_eval_accumulator(acc)

    return finalized


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device
    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        print(f"Checkpoint: {cfg.checkpoint}")
        print(
            f"Subsample: frac={cfg.subsample_frac} passes={cfg.subsample_passes} seed={cfg.subsample_seed}"
        )

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

    run = None
    if state.is_main:
        run_name = cfg.wandb_name or f"{cfg.agent}/h240-h183-tta-stack"
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            group=cfg.wandb_group,
            name=run_name,
            config={
                **asdict(cfg),
                "checkpoint_run_id": cfg.run_id,
                "checkpoint_epoch": ck.get("epoch"),
            },
            tags=["h240", "tta", "mesh-subsample", "h183", "eval-only", cfg.agent],
            reinit="finish_previous",
        )

    splits = [
        ("val_surface", val_loaders["val_surface"], "full_val"),
        ("test_surface", test_loaders["test_surface"], "test"),
    ]

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name, loader, log_prefix in splits:
        if state.is_main:
            print(f"\n=== Evaluating split={name} ===")
        t0 = time.time()
        mode_metrics = evaluate_tta_split(
            model=model,
            loader=loader,
            transform=transform,
            device=device,
            amp_mode=cfg.amp_mode,
            distributed_state=state,
            subsample_frac=cfg.subsample_frac,
            subsample_passes=cfg.subsample_passes,
            subsample_seed=cfg.subsample_seed,
        )
        dt = time.time() - t0
        if state.is_main:
            print(f"  done in {dt:.1f}s")
            for mode in MODES:
                print(f"  -- {mode} --")
                print_metrics(name, mode_metrics[mode])
            summary[name] = mode_metrics

            log_obj: dict[str, float] = {}
            for mode, metrics in mode_metrics.items():
                log_obj.update(primary_metric_log(f"{log_prefix}_primary/{mode}", metrics))
                log_obj.update({f"{log_prefix}_extra/{mode}/loss": metrics["loss"]})
            if run is not None:
                wandb.log(log_obj)

    if state.is_main and summary:
        print("\n=== Summary (rel_l2_pct lower-is-better) ===")
        for split, modes in summary.items():
            print(f"\n[{split}]")
            keys = (
                "abupt_axis_mean_rel_l2_pct",
                "surface_pressure_rel_l2_pct",
                "wall_shear_rel_l2_pct",
                "wall_shear_x_rel_l2_pct",
                "wall_shear_y_rel_l2_pct",
                "wall_shear_z_rel_l2_pct",
                "volume_pressure_rel_l2_pct",
            )
            header = f"  {'metric':<36s}"
            for mode in MODES:
                header += f" {mode[:18]:>18s}"
            print(header)
            for k in keys:
                row = f"  {k:<36s}"
                for mode in MODES:
                    row += f" {modes[mode][k]:>18.4f}"
                print(row)

        if run is not None:
            for split, modes in summary.items():
                for mode, metrics in modes.items():
                    for k, v in metrics.items():
                        try:
                            run.summary[f"{split}/{mode}/{k}"] = float(v)
                        except Exception:
                            pass
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
