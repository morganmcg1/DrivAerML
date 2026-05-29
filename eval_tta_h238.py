# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H238: Weighted TTA mirror-blend eval on H185 EP13 checkpoint (eval-only, no training).

Same pipeline as H209 (eval_tta_h209.py) but the TTA blend weight α is configurable
via ``--mirror-blend-weight``. The final TTA prediction (in normalized space) is::

    pred_tta = (1 - α) * pred_orig + α * pred_mirror_unmirrored

α=0.5 reproduces H209 exactly. α=0.0 = no TTA (orig only). α=1.0 = pure mirror.

Mirror convention (H148/H183/H209):
    surface_x [x, y, z, nx, ny, nz, area] -> negate y(1) and ny(4)
    volume_x  [x, y, z, sdf]              -> negate y(1)
    surface_y predictions [cp, tau_x, tau_y, tau_z] -> un-mirror by negating tau_y(2)
    volume_y predictions [volume_pressure] -> invariant
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
    """Minimal config mirroring train.Config — only the fields we need for eval.

    Defaults match the H185 training-time config so the reconstructed model
    matches the checkpoint exactly.
    """

    checkpoint: str = ""
    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h238_eval"
    wandb_group: str = "h238-alphonse-blend-sweep"
    wandb_name: str = ""
    agent: str = "alphonse"

    # H238-specific knob: blend weight α for (1-α)*orig + α*mirror in normalized space.
    mirror_blend_weight: float = 0.5

    # Eval-only loader params (no train sampling)
    batch_size: int = 4
    eval_surface_points: int = 65536
    eval_volume_points: int = 65536
    train_surface_points: int = 65536  # only used by load_data to build the train_ds; unused by us
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
    parser = argparse.ArgumentParser(description="H238 weighted-TTA mirror-blend eval")
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


# --- TTA mirror helpers (H148/H209 convention) ---


def mirror_inputs(batch: SurfaceBatch) -> SurfaceBatch:
    """Negate y/normal_y in surface_x, y in volume_x. Keep masks and targets unchanged."""
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


# --- TTA accumulation: reuse the standard eval pipeline 3x in parallel ---


def _accumulate_outputs(
    acc: EvalAccumulator,
    batch: SurfaceBatch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> None:
    """Same logic as accumulate_eval_batch but takes precomputed predictions.

    Inputs are normalized predictions; we denormalize them here for the MAE / relL2
    accumulators exactly like ``accumulate_eval_batch`` does.
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


def evaluate_tta_split(
    *,
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    distributed_state,
    blend: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Run three eval modes in a single pass through the loader.

    Returns ``(orig_metrics, mirror_metrics, tta_metrics)``.

    The TTA prediction is computed in **normalized space** as
    ``(1-α)*orig + α*mirror`` where α = ``blend``; denormalization happens inside
    ``_accumulate_outputs`` exactly like the standard pipeline.
    """
    acc_orig = EvalAccumulator()
    acc_mirror = EvalAccumulator()
    acc_tta = EvalAccumulator()

    model.eval()
    eval_module = unwrap_model(model)

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

            surface_pred_b_raw = out_b["surface_preds"].float()
            volume_pred_b_raw = out_b["volume_preds"].float()
            surface_pred_b = unmirror_surface_pred(surface_pred_b_raw)
            volume_pred_b = unmirror_volume_pred(volume_pred_b_raw)

            surface_pred_tta = (1.0 - blend) * surface_pred_a + blend * surface_pred_b
            volume_pred_tta = (1.0 - blend) * volume_pred_a + blend * volume_pred_b

            _accumulate_outputs(acc_orig, batch, surface_pred_a, volume_pred_a, transform)
            _accumulate_outputs(acc_mirror, batch, surface_pred_b, volume_pred_b, transform)
            _accumulate_outputs(acc_tta, batch, surface_pred_tta, volume_pred_tta, transform)

    metrics = []
    for acc in (acc_orig, acc_mirror, acc_tta):
        if distributed_state is not None and distributed_state.enabled:
            gathered = [None for _ in range(distributed_state.world_size)]
            dist.all_gather_object(gathered, acc)
            if distributed_state.is_main:
                acc = merge_eval_accumulators(g for g in gathered if g is not None)
                metrics.append(finalize_eval_accumulator(acc))
            else:
                metrics.append({})
        else:
            metrics.append(finalize_eval_accumulator(acc))

    return metrics[0], metrics[1], metrics[2]


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device
    blend = float(cfg.mirror_blend_weight)
    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        print(f"Checkpoint: {cfg.checkpoint}")
        print(f"Mirror blend weight α: {blend}")

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
        run_name = cfg.wandb_name or f"{cfg.agent}/h238-blend-{blend:g}"
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            group=cfg.wandb_group,
            name=run_name,
            config={
                **asdict(cfg),
                "checkpoint_run_id": "yw2a5dyl",
                "checkpoint_epoch": ck.get("epoch"),
            },
            tags=["h238", "tta", "mirror-aug", "weighted-blend", "eval-only", cfg.agent],
            reinit="finish_previous",
        )

    splits = [
        ("val_surface", val_loaders["val_surface"], "full_val"),
        ("test_surface", test_loaders["test_surface"], "test"),
    ]

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name, loader, log_prefix in splits:
        if state.is_main:
            print(f"\n=== Evaluating split={name} α={blend} ===")
        t0 = time.time()
        orig, mirror, tta = evaluate_tta_split(
            model=model,
            loader=loader,
            transform=transform,
            device=device,
            amp_mode=cfg.amp_mode,
            distributed_state=state,
            blend=blend,
        )
        dt = time.time() - t0
        if state.is_main:
            print(f"  done in {dt:.1f}s")
            print("  -- orig --")
            print_metrics(name, orig)
            print("  -- mirror --")
            print_metrics(name, mirror)
            print(f"  -- tta (α={blend}) --")
            print_metrics(name, tta)
            summary[name] = {"orig": orig, "mirror": mirror, "tta": tta}

            log_obj: dict[str, float] = {"mirror_blend_weight": blend}
            for mode, metrics in (("orig", orig), ("mirror", mirror), ("tta", tta)):
                log_obj.update(primary_metric_log(f"{log_prefix}_primary/{mode}", metrics))
                log_obj.update({f"{log_prefix}_extra/{mode}/loss": metrics["loss"]})
            if run is not None:
                wandb.log(log_obj)

    if state.is_main and summary:
        print(f"\n=== Summary (rel_l2_pct lower-is-better) α={blend} ===")
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
            print(f"  {'metric':<36s} {'orig':>10s} {'mirror':>10s} {'tta':>10s} {'tta_vs_orig':>12s}")
            for k in keys:
                o = modes["orig"][k]
                m = modes["mirror"][k]
                t = modes["tta"][k]
                d = t - o
                print(f"  {k:<36s} {o:>10.4f} {m:>10.4f} {t:>10.4f} {d:>+12.4f}")

        if run is not None:
            for split, modes in summary.items():
                for mode, metrics in modes.items():
                    for k, v in metrics.items():
                        try:
                            run.summary[f"{split}/{mode}/{k}"] = float(v)
                        except Exception:
                            pass
            run.summary["mirror_blend_weight"] = blend
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
