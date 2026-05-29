# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H223: Rotational TTA on H185 EP13 — orthogonal-augmentation TTA (eval-only).

Extends H209's mirror TTA with small-angle rotations around the x-axis.

Rotation convention (right-hand rule about +x):
    point (y, z) -> ( y*cos(θ) - z*sin(θ),  y*sin(θ) + z*cos(θ) )
    normal (n_y, n_z) -> same transform
    wall_shear (tau_y, tau_z) -> same transform (vector field rotates with frame)

To bring rotated-frame predictions back to the original frame we apply the
inverse rotation (-θ) to the y,z components of the wall-shear vector. Scalar
channels (cp, volume_pressure) and SDF are invariant.

Eval modes (passed as a comma-separated string to --eval-modes):
    original         : single pass on the original geometry (sanity)
    mirror           : single un-mirrored mirror pass
    mirror_rotation  : 4-pass average = (orig + mirror + rot+θ + rot-θ) / 4
    rotation_only    : 3-pass average = (orig + rot+θ + rot-θ) / 3
"""

from __future__ import annotations

import argparse
import math
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

VALID_MODES = ("original", "mirror", "mirror_rotation", "rotation_only")


@dataclass
class EvalConfig:
    checkpoint: str = ""
    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h223_eval"
    wandb_group: str = "h223-askeladd-rotation-tta"
    wandb_name: str = ""
    agent: str = "askeladd"

    # Eval mode + rotation knob
    eval_modes: str = "original,mirror,mirror_rotation,rotation_only"
    rotation_angle: float = 2.0  # degrees, applied as ±rotation_angle

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
    max_batches: int = 0  # 0 = unlimited; >0 truncates loader for smoke tests


def parse_args(argv: Iterable[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="H223 rotation TTA eval")
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


def parse_modes(spec: str) -> list[str]:
    modes = [m.strip() for m in spec.split(",") if m.strip()]
    for m in modes:
        if m not in VALID_MODES:
            raise ValueError(f"Unknown eval mode {m!r}, valid: {VALID_MODES}")
    # de-duplicate but preserve order
    seen = set()
    deduped: list[str] = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


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


# --- Geometric TTA operators ---


def mirror_inputs(batch: SurfaceBatch) -> SurfaceBatch:
    """y-mirror, same as H209/H148/H183."""
    surface_x = batch.surface_x.clone()
    surface_x[..., 1].neg_()  # y
    surface_x[..., 4].neg_()  # normal_y
    volume_x = batch.volume_x.clone()
    volume_x[..., 1].neg_()
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


def rotate_inputs(batch: SurfaceBatch, theta_rad: float) -> SurfaceBatch:
    """Rotate y,z about the x-axis by +theta_rad in both inputs.

    Channels affected:
      surface_x[..., 1] = y, [..., 2] = z, [..., 4] = n_y, [..., 5] = n_z
      volume_x[..., 1] = y, [..., 2] = z
      area, sdf, cp, tau_*  are unchanged for INPUTS.
    """
    cos = math.cos(theta_rad)
    sin = math.sin(theta_rad)

    surface_x = batch.surface_x.clone()
    y = surface_x[..., 1].clone()
    z = surface_x[..., 2].clone()
    surface_x[..., 1] = y * cos - z * sin
    surface_x[..., 2] = y * sin + z * cos
    ny = surface_x[..., 4].clone()
    nz = surface_x[..., 5].clone()
    surface_x[..., 4] = ny * cos - nz * sin
    surface_x[..., 5] = ny * sin + nz * cos

    volume_x = batch.volume_x.clone()
    vy = volume_x[..., 1].clone()
    vz = volume_x[..., 2].clone()
    volume_x[..., 1] = vy * cos - vz * sin
    volume_x[..., 2] = vy * sin + vz * cos

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


def unrotate_surface_pred(pred: torch.Tensor, theta_rad: float) -> torch.Tensor:
    """Rotate wall_shear (tau_y, tau_z) by -theta_rad to bring back to orig frame.

    surface_pred channels: [cp, tau_x, tau_y, tau_z]. cp and tau_x untouched.
    """
    cos = math.cos(-theta_rad)
    sin = math.sin(-theta_rad)
    out = pred.clone()
    tau_y = pred[..., 2].clone()
    tau_z = pred[..., 3].clone()
    out[..., 2] = tau_y * cos - tau_z * sin
    out[..., 3] = tau_y * sin + tau_z * cos
    return out


# --- Eval accumulation (mirrors H209 _accumulate_outputs verbatim) ---


def _accumulate_outputs(
    acc: EvalAccumulator,
    batch: SurfaceBatch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> None:
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
    modes: list[str],
    theta_rad: float,
    max_batches: int = 0,
) -> dict[str, dict[str, float]]:
    """Run the requested eval modes in a single pass through the loader.

    Forward passes are computed lazily based on which modes are requested:
      - original             → needs orig
      - mirror               → needs orig (for the loop) + mirror
      - mirror_rotation      → needs orig + mirror + rot+θ + rot-θ
      - rotation_only        → needs orig + rot+θ + rot-θ
    """
    needs_mirror = any(m in modes for m in ("mirror", "mirror_rotation"))
    needs_rot = any(m in modes for m in ("mirror_rotation", "rotation_only"))

    accumulators = {m: EvalAccumulator() for m in modes}

    model.eval()
    eval_module = unwrap_model(model)

    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            with autocast_context(device, amp_mode):
                out_orig = eval_module(
                    surface_x=batch.surface_x,
                    surface_mask=batch.surface_mask,
                    volume_x=batch.volume_x,
                    volume_mask=batch.volume_mask,
                )
                surface_orig = out_orig["surface_preds"].float()
                volume_orig = out_orig["volume_preds"].float()

                surface_mirror = None
                volume_mirror = None
                if needs_mirror:
                    mirrored = mirror_inputs(batch)
                    out_m = eval_module(
                        surface_x=mirrored.surface_x,
                        surface_mask=mirrored.surface_mask,
                        volume_x=mirrored.volume_x,
                        volume_mask=mirrored.volume_mask,
                    )
                    surface_mirror = unmirror_surface_pred(out_m["surface_preds"].float())
                    volume_mirror = out_m["volume_preds"].float()

                surface_rot_p = surface_rot_m = None
                volume_rot_p = volume_rot_m = None
                if needs_rot:
                    rot_p = rotate_inputs(batch, theta_rad=+theta_rad)
                    out_rp = eval_module(
                        surface_x=rot_p.surface_x,
                        surface_mask=rot_p.surface_mask,
                        volume_x=rot_p.volume_x,
                        volume_mask=rot_p.volume_mask,
                    )
                    surface_rot_p = unrotate_surface_pred(
                        out_rp["surface_preds"].float(), theta_rad=+theta_rad
                    )
                    volume_rot_p = out_rp["volume_preds"].float()

                    rot_m = rotate_inputs(batch, theta_rad=-theta_rad)
                    out_rm = eval_module(
                        surface_x=rot_m.surface_x,
                        surface_mask=rot_m.surface_mask,
                        volume_x=rot_m.volume_x,
                        volume_mask=rot_m.volume_mask,
                    )
                    surface_rot_m = unrotate_surface_pred(
                        out_rm["surface_preds"].float(), theta_rad=-theta_rad
                    )
                    volume_rot_m = out_rm["volume_preds"].float()

            for m in modes:
                if m == "original":
                    sp, vp = surface_orig, volume_orig
                elif m == "mirror":
                    sp, vp = surface_mirror, volume_mirror
                elif m == "mirror_rotation":
                    sp = (surface_orig + surface_mirror + surface_rot_p + surface_rot_m) / 4.0
                    vp = (volume_orig + volume_mirror + volume_rot_p + volume_rot_m) / 4.0
                elif m == "rotation_only":
                    sp = (surface_orig + surface_rot_p + surface_rot_m) / 3.0
                    vp = (volume_orig + volume_rot_p + volume_rot_m) / 3.0
                else:
                    raise AssertionError(m)
                _accumulate_outputs(accumulators[m], batch, sp, vp, transform)

            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break

    results: dict[str, dict[str, float]] = {}
    for m, acc in accumulators.items():
        if distributed_state is not None and distributed_state.enabled:
            gathered = [None for _ in range(distributed_state.world_size)]
            dist.all_gather_object(gathered, acc)
            if distributed_state.is_main:
                merged = merge_eval_accumulators(g for g in gathered if g is not None)
                results[m] = finalize_eval_accumulator(merged)
            else:
                results[m] = {}
        else:
            results[m] = finalize_eval_accumulator(acc)
    return results


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device
    modes = parse_modes(cfg.eval_modes)
    theta_deg = float(cfg.rotation_angle)
    theta_rad = math.radians(theta_deg)

    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        print(f"Checkpoint: {cfg.checkpoint}")
        print(f"Modes: {modes}  |  rotation_angle: {theta_deg}° ({theta_rad:.6f} rad)")
        if cfg.max_batches:
            print(f"DEBUG: capped to max_batches={cfg.max_batches} per split")

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
        run_name = cfg.wandb_name or f"{cfg.agent}/h223-h185-rotation-tta"
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            group=cfg.wandb_group,
            name=run_name,
            config={
                **asdict(cfg),
                "checkpoint_run_id": "yw2a5dyl",
                "checkpoint_epoch": ck.get("epoch"),
                "modes": modes,
                "rotation_angle_deg": theta_deg,
                "rotation_angle_rad": theta_rad,
            },
            tags=["h223", "tta", "rotation-aug", "mirror-aug", "eval-only", cfg.agent],
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
        per_mode = evaluate_tta_split(
            model=model,
            loader=loader,
            transform=transform,
            device=device,
            amp_mode=cfg.amp_mode,
            distributed_state=state,
            modes=modes,
            theta_rad=theta_rad,
            max_batches=cfg.max_batches,
        )
        dt = time.time() - t0
        if state.is_main:
            print(f"  done in {dt:.1f}s")
            for m, metrics in per_mode.items():
                print(f"  -- {m} --")
                print_metrics(name, metrics)
            summary[name] = per_mode

            log_obj: dict[str, float] = {}
            for m, metrics in per_mode.items():
                log_obj.update(primary_metric_log(f"{log_prefix}_primary/{m}", metrics))
                if "loss" in metrics:
                    log_obj.update({f"{log_prefix}_extra/{m}/loss": metrics["loss"]})
            if run is not None and log_obj:
                wandb.log(log_obj)

    if state.is_main and summary:
        print("\n=== Summary (rel_l2_pct lower-is-better) ===")
        for split, modes_metrics in summary.items():
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
            col_names = list(modes_metrics.keys())
            header = f"  {'metric':<36s} " + " ".join(f"{c:>16s}" for c in col_names)
            print(header)
            for k in keys:
                row_vals = [modes_metrics[c].get(k, float('nan')) for c in col_names]
                row = f"  {k:<36s} " + " ".join(f"{v:>16.4f}" for v in row_vals)
                print(row)

        if run is not None:
            for split, modes_metrics in summary.items():
                for m, metrics in modes_metrics.items():
                    for k, v in metrics.items():
                        try:
                            run.summary[f"{split}/{m}/{k}"] = float(v)
                        except Exception:
                            pass
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
