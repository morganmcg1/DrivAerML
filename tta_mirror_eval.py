# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""H200: TTA mirror-augmentation eval for the DrivAerML surrogate.

Loads a single trained checkpoint and evaluates it in three modes against
the held-out val and test splits:

* ``original``  - vanilla forward pass on the geometry as loaded.
* ``mirrored``  - forward pass on the y-mirrored geometry, with the
  predicted ``tau_y`` component negated so the prediction is reported in
  the original frame.
* ``tta``       - average of the original and the un-mirrored prediction
  in normalised space; the standard 2-mode mirror TTA estimator.

Each batch incurs only two model forward passes; the three modes share
those passes so a full 8-GPU eval is roughly the cost of two normal
evals (not three).

The script does not modify any source file. It re-uses the existing
``EvalAccumulator`` / ``finalize_eval_accumulator`` infrastructure from
``trainer_runtime`` to guarantee metric parity with ``train.py``.

Metric reporting in W&B uses the prefixes:

  ``<split>_original/*``  ``<split>_mirrored/*``  ``<split>_tta/*``

Use case for H200 vs H192: if H192 (mirror-trained) shows a meaningful
test_WSS gain under TTA but H200 (this run, non-mirror-trained) does
not, the TTA gain is mechanism-specific and depends on the model having
learned mirror invariance, not generic averaging.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import wandb
import yaml

from data import load_data, pad_collate
from data.loader import SurfaceBatch
from model import SurfaceTransolver
from trainer_runtime import (
    DistributedState,
    EvalAccumulator,
    StridedDistributedSampler,
    TargetTransform,
    _accumulate_case_rel_l2,
    _masked_sse_count,
    autocast_context,
    cleanup_distributed,
    distributed_barrier,
    finalize_eval_accumulator,
    init_distributed,
    merge_eval_accumulators,
    print_metrics,
)

# Surface input layout (data/loader.py): [x, y, z, nx, ny, nz, area].
SURFACE_INPUT_Y_COL = 1
SURFACE_INPUT_NORMAL_Y_COL = 4
# Volume input layout: [x, y, z, sdf].
VOLUME_INPUT_Y_COL = 1
# Surface target/prediction layout: [cp, tau_x, tau_y, tau_z].
SURFACE_OUTPUT_TAU_Y_COL = 2

EVAL_MODES = ("original", "mirrored", "tta")


@dataclass
class EvalConfig:
    checkpoint: str
    config_yaml: str
    data_root: str
    manifest: str
    batch_size: int
    eval_surface_points: int
    eval_volume_points: int
    num_workers: int
    amp_mode: str
    splits: tuple[str, ...]
    wandb_group: str
    wandb_name: str
    eval_modes: tuple[str, ...]


def parse_args(argv: Iterable[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="H200 TTA mirror-aug eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-surface-points", type=int, default=65536)
    parser.add_argument("--eval-volume-points", type=int, default=65536)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--amp-mode", default="bf16")
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--eval-modes", default="original,mirrored,tta")
    args = parser.parse_args(list(argv) if argv is not None else None)
    modes = tuple(m.strip() for m in args.eval_modes.split(",") if m.strip())
    for m in modes:
        if m not in EVAL_MODES:
            raise ValueError(f"unknown eval mode {m!r}; choose from {EVAL_MODES}")
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    for s in splits:
        if s not in ("val", "test"):
            raise ValueError(f"unknown split {s!r}; expected val or test")
    return EvalConfig(
        checkpoint=args.checkpoint,
        config_yaml=args.config_yaml,
        data_root=args.data_root,
        manifest=args.manifest,
        batch_size=args.batch_size,
        eval_surface_points=args.eval_surface_points,
        eval_volume_points=args.eval_volume_points,
        num_workers=args.num_workers,
        amp_mode=args.amp_mode,
        splits=splits,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        eval_modes=modes,
    )


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


def build_model_from_yaml(config_yaml: Path, state_dict: dict) -> SurfaceTransolver:
    with config_yaml.open("r") as fh:
        config = yaml.safe_load(fh)
    use_aux_decoder_heads = "surface_out.0.weight" in state_dict
    return SurfaceTransolver(
        n_layers=int(config.get("model_layers", 3)),
        n_hidden=int(config.get("model_hidden_dim", 192)),
        dropout=float(config.get("model_dropout", 0.0)),
        n_head=int(config.get("model_heads", 3)),
        mlp_ratio=int(config.get("model_mlp_ratio", 4)),
        slice_num=int(config.get("model_slices", 96)),
        rff_num_features=int(config.get("rff_num_features", 0)),
        rff_sigma=float(config.get("rff_sigma", 1.0)),
        rff_init_sigmas=parse_rff_init_sigmas(config.get("rff_init_sigmas")),
        pos_encoding_mode=str(config.get("pos_encoding_mode", "sincos")),
        use_qk_norm=bool(config.get("use_qk_norm", False)),
        use_surf_to_vol_xattn=bool(config.get("use_surf_to_vol_xattn", False)),
        use_aux_decoder_heads=use_aux_decoder_heads,
        drop_path_max=float(config.get("drop_path_max", 0.0)),
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[SurfaceTransolver, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint {path} missing 'model' state dict")
    config_yaml = path.parent / "config.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(f"Expected config.yaml next to checkpoint: {config_yaml}")
    state_dict = dict(checkpoint["model"])
    model = build_model_from_yaml(config_yaml, state_dict).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch loading {path}: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model, checkpoint


def mirror_batch(batch: SurfaceBatch) -> SurfaceBatch:
    """Return a y-mirrored shallow copy of ``batch``.

    Negates ``y`` and ``n_y`` in the surface inputs and ``y`` in the
    volume inputs. Other channels (cp, tau, normals/coords x and z, area,
    sdf) are invariant under y-reflection.
    """

    surface_x = batch.surface_x.clone()
    surface_x[..., SURFACE_INPUT_Y_COL] = -surface_x[..., SURFACE_INPUT_Y_COL]
    surface_x[..., SURFACE_INPUT_NORMAL_Y_COL] = -surface_x[..., SURFACE_INPUT_NORMAL_Y_COL]
    volume_x = batch.volume_x.clone()
    if volume_x.numel() > 0:
        volume_x[..., VOLUME_INPUT_Y_COL] = -volume_x[..., VOLUME_INPUT_Y_COL]
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


def unmirror_surface_pred_norm(
    pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> torch.Tensor:
    """Apply the y-reflection sign flip to a normalised surface prediction.

    The lateral sign flip (``tau_y -> -tau_y``) is a physical-space
    operation. The model output is in normalised space, where channel
    ``c`` satisfies ``y_norm = (y_phys - mu_c) / sigma_c``. Negating the
    physical value gives ``-y_phys = -(y_norm * sigma_c + mu_c)``, so

        y_norm_new = -y_norm - 2 * mu_c / sigma_c

    For DrivAerML's tau_y, ``mu / sigma ~= 0.0015 / 1.356 ~= 1.1e-3``,
    so the offset is small but we include it for physical fidelity.
    """

    out = pred_norm.clone()
    mean = transform.surface_y_mean.to(out.device)
    std = transform.surface_y_std.to(out.device)
    c = SURFACE_OUTPUT_TAU_Y_COL
    out[..., c] = -out[..., c] - 2.0 * mean[c] / std[c]
    return out


@torch.no_grad()
def forward_pred_norm(
    model: SurfaceTransolver,
    batch: SurfaceBatch,
    device: torch.device,
    amp_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_context(device, amp_mode):
        out = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
    surface_pred = out["surface_preds"].float()
    volume_pred = out["volume_preds"].float()
    return surface_pred, volume_pred


def accumulate_batch_with_predictions(
    accumulator: EvalAccumulator,
    *,
    batch: SurfaceBatch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> None:
    """Mirror of trainer_runtime.accumulate_eval_batch using precomputed preds.

    The forward pass is decoupled so a single batch can be scored under
    multiple TTA averaging strategies without duplicating model work.
    """

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
def evaluate_split_tta(
    model: SurfaceTransolver,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    eval_modes: tuple[str, ...],
    distributed_state: DistributedState,
) -> dict[str, dict[str, float]]:
    """Run one eval pass yielding metrics for each requested TTA mode."""

    model.eval()
    accumulators = {mode: EvalAccumulator() for mode in eval_modes}
    needs_mirrored = ("mirrored" in eval_modes) or ("tta" in eval_modes)

    n_batches = 0
    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        surface_a, volume_a = forward_pred_norm(model, batch, device, amp_mode)
        surface_b_unmirrored = None
        volume_b = None
        if needs_mirrored:
            mirrored = mirror_batch(batch)
            surface_b_raw, volume_b = forward_pred_norm(
                model, mirrored, device, amp_mode
            )
            surface_b_unmirrored = unmirror_surface_pred_norm(surface_b_raw, transform)
        if "original" in eval_modes:
            accumulate_batch_with_predictions(
                accumulators["original"],
                batch=batch,
                surface_pred_norm=surface_a,
                volume_pred_norm=volume_a,
                transform=transform,
            )
        if "mirrored" in eval_modes:
            accumulate_batch_with_predictions(
                accumulators["mirrored"],
                batch=batch,
                surface_pred_norm=surface_b_unmirrored,
                volume_pred_norm=volume_b,
                transform=transform,
            )
        if "tta" in eval_modes:
            tta_surface = 0.5 * (surface_a + surface_b_unmirrored)
            tta_volume = 0.5 * (volume_a + volume_b)
            accumulate_batch_with_predictions(
                accumulators["tta"],
                batch=batch,
                surface_pred_norm=tta_surface,
                volume_pred_norm=tta_volume,
                transform=transform,
            )
        n_batches += 1

    if distributed_state.enabled:
        finalized: dict[str, dict[str, float]] = {}
        for mode in eval_modes:
            gathered: list[EvalAccumulator | None] = [
                None for _ in range(distributed_state.world_size)
            ]
            dist.all_gather_object(gathered, accumulators[mode])
            if not distributed_state.is_main:
                finalized[mode] = {}
                continue
            merged = merge_eval_accumulators(a for a in gathered if a is not None)
            finalized[mode] = finalize_eval_accumulator(merged)
        if distributed_state.is_main:
            finalized["_runtime_seconds"] = {"value": time.time() - t0, "n_batches": float(n_batches)}
        return finalized

    out: dict[str, dict[str, float]] = {
        mode: finalize_eval_accumulator(accumulators[mode]) for mode in eval_modes
    }
    out["_runtime_seconds"] = {"value": time.time() - t0, "n_batches": float(n_batches)}
    return out


def init_wandb(cfg: EvalConfig, checkpoint_metadata: dict) -> "wandb.sdk.wandb_run.Run":
    init_kwargs = dict(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        group=cfg.wandb_group or None,
        name=cfg.wandb_name or None,
        config={
            "checkpoint": cfg.checkpoint,
            "config_yaml": cfg.config_yaml,
            "data_root": cfg.data_root,
            "manifest": cfg.manifest,
            "batch_size": cfg.batch_size,
            "eval_surface_points": cfg.eval_surface_points,
            "eval_volume_points": cfg.eval_volume_points,
            "amp_mode": cfg.amp_mode,
            "splits": list(cfg.splits),
            "eval_modes": list(cfg.eval_modes),
            "checkpoint_epoch": checkpoint_metadata.get("epoch"),
            "checkpoint_source": checkpoint_metadata.get("checkpoint_source"),
        },
        tags=["h200", "tta-mirror", "edward"],
        mode=os.environ.get("WANDB_MODE", "online"),
        job_type="eval",
    )
    return wandb.init(**init_kwargs)


def format_metric_log(
    split_name: str,
    mode: str,
    metrics: dict[str, float],
) -> dict[str, float]:
    prefix_primary = f"{split_name}_primary_{mode}"
    prefix_raw = f"{split_name}_{mode}"
    log: dict[str, float] = {}
    primary_keys = (
        "abupt_axis_mean_rel_l2_pct",
        "surface_pressure_rel_l2_pct",
        "wall_shear_rel_l2_pct",
        "wall_shear_x_rel_l2_pct",
        "wall_shear_y_rel_l2_pct",
        "wall_shear_z_rel_l2_pct",
        "volume_pressure_rel_l2_pct",
        "surface_pressure_mae",
        "wall_shear_mae",
        "volume_pressure_mae",
    )
    for key in primary_keys:
        if key in metrics:
            log[f"{prefix_primary}/{key}"] = float(metrics[key])
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            log[f"{prefix_raw}/{key}"] = float(value)
    return log


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device
    if state.is_main:
        print(f"Device: {device}")
        if state.enabled:
            print(f"DDP world_size={state.world_size}")

    checkpoint_path = Path(cfg.checkpoint).resolve()
    config_yaml = Path(cfg.config_yaml).resolve()
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    if state.is_main:
        print(
            f"Loaded checkpoint epoch={checkpoint.get('epoch')} "
            f"source={checkpoint.get('checkpoint_source')} "
            f"params={n_params / 1e6:.2f}M"
        )

    # Match train.py's eval-time config block: 65536 surface/volume chunked.
    # ``load_data`` only needs ``eval_*_points``; train_* parameters are
    # ignored downstream because we never instantiate the train loader.
    _, val_splits, test_splits, stats = load_data(
        manifest_path=cfg.manifest,
        root=cfg.data_root or None,
        train_surface_points=cfg.eval_surface_points,
        eval_surface_points=cfg.eval_surface_points,
        train_volume_points=cfg.eval_volume_points,
        eval_volume_points=cfg.eval_volume_points,
        debug=False,
    )
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    split_loaders: dict[str, torch.utils.data.DataLoader] = {}
    if "val" in cfg.splits:
        split_loaders["val"] = build_eval_loader(
            val_splits["val_surface"], cfg, state=state
        )
    if "test" in cfg.splits:
        split_loaders["test"] = build_eval_loader(
            test_splits["test_surface"], cfg, state=state
        )

    run = None
    if state.is_main:
        run = init_wandb(cfg, checkpoint)
        wandb.summary["model/n_params"] = n_params

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, loader in split_loaders.items():
        if state.is_main:
            print(f"\n=== Evaluating {split_name} ({len(loader.dataset)} views) ===")
        results = evaluate_split_tta(
            model=model,
            loader=loader,
            transform=transform,
            device=device,
            amp_mode=cfg.amp_mode,
            eval_modes=cfg.eval_modes,
            distributed_state=state,
        )
        if state.is_main:
            runtime = results.pop("_runtime_seconds", None)
            all_metrics[split_name] = results
            for mode in cfg.eval_modes:
                metrics = results[mode]
                print_metrics(f"{split_name}_{mode}", metrics)
                print(
                    f"  {split_name}_{mode} -> "
                    f"abupt={metrics['abupt_axis_mean_rel_l2_pct']:.4f}% | "
                    f"SP={metrics['surface_pressure_rel_l2_pct']:.4f}% | "
                    f"WSS={metrics['wall_shear_rel_l2_pct']:.4f}% | "
                    f"VP={metrics['volume_pressure_rel_l2_pct']:.4f}% | "
                    f"WSS_x={metrics['wall_shear_x_rel_l2_pct']:.4f}% | "
                    f"WSS_y={metrics['wall_shear_y_rel_l2_pct']:.4f}% | "
                    f"WSS_z={metrics['wall_shear_z_rel_l2_pct']:.4f}%"
                )
            if runtime is not None:
                print(
                    f"  runtime={runtime['value']:.1f}s "
                    f"n_batches={int(runtime['n_batches'])}"
                )
        distributed_barrier(state)

    if state.is_main and run is not None:
        log: dict[str, float] = {}
        for split_name, results in all_metrics.items():
            for mode in cfg.eval_modes:
                log.update(format_metric_log(split_name, mode, results[mode]))
        wandb.log(log, step=0)
        wandb.summary.update(log)
        if "test" in all_metrics and "tta" in cfg.eval_modes and "original" in cfg.eval_modes:
            o = all_metrics["test"]["original"]
            t = all_metrics["test"]["tta"]
            deltas = {
                "test_delta_tta_minus_original/abupt": t["abupt_axis_mean_rel_l2_pct"]
                - o["abupt_axis_mean_rel_l2_pct"],
                "test_delta_tta_minus_original/wall_shear": t["wall_shear_rel_l2_pct"]
                - o["wall_shear_rel_l2_pct"],
                "test_delta_tta_minus_original/wall_shear_x": t["wall_shear_x_rel_l2_pct"]
                - o["wall_shear_x_rel_l2_pct"],
                "test_delta_tta_minus_original/wall_shear_y": t["wall_shear_y_rel_l2_pct"]
                - o["wall_shear_y_rel_l2_pct"],
                "test_delta_tta_minus_original/wall_shear_z": t["wall_shear_z_rel_l2_pct"]
                - o["wall_shear_z_rel_l2_pct"],
                "test_delta_tta_minus_original/surface_pressure": t["surface_pressure_rel_l2_pct"]
                - o["surface_pressure_rel_l2_pct"],
                "test_delta_tta_minus_original/volume_pressure": t["volume_pressure_rel_l2_pct"]
                - o["volume_pressure_rel_l2_pct"],
            }
            wandb.log(deltas, step=0)
            wandb.summary.update(deltas)
            print("\n=== TTA vs original (test) ===")
            for k, v in deltas.items():
                print(f"  {k.split('/', 1)[1]}: {v:+.4f}pp")
        wandb.finish()

    cleanup_distributed(state)


def build_eval_loader(
    dataset,
    cfg: EvalConfig,
    *,
    state: DistributedState,
) -> torch.utils.data.DataLoader:
    sampler = None
    if state.enabled:
        sampler = StridedDistributedSampler(
            dataset, num_replicas=state.world_size, rank=state.rank
        )
    num_workers = cfg.num_workers if cfg.num_workers >= 0 else min(4, os.cpu_count() or 4)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=pad_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


if __name__ == "__main__":
    main()
