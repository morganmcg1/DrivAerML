# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H239: Mesh-subsample TTA on H148 EP13 checkpoint (eval-only, no training).

Evaluates three TTA modes on val + test splits in a single pass through the loader:
    * mirror_only        : 0.5 * (orig + mirror)   — reproduces H230 H148+mirror
    * subsample_only     : 0.5 * orig + 0.5 * subsample_avg_MC
    * mirror_x_subsample : (orig + mirror + subsample_avg_MC) / 3 — 6-pass stack

Subsample mechanism (parallel to askeladd H231 on H185):
    Per pass, independently sample ~80% of the valid surface and volume points.
    `subsample_mask = original_mask & uniform_random_keep(p=0.8)` — points outside the
    subset are treated as padding, so the model attends only to the subset and emits
    zero predictions there. We aggregate across 4 passes using MC-coverage averaging:
        sub_avg_MC[j] = sum_i pred_i[j] / max(1, sum_i kept_i[j])
    This removes the ~20% downward bias of naive /N averaging (each point is unmasked
    in only ~80% of passes, so naive mean = 0.8 * true). For the rare (<0.2%) cases
    where a point is never kept across all 4 passes, we fall back to the orig pred so
    we never inject true zeros into the TTA output.

Mirror convention (H148 same as H185):
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

    Defaults match H148 yw2a5dyl/2qr5guel training-time config so the reconstructed
    model matches the checkpoint exactly. H148 shares architecture with H185.
    """

    checkpoint: str = "outputs/ensemble_cache/2qr5guel/checkpoint.pt"
    run_id: str = "2qr5guel"
    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h239_eval"
    wandb_group: str = "h239-thorfinn-mesh-subsample-h148"
    wandb_name: str = ""
    agent: str = "thorfinn"

    # TTA params
    subsample_frac: float = 0.8
    subsample_passes: int = 4
    seed: int = 42

    # Eval-only loader params (no train sampling)
    batch_size: int = 4
    eval_surface_points: int = 65536
    eval_volume_points: int = 65536
    train_surface_points: int = 65536  # unused for eval
    train_volume_points: int = 65536
    num_workers: int = -1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Model arch (must match H148 2qr5guel — identical to H185)
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
    parser = argparse.ArgumentParser(description="H239 mesh-subsample + mirror TTA eval on H148")
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
    # legacy alias accepted by some advisor templates
    parser.add_argument("--mode", type=str, default=None, help="ignored; all 3 modes run together")
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


# --- TTA mirror helpers (H148/H185 convention, same as H209) ---


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
    out[..., 2].neg_()
    return out


def unmirror_volume_pred(pred: torch.Tensor) -> torch.Tensor:
    return pred


# --- Mesh-subsample TTA helpers ---


def make_subsample_mask(
    base_mask: torch.Tensor,
    keep_frac: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate a per-pass mask by keeping `keep_frac` of base_mask=True positions.

    Returns a boolean mask of the same shape as base_mask. Positions that were already
    False in base_mask remain False. Positions that were True are kept independently
    with probability `keep_frac` (so the expected fraction of True positions in the
    output is `keep_frac * base_mask.float().mean()`).
    """
    rand = torch.empty(base_mask.shape, dtype=torch.float32, device=base_mask.device)
    rand.uniform_(0.0, 1.0, generator=generator)
    keep = rand < keep_frac
    return base_mask & keep


# --- Eval accumulator reuse: 3 TTA modes per batch ---


def _accumulate_outputs(
    acc: EvalAccumulator,
    batch: SurfaceBatch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> None:
    """Same as eval_tta_h209._accumulate_outputs — denormalize, accumulate MAE/relL2."""
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


# All modes we compute per batch — kept stable so the W&B summary keys match the table.
TTA_MODES = (
    "orig",
    "mirror",
    "subsample_avg",
    "mirror_only",  # 0.5 * (orig + mirror)
    "subsample_only",  # 0.5 * orig + 0.5 * subsample_avg
    "mirror_x_subsample",  # (orig + mirror + subsample_avg) / 3
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
    rank_seed: int,
) -> dict[str, dict[str, float]]:
    """Run all TTA forward passes in one pass through the loader, compute all modes.

    Returns: dict[mode_name -> finalized metrics dict].
    """
    accs: dict[str, EvalAccumulator] = {m: EvalAccumulator() for m in TTA_MODES}

    model.eval()
    eval_module = unwrap_model(model)

    # Use a CPU generator so the same seed -> same masks regardless of CUDA RNG state.
    # We bump per-rank by rank_seed so different ranks see independent subsamples.
    gen = torch.Generator(device=device)
    gen.manual_seed(rank_seed)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mirrored = mirror_inputs(batch)

            # orig + mirror passes
            with autocast_context(device, amp_mode):
                out_orig = eval_module(
                    surface_x=batch.surface_x,
                    surface_mask=batch.surface_mask,
                    volume_x=batch.volume_x,
                    volume_mask=batch.volume_mask,
                )
                out_mirror = eval_module(
                    surface_x=mirrored.surface_x,
                    surface_mask=mirrored.surface_mask,
                    volume_x=mirrored.volume_x,
                    volume_mask=mirrored.volume_mask,
                )

            surface_pred_orig = out_orig["surface_preds"].float()
            volume_pred_orig = out_orig["volume_preds"].float()
            surface_pred_mirror = unmirror_surface_pred(out_mirror["surface_preds"].float())
            volume_pred_mirror = unmirror_volume_pred(out_mirror["volume_preds"].float())

            # Mesh-subsample passes (independent uniform per surface / volume).
            sum_surface = torch.zeros_like(surface_pred_orig)
            sum_volume = torch.zeros_like(volume_pred_orig)
            cnt_surface = torch.zeros_like(batch.surface_mask, dtype=torch.float32)
            cnt_volume = torch.zeros_like(batch.volume_mask, dtype=torch.float32)
            for _ in range(subsample_passes):
                sub_surface_mask = make_subsample_mask(
                    batch.surface_mask, subsample_frac, gen
                )
                sub_volume_mask = make_subsample_mask(batch.volume_mask, subsample_frac, gen)
                with autocast_context(device, amp_mode):
                    out_sub = eval_module(
                        surface_x=batch.surface_x,
                        surface_mask=sub_surface_mask,
                        volume_x=batch.volume_x,
                        volume_mask=sub_volume_mask,
                    )
                sum_surface = sum_surface + out_sub["surface_preds"].float()
                sum_volume = sum_volume + out_sub["volume_preds"].float()
                cnt_surface = cnt_surface + sub_surface_mask.float()
                cnt_volume = cnt_volume + sub_volume_mask.float()

            # MC-coverage subsample average. For points never sampled (very rare, prob
            # = 0.2^4 = 0.16% at keep=0.8/passes=4), fall back to the orig pred so we
            # don't inject zeros into the TTA output.
            denom_surface = cnt_surface.clamp(min=1.0).unsqueeze(-1)
            denom_volume = cnt_volume.clamp(min=1.0).unsqueeze(-1)
            sub_surface_mc = sum_surface / denom_surface
            sub_volume_mc = sum_volume / denom_volume
            never_sampled_surf = (cnt_surface == 0).unsqueeze(-1)
            never_sampled_vol = (cnt_volume == 0).unsqueeze(-1)
            sub_surface_avg = torch.where(
                never_sampled_surf, surface_pred_orig, sub_surface_mc
            )
            sub_volume_avg = torch.where(
                never_sampled_vol, volume_pred_orig, sub_volume_mc
            )

            # Compose the 3 final TTA modes.
            mirror_only_s = 0.5 * (surface_pred_orig + surface_pred_mirror)
            mirror_only_v = 0.5 * (volume_pred_orig + volume_pred_mirror)
            subsample_only_s = 0.5 * (surface_pred_orig + sub_surface_avg)
            subsample_only_v = 0.5 * (volume_pred_orig + sub_volume_avg)
            mxs_s = (surface_pred_orig + surface_pred_mirror + sub_surface_avg) / 3.0
            mxs_v = (volume_pred_orig + volume_pred_mirror + sub_volume_avg) / 3.0

            preds_by_mode: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "orig": (surface_pred_orig, volume_pred_orig),
                "mirror": (surface_pred_mirror, volume_pred_mirror),
                "subsample_avg": (sub_surface_avg, sub_volume_avg),
                "mirror_only": (mirror_only_s, mirror_only_v),
                "subsample_only": (subsample_only_s, subsample_only_v),
                "mirror_x_subsample": (mxs_s, mxs_v),
            }
            for mode_name, (sp, vp) in preds_by_mode.items():
                _accumulate_outputs(accs[mode_name], batch, sp, vp, transform)

    finalized: dict[str, dict[str, float]] = {}
    for mode_name in TTA_MODES:
        acc = accs[mode_name]
        if distributed_state is not None and distributed_state.enabled:
            gathered: list[EvalAccumulator | None] = [None for _ in range(distributed_state.world_size)]
            dist.all_gather_object(gathered, acc)
            if distributed_state.is_main:
                merged = merge_eval_accumulators(g for g in gathered if g is not None)
                finalized[mode_name] = finalize_eval_accumulator(merged)
            else:
                finalized[mode_name] = {}
        else:
            finalized[mode_name] = finalize_eval_accumulator(acc)
    return finalized


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device
    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        print(f"Checkpoint: {cfg.checkpoint}")
        print(f"Subsample frac={cfg.subsample_frac} passes={cfg.subsample_passes}")

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
        run_name = cfg.wandb_name or f"{cfg.agent}/h239-h148-mesh-subsample"
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            group=cfg.wandb_group,
            name=run_name,
            config={
                **asdict(cfg),
                "checkpoint_run_id": cfg.run_id,
                "checkpoint_epoch": ck.get("epoch"),
                "tta_modes": list(TTA_MODES),
            },
            tags=["h239", "tta", "mesh-subsample", "mirror-aug", "eval-only", cfg.agent],
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
        # Stagger the per-rank seed by split so val/test see different subsamples.
        rank_seed = cfg.seed + state.rank * 1000 + (0 if name == "val_surface" else 500)
        finalized = evaluate_tta_split(
            model=model,
            loader=loader,
            transform=transform,
            device=device,
            amp_mode=cfg.amp_mode,
            distributed_state=state,
            subsample_frac=cfg.subsample_frac,
            subsample_passes=cfg.subsample_passes,
            rank_seed=rank_seed,
        )
        dt = time.time() - t0
        if state.is_main:
            print(f"  done in {dt:.1f}s")
            for mode_name in TTA_MODES:
                print(f"  -- {mode_name} --")
                print_metrics(name, finalized[mode_name])
            summary[name] = finalized

            log_obj: dict[str, float] = {}
            for mode_name in TTA_MODES:
                metrics = finalized[mode_name]
                log_obj.update(primary_metric_log(f"{log_prefix}_primary/{mode_name}", metrics))
                log_obj.update({f"{log_prefix}_extra/{mode_name}/loss": metrics["loss"]})
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
            for mode_name in TTA_MODES:
                header += f" {mode_name:>20s}"
            print(header)
            for k in keys:
                row = f"  {k:<36s}"
                for mode_name in TTA_MODES:
                    row += f" {modes[mode_name][k]:>20.4f}"
                print(row)

        if run is not None:
            for split, modes in summary.items():
                for mode_name, metrics in modes.items():
                    for k, v in metrics.items():
                        try:
                            run.summary[f"{split}/{mode_name}/{k}"] = float(v)
                        except Exception:
                            pass
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
