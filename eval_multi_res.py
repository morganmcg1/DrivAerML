# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H236: Multi-resolution TTA — eval at varied volume point counts and average.

For each case, accumulates per-point predictions across multiple
``eval_volume_points`` resolutions (optionally combined with mirror TTA),
averages them per-point in real space, and reports the standard relL2/MAE
metrics against the unchanged ground truth.

Cases are sharded across DDP ranks (case_ids[rank::world_size]); each rank
processes its assigned cases sequentially with single-case loaders. Per-case
scratch buffers live on CPU (~200MB per case).

Modes:
    res_avg:        average over N resolutions, no mirror
    mirror_res_avg: 2N-pass = orig + mirror at each of N resolutions

Mirror convention (H148/H183/H209):
    surface_x [x, y, z, nx, ny, nz, area] -> negate y(1) and ny(4)
    volume_x  [x, y, z, sdf]              -> negate y(1)
    surface_y predictions [cp, tau_x, tau_y, tau_z] -> un-mirror by negating tau_y(2)
    volume_y predictions [volume_pressure] -> invariant
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb

from data import SurfaceBatch, pad_collate
from data.loader import DrivAerMLCaseStore, DrivAerMLSurfaceDataset
from model import SurfaceTransolver
from trainer_runtime import (
    EVAL_KEYS,
    EvalAccumulator,
    TargetTransform,
    _accumulate_case_rel_l2,
    _masked_sse_count,
    autocast_context,
    cleanup_distributed,
    finalize_eval_accumulator,
    init_distributed,
    merge_eval_accumulators,
    primary_metric_log,
    print_metrics,
    unwrap_model,
)


@dataclass
class EvalConfig:
    """H236 eval config. Defaults match the H185 yw2a5dyl training-time config."""

    # Checkpoint source — either a direct path (preferred when present) or
    # fetch from W&B by run_id + artifact alias.
    ckpt_path: str = ""  # if set, load this checkpoint directly (skip W&B download)
    run_id: str = "yw2a5dyl"
    checkpoint: str = "epoch-13"  # artifact alias; can also be "best"
    use_ema: bool = True  # validates that loaded checkpoint_source == "ema"
    cache_root: str = "outputs/h236_eval/_artifacts"

    # Multi-resolution TTA configuration
    resolutions: str = "49152,65536,81920"
    eval_modes: str = "res_avg,mirror_res_avg"  # comma-separated: orig, mirror, res_avg, mirror_res_avg
    # Surface resolution stays fixed; only volume is varied (except for orig/mirror modes,
    # which run at the single resolution ``eval_volume_points``).
    eval_surface_points: int = 65536
    eval_volume_points: int = 65536  # used by single-resolution modes (orig, mirror)

    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h236_eval"
    wandb_group: str = "h236-nezuko-multi-res-tta"
    wandb_name: str = ""
    wandb_project: str = "senpai-v1-drivaerml-ddp8"
    wandb_entity: str = "wandb-applied-ai-team"
    agent: str = "nezuko"

    # Eval-only loader params
    batch_size: int = 2
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
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
    debug: bool = False  # debug: 2 val + 2 test cases only


def parse_args(argv: Iterable[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="H236 multi-resolution TTA eval")
    defaults = EvalConfig()
    for f in fields(EvalConfig):
        v = getattr(defaults, f.name)
        dashed = f"--{f.name.replace('_', '-')}"
        underscored = f"--{f.name}"
        cli_args = [dashed] if dashed == underscored else [dashed, underscored]
        if isinstance(v, bool):
            parser.add_argument(*cli_args, action="store_true", default=v, dest=f.name)
            neg_dashed = f"--no-{f.name.replace('_', '-')}"
            neg_underscored = f"--no_{f.name}"
            neg_args = [neg_dashed] if neg_dashed == neg_underscored else [neg_dashed, neg_underscored]
            parser.add_argument(*neg_args, action="store_false", dest=f.name)
        else:
            parser.add_argument(*cli_args, type=type(v), default=v, dest=f.name)
    ns = parser.parse_args(argv)
    cfg = EvalConfig(**{f.name: getattr(ns, f.name) for f in fields(EvalConfig)})
    return cfg


def parse_rff_init_sigmas(spec: str) -> list[float] | None:
    if not spec:
        return None
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_resolutions(spec: str) -> list[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_modes(spec: str) -> list[str]:
    return [m.strip() for m in spec.split(",") if m.strip()]


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


def download_checkpoint(
    entity: str,
    project: str,
    run_id: str,
    alias: str,
    cache_root: Path,
) -> Path:
    """Download model artifact for run_id with the given alias (e.g. 'best', 'epoch-13')."""
    cache_dir = cache_root / run_id / alias
    ckpt_path = cache_dir / "checkpoint.pt"
    if ckpt_path.exists():
        return ckpt_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    matched = None
    for art in run.logged_artifacts():
        if art.type != "model":
            continue
        if alias in art.aliases:
            matched = art
            break
    if matched is None:
        avail = []
        for art in run.logged_artifacts():
            if art.type == "model":
                avail.append((art.name, list(art.aliases)))
        raise RuntimeError(
            f"No model artifact with alias '{alias}' for run {run_id}; "
            f"available={avail}"
        )
    matched.download(root=str(cache_dir))
    return ckpt_path


# --- mirror helpers (H209 convention) ---


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


def unmirror_surface_real(pred: torch.Tensor) -> torch.Tensor:
    """surface real-units [cp, tau_x, tau_y, tau_z]: negate tau_y to un-mirror."""
    out = pred.clone()
    out[..., 2].neg_()
    return out


# --- global indices for a chunk in eval_chunk mode ---


def chunk_global_indices(view_index: int, view_count: int, n_full: int) -> torch.Tensor:
    """Reconstruct the global stride indices for a chunk emitted by ``eval_chunk`` sampling."""
    if view_index >= view_count:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(view_index, n_full, view_count, dtype=torch.long)


# --- per-case multi-resolution TTA ---


@torch.no_grad()
def process_case_multires(
    *,
    case_id: str,
    model: nn.Module,
    transform: TargetTransform,
    device: torch.device,
    store: DrivAerMLCaseStore,
    cfg: EvalConfig,
    resolutions: list[int],
    use_mirror: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, dict]:
    """Run multi-res TTA passes for one case and return per-point averaged predictions.

    Returns (surface_avg_real[n_surf, 4], volume_avg_real[n_vol, 1],
             surface_y[n_surf, 4], volume_y[n_vol, 1], n_passes, timing).
    Predictions are in **real units** (already denormalized).
    """
    counts = store.case_point_counts(case_id)
    n_surf = counts["n_surface"]
    n_vol = counts["n_volume"]

    surf_sum = torch.zeros(n_surf, 4, dtype=torch.float32)
    surf_cnt = torch.zeros(n_surf, dtype=torch.int32)
    vol_sum = torch.zeros(n_vol, 1, dtype=torch.float32)
    vol_cnt = torch.zeros(n_vol, dtype=torch.int32)

    eval_module = unwrap_model(model)
    timing: dict[str, float] = {"forward_seconds": 0.0, "io_seconds": 0.0, "n_forwards": 0}

    for K in resolutions:
        dataset = DrivAerMLSurfaceDataset(
            case_ids=[case_id],
            store=store,
            max_surface_points=cfg.eval_surface_points,
            max_volume_points=K,
            sampling_mode="eval_chunk",
        )
        loader_iter = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=pad_collate,
            persistent_workers=False,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        )

        mirror_flags = (False, True) if use_mirror else (False,)
        for batch in loader_iter:
            batch = batch.to(device)
            metas_cached = list(batch.metadata)
            for mirror in mirror_flags:
                t0 = time.time()
                model_input = mirror_inputs(batch) if mirror else batch

                with autocast_context(device, cfg.amp_mode):
                    out = eval_module(
                        surface_x=model_input.surface_x,
                        surface_mask=model_input.surface_mask,
                        volume_x=model_input.volume_x,
                        volume_mask=model_input.volume_mask,
                    )
                surface_pred_real = transform.invert_surface(out["surface_preds"].float())
                volume_pred_real = transform.invert_volume(out["volume_preds"].float())
                if mirror:
                    surface_pred_real = unmirror_surface_real(surface_pred_real)
                    # volume_pressure is scalar and invariant under y-mirror.

                timing["forward_seconds"] += time.time() - t0
                timing["n_forwards"] += int(batch.surface_x.shape[0])

                t1 = time.time()
                surface_pred_real_cpu = surface_pred_real.detach().cpu()
                volume_pred_real_cpu = volume_pred_real.detach().cpu()
                surface_mask_cpu = batch.surface_mask.detach().cpu()
                volume_mask_cpu = batch.volume_mask.detach().cpu()
                for i in range(len(batch.case_ids)):
                    meta = metas_cached[i]
                    s_view_idx = int(meta["surface_view_index"])
                    s_view_count = int(meta["surface_view_count"])
                    v_view_idx = int(meta["volume_view_index"])
                    v_view_count = int(meta["volume_view_count"])

                    if s_view_idx < s_view_count:
                        s_global = chunk_global_indices(s_view_idx, s_view_count, n_surf)
                        s_mask_i = surface_mask_cpu[i].bool()
                        s_chunk = surface_pred_real_cpu[i][s_mask_i]
                        if s_global.shape[0] != s_chunk.shape[0]:
                            raise RuntimeError(
                                f"surface global-index mismatch case={case_id} K={K} "
                                f"view_idx={s_view_idx}/{s_view_count}: "
                                f"global={s_global.shape[0]} vs chunk={s_chunk.shape[0]}"
                            )
                        if s_global.numel() > 0:
                            surf_sum.index_add_(0, s_global, s_chunk)
                            surf_cnt.index_add_(
                                0, s_global, torch.ones_like(s_global, dtype=torch.int32)
                            )

                    if v_view_idx < v_view_count:
                        v_global = chunk_global_indices(v_view_idx, v_view_count, n_vol)
                        v_mask_i = volume_mask_cpu[i].bool()
                        v_chunk = volume_pred_real_cpu[i][v_mask_i]
                        if v_global.shape[0] != v_chunk.shape[0]:
                            raise RuntimeError(
                                f"volume global-index mismatch case={case_id} K={K} "
                                f"view_idx={v_view_idx}/{v_view_count}: "
                                f"global={v_global.shape[0]} vs chunk={v_chunk.shape[0]}"
                            )
                        if v_global.numel() > 0:
                            vol_sum.index_add_(0, v_global, v_chunk)
                            vol_cnt.index_add_(
                                0, v_global, torch.ones_like(v_global, dtype=torch.int32)
                            )
                timing["io_seconds"] += time.time() - t1

    if int(surf_cnt.min()) == 0:
        raise RuntimeError(
            f"surface coverage incomplete for case {case_id}: "
            f"{int((surf_cnt == 0).sum())} points have zero contributing chunks"
        )
    if int(vol_cnt.min()) == 0:
        raise RuntimeError(
            f"volume coverage incomplete for case {case_id}: "
            f"{int((vol_cnt == 0).sum())} points have zero contributing chunks"
        )

    surf_avg = surf_sum / surf_cnt.unsqueeze(-1).to(torch.float32)
    vol_avg = vol_sum / vol_cnt.unsqueeze(-1).to(torch.float32)

    # Load full ground truth once for metric computation.
    case = store.load_case(case_id)
    surf_y = case.surface_y.float()
    vol_y = case.volume_y.float()

    n_passes = len(resolutions) * (2 if use_mirror else 1)
    return surf_avg, vol_avg, surf_y, vol_y, n_passes, timing


def add_case_to_accumulator(
    acc: EvalAccumulator,
    *,
    case_id: str,
    surface_pred_real: torch.Tensor,
    volume_pred_real: torch.Tensor,
    surface_y: torch.Tensor,
    volume_y: torch.Tensor,
    transform: TargetTransform,
) -> None:
    """Accumulate metrics for one fully-predicted case into ``acc``.

    Mirrors the accumulation logic from ``_accumulate_outputs`` in eval_tta_h209.py
    but for a single full-case prediction (no chunk masks: every point is valid).
    """
    surface_pred_norm = transform.apply_surface(surface_pred_real)
    volume_pred_norm = transform.apply_volume(volume_pred_real)
    surface_target_norm = transform.apply_surface(surface_y)
    volume_target_norm = transform.apply_volume(volume_y)

    mask_surface = torch.ones(surface_pred_norm.shape[0], dtype=torch.bool)
    mask_volume = torch.ones(volume_pred_norm.shape[0], dtype=torch.bool)
    surface_sse, surface_count = _masked_sse_count(
        surface_pred_norm, surface_target_norm, mask_surface
    )
    volume_sse, volume_count = _masked_sse_count(
        volume_pred_norm, volume_target_norm, mask_volume
    )
    acc.surface_loss_sse += surface_sse
    acc.surface_loss_count += surface_count
    acc.volume_loss_sse += volume_sse
    acc.volume_loss_count += volume_count

    surface_abs = (surface_pred_real - surface_y).abs()
    acc.abs_sums["surface_pressure"] += float(surface_abs[:, 0].sum().item())
    acc.abs_counts["surface_pressure"] += int(surface_abs[:, 0].numel())
    wall_abs = surface_abs[:, 1:4]
    acc.abs_sums["wall_shear"] += float(wall_abs.sum().item())
    acc.abs_counts["wall_shear"] += int(wall_abs.numel())
    for offset, axis in enumerate(("x", "y", "z")):
        channel = wall_abs[:, offset]
        acc.abs_sums[f"wall_shear_{axis}"] += float(channel.sum().item())
        acc.abs_counts[f"wall_shear_{axis}"] += int(channel.numel())
    wall_vector_error = torch.linalg.vector_norm(
        surface_pred_real[:, 1:4] - surface_y[:, 1:4],
        dim=-1,
    )
    acc.wall_shear_vector_abs_sum += float(wall_vector_error.sum().item())
    acc.wall_shear_vector_count += int(wall_vector_error.numel())

    volume_abs = (volume_pred_real - volume_y).abs()
    acc.abs_sums["volume_pressure"] += float(volume_abs[:, 0].sum().item())
    acc.abs_counts["volume_pressure"] += int(volume_abs[:, 0].numel())

    _accumulate_case_rel_l2(
        acc.case_sums["surface_pressure"],
        case_id=case_id,
        pred=surface_pred_real[:, 0:1],
        target=surface_y[:, 0:1],
    )
    _accumulate_case_rel_l2(
        acc.case_sums["wall_shear"],
        case_id=case_id,
        pred=surface_pred_real[:, 1:4],
        target=surface_y[:, 1:4],
    )
    for channel, axis in enumerate(("x", "y", "z"), start=1):
        _accumulate_case_rel_l2(
            acc.case_sums[f"wall_shear_{axis}"],
            case_id=case_id,
            pred=surface_pred_real[:, channel : channel + 1],
            target=surface_y[:, channel : channel + 1],
        )
    _accumulate_case_rel_l2(
        acc.case_sums["volume_pressure"],
        case_id=case_id,
        pred=volume_pred_real,
        target=volume_y,
    )


def evaluate_split_multires(
    *,
    split_name: str,
    case_ids: list[str],
    model: nn.Module,
    transform: TargetTransform,
    device: torch.device,
    store: DrivAerMLCaseStore,
    cfg: EvalConfig,
    resolutions: list[int],
    use_mirror: bool,
    distributed_state,
) -> dict[str, float]:
    """Distribute cases across ranks, run multi-res TTA per case, return finalized metrics."""
    rank = distributed_state.rank if distributed_state and distributed_state.enabled else 0
    world_size = (
        distributed_state.world_size
        if distributed_state and distributed_state.enabled
        else 1
    )
    my_cases = case_ids[rank::world_size]

    acc = EvalAccumulator()
    total_t0 = time.time()
    total_forward = 0.0
    total_io = 0.0
    total_forwards = 0
    for case_idx, case_id in enumerate(my_cases):
        t0 = time.time()
        surf_avg, vol_avg, surf_y, vol_y, n_passes, timing = process_case_multires(
            case_id=case_id,
            model=model,
            transform=transform,
            device=device,
            store=store,
            cfg=cfg,
            resolutions=resolutions,
            use_mirror=use_mirror,
        )
        add_case_to_accumulator(
            acc,
            case_id=case_id,
            surface_pred_real=surf_avg,
            volume_pred_real=vol_avg,
            surface_y=surf_y,
            volume_y=vol_y,
            transform=transform,
        )
        total_forward += timing["forward_seconds"]
        total_io += timing["io_seconds"]
        total_forwards += timing["n_forwards"]
        dt = time.time() - t0
        print(
            f"  [rank {rank}] {split_name} {case_id} ({case_idx + 1}/{len(my_cases)}): "
            f"n_passes={n_passes} forwards={timing['n_forwards']} "
            f"forward={timing['forward_seconds']:.1f}s io={timing['io_seconds']:.1f}s "
            f"total={dt:.1f}s",
            flush=True,
        )

    print(
        f"[rank {rank}] {split_name} complete: {len(my_cases)} cases in "
        f"{time.time() - total_t0:.1f}s "
        f"(forward={total_forward:.1f}s io={total_io:.1f}s n_forwards={total_forwards})",
        flush=True,
    )

    # Gather accumulators across ranks.
    if distributed_state is not None and distributed_state.enabled:
        gathered: list[EvalAccumulator | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, acc)
        if not distributed_state.is_main:
            return {}
        merged = merge_eval_accumulators(g for g in gathered if g is not None)
    else:
        merged = acc

    return finalize_eval_accumulator(merged)


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device

    resolutions = parse_resolutions(cfg.resolutions)
    modes = parse_modes(cfg.eval_modes)
    valid_modes = ("orig", "mirror", "res_avg", "mirror_res_avg")
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Unknown eval mode: {mode!r} (valid: {valid_modes})")

    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        if cfg.ckpt_path:
            print(f"Source ckpt_path={cfg.ckpt_path} use_ema={cfg.use_ema}")
        else:
            print(f"Source run_id={cfg.run_id} alias={cfg.checkpoint} use_ema={cfg.use_ema}")
        print(f"Resolutions: {resolutions}")
        print(f"Single-res volume points (for orig/mirror): {cfg.eval_volume_points}")
        print(f"Modes: {modes}")

    # Resolve checkpoint path on rank 0 only, then broadcast.
    cache_root = Path(cfg.cache_root)
    if state.is_main:
        if cfg.ckpt_path:
            ckpt_path = Path(cfg.ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"--ckpt-path not found: {ckpt_path}")
        else:
            ckpt_path = download_checkpoint(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                run_id=cfg.run_id,
                alias=cfg.checkpoint,
                cache_root=cache_root,
            )
        ckpt_path_str = str(ckpt_path)
    else:
        ckpt_path_str = ""
    if state.enabled:
        obj = [ckpt_path_str]
        dist.broadcast_object_list(obj, src=0)
        ckpt_path_str = obj[0]
    ckpt_path = Path(ckpt_path_str)

    # Build store and normalization stats (matching make_loaders / load_data).
    store = DrivAerMLCaseStore(
        manifest_path=cfg.manifest, root=cfg.data_root or None
    )
    from data.loader import target_stats_from_normalizers  # noqa: PLC0415 — late import
    stats = target_stats_from_normalizers(store)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    # Build model.
    model = build_model(cfg).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ck["model"]
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if state.is_main:
        print(f"Loaded checkpoint epoch={ck.get('epoch')} source={ck.get('checkpoint_source')}")
        if cfg.use_ema and ck.get("checkpoint_source") != "ema":
            print(
                f"  WARNING: --use_ema set but checkpoint_source={ck.get('checkpoint_source')!r}"
            )
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  missing[:5]={missing[:5]}")
        if unexpected:
            print(f"  unexpected[:5]={unexpected[:5]}")
    model.eval()

    val_case_ids = store.case_ids("val")
    test_case_ids = store.case_ids("test")
    if cfg.debug:
        val_case_ids = val_case_ids[:2]
        test_case_ids = test_case_ids[:2]
        if state.is_main:
            print(f"DEBUG: limiting to {len(val_case_ids)} val + {len(test_case_ids)} test cases")

    # W&B init on rank 0.
    run = None
    if state.is_main:
        run_name = cfg.wandb_name or f"{cfg.agent}/h236-multi-res-tta-h185-ep{ck.get('epoch')}"
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", cfg.wandb_project),
            entity=os.environ.get("WANDB_ENTITY", cfg.wandb_entity),
            group=cfg.wandb_group,
            name=run_name,
            config={
                **asdict(cfg),
                "checkpoint_run_id": cfg.run_id,
                "checkpoint_epoch": ck.get("epoch"),
                "checkpoint_source": ck.get("checkpoint_source"),
                "resolutions_list": resolutions,
                "modes_list": modes,
                "n_val_cases": len(val_case_ids),
                "n_test_cases": len(test_case_ids),
            },
            tags=[
                "h236",
                "tta",
                "multi-res",
                "eval-only",
                cfg.agent,
            ],
            reinit="finish_previous",
        )

    splits = [
        ("val_surface", val_case_ids, "full_val"),
        ("test_surface", test_case_ids, "test"),
    ]

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, case_ids, log_prefix in splits:
        summary[split_name] = {}
        for mode in modes:
            use_mirror = mode in ("mirror", "mirror_res_avg")
            if mode in ("orig", "mirror"):
                mode_resolutions = [cfg.eval_volume_points]
            else:
                mode_resolutions = resolutions
            if state.is_main:
                print(
                    f"\n=== Evaluating split={split_name} mode={mode} "
                    f"resolutions={mode_resolutions} mirror={use_mirror} ===",
                    flush=True,
                )
            t0 = time.time()
            metrics = evaluate_split_multires(
                split_name=f"{split_name}/{mode}",
                case_ids=case_ids,
                model=model,
                transform=transform,
                device=device,
                store=store,
                cfg=cfg,
                resolutions=mode_resolutions,
                use_mirror=use_mirror,
                distributed_state=state,
            )
            dt = time.time() - t0
            if state.is_main:
                print(f"  total {split_name}/{mode}: {dt:.1f}s")
                print_metrics(f"{split_name}/{mode}", metrics)
                summary[split_name][mode] = metrics

                log_obj: dict[str, float] = {}
                log_obj.update(primary_metric_log(f"{log_prefix}_primary/{mode}", metrics))
                log_obj.update({f"{log_prefix}_extra/{mode}/loss": metrics["loss"]})
                log_obj[f"{log_prefix}_extra/{mode}/seconds"] = dt
                if run is not None:
                    wandb.log(log_obj)

    if state.is_main and summary:
        print("\n=== Summary (rel_l2_pct lower-is-better) ===")
        for split, modes_dict in summary.items():
            print(f"\n[{split}]")
            keys_to_show = (
                "abupt_axis_mean_rel_l2_pct",
                "surface_pressure_rel_l2_pct",
                "wall_shear_rel_l2_pct",
                "wall_shear_x_rel_l2_pct",
                "wall_shear_y_rel_l2_pct",
                "wall_shear_z_rel_l2_pct",
                "volume_pressure_rel_l2_pct",
            )
            mode_keys = list(modes_dict.keys())
            header = f"  {'metric':<36s} " + " ".join(f"{m:>16s}" for m in mode_keys)
            print(header)
            for k in keys_to_show:
                row = f"  {k:<36s} " + " ".join(
                    f"{modes_dict[m][k]:>16.4f}" for m in mode_keys
                )
                print(row)

        if run is not None:
            for split, modes_dict in summary.items():
                for mode, metrics in modes_dict.items():
                    for k, v in metrics.items():
                        try:
                            run.summary[f"{split}/{mode}/{k}"] = float(v)
                        except Exception:
                            pass
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
