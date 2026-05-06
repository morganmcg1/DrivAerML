# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Phase 0 diagnostic: per-case + per-region test_vol_p decomposition.

Loads a single checkpoint by W&B run_id from either the ddp8 or yi project,
runs deterministic eval-chunk inference over the test split, and records:

- per-case error_sq / target_sq for surface_pressure, wall_shear, tau_x/y/z, volume_pressure
- per-(case, region) error_sq / target_sq for x_rel / z_rel regions and SDF bands
- aggregate point counts per region

Output: ``analysis/per_case_test_predictions_<run_id>.npz`` which contains all
quantities needed to reconstruct Tables A-E without re-running inference.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_data, pad_collate
from model import (
    SurfaceTransolver,
    StringSeparableEncoding,
    ContinuousSincosEmbed,
)
from trainer_runtime import (
    EvalAccumulator,
    TargetTransform,
    autocast_context,
    finalize_eval_accumulator,
)
from ensemble_eval import (
    parse_rff_init_sigmas,
    build_model_from_config,
    download_checkpoint,
    make_eval_loader,
)


# ---------------------------------------------------------------------------
# yi-compat learnable position encoding
# ---------------------------------------------------------------------------


class LearnableContinuousSincosEmbed(nn.Module):
    """Replicates the yi-project ContinuousSincosEmbed(learnable=True) layout.

    Stored params match the legacy state_dict keys ``pos_embed.log_freq`` and
    ``pos_embed.phase`` so a yi-style checkpoint loads without rewiring.
    """

    def __init__(self, hidden_dim: int, input_dim: int, max_wavelength: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        padding = hidden_dim % input_dim
        dim_per_axis = (hidden_dim - padding) // input_dim
        sincos_padding = dim_per_axis % 2
        self.padding = padding + sincos_padding * input_dim
        effective_dim_per_axis = (hidden_dim - self.padding) // input_dim
        if effective_dim_per_axis <= 0:
            raise ValueError("hidden_dim must be large enough for the requested input dimension")
        arange = torch.arange(0, effective_dim_per_axis, 2, dtype=torch.float32)
        init_omega = 1.0 / max_wavelength ** (arange / effective_dim_per_axis)
        init_log = torch.log(init_omega)
        self.log_freq = nn.Parameter(init_log.unsqueeze(0).expand(input_dim, -1).clone())
        self.phase = nn.Parameter(torch.zeros(input_dim, len(arange)))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.float()
        omega = torch.exp(self.log_freq)
        out = coords.unsqueeze(-1) * omega
        out = out + self.phase
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        emb = emb.flatten(start_dim=-2)
        if self.padding > 0:
            pad = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=-1)
        return emb


def is_yi_style_state_dict(state_dict: dict) -> bool:
    return "pos_embed.log_freq" in state_dict and "pos_embed.phase" in state_dict


def build_model_for_checkpoint(config: dict, state_dict: dict) -> SurfaceTransolver:
    """Build a model that matches the saved state_dict.

    Falls back to the yi-style learnable position embedding when the checkpoint
    contains ``pos_embed.log_freq`` (yi project legacy layout).
    """

    if is_yi_style_state_dict(state_dict):
        # yi-style: learnable_pe ContinuousSincosEmbed, no RFF, no string-sep,
        # no qk_norm. Build with sincos defaults then replace pos_embed.
        model = SurfaceTransolver(
            n_layers=int(config.get("model_layers", 4)),
            n_hidden=int(config.get("model_hidden_dim", 512)),
            dropout=float(config.get("model_dropout", 0.0)),
            n_head=int(config.get("model_heads", 8)),
            mlp_ratio=int(config.get("model_mlp_ratio", 4)),
            slice_num=int(config.get("model_slices", 128)),
            rff_num_features=0,
            rff_sigma=1.0,
            rff_init_sigmas=None,
            pos_encoding_mode="sincos",
            use_qk_norm=False,
        )
        max_wavelength = int(config.get("pos_max_wavelength", 1000))
        model.pos_embed = LearnableContinuousSincosEmbed(
            hidden_dim=int(config.get("model_hidden_dim", 512)),
            input_dim=3,
            max_wavelength=max_wavelength,
        )
        return model
    return build_model_from_config(config)


def load_checkpoint_for_diagnostic(
    run_id: str,
    artifact_dir: Path,
    device: torch.device,
) -> tuple[SurfaceTransolver, dict]:
    config_path = artifact_dir / "config.yaml"
    checkpoint_path = artifact_dir / "checkpoint.pt"
    with config_path.open("r") as fh:
        config = yaml.safe_load(fh)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise RuntimeError(f"Checkpoint for run {run_id} missing 'model' state dict")
    state_dict = {k: v for k, v in ckpt["model"].items()}
    model = build_model_for_checkpoint(config, state_dict).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch for {run_id}: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    print(
        f"Loaded {run_id}: epoch={ckpt.get('epoch')} src={ckpt.get('checkpoint_source')} "
        f"pos={'yi-learnable' if is_yi_style_state_dict(state_dict) else config.get('pos_encoding_mode')} "
        f"layers={config.get('model_layers')} hidden={config.get('model_hidden_dim')} "
        f"heads={config.get('model_heads')} qk_norm={config.get('use_qk_norm')}"
    )
    return model, config


# ---------------------------------------------------------------------------
# Region geometry
# ---------------------------------------------------------------------------


# Vehicle-dataset-mean bbox values from #763 (a4c516e fallback).
DATASET_MEAN_CX = 1.5046
DATASET_MEAN_CZ = 0.3942
DATASET_MEAN_S_REF = 4.6720

# Region masks operate on:
#   x_rel = (x - cx) / s_ref
#   z_rel = (z - cz) / s_ref
# Region definitions (from PR #767):
#   upstream:  x_rel <= 0.5
#   near-wake: 0.5 < x_rel < 3.0 AND |z_rel| < 1.5
#   far-wake:  x_rel >= 3.0
#   under-body: z_rel < 0
#   roof:       z_rel >= 1.0
# SDF bands operate on raw |sdf| in metres:
#   near-surface:    |sdf| <= 0.05
#   boundary-layer:  0.05 < |sdf| <= 0.5
#   far-field:       |sdf| > 0.5

X_REL_REGIONS = ("upstream", "near_wake", "far_wake", "under_body", "roof")
SDF_BANDS = ("near_surface", "boundary_layer", "far_field")


def compute_xz_region_masks(volume_xyz: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return a dict of bool masks shape [N] for each region label."""

    x = volume_xyz[..., 0]
    z = volume_xyz[..., 2]
    x_rel = (x - DATASET_MEAN_CX) / DATASET_MEAN_S_REF
    z_rel = (z - DATASET_MEAN_CZ) / DATASET_MEAN_S_REF
    upstream = x_rel <= 0.5
    far_wake = x_rel >= 3.0
    near_wake = (x_rel > 0.5) & (x_rel < 3.0) & (z_rel.abs() < 1.5)
    under_body = z_rel < 0
    roof = z_rel >= 1.0
    return {
        "upstream": upstream,
        "near_wake": near_wake,
        "far_wake": far_wake,
        "under_body": under_body,
        "roof": roof,
    }


def compute_sdf_band_masks(volume_sdf: torch.Tensor) -> dict[str, torch.Tensor]:
    abs_sdf = volume_sdf.abs()
    near = abs_sdf <= 0.05
    layer = (abs_sdf > 0.05) & (abs_sdf <= 0.5)
    far = abs_sdf > 0.5
    return {
        "near_surface": near,
        "boundary_layer": layer,
        "far_field": far,
    }


# ---------------------------------------------------------------------------
# Per-case + per-region accumulators
# ---------------------------------------------------------------------------


class DiagnosticAccumulator:
    """Accumulates per-case and per-region squared errors / squared targets.

    Storage shape:
      case_metric_sums[case_id][metric] = [sum_err_sq, sum_target_sq, point_count]
      region_sums[region][case_id] = [sum_err_sq, sum_target_sq, point_count]
      sdf_sums[band][case_id] = [sum_err_sq, sum_target_sq, point_count]

    All metrics are computed in original target units after ``transform.invert_*``.
    """

    SURFACE_METRICS = ("surface_pressure", "wall_shear", "wall_shear_x", "wall_shear_y", "wall_shear_z")
    VOLUME_METRICS = ("volume_pressure",)

    def __init__(self) -> None:
        self.case_metric_sums: dict[str, dict[str, list[float]]] = {}
        self.region_sums: dict[str, dict[str, list[float]]] = {
            r: {} for r in X_REL_REGIONS
        }
        self.sdf_sums: dict[str, dict[str, list[float]]] = {
            b: {} for b in SDF_BANDS
        }
        self.region_total_points = {r: 0 for r in X_REL_REGIONS}
        self.sdf_total_points = {b: 0 for b in SDF_BANDS}
        self.total_volume_points = 0

    def _bump_metric(self, store: dict[str, list[float]], err_sq: float, tgt_sq: float, count: int) -> None:
        if count == 0:
            return
        if not store:
            store.extend([0.0, 0.0, 0])
        store[0] += err_sq
        store[1] += tgt_sq
        store[2] += count

    def _store(self, top: dict, case_id: str) -> list:
        return top.setdefault(case_id, [])

    def update(
        self,
        *,
        case_id: str,
        surface_pred: torch.Tensor,  # [N_surface_valid, 4] denorm
        surface_target: torch.Tensor,  # [N_surface_valid, 4] denorm
        volume_pred: torch.Tensor,  # [N_volume_valid, 1] denorm
        volume_target: torch.Tensor,  # [N_volume_valid, 1] denorm
        volume_xyz: torch.Tensor,  # [N_volume_valid, 3]
        volume_sdf: torch.Tensor,  # [N_volume_valid]
    ) -> None:
        per_case = self.case_metric_sums.setdefault(case_id, {m: [] for m in (*self.SURFACE_METRICS, *self.VOLUME_METRICS)})

        # ---- Surface per-case metrics ----
        if surface_target.numel() > 0:
            sp_pred = surface_pred[:, 0:1]
            sp_tgt = surface_target[:, 0:1]
            err = (sp_pred - sp_tgt).float().square().sum().item()
            tgt = sp_tgt.float().square().sum().item()
            self._bump_metric(per_case["surface_pressure"], err, tgt, sp_pred.numel())

            ws_pred = surface_pred[:, 1:4]
            ws_tgt = surface_target[:, 1:4]
            err = (ws_pred - ws_tgt).float().square().sum().item()
            tgt = ws_tgt.float().square().sum().item()
            self._bump_metric(per_case["wall_shear"], err, tgt, ws_pred.numel())

            for ch, axis in enumerate(("x", "y", "z"), start=1):
                ax_pred = surface_pred[:, ch:ch + 1]
                ax_tgt = surface_target[:, ch:ch + 1]
                err = (ax_pred - ax_tgt).float().square().sum().item()
                tgt = ax_tgt.float().square().sum().item()
                self._bump_metric(per_case[f"wall_shear_{axis}"], err, tgt, ax_pred.numel())

        # ---- Volume per-case metrics ----
        if volume_target.numel() > 0:
            err_full = (volume_pred - volume_target).float().square()
            tgt_full = volume_target.float().square()
            err = err_full.sum().item()
            tgt = tgt_full.sum().item()
            self._bump_metric(per_case["volume_pressure"], err, tgt, volume_pred.numel())
            self.total_volume_points += int(volume_pred.numel())

            err_per_pt = err_full[:, 0]
            tgt_per_pt = tgt_full[:, 0]

            # ---- Per-region (x_rel/z_rel) ----
            xz_masks = compute_xz_region_masks(volume_xyz)
            for region, mask in xz_masks.items():
                if not bool(mask.any()):
                    continue
                err_r = err_per_pt[mask].sum().item()
                tgt_r = tgt_per_pt[mask].sum().item()
                cnt_r = int(mask.sum().item())
                store = self._store(self.region_sums[region], case_id)
                self._bump_metric(store, err_r, tgt_r, cnt_r)
                self.region_total_points[region] += cnt_r

            # ---- Per-SDF-band ----
            sdf_masks = compute_sdf_band_masks(volume_sdf)
            for band, mask in sdf_masks.items():
                if not bool(mask.any()):
                    continue
                err_r = err_per_pt[mask].sum().item()
                tgt_r = tgt_per_pt[mask].sum().item()
                cnt_r = int(mask.sum().item())
                store = self._store(self.sdf_sums[band], case_id)
                self._bump_metric(store, err_r, tgt_r, cnt_r)
                self.sdf_total_points[band] += cnt_r


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_test_eval(
    *,
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    log_every: int = 50,
) -> tuple[DiagnosticAccumulator, EvalAccumulator, dict[str, float]]:
    diag = DiagnosticAccumulator()
    full_acc = EvalAccumulator()
    n_batches = len(loader)
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        with autocast_context(device, amp_mode):
            out = model(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
        surface_pred_norm = out["surface_preds"].float()
        volume_pred_norm = out["volume_preds"].float()
        surface_target_norm = transform.apply_surface(batch.surface_y)
        volume_target_norm = transform.apply_volume(batch.volume_y)

        # Aggregate-equivalent metrics (mirrors trainer_runtime accumulate_eval_batch).
        from trainer_runtime import _masked_sse_count
        s_sse, s_count = _masked_sse_count(surface_pred_norm, surface_target_norm, batch.surface_mask)
        v_sse, v_count = _masked_sse_count(volume_pred_norm, volume_target_norm, batch.volume_mask)
        full_acc.surface_loss_sse += s_sse
        full_acc.surface_loss_count += s_count
        full_acc.volume_loss_sse += v_sse
        full_acc.volume_loss_count += v_count

        surface_pred = transform.invert_surface(surface_pred_norm)
        volume_pred = transform.invert_volume(volume_pred_norm)

        if bool(batch.surface_mask.any()):
            surf_abs = (surface_pred - batch.surface_y).abs()
            valid_surf = surf_abs[batch.surface_mask]
            full_acc.abs_sums["surface_pressure"] += float(valid_surf[:, 0].sum().item())
            full_acc.abs_counts["surface_pressure"] += int(valid_surf[:, 0].numel())
            wall_abs = valid_surf[:, 1:4]
            full_acc.abs_sums["wall_shear"] += float(wall_abs.sum().item())
            full_acc.abs_counts["wall_shear"] += int(wall_abs.numel())
            for offset, axis in enumerate(("x", "y", "z")):
                ch = wall_abs[:, offset]
                full_acc.abs_sums[f"wall_shear_{axis}"] += float(ch.sum().item())
                full_acc.abs_counts[f"wall_shear_{axis}"] += int(ch.numel())
            wsv_err = torch.linalg.vector_norm(
                surface_pred[batch.surface_mask][:, 1:4] - batch.surface_y[batch.surface_mask][:, 1:4],
                dim=-1,
            )
            full_acc.wall_shear_vector_abs_sum += float(wsv_err.sum().item())
            full_acc.wall_shear_vector_count += int(wsv_err.numel())
        if bool(batch.volume_mask.any()):
            vol_abs = (volume_pred - batch.volume_y).abs()[batch.volume_mask]
            full_acc.abs_sums["volume_pressure"] += float(vol_abs[:, 0].sum().item())
            full_acc.abs_counts["volume_pressure"] += int(vol_abs[:, 0].numel())

        # Per-case + per-region diagnostic accumulation.
        for case_idx, case_id in enumerate(batch.case_ids):
            sm = batch.surface_mask[case_idx].bool()
            vm = batch.volume_mask[case_idx].bool()
            sp_valid = surface_pred[case_idx][sm] if bool(sm.any()) else surface_pred[case_idx][:0]
            st_valid = batch.surface_y[case_idx][sm] if bool(sm.any()) else batch.surface_y[case_idx][:0]
            vp_valid = volume_pred[case_idx][vm] if bool(vm.any()) else volume_pred[case_idx][:0]
            vt_valid = batch.volume_y[case_idx][vm] if bool(vm.any()) else batch.volume_y[case_idx][:0]
            v_xyz = batch.volume_x[case_idx, :, :3][vm] if bool(vm.any()) else batch.volume_x[case_idx, :0, :3]
            v_sdf = batch.volume_x[case_idx, :, 3][vm] if bool(vm.any()) else batch.volume_x[case_idx, :0, 3]

            # Also feed the standard EvalAccumulator's per-case rel_l2 path.
            from trainer_runtime import _accumulate_case_rel_l2
            if bool(sm.any()):
                _accumulate_case_rel_l2(
                    full_acc.case_sums["surface_pressure"], case_id=case_id,
                    pred=sp_valid[:, 0:1], target=st_valid[:, 0:1],
                )
                _accumulate_case_rel_l2(
                    full_acc.case_sums["wall_shear"], case_id=case_id,
                    pred=sp_valid[:, 1:4], target=st_valid[:, 1:4],
                )
                for ch, axis in enumerate(("x", "y", "z"), start=1):
                    _accumulate_case_rel_l2(
                        full_acc.case_sums[f"wall_shear_{axis}"], case_id=case_id,
                        pred=sp_valid[:, ch:ch + 1], target=st_valid[:, ch:ch + 1],
                    )
            if bool(vm.any()):
                _accumulate_case_rel_l2(
                    full_acc.case_sums["volume_pressure"], case_id=case_id,
                    pred=vp_valid, target=vt_valid,
                )

            diag.update(
                case_id=case_id,
                surface_pred=sp_valid,
                surface_target=st_valid,
                volume_pred=vp_valid,
                volume_target=vt_valid,
                volume_xyz=v_xyz,
                volume_sdf=v_sdf,
            )

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.time() - t0
            eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
            mem_gb = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0.0
            )
            print(
                f"  [{batch_idx + 1}/{n_batches}] {elapsed:.1f}s elapsed | ETA {eta:.0f}s | "
                f"peak mem {mem_gb:.1f} GB | seen cases={len(diag.case_metric_sums)}"
            )

    aggregate = finalize_eval_accumulator(full_acc)
    aggregate["_eval_seconds"] = time.time() - t0
    aggregate["_n_batches"] = float(n_batches)
    return diag, full_acc, aggregate


# ---------------------------------------------------------------------------
# Save / aggregate / output
# ---------------------------------------------------------------------------


def case_metric_to_rel_l2_pct(store: list[float]) -> float:
    if not store:
        return float("nan")
    err_sq, tgt_sq, _ = store
    if tgt_sq <= 0:
        return float("nan")
    return 100.0 * math.sqrt(err_sq / tgt_sq)


def save_diagnostic_outputs(
    *,
    out_path: Path,
    run_id: str,
    project: str,
    config: dict,
    diag: DiagnosticAccumulator,
    aggregate: dict[str, float],
) -> None:
    """Pickle all per-case, per-region, per-band sums into a single .npz."""

    case_ids = sorted(diag.case_metric_sums.keys())
    metric_names = list(diag.SURFACE_METRICS) + list(diag.VOLUME_METRICS)

    case_table = np.full(
        (len(case_ids), len(metric_names), 3), fill_value=np.nan, dtype=np.float64
    )
    for ci, case_id in enumerate(case_ids):
        for mi, metric in enumerate(metric_names):
            store = diag.case_metric_sums[case_id].get(metric, [])
            if store:
                case_table[ci, mi, 0] = store[0]
                case_table[ci, mi, 1] = store[1]
                case_table[ci, mi, 2] = store[2]

    region_table = np.full(
        (len(X_REL_REGIONS), len(case_ids), 3), fill_value=np.nan, dtype=np.float64
    )
    for ri, region in enumerate(X_REL_REGIONS):
        for ci, case_id in enumerate(case_ids):
            store = diag.region_sums[region].get(case_id, [])
            if store:
                region_table[ri, ci, 0] = store[0]
                region_table[ri, ci, 1] = store[1]
                region_table[ri, ci, 2] = store[2]

    sdf_table = np.full(
        (len(SDF_BANDS), len(case_ids), 3), fill_value=np.nan, dtype=np.float64
    )
    for bi, band in enumerate(SDF_BANDS):
        for ci, case_id in enumerate(case_ids):
            store = diag.sdf_sums[band].get(case_id, [])
            if store:
                sdf_table[bi, ci, 0] = store[0]
                sdf_table[bi, ci, 1] = store[1]
                sdf_table[bi, ci, 2] = store[2]

    np.savez(
        out_path,
        run_id=np.array(run_id),
        project=np.array(project),
        config_yaml=np.array(yaml.safe_dump(config)),
        case_ids=np.array(case_ids),
        metric_names=np.array(metric_names),
        # case_table[case_idx, metric_idx, (err_sq, tgt_sq, count)]
        case_table=case_table,
        region_names=np.array(X_REL_REGIONS),
        # region_table[region_idx, case_idx, (err_sq, tgt_sq, count)]
        region_table=region_table,
        sdf_band_names=np.array(SDF_BANDS),
        # sdf_table[band_idx, case_idx, (err_sq, tgt_sq, count)]
        sdf_table=sdf_table,
        region_total_points=np.array(
            [diag.region_total_points[r] for r in X_REL_REGIONS], dtype=np.int64
        ),
        sdf_total_points=np.array(
            [diag.sdf_total_points[b] for b in SDF_BANDS], dtype=np.int64
        ),
        total_volume_points=np.array(diag.total_volume_points, dtype=np.int64),
        aggregate_keys=np.array(list(aggregate.keys())),
        aggregate_values=np.array([float(v) for v in aggregate.values()], dtype=np.float64),
    )
    print(f"Saved diagnostic to {out_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase 0 per-case + per-region test_vol_p diagnostic")
    p.add_argument("--run-id", required=True, help="W&B run ID for the checkpoint")
    p.add_argument(
        "--project",
        default="senpai-v1-drivaerml-ddp8",
        help="W&B project that hosts the run (e.g. senpai-v1-drivaerml for yi-era runs).",
    )
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    p.add_argument("--cache-root", default="outputs/diagnostic_cache")
    p.add_argument("--output-dir", default="analysis")
    p.add_argument("--manifest", default="data/split_manifest.json")
    p.add_argument("--data-root", default="")
    p.add_argument("--eval-surface-points", type=int, default=65536)
    p.add_argument("--eval-volume-points", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp-mode", default="bf16", choices=["bf16", "none"])
    p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", "online"),
        help="Set to 'offline' or 'disabled' to skip uploads.",
    )
    p.add_argument("--wandb-group", default="askeladd-phase0-diagnostic")
    p.add_argument("--wandb-name", default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    artifact_dir = download_checkpoint(api, args.entity, args.project, args.run_id, cache_root)
    print(f"Artifact for {args.run_id}: {artifact_dir}")

    model, config = load_checkpoint_for_diagnostic(args.run_id, artifact_dir, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    print("Loading data...")
    _, _, test_splits, stats = load_data(
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
    test_dataset = test_splits["test_surface"]
    loader = make_eval_loader(test_dataset, args.batch_size, args.num_workers)
    print(f"Test split: {len(test_dataset)} views over {len(set(v.case_id for v in test_dataset.views))} cases, {len(loader)} batches")

    if args.limit_batches > 0:
        # Lightweight cap for debug runs — wraps the loader.
        from itertools import islice

        class _LimitedLoader:
            def __init__(self, base, n):
                self.base = base
                self.n = n
            def __iter__(self):
                return islice(iter(self.base), self.n)
            def __len__(self):
                return self.n
        loader = _LimitedLoader(loader, args.limit_batches)

    run = None
    if args.wandb_mode not in ("disabled", "offline"):
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.wandb_group,
            name=args.wandb_name or f"{args.run_id}-test-eval",
            tags=["askeladd", "phase0", "diagnostic", "test-eval"],
            mode=args.wandb_mode,
            config={
                "diagnostic_run_id": args.run_id,
                "model_n_params": n_params,
                "eval_surface_points": args.eval_surface_points,
                "eval_volume_points": args.eval_volume_points,
                "batch_size": args.batch_size,
                "amp_mode": args.amp_mode,
                "data_root": args.data_root,
            },
        )

    print("\nRunning inference...")
    diag, full_acc, aggregate = run_test_eval(
        model=model,
        loader=loader,
        transform=transform,
        device=device,
        amp_mode=args.amp_mode,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_case_test_predictions_{args.run_id}.npz"
    save_diagnostic_outputs(
        out_path=out_path,
        run_id=args.run_id,
        project=args.project,
        config=config,
        diag=diag,
        aggregate=aggregate,
    )

    print("\n=== Aggregate test metrics ===")
    for k in (
        "abupt_axis_mean_rel_l2_pct",
        "surface_pressure_rel_l2_pct",
        "wall_shear_rel_l2_pct",
        "volume_pressure_rel_l2_pct",
        "wall_shear_x_rel_l2_pct",
        "wall_shear_y_rel_l2_pct",
        "wall_shear_z_rel_l2_pct",
    ):
        v = aggregate.get(k, float("nan"))
        print(f"  {k}: {v:.4f}")

    print("\n=== Per-region (point-weighted) volume_pressure rel_l2 ===")
    region_payload: dict[str, float] = {}
    for region in X_REL_REGIONS:
        sums = diag.region_sums[region]
        total_err = sum(s[0] for s in sums.values() if s)
        total_tgt = sum(s[1] for s in sums.values() if s)
        total_pts = sum(s[2] for s in sums.values() if s)
        share = total_pts / max(diag.total_volume_points, 1)
        rel_l2 = 100.0 * math.sqrt(total_err / total_tgt) if total_tgt > 0 else float("nan")
        print(f"  {region:11s} | share={share:6.2%} | rel_l2={rel_l2:.4f}%")
        region_payload[f"region/{region}/point_share"] = float(share)
        region_payload[f"region/{region}/volume_pressure_rel_l2_pct"] = float(rel_l2)

    print("\n=== Per-SDF-band (point-weighted) volume_pressure rel_l2 ===")
    band_payload: dict[str, float] = {}
    for band in SDF_BANDS:
        sums = diag.sdf_sums[band]
        total_err = sum(s[0] for s in sums.values() if s)
        total_tgt = sum(s[1] for s in sums.values() if s)
        total_pts = sum(s[2] for s in sums.values() if s)
        share = total_pts / max(diag.total_volume_points, 1)
        rel_l2 = 100.0 * math.sqrt(total_err / total_tgt) if total_tgt > 0 else float("nan")
        print(f"  {band:13s} | share={share:6.2%} | rel_l2={rel_l2:.4f}%")
        band_payload[f"sdf_band/{band}/point_share"] = float(share)
        band_payload[f"sdf_band/{band}/volume_pressure_rel_l2_pct"] = float(rel_l2)

    print("\n=== Top-10 worst test cases by volume_pressure rel_l2 ===")
    case_vp = []
    metric_idx = {m: i for i, m in enumerate((*diag.SURFACE_METRICS, *diag.VOLUME_METRICS))}
    for case_id, sums in diag.case_metric_sums.items():
        store = sums.get("volume_pressure", [])
        case_vp.append((case_id, case_metric_to_rel_l2_pct(store)))
    case_vp_sorted = sorted(case_vp, key=lambda x: -x[1] if not math.isnan(x[1]) else 0)
    for case_id, vp in case_vp_sorted[:10]:
        print(f"  {case_id:12s}  vol_p={vp:.4f}%")

    if run is not None:
        log_payload = {f"test_primary/{k}": float(v) for k, v in aggregate.items() if isinstance(v, (int, float))}
        log_payload.update(region_payload)
        log_payload.update(band_payload)
        log_payload["peak_memory_gb"] = (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        )
        wandb.log(log_payload)
        wandb.summary.update(log_payload)
        wandb.finish()


if __name__ == "__main__":
    main()
