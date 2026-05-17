"""H10 direction-vs-magnitude WSS error decomposition (PR #1148).

Loads BEST ckpt from a run-<id> dir, runs DDP inference over val_surface and
test_surface, and decomposes the wall-shear rel-L2 error orthogonally into a
magnitude component (along the GT direction) and a tangential component
(direction component perpendicular to GT direction).

Per-vertex (in physical units):
    u_gt    = tau_gt / |tau_gt|
    proj    = tau_pred . u_gt            (signed scalar)
    mag_err = proj - |tau_gt|            (signed)
    tan_err = tau_pred - proj * u_gt     (perpendicular to u_gt)

Per case (Pythagoras):
    total_err^2 = sum_v ||tau_pred - tau_gt||^2 == sum_v (mag_err^2 + ||tan_err||^2)
    mag_rel_l2  = sqrt(sum_v mag_err^2  / sum_v ||tau_gt||^2)
    dir_rel_l2  = sqrt(sum_v ||tan_err||^2 / sum_v ||tau_gt||^2)
    total_rel_l2= sqrt(mag_rel_l2^2 + dir_rel_l2^2)   (== wall_shear case rel L2)

Aggregated as the mean of per-case rel L2 values, matching evaluate_split().

Usage:
    cd target/
    torchrun --standalone --nproc-per-node=8 eval_direction_magnitude.py \\
        --run-dir outputs/drivaerml/run-qncb8ikl --eval-volume-points 65536
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from data import load_data
from train import Config, build_model
from trainer_runtime import (
    TargetTransform,
    autocast_context,
    cleanup_distributed,
    eval_loader_for_dataset,
    init_distributed,
    unwrap_model,
)


def _per_vertex_errs(tau_pred: torch.Tensor, tau_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mag_err_sq, tan_err_sq, target_sq, cos_sim per vertex.

    Shapes: all inputs/outputs are [..., 3] for tau and [...] (scalar per vertex) for outputs.
    """
    gt_norm = torch.linalg.vector_norm(tau_gt, dim=-1)
    safe_gt_norm = gt_norm.clamp(min=1e-12)
    u_gt = tau_gt / safe_gt_norm.unsqueeze(-1)
    proj = (tau_pred * u_gt).sum(dim=-1)
    mag_err = proj - gt_norm
    tan_err_vec = tau_pred - proj.unsqueeze(-1) * u_gt
    tan_err_sq = (tan_err_vec * tan_err_vec).sum(dim=-1)
    target_sq = gt_norm * gt_norm
    pred_norm = torch.linalg.vector_norm(tau_pred, dim=-1).clamp(min=1e-12)
    cos_sim = proj / pred_norm
    return mag_err * mag_err, tan_err_sq, target_sq, cos_sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-volume-points", type=int, default=None)
    parser.add_argument("--eval-surface-points", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-full-val", action="store_true")
    args = parser.parse_args()

    state = init_distributed()
    device = state.device
    is_main = state.is_main

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    config = Config(**cfg_dict)
    if args.eval_volume_points is not None:
        config.eval_volume_points = args.eval_volume_points
    if args.eval_surface_points is not None:
        config.eval_surface_points = args.eval_surface_points
    config.batch_size = args.batch_size
    config.num_workers = 2

    if is_main:
        print(f"World size: {state.world_size}  Device: {device}", flush=True)

    _train_ds, val_splits, test_splits, stats = load_data(
        manifest_path=config.manifest,
        root=config.data_root or None,
        train_surface_points=config.train_surface_points,
        eval_surface_points=config.eval_surface_points,
        train_volume_points=config.train_volume_points,
        eval_volume_points=config.eval_volume_points,
        debug=config.debug,
    )
    full_val_loaders = {
        name: eval_loader_for_dataset(ds, config, distributed_state=state)
        for name, ds in val_splits.items()
    }
    full_test_loaders = {
        name: eval_loader_for_dataset(ds, config, distributed_state=state)
        for name, ds in test_splits.items()
    }
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )
    model = build_model(config).to(device)
    ckpt_path = run_dir / "checkpoint.pt"
    if is_main:
        print(f"Loading: {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if is_main:
        print(f"  epoch={checkpoint['epoch']} source={checkpoint['checkpoint_source']}", flush=True)
    model.eval()

    def run_split(name: str, loader) -> dict[str, float]:
        # case_sums[case_id] = [mag_err_sq, tan_err_sq, target_sq, n_verts, cos_sum]
        case_sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0.0])
        eval_module = unwrap_model(model)
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad(), autocast_context(device, config.amp_mode):
                out = eval_module(
                    surface_x=batch.surface_x,
                    surface_mask=batch.surface_mask,
                    volume_x=batch.volume_x,
                    volume_mask=batch.volume_mask,
                )
            surface_pred_norm = out["surface_preds"].float()
            surface_pred = transform.invert_surface(surface_pred_norm)
            mask = batch.surface_mask  # [B, N]
            tau_pred = surface_pred[..., 1:4]
            tau_gt = batch.surface_y[..., 1:4]
            mag_err_sq, tan_err_sq, target_sq, cos_sim = _per_vertex_errs(tau_pred, tau_gt)
            for b, case_id in enumerate(batch.case_ids):
                m = mask[b]
                if not bool(m.any()):
                    continue
                state_arr = case_sums[case_id]
                state_arr[0] += float(mag_err_sq[b][m].sum().detach().cpu().item())
                state_arr[1] += float(tan_err_sq[b][m].sum().detach().cpu().item())
                state_arr[2] += float(target_sq[b][m].sum().detach().cpu().item())
                state_arr[3] += int(m.sum().item())
                state_arr[4] += float(cos_sim[b][m].sum().detach().cpu().item())

        if state.enabled:
            payload: dict[str, list[float]] = {k: list(v) for k, v in case_sums.items()}
            gathered: list[dict | None] = [None for _ in range(state.world_size)]
            dist.all_gather_object(gathered, payload)
            if not is_main:
                return {}
            merged: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0.0])
            for shard in gathered:
                if shard is None:
                    continue
                for k, v in shard.items():
                    m = merged[k]
                    m[0] += v[0]
                    m[1] += v[1]
                    m[2] += v[2]
                    m[3] += int(v[3])
                    m[4] += v[4]
            case_sums = merged

        mag_rel = []
        dir_rel = []
        total_rel = []
        cos_avg = []
        for case_id, (m_sq, d_sq, t_sq, n_verts, cos_s) in case_sums.items():
            if t_sq <= 0.0 or n_verts == 0:
                continue
            mag_rel.append(math.sqrt(m_sq / t_sq))
            dir_rel.append(math.sqrt(d_sq / t_sq))
            total_rel.append(math.sqrt((m_sq + d_sq) / t_sq))
            cos_avg.append(cos_s / n_verts)
        if not mag_rel:
            return {}
        return {
            "n_cases": len(mag_rel),
            "mag_rel_l2_pct": 100.0 * sum(mag_rel) / len(mag_rel),
            "dir_rel_l2_pct": 100.0 * sum(dir_rel) / len(dir_rel),
            "total_rel_l2_pct": 100.0 * sum(total_rel) / len(total_rel),
            "mean_cos_sim": sum(cos_avg) / len(cos_avg),
            "mag_share_sq": (
                sum(m * m for m in mag_rel) / sum(t * t for t in total_rel)
                if total_rel else 0.0
            ),
            "dir_share_sq": (
                sum(d * d for d in dir_rel) / sum(t * t for t in total_rel)
                if total_rel else 0.0
            ),
        }

    results: dict[str, dict[str, float]] = {}
    if not args.skip_full_val:
        if is_main:
            print("\n=== full_val direction/magnitude decomposition ===", flush=True)
        for name, loader in full_val_loaders.items():
            metrics = run_split(name, loader)
            if is_main and metrics:
                results[f"full_val.{name}"] = metrics
                print(
                    f"  {name}: n={metrics['n_cases']} total={metrics['total_rel_l2_pct']:.4f}%"
                    f" mag={metrics['mag_rel_l2_pct']:.4f}% dir={metrics['dir_rel_l2_pct']:.4f}%"
                    f" cos={metrics['mean_cos_sim']:.6f}"
                    f" mag_share_sq={metrics['mag_share_sq']:.3f}"
                    f" dir_share_sq={metrics['dir_share_sq']:.3f}",
                    flush=True,
                )

    if is_main:
        print("\n=== test direction/magnitude decomposition ===", flush=True)
    for name, loader in full_test_loaders.items():
        metrics = run_split(name, loader)
        if is_main and metrics:
            results[f"test.{name}"] = metrics
            print(
                f"  {name}: n={metrics['n_cases']} total={metrics['total_rel_l2_pct']:.4f}%"
                f" mag={metrics['mag_rel_l2_pct']:.4f}% dir={metrics['dir_rel_l2_pct']:.4f}%"
                f" cos={metrics['mean_cos_sim']:.6f}"
                f" mag_share_sq={metrics['mag_share_sq']:.3f}"
                f" dir_share_sq={metrics['dir_share_sq']:.3f}",
                flush=True,
            )

    if is_main:
        out_path = run_dir / "direction_magnitude_decomp.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nWrote: {out_path}", flush=True)

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
