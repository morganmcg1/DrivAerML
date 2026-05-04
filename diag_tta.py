"""Diagnostic for Y-mirror TTA: compare per-channel pred statistics on
original vs mirrored input for a single batch."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import wandb

from data import load_data, pad_collate
from ensemble_eval import download_checkpoint, load_member, make_eval_loader
from tta_eval import (
    SURFACE_OUTPUT_TAU_Y_IDX,
    y_mirror_surface_x,
    y_mirror_volume_x,
)
from trainer_runtime import TargetTransform, autocast_context


@torch.no_grad()
def main():
    entity = "wandb-applied-ai-team"
    project = "senpai-v1-drivaerml-ddp8"
    run_id = "nh96x7m4"
    device = torch.device("cuda")
    cache_root = Path("outputs/tta_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    member_dir = download_checkpoint(api, entity, project, run_id, cache_root)
    model, cfg = load_member(run_id, member_dir, device)
    model.eval()

    _, val_splits, _, stats = load_data(
        train_surface_points=65536,
        eval_surface_points=65536,
        train_volume_points=65536,
        eval_volume_points=65536,
        debug=False,
    )
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    loader = make_eval_loader(val_splits["val_surface"], batch_size=1, num_workers=0)
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        out_o = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
        sx_m = y_mirror_surface_x(batch.surface_x)
        vx_m = y_mirror_volume_x(batch.volume_x)
        out_m = model(
            surface_x=sx_m,
            surface_mask=batch.surface_mask,
            volume_x=vx_m,
            volume_mask=batch.volume_mask,
        )

        # Surface input check
        sm = batch.surface_mask.bool()[0]
        sx0 = batch.surface_x[0]
        sx0_m = sx_m[0]
        print(f"\nBatch {batch_idx}: case={batch.case_ids[0]} surface_n={int(sm.sum())} volume_n={int(batch.volume_mask.bool()[0].sum())}")
        print(f"  surface y range orig: {sx0[sm,1].min().item():.4f} to {sx0[sm,1].max().item():.4f}")
        print(f"  surface y range mirror: {sx0_m[sm,1].min().item():.4f} to {sx0_m[sm,1].max().item():.4f}")
        print(f"  surface ny range orig: {sx0[sm,4].min().item():.4f} to {sx0[sm,4].max().item():.4f}")
        print(f"  surface ny range mirror: {sx0_m[sm,4].min().item():.4f} to {sx0_m[sm,4].max().item():.4f}")

        # Predictions in physical space
        sp_orig = transform.invert_surface(out_o["surface_preds"].float())
        sp_mirror = transform.invert_surface(out_m["surface_preds"].float())
        sp_mirror_corr = sp_mirror.clone()
        sp_mirror_corr[..., SURFACE_OUTPUT_TAU_Y_IDX] = -sp_mirror_corr[..., SURFACE_OUTPUT_TAU_Y_IDX]

        target = batch.surface_y[0][sm]
        pred_o = sp_orig[0][sm]
        pred_m = sp_mirror[0][sm]
        pred_mc = sp_mirror_corr[0][sm]

        names = ["cp", "tau_x", "tau_y", "tau_z"]
        print("  Per-channel residual norms (root mean squared error vs target):")
        for i, name in enumerate(names):
            rmse_orig = (pred_o[:, i] - target[:, i]).pow(2).mean().sqrt().item()
            rmse_mirror_raw = (pred_m[:, i] - target[:, i]).pow(2).mean().sqrt().item()
            rmse_mirror_corr = (pred_mc[:, i] - target[:, i]).pow(2).mean().sqrt().item()
            avg_corr = 0.5 * (pred_o[:, i] + pred_mc[:, i])
            rmse_avg = (avg_corr - target[:, i]).pow(2).mean().sqrt().item()
            print(
                f"    {name}: rmse_orig={rmse_orig:.4f}  rmse_mirror_raw={rmse_mirror_raw:.4f}  "
                f"rmse_mirror_corr={rmse_mirror_corr:.4f}  rmse_avg_tta={rmse_avg:.4f}"
            )

        # Stats: how similar are pred_orig and pred_mirror_corr in physical space?
        print("  agreement (pred_orig vs sign-corrected pred_mirror):")
        for i, name in enumerate(names):
            d = pred_o[:, i] - pred_mc[:, i]
            corr_xy = torch.corrcoef(torch.stack([pred_o[:, i], pred_mc[:, i]])).flatten()[1].item()
            print(
                f"    {name}: |diff|_mean={d.abs().mean().item():.4f}  "
                f"|orig|_mean={pred_o[:, i].abs().mean().item():.4f}  "
                f"corr={corr_xy:.4f}"
            )

        # Volume
        vp_orig = transform.invert_volume(out_o["volume_preds"].float())
        vp_mirror = transform.invert_volume(out_m["volume_preds"].float())
        vm = batch.volume_mask.bool()[0]
        target_v = batch.volume_y[0][vm]
        pred_vo = vp_orig[0][vm]
        pred_vm = vp_mirror[0][vm]
        avg_v = 0.5 * (pred_vo + pred_vm)
        print(f"  vp: rmse_orig={(pred_vo-target_v).pow(2).mean().sqrt().item():.4f}  "
              f"rmse_mirror={(pred_vm-target_v).pow(2).mean().sqrt().item():.4f}  "
              f"rmse_avg={(avg_v-target_v).pow(2).mean().sqrt().item():.4f}")

        if batch_idx >= 2:
            break


if __name__ == "__main__":
    main()
