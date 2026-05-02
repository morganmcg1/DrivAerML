"""Inference-only script for the haku tau_y/z error-mode diagnostic (PR #363).

Loads the kohaku/seed2024 (k7wq5uxx) checkpoint as a PR #222-class stand-in
(canonical PR #222 W&B run `ut1qmc3i` is no longer accessible from W&B).
Runs full-fidelity validation inference on all 34 val cases and saves raw
per-point predictions for downstream analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Make target/ importable
TARGET_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TARGET_ROOT))

from data import load_data, pad_collate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from train import (  # noqa: E402
    Config,
    SurfaceTransolver,
    TargetTransform,
    autocast_context,
)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--checkpoint-config", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-surface-points", type=int, default=65536)
    p.add_argument("--eval-volume-points", type=int, default=65536)
    p.add_argument("--manifest", default="data/split_manifest.json")
    p.add_argument("--data-root", default=None)
    p.add_argument("--amp-mode", default="bf16")
    return p.parse_args()


def main() -> None:
    args = parse()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load training config from artifact's config.yaml
    with open(args.checkpoint_config) as f:
        ck_cfg = yaml.safe_load(f)
    print("checkpoint config (selected):")
    for k in [
        "agent",
        "lr",
        "batch_size",
        "model_layers",
        "model_hidden_dim",
        "model_heads",
        "model_slices",
        "model_mlp_ratio",
        "use_film",
        "film_encoder_dim",
        "stochastic_depth_prob",
        "pos_max_wavelength",
        "use_ema",
        "ema_decay",
        "wallshear_y_weight",
        "wallshear_z_weight",
    ]:
        if k in ck_cfg:
            print(f"  {k}: {ck_cfg[k]}")

    # Build a Config from defaults but override architecture from the checkpoint
    cfg = Config()
    cfg.model_layers = ck_cfg["model_layers"]
    cfg.model_hidden_dim = ck_cfg["model_hidden_dim"]
    cfg.model_heads = ck_cfg["model_heads"]
    cfg.model_slices = ck_cfg["model_slices"]
    cfg.model_mlp_ratio = ck_cfg.get("model_mlp_ratio", cfg.model_mlp_ratio)
    cfg.model_dropout = ck_cfg.get("model_dropout", cfg.model_dropout)
    cfg.stochastic_depth_prob = ck_cfg.get("stochastic_depth_prob", cfg.stochastic_depth_prob)
    cfg.use_film = ck_cfg.get("use_film", False)
    cfg.film_encoder_dim = ck_cfg.get("film_encoder_dim", cfg.film_encoder_dim)
    cfg.pos_max_wavelength = ck_cfg.get("pos_max_wavelength", cfg.pos_max_wavelength)
    cfg.eval_surface_points = args.eval_surface_points
    cfg.eval_volume_points = args.eval_volume_points
    cfg.train_surface_points = args.eval_surface_points
    cfg.train_volume_points = args.eval_volume_points
    cfg.batch_size = args.batch_size
    cfg.manifest = args.manifest
    if args.data_root is not None:
        cfg.data_root = args.data_root
    cfg.amp_mode = args.amp_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Build model and load checkpoint
    model = SurfaceTransolver(
        n_layers=cfg.model_layers,
        n_hidden=cfg.model_hidden_dim,
        dropout=cfg.model_dropout,
        n_head=cfg.model_heads,
        mlp_ratio=cfg.model_mlp_ratio,
        slice_num=cfg.model_slices,
        stochastic_depth_prob=cfg.stochastic_depth_prob,
        use_film=cfg.use_film,
        film_encoder_dim=cfg.film_encoder_dim,
        pos_max_wavelength=cfg.pos_max_wavelength,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"WARNING: missing keys {missing[:3]}.. unexpected {unexpected[:3]}..")
    print(
        f"loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
        f"val abupt={ckpt.get('val_metrics', {}).get('val_surface', {}).get('abupt_axis_mean_rel_l2_pct', '?')}"
    )
    model.eval()

    # Build data loaders -- use full-fidelity strided eval
    _train_ds, val_splits, test_splits, stats = load_data(
        manifest_path=cfg.manifest,
        root=cfg.data_root or None,
        train_surface_points=cfg.train_surface_points,
        eval_surface_points=cfg.eval_surface_points,
        train_volume_points=cfg.train_volume_points,
        eval_volume_points=cfg.eval_volume_points,
        debug=False,
    )
    if args.split == "val":
        target_split = val_splits["val_surface"]
    elif args.split == "test":
        target_split = test_splits["test_surface"]
    else:
        raise ValueError(f"unknown split {args.split}")
    loader = DataLoader(
        target_split,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    # Aggregate per-case across views (eval_chunk strided sampling visits each point exactly once)
    case_chunks: dict[str, dict[str, list[np.ndarray]]] = {}
    n_views = len(target_split)
    print(f"running inference over {n_views} views ({len(target_split.case_ids)} cases)...")

    t0 = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch = batch.to(device)
            with autocast_context(device, cfg.amp_mode):
                out = model(
                    surface_x=batch.surface_x,
                    surface_mask=batch.surface_mask,
                    volume_x=batch.volume_x,
                    volume_mask=batch.volume_mask,
                )
            surface_pred_norm = out["surface_preds"].float()
            surface_pred = transform.invert_surface(surface_pred_norm)

            for case_idx, case_id in enumerate(batch.case_ids):
                mask = batch.surface_mask[case_idx].bool().cpu().numpy()
                if not mask.any():
                    continue
                surface_x = batch.surface_x[case_idx][mask].cpu().numpy()  # [N, 7]
                surface_y = batch.surface_y[case_idx][mask].cpu().numpy()  # [N, 4]
                pred = surface_pred[case_idx][mask].cpu().numpy()  # [N, 4]

                case = case_chunks.setdefault(case_id, {"xyz": [], "normals": [], "tau_pred": [], "tau_target": [], "ps_pred": [], "ps_target": []})
                case["xyz"].append(surface_x[:, 0:3].copy())
                case["normals"].append(surface_x[:, 3:6].copy())
                case["tau_pred"].append(pred[:, 1:4].copy())
                case["tau_target"].append(surface_y[:, 1:4].copy())
                case["ps_pred"].append(pred[:, 0:1].copy())
                case["ps_target"].append(surface_y[:, 0:1].copy())

            if idx % 50 == 0 or idx + 1 == n_views:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 1e-6)
                eta = (n_views - idx - 1) / max(rate, 1e-6)
                print(f"  view {idx + 1}/{n_views} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"inference done in {elapsed:.1f}s for {len(case_chunks)} cases")

    # Concatenate chunks per case and persist
    case_records = {}
    for case_id, chunks in case_chunks.items():
        rec = {
            "case_id": case_id,
            "surface_xyz": np.concatenate(chunks["xyz"], axis=0).astype(np.float32),
            "surface_normals": np.concatenate(chunks["normals"], axis=0).astype(np.float32),
            "tau_pred": np.concatenate(chunks["tau_pred"], axis=0).astype(np.float32),
            "tau_target": np.concatenate(chunks["tau_target"], axis=0).astype(np.float32),
            "ps_pred": np.concatenate(chunks["ps_pred"], axis=0).astype(np.float32),
            "ps_target": np.concatenate(chunks["ps_target"], axis=0).astype(np.float32),
        }
        case_records[case_id] = rec
        out_path = args.out_dir / f"{case_id}.npz"
        np.savez_compressed(out_path, **{k: v for k, v in rec.items() if k != "case_id"})

    # Compute and report headline metrics for sanity check
    print("\nper-case headline metrics (rel-L2 over each axis):")
    overall_y_num = overall_y_den = 0.0
    overall_z_num = overall_z_den = 0.0
    overall_x_num = overall_x_den = 0.0
    for case_id in sorted(case_records.keys()):
        rec = case_records[case_id]
        target = rec["tau_target"]
        pred = rec["tau_pred"]
        diff = pred - target
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            num = float(np.sum(diff[:, axis_idx] ** 2))
            den = float(np.sum(target[:, axis_idx] ** 2))
            if axis_name == "x":
                overall_x_num += num
                overall_x_den += den
            elif axis_name == "y":
                overall_y_num += num
                overall_y_den += den
            else:
                overall_z_num += num
                overall_z_den += den

    def pct(num: float, den: float) -> float:
        return 100.0 * np.sqrt(num / max(den, 1e-12))

    print(f"  global tau_x rel-L2: {pct(overall_x_num, overall_x_den):.4f}%")
    print(f"  global tau_y rel-L2: {pct(overall_y_num, overall_y_den):.4f}%")
    print(f"  global tau_z rel-L2: {pct(overall_z_num, overall_z_den):.4f}%")
    # Save summary
    summary = {
        "split": args.split,
        "n_cases": len(case_records),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "tau_x_rel_l2_pct_global": pct(overall_x_num, overall_x_den),
        "tau_y_rel_l2_pct_global": pct(overall_y_num, overall_y_den),
        "tau_z_rel_l2_pct_global": pct(overall_z_num, overall_z_den),
    }
    with (args.out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(summary, f)
    print(f"saved {len(case_records)} cases to {args.out_dir}")


if __name__ == "__main__":
    main()
