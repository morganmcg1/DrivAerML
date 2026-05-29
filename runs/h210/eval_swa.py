# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H210 SWA single-checkpoint eval — full val + test on a single GPU.

This script loads one SWA-averaged checkpoint, builds the matching model
from the donor config, and runs the same full_val + test evaluation as
`run_final_evaluation` in trainer_runtime. It logs val_primary/*,
full_val_primary/* and test_primary/* W&B metrics with the same naming
contract as `train.py`, so SWA configs can be compared 1:1 to the EP13
"best" runs that produced the input checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

# Make the project root (target/) importable regardless of where this script
# is invoked from. The script lives at <root>/runs/h210/eval_swa.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import wandb
import yaml

from data import load_data, pad_collate
from model import SurfaceTransolver
from trainer_runtime import (
    TargetTransform,
    evaluate_split,
    metric_namespace,
    primary_metric_log,
    print_metrics,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H210 SWA checkpoint eval")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--donor-config", required=True, type=str,
                        help="Path to one of the 5 donor config.yaml files "
                             "(all donors share the same architecture).")
    parser.add_argument("--wandb-group", default="h210-tanjiro-swa-cohort")
    parser.add_argument("--wandb-name", required=True, type=str)
    parser.add_argument("--wandb-tags", nargs="*", default=["h210", "swa", "tanjiro", "eval"])
    parser.add_argument("--eval-surface-points", type=int, default=65536)
    parser.add_argument("--eval-volume-points", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-mode", default="bf16", choices=["bf16", "none"])
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def build_model_from_donor_config(cfg: dict) -> SurfaceTransolver:
    rff_sigmas_raw = cfg.get("rff_init_sigmas")
    if isinstance(rff_sigmas_raw, str) and rff_sigmas_raw.strip():
        init_sigmas = [float(v) for v in rff_sigmas_raw.split(",") if v.strip()]
    elif isinstance(rff_sigmas_raw, (list, tuple)) and rff_sigmas_raw:
        init_sigmas = [float(v) for v in rff_sigmas_raw]
    else:
        init_sigmas = None
    return SurfaceTransolver(
        n_layers=int(cfg["model_layers"]),
        n_hidden=int(cfg["model_hidden_dim"]),
        dropout=float(cfg.get("model_dropout", 0.0)),
        n_head=int(cfg["model_heads"]),
        mlp_ratio=int(cfg.get("model_mlp_ratio", 4)),
        slice_num=int(cfg["model_slices"]),
        rff_num_features=int(cfg.get("rff_num_features", 0)),
        rff_sigma=float(cfg.get("rff_sigma", 1.0)),
        rff_init_sigmas=init_sigmas,
        pos_encoding_mode=str(cfg.get("pos_encoding_mode", "sincos")),
        use_qk_norm=bool(cfg.get("use_qk_norm", False)),
        use_surf_to_vol_xattn=bool(cfg.get("use_surf_to_vol_xattn", False)),
    )


def make_eval_loader(dataset, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        collate_fn=pad_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Loading donor config: {args.donor_config}")
    with open(args.donor_config, "r") as fh:
        donor_cfg = yaml.safe_load(fh)
    model = build_model_from_donor_config(donor_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built: {n_params/1e6:.2f}M params")

    print(f"Loading SWA checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    swa_meta = ckpt.get("swa", {})
    print(f"SWA members={swa_meta.get('members')} weights={swa_meta.get('weights')}")

    data_root = args.data_root or donor_cfg.get("data_root") or ""
    print(f"Loading data from manifest={args.manifest} root={data_root!r}")
    _, val_splits, test_splits, stats = load_data(
        manifest_path=args.manifest,
        root=data_root or None,
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

    val_loaders = {
        name: make_eval_loader(ds, args.batch_size, args.num_workers)
        for name, ds in val_splits.items()
    }
    test_loaders = {
        name: make_eval_loader(ds, args.batch_size, args.num_workers)
        for name, ds in test_splits.items()
    }

    run = None
    if not args.no_wandb:
        wb_cfg = {
            "checkpoint": args.checkpoint,
            "swa_members": swa_meta.get("members"),
            "swa_weights": swa_meta.get("weights"),
            "donor_config": args.donor_config,
            "eval_surface_points": args.eval_surface_points,
            "eval_volume_points": args.eval_volume_points,
            "batch_size": args.batch_size,
            "amp_mode": args.amp_mode,
            "donor_run_epoch": int(ckpt.get("epoch", -1)),
        }
        run = wandb.init(
            entity=entity,
            project=project,
            group=args.wandb_group,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=wb_cfg,
            mode=os.environ.get("WANDB_MODE", "online"),
        )

    log_payload: dict[str, float] = {}
    summary: dict[str, float] = {}

    # Full val
    full_val_metrics = {}
    for name, loader in val_loaders.items():
        t0 = time.time()
        m = evaluate_split(model, loader, transform, device, amp_mode=args.amp_mode)
        m["_seconds"] = time.time() - t0
        full_val_metrics[name] = m
        print_metrics(f"full_val/{name}", m)
        print(f"  ({m['_seconds']:.1f}s)")
    primary_val_name = "val_surface" if "val_surface" in full_val_metrics else next(iter(full_val_metrics))
    full_val_log = primary_metric_log("full_val_primary", full_val_metrics[primary_val_name])
    for split_name, m in full_val_metrics.items():
        full_val_log.update(metric_namespace("full_val", split_name, m))
    log_payload.update(full_val_log)
    summary.update(full_val_log)

    # Test
    test_metrics = {}
    for name, loader in test_loaders.items():
        t0 = time.time()
        m = evaluate_split(model, loader, transform, device, amp_mode=args.amp_mode)
        m["_seconds"] = time.time() - t0
        test_metrics[name] = m
        print_metrics(f"test/{name}", m)
        print(f"  ({m['_seconds']:.1f}s)")
    primary_test_name = "test_surface" if "test_surface" in test_metrics else next(iter(test_metrics))
    test_log = primary_metric_log("test_primary", test_metrics[primary_test_name])
    for split_name, m in test_metrics.items():
        test_log.update(metric_namespace("test", split_name, m))
    log_payload.update(test_log)
    summary.update(test_log)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        summary["peak_memory_gb"] = peak
        log_payload["peak_memory_gb"] = peak
        print(f"Peak GPU memory: {peak:.2f} GB")

    if run is not None:
        wandb.log(log_payload)
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
