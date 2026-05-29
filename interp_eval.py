# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Linear weight-interpolation eval for DrivAerML (PR #1380, H207).

Loads two checkpoints (H112 EP13 and H183 EP13 best artifacts), builds
``w_alpha = (1 - alpha) * w_a + alpha * w_b`` for each requested alpha,
runs full val+test eval per alpha, and logs metrics to W&B. No training,
no source-file modifications.

Run a single alpha:

    python interp_eval.py --alphas 0.5 --wandb-name h207-alpha-0.5

Or run a sweep sequentially on one GPU:

    python interp_eval.py --alphas 0.0,0.25,0.5,0.75,1.0
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import wandb
import yaml

from data import load_data
from ensemble_eval import (
    build_model_from_config,
    download_checkpoint,
    evaluate_ensemble_split,
    make_eval_loader,
    primary_log_payload,
)
from trainer_runtime import TargetTransform


ARCH_KEYS = (
    "model_layers",
    "model_hidden_dim",
    "model_heads",
    "model_mlp_ratio",
    "model_slices",
    "rff_num_features",
    "rff_sigma",
    "rff_init_sigmas",
    "pos_encoding_mode",
    "use_qk_norm",
    "use_surf_to_vol_xattn",
    "enable_residual_positions",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0",
                   help="Comma-separated alphas in [0,1] (w_a@alpha=0, w_b@alpha=1).")
    p.add_argument("--run-id-a", default="u9ue2ryb",
                   help="W&B run id contributing weights at alpha=0 (H112 EP13).")
    p.add_argument("--run-id-b", default="5k58uzqc",
                   help="W&B run id contributing weights at alpha=1 (H183 EP13).")
    p.add_argument("--eval-surface-points", type=int, default=65536)
    p.add_argument("--eval-volume-points", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp-mode", default="bf16", choices=["bf16", "none"])
    p.add_argument("--manifest", default="data/split_manifest.json")
    p.add_argument("--data-root", default="")
    p.add_argument("--cache-root", default="outputs/ensemble_cache",
                   help="Where to cache downloaded W&B artifacts.")
    p.add_argument("--wandb-group", default="h207-askeladd-interp-h112-h183")
    p.add_argument("--wandb-name-prefix", default="h207-alpha")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--limit-batches", type=int, default=0,
                   help="Optional cap on batches per split (debug).")
    return p.parse_args()


def _verify_architecture_match(cfg_a: dict, cfg_b: dict) -> None:
    diffs = []
    for k in ARCH_KEYS:
        if cfg_a.get(k) != cfg_b.get(k):
            diffs.append(f"  {k}: A={cfg_a.get(k)!r} vs B={cfg_b.get(k)!r}")
    if diffs:
        raise RuntimeError(
            "Architecture mismatch between A and B checkpoints — interpolation "
            "is not well-defined:\n" + "\n".join(diffs)
        )


def _load_state_dict(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise RuntimeError(f"Checkpoint {checkpoint_path} missing 'model' key")
    state = {k: v for k, v in ckpt["model"].items()}
    src = ckpt.get("checkpoint_source")
    if src != "ema":
        print(f"WARNING: {checkpoint_path} checkpoint_source={src!r} (expected 'ema')")
    return state


def _verify_state_dicts_compatible(state_a: dict, state_b: dict) -> None:
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        raise RuntimeError(
            "State-dict key mismatch.\n"
            f"  Only in A ({len(only_a)}): {only_a[:10]}{'...' if len(only_a) > 10 else ''}\n"
            f"  Only in B ({len(only_b)}): {only_b[:10]}{'...' if len(only_b) > 10 else ''}"
        )
    for k in keys_a:
        if state_a[k].shape != state_b[k].shape:
            raise RuntimeError(
                f"Shape mismatch for {k}: A={tuple(state_a[k].shape)} "
                f"vs B={tuple(state_b[k].shape)}"
            )


def _interpolate(state_a: dict, state_b: dict, alpha: float) -> dict:
    if alpha == 0.0:
        return {k: v.clone() for k, v in state_a.items()}
    if alpha == 1.0:
        return {k: v.clone() for k, v in state_b.items()}
    interp: dict[str, torch.Tensor] = {}
    for k in state_a:
        a = state_a[k]
        b = state_b[k]
        # Float interpolation; cast back to the original dtype to match the
        # checkpoint precision used during training (typically float32 for
        # the SurfaceTransolver backbone).
        v = (1.0 - alpha) * a.float() + alpha * b.float()
        interp[k] = v.to(a.dtype)
    return interp


def main() -> None:
    args = parse_args()

    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    if not alphas:
        raise ValueError("--alphas parsed to empty list")
    for a in alphas:
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"alpha must be in [0, 1]; got {a}")

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading A checkpoint (run {args.run_id_a})...")
    api = wandb.Api()
    dir_a = download_checkpoint(api, entity, project, args.run_id_a, cache_root)
    print(f"Downloading B checkpoint (run {args.run_id_b})...")
    dir_b = download_checkpoint(api, entity, project, args.run_id_b, cache_root)

    cfg_a = yaml.safe_load((dir_a / "config.yaml").open())
    cfg_b = yaml.safe_load((dir_b / "config.yaml").open())
    _verify_architecture_match(cfg_a, cfg_b)

    state_a = _load_state_dict(dir_a / "checkpoint.pt")
    state_b = _load_state_dict(dir_b / "checkpoint.pt")
    _verify_state_dicts_compatible(state_a, state_b)
    print(f"State dicts compatible: {len(state_a)} tensors, total params "
          f"{sum(v.numel() for v in state_a.values()):,}")

    use_aux_decoder_heads = "surface_out.0.weight" in state_a
    model = build_model_from_config(
        cfg_a, use_aux_decoder_heads=use_aux_decoder_heads
    ).to(device)

    print("Loading data splits (val + test)...")
    _, val_splits, test_splits, stats = load_data(
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
    val_loader = make_eval_loader(val_splits["val_surface"], args.batch_size, args.num_workers)
    test_loader = make_eval_loader(test_splits["test_surface"], args.batch_size, args.num_workers)

    for alpha in alphas:
        print(f"\n=== alpha = {alpha} ===")
        t0 = time.time()
        interp_state = _interpolate(state_a, state_b, alpha)
        missing, unexpected = model.load_state_dict(interp_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"State-dict load issue at alpha={alpha}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        model.eval()

        val_metrics = evaluate_ensemble_split(
            [model], val_loader, transform, device, args.amp_mode
        )
        test_metrics = evaluate_ensemble_split(
            [model], test_loader, transform, device, args.amp_mode
        )

        peak_mem_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if torch.cuda.is_available() else 0.0
        )
        elapsed = time.time() - t0

        run_name = f"{args.wandb_name_prefix}-{alpha}"
        print(f"  val_abupt = {val_metrics['abupt_axis_mean_rel_l2_pct']:.4f}")
        print(f"  test_abupt = {test_metrics['abupt_axis_mean_rel_l2_pct']:.4f}")
        print(f"  test_WSS = {test_metrics['wall_shear_rel_l2_pct']:.4f}  "
              f"test_VP = {test_metrics['volume_pressure_rel_l2_pct']:.4f}  "
              f"test_SP = {test_metrics['surface_pressure_rel_l2_pct']:.4f}")
        print(f"  elapsed = {elapsed:.1f}s  peak_mem = {peak_mem_gb:.1f} GiB")

        if not args.no_wandb:
            run = wandb.init(
                entity=entity,
                project=project,
                group=args.wandb_group,
                name=run_name,
                tags=["h207", "interp", "askeladd", "eval-only"],
                config={
                    "alpha": alpha,
                    "run_id_a": args.run_id_a,
                    "run_id_b": args.run_id_b,
                    "agent": "askeladd",
                    "pr": 1380,
                    "hypothesis": "H207-weight-interp-H112-H183",
                    "eval_surface_points": args.eval_surface_points,
                    "eval_volume_points": args.eval_volume_points,
                    "batch_size": args.batch_size,
                    "amp_mode": args.amp_mode,
                    "data_root": args.data_root,
                },
                reinit=True,
                mode=os.environ.get("WANDB_MODE", "online"),
            )
            payload = {
                "alpha": alpha,
                "elapsed_seconds": elapsed,
                "peak_memory_gib": peak_mem_gb,
            }
            payload.update(primary_log_payload("full_val_primary", val_metrics))
            payload.update(primary_log_payload("test_primary", test_metrics))
            wandb.log(payload)
            for k, v in payload.items():
                wandb.run.summary[k] = v
            wandb.finish()


if __name__ == "__main__":
    main()
