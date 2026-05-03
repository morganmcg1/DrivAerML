# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Top-K best-val checkpoint soup evaluation for DrivAerML (PR #554).

Wortsman-style "model soup": uniformly averages the state_dicts of the K
best-val checkpoints across architecturally compatible W&B runs and
evaluates the resulting model on val + test. Single-GPU, post-hoc.

Reproduce:
    python target/scripts/soup_eval.py --k-list 1,2,3,4,5,6,7,8 \\
        --output-json target/outputs/soup_eval_results.json

The candidate set is hard-coded to the SOTA architecture stack:
    Lion, 4L/512d/4h/128sl, STRING-separable PE, QK-norm, RFF feat=16,
    EMA decay=0.999.

Saved checkpoint weights were already EMA-copied at best-val time
(see train.py around the `ema.copy_to(base_model)` + `torch.save` block),
so soup-input is the EMA-evaluated model.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import Config, build_model  # noqa: E402
from trainer_runtime import (  # noqa: E402
    TargetTransform,
    evaluate_split,
    full_eval_loaders_from,
    make_loaders,
)


# Candidate set: architecturally compatible with PR #523 SOTA, ranked by
# best_val_primary/abupt_axis_mean_rel_l2_pct ascending. All runs share
# 4L/512d/4h/128sl, string_separable, use_qk_norm=True, rff_num_features=16,
# Lion, ema_decay=0.999, model_dropout=0, model_mlp_ratio=4. Differences
# (lr, rff_init_sigmas, vol_points_schedule, surface_loss_weight) are
# train-time / init-time only and do not affect parameter shapes.
CANDIDATES: list[dict] = [
    {"rank": 1, "run_id": "9mm3sz7x", "agent": "askeladd", "best_val": 6.8701, "test": 8.1229,
     "label": "askeladd-tau-reweight-micro slw=2 lr=9e-5 vol-curric"},
    {"rank": 2, "run_id": "49aimdiz", "agent": "alphonse", "best_val": 6.8994, "test": 8.2972,
     "label": "alphonse-slw2-13ep multi-sigma vol-curric"},
    {"rank": 3, "run_id": "wyz68o8r", "agent": "thorfinn", "best_val": 6.9246, "test": 8.2355,
     "label": "PR#523 SOTA gradnorm-ema alpha=0.5"},
    {"rank": 4, "run_id": "qqtdnlwq", "agent": "alphonse", "best_val": 7.0063, "test": 8.2921,
     "label": "PR#510 slw=2.0"},
    {"rank": 5, "run_id": "5o7jc7wi", "agent": "edward", "best_val": 7.0134, "test": 8.3130,
     "label": "PR#511 extended cosine ep13"},
    {"rank": 6, "run_id": "nh2ke150", "agent": "edward", "best_val": 7.1030, "test": 8.3187,
     "label": "edward-cosine-ep15"},
    {"rank": 7, "run_id": "qawfhlu6", "agent": "frieren", "best_val": 7.1614, "test": 8.4886,
     "label": "frieren-aniso-string-vs511"},
    {"rank": 8, "run_id": "r5rw40rn", "agent": "thorfinn", "best_val": 7.1792, "test": 8.4969,
     "label": "PR#489 vol-curric 16k->65k"},
]


WANDB_PROJECT = "wandb-applied-ai-team/senpai-v1-drivaerml-ddp8"


# Eval-time architecture flags. These are the SOTA stack from PR #523 /
# PR #510 — all candidate runs share these exact values.
SOTA_ARCH = dict(
    model_layers=4,
    model_hidden_dim=512,
    model_heads=4,
    model_slices=128,
    model_dropout=0.0,
    model_mlp_ratio=4,
    rff_num_features=16,
    rff_sigma=1.0,
    rff_init_sigmas="",  # init-time only; trained log_freq overrides at load
    pos_encoding_mode="string_separable",
    use_qk_norm=True,
)


# Eval data flags from SOTA training (full-fidelity 65536 points per modality).
EVAL_DATA = dict(
    batch_size=4,
    train_surface_points=65536,
    eval_surface_points=65536,
    train_volume_points=65536,
    eval_volume_points=65536,
    amp_mode="bf16",
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2,
    manifest="data/split_manifest.json",
    data_root="",
    debug=False,
)


PRIMARY_METRIC_KEYS = (
    "abupt_axis_mean_rel_l2_pct",
    "surface_pressure_rel_l2_pct",
    "wall_shear_rel_l2_pct",
    "volume_pressure_rel_l2_pct",
    "wall_shear_x_rel_l2_pct",
    "wall_shear_y_rel_l2_pct",
    "wall_shear_z_rel_l2_pct",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-K checkpoint soup evaluation")
    p.add_argument(
        "--k-list",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated K values to sweep (top-K best-val).",
    )
    p.add_argument(
        "--ckpt-cache",
        type=str,
        default=str(REPO_ROOT / "outputs" / "soup_ckpts"),
        help="Cache directory for downloaded W&B checkpoints.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "outputs" / "soup_eval_results.json"),
        help="Where to write the results table as JSON.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation.",
    )
    p.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="Optional cap on number of candidates pulled (0 = use full set).",
    )
    p.add_argument(
        "--custom-recipe",
        type=str,
        default="",
        help=(
            "Probe a custom subset (overrides --k-list). Format: "
            "'run_id1,run_id2,...' for uniform, or "
            "'run_id1:w1,run_id2:w2,...' for weighted (weights normalized to sum 1)."
        ),
    )
    p.add_argument(
        "--probe-tag",
        type=str,
        default="probe",
        help="Tag for the custom-recipe row in the output JSON.",
    )
    return p.parse_args(argv)


def download_checkpoint(run_id: str, cache_root: Path) -> Path:
    """Download W&B checkpoint (cached) and return path to checkpoint.pt."""
    cache_root.mkdir(parents=True, exist_ok=True)
    run_cache = cache_root / run_id
    ckpt_path = run_cache / "checkpoint.pt"
    if ckpt_path.exists():
        return ckpt_path
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    art = next(a for a in run.logged_artifacts() if a.type == "model")
    art.download(root=str(run_cache))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Expected {ckpt_path} after artifact download")
    return ckpt_path


def load_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["model"]
    # Strip torch.compile prefix if present (none of our candidates have it,
    # but defensive).
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def soup_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted average of K state_dicts. All sds must share keys/shapes/dtypes.

    If `weights` is None, uses uniform mean. Otherwise weights are normalized
    to sum to 1 and applied per-tensor.
    """
    if not state_dicts:
        raise ValueError("Empty state_dict list")
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    if len(weights) != len(state_dicts):
        raise ValueError("weights length must match state_dicts length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Sum of weights must be positive")
    norm_weights = [w / total for w in weights]
    ref = state_dicts[0]
    soup: dict[str, torch.Tensor] = {}
    for k, ref_v in ref.items():
        for i, sd in enumerate(state_dicts[1:], start=1):
            if k not in sd:
                raise KeyError(f"Key '{k}' missing in state_dict #{i}")
            other = sd[k]
            if other.shape != ref_v.shape:
                raise ValueError(
                    f"Shape mismatch on '{k}': ref {tuple(ref_v.shape)} vs "
                    f"#{i} {tuple(other.shape)}"
                )
        if not torch.is_floating_point(ref_v):
            # Integer tensors (counters etc.): take from rank-0 unchanged.
            soup[k] = ref_v.clone()
            continue
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        weight_tensor = torch.tensor(norm_weights, dtype=stacked.dtype).view(
            -1, *([1] * (stacked.ndim - 1))
        )
        averaged = (stacked * weight_tensor).sum(dim=0)
        soup[k] = averaged.to(ref_v.dtype)
    return soup


def build_eval_config() -> Config:
    cfg = Config(**{**SOTA_ARCH, **EVAL_DATA})
    return cfg


def evaluate_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: Config,
    val_loaders: dict,
    test_loaders: dict,
    transform: TargetTransform,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model = build_model(config).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict load mismatch — missing={missing} unexpected={unexpected}"
        )
    val_metrics = {
        name: evaluate_split(
            model, loader, transform, device,
            amp_mode=config.amp_mode, distributed_state=None,
        )
        for name, loader in val_loaders.items()
    }
    test_metrics = {
        name: evaluate_split(
            model, loader, transform, device,
            amp_mode=config.amp_mode, distributed_state=None,
        )
        for name, loader in test_loaders.items()
    }
    # Free GPU memory between K's.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {"val": val_metrics["val_surface"], "test": test_metrics["test_surface"]}


def primary_summary(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(metrics[k]) for k in PRIMARY_METRIC_KEYS if k in metrics}


def parse_custom_recipe(spec: str) -> tuple[list[str], list[float]]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    run_ids: list[str] = []
    weights: list[float] = []
    for p in parts:
        if ":" in p:
            rid, w = p.split(":", 1)
            run_ids.append(rid.strip())
            weights.append(float(w))
        else:
            run_ids.append(p)
            weights.append(1.0)
    return run_ids, weights


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    use_recipe = bool(args.custom_recipe)
    if use_recipe:
        recipe_ids, recipe_weights = parse_custom_recipe(args.custom_recipe)
        recipe_set = [{"rank": -1, "run_id": rid, "agent": "?", "best_val": float("nan"),
                       "test": float("nan"), "label": "custom-recipe"} for rid in recipe_ids]
        candidate_set = recipe_set
        max_k = len(recipe_set)
        k_values = [max_k]
    else:
        recipe_weights = None
        k_values = sorted({int(x) for x in args.k_list.split(",") if x.strip()})
        if not k_values:
            raise ValueError("--k-list parsed to empty set")
        candidate_set = CANDIDATES
        if args.candidate_limit > 0:
            candidate_set = candidate_set[: args.candidate_limit]
        max_k = max(k_values)
        if max_k > len(candidate_set):
            raise ValueError(
                f"Requested K={max_k} but only {len(candidate_set)} candidates available"
            )

    cache_root = Path(args.ckpt_cache)

    # Step 1: download all checkpoints up to max_k.
    print(f"[soup_eval] Candidates (top-{max_k} by val_abupt):", flush=True)
    for cand in candidate_set[:max_k]:
        print(
            f"  rank={cand['rank']:2d} run={cand['run_id']:>10s} "
            f"agent={cand['agent']:>9s} val={cand['best_val']:.4f} "
            f"test={cand['test']:.4f}  ({cand['label']})",
            flush=True,
        )

    print("\n[soup_eval] Downloading checkpoints from W&B...", flush=True)
    for cand in candidate_set[:max_k]:
        ckpt_path = download_checkpoint(cand["run_id"], cache_root)
        print(f"  {cand['run_id']}: {ckpt_path}", flush=True)

    # Step 2: load all state_dicts up to max_k.
    print("\n[soup_eval] Loading state_dicts...", flush=True)
    state_dicts: list[dict[str, torch.Tensor]] = []
    ref_keys: set[str] | None = None
    for cand in candidate_set[:max_k]:
        ckpt_path = cache_root / cand["run_id"] / "checkpoint.pt"
        sd = load_state_dict(ckpt_path)
        keys = set(sd.keys())
        if ref_keys is None:
            ref_keys = keys
        else:
            missing = ref_keys - keys
            extra = keys - ref_keys
            if missing or extra:
                raise RuntimeError(
                    f"Key set differs for {cand['run_id']} — missing={missing} extra={extra}"
                )
        state_dicts.append(sd)
        print(f"  {cand['run_id']}: {len(sd)} tensors", flush=True)

    # Step 3: build the data loaders once (shared across all K).
    config = build_eval_config()
    print("\n[soup_eval] Building eval loaders...", flush=True)
    _train_loader, base_val_loaders, base_test_loaders, stats = make_loaders(
        config, distributed_state=None
    )
    val_loaders = full_eval_loaders_from(base_val_loaders, config)
    test_loaders = full_eval_loaders_from(base_test_loaders, config)

    device = torch.device(args.device)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    print(f"[soup_eval] Device={device}", flush=True)
    print(
        f"[soup_eval] Data: val_surface batches={len(val_loaders['val_surface'])} "
        f"test_surface batches={len(test_loaders['test_surface'])}",
        flush=True,
    )

    # Step 4: K sweep.
    results: list[dict] = []
    for K in k_values:
        t0 = time.time()
        members = candidate_set[:K]
        sds = state_dicts[:K]
        print(f"\n[soup_eval] === K = {K} ===", flush=True)
        print(
            "  members:",
            ", ".join(f"{c['run_id']}({c['best_val']:.3f})" for c in members),
            flush=True,
        )
        if K == 1:
            soup_sd = sds[0]
        elif use_recipe:
            soup_sd = soup_state_dicts(sds, weights=recipe_weights)
        else:
            soup_sd = soup_state_dicts(sds)
        eval_out = evaluate_state_dict(
            soup_sd, config, val_loaders, test_loaders, transform, device
        )
        val = primary_summary(eval_out["val"])
        test = primary_summary(eval_out["test"])
        elapsed = time.time() - t0
        results.append(
            {
                "K": K,
                "members": [c["run_id"] for c in members],
                "member_labels": [c["label"] for c in members],
                "val": val,
                "test": test,
                "elapsed_s": elapsed,
            }
        )
        print(
            f"  val_abupt = {val['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"test_abupt = {test['abupt_axis_mean_rel_l2_pct']:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        print(
            "  val:  "
            + " ".join(f"{k.removesuffix('_rel_l2_pct')}={val[k]:.4f}" for k in PRIMARY_METRIC_KEYS),
            flush=True,
        )
        print(
            "  test: "
            + " ".join(f"{k.removesuffix('_rel_l2_pct')}={test[k]:.4f}" for k in PRIMARY_METRIC_KEYS),
            flush=True,
        )

    # Final table.
    print("\n[soup_eval] === Final Soup-K results ===", flush=True)
    header = f"{'K':>2s}  {'val_abupt':>9s}  {'test_abupt':>10s}  {'val_sp':>7s}  {'val_vp':>7s}  {'test_sp':>8s}  {'test_vp':>8s}"
    print(header, flush=True)
    for r in results:
        v, t = r["val"], r["test"]
        print(
            f"{r['K']:>2d}  {v['abupt_axis_mean_rel_l2_pct']:>9.4f}  "
            f"{t['abupt_axis_mean_rel_l2_pct']:>10.4f}  "
            f"{v['surface_pressure_rel_l2_pct']:>7.4f}  "
            f"{v['volume_pressure_rel_l2_pct']:>7.4f}  "
            f"{t['surface_pressure_rel_l2_pct']:>8.4f}  "
            f"{t['volume_pressure_rel_l2_pct']:>8.4f}",
            flush=True,
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidates": candidate_set,
        "k_values": k_values,
        "wandb_project": WANDB_PROJECT,
        "config": asdict(config),
        "results": results,
        "sota_reference": {
            "PR": "#523",
            "run_id": "wyz68o8r",
            "test_abupt": 8.2355,
            "val_abupt": 6.9246,
        },
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[soup_eval] Wrote results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
