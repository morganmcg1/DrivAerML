# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H352 raw-eval diagnostic gate runner.

Minimal val/test raw evaluation for the H352 pre-committed gate. Loads a
checkpoint (final EP16 EMA or an SWA-averaged file produced by
``swa_average.py``), builds the H336 model arch, and runs a single forward
pass over the canonical val/test loaders. No TTA, no calibration; the only
metric of interest is val_abupt RAW (``abupt_axis_mean_rel_l2_pct``) for the
intra-run diagnostic gate.

Usage (single GPU is sufficient; the run finishes in ~1 min):

    python eval_raw.py --checkpoint outputs/h352/swa_armA.pt \
        --label armA

Use ``torchrun --nproc-per-node=8 eval_raw.py ...`` to parallelize across
the same DDP topology as training; ``StridedDistributedSampler`` shards the
val/test splits without padding so the merged metric matches the single-GPU
result to numerical precision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from data.loader import DrivAerMLCaseStore, target_stats_from_normalizers
from model import SurfaceTransolver
from trainer_runtime import (
    TargetTransform,
    cleanup_distributed,
    evaluate_split,
    init_distributed,
    make_loaders,
)


@dataclass
class EvalRawConfig:
    """Minimal config to drive ``make_loaders`` + ``evaluate_split`` for H352."""

    checkpoint: str = ""
    label: str = "raw"
    eval_splits: str = "val,test"
    output_json: str = ""

    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"

    # H336 model arch (must exactly match train.py defaults used for H336).
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

    # Loader knobs (match train.py defaults for canonical eval).
    train_surface_points: int = 65536
    eval_surface_points: int = 65536
    train_volume_points: int = 65536
    eval_volume_points: int = 65536
    batch_size: int = 2
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    amp_mode: str = "bf16"
    debug: bool = False


def parse_args() -> EvalRawConfig:
    parser = argparse.ArgumentParser(description="H352 raw eval (no TTA, no cal)")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt file")
    parser.add_argument("--label", default="raw", help="Label for printed/JSON output")
    parser.add_argument(
        "--eval-splits",
        default="val,test",
        help="Comma-separated subset of {val,test}. Default 'val,test'.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to append a one-line JSON result for downstream parsing.",
    )
    ns = parser.parse_args()
    return EvalRawConfig(
        checkpoint=ns.checkpoint,
        label=ns.label,
        eval_splits=ns.eval_splits,
        output_json=ns.output_json,
    )


def build_model(cfg: EvalRawConfig) -> SurfaceTransolver:
    rff = [float(x) for x in cfg.rff_init_sigmas.split(",") if x.strip()]
    return SurfaceTransolver(
        n_layers=cfg.model_layers,
        n_hidden=cfg.model_hidden_dim,
        dropout=cfg.model_dropout,
        n_head=cfg.model_heads,
        mlp_ratio=cfg.model_mlp_ratio,
        slice_num=cfg.model_slices,
        rff_num_features=cfg.rff_num_features,
        rff_sigma=cfg.rff_sigma,
        rff_init_sigmas=rff,
        pos_encoding_mode=cfg.pos_encoding_mode,
        use_qk_norm=cfg.use_qk_norm,
        use_surf_to_vol_xattn=cfg.use_surf_to_vol_xattn,
        drop_path_max=cfg.drop_path_max,
    )


def main() -> None:
    cfg = parse_args()
    state = init_distributed()
    device = state.device

    if state.is_main:
        print(
            f"H352 raw eval: label={cfg.label} checkpoint={cfg.checkpoint} "
            f"splits={cfg.eval_splits} world={state.world_size}"
        )

    store = DrivAerMLCaseStore(manifest_path=cfg.manifest, root=cfg.data_root or None)
    stats = target_stats_from_normalizers(store)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    _, val_loaders, test_loaders, _ = make_loaders(cfg, distributed_state=state)

    model = build_model(cfg).to(device)
    ck = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        state_dict = ck["model"]
    elif isinstance(ck, dict):
        state_dict = ck
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {type(ck)}")
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if state.is_main:
        ep = ck.get("epoch") if isinstance(ck, dict) else None
        src = ck.get("checkpoint_source") if isinstance(ck, dict) else None
        h352 = ck.get("h352_swa") if isinstance(ck, dict) else None
        print(
            f"Loaded checkpoint epoch={ep} source={src} missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        if h352:
            print(f"  SWA provenance: {h352}")
    model.eval()

    wanted = {s.strip() for s in cfg.eval_splits.split(",") if s.strip()}
    output_rows: list[dict] = []

    if "val" in wanted:
        val_metrics_by_name = {
            name: evaluate_split(
                model,
                loader,
                transform,
                device,
                amp_mode=cfg.amp_mode,
                distributed_state=state,
            )
            for name, loader in val_loaders.items()
        }
        if state.is_main:
            for split_name, m in val_metrics_by_name.items():
                abupt = float(m["abupt_axis_mean_rel_l2_pct"])
                row = {"label": cfg.label, "split": split_name, "abupt_pct": abupt, **{k: float(v) for k, v in m.items() if isinstance(v, (int, float))}}
                output_rows.append(row)
                print(f"  {split_name}: abupt_pct={abupt:.4f}")

    if "test" in wanted:
        test_metrics_by_name = {
            name: evaluate_split(
                model,
                loader,
                transform,
                device,
                amp_mode=cfg.amp_mode,
                distributed_state=state,
            )
            for name, loader in test_loaders.items()
        }
        if state.is_main:
            for split_name, m in test_metrics_by_name.items():
                abupt = float(m["abupt_axis_mean_rel_l2_pct"])
                row = {"label": cfg.label, "split": split_name, "abupt_pct": abupt, **{k: float(v) for k, v in m.items() if isinstance(v, (int, float))}}
                output_rows.append(row)
                print(f"  {split_name}: abupt_pct={abupt:.4f}")

    if state.is_main and cfg.output_json and output_rows:
        out_path = Path(cfg.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as f:
            for row in output_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Appended {len(output_rows)} row(s) to {out_path}")

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
