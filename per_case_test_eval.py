# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Per-case test eval for the SDF data-fix run (PR #941).

Loads the EP12 best-EMA checkpoint from outputs/drivaerml/run-<run_id>/checkpoint.pt
and runs evaluation against the test split, emitting per-case rel_l2_pct
metrics for surface_pressure, wall_shear (axis-mean), wall_shear_{x,y,z},
and volume_pressure. Highlights the 4 OOD cases (run_133, run_158,
run_203, run_226) and contrasts them against the other 46 cases.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml

from data import load_data, pad_collate
from model import SurfaceTransolver
from torch.utils.data import DataLoader
from trainer_runtime import (
    EvalAccumulator,
    TargetTransform,
    accumulate_eval_batch,
)

OOD_CASES = {"run_133", "run_158", "run_203", "run_226"}


def per_case_rel_l2_pct(store: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for case_id, (err_sq, tgt_sq) in store.items():
        if tgt_sq <= 0.0:
            continue
        out[case_id] = 100.0 * math.sqrt(err_sq / tgt_sq)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/drivaerml/run-2ub8dmy7")
    parser.add_argument("--manifest", default="data/split_manifest.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--eval-surface-points", type=int, default=65536)
    parser.add_argument("--eval-volume-points", type=int, default=65536)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    cfg_dict = yaml.safe_load(config_path.read_text())

    device = torch.device(args.device)
    print(f"Loading checkpoint from {run_dir}/checkpoint.pt on {device}")

    train_ds, val_splits, test_splits, stats = load_data(
        manifest_path=cfg_dict.get("manifest", args.manifest),
        root=args.data_root or None,
        train_surface_points=int(cfg_dict.get("train_surface_points", 65536)),
        eval_surface_points=args.eval_surface_points,
        train_volume_points=int(cfg_dict.get("train_volume_points", 16384)),
        eval_volume_points=args.eval_volume_points,
        debug=False,
    )
    del train_ds  # not needed

    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    def _parse_rff_sigmas(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            return [float(v.strip()) for v in raw.split(",") if v.strip()] or None
        if isinstance(raw, (list, tuple)):
            return [float(v) for v in raw] or None
        return None

    model = SurfaceTransolver(
        n_layers=int(cfg_dict["model_layers"]),
        n_hidden=int(cfg_dict["model_hidden_dim"]),
        dropout=float(cfg_dict.get("model_dropout", 0.0)),
        n_head=int(cfg_dict["model_heads"]),
        mlp_ratio=int(cfg_dict.get("model_mlp_ratio", 4)),
        slice_num=int(cfg_dict["model_slices"]),
        rff_num_features=int(cfg_dict.get("rff_num_features", 16)),
        rff_sigma=float(cfg_dict.get("rff_sigma", 1.0)),
        rff_init_sigmas=_parse_rff_sigmas(cfg_dict.get("rff_init_sigmas")),
        pos_encoding_mode=str(cfg_dict.get("pos_encoding_mode", "string_separable")),
        use_qk_norm=bool(cfg_dict.get("use_qk_norm", True)),
        use_surf_to_vol_xattn=bool(cfg_dict.get("use_surf_to_vol_xattn", False)),
    ).to(device)

    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(
        f"Loaded checkpoint epoch={ckpt.get('epoch', '?')} "
        f"source={ckpt.get('checkpoint_source', '?')} "
        f"best_val_abupt={ckpt.get('best_val', '?')}"
    )

    test_ds = test_splits["test_surface"]
    print(f"Test split: {len(test_ds)} views")
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_collate,
    )

    model.eval()
    accumulator = EvalAccumulator()
    import time
    t0 = time.time()
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            accumulate_eval_batch(
                accumulator,
                model=model,
                batch=batch,
                transform=transform,
                device=device,
                amp_mode=cfg_dict.get("amp_mode", "bf16"),
            )
            n_batches += 1
    elapsed = time.time() - t0
    print(f"Eval done: {n_batches} batches, {elapsed:.1f}s")

    surf_per = per_case_rel_l2_pct(accumulator.case_sums["surface_pressure"])
    ws_per = per_case_rel_l2_pct(accumulator.case_sums["wall_shear"])
    wsx_per = per_case_rel_l2_pct(accumulator.case_sums["wall_shear_x"])
    wsy_per = per_case_rel_l2_pct(accumulator.case_sums["wall_shear_y"])
    wsz_per = per_case_rel_l2_pct(accumulator.case_sums["wall_shear_z"])
    vol_per = per_case_rel_l2_pct(accumulator.case_sums["volume_pressure"])

    all_cases = sorted(vol_per.keys() | surf_per.keys())
    print(f"\nPer-case test metrics ({len(all_cases)} cases):")
    print(f"{'case':10s} {'surf_p%':>9s} {'wshear%':>9s} {'wsx%':>9s} {'wsy%':>9s} {'wsz%':>9s} {'vol_p%':>9s}  OOD")
    ood_vol = []
    other_vol = []
    ood_surf = []
    other_surf = []
    ood_ws = []
    other_ws = []
    for c in all_cases:
        is_ood = c in OOD_CASES
        sp = surf_per.get(c, float("nan"))
        ws = ws_per.get(c, float("nan"))
        wsx = wsx_per.get(c, float("nan"))
        wsy = wsy_per.get(c, float("nan"))
        wsz = wsz_per.get(c, float("nan"))
        vp = vol_per.get(c, float("nan"))
        marker = "  *OOD*" if is_ood else ""
        print(f"{c:10s} {sp:>9.4f} {ws:>9.4f} {wsx:>9.4f} {wsy:>9.4f} {wsz:>9.4f} {vp:>9.4f}{marker}")
        if math.isfinite(vp):
            (ood_vol if is_ood else other_vol).append(vp)
        if math.isfinite(sp):
            (ood_surf if is_ood else other_surf).append(sp)
        if math.isfinite(ws):
            (ood_ws if is_ood else other_ws).append(ws)

    def _stats(arr):
        if not arr:
            return (float("nan"), float("nan"), float("nan"), float("nan"))
        return (
            min(arr),
            max(arr),
            sum(arr) / len(arr),
            sorted(arr)[len(arr) // 2],
        )

    print("\n=== Summary ===")
    for label, arr in [("OOD-4 vol_p", ood_vol), ("Other-46 vol_p", other_vol),
                       ("OOD-4 surf_p", ood_surf), ("Other-46 surf_p", other_surf),
                       ("OOD-4 wshear", ood_ws), ("Other-46 wshear", other_ws)]:
        mn, mx, mean, med = _stats(arr)
        print(f"  {label:18s}: n={len(arr):3d} mean={mean:7.4f}% median={med:7.4f}% min={mn:7.4f}% max={mx:7.4f}%")

    # Aggregate (matches wandb-logged test metric per case-mean aggregation)
    all_vol = ood_vol + other_vol
    all_surf = ood_surf + other_surf
    all_ws = ood_ws + other_ws
    print(f"\n=== Aggregate (50-case mean) ===")
    if all_surf:
        print(f"  surface_pressure: {sum(all_surf)/len(all_surf):.4f}%")
    if all_ws:
        print(f"  wall_shear:       {sum(all_ws)/len(all_ws):.4f}%")
    if all_vol:
        print(f"  volume_pressure:  {sum(all_vol)/len(all_vol):.4f}%")
    # abupt = mean of (surf_p, wshear, vol_p)
    abupt = []
    for c in all_cases:
        if c in surf_per and c in ws_per and c in vol_per:
            abupt.append((surf_per[c] + ws_per[c] + vol_per[c]) / 3.0)
    if abupt:
        print(f"  abupt (cases):    {sum(abupt)/len(abupt):.4f}%")
    # Or the canonical: mean each channel then mean
    if all_surf and all_ws and all_vol:
        sp_mean = sum(all_surf)/len(all_surf)
        ws_mean = sum(all_ws)/len(all_ws)
        vp_mean = sum(all_vol)/len(all_vol)
        print(f"  abupt (axis):     {(sp_mean+ws_mean+vp_mean)/3.0:.4f}%")


if __name__ == "__main__":
    main()
