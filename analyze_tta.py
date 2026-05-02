"""Analyze TTA experiment results from W&B runs.

Usage:
    python analyze_tta.py <run_id1> [<run_id2> ...]
"""
from __future__ import annotations

import sys
import wandb


PRIMARY_KEYS = [
    "abupt_axis_mean_rel_l2_pct",
    "surface_pressure_rel_l2_pct",
    "wall_shear_rel_l2_pct",
    "wall_shear_x_rel_l2_pct",
    "wall_shear_y_rel_l2_pct",
    "wall_shear_z_rel_l2_pct",
    "volume_pressure_rel_l2_pct",
]


def fmt(v):
    if v is None:
        return "—"
    return f"{v:.4f}"


def fmt_delta(t, n):
    if t is None or n is None:
        return "—"
    return f"{t - n:+.4f}"


def fmt_rel(t, n):
    if t is None or n is None or abs(n) < 1e-9:
        return "—"
    return f"{(t - n) / n * 100:+.2f}%"


def per_epoch_table(history):
    """Build per-epoch table of TTA-on vs TTA-off val_primary."""
    rows = []
    for entry in history:
        if "val_tta/val_surface/abupt_axis_mean_rel_l2_pct" not in entry:
            continue
        if "val_no_tta/val_surface/abupt_axis_mean_rel_l2_pct" not in entry:
            continue
        gs = entry.get("global_step")
        tta = entry["val_tta/val_surface/abupt_axis_mean_rel_l2_pct"]
        ctrl = entry["val_no_tta/val_surface/abupt_axis_mean_rel_l2_pct"]
        rows.append((gs, ctrl, tta))
    return rows


def summary_dict(summary, prefix):
    out = {}
    for key in PRIMARY_KEYS:
        full = f"{prefix}/{key}"
        if full in summary:
            out[key] = summary[full]
    return out


def analyze(run_id, project="wandb-applied-ai-team/senpai-v1-drivaerml"):
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    print(f"\n{'=' * 80}")
    print(f"Run: {run.name}  ({run.id})  state={run.state}")
    print(f"  group: {run.group}")
    print(f"  config: max_train_cases={run.config.get('max_train_cases')} "
          f"batch_size={run.config.get('batch_size')} "
          f"epochs={run.config.get('epochs')} "
          f"lr={run.config.get('lr')}")
    print('=' * 80)

    # Per-epoch comparison
    history = list(run.scan_history(
        keys=[
            "global_step",
            "val_tta/val_surface/abupt_axis_mean_rel_l2_pct",
            "val_no_tta/val_surface/abupt_axis_mean_rel_l2_pct",
            "val_tta/val_surface/surface_pressure_rel_l2_pct",
            "val_no_tta/val_surface/surface_pressure_rel_l2_pct",
            "val_tta/val_surface/wall_shear_rel_l2_pct",
            "val_no_tta/val_surface/wall_shear_rel_l2_pct",
            "val_tta/val_surface/wall_shear_x_rel_l2_pct",
            "val_no_tta/val_surface/wall_shear_x_rel_l2_pct",
            "val_tta/val_surface/wall_shear_y_rel_l2_pct",
            "val_no_tta/val_surface/wall_shear_y_rel_l2_pct",
            "val_tta/val_surface/wall_shear_z_rel_l2_pct",
            "val_no_tta/val_surface/wall_shear_z_rel_l2_pct",
            "val_tta/val_surface/volume_pressure_rel_l2_pct",
            "val_no_tta/val_surface/volume_pressure_rel_l2_pct",
        ],
    ))
    print("\n## Per-epoch table: val abupt_axis_mean_rel_l2_pct")
    print(f"| Epoch | step | val_no_tta | val_tta | Δ(TTA−ctrl) | Δ% |")
    print(f"|-------|------|------------|---------|-------------|-----|")
    for i, entry in enumerate(history):
        gs = entry.get("global_step", "?")
        tta = entry.get("val_tta/val_surface/abupt_axis_mean_rel_l2_pct")
        ctrl = entry.get("val_no_tta/val_surface/abupt_axis_mean_rel_l2_pct")
        if tta is None or ctrl is None:
            continue
        delta = tta - ctrl
        rel = (delta / ctrl) * 100 if abs(ctrl) > 1e-9 else 0
        print(f"| {i+1} | {gs} | {ctrl:.4f}% | {tta:.4f}% | {delta:+.4f} | {rel:+.2f}% |")

    print("\n## Per-component breakdown (last logged epoch)")
    if history:
        last = history[-1]
        print(f"| Component | val_no_tta | val_tta | Δ(TTA−ctrl) | Δ% |")
        print(f"|-----------|------------|---------|-------------|-----|")
        for key in PRIMARY_KEYS:
            ctrl = last.get(f"val_no_tta/val_surface/{key}")
            tta = last.get(f"val_tta/val_surface/{key}")
            print(f"| {key} | {fmt(ctrl)} | {fmt(tta)} | {fmt_delta(tta, ctrl)} | {fmt_rel(tta, ctrl)} |")

    summary = run.summary
    print("\n## Final full_val (best checkpoint)")
    full_val_tta = summary_dict(summary, "full_val_tta/val_surface")
    full_val_no = summary_dict(summary, "full_val_no_tta/val_surface")
    print(f"| Component | full_val_no_tta | full_val_tta | Δ | Δ% |")
    print(f"|-----------|-----------------|--------------|---|-----|")
    for key in PRIMARY_KEYS:
        ctrl = full_val_no.get(key)
        tta = full_val_tta.get(key)
        print(f"| {key} | {fmt(ctrl)} | {fmt(tta)} | {fmt_delta(tta, ctrl)} | {fmt_rel(tta, ctrl)} |")

    print("\n## Final test (best checkpoint)")
    test_tta = summary_dict(summary, "test_tta/test_surface")
    test_no = summary_dict(summary, "test_no_tta/test_surface")
    print(f"| Component | test_no_tta | test_tta | Δ | Δ% |")
    print(f"|-----------|-------------|----------|---|-----|")
    for key in PRIMARY_KEYS:
        ctrl = test_no.get(key)
        tta = test_tta.get(key)
        print(f"| {key} | {fmt(ctrl)} | {fmt(tta)} | {fmt_delta(tta, ctrl)} | {fmt_rel(tta, ctrl)} |")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for run_id in sys.argv[1:]:
        analyze(run_id)


if __name__ == "__main__":
    main()
