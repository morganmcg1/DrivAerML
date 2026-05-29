"""H235: aggregate W&B summary metrics from the per-checkpoint TTA evals.

Each ``frieren/h235-tta-<LABEL>-<RUNID>`` run logs ``full_val`` and ``test``
metrics under ``{split}/{mode}/{metric}`` summary keys for mode in
{orig, mirror, tta}. This script pulls them and renders the comparison
table requested by the PR:

  checkpoint | val_abupt orig | val_abupt TTA | test_abupt orig | test_abupt TTA | TTA_delta (bp) | WSS_x slope
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

CANDIDATES = [
    ("H185", "yw2a5dyl"),
    ("H183", "5k58uzqc"),
    ("H190", "9f2jtrg2"),
    ("H188", "18t5rx2t"),
    ("H148", "2qr5guel"),
    ("H191", "5y5a5tgr"),
    ("H181b", "w7w92npw"),
    ("H186", "d15dm825"),
]


def _summary(api, label: str, source_run_id: str) -> dict:
    runs = list(
        api.runs(
            "wandb-applied-ai-team/senpai-v1-drivaerml-ddp8",
            {"display_name": {"$regex": f"^frieren/h235-tta-{label}-{source_run_id}$"}},
        )
    )
    if not runs:
        return {}
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return dict(runs[0].summary), runs[0].id


def main(argv: list[str]) -> int:
    api = wandb.Api()
    out_rows = []
    for label, source_run_id in CANDIDATES:
        result = _summary(api, label, source_run_id)
        if not result:
            print(f"!! no H235 eval run found for {label} ({source_run_id})", file=sys.stderr)
            continue
        summary, h235_run_id = result

        def f(key: str) -> float | None:
            v = summary.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        v_orig = f("val_surface/orig/abupt_axis_mean_rel_l2_pct")
        v_tta = f("val_surface/tta/abupt_axis_mean_rel_l2_pct")
        t_orig = f("test_surface/orig/abupt_axis_mean_rel_l2_pct")
        t_tta = f("test_surface/tta/abupt_axis_mean_rel_l2_pct")
        wssx_orig = f("test_surface/orig/wall_shear_x_rel_l2_pct")
        wssx_tta = f("test_surface/tta/wall_shear_x_rel_l2_pct")
        sp_tta = f("test_surface/tta/surface_pressure_rel_l2_pct")
        vp_tta = f("test_surface/tta/volume_pressure_rel_l2_pct")
        wss_tta = f("test_surface/tta/wall_shear_rel_l2_pct")
        wssy_tta = f("test_surface/tta/wall_shear_y_rel_l2_pct")
        wssz_tta = f("test_surface/tta/wall_shear_z_rel_l2_pct")
        val_tta_delta_bp = (v_tta - v_orig) * 100 if v_orig is not None and v_tta is not None else None
        test_tta_delta_bp = (t_tta - t_orig) * 100 if t_orig is not None and t_tta is not None else None
        wssx_delta_bp = (wssx_tta - wssx_orig) * 100 if wssx_orig is not None and wssx_tta is not None else None
        out_rows.append({
            "label": label,
            "source_run": source_run_id,
            "h235_run": h235_run_id,
            "val_orig": v_orig,
            "val_tta": v_tta,
            "test_orig": t_orig,
            "test_tta": t_tta,
            "val_tta_delta_bp": val_tta_delta_bp,
            "test_tta_delta_bp": test_tta_delta_bp,
            "wssx_orig": wssx_orig,
            "wssx_tta": wssx_tta,
            "wssx_delta_bp": wssx_delta_bp,
            "test_tta_sp": sp_tta,
            "test_tta_vp": vp_tta,
            "test_tta_wss": wss_tta,
            "test_tta_wssy": wssy_tta,
            "test_tta_wssz": wssz_tta,
        })

    print("\n=== H235 cross-checkpoint TTA mirror sweep (corrected split, full_val + test_primary) ===\n")
    hdr = ("ckpt", "val_orig", "val_tta", "Δv(bp)", "test_orig", "test_tta", "Δt(bp)",
           "wssx_o", "wssx_tta", "Δwx(bp)", "sp_tta", "vp_tta", "wss_tta")
    fmt_hdr = "{:<6} {:>9} {:>9} {:>7} {:>10} {:>10} {:>7} {:>8} {:>9} {:>8} {:>7} {:>7} {:>8}"
    fmt_row = "{:<6} {:>9.4f} {:>9.4f} {:>+7.2f} {:>10.4f} {:>10.4f} {:>+7.2f} {:>8.4f} {:>9.4f} {:>+8.2f} {:>7.4f} {:>7.4f} {:>8.4f}"
    print(fmt_hdr.format(*hdr))
    sota_gate_passes = []
    for row in out_rows:
        if any(row[k] is None for k in ("val_orig", "val_tta", "test_orig", "test_tta", "wssx_orig", "wssx_tta")):
            print(f"{row['label']:<6} (missing summary metrics — check run)")
            continue
        print(fmt_row.format(
            row['label'],
            row['val_orig'], row['val_tta'], row['val_tta_delta_bp'],
            row['test_orig'], row['test_tta'], row['test_tta_delta_bp'],
            row['wssx_orig'], row['wssx_tta'], row['wssx_delta_bp'],
            row['test_tta_sp'] or 0.0, row['test_tta_vp'] or 0.0, row['test_tta_wss'] or 0.0,
        ))
        if row['val_tta'] < 5.9755 and row['test_tta'] < 5.8221:
            sota_gate_passes.append(row['label'])
    print()
    if sota_gate_passes:
        print(f"*** SOTA-candidate gate hits: {sota_gate_passes} ***")
    else:
        print("No checkpoint+TTA passes both SOTA gates (val < 5.9755 AND test < 5.8221).")
    print()
    out_path = Path("outputs/h235_eval/h235_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
