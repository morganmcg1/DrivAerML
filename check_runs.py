"""
Query W&B run statuses for survey.
"""
import os
import sys
from pathlib import Path

# Add skill scripts to path
skill_dir = os.environ.get("CLAUDE_SKILL_DIR", "")
if skill_dir:
    sys.path.insert(0, str(Path(skill_dir) / "scripts"))

import wandb
import pandas as pd

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

METRICS = [
    "val_primary/abupt_axis_mean_rel_l2_pct",
    "val_primary/wall_shear_y_rel_l2_pct",
    "val_primary/wall_shear_z_rel_l2_pct",
    "val_primary/surface_pressure_rel_l2_pct",
    "val_primary/volume_pressure_rel_l2_pct",
]

# Specific run IDs to check
SPECIFIC_RUNS = {
    "hph6eaky": "fern / PR#409 coord-norm fix",
    "5ifnf1wc": "thorfinn / PR#382 6L/512d/8H",
    "4632xosf": "kohaku / PR#417 EMA",
    "0xi2n4oo": "alphonse / PR#437 6L/256d",
    "jj9r7x0o": "senku / PR#442 OHEM",
    "vyhpqruv": "tanjiro / PR#443 mirror+SW=2.0",
}

print("=" * 80)
print("SPECIFIC RUN STATUS")
print("=" * 80)

rows = []
for run_id, label in SPECIFIC_RUNS.items():
    try:
        run = api.run(f"{path}/{run_id}")
        summary = run.summary_metrics

        row = {
            "run_id": run_id,
            "label": label,
            "state": run.state,
            "epoch": summary.get("epoch", summary.get("trainer/epoch", "N/A")),
        }
        for m in METRICS:
            short = m.split("/")[-1].replace("_rel_l2_pct", "")
            row[short] = round(summary.get(m, float("nan")), 4) if summary.get(m) is not None else float("nan")
        rows.append(row)

    except Exception as e:
        rows.append({
            "run_id": run_id,
            "label": label,
            "state": f"ERROR: {e}",
            "epoch": "N/A",
        })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:.4f}".format)
print(df.to_string(index=False))

print()
print("=" * 80)
print("WAVE 19 / WAVE 20 RUNS")
print("=" * 80)

for wave in ["bengio-wave19", "bengio-wave20"]:
    print(f"\n--- Group: {wave} ---")
    try:
        runs = api.runs(path, filters={"group": wave})
        wave_rows = []
        for run in runs[:20]:
            summary = run.summary_metrics
            row = {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "epoch": summary.get("epoch", summary.get("trainer/epoch", "N/A")),
            }
            for m in METRICS:
                short = m.split("/")[-1].replace("_rel_l2_pct", "")
                row[short] = round(summary.get(m, float("nan")), 4) if summary.get(m) is not None else float("nan")
            wave_rows.append(row)

        if wave_rows:
            wdf = pd.DataFrame(wave_rows)
            print(wdf.to_string(index=False))
        else:
            print("  No runs found.")
    except Exception as e:
        print(f"  Error: {e}")

print()
print("Done.")
