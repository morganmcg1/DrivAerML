"""
Check W&B runs for waves 14-20 on bengio.
"""
import os
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

# Try all wave groups 1-25
waves_found = []
for w in range(14, 26):
    group = f"bengio-wave{w}"
    try:
        runs = api.runs(path, filters={"group": group})
        run_list = list(runs[:10])
        if run_list:
            waves_found.append((group, run_list))
    except Exception as e:
        pass

print(f"Found waves: {[w[0] for w in waves_found]}")

for group, runs in waves_found:
    print(f"\n{'='*80}")
    print(f"Group: {group}")
    print('='*80)
    rows = []
    for run in runs:
        summary = run.summary_metrics
        row = {
            "run_id": run.id,
            "name": run.name[:30],
            "state": run.state,
            "step": run.lastHistoryStep,
        }
        for m in METRICS:
            short = m.split("/")[-1].replace("_rel_l2_pct", "").replace("abupt_axis_mean", "abupt")
            row[short] = round(summary.get(m, float("nan")), 4) if summary.get(m) is not None else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
