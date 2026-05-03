"""
Find all running W&B runs for bengio research - search by run name prefix.
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
]

# Get all running runs, ordered by recency
print("=== All currently running runs (most recent 50) ===")
runs = api.runs(path, filters={"state": "running"}, order="-created_at")
run_list = list(runs[:50])
print(f"Total running runs found: {len(run_list)}")

rows = []
for r in run_list:
    s = r.summary_metrics
    abupt = s.get("val_primary/abupt_axis_mean_rel_l2_pct")
    abupt_str = f"{abupt:.4f}" if isinstance(abupt, float) else "N/A"
    rows.append({
        "run_id": r.id,
        "name": r.name[:45],
        "group": r.group or "",
        "step": r.lastHistoryStep,
        "abupt": abupt_str,
        "created": str(r.createdAt)[:16] if r.createdAt else "?",
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 50)
print(df.to_string(index=False))
