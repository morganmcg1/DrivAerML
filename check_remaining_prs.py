"""
Find W&B runs for remaining PRs by student name in recent runs.
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

# Check remaining wave groups and student-specific groups
# frieren PR#361 (wd sweep), alphonse PR#437 6L/256d,
# edward PR#468 (Muon), askeladd PR#495 (CoordConv dist-to-surface)
# Wave 9, 10, 11 runs, and the very new wave 19/20 PRs

GROUPS_TO_CHECK = [
    "bengio-wave9", "bengio-wave10", "bengio-wave11", "bengio-wave12",
    "bengio-wave13", "bengio-wave16", "bengio-wave17",
    "bengio-wave19", "bengio-wave20", "bengio-wave21", "bengio-wave22",
]

STUDENT_RUN_SEARCH = [
    "frieren", "edward", "nezuko", "emma", "gilbert", "violet", "chihiro",
]

print("=== Checking wave groups ===")
for group in GROUPS_TO_CHECK:
    try:
        runs = api.runs(path, filters={"group": group})
        run_list = list(runs[:5])
        if run_list:
            print(f"\nGroup: {group} ({len(run_list)} runs)")
            for r in run_list:
                s = r.summary_metrics
                abupt = s.get("val_primary/abupt_axis_mean_rel_l2_pct", "N/A")
                abupt_str = f"{abupt:.4f}" if isinstance(abupt, float) else "N/A"
                print(f"  {r.id} {r.name[:40]:40s} state={r.state} step={r.lastHistoryStep} abupt={abupt_str}")
    except Exception as e:
        pass

print("\n=== Searching by student name (recent running runs) ===")
# Search for running runs by username prefix
for student in STUDENT_RUN_SEARCH:
    try:
        runs = api.runs(
            path,
            filters={"username": student, "state": "running"},
            order="-created_at"
        )
        run_list = list(runs[:3])
        if run_list:
            print(f"\nStudent: {student}")
            for r in run_list:
                s = r.summary_metrics
                abupt = s.get("val_primary/abupt_axis_mean_rel_l2_pct", "N/A")
                abupt_str = f"{abupt:.4f}" if isinstance(abupt, float) else "N/A"
                print(f"  {r.id} {r.name[:40]:40s} group={r.group} step={r.lastHistoryStep} abupt={abupt_str}")
    except Exception as e:
        print(f"  {student}: ERROR {e}")
