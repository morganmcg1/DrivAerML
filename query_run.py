import wandb
import numpy as np
import sys
import os

api = wandb.Api()

entity = "wandb-applied-ai-team"
project = "senpai-v1-drivaerml-ddp8"
run_id = "ze0bohdu"

path = f"{entity}/{project}/{run_id}"
run = api.run(path)

print(f"=== RUN STATE ===")
print(f"ID: {run.id}")
print(f"Name: {run.name}")
print(f"State: {run.state}")

s = run.summary_metrics
print(f"\n=== LATEST STEP ===")
print(f"_step: {s.get('_step')}")

print(f"\n=== VAL PRIMARY METRICS ===")
val_keys = [k for k in s.keys() if 'val' in k.lower() or 'surface_pressure' in k.lower() or 'SP' in k or 'VP' in k or 'WSS' in k or 'abupt' in k.lower()]
for k in sorted(val_keys):
    print(f"  {k}: {s.get(k)}")

print(f"\n=== ALL SUMMARY KEYS (filtered) ===")
for k in sorted(s.keys()):
    v = s.get(k)
    if any(x in k for x in ['val', 'test', 'loss', 'grad', 'nan', 'NaN', 'SP', 'VP', 'WSS', 'abupt', 'surface', 'volume']):
        print(f"  {k}: {v}")

print(f"\n=== RUNTIME ===")
print(f"runtime: {s.get('_runtime')} seconds")
if s.get('_runtime'):
    rt = s.get('_runtime')
    print(f"  = {rt/3600:.2f} hours")
print(f"timestamp: {s.get('_timestamp')}")

# Check test metrics
print(f"\n=== TEST METRICS IN SUMMARY ===")
test_keys = [k for k in s.keys() if k.startswith('test')]
if test_keys:
    for k in sorted(test_keys):
        print(f"  {k}: {s.get(k)}")
else:
    print("  (none found)")

# Now scan history for loss and grad norm
print(f"\n=== SCANNING HISTORY FOR LOSS/GRAD ===")
keys_to_scan = [
    "loss/surface_loss", "loss/volume_loss", "loss/total_loss",
    "train/grad/global_norm", "train/loss/surface_loss", "train/loss/volume_loss",
    "grad/global_norm",
]
# Try to get latest values
try:
    rows = list(run.scan_history(keys=keys_to_scan, page_size=1000))
    if rows:
        last = rows[-1]
        print(f"  Last row step: {last.get('_step')}")
        for k in keys_to_scan:
            if k in last:
                print(f"  {k}: {last[k]}")

        # Check surface/volume ratio
        surf = None
        vol = None
        for k in ['loss/surface_loss', 'train/loss/surface_loss']:
            if k in last and last[k] is not None:
                surf = last[k]
                break
        for k in ['loss/volume_loss', 'train/loss/volume_loss']:
            if k in last and last[k] is not None:
                vol = last[k]
                break
        if surf and vol:
            print(f"  surf/vol ratio: {surf/vol:.4f}")
    else:
        print("  No rows returned")
except Exception as e:
    print(f"  Error: {e}")

# Get all loss-related keys from summary
print(f"\n=== LOSS VALUES FROM SUMMARY ===")
for k in sorted(s.keys()):
    if 'loss' in k.lower():
        print(f"  {k}: {s.get(k)}")

print(f"\n=== GRAD NORM FROM SUMMARY ===")
for k in sorted(s.keys()):
    if 'grad' in k.lower() or 'norm' in k.lower():
        print(f"  {k}: {s.get(k)}")

print(f"\n=== NaN COUNTS FROM SUMMARY ===")
for k in sorted(s.keys()):
    if 'nan' in k.lower():
        print(f"  {k}: {s.get(k)}")

# Best val checkpoint
print(f"\n=== BEST CHECKPOINT METRICS ===")
for k in sorted(s.keys()):
    if 'best' in k.lower():
        print(f"  {k}: {s.get(k)}")
