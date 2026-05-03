"""
Get current epoch for W&B runs - check summary keys directly.
"""
import os
import wandb

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

RUNS = {
    "hph6eaky": "fern / PR#409",
    "5ifnf1wc": "thorfinn / PR#382",
    "4632xosf": "kohaku / PR#417",
    "0xi2n4oo": "alphonse / PR#437",
    "jj9r7x0o": "senku / PR#442",
    "vyhpqruv": "tanjiro / PR#443",
}

for run_id, label in RUNS.items():
    try:
        run = api.run(f"{path}/{run_id}")
        summary = run.summary_metrics
        # Print all epoch-related keys
        epoch_keys = {k: v for k, v in summary.items() if "epoch" in k.lower() or k == "_step"}
        print(f"{run_id} ({label}): _step={run.lastHistoryStep}, epoch_keys={epoch_keys}")
    except Exception as e:
        print(f"{run_id}: ERROR {e}")
