"""
Get current epoch for W&B runs by scanning recent history.
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

# Epoch keys to check
EPOCH_KEYS = ["epoch", "trainer/epoch", "_step", "val_epoch", "train/epoch"]

for run_id, label in RUNS.items():
    try:
        run = api.run(f"{path}/{run_id}")
        summary = run.summary_metrics

        # Check all keys in summary for epoch
        epoch_val = None
        for k in EPOCH_KEYS:
            v = summary.get(k)
            if v is not None:
                epoch_val = f"{k}={v}"
                break

        # Also scan last few history rows
        history_rows = list(run.scan_history(keys=["epoch", "_step"], min_step=max(0, run.lastHistoryStep - 5)))
        last_row = history_rows[-1] if history_rows else {}
        step = last_row.get("_step", "?")
        ep_from_hist = last_row.get("epoch", "?")

        print(f"{run_id} ({label}): summary_epoch={epoch_val}, hist_epoch={ep_from_hist}, _step={step}, state={run.state}")

    except Exception as e:
        print(f"{run_id}: ERROR {e}")
