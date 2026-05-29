"""H235 helper: download the EP13 EMA checkpoint for each mirror candidate run.

Each W&B run logs a single `model` artifact aliased `best` / `epoch-13` / `latest`,
which is the EP13 EMA checkpoint we want. Drop them under ``artifacts/`` so
``eval_tta_h209.py --checkpoint`` can pick them up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import wandb

CANDIDATES = [
    ("yw2a5dyl", "H185"),
    ("5k58uzqc", "H183"),
    ("9f2jtrg2", "H190"),
    ("18t5rx2t", "H188"),
    ("2qr5guel", "H148"),
    ("5y5a5tgr", "H191"),
    ("w7w92npw", "H181b"),
    ("d15dm825", "H186"),
]


def main() -> None:
    api = wandb.Api()
    entity = "wandb-applied-ai-team"
    project = "senpai-v1-drivaerml-ddp8"
    out_root = Path("artifacts")
    out_root.mkdir(exist_ok=True)
    paths: list[tuple[str, str, str]] = []
    for run_id, label in CANDIDATES:
        run = api.run(f"{entity}/{project}/{run_id}")
        match = None
        for art in run.logged_artifacts():
            if art.type != "model":
                continue
            if "epoch-13" in art.aliases or "best" in art.aliases:
                match = art
                break
        if match is None:
            print(f"!! no model artifact with alias epoch-13/best on run {run_id} ({label})", file=sys.stderr)
            continue
        download_dir = match.download()
        ckpt = Path(download_dir) / "checkpoint.pt"
        if not ckpt.exists():
            print(f"!! checkpoint.pt missing under {download_dir} for {run_id} ({label})", file=sys.stderr)
            continue
        paths.append((run_id, label, str(ckpt)))
        print(f"OK {label:8s} {run_id} -> {ckpt}")
    print("\n# all_paths")
    for run_id, label, p in paths:
        print(f"{label}\t{run_id}\t{p}")


if __name__ == "__main__":
    main()
