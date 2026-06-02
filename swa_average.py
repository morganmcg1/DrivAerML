# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H352: uniform Stochastic Weight Averaging over cosine-tail snapshots.

Loads all ``swa_step_<step>.pt`` files written by ``train.py`` during the
H336 cosine-tail re-run and produces a single averaged checkpoint suitable
for downstream eval (``eval_tta_h252.py``). Snapshots are accumulated in
float64 to preserve precision across ~30 weights; the final averaged
state-dict is cast back to the snapshot dtype before saving.

Two averaging modes, matching the H352 PR arms:

* ``--mode uniform_all``  → Arm A: uniform mean over every snapshot in the
  directory.
* ``--mode last_n --last-n 10`` → Arm B: uniform mean over the ``--last-n``
  snapshots with the largest step indices (the lowest-LR tail).

Optional ``--ref-checkpoint`` copies non-model fields (config, epoch,
checkpoint_source) from a reference checkpoint so the resulting file looks
like a standard train.py checkpoint to eval scripts. When not provided,
the output checkpoint is a minimal ``{"model": <averaged-state-dict>}``
plus SWA provenance metadata.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import torch


SNAPSHOT_RE = re.compile(r"swa_step_(\d+)\.pt$")


@dataclass
class SwaArgs:
    snapshot_dir: str
    output: str
    mode: str
    last_n: int
    ref_checkpoint: str
    dtype: str


def parse_args() -> SwaArgs:
    parser = argparse.ArgumentParser(
        description="H352: uniform SWA averaging over cosine-tail snapshots"
    )
    parser.add_argument(
        "--snapshot-dir",
        required=True,
        help="Directory containing swa_step_<step>.pt snapshot files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the averaged checkpoint (.pt).",
    )
    parser.add_argument(
        "--mode",
        choices=["uniform_all", "last_n"],
        default="uniform_all",
        help=(
            "Averaging policy. 'uniform_all' averages every snapshot in "
            "the directory (Arm A). 'last_n' averages the --last-n "
            "snapshots with the largest step indices (Arm B = late-tail)."
        ),
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="Number of late-tail snapshots when --mode=last_n (default 10).",
    )
    parser.add_argument(
        "--ref-checkpoint",
        default="",
        help=(
            "Optional reference checkpoint (.pt) to copy non-model fields "
            "from (config, epoch, etc.). When provided, the output looks "
            "like a standard train.py checkpoint to eval scripts."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help=(
            "Output tensor dtype (default float32). Internal accumulation "
            "is always float64 regardless."
        ),
    )
    ns = parser.parse_args()
    return SwaArgs(
        snapshot_dir=ns.snapshot_dir,
        output=ns.output,
        mode=ns.mode,
        last_n=ns.last_n,
        ref_checkpoint=ns.ref_checkpoint,
        dtype=ns.dtype,
    )


def discover_snapshots(snapshot_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for path in snapshot_dir.iterdir():
        match = SNAPSHOT_RE.search(path.name)
        if match is None:
            continue
        out.append((int(match.group(1)), path))
    out.sort(key=lambda kv: kv[0])
    return out


def average_snapshots(paths: list[Path], out_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    if not paths:
        raise ValueError("No snapshots provided to average.")
    sum_state: dict[str, torch.Tensor] = {}
    n = 0
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not sum_state:
            for key, tensor in state.items():
                if torch.is_tensor(tensor) and tensor.is_floating_point():
                    sum_state[key] = tensor.detach().to(torch.float64).clone()
                else:
                    sum_state[key] = tensor
        else:
            for key, tensor in state.items():
                if key not in sum_state:
                    raise KeyError(f"Snapshot {path} has unexpected key {key!r}")
                if torch.is_tensor(tensor) and tensor.is_floating_point():
                    sum_state[key].add_(tensor.detach().to(torch.float64))
        n += 1
    averaged: dict[str, torch.Tensor] = {}
    for key, value in sum_state.items():
        if torch.is_tensor(value) and value.is_floating_point():
            averaged[key] = (value / n).to(out_dtype)
        else:
            averaged[key] = value
    return averaged


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    if not snapshot_dir.is_dir():
        raise SystemExit(f"Snapshot directory does not exist: {snapshot_dir}")

    snapshots = discover_snapshots(snapshot_dir)
    if not snapshots:
        raise SystemExit(f"No swa_step_*.pt snapshots found in {snapshot_dir}")

    if args.mode == "uniform_all":
        selected = snapshots
    elif args.mode == "last_n":
        if args.last_n <= 0:
            raise SystemExit(f"--last-n must be positive, got {args.last_n}")
        selected = snapshots[-args.last_n :]
    else:
        raise SystemExit(f"Unknown --mode: {args.mode}")

    selected_paths = [path for _, path in selected]
    out_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    print(
        f"H352 SWA averaging: dir={snapshot_dir} mode={args.mode} "
        f"selected={len(selected_paths)}/{len(snapshots)} "
        f"step_range=[{selected[0][0]}, {selected[-1][0]}] "
        f"dtype={args.dtype}"
    )
    averaged = average_snapshots(selected_paths, out_dtype)

    checkpoint: dict[str, object] = {
        "model": averaged,
        "h352_swa": {
            "mode": args.mode,
            "last_n": args.last_n if args.mode == "last_n" else None,
            "n_snapshots": len(selected_paths),
            "step_range": [selected[0][0], selected[-1][0]],
            "snapshot_steps": [step for step, _ in selected],
            "snapshot_dir": str(snapshot_dir),
        },
    }

    if args.ref_checkpoint:
        ref_path = Path(args.ref_checkpoint)
        if not ref_path.is_file():
            raise SystemExit(f"--ref-checkpoint not found: {ref_path}")
        ref = torch.load(ref_path, map_location="cpu", weights_only=False)
        for key in ("config", "epoch", "val_metrics", "selection_metric"):
            if key in ref:
                checkpoint[key] = ref[key]
        checkpoint["checkpoint_source"] = f"swa_{args.mode}"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Wrote averaged checkpoint to {output}")
    print(
        f"  averaged_keys={sum(1 for v in averaged.values() if torch.is_tensor(v) and v.is_floating_point())} "
        f"float-tensors; mode={args.mode}; n_snapshots={len(selected_paths)}"
    )


if __name__ == "__main__":
    main()
