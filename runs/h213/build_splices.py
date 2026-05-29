"""Build 6 block-spliced checkpoints from H112 and H183 EP13 weights.

k = number of H112 blocks at front; (5-k) blocks at end come from H183.
k=5 = all H112 (sanity), k=0 = all H183 (sanity).

Non-block keys (embedding, projection heads, output heads, cross-attn,
positional encoding, final norm) follow H112 — this is the default consistent
with PR #1387 instructions and makes k=5 reproduce H112 exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


BLOCK_PREFIX = "backbone.blocks"
NUM_BLOCKS = 5


def splice_k(w_h112: dict, w_h183: dict, k: int) -> dict:
    out: dict = {}
    for key, val in w_h112.items():
        if key.startswith(f"{BLOCK_PREFIX}."):
            rest = key.split(f"{BLOCK_PREFIX}.")[1]
            block_idx = int(rest.split(".")[0])
            out[key] = w_h112[key] if block_idx < k else w_h183[key]
        else:
            out[key] = w_h112[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h112", default="runs/h210/artifacts/h112/checkpoint.pt")
    parser.add_argument("--h183", default="runs/h210/artifacts/h183/checkpoint.pt")
    parser.add_argument("--out-dir", default="runs/h213/splices")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck112 = torch.load(args.h112, map_location="cpu", weights_only=False)
    ck183 = torch.load(args.h183, map_location="cpu", weights_only=False)
    sd112 = ck112["model"]
    sd183 = ck183["model"]

    block_keys_h112 = {k: v for k, v in sd112.items() if k.startswith(f"{BLOCK_PREFIX}.")}
    block_keys_h183 = {k: v for k, v in sd183.items() if k.startswith(f"{BLOCK_PREFIX}.")}
    non_block_keys = {k: v for k, v in sd112.items() if not k.startswith(f"{BLOCK_PREFIX}.")}
    print(
        f"H112: {len(sd112)} keys -> {len(block_keys_h112)} block, "
        f"{len(non_block_keys)} non-block."
    )

    block_indices = sorted({int(k.split(".")[2]) for k in block_keys_h112})
    if block_indices != list(range(NUM_BLOCKS)):
        raise RuntimeError(
            f"Expected blocks {list(range(NUM_BLOCKS))} but found {block_indices}"
        )

    metadata = {
        "config": ck112["config"],
        "epoch": ck112["epoch"],
        "selection_metric": ck112.get("selection_metric"),
        "checkpoint_source": "splice",
    }

    for k in range(NUM_BLOCKS + 1):
        spliced = splice_k(sd112, sd183, k)
        out_path = out_dir / f"splice_k{k}.pt"
        torch.save({**metadata, "model": spliced, "splice_k": k}, out_path)
        print(f"saved {out_path}")

    print("\n=== sanity check ===")
    # Verify k=5 reproduces H112 (5 spot checks across key types).
    ck5 = torch.load(out_dir / "splice_k5.pt", map_location="cpu", weights_only=False)
    sd5 = ck5["model"]
    spot_keys = [
        "backbone.blocks.0.norm1.weight",
        "backbone.blocks.4.norm1.weight",
        "pos_embed.omega",
        "surface_out.0.weight",
        "norm.weight",
    ]
    for key in spot_keys:
        match = torch.equal(sd5[key], sd112[key])
        print(f"  k=5 ⇄ H112 [{key}]: {match}")

    # Verify k=0 reproduces H183 blocks
    ck0 = torch.load(out_dir / "splice_k0.pt", map_location="cpu", weights_only=False)
    sd0 = ck0["model"]
    for key in [
        "backbone.blocks.0.norm1.weight",
        "backbone.blocks.4.norm1.weight",
    ]:
        match = torch.equal(sd0[key], sd183[key])
        print(f"  k=0 ⇄ H183 [{key}]: {match}")
    # Confirm k=0 non-block keys still equal H112 (per advisor's note)
    print(
        f"  k=0 pos_embed equals H112: "
        f"{torch.equal(sd0['pos_embed.omega'], sd112['pos_embed.omega'])}"
    )

    # Verify mid-k: k=2 has blocks 0,1 from H112 and 2,3,4 from H183
    ck2 = torch.load(out_dir / "splice_k2.pt", map_location="cpu", weights_only=False)
    sd2 = ck2["model"]
    for blk in [0, 1]:
        key = f"backbone.blocks.{blk}.norm1.weight"
        print(f"  k=2 block {blk} from H112: {torch.equal(sd2[key], sd112[key])}")
    for blk in [2, 3, 4]:
        key = f"backbone.blocks.{blk}.norm1.weight"
        print(f"  k=2 block {blk} from H183: {torch.equal(sd2[key], sd183[key])}")


if __name__ == "__main__":
    main()
