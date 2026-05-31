"""H307: Linear weight-space averaging of two checkpoints (model soup).

Loads two checkpoints, asserts identical state_dict keys/shapes, then writes
out  alpha*A + (1 - alpha)*B  as a new checkpoint with the same on-disk
structure expected by eval_tta_h252.py (top-level keys: model, epoch,
checkpoint_source, config, selection_metric, val_metrics).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def soup(ckpt_a: Path, ckpt_b: Path, alpha: float, output: Path) -> None:
    ck_a = torch.load(ckpt_a, map_location="cpu", weights_only=False)
    ck_b = torch.load(ckpt_b, map_location="cpu", weights_only=False)
    sd_a: dict[str, torch.Tensor] = ck_a["model"]
    sd_b: dict[str, torch.Tensor] = ck_b["model"]

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    if keys_a != keys_b:
        raise RuntimeError(
            f"state_dict key mismatch: A-B={list(keys_a - keys_b)[:5]} "
            f"B-A={list(keys_b - keys_a)[:5]}"
        )

    soup_sd: dict[str, torch.Tensor] = {}
    n_tensor = 0
    n_float = 0
    n_nonfloat_diff = 0
    delta_norm_total = 0.0
    soup_norm_total = 0.0
    for k in sorted(keys_a):
        ta = sd_a[k]
        tb = sd_b[k]
        if ta.shape != tb.shape:
            raise RuntimeError(f"shape mismatch on {k}: {ta.shape} vs {tb.shape}")
        n_tensor += 1
        if ta.dtype.is_floating_point:
            ta_f = ta.float()
            tb_f = tb.float()
            soup_t = alpha * ta_f + (1.0 - alpha) * tb_f
            soup_sd[k] = soup_t.to(ta.dtype)
            n_float += 1
            delta_norm_total += float((ta_f - tb_f).norm().item() ** 2)
            soup_norm_total += float(soup_t.norm().item() ** 2)
        else:
            # Integer buffers (e.g., num_batches_tracked): require exact match
            # so the soup is well-defined; otherwise prefer A as the canonical.
            if not torch.equal(ta, tb):
                n_nonfloat_diff += 1
            soup_sd[k] = ta.clone()

    print(
        f"Soup: tensors={n_tensor} float={n_float} non-float-mismatches={n_nonfloat_diff}\n"
        f"  ||A-B||_2 (float params only) = {delta_norm_total ** 0.5:.6f}\n"
        f"  ||soup||_2                    = {soup_norm_total ** 0.5:.6f}\n"
        f"  alpha = {alpha:.4f}  (A weight = alpha, B weight = 1 - alpha)"
    )

    # Preserve metadata structure from A so downstream eval can read epoch /
    # config / etc.  Tag provenance so future tooling can detect a soup ckpt.
    out_ck: dict = {
        "model": soup_sd,
        "epoch": ck_a.get("epoch"),
        "checkpoint_source": ck_a.get("checkpoint_source", "ema"),
        "config": ck_a.get("config", {}),
        "selection_metric": ck_a.get("selection_metric"),
        "val_metrics": ck_a.get("val_metrics"),
        "soup": {
            "alpha": alpha,
            "ckpt_a": str(ckpt_a),
            "ckpt_b": str(ckpt_b),
            "epoch_a": ck_a.get("epoch"),
            "epoch_b": ck_b.get("epoch"),
            "source_a": ck_a.get("checkpoint_source"),
            "source_b": ck_b.get("checkpoint_source"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ck, output)
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Saved soup checkpoint -> {output}  ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="H307 weight-space soup")
    p.add_argument("--ckpt-a", required=True, type=Path, help="checkpoint A (weight alpha)")
    p.add_argument("--ckpt-b", required=True, type=Path, help="checkpoint B (weight 1 - alpha)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit(f"alpha must be in [0,1], got {args.alpha}")
    soup(args.ckpt_a, args.ckpt_b, args.alpha, args.output)


if __name__ == "__main__":
    main()
