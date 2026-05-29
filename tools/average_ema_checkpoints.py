"""Average EMA tensor weights from multiple training checkpoints.

Produces a single checkpoint whose `model` state-dict is the element-wise
mean of the input checkpoints' state-dicts. Non-tensor / non-numeric values
(integer counters, strings, NoneType) are taken from the first input.

The output dict preserves the surrounding metadata from the first input
(epoch, config, val_metrics, checkpoint_source, selection_metric) so the
eval pipeline (`eval_multi_res.py`) can load it without modification. We
add an `averaged_from` list recording the input file paths for traceability.
"""

import argparse
import os
import torch


def _avg_state_dict(state_dicts: list[dict]) -> dict:
    """Element-wise mean of a list of state-dicts.

    All inputs must share keys. Non-floating-point tensors (e.g. integer
    buffers) are taken from the first input verbatim; floating-point
    tensors are averaged in fp32 then cast back to the original dtype.
    """
    keys = list(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != set(keys):
            extra = set(sd.keys()) - set(keys)
            missing = set(keys) - set(sd.keys())
            raise ValueError(
                f"Input {i} state-dict keys differ; extra={list(extra)[:5]} missing={list(missing)[:5]}"
            )

    out: dict = {}
    n = len(state_dicts)
    for k in keys:
        t0 = state_dicts[0][k]
        if not torch.is_tensor(t0):
            out[k] = t0
            continue
        if not t0.is_floating_point():
            out[k] = t0.clone()
            continue
        target_dtype = t0.dtype
        acc = torch.zeros_like(t0, dtype=torch.float32)
        for sd in state_dicts:
            acc.add_(sd[k].to(torch.float32))
        acc.div_(n)
        out[k] = acc.to(target_dtype)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to checkpoint files to average (>=2).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the averaged checkpoint.",
    )
    parser.add_argument(
        "--key",
        default=None,
        help=(
            "Top-level key holding the state-dict to average. If omitted, "
            "auto-detect: tries 'ema_state_dict', 'model_ema_state_dict', "
            "then 'model'."
        ),
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("Need at least 2 input checkpoints to average")

    print(f"Loading {len(args.inputs)} checkpoints...")
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in args.inputs]

    if args.key is None:
        candidates = ["ema_state_dict", "model_ema_state_dict", "model"]
        chosen = None
        for c in candidates:
            if c in ckpts[0] and isinstance(ckpts[0][c], dict):
                chosen = c
                break
        if chosen is None:
            raise SystemExit(
                f"Could not auto-detect state-dict key; tried {candidates}; "
                f"top-level keys in first ckpt: {list(ckpts[0].keys())}"
            )
        key = chosen
    else:
        key = args.key

    print(f"Using state-dict key: '{key}'")
    for p, ck in zip(args.inputs, ckpts):
        if key not in ck:
            raise SystemExit(f"Key '{key}' not in {p}; available: {list(ck.keys())}")
        src = ck.get("checkpoint_source")
        ep = ck.get("epoch")
        print(f"  {os.path.basename(p)}: epoch={ep} source={src} n_params={len(ck[key])}")

    state_dicts = [ck[key] for ck in ckpts]
    averaged = _avg_state_dict(state_dicts)

    out = dict(ckpts[0])
    out[key] = averaged
    out["averaged_from"] = [os.path.abspath(p) for p in args.inputs]
    out["averaged_n"] = len(args.inputs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved averaged checkpoint -> {args.output}")
    print(f"  averaged_from = {out['averaged_from']}")


if __name__ == "__main__":
    main()
