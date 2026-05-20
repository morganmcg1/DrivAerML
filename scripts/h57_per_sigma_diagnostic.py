"""H57 per-sigma frequency-band utilization diagnostic.

For a StringSeparableEncoding checkpoint, report (per init sigma group):

  1) Learned log-frequency stats — how far each sigma's features have drifted
     from their init value.  If a sigma stays near init, gradient descent has
     not had reason to move it.  If it drifts substantially, the optimizer is
     actively re-shaping that frequency band.

  2) Activation L2 norm — feed random coords drawn uniformly from the
     normalized cube [-1, 1]^3 (matches the normalized-canon range the model
     sees during training) and group the encoded output columns by their init
     sigma.  Tells us how much energy each sigma band carries in the encoded
     representation.

Usage:

    python scripts/h57_per_sigma_diagnostic.py \\
        --ckpt outputs/drivaerml/run-<id>/checkpoint.pt \\
        [--init-sigmas 0.125,0.25,0.5,1.0,2.0,4.0,8.0,16.0] \\
        [--n-coords 65536]

If --init-sigmas is omitted, the value is read from the checkpoint's config.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import StringSeparableEncoding  # noqa: E402


def parse_sigmas(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def per_sigma_stats(
    log_freq: torch.Tensor, init_sigmas: list[float]
) -> list[dict[str, float]]:
    """log_freq: [in_dim, num_features]. Returns one dict per init sigma."""
    in_dim, num_features = log_freq.shape
    rows = []
    for s_idx, sigma in enumerate(init_sigmas):
        feat_idx = [f for f in range(num_features) if f % len(init_sigmas) == s_idx]
        if not feat_idx:
            continue
        subset = log_freq[:, feat_idx]  # [in_dim, n_per_sigma]
        learned_freq = subset.exp()
        init_log = math.log(sigma)
        drift = (subset - init_log).abs()
        rows.append({
            "sigma_idx": s_idx,
            "init_sigma": sigma,
            "n_features": subset.numel(),
            "learned_freq_mean": float(learned_freq.mean()),
            "learned_freq_std": float(learned_freq.std()),
            "learned_freq_min": float(learned_freq.min()),
            "learned_freq_max": float(learned_freq.max()),
            "abs_log_drift_mean": float(drift.mean()),
            "abs_log_drift_max": float(drift.max()),
            "per_axis_learned_freq_mean": [
                float(learned_freq[d].mean()) for d in range(in_dim)
            ],
        })
    return rows


def per_sigma_activation_l2(
    enc: StringSeparableEncoding,
    init_sigmas: list[float],
    n_coords: int,
    device: torch.device,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Run uniform random coords in [-1, 1]^3 through enc and group output
    columns by init sigma. enc output is [N, 2 * in_dim * num_features] with
    layout (after flatten): for each axis d, [sin(0..F-1), cos(0..F-1)]."""
    in_dim = enc.in_dim
    num_features = enc.num_features
    g = torch.Generator(device="cpu").manual_seed(seed)
    coords = (torch.rand(n_coords, in_dim, generator=g) * 2.0 - 1.0).to(device)
    with torch.no_grad():
        out = enc(coords)  # [N, 2 * in_dim * num_features]
    out = out.float().cpu()
    # Reshape: [N, in_dim, 2 * num_features] where last 2F splits into [sin(F), cos(F)]
    out = out.view(n_coords, in_dim, 2 * num_features)
    sin_part = out[..., :num_features]   # [N, in_dim, F]
    cos_part = out[..., num_features:]   # [N, in_dim, F]
    sin_sq = (sin_part ** 2).sum(dim=0)   # [in_dim, F]
    cos_sq = (cos_part ** 2).sum(dim=0)   # [in_dim, F]
    rows = []
    total_sq = float((sin_sq + cos_sq).sum())
    for s_idx, sigma in enumerate(init_sigmas):
        feat_idx = [f for f in range(num_features) if f % len(init_sigmas) == s_idx]
        if not feat_idx:
            continue
        feat_idx_t = torch.tensor(feat_idx, dtype=torch.long)
        sin_g = sin_sq[:, feat_idx_t].sum()
        cos_g = cos_sq[:, feat_idx_t].sum()
        group_sq = float(sin_g + cos_g)
        per_axis_l2 = [
            math.sqrt(float((sin_sq[d, feat_idx_t] + cos_sq[d, feat_idx_t]).sum()))
            for d in range(in_dim)
        ]
        rows.append({
            "sigma_idx": s_idx,
            "init_sigma": sigma,
            "n_columns": 2 * len(feat_idx) * in_dim,
            "activation_l2": math.sqrt(group_sq),
            "frac_of_total_l2sq": group_sq / total_sq if total_sq > 0 else 0.0,
            "per_axis_l2": per_axis_l2,
        })
    return rows


def per_sigma_projection_weight_l2(
    proj_weight: torch.Tensor,
    init_sigmas: list[float],
    num_features: int,
    in_dim: int,
    extra_dim: int,
) -> list[dict[str, float]]:
    """proj_weight: [hidden_dim, extra_dim + 2*in_dim*num_features]. The columns
    after extra_dim correspond to the string_sep flattened output. Layout per
    axis d: columns [extra + d*2F .. extra + d*2F + F) are sin; the next F are
    cos. For sigma index s, the relevant feature indices f satisfy f % S == s
    where S = len(init_sigmas).

    Per-sigma weight L2 measures how strongly the downstream projection reads
    each frequency band. At init (trunc_normal std=0.02), every group starts
    at the same L2; deviations mean the model has learned to amplify or
    suppress that band.
    """
    F = num_features
    rows = []
    total_string_sep_l2sq = float((proj_weight[:, extra_dim:] ** 2).sum())
    for s_idx, sigma in enumerate(init_sigmas):
        feat_idx = [f for f in range(F) if f % len(init_sigmas) == s_idx]
        if not feat_idx:
            continue
        cols = []
        for d in range(in_dim):
            base = extra_dim + d * 2 * F
            for f in feat_idx:
                cols.append(base + f)            # sin
                cols.append(base + F + f)        # cos
        col_idx = torch.tensor(cols, dtype=torch.long)
        block = proj_weight[:, col_idx]
        group_l2sq = float((block ** 2).sum())
        rows.append({
            "sigma_idx": s_idx,
            "init_sigma": sigma,
            "n_columns": len(cols),
            "weight_l2_per_column_mean": math.sqrt(group_l2sq / len(cols)),
            "weight_l2_total": math.sqrt(group_l2sq),
            "frac_of_string_sep_l2sq": group_l2sq / total_string_sep_l2sq
                if total_string_sep_l2sq > 0 else 0.0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--init-sigmas", type=str, default=None,
                        help="Override init_sigmas from checkpoint config")
    parser.add_argument("--n-coords", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    cfg = ckpt.get("config", {}) or {}

    if args.init_sigmas:
        init_sigmas = parse_sigmas(args.init_sigmas)
    else:
        raw = cfg.get("rff_init_sigmas") if isinstance(cfg, dict) else getattr(cfg, "rff_init_sigmas", None)
        if isinstance(raw, str):
            init_sigmas = parse_sigmas(raw)
        elif isinstance(raw, (list, tuple)):
            init_sigmas = [float(x) for x in raw]
        else:
            raise SystemExit("Could not read init_sigmas from ckpt; pass --init-sigmas")

    epoch = ckpt.get("epoch")
    surface_lf = state["surface_string_sep.log_freq"].cpu()
    volume_lf = state["volume_string_sep.log_freq"].cpu()
    in_dim, num_features = surface_lf.shape

    # Build fresh encoders with checkpoint weights for activation analysis.
    surf_enc = StringSeparableEncoding(
        in_dim=in_dim, num_features=num_features, init_sigmas=init_sigmas
    ).to(device)
    surf_enc.load_state_dict({
        "log_freq": state["surface_string_sep.log_freq"].to(device),
        "phase": state["surface_string_sep.phase"].to(device),
    })
    surf_enc.eval()
    vol_enc = StringSeparableEncoding(
        in_dim=in_dim, num_features=num_features, init_sigmas=init_sigmas
    ).to(device)
    vol_enc.load_state_dict({
        "log_freq": state["volume_string_sep.log_freq"].to(device),
        "phase": state["volume_string_sep.phase"].to(device),
    })
    vol_enc.eval()

    surf_proj_w = state.get("project_surface_features.project.weight")
    vol_proj_w = state.get("project_volume_features.project.weight")
    string_sep_out_dim = 2 * in_dim * num_features
    surf_extra = (surf_proj_w.shape[1] - string_sep_out_dim) if surf_proj_w is not None else 0
    vol_extra = (vol_proj_w.shape[1] - string_sep_out_dim) if vol_proj_w is not None else 0

    report = {
        "ckpt": str(args.ckpt),
        "epoch": epoch,
        "in_dim": in_dim,
        "num_features": num_features,
        "init_sigmas": init_sigmas,
        "n_coords": args.n_coords,
        "surface": {
            "log_freq_stats": per_sigma_stats(surface_lf, init_sigmas),
            "activation_l2": per_sigma_activation_l2(
                surf_enc, init_sigmas, args.n_coords, device, seed=args.seed
            ),
            "proj_weight_l2": (
                per_sigma_projection_weight_l2(
                    surf_proj_w.cpu(), init_sigmas, num_features, in_dim, surf_extra
                ) if surf_proj_w is not None else []
            ),
            "extra_dim": surf_extra,
        },
        "volume": {
            "log_freq_stats": per_sigma_stats(volume_lf, init_sigmas),
            "activation_l2": per_sigma_activation_l2(
                vol_enc, init_sigmas, args.n_coords, device, seed=args.seed
            ),
            "proj_weight_l2": (
                per_sigma_projection_weight_l2(
                    vol_proj_w.cpu(), init_sigmas, num_features, in_dim, vol_extra
                ) if vol_proj_w is not None else []
            ),
            "extra_dim": vol_extra,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"=== H57 per-sigma diagnostic ===")
    print(f"ckpt: {args.ckpt}")
    print(f"epoch: {epoch}")
    print(f"init_sigmas: {init_sigmas}")
    print(f"in_dim={in_dim}, num_features={num_features}, n_coords={args.n_coords}")
    print()

    for branch in ("surface", "volume"):
        print(f"--- {branch.upper()} encoder ---")
        print(
            f"{'sigma':>8} | {'learned freq':>14} | {'fr std':>8} | "
            f"{'|log drift|':>12} | {'proj W L2/col':>14} | {'proj frac':>10}"
        )
        stats = {s["sigma_idx"]: s for s in report[branch]["log_freq_stats"]}
        proj = {p["sigma_idx"]: p for p in report[branch]["proj_weight_l2"]}
        for s_idx, sigma in enumerate(init_sigmas):
            st = stats.get(s_idx)
            pj = proj.get(s_idx)
            if not st:
                continue
            pj_l2 = pj["weight_l2_per_column_mean"] if pj else float("nan")
            pj_frac = (pj["frac_of_string_sep_l2sq"] * 100) if pj else float("nan")
            print(
                f"{sigma:>8.3f} | {st['learned_freq_mean']:>14.4f} | "
                f"{st['learned_freq_std']:>8.4f} | {st['abs_log_drift_mean']:>12.4f} | "
                f"{pj_l2:>14.4f} | {pj_frac:>9.2f}%"
            )
        print()


if __name__ == "__main__":
    main()
