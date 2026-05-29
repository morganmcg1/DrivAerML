# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""H268: Antithetic weight-noise TTA on H185 EP13 — variance reduction via ±δ pairs.

Eval-only probe of antithetic-variates noise sampling. The H242 noise harness draws
K *independent* random weight perturbations δ_k and averages f(w+δ_k). H268 draws K
*paired* perturbations: for each pair index k, sample one δ_k and evaluate both
f(w+δ_k) and f(w−δ_k). The 2K predictions are averaged.

Theory: f(w+δ) = f(w) + ∇f·δ + ½δᵀHδ + O(δ³). The antisymmetric (linear-gradient)
term cancels in the pair average, leaving the curvature term + higher-order noise.
For smooth activations (GELU/SiLU here — confirmed in model.py), f(w+δ) and f(w−δ)
are strongly negatively correlated (ρ ≈ -1 + O(σ²)), so the variance of the pair
mean approaches 0 in the linear regime — up to a 2× reduction at equal compute
relative to K=2 independent samples.

Modes:
    antithetic_noise_only : K pairs (2K passes total). Predictions averaged.
    noise_only            : K independent samples (matches H242 harness exactly,
                            used as a control to confirm equivalence to H242 ref).
    Both modes also report `orig` (clean weights, single pass) for reference.

Mirror convention (H148/H183/H209), unchanged:
    surface_x [x, y, z, nx, ny, nz, area] -> negate y(1) and ny(4)
    volume_x  [x, y, z, sdf]              -> negate y(1)
    surface_y predictions [cp, tau_x, tau_y, tau_z] -> un-mirror by negating tau_y(2)

DDP correctness: weight perturbations must be identical across ranks. Each rank
draws noise from a device-local Generator seeded by `(seed_base + k) * 100003`,
where k is the pair index for antithetic mode or pass index for noise_only mode.
Relative noise: `δ = randn(seed) * sigma * |clean_p|` — exact H242 mechanism.

Efficient antithetic application (per pair):
    p.add_(δ)        # +δ pass
    forward; accumulate
    p.add_(-2δ)      # flip to -δ in place
    forward; accumulate
    p.add_(δ)        # restore clean weights
This avoids the snapshot/restore round-trips and keeps |w| computation once per
pair (delta buffer is materialized once and reused for the +/- flip).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass, fields
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb

from data import SurfaceBatch
from model import SurfaceTransolver
from trainer_runtime import (
    EvalAccumulator,
    TargetTransform,
    _accumulate_case_rel_l2,
    _masked_sse_count,
    autocast_context,
    cleanup_distributed,
    finalize_eval_accumulator,
    init_distributed,
    make_loaders,
    merge_eval_accumulators,
    primary_metric_log,
    print_metrics,
    unwrap_model,
)


@dataclass
class EvalConfig:
    """H268 antithetic-pair noise TTA config."""

    checkpoint: str = "runs/h210/artifacts/h185/checkpoint.pt"
    manifest: str = "data/split_manifest.json"
    data_root: str = "/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
    output_dir: str = "outputs/h268_eval"
    wandb_group: str = "h268-askeladd-antithetic-noise"
    wandb_name: str = ""
    agent: str = "askeladd"

    # Eval-only loader params (match H242)
    batch_size: int = 4
    eval_surface_points: int = 65536
    eval_volume_points: int = 65536
    train_surface_points: int = 65536
    train_volume_points: int = 65536
    num_workers: int = -1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Model arch (must match H185 yw2a5dyl exactly)
    model_layers: int = 5
    model_hidden_dim: int = 512
    model_heads: int = 4
    model_mlp_ratio: int = 4
    model_slices: int = 128
    model_dropout: float = 0.0
    rff_num_features: int = 16
    rff_sigma: float = 1.0
    rff_init_sigmas: str = "0.25,0.5,1.0,2.0,4.0"
    pos_encoding_mode: str = "string_separable"
    use_qk_norm: bool = True
    use_surf_to_vol_xattn: bool = True
    drop_path_max: float = 0.1

    amp_mode: str = "bf16"
    debug: bool = False

    # H268 noise-TTA knobs
    weight_noise_sigma: float = 5e-4
    weight_noise_pairs: int = 3                 # antithetic pair count (2K passes)
    weight_noise_passes: int = 5                # independent-pass count for noise_only control
    weight_noise_seed_base: int = 42
    eval_modes: str = "antithetic_noise_only"   # "antithetic_noise_only" | "noise_only" | both comma-separated


def parse_args(argv: Iterable[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="H268 antithetic weight-noise TTA eval")
    defaults = EvalConfig()
    for f in fields(EvalConfig):
        v = getattr(defaults, f.name)
        cli = f"--{f.name.replace('_', '-')}"
        if isinstance(v, bool):
            parser.add_argument(cli, action="store_true", default=v, dest=f.name)
            parser.add_argument(
                f"--no-{f.name.replace('_', '-')}",
                action="store_false",
                dest=f.name,
            )
        else:
            parser.add_argument(cli, type=type(v), default=v)
    ns = parser.parse_args(argv)
    cfg = EvalConfig(**{f.name: getattr(ns, f.name) for f in fields(EvalConfig)})
    return cfg


def parse_rff_init_sigmas(spec: str) -> list[float] | None:
    if not spec:
        return None
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_eval_modes(spec: str) -> list[str]:
    return [m.strip() for m in spec.split(",") if m.strip()]


def build_model(cfg: EvalConfig) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=cfg.model_layers,
        n_hidden=cfg.model_hidden_dim,
        dropout=cfg.model_dropout,
        n_head=cfg.model_heads,
        mlp_ratio=cfg.model_mlp_ratio,
        slice_num=cfg.model_slices,
        rff_num_features=cfg.rff_num_features,
        rff_sigma=cfg.rff_sigma,
        rff_init_sigmas=parse_rff_init_sigmas(cfg.rff_init_sigmas),
        pos_encoding_mode=cfg.pos_encoding_mode,
        use_qk_norm=cfg.use_qk_norm,
        use_surf_to_vol_xattn=cfg.use_surf_to_vol_xattn,
        drop_path_max=cfg.drop_path_max,
    )


# --- Eval accumulation shared with H209/H242 ---


def _accumulate_outputs(
    acc: EvalAccumulator,
    batch: SurfaceBatch,
    surface_pred_norm: torch.Tensor,
    volume_pred_norm: torch.Tensor,
    transform: TargetTransform,
) -> None:
    surface_target_norm = transform.apply_surface(batch.surface_y)
    volume_target_norm = transform.apply_volume(batch.volume_y)

    surface_sse, surface_count = _masked_sse_count(
        surface_pred_norm, surface_target_norm, batch.surface_mask
    )
    volume_sse, volume_count = _masked_sse_count(
        volume_pred_norm, volume_target_norm, batch.volume_mask
    )
    acc.surface_loss_sse += surface_sse
    acc.surface_loss_count += surface_count
    acc.volume_loss_sse += volume_sse
    acc.volume_loss_count += volume_count

    surface_pred = transform.invert_surface(surface_pred_norm)
    volume_pred = transform.invert_volume(volume_pred_norm)

    if bool(batch.surface_mask.any()):
        surface_abs = (surface_pred - batch.surface_y).abs()
        valid_surface_abs = surface_abs[batch.surface_mask]
        acc.abs_sums["surface_pressure"] += float(
            valid_surface_abs[:, 0].sum().detach().cpu().item()
        )
        acc.abs_counts["surface_pressure"] += int(valid_surface_abs[:, 0].numel())
        wall_abs = valid_surface_abs[:, 1:4]
        acc.abs_sums["wall_shear"] += float(wall_abs.sum().detach().cpu().item())
        acc.abs_counts["wall_shear"] += int(wall_abs.numel())
        for offset, axis in enumerate(("x", "y", "z")):
            channel = wall_abs[:, offset]
            acc.abs_sums[f"wall_shear_{axis}"] += float(channel.sum().detach().cpu().item())
            acc.abs_counts[f"wall_shear_{axis}"] += int(channel.numel())
        wall_vector_error = torch.linalg.vector_norm(
            surface_pred[batch.surface_mask][:, 1:4]
            - batch.surface_y[batch.surface_mask][:, 1:4],
            dim=-1,
        )
        acc.wall_shear_vector_abs_sum += float(
            wall_vector_error.sum().detach().cpu().item()
        )
        acc.wall_shear_vector_count += int(wall_vector_error.numel())

    if bool(batch.volume_mask.any()):
        volume_abs = (volume_pred - batch.volume_y).abs()[batch.volume_mask]
        acc.abs_sums["volume_pressure"] += float(
            volume_abs[:, 0].sum().detach().cpu().item()
        )
        acc.abs_counts["volume_pressure"] += int(volume_abs[:, 0].numel())

    for case_idx, case_id in enumerate(batch.case_ids):
        surface_valid = batch.surface_mask[case_idx].bool()
        if bool(surface_valid.any()):
            surface_pred_valid = surface_pred[case_idx][surface_valid]
            surface_target_valid = batch.surface_y[case_idx][surface_valid]
            _accumulate_case_rel_l2(
                acc.case_sums["surface_pressure"],
                case_id=case_id,
                pred=surface_pred_valid[:, 0:1],
                target=surface_target_valid[:, 0:1],
            )
            _accumulate_case_rel_l2(
                acc.case_sums["wall_shear"],
                case_id=case_id,
                pred=surface_pred_valid[:, 1:4],
                target=surface_target_valid[:, 1:4],
            )
            for channel, axis in enumerate(("x", "y", "z"), start=1):
                _accumulate_case_rel_l2(
                    acc.case_sums[f"wall_shear_{axis}"],
                    case_id=case_id,
                    pred=surface_pred_valid[:, channel : channel + 1],
                    target=surface_target_valid[:, channel : channel + 1],
                )
        volume_valid = batch.volume_mask[case_idx].bool()
        if bool(volume_valid.any()):
            _accumulate_case_rel_l2(
                acc.case_sums["volume_pressure"],
                case_id=case_id,
                pred=volume_pred[case_idx][volume_valid],
                target=batch.volume_y[case_idx][volume_valid],
            )


# --- Weight-noise helpers ---


@torch.no_grad()
def snapshot_clean_params(module: nn.Module) -> dict[str, torch.Tensor]:
    """Pre-copy floating-point parameter tensors for fast restore between passes."""
    snap: dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        if p.dtype.is_floating_point:
            snap[name] = p.detach().clone()
    return snap


@torch.no_grad()
def restore_clean_params(module: nn.Module, clean: dict[str, torch.Tensor]) -> None:
    for name, p in module.named_parameters():
        if name in clean:
            p.data.copy_(clean[name])


@torch.no_grad()
def materialize_delta(
    module: nn.Module,
    clean: dict[str, torch.Tensor],
    sigma: float,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build per-parameter relative-noise delta tensors deterministically.

    delta_p = randn(seed) * sigma * |clean_p|
    Uses a device-local Generator so the draw is identical across DDP ranks
    (same param order and shapes => same delta on every rank).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    delta: dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        clean_p = clean[name]
        noise = torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype)
        noise.mul_(sigma).mul_(clean_p.abs())
        delta[name] = noise
    return delta


@torch.no_grad()
def perturb_relative_(
    module: nn.Module,
    clean: dict[str, torch.Tensor],
    sigma: float,
    seed: int,
    device: torch.device,
) -> None:
    """In-place absolute set: p <- clean_p + randn(seed) * sigma * |clean_p|."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for name, p in module.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        clean_p = clean[name]
        noise = torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype)
        noise.mul_(sigma).mul_(clean_p.abs())
        p.data.copy_(clean_p).add_(noise)


@torch.no_grad()
def apply_delta_(module: nn.Module, delta: dict[str, torch.Tensor], scale: float) -> None:
    """p <- p + scale * delta[name] for every floating-point parameter."""
    for name, p in module.named_parameters():
        if name in delta:
            p.data.add_(delta[name], alpha=scale)


# --- Forward pass helper ---


def _forward(model: nn.Module, batch: SurfaceBatch, amp_mode: str, device: torch.device):
    with autocast_context(device, amp_mode):
        out = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
    return out["surface_preds"].float(), out["volume_preds"].float()


# --- Eval drivers ---


def evaluate_split(
    *,
    model: nn.Module,
    loader,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    distributed_state,
    clean: dict[str, torch.Tensor],
    sigma: float,
    seed_base: int,
    modes: list[str],
    K_pairs: int,
    K_passes: int,
) -> dict[str, dict[str, float]]:
    """Run all requested modes in a single sweep through the loader.

    Active accumulators (only those needed for the requested `modes`):
        orig                  : single clean-weights pass (always produced).
        antithetic_noise_only : K_pairs antithetic ±δ pairs (2*K_pairs passes).
        noise_only            : K_passes independent samples (matches H242).
    """
    want_anti = "antithetic_noise_only" in modes
    want_indep = "noise_only" in modes
    acc_orig = EvalAccumulator()
    acc_anti = EvalAccumulator() if want_anti else None
    acc_indep = EvalAccumulator() if want_indep else None

    model.eval()
    eval_module = unwrap_model(model)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # --- orig pass on clean weights (always) ---
            surface_orig, volume_orig = _forward(eval_module, batch, amp_mode, device)
            _accumulate_outputs(acc_orig, batch, surface_orig, volume_orig, transform)

            # --- antithetic ±δ pairs ---
            if want_anti:
                surface_sum: torch.Tensor | None = None
                volume_sum: torch.Tensor | None = None
                n_anti_passes = 0
                for k in range(K_pairs):
                    seed = (seed_base + k) * 100003
                    delta = materialize_delta(eval_module, clean, sigma, seed, device)

                    # +delta pass
                    apply_delta_(eval_module, delta, +1.0)
                    surface_plus, volume_plus = _forward(eval_module, batch, amp_mode, device)
                    # flip in place to -delta
                    apply_delta_(eval_module, delta, -2.0)
                    surface_minus, volume_minus = _forward(eval_module, batch, amp_mode, device)
                    # restore clean weights
                    apply_delta_(eval_module, delta, +1.0)
                    del delta

                    if surface_sum is None:
                        surface_sum = surface_plus + surface_minus
                        volume_sum = volume_plus + volume_minus
                    else:
                        surface_sum.add_(surface_plus).add_(surface_minus)
                        volume_sum.add_(volume_plus).add_(volume_minus)
                    n_anti_passes += 2

                assert surface_sum is not None and volume_sum is not None
                surface_anti = surface_sum / float(n_anti_passes)
                volume_anti = volume_sum / float(n_anti_passes)
                _accumulate_outputs(acc_anti, batch, surface_anti, volume_anti, transform)

            # --- independent noise passes (H242 control) ---
            if want_indep:
                surface_sum_i: torch.Tensor | None = None
                volume_sum_i: torch.Tensor | None = None
                for k in range(K_passes):
                    seed = (seed_base + k) * 100003
                    perturb_relative_(eval_module, clean, sigma=sigma, seed=seed, device=device)
                    surface_p, volume_p = _forward(eval_module, batch, amp_mode, device)
                    if surface_sum_i is None:
                        surface_sum_i = surface_p.clone()
                        volume_sum_i = volume_p.clone()
                    else:
                        surface_sum_i.add_(surface_p)
                        volume_sum_i.add_(volume_p)
                restore_clean_params(eval_module, clean)
                assert surface_sum_i is not None and volume_sum_i is not None
                surface_ind = surface_sum_i / float(K_passes)
                volume_ind = volume_sum_i / float(K_passes)
                _accumulate_outputs(acc_indep, batch, surface_ind, volume_ind, transform)

    out: dict[str, dict[str, float]] = {}
    for label, acc in (
        ("orig", acc_orig),
        ("antithetic_noise_only", acc_anti),
        ("noise_only", acc_indep),
    ):
        if acc is None:
            continue
        if distributed_state is not None and distributed_state.enabled:
            gathered = [None for _ in range(distributed_state.world_size)]
            dist.all_gather_object(gathered, acc)
            if distributed_state.is_main:
                merged = merge_eval_accumulators(g for g in gathered if g is not None)
                out[label] = finalize_eval_accumulator(merged)
            else:
                out[label] = {}
        else:
            out[label] = finalize_eval_accumulator(acc)
    return out


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    cfg = parse_args(argv)
    device = state.device
    modes = parse_eval_modes(cfg.eval_modes)
    if state.is_main:
        ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
        print(f"Device: {device}{ddp_suffix}")
        print(f"Checkpoint: {cfg.checkpoint}")
        print(f"Modes: {modes}")
        if "antithetic_noise_only" in modes:
            print(
                f"  antithetic: sigma_rel={cfg.weight_noise_sigma}, pairs={cfg.weight_noise_pairs} "
                f"({2 * cfg.weight_noise_pairs} effective passes)"
            )
        if "noise_only" in modes:
            print(
                f"  independent: sigma_rel={cfg.weight_noise_sigma}, K_passes={cfg.weight_noise_passes}"
            )
        print(f"  seed_base={cfg.weight_noise_seed_base}")

    train_loader, val_loaders, test_loaders, stats = make_loaders(cfg, distributed_state=state)
    transform = TargetTransform(
        surface_y_mean=stats["surface_y_mean"].to(device),
        surface_y_std=stats["surface_y_std"].to(device),
        volume_y_mean=stats["volume_y_mean"].to(device),
        volume_y_std=stats["volume_y_std"].to(device),
    )

    model = build_model(cfg).to(device)
    ck = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ck["model"]
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if state.is_main:
        print(f"Loaded checkpoint epoch={ck.get('epoch')} source={ck.get('checkpoint_source')}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  missing[:5]={missing[:5]}")
        if unexpected:
            print(f"  unexpected[:5]={unexpected[:5]}")
    model.eval()

    eval_module = unwrap_model(model)
    clean = snapshot_clean_params(eval_module)
    if state.is_main:
        n_perturbed = sum(t.numel() for t in clean.values())
        print(f"Snapshotted {len(clean)} floating-point tensors ({n_perturbed:,} elements)")

    run = None
    if state.is_main:
        default_name_suffix = (
            f"pairs-{cfg.weight_noise_pairs}"
            if "antithetic_noise_only" in modes
            else f"K-{cfg.weight_noise_passes}"
        )
        run_name = cfg.wandb_name or f"{cfg.agent}/h268-{default_name_suffix}"
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            group=cfg.wandb_group,
            name=run_name,
            config={
                **asdict(cfg),
                "checkpoint_run_id": "yw2a5dyl",
                "checkpoint_epoch": ck.get("epoch"),
                "checkpoint_source": ck.get("checkpoint_source"),
                "modes": modes,
                "antithetic_effective_passes": (
                    2 * cfg.weight_noise_pairs if "antithetic_noise_only" in modes else 0
                ),
            },
            tags=["h268", "tta", "antithetic", "weight-noise", "eval-only", cfg.agent],
            reinit="finish_previous",
        )

    splits = [
        ("val_surface", val_loaders["val_surface"], "full_val"),
        ("test_surface", test_loaders["test_surface"], "test"),
    ]

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name, loader, log_prefix in splits:
        if state.is_main:
            print(f"\n=== Evaluating split={name} ({','.join(modes)}) ===")
        t0 = time.time()
        mode_metrics = evaluate_split(
            model=model,
            loader=loader,
            transform=transform,
            device=device,
            amp_mode=cfg.amp_mode,
            distributed_state=state,
            clean=clean,
            sigma=cfg.weight_noise_sigma,
            seed_base=cfg.weight_noise_seed_base,
            modes=modes,
            K_pairs=cfg.weight_noise_pairs,
            K_passes=cfg.weight_noise_passes,
        )
        dt = time.time() - t0
        if state.is_main:
            print(f"  done in {dt:.1f}s")
            for mode_label, metrics in mode_metrics.items():
                print(f"  -- {mode_label} --")
                print_metrics(name, metrics)
            summary[name] = mode_metrics

            log_obj: dict[str, float] = {}
            for mode_label, metrics in mode_metrics.items():
                log_obj.update(primary_metric_log(f"{log_prefix}_primary/{mode_label}", metrics))
                log_obj.update({f"{log_prefix}_extra/{mode_label}/loss": metrics["loss"]})
            if run is not None:
                wandb.log(log_obj)

    restore_clean_params(eval_module, clean)

    if state.is_main and summary:
        print("\n=== Summary (rel_l2_pct lower-is-better) ===")
        for split, modes_metrics in summary.items():
            print(f"\n[{split}]")
            keys = (
                "abupt_axis_mean_rel_l2_pct",
                "surface_pressure_rel_l2_pct",
                "wall_shear_rel_l2_pct",
                "wall_shear_x_rel_l2_pct",
                "wall_shear_y_rel_l2_pct",
                "wall_shear_z_rel_l2_pct",
                "volume_pressure_rel_l2_pct",
            )
            header = " ".join(f"{m:>14s}" for m in modes_metrics.keys())
            print(f"  {'metric':<36s} {header}")
            for k in keys:
                values = " ".join(f"{modes_metrics[m][k]:>14.4f}" for m in modes_metrics.keys())
                print(f"  {k:<36s} {values}")

        if run is not None:
            for split, modes_metrics in summary.items():
                for mode_label, metrics in modes_metrics.items():
                    for k, v in metrics.items():
                        try:
                            run.summary[f"{split}/{mode_label}/{k}"] = float(v)
                        except Exception:
                            pass
            run.finish()

    cleanup_distributed(state)


if __name__ == "__main__":
    main()
