# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Train a grouped surface/volume Transolver on DrivAerML.

Primary metric:
    val_primary/abupt_axis_mean_rel_l2_pct

Usage:
    python train.py --epochs 50 --agent <name> --wandb_name "<name>/<experiment>"
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import DrivAerMLSurfaceDataset
from model import SurfaceTransolver
from trainer_runtime import (
    EMA,
    MetricSlopeTracker,
    TargetTransform,
    autocast_context,
    build_lr_scheduler,
    check_kill_thresholds,
    cleanup_distributed,
    collect_gradient_metrics,
    collect_weight_metrics,
    distributed_any,
    distributed_barrier,
    evaluate_split,
    full_eval_loaders_from,
    global_grad_norm,
    init_distributed,
    init_wandb_run,
    is_valid_primary_metric,
    loader_kwargs,
    make_loaders,
    masked_mse,
    metric_namespace,
    weighted_channel_mse,
    parse_kill_thresholds,
    primary_metric_log,
    print_metrics,
    run_final_evaluation,
    should_update_best_checkpoint,
    timeout_budget_minutes,
    unwrap_model,
)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


@dataclass
class Config:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 50
    train_surface_points: int = 40_000
    eval_surface_points: int = 40_000
    train_volume_points: int = 40_000
    eval_volume_points: int = 40_000
    validation_every: int = 1
    surface_loss_weight: float = 1.0
    volume_loss_weight: float = 1.0
    manifest: str = "data/split_manifest.json"
    data_root: str = ""
    output_dir: str = "outputs/drivaerml"
    wandb_group: str = ""
    wandb_name: str = ""
    agent: str = ""
    model_layers: int = 3
    model_hidden_dim: int = 192
    model_heads: int = 3
    model_mlp_ratio: int = 4
    model_slices: int = 96
    model_dropout: float = 0.0
    rff_num_features: int = 0
    rff_sigma: float = 1.0
    rff_init_sigmas: str = ""
    pos_encoding_mode: str = "sincos"
    use_qk_norm: bool = False
    use_surf_to_vol_xattn: bool = False
    drop_path_max: float = 0.0
    tau_y_loss_weight: float = 1.0
    tau_z_loss_weight: float = 1.0
    amp_mode: str = "bf16"
    num_workers: int = -1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_start_step: int = 50
    eval_raw_vs_ema: bool = False
    lr_warmup_epochs: int = 0
    lr_cosine_t_max: int = 0
    lr_min: float = 1e-6
    grad_clip_norm: float = 1.0
    gradient_log_every: int = 250
    log_gradient_histograms: bool = False
    weight_log_every: int = 250
    log_weight_histograms: bool = False
    ddp_log_model_telemetry_all_ranks: bool = False
    slope_log_fraction: float = 0.05
    kill_thresholds: str = ""
    compile_model: bool = True
    optimizer: str = "adamw"
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    vol_points_schedule: str = ""
    use_gradnorm: bool = False
    gradnorm_mode: str = "ema_proxy"
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 1e-3
    gradnorm_init_warmup_steps: int = 10
    gradnorm_log_clip: float = 4.0
    gradnorm_ema_beta: float = 0.9
    gradnorm_min_weight: float = 0.0
    # H244: resume + extension support
    mirror_augmentation: bool = False
    resume_from_wandb: str = ""
    resume_alias: str = ""
    epochs_already_done: int = 0
    save_every_epoch: bool = False
    # H347: boundary-layer physics-prior auxiliary losses on WSS
    physics_prior_normals: bool = False
    physics_prior_smooth: bool = False
    lambda_normals: float = 1e-3
    lambda_smooth: float = 1e-3
    physics_knn_k: int = 6
    physics_smooth_anchors: int = 8192
    physics_knn_chunk_size: int = 1024
    debug: bool = False


def parse_args(argv: Iterable[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="DrivAerML surface/volume trainer")
    defaults = Config()
    help_text = {
        "kill_thresholds": (
            "Optional early-stop checks. Format: "
            "'STEP:METRIC<NUMBER[,STEP:METRIC>=NUMBER...]'; commas or semicolons "
            "separate checks. STEP is a global optimizer step and METRIC must match "
            "a logged W&B key exactly, for example "
            "'500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25'."
        ),
        "vol_points_schedule": (
            "Optional epoch-based curriculum for the train-volume-points view "
            "size. Format: 'EPOCH:POINTS:EPOCH:POINTS:...' (colon-separated). "
            "The train DataLoader is rebuilt at any epoch where the value "
            "changes. Must start at epoch 0; the value applies from that "
            "epoch onwards (inclusive) until the next breakpoint. Empty "
            "string disables the curriculum and `--train-volume-points` is "
            "used unchanged. Example: '0:16384:3:32768:6:49152:9:65536'."
        ),
        "use_gradnorm": (
            "Enable GradNorm dynamic per-task loss balancing (Chen et al., "
            "NeurIPS 2018, https://arxiv.org/abs/1711.02132). Splits the "
            "surface MSE into 4 per-channel losses (sp, tau_x, tau_y, tau_z) "
            "and treats volume_pressure as the 5th task; per-task weights "
            "are updated each step (algorithm controlled by --gradnorm-mode). "
            "When disabled, the legacy --surface-loss-weight / "
            "--volume-loss-weight scalars apply."
        ),
        "gradnorm_mode": (
            "GradNorm balancing algorithm. 'full' is the paper-faithful "
            "variant: per-task gradient norms at the last shared encoder "
            "block via autograd.grad with retain_graph (5x backward overhead "
            "for 5 tasks). 'ema_proxy' is the lightweight loss-magnitude "
            "approximation: maintains an EMA of per-task losses, computes "
            "relative training rate r_i = ema_i / initial_i, and assigns "
            "weights w_i proportional to r_i**alpha (renormalised so that "
            "mean weight = 1). Trades the formal gradient-alignment "
            "guarantee for ~1x overhead. Default: ema_proxy."
        ),
        "gradnorm_alpha": (
            "Restoring-force exponent alpha. Higher values pull "
            "slow-converging tasks back harder. Paper-recommended range "
            "[0.5, 2.0]; default 1.5 for tasks with different rates. "
            "Applied to relative training rate r_i in both modes."
        ),
        "gradnorm_lr": (
            "Learning rate for the Adam optimizer that updates the per-task "
            "log-weights in mode='full'. Unused in mode='ema_proxy' (weights "
            "are computed in closed form). The brianlan reference uses "
            "~2.5e-2; we default to 1e-3 for slower / stabler movement."
        ),
        "gradnorm_init_warmup_steps": (
            "Number of training steps over which to average the per-task "
            "losses to define L_i(0). Skips step 0 to avoid random-init "
            "spikes; default 10 steps. Applied in both modes."
        ),
        "gradnorm_log_clip": (
            "Symmetric clamp applied to the GradNorm log-weights each step "
            "(weight = exp(log_w) so |log_w|<=clip => weight in "
            "[exp(-clip), exp(clip)]). Prevents runaway divergence and "
            "task collapse. Default 4.0 => weights in [0.018, 54.6]. "
            "Applied in both modes."
        ),
        "gradnorm_ema_beta": (
            "EMA decay for the per-task loss tracker used in "
            "mode='ema_proxy' (ema = beta * ema + (1-beta) * L_i.detach()). "
            "Default 0.9 matches the advisor spec. Unused in mode='full'."
        ),
        "gradnorm_min_weight": (
            "Asymmetric per-task weight floor applied in mode='ema_proxy'. "
            "Each pre-normalisation w_raw_i is clamped to >= min_weight "
            "before the closed-form sum-T renormalisation; the up-weighting "
            "ceiling is left free. Prevents over-aggressive down-weighting "
            "of fast-converging tasks (sp/vp) while still allowing slow "
            "tasks (tau_y/tau_z) to climb. Default 0.0 disables the floor "
            "(matches Run 1 behavior). Recommended 0.7 for soft "
            "redistribution. Unused in mode='full'."
        ),
        "use_surf_to_vol_xattn": (
            "Enable surface->volume cross-attention (PR #823). After the "
            "shared Transolver backbone, a single nn.MultiheadAttention "
            "block lets volume hidden states (Q) attend to surface hidden "
            "states (K/V) before the volume output head. The out_proj "
            "weight and bias are zero-initialised so the layer is identity "
            "at init (preserves baseline at epoch 0). embed_dim follows "
            "--model-hidden-dim and num_heads follows --model-heads."
        ),
        "drop_path_max": (
            "H112: Maximum DropPath (stochastic depth, Huang et al. 2016) "
            "probability applied to the final backbone block's residual "
            "branches. Each TransformerBlock has independent DropPath on "
            "the attention and MLP branches with a linear schedule from "
            "0.0 at block 0 to drop_path_max at block (depth-1). Identity "
            "at eval so adds zero inference cost. 0.0 disables (default)."
        ),
        "mirror_augmentation": (
            "H148/H183/H185: per-sample y=0 yaw mirror augmentation with "
            "p=0.5. Negates surface_x[..., 1] (y) and surface_x[..., 4] "
            "(ny), surface_y[..., 2] (tau_y), and volume_x[..., 1] (y); "
            "cp, tau_x, tau_z, volume_pressure, and sdf are invariant. "
            "Applied to each sample independently after batch.to(device)."
        ),
        "resume_from_wandb": (
            "H244: optional W&B run_id whose 'model' artifact will be "
            "downloaded and loaded into base_model before training. Used "
            "together with --resume-alias (e.g. 'epoch-13') and "
            "--epochs-already-done to continue an existing cosine schedule."
        ),
        "resume_alias": (
            "H244: artifact alias of the W&B checkpoint to resume from "
            "(e.g. 'epoch-13', 'best'). Required with --resume-from-wandb."
        ),
        "epochs_already_done": (
            "H244: number of epochs the resumed checkpoint had completed. "
            "Advances the LR scheduler by this many steps so the resumed "
            "training continues the existing cosine schedule, and the main "
            "epoch loop starts at this index instead of zero."
        ),
        "save_every_epoch": (
            "H244: save EMA checkpoint as 'checkpoint_ep{N}.pt' alongside "
            "the best-only checkpoint after every validation event. Used "
            "to keep EP14/15/16 EMA checkpoints for downstream TTA eval."
        ),
        "physics_prior_normals": (
            "H347: enable the tangent-plane physics prior on wall shear "
            "stress. Adds an auxiliary loss term that penalises the cosine "
            "of the angle between the predicted WSS vector (denormalised) "
            "and the surface normal at every valid surface point. By "
            "definition WSS lies in the tangent plane (tau . n = 0); this "
            "regularises the model toward physically valid predictions in "
            "geometric-complexity regions (A-pillar, wheel arch, etc.) "
            "where the WSS_z residual is concentrated."
        ),
        "physics_prior_smooth": (
            "H347: enable the graph-Laplacian smoothness physics prior on "
            "tau_z. For every batch a chunked on-the-fly kNN graph is "
            "computed in 3D coordinate space (random anchor subsample for "
            "memory) and tau_z differences along edges are penalised. "
            "Encourages attached-boundary-layer continuity along "
            "streamlines without explicit edge metadata."
        ),
        "lambda_normals": (
            "H347: scalar weight for the tangent-plane loss. The loss is "
            "the mean squared cosine of the angle between predicted-WSS "
            "and surface normal so its magnitude is O(1); 1e-3 keeps the "
            "auxiliary term well below the primary MSE during cosine-tail "
            "fine-tuning. Default 1e-3."
        ),
        "lambda_smooth": (
            "H347: scalar weight for the kNN smoothness loss on tau_z. The "
            "loss is mean squared neighbour difference on the model output "
            "(normalised space) so its magnitude is O(1); default 1e-3 "
            "mirrors lambda_normals."
        ),
        "physics_knn_k": (
            "H347: number of nearest neighbours used by the smoothness "
            "prior. Default 6 matches local mesh connectivity for "
            "DrivAerML's near-uniform surface sampling."
        ),
        "physics_smooth_anchors": (
            "H347: number of random anchor points per batch sample used "
            "for the smoothness loss. Computing kNN over all N=65k surface "
            "points per step is too memory-heavy; sampling 8k anchors gives "
            "dense spatial coverage across many steps while keeping the "
            "physics-prior overhead under ~10 percent wall-time."
        ),
        "physics_knn_chunk_size": (
            "H347: chunk size for the pairwise distance computation in the "
            "smoothness loss (anchors are processed in chunks of this size "
            "against the full sampled point cloud). Trade-off between "
            "peak memory and kernel-launch overhead; default 1024 keeps "
            "the per-chunk distance matrix around 256MB at N=65k."
        ),
    }
    for field in fields(Config):
        value = getattr(defaults, field.name)
        arg_name = f"--{field.name.replace('_', '-')}"
        help_value = help_text.get(field.name)
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value, help=help_value)
            parser.add_argument(f"--no-{field.name.replace('_', '-')}", action="store_false", dest=field.name)
        else:
            parser.add_argument(arg_name, type=type(value), default=value, help=help_value)
    namespace = parser.parse_args(argv)
    return Config(**vars(namespace))


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "lion":
        from lion_pytorch import Lion

        return Lion(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.lion_beta1, config.lion_beta2),
            use_triton=False,
        )
    raise ValueError(f"Unknown optimizer '{config.optimizer}'. Supported: adamw, lion.")


def parse_rff_init_sigmas(spec: str) -> list[float] | None:
    spec = (spec or "").strip()
    if not spec:
        return None
    sigmas = [float(token.strip()) for token in spec.split(",") if token.strip()]
    if not sigmas:
        return None
    if any(s <= 0.0 for s in sigmas):
        raise ValueError(f"--rff-init-sigmas must be positive: {spec!r}")
    return sigmas


def collect_string_sep_metrics(model: nn.Module) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for attr in ("surface_string_sep", "volume_string_sep"):
        module = getattr(model, attr, None)
        if module is None:
            continue
        log_freq = module.log_freq.detach().float()
        prefix = f"string_sep/{attr}"
        metrics[f"{prefix}/log_freq_mean"] = float(log_freq.mean().item())
        metrics[f"{prefix}/log_freq_std"] = float(log_freq.std().item())
        metrics[f"{prefix}/log_freq_min"] = float(log_freq.min().item())
        metrics[f"{prefix}/log_freq_max"] = float(log_freq.max().item())
        for axis in range(log_freq.shape[0]):
            metrics[f"{prefix}/freq_axis_{axis}_mean"] = float(log_freq[axis].mean().item())
            metrics[f"{prefix}/freq_axis_{axis}_std"] = float(log_freq[axis].std().item())
    return metrics


def download_resume_checkpoint(
    entity: str,
    project: str,
    run_id: str,
    alias: str,
    cache_root: Path,
) -> Path:
    """Download the W&B model artifact for run_id with the given alias.

    Returns the local path to checkpoint.pt. Mirrors the convention from
    ``eval_multi_res.py``: artifacts of type 'model' with the alias.
    """

    cache_dir = cache_root / run_id / alias
    ckpt_path = cache_dir / "checkpoint.pt"
    if ckpt_path.exists():
        return ckpt_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    matched = None
    for art in run.logged_artifacts():
        if art.type != "model":
            continue
        if alias in art.aliases:
            matched = art
            break
    if matched is None:
        avail = []
        for art in run.logged_artifacts():
            if art.type == "model":
                avail.append((art.name, list(art.aliases)))
        raise RuntimeError(
            f"No model artifact with alias '{alias}' for run {run_id}; "
            f"available={avail}"
        )
    matched.download(root=str(cache_dir))
    return ckpt_path


def mirror_augment_batch(batch, p: float = 0.5):
    """Apply per-sample y=0 yaw mirror with probability ``p`` to a SurfaceBatch.

    Mirrors negate y-axis components: surface_x[..., 1] (y) and
    surface_x[..., 4] (ny); surface_y[..., 2] (tau_y); volume_x[..., 1] (y).
    cp, tau_x, tau_z, sdf, and volume_pressure are invariant under this
    bilateral symmetry of DrivAerML geometry.

    The decision is per-sample and independent across ranks, which matches
    the loader-level implementation on askeladd/h148-mirror-augmentation
    (used by H171/H183/H185).
    """

    batch_size = batch.surface_x.shape[0]
    flips = (torch.rand(batch_size, device=batch.surface_x.device) < p).float()
    if flips.sum().item() == 0:
        return batch
    surface_x = batch.surface_x.clone()
    surface_y = batch.surface_y.clone()
    volume_x = batch.volume_x.clone()
    sign = (1.0 - 2.0 * flips).view(batch_size, 1).to(surface_x.dtype)
    surface_x[..., 1] = surface_x[..., 1] * sign
    surface_x[..., 4] = surface_x[..., 4] * sign
    surface_y[..., 2] = surface_y[..., 2] * sign
    volume_x[..., 1] = volume_x[..., 1] * sign
    return type(batch)(
        case_ids=list(batch.case_ids),
        surface_x=surface_x,
        surface_y=surface_y,
        surface_mask=batch.surface_mask,
        volume_x=volume_x,
        volume_y=batch.volume_y,
        volume_mask=batch.volume_mask,
        metadata=list(batch.metadata),
    )


def build_model(config: Config) -> SurfaceTransolver:
    return SurfaceTransolver(
        n_layers=config.model_layers,
        n_hidden=config.model_hidden_dim,
        dropout=config.model_dropout,
        n_head=config.model_heads,
        mlp_ratio=config.model_mlp_ratio,
        slice_num=config.model_slices,
        rff_num_features=config.rff_num_features,
        rff_sigma=config.rff_sigma,
        rff_init_sigmas=parse_rff_init_sigmas(config.rff_init_sigmas),
        pos_encoding_mode=config.pos_encoding_mode,
        use_qk_norm=config.use_qk_norm,
        use_surf_to_vol_xattn=config.use_surf_to_vol_xattn,
        drop_path_max=config.drop_path_max,
    )


def _physics_prior_loss(
    surface_pred_norm: torch.Tensor,
    batch,
    transform: TargetTransform,
    *,
    enable_normals: bool,
    enable_smooth: bool,
    lambda_normals: float,
    lambda_smooth: float,
    knn_k: int,
    smooth_anchors: int,
    knn_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """H347 boundary-layer physics priors on wall shear stress.

    Two auxiliary losses, both optional:

      (1) Tangent-plane: WSS must lie in the tangent plane of the surface,
          i.e. ``tau . n = 0`` after denormalisation. Enforced as the mean
          squared cosine of the angle between predicted-WSS and the
          per-point surface normal (numerically safe at small |tau|).
      (2) Graph-Laplacian smoothness on tau_z: penalise differences of
          ``tau_z`` along edges of a kNN graph built on-the-fly in 3D
          coordinate space. Uses a random anchor subsample per batch for
          memory; ``knn_chunk_size`` controls the peak distance-matrix
          size. The graph indices are detached so gradients flow only
          through the predicted ``tau_z`` values.

    Returns the (already weighted) auxiliary loss and a metrics dict with
    raw and weighted components and physical-violation magnitudes for
    monitoring.
    """

    metrics: dict[str, float] = {}
    device = surface_pred_norm.device
    L_aux = surface_pred_norm.new_zeros(())

    surface_mask = batch.surface_mask
    if surface_mask.dtype != torch.bool:
        mask_bool = surface_mask.bool()
    else:
        mask_bool = surface_mask
    mask_f = mask_bool.to(surface_pred_norm.dtype)
    n_valid = mask_f.sum().clamp(min=1.0)

    pred_denorm = transform.invert_surface(surface_pred_norm)
    wss_pred = pred_denorm[..., 1:4]

    if enable_normals:
        normals = batch.surface_x[..., 3:6]
        wss_norm = wss_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        wss_dir = wss_pred / wss_norm
        n_norm = normals.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        n_dir = normals / n_norm
        dot = (wss_dir * n_dir).sum(-1)
        dot_sq_masked = (dot * dot) * mask_f
        L_normals = dot_sq_masked.sum() / n_valid
        L_aux = L_aux + lambda_normals * L_normals
        dot_abs_mean = ((dot.detach().abs()) * mask_f).sum() / n_valid
        metrics["physics/normals_loss"] = float(L_normals.detach().item())
        metrics["physics/normals_loss_weighted"] = float(
            (lambda_normals * L_normals).detach().item()
        )
        metrics["physics/normals_violation_cos"] = float(dot_abs_mean.item())

    if enable_smooth:
        coords = batch.surface_x[..., 0:3].detach()
        tau_z_norm = surface_pred_norm[..., 3]
        B, _, _ = coords.shape
        smooth_terms: list[torch.Tensor] = []
        for b in range(B):
            valid_idx = mask_bool[b].nonzero(as_tuple=False).squeeze(-1)
            n_valid_b = valid_idx.numel()
            if n_valid_b <= knn_k + 1:
                continue
            n_anchor = min(int(smooth_anchors), n_valid_b)
            perm = torch.randperm(n_valid_b, device=device)[:n_anchor]
            anchor_idx = valid_idx[perm]
            anchor_coords = coords[b].index_select(0, anchor_idx)
            full_coords = coords[b].index_select(0, valid_idx)
            full_tau_z = tau_z_norm[b].index_select(0, valid_idx)
            anchor_tau_z = tau_z_norm[b].index_select(0, anchor_idx)
            chunk = max(1, int(knn_chunk_size))
            neighbor_chunks: list[torch.Tensor] = []
            for start in range(0, n_anchor, chunk):
                end = min(start + chunk, n_anchor)
                q = anchor_coords[start:end]
                with torch.no_grad():
                    dists = torch.cdist(q, full_coords)
                    _, neigh = dists.topk(knn_k + 1, dim=-1, largest=False)
                    neigh = neigh[..., 1:]
                neighbor_chunks.append(neigh)
            knn_idx = torch.cat(neighbor_chunks, dim=0)
            tau_z_neighbors = full_tau_z[knn_idx]
            diff = anchor_tau_z.unsqueeze(-1) - tau_z_neighbors
            smooth_terms.append((diff * diff).mean())
        if smooth_terms:
            L_smooth = torch.stack(smooth_terms).mean()
            L_aux = L_aux + lambda_smooth * L_smooth
            metrics["physics/smooth_loss"] = float(L_smooth.detach().item())
            metrics["physics/smooth_loss_weighted"] = float(
                (lambda_smooth * L_smooth).detach().item()
            )
            metrics["physics/smooth_violation_rms"] = float(
                L_smooth.detach().sqrt().item()
            )
        else:
            metrics["physics/smooth_loss"] = 0.0
            metrics["physics/smooth_loss_weighted"] = 0.0
            metrics["physics/smooth_violation_rms"] = 0.0

    return L_aux, metrics


def train_loss(
    model: nn.Module,
    batch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
    *,
    surface_loss_weight: float = 1.0,
    volume_loss_weight: float = 1.0,
    surface_channel_weights: torch.Tensor | None = None,
    physics_prior_normals: bool = False,
    physics_prior_smooth: bool = False,
    lambda_normals: float = 0.0,
    lambda_smooth: float = 0.0,
    physics_knn_k: int = 6,
    physics_smooth_anchors: int = 8192,
    physics_knn_chunk_size: int = 1024,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = batch.to(device)
    surface_target = transform.apply_surface(batch.surface_y)
    volume_target = transform.apply_volume(batch.volume_y)
    with autocast_context(device, amp_mode):
        out = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
        surface_pred = out["surface_preds"]
        if surface_channel_weights is not None:
            surface_loss = weighted_channel_mse(
                surface_pred,
                surface_target,
                batch.surface_mask,
                surface_channel_weights,
            )
        else:
            surface_loss = masked_mse(surface_pred, surface_target, batch.surface_mask)
        volume_loss = masked_mse(out["volume_preds"], volume_target, batch.volume_mask)
        weighted_surface_loss = surface_loss_weight * surface_loss
        weighted_volume_loss = volume_loss_weight * volume_loss
        loss = weighted_surface_loss + weighted_volume_loss
        base_mse_loss = surface_loss + volume_loss
    metrics = {
        "base_mse_loss": float(base_mse_loss.detach().cpu().item()),
        "surface_loss": float(surface_loss.detach().cpu().item()),
        "volume_loss": float(volume_loss.detach().cpu().item()),
        "surface_loss_weighted": float(weighted_surface_loss.detach().cpu().item()),
        "volume_loss_weighted": float(weighted_volume_loss.detach().cpu().item()),
    }
    if physics_prior_normals or physics_prior_smooth:
        L_phys, phys_metrics = _physics_prior_loss(
            surface_pred,
            batch,
            transform,
            enable_normals=physics_prior_normals,
            enable_smooth=physics_prior_smooth,
            lambda_normals=lambda_normals,
            lambda_smooth=lambda_smooth,
            knn_k=physics_knn_k,
            smooth_anchors=physics_smooth_anchors,
            knn_chunk_size=physics_knn_chunk_size,
        )
        loss = loss + L_phys
        metrics["physics/total_loss_weighted"] = float(L_phys.detach().cpu().item())
        metrics.update(phys_metrics)
    return loss, metrics


GRADNORM_TASK_NAMES = ("sp", "tau_x", "tau_y", "tau_z", "vp")


def per_task_train_losses(
    model: nn.Module,
    batch,
    transform: TargetTransform,
    device: torch.device,
    amp_mode: str,
) -> list[torch.Tensor]:
    """Compute the 5 per-axis task losses used by GradNorm.

    Returns scalar tensors in order: [sp, tau_x, tau_y, tau_z, vp].
    All five share the same forward graph so the GradNorm balancer can
    extract per-task gradient norms via ``torch.autograd.grad`` with
    ``retain_graph=True``.
    """

    batch = batch.to(device)
    surface_target = transform.apply_surface(batch.surface_y)
    volume_target = transform.apply_volume(batch.volume_y)
    with autocast_context(device, amp_mode):
        out = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
        surface_pred = out["surface_preds"]
        sp_loss = masked_mse(surface_pred[..., 0:1], surface_target[..., 0:1], batch.surface_mask)
        taux_loss = masked_mse(surface_pred[..., 1:2], surface_target[..., 1:2], batch.surface_mask)
        tauy_loss = masked_mse(surface_pred[..., 2:3], surface_target[..., 2:3], batch.surface_mask)
        tauz_loss = masked_mse(surface_pred[..., 3:4], surface_target[..., 3:4], batch.surface_mask)
        vp_loss = masked_mse(out["volume_preds"], volume_target, batch.volume_mask)
    return [sp_loss, taux_loss, tauy_loss, tauz_loss, vp_loss]


GRADNORM_MODES = ("full", "ema_proxy")


class GradNormBalancer:
    """GradNorm dynamic multi-task loss balancing (Chen et al., NeurIPS 2018).

    Two operating modes share the same per-task log-weight state and the
    same warmup mechanism for L_i(0); they differ in how the weights are
    updated each step.

    mode='full' (paper-faithful):
        Maintains a learnable log-weight per task and updates it via Adam
        on the GradNorm L1 loss using gradient norms measured at the last
        shared encoder block. Cost: 5x backward (one autograd.grad per
        task with retain_graph=True) on top of the main backward.

        Notes vs the paper:
          * ``log_weights`` parameterisation; ``task_weights = exp(log_w) *
            T / sum(exp(log_w))`` keeps the mean-1 invariant by construction.
          * Avoids ``create_graph=True``: since ``||w * g||_2 = w * ||g||_2``
            for ``w >= 0``, the GradNorm loss is computed as
            ``W * grad_norm_detached`` and is differentiable in
            ``log_weights`` directly. Equivalent and ~2x cheaper than the
            second-order graph variant.

    mode='ema_proxy' (lightweight loss-magnitude approximation):
        Maintains an exponential moving average of per-task losses and
        sets weights in closed form each step:

            ema_i      = beta * ema_i + (1 - beta) * L_i.detach()
            r_i        = ema_i / initial_loss_i              # relative rate
            r_norm_i   = r_i / mean(r)                       # mean-1 rate
            w_raw_i    = max(r_norm_i ** alpha, min_weight)  # asym floor
            w_i        = w_raw_i * T / sum(w_raw_j)          # sum=T => mean=1

        No autograd over the balancer => ~1x overhead. Tasks whose losses
        decrease slower than average (high r_norm_i) get up-weighted; the
        alpha exponent controls how aggressively. ``log_weights.data`` is
        set from log(w_i) each step for unified logging and DDP broadcast.

        ``min_weight`` (default 0.0) is an asymmetric pre-normalisation
        floor applied to ``w_raw_i``: it caps how aggressively a fast
        task can be down-weighted while leaving the up-weighting ceiling
        free. Recommended 0.7 for soft redistribution.

        Note on the advisor spec:
            The original spec wrote ``w_i = (1/ema_i) / sum(1/ema_j) * T``
            but described the intent as "tasks with larger current loss
            get more weight" (which matches the GradNorm goal of
            up-weighting slow-converging tasks like tau_y/tau_z). The
            literal inverse-loss formula has the opposite sign (it
            DOWN-weights high-loss tasks). We implement the intent: weights
            proportional to r_i**alpha, which is the GradNorm relative
            training rate dressed up as a closed-form computation.

    DDP synchronisation (both modes): per-task losses (and per-task
    gradient norms in 'full' mode) are all-reduced (AVG) so every rank
    runs the same balancer step; ``log_weights`` are broadcast from rank 0
    each step as a belt-and-suspenders guard against numerical drift.

    L_i(0) is captured as the running average of the synced losses
    (mode='full') or the EMA snapshot at warmup completion
    (mode='ema_proxy') over the first ``init_warmup_steps`` steps.
    """

    def __init__(
        self,
        num_tasks: int,
        *,
        mode: str,
        alpha: float,
        lr: float,
        device: torch.device,
        init_warmup_steps: int,
        log_clip: float,
        ema_beta: float,
        min_weight: float = 0.0,
    ):
        if mode not in GRADNORM_MODES:
            raise ValueError(
                f"Unknown gradnorm_mode '{mode}'. Supported: {GRADNORM_MODES}."
            )
        self.mode = mode
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device
        self.init_warmup_steps = max(1, init_warmup_steps)
        self.log_clip = float(log_clip)
        self.ema_beta = float(ema_beta)
        self.min_weight = float(min_weight)
        self.log_weights = nn.Parameter(torch.zeros(num_tasks, device=device))
        if mode == "full":
            self.optimizer: torch.optim.Optimizer | None = torch.optim.Adam(
                [self.log_weights], lr=lr
            )
        else:
            self.optimizer = None
        self._init_loss_sum = torch.zeros(num_tasks, device=device)
        self._init_loss_count = 0
        self.initial_loss: torch.Tensor | None = None
        self.ema_loss: torch.Tensor | None = None
        self.last_floor_active: int = 0

    @property
    def ready(self) -> bool:
        return self.initial_loss is not None

    def task_weights(self) -> torch.Tensor:
        w = self.log_weights.exp()
        return w * (self.num_tasks / w.sum().clamp(min=1e-12))

    def task_weights_detached(self) -> torch.Tensor:
        return self.task_weights().detach()

    def update_initial_loss(self, per_task_losses_synced: torch.Tensor) -> None:
        if self.initial_loss is not None:
            return
        L = per_task_losses_synced.detach().to(self.device)
        if self.mode == "ema_proxy":
            if self.ema_loss is None:
                self.ema_loss = L.clone()
            else:
                self.ema_loss = self.ema_beta * self.ema_loss + (1.0 - self.ema_beta) * L
        self._init_loss_sum = self._init_loss_sum + L
        self._init_loss_count += 1
        if self._init_loss_count >= self.init_warmup_steps:
            if self.mode == "ema_proxy" and self.ema_loss is not None:
                self.initial_loss = self.ema_loss.detach().clamp(min=1e-12).clone()
            else:
                self.initial_loss = (
                    self._init_loss_sum / self._init_loss_count
                ).clamp(min=1e-12)

    def step(
        self,
        per_task_losses_synced: torch.Tensor,
        per_task_grad_norms_synced: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Run one GradNorm update on ``log_weights``.

        Inputs are expected to already be cross-rank synced (mean over
        ranks). In mode='ema_proxy' ``per_task_grad_norms_synced`` is
        ignored. Returns a scalar diagnostic tensor (the L1 GradNorm loss
        in 'full' mode, the per-task r_i sum-deviation in 'ema_proxy'
        mode), or ``None`` if the initial-loss warmup has not finished
        yet.
        """

        if self.initial_loss is None:
            return None
        L = per_task_losses_synced.detach().to(self.device)
        if self.mode == "ema_proxy":
            assert self.ema_loss is not None
            self.ema_loss = (
                self.ema_beta * self.ema_loss + (1.0 - self.ema_beta) * L
            )
            r = (self.ema_loss / self.initial_loss).clamp(min=1e-12)
            r = r / r.mean().clamp(min=1e-12)
            w_raw_pre = r.pow(self.alpha).clamp(min=1e-12)
            if self.min_weight > 0.0:
                self.last_floor_active = int(
                    (w_raw_pre < self.min_weight).sum().item()
                )
                w_raw = w_raw_pre.clamp(min=self.min_weight)
            else:
                self.last_floor_active = 0
                w_raw = w_raw_pre
            w = w_raw * (self.num_tasks / w_raw.sum().clamp(min=1e-12))
            with torch.no_grad():
                self.log_weights.data = w.log().clamp(
                    -self.log_clip, self.log_clip
                )
            return (r - 1.0).abs().sum().detach()

        if per_task_grad_norms_synced is None:
            raise ValueError(
                "GradNormBalancer.step: gradient norms are required in mode='full'."
            )
        G_detached = per_task_grad_norms_synced.detach().to(self.device)
        r = (L / self.initial_loss).clamp(min=1e-12)
        r = r / r.mean().clamp(min=1e-12)
        target = (G_detached.mean() * (r ** self.alpha)).detach()

        W = self.task_weights()
        G = W * G_detached
        gradnorm_loss = (G - target).abs().sum()

        self.optimizer.zero_grad()
        gradnorm_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.log_weights.clamp_(-self.log_clip, self.log_clip)
        return gradnorm_loss.detach()


def synced_per_task_tensor(
    values: list[torch.Tensor] | torch.Tensor,
    *,
    state,
    device: torch.device,
) -> torch.Tensor:
    """Stack per-task scalar tensors and all-reduce(AVG) across DDP ranks."""

    if isinstance(values, torch.Tensor):
        stacked = values.to(device=device, dtype=torch.float32)
    else:
        stacked = torch.stack([v.to(device=device, dtype=torch.float32) for v in values])
    if state.enabled:
        dist.all_reduce(stacked, op=dist.ReduceOp.AVG)
    return stacked


def parse_vol_points_schedule(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    parts = [p.strip() for p in text.split(":") if p.strip()]
    if len(parts) % 2 != 0 or len(parts) == 0:
        raise ValueError(
            f"--vol-points-schedule must be 'EPOCH:POINTS' pairs joined by ':'; got '{text}'"
        )
    schedule: list[tuple[int, int]] = []
    for i in range(0, len(parts), 2):
        try:
            epoch = int(parts[i])
            points = int(parts[i + 1])
        except ValueError as exc:
            raise ValueError(
                f"--vol-points-schedule values must be integers; got '{text}'"
            ) from exc
        if epoch < 0 or points <= 0:
            raise ValueError(
                f"--vol-points-schedule entries must satisfy epoch>=0 and points>0; "
                f"got {epoch}:{points}"
            )
        schedule.append((epoch, points))
    schedule.sort(key=lambda x: x[0])
    seen: set[int] = set()
    for start_epoch, _ in schedule:
        if start_epoch in seen:
            raise ValueError(
                f"--vol-points-schedule has duplicate epoch breakpoint {start_epoch}"
            )
        seen.add(start_epoch)
    if schedule[0][0] != 0:
        raise ValueError(
            f"--vol-points-schedule must start at epoch 0; got first entry {schedule[0]}"
        )
    return schedule


def vol_points_for_epoch(
    schedule: list[tuple[int, int]],
    epoch: int,
    fallback: int,
) -> int:
    if not schedule:
        return fallback
    current = schedule[0][1]
    for start_epoch, points in schedule:
        if epoch >= start_epoch:
            current = points
        else:
            break
    return current


def rebuild_train_loader_with_vol_points(
    config: Config,
    old_train_loader: DataLoader,
    n_points: int,
    distributed_state,
) -> DataLoader:
    """Rebuild the training DataLoader with a new max_volume_points value.

    Reuses the existing ``DrivAerMLCaseStore`` so cached point counts and
    artifact-path resolutions survive the swap. The view list is recomputed
    because ``max_volume_points`` changes the per-case view count, which in
    turn changes the dataset length that the distributed sampler must see.
    """

    old_ds = old_train_loader.dataset
    sampling_mode = (
        "train_random" if (config.train_surface_points > 0 or n_points > 0) else "full"
    )
    train_ds = DrivAerMLSurfaceDataset(
        old_ds.case_ids,
        store=old_ds.store,
        max_surface_points=config.train_surface_points,
        max_volume_points=n_points,
        sampling_mode=sampling_mode,
    )
    train_sampler = None
    train_shuffle = True
    if distributed_state is not None and distributed_state.enabled:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=distributed_state.world_size,
            rank=distributed_state.rank,
            shuffle=True,
            drop_last=True,
        )
        train_shuffle = False
    return DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs(config),
    )


def main(argv: Iterable[str] | None = None) -> None:
    state = init_distributed()
    run = None
    try:
        config = parse_args(argv)
        kill_thresholds = parse_kill_thresholds(config.kill_thresholds)
        vol_points_schedule = parse_vol_points_schedule(config.vol_points_schedule)
        requested_epochs = config.epochs
        if os.environ.get("SENPAI_MAX_EPOCHS"):
            requested_epochs = min(requested_epochs, int(os.environ["SENPAI_MAX_EPOCHS"]))
        max_epochs = min(requested_epochs, 3) if config.debug else requested_epochs
        timeout_minutes, val_budget_minutes, train_timeout_minutes = timeout_budget_minutes()
        device = state.device
        if state.is_main:
            ddp_suffix = f", DDP world_size={state.world_size}" if state.enabled else ""
            print(f"Device: {device}{ddp_suffix}" + (" [DEBUG]" if config.debug else ""))

        if vol_points_schedule:
            initial_vol_points = vol_points_for_epoch(
                vol_points_schedule, 0, config.train_volume_points
            )
            if config.debug:
                initial_vol_points = min(initial_vol_points, 8_192)
            config.train_volume_points = initial_vol_points
            if state.is_main:
                print(
                    "Volume-points curriculum: "
                    + ", ".join(f"ep{e}->{p}" for e, p in vol_points_schedule)
                )
        current_train_vol_points = config.train_volume_points

        train_loader, val_loaders, test_loaders, stats = make_loaders(config, distributed_state=state)
        final_val_loaders = full_eval_loaders_from(val_loaders, config) if state.is_main else {}
        final_test_loaders = full_eval_loaders_from(test_loaders, config) if state.is_main else {}
        transform = TargetTransform(
            surface_y_mean=stats["surface_y_mean"].to(device),
            surface_y_std=stats["surface_y_std"].to(device),
            volume_y_mean=stats["volume_y_mean"].to(device),
            volume_y_std=stats["volume_y_std"].to(device),
        )

        model: nn.Module = build_model(config).to(device)
        n_params = sum(param.numel() for param in model.parameters())

        # H244: resume from W&B artifact before torch.compile and DDP wrap so
        # the load_state_dict() call sees the plain nn.Module. The checkpoint
        # saved by trainer_runtime is the EMA shadow (best_checkpoint_source
        # = "ema"), so loading it here seeds the model with the EP13 EMA.
        resume_epoch_info: dict[str, object] = {}
        if config.resume_from_wandb:
            if not config.resume_alias:
                raise ValueError("--resume-from-wandb requires --resume-alias")
            cache_root = Path(config.output_dir) / "resume_cache"
            if state.is_main:
                ckpt_path = download_resume_checkpoint(
                    entity=os.environ.get(
                        "WANDB_ENTITY", "wandb-applied-ai-team"
                    ),
                    project=os.environ.get(
                        "WANDB_PROJECT", "senpai-v1-drivaerml-ddp8"
                    ),
                    run_id=config.resume_from_wandb,
                    alias=config.resume_alias,
                    cache_root=cache_root,
                )
                ckpt_path_str = str(ckpt_path)
            else:
                ckpt_path_str = ""
            if state.enabled:
                obj = [ckpt_path_str]
                dist.broadcast_object_list(obj, src=0)
                ckpt_path_str = obj[0]
            ck = torch.load(ckpt_path_str, map_location="cpu", weights_only=False)
            state_dict = ck["model"]
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            resume_epoch_info = {
                "resume_epoch": ck.get("epoch"),
                "resume_checkpoint_source": ck.get("checkpoint_source"),
                "resume_run_id": config.resume_from_wandb,
                "resume_alias": config.resume_alias,
                "resume_missing": len(missing),
                "resume_unexpected": len(unexpected),
            }
            if state.is_main:
                print(
                    f"Resumed from W&B {config.resume_from_wandb}:{config.resume_alias} "
                    f"epoch={ck.get('epoch')} source={ck.get('checkpoint_source')} "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )

        # Surface targets are [cp, tau_x, tau_y, tau_z] — channels 2 and 3 carry
        # the largest gap to AB-UPT, so we expose per-channel weights here.
        surface_channel_weights = torch.tensor(
            [1.0, 1.0, config.tau_y_loss_weight, config.tau_z_loss_weight],
            device=device,
            dtype=torch.float32,
        )
        use_channel_weights = bool(
            (config.tau_y_loss_weight != 1.0) or (config.tau_z_loss_weight != 1.0)
        )
        if config.compile_model:
            model = torch.compile(model)
        if state.enabled:
            ddp_kwargs = {}
            if device.type == "cuda":
                ddp_kwargs = {"device_ids": [state.local_rank], "output_device": state.local_rank}
            # The surface->volume cross-attention sublayer (PR #823) gates on
            # `surface_tokens > 0`, but DrivAerML's loader returns
            # max(surface_views, volume_views) views per case, so views with
            # view_index >= surface_views have N_S=0 (and the gate skips the
            # layer). Different ranks hit different sub-graphs in a single
            # batch, so DDP's default find_unused_parameters=False deadlocks
            # at the gradient bucket allreduce. Enabling find_unused_parameters
            # whenever the cross-attention is on lets DDP synchronize unused
            # params across ranks, at a small per-step overhead.
            if config.use_surf_to_vol_xattn:
                ddp_kwargs["find_unused_parameters"] = True
            model = DistributedDataParallel(model, **ddp_kwargs)
        base_model = unwrap_model(model)
        if state.is_main:
            print(f"Model: SurfaceTransolver grouped surface+volume ({n_params / 1e6:.2f}M params)")

        optimizer = build_optimizer(base_model, config)
        scheduler = build_lr_scheduler(optimizer, config, max_epochs)
        # H244: when resuming, advance the cosine scheduler so the LR matches
        # the position in the original schedule. Optimizer state is fresh; the
        # cosine LR is determined entirely by ``last_epoch``.
        if config.epochs_already_done > 0:
            for _ in range(config.epochs_already_done):
                scheduler.step()
            if state.is_main:
                print(
                    f"Advanced LR scheduler by epochs_already_done="
                    f"{config.epochs_already_done}; current_lr="
                    f"{scheduler.get_last_lr()[0]:.3e}"
                )
        ema = EMA(base_model, decay=config.ema_decay, start_step=config.ema_start_step) if config.use_ema else None

        balancer: GradNormBalancer | None = None
        gradnorm_shared_params: list[nn.Parameter] = []
        if config.use_gradnorm:
            if config.gradnorm_mode == "full" and config.compile_model:
                raise ValueError(
                    "--use-gradnorm --gradnorm-mode full requires --no-compile-model "
                    "(autograd.grad with retain_graph is not supported through "
                    "torch.compile)."
                )
            balancer = GradNormBalancer(
                num_tasks=len(GRADNORM_TASK_NAMES),
                mode=config.gradnorm_mode,
                alpha=config.gradnorm_alpha,
                lr=config.gradnorm_lr,
                device=device,
                init_warmup_steps=config.gradnorm_init_warmup_steps,
                log_clip=config.gradnorm_log_clip,
                ema_beta=config.gradnorm_ema_beta,
                min_weight=config.gradnorm_min_weight,
            )
            if config.gradnorm_mode == "full":
                gradnorm_shared_params = [
                    p for p in base_model.backbone.blocks[-1].parameters() if p.requires_grad
                ]
                if not gradnorm_shared_params:
                    raise RuntimeError(
                        "GradNorm: no trainable parameters in the last shared encoder block."
                    )
            if state.is_main:
                if config.gradnorm_mode == "full":
                    shared_param_count = sum(p.numel() for p in gradnorm_shared_params)
                    print(
                        f"GradNorm enabled (mode=full): {len(GRADNORM_TASK_NAMES)} tasks "
                        f"({', '.join(GRADNORM_TASK_NAMES)}); alpha={config.gradnorm_alpha}; "
                        f"lr={config.gradnorm_lr}; warmup={config.gradnorm_init_warmup_steps} "
                        f"steps; log_clip={config.gradnorm_log_clip}; "
                        f"shared_params={shared_param_count:,d} (last backbone block)"
                    )
                else:
                    print(
                        f"GradNorm enabled (mode=ema_proxy): {len(GRADNORM_TASK_NAMES)} tasks "
                        f"({', '.join(GRADNORM_TASK_NAMES)}); alpha={config.gradnorm_alpha}; "
                        f"ema_beta={config.gradnorm_ema_beta}; "
                        f"warmup={config.gradnorm_init_warmup_steps} steps; "
                        f"log_clip={config.gradnorm_log_clip}; "
                        f"min_weight={config.gradnorm_min_weight}; "
                        f"closed-form weights from r_i = ema_loss_i / initial_loss_i"
                    )
        total_estimated_steps = max(1, max_epochs * max(len(train_loader), 1))
        if kill_thresholds and state.is_main:
            print("Kill thresholds:", "; ".join(threshold.describe() for threshold in kill_thresholds))
        train_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)
        val_slope_tracker = MetricSlopeTracker(total_estimated_steps, config.slope_log_fraction)

        # Log multi-sigma RFF init layout for verification.
        if state.is_main:
            ss = getattr(base_model, "surface_string_sep", None)
            if ss is not None and getattr(ss, "init_sigmas", None) is not None:
                print(
                    f"StringSep multi-sigma init: sigmas={ss.init_sigmas} "
                    f"num_features={ss.num_features} layout=round-robin per axis "
                    f"(log_freq[d, f] = log(sigmas[f % len(sigmas)]))"
                )
            if use_channel_weights:
                print(
                    f"Surface channel weights: cp=1.0 tau_x=1.0 "
                    f"tau_y={config.tau_y_loss_weight} tau_z={config.tau_z_loss_weight}"
                )

        run = init_wandb_run(
            config=config,
            state=state,
            n_params=n_params,
            train_loader=train_loader,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            total_estimated_steps=total_estimated_steps,
            max_epochs=max_epochs,
            train_timeout_minutes=train_timeout_minutes,
            val_budget_minutes=val_budget_minutes,
        )

        output_dir = Path(config.output_dir) / f"run-{run.id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "checkpoint.pt"
        config_path = output_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(asdict(config), f)

        if state.is_main:
            ss = getattr(base_model, "surface_string_sep", None)
            init_sigmas_for_log = (
                list(ss.init_sigmas)
                if ss is not None and getattr(ss, "init_sigmas", None) is not None
                else []
            )
            wandb.summary["model/string_sep_init_sigmas"] = init_sigmas_for_log
            wandb.summary["loss/tau_y_loss_weight"] = config.tau_y_loss_weight
            wandb.summary["loss/tau_z_loss_weight"] = config.tau_z_loss_weight
            backbone = getattr(base_model, "backbone", None)
            if backbone is not None and hasattr(backbone, "blocks"):
                drop_path_schedule = [
                    float(blk.drop_path_attn.drop_prob) for blk in backbone.blocks
                ]
                wandb.summary["drop_path/max"] = config.drop_path_max
                wandb.summary["drop_path/schedule"] = drop_path_schedule
                if config.drop_path_max > 0.0:
                    print(
                        f"DropPath (H112): per-block schedule "
                        f"(attn=mlp) = {drop_path_schedule}"
                    )
            for k, v in resume_epoch_info.items():
                wandb.summary[f"resume/{k}"] = v
            wandb.summary["mirror_augmentation"] = config.mirror_augmentation
            wandb.summary["epochs_already_done"] = config.epochs_already_done
            wandb.summary["save_every_epoch"] = config.save_every_epoch
            wandb.summary["physics/prior_normals"] = config.physics_prior_normals
            wandb.summary["physics/prior_smooth"] = config.physics_prior_smooth
            wandb.summary["physics/lambda_normals"] = config.lambda_normals
            wandb.summary["physics/lambda_smooth"] = config.lambda_smooth
            wandb.summary["physics/knn_k"] = config.physics_knn_k
            wandb.summary["physics/smooth_anchors"] = config.physics_smooth_anchors
            wandb.summary["physics/knn_chunk_size"] = config.physics_knn_chunk_size
            if config.physics_prior_normals or config.physics_prior_smooth:
                priors = []
                if config.physics_prior_normals:
                    priors.append(f"normals(lambda={config.lambda_normals})")
                if config.physics_prior_smooth:
                    priors.append(
                        f"smooth(lambda={config.lambda_smooth},k={config.physics_knn_k},"
                        f"anchors={config.physics_smooth_anchors})"
                    )
                print("H347 physics priors: " + " + ".join(priors))

        best_val = float("inf")
        best_metrics: dict[str, float] = {}
        best_checkpoint_source = "ema" if ema is not None else "raw"
        global_step = 0
        early_stop_reason: str | None = None
        timeout_hit = False
        train_start = time.time()

        for epoch in range(config.epochs_already_done, max_epochs):
            if vol_points_schedule:
                desired_vol_points = vol_points_for_epoch(
                    vol_points_schedule, epoch, config.train_volume_points
                )
                if config.debug:
                    desired_vol_points = min(desired_vol_points, 8_192)
                if desired_vol_points != current_train_vol_points:
                    if state.is_main:
                        print(
                            f"Volume-points curriculum: epoch {epoch} -> "
                            f"train_volume_points={desired_vol_points} "
                            f"(was {current_train_vol_points})"
                        )
                    config.train_volume_points = desired_vol_points
                    train_loader = rebuild_train_loader_with_vol_points(
                        config, train_loader, desired_vol_points, state
                    )
                    current_train_vol_points = desired_vol_points
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            timeout_hit = distributed_any(
                state,
                (time.time() - train_start) / 60.0 >= timeout_minutes,
                device,
            )
            if timeout_hit:
                if state.is_main:
                    print(f"Timeout ({timeout_minutes:.1f} min). Stopping.")
                break

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            model.train()
            train_loss_sum = 0.0
            n_batches = 0

            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{max_epochs}",
                leave=False,
                disable=not state.is_main,
            ):
                if config.mirror_augmentation:
                    batch = mirror_augment_batch(batch.to(device), p=0.5)
                gradnorm_metrics: dict[str, float] = {}
                if balancer is not None:
                    per_task_losses = per_task_train_losses(
                        model, batch, transform, device, config.amp_mode
                    )
                    synced_losses = synced_per_task_tensor(
                        [t.detach() for t in per_task_losses], state=state, device=device
                    )
                    balancer.update_initial_loss(synced_losses)

                    synced_norms: torch.Tensor | None = None
                    if balancer.mode == "full":
                        per_task_grad_norms: list[torch.Tensor] = []
                        for L_i in per_task_losses:
                            grads = torch.autograd.grad(
                                L_i,
                                gradnorm_shared_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            norm_sq = torch.zeros((), device=device, dtype=torch.float32)
                            for g in grads:
                                if g is not None:
                                    norm_sq = norm_sq + g.float().square().sum()
                            per_task_grad_norms.append(norm_sq.sqrt())
                        grad_norms_tensor = torch.stack(per_task_grad_norms)
                        synced_norms = synced_per_task_tensor(
                            grad_norms_tensor, state=state, device=device
                        )
                    gradnorm_loss_tensor = balancer.step(synced_losses, synced_norms)
                    if state.enabled:
                        dist.broadcast(balancer.log_weights.data, src=0)

                    W_detached = balancer.task_weights_detached()
                    weighted_terms = [W_detached[i] * per_task_losses[i] for i in range(len(per_task_losses))]
                    loss = weighted_terms[0]
                    for term in weighted_terms[1:]:
                        loss = loss + term
                    surface_unweighted = per_task_losses[0]
                    for term in per_task_losses[1:4]:
                        surface_unweighted = surface_unweighted + term
                    base_mse_total = surface_unweighted + per_task_losses[4]
                    weighted_surface = (
                        W_detached[:4] * torch.stack([t.detach() for t in per_task_losses[:4]])
                    ).sum()
                    weighted_volume = W_detached[4] * per_task_losses[4].detach()
                    batch_loss_metrics = {
                        "base_mse_loss": float(base_mse_total.detach().cpu().item()),
                        "surface_loss": float(surface_unweighted.detach().cpu().item()),
                        "volume_loss": float(per_task_losses[4].detach().cpu().item()),
                        "surface_loss_weighted": float(weighted_surface.cpu().item()),
                        "volume_loss_weighted": float(weighted_volume.cpu().item()),
                    }
                    weights_cpu = W_detached.cpu().tolist()
                    losses_cpu = synced_losses.detach().cpu().tolist()
                    for i, name in enumerate(GRADNORM_TASK_NAMES):
                        gradnorm_metrics[f"gradnorm/weight_{name}"] = weights_cpu[i]
                        gradnorm_metrics[f"train/loss_{name}"] = losses_cpu[i]
                    if synced_norms is not None:
                        norms_cpu = synced_norms.detach().cpu().tolist()
                        for i, name in enumerate(GRADNORM_TASK_NAMES):
                            gradnorm_metrics[f"gradnorm/gnorm_{name}"] = norms_cpu[i]
                    if balancer.mode == "ema_proxy" and balancer.ema_loss is not None:
                        ema_cpu = balancer.ema_loss.detach().cpu().tolist()
                        for i, name in enumerate(GRADNORM_TASK_NAMES):
                            gradnorm_metrics[f"gradnorm/ema_loss_{name}"] = ema_cpu[i]
                        if balancer.initial_loss is not None:
                            r_vec = (
                                balancer.ema_loss / balancer.initial_loss.clamp(min=1e-12)
                            ).detach().cpu().tolist()
                            for i, name in enumerate(GRADNORM_TASK_NAMES):
                                gradnorm_metrics[f"gradnorm/r_{name}"] = r_vec[i]
                    gradnorm_metrics["gradnorm/ready"] = 1.0 if balancer.ready else 0.0
                    if balancer.initial_loss is not None:
                        for i, name in enumerate(GRADNORM_TASK_NAMES):
                            gradnorm_metrics[f"gradnorm/initial_loss_{name}"] = float(
                                balancer.initial_loss[i].cpu().item()
                            )
                    if balancer.mode == "ema_proxy" and balancer.min_weight > 0.0:
                        gradnorm_metrics["gradnorm/min_weight_floor_count"] = float(
                            balancer.last_floor_active
                        )
                    if gradnorm_loss_tensor is not None:
                        gradnorm_metrics["gradnorm/loss"] = float(gradnorm_loss_tensor.cpu().item())
                else:
                    loss, batch_loss_metrics = train_loss(
                        model,
                        batch,
                        transform,
                        device,
                        config.amp_mode,
                        surface_loss_weight=config.surface_loss_weight,
                        volume_loss_weight=config.volume_loss_weight,
                        surface_channel_weights=surface_channel_weights if use_channel_weights else None,
                        physics_prior_normals=config.physics_prior_normals,
                        physics_prior_smooth=config.physics_prior_smooth,
                        lambda_normals=config.lambda_normals,
                        lambda_smooth=config.lambda_smooth,
                        physics_knn_k=config.physics_knn_k,
                        physics_smooth_anchors=config.physics_smooth_anchors,
                        physics_knn_chunk_size=config.physics_knn_chunk_size,
                    )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                loss_is_nonfinite = not bool(torch.isfinite(loss.detach()).item())
                skip_step = distributed_any(state, loss_is_nonfinite, device)
                should_log_model_telemetry = (
                    state.is_main or config.ddp_log_model_telemetry_all_ranks
                )
                should_log_gradients = (
                    should_log_model_telemetry
                    and config.gradient_log_every > 0
                    and global_step % config.gradient_log_every == 0
                )
                should_log_weights = (
                    should_log_model_telemetry
                    and config.weight_log_every > 0
                    and global_step % config.weight_log_every == 0
                )
                train_log: dict[str, object] = {
                    "global_step": global_step,
                    "train/lr": current_lr,
                    "lr": current_lr,
                    "train/vol_points": float(current_train_vol_points),
                    "train/step_skipped": 1.0 if skip_step else 0.0,
                    "train/nonfinite_loss": 1.0 if loss_is_nonfinite else 0.0,
                }
                if not loss_is_nonfinite:
                    train_log.update(
                        {
                            "train/loss": float(loss.detach().cpu().item()),
                            "train/base_mse_loss": batch_loss_metrics["base_mse_loss"],
                            "train/surface_loss": batch_loss_metrics["surface_loss"],
                            "train/volume_loss": batch_loss_metrics["volume_loss"],
                            "train/surface_loss_weighted": batch_loss_metrics["surface_loss_weighted"],
                            "train/volume_loss_weighted": batch_loss_metrics["volume_loss_weighted"],
                            "train/tau_y_loss_weight": config.tau_y_loss_weight,
                            "train/tau_z_loss_weight": config.tau_z_loss_weight,
                        }
                    )
                    for key, value in batch_loss_metrics.items():
                        if key.startswith("physics/"):
                            train_log[f"train/{key}"] = value
                if gradnorm_metrics:
                    train_log.update(gradnorm_metrics)

                if skip_step:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if config.grad_clip_norm > 0.0:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            base_model.parameters(),
                            max_norm=config.grad_clip_norm,
                        )
                    else:
                        grad_norm_tensor = global_grad_norm(base_model.parameters(), device)
                    grad_norm_pre_clip = float(grad_norm_tensor.detach().cpu().item())
                    grad_is_nonfinite = not math.isfinite(grad_norm_pre_clip)
                    skip_step = distributed_any(state, grad_is_nonfinite, device)
                    clipped = (
                        config.grad_clip_norm > 0.0
                        and math.isfinite(grad_norm_pre_clip)
                        and grad_norm_pre_clip > config.grad_clip_norm
                    )
                    train_log.update(
                        {
                            "train/grad/global_norm_pre_clip": grad_norm_pre_clip,
                            "train/grad/clipped": 1.0 if clipped else 0.0,
                            "train/nonfinite_grad": 1.0 if grad_is_nonfinite else 0.0,
                            "train/step_skipped": 1.0 if skip_step else 0.0,
                        }
                    )
                    gradient_metrics = (
                        collect_gradient_metrics(
                            model,
                            log_histograms=config.log_gradient_histograms,
                        )
                        if should_log_gradients and not skip_step
                        else {}
                    )
                    if skip_step:
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.step()
                        if ema is not None:
                            ema.update(base_model)
                        weight_metrics = (
                            collect_weight_metrics(
                                model,
                                log_histograms=config.log_weight_histograms,
                            )
                            if should_log_weights
                            else {}
                        )
                        train_loss_sum += float(loss.detach().cpu().item())
                        n_batches += 1
                        train_log.update(gradient_metrics)
                        train_log.update(weight_metrics)

                train_log.update(
                    train_slope_tracker.update(
                        global_step=global_step,
                        metrics=train_log,
                        namespace="train",
                    )
                )
                local_stop_reason = check_kill_thresholds(
                    global_step=global_step,
                    metrics=train_log,
                    thresholds=kill_thresholds,
                )
                if local_stop_reason is not None:
                    early_stop_reason = local_stop_reason
                stop_requested = distributed_any(state, early_stop_reason is not None, device)
                if stop_requested and early_stop_reason is None:
                    early_stop_reason = "distributed peer requested early stop"
                if early_stop_reason is not None:
                    train_log["early_stop/triggered"] = 1.0
                wandb.log(train_log)
                if early_stop_reason is not None:
                    if state.is_main:
                        print(early_stop_reason)
                    break
                timeout_hit = distributed_any(
                    state,
                    (time.time() - train_start) / 60.0 >= train_timeout_minutes,
                    device,
                )
                if timeout_hit:
                    if state.is_main:
                        print(
                            f"Train timeout ({train_timeout_minutes:.1f} min) mid-epoch "
                            f"at step {global_step}. Forcing validation and stopping."
                        )
                    break

            scheduler.step()
            epoch_train_loss = train_loss_sum / max(n_batches, 1)
            dt = time.time() - t0
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
            should_validate = (
                epoch == 0
                or (epoch + 1) % max(config.validation_every, 1) == 0
                or epoch + 1 == max_epochs
                or (timeout_hit and n_batches > 0)
            )

            log_metrics: dict[str, object] = {
                "train/epoch_loss": epoch_train_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "lr": scheduler.get_last_lr()[0],
                "epoch_time_s": dt,
                "global_step": global_step,
            }
            if early_stop_reason is not None:
                log_metrics["early_stop/triggered"] = 1.0
                wandb.log(log_metrics)
                break

            if not should_validate:
                local_stop_reason = check_kill_thresholds(
                    global_step=global_step,
                    metrics=log_metrics,
                    thresholds=kill_thresholds,
                )
                if local_stop_reason is not None:
                    early_stop_reason = local_stop_reason
                stop_requested = distributed_any(state, early_stop_reason is not None, device)
                if stop_requested and early_stop_reason is None:
                    early_stop_reason = "distributed peer requested early stop"
                if early_stop_reason is not None:
                    log_metrics["early_stop/triggered"] = 1.0
                wandb.log(log_metrics)
                if state.is_main:
                    print(
                        f"Epoch {epoch + 1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB] "
                        f"train_loss={epoch_train_loss:.5f}"
                    )
                if early_stop_reason is not None:
                    if state.is_main:
                        print(early_stop_reason)
                    break
                continue

            # Ease VRAM pressure before val to reduce OOM risk on the SOTA stack
            # (65k surface + 65k volume points / DDP=8). Logs peak memory pre/post-clear.
            if torch.cuda.is_available():
                pre_clear_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                torch.cuda.empty_cache()
                gc.collect()
                post_clear_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                if state.is_main:
                    log_metrics["val/pre_clear_reserved_gb"] = pre_clear_gb
                    log_metrics["val/post_clear_reserved_gb"] = post_clear_gb

            raw_val_metrics = None
            if config.eval_raw_vs_ema and ema is not None:
                raw_val_metrics = {
                    name: evaluate_split(
                        model,
                        loader,
                        transform,
                        device,
                        amp_mode=config.amp_mode,
                        distributed_state=state,
                    )
                    for name, loader in val_loaders.items()
                }
            if ema is not None:
                ema.store(base_model)
                ema.copy_to(base_model)
            val_metrics = {
                name: evaluate_split(
                    model,
                    loader,
                    transform,
                    device,
                    amp_mode=config.amp_mode,
                    distributed_state=state,
                )
                for name, loader in val_loaders.items()
            }

            if state.is_main:
                if raw_val_metrics is not None:
                    raw_surface = raw_val_metrics["val_surface"]
                    log_metrics.update(primary_metric_log("val_raw_primary", raw_surface))
                    for split_name, metrics in raw_val_metrics.items():
                        log_metrics.update(metric_namespace("val_raw", split_name, metrics))
                primary_val = val_metrics["val_surface"]["abupt_axis_mean_rel_l2_pct"]
                log_metrics.update(primary_metric_log("val_primary", val_metrics["val_surface"]))
                for split_name, metrics in val_metrics.items():
                    log_metrics.update(metric_namespace("val", split_name, metrics))
                log_metrics.update(collect_string_sep_metrics(base_model))
                log_metrics.update(
                    val_slope_tracker.update(
                        global_step=global_step,
                        metrics=log_metrics,
                        namespace="val",
                        force=True,
                    )
                )
                local_stop_reason = check_kill_thresholds(
                    global_step=global_step,
                    metrics=log_metrics,
                    thresholds=kill_thresholds,
                )
                if local_stop_reason is not None:
                    early_stop_reason = local_stop_reason
                if early_stop_reason is not None:
                    log_metrics["early_stop/triggered"] = 1.0
                improved = should_update_best_checkpoint(primary_val, best_val)
                if improved:
                    best_val = primary_val
                    best_metrics = {"epoch": float(epoch + 1), **val_metrics["val_surface"]}
                    torch.save(
                        {
                            "model": base_model.state_dict(),
                            "config": asdict(config),
                            "epoch": epoch + 1,
                            "val_metrics": val_metrics,
                            "checkpoint_source": best_checkpoint_source,
                            "selection_metric": "val_primary/abupt_axis_mean_rel_l2_pct",
                        },
                        model_path,
                    )
                log_metrics["best_checkpoint/updated"] = 1.0 if improved else 0.0
                log_metrics["best_checkpoint/valid_primary"] = 1.0 if is_valid_primary_metric(primary_val) else 0.0
                if config.save_every_epoch:
                    epoch_ckpt_path = output_dir / f"checkpoint_ep{epoch + 1}.pt"
                    torch.save(
                        {
                            "model": base_model.state_dict(),
                            "config": asdict(config),
                            "epoch": epoch + 1,
                            "val_metrics": val_metrics,
                            "checkpoint_source": best_checkpoint_source,
                            "selection_metric": "val_primary/abupt_axis_mean_rel_l2_pct",
                        },
                        epoch_ckpt_path,
                    )
                    log_metrics["per_epoch_checkpoint/path"] = str(epoch_ckpt_path)
                wandb.log(log_metrics)
                tag = " *" if improved else ""
                print(
                    f"Epoch {epoch + 1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB] "
                    f"train_loss={epoch_train_loss:.5f} "
                    f"val_abupt_axis_rel_l2_pct={primary_val:.4f}{tag}"
                )
                print_metrics("val_surface", val_metrics["val_surface"])
            else:
                wandb.log(log_metrics)
            if ema is not None:
                ema.restore(base_model)
            stop_requested = distributed_any(state, early_stop_reason is not None, device)
            if stop_requested and early_stop_reason is None:
                early_stop_reason = "rank 0 requested early stop"
            if early_stop_reason is not None:
                if state.is_main:
                    print(early_stop_reason)
                break
            if timeout_hit:
                break

        total_minutes = (time.time() - train_start) / 60.0
        if state.is_main:
            print(f"\nTraining done in {total_minutes:.1f} min")

        if early_stop_reason is not None:
            wandb.summary.update(
                {
                    "early_stop/triggered": 1.0,
                    "early_stop/reason": early_stop_reason,
                    "early_stop/global_step": global_step,
                    "total_train_minutes": total_minutes,
                }
            )
            wandb.finish()
            return

        distributed_barrier(state)
        if not state.is_main:
            wandb.summary.update({"total_train_minutes": total_minutes})
            wandb.finish()
            return

        if not best_metrics:
            wandb.summary.update(
                {
                    "run_invalid": 1.0,
                    "run_invalid/reason": "no finite positive validation checkpoint was saved",
                    "total_train_minutes": total_minutes,
                }
            )
            print("No finite positive validation checkpoint was saved.")
            wandb.finish()
            return

        run_final_evaluation(
            run=run,
            model=base_model,
            model_path=model_path,
            config_path=config_path,
            config=config,
            transform=transform,
            device=device,
            final_val_loaders=final_val_loaders,
            final_test_loaders=final_test_loaders,
            best_metrics=best_metrics,
            best_checkpoint_source=best_checkpoint_source,
            n_params=n_params,
            global_step=global_step,
            total_minutes=total_minutes,
        )
        wandb.finish()
    finally:
        cleanup_distributed(state)


if __name__ == "__main__":
    main()
