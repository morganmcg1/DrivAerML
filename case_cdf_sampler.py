# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""H378 — Online hard-case mining: case-CDF reweighted sampling for τ_z.

The dataset is indexed at the *view* level (each case is split into multiple
point-limited views by ``DrivAerMLSurfaceDataset``); the sampler in this module
operates at the *case* level and projects per-case weights onto per-view
weights uniformly across that case's views.

DDP pattern: rank 0 generates a global index list via ``torch.multinomial``,
broadcasts to all ranks, then each rank slices ``[rank::world_size]``. This
avoids the failure mode where independent per-rank ``WeightedRandomSampler``
instances draw the same case onto multiple ranks in the same step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


# Mapping from channel spec strings to (modality, channel index).
# "modality" is either "surface" (channel of the 4-dim surface_y tensor:
# [cp, tau_x, tau_y, tau_z]) or "volume" (channel of the 1-dim volume_y
# tensor: [volume_pressure]).
CHANNEL_SPECS: dict[str, tuple[str, int]] = {
    "sp": ("surface", 0),
    "surface_pressure": ("surface", 0),
    "wss_x": ("surface", 1),
    "tau_x": ("surface", 1),
    "wss_y": ("surface", 2),
    "tau_y": ("surface", 2),
    "wss_z": ("surface", 3),
    "tau_z": ("surface", 3),
    "vp": ("volume", 0),
    "volume_pressure": ("volume", 0),
}


def parse_channel_spec(spec: str) -> list[tuple[str, int]]:
    """Parse '--case-cdf-sampler-channels' into [(modality, channel_idx), ...]."""

    parts = [token.strip() for token in spec.split(",") if token.strip()]
    if not parts:
        raise ValueError(f"--case-cdf-sampler-channels must list >=1 channel, got '{spec}'")
    out: list[tuple[str, int]] = []
    for token in parts:
        key = token.lower()
        if key not in CHANNEL_SPECS:
            raise ValueError(
                f"Unknown case-CDF channel '{token}'. Supported: {sorted(CHANNEL_SPECS)}"
            )
        out.append(CHANNEL_SPECS[key])
    return out


@dataclass
class CaseCDFState:
    """Per-epoch accumulator for case-level MAE on selected channels.

    sum_abs: [N_cases, n_channels] sum of |pred - target| in normalized space.
    count:   [N_cases, n_channels] number of valid (unmasked) point-channel
             contributions accumulated.

    Both tensors live on the training device for cheap in-place adds during
    the batch loop, then are all-reduced at epoch end.
    """

    sum_abs: torch.Tensor
    count: torch.Tensor
    case_id_to_idx: dict[str, int]
    channels: list[tuple[str, int]]

    @classmethod
    def init(
        cls,
        *,
        case_ids: Sequence[str],
        channels: list[tuple[str, int]],
        device: torch.device,
    ) -> "CaseCDFState":
        n_cases = len(case_ids)
        n_channels = len(channels)
        return cls(
            sum_abs=torch.zeros(n_cases, n_channels, device=device, dtype=torch.float64),
            count=torch.zeros(n_cases, n_channels, device=device, dtype=torch.float64),
            case_id_to_idx={cid: i for i, cid in enumerate(case_ids)},
            channels=channels,
        )

    def reset(self) -> None:
        self.sum_abs.zero_()
        self.count.zero_()

    @torch.no_grad()
    def accumulate(
        self,
        *,
        case_ids: Sequence[str],
        surface_pred_norm: torch.Tensor,
        surface_target_norm: torch.Tensor,
        surface_mask: torch.Tensor,
        volume_pred_norm: torch.Tensor,
        volume_target_norm: torch.Tensor,
        volume_mask: torch.Tensor,
    ) -> None:
        """Accumulate |pred - target| per (case, channel) in normalized space.

        Normalized space is fine because all cases share the same
        train-split normalization constants. The relative ranking of cases by
        per-channel MAE is identical between normalized and original units.
        """

        surface_pred_norm = surface_pred_norm.detach().float()
        surface_target_norm = surface_target_norm.detach().float()
        volume_pred_norm = volume_pred_norm.detach().float()
        volume_target_norm = volume_target_norm.detach().float()
        surf_diff_abs = (surface_pred_norm - surface_target_norm).abs()
        vol_diff_abs = (volume_pred_norm - volume_target_norm).abs()
        for b, cid in enumerate(case_ids):
            c_idx = self.case_id_to_idx.get(cid)
            if c_idx is None:
                continue
            surf_valid = surface_mask[b]
            vol_valid = volume_mask[b]
            surf_n = int(surf_valid.sum().item())
            vol_n = int(vol_valid.sum().item())
            for ch_pos, (modality, ch) in enumerate(self.channels):
                if modality == "surface":
                    if surf_n == 0:
                        continue
                    s = surf_diff_abs[b, surf_valid, ch].sum().double()
                    self.sum_abs[c_idx, ch_pos] += s
                    self.count[c_idx, ch_pos] += float(surf_n)
                else:  # volume
                    if vol_n == 0:
                        continue
                    s = vol_diff_abs[b, vol_valid, ch].sum().double()
                    self.sum_abs[c_idx, ch_pos] += s
                    self.count[c_idx, ch_pos] += float(vol_n)

    def all_reduce(self) -> None:
        """Sum partial accumulators across DDP ranks (no-op if not distributed)."""

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.sum_abs, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

    def per_case_mae(self) -> torch.Tensor:
        """Compute per-case mean absolute error averaged across the selected channels.

        Returns: [N_cases] float64 tensor (on the same device as sum_abs).
        Cases with zero contributions on every channel get MAE = 0; weight
        normalization keeps them sampled at the eps floor so they are not
        silently dropped.
        """

        per_channel_mae = self.sum_abs / self.count.clamp(min=1.0)
        valid = (self.count > 0).double()
        per_channel_mae = per_channel_mae * valid
        denom = valid.sum(dim=1).clamp(min=1.0)
        return per_channel_mae.sum(dim=1) / denom


def compute_case_weights(
    per_case_mae: torch.Tensor,
    *,
    alpha: float,
    eps: float,
    clip_factor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute normalized per-case sampling weights with soft clipping.

    Args:
        per_case_mae: [N_cases] float tensor of per-case MAE (or any positive
            hardness score).
        alpha: priority exponent. alpha=0 → uniform (returns 1/N).
        eps: numerical floor added before exponentiation to keep
            zero-MAE cases samplable.
        clip_factor: soft ceiling on per-case weight, expressed as a multiple
            of the mean raw weight. Recommended 10-20 to bound ESS collapse
            when one case dominates. Set to 0 or non-positive to disable.

    Returns:
        weights: [N_cases] normalized weights, sums to 1.
        diagnostics: dict of scalars (ESS, entropy, min/max/median weight,
            clip count).
    """

    p = per_case_mae.clamp(min=0.0).double()
    n = p.numel()
    if alpha == 0.0:
        w = torch.full_like(p, 1.0 / float(n))
        clip_active = 0
    else:
        w_raw = (p + eps) ** alpha
        clip_active = 0
        if clip_factor > 0.0:
            # Soft cap relative to the MEDIAN raw weight. Median is robust
            # to extreme outliers (which the cap is meant to constrain),
            # so a single 1000x-overshooting case does not pull the cap
            # threshold up to its own value.
            median_raw = float(w_raw.median().item())
            cap = float(clip_factor) * max(median_raw, 1e-30)
            over = w_raw > cap
            clip_active = int(over.sum().item())
            if clip_active > 0:
                w_raw = w_raw.clamp(max=cap)
        w = w_raw / w_raw.sum().clamp(min=1e-30)
    n = float(w.numel())
    p_normalized = w  # by construction sums to 1
    ess = float(1.0 / (p_normalized * p_normalized).sum().clamp(min=1e-30).item())
    safe_log = (p_normalized.clamp(min=1e-30)).log()
    entropy = float((-(p_normalized * safe_log)).sum().item())
    entropy_uniform = float(torch.log(torch.tensor(n)).item())
    diagnostics = {
        "case_cdf/ess": ess,
        "case_cdf/ess_frac": ess / n,
        "case_cdf/entropy": entropy,
        "case_cdf/entropy_uniform": entropy_uniform,
        "case_cdf/entropy_ratio": entropy / max(entropy_uniform, 1e-30),
        "case_cdf/w_min": float(p_normalized.min().item()),
        "case_cdf/w_max": float(p_normalized.max().item()),
        "case_cdf/w_median": float(p_normalized.median().item()),
        "case_cdf/w_mean": float(p_normalized.mean().item()),
        "case_cdf/clip_active": float(clip_active if alpha != 0.0 else 0),
        "case_cdf/n_cases": float(n),
    }
    return w, diagnostics


def case_to_view_weights(
    case_weights: torch.Tensor,
    case_id_to_idx: dict[str, int],
    views: Iterable,  # iterable of PointView objects with `.case_id`
) -> torch.Tensor:
    """Project per-case weights onto per-view weights.

    For each case c with weight w_c and v_c views, every view of c gets
    view weight w_c / v_c so the total sampling mass of case c remains w_c
    regardless of view count.
    """

    n_cases = case_weights.numel()
    views_per_case = torch.zeros(n_cases, dtype=torch.float64)
    case_idx_per_view: list[int] = []
    for view in views:
        c_idx = case_id_to_idx[view.case_id]
        views_per_case[c_idx] += 1.0
        case_idx_per_view.append(c_idx)
    case_idx_tensor = torch.tensor(case_idx_per_view, dtype=torch.long)
    cw_cpu = case_weights.detach().cpu().double()
    per_case_share = cw_cpu / views_per_case.clamp(min=1.0)
    view_weights = per_case_share[case_idx_tensor]
    # WeightedRandomSampler tolerates zeros if any case has zero weight,
    # but we want every case still potentially samplable, so floor at a
    # tiny value relative to the mean. The weights themselves were already
    # produced with an eps floor; this is belt-and-suspenders.
    floor = 1e-12 * view_weights.mean().clamp(min=1e-30)
    return view_weights.clamp(min=floor).double()


class DistributedCaseCDFSampler(Sampler[int]):
    """DDP-safe weighted view sampler driven by per-case weights.

    Each ``__iter__`` call:
      1. (rank 0) draws ``num_samples_total`` view indices via
         ``torch.multinomial`` with the current ``view_weights``.
      2. broadcasts the index tensor to all ranks.
      3. each rank slices ``[rank::world_size]`` (interleaved shard) and
         iterates its slice. Length per rank is exactly
         ``num_samples_total // world_size`` so DDP batch counts stay aligned.

    ``set_epoch(epoch)`` advances the deterministic generator seed so a
    smoke run with the same ``--case-cdf-sampler-alpha 0.0`` and equal
    weights is reproducible across epochs.

    Weights are stored on CPU as float64 to dodge bfloat16 multinomial
    instability; the actual sampling is also on CPU since multinomial
    is generally cheap at this dataset size (~2000 views, 400 cases).
    """

    def __init__(
        self,
        *,
        num_samples_total: int,
        world_size: int,
        rank: int,
        seed: int = 0,
        view_weights: torch.Tensor | None = None,
    ):
        if num_samples_total <= 0:
            raise ValueError("num_samples_total must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} not in [0, {world_size})")
        # Round num_samples_total down to a multiple of world_size so each
        # rank has the same number of samples (drop_last=True semantics).
        self.num_samples_total = (num_samples_total // world_size) * world_size
        if self.num_samples_total == 0:
            self.num_samples_total = world_size
        self.world_size = world_size
        self.rank = rank
        self.seed = int(seed)
        self.epoch = 0
        if view_weights is None:
            raise ValueError("DistributedCaseCDFSampler requires initial view_weights")
        self._view_weights = view_weights.detach().cpu().double().contiguous()

    def __len__(self) -> int:
        return self.num_samples_total // self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def view_weights(self) -> torch.Tensor:
        return self._view_weights

    def set_view_weights(self, view_weights: torch.Tensor) -> None:
        if view_weights.numel() != self._view_weights.numel():
            raise ValueError(
                f"set_view_weights: expected {self._view_weights.numel()} values, "
                f"got {view_weights.numel()}"
            )
        self._view_weights = view_weights.detach().cpu().double().contiguous()

    def __iter__(self):
        # All ranks generate identical indices using the same seed; this is
        # both cheap (multinomial on a few-thousand-element distribution)
        # and avoids any broadcast cost.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + 100003 * self.epoch)
        indices = torch.multinomial(
            self._view_weights,
            self.num_samples_total,
            replacement=True,
            generator=gen,
        )
        my_slice = indices[self.rank :: self.world_size].tolist()
        return iter(my_slice)
