# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Spatial-prior surface point sampling for DrivAerML.

Wraps ``DrivAerMLSurfaceDataset`` so that during training the
``--train-surface-points`` draw is biased toward (a) the front of the
vehicle in the canonical x-axis and (b) far-from-ground-plane regions
along the canonical z-axis. These coordinate priors carry the highest
empirical Pearson correlation with |WSS| on DrivAerML
(``pearson(-x, |WSS|) ≈ +0.236``, ``pearson(|z|, |WSS|) ≈ +0.176``;
≈4x κ on the same 7-case sample, PR #1113 diagnostic).

Per-point importance weight:

    front_bias  = (x_max - x) / (x_max - x_min)   # 0 back, 1 front
    ground_bias = (|z| - |z|_min) / (|z|_max - |z|_min)  # 0 ground, 1 far
    weight = 1 + alpha * (front_bias + ground_bias) / 2

Computed once per case (lazy on first access) and cached per worker.
Eval ("eval_chunk") and "full" sampling modes are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .loader import DrivAerMLCase, DrivAerMLSurfaceDataset, _resolve_artifact_path


def compute_spatial_prior_weights_np(
    xyz: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Return per-point spatial-prior weights given full surface xyz.

    ``xyz`` is expected as float32 ``[N, 3]`` in the canonical vehicle frame:
    +x toward the rear (so ``-x`` ranks front-of-car high), z aligned with
    the vertical axis (|z| ranks far-from-ground high).
    """

    if alpha <= 0.0:
        return np.ones(xyz.shape[0], dtype=np.float32)
    x = xyz[:, 0]
    z = xyz[:, 2]
    x_min = float(x.min())
    x_max = float(x.max())
    x_range = max(x_max - x_min, 1e-8)
    front_bias = (x_max - x) / x_range  # 0 at back, 1 at front
    z_abs = np.abs(z)
    z_abs_min = float(z_abs.min())
    z_abs_max = float(z_abs.max())
    z_range = max(z_abs_max - z_abs_min, 1e-8)
    ground_bias = (z_abs - z_abs_min) / z_range  # 0 near ground, 1 far
    combined = 0.5 * (front_bias + ground_bias)
    weights = 1.0 + alpha * combined
    return weights.astype(np.float32, copy=False)


class SpatialPriorSurfaceDataset(DrivAerMLSurfaceDataset):
    """Surface dataset that oversamples high spatial-prior surface regions.

    During ``train_random`` sampling, surface row indices are drawn from
    ``torch.multinomial(weights, num_samples, replacement=True)`` where
    ``weights`` follow ``compute_spatial_prior_weights_np``. Volume
    sampling and evaluation are unchanged.

    ``spatial_alpha = 0`` recovers uniform behavior identical to the
    parent class. ``replacement=True`` is intentional: it matches the
    parent uniform sampler (which uses ``torch.randint`` with
    replacement) and is ~10x faster than ``replacement=False`` on
    N~10M surface meshes; expected duplicate count at 65k/8.8M is
    negligible against gradient noise (PR #1113 throughput proven).
    """

    def __init__(
        self,
        *args: Any,
        spatial_alpha: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_alpha = float(spatial_alpha)
        self._weight_cache: dict[str, torch.Tensor] = {}

    def _load_full_xyz(self, case_id: str) -> np.ndarray:
        case_dir = self.store.root / case_id
        path = _resolve_artifact_path(case_dir / "surface_xyz.npy")
        arr = np.load(path, mmap_mode="r")
        return np.asarray(arr, dtype=np.float32)

    def _surface_weights(self, case_id: str) -> torch.Tensor:
        cached = self._weight_cache.get(case_id)
        if cached is not None:
            return cached
        xyz = self._load_full_xyz(case_id)
        w = compute_spatial_prior_weights_np(xyz, alpha=self.spatial_alpha)
        tensor = torch.from_numpy(w)
        self._weight_cache[case_id] = tensor
        return tensor

    def __getitem__(self, idx: int) -> DrivAerMLCase:
        if self.spatial_alpha <= 0.0:
            return super().__getitem__(idx)
        view = self.views[idx]
        counts = self.store.case_point_counts(view.case_id)
        n_surface = int(counts["n_surface"])
        max_surface = int(self.max_surface_points)
        use_weighted = (
            view.sampling_mode == "train_random"
            and max_surface > 0
            and n_surface > max_surface
            and view.view_index < view.surface_view_count
        )
        if use_weighted:
            weights = self._surface_weights(view.case_id)
            sampled = torch.multinomial(weights, max_surface, replacement=True)
            surface_idx = sampled.sort().values
        else:
            surface_idx = self._indices(
                n_surface,
                max_surface,
                view,
                group_view_count=view.surface_view_count,
            )
        volume_idx = self._indices(
            int(counts["n_volume"]),
            int(self.max_volume_points),
            view,
            group_view_count=view.volume_view_count,
        )
        case = self.store.load_case(
            view.case_id,
            surface_rows=None if surface_idx is None else surface_idx.numpy(),
            volume_rows=None if volume_idx is None else volume_idx.numpy(),
        )
        metadata = dict(case.metadata)
        metadata["n_surface_full"] = int(counts["n_surface"])
        metadata["n_surface_loaded"] = int(case.surface_x.shape[0])
        metadata["surface_view_index"] = int(view.view_index)
        metadata["surface_view_count"] = int(view.surface_view_count)
        metadata["surface_sampling_mode"] = (
            "train_spatial_prior" if use_weighted else view.sampling_mode
        )
        metadata["n_volume_full"] = int(counts["n_volume"])
        metadata["n_volume_loaded"] = int(case.volume_x.shape[0])
        metadata["volume_view_index"] = int(view.view_index)
        metadata["volume_view_count"] = int(view.volume_view_count)
        metadata["volume_sampling_mode"] = view.sampling_mode
        metadata["joint_view_count"] = int(view.view_count)
        return DrivAerMLCase(
            case_id=case.case_id,
            surface_x=case.surface_x,
            surface_y=case.surface_y,
            volume_x=case.volume_x,
            volume_y=case.volume_y,
            metadata=metadata,
        )
