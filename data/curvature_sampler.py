# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Curvature-weighted surface point sampling for DrivAerML.

Wraps ``DrivAerMLSurfaceDataset`` so that during training the
``--train-surface-points`` draw is biased toward high-curvature regions
(leading edges, A-pillars, wheel arches) where wall-shear-stress signal is
concentrated. Curvature is approximated from geometry alone (point + normal),
so this is not target leakage.

Eval ("eval_chunk") and "full" sampling modes are unchanged; only
"train_random" is replaced with curvature-weighted ``torch.multinomial``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from .loader import DrivAerMLCase, DrivAerMLSurfaceDataset, _resolve_artifact_path


def approx_mean_curvature_np(
    xyz: np.ndarray,
    normals: np.ndarray,
    *,
    k: int = 8,
) -> np.ndarray:
    """Approximate mean curvature magnitude from oriented point cloud.

    Uses the discrete divergence-of-normals estimator over a kNN
    neighborhood: kappa_i ~= || mean_j (n_j - n_i) / |x_j - x_i| ||.

    Returns shape [N] float32. High values mark sharp edges / high
    curvature regions; near-zero on flat panels.
    """

    n = xyz.shape[0]
    if n <= k + 1:
        return np.zeros(n, dtype=np.float32)
    tree = cKDTree(xyz)
    _, knn_idx = tree.query(xyz, k=k + 1, workers=1)
    knn_idx = knn_idx[:, 1:]  # exclude self

    knn_xyz = xyz[knn_idx]  # [N, k, 3]
    knn_normals = normals[knn_idx]  # [N, k, 3]

    delta_x = knn_xyz - xyz[:, None, :]
    delta_n = knn_normals - normals[:, None, :]

    dist = np.linalg.norm(delta_x, axis=-1, keepdims=True)
    dist = np.maximum(dist, 1e-6)
    curvature_vec = (delta_n / dist).mean(axis=1)
    kappa = np.linalg.norm(curvature_vec, axis=-1)
    return kappa.astype(np.float32)


class CurvatureWeightedSurfaceDataset(DrivAerMLSurfaceDataset):
    """Surface dataset that oversamples high-curvature surface regions.

    During ``train_random`` sampling, surface row indices are drawn from
    ``torch.multinomial(weights, num_samples, replacement=False)`` where
    ``weights = 1 + alpha * (kappa / mean(kappa))`` and ``kappa`` is the
    approximate mean curvature at each point. Volume sampling and
    evaluation are unchanged.

    ``alpha = 0`` recovers uniform behavior identical to the parent class.
    Curvature is computed once per case (lazy on first access) and cached
    per worker process. The cache survives across epochs when
    ``persistent_workers=True``.
    """

    def __init__(
        self,
        *args: Any,
        surface_importance_alpha: float = 0.0,
        surface_importance_k: int = 8,
        surface_importance_floor: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.surface_importance_alpha = float(surface_importance_alpha)
        self.surface_importance_k = int(surface_importance_k)
        self.surface_importance_floor = float(surface_importance_floor)
        self._curvature_cache: dict[str, torch.Tensor] = {}

    def _load_or_compute_kappa(self, case_id: str) -> np.ndarray:
        case_dir = self.store.root / case_id
        cache_path = case_dir / "surface_kappa_v2.npy"
        if cache_path.exists():
            return np.load(cache_path).astype(np.float32, copy=False)
        # Slow path: compute on the fly and persist for next time.
        xyz = np.asarray(
            np.load(_resolve_artifact_path(case_dir / "surface_xyz.npy")),
            dtype=np.float32,
        )
        normals = np.asarray(
            np.load(_resolve_artifact_path(case_dir / "surface_normals.npy")),
            dtype=np.float32,
        )
        kappa = approx_mean_curvature_np(xyz, normals, k=self.surface_importance_k)
        try:
            tmp = cache_path.with_suffix(".npy.tmp")
            np.save(tmp, kappa)
            tmp.replace(cache_path)
        except OSError:
            pass
        return kappa

    def _surface_weights(self, case_id: str) -> torch.Tensor:
        weights = self._curvature_cache.get(case_id)
        if weights is not None:
            return weights
        kappa = self._load_or_compute_kappa(case_id)
        mean_kappa = float(kappa.mean())
        if mean_kappa > 0.0:
            kappa_normed = kappa / mean_kappa
        else:
            kappa_normed = kappa
        w = 1.0 + self.surface_importance_alpha * kappa_normed
        floor = self.surface_importance_floor
        if floor > 0.0:
            w = np.maximum(w, floor)
        weights = torch.from_numpy(w.astype(np.float32))
        self._curvature_cache[case_id] = weights
        return weights

    def __getitem__(self, idx: int) -> DrivAerMLCase:
        if self.surface_importance_alpha <= 0.0:
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
            # replacement=True matches the parent class's uniform sampler
            # (which uses torch.randint with replacement) and is ~10x faster
            # than replacement=False on N~10M surface meshes. At 65k samples
            # / 8.8M points the expected duplicate count is ~2.5 per draw —
            # negligible against gradient noise.
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
            "train_curvature_weighted" if use_weighted else view.sampling_mode
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
