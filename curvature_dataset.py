# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""H348: per-vertex surface-curvature features concatenated to ``surface_x``.

The DrivAerML reference loader returns surface features
``[x, y, z, nx, ny, nz, area]`` (``SURFACE_X_DIM = 7``). This module wraps
:class:`DrivAerMLSurfaceDataset` and appends one or two per-vertex curvature
channels after ``area``, so the model receives
``[x, y, z, nx, ny, nz, area, curv_0, (curv_1)]``.

Channel selection is controlled by ``--curvature-mode``:

- ``none``: pass-through (this wrapper is not used).
- ``H``: append normalized mean curvature.
- ``K``: append normalized Gaussian curvature.
- ``k1k2``: append normalized principal curvatures (sign-canonicalized ``k1 ≥ k2``).

Curvature is loaded from one of two precomputed roots:

- ``hk_root/<case>.npy`` shape ``(N_full, 2)`` with col 0 = H, col 1 = K
- ``k1k2_root/<case>.npy`` shape ``(N_full, 2)`` with col 0 = k1, col 1 = k2

Per-channel global p5/p95 stats live in ``curvature_normalization.pt`` (see
``precompute_curvature_stats.py``) and are applied as a clip-to-[-3, 3] of the
linear remap ``(x - center) / half_range`` where ``center = (p5+p95)/2``,
``half_range = (p95-p5)/2``.

The sampling alignment with the base loader is critical: the base loader
samples ``surface_rows`` from the canonical mesh; this wrapper reproduces the
same RNG / chunk strategy so the appended channels stay row-aligned with the
xyz/normals/area returned by the parent ``store.load_case``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data import SURFACE_Y_DIM, VOLUME_X_DIM, VOLUME_Y_DIM, SurfaceBatch
from data.loader import (
    DrivAerMLCase,
    DrivAerMLCaseStore,
    DrivAerMLSurfaceDataset,
    PointView,
)


def pad_collate_variable_surface(samples: list[DrivAerMLCase]) -> SurfaceBatch:
    """pad_collate variant that uses the sample's surface_x last-dim instead of SURFACE_X_DIM.

    The base ``pad_collate`` hardcodes ``SURFACE_X_DIM`` (= 7) which truncates
    or shape-mismatches when curvature columns extend the per-sample width.
    This variant infers the width from sample[0] and asserts uniformity.
    """
    if not samples:
        raise ValueError("pad_collate_variable_surface received an empty batch")
    surface_x_dim = samples[0].surface_x.shape[-1]
    for s in samples:
        if s.surface_x.shape[-1] != surface_x_dim:
            raise RuntimeError(
                f"Inconsistent surface_x widths in batch: {surface_x_dim} vs {s.surface_x.shape[-1]}"
            )
    max_surface_n = max(s.surface_x.shape[0] for s in samples)
    max_volume_n = max(s.volume_x.shape[0] for s in samples)
    batch_size = len(samples)
    surface_x = torch.zeros(batch_size, max_surface_n, surface_x_dim, dtype=samples[0].surface_x.dtype)
    surface_y = torch.zeros(batch_size, max_surface_n, SURFACE_Y_DIM, dtype=samples[0].surface_y.dtype)
    surface_mask = torch.zeros(batch_size, max_surface_n, dtype=torch.bool)
    volume_x = torch.zeros(batch_size, max_volume_n, VOLUME_X_DIM, dtype=samples[0].volume_x.dtype)
    volume_y = torch.zeros(batch_size, max_volume_n, VOLUME_Y_DIM, dtype=samples[0].volume_y.dtype)
    volume_mask = torch.zeros(batch_size, max_volume_n, dtype=torch.bool)
    for i, sample in enumerate(samples):
        sn = sample.surface_x.shape[0]
        vn = sample.volume_x.shape[0]
        surface_x[i, :sn] = sample.surface_x
        surface_y[i, :sn] = sample.surface_y
        surface_mask[i, :sn] = True
        volume_x[i, :vn] = sample.volume_x
        volume_y[i, :vn] = sample.volume_y
        volume_mask[i, :vn] = True
    return SurfaceBatch(
        case_ids=[s.case_id for s in samples],
        surface_x=surface_x,
        surface_y=surface_y,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_y=volume_y,
        volume_mask=volume_mask,
        metadata=[dict(s.metadata) for s in samples],
    )

CURV_CHANNEL_NAMES: dict[str, tuple[str, ...]] = {
    "none": (),
    "H": ("H",),
    "K": ("K",),
    "k1k2": ("k1", "k2"),
}

DEFAULT_HK_ROOT = "/mnt/new-pvc/Processed/curvatures_haku_v1"
DEFAULT_K1K2_ROOT = "/mnt/new-pvc/Processed/curvatures_haku_v1_k1k2"
DEFAULT_STATS_PATH = "curvature_normalization.pt"


def n_curvature_channels(mode: str) -> int:
    if mode not in CURV_CHANNEL_NAMES:
        raise ValueError(f"Unknown --curvature-mode {mode!r}; expected one of {sorted(CURV_CHANNEL_NAMES)}")
    return len(CURV_CHANNEL_NAMES[mode])


class CurvatureNormalizer:
    """Robust p5/p95 clip-and-rescale per channel."""

    def __init__(self, stats_path: str | Path = DEFAULT_STATS_PATH):
        payload = torch.load(stats_path, weights_only=False)
        self.stats: dict[str, dict] = payload["stats"]
        self.hk_root: Path = Path(payload.get("hk_root", DEFAULT_HK_ROOT))
        self.k1k2_root: Path = Path(payload.get("k1k2_root", DEFAULT_K1K2_ROOT))
        # Precompute affine constants once.
        self._affine: dict[str, tuple[float, float]] = {}
        for name, s in self.stats.items():
            p5 = float(s["p5"])
            p95 = float(s["p95"])
            half = max((p95 - p5) / 2.0, 1e-8)
            center = (p95 + p5) / 2.0
            self._affine[name] = (center, half)

    def apply(self, arr: np.ndarray, channel: str) -> np.ndarray:
        center, half = self._affine[channel]
        out = (arr - center) / half
        return np.clip(out, -3.0, 3.0).astype(np.float32)


class CurvatureAugmentedDataset(DrivAerMLSurfaceDataset):
    """Wrap the base surface dataset and concatenate curvature channels to surface_x.

    ``base.__getitem__`` returns a :class:`DrivAerMLCase` whose ``surface_x`` is the
    row-aligned ``[xyz | normals | area]`` (after subsampling). To keep the
    appended curvature row-aligned, we reproduce the exact ``surface_rows``
    that the base loader would have generated, load curvature with those rows,
    normalize, and append columns.

    The implementation overrides ``__getitem__`` end-to-end (rather than
    augmenting after a parent call) so the random surface sampling stays in
    lockstep across modalities without depending on private state.
    """

    def __init__(
        self,
        case_ids: list[str],
        *,
        store: DrivAerMLCaseStore | None = None,
        manifest_path: str = "data/split_manifest.json",
        root: str | None = None,
        max_points: int | None = None,
        max_surface_points: int = 0,
        max_volume_points: int = 0,
        sampling_mode: str = "full",
        curvature_mode: str = "H",
        normalizer: CurvatureNormalizer | None = None,
    ):
        super().__init__(
            case_ids,
            store=store,
            manifest_path=manifest_path,
            root=root,
            max_points=max_points,
            max_surface_points=max_surface_points,
            max_volume_points=max_volume_points,
            sampling_mode=sampling_mode,
        )
        if curvature_mode == "none":
            raise ValueError("CurvatureAugmentedDataset should not be used when curvature_mode='none'")
        if curvature_mode not in CURV_CHANNEL_NAMES:
            raise ValueError(f"Unknown curvature_mode={curvature_mode!r}")
        self.curvature_mode = curvature_mode
        self.normalizer = normalizer or CurvatureNormalizer()
        self.channels = CURV_CHANNEL_NAMES[curvature_mode]

        if curvature_mode == "k1k2":
            self.curv_root = self.normalizer.k1k2_root
            self._col_indices = (0, 1)  # k1, k2
        else:
            self.curv_root = self.normalizer.hk_root
            self._col_indices = (0,) if curvature_mode == "H" else (1,)  # H -> 0, K -> 1

        for cid in self.case_ids:
            p = self.curv_root / f"{cid}.npy"
            if not p.exists():
                raise FileNotFoundError(f"Missing curvature for case {cid!r} at {p}")

    def _load_curvature(self, case_id: str, surface_rows: torch.Tensor | None) -> np.ndarray:
        """Load curvature for a case, row-aligned with the parent loader's sampling."""
        arr = np.load(self.curv_root / f"{case_id}.npy", mmap_mode="r")
        if surface_rows is None:
            slab = np.asarray(arr, dtype=np.float32)
        else:
            slab = np.asarray(arr[surface_rows.numpy()], dtype=np.float32)
        # Select requested columns and normalize.
        cols = []
        for ch_name, col in zip(self.channels, self._col_indices):
            cols.append(self.normalizer.apply(slab[:, col], ch_name))
        return np.stack(cols, axis=-1)  # (n_loaded, n_curv)

    def __getitem__(self, idx: int) -> DrivAerMLCase:
        view: PointView = self.views[idx]
        counts = self.store.case_point_counts(view.case_id)
        surface_idx = self._indices(
            counts["n_surface"],
            self.max_surface_points,
            view,
            group_view_count=view.surface_view_count,
        )
        volume_idx = self._indices(
            counts["n_volume"],
            self.max_volume_points,
            view,
            group_view_count=view.volume_view_count,
        )
        case = self.store.load_case(
            view.case_id,
            surface_rows=None if surface_idx is None else surface_idx.numpy(),
            volume_rows=None if volume_idx is None else volume_idx.numpy(),
        )

        n_surface_loaded = case.surface_x.shape[0]
        if n_surface_loaded > 0:
            curv = self._load_curvature(view.case_id, surface_idx)
            if curv.shape[0] != n_surface_loaded:
                raise RuntimeError(
                    f"Curvature row count {curv.shape[0]} mismatches surface row count "
                    f"{n_surface_loaded} for case {view.case_id!r}"
                )
            curv_t = torch.from_numpy(curv)
            surface_x = torch.cat([case.surface_x, curv_t], dim=-1)
        else:
            surface_x = case.surface_x.new_zeros((0, case.surface_x.shape[1] + len(self.channels)))

        metadata = dict(case.metadata)
        metadata["n_surface_full"] = int(counts["n_surface"])
        metadata["n_surface_loaded"] = int(n_surface_loaded)
        metadata["surface_view_index"] = int(view.view_index)
        metadata["surface_view_count"] = int(view.surface_view_count)
        metadata["surface_sampling_mode"] = view.sampling_mode
        metadata["n_volume_full"] = int(counts["n_volume"])
        metadata["n_volume_loaded"] = int(case.volume_x.shape[0])
        metadata["volume_view_index"] = int(view.view_index)
        metadata["volume_view_count"] = int(view.volume_view_count)
        metadata["volume_sampling_mode"] = view.sampling_mode
        metadata["joint_view_count"] = int(view.view_count)
        metadata["curvature_mode"] = self.curvature_mode
        metadata["curvature_channels"] = list(self.channels)

        return DrivAerMLCase(
            case_id=case.case_id,
            surface_x=surface_x,
            surface_y=case.surface_y,
            volume_x=case.volume_x,
            volume_y=case.volume_y,
            metadata=metadata,
        )
