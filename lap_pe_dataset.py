# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""H360: surface graph Laplacian eigenfunction positional encoding (LapPE).

Concatenates a precomputed 32-channel surface Laplacian PE to ``surface_x``.
The reference loader returns ``[x, y, z, nx, ny, nz, area]``
(``SURFACE_X_DIM = 7``); this wrapper appends ``lap_pe_0..lap_pe_K-1`` after
``area`` so the model receives
``[x, y, z, nx, ny, nz, area, lap_pe_0, ..., lap_pe_K-1]``.

LapPE was precomputed once on a per-case 8192-point random subgraph (k_graph=8
kNN) with 33 smallest eigenpairs of the unnormalized combinatorial Laplacian;
the trivial mode is dropped and the next 32 are kept (k=1..32, sign-canonical,
per-channel std=1). Per-case files:

- ``<lap_pe_root>/<case_id>_lappe.npy`` shape ``(n_sub=8192, 32)`` float32
- ``<lap_pe_root>/<case_id>_nnidx.npy`` shape ``(n_full,)`` int32: NN index
  mapping each full-surface vertex to its nearest subgraph vertex, used to
  interpolate LapPE back to the full surface in O(1) per point.

Row alignment with the parent loader is identical to the H348 curvature
wrapper: the base ``DrivAerMLSurfaceDataset`` samples ``surface_rows`` from
the canonical mesh; we reproduce the same row indices and gather LapPE via
``lappe[nnidx[surface_rows]]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data import SURFACE_Y_DIM, VOLUME_X_DIM, VOLUME_Y_DIM, SurfaceBatch
from data.loader import (
    DrivAerMLCase,
    DrivAerMLCaseStore,
    DrivAerMLSurfaceDataset,
    PointView,
)


DEFAULT_LAP_PE_ROOT = "/mnt/new-pvc/Processed/lap_pe_v1"
LAP_PE_MAX_CHANNELS = 32


def pad_collate_variable_surface(samples: list[DrivAerMLCase]) -> SurfaceBatch:
    """pad_collate variant that infers surface_x last-dim from the samples.

    The base ``pad_collate`` hardcodes ``SURFACE_X_DIM`` (= 7) which causes a
    shape mismatch when LapPE columns extend the per-sample width. This
    variant infers the width from sample[0] and asserts uniformity.
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


class LapPeAugmentedDataset(DrivAerMLSurfaceDataset):
    """Wrap the base surface dataset and append LapPE channels to ``surface_x``.

    See module docstring for the precomputed file layout. The implementation
    overrides ``__getitem__`` end-to-end (rather than augmenting after a
    parent call) so the surface row sampling stays in lockstep with the base
    loader's RNG / chunk strategy without depending on private state.
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
        lap_pe_root: str | Path = DEFAULT_LAP_PE_ROOT,
        n_channels: int = LAP_PE_MAX_CHANNELS,
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
        if not (1 <= n_channels <= LAP_PE_MAX_CHANNELS):
            raise ValueError(
                f"n_channels must be in [1, {LAP_PE_MAX_CHANNELS}], got {n_channels}"
            )
        self.lap_pe_root = Path(lap_pe_root)
        self.n_channels = int(n_channels)

        for cid in self.case_ids:
            lp = self.lap_pe_root / f"{cid}_lappe.npy"
            ni = self.lap_pe_root / f"{cid}_nnidx.npy"
            if not lp.exists():
                raise FileNotFoundError(f"Missing LapPE for case {cid!r} at {lp}")
            if not ni.exists():
                raise FileNotFoundError(f"Missing LapPE nnidx for case {cid!r} at {ni}")

    def _load_lap_pe(self, case_id: str, surface_rows: torch.Tensor | None) -> np.ndarray:
        """Load LapPE for the rows the base loader will return."""
        lappe_sub = np.load(self.lap_pe_root / f"{case_id}_lappe.npy", mmap_mode="r")
        nnidx = np.load(self.lap_pe_root / f"{case_id}_nnidx.npy", mmap_mode="r")
        if surface_rows is None:
            sub_idx = np.asarray(nnidx)
        else:
            sub_idx = np.asarray(nnidx[surface_rows.numpy()])
        # Gather (n_loaded, K) and truncate to n_channels.
        feats = np.asarray(lappe_sub[sub_idx], dtype=np.float32)
        return feats[:, : self.n_channels]

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
            lap = self._load_lap_pe(view.case_id, surface_idx)
            if lap.shape[0] != n_surface_loaded:
                raise RuntimeError(
                    f"LapPE row count {lap.shape[0]} mismatches surface row count "
                    f"{n_surface_loaded} for case {view.case_id!r}"
                )
            lap_t = torch.from_numpy(lap)
            surface_x = torch.cat([case.surface_x, lap_t], dim=-1)
        else:
            surface_x = case.surface_x.new_zeros((0, case.surface_x.shape[1] + self.n_channels))

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
        metadata["lap_pe_channels"] = int(self.n_channels)
        metadata["lap_pe_root"] = str(self.lap_pe_root)

        return DrivAerMLCase(
            case_id=case.case_id,
            surface_x=surface_x,
            surface_y=case.surface_y,
            volume_x=case.volume_x,
            volume_y=case.volume_y,
            metadata=metadata,
        )
