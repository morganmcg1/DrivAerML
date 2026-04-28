# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""DrivAerML dataset loaders for the packaged processed PVC arrays.

The processed dataset provides one directory per case containing `.npy` arrays:

- `surface_xyz.npy`
- `surface_normals.npy`
- `surface_area.npy`
- `surface_cp.npy`
- `surface_wallshearstress.npy`
- `volume_xyz.npy`
- `volume_sdf.npy`
- `volume_pressure.npy`

This repo predicts surface pressure, wall shear / friction, and volume pressure.
"""

from __future__ import annotations

import functools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .split_utils import expand_pvc_candidates, first_existing

DEFAULT_MANIFEST = Path(__file__).with_name("split_manifest.json")
SURFACE_X_DIM = 7  # xyz(3) + normals(3) + panel area(1)
SURFACE_Y_DIM = 4  # cp(1) + wall shear stress(3)
VOLUME_X_DIM = 4  # xyz(3) + sdf(1)
VOLUME_Y_DIM = 1  # volume pressure
SURFACE_TARGET_NAMES = ("surface_pressure", "wall_shear_x", "wall_shear_y", "wall_shear_z")
VOLUME_TARGET_NAMES = ("volume_pressure",)
EXPECTED_SURFACE_SPLIT_COUNTS = {"train": 400, "val": 34, "test": 50}
EXPECTED_EXCLUDED_CASE_COUNT = 0
REQUIRED_RESTORED_CASE_IDS = frozenset(
    {
        "run_44",
        "run_133",
        "run_158",
        "run_184",
        "run_203",
        "run_226",
        "run_249",
        "run_310",
        "run_416",
        "run_484",
    }
)


@dataclass(frozen=True)
class DrivAerMLCase:
    case_id: str
    surface_x: torch.Tensor
    surface_y: torch.Tensor
    volume_x: torch.Tensor
    volume_y: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PointView:
    case_id: str
    view_index: int
    view_count: int
    surface_view_count: int
    volume_view_count: int
    sampling_mode: str


@dataclass
class SurfaceBatch:
    case_ids: list[str]
    surface_x: torch.Tensor
    surface_y: torch.Tensor
    surface_mask: torch.Tensor
    volume_x: torch.Tensor
    volume_y: torch.Tensor
    volume_mask: torch.Tensor
    metadata: list[dict[str, Any]]

    def to(self, device: torch.device | str) -> "SurfaceBatch":
        return SurfaceBatch(
            case_ids=list(self.case_ids),
            surface_x=self.surface_x.to(device),
            surface_y=self.surface_y.to(device),
            surface_mask=self.surface_mask.to(device),
            volume_x=self.volume_x.to(device),
            volume_y=self.volume_y.to(device),
            volume_mask=self.volume_mask.to(device),
            metadata=list(self.metadata),
        )

    @property
    def x(self) -> torch.Tensor:
        return self.surface_x

    @property
    def y(self) -> torch.Tensor:
        return self.surface_y

    @property
    def mask(self) -> torch.Tensor:
        return self.surface_mask


def read_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_manifest(manifest: dict, manifest_path: str | Path) -> None:
    surface_splits = manifest.get("surface_splits")
    if not isinstance(surface_splits, dict):
        raise ValueError(f"DrivAerML manifest {manifest_path} is missing surface_splits")

    missing_splits = sorted(set(EXPECTED_SURFACE_SPLIT_COUNTS) - set(surface_splits))
    if missing_splits:
        raise ValueError(f"DrivAerML manifest {manifest_path} is missing splits: {missing_splits}")

    actual_counts = {split: len(surface_splits[split]) for split in EXPECTED_SURFACE_SPLIT_COUNTS}
    if actual_counts != EXPECTED_SURFACE_SPLIT_COUNTS:
        raise ValueError(
            "DrivAerML manifest does not match the public processed split: "
            f"{actual_counts} vs {EXPECTED_SURFACE_SPLIT_COUNTS} ({manifest_path})"
        )

    split_sets = {split: set(surface_splits[split]) for split in EXPECTED_SURFACE_SPLIT_COUNTS}
    surface_case_ids = set().union(*split_sets.values())
    if len(surface_case_ids) != sum(actual_counts.values()):
        raise ValueError(f"DrivAerML manifest {manifest_path} has overlapping surface splits")

    excluded_count = int(manifest.get("excluded_case_count", len(manifest.get("excluded_case_ids", []))))
    if excluded_count != EXPECTED_EXCLUDED_CASE_COUNT:
        raise ValueError(
            "DrivAerML manifest still excludes repaired public cases: "
            f"{excluded_count} excluded in {manifest_path}"
        )

    missing_required = sorted(REQUIRED_RESTORED_CASE_IDS - surface_case_ids)
    if missing_required:
        raise ValueError(
            f"DrivAerML manifest {manifest_path} is missing restored public cases: {missing_required}"
        )


def _resolve_case_root(manifest: dict, override_root: str | Path | None = None) -> Path:
    if override_root is not None:
        root = Path(override_root)
        if not root.exists():
            raise FileNotFoundError(f"DrivAerML root does not exist: {root}")
        return root

    candidates = list(manifest.get("case_root_candidates", []))
    case_root = manifest.get("case_root")
    if case_root:
        candidates.append(case_root)
        if isinstance(case_root, str) and case_root.startswith("/mnt/pvc/"):
            candidates.append(case_root.replace("/mnt/pvc/", "/mnt/new-pvc/", 1))
        if isinstance(case_root, str) and case_root.startswith("/mnt/new-pvc/"):
            candidates.append(case_root.replace("/mnt/new-pvc/", "/mnt/pvc/", 1))
    candidates = expand_pvc_candidates(candidates)

    root = first_existing(candidates)
    if root is not None:
        return root
    raise FileNotFoundError(
        f"Could not resolve DrivAerML root from candidates: {manifest.get('case_root_candidates', [])}"
    )


def _case_dir(root: Path, case_id: str) -> Path:
    path = root / case_id
    if not path.exists():
        raise FileNotFoundError(f"DrivAerML case directory not found: {path}")
    return path


def _candidate_artifact_paths(path: Path) -> list[Path]:
    texts: list[str] = []
    seen: set[str] = set()

    def add(value: str | Path | None) -> None:
        if not value:
            return
        text = str(value)
        if text in seen:
            return
        seen.add(text)
        texts.append(text)

    add(path)
    add(path.resolve(strict=False))
    if str(path).startswith("/mnt/pvc/"):
        add(str(path).replace("/mnt/pvc/", "/mnt/new-pvc/", 1))
    if str(path).startswith("/mnt/new-pvc/"):
        add(str(path).replace("/mnt/new-pvc/", "/mnt/pvc/", 1))

    if path.is_symlink():
        raw_target = os.readlink(path)
        add(raw_target)
        target_path = Path(raw_target)
        if not target_path.is_absolute():
            target_path = (path.parent / target_path).resolve(strict=False)
        add(target_path)
        target_text = str(target_path)
        if target_text.startswith("/rsyncd-munged/"):
            stripped = target_text.removeprefix("/rsyncd-munged/").lstrip("/")
            add("/" + stripped)
            if stripped.startswith("mnt/pvc/"):
                add("/" + stripped.replace("mnt/pvc/", "mnt/new-pvc/", 1))
            if stripped.startswith("mnt/new-pvc/"):
                add("/" + stripped.replace("mnt/new-pvc/", "mnt/pvc/", 1))

    ordered: list[Path] = []
    for candidate in expand_pvc_candidates(texts):
        resolved = Path(candidate)
        if resolved not in ordered:
            ordered.append(resolved)
    return ordered


@functools.lru_cache(maxsize=16384)
def _resolve_artifact_path(path: Path) -> Path:
    for candidate in _candidate_artifact_paths(path):
        if candidate.exists():
            return candidate
    checked = [str(candidate) for candidate in _candidate_artifact_paths(path)]
    raise FileNotFoundError(f"Could not resolve DrivAerML artifact {path}; checked {checked}")


def _load_npy_rows(path: Path, rows: np.ndarray | None = None) -> np.ndarray:
    resolved = _resolve_artifact_path(path)
    if rows is None:
        return np.asarray(np.load(resolved), dtype=np.float32)
    mapped = np.load(resolved, mmap_mode="r")
    return np.asarray(mapped[rows], dtype=np.float32)


def _npy_row_count(path: Path) -> int:
    arr = np.load(_resolve_artifact_path(path), mmap_mode="r")
    if arr.ndim == 0:
        return 1
    return int(arr.shape[0])


def _column(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    return arr[:, None] if arr.ndim == 1 else arr


def _three_column(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {arr.shape}")
    return arr


def load_case(
    root: str | Path,
    case_id: str,
    *,
    surface_rows: np.ndarray | None = None,
    volume_rows: np.ndarray | None = None,
) -> DrivAerMLCase:
    case_dir = _case_dir(Path(root), case_id)
    xyz = _load_npy_rows(case_dir / "surface_xyz.npy", surface_rows)
    normals = _load_npy_rows(case_dir / "surface_normals.npy", surface_rows)
    area = _column(_load_npy_rows(case_dir / "surface_area.npy", surface_rows))
    cp = _column(_load_npy_rows(case_dir / "surface_cp.npy", surface_rows))
    wall_shear = _three_column(
        _load_npy_rows(case_dir / "surface_wallshearstress.npy", surface_rows),
        "surface_wallshearstress.npy",
    )
    surface_x = np.concatenate([xyz, normals, area], axis=1)
    surface_y = np.concatenate([cp, wall_shear], axis=1)
    volume_x = np.concatenate(
        [
            _load_npy_rows(case_dir / "volume_xyz.npy", volume_rows),
            _column(_load_npy_rows(case_dir / "volume_sdf.npy", volume_rows)),
        ],
        axis=1,
    )
    volume_y = _column(_load_npy_rows(case_dir / "volume_pressure.npy", volume_rows))
    return DrivAerMLCase(
        case_id=case_id,
        surface_x=torch.from_numpy(surface_x),
        surface_y=torch.from_numpy(surface_y),
        volume_x=torch.from_numpy(volume_x),
        volume_y=torch.from_numpy(volume_y),
        metadata={
            "case_id": case_id,
            "n_surface": int(surface_x.shape[0]),
            "n_volume": int(volume_x.shape[0]),
        },
    )


def load_case_point_counts(root: str | Path, case_id: str) -> dict[str, int]:
    case_dir = _case_dir(Path(root), case_id)
    return {
        "case_id": case_id,
        "n_surface": _npy_row_count(case_dir / "surface_xyz.npy"),
        "n_volume": _npy_row_count(case_dir / "volume_xyz.npy"),
    }


class DrivAerMLCaseStore:
    """Manifest-backed store for the processed DrivAerML PVC layout."""

    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST, root: str | Path | None = None):
        self.manifest_path = Path(manifest_path)
        self.manifest = read_json(self.manifest_path)
        validate_manifest(self.manifest, self.manifest_path)
        self.root = _resolve_case_root(self.manifest, override_root=root)
        self.normalizers_path = self.root / "normalizers.json"
        self._point_count_cache: dict[str, dict[str, int]] = {}

    def case_ids(self, split: str) -> list[str]:
        return list(self.manifest["surface_splits"][split])

    def load_case(
        self,
        case_id: str,
        *,
        surface_rows: np.ndarray | None = None,
        volume_rows: np.ndarray | None = None,
    ) -> DrivAerMLCase:
        return load_case(self.root, case_id, surface_rows=surface_rows, volume_rows=volume_rows)

    def case_point_counts(self, case_id: str) -> dict[str, int]:
        cached = self._point_count_cache.get(case_id)
        if cached is None:
            cached = load_case_point_counts(self.root, case_id)
            self._point_count_cache[case_id] = cached
        return dict(cached)

    def load_normalizers(self) -> dict:
        with self.normalizers_path.open() as f:
            return json.load(f)


class DrivAerMLSurfaceDataset(Dataset):
    """DrivAerML cases, optionally split into point-limited surface/volume views."""

    def __init__(
        self,
        case_ids: list[str],
        *,
        store: DrivAerMLCaseStore | None = None,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        root: str | Path | None = None,
        max_points: int | None = None,
        max_surface_points: int = 0,
        max_volume_points: int = 0,
        sampling_mode: str = "full",
    ):
        self.store = store or DrivAerMLCaseStore(manifest_path=manifest_path, root=root)
        self.case_ids = list(case_ids)
        if max_points is not None:
            max_surface_points = max_points
            max_volume_points = max_points
        self.max_surface_points = max_surface_points
        self.max_volume_points = max_volume_points
        self.sampling_mode = sampling_mode
        self.views = self._build_views()

    def __len__(self) -> int:
        return len(self.views)

    @staticmethod
    def _view_count(total: int, points_per_view: int) -> int:
        if points_per_view <= 0 or total <= points_per_view:
            return 1
        return max(1, math.ceil(total / points_per_view))

    def _build_views(self) -> list[PointView]:
        views: list[PointView] = []
        for case_id in self.case_ids:
            counts = self.store.case_point_counts(case_id)
            surface_views = self._view_count(counts["n_surface"], self.max_surface_points)
            volume_views = self._view_count(counts["n_volume"], self.max_volume_points)
            view_count = max(surface_views, volume_views)
            for view_index in range(view_count):
                views.append(
                    PointView(
                        case_id=case_id,
                        view_index=view_index,
                        view_count=view_count,
                        surface_view_count=surface_views,
                        volume_view_count=volume_views,
                        sampling_mode=self.sampling_mode,
                    )
                )
        return views

    def _indices(
        self,
        total: int,
        count: int,
        view: PointView,
        *,
        group_view_count: int,
    ) -> torch.Tensor | None:
        if view.view_index >= group_view_count:
            return torch.empty(0, dtype=torch.long)
        if count <= 0 or total <= count:
            return None if view.view_index == 0 else torch.empty(0, dtype=torch.long)
        if view.sampling_mode == "train_random":
            return torch.randint(total, (count,), dtype=torch.long).sort().values
        if view.sampling_mode == "eval_chunk":
            return torch.arange(view.view_index, total, group_view_count, dtype=torch.long)
        return None

    def __getitem__(self, idx: int) -> DrivAerMLCase:
        view = self.views[idx]
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
        metadata = dict(case.metadata)
        metadata["n_surface_full"] = int(counts["n_surface"])
        metadata["n_surface_loaded"] = int(case.surface_x.shape[0])
        metadata["surface_view_index"] = int(view.view_index)
        metadata["surface_view_count"] = int(view.surface_view_count)
        metadata["surface_sampling_mode"] = view.sampling_mode
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


def pad_collate(samples: list[DrivAerMLCase]) -> SurfaceBatch:
    if not samples:
        raise ValueError("pad_collate received an empty batch")
    max_surface_n = max(sample.surface_x.shape[0] for sample in samples)
    max_volume_n = max(sample.volume_x.shape[0] for sample in samples)
    batch_size = len(samples)
    surface_x = torch.zeros(batch_size, max_surface_n, SURFACE_X_DIM, dtype=samples[0].surface_x.dtype)
    surface_y = torch.zeros(batch_size, max_surface_n, SURFACE_Y_DIM, dtype=samples[0].surface_y.dtype)
    surface_mask = torch.zeros(batch_size, max_surface_n, dtype=torch.bool)
    volume_x = torch.zeros(batch_size, max_volume_n, VOLUME_X_DIM, dtype=samples[0].volume_x.dtype)
    volume_y = torch.zeros(batch_size, max_volume_n, VOLUME_Y_DIM, dtype=samples[0].volume_y.dtype)
    volume_mask = torch.zeros(batch_size, max_volume_n, dtype=torch.bool)
    for i, sample in enumerate(samples):
        surface_n = sample.surface_x.shape[0]
        volume_n = sample.volume_x.shape[0]
        surface_x[i, :surface_n] = sample.surface_x
        surface_y[i, :surface_n] = sample.surface_y
        surface_mask[i, :surface_n] = True
        volume_x[i, :volume_n] = sample.volume_x
        volume_y[i, :volume_n] = sample.volume_y
        volume_mask[i, :volume_n] = True
    return SurfaceBatch(
        case_ids=[sample.case_id for sample in samples],
        surface_x=surface_x,
        surface_y=surface_y,
        surface_mask=surface_mask,
        volume_x=volume_x,
        volume_y=volume_y,
        volume_mask=volume_mask,
        metadata=[dict(sample.metadata) for sample in samples],
    )


def _normalizer_tensor(raw: dict, name: str, expected_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    stats = raw.get(name)
    if not isinstance(stats, dict):
        raise ValueError(f"{name!r} is missing from normalizers.json")
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(stats["std"], dtype=torch.float32).reshape(-1)
    if mean.numel() == 1 and expected_dim > 1:
        mean = mean.repeat(expected_dim)
    if std.numel() == 1 and expected_dim > 1:
        std = std.repeat(expected_dim)
    if mean.numel() != expected_dim or std.numel() != expected_dim:
        raise ValueError(
            f"normalizers.json entry {name!r} must have {expected_dim} values, "
            f"got mean={mean.numel()} std={std.numel()}"
        )
    return mean, std


def target_stats_from_normalizers(store: DrivAerMLCaseStore) -> dict[str, torch.Tensor]:
    """Read target stats for all required prediction fields."""

    if not store.normalizers_path.exists():
        raise FileNotFoundError(f"Missing DrivAerML normalizers file: {store.normalizers_path}")
    raw = store.load_normalizers()
    surface_cp_mean, surface_cp_std = _normalizer_tensor(raw, "surface_cp", 1)
    wall_shear_mean, wall_shear_std = _normalizer_tensor(raw, "surface_wallshearstress", 3)
    volume_pressure_mean, volume_pressure_std = _normalizer_tensor(raw, "volume_pressure", 1)
    return {
        "surface_y_mean": torch.cat([surface_cp_mean, wall_shear_mean]),
        "surface_y_std": torch.cat([surface_cp_std, wall_shear_std]),
        "volume_y_mean": volume_pressure_mean,
        "volume_y_std": volume_pressure_std,
    }


def load_data(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    root: str | Path | None = None,
    *,
    train_surface_points: int = 40_000,
    eval_surface_points: int = 40_000,
    train_volume_points: int = 40_000,
    eval_volume_points: int = 40_000,
    debug: bool = False,
) -> tuple[DrivAerMLSurfaceDataset, dict[str, DrivAerMLSurfaceDataset], dict[str, DrivAerMLSurfaceDataset], dict[str, torch.Tensor]]:
    """Return train, validation, test datasets and target normalization stats."""

    store = DrivAerMLCaseStore(manifest_path=manifest_path, root=root)
    train_ids = store.case_ids("train")
    val_ids = store.case_ids("val")
    test_ids = store.case_ids("test")
    if debug:
        train_ids = train_ids[:4]
        val_ids = val_ids[:2]
        test_ids = test_ids[:2]
        train_surface_points = min(train_surface_points, 8_192)
        eval_surface_points = min(eval_surface_points, 8_192)
        train_volume_points = min(train_volume_points, 8_192)
        eval_volume_points = min(eval_volume_points, 8_192)

    train_sampling = "train_random" if train_surface_points > 0 or train_volume_points > 0 else "full"
    eval_sampling = "eval_chunk" if eval_surface_points > 0 or eval_volume_points > 0 else "full"
    train_ds = DrivAerMLSurfaceDataset(
        train_ids,
        store=store,
        max_surface_points=train_surface_points,
        max_volume_points=train_volume_points,
        sampling_mode=train_sampling,
    )
    val_splits = {
        "val_surface": DrivAerMLSurfaceDataset(
            val_ids,
            store=store,
            max_surface_points=eval_surface_points,
            max_volume_points=eval_volume_points,
            sampling_mode=eval_sampling,
        )
    }
    test_splits = {
        "test_surface": DrivAerMLSurfaceDataset(
            test_ids,
            store=store,
            max_surface_points=eval_surface_points,
            max_volume_points=eval_volume_points,
            sampling_mode=eval_sampling,
        )
    }
    stats = target_stats_from_normalizers(store)
    print(
        f"DrivAerML: train={len(train_ds)} views/{len(train_ids)} cases, "
        f"val={len(val_splits['val_surface'])} views/{len(val_ids)} cases, "
        f"test={len(test_splits['test_surface'])} views/{len(test_ids)} cases"
    )
    return train_ds, val_splits, test_splits, stats
