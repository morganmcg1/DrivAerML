"""End-to-end data loader smoke test for curvature integration.

Loads one case via ``CurvatureAugmentedCaseStore`` (using a dynamically-computed
partial-stats file derived from currently-cached cases) and runs it through the
``curvature_pad_collate`` + ``SurfaceTransolver`` forward path with the
``--use-curvature-attention-bias`` setting. Verifies the curvature tensor
reaches the model and zero-init still produces baseline-identical output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import DEFAULT_MANIFEST  # noqa: E402
from model import SurfaceTransolver  # noqa: E402
from trainer_runtime import (  # noqa: E402
    CURVATURE_CACHE_FILENAME,
    CURVATURE_DIM,
    CurvatureAugmentedCaseStore,
    curvature_pad_collate,
)


def main() -> None:
    data_root = Path("/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511")
    cached_cases = sorted({p.parent.name for p in data_root.glob("*/surface_curvature_proxy_k16_v1.npy")})
    if not cached_cases:
        raise SystemExit("No cached curvature cases available — wait for precompute")
    print(f"Cached cases available: {len(cached_cases)}; using first 3 for stats build")

    # Build partial stats from the first 3 cached cases.
    sum_x = np.zeros(CURVATURE_DIM, dtype=np.float64)
    sum_x2 = np.zeros(CURVATURE_DIM, dtype=np.float64)
    n_total = np.zeros((), dtype=np.float64)
    for case_id in cached_cases[:3]:
        curv = np.load(data_root / case_id / CURVATURE_CACHE_FILENAME)
        sum_x += curv.sum(axis=0, dtype=np.float64)
        sum_x2 += (curv.astype(np.float64) ** 2).sum(axis=0)
        n_total += curv.shape[0]
    mean = (sum_x / n_total).astype(np.float64)
    var = (sum_x2 / n_total) - mean * mean
    std = np.sqrt(np.clip(var, 0.0, None))
    std = np.maximum(std, 1e-6)
    stats = {
        "version": "k16_v1",
        "k": 16,
        "channels": ["kappa_H", "kappa_G", "kappa_mag"],
        "n_train_points": int(n_total),
        "n_train_cases": 3,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    print(f"Partial stats: mean={mean}, std={std}")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(stats, tf)
        stats_path = Path(tf.name)

    # Build the store with partial stats.
    store = CurvatureAugmentedCaseStore(
        manifest_path=DEFAULT_MANIFEST,
        root=data_root,
        stats=stats,
    )

    # Load one cached case with a small surface row subsample (to avoid OOM).
    case_id = cached_cases[0]
    rng = np.random.default_rng(0)
    n_surface = np.load(data_root / case_id / "surface_xyz.npy", mmap_mode="r").shape[0]
    n_vol = np.load(data_root / case_id / "volume_xyz.npy", mmap_mode="r").shape[0]
    surface_rows = np.sort(rng.choice(n_surface, size=8192, replace=False)).astype(np.int64)
    volume_rows = np.sort(rng.choice(n_vol, size=8192, replace=False)).astype(np.int64)

    case = store.load_case(case_id, surface_rows=surface_rows, volume_rows=volume_rows)
    print(f"Loaded case {case_id}: surface={case.surface_x.shape}, volume={case.volume_x.shape}")
    curv = case.metadata["surface_curvature"]
    print(f"  curvature: shape={tuple(curv.shape)}, dtype={curv.dtype}")
    print(f"  curvature stats: mean={curv.mean(dim=0).tolist()}, std={curv.std(dim=0).tolist()}")
    assert curv.shape == (case.surface_x.shape[0], 3), "Curvature shape mismatch"

    # Collate single sample (test pad path with B=2 of the same case for shape).
    case2 = store.load_case(case_id, surface_rows=surface_rows[:7000], volume_rows=volume_rows[:5000])
    batch = curvature_pad_collate([case, case2])
    print(f"Batch surface_x: {tuple(batch.surface_x.shape)}, surface_mask sum={batch.surface_mask.sum().item()}")
    print(f"Batch surface_curvature: {tuple(batch.surface_curvature.shape)}, "
          f"dtype={batch.surface_curvature.dtype}")
    assert hasattr(batch, "surface_curvature")
    assert batch.surface_curvature.shape[0] == 2
    assert batch.surface_curvature.shape[1] == batch.surface_x.shape[1]
    assert batch.surface_curvature.shape[2] == 3
    # Padded rows for case2 should be zero curvature.
    n_case2 = case2.surface_x.shape[0]
    pad_curv = batch.surface_curvature[1, n_case2:]
    assert pad_curv.abs().max().item() == 0.0, "Padded curvature should be zero"
    print(f"  pad rows zero: OK (n={pad_curv.shape[0]} pad rows)")

    # Move to device and forward through both models (with/without curvature bias)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SurfaceBatch.to() returns a new instance without dynamic attributes, so
    # we follow the same pattern as train_loss/accumulate_eval_batch.
    surface_curvature_cpu = batch.surface_curvature
    batch = batch.to(device)
    surface_curvature = surface_curvature_cpu.to(device)
    setattr(batch, "surface_curvature", surface_curvature)

    torch.manual_seed(123)
    common_kwargs = dict(
        space_dim=3,
        surface_input_dim=7,
        volume_input_dim=4,
        n_hidden=64,
        n_layers=2,
        n_head=4,
        slice_num=16,
        surface_output_dim=4,
        volume_output_dim=1,
    )
    m_off = SurfaceTransolver(**common_kwargs, use_curvature_attention_bias=False).to(device).eval()
    m_on = SurfaceTransolver(**common_kwargs, use_curvature_attention_bias=True).to(device).eval()
    # Copy shared params
    state_off = m_off.state_dict()
    state_on = m_on.state_dict()
    for k in state_on:
        if k in state_off:
            state_on[k] = state_off[k].clone()
    m_on.load_state_dict(state_on, strict=True)

    with torch.no_grad():
        out_off = m_off(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
        out_on = m_on(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
            surface_curvature=surface_curvature,
        )
    diff_s = (out_off["surface_preds"] - out_on["surface_preds"]).abs().max().item()
    diff_v = (out_off["volume_preds"] - out_on["volume_preds"]).abs().max().item()
    print(f"end-to-end zero-init surface_preds diff: {diff_s:.3e}")
    print(f"end-to-end zero-init volume_preds  diff: {diff_v:.3e}")
    assert diff_s == 0.0, "End-to-end zero-init identity violated for surface"
    assert diff_v == 0.0, "End-to-end zero-init identity violated for volume"
    print("OK: end-to-end data loader + zero-init forward identity verified.")

    # Cleanup
    stats_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
