from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import DrivAerMLCase, DrivAerMLSurfaceDataset, SurfaceBatch  # noqa: E402
from train import (  # noqa: E402
    MetricSlopeTracker,
    SurfaceTransolver,
    TargetTransform,
    collect_gradient_metrics,
    evaluate_split,
)


class FakeStore:
    def __init__(self):
        surface = torch.arange(10, dtype=torch.float32).unsqueeze(-1).repeat(1, 7)
        surface_target = torch.cat(
            [
                torch.arange(10, dtype=torch.float32).unsqueeze(-1),
                torch.ones(10, 3, dtype=torch.float32),
            ],
            dim=1,
        )
        volume = torch.arange(12, dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
        volume_target = torch.arange(12, dtype=torch.float32).unsqueeze(-1)
        self.calls: list[dict[str, object]] = []
        self.case = DrivAerMLCase(
            case_id="case-a",
            surface_x=surface,
            surface_y=surface_target,
            volume_x=volume,
            volume_y=volume_target,
            metadata={},
        )

    def case_point_counts(self, case_id: str) -> dict[str, int]:
        assert case_id == "case-a"
        return {
            "case_id": case_id,
            "n_surface": int(self.case.surface_x.shape[0]),
            "n_volume": int(self.case.volume_x.shape[0]),
        }

    def load_case(self, case_id: str, *, surface_rows=None, volume_rows=None) -> DrivAerMLCase:
        assert case_id == "case-a"
        surface_rows_tensor = None if surface_rows is None else torch.as_tensor(surface_rows)
        volume_rows_tensor = None if volume_rows is None else torch.as_tensor(volume_rows)
        self.calls.append(
            {
                "case_id": case_id,
                "surface_rows": None if surface_rows_tensor is None else surface_rows_tensor.tolist(),
                "volume_rows": None if volume_rows_tensor is None else volume_rows_tensor.tolist(),
            }
        )
        return DrivAerMLCase(
            case_id=self.case.case_id,
            surface_x=self.case.surface_x
            if surface_rows_tensor is None
            else self.case.surface_x.index_select(0, surface_rows_tensor),
            surface_y=self.case.surface_y
            if surface_rows_tensor is None
            else self.case.surface_y.index_select(0, surface_rows_tensor),
            volume_x=self.case.volume_x
            if volume_rows_tensor is None
            else self.case.volume_x.index_select(0, volume_rows_tensor),
            volume_y=self.case.volume_y
            if volume_rows_tensor is None
            else self.case.volume_y.index_select(0, volume_rows_tensor),
            metadata={},
        )


class IdentityModel(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, *, surface_x=None, surface_mask=None, volume_x=None, volume_mask=None, x=None, mask=None):
        del surface_mask, volume_mask, mask
        if surface_x is None:
            surface_x = x
        return {
            "preds": surface_x[..., :4],
            "surface_preds": surface_x[..., :4],
            "volume_preds": volume_x[..., :1],
        }


def test_eval_chunk_covers_every_surface_and_volume_point_once():
    store = FakeStore()
    dataset = DrivAerMLSurfaceDataset(
        ["case-a"],
        store=store,
        max_points=4,
        sampling_mode="eval_chunk",
    )

    assert len(dataset) == 3
    seen_surface: list[int] = []
    seen_volume: list[int] = []
    for sample in dataset:
        assert sample.surface_x.shape[0] <= 4
        assert sample.volume_x.shape[0] <= 4
        assert sample.metadata["surface_view_count"] == 3
        assert sample.metadata["volume_view_count"] == 3
        seen_surface.extend(int(v) for v in sample.surface_y[:, 0].tolist())
        seen_volume.extend(int(v) for v in sample.volume_y[:, 0].tolist())

    assert sorted(seen_surface) == list(range(10))
    assert sorted(seen_volume) == list(range(12))
    assert store.calls[0]["surface_rows"] == [0, 3, 6, 9]
    assert store.calls[0]["volume_rows"] == [0, 3, 6, 9]


def test_eval_chunk_does_not_repeat_smaller_modality_when_view_counts_differ():
    store = FakeStore()
    dataset = DrivAerMLSurfaceDataset(
        ["case-a"],
        store=store,
        max_surface_points=20,
        max_volume_points=4,
        sampling_mode="eval_chunk",
    )

    assert len(dataset) == 3
    seen_surface: list[int] = []
    seen_volume: list[int] = []
    for sample in dataset:
        seen_surface.extend(int(v) for v in sample.surface_y[:, 0].tolist())
        seen_volume.extend(int(v) for v in sample.volume_y[:, 0].tolist())

    assert sorted(seen_surface) == list(range(10))
    assert sorted(seen_volume) == list(range(12))
    assert store.calls[0]["surface_rows"] is None
    assert store.calls[1]["surface_rows"] == []
    assert store.calls[2]["surface_rows"] == []


def test_train_random_repeats_case_enough_times():
    store = FakeStore()
    torch.manual_seed(0)
    dataset = DrivAerMLSurfaceDataset(
        ["case-a"],
        store=store,
        max_points=4,
        sampling_mode="train_random",
    )

    assert len(dataset) == 3
    total_surface_loaded = 0
    total_volume_loaded = 0
    for sample in dataset:
        assert sample.surface_x.shape[0] == 4
        assert sample.volume_x.shape[0] == 4
        assert sample.metadata["surface_sampling_mode"] == "train_random"
        assert sample.metadata["volume_sampling_mode"] == "train_random"
        total_surface_loaded += sample.surface_x.shape[0]
        total_volume_loaded += sample.volume_x.shape[0]

    assert total_surface_loaded >= 10
    assert total_volume_loaded >= 12
    assert all(call["surface_rows"] is not None for call in store.calls)
    assert all(call["volume_rows"] is not None for call in store.calls)


def test_chunked_eval_reaggregates_per_case_relative_l2_for_surface_and_volume():
    def batch(
        case_id: str,
        preds: list[float],
        targets: list[float],
    ) -> SurfaceBatch:
        pred_tensor = torch.tensor(preds, dtype=torch.float32).view(1, -1, 1)
        target_tensor = torch.tensor(targets, dtype=torch.float32).view(1, -1, 1)
        surface_x = torch.zeros(1, len(preds), 7)
        surface_x[..., 0:1] = pred_tensor
        surface_x[..., 1:4] = 1.0
        surface_y = torch.cat([target_tensor, torch.ones(1, len(preds), 3)], dim=-1)
        volume_x = torch.zeros(1, len(preds), 4)
        volume_x[..., 0:1] = pred_tensor
        volume_y = target_tensor
        mask = torch.ones(1, len(preds), dtype=torch.bool)
        return SurfaceBatch(
            case_ids=[case_id],
            surface_x=surface_x,
            surface_y=surface_y,
            surface_mask=mask,
            volume_x=volume_x,
            volume_y=volume_y,
            volume_mask=mask,
            metadata=[],
        )

    loader = [
        batch("case-a", [1.0, 2.0], [1.0, 2.0]),
        batch("case-a", [3.0, 5.0], [3.0, 4.0]),
        batch("case-b", [4.0], [2.0]),
        batch("case-b", [2.0], [2.0]),
    ]
    metrics = evaluate_split(
        IdentityModel(),
        loader,
        TargetTransform(
            surface_y_mean=torch.zeros(4),
            surface_y_std=torch.ones(4),
            volume_y_mean=torch.zeros(1),
            volume_y_std=torch.ones(1),
        ),
        torch.device("cpu"),
    )

    case_a = math.sqrt(1.0 / 30.0)
    case_b = math.sqrt(4.0 / 8.0)
    expected = (case_a + case_b) / 2.0
    assert metrics["surface_rel_l2"] == expected
    assert metrics["surface_pressure_rel_l2"] == expected
    assert metrics["volume_pressure_rel_l2"] == expected
    assert metrics["wall_shear_rel_l2"] == 0.0
    assert metrics["surface_pressure_mae"] == 0.5
    assert metrics["volume_pressure_mae"] == 0.5
    assert metrics["wall_shear_mae"] == 0.0


def test_slope_tracker_logs_five_percent_update_slope():
    tracker = MetricSlopeTracker(total_steps=100, fraction=0.05)

    assert tracker.update(global_step=1, metrics={"train/loss": 10.0}, namespace="train") == {}
    slopes = tracker.update(global_step=5, metrics={"train/loss": 6.0}, namespace="train")

    assert slopes["train/slope/loss/per_step"] == -1.0
    assert slopes["train/slope/loss/per_1k_steps"] == -1000.0


def test_gradient_telemetry_exposes_aggregate_layer_type_module_and_param_keys():
    model = SurfaceTransolver(
        n_layers=1,
        n_hidden=12,
        n_head=3,
        slice_num=4,
        mlp_ratio=1,
    )
    x = torch.randn(2, 5, 7)
    mask = torch.ones(2, 5, dtype=torch.bool)
    loss = model(x=x, mask=mask)["preds"].square().mean()
    loss.backward()

    metrics = collect_gradient_metrics(model, log_histograms=False)

    assert "train/grad/global_norm" in metrics
    assert "train/grad/grad_to_param_norm" in metrics
    assert any(key.startswith("train/grad_type/LinearProjection/") for key in metrics)
    assert any(key.startswith("train/grad_module/TransolverAttention/") for key in metrics)
    assert any(key.startswith("train/grad_param/LinearProjection/") for key in metrics)
