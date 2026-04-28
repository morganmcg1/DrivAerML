from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import DrivAerMLCase, DrivAerMLSurfaceDataset, SurfaceBatch  # noqa: E402
from train import TargetTransform, evaluate_split  # noqa: E402


class FakeStore:
    def __init__(self):
        surface = torch.arange(10, dtype=torch.float32).unsqueeze(-1).repeat(1, 7)
        target = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
        self.calls: list[dict[str, object]] = []
        self.case = DrivAerMLCase(
            case_id="case-a",
            surface_x=surface,
            surface_y=target,
            metadata={},
        )

    def case_point_counts(self, case_id: str) -> dict[str, int]:
        assert case_id == "case-a"
        return {"case_id": case_id, "n_surface": int(self.case.surface_x.shape[0])}

    def load_case(self, case_id: str, *, surface_rows=None) -> DrivAerMLCase:
        assert case_id == "case-a"
        rows = None if surface_rows is None else torch.as_tensor(surface_rows)
        self.calls.append({"case_id": case_id, "surface_rows": None if rows is None else rows.tolist()})
        return DrivAerMLCase(
            case_id=self.case.case_id,
            surface_x=self.case.surface_x if rows is None else self.case.surface_x.index_select(0, rows),
            surface_y=self.case.surface_y if rows is None else self.case.surface_y.index_select(0, rows),
            metadata={},
        )


class IdentityModel(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, *, x, mask):
        del mask
        return {"preds": x[..., :1]}


def test_eval_chunk_covers_every_surface_point_once():
    store = FakeStore()
    dataset = DrivAerMLSurfaceDataset(
        ["case-a"],
        store=store,
        max_points=4,
        sampling_mode="eval_chunk",
    )

    assert len(dataset) == 3
    seen: list[int] = []
    for sample in dataset:
        assert sample.surface_x.shape[0] <= 4
        assert sample.metadata["surface_view_count"] == 3
        seen.extend(int(v) for v in sample.surface_y[:, 0].tolist())

    assert sorted(seen) == list(range(10))
    assert store.calls[0]["surface_rows"] == [0, 3, 6, 9]


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
    total_loaded = 0
    for sample in dataset:
        assert sample.surface_x.shape[0] == 4
        assert sample.metadata["surface_sampling_mode"] == "train_random"
        total_loaded += sample.surface_x.shape[0]

    assert total_loaded >= 10
    assert all(call["surface_rows"] is not None for call in store.calls)


def test_chunked_eval_reaggregates_per_case_relative_l2():
    def batch(case_id: str, preds: list[float], targets: list[float]) -> SurfaceBatch:
        pred_tensor = torch.tensor(preds, dtype=torch.float32).view(1, -1, 1)
        target_tensor = torch.tensor(targets, dtype=torch.float32).view(1, -1, 1)
        mask = torch.ones(1, len(preds), dtype=torch.bool)
        return SurfaceBatch(
            case_ids=[case_id],
            x=pred_tensor.repeat(1, 1, 7),
            y=target_tensor,
            mask=mask,
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
        TargetTransform(y_mean=torch.tensor([0.0]), y_std=torch.tensor([1.0])),
        torch.device("cpu"),
    )

    case_a = math.sqrt(1.0 / 30.0)
    case_b = math.sqrt(4.0 / 8.0)
    expected = (case_a + case_b) / 2.0
    assert metrics["surface_rel_l2"] == expected
    assert metrics["surface_rel_l2_pct"] == expected * 100.0
