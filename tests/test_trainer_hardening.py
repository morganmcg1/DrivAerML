import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("wandb")

import train


def test_parser_operational_defaults():
    config = train.parse_args([])
    assert config.validation_every == 1
    assert config.gradient_log_every == 250
    assert config.weight_log_every == 250
    assert config.log_gradient_histograms is False
    assert config.grad_clip_norm == 1.0
    assert config.lr_warmup_epochs == 0
    assert config.lr_cosine_t_max == 0
    assert config.lr_min == 1e-6
    assert config.eval_raw_vs_ema is False


def test_masked_losses_are_finite_for_empty_masks():
    pred = torch.ones(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    mse = train.masked_mse(pred, target, mask)
    rel_l2 = train.squared_relative_l2_loss(pred, target, mask)
    assert torch.isfinite(mse)
    assert torch.isfinite(rel_l2)
    assert mse.item() == 0.0
    assert rel_l2.item() == 0.0


def test_masked_mse_uses_only_valid_tokens():
    pred = torch.tensor([[[1.0], [3.0], [100.0]]])
    target = torch.tensor([[[0.0], [1.0], [0.0]]])
    mask = torch.tensor([[True, True, False]])
    assert train.masked_mse(pred, target, mask).item() == pytest.approx(2.5)


def test_final_metric_contract_rejects_missing_or_nonfinite_values():
    log = {f"test_primary/{key}": 1.0 for key in train.PRIMARY_METRIC_KEYS}
    train.assert_required_finite_metrics(log, "test_primary")
    log["test_primary/volume_pressure_rel_l2_pct"] = math.nan
    with pytest.raises(RuntimeError):
        train.assert_required_finite_metrics(log, "test_primary")
    del log["test_primary/volume_pressure_rel_l2_pct"]
    with pytest.raises(RuntimeError):
        train.assert_required_finite_metrics(log, "test_primary")


def test_scheduler_supports_warmup_and_cosine():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = train.Config(lr_warmup_epochs=2, lr_cosine_t_max=5, lr_min=1e-5)
    scheduler = train.build_lr_scheduler(optimizer, config, max_epochs=10)
    assert scheduler.get_last_lr()[0] == pytest.approx(5e-5)
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[0] > 5e-5


def test_best_checkpoint_guard_requires_finite_positive_improvement():
    assert train.should_update_best_checkpoint(1.0, 2.0)
    assert not train.should_update_best_checkpoint(3.0, 2.0)
    assert not train.should_update_best_checkpoint(0.0, 2.0)
    assert not train.should_update_best_checkpoint(math.nan, 2.0)
    assert not train.should_update_best_checkpoint(math.inf, 2.0)


def test_timeout_budget_keeps_harvest_reserve_for_tiny_smokes():
    total, reserve, train_budget = train.timeout_budget_minutes(
        {
            "SENPAI_TIMEOUT_MINUTES": "0.25",
            "SENPAI_VAL_BUDGET_MINUTES": "0.1",
        }
    )
    assert total == pytest.approx(0.25)
    assert reserve == pytest.approx(0.1)
    assert train_budget == pytest.approx(0.15)

    _, _, train_budget = train.timeout_budget_minutes(
        {
            "SENPAI_TIMEOUT_MINUTES": "0.1",
            "SENPAI_VAL_BUDGET_MINUTES": "0.2",
        }
    )
    assert train_budget == 0.0


def test_ddp_wandb_names_use_shared_group_and_rank_suffix():
    state = train.DistributedState(
        enabled=True,
        rank=3,
        local_rank=3,
        world_size=8,
        device=torch.device("cpu"),
    )
    config = train.Config(wandb_name="baseline", agent="student-a")
    assert train.run_name_for_rank(config, state) == "baseline-rank3"
    assert train.wandb_group_for_rank(config, state) == "baseline"

    config = train.Config(agent="student-a")
    assert train.run_name_for_rank(config, state) == "student-a-rank3"
    assert train.wandb_group_for_rank(config, state) == "student-a"


def test_strided_distributed_sampler_partitions_without_duplicates():
    dataset = list(range(10))
    shards = [
        list(train.StridedDistributedSampler(dataset, num_replicas=3, rank=rank))
        for rank in range(3)
    ]
    flattened = [item for shard in shards for item in shard]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))
    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
