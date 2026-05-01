# DrivAerML Baseline Metrics

## Current Best: PR #74 — alphonse 4L/256d Fourier PE baseline (2026-05-01)

The current best result on the bengio branch is from PR #74 (alphonse), 4L/256d Transformer
with Fourier Positional Encoding, no-EMA, cosine LR with T_max=30. Val metrics at best checkpoint
(ep30, step 552,326, W&B run `m9775k1v`).

Note: These are **validation** metrics — test_primary eval from the ep30 checkpoint is pending.

**IMPORTANT val/test gap**: A systematic ~2x degradation on vol_p has been observed across experiments (e.g., val=4.17% → test~8-12%). Do not claim AB-UPT wins based solely on val metrics — test_primary confirmation required for all axis metrics before submission.

| Metric | Current Best (val) | AB-UPT Target |
|--------|-------------------|--------------|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **7.2091** | 4.51 |
| `val_primary/surface_pressure_rel_l2_pct` | 4.802 | 3.82 |
| `val_primary/wall_shear_rel_l2_pct` | 8.160 | 7.29 |
| `val_primary/volume_pressure_rel_l2_pct` | **4.166** ✓ | 6.08 |
| `val_primary/wall_shear_x_rel_l2_pct` | 7.109 | 5.35 |
| `val_primary/wall_shear_y_rel_l2_pct` | 9.100 | 3.65 |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.869 | 3.63 |

**vol_p beats AB-UPT target.** abupt is 7.21% vs 4.51% target — 2.7pp gap remains.

### Reproduce command (best config)

```bash
cd target/ && python train.py \
  --model-num-layers 4 \
  --model-hidden-dim 256 \
  --no-use-ema \
  --lr-cosine-t-max 30 \
  --fourier-pe \
  --no-compile-model \
  --nproc_per_node 4 \
  --wandb_group bengio-wave2
```

## AB-UPT Public Reference Targets

| Metric | AB-UPT Target |
|--------|--------------|
| `test_primary/surface_pressure_rel_l2_pct` | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | 3.63 |
| `test_primary/abupt_axis_mean_rel_l2_pct` | ~4.51 (mean of 5 axis metrics) |

## Historical Reference: radford branch (PR #2593)

Approximately 12.96% abupt — the bengio Wave 1 result (7.21%) is a significant improvement.

## Update Log

- 2026-04-30: Branch initialized. No experiments merged yet. AB-UPT reference values are the targets.
- 2026-05-01: PR #74 (alphonse) merged as Wave 1 leader. New best val_abupt = 7.2091% (ep30, run m9775k1v). vol_p beats AB-UPT target at 4.166%.
