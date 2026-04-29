# DrivAerML Baseline

**Branch:** `yi` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml`

## Status: first baseline merged 2026-04-29 — PR #11 (kohaku, tangential wall-shear projection)

Tangential wall-shear projection loss merged from PR #11. Run `uy0ds6iz`
(state=finished, 1 epoch reached before pre-fix timeout). All `test_primary/*`
metrics non-NaN. Subsequent PRs must beat these numbers.

## Reference baseline targets (must beat — AB-UPT public reference)

| Target | This-repo metric | AB-UPT |
|---|---|---:|
| Surface pressure `p_s` | `test_primary/surface_pressure_rel_l2_pct` | **3.82** |
| Vector wall shear `tau` | `test_primary/wall_shear_rel_l2_pct` | **7.29** |
| Volume pressure `p_v` | `test_primary/volume_pressure_rel_l2_pct` | **6.08** |
| Wall shear `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **5.35** |
| Wall shear `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **3.65** |
| Wall shear `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **3.63** |

Lower is better. Final claims must come from `test_primary/*` after best-validation
checkpoint reload.

## Current best on `yi`

| Metric | Best | PR | W&B run | Date |
|---|---:|---|---|---|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **35.1239** | #11 | uy0ds6iz | 2026-04-29 |
| `test_primary/surface_pressure_rel_l2_pct` | **10.0667** | #11 | uy0ds6iz | 2026-04-29 |
| `test_primary/wall_shear_rel_l2_pct` | **43.0515** | #11 | uy0ds6iz | 2026-04-29 |
| `test_primary/volume_pressure_rel_l2_pct` | **14.9879** | #11 | uy0ds6iz | 2026-04-29 |
| `test_primary/wall_shear_x_rel_l2_pct` | **30.8498** | #11 | uy0ds6iz | 2026-04-29 |
| `test_primary/wall_shear_y_rel_l2_pct` | **42.0609** | #11 | uy0ds6iz | 2026-04-29 |
| `test_primary/wall_shear_z_rel_l2_pct` | **77.6541** | #11 | uy0ds6iz | 2026-04-29 |

**Reproduce (PR #11 merged config):**

```bash
cd target/
python train.py \
  --use-tangential-wallshear-loss \
  --lr 2e-4 --weight-decay 5e-4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 256 --model-heads 4 --model-slices 128 \
  --ema-decay 0.9995 \
  --gradient-log-every 100 --weight-log-every 100 --no-log-gradient-histograms \
  --validation-every 1
```

**Distance from AB-UPT targets (multiple of target):**

| Metric | yi best | AB-UPT | Ratio |
|---|---:|---:|---:|
| surface_pressure | 10.07 | 3.82 | 2.6× |
| wall_shear | 43.05 | 7.29 | 5.9× |
| volume_pressure | 14.99 | 6.08 | 2.5× |
| wall_shear_x | 30.85 | 5.35 | 5.8× |
| wall_shear_y | 42.06 | 3.65 | 11.5× |
| wall_shear_z | 77.65 | 3.63 | 21.4× |

The wall-shear axes (especially `tau_z`) are the largest gap to close. PR #11
ran only 1 epoch — the per-step timeout fix on `yi` (commit `af92e9a`) plus
`--validation-every 1` should let subsequent runs reach 4–5 epochs and likely
crush these numbers further.

## Reference config (`train.py` defaults on `yi`)

```
lr=3e-4  weight_decay=1e-4  batch_size=2  epochs=50
train_/eval_ surface_points=40_000  train_/eval_ volume_points=40_000
model: 3 layers · 192 hidden · 3 heads · 96 slices · mlp_ratio=4
amp=bf16  ema_decay=0.999  validation_every=10
```
