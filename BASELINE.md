# DrivAerML Baseline

**Branch:** `yi` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml`

## Status: clean slate — no completed runs yet

We have no `test_primary/*` numbers on this project. The first wave will calibrate
both the reference defaults and a stronger known-good config, then layer single-delta
experiments on top.

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

_Nothing merged yet. All metrics N/A._

| Metric | Best | PR | W&B run | Date |
|---|---:|---|---|---|
| `test_primary/abupt_axis_mean_rel_l2_pct` | — | — | — | — |
| `test_primary/surface_pressure_rel_l2_pct` | — | — | — | — |
| `test_primary/wall_shear_rel_l2_pct` | — | — | — | — |
| `test_primary/volume_pressure_rel_l2_pct` | — | — | — | — |

## Reference config (`train.py` defaults on `yi`)

```
lr=3e-4  weight_decay=1e-4  batch_size=2  epochs=50
train_/eval_ surface_points=40_000  train_/eval_ volume_points=40_000
model: 3 layers · 192 hidden · 3 heads · 96 slices · mlp_ratio=4
amp=bf16  ema_decay=0.999  validation_every=10
```
