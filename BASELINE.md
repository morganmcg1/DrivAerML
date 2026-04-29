# DrivAerML Baseline

**Branch:** `yi` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml`

## Status: gilbert PR #9 wins — new baseline 2026-04-29 03:57 UTC

PR #9 (gilbert, vol_w=2.0 + protocol fixes) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 35.12 (kohaku PR #11) to
**17.39** — a 50.5% improvement on the headline metric. Wall-shear axes saw
~50–70% reductions. Surface pressure regressed slightly (+1 pp). Run
`y2gigs61`, state=finished, 6 epochs reached, best_epoch=3.

PR #9 was a CLI-flag-only change (no code diff). The win compounds the
existing PR #11 projection-loss code on `yi` with: `--volume-loss-weight 2.0`,
`--batch-size 8`, `--validation-every 1`, `--gradient-log-every 100
--weight-log-every 100`. **Future PRs should adopt this base config.**

**Important caveat** — gilbert's run did **not** include
`--use-tangential-wallshear-loss`, yet still beat kohaku's projection-loss
run by 50%. This means the bulk of the win came from the protocol fixes
(bs=8 + validation-every=1 + log cadence), not the loss form. A follow-up
combining all three (projection + vol_w=2.0 + protocol) should be even
better.

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
| `test_primary/abupt_axis_mean_rel_l2_pct` | **17.3933** | #9 | y2gigs61 | 2026-04-29 |
| `test_primary/surface_pressure_rel_l2_pct` | **11.0733** | #9 | y2gigs61 | 2026-04-29 |
| `test_primary/wall_shear_rel_l2_pct` | **18.3180** | #9 | y2gigs61 | 2026-04-29 |
| `test_primary/volume_pressure_rel_l2_pct` | **15.2059** | #9 | y2gigs61 | 2026-04-29 |
| `test_primary/wall_shear_x_rel_l2_pct` | **15.6465** | #9 | y2gigs61 | 2026-04-29 |
| `test_primary/wall_shear_y_rel_l2_pct` | **21.8605** | #9 | y2gigs61 | 2026-04-29 |
| `test_primary/wall_shear_z_rel_l2_pct` | **23.1803** | #9 | y2gigs61 | 2026-04-29 |

(`p_s = 10.07` from PR #11 is a marginally better single-axis number, but the
abupt_axis_mean win is decisive and `tau_*` improvements dominate the
composite metric.)

**Reproduce (PR #9 winning config — recommended for all future PRs):**

```bash
cd target/
python train.py \
  --volume-loss-weight 2.0 \
  --batch-size 8 \
  --validation-every 1 \
  --lr 2e-4 --weight-decay 5e-4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 256 --model-heads 4 --model-slices 128 \
  --ema-decay 0.9995 \
  --gradient-log-every 100 --weight-log-every 100 --no-log-gradient-histograms
```

**Note:** PR #9 did **not** include `--use-tangential-wallshear-loss`. PR #11
showed the projection helps; combining both should compose. Future Round-2
experiments should layer on top of the gilbert config and add
`--use-tangential-wallshear-loss` if the hypothesis is wall-shear-related.

**Distance from AB-UPT targets (multiple of target):**

| Metric | yi best | AB-UPT | Ratio |
|---|---:|---:|---:|
| surface_pressure | 11.07 | 3.82 | 2.9× |
| wall_shear | 18.32 | 7.29 | 2.5× |
| volume_pressure | 15.21 | 6.08 | 2.5× |
| wall_shear_x | 15.65 | 5.35 | 2.9× |
| wall_shear_y | 21.86 | 3.65 | 6.0× |
| wall_shear_z | 23.18 | 3.63 | 6.4× |

The wall-shear axes (esp. `tau_y`, `tau_z`) remain the largest gap to AB-UPT
but have collapsed by ~60–70% from where we started. Volume pressure and
surface pressure are now the rate-limiting axes by absolute ratio.

**Known training-stability bug (gilbert flagged in PR #9):** `train.py` has
**no gradient clipping**. Run B (vol_w=3.0) and several other Round-1 PRs
diverged on this exact mechanism. Follow-up PR #22 (gilbert) is opened to
add `torch.nn.utils.clip_grad_norm_` between `loss.backward()` and
`optimizer.step()`. Once landed, larger LR / vol_w / batch sweeps become
safe.

## Reference config (`train.py` defaults on `yi`)

```
lr=3e-4  weight_decay=1e-4  batch_size=2  epochs=50
train_/eval_ surface_points=40_000  train_/eval_ volume_points=40_000
model: 3 layers · 192 hidden · 3 heads · 96 slices · mlp_ratio=4
amp=bf16  ema_decay=0.999  validation_every=10
```
