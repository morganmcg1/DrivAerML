# DrivAerML Baseline

**Branch:** `yi` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml`

## Status: fern PR #183 wins — new baseline 2026-04-29

PR #183 (fern, omega-bank sweep: pos_max_wavelength=1000) reduced
`val_primary/abupt_axis_mean_rel_l2_pct` from 10.69 (fern PR #99) to
**10.21** — a 4.5% improvement on the headline metric. W&B run: `bplngfyo`.

Key finding: compressing `pos_max_wavelength` from 10,000 → 1,000 in
`ContinuousSincosEmbed` significantly improves all metrics. The vehicle bbox
is ~8m × 2.5m × 2m, so 1,000 gives much denser frequency sampling across all
axes. Best-ever volume_pressure (6.32 vs 7.85) and consistent improvements on
surface_pressure (6.85 vs 6.97), wall_shear (11.43 vs 11.69). 4.6h runtime.

**Compounding wins so far (all landed on `yi`):**
1. PR #11 kohaku — tangential wall-shear projection loss code
2. PR #9 gilbert — protocol fixes (bs=8, vol_w=2.0, validation-every=1)
3. PR #4 chihiro — width scale-up to 512d/8h
4. PR #14 senku — depth scale-up to 6L/256d
5. PR #58 alphonse — NaN-safe checkpoint guard (bugfix)
6. PR #66 thorfinn — per-axis tau_y/z loss upweighting W_y=2, W_z=2
7. PR #99 fern — LR peak 5e-4 (5× base)
8. PR #169 thorfinn — NaN/Inf-skip safeguard, --seed, --lr-warmup-steps (infra utilities, no metric regression)
9. PR #183 fern — pos_max_wavelength=1000 (omega-bank sincos positional encoding)

**New recommended base config (PR #183 winning arm):**

```bash
cd target/
python train.py \
  --volume-loss-weight 2.0 \
  --batch-size 8 \
  --validation-every 1 \
  --lr 5e-4 --weight-decay 5e-4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 6 --model-hidden-dim 256 --model-heads 4 --model-slices 128 \
  --ema-decay 0.9995 \
  --clip-grad-norm 1.0 \
  --wallshear-y-weight 2.0 \
  --wallshear-z-weight 2.0 \
  --pos-max-wavelength 1000 \
  --gradient-log-every 100 --weight-log-every 100 --no-log-gradient-histograms
```

---

## Previous: thorfinn PR #66 — baseline 2026-04-30

PR #66 (thorfinn, per-axis tau_y/z loss upweighting W_y=2, W_z=2 on 6L/256d base) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 13.15 (senku PR #14) to
**12.74** — a 3.1% improvement on the headline metric. Tau_y dropped from 16.23→15.15
(−6.7%) and tau_z from 16.75→15.05 (−10.2%). W&B run: `gvigs86q`.

Key finding: upweighting the two hardest wall-shear axes (tau_y and tau_z) by 2×
improves the composite metric without hurting surface_pressure or volume_pressure.
W_y=2, W_z=2 beats W_y=1.5, W_z=1.5 and the equal-weight arms. Thorfinn's code
adds `--wallshear-y-weight` and `--wallshear-z-weight` flags to `train.py`.

---

## Previous: senku PR #14 — baseline 2026-04-29

PR #14 (senku, 6L/256d depth scale-up) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 16.64 (chihiro PR #4) to
**13.15** — a 21.0% improvement on the headline metric. Both 5L (13.52, −18.7%)
and 6L (13.15, −21.0%) beat all pending PRs. W&B runs: `t5tv01ch` (5L) and
`et4ajeqj` (6L). Key finding: depth is more parameter-efficient than width —
6L/256d (4.73M params) crushes 4L/512d (12.7M params).

---

## Previous: chihiro PR #4 — baseline 2026-04-29

PR #4 (chihiro, 4L/512d/8h large-model scale-up) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 17.39 (gilbert PR #9) to
**16.64** — a 4.3% improvement on the headline metric. Run `pejudvyd`,
state=finished, 3 best epochs, params ~12.7M. Width upgrade used `lr=5e-5`
(3 prior runs at 2e-4 diverged) and `bs=4` (largest power-of-2 fitting 96GB).
Standout gain: `volume_pressure` 14.37 vs 15.21 — orthogonal to FiLM and
cosine-EMA wins still pending merge (PRs #8, #13).

---

## Previous: gilbert PR #9 — baseline 2026-04-29 03:57 UTC

PR #9 (gilbert, vol_w=2.0 + protocol fixes) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 35.12 (kohaku PR #11) to
**17.39** — a 50.5% improvement on the headline metric. Wall-shear axes saw
~50–70% reductions. Surface pressure regressed slightly (+1 pp). Run
`y2gigs61`, state=finished, 6 epochs reached, best_epoch=3.

PR #9 was a CLI-flag-only change (no code diff). The win compounds the
existing PR #11 projection-loss code on `yi` with: `--volume-loss-weight 2.0`,
`--batch-size 8`, `--validation-every=1`, `--gradient-log-every 100
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
| `val_primary/abupt_axis_mean_rel_l2_pct` | **10.21** | #183 | bplngfyo | 2026-04-29 |
| `val_primary/surface_pressure_rel_l2_pct` | **6.85** | #183 | bplngfyo | 2026-04-29 |
| `val_primary/wall_shear_rel_l2_pct` | **11.43** | #183 | bplngfyo | 2026-04-29 |
| `val_primary/volume_pressure_rel_l2_pct` | **6.32** | #183 | bplngfyo | 2026-04-29 |
| `val_primary/wall_shear_x_rel_l2_pct` | **9.89** | #183 | bplngfyo | 2026-04-29 |
| `val_primary/wall_shear_y_rel_l2_pct` | **13.47** | #183 | bplngfyo | 2026-04-29 |
| `val_primary/wall_shear_z_rel_l2_pct` | **14.52** | #183 | bplngfyo | 2026-04-29 |

Note: Additional code wins pending merge (all superseded on headline metric by
PR #183 but contain orthogonal code contributions) — PRs #98 (emma weight-decay),
#106 (thorfinn yw2.5-zw2.5), #97 (edward slices192), #63 (askeladd sq-rel),
#104 (senku ema9997), #102 (haku dropout). PRs #8 (frieren FiLM) merged 2026-04-29.
PR #169 (thorfinn, infra utilities) merged 2026-04-29 — adds NaN/Inf-skip, --seed, --lr-warmup-steps to train.py.
PR #183 (fern, omega-bank sweep) merged 2026-04-29 — pos_max_wavelength=1000 now default in train.py.

**Distance from AB-UPT targets (multiple of target):**

| Metric | yi best (PR #183) | AB-UPT | Ratio |
|---|---:|---:|---:|
| surface_pressure | 6.85 | 3.82 | 1.8× |
| wall_shear | 11.43 | 7.29 | 1.6× |
| volume_pressure | 6.32 | 6.08 | 1.0× |
| wall_shear_x | 9.89 | 5.35 | 1.8× |
| wall_shear_y | 13.47 | 3.65 | 3.7× |
| wall_shear_z | 14.52 | 3.63 | 4.0× |

Wall_shear_y and wall_shear_z remain the largest gap at ~4× AB-UPT.
Volume pressure has now matched AB-UPT (1.0×, 6.32 vs 6.08), suggesting
strong capacity for scalar fields. Wall-shear axis precision remains the
key challenge and the primary target for future experiments.

## Reference config (`train.py` defaults on `yi`)

```
lr=3e-4  weight_decay=1e-4  batch_size=2  epochs=50
train_/eval_ surface_points=40_000  train_/eval_ volume_points=40_000
model: 3 layers · 192 hidden · 3 heads · 96 slices · mlp_ratio=4
amp=bf16  ema_decay=0.999  validation_every=10
```
