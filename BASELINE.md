# DrivAerML Baseline

**Branch:** `yi` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml`

## Status: thorfinn PR #66 wins — new baseline 2026-04-30

PR #66 (thorfinn, per-axis tau_y/z loss upweighting W_y=2, W_z=2 on 6L/256d base) reduced
`test_primary/abupt_axis_mean_rel_l2_pct` from 13.15 (senku PR #14) to
**12.74** — a 3.1% improvement on the headline metric. Tau_y dropped from 16.23→15.15
(−6.7%) and tau_z from 16.75→15.05 (−10.2%). W&B run: `gvigs86q`.

Key finding: upweighting the two hardest wall-shear axes (tau_y and tau_z) by 2×
improves the composite metric without hurting surface_pressure or volume_pressure.
W_y=2, W_z=2 beats W_y=1.5, W_z=1.5 and the equal-weight arms. Thorfinn's code
adds `--wallshear-y-weight` and `--wallshear-z-weight` flags to `train.py`.

**Compounding wins so far (all landed on `yi`):**
1. PR #11 kohaku — tangential wall-shear projection loss code
2. PR #9 gilbert — protocol fixes (bs=8, vol_w=2.0, validation-every=1)
3. PR #4 chihiro — width scale-up to 512d/8h
4. PR #14 senku — depth scale-up to 6L/256d
5. PR #58 alphonse — NaN-safe checkpoint guard (bugfix)
6. PR #66 thorfinn — per-axis tau_y/z loss upweighting W_y=2, W_z=2 (this PR)

**New recommended base config (PR #66 winning arm):**

```bash
cd target/
python train.py \
  --volume-loss-weight 2.0 \
  --batch-size 8 \
  --validation-every 1 \
  --lr 2e-4 --weight-decay 5e-4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 6 --model-hidden-dim 256 --model-heads 4 --model-slices 128 \
  --ema-decay 0.9995 \
  --clip-grad-norm 1.0 \
  --wallshear-y-weight 2.0 \
  --wallshear-z-weight 2.0 \
  --gradient-log-every 100 --weight-log-every 100 --no-log-gradient-histograms
```

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

**Compounding wins so far (all landed on `yi`):**
1. PR #11 kohaku — tangential wall-shear projection loss
2. PR #9 gilbert — protocol fixes (bs=8, vol_w=2.0, validation-every=1)
3. PR #4 chihiro — width scale-up to 512d/8h (this PR)

PRs #8 (frieren FiLM, 16.53) and #13 (norman cosine EMA, 15.82) are pending
rebase+merge and should push the composite lower still.

---

## Previous: gilbert PR #9 — baseline 2026-04-29 03:57 UTC

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
| `test_primary/abupt_axis_mean_rel_l2_pct` | **12.74** | #66 | gvigs86q | 2026-04-30 |
| `test_primary/surface_pressure_rel_l2_pct` | **7.64** | #14 | et4ajeqj | 2026-04-29 |
| `test_primary/wall_shear_rel_l2_pct` | **12.86** | #66 | gvigs86q | 2026-04-30 |
| `test_primary/volume_pressure_rel_l2_pct` | **13.14** | #66 | gvigs86q | 2026-04-30 |
| `test_primary/wall_shear_x_rel_l2_pct` | **11.29** | #66 | gvigs86q | 2026-04-30 |
| `test_primary/wall_shear_y_rel_l2_pct` | **15.15** | #66 | gvigs86q | 2026-04-30 |
| `test_primary/wall_shear_z_rel_l2_pct` | **15.05** | #66 | gvigs86q | 2026-04-30 |

Note: Additional code wins pending merge (all superseded on headline metric by
PR #14 but contain orthogonal code contributions) — PRs #22 (gilbert clip=1.0,
14.80), #24 (emma sq-rel-L2, 14.81), #3 (askeladd codex-lineage, 15.27),
#13 (norman cosine EMA, 15.82). PR #8 (frieren FiLM, 16.53) merged 2026-04-29 —
FiLM geometry conditioning code now on `yi`.

(`p_s = 10.07` from PR #11 is a marginally better single-axis number, but the
abupt_axis_mean win is decisive and `tau_*` improvements dominate the
composite metric.)

**Reproduce (PR #4 winning config — new recommended base for 512d experiments):**

```bash
cd target/
python train.py \
  --volume-loss-weight 2.0 \
  --batch-size 4 \
  --validation-every 1 \
  --lr 5e-5 --weight-decay 5e-4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.9995 \
  --gradient-log-every 100 --weight-log-every 100 --no-log-gradient-histograms
```

**Note:** `lr=5e-5` is necessary for 512d — three runs at 2e-4 diverged.
`bs=4` is the largest batch fitting 96GB VRAM at 512d. For 256d experiments
still use `bs=8 lr=2e-4` (gilbert PR #9 config).

**Previous 256d baseline reproduce (PR #9):**
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

| Metric | yi best (PR #66) | AB-UPT | Ratio |
|---|---:|---:|---:|
| surface_pressure | 7.64 | 3.82 | 2.0× |
| wall_shear | 12.86 | 7.29 | 1.8× |
| volume_pressure | 13.14 | 6.08 | 2.2× |
| wall_shear_x | 11.29 | 5.35 | 2.1× |
| wall_shear_y | 15.15 | 3.65 | 4.2× |
| wall_shear_z | 15.05 | 3.63 | 4.1× |

Wall_shear_y and wall_shear_z remain the largest gap at ~4× AB-UPT despite
thorfinn's W_y=W_z=2 win. Volume pressure shows a known val→test gap (val
~6.9 ≈ AB-UPT level, test 13.14 = 2.2×) — test generalization is the
remaining challenge, not model capacity per se.

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
