# DrivAerML Baseline

**Branch:** `yi` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml`

## Status: edward PR #311 wins — new baseline 2026-05-02

PR #311 (edward, STRING-separable learnable position encoding) reduced
`val_primary/abupt_axis_mean_rel_l2_pct` from 9.039% (PR #309 thorfinn grad-clip=0.5) to
**7.546%** — a 16.5% improvement on the headline metric, and a 13.9% improvement
on `test_primary/abupt_axis_mean_rel_l2_pct` (10.190% → **8.771%**). W&B run: `gcwx9yaa`,
group: `tay-round18-grape-ablation`, Arm B.

Key finding: replacing isotropic RFF-32 (or no spectral encoding) with a
**STRING-separable** position encoding — axis-aligned frequencies with per-axis
**learnable** `log_freq` and `phase` parameters — yields the cleanest
architectural win in many rounds. Test deltas vs prior SOTA: surface_pressure
−0.98pp, wall_shear −1.68pp, volume_pressure −0.22pp. All val slopes still
negative at terminal epoch (model still converging) → more epochs would likely
extend the gain. This is the **first new-architecture (vs HP) win** in several
rounds and should compound with the existing 4L/512d Lion+warmup stack.

## Previous: fern PR #222 wins — baseline 2026-05-01

PR #222 (fern, 1-epoch LR warmup before cosine decay) reduced
`val_primary/abupt_axis_mean_rel_l2_pct` from 9.484% (fern PR #208 / prior best) to
**9.2910%** — a 2.0% improvement on the headline metric. W&B run: `ut1qmc3i`, group: `tay-round12-lr-warmup-1ep`.

Key finding: adding a single epoch of linear LR warm-up before cosine annealing stabilises
Lion training on the 4L/512d architecture (lr=1e-4, batch=4, torchrun 8-GPU DDP).
Smooth loss descent across all 9 epochs with no instability. Best-epoch=9.
Operates on the **4L/512d** architecture (not 6L/256d), which had been the
previous SoTA architecture.

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
10. PR #222 fern — lr_warmup_epochs=1 (Lion stability, 4L/512d architecture)
11. PR #309 thorfinn — gradient clipping max_norm=0.5 (val 9.039%)
12. PR #311 edward — STRING-separable learnable position encoding (val 7.546%, test 8.771%)
13. PR #355 emma — DDP infrastructure restored (cherry-pick bfbe975+1a8f7b7: init_process_group, DistributedSampler, DDP wrap, Lion optimizer wiring; unblocks full fleet 4/8-GPU torchrun; metrics bar unchanged)

**New recommended base config (PR #222 winning arm):**

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --agent fern \
  --optimizer lion \
  --lr 1e-4 \
  --weight-decay 5e-4 \
  --no-compile-model \
  --batch-size 4 \
  --validation-every 1 \
  --train-surface-points 65536 \
  --eval-surface-points 65536 \
  --train-volume-points 65536 \
  --eval-volume-points 65536 \
  --model-layers 4 \
  --model-hidden-dim 512 \
  --model-heads 8 \
  --model-slices 128 \
  --ema-decay 0.999 \
  --lr-warmup-epochs 1
```

**PR #222 epoch-by-epoch metrics (W&B run `ut1qmc3i`):**

| Epoch | Step  | val_abupt  | surf_pres | vol_pres  | wall_shear |
|-------|-------|-----------|-----------|-----------|------------|
| 1     | 2720  | 67.7263%  | 52.1452%  | 59.6851%  | 68.9568%   |
| 2     | 5441  | 41.9288%  | 31.5194%  | 25.0452%  | 46.0780%   |
| 3     | 8162  | 19.3033%  | 13.3635%  | 11.4773%  | 21.4948%   |
| 4     | 10883 | 13.7327%  | 9.2451%   | 8.0774%   | 15.3581%   |
| 5     | 13604 | 11.6016%  | 7.6196%   | 6.8622%   | 13.0241%   |
| 6     | 16325 | 10.5020%  | 6.7791%   | 6.2569%   | 11.7950%   |
| 7     | 19046 | 9.8759%   | 6.3077%   | 6.0145%   | 11.0603%   |
| 8     | 21767 | 9.4516%   | 6.0019%   | 5.7614%   | 10.5847%   |
| **9** | **23544** | **9.2910%** | **5.8707%** | **5.8789%** | **10.3423%** |

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

| Metric | Best (val) | Best (test) | PR | W&B run | Date |
|---|---:|---:|---|---|---|
| `abupt_axis_mean_rel_l2_pct` | **7.546** | **8.771** | #311 | gcwx9yaa | 2026-05-02 |
| `surface_pressure_rel_l2_pct` | — | **4.485** | #311 | gcwx9yaa | 2026-05-02 |
| `wall_shear_rel_l2_pct` | — | **8.227** | #311 | gcwx9yaa | 2026-05-02 |
| `volume_pressure_rel_l2_pct` | — | **12.438** | #311 | gcwx9yaa | 2026-05-02 |
| `wall_shear_x_rel_l2_pct` | — | **7.253** | #311 | gcwx9yaa | 2026-05-02 |
| `wall_shear_y_rel_l2_pct` | — | **9.233** | #311 | gcwx9yaa | 2026-05-02 |
| `wall_shear_z_rel_l2_pct` | — | **10.449** | #311 | gcwx9yaa | 2026-05-02 |

Note: PR #311 (edward, STRING-separable position encoding) merged 2026-05-02 —
the first new-architecture win in many rounds. Prior compounding wins:
PR #309 (thorfinn, grad-clip=0.5, val 9.039%), PR #222 (fern, lr_warmup_epochs=1, val 9.291%).
Additional code wins in history: PRs #98 (emma weight-decay), #106 (thorfinn yw2.5-zw2.5),
#97 (edward slices192), #63 (askeladd sq-rel), #104 (senku ema9997), #102 (haku dropout),
#8 (frieren FiLM), #169 (thorfinn infra), #183 (fern omega-bank).
**Merge bar: val_abupt 7.546% — any PR must beat this to merge.**

**Distance from AB-UPT targets (test, multiple of target):**

| Metric | yi best test (PR #311) | AB-UPT | Ratio |
|---|---:|---:|---:|
| surface_pressure | 4.485 | 3.82 | 1.17× |
| wall_shear | 8.227 | 7.29 | 1.13× |
| volume_pressure | 12.438 | 6.08 | 2.05× |
| wall_shear_x | 7.253 | 5.35 | 1.36× |
| wall_shear_y | 9.233 | 3.65 | 2.53× |
| wall_shear_z | 10.449 | 3.63 | 2.88× |
| abupt_axis_mean | 8.771 | — | — |

Surface pressure and wall_shear (vector) have closed substantially with PR #311.
The dominant remaining gaps are **wall_shear_y/z (2.5×, 2.9×)** and
**volume_pressure (2.0×)** — these are the key research targets for upcoming
rounds. PR #311 STRING-sep also showed all val slopes still negative at terminal
epoch, so longer training on this architecture is itself a candidate.

## Reference config (`train.py` defaults on `yi`)

```
lr=3e-4  weight_decay=1e-4  batch_size=2  epochs=50
train_/eval_ surface_points=40_000  train_/eval_ volume_points=40_000
model: 3 layers · 192 hidden · 3 heads · 96 slices · mlp_ratio=4
amp=bf16  ema_decay=0.999  validation_every=10
```
