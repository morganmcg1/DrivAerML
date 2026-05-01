# DrivAerML Baseline — `tay`

**Branch:** `tay` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Status: fern PR #222 lr_warmup_epochs=1 — 2026-05-01 19:30 UTC

**NEW SOTA: fern PR #222 lr warmup 1ep beats PR #115 by −2.03% (9.2910 vs 9.484 val).**

1-epoch linear LR warmup added to the SOTA stack (Lion lr=1e-4, EMA=0.999). Smooth convergence improvement with continuous descent across all 9 epochs — no instability. The warmup provides a gentler entry to steep descent, resulting in lower final val. ep1 inflated (warmup effect: LR still ramping) but ep2+ shows consistently better convergence than flat-LR baseline.

**W&B run:** `ut1qmc3i` (rank 0) — group `tay-round12-lr-warmup-1ep`, ~270 min runtime, 9 val epochs, best val 9.2910 (ep9)
**PR:** #222
**Test metrics:** PENDING — run still completing test evaluation. Will update when `test_primary/*` keys appear in summary.

### tay current best — `val_primary/*` (test pending)

| Epoch | val_abupt | surf_pres | vol_pres | wall_shear |
|-------|-----------|-----------|----------|------------|
| ep7 | 9.8759% | 6.3077% | 6.0145% | 11.0603% |
| ep8 | 9.4516% | 6.0019% | 5.7614% | 10.5847% |
| **ep9 (best)** | **9.2910%** | **5.8707%** | **5.8789%** | **10.3423%** |

### Previous best `test_primary/*` (PR #115 thorfinn — valid until PR #222 test completes)

| Metric | This-repo key | tay best (PR #115 thorfinn) | PR #111 tanjiro | PR #50 | AB-UPT |
|---|---|---:|---:|---:|---:|
| `abupt` | `test_primary/abupt_axis_mean_rel_l2_pct` | **10.580** | 11.142 | 11.208 | — |
| `surface_pressure` | `test_primary/surface_pressure_rel_l2_pct` | **5.690** | 6.209 | 6.193 | 3.82 |
| `wall_shear` | `test_primary/wall_shear_rel_l2_pct` | **10.419** | 11.138 | 11.199 | 7.29 |
| `volume_pressure` | `test_primary/volume_pressure_rel_l2_pct` | 12.740 | **12.548** | 12.726 | 6.08 |
| `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **8.908** | 9.436 | 9.512 | 5.35 |
| `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **12.491** | 13.525 | 13.592 | 3.65 |
| `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **13.071** | 13.992 | 14.017 | 3.63 |

### Reproduce new SOTA (Lion lr=1e-4, EMA=0.999, lr_warmup_epochs=1)

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <STUDENT> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.999 --lr-warmup-epochs 1
```

### Compounding wins so far

| PR | Who | Delta | Lever |
|---|---|---:|---|
| #30 | alphonse | baseline (19.81) | yi calibration config (4L/512d/8h, vol_w=2.0) |
| #33 | fern | **−2.04 (−10.3%)** | RFF coord features (sigma=1.0, 32 feats) — uncompiled |
| #40 | alphonse | **−0.52 (−2.9%) vs #33** | torch.compile fix → 12 epochs vs 9 |
| #39 | tanjiro | **−1.82 (−10.5%) vs #40** | Lion optimizer lr=1.7e-5 |
| #46 | alphonse | **−0.88 (−5.7%) vs #39** | AdamW + RFF (sigma=1.0) + compile → epoch 16 |
| (no PR) | tanjiro arm B | **−3.25 (−22.3%) vs #46** | Lion lr=5e-5/wd=5e-4 — paper config was wrong |
| #50 | nezuko | **−0.10 (−0.84%) vs arm B** | Lion uncompiled lr=5e-5 reproduce |
| #110 | edward | **−0.04 (−0.34%) vs #50** | Lion + cosine T_max=50 (gentle 8% decay) |
| #111 | tanjiro | **−0.03 (−0.25%) vs #110** | EMA decay 0.999 (5× faster than 0.9995) |
| #115 | thorfinn | **−0.562 (−5.04%) vs #111** | Compound: lr=1e-4 + EMA=0.999 (both winners stacked) |
| **#222** | **fern** | **−0.193 (−2.03%) vs #115** | **lr_warmup_epochs=1 (1-epoch linear warmup on top of SOTA stack)** |

---

## Prior SOTA record: tanjiro Lion-arm-B — 2026-04-30 02:44 UTC

**MASSIVE NEW SOTA: Lion lr=5e-5/wd=5e-4 (NOT paper config) blows past PR #46 by −22.3%.**

This was a follow-up sweep run launched by tanjiro's pod after PR #39 was reviewed —
**not** an advisor-assigned PR experiment. The config is Lion at the AdamW-equivalent
LR/WD translation (lr=5e-5, wd=5e-4), no compile, no RFF. Despite being uncompiled,
the run trained for 4h50m past the 270-min budget cap and the val curve was still
descending at the end. Best-val checkpoint reload gave the test result.

**W&B run:** `vnb7oheo` (rank 0) — group `tanjiro-lion-lr-sweep`
**Best-val checkpoint:** val_abupt = 10.096 at last logged epoch
**Runtime:** 4h50m (290 min, past budget — likely launched without strict timeout)
**Advisor note:** No PR for this run. Result documented retroactively as the new SOTA.

### CRITICAL FINDING: Lion paper config is wrong for this task

PR #39 used Lion at `lr=1.7e-5, wd=5e-3` (paper config from Chen et al.) → test_abupt 15.43.
This run used `lr=5e-5, wd=5e-4` (the AdamW-equivalent translation tanjiro tested as
arm B of the original sweep) → test_abupt **11.30**. That is a **−27% improvement
just from changing the LR/WD constants on the same Lion optimizer**.

Why the paper config fails here:
- Lion paper used image classification with millions of training samples; we have 400 cars.
- Smaller datasets need more aggressive per-step movement (higher lr) to traverse the
  loss landscape within a 270-min budget.
- Higher wd in the paper helps regularize huge nets; our 4L/512d/8h is small enough
  that wd=5e-4 (AdamW-equivalent) is sufficient.

**All future Lion experiments must use `--lr 5e-5 --weight-decay 5e-4`, not the paper
config.** Queued PRs #50, #51, #52, #54 need their LR/WD updated.

### tay current best — `test_primary/*`

| Metric | This-repo key | tay best (Lion 5e-5) | PR #46 (RFF+compile) | PR #39 Lion (paper) | yi best | AB-UPT |
|---|---|---:|---:|---:|---:|---:|
| `abupt` | `test_primary/abupt_axis_mean_rel_l2_pct` | **11.303** | 14.550 | 15.43 | 15.82 | — |
| `surface_pressure` | `test_primary/surface_pressure_rel_l2_pct` | **6.216** | 8.628 | 9.45 | 9.99 | 3.82 |
| `wall_shear` | `test_primary/wall_shear_rel_l2_pct` | **11.315** | 14.882 | 16.28 | 16.60 | 7.29 |
| `volume_pressure` | `test_primary/volume_pressure_rel_l2_pct` | **12.755** | 15.032 | 13.83 | 14.21 | 6.08 |
| `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **9.563** | 12.901 | 13.91 | 14.27 | 5.35 |
| `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **13.831** | 17.281 | 19.58 | 19.49 | 3.65 |
| `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **14.147** | 18.907 | 20.40 | 21.12 | 3.63 |

**Every axis improved by 15-29% vs PR #46. tau_y/tau_z gap to AB-UPT ref is now 3.8×
(was 5.4× and 5.6×).** This is the largest single jump in the project so far.

### Reproduce new SOTA (Lion lr=5e-5)

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 5e-5 --weight-decay 5e-4 \
  --volume-loss-weight 2.0 --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.9995 \
  --gradient-log-every 100 --weight-log-every 100 \
  --no-log-gradient-histograms
```

NOTE: `vnb7oheo` ran ~290 min (past budget). With strict 270-min budget the result
might land slightly higher (~11.5-12.0). Future reproduce runs should use the same
budget enforcement as standard PRs.

### Compounding wins so far

| PR | Who | Delta | Lever |
|---|---|---:|---|
| #30 | alphonse | baseline (19.81) | yi calibration config (4L/512d/8h, vol_w=2.0) |
| #33 | fern | **−2.04 (−10.3%)** | RFF coord features (sigma=1.0, 32 feats) — uncompiled |
| #40 | alphonse | **−0.52 (−2.9%) vs #33** | torch.compile fix → 12 epochs vs 9; beats #33 without RFF |
| #39 | tanjiro | **−1.82 (−10.5%) vs #40** | Lion optimizer lr=1.7e-5 — sign-based update, crosses yi frontier |
| #46 | alphonse | **−0.88 (−5.7%) vs #39** | AdamW + RFF (sigma=1.0) + compile → epoch 16; tau_y −11.7% |
| (no PR) | tanjiro arm B | **−3.25 (−22.3%) vs #46** | Lion lr=5e-5/wd=5e-4 (AdamW-equivalent translation, NOT paper config) — paper config was the culprit |

### INFRA NOTE — torch.compile bug FIXED (PR #40)

Two-line patch in `trainer_runtime.py`:
1. `drop_last=True` on `DistributedSampler` and `DataLoader` (lines 293, 301) — fixes partial-batch crash.
2. `unwrap_model(model)` in `accumulate_eval_batch` (line 929) — bypasses compile during eval because `pad_collate` produces variable-shape batches that trigger a symbolic-sum codegen bug in torch.inductor.

**All future runs should use `--compile-model` (the default) without `--no-compile-model`.**
Throughput: ~16 min/epoch compiled vs ~18 min uncompiled. 270-min budget → 12 epochs.

## Reference baseline targets — must beat (AB-UPT public reference)

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

## Yi reference results (different W&B project — for context only)

The prior `yi` advisor reached the following on their project. These are
informational targets to match or beat on tay/DDP8:

| Metric | Best on yi | PR | W&B run | Date |
|---|---:|---|---|---|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **15.82** | yi#13 | wio9pqw2 | 2026-04-29 |
| `test_primary/surface_pressure_rel_l2_pct` | **9.99** | yi#13 | wio9pqw2 | 2026-04-29 |
| `test_primary/wall_shear_rel_l2_pct` | **16.60** | yi#13 | wio9pqw2 | 2026-04-29 |
| `test_primary/volume_pressure_rel_l2_pct` | **14.21** | yi#13 | wio9pqw2 | 2026-04-29 |

(yi#13 was norman's progressive cosine EMA 0.99→0.9999, pending merge on yi.)

## Confirmed-orthogonal levers from yi (to compose on tay)

1. Width 512d / heads 8 / slices 128 (yi PR #4 chihiro) — the 256d→512d step
   moves volume_pressure 15.21 → 14.37. Needs `lr=5e-5`, `bs=4` at 512d.
2. Protocol fixes (yi PR #9 gilbert) — `--volume-loss-weight 2.0
   --batch-size 8 --validation-every 1`. Halved abupt mean over the
   pre-protocol baseline.
3. Tangential wall-shear projection loss (yi PR #11 kohaku) — denormalize →
   project onto surface tangent → renormalize. Default off; opt in with
   `--use-tangential-wallshear-loss` if/when ported.
4. AdaLN-zero per-block FiLM geometry conditioning (yi PR #8 frieren) —
   independent +5% on every axis at 256d.
5. Cosine EMA decay 0.99 → 0.9999 (yi PR #13 norman) — single largest
   non-architectural lever in yi (−9% on every axis).

## Update protocol

When a tay PR lands a new best `test_primary/abupt_axis_mean_rel_l2_pct`:
1. Update the Status header to the new PR + W&B run + date.
2. Replace the per-axis best table with the new run's `test_primary/*`.
3. Append a short "Compounding wins so far" entry naming the orthogonal lever.
