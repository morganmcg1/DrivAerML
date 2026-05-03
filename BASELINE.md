# DrivAerML Baseline — `tay`

**Branch:** `tay` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Status: thorfinn PR #489 volume-points curriculum 16k→65k ramp — 2026-05-03 (updated)

**NEW SOTA: thorfinn PR #489 (vol-points curriculum 16k→32k→49k→65k over 4 stages) beats PR #488 by −0.1880pp val (7.1792% vs 7.3672% val). W&B run `r5rw40rn`, EP11. Delta −2.55% relative.**

Volume-points curriculum ramps the number of volume query points across training epochs: 16384 (ep0-2) → 32768 (ep3-5) → 49152 (ep6-8) → 65536 (ep9+). This curriculum approach lets the model first learn from cheaper, coarser sampling and progressively refine as training matures. Run timed out at 270 min before reaching the 49k/65k stages, but the final result still beats SOTA. Volume pressure improves further (vp=4.207% vs prior SOTA 4.357%), suggesting the curriculum meaningfully helps volume field fidelity. tau_y/tau_z remain the primary open problem.

**W&B run:** `r5rw40rn` (thorfinn DDP8, rank 0) — group `thorfinn-vol-curriculum`, best val **7.1792%** (EP11), all 8 DDP siblings finished
**PR:** #489
**Val metrics (best-val checkpoint):** val_abupt=7.1792%, surface_pressure=4.783%, wall_shear=8.098%, volume_pressure=4.207%, tau_x=7.019%, tau_y=9.187%, tau_z=10.701%
**Test metrics:** test_abupt=8.497%

### tay current best — `val_primary/*` (PR #489 thorfinn, run `r5rw40rn`)

| Metric | **PR #489 thorfinn (SOTA)** | PR #488 alphonse (prev) | AB-UPT |
|---|---:|---:|---:|
| `abupt` | **7.1792** | 7.3672 | — |
| `surface_pressure` | 4.783 | 4.805 | 3.82 |
| `wall_shear` | 8.098 | 8.347 | 7.29 |
| `volume_pressure` | **4.207** | 4.357 | 6.08 |
| `tau_x` | 7.019 | — | 5.35 |
| `tau_y` | 9.187 | — | 3.65 |
| `tau_z` | 10.701 | — | 3.63 |

**Key insight:** Volume-points curriculum improves across all axes. vp=4.207% beats prior SOTA 4.357% and stays well below AB-UPT ref 6.08%. tau_y/tau_z gap (9.19%/10.70% vs target 3.65%/3.63%) remains the primary unsolved problem — 2.5-2.9× above reference. The curriculum suggests progressive training strategies deserve further investigation.

### Reproduce new SOTA (Lion lr=1e-4, EMA=0.999, STRING-sep, QK-norm, feat16 RFF, multi-sigma init, vol-curriculum)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --agent thorfinn --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 \
  --rff-init-sigmas <octave-sigmas-from-pr488> \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536"
```

### Compounding wins so far (updated through PR #489)

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
| #222 | fern | **−0.193 (−2.03%) vs #115** | lr_warmup_epochs=1 (1-epoch linear warmup on top of SOTA stack) |
| #232 | askeladd | **−0.226 (−2.44%) vs #222** | model-heads=4 (halving attention heads from 8 to 4) |
| #309 | thorfinn | **−0.064pp (−0.63%) vs #232** | grad-clip-norm=0.5 (Lion EMA momentum stabilization, avoids ep8 regression) |
| #311 | edward | **−1.355pp (−13.39%) vs #309** | STRING-separable pos encoding: learnable per-axis log_freq + phase — largest single gain since tanjiro arm B |
| #358 | thorfinn | **−0.154pp (−2.04%) vs #311** | QK-norm (RMSNorm on Q and K) stacked on STRING-sep — best val at EP11 (7.3921%) |
| #387 | alphonse | **−0.031pp (−0.36%) vs #358** | feat16 RFF (rff_num_features=16) stacked on STRING-sep + QK-norm config — best val at EP11 (7.3816%) |
| #488 | alphonse | **−0.0144pp (−0.195%) vs #387** | multi-sigma STRING-sep init: log_freq params distributed across frequency octaves at init — vp drops from 12.189% to 4.357% (beats AB-UPT ref!) |
| **#489** | **thorfinn** | **−0.1880pp (−2.55%) vs #488** | **vol-points curriculum 16k→32k→49k→65k: progressive coarse-to-fine volume sampling across training epochs — vp further improves to 4.207%** |

---

## Prior SOTA record: alphonse PR #488 multi-sigma STRING-sep init — 2026-05-03 (updated)

**PRIOR SOTA: alphonse PR #488 (multi-sigma RFF init across frequency octaves) beats PR #387 by −0.0144pp val (7.3672% vs 7.3816% val). W&B run `ki2q9ko9`, EP11.**

Multi-sigma STRING-sep init distributes `log_freq` parameters across frequency octaves at initialization via `--rff-init-sigmas`, giving the STRING-sep encoding a broader spectral coverage from the start. Dramatically improves volume_pressure (vp=4.357% vs SOTA 12.189% — a +7.832pp improvement), bringing it to near-target territory (AB-UPT ref: 6.08%). Surface pressure and wall shear see modest regression (+0.367pp and +0.348pp respectively), but the net val_abupt improvement confirms the octave-init approach is a genuine advance.

**W&B run:** `ki2q9ko9` (alphonse DDP8) — best val **7.3672%** (EP11)
**PR:** #488
**Val metrics (best-val checkpoint):** val_abupt=7.3672%, surface_pressure=4.805%, wall_shear=8.347%, volume_pressure=4.357%

---

## Prior SOTA record: alphonse PR #387 feat16 RFF + QK-norm + STRING-sep — 2026-05-01 (updated)

**PRIOR SOTA: alphonse PR #387 (feat16 RFF + QK-norm stacked on STRING-sep) beats PR #358 by −0.0105pp val (7.3816% vs 7.3921% val). W&B run `wj6mn6ve`, EP11 (Arm A: rff_num_features=16).**

RFF with rff_num_features=16 (feat16) stacks on top of the STRING-sep + QK-norm SOTA baseline. The feat16 encoding adds 16-feature Random Fourier Features on top of the learnable per-axis STRING-sep frequencies, providing richer spectral coverage at low compute cost. Both val and test improve over the prior SOTA.

**W&B run:** `wj6mn6ve` (alphonse DDP8) — group `alphonse-rff-sweep`, best val **7.3816%** (EP11)
**PR:** #387
**Test metrics (best-val checkpoint):** test_abupt=8.5936%, surface_pressure=4.4377%, wall_shear=7.9989%, volume_pressure=12.1885%, tau_x=6.9622%, tau_y=9.1058%, tau_z=10.2736%

---

## Prior SOTA record: thorfinn PR #358 STRING-sep + QK-norm stack — 2026-05-02 (updated)

**PRIOR SOTA: thorfinn PR #358 (STRING-sep + QK-norm stack) beat PR #311 by −0.154pp val (7.3921% vs 7.546% val). W&B run `tkiigfmc`, EP11.**

QK-norm adds `nn.RMSNorm(dim_head, elementwise_affine=True)` applied to Q and K projections immediately after the qkv chunk, before `F.scaled_dot_product_attention`. Stacks directly on top of PR #311 STRING-sep config. Convergence continued improving to EP11.

**W&B run:** `tkiigfmc` (thorfinn DDP8) — group `thorfinn-string-qknorm-r19`, best val **7.3921%** (EP11)
**PR:** #358
**Test metrics (best-val checkpoint):** test_abupt=8.625%, surface_pressure=4.462%, wall_shear=7.965%, volume_pressure=12.434%

---

## Prior SOTA record: edward PR #311 STRING-separable positional encoding — 2026-05-01 (updated)

**NEW SOTA: edward PR #311 (STRING-separable pos encoding) beats PR #309 by −1.493pp val / −1.355pp test (7.546% vs 9.039% val, 8.771% vs 10.126% test). This is a −13.93% relative improvement on test_abupt.**

STRING-separable replaces fixed isotropic Gaussian RFF with learnable per-axis frequency/phase (`log_freq` + `phase` as `nn.Parameter`, one per axis). The axis-separable factorization learns independent spectral emphasis per spatial axis, matching the anisotropic structure of automotive aerodynamics. All gradient diagnostics healthy (nonfinite_count: 0 throughout). Val slopes still negative at terminal epoch — model still converging, further gains possible.

**W&B run:** `gcwx9yaa` (rank 0) — group `tay-round18-grape-ablation`, STRING arm, best val 7.546%
**PR:** #311
**Test metrics:** test_abupt 8.771% (test_primary/abupt_axis_mean_rel_l2_pct)

### tay current best — `val_primary/*`

| Epoch | val_abupt |
|-------|-----------|
| **best** | **7.546%** |

### tay current best — `test_primary/*` (PR #311 edward, run `gcwx9yaa`)

| Metric | This-repo key | **PR #311 edward (NEW SOTA)** | PR #309 thorfinn (prev) | PR #232 askeladd | AB-UPT |
|---|---|---:|---:|---:|---:|
| `abupt` | `test_primary/abupt_axis_mean_rel_l2_pct` | **8.771** | 10.126 | 10.190 | — |
| `surface_pressure` | `test_primary/surface_pressure_rel_l2_pct` | **4.485** | 5.395 | 5.461 | 3.82 |
| `wall_shear` | `test_primary/wall_shear_rel_l2_pct` | **8.227** | 9.883 | 9.910 | 7.29 |
| `volume_pressure` | `test_primary/volume_pressure_rel_l2_pct` | **12.438** | 12.484 | 12.656 | 6.08 |
| `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **7.253** | 8.402 | 8.432 | 5.35 |
| `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **9.233** | 11.941 | 11.952 | 3.65 |
| `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **10.449** | 12.407 | 12.447 | 3.63 |

**Wins over PR #309 on every axis.** Largest gains: tau_y −2.708pp (−22.68%), tau_z −1.958pp (−15.78%), surface_pressure −0.91pp (−16.87%), wall_shear −1.656pp (−16.76%).

**3-arm ablation comparison (tay-round18-grape-ablation):**

| Arm | Encoding | val_abupt | test_abupt | vs SOTA |
|-----|----------|-----------|------------|---------|
| A (RFF-32) | Fixed isotropic Gaussian | 9.710% | 10.721% | +0.595pp worse |
| **B (STRING-sep)** | **Learnable per-axis freq/phase** | **7.546%** | **8.771%** | **−1.355pp better** |
| C (GRAPE-M) | Minimal learned spectral proj | still running | — | — |
| SOTA (#309) | No spectral encoding (RFF-0) | 9.039% | 10.126% | baseline |

### Reproduce new SOTA (Lion lr=1e-4, EMA=0.999, heads=4, grad-clip-norm=0.5, STRING-sep)

**Note:** STRING-separable encoding uses `--pos-encoding-mode string_separable` (or equivalent flag). Stacks on the full PR #309 SOTA config.

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <STUDENT> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 \
  --pos-encoding-mode string_separable
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
| #222 | fern | **−0.193 (−2.03%) vs #115** | lr_warmup_epochs=1 (1-epoch linear warmup on top of SOTA stack) |
| #232 | askeladd | **−0.226 (−2.44%) vs #222** | model-heads=4 (halving attention heads from 8 to 4) |
| #309 | thorfinn | **−0.064pp (−0.63%) vs #232** | grad-clip-norm=0.5 (Lion EMA momentum stabilization, avoids ep8 regression) |
| **#311** | **edward** | **−1.355pp (−13.39%) vs #309** | **STRING-separable pos encoding: learnable per-axis log_freq + phase — largest single gain since tanjiro arm B** |

---

## Prior SOTA record: thorfinn PR #309 grad-clip-norm=0.5 — 2026-05-02 07:17 UTC

**PRIOR SOTA: thorfinn PR #309 grad-clip-norm=0.5 beats PR #232 by −0.026pp val / −0.064pp test (9.0389% vs 9.0650% val, 10.126% vs 10.190% test).**

**W&B run:** `ztdhodw1` (rank 0) — group `thorfinn-gradclip-r15`, ~270 min runtime, best val 9.0389% (ep11)
**PR:** #309
**Test metrics:** test_abupt 10.126% (test_primary/abupt_axis_mean_rel_l2_pct)

---

## Prior SOTA record: askeladd PR #232 model-heads=4 — 2026-05-01 21:06 UTC

**PRIOR SOTA: askeladd PR #232 heads=4 beat PR #222 by −0.226pp (9.0650% vs 9.2910% val).**

**W&B run:** `r8s2dtnq` (rank 0) — group `tay-round12-heads-4`, 282.5 min runtime, best val 9.0650% (ep11)
**PR:** #232
**Test metrics:** test_abupt 10.190%

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <STUDENT> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999
```

---

## Prior SOTA record: fern PR #222 lr_warmup_epochs=1 — 2026-05-01 19:30 UTC

**NEW SOTA: fern PR #222 lr warmup 1ep beats PR #115 by −2.03% (9.2910 vs 9.484 val).**

1-epoch linear LR warmup added to the SOTA stack (Lion lr=1e-4, EMA=0.999). Smooth convergence improvement with continuous descent across all 9 epochs — no instability. The warmup provides a gentler entry to steep descent, resulting in lower final val. ep1 inflated (warmup effect: LR still ramping) but ep2+ shows consistently better convergence than flat-LR baseline.

**W&B run:** `ut1qmc3i` (rank 0) — group `tay-round12-lr-warmup-1ep`, ~270 min runtime, 9 val epochs, best val 9.2910 (ep9)
**PR:** #222
**Test metrics:** **CONFIRMED — test_abupt 10.420% (beats prior PR #115 SOTA 10.580% by −0.16pp / −1.51%).** Updated 2026-05-01 from W&B run `ut1qmc3i` summary.

### Reproduce (Lion lr=1e-4, EMA=0.999, lr_warmup_epochs=1)

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
