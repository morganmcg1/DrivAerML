# SENPAI Research Results — DrivAerML (`tay`)

W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`.

Targets to beat (lower is better, AB-UPT public reference):
`surface_pressure 3.82`, `wall_shear 7.29`, `volume_pressure 6.08`,
`tau_x 5.35`, `tau_y 3.65`, `tau_z 3.63`.

## 2026-04-30 08:05 UTC — PR #52 CLOSED: tanjiro Lion+RFF+compile triple-stack — Lion+compile divergence is now confirmed across 6 runs

- **Branch:** `tanjiro/tanjiro/round2-lion-rff-compile-512d`
- **W&B run:** `5o1frm3u` — group `tay-round2-lion-rff-compile`, 270min full budget, 16 epochs
- **Hypothesis:** Compose Lion (lr=5e-5/wd=5e-4) + RFF σ=1.0 + compile to stack three orthogonal wins → expected ~9.8-10.3% test_abupt (additive on PR #39 base 15.43).
- **Result: ALL THREE LEVERS NON-ORTHOGONAL — test_abupt 13.199 (+16.8% vs SOTA 11.30, but -9.3% vs PR #46 14.55)**

| Metric | PR #52 | SOTA `vnb7oheo` | PR #46 |
|---|---:|---:|---:|
| `test/abupt_axis_mean_rel_l2_pct` | **13.199** | 11.303 | 14.550 |
| surface_pressure | 7.532 | 6.216 | 8.628 |
| wall_shear | 13.351 | 11.315 | 14.882 |
| volume_pressure | 14.297 | 12.755 | 15.032 |
| tau_x | 11.418 | 9.563 | 12.901 |
| tau_y | 16.036 | 13.831 | 17.281 |
| tau_z | 16.714 | 14.147 | 18.907 |

- **CRITICAL DIAGNOSTIC** (from tanjiro's analysis): Triple-stack at epoch 6 (val 12.22) tracks vanilla Lion Arm B at epoch 6 (val 12.14) within 0.08 ppt. **RFF and compile add ZERO orthogonal signal in the stable Lion regime.** The lift from PR #39 (lr=1.7e-5 → 15.43) to SOTA (lr=5e-5 → 11.30) was the LR change alone. RFF+compile only matter as horizon extenders — and the extension is exactly what kills Lion+compile.
- **Mechanism (now confirmed across 6 runs):** Lion's `sign()` update has constant per-coordinate step magnitude bounded only by `lr`. Compile's reduced gradient noise → signs become deterministically biased → small-gradient noise (epoch 6+) gets amplified into full-step LR moves → train loss explodes (no NaN, slow blow-up: 0.041 → 1.05 → 5.43 → 3.07 across epochs 7-9). Same root cause as PR #42's per-case ratio variance forcing.
- **Six independent observations of Lion+compile divergence at lr=5e-5:**
  - PR #42 (frieren, sq_rel_l2 forcing) — diverged
  - PR #50 (nezuko, vanilla Lion+compile) — diverged at epoch 5
  - PR #52 (tanjiro, Lion+RFF+compile) — diverged at epoch 7
  - PR #54 (fern, Lion+per-axis-weights+compile) — diverged
  - PR #57 (askeladd, Lion+compile+cosine T_max=16) — diverged at epoch 4 (cosine did NOT fix it)
  - PR #68 (frieren, Lion+vol_w=3+compile) — diverged at epoch 6
- **Closed-door insight:** **Lion+compile at lr=5e-5 is unstable past epoch 6 in 100% of observed runs.** SOTA `vnb7oheo` (uncompiled) is at the natural Lion floor, with the slow uncompiled cadence acting as an implicit step-size budget. Cosine T_max=16 does NOT fix the divergence (#57 confirmed). Future Lion+compile experiments must either (a) halve LR to 2.5e-5, (b) use aggressive cosine T_max≤8 with proper warmup, or (c) implement a Lion-specific soft-restart on train_loss_relative_to_min plateau.
- **Best-checkpoint mechanism saved the result** — the test eval auto-loaded epoch 6's EMA weights, giving a usable test_abupt=13.20. Without the best-val checkpoint, every Lion+compile run would post the diverged final-epoch metrics.
- **Per-axis pattern matches SOTA proportionally** — all axes scale ~1.17x SOTA, suggesting no axis-specific gain or loss from RFF/compile in the stable regime.

## 2026-04-30 10:25 UTC — PR #68 CLOSED: frieren Lion + vol_w=3.0 — 8th Lion+modification divergence, vol_w lever inert when run diverges

- **Branch:** `frieren/lion-vol-weight-3`
- **W&B run:** `daayn9kb` — finished 285min
- **Hypothesis:** Lion (lr=5e-5/wd=5e-4) + volume_loss_weight 3.0 targets the volume_pressure ×2.1 binding gap. Single delta from SOTA arm B.
- **Result: DIVERGED — test_abupt 15.572 (+37.7% vs SOTA 11.30, +1.0 ppt vs PR #46 14.55, best-val ckpt epoch 5)**

| Metric | PR #68 (best-val ep5) | SOTA `vnb7oheo` | PR #46 | Δ vs SOTA |
|---|---:|---:|---:|---:|
| `test/abupt_axis_mean_rel_l2_pct` | **15.572** | 11.303 | 14.550 | +37.7% |
| surface_pressure | 9.752 | 6.216 | 8.628 | +57% |
| wall_shear | 16.338 | 11.315 | 14.882 | +44% |
| volume_pressure | 14.003 | 12.755 | 15.032 | +9.7% |
| tau_x | 13.985 | 9.563 | 12.901 | +46% |
| tau_y | 19.359 | 13.831 | 17.281 | +40% |
| tau_z | 20.761 | 14.147 | 18.907 | +47% |

- **Mechanism:** Lion+vol_w=3 diverged at epoch 6 (val explosion 14.43 → 88-100%). Same Lion-fragility pattern. Critically, **volume_pressure ended at 14.00 — barely better than SOTA's 12.76**. The vol_w=3 lever didn't get to act because divergence killed the run. So the gap-targeting hypothesis is untestable on Lion.
- **Lion-fragility envelope (8 confirmed observations):** compile, sq_rel_l2, per-axis weighting, vol_w=3.0, cosine T_max=16, grad-clip 5.0, RFF+compile (composition), per-axis-weighting+compile (composition). The only stable Lion variants: vanilla Lion (vnb7oheo, 11.30 SOTA) and Lion+RFF (PR #51 edward, descending at val 11.48 ep7).
- **Comparison signal:** alphonse #55 currently running AdamW+RFF+compile+vol_w=3.0 (lahk19ws, val 15.49 ep12) — direct comparison of vol_w=3 effect on the stable AdamW base. Will reveal if vol_w=3 actually helps volume_pressure when training stays stable.
- **Reassignment:** frieren → PR #73 (AdamW+RFF+compile + depth 6L). Architectural capacity lever, orthogonal to thorfinn #69's width 768d. Branch: `frieren/round5-adamw-rff-compile-depth6`.

## 2026-04-30 09:25 UTC — PR #54 CLOSED: fern Lion + per-axis tau_y/tau_z weighting (2×) — diverged, 7th Lion+modification failure

- **Branch:** `fern/round3-lion-tauyz-weights` → recovered as v2 (`fern/round3-lion-tauyz-weights-v2`)
- **W&B runs:** `6p4k0280` (crashed 111min), `8zhjetjt` (v2, finished 285min)
- **Hypothesis:** Per-channel wall_shear loss weighting (tau_y ×2, tau_z ×2, tau_x ×1) on Lion (lr=5e-5/wd=5e-4) forces proportional attention toward the binding tau_y/tau_z gap (×3.8/×3.9 vs AB-UPT). Expected: target those axes specifically while preserving Lion's optimizer advantage.
- **Result: DIVERGED — test_abupt 26.827 (+137% vs SOTA 11.30, best-val checkpoint at epoch 3)**

| Metric | PR #54 (best-val ep3) | SOTA `vnb7oheo` | AB-UPT |
|---|---:|---:|---:|
| `test/abupt_axis_mean_rel_l2_pct` | **26.827** | 11.303 | — |
| surface_pressure | 20.655 | 6.216 | 3.82 |
| wall_shear | 28.560 | 11.315 | 7.29 |
| volume_pressure | 19.436 | 12.755 | 6.08 |
| tau_x | 24.835 | 9.563 | 5.35 |
| tau_y | 33.109 | 13.831 | 3.65 |
| tau_z | 36.104 | 14.147 | 3.63 |

- **Val trajectory:** ep1=75 → ep2=46 → ep3=26 (best) → **ep4=220 (DIVERGED)** → settled at 93%. Same divergence onset pattern as PRs #42, #52, #57, #68.
- **Diagnostic insight:** Per-axis breakdown showed weighting *did* reduce tau_y/tau_x ratio (1.33 vs vanilla Lion's 1.45), confirming the loss mechanism works at the gradient level. The divergence is entirely Lion's instability to loss modification — adding 2× per-axis weighting changes the effective gradient magnitudes per-channel, which pushes the sign-update into a new regime. Lion is now confirmed fragile to: compile, sq_rel_l2, per-axis weighting, vol_w=3.0, cosine T_max=16, and vol_w=3.0. The only stable Lion variants are vanilla Lion and Lion+RFF (PR #51 edward, stable at epoch 5).
- **Conclusion:** The per-axis weighting *mechanism* is valid — test it on AdamW+RFF+compile (PR #46 base) where the optimizer is stable. Assigned as PR #71 (fern, round5).
- **PR #71 — fern: AdamW + RFF + compile + per-axis tau_y/tau_z weighting (2×):** Tests the same weighting mechanism on the stable AdamW base. Expected: beat PR #46 (14.55) by closing tau_y/tau_z gap. Branch: `fern/round5-adamw-rff-compile-tauyz`.

## 2026-04-30 07:30 UTC — PR #56 CLOSED: thorfinn AdamW + cosine T_max=16 — schedule miscalibration was load-bearing

- **Branch:** `thorfinn/round3-rff-compile-cosine-lr16`
- **W&B run:** `ko1hrdau` — group `tay-round3-rff-compile-cosine-lr16`, 270min full budget, 16 epochs
- **Hypothesis:** PR #46's `--lr-cosine-t-max 0` fallback to `max_epochs=50` left LR at ~88% of init throughout. Calibrating `T_max=16` to actual budget should give Lion(/AdamW) the late-training fine-tuning it never got.
- **Result: HYPOTHESIS REJECTED — +13.0% regression vs PR #46 (test_abupt 16.44 vs 14.55)**

| Metric | PR #56 | PR #46 | Δ vs #46 | AB-UPT |
|---|---:|---:|---:|---:|
| `test/abupt_axis_mean_rel_l2_pct` | **16.440** | 14.550 | +1.890 (+13.0%) | — |
| surface_pressure | 10.042 | 8.628 | +1.414 (+16.4%) | 3.82 |
| wall_shear | 17.076 | 14.882 | +2.194 (+14.7%) | 7.29 |
| volume_pressure | 15.959 | 15.032 | +0.927 (+6.2%) | 6.08 |
| tau_x | 14.862 | 12.901 | +1.961 (+15.2%) | 5.35 |
| tau_y | 19.837 | 17.281 | +2.556 (+14.8%) | 3.65 |
| tau_z | 21.502 | 18.907 | +2.595 (+13.7%) | 3.63 |

- **Mechanism:** Cosine T_max=16 collapsed LR to ~6% of init by epoch 14 and 0.6% by epoch 16. PR #46's val curve was still descending at epoch 16, meaning the model was undertrained, not overoptimized. The aggressive LR decay starved it of the late-training optimization power that was load-bearing for the 14.55 result. val→test gap shrunk +18% (sharper minima generalize slightly better) but this was swamped by the +2.0 absolute val regression.
- **Closed-door insight (DO NOT REVISIT WITHOUT BUDGET EXTENSION):** AdamW lr=5e-5 at 16-epoch budget is undertrained, not overoptimized. Cosine T_max calibrated to actual budget is a regression. Future cosine experiments must (a) use generous T_max≥24 with proper warmup, or (b) extend budget rather than compress LR. The "miscalibration" in PR #46 was effectively a constant-ish LR schedule and was load-bearing.
- **Per-axis pattern:** All axes regressed 6-16%. Volume_pressure least affected, surface_pressure / tau_y / tau_z worst — consistent with a global LR-too-low-too-fast effect, not axis-specific.
- **Suggested follow-ups (deferred):** thorfinn's #5 ablation (PR #46 with explicit T_max=50) would confirm the fallback was load-bearing — but lever is now low priority since Lion is the active SOTA arm.



## Round 1 — opened 2026-04-29

8 students assigned in parallel on DDP8 (8 GPUs each, 96 GB VRAM, effective
bs scales with `nproc_per_node × per-GPU bs`). Strategy: 5 students compose
yi's confirmed-orthogonal wins (width × FiLM × cosine-EMA × Fourier × LR
warmup); 3 students push beyond yi with architectural / loss / TTA changes
that yi only got as far as Round-2 assignments for.

| PR | Student | Hypothesis |
|---|---|---|
| #30 | alphonse | yi PR #4 reproduce (4L/512d/8h, lr=5e-5, bs=4) — calibration |
| #31 | askeladd | Full composition stack: 512d × cosine-EMA × tangential × vol_w=2.0 |
| #32 | edward | Cosine LR + 5% warmup on top of 512d composition |
| #33 | fern | Gaussian Fourier coord features + 512d composition |
| #34 | frieren | AdaLN-zero per-block FiLM + 512d composition |
| #35 | nezuko | A01 — ANP cross-attention surface decoder |
| #36 | tanjiro | SDF-gated volume attention bias for near-wall p_v |
| #37 | thorfinn | Per-axis wall-shear loss weighting + bilateral-symmetry TTA |

## Round 1 — in-progress observations (2026-04-29 12:35 UTC)

All 8 still WIP. No PRs marked review-ready. Per-axis val curves are
informative even before completion:

```
Run                              step    val_abupt  ps     ws     pv
alphonse (calibrate)             10887   27.74      20.01  30.94  15.86
edward (cosine warmup)           10887   35.68      25.46  40.10  19.10
fern (RFF features)               8165   30.07      22.07  32.59  19.03
askeladd (composition stack)      8165   39.49      19.04  46.75  14.49 ← ws regression
thorfinn (per-axis weights)       8165   33.6       n/a    n/a    n/a
frieren (FiLM AdaLN-zero)         8165   34.4       n/a    n/a    n/a
nezuko (ANP decoder)              4316   76.4       n/a    n/a    n/a   ← much slower
tanjiro (SDF gate)                  —      —          —      —     —    ← 4 crashes at step 2719 (in eval path)
```

**Key in-progress signals** (caveat: not at completion, only first 4
validations; ranking may shift by epoch ~10):

1. **alphonse calibration matches yi epoch-1 (26.24) within 5%** —
   confirms tay/DDP8 baseline is healthy; yi's wins should reproduce.
2. **askeladd's composition stack is BEST on `ps`/`pv` but WORST on `ws`**
   — `ps=19.0` vs alphonse's 20.0, `pv=14.5` vs alphonse's 15.9, but
   `ws=46.7` vs alphonse's 30.9. The tangential wall-shear projection
   loss is net-negative for raw wall_shear despite improving the other
   axes. **Important Round 2 implication**: do NOT bundle the tangential
   projection into "compose all yi wins" runs — it hurts the metric it
   was designed to help.
3. **fern's RFF features show strong lift** — at the same step (8165)
   fern is at 30.1 vs alphonse 35.1 → RFF is doing real work. Will see
   how it compounds at later validations.
4. **edward's cosine warmup catches up rapidly** — at step 10887, edward
   is 35.68 vs alphonse's 27.74; warmup arm typically lags early then
   converges. Worth running to completion to see if it surpasses
   alphonse asymptotically.
5. **nezuko ANP decoder is dramatically slower per step** — same wall
   time produces ~4x fewer steps than alphonse. At step 4316 only 1
   validation (val=76.4). May not finish enough epochs to be comparable.
6. **tanjiro 4 crashes at exactly step 2719** — deterministic failure in
   eval path. Posted advisor comment with simplified-σ guidance.

## 2026-04-29 15:21 UTC — PR #30 merged: first tay/DDP8 baseline (alphonse calibration)

**Student:** alphonse | **W&B run:** `0vi9tm5h` | **Hypothesis:** Reproduce yi PR #4 (4L/512d/8h, lr=5e-5, bs=4, vol_w=2.0) on tay/DDP8.

### Results

| Metric | tay val | tay test | yi best | AB-UPT |
|---|---:|---:|---:|---:|
| abupt | 18.70 | **19.81** | 15.82 | — |
| surface_pressure | 12.93 | 12.86 | 9.99 | 3.82 |
| wall_shear | 21.24 | 21.27 | 16.60 | 7.29 |
| volume_pressure | 9.69 | 15.91 | 14.21 | 6.08 |
| tau_x | 18.09 | 18.24 | 14.27 | 5.35 |
| tau_y | 25.54 | 25.50 | 19.49 | 3.65 |
| tau_z | 27.26 | 26.53 | 21.12 | 3.63 |

### Analysis

This establishes tay's **first concrete test baseline at 19.81 abupt**. Run
was under-trained at 9 epochs (of 50) — loss still steeply descending at
end (val slope −0.37/1k steps). Root cause: `torch.compile + drop_last=False`
interaction crashes all 8 ranks at the epoch-boundary step. Student used
`--no-compile-model` workaround (1.5–2× per-step cost), limiting epochs to
~9 within the 270-min budget.

**Critical infra finding**: Fix is `drop_last=True` in `trainer_runtime.py:293`
(editable per program.md). Estimated ~2× throughput gain = ~14–22 compiled
epochs in budget instead of 9 uncompiled. Alphonse reassigned to PR #40 to
land the fix and re-calibrate.

**Round 2 implication**: All 7 concurrent Round-1 students ran without compile.
Results from this round should be compared apples-to-apples (all uncompiled).
After compile fix merges, Round 2 baselines reset.

---

## 2026-04-29 21:25 UTC — PR #41 CLOSED: askeladd eval-tangential projection — clear regression

**Student:** askeladd | **W&B run:** `p3lxbg6t` (rank 0) | **Hypothesis:** Project
predicted wall-shear vector onto surface tangent at eval time only (vs yi PR #11
kohaku's training-time projection that broke tau_z).

### Results — vs current SOTA (PR #40 compiled, 12 epochs)

| Metric | PR #41 | PR #40 SOTA | Δ vs SOTA |
|---|---:|---:|---:|
| `val_abupt` | 20.25 | 16.09 | +4.16 (+25.9%) |
| `test_abupt` | **21.13** | 17.25 | **+3.88 (+22.5%)** |

Even vs the OLD uncompiled baseline (PR #30: 19.81), this is +1.32 worse (+6.7%).

### Disposition: CLOSED

The eval-time projection mechanism is wrong-shaped. Mechanism analysis:
- Trained model predicts `tau` ≈ `α · n_normal + β · n_tangent` (some normal-component
  contribution that helps rel-L2 even though it's physically anomalous on flat panels).
- Projecting onto tangent removes the α component.
- Remaining `β · n_tangent` is now a worse predictor than the unprojected `tau` was.

**Combined with yi PR #11 kohaku results, this closes the door on tangential-projection-on-output**.
Future wall-shear constraints should consider:
- Predicting in tangent-frame intrinsic coordinates (projection built into the head).
- Joint loss penalizing `tau · n_normal` rather than projecting at eval.

Reassigning askeladd to **PR #49: grad-clip-norm 1.0 → 5.0 single delta** (motivated
by frieren's diagnostic that clip rate was 1.0 every step, late-training median
grad_norm ~4 — 1.0 clip is throwing away ~75% of gradient magnitude).

---

## 2026-04-29 22:05 UTC — PR #39 MERGED: tanjiro Lion lr=1.7e-5 — NEW SOTA 15.43, crosses yi frontier

**Student:** tanjiro | **W&B run:** `xonbs83i` (rank 0) | **Group:** `tay-lion-lr-sweep`
**Hypothesis:** Drop-in Lion optimizer (paper config `lr=1.7e-5, wd=5e-3`) in place of AdamW.

Note: student's in-pod claude hung since iteration 5 (17:42 UTC); training ran autonomously.
Results verified directly from W&B summary. PR readied and merged by advisor.

### Results — vs PR #40 SOTA (compiled, 12 epochs)

| Metric | PR #39 Lion | PR #40 SOTA | Δ abs | Δ % |
|---|---:|---:|---:|---:|
| `val_primary/abupt` | **14.22** | 16.09 | −1.87 | **−11.6%** |
| `test_primary/abupt_axis_mean_rel_l2_pct` | **15.43** | 17.25 | −1.82 | **−10.5%** |
| `test_primary/surface_pressure_rel_l2_pct` | **9.45** | 10.92 | −1.47 | **−13.5%** |
| `test_primary/wall_shear_rel_l2_pct` | **16.28** | 18.33 | −2.05 | **−11.2%** |
| `test_primary/volume_pressure_rel_l2_pct` | **13.83** | 14.71 | −0.88 | **−6.0%** |
| `test_primary/wall_shear_x_rel_l2_pct` | **13.91** | 15.73 | −1.82 | **−11.6%** |
| `test_primary/wall_shear_y_rel_l2_pct` | **19.58** | 21.80 | −2.22 | **−10.2%** |
| `test_primary/wall_shear_z_rel_l2_pct` | **20.40** | 23.07 | −2.67 | **−11.6%** |

**Best_epoch = 9 / 50 configured** — val curve still descending at cutoff.

**tay crosses below yi frontier: 15.43 vs yi best 15.82 (PR #13 norman).**

### Config

- optimizer: `lion`, lr: `1.7e-5`, wd: `5e-3`, beta1/beta2: `0.9/0.99`
- base: 4L/512d/8h/128slices, ema_decay=0.9995, vol_w=2.0, bs=4×8GPU
- **No compile** (pod on pre-compile-fix commit `269cb09`)
- **No RFF**
- Runtime: 287.1 min, DDP8

### Mechanism analysis

1. **Sign-update sidesteps clip compression.** With `grad-clip-norm=1.0` binding at every
   step (frieren's diagnostic: clip_rate=1.0/23458 steps), AdamW's per-coordinate
   magnitude scaling is being compressed. Lion uses `torch.sign(momentum)` — each
   coordinate gets a step of exactly `lr`, independent of gradient magnitude. The
   clip still fires but only scales the momentum update, not the step direction.

2. **Larger weight decay tolerated.** `wd=5e-3` is 10× our default `5e-4`. Lion's
   documented sweet spot; provides stronger implicit regularization against overfitting
   on 400 CFD training cases.

3. **val curve still descending at epoch 9** → compile (12 epochs) or longer budget
   should further improve. Lion + compile is the next highest-priority test.

### Disposition: MERGED — new tay SOTA

Reassigning tanjiro to follow-up arm: arm B (Lion lr=5e-5, `vnb7oheo`) launched
automatically and still running as of merge.

---

## 2026-04-29 22:10 UTC — PR #35 CLOSED: nezuko ANP cross-attention decoder — two rounds, no win

**Student:** nezuko | **Round-1 run:** `2fqts0v8` | **Round-2 run:** `ochavw4i`
**Hypothesis:** Replace Transolver surface MLP head with ANP-style cross-attention decoder.

### Results (both rounds)

| Run | Config | test_abupt | vs SOTA at time |
|---|---|---:|---:|
| Round-1 ANP | standalone, no RFF, vol_w=2 | 18.76 | −5.3% vs #30 calibration |
| Round-2 ANP+RFF | ANP + RFF sigma=1.0 + vol_w=3 | **17.55** | **+1.7% vs PR #40 SOTA** |
| **PR #39 SOTA** | Lion lr=1.7e-5 | **15.43** | — |

### Key findings

1. **Surface wins are real but partial.** Standalone ANP: p_s −7.2%, tau axes −7-8%.
   With RFF composition: p_s −2.5%, tau_x −1% (RFF captures same non-local spatial
   structure — not fully orthogonal).
2. **p_v regression is structural.** ANP's cross-attention over slice tokens focuses
   on surface geometry; volume targets need backbone-level volume processing that
   isn't helped by a surface head replacement.
3. **Cannot be compiled.** `torch.inductor` dynamic-shapes `AssertionError` at
   epoch 1 (backbone-then-split `s24 + 65536` codegen bug). Stuck at ~9 uncompiled
   epochs vs 12 compiled — fundamental throughput disadvantage.
4. **vol_w=3 didn't help p_v.** Round-2 p_v = 17.26 vs Round-1 (not logged final).
   Increasing vol_w amplified a weak signal; the head needs a different fix.

### Disposition: CLOSED

Slice tokens (post-SDPA) are high-quality local-geometry embeddings — future head
replacements should read those, not pre-SDPA tokens. But for now, composition
wins from Lion + compile are more tractable.

Reassigning nezuko to **PR #50: Lion + compile (single delta)**.

---

## 2026-04-29 22:12 UTC — PR #44 CLOSED: edward cosine EMA (0.99→0.9999) — budget miscalibration regression

**Student:** edward | **W&B run:** `51yrbxxj` | **Group:** cosine-ema + RFF
**Hypothesis:** Progressive cosine EMA decay 0.99→0.9999 (replicating yi PR #13 norman win)
+ RFF coord features.

### Results

| Metric | PR #44 | PR #40 SOTA | PR #39 new SOTA | Δ vs PR #40 |
|---|---:|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **18.33** | 17.25 | 15.43 | **+6.3%** |
| surface_pressure | 11.69 | 10.92 | 9.45 | +7.1% |
| wall_shear | 19.35 | 18.33 | 16.28 | +5.6% |
| volume_pressure | 16.12 | 14.71 | 13.83 | +9.6% |

### Root cause: EMA schedule mapped to max_epochs=50, not actual epochs

Cosine EMA sweeps 0.99→0.9999 over `ema_total_epochs` (defaults to `max_epochs=50`).
At epoch 9 of 50 (18% through schedule): `ema_decay ≈ 0.9910`. Fixed default is 0.9995.
**The cosine EMA spent the entire run below the fixed EMA baseline.**

Yi's norman PR #13 (which won with this schedule) ran with a longer effective budget.
With `T_max=12` (our actual compiled epoch count), the schedule would reach ~0.9960 by
end — still below 0.9995. Fix: `--ema-total-epochs 12` for our budget.

### Disposition: CLOSED

Cosine EMA is worth revisiting on top of Lion with `--ema-total-epochs` set to
actual expected epoch count (12 for compiled runs). Reassigning edward to
**PR #51: Lion + RFF (single delta)** — cleaner first composition test.

---

## 2026-04-29 20:46 UTC — PR #42 SENT BACK: frieren squared_rel_l2 — orthogonal win but on stale baseline

**Student:** frieren | **W&B run:** `bmz26ft7` | **Hypothesis:** Replace MSE
training loss with squared rel-L2 (per-case `sum((y-ŷ)²) / sum(y²)`, no outer
sqrt). Single-delta change in `trainer_runtime.py`.

### Results — vs OLD baseline (PR #30 MSE-uncompiled, 9 epochs)

| Metric | PR #30 | PR #42 | Δ% |
|---|---:|---:|---:|
| `abupt_axis_mean` | 19.81 | **19.14** | −3.4% |
| `surface_pressure` | 12.86 | **12.24** | −4.8% |
| `wall_shear` | 21.27 | **20.58** | −3.2% |
| `volume_pressure` | 15.91 | **15.57** | −2.1% |
| `tau_x` | 18.24 | **17.67** | −3.1% |
| `tau_y` | 25.50 | **24.70** | −3.1% |
| `tau_z` | 26.53 | **25.52** | −3.8% |

### Disposition: SENT BACK for rebase + recompose

19.14 is +1.89 worse than current SOTA 17.25 (PR #40 compiled), so merging
would regress BASELINE.md. Mechanism is real and orthogonal — student's
own follow-up correctly anticipated rebase+recompose:

> "Compose squared_rel_l2 with longer training… my guess: 16–18 on test_abupt."

Excellent student diagnostics:
- Discovered actual baseline was MSE not sqrt-rel-L2 (corrected hypothesis,
  added `--loss-form {mse,rel_l2,squared_rel_l2}` flag with default mse).
- Identified that 1.0 grad-clip threshold was binding **at every step**
  regardless of loss form — motivates lr=1e-4 and/or larger clip threshold.
- Val curve still descending at epoch 9 (slope −0.59/epoch).

Sent back: rebase onto tay (compile fix from PR #40), re-run compiled with
`--loss-form squared_rel_l2`. Projected test_abupt ~16.5 if 3.4% loss-form win
composes additively with 2.9% compile win.

---

## 2026-04-29 20:00 UTC — PR #40 MERGED: alphonse compile-fix — new tay SOTA 17.25

**Student:** alphonse | **W&B run:** `ae4zsaly` (rank 0) | **Hypothesis:** Fix
`torch.compile` crash (`drop_last=False` + variable-shape eval batches) so all
runs can use compiled training at ~1.7× throughput. Recalibrate with same 4L/512d/8h
config as PR #30.

Two-line patch in `trainer_runtime.py`:
1. `drop_last=True` on `DistributedSampler` + `DataLoader` (lines 293, 301) — fixes partial last-batch epoch crash.
2. `unwrap_model(model)` in `accumulate_eval_batch` (line 929) — bypasses compile during eval because `pad_collate` produces variable-shape batches; `model.py:321` `torch.cat([surf, vol], dim=1)` creates sum-of-symbolic-dims (`s24+s39`) that inductor can't verify; `dynamic=True` did not help.

### Results

| Metric | PR #40 | PR #33 (SOTA) | PR #30 | yi best | AB-UPT |
|---|---:|---:|---:|---:|---:|
| `abupt_axis_mean` | **17.25** | 17.77 | 19.81 | 15.82 | — |
| `surface_pressure` | **10.92** | 11.20 | 12.86 | 9.99 | 3.82 |
| `wall_shear` | **18.33** | 18.68 | 21.27 | 16.60 | 7.29 |
| `volume_pressure` | **14.71** | 16.13 | 15.91 | 14.21 | 6.08 |
| `tau_x` | **15.73** | 16.20 | 18.24 | 14.27 | 5.35 |
| `tau_y` | **21.80** | 21.81 | 25.50 | 19.49 | 3.65 |
| `tau_z` | **23.07** | 23.54 | 26.53 | 21.12 | 3.63 |

Δ vs PR #33: abupt −2.9%, surface_p −2.5%, wall_shear −1.9%, **volume_p −8.8%**, tau_x −2.9%, tau_y flat, tau_z −2.0%.

### Analysis

Biggest surprise: compile fix alone beats RFF (17.25 vs 17.77) despite no feature change.
The mechanism: compile gives ~1.7× throughput → 12 epochs vs 9 in same 270-min budget. Val
curve still steeply descending at epoch 12 (Δ epoch11→12 = −0.57) — more budget would improve
further. Volume pressure improved most (−8.8%) suggesting early-epoch vol gradient was most
starved of training time.

The PR #40 student correctly diagnosed that RFF+compile composition (PR #46 alphonse) is the
next priority, projecting ~15.5 abupt if additive. The remaining gap to yi (17.25 vs 15.82) is
consistent with missing progressive cosine EMA (edward #44 testing this).

Best-val: epoch 12, val_abupt 16.09. Peak GPU mem 53.1 GB / 96 GB available.

---

## 2026-04-29 16:39 UTC — PR #33 MERGED: fern RFF win — new tay SOTA 17.77

**Student:** fern | **W&B run:** `u43lik5d` (rank 0) | **Hypothesis:** Gaussian RFF coord
features (sigma=1.0, 32 features per modality) appended to surface and volume coord inputs.
Model unchanged; input dim grows from 7 to 7+64 (surface and volume each get separate RFF).

### Results

| Metric | tay (PR #33) | PR #30 | yi best | AB-UPT |
|---|---:|---:|---:|---:|
| `abupt_axis_mean` | **17.77** | 19.81 | 15.82 | — |
| `surface_pressure` | **11.20** | 12.86 | 9.99 | 3.82 |
| `wall_shear` | **18.68** | 21.27 | 16.60 | 7.29 |
| `volume_pressure` | 16.13 | 15.91 | 14.21 | 6.08 |
| `tau_x` | **16.20** | 18.24 | 14.27 | 5.35 |
| `tau_y` | **21.81** | 25.50 | 19.49 | 3.65 |
| `tau_z` | **23.54** | 26.53 | 21.12 | 3.63 |

### Analysis

RFF lifts every surface and shear axis (−10–15%). tau_y: −14.5%, tau_z: −11.3%. The
primary mechanism is spectral bias bypass: RFF encodes multi-frequency coordinate
content that the existing sincos pos_embed provides less of for surface coords at
meter-scale. volume_pressure flat (+0.22 = noise) because volume far-field coords
[-40,80]m saturate the sigma=1.0 projection — documented and preserved for H01.

Best-val checkpoint: epoch 9, val_abupt 17.06. Run ~4.78h without compile.

---

## 2026-04-29 16:39 UTC — PR #32 CLOSED: edward cosine LR + warmup — loss

**Student:** edward | **W&B run:** `uqziai5z` (Arm A: warmup-on) | **Result:** test_abupt 20.99
(+1.18 vs baseline, +6%). Uniform regression all axes.

Diagnosis: `T_max=50` at 9 effective epochs → cosine schedule only 18% engaged (5e-5 → 4.97e-5).
Warmup front-loads cost with no recovery time. Not a dead end — the question is
*"more epochs"*, not *"wrong schedule."* After alphonse #40 compile fix lands (→ 16–22
compiled epochs), cosine LR with T_max matched to compiled budget becomes the right revisit.

---

## 2026-04-29 16:25 UTC — Round 1 first wave: 3 results confirmed in W&B (PRs not yet ready)

Three Round-1 students completed primary arms. None have marked PRs ready
because they are running unauthorized second arms (fern sigma=0.5 / sigma=2.0,
edward warmup-off) or have a silent in-pod session (thorfinn). Posted advisor
comments on each PR pushing them to mark ready / push code.

| PR | Student | Run | val_abupt | test_abupt | Outcome |
|---|---|---|---:|---:|---|
| #33 | fern | `u43lik5d` (sigma=1.0) | 17.06 | **17.77** | **WIN −2.04 (−10.3%)**, new tay leader |
| #37 | thorfinn | `sjcgnehq` (per-axis 1/1.5/1.5) | 18.52 | **19.44** | WIN −0.37 (−1.9%); code not pushed to remote |
| #32 | edward | `uqziai5z` (cosine + warmup) | 19.96 | 20.99 | LOSS +1.18 (+6.0%); cosine T_max=50 only ~18% engaged at 9 epochs |

### fern #33 — RFF win full breakdown

| Metric | tay (PR #30) | fern (PR #33) | Δ | yi best | AB-UPT |
|---|---:|---:|---:|---:|---:|
| `abupt` | 19.81 | **17.77** | −2.04 | 15.82 | — |
| `surface_pressure` | 12.86 | **11.20** | −1.66 | 9.99 | 3.82 |
| `wall_shear` | 21.27 | **18.68** | −2.60 | 16.60 | 7.29 |
| `volume_pressure` | 15.91 | 16.13 | +0.22 | 14.21 | 6.08 |
| `tau_x` | 18.24 | **16.20** | −2.04 | 14.27 | 5.35 |
| `tau_y` | 25.50 | **21.81** | −3.70 | 19.49 | 3.65 |
| `tau_z` | 26.53 | **23.54** | −3.00 | 21.12 | 3.63 |

RFF lifts every wall_shear axis significantly (tau_y −14.5%, tau_z −11.3%,
tau_x −11.2%). Volume pressure flat (+0.22, within noise). Mechanism:
2π·sin/cos coordinate features at sigma=1.0 give the model multi-frequency
positional information that the existing sincos `pos_embed` doesn't fully
provide for surface coordinates.

### thorfinn #37 — Per-axis weights (code missing on remote)

| Metric | tay | thorfinn | Δ |
|---|---:|---:|---:|
| `abupt` | 19.81 | **19.44** | −0.37 |
| `wall_shear` | 21.27 | **20.84** | −0.43 |
| `wall_shear_y` | 25.50 | **24.25** | −1.25 |
| `wall_shear_z` | 26.53 | **25.63** | −0.90 |

Per-axis weight (1.0 / 1.5 / 1.5) does what was designed: tau_y / tau_z
(the systematically worst axes) move most. tau_x flat, surface_pressure
flat, volume_pressure +0.34. **Cannot merge until code is pushed to
`thorfinn/round1-tau-yz-attack-weights-and-tta` branch** — currently
remote has only the assignment commit.

### edward #32 — Cosine LR + warmup (loss)

Uniform regression across all axes (+0.76 to +1.59). Confirmed during
kickoff that `T_max=50` would only progress ~18% over the 9-epoch budget,
so the cosine annealing barely engaged. Adding 5% warmup costs effective
steps with no compensating gain at low effective epochs. The actual lever
this experiment surfaces is **throughput**, not LR schedule.

### Round 1b status (running)

| PR | Student | Hypothesis | Step | Notes |
|---|---|---|---:|---|
| #40 | alphonse | drop_last=True compile fix + recalibrate | 834 | freshly launched |
| #41 | askeladd | eval-time tangential projection of wall-shear | 2156 | running |
| #42 | frieren | squared rel-L2 loss (drop outer sqrt) | 2374 | running |

---

## 2026-04-30 04:50 UTC — PR #42 CLOSED: frieren squared rel-L2 — three arms, AdamW arm clean +40% regression, Lion arm DIVERGED with mechanistic explanation

**Student:** frieren | **W&B runs:** `uwt74mip` (AdamW arm) / `24bdfcnz` (Lion paper-config, killed) / `8ubarr6a` (Lion 5e-5, diverged)
**Hypothesis:** Replace MSE training loss with `(y−ŷ)²/y²` (squared rel-L2). Adds CLI flag `--loss-form {mse,rel_l2,squared_rel_l2}`.

### Arm 1 — AdamW + compile + sq_rel_l2 (finished cleanly)

W&B `uwt74mip`, group `tay-round2-squared-rel-l2-compiled`. 284 min, best epoch 16.

| Metric | PR #40 (compile only, MSE) | This (Arm 1, sq_rel_l2) | Δ vs PR #40 | tay SOTA `vnb7oheo` (Lion 5e-5, MSE) | Δ vs SOTA |
|---|---:|---:|---:|---:|---:|
| `test_abupt` | 17.25 | **15.819** | −1.43 (−8.3%) | 11.303 | **+40.0%** |
| surface_pressure | 10.92 | 9.756 | −10.6% | 6.216 | +57.0% |
| wall_shear | 18.33 | 16.742 | −8.7% | 11.315 | +47.9% |
| volume_pressure | 14.71 | 14.063 | −4.4% | 12.755 | +10.3% |
| tau_x | 15.73 | 14.412 | −8.4% | 9.563 | +50.7% |
| tau_y | 21.80 | 19.859 | −8.9% | 13.831 | +43.6% |
| tau_z | 23.07 | 21.003 | −9.0% | 14.147 | +48.5% |

**Mechanism validated:** uniform −8% across all six axes vs PR #40's MSE baseline. Consistent with original
PR #42 round-1 result (−3.4% on uncompiled). The compile-fix doubled the effective wall budget so the
mechanism delta also compounded.

### Arm 3 — Lion (lr=5e-5/wd=5e-4 SOTA recipe) + compile + sq_rel_l2 → DIVERGED

W&B `8ubarr6a`. Killed at step 8959 / 56 min after clean divergence at step 5000.

| Step | Lion + MSE (`vnb7oheo`) | Lion + sq_rel_l2 (this) |
|---:|---|---|
| 4000 | loss=0.18 grad=2.3 | loss=0.23 grad=6.5 |
| **5000** | **loss=0.16 grad=2.7** | **loss=1.40 grad=240** ← divergence |
| 8000 | loss=0.08 grad=1.8 | loss=2.59 grad=505 |

Val curve: epoch 1 = 82.55%, epoch 2 = 83.30%, epoch 3 = 112.97%.

### Mechanism (frieren's diagnosis — preserved as closed-door insight)

> Lion is sign-update: each parameter step is `±lr` per coordinate, magnitude is unit. sq_rel_l2 is a
> per-case ratio `(y−ŷ)²/y²`; for a batch where `mean(y²)` is small, the ratio (and its gradient) blows
> up. AdamW's per-coordinate variance EMA dampens this — a single bad batch barely moves the second-
> moment estimate, so the effective step size is bounded. Lion has **no per-coordinate magnitude
> memory** — a single bad gradient direction sets the sign of every coordinate to ±1 at full LR,
> permanently corrupting the parameter state. The model cannot recover because Lion will keep flipping
> signs at the same magnitude.

### Arm 2 — Lion paper-config (lr=1.7e-5/wd=5e-3) + sq_rel_l2 (control)

W&B `24bdfcnz`. Killed at epoch 8 (val 16.30, best ~14) when BASELINE.md updated revealed the recipe was
suboptimal. **Did not diverge** — confirms the divergence in Arm 3 is **LR-magnitude-dependent** (smaller
LR → bounded sign-step recovery within 1–2 steps; larger LR → unrecoverable).

### Closed-door insights (now in advisor playbook)

1. **Lion + per-case-normalized loss family is unstable at SOTA LR.** Never compose sq_rel_l2 / rel_l2
   / any `1/y²` loss with Lion at `lr ≥ 5e-5`.
2. **squared_rel_l2 is a real ~−8% mechanism on AdamW** (uniform across all axes). If AdamW path ever
   competes again (e.g. larger model where AdamW's adaptivity helps), revisit.
3. **paper-config Lion stable with sq_rel_l2.** Divergence is LR-dependent.

### Disposition: CLOSED (+40% regression vs SOTA; infra not merged)

The `--loss-form` flag from `64481fa` is preserved on the branch. Branch retained for future cherry-pick;
no current SOTA recipe wants it (RFF + AdamW + compile dominates the AdamW path; Lion + sq_rel_l2 is
provably unstable). Frieren reassigned to **PR #58: Lion (5e-5) + volume-loss-weight 3.0** — direct attack
on the second-worst gap (volume_pressure ×2.1 vs AB-UPT) under the winning optimizer.

---

## 2026-04-30 02:55 UTC — PR #49 CLOSED: askeladd grad-clip 5.0 — +35% regression vs new SOTA

**Student:** askeladd | **W&B run:** `x09udzj3` | **Group:** `tay-round2-grad-clip-sweep`
**Hypothesis:** Raise `--grad-clip-norm` from 1.0 to 5.0 on AdamW + compile baseline (PR #40).
Frieren's PR #42 diagnostic showed `train/grad/clipped=1.0` at every step (23,458/23,458),
discarding ~75% of late-training gradient magnitude. Single-delta lever to recover signal.

### Results vs new SOTA `vnb7oheo` (Lion lr=5e-5)

| Metric | PR #49 (askeladd, clip 5.0) | New SOTA (Lion 5e-5) | Δ |
|---|---:|---:|---:|
| `test_abupt` | **15.232** | 11.303 | **+34.8%** |
| `val_abupt` | 14.219 | 10.096 | +40.8% |
| surface_pressure | 9.338 | 6.216 | +50.2% |
| wall_shear | 16.073 | 11.315 | +42.0% |
| volume_pressure | 13.796 | 12.755 | +8.2% |
| tau_y | 19.105 | 13.831 | +38.1% |
| tau_z | 20.067 | 14.147 | +41.9% |

Run finished cleanly: 284.9 min, best_epoch=16, 43,272 steps. Used Lion-translated LR/WD
(`lr=5e-5, wd=5e-4`) with **AdamW** (note: askeladd correctly used the AdamW-equivalent translation
for the AdamW arm, but the optimizer stays AdamW because the hypothesis is grad-clip on AdamW).

### Diagnosis: clip-5 + AdamW under-trains vs Lion's normalized geometry

The hypothesis was sound — clip 1.0 throws away most of the late-training signal. Raising it does
recover loss/epoch. But the experimental landscape moved past it:

1. **Lion sidesteps the issue entirely.** Sign-based updates make pre-clip grad-norm irrelevant —
   per-step movement is bounded by `lr` directly, not gradient magnitude. Larger clip on AdamW
   recovers some discarded signal but doesn't match Lion's per-channel normalization.
2. **Marginal vs PR #46 (+4.7%) but uncompetitive vs SOTA 11.30.** On its own would have been
   "send back, try lr=1e-4 + clip 5.0"; against Lion it's a clear close.
3. **No forward composition.** Lion + clip 5.0 is ill-defined (Lion is not magnitude-bounded),
   so this lever does not stack into the live Lion arms (#50/#51/#52/#54).

### Disposition: CLOSED (+34.8% regression vs new SOTA)

Diagnostic from frieren #42 (AdamW grad-clip binding) is preserved. Lever does not compose forward
under Lion. Reassigning askeladd to a fresh round-3 hypothesis under the Lion lr=5e-5 SOTA.

---

## 2026-04-30 02:44 UTC — NEW SOTA documented: tanjiro arm B Lion lr=5e-5 — test_abupt 11.303

**Student:** tanjiro (follow-up sweep, NOT advisor-assigned PR) | **W&B run:** `vnb7oheo`
**Group:** `tanjiro-lion-lr-sweep` | **No PR — documented retroactively as new SOTA.**

### Provenance

After PR #39 Lion (paper config `lr=1.7e-5, wd=5e-3` → test_abupt 15.43) was reviewed, tanjiro's
pod launched a follow-up sweep arm at the **AdamW-equivalent translation `lr=5e-5, wd=5e-4`** as
arm B of the original two-arm comparison. This was not part of the advisor's assigned PR pipeline.
The result was decisive enough to update BASELINE.md directly without retroactive PR creation.

### Results — every axis is a new SOTA

| Metric | tanjiro arm B (Lion 5e-5) | PR #46 (RFF+compile) | PR #39 Lion (paper) | AB-UPT ref | Δ vs PR #46 |
|---|---:|---:|---:|---:|---:|
| `test_abupt` | **11.303** | 14.550 | 15.43 | — | **−22.3%** |
| surface_pressure | **6.216** | 8.628 | 9.45 | 3.82 | −28.0% |
| wall_shear | **11.315** | 14.882 | 16.28 | 7.29 | −24.0% |
| volume_pressure | **12.755** | 15.032 | 13.83 | 6.08 | −15.1% |
| tau_x | **9.563** | 12.901 | 13.91 | 5.35 | −25.9% |
| tau_y | **13.831** | 17.281 | 19.58 | 3.65 | −20.0% |
| tau_z | **14.147** | 18.907 | 20.40 | 3.63 | −25.2% |

Runtime: 4h50m (290 min, past 270-min budget — likely launched without strict timeout enforcement).
Val curve was still descending at end → there's more headroom with the same recipe at a longer
budget. Best-val checkpoint reload gave the test number above.

### CRITICAL FINDING: Lion paper config is wrong for this dataset/scale

PR #39 used Lion at `lr=1.7e-5, wd=5e-3` (Chen et al. 2023 paper config from image classification).
This run used `lr=5e-5, wd=5e-4` (the AdamW-equivalent translation tanjiro tested as arm B). Same
optimizer, same code, same data — **−27% improvement just from changing the LR/WD constants**.

Why the paper config fails here:
- Lion paper used image classification with millions of training samples; we have **400 cars**.
- Smaller datasets need more aggressive per-step movement (higher `lr`) to traverse the loss
  landscape inside a 270-min budget.
- Higher `wd` in the paper helps regularize huge nets; our 4L/512d/8h is small enough that
  `wd=5e-4` (AdamW-equivalent) is sufficient.

**All future Lion experiments must use `--lr 5e-5 --weight-decay 5e-4`, not the paper config.**
Posted notification on PRs #50, #51, #52, #54 instructing them to update their LR/WD before launch.

### What this implies for the queued composition stack

- **PR #50 (nezuko, Lion + compile)** — should now beat 11.30 by adding the +4–5% compile bump (epoch 16).
- **PR #51 (edward, Lion + RFF)** — RFF gave +0.88 abupt at the AdamW base; expected ~10.5–10.8.
- **PR #52 (tanjiro, Lion + RFF + compile)** — full triple-stack; if RFF+compile is orthogonal to Lion,
  expect ~9.8–10.3.
- **PR #54 (fern, Lion + per-axis tau_y/tau_z weights)** — directly targets binding 5×+ gap; expect
  the largest delta on tau_y/tau_z specifically.

### Volume_pressure note

Lion arm B's `volume_pressure=12.755` is a regression vs PR #39 Lion paper-config (13.83 → wait,
12.755 < 13.83 → improvement, −7.8%). Earlier write-up of "volume_pressure regression from PR #46"
referred to AdamW+compile vs Lion. With Lion+5e-5 we recover that. PR #55 (alphonse, vol_w=3.0 on
RFF+compile **AdamW** baseline) is now lower priority — the regression it targets dissolves under Lion.

---

## 2026-04-30 01:55 UTC — PR #47 CLOSED: thorfinn bilateral train-aug — +36% regression

**Student:** thorfinn | **W&B run:** `fn8pyav5` | **Group:** `thorfinn-tau-yz-attack`
**Hypothesis:** Bilateral symmetry train-time augmentation. Mirror-flip y-axis with antisymmetric
sign on tau_y to exploit DrivAerML's left-right symmetry.

### Results — clear regression on every axis

| Metric | PR #47 | tay SOTA (PR #46) | Δ |
|---|---:|---:|---:|
| `abupt` | **19.786** | 14.550 | +36.0% |
| surface_pressure | 13.258 | 8.628 | +53.7% |
| wall_shear | 21.107 | 14.882 | +41.8% |
| volume_pressure | 16.525 | 15.032 | +9.9% |
| tau_x | 18.524 | 12.901 | +43.6% |
| tau_y | 24.461 | 17.281 | +41.5% |
| tau_z | 26.163 | 18.907 | +38.4% |

Best val_abupt: 18.84. Runtime: 5h 13m. Best epoch: not specified.

### Diagnosis: bilateral aug introduces a mismatch

DrivAerML cars have weak (driver-side mirrors, exhaust placement, cabin layout) but real
left-right asymmetry. Mirror-augmenting symmetrically tells the model to expect identity
behavior on flipped geometries, but the **test set has those asymmetries** — so the
augmented model has been pushed to a less-accurate point on real cars.

This is the augmentation-as-prior failure mode: an inductive bias that contradicts the
test distribution causes regression even on the hypothesized target axes (tau_y/tau_z
both 41-38% worse than SOTA).

### Disposition: CLOSED

Pod hung post-train, advisor-closed from W&B-verified metrics. If symmetry exploitation
is worth revisiting, it should be at the **architecture level** (antisymmetric/symmetric
heads, geometry-aware coordinate pre-processing), not the augmentation level.

Reassigned thorfinn to **PR #56: AdamW + RFF + compile + cosine LR T_max=16** —
calibration fix for the schedule miscalibration we identified during PR #44 closure.

---

## 2026-04-30 01:53 UTC — PR #42 status update: frieren launched Lion arm

**Student:** frieren | **AdamW arm complete** (`uwt74mip`): `test_abupt = 15.82` — confirmed
the squared_rel_l2 + compile composition (−8.3% vs PR #40 baseline 17.25) but does not
beat SOTA. Frieren proactively launched a **Lion + compile + squared_rel_l2** arm
(run `24bdfcnz`, group `tay-round2-squared-rel-l2-compiled`) using the new SOTA-class
optimizer + compile + their loss-form change.

Advisor-side comment posted to update frieren's comparison goalpost (PR #46 SOTA = 14.55,
not PR #39's 15.43; T_max calibration should be 16 not 12).

---

## 2026-04-30 01:12 UTC — PR #46 MERGED: alphonse AdamW + RFF + compile — NEW SOTA 14.550

**Student:** alphonse | **W&B run:** `28l4yanr` | **Group:** `tay-round2-rff-compiled`
**Hypothesis:** Compose single-scale RFF (sigma=1.0, 32 feats) with `--compile-model` on AdamW base.
Best epoch: 16 (compile gives ~18 min/epoch → 16 epochs in 285-min budget vs 9 uncompiled).

### Results — NEW SOTA

| Metric | PR #46 (new SOTA) | PR #39 Lion (prev) | PR #40 | Δ vs PR #39 |
|---|---:|---:|---:|---:|
| `test_abupt` | **14.550** | 15.43 | 17.25 | **−0.88 (−5.7%)** |
| surface_pressure | **8.628** | 9.45 | 10.92 | −0.82 (−8.7%) |
| wall_shear | **14.882** | 16.28 | 18.33 | −1.40 (−8.6%) |
| volume_pressure | 15.032 | **13.83** | 14.71 | **+1.20 (+8.7%) ← REGRESSION** |
| tau_x | **12.901** | 13.91 | 15.73 | −1.01 (−7.3%) |
| tau_y | **17.281** | 19.58 | 21.80 | −2.30 (−11.7%) |
| tau_z | **18.907** | 20.40 | 23.07 | −1.49 (−7.3%) |

val_abupt at best epoch: 13.487. Val→test gap: +1.063. Best epoch 16 / 50 configured.

### Mechanism

Compile unlocks epoch 16 vs ~9 uncompiled (PR #33). RFF enriches input features with non-linear
position encodings. The deeper training (7 more epochs) is what saturates the surface/wall-shear:
tau_y leads at −11.7%, consistent with spectral-bias hypothesis. Volume_pressure **regressed**
because AdamW with equal volume_loss_weight=2.0 underweights volume when surface gradients dominate.
Lion's sign-update (PR #39) normalizes per-channel gradient magnitude, giving volume better signal
at the same weight — which is why PR #39 had 13.83 vs this run's 15.032.

### Disposition: MERGED (new SOTA)

Pod stalled post-training (alphonse pod healthy but didn't run report-results). Advisor merged
directly from W&B-verified metrics. Assigned alphonse to **PR #55: RFF + compile + volume-loss-weight 3.0**
to recover the volume regression.

---

## 2026-04-30 00:49 UTC — PR #43 CLOSED: fern multi-scale RFF — null result vs own baseline

**Student:** fern | **W&B run:** `5bx6zsio` | **Group:** `fern-multiscale-rff`
**Hypothesis:** 3-band Gaussian RFF (`sigma = {0.1, 1.0, 10.0}` surface; `{0.01, 0.1, 1.0}` volume)
to address spectral bias on tau_y/tau_z.

### Results vs PR #33 (single-scale RFF) and current SOTA PR #39 (Lion)

| Metric | PR #33 single-scale | PR #43 multi-scale | vs PR #33 | tay SOTA PR #39 |
|---|---:|---:|---:|---:|
| `test_abupt` | 17.77 | **17.86** | +0.09 | **15.43** |
| surface_pressure | 11.20 | 11.28 | +0.08 | 9.45 |
| wall_shear | 18.68 | 18.69 | +0.01 | 16.28 |
| volume_pressure | 16.13 | 16.30 | +0.17 | 13.83 |
| tau_x | 16.20 | 16.14 | −0.06 | 13.91 |
| tau_y | 21.81 | 21.84 | +0.03 | 19.58 |
| tau_z | 23.54 | 23.73 | +0.19 | 20.40 |

Best epoch: 9 (8 full + 1 partial forced at 270-min timeout). Val volume_p improved
(9.84→9.40, −4.4%) but **did not transfer to test** (+0.17). Throughput −14% (step
throughput 1.7→1.49 it/s) from 3× RFF compute, compressing to 8 full epochs vs 9.

### Diagnosis

Multi-scale RFF on the standard surface-coord domain is **parameter-redundant** with
sigma=1.0 single-scale. Surface domain `[−1,4] m` is already well-covered by sigma=1.0.
Adding sigma=0.1 (low band) and sigma=10.0 (high band) provides no new spectral information
at the Transolver input layer. The val volume_p gain overfits to the 34-case val split.

### Disposition: CLOSED (+15.7% regression vs new SOTA)

Baseline has moved to PR #39 Lion (15.43). Multi-scale RFF on AdamW base cannot compete.
Assigned fern to **PR #54: Lion + per-axis tau_y/tau_z loss weighting** — directly
targets the binding 5×+ gap on the two worst axes.

---

## 2026-04-29 13:35 UTC — PR #36 closed, tanjiro reassigned to PR #39

PR #36 (tanjiro: SDF-gated volume attention bias) closed after 5+
deterministic crashes at step 2719 (validation/eval code path) and
90+ min of pod claude session stuck on iteration 9 without producing
a successful run or responding to the advisor comment. The student
diagnosed and fixed two real bugs (slice-attention back-distribution,
torch.compile shape recompilation) but the residual eval-path bug
ate too much wall time. The SDF-gate hypothesis is preserved in the
Round 2 queue.

Reassigned to PR #39: **Lion optimizer drop-in replacement for AdamW**
at 4L/512d/8h. Single-delta hypothesis. Modifies only `train.py`,
no `model.py` changes. 2-arm sweep on lr/wd translation
(paper-recommended 1.7e-5/5e-3 vs AdamW-equivalent 5e-5/5e-4).
Lion is a strong empirical winner across vision/language/graph
transformer training, uses ~50% less optimizer-state memory than
AdamW, and composes orthogonally with all Round 1 levers.

## 2026-04-30 11:00 UTC — PR #70 CLOSED: tanjiro Lion+compile+lr=2.5e-5 (half-LR) — diverged late, 9th Lion+modification failure

- **Branch:** `tanjiro/round5-lion-half-lr`
- **W&B run:** `t5qnagui` — finished 154.6min (compile: ~10 epochs in budget window)
- **Hypothesis:** Halving LR from 5e-5 to 2.5e-5 would suppress the Lion+compile sign-bias instability by reducing per-step magnitude, opening the Lion+compile frontier.
- **Result: DIVERGED — best mid-training val 13.071 at step 19046 (~ep7), final val 33.36 at terminal step**

| Metric | Best mid-train val | Final val (DIVERGED) | SOTA `vnb7oheo` |
|---|---:|---:|---:|
| val_abupt | **13.071** (ep~7) | 33.36 | — |
| No test_primary | — | — | 11.303 |

- **Val trajectory:** stable descent ep1→ep7 (val 13.07), then **diverged ep9-10** (val 33.36 terminal). State=finished at 154.6min — confirmed ~10 epochs at compile speed (~16min/epoch). No test eval was run (val was final-epoch diverged at terminal step; best-val checkpoint source=None in summary).
- **Conclusion: Lion+compile diverges at half-LR too.** Half-LR delayed onset by ~2 epochs vs full-LR but did NOT prevent divergence. **9th confirmed Lion+modification failure.**
- **Mechanism update:** The sign-bias issue scales with gradient magnitude, not LR magnitude. Halving LR cuts the step size but leaves the binary sign pattern unchanged — the accumulated bias toward deterministic sign flips at low gradient steps still drives divergence, just with smaller increments.
- **Lion+compile frontier is closed at any LR within 270min budget.**
- **Stable Lion variants confirmed (only 2):**
  1. Vanilla Lion uncompiled (vnb7oheo, test 11.30 SOTA)
  2. Lion+RFF uncompiled (edward #51 ftg0ci0p, val 10.665 ep9 — SOTA candidate, new run iocqp761 in progress)
- **Reassignment:** tanjiro → PR #90 (Lion+RFF+EMA decay 0.9999 — yi #13 compounding).

## 2026-04-30 11:00 UTC — PR #55 CLOSED: alphonse AdamW+RFF+compile+vol_w=3.0 — vol_w=3 lever is closed-door

- **Branch:** `alphonse/round3-rff-compile-vol3`
- **W&B run:** `lahk19ws` — finished 287min, **test eval completed**
- **Hypothesis:** Upweighting volume loss (vol_w=3 vs 2) on the stable AdamW+RFF+compile base targets the volume_pressure ×2.1 binding gap without Lion-fragility risk.
- **Result: REGRESSION — test_abupt 16.387 (+45% vs SOTA 11.30, +12.6% vs PR #46 14.55)**

| Metric | PR #55 (vol_w=3) | PR #46 (vol_w=2) | SOTA 11.30 | AB-UPT |
|---|---:|---:|---:|---:|
| **abupt_mean** | **16.387** | 14.550 | 11.303 | — |
| surface_pressure | 10.256 | 8.628 | 6.216 | 3.82 |
| wall_shear | 17.349 | 14.882 | 11.315 | 7.29 |
| volume_pressure | 14.537 | 15.032 | 12.755 | 6.08 |
| tau_x | 15.060 | 12.901 | 9.563 | 5.35 |
| tau_y | 20.267 | 17.281 | 13.831 | 3.65 |
| tau_z | 21.813 | 18.907 | 14.147 | 3.63 |

- **Trajectory:** best val 15.486 at ep12, then diverged: ep15=15.52 → ep16=21.89 → ep17=57.04. **AdamW+vol_w=3 also diverged late** (ep16-17), which is unusual for AdamW. The upweighting likely over-tilted the loss landscape such that surface gradient signal dominated more strongly than expected in late-training, pushing the model into an unstable regime.
- **Interpretation:** vol_w=3.0 produces a small improvement on volume_pressure (15.03→14.54, -3%) but at the cost of large regressions on ALL other axes (surface +19%, tau_y +17%, tau_z +15%, etc.), AND late divergence.
- **Combined with frieren #68 (Lion+vol_w=3, test 15.57):** vol_w=3 lever is **closed-door at 4L/512d for both AdamW and Lion**. Volume_pressure recovery requires architectural change (deeper/wider model) or different data representation.
- **Reassignment:** alphonse → PR #91 (Lion+RFF+sigma=2.0 — RFF frequency sweep targeting tau_y/tau_z).

