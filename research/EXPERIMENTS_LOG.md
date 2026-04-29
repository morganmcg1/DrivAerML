# SENPAI Research Results — DrivAerML (`tay`)

W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`.

Targets to beat (lower is better, AB-UPT public reference):
`surface_pressure 3.82`, `wall_shear 7.29`, `volume_pressure 6.08`,
`tau_x 5.35`, `tau_y 3.65`, `tau_z 3.63`.

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

