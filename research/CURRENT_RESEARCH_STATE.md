# SENPAI Research State
- 2026-05-09 ~13:30 UTC

## Human Research Directive (Issue #882)
**TOP PRIORITY — Volume Pressure Focus:**
- The **TEST volume pressure L2 error** is the only metric that matters for new experiment design
- Do NOT degrade surface error or wall shear stress metrics
- Published SOTA models show significantly better volume pressure test metrics are achievable — large headroom to close
- All new student assignments must be designed with volume pressure improvement as the singular focus

## Wave SOTA (Test — Merged on Branch)

**PR #740** (dl24-fern, run `5x8wofzm`): `abupt_axis_mean_rel_l2_pct` = **7.5195%**
| Metric | Value |
|--------|-------|
| test_abupt | 7.5195% |
| surf_p | 3.8810% |
| vol_p | 10.7580% |
| wall_shear | 7.0610% |

**Val wave leader (active):** fern #968 EP15 abupt=**6.2806%**, vol_p=**3.9989%** — FIRST sub-4% val_vol_p in codebase history.

**Central unsolved problem:** val vol_p ≈ 4.0–4.5%, test vol_p ≈ 10.7–11% — systematic +7pp val→test gap confirmed across ALL completed long runs. All active experiments are designed to close this gap.

**Vol-loss-weighting direction CLOSED:** PR #911 (GradNorm+static weight=no-op) + PR #936 (weight=2.0, no GradNorm=harmful) + PR #964 (weight=3.0, no GradNorm=WORST EVER: gap +8.12pp, test_abupt=8.0190%). Static vol loss upweighting conclusively does NOT close the val→test gap. This axis is FULLY CLOSED.

**EMA AXIS CLOSED:** PR #954 (EMA decay=0.999 + eval-raw-vs-ema) showed test_vol_p=11.28% and test_abupt=7.55% — confirming EMA weight averaging does not reduce the val→test gap. CLOSED.

## Active Experiments (2026-05-09 ~13:30 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Known Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #968 | dl24-fern | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym + **stochastic vol subsampling** (fresh random draw every batch) | `a0yoxy85` | **Running** — EP20 CLEARED (6.2973%), plateau EP15–EP20, terminal sprint to EP30 ~14:15Z | abupt=**6.2806%** (EP15 best), vol_p=**3.9989%** | Plateau confirmed EP15→EP20 (6.28–6.30% band). GradNorm w_vol_p drifting down: 0.104→0.095→0.076 (vol_p naturally easy for stochastic subsampling). Test eval auto-runs at terminal EP30. |
| #972 | dl24-frieren | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym + **SDF-stratified importance sampling** (bias toward far-field cells, α=2.0) | `56bcqp3m` | **Running** — EP8 complete, strong monotone descent | abupt=**6.2296%** (EP8 BEST), vol_p=**3.8798%** (WAVE LEADER) | SDF hypothesis for gap-closing FALSIFIED (EP7 test: val→test vol_p gap = +8.00pp ≈ +8.12pp baseline). Run continues for val_abupt wave leadership. First ever sub-3.88% val_vol_p. ETA EP30 ~05:30Z May 10. |
| #987 | dl24-tanjiro | **Lookahead optimizer** — Lookahead(k=5, α=0.5) wrapping Lion lr=1e-4; slow-weight buffer for OOD generalization | TBD | **WIP** — assignment posted, awaiting student pickup | — | DropPath FALSIFIED (EP5=7.885%, test_vol_p=14.278%, gap +7.91pp unchanged). New assignment: Lookahead optimizer targeting val→test vol_p gap via slow-weight averaging. |
| #990 | dl24-nezuko | **Vol coordinate noise augmentation** (σ=0.005, inject ε~N(0,σ²) on volume query coords during training only) | `um3tuyvy` | **Running** — launched 09:52Z May 11, EP1 in progress (~10:33Z ETA), EP5 gate ~11:35Z (≤7.5%) | — | Implementation confirmed healthy: `train/aug/vol_coord_noise_rms ≈ 0.005` ✅. Only xyz dims (0..2) perturbed; SDF (dim 3) untouched; `model.training` guard prevents eval contamination. |

## Closed This Wave (Recent)

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #964 | 6L + vol_loss_weight=3.0 + NO GradNorm | **CLOSED** — test_abupt=8.0190%, test_vol_p=12.52%, gap +8.12pp WORST EVER | Static vol upweighting (any value) does NOT close val→test gap. Gap is a DATA DISTRIBUTION problem not a loss-weighting problem. Axis FULLY CLOSED. |
| #951 | 6L + proportional sampling (96k vol + 60k surf) | **CLOSED** — slower convergence, EP15 gate AT RISK | Proportional 1.6:1 vol:surf ratio shows vol_p oscillation. Fixed-seed 65k vol standard is better. |
| #954 | EMA decay=0.999 + eval-raw-vs-ema 6L | **CLOSED** — test_abupt=7.5476%, test_vol_p=11.2803% | EMA does not close val→test gap. EMA axis FULLY REJECTED. |
| #946 | Extended cosine T_max=60 | **CLOSED** — EP7 regression +0.21pp | T_max=60 keeps LR too high in tail. Default T_max=50 confirmed optimal. |
| #944 | EMA decay=0.9999 + WD=0.005 6L | **CLOSED** — wrong EMA config | Corrected as #954 (also rejected). |
| #914 | 5L + GradNorm α=0.5 + WD=0.005 + Y-sym | **CLOSED** — val=6.5290% EP41 terminal | WD=0.005 matched WD=0.01; WD axis closed. |
| #900 | 6L + GradNorm α=0.5 + WD=0.01 + Y-sym | EP27 best=6.6065%, EP30 gate MISS | WD=0.01 does not close val→test gap. |
| #873 | 7L depth (without WD=0.005) | Catastrophic bounce EP12→EP15 | 7L without full regularization: catastrophic bounce. |

## Key Insights

1. **Weight decay is load-bearing.** PR #898 (no WD) overfits at EP9. WD is necessary but the WD axis (0.01 and 0.005) is fully exhausted — neither closes the val→test vol_p gap.

2. **The val→test vol_p gap is structural and unsolved.** Val vol_p ≈ 4–4.5%, test vol_p ≈ 10.7–11%. Gap persists across all WD values, EMA, loss-weighting, architecture-depth changes tried so far. Almost certainly covariate shift between training and test aerodynamic configurations.

3. **Vol-loss-weighting FULLY CLOSED:** PR #911 (GradNorm+static=no-op) + PR #936 (weight=2.0, no GradNorm=harmful) + PR #964 (weight=3.0, no GradNorm=WORST EVER: test_abupt=8.0190%, gap +8.12pp). Static vol loss upweighting does NOT close the val→test gap — confirmed at weight=2.0 and 3.0. Axis fully closed.

4. **EMA weight averaging rejected:** PR #954 (decay=0.999, corrected from #944's 0.9999) confirmed test_vol_p=11.28% — same gap as baseline. EMA axis fully closed.

5. **Extended cosine T_max=60 destabilizing:** #946 showed EP7 regression (+0.21pp), consistent with LR staying too high during the training tail. Default T_max=50 is confirmed optimal.

6. **Proportional sampling (60k surf + 96k vol):** Fern #951 showing slower convergence than standard 40k/65k per-epoch (EP10=6.86% vs nezuko 6.70% at EP10). vol_p oscillation pattern present. EP15 gate ≤6.80% AT RISK.

7. **String multisigma PE (5-octave) is confirmed best.** σ=[0.25, 0.5, 1.0, 2.0, 4.0].

8. **GradNorm α=0.5 is optimal.** α=0.25 causes test regression; α=0.75 causes catastrophic instability at EP16.

## Gate Schedule

| Gate | Standard Threshold | Steps (std 5493/ep) | Fern Steps (3719/ep) |
|------|--------------------|---------------------|----------------------|
| EP5  | ≤7.5% | ~27,465 | EP5 @ ~18,595 |
| EP10 | ≤7.2% | ~54,930 | EP10 @ ~37,190 |
| EP15 | ≤6.80% | ~82,395 | EP15 @ ~55,785 |
| EP20 | ≤6.70% | ~109,860 | — |
| EP25 | ≤6.65% | ~137,325 | — |
| EP30 | ≤6.60% | ~164,790 | — |
| EP35 | ≤6.58% | ~192,255 | — |
| EP40 | ≤6.55% | ~219,720 | — |

## Critical Config Constraints

1. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
2. **`--train-volume-points 65000` (or higher)** REQUIRED: default 16384 inverts volume:surface gradient ratio.
3. **`--lr-warmup-steps 500`** (NOT `--lr-warmup-epochs 1`): epoch-based = 43k steps, far too long.
4. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion optimizer.
5. **`--volume-loss-weight` BUG**: When `--use-gradnorm` is active, `--volume-loss-weight` is a no-op. Must use either: (a) disable GradNorm and use static weight, OR (b) apply fix from #911 branch.
6. **`--model-pe string_multisigma` REQUIRED when using STRING PE**: omitting causes `--pe-init-sigmas` to be silently ignored.
7. **`--pe-init-sigmas` must be COMMA-separated**: `0.25,0.5,1.0,2.0,4.0` NOT space-separated.
8. **96k vol points data loader bug**: `_indices()` produced 30.9% empty volume views at 96k. Fixed in nezuko #912 relaunch — must verify fix is present in any new 96k run.
9. **Proportional sampling (96k vol + 60k surf) uses STANDARD lr=1e-4**: prior `lr=5e-5` was for 5L runs.
10. **Kill threshold operator is `<`**: NOT `>`. PR #945 had operator bug causing inverted logic.
11. **EMA decay=0.999 (NOT 0.9999)**: 0.9999 gives ~10,000-step lookback (too slow); 0.999 gives ~1,000-step lookback. Note: EMA does not close val→test gap regardless.

## Confirmed Dead Ends (Do Not Retry)

- No weight decay: overfits at EP9 (PR #898)
- 7L depth without full regularization: catastrophic bounce (PR #873)
- 8L depth: EP11 bounce +0.222pp — EP12 is kill/continue decision point (PR #965, IN PROGRESS)
- Dropout=0.1: consistent degradation (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- vol-loss-weight=2.0 WITHOUT GradNorm: actively harmful, EP5=9.010% (PR #936)
- vol-loss-weight=3.0 WITHOUT GradNorm: WORST EVER test_abupt=8.0190%, gap +8.12pp (PR #964) — AXIS FULLY CLOSED
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- 96k+60k balanced sampling on 5L at default lr=1e-4: catastrophic divergence EP5=15.22% (PR #938)
- Proportional sampling 96k+60k at 6L: slower convergence, vol_p oscillation (PR #951)
- WD=0.01: does not close val→test vol_p gap (PR #900)
- WD=0.005: val matches WD=0.01; test gap persists (PR #914)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16
- Extended cosine T_max=60 (PR #946): destabilizing in training tail
- EMA decay=0.999 (PR #954): does not close val→test vol_p gap (test_vol_p=11.28%)
- TTA Y-symmetry (PR #979): gap +7.860pp → +7.863pp UNCHANGED (noise-level -0.012pp). AXIS CLOSED. Add `--use-tta` as free-lunch only.
- SDF-stratified importance sampling (PR #972, EP7 test eval): val→test vol_p gap = +8.00pp — identical to +8.12pp baseline. HYPOTHESIS FALSIFIED. Gap is not addressable by sampling strategy. Axis CLOSED for sampling-based gap-closing approaches.
- DropPath regularization (PR #987, run `rydn7aqb`): EP5=7.8846% (FAIL gate ≤7.5%); test_abupt=9.1036%, test_vol_p=14.278%, gap +7.91pp UNCHANGED. MECHANISM: DropPath adversarially interacts with GradNorm — drives w_vol_p ~0.14–0.18 lower than baseline throughout training, accelerating vol_p down-weighting when GradNorm should fight for it. FALSIFIED. Stochastic depth regularization does not help OOD vol_p generalization.

## Potential Next Directions (Not Yet Assigned)

**Targeting val→test vol_p gap (primary unsolved problem):**

1. **Vol coordinate noise augmentation (nezuko #990 — ACTIVE)** — Inject Gaussian noise ε~N(0, σ²) on volume query coordinates during training (σ=0.005 normalized coords) to force spatially smooth pressure field learning. NeRF-inspired spatial smoothness inductive bias. Only remaining spatial-regularization hypothesis untested. Requires `--vol-coord-noise-std` flag in train.py.
2. **DropPath regularization** — FALSIFIED (PR #987). test_vol_p=14.278%, gap +7.91pp unchanged. Adversarial GradNorm interaction confirmed.
3. **Data distribution analysis** — Profile train vs test aerodynamic configurations. What makes test OOD? Build augmentations that explicitly mimic test distribution shift.
4. **Physics-informed regularization** — Poisson residual on pressure field as auxiliary loss; direct physics constraint for vol_p generalization.
5. **Feature disentanglement** — Explicit bottleneck between surface and volume prediction paths; train surface and volume heads with independent gradient flows.
6. **Domain adaptation** — If we can identify what makes the test split OOD, train a domain discriminator and use adversarial training to make backbone features distribution-agnostic.
7. **Checkpoint averaging (top-3 val)** — Average top-3 val checkpoints instead of best single; known to reduce overfit to val noise. `--use-tta` free-lunch confirmed; checkpoint averaging is the next eval-time lever.
8. **Lookahead optimizer (tanjiro #987 — ACTIVE)** — Wraps Lion, adds slow-weight buffer (k=5, α=0.5); known to improve generalization on OOD sets. May directly target val→test gap. Requires code change.
9. **If tanjiro/nezuko both fail** — All training-side gap-closing hypotheses exhausted. Escalate to plateau protocol: (a) architecture changes (separate surface/volume encoder), (b) physics-based regularization (Poisson loss), (c) domain adaptation (adversarial OOD training).

_Last updated: 2026-05-09 ~13:30 UTC. Key events: (1) DropPath FALSIFIED (PR #987): EP5=7.885% gate FAIL, test_vol_p=14.278%, gap +7.91pp UNCHANGED. Mechanism: adversarial GradNorm interaction drives w_vol_p lower than baseline. (2) tanjiro reassigned to Lookahead optimizer (k=5, α=0.5) wrapping Lion. (3) frieren #972 heading to EP15 gate, wave val leader EP8 abupt=6.2296%. (4) fern #968 plateau EP15→EP20, terminal sprint to EP30. (5) nezuko #990 EP3 complete (abupt=9.43%), EP5 gate pending with test eval on best-val checkpoint._
