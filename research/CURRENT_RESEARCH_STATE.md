# SENPAI Research State

- 2026-05-11 ~00:21 UTC

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

**Val wave leader:** nezuko #939 EP33 abupt=**6.5522%**, vol_p=4.2202% (val only — test not yet run)

**Strongest early-wave signal:** fern #968 EP6=**6.3937%**, vol_p=4.1712% — stochastic vol subsampling running ~1.0pp ahead of yfitnqia at matched epoch.

**Central unsolved problem:** val vol_p ≈ 4.0–4.5%, test vol_p ≈ 10.7–11% — systematic +7pp val→test gap confirmed across ALL completed long runs. All active experiments are designed to close this gap.

**Weight decay axis FULLY EXPLORED AND CLOSED:** Both WD=0.01 (#900 tanjiro) and WD=0.005 (#914 frieren) failed to close the val→test gap.

**Vol-loss-weighting direction CLOSED:** PR #911 (GradNorm+static weight=no-op) + PR #936 (weight=2.0, no GradNorm=harmful) + PR #964 (weight=3.0, no GradNorm=WORST EVER: gap +8.12pp, test_abupt=8.0190%). Static vol loss upweighting conclusively does NOT close the val→test gap. This axis is FULLY CLOSED.

**EMA AXIS CLOSED:** PR #954 (EMA decay=0.999 + eval-raw-vs-ema) showed test_vol_p=11.28% and test_abupt=7.55% — confirming EMA weight averaging does not reduce the val→test gap. CLOSED.

## Active Experiments (2026-05-11 ~00:21 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Known Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #939 | dl24-nezuko | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym | `yfitnqia` | **Running** — EP35 complete | abupt=**6.5522%** (EP33 run best), EP35=6.5657% | Deep plateau since EP23. EP35 gate PASS (6.5657% ≤6.58% ✓). EP40 gate ≤6.55% needs -0.003pp recovery from EP33 run best. ETA EP40 ~04:00Z. |
| #965 | dl24-tanjiro | 6L→**8L** STRING + GradNorm α=0.5 + WD=0.005 + Y-sym | `pgy0xbyw` | **Running** — EP11 ⚠BOUNCE | abupt=EP10=6.5653% (run best), EP11=6.7869% (+0.222pp bounce) | EP11 bounce below 0.3pp kill threshold. EP12 is decision point (ETA ~00:50Z). If EP12 <6.65%: noise, continue. If EP12 >6.70%: KILL — same failure mode as 7L #873. |
| #968 | dl24-fern | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym + **stochastic vol subsampling** (fresh random draw every batch) | `a0yoxy85` | **Running** — EP6 complete | abupt=**6.3937%** (EP6), vol_p=4.1712% | Strongest early-wave signal. ~1.0pp ahead of yfitnqia at matched epoch. Monotonic descent all metrics. EP10 ETA ~01:50Z. Gate ≤7.2% pre-cleared. |
| #972 | dl24-frieren | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym + **SDF-stratified importance sampling** (bias toward far-field cells) | TBD | **Starting** — pod active, no comments yet | — (training not yet started as of 00:21Z) | Agent invoked ~00:09Z; setup expected ~00:30Z; first metrics ~01:00Z. Hypothesis: reduce spatial memorization by weighting vol draws by `1 + α*|sdf|`. |

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

## Potential Next Directions (Not Yet Assigned)

**Targeting val→test vol_p gap (primary unsolved problem):**

**Active (IN PROGRESS):**
- **Stochastic vol subsampling** — Fresh random vol point draw every batch (PR #968 / fern). Early signal EP6=6.3937%, vol_p=4.1712% — ~1.0pp ahead of yfitnqia at matched epoch. Strongest wave signal.
- **SDF-stratified importance sampling** — Weight vol draws by `1 + α*|sdf|` to bias toward far-field cells (PR #972 / frieren). Training startup in progress.

**Not yet assigned:**
1. **Data distribution analysis** — Profile train vs test aerodynamic configurations. What makes test OOD? Build augmentations that explicitly mimic test distribution shift (Mixup of configurations, random Reynolds number perturbations, geometry morphing).
2. **Adaptive test-time augmentation (TTA)** — At inference, average predictions across multiple random samplings of the same point cloud. If the gap is sampling variance, TTA will close it.
3. **Feature disentanglement** — Explicit bottleneck between surface and volume prediction paths; train surface and volume heads with independent gradient flows to prevent one from dominating.
4. **Physics-informed regularization** — Poisson residual on pressure field as auxiliary loss; direct physics constraint for vol_p generalization.
5. **Domain adaptation** — If we can identify what makes the test split OOD, train a domain discriminator and use adversarial training to make backbone features distribution-agnostic.
6. **Ensemble / checkpoint averaging** — Average the top-3 val checkpoints instead of best single checkpoint; known to reduce overfit to val noise.
7. **Lookahead optimizer** — Wraps Lion, adds slow-weight buffer; known to improve generalization on OOD sets. May directly target val→test gap. Requires code change.

_Last updated: 2026-05-11 ~00:24 UTC. Key events: (1) PR #964 CLOSED — vol-loss-weighting axis fully rejected (WORST EVER: test_abupt=8.0190%, val→test gap +8.12pp; all static upweighting CLOSED). (2) PR #951 CLOSED — proportional sampling 96k+60k slower convergence than standard 65k/40k. (3) PR #968 assigned to fern — stochastic vol subsampling; strongest early-wave signal EP6=6.3937% (1.0pp ahead of yfitnqia). (4) PR #972 assigned to frieren — SDF-stratified importance sampling; training startup in progress ~00:09Z. (5) Tanjiro #965 EP11 bounce (+0.222pp): EP12 decision point ETA ~00:50Z (if >6.70%: KILL). (6) Nezuko #939 EP35 PASS (6.5657%), run best EP33=6.5522%, EP40 gate ≤6.55% ETA ~04:00Z._
