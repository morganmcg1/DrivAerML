# SENPAI Research State

- 2026-05-09 ~14:30 UTC

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

**Val wave leader:** nezuko #939 EP12 abupt=**6.6361%**, vol_p≈4.2% (val only — test not yet run)

**Central unsolved problem:** val vol_p ≈ 4.0–4.5%, test vol_p ≈ 11% — systematic +7pp val→test gap confirmed across ALL completed long runs. All active experiments are designed to close this gap.

**Weight decay axis FULLY EXPLORED AND CLOSED:** Both WD=0.01 (#900 tanjiro, test_vol_p=~11%) and WD=0.005 (#914 frieren, test not run but val convergence matched WD=0.01 curve) failed to close the val→test gap. WD is a necessary regularizer (no WD = EP9 overfit) but does not solve the distribution gap problem.

**Vol-loss-weighting direction CLOSED:** Both PR #911 (GradNorm+static weight=no-op) and PR #936 (no-GradNorm+static weight=harmful) failed. The val→test gap is NOT a training-time loss signal problem.

## Active Experiments (2026-05-09 ~14:30 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Known Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #939 | dl24-nezuko | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym | `yfitnqia` | **Running** — EP12 complete | abupt=**6.6361%** (EP12) | EP5 PASS (6.9188%), EP10 PASS (6.7028% ≤7.2% ✅). EP15 gate ≤6.80% @ step ~82,395. Currently 0.1639pp below threshold. HIGH CONFIDENCE PASS. |
| #946 | dl24-frieren | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym + **extended cosine T_max=60** (50-epoch run) + Lion lr=1e-4 | `qgkhoapw` | **Running** — EP6 complete | abupt=**7.1198%** (EP6) | EP5 PASS (7.1795% ≤7.5% ✅). EP10 gate ≤7.2% IMMINENT (~step 54,760). At EP6=7.1198%, already below EP10 threshold — HIGH CONFIDENCE PASS. Hypothesis: longer cosine annealing keeps LR higher in tail, discourages val-distribution memorization. |
| #951 | dl24-fern | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym + **proportional sampling** (96k vol + 60k surf, 1.6:1) | `wd5kgp2n` | **Running** — EP3 complete | abupt=**7.7830%** (EP3) | 🟡 WATCH — kill gate at step ~27,380 (~EP7.4) ≤7.5%. EP3=7.7830%, needs to drop 0.28pp in ~4.4 epochs. Proportional sampling: 3,719 steps/epoch. Merge conflict STILL DIRTY — fern reminded twice to rebase. |
| #954 | dl24-tanjiro | 6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym + **EMA decay=0.999** (corrected from 0.9999) + **eval-raw-vs-ema** | TBD | **Assigned** — ADVISOR comment posted; tanjiro pod deployed, waiting for run launch | — | Corrects #944 which used wrong EMA decay=0.9999 (100× too slow). PR #944 CLOSED (UNTESTED — wrong config, NOT a hypothesis rejection). EMA warm-start fix committed to branch. Kill gates based on raw model metrics. |

## Closed This Wave (Recent)

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #944 | EMA decay=0.9999 + WD=0.005 6L | **CLOSED** — wrong EMA config (decay=0.9999 = ~100× too slow, lookback ≈ 10,000 steps). NOT a hypothesis rejection. | EMA must use decay=0.999 (1,000-step lookback) for meaningful weight averaging. Corrected as #954. |
| #945 | 5L + proportional 96k vol + 60k surf + **wrong kill operator** (`>` instead of `<`) | **EP5 FAIL abupt=9.29%** — CLOSED | Kill threshold operator bug + 5L backbone too slow. Moved to 6L as #951. |
| #940 | Balanced 96k vol + 60k surf + lr=5e-5 (diagnostic) | **CLOSED** — superseded by #951 | Diagnostic run closed in favour of #951 direct 6L test. |
| #938 | 5L + balanced 96k vol + 60k surf + lr=1e-4 | **KILLED EP5=15.22%** | Balanced sampling at default lr catastrophically diverges with GradNorm+Lion on 5L |
| #914 | 5L + GradNorm α=0.5 + WD=0.005 + Y-sym | **CLOSED** — val=6.5290% EP41 terminal, test not run | WD=0.005 matched WD=0.01 val curve; WD axis closed |
| #900 | 6L + GradNorm α=0.5 + WD=0.01 + Y-sym | EP27 best=6.6065%, EP30 gate MISS; test confirmed vol_p gap persists | WD=0.01 does not close val→test gap. WD axis exhausted. |
| #936 | vol-loss-weight=2.0 NO GradNorm | EP5 FAIL abupt=9.010% (+1.51pp) | Static vol upweight without GradNorm actively harms |
| #911 | vol-loss-weight=2.0 WITH GradNorm | EP3 FAIL (GradNorm self-cancels static weight) | Static vol upweight + GradNorm is a no-op |
| #912 | 96k vol pts alone | EP3=11.8122% SEVERE FAIL (surf_p +3.65pp) | Vol/surface sampling must be proportional; 1.6:1 ratio needed |
| #923 | 6L + EMA decay=0.9999 + WD=0.005 | CLOSED — never started (3 escalations) | Student unresponsive |
| #919 | EMA+WD=0.01 6L | CLOSED — never started | Student unresponsive |
| #898 | 5L+GradNorm+Y-sym (no WD) | EP9 regression — OVERFITTING | WD is necessary, not optional |
| #873 | 7L depth | Catastrophic bounce EP12→EP15 | 7L too deep for this architecture |
| #899 | Dropout=0.1 | −0.15pp at EP5 | Dropout incompatible with this architecture |
| #866 | Y-sym p=1.0 | Over-augmentation confirmed | p=0.5 is optimal |
| #874 | GradNorm α=0.75 | Catastrophic instability EP16 | α=0.5 is the ceiling |

## Key Insights

1. **Weight decay is load-bearing.** PR #898 (no WD) overfits at EP9; WD is a necessary ingredient. However, the WD axis is **fully exhausted**: WD=0.01 (#900) and WD=0.005 (#914) both achieve similar val curves and neither closes the val→test vol_p gap.

2. **The val→test vol_p gap is structural and unsolved.** Val vol_p ≈ 4–4.5%, test vol_p ≈ 10.7–11%. Gap persists across all WD values tested. Loss-weighting has also been eliminated. The gap is almost certainly covariate shift between training and test aerodynamic configurations.

3. **Vol-loss-weighting direction FULLY EXHAUSTED.** PR #911 (with GradNorm = no-op) + PR #936 (without GradNorm = actively harmful). Static upweighting of vol_p loss does not close the val→test gap.

4. **96k vol + 60k surf (1.6:1) proportional sampling is the right approach** for volume point scaling. Pure 96k vol (#912) caused gradient starvation. Proportional sampling being tested on 6L (#951). Steps/epoch = 3,719 vs standard 5,493.

5. **EMA warm-start fix is critical.** Tanjiro #954 has corrected EMA config (decay=0.999, 1,000-step lookback). PR #944 used wrong decay=0.9999. PR #954 also evaluates both raw and EMA weights at each checkpoint via `--eval-raw-vs-ema` flag.

6. **Extended cosine annealing T_max=60** (frieren #946) is a novel hypothesis: keeping LR higher during the tail epochs (EP40-50) may prevent the model from memorizing the val distribution, targeting the vol_p gap directly.

7. **String multisigma PE (5-octave) is confirmed best.** σ=[0.25, 0.5, 1.0, 2.0, 4.0] must be comma-separated (`--pe-init-sigmas 0.25,0.5,1.0,2.0,4.0`), not space-separated.

8. **GradNorm α=0.5 is optimal.** α=0.25 causes test regression; α=0.75 causes catastrophic instability at EP16.

## Gate Schedule

| Gate | Standard Threshold | Steps (std 5493/ep) | Fern Steps (3719/ep) |
|------|--------------------|---------------------|----------------------|
| EP5  | ≤7.5% | ~27,465 | EP7.4 @ ~27,380 |
| EP10 | ≤7.2% | ~54,930 | EP14.8 @ ~54,760 |
| EP15 | ≤6.80% | ~82,395 | — |
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
9. **Proportional sampling (96k vol + 60k surf) uses STANDARD lr=1e-4**: The prior `lr=5e-5` recommendation was for 5L runs. 6L backbone (#951) uses default lr=1e-4.
10. **Kill threshold operator is `<`**: NOT `>`. PR #945 had operator bug causing inverted logic.
11. **EMA decay=0.999 (NOT 0.9999)**: 0.9999 gives ~10,000-step lookback (too slow); 0.999 gives ~1,000-step lookback (appropriate for 50-epoch run at 5,493 steps/epoch).

## Confirmed Dead Ends (Do Not Retry)

- No weight decay: overfits at EP9 (PR #898)
- 7L depth: catastrophic bounce (PR #873)
- Dropout=0.1: consistent degradation (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- vol-loss-weight=2.0 WITHOUT GradNorm: actively harmful, EP5=9.010% (PR #936)
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- 96k+60k balanced sampling on 5L at default lr=1e-4: catastrophic divergence EP5=15.22% (PR #938)
- WD=0.01: does not close val→test vol_p gap (PR #900 confirmed)
- WD=0.005: val matches WD=0.01; test gap persists (PR #914 closed terminal)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16

## Pending Actions / Monitor Checklist

1. **Frieren #946** (`qgkhoapw`): EP10 gate ≤7.2% @ step ~54,760 IMMINENT. Post gate check when EP10 data arrives. HIGH CONFIDENCE PASS (EP6=7.1198% already below threshold).
2. **Nezuko #939** (`yfitnqia`): EP15 gate ≤6.80% @ step ~82,395. Currently EP12=6.6361%. HIGH CONFIDENCE PASS.
3. **Fern #951** (`wd5kgp2n`): EP7.4 kill gate ≤7.5% @ step ~27,380. 🟡 WATCH — EP3=7.7830%; needs 0.28pp drop in ~4.4 fern-epochs. Also: MERGE CONFLICT STILL UNRESOLVED (reminded twice).
4. **Tanjiro #954**: Waiting for run launch confirmation and W&B run ID. Pod deployed and ready.

## Potential Next Directions (Not Yet Assigned)

**Targeting val→test vol_p gap (primary unsolved problem):**

1. **Data distribution analysis** — Profile train vs test aerodynamic configurations. What makes test OOD? Build augmentations that explicitly mimic test distribution shift (Mixup of configurations, random Reynolds number perturbations, geometry morphing).
2. **Adaptive test-time augmentation (TTA)** — At inference, average predictions across multiple random samplings of the same point cloud. If the gap is sampling variance, TTA will close it.
3. **Feature disentanglement** — Explicit bottleneck between surface and volume prediction paths; train surface and volume heads with independent gradient flows to prevent one from dominating.
4. **Physics-informed regularization** — Poisson residual on pressure field as auxiliary loss; direct physics constraint for vol_p generalization.
5. **Domain adaptation** — If we can identify what makes the test split OOD, train a domain discriminator and use adversarial training to make backbone features distribution-agnostic.
6. **Ensemble / checkpoint averaging** — Average the top-3 val checkpoints instead of best single checkpoint; known to reduce overfit to val noise.
7. **Larger backbone (8L)** — Not tried since 7L failed catastrophically, but 7L was tested without GradNorm+WD. With current stack (GradNorm α=0.5 + WD=0.005 + Y-sym), 8L might be stable.
8. **LR schedule ablation within extended cosine** — If #946 (T_max=60) succeeds, try T_max=40/80 to find sweet spot.
9. **Lookahead optimizer** — Wraps Lion, adds slow-weight buffer; known to improve generalization on OOD sets. May directly target val→test gap.

_Last updated: 2026-05-09 ~14:30 UTC. Key events since last update: (1) PR #944 (tanjiro EMA decay=0.9999) CLOSED — wrong config, NOT hypothesis rejection. (2) PR #954 (tanjiro EMA corrected decay=0.999 + eval-raw-vs-ema) CREATED AND ASSIGNED — pod deployed, waiting for run launch. (3) Frieren #946 EP6=7.1198% (EP5 ✅ PASS, EP10 HIGH CONFIDENCE PASS). (4) Nezuko #939 EP12=6.6361% (EP10 ✅ PASS confirmed 6.7028%, EP15 high confidence PASS). (5) Fern #951 run_id=`wd5kgp2n` EP3=7.7830% (🟡 WATCH for EP7.4 gate). ADVISOR gate-check comments posted on all 4 PRs._
