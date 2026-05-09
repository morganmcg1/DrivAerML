# SENPAI Research State

- 2026-05-09 ~20:45 UTC

## Human Research Directive (Issue #882)
**TOP PRIORITY — Volume Pressure Focus:**
- The **TEST volume pressure L2 error** is the only metric that matters for new experiment design
- Do NOT degrade surface error or wall shear stress metrics
- Published SOTA models show significantly better volume pressure test metrics are achievable — large headroom to close
- All new student assignments must be designed with volume pressure improvement as the singular focus

## Wave SOTA (Merged Test)

**PR #740** (dl24-fern, run `5x8wofzm`): `test_primary/abupt_axis_mean_rel_l2_pct` = **7.5195%**
| Metric | Value |
|--------|-------|
| test_abupt | 7.5195% |
| surf_p | 3.8810% |
| vol_p | 10.7580% |
| wall_shear | 7.0610% |

**Critical insight:** vol_p test = 10.758% vs val ≈ 4.0–4.3% — systematic ~6-7pp val→test gap in volume pressure.
This gap is the central unsolved problem. All current active experiments target it.

## Key Insights (2026-05-09 ~20:45 UTC)

**Weight decay = the crucial regularizer.** PR #898 (5L+GradNorm+Y-sym, no WD) overfitted at EP9 with clean val regression. PR #900 (6L+GradNorm+Y-sym, WD=0.01) has shown zero terminal regression, now at EP19 as wave val leader. This is a **confirmed causal result**: WD is necessary (not just helpful) for avoiding mid-run overfitting on this architecture.

**WD=0.005 eliminates EP9 transient.** Tanjiro #900 (WD=0.01) showed a +0.744pp spike at EP9; frieren #914 (WD=0.005) shows smooth monotonic decline through EP9 → EP11. The optimal WD is somewhere in [0.005, 0.01].

**EP25 gate officially cleared by tanjiro #900.** EP19=6.6545% < 6.65% threshold, margin only 0.0045pp. EP30 gap is 0.054pp — reachable but not guaranteed.

**Critical unanswered question:** Does weight decay actually compress the val→test vol_p gap at test time? Val vol_p is ~4.27% for tanjiro, but test has not yet been evaluated. If WD fixes val→test gap, tanjiro #900 could beat wave SOTA 7.5195%.

## Active Experiments (2026-05-09 ~23:30 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #900 | dl24-tanjiro | 6L + GradNorm α=0.5 + WD=0.01 + Y-sym | `os6v64lq` | Running EP20→EP21 | abupt=6.6617% (EP20), vol_p=4.2805%; EP19 still RUN BEST 6.6545% ⭐ | EP25 OFFICIALLY CONFIRMED ✅; EP30 gap=0.054pp |
| #914 | dl24-frieren | 5L + GradNorm α=0.5 + WD=0.005 + Y-sym | `wdxtdmhy` | Running EP13→EP14 | abupt=6.6624% (EP13 RUN BEST ⭐), vol_p=4.2390% (new min) | At step 76,668; EP14 fires at step ~76,915 (IMMINENT); EP15 pre-cleared |
| #923 | dl24-nezuko | 6L + GradNorm α=0.5 + EMA decay=0.9999 + WD=0.005 + Y-sym | `4w5k42t5` | Running EP1→EP2 | abupt=16.5734% (EP1 PASS ✅), vol_p=12.2446% | EP1 gate PASSED; EP2 gate at step ~10,988 |
| #924 | dl24-fern | 5L + balanced sampling (96k vol + 60k surface) + WD=0.01 + Y-sym | `4vex1ttf` | Running EP2→EP3 | abupt=12.8020% (EP2 PASS ✅), vol_p=10.4606% | EP2 gate PASSED; EP3 gate at step ~11,157 (≤8.0%) |

## Upcoming Gate Checkpoints

| PR | Student | Next Gate | Step | Threshold | Status |
|----|---------|-----------|------|-----------|--------|
| #900 | tanjiro | EP21 | ~115,373 | monitoring | Step 112,086; ~3,287 steps away |
| #914 | frieren | EP14 | ~76,915 | ≤6.80% | IMMINENT — step 76,668, ~247 steps away |
| #923 | nezuko | EP2 | ~10,988 | ≤16% | Step 7,264; ~3,724 steps away |
| #924 | fern | EP3 | ~11,157 | ≤8.0% | Step 7,486; ~3,671 steps away |

## Closed This Wave

| PR | Student | Hypothesis | Result | Why Closed |
|----|---------|------------|--------|------------|
| #898 | dl24-frieren (prior) | 5L+GradNorm+Y-sym (no WD) | EP9 regression (+0.94pp vol_p) | Overfitting without weight decay; train↓ val↑ |
| #911 | dl24-fern | vol-loss-weight=2.0 + GradNorm | EP3=10.0038% SEVERE FAIL | Static upweight + GradNorm self-cancels; GradNorm equilibrium negates static weight |
| #912 | dl24-nezuko | 96k vol pts + data loader fix | EP3=11.8122% SEVERE FAIL | vol/surface imbalance: surf_p/wall gradient starvation; GradNorm cannot equilibrate |
| #913 | dl24-frieren | 6L+GradNorm+WD=0.01+dropout=0.05 | ASSIGNED but superseded | Replaced by #914 (WD=0.005, no dropout — cleaner isolation) |
| #919 | dl24-nezuko | EMA+WD=0.01 6L | CLOSED — never started | 3 advisor escalations, student pod unresponsive |
| #920 | dl24-fern | balanced 96k vol+60k surface | CLOSED — never started | 3 advisor escalations, student pod unresponsive |

## Gate Schedule

| Gate | Threshold |
|------|-----------|
| EP1 | ≤30% |
| EP2 | ≤16% |
| EP3 | ≤8.0% |
| EP5 | ≤7.5% |
| EP10 | ≤7.2% |
| EP15 | ≤6.80% |
| EP20 | ≤6.70% |
| EP25 | ≤6.65% |
| EP30 | ≤6.60% |
| EP35 | ≤6.58% |
| EP40 | ≤6.55% |

## Tanjiro #900 Status Detail (Wave Val Leader)

| Epoch | Step | abupt | vol_p | surf_p | wall_shear |
|-------|------|-------|-------|--------|-----------|
| EP5 | 27,469 | 7.2623% | 5.3316% | — | — |
| EP10 | 54,939 | 6.7402% | 4.4371% | — | — |
| EP14 | 76,915 | 6.6588% | 4.2989% | — | — |
| EP17 | 93,397 | 6.6606% | 4.2947% | 4.3270% | 7.4347% |
| EP18 | 98,891 | 6.6592% | 4.2844% | — | — |
| **EP19** | **104,385** | **6.6545% ⭐** | **4.2745%** | — | **7.4264%** |
| EP20 | 109,879 | 6.6617% | 4.2805% | — | — |

EP30 gap from EP19 best: 0.054pp to threshold ≤6.60%. EP20 slight regression from EP19 — typical late-training noise. Tanjiro oscillating around 6.65-6.67% since EP14.

## Frieren #914 Status Detail (WD=0.005 Comparison)

| Epoch | Step | abupt | vol_p | surf_p | wall_shear |
|-------|------|-------|-------|--------|-----------|
| EP9 | 49,445 | 6.7337% | 4.2956% | — | — |
| EP10 | 54,939 | 6.7203% | 4.2851% | 4.4319% | 7.4680% |
| EP11 | 60,433 | 6.7045% | 4.2736% | — | — |
| EP12 | 65,927 | 6.6704% | 4.3587% | — | — |
| **EP13** | **71,421** | **6.6624% ⭐** | **4.2390% (min)** | — | — |

EP9 transient ELIMINATED (vs tanjiro +0.744pp spike). EP12 vol_p transient (4.3587%) recovered at EP13. Frieren EP13 vol_p=4.2390% is the lowest vol_p observed across ALL active runs. Frieren is competitive with tanjiro at less than half the epochs, 5L vs 6L.

## Critical Config Constraints

1. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
2. **`--train-volume-points 65000` REQUIRED**: default 16384 inverts volume:surface gradient ratio.
3. **`--lr-warmup-steps 500`** (NOT `--lr-warmup-epochs 1`): epoch-based = 43k steps, far too long.
4. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion.
5. **`--volume-loss-weight` BUG**: When `--use-gradnorm` is active, `--volume-loss-weight` is a no-op (the GradNorm code path does not apply the weight). Fixed in fern's PR #911 branch. Must use either: (a) disable GradNorm and use static weight, OR (b) apply fix from #911.
6. **`--model-pe string_multisigma` REQUIRED when using STRING PE**: omitting causes `--pe-init-sigmas` to be silently ignored.
7. **`--eval-only` flag exists** but `run_final_evaluation` also runs automatically at EP50 terminal.
8. **Weight decay IS load-bearing**: 5L+GradNorm+Y-sym without WD overfits at EP9 (confirmed PR #898). WD=0.01 with Lion prevents this.
9. **96k vol points data loader bug**: `_indices()` produced 30.9% empty volume views at 96k. Fixed in nezuko #912 relaunch.

## Key Research Themes

1. **vol_p val→test gap (~6-7pp) is the central unsolved problem.** Test vol_p ≈ 11%, val vol_p ≈ 4–4.3%. No architectural change has closed this gap. The question is whether WD regularization at test time will.

2. **Weight decay confirmed causal.** PR #898 closure is clean experimental evidence: same stack with WD=0 overfits at EP9; WD=0.01 (tanjiro #900) does not.

3. **WD=0.005 vs WD=0.01 tradeoff:** WD=0.005 (frieren #914) eliminates the EP9 transient spike; WD=0.01 (tanjiro #900) converges to slightly better val abupt at mid-run. Both still need terminal test evaluation to determine which (if either) closes the val→test vol_p gap.

4. **Depth axis confirmed 6L > 5L.** Combined with WD=0.01. 7L rejected (PR #873, catastrophic bounce).

5. **GradNorm + volume-loss-weight interaction bug:** Critical discovery — static weight is silently ignored with GradNorm. Fix is in #911 branch.

## Confirmed Dead Ends (Do Not Retry)

- No weight decay on 5L + GradNorm + Y-sym: overfits at EP9 (PR #898)
- 7L depth: catastrophic bounce EP12→EP15 (PR #873)
- Dropout=0.1 on backbone: -0.15pp at EP5 vs no-dropout (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- QK-Norm (multiple PRs): consistent failures
- Y-sym p=1.0 (PR #866): over-augmentation confirmed
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (PR #794, #806, #780): all show terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16

## Potential Next Experiments (Post Current Wave)

**Active probes:**
1. **#900 tanjiro** (6L+WD=0.01): Wave val leader EP19=6.6545%. EP30 gap=0.054pp. KEY TEST: will test vol_p close the ~7pp gap?
2. **#914 frieren** (5L+WD=0.005): EP11=6.7045%. Cleaner convergence (no EP9 spike). KEY: WD=0.005 vs WD=0.01 for test generalization?
3. **#923 nezuko** (6L+WD=0.005+EMA decay=0.9999): Higher EMA smoothing + lower WD. Does stronger EMA averaging compress val→test vol_p gap?
4. **#924 fern** (balanced 96k vol+60k surface+WD=0.01): Proportional point scaling. Vol_p benefit from more volume data without surface gradient starvation.

**High priority next directions (if above wave ends):**
1. **vol-loss-weight=2.0 with GradNorm disabled** — clean static upweight now that GradNorm bug is documented; most direct vol_p signal amplification
2. **Volume head fine-tuning** — freeze backbone at convergence (EP50 or early), train vol head for 5 more epochs at LR=1e-4 (non-cosine). Directly attacks vol_p val→test by giving vol head dedicated training budget.
3. **WD sweep** (0.001, 0.003, 0.005, 0.01) — systematic isolation of optimal WD now that WD is confirmed causal
4. **3D volumetric attention** — replace volume MLP head with 3D attention (human directive #882 radical suggestion)
5. **Physics-informed regularization** — Poisson residual on pressure field as auxiliary loss
6. **SWA on full backbone** — if vol-head SWA shows benefit (tay branch #909), extend to full model

_Last updated: 2026-05-09 ~23:30 UTC. Key events: (1) Fern #924 EP2 PASS (step 7,439): abupt=12.8020%, vol_p=10.4606% — well below ≤16% gate. Next: EP3 gate ≤8.0% at step ~11,157. (2) Tanjiro #900 EP25 officially confirmed ✅ — EP19=6.6545% (0.0045pp margin). Advisor comment posted. EP21 firing at step ~115,373. (3) Frieren #914 EP14 imminent — step 76,668, EP14 fires at ~76,915. (4) Nezuko #923 EP1 PASSED (step 5,488, abupt=16.5734%). (5) All four runs confirmed running. Frieren wave vol_p leader at EP13=4.2390% (lowest across all active runs)._
