# SENPAI Research State

- 2026-05-10 ~03:00 UTC

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

**Val wave leader:** frieren #914 EP24 abupt=**6.5958%**, vol_p=4.1761% (val only — test not yet run)

**Central unsolved problem:** val vol_p ≈ 4.0–4.3%, test vol_p ≈ 11% — systematic +7pp val→test gap confirmed across ALL completed long runs. All active experiments are designed to close this gap.

**Vol-loss-weighting direction CLOSED:** Both PR #911 (GradNorm+static weight=no-op) and PR #936 (no-GradNorm+static weight=harmful) failed. The val→test gap is NOT a training-time loss signal problem. It is almost certainly a covariate shift / data distribution problem.

## Active Experiments (2026-05-10 ~03:00 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Known Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #900 | dl24-tanjiro | 6L + GradNorm α=0.5 + WD=0.01 + Y-sym | `os6v64lq` | Running, EP29 complete, EP30 imminent | abupt=6.6065% (EP27 best); EP28=6.6231%, EP29=6.6252% regression | EP30 gate ≤6.60% NOT expected; test eval on EP27 checkpoint pending |
| #914 | dl24-frieren | 5L + GradNorm α=0.5 + WD=0.005 + Y-sym | `wdxtdmhy` | Running EP24+; step=133,638 | abupt=**6.5958%** (EP24, val wave leader ⭐), vol_p=4.1761% | EP21-EP23 oscillation was transient; EP24 broke through cleanly; EP25 gate imminent |
| #938 | dl24-fern | 5L + balanced 96k vol + 60k surface + WD=0.005 + GradNorm α=0.5 + Y-sym | `md3vhhd8` | Running, EP1=36.43% (normal early training); step=4,751 | EP5 gate ≤7.5% expected ~03:00Z | Watch surf_p — gradient starvation risk; PR #912 warning: surf_p +3.65pp without proportional surface |
| #939 | dl24-nezuko | 6L + GradNorm α=0.5 + WD=0.005 + Y-sym (6L×WD cross) | TBD | Assigned, not yet started | — | Cross-product of tanjiro #900 (6L) and frieren #914 (WD=0.005); untested quadrant |

## Key Closed/Falsified Hypotheses This Wave

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #936 | vol-loss-weight=2.0 NO GradNorm | EP5 FAIL abupt=9.010% (+1.51pp), vol_p=5.691% WORSE than frieren | Static vol upweight without GradNorm actively harms; direction fully closed |
| #911 | vol-loss-weight=2.0 WITH GradNorm | EP3 FAIL (GradNorm self-cancels static weight) | Static vol upweight + GradNorm is a no-op; must choose one |
| #912 | 96k vol pts alone | EP3=11.8122% SEVERE FAIL (surf_p +3.65pp) | Vol/surface sampling must be proportional; 1.6:1 ratio needed |
| #923 | 6L + EMA decay=0.9999 + WD=0.005 | CLOSED — never started (3 escalations) | Student unresponsive |
| #924 | Balanced 96k vol + 60k surface + WD=0.01 | CLOSED — never started (3 escalations) | Revived as #938 with WD=0.005 |
| #919 | EMA+WD=0.01 6L | CLOSED — never started | Student unresponsive |
| #898 | 5L+GradNorm+Y-sym (no WD) | EP9 regression — OVERFITTING | WD is necessary, not optional |
| #873 | 7L depth | Catastrophic bounce EP12→EP15 | 7L too deep for this architecture |
| #899 | Dropout=0.1 | −0.15pp at EP5 | Dropout incompatible with this architecture |
| #866 | Y-sym p=1.0 | Over-augmentation confirmed | p=0.5 is optimal |
| #874 | GradNorm α=0.75 | Catastrophic instability EP16 | α=0.5 is the ceiling |

## Key Insights

1. **Weight decay is load-bearing.** PR #898 (no WD) overfits at EP9; WD=0.01 (tanjiro #900) prevents it entirely. WD is a necessary ingredient.

2. **WD=0.005 (frieren) beats WD=0.01 (tanjiro) on val.** WD=0.005 eliminates the EP9 +0.744pp transient spike. Both need terminal test evaluation to determine which closes the val→test vol_p gap.

3. **Vol-loss-weighting direction FULLY EXHAUSTED.** PR #911 (with GradNorm = no-op) + PR #936 (without GradNorm = actively harmful). Static upweighting of vol_p loss does not close the val→test gap. The gap is a covariate shift / data distribution problem, NOT a loss signal problem.

4. **96k vol points: vol_p benefit is real but requires proportional surface scaling.** PR #912 confirmed vol_p −0.35pp at EP3, but surf_p/wall regressed +3.65pp/+5.53pp due to gradient starvation. Fix is 96k vol + 60k surface (1.6:1 ratio) — now being tested in PR #938.

5. **EP9 transient spike:** WD=0.01 (tanjiro) had +0.744pp spike; WD=0.005 (frieren) had zero spike. The spike origin is likely an LR schedule inflection point interacting with weight decay magnitude.

6. **Frieren EP24 oscillation resolved.** EP21-EP23 showed a +0.015pp micro-uptick before EP24 broke through to new run best 6.5958%. Transient oscillations do NOT indicate plateau.

## Gate Schedule

| Gate | Threshold |
|------|-----------|
| EP5  | ≤7.5% |
| EP10 | ≤7.2% |
| EP15 | ≤6.80% |
| EP20 | ≤6.70% |
| EP25 | ≤6.65% |
| EP30 | ≤6.60% |
| EP35 | ≤6.58% |
| EP40 | ≤6.55% |

## Critical Config Constraints

1. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
2. **`--train-volume-points 65000` (or higher)** REQUIRED: default 16384 inverts volume:surface gradient ratio.
3. **`--lr-warmup-steps 500`** (NOT `--lr-warmup-epochs 1`): epoch-based = 43k steps, far too long.
4. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion optimizer.
5. **`--volume-loss-weight` BUG**: When `--use-gradnorm` is active, `--volume-loss-weight` is a no-op. Must use either: (a) disable GradNorm and use static weight, OR (b) apply fix from #911 branch.
6. **`--model-pe string_multisigma` REQUIRED when using STRING PE**: omitting causes `--pe-init-sigmas` to be silently ignored.
7. **96k vol points data loader bug**: `_indices()` produced 30.9% empty volume views at 96k. Fixed in nezuko #912 relaunch — must verify fix is present in any new 96k run.

## Confirmed Dead Ends (Do Not Retry)

- No weight decay: overfits at EP9 (PR #898)
- 7L depth: catastrophic bounce (PR #873)
- Dropout=0.1: consistent degradation (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- vol-loss-weight=2.0 WITHOUT GradNorm: actively harmful, EP5=9.010% (PR #936)
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16

## Potential Next Directions (After Current Wave Completes)

**Targeting val→test vol_p gap (primary unsolved problem):**
1. **Data distribution alignment** — The gap is likely covariate shift. Analyze train vs test aerodynamic configurations. Do augmentations that push the model toward OOD robustness (random geometric perturbations, stochastic mesh sampling, flow regime mixing).
2. **SWA (Stochastic Weight Averaging)** — Average weights across late epochs; known to improve generalization for OOD test sets. Low implementation cost, well-motivated.
3. **WD systematic sweep** (0.001, 0.003, 0.005, 0.01) — now that WD is confirmed causal, find the optimal value more precisely
4. **Volume head fine-tuning** — freeze backbone at convergence, train vol head for additional epochs at lower LR; directly attacks vol_p val→test by giving vol head dedicated budget
5. **Physics-informed regularization** — Poisson residual on pressure field as auxiliary loss; direct physics constraint for vol_p generalization
6. **Label smoothing / uncertainty** — soft vol_p targets or uncertainty weighting to reduce overfitting to exact training values
7. **Temperature scaling post-hoc** — calibrate vol_p predictions after training to reduce val→test confidence gap
8. **3D volumetric attention** — replace volume MLP head with spatially-aware attention (radical architecture change)
9. **LR schedule ablation** — cosine vs linear decay: does cosine annealing favor val at expense of test generalization?

_Last updated: 2026-05-10 ~03:00 UTC. Key events since last update: (1) PR #936 (nezuko vol-loss-weight=2.0 no-GradNorm) FALSIFIED — EP5=9.010% gate miss, vol_p=5.691% WORSE than baseline, auto-killed and CLOSED. (2) Vol-loss-weighting direction now FULLY CLOSED (both #911 and #936 failed). (3) PR #914 (frieren) new run best EP24=6.5958% — beats EP19 by 0.0098pp, wave val leader confirmed. (4) PR #900 (tanjiro) EP27=6.6065% best, EP28-29 regression; EP30 imminent, unlikely to gate at ≤6.60%. (5) PR #938 (fern) launched, run md3vhhd8, EP1=36.4%, EP5 expected ~03:00Z. (6) PR #939 (nezuko) assigned: 6L + WD=0.005 + GradNorm α=0.5 + Y-sym — untested cross-product of tanjiro's 6L depth and frieren's WD=0.005 regularization._
