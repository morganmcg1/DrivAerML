# SENPAI Research State

- 2026-05-10 ~06:00 UTC

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

**Val wave leader:** frieren #914 EP26 abupt=**6.5932%**, vol_p=4.2099% (val only — test not yet run)

**Central unsolved problem:** val vol_p ≈ 4.0–4.3%, test vol_p ≈ 11% — systematic +7pp val→test gap confirmed across ALL completed long runs. All active experiments are designed to close this gap.

**Vol-loss-weighting direction CLOSED:** Both PR #911 (GradNorm+static weight=no-op) and PR #936 (no-GradNorm+static weight=harmful) failed. The val→test gap is NOT a training-time loss signal problem. It is almost certainly a covariate shift / data distribution problem.

## Active Experiments (2026-05-10 ~06:00 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Known Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #900 | dl24-tanjiro | 6L + GradNorm α=0.5 + WD=0.01 + Y-sym | `os6v64lq` | **EP30 GATE MISS** (abupt=6.6135% > 6.60%); test eval on EP27 checkpoint PENDING | abupt=6.6065% (EP27 best); EP28-29-30 regression | Gate miss comment posted. Tanjiro directed to run test eval on EP27 checkpoint (step 148,337: abupt=6.6065%, vol_p=4.2423%, surf_p=4.3094%, wall=7.3713%) AND EP19 backup. |
| #914 | dl24-frieren | 5L + GradNorm α=0.5 + WD=0.005 + Y-sym | `wdxtdmhy` | Running EP26+; wave val leader | abupt=**6.5932%** (EP26, wave val leader ⭐), vol_p=4.2099% | EP35 gate (≤6.58%) imminent; need −0.0132pp more from EP27+ |
| #939 | dl24-nezuko | 6L + GradNorm α=0.5 + WD=0.005 + Y-sym (6L×WD cross) | `yfitnqia` | Running; step=3,874 (~EP0.7), EP1 sanity pending | No val yet (pre-EP1) | EP1 gate ≤30%; config: 6L STRING, GradNorm, WD=0.005, Y-sym, 40k surf/65k vol |
| #940 | dl24-fern | Balanced 96k vol + 60k surf + lr=5e-5 + warmup=1000 (diagnostic for #938 failure) | TBD | Just assigned; not yet started | — | If EP5 clears ≤7.5%: LR scaling is root cause of #938 failure. If EP5 fails again: balanced sampling is fundamentally broken. EP1 train_loss diagnostic: should be < 1.0 if LR fix works (vs 1.567 in #938). |

## Key Closed/Falsified Hypotheses This Wave

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #938 | 5L + balanced 96k vol + 60k surf + lr=1e-4 (default) | **KILLED EP5=15.22%** (gate ≤7.5%). EP1 train_loss=1.567 (3.7× frieren's 0.427) | Balanced sampling puts GradNorm+Lion in different gradient regime at default lr; must halve lr to 5e-5 |
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

4. **96k vol points: vol_p benefit is real but requires proportional surface scaling.** PR #912 confirmed vol_p −0.35pp at EP3, but surf_p/wall regressed +3.65pp/+5.53pp due to gradient starvation. Fix is 96k vol + 60k surface (1.6:1 ratio) — now being tested in PR #940 with corrected LR.

5. **Balanced sampling requires reduced lr.** PR #938 trained at 1e-4 (default): EP1 train_loss=1.567 (3.7× frieren's 0.427); EP5=15.22% >> 7.5% gate. Root cause: more diverse/harder vol points in 96k set puts GradNorm in a higher-loss regime, making the standard LR too aggressive for the Lion optimizer. Diagnostic PR #940 uses lr=5e-5 + warmup=1000 steps to compensate.

6. **EP9 transient spike:** WD=0.01 (tanjiro) had +0.744pp spike; WD=0.005 (frieren) had zero spike. The spike origin is likely an LR schedule inflection point interacting with weight decay magnitude.

7. **Tanjiro EP30 gate miss:** EP27=6.6065% was the best checkpoint; EP28-EP30 showed regression. EP30 abupt=6.6135% > 6.60% gate. Training should not be extended. Test eval on EP27 checkpoint (step 148,337) + EP19 backup will provide the first WD=0.01 test data point for vol_p gap analysis.

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
8. **Balanced 96k+60k sampling requires lr=5e-5 (not default 1e-4)**: Default LR causes EP1 train_loss ~1.5 (3.7× normal), catastrophic divergence. Always use `--lr 5e-5 --lr-warmup-steps 1000` with balanced sampling configs.

## Confirmed Dead Ends (Do Not Retry)

- No weight decay: overfits at EP9 (PR #898)
- 7L depth: catastrophic bounce (PR #873)
- Dropout=0.1: consistent degradation (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- vol-loss-weight=2.0 WITHOUT GradNorm: actively harmful, EP5=9.010% (PR #936)
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- 96k+60k balanced sampling with default lr=1e-4: catastrophic divergence EP5=15.22% (PR #938)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16

## Pending Decisions / Awaiting Student Actions

1. **Tanjiro #900**: Test eval on EP27 (step 148,337) + EP19 checkpoint — must produce terminal SENPAI-RESULT. This is the first WD=0.01 test data point for the vol_p gap analysis (Issue #882 directive).
2. **Frieren #914**: EP35 gate (≤6.58%) — currently at EP26=6.5932%, needs −0.0132pp more. Monitor for student comments.
3. **Nezuko #939**: EP1 sanity check (≤30%) then standard gate progression. Run `yfitnqia`.
4. **Fern #940**: EP1 train_loss diagnostic (should be < 1.0 to confirm LR fix) then EP5 gate (≤7.5%).

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

_Last updated: 2026-05-10 ~06:00 UTC. Key events since last update: (1) PR #900 (tanjiro) EP30 GATE MISS — abupt=6.6135% > 6.60%; gate miss comment posted; tanjiro directed to run test eval on EP27 best checkpoint (step 148,337: 6.6065%). (2) PR #914 (frieren) EP26 new run best abupt=6.5932%, wave val leader; EP35 gate imminent. (3) PR #938 (fern) KILLED EP5=15.22% — catastrophic divergence; root cause: default lr=1e-4 too aggressive for balanced 96k+60k sampling regime. (4) PR #940 (fern) NEWLY ASSIGNED — same balanced 96k+60k config but lr=5e-5 + warmup=1000 + 6L model; diagnostic to confirm whether LR scaling fixes convergence. (5) PR #939 (nezuko) run yfitnqia, step=3,874 at last check (~EP0.7), EP1 val pending._
