# SENPAI Research State

- 2026-05-09 ~22:30 UTC

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

**Val wave leader:** frieren #914 EP19 abupt=**6.6056%**, vol_p=4.1732% (val only — test not yet run)

**Central unsolved problem:** val vol_p ≈ 4.0–4.3%, test vol_p ≈ 11% — systematic +7pp val→test gap confirmed across ALL completed long runs. All active experiments are designed to close this gap.

## Active Experiments (2026-05-09 ~22:30 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Known Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #900 | dl24-tanjiro | 6L + GradNorm α=0.5 + WD=0.01 + Y-sym | `os6v64lq` | Running EP28→EP30 | abupt=6.6065% (EP27 best), EP28=6.6231% regression | EP30 gate ≤6.60% very unlikely; advisor: continue to EP30 then test eval on EP27 checkpoint |
| #914 | dl24-frieren | 5L + GradNorm α=0.5 + WD=0.005 + Y-sym | `wdxtdmhy` | Running EP19+ | abupt=6.6056% (EP19 wave leader ⭐), vol_p=4.1732% | Last known EP19; EP20+ status unknown; EP9 transient ELIMINATED |
| #936 | dl24-nezuko | 5L + vol-loss-weight=2.0 + NO GradNorm + WD=0.005 + Y-sym | `6gd9u34e` | Running EP2→EP5 | abupt EP2=11.858%, vol_p=6.805% | EP5 gate ≤7.5% upcoming; strong descent from EP2 |
| #938 | dl24-fern | 5L + balanced 96k vol + 60k surface + WD=0.005 + GradNorm α=0.5 + Y-sym | TBD | Newly assigned 2026-05-09T22:30Z | — | Revives closed #924; combines PR#912 vol_p benefit with proportional surf fix + WD=0.005 |

## Key Closed/Falsified Hypotheses This Wave

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
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

3. **GradNorm + static vol-loss-weight = self-cancelling.** Proven in PR #911. The fix is either: (a) disable GradNorm and use static weight (PR #936 strategy), or (b) proportional point sampling to provide natural vol signal without loss weighting (PR #938 strategy).

4. **96k vol points: vol_p benefit is real but requires proportional surface scaling.** PR #912 confirmed vol_p −0.35pp at EP3, but surf_p/wall regressed +3.65pp/+5.53pp due to gradient starvation. Fix is 96k vol + 60k surface (1.6:1 ratio) — now being tested in PR #938.

5. **EP9 transient spike:** WD=0.01 (tanjiro) had +0.744pp spike; WD=0.005 (frieren) had zero spike. The spike origin is likely an LR schedule inflection point interacting with weight decay magnitude.

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
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16

## Potential Next Directions (After Current Wave Completes)

1. **WD systematic sweep** (0.001, 0.003, 0.005, 0.01) — now that WD is confirmed causal, find the optimal value more precisely
2. **Volume head fine-tuning** — freeze backbone at convergence, train vol head for additional epochs at lower LR; directly attacks vol_p val→test by giving vol head dedicated budget
3. **Physics-informed regularization** — Poisson residual on pressure field as auxiliary loss; direct physics constraint for vol_p generalization
4. **Label smoothing / uncertainty** — soft vol_p targets or uncertainty weighting to reduce overfitting to exact training values
5. **Temperature scaling post-hoc** — calibrate vol_p predictions after training to reduce val→test confidence gap
6. **SWA (Stochastic Weight Averaging)** — average weights across late epochs; known to improve generalization, especially for out-of-distribution test
7. **3D volumetric attention** — replace volume MLP head with spatially-aware attention (radical architecture change)
8. **LR schedule ablation** — cosine vs linear decay: does cosine annealing favor val at expense of test generalization?

_Last updated: 2026-05-09 ~22:30 UTC. Key events since last update: (1) PR #923 and #924 CLOSED (never started). (2) PR #936 (nezuko vol-loss-weight=2.0 no-GradNorm) assigned and running, EP2=11.858%. (3) PR #900 (tanjiro) reached EP27=6.6065% wave leader, EP28 regression; advisor instructed continue to EP30 then test eval EP27 checkpoint. (4) PR #914 (frieren) reached EP19=6.6056% current val wave leader. (5) PR #938 (fern) newly assigned: balanced 96k vol + 60k surface + WD=0.005._
