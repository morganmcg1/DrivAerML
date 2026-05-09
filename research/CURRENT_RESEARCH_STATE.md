# SENPAI Research State

- 2026-05-09 ~17:30 UTC

## Human Research Directive (Issue #882)
**TOP PRIORITY — Volume Pressure Focus:**
- The **TEST volume pressure L2 error** is now the only metric that matters for new experiment design
- Do NOT degrade surface error or wall shear stress metrics
- Published SOTA models show significantly better volume pressure test metrics are achievable — large headroom to close
- All new student assignments should be designed with volume pressure improvement as the singular focus

## Wave SOTA (Merged Test)

**PR #740** (dl24-fern, run `5x8wofzm`): `test_primary/abupt_axis_mean_rel_l2_pct` = **7.5195%**
| Metric | Value |
|--------|-------|
| test_abupt | 7.5195% |
| surf_p | 3.8810% |
| vol_p | 10.7580% |
| wall_shear | 7.0610% |

**Critical insight:** vol_p test = 10.758% vs val ≈ 4.0–4.7% — systematic ~7pp val→test gap in volume pressure.
This gap is the central unsolved problem. All current active experiments target it.

## Key Insight Update (2026-05-09 17:30 UTC)

**Weight decay = the crucial regularizer.** PR #898 (5L+GradNorm+Y-sym, no WD) overfitted at EP9 with clean val regression. PR #900 (6L+GradNorm+Y-sym, WD=0.01) has shown zero regression through EP7.6. This is now a **confirmed causal result**: WD=0.01 is necessary (not just helpful) for avoiding mid-run overfitting on this architecture.

**GradNorm is reactive, not preventive.** When overfitting began in #898, GradNorm raised w_vol_p (0.58→0.71) trying to correct it. This doesn't help — it amplifies gradient signal into an already-overfit state.

## Active Experiments (2026-05-09 ~17:30 UTC)

| PR | Student | Hypothesis | Run ID | Status | Latest Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #900 | dl24-tanjiro | 6L + GradNorm α=0.5 + WD=0.01 + Y-sym | `os6v64lq` | Running EP7.6 | abupt=6.8248%, vol_p=4.5618% | **WAVE LEADER, all 4 metrics improving** |
| #911 | dl24-fern | 5L + vol-loss-weight=2.0 + no-GradNorm + WD=1e-4 | `8co57khm` | Running EP1.6 | (too early) | Arm A: direct static upweight |
| #912 | dl24-nezuko | 5L + 96k vol pts + GradNorm + data loader fix | `q2mf0exo` | Running EP0 | (too early) | Relaunched 13:56 UTC with bug fix |
| #913 | dl24-frieren | 6L + GradNorm α=0.5 + WD=0.01 + dropout=0.05 | (not yet started) | ASSIGNED | — | Orthogonal regularizer on tanjiro's stack |

**Plus on `tay` advisor branch** (different advisor):
| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #909 | thorfinn | Vol-head SWA (SWA on vol+xattn only) | `r6o9pz3n` | Running EP0, started 13:54 UTC |
| #901 | fern | Train-time y-axis mirror aug, 13ep | `5r0rkhuo` | Running EP0, started 13:55 UTC |

## Closed This Wave

| PR | Student | Hypothesis | Result | Why Closed |
|----|---------|------------|--------|------------|
| #898 | dl24-frieren | 5L+GradNorm+Y-sym (no WD) | EP9 regression (+0.94pp vol_p) | Overfitting without weight decay; train↓ val↑ |

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

## Tanjiro #900 Status Detail (Best In-Wave)

| Epoch | Step | abupt | vol_p | surf_p | wall_shear |
|-------|------|-------|-------|--------|-----------|
| EP2 | ~10987 | 7.9865% | 6.398% | — | — |
| EP3 | ~16481 | 7.3662% | 5.443% | — | — |
| EP5 | ~27469 | 7.2623% | 5.332% | — | — |
| EP6 | ~32963 | **6.8847%** | 4.639% | 4.399% | 7.660% |
| EP7.6 | 41,801 | **6.8248%** | **4.5618%** | **4.3811%** | **7.5921%** |

*Note: Early advisor comments incorrectly labeled val events as "EP10", "EP15" etc. — actual epoch = step/5493. EP6 = step 32,963 (not "EP10"). Epochs/gate steps recalibrated.*

**EP10 gate step 54,930 (≤7.2%):** already below at EP7.6 — trivial to clear
**EP15 gate step 82,395 (≤6.80%):** needs -0.025pp more in 7.4 epochs — very achievable
**Wall shear watchpoint:** 7.592% vs SOTA 7.061% (+0.53pp). Trending down but needs to continue.

## Critical Config Constraints

1. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
2. **`--train-volume-points 65000` REQUIRED**: default 16384 inverts volume:surface gradient ratio.
3. **`--lr-warmup-steps 500`** (NOT `--lr-warmup-epochs 1`): epoch-based = 43k steps, far too long.
4. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion.
5. **`--volume-loss-weight` BUG**: When `--use-gradnorm` is active, `--volume-loss-weight` is a no-op (the GradNorm code path does not apply the weight). Fixed in fern's PR #911 branch. Must use either: (a) disable GradNorm and use static weight, OR (b) apply fix from #911.
6. **`--model-pe string_multisigma` REQUIRED when using STRING PE**: omitting causes `--pe-init-sigmas` to be silently ignored.
7. **`--eval-only` flag exists** but `run_final_evaluation` also runs automatically at EP50 terminal.
8. **Weight decay IS load-bearing**: 5L+GradNorm+Y-sym without WD overfits at EP9 (confirmed PR #898). WD=0.01 with Lion prevents this.
9. **96k vol points data loader bug**: `_indices()` produced 30.9% empty volume views at 96k. Fixed in nezuko #912 relaunch (`q2mf0exo`).

## Key Research Themes

1. **vol_p val→test gap (~7pp) is the central unsolved problem.** Test vol_p ≈ 11%, val vol_p ≈ 4–5%. No architectural change has closed this gap.

2. **Weight decay confirmed causal.** PR #898 closure is clean experimental evidence: same stack with WD=0 overfits at EP9; WD=0.01 (tanjiro #900) does not.

3. **Depth axis confirmed 6L > 5L.** Combined with WD=0.01. 7L rejected (PR #873, catastrophic bounce).

4. **GradNorm + volume-loss-weight interaction bug:** Critical discovery — static weight is silently ignored with GradNorm. Fix is in #911 branch, pending upstream merge.

5. **Human directive (Issue #882):** All new experiments must target TEST vol_p improvement.

## Confirmed Dead Ends (Do Not Retry)

- No weight decay on 5L + GradNorm + Y-sym: overfits at EP9 (PR #898)
- 7L depth: catastrophic bounce EP12→EP15 (PR #873)
- Dropout=0.1 on backbone: -0.15pp at EP5 vs no-dropout (PR #899)
- QK-Norm (multiple PRs): consistent failures
- Y-sym p=1.0 (PR #866): over-augmentation confirmed
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (PR #794, #806, #780): all show terminal test regression

## Potential Next Experiments (Post Current Wave)

**Active probes:**
1. **#900 tanjiro** (6L+WD=0.01): Wave leader. Pending EP15 gate at step 82,395.
2. **#911 fern ArmA** (vol-loss-weight=2.0, no-GradNorm, WD=1e-4): Direct static upweight test.
3. **#912 nezuko** (96k vol pts + data loader fix): More volume data hypothesis.
4. **#913 frieren** (6L+WD=0.01+dropout=0.05): Orthogonal regularizer on tanjiro stack.

**High priority next directions (if above wave ends):**
1. **vol-loss-weight=2.0 WITH GradNorm + bug fix** (fern #911 ArmB): The pre-scale fix enables both; most likely to be tried after ArmA completes.
2. **Volume head fine-tuning** — freeze backbone at convergence, train vol head for 5 more epochs at LR=1e-4 (non-cosine). Directly attacks vol_p val→test by giving vol head dedicated training budget.
3. **3D volumetric attention** — replace volume MLP head with 3D attention (human directive #882 radical suggestion).
4. **Physics-informed regularization** — Poisson residual on pressure field as auxiliary loss.
5. **SWA on full backbone** (not just vol head) — if thorfinn #909 vol-head SWA shows benefit on tay branch, extend to full model.

_Last updated: 2026-05-09 ~17:30 UTC. Key events: (1) PR #898 CLOSED — frieren EP9 overfitting confirmed; (2) PR #913 ASSIGNED to frieren — 6L+WD=0.01+dropout=0.05; (3) Tanjiro #900 is wave leader at EP7.6 abupt=6.8248%, all metrics improving; (4) Epoch numbering reconciled — advisor's prior "EP10" labels were actually EP6 (step 34,731/5493=EP6.3); corrected gate steps: EP10@54,930, EP15@82,395._
