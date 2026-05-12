# SENPAI Research State
- 2026-05-12 ~20:30 UTC

## Human Research Directive (Issue #882)
**TOP PRIORITY — Volume Pressure Focus:**
- The **TEST volume pressure L2 error** is the only metric that matters for new experiment design
- Do NOT degrade surface error or wall shear stress metrics
- Published SOTA models show significantly better volume pressure test metrics are achievable — large headroom to close
- All new student assignments must be designed with volume pressure improvement as the singular focus

## DATASET ARTIFACT RESOLVED (2026-05-12) — Issue #1053

**CRITICAL — All prior val→test vol_p gap analysis is INVALIDATED.**

The persistent +7–8pp val→test volume pressure gap reported across all prior experiments was a **DATASET ARTIFACT** caused by a case-split bug in the pre-20260511 dataset. Under the corrected split (`rawcanon_20260511`), the gap disappears entirely.

**Corrected dataset path (MANDATORY for all new and active experiments):**
`/mnt/pvc/Processed/drivaerml_processed_rawcanon_20260511`

**Corrected eval parameters:** `eval_surface_points=65536`, `eval_volume_points=65536` (chunk sizes, not caps), 34 val cases / 7,295 views, 50 test cases / 11,091 views.

**Impact on prior results:**
- PRs #972 and #968, previously labeled FALSIFIED, are the **TOP TWO PERFORMERS** on the corrected split.
- PR #740 (old SOTA at 7.5195%) re-evaluates to 8.165% — **rank 22** under the corrected split.
- All "AXIS FULLY CLOSED" labels on sampling strategies are **RETRACTED**.
- The 15+ "falsified" interventions were not tested on a valid dataset split.

## Wave SOTA (Corrected Split — rawcanon_20260511)

**PR #972** (SDF-stratified far-field sampling, α=2.0, run `56bcqp3m`, eval `zxnhtagj`): `abupt_axis_mean_rel_l2_pct` = **5.844%**

| Metric | Value |
|--------|-------|
| test_abupt | **5.844%** |
| test_surf_p | 3.577% |
| test_vol_p | **3.643%** |
| test_wss | 6.727% |

> Note: PR #972 is CLOSED (was falsely closed under old dataset). The SDF-stratified sampling technique is the current SOTA and must be treated as the baseline configuration for new experiments.

**Top 5 on corrected split:**

| Rank | PR | test_ABUPT | test_vol_p | Notes |
|------|----|------------|------------|-------|
| 1 | **#972** | **5.844%** | 3.643% | SDF-stratified far-field sampling (α=2.0) — **CLOSED, FALSELY** |
| 2 | #968 | 5.986% | 3.957% | Stochastic vol subsampling — **CLOSED, FALSELY** |
| 3 | #880 | 6.010% | 4.501% | (see BASELINE.md for full run IDs) — CLOSED |
| 4 | #958 | 6.107% | 3.818% | — CLOSED |
| 5 | #939 | 6.242% | — | — CLOSED |
| … | … | … | … | … |
| 22 | #740 | 8.165% | 13.660% | Old "SOTA" — NOW ARTIFACT |

## Active Experiments (2026-05-12 ~20:30 UTC)

> **ACTION REQUIRED FOR ALL ACTIVE EXPERIMENTS**: Switch dataset to corrected split path
> `/mnt/pvc/Processed/drivaerml_processed_rawcanon_20260511`
> Existing runs trained on old split should be restarted from scratch on corrected split.

| PR | Student | Hypothesis | Run ID | Status | Latest Val | Notes |
|----|---------|------------|--------|--------|------------|-------|
| #1025 | dl24-frieren | **Vol-token LayerNorm WITHOUT GradNorm** | `ttnva184` | Running — EP23.2 | val_abupt=6.3639%, val_vol_p=3.5467% | **MUST RESTART** on corrected dataset. Old-split results not comparable. |
| #1035 | dl24-fern | **Independent vol_p transformer tower** | `1dijs6g1` | Running — EP2.02 | val_abupt=10.378%, val_vol_p=11.213% | **MUST RESTART** on corrected dataset. Early enough to restart cleanly. |
| TBD | dl24-tanjiro | **DETR-style learned query positions** | TBD | Assigning | — | **ASSIGN ON CORRECTED DATASET from the start.** |
| TBD | dl24-nezuko | **Pure FNO for vol_p** | TBD | Assigning | — | **ASSIGN ON CORRECTED DATASET from the start.** |

## Key Insights (Post-Artifact-Resolution)

1. **The val→test vol_p gap was entirely artificial.** Under `rawcanon_20260511`, test_vol_p tracks val_vol_p closely (~3.6–4.0% range). No covariate shift hypothesis needed.

2. **SDF-stratified far-field sampling (PR #972) is the SOTA technique.** α=2.0 upweighting of far-field volume points yields test_abupt=5.844%. This must be treated as the baseline config.

3. **Stochastic vol subsampling (PR #968) is the #2 technique.** Fresh random draw every batch: test_abupt=5.986%. Complementary to SDF stratification — combining them is Tier 1.

4. **The 15+ "falsified" interventions need re-evaluation.** WD variants, GradNorm α-variants, EMA, BBox norm, DropPath, vol coord noise, etc. — all were tested on a broken dataset. Some may still be dead ends; others may show benefit on the corrected split.

5. **GradNorm + AdamW = catastrophic instability**: CONFIRMED HARDWARE-LEVEL. Not dataset-dependent. Use Lion when using GradNorm.

6. **String multisigma PE (5-octave) is confirmed best.** σ=[0.25, 0.5, 1.0, 2.0, 4.0]. Not dataset-dependent.

7. **Vol-token LayerNorm** (PR #1025) showed val_vol_p=3.547% on old dataset at EP23 — promising but needs re-run on corrected split to determine true rank.

## Tier 1 Follow-Ups (Highest Priority — Assign Immediately)

1. **SDF-stratified + Stochastic combined** — combine PR #972's α=2.0 SDF upweighting with PR #968's fresh random draw every batch. Neither was tried together. Expected: ≤5.7% test_abupt.
2. **SDF-stratified + dedicated vol decoder (6L vol tower)** — combine the #1 sampling technique with the independent vol tower architecture (#1035). Tests whether the vol tower benefit stacks with better sampling.
3. **SDF α sweep on corrected split** — try α=1.5, 3.0, 4.0 to identify optimal far-field upweighting. α=2.0 was the only value tested.

## Tier 2 Follow-Ups (After Tier 1 Results)

4. **Re-run vol-token LN (PR #1025 config) on corrected split from scratch** — corrected-split val_vol_p likely to be ~3.5% (already promising on old split).
5. **SDF + 5L depth** — lighter model may generalize better with strong SDF sampling.
6. **Ensemble refresh on corrected split** — new ensemble from top corrected-split checkpoints (#972, #968, #880, #958) could push below 5.5%.
7. **WD=0.005 re-test on corrected split** — may show genuine benefit now that the dataset artifact is removed.
8. **EMA decay=0.999 re-test on corrected split** — previously falsified but could benefit from clean evaluation.

## Gate Schedule

| Gate | Standard Threshold | Steps (std, bs=2, DDP8) | Steps (bs=1, DDP8) |
|------|--------------------|---------------------|----------------------|
| EP5  | ≤7.5% | ~27,469 | ~54,930 |
| EP10 | ≤7.2% | ~54,938 | ~109,860 |
| EP15 | ≤6.80% | ~82,407 | ~164,790 |
| EP20 | ≤6.70% | ~109,876 | ~219,720 |
| EP25 | ≤6.65% | ~137,345 | — |
| EP30 | ≤6.60% | ~164,814 | — |

> Note: Gate thresholds calibrated to old dataset. With corrected split showing test_abupt=5.844% as current SOTA, consider tightening gates in future rounds (e.g., EP20 ≤6.30%, EP30 ≤5.95%).

## Critical Config Constraints

1. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
2. **`--train-volume-points 65000` (or higher)** REQUIRED: default 16384 inverts volume:surface gradient ratio.
3. **`--lr-warmup-epochs 1` NOT `--lr-warmup-steps 500`**: epoch-based warmup is correct at 6L 65k.
4. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion optimizer.
5. **`--volume-loss-weight` BUG**: When `--use-gradnorm` is active, `--volume-loss-weight` is a no-op.
6. **`--model-pe string_multisigma` REQUIRED when using STRING PE**: omitting causes `--pe-init-sigmas` to be silently ignored.
7. **`--pe-init-sigmas` must be COMMA-separated**: `0.25,0.5,1.0,2.0,4.0` NOT space-separated.
8. **lion-pytorch pod environment drift**: `ModuleNotFoundError: No module named 'lion_pytorch'` can occur. Fix: `uv pip install --system lion-pytorch` (resolves to 0.2.4).
9. **Kill threshold operator is `<`**: NOT `>`. PR #945 had operator bug causing inverted logic.
10. **EMA decay=0.999 (NOT 0.9999)**: 0.9999 gives ~10,000-step lookback (too slow); 0.999 gives ~1,000-step lookback.
11. **Vol curriculum steps/epoch** (measured from chunked data loading, 400 cases × views ÷ 8 ranks ÷ batch 2): 16,384→10,864; 32,768→5,435; 49,152→3,625; 65,536→2,720.
12. **Steps/epoch at bs=1 DDP8**: ~10,975–10,986.
13. **Steps/epoch at bs=2 DDP8 (standard)**: ~10,975.

## Confirmed Dead Ends (Hardware/Architecture — Not Dataset-Dependent)

These results are confirmed on multiple runs and are likely dataset-independent:

- No weight decay: overfits at EP9 (PR #898)
- 7L depth without full regularization: catastrophic bounce (PR #873)
- 8L depth: EP11 bounce +0.222pp (PR #965)
- Dropout=0.1: consistent degradation (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- vol-loss-weight=2.0 WITHOUT GradNorm: actively harmful (PR #936)
- vol-loss-weight=3.0 WITHOUT GradNorm: WORST EVER on old dataset (PR #964)
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- 96k+60k balanced sampling on 5L at default lr=1e-4: catastrophic divergence (PR #938)
- Proportional sampling 96k+60k at 6L: slower convergence, vol_p oscillation (PR #951)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16
- Extended cosine T_max=60 (PR #946): destabilizing in training tail
- DropPath regularization (PR #987): EP5 gate FAIL (likely architecture-level, not dataset-dependent)
- Lookahead Lion (PR #998): EP5 FAIL (optimizer-level)
- Online focal vol reweighting (#1026, #1033): scale=3 ceiling degeneration — EMA ratio approach AXIS CLOSED
- InstanceNorm across vol tokens (PR #1015): closed before terminal — vol-token LN (#1025) is successor

## Removed from Dead Ends (RETRACTED — Dataset Artifact)

The following were previously listed as dead ends but were tested on the broken pre-20260511 dataset split. They are **NOT confirmed dead ends** and some are now SOTA techniques:

- ~~SDF-stratified importance sampling (PR #972)~~: **NOW RANK 1 SOTA** (test_abupt=5.844%)
- ~~Stochastic vol subsampling (PR #968)~~: **NOW RANK 2 SOTA** (test_abupt=5.986%)
- WD=0.01 (PR #900): needs re-test on corrected split before concluding
- WD=0.005 (PR #914): needs re-test on corrected split before concluding
- TTA Y-symmetry (PR #979): may still be neutral; re-test on corrected split before concluding
- Vol coordinate noise (PR #990): EP5 gate FAIL — may be architecture-level; re-test on corrected split
- Bbox normalization (PR #978): may need re-test
- EMA decay=0.999 (PR #954): needs re-test on corrected split

_Last updated: 2026-05-12 ~20:30 UTC. MAJOR UPDATE: Dataset artifact resolved (Issue #1053). Corrected split `rawcanon_20260511` shows no val→test vol_p gap. PR #972 (SDF-stratified sampling) is new SOTA at test_abupt=5.844%. All "AXIS FULLY CLOSED" labels on sampling strategies RETRACTED. PR #740 old SOTA now ranks 22 at 8.165% (artifact). All active experiments must restart on corrected dataset path._
