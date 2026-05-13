# SENPAI Research State
- 2026-05-13 ~14:30 UTC

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
`/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`

**Corrected eval parameters:** `eval_surface_points=65536`, `eval_volume_points=65536` (chunk sizes, not caps), 34 val cases / 7,295 views, 50 test cases / 11,091 views.

## CRITICAL DISCOVERY (2026-05-12 ~22:00 UTC): SDF Monkey-Patch Was a No-Op

**PR #972 did NOT implement SDF-stratified sampling. It used uniform sampling.**

Two bugs were discovered in the `types.MethodType` monkey-patch used in PR #972 and copied to PRs #1054 and #1055:

- **Bug 1 (Python dunder no-op)**: `types.MethodType(__getitem__, dataset)` installs in instance dict. Python resolves dunders via type MRO, not instance dict. Fix: use `__class__` reassignment to a dynamically-created subclass.
- **Bug 2 (formula inversion)**: Original formula `weight = 1.0 + α * |sdf|` upweights far-field. Correct: `weight = 1.0 / (1.0 + α * |sdf|)` (upweights near-surface, sdf≈0).

**ALL prior "SDF" experiments used uniform sampling. True SDF near-surface emphasis has never been tested until the DL24 branch (PRs #1054+).**

## Wave SOTA (Corrected Split — rawcanon_20260511)

**PR #972** (run `56bcqp3m`, eval `zxnhtagj`): `abupt_axis_mean_rel_l2_pct` = **5.844%**

| Metric | Value |
|--------|-------|
| test_abupt | **5.844%** |
| test_surf_p | 3.577% |
| test_vol_p | **3.643%** |
| test_wss | 6.727% |

> Note: PR #972 used uniform sampling (monkey-patch was a no-op). This is the baseline to beat.

**GradNorm v4 partial SOTA beat (PR #1058 arm `ysycg6xc`, terminal=false pending_arms=true):**
- val_abupt=6.1855%, test_vol_p=**3.6328%** (BEATS SOTA by 0.010pp), test_abupt=5.9950%
- This arm is NOT terminal; v5b arm still running. Final merge decision pending.

**Top 5 on corrected split:**

| Rank | PR | test_ABUPT | test_vol_p | Notes |
|------|----|------------|------------|-------|
| 1 | #972 | 5.844% | 3.643% | Uniform (monkey-patch no-op) — CLOSED |
| 2 | #968 | 5.986% | 3.957% | Stochastic vol subsampling — CLOSED |
| 3 | #880 | 6.010% | 4.501% | CLOSED |
| 4 | #958 | 6.107% | 3.818% | CLOSED |
| 5 | #939 | 6.242% | — | CLOSED |

## Active Experiments (2026-05-13 ~14:30 UTC)

### Pod Assignments

| Student | PR | Hypothesis | W&B Run | Current Step |
|---------|-----|-----------|---------|-------------|
| tanjiro | #1076 | SDF near-surface α=3.0 (corrected patch + inverse formula) | `ed01yw3z` | ~23,400 |
| nezuko | #1072 | SDF near-surface α=0.5 (corrected patch + inverse formula) | `yp383yq2` | ~72,700 |
| fern | #1063 | SDF near-surface α=0.25 (corrected patch + inverse formula) | `xfykblf9` | ~150,000 |
| frieren | #1058 | GradNorm v5b: α=1.5, gradnorm_mode=ema_proxy | `4uyc5dyl` | ~94,900 |

### Val Checkpoint Histories (abupt_axis_mean_rel_l2_pct)

**Tanjiro — PR #1076 — SDF α=3.0 — run `ed01yw3z`:**

| Gate | Step | Val abupt | Status |
|------|------|-----------|--------|
| EP1 | 10,975 | 24.11% | PASS (≤30%) |
| EP2 | 21,950 | **8.885%** | PASS (≤16%) |
| EP3 | 32,925 | — | **PENDING** (threshold ≤8.0%) |

Note: massive EP1→EP2 drop (24.11% → 8.89%). α=3.0 converging aggressively. EP3 outcome uncertain (0.89pp above threshold).

**Nezuko — PR #1072 — SDF α=0.5 — run `yp383yq2`:**

| Gate | Step | Val abupt | Status |
|------|------|-----------|--------|
| EP1 | 10,975 | 21.54% | PASS (≤30%) |
| EP2 | 21,950 | 7.29% | PASS (≤16%) |
| EP3 | 32,925 | 6.67% | PASS (≤8%) |
| EP5 | 54,875 | 6.43% | PASS (≤7.5%) |
| EP6 | 65,850 | 6.42% | — |
| EP10 | 109,750 | — | **PENDING** (threshold ≤7.2%) |

**Fern — PR #1063 — SDF α=0.25 — run `xfykblf9`:**

| Gate | Step | Val abupt | Status |
|------|------|-----------|--------|
| EP1 | 10,975 | 20.94% | PASS (≤30%) |
| EP2 | 21,950 | 7.48% | PASS (≤16%) |
| EP3 | 32,925 | 6.62% | PASS (≤8%) |
| EP5 | 54,875 | 6.35% | PASS (≤7.5%) |
| EP8 | 87,800 | 6.31% | — |
| EP9 | 98,775 | 6.27% | — |
| EP10 | 109,750 | 6.28% | PASS (≤7.2%) |
| EP11 | 120,725 | 6.26% | — |
| EP12 | 131,700 | 6.27% | — |
| EP13 | 142,675 | 6.31% | — |
| EP15 | 164,625 | — | **PENDING** (threshold ≤6.80%) |

**Frieren — PR #1058 — GradNorm v5b α=1.5 — run `4uyc5dyl`:**

| Gate | Step | Val abupt | Status |
|------|------|-----------|--------|
| EP1 | 10,974 | 19.13% | PASS (≤30%) |
| EP2 | 21,948 | 7.53% | PASS (≤16%) |
| EP3 | 32,922 | 6.77% | PASS (≤8%) |
| EP5 | 54,870 | 6.42% | PASS (≤7.5%) |
| EP7 | 76,818 | 6.26% | — |
| EP8 | 87,792 | 6.23% | — |
| EP10 | 109,750 | — | **PENDING** (threshold ≤7.2%) |

### SDF α Sweep Status

| α value | Student | PR | Status |
|---------|---------|-----|--------|
| 0.25 | fern | #1063 | EP13 PASS (6.31%), EP15 gate PENDING |
| 0.5 | nezuko | #1072 | EP6 PASS (6.42%), EP10 gate PENDING |
| 1.0 | — | #1077 | **UNSTARTED** — frieren pod occupied; assign when free |
| 3.0 | tanjiro | #1076 | EP2 PASS (8.885%), EP3 gate PENDING |

### GradNorm Status

- **v4 (run `ysycg6xc`)**: COMPLETED with SENPAI-RESULT terminal=false, pending_arms=true. val_abupt=6.1855%, test_vol_p=3.6328% (BEATS SOTA by 0.010pp). Waiting for v5b completion to pick best arm.
- **v5b (run `4uyc5dyl`)**: In progress — EP8 abupt=6.233%. EP10 gate PENDING (≤7.2% at step ~109,750). On track.

## Next Pending Gates (in order)

1. **Tanjiro EP3** — step 32,925, threshold ≤8.0%. Current EP2=8.885% (0.89pp above). Uncertain — needs continued convergence.
2. **Nezuko EP10** — step 109,750, threshold ≤7.2%. Current EP6=6.42%. Well on track.
3. **Frieren GradNorm v5b EP10** — step ~109,750, threshold ≤7.2%. Current EP8=6.23%. Well on track.
4. **Fern EP15** — step 164,625, threshold ≤6.80%. Current EP13=6.31%. Well on track.

## Pending Assignments

- **PR #1077 (SDF α=1.0)**: unstarted — assign to frieren pod once GradNorm v5b completes (EP10 gate ~15k steps away from frieren's current position)
- **Post-GradNorm v5b**: Post terminal SENPAI-RESULT for PR #1058 picking best arm (v4 vs v5b), then merge if beats baseline

## Gate Schedule

| Gate | Standard Threshold | Steps (EBS=32) | Steps (EBS=8) |
|------|--------------------|----------------|----------------|
| EP1  | ≤30%               | 2,743          | 10,975         |
| EP2  | ≤16%               | 5,486          | 21,950         |
| EP3  | ≤8%                | 8,229          | 32,925         |
| EP5  | ≤7.5%              | 13,715         | 54,875         |
| EP10 | ≤7.2%              | 27,430         | 109,750        |
| EP15 | ≤6.80%             | 41,145         | 164,625        |
| EP20 | ≤6.70%             | 54,860         | 219,500        |
| EP25 | ≤6.65%             | 68,575         | 274,375        |
| EP30 | ≤6.60%             | 82,290         | 329,250        |

## Critical Config Constraints

1. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
2. **`--train-volume-points 65000` (or higher)** REQUIRED: default 16384 inverts volume:surface gradient ratio.
3. **`--lr-warmup-epochs 1` NOT `--lr-warmup-steps 500`**: epoch-based warmup is correct at 6L 65k.
4. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion optimizer.
5. **`--volume-loss-weight` BUG**: When `--use-gradnorm` is active, `--volume-loss-weight` is a no-op.
6. **`--model-pe string_multisigma` REQUIRED when using STRING PE on DL24**: omitting causes `--pe-init-sigmas` to be silently ignored.
7. **`--pe-init-sigmas` must be COMMA-separated**: `0.25,0.5,1.0,2.0,4.0` NOT space-separated.
8. **Kill threshold operator on DL24**: `<=` is correct (`passes()` returns True when observed<=threshold; kills when not passes).
9. **EMA decay=0.999 (NOT 0.9999)**: 0.9999 gives ~10,000-step lookback (too slow); 0.999 gives ~1,000-step lookback.
10. **Steps/epoch at bs=1 DDP8 (EBS=8)**: ~10,975.
11. **Steps/epoch at bs=4 DDP8 (EBS=32)**: ~2,743.
12. **SDF monkey-patch MUST use `__class__` reassignment**: `types.MethodType(__getitem__, dataset)` is a Python dunder no-op.
13. **SDF near-surface formula**: `weight = 1.0 / (1.0 + α * |sdf|)`. NOT `1.0 + α * |sdf|`.
14. **Branch flag incompatibility**: tay and DL24 branches use DIFFERENT CLI flags. Never mix.
15. **Dataset path**: `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`. NOT `/mnt/pvc/`.

## Confirmed Dead Ends (Hardware/Architecture — Not Dataset-Dependent)

- No weight decay: overfits at EP9 (PR #898)
- 7L depth without full regularization: catastrophic bounce (PR #873)
- 8L depth: EP11 bounce +0.222pp (PR #965)
- Dropout=0.1: consistent degradation (PR #899)
- vol-loss-weight=2.0 WITH GradNorm: self-cancelling (PR #911)
- vol-loss-weight=2.0 WITHOUT GradNorm: actively harmful (PR #936)
- vol-loss-weight=3.0 WITHOUT GradNorm: WORST EVER on old dataset (PR #964)
- 96k vol points without proportional surface increase: gradient starvation (PR #912)
- QK-Norm: consistent failures (multiple PRs)
- Y-sym p=1.0: over-augmentation (PR #866)
- 7-octave STRING PE (PR #843): σ=16.0 destabilization
- 6-octave STRING PE (PR #818): does not beat 5-octave
- GradNorm α=0.25 (multiple PRs): terminal test regression
- GradNorm α=0.75 (PR #874): catastrophic instability at EP16
- Extended cosine T_max=60 (PR #946): destabilizing in training tail
- DropPath regularization (PR #987): EP5 gate FAIL
- Lookahead Lion (PR #998): EP5 FAIL
- Online focal vol reweighting (#1026, #1033): scale=3 ceiling degeneration — EMA ratio approach AXIS CLOSED
- SDF near-surface sampling α=2.0 (nezuko PR #1054): EP15 FAIL (6.9264% > 6.80%) — too aggressive near-surface concentration

## Removed from Dead Ends (RETRACTED — Dataset Artifact)

- ~~SDF-stratified importance sampling (PR #972)~~: **NOW RANK 1 SOTA** (test_abupt=5.844%, but this was actually uniform)
- ~~Stochastic vol subsampling (PR #968)~~: **NOW RANK 2 SOTA** (test_abupt=5.986%)
- WD=0.01 (PR #900): needs re-test on corrected split
- WD=0.005 (PR #914): needs re-test on corrected split
- TTA Y-symmetry (PR #979): may still be neutral; re-test needed
- Vol coordinate noise (PR #990): EP5 gate FAIL — may be architecture-level; re-test needed
- Bbox normalization (PR #978): may need re-test
- EMA decay=0.999 (PR #954): needs re-test on corrected split

_Last updated: 2026-05-13 ~14:30 UTC. 4 active DL24 experiments: tanjiro #1076 α=3.0 EP3 PENDING (8.885% at EP2, 0.89pp above threshold), nezuko #1072 α=0.5 EP10 PENDING (6.42% at EP6), fern #1063 α=0.25 EP15 PENDING (6.31% at EP13), frieren #1058 GradNorm v5b EP10 PENDING (6.23% at EP8). PR #1077 α=1.0 unstarted (frieren pod occupied). GradNorm v4 partial SOTA beat on test_vol_p (3.6328% vs 3.643%)._
