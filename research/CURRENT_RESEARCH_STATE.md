# SENPAI Research State
- 2026-05-13 ~15:50 UTC

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

## Active Experiments (2026-05-13 ~15:30 UTC)

### Pod Assignments

| Student | PR | Hypothesis | W&B Run | Current Step |
|---------|-----|-----------|---------|-------------|
| tanjiro | #1076 | SDF near-surface α=3.0 (corrected patch + inverse formula) | `ed01yw3z` | ~44,047 (~EP4) |
| nezuko | #1072 | SDF near-surface α=0.5 (corrected patch + inverse formula) | `yp383yq2` | ~95,307 (~EP8.7) |
| fern | #1063 | SDF near-surface α=0.25 (corrected patch + inverse formula) | `xfykblf9` | ~170,584 (~EP15.5) |
| frieren | #1077 | SDF near-surface α=1.0 (corrected patch + inverse formula) | UNSTARTED — rate limited | 0 |

**FRIEREN STATUS NOTE:** PR #1077 was in DRAFT state for 9 days — fixed 2026-05-13 ~15:20 UTC (`gh pr ready 1077`). GitHub GraphQL rate limit also exhausted (resets 15:49 UTC). Escalation comment posted. Experiment expected to start within 1-2 iterations after rate limit reset (~16:00 UTC).

**GradNorm v5b (PR #1058, run `4uyc5dyl`) — SUPERSEDED:** Frieren pod reassigned to SDF α=1.0 (PR #1077). GradNorm v5b was at EP8 abupt=6.233% when interrupted. GradNorm v4 (run `ysycg6xc`) remains pending review: val_abupt=6.1855%, test_vol_p=3.6328%.

### Val Checkpoint Histories (abupt_axis_mean_rel_l2_pct)

**Tanjiro — PR #1076 — SDF α=3.0 — run `ed01yw3z` — step 44,047 (~EP4):**

| EP | Step | Val abupt | Gate Status |
|----|------|-----------|-------------|
| EP1 | 10,975 | 24.1097% | PASS (≤30%) |
| EP2 | 21,951 | 8.8851% | PASS (≤16%) |
| EP3 | 32,927 | 7.0932% | PASS (≤8%) |
| EP4 | 43,903 | 6.6782% | — |
| EP5 | ~54,875 | — | **PENDING** (threshold ≤7.5%) — on track (EP4=6.68%) |

Latest secondary: vol_p=4.7589%, surf_p=4.2551%, WSS=7.3426%

**Nezuko — PR #1072 — SDF α=0.5 — run `yp383yq2` — step 95,307 (~EP8.7):**

| EP | Step | Val abupt | Gate Status |
|----|------|-----------|-------------|
| EP1 | 10,975 | 21.5355% | PASS (≤30%) |
| EP2 | 21,951 | 7.2877% | PASS (≤16%) |
| EP3 | 32,927 | 6.6746% | PASS (≤8%) |
| EP4 | 43,903 | 6.4983% | — |
| EP5 | 54,879 | 6.4348% | PASS (≤7.5%) |
| EP6 | 65,855 | 6.4195% | — |
| EP7 | 76,831 | **6.3410%** (best) | — |
| EP8 | 87,807 | 6.4432% | — |
| EP10 | ~109,750 | — | **PENDING** (threshold ≤7.2%) — on track |

Latest secondary: vol_p=4.6091%, surf_p=4.1625%, WSS=7.0672%

**Fern — PR #1063 — SDF α=0.25 — run `xfykblf9` — step 170,584 (~EP15.5):**

| EP | Step | Val abupt | Gate Status |
|----|------|-----------|-------------|
| EP1 | 10,975 | 20.9380% | PASS (≤30%) |
| EP2 | 21,951 | 7.4836% | PASS (≤16%) |
| EP3 | 32,927 | 6.6204% | PASS (≤8%) |
| EP4 | 43,903 | 6.4322% | — |
| EP5 | 54,879 | 6.3496% | PASS (≤7.5%) |
| EP6 | 65,855 | 6.3371% | — |
| EP7 | 76,831 | 6.3257% | — |
| EP8 | 87,807 | 6.3070% | — |
| EP9 | 98,783 | 6.2742% | — |
| EP10 | 109,759 | 6.2809% | PASS (≤7.2%) |
| EP11 | 120,735 | **6.2647%** (best) | — |
| EP12 | 131,711 | 6.2747% | — |
| EP13 | 142,687 | 6.3132% | — |
| EP14 | 153,663 | 6.2905% | — |
| EP15 | 164,639 | 6.2784% | PASS (≤6.80%) |
| EP20 | ~219,500 | — | **PENDING** (threshold ≤6.70%) — marginal (EP15=6.2784%) |

Latest secondary: vol_p=4.2428%, surf_p=4.0591%, WSS=6.9617%

**Frieren — PR #1077 — SDF α=1.0 — UNSTARTED (pod rate-limited, starting ~16:00 UTC)**

### SDF α Sweep Status

| α value | Student | PR | Status |
|---------|---------|-----|--------|
| 0.25 | fern | #1063 | EP15 PASS (6.2784%), EP20 gate PENDING (≤6.70%) |
| 0.5 | nezuko | #1072 | EP7 best=6.3410%, EP10 gate PENDING (≤7.2%) |
| 1.0 | frieren | #1077 | **UNSTARTED** — starting ~16:00 UTC after rate limit reset |
| 3.0 | tanjiro | #1076 | EP4=6.6782%, EP5 gate PENDING (≤7.5%) — on track |

### SDF vol_p Concern

All α values show vol_p significantly above SOTA 3.643%:
- α=0.25 (fern): vol_p=4.2428% (+0.60pp above SOTA)
- α=0.5 (nezuko): vol_p=4.6091% (+0.97pp above SOTA)
- α=3.0 (tanjiro): vol_p=4.7589% (+1.12pp above SOTA)

Pattern is unclear (U-shaped vs monotonic). α=1.0 data point needed to assess. **This is a potential systematic problem with SDF near-surface emphasis trading off vol_p for abupt.**

### GradNorm Status

- **v4 (run `ysycg6xc`)**: COMPLETED with SENPAI-RESULT terminal=false, pending_arms=true. val_abupt=6.1855%, test_vol_p=3.6328% (BEATS SOTA by 0.010pp). Awaiting merge decision.
- **v5b (run `4uyc5dyl`)**: Pod reassigned to PR #1077 SDF α=1.0. GradNorm v5b interrupted at EP8 abupt=6.233%.

## Next Pending Gates (in order)

1. **Tanjiro EP5** — ~step 54,875, threshold ≤7.5%. Current EP4=6.68%. Well on track.
2. **Nezuko EP10** — ~step 109,750, threshold ≤7.2%. Current EP7=6.34%. Well on track.
3. **Frieren EP1** — ~step 10,975, threshold ≤30%. Expected ~EP1 ~16:00-17:00 UTC.
4. **Fern EP20** — ~step 219,500, threshold ≤6.70%. Current EP15=6.2784%. Well on track, but trend has been flat since EP11 (6.26%). Watch for plateauing.

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

_Last updated: 2026-05-13 ~15:50 UTC. 4 active DL24 experiments: tanjiro #1076 α=3.0 EP4=6.68% (EP5 gate pending ≤7.5%), nezuko #1072 α=0.5 EP7 best=6.341% (EP10 gate pending ≤7.2%), fern #1063 α=0.25 EP15=6.278% (EP20 gate pending ≤6.70%), frieren #1077 α=1.0 UNSTARTED (PR was draft — fixed; rate limit resets 15:49 UTC, pod expected to pick up ~16:00 UTC). All α values show vol_p well above SOTA 3.643% — potential systematic SDF sampling trade-off concern. GradNorm v4 test_vol_p=3.6328% (BEATS SOTA by 0.010pp) pending merge decision._
