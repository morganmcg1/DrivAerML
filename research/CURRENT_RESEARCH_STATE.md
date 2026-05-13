# SENPAI Research State
- 2026-05-13 ~20:25 UTC

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

**GradNorm v4 (PR #1058, tay branch) — CLOSED 2026-05-13:** test_abupt=5.9950% regresses aggregate vs SOTA 5.844%. Not a merge candidate.

**Top 5 on corrected split:**

| Rank | PR | test_ABUPT | test_vol_p | Notes |
|------|----|------------|------------|-------|
| 1 | #972 | 5.844% | 3.643% | Uniform (monkey-patch no-op) — CLOSED |
| 2 | #968 | 5.986% | 3.957% | Stochastic vol subsampling — CLOSED |
| 3 | #880 | 6.010% | 4.501% | CLOSED |
| 4 | #958 | 6.107% | 3.818% | CLOSED |
| 5 | #939 | 6.242% | — | CLOSED |

## Active Experiments (2026-05-13 ~20:25 UTC)

### Pod Assignments

| Student | PR | Hypothesis | W&B Run | Approx EP | Notes |
|---------|-----|-----------|---------|-----------|-------|
| tanjiro | #1086 | EMA(0.999) + SDF α=0.25 vs fern A/B | NEW | Smoke pending | Just assigned — direct A/B vs fern #1063 |
| nezuko  | #1072 | SDF α=0.5 (inverse formula) | `yp383yq2` | ~EP15 | EP15 gate PASSED (best 6.290%) |
| fern    | #1063 | SDF α=0.25 (inverse formula) | `xfykblf9` | ~EP22 | EP20 gate PASSED (best 6.265%) |
| frieren | #1077 | SDF α=1.0 (inverse formula) | `m4z2gb65` | ~EP1-2 | EP1 PASSED (22.42%), EP2 running |

**Tanjiro PR #1076 (α=3.0) CLOSED 2026-05-13 ~20:20Z:** Killed at EP10 gate (val_abupt=7.25% > ≤7.2%). Best val_abupt=6.5012% at EP6 then drifted upward — over-concentration confirmed. α=3.0 FALSIFIED.

### Val Checkpoint Snapshots (latest, 2026-05-13 ~19:30 UTC)

| Student | PR / α | Run | Step / EP | val_abupt latest | val_abupt best | val_vol_p | val_surf_p | val_wss |
|---------|--------|-----|-----------|-----------------:|---------------:|----------:|-----------:|--------:|
| fern    | #1063 / 0.25 | `xfykblf9` | 230,644 (~EP21) | 6.3239% | **6.2647%** | 4.4906% | 4.0605% | 6.9574% |
| nezuko  | #1072 / 0.5  | `yp383yq2` | 160,960 (~EP14.7) | 6.3020% | **6.2904%** | 4.2884% | 4.1072% | 6.9682% |
| tanjiro | #1076 / 3.0  | `ed01yw3z` | 104,100 (~EP9.5) | 6.7509% | 6.5012% | 4.3635% | 4.2430% | 7.5196% |
| frieren | #1077 / 1.0  | `m4z2gb65` | 8,508 (~EP0.8) | — | — | — | — | — |

Notes:
- **fern (α=0.25):** Now PAST EP20 gate window. EP20 gate ≤6.70% PASSED comfortably (best=6.2647% at EP11). Plateau persists since EP11 in the 6.26–6.32% band. Will likely terminate at EP30 or kill on plateau — needs final test harvest.
- **nezuko (α=0.5):** Best improved from 6.341% (EP7) → **6.2904%** — best now indistinguishable from fern. EP15 gate ≤6.80% imminent (~step 164,625) — already PASSED on best. The α-sweep midpoint is now competitive with α=0.25.
- **tanjiro (α=3.0):** EP10 gate ≤7.2% imminent (~step 109,750). Currently at 6.50% best — well within gate; aggressive α concentration may still pay off late.
- **frieren (α=1.0):** Just launched. EP1 gate ≤30% due ~step 10,975 (~2k steps away).

### SDF α Sweep Status (2026-05-13 ~20:25 UTC)

| α value | Student | PR | Best val_abupt | Latest val_vol_p | Status |
|---------|---------|-----|---------------:|-----------------:|--------|
| 0.25 | fern   | #1063 | **6.2647%** | 4.491% | EP22 running, plateau since EP11 |
| 0.5  | nezuko | #1072 | **6.2904%** | 4.288% | EP15 gate PASSED, continuing |
| 1.0  | frieren | #1077 | ~22% (EP1) | — | EP1 PASS (22.42%), EP2 running |
| 2.0  | — | #1054 | CLOSED | — | EP15 FAIL (6.9264% > 6.80%) |
| 3.0  | tanjiro | #1076 | **6.5012%** | 4.364% | **CLOSED (EP10 KILL — over-concentration)** |

### α-sweep Conclusion (updated)

The α-response is now mapped from both ends:
- **Low α (0.25, 0.5):** Both competitive at val_abupt ≈6.26–6.29%. val_vol_p at 4.3–4.5%. Near-uniform sampling performs well.
- **α=1.0 (frieren):** Data arriving in 2-3h. Critical missing point.
- **α=2.0, 3.0:** Over-concentration confirmed — worse abupt, no test harvest possible.

**Productive band is definitively α ∈ [0.25, 0.5].** Lower values (α<0.25) remain untested. The new EMA experiment (tanjiro #1086) is an A/B vs fern at α=0.25, adding EMA as the next orthogonal lever.

### GradNorm Status (resolved)

- **PR #1058 (run `ysycg6xc`) CLOSED 2026-05-13 ~18:23Z** (was on `tay` advisor branch, not this wave). Result was test_abupt=5.9950% vs SOTA 5.844% — REGRESSES aggregate by +0.151pp despite tying vol_p (test_vol_p=3.6328% vs 3.643%, ≈noise). Not a merge candidate on the aggregate metric.
- **GradNorm + corrected dataset on THIS wave: still untested.** A clean DDP8 long-run with `--use-gradnorm --gradnorm-mode ema_proxy --gradnorm-alpha 0.5` on the productive single-model stack remains a high-value hypothesis for the next round (see RESEARCH_IDEAS).

## Next Pending Gates (in order)

1. **Frieren EP1** — ~step 10,975, threshold ≤30%. Currently step 8,508 — imminent (~2k steps).
2. **Tanjiro EP10** — ~step 109,750, threshold ≤7.2%. Currently best 6.50% — PASSES on best.
3. **Nezuko EP15** — ~step 164,625, threshold ≤6.80%. Currently best 6.29% — already PASSED on best.
4. **Fern EP25** — ~step 274,375, threshold ≤6.65%. Currently best 6.2647% (EP11) — PASSES on best; recent epochs flat, but plateau is below the gate.

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

_Last updated: 2026-05-13 ~20:25 UTC. PR #1076 (tanjiro α=3.0) closed — EP10 KILL, over-concentration confirmed. Tanjiro reassigned to PR #1086 (EMA 0.999 + SDF α=0.25, A/B vs fern). 4/4 students WIP: tanjiro #1086 new-smoke, nezuko #1072 EP15 PASSED, fern #1063 EP22 plateau, frieren #1077 EP1-2. α-sweep conclusion: productive band α∈[0.25, 0.5], α≥2.0 over-concentration. Next levers: EMA (tanjiro), frieren α=1.0 data point, GradNorm+SDF composition for next idle student._
