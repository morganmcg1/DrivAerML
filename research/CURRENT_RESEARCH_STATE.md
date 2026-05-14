# SENPAI Research State
- 2026-05-14 ~05:30 UTC

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

## INFRASTRUCTURE FIX MERGED (2026-05-13 ~21:11 UTC): PR #1087 — EMA warm-start shadow re-init

**Backport from main commit `860d08f` merged to advisor branch.** Bug: `EMA.update()` never re-synced shadow buffers at `step_counter == start_step` — shadow stayed at random-init for entire warmup window. Fix re-initializes shadow from live params at the trigger step.

**Implication:** Past EMA verdicts on this branch (PR #954) are suspect — early-epoch metrics fed kill-thresholds with contaminated shadow. Long runs (>>7400 steps) decay contamination to ~0%, so PR #972 SOTA itself is unaffected. PR #1086 (tanjiro EMA re-test) is the first clean EMA run on this branch.

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

**Top 5 on corrected split:**

| Rank | PR | test_ABUPT | test_vol_p | Notes |
|------|----|------------|------------|-------|
| 1 | #972 | 5.844% | 3.643% | Uniform (monkey-patch no-op) — CLOSED |
| 2 | #968 | 5.986% | 3.957% | Stochastic vol subsampling — CLOSED |
| 3 | #880 | 6.010% | 4.501% | CLOSED |
| 4 | #958 | 6.107% | 3.818% | CLOSED |
| 5 | #939 | 6.242% | — | CLOSED |

## Active Experiments (2026-05-14 ~05:30 UTC)

### Pod Assignments

| Student | PR | Hypothesis | W&B Run | Approx EP | Notes |
|---------|-----|-----------|---------|-----------|-------|
| dl24-tanjiro | #1086 | EMA(0.999) + SDF α=0.25 vs fern A/B | `fby84xtu` | ~EP7.5 | val_abupt=6.274% (wave-best), val_vol_p=4.118%, terminal ~10-12h |
| dl24-nezuko  | #1072 | SDF α=0.5 (inverse formula) | `yp383yq2` | ~EP25.4 | EP25 gate PASSED (6.399%), terminal ~1-2h |
| dl24-fern    | #1098 | WD=0.01 isolated re-test (orthogonal pivot) | TBD (not launched) | smoke pending | **NEW: assigned 2026-05-14 05:30Z, prev #1063 CLOSED** |
| dl24-frieren | #1077 | SDF α=1.0 (inverse formula) | `m4z2gb65` | ~EP13 | continuing |

### SDF Wave Terminal Results

**PR #1063 CLOSED 2026-05-14 ~05:00Z — fern SDF α=0.25 — NOT A WINNER**

| Metric | Fern terminal (xfykblf9) | PR #972 SOTA | Δ |
|--------|------------------------:|-------------:|---:|
| test_abupt | 5.955% | 5.844% | +0.111pp (+1.9% rel) |
| test_vol_p | 3.990% | 3.643% | **+0.347pp (+9.5% rel)** |
| test_surf_p | 3.707% | 3.577% | +0.130pp |
| test_wss | 6.746% | 6.727% | ~noise |

All four test metrics regress vs SOTA. test_vol_p +9.5% is the disqualifying factor per Issue #882 priority. Closed.

### SDF α Sweep Status (BROADLY FALSIFIED)

| α value | PR | Best val_abupt | Test result | Status |
|---------|-----|---------------:|-------------|--------|
| 0.25 | #1063 | 6.2647% (EP11) | test_abupt=5.955% — **CLOSED** | test_vol_p regressed +9.5% |
| 0.5  | #1072 | 6.2904% (EP10) | terminal pending (~06:30Z) | EP25 PASS, converging to ~6.40% |
| 1.0  | #1077 | TBD | running ~EP13 | midpoint arm |
| 2.0  | #1054 | CLOSED | EP15 FAIL | over-concentration |
| 3.0  | #1076 | 6.5012% (EP6) | KILLED EP10 | over-concentration |

**α-sweep conclusion:** Productive band is α ∈ [0.25, 0.5] for val_abupt, but even α=0.25 regresses all test metrics. SDF concentration is NOT the right lever on the corrected split. The SOTA was achieved with uniform sampling (α=∞ effectively) and the corrected dataset; adding near-surface emphasis seems to hurt test vol_p generalization.

### Val Checkpoint Snapshots (latest ~05:20 UTC)

| Student | PR | Run | EP | val_abupt (latest) | val_abupt (best) | val_vol_p (latest) |
|---------|-----|-----|----|-------------------:|-----------------:|-----------------:|
| dl24-fern    | #1098 | TBD | — | — | — | — |
| dl24-nezuko  | #1072 | `yp383yq2` | ~EP25.4 | 6.399% | **6.2904%** (EP10) | 4.753% |
| dl24-tanjiro | #1086 | `fby84xtu` | ~EP7.5 | **6.2742%** | **6.2742%** (EP7.5) | **4.118%** |
| dl24-frieren | #1077 | `m4z2gb65` | ~EP12-13 | ~6.36% | ~6.35% | ~4.3% |

**Tanjiro (#1086) is current wave val leader at 6.2742% with outstanding vol_p of 4.118%.** EMA may be decoupling the plateau-regression pattern — this is the hypothesis to watch for terminal test metrics.

## Strategic Themes & Next Directions

### What the SDF wave taught us
1. **α=0.25 and α=0.5** produce near-identical best val_abupt (~6.27–6.29%) — the SDF concentration parameter has weak discriminating signal in this range.
2. **All SDF val improvements regress at test** — SDF concentration doesn't help vol_p generalization to OOD test geometries.
3. **The SOTA (uniform sampling + corrected split) is the right baseline** — the SDF mechanism is not the key ingredient.

### Orthogonal levers being tested
- **EMA (tanjiro #1086):** First clean EMA run after the warm-start bug fix. val_vol_p=4.118% at EP7.5 is wave-best. Terminal test will tell if EMA decouples val→test transfer. Terminal expected ~10-12h.
- **WD=0.01 (fern #1098, NEW):** Weight decay A/B vs SOTA WD=0.005. Hypothesis: higher WD penalizes geometry-specific memorization, improving OOD test generalization. Clean single-variable change on corrected split. No prior clean test on corrected data.

### Pending hypothesis queue (once slots open)
1. **H1:** GradNorm + SDF α=0.5 composition — if nezuko #1072 terminals with strong vol_p (best so far: val_vol_p=4.231% EP10), compose with GradNorm. Assign to next idle student.
2. **H5:** LR=9e-5 isolated control — lower LR may extend useful learning window; never cleanly tested on corrected split.
3. **Surface-point density / eval-point increase** — if test vol_p gap persists, investigate whether eval_surface_points=131k (double) improves test surface metrics without training change.

## Next Pending Gates (in order)

1. **Nezuko #1072 terminal (~06:30Z):** EP30. Test from EP10 best checkpoint (val_abupt=6.2904%, val_vol_p=4.231%) is the deliverable. Compare test_vol_p vs SOTA 3.643%.
2. **Fern #1098 smoke (~2h after student picks up):** EP3 screen, val_abupt < 8.5%.
3. **Tanjiro #1086 terminal (~10-12h):** EMA(0.999) first clean test — wave-best val_abupt=6.274%, val_vol_p=4.118%. KEY experiment.
4. **Frieren #1077 EP15 gate (~6-8h):** val_abupt ≤ 6.80%. Currently tracking ~6.35%.
