# SENPAI Research State

**Updated**: 2026-05-27 ~05:25Z | Branch: `tay` | SOTA: H112 PR #1283

## PROGRAM-CRITICAL OPEN QUESTION (banked 04:58Z, H164 SWA terminal)

**Is the slope-cohort framework REAL or RNG-LEVEL NOISE?** H164 surfaced the hypothesis that slope-flattening/preservation may be an artifact of any trajectory deviation from H112's exact RNG path, NOT a mechanism signal. The 4-axis slope-preservation cohort, z-axis flattening sub-cohort, and WSS_x reversal signature are all PROVISIONAL pending H164d (frieren PR #1357 — H112-recipe rerun for RNG noise floor measurement).

H164d outcome reframes every cohort closure decision:
- A: slope WITHIN ±0.02pp of H112 → cohort framework REAL, continue compounding (H183 etc.)
- B: slope DEVIATES by ±0.05-0.15pp → cohort framework COLLAPSES, refocus on absolute test metrics only

---

## Primary Objective

**test_WSS < 5.85%** (Transolver-3 SOTA, per Morgan Issue #1056)

Current SOTA (H112, PR #1283, W&B `u9ue2ryb`): val_abupt 6.1358% / test_WSS 6.752% / test_WSS_z 8.720%
Gap to target: test_WSS needs −0.90pp improvement from H112.

Merge gate: val_abupt < 6.1358%, test_WSS ≤ 6.727%, test_VP ≤ 3.421%, test_SP ≤ 3.577%

---

## Program-Defining Framework (locked 2026-05-27 01:05Z)

### Slope-Preservation Cohort — 4 independent axes confirmed

| Member | Mechanism axis | Δ slope WSS agg | Δ slope WSS_z | Status |
|---|---|---:|---:|---|
| H145 (tau_y=3.0) | y-axis loss-weight ESCALATE | −0.048pp STEEPER | −0.026pp STEEPER | CLOSED C NULL |
| H148 (mirror-aug) | y-axis data invariance | −0.081pp STEEPER | −0.013pp STEEPER | CLOSED C NULL |
| H149 (AdamW) | optimizer-axis | −0.036pp STEEPER | −0.060pp STEEPER | CLOSED C NULL |
| **H157 (SGDR T_0=4)** | **scheduler-axis** | **−0.036pp STEEPER** | **−0.075pp STEEPER** | **CLOSED C NULL (01:05Z)** |

Framework locked: slope-steepening ~−0.04 to −0.08pp WSS aggregate, axis-orthogonal, channel-uniform, reproducible. **None close val gap alone.** Compounding required for SOTA.

### Wave 38+39 Refined Closure

**OLD framing**: "Any perturbation off H112 causes slope-flattening."
**REVISED (permanent)**: "Any perturbation in a CAPACITY-INCREASING or Z-AXIS-PERTURBING direction causes slope-flattening. Orthogonal-axis perturbations (y-loss, data-aug, optimizer, scheduler) PRESERVE slope."

**"lr_min freeze" hypothesis FALSIFIED** (banked 2026-05-27 01:05Z): H112 val improves through EP13 under near-zero cosine LR. Late-train flatness = basin-floor reached, NOT LR-induced freeze. Do not propose scheduler variants that interrupt late-train basin convergence.

### Slope-FLATTENING class (closed permanently)

| Class | Experiments | Mechanism |
|---|---|---|
| z-axis loss-weight ESCALATE | H143 (tau_z=4.0), H144 (tau_z=6.0) | Monotone test regression; slope flattening on both aggregate and WSS_z |
| **z-axis loss-weight DE-escalate** | **H165 (tau_z=1.5)** | **slope-FLATTER bilaterally vs H112; cohort #4 of 4 on test_WSS_z (anti-aligned)** |
| Dynamic loss balancing | H147 (GradNorm) | tau_z ceiling [1.65,1.70] BELOW H112's 2.0; dynamic = static pathology |
| Architectural split | H138 (split-z), H146 (split-y) | +525K overhead drives slope flattening regardless of channel |
| Capacity addition | H118/H120/H121/H125 | Val-overfit slope catastrophe; H112 depth-5 is test optimum |

### z-axis 4-point closure LOCKED (2026-05-27 ~04:25Z, H165 terminal)

| Hypothesis | tau_z | val_abupt | test_WSS | test_WSS_z | val→test WSS slope |
|---|---:|---:|---:|---:|---:|
| H112 SOTA | 2.0 | 6.136 | **6.752** | **8.720** | **−0.215** |
| H143 (closed) | 4.0 | 6.200 | 6.955 | 8.895 | −0.082 |
| H165 (closed) | 1.5 | 6.187 | 6.980 | 9.055 | −0.046 |
| H144 (closed) | 6.0 | 6.199 | 7.079 | 8.984 | −0.013 |

H112 is test-optimum on BOTH aggregate AND target channel. ALL 3 perturbed arms show FLATTER slope + WORSE absolute test_WSS. **z-axis tau_z is permanently LOCKED at 2.0 ± 0.0.**

### NEW cohort signature emerging — WSS_x val→test REVERSAL (validation pending)

Every z-axis perturbation (H143, H144, H165) shows positive WSS_x slope (test_WSS_x > val_WSS_x); H112 alone has negative slope (−0.093). MAY be a bowl-floor-proximity indicator. Validation pending on slope-preservation cohort (H145/H148/H149/H157 terminal slopes).

---

## Active Fleet (2026-05-27 ~05:25Z)

| Student | PR | Hypothesis | Status | ETA |
|---|---|---|---|---|
| **frieren** | **#1357** | **H164d H112-rerun RNG-baseline (PROGRAM-CRITICAL slope-cohort sanity check)** | **Freshly assigned** | **terminal ~19:25Z+1** |
| fern | #1356 | H183 mirror-aug + tau_y=3.0 compound (slope-preservation x slope-preservation factorial) | Freshly assigned | terminal ~22:00Z |
| alphonse | #1349 | H166 tau_y=1.0 DE-escalate | EP7-range step 60,204 val_abupt 6.328% (W&B-direct read) | terminal ~06:25Z |
| edward | #1350 | H170 surface:vol 8:1 rebalance | EP9-range step 51,031 val_abupt 6.310% (W&B-direct read) | terminal ~06:40Z |
| thorfinn | #1353 | H171 plateau-exact static (vol=0.5, tau_y=1.30, tau_z=1.67) | v2 r6zoibfi at EP2.5 step 27,221 val_abupt 7.97% | terminal ~14:30Z |
| askeladd | #1354 | H181b H148+ema=0.9999 (corrected gates) | EP1 val 67.97% PASS (DROPPED gate); EP3 first real gate ~05:50Z | terminal ~12:55Z |
| tanjiro | #1351 | H180 Lookahead(AdamW k=5 alpha=0.5) | EP3.5 step 38030 val 8.01% (catchup decay 1.34×); EP6 binding ~05:07Z | terminal ~13:00Z |
| nezuko | #1355 | H167 tau_y=4.0 ESCALATE (y-axis extension) | EP1 val 26.83% PASS | terminal ~15:15Z |

**Closed at 04:25Z**: PR #1348 H165 fern (tau_z=1.5 DE-escalate) C NULL slope-FLATTER bilaterally — locks 4-point z-axis cohort.
**Closed at 04:58Z**: PR #1347 H164 frieren (SWA averaging) C NULL — basin-geometry hypothesis FALSIFIED, slope-as-trajectory-drift hypothesis PROGRAM-CRITICAL.

---

## Priority Hypotheses in Flight

### HIGHEST EV — path to first single-model SOTA
**H181b (askeladd PR #1354)**: H148 + ema=0.9999. EMA longer-averaging window closes val noise floor (~83bp) while preserving gradient-reallocation mechanism (upstream of EMA wrapper). Projected: val_abupt <= H112, slope preserved → test_WSS <= 6.671% = first SOTA via slope-preservation path. EMA-composition-aware gates: EP1 DROPPED, EP3 <25%, EP6 <11%, EP9 <8%.

### HIGH EV — mechanism axis extension
**H167 (nezuko PR #1355)**: tau_y=4.0 ESCALATE. Extends H145 (tau_y=3.0, slope-preservation cohort #1). Key questions: does val_abupt improve vs H145 (6.388%), and does slope stay steeper than H112? If yes, higher tau_y is a better operating point and compounds with H181b.

**H171 (thorfinn PR #1353)**: plateau-exact static (vol=0.5, tau_y=1.30, tau_z=1.67). Tests values-vs-dynamics: if slope-FLATTENS like H147 GradNorm, z-axis perturbation-itself-is-issue framework LOCKED permanently. If slope-PRESERVES or A WIN, GradNorm's auto-discovered values are the new static optimum.

**H165 (fern PR #1348)**: tau_z=1.5 DE-escalation. Direct test of H147 GradNorm-discovered tau_z optimum (1.631) below H112's 2.0. EP9 BINDING: 3 consecutive monotone narrowing epochs, gradient-pressure asymmetry finding. EP10 vol-bump diagnostic is critical (~01:15Z+1).

### MECHANISM DIAGNOSTIC
**H164 (frieren PR #1347)**: SWA averaging-axis. EP10 is first SWA activation epoch (internal epoch=9, display "Epoch 10 end", ~01:45-01:50Z). Binding event: swa/active=1 + n_averaged>=1.

**H166 (alphonse PR #1349)**: tau_y=1.0 DE-escalation. Bilateral y-axis cohort with H145 (3.0 escalate) + H166 (1.0 de-escalate).

**H170 (edward PR #1350)**: surface:vol 8:1 rebalancing. First surface:volume axis test, informed by H147 GradNorm auto-discovery.

**H180 (tanjiro PR #1351)**: Lookahead(AdamW). Tests slow-weight EMA around AdamW recovering Lion's val convergence while preserving slope steepening.

---

## Banked Wave 40+ Hypotheses (not yet assigned)

- H172: param-budget-neutral split decoder (avoids +525K overhead pathology)
- H175: EMA-decay sweep (0.9995, 0.9998, 0.9999)
- H177: backbone width 320 (param-neutral at depth-5)
- H179: single late-restart cosine T_0=10 (SGDR revisit with T_0>=75% budget)
- H183: H148 x H145 y-axis stacking factorial
- H184: H148 x DropPath rate escalation (slope + dropout compound)
- H185: asymmetric mirror p=0.25 (softer version of H148)

---

## Diagnostic Invariants (DO NOT VIOLATE)

- No capacity additions — any param overhead >=+1% risks slope flattening
- No ensembles
- Cross-run comparison: W&B raw val_primary/* at same step (NOT EMA, NOT student tables)
- Kill thresholds: global_step 10864/32592/48897 (standard; EMA=0.9999 requires EP1 gate dropped)
- `<threshold` in kill-threshold string is PASS condition; never write `>threshold`
- SWA / EMA phase-activation: verify 0-indexed internal epoch vs display 1-indexed before alerting
- DrivAerML axes: x=streamwise, y=lateral (mirror axis), z=vertical

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-24)**: "test_WSS < 5.85% is THE objective. val_abupt is steering metric only."
- No ensembles — lazy route, want genuine breakthroughs
- All advisor work on `tay` branch; DDP 8 GPUs every run
