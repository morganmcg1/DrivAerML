# SENPAI Research State

**Updated**: 2026-05-28 ~11:05Z | Branch: `tay` | SOTA: H112 PR #1283 (single-model) / PR #1102 (K=8 ensemble)
**Constraint**: ~24 hours of training compute remain (Issue #1056 human directive, 2026-05-27)

## Primary Objective

**test_WSS < 5.85%** (Transolver-3 SOTA, per Morgan Issue #1056)

| Model | val_abupt | test_WSS | Gap to target | Notes |
|---|---:|---:|---:|---|
| H112 single (PR #1283) | 6.1358% | **6.752%** | −0.90pp | current canonical single-model SOTA |
| PR #1102 K=8 ensemble | 5.7452% | **6.3263%** | −0.48pp | ensemble path — humans flagged "no ensembles" preference |
| H183 single (fern PR #1356, in flight ~11Z) | 6.0388% | 6.8287% | — | **NEW PROGRAM VAL SOTA** but test MISS due to WSS_x sign-flip |

Merge gate: val_abupt < 6.1358% AND test_WSS ≤ 6.727% AND test_VP ≤ 3.421% AND test_SP ≤ 3.577%.

---

## Program-Critical Finding — H164d RNG Floor Calibration (banked 2026-05-28 ~10:30Z)

**H164d (frieren PR #1357, completed)**: H112-recipe rerun with different RNG seed.

| Channel | Δ slope (H164d vs H112) | Implication |
|---|---:|---|
| WSS aggregate | +0.040pp | RNG floor ≈ ±0.040pp single-draw |
| WSS_z | −0.009pp | Most stable — RNG floor ≈ ±0.01pp |
| WSS_x | ~±0.05pp | RNG floor ≈ ±0.05pp |
| WSS_y | ~±0.06pp | RNG floor ≈ ±0.06pp |
| VP | ~±0.09pp | Largest channel-specific RNG floor |

### Framework recalibration (PERMANENT)

- **Slope claims smaller than 3× channel-specific floor are now under-powered with N=1**:
  - WSS_agg < 0.12pp → noise
  - WSS_z < 0.03pp → noise
  - WSS_x < 0.15pp → noise
- **WSS_z slope is the canonical cohort-screening axis** (4–9× better RNG reproducibility than other channels). Single-experiment comparisons reliable on WSS_z.
- **Many prior cohort closures retain status as suggestive but not statistically definitive**. The 4-axis slope-preservation cohort (H145 −0.048pp, H148 −0.081pp, H149 −0.036pp, H157 −0.036pp) has only H148's signal clearly above 2× RNG floor; H145/H149/H157 are within or near floor and may be RNG noise.

### What's still SOLID

- **z-axis tau_z LOCKED at 2.0** (4-point closure H112=2.0, H143=4.0, H144=6.0, H165=1.5 — all 3 perturbations show monotone regression, signal exceeds RNG floor)
- **H112 depth-5 is test-optimum on depth axis** (H120 depth-6, H125 depth-7 both regress test_WSS by 0.08–0.09pp, well above RNG floor)
- **Capacity additions cause test regression** (H118/H120/H121/H125 — slope-flattening exceeds RNG floor)
- **H183 finding REAL**: shared-axis stacking (mirror+tau_y=3.0) triggered WSS_x slope sign-flip from −0.093pp to +0.064pp — a 16bp swing on a channel with ±0.05pp floor (3× floor, real)

---

## EP3 Cross-Axis Cohort Leaderboard (banked 2026-05-28 18:00Z)

| PR | Student | Hypothesis | EP3 val_abupt | Δ vs H112 (6.978%) | WSS_x slope | Status |
|---|---|---|---:|---:|---|---|
| **#1353** | **thorfinn** | **H185: GradNorm × mirror-aug** | **6.8665** | **−0.111pp LEADING** | −0.162pp (intact) | **strongest at EP3** |
| #1362 | alphonse | H188: mirror × DropPath=0.15 | 7.34 | +0.36pp | tba | strong |
| #1365 | tanjiro | H191: mirror × tau_y=2.0 | 7.18 | +0.20pp | tba | moderate |
| #1363 | askeladd | H189: AdamW × tau_y=3.0 no mirror | 8.05 | +1.07pp | −0.160pp (intact) | mechanism confirmed |
| #1356 | fern | H186: mirror × AdamW | 8.28 | +1.30pp | tba | restart cancelled at 19:05Z (let run finish) |
| #1357 | frieren | H187: H164e RNG rerun | 6.94 | −0.04pp | (calibration) | baseline N=2 |

**H185 KEY FINDINGS**:
- LEADING H112 on ALL channels at EP3 (abupt, WSS_agg, WSS_x, WSS_y, WSS_z, SP); VP +5.8bp (expected from H171 vol downweight)
- WSS_x basin invariant strongly negative (−16.2bp ahead of H112)
- Synergistic effect: mirror alone H148 gave +9.5bp at EP3 (noise); H185 compound gives −16.5bp (genuine signal)
- EP4 bonus: still leading −0.063pp on val_abupt
- Lead NARROWING EP3→EP4 (basin convergence); terminal trajectory uncertain
- Terminal ~01:30Z tomorrow

## ~Step 46k Snapshot (banked 19:05Z via W&B — students bot-silent on PRs)

H112 terminal val_abupt gate = 6.1358%. Latest mid-EP6 trajectory:

| PR | Student | Hypothesis | step | val_abupt | Δ vs gate |
|---|---|---|---:|---:|---:|
| #1362 | alphonse | H188: mirror × DropPath=0.15 | 46245 | **6.525%** | +0.39pp |
| #1350 | edward | H184: surface:vol 4:0.5 + tau_y=3.0 | 46001 | 6.579% | +0.44pp |
| #1364 | nezuko | H190: mirror at p=0.25 | 42452 | 6.582% | +0.45pp |
| #1365 | tanjiro | H191: mirror × tau_y=2.0 | 45943 | 6.742% | +0.61pp |
| #1363 | askeladd | H189: AdamW × tau_y=3.0 no mirror | 46824 | 7.090% | +0.95pp |
| #1356 | fern | H186: mirror × AdamW | 45981 | 7.299% | +1.16pp (worst) |

Three students (alphonse, edward, nezuko) clustering tightly at 6.5-6.6% val_abupt with ~20k steps remaining (~25-30% of training). Each plausibly drops 0.4-0.5pp by terminal → breakthrough beyond H112 SOTA. thorfinn H185 was at 6.535 at EP4 (step 38030), likely similar or better trajectory at step 46k.

**Multiple winners possible**. First terminal expected ~22:30Z (frieren H187), main cohort ~01:00-02:30Z tomorrow.

---

## H183 Val SOTA Path Analysis (program-defining)

H183 banked NEW PROGRAM VAL SOTA val_abupt=6.0388% (−97bp below H112 gate), but test_WSS missed at 6.8287% due to WSS_x sign-flip from shared y-axis stacking.

**Diagnosis**: tau_y=3.0 (y-loss-weight escalation) + mirror-augmentation (y-axis data invariance) over-allocated gradient mass on the y-channel, disrupting the basin invariant for WSS_x. WSS_x slope went from −0.093pp (H112, basin intact) to +0.064pp (basin disrupted).

**Current frontier — cross-axis compounds that AVOID shared-axis stacking**:

| PR | Student | Hypothesis | Cross-axis pair | Status |
|---|---|---|---|---|
| #1356 | fern | H186: mirror-aug + AdamW | data-invariance × optimizer | follow-up arm running |
| #1353 | thorfinn | H185: GradNorm static plateau + mirror-aug | dynamic-balance × data-invariance | follow-up arm running |
| #1350 | edward | H184: surface:vol 4:0.5 + tau_y=3.0 | loss-budget × y-axis | follow-up arm running |
| #1357 | frieren | H187: H164e — 2nd H112 RNG rerun | calibration | follow-up arm running |

**4 new compounds assigned 2026-05-28 ~11:05Z**:

| PR | Student | Hypothesis | Cross-axis test |
|---|---|---|---|
| **#1362** | **alphonse** | **H188: mirror-aug + DropPath_max=0.15** | data-invariance × architecture-stochasticity |
| **#1363** | **askeladd** | **H189: AdamW + tau_y=3.0 (no mirror)** | optimizer × y-loss-weight (H183 ablation) |
| **#1364** | **nezuko** | **H190: mirror-aug at p=0.25** | data-invariance strength sweep |
| **#1365** | **tanjiro** | **H191: mirror-aug + tau_y=2.0** | data-invariance × moderated y-axis (H183 boundary probe) |

---

## Active Fleet (2026-05-28 ~11:05Z)

| Student | PR | Hypothesis | Notes |
|---|---|---|---|
| frieren | #1357 | H187 (H164e RNG variance check) | follow-up arm; N=2 draws give variance estimate |
| fern | #1356 | H186 (mirror + AdamW) | cross-axis compound, follows H183 NEW VAL SOTA |
| thorfinn | #1353 | H185 (GradNorm + mirror) | cross-axis compound, follows H171 val PASS |
| edward | #1350 | H184 (surface:vol 4:0.5 + tau_y=3.0) | cross-axis compound, follows H170 val PASS, Interpretation B confirmed |
| alphonse | #1362 | H188 (mirror + DropPath=0.15) | NEW assignment, cross-axis reg compound |
| askeladd | #1363 | H189 (AdamW + tau_y=3.0, NO mirror) | NEW assignment, H183 cross-axis ablation |
| nezuko | #1364 | H190 (mirror at p=0.25) | NEW assignment, augmentation strength sweep |
| tanjiro | #1365 | H191 (mirror + tau_y=2.0) | NEW assignment, H183 boundary probe |

Zero idle students.

---

## Recent Closures (2026-05-28 ~10:30Z batch)

- #1349 H166 (alphonse) — tau_y=1.0 de-escalation: closed
- #1351 H180 (tanjiro) — Lookahead(AdamW): closed
- #1354 H181b (askeladd) — H148+EMA=0.9999: closed C NULL (EMA=0.9999 universally worse than 0.999; EMA axis exhausted at this base)
- #1355 H167 (nezuko) — tau_y=4.0 escalation: closed

---

## Diagnostic Invariants (DO NOT VIOLATE)

- No capacity additions — any param overhead ≥+1% risks slope flattening
- No ensembles (Morgan directive: lazy route)
- Cross-run comparison: W&B raw `val_primary/*` at same step (NOT EMA, NOT student tables)
- Kill thresholds: global_step 10864/32592/48897 (standard; EMA=0.9999 requires EP1 gate dropped)
- `<threshold` in kill-threshold string is PASS condition; never write `>threshold`
- WSS_x slope sign is the BASIN-DISRUPTION DIAGNOSTIC — positive slope indicates shared-axis stacking failure
- DrivAerML axes: x=streamwise, y=lateral (mirror axis), z=vertical
- RNG floor on WSS_agg slope = ±0.040pp; treat as the minimum effect size for single-draw mechanism claims
- WSS_z slope is the canonical cohort-screening axis (4–9× better RNG reproducibility)

---

## Banked Hypotheses (not assigned) — prioritized for next-wave after H188–H191 land

- **SAM (Sharpness-Aware Minimization)** [BOLD]: optimizer wrapper around AdamW/Lion; ascends to worst-case loss neighborhood then descends. Known to improve test generalization markedly (flatter minima). No capacity addition; 2× forward/backward cost. **Top candidate for next-wave bold swing per Morgan's PUSH HARD directive.**
- **TTA (test-time augmentation)** [CHEAP]: eval-time mirror-aug forward + average; pure inference code change, no retrain. Potentially applies to ANY existing trained model (including H112) — could yield free test_WSS reduction with no GPU cost.
- **Cross-axis 3-compounds (after H188–H191 results)**: e.g., mirror + AdamW + DropPath=0.15 if H188 + H186 both succeed
- **EMA-best checkpoint selection**: per H181b suggested follow-up — multi-checkpoint EMA tracking + select best EP9–EP13 snapshot
- **Lookahead + mirror-aug**: H180 Lookahead closed; cross-axis with mirror untested
- **Schedule-Free AdamW**: optimizer code change; may avoid LR-schedule confounds
- **SWA (Stochastic Weight Averaging)** [SINGLE-MODEL]: average weights of last K epochs into single model. Distinct from ensembles (one set of weights). Could compound with H112 recipe.

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-27)**: ~2 days until experiment window closes (now ~24h remaining). test_WSS < 5.85% is THE objective. val_abupt is steering metric only.
- **Morgan (Issue #1056, 2026-05-28 15:27Z)**: Ensembles RE-CONFIRMED BANNED. "PUSH HARD" with <2 days remaining. Looking for genuine breakthroughs, not shortcuts.
- No ensembles (Morgan directive — lazy route, want genuine breakthroughs)
- All advisor work on `tay` branch; DDP 8 GPUs every run
- Last advisor response to Morgan: 2026-05-28 15:35Z (full tay-track status: 8/8 fleet active, H183 NEW VAL SOTA finding, cross-axis ablation strategy, key RNG calibration)
