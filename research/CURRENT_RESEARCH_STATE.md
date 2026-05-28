# SENPAI Research State

**Updated**: 2026-05-28 23:55Z | Branch: `tay` | SOTA: H112 PR #1283 (single-model) / PR #1102 (K=8 ensemble)
**Constraint**: ~12-15 hours of training compute remain (Issue #1056 human directive, 2026-05-27)
**Active fleet snapshot**: 8/8 students training healthy; first terminal frieren H164e ~01:08Z+1; H185 thorfinn merge-eligible candidate terminal ~03:00-04:00Z+1.

## ~23:55Z Active Fleet Health (W&B-verified)

| PR | Student | Hypothesis | step (mid-EP9/10) | val_abupt | Notes |
|---|---|---|---:|---:|---|
| #1353 | thorfinn | H185 GradNorm × mirror-aug | EP9 banked (step 59780) | **6.0948 LEAD −0.108pp** | TERMINAL MERGE CANDIDATE (~03:00-04:00Z+1) |
| #1357 | frieren | H164e RNG calibration | step ~65875 mid-EP12 | 6.05% trajectory | terminal ~01:08Z+1; test eval ~02:35Z+1 |
| #1364 | nezuko | H190 mirror p=0.25 | EP9 (step 59833) | **6.1386 LEAD −0.011pp (within noise)** | terminal ~03:00Z+1 (50/50 within RNG floor) |
| #1365 | tanjiro | H191 mirror × tau_y=2.0 | step 59780 mid-EP10 | 6.3982 +0.26pp behind | terminal ~04:30Z+1 (NULL-expected boundary probe) |
| #1362 | alphonse | H188 mirror × DropPath=0.15 | step 64690 mid-EP9 | 6.20% / WSS_agg 7.05% | HEALTHY heartbeat live; in band with H112 EP9 |
| #1363 | askeladd | H189 AdamW × tau_y=3.0 no-mirror | step 65221 EP9-boundary | 6.50% / WSS_agg 7.36% | HEALTHY |
| #1356 | fern | H186 mirror × AdamW (kill+restart cancelled 19:05Z) | step 64504 mid-EP9 | EP9 boundary pending | HEALTHY |
| #1350 | edward | H184 surface:vol 4:0.5 + tau_y=3.0 | step 64928 EP9-boundary | 6.26% / WSS_agg 7.03% | HEALTHY |

**Stale-WIP cluster analysis**: alphonse / edward / askeladd / fern all heartbeat live; comment-lag is the explanation, not pod stall. Pattern: student bots only poll PR comments at major checkpoint boundaries (EP3/EP6/EP9), not continuously.

## Primary Objective

**test_WSS < 5.85%** (Transolver-3 SOTA, per Morgan Issue #1056)

| Model | val_abupt | test_WSS | Gap to target | Notes |
|---|---:|---:|---:|---|
| H112 single (PR #1283) | 6.1358% | **6.752%** | −0.90pp | current canonical single-model SOTA |
| PR #1102 K=8 ensemble | 5.7452% | **6.3263%** | −0.48pp | ensemble path — humans flagged "no ensembles" preference |
| H183 single (fern PR #1356, in flight ~11Z) | 6.0388% | 6.8287% | — | **NEW PROGRAM VAL SOTA** but test MISS due to WSS_x sign-flip |

Merge gate: val_abupt < 6.1358% AND test_WSS ≤ 6.727% AND test_VP ≤ 3.421% AND test_SP ≤ 3.577%.

---

## Program-Critical Finding — RNG Floor Calibration MATURE (frieren H164d+H164e N=2 EP11)

**Banked 23:47Z** via frieren EP9/10/11 catch-up report. N=2 H112-recipe brackets (H164d + H164e) at EP11:

| Channel | EP11 half-range | RNG floor implication |
|---|---:|---|
| **abupt** | **±0.053pp** | minimum effect size for val signal |
| WSS aggregate | ±0.063pp | |
| WSS_x | ±0.059pp | basin-diagnostic still readable above ±0.06 floor |
| WSS_y | ~±0.06pp | |
| **WSS_z** | **±0.083pp (largest)** | NOT canonical screening axis; revise earlier hypothesis |
| VP | ±0.036pp | smallest — cross-axis compounds clear test_VP gate |

**Recalibration impact on active leaders**:
- **H185 thorfinn EP9 lead −0.108pp = 2× RNG floor → REAL signal** (sole compound clearly above noise)
- **H190 nezuko EP9 lead −0.011pp = well below RNG floor → noise-band 50/50 outcome**
- **H164e** LEADS H112 at EP11 (H112's reported baseline upper-end of RNG distribution)
- VP universally negative across H164d/H164e → cross-axis compounds expected to clear test_VP gate

### Earlier (H164d single-draw) framework — superseded by N=2 calibration above

| Channel | Δ slope (H164d vs H112) | Implication |
|---|---:|---|
| WSS aggregate | +0.040pp | RNG floor ≈ ±0.040pp single-draw |
| WSS_z | −0.009pp | (revised at N=2: actually LARGEST not stablest) |
| WSS_x | ~±0.05pp | (confirmed at N=2: ±0.059pp) |
| WSS_y | ~±0.06pp | (confirmed at N=2) |
| VP | ~±0.09pp | (revised at N=2: actually SMALLEST at ±0.036pp) |

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

## EP6 Cohort Snapshot (banked 19:13Z + 19:10Z)

| PR | Student | Hypothesis | EP6 val_abupt | Δ vs H112 EP6 (6.3741) | Notes |
|---|---|---|---:|---:|---|
| **#1353** | **thorfinn** | **H185: GradNorm × mirror-aug** | **6.2811** | **−0.093pp LEADING** | All 5 channels uniform 9-14bp lead; WSS_x basin invariant intact (−0.133pp); terminal projection ~6.06-6.08% |
| #1357 | frieren | H164e (H112 recipe RNG draw 2) | 6.3132 | −0.061pp | RNG calibration — bracketing established |
| (H112 baseline) | — | reference | 6.3741 | 0 | upper-end of recipe RNG distribution |
| (H164d) | — | H112 recipe RNG draw 1 | 6.4400 | +0.066pp | lower-end side of distribution |

**RNG bracketing finding** (frieren H164e EP6 vs H164d EP6): H112-recipe true mean ≈ 6.075% terminal val_abupt. H112's reported baseline is upper-end of recipe distribution. Merge gates remain set against H112's reported 6.1358%, but interpretation of "winning" should account for ~0.07pp RNG floor on val_abupt.

**H185 EP6 lead validates cross-axis hypothesis at EP6 binding diagnostic**:
- z×y compound (tau_z × mirror-aug) preserves basin (no WSS_x sign-flip)
- Synergistic effect amplifies vs H171 solo by −7.2bp at EP6
- Falsifies H147 EP6 LAG framework — slope-preservation hypothesis ALIVE

**H185 EP9 (banked 22:17Z)** — val_abupt **6.0948% LEADING H112 (6.2024) by −0.108pp** at EP9. WSS_x lead −0.150pp (basin invariant past H112 terminal). Lead consolidating (10-15bp band) not narrowing. Terminal projection val_abupt ~6.028 (108bp below merge gate). Test_WSS projection 6.528-6.675 across 3 slope scenarios — **ALL PASS 6.752 merge gate**. **H185 IS THE FIRST TAY-TRACK MERGE-ELIGIBLE CANDIDATE SINCE H112**. ETA terminal ~03:00-04:00Z+1.

**Terminal projection** (using cohort EP6→terminal mean drop −22.6bp):
- H185 → ~6.06-6.08% val_abupt → **beats H112 gate 6.1358% by 5-8bp** if realized → REVISED via EP9 actual: **~6.028 → beats by 108bp**
- First val-side merge-eligible result expected

**Updates banked 19:50Z**:
- **H190 nezuko EP5 LEADING H112 by −0.054pp** (val_abupt 6.4162 at step 43466). Mirror p=0.25 SUPPRESSES H148's over-regularization (VP +0.638pp tighter than p=0.5). EP5→terminal projection: ~6.082% val_abupt → beats H112 gate by ~5bp. **Second cross-axis compound to clear H112 mid-training**.
- **H191 tanjiro EP6 BEHIND H112 by +0.296pp** (val_abupt 6.6702). Not winner trajectory but WSS_x slope intact (−0.0128/1k_steps). **Mechanism finding**: tau_y=2.0+mirror = safe (basin intact); tau_y=3.0+mirror (H183) = unsafe (sign-flip). Threshold lies between 2.0 and 3.0.

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
- RNG floor on WSS_agg slope = ±0.063pp at N=2 EP11; treat as minimum effect size for single-draw mechanism claims
- RNG floor on **val_abupt = ±0.053pp at N=2 EP11** — claims must exceed 2× floor (±0.106pp) to count as real signal
- **WSS_z slope is NOT the canonical cohort-screening axis** (N=2 finding: ±0.083pp half-range, LARGEST not smallest) — supersedes earlier H164d single-draw hypothesis
- VP smallest RNG floor at ±0.036pp → cross-axis compounds clear test_VP gate

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
