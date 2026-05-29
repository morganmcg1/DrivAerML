# SENPAI Research State

**Updated**: 2026-05-29 03:00Z | Branch: `tay` | SOTA: H112 PR #1283 (single-model) | ~8.5h compute remaining

---

## 🚨 INFRASTRUCTURE BLOCKER + STRATEGY DOUBLE PIVOT — EP13-Only Eval Sprint Activated

The slope-preservation cohort strategy exhausted at 02:00Z. The eval-only sprint pivot at 02:30Z (6 PRs) was itself **immediately invalidated at 02:43Z** by Alphonse's H204 diagnostic (PR #1372): **mid-EP EMA checkpoints were never saved program-wide** — standard `train.py` writes only single "best" EMA snapshot, overwritten on every val improvement. All W&B artifact aliases for every run: `['best', 'latest', 'epoch-13']`. PVC scan confirms no per-EP files.

5 of 6 eval-sprints (H198 askeladd, H199 fern, H204 alphonse, H205 frieren, H195 tanjiro) **proactively closed** before any GPU was wasted. The 6th (H200 edward) is still viable since it doesn't depend on mid-EP artifacts.

**Current primary objective**: test_WSS < 5.85% (Transolver-3 SOTA, Morgan Issue #1056)

| Model | val_abupt | test_WSS | test_VP | test_SP | Notes |
|---|---:|---:|---:|---:|---|
| H112 single (PR #1283) | 6.1358% | **6.752%** | 3.421% | 3.695% | **CURRENT CANONICAL SINGLE-MODEL SOTA** |
| Target | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% | Transolver-3 SOTA (Morgan directive) |

Gap: **−0.902pp on test_WSS** from SOTA to target.

Merge gates: val_abupt < 6.1358% AND test_WSS ≤ 6.752% AND test_VP ≤ 3.421% AND test_SP ≤ 3.577%

---

## Active Fleet (as of 02:55Z) — 8 students, all EP13-only sprints

| PR | Student | Hypothesis | Type | Status | Notes |
|---|---|---|---|---|---|
| #1370 | thorfinn | H192: TTA mirror-aug on H112 | TTA | WIP | 2× forward pass average; assigned 02:00Z |
| #1374 | edward | H200: TTA control on H189 (non-mirror) | TTA control | WIP | Mechanism attribution for H192 |
| #1379 | alphonse | H206: TTA mirror-aug on H183 EP13 | TTA recovery | WIP | val SOTA 6.039%, basin-flipped on test |
| #1380 | askeladd | H207: weight interp H112 ↔ H183 EP13 | parameter manifold | WIP | 5-alpha sweep, basin-traversal diagnostic |
| #1381 | fern | H208: weight interp H112 ↔ H190 EP13 | parameter manifold | WIP | 5-alpha sweep across mirror p=0.25 axis |
| #1382 | frieren | H209: TTA mirror-aug on H185 EP13 | TTA recovery | WIP | program val SOTA 6.017% |
| #1383 | tanjiro | H210: cross-recipe SWA over 5 EP13 artifacts | consensus basin | WIP | 4 configs: SWA5, SWA3-baseline, SWA3-val-SOTA, weighted |
| #1384 | nezuko | H211: TTA mirror-aug on H190 EP13 | TTA recovery | WIP | own recipe — bimodal mechanism comparator |

Zero idle students. All 8 GPUs occupied. Expected runtimes: TTA ~1h, weight-interp ~1.5h, cohort SWA ~2h.

---

## Program-Permanent Findings (this invocation cycle)

### Finding L — Mirror-Aug is BIMODAL (this cycle, H190 nezuko terminal)

Three-point mirror-aug sweep CONCLUSIVE:

| p | val_abupt | test_WSS | WSS slope | Mechanism |
|---|---:|---:|---:|---|
| 0 (H112) | 6.136 | 6.752 | −0.215pp | baseline |
| 0.25 (H190) | **6.077 BETTER** | 6.820 WORSE | **−0.085pp FLATTEST** | slope mechanism COLLAPSED |
| 0.5 (H148) | 6.219 WORSE | 6.775 ~neutral | **−0.296pp STEEPEST** | slope mechanism active |

There is NO intermediate p where both val improves AND slope steepens. p=0.25 collapses slope mechanism (val gain is val-set over-reg); p=0.5 is canonical operating point.

**Retroactive explanation**: H183 (val SOTA at p=0.5+tau_y=3) val gain was tau_y-driven, NOT slope-mechanism-driven.

### Finding M — Mid-EP EMA Checkpoints Never Saved (PROGRAM-PERMANENT, this cycle)

Standard `train.py` saves single best EMA snapshot, overwritten every val improvement. For monotone-improving runs (H112, H164e, H183, H185, H188, H189, H190, H191), only EP13 "best" exists. **Future trainer should save per-EP EMA artifacts.** Outside current compute window.

### Findings E-K (banked, prior 02:30Z cycle)

- **E**: Slope-pres compounds fail at terminal regardless of axis topology (shared y / cross-axis)
- **F**: Val→test slope anti-compounds on slope-pres interventions
- **G**: Program val SOTA achievable but undeployable
- **H**: Mirror-aug × capacity-regularization anti-compound on slope (H188 DropPath)
- **I**: Mirror-aug is the load-bearing component (H189 AdamW control)
- **J**: tau_y stacking failure threshold ∈ (2.0, 3.0]
- **K**: Surface:vol rebalance — WSS_x −2pp gain but WSS_y regression

---

## RNG Floor Calibration (N=2 EP12, PERMANENT)

| Channel | N=2 val half-range | N=2 slope half-range |
|---|---:|---:|
| WSS_agg | ±0.065pp | **±0.001pp ← canonical slope screening** |
| abupt | ±0.053pp | ±0.006pp |
| SP | **±0.020pp ← tightest val** | ±0.004pp |
| WSS_z | ±0.082pp (LARGEST) | ±0.006pp |
| VP | ±0.037pp | ±0.009pp |
| WSS_x | ±0.065pp | ±0.012pp |
| WSS_y | ±0.060pp | ±0.021pp |

---

## Eval-Sprint Priority Ladder (descending EV, 02:55Z)

1. **H209 frieren** — TTA on H185 (strongest val 6.017%): if mirror invariance recovers basin → SOTA
2. **H206 alphonse** — TTA on H183 (val 6.039%): same logic, 2nd-strongest val
3. **H192 thorfinn** — TTA on H112: baseline TTA, validates the technique
4. **H210 tanjiro** — Cross-recipe SWA over 5 EP13 artifacts: consensus basin from 5 distinct mechanism failures
5. **H207 askeladd** — H112 ↔ H183 weight interp: parameter-manifold basin-traversal diagnostic
6. **H208 fern** — H112 ↔ H190 weight interp: parameter-manifold across collapsed-mechanism axis
7. **H211 nezuko** — TTA on H190: own recipe, bimodal-mechanism comparator
8. **H200 edward** — TTA control on H189 (non-mirror-trained): mechanism attribution for H192

If H209 OR H206 finds a TTA-recovered SOTA → IMMEDIATE merge candidate.
H192 sets the TTA baseline. H210 provides consensus basin if no single TTA wins.

---

## Strategic Logic — Why Eval-Only

No new training will complete in the ~8.5h window. We extract SOTA from existing weights via:
1. **TTA mirror-aug** (H192, H200, H206, H209, H211): exploit learned mirror invariance at inference, no training cost
2. **Weight-arithmetic** (H207, H208, H210): traverse parameter manifold between val-SOTA basin-flipped runs and basin-intact H112

The TTA arm has highest EV: 3 cross-validation runs (H183/H185/H190) all share mirror-aug training but at different intensities. If TTA recovers WSS_x basin on ANY of these → confirms learned mirror invariance was present but underutilized at inference.

The weight-interp arm probes WHERE in parameter space the basin disrupts: alpha=0 (H112, slope −0.215pp) → alpha=1 (H183/H190, slope flipped). Is disruption localized or distributed?

---

## Diagnostic Invariants (DO NOT VIOLATE)

- No capacity additions (≥+1% param overhead → slope flattening)
- No ensembles (Morgan directive — single model only)
- DDP 8 GPUs every run
- z-axis tau_z LOCKED at 2.0 (4-point monotone closure)
- WSS_x slope sign is the BASIN-DISRUPTION DIAGNOSTIC (negative = intact, positive = disrupted)
- data/loader.py, data/preload.py, data/split_manifest.json — READ-ONLY
- Actual EP boundaries (tanjiro calibration): EP6=48902, EP9=59780, EP10=62501, EP11=65184, EP13≈70657

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-28 15:27Z)**: Ensembles BANNED. "PUSH HARD". test_WSS < 5.85% target.
- ADVISOR replies to Morgan: 2026-05-29 02:09Z (H185 terminal miss → TTA pivot), 03:00Z (infrastructure blocker + 8-student eval roster — PENDING POST)
