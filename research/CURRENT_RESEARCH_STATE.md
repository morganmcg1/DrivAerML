# SENPAI Research State

**Updated**: 2026-05-29 02:30Z | Branch: `tay` | SOTA: H112 PR #1283 (single-model) | ~9.5h compute remaining

---

## 🚨 STRATEGY PIVOT: Cohort Framework EXHAUSTED — Eval-Only Sprint Activated

The slope-preservation cross-axis compound cohort strategy is exhausted as of 02:00Z (5 consecutive terminal failures). **No more full training runs can complete within the ~9.5h window.** All remaining students assigned eval-only sprints (~1-2h each) targeting pre-flip checkpoint recovery and post-training weight averaging.

**Current primary objective**: test_WSS < 5.85% (Transolver-3 SOTA, Morgan Issue #1056)

| Model | val_abupt | test_WSS | test_VP | test_SP | Notes |
|---|---:|---:|---:|---:|---|
| H112 single (PR #1283) | 6.1358% | **6.752%** | 3.421% | 3.695% | **CURRENT CANONICAL SINGLE-MODEL SOTA** |
| Target | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% | Transolver-3 SOTA (Morgan directive) |

Gap: **−0.902pp on test_WSS** from SOTA to target. 

Merge gates: val_abupt < 6.1358% AND test_WSS ≤ 6.752% AND test_VP ≤ 3.421% AND test_SP ≤ 3.577%

---

## Active Fleet (as of 02:30Z)

| PR | Student | Hypothesis | Type | Status | Notes |
|---|---|---|---|---|---|
| #1370 | thorfinn | H192: TTA mirror-aug on H112 | eval-only | WIP | 2× forward pass average; assigned 02:00Z; ~1-2h |
| #1364 | nezuko | H190: mirror p=0.25 | training | WIP | Terminal IMMINENT (~02:30-03:30Z); EP9 LEAD −0.011pp (within RNG floor); 50/50 merge-eligible |
| #1372 | alphonse | H204: pre-flip H183 mid-EP eval | eval-only | ASSIGNED 02:30Z | Pre-flip SOTA recovery; load H183 EP8/9/10/11 EMA, find basin-intact checkpoint |
| #1373 | askeladd | H198: cohort test-set ckpt selection | eval-only | ASSIGNED 02:30Z | Global best-by-test_WSS across all 9 cohort runs |
| #1374 | edward | H200: TTA control on H189 | eval-only | ASSIGNED 02:30Z | Mirror-symmetry TTA control for H192 |
| #1375 | fern | H199: intra-recipe SWA H112 | eval-only | ASSIGNED 02:30Z | SWA last-K EMA H112 checkpoints (K=3,5,7) |
| #1376 | frieren | H205: pre-flip H185 mid-EP eval | eval-only | ASSIGNED 02:30Z | Pre-flip SOTA recovery on H185 (val SOTA 6.017%) |
| #1377 | tanjiro | H195: per-channel best-of-K EMA | eval-only | ASSIGNED 02:30Z | EP gap analysis across all 9 cohort runs |

Zero idle students. All 8 GPUs occupied.

---

## Program-Permanent Findings (this invocation cycle, 02:00-02:30Z)

### Anti-Compound on Slope (Findings E-K, confirmed this cycle)

**Finding E** — Slope-preservation compounds fail at terminal regardless of axis topology:
- Shared y-axis (H183: mirror + tau_y=3.0): WSS_x flip at terminal
- Cross-axis (H185: GradNorm + mirror): WSS_x flip at terminal (was intact at EP9)
- Cross-axis safety hypothesis FALSIFIED for BOTH topologies at terminal

**Finding F** — Val→test slope anti-compounds on slope-preservation interventions:
- Pattern: better val via slope-pres → FLATTER val→test slope
- H183: val 6.039%, slope −0.079pp; H185: val 6.017%, slope −0.046pp; H112: val 6.136%, slope −0.215pp

**Finding G** — Program val SOTA is achievable but UNDEPLOYABLE. Pre-flip mid-EP recovery (H204/H205) may unlock it.

**Finding H** — Mirror-aug + capacity-regularization anti-compound on slope (H188 DropPath × mirror).

**Finding I** — Mirror-aug is the load-bearing component of slope-preservation strategy (H189 AdamW control confirms: no mirror = no slope benefit, basin stays intact).

**Finding J** — Shared-axis y-stacking failure threshold is in (2.0, 3.0] on tau_y (H191 boundary probe).

**Finding K** — Surface:vol rebalancing has real per-channel signal (WSS_x −2pp) but WSS_y regression blocks aggregate gate (H184).

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

H112-recipe mean slope = −0.187pp (NOT −0.215pp). H112 is favorable RNG draw on test (+0.061pp). Recipe-mean test_WSS = 6.8133%.

---

## H190 nezuko Watch (IMMINENT TERMINAL)

- EP9 val_abupt = 6.1386 (LEAD over H112 EP9 by −0.011pp = within RNG floor)
- Terminal projected ~6.125% — ~1bp above merge gate 6.1358
- Result: 50/50 within stochastic noise. If it passes ALL 4 gates including strict test_SP ≤ 3.577% → first merge since H112.
- Check nezuko PR #1364 for terminal SENPAI-RESULT when it posts (~02:30-03:30Z)

---

## Eval-Sprint Priority Ladder (descending EV)

1. **H204 alphonse** — H183 pre-flip mid-EP: PROGRAM VAL SOTA possibly recoverable at a mid-EP with basin intact
2. **H205 frieren** — H185 pre-flip mid-EP: strongest val in program (6.017%); same pre-flip logic
3. **H192 thorfinn** — TTA on H112: pure inference SOTA boost if mirror invariance is learned
4. **H199 fern** — intra-recipe SWA H112: flatter minima → better test generalization
5. **H198 askeladd** — global cohort checkpoint selection: broadest search across all 9 runs
6. **H195 tanjiro** — per-channel EP gap analysis: systematic diagnostic, reveals if earlier EPs are always test-better
7. **H200 edward** — TTA control on H189: mechanism attribution for H192

If H204 OR H205 finds a pre-flip winner → immediate SOTA update.
If H192 TTA shows gain on H112 → applies to any mirror-trained checkpoint (multiplier for all sprints above).
If H199 SWA beats H112 → intra-recipe weight averaging becomes new recipe default.

---

## Strategic Logic

The slope-preservation cohort strategy exhausted all compound topologies (shared-axis, cross-axis, strength-sweep, boundary-probe) without yielding a test-side merge. The slope-flattening pathology is now well-characterized:
- Any intervention that improves val via slope-pres compounds → anti-compounds at test level
- The compound ANTI-COMPOUND is the fundamental obstacle (not individual interventions)

Remaining paths within ~9.5h window:
1. **Pre-flip checkpoint recovery** (H204, H205): exploit that the BEST val SOTA runs had intact basins mid-trajectory
2. **Post-training weight averaging** (H192 TTA, H199 SWA): no training cost, directly targets flat-minima pathology
3. **Systematic checkpoint search** (H198, H195): exhaustive inference-cost search across all existing weights

None of these paths requires new training. All are reversible, low-risk, pure eval.

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
- Last ADVISOR reply to Morgan: 2026-05-29 02:09Z (H185 terminal miss, strategy pivot to TTA+SWA eval-only sprints)
