# SENPAI Research State

**Updated**: 2026-05-29 05:30Z | Branch: `tay` | **SOTA: H185+TTA PR #1382** | Round 4 fleet: 8/8 active

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| Prior SOTA H112 (PR #1283) | 6.1358% | 5.839% | 6.752% | 3.421% | 3.695% |
| **SOTA H185+TTA (PR #1382)** | **5.9755%** | **5.8221%** | **6.7214%** | 3.4400% | 3.6806% |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Gap to test_WSS target**: 0.87pp (13% relative) — ~6h compute remaining.

**Merge gate**: val_abupt < **5.9755%** AND test_abupt < **5.8221%**
**Paper floor**: test_VP ≤ 3.421, test_SP ≤ 3.577, test_WSS ≤ 6.727

Source: W&B `bx3t1vdw` | H185 checkpoint: `yw2a5dyl` EP13 EMA

---

## Round 4 Active Fleet (all 8 students WIP as of 05:30Z)

| PR | Student | Hypothesis | Type | ETA |
|---|---|---|---|---|
| #1389 | alphonse | H215: Ultra-low-LR continuation EP13→EP16 (lr=1e-6 constant) | training-light | ~2h |
| #1390 | askeladd | H223: Rotational TTA ±2° on H185 (eval-only) | eval-only | ~45min |
| #1391 | edward | H222: Tau_x channel upweight 2.0 — address WSS_x slope flip at training time | training-heavy | ~3.5h |
| #1392 | fern | H216: H185 + tau_y=2.0 retrain (de-escalate below Finding J boundary) | training-heavy | ~3.5h |
| #1393 | frieren | H217: H185 EP15 cosine extension from EP13 checkpoint | training-light | ~2h |
| #1394 | nezuko | H218: Mirror p=0.5 + tau_y=2.0 — untested (p × tau_y) grid cell | training-heavy | ~4h |
| #1395 | tanjiro | H221: Lion→AdamW switch at EP10 — late-epoch optimizer refinement | training-heavy | ~3.5h |
| #1396 | thorfinn | H219: Fresh H185 seed + within-recipe LERP test | training-heavy | ~5h |

**Expected first result**: H223 (askeladd, ~45min)

---

## Current Research Focus

### Primary Lever: WSS_x Slope Repair
Finding Q established TTA cannot recover the WSS_x slope flip from H185. **Training-time interventions targeting the slope** are the only known path to further WSS improvement:
- H222 (edward): tau_x upweight 2.0 — direct channel gradient pressure
- H221 (tanjiro): Lion→AdamW switch at EP10 — late-epoch optimizer change
- H216 (fern): tau_y=2.0 retrain — reduce tau_y below Finding J boundary
- H218 (nezuko): p=0.5 + tau_y=2.0 — untested grid cell combining both mitigations

If WSS_x slope is restored AND val_abupt is competitive → TTA stacks on top → SOTA candidate.

### Secondary Lever: Continuation Strategies
H185 EP13 is the strongest weight starting point in program history. Low-risk extensions:
- H215 (alphonse): lr=1e-6 constant, 3 EP — optimizer-anchored refinement
- H217 (frieren): cosine extended to EP15 — unconverged trajectory
- H219 (thorfinn): fresh seed + within-recipe LERP — test same-recipe basin connectivity

### Tertiary Lever: New TTA Geometry
- H223 (askeladd): Rotational TTA ±2° — extends TTA equivariance to rotation axis model hasn't absorbed

---

## Findings Banked (program-permanent)

| Finding | Cycle | Summary |
|---|---|---|
| E-K | prior | slope-pres compound failures, mirror-aug load-bearing, tau_y stacking threshold |
| L | prior | Mirror-aug bimodal: p=0.25 collapses slope, p=0.5 canonical |
| M | prior | Mid-EP EMA checkpoints never saved program-wide |
| N | this cycle | TTA on non-mirror-trained = +1.27-1.38pp control floor (N=3) |
| O | this cycle | Cross-recipe SWA destroys models (permutation symmetry) |
| Q | this cycle | TTA on mirror-aug = +4-5bp gain, WSS_x slope NOT recovered |
| R | this cycle | Linear mode connectivity absent for tay-track checkpoints (N=2 pairs) |
| S | this cycle | delta_mirror_WSS diagnostic: p=0.5 → ~0, p=0.25 → +0.002pp |
| T | this cycle | Permutation barrier at sub-block granularity (even 1 cross-recipe block destroys) |
| U | this cycle | H112 basin radius < 0.005 in H183 direction — definitively closes linear-interp arm |

---

## Diagnostic Invariants

- No capacity additions (≥+1% param overhead → slope flattening)
- **No ensembles** (Morgan directive). TTA is NOT ensembling.
- DDP 8 GPUs every training run
- z-axis tau_z LOCKED at 2.0
- WSS_x slope sign = BASIN-DISRUPTION DIAGNOSTIC (positive = flipped = bad)
- data/loader.py, data/preload.py, data/split_manifest.json — READ-ONLY
- EP boundaries: EP6=48902, EP9=59780, EP10=62501, EP11=65184, EP13≈70657

---

## H185 Recipe (current SOTA backbone)

Lion, lr=9e-5, β1=0.9 β2=0.99, wd=5e-4, batch=4, 13 EP cosine, tau_y=3.0, tau_z=2.0, mirror p=0.25, compound H150-β, lr-warmup=1 EP, ema-decay=0.999 | TTA via eval_tta_h209.py (2-pass y-mirror average)

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-28 15:27Z)**: Ensembles BANNED. "PUSH HARD". test_WSS < 5.85% target.
- Last advisor update to Morgan: 2026-05-29 05:30Z (Round 4 fleet deployed)
