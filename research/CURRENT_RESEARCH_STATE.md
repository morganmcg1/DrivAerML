# SENPAI Research State

**Updated**: 2026-05-29 05:05Z | Branch: `tay` | **NEW SOTA: H185+TTA PR #1382 merged** | ~6h compute remaining

---

## 🎯 NEW SOTA MERGED — First Sub-6.0% val_abupt, Closing Gap to Transolver-3 SOTA

H209 frieren MERGED at 04:35Z. H185+TTA pulls **val_abupt under 6.0%** for the first time in program history and improves test on every primary channel except a paper-floor near-miss on VP.

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| Prior SOTA H112 (PR #1283) | 6.1358% | 5.839% | 6.752% | 3.421% | 3.695% |
| **NEW SOTA H185+TTA (PR #1382)** | **5.9755%** | **5.8221%** | **6.7214%** | 3.4400% | 3.6806% |
| Δ | −16bp | −1.7bp | −3.1bp | +1.9bp | −14.4bp |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Gap to target test_WSS = 5.85**: now 0.87pp (~13% relative) from 0.90pp at session start. ~6h compute remaining.

**Merge gate (updated)**: val_abupt < 5.9755% AND test_abupt < 5.8221%. Test floors for paper claims: VP ≤ 3.421, SP ≤ 3.577, WSS ≤ 6.727.

---

## Findings Q + R Banked (this cycle)

### Finding Q — TTA on mirror-aug-trained models (N=3 across H183/H185/H190)
- ~4-5bp **uniform gain** on test_WSS for ANY mirror-aug-trained checkpoint
- WSS_y benefits most (−8 to −10bp) — direct y-mirror equivariance signal
- **WSS_x slope flip NOT recovered** by TTA — basin disruption is a weight-trajectory property
- TTA = STANDARD INFERENCE RECIPE for any future mirror-aug winner

### Finding R — Linear mode connectivity ABSENT (N=2 pairs)
- H112↔H183 (askeladd H207): val 88.78 at α=0.5
- H112↔H190 (fern H208): val 88.78 at α=0.5
- Both collapse to ~89% peak — wide high-loss ridge between basins
- Permutation-symmetry per Ainsworth 2022. Naive LERP not viable without rebasin.

### Finding N — TTA control floor (N=3 non-mirror-trained: H189/H112/H164e)
- +1.27 to +1.38pp test_WSS degradation
- Establishes control floor for TTA technique

### Finding O — Cross-recipe SWA destroys models (H210 tanjiro)
- Naive parameter averaging across 5 EP13 artifacts: 88-95% rel-L2
- Same mechanism as Finding R (permutation symmetry)

---

## Active Fleet (as of 05:05Z) — 3 WIP eval-only sprints, 5 idle awaiting assignment

| PR | Student | Hypothesis | Type | Status | ETA |
|---|---|---|---|---|---|
| #1386 | edward | H212: mirror invariance profile (5 EP13 cohort) | eval | WIP | ~25 min |
| #1387 | tanjiro | H213: block-wise splice H112↔H183 k=0..5 | eval | WIP | ~30 min |
| #1388 | askeladd | H214: sub-alpha sweep α ∈ {0.005..0.2} | eval | WIP | ~25 min |

**5 idle students** (alphonse, fern, frieren, nezuko, thorfinn) — researcher-agent generating next-round hypotheses.

---

## Next-Round Strategy Priorities (descending EV, 6h window)

1. **Training-heavy: H185 recipe extensions** (longer epochs, mirror p variants, different tau weighting) — direct extension of merged winner
2. **Training-heavy: WSS_x slope penalty / trajectory regularization** — the ONLY known mechanism to address Finding Q's "TTA doesn't recover WSS_x slope" gap
3. **Training-heavy: H112 fine-tune with mirror p=0.25 + L2 anchor** (fern's own H208 suggestion) — bypasses Finding R via optimizer-anchored continuation
4. **Eval-only: rotational TTA on H185** — extend Finding Q with geometric augmentations beyond y-mirror
5. **Eval-only: snapshot averaging WITHIN single H185 trajectory** (intra-recipe, late EP only) — not banned by ensemble policy

---

## Recipe for current SOTA H185+TTA

- **H185 training**: Lion, lr=9e-5, β1=0.9 β2=0.99, weight_decay=5e-4, batch=4, 13 EP, tau_y=3.0, tau_z=2.0, mirror p=0.25, compound H150-β, lr-warmup=1 EP, ema-decay=0.999
- **Source W&B run**: `yw2a5dyl` EP13 EMA artifact alias `epoch-13`/`best`
- **TTA**: `eval_tta_h209.py` 2-pass — original + y-mirrored pass, 0.5 avg in normalized space. Mirror op flips y in surface_x/n_y/volume_x, un-mirrors tau_y output. Eval-only at inference.

---

## Diagnostic Invariants (DO NOT VIOLATE)

- No capacity additions (≥+1% param overhead → slope flattening)
- **No ensembles** (Morgan directive). TTA is NOT ensembling (single model, single checkpoint, deterministic geometric averaging).
- DDP 8 GPUs every training run
- z-axis tau_z LOCKED at 2.0
- WSS_x slope sign is the BASIN-DISRUPTION DIAGNOSTIC
- data/loader.py, data/preload.py, data/split_manifest.json — READ-ONLY
- EP boundaries: EP6=48902, EP9=59780, EP10=62501, EP11=65184, EP13≈70657

---

## RNG Floor Calibration (N=2 EP12, PERMANENT)

| Channel | N=2 val half-range | N=2 slope half-range |
|---|---:|---:|
| WSS_agg | ±0.065pp | **±0.001pp ← canonical slope screening** |
| abupt | ±0.053pp | ±0.006pp |
| SP | ±0.020pp ← tightest val | ±0.004pp |
| WSS_z | ±0.082pp (LARGEST) | ±0.006pp |
| VP | ±0.037pp | ±0.009pp |
| WSS_x | ±0.065pp | ±0.012pp |
| WSS_y | ±0.060pp | ±0.021pp |

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-28 15:27Z)**: Ensembles BANNED. "PUSH HARD". test_WSS < 5.85% target.
- ADVISOR replies: 2026-05-29 02:09Z, 03:00Z, **05:05Z (this cycle — new SOTA report)**

## Findings Banked (full program)

- **E-K** (prior cycles): slope-pres compound failures, mirror-aug as load-bearing, tau_y stacking threshold
- **L** (cycle): Mirror-aug is BIMODAL (p=0.25 collapses slope, p=0.5 canonical)
- **M** (cycle): Mid-EP EMA checkpoints never saved program-wide
- **N** (this cycle): TTA on non-mirror-trained = +1.27-1.38pp control floor (N=3)
- **O** (this cycle): Cross-recipe SWA destroys models (Ainsworth permutation symmetry)
- **P** (this cycle): Linear mode connectivity ABSENT (covered by R below — finer naming)
- **Q** (this cycle): TTA on mirror-aug = +4-5bp gain, WSS_x slope NOT recovered (N=3)
- **R** (this cycle): Linear mode connectivity ABSENT for tay-track checkpoints (N=2)
