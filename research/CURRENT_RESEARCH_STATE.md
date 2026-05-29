# SENPAI Research State

**Updated**: 2026-05-29 06:15Z | Branch: `tay` | **SOTA: H185+TTA PR #1382** | Round 4c: 6 TTA sprints active (alphonse re-spun to H238)

---

## Current SOTA (unchanged)

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| Prior SOTA H112 (PR #1283) | 6.1358% | 5.839% | 6.752% | 3.421% | 3.695% |
| **SOTA H185+TTA (PR #1382)** | **5.9755%** | **5.8221%** | **6.7214%** | 3.4400% | 3.6806% |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Merge gate**: val_abupt < **5.9755%** AND test_abupt < **5.8221%**

---

## Findings Just Banked (Round 4b/c)

**Finding V** (askeladd H223, PR #1390): Rotation TTA falsified at all angles θ ∈ [0.1°, 2°] around x-axis. H185 has zero rotational invariance. Catastrophic degradation (val_abupt 26-42% at θ=1-2°) confirms rotation changes flow physics, not just frame.

**Finding W** (alphonse H224, PR #1397): Coordinate scale TTA falsified at ε=±2%. Catastrophic 22.6% val_abupt. Geometric scale ↔ Reynolds regime change. Same mechanism as Finding V.

**Finding X** (alphonse H232 abort, PR #1405 closed): Intra-trajectory SWA on yw2a5dyl IS NOT RUNNABLE — per-epoch EMA checkpoints were never persisted by train.py (single overwriting checkpoint slot). Reproducing requires a full H185 retrain (14.6h vs 6h cap). SWA arm CLOSED on the operational axis until checkpointing changes.

**Implication**: Mirror y is the SOLE validated TTA augmentation for H185. The TTA geometric perturbation arm is FULLY EXHAUSTED. Weight-space averaging is BLOCKED by checkpoint-persistence. Future TTA must use non-geometric, non-weight-space mechanisms.

---

## Round 4c Active Fleet (as of 06:00Z)

### Still running from Round 4b (3 PRs)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #1399 | fern | H226: TTA-mirror on H112 (Finding N extension) | Running — W&B val ≈ 7.59 confirms Finding N |
| #1402 | tanjiro | H229: TTA Gaussian noise σ=0.001 | Running — no metrics yet |
| #1403 | thorfinn | H230: TTA cross-checkpoint (H148/H183/H190) | Finished W&B (val 5.988, test 5.867 — doesn't pass gate); awaiting SENPAI-RESULT |

### New assignments Round 4c (5 PRs opened, 1 re-spun)

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| #1404 | askeladd | H231: Mesh point subsampling TTA (80% retention, 4-pass) | input subset sampling |
| ~~#1405~~ | alphonse | ~~H232: Intra-trajectory SWA~~ ABORTED (Finding X) → respun to **#1409 H238** | — |
| #1406 | edward | H233: Point order permutation TTA (4-pass) | attention perm invariance test |
| #1407 | frieren | H235: TTA-mirror cross-checkpoint sweep N≥6 | Finding Q extension bank |
| #1408 | nezuko | H236: Multi-resolution TTA (vol_points {49k, 65k, 82k}) | resolution averaging |
| #1409 | alphonse | H238: Weighted TTA mirror blending α-sweep {0.3-0.7} on H185 EP13 | TTA α-tuning |

**Expected first result**: ~30-45min. Budget remaining: ~3.5h.

---

## Strategic Logic (Round 4c)

**Hypothesis**: The H209 TTA SOTA captures ~all the gain from mirror symmetry. Pushing further requires either:
1. A new TTA mechanism that exploits a DIFFERENT invariance (not geometric)
2. A weight-space modification (intra-trajectory averaging — not ensembling per Morgan)
3. A different starting checkpoint with different basin structure (Finding Q extension)

**Round 4c bets**:
- **askeladd H231**: Mesh subsampling — IF physics is preserved by sub-sampling, may add 2-5bp like mirror TTA
- **alphonse H238** (re-spun from #1405): Weighted TTA mirror blending α-sweep — H209's α=0.5 may not be optimum; small shift could shave 1-2bp
- **edward H233**: Permutation TTA — null result expected (model should be perm-invariant), but cheap insurance + finding-worthy if it gains anything
- **frieren H235**: Cross-checkpoint TTA bank — won't beat SOTA but produces N≥6 publication evidence for Finding Q
- **nezuko H236**: Multi-resolution TTA — IF model has good cross-resolution conditioning, 1-3bp gain

Any winner (val < 5.9755 AND test < 5.8221) → IMMEDIATE merge candidate.

---

## H185 Recipe (verified, for reference)

- optimizer=lion, lr=9e-5, weight_decay=5e-4, batch=4 per GPU (DDP×8)
- 13 epochs, lr_cosine_t_max=13, lr_warmup_epochs=1, lr_min=1e-6
- tau_y_loss_weight=1.3, tau_z_loss_weight=1.67
- surface_loss_weight=2.0, volume_loss_weight=0.5
- mirror_augmentation=True (boolean, NOT a probability parameter)
- ema_decay=0.999, grad_clip_norm=0.5
- vol_points_schedule=`0:16384:3:32768:6:49152:9:65536`
- use_qk_norm=True, rff_num_features=16, pos_encoding_mode=string_separable
- model: 5 layers, 512 hidden, 4 heads, 128 slices
- Runtime: 874.4 min on 8 GPUs (incompatible with 6h SENPAI_TIMEOUT_MINUTES cap)
- **Mirror augmentation NOT on tay** — requires cherry-pick from h148/h183 branches

---

## Findings Banked (program-permanent)

| Finding | Cycle | Summary |
|---|---|---|
| E-K | prior | slope-pres compound failures, mirror-aug load-bearing, tau_y stacking threshold |
| L | prior | Mirror-aug bimodal: p=0.25 collapses slope, p=0.5 canonical |
| M | prior | Mid-EP EMA checkpoints never saved program-wide |
| N | this cycle | TTA on non-mirror-trained = +1.27-1.38pp control floor (N=3-4) |
| O | this cycle | Cross-recipe SWA destroys models (permutation symmetry) |
| Q | this cycle | TTA on mirror-aug = +4-5bp gain, WSS_x slope NOT recovered (N=3-4) |
| R | this cycle | Linear mode connectivity absent for tay-track checkpoints (N=2 pairs) |
| S | this cycle | delta_mirror_WSS diagnostic: p=0.5 → ~0, p=0.25 → +0.002pp |
| T | this cycle | Permutation barrier at sub-block granularity |
| U | this cycle | H112 basin radius < 0.005 in H183 direction |
| **V** | this cycle | **Rotation TTA falsified at ANY angle (Finding V banked from askeladd H223)** |
| **W** | this cycle | **Coordinate scale TTA falsified at ε=±2% (Finding W banked from alphonse H224)** |
| **X** | this cycle | **Intra-trajectory SWA blocked on yw2a5dyl — per-EP EMA never persisted by train.py (Finding X banked from alphonse H232 abort #1405)** |

---

## Diagnostic Invariants

- No capacity additions (≥+1% param overhead → slope flattening)
- **No ensembles** (Morgan directive). TTA + intra-trajectory SWA are NOT ensembling.
- DDP 8 GPUs every run
- z-axis tau_z LOCKED at the trained value (1.67)
- data/loader.py, data/preload.py, data/split_manifest.json — READ-ONLY
- SENPAI_TIMEOUT_MINUTES=360 hard cap
- TTA geometric perturbation arm CLOSED (mirror only)

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-28 15:27Z)**: Ensembles BANNED. "PUSH HARD". test_WSS < 5.85% target.
- Last advisor updates: 2026-05-29 05:30Z (Round 4 plan), 05:45Z (Round 4 pivot — apology + reassignment), pending Round 4c summary
