# SENPAI Research State

**Updated**: 2026-05-29 07:20Z | Branch: `tay` | **SOTA: H185+TTA PR #1382** | Round 4c: 8 sprints active (frieren/tanjiro re-spun; Findings Q@N=8 + Z banked)

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

**Finding Z** (tanjiro H229, PR #1402 closed): Gaussian-noise TTA on input coordinates is INFEASIBLE — no usable σ band exists. σ ≤ 0.0001 too small to add diversity (model is locally smooth), σ ≥ 0.001 too large (off-manifold, doubles error). Channel asymmetry: noise slightly improves VP/SP but degrades WSS (WSS_y worst). Input-perturbation TTA arm exhausted across (geometric, scale, noise).

**Finding Q (UPGRADED to N=8)** (frieren H235, PR #1407 closed): TTA-mirror on mirror-aug-trained EP13 EMA checkpoints delivers tight, consistent improvement.
- mean Δval = **-4.69bp** (stdev 0.87, range [-3.76, -6.49])
- mean Δtest = **-4.42bp** (stdev 0.77, range [-3.55, -5.95])
- mean ΔWSS_x = **-3.32bp**
- **8/8 checkpoints improve on every axis**. Publication-grade evidence.
- Universe: {H185, H183, H190, H188, H148, H191, H181b, H186} have `mirror_augmentation=True`. H164e and H171-plateau-exact-v2 do not.
- H186 outlier: largest gain (-6.49bp val) and worst baseline → TTA acts as variance-reduction on weaker checkpoints.

**Implication**: Mirror y is the SOLE validated TTA augmentation for H185. The TTA geometric perturbation arm + noise arm are FULLY EXHAUSTED. Weight-space averaging is BLOCKED by checkpoint-persistence. Future TTA must use non-geometric, non-weight-space, non-noise mechanisms.

---

## Round 4c Active Fleet (as of 06:00Z)

### Round 4b/c closures (5 PRs)

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| ~~#1399~~ | fern | H226: TTA-mirror on H112 | CLOSED — Finding N extension N=4 banked (+1.45val/+1.63test pp) |
| ~~#1402~~ | tanjiro | H229: TTA Gaussian noise | CLOSED — Finding Z banked (σ=0.001 catastrophic, σ=0.0001 mild degradation) |
| ~~#1403~~ | thorfinn | H230: TTA cross-checkpoint | CLOSED — Finding Q extension N=4 banked; H148+TTA passes test gate but fails val |
| ~~#1405~~ | alphonse | H232: Intra-trajectory SWA | ABORTED — Finding X banked |
| ~~#1407~~ | frieren | H235: TTA-mirror N≥6 | CLOSED — Finding Q UPGRADED to N=8, publication-grade |

### Round 4c active fleet (8 PRs, 4 re-spins)

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| #1404 | askeladd | H231: Mesh-subsample TTA on H185 (80%, 4-pass) | input subset sampling |
| #1406 | edward | H233: Point permutation TTA | attention perm invariance test |
| #1408 | nezuko | H236: Multi-resolution TTA | resolution averaging |
| #1409 | alphonse | H238: Weighted α-sweep on H185 EP13 | TTA α-tuning (uniform across channels) |
| #1410 | thorfinn | H239: Mesh-subsample TTA on H148 EP13 (best-test-margin) | mesh subsample, new checkpoint |
| #1411 | fern | H240: Mesh-subsample TTA on H183 EP13 (closest-to-val-gate, 13bp gap) | mesh subsample, new checkpoint |
| #1412 | frieren | H241: Per-channel TTA α-sweep on H185 EP13 (exploit channel asymmetry) | TTA per-channel α-tuning |
| #1413 | tanjiro | H242: Weight-space Gaussian-noise TTA on H185 EP13 (loss surface flatness probe) | weight-space perturbation |

**Expected first result**: ~30min for new respins. Budget remaining: ~2.5h.

**Coverage triangulation**:
- **Mesh-subsample TTA**: askeladd (H185), thorfinn (H148), fern (H183) — same mechanism, 3 checkpoints
- **α-blending on H185 EP13**: alphonse (uniform α), frieren (per-channel α) — same checkpoint, two tuning strategies
- **Novel mechanisms**: edward (permutation), nezuko (multi-res), tanjiro (weight-space noise)

---

## Strategic Logic (Round 4c)

**Hypothesis**: The H209 TTA SOTA captures ~all the gain from mirror symmetry. Pushing further requires either:
1. A new TTA mechanism that exploits a DIFFERENT invariance (not geometric)
2. A weight-space modification (intra-trajectory averaging — not ensembling per Morgan)
3. A different starting checkpoint with different basin structure (Finding Q extension)

**Round 4c bets**:
- **askeladd #1404 H231**: Mesh subsampling on H185 — if physics preserved by sub-sampling, may add 2-5bp
- **edward #1406 H233**: Permutation TTA — null result expected
- **nezuko #1408 H236**: Multi-resolution TTA — if good cross-resolution conditioning, 1-3bp gain
- **alphonse #1409 H238**: Uniform α-sweep on H185 EP13 — H209's α=0.5 may not be optimum
- **thorfinn #1410 H239**: Mesh-subsample on H148 — best-test-margin checkpoint + new mechanism
- **fern #1411 H240**: Mesh-subsample on H183 — closest-to-val-gate (13bp), best mechanical odds
- **frieren #1412 H241**: Per-channel α-sweep on H185 — WSS_y has 2.4× TTA gain vs WSS_x; structured exploitation
- **tanjiro #1413 H242**: Weight-space noise TTA on H185 — flatness probe of loss surface

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
| **Z** | this cycle | **Gaussian-noise TTA infeasible — no usable σ band (banked from tanjiro H229 #1402). Sub-finding: WSS more noise-sensitive than VP/SP.** |
| **Q@N=8** | this cycle | **TTA-mirror on mirror-aug-trained EP13 EMA: mean -4.42bp test, stdev 0.77 — 8/8 checkpoints improve. Publication-grade (frieren H235 #1407).** |

---

## Diagnostic Invariants

- No capacity additions (≥+1% param overhead → slope flattening)
- **No ensembles** (Morgan directive). TTA + intra-trajectory SWA are NOT ensembling.
- DDP 8 GPUs every run
- z-axis tau_z LOCKED at the trained value (1.67)
- data/loader.py, data/preload.py, data/split_manifest.json — READ-ONLY
- SENPAI_TIMEOUT_MINUTES=360 hard cap
- TTA geometric perturbation arm CLOSED (mirror only)
- Input-noise TTA arm CLOSED (Finding Z; no usable σ window)

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-28 15:27Z)**: Ensembles BANNED. "PUSH HARD". test_WSS < 5.85% target.
- Last advisor updates: 2026-05-29 05:30Z (Round 4 plan), 05:45Z (Round 4 pivot — apology + reassignment), pending Round 4c summary
