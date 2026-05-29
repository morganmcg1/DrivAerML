# SENPAI Research State

**Updated**: 2026-05-29 09:50Z | Branch: `tay` | **NEW SOTA: H236 multi-res TTA PR #1408 MERGED** | Round 4g: 9 active | **STACKING ARM active (#1413 tanjiro — highest EV)**

---

## Current SOTA (UPDATED — H236 just merged)

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| Prior SOTA H112 (PR #1283) | 6.1358% | 5.839% | 6.752% | 3.421% | 3.695% |
| SOTA H185+TTA (PR #1382) | 5.9755% | 5.8221% | 6.7214% | 3.4400% | 3.6806% |
| **NEW SOTA H236 multi-res TTA (PR #1408)** | **5.9613%** | **5.8081%** | **6.7130%** | **3.4033%** | 3.6759% |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Merge gate (updated)**: val_abupt < **5.9613%** AND test_abupt < **5.8081%**

**Paper floors crossed**: test_VP 3.4033 ≤ 3.421 ✓ | test_WSS 6.7130 ≤ 6.727 ✓ | test_SP 3.6759 > 3.577 ✗

**Method**: {49152, 65536, 81920} × {orig, mirror-y} = 6-pass TTA on H185 EP13 EMA. -14bp val, -14bp test over H209.

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

### Round 4c CLOSED (multiple Findings banked)

| PR | Student | Outcome |
|---|---|---|
| ~~#1404~~ | askeladd H231 | CLOSED — Finding AA: mesh-subsample TTA falsified (no point-density invariance) |
| ~~#1406~~ | edward H233 | CLOSED — Finding BB: permutation invariant, null TTA gain |
| **#1408** | **nezuko H236** | **MERGED NEW SOTA — multi-res TTA val 5.9613 / test 5.8081** |
| ~~#1410~~ | thorfinn H239 | Running (mesh-subsample on H148, will confirm Finding AA at N=3) |
| ~~#1411~~ | fern H240 | CLOSED — Finding AA confirmed on H183 at N=2 |

### Round 4d/e closures (2 PRs)

| PR | Student | Outcome |
|---|---|---|
| ~~#1409~~ | alphonse H238 | CLOSED — Finding DD banked: α-sweep U-shapes at 0.5, α-tuning futile on equal-bias TTA pairs |
| ~~#1410~~ | thorfinn H239 | CLOSED — Finding AA confirmed at N=3; **Finding CC banked** (H148 density robustness) |

### Round 4e active fleet (9 PRs)

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| #1412 | frieren | H241: Per-channel TTA α on H185 EP13 | per-channel α (vs new gate) |
| #1413 | tanjiro | H242: Weight-space noise TTA on H185 | weight-space perturbation |
| #1414 | askeladd | H243: Extended multi-res range {32k-98k}×{orig,mirror}=10-pass | eval-only ~25min |
| #1415 | edward | H244: H185 EP14-16 cosine extension (TRAINING) | ~340min training + TTA |
| #1416 | fern | H245: Multi-res TTA on H183 EP13 | eval-only ~20min |
| **#1417** | **thorfinn** | **H246: Multi-res TTA on H148 EP13 — best test_abupt checkpoint** | eval-only ~20min |
| **#1418** | **nezuko** | **H247: Per-channel α multi-res TTA on H185 EP13** | per-channel α on H236 mechanism |
| **#1419** | **alphonse** | **H248: Mirror × multi-res × point-jitter TTA on H185 EP13** | third orthogonal TTA axis |
| ~~#1416~~ | ~~fern~~ | ~~H245: Multi-res on H183~~ | CLOSED — Finding EE(N=2): +14bp checkpoint-agnostic |
| ~~#1412~~ | ~~frieren~~ | ~~H241: Per-channel TTA α H185~~ | CLOSED — Finding DD-extension: per-channel α also collapses to 0.5 |
| ~~#1417~~ | ~~thorfinn~~ | ~~H246: H148 multi-res TTA~~ | CLOSED — Finding GG: val +76bp/test −6bp on H148 (density-robust outlier) |
| ~~#1418~~ | ~~nezuko~~ | ~~H247: Per-channel α multi-res TTA~~ | CLOSED — Finding DD-extension to multi-res: uniform is local optimum |
| **#1421** | **fern** | **H249: Tight-range multi-res TTA on H185 {57k,65k,73k}** | resolution spacing sensitivity |
| **#1422** | **frieren** | **H250: Frequency-weighted multi-res TTA — heavy K=65536** | resolution weighting by training-frequency bias |
| **#1425** | **nezuko** | **H251: Multi-res TTA on H188 EP13 — Finding GG N=4** | checkpoint portability probe |
| **#1426** | **thorfinn** | **H252: Weight-noise TTA on H148 EP13 — flat-basin probe** | Finding FF generalization to density-robust checkpoint |

**Budget remaining**: ~7.0h. edward H244 training sprint uses ~5.7h.

**Findings banked (input-space TTA exhaustion map)**:
| Mechanism | Result |
|---|---|
| Mirror-y | ✓ VALID (only valid input-space TTA) |
| Rotation θ≥0.1° | ✗ FALSIFIED (changes physics) |
| Coordinate scale ε=±2% | ✗ FALSIFIED (Reynolds regime change) |
| Mesh-subsample 80-95% | ✗ FALSIFIED (no point-density invariance) |
| Point permutation | NULL (invariant, no diversity) |
| Gaussian input noise | ✗ FALSIFIED (no usable σ band) |
| Multi-resolution | ✓ VALID — −14bp (H236 MERGED) |

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
| **AA** | this cycle | **Mesh-subsample TTA falsified at 80%+95% on H185 (H231), H183 (H240), H148 (H239). No point-density invariance. N=3 checkpoints. Finding confirmed.** |
| **BB** | this cycle | **Permutation TTA null — H185 slice-attention empirically permutation-invariant. Null signal, dilutes mirror weight (edward H233 #1406).** |
| **CC** | this cycle | **H148 density robustness: H148 loses only −0.6bp under 20% subsampling vs H185 losing ~100bp. H148 backbone is materially more point-density robust. Backbone selection criterion for future density-perturbing TTA.** |
| **DD** | this cycle | **α-sweep on equal-bias TTA pairs (same-model orig+mirror) is U-shaped at α=0.5 — theoretically guaranteed. Skip α-sweeps on same-architecture TTA pairs. (alphonse H238 #1409)** |
| **EE (prelim N=2)** | this cycle | **Multi-res TTA bonus is CHECKPOINT-INVARIANT at ~14bp. H185 (8bp mirror) +14bp; H183 (50bp mirror) +14bp. Mirror and multi-res mechanisms are independent — bonus does NOT scale with mirror sensitivity. N=3 pending (thorfinn H246).** |
| **DD-extension** | this cycle | **Per-channel α-sweep on same-checkpoint TTA ALSO collapses to 0.5 per channel (frieren H241). H235 Δ measures prediction disagreement, NOT bias asymmetry. α-tuning is futile at both global and per-channel levels for same-checkpoint mirror TTA.** |
| **FF (prelim)** | this cycle | **Weight-space noise σ ∈ [1e-5, 5e-4] gives MONOTONIC improvement on H185 EP13 (tanjiro H242 #1413). σ=5e-4: noise_only val 5.9845 (vs orig 6.0172, −33bp); +mirror→val 5.9674 (8bp better than H209). H185 sits in wide flat basin. Mechanism ORTHOGONAL to mirror+multi-res — stacked version (H242 extension) is highest-EV SOTA push.** |
| **GG** | this cycle | **Multi-res TTA portability is checkpoint-specific. H148: val +76bp WORSE / test −6bp (val-test sign flip). H185/H183: both +~14bp. H148 density robustness (Finding CC) already encodes cross-res signal → multi-res adds noise on val. H246 thorfinn #1417.** |
| **DD-ext2 (multi-res α)** | this cycle | **Per-channel α blending of multi-res vs single-res also sits at α=1.0 optimum (= H236 uniform). Surface channels benefit from multi-res too (just less than VP), so any pull-back of α_surface degrades aggregate. Nezuko H247 #1418.** |

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
