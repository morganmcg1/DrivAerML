# SENPAI Research State

**Updated**: 2026-06-01 16:35Z | Branch: `tay` | **SOTA: H336 K=5+Student-t ν=4+8-res+mirror+cal — val_cal 5.8978 / test_cal 5.7379**

---

## Current SOTA (H336 calibrated)

| Model | val_cal | test_cal | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| **H336 K=5+Student-t ν=4+8-res+cal ← CURRENT SOTA** | **5.8978%** | **5.7379%** | **6.6382%** | **3.3735%** | **3.6133%** | 348i3z1v |
| Transolver-3 target | — | — | **< 5.850%** | ≤ 3.421% | ≤ 3.577% | — |

**Merge gate**: val_abupt_calibrated < **5.8978%** AND test_abupt_calibrated < **5.7379%**

**WSS gap**: test_WSS_z 8.6175% → 5.85% Transolver target = **0.788pp remaining** (primary obstacle)
**test_SP gap**: 3.6133% vs floor 3.577% = **+3.6bp** (edward H338 targeting this)
**test_VP**: 3.3735% ✓ (safely below 3.421% floor)

---

## Closed axes (do NOT revisit)

| Axis | Finding | Closed by |
|---|---|---|
| TTA resolution R | `res-density-saturated-8res` — R=8 at 32k–131k is the Pareto point | H267, H291, H331 |
| TTA K (Gaussian) | `K5-cal-redundant-at-h312-budget` under Gaussian noise | H330 |
| TTA noise-family ν | ν=4 Student-t is optimal; ν=3/6/8 all worse | H314 merged |
| **TTA noise σ at ν=4** | `sigma-axis-closed-nu4` — σ=5e-4 unimodal; σ=2.5e-4 and σ=1e-3 both regress all 5 channels (val + test). σ × ν joint axis fully saturated. | **H340 closed (this loop)** |
| Per-channel cal (affine) | OLS-MLE is the global optimum at val_N=34 | H316, H319, H323, H328, H329, H332, H333, H334 |
| Cal yield ceiling | `cal-cannot-rescue-train-raw-regression` — cal extracts ~7-8bp val regardless of starting raw; cannot close >10bp raw deficit. Use as fast-path close criterion. | **H343 closed (this loop)** |
| Weight-space soup (cross-seed) | α-soup axis exhausted; val gate irreducible via cross-seed averaging | H307 |
| WSS loss reweight (uniform) | `wz-reweight-monotone-nogate` — H339 closed. All channels regress uniformly | H339 closed |
| WSS loss reweight (wz-only) | `wz-reweight-monotone-nogate` — H341 CLOSED. Even isolated wz reweight regresses all channels | H341 closed |
| K-axis continuation at ν=4 | `K6-vs-K5-noise-floor-tie` — K=6 ties K=5 at ±0.02bp noise floor; K-axis saturated at K=5 | H344 closed |
| Gradient-direction conflict | `pcgrad-no-conflict-falsifies` — mean cos(g_wss, g_pres) = +0.30, 99.55% positive. PCGrad would be a no-op. | H345 closed |
| WSS gradient magnitude lever | `wss-gradient-already-dominant` — |g_wss|/|g_pres| = 5.12 at H336 base. Loss-weight up/down both unproductive. | H345 closed |
| WSS_z per-batch focal weighting | `focal-batch-collapsing` — γ∈{2.0, 1.0} both trip 50× collapse rule; even identity exponent collapses to 1-2 vertices on 88.5% of batches | H346 closed |
| **WSS_z loss-reweight axis (combined)** | `wss-z-loss-weight-axis-closed` — H339 uniform + H341 wz-only + H346 per-vertex focal all fail. **TRIPLY CLOSED.** WSS_z floor at 8.62% test_cal is NOT addressable by any loss-reweight scheme at the EP13 cosine-tail base — bottleneck is representational, not gradient-budget | H346 closed |
| **SAM cosine-tail at any ρ** | `sam-flatness-pessimal-wssz` + `sam-monotone-regression-rho` — ρ∈{0.05, 0.02, 0.002} all regress; WSS_z pessimal (+50bp@ρ=0.002). H336 basin is already flat enough; SAM perturbations push OUT of the WSS_z local fit. Flatness-regularization axis closed. | **H343 closed (this loop)** |

---

## Active Fleet (2026-06-01 16:35Z — 7 students with open PRs, frieren idle pending H353)

| PR | Student | Hypothesis | Status | Theme |
|---|---|---|---|---|
| **#1544** | thorfinn | H352: SWA-within-cosine-tail — weight-space averaging of ~30 fine-grained snapshots along H336 cosine-tail; distinct from H307 cross-seed soup and H342 output-avg. Zero param overhead. — NEW | 🆕 just assigned | Weight-space averaging (same trajectory) |
| **#1543** | tanjiro | H351: NGSB (Normal-Relative Geometric Slice Bias) — 24-param zero-init `nn.Linear(3 → num_heads)` bias on Transolver slice_logits via surface normals; attacks encoder-resident attention-slice-routing axis. References GeoTransolver Adams et al. Dec 2025. — NEW | 🆕 just assigned | Encoder slice-routing |
| **#1542** | askeladd | H350: Channel-isolated FiLM-conditioned decoder (Phase A frozen-backbone diagnostic gate, then Phase B full finetune) | 🟡 WIP — implementing | Architecture / per-channel decoder capacity |
| **#1539** | fern | H348: Surface curvature input features (H, K, k1k2) — Arm A H train terminal raw val_abupt **6.0088** (essentially TIED with H336 raw), val_wss_z 9.1847 tied, test_abupt 5.8539 tied. **Pending TTA cal eval** (sent back 16:30Z, nudge for SENPAI-RESULT and TTA stack) | 🟡 WIP — awaiting student TTA eval | Input geometry |
| **#1538** | nezuko | H347: Boundary-layer physics priors (τ⊥n normals + kNN smoothness) — running 1.5h, ETA ~17:30Z | 🟡 WIP — training | Physical constraint |
| **#1526** | alphonse | H342: Multi-checkpoint output averaging (ep13+ep14+ep15 TTA) | 🟡 WIP | Output-space averaging |
| **#1522** | edward | H338: Arm D compositional eval (Arm C SP-reweight EP15 × H336 K=5+ν=4+8-res+mirror TTA recipe) `9t27gag4`, ETA ~21:30Z | 🟡 WIP — eval | SP floor gap 3.6bp via composition |

**Closed this loop**:
- PR #1524 tanjiro H340 σ-sweep at ν=4: `sigma-axis-closed-nu4` + `per-channel-alpha-sigma-drift`. TTA hyperparameter family fully saturated.
- PR #1528 thorfinn H343 SAM cosine-tail: `sam-flatness-pessimal-wssz` + `sam-monotone-regression-rho` + `cal-cannot-rescue-train-raw-regression`. Flatness-regularization axis closed.
- PR #1540 frieren H349 arcsinh-wss_z target: `arcsinh-wssz-target-pessimal` + `target-transform-direction-falsified-compressive`. +72bp val_wss_z catastrophic. Direction-of-effect of target-transform on heavy-tailed channel is OPPOSITE to compression — H353 frieren next assignment tests the expansive converse.

---

## Current Research Focus: 5-axis WSS_z attack on orthogonal mechanisms (loss-reweight + flatness axes closed)

The **primary obstacle** is test_WSS_z = 8.6175% (277bp above Transolver-3's target 5.85%). With H336+: loss-reweight axis TRIPLY closed (H339+H341+H346), flatness/SAM closed (H343), gradient-direction/magnitude closed (H345), TTA hyperparameter family fully saturated (H330+H340+H344). **The WSS_z bottleneck is structurally representational, not optimization-side.** Active attack continues on input / target-transform / output-decoder / physics-constraint / encoder-routing / weight-trajectory mechanisms:

**Active 5-axis WSS_z attack:**
1. **H347 nezuko physics priors** — boundary-layer geometric constraint (τ⊥n: shear tangent to surface normal) and kNN smoothness regularization. Training-time physical constraint.
2. **H348 fern curvature features** — per-vertex (H, K, k1, k2) surface curvature as new INPUT channels. WSS_z error may concentrate on high-curvature surfaces (Falkner-Skan).
3. **H349 frieren arcsinh-target** — `arcsinh(τ_z)` regression target. Addresses linear-MSE vs rel_l2 mismatch. **Train-raw catastrophic +100bp**, likely close.
4. **H350 askeladd FiLM decoder** — channel-isolated MLP heads via FiLM conditioning. Decoder capacity for τ_z separated from cp/τ_x/τ_y.
5. **H351 tanjiro NGSB (Normal-Relative Geometric Slice Bias)** — 24-parameter zero-init bias on Transolver's `slice_logits` from surface normals, attacks encoder-resident token-routing mechanism. References GeoTransolver Adams et al. Dec 2025.

**Parallel weight-trajectory probe**:
6. **H352 thorfinn SWA-within-cosine-tail** — averages ~30 fine-grained weight snapshots along the SAME H336 cosine-tail trajectory. Mechanistically distinct from H307 cross-seed soup (closed) and H342 output averaging (in-flight). Zero param overhead. Diagnostic gate via train-raw delta.

**Orthogonal non-WSS supplements:**
7. **H338 edward SP reweight Arm D** — composition test of Arm C EP15 with H336 TTA recipe. Just launched, ETA ~21:30Z.
8. **H342 alphonse multi-checkpoint TTA** — output-space average ep13+ep14+ep15.

**Triangulation logic**: H348 = INPUT; H349 = OUTPUT-TRANSFORM; H350 = OUTPUT-DECODER; H347 = PHYSICS-CONSTRAINT; H351 = ENCODER-ROUTING; H352 = WEIGHT-TRAJECTORY. If any of these break the WSS_z floor, the mechanism is unambiguous. If all six fail, the bottleneck is encoder-feature-extraction itself and the attack moves to deep-tier architecture changes (attention head reweighting, hierarchical attention, GeoTransolver-style geometric cross-attention).

---

## Findings bank highlights (WSS_z-relevant)

| Finding | Implication |
|---|---|
| `wz-reweight-monotone-nogate` | Diagonal training-time per-channel WSS loss reweight is CLOSED — don't assign wy-only, wx-only variants |
| `K5-studentt-superadditive` | K-axis reopens under Student-t ν=4; K=5 is optimal (K=6 ties noise floor per H344) |
| `sigma-axis-closed-nu4` | σ=5e-4 unimodal at ν=4; ±1 step regresses everything; TTA-noise family fully saturated |
| `cal-cannot-rescue-train-raw-regression` | Cal extracts ~7-8bp val; cannot close >10bp raw deficit. Fast-path close rule. |
| `sam-flatness-pessimal-wssz` | H336 basin is already flat enough; SAM perturbations cost 20-50bp on WSS_z. Flatness lever closed. |
| `cal-bootstrap-marginal-test-gain` | Post-hoc cal axis fully saturated at diagonal-affine OLS; no further cal improvements possible |
| `soup-wz-improves-mildly` | Weight-space soup (α=0.85) gave test_WSS_z 8.617 (−0.22bp); signals wz IS the architectural bottleneck |

---

## Next research directions (if 5-axis attack nulls)

If H347/H348/H349/H350/H351/H352 all null, escalate to deeper architecture-tier (plateau protocol):

1. **GeoTransolver-style geometric cross-attention** — cross-attend tokens to a separate geometric encoding of the surface mesh (NVIDIA arxiv:2512.20399 achieves test_WSS ~4.9% on DrivAerML).
2. **Hierarchical multi-scale attention** — coarse-to-fine WSS pyramid with axis-specific heads at fine scales.
3. **Direction-magnitude decomposed WSS loss** — `L = α·cos_sim_loss(τ_pred, τ_true) + β·magnitude_loss(|τ_pred|, |τ_true|)` separating shear direction from magnitude. (Note: passes "loss-side closed" because it changes the loss STRUCTURE, not the per-channel weights.)
4. **Adversarial geometric augmentation** — random surface perturbations to force geometry-invariant flow features.
5. **Hierarchical encoder layer scaling** (LayerScale, CaiT) — gradient-flow regularization for the deep encoder.
