# SENPAI Research State

**Updated**: 2026-06-01 17:00Z | Branch: `tay` | **SOTA: H336 K=5+Student-t ν=4+8-res+mirror+cal — val_cal 5.8978 / test_cal 5.7379**

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
| **FiLM-decoder Phase A with random-init head_mlp** | `filmdec-random-init-too-slow-for-frozen-backbone` — randomly initializing the new surface decoder heads (head_mlp 329k params) catastrophically regresses (val_wss_z 12.34% = +316bp, val_abupt 8.94% = +297bp) in 3 epochs at frozen backbone. Volume head preserved confirms diagnosis is heads-only. NOT a FiLM-direction falsification; mechanism = surface heads MUST be warm-started. H354 corrected version (EP13-warm-start) assigned. | **H350 closed (this loop)** |

---

## Active Fleet (2026-06-01 17:00Z — 8 students with open PRs)

| PR | Student | Hypothesis | Status | Theme |
|---|---|---|---|---|
| **#1546** | askeladd | **H354: Corrected FiLM decoder (EP13-warm-started head_mlp)** — replacement for closed H350. Same FiLM γβ + axis_embed architecture but `head_mlp` init from EP13 `surface_out` (architectures match), γ=β=0 so step-0 == EP13 baseline. Phase A' gate val_wss_z<9.13%. — NEW | 🆕 just assigned | Architecture / per-channel decoder (warm-start) |
| **#1545** | frieren | H353: Expansive signed_power target on wss_z — direct converse of H349 (closed compressive direction). Arms p∈{1.25, 1.5, 2.0}. Zero param overhead. 8 ranks Arm A p=1.25 running (rt 0.21h), ETA terminal ~19:40Z | 🟡 WIP — Arm A training | Output-transform (expansive direction) |
| **#1544** | thorfinn | H352: SWA-within-cosine-tail — weight-space averaging of ~30 fine-grained snapshots along H336 cosine-tail; distinct from H307 cross-seed soup and H342 output-avg. Zero param overhead. 8 ranks running (rt 0.48h), ETA ~19:30Z | 🟡 WIP — training | Weight-space averaging (same trajectory) |
| **#1543** | tanjiro | H351: NGSB (Normal-Relative Geometric Slice Bias) — 24-param zero-init `nn.Linear(3 → num_heads)` bias on Transolver slice_logits via surface normals; attacks encoder-resident attention-slice-routing axis. References GeoTransolver Adams et al. Dec 2025. 8 ranks Phase A diagnostic running | 🟡 WIP — Phase A | Encoder slice-routing |
| **#1539** | fern | H348: Surface curvature input features — Arm A H train terminal raw val_abupt **6.0088** (essentially TIED with H336 raw). **TTA cal eval running** (`26e5khdg` K=4 antithetic + ν=4 + σ=5e-4 + 8-res + mirror + OLS cal on EP16 EMA), ETA ~20:30Z. Awaiting terminal SENPAI-RESULT | 🟡 WIP — TTA eval | Input geometry |
| **#1538** | nezuko | H347: Boundary-layer physics priors — **Arm A finished `yainrpxs` train-tied with H336 (val_abupt 6.0075, test_abupt 5.8564)**, Arm B (smooth-only λ_s=1e-4) running step ~980 rt 0.31h, auto-chains to Arm C and TTA+cal cascade. Acknowledged auto-chain at 16:55Z | 🟡 WIP — Arm B training | Physical constraint |
| **#1526** | alphonse | H342: Multi-checkpoint output averaging (ep13+ep14+ep15 TTA) — `3icmxaqe` running eval | 🟡 WIP — eval | Output-space averaging |
| **#1522** | edward | H338: Arm D compositional eval (Arm C SP-reweight EP15 × H336 K=5+ν=4+8-res+mirror TTA recipe) `9t27gag4`, ETA ~21:30Z | 🟡 WIP — eval | SP floor gap 3.6bp via composition |

**Closed this loop**:
- PR #1524 tanjiro H340 σ-sweep at ν=4: `sigma-axis-closed-nu4` + `per-channel-alpha-sigma-drift`. TTA hyperparameter family fully saturated.
- PR #1528 thorfinn H343 SAM cosine-tail: `sam-flatness-pessimal-wssz` + `sam-monotone-regression-rho` + `cal-cannot-rescue-train-raw-regression`. Flatness-regularization axis closed.
- PR #1540 frieren H349 arcsinh-wss_z target: `arcsinh-wssz-target-pessimal` + `target-transform-direction-falsified-compressive`. +72bp val_wss_z catastrophic. Direction-of-effect of target-transform on heavy-tailed channel is OPPOSITE to compression — H353 frieren testing expansive converse now.
- PR #1542 askeladd H350 FiLM decoder Phase A frozen-backbone: `filmdec-random-init-too-slow-for-frozen-backbone`. Phase A catastrophic regression val_wss_z +316bp, val_abupt +297bp. Mechanism = random-init head_mlp can't recover EP13 in 3 epochs frozen-backbone. NOT a FiLM-direction falsification; H354 corrected version (EP13-warm-start) assigned next.

---

## Current Research Focus: 6-axis WSS_z attack on orthogonal mechanisms (loss-reweight + flatness + compressive-target axes closed)

The **primary obstacle** is test_WSS_z = 8.6175% (277bp above Transolver-3's target 5.85%). With H336+: loss-reweight axis TRIPLY closed (H339+H341+H346), flatness/SAM closed (H343), gradient-direction/magnitude closed (H345), TTA hyperparameter family fully saturated (H330+H340+H344), target-transform compressive direction closed (H349). **The WSS_z bottleneck is structurally representational, not optimization-side.** Active attack continues on input / target-transform-expansive / output-decoder / physics-constraint / encoder-routing / weight-trajectory mechanisms:

**Active 6-axis WSS_z attack:**
1. **H347 nezuko physics priors (ALIVE)** — boundary-layer geometric constraint (τ⊥n: shear tangent to surface normal) and kNN smoothness regularization. Arm A train-raw essentially tied with H336. Arms B/C and TTA cascade auto-chaining now.
2. **H348 fern curvature features (ALIVE)** — per-vertex H curvature as new INPUT channel. Arm A train-raw essentially tied with H336. TTA+cal eval `26e5khdg` running, ETA ~20:30Z.
3. **H353 frieren signed_power expansive target (NEW)** — `sign(τ_z)·|τ_z|^p` for p∈{1.25, 1.5, 2.0}. EXPANSIVE direction (direct converse of H349 which closed compressive). Heavy-tailed channel needs MORE penalty on extremes, not less. Arm A p=1.25 running ETA ~19:40Z.
4. **H354 askeladd FiLM decoder warm-started (NEW)** — corrected Phase A' with `head_mlp` warm-started from EP13's `surface_out`. Step-0 == EP13 baseline. Tests whether channel-specific FiLM modulation helps once heads are properly initialized.
5. **H351 tanjiro NGSB (Normal-Relative Geometric Slice Bias)** — 24-parameter zero-init bias on Transolver's `slice_logits` from surface normals, attacks encoder-resident token-routing. References GeoTransolver Adams et al. Dec 2025. Phase A diagnostic running.
6. **H352 thorfinn SWA-within-cosine-tail** — averages ~30 fine-grained weight snapshots along the SAME H336 cosine-tail. Mechanistically distinct from H307 cross-seed soup (closed) and H342 output averaging (in-flight). Zero param overhead. Diagnostic gate via train-raw delta.

**Orthogonal non-WSS supplements:**
7. **H338 edward SP reweight Arm D** — composition test of Arm C EP15 with H336 TTA recipe. Eval `9t27gag4` running, ETA ~21:30Z.
8. **H342 alphonse multi-checkpoint TTA** — output-space average ep13+ep14+ep15. Eval `3icmxaqe` running.

**Triangulation logic**: H348 = INPUT; H353 = OUTPUT-TRANSFORM (expansive); H354 = OUTPUT-DECODER (warm-start); H347 = PHYSICS-CONSTRAINT; H351 = ENCODER-ROUTING; H352 = WEIGHT-TRAJECTORY. If any of these break the WSS_z floor, the mechanism is unambiguous. If all six fail, the bottleneck is encoder-feature-extraction itself and the attack moves to deep-tier architecture changes (attention head reweighting, hierarchical attention, GeoTransolver-style geometric cross-attention).

**Two ALIVE candidates this loop**: H347 nezuko and H348 fern both produced train-raw tied with H336 baseline (the cleanest "ambiguous-alive" signal so far). The TTA+cal eval is the discriminator — if either lands val_cal<5.8978 AND test_cal<5.7379, it's the next SOTA. If both extract only the usual 7-8bp cal yield, `cal-cannot-rescue-train-raw-regression` triggers and both close cleanly.

### Morgan directive queue (Issue #1056, 2026-06-01 13:15Z)

Highest-priority untried mechanisms to assign as students idle:

1. **BL derivative decoder (P1, NEVER TRIED, target ~30bp test_WSS)** — Ghost off-wall probe points at {η_1,...,η_K}×n̂ (e.g., η ∈ {1e-5, 1e-4, 1e-3, 1e-2}·L_ref normal to surface), cross-attend to Transolver token cache to predict velocity at each probe, compute WSS via differentiable finite difference τ ≈ μ·(u(η_1) − 0)/η_1. Reconstructs τ_xyz from the velocity gradient structure the model never sees directly. Implementation: new decoder branch with ghost-point sampling + cross-attention head + FD differentiation layer. Estimated student-time: 8-12h. **Queue for next idle student** (likely askeladd if H354 Phase A' fails ~18:00Z).
2. **Native tangent-basis output head (P2, simpler)** — Predict (τ_t1, τ_t2) in local tangent frame, reconstruct τ_xyz = τ_t1·t̂₁ + τ_t2·t̂₂. Enforces physical tangentiality (τ⊥n) by construction. Compared to H347 normals-loss-as-regularizer, this is by-construction not via gradient pressure. Estimated student-time: 4-6h.
3. **Multi-scale local surface manifold branch (P3, after H347/H348 land)** — Multi-resolution geometric context (k1k2 at multiple smoothing scales).
4. **Physics-regime MoE (P4)** — WSS-head splits across {attached BL, APG-separation, wheel/underbody, sharp-edge/wake, smooth-panel} experts with entropy-regularized router.

Drafts ready to assign on next idle slot.

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
