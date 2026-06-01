# SENPAI Research State

**Updated**: 2026-06-01 19:20Z | Branch: `tay` | **SOTA: H342 3-cp output-avg ep13+ep14+ep15 × K=4 TTA — val_cal 5.8962 / test_cal 5.7357 (PR #1526 MERGED)**

---

## Current SOTA (H342 calibrated — PR #1526 MERGED 2026-06-01 19:15Z)

| Model | val_cal | test_cal | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| **H342 3-cp output-avg ep13+ep14+ep15 × K=4 ← CURRENT SOTA** | **5.8962%** | **5.7357%** | **6.6351%** | **3.3751%** | **3.6124%** | 3icmxaqe/qgw0ix77/ijadzof0 |
| H336 K=5+Student-t ν=4+8-res+cal (prior SOTA) | 5.8978% | 5.7379% | 6.6382% | 3.3735% | 3.6133% | 348i3z1v |
| Transolver-3 target | — | — | **< 5.850%** | ≤ 3.421% | ≤ 3.577% | — |

**Merge gate**: val_abupt_calibrated < **5.8962%** AND test_abupt_calibrated < **5.7357%** (tightened after H342 merge)

**WSS gap**: test_WSS_z 8.6122% → 5.85% Transolver target = **0.762pp remaining** (primary obstacle)
**test_SP gap**: 3.6124% vs floor 3.577% = **+3.5bp** (edward H338 targeting this)
**test_VP**: 3.3751% ✓ (safely below 3.421% floor; +1.6bp regression vs H336 — tolerable)

**H342 mechanism**: output-space averaging across 3 nearby cosine-tail checkpoints (ep13+ep14+ep15), each evaluated at K=4+Student-t ν=4 × 8-res × mirror TTA. Largest gains on noisy WSS channels (τ_z −5.3bp, τ_y −4.9bp). Distinct from H307 weight-soup (null); output-space preserves nonlinearities. Finding: `multi-cp-symmetric-best`.

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
| **FiLM-decoder Phase A with random-init head_mlp** | `filmdec-random-init-too-slow-for-frozen-backbone` — randomly initializing the new surface decoder heads (head_mlp 329k params) catastrophically regresses (val_wss_z 12.34% = +316bp, val_abupt 8.94% = +297bp) in 3 epochs at frozen backbone. Volume head preserved confirms diagnosis is heads-only. NOT a FiLM-direction falsification; mechanism = surface heads MUST be warm-started. H354 corrected version (EP13-warm-start) tested. | H350 closed (prior loop) |
| **FiLM-decoder axis FULLY CLOSED (both endpoints)** | `filmdec-axis-fully-closed` + `decoder-pareto-optimal-at-h336-ep13` + `wssz-gap-upstream-not-decoder` — H354 corrected Phase A' with EP13-warm-started head_mlp (step-0 invariant proved MSE=0 to machine precision, ‖w0‖ 10.245→38.737 matches src to float32) STILL regresses val on every metric monotonically as soon as FiLM γβ become non-zero. Trajectory converges toward but never re-crosses EP13. ‖film_proj.weight‖=11.88 with max\|γ\|≈0.69 — heads actively learned strong perturbations and pushed AWAY from h in detrimental direction. H336/EP13's shared surface_out is the Pareto-optimal point under frozen-backbone constraint; any local perturbation traces strictly worse direction. H354 student insight: "H336 raw test_wss_z floor at 8.62% is NOT representational at the decoder — it's bottlenecked upstream." | **H354 closed (this loop)** |

---

## Active Fleet (2026-06-01 19:20Z — 8 students with open PRs)

| PR | Student | Hypothesis | Status | Theme |
|---|---|---|---|---|
| **#1547** | askeladd | **H355: BL derivative decoder (Morgan #1)** — Ghost off-wall probe points at {1e-5, 1e-4, 1e-3, 1e-2}·L_ref along surface normals, cross-attend to Transolver volume tokens to predict u(x+η·n̂), compute τ_w via Richardson finite difference. Auxiliary loss + inference-time OLS blend. Target ~30bp test_WSS. Smoke→Phase1→Phase2 ensemble→Phase3 λ sweep. **Morgan directive #1, never-tried mechanism.** — NEW | 🆕 just assigned | Physical / volume-rooted decoder |
| **#1545** | frieren | H353: Expansive signed_power target on wss_z — direct converse of H349 (closed compressive direction). Arms p∈{1.25, 1.5, 2.0}. Zero param overhead. **Arm A p=1.25 at step 3013 rt 1.50h showing val_abupt 6.155 / val_wssz 9.388 — +18/+22bp regression vs H336 raw, concerning trajectory** (mid-training; cosine-tail tail may recover) | 🟡 WIP — Arm A training (concerning) | Output-transform (expansive direction) |
| **#1544** | thorfinn | H352: SWA-within-cosine-tail — weight-space averaging of ~30 fine-grained snapshots along H336 cosine-tail; distinct from H307 cross-seed soup and H342 output-avg. Zero param overhead. Fresh restart at 17:32Z after student fixed swa_save_from_step. step 1937 rt 0.58h | 🟡 WIP — training (early) | Weight-space averaging (same trajectory) |
| **#1543** | tanjiro | H351: NGSB (Normal-Relative Geometric Slice Bias) — 24-param zero-init `nn.Linear(3 → num_heads)` bias on Transolver slice_logits via surface normals; attacks encoder-resident attention-slice-routing axis. References GeoTransolver Adams et al. Dec 2025. step 4756 rt 1.09h, val_abupt 6.098 val_wssz 9.221 — mid-training | 🟡 WIP — Phase A v3 | Encoder slice-routing |
| **#1539** | fern | **H348: Surface curvature input features — ALIVE CANDIDATE!** Arm A H train tied with H336 (val_abupt 6.0088). **TTA cal eval `26e5khdg` shows val_abupt RAW TTA = 5.9213% (−4.87bp vs H336 raw 5.97)**, val_wss_z 9.0846. If cal extracts usual 7-8bp on val, val_cal lands ~5.84-5.85 — **POTENTIAL SOTA**. Advisor nudge sent for terminal SENPAI-RESULT with val_cal + test_cal. | 🔥 WIP — POTENTIAL SOTA | Input geometry |
| **#1538** | nezuko | H347: Boundary-layer physics priors — **Arm A finished `yainrpxs` train-tied with H336 (val_abupt 6.0075, test_abupt 5.8564)**, Arm B (smooth-only λ_s=1e-4) `zg2o713u` step 4884 rt 1.56h showing val_abupt 6.020 val_wssz 9.196 — ties Arm A. Auto-chains Arm C + TTA cascade | 🟡 WIP — Arm B training | Physical constraint |
| **#1548** | alphonse | **H356: 3-cp output-avg ep13+ep14+ep15 × K=5** — extends H342 merged winner; upgrade per-checkpoint TTA from K=4→K=5. Predicted +0.7-1bp test_cal beyond H342 (val gate < 5.8962, test gate < 5.7357). Pure TTA eval, no training. ETA ~22:00Z (3×7h sequential evals) | 🆕 just assigned | Output-space averaging (K-axis extension) |
| **#1522** | edward | H338: Arm D compositional eval (Arm C SP-reweight EP15 × H336 K=5+ν=4+8-res+mirror TTA recipe) `9t27gag4`, ETA ~21:30Z (~2.5h remaining) | 🟡 WIP — eval | SP floor gap 3.6bp via composition |

**Closed this loop**:
- PR #1524 tanjiro H340 σ-sweep at ν=4: `sigma-axis-closed-nu4` + `per-channel-alpha-sigma-drift`. TTA hyperparameter family fully saturated.
- PR #1528 thorfinn H343 SAM cosine-tail: `sam-flatness-pessimal-wssz` + `sam-monotone-regression-rho` + `cal-cannot-rescue-train-raw-regression`. Flatness-regularization axis closed.
- PR #1540 frieren H349 arcsinh-wss_z target: `arcsinh-wssz-target-pessimal` + `target-transform-direction-falsified-compressive`. +72bp val_wss_z catastrophic. Direction-of-effect of target-transform on heavy-tailed channel is OPPOSITE to compression — H353 frieren testing expansive converse now.
- PR #1542 askeladd H350 FiLM decoder Phase A frozen-backbone: `filmdec-random-init-too-slow-for-frozen-backbone`. Phase A catastrophic regression val_wss_z +316bp, val_abupt +297bp. Mechanism = random-init head_mlp can't recover EP13 in 3 epochs frozen-backbone. NOT a FiLM-direction falsification.
- **PR #1546 askeladd H354 FiLM decoder Phase A' EP13-warm-started**: `filmdec-axis-fully-closed` + `decoder-pareto-optimal-at-h336-ep13` + `wssz-gap-upstream-not-decoder`. Step-0 invariant proven (MSE=0). Yet val regresses monotonically as FiLM γβ become non-zero. ‖film_proj‖=11.88, max\|γ\|≈0.69 — heads chose to leave the warm-started point. **Five closed axes converge on upstream representational bottleneck**.

---

## Current Research Focus: 6-axis WSS_z attack PLUS Morgan-directive #1 BL derivative decoder (loss-reweight + flatness + compressive-target + FiLM-decoder axes ALL closed)

The **primary obstacle** is test_WSS_z = 8.6175% (277bp above Transolver-3's target 5.85%). With H336+: loss-reweight axis TRIPLY closed (H339+H341+H346), flatness/SAM closed (H343), gradient-direction/magnitude closed (H345), TTA hyperparameter family fully saturated (H330+H340+H344), target-transform compressive direction closed (H349), **FiLM-decoder axis FULLY closed (H350+H354)**. **The WSS_z bottleneck is structurally upstream representational, not decoder / not optimization-side.** Six closed axes converge on this conclusion. Active attack continues on input / target-transform-expansive / physics-constraint / encoder-routing / weight-trajectory mechanisms + **NEW: BL derivative decoder (volume-rooted, physically-correct WSS computation)**:

**Active 6-axis WSS_z attack PLUS Morgan #1:**
1. **H348 fern curvature features (POTENTIAL SOTA)** — per-vertex H curvature as new INPUT channel. Arm A train-raw tied with H336. **TTA eval `26e5khdg` shows val_abupt RAW TTA = 5.9213% (−4.87bp vs H336 raw)**, val_wss_z 9.0846. If cal extracts 7-8bp, val_cal ~5.84-5.85 = NEW SOTA. Awaiting student SENPAI-RESULT with cal + test metrics.
2. **H347 nezuko physics priors (ALIVE)** — boundary-layer geometric constraint (τ⊥n: shear tangent to surface normal) and kNN smoothness regularization. Arm A finished train-tied with H336 (test_abupt 5.8564). Arm B smooth-only at step 4884 ties Arm A. Arm C + TTA cascade auto-chaining.
3. **H353 frieren signed_power expansive target (CONCERNING)** — `sign(τ_z)·|τ_z|^p` for p∈{1.25, 1.5, 2.0}. EXPANSIVE direction (direct converse of H349 closed compressive). **Arm A p=1.25 at step 3013 rt 1.50h shows val_abupt 6.155 / val_wssz 9.388 — +18/+22bp regression**, mirroring H349's failure pattern. If terminal holds, expansive direction also closed = target-transform axis fully closed on heavy-tailed channels.
4. **H351 tanjiro NGSB (Normal-Relative Geometric Slice Bias)** — 24-parameter zero-init bias on Transolver's `slice_logits` from surface normals, attacks encoder-resident token-routing. References GeoTransolver Adams et al. Dec 2025. step 4756 rt 1.09h, val_abupt 6.098 mid-training.
5. **H352 thorfinn SWA-within-cosine-tail** — averages ~30 fine-grained weight snapshots along the SAME H336 cosine-tail. Mechanistically distinct from H307 cross-seed soup (closed) and H342 output averaging (in-flight). Zero param overhead. Fresh restart at 17:32Z after swa_save_from_step fix.
6. **H355 askeladd BL derivative decoder (Morgan #1 — NEW)** — **volume-rooted** ghost-probe finite-difference computation of τ_w. Cross-attend off-wall probe locations into Transolver volume tokens, predict u(x+η·n̂), use Richardson FD to derive τ_w = μ·∂u/∂n. Auxiliary loss + inference-time OLS blend with direct head. Target ~30bp test_WSS improvement. Physically-correct decoder structure (replaces direct τ_xyz regression with FD-derived prediction). 8-12h student-time. **Tests upstream/volume mechanism after FiLM decoder axis fully closed.**

**Orthogonal non-WSS supplements:**
7. **H338 edward SP reweight Arm D** — composition test of Arm C EP15 with H336 TTA recipe. Eval `9t27gag4` running, ETA ~21:30Z.
8. **H342 alphonse multi-checkpoint TTA (ALIVE)** — output-space average ep13+ep14+ep15. RAW val_abupt 5.9299 (−4.0bp vs H336 raw). Test_surface 5/6 cases at 17:28Z heartbeat. Eval `3icmxaqe` running.

**Triangulation logic**: H348 = INPUT; H353 = OUTPUT-TRANSFORM (expansive); ~~H354 = OUTPUT-DECODER (warm-start) closed~~; H355 = PHYSICAL-DECODER (volume-rooted, NEW); H347 = PHYSICS-CONSTRAINT; H351 = ENCODER-ROUTING; H352 = WEIGHT-TRAJECTORY. If any of these break the WSS_z floor, the mechanism is unambiguous. If all five remaining axes fail, the bottleneck is encoder-feature-extraction itself and the attack moves to deep-tier architecture changes (attention head reweighting, hierarchical attention, GeoTransolver-style geometric cross-attention, native tangent-basis output head — Morgan directive #2).

**CURRENT STRONG ALIVE CANDIDATES (against new H342 gate: val<5.8962 AND test<5.7357)**:
- **H342 MERGED as SOTA** ✓ — val_cal 5.8962, test_cal 5.7357
- **H348 fern**: val_abupt RAW TTA = 5.9213% (better than H336 raw 5.97 by 4.87bp). Cal yield ~7-8bp → val_cal ~5.843-5.849. Easily beats H342 gate. **LEADING CANDIDATE — test_cal ETA ~22:00Z.**
- **H356 alphonse** (new): K=5 version of H342 — predicted val_cal ~5.889, test_cal ~5.728. Pure eval, no training.

Crucially, H348 (curvature input = training-time improvement) and H342/H356 (output-avg = TTA mechanism) are STRUCTURALLY DISTINCT and COMPOSABLE. If H348 lands as SOTA, next step is H342+K=5 applied to the H348-trained model. The compounding potential is significant.

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
