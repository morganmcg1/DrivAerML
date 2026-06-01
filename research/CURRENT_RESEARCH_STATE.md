# SENPAI Research State

**Updated**: 2026-06-01 22:52Z | Branch: `tay` | **SOTA: H342 3-cp output-avg ep13+ep14+ep15 × K=4 TTA — val_cal 5.8962 / test_cal 5.7357 (PR #1526 MERGED)**

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
| **Surface curvature input features (H, K, k1, k2)** | `curvature-features-null` — per-vertex H curvature adds 0 signal: val_cal +0.06bp miss, test_WSS_z +0.07bp WORSE. Sub-bp null on every channel under calibration. WSS_z floor is robust to LOCAL geometric shape information. Global geodesic structure (H360 LapPE-32) still open. | H348 closed |
| **WSS_z loss-reweight axis (combined)** | `wss-z-loss-weight-axis-closed` — H339 uniform + H341 wz-only + H346 per-vertex focal all fail. **TRIPLY CLOSED.** WSS_z floor at 8.62% test_cal is NOT addressable by any loss-reweight scheme at the EP13 cosine-tail base — bottleneck is representational, not gradient-budget | H346 closed |
| **SAM cosine-tail at any ρ** | `sam-flatness-pessimal-wssz` + `sam-monotone-regression-rho` — ρ∈{0.05, 0.02, 0.002} all regress; WSS_z pessimal (+50bp@ρ=0.002). H336 basin is already flat enough; SAM perturbations push OUT of the WSS_z local fit. Flatness-regularization axis closed. | **H343 closed (this loop)** |
| **FiLM-decoder Phase A with random-init head_mlp** | `filmdec-random-init-too-slow-for-frozen-backbone` — randomly initializing the new surface decoder heads (head_mlp 329k params) catastrophically regresses (val_wss_z 12.34% = +316bp, val_abupt 8.94% = +297bp) in 3 epochs at frozen backbone. Volume head preserved confirms diagnosis is heads-only. NOT a FiLM-direction falsification; mechanism = surface heads MUST be warm-started. H354 corrected version (EP13-warm-start) tested. | H350 closed (prior loop) |
| **FiLM-decoder axis FULLY CLOSED (both endpoints)** | `filmdec-axis-fully-closed` + `decoder-pareto-optimal-at-h336-ep13` + `wssz-gap-upstream-not-decoder` — H354 corrected Phase A' with EP13-warm-started head_mlp (step-0 invariant proved MSE=0 to machine precision, ‖w0‖ 10.245→38.737 matches src to float32) STILL regresses val on every metric monotonically as soon as FiLM γβ become non-zero. Trajectory converges toward but never re-crosses EP13. ‖film_proj.weight‖=11.88 with max\|γ\|≈0.69 — heads actively learned strong perturbations and pushed AWAY from h in detrimental direction. H336/EP13's shared surface_out is the Pareto-optimal point under frozen-backbone constraint; any local perturbation traces strictly worse direction. H354 student insight: "H336 raw test_wss_z floor at 8.62% is NOT representational at the decoder — it's bottlenecked upstream." | **H354 closed (this loop)** |

---

## Active Fleet (2026-06-01 19:45Z — 8 students with open PRs)

| PR | Student | Hypothesis | Status | Theme |
|---|---|---|---|---|
| **#1551** | askeladd | **H359: Multi-scale local surface kNN branch (FRESH — Morgan directive #3 proxy)** — MultiScaleSurfaceAggregation: kNN at k=8,16,32 on surface point cloud → mean-pool per scale → project → zero-init residual added to surface tokens BEFORE Transolver slicing. ~200k params (0.4%). Smoke → Phase 1 EP14-16 from EP13. Encoder-side attack on WSS_z bottleneck diagnosed by 8 closed decoder-side axes. **PR #1551 just assigned.** | 🆕 just assigned | Encoder representation (multi-scale local geometry) |
| **#1550** | frieren | **H358: Native tangent-basis residual output head (FRESH)** — TangentResidualHead predicts (τ_t1, τ_t2) in local tangent basis, rotates to xyz, added to existing 3-channel surface_out. Zero-init output → step-0 invariant. Tests OUTPUT BASIS axis (untouched). Physics constraint: τ·n=0 by no-slip wall BC. Fine-tune 3 epochs from EP13. **PR #1550 just assigned.** | 🆕 just assigned | Output basis (physics-correct τ·n=0) |
| **#1544** | thorfinn | H352: SWA-within-cosine-tail — fresh restart at 17:32Z. Heartbeat sent 19:35Z. ETA terminal ~20:15Z. Gate tightened to val_cal < 5.8962 (H342 SOTA). | 🟡 WIP — training (ETA ~20:15Z) | Weight-space averaging (same trajectory) |
| **#1549** | tanjiro | **H357: GeoTransolver geometric content embedding (FRESH)** — GeometricContentNet: [nx,ny,nz,log_area]→64→d_model, zero-init output, added to surface token content x BEFORE Transolver slicing. Warm-start from H185 EP15 + fine-tune 3 cosine-tail epochs. Directly tests CONTENT PATH (distinct from H351's routing path). **PR #1549 just assigned.** | 🆕 just assigned | Encoder content (geometric inductive bias) |
| **#1552** | fern | **H360: Surface Laplacian eigenfunction PE (LapPE-32) — FRESH ASSIGNED 22:52Z.** Global geodesic surface structure via lowest 32 eigenfunctions of surface graph Laplacian. Orthogonal to H348 curvature (local shape) and H359 kNN (local multi-scale). Tests whether WSS_z floor is a GLOBAL information-deficit problem. Smoke → Phase 1 EP13→EP16 cosine-tail → TTA+cal eval. Pre-committed close: val_abupt RAW > 6.10% → close early. | 🆕 just assigned | Encoder input (global geodesic structure) |
| **#1538** | nezuko | H347: BL physics priors — **Arm B smooth-only `zg2o713u` DONE: test_abupt_raw 5.8500 (−0.64bp vs Arm A).** Arm C (both priors λ_n+λ_s) launched 19:33Z, ETA terminal ~22:07Z. TTA cascade starts after Arm C. | 🟡 WIP — Arm C training | Physical constraint |
| **#1548** | alphonse | **H356: 3-cp output-avg × K=5** — ep15 K=5 eval `0n1xkwic` active, ep14+ep13 to follow sequentially. Total ETA ~22:00-04:00Z. | 🟡 WIP — ep15 K=5 eval | Output-space averaging (K-axis extension) |
| **#1522** | edward | H338: Arm D compositional eval `9t27gag4`. ETA ~21:30Z. | 🟡 WIP — eval | SP floor gap 3.6bp via composition |

**Closed this loop**:
- PR #1524 tanjiro H340 σ-sweep at ν=4: `sigma-axis-closed-nu4` + `per-channel-alpha-sigma-drift`. TTA hyperparameter family fully saturated.
- PR #1528 thorfinn H343 SAM cosine-tail: `sam-flatness-pessimal-wssz` + `sam-monotone-regression-rho` + `cal-cannot-rescue-train-raw-regression`. Flatness-regularization axis closed.
- PR #1540 frieren H349 arcsinh-wss_z target: `arcsinh-wssz-target-pessimal` + `target-transform-direction-falsified-compressive`. +72bp val_wss_z catastrophic. H353 testing expansive converse.
- PR #1542 askeladd H350 FiLM decoder Phase A: `filmdec-random-init-too-slow-for-frozen-backbone`. Catastrophic regression val_wss_z +316bp.
- PR #1546 askeladd H354 FiLM decoder Phase A' EP13-warm-started: `filmdec-axis-fully-closed` + `decoder-pareto-optimal-at-h336-ep13` + `wssz-gap-upstream-not-decoder`. **FiLM-decoder axis fully closed.**
- **PR #1543 tanjiro H351 NGSB**: `ngsb-normal-only-routing-pessimal` + `encoder-routing-axis-coarse-normals-falsified`. +69.8bp val abupt monotone regression; WSS_z no improvement at any epoch. **Routing path via normals is pessimal; H357 tests CONTENT path.**
- **PR #1545 frieren H353 signed_power expansive**: `target-transform-axis-closed-wssz` + `signed-power-expansive-also-pessimal` + `wss-z-residuals-not-heavy-tail-loss-problem`. Arm A p=1.25 +15bp val_abupt/+15bp val_wss_z. Arms B/C skipped. **Both directions of target-transform now closed — 7th converging axis. Loss-shape is NOT the lever; representation IS.**
- **PR #1547 askeladd H355 BL derivative decoder (Morgan #1 FALSIFIED)**: `bl-derivative-decoder-aux-neutral`. val_wss_z +0.4bp (neutral), val_abupt +4.2bp. Close rule triggered (wss_z improvement < 5bp). BL head trains in isolation, not transferring to surface decoder. **8th converging closed decoder-side axis — Plateau Protocol escalation: pivot fully to encoder/input-side attacks.**
- **PR #1539 fern H348 curvature input features (22:52Z)**: `curvature-features-null`. val_cal +0.06bp (MISS by 0.06bp), test_WSS_z_cal +0.07bp (WORSE). All deltas sub-bp. 3-arm chain killed after Arm A null — Arms B (K) and C (k1k2) skipped. **9th converging closed axis: WSS_z floor is robust to local geometric shape information (curvature cannot help).** H360 LapPE-32 (global geodesic structure) assigned.

---

## Current Research Focus: ENCODER/INPUT-SIDE ATTACK — 8 decoder-side axes fully closed (Plateau Protocol escalated)

The **primary obstacle** is test_WSS_z = 8.6175% (277bp above Transolver-3's target 5.85%). **Eight consecutive decoder-side experiments now closed with the same signature**: loss-reweight (H339+H341+H346), TTA-saturation (H330+H340+H344), SAM (H343), target-transform (H349+H353), FiLM-decoder (H350+H354), NGSB encoder-routing (H351), BL-derivative-decoder (H355). **Plateau Protocol escalation in effect: the WSS_z bottleneck is structurally upstream representational — the volume backbone is the bottleneck, not the decoder.** All active experiments now target input, encoder, or output-basis architectural changes that operate BEFORE or independently of the frozen-backbone slice-pool. H357/H358 also test encoder-content and output-basis respectively.

**Active encoder/input-side WSS_z attack (9 input/decoder-side axes closed, Plateau Protocol escalated):**
1. **~~H348 fern curvature input~~ CLOSED → H360 fern LapPE-32 (FRESH 22:52Z)** — H348 closed (`curvature-features-null`): val_cal +0.06bp miss, test_WSS_z +0.07bp WORSE. 9th converging closed axis. H360 pivots to GLOBAL geodesic structure: 32 surface graph Laplacian eigenfunctions precomputed on canonical mesh surface, concatenated as input. Orthogonal to curvature (local) and kNN (local multi-scale). **PR #1552.**
2. **H347 nezuko physics priors (ALIVE)** — boundary-layer geometric constraint (τ⊥n: shear tangent to surface normal) and kNN smoothness regularization. Arm A finished train-tied with H336 (test_abupt 5.8564). Arm B smooth-only ties Arm A. Arm C + TTA cascade auto-chaining.
3. **~~H353 frieren signed_power expansive~~ CLOSED → H358 frieren tangent-basis residual head (FRESH, Phase 1 nearly complete)** — H353 closed (`target-transform-axis-closed-wssz`). H358 pivots to OUTPUT BASIS axis (never touched): TangentResidualHead predicts (τ_t1, τ_t2) in per-point local tangent basis, projects to global xyz, added to existing 3-channel head. 7/8 DDP ranks finished Phase 1 at step 8162, rank-0 finishing (~22:45Z). **PR #1550.**
4. **~~H351 tanjiro NGSB~~ CLOSED → H357 tanjiro GeoTransolver content embedding (FRESH)** — H351 closed (`ngsb-normal-only-routing-pessimal`). H357 replaces routing path with CONTENT path: GeometricContentNet([nx,ny,nz,log_area])→64→d_model, zero-init output, added to surface token `x` before Transolver slicing. Warm-start EP15, fine-tune 3 cosine-tail epochs. **PR #1549.**
5. **H352 thorfinn SWA-within-cosine-tail** — averages ~30 fine-grained weight snapshots along the SAME H336 cosine-tail. Mechanistically distinct from H307 cross-seed soup (closed) and H342 output averaging (merged). Zero param overhead. ETA Arm B TTA cal ~04:25Z June 2.
6. **~~H355 askeladd BL derivative decoder CLOSED~~ → H359 askeladd Multi-scale surface kNN branch (FRESH)** — H355 closed (`bl-derivative-decoder-aux-neutral`). H359 attacks the ENCODER directly: MultiScaleSurfaceAggregation kNN at k=8,16,32 on surface point cloud → mean-pool per scale → project → zero-init residual added to surface tokens BEFORE Transolver slicing. ~200k params (0.4%). **PR #1551.**

**Orthogonal non-WSS supplements:**
7. **H338 edward SP reweight Arm D** — composition test of Arm C EP15 with H336 TTA recipe. Eval `9t27gag4` running (runtime 12492s at 22:43Z, no cal metrics yet). ETA ~23:00Z.
8. **H356 alphonse 3-cp output-avg × K=5** — pure eval, no training. K=5 version of H342. ep15 eval `0n1xkwic` active, ep14+ep13 to follow. Predicted val_cal ~5.889, test_cal ~5.728.

**Triangulation logic**: ~~H348 = INPUT (curvature CLOSED)~~; H360 = INPUT (global LapPE FRESH); ~~H353 = OUTPUT-TRANSFORM CLOSED~~; ~~H354 = OUTPUT-DECODER CLOSED~~; ~~H355 = PHYSICAL-DECODER CLOSED~~; H347 = PHYSICS-CONSTRAINT; ~~H351 = ENCODER-ROUTING CLOSED~~; H357 = ENCODER-CONTENT; H358 = OUTPUT-BASIS; H352 = WEIGHT-TRAJECTORY; H359 = ENCODER-MULTI-SCALE-LOCAL; H360 = ENCODER-INPUT-GLOBAL-SPECTRAL. If all null, escalate to Physics-regime MoE (Morgan P4) or hierarchical attention.

**CURRENT CANDIDATES (against H342 gate: val<5.8962 AND test<5.7357)**:
- **H342 MERGED as SOTA** ✓ — val_cal 5.8962, test_cal 5.7357
- **H356 alphonse**: K=5 version of H342 — predicted val_cal ~5.889, test_cal ~5.728 (~several bp improvement if K-axis still alive at K=5 over K=4). Pure eval.
- H358 frieren: Phase 1 nearly complete, cal eval to follow when rank-0 lands.

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
