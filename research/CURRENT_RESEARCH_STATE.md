# SENPAI Research State

**Updated**: 2026-06-01 15:15Z | Branch: `tay` | **SOTA: H336 K=5+Student-t ν=4+8-res+mirror+cal — val_cal 5.8978 / test_cal 5.7379**

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
| Per-channel cal (affine) | OLS-MLE is the global optimum at val_N=34 | H316, H319, H323, H328, H329, H332, H333, H334 |
| Weight-space soup | α-soup axis exhausted; val gate irreducible via cross-seed averaging | H307 |
| WSS loss reweight (uniform) | `wz-reweight-monotone-nogate` — H339 closed. All channels regress uniformly | H339 closed |
| WSS loss reweight (wz-only) | `wz-reweight-monotone-nogate` — H341 CLOSED. Even isolated wz reweight regresses all channels | H341 closed |
| K-axis continuation at ν=4 | `K6-vs-K5-noise-floor-tie` — K=6 ties K=5 at ±0.02bp noise floor; K-axis saturated at K=5 | H344 closed |
| Gradient-direction conflict | `pcgrad-no-conflict-falsifies` — mean cos(g_wss, g_pres) = +0.30, 99.55% positive. PCGrad would be a no-op. | H345 closed |
| WSS gradient magnitude lever | `wss-gradient-already-dominant` — |g_wss|/|g_pres| = 5.12 at H336 base. Loss-weight up/down both unproductive. | H345 closed |
| WSS_z per-batch focal weighting | `focal-batch-collapsing` — γ∈{2.0, 1.0} both trip 50× collapse rule; even identity exponent collapses to 1-2 vertices on 88.5% of batches | H346 CLOSED this loop |
| **WSS_z loss-reweight axis (combined)** | `wss-z-loss-weight-axis-closed` — H339 uniform + H341 wz-only + H346 per-vertex focal all fail. **TRIPLY CLOSED.** WSS_z floor at 8.62% test_cal is NOT addressable by any loss-reweight scheme at the EP13 cosine-tail base — bottleneck is representational, not gradient-budget | H346 CLOSED this loop |

---

## Active Fleet (2026-06-01 15:15Z — 8 students with open PRs)

| PR | Student | Hypothesis | Status | Theme |
|---|---|---|---|---|
| **#1542** | askeladd | H350: Channel-isolated FiLM-conditioned decoder (Phase A frozen-backbone diagnostic gate, then Phase B full finetune) — NEW | 🆕 just assigned | Architecture / per-channel decoder capacity |
| **#1540** | frieren | H349: arcsinh target transform on WSS_z (3 arms: wss_z, wss, all) | 🟡 WIP — training | Target-space geometry |
| **#1539** | fern | H348: Surface curvature input features (H, K, k1k2) — Arm A H-only running at 7944 steps, ETA ~16:30Z | 🟡 WIP — training | Input geometry |
| **#1538** | nezuko | H347: Boundary-layer physics priors (τ⊥n normals + kNN smoothness) — running 1.49h | 🟡 WIP — training | Physical constraint |
| **#1528** | thorfinn | H343: SAM cosine-tail Arm B TTA eval (`s2ap2tb6`, 2.77h in, near-terminal). Train-raw +14.1bp val_abupt regression — close-likely | 🟡 WIP — eval near-terminal | Optimizer curvature |
| **#1526** | alphonse | H342: Multi-checkpoint output averaging (ep13+ep14+ep15 TTA) | 🟡 WIP | Output-space averaging |
| **#1524** | tanjiro | H340: σ-sweep at ν=4 (arms σ∈{2.5e-4, 5e-4 ref, 1e-3}) | 🟡 WIP | Noise scale at Student-t |
| **#1522** | edward | H338: Arm D compositional eval (Arm C SP-reweight EP15 × H336 K=5+ν=4+8-res+mirror TTA recipe), just launched `9t27gag4`, ETA ~21:30Z | 🟡 WIP — eval | SP floor gap 3.6bp via composition |

**Closed this loop (PR #1537, askeladd H346)**: per-batch focal WSS_z weighting collapsed at γ∈{2.0, 1.0}; banked `focal-batch-collapsing` and the combined finding `wss-z-loss-weight-axis-closed` (H339+H341+H346 triply close the loss-reweight axis).

---

## Current Research Focus: 4-axis WSS_z attack on orthogonal mechanisms (loss-reweight axis TRIPLY CLOSED)

The **primary obstacle** is test_WSS_z = 8.6175% (277bp above Transolver-3's target 5.85%). **The loss-reweight axis is now triply closed** (H339 uniform + H341 wz-only + H346 per-vertex focal — all banked under combined finding `wss-z-loss-weight-axis-closed`). Combined with H345's gradient-direction and gradient-magnitude null findings, every diagonal training-time loss/gradient lever has been exhausted. **The WSS_z bottleneck is structurally representational, not a gradient/loss-budget problem.** The 4-axis attack continues on input / target-transform / output-decoder / physics-constraint mechanisms:

**4-axis parallel WSS_z attack (in-flight):**
1. **H347 nezuko physics priors** — geometric constraint (τ⊥n: shear tangent to surface normal) and kNN smoothness regularization. Boundary-layer physics injected as training-time constraint.
2. **H348 fern curvature features** — adds per-vertex (H, K, k1, k2) surface curvature as new INPUT channels. WSS_z error may concentrate on high-curvature surfaces; Falkner-Skan boundary layer scales with curvature.
3. **H349 frieren arcsinh-target** — transforms regression target via `arcsinh(τ_z)`. Compresses dynamic range; addresses mismatch between linear-MSE training (over-weights high-|τ_z|) and rel_l2 eval (over-weights low-|τ_z|).
4. **H350 askeladd FiLM decoder** — replaces shared `Linear(d_model→4)` output projection with 4 axis-conditioned MLP heads via FiLM (Perez et al. 2018). Gives τ_z its own decoder capacity, separated from cp/τ_x/τ_y. Phase-A frozen-backbone diagnostic at 6h GPU gates Phase-B 20-24h finetune.

The triangulation: H348 attacks the INPUT side, H349 attacks the OUTPUT-TRANSFORM side, H350 attacks the OUTPUT-DECODER side, H347 attacks the PHYSICS-CONSTRAINT side. If any of these break the WSS_z floor, the mechanism will be unambiguous. If all four fail, the WSS_z bottleneck is encoder-resident (i.e. the deep feature extraction doesn't represent τ_z well from inputs alone) and the attack moves to encoder-architecture revisions (token mixing, geometric attention, etc.).

**Orthogonal non-WSS supplements:**
5. **H338 edward SP reweight Arm D** — Arm C train (5.7345 test < gate but 5.9001 val > gate) was asymmetric. Arm D evaluates Arm C EP15 with H336's K=5+ν=4+8-res+mirror+cal TTA recipe (composition test). Just launched, ~6.8h, ETA ~21:30Z. If val_cal < 5.8978 → new SOTA; if not → close.
6. **H340 tanjiro σ-sweep** — optimal noise sigma at ν=4, may compound with K=5 finding.
7. **H342 alphonse multi-checkpoint TTA** — output-space average of ep13+ep14+ep15.
8. **H343 thorfinn SAM** — Arm B train raw +14.1bp val_abupt regression. TTA eval `s2ap2tb6` near-terminal. Even with TTA stack, +14bp raw deficit is unlikely to invert into a SOTA. Close-pending result.

---

## Findings bank highlights (WSS_z-relevant)

| Finding | Implication |
|---|---|
| `wz-reweight-monotone-nogate` | Diagonal training-time per-channel WSS loss reweight is CLOSED — don't assign wy-only, wx-only variants |
| `K5-studentt-superadditive` | K-axis reopens under Student-t ν=4; K=5 is optimal (K=6 ties noise floor per H344) |
| `cal-bootstrap-marginal-test-gain` | Post-hoc cal axis fully saturated at diagonal-affine OLS; no further cal improvements possible |
| `soup-wz-improves-mildly` | Weight-space soup (α=0.85) gave test_WSS_z 8.617 (−0.22bp); signals wz IS the architectural bottleneck |

---

## Next research directions (if current fleet nulls)

If H346/H347/H348/H349 all null, escalate to architecture-tier (plateau protocol ≥5 consecutive nulls since H336):

1. **Per-axis decoder head for WSS** — channel-specific attention/MLP branch conditioned on axis label {x, y, z}. Gives the model dedicated capacity for z-axis shear without sharing parameters with pressure.
2. **Anisotropic neighborhood sampling** — bias kNN toward streamwise vehicle axis direction where τ_z concentrates. Fundamentally changes the receptive field geometry.
3. **Multi-scale WSS prediction head** — coarse-to-fine WSS pyramid; higher scales capture global boundary layer, fine scale refines high-curvature regions.
4. **Direction-magnitude decomposed loss** — `L = α·cos_sim_loss(τ_pred, τ_true) + β·magnitude_loss(|τ_pred|, |τ_true|)`. Separates the geometry of shear direction from magnitude.
5. **Adversarial geometric augmentation** — augment with random surface perturbations to force the model to learn geometry-invariant flow features.
