# SENPAI Research State

**Updated**: 2026-06-01 12:15Z | Branch: `tay` | **SOTA: H336 K=5+Student-t ν=4+8-res+mirror+cal — val_cal 5.8978 / test_cal 5.7379**

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
| Gradient-direction conflict | `pcgrad-no-conflict-falsifies` — mean cos(g_wss, g_pres) = +0.30, 99.55% positive. PCGrad would be a no-op. | H345 CLOSED this loop |
| WSS gradient magnitude lever | `wss-gradient-already-dominant` — |g_wss|/|g_pres| = 5.12 at H336 base. Loss-weight up/down both unproductive. | H345 CLOSED this loop |

---

## Active Fleet (2026-06-01 12:15Z — 8 students with open PRs)

| PR | Student | Hypothesis | Status | Theme |
|---|---|---|---|---|
| **#1540** | frieren | H349: arcsinh target transform on WSS_z (3 arms: wss_z, wss, all) — NEW | 🆕 just assigned | Target-space geometry |
| **#1537** | askeladd | H346: Per-batch focal weighting on WSS_z residuals (Option A) (γ=1,2,3) | 🟡 WIP (revised) | Spatial concentration |
| **#1538** | nezuko | H347: Boundary-layer physics priors (τ⊥n normals + kNN smoothness) | 🟡 WIP | Physical constraint |
| **#1528** | thorfinn | H343: SAM cosine-tail (ρ∈{0.05, 0.02}, 2-arm chained) | 🟡 WIP | Optimizer curvature |
| **#1526** | alphonse | H342: Multi-checkpoint output averaging (ep13+ep14+ep15 TTA) | 🟡 WIP | Output-space averaging |
| **#1524** | tanjiro | H340: σ-sweep at ν=4 (arms σ∈{2.5e-4, 5e-4 ref, 1e-3}) | 🟡 WIP | Noise scale at Student-t |
| **#1522** | edward | H338: SP-targeted loss reweight cosine-tail ({1.05, 1.10, 1.20}) | 🟡 WIP | SP floor gap 3.6bp |
| **#1539** | fern | H348: Surface curvature input features (H, K, k1k2) | 🟡 WIP | Input geometry |

**Closed this loop**: PR #1536 frieren H345 — `pcgrad-no-conflict-falsifies` + `wss-gradient-already-dominant`. Diagnostic gate hit (mean cos(g_wss, g_pres) = +0.30, 99.55% positive); Arms B/C skipped saving ~30h GPU.

---

## Current Research Focus: 4-axis WSS_z attack on orthogonal mechanisms

The **primary obstacle** is test_WSS_z = 8.6175% (277bp above Transolver-3's target 5.85%). Three independent loss-side closures (H339 uniform reweight, H341 wz-only reweight, H345 PCGrad direction) have established that diagonal training-time gradient interventions are structurally closed. **The WSS_z bottleneck is not a gradient budget, routing, or magnitude problem.** The attack has shifted to representation/target/input mechanisms:

**4-axis parallel WSS_z attack (all in-flight):**
1. **H346 askeladd focal-batch** — per-batch focal weighting on τ_z residuals (Option A, no cross-step EMA). Spatial concentration: hard points within each batch get exponentially higher loss weight.
2. **H347 nezuko physics priors** — geometric constraint (τ⊥n: shear tangent to surface normal) and kNN smoothness regularization. Boundary-layer physics injected as training-time constraint.
3. **H348 fern curvature features** — adds per-vertex (H, K, k1, k2) surface curvature as new INPUT channels. WSS_z error may concentrate on high-curvature surfaces; Falkner-Skan boundary layer scales with curvature.
4. **H349 frieren arcsinh-target** — transforms regression target via `arcsinh(τ_z)`. Compresses dynamic range; addresses mismatch between linear-MSE training (over-weights high-|τ_z|) and rel_l2 eval (over-weights low-|τ_z|).

**Orthogonal non-WSS supplements:**
5. **H338 edward SP reweight** — SP channel gap (3.6bp above floor) attacked separately via SP-targeted loss multiplier.
6. **H340 tanjiro σ-sweep** — optimal noise sigma at ν=4, may compound with K=5 finding.
7. **H342 alphonse multi-checkpoint TTA** — output-space average of ep13+ep14+ep15.
8. **H343 thorfinn SAM** — sharper loss basin from SAM optimizer may improve generalization.

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
