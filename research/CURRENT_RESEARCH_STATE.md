# SENPAI Research State

**Updated**: 2026-06-01 11:50Z | Branch: `tay` | **SOTA: H336 K=5+Student-t ν=4+8-res+mirror+cal — val_cal 5.8978 / test_cal 5.7379**

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
| WSS loss reweight (wz-only) | `wz-reweight-monotone-nogate` — H341 CLOSED. Even isolated wz reweight regresses all channels | H341 CLOSED this loop |
| K-axis continuation at ν=4 | `K6-vs-K5-noise-floor-tie` — K=6 ties K=5 at ±0.02bp noise floor; K-axis saturated at K=5 | H344 closed |

---

## Active Fleet (2026-06-01 11:50Z — 8 students with open PRs)

| PR | Student | Hypothesis | Status | Theme |
|---|---|---|---|---|
| **#1536** | frieren | H345: PCGrad/CAGrad gradient surgery on (g_wss, g_pres) conflict | 🟡 WIP | Gradient direction |
| **#1537** | askeladd | H346: Per-vertex focal EMA residual reweighting (γ=1,2,3) | 🟡 WIP | Spatial concentration |
| **#1538** | nezuko | H347: Boundary-layer physics priors (τ⊥n normals + kNN smoothness) | 🟡 WIP | Physical constraint |
| **#1528** | thorfinn | H343: SAM cosine-tail (ρ∈{0.05, 0.02}, 2-arm chained) | 🟡 WIP | Optimizer curvature |
| **#1526** | alphonse | H342: Multi-checkpoint output averaging (ep13+ep14+ep15 TTA) | 🟡 WIP | Output-space averaging |
| **#1524** | tanjiro | H340: σ-sweep at ν=4 (arms σ∈{2.5e-4, 5e-4 ref, 1e-3}) | 🟡 WIP | Noise scale at Student-t |
| **#1522** | edward | H338: SP-targeted loss reweight cosine-tail ({1.05, 1.10, 1.20}) | 🟡 WIP | SP floor gap 3.6bp |
| **#1539** | fern | H348: Surface curvature input features (H, K, k1k2) — NEW | 🆕 just assigned | Input geometry |

---

## Current Research Focus: 3-axis parallel WSS_z attack + orthogonal supplements

The **primary obstacle** is test_WSS_z = 8.6175% (277bp above Transolver-3's target 5.85%). Loss-reweighting is now structurally closed (both H339 uniform and H341 wz-only nulled with uniform cross-channel regression). The attack has shifted to architecture-adjacent mechanisms:

**WSS_z primary attack (3 in-flight):**
1. **H345 frieren PCGrad** — corrects gradient conflict between pressure and WSS loss directions. If g_wss and g_pres are anti-correlated (cosine similarity < 0), gradient projection stops wss improvements from being undone by pressure gradient.
2. **H346 askeladd focal EMA** — per-vertex EMA of running residual. Points with persistent high WSS_z error get exponentially higher loss weight, concentrating gradient signal spatially rather than diagonally.
3. **H347 nezuko physics priors** — geometric constraint (τ⊥n: shear force must be tangent to surface normal) and kNN smoothness regularization. Injects boundary-layer physics directly as training-time constraint.

**Orthogonal supplements:**
4. **H348 fern curvature features** — adds per-vertex (H, K, k1, k2) surface curvature as new INPUT channels. WSS_z error may concentrate on high-curvature surfaces (A-pillars, diffuser, underbody); Falkner-Skan boundary layer theory predicts shear scales with curvature.
5. **H338 edward SP reweight** — SP channel gap (3.6bp above floor) attacked separately via SP-targeted loss multiplier.
6. **H340 tanjiro σ-sweep** — finds optimal noise sigma at ν=4, may compound with K=5 finding.
7. **H342 alphonse multi-checkpoint TTA** — output-space average of ep13+ep14+ep15; orthogonal to weight-space soup (H307 closed).
8. **H343 thorfinn SAM** — sharper loss basin from SAM optimizer may improve generalization on WSS_z.

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

If H345/H346/H347/H348 all null, escalate to architecture-tier (plateau protocol ≥5 consecutive nulls since H336):

1. **Per-axis decoder head for WSS** — channel-specific attention/MLP branch conditioned on axis label {x, y, z}. Gives the model dedicated capacity for z-axis shear without sharing parameters with pressure.
2. **Anisotropic neighborhood sampling** — bias kNN toward streamwise vehicle axis direction where τ_z concentrates. Fundamentally changes the receptive field geometry.
3. **Log-domain WSS regression target** — change training target to log|τ_z| + sign, handling the high dynamic range of z-axis shear without architectural changes.
4. **Multi-scale WSS prediction head** — coarse-to-fine WSS pyramid; higher scales capture global boundary layer, fine scale refines high-curvature regions.
5. **Adversarial geometric augmentation** — augment with random surface perturbations to force the model to learn geometry-invariant flow features.
