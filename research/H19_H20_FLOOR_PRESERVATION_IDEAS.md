# H19 & H20: Orthogonal vol_p Floor-Preservation Mechanisms

**Date:** 2026-05-17
**Context:** H10b beats wss SOTA (test_wss=6.665%) but breaches vol_p floor (+0.517pp → 4.160%). H9b preserves vol_p floor (3.646%) but costs +0.103pp wss (6.781%). H18 tests the H10b+H9b composition. These two hypotheses explore orthogonal alternatives that stack on H10b without relying on MAE_aux.

**Baseline (H10b, PR #1159):** test_wss=6.665%, test_vol_p=4.160%, test_surf_p=3.559%
**Floor targets:** test_vol_p ≤ 3.643%, test_surf_p ≤ 3.577%, test_wss < 6.727%

---

## H19: Charbonnier Loss on vol_p Channel (dl24-frieren)

**Hypothesis:** H10b applies Charbonnier loss to τ_z to make the wss head more robust to high-shear outliers (weighted/MSE ratio=12.7% at EP1). The same mechanism applied symmetrically to the vol_p head should provide analogous robustness on the volume side — shifting the loss landscape to penalize large vol_p residuals smoothly rather than quadratically — without adding any gradient mass via an auxiliary head. Unlike MAE_aux, Charbonnier does not inject extra backward passes through the shared encoder; it only reshapes the vol_p loss surface. GradNorm therefore does not need to compensate by raising w_vol_p, keeping the wss/vol_p gradient budget balance intact.

**Why preferred over MAE_aux:** MAE_aux (H9b) raises w_vol_p by flooding the shared encoder with L1 gradients, which competes with wss gradient mass. Charbonnier on vol_p reshapes the vol_p loss without increasing its absolute gradient norm relative to wss — it should recover the outlier sensitivity problem that causes vol_p to breach without the zero-sum wss cost.

**Implementation (train.py / loss.py, ~15 min):**
Add a `--vol_p_charbonnier_weight` flag (default 0.0). When nonzero, replace the vol_p MSE term with `Charbonnier(pred_vol_p, target_vol_p, eps=1e-3) * vol_p_charbonnier_weight + MSE(pred_vol_p, target_vol_p) * (1 - vol_p_charbonnier_weight)`. Mirror exactly the existing `wss_charbonnier_weight` path already present for τ_z. No model.py changes needed.

**Suggested starting value:** `--vol_p_charbonnier_weight 0.1` (same as H10b τ_z Charb weight).

**Expected outcome:** test_vol_p recovers toward 3.65-3.85% (below floor 3.643% attainable), test_wss holds near 6.665-6.72% (no gradient-mass shift). If w_vol_p telemetry (`train/gradnorm/w_vol_p`) remains stable (no sharp drop) during EP1-5, the mechanism is working as expected.

**Falsification:** If test_vol_p does not improve vs H10b (remains > 4.0%) or test_wss regresses > 0.15pp vs H10b baseline, the hypothesis is falsified. If w_vol_p drops below 0.30 by EP5 (same pattern as H10b), Charbonnier on vol_p is not holding the GradNorm balance.

**NaN risk:** Very low. Charbonnier is smooth at zero (eps=1e-3 prevents gradient explosion). Identical to the τ_z path already battle-tested under Lion lr=1e-4.

---

## H20: Volume-Side SDF-Gradient Attention Bias (dl24-nezuko)

**Hypothesis:** H9 (PR #1145) showed that surface curvature attention bias improves wss by +0.586pp by steering surface tokens toward geometrically complex regions. The same principle applied to the volume Transolver block should help vol_p: volume tokens near the surface (where pressure gradients are sharpest) have high ‖∇SDF‖ ≈ 1, while interior tokens have lower SDF gradients. Adding a small learned or fixed additive bias proportional to ‖∇SDF‖ to the volume attention logits will concentrate volume token capacity near the surface boundary layer — precisely where the vol_p prediction error tends to be highest — without touching any loss formulation or wss gradient paths.

**Why preferred over MAE_aux:** This is a purely architectural mechanism. It does not modify any loss weights, add any auxiliary terms, or change GradNorm dynamics. It is also orthogonal to H19 (loss-shape vs attention-geometry), so if both mechanisms survive independently, they can be composed in a future round.

**Implementation (model.py, ~25 min):**
In the volume Transolver attention block, compute `sdf_grad_norm = torch.norm(volume_x[:, :, 3:4], dim=-1, keepdim=True)` (SDF channel is index 3 of volume features [x,y,z,sdf]). Note: ‖∇SDF‖ is not directly available from the SDF scalar alone — use the approximation that SDF value near zero ≈ boundary, so use `bias = -alpha * volume_x[:, :, 3].abs()` (tokens with |SDF|≈0 are near-surface, get highest attention). Add this scalar bias to attention logits before softmax. Expose `--vol_attn_sdf_bias_alpha` flag (default 0.0, suggested start 0.5). One clean alternative: use `exp(-alpha * |SDF|)` as a multiplicative scale on the value vectors rather than additive logit bias — reduces risk of softmax saturation.

**Suggested starting value:** `--vol_attn_sdf_bias_alpha 0.5` with additive logit formulation, or `--vol_attn_sdf_bias_alpha 1.0` with multiplicative value scaling.

**Expected outcome:** test_vol_p recovers toward 3.65-3.90%, test_wss unchanged vs H10b (no wss path modified). Volume token utilization should shift visibly toward low-|SDF| tokens.

**Falsification:** If test_vol_p does not improve vs H10b (remains > 4.0%), the attention geometry hypothesis is falsified. If training diverges or produces NaN (softmax saturation from large additive bias), reduce alpha or switch to multiplicative formulation.

**NaN risk:** Low with additive logit bias at alpha ≤ 1.0. Recommend clamping the bias term to [-2, 0] range as a safeguard against softmax collapse on edge cases.

---

## Assignment Summary

| Student | Hypothesis | Mechanism | Base | Key flag |
|---|---|---|---|---|
| dl24-frieren | H19 | Charbonnier on vol_p | H10b | `--vol_p_charbonnier_weight 0.1` |
| dl24-nezuko | H20 | SDF-gradient attention bias (volume) | H10b | `--vol_attn_sdf_bias_alpha 0.5` |

Both stack on H10b (curvature+Charb τ_z). Neither uses MAE_aux. Mechanistically orthogonal: loss-shape vs attention-geometry. Both have <10% NaN risk under Lion lr=1e-4 + GradNorm α=0.5.
