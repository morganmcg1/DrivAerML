<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# WSS-Reduction Research Ideas — 2026-05-14 14:30

**Research context:** SOTA on corrected split (rawcanon_20260511) is PR #972 (run `56bcqp3m`):
test_abupt=5.844%, test_vol_p=3.643%, test_surf_p=3.577%, **test_wss=6.727%**.
Target: test_wss < 5.85% (Transolver-3 reference), requiring ~13% relative reduction (~0.877pp).

**Root cause of WSS gap:** tau_y=7.94% (2.2× AB-UPT target of 3.65%) and tau_z=9.54% (2.6× AB-UPT
target of 3.63%) dominate total WSS error. tau_x=6.45% (1.2× target) is secondary.
Cross-flow shear components are the bottleneck — not streamwise stress.

**Base training stack (all experiments build on):**
Lion lr=1e-4 + 6L STRING (5-sigma octave) + GradNorm α=0.5 + EMA decay=0.999 +
Y-sym p=0.5 + bs=1 + 65k surface + 65k volume points + DDP8 + SDF-stratified vol α=2.0 +
~30ep / 24h wall.

**Hard constraints (do not violate):**
- No ensemble, model soup, greedy K, NNLS — single model only.
- No hard tangent projection or output-coordinate projection losses.
- No static/scalar global loss weights >2.0 or <0.5.
- No backbone replacements (Mamba, S4D, Perceiver).
- Every proposal must explicitly protect test_vol_p and test_surf_p.

**Active WIP (do NOT duplicate):**
#1068 dedicated WSS surface decoder head, #1066 tau Y/Z loss weight sweep,
#1069 combined SDF+curvature stratified SAMPLING, #1058 GradNorm on corrected dataset,
#1055 SDF α sweep 1.5/3.0/4.0, #1063 SDF near-surface α, #1065 SDF extended 45ep,
#1061 stochastic per-batch vol points, #1060 vol-loss-weight sweep, #1071 surface 131k points,
#1070 dedicated vol_p aux head, #1067 RFF octave ladder, #1050 dropout p=0.1.

---

## Ranking Summary

| Rank | Hypothesis | One-line claim | EV × P(success) |
|------|-----------|----------------|-----------------|
| 1 | H1: Wind-exposure geometric proxy | Normal·freestream dot product directly encodes cross-flow forcing that tau_y/z errors track | High |
| 2 | H2: Surface curvature input features | kappa_H + kappa_G as 9-channel input improve cross-flow curvature turn prediction | High |
| 3 | H3: Near-wall volume feature injection | SDF-stratified volume points carry boundary-layer velocity gradient signal absent from surface alone | High |
| 4 | H4: Per-task GradNorm α | Global α=0.5 underweights tau_y/z specifically; per-head α unlocks finer balance | Medium-High |
| 5 | H5: Darboux / principal-curvature frame | Encoding tangent-frame basis vectors as surface input makes network basis-invariant for cross-flow WSS | Medium-High |
| 6 | H6: PCGrad gradient surgery | Conflicting gradients between vol_p and tau_y/z waste capacity; orthogonalizing them frees WSS learning | Medium |
| 7 | H7: Adaptive per-channel gradient clipping | tau_y/z have larger gradient norm variance than tau_x; per-channel clipping stabilizes their learning | Medium |
| 8 | H8: Multi-scale local k-NN surface aggregation | Fine-scale neighborhood messages below STRING's global attention resolution encode near-separation geometry | Medium |
| 9 | H9: Trainable per-WSS-channel activations | sigma-weighted activation per tau channel allows asymmetric nonlinearity in cross-flow vs streamwise | Low-Medium |
| 10 | H10: Curvature-only surface point sampling | Oversample high-kappa surface regions where tau_y/z error concentrates, without touching volume stack | Low-Medium |

---

## H1: Wind-Exposure Geometric Proxy Feature

**One-line claim:** Adding the dot product of the outward surface normal with the freestream direction
(-X axis) as a 10th surface input channel encodes cross-flow attack angle directly, targeting the
tau_y/z gap without modifying architecture or loss.

**Mechanism:** The current 7-channel surface input [x,y,z,nx,ny,nz,area] has no explicit signal
about how each surface element is oriented relative to the flow. tau_y and tau_z errors concentrate
at swept surfaces, A-pillars, underbody transitions, and rear diffuser where the local inflow angle
is oblique to the surface and cross-flow gradients are steep. The freestream dot product
`n_dot_u = -(nx)` (negative since freestream is -X) is a scalar in [-1,1] that:
- Equals +1 on blunt frontal faces (stagnation point, where tau is small — low-risk region)
- Equals 0 on side/underbody faces (where tau_y/z are largest and most variable)
- Equals -1 on rear-facing surfaces (wake, base bleed)
The network can use this scalar to specialise its representation of shear for oblique-flow zones.
A second useful proxy is `|ny|` (absolute cross-flow normal component), capturing lateral obliquity.
Both are zero-cost: computed from the existing nx,ny,nz channels at data-load time.

**vol_p / surf_p protection:** The new feature is additive to the existing input embedding; the
volume branch is completely unmodified (its input remains [x,y,z,sdf]). surf_p uses the same
surface encoder as WSS — the dot-product feature provides monotone ordering information that cannot
hurt pressure since stagnation geometry is already well-captured. GradNorm α=0.5 is retained and
continues to balance the three loss heads dynamically.

**DDP8 torchrun CLI flags:**

```bash
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --surface_input_extra_features wind_dot_product abs_cross_normal \
  --optimizer lion \
  --lr 1e-4 \
  --epochs 30 \
  --batch_size 1 \
  --surface_points 65000 \
  --volume_points 65000 \
  --string_sigma_octaves 5 \
  --string_layers 6 \
  --gradnorm_alpha 0.5 \
  --ema_decay 0.999 \
  --y_sym_prob 0.5 \
  --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h1-wind-exposure-proxy
```

**Smoke test plan (EP3 gate):** val_wss <= 7.3% and val_vol_p <= 5.0% and val_surf_p <= 4.5%.
If wind_dot_product feature causes training instability (loss spike) by EP3, abort immediately.

**Gate schedule:**
- EP6: val_wss < 7.2%, val_vol_p < 4.8%, val_surf_p < 4.3% — continue
- EP10: val_wss < 7.0%, val_vol_p < 4.5%, val_surf_p < 4.1%
- EP15: val_wss < 6.9%, val_vol_p < 4.3%
- EP20: val_wss < 6.75%, approaching baseline
- EP25: val_wss < 6.65%
- EP30: final — must show test_wss improvement vs 6.727% baseline

**Success criterion:** test_wss < 6.5% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- The normal is already in the input — network may learn this combination implicitly and gain nothing.
- Feature collinearity: nx is already in the input, so `n_dot_u = -nx` is algebraically redundant unless nonlinear interactions are the bottleneck.
- Mitigation: include `|ny|` (not redundant with any existing channel) as the primary test feature.

**Suggested student:** dl24-nezuko (currently idle)

---

## H2: Surface Curvature Input Features (kappa_H, kappa_G)

**One-line claim:** Adding mean curvature kappa_H and Gaussian curvature kappa_G as 2 additional
surface input channels (9-channel input) gives the model explicit local shape curvature signal,
improving prediction at separation edges and curved underbody panels where cross-flow WSS spikes.

**Mechanism:** PR #1074 was PROPOSED but NEVER RAN — this is the first test of this idea.
WSS magnitudes change sharply at regions of high local curvature: leading edges (high |kappa_H|),
rounded A-pillar transitions, wheel arch lips, and underbody strakes. The current 7-channel input
encodes only the local normal and area element, not the surface curvature tensor. kappa_H (mean
curvature = half the trace of the shape operator) and kappa_G (Gaussian = determinant) together
characterise the local shape type (elliptic, hyperbolic, parabolic). Regions with |kappa_H| >> 0
are precisely where the attached/separated shear boundary lives and where tau_y/z errors are worst.
The features are precomputed from the mesh triangle adjacency (cotangent-weighted Laplace-Beltrami
operator) and stored as two scalar channels, zero cost at train time.

**vol_p / surf_p protection:** Volume branch unchanged. For surf_p, high-curvature regions are also
where pressure changes fastest — curvature features should help surf_p at edges too, providing a
positive side effect rather than a trade-off. GradNorm α=0.5 retained.

**DDP8 torchrun CLI flags:**

```bash
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --surface_extra_channels kappa_mean kappa_gaussian \
  --optimizer lion \
  --lr 1e-4 \
  --epochs 30 \
  --batch_size 1 \
  --surface_points 65000 \
  --volume_points 65000 \
  --string_sigma_octaves 5 \
  --string_layers 6 \
  --gradnorm_alpha 0.5 \
  --ema_decay 0.999 \
  --y_sym_prob 0.5 \
  --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h2-curvature-features
```

**Smoke test plan (EP3 gate):** val_wss <= 7.3%. Check that kappa_H and kappa_G are finite and
not NaN after mesh preprocessing (curvature computation can produce NaN at non-manifold edges —
student should clamp to [-10, 10] in physical units before normalization).

**Gate schedule:**
- EP6: val_wss < 7.1%
- EP10: val_wss < 6.95%
- EP15: val_wss < 6.8%
- EP20: val_wss < 6.70%
- EP25: val_wss < 6.60%
- EP30: test_wss < 6.5%

**Success criterion:** test_wss < 6.5% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- Curvature estimation on raw triangulated CFD meshes is noisy at coarse-resolution panels.
  Cotangent Laplacian is preferred over simple angle-weighted; student must verify.
- Large kappa values at sharp geometric edges (wheel arch rims) may dominate normalization.
  Robust normalization (clip at 99th percentile, then standardize) is essential.
- Model may not use curvature channels if STRING already captures geometric variation via spatial
  sinusoids; an ablation on val_wss vs. without curvature at EP10 is valuable.

**Suggested student:** dl24-fern (when next idle)

---

## H3: Near-Wall Volume Feature Injection Into Surface Head

**One-line claim:** Cross-attending the surface decoder to SDF-stratified near-wall volume tokens
(SDF < 0.05m) injects boundary-layer velocity-gradient signal into WSS prediction, closing the
gap between the surface-only encoder and the true near-wall physics.

**Mechanism:** The current model computes surface predictions from surface tokens alone — the volume
branch feeds a separate head and the two branches share only the backbone. But WSS = mu * dU/dy|wall
is determined by the near-wall velocity gradient, which is encoded in the volume tokens near the
wall (SDF < 0.05m). Those volume tokens already exist in the batch (65k points, many in the
near-wall layer via SDF-stratified α=2.0 sampling). A lightweight cross-attention layer in the
surface decoder queries position-encoded surface tokens against the near-wall volume subset,
retrieving boundary-layer shape information without any new data cost. This is conceptually similar
to boundary-layer feature injection in DoMINO (NVIDIA 2025) but implemented as cross-attention
rather than explicit near-wall meshing.
The cross-attention is applied only after the final transformer layer, before the surface MLP head.
The query is the surface token, key/value are volume tokens with SDF < threshold (e.g., 5cm).

**vol_p / surf_p protection:** Volume head is unmodified. surf_p shares the surface encoder —
the cross-attention uses the same surface token representation, so surf_p implicitly benefits from
boundary-layer signal at high-pressure-gradient regions too. GradNorm α=0.5 retained.
If the cross-attention gate learns low weights for surf_p relevant regions, no harm is done.

**DDP8 torchrun CLI flags:**

```bash
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --surface_decoder_vol_cross_attn true \
  --surface_decoder_vol_cross_attn_sdf_threshold 0.05 \
  --surface_decoder_vol_cross_attn_heads 4 \
  --optimizer lion \
  --lr 1e-4 \
  --epochs 30 \
  --batch_size 1 \
  --surface_points 65000 \
  --volume_points 65000 \
  --string_sigma_octaves 5 \
  --string_layers 6 \
  --gradnorm_alpha 0.5 \
  --ema_decay 0.999 \
  --y_sym_prob 0.5 \
  --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h3-nearwall-vol-injection
```

**Smoke test plan (EP3 gate):** val_wss <= 7.3%, no OOM. The cross-attention over near-wall subset
adds O(N_surf × N_nearwall) attention ops — student must verify GPU memory headroom at bs=1, 65k
surface + ~10k near-wall points. If OOM, reduce near-wall budget to 8k points.

**Gate schedule:**
- EP6: val_wss < 7.2%
- EP10: val_wss < 6.9%
- EP15: val_wss < 6.75%
- EP20: val_wss < 6.65%
- EP25: val_wss < 6.55%
- EP30: test_wss < 6.45%

**Success criterion:** test_wss < 6.4% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- Memory budget: near-wall subset at SDF < 0.05m may be large; start with 0.02m threshold.
- Cross-attention adds meaningful parameters; with only 400 training cases, overfitting is possible.
  Apply dropout=0.1 on the cross-attention value projection only.
- Gradient flow: backprop through cross-attention into volume tokens may destabilize vol_p head.
  Detach volume tokens at the cross-attention input (stop gradient for volume head only).

**Suggested student:** dl24-tanjiro (when idle)

---

## H4: Per-Task GradNorm Alpha (Separate alpha Per Output Head)

**One-line claim:** Using different GradNorm learning-rate coefficients for the WSS head (alpha_wss)
vs. the surface-pressure head (alpha_surf) vs. the volume head (alpha_vol) replaces the global
alpha=0.5 with a richer balancing that can account for the very different training dynamics of
tau_y/z versus pressure.

**Mechanism:** The current GradNorm uses a single alpha=0.5 for ALL three output tasks equally.
But alpha controls the speed at which GradNorm returns a task to its "fair share" of gradient
magnitude relative to its training rate. tau_y and tau_z have approximately 2× higher absolute
error than surf_p/vol_p and their loss curves evolve much more slowly — this is exactly the regime
where a HIGHER alpha_wss (faster rebalancing toward WSS when it falls behind) could help. Global
alpha=0.5 may be the right value for surf_p and vol_p but slightly too slow for WSS.
Proposed sweep: alpha_wss in {0.5, 0.75, 1.0} with alpha_surf=alpha_vol=0.5 fixed.
Note: alpha=0.75 GLOBALLY failed catastrophically (EP16 blowup) — but that may have been because
it simultaneously raised all three heads. Per-task alpha allows WSS to be more aggressive while
keeping pressure heads conservative.

**vol_p / surf_p protection:** alpha_surf=0.5 and alpha_vol=0.5 are explicitly held at the known-
safe value. The only risk is indirect: if alpha_wss=1.0 drives a large WSS loss reduction that
GradNorm then interprets as WSS "getting better faster", it will re-allocate gradient back toward
surf_p/vol_p. This is the correct behavior and should protect them.

**DDP8 torchrun CLI flags (3-arm sweep):**

```bash
# Arm A: alpha_wss=0.75
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --gradnorm_alpha_wss 0.75 \
  --gradnorm_alpha_surf 0.5 \
  --gradnorm_alpha_vol 0.5 \
  --optimizer lion --lr 1e-4 --epochs 30 --batch_size 1 \
  --surface_points 65000 --volume_points 65000 \
  --string_sigma_octaves 5 --string_layers 6 \
  --ema_decay 0.999 --y_sym_prob 0.5 --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h4-pertask-gradnorm-alpha-wss0.75

# Arm B: alpha_wss=1.0
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --gradnorm_alpha_wss 1.0 \
  --gradnorm_alpha_surf 0.5 \
  --gradnorm_alpha_vol 0.5 \
  --optimizer lion --lr 1e-4 --epochs 30 --batch_size 1 \
  --surface_points 65000 --volume_points 65000 \
  --string_sigma_octaves 5 --string_layers 6 \
  --ema_decay 0.999 --y_sym_prob 0.5 --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h4-pertask-gradnorm-alpha-wss1.0
```

**Smoke test plan (EP3 gate):** Both arms: val_wss <= 7.3%, no divergence. Compare Arm A vs B
gradient norm ratio (wss/surf) — Arm B should show higher wss gradient weight at EP3.

**Gate schedule:**
- EP6: both arms val_wss < 7.1%
- EP10: best arm val_wss < 6.95%
- EP15: val_wss < 6.80%
- EP20: drop weaker arm; advance better one
- EP30: test eval on winner

**Success criterion:** Best arm: test_wss < 6.5% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- The global alpha=0.75 blowup at EP16 was the key precedent — student must monitor loss curves
  carefully through EP12-16 window and abort Arm B immediately if divergence appears.
- Per-task alpha changes GradNorm internals — student must confirm the GradNorm implementation
  exposes per-task alpha control, or implement a simple wrapper.

**Suggested student:** dl24-alphonse (when idle)

---

## H5: Darboux Frame / Principal Curvature Direction Encoding

**One-line claim:** Encoding the two principal curvature direction vectors (e1, e2) of the surface
as 6 additional input channels (alongside their curvature magnitudes kappa_1, kappa_2) gives the
network a local tangent frame aligned with the surface's shape operator, making cross-flow WSS
prediction invariant to arbitrary parameterization.

**Mechanism:** Motivated by the Intrinsic Vector Heat Network (Gao et al., arXiv 2406.09648, 2024)
and Mesh CNN WSS (Suk et al., arXiv 2109.04797). WSS is a surface tangent vector field — its
decomposition into tau_y/tau_z is an artifact of the global Cartesian frame, not the intrinsic
geometry. The network currently learns in Cartesian coordinates where a 45-degree rotation of a
car panel would produce completely different tau_y/tau_z values for the same physical shear stress.
The principal curvature directions e1 (max curvature direction) and e2 (min curvature direction)
form a natural local frame on the surface that is tied to the geometry, not the global orientation.
Encoding e1 and e2 as 6 channels (3D vectors, normalized) plus kappa_1 and kappa_2 (scalars)
gives the network a local basis it can use to decompose WSS into a physically meaningful frame.
Total new input: 7 + 6 + 2 = 15 channels. Input projection MLP grows accordingly.

**vol_p / surf_p protection:** Volume branch unchanged. The Darboux frame is surface-only.
surf_p uses the same enlarged surface encoder but pressure is a scalar — the extra frame channels
add expressivity that can only help. GradNorm α=0.5 retained.

**DDP8 torchrun CLI flags:**

```bash
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --surface_extra_channels principal_dir_1 principal_dir_2 kappa_1 kappa_2 \
  --optimizer lion \
  --lr 1e-4 \
  --epochs 30 \
  --batch_size 1 \
  --surface_points 65000 \
  --volume_points 65000 \
  --string_sigma_octaves 5 \
  --string_layers 6 \
  --gradnorm_alpha 0.5 \
  --ema_decay 0.999 \
  --y_sym_prob 0.5 \
  --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h5-darboux-frame
```

**Smoke test plan (EP3 gate):** val_wss <= 7.3%. Verify principal direction computation does not
produce NaN at flat panels (degenerate when kappa_1 ≈ kappa_2 — random perturbation needed).

**Gate schedule:**
- EP6: val_wss < 7.1%
- EP10: val_wss < 6.9%
- EP15: val_wss < 6.75%
- EP20: val_wss < 6.65%
- EP30: test_wss < 6.45%

**Success criterion:** test_wss < 6.45% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- Principal direction computation is numerically unstable at umbilic points (kappa_1 = kappa_2).
  Must handle gracefully: set e1=[1,0,0], e2=[0,1,0] when |kappa_1 - kappa_2| < epsilon.
- Sign convention for e1/e2 is ambiguous (both e1 and -e1 are valid principal directions).
  The network must learn to handle sign ambiguity, or apply random sign flipping during training
  as augmentation. Y-symmetry augmentation already in the stack will help here.
- 15-channel input expands the model's first projection layer. Verify that parameter budget does
  not push the model over GPU memory at 65k surface points, bs=1.

**Suggested student:** dl24-askeladd (when idle)

---

## H6: PCGrad Gradient Surgery for WSS vs Pressure Conflict

**One-line claim:** PCGrad (Yu et al., NeurIPS 2020) orthogonalizes gradients between conflicting
task pairs (wss, surf_p, vol_p) before the optimizer step, eliminating the gradient interference
that forces GradNorm to zero out one task to let another learn.

**Mechanism:** PR #1003 was PROPOSED but NEVER RAN. At each optimizer step, PCGrad:
1. Computes per-task gradients g_wss, g_surfp, g_volp separately.
2. For each pair (i,j), if cos(g_i, g_j) < 0 (conflicting), projects g_i onto the plane normal to
   g_j, replacing the conflicting component with zero.
3. Sums the projected gradients and uses this for the optimizer step.
This is orthogonal to GradNorm: GradNorm scales loss magnitudes to equalize task learning rates;
PCGrad removes directional conflicts in the gradient space. Both can be used simultaneously.
The key hypothesis: tau_y/z gradients may conflict with vol_p gradients in the backbone because
cross-flow shear prediction requires different representational specialization than volume pressure.
GradNorm compensates by reducing vol_p weight when wss is struggling, but this risks vol_p regression.
PCGrad instead allows the optimizer to find a direction that progresses both simultaneously.

**vol_p / surf_p protection:** PCGrad is explicitly designed to prevent one task from regressing
another. Projection ensures that wss gradient updates don't move the model in a direction that
hurts vol_p. This is the strongest explicit protection mechanism available short of separate models.

**DDP8 torchrun CLI flags:**

```bash
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --gradient_surgery pcgrad \
  --gradient_surgery_tasks wss surf_p vol_p \
  --optimizer lion \
  --lr 1e-4 \
  --epochs 30 \
  --batch_size 1 \
  --surface_points 65000 \
  --volume_points 65000 \
  --string_sigma_octaves 5 \
  --string_layers 6 \
  --gradnorm_alpha 0.5 \
  --ema_decay 0.999 \
  --y_sym_prob 0.5 \
  --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h6-pcgrad-gradient-surgery
```

**Smoke test plan (EP3 gate):** val_wss <= 7.3%, monitor per-task gradient cosine similarity —
if most pairs are already positive (not conflicting), PCGrad adds overhead but no benefit.
Check: if <10% of steps have negative cosine similarity, consider the approach uninformative.

**Gate schedule:**
- EP6: val_wss < 7.2%
- EP10: val_wss < 6.95%
- EP15: val_wss < 6.80%
- EP20: val_wss < 6.70%
- EP30: test_wss < 6.50%

**Success criterion:** test_wss < 6.5% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- PCGrad requires computing per-task gradients separately (backward pass per task), roughly
  tripling backward pass time. With bs=1, DDP8, and 30ep budget, this may push over 24h wall time.
  Student should verify wall-clock estimate at EP1 and abort if >72h projected.
- With Lion optimizer (which normalizes gradient direction to sign(g)), the cosine-similarity
  condition for PCGrad may behave differently. Student should test with a brief AdamW baseline run
  before committing to Lion+PCGrad.
- PCGrad is designed for loss landscapes where tasks share a backbone. If the backbone is already
  adequately separating task representations (which GradNorm may have accomplished), PCGrad's
  benefit will be marginal.

**Suggested student:** dl24-frieren (when idle)

---

## H7: Adaptive Per-Channel Gradient Clipping for tau_y and tau_z

**One-line claim:** Applying per-channel gradient norm clipping with separate budgets for tau_y
and tau_z (larger clip norm = 5.0) vs tau_x and surf_p/vol_p (baseline clip norm = 1.0) allows
the cross-flow components to receive proportionally larger gradient updates without destabilizing
the better-performing heads.

**Mechanism:** The global gradient clipping in the current stack treats all output channels
identically. But tau_y and tau_z have ~2.2× and ~2.6× higher relative error than tau_x — they
are in a materially different optimization regime. Allowing a larger gradient norm for the
parameters that primarily affect tau_y/z (the surface MLP output layer rows corresponding to
channels 2 and 3) while keeping the clip norm tight for other parameters prevents the well-trained
channels from dominating the gradient update budget and leaving the poorly-trained ones behind.
Implementation: in the backward hook, identify the gradient of the surface output head's final
linear layer and apply separate clip norms per output channel row. This does not require
architectural changes — only a post-backward gradient hook.

**vol_p / surf_p protection:** The per-channel clip is applied only to the surface output MLP's
final layer. The volume head and the surface pressure channel (channel 0 of surface_preds) retain
their original clip norm. GradNorm α=0.5 continues to balance task losses. The risk of
cross-channel interference through the shared backbone is mitigated by the selective clipping.

**DDP8 torchrun CLI flags:**

```bash
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --per_channel_grad_clip true \
  --grad_clip_tau_y 5.0 \
  --grad_clip_tau_z 5.0 \
  --grad_clip_tau_x 1.0 \
  --grad_clip_surf_p 1.0 \
  --grad_clip_global 1.0 \
  --optimizer lion \
  --lr 1e-4 \
  --epochs 30 \
  --batch_size 1 \
  --surface_points 65000 \
  --volume_points 65000 \
  --string_sigma_octaves 5 \
  --string_layers 6 \
  --gradnorm_alpha 0.5 \
  --ema_decay 0.999 \
  --y_sym_prob 0.5 \
  --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h7-perchannel-grad-clip
```

**Smoke test plan (EP3 gate):** val_wss <= 7.3%. Monitor per-channel gradient norms at EP1 to
verify that tau_y/z gradients are being clipped more permissively and tau_x is not affected.
If all channels have norms < 1.0 (below baseline clip), the per-channel clipping is a no-op —
abort and diagnose.

**Gate schedule:**
- EP6: val_wss < 7.1%
- EP10: val_wss < 6.95%
- EP15: val_wss < 6.80%
- EP20: val_wss < 6.70%
- EP30: test_wss < 6.50%

**Success criterion:** test_wss < 6.5% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- The hypothesis assumes that tau_y/z gradients are being clipped more aggressively than tau_x
  under the current global clip budget. This must be verified empirically at EP1. If the current
  gradient norms are already similar across channels, this is a no-op.
- Larger clip budget for tau_y/z could cause instability if combined with GradNorm — the adaptive
  loss weight and the adaptive gradient norm now both act on the same parameters. Monitor
  GradNorm weight oscillation.
- Lion optimizer already normalizes gradient direction, which partially mitigates clip-norm effects.

**Suggested student:** dl24-edward (when idle)

---

## H8: Multi-Scale Local k-NN Surface Aggregation

**One-line claim:** Prepending a lightweight k-NN message-passing stage (k=16, r=2cm; k=64, r=5cm)
to the surface encoder adds fine-scale neighborhood context below STRING's effective resolution,
capturing near-separation geometry features that STRING sinusoids cannot resolve.

**Mechanism:** DoMINO (NVIDIA, 2025) showed that multi-scale local neighborhood information is
critical for accurate DrivAerML predictions. STRING operates as a global attention mechanism with
sinusoidal positional encoding — its effective resolution is set by the sigma parameters (3cm to
~1m in the 5-octave stack). At the scale of separation geometry (boundary layer thickness ~1-3mm,
pressure gradient variation over 1-2cm), STRING may not have sufficient resolution even at its
smallest sigma. A lightweight 2-scale PointNet-style neighborhood aggregation:
- Scale 1: k=16 neighbors within r=2cm — captures micro-scale surface variation
- Scale 2: k=64 neighbors within r=5cm — captures meso-scale panel geometry
These produce local feature vectors that augment the per-point input embedding BEFORE the STRING
transformer. The aggregation uses max-pool + MLP (no learnable positions, no added STRING PEs).
Additional parameters ~500k (small relative to backbone).

**vol_p / surf_p protection:** Volume branch completely unmodified. surf_p shares the surface encoder
and will benefit from local neighborhood signal at pressure gradient regions too. GradNorm α=0.5
retained. The local aggregation adds no new loss terms.

**DDP8 torchrun CLI flags:**

```bash
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --surface_local_agg true \
  --surface_local_agg_scales "16,0.02;64,0.05" \
  --optimizer lion \
  --lr 1e-4 \
  --epochs 30 \
  --batch_size 1 \
  --surface_points 65000 \
  --volume_points 65000 \
  --string_sigma_octaves 5 \
  --string_layers 6 \
  --gradnorm_alpha 0.5 \
  --ema_decay 0.999 \
  --y_sym_prob 0.5 \
  --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h8-multiscale-knn-agg
```

**Smoke test plan (EP3 gate):** val_wss <= 7.3%, no OOM. k-NN computation on 65k points at k=64
is the memory bottleneck — use torch_cluster or open3d ball_query, check GPU memory at EP1.

**Gate schedule:**
- EP6: val_wss < 7.1%
- EP10: val_wss < 6.9%
- EP15: val_wss < 6.75%
- EP20: val_wss < 6.65%
- EP30: test_wss < 6.45%

**Success criterion:** test_wss < 6.45% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- k-NN computation on 65k points per sample per forward pass adds significant CPU/GPU time.
  Use a precomputed neighbor index (stored with the processed dataset) if available, otherwise
  accept ~10-20% wall-clock overhead and verify budget stays within 24h.
- With only 400 training samples and additional parameters, overfitting is a risk.
  The max-pool aggregation is a limited expressive power choice that partially mitigates this.

**Suggested student:** dl24-nezuko (second assignment priority if H1 is already in-flight)

---

## H9: Trainable Per-WSS-Channel Activation Functions in Surface Head

**One-line claim:** Replacing fixed ReLU/GELU activations in the surface MLP head with per-output-
channel trainable rational activations (Boulle et al. 2020) allows the model to learn asymmetric
nonlinearities specific to tau_y and tau_z, whose distributions are highly skewed near separation.

**Mechanism:** Motivated by Farea et al. (arXiv 2509.14437, 2025) on trainable activations for
multi-task Navier-Stokes surrogates. The surface MLP head currently uses a fixed activation
function applied uniformly across all output channels. tau_y and tau_z have highly skewed,
long-tailed distributions near separation edges (sudden sign flips, large positive excursions on
windward faces, near-zero in the lee). A trainable rational activation function
f(x) = P(x)/Q(x) = (a0 + a1*x + a2*x^2 + ...) / (b0 + b1*x + ...) with learnable coefficients
allows each output channel to develop its own nonlinear response curve. This adds only ~20
parameters per activation (degree-3 rational = 8 numerator + 4 denominator coefficients), is fully
differentiable, and runs at negligible compute overhead.
Implementation: apply one trainable activation per output channel in the LAST layer of the surface
MLP (not in the backbone). Total new parameters: 4 channels × ~12 coefficients = 48 scalars.

**vol_p / surf_p protection:** The trainable activations are only in the surface head's final layer.
The volume head retains its original activation. surf_p will get its own trainable activation too
(channel 0), which can only help. GradNorm α=0.5 retained.

**DDP8 torchrun CLI flags:**

```bash
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --surface_head_trainable_activations true \
  --surface_head_activation_type rational_p3q2 \
  --optimizer lion \
  --lr 1e-4 \
  --epochs 30 \
  --batch_size 1 \
  --surface_points 65000 \
  --volume_points 65000 \
  --string_sigma_octaves 5 \
  --string_layers 6 \
  --gradnorm_alpha 0.5 \
  --ema_decay 0.999 \
  --y_sym_prob 0.5 \
  --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h9-trainable-activations
```

**Smoke test plan (EP3 gate):** val_wss <= 7.3%. Verify denominator Q(x) != 0 numerically — the
rational activation can develop poles during training. Use the stabilized variant with
|Q(x)| + epsilon in the denominator. Check learned activation shape at EP5: should show
asymmetry between tau_y+ and tau_y-.

**Gate schedule:**
- EP6: val_wss < 7.1%
- EP10: val_wss < 6.9%
- EP15: val_wss < 6.80%
- EP20: val_wss < 6.68%
- EP30: test_wss < 6.50%

**Success criterion:** test_wss < 6.5% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- Rational activation denominator can develop near-zero values; must use stability regularization.
- The benefit depends on whether the surface head's activation is actually a bottleneck. If the
  MLP has multiple layers, the learned representation may already be sufficiently flexible.
  Ablation at EP10 vs standard GELU will clarify.

**Suggested student:** dl24-alphonse (second priority after H4)

---

## H10: Curvature-Stratified Surface Point Sampling (Standalone)

**One-line claim:** Oversampling surface points in proportion to local mean curvature |kappa_H|
(independent of SDF-stratified volume sampling) allocates more training attention to separated-flow
regions where tau_y/z error is highest, without modifying architecture or loss formulation.

**Mechanism:** PR #1069 tests SDF+curvature COMBINED stratification for VOLUME points. This is
distinct: curvature-stratified SURFACE point sampling, applied independently. The current surface
sampling is uniform random — each of the ~400k surface triangles is equally likely to be in the
65k-point sample. But tau_y/z errors are concentrated at high-curvature regions (A-pillars,
underbody edges, wheel arches), which represent a small fraction of total surface area. By sampling
with probability proportional to |kappa_H|^beta (beta=1.0 to 2.0) plus uniform background, the
training mini-batches will contain proportionally more points from the difficult high-curvature
regions per epoch. This is analogous to focal sampling: simple, cheap, and directly addresses
the "hard example" regime for WSS.
At eval, continue using the standard uniform/strided sampler (no distributional shift at test time).

**vol_p / surf_p protection:** Volume sampling is unchanged — SDF-stratified α=2.0 stack retained.
surf_p operates on the same surface points as WSS; if high-curvature regions are also regions of
high pressure gradient (which they generally are at leading edges), surf_p benefits too.
GradNorm α=0.5 retained.

**DDP8 torchrun CLI flags (2-arm sweep: beta=1.0 and beta=2.0):**

```bash
# Arm A: beta=1.0
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --surface_curvature_sampling_beta 1.0 \
  --surface_curvature_sampling_bg_frac 0.3 \
  --optimizer lion --lr 1e-4 --epochs 30 --batch_size 1 \
  --surface_points 65000 --volume_points 65000 \
  --string_sigma_octaves 5 --string_layers 6 \
  --gradnorm_alpha 0.5 --ema_decay 0.999 --y_sym_prob 0.5 --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h10-curvature-sampling-beta1

# Arm B: beta=2.0
torchrun --nproc_per_node=8 train.py \
  --dataset_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --surface_curvature_sampling_beta 2.0 \
  --surface_curvature_sampling_bg_frac 0.3 \
  --optimizer lion --lr 1e-4 --epochs 30 --batch_size 1 \
  --surface_points 65000 --volume_points 65000 \
  --string_sigma_octaves 5 --string_layers 6 \
  --gradnorm_alpha 0.5 --ema_decay 0.999 --y_sym_prob 0.5 --vol_sdf_stratified_alpha 2.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group h10-curvature-sampling-beta2
```

**Smoke test plan (EP3 gate):** val_wss <= 7.3% for both arms. Verify that curvature values are
precomputed and stored in the processed dataset, or that the preprocessing overhead at data-load
is <5% of step time. If curvature is not precomputed, recommend running offline preprocessing first.

**Gate schedule:**
- EP6: best arm val_wss < 7.1%
- EP10: val_wss < 6.9%
- EP15: val_wss < 6.75%
- EP20: val_wss < 6.65%
- EP30: test_wss < 6.45%

**Success criterion:** test_wss < 6.45% AND test_vol_p <= 3.75% AND test_surf_p <= 3.65%.

**Risks:**
- If curvature values are not precomputed in the rawcanon_20260511 dataset, the student must add an
  offline preprocessing step. This should be done before the run starts, not during training.
- Very aggressive curvature weighting (beta=3+) can cause the sampler to undersample flat panels
  almost entirely, causing the model to degrade on those regions. beta=2.0 with 30% background
  uniform floor is a safe upper bound.
- This partially overlaps with #1069 (which combines SDF+curvature for volume). Student must
  confirm that #1069 does NOT touch surface sampling — if it does, this PR must be rebased to
  differ only in the surface-sampling component.

**Suggested student:** dl24-edward (second priority after H7)

---

## Appendix: Key Literature

1. **DoMINO (NVIDIA, 2025):** Multi-scale local geometric conditioning on DrivAerML. Validates
   the value of local neighborhood information beyond global attention.
   https://arxiv.org/abs/2501.02992

2. **Intrinsic Vector Heat Network (Gao et al., arXiv 2406.09648, 2024):** Intrinsic tangent
   vector field learning on manifolds. Directly relevant to WSS as a surface vector field.
   Architecture-level inspiration for H5.

3. **Mesh CNN for 3D WSS Estimation (Suk et al., arXiv 2109.04797, 2021):** Shows that local
   mesh geometry features are the key driver of 3D WSS vector prediction accuracy in vascular flow.
   Motivates H2, H5, H8.

4. **PCGrad: Gradient Surgery (Yu et al., NeurIPS 2020):** Project conflicting gradients to
   prevent catastrophic interference in multi-task learning. Official code at
   https://github.com/WeiChengTseng/Pytorch-PCGrad. Motivates H6.

5. **Rational Activations (Boulle et al., NeurIPS 2020):** Trainable rational activation
   functions, numerically stable implementation available. Motivates H9.

6. **Multi-objective loss balancing PINNs (Farea et al., arXiv 2509.14437, 2025):** Trainable
   activations + loss balancing for complex fluid flow surrogates. Motivates H9.
