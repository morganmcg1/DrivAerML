# Research Ideas — 2026-05-24 12:30 UTC

**Primary objective**: test_WSS < 5.85% (Transolver-3 SOTA).
**Current best**: test_WSS 6.752% (H112/PR #1283, merged 2026-05-24). Gap: 0.902pp.
**Hard constraints**: test_VP ≤ 3.643%, test_SP ≤ 3.577% (must not regress).

**Closed classes (do NOT propose anything in these families):**
1. Tangency-imposition: hard projection of τ onto tangent plane at output (PRs #351, #680, #713, #1299/H123) — model never learns to predict tangent; hard projection zeros gradients
2. WSS-dedicated head: separate decoder module for WSS (PRs #1068, #1123, #1261/H96, #1265/H100)
3. Curvature-aware input features: Gaussian/mean curvature, principal-direction features (PRs #1084, #1146, #1223/H69)
4. Loss-balancing mechanisms: heteroscedastic uncertainty weighting, per-axis decomposition weighting (PRs #1285/H113, #1133) — SP plateau is HARDNESS-BOUND per H113 diagnostic
5. Per-axis WSS reweighting (H114 panel-area): amplifies spurious attractors, destroys gradient signal

---

## Hypotheses Ranked by Expected Information Value

---

### Rank 1 — H-A: Surface-Intrinsic Tangent-Frame Input Encoding

**One sentence**: Replace global (x,y,z) normal components in the surface input feature vector with a per-point local orthonormal frame {t₁, t₂, n}, expressing the model's surface inputs in surface-intrinsic coordinates rather than world frame.

**Specific mechanism**:
At each surface point, construct a local orthonormal frame from the existing surface normal `n_hat = [nx, ny, nz]` (already in input features). Pick a reference tangent t₁ = normalize(cross(n_hat, [0, 0, 1])) (or fallback [0, 1, 0] if n_hat ≈ [0, 0, 1]); t₂ = cross(n_hat, t₁). Now re-express the position offset δp = p - p_centroid in this local frame as {δp·t₁, δp·t₂, δp·n_hat} and use these 3 frame-local coordinates instead of (or alongside) world (x, y, z). Similarly, the model's output WSS prediction in (τ_x, τ_y, τ_z) world frame can be decoded in the local frame as (τ·t₁, τ·t₂, τ·n) then projected back to world. This is REPRESENTATION, not CONSTRAINT — gradients flow freely through the linear projection; the model can learn whatever output it wants.

**Why this is NOT in a closed class**:
- NOT tangency-imposition: no hard projection of the output. The model predicts raw (τ·t₁, τ·t₂, τ·n) and all three are supervised. Gradients are never zeroed.
- NOT curvature-aware features: no curvature tensor, no Gaussian/mean curvature. Uses only the existing surface normal, which is already in the input feature vector.
- NOT loss-balancing: no change to the loss function.
- It is a change to the INPUT REPRESENTATION and optionally the OUTPUT PARAMETERIZATION.

**Why it might help WSS specifically**:
WSS τ is a tangential vector field — by construction τ·n = 0 (zero wall-normal component). Yet the model currently predicts in world (x, y, z) frame where τ_z (vertical) and τ_x (streamwise) are entangled with normal direction depending on panel orientation. Expressing inputs in the local tangent frame gives the model a consistent coordinate reference for each surface patch, analogous to how humans decompose BL quantities into wall-normal / streamwise / spanwise. The SE(3)-equivariant WSS prediction literature (arXiv 2212.05023, Dalton 2022) shows that rotating WSS inputs with the mesh achieves ~7.6% approximation error vs ~34% for non-equivariant baselines on hemodynamic surfaces. The Intrinsic Vector Heat Network (arXiv 2406.09648) shows that basis-invariant manifold operations systematically outperform frame-dependent baselines on surfaces.

**Falsifiable prediction**:
- Primary: test_WSS improves ≥ 0.15pp vs H112 baseline (6.752% → ≤ 6.60%).
- Mechanism observable: the τ_z (vertical WSS) component should show the largest gain vs the baseline, because τ_z panels are the ones whose world-frame representation is most misaligned with local tangent (e.g. near-horizontal roof panels where n_hat ≈ [0,0,1] and the world-z axis IS the normal, so τ_z ~0 by definition but the model must learn this implicitly).
- Falsifying result: if val_WSS_z improves less than val_WSS_x, the frame-alignment explanation is incorrect.

**Implementation effort**: Medium.
- Compute t₁, t₂ from existing `nx, ny, nz` features in the data loader or model forward pass. ~15 lines.
- Replace or augment the (x, y, z) input channels with (δp·t₁, δp·t₂, δp·n). Experiment A: input-only (simplest, ~15 lines, zero architectural change). Experiment B: input + output frame change (decode τ in local frame, project to world for loss computation).
- Start with Experiment A (input-only) to isolate the representation effect.

**Risk class**: Medium.
- Known risk: t₁ direction is arbitrary modulo rotation in the tangent plane (degenerate if n_hat = [0,0,1]). Use consistent orientation convention (e.g. always project to [0,0,1] fallback [0,1,0]) and keep t₁ continuous across the mesh.
- The model's attention operates on positional inputs; changing their frame may affect how well learned positional encodings transfer. Monitor val_SP and val_VP for regression.

**Literature support**:
- arXiv 2406.09648 (Sharp et al., Intrinsic Vector Heat Networks, 2024): basis-invariant vector diffusion on triangle meshes; explicitly shows that frame-invariant representations outperform frame-dependent baselines for surface vector fields.
- arXiv 2212.05023 (Dalton et al., 2022): SE(3)-equivariant graph convolution on triangular meshes achieves 7.6% relative error for WSS prediction vs baseline; WSS rotates with the mesh.

---

### Rank 2 — H-B: Two-Stage WSS Magnitude + Direction Decode (Within Existing Single Head)

**One sentence**: Within the existing single surface head, decompose the WSS output into (a) log-magnitude log‖τ‖ and (b) angular direction (θ, φ) in the local tangent frame, train both via separate loss terms, and reconstruct (τ_x, τ_y, τ_z) at inference.

**Specific mechanism**:
The surface head currently outputs 4 channels: [cp, τ_x, τ_y, τ_z]. Replace the τ channels with [log‖τ‖, cos θ, sin θ, sin φ] where ‖τ‖ = norm of shear vector, θ = angle in tangent plane (azimuthal), φ = wall-normal tilt angle. The loss for WSS becomes L_WSS = α·MSE(log‖τ‖, log‖τ_gt‖) + β·L_angle(θ, θ_gt) + γ·L_norm_tilt(φ, 0). The normal tilt term penalizes φ ≠ 0, which encodes the physical prior that WSS is tangential — but as a SOFT LOSS TERM not a hard constraint, so gradients always flow. At inference, reconstruct τ_world = exp(log‖τ‖) · (cos θ · t₁ + sin θ · t₂).

**Why this is NOT in a closed class**:
- NOT tangency-imposition: no hard projection; the model predicts φ as a free variable; the loss merely penalizes large φ values with weight γ (tunable to 0 as ablation).
- NOT WSS-dedicated head: same output head, just different output parameterization of the 3 WSS channels. No added module. The head architecture is unchanged.
- NOT curvature features: no new input.
- NOT loss-balancing across tasks: the restructuring is internal to the WSS output channels within the existing loss term.

**Why it might help WSS specifically**:
The magnitude and direction of WSS are physically distinct quantities governed by different physics — magnitude reflects BL thickness and pressure gradient magnitude; direction is aligned with near-wall streamlines. Predicting raw (τ_x, τ_y, τ_z) forces the model to jointly handle both simultaneously in Cartesian space. Log-magnitude decomposition is standard practice in regression for heavy-tailed positive quantities (prevents mean-regression bias for low-WSS regions). The rotation representation discontinuity literature (Bregier 2021, Abbruzzese 2022) shows that predicting continuous angular representations (cos θ, sin θ) systematically outperforms predicting raw τ_x/τ_y on rotation-like quantities. The magnitude term benefits from log-space training because WSS spans 3–4 orders of magnitude across the vehicle surface.

**Falsifiable prediction**:
- Primary: test_WSS improves ≥ 0.10pp vs H112 (6.752% → ≤ 6.65%).
- Mechanism observable: val_WSS_z should improve most (τ_z on near-horizontal panels is the hardest direction to predict in world frame but easiest in tangent-frame angle θ).
- Quantitative prediction: the log-magnitude ablation alone (set β=γ=0; only use log‖τ‖ + raw direction) should improve WSS even without the angular decomposition, because log-space magnitude prediction is the lower-risk component.
- Falsifying result: if log-magnitude-only ablation shows no WSS improvement, the raw-space regression is not the bottleneck and this hypothesis is closed.

**Implementation effort**: Low-to-medium.
- Output reparameterization: ~30 lines in model forward + loss.
- Two ablations: (1) log‖τ‖ only (simplest, most conservative), (2) full log‖τ‖ + (cos θ, sin θ, φ) decomposition.
- Start with ablation 1; if it helps, proceed to ablation 2.
- Key hyperparameter: α/β/γ balance. Start with α=1.0, β=1.0, γ=0.1 (weak normal-tilt penalty).

**Risk class**: Medium.
- Angle discontinuity at ‖τ‖ → 0: use log‖τ‖ with a small epsilon floor (ε=1e-6) to avoid log(0). Angular predictions become undefined at τ=0 but DrivAerML surfaces should not have exact zero-WSS stagnation on training samples.
- The existing loss scale for WSS may need recalibration when switching to log-space; monitor for divergence in first 5k steps.
- Stagnation-point regions (vehicle front) have near-zero WSS magnitude — these regions may worsen under log-space with an ε floor. Monitor per-region error.

**Literature support**:
- Bregier (2021), "Deep Regression on Manifolds: A 3D Rotation Case Study": shows atan2-based continuous angle representations outperform raw Cartesian decomposition for rotation-like regression.
- Neural Vector Fields (Ben-Shabat et al., CVPR 2023): encodes both distance and direction as separate outputs for point cloud tasks; shows that decomposed decode improves accuracy on direction-sensitive targets.
- DoMINO (Romero et al., 2024, Nvidia): multi-scale point cloud for CFD surrogates; uses magnitude-separated features for WSS.

---

### Rank 3 — H-C: SDF-Gradient-Stratified Training Point Sampling

**One sentence**: Weight training-time surface point sampling probability proportional to the local boundary-layer thickness proxy |∇SDF|² so that each training batch oversamples high-WSS-variance surface regions.

**Specific mechanism**:
The model currently uses replacement sampling during training, sampling ~63% unique surface points per epoch with approximately uniform probability. For WSS prediction, high-variance regions (wheel arches, A-pillar, door gaps, hood-roof junction) are systematically underrepresented relative to large flat panels (doors, roof center) which contribute little WSS error. Compute a per-point sampling weight w_i = softmax(|∇SDF_i|² / T) where |∇SDF_i| is estimated from the near-wall SDF gradient (already available for volume points; approximate for surface points as distance to nearest mesh node from the volume grid). T is a temperature hyperparameter. Oversample high-gradient points; keep total number of sampled points identical. No architectural change; no input feature change; no loss change.

**Why this is NOT in a closed class**:
- NOT curvature-aware input features: sampling weight is based on SDF gradient (a quantity from the volume field), not surface curvature. No new feature added to the input vector.
- NOT loss-balancing: no change to the loss function; per-point MSE is identical. Only the sampling distribution changes.
- NOT curvature features: SDF gradient is a volumetric quantity reflecting boundary-layer structure, fundamentally different from surface Gaussian/mean curvature.

**Why it might help WSS specifically**:
CFD turbulence closure literature consistently shows that WSS estimation error concentrates in separated flow regions and pressure-gradient-dominated patches (Spalart 2009, Rumsey 2022). These regions are geometrically sparse (small area fraction) but contribute disproportionately to the relative L2 error. Uniform point sampling means the model receives very few gradient updates from these regions per epoch. Stratified sampling to oversample high-gradient regions is a standard data-centric technique in active learning and physics-informed ML (Weng et al., 2022: importance-weighted collocation for PINNs). For DrivAerML specifically, wheel arches are known to be the hardest region (highest WSS_z error) — these regions also have the steepest SDF gradients.

**Falsifiable prediction**:
- Primary: test_WSS improves ≥ 0.08pp vs H112 (6.752% → ≤ 6.67%).
- Mechanism observable: val_WSS_z should improve more than val_WSS_x (wheel arch / vertical panel patches drive the z-component error; these are the regions being oversampled).
- Secondary observable: val_SP should NOT degrade significantly (sampling change is task-agnostic — SP also benefits from better coverage of high-gradient regions).
- Falsifying result: if val_WSS_z and val_WSS_x improve proportionally (no z-dominance), the mechanism is not operating as hypothesized.

**Implementation effort**: Low.
- Compute per-point SDF gradient proxy from existing volume SDF data: ~20 lines in data loader preprocessing.
- Modify the sampling distribution in the training loop: ~10 lines.
- Key hyperparameters: temperature T (start T=0.5; T→∞ = uniform; T→0 = deterministic top-k oversampling), oversampling ratio (start at 2× for high-gradient quartile).
- Zero architectural change; zero inference overhead.

**Risk class**: Low.
- Main risk: if high-gradient regions coincide with geometrically complex areas with noisy surface normals, oversampling them may increase training instability. Monitor training loss variance in first 5k steps.
- The SDF gradient proxy computed from volume SDF may not perfectly align with surface boundary-layer gradient. Ablation: try uniform sampling as control vs stratified.

**Literature support**:
- Weng et al. (2022): importance-weighted collocation for PINNs; shows that oversampling high-residual regions consistently reduces test error on PDE-constrained problems.
- Romero et al., DoMINO (2024, Nvidia): multi-scale point cloud sampling for CFD surfaces; identifies that wheel-arch regions drive WSS error when uniformly sampled.
- Adaptive collocation / active learning sampling (Deshpande 2024): systematic oversampling of difficult regions in physics-constrained ML reduces generalization error on complex geometries.

---

### Rank 4 — H-D: Volume-to-Surface SDF-Gated Cross-Attention for Near-Wall Feature Propagation

**One sentence**: For each surface attention slice, add a lightweight cross-attention layer that reads from the K nearest volume points (selected by SDF proximity), allowing the model to propagate near-wall velocity-gradient information to surface WSS predictions.

**Specific mechanism**:
The Transolver architecture processes surface and volume tokens in separate branches. WSS at a surface point is physically governed by the velocity gradient ∂u/∂n|_{wall} — information that lives in the volume just above the wall, not on the surface itself. Add a cross-attention module: for each surface token i, attend over the K=8 nearest volume tokens (by |SDF_j| ≤ δ, with δ = 0.05 * L_ref, the near-wall shell). Surface token query, volume token keys/values. Gating: weight each volume key by exp(-|SDF_j| / σ) to concentrate attention on the near-wall shell. This is a single cross-attention layer inserted after the last surface self-attention block, before the surface decode. The volume branch remains unchanged; only the surface decode path changes.

**Why this is NOT in a closed class**:
- NOT curvature-aware features: volume SDF features are volumetric, not surface curvature.
- NOT WSS-dedicated head: the cross-attention module is prepended to the existing surface head output; no separate decode module added.
- NOT tangency-imposition: no constraint on the output.
- NOT loss-balancing: no change to the loss.
- It is a cross-modal attention mechanism that uses VOLUME information to supplement SURFACE prediction — a fundamentally different architectural direction from any closed class.

**Why it might help WSS specifically**:
WSS = μ · (∂u/∂n)|_{wall}. The most direct WSS predictor is the near-wall velocity gradient, which lives in the first few mm of the boundary layer — exactly the volume points with the smallest |SDF| values. The current architecture cannot access this information directly; the surface branch only sees surface geometry. Cross-modal attention is the architecturally principled way to inject this information. The AMSPINN paper uses anisotropic multi-head attention that explicitly penalizes wall-normal differences more than streamwise, validating the concept that volume-to-surface information flow helps WSS. AB-UPT (arXiv 2502.09692) for DrivAerML uses branched architecture with shared representations between surface and volume — showing the community is converging on cross-modal approaches.

**Falsifiable prediction**:
- Primary: test_WSS improves ≥ 0.12pp vs H112 (6.752% → ≤ 6.63%).
- Mechanism observable: test_VP should also improve (the volume branch now receives implicit feedback through the cross-attention gradient — the volume encoder learns to produce better near-wall representations).
- If test_VP does NOT improve alongside test_WSS, the volume-branch information is not being utilized meaningfully (mechanism fails).
- Falsifying result: test_WSS improves but test_VP regresses — would suggest the cross-attention is extracting spurious volume features rather than near-wall gradient information.

**Implementation effort**: High.
- Requires modification of the Transolver forward pass to implement cross-attention between surface and volume token streams.
- SDF-proximity neighbor lookup is O(N_surface × N_volume) naively; need KD-tree or pre-computed K-NN index for the near-wall shell. ~100 lines.
- Key hyperparameters: K (number of volume neighbors, start K=8), δ (SDF cutoff, start 0.05 × L_ref), σ (SDF gating bandwidth), cross-attention heads (start 4).
- High implementation complexity relative to the other hypotheses; run AFTER H-A and H-B to get early signal on simpler mechanisms.

**Risk class**: Medium-high.
- Memory: K=8 cross-attention per surface point adds ~8× memory for the cross-attention keys/values. Monitor VRAM budget.
- The near-wall volume point density in the DrivAerML dataset may be insufficient for meaningful cross-attention in some regions (adaptive mesh refinement near walls is dataset-specific).
- If volume SDF points are not densely distributed in the near-wall shell (<0.05 L_ref), the cross-attention will degenerate. Pre-validate the volume point density distribution near the surface before launching this experiment.

**Literature support**:
- AMSPINN (2024): anisotropic multi-head attention penalizing wall-normal differences more than streamwise for turbulent flow; cross-attention between volume and surface branches.
- arXiv 2502.09692 (AB-UPT, 2025): Anchored-Branched Universal Physics Transformers for DrivAerML; branched architecture with cross-modal feature sharing between surface and volume.
- Transolver original (Wu et al., 2024): grouped slice-attention on physics states; the architecture naturally supports extension to cross-modal attention by treating volume slices as additional key/value sources.

---

## Decision Tree

```
Start: test_WSS = 6.752% (H112 baseline, gap 0.902pp to SOTA)

Launch H-A (input frame encoding) + H-C (stratified sampling) in parallel
(both are low-medium effort, architecturally clean, clearly outside all closed classes)

H-A result:
  SUCCESS (test_WSS ≤ 6.60%, ≥0.15pp gain):
    → Launch H-A + H-B compound (frame encoding + magnitude/direction decode)
    → H-A mechanism confirmed; layer B on top
  PARTIAL (0.05–0.15pp gain, val_WSS_z improves as predicted):
    → Still merge (compound gains); proceed to H-B as additive
  FAIL (no gain OR val_WSS_z does not improve more than val_WSS_x):
    → Close H-A; frame-alignment explanation falsified
    → Mechanism constraint: representation change alone insufficient
    → Accelerate H-D (cross-modal) as next architectural lever

H-C result:
  SUCCESS (test_WSS ≤ 6.67%, ≥0.08pp gain):
    → Merge; combine with H-A if H-A also succeeded
  FAIL (no gain OR val_WSS_z does NOT dominate):
    → Sampling distribution not the bottleneck
    → Data-centric tier ruled out; focus on architecture (H-A, H-D)

If both H-A and H-C succeed:
  → Launch H-A + H-C compound (frame encoding + stratified sampling)
  → Then layer H-B on top of winners

H-B result (after H-A baseline established):
  SUCCESS (additional ≥0.10pp on top of H-A):
    → This is the direction to push: frame encoding + decomposed decode
    → H-D (cross-modal) becomes optional enhancement
  LOG-MAGNITUDE-ONLY ablation fails:
    → Close H-B; raw-space regression not the bottleneck; Cartesian τ is fine

H-D launch condition:
  → Launch ONLY if H-A + H-B + H-C combined have closed less than 0.4pp of the 0.902pp gap
  → H-D is the high-effort, high-ceiling bet; do not run before confirming simpler mechanisms fail
  → Pre-validate near-wall volume point density before launching

Final compounding target:
  H-A (frame encoding) + H-B (decomposed decode) + H-C (stratified sampling) + capacity axis (H120/H118)
  Projected additive: 0.15 + 0.10 + 0.08 + (capacity axis TBD) ≈ 0.33pp minimum
  Target: test_WSS ≤ 6.42% as nearer milestone; 5.85% requires continued stacking
```

---

## Stop Conditions

- H-A: if val_WSS_z does not improve more than val_WSS_x after implementing frame encoding → falsified. Close without further ablation.
- H-B: if log-magnitude-only ablation (zero angular decomposition) shows no WSS improvement → close the magnitude/direction decode class.
- H-C: if stratified sampling experiment shows val_WSS_x and val_WSS_z improve proportionally (not z-dominated) → sampling distribution is not the bottleneck. Close.
- H-D: if near-wall volume point density is < 5 points per surface point within the 0.05 L_ref shell → abort before training begins (architecture assumption invalid on this dataset).

---

## Research State Update

**Current best explanation for 0.902pp WSS gap:**
The model predicts WSS in world Cartesian frame (τ_x, τ_y, τ_z) without any explicit coordinate-frame alignment to surface geometry. For panels where the surface normal has a large z-component (roof, hood), τ_z ≈ 0 by physics but the model must learn this implicitly. This representation mismatch is the most likely remaining bottleneck given that: (a) capacity scaling (H120 depth-6) has been confirmed to help but is not sufficient alone (current WSS gap 0.9pp with depth-6 still in flight), (b) loss-balancing (H113) has been falsified as a mechanism, (c) the most successful intervention so far (Y-mirror / H116) is a data-level symmetry augmentation, suggesting the model benefits from symmetry-aware structure but the baseline lacks coordinate frame alignment.

**Open uncertainties:**
1. How much of the WSS gap is due to representation mismatch (addressable by H-A/H-B) vs insufficient capacity (addressable by H122 capacity stack) vs insufficient data augmentation?
2. Is the near-wall volume information in DrivAerML dense enough to support meaningful cross-attention for H-D?
3. Does stratified sampling improve WSS uniformly or only in wheel-arch/separated-flow regions?

**Evidence base:**
- H113 diagnostic: SP plateau HARDNESS-BOUND → loss-balancing closed.
- H116 Y-mirror: B PARTIAL on test_WSS → data augmentation (symmetry) positive signal.
- H120 depth-6 EP3: capacity-axis dominance confirmed.
- arXiv 2406.09648, 2212.05023: surface-intrinsic frame encoding is the active frontier for manifold WSS prediction.
