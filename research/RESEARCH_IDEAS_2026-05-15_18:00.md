# Wave 30 Research Ideas — Architectural Hypotheses for the τ_z Bottleneck
**Generated:** 2026-05-15 18:00
**Context:** EIGHTFOLD-CONFIRMED structural finding: τz/τx ratio 1.50–1.57 is an attractor across all loss-side and data-side interventions. Wave 30 pivots to backbone-level architectural changes.

---

## Structural Finding Summary

Eight mechanistically distinct experiments targeting τ_z have all converged to the same ratio band:
1. τ_z focal loss weight (PR #1109) — null
2. Learnable WSS channel weights (PR #1114) — null
3. Normal-frame WSS prediction (PR #1094) — marginal
4. Tangent frame features (PR #1096) — null
5. Doubled slice count to 256 (PR #1100) — null
6. Curvature-based spatial loss (PR #1085) — null
7. Per-channel decoder heads (PR #1116) — null/regressed
8. Per-axis WSS magnitude decomp (PR #1133) — in flight (frieren)

**Conclusion:** The bottleneck is architectural, not loss-formulated or data-side. The representation layer cannot cleanly separate vertical shear from streamwise shear because the backbone has no mechanism to do so. The target is τz/τx ≤ 1.40, which requires a genuine representational change.

---

## Key Architecture Gaps (from model.py analysis)

1. **StringSeparableEncoding** already gives x, y, z separate per-axis spectral parameters — axis conflation is NOT in the positional encoding layer specifically.
2. **Surface normals (nx,ny,nz)** are passed as raw features via a linear projection — they do NOT receive spectral encoding. This is unexplored.
3. **Slice-attention** groups ALL surface tokens together regardless of normal orientation — no axis-aware or τ-direction-aware grouping.
4. **Surface decoder head** is a single shared 2-layer MLP → [cp, τx, τy, τz] — no structural inductive bias separating horizontal from vertical shear.
5. **No cylindrical/vehicle-frame coordinate experiments** have ever been run across 523 PRs.
6. **No pretraining or auxiliary task experiments** exist in the entire programme history.

---

## CATEGORY 1: Coordinate-System / Input Representation Changes

### H1: Cylindrical Vehicle-Frame Coordinates (r, θ, z)
**Title:** cylindrical-coords-rtheta-z-input

**Mechanism:**
Replace the raw Cartesian positional inputs (x, y, z) fed to StringSeparableEncoding with cylindrical coordinates (r, θ, z) where r = sqrt(x²+y²) is the lateral distance from the vehicle centerline, θ = atan2(y, x) is the circumferential angle, and z is the vertical axis. The vehicle has bilateral symmetry in the x-z plane, so τ_z corresponds to the purely vertical cylinder axis while τ_x and τ_y blend cylindrical and radial components. In cylindrical frame, τ_z is the stress component ALIGNED with the z-axis of the coordinate system, while in Cartesian frame τ_z must be inferred from a projection mixing all three axes in the backbone's learned representation. Giving the network a coordinate system where one axis aligns with the problematic shear direction may reduce the representational burden.

**Theoretical Basis:**
Cylindrical coordinates are standard in vortex-dominated and rotational flows (Jameson, 1991; Pope, Turbulent Flows, 2000). In CFD, surface quantities on bluff bodies with streamwise symmetry are routinely decomposed in cylindrical coordinates because the curl operator separates cleanly. Recent physics-aware networks (EquiFormer, Clebsch-Gordan networks) show that coordinate frames aligned with the dominant symmetry group of the problem consistently outperform isotropic Cartesian representations on directional output quantities.

**Implementation Sketch:**
- In `model.py`, inside `SurfaceTransolver.__init__`, add a flag `self.use_cylindrical_coords` (driven by `--use-cylindrical-coords` CLI flag in `train.py`)
- In `SurfaceTransolver._encode_group`, before calling `pos_embed` and `string_sep`, transform pos coordinates:
  ```python
  if self.use_cylindrical_coords:
      x_cyl = pos[:, :, 0]
      y_cyl = pos[:, :, 1]
      z_cyl = pos[:, :, 2]
      r = torch.sqrt(x_cyl**2 + y_cyl**2 + 1e-8)
      theta = torch.atan2(y_cyl, x_cyl)
      pos = torch.stack([r, theta, z_cyl], dim=-1)
  ```
- Apply this transformation to BOTH surface and volume positional inputs in their respective `_encode_group` calls
- The rest of the encoding pipeline (StringSeparableEncoding) is unchanged — it still learns per-axis spectral parameters, but now axis 0 = radial, axis 1 = circumferential, axis 2 = vertical
- Estimated: ~25 LOC in `model.py`, ~10 LOC in `train.py`
- No change to data loading, output heads, or loss

**Falsifiability:**
- Success: τz/τx ratio drops below 1.40 AND val_abupt improves vs baseline (6.228%)
- Fail: τz/τx ratio stays in 1.50–1.57 band → coordinate frame is not the bottleneck, ruling out geometric alignment as the mechanism
- Diagnostic: per-axis WSS error breakdown in W&B; if r and θ channels improve while z stays flat, the hypothesis is partially supported

**Why This Targets the Structural Finding:**
The structural finding is that τ_z is persistently harder than τ_x regardless of loss or data manipulation. This implies the backbone representation aligns better with the streamwise axis than the vertical axis — a Cartesian geometry effect. In Cartesian frame, the vehicle body extends primarily along x with moderate z extent, so the token distribution is denser in x. Cylindrical frame rebalances this: z is now a dedicated coordinate axis with no interference from lateral geometry, and the StringSeparableEncoding's per-axis spectral parameters will learn purely vertical frequency content for the z axis rather than mixing vertical and lateral content.

---

### H2: Normal-Vector Spectral Encoding (StringSeparable on Normals)
**Title:** normal-spectral-encoding-stringsep-normals

**Mechanism:**
Apply StringSeparableEncoding to the surface normal vector (nx, ny, nz) using the same per-axis learnable log_freq and phase parameterization currently used only for (x, y, z) positions. Currently, normals go through a raw linear projection in `_encode_group`. Surface normals encode orientation information that is critical for predicting directional shear: a surface patch with nz ≈ ±1 (horizontal roof or floor) has fundamentally different τ_z characteristics than a patch with nz ≈ 0 (vertical side wall). Giving the network learnable spectral basis functions over normal space lets it represent "normal-frequency" features — the idea that patches with similar normal orientation should have similar shear projections.

**Theoretical Basis:**
Fourier features for directional quantities on the sphere (spherical harmonics) are a classical tool in computational physics. The connection between surface normals and shear stress tensor projections is:
  τ_i = Σ_j σ_ij * n_j
where σ_ij is the viscous stress tensor. This is a linear map from normals to shear, but only for the LOCAL shear tensor; the global flow interaction means the effective map is highly nonlinear. However, spectral encoding of the normal vector should capture the dominant harmonic structure of this map better than a raw linear projection. Recent work on geometric deep learning (Batzner et al., NequIP 2022; Brandstetter et al., Clifford Neural Layers 2023) shows that spectral treatment of directional inputs consistently outperforms linear projection when the output has directional dependence.

**Implementation Sketch:**
- In `SurfaceTransolver.__init__`, add a second `StringSeparableEncoding` instance `self.normal_string_sep` with `in_dim=3` (for nx,ny,nz) and configurable `num_features` (default 16, same as current pos encoding)
- In `SurfaceTransolver._encode_group` (surface path only — volume has SDF not normals):
  ```python
  normals = x[:, :, 3:6]  # nx, ny, nz
  area = x[:, :, 6:7]
  # Replace: features = cat([x[:,:,space_dim:], string_sep(pos)])
  # With:
  normal_enc = self.normal_string_sep(normals)
  features = cat([area, string_sep(pos), normal_enc])
  ```
- Update `in_features` dimension calculation accordingly in `project_features` linear
- The volume path is unchanged (its extras are just SDF, not normals)
- Estimated: ~35 LOC in `model.py`, ~5 LOC in `train.py` for the flag `--normal-spectral-encoding`

**Falsifiability:**
- Success: τz/τx < 1.45 AND val_abupt improves
- Fail: ratio stays above 1.47 → normal orientation encoding is not the limiting factor
- Diagnostic: ablation — if spectral normal encoding helps on τ_z but not τ_x, the mechanism is confirmed (normals encode z-direction orientation information)

**Why This Targets the Structural Finding:**
The τ_z bottleneck exists because τ_z requires the network to reason about vertical shear, which depends critically on the local normal orientation (whether a patch faces up/down vs sideways). Currently the normals feed through a linear projection, which can represent the mean but not the frequency structure of the normal-to-shear mapping. Spectral encoding over normal space gives the backbone representations that explicitly encode "this patch has predominantly vertical-facing normal" vs "this patch has predominantly sideways-facing normal" in a rich frequency basis that can interact with the τ_z prediction pathway.

---

## CATEGORY 2: Attention Mechanism Modifications

### H3: Normal-Aligned Slice Groups (Axis-Aware Soft Assignment)
**Title:** normal-aligned-slice-groups-axis-aware

**Mechanism:**
Modify the slice-attention assignment in `TransolverAttention` so that each slice token is biased toward tokens with similar surface normal orientation. Currently, slice assignment is based purely on the learned projection `in_project_slice(x_mid)`, which can learn any grouping but in practice groups by spatial proximity and feature similarity without any explicit orientation bias. Add a normal-orientation term to the slice logits:

```python
# Current: logits = in_project_slice(x_mid) / temperature
# Proposed: logits = in_project_slice(x_mid) / temperature + alpha * normal_align_bias(normals_mid)
```

Where `normal_align_bias` projects normals into slice space, making it geometrically easier for the softmax to form orientation-coherent slices. The intuition: a slice token that represents "upward-facing surfaces" will have all the vertical-normal patches attend to it, concentrating τ_z-relevant information in that token for efficient QKV attention to compute vertical shear interactions.

**Theoretical Basis:**
Point cloud transformers with geometric attention (PointBERT, Point-MAE, 2022; DGCNN 2019) consistently show that grouping/sampling strategies based on local geometry outperform purely feature-based grouping for shape-dependent outputs. In CFD specifically, Transolver's original paper (Wu et al., 2024) showed that "physics-informed slicing" outperforms random token grouping — the same principle applies here but for ORIENTATION rather than spatial position. The connection to τ_z: WSS direction is determined by flow direction projected onto the surface tangent plane, so tokens with similar normal orientation should have correlated τ-direction regardless of spatial position.

**Implementation Sketch:**
- Add `self.normal_slice_bias` linear layer in `TransolverAttention.__init__`: `Linear(normal_dim, heads * num_slices)`
- Modify `forward` to accept optional `normals` argument (passed down from `SurfaceTransolver.forward`)
- In `TransolverAttention.forward`:
  ```python
  slice_logits = self.in_project_slice(x_mid) / self.temperature
  if normals is not None and self.normal_slice_bias is not None:
      normal_bias = self.normal_slice_bias(normals)  # [B, N, heads*slices]
      normal_bias = normal_bias.view(B, N, self.heads, self.num_slices)
      slice_logits = slice_logits + self.normal_slice_alpha * normal_bias
  slice_weights = softmax(slice_logits, dim=-2)
  ```
- In `SurfaceTransolver.forward`, extract normals from surface_x and pass to backbone layers for surface token positions
- `--normal-slice-alpha` CLI flag controls the initial bias weight (default 0.1, learnable scale)
- Estimated: ~50 LOC in `model.py`, ~5 LOC in `train.py`

**Falsifiability:**
- Success: τz/τx drops below 1.43 AND val_abupt improves
- Fail: τz/τx stays above 1.47 → orientation-coherent slicing is not the missing inductive bias
- Diagnostic: visualize slice weight distribution — do normal-aligned slices show lower entropy (more orientation-specific grouping) compared to baseline?

**Why This Targets the Structural Finding:**
The τ_z bottleneck is consistent with the slice-attention mechanism failing to form orientation-specific token groups. Without an orientation prior, slices likely form by spatial proximity, mixing upward-facing (τ_z-dominant) and sideways-facing (τ_x-dominant) surfaces in the same slice token. This averages out the directional signal. By biasing slice assignment toward orientation coherence, we create "τ_z-specialized" and "τ_x-specialized" slice tokens that can independently learn the correct shear magnitude for their orientation class — directly addressing the structural origin of the ratio.

---

### H4: Dedicated τ_z Slice Partition (Hard Orientation Routing)
**Title:** hard-normal-routing-dedicated-tauZ-slices

**Mechanism:**
Partition the `num_slices` budget into two groups: `num_slices_xy` slices for predominantly horizontal/lateral-normal surfaces (|nz| < threshold) and `num_slices_z` slices for predominantly vertical-normal surfaces (|nz| ≥ threshold). Use hard routing (Gumbel-softmax or simple top-k masking) to assign each surface token to its orientation group, then run independent slice-attention within each group. This is essentially a mixture-of-experts over surface orientation.

**Theoretical Basis:**
Hard routing (Switch Transformer, Fedus et al. 2022; Expert Choice, Zhou et al. 2022) outperforms soft routing when the problem has discrete structural modes. In aerodynamics, surface normals DO have a discrete structure: the vehicle has a roof (nz ≈ +1), underbody (nz ≈ -1), and sides (nz ≈ 0), and the WSS profiles on these regions are qualitatively different. This is closer to a "routing problem" than a "mixing problem." The Transolver paper itself notes that physics slices correspond to physical regimes — explicitly routing by normal orientation extends this principle to orientation regimes rather than spatial regimes.

**Implementation Sketch:**
- Add `self.nz_threshold = 0.5` and `self.z_slice_fraction = 0.25` (give 25% of slices to z-normal surfaces)
- In `TransolverAttention.forward`, compute |nz| for each token, create a binary mask `is_z_surface`
- Run separate slice-attention sub-calls on the two groups using slice count proportional to `z_slice_fraction`
- Concatenate results back via the mask
- `--z-slice-fraction 0.25` and `--nz-routing-threshold 0.5` CLI flags
- Estimated: ~70 LOC in `model.py`, ~10 LOC in `train.py`
- Note: this is more invasive than H3 but more directly testable

**Falsifiability:**
- Success: τz/τx < 1.40 (reaching the target directly) AND val_abupt improves
- Fail: τz/τx unchanged → hard orientation routing is not the missing structural element
- Diagnostic: compare τ_z error on high-|nz| surfaces vs low-|nz| surfaces before and after — if routing helps only on high-|nz| surfaces, the mechanism is confirmed

**Why This Targets the Structural Finding:**
This is the most direct architectural test of the orientation-routing hypothesis: do surfaces with nz ≈ ±1 (roofs, floors) suffer more τ_z error than surfaces with nz ≈ 0 (sides)? If so, dedicated attention capacity for z-normal surfaces should directly reduce the error. This is falsifiable in a way that soft methods (H3) are not — if hard routing doesn't help, it rules out the orientation-capacity-sharing hypothesis entirely.

---

## CATEGORY 3: Backbone Architectural Changes

### H5: Y-Architecture — Dual Backbone Branch for WSS vs Pressure
**Title:** y-arch-dual-backbone-wss-pressure-branches

**Mechanism:**
Split the backbone after the first encoder layer into two parallel transformer stacks: one branch specializes on scalar pressure (cp) and one branch specializes on WSS (τ_x, τ_y, τ_z). The two branches share initial feature extraction (first layer) but have independent intermediate layers (layers 2-4) with separate slice-attention parameters. Final layer can optionally merge or remain separate. This creates a structural Y-architecture where the WSS branch can develop representations optimized for directional shear without being contaminated by the pressure-optimal representations that dominate the shared backbone.

**Theoretical Basis:**
Y-networks for multi-task learning (Misra et al., Cross-Stitch Networks, CVPR 2016; Vandenhende et al., Multi-Task Learning Survey 2021) consistently outperform single-backbone multi-task models when tasks have structurally distinct optimal representations. Pressure and WSS in aerodynamics are fundamentally different: pressure is a scalar potential field governed by the Bernoulli-type equations (irrotational contribution), while WSS is a vector gradient of velocity at the wall (rotational contribution). These correspond to different physical modes of the flow. Giving them separate backbone branches allows each to develop appropriate spectral and spatial inductive biases independently.

**Implementation Sketch:**
- In `SurfaceTransolver.__init__`, replace `self.blocks = ModuleList([TransformerBlock(...) for _ in range(depth)])` with:
  ```python
  self.shared_blocks = ModuleList([TransformerBlock(...) for _ in range(1)])  # first layer shared
  self.pressure_blocks = ModuleList([TransformerBlock(...) for _ in range(depth-2)])
  self.wss_blocks = ModuleList([TransformerBlock(...) for _ in range(depth-2)])
  self.final_block = TransformerBlock(...)  # optional: merge or keep separate
  ```
- In `forward`:
  ```python
  # Shared stem
  for blk in self.shared_blocks:
      x = blk(x)
  # Split
  x_pressure = x
  x_wss = x
  for blk in self.pressure_blocks:
      x_pressure = blk(x_pressure)
  for blk in self.wss_blocks:
      x_wss = blk(x_wss)
  # Merge for volume (uses combined features) or keep separate for surface heads
  ```
- Surface head: `cp` from `x_pressure`, `[τx, τy, τz]` from `x_wss`
- Volume stays on merged or pressure branch (pressure governs volume)
- `--y-arch-split-layer 1` CLI flag to control where split happens
- Estimated: ~80 LOC in `model.py`, ~10 LOC in `train.py`
- Compute: ~1.5x backbone parameters for the split layers, but same depth → similar wall-clock

**Falsifiability:**
- Success: WSS error (especially τ_z) improves while cp/volume metrics maintain, τz/τx < 1.45
- Fail: WSS error unchanged or pressure degrades → task interference hypothesis refuted, or split creates optimization instability
- Diagnostic: compare WSS branch loss vs pressure branch loss curves; if WSS branch converges faster than baseline, task separation is working

**Why This Targets the Structural Finding:**
The τ_z bottleneck may exist because the shared backbone optimizes primarily for the easiest-to-predict quantities (pressure, τ_x) and leaves WSS — especially the vertical shear — as a secondary optimization target. Separating the backbone gives WSS its own representational budget and optimization signal, breaking the regime where pressure prediction dominates the shared backbone's spectral content.

---

## CATEGORY 4: Output Head Architectural Changes

### H6: Axis-Decomposed WSS Head with Normal Projection
**Title:** axis-decomposed-wss-head-normal-projection

**Mechanism:**
Replace the shared 2-layer MLP surface head ([cp, τx, τy, τz]) with a physics-motivated factored head. Instead of predicting τ_x, τ_y, τ_z directly in global Cartesian frame, predict the WSS vector in a LOCAL surface coordinate system (tangent-1, tangent-2, normal) and then rotate back to global frame using the surface normals. The local-frame prediction is easier because: (1) the normal component of WSS is identically zero (no-penetration condition) providing a built-in constraint, (2) the two tangential components are physically isotropic in the local frame (no directional bias), so the network learns a coordinate-free magnitude and direction rather than a globally-biased directional split.

**Theoretical Basis:**
Local frame predictions are fundamental in computational mechanics and graphics. For WSS specifically, the Navier-Stokes boundary condition requires WSS to lie in the tangent plane — it has zero normal component by definition. A global Cartesian head can violate this constraint, wasting model capacity on learning to suppress the spurious normal component. Local frame prediction enforces this constraint exactly. Related: NequIP (Batzner et al. 2022), Equiformer (Liao et al. 2022), and SE(3)-Transformers (Fuchs et al. 2020) all show that equivariant/local-frame outputs consistently reduce error on directional quantities in physics simulations compared to global Cartesian outputs.

**Implementation Sketch:**
- Compute local surface frame from normals: given n, construct orthonormal basis (t1, t2, n) using Gram-Schmidt on (n, [1,0,0] or [0,1,0] if degenerate)
- In `SurfaceTransolver.__init__`:
  ```python
  self.surface_out_cp = nn.Linear(hidden_dim, 1)
  self.surface_out_wss_local = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
      nn.Linear(hidden_dim, 2)  # (wss_t1, wss_t2) in local tangent frame
  )
  ```
- In `forward`:
  ```python
  cp = self.surface_out_cp(surface_feats)
  wss_local = self.surface_out_wss_local(surface_feats)  # [B, N, 2]
  # Reconstruct global from local: wss_global = wss_local[:,:,0:1]*t1 + wss_local[:,:,1:2]*t2
  # where t1, t2 derived from surface normals
  tau_global = wss_local[..., 0:1] * t1 + wss_local[..., 1:2] * t2  # [B, N, 3]
  out = cat([cp, tau_global], dim=-1)  # [B, N, 4]
  ```
- `--local-frame-wss-head` CLI flag
- Estimated: ~60 LOC in `model.py`, ~5 LOC in `train.py`
- Key implementation detail: the local frame construction must be numerically stable (handle n ≈ [0,0,1] case carefully)

**Falsifiability:**
- Success: τz/τx drops below 1.40 (most direct test — if the bottleneck was global-frame decomposition, this should resolve it) AND val_abupt improves
- Fail: τz/τx unchanged → global-frame decomposition is not the bottleneck; the issue is upstream in the backbone representation
- Diagnostic: check if the no-penetration constraint is being violated in baseline (compute τ · n for baseline predictions); if it's non-negligible, this hypothesis has strong prior support

**Why This Targets the Structural Finding:**
The τ_z bottleneck emerges because in global Cartesian frame, the network must learn that τ_z on a side wall (where n ≈ [0,1,0]) is a TANGENTIAL component while τ_z on a roof (where n ≈ [0,0,1]) is also a TANGENTIAL component but in a different direction — the same global τ_z label maps to fundamentally different physical meanings depending on the local surface orientation. A local-frame head sees both cases as "tangential WSS magnitude" and removes this representational asymmetry. This directly explains why τ_z is harder: it requires the most orientation-conditional reasoning of the three Cartesian components.

---

## CATEGORY 5: Pretraining and Auxiliary Tasks

### H7: Normal-Prediction Auxiliary Head (Surface Orientation Pretraining)
**Title:** normal-prediction-auxiliary-pretraining

**Mechanism:**
Add an auxiliary prediction head that predicts the surface normal vector (nx, ny, nz) from the backbone representations, trained jointly with the main WSS/pressure task. The normal vector is available in the input features, so this is a reconstruction/consistency task rather than a new label requirement. The mechanism: by explicitly training the backbone to maintain orientation information throughout its layers (via the aux loss gradient signal), we prevent the backbone from discarding normal information that is critical for τ_z prediction. Additionally, pretrain for 1-2 epochs on normal prediction alone before the main task — this forces the backbone to learn orientation-sensitive representations as initialization.

**Theoretical Basis:**
Auxiliary task learning for geometric features improves backbone representations in 3D point cloud models (PointMAE, 2022; Point-BERT, 2022; GeoAE, 2022). The key insight: self-supervised pretraining on geometric properties (surface normals, curvature) improves downstream physics prediction because these geometric properties encode the same physics constraints that the main task requires. In DrivAerML specifically, the no-penetration constraint (τ · n = 0) means that predicting τ correctly IS equivalent to predicting the tangential projection, which requires an accurate internal representation of n. Auxiliary normal prediction forces the backbone to maintain this representation.

**Implementation Sketch:**
- Add `self.normal_aux_head = nn.Linear(hidden_dim, 3)` in `SurfaceTransolver.__init__`
- In `forward`, compute `normal_pred = self.normal_aux_head(surface_feats)` and normalize to unit sphere
- In `trainer_runtime.py`, add auxiliary loss term:
  ```python
  normal_loss = F.cosine_embedding_loss(
      normal_pred.reshape(-1, 3),
      surface_normals.reshape(-1, 3),
      torch.ones(B*N, device=device)
  )
  total_loss = main_loss + aux_normal_weight * normal_loss
  ```
- `--aux-normal-weight 0.1` CLI flag; `--aux-normal-pretrain-epochs 2` for warmup phase
- Estimated: ~40 LOC in `model.py`, ~30 LOC in `trainer_runtime.py`, ~10 LOC in `train.py`
- Key detail: the normal vectors are in surface_x[:, :, 3:6] — they are available as input labels without any new data requirement

**Falsifiability:**
- Success: τz/τx drops below 1.45 AND val_abupt improves; normal prediction cosine similarity should be near 1.0 by epoch 3
- Fail: τz/τx unchanged → backbone already retains sufficient normal information; auxiliary task adds no gradient signal the backbone didn't have from WSS direction (which depends on normals implicitly)
- Diagnostic: compare baseline backbone's ability to predict normals from intermediate features (zero-shot linear probe) vs after auxiliary training — if baseline normal prediction is already good, this hypothesis has low prior

**Why This Targets the Structural Finding:**
The τ_z bottleneck is consistent with the backbone learning representations that encode spatial position well (τ_x and τ_y) but fail to maintain explicit orientation information through 5 transformer layers (needed for τ_z). Auxiliary normal prediction provides a direct gradient path that rewards orientation-preserving representations at every layer, potentially breaking the gradient dynamics that currently let orientation information diffuse away in the backbone.

---

### H8: Contrastive Orientation Regularization (Orientation-Aware Loss)
**Title:** contrastive-orientation-regularization-normals

**Mechanism:**
Add a contrastive regularization loss in feature space that encourages tokens with similar surface normal orientation to have similar backbone representations, and tokens with different normals to have dissimilar representations. This is a metric learning objective over the backbone's internal representation space, separate from the main prediction loss. The idea: if the backbone representation encodes orientation information contrastively, the downstream τ_z head can easily separate "upward-facing" from "sideways-facing" tokens, reducing the ambiguity that causes the τ_z bottleneck.

**Theoretical Basis:**
Contrastive learning on geometric features improves shape understanding (ContrastScene, 2023; PointContrast, Xie et al. 2020; CSC, Hou et al. 2021). The key paper: PointContrast showed that contrastive pretraining with spatial correspondence significantly improves 3D scene understanding benchmarks. Applied to surface normals: we define positive pairs as surface tokens with cosine(n_i, n_j) > 0.9 (similar orientation) and negative pairs as tokens with cosine(n_i, n_j) < 0.1 (different orientation). The backbone representation space should cluster by orientation class because WSS prediction requires orientation-aware computation.

**Implementation Sketch:**
- Add `self.orientation_proj = nn.Linear(hidden_dim, 64)` (projection head for contrastive space)
- In `trainer_runtime.py`, after computing backbone features:
  ```python
  # Sample orientation pairs from surface tokens within each batch element
  z = F.normalize(model.orientation_proj(surface_feats), dim=-1)  # [B, N, 64]
  normal_sim = torch.bmm(normals, normals.transpose(1, 2))  # [B, N, N] cosine sim
  pos_mask = (normal_sim > 0.9).float()
  neg_mask = (normal_sim < 0.1).float()
  feat_sim = torch.bmm(z, z.transpose(1, 2))  # [B, N, N]
  contrastive_loss = -(feat_sim * pos_mask).sum() / (pos_mask.sum() + 1e-6) \
                    + F.relu(feat_sim * neg_mask + margin).sum() / (neg_mask.sum() + 1e-6)
  total_loss = main_loss + contrastive_weight * contrastive_loss
  ```
- `--orientation-contrastive-weight 0.05` and `--contrastive-margin 0.5` CLI flags
- Estimated: ~50 LOC in `trainer_runtime.py`, ~20 LOC in `model.py`, ~10 LOC in `train.py`

**Falsifiability:**
- Success: τz/τx drops below 1.45 AND val_abupt improves; feature-space normal clustering (measured by silhouette score over orientation groups) should increase vs baseline
- Fail: τz/τx unchanged → orientation clustering in feature space is not the bottleneck
- Diagnostic: this experiment produces an interpretable diagnostic — we can visualize whether backbone features cluster by orientation after training, which is informative regardless of outcome

**Why This Targets the Structural Finding:**
Contrastive orientation regularization forces the backbone to maintain a latent space where orientation is a first-class citizen. If the τ_z bottleneck arises because the backbone learns orientation-agnostic features (optimizing for the dominant streamwise flow signal), this regularization directly counteracts that by imposing an orientation-preserving structure on the latent space.

---

## Top-3 Recommendations

### Rank 1: H6 — Axis-Decomposed WSS Head with Normal Projection
**Rationale:** This is the most mechanistically direct intervention. The τ_z bottleneck has a concrete physical explanation: in global Cartesian frame, the same τ_z label has different physical meanings depending on the local surface orientation (tangential on sides, tangential-but-different on roofs). A local-frame head removes this ambiguity EXACTLY, not approximately. The implementation is clean (~60 LOC), does not change the backbone architecture (lower risk), and the falsification is precise. If τz/τx doesn't improve with a local-frame head, it definitively rules out global-frame decomposition as the bottleneck and points upstream to the backbone representation. The no-penetration check (τ · n = 0) is a free diagnostic that either strongly supports or weakens the hypothesis before a single training run. **Taste scores: Mechanistic grounding = 4 (precise physical mechanism directly targeting the bottleneck), Research-state value = 4 (either confirms or rules out global-frame decomposition entirely), Execution value = 3 (cheap to implement, head-only change, fast ablation possible).**

### Rank 2: H2 — Normal-Vector Spectral Encoding (StringSeparable on Normals)
**Rationale:** This is the most surprising gap in the current architecture: the position encoding already applies StringSeparableEncoding to (x,y,z) but the normal vector (nx,ny,nz) — which encodes orientation information critical for τ_z — gets only a linear projection. This is an obvious asymmetry. Applying the same spectral encoding to normals costs ~35 LOC and zero architecture restructuring. The mechanism is clear: richer normal representations → better orientation-conditional features → better τ_z prediction. The risk is that the existing combination of linear normal projection + spectral position encoding is sufficient and normals are already implicitly handled. This should be tried early because it's cheap and closes an obvious gap. **Taste scores: Mechanistic grounding = 3 (clear gap in architecture, plausible but not proven to be the bottleneck), Research-state value = 3 (closes a specific architectural gap), Execution value = 4 (minimal code change, fast to test, informative either way).**

### Rank 3: H5 — Y-Architecture Dual Backbone Branch
**Rationale:** Task interference between pressure and WSS is a standard multi-task learning problem and the Y-architecture is a well-validated solution. The theoretical case is strong: pressure (potential flow) and WSS (boundary layer gradient) represent fundamentally different physical modes. The main risk is optimization instability from the split (increased parameters, gradient conflicts at the merge point) and the moderate implementation complexity (~80 LOC). This is a "tier shift" experiment — bigger architectural bet — that should come after H6 and H2 have been tested. If H6 and H2 both fail, H5 is the next discriminating experiment at a higher level of abstraction. **Taste scores: Mechanistic grounding = 3 (strong theoretical basis from multi-task learning literature, but task interference unproven in this specific setting), Research-state value = 4 (would either confirm task interference as the bottleneck or rule it out, significantly updating the research map), Execution value = 2 (more invasive, harder to debug, more moving parts).**

---

## WAVE 30 ASSIGNMENT FOR TANJIRO

**Assign H6: Axis-Decomposed WSS Head with Normal Projection**

**Justification:** H6 is the highest-EV experiment because:
1. It tests the most specific and directly falsifiable mechanism for the τ_z bottleneck
2. It is the only change that ENFORCES the physical no-penetration constraint (τ · n = 0) rather than merely encouraging it through loss weighting
3. The implementation is contained in the output head (~60 LOC in model.py), minimizing risk of destabilizing the baseline recipe
4. The falsification is clean: either the local-frame head reduces τz/τx below 1.45 or it doesn't — there is no ambiguous middle ground
5. Free pre-run diagnostic: compute the no-penetration violation τ · n on baseline predictions; if it's ≥ 1% of τ magnitude, the hypothesis has strong prior support

**Pre-run diagnostic (tanjiro should run before full training):**
Load the baseline checkpoint and compute mean |τ · n| / |τ| across all validation surface points. Report this value. If it's > 0.01 (1% violation), the local-frame hypothesis is strongly supported.

**Baseline to beat:**
- val_abupt: 6.228% (tanjiro #1124)
- test_WSS: 6.727% (PR #972 baseline)
- τz/τx target: ≤ 1.40 (currently 1.50–1.57)

**Recommended training flags:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --optimizer lion --lr 9e-5 \
  --pos-encoding-mode string_separable \
  --use-qk-norm \
  --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --batch-size 4 --epochs 13 \
  --tau-z-loss-weight 2.0 --surface-loss-weight 2.0 \
  --use-surf-to-vol-xattn \
  --use-ema --ema-decay 0.999 \
  --local-frame-wss-head \
  --wandb_group wave30_local_frame_wss
```

---

## Research State After Wave 30 Launch

### Current Best Explanation
The τ_z bottleneck (ratio 1.50–1.57) is structural and originates in the decoder head: the global Cartesian frame forces the network to learn orientation-conditional τ decomposition implicitly, which it handles poorly for the vertical component. Supporting evidence: 8 loss-side and data-side interventions all null; the bottleneck is isotropic to changes in training signal, suggesting the model has simply exhausted its capacity to represent τ_z in the current parameterization.

### Evidence
- PRs #1085, #1094, #1096, #1100, #1109, #1114, #1116, #1133 (8 null experiments)
- Ratio band 1.50–1.57 consistent across all attempts
- Current best single-model: test_WSS=6.727%, test_τ_z=8.2585%

### Ruled-Out Paths
- Loss weighting (any form): confirmed null ×4
- Coordinate frame rotation (pre-head): confirmed null/marginal ×2
- Slice count scaling: confirmed null ×1
- Spatial/curvature loss: confirmed null ×1
- Extra decoder capacity (per-channel heads): confirmed null ×1

### Open Uncertainties
1. Is the bottleneck in the decoder (H6 tests this) or the backbone representation (H5 tests this)?
2. Does spectral encoding of normals provide representations sufficient for orientation-conditional τ prediction (H2 tests this)?
3. Is the τ_z error concentrated on specific surface regions (high |nz| roofs/floors) or distributed across all orientations?

### Next Discriminating Experiment
H6 (local-frame WSS head) → if τz/τx < 1.45, continue with backbone-level changes (H2, H3); if null, move to H5 (Y-architecture) or H3 (normal-aligned slices).

### Stop Condition for Wave 30 Architecture Direction
If H6, H2, H3, H4, AND H5 all fail to move τz/τx below 1.47, the hypothesis that the τ_z bottleneck is architectural is refuted. At that point, the bottleneck is likely DATA — the training distribution lacks sufficient vertical-shear diversity — and the research direction should pivot to data augmentation (synthetic vertical flow perturbations, re-weighting by surface orientation) or to accepting the current τ_z floor and targeting other metric improvements.

---

## Experiment Decision Tree

```
H6 (local-frame WSS head)
├── SUCCESS (τz/τx < 1.45, val_abupt improves)
│   ├── Merge H6
│   ├── Test H2 (normal spectral encoding) on top of H6
│   └── Test H5 (Y-arch) as additional backbone separation
└── FAIL (τz/τx > 1.47)
    ├── Bottleneck is NOT in head decomposition → head is not the bottleneck
    ├── Run H2 (normal spectral encoding) — tests backbone representation quality
    │   ├── H2 SUCCESS → run H3 (normal-aligned slices) to compound
    │   └── H2 FAIL → run H5 (Y-architecture) — task interference hypothesis
    │       ├── H5 SUCCESS → compound with H6 and H2
    │       └── H5 FAIL → ALL ARCH INTERVENTIONS FAILED → pivot to DATA direction
    └── In parallel: run H3 (normal-aligned slices) — directly tests grouping hypothesis
        (H3 and H2 are orthogonal enough to run concurrently on separate students)
```

---

## Taste Rubric Summary

| Hypothesis | Research Mode | Mechanistic Grounding | Research-State Value | Execution Value | Total |
|---|---|---|---|---|---|
| H6: Local-frame WSS head | Diagnostic | 4 | 4 | 3 | **11** |
| H2: Normal spectral encoding | Diagnostic | 3 | 3 | 4 | **10** |
| H5: Y-architecture backbone | Tier shift | 3 | 4 | 2 | **9** |
| H3: Normal-aligned slices | Diagnostic | 3 | 3 | 3 | **9** |
| H1: Cylindrical coordinates | Tier shift | 2 | 3 | 3 | **8** |
| H7: Normal prediction aux | Diagnostic | 2 | 3 | 3 | **8** |
| H4: Hard normal routing | Diagnostic | 3 | 3 | 2 | **8** |
| H8: Contrastive orientation | Tier shift | 2 | 3 | 2 | **7** |

**Confidence:** H6 and H2 — Strong evidence from physics principles and architecture analysis. H5 — Strong evidence from multi-task learning literature applied to analogous settings. H1, H3, H4, H7, H8 — Promising theory, no validation in this specific setting yet.
