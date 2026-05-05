# Research Ideas — Round 40 (2026-05-05)

**Branch:** yi  
**Merge bar:** val_abupt = 7.3914% / test_abupt = 8.7189% (PR #658)  
**Yi SOTA:** val_abupt = 7.3767% (PR #681, nezuko, terminal LR polish lr=3e-7)  
**Aspirational target:** ~7.0% (tay branch SOTA)  
**Primary gaps vs AB-UPT:** τ_y = 9.6123% val (2.63×), τ_z = 11.0573% val (3.05×), vol_p = 11.46% test (1.88×)

---

## Card 1: Selective Channel-Wise TTA (τ_y Only)

**Title:** Selective channel-wise test-time flip augmentation targeting τ_y

**Hypothesis:**
Bilateral TTA (PR #286) was a net negative overall (−0.644% abupt below merge bar) because flip-averaging harmed surface pressure (sp) and volume pressure (vp) while helping τ_y (+2.79%). This is expected: τ_y has left-right antisymmetry under a lateral flip (x → −x), meaning the average of original and flipped predictions for τ_y cancels noise without canceling signal. By contrast, sp and vp are symmetric under that flip, so averaging the original with the flipped prediction introduces no gain but can hurt if the model is not perfectly symmetric. Selectively applying TTA only to the τ_y channel, while keeping original predictions for all other channels, should capture the τ_y improvement without the cross-channel regression.

**Expected mechanism:**
τ_y measures lateral wall shear. A vehicle has (approximate) bilateral symmetry about the x-z midplane, so the physics dictates τ_y(x,y,z) ≈ −τ_y(−x,y,z). Averaging `pred_tau_y(original) + (−pred_tau_y(flipped)))/2` exploits this antisymmetry to reduce prediction variance near the separation bubbles that dominate the τ_y error. No training change required — this is pure inference-time.

**Implementation sketch:**
1. During eval/test, for each batch, also forward a laterally flipped copy of the surface point cloud (x → −x; normals nx → −nx).
2. Compute final predictions as:
   - `cp_final = cp_original` (symmetric — no TTA)
   - `tau_x_final = tau_x_original` (symmetric — no TTA)
   - `tau_y_final = (tau_y_original - tau_y_flipped) / 2` (antisymmetric — sign-flip before averaging)
   - `tau_z_final = tau_z_original` (symmetric — no TTA, avoid regression)
   - `vp_final = vp_original` (symmetric — no TTA)
3. Check that the surface feature flip is applied correctly: `surface_x[:, 0] *= -1; surface_x[:, 3] *= -1` (x and nx columns).
4. No volume flip needed (volume pressure is symmetric under lateral flip).

**Experimental protocol:**
Smoke run (3 epochs, viability check — primarily a code correctness test since TTA applies only at eval):
```bash
python train.py \
  --model-layers 6 --model-hidden-dim 448 --model-heads 8 --model-slices 32 \
  --optimizer lion --lr 1e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --swa-start-fraction 0.6 --swa-lr 5e-8 --swa-anneal-epochs 1 \
  --learnable-pe --surface-curvature-features \
  --beta-nll-beta 0.5 \
  --wallshear-huber-delta 0.3 \
  --grad-ema-alpha 0.5 \
  --selective-tta-tau-y \
  --resume-from <yi_sota_checkpoint> \
  --wandb_group round40-selective-tta \
  --kill-thresholds "500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25" \
  SENPAI_MAX_EPOCHS=3
```
(Exact flag name `--selective-tta-tau-y` TBD — student to implement and name.)

Gate: τ_y val metric must improve vs baseline without sp/vp/τ_z regression. If τ_y improves by ≥1% and no other channel degrades by >0.3%, proceed to a full-length confirmation run.

**Risk:**
Low-medium. The mechanism is well-understood from PR #286 data. The main risks are: (a) incorrect sign convention for the flip (student must verify the antisymmetry direction); (b) the model's internal representation is not exactly laterally symmetric so the flip average may not decompose cleanly. If the +2.79% τ_y benefit from PR #286 is confirmed on this stack, abupt should improve.

**Priority:** HIGH (1st priority). Zero training cost, theoretically grounded, directly targets the primary gap (τ_y), builds on existing PR data. If correct sign convention is applied, near-certain improvement in τ_y.

---

## Card 2: Volume Pressure Distribution Shift Diagnostic + SDF-Stratified Normalization

**Title:** Val/test volume pressure anomaly investigation with SDF-stratified normalization

**Hypothesis:**
The volume_pressure error has a striking val/test gap: 4.41% val vs 11.46% test (2.6× ratio). This is not explained by the standard train/val/test split alone. The most likely cause is a covariate shift in the volume point cloud SDF distribution between val and test cars — either the test geometries have more complex under-hood/interior flow, different SDF distance ranges, or the sampled volume points cluster more heavily in regions the model has not learned well. A secondary hypothesis is that the current global pressure normalization (mean/std over all training volume points) masks SDF-stratified distribution drift. Fixing requires: (1) diagnosing the shift, (2) applying SDF-stratified normalization or SDF-conditional positional encoding to make volume predictions robust to density shifts.

**Expected mechanism:**
SDF (signed distance field) encodes proximity to surface. Near-surface volume points (small SDF) have strong pressure gradients and are physically coupled to wall pressure (cp). Far-field points (large SDF) are governed by freestream + wake dynamics. If test cars have more near-surface volume points or a different SDF histogram, the model's single global normalization will be miscalibrated for those points. A per-bucket normalization (e.g., 4–8 SDF percentile buckets, each with its own running mean/std) would remove this bias. Alternatively, concatenating a learned SDF-embedding to the positional encoding could let the model adapt.

**Implementation sketch:**
Step 1 (Diagnostic — no training, just data analysis):
```python
# For each split, compute SDF histogram of volume points
# Compare val vs test SDF distributions
# Plot pressure prediction error vs SDF bucket
# Hypothesis confirmed if test has heavier near-surface SDF tail or different error profile
```

Step 2 (Training fix):
- Add `--volume-sdf-norm-buckets N` flag (e.g., N=8) to apply per-SDF-percentile normalization.
- Alternatively, augment volume features from `[x,y,z,sdf]` to `[x,y,z,sdf,sdf^2,log(|sdf|+ε)]` to give the model explicit SDF expressiveness.
- Keep surface predictions identical (no surface changes).

Smoke run (3 epochs):
```bash
python train.py \
  --model-layers 6 --model-hidden-dim 448 --model-heads 8 --model-slices 32 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --swa-start-fraction 0.6 --swa-lr 1e-7 \
  --learnable-pe --surface-curvature-features \
  --beta-nll-beta 0.5 --wallshear-huber-delta 0.3 \
  --volume-sdf-norm-buckets 8 \
  --resume-from <yi_sota_checkpoint> \
  --wandb_group round40-sdf-norm \
  --kill-thresholds "500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25"
```

Gate: val vp metric must not regress. If val vp holds and diagnostic confirms test SDF shift, run full confirmation and check if test_abupt closes toward val_abupt ratio (test/val ratio should decrease from 2.6× toward 1.3–1.5×).

**Risk:**
Medium. The diagnostic step is cheap but the normalization change requires careful implementation (bucket boundaries must be computed on training set only, applied consistently at inference). If the val/test gap is not due to SDF distribution shift (e.g., it is a genuine geometric complexity difference), this will not help vp. The SDF extended features variant (`sdf^2, log|sdf|`) is low-risk and may improve near-surface pressure accuracy regardless.

**Priority:** HIGH (2nd priority). The val/test gap for volume pressure is the largest unexplained anomaly in the current results. Even a partial fix on test_abupt has high paper-facing value.

---

## Card 3: Surface-Tangent Frame Decomposition for Wall Shear Prediction

**Title:** Local surface-tangent coordinate frame for τ_y/τ_z target decomposition

**Hypothesis:**
Wall shear τ = (τ_x, τ_y, τ_z) is currently predicted in the global Cartesian frame. This is suboptimal because the physically meaningful components of wall shear are tangent to the surface (two tangential components) and normal (which should be zero by no-slip). The global y and z projections of the tangential stress mix together depending on surface orientation, making them harder to learn. Decomposing τ into a local surface-tangent frame (t1, t2, n) — where t1/t2 span the tangent plane and n is the surface normal — before training, then re-projecting to Cartesian at inference, should give the model a more natural prediction basis. This is distinct from PR #603 (Mahalanobis local-frame loss rotation, negative) and PR #680 (tangential-loss penalty, different mechanism) and PR #713 (normal-penalty regularizer).

**Expected mechanism:**
At each surface point with normal n̂, define an orthonormal tangent frame (t1, t2) using Gram-Schmidt on the global x-axis and n̂. The shear τ in this frame has no normal component by fluid physics. Predicting (τ_t1, τ_t2) instead of (τ_x, τ_y, τ_z) reduces the effective dimensionality of the target from 3D (constrained) to 2D (unconstrained), removing a spurious degree of freedom. The model should learn a lower-variance mapping. At inference, rotate (τ_t1, τ_t2, 0) back to Cartesian using the stored rotation matrix. Expected benefit: τ_y and τ_z should converge faster and to a better minimum because their entanglement through surface-orientation variation is removed.

**Implementation sketch:**
1. Preprocessing: for each surface point with features `[x,y,z,nx,ny,nz,area]`, compute the 3×2 tangent matrix T from (nx,ny,nz). Store T as additional metadata.
2. Target transformation at training time: `tau_tangent = T^T @ tau_cartesian` (2D tangent components).
3. Predict (cp, τ_t1, τ_t2) as a 3-channel output instead of (cp, τ_x, τ_y, τ_z) as 4-channel. Or keep 4-channel and add a no-normal-component regularization loss.
4. At inference: `tau_cartesian = T @ tau_tangent`.
5. Loss: standard β-NLL on tangent-frame targets.

Smoke run (3 epochs):
```bash
python train.py \
  --model-layers 6 --model-hidden-dim 448 --model-heads 8 --model-slices 32 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 \
  --swa-start-fraction 0.6 --swa-lr 1e-7 \
  --learnable-pe --surface-curvature-features \
  --beta-nll-beta 0.5 \
  --surface-tangent-frame-targets \
  --resume-from <yi_sota_checkpoint> \
  --wandb_group round40-tangent-frame \
  --kill-thresholds "500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25"
```

Gate: τ_y + τ_z combined must improve by ≥0.5% without sp regression.

**Risk:**
Medium-high. The rotation must be applied consistently (training, validation, test, and must handle degenerate normals). The conceptual motivation is strong but the implementation is invasive. PR #603 showed that loss-space rotation was negative — however, that was a loss rotation, not a target-space rotation. The critical distinction is that this card changes the prediction target itself, not just the loss weighting. If the model struggles to learn the tangent-frame targets from scratch, warm-starting from yi SOTA should mitigate this.

**Priority:** MEDIUM (3rd priority). Strong physical motivation but invasive implementation. Best assigned to a student who can test carefully with a quick sanity check (manually verify that a random surface patch's rotation matrix is orthonormal and correctly recovers τ_cartesian).

---

## Card 4: CRPS / Energy Score Distributional Loss for τ_y/τ_z

**Title:** Continuous Ranked Probability Score loss targeting multimodal wall shear distributions

**Hypothesis:**
The current β-NLL loss assumes unimodal Gaussian residuals. Near flow separation bubbles on the vehicle sides, τ_y can have bimodal or fat-tailed distributions across the training set (cars with vs. without side mirrors, different A-pillar radii). A proper scoring rule that does not assume Gaussianity — the Continuous Ranked Probability Score (CRPS) or the energy score — would penalize miscalibrated predictions differently and may find lower-variance solutions for τ_y/τ_z. CRPS minimization is equivalent to finding the predictive median (or the full conditional CDF), making it more robust to outlier geometries.

**Expected mechanism:**
CRPS = E[|F(y) − 1{y≥t}|] integrated over t, which reduces to MAE for deterministic predictions and to a proper scoring rule for distributional predictions. For deterministic predictions (single point estimate), CRPS ≡ MAE. The experiment can first test CRPS-as-MAE (no distributional prediction, just a robustness check vs. MSE/β-NLL) for τ_y/τ_z while keeping β-NLL for cp and vp. If per-point CRPS improves τ_y/τ_z at val, it suggests the β-NLL variance-weighted loss is over-smoothing high-variance regions. A follow-up would use MC-dropout or multi-head predictions to get a sample-based estimate and compute CRPS with the full distributional form.

**Implementation sketch:**
Phase 1 (deterministic CRPS = MAE, low risk):
```bash
--wallshear-loss crps  # new flag: use CRPS (MAE form) for tau_y/tau_z
--beta-nll-beta 0.5    # keep for cp and vp
```

Phase 2 (distributional, if phase 1 shows τ_y improvement):
- Add N=8 MC-dropout samples at inference for τ_y/τ_z.
- Compute energy score E[||τ_pred − τ_gt||] − 0.5*E[||τ_pred − τ_pred'||] over samples.
- This requires adding dropout to the wall-shear head only.

Smoke run (3 epochs, Phase 1):
```bash
python train.py \
  --model-layers 6 --model-hidden-dim 448 --model-heads 8 --model-slices 32 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 \
  --swa-start-fraction 0.6 --swa-lr 1e-7 \
  --learnable-pe --surface-curvature-features \
  --wallshear-loss crps \
  --beta-nll-beta 0.5 \
  --resume-from <yi_sota_checkpoint> \
  --wandb_group round40-crps-loss \
  --kill-thresholds "500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25"
```

Gate (Phase 1): τ_y + τ_z combined must improve ≥0.3% vs baseline. If not, CRPS on this task collapses to MAE which is likely weaker than β-NLL and the phase 2 distributional variant may not be worth pursuing.

**Risk:**
Medium. Phase 1 (deterministic CRPS) is a simple loss swap. The main question is whether the β-NLL variance weighting (which upweights high-uncertainty regions) is actually hurting τ_y or helping. If β-NLL is already well-tuned at β=0.5, MAE form will likely underperform slightly on sp (which has clean signal). Recommend using CRPS only for the wall-shear head and keeping β-NLL for cp/vp.

**Priority:** MEDIUM (4th priority). Well-motivated by the multimodal separation bubble physics, but requires a new loss implementation. Phase 1 is low-risk and discriminating.

---

## Card 5: Multigrid Hierarchical Volume Attention

**Title:** Two-resolution coarse-to-fine attention for volume pressure prediction

**Hypothesis:**
Volume pressure in CFD is governed by an elliptic PDE (Laplace/Poisson). Elliptic solvers converge fastest with multigrid methods: a coarse grid removes low-frequency errors cheaply, and a fine grid resolves high-frequency residuals. The current Transolver applies a single-resolution attention over all volume points, which is inefficient for the long-range pressure correlations that dominate the elliptic mode. Adding a hierarchical attention that first computes a coarse global context (e.g., over a voxel-downsampled 10% subset of volume points) and then uses that as a cross-attention key/value for the full-resolution volume head should propagate global pressure gradients more efficiently, directly reducing the volume_pressure error.

**Expected mechanism:**
Coarse-level attention (N_coarse = ~1000 points, randomly or uniformly subsampled) captures the global pressure field structure (wake, stagnation zone, far-field). Fine-level attention (N_fine = full ~10000 points) attends to coarse keys via cross-attention, inheriting the global context. This mirrors the coarse-to-fine message passing in multigrid: the "restriction" step is the coarse subsampling, the "prolongation" is the cross-attention broadcast. The two-level hierarchy adds O(N_coarse^2 + N_coarse * N_fine) attention cost rather than O(N_fine^2), which is cheaper than a single full-resolution attention for large N.

**Implementation sketch:**
1. In the volume branch, before the final prediction head:
   - Subsample to N_coarse points (e.g., every 10th point, deterministic stride).
   - Apply 2–4 self-attention layers on coarse points → coarse context tokens.
   - Apply cross-attention from fine points (query) to coarse context (key/value) → augmented fine features.
   - Pass augmented fine features to the volume prediction MLP.
2. No changes to surface branch.
3. New flags: `--volume-multigrid-coarse-ratio 0.1` (fraction of volume points for coarse level), `--volume-multigrid-attn-layers 2`.

Smoke run (3 epochs):
```bash
python train.py \
  --model-layers 6 --model-hidden-dim 448 --model-heads 8 --model-slices 32 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 \
  --swa-start-fraction 0.6 --swa-lr 1e-7 \
  --learnable-pe --surface-curvature-features \
  --beta-nll-beta 0.5 \
  --volume-multigrid-coarse-ratio 0.1 --volume-multigrid-attn-layers 2 \
  --resume-from <yi_sota_checkpoint> \
  --wandb_group round40-multigrid-volume \
  --kill-thresholds "500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25"
```

Gate: val vp must improve ≥0.5%. Also check: does the val/test vp gap narrow (i.e., does test vp improve more than val vp)?

**Risk:**
Medium-high. This requires architectural changes to the volume branch. The cross-attention implementation must handle variable-length point clouds correctly (masked attention). The coarse subsampling adds a stochastic element during training that must be seeded consistently at eval. If the volume branch already has sufficient receptive field from the slice-attention mechanism, the coarse context may not add information. Recommend testing with a very small coarse ratio first (5%) to confirm the mechanism before scaling.

**Priority:** MEDIUM (5th priority). Strong physical motivation from elliptic PDE structure. Invasive but not a full architectural rewrite. Best paired with the SDF-normalization diagnostic (Card 2) since both target volume pressure.

---

## Card 6: Residual Correction Network on Frozen Base

**Title:** Lightweight correction MLP trained on frozen yi SOTA residuals

**Hypothesis:**
The yi SOTA model (PR #681) has a characteristic residual pattern: it systematically underestimates τ_y at high-curvature separation regions and overestimates volume pressure near the vehicle underbody. These residuals are not random — they have structure that a small auxiliary model can learn. By freezing the yi SOTA checkpoint and training only a lightweight 3-layer MLP to predict the residual (GT − base_pred) as a function of enhanced geometry features (curvature, SDF gradient, point density, surface tangent deviation), we can target the structured bias without disturbing the base model's calibration. This is distinct from PR #676 (knowledge distillation, teacher→student), which trains a smaller student model. Here the base model is kept intact and augmented.

**Expected mechanism:**
The correction MLP takes as input: [base_pred, x, y, z, nx, ny, nz, curvature_h1, curvature_h2, area, sdf_gradient_magnitude] and predicts the residual for each of the 5 output channels. The base_pred itself is a strong feature — the MLP only needs to learn the bias pattern. Training on 400 cars × all surface/volume points with a simple L1 or Huber loss on the residual should converge in 2–5 epochs. At inference: final_pred = base_pred + correction_pred. The correction MLP has ~50K parameters vs ~15M for the base model, so inference cost is negligible.

**Implementation sketch:**
1. Freeze all base model parameters: `for p in model.parameters(): p.requires_grad = False`.
2. Add a `CorrectionMLP` head with 3 hidden layers (256→256→256→output_dim) + ReLU + residual connection.
3. Features to correction MLP: base_pred concatenated with geometry features.
4. Train only correction MLP parameters.
5. New flag: `--residual-correction-mlp --correction-mlp-hidden 256 --correction-mlp-layers 3`.

Smoke run (3 epochs):
```bash
python train.py \
  --model-layers 6 --model-hidden-dim 448 --model-heads 8 --model-slices 32 \
  --optimizer adamw --lr 3e-4 --weight-decay 0.01 \
  --freeze-base-model \
  --residual-correction-mlp --correction-mlp-hidden 256 --correction-mlp-layers 3 \
  --surface-curvature-features \
  --resume-from <yi_sota_checkpoint> \
  --wandb_group round40-residual-correction \
  --kill-thresholds "500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25"
```

Note: use AdamW for the correction MLP (Lion is tuned for the full model; AdamW with higher lr converges faster for a small MLP from scratch).

Gate: val abupt must improve ≥0.1% (very low bar — this is a bias correction on an already-good model, gains should be small but consistent).

**Risk:**
Medium. The correction MLP approach is well-established in ensemble and calibration literature, but requires careful feature engineering. If the residuals are truly random rather than geometrically structured, the MLP will overfit. Strong regularization (weight decay 0.01, dropout 0.2 on MLP hidden layers) is essential. Recommend inspecting the residual heatmaps first (no training required) to confirm structure before running the full experiment.

**Priority:** MEDIUM-LOW (6th priority). Novel enough to be interesting; low training cost; but gain ceiling is inherently limited by the structured component of the residual. Best run as a quick 3-epoch sanity check before committing to a full confirmation run.

---

## Card 7: SAM (Sharpness-Aware Minimization) Optimizer

**Title:** SAM optimizer with rho sweep for flatter τ_y/τ_z minima

**Hypothesis:**
The yi SOTA sits in a sharp minimum for τ_y/τ_z — the loss landscape around the current optimum has high curvature in the directions corresponding to these metrics. Sharpness-Aware Minimization (SAM) finds flatter minima by performing an ascent step (finding the worst-case perturbation within an L2 ball of radius ρ) followed by a descent step. Flatter minima generalize better, and for a surrogate model where the test set covers geometrically different cars, generalization is the dominant challenge. The hypothesis is that SAM will find a checkpoint that generalizes better to both the validation τ_y/τ_z cases and the test volume_pressure distribution shift.

**Expected mechanism:**
SAM's perturbation step: θ̂ = θ + ρ * ∇L/||∇L||. This biases the optimizer toward flat regions of the loss surface where nearby weight perturbations do not increase loss. For a model that has been trained to convergence (as yi SOTA has), SAM from a warm start (low lr, small ρ) acts as a fine-tuning sharpness annealing step. Hyperparameter recommendation: ρ ∈ {0.02, 0.05, 0.1}; use with Lion base optimizer (SAM + Lion is compatible via the m-SAM variant); run for 3–5 epochs starting from yi SOTA checkpoint.

**Implementation sketch:**
```python
# SAM wrapper around Lion/AdamW
from sam import SAM  # github.com/davda54/sam
optimizer = SAM(model.parameters(), base_optimizer=Lion, rho=0.05, lr=1e-7)

# Training step:
def training_step(x, y):
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)  # ascent
    criterion(model(x), y).backward()
    optimizer.second_step(zero_grad=True)  # descent
```

New flag: `--sam-rho 0.05` (0.0 = disable, >0 = enable SAM with given radius).

Smoke run (3 epochs, 3-arm sweep over ρ):
```bash
# Arm 1: rho=0.02
python train.py --optimizer lion --lr 1e-7 --sam-rho 0.02 \
  --model-layers 6 --model-hidden-dim 448 --model-heads 8 --model-slices 32 \
  --use-ema --ema-decay 0.9999 --swa-start-fraction 0.6 --swa-lr 5e-8 \
  --learnable-pe --surface-curvature-features \
  --beta-nll-beta 0.5 --wallshear-huber-delta 0.3 \
  --resume-from <yi_sota_checkpoint> \
  --wandb_group round40-sam

# Arm 2: rho=0.05  (same, sam-rho 0.05)
# Arm 3: rho=0.10  (same, sam-rho 0.10)
```

Gate: any arm must show τ_y or τ_z improvement ≥0.3% without abupt regression. Best arm proceeds to 10-epoch confirmation.

**Risk:**
Medium. SAM doubles the forward-backward passes per step (2× compute). In a 4-GPU pod this is manageable. Main risks: (a) SAM implementation for Lion optimizer (not standard; m-SAM or GSAM may be needed); (b) warm-start SAM from a converged model may still disrupt the current optimum. Recommend starting with ρ=0.02 and checking that training loss does not spike in the first 100 steps.

**Priority:** MEDIUM-LOW (7th priority). Well-motivated from generalization theory, but 2× compute cost and uncertain benefit at very low lr fine-tuning makes this a tier-2 experiment after the lower-cost cards.

---

## Card 8: Geometry-Aware Mixup for τ_y/τ_z

**Title:** Geometry-similarity-constrained mixup targeting high-variance τ_y/τ_z regions

**Hypothesis:**
Standard random-pair mixup on vehicle surface point clouds creates physically implausible chimera geometries (e.g., interpolating between a sedan and an SUV), which adds noise rather than signal. Geometry-aware mixup pairs vehicles that are geometrically similar (measured by a PCA embedding of surface normal histograms), then interpolates their surface pressure and wall shear predictions in a geometrically meaningful way. This should act as a structured data augmentation that improves τ_y/τ_z generalization by interpolating between geometrically plausible car variants.

**Expected mechanism:**
Compute a 32-dimensional feature vector per car from histograms of surface normal orientations (8 azimuth × 4 elevation bins). Apply PCA to get a 10D embedding. Pair cars within the top-5 nearest neighbors in this embedding space. Mix (x_a, y_a) and (x_b, y_b) with λ ~ Beta(0.4, 0.4). The key difference from random mixup: the mixed geometry is physically plausible (two similar body styles), so the mixed τ_y/τ_z target is a valid interpolation. Apply only in the surface domain (volume mixup is more complex, skip for now).

**Implementation sketch:**
1. Precompute PCA geometry embeddings at dataset initialization.
2. During training batch construction, replace random pairing with kNN-based geometry pairing.
3. Apply input interpolation: `x_mix = λ*x_a + (1−λ)*x_b`, target interpolation: `y_mix = λ*y_a + (1−λ)*y_b`.
4. New flags: `--geometry-mixup-alpha 0.4 --geometry-mixup-knn 5`.

Smoke run (3 epochs):
```bash
python train.py \
  --model-layers 6 --model-hidden-dim 448 --model-heads 8 --model-slices 32 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 \
  --swa-start-fraction 0.6 --swa-lr 1e-7 \
  --learnable-pe --surface-curvature-features \
  --beta-nll-beta 0.5 \
  --geometry-mixup-alpha 0.4 --geometry-mixup-knn 5 \
  --resume-from <yi_sota_checkpoint> \
  --wandb_group round40-geometry-mixup \
  --kill-thresholds "500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25"
```

Gate: τ_y + τ_z combined must improve ≥0.3% over 3 epochs. If no signal at 3 epochs, close (mixup typically needs more training to show benefit; if not visible at 3 epochs on a warm start, the mechanism is likely inactive on this dataset size).

**Risk:**
Medium-high. The dataset has only 400 training cars — at kNN=5, each car has a fixed pool of 5 mixup partners, which limits diversity. The geometric similarity computation (surface normal histogram PCA) may not capture the features most relevant to τ_y (e.g., door mirror geometry is more relevant than overall body style orientation). Recommend inspecting the kNN clusters visually before running training.

**Priority:** LOW (8th priority). Speculative mechanism on a small dataset. Reserve for a round after the higher-confidence cards have been evaluated.

---

## Summary Priority Ranking

| Rank | Card | 1-Line Summary | Risk | Compute |
|------|------|----------------|------|---------|
| 1 | Selective Channel-Wise TTA | Flip-average τ_y only at inference; zero training cost; +2.79% τ_y demonstrated in PR #286 | Low | Minimal |
| 2 | SDF-Stratified Normalization | Fix val/test vp gap (4.41% → 11.46%) via per-SDF-bucket normalization; closes biggest unexplained anomaly | Medium | Low |
| 3 | Surface-Tangent Frame Decomposition | Predict τ in local tangent frame to remove spurious Cartesian entanglement in τ_y/τ_z | Med-High | Medium |
| 4 | CRPS Distributional Loss | Replace β-NLL with CRPS for τ_y/τ_z; targets multimodal shear distributions at separation bubbles | Medium | Low |
| 5 | Multigrid Hierarchical Volume Attention | Two-resolution coarse-to-fine attention for volume pressure; FEM/elliptic PDE motivation | Med-High | Medium |
| 6 | Residual Correction Network | Lightweight MLP on frozen yi SOTA residuals; targets structured geometric bias | Medium | Low |
| 7 | SAM Optimizer | Sharpness-aware fine-tuning from yi SOTA; 2× compute but flatter minima | Med-Low | High (2×) |
| 8 | Geometry-Aware Mixup | kNN-constrained mixup on geometrically similar vehicle pairs | Med-High | Low |

**Recommended assignments for 4 idle students (alphonse, kohaku, nezuko, thorfinn):**
- **alphonse** → Card 1 (Selective TTA): zero training cost, quick to implement, near-certain τ_y improvement
- **kohaku** → Card 2 (SDF Normalization): run diagnostic + training fix in parallel; directly targets test metric anomaly
- **nezuko** → Card 3 (Tangent Frame Decomposition): most physically principled architectural change; nezuko has demonstrated strong implementation reliability
- **thorfinn** → Card 4 (CRPS Loss): clean loss-swap experiment; quick to implement and evaluate
