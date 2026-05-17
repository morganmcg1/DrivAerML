# WSS Research Hypotheses Wave 2 — 2026-05-17 01:30

**Target**: test_wss < 5.85% (current SOTA: 6.727%)
**Floors**: test_vol_p ≤ 3.643, test_surf_p ≤ 3.577 (must NOT regress)
**Dataset**: `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`
**Optimizer**: Lion, lr=5e-4, DDP8
**Budget**: 30 epochs, ≈33h wall-clock

## Background: What We Know

### Confirmed Mechanisms (stack these)
- **Curvature additive attention bias** (H5/H9): curvature computed from surface normals injected as additive bias into slice attention logits; `curvature_bias_scale=1.0`. NOT as input channel (gradnorm imbalance).
- **MAE aux loss + gradient clamp=0.15** (H9b): vol_p floor recovery via gradient mass floor; prevents vol_p regression when surface loss dominates.
- **Charbonnier loss on τ_z** (H10b): robustness against τ_z outliers; slight τ_z slope lead confirmed.
- **GradNorm α=0.5**: adaptive loss balancing, confirmed winner.
- **Y-axis symmetry augmentation** (p=0.5): confirmed winner.
- **6L STRING backbone** (5-sigma octave span): SOTA config.
- **SDF-stratified volume sampling** (α=2.0): confirmed winner.

### Refuted Mechanisms (do not retry without new evidence)
- Per-axis WSS loss upweighting (H4, H11): neither τ_x/τ_y/τ_z individual weights nor uniform up-weight helped.
- Uniform surface_loss_weight=1.5 (H7): confirmed REFUTED.
- Raw curvature input channels (H2): gradnorm imbalance, refuted.
- Wind-exposure raw channels (H1): refuted.
- Near-wall volume cross-attention (H3): refuted.

### Currently Running (H9b/H10b/H11b) — do NOT duplicate
- **H9b (tanjiro, PR #1157)**: curvature_bias + MAE_aux + clamp=0.15 stack.
- **H10b (frieren, PR #1159)**: H9b stack + Charbonnier loss on τ_z specifically.
- **H11b (nezuko, PR #1160)**: AdamW optimizer sweep + per-axis τ loss weights (revisit of H4 but with H9b stack).

### Key Architectural Observation
The surface output head is a SINGLE `LinearProjection(n_hidden, 4)` mapping to [cp, τ_x, τ_y, τ_z]. All four channels compete in the same output projection. This is the primary architectural bottleneck candidate that NO current hypothesis addresses.

---

## H12: Separate WSS Decoder Head with Deeper MLP

### Hypothesis Statement
Replacing the single shared `LinearProjection(n_hidden, 4)` surface head with two independent heads — a shallow linear cp head and a deeper 2-layer MLP τ head — will allow the model to learn WSS-specific representations without cp competing for the same output projection weights, reducing τ error by 0.5–1.0pp.

### Expected Mechanism
Currently, gradient signals from the cp loss and τ losses are summed before flowing through the shared `surface_out` projection. The cp target has different statistical structure (scalar pressure field, correlated with global flow) vs. τ (3D vector shear, dominated by local geometry and near-wall velocity gradients). A single linear projection forces these to share a common final-layer feature basis.

By giving τ its own 2-layer MLP (`Linear(n_hidden, 2*n_hidden) → GELU → Linear(2*n_hidden, 3)`), τ can compose geometry-sensitive features from the backbone hidden state without cp pulling the shared weights toward pressure-optimized subspaces. The extra MLP depth in the τ head provides capacity for the non-linear feature combination that cross-flow components (τ_y, τ_z) likely require.

This is orthogonal to H9b/H10b/H11b: those experiments modify loss terms, curvature injection, and optimizer. This modifies the output architecture only.

### Implementation Details

**File to modify**: `model.py` (ONLY)

**Change**: In `SurfaceTransolver.__init__`:
```python
# REMOVE:
self.surface_out = LinearProjection(n_hidden, self.surface_output_dim)

# ADD:
self.surface_cp_out = LinearProjection(n_hidden, 1)  # cp channel only
self.surface_tau_out = nn.Sequential(
    nn.Linear(n_hidden, 2 * n_hidden),
    nn.GELU(),
    nn.Linear(2 * n_hidden, 3),  # tau_x, tau_y, tau_z
)
# initialize tau_out final layer near zero for stable start:
nn.init.normal_(self.surface_tau_out[-1].weight, std=0.01)
nn.init.zeros_(self.surface_tau_out[-1].bias)
```

**Change**: In `SurfaceTransolver.forward`:
```python
# REMOVE:
surface_preds = self.surface_out(surface_hidden) * surface_mask.unsqueeze(-1)

# ADD:
cp_preds = self.surface_cp_out(surface_hidden)  # [B, N_surf, 1]
tau_preds = self.surface_tau_out(surface_hidden)  # [B, N_surf, 3]
surface_preds = torch.cat([cp_preds, tau_preds], dim=-1)  # [B, N_surf, 4]
surface_preds = surface_preds * surface_mask.unsqueeze(-1)
```

**CLI flags**: Identical to H9b baseline; no new flags needed. Use H9b full confirmed stack:
```bash
torchrun --nproc_per_node=8 train.py \
  --data_path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --n_layers 6 --n_hidden 192 --n_head 3 --slice_num 96 \
  --pe_kind string_separable --pe_num_features 16 \
  --pe_init_sigmas 0.1 0.178 0.316 0.562 1.0 \
  --n_surface_points 65000 --n_volume_points 65000 \
  --volume_sdf_alpha 2.0 \
  --optimizer lion --lr 5e-4 \
  --use_gradnorm --gradnorm_alpha 0.5 \
  --use_ema --ema_decay 0.999 \
  --y_sym_aug_prob 0.5 \
  --use_curvature_bias --curvature_bias_scale 1.0 \
  --use_mae_aux --grad_clamp 0.15 \
  --epochs 30 \
  --wandb_group H12-separate-tau-head
```

**Parameter overhead**: ~192*2*192 + 192*3 ≈ 74,304 extra parameters (< 0.1% model size). Negligible compute increase.

### EP10 Viability Gate
- val_wss ≤ 6.3% (vs. H9b EP10 baseline of ~6.5%; need ≥0.2pp improvement to justify continuation)
- val_vol_p ≤ 4.5% (floor check; failure here = stop)
- τ_z slope must be negative (still improving at EP10)

### Non-Duplication Argument
H9b: curvature bias + MAE_aux → gradient-level mechanism.
H10b: Charbonnier τ_z → loss-level mechanism.
H11b: AdamW + per-axis τ weights → optimizer-level mechanism.
H12: separate τ head (deeper MLP) → output architecture-level mechanism.
These are orthogonal axes of the pipeline. H12 can be stacked on top of any winner.

---

## H13: Magnitude-Decoupled τ Loss

### Hypothesis Statement
Decomposing the τ prediction into independently supervised magnitude (`|τ|`) and direction (`τ / |τ|`) components — with separate loss terms — will reduce τ error by enabling the model to separately optimize scalar magnitude estimation (dominated by global car geometry) and vector direction estimation (dominated by local surface curvature), improving the τ_y/τ_z cross-flow bottleneck by 0.3–0.7pp.

### Expected Mechanism
The current relative L2 loss on τ = [τ_x, τ_y, τ_z] treats the three components as an undifferentiated vector. In physical reality:
- `|τ|` is determined by near-wall velocity gradient magnitude, which depends on global flow attachment/separation topology.
- `τ/|τ|` (unit direction) is determined by local surface geometry (principally the curvature-induced flow steering in τ_y and τ_z).

The cross-flow components (τ_y, τ_z) have SOTA errors of 7.94% and 9.54% vs. AB-UPT 3.65%/3.63% — roughly 2-3x worse than the streamwise τ_x. This suggests the directional prediction is the bottleneck, not the magnitude. By isolating `cos_sim_loss = 1 - (τ_pred · τ_gt) / (|τ_pred| |τ_gt|)` as a separate loss term, gradient signal on direction is not swamped by magnitude error.

Decomposition:
```
L_tau_mag  = RelL2(|τ_pred|, |τ_gt|)      # magnitude term
L_tau_dir  = mean(1 - cos_sim(τ_pred, τ_gt))  # direction term
L_tau = λ_mag * L_tau_mag + λ_dir * L_tau_dir
```
With GradNorm already in the stack, λ_mag and λ_dir can be auto-balanced via GradNorm's per-loss scaling (treat them as two separate task losses from GradNorm's perspective).

### Implementation Details

**File to modify**: `train.py` (loss computation section)

The loss function must compute:
```python
# In loss computation block:
tau_pred = surface_preds[..., 1:4]  # [B, N_surf, 3]
tau_gt   = surface_y[..., 1:4]

# Magnitude loss
tau_mag_pred = torch.norm(tau_pred, dim=-1, keepdim=True)  # [B, N_surf, 1]
tau_mag_gt   = torch.norm(tau_gt,   dim=-1, keepdim=True)
loss_tau_mag = relative_l2(tau_mag_pred, tau_mag_gt, mask)

# Direction loss (cosine)
tau_pred_unit = F.normalize(tau_pred, dim=-1, eps=1e-8)
tau_gt_unit   = F.normalize(tau_gt,   dim=-1, eps=1e-8)
loss_tau_dir  = (1.0 - (tau_pred_unit * tau_gt_unit).sum(-1))[mask].mean()

# GradNorm treats loss_tau_mag and loss_tau_dir as separate task losses
```

**Important gotcha**: Zero-|τ| regions (separation bubbles, far-field padding) can cause NaN in `F.normalize`. Use `eps=1e-8` and mask out points where `|τ_gt| < 1e-6` before the cosine loss.

**CLI flags**: Add `--use_mag_dir_tau_loss` (new flag in train.py); rest is H9b stack:
```bash
torchrun --nproc_per_node=8 train.py \
  [... H9b stack flags ...] \
  --use_mag_dir_tau_loss \
  --wandb_group H13-mag-dir-tau-loss
```

**GradNorm integration**: Add `loss_tau_mag` and `loss_tau_dir` as two separate entries in GradNorm's loss dictionary rather than pre-combining them. GradNorm will auto-balance.

### EP10 Viability Gate
- val_wss ≤ 6.2% (need ≥0.3pp improvement over H9b EP10)
- val_tau_y + val_tau_z combined must improve (the direction loss is specifically targeting these)
- If val_tau_y/tau_z do NOT improve vs. H9b at EP10, direction decomposition is not helping → stop

### Non-Duplication Argument
H9b/H10b/H11b all apply loss modifications at the per-output-channel level (e.g., Charbonnier on τ_z). H13 changes the geometric decomposition of what is being predicted — magnitude vs. direction — which is a fundamentally different formulation of the prediction problem. It does not change channel selection, optimizer, or architecture.

---

## H14: Surface Normal Prediction Auxiliary Task (Geometric Denoising)

### Hypothesis Statement
Adding a self-supervised auxiliary task where the backbone predicts the surface normal [nx, ny, nz] from `surface_hidden` — given only [x, y, z] as positional input (without the normals) — will force the model to learn richer geometric representations that benefit WSS prediction, reducing test_wss by 0.4–0.8pp.

### Expected Mechanism
Surface normals are currently provided as INPUT channels `[x, y, z, nx, ny, nz, area]`. The auxiliary task flips this: we withhold normals from the input (provide only `[x, y, z, area]`) and force the backbone to predict them.

This creates a "geometric denoising" pretraining signal: the backbone must reconstruct the local surface geometry (normals) from raw point positions. The learned geometry-aware representations then improve the downstream WSS prediction, because WSS direction is fundamentally determined by near-wall flow steering which is governed by surface curvature — precisely the information encoded in normals.

The auxiliary loss is only backpropagated through `surface_hidden`; the volume branch is unaffected. This is structurally similar to MAE-style auxiliary tasks but targets surface geometry specifically, not masked point reconstruction.

Variant: instead of completely withholding normals, use the normals in input but add an auxiliary head that predicts normals from hidden state anyway. This "redundant prediction" forces the hidden state to maintain geometry-decodable representations even when processing non-normal features. This is easier to implement (no input masking) and more stable.

**Recommended variant**: redundant prediction (normals in input AND predicted from hidden).

### Implementation Details

**Files to modify**: `model.py` and `train.py`

In `model.py`, add to `SurfaceTransolver.__init__`:
```python
# Auxiliary normal prediction head
self.normal_aux_out = nn.Sequential(
    nn.Linear(n_hidden, n_hidden // 2),
    nn.GELU(),
    nn.Linear(n_hidden // 2, 3),  # predict [nx, ny, nz]
)
```

In `SurfaceTransolver.forward`, add to return dict:
```python
normal_preds = self.normal_aux_out(surface_hidden) * surface_mask.unsqueeze(-1)
return {
    "surface_preds": surface_preds,
    "volume_preds": volume_preds,
    "normal_preds": normal_preds,  # NEW: [B, N_surf, 3]
    "hidden": hidden, ...
}
```

In `train.py`, add aux loss:
```python
# surface_x channels: [x, y, z, nx, ny, nz, area]  (normals are at indices 3,4,5)
normal_gt = surface_x[..., 3:6]  # [B, N_surf, 3]
normal_pred = model_output["normal_preds"]

# Cosine similarity aux loss (normals are unit vectors by definition)
normal_pred_unit = F.normalize(normal_pred, dim=-1, eps=1e-8)
normal_gt_unit   = F.normalize(normal_gt,   dim=-1, eps=1e-8)
loss_normal_aux  = (1.0 - (normal_pred_unit * normal_gt_unit).sum(-1))[surface_mask].mean()

# Weight: treat as separate GradNorm task with initial weight=0.1
# (small weight — this is auxiliary, not primary)
```

**GradNorm integration**: `loss_normal_aux` enters GradNorm as a new task. Initialize its weight to 0.1 so it starts subdominant to the primary WSS/cp/vol_p losses.

**CLI flags**:
```bash
torchrun --nproc_per_node=8 train.py \
  [... H9b stack flags ...] \
  --use_normal_aux --normal_aux_weight 0.1 \
  --wandb_group H14-normal-aux
```

**Critical gotcha**: If you run GradNorm with 5 tasks (cp, τ, vol_p, MAE, normals), ensure GradNorm's initial task weights don't cause any single loss to blow up. Monitor per-task GradNorm weights at EP1 and EP2 closely.

### EP10 Viability Gate
- val_wss ≤ 6.3% (need ≥0.2pp over H9b EP10)
- val_normal_aux ≤ 5% cosine loss (sanity check — if the aux task isn't being learned, the aux loss is providing no useful gradient)
- vol_p floor: val_vol_p ≤ 4.5%

### Non-Duplication Argument
H9b uses MAE aux (masked autoencoder style). H14 uses a normal prediction aux task — a geometric denoising task targeting surface geometry rather than point masking recovery. The mechanism is different: MAE provides gradient mass to prevent vol_p collapse; normal prediction forces geometry-aware representations that directly benefit WSS direction prediction.

---

## H15: Layer-wise Curvature Bias Scaling (Late-Layer Amplification)

### Hypothesis Statement
The current uniform `curvature_bias_scale=1.0` across all 6 Transolver layers may be suboptimal: early layers should see small curvature bias to learn global flow structure, while late layers (closest to the output) should see amplified curvature bias to specialize in geometry-driven local WSS variations. Using a linearly increasing bias scale (0.25 → 1.5 across layers 1–6) will improve WSS without degrading pressure fields.

### Expected Mechanism
In transformer-like architectures, early layers typically capture global structure (positional relationship between surface patches, global flow attachment) while later layers handle local feature refinement. The curvature attention bias is designed to make the model attend to geometrically salient regions — but its optimal strength may vary by layer depth.

In H5/H9, the single global `curvature_bias_scale=1.0` was confirmed effective vs. no curvature. But this is not an ablation of per-layer scaling. A layer-wise ramp (small early, large late) should:
1. Allow early layers to route information globally without geometric over-fixation.
2. Allow late layers to strongly specialize final representations to high-curvature regions where WSS gradients are sharpest.

This is analogous to "progressive resolution" in image transformers (coarse → fine attention), applied to curvature-driven geometry attention.

### Implementation Details

**File to modify**: `model.py`

The `curvature_bias_scale` currently enters `TransolverAttention.forward` as a fixed scalar. Layer index must be passed to each attention layer:

In `SurfaceTransolver.__init__`, replace:
```python
self.curvature_bias_scale = curvature_bias_scale  # scalar
```
With:
```python
# Linear ramp from scale_min to scale_max across n_layers
import torch
scale_min, scale_max = 0.25, 1.5
self.curvature_bias_scales = torch.linspace(scale_min, scale_max, n_layers)
# [0.25, 0.525, 0.8, 1.075, 1.35, 1.5] for n_layers=6
```

Then in the backbone forward loop, pass the per-layer scale:
```python
for i, layer in enumerate(self.layers):
    x = layer(x, curvature_bias=curvature_bias,
               curvature_bias_scale=self.curvature_bias_scales[i].item())
```

**CLI flags**: Add `--curvature_bias_scale_min 0.25 --curvature_bias_scale_max 1.5`; mutually exclusive with `--curvature_bias_scale` (single scalar). Default behavior unchanged if only `--curvature_bias_scale` is passed:
```bash
torchrun --nproc_per_node=8 train.py \
  [... H9b stack flags, WITHOUT --curvature_bias_scale 1.0 ...] \
  --curvature_bias_scale_min 0.25 --curvature_bias_scale_max 1.5 \
  --wandb_group H15-layerwise-curvature-bias
```

**Alternative quick-test**: Instead of full implementation, test with `--curvature_bias_scale 1.5` (amplified uniform) vs. baseline `1.0` first. If 1.5 wins, layer-wise ramp is warranted. If 1.5 loses, the ramp may still outperform both via early regularization.

### EP10 Viability Gate
- val_wss ≤ 6.3% (need ≥0.2pp over H9b EP10, which uses scale=1.0)
- val_surf_p ≤ 3.7% (cp should not be harmed by more curvature bias in late layers)
- Attention weight visualization (optional): curvature high-attention regions should be geometrically coherent (A/C pillars, wheel arches, hood leading edge)

### Non-Duplication Argument
H9b/H10b use `curvature_bias_scale=1.0` uniformly. H15 changes the distribution of curvature attention strength across transformer depth. This is a depth-wise hyperparameter not touched by any running experiment. The mechanism (layer-depth specialization) is distinct from loss/optimizer/architecture changes in H9b/H10b/H11b.

---

## H16: Focal-Power Surface Point Loss Weighting

### Hypothesis Statement
Weighting the surface loss for each point proportional to `(|τ_gt| / mean_|τ_gt|)^γ` (with γ ≈ 0.5–1.0) will focus gradient signal on high-shear regions (wheel arches, A-pillars, underbody edges) where cross-flow τ_y/τ_z errors dominate, reducing test_wss by 0.3–0.6pp without requiring new architecture or aux tasks.

### Expected Mechanism
The current relative L2 loss computes `||τ_pred - τ_gt||² / ||τ_gt||²` globally — which effectively already normalizes by the target magnitude. However, within the per-point mean reduction, regions with very low |τ| (far-field surface, flat top) contribute equal weight to regions with high |τ| (wheel arch, leading edges).

The key physical insight: τ_y and τ_z errors are largest in regions of strong lateral flow steering — high-curvature junctions, separation lines, and underbody transitions. These regions also have the highest |τ| values. By up-weighting high-|τ| points in the loss (proportional to normalized magnitude), we increase gradient signal specifically where the physics are most complex and our cross-flow errors are largest.

This is different from per-axis channel weighting (H4/H11, refuted): H4/H11 applied uniform scalar multipliers to τ_y or τ_z channels. H16 applies data-driven, geometry-responsive spatial weighting based on local τ magnitude. It functions as a soft analog of importance sampling in loss space, without requiring curvature computation.

### Implementation Details

**File to modify**: `train.py`

In the surface loss computation:
```python
# Compute per-point τ magnitude from ground truth (normalized within batch)
with torch.no_grad():
    tau_gt_mag = torch.norm(surface_y[..., 1:4], dim=-1)  # [B, N_surf]
    tau_gt_mag_mean = tau_gt_mag[surface_mask].mean().clamp(min=1e-8)
    focal_weights = (tau_gt_mag / tau_gt_mag_mean).clamp(min=0.1, max=10.0)
    focal_weights = focal_weights ** gamma  # γ in {0.5, 0.75, 1.0}

# Apply per-point weight to surface loss before mean reduction
# (internal to the surface loss computation, not as an outer multiplier)
surface_loss = (focal_weights * per_point_surface_loss)[surface_mask].mean()
```

**Recommended γ**: Start with γ=0.75 (soft — not fully linear). γ=1.0 is linear and may over-penalize; γ=0.5 is sqrt and gentler.

**CLI flags**:
```bash
torchrun --nproc_per_node=8 train.py \
  [... H9b stack flags ...] \
  --surface_focal_gamma 0.75 \
  --wandb_group H16-focal-tau-weight
```

**Critical gotcha**: With GradNorm in the stack, the per-point focal weighting changes the effective gradient magnitude for the surface task. This will cause GradNorm to re-balance the surface task weight. Monitor GradNorm's surface task weight at EP1; if it collapses rapidly, reduce γ. Also: the `vol_p` floor safeguard is GradNorm's main mechanism here — if focal weighting pushes more gradient toward surface, GradNorm should auto-compensate for vol_p. Still, add gradient clamp=0.15 (from H9b) to prevent spikes.

**Important**: Do NOT compute focal weights from `surface_preds` (prediction magnitudes) — use ONLY `surface_y` (ground truth). Using predictions creates a non-stationary loss that can cause runaway feedback.

### EP10 Viability Gate
- val_wss ≤ 6.2% (need ≥0.3pp over H9b EP10)
- val_tau_y and val_tau_z must BOTH improve (focal weighting targets high-shear regions where cross-flow errors are largest)
- vol_p floor: val_vol_p ≤ 4.5%
- If only τ_x improves and τ_y/τ_z do not, the mechanism is not targeting the right region → stop

### Non-Duplication Argument
H4/H11: per-channel scalar weights on τ_y, τ_z axes (refuted). H16: per-POINT spatial weights based on local τ magnitude (not per-channel). The refuted approaches applied uniform channel multipliers; H16 applies spatially-adaptive magnitude-responsive weights. The failure mode of H4/H11 was that uniform channel weighting worsened other channels; H16's point-level weighting preserves relative per-channel loss balance while increasing gradient in physically meaningful regions.

---

## Summary Ranking

| Rank | Hypothesis | Target mechanism | Effort | Expected gain | Risk |
|------|-----------|-----------------|--------|---------------|------|
| 1 | **H12**: Separate τ head (2-layer MLP) | Output architecture bottleneck | Low (10 LOC model.py) | 0.5–1.0pp wss | Low — no loss/opt changes |
| 2 | **H13**: Magnitude-decoupled τ loss | Loss landscape (magnitude vs. direction) | Medium (20 LOC train.py) | 0.3–0.7pp wss | Medium — GradNorm interaction |
| 3 | **H14**: Normal aux task | Self-supervised geometry awareness | Medium (30 LOC model+train) | 0.4–0.8pp wss | Medium — GradNorm with 5 tasks |
| 4 | **H16**: Focal-power point weighting | Loss spatial up-weighting | Low (10 LOC train.py) | 0.3–0.6pp wss | Medium — GradNorm re-balance |
| 5 | **H15**: Layer-wise curvature bias | Per-layer attention geometry | Low (5 LOC model.py) | 0.2–0.4pp wss | Low — small perturbation |

**TOP RECOMMENDATION for dl24-fern: H12** — separate WSS decoder head with deeper MLP.

Rationale: The single `LinearProjection(n_hidden, 4)` is the only significant architectural bottleneck not yet addressed by any running or prior experiment. It requires the smallest code change (≈10 lines in `model.py`, zero changes to `train.py`), stacks cleanly on the confirmed H9b configuration (curvature bias + MAE_aux + GradNorm + Lion lr=5e-4), and is fully orthogonal to H9b/H10b/H11b. If the architecture bottleneck is real, the effect should be visible at EP5 as a diverging τ trajectory. If it fails (τ does not improve by EP10), it strongly implies the representation in `surface_hidden` is already τ-expressive and the bottleneck is elsewhere (loss formulation or data), which is itself a valuable result for directing H13/H14.

**Runner-up for second idle student: H13** (magnitude-decoupled τ loss) — complements H12 by attacking the loss formulation axis rather than architecture.
