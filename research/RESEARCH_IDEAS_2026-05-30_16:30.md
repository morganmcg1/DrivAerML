# SENPAI Research Ideas — 2026-05-30 16:30

**ADVISOR CORRECTION 16:40Z**: H304 (lateral bilateral y-flip TTA) is **REDUNDANT** — the existing `mirror` in `eval_tta_h252.py:209-230` IS already y-flip (negates surface_x[..,1]=y, surface_x[..,4]=ny, volume_x[..,1]=y; un-mirrors tau_y[..,2]). The researcher-agent misidentified mirror as x-mirror (front-back); it is in fact y-mirror (left-right bilateral). Cars are NOT front-back symmetric (wake asymmetry), so x-mirror is not a valid TTA augmentation. **Skip H304. Proceed with H305 → H306 → H307 (pending checkpoint audit) → H308.**

Generated for: H304+ (next wave, 4–5 students to become idle ~18:30–21:30Z after H297–H303 land)
Current SOTA: H295 EP15+anti-K5+6-res+mirror (PR #1483) — val_abupt=5.9231%, test_abupt=5.7679%
Active in-flight: H296 (fern K=4+8-res), H297 (thorfinn per-layer noise), H298 (alphonse coord-noise), H299 (askeladd embed-noise), H300 (edward per-channel calibration), H301 (nezuko heterogeneous best-of-K), H302 (frieren asymmetric modality resolution), H303 (tanjiro σ=3e-4 at K=5)
Hard constraints: TTA-only (no retraining), tau_z LOCKED, SENPAI_TIMEOUT_MINUTES=360, DDP×8 GPUs, read-only: data/loader.py, data/preload.py, data/split_manifest.json
Primary bottleneck: test_WSS=6.6732% (paper target <5.850%, gap ~82bp) — largest remaining sub-metric gap

Context: The K/res/noise-pattern cluster is saturated. Findings AAA–SSS cover: σ-axis, K-axis, EP-axis, aggregation operators, noise families (Gaussian/Laplace/Student-t), Sobol QMC, multi-σ diversity, resolution extent (upper+lower), resolution density (+40K/+57K midpoints), per-layer noise stratification (H297 in-flight), per-channel calibration (H300 in-flight), embed-only noise (H299 in-flight), coord-noise (H298 in-flight), Hutchinson curvature-σ (TT), Taylor-2 correction (SS). What follows must attack from a categorically different angle.

---

## Ranking by Expected Value

1. **H304** (lateral symmetry TTA — bilateral flip) — Highest EV. Novel mechanism (geometric symmetry), zero compute cost, physically principled (cars are bilaterally symmetric). Directly attacks WSS_z asymmetry error which is the biggest sub-metric gap.
2. **H305** (WSS normal-component zeroing — BC enforcement) — Highest mechanistic grounding. Physical boundary condition τ·n̂≈0 at wall is guaranteed by physics, making this a free correction with zero extra passes.
3. **H306** (variance-adaptive inverse-variance weighting) — Medium EV. Replaces uniform mean with per-point confidence-weighted mean. Orthogonal to K selection, orthogonal to noise family, orthogonal to aggregation-operator sweep (which only tested homogeneous pooling).
4. **H307** (ensemble checkpoint interpolation — linear loss-landscape SWA on 2 models) — Lower EV but high upside if the two EP15 checkpoint families sit on a low-loss valley. Different from prior SWA (LL-SWA-null which averaged adjacent epochs of the same run).
5. **H308** (surface–volume prediction consistency self-correction) — Exploratory. Physically motivated but implementation cost is higher; recommended only if H304–H306 are exhausted.

---

## H304 — Lateral Bilateral Symmetry TTA (Left↔Right Flip)

**Target students: any idle student (assign first)**
**EV rank: 1st (Highest)**

### Mechanism

Cars are bilaterally symmetric about the x-z plane (y=0 in DrivAerML convention). The model has no explicit symmetry constraint — it learned approximate symmetry from data but not exactly. At test time, for any input point cloud X, we can compute:

```
prediction(X) averaged with flip(prediction(flip(X)))
```

where `flip` negates the y-coordinate (and the y-component of normals and wall-shear). This is a zero-cost, zero-noise augmentation that does not require any retraining, extra parameters, or stochastic sampling. It is NOT a noise-based TTA (K passes); it is a single deterministic augmentation applied once.

Why it helps WSS_z: the z-axis WSS channel (tau_z) measures shear stress in the car-roof/hood direction. Under a y-flip, tau_z is symmetric (even function of y), meaning the model's asymmetric prediction errors for tau_z will cancel out when averaged over the flip. The y-WSS channel (tau_y) is antisymmetric (odd function of y), so it averages to the true antisymmetric value, canceling left-right bias. The x-WSS channel (tau_x) is symmetric, also benefiting. Surface pressure (symmetric) also benefits marginally.

This is mechanistically distinct from the x-mirror (mirror_x) augmentation already in the codebase: mirror_x reflects the car about the x=0 plane (front-back, not left-right). Bilateral y-flip has never been tested.

### Why it is orthogonal to all existing findings

- Findings AAA–SSS cover stochastic noise perturbation and its variants. H304 adds zero noise.
- Finding PP (mirror×noise interaction): mirror_x is the front-back flip; this is mirror_y (left-right flip), a completely different symmetry axis.
- The model was trained on DrivAerML's 400 cars, which include both left-right symmetric and slightly asymmetric configurations. A bilateral y-flip should cancel model-induced asymmetry errors.
- Zero extra GPU passes: only adds one additional model forward pass (the flip), so total compute is 2× per existing configuration, approximately doubling wall-clock for the flip itself.

### Implementation plan

**Eval script**: Modify `trainer_runtime.py` or add a small `eval_h304_bilateral_tta.py` wrapper.

**Key transform functions**:
```python
def flip_surface_input(surface_x, surface_normals):
    # Negate y-coordinate of surface points
    flipped_x = surface_x.clone()
    flipped_x[..., 1] = -flipped_x[..., 1]  # negate y
    # Also negate y-component of normals
    flipped_normals = surface_normals.clone()
    flipped_normals[..., 1] = -flipped_normals[..., 1]
    return flipped_x, flipped_normals

def flip_surface_prediction(surface_pred):
    # surface_pred: [B, N, 4] — ch0=cp, ch1=tau_x, ch2=tau_y, ch3=tau_z
    # tau_y is antisymmetric under y-flip; tau_x, tau_z, cp are symmetric
    flipped = surface_pred.clone()
    flipped[..., 2] = -flipped[..., 2]  # negate tau_y
    return flipped

def bilateral_tta_average(model, batch):
    # Standard forward pass
    pred_orig = model(surface_x=batch.surface_x, ...)
    
    # Flip y-axis of input
    flipped_batch = flip_batch_y(batch)
    pred_flip = model(surface_x=flipped_batch.surface_x, ...)
    
    # Unflip the prediction (negate tau_y)
    pred_flip_unflipped = flip_surface_prediction(pred_flip["surface_preds"])
    
    # Average
    pred_avg = (pred_orig["surface_preds"] + pred_flip_unflipped) / 2
    return pred_avg
```

**Stack this on top of the existing K=5 anti-thetic + 6-res + mirror (H295 config)**. The bilateral flip is an additional outer augmentation: for each (resolution, noise draw, mirror state), also flip — this doubles the number of passes. Alternatively, test it first in isolation (K=1, no noise, 1 resolution, with/without flip) to confirm the mechanism is alive before adding to the full stack.

**Suggested staging**:
- Arm A (cheap diagnostic, ~30 min): K=1, σ=0, 1 resolution (65536), no x-mirror, with y-bilateral flip. Compare to the un-augmented single-pass baseline. If WSS_z improves by >1bp, the mechanism is alive.
- Arm B (full stack, ~8h): Fold y-flip into the full H295 config (K=5, 6-res, x-mirror, y-flip). This doubles wall-clock to ~8h — may exceed 360-min timeout.

**IMPORTANT**: Arm B will likely exceed 360-min timeout if applied naively (K=5×6-res×2×2 = 120 passes). Consider K=3 anti-thetic + y-flip + 6-res + x-mirror (K=3×6×2×2=72 passes ≈ 290 min) as the compound arm. The bilateral flip is free compared to adding more K passes, so K can be reduced slightly to stay within budget.

**Suggested CLI for Arm A (diagnostic)**:
```bash
H295_CKPT="outputs/ensemble_cache/run-yw2a5dyl-epoch-15-ema/checkpoint.pt"
torchrun --standalone --nproc_per_node=8 target/eval_h304_bilateral_tta.py \
  --checkpoint $H295_CKPT \
  --resolutions "65536" \
  --eval-modes "bilateral_flip_only" \
  --batch-size 4 --num-workers 4 \
  --wandb-name "student/h304-bilateral-flip-arm-a-diagnostic" \
  --wandb-group "h304-bilateral-symmetry-tta"
```

**Suggested CLI for Arm B (full stack)**:
```bash
torchrun --standalone --nproc_per_node=8 target/eval_h304_bilateral_tta.py \
  --checkpoint $H295_CKPT \
  --resolutions "32768,49152,65536,81920,98304,131072" \
  --eval-modes "anti_bilateral_res_mirror_avg" \
  --n-antithetic-pairs 3 \
  --weight-noise-sigma 5e-4 \
  --bilateral-flip true \
  --batch-size 2 --num-workers 4 \
  --wandb-name "student/h304-bilateral-flip-arm-b-k3-6res-xmirror-yflip" \
  --wandb-group "h304-bilateral-symmetry-tta"
```

### Wall-clock estimate

- Arm A (diagnostic): ~20–30 min on DDP×8
- Arm B (K=3+6-res+x-mirror+y-flip): ~6×2×2×3 = 72 passes × ~4.5 min/pass = ~320 min (within 360-min limit)

### Expected fail mode

If DrivAerML cars have significant y-asymmetry in their geometries (e.g., steering wheel, exhaust), the y-flip introduces geometric inconsistency and errors may cancel asymmetries that should not cancel. Check that the car geometries are dominantly symmetric by inspection of the normals; if not, the flip will hurt VP/SP as well.

---

## H305 — Physical BC Enforcement: Normal-Component WSS Zeroing

**Target students: second idle student**
**EV rank: 2nd (Highest mechanistic grounding)**

### Mechanism

The no-penetration boundary condition at a solid wall states that the fluid velocity normal to the wall is zero. This implies that wall shear stress τ is tangential to the wall surface — the normal component τ·n̂ should be zero exactly (to machine precision in the CFD solver). The model predicts τ = (τ_x, τ_y, τ_z) as a 3-vector at each surface point, but it has no explicit constraint enforcing τ·n̂ = 0.

The fix is trivially cheap: at inference, project the predicted τ vector onto the tangent plane of the surface at each point:

```python
def enforce_wss_tangential(wss_pred, surface_normals):
    # wss_pred:       [B, N, 3]  (tau_x, tau_y, tau_z)
    # surface_normals: [B, N, 3] (unit normals, available in surface_x features)
    # Remove normal component: tau_tangential = tau - (tau·n̂)n̂
    n_hat = F.normalize(surface_normals, dim=-1)
    normal_component = (wss_pred * n_hat).sum(-1, keepdim=True)  # [B, N, 1]
    wss_corrected = wss_pred - normal_component * n_hat           # [B, N, 3]
    return wss_corrected
```

This is a zero-cost post-processing step applied after the model forward pass. It is guaranteed to reduce WSS normal-component error; the only question is whether the model's normal-component error is large enough to affect the relative L2 metric materially.

Why WSS_z specifically: z-axis normal vectors (pointing roughly in the gravity/up direction on the roof) will have the largest cross-component leakage because the model cannot distinguish between WSS_z (shear force in z direction) and the normal-z component (which should be zero). On the hood/roof surfaces where n̂ ≈ (0,0,1), the model's tau_z prediction conflates true tangential z-shear with normal-z force. Projecting out the normal removes this leakage.

### Why it is orthogonal to all existing findings

This is a physics-informed post-processing step, not a noise/sampling/aggregation change. No finding in the bank (AAA–SSS) has tested this. It costs zero extra forward passes — it is applied after any TTA aggregation is complete. Stack directly on top of H295 config.

### Implementation plan

**Key requirement**: The surface point cloud feature tensor `surface_x` must include surface normals. According to `program.md`, surface_x has the layout: [x, y, z, nx, ny, nz, area, ...]. Confirm the exact normal channels from `data/loader.py` (read-only: can inspect, not modify).

```python
# In trainer_runtime.py, after the TTA aggregation produces surface_pred_agg:
surface_normals = batch.surface_x[..., 3:6]  # nx, ny, nz channels
wss_pred = surface_pred_agg[..., 1:4]         # tau_x, tau_y, tau_z channels
wss_corrected = enforce_wss_tangential(wss_pred, surface_normals)
surface_pred_agg = torch.cat([surface_pred_agg[..., :1], wss_corrected], dim=-1)
```

This is a 5-line change to `trainer_runtime.py` with no hyperparameters.

**Suggested CLI (same as H295, plus a --enforce-wss-tangential flag)**:
```bash
H295_CKPT="outputs/ensemble_cache/run-yw2a5dyl-epoch-15-ema/checkpoint.pt"
torchrun --standalone --nproc_per_node=8 target/eval_h305_bc_enforcement.py \
  --checkpoint $H295_CKPT \
  --resolutions "32768,49152,65536,81920,98304,131072" \
  --eval-modes "anti_res_mirror_avg" \
  --n-antithetic-pairs 5 \
  --weight-noise-sigma 5e-4 \
  --enforce-wss-tangential true \
  --batch-size 2 --num-workers 4 \
  --wandb-name "student/h305-bc-enforcement-wss-normal-zero-k5-6res" \
  --wandb-group "h305-bc-enforcement-wss"
```

### Wall-clock estimate

~3.5h on DDP×8 (same pass count as H295 baseline — this adds negligible compute).

### Expected fail mode

If the surface normals in `surface_x` are not unit-normalized, the projection formula needs an extra normalization step. Check `data/loader.py` to confirm. Also: if the model's normal-component WSS error is small (e.g., model has already learned near-zero normal component from data), this correction will be near-zero — the experiment provides information either way.

---

## H306 — Per-Point Inverse-Variance Weighting of TTA Passes

**Target students: third idle student**
**EV rank: 3rd (Medium)**

### Mechanism

The current TTA aggregation computes a uniform mean over K passes:

```
f_final(x_i) = (1/K) * sum_k f(w + ε_k)(x_i)
```

This treats every surface point x_i equally regardless of how consistent the model is across passes at that point. Points in high-curvature regions (sharp edges, separation lines) have high cross-pass variance and high prediction error. These points are exactly the ones where the model is uncertain, yet they contribute equally to the aggregate.

Inverse-variance weighting gives higher weight to passes where the model is more confident about a given point:

```
f_final(x_i) = sum_k [ w_k(x_i) * f_k(x_i) ]

where w_k(x_i) = 1 / var_k(x_i) / sum_j 1/var_j(x_i)
and var_k(x_i) = || f_k(x_i) - mean_k f_k(x_i) ||^2  (computed across the K pass dimension)
```

Note: this is NOT the same as Findings OOO (aggregation operators sweep). Finding OOO tested homogeneous operators (median, trimmed mean, Huber) that treat all points the same way. H306 is heterogeneous: the weight w_k(x_i) varies per point AND per pass, based on the observed cross-pass variance. It is a point-adaptive aggregation.

Why this targets WSS: WSS_z has the highest cross-pass variance (curvature and separation regions on the car roof have higher flow sensitivity). Downweighting high-variance points adaptively should reduce the contribution of uncertain predictions to the metric.

### Why it is distinct from Finding OOO (aggregation null result)

Finding OOO tested: mean, median, trimmed_mean_10, trimmed_mean_20, Huber. All use the same weight for every point across all K passes. H306 uses a different weight for every point×pass pair. The two are mechanistically orthogonal: Finding OOO says "homogeneous pooling operator doesn't matter"; H306 says "spatially heterogeneous pooling based on cross-pass confidence can help."

### Implementation plan

```python
def inverse_variance_aggregate(preds_stack):
    # preds_stack: [K, B, N, C]  (stacked TTA predictions)
    K, B, N, C = preds_stack.shape
    mean_pred = preds_stack.mean(0)              # [B, N, C]
    variance = ((preds_stack - mean_pred.unsqueeze(0)) ** 2).mean(0)  # [B, N, C]
    inv_var = 1.0 / (variance + 1e-8)            # [B, N, C], eps for stability
    weights = inv_var / inv_var.sum(0, keepdim=True)  # [K, B, N, C] if broadcast
    # Actually: per-point weights are the same across K (it's a post-hoc reweighting)
    # So: aggregate per-point with confidence weighting across K
    weighted = (preds_stack * inv_var.unsqueeze(0)).sum(0) / inv_var.sum(0)
    return weighted
```

Wait — the per-pass weight w_k(x_i) should reflect how close pass k's prediction is to the cross-pass mean, not just the variance. A cleaner formulation: use the inverse of per-pass deviation from mean as the per-pass weight:

```python
def confidence_weighted_aggregate(preds_stack):
    # preds_stack: [K, B, N, C]
    mean_pred = preds_stack.mean(0, keepdim=True)  # [1, B, N, C]
    per_pass_deviation = ((preds_stack - mean_pred) ** 2).mean(-1, keepdim=True)  # [K, B, N, 1]
    weights = torch.softmax(-per_pass_deviation * temperature, dim=0)  # [K, B, N, 1]
    weighted = (preds_stack * weights).sum(0)  # [B, N, C]
    return weighted
```

Temperature `temperature` controls sharpness: temperature→0 gives argmax (best-of-K), temperature→∞ gives uniform mean. Sweep {0.5, 1.0, 2.0, 5.0} on val before running test.

**Suggested CLI**:
```bash
H295_CKPT="outputs/ensemble_cache/run-yw2a5dyl-epoch-15-ema/checkpoint.pt"
torchrun --standalone --nproc_per_node=8 target/eval_h306_invvar_tta.py \
  --checkpoint $H295_CKPT \
  --resolutions "32768,49152,65536,81920,98304,131072" \
  --eval-modes "anti_res_mirror_invvar_avg" \
  --n-antithetic-pairs 5 \
  --weight-noise-sigma 5e-4 \
  --aggregation-mode "confidence_weighted" \
  --aggregation-temperature 1.0 \
  --batch-size 2 --num-workers 4 \
  --wandb-name "student/h306-invvar-k5-6res-temp1.0" \
  --wandb-group "h306-invvar-confidence-weighted"
```

Run temperature sweep on val first (very cheap: recompute from cached K-pass predictions), then run test with the best temperature.

### Wall-clock estimate

~3.5–4h on DDP×8 (same pass count as H295; slight overhead for per-pass deviation computation).

### Expected fail mode

If the TTA predictions are already tightly clustered (low cross-pass variance due to the anti-thetic pairing reducing variance), the confidence weights will be nearly uniform and this degenerates to the mean. Finding OOO showed mean ≈ all alternatives — if this mechanism gives the same result, it confirms the TTA predictions are uniformly low-variance and no point-adaptive refinement helps.

---

## H307 — Loss-Landscape Linear Interpolation Between Two Independently-Trained Models

**Target students: fourth idle student**
**EV rank: 4th (Medium-high — orthogonal to all TTA findings)**

### Mechanism

Finding LL-SWA-null showed that SWA (averaging adjacent epochs EP13,EP14,EP15 of the same run) gives no improvement. That is averaging weights along a single training trajectory. This is different.

H307 linearly interpolates between two independently-trained models at the optimal checkpoint:

```
w_α = α * w_A + (1-α) * w_B
```

where w_A and w_B are the weights of two different EP15 checkpoints from different training runs (if multiple trained models exist in `outputs/ensemble_cache/`). The loss landscape literature (Frankle+Carlin ICLR 2020, Garipov+Vetrov NeurIPS 2018) shows that independently trained models often lie on a low-loss valley connected by a curved path in weight space — and the linear midpoint is on or near this valley for models trained with the same data.

This is NOT ensembling (which keeps both models and averages predictions). It is weight-space interpolation producing a single model that can then be evaluated with the full H295 TTA stack.

Why it may help: the interpolated model at α=0.5 may occupy a point in the loss landscape with better generalization than either endpoint alone — a "loss valley" midpoint. This is the Stochastic Weight Averaging (SWA) intuition but applied across different training runs rather than epochs of the same run.

Note: prior Greedy Ensemble work (Findings in PRs #556, #562, #880, #1030, #1059) uses an ensemble of predictions, not weight interpolation. H307 creates a new single model via weight averaging, which is different.

### Implementation plan

1. Identify two independently-trained EP15 checkpoints from the cache directory. From the PR history, H253 used `run-yw2a5dyl-epoch-13-ema` and the EP15 checkpoint is presumably from the same run or a newer run. Check if a second independent EP15-quality checkpoint exists in `outputs/ensemble_cache/`.

2. Weight interpolation:
```python
def interpolate_weights(checkpoint_A, checkpoint_B, alpha=0.5):
    state_A = torch.load(checkpoint_A)["model"]
    state_B = torch.load(checkpoint_B)["model"]
    interpolated = {}
    for key in state_A:
        interpolated[key] = alpha * state_A[key] + (1-alpha) * state_B[key]
    return interpolated

# Save interpolated checkpoint and evaluate with full H295 TTA stack
```

3. Alpha sweep: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} on val, then run test with best alpha.

**Caveat**: This requires two high-quality independent EP15 checkpoints. If the cache only has one unique model family, this hypothesis cannot be run. Check checkpoint availability before assigning.

**Suggested CLI (after confirming checkpoints exist)**:
```bash
torchrun --standalone --nproc_per_node=8 target/eval_h307_weight_interp.py \
  --checkpoint-a $H295_CKPT_A \
  --checkpoint-b $H295_CKPT_B \
  --interpolation-alpha 0.5 \
  --resolutions "32768,49152,65536,81920,98304,131072" \
  --eval-modes "anti_res_mirror_avg" \
  --n-antithetic-pairs 5 \
  --weight-noise-sigma 5e-4 \
  --batch-size 2 --num-workers 4 \
  --wandb-name "student/h307-weight-interp-alpha0.5-k5-6res" \
  --wandb-group "h307-weight-interpolation"
```

### Wall-clock estimate

~3.5–4h per alpha value on DDP×8. Alpha sweep: 9 values × 3.5h = 31.5h total, but these can be parallelized: run alpha sweep at K=1 + 1-res (cheap, ~30 min each on one set of K pass configs) to pick best alpha before the full stack run.

### Expected fail mode

If only one independent training run exists in the checkpoint cache, this experiment cannot run. If the two checkpoints are from the same training trajectory (just different epochs), this degenerates to SWA (already null, Finding LL). Confirm checkpoint independence (different W&B run IDs) before assigning.

---

## H308 — Surface-Volume Prediction Consistency as Self-Correction Signal

**Target students: fifth idle student (lower priority, assign last)**
**EV rank: 5th (Exploratory — higher implementation cost)**

### Mechanism

The model simultaneously predicts surface pressure (sp), wall shear stress (wss_x/y/z), and volume pressure (vp) from shared representations. These predictions are NOT independently generated — they share a backbone. The physical relationship between surface pressure gradients and volume pressure is governed by Bernoulli's equation (approximately, for inviscid flow regions):

```
p_surface ≈ p0 - 0.5 * ρ * v²
```

More concretely: on the car body surface, there is a continuity relationship between the pressure just inside the boundary layer (volume_pressure near the wall) and the surface_cp (coefficient of pressure). If the model predicts a surface_cp pattern inconsistent with its own volume_pressure pattern near the wall, this inconsistency is a signal that predictions can be corrected.

The TTA version: for each test case, use the gradient of the volume_pressure field near the wall surface as a consistency check for the surface_pressure prediction. If surface_cp gradient disagrees with vol_p gradient direction in the near-wall region, apply a light correction.

This is a zero-extra-forward-pass approach — the model already produces both surface and volume predictions from a single forward pass, so no additional computation is required.

### Why this targets SP and VP jointly

test_SP=3.6421% (>3.577% paper target, 6.5bp gap) and test_VP=3.3781% (<3.421% target, met). However, SP is the binding constraint on the Morgan target. Enforcing surface-volume consistency may improve SP via the pressure continuity regularizer.

### Implementation plan (high-level, student fills details)

1. Identify near-wall volume points: for each surface point, find the k-nearest volume points (within a distance threshold δ) using precomputed distances.
2. Compute the predicted pressure gradient direction from volume predictions at these near-wall points.
3. Compare to the surface_cp gradient direction.
4. Apply a small correction to surface_cp that reduces the inconsistency: `cp_corrected = cp_pred + λ * consistency_correction`.

This requires access to the volume predictions alongside surface predictions at test time — which the model already provides. The main implementation challenge is efficiently computing surface↔volume point correspondences at test time without modifying read-only data loading code.

**Constraint check**: `data/loader.py` and `data/preload.py` are read-only. The correspondence computation must be done inline at eval time using the raw point coordinates in `batch.surface_x` and `batch.volume_x`.

### Wall-clock estimate

~3.5–4h on DDP×8 for the base evaluation, plus correspondence computation overhead (~30 min for nearest-neighbor computation on 50K surface + 300K volume points).

### Expected fail mode

The near-wall volume point density may be too sparse to compute a reliable pressure gradient. If volume points are not concentrated near the wall (some may be distributed throughout the volume), the gradient estimation will be noisy. This is the most speculative mechanism in this list.

---

## Summary Table

| ID | Title | EV Rank | Target Metric | Wall Clock | Mechanism | Novel vs. Existing? |
|----|-------|---------|---------------|------------|-----------|---------------------|
| H304 | Lateral bilateral y-flip TTA | 1st (High) | WSS_z (primary), WSS_y | Arm A ~30min, Arm B ~5h | Geometric symmetry augmentation, zero noise | Never tried (mirror_x ≠ mirror_y) |
| H305 | BC enforcement: WSS normal zeroing | 2nd (High) | WSS_z specifically | ~3.5h | Physics: τ·n̂=0 at wall | Never tried |
| H306 | Per-point inverse-variance weighting | 3rd (Medium) | WSS (high-variance regions) | ~4h | Heterogeneous confidence-weighted pooling | Distinct from OOO (homogeneous operators) |
| H307 | Weight-space interpolation (2 models) | 4th (Med-High) | All axes | ~4h + alpha sweep | Loss-landscape valley interpolation | Distinct from SWA (LL, adjacent epochs) |
| H308 | Surface-volume consistency correction | 5th (Exploratory) | SP, VP jointly | ~4h + overhead | Physics: pressure continuity BC | Never tried; higher implementation cost |

---

## Assign Priority Queue

**Assign when students become idle (expected 18:30–21:30Z)**:

1. H304 (bilateral y-flip) — assign to first idle student. Start with Arm A diagnostic (~30 min), then Arm B if Arm A shows WSS_z improvement.
2. H305 (BC enforcement) — assign to second idle student. Confirm surface_x has normal channels first.
3. H306 (inverse-variance weighting) — assign to third idle student.
4. H307 (weight interpolation) — assign to fourth idle student ONLY IF a second independent EP15 checkpoint exists in the cache.
5. H308 (surface-volume consistency) — assign to fifth idle student, lower priority.

---

## Non-Overlap Verification

- H297 (thorfinn in-flight): per-layer σ_attn=0/σ_mlp=5e-4. H304/305/306/308 add no noise. H307 modifies weights before TTA, not during.
- H298 (alphonse in-flight): input-coord noise. None of H304–H308 add input noise.
- H299 (askeladd in-flight): embedding-only noise. None of H304–H308 modify embeddings.
- H300 (edward in-flight): per-channel affine calibration. H305/H306 are post-processing but not affine — different mechanism.
- H301 (nezuko in-flight): per-channel best-of-K selection. H306 is per-POINT confidence weighting — not per-channel selection.
- H302 (frieren in-flight): asymmetric modality resolution. H304 is a different axis (symmetry, not resolution).
- H303 (tanjiro in-flight): σ=3e-4 at K=5. H304–H308 use σ=5e-4 (optimal per Finding AAA) or zero (H305).
- Findings bank check: AAA (σ-flat), BBB (K-saturates-EP13), ..., SSS (K-axis-saturation): none cover geometric symmetry augmentation, physical BC enforcement, point-adaptive confidence weighting, cross-run weight interpolation, or surface-volume consistency correction.
