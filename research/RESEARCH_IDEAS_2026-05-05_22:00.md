# Research Ideas — Round 42 (2026-05-05 22:00 UTC)

**Branch:** yi
**Merge bar:** val_abupt = **7.3767%** / test_abupt = **8.7015%** (PR #681, nezuko, run `dc031qpt`)
**Pending merge candidates:** Norman #724 (7.3588%), Edward #672 (7.3660% at EP2, still training)
**Post-merge bar estimate (if both merge):** ~7.35%
**Aspirational target:** ~7.0% (tay branch SOTA PR #511, `5o7jc7wi`)
**Primary gaps vs AB-UPT:**
- τ_y: 9.5832% val (2.63×), τ_z: 11.0377% val (3.04×), surface_p: 4.8515% (1.27×)

**In-flight (do NOT duplicate):**
#724 norman (residual MLP), #672 edward (decoupled τ head), #746 haku (y-flip aug), #743 senku (multi-ckpt ensemble), #744 tanjiro (hard-mining polish), #725 violet (multigrid vol attn), #726 gilbert (SAM polish), #739 chihiro (curvature-weighted loss), #731 alphonse (EMA snapshot ensemble), #721 thorfinn (CRPS/MAE loss), #720 nezuko (surface-tangent frame), #719 kohaku (SDF Phase 2), #715 askeladd (annealed wallshear weighting), #713 fern (normal-penalty tangency), #733 emma (dual-tower bridge), #652 frieren (Muon+Lion chain)

**Confirmed null (do NOT repeat):** RFF surface features (#674/#661), geometry-aware mixup (#727), selective TTA without y-aug training (#718), asinh normalization (#668), Perceiver-IO backbone (#675), 4L/768d depth (#659), 6L/512d depth (#714), curvature k1/k2 features (#662)

---

## Card 1: Y-Symmetry Pair Loss Stabilized (Tanjiro Revisit)

**Priority: 1 — Highest**

**Title:** Y-symmetry pair loss with cosine LR decay and tight gradient clip — stabilized rerun

**Hypothesis:**
Tanjiro PR #671 demonstrated that the y-symmetry pair loss *mechanism works* — val reached ~8.17% at EP2 before diverging. The divergence was caused by Lion optimizer operating at a fixed LR (3e-7) with no decay, allowing accumulation of gradient noise from the antisymmetric loss component as training progressed. The fix is: (a) cosine LR decay from 1e-4 down to 1e-7 over 30 epochs (prevents late-run divergence), (b) clip_grad_norm=0.25 (tight, not 1.0), (c) kill gate at step 10k if val > 11% (catch early divergence). With these stabilizers, the pair loss should enforce bilateral antisymmetry on τ_y throughout training, directly reducing the 2.63× τ_y gap.

**Expected mechanism:**
The y-symmetry pair loss adds a penalty `|τ_y(x,y,z) + τ_y(x,-y,z)|` (τ_y is ODD under y-flip) for paired surface point batches. This forces the model to learn the correct antisymmetric structure of lateral wall shear. Without it, the model may converge to a y-symmetric solution that systematically underpredicts τ_y magnitude near side-body separation bubbles. The peak val 8.17% at EP2 in #671 proves the signal is there — stabilization is purely an optimization problem, not a mechanism problem.

**Implementation sketch:**
```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 1e-4 --lr-scheduler cosine --lr-min 1e-7 --lr-warmup-steps 500 \
  --weight-decay 0.05 --clip-grad-norm 0.25 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --y-symmetry-pair-loss --y-symmetry-weight 0.1 \
  --wandb_group round42-y-pair-loss-stabilized \
  --kill-thresholds "500:train/loss<5,10000:val_primary/abupt_axis_mean_rel_l2_pct<11"
```
SENPAI_MAX_EPOCHS=30 (full budget needed for cosine decay effect)

**Gate:** val_abupt < 9.0% by step 15k (else close). Primary success: val_abupt < 7.35% at any checkpoint.

**Risk:** Medium. The mechanism confirmed at peak EP2 ~8.17%, but this is a cold-start 30-epoch run. If the model fails EP1 gate, close immediately. The tight clip_grad_norm=0.25 is the key stabilizer vs #671.

**Difference from #671:** Cosine LR (1e-4→1e-7, 30 epochs) vs fixed 3e-7. clip_grad_norm=0.25 vs 1.0. Kill gate at step 10k val>11%. Full SOTA stack (learnable STRING-sep PE + grad-EMA + β-NLL). Cold-start vs resume.

---

## Card 2: Cosine LR Warm-Restart on Lion from SOTA Checkpoint

**Priority: 2 — High**

**Title:** Cosine annealing with warm restarts on Lion optimizer, starting from yi SOTA

**Hypothesis:**
The yi SOTA was reached by a monotonic LR decay schedule. Multiple experiments (tanjiro #671, askeladd #715) show divergence or plateau at late training with constant Lion LR. Cosine warm restarts (SGDR, Loshchilov 2017) allow the optimizer to escape shallow local minima by periodically increasing then decaying LR. Applied on top of the current SOTA checkpoint, a short warm-restart schedule (T_0=2 epochs, T_mult=2, lr_max=5e-5, lr_min=1e-8) should allow the SOTA weights to explore nearby optima that the monotonic schedule missed, specifically around τ_y/τ_z loss basins where the pair-loss path diverged.

**Expected mechanism:**
The SOTA checkpoint is already near a local optimum. Standard cosine decay (single cycle) cannot escape if the optimum is wide-but-suboptimal. Warm restarts briefly raise LR to allow exploration, then decay to a lower floor. For a 4-GPU Polish run starting from dc031qpt, T_0=2 epochs (the full dataset pass) gives 2 full restarts in 6 epochs. The τ_y/τ_z channels have the largest prediction variance — a brief LR bump may allow the model to reach a better saddle point for these channels while the well-converged p channels re-stabilize.

**Implementation sketch:**
```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 5e-5 --lr-scheduler cosine_warm_restarts --lr-T0 2 --lr-T-mult 2 --lr-min 1e-8 \
  --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --resume-from <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-cosine-restart-lion \
  --kill-thresholds "500:train/loss<5,4000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=6 (3 restart cycles, enough to see benefit)

Note: If `--lr-scheduler cosine_warm_restarts` is not implemented, the student should implement via `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` wrapping the Lion optimizer.

**Gate:** val_abupt < 7.35% by EP2 restart point. If the first restart cycle (EP2) shows no improvement vs SOTA, close.

**Risk:** Medium. If the SOTA is already at a global optimum for this architecture, the restart will add noise without benefit. The kill gate at step 4000 val>8.5% (well above SOTA) is protective.

**Difference from prior LR experiments:** CosineAnnealingWarmRestarts (not single-cycle decay). Starting from SOTA checkpoint (not cold start). T_mult=2 means restarts progressively lengthen.

---

## Card 3: Cholesky Correlated NLL for Joint τ_y/τ_z Uncertainty

**Priority: 3 — High**

**Title:** Full lower-triangular covariance NLL loss coupling τ_y and τ_z predictions

**Hypothesis:**
The current β-NLL loss treats each output channel independently (diagonal covariance). But τ_y and τ_z are physically coupled — both arise from the same velocity gradient tensor, and their errors are correlated (if the model mispredicts the separation bubble on the side of a vehicle, it typically errs on both). A lower-triangular (Cholesky) parameterization of the joint 2×2 covariance matrix for (τ_y, τ_z) allows the model to explicitly learn this correlation and calibrate its uncertainty accounting for the joint distribution. This should reduce τ_y+τ_z combined error by allowing the model to trade variance between channels intelligently.

**Expected mechanism:**
The NLL for a bivariate Gaussian is `0.5 * [(y-mu)^T Σ^{-1} (y-mu) + log|Σ|]`. Parameterize Σ = L L^T where L is lower-triangular (Cholesky factor), with diagonal elements exp(l_ii) > 0. The off-diagonal `l_21` captures τ_y/τ_z covariance. At initialization, set L = I (equivalent to current diagonal β-NLL). The model learns to use the off-diagonal term to reduce joint uncertainty. Per-point covariance — the head predicts L parameters (3 scalars: l_11, l_22, l_21) per point in addition to the 5 channel means.

**Implementation sketch:**
- Modify the output head to predict `[tau_y_mu, tau_z_mu, ..., l_11, l_21, l_22, ...]` where l-values are the Cholesky factors
- Apply correlated NLL loss only on (τ_y, τ_z) pair; keep current β-NLL for τ_x, surface_p, vol_p
- β-schedule: apply β=0.5 uncertainty weighting to the Cholesky covariance magnitude as currently done
- Initialization: set l_11=l_22=1, l_21=0 initially (identity covariance, matching current diagonal loss)
- Mixed loss: `loss = cholesky_nll(τ_y, τ_z) + beta_nll(τ_x) + beta_nll(surface_p) + beta_nll(vol_p)`

```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --cholesky-tau-yz-nll --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --resume-from <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-cholesky-nll \
  --kill-thresholds "500:train/loss<5,3000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=4

**Gate:** τ_y or τ_z must individually improve vs SOTA by EP2. If both regress, close.

**Risk:** Medium-high. Cholesky NLL adds 3 predicted scalars per point and a matrix inverse at every step. Memory overhead is small (~3/7 = 43% more head outputs), but the training stability depends on initialization. The identity initialization should prevent early divergence.

**Difference from β-NLL:** Explicitly models τ_y/τ_z covariance (off-diagonal). β-NLL is purely diagonal. This is orthogonal to Edward's decoupled-head approach and can be composed.

---

## Card 4: Boundary Layer Thickness Feature (δ_99 Estimate from SDF Gradient)

**Priority: 4 — High**

**Title:** Boundary layer height feature derived from SDF gradient profile as per-point surface input

**Hypothesis:**
The τ_y/τ_z gap is largest in regions of boundary layer separation and reattachment (side body, A-pillars, wake). These regions are also where the boundary layer thickness δ_99 changes rapidly. Adding a per-surface-point estimate of δ_99 — derived from the SDF gradient decay profile normal to the surface — gives the model explicit spatial conditioning on where the boundary layer is thick vs thin. Thick δ_99 → large τ_y/τ_z gradients; thin δ_99 → near-linear profile, smaller τ magnitude. This is a physics-grounded feature that explicitly encodes information the model currently must infer from geometry alone.

**Expected mechanism:**
At each surface point, the SDF gradient `∇SDF` points normal to the surface. Walk along this gradient in the volume point cloud and find the distance at which the SDF magnitude crosses a threshold (e.g., 0.01×car_length). This gives a local curvature-adjusted δ estimate. Concatenate this as a scalar feature to the surface input: `[nx, ny, nz, area, delta_99_estimate]`. The model then directly observes where separation is expected and can allocate more representational capacity to τ_y/τ_z there.

**Implementation sketch:**
Pre-processing step (data pipeline, not model change):
```python
# For each surface point, find the SDF gradient direction
# Walk 50 steps of 0.002*L_car along normal direction in volume point cloud
# Find first crossing of SDF > threshold (e.g., 0.05)
# delta_99 = step_distance where crossing occurs
# Normalize delta_99 by mean delta across training set
```
Flag: `--delta-99-surface-feature` — enables pre-computation and concatenation.
Model change: input surface dim increases from 4 to 5 → first linear layer weights for the new dim initialized near zero.

```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --delta-99-surface-feature \
  --resume-from <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-delta99-feature \
  --kill-thresholds "500:train/loss<5,3000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=4

**Gate:** val_abupt < 7.35% by EP2. Per-axis: τ_y or τ_z must improve, not regress.

**Risk:** Medium. Pre-computation cost is small (nearest-neighbor in SDF field). The risk is that the SDF gradient walk in the volume point cloud is noisy for complex geometries (undercarriage, mirrors). The student should implement with a fallback to constant δ=0 for points where the threshold is never crossed.

**Difference from SDF input feature experiments:** This is a surface feature derived from the volume SDF, not volume SDF concatenation. It encodes local turbulent boundary layer thickness rather than distance-to-wall. No prior experiment has tried this specific feature.

---

## Card 5: Slice-Token Rotary Positional Encoding (RoPE on Transolver Latents)

**Priority: 5 — High**

**Title:** Rotary positional encoding (RoPE) applied to Transolver slice centroid positions

**Hypothesis:**
The current Transolver architecture uses STRING-sep learned positional encodings on surface/volume *input points*. But the transformer's attention operates on *slice tokens* — abstract latent representations of groups of points sampled by the slice mechanism. These slice tokens are assigned positional meaning only implicitly, through the mean coordinates of their assigned points. Applying RoPE directly to the slice token positions (using slice centroid x,y,z coordinates as RoPE dimensions) would give the transformer explicit relative positional awareness at the latent level, allowing attention heads to attend more precisely to spatially adjacent slices. This directly addresses τ_y/τ_z, which require spatially coherent wall-shear gradients across adjacent slices.

**Expected mechanism:**
RoPE encodes relative position by rotating query/key vectors by an angle proportional to position difference. For slice tokens at positions p_i and p_j, attention weight becomes sensitive to |p_i - p_j| via the rotation angle. This biases attention toward nearby slices, which is physically appropriate for wall shear (τ_y/τ_z are smooth fields with spatial coherence length ~0.01L_car). The STRING-sep PE acts on input points before aggregation; RoPE on slice tokens acts on the aggregated latent representations — a complementary level of positional encoding.

**Implementation sketch:**
- Compute slice centroids as mean of assigned surface points (done naturally in Transolver sampling)
- Apply 3D RoPE: encode centroid (cx, cy, cz) as RoPE position, with separate frequency bands for each spatial dimension
- Modify the transformer self-attention Q, K projections to apply the RoPE rotation before dot-product
- Keep the existing STRING-sep PE on input points unchanged
- Use RoPE `dim_per_head=16` (6 dims for xyz × 2 sin/cos, rounded to next multiple of 2)

```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --slice-rope \
  --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --resume-from <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-slice-rope \
  --kill-thresholds "500:train/loss<5,3000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=4

**Gate:** val_abupt < 7.35% by EP2. Any individual-axis improvement with no regression counts as partial success.

**Risk:** Medium-high. RoPE on 3D positions is non-trivial to implement correctly — the standard 1D RoPE extends to 3D via interleaved frequency pairs per axis. Key implementation risk: slice centroids change across batches (stochastic sampling), so RoPE must be recomputed per batch rather than cached. Student must confirm no weight initialization is needed for the RoPE rotation (it is parameter-free).

**Difference from STRING-sep PE:** STRING-sep PE operates on input points before aggregation. Slice-token RoPE operates on the aggregated latent vectors. Orthogonal levels of the architecture.

---

## Card 6: Gradient Surgery (PCGrad) for Surface vs Volume Multi-Task Conflict

**Priority: 6 — Medium-High**

**Title:** PCGrad gradient surgery to eliminate τ_y/τ_z vs vol_p gradient conflicts in multi-task training

**Hypothesis:**
Surface wall shear (τ_y, τ_z) and volume pressure (vol_p) have competing gradient directions during backpropagation through the shared trunk. When their gradients are negatively correlated (pointing in opposite directions in weight space), the optimizer makes small net updates that benefit neither task. Yu et al. (2020) PCGrad projects each task's gradient onto the normal plane of any conflicting task gradient, eliminating the negative correlation. Applied to the (τ_y, τ_z) task group vs the vol_p task group, this should improve both without adding parameters.

**Expected mechanism:**
PCGrad detects when `g_A · g_B < 0` (dot product negative) for task A and B gradients, and replaces `g_A` with `g_A - (g_A · g̃_B / |g̃_B|²) g̃_B` where g̃_B is the unit vector of g_B. This removes the component of g_A that conflicts with g_B. For our 5-output prediction problem, the natural task groups are: (τ_y, τ_z) as Task A and (τ_x, surface_p, vol_p) as Task B. Task group A is the bottleneck. PCGrad operates at the loss level, not the architecture level — zero architectural change required.

**Implementation sketch:**
```python
# PCGrad implementation (PyTorch)
class PCGrad:
    def __init__(self, optimizer, task_groups):
        # task_groups: list of output channel indices
        # e.g., [[1, 2], [0, 3, 4]] for (tau_y, tau_z) vs rest
        ...
    def step(self, losses):
        # Compute per-task gradient vectors
        # Project conflicting gradients
        # Apply net gradient
```
Flag: `--pcgrad --pcgrad-task-groups "1,2:0,3,4"` (tau_y/tau_z vs tau_x/sp/vp)

```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --pcgrad --pcgrad-task-groups "1,2:0,3,4" \
  --resume-from <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-pcgrad-surface-volume \
  --kill-thresholds "500:train/loss<5,3000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=4

**Gate:** τ_y + τ_z combined must improve vs SOTA by EP2. If PCGrad increases training time by >30%, close (compute budget exceeded).

**Risk:** Medium. PCGrad requires computing per-task gradients separately (2× backward passes per step or per-channel loss splitting). This ~doubles per-step training time. If the training speed drops significantly, this approach is not viable within the epoch limit. Student should profile before committing to full run.

**Implementation reference:** PCGrad GitHub: https://github.com/WeiChengTseng/Pytorch-PCGrad

---

## Card 7: Width Expansion to 640d/10h with Constant FLOP Budget

**Priority: 7 — Medium**

**Title:** Transolver width scale-up to hidden_dim=640 / heads=10 with reduced slices

**Hypothesis:**
The current SOTA uses 4L/512d/8h/128sl. The yi branch has not explored width > 512d — only depth (6L/512d in #714, 4L/768d in #659, both failed in cold-start). A width expansion that stays within VRAM budget by reducing slices from 128 to 96 may find a better parameter allocation: wider attention heads may better capture the spatial correlations needed for τ_y/τ_z. The key insight from prior failures (#659, #714) is that they were cold-start runs; a warm-start from SOTA with the extra width capacity added incrementally (pad existing 512d weights to 640d with zero initialization) may stabilize the transition.

**Expected mechanism:**
Wider hidden dimensions increase the expressivity of each attention head's Q/K/V projections and the MLP per-token transform. For τ_y/τ_z, which depend on fine-grained spatial gradients near separation points, additional hidden dimension allows richer pairwise feature comparisons across slice tokens. The slices-from-128-to-96 tradeoff: fewer slices per sample, but each slice attends to a richer embedding. VRAM calculation: 640×640×4L×2 (Q+K) ≈ +35% attention param count, -25% slice tokens → net ~0% FLOP change.

**Implementation sketch:**
```python
# Weight initialization from 512d SOTA checkpoint:
# - For all linear layers: pad existing [512×...] weights to [640×...] with N(0, 0.001) noise
# - For attention Q/K/V: pad the extra 128 head dims with near-zero weights
# - This ensures model starts close to SOTA behavior with extra capacity near zero
```
Flag: `--model-hidden-dim 640 --model-heads 10 --model-slices 96 --expand-from-512d`

```bash
python train.py \
  --model-layers 4 --model-hidden-dim 640 --model-heads 10 --model-slices 96 \
  --optimizer lion --lr 1e-6 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --expand-from-512d <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-640d-10h-warm \
  --kill-thresholds "500:train/loss<5,5000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=4 (check if 640d VRAM fits on 4×A100 96GB)

**Gate:** val_abupt < 7.35% by EP2 (warm start must recover SOTA within 1 epoch). If EP1 val > 8.0%, close.

**Risk:** High. Weight padding from 512d to 640d is non-trivial to implement without corruption. The VRAM budget for 640d on 4-GPU DDP needs verification. Student should run a 100-step smoke test first to confirm VRAM does not OOM.

---

## Card 8: Temperature-Annealed β-NLL (β Curriculum During Training)

**Priority: 8 — Medium**

**Title:** Anneal the β heteroscedastic loss weight from 0.0→0.5 over the first 10k steps

**Hypothesis:**
The β-NLL loss (Seitzer et al. 2022) with β=0.5 down-weights high-uncertainty predictions. At initialization, the model has high uniform uncertainty everywhere, so β=0.5 immediately reduces gradients for all points equally — including τ_y/τ_z which should receive stronger gradient signal early in training. Annealing β from 0.0 (standard MSE, full gradient) at step 0 to 0.5 at step 10k would allow the model to learn the mean function first (full gradient signal for τ_y/τ_z) before trusting its uncertainty estimates to down-weight gradients. This is analogous to the β-VAE curriculum used in generative models.

**Expected mechanism:**
Early training: β=0 → `loss = (y-mu)^2` for all points. The model receives full gradient signal even in high-uncertainty regions. Late training: β ramps to 0.5 → uncertainty-weighted loss reduces the gradient from well-predicted easy points, emphasizing hard τ_y/τ_z regions. The ramp prevents the "uncertainty collapse" where the model learns to inflate uncertainty everywhere to reduce loss rather than improve predictions.

**Implementation sketch:**
```python
# In training loop:
beta_current = min(0.5, 0.5 * step / 10000)  # linear ramp over 10k steps
loss = beta_nll(predictions, targets, beta=beta_current)
```
Flag: `--beta-nll-beta 0.5 --beta-nll-anneal-steps 10000` (current: `--beta-nll-beta 0.5` fixed throughout)

```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --beta-nll-beta 0.5 --beta-nll-anneal-steps 10000 \
  --grad-ema-alpha 0.5 \
  --resume-from <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-beta-anneal \
  --kill-thresholds "500:train/loss<5,3000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=4

**Gate:** val_abupt < 7.35% by EP2. τ_y improvement primary signal.

**Risk:** Low-medium. This is a purely training dynamics change, no architectural modification. The warm-start from SOTA means β starts at 0.0 for the first ~3k steps (resetting the uncertainty weighting) then ramps. Risk: briefly setting β=0 on a converged model may cause a transient loss spike.

---

## Card 9: Pressure-Gradient Surface Feature (FD Estimate of ∇p_s)

**Priority: 9 — Medium**

**Title:** Surface pressure gradient as an additional point-wise input feature for wall shear prediction

**Hypothesis:**
Wall shear stress is physically coupled to the local pressure gradient through the boundary layer momentum equation: τ_wall ∝ (dp/ds) integrated across the boundary layer. Regions of adverse pressure gradient (∇p > 0 in flow direction) tend to have reduced τ, while favorable gradient regions have elevated τ. Estimating ∇p_s at each surface point from the training set (using finite differences on a local surface patch of ~10 nearest neighbors) and providing this as an additional surface input feature would give the model a physics-grounded spatial cue for where τ_y and τ_z are expected to be large or small.

**Expected mechanism:**
For each surface point, estimate `dp/ds ≈ (p_neighbor - p_center) / |x_neighbor - x_center|` using k=6 nearest surface neighbors, projecting onto the estimated local flow direction (approximated as y-axis for baseline, or from freestream direction). Concatenate [dp/ds_x, dp/ds_y, dp/ds_z] to surface point features. The model can then condition τ_y/τ_z predictions on the local pressure gradient state. This feature is computable at inference time since surface_p predictions are made simultaneously — but as an input feature it requires ground-truth p_s during training.

**Implementation note:** This is a training-only feature (uses GT surface p labels as input). At inference, must use predicted p_s from a first forward pass or the model's own p_s output. This creates a two-pass inference requirement OR the model must learn to use its own p_s output as an auxiliary conditioning signal (similar to teacher forcing). The simpler single-pass approach: predict p_s in the first half of the forward pass, compute ∇p_s from predicted p_s, then pass ∇p_s to the τ head. This requires architectural modification to the head.

```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --pressure-gradient-feature \
  --resume-from <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-pressure-grad-feature \
  --kill-thresholds "500:train/loss<5,3000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=4

**Gate:** τ_y + τ_z combined must improve by EP2. If both regress, close.

**Risk:** High. The two-pass inference requirement is a significant implementation complexity. The student should implement the simpler "GT p_s as additional input at train time" first (teacher-forcing variant), with explicit documentation that test-time behavior will differ. This is an exploratory high-risk card.

---

## Card 10: Decoder-Level Manifold Mixup for τ_y/τ_z

**Priority: 10 — Medium**

**Title:** Manifold mixup in the slice-token embedding space, restricted to same-shape-class pairs

**Hypothesis:**
Data augmentation via mixup on input points (#727) failed catastrophically because it created chimeric vehicles at the input level (cross-stream trunk corruption). Manifold mixup (Verma et al. 2019) applies interpolation in the *latent* space — specifically between the slice token embeddings of two training samples — rather than between input points. The mixed embedding is still within the learned latent manifold (approximately), avoiding the chimera problem at the input level. To further prevent semantically invalid mixing, restrict pairs to same-shape-class vehicles (hatchback+hatchback, fastback+fastback) identified by a clustering of geometry features.

**Expected mechanism:**
After the encoder's final transformer layer, the slice token embeddings H_A and H_B for two same-class vehicles are mixed: H_mixed = λ H_A + (1-λ) H_B, with λ ~ Beta(0.2, 0.2). The decoder produces predictions for H_mixed. Labels are mixed: y_mixed = λ y_A + (1-λ) y_B. The mixed latent still captures a plausible vehicle aerodynamics because both A and B are geometrically similar. The augmentation doubles effective training data and regularizes the τ_y/τ_z prediction by interpolating between physically valid solutions.

**Implementation sketch:**
Flag: `--manifold-mixup --mixup-alpha 0.2 --mixup-layer-idx -1 --mixup-shape-class-aware`
- `--mixup-layer-idx -1` applies mixup after the last transformer layer (before decoder heads)
- `--mixup-shape-class-aware` restricts mixing to pre-clustered shape groups (student must implement simple k-means on geometry features at dataset load time)

```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --manifold-mixup --mixup-alpha 0.2 \
  --resume-from <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-manifold-mixup \
  --kill-thresholds "500:train/loss<5,3000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=4

**Gate:** val_abupt < 7.35% by EP2. If EP1 gate triggers (val > 9%), close immediately.

**Risk:** Medium-high. Manifold mixup changes the loss surface unpredictably at late training. The shape-class-aware pairing requires implementation. The core risk: if the decoder is highly nonlinear, mixed latents may not decode to valid interpolations.

---

## Card 11: Checkpoint Voting Ensemble (Per-Epoch Majority Vote)

**Priority: 11 — Low-Medium**

**Title:** Ensemble N per-epoch checkpoints from the yi SOTA run using coordinate-wise median voting

**Hypothesis:**
The yi SOTA run (dc031qpt) saved checkpoints every epoch for 4 epochs. Each checkpoint represents a slightly different point in weight space. Unlike EMA averaging (which is a running mean along the trajectory), a per-epoch checkpoint ensemble can combine checkpoints that may have individually learned different aspects of the prediction (EP2 may have better τ_y; EP3 may have better τ_z). Using coordinate-wise median voting on predictions (rather than mean) is more robust to individual checkpoint outliers.

**Expected mechanism:**
For N checkpoint predictions [y1, y2, ..., yN] at each point, compute the coordinate-wise median. The median is more robust than the mean to any single checkpoint with a pathological prediction. This is computationally cheap (inference only, no training) and requires only the saved checkpoint files. The expected gain is ~0.1-0.3pp from checkpoint diversity.

**Implementation sketch:**
```python
# Load K checkpoints from the SOTA run
# Forward pass all K checkpoints on val/test set
# Per-point coordinate-wise median of predictions
# Report val_abupt on ensemble vs individual checkpoints
```
Flag: `--ensemble-checkpoint-voting --checkpoints "ep1.pt ep2.pt ep3.pt ep4.pt"`

**Gate:** val_abupt < 7.35% (better than best single checkpoint 7.3767%). If no improvement over best individual, close.

**Risk:** Low. Inference-only change, no training required. Main risk: the K checkpoints from dc031qpt may not be available (only terminal checkpoint saved). Student must verify that intermediate checkpoints are accessible from the W&B artifact.

---

## Card 12: GELU → SiLU Activation Swap in All MLP Blocks

**Priority: 12 — Low**

**Title:** Replace GELU activations with SiLU (Swish) in all MLP blocks

**Hypothesis:**
SiLU (x·σ(x)) and GELU (x·Φ(x)) are nearly identical in expectation but differ in higher derivatives. SiLU has smoother gradients in the positive regime and a slightly different negative tail. In recent language models (LLaMA, Mistral), SiLU has consistently outperformed GELU on downstream tasks at equivalent compute. For the Transolver MLP blocks, the difference may be minimal — but as a near-zero-cost experiment (one-line change), the expected reward/risk ratio is favorable if Lion optimizer's per-weight adaptive steps interact slightly differently with SiLU gradients.

**Implementation sketch:**
Flag: `--mlp-activation silu` (current: GELU)

```bash
python train.py \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --optimizer lion --lr 3e-7 --weight-decay 0.05 --clip-grad-norm 1.0 \
  --use-ema --ema-decay 0.9999 --ema-start-step 0 \
  --learnable-pe --string-sep-pe \
  --beta-nll-beta 0.5 \
  --grad-ema-alpha 0.5 \
  --mlp-activation silu \
  --resume-from <yi_sota_dc031qpt_checkpoint> \
  --wandb_group round42-silu-activation \
  --kill-thresholds "500:train/loss<5,3000:val_primary/abupt_axis_mean_rel_l2_pct<8.5"
```
SENPAI_MAX_EPOCHS=3 (short viability check only)

**Gate:** val_abupt < 7.35% by EP2. If no improvement, close — do not extend.

**Risk:** Low. SiLU is a drop-in replacement. Main risk: loading a GELU-trained checkpoint and switching activations mid-training. Student should verify that the SOTA checkpoint's MLP block activations are indeed GELU before applying this flag, and note that a brief initial val regression is expected as the activation outputs re-calibrate.

---

## Summary Rankings

| Rank | Card | Title | Primary Target | Risk | Expected Impact |
|------|------|-------|---------------|------|----------------|
| 1 | Y-Symmetry Pair Loss (stabilized) | Mechanism confirmed (#671 EP2 ~8.17%), needs stabilization | τ_y, τ_z | Medium | Very High |
| 2 | Cosine LR Warm-Restart | Escape shallow optima; addresses Lion plateau (#671, #715) | τ_y, τ_z (all) | Medium | High |
| 3 | Cholesky Correlated NLL | Model joint τ_y/τ_z covariance; orthogonal to β-NLL | τ_y, τ_z | Med-High | High |
| 4 | δ_99 BL Thickness Feature | Physics-grounded wall-shear spatial conditioning | τ_y, τ_z | Medium | High |
| 5 | Slice-Token RoPE | RoPE at latent level (untried), spatial coherence for τ | τ_y, τ_z | Med-High | Medium-High |
| 6 | PCGrad Gradient Surgery | Eliminate multi-task gradient conflicts | τ_y, τ_z | Medium | Medium-High |
| 7 | 640d/10h Width Expansion | More representational capacity at constant FLOP | All | High | Medium |
| 8 | Annealed β-NLL | Better early gradient signal for τ_y/τ_z | τ_y, τ_z | Low-Med | Medium |
| 9 | Pressure-Gradient Feature | Physics-coupled input feature for wall shear | τ_y, τ_z | High | Medium |
| 10 | Manifold Mixup | Latent-space augmentation (avoids chimera problem of #727) | τ_y, τ_z | Med-High | Medium |
| 11 | Checkpoint Voting Ensemble | Zero-training inference ensemble; needs checkpoint availability | All | Low | Low-Med |
| 12 | GELU → SiLU Activation | Near-zero-cost activation swap | All | Low | Low |
