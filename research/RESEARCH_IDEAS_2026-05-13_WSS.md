# WSS-Focused Research Hypotheses
## Generated: 2026-05-13 — Response to Issue #1056 Directive

**Directive:** Beat Wall Shear Stress SOTA — drive `test_WSS` below Transolver-3's 5.85%, while simultaneously keeping `test_vol_p ≤ 3.643%` and `test_SP ≤ 3.577%` (PR #972 levels).

---

## WSS Gap Summary

| Configuration | test_WSS | test_vol_p | test_SP | test_ABUPT |
|---|---:|---:|---:|---:|
| **Target** | **< 5.85%** | ≤ 3.643% | ≤ 3.577% | — |
| PR #1059 K=4 ensemble (current SOTA) | 6.330% | 3.889% ⚠ | 3.366% | 5.594% |
| PR #972 SDF-stratified (best single) | 6.727% | 3.643% | 3.577% | 5.844% |
| PR #968 stoch vol subsampling | 6.825% | 3.957% | 3.673% | 5.986% |
| PR #958 vol_p aux head | 6.985% | 3.818% | 3.911% | 6.107% |
| PR #823 surf→vol xattn | ~7.05% | — | — | ~6.2% |

**Gap to close:** 6.330% → 5.85% = **0.48pp** (ensemble-level), 6.727% → 5.85% = **0.88pp** (single-model).

---

## Historical WSS Data (for context)

**Per-axis old dataset results (WSS = mean of tau_x, tau_y, tau_z):**
- PR #958: tau_x=6.17%, tau_y=7.66%, tau_z=9.51% (val) → WSS=7.78% mean
- PR #823: tau_x=5.78%, tau_y=7.60%, tau_z=9.01% (val) → WSS=7.46% mean — best per-axis historically
- PR #571 (tau_y×1.5, tau_z×2.0): tau_z improved 10.27%→10.00% (−0.27pp on that axis)
- PR #516 (tau_y×1.2, tau_z×1.3): WSS=7.757%

**Key insight:** tau_z is consistently the worst axis (~9–10% old dataset, still worst on corrected split: GradNorm at EP1 of PR #1058 shows tau_z weight=1.59, tau_y=1.28, tau_x=0.83 — tau_z upweighted 1.9× more than tau_x). Any WSS improvement strategy must prioritize **tau_z improvement**.

**What has moved WSS historically:**
1. Surf→vol cross-attention (PR #823): architecture win, best per-axis tau_x ever (5.78% val)
2. SDF-stratified vol sampling (PR #972): −0.26pp test_WSS vs PR #958
3. Fixed tau loss weights (PR #571): −0.27pp tau_z, −0.48pp test_WSS vs baseline
4. Ensemble diversity (PR #1059): 6.727% → 6.330%, −0.40pp vs single-model SOTA

**WSS gap = tau_z problem.** Closing tau_z from ~current to competitive is the bottleneck.

---

## Hypotheses — Ranked by Expected Impact

### H1: Stronger tau_z loss weighting — stack with PR #972 baseline (HIGH PRIORITY)

**Hypothesis:** Current baseline uses tau_y×1.5, tau_z×2.0. These weights were set on the old dataset where tau_z was ~9–10%. On the corrected split, the relative hardness of tau_z has increased (GradNorm sees tau_z=1.59 vs tau_x=0.83 at EP1). Increasing tau_z weight to 3.0–4.0 and tau_y to 2.0–3.0 should push the model to allocate more capacity to WSS axes without harming vol_p (which is now stable due to SDF-stratified sampling).

**Design:**
- Arm A: tau_y×2.0, tau_z×3.0 (moderate increase)
- Arm B: tau_y×2.5, tau_z×4.0 (aggressive increase)
- Base: PR #972 SDF-stratified stack (best single-model corrected-split baseline)

**Gate:** EP3 val_abupt ≤ 8.0%; full run to EP13.
**Win criterion:** test_WSS < 6.50% AND test_vol_p ≤ 3.65% (small improvement is a start; must not break vol_p constraint).

**Reproduce (Arm A):**
```bash
cd target/ && torchrun --standalone --nproc_per_node=8 train.py \
  --agent <student> --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --tau-y-loss-weight 2.0 --tau-z-loss-weight 3.0 --surface-loss-weight 2.0 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --lr-cosine-t-max 13 --epochs 13 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --no-compile-model \
  --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --use-surf-to-vol-xattn \
  --sdf-importance-sampling --sdf-alpha 4.0 \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --wandb-group wss-tau-weight-sweep
```
(Arm B: same but `--tau-y-loss-weight 2.5 --tau-z-loss-weight 4.0`)

---

### H2: WSS-targeted greedy ensemble — rerun selection optimizing val_WSS (HIGH PRIORITY)

**Hypothesis:** PR #1059 K=4 ensemble was selected to minimize val_abupt (mean of 5 channels). WSS is 1 of 5 channels and is consistently the worst, so val_abupt-optimal selection may deprioritize WSS diversity. Selecting members by val_WSS instead of val_abupt should yield an ensemble with better WSS coverage — models that disagree on WSS (error correlation < 1.0) provide the most diversity benefit.

**Design:**
- Pool: all corrected-split trained models (same pool as PR #1059 + any new PR #1058/1065/1057 trained models)
- Selection: greedy forward selection minimizing val_WSS (instead of val_abupt)
- Report: val_WSS, test_WSS, val_abupt, test_abupt, test_vol_p
- Critical constraint: test_vol_p ≤ 3.643% AND test_SP ≤ 3.577% must hold after selection

**Produce command:**
```bash
cd target/
uv run python ensemble_eval.py \
  --greedy \
  --greedy-metric val_WSS \
  --pred-cache-dir outputs/ensemble_cache_corrected_20260513/pred_cache \
  --candidate-run-ids 56bcqp3m 29nohj67 a0yoxy85 ghh0s4ne <any_new_corrected_runs> \
  --max-k 8 \
  --wandb-group wss-targeted-ensemble \
  --wandb-name greedy-wss-metric-corrected
```

**Note:** If `ensemble_eval.py` doesn't support `--greedy-metric`, the student should modify the greedy selection criterion to minimize the WSS component of the ensemble predictions. The cached predictions from PR #1059 can be reused — no new training needed for the baseline pool.

---

### H3: Surface-normal-frame WSS prediction (NEW ARCHITECTURE IDEA, MEDIUM-HIGH PRIORITY)

**Hypothesis:** Current model predicts WSS in global (x, y, z) Cartesian frame. Wall shear stress is a tangential force — it lives in the 2D tangent plane of the surface (perpendicular to the surface normal). Predicting WSS in local surface-tangent frame (t₁, t₂) — two orthogonal tangent vectors at each surface point — and converting back to global Cartesian at eval time should reduce the learning burden: the model no longer needs to learn that WSS has zero component along the surface normal.

This is analogous to haku PR #699 (surface-tangent frame) which had promising initial results. The key constraint is that the surface normal field is already available in the dataset (`surface_normals` tensor).

**Implementation:**
1. At each surface point, compute local frame: (t₁, t₂, n) from `surface_normals` using Gram-Schmidt
2. Rotate wall_shear target from (x, y, z) → (t₁, t₂) during training (project onto tangent plane)
3. Predict 2D tangential WSS; convert back to 3D (x, y, z) via inverse rotation for loss/eval
4. The surface normal channel is already in the model's input features — no new data loading needed

**Win criterion:** val_WSS < 6.2% (current single-model best is ~6.5% on corrected split implied by PR #972 test_WSS=6.727%).

---

### H4: SDF-stratified surface point sampling (extension of PR #972 to surface) — MEDIUM-HIGH PRIORITY

**Hypothesis:** PR #972 applied SDF-importance-sampling to volume points and got the single-model SOTA. The same idea applied to **surface** sampling: oversample high-SDF-gradient regions (edges, wheel arches, underbody) which are physically where WSS is highest and most variable. A-pillar and wheel-arch/underbody surfaces are WSS hotspots; oversampling them during training reduces the effective mean-squared-error contribution from low-WSS flat regions.

**Design:**
- Apply importance sampling to `train-surface-points` selection based on local surface geometry (curvature proxy from normals, or use surface_sdf gradient magnitude if available)
- Alternative: use existing curvature features from the geometry preprocessor to construct sampling weights
- Arm A: curvature-based surface importance sampling (κ proportional weight)
- Arm B: geometric-magnitude-based (sample proportional to |surface_normal × gradient| or similar proxy)

**Note:** If SDF information is not directly available per surface point in the dataset, use point-cloud local variation (via KNN distance variance) as a curvature proxy.

---

### H5: WSS-specific auxiliary prediction head with direction decomposition (MEDIUM PRIORITY)

**Hypothesis:** WSS is a 3-vector field (tau_x, tau_y, tau_z). Instead of predicting all 3 jointly, predict: (a) WSS magnitude |τ| and (b) WSS direction unit-vector (τ/|τ|) separately, then multiply at inference. Magnitude and direction may have different optimization landscapes — magnitude is a scalar regression problem while direction is better framed as a unit-vector regression (e.g., with a cosine loss). This decomposition may reduce the effective learning difficulty, especially for tau_z which has highest variance.

**Implementation:**
1. Surface decoder outputs: 1-channel magnitude head + 3-channel direction head (L2-normalized)
2. Training loss: (a) MSE on log(1 + |τ|) [log-space magnitude regression] + (b) cosine similarity loss on direction (1 − cos_sim)
3. Eval: multiply magnitude × direction, compute standard Cartesian L2 metric
4. Base: PR #972 SDF-stratified stack

---

### H6: Hard example mining (OHEM) for high-WSS surface patches — MEDIUM PRIORITY

**Hypothesis:** WSS error is concentrated in high-shear regions (wheel arches, underbody, A-pillar). Most surface points have low WSS magnitude — the MSE loss is dominated by these easy low-shear regions. Online Hard Example Mining (OHEM) applied to surface points: for each batch, select the K% of surface points with highest current WSS prediction error and upweight their contribution to the loss.

**Design:**
- Maintain a per-surface-point running EMA of recent prediction errors
- At each training step, use the top-K% hardest points (by EMA error) as a supplementary loss term: `L_wss = L_wss_all + λ * L_wss_hard_top20pct`
- λ = 0.5 (moderate), top-K% = 20% of surface points

**Prior work:** Related to PR #509 surface OHEM — check that PR for implementation notes and lessons.

---

### H7: Mild yaw-only rotation augmentation for WSS (MEDIUM PRIORITY)

**Hypothesis:** PR #925 (yaw±5°/pitch±3°/p=0.5) failed EP3 gate; the EXPERIMENTS_LOG explicitly notes the wall_shear channel degrades most under aggressive rotation aug. The student's suggested follow-up: yaw-only (pitch=0), yaw±3°, p=0.3. Physical motivation: DrivAerML test cases likely differ primarily in yaw incidence angle. Mild yaw-only aug should help WSS generalization without introducing the high-entropy noise of pitch+yaw combined.

**Design (yaw-only mild, EP4 screen first):**
```bash
  --yaw-aug-max 3.0 --pitch-aug-max 0.0 --aug-prob 0.3 \
  --epochs 4 --lr-cosine-t-max 4 \
  --wandb-group wss-yaw-aug-mild
```
Win criterion at EP4: val_abupt < 6.9% AND val_WSS < 6.5%. If passes, run full 13-ep.

---

### H8: WSS-optimized pool expansion then WSS-targeted greedy ensemble (MEDIUM PRIORITY, COMBINATORIAL)

**Hypothesis:** The ensemble WSS benefit is limited by the diversity of the candidate pool. All current pool members were trained on the same PR #972/958/968 training recipe — similar training recipes produce correlated WSS errors. Explicitly training 2–3 new models with different WSS-targeted training recipes (from H1, H3, H5 above) and adding them to the ensemble pool should provide genuinely diverse WSS predictions.

**Design:**
1. Wait for H1 (stronger tau weights), H3 (surface normal frame), H5 (magnitude+direction head) results
2. Add any passing models to the corrected-split ensemble pool
3. Re-run greedy selection (both val_abupt and val_WSS criteria)
4. Report whether WSS-diverse pool yields test_WSS improvement

**This is a combinatorial hypothesis — only viable after H1-H5 produce at least 1-2 new passing models.**

---

### H9: RevIN-style per-case WSS normalization at inference — MEDIUM PRIORITY

**Hypothesis:** Per-case distribution shift in WSS: different geometry instances (body styles, ride heights, wheel configs) have different WSS magnitude distributions. At test time, normalizing WSS predictions per geometry instance (subtract instance mean, divide by instance std, then rescale back using test-time estimates of mean/std) could reduce distribution mismatch.

**Implementation:**
- Compute per-case mean/std from the model's own WSS predictions at test time (RevIN: use the predicted distribution itself)
- Apply linear recalibration: `τ_corrected = τ_pred * (σ_val_typical / σ_pred_test) + (μ_val_typical − μ_pred_test * ratio)`
- This requires no retraining — pure inference-time post-processing on the corrected-split test set

---

### H10: Longer training (45 epochs) on corrected split — LOWER PRIORITY (already running PR #1065)

**Note:** PR #1065 (stark, 45-epoch SDF-stratified baseline) is already testing this. Monitor closely for WSS trend — if WSS improves past EP13, this axis is confirmed and all future models should train longer.

---

## Assignment Queue (when students become idle)

**Priority order for WSS experiments:**

1. **H1 (stronger tau weighting)** — First idle student. Both Arm A and Arm B simultaneously if 2 students available. Directly attacks the tau_z bottleneck with minimal code change.

2. **H2 (WSS-targeted ensemble)** — Second idle student. No retraining needed — just rerun greedy selection. Fast experiment, high expected value if the pool already has diverse WSS error patterns.

3. **H7 (mild yaw-only aug)** — Third idle student. Start with 4-ep screen (fast, cheap). Prior work (PR #925) shows the failure mode clearly; this is a targeted fix.

4. **H3 (surface-normal-frame WSS)** — Fourth idle student. Requires implementation effort but is architecturally motivated and untested.

5. **H5 (magnitude + direction decomposition)** — Fifth idle student. Medium implementation complexity, interesting decomposition idea.

---

## Constraints (non-negotiable)

All WSS-targeted experiments MUST track:
- test_vol_p ≤ 3.643% (PR #972 baseline — must not regress)
- test_SP ≤ 3.577% (PR #972 baseline — must not regress)
- val_WSS as primary improvement metric (in addition to val_abupt)

If any PR achieves test_WSS < 6.0% (even a single checkpoint), report it immediately — this is significant directional progress.

---

## Notes on Currently Active WSS-Relevant PRs

- **PR #1058 (frieren GradNorm dynamic balancing)**: GradNorm naturally upweighting tau_z=1.59×, tau_y=1.28× at EP1 — directly WSS-relevant. Watch for EP3 gate results and per-axis tau metrics.
- **PR #1065 (stark 45-epoch extended training)**: Longer training on SDF-stratified baseline — will show if more epochs help WSS convergence on corrected split.
- **PR #1064 (nezuko K=3 ensemble)**: Dropping `ghh0s4ne` from pool should recover test_vol_p ≤ 3.643%; if it also tracks val_WSS we get a natural WSS-focused selection baseline.
- **PR #1057 (alphonse mild rotation aug)**: Similar to H7 — watch for WSS channel metrics especially.
