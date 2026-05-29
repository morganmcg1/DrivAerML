# SENPAI Research Ideas — 2026-05-29 17:50

Generated for: frieren, nezuko, askeladd (3 idle students)
Current SOTA: H253 val_abupt=5.9418% / test_abupt=5.7847%
Active in-flight: H267 (edward EP15+full-stack), H269 (alphonse anti-thetic+multi-res stacked), H270 (tanjiro TBD), H257 (thorfinn TBD)
Hard constraints: no ensembles, no capacity additions ≥+1% params, tau_z=1.67 locked, SENPAI_TIMEOUT_MINUTES=360, read-only: train.py, data/loader.py, data/preload.py, data/split_manifest.json

---

## Ranking by Expected Value

1. **H272** (Hutchinson curvature-inverse noise — nezuko) — Highest EV. Novel principled mechanism, never tried, directly addresses the "flat vs sharp curvature" heterogeneity that makes isotropic noise wasteful. The strongest theoretical grounding.
2. **H271** (Sobol QMC weight perturbation — frieren) — Medium EV. Distinct from anti-thetic (not just ±δ negation, but full low-discrepancy coverage); directly attacks the K=10 plateau identified in Finding KK-noise-saturation. Lower risk than H272.
3. **H273** (Taylor second-order correction — askeladd) — Exploratory. Theoretically motivated but the curvature correction adds variance; benefit uncertain. Most likely to regress, but has non-zero upside (~2bp) if the quadratic bias is systematic and stable.

---

## H271 — Sobol QMC Weight Perturbation

**Assigned to: frieren**
**EV rank: 2nd (Medium)**

### Mechanism

Finding KK-noise-saturation shows that random K=5→10 gains only +1.36bp and K=10→20 gains only +0.42bp: the Monte Carlo noise floor is hit quickly because i.i.d. Gaussian draws cluster. The core fix is to replace independent Gaussian samples with a scrambled Sobol low-discrepancy sequence via `torch.quasirandom.SobolEngine(d, scramble=True).draw(K)`, projected to standard normal via inverse CDF (`torch.distributions.Normal(0,1).icdf`). A Sobol sequence achieves O((log K)^d / K) star-discrepancy vs O(1/√K) for random, meaning the K draws cover the weight-perturbation ball much more evenly. This is fundamentally different from anti-thetic sampling (Finding NN-antithetic, H269): anti-thetic only cancels the linear gradient term by pairing ±δ for each draw, while Sobol additionally spreads draws across higher-order structure and fills gaps that random sampling leaves. Effective K increases without adding passes. In the 5e-4-σ regime the weight perturbation dimension is bounded by the number of effective parameters that actually matter (empirically, the Sobol engine needs `d` = a manageable projection dim, not the full 15.9M parameter count — we project to a d=512 or d=1024 subspace via a fixed random basis, then scale to the correct σ per-layer).

### Why it is complementary to in-flight work

H269 (alphonse) tests anti-thetic pairing stacked into 6-res+mirror. H271 tests a different mechanism: quasi-random coverage rather than anti-thetic cancellation. Both aim at the same K-plateau from different angles. These two results together will definitively characterize what fraction of the K-plateau is due to linear-term waste (anti-thetic) vs coarse coverage (QMC). They are non-overlapping by mechanism.

### Predicted val_abupt

5.929–5.941% (honest range: +0.5–1.5bp improvement over H253 K=5 random in stacked context; conservative because the projection subspace adds an approximation)

### Implementation plan

**Eval script**: New `eval_h271_sobol_tta.py` derived from `eval_multi_res.py`.

**Checkpoint**: `outputs/ensemble_cache/run-yw2a5dyl-epoch-13-ema/checkpoint.pt` (H185 EP13 EMA, same as H253)

**Key changes**:
- Add a `--weight-noise-mode` flag: `random` (baseline) vs `sobol` (new)
- Add `--sobol-proj-dim` flag (default: 1024)
- In the noise sampling function, replace:
  ```python
  delta = {k: torch.randn_like(v) * sigma for k, v in named_params}
  ```
  with:
  ```python
  engine = SobolEngine(dimension=proj_dim, scramble=True)
  sobol_pts = engine.draw(K)  # (K, proj_dim), uniform [0,1]
  normal_pts = Normal(0,1).icdf(sobol_pts.clamp(1e-6, 1-1e-6))  # (K, proj_dim)
  # project to per-param noise via a fixed random basis B (proj_dim, total_params)
  # B is generated once with seed=42 and cached
  flat_delta = normal_pts @ B * sigma  # (K, total_params)
  # then split flat_delta back into per-param tensors
  ```
- All other aspects of the 6-res + mirror + K=5 stack identical to H253

**Hyperparameters to test**:
- Primary: K=5, σ=5e-4, proj_dim=1024
- Secondary arm (if time): K=10, σ=5e-4, proj_dim=1024 (to test if QMC extends the useful K range)

**Suggested CLI**:
```bash
H185_CKPT="outputs/ensemble_cache/run-yw2a5dyl-epoch-13-ema/checkpoint.pt"
torchrun --standalone --nproc_per_node=8 target/eval_h271_sobol_tta.py \
  --checkpoint $H185_CKPT \
  --resolutions "32768,49152,65536,81920,98304,131072" \
  --eval-modes "mirror_res_weight_noise_avg" \
  --weight-noise-mode sobol \
  --weight-noise-sigma 5e-4 \
  --n-weight-noise-passes 5 \
  --sobol-proj-dim 1024 \
  --batch-size 2 --num-workers 4 \
  --wandb-name "frieren/h271-sobol-qmc-k5-proj1024" \
  --wandb-group "h271-frieren-sobol-qmc"
```

### Wall-clock estimate

~2.5–3h on DDP×8 (same pass count as H253: 5 noise × 6 res × 2 mirror = 60 passes; overhead is negligible — the Sobol draw is computed once before the pass loop)

---

## H272 — Hutchinson Diagonal Curvature-Inverse Noise Scaling

**Assigned to: nezuko**
**EV rank: 1st (Highest EV)**

### Mechanism

All prior weight-noise TTA (H253, H269) uses isotropic Gaussian noise: every parameter receives the same σ=5e-4 regardless of whether it sits in a flat or sharp region of the loss landscape. This is provably suboptimal. Parameters in sharp curvature regions should receive smaller perturbations (large H_ii means f changes rapidly; large δ adds noise to predictions), while flat-region parameters can tolerate larger perturbations (safe exploration). The Hutchinson trace estimator makes this tractable without computing the full Hessian: for M random Rademacher vectors {z_j}_{j=1}^M with z_j ∈ {±1}^n, the diagonal Hessian estimate is `diag(H) ≈ (1/M) Σ_j z_j ⊙ (H z_j)` where `H z_j = ∂²L/∂w² z_j` is the Hessian-vector product computed via two backward passes (`grad(grad(L, w) · z_j, w)`). Each HVP costs one forward+backward pair (same cost as a training step). With M=5–10 vectors on a single validation batch, we get a noisy but unbiased estimate of per-parameter curvature. The per-parameter σ is then:

```
σ_i = σ_base / sqrt(1 + β · |diag_H_i|)
```

where β≈1e4 (tunable). This is TTA with principled importance-weighted weight-space exploration: the same total "noise energy" is redistributed toward flat (robust) directions. No training required; the Hutchinson pass runs once before the TTA loop.

### Why it is complementary to in-flight work

H271 and H269 both address the K-plateau from a sampling perspective. H272 addresses a completely different dimension: the noise distribution shape, not the sampling strategy. Even with perfect QMC sampling, isotropic noise is suboptimal if the loss landscape is anisotropic. H272 is the first experiment to use gradient information about the model's landscape to guide TTA.

### Predicted val_abupt

5.928–5.942% (honest range: up to 3bp improvement if curvature is significantly heterogeneous; could break even if the landscape is nearly isotropic at this scale; unlikely to regress materially because the Hutchinson step only rescales σ downward for sharp params)

### Implementation plan

**Eval script**: New `eval_h272_hutchinson_tta.py` derived from `eval_multi_res.py`.

**Checkpoint**: `outputs/ensemble_cache/run-yw2a5dyl-epoch-13-ema/checkpoint.pt` (H185 EP13 EMA)

**Key changes**:
1. Pre-loop Hutchinson pass:
   ```python
   def hutchinson_diag_hessian(model, val_batch, M=8):
       model.eval()
       params = [p for p in model.parameters() if p.requires_grad]
       diag_H = [torch.zeros_like(p) for p in params]
       for _ in range(M):
           z = [torch.randint(0, 2, p.shape).float() * 2 - 1 for p in params]
           loss = compute_loss(model, val_batch)  # forward pass
           grads = torch.autograd.grad(loss, params, create_graph=True)
           gz = sum((g * zv).sum() for g, zv in zip(grads, z))
           hvp = torch.autograd.grad(gz, params, retain_graph=False)
           for i, (h, zv) in enumerate(zip(hvp, z)):
               diag_H[i] += zv * h / M
       return [dh.abs() for dh in diag_H]
   ```
2. Per-parameter σ scaling:
   ```python
   beta = 1e4
   sigma_per_param = [sigma_base / (1 + beta * dh).sqrt() for dh in diag_H]
   ```
3. Replace isotropic `torch.randn_like(v) * sigma` with `torch.randn_like(v) * sigma_i` in noise injection

**Hyperparameters to test**:
- Primary: M=8 Hutchinson vectors, σ_base=5e-4, β=1e4, K=5, 6-res+mirror stack
- Secondary: β sweep {1e3, 1e4, 1e5} if primary shows sensitivity (cheaply done with the same diag_H cached)

**Suggested CLI**:
```bash
H185_CKPT="outputs/ensemble_cache/run-yw2a5dyl-epoch-13-ema/checkpoint.pt"
torchrun --standalone --nproc_per_node=8 target/eval_h272_hutchinson_tta.py \
  --checkpoint $H185_CKPT \
  --resolutions "32768,49152,65536,81920,98304,131072" \
  --eval-modes "mirror_res_weight_noise_avg" \
  --weight-noise-sigma 5e-4 \
  --hutchinson-m 8 \
  --hutchinson-beta 1e4 \
  --n-weight-noise-passes 5 \
  --batch-size 2 --num-workers 4 \
  --wandb-name "nezuko/h272-hutchinson-curvature-k5-m8-beta1e4" \
  --wandb-group "h272-nezuko-hutchinson-curvature"
```

### Wall-clock estimate

~3–3.5h on DDP×8. Hutchinson pre-pass: M=8 HVPs × 8 val batches ≈ 15–20 min overhead. Main TTA: same as H253 (~230 min). Total: ~250–260 min, within the 360-min timeout.

**Note on DDP**: The Hutchinson HVP requires `create_graph=True` which may interact with DDP. Recommend running Hutchinson on rank-0 only, then broadcasting `diag_H` to all ranks before the TTA loop. This is clean and avoids DDP graph issues.

---

## H273 — Higher-Order Taylor Curvature Correction to Base Prediction

**Assigned to: askeladd**
**EV rank: 3rd (Exploratory)**

### Mechanism

Standard weight-noise TTA estimates `E[f(w+ε)] ≈ f(w) + ½σ²·tr(H_w)` to second order. The correction term `½σ²·tr(H_w)` is a systematic bias that shifts ensemble predictions relative to the point estimate `f(w)`. Anti-thetic sampling (H269) cancels the linear term `∇f(w)·ε` by pairing ±δ. H273 goes further: it explicitly estimates and removes (or corrects for) the quadratic bias. For K paired samples (w+ε_k, w−ε_k), the second-order residual per pair is:

```
r_k = [f(w+ε_k) + f(w−ε_k) - 2·f(w)] / 2
    ≈ ε_k^T H_w ε_k / 2    (the curvature bias for this draw direction)
```

The corrected prediction is:

```
f_corrected = f(w) - λ · mean_k(r_k)
```

where λ ∈ [0, 1] is a blending coefficient (λ=1 is full correction, λ=0 is base prediction). The key question is whether this curvature bias is systematic across the val set. If H_w is consistently positive-definite (a convex bowl near the optimum), ensemble predictions are consistently upward-biased relative to the true optimum, and r_k > 0 systematically — then correction helps. λ is tuned on val before final test.

This is distinct from anti-thetic (H269): H269 averages f(w+ε)+f(w−ε) to get a better estimate of f(w); H273 uses the same paired samples to estimate and correct the second-order Taylor bias in f(w) itself.

### Why it is complementary to in-flight work

H269 (alphonse) tests anti-thetic averaging; H273 tests what to do with the second-order residual that anti-thetic averaging discards. If H269 succeeds, H273 is a follow-up that squeezes additional signal from the same pass budget.

### Predicted val_abupt

5.930–5.945% (honest range: could gain ~2bp if curvature bias is systematic and positive, could regress ~1bp if r_k is noisy and adds variance; λ grid search on val provides a safety net at λ=0)

### Implementation plan

**Eval script**: New `eval_h273_taylor2_tta.py` derived from `eval_multi_res.py`.

**Checkpoint**: `outputs/ensemble_cache/run-yw2a5dyl-epoch-13-ema/checkpoint.pt` (H185 EP13 EMA)

**Key changes**:
1. Compute base prediction `f_base = f(w)` (1 pass, no noise)
2. Compute K=3 anti-thetic pairs: `f_plus[k], f_minus[k]` = f(w+ε_k), f(w−ε_k)
3. Compute anti-thetic mean (same as H269 output): `f_anti = mean_k[(f_plus[k]+f_minus[k])/2]`
4. Compute curvature residuals: `r_k = (f_plus[k] + f_minus[k] - 2*f_base) / 2`
5. Final output: `f_out = f_anti - λ * mean_k(r_k)`
6. λ grid search: {0.0, 0.05, 0.1, 0.15, 0.2, 0.3} — pick best on val, then run test

**Pass budget**: 1 base + 6 noise (K=3 pairs × 2) = 7 passes × 6-res × 2-mirror = 84 total passes vs H253's 60. Approximately 30% more compute.

**Suggested CLI**:
```bash
H185_CKPT="outputs/ensemble_cache/run-yw2a5dyl-epoch-13-ema/checkpoint.pt"
torchrun --standalone --nproc_per_node=8 target/eval_h273_taylor2_tta.py \
  --checkpoint $H185_CKPT \
  --resolutions "32768,49152,65536,81920,98304,131072" \
  --eval-modes "mirror_res_weight_noise_avg" \
  --weight-noise-sigma 5e-4 \
  --n-antithetic-pairs 3 \
  --taylor-lambda 0.1 \
  --batch-size 2 --num-workers 4 \
  --wandb-name "askeladd/h273-taylor2-correction-k3-lam0.1" \
  --wandb-group "h273-askeladd-taylor2"
```

Run λ sweep on val first (cheaply, using cached per-sample predictions), then run test with the best λ.

### Wall-clock estimate

~3h on DDP×8 (84 passes × overhead factor ≈ ~280 min; borderline on the 360-min budget; recommend 6-res only, skip mirror on first arm to test the mechanism cheaply in ~160 min, then add mirror only if val result is promising)

---

## Summary Table

| ID | Title | Assigned | EV Rank | Predicted val_abupt | Wall Clock | Mechanism |
|----|-------|----------|---------|---------------------|------------|-----------|
| H271 | Sobol QMC weight perturbation | frieren | 2nd (Medium) | 5.929–5.941% | ~2.5–3h | Low-discrepancy coverage of weight-noise ball; attacks K-plateau differently than anti-thetic |
| H272 | Hutchinson curvature-inverse noise | nezuko | 1st (Highest) | 5.928–5.942% | ~3–3.5h | Per-parameter σ inversely scaled by diagonal Hessian; spends noise budget in flat-landscape directions |
| H273 | Taylor 2nd-order correction | askeladd | 3rd (Exploratory) | 5.930–5.945% | ~3h | Estimates and subtracts quadratic ensemble bias using anti-thetic residuals; λ tuned on val |

## Non-overlap verification

- H269 (alphonse): anti-thetic ±δ pairing in stacked context. NOT H271 (Sobol coverage), NOT H272 (curvature-weighted σ), NOT H273 (quadratic residual correction).
- H267 (edward): EP15 extension + full stack. None of H271/H272/H273 touch the checkpoint or training epochs.
- H270 (tanjiro): TBD (not anti-thetic, not Sobol, not curvature — assumed distinct).
- H257 (thorfinn): TBD (assumed distinct).
- Banked: mirror-y ✓, 6-res ✓, EP15 ✓, noise K=5 ✓, stacking ✓, SWA-null ✓, anti-thetic standalone ✓. None of the banked findings covers QMC sampling, curvature-weighted noise, or Taylor correction.
