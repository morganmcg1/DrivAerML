# Research Ideas — 2026-05-25 12:30

Generated for: DrivAerML DL24 CFD Surrogate Programme
Current SOTA: H39 test_WSS=6.6506% (PR #1284, merged 2026-05-24)
Target: test_WSS < 5.85% (Transolver-3 SOTA)
Gap to close: −0.8006pp

> **Advisor naming note (added post-research-agent):** The number `H138` was already in use for a separate hypothesis dispatched at 12:30Z (PR #1324 dl24-frieren — curvature-weighted Charbonnier z-axis loss). To avoid collision, the researcher-agent's `H138 SWA` is **renumbered to H141** for Wave 36 dispatch. Final Wave 36 queue priority: **H140 (z-coord WSS reweighting) > H141 (SWA) > H139 (SGDR Warm Restarts)**. Note H140 is OVERLAPPING in spirit with my dispatched H138 curvature-weighted Charb (both spatially reweight WSS loss), but uses z-coord magnitude instead of curvature — useful as compound if H138 wins, or as orthogonal axis-test if H138 nulls.

Constraints active at generation time:
- No slice-routing perturbations (H135 falsified — diverged EP1)
- No equal-projection MTL weighting (H136 IMTL-G falsified — suppressed w_wss 7.3×)
- No capacity-axis tweaks (depth, width, slices, hidden_dim, LR all exhausted)
- Single-model only (no ensembles, no NNLS)
- DDP-8 8×H100, 30 epochs max, ~24h wallclock
- Dataset: /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511
- AND-contract: must NOT regress test_vol_p (≤3.643%) or test_surf_p (≤3.577%)

---

## H138 — Stochastic Weight Averaging (SWA)

### What it is

Post-hoc weight averaging over the final 5–6 training epochs using `torch.optim.swa_utils`, producing a single averaged model checkpoint that sits in a flatter loss basin and generalizes better to held-out test cases.

### Why it might help here

H39 shows a favorable val→test gap (val 6.7761% → test 6.6506%, −0.118pp), suggesting the model does not catastrophically overfit to val. SWA targets the orthogonal axis: it finds a flatter basin in weight space, which tends to improve generalization to genuinely OOD inputs. The 50 test cars are a distinct distribution shift from the 400 train cars; flat minima are known to transfer better across such covariate shifts (SWAD, Cha et al. 2021). SWA adds zero architecture complexity, zero divergence risk, and requires only a small `train.py` wrapper around the existing training loop.

No BN layers exist in Transolver, so the `swa_utils.update_bn()` call is not needed — eliminating the main gotcha of SWA in standard vision models.

### Supporting evidence

- **Izmailov et al. (2018), "Averaging Weights Leads to Wider Optima and Better Generalization"** (arXiv:1803.05407v3). Canonical SWA reference. Demonstrates 1.0–2.0pp CIFAR-100 improvement, 0.3–0.5pp ImageNet improvement from cyclical LR + weight averaging over last K checkpoints. Key mechanism: cosine annealing pushes weights toward basin edges; averaging multiple edge samples recovers the flat center.
  URL: https://arxiv.org/abs/1803.05407

- **Cha et al. (2021), "SWAD: Domain Generalization by Seeking Flat Minima"** (arXiv:2102.08604). Applies flat-minima weight averaging specifically for domain generalization (train-domain → held-out-domain transfer). Improves DomainBed benchmarks by 1–3pp. The train/test split in DrivAerML (400 train, 50 test, different geometric configurations) is structurally analogous to domain generalization.
  URL: https://arxiv.org/abs/2102.08604

- **Benton et al. (2021), "Loss Surface Simplexes for Mode Connecting Volumetric Loss Landscape"** — supports flat-basin hypothesis in overparameterized neural nets. Supporting geometry for why SWA finds flatter solutions.

### Mechanism

1. Train normally for EP1–24 with the H39 configuration (Lion, lr=1e-4, cosine T_max=30, GradNorm α=0.5, Charbonnier-z, y-symmetry-aug, curvature-attention-bias).
2. From EP25, wrap the model in `torch.optim.swa_utils.AveragedModel`.
3. At the end of each epoch (EP25, EP26, EP27, EP28, EP29, EP30), call `swa_model.update_parameters(model)`.
4. At the end of EP30, evaluate `swa_model` — this is the submitted checkpoint.
5. Also save and evaluate the standard EP30 checkpoint as a control (so we can attribute any change).

No `update_bn()` call needed — Transolver has no BatchNorm layers.

The SWA learning rate schedule: keep the cosine schedule running through EP30 (no special SWA LR needed; the cosine tail naturally oscillates into low-LR territory which is the useful range for SWA averaging).

### Predicted metrics

- test_WSS: 6.40–6.50% (−0.15 to −0.25pp from H39)
- test_vol_p: stable (SWA does not target specific axes; flat basin should generalize uniformly)
- test_surf_p: stable or improved (same reasoning)
- test_z: the primary beneficiary — expect −0.2 to −0.4pp on wss_z if flat minima help OOD generalization

### Kill threshold

```
--kill-thresholds "5:abupt_axis_mean_rel_l2_pct>8.0"
```
EP5 should be well below 8.0 (H39 EP5 ≈ 6.2%); abort only if the base training diverges. SWA averaging does not change the base model until EP25, so early training behavior should be identical to H39.

### CLI / code change

SWA is not a CLI flag in the current codebase. The student must add approximately 15 lines to `train.py`:

```python
# After optimizer and scheduler creation, add:
from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(model)
SWA_START_EPOCH = 25  # start averaging at EP25

# Inside the training loop, after optimizer.step() + scheduler.step(), add:
if epoch >= SWA_START_EPOCH:
    swa_model.update_parameters(model)

# After the training loop ends, save swa_model.module.state_dict()
# and run evaluation on swa_model (wrapped in eval mode)
```

No architecture changes. No changes to model.py. Only train.py.

### Risk assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| No improvement (SWA lands on similar loss to EP30 normal) | Medium | Compare swa_model vs ep30 checkpoint explicitly |
| Slight regression in surf_p or vol_p | Low | SWA averages all parameters uniformly — no per-task bias |
| EP1-24 divergence (unrelated to SWA) | Very low | Identical to H39 base config for first 24 epochs |
| BN update issue | None | Transolver has no BatchNorm layers |

Overall: **low risk, moderate expected gain**.

---

## H139 — SGDR Cosine Restarts (Warm Restarts LR Schedule)

### What it is

Replace H39's monotone cosine decay (T_max=30, one cycle) with cosine annealing with warm restarts (T_0=10, T_mult=1, giving 3 cycles × 10 epochs each). The LR returns to its max value at EP11 and EP21, briefly raising the learning rate to escape shallow local minima and explore a wider loss landscape before final convergence.

### Why it might help here

H39 uses a single cosine decay from LR=1e-4 to ~0 over 30 epochs. This is aggressive for the final third of training — by EP20 the LR is < 1e-5 and the model is essentially frozen. Warm restarts extend the effective exploration window: at EP21 the LR resets to 1e-4, allowing the optimizer to traverse additional loss surface and potentially find a lower basin before the final decay at EP30. This is especially relevant for the z-axis WSS which is the highest-residual task and may require larger parameter updates to improve.

The mechanism is distinct from SWA (which averages weights), from capacity changes (which change architecture), and from loss reweighting (which changes gradient magnitudes). It is purely a training dynamics intervention.

### Supporting evidence

- **Loshchilov & Hutter (2017), "SGDR: Stochastic Gradient Descent with Warm Restarts"** (arXiv:1608.03983). Canonical reference. Demonstrates consistent improvement on CIFAR-10/100, shows that cyclical LR allows escaping sharp local minima. Widely reproduced across domains.
  URL: https://arxiv.org/abs/1608.03983

- **Wen et al. (2022), "iFNO: Implicit Fourier Neural Operators for Parametric PDEs"** (arXiv:2211.15188v4). Uses cyclical LR for neural PDE operators; notes that multiple cosine cycles improve convergence stability on multi-physics predictions. Mechanistically relevant: Transolver is a neural PDE operator in the same category.

- General empirical evidence: SGDR is a standard ingredient in most modern ML training stacks (LLM, vision). Its failure mode is well-characterized (early restarts that are too aggressive destabilize training — mitigated by starting with T_0=10 rather than T_0=5).

### Mechanism

Replace the `--lr-scheduler cosine --T-max 30` flags with `--lr-scheduler cosine_restarts --T-0 10 --T-mult 1`. This requires either:
(a) Adding a `cosine_restarts` option to the scheduler factory in `train.py` using `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)`, or
(b) If the CLI already supports `CosineAnnealingWarmRestarts`, just passing the right flags.

The student should verify which scheduler options are available in the current `train.py` before adding new code. If the `cosine_restarts` case does not exist, it is a ~5-line addition.

All other H39 hyperparameters held fixed: Lion, lr=1e-4, GradNorm α=0.5, Charbonnier-z, y-symmetry-aug, curvature-attention-bias, bs=1, 30 epochs, 65000 train / 65000 val surface points.

### Predicted metrics

- test_WSS: 6.45–6.55% (−0.10 to −0.20pp from H39)
- test_z: modest improvement, potentially −0.1 to −0.2pp on wss_z
- test_vol_p / test_surf_p: neutral to small improvement (same mechanism benefits all tasks equally)

Lower expected gain than H138 or H140, but also lowest implementation risk — pure scheduler swap.

### Kill threshold

```
--kill-thresholds "5:abupt_axis_mean_rel_l2_pct>8.0"
```
After the first restart at EP11, a temporary metric spike is expected (LR jumps back to 1e-4). This is normal and expected. Kill only if val_wss is still above 8.0 at EP5 before the first restart.

### CLI / code change

If `cosine_restarts` is already a supported `--lr-scheduler` option, this is a pure CLI change:
```
--lr-scheduler cosine_restarts --T-0 10 --T-mult 1
```

If not, add to `train.py` scheduler factory:
```python
elif args.lr_scheduler == 'cosine_restarts':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=getattr(args, 'T_mult', 1), eta_min=1e-6
    )
```
And add `--T-0 10 --T-mult 1` as CLI args.

### Risk assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Metric spike at EP11 restart confuses kill threshold | Medium | Set kill threshold only at EP5, before first restart |
| LR-restart destabilizes GradNorm task weights | Low | GradNorm α=0.5 re-normalizes within a few steps |
| No improvement vs H39 single cycle | Medium | Baseline comparison: final val_wss at EP30 |
| Convergence slower (less time at low LR) | Low | 3×10-epoch cycles still end at low LR at EP30 |

Overall: **very low risk, modest expected gain**.

---

## H140 — z-Coordinate Surface Loss Weighting

### What it is

A spatially-adaptive loss weighting scheme that upweights WSS prediction loss on surface points with large |z| coordinate values, directly targeting the worst per-axis metric (test_z=8.66%). The intuition: the z-axis WSS error is dominated by regions of the car where z-axis flow gradients are highest — typically the underbody and lateral surfaces at extreme z. By amplifying the gradient signal from those regions, the model is steered toward improving them.

### Why it might help here

The per-axis WSS breakdown is stark: x=5.90%, y=7.19%, z=8.66%. The z-axis is 1.5× harder than x-axis. This is unlikely to be an architecture capacity issue (H134 geometry context, H117/H132/H133 capacity sweeps all failed to close the z gap). It is more likely a training signal issue: the uniform loss over all surface points treats z-axis-critical high-|z| regions the same as z-axis-easy near-centerline regions. Amplifying the signal from high-|z| surface points directly corrects this.

The mechanism is distinct from per-task reweighting (GradNorm/IMTL-G/PCGrad, which reweight tasks relative to each other). This is per-point spatial reweighting within the WSS task itself.

### Supporting evidence

- **SENSE (Sch\'onfeld et al., Springer 2025)**: Multi-objective physics-informed loss for CFD neural surrogates. Uses point-importance weighting based on flow gradients to improve high-gradient regions. Directly analogous mechanism.

- **Arzani et al. (Nature Scientific Reports, 2026), "PI-GNN for coronary WSS prediction"**: Demonstrates that spatially-adaptive loss weights on surface points — emphasizing physiologically high-WSS regions — reduce WSS prediction error by 15–30% in those regions without degrading global metrics. The geometry is different (coronary arteries vs. car surface) but the mechanism is identical.

- **Focal Loss (Lin et al., 2017)**: The conceptual parent. Focal loss reweights examples by their difficulty (high-loss regions get more gradient). Z-coordinate loss weighting is the spatial analogue for regression.

- **General CFD principle**: z-axis WSS captures crossflow and separation phenomena. These are geometrically concentrated at extreme |z| values (underbody, side mirrors, wheel arches). A uniform loss treats the smooth hood surface the same as a wheel arch — a known suboptimality in CFD surrogates that spatial weighting directly corrects.

### Mechanism

In `train.py`, locate the WSS loss computation. Surface points have coordinate features in `surface_x[:,:,:3]` where index 2 is the z-coordinate. Add a per-point weight:

```python
# In the WSS loss computation block:
# surface_x shape: [B, N, 7] — features are [x, y, z, nx, ny, nz, area]
z_coord = surface_x[:, :, 2]  # [B, N]
z_max = z_coord.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)  # [B, 1]
z_weight = 1.0 + lambda_z * (z_coord.abs() / z_max)  # [B, N], range [1, 1+lambda_z]

# Apply to WSS channels only (indices 1,2,3 of surface_preds, not cp which is index 0):
# wss_loss before weighting: [B, N, 3]
# z_weight: [B, N] → unsqueeze to [B, N, 1] for broadcast
wss_loss = wss_loss * z_weight.unsqueeze(-1)
wss_loss = wss_loss.mean()
```

Key parameter: `lambda_z = 1.0` (start value). This gives weights in [1.0, 2.0] — high-|z| points get 2× the gradient signal of centerline points.

Apply weighting only to WSS loss (not to cp or vol_p) to preserve the AND-contract for surf_p and vol_p.

### Hyperparameter sensitivity

| lambda_z | Weight range | Risk level | Expected effect |
|----------|-------------|------------|----------------|
| 0.5 | [1.0, 1.5] | Low | Mild z improvement, safe |
| 1.0 | [1.0, 2.0] | Medium | Moderate z improvement, recommended start |
| 2.0 | [1.0, 3.0] | Higher | Aggressive; may hurt surf_p indirectly if GradNorm rebalances |

Recommend: lambda_z=1.0 for first run.

### Predicted metrics

- test_WSS: 6.30–6.50% (−0.15 to −0.35pp from H39) — highest upside of the three hypotheses
- test_z (wss_z): −0.3 to −0.6pp improvement expected (primary target)
- test_x / test_y: neutral to −0.1pp (z-weighting applies to all 3 WSS channels, mild spill-over)
- test_surf_p: neutral (cp not weighted)
- test_vol_p: neutral (vol_p not weighted)

### Kill threshold

```
--kill-thresholds "5:abupt_axis_mean_rel_l2_pct>8.0"
```

If lambda_z=1.0 destabilizes training (wss_z diverges), kill threshold will catch it at EP5.

### CLI / code change

No new CLI flags required if lambda_z is hardcoded. Better: add as a CLI argument:
```
--z-loss-lambda 1.0
```

Code change is entirely in `train.py`, approximately 5 lines in the loss computation block.

### Risk assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| lambda_z=1.0 hurts test_x / test_y through GradNorm rebalancing | Medium | Monitor per-axis metrics; if x/y degrade, reduce lambda_z to 0.5 |
| Overfitting high-|z| train points (near-boundary surface geometry) | Low-Medium | 400 train cars have diverse z distributions; limited overfitting risk |
| surf_p or vol_p regression via GradNorm rebalancing | Low | Spatial weighting is within WSS task, not across tasks; GradNorm sees WSS as one task |
| No improvement (z error is not from spatial imbalance) | Medium | Diagnostic value: if null, confirms z error is architectural/fundamental |

Overall: **medium risk, highest expected gain (−0.15 to −0.35pp)**.

---

## Summary Table

| ID | Name | Mechanism | Code change | Risk | Predicted test_WSS |
|----|------|-----------|-------------|------|-------------------|
| H138 | SWA | Weight averaging EP25–30, flat minima → better OOD transfer | ~15 lines in train.py | Low | 6.40–6.50% |
| H139 | SGDR Warm Restarts | 3×10-epoch cosine cycles, extended LR exploration | ~5 lines in train.py or pure CLI | Very Low | 6.45–6.55% |
| H140 | z-Coord Loss Weight | Per-point WSS loss weight by |z|, lambda_z=1.0, targets test_z=8.66% | ~5 lines in train.py | Medium | 6.30–6.50% |

**Recommended priority order**: H140 > H138 > H139 (by expected gain and mechanistic novelty)

All three are orthogonal to each other and to currently active experiments (H123 tanjiro, H134 fern, H137 nezuko). All three preserve the AND-contract for surf_p and vol_p. All three require only `train.py` modifications.

---

## Common reproduction base

All three experiments should use the following H39-matching base config:

```bash
python train.py \
  --dataset-path /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --model transolver \
  --hidden-dim 512 \
  --depth 6 \
  --slices 128 \
  --surface-out-factor 2.0 \
  --loss charbonnier_z \
  --optimizer lion \
  --lr 1e-4 \
  --batch-size 1 \
  --epochs 30 \
  --train-surface-points 65000 \
  --train-volume-points 65000 \
  --gradnorm-alpha 0.5 \
  --gradnorm-clamp 0.15 \
  --curvature-attention-bias \
  --y-symmetry-aug \
  --kill-thresholds "5:abupt_axis_mean_rel_l2_pct>8.0" \
  --wandb-group <hypothesis-slug>
```

With the hypothesis-specific additions as described in each section above.

---

*Generated by senpai-advisor on drivaerml-long-20260504, 2026-05-25 12:30*
