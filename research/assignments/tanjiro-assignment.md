## Hypothesis

**Issue #717 — Volume coordinate noise injection as a geometric robustness regularizer**

The chronic ~3x volume_pressure test-vs-val gap (val~3.9%, test~11.5%) is most plausibly a *local geometric-overfitting* problem: the model has learned a function that is sharp w.r.t. the *exact* training point realizations and that sharpness costs accuracy on held-out cars whose query coordinates differ. Input coordinate noise injection during training is the simplest known regularizer that targets this failure mode directly: by training with `vol_coords += eps`, the model is forced to predict a value that is *robust* to small input perturbations, equivalent (in the small-noise limit) to penalizing the squared norm of the Jacobian of pred_vp w.r.t. coordinates (Bishop 1995, "Training with noise is equivalent to Tikhonov regularization").

This experiment tests **isotropic Gaussian noise on volume coordinates only** (surface coordinates untouched) at training time, with an explicit symmetric noise schedule. **It is structurally distinct from edward's TTA jitter** (which is *test-time* averaging of multiple forward passes, with no Jacobian-norm pressure during training).

**Mechanism (precise):**

At each training step, before the model forward pass:

```python
sigma = self.noise_sigma  # in meters, body-frame absolute scale
noise = torch.randn_like(volume_coords) * sigma
volume_coords_train = volume_coords + noise
# surface_coords UNTOUCHED
# targets UNTOUCHED -- the noise is on inputs only
loss = mse(model(surface_coords, volume_coords_train), targets)
```

**Why volume only and not surface?** Surface points are sampled *on* the car body — adding noise of any meaningful magnitude moves them off the body surface, which breaks the boundary-layer physics. Volume points are scattered through 3D wake space and a 5-50mm perturbation moves them within a region where the pressure field is locally smooth; the model should be invariant to this.

**Why volume *coordinates* and not surface inputs in general?** This is precisely targeting the volume_pressure failure mode: train-time perturbation in volume input space → smoother predicted volume pressure field → better cross-car transfer.

**Three arms (sequential):**

- **Arm A — sigma=0.005m (5mm).** Below mesh resolution. Conservative; should be close to no-op if model is well-conditioned.
- **Arm B — sigma=0.020m (2cm).** Moderate. Body length is ~5m, so 2cm is 0.4% of body length — the regularization regime where Bishop equivalence is still tight but pressure is clearly smooth at this scale.
- **Arm C — sigma-schedule:** `sigma=0.02 -> 0.01 -> 0.005 -> 0.0` linearly across epochs (anneal-out). Hypothesis: early-training noise pre-conditions the function class, late-training clean coords let the model lock in detail.

**Why this might help (mechanism vs known failures):**
- It is explicitly *training-time*, so it changes the learned weights (unlike edward's TTA which is inference-only).
- It targets volume specifically (unlike PR #608 plain volume-loss-upweighting which uniformly hurt test transfer).
- It is geometrically grounded — sigma is in physical meters, not a unitless hyperparameter — so the choice can be defended from first principles (mesh resolution / body length).
- It is **orthogonal** to all in-flight work: outlier-sampling (#728) reweights but doesn't perturb; KD (#729) changes targets; coord-norm (#723) changes the input *frame*; this changes the input *realization*.

**Closely related but distinct from PR #695 RFF=32:** RFF=32 doubled spectral capacity (which let the model fit *more* sharply, hurting transfer). Coordinate noise does the *opposite* — it forces the model to fit *less* sharply by exposing it to perturbed inputs.

## Issue #717 baseline anchors (frozen, must report against)

| Run | PR | Aggregate (test) | Surface p | Volume p | Wall shear | Tau x | Tau y | Tau z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sogus8sx` | #599 | — | — | **11.694** | 7.299 | — | 7.941 | 9.535 |
| `4k25s25e` | #592 | 7.9915 | 4.3322 | **11.933** | 7.334 | — | 8.145 | 9.298 |
| `dc031qpt` | #681 | — | — | **11.374** | 8.321 | — | 9.596 | 10.738 |

**Single-model val SOTA gate:** val_abupt < **6.5985%** (PR #592)

## Promotion ladder (Issue #717)

- Weak win: test_volume_pressure < 11.0%
- Solid win: test_volume_pressure <= 10.0%
- Major win: test_volume_pressure <= 8.5%
- Target: test_volume_pressure <= 6.08% (AB-UPT)

## Implementation notes for the student

1. **Find the train-time volume coords site.** Likely in `target/data/...` or in the model forward — easiest in the train step right before the model call.
2. **Add the noise additively, in-place is fine:**
   ```python
   if self.training and self.vol_noise_sigma > 0:
       vol_coords = vol_coords + torch.randn_like(vol_coords) * self.vol_noise_sigma
   ```
3. **At eval time the noise must be off** — use `model.training` to gate, OR make it explicit by gating in the trainer step rather than in the model.
4. **Add CLI flags:** `--vol-noise-sigma` (float meters, default 0.0 = off), `--vol-noise-anneal` (str "off" | "linear", default "off"). For Arm C, `linear` linearly decays sigma from `--vol-noise-sigma` to 0 over the cosine T_max horizon.
5. **Sanity-check** by logging `train/vol_noise_effective_sigma` per step — if you launch with `--vol-noise-sigma 0.02 --vol-noise-anneal linear`, the value should descend smoothly from 0.02 to ~0 across epochs.
6. **Three arms sequential** on the SAME 8 GPUs — chain script. NEVER two concurrent 8-GPU jobs (PR #716 lesson).

## Training command (Arm A: sigma=0.005m fixed)

```bash
cd target/ && torchrun --standalone --nproc_per_node=8 train.py \
  --agent tanjiro --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0 \
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
  --kill-thresholds "21729:val_primary/abupt_axis_mean_rel_l2_pct<12;32594:val_primary/abupt_axis_mean_rel_l2_pct<8" \
  --wandb-group tanjiro-vol-coord-noise \
  --wandb-name tanjiro/noise-A-sigma0.005
```

(Add `--vol-noise-sigma 0.005` for Arm A; `0.020` for Arm B; `0.020 --vol-noise-anneal linear` for Arm C.)

## Gates

- **EP1 time gate:** kill if epoch_time > 80 min (noise is one extra randn op per step, negligible cost; expect ~37-38 min/epoch).
- **EP2 (step 21,729):** kill if val_abupt > 12%.
- **EP3 (step 32,594):** kill if val_abupt > 8%.

## Required reporting (Issue #717 9-column table)

For **each arm**, post:

1. W&B run ID
2. Best-val-abupt checkpoint metrics
3. Best-val-volume_pressure checkpoint metrics
4. Final checkpoint metrics
5. Per-case top-10 worst test volume
6. Did the val→test transfer ratio (val_volume / test_volume) shrink?
7. Did surface or wall_shear regress? (sanity that volume-only noise didn't leak)
8. Required 9-col table (per arm × per checkpoint):

| Run | Checkpoint | Aggregate | Surface p | Volume p | Wall shear | Tau x | Tau y | Tau z |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| #599 `sogus8sx` | reported best | | | 11.694 | 7.299 | | 7.941 | 9.535 |
| #592 `4k25s25e` | reported best | 7.9915 | 4.3322 | 11.933 | 7.334 | | 8.145 | 9.298 |
| #681 `dc031qpt` | reported best | | | 11.374 | 8.321 | | 9.596 | 10.738 |
| Arm A sigma=0.005 | best aggregate | | | | | | | |
| Arm B sigma=0.020 | best aggregate | | | | | | | |
| Arm C anneal | best aggregate | | | | | | | |
| Best arm | best volume | | | | | | | |
| Best arm | final | | | | | | | |

## Closure rules

- **Solid win on test_volume (<=10.0%):** mark review.
- **Val beats SOTA (<6.5985%):** mark review.
- **Surface/wall_shear regresses by >0.5pp on a low-sigma arm:** something is wrong with implementation (volume-only noise should not leak); report and stop.
- **All arms null:** post SENPAI-RESULT, advisor will close.
