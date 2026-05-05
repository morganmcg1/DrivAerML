## Hypothesis

**Issue #717 — Inter-sample volume mixup as a single-model regularizer for test transfer**

The chronic ~3x volume_pressure test-vs-val gap (val~3.9%, test~11.5%) is a textbook *generalization* failure — the model fits the training cars' wake structure tightly but the per-test-car wake field is genuinely different. Mixup (Zhang et al. 2017, https://arxiv.org/abs/1710.09412) is the canonical regularizer for this exact setting: it forces the model to learn a *linear interpolation* of input-output behavior across training samples, which reduces the model's local Lipschitz constant in input-output space and is empirically the strongest single-knob regularizer for cross-distribution transfer in vision and tabular tasks.

This experiment tests **input-mixup applied only to volume points and volume_pressure targets** (not surface, which already transfers well at ~3.5% test gap) on the SOTA single-model stack.

**Mechanism (precise):**

For each minibatch of 4 cars, with probability `p_mix = 0.5`, sample a `lambda ~ Beta(alpha, alpha)` and a permutation `pi` of the batch. Then:

1. Mix volume coordinates: `vol_coords_mixed = lambda * vol_coords + (1 - lambda) * vol_coords[pi]`
2. Mix volume targets: `vol_pressure_mixed = lambda * vol_pressure + (1 - lambda) * vol_pressure[pi]`
3. **Surface inputs and targets unchanged** — surface is a boundary task and mixing surface coordinates of two different cars yields a non-physical car silhouette. Volume points are scattered through 3D wake space and mixing them yields a still-meaningful (interpolated) volume query position.
4. Loss: standard volume MSE on the mixed predictions.

**Two arms (sequential):**

- **Arm A — alpha=0.2** (gentle mixup; mode of Beta(0.2,0.2) is at 0/1, so lambda is usually near 0 or 1 — small perturbation regime). This is the conservative starting point. Standard imagenet mixup recipe.
- **Arm B — alpha=0.4** (stronger mixup; closer to uniform in lambda). Larger regularization.

**Why this might help:**
- Test cars have fundamentally different wake geometry. Mixup teaches the model "the volume pressure at an interpolated wake position is the interpolation of the two end pressures" — this is a learned smoothness prior that *should* transfer because it constrains the local function shape, not the specific car shape.
- Surface-stays-clean is critical: PR #695 (RFF=32 doubling on surface and volume) regressed because doubling spectral capacity hurt the strong surface task. Mixup on volume only sidesteps this.
- This is **completely orthogonal** to in-flight outlier-sampling (PR #728 frieren — emphasizes hard cases more) and KD (PR #729 alphonse — uses ensemble soft targets). Mixup mixes *input-output pairs*, KD doesn't change inputs, outlier-sampling reweights but doesn't mix.

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

1. **In the training loop / collate fn**, after the standard batch is constructed:
   ```python
   if torch.rand(1).item() < self.mixup_p:
       lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(device)
       pi = torch.randperm(batch_size, device=device)
       batch.volume_coords = lam * batch.volume_coords + (1-lam) * batch.volume_coords[pi]
       batch.volume_pressure = lam * batch.volume_pressure + (1-lam) * batch.volume_pressure[pi]
       # surface untouched
   ```
2. **Critical: only do mixup at training time.** All val/test eval must use clean (un-mixed) inputs — the mixup is a regularizer, not a target distribution change.
3. Add CLI flags: `--mixup-alpha` (float, default 0.0 = off), `--mixup-prob` (float, default 0.5).
4. Ensure mixed batch sizes match for `pi` indexing — use the same batch dim layout as the existing pipeline.
5. **Two arms run sequentially** on the SAME 8 GPUs — never two `torchrun --nproc_per_node=8` jobs concurrently (PR #716 lesson). Use the chain script pattern: run Arm A, on completion launch Arm B.

## Training command (Arm A: alpha=0.2)

```bash
cd target/ && torchrun --standalone --nproc_per_node=8 train.py \
  --agent fern --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
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
  --wandb-group fern-vol-mixup \
  --wandb-name fern/mixup-alpha0.2
```

(Add `--mixup-alpha 0.2 --mixup-prob 0.5` for Arm A; `--mixup-alpha 0.4` for Arm B.)

## Gates

- **EP1 time gate:** kill if epoch_time > 80 min (mixup is free at collate level; expect ~37-38 min/epoch).
- **EP2 (step 21,729):** kill if val_abupt > 12%.
- **EP3 (step 32,594):** kill if val_abupt > 8%.

## Required reporting (Issue #717 9-column table)

For **each arm**, post:

1. W&B run ID
2. Best-val-abupt checkpoint metrics
3. Best-val-volume_pressure checkpoint metrics
4. Final checkpoint metrics
5. Per-case top-10 worst test volume
6. Did mixup specifically help volume vs other channels?
7. Did the val→test transfer ratio shrink?
8. Required 9-col table (per arm × per checkpoint):

| Run | Checkpoint | Aggregate | Surface p | Volume p | Wall shear | Tau x | Tau y | Tau z |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| #599 `sogus8sx` | reported best | | | 11.694 | 7.299 | | 7.941 | 9.535 |
| #592 `4k25s25e` | reported best | 7.9915 | 4.3322 | 11.933 | 7.334 | | 8.145 | 9.298 |
| #681 `dc031qpt` | reported best | | | 11.374 | 8.321 | | 9.596 | 10.738 |
| Arm A alpha=0.2 | best aggregate | | | | | | | |
| Arm A alpha=0.2 | best volume | | | | | | | |
| Arm A alpha=0.2 | final | | | | | | | |
| Arm B alpha=0.4 | best aggregate | | | | | | | |
| Arm B alpha=0.4 | best volume | | | | | | | |
| Arm B alpha=0.4 | final | | | | | | | |

## Closure rules

- **Solid/major win on test_volume (<=10.0%):** mark review.
- **Val beats SOTA (<6.5985%):** mark review.
- **Both arms null and val regresses:** post SENPAI-RESULT, advisor will close.
- **Surface or wall_shear regresses by >0.5pp:** mixup is hurting the working channels — report and stop.
