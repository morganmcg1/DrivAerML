## Hypothesis

**Issue #717 — Region-weighted volume loss: "where the test error lives" hypothesis**

The chronic ~3x volume_pressure test-vs-val gap (val~3.9%, test~11.5%) is geometrically structured: the volume pressure field has very different statistics in the **near-wake** (high-magnitude pressure recovery, complex vortex shedding) vs the **far-wake / open-flow** regions (low-magnitude, smoother). Issue #717 explicitly requests Phase 0 diagnostics on "Volume error by region: near wake, far wake, roof/underbody/side". The current uniform-MSE loss treats all 65,536 volume points equally — a far-wake point at 5m downstream contributes the same gradient as a critical wake-vortex point 0.3m behind the rear bumper.

This experiment **explicitly upweights the volume loss in the high-uncertainty region of the wake**, defined geometrically (no per-point residual EMA — that is PR #728 frieren's outlier-sampling territory). The geometric region weights are a *static, sample-invariant* function of `(x, y, z)` — orthogonal to PR #728's *dynamic* per-point residual reweighting and to PR #729's KD soft targets.

**Mechanism (precise):**

For each volume query point `v_i = (x_i, y_i, z_i)` in normalized car-relative coordinates (centroid-subtracted, body-length-normalized — let `s_ref` be the body-x extent):

1. Compute `x_rel = (x_i - centroid_x) / s_ref` and `z_rel = (z_i - centroid_z) / s_ref`.
2. Define **near-wake mask** = `1.0 < x_rel < 3.0 AND -1.5 < z_rel < 1.5` (1 to 3 body-lengths behind, within +/- 1.5 body-lengths vertically).
3. Per-point weight: `w_i = w_near` if in near-wake mask else `w_far`.
4. Weighted MSE: `loss_volume = sum_i (w_i * (pred_i - target_i)^2) / sum_i w_i`.

**Three arms (sequential, NOT concurrent):**

- **Arm A — `w_near=1.5, w_far=1.0`** (mild near-wake emphasis, baseline). Tests whether near-wake matters.
- **Arm B — `w_near=2.0, w_far=1.0`** (moderate). The "primary signal" arm.
- **Arm C — `w_near=2.0, w_far=0.7`** (combined: near up + far down). Tests whether de-emphasizing easy far-wake also helps.

**Why this might help:**
- Per-region diagnostics from Issue #717 1B-style outlier studies suggest the dominant test error is in the near-wake/recirculation zone, not the far field. Upweighting that region forces the model to spend more capacity where the error lives.
- This is a **static loss reweighting** at training time only. At inference, no change. Surface unchanged — only the volume head's per-point loss is reweighted.
- Strictly orthogonal to in-flight: outlier-sampling (#728) emphasizes by per-point *residual EMA* (data-dependent), this emphasizes by *position* (data-independent). KD (#729) changes the *target*, this changes the *weight*. Coord-norm (#723) changes the *input encoding*, this changes the *loss aggregation*.

**Critical: this is NOT plain volume-loss upweighting** (which Issue #717 explicitly bans, PR #608 went 14.005% test). The total volume loss magnitude is held approximately constant by the normalizing `sum_i w_i` denominator — only the *spatial allocation* of gradient changes.

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

1. **Find the volume MSE loss site** in `target/`. It is currently a clean `F.mse_loss(pred_vp, target_vp)`.
2. **Compute centroid + body-length per sample** (you can use the surface coords' bounding box: `s_ref = surface_coords[..., 0].max() - surface_coords[..., 0].min()`).
3. **Mask construction** is purely on volume coordinates; vectorized with broadcasting:
   ```python
   x_rel = (vol_coords[..., 0] - cx) / s_ref
   z_rel = (vol_coords[..., 2] - cz) / s_ref
   near_mask = ((x_rel > 1.0) & (x_rel < 3.0) & (z_rel > -1.5) & (z_rel < 1.5)).float()
   weight = near_mask * (w_near - w_far) + w_far  # broadcast (B, V)
   loss_vp = (weight * (pred - target).pow(2)).sum() / weight.sum().clamp_min(1e-6)
   ```
4. **Add CLI flags:** `--vol-loss-w-near` (default 1.0 = off), `--vol-loss-w-far` (default 1.0 = off).
5. **Three arms sequential** on the same 8 GPUs — chain script. NEVER two concurrent 8-GPU jobs (PR #716 lesson, doubled epoch time to 180 min).
6. Sanity-check the mask coverage logging: log `mean(near_mask)` per step — should be roughly 5-15% of points (depends on volume sampler).

## Training command (Arm A: w_near=1.5, w_far=1.0)

```bash
cd target/ && torchrun --standalone --nproc_per_node=8 train.py \
  --agent nezuko --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
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
  --wandb-group nezuko-region-vp-loss \
  --wandb-name nezuko/region-A-near1.5-far1.0
```

(Add `--vol-loss-w-near 1.5 --vol-loss-w-far 1.0` for Arm A; `2.0/1.0` for Arm B; `2.0/0.7` for Arm C.)

## Gates

- **EP1 time gate:** kill if epoch_time > 80 min (mask compute is ~free; expect ~37-38 min/epoch).
- **EP2 (step 21,729):** kill if val_abupt > 12%.
- **EP3 (step 32,594):** kill if val_abupt > 8%.
- **Per-arm gate:** if Arm A has identical val curve to SOTA (within 0.05pp), skip Arm B (signal too weak); jump straight to Arm C combined emphasis.

## Required reporting (Issue #717 9-column table)

For **each arm**, post:

1. W&B run ID
2. Best-val-abupt checkpoint metrics
3. Best-val-volume_pressure checkpoint metrics
4. Final checkpoint metrics
5. **Per-region test volume error breakdown** (the diagnostic Issue #717 explicitly requests):
   - near-wake band (1.0 < x_rel < 3.0, |z_rel| < 1.5): rel_l2 + point count
   - far-wake band (x_rel >= 3.0): rel_l2 + point count
   - upstream/cabin band (x_rel <= 1.0): rel_l2 + point count
6. Per-case top-10 worst test volume
7. Did near-wake error specifically drop? Did upweighting transfer?
8. Required 9-col table (per arm × per checkpoint):

| Run | Checkpoint | Aggregate | Surface p | Volume p | Wall shear | Tau x | Tau y | Tau z |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| #599 `sogus8sx` | reported best | | | 11.694 | 7.299 | | 7.941 | 9.535 |
| #592 `4k25s25e` | reported best | 7.9915 | 4.3322 | 11.933 | 7.334 | | 8.145 | 9.298 |
| #681 `dc031qpt` | reported best | | | 11.374 | 8.321 | | 9.596 | 10.738 |
| Arm A 1.5/1.0 | best aggregate | | | | | | | |
| Arm B 2.0/1.0 | best aggregate | | | | | | | |
| Arm C 2.0/0.7 | best aggregate | | | | | | | |
| Best arm | best volume | | | | | | | |
| Best arm | final | | | | | | | |

## Closure rules

- **Any arm: solid win on test_volume (<=10.0%):** mark review.
- **Any arm: val beats SOTA (<6.5985%):** mark review.
- **Near-wake band rel_l2 drops but overall aggregate volume_p does not:** report — mechanism partial-success, advisor will combine with a follow-up.
- **All arms null:** post SENPAI-RESULT, advisor will close.
