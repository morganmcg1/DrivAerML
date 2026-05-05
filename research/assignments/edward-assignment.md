## Hypothesis

**Issue #717 — Test-time augmentation (TTA) for single-model volume_pressure transfer**

The chronic ~3x volume_pressure test-vs-val gap (val~3.9%, test~11.5%) cannot be closed at val-time only — it is by definition a *test-distribution* problem. Issue #717 H1 explicitly says "every volume claim needs early test evaluation" and the K=7 ensemble (PR #612) reduces the gap because it averages out per-model noise on the held-out test cars. **TTA is the natural single-model analogue of ensembling**: run the *same* trained model multiple times on the same test sample with cheap, label-preserving input perturbations and average the predictions. No second model, no inference-time ensembling across distinct checkpoints — single model, multiple input views.

This experiment tests **TTA on the existing PR #592 SOTA single-model checkpoint AND a freshly trained single-model run**, with two cheap, physically valid augmentations:

1. **Y-mirror.** DrivAerML cars are nominally symmetric across the y=0 plane (driver-passenger). Mirror the input geometry (`y -> -y` on coords; flip sign of `wall_shear_y` predictions; do NOT flip sign of `volume_pressure` or `surface_pressure`; flip `wall_shear_y` axis prediction sign back at output). One mirrored forward pass per sample.
2. **Coordinate jitter.** Add small isotropic Gaussian noise `eps ~ N(0, sigma^2)` with `sigma = 0.005` (5mm at body scale, well below mesh resolution) to the input volume *and* surface coordinates. Run 4 jittered passes. Predictions averaged.

**Total TTA cost:** 1 (clean) + 1 (Y-mirror) + 4 (jitter) = **6 forward passes per test sample**. Eval-time only — training is unchanged.

**Why this might help:**
- The val_pressure model already nails val (3.9%) but fails test (11.5%) — the *function* the model has learned is locally too sharp w.r.t. input geometry. TTA averaging smooths sharp local deviations *only at test time*, which is exactly the failure mode.
- Y-mirroring is **free signal** the training pipeline doesn't currently exploit: the model sees each car only in its native orientation. At test, asking the model to predict on the mirrored car and averaging the two predictions is a Bayesian model-average over a learned-but-not-enforced symmetry.
- Coordinate jitter (sigma=5mm) is below the simulation mesh resolution; the *true* pressure field is locally smooth at this scale, so the average of jittered predictions should converge to the true mean. If the model is overfit to specific point realizations (sampling noise), jitter-TTA exposes and averages it out.
- This is **completely orthogonal** to all in-flight experiments (KD #729, outlier-sampling #728, coord-norm #723) — it operates at inference time on a fixed weights snapshot.

**Two arms (sequential, NOT concurrent — single 8-GPU pod):**

- **Arm A** — TTA on existing best single-model SOTA checkpoint (`4k25s25e`, PR #592, val_abupt=6.5985%). Pure inference-only, ~30 min wall clock for full test eval. Gates immediate signal: does TTA help any existing checkpoint?
- **Arm B** — Train a fresh single-model run with **train-time mirror augmentation** (random Y-flip with p=0.5 per sample at training; same sign-flip rule for wall_shear_y target) AND apply test-time TTA at inference. This trains the model to be more *consistent* under mirror, which sharpens the TTA averaging gain.

If Arm A shows weak/no signal, Arm B may still win because the trained-with-mirror model can learn the symmetry consistency loss implicitly.

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

**Arm A (inference-only, do this first — fast):**

1. Locate the `eval_pass` / `validation_step` code that runs `model(batch) -> preds`.
2. Wrap it with a TTA loop that runs 6 forward passes per batch and averages predictions.
3. Each pass: clone `batch`, optionally apply (a) Y-mirror (negate y-coordinate; remember to flip sign of `wall_shear_y` predictions back); (b) jitter (add `torch.randn_like(coords) * 0.005`).
4. The W&B logging key for TTA evals should be `test_primary/...` so the existing best-ckpt selection logic remains intact.
5. Reproduce eval with TTA on the `4k25s25e` checkpoint — the model artifact `model-alphonse-depth-L5-4k25s25e` is in W&B.

**Arm B (train-time mirror + test-time TTA):**

6. Add `--use-mirror-aug` flag to `train.py`. In the train collate / dataloader, with `p=0.5` per sample, flip y-coordinate of inputs and flip target sign of `wall_shear_y` GT. Surface and volume coords mirrored together.
7. Train using SOTA stack (command below). At eval, run the same 6-pass TTA from Arm A.
8. **Sequential arms** — run Arm A first (~30 min), then Arm B (~270 min training + ~30 min TTA eval). Total wall-clock ~5h.

## Training command (Arm B — Arm A is inference-only)

```bash
cd target/ && torchrun --standalone --nproc_per_node=8 train.py \
  --agent edward --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
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
  --wandb-group edward-tta-mirror-jitter \
  --wandb-name edward/tta-armB-trainmirror
```

(Add the `--use-mirror-aug` flag once implemented.)

## Gates

- **Arm A (inference-only):** no time/val gate. Just report TTA-on/TTA-off side-by-side on test.
- **Arm B (training):**
  - **EP1 time gate:** kill if epoch_time > 80 min (mirror aug is ~free at dataloader level; expect ~37-38 min/epoch).
  - **EP2 (step 21,729):** kill if val_abupt > 12%.
  - **EP3 (step 32,594):** kill if val_abupt > 8%.

## Required reporting (Issue #717 9-column table)

For **each arm**, post:

1. W&B run ID
2. TTA-off test metrics (the no-TTA single-model number on the same checkpoint)
3. TTA-on test metrics (Y-mirror only / jitter only / combined)
4. Best-val-abupt checkpoint
5. Final checkpoint
6. Per-case top-10 worst test volume
7. Did TTA help volume specifically vs other channels?
8. Required 9-col table:

| Run | Checkpoint | Aggregate | Surface p | Volume p | Wall shear | Tau x | Tau y | Tau z |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| #599 `sogus8sx` | reported best | | | 11.694 | 7.299 | | 7.941 | 9.535 |
| #592 `4k25s25e` | reported best | 7.9915 | 4.3322 | 11.933 | 7.334 | | 8.145 | 9.298 |
| #681 `dc031qpt` | reported best | | | 11.374 | 8.321 | | 9.596 | 10.738 |
| Arm A `4k25s25e` no-TTA | best aggregate | 7.9915 | 4.3322 | 11.933 | 7.334 | | 8.145 | 9.298 |
| Arm A `4k25s25e` with-TTA | best aggregate | | | | | | | |
| Arm B fresh-trained no-TTA | best aggregate | | | | | | | |
| Arm B fresh-trained with-TTA | best aggregate | | | | | | | |
| Arm B fresh-trained with-TTA | best volume | | | | | | | |

## Closure rules

- **Arm A solid win on test_volume (<=10.0%):** strong signal regardless of Arm B — mark review.
- **Arm A weak (no movement) but Arm B solid:** mark review on Arm B; the train-time mirror is the lever.
- **Both arms null:** post SENPAI-RESULT, advisor will close.
- **Crash/divergence:** report root cause.
