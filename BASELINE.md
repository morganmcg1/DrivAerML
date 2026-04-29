# DrivAerML Baseline — `tay`

**Branch:** `tay` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Status: PR #33 — 2026-04-29 16:39 UTC

New tay SOTA set by fern's RFF coordinate encoding: Gaussian random Fourier
features (sigma=1.0, 32 features per modality) appended to surface and volume
coord inputs. Lifts every surface/wall-shear axis by 10-15%, volume pressure
flat (expected — far-field coords saturate sigma=1.0 RFF). Run under
`--no-compile-model` (torch.compile bug open), 9 compiled-equiv epochs.

**W&B run:** `u43lik5d` (rank 0) — group `fern-rff-sigma-sweep`
**Best-val checkpoint:** epoch 9 (val_abupt=17.06)

### tay current best — `test_primary/*`

| Metric | This-repo key | tay best (PR #33) | PR #30 | yi best | AB-UPT |
|---|---|---:|---:|---:|---:|
| `abupt` | `test_primary/abupt_axis_mean_rel_l2_pct` | **17.77** | 19.81 | 15.82 | — |
| `surface_pressure` | `test_primary/surface_pressure_rel_l2_pct` | **11.20** | 12.86 | 9.99 | 3.82 |
| `wall_shear` | `test_primary/wall_shear_rel_l2_pct` | **18.68** | 21.27 | 16.60 | 7.29 |
| `volume_pressure` | `test_primary/volume_pressure_rel_l2_pct` | **16.13** | 15.91 | 14.21 | 6.08 |
| `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **16.20** | 18.24 | 14.27 | 5.35 |
| `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **21.81** | 25.50 | 19.49 | 3.65 |
| `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **23.54** | 26.53 | 21.12 | 3.63 |

### Reproduce PR #33 config

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --volume-loss-weight 2.0 --batch-size 4 --validation-every 1 \
  --lr 5e-5 --weight-decay 5e-4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.9995 \
  --rff-num-features 32 --rff-sigma 1.0 \
  --gradient-log-every 100 --weight-log-every 100 \
  --no-log-gradient-histograms --no-compile-model
```

### Compounding wins so far

| PR | Who | Delta | Lever |
|---|---|---:|---|
| #30 | alphonse | baseline (19.81) | yi calibration config (4L/512d/8h, vol_w=2.0) |
| #33 | fern | **−2.04 (−10.3%)** | RFF coord features (sigma=1.0, 32 feats) |

### INFRA NOTE — torch.compile bug (high priority fix)

`compile_model=True` default in `train.py` + `drop_last=False` in
`trainer_runtime.py:293` → last partial batch of each epoch kills all ranks
via `torch._inductor.exc.InductorError` at the epoch-boundary step.

**Fix (either in train.py or trainer_runtime.py — both editable per program.md):**
- Option A (train.py): pass `dynamic=True` to `torch.compile(model, ...)`
- Option B (trainer_runtime.py): set `drop_last=True` on the train DataLoader

Without the fix, `--no-compile-model` is required, costing ~50% of training
throughput and limiting tay runs to ~9 epochs per budget slot.

The yi advisor's best results (different W&B project) are informational targets
to match or beat on tay/DDP8:

## Reference baseline targets — must beat (AB-UPT public reference)

| Target | This-repo metric | AB-UPT |
|---|---|---:|
| Surface pressure `p_s` | `test_primary/surface_pressure_rel_l2_pct` | **3.82** |
| Vector wall shear `tau` | `test_primary/wall_shear_rel_l2_pct` | **7.29** |
| Volume pressure `p_v` | `test_primary/volume_pressure_rel_l2_pct` | **6.08** |
| Wall shear `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **5.35** |
| Wall shear `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **3.65** |
| Wall shear `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **3.63** |

Lower is better. Final claims must come from `test_primary/*` after best-validation
checkpoint reload.

## Yi reference results (different W&B project — for context only)

The prior `yi` advisor reached the following on their project. These are
informational targets to match or beat on tay/DDP8:

| Metric | Best on yi | PR | W&B run | Date |
|---|---:|---|---|---|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **15.82** | yi#13 | wio9pqw2 | 2026-04-29 |
| `test_primary/surface_pressure_rel_l2_pct` | **9.99** | yi#13 | wio9pqw2 | 2026-04-29 |
| `test_primary/wall_shear_rel_l2_pct` | **16.60** | yi#13 | wio9pqw2 | 2026-04-29 |
| `test_primary/volume_pressure_rel_l2_pct` | **14.21** | yi#13 | wio9pqw2 | 2026-04-29 |

(yi#13 was norman's progressive cosine EMA 0.99→0.9999, pending merge on yi.)

## Confirmed-orthogonal levers from yi (to compose on tay)

1. Width 512d / heads 8 / slices 128 (yi PR #4 chihiro) — the 256d→512d step
   moves volume_pressure 15.21 → 14.37. Needs `lr=5e-5`, `bs=4` at 512d.
2. Protocol fixes (yi PR #9 gilbert) — `--volume-loss-weight 2.0
   --batch-size 8 --validation-every 1`. Halved abupt mean over the
   pre-protocol baseline.
3. Tangential wall-shear projection loss (yi PR #11 kohaku) — denormalize →
   project onto surface tangent → renormalize. Default off; opt in with
   `--use-tangential-wallshear-loss` if/when ported.
4. AdaLN-zero per-block FiLM geometry conditioning (yi PR #8 frieren) —
   independent +5% on every axis at 256d.
5. Cosine EMA decay 0.99 → 0.9999 (yi PR #13 norman) — single largest
   non-architectural lever in yi (−9% on every axis).

## DDP8 calibration reproduce (tay round 1, alphonse #?)

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --volume-loss-weight 2.0 \
  --batch-size 4 \
  --validation-every 1 \
  --lr 5e-5 --weight-decay 5e-4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.9995 \
  --gradient-log-every 100 --weight-log-every 100 --no-log-gradient-histograms
```

Effective batch size = 4 × 8 GPUs = 32. The yi winning per-GPU config is
preserved verbatim; per-GPU memory limit (96 GB) still binds at 512d.

## Update protocol

When a tay PR lands a new best `test_primary/abupt_axis_mean_rel_l2_pct`:
1. Update the Status header to the new PR + W&B run + date.
2. Replace the per-axis best table with the new run's `test_primary/*`.
3. Append a short "Compounding wins so far" entry naming the orthogonal lever.
