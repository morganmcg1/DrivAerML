# DrivAerML Baseline — `tay`

**Branch:** `tay` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Status: PR #39 — 2026-04-29 22:05 UTC

New tay SOTA set by tanjiro's Lion optimizer (paper config `lr=1.7e-5, wd=5e-3`).
Drop-in AdamW replacement using sign-based updates with momentum. Lion's sign-update
sidesteps the per-coordinate gradient-magnitude compression caused by our
`grad-clip-norm=1.0` binding at every step. Best-val at epoch 9 (of 50 configured);
val curve still descending at cutoff — more budget or compile would improve further.

**W&B run:** `xonbs83i` (rank 0) — group `tay-lion-lr-sweep`
**Best-val checkpoint:** epoch 9 (val_abupt=14.22)

**NEW: tay crosses below yi frontier (15.82 → 15.43)**

### tay current best — `test_primary/*`

| Metric | This-repo key | tay best (PR #39) | PR #40 | yi best | AB-UPT |
|---|---|---:|---:|---:|---:|
| `abupt` | `test_primary/abupt_axis_mean_rel_l2_pct` | **15.43** | 17.25 | 15.82 | — |
| `surface_pressure` | `test_primary/surface_pressure_rel_l2_pct` | **9.45** | 10.92 | 9.99 | 3.82 |
| `wall_shear` | `test_primary/wall_shear_rel_l2_pct` | **16.28** | 18.33 | 16.60 | 7.29 |
| `volume_pressure` | `test_primary/volume_pressure_rel_l2_pct` | **13.83** | 14.71 | 14.21 | 6.08 |
| `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | **13.91** | 15.73 | 14.27 | 5.35 |
| `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | **19.58** | 21.80 | 19.49 | 3.65 |
| `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | **20.40** | 23.07 | 21.12 | 3.63 |

### Reproduce PR #39 config

```bash
cd target/
torchrun --standalone --nproc_per_node=8 train.py \
  --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 1.7e-5 --weight-decay 5e-3 \
  --volume-loss-weight 2.0 --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --ema-decay 0.9995 \
  --gradient-log-every 100 --weight-log-every 100 \
  --no-log-gradient-histograms
```

Note: run `xonbs83i` did NOT use `--compile-model` (pod was on pre-compile-fix commit).
Future Lion runs should add compile for ~5-15% additional throughput gain.

### Compounding wins so far

| PR | Who | Delta | Lever |
|---|---|---:|---|
| #30 | alphonse | baseline (19.81) | yi calibration config (4L/512d/8h, vol_w=2.0) |
| #33 | fern | **−2.04 (−10.3%)** | RFF coord features (sigma=1.0, 32 feats) — uncompiled |
| #40 | alphonse | **−0.52 (−2.9%) vs #33** | torch.compile fix → 12 epochs vs 9; beats #33 without RFF |
| #39 | tanjiro | **−1.82 (−10.5%) vs #40** | Lion optimizer lr=1.7e-5 — sign-based update, crosses yi frontier |

### INFRA NOTE — torch.compile bug FIXED (PR #40)

Two-line patch in `trainer_runtime.py`:
1. `drop_last=True` on `DistributedSampler` and `DataLoader` (lines 293, 301) — fixes partial-batch crash.
2. `unwrap_model(model)` in `accumulate_eval_batch` (line 929) — bypasses compile during eval because `pad_collate` produces variable-shape batches that trigger a symbolic-sum codegen bug in torch.inductor.

**All future runs should use `--compile-model` (the default) without `--no-compile-model`.**
Throughput: ~16 min/epoch compiled vs ~18 min uncompiled. 270-min budget → 12 epochs.

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

## Update protocol

When a tay PR lands a new best `test_primary/abupt_axis_mean_rel_l2_pct`:
1. Update the Status header to the new PR + W&B run + date.
2. Replace the per-axis best table with the new run's `test_primary/*`.
3. Append a short "Compounding wins so far" entry naming the orthogonal lever.
