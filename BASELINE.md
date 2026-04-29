# DrivAerML Baseline — `tay`

**Branch:** `tay` · **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Status: tay-bootstrap — 2026-04-29

Tay is a fresh advisor branch on top of `main` + the DDP8-hardened trainer
(commit `ceca448`, *Harden trainer defaults and add DDP8 support*). No tay
runs have completed yet, so the first wave establishes our baseline by
porting yi's proven configs onto DDP8 hardware.

The yi advisor's best results (different W&B project) are documented as our
reference floor while we calibrate. We will replace this section the moment
the first tay run finishes.

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
