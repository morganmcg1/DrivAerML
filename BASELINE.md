# DrivAerML Baseline Metrics

## Current Best: PR #74 — alphonse 4L/256d baseline (2026-05-01)

The current best result on the bengio branch is from PR #74 (alphonse), 4L/256d Transformer
with **`ContinuousSincosEmbed` PE** (the existing default), no-EMA, cosine LR with T_max=30.
Val metrics at best checkpoint (ep30, step 552,326, W&B run `m9775k1v`).

**Baseline correction (2026-05-01 16:14Z, frieren PR #218 audit)**: The Wave 1 alphonse PR was
a squash merge of an assignment commit only — no model code landed. FourierEmbed (chihiro PR
#176) was added to bengio later. So the 7.21% baseline does NOT use Fourier PE; it uses the
existing `ContinuousSincosEmbed` default. The `--fourier-pe` flag is now wired but has not
been validated as a standalone improvement on the 4L/256d recipe.

Note: These are **validation** metrics — test_primary eval from the ep30 checkpoint is pending.

**IMPORTANT val/test gap**: A systematic ~2x degradation on vol_p has been observed across experiments (e.g., val=4.17% → test~8-12%). Do not claim AB-UPT wins based solely on val metrics — test_primary confirmation required for all axis metrics before submission.

| Metric | Current Best (val) | AB-UPT Target |
|--------|-------------------|--------------|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **7.2091** | 4.51 |
| `val_primary/surface_pressure_rel_l2_pct` | 4.802 | 3.82 |
| `val_primary/wall_shear_rel_l2_pct` | 8.160 | 7.29 |
| `val_primary/volume_pressure_rel_l2_pct` | **4.166** ✓ | 6.08 |
| `val_primary/wall_shear_x_rel_l2_pct` | 7.109 | 5.35 |
| `val_primary/wall_shear_y_rel_l2_pct` | 9.100 | 3.65 |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.869 | 3.63 |

**vol_p beats AB-UPT target.** abupt is 7.21% vs 4.51% target — 2.7pp gap remains.

### Reproduce command (best config)

```bash
cd target/ && torchrun --standalone --nproc-per-node=4 train.py \
  --model-layers 4 \
  --model-hidden-dim 256 \
  --model-heads 4 \
  --no-use-ema \
  --lr 3e-4 \
  --lr-cosine-t-max 30 \
  --no-compile-model \
  --wandb-group bengio-wave2
```

(no `--fourier-pe` — the m9775k1v baseline used `ContinuousSincosEmbed`, the existing default)

## AB-UPT Public Reference Targets

| Metric | AB-UPT Target |
|--------|--------------|
| `test_primary/surface_pressure_rel_l2_pct` | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | 3.63 |
| `test_primary/abupt_axis_mean_rel_l2_pct` | ~4.51 (mean of 5 axis metrics) |

## Historical Reference: radford branch (PR #2593)

Approximately 12.96% abupt — the bengio Wave 1 result (7.21%) is a significant improvement.

## Update Log

- 2026-04-30: Branch initialized. No experiments merged yet. AB-UPT reference values are the targets.
- 2026-05-01: PR #74 (alphonse) merged as Wave 1 leader. New best val_abupt = 7.2091% (ep30, run m9775k1v). vol_p beats AB-UPT target at 4.166%.
- 2026-05-01 16:14Z: Corrected baseline description — alphonse `m9775k1v` used `ContinuousSincosEmbed` not FourierEmbed (frieren PR #218 audit confirmed via W&B run config); PR #74 was a squash merge of the assignment only, no model code. FourierEmbed was added later via chihiro PR #176. Reproduce command updated to drop `--fourier-pe`.
- 2026-05-02: PR #176 (chihiro) squash-merged — canonical FourierEmbed implementation and `--fourier-pe` flag are now on bengio. Baseline does NOT change (7.2091%). LR sweep result: lr=3e-4 confirmed optimal within {1e-4, 3e-4, 5e-4}; lr=5e-4 Trial B best at ep11=9.122% (worse than baseline), lr=1e-4 Trial A diverged. FourierEmbed standalone validation vs ContinuousSincosEmbed is a Wave 4 experiment target.
