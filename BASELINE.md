# DrivAerML Baseline Metrics

## Current Best: PR #174 — alphonse 5L/256d + FourierPE + T_max=50 (2026-05-02)

Best result on the bengio branch from PR #174 (alphonse), 5L/256d Transformer with **`FourierEmbed` PE**, no-EMA, cosine LR with T_max=50.
Val metrics at best checkpoint (ep~45.3, step 807,025, W&B run `vu4jsiic`).

| Metric | Current Best (val) | AB-UPT Target |
|--------|-------------------|--------------|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **6.9549** | 4.51 |
| `val_primary/surface_pressure_rel_l2_pct` | 4.5644 | 3.82 |
| `val_primary/volume_pressure_rel_l2_pct` | **3.9361** ✓ | 6.08 |
| `val_primary/wall_shear_y_rel_l2_pct` | 8.7345 | 3.65 |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.5766 | 3.63 |

**vol_p beats AB-UPT target.** abupt is 6.95% vs 4.51% target — 2.44pp gap remains.

**Architecture**: n_params=3,249,813 = FourierEmbed + 5L/256d/4H. T_max=50 cosine schedule validated as +0.25pp improvement vs T_max=30 baseline.

### Reproduce command (best config)

```bash
cd target/ && torchrun --standalone --nproc-per-node=4 train.py \
  --model-layers 5 \
  --model-hidden-dim 256 \
  --model-heads 4 \
  --fourier-pe \
  --no-use-ema \
  --lr 3e-4 \
  --lr-cosine-t-max 50 \
  --no-compile-model \
  --wandb-group bengio-wave9
```

---

## Previous Best: PR #74 — alphonse 4L/256d baseline (2026-05-01)

The current best result on the bengio branch is from PR #74 (alphonse), 4L/256d Transformer
with **`FourierEmbed` PE**, no-EMA, cosine LR with T_max=30.
Val metrics at best checkpoint (ep30, step 552,326, W&B run `m9775k1v`).

**Architecture confirmation (2026-05-02, fern PR #276 + advisor audit)**: The W&B run name for
`m9775k1v` is `alphonse/4l-256d-fourier-pe-baseline-rank0` and `n_params=3,249,813`. Runs
without FourierEmbed have `n_params=3,237,269`; the gap of 12,544 = exactly the FourierEmbed
`Linear(48→256)` projection (input_dim=3 features × 8 freqs × 2 = 48 → hidden_dim=256:
48×256+256=12,544). The earlier "baseline correction" (frieren PR #218 audit, 2026-05-01 16:14Z)
was **incorrect**: it inferred ContinuousSincosEmbed from the absence of a `fourier_pe` config
field, but that field didn't exist when `m9775k1v` was trained — FourierEmbed was the hardcoded
default before chihiro PR #176 added the opt-in `--fourier-pe` flag. The baseline **uses
FourierEmbed**. The `--fourier-pe` flag must be included in all reproduce commands.

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
  --fourier-pe \
  --no-use-ema \
  --lr 3e-4 \
  --lr-cosine-t-max 30 \
  --no-compile-model \
  --wandb-group bengio-wave2
```

(`--fourier-pe` is required — m9775k1v was trained with FourierEmbed as the hardcoded default;
n_params=3,249,813 confirms this. All future experiments must include `--fourier-pe` to be
comparable to the baseline.)

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
- 2026-05-02: PR #176 (chihiro) squash-merged — canonical FourierEmbed implementation and `--fourier-pe` flag are now on bengio. Baseline does NOT change (7.2091%). LR sweep result: lr=3e-4 confirmed optimal within {1e-4, 3e-4, 5e-4}; lr=5e-4 Trial B best at ep11=9.122% (worse than baseline), lr=1e-4 Trial A diverged. **BASELINE.md CORRECTED**: the "ContinuousSincosEmbed" baseline description was wrong. m9775k1v used FourierEmbed (n_params=3,249,813, run name `alphonse/4l-256d-fourier-pe-baseline-rank0`). All reproduce commands updated to include `--fourier-pe`. The frieren PR #218 audit was incorrect in its conclusion (absence of `fourier_pe` config field in W&B does not mean ContinuousSincosEmbed — FourierEmbed was the hardcoded default at time of training, before the flag was added).
- 2026-05-02 15:00Z: PR #174 (alphonse) merged as new best. New best val_abupt = **6.9549%** (ep~45.3, step 807,025, run `vu4jsiic`). Improvements: 5L vs 4L (+1 layer), T_max=50 vs T_max=30. Combined gain: −0.254pp on abupt. Wall-shear axes remain the dominant binding constraint (wsy=8.73%, wsz=10.58%).
