## Hypothesis

**Issue #717 Exp 1C — Geometry conditioning for volume (priority 3): soft distance-to-surface scalar feature**

The chronic 3x volume_pressure test-vs-val gap (val~3.9%, test~11.5%) appears to be a *geometric generalization* failure: at training time the model sees the same vehicle silhouettes from multiple sampling realizations, but at test the held-out cars have different far-wake/under-body geometry that the model has never been told *anything about explicitly*. The volume encoder receives only raw `(x, y, z)` coordinates plus the spectral RFF expansion of those coordinates, with **no information about whether a query point is 5cm away from the car body or 5m downstream** — that distinction is left to the slice attention to discover.

This experiment tests **Issue #717 1C priority 3**: add a **soft distance-to-surface scalar feature** (signed distance / nearest-neighbor distance proxy) as an *extra input feature* to every volume query point.

**Mechanism (precise):**

1. For each volume query point `v_i`, compute `d_i = min over surface points s_j of ||v_i - s_j||_2` (per-batch, per-sample, on-GPU via `cdist`). The surface points used are the same 65,536-point sample already loaded for the surface head — no extra IO. This is a near-surface distance proxy, not a true SDF (no inside/outside sign), but for the DrivAerML wake field the wake is exclusively *outside* the body so unsigned distance is sufficient.
2. Apply a length-scale-aware encoding: `f_i = [log1p(d_i / s_ref), exp(-d_i / s_near), exp(-d_i / s_mid), exp(-d_i / s_far)]` with `s_ref = 1.0`, `s_near = 0.1`, `s_mid = 0.5`, `s_far = 2.0` (vehicle body length is ~5m in DrivAerML, so these are 2cm/10cm/0.5m/2m bands at body scale). This gives the encoder a multiscale "how close to the car am I" signal without committing to a single bandwidth.
3. Concatenate the 4-channel distance feature to the existing volume input feature vector **before** the input projection; widen the input linear `nn.Linear(in_features+4, hidden)`. Surface inputs are unchanged.

**Why this might help:**
- The volume encoder currently has to *infer* near-vs-far from raw coordinates and slice attention. Slice token aggregation is global; near-surface boundary-layer points are statistically rare in the 65k uniform volume sample, so the relevant inductive signal is weak.
- Test-set cars have *different* surface geometry, which means `cdist`-derived distance is a **car-relative** feature: a far-wake test point at 4m behind a held-out car gives the same `d_i` as a 4m far-wake training point — the feature is geometry-invariant in a way that raw `(x, y, z)` is not.
- This is exactly the "soft geometry feature" lever Issue #717 H3 explicitly calls out, and it is *orthogonal* to the dual-tower (PR #722, null), BC-type (PR #716, broken), coord-norm (PR #723, in flight), and outlier-sampling (PR #728, in flight) directions.

**Single arm** — simplest, cheapest soft-geometry feature first. If positive, follow-ups can stack curvature, true signed SDF, or learnable kernel widths.

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

You will need to make code changes in `target/`:

1. **`target/data/...` or wherever volume points are loaded:** compute and return per-point `d_i` between volume query and the *current sample's* surface points. Easiest: do this in the model forward, not the dataloader, so you have access to the surface tensor that's already on-GPU (avoids extra IO).
2. **`target/transolver_model.py` (or wherever the volume input projection lives):** widen the volume input linear by 4 channels and concat the distance encoding. Use `torch.cdist(volume_coords, surface_coords)` and reduce with `.min(dim=-1).values`. Profile: `cdist` for `(B=4, V=65536, S=65536)` is ~17 GFLOP per forward and runs in ~3-6ms on H100; small fraction of the ~16ms layer cost.
3. **No new CLI flag required if always-on**, but add a flag `--use-distance-feature` (bool, default True for this PR) so the no-distance control is just a flag flip in a follow-up.
4. **Test the `cdist` for memory:** at FP32 the distance matrix is 4*65536*65536*4 = 64 GB which will OOM. **Compute in FP16/BF16 and chunk over the volume dim if needed** (e.g. process volume in chunks of 8192 to keep peak at ~8 GB). On H100 with BF16 cdist this is fine.

## Training command

```bash
cd target/ && torchrun --standalone --nproc_per_node=8 train.py \
  --agent askeladd --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
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
  --wandb-group askeladd-sdf-distance-feature \
  --wandb-name askeladd/sdf-dist-v1
```

## Gates

- **EP1 time gate:** kill if epoch_time > 80 min (current SOTA stack ~37-38 min/epoch; cdist adds ~3-6ms per layer, expect ~40 min/epoch).
- **EP2 (step 21,729):** kill if val_abupt > 12%.
- **EP3 (step 32,594):** kill if val_abupt > 8%.

## Required reporting (Issue #717 9-column table)

After training, post the SENPAI-RESULT comment with:

1. W&B run ID
2. Best-val-abupt checkpoint metrics (val and test, all 7 channels including aggregate)
3. Best-val-volume_pressure checkpoint metrics
4. Final-epoch checkpoint metrics
5. Per-case test volume diagnostics: top 10 worst test cases by `volume_pressure_rel_l2_pct`
6. Per-region test volume error (slabs along x: front/cabin/wake; bands by distance-to-surface: 0-0.1m / 0.1-0.5m / 0.5-2m / 2m+)
7. Statement: did val volume gain transfer to test volume?
8. Required 9-col table:

| Run | Checkpoint | Aggregate | Surface p | Volume p | Wall shear | Tau x | Tau y | Tau z |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| #599 `sogus8sx` | reported best | | | 11.694 | 7.299 | | 7.941 | 9.535 |
| #592 `4k25s25e` | reported best | 7.9915 | 4.3322 | 11.933 | 7.334 | | 8.145 | 9.298 |
| #681 `dc031qpt` | reported best | | | 11.374 | 8.321 | | 9.596 | 10.738 |
| candidate | best aggregate | | | | | | | |
| candidate | best volume | | | | | | | |
| candidate | final | | | | | | | |

## Closure rules

- **Solid/major win on test_volume_pressure (<=10.0%):** mark `status:review`, this is a winner.
- **Weak win (<11.0%) but val regresses:** mark `status:review`, advisor will weigh.
- **Val beats SOTA (<6.5985%) but test_volume regresses:** mark `status:review`, this is single-model val SOTA.
- **Both val and test miss:** post SENPAI-RESULT, leave open for advisor close.
- **Crash/divergence/OOM:** report root cause clearly; advisor will decide on rerun.
