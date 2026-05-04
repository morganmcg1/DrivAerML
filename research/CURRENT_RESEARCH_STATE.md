# SENPAI Research State
- 2026-05-04 10:52 UTC (Round 33/34 — advisor cycle)
- **CURRENT SOTA (yi branch): PR #517 (askeladd, Lion lr=1e-4 clip=0.5) — val_abupt 9.032%**. Active yi merge bar: **9.032%** (run `brat65z4`).
- **PR #490 (frieren STRING-sep learnable PE) MERGED to yi at 15:48 UTC 2026-05-03.** The `--learnable-pe` flag is now available in yi `train.py`. Requires `--no-compile-model` (torch.compile inductor broadcast bug).
- **Tay SOTA (reference track, not yi): PR #511 (edward) — val_abupt 7.013% / test_abupt 8.313%** (tay branch only).

## Most Recent Research Direction from Human Researcher Team

**Issue #18** (open, standing directive):
> "Stop incremental tuning. Be bold with architecture. Empower students to replace the model backbone while maintaining logging/validation/checkpointing."

## Current Baseline: PR #517 (askeladd) — yi — val_abupt 9.032%

**Active yi merge bar: val_abupt < 9.032%** (run `brat65z4`)

| Metric | yi bar (test, PR #517) | Tay SOTA ref (PR #511 test) | AB-UPT target | Gap |
|---|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | 10.119 | 8.313 | — | — |
| `surface_pressure_rel_l2_pct` | — | 4.271 | 3.82 | 1.12× |
| `wall_shear_rel_l2_pct` | — | 7.786 | 7.29 | 1.07× |
| `volume_pressure_rel_l2_pct` | — | 11.867 | 6.08 | **1.95×** |
| `wall_shear_y_rel_l2_pct` | — | 8.582 | 3.65 | **2.35×** |
| `wall_shear_z_rel_l2_pct` | — | 9.927 | 3.63 | **2.73×** |

Dominant open gaps: τ_z (2.73×), τ_y (2.35×), vol_p (1.95×).

## Active WIP PRs (Round 33/34 — 16 PRs as of 2026-05-04 10:52 UTC)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #597 | fern | RANS Laplace regularizer on far-field volume pressure (vol_p 1.95× gap) | Running |
| #596 | emma | 1-cycle LR with 1e-3 peak for Lion optimizer | Running |
| #595 | alphonse | asinh target-space normalization for wall-shear loss (τ_y/τ_z gap) | Running |
| #591 | violet | Model Soup: weight-space checkpoint averaging | Running |
| #590 | thorfinn | Gradient EMA smoothing before optimizer update | Running |
| #589 | tanjiro | SDF gradient direction as auxiliary volume input | Running (Arm A in-flight `i7hi8rgu`) |
| #588 | senku | ADOPT optimizer (provably convergent Adam variant) | Running |
| #587 | norman | Dual-tower vol/surface cross-attention encoder | Running |
| #586 | nezuko | Progressive resolution curriculum | Running |
| #585 | kohaku | Angle-to-freestream + log local area surface features | Running (Arm A in-flight `425gcyt7`) |
| #584 | gilbert | Inverse local point density loss weighting | Running |
| #583 | edward | β-NLL heteroscedastic loss | Running |
| #582 | chihiro | Multi-EMA ensemble (3 shadow decays) | Running |
| #580 | haku | Principal surface curvatures as input features | Running |
| #578 | askeladd | Streamline-aligned wall-shear target frame | Running |
| #576 | frieren | STRING-sep + Lion from-scratch anchor run | Running |

**Note on "stale" flags**: PRs #589 (tanjiro) and #585 (kohaku) are flagged stale by GitHub update time but confirmed active — both have W&B experiments in-flight. Arm A runs launched ~07:30–07:35 UTC; results expected within the 6h budget window.

## Research Themes (Round 33/34 priority order)

1. **Lock in STRING-sep + Lion combination** (PR #576 frieren) — anchor run combining both known wins, expected ~8.1% if orthogonal
2. **τ_y/τ_z structural attack (multi-prong):**
   - asinh target normalization for wall-shear (PR #595 alphonse) — compresses heavy tail
   - SDF gradient direction as aux vol input (PR #589 tanjiro) — geometry-informed representation
   - Streamline-aligned target frame (PR #578 askeladd) — coordinate frame rotation
   - β-NLL heteroscedastic loss (PR #583 edward) — per-point uncertainty modeling
   - Inverse density loss weighting (PR #584 gilbert) — upweight sparse high-error regions
3. **Volume pressure gap attack:**
   - RANS Laplace regularizer (PR #597 fern) — physics-informed far-field ∇²p ≈ 0 penalty
   - Dual-tower cross-attention vol/surf (PR #587 norman) — cross-stream coupling
4. **Architecture/input features:**
   - Angle-to-freestream + log area (PR #585 kohaku) — 9-channel surface + 7-channel volume
   - Principal surface curvatures (PR #580 haku) — geometric input enrichment
   - Dual-tower cross-attention (PR #587 norman) — architectural rethink
   - ADOPT optimizer (PR #588 senku) — provably convergent Adam variant
5. **Training dynamics:**
   - 1-cycle LR with 1e-3 peak (PR #596 emma) — aggressive LR schedule
   - Gradient EMA smoothing (PR #590 thorfinn) — gradient conditioning
   - Progressive resolution curriculum (PR #586 nezuko) — curriculum learning
   - Multi-EMA ensemble (PR #582 chihiro) — 3 shadow decays for model averaging
   - Model soup checkpoint averaging (PR #591 violet) — weight-space ensembling

## Known Null Results (do not repeat)

- Volume coord transforms: signed-log, asinh on volume stream input — flat (#520)
- BF16→FP32 final epoch — slight regression at <3 epoch horizon (#522)
- BERT-direction LLRD (decay < 1 on later layers) — hurts from-scratch training (#519 Arm A 11.17%)
- Model-only checkpoint resume — EMA/optimizer reset undoes gains (#472)
- Per-axis static loss weight sweeps (#244, #454)
- Uncertainty weighting Kendall (#496) — inverts gradient for lagging tasks
- LLRD: tentatively null for all 3 arms if B and C also miss by >10%
- Coordinate frame transforms on vol stream (#520): signed-log, asinh flat
- FP32 final-epoch upcast (#522): slight regression

## Fleet Stability Constants

- Lion optimizer: lr=1e-4, warmup=1 epoch, clip=0.5, wd=5e-4
- 4×H100/H200 DDP: ~1.25 s/it × 5442 steps/epoch → ~113–191 min/epoch → ~2–3 epochs in 6h budget
- `--no-compile-model` required when `--learnable-pe` is active (torch.compile inductor bug with multi-axis broadcast in ContinuousSincosEmbed)
- Flag name: `--clip-grad-norm 0.5` (NOT `--grad-clip-norm`) — verified via `python train.py --help`
- OOM mitigation: chunked cdist (chunk_size=2048–4096) for pairwise distance computations over 65k-point clouds

## Recommended Base Config (from PR #222 + #490 learnable-PE)

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --agent <student> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --clip-grad-norm 0.5 --no-compile-model --learnable-pe \
  --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --model-slices 128 --ema-decay 0.999 --lr-warmup-epochs 1
```
