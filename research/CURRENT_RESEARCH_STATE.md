# SENPAI Research State
- 2026-05-04 01:36 UTC (Round 34 — advisor cycle)
- **CURRENT SOTA (yi branch): PR #517 (askeladd, Lion lr=1e-4 clip=0.5) — val_abupt 9.032%**. Active yi merge bar: **9.032%** (run `brat65z4`).
- **PR #490 (frieren STRING-sep learnable PE) MERGED to yi at 15:48 UTC 2026-05-03.** The `--learnable-pe` flag is now available in yi `train.py`. The next highest-priority merge is a from-scratch run combining STRING-sep + Lion lr=1e-4 clip=0.5 (PR #539 frieren, in-flight).
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

## Active WIP PRs (Round 34 — 14 PRs as of 2026-05-04 01:36 UTC)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #563 | edward | Variance-matching aux loss for τ_y/τ_z std collapse | **NEW — just assigned** |
| #566 | norman | Vol-to-surface cross-attention (near-wall pressure coupling) | **NEW — just assigned** |
| #567 | thorfinn | SDF-proximity surface loss weighting (geometry-based point weights) | **NEW — just assigned** |
| #547 | tanjiro | Coupled yz prediction head (Cholesky NLL, recovering from compile bugs) | Running |
| #546 | violet | Online hard example mining for τ_y/τ_z (Arm B k=0.25 running) | Running |
| #545 | senku | RoPE on Transolver slice tokens (3D centroid keys) | Approved, not launched — prodded 01:36 UTC |
| #544 | fern | y-symmetry paired loss + TTA (Arm B v3 with --no-compile-model running) | Running |
| #543 | haku | Surface curvature features (Arm A done 10.39%, Arm B running) | Running |
| #542 | alphonse | Laplace far-field soft penalty on vol pressure | Approved, not launched — prodded 01:36 UTC |
| #541 | askeladd | Streamline-aligned wall-shear target frame (tau.n non-zero finding) | Running |
| #540 | emma | GradNorm adaptive τ_y/τ_z up-weighting | Approved, not launched — prodded 01:36 UTC |
| #539 | frieren | STRING-sep + Lion lr=1e-4 clip=0.5 from scratch (run `t7g7hhed`) | Running |
| #528 | nezuko | 1-cycle LR pct_start=0.05 + STRING-sep (Arm C `up4xxvt7` running) | Running |
| #519 | gilbert | LLRD sweep (Arm A done 11.17%, Arm C running, Arm B to relaunch) | Running |
| #518 | kohaku | Loss-side asinh wallshear delta sweep | Running/stale |
| #478 | chihiro | Per-step cosine LR + eta_min fix (Arm B' `u72dousk`, ETA ~07:00 UTC) | Running |

## Students with Reassignment Deadline (60 min from ~01:36 UTC)

These students were approved >5h ago with no launch. They have until ~02:36 UTC to post launch confirmation or their PRs will be reassigned:
- **senku** (#545) — RoPE on slice tokens
- **alphonse** (#542) — Laplace far-field penalty
- **emma** (#540) — GradNorm adaptive weighting

## Recent Closed PRs (this session)

| PR | Reason |
|---|---|
| #551 (stark) | No student pod exists — hypothesis reassigned to edward (#563) |
| #520 (norman) | Null: coord transforms flat-to-worse on vol_pressure (6.12–6.19%) |
| #522 (thorfinn) | Null: FP32 last-epoch regresses val_abupt by +0.25pp (model not near basin) |
| #564 (norman duplicate) | Duplicate of askeladd #541 (streamline frame already running) — closed immediately |
| #565 (thorfinn duplicate) | Duplicate of haku #543 (surface curvature already running) — closed immediately |

## Research Themes (Round 34 priority order)

1. **Lock in STRING-sep + Lion combination** (PR #539 frieren) — highest confidence, locks bar to ~8.1%
2. **τ_y/τ_z structural attack (multi-prong):**
   - Variance-matching aux loss (PR #563 edward) — 2nd-moment regularizer
   - Streamline frame target rotation (PR #541 askeladd) — coordinate frame
   - GradNorm adaptive task weighting (PR #540 emma) — principled up-weighting
   - OHEM hard example mining (PR #546 violet) — gradient-based mining
   - Coupled yz head (PR #547 tanjiro) — joint modeling
   - y-symmetry paired loss (PR #544 fern) — equivariance enforcement
   - SDF-proximity loss weighting (PR #567 thorfinn) — geometry-based mining
3. **Volume pressure gap attack:**
   - Laplace far-field penalty (PR #542 alphonse) — physics-informed regularizer
   - Vol-to-surface cross-attention (PR #566 norman) — cross-stream coupling
4. **Architecture/geometric features:**
   - RoPE on slice tokens (PR #545 senku) — attention PE
   - Surface curvature features (PR #543 haku) — geometric input (Arm B running)
5. **Optimizer/LR confirmations:**
   - 1-cycle LR + STRING-sep (PR #528 nezuko, Arm C)
   - LLRD sweep (PR #519 gilbert — likely null, all arms may miss bar)
   - Per-step cosine eta_min fix (PR #478 chihiro, ETA ~07:00 UTC)

## Known Null Results (do not repeat)

- Volume coord transforms: signed-log, asinh on volume stream — flat (#520)
- BF16→FP32 final epoch — slight regression at <3 epoch horizon (#522)
- BERT-direction LLRD (decay < 1 on later layers) — hurts from-scratch training (#519 Arm A 11.17%)
- Model-only checkpoint resume — EMA/optimizer reset undoes gains (#472)
- Per-axis static loss weight sweeps (#244, #454)
- Uncertainty weighting Kendall (#496) — inverts gradient for lagging tasks
- LLRD: tentatively null for all 3 arms if B and C also miss by >10%

## Fleet Stability Constants

- Lion optimizer: lr=1e-4, warmup=1 epoch, clip=0.5, wd=5e-4
- 4×H100 DDP: ~1.25 s/it × 5442 steps/epoch → ~113 min/epoch → ~3 epochs in 6h budget
- `--no-compile-model` required when `--learnable-pe` is active (torch.compile inductor bug with multi-axis broadcast in ContinuousSincosEmbed)
- Flag name: `--clip-grad-norm 0.5` (NOT `--grad-clip-norm`) — verified via `python train.py --help`
