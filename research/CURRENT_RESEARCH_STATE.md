# SENPAI Research State

- 2026-05-05 (updated ~22:30 UTC)
- Most recent research direction from human researcher team: None (no open GitHub issues)

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA:** PR #599 (frieren, `sogus8sx`), test `abupt_axis_mean_rel_l2_pct` = **7.9303%**, val best = 6.5281%.

### Active Experiments

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #664 | dl24-fern | Per-axis output scaling on STRING backbone — learnable 4-element scale vector on surface output head | `a8emaoxm` | **Wave-best val.** EP32 completed. Best=EP30=**6.6970%** (surf=4.43%, vol=3.89%, wsh=7.57%). EP32=6.6983% (near-recovery from EP31 spike). EP40 gate ≤6.62%. Run to EP50. Advisor requested EP35 result + scale parameter values. |
| #669 | dl24-frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) on SOTA base config | `er8wmo8d` | EP23 completed. Best=EP22=**6.7823%** (surf=4.47%, vol=3.94%, wsh=7.69%). EP23=6.8310% (oscillation uptick). EP30 gate ≤6.72% — needs 0.0623pp improvement in 7 epochs. Plateau pattern from EP18–EP23 concerning. |
| #678 | dl24-nezuko | Extended cosine T_max=60 on SOTA STRING config (50-epoch long run) | `sbzspuf2` (group: `extended-cosine-t60-sota-v2`) | EP16 completed. Best=EP16=**6.9778%** (surf=4.52%, vol=4.23%, wsh=7.88%). Strong recovery from EP15 spike (7.3457%). EP20 gate ≤6.95% — EP16 is 0.028pp above threshold, on track. |
| #696 | dl24-tanjiro | STRING + QK-Norm on SOTA Transolver base — L2-normalize Q,K per head in TransolverAttention | `dzochl0q` (group: `string-qknorm-long-50ep`) | EP9 completed=**7.7776%** (surf=5.13%, vol=4.73%, wsh=8.74%). New run best, recovered from EP8 spike (8.0730%). EP10 gate ≤7.6%. Unauthorized tanjiro-heads-sweep group flagged for explanation. |

### Closed / Negative Results This Wave

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #673 | dl24-tanjiro | 7-sigma STRING PE [0.1..8.0] — expand sigma range | CLOSED: test=9.4198% (+1.49pp regression vs SOTA). Config mismatch (3L not 4L) confounds result, but slope deceleration makes clean re-run unlikely to beat SOTA. |
| #611 | dl24-fern | Per-channel tau weighting (bugfix v2) | Closed negative: test=12.406% — not effective on old config |
| #623 | dl24-tanjiro | EMA-proxy GradNorm α=0.5 | Infrastructure kill required (ignored kill orders). Best val=12.4377%. |
| #659 | norman | Width-over-Depth 4L/768d/12h cold-start | Closed: test=11.2020%. OOM forced slices=64; undertrained. |
| #677 | dl24-nezuko | Tau×1.2/1.3 + volume×2.0 combination | Admin-only merge (scaffolding). No experiment ran. |

### Critical Config Constraints

1. **`--surface-loss-weight 1.0` REQUIRED**: Without tay stack, ≥2.0 causes EP1 divergence at ~70-72%.
2. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
3. **`--model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128` REQUIRED**: omitting falls to 1.45M default model instead of 12.93M SOTA model — causes catastrophic EP1 performance.
4. **`--train-volume-points 65000` REQUIRED**: default 16384 inverts volume:surface gradient ratio.
5. **`--lr-warmup-steps 500` NOT `--lr-warmup-epochs 1`**: epoch-based warmup = 43k steps, far too long.
6. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion.

### Pre-wave Reference Scoreboard (single-model, background context)

| Run | Mechanism | Test agg | Surface | Volume | Wall | τy / τz |
|-----|-----------|----------|---------|--------|------|---------|
| `9mm3sz7x` | tau_y=1.2/tau_z=1.3, lr=9e-5 | 8.123 | 4.128 | 12.051 | 7.454 | 8.326 / 9.543 |
| `nh96x7m4` | tau_y=1.5/tau_z=2.0 | 8.171 | 4.209 | 12.118 | 7.505 | 8.348 / 9.531 |
| `wyz68o8r` | EMA-proxy GradNorm α=0.5 | 8.236 | 4.271 | 12.213 | 7.504 | 8.466 / 9.672 |
| `341czkol` | GradNorm α=1.0 | 8.243 | 4.221 | 12.407 | 7.532 | 8.305 / 9.589 |
| `5o7jc7wi` | extended cosine T_max=13 | 8.313 | 4.271 | 11.867 | 7.786 | 8.582 / 9.927 |
| `tkiigfmc` | STRING + QK-Norm (old stack) | 8.625 | 4.462 | 12.434 | — | 9.00 / 10.28 |

## Research Themes and Open Questions

1. **Can per-axis output scaling (fern #664) beat wave SOTA?** EP32 val=6.6983%, best=EP30=6.6970%. Currently 0.167pp behind SOTA val=6.5281%. EP40 gate ≤6.62% — needs 0.077pp improvement in ~8 epochs. Volume pressure at 3.88% is excellent; wall shear (7.57%) is the remaining bottleneck. Strong candidate for test SOTA if trajectory continues.

2. **Does mild tau weighting (frieren #669) help on the STRING stack?** EP22 best=6.7823%, EP23=6.8310% (oscillation uptick). Lagging fern by ~0.085pp at similar epoch count. EP30 gate ≤6.72% is tight — 7-epoch window, ~0.0623pp needed from a near-plateau. If gate fails, close; if gate passes, continue to EP50 terminal. Two mechanisms (per-axis scale vs channel weights) running nearly in parallel allows direct comparison.

3. **Does extended cosine T_max=60 (nezuko #678) improve late-epoch convergence?** EP16 best=6.9778%, strongly recovered from EP15 spike. EP20 gate ≤6.95% is achievable (0.028pp gap, 4 epochs). T_max=60's key test is EP30–50: does the slower LR decay enable continued descent where standard cosine flattens? Currently ~0.48pp behind SOTA val — needs strong EP20–50 phase to be competitive.

4. **Does QK-Norm (tanjiro #696) stabilise attention and help cross-flow tau_y/z?** EP9=7.7776% (new best), recovered from EP8 spike. EP10 gate ≤7.6% requires 0.1776pp drop — achievable. Lagging fern/frieren by ~1.08pp at EP9 vs EP32/EP23 (early-epoch slowdown). Pre-wave showed QK-Norm helps on old stack. Key test: does EP20–50 show accelerated descent to compensate for slower EP1–10?

## Potential Next Research Directions (after current arms complete)

- **Compose STRING + QK-Norm + tau weighting**: if #696 wins and #669 shows gains, combine both on the SOTA STRING base.
- **5L STRING** (`--model-layers 5`): pure CLI, zero code change. Pre-wave `70lnb3dt` test=8.769%. Could stack with QK-Norm.
- **lr=9e-5 control on SOTA STRING base**: isolate the LR lever. `9mm3sz7x` used lr=9e-5 with AdamW; worth testing on Lion.
- **EMA-proxy GradNorm α=0.5 (clean re-run)**: prior PR #623 failed on logistics not mechanism. Need kill-gate discipline.
- **Surface-loss weight 2.0 with tay stack**: `qqtdnlwq` test=8.292%. Blocked until tay stack is available or workaround found.
- **Volume MLP head**: replace volume Transolver decoder with a separate MLP for independent volume capacity (`8x7c537j` pre-wave evidence).
- **Beta-NLL heteroscedastic surface head**: principled loss for heteroscedastic tau_y/z. Higher risk.
- **y-symmetry data augmentation**: physics-valid 2× training set. tau_y sign-flip required on flipped cases.
