# SENPAI Research State

- 2026-05-05 (updated ~18:00 UTC)
- Most recent research direction from human researcher team: None (no open GitHub issues)

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA:** PR #599 (frieren, `sogus8sx`), test `abupt_axis_mean_rel_l2_pct` = **7.9303%**, val best = 6.5281%.

### Active Experiments

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #664 | dl24-fern | Per-axis output scaling on STRING backbone — learnable 4-element scale vector on surface output head | `a8emaoxm` | **Wave-best val.** EP29 completed. Best=EP29=**6.7337%** (in-wave val record). Merge conflict resolved. EP30 gate ≤6.745% (relaxed — improvement still ongoing). EP35 gate ≤6.72%. EP40 gate ≤6.70%. Run to EP50. |
| #669 | dl24-frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) on SOTA base config | `er8wmo8d` | EP21 completed (6.7888%). Best=EP20=**6.7885%**. Near-flat oscillation in EP20–EP21. EP30 gate ≤6.85%. Run to EP50. |
| #678 | dl24-nezuko | Extended cosine T_max=60 on SOTA STRING config (50-epoch long run) | `sbzspuf2`+7 ranks (group: `extended-cosine-t60-sota-v2`) | EP14 completed. Best=EP14=**7.0122%** (new best). EP20 gate ≤7.0% (critical boundary — currently 0.0122pp above gate). All 8 ranks healthy. |
| #696 | dl24-tanjiro | STRING + QK-Norm on SOTA Transolver base — L2-normalize Q,K per head in TransolverAttention | `dzochl0q`+7 ranks (group: `string-qknorm-long-50ep`) | EP7 completed=**7.9489%** (EP8 gate pre-cleared at EP7). Full 50-epoch run authorized. EP20 gate ≤7.2%. |

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

1. **Can per-axis output scaling (fern #664) beat wave SOTA?** EP29 val=6.7337% is the current wave-best val, 0.191pp behind SOTA val=6.5281%. Still improving — new record at EP29 after a brief micro-plateau in EP26–EP28. EP30 gate ≤6.745% (relaxed), EP35 gate ≤6.72%. High confidence this will beat 7.9303% on test. Trajectory still slightly downward — potential to reach low-6.7% by EP50.

2. **Does mild tau weighting (frieren #669) help on the STRING stack?** EP20 best=6.7885% is strong, near-matching fern's 6.7337% with a different mechanism. EP21=6.7888% is essentially flat, indicating an oscillation plateau in EP20–EP21 window. First clean test on Lion+STRING; prior evidence from `9mm3sz7x` (test=8.12%) used AdamW+non-STRING. EP30 gate ≤6.85%. If frieren continues descent past EP25 and surpasses fern, we have two competing mechanisms to compare for composition.

3. **Does extended cosine T_max=60 (nezuko #678) improve late-epoch convergence?** EP14 best=7.0122%. EP20 gate ≤7.0% is the critical checkpoint — currently 0.0122pp above gate. The T_max=60 LR schedule's theoretical value is superior late-epoch convergence past EP30–50. The early-epoch cost appears real (slower initial descent vs peers). Key question: does the slower LR enable continued descent through EP30–50 where standard cosine would plateau?

4. **Does QK-Norm (tanjiro #696) stabilise attention and help cross-flow tau_y/z?** EP7=7.9489% cleared the EP8 gate (≤8.0%) one epoch early. Full 50-epoch run authorized. EP20 gate ≤7.2%. Pre-wave `tkiigfmc` showed promise on old stack (test=8.625% vs SOTA 9.x% at time). Currently lagging fern/frieren by ~1.2pp at same epoch count — QK-Norm appears to slow early learning but may enable better late convergence.

## Potential Next Research Directions (after current arms complete)

- **Compose STRING + QK-Norm + tau weighting**: if #696 wins and #669 shows gains, combine both on the SOTA STRING base.
- **5L STRING** (`--model-layers 5`): pure CLI, zero code change. Pre-wave `70lnb3dt` test=8.769%. Could stack with QK-Norm.
- **lr=9e-5 control on SOTA STRING base**: isolate the LR lever. `9mm3sz7x` used lr=9e-5 with AdamW; worth testing on Lion.
- **EMA-proxy GradNorm α=0.5 (clean re-run)**: prior PR #623 failed on logistics not mechanism. Need kill-gate discipline.
- **Surface-loss weight 2.0 with tay stack**: `qqtdnlwq` test=8.292%. Blocked until tay stack is available or workaround found.
- **Volume MLP head**: replace volume Transolver decoder with a separate MLP for independent volume capacity (`8x7c537j` pre-wave evidence).
- **Beta-NLL heteroscedastic surface head**: principled loss for heteroscedastic tau_y/z. Higher risk.
- **y-symmetry data augmentation**: physics-valid 2× training set. tau_y sign-flip required on flipped cases.
