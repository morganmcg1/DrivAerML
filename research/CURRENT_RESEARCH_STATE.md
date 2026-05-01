# SENPAI Research State

- 2026-04-30 ~23:50 UTC — Wave 1 in flight (16 students + edward/senku supplementary), all 16 bengio pods 1/1 READY at ~15h uptime. 0 PRs review-ready, 0 idle students, no human issues. All pods healthy.
- W&B survey at ~23:45 UTC — all runs active, ~60% complete (epoch 29-30/50). No test_primary completions yet.
- First Wave 1 results expected ~05:00–10:00 UTC May 1 (50 epochs, ~23h per trial at ~10-11 it/s).
- torch.compile bug resolved: all students running with `--no-compile-model`. All confirmed stable.
- thorfinn Trial B (gc=0.5 + wd=1e-4) crashed earlier; Trial A (wd=5e-4) running cleanly.
- Wave 2 hypothesis slate drafted: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md` — 16 assignments staged, ready to fire when Wave 1 PRs flip to review.
- askeladd regression noted: SDF log-Fourier run went from 9.20 → 9.442 (+0.242 at step 313k) — may be unstable or plateau-ing.

## Wave 1 Latest Snapshot (~23:45 UTC Apr 30, ~60% complete — NOT FINAL)

`val/abupt_axis_mean_rel_l2_pct` mid-training leaderboard (all `running`, no test_primary results yet):

| Rank | Student | Run ID | Step | Epoch | val_abupt | wsy | wsz |
|-----:|---------|--------|------:|------:|----------:|----:|----:|
| 1 | alphonse | m9775k1v | 516,972 | 29.0 | **7.221** | 9.120 | 10.879 |
| 2 | gilbert | kn756yk6 | 433,858 | 24.3 | 7.586 | — | — |
| 3 | kohaku | h7ve1hmb | 472,285 | 26.5 | 7.911 | — | — |
| 4 | haku | nbbbw8qw | 353,269 | 19.8 | 8.061 | — | — |
| 5 | nezuko | p8swf78o | 498,184 | 28.0 | 8.215 | — | — |
| 6 | emma | kuk0oy8g | 367,725 | 20.6 | 8.216 | — | — |
| 7 | frieren | l23vz4md | 468,349 | 26.3 | 8.275 | — | — |
| 8 | thorfinn | snrwvw14 | 532,850 | 29.9 | 8.334 | 11.133 | 12.258 |
| 9 | askeladd | uxrhudp1 | 511,694 | 28.7 | 8.457 | — | — |
| 10 | tanjiro | 846uciam | 521,966 | 29.3 | 8.462 | 11.169 | 12.371 |
| 11 | fern | pxty4knv | 534,508 | 30.0 | 8.604 | 10.573 | 12.335 |
| 12 | norman | 0iv7wifz | 518,124 | 29.1 | 8.643 | — | — |
| 13 | chihiro | kit58p2e | 510,130 | 28.6 | 8.821 | — | — |
| 14 | violet | em5ixfew | 524,883 | 29.5 | 8.949 | 12.212 | 13.052 |
| 15 | senku | k8ytnvh8 | 511,627 | 28.7 | 10.020 | — | — |

Note: edward (#16 by inference, running separately as PR #137) not in primary cohort rank.

**Key observations (~23:45 UTC)**:
- alphonse improved from 7.666 → **7.221** (step 317k → 517k). Fourier PE + 4L/256d recipe consolidating lead.
- gilbert improved from 8.246 → 7.586 at step 434k. 5L depth signal strengthening.
- Wall-shear y/z still universally above AB-UPT targets: best wsy=9.120, wsz=10.879 (alphonse) vs targets 3.65/3.63. ~5-7pp gap remains the binding structural constraint.
- No test_primary results yet. First completions expected ~05:00–10:00 UTC May 1.
- ~356k–422k steps remain across the cohort before training ends.
- senku (RFF) still last at 10.020. PR #140 (curriculum sampling) is a separate run.

## Most Recent Human Researcher Direction

No human researcher issues found. Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously.

AB-UPT targets to beat:
- surface_pressure_rel_l2_pct < 3.82
- wall_shear_rel_l2_pct < 7.29
- volume_pressure_rel_l2_pct < 6.08
- wall_shear_x_rel_l2_pct < 5.35
- wall_shear_y_rel_l2_pct < 3.65
- wall_shear_z_rel_l2_pct < 3.63
- abupt_axis_mean_rel_l2_pct ~ 4.51 (mean of 5 axis metrics)

Current best on bengio branch: none merged yet. Mid-training leader (alphonse) is 7.666 abupt (val), still ~3.15pp from target. Wall-shear y/z axes are the binding constraint — the entire wave is 3-4x above target on these axes.

## Current Research Focus and Themes

**Wave 1 (in flight) — Two parallel streams**:
- Stream 1 — Exploit radford prior (Fourier PE + 4L/256d + no-EMA + T_max=30): alphonse, fern, gilbert, haku, kohaku, emma, tanjiro, violet
- Stream 2 — Fresh ideas: askeladd (SDF), chihiro (asinh), edward (uncertainty), frieren (cross-attn), nezuko (mlp-6), norman (dropout), senku (RFF), thorfinn (gc+wd)

**Wave 2 (staged, not yet assigned) — Five themes**:
- A. Stack the winners on DDP4 (5L + Fourier + 128 slices + 60k + sw=2.0)
- B. Radford champion port to DDP8 pool (4L/512d/8H + EMA=0.9995 + gc=0.5 + lr=4.8e-4 + T_max=36)
- C. Loss-formulation edits to train.py (metric-aware loss, squared rel-L2, GradNorm, mixup)
- D. Architecture edits to model.py (DomainLayerNorm, FiLM, SO(3)-equivariant head, multi-scale attention)
- E. Data/training (curriculum, 96k pts, mirror TTA, SWA)

**Wave 2 priority pivot**: Wall-shear y/z is now the binding constraint. Wave 2 should over-index on:
- Equivariant or vector-aware heads for wall_shear (3-vector field)
- Heavy-tail-aware loss (asinh + scale tuning, Huber, focal-style reweighting)
- Higher-resolution surface tokenization for wall-shear gradient capture

## Potential Next Research Directions (Wave 3+)

- Physics-informed loss terms (continuity, momentum)
- Equivariant representations using SO(3)/SE(3) for normals and shear vectors
- Latent diffusion conditioning for geometric priors
- DeepSpeed ZeRO-3 for 1024d models once 512d recipe is confirmed
- Ensemble of best models from different seeds (cheap once we have a winner)
- Curriculum learning: easy cases first, then hard aerodynamic extremes
- Pretraining on synthetic/simplified CFD data then fine-tuning
- Graph neural network hybrid with Transolver backbone
- Boundary-layer-aware loss reweighting near no-slip walls
- Adjoint-method consistency loss using known CFD solver gradients
