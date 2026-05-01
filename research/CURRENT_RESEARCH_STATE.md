# SENPAI Research State

- 2026-04-30 UTC — Wave 1 in flight (16 students active). 0 PRs review-ready, 0 idle students, no human issues. All 16 bengio pods healthy.
- PR #145 (senku) sent back for rebase due to merge conflict with bengio base.
- All 16 Wave 1 experiments running. No test_primary completions yet.
- First Wave 1 results expected within ~6–12h (50 epochs, ~23h per trial). Leaders (alphonse, fern, thorfinn) approaching ~62–65% completion.
- torch.compile bug resolved: all students running with `--no-compile-model`. All confirmed stable.
- thorfinn Trial B (gc=0.5 + wd=1e-4) crashed earlier; Trial A (wd=5e-4) running cleanly.
- Wave 2 hypothesis slate: 16 assignments staged (see below), ready to fire when Wave 1 PRs flip to review.
- askeladd (SDF): recovered to 8.497% at step 539k — stable but below leaders.
- senku (metric-aware loss, PR #145): 9.983% at step 539k — last in cohort, auxiliary rel-L2 term not helping wall shear.

## Wave 1 Latest Snapshot (~05:30 UTC May 1, ~62-65% complete — NOT FINAL)

`val_primary/abupt_axis_mean_rel_l2_pct` mid-training leaderboard (all `running`, no test_primary results yet):

| Rank | PR# | Student | Run ID | Step | val_abupt | wsy | wsz |
|-----:|-----|---------|--------|------:|----------:|----:|----:|
| 1 | #74 | alphonse | m9775k1v | 545,120 | **7.212** | 9.103 | 10.869 |
| 2 | #76 | gilbert | kn756yk6 | 456,876 | 7.571 | 9.114 | 11.052 |
| 3 | #78 | kohaku | h7ve1hmb | 497,513 | 7.881 | 9.570 | 11.456 |
| 4 | #77 | haku | nbbbw8qw | 371,872 | 8.031 | 10.878 | 11.806 |
| 5 | #86 | nezuko | p8swf78o | 524,011 | 8.163 | 10.887 | 11.979 |
| 6 | #79 | emma | kuk0oy8g | 386,689 | 8.214 | 9.904 | 11.828 |
| 7 | #85 | frieren | l23vz4md | 493,269 | 8.249 | 11.057 | 12.053 |
| 8 | #89 | thorfinn | snrwvw14 | 558,956 | 8.322 | 11.108 | 12.239 |
| 9 | #80 | tanjiro | 846uciam | 550,086 | 8.446 | 11.147 | 12.347 |
| 10 | #82 | askeladd | uxrhudp1 | 539,366 | 8.497 | 11.338 | 12.462 |
| 11 | #75 | fern | pxty4knv | 560,775 | 8.583 | 10.537 | 12.297 |
| 12 | #87 | norman | 0iv7wifz | 545,961 | 8.612 | 11.422 | 12.687 |
| 13 | #83 | chihiro | kit58p2e | 537,654 | 8.769 | 12.112 | 13.038 |
| 14 | #81 | violet | em5ixfew | 552,422 | 8.922 | 12.056 | 13.073 |
| 15 | #145 | senku | k8ytnvh8 | 539,403 | 9.983 | 12.599 | 14.932 |

Note: edward (PR #137, GradNorm) running separately as supplementary; W&B run not found by student filter at this survey pass.

**Key observations (~05:30 UTC May 1)**:
- alphonse continues to lead at **7.212%** (step 545k, ~30.6 epochs). Fourier PE + 4L/256d recipe is the strongest so far.
- gilbert closing gap: 7.571% at step 457k. 5L depth scaling shows real signal.
- kohaku (128-slice): 7.881% at step 498k — surface tokenization resolution clearly matters.
- emma (60k points): 8.214% — higher point density helping but less than depth/slice scaling.
- Wall-shear y/z structurally stuck: best wsy=9.103%, wsz=10.869% (alphonse) vs targets 3.65/3.63. Gap is ~2.5x on y and ~3x on z.
- chihiro (asinh normalization): 8.769% — not beating vanilla Fourier PE; wall_shear_y still 12.112%.
- senku (metric-aware loss, PR #145): 9.983%, last in cohort — auxiliary rel-L2 loss is not helping.
- No test_primary metrics logged for any run. First completions expected ~08:00–12:00 UTC May 1.

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

Current best on bengio branch: none merged yet. Mid-training leader (alphonse) is 7.212% abupt (val), still ~2.7pp from target. Wall-shear y/z axes are the binding constraint — the entire wave is 2.5–3x above target on these axes.

## Current Research Focus and Themes

**Wave 1 (in flight) — Two parallel streams**:
- Stream 1 — Exploit radford prior (Fourier PE + 4L/256d + no-EMA + T_max=30): alphonse, fern, gilbert, haku, kohaku, emma, tanjiro, violet
- Stream 2 — Fresh ideas: askeladd (SDF), chihiro (asinh), edward (gradnorm), frieren (cross-attn), nezuko (mlp-6), norman (dropout), senku (metric-aware loss), thorfinn (gc+wd)

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
