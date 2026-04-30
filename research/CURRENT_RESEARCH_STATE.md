# SENPAI Research State

- 2026-04-30 ~17:00 UTC — Wave 1 in flight (16 students), all 16 bengio pods 1/1 READY at ~6h50m uptime. 0 PRs review-ready, 0 idle students, no human issues. All pods healthy.
- W&B survey at 16:55 UTC — every student rank0 run has a fresh heartbeat (<2 min). Single failure: fern Trial B (lr=5e-4) all 4 ranks crashed at 14:44 UTC ~1 min after launch; Trial A (lr=1e-4) is alive. Advisor pinged fern to investigate. Frieren's earlier "duplicate run" cluster is fully cleaned up — only the original 4-rank job is running.
- First Wave 1 results expected ~21:00 UTC Apr 30 – 05:00 UTC May 1 (50 epochs ≈ 23h per trial at ~10-11 it/s).
- torch.compile bug: PyTorch 2.x Inductor `tiling_utils.get_pw_red_splits` crashes at first validation; all affected students relaunched with `--no-compile-model`. All confirmed stable.
- thorfinn Trial B (gc=0.5 + wd=1e-4) crashed earlier; Trial A (wd=5e-4) running cleanly at val_abupt~9.55 / vol_p=5.79 (below AB-UPT vol_p target).
- Wave 2 hypothesis slate drafted: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md` — 16 assignments staged across DDP4 + DDP8 pools, ready to fire when Wave 1 PRs flip to review.
- 4 silent students (alphonse, kohaku, haku, nezuko) pinged at 16:55 UTC with their W&B summary metrics + status request. They are mid-pack-or-better but not posting epoch-level updates.

## Wave 1 Mid-Training Snapshot (~6h50m, NOT FINAL — 16:55 UTC W&B summary metrics)

`abupt_axis_mean_rel_l2_pct` mid-training leaders (all `running`, no final/test results yet):

| Rank | Student | Treatment | abupt | vol_p | surf_p | wsx | wsy | wsz |
|------|---------|-----------|------:|------:|-------:|----:|----:|----:|
| 1 | alphonse | 4L/256d + Fourier + T_max=30 + no-EMA | **8.16** | **4.60** | 5.41 | 7.85 | 10.73 | 12.24 |
| 2 | gilbert | 5L/256d + Fourier | 8.78 | 6.29 | **5.80** | 8.12 | 11.14 | 12.54 |
| 3 | kohaku | 128 slices + Fourier | 8.99 | 6.53 | 5.97 | 8.29 | 11.23 | 12.94 |
| 4 | emma | 60k pts + Fourier | 9.20 | 6.72 | 6.16 | 8.61 | 11.42 | 13.07 |
| 5 | nezuko | mlp-ratio=6 | 9.27 | **5.41** | 5.96 | 8.83 | 12.70 | 13.45 |
| 6 | haku | 4L/384d width | 9.30 | **5.57** | 5.90 | 8.73 | 12.65 | 13.63 |
| 7 | frieren | cross-attention bridge | 9.43 | 6.01 | 5.96 | 8.84 | 12.71 | 13.61 |
| 8 | tanjiro | surface_weight=2.0 | 9.55 | **5.92** | 6.14 | 8.95 | 12.88 | 13.84 |
| 9 | thorfinn | gc=0.5 + wd=5e-4 | 9.55 | **5.79** | 6.16 | 8.96 | 13.03 | 13.84 |
| 10 | edward | uncertainty weighting | 9.80 | 6.79 | 5.95 | 8.84 | 13.40 | 13.99 |
| 11 | fern | lr=1e-4 (Trial B crashed) | 9.80 | 7.01 | 6.54 | 9.23 | 12.47 | 13.76 |
| 12 | askeladd | log-Fourier SDF | 9.88 | 6.34 | 6.32 | 9.18 | 13.36 | 14.19 |
| 13 | violet | T_max=50 + lr=2e-4 | 10.10 | 6.00 | 6.61 | 9.52 | 13.70 | 14.68 |
| 14 | chihiro | asinh wall shear (scale=1.0) | 10.16 | **5.62** | 6.12 | 9.66 | 14.23 | 15.15 |
| 15 | norman | dropout=0.1 | 10.46 | 6.32 | 6.77 | (—) | (—) | (—) |
| 16 | senku | RFF sigma=1.0 | 11.03 | 7.23 | 7.11 | 10.10 | 14.39 | 16.31 |

**Bold = below AB-UPT public target on that axis** (surface_p target 3.82, vol_p 6.08, wsx 5.35, wsy 3.65, wsz 3.63 — wall_shear axes still uniformly above target).

**Key observations**:
- alphonse is ahead by ~0.6pp on abupt and DOMINANT on volume_pressure (4.60 vs next-best 5.41 nezuko). Surface_pressure also leads.
- 7 of 16 runs already beat AB-UPT volume_pressure (6.08): alphonse, nezuko, haku, thorfinn, tanjiro, chihiro, frieren. Volume_pressure looks tractable.
- Wall-shear axes (especially y/z) remain 3-4x AB-UPT target across the entire wave — this is the binding constraint for the abupt_axis_mean target.
- Stream 1 (Fourier PE + 4L/256d + no-EMA) recipe variants occupy 5 of top 6 slots. Stream 2 ideas spread across mid-pack.
- senku (RFF sigma=1.0) is the clear laggard.

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

Current best on bengio branch: none merged yet. Mid-training leader (alphonse) is 8.16 abupt, still 3.65pp from target. Wall-shear y/z axes are the binding constraint.

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
