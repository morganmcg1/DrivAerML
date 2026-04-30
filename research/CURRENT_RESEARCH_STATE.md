# SENPAI Research State

- 2026-04-30 ~21:30 UTC — Wave 1 in flight (16 students), all 16 bengio pods 1/1 READY at ~9h uptime. 0 PRs review-ready, 0 idle students, no human issues. All pods healthy.
- W&B survey at ~21:30 UTC — all 16 primary rank0 runs active and improving. No test_primary completions yet.
- First Wave 1 results expected ~05:00–10:00 UTC May 1 (50 epochs, ~23h per trial at ~10-11 it/s).
- torch.compile bug resolved: all students running with `--no-compile-model`. All confirmed stable.
- thorfinn Trial B (gc=0.5 + wd=1e-4) crashed earlier; Trial A (wd=5e-4) running cleanly.
- Wave 2 hypothesis slate drafted: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md` — 16 assignments staged, ready to fire when Wave 1 PRs flip to review.
- askeladd regression noted: SDF log-Fourier run went from 9.20 → 9.442 (+0.242 at step 313k) — may be unstable or plateau-ing.

## Wave 1 Latest Snapshot (~21:30 UTC Apr 30, ~9h uptime — NOT FINAL)

`val/abupt_axis_mean_rel_l2_pct` mid-training leaderboard (all `running`, no test_primary results yet):

| Rank | Student | Step | val_abupt | surf_p | vol_p | wsx | wsy | wsz |
|-----:|---------|------:|----------:|-------:|------:|----:|----:|----:|
| 1 | alphonse | 317k | **7.666** | **5.090** | **4.383** | **7.492** | 9.913 | 11.449 |
| 2 | gilbert | 269k | 8.246 | 5.472 | 5.800 | 7.743 | 10.332 | 11.884 |
| 3 | kohaku | 292k | 8.437 | 5.629 | 6.116 | 7.917 | 10.399 | 12.123 |
| 4 | emma | 230k | 8.585 | 5.735 | 6.254 | 8.130 | 10.517 | 12.288 |
| 5 | haku | 220k | 8.753 | 5.586 | **5.272** | 8.237 | 11.944 | 12.723 |
| 6 | thorfinn | 332k | 8.854 | 5.711 | **5.218** | 8.400 | 11.987 | 12.953 |
| 7 | edward | 291k | 8.870 | **5.474** | 5.826 | 8.145 | 12.044 | 12.860 |
| 8 | nezuko | 311k | 8.895 | 5.684 | **5.153** | 8.420 | 12.232 | 12.985 |
| 9 | frieren | 290k | 8.946 | 5.675 | **5.427** | 8.481 | 12.161 | 12.985 |
| 10 | tanjiro | 323k | 8.978 | 5.778 | **5.500** | 8.489 | 12.048 | 13.077 |
| 11 | fern | 333k | 9.155 | 6.127 | 6.502 | 8.693 | 11.425 | 13.026 |
| 12 | norman | 321k | 9.292 | 6.021 | 5.625 | 8.765 | 12.463 | 13.584 |
| 13 | chihiro | 311k | 9.423 | 5.735 | **5.124** | 9.125 | 13.137 | 13.991 |
| 14 | askeladd | 313k | 9.442 | 6.193 | **5.355** | 8.891 | 12.998 | 13.773 |
| 15 | violet | 326k | 9.509 | 6.165 | 5.765 | 9.005 | 12.938 | 13.671 |
| 16 | senku | 312k | 10.521 | 6.768 | 7.046 | 9.661 | 13.517 | 15.613 |

**Bold = below AB-UPT target on that axis** (surf_p: 3.82, vol_p: 6.08, wsx: 5.35, wsy: 3.65, wsz: 3.63)

**Key observations (~21:30 UTC)**:
- alphonse is clear leader at 7.666 abupt (vol_p=4.383, an exceptional result). Gap over #2 gilbert is ~0.58pp.
- 11 of 16 runs already beat AB-UPT vol_p (6.08): alphonse, thorfinn, nezuko, frieren, tanjiro, haku, chihiro, askeladd, violet, edward, norman. Volume_pressure essentially solved.
- Wall-shear y/z axes still uniformly above target across all 16 runs (wsy range: 9.91–13.52, wsz range: 11.45–15.61 vs targets 3.65/3.63). This is the binding structural constraint.
- Fourier PE variants dominate top-4: alphonse (#1), gilbert (#2), kohaku (#3), emma (#4). Stream 1 recipe proven.
- askeladd (SDF-conditioned) regressed from 9.20 → 9.442 — may be plateau-ing or unstable.
- senku (RFF sigma=1.0) confirmed last at 10.521, ~2.9pp behind alphonse.

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
