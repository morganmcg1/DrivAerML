# SENPAI Research State

- **2026-05-01 ~10:00 UTC** — haku assigned PR #190 (radford-champion-512d-ddp8): 4L/512d/8H + EMA=0.9995 + gc=0.5 + lr=4.8e-4 + T_max=36 on DDP8. This is the highest-priority Wave 2 experiment, porting the radford branch champion recipe. 0 review-ready; 0 idle students. All 16 Wave 1 students + 2 Wave 2 students (senku PR #145, haku PR #190) now running.
- **2026-05-01 ~09:30 UTC** — PR #145 (senku metric-aware loss, run `39dekqil`) at ep7 with val_abupt=10.60% — new all-channel best, ep5 spike confirmed transient. Trend line: ep8≈10.26%, ep9≈9.92%, ep10≈9.58% (targeting 'Excellent' zone ≤9.5%). ep10 kill threshold: >11.0% → relaunch at w=0.01. All 16 student pods healthy (1/1 READY). 15 Wave 1 runs still active + 1 Wave 2 run (senku). 0 review-ready; 0 idle students. Wave 1 completions expected 12:00-16:00Z May 1.
- **2026-05-01 ~08:45 UTC** — PR #137 (edward GradNorm, run `v5ybmwra`) closed: diverged, val_abupt=33.43%. Edward reassigned to PR #160: split surface output head — dedicated cp MLP + wall-shear MLP on alphonse base. PR #145 (senku metric-aware loss, run `39dekqil`) at step ~43k (ep~2-3), val_abupt=15.045%; ep5 gate: if >13% kill+relaunch at w=0.01. 14 Wave 1 runs still active, 1 Wave 2 run active. 0 review-ready; 0 idle students. Wave 1 completions expected 12:00-16:00Z May 1.
- **2026-05-01 ~07:30 UTC** — Wave 1 still running (16 active, all healthy on bengio). 0 PRs review-ready; 0 idle students. Latest W&B: alphonse at ep~38, val_abupt=7.263, surf_p=4.828, vol_p=4.186. **alphonse has peaked** at ep~31.8 (step 552,326): val_abupt=7.209 (slope now positive +0.001/1k_steps, trending up). gilbert close 2nd at ep~31.6 / 7.476. Senku PR #145 (run `39dekqil`) at ep~2.2: abupt=15.05% (down from 17.79 at ep~2.0); old senku trajectory was ep1=16.69→ep2=13.83→ep3=12.79→ep5=~9.98 — new senku is ~2pp behind, gate decision deferred to ep5. Edward GradNorm PR #137 (run `09kojb6q`) at ep~12, abupt=8.85% (descending). All Wave 1 runs ~5-6h to ep50 completion. **First test_primary metrics expected 12:00-14:00Z May 1.** Wave 2 assignment slate ready.
- **2026-05-01 ~07:00 UTC** — Wave 1 still running (15 active, all healthy on bengio). 0 PRs review-ready; 0 idle students. Latest W&B: alphonse leads at ep37, abupt=7.24, surf_p=4.83, vol_p=4.19. Senku PR #145 metric-aware loss (run `39dekqil`) just completed ep1 — abupt=17.79% at w=0.05, ~3pp behind cohort baseline (suggesting w=0.05 is borderline-heavy; will reassess at ep5). Edward GradNorm PR #137 (run `09kojb6q`) at ep11, abupt=8.95% (plateau forming, still far from AB-UPT targets). Wave 1 first completions expected 12:00-14:00Z May 1. Wave 2 assignment slate ready.
- **2026-05-01 ~03:30 UTC** — Wave 1 still running (16 active, survey confirms all healthy). 0 PRs review-ready; 0 idle students. No `test_primary/*` metrics yet. W&B survey shows alphonse at step 592k (val_abupt=7.22, leader). Wave 1 completions expected 08:00-13:00Z May 1. Wave 2 assignment slate ready.
- **2026-05-01 ~03:30 UTC** — Wave 1 still running (17 active, all pods healthy). 0 PRs review-ready; 0 idle students. Wave 1 completions expected 09:00-13:00Z May 1. Wave 2 assignment slate ready in `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`.
- **2026-05-01 ~01:47 UTC** — Wave 1 still running (15 active). PR #140 (senku/curriculum mass-weighted sampling) merged. Senku reassigned to C1 metric-aware loss (PR #145, branch `senku/metric-aware-loss-rel-l2`, Wave 2 Theme C1). 0 PRs review-ready; 0 idle students.
- **2026-04-30 ~23:30 UTC** — Wave 1 running (all 16 active healthy on bengio). PR #84 (edward/UW) reviewed and closed. PR #88 (senku/RFF) closed (dead end). edward reassigned to GradNorm (PR #137, branch `edward/gradnorm-shear-balance`). senku reassigned to curriculum mass-weighted sampling (PR #140, branch `senku/curriculum-mass-weighted-sampling`, Wave 2 Theme E1). 0 PRs review-ready; 0 idle students.
- **Wave 1 latest W&B** (~23:10 UTC): alphonse val_abupt=7.33 at step 421k (leader). fern at step 437k (closest to finish, ~500k steps total). Edward GradNorm PR #137 just launched (run `v5ybmwra`, ~step 0). Senku PR #140 curriculum sampling merged — senku now on C1 metric-aware loss (PR #145).
- **First Wave 1 final test results expected soon**: fern and thorfinn within 1-3h. No test_primary metrics yet.
- torch.compile bug: PyTorch 2.x Inductor `tiling_utils.get_pw_red_splits` crashes at first validation; all affected students relaunched with `--no-compile-model`. All confirmed stable.
- Wave 2 hypothesis slate drafted: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md` — staged across DDP4 + DDP8 pools, ready to fire when Wave 1 PRs flip to review.
- **Stale RFF sweep runs flagged for cleanup**: `3s9qatve` (edward-rff64-s50, abupt=130 diverged) and `fig141q6` (edward-rff128-s20, abupt=17.4) from old `edward-rff-sweep` group — unrelated to current PRs. Initially mis-identified as PR #137 GradNorm; correction comment posted on #137.
- **Wave 2 C1 now in flight**: senku PR #145 — metric-aware loss `mse_plus_raw_rel_l2` w=0.05 on alphonse base (4L/256d/4H + Fourier PE + T_max=30 + no-EMA). Radford PR #3302 validated this approach at val_abupt=3.700% (below all AB-UPT targets). DDP4 experiment.

## Wave 1 Updated Snapshot (~52h uptime, NOT FINAL — 2026-05-01 ~05:00 UTC W&B metrics)

Full per-axis val metrics:

| Student | Run ID | Step | val_abupt | surf_p | vol_p | wsx | wsy | wsz |
|---------|--------|-----:|---------:|-------:|------:|----:|----:|----:|
| alphonse | `m9775k1v` | 562,029 | **7.21** | **4.80** | **4.17** | 7.11 | 9.10 | 10.87 |
| gilbert | `kn756yk6` | 470,821 | 7.54 | 5.04 | **5.32** | 7.25 | 9.07 | 11.03 |
| kohaku | `h7ve1hmb` | 512,811 | 7.87 | 5.25 | **5.58** | 7.51 | 9.52 | 11.46 |
| haku | `nbbbw8qw` | 383,000 | 7.89 | 5.06 | **4.61** | 7.57 | 10.60 | 11.61 |
| nezuko | `p8swf78o` | 539,770 | 8.15 | 5.23 | **4.87** | 7.84 | 10.86 | 11.96 |
| frieren | `l23vz4md` | 508,230 | 8.23 | 5.24 | **5.00** | 7.88 | 11.03 | 12.02 |
| emma | `kuk0oy8g` | 397,375 | 8.22 | 5.53 | 5.92 | 7.90 | 9.92 | 11.85 |
| thorfinn | `snrwvw14` | 574,476 | 8.33 | 5.41 | **4.92** | 7.94 | 11.11 | 12.25 |
| askeladd | `uxrhudp1` | 554,713 | 8.42 | 5.53 | **4.71** | 8.07 | 11.32 | 12.45 |
| tanjiro | `846uciam` | 565,459 | 8.44 | 5.45 | **5.19** | 8.08 | 11.12 | 12.34 |
| fern | `pxty4knv` | 576,393 | 8.60 | 5.79 | 6.03 | 8.29 | 10.57 | 12.32 |
| norman | `0iv7wifz` | 561,176 | 8.61 | 5.62 | **5.12** | 8.22 | 11.42 | 12.68 |
| chihiro | `kit58p2e` | 553,167 | 8.77 | 5.30 | **4.82** | 8.58 | 12.12 | 13.03 |
| violet | `em5ixfew` | 568,340 | 8.92 | 5.79 | **5.17** | 8.53 | 12.06 | 13.07 |
| senku | `k8ytnvh8` | 554,802 | 9.98 | 6.38 | 6.82 | 9.17 | 12.59 | 14.94 |
| edward(v1) | `v5ybmwra` | 17,816 | 33.43 (CRASHED) | — | — | — | — | — |
| edward(v2) | `09kojb6q` | 113,011 | 9.70 | 6.38 | 7.94 | 8.70 | 12.05 | 13.42 |

(Bold vol_p = at or below 6.08 target; alphonse leads all axes. wsy/wsz still universally 2.5-4x above target.)

`abupt_axis_mean_rel_l2_pct` mid-training leaders (all `running`, no real test_primary metrics yet):

| Rank | Student | Run ID | step | abupt | surf_p | vol_p | wsx | wsy | wsz |
|-----:|---------|--------|------:|------:|-------:|------:|----:|----:|----:|
| 1 | alphonse | `m9775k1v` | 421k | **7.33** | **4.87** | **4.23** | – | – | – |
| 2 | gilbert | `kn756yk6` | 355k | 7.83 | 5.20 | 5.57 | – | – | – |
| 3 | kohaku | `h7ve1hmb` | 386k | 8.13 | 5.41 | 5.82 | – | – | – |
| 4 | haku | `nbbbw8qw` | 289k | 8.17 | 5.22 | **4.78** | – | – | – |
| 5 | emma | `kuk0oy8g` | 302k | 8.33 | 5.59 | 5.99 | – | – | – |
| 6 | tanjiro | `846uciam` | 427k | 8.66 | 5.60 | **5.32** | – | – | – |
| 7 | violet | `em5ixfew` | 430k | 9.07 | 5.84 | **5.29** | – | – | – |
| 8 | fern | `pxty4knv` | 437k | 8.76 | 5.87 | 6.15 | – | – | – |
| 9 | nezuko | `p8swf78o` | 409k | 8.39 | 5.36 | **4.94** | – | – | – |
| 10 | frieren | `l23vz4md` | 383k | 8.55 | 5.42 | **5.19** | – | – | – |
| 11 | thorfinn | `snrwvw14` | 436k | 8.61 | 5.57 | **5.00** | – | – | – |
| 12 | askeladd | `uxrhudp1` | 416k | 8.61 | 5.64 | **4.82** | – | – | – |
| 13 | norman | `0iv7wifz` | 424k | 8.90 | 5.78 | **5.28** | – | – | – |
| 14 | chihiro | `kit58p2e` | 415k | 8.98 | 5.42 | **4.93** | – | – | – |
| 15 | senku | `k8ytnvh8` | 416k | 10.15 | 6.49 | 6.89 | – | – | – |
| 16 | edward | `v5ybmwra` | ~0 | – | – | – | GradNorm PR #137 just launched |
| — | senku | — | — | – | – | – | PR #140 Wave 2 E1 curriculum sampling just launched |

**Bold numeric = at or below AB-UPT target for that axis (vol_p target=6.08)**. alphonse leads all axes. **10/16 already below the 6.08 vol_p target** (vol_p solved). Wall-shear y/z universally 2.5–4x above target — decisive binding constraint.

**AB-UPT targets to beat (all 6 axes, current best in parens)**:
- surface_pressure < 3.82 (best 4.87 — 1.05pp gap)
- volume_pressure < 6.08 (alphonse 4.23 beats target; **10+ students below target** — effectively solved)
- wall_shear_x < 5.35 (best 7.19 — 1.84pp gap, closing)
- wall_shear_y < 3.65 (best 9.33 — 5.68pp gap — **hard binding constraint**)
- wall_shear_z < 3.63 (best 11.01 — 7.38pp gap — **hardest binding constraint**)
- abupt_axis_mean ~ 4.51 (best 7.33 — 2.82pp gap)

**Key observations (~42h W&B refresh)**:
- alphonse still leads all axes (abupt=7.33, improved from 7.55 at 365k). Fourier PE + 4L/256d recipe dominant.
- gilbert rank 2 (7.83) — 5L depth signal confirmed.
- haku/nezuko/askeladd/chihiro: vol_p < 6.08 (below AB-UPT target). 
- thorfinn (425k) and fern (426k) approaching epoch 50 — first test_primary metrics imminent.
- Wall-shear y/z: NO run is within 5pp of AB-UPT target. This is the defining challenge.
- edward GradNorm PR #137 launching from bengio pod (iteration 67). Stale RFF sweep runs unrelated — cleanup later.

**Strong Wave 1 signals** (validated at ~42h): Fourier PE dominant. 5L > 4L (gilbert). 128 slices viable (kohaku). 60k pts useful (emma). SDF (askeladd) strong vol_p. Asinh (chihiro) competitive vol_p. mlp-ratio=6 viable (nezuko). Tanjiro (SW=2.0) recovering. RFF/dropout/uncertainty bottom half. Wall-shear y/z universally far from target — Wave 2 must address this directly.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — ADVISOR response drafted.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Incorporate into Wave 2+ design. Don't lean too heavily on reference train.py; empower students to make architectural leaps.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously.

AB-UPT targets to beat:
- surface_pressure_rel_l2_pct < 3.82
- wall_shear_rel_l2_pct < 7.29
- volume_pressure_rel_l2_pct < 6.08
- wall_shear_x_rel_l2_pct < 5.35
- wall_shear_y_rel_l2_pct < 3.65
- wall_shear_z_rel_l2_pct < 3.63
- abupt_axis_mean_rel_l2_pct ~ 4.51 (mean of 5 axis metrics)

Current best on bengio branch: none merged. Mid-training leader (alphonse) is at abupt=7.33, 2.82pp from target.

## Current Research Focus and Themes

**Wave 1 (in flight, ~42h) — Two parallel streams**:
- Stream 1 — Exploit radford prior (Fourier PE + 4L/256d + no-EMA + T_max=30): alphonse, fern, gilbert, haku, kohaku, emma, tanjiro, violet
- Stream 2 — Fresh ideas: askeladd (SDF), chihiro (asinh), frieren (cross-attn), nezuko (mlp-6), norman (dropout), senku (RFF), thorfinn (gc+wd)
- edward (PR #137 GradNorm CLOSED — diverged val_abupt=33.43%) → **edward (PR #160 split-output-heads-shear) — dedicated cp + wall-shear MLP heads on alphonse base**

**Wave 2 (staged, not yet assigned) — Five themes**:
- A. Stack the winners on DDP4 (5L + Fourier + 128 slices + 60k + sw=2.0)
- B. Radford champion port to DDP8 pool (4L/512d/8H + EMA=0.9995 + gc=0.5 + lr=4.8e-4 + T_max=36)
- C. Loss-formulation edits (GradNorm with correct lr+isolation, metric-aware loss, squared rel-L2, mixup)
- D. Architecture edits (DomainLayerNorm, FiLM, SO(3)-equivariant head, multi-scale attention)
- E. Data/training (curriculum, 96k pts, mirror TTA, SWA)

**CRITICAL GAP for Wave 2**: Wall-shear y/z axes are 3-4x above AB-UPT targets across ALL Wave 1 runs. Wave 2 MUST specifically target wall-shear improvement. Candidate approaches:
- Dedicated wall-shear loss upweighting (wsy/wsz separate from wsx and surface/volume)
- Surface-normal-aligned coordinate systems for shear prediction (physics-motivated)
- Log-transform of wall shear targets (not just asinh) to better capture extreme values
- Boundary layer-aware attention (thinner surface regions with higher gradients)
- Separate wsy/wsz prediction heads with specialized architectures
- GradNorm (edward) — once restarted correctly with isolated controller and lr=3e-4, primary Wave 2 shear-balance tool

**Physical explanation for wsy/wsz gap**: Car geometry has dominant side-flow vs axial-flow structure. Wall-shear y (lateral) and z (vertical/normal) capture cross-flow vortices and boundary layer separation that are harder to predict without explicit physics encoding. wsx (streamwise) aligns with primary flow direction and is easier to learn.

## Potential Next Research Directions (Wave 3+)

- Physics-informed loss terms (continuity, momentum residuals as auxiliary losses)
- Equivariant representations using SO(3)/SE(3) for normals and shear vectors
- Latent diffusion conditioning for geometric priors
- DeepSpeed ZeRO-3 for 1024d models once 512d recipe is confirmed
- Ensemble of best models from different seeds (cheap once we have a winner)
- Curriculum learning: easy cases first, then hard aerodynamic extremes
- Pretraining on synthetic/simplified CFD data then fine-tuning
- Graph neural network hybrid with Transolver backbone
- Boundary-layer-aware loss reweighting near no-slip walls
- Adjoint-method consistency loss using known CFD solver gradients
- Separate prediction head architectures for wsy/wsz (these axes have distinctly different error profiles)
- 6L or 7L depth scaling (gilbert confirms depth signal; 5L→6L may compound)
- Combined recipe: 5L + 384d + Fourier PE + 128 slices + 60k pts (stacking all Wave 1 winners)
- Investigate wall-shear y/z gap: why is wsy gap 5.68pp but wsx gap only 1.84pp? Physical + architectural explanation needed
- 4L/512d/8H + EMA=0.9995 + gc=0.5 + lr=4.8e-4 + T_max=36 (radford champion port, highest priority Wave 2 experiment)
- SDF features follow-up (askeladd rank 9, vol_p=4.97) — combine with Fourier PE for hybrid geometric encoding
- Asinh transform follow-up (chihiro rank 13, vol_p=4.92) — try on surface pressure too
- UW follow-ups (from edward PR #84 post-mortem): wider clamp [-10,10] or unclamped log_vars; loss-scale-aware init; drop regularizer; higher grad-clip floor (10–50)
