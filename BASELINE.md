# Baseline (drivaerml-long-20260504)

Wave: DrivAerML long-run single-model DDP8 validation, started 2026-05-04. Branch was cut from `main`. No experiments have merged into this branch yet, so no in-wave single-model SOTA has been established here.

This file tracks the **single-model** scoreboard for this wave only. Ensemble-tier results (e.g. `18oalu1h`) are background context, not training targets — see `program.md` and `instructions/prompt-advisor.md`.

## Public reference targets to beat (AB-UPT, lower is better)

| Target | This repo metric | AB-UPT |
|--------|------------------|-----:|
| Surface pressure `p_s` | `test_primary/surface_pressure_rel_l2_pct` | 3.82 |
| Vector wall shear `tau` | `test_primary/wall_shear_rel_l2_pct` | 7.29 |
| Volume pressure `p_v` | `test_primary/volume_pressure_rel_l2_pct` | 6.08 |
| Wall shear `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | 5.35 |
| Wall shear `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | 3.65 |
| Wall shear `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | 3.63 |

## Pre-wave reference single-model evidence (background, not on this branch)

These W&B runs are the strongest verified single-model points from earlier branches and motivate this wave's hypotheses. They are **not** merged baselines on `drivaerml-long-20260504`; treat them as targets to validate under controlled long DDP8.

| Run | Mechanism | Test aggregate | Surface | Volume | Wall | tau_x / tau_y / tau_z |
|---|---|---:|---:|---:|---:|---:|
| `9mm3sz7x` | mild tau_y=1.2 / tau_z=1.3, lr 9e-5 | 8.123 | 4.128 | 12.051 | 7.454 | 6.566 / 8.326 / 9.543 |
| `nh96x7m4` | tau_y=1.5 / tau_z=2.0 | 8.171 | 4.209 | 12.118 | 7.505 | 6.649 / 8.348 / 9.531 |
| `341czkol` | GradNorm α=1.0 | 8.243 | 4.221 | 12.407 | 7.532 | 6.695 / 8.305 / 9.589 |
| `wyz68o8r` | EMA-proxy GradNorm α=0.5 | 8.236 | 4.271 | 12.213 | 7.504 | 6.556 / 8.466 / 9.672 |
| `5o7jc7wi` | extended cosine T_max=13 | 8.313 | 4.271 | 11.867 | 7.786 | 6.918 / 8.582 / 9.927 |
| `qqtdnlwq` | surface-loss weight 2.0 | 8.292 | 4.238 | 12.105 | 7.634 | 6.666 / 8.645 / 9.807 |
| `ki2q9ko9` | multi-sigma STRING init | 8.479 | 4.449 | 11.503 | 8.139 | 7.090 / 9.066 / 10.282 |
| `r5rw40rn` | volume-point curriculum (censored) | 8.497 | 4.363 | 12.199 | 7.972 | 6.897 / 8.949 / 10.077 |
| `wgvvevb9` | per-axis output scaling | 8.618 | 4.408 | 12.254 | 8.125 | 7.083 / 9.092 / 10.260 |

## Provenance policy

Run IDs and explicit configs are stronger evidence than PR merge state. The map flagged that PR #488's reported idea (`ki2q9ko9` multi-sigma STRING) was real as a W&B run but the merged code drifted; multi-sigma STRING was only fully present after #511/#516. Students and the advisor must verify that copied code actually implements the run config before claiming a replication win.

## Update protocol

When a wave PR merges with terminal `SENPAI-RESULT` and lower `test_primary/abupt_axis_mean_rel_l2_pct` than the in-wave best, append a new line to the table below and bump the "Current single-model best" pointer.

### In-wave merged single-model results

| PR | Run | Mechanism | Test aggregate | Surface | Volume | Wall | tau_x / tau_y / tau_z |
|---|---|---|---:|---:|---:|---:|---:|
| #599 | `sogus8sx` | multi-sigma STRING PE (`pe_init_sigmas=[0.25,0.5,1.0,2.0,4.0]`) | 7.9303 | — | — | — | — |
| #741 | `lszc4ri7` / `1tal40wr` | Y-axis symmetry augmentation (bilateral car geometry) | 7.8232 | 3.9821 | 11.3345 | 7.3076 | 6.5304 / 7.9248 / 9.3444 |
| #740 | `5x8wofzm` | GradNorm α=0.5 adaptive loss balancing | **7.5195** | 3.8810 | 10.7580 | 7.0610 | 6.3490 / 7.5600 / 9.0480 |

### Current single-model best on `drivaerml-long-20260504`

PR #740 (`5x8wofzm`), test_primary/abupt_axis_mean_rel_l2_pct = **7.5195%** (fern, GradNorm α=0.5 adaptive loss balancing) — improved from 7.8232% (+0.3037pp)

### In-wave val target (to beat before merging)

In-wave val target: **6.6507%** (PR #818 dl24-tanjiro, EP23, run `dy2z6o4a`) — wave val leader as of 2026-05-08T~10:00Z; tanjiro EP23=6.6507% (new wave val leader, 0.0066pp ahead of frieren EP28=6.6573%; EP20=6.6532%, EP21=6.6543%, EP22=6.6930% vp-spike transient, EP23=6.6507% recovery+new-best; surf_p=4.3979%, vol_p=4.3546%, ws=7.3589%, τx=6.3833%, τy=8.0033%, τz=10.1143%; active descent, EP24 in progress); frieren #806 EP42=6.6792% (best EP28=6.6573%, plateau EP28-42 range 6.6573–6.6808%; cosine tail EP43-50 remaining; vol_p=4.0577% WAVE LOW; EP50 ETA ~16:00Z May 8); nezuko #800 EP37=6.6917% (best EP25=6.6828%, plateau EP25-37 range 6.6828–6.6999%; ~13 epochs remaining; ws_total=7.4616% wave-low; τz=10.1428%); fern #831 EP7=6.8581% (6L, run best EP7; EP8=7.1219% vp-spike transient, EP9 in progress; ~32.5 min/epoch, wall-clock cutoff ~EP43-44)

### In-wave validation tracking (not yet merged; val metrics are mid-run, not terminal test)

| PR | Run | Student | Epoch | val_abupt | Notes |
|---|---|---|---:|---:|---|
| #741 | `lszc4ri7` | nezuko | EP33 (best) | **6.4984%** | Y-axis symmetry aug; MERGED PR #741; test=7.8232% |
| #745 | `co0xlqap` | frieren | EP30 (best) | **6.5097%** | 5L STRING PE; TERMINAL EP50 complete; test=7.845%; did NOT beat merged best (7.8232%); val beats pre-wave SOTA 6.5281% by 0.018pp |
| #749 | `oi2a01zy` | tanjiro | EP27 (best) | **6.8479%** | Lion lr=9e-5 control; CLOSED — plateau EP35-47 |
| #780 | `20n1fvwn` | dl24-tanjiro | EP49 (terminal) | **6.7669%** | GradNorm α=0.25, no Y-sym; TERMINAL SENPAI-RESULT 16:55Z May 7: full_val=6.7669% / **test=8.0647%** — does NOT beat wave best 7.5195% (+0.5452pp regression); every channel worse vs α=0.5; converged GradNorm: w_vol=2.351 (over-weighted), w_τz=0.487 (under-weighted); confirms α=0.5 is the unimodal optimum on the α-axis. **PR CLOSED** (not merged). |
| #784 | `sd59a9dq` | dl24-nezuko | EP18 (terminal) | **7.5605%** | QK-Norm + Y-axis aug; TERMINATED — EP20 gate (≤7.2%) MISSED. PR CLOSED. dl24-nezuko IDLE. |
| #791 | `g0um26ek` | dl24-frieren | EP15 (last logged) | **6.9102%** | GradNorm α=0.5 + Y-axis symmetry on 4L; W&B state=crashed at step 89,323 (~EP16); never reached cosine-tail descent; configuration superseded by nezuko #800 (5L STRING + α=0.5 + Y-sym). **PR CLOSED** (no terminal SENPAI-RESULT). |
| #794 | `em7eupj5` | dl24-fern | EP25 (best) / EP50 (terminal) | **6.7064%** | GradNorm α=0.25 + Y-axis symmetry composition (4L); EP25=6.7064% (run best); EP50=6.7599% (terminal); cosine-tail re-engagement did NOT materialize; TERMINAL test=**7.9011%** (+0.382pp regression vs 7.5195%); 4L architecture deficit vs 5L; **PR CLOSED** (not merged). |
| #800 | `hmhfnedy` | dl24-nezuko | EP25 (best) / EP37 (latest) | **6.6828%** | 5L STRING + GradNorm α=0.5 + Y-sym; EP25=6.6828% (run best); EP37=6.6917% (latest); plateau EP25-37 (range 6.6828%–6.6999%); ws_total=7.4616% (WAVE LOW); vol_p=4.2757%; τz=10.1428%; best_ckpt_updated=0; ~13 epochs remaining; EP50 terminal ETA ~14:00Z May 8 |
| #806 | `gui4ceed` | dl24-frieren | EP28 (best) / EP42 (latest) | **6.6573%** | 5L STRING + GradNorm α=0.25 + Y-sym (triple compose); EP28=6.6573% (run best); EP42=6.6792% (latest); plateau EP28-42 (range 6.6573%–6.6808%); EP38=6.8199% vp-spike transient, recovery EP39-42; vol_p=4.0577% (WAVE LOW); ws_total=7.4854%; τz=10.2787%; best_ckpt_updated=0; ~8 epochs remaining; EP50 ETA ~16:00Z May 8 |
| #818 | `dy2z6o4a` | dl24-tanjiro | EP23 (best) / EP24 (in progress) | **6.6507% ← WAVE VAL LEADER** | 6-octave STRING PE + GradNorm α=0.5 + Y-sym; EP23=6.6507% (NEW WAVE VAL LEADER, 0.0066pp ahead of frieren best); EP20=6.6532%, EP21=6.6543%, EP22=6.6930% vp-spike transient, EP23=6.6507% recovery+new-best; surf_p=4.3979%, vol_p=4.3546%, ws=7.3589%, τx=6.3833%, τy=8.0033%, τz=10.1143%; EP24 in progress; active descent; EP50 ETA ~17:30Z May 8 |
| #831 | `pnrgixj1` | dl24-fern | EP7 (best) / EP9 (in progress) | 6.8581% | 6L STRING + GradNorm α=0.5 + Y-sym; EP7=6.8581% (run best); trajectory EP1→EP7: 10.5984%→7.7839%→7.4178%→7.0987%→7.0461%→6.9071%→6.8581%; EP8=7.1219% vp-spike transient; EP9 in progress; surf_p=4.5281%, vol_p=4.6025%, ws=7.6185%, τz=10.3878% (EP7); 32.5 min/epoch; wall-clock cutoff ~EP43-44 |

_Last updated: 2026-05-08T~10:00Z. Major events: (1) tanjiro #818 (`dy2z6o4a`) EP23=6.6507% — NEW WAVE VAL LEADER (displaces frieren EP28=6.6573% by 0.0066pp); EP22=6.6930% was transient vp-spike, EP23 recovered and set new run best; surf_p=4.3979%, vol_p=4.3546%, ws=7.3589%, τz=10.1143%; EP24 in progress, active descent; most promising run in wave; (2) frieren #806 (`gui4ceed`) EP42=6.6792% (latest), plateau EP28-42 confirmed (range 6.6573–6.6808%); best remains EP28=6.6573%; EP38=6.8199% was transient spike, recovered EP39-42; ~8 epochs to EP50; (3) nezuko #800 (`hmhfnedy`) EP37=6.6917% (latest), plateau EP25-37 confirmed (range 6.6828–6.6999%); best remains EP25=6.6828%; ~13 epochs to EP50; (4) fern #831 (`pnrgixj1`) EP7=6.8581% run best; EP8=7.1219% transient spike, EP9 in progress; 6L slower at 32.5 min/epoch, wall-clock cutoff ~EP43-44; solid descent trajectory from EP1=10.5984%._
