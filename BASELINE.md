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

In-wave val target: **6.6053%** (PR #818 dl24-tanjiro, EP28/step~159324, run `dy2z6o4a`) — wave val leader as of 2026-05-08; tanjiro EP30=6.6089% (latest, tight oscillation near floor; best=6.6053% @ EP28/step~159324); fern #831 EP17=6.6378% (latest, plateau EP13-17; best=6.6164% @ EP13); nezuko #843 EP6=7.0719% (best, strong descent; EP5 ≤7.5% CLEARED, EP10 pre-cleared ≤7.2% CLEARED); frieren #844 EP2=8.2506% (<16% gate CLEARED, ~22 epochs remaining to EP5 gate)

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
| #800 | `hmhfnedy` | dl24-nezuko | EP25 (best) / EP50 (terminal) | **6.6828%** | 5L STRING + GradNorm α=0.5 + Y-sym; EP25=6.6828% (run best); EP50=6.6975% (terminal); TERMINAL test=**7.8981%** (+0.3786pp regression vs 7.5195%); critical vol_p val→test gap: val=4.2757% vs test=12.0379% (7.76pp); surf_p=3.9804%, ws=7.1561%, τx=6.2955%, τy=7.8062%, τz=9.3705%; GradNorm final: w_τx=0.719, w_τy=0.995, w_τz=1.608; **PR CLOSED** (not merged — does NOT beat wave SOTA) |
| #806 | `gui4ceed` | dl24-frieren | EP50 (terminal) | **6.6573%** (EP28 best) | 5L STRING + GradNorm α=0.25 + Y-sym; EP50 terminal: val=6.6573% (EP28 best), test=**7.9323%** (+0.413pp vs SOTA); vol_p val→test gap 2.95× (val=4.07%, test=12.03%); surf_p=3.95% ✓, ws=7.25% ✓; GradNorm w_vol_p surge (0.88→1.15) during cosine tail drove overfit; **PR CLOSED — does NOT beat wave SOTA 7.5195%** |
| #818 | `dy2z6o4a` | dl24-tanjiro | EP28 (best) / EP30 (latest) | **6.6053% ← WAVE VAL LEADER** | 6-octave STRING PE + GradNorm α=0.5 + Y-sym; val history: EP0.5=11.021%, EP1=8.120%, EP2=7.239%, EP3=7.093%, EP4=6.834%, EP5=6.818%, EP6=6.867%, EP7=6.851%, EP8=6.744%, EP9=6.685%, EP10=6.653%, EP11=6.693%, EP12=6.647%, EP13=6.609%, EP14=6.6053%→6.6143%, **EP28/step~159324=6.6053% (BEST RUN)**, EP30/step164819=6.6089% (latest); surf_p~4.382%, vol_p~4.301%, ws~7.308% at run best; tight oscillation near floor (noise band ~0.004pp); SENPAI-RESULT terminal:false posted; ~20 epochs remaining |
| #831 | `pnrgixj1` | dl24-fern | EP13 (best) / EP17 (latest) | **6.6164%** | 6L STRING + GradNorm α=0.5 + Y-sym; val history: EP5=6.993%, EP5.5=6.702%, EP6=6.840%, EP6.50=6.6164%, EP7=6.6245%, EP13=6.6164% (best), EP14=6.6245%, EP15=6.6333% (vol_p=4.2625% run low), EP16=6.6337%, EP17=6.6378% (latest); plateau sustained EP13-17 (0.0111pp behind wave leader; ascending 0.0214pp in 4 epochs); vol_p descending separately (4.2625% at EP15 run low); wall-clock cutoff ~EP43-44; strongly descending early but now plateaued |
| #843 | `hyzdxrj2` | dl24-nezuko | EP6 (best/latest) | **7.0719%** | 7-octave STRING PE (`pe_init_sigmas=[0.25,0.5,1.0,2.0,4.0,8.0,16.0]`) + GradNorm α=0.5 + Y-sym; val history: EP1/step5493=10.8989%, EP2/step10986=7.9426%, EP3/step16479=7.4226%, EP4/step21972=7.4031%, **EP5/step27465=7.1003% (≤7.5% gate CLEARED)**, **EP6/step32958=7.0719% (≤7.2% gate CLEARED EARLY)**; GradNorm weights at EP1: w_cp=0.834, w_vp=0.874, w_tx=1.032, w_ty=1.037, w_tz=1.223; strong descent continuing |
| #838 | `84skr4yq` | dl24-fern | EP4.1 (tay terminal) | 7.4255% | STRING RFF σ=0.125, 4-epoch tay screen; EP4.1=7.4255% (all tay gates passed); **CLOSED** — merge gate FAILED (tay EP4≤6.5985% required); promoted to long-run but test=8.7190% (+1.2pp regression vs SOTA); σ<0.25 axis confirmed dead at 65k surface points |
| #844 | `7dqsxvbq` | dl24-frieren | EP2 (latest) | 8.2506% | 5L STRING PE + GradNorm α=0.5 + no Y-sym; original run `3054rc61` intentionally killed by student at EP1 — wrong `--train-volume-points 16384` flag in PR body; relaunched 2026-05-08T07:44Z with corrected `--train-volume-points 65000 --train-surface-points 40000`; EP1/step10987=12.4553%, **EP2/step16413=8.2506% (<16% gate CLEARED)**; EP3 gate (<8%) pending ~step 19924; ~27.8 min/epoch → ~23.2h total EP50 |
| #842 | `3487klz8` | thorfinn (tay) | EP4 (terminal) | 7.610% | tay 4-epoch screen; **CLOSED 2026-05-08T09:03Z** — EP4=7.610%, gate ≤6.5985% FAILED by 1.011pp |
| #840 | `oiptel6p` | tanjiro-tay (tay) | EP4 (terminal) | 7.8558% | tay 4-epoch screen; **CLOSED 2026-05-08T09:03Z** — EP4=7.8558%, gate ≤6.5985% FAILED by 1.256pp |
| #823 | `ghh0s4ne` | nezuko-tay (tay) | EP1 (in progress, relaunched) | — | tay 4-epoch screen with --use-surf-to-vol-xattn; RELAUNCHED 2026-05-08T08:33Z (prior run `lp7u9r8g` killed by advisor error); EP1 in progress |
| #847 | `7gzie3gj` | dl24-frieren | EP1 (in progress) | — | LR warmup 2ep variant; EP1/step10987=26.3602% (gate <30% passed 2026-05-08T09:04Z); EP2 running with 2-epoch warmup |
| #848 | TBD | alphonse (tay) | — | — | tay 4-epoch screen: Lion lr=8e-5 vs 7e-5 downsweep; sequential plan approved by advisor; Arm A pending launch |
| #845 | `thimjhnd` (rank0) | dl24-fern | EP~1 (running) | — | RFF num_features=24, rff_init_sigmas=0.25,0.5,1.0,2.0,4.0 (sota-spectrum); group=rff-capacity-sweep; 8-GPU DDP; launched 2026-05-09T07:32Z |
| #846 | `px719275` | dl24-edward | EP~1 (running) | — | RFF num_features=32, rff_init_sigmas=0.25,0.5,1.0,2.0,4.0 (sota-spectrum); group=rff-capacity-sweep; launched 2026-05-09T07:33Z; ETA ~12:35Z |
| #849 | TBD | askeladd (tay) | — | — | tay 4-epoch screen: τy=2.0/2.5 vs τz=1.5 differential; Arm A running; student acknowledged EP2 gate correction (<16%) |

_Last updated: 2026-05-08 09:32Z. Major events: (1) tanjiro #818 (`dy2z6o4a`) run best=**6.6053%** @ EP28/step~159324 (WAVE VAL LEADER); EP30/step164819=6.6089% (latest); ~20 epochs remaining; surf_p~4.382%, vol_p~4.301%, ws~7.308% at run best; (2) fern #831 (`pnrgixj1`) run best=**6.6164%** @ EP13; plateau EP13-17 (6.6164-6.6378%); vol_p=4.2625% at EP15 (descending separately); wall-clock cutoff ~EP43-44; (3) nezuko #843 (`hyzdxrj2`) EP6=7.0719% (run best); EP5 gate ≤7.5% CLEARED, EP10 pre-cleared (7.0719% ≤ 7.2%); strong descent; (4) frieren #844 (`7dqsxvbq`) EP2=8.2506% (<16% gate CLEARED); EP3 gate (<8%) pending ~step 19924; ~23.2h projected EP50; (5) thorfinn #842 CLOSED 09:03Z — EP4=7.610% gate FAILED by 1.011pp; (6) tanjiro-tay #840 CLOSED 09:03Z — EP4=7.8558% gate FAILED by 1.256pp; (7) nezuko-tay #823 RELAUNCHED as `ghh0s4ne` at 08:33Z; (8) frieren #847 EP1=26.3602% gate <30% PASSED at 09:04Z._
