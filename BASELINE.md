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

In-wave val target: **6.6573%** (PR #806 dl24-frieren, EP28, run `gui4ceed`) — wave val leader as of 2026-05-07T~22:10Z; EP28=6.6573% (new run-best, 3-epoch monotonic descent EP26-28 confirmed); vol_p=4.0890% wave low; ws_total=7.4590%; 5L STRING + GradNorm α=0.25 + Y-sym (triple compose); nezuko #800 EP25=6.6828% (personal best, 0.026pp behind frieren); fern #794 EP25=6.7064% (run-best, plateau+drift EP26-EP45, terminal imminent ~22:35Z May 7); tanjiro #818 EP9=6.9372% (oscillation, EP10 gate due ~02:00Z May 8); EP50 ETAs: frieren ~07:30Z May 8, nezuko ~02:30Z May 8

### In-wave validation tracking (not yet merged; val metrics are mid-run, not terminal test)

| PR | Run | Student | Epoch | val_abupt | Notes |
|---|---|---|---:|---:|---|
| #741 | `lszc4ri7` | nezuko | EP33 (best) | **6.4984%** | Y-axis symmetry aug; MERGED PR #741; test=7.8232% |
| #745 | `co0xlqap` | frieren | EP30 (best) | **6.5097%** | 5L STRING PE; TERMINAL EP50 complete; test=7.845%; did NOT beat merged best (7.8232%); val beats pre-wave SOTA 6.5281% by 0.018pp |
| #749 | `oi2a01zy` | tanjiro | EP27 (best) | **6.8479%** | Lion lr=9e-5 control; CLOSED — plateau EP35-47 |
| #780 | `20n1fvwn` | dl24-tanjiro | EP49 (terminal) | **6.7669%** | GradNorm α=0.25, no Y-sym; TERMINAL SENPAI-RESULT 16:55Z May 7: full_val=6.7669% / **test=8.0647%** — does NOT beat wave best 7.5195% (+0.5452pp regression); every channel worse vs α=0.5; converged GradNorm: w_vol=2.351 (over-weighted), w_τz=0.487 (under-weighted); confirms α=0.5 is the unimodal optimum on the α-axis. **PR CLOSED** (not merged). |
| #784 | `sd59a9dq` | dl24-nezuko | EP18 (terminal) | **7.5605%** | QK-Norm + Y-axis aug; TERMINATED — EP20 gate (≤7.2%) MISSED. PR CLOSED. dl24-nezuko IDLE. |
| #791 | `g0um26ek` | dl24-frieren | EP15 (last logged) | **6.9102%** | GradNorm α=0.5 + Y-axis symmetry on 4L; W&B state=crashed at step 89,323 (~EP16); never reached cosine-tail descent; configuration superseded by nezuko #800 (5L STRING + α=0.5 + Y-sym). **PR CLOSED** (no terminal SENPAI-RESULT). |
| #794 | `em7eupj5` | dl24-fern | EP25 (best) / EP45 (latest) | **6.7064%** | GradNorm α=0.25 + Y-axis symmetry composition; EP25=6.7064% (run best); EP45=6.7542% (latest, +0.048pp above best); plateau+drift worsening EP26-45; wall shear drifted 7.5987%→7.6540%; τz rising; vol_p=3.9946% (run-low); cosine-tail re-engagement did NOT materialize; EP50 terminal IMMINENT ~22:35Z May 7 — SENPAI-RESULT due within ~30 min; test must use EP25 best-val checkpoint; baseline to beat: test=7.5195% |
| #800 | `hmhfnedy` | dl24-nezuko | EP25 (best) | **6.6828%** | 5L STRING + GradNorm α=0.5 + Y-sym; EP25=6.6828% (run best, all 5 channels at run-best simultaneously); slow descent re-engaged at EP25 (−0.014pp/ep after EP21-EP24 plateau); wall=7.4536% (WAVE LOW); vol_p=4.2757%; trailing frieren #806 EP28 by 0.026pp; EP50 terminal ETA ~02:30Z May 8 |
| #806 | `gui4ceed` | dl24-frieren | EP28 (best) | **6.6573% ← WAVE VAL LEADER** | 5L STRING + GradNorm α=0.25 + Y-sym (triple compose); EP28=6.6573% (wave val leader); EP26→EP28 clean 3-epoch monotonic descent −0.025pp; all channels improved EP27→EP28; GradNorm rebalancing: w_τz dropped 1.284→1.138 as τz improving, w_τx/w_τy crossed above 1.0; ws_total=7.4590% (closing on AB-UPT 7.29%); EP50 ETA ~07:30Z May 8 |
| #818 | `dy2z6o4a` | dl24-tanjiro | EP9 (latest) | EP9=6.9372% | 6-octave STRING PE + GradNorm α=0.5 + Y-sym (`--pe-init-sigmas "0.25,0.5,1.0,2.0,4.0,8.0"`); EP8=6.8343% (run best); EP9=6.9372% (oscillation, normal); EP5 gate ≤7.5% CLEARED; EP10 gate (≤7.2%) at step 54,930 ~02:00Z May 8 — 0.74pp clear at EP9, passage certain; EP7 showed wall advantage vs frieren on all 4 shear channels; vol_p lagging (+0.35pp vs frieren at EP7); EP50 ETA ~17:30Z May 8 |

_Last updated: 2026-05-07T~22:10Z. Major events: (1) frieren #806 (`gui4ceed`) EP28=**6.6573%** NEW WAVE VAL LEADER (beat EP24=6.6775%); EP26-28 monotonic 3-epoch descent −0.025pp; ws=7.4590% closing on AB-UPT 7.29%; EP50 ETA ~07:30Z May 8; (2) nezuko #800 EP25=**6.6828%** new personal best (all 5 channels simultaneous), slow descent re-engaged, wall=7.4536% wave low; trailing frieren by 0.026pp; EP50 ETA ~02:30Z May 8; (3) fern #794 EP45=6.7542% plateau+drift confirmed (+0.048pp from EP25 best=6.7064%); EP50 terminal IMMINENT ~22:35Z May 7 — expect SENPAI-RESULT posting within ~30 mins; test must use EP25 best-val checkpoint; (4) tanjiro #818 EP9=6.9372% (oscillation, EP8=6.8343% run best); EP10 gate (≤7.2%) due ~02:00Z May 8, passage certain (0.74pp margin)._
