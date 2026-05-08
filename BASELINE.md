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

In-wave val target: **6.6090%** (PR #818 dl24-tanjiro, EP26, run `dy2z6o4a`) — wave val leader as of 2026-05-08T~18:00Z; tanjiro EP27.3=6.6132% (latest, slight rebound from EP26 best 6.6090%; tight range confirms sustained descent trend; EP50 ETA ~17:30Z May 8); frieren #806 EP47=6.6857% (latest, best EP28=6.6573%, deep plateau EP28-47; ~3 epochs to EP50 terminal ~17:00Z May 8); nezuko #800 EP50 TERMINAL=6.6975% (best EP25=6.6828%; test=7.8981%; PR CLOSED, not merged); fern #831 EP12.1 (run best EP11=6.7016%; EP20 gate cleared at EP9=6.7836%; wall-clock cutoff ~EP43-44)

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
| #806 | `gui4ceed` | dl24-frieren | EP28 (best) / EP47.0 (latest) | **6.6573%** | 5L STRING + GradNorm α=0.25 + Y-sym (triple compose); EP28=6.6573% (run best); EP47=6.6857% (latest, step=258,352); deep plateau EP28-47 (range 6.6573%–6.6857%); slope essentially flat; vol_p=4.0469% (WAVE LOW); ws_total=7.4975%; τz=10.2979%; ~3 epochs remaining; EP50 ETA ~17:00Z May 8 |
| #818 | `dy2z6o4a` | dl24-tanjiro | EP26.0 (best) / EP27.3 (latest) | **6.6090% ← WAVE VAL LEADER** | 6-octave STRING PE + GradNorm α=0.5 + Y-sym; EP26=6.6090% (new run best, improved from EP25=6.6150%, EP24=6.6467%); EP27=6.6132% (slight rebound but tight); strongly descending trend; surf_p=4.3835%, vol_p=4.3064%, ws=7.3215%, τx=6.3439%, τy=7.9653%, τz=10.0760% [EP25 channel]; EP50 ETA ~17:30Z May 8 |
| #831 | `pnrgixj1` | dl24-fern | EP11.0 (best) / EP12.1 (latest) | **6.7016%** | 6L STRING + GradNorm α=0.5 + Y-sym; EP11=6.7016% (run best); EP12.0=6.8403% (latest); EP20 gate (≤6.9%) CLEARED at EP9=6.7836%; oscillation pattern (EP7=6.8581%→EP8=7.1219% spike→EP9=6.7836%→EP10=6.9925%→EP11=6.7016%) expected for GradNorm; wall-clock cutoff ~EP43-44; 32.5 min/epoch |
| #843 | `hyzdxrj2` | dl24-nezuko | EP1.0 (latest) | 10.8989% | 7-octave STRING PE (`pe_init_sigmas=[0.25,0.5,1.0,2.0,4.0,8.0,16.0]`) + GradNorm α=0.5 + Y-sym; EP1=10.8989% (well within EP5 gate ≤7.5%); run just started 2026-05-08; EP5 gate checkpoint ~22:00Z May 8 |
| #838 | `84skr4yq` | dl24-fern | EP4.1 (tay terminal) | 7.4255% | STRING RFF σ=0.125, 4-epoch tay screen; EP2=25.49% ✓, EP3=10.58% ✓, EP3.6=8.025% ✓, EP4.1=7.4255% ✓ — all tay gates PASSED; SENPAI-RESULT pending; promote to EP50 long-run recommended |

_Last updated: 2026-05-08T~18:00Z. Major events: (1) tanjiro #818 (`dy2z6o4a`) EP26=6.6090% WAVE VAL LEADER; EP27.3 running, slight rebound to 6.6132% but trend sustained; EP50 ETA ~17:30Z May 8; (2) frieren #806 (`gui4ceed`) EP47=6.6857% (latest), deep plateau EP28-47; best EP28=6.6573%; ~3 epochs to EP50 terminal (~17:00Z May 8); (3) fern #831 (`pnrgixj1`) EP12.1, run best EP11=6.7016%; EP20 gate CLEARED at EP9=6.7836%; 6L run continuing with oscillation-then-descend pattern; (4) nezuko #843 (`hyzdxrj2`) EP1.7 — 7-octave STRING PE; EP5 gate checkpoint ~22:00Z May 8; (5) fern #838 (`84skr4yq`) EP4 tay PASSED: EP4.1=7.4255%; all gates cleared; SENPAI-RESULT pending; promote to EP50 recommended; (6) tay-wave screening: askeladd #836, nezuko #823, tanjiro #840, thorfinn #842 — all EP1.3-2.0, no val checkpoints yet._
