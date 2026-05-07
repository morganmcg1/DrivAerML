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

In-wave val target: **6.8315%** (PR #794 dl24-fern, EP11, run `em7eupj5`) — wave val leader as of 2026-05-07T07:59Z; crossed tanjiro #780 EP25=6.8334% by 0.0019pp at 14 fewer epochs; training to EP50 terminal

### In-wave validation tracking (not yet merged; val metrics are mid-run, not terminal test)

| PR | Run | Student | Epoch | val_abupt | Notes |
|---|---|---|---:|---:|---|
| #741 | `lszc4ri7` | nezuko | EP33 (best) | **6.4984%** | Y-axis symmetry aug; MERGED PR #741; test=7.8232% |
| #745 | `co0xlqap` | frieren | EP30 (best) | **6.5097%** | 5L STRING PE; TERMINAL EP50 complete; test=7.845%; did NOT beat merged best (7.8232%); val beats pre-wave SOTA 6.5281% by 0.018pp |
| #749 | `oi2a01zy` | tanjiro | EP27 (best) | **6.8479%** | Lion lr=9e-5 control; CLOSED — plateau EP35-47 |
| #780 | `20n1fvwn` | dl24-tanjiro | EP25 (best/latest) | **6.8334%** | GradNorm α=0.25; EP25=6.8334% (EP22=6.8565%, EP20=6.9330%); EP20 gate cleared; rebase clean (zero diff); w_cp=0.923, w_τx=0.969, w_τy=0.246, w_τz=0.498, w_vol=2.364 (α=0.25 no Y-sym routes heavily to volume); plateau slope -0.0045pp/ep, proj EP50≈6.73%; training to EP50 terminal |
| #784 | `sd59a9dq` | dl24-nezuko | EP18 (terminal) | **7.5605%** | QK-Norm + Y-axis aug; TERMINATED — EP20 gate (≤7.2%) MISSED. PR CLOSED. dl24-nezuko IDLE. |
| #791 | `g0um26ek` | dl24-frieren | EP13 (best) / EP15 (latest) | **6.9635%** | GradNorm α=0.5 + Y-axis symmetry composition; EP13 best=6.9635% (cp=4.5765%, vol=4.4188%, τx=6.8010%, τy=8.4882%, τz=10.5332%); w_cp=0.762, w_τx=0.908, w_τy=1.052, w_τz=1.451, w_vol=0.827 (α=0.5 + Y-sym routes to τz, Y-sym relieves volume pressure); rebase clean (zero diff, 422da712); EP15 in progress; projecting EP20-EP25 in 6.6-6.7% |
| #794 | `em7eupj5` | dl24-fern | EP13 (best/latest) | **6.7834% ← WAVE VAL LEADER** | GradNorm α=0.25 + Y-axis symmetry composition; EP13=6.7834% (new run best, -0.030pp vs EP12); EP14 uptick=6.8195% (noise); EP11 crossed tanjiro #780; all gates cleared early; fastest convergence in wave; w_vol=1.058, w_τz=1.274; projecting EP20≈6.59%, EP50≈6.2-6.4%; ~EP15 in progress |
| #780 | `20n1fvwn` | dl24-tanjiro | EP28-30 (best/latest) | **6.8154%** | GradNorm α=0.25 no Y-sym; EP28=6.8216% run best; EP29=6.8264%; EP30=6.8154% (latest); tail slope -0.006pp/ep; w_vol=2.351 (strongly elevated, volume hardest component); α=0.25 no Y-sym ablation vs fern α=0.25+Y-sym; EP50 terminal target ~6.7-6.8% |
| #800 | `hmhfnedy` | dl24-nezuko | EP4 (best/latest) | **7.1180%** | 5L STRING + GradNorm α=0.5, no Y-sym; EP4=7.1180% (Δ=-0.41pp vs EP3); already 0.08pp below EP20 gate (≤7.2%); w_τz=1.5899 (climbing toward 10.54% shear z); sharp convergence ahead of all gates; EP5 gate ~09:10 UTC; CP=4.658%, vol=4.651%, τz=10.541% |
| #806 | `gui4ceed` | dl24-frieren | EP1 (best/latest) | **11.1953%** | 5L STRING + GradNorm α=0.25 + Y-sym (triple compose); EP1=11.195%, 1.13pp ahead of fern #794 at same epoch; GradNorm weights nearly uniform (expected at EP1); w_τz/τy/τx climbing; EP5 gate ~10:30 UTC; highest-complexity composition in wave |

_Last updated: 2026-05-07T09:05Z (fern #794 EP13=6.7834% WAVE VAL LEADER — all 4 runs healthy; tanjiro #780 EP30=6.8154% on tail slope -0.006pp/ep; nezuko #800 EP4=7.1180% gates clear; frieren #806 EP1=11.195% triple-compose 1.13pp ahead of fern at same epoch; GradNorm routing confirmed: α=0.25 no Y-sym → w_vol=2.351 (tanjiro), α=0.5+no Y-sym → w_τz=1.590 (nezuko), α=0.25+Y-sym → w_vol=1.058/w_τz=1.274 (fern); all pods 1/1 ready)_
