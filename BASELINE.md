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

In-wave val target: **6.7064%** (PR #794 dl24-fern, EP25, run `em7eupj5`) — wave val leader as of 2026-05-07T13:18Z; EP25=6.7064% new run best (-0.004pp vs EP24=6.7104%); 5/7 channels at or near run-bests; vol_p=4.0189% (run low), τy=8.2687% (run low); GradNorm w_τz=1.438 (climbing); EP30 expected ~15:30Z; training to EP50 terminal

### In-wave validation tracking (not yet merged; val metrics are mid-run, not terminal test)

| PR | Run | Student | Epoch | val_abupt | Notes |
|---|---|---|---:|---:|---|
| #741 | `lszc4ri7` | nezuko | EP33 (best) | **6.4984%** | Y-axis symmetry aug; MERGED PR #741; test=7.8232% |
| #745 | `co0xlqap` | frieren | EP30 (best) | **6.5097%** | 5L STRING PE; TERMINAL EP50 complete; test=7.845%; did NOT beat merged best (7.8232%); val beats pre-wave SOTA 6.5281% by 0.018pp |
| #749 | `oi2a01zy` | tanjiro | EP27 (best) | **6.8479%** | Lion lr=9e-5 control; CLOSED — plateau EP35-47 |
| #780 | `20n1fvwn` | dl24-tanjiro | EP40 (best EP39=6.7702%) | **6.7702%** | GradNorm α=0.25, no Y-sym; noise floor ~6.770%; EP39=6.7702% run-best, EP40=6.7706% (+0.0004pp noise); vol_p=3.8509% at EP40 (best vol across wave); GradNorm w_vol=2.367, w_τz=0.491 (structurally inverted vs α=0.5: vol-heavy not τz-heavy); ~0.4pp structural deficit vs fern due to no Y-sym; terminal SENPAI-RESULT ~16:00-16:30Z May 7 |
| #784 | `sd59a9dq` | dl24-nezuko | EP18 (terminal) | **7.5605%** | QK-Norm + Y-axis aug; TERMINATED — EP20 gate (≤7.2%) MISSED. PR CLOSED. dl24-nezuko IDLE. |
| #791 | `g0um26ek` | dl24-frieren | EP13 (best) / EP15 (latest) | **6.9635%** | GradNorm α=0.5 + Y-axis symmetry composition; EP13 best=6.9635% (cp=4.5765%, vol=4.4188%, τx=6.8010%, τy=8.4882%, τz=10.5332%); w_cp=0.762, w_τx=0.908, w_τy=1.052, w_τz=1.451, w_vol=0.827 (α=0.5 + Y-sym routes to τz, Y-sym relieves volume pressure); rebase clean (zero diff, 422da712); EP15 in progress; projecting EP20-EP25 in 6.6-6.7% |
| #794 | `em7eupj5` | dl24-fern | EP25 (best) | **6.7064% ← WAVE VAL LEADER** | GradNorm α=0.25 + Y-axis symmetry composition; EP25=6.7064% (run best, -0.004pp from EP24=6.7104%); trajectory EP20=6.7135%→EP21=6.7236%→EP22=6.7642%→EP23=6.7811%→EP24=6.7104%→EP25=6.7064%; EP25 per-channel: cp=4.3238%, vol_p=4.0189% (run low), τx=6.6381%, τy=8.2687% (run low), τz=10.2824%; GradNorm: w_cp=0.819, w_vol_p=0.913, w_τx=0.905, w_τy=0.925, w_τz=1.438 (climbing); EP30 expected ~15:30Z; EP50≈6.25-6.40%; training to EP50 terminal |
| #800 | `hmhfnedy` | dl24-nezuko | EP12 (best/latest) | **6.7797%** | 5L STRING + GradNorm α=0.5 + Y-sym; EP12=6.7797% (new run best, -0.107pp from EP11); vol_p=4.4863% full recovery (better than pre-spike EP9=4.6846%); all 5 channels at simultaneous run-bests; w_τz=1.4903 (easing); EP20 gate pre-cleared; projections EP20≈6.30-6.40%, EP50≈5.90-6.10%; EP15 expected ~15:00Z |
| #806 | `gui4ceed` | dl24-frieren | EP10 (best/latest) | **6.8007%** | 5L STRING + GradNorm α=0.25 + Y-sym (triple compose); trajectory EP1=11.1953%→EP5=7.0526%→EP7=6.8994%→EP10=6.8007% (−0.099pp EP7→EP10); EP10 per-channel: cp=4.4674%, vol_p=4.2452%, τx=6.5864%, τy=8.3287%, τz=10.3757%; GradNorm: w_τz=1.2496 (climbing EP1=0.98→EP5=1.15→EP7=1.29→EP10=1.25); cross-run: frieren triple (6.8007%) < fern pairwise (6.8631%) < nezuko 5L (6.8733%); EP15 expected ~15:11Z, EP20 ~17:31Z, EP50 ETA ~07:30Z May 8; projections: EP25≈6.4-6.5%, EP50≈6.0-6.3% |

_Last updated: 2026-05-07T13:45Z (nezuko #800 EP12=6.7797% new run-best — vol_p noise diagnosis closed, all channels at run-bests, slope -0.107pp/epoch re-accelerated, EP20 gate pre-cleared, EP50 projection ~5.90-6.10%; fern #794 EP25=6.7064% still wave val leader; tanjiro #780 EP40 noise floor ~6.770%, terminal ~17:40Z; frieren #806 EP10=6.8007%, EP15 expected ~15:11Z; all advisor responses current)_
