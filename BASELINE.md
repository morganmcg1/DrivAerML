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

In-wave val target: **6.4667%** (PR #874 dl24-nezuko, EP14/step 78742, run `rm6u10ro`) — **nezuko #874 NEW WAVE VAL LEADER, BELOW merge threshold 6.5985% by 0.1318pp** as of 2026-05-09 ~05:45 UTC live query. 6L STRING + GradNorm α=0.75 + Y-sym p=0.5; vol_p=4.3041%, surf_p=4.2100%, wsh=7.1772%, wsh_z=9.7667%; slope=-0.01299%/1k steps (still declining). **Run ACTIVE** — EP20 checkpoint review due ~07:55 UTC. Former wave leader frieren #844 run `7dqsxvbq`: EP35=6.5290% run best, EP41 terminal (24h timeout); **UNAUTHORIZED CONTINUATION** launched by frieren at 03:52 UTC (xattn-detach-kv-r21 group, 8 DDP8 runs); escalation comment posted 04:35Z requiring immediate stop and eval-only pass before merge. fern #881 run `k59gu9o5` EP11=6.6581% (EP9=6.6786% run best); vol_p=4.0866% (run best monotonic), GradNorm equilibrium achieved; EP15 gate ≤6.65% approaching by 0.008pp. tanjiro #873 active run `2oweovb3` EP10 gate CLEARED (EP8=6.6593%, slope=-0.07120%/1k steps); 7L depth axis progression active.

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
| #818 | `dy2z6o4a` | dl24-tanjiro | EP~34.2 (best) / EP~40.6 (latest) | **6.6052%** | 6-octave STRING PE + GradNorm α=0.5 + Y-sym; val history: EP0.5=11.021%, EP1=8.120%, EP2=7.239%, EP3=7.093%, EP4=6.834%, EP5=6.818%, EP6=6.867%, EP7=6.851%, EP8=6.744%, EP9=6.685%, EP10=6.653%, EP11=6.693%, EP12=6.647%, EP13=6.609%, EP14=6.6053%→6.6143%, EP28/step~159324=6.6053%, **EP~34.2/step186795=6.6052% (RUN BEST)**, EP~40.2/step219759=6.6159% (latest); **ASCENDING** past valley EP34; current step=222185 (~EP40.6); will NOT beat 6.5985% threshold |
| #831 | `pnrgixj1` | dl24-fern | EP29 (best) / EP38 (latest) | **6.5477% (EP29 best) / 6.5866% (EP38 latest) — ACTIVE TAIL DIVERGENCE** | 6L STRING + GradNorm α=0.5 + Y-sym; val history: EP5=6.993%, EP5.5=6.702%, EP6=6.840%, EP6.50=6.6164%, EP7=6.6245%, EP13=6.6164%, EP15=6.6333%, EP17=6.6378%, EP~23.1=6.5644%, **EP29=6.5477% (RUN BEST)**, EP32=6.5772%, EP34.4=6.5555%, EP37=6.5726%, **EP38=6.5866% (LATEST — 39bp ABOVE EP29 BEST)**; step>200000; vol_p=4.1086% stable; **ACTIVE COSINE TAIL DIVERGENCE** — EP38 worse than EP29 by 39bp, ascending trend EP29→EP38; EP40 hard decision point (early termination if no new best); frieren #844 has surpassed fern as wave leader |
| #843 | `hyzdxrj2` | dl24-nezuko | EP27.6 (best EP26~) / EP27.6 (latest) | **6.6554%** | 7-octave STRING PE (`pe_init_sigmas=[0.25,0.5,1.0,2.0,4.0,8.0,16.0]`) + GradNorm α=0.5 + Y-sym; val history: EP1=10.8989%, EP2=7.9426%, EP3=7.4226%, EP4=7.4031%, EP5=7.1003% (≤7.5% gate CLEARED), EP6=7.0719% (≤7.2% gate CLEARED EARLY), EP~14.1=6.7644%, EP20=6.6994%, EP25=6.6588%, **EP27.6 latest best=6.6554%** — vol_p=4.4696% (oscillating 4.45–4.95% range since EP19; σ=16 destabilization hypothesis); EP30 KILL GATE (~2.4ep remaining): (1) vol_p < 4.30% OR primary < 6.6005% required to continue; currently 0.108pp above fern wave leader; step=151420 |
| #838 | `84skr4yq` | dl24-fern | EP4.1 (tay terminal) | 7.4255% | STRING RFF σ=0.125, 4-epoch tay screen; EP4.1=7.4255% (all tay gates passed); **CLOSED** — merge gate FAILED (tay EP4≤6.5985% required); promoted to long-run but test=8.7190% (+1.2pp regression vs SOTA); σ<0.25 axis confirmed dead at 65k surface points |
| #844 | `7dqsxvbq` | dl24-frieren | EP35 (best) / EP41 (terminal) | **6.5290% ← WAVE VAL LEADER, AWAITING TEST** | 5L STRING PE + GradNorm α=0.5 + no Y-sym; relaunched 2026-05-08T07:44Z with corrected `--train-volume-points 65000 --train-surface-points 40000`; EP1=10.8430%, EP5=6.95%, EP10=6.68%, EP20=6.5644%, EP27=6.5379%, EP33=6.5294%, **EP35 confirmed run best=6.5290%, vol_p=4.398%**; EP36-EP41 plateau 6.5385%-6.5599% (typical loss-basin oscillation); run state=**FINISHED at EP41/step 225,253** (24h timeout interrupted before final test eval); **NO test_primary metrics logged on `7dqsxvbq` — PR sent back 2026-05-09 for `--eval-only --eval-checkpoint outputs/drivaerml/run-7dqsxvbq/checkpoint.pt` test pass before merge eligibility** |
| #866 | `gb73kgzz` | dl24-tanjiro | EP4 (terminal) | CLOSED | 6L STRING + GradNorm a=0.5 + Y-sym p=1.0 (full symmetry); EP4 gate DEFINITIVELY FAILED: EP4 best still ~13-12% range, projected EP5~8.2% at max recovery rate — gate requires <=7.5%; severe deceleration vs p=0.5 baseline; Y-sym p=1.0 (every batch flipped) falsified as over-augmentation; p=0.5 confirmed optimal (PR #741). **PR CLOSED 2026-05-05.** |
| #873 | `2oweovb3` (active) / `59bcgz40` (killed) | dl24-tanjiro | EP6 (latest) / running | 6.8467% (EP6) | 7L STRING + GradNorm a=0.5 + Y-sym p=0.5; depth axis 4L → 5L (#844 leader) → 6L (#831) → 7L (THIS). Original run `59bcgz40` killed at EP5=6.9789% by inverted kill-threshold operator student bug (`>=7.5` instead of `<7.5`). Relaunched 2026-05-08T23:39Z as `2oweovb3` with corrected operator; EP1=10.5138%, EP2=8.2038%, EP3=7.3671%, EP4=6.9672%, EP5=6.8718% (≤7.5% gate CLEARED), **EP6=6.8467%** (latest), vol_p=4.4760%, w_vol_p=0.6079. Currently mid-EP7 (step ~34,611), EP10 gate (≤7.2%) pending. |
| #874 | `rm6u10ro` | dl24-nezuko | EP14 (latest, RUN BEST) / running | **6.4667% (EP14 NEW BEST)** | 6L STRING + GradNorm α=0.75 + Y-sym p=0.5; α-axis upper half. EP1=10.7455%, EP5=6.8164% (gate cleared), EP7=6.6892%, EP10=6.5901%, EP11=6.5686%, EP12=6.6217% (single-epoch bounce), EP13=6.5381%, **EP14=6.4667% NEW RUN BEST** ⭐ (step=78742), vol_p=4.3041%, surf_p=4.2100%, wsh=7.1772%, wsh_z=9.7667%. **Below in-wave merge threshold 6.5985%** — new wave val leader. α=0.75 6L converging faster than α=0.5 5L frieren #844 (which required EP35). w_vol_p healthy, slope=-0.01299%/1k steps (still declining). Continue training. |
| #881 | `k59gu9o5` | dl24-fern | EP10 (latest, EP9 best) / running | **6.6786% (EP9 best) / 6.6867% (EP10 latest)** | 5L STRING PE + GradNorm α=0.5 + vol-mlp-head (deeper MLP volume decoder, no Y-sym). EP1=10.7116%, EP2=7.7931%, EP4=7.0796%, EP5=7.3212% (transient — recovered), EP6=6.7679%, EP7=6.7105%, EP8=6.6957%, **EP9=6.6786% (run best)**, EP10=6.6867% (gate PASS, +0.51pp margin under 7.20%); vol_p=4.1088% (run best, monotonic descent EP6-EP10), surf_p=4.3885%, wsh_z=10.10%; **w_vol_p settled into target [0.80-1.20] band at ~1.14** (was 1.484 at EP7); r_vol_p~0.45 (well below 0.65 stability threshold). MLP head's early dominance resolved — GradNorm equilibrium reached. Continue to EP15/EP20/EP25/EP30 milestones; EP30 hard gate ≤6.6005% (in-wave merge threshold). |
| #842 | `3487klz8` | thorfinn (tay) | EP4 (terminal) | 7.610% | tay 4-epoch screen; **CLOSED 2026-05-08T09:03Z** — EP4=7.610%, gate ≤6.5985% FAILED by 1.011pp |
| #840 | `oiptel6p` | tanjiro-tay (tay) | EP4 (terminal) | 7.8558% | tay 4-epoch screen; **CLOSED 2026-05-08T09:03Z** — EP4=7.8558%, gate ≤6.5985% FAILED by 1.256pp |
| #823 | `ghh0s4ne` | nezuko-tay (tay) | EP1 (in progress, relaunched) | — | tay 4-epoch screen with --use-surf-to-vol-xattn; RELAUNCHED 2026-05-08T08:33Z (prior run `lp7u9r8g` killed by advisor error); EP1 in progress |
| #847 | `7gzie3gj` | dl24-frieren | EP1 (in progress) | — | LR warmup 2ep variant; EP1/step10987=26.3602% (gate <30% passed 2026-05-08T09:04Z); EP2 running with 2-epoch warmup |
| #848 | TBD | alphonse (tay) | — | — | tay 4-epoch screen: Lion lr=8e-5 vs 7e-5 downsweep; sequential plan approved by advisor; Arm A pending launch |
| #845 | `thimjhnd` (rank0) | dl24-fern | EP~1 (running) | — | RFF num_features=24, rff_init_sigmas=0.25,0.5,1.0,2.0,4.0 (sota-spectrum); group=rff-capacity-sweep; 8-GPU DDP; launched 2026-05-09T07:32Z |
| #846 | `px719275` | dl24-edward | EP~1 (running) | — | RFF num_features=32, rff_init_sigmas=0.25,0.5,1.0,2.0,4.0 (sota-spectrum); group=rff-capacity-sweep; launched 2026-05-09T07:33Z; ETA ~12:35Z |
| #849 | TBD | askeladd (tay) | — | — | tay 4-epoch screen: τy=2.0/2.5 vs τz=1.5 differential; Arm A running; student acknowledged EP2 gate correction (<16%) |

_Last updated: 2026-05-09 ~04:30 UTC (live data refresh — nezuko EP13 NEW BEST, fern EP10 gate PASS, tanjiro EP6, frieren eval-only pending). Major events: (1) **nezuko #874 (`rm6u10ro`) EP13=6.5381% NEW RUN BEST** ⭐ — within 0.0091pp of frieren wave leader at EP13 vs EP35; α=0.75 6L converging faster than α=0.5 5L; every channel run-best; w_vol_p recovered to 0.7029; continuing to EP20 ETA ~07:55 UTC; (2) **fern #881 (`k59gu9o5`) EP10 gate PASS=6.6867% (EP9 best=6.6786%)** — w_vol_p settled into target [0.80-1.20] band (~1.14, was 1.484 at EP7); GradNorm equilibrium achieved; vol_p=4.1088% monotonic; continuing to EP30 milestones; (3) **tanjiro #873 (`2oweovb3`) EP6=6.8467%** — descending healthily, mid-EP7, EP10 gate (≤7.2%) on track; (4) **frieren #844 (`7dqsxvbq`) PR sent back for eval-only test pass** at 03:37Z — no student response yet; required for merge eligibility (no test_primary metrics on terminal run); (5) **wave leader board: frieren EP35=6.5290% val (TERMINAL, awaiting test) > nezuko EP13=6.5381% (climbing) > fern EP9=6.6786% > tanjiro EP6=6.8467%**._
