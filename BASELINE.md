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

### In-wave merged single-model results (OLD DATASET — rawcanon pre-20260511, ARTIFACT)

> **IMPORTANT (2026-05-12):** A case-split bug in the dataset was identified on 2026-05-12 (Issue #1053). The results below were produced on the **old split** and are affected by a dataset artifact that created a spurious +7–8pp val→test volume pressure gap. These results are retained for historical continuity but should NOT be used as comparison targets. See the corrected-split table below.

| PR | Run | Mechanism | Test aggregate (OLD) | Surface | Volume | Wall | tau_x / tau_y / tau_z |
|---|---|---|---:|---:|---:|---:|---:|
| #599 | `sogus8sx` | multi-sigma STRING PE (`pe_init_sigmas=[0.25,0.5,1.0,2.0,4.0]`) | 7.9303 | — | — | — | — |
| #741 | `lszc4ri7` / `1tal40wr` | Y-axis symmetry augmentation (bilateral car geometry) | 7.8232 | 3.9821 | 11.3345 | 7.3076 | 6.5304 / 7.9248 / 9.3444 |
| #740 | `5x8wofzm` | GradNorm α=0.5 adaptive loss balancing | 7.5195 | 3.8810 | 10.7580 | 7.0610 | 6.3490 / 7.5600 / 9.0480 |

> Under the corrected split, PR #740 (`5x8wofzm`) re-evaluates to test_ABUPT=**8.165%** (rank 22) and PR #741 to 8.156% (rank 21). The 7.52% result was an artifact of the old split.

---

### Corrected-split results (rawcanon_20260511, 2026-05-12) — AUTHORITATIVE

Dataset: `/mnt/pvc/Processed/drivaerml_processed_rawcanon_20260511`  
Eval params: `eval_surface_points=65536`, `eval_volume_points=65536`  
Split: 34 val cases / 7,295 views; 50 test cases / 11,091 views  
Source: Issue #1053 revalidation run 2026-05-12 (all PRs re-evaluated on corrected split)

| Rank | PR | Source run | Eval run | test_ABUPT | test_VP | test_SP | test_WSS |
|------|-----|-----------|---------|------------|---------|---------|---------|
| 1 ⭐ | **#1284 (MERGED)** | `yym5oa8x` | terminal | 5.801% | 3.603% | 3.650% | **6.6506%** ⭐ NEW WSS SOTA |
| 2 | **#972** | `56bcqp3m` | `zxnhtagj` | **5.844%** | 3.643% | 3.577% | 6.727% |
| 3 | #968 | `a0yoxy85` | `qbg9pkmx` | 5.986% | 3.957% | 3.673% | 6.825% |
| 4 | #880 | `zst3y2mp` | `x78xbsfn` | 6.010% | 4.501% | 3.611% | 6.708% |
| 5 | #958 | `29nohj67` | `fkjc12c8` | 6.107% | 3.818% | 3.911% | 6.985% |
| 6 | #939 | `yfitnqia` | — | 6.242% | — | — | — |
| … | … | … | … | … | … | … | … |
| 21 | #741 | `lszc4ri7`/`1tal40wr` | — | 8.156% | 11.744% | — | — |
| 22 | #740 | `5x8wofzm` | — | 8.165% | 13.660% | — | — |

> **Note on PRs #972, #968, #880, #958:** These PRs were CLOSED as FALSIFIED under the old dataset due to the spurious +7–8pp val→test vol_p gap. Under the corrected split, they are the **top 4 performers** in the entire wave. They were NOT merged (already closed), so their techniques should be used as the foundation for new experiments on the corrected dataset.

### Current single-model best on `drivaerml-long-20260504`

**MERGED dl24 SOTA (2026-05-24 10:57Z):** PR #1284 (`yym5oa8x`) — H39 wider surface_out MLP (factor=2.0) — **FIRST in-wave merge on `drivaerml-long-20260504`**  
test_primary/wall_shear_rel_l2_pct = **6.6506%** ⭐ (beats prior SOTA #972 6.727% by −0.0764pp — NEW dl24 single-model WSS SOTA)  
test_primary/volume_pressure_rel_l2_pct = **3.6033%** (clears 3.643% floor) ✓  
test_primary/surface_pressure_rel_l2_pct = **3.6498%** (misses 3.577% floor by +0.0728pp) ⚠️  
test_primary/abupt_axis_mean_rel_l2_pct = **5.8010%** (clears 5.844% floor; also beats prior abupt SOTA #972 5.844% by −0.043pp)  
Per-axis WSS: wss_x=5.9031%, wss_y=7.1901%, wss_z=8.6587%  
Best checkpoint: EP24 EMA (selection metric: val_primary/abupt_axis_mean_rel_l2_pct); runtime 73037s within 1300min budget; ended at step 305904 / EP27.87  

**PR #972 (closed, corrected-split re-eval)** remains the 3-of-4 floor reference; H39 BEATS the WSS objective (Issue #1056 primary) but doesn't clear the surface_p floor for the full 4-floor contract.

Per Issue #1056: **test_WSS is the PRIMARY objective**, hard constraints test_VP ≤ 3.643, test_SP ≤ 3.577, test_abupt ≤ 5.844. H39 clears 3-of-4. Next assignments target compound mechanisms (H39 + per-axis Charb, deeper trunk, etc.) that hold WSS lead AND close the SP floor.

All new experiments must be evaluated on the corrected dataset and target beating **test_WSS=6.6506%** while maintaining the 3-of-4 floor clearance and ideally closing the SP gap.

### In-wave val target (to beat before merging)

In-wave val leader (historical, CLOSED): **6.4402%** (PR #874 dl24-nezuko, EP15/step 82,409, run `rm6u10ro`). **PR #874 CLOSED 2026-05-09**: eval-only run `zy07iley` completed 06:07Z; test_primary/abupt=**7.7116%** — ABOVE wave SOTA 7.5195% by +0.1921pp; does NOT beat SOTA; no merge path. Same vol_p val→test overfitting pattern (+7.2pp gap) as PR#844 and PR#800. Frieren #844 `7dqsxvbq`: CLOSED 2026-05-09; test=7.7804%; does NOT beat SOTA 7.5195%. Vol_p val→test anomaly confirmed as systematic overfitting across all runs (val≈4%, test≈11%, +7pp gap). To beat SOTA, a run must achieve test_primary/abupt ≤7.5195% — val leadership alone is insufficient. **Active in-wave val leader: tanjiro #900 `os6v64lq` EP10=6.7402% (WD=0.01 regularization hypothesis; continuing to EP15+).**

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
| #844 | `7dqsxvbq` | dl24-frieren | EP35 (best) / EP41 (terminal) | **6.5290% (val best); test=7.7804% — CLOSED** | 5L STRING PE + GradNorm α=0.5 + no Y-sym; relaunched 2026-05-08T07:44Z; EP35 confirmed run best=6.5290%; EP41 terminal; **test_primary=7.7804%** confirmed from training run `7dqsxvbq` (NOT from eval-only run `25tt8r47` which was redundant). Val→test gap +1.25pp. Vol_p val→test anomaly: val=4.063% vs test=11.164% (+7.1pp) — volume pressure head overfitting to val distribution. Does NOT beat wave SOTA 7.5195%. **PR CLOSED 2026-05-09** |
| #866 | `gb73kgzz` | dl24-tanjiro | EP4 (terminal) | CLOSED | 6L STRING + GradNorm a=0.5 + Y-sym p=1.0 (full symmetry); EP4 gate DEFINITIVELY FAILED: EP4 best still ~13-12% range, projected EP5~8.2% at max recovery rate — gate requires <=7.5%; severe deceleration vs p=0.5 baseline; Y-sym p=1.0 (every batch flipped) falsified as over-augmentation; p=0.5 confirmed optimal (PR #741). **PR CLOSED 2026-05-05.** |
| #873 | `2oweovb3` / `59bcgz40` (killed) | dl24-tanjiro | EP15 (terminal gated) | **6.5874% (EP12, run best) → CLOSED** | 7L STRING + GradNorm α=0.5 + Y-sym p=0.5; depth axis explored 4L→5L→6L→7L. EP15=6.8505% — FAILED gate ≤6.55% by 0.3005pp; catastrophic bounce confirmed (EP12=6.5874% → EP15=6.8505%); 7L depth hypothesis REJECTED; 6L confirmed as depth ceiling. **PR #873 CLOSED 2026-05-09.** |
| #874 | `rm6u10ro` / eval-only `zy07iley` | dl24-nezuko | EP15 (best) / EP16 (BLOWUP — FINISHED) | **6.4402% (EP15/step 82,409, RUN BEST ⭐)** | 6L STRING + GradNorm α=0.75 + Y-sym p=0.5; α-axis upper half. EP1=10.7455%, EP5=6.8164%, EP7=6.6892%, EP10=6.5901%, EP11=6.5686%, EP12=6.6217% (bounce), EP13=6.5381%, EP14=6.4667%, **EP15=6.4402% (RUN BEST, step 82,409)** ⭐, EP16=9.7575% (CATASTROPHIC BLOWUP); **run state=FINISHED**. **EVAL-ONLY COMPLETE** (run `zy07iley`, finished 2026-05-09T06:07Z): **test_primary/abupt=7.7116%** — ABOVE wave SOTA 7.5195% by +0.1921pp; full_val_primary/abupt=6.4402% confirmed. Test breakdown: surf_p=3.9449%, vol_p=11.3563%, wsh=7.1280%, τx=6.3557%, τy=7.6844%, τz=9.2164%. Vol_p val→test gap: val≈4.13% vs test=11.36% (+7.2pp) — same overfitting pattern as #844. α=0.75 GradNorm instability mechanism confirmed. **DOES NOT BEAT WAVE SOTA 7.5195%. PR #874 CLOSED.** Wave val leader record only (6.4402% EP15). |
| #881 | `k59gu9o5` | dl24-fern | EP22 (last) / **KILLED** | **6.6055% (EP20/step 109,879, run best)** | 5L STRING PE + GradNorm α=0.5 + vol-mlp-head (deeper MLP volume decoder, no Y-sym). Full val history: EP1=10.7116%, ..., EP16=6.6106%, EP17=6.6163%, EP18=6.6193%, EP19=6.6108%, **EP20=6.6055% (RUN BEST)**, EP21=6.6066%, EP22=6.6076% (UPWARD DRIFT). EP20 gate ≤6.60% CLEARED; EP25 gate ≤6.58% UNREACHABLE (ascending trend EP20→EP22); vol-mlp-head hypothesis REJECTED. **PR #881 KILLED at EP22 — upward drift confirmed; dl24-fern now IDLE.** |
| #883 | `3jxymrmm` | dl24-tanjiro | EP1 FIRED ✓ / EP2 running | **28.7169% (EP1.98/step 21,436, first snapshot)** | xattn-rff-pos: STRING sep token + RFF-positional encoding in cross-attention. Run confirmed as `tanjiro/xattn-rff-pos-13ep-rank0`; agent=tanjiro (not fern). Screen-run: SPE=10,864 steps/epoch. EP1.98/step=21,436=28.7169% — **EP1 gate (≤30%) PASS ✓**. `model/string_sep_init_sigmas: []` (empty sigma list in W&B config) appears to be config-logging artifact; run is executing normally. step=12,512 (EP=1.15); EP2 gate (≤16%) at step 21,729 (~9,217 steps from step 12,512). ACTIVE, monitoring EP2. |
| #888 | `dz1pjkhw` | dl24-tanjiro | EP2 (best) / EP2.80 mid (latest) / running | **8.4952% (EP2/step 21,729, run best)** | OOD-weighting experiment. EP1=30.2033% (borderline FAIL >30% by 0.2033pp — run continues due to strong EP2 recovery); **EP2=8.4952% (gate ≤16% CLEARED ✓)**; mid-epoch EP2.80/step=30,452=7.4232% (approaching EP3 boundary); step=30,453 (EP=2.80). vol_p=5.254% at EP2. **ACTIVE, monitoring EP3 gate (≤8.0%)** (step 32,592 — ~2,139 steps away). EP3 gate LIKELY TO CLEAR: mid-epoch EP2.80=7.4232% already BELOW ≤8.0% threshold by 0.5768pp; strong descent trajectory. |
| #890 | `81x541h2` | dl24-nezuko | EP2.08 (running — EP2 gate anomaly RESOLVED) | **7.5868% (EP2.08/step 22,647, latest)** | xattn-detach-kv: detach K/V in cross-attention. **EP1=29.9332% (gate ≤30% PASS ✓)**; EP1.50=11.3480%, EP1.83=8.2934%, **EP2.08/step=22,647=7.5868% (gate ≤16% PASS ✓)**; step=22,648 (EP=2.08). EP2 snapshot logged 918 steps after exact boundary (21,729) — timing anomaly resolved; value well below gate threshold. **EP3 gate (≤8.0%) next** at step 32,592 (~9,944 steps from step 22,648). Strong EP1.83→EP2.08 descent: 8.29%→7.59%; EP3 gate likely to clear. ACTIVE, monitoring EP3. |
| #891 | `c3jvc0s1` | dl24-frieren | EP1.50 (running) | **26.5136% (EP1/step 10,864, only snapshot)** | post-xattn-ffn: add FFN after cross-attention block. **EP1=26.5136% (gate ≤30% PASS ✓)**; step=16,299 (EP=1.50). vol_p=15.7122% at EP1. **EP2 gate (≤16%) next** (step 21,729 — ~5,430 steps away). **ACTIVE**. |
| #892 | `wu634y5m` | dl24-edward | **FAILED — ALL 8 RANKS CRASHED** | — | xattn-mid-backbone: inject cross-attention in middle of backbone. All 8 ranks failed at step=10,863 (just before EP1 boundary); no val snapshots produced; no restart. **PR #892 CLOSED — experiment failed at EP1 boundary.** |
| #893 | `7jqz957i` | dl24-alphonse | EP<1 (running — EP1 IMMINENT) | — | GQA-xattn Arm A: grouped-query attention in cross-attention. step=~10,401 (EP=0.96), no val snapshots yet. **EP1 gate (≤30%) pending** (~463 steps to step 10,864 — IMMINENT). Arm B (n_kv_heads=2) not yet launched. **ACTIVE — EP1 ANY MINUTE**. |
| TBD | `x3c2a2jt` | dl24-edward | EP<1 (running) | — | xattn-depth-L6-512: 6L backbone with hidden_dim=512 expansion. Launched 2026-05-09T07:50Z; step=3,914 (EP=0.36); no val snapshots yet. **EP1 gate (≤30%) pending** (~6,950 steps to step 10,864). **ACTIVE**. |
| TBD | `v85mklr7` | dl24-askeladd | EP<1 (running) | — | learned-surf-pool-xattn-256: learned surface pooling + cross-attention with dim=256. Launched 2026-05-09T07:43Z; step=6,293 (EP=0.58); no val snapshots yet. **EP1 gate (≤30%) pending** (~4,571 steps to step 10,864). **ACTIVE**. |
| #897 | `5g9i6s7p` | dl24-frieren | EP3 (killed) | — | **mlp-ratio=2.0** — doubled MLP ratio in Transolver backbone (2.0 vs default 1.0). EP1=15.5097%, EP2=11.706%, EP3=11.1813% vs gate ≤8.0% (MISS by 3.18pp). Insufficient convergence from wider MLP. **PR #897 KILLED at EP3 — hypothesis REJECTED; dl24-frieren IDLE.** |
| #895 | `x3c2a2jt` | dl24-edward | EP3 (narrow FAIL) / EP4 (IMMINENT) | **8.3269% (EP3/step 19,926)** | xattn-depth-L6-512: 6L backbone + cross-attention with hidden_dim=512 expansion. EP1=27.1502% PASS (≤30%); EP2=11.5452% PASS (≤16%); EP3=8.3269% NARROW FAIL vs ≤8.0% gate (miss 0.33pp) — **EXTENDED to EP4** (≤6.4407%, SOTA). EP4 boundary at step ~21,728; current step=20,610 (~1,118 steps away). ACTIVE, monitoring EP4. |
| #896 | `vf9dprlh` | dl24-frieren (xattn) | EP2 PASS / EP3 approaching | **8.9539% (step 39,851, non-epoch snapshot)** | xattn KV gradient scaling α=0.25. Epoch size=21,728 (batch_size=2). EP1=13.4487% PASS (≤30%); EP2=9.3948% PASS (≤16%); last snapshot 8.9539% at step 39,851 (non-epoch midpoint). EP3 boundary at ~43,456 (~1,941 steps from step 41,515). Gate EP3≤8.0% — needs ~0.95pp drop. ACTIVE, monitoring EP3. |
| #898 | `ylrp8f97` | dl24-frieren | EP5 PASS | **7.2169% (EP5/step 27,469)** | **5L STRING (5-oct) + GradNorm α=0.5 + Y-sym p=0.5 + vol-points=65k** — first test of validated triple stack together. Gate schedule: EP5≤7.5%✓, EP10≤7.2%, EP15≤6.80%, EP20≤6.70%, EP25≤6.65%, EP30≤6.60%, EP35≤6.58%, EP40≤6.55%. Full val history: EP1=11.0024% PASS, EP2=8.0667% PASS, EP3=7.5575% PASS, EP4=7.3231% PASS, **EP5=7.2169% PASS (gate ≤7.5% ✓)**. EP5 cleared by 0.28pp. Next gate EP10 (step ~54,938) ≤7.2%. ACTIVE. |
| #899 | `mbi0cac0` | dl24-nezuko | EP5 FAIL / **KILLED** | **7.3501% (EP5/step 27,469)** | 5L STRING + GradNorm α=0.5 + dropout=0.1 + Y-sym p=0.5 + vol-points=65k. Val history: EP1=11.7844% PASS, EP2=8.5153% PASS, EP3=7.8459% PASS, EP4=7.5819% FAIL (extended), **EP5=7.3501% FAIL vs extended gate ≤7.2% (miss 0.150pp)**. Dropout=0.1 slows convergence without offsetting benefit vs #898 (no dropout, 0.133pp better at EP5). Structural approaches needed for vol_p overfitting. **PR #899 KILLED at EP5 — dropout hypothesis REJECTED; dl24-nezuko IDLE.** |
| #900 | `os6v64lq`/`wps85twd`/`2xpnbg06` | dl24-tanjiro | EP27 (run best) / TERMINAL — **CLOSED** | **6.5911% (val_abupt terminal) / test_abupt=7.8809% / test_vol_p=11.8441%** | **6L STRING (5-oct) + GradNorm α=0.5 + weight-decay=0.01 + Y-sym p=0.5 + vol-points=65k**. Val trajectory: EP19=6.6545%, EP20=6.6353%, EP21=6.6325%, EP22=6.6208%, EP23=6.6024% (new best), EP24=6.6085%, EP25=6.6005% (new best), EP26=6.6215%, **EP27=6.5911% (RUN BEST ⭐)**. Terminal test: val_abupt=6.5911%, **test_abupt=7.8809%** (+1.29pp vs val), **test_vol_p=11.8441%** (+7.4pp val→test gap — same structural overfitting pattern). WD=0.01 REJECTED: compressed val vol_p to ~4.0% but test vol_p still 11.84% — gap is structural, not weight-magnitude driven. **PR #900 CLOSED 2026-05-10. DOES NOT BEAT WAVE SOTA 7.5195%.** |
| #911 | `8co57khm` / `pwmc52v7` | dl24-fern | EP3 FAIL — **CLOSED** | **10.0038% (Arm B EP3 best — FAIL)** | **5L STRING + GradNorm α=0.5 + volume-loss-weight=2.0 + Y-sym p=0.5 + vol-points=65k**. Arm A (`8co57khm`) KILLED at EP3=11.5646%; Arm B (`pwmc52v7`, bug-fixed) KILLED at EP3=10.0038%. Both failed EP3 gate ≤8.0%. Cancellation hypothesis CONFIRMED: static vol_loss×2 + GradNorm is self-cancelling; GradNorm drives equilibrium back toward original weights, transient adds gradient noise. EP3 vol_p Arm A=7.7451%, Arm B=7.7175% vs SOTA 5.1329% (+2.6pp WORSE on vol_p). Bug fix commit `7904c58` valid and must be cherry-picked to main. **PR #911 CLOSED 2026-05-09.** |
| #912 | `91tmhv7w` | dl24-nezuko | EP3 FAIL — **CLOSED** | **11.8122% (EP3 — SEVERE FAIL)** | **5L STRING + GradNorm α=0.5 + 96k vol-points (65k→96k) + Y-sym p=0.5**. Prior runs `irq1xl5a` (vol_p task_loss=0 bug), `q2mf0exo` (operator killed). Bug fixed: monkey-patch `DrivAerMLSurfaceDataset._indices`. EP1=18.0022% PASS, EP2=13.0275% PASS, **EP3=11.8122% SEVERE FAIL** vs gate ≤8.0% (miss +3.81pp). EP3 metrics: vol_p=4.7841% (−0.35pp vs 65k baseline ✓), surf_p=8.5218% (+3.65pp WORSE), wall=13.8653% (+5.53pp WORSE). **Root cause: vol/surface imbalance** (96k vol vs 40k surface) — GradNorm cannot equilibrate; surf_p/wall gradient starvation overwhelms vol_p benefit. vol_p advantage is real and widening (EP1: +0.09pp, EP2: −0.25pp, EP3: −0.35pp) but insufficient alone. **Future direction: scale surface pts proportionally (96k vol + 60k surface) to maintain 1.6:1 ratio**. Bug fix (DrivAerMLSurfaceDataset._indices monkey-patch) must be cherry-picked to main. **PR #912 CLOSED 2026-05-09.** |
| #936 | `unknown` | dl24-nezuko | EP5 (terminal) — **CLOSED/FALSIFIED** | **9.010% (EP5 — FAIL vs gate ≤7.5%)** | **5L STRING + GradNorm α=0.5 + vol-loss-weight=2.0 (no static cancellation fix) + WD=0.005 + Y-sym p=0.5 + vol-points=65k**. Static vol_loss upweighting=2.0 without disabling GradNorm. Val trajectory: EP1=17.878%, EP2=11.858% (vol_p=6.805%), EP3=10.439%, EP4=9.765%, EP5=9.010% (vol_p=5.691%). EP5 FAIL: 9.010% vs gate ≤7.5% (+1.51pp miss). Convergence too slow — static upweighting without GradNorm adjustment cannot converge at learning rate required by gates. **Hypothesis FALSIFIED: static vol_loss_weight=2.0 alone without GradNorm modification is insufficient. PR #936 CLOSED 2026-05-10.** |
| #938 | `md3vhhd8` (bug) / `8u8r1e6i` (fix, ORPHANED) | dl24-fern | EP5 pending (orphaned) | **CLOSED (lr bug) / orphaned run `8u8r1e6i` running** | **5L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym p=0.5 + 96k vol + 60k surf + vol-points=96k**. Proportional point sampling to maintain 1.6:1 ratio (addresses PR#912 surf_p gradient starvation). First run `md3vhhd8` used lr=3e-4 (BUG — should be 1e-4), causing EP5=15.22% FAIL. PR closed by advisor 03:01:20Z. Fern relaunched as `8u8r1e6i` with lr=1e-4 fix 35 seconds later (03:01:55Z) — run ORPHANED under closed PR. `8u8r1e6i` EP5 result not yet known at PR closure. **PR #938 CLOSED 2026-05-10 (lr bug). Orphaned run `8u8r1e6i` running; needs new tracking PR.** |
| #939 | `yfitnqia` | dl24-nezuko | EP20 PASS / EP25 gate next | **6.5829% (EP20 PASS ≤6.70% ✓ — plateau 6.58–6.61%)** | **6L STRING (5-oct) + GradNorm α=0.5 + WD=0.005 + Y-sym p=0.5 + vol-points=65k**. WD=0.005 on 6L backbone. Full trajectory: EP1=10.6323%, EP5=6.9188% PASS ✓, EP10 PASS, EP11=6.6159%, EP12=6.5989%, EP13=6.6006%, EP14=6.5973%, EP15=6.6082%, EP16=6.5971% (run best so far), EP17=6.5925%, EP18=6.5953%, EP19=6.5794%, **EP20=6.5829% PASS (gate ≤6.70% ✓)**. Plateau 6.58–6.61% range since EP11. EP25 gate (≤6.65%) next — EP16=6.5971% already below, likely PASS. EP30 gate (≤6.60%) very tight — only 0.0029pp headroom from EP16 best. **ACTIVE — monitoring EP25 gate.** |
| #914 | `wdxtdmhy`/`qjmyzmby` | dl24-frieren | EP30 (terminal) — **CLOSED** | **6.5771% (val_abupt terminal) / test_abupt=7.9318% — REJECTED** | **5L STRING (5-oct) + GradNorm α=0.5 + weight-decay=0.005 + Y-sym p=0.5 + vol-points=65k**. WD=0.005 vs codebase default WD=1e-4 (50x increase). Terminal test: val_abupt=6.5771%, **test_abupt=7.9318%** (+1.355pp vs val), **test_vol_p=11.8961%** (+7.724pp val→test gap — WIDER than SOTA's gap), test_surf_p=4.0476%, test_wall=7.2446%. WD=0.005 REJECTED: compressed val vol_p to 4.1722% but test vol_p still 11.90% — gap is structural, not weight-magnitude driven. DOES NOT BEAT wave SOTA 7.5195% (+0.412pp WORSE). **PR #914 CLOSED 2026-05-10.** |
| #901 | `f9gz7i0o` | dl24-mirroraug | EP1 PASS / EP2 approaching | **29.5643% (EP1/step 10,864)** | Y-axis mirror augmentation variant (PR#901_mirroraug). EP1=29.5643% PASS (gate ≤35%). EP2 boundary at step ~21,729; current step=13,522 (~8,207 steps away). Gate EP2≤16%. ACTIVE. |
| #893B | `eqp1873z` | dl24-alphonse | EP<1 (running) | — | GQA Arm B: n_kv_heads=2 in cross-attention. 10,864-step epoch size. No val snapshots yet; step=7,986 (~2,878 steps to EP1 boundary). EP1 gate ≤30% pending. ACTIVE. |
| #842 | `3487klz8` | thorfinn (tay) | EP4 (terminal) | 7.610% | tay 4-epoch screen; **CLOSED 2026-05-08T09:03Z** — EP4=7.610%, gate ≤6.5985% FAILED by 1.011pp |
| #840 | `oiptel6p` | tanjiro-tay (tay) | EP4 (terminal) | 7.8558% | tay 4-epoch screen; **CLOSED 2026-05-08T09:03Z** — EP4=7.8558%, gate ≤6.5985% FAILED by 1.256pp |
| #823 | `ghh0s4ne` | nezuko-tay (tay) | EP1 (in progress, relaunched) | — | tay 4-epoch screen with --use-surf-to-vol-xattn; RELAUNCHED 2026-05-08T08:33Z (prior run `lp7u9r8g` killed by advisor error); EP1 in progress |
| #847 | `7gzie3gj` | dl24-frieren | EP1 (in progress) | — | LR warmup 2ep variant; EP1/step10987=26.3602% (gate <30% passed 2026-05-08T09:04Z); EP2 running with 2-epoch warmup |
| #848 | TBD | alphonse (tay) | — | — | tay 4-epoch screen: Lion lr=8e-5 vs 7e-5 downsweep; sequential plan approved by advisor; Arm A pending launch |
| #845 | `thimjhnd` (rank0) | dl24-fern | EP~1 (running) | — | RFF num_features=24, rff_init_sigmas=0.25,0.5,1.0,2.0,4.0 (sota-spectrum); group=rff-capacity-sweep; 8-GPU DDP; launched 2026-05-09T07:32Z |
| #846 | `px719275` | dl24-edward | EP~1 (running) | — | RFF num_features=32, rff_init_sigmas=0.25,0.5,1.0,2.0,4.0 (sota-spectrum); group=rff-capacity-sweep; launched 2026-05-09T07:33Z; ETA ~12:35Z |
| #849 | TBD | askeladd (tay) | — | — | tay 4-epoch screen: τy=2.0/2.5 vs τz=1.5 differential; Arm A running; student acknowledged EP2 gate correction (<16%) |

| #944 | `234rrtoo` | dl24-tanjiro | EP5 FAIL — **CLOSED** | **8.7923% (EP5 FAIL vs gate ≤8.0%)** | **6L STRING + GradNorm α=0.5 + WD=0.005 + Y-sym p=0.5 + EMA decay=0.9999 + ema-start-step=500**. EMA warm-start bug (shadow initialized from random weights) + decay=0.9999 → lookback ~10k steps → 10.67% shadow contamination at EP5. Hypothesis UNTESTED (not rejected). Retested as PR #954 with decay=0.999 (lookback ~1k steps). **PR #944 CLOSED 2026-05-09. Hypothesis: UNTESTED.** |
| #945 | `6wkvxmo3` | dl24-fern | EP5 FAIL — **CLOSED** | **9.29% (EP5 FAIL vs gate ≤7.5%)** | **5L STRING (5-oct) + GradNorm α=0.5 + WD=0.005 + Y-sym p=0.5 + 96k vol + 60k surf + Lion lr=1e-4**. Proportional point sampling (1.6:1 ratio) on 5L backbone. Kill threshold operator bug (`>` instead of `<`) meant run was NOT killed at EP5=9.29% despite missing gate. 5L backbone too weak for proportional sampling convergence speed. **PR #945 CLOSED 2026-05-10 (EP5 FAIL 9.29%). Hypothesis NOT rejected — moved to 6L backbone as PR #951.** |
| #954 | `gwiucb33` / eval `wiroo9ux` | dl24-tanjiro | EP8 EMA (best) / EP10 KILLED | **6.6958% (EP8 EMA val best) / test_abupt=7.9238% — CLOSED, HYPOTHESIS REJECTED** | **6L STRING (5-oct) + GradNorm α=0.5 + WD=0.005 + Y-sym p=0.5 + EMA decay=0.999 + ema-start-step=500 + eval-raw-vs-ema + Lion lr=1e-4**. Kill cause: EP10 raw gate failure — val_raw=7.390% failed gate ≤7.2%; raw model regressed from 7.057% at EP8 to 7.390% at EP10 (overfitting onset). Eval-only run `wiroo9ux` on EP8 EMA checkpoint: test_abupt=7.9238% (+0.404pp vs SOTA), test_vol_p=12.179% (+1.421pp vs SOTA). Val→test vol_p gap: val=4.671% → test=12.179% = **+7.51pp** — identical to all prior completed runs. EMA hypothesis REJECTED: weight averaging does NOT close structural vol_p gap. **PR #954 CLOSED 2026-05-10.** |
| #951 | `wd5kgp2n` | dl24-fern | EP11 / EP15 gate next | **6.8634% (EP10 PASS ≤7.20% ✓) / EP11=6.9149% regression** | **6L STRING (5-oct) + GradNorm α=0.5 + WD=0.005 + Y-sym p=0.5 + 96k vol + 60k surf + Lion lr=1e-4**. Proportional point sampling (1.6:1 ratio) on 6L backbone — single-variable ablation vs #939. Trajectory: EP5=7.4688% PASS ✓, EP6=7.3665%, EP7=7.4112% (vol_p spike), EP8=7.2141% (recovery), EP9=6.9590%, **EP10=6.8634% PASS (gate ≤7.20% ✓)**, EP11=6.9149% (regression +0.051pp). Two-epoch vol_p oscillation pattern observed at EP7 (structural, not fatal — full recovery by EP8). EP15 gate (≤6.80%) next — needs ~0.029pp/epoch over 4 epochs from EP11. **MERGE CONFLICT UNRESOLVED — 6th reminder sent; cannot merge without rebase resolution.** **ACTIVE — monitoring EP15 gate.** |
| #946 | `qgkhoapw` | dl24-frieren | EP16 TERMINAL — **CLOSED** | **7.9972% (EP16 test; val_vol_p=4.52%, test_vol_p=12.20%)** | **6L STRING (5-oct) + GradNorm α=0.5 + WD=0.005 + Y-sym p=0.5 + extended cosine T_max=60 + Lion lr=1e-4**. Auto-kill fired at step 82,407 (EP16 boundary) overriding verbal "do not kill" instruction. EP16 terminal: test_abupt=7.9972%, test_vol_p=12.20%, val→test vol_p gap=+7.68pp. Extended cosine did not compress the gap. **FALSIFIED for wave-era 6L config. PR #946 CLOSED 2026-05-10.** |
| #964 | `gybtgqus` / eval `hm5z9oo5` | dl24-frieren | EP40 (terminal) — **CLOSED/FALSIFIED** | **test_abupt=8.0190% — REJECTED, val→test vol_p gap WIDENED to +8.12pp** | **6L STRING (5-oct) + WD=0.005 + Y-sym p=0.5 + fixed vol_loss_weight=3.0 + NO GradNorm + Lion lr=1e-4**. GradNorm disabled entirely; fixed 3× vol pressure upweight permanent. Terminal results: val_abupt=6.5381% (run best EP13), val_vol_p=3.806% (fleet-best val), test_abupt=8.0190%, test_vol_p=11.929%, test_surf_p=4.041%, test_wall=7.359%. Val→test vol_p gap=**+8.12pp** — WORST ever observed across all completed runs. Hypothesis FALSIFIED: amplifying vol_p gradient drove val overfitting but WIDENED the test gap. Fixed 3× vol weight WITHOUT GradNorm is strictly WORSE for test generalization than GradNorm-balanced runs. Val→test vol_p gap is a DATA DISTRIBUTION problem, not a loss weighting problem. **PR #964 CLOSED 2026-05-10.** |
| #978 | `b2lryieh` | dl24-tanjiro | EP5 FAIL — **CLOSED/FALSIFIED** | **8.4488% (EP5 FAIL vs gate ≤7.5%)** | **6L STRING (5-oct) + GradNorm α=0.5 + BBox canonical coordinate normalization + Y-sym p=0.5 + vol-points=65k**. Canonical frame normalization of point coordinates to [-1,+1]. Root cause of failure: STRING PE sigmas calibrated for ~5m raw coordinate range; normalizing to [-1,+1] rescales effective sigma coverage by ~5×, miscalibrating PE frequency spectrum. EP5=8.4488% FAIL (+0.95pp vs gate). Val_vol_p=6.4548% at EP5. Hypothesis FALSIFIED: BBox normalization incompatible with STRING PE sigmas without sigma recalibration. **PR #978 CLOSED 2026-05-11.** |
| #979 | `msmccvne` | dl24-nezuko | EP5 FAIL — **CLOSED/FALSIFIED** | **7.6317% (EP5 FAIL vs gate ≤7.5%); TTA A/B gap=+7.86pp UNCHANGED** | **6L STRING (5-oct) + GradNorm α=0.5 + TTA Y-symmetry ensemble + Y-sym p=0.5 + vol-points=65k**. Test-time augmentation: average predictions over original + Y-flipped inputs. EP5=7.6317% FAIL (training-time convergence not affected). TTA A/B eval on EP4 checkpoint: control (no TTA) test_abupt=8.3435%, test_vol_p=12.7757%; TTA: test_abupt=8.2991%, test_vol_p=12.7642% — val→test vol_p gap +7.86pp COMPLETELY UNCHANGED. Hypothesis FALSIFIED: TTA Y-symmetry does NOT close the structural val→test vol_p gap. Gap is DATA DISTRIBUTION problem, not test-time prediction symmetry. **PR #979 CLOSED 2026-05-11.** |
| #987 | `rydn7aqb` / eval `ky25tufm` | dl24-tanjiro | EP5 FAIL — **CLOSED/FALSIFIED** | **7.8846% (EP5 FAIL vs gate ≤7.5%); test_abupt=9.1036%, test_vol_p=14.278%, gap=+7.91pp** | **6L STRING (5-oct) + GradNorm α=0.5 + DropPath rate=0.1 (stochastic depth) + Y-sym p=0.5 + vol-points=65k**. DropPath drops entire residual branches per sample (stochastic depth regularization). First launch `m3jycncs` killed at 3 min (wrong dataset config — 347k views vs 87k). Relaunched `rydn7aqb`. Full EP trajectory: EP1=12.6253%, EP2=9.7335%, EP3=8.9169%, EP4=8.1520%, EP5=7.8846% FAIL (+0.385pp vs gate). GradNorm w_vol_p at EP5=0.672 vs baseline ~0.82 — DropPath adversarially interacts with GradNorm, accelerating down-weighting of vol_p. Eval `ky25tufm`: test_abupt=9.1036%, test_vol_p=14.278%, test_surf_p=4.584%, test_wall=8.111%. Val→test vol_p gap: 6.3659% → 14.278% = **+7.91pp** UNCHANGED. Hypothesis FALSIFIED: DropPath slows convergence AND has adversarial GradNorm interaction. **PR #987 CLOSED 2026-05-11.** |
| #990 | `um3tuyvy` | dl24-nezuko | EP5 FAIL — **CLOSED/FALSIFIED** | **8.5416% (EP5 FAIL vs gate ≤7.5%); val_vol_p=9.9228%** | **6L STRING (5-oct) + GradNorm α=0.5 + vol coord noise σ=0.005 + Y-sym p=0.5 + vol-points=65k**. Gaussian noise σ=0.005 on volume query xyz coordinates during training only (data augmentation on vol point positions). EP5=8.5416% FAIL (+1.04pp vs gate). Val_vol_p=9.9228% at EP5 (vs baseline ~6.4% — 3.5pp WORSE). Noise slows convergence significantly without reducing val→test gap. Hypothesis FALSIFIED: vol coord noise σ=0.005 adds training-time noise without compressing structural DATA DISTRIBUTION gap. **PR #990 CLOSED 2026-05-11.** |
| #968 | `a0yoxy85` | dl24-fern | EP15 (val best) / EP30 (terminal) — **CLOSED/FALSIFIED** | **test_abupt=7.6157%, test_vol_p=12.1140% — DOES NOT BEAT WAVE SOTA** | **6L STRING (5-oct) + GradNorm α=0.5 + stochastic vol subsampling (random 65k subset each step, no caching) + Y-sym p=0.5**. Hypothesis: random vol-point sampling each step forces the model to generalise over the full vol distribution rather than memorising fixed-index subsets. Val best EP15: ABUPT=6.2806%, val_vol_p=3.999% (run best). Terminal EP30: val_abupt=6.4622%, val_vol_p=4.166%. Test (from best-val EP15 checkpoint): test_abupt=7.6157%, test_vol_p=12.1140%, test_surf_p=3.9440%, test_wall=7.0470%. Val→test vol_p gap: **+8.115pp** — WIDEST ever observed across all wave runs. Stochastic subsampling compressed val vol_p to 4.0% (fleet-best at EP15) but WIDENED the test gap beyond all prior approaches. Hypothesis FALSIFIED: val vol_p improvement is entirely overfitting to the specific subset of val vol points presented; test evaluates ALL vol points, exposing the memorisation. **PR #968 CLOSED 2026-05-12.** |

_Last updated: 2026-05-24 10:57Z (session). **MERGED dl24 SOTA: PR #1284 run `yym5oa8x` test_WSS=6.6506% — first wave-33 merge into `drivaerml-long-20260504` (H39 wider surface_out MLP factor=2.0). Beats prior reference #972 (closed) WSS 6.727% by −0.0764pp.** Active WIP (drivaerml-long, 13 open): **#1071** (surface point density 131k), **#1070** (SDF vol α=2.0 + dedicated vol_p aux head, frieren), **#1069** (combined SDF + curvature stratified sampling, frieren), **#1068** (dedicated WSS surface decoder head 2-arm sweep), **#1067** (RFF octave ladder EP30 confirmation), **#1066** (nezuko tau Y/Z loss weight sweep), **#1065** (stark SDF-stratified extended 45ep corrected dataset), **#1063** (fern SDF near-surface α sweep EP5 PASS 6.350%), **#1061** (tanjiro stochastic per-batch vol points), **#1060** (askeladd vol-loss-weight sweep 1.5/2.0/3.0), **#1058** (frieren GradNorm on corrected dataset), **#1055** (tanjiro SDF α sweep 1.5/3.0/4.0; EP11=6.388% PASS ≤7.2% ✓; monitoring EP15 gate ≤6.80%; MERGE CONFLICT — FINAL WARNING), **#1050** (dropout regularization p=0.1). RECENTLY CLOSED: **#1054** (nezuko DANN domain adaptation — EP15 FAIL val_abupt=6.9264% vs gate ≤6.80%; CLOSED 2026-05-12), **#968** (fern stochastic vol subsampling — test_vol_p gap +8.115pp WIDENED, FALSIFIED, CLOSED 2026-05-12). Nezuko (dl24-nezuko) IDLE — needs new assignment._

---

## 2026-05-28 — PR #1344: WSS H147 — lion-β-only ablation (clean H39 + β=0.95/0.98 only) — **NEW WSS SOTA**

- **W&B run:** `k6q4c3on`
- **test_WSS:** **6.5409%** (−0.110pp vs H39 SOTA 6.6506%) ⭐ NEW WSS SOTA
- **test_ABUPT:** 5.6648% ✓ (clears 5.844% floor)
- **test_VP:** **3.4014%** ✓ (clears 3.643% floor by 0.242pp — **CORRECTED 2026-05-29: prior commit ea99dda copy-pasted H39's value 3.6033% by mistake**)
- **test_SP:** **3.5634%** ✓ (clears 3.577% floor by 0.014pp — **CORRECTED 2026-05-29: H147 actually CLEARS the SP floor; prior commit ea99dda copy-pasted H39's value 3.6498%**)
- **Per-axis WSS (corrected from W&B):** wss_x=5.8155%, wss_y=7.0556%, wss_z=8.4882%
- **Config:** Clean H39 stack (hidden_dim=512, layers=6, heads=4, slices=128, surface_out_width_factor=2.0, PE=string_multisigma) + `--lion-beta1 0.95 --lion-beta2 0.98` ONLY. No weight-decay drift, no loss modifications.
- **Key finding:** Lion β1=0.95/β2=0.98 (vs canonical 0.90/0.99) is the **confirmed single driver** of WSS improvement from the H138 complex. EP1 val exactly matched H138 (12.83%), confirming mechanism replication. β-drift is now the canonical optimizer config.
- **Reproduce:** `cd "target/" && python train.py --hidden-dim 512 --layers 6 --heads 4 --slices 128 --surface-out-width-factor 2.0 --pe-type string_multisigma --lion-beta1 0.95 --lion-beta2 0.98 --ema-decay 0.999 --ema-start-step 500 --epochs 30`

### Current single-model best on `drivaerml-long-20260504` (updated 2026-05-28)

**MERGED dl24 SOTA (2026-05-28; metrics CORRECTED 2026-05-29):** PR #1344 (`k6q4c3on`) — H147 Lion β1=0.95/β2=0.98 ablation — **NEW WSS SOTA, CLEARS ALL 4 FLOORS**
test_primary/wall_shear_rel_l2_pct = **6.5409%** ⭐ (beats H39 6.6506% by −0.110pp)
test_primary/volume_pressure_rel_l2_pct = **3.4014%** ✓ (clears 3.643% floor by 0.242pp)
test_primary/surface_pressure_rel_l2_pct = **3.5634%** ✓ (clears 3.577% floor by 0.014pp)
test_primary/abupt_axis_mean_rel_l2_pct = **5.6648%** ✓ (clears 5.844% floor)
Canonical optimizer: Lion lr=1e-4, β1=0.95, β2=0.98

> **CORRECTION 2026-05-29 (verified by W&B query `k6q4c3on` + cross-check by nezuko in PR #1360):** The prior BASELINE.md update (commit ea99dda, 2026-05-28) listed `test_VP=3.6033%` and `test_SP=3.6498%` for H147. Those values are actually H39's (run `yym5oa8x`), copy-pasted in error. The true H147 test metrics are `test_VP=3.4014%` and `test_SP=3.5634%`. **Material change in interpretation:** H147 actually CLEARS all 4 floors including SP (3.5634% < 3.577%); does NOT miss the SP floor by +0.073pp as previously documented. This corrects analyses banked from H150 (PR #1359 closure) and any other comparisons drawn against the old H147 numbers. New candidates must still beat test_WSS < 6.5409% (primary) and may use the PR #972 floor envelope (test_VP ≤ 3.643%, test_SP ≤ 3.577%, test_ABUPT ≤ 5.844%) as the merge-eligibility floor, but conclusions about whether H150-style β-shifts "improved" pressure heads against H147 must be re-derived from the corrected baseline.

---

## 2026-06-01 — PR #1510: H183 — Per-channel surface decoder heads (split shared MLP 4×1) — **NEW SOTA, ALL 4 FLOORS CLEARED**

- **W&B run:** `guw83mge` (rank0, `dl24-tanjiro/h183-main-30ep`)
- **Group:** `h183-per-channel-decoder-heads`
- **Best checkpoint:** EP24 EMA (selection metric: `val_primary/abupt_axis_mean_rel_l2_pct=5.8686`)
- **Runtime:** ~22.8h (82069s)

| Metric | H183 | H147 SOTA | Δ vs H147 | Floor | Status |
|---|---:|---:|---:|---:|---|
| test_WSS | **6.4427%** | 6.5409% | **−0.098pp** | — | ✅ NEW SOTA |
| test_VP | 3.4415% | 3.4014% | +0.040pp | ≤ 3.643% | ✅ clears |
| test_SP | **3.5187%** | 3.5634% | **−0.045pp** | ≤ 3.577% | ✅ clears by 0.058pp |
| test_ABU | **5.6152%** | 5.6648% | **−0.050pp** | ≤ 5.844% | ✅ clears |
| test_τ_x | 5.6983% | 5.8155% | −0.117pp | — | ✅ improved |
| test_τ_y | 6.9813% | 7.0556% | −0.074pp | ≤ 3.65% (paper) | improved |
| test_τ_z | 8.4364% | 8.4882% | −0.052pp | ≤ 3.63% (paper) | improved |

**Key finding — val→test mapping changes with architecture topology:**
H183's val trajectory looked ~+0.04pp BEHIND H147 on WSS at EP20-30, but test shows −0.098pp IMPROVEMENT. val→test pattern is NOT constant: H147 pattern ≈ 0pp gap; H183 pattern = −0.14pp gap on WSS and −0.32pp on SP. Projections using H147's val→test map must be bracketed for alternative architectures.

Per-channel heads act as a mild generalization regularizer — each independent head (cp, τ_x, τ_y, τ_z) commits to one channel's loss landscape rather than diluting capacity across four targets.

**Delta vs H147 stack:** only one flag added — `--per-channel-surface-heads`

**Reproduce command:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 --lion-beta1 0.95 --lion-beta2 0.98 \
  --batch-size 1 --lr-warmup-epochs 1 --lr-cosine-t-max 30 --epochs 30 \
  --model-pe string_multisigma --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 --pe-num-features 16 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --vol-p-charbonnier-weight 0.1 --wss-charbonnier-weight 0.1 --wss-charbonnier-axes z \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15 \
  --use-curvature-attention-bias --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --surface-out-width-factor 2.0 \
  --per-channel-surface-heads \
  --train-surface-points 65000 --train-volume-points 65000 \
  --eval-surface-points 65536 --eval-volume-points 65536 \
  --ema-decay 0.999 --ema-start-step 500
```

### Current single-model best on `drivaerml-long-20260504` (updated 2026-06-01)

**MERGED dl24 SOTA (2026-06-01):** PR #1510 (`guw83mge`) — H183 per-channel surface decoder heads — **NEW WSS SOTA, ALL 4 FLOORS CLEARED**
test_primary/wall_shear_rel_l2_pct = **6.4427%** ⭐ (beats H147 6.5409% by −0.098pp)
test_primary/volume_pressure_rel_l2_pct = **3.4415%** ✓ (clears 3.643% floor)
test_primary/surface_pressure_rel_l2_pct = **3.5187%** ✓ (clears 3.577% floor by 0.058pp)
test_primary/abupt_axis_mean_rel_l2_pct = **5.6152%** ✓ (clears 5.844% floor)
Per-axis WSS: wss_x=5.6983%, wss_y=6.9813%, wss_z=8.4364%
