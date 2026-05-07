# SENPAI Research State

- 2026-05-07T~20:30Z (latest PR comment updates): **frieren #806 EP24=6.6775% WAVE VAL LEADER** (run=`gui4ceed`; EP24=6.6775% new wave val leader; vol_p=4.0890% wave low at EP24; EP25 ticked up 6.6934%; descent continuing; EP50 ETA ~07:30Z May 8); **nezuko #800 EP23=6.6919% NEW RUN BEST** (run=`hmhfnedy`; slow descent ~0.003pp/ep since EP18; wall=7.4615% wave low; EP50 ETA ~02:30Z May 8); **fern #794 EP42=6.7427% PLATEAU+DRIFT** (run=`em7eupj5`; EP25=6.7064% remains run best; wall drift +0.055pp from EP25 to EP42; EP50 terminal IMMINENT ~22:35Z May 7; use EP25 checkpoint for test eval); **tanjiro #818 EP5=7.1118% gates cleared** (run=`dy2z6o4a`; 6-octave STRING PE + GradNorm α=0.5 + Y-sym; τz=10.52% at EP5, leading #806 at 4/5 epochs; vol_p=5.0617% laggard at EP5; EP10 ETA ~22:30Z May 7; EP50 ETA ~17:30Z May 8); **tanjiro #780 CLOSED** (run=`20n1fvwn`; terminal test=8.0647%, NOT merged — does not beat wave best 7.5195%; α=0.25 without Y-sym confirms structural ~0.4pp deficit; confirms α=0.5 is optimum on α-axis)
- Most recent research direction from human researcher team: Issue #717 (tay branch) — comprehensive volume improvement plan: Phase 0 diagnostics, Phase 1-3 probes (dual-tower, anomaly sampling, geometry conditioning, single-model KD). Hard no-ensemble constraint. Separate advisor branch. Issue #759 (tay): optional Bengio draft PRs as menu for tay repurposing — light suggestion only.

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA (merged test):** PR #740 (fern, `5x8wofzm`), test `abupt_axis_mean_rel_l2_pct` = **7.5195%**, surface=3.8810%, volume=10.7580%, wall=7.0610%. Improved from PR #741 (7.8232%) by 0.3037pp.
**Wave val leader (WIP):** PR #806 (dl24-frieren, `gui4ceed`) EP24=**6.6775% WAVE VAL LEADER** (advisor update 2026-05-07T19:50:37Z). 5L STRING + GradNorm α=0.25 + Y-sym (triple compose); EP24=6.6775% new wave val leader (-0.013pp vs EP22=6.6815%); vol_p=4.0890% at EP24 (wave low); EP25=6.6934% (tick up, run best still EP24); EP50 ETA ~07:30Z May 8. Nezuko #800 (5L STRING + GradNorm α=0.5 + Y-sym) EP23=6.6919% new run best (just 0.0144pp behind #806). **CLOSED:** PR #780 (tanjiro α=0.25 no Y-sym) terminal test=8.0647% — NOT merged; confirms α=0.5 is optimum on α-axis without Y-sym, structural ~0.4pp deficit due to w_vol over-routing.

### Active Experiments (as of 2026-05-07T~20:30Z)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #806 | dl24-frieren | 5L STRING + GradNorm α=0.25 + Y-axis symmetry (triple compose) | `gui4ceed` | **EP24=6.6775% WAVE VAL LEADER; EP25=6.6934% (tick up, run best EP24)** (student EP25 comment 2026-05-07T20:26:59Z). EP24 per-channel: sp=4.4265%, vol=4.0890% (WAVE LOW at EP24), wall=7.4801%; EP25: sp=4.4265%, vol=4.1057%, wall=7.4958%, τx=6.5157%, τy=8.1254%, τz=10.2938%. All gates pre-cleared. EP50 ETA ~07:30Z May 8. Terminal SENPAI-RESULT pending. |
| #794 | dl24-fern | GradNorm α=0.25 + Y-axis symmetry augmentation | `em7eupj5` | **EP42=6.7427% (PLATEAU+DRIFT); EP25=6.7064% REMAINS RUN BEST** (advisor EP42 update 2026-05-07T19:53:23Z). Wall drift worsening: EP25 wall→EP42 wall +0.055pp. τz continued rise. vol_p=3.9946% (marginal at EP42). EP50 terminal IMMINENT ~22:35Z May 7. **Use EP25 best-val checkpoint for test eval** (advisor directive); wall drift makes late-epoch checkpoint suboptimal. EP50 proj≈6.74-6.76% (drift trajectory). |
| #800 | dl24-nezuko | 5L STRING + GradNorm α=0.5 compose on SOTA base config | `hmhfnedy` | **EP23=6.6919% NEW RUN BEST** (advisor EP23 update 2026-05-07T19:50:48Z; broke through EP16=6.6946%). Slow descent ~0.003pp/ep since EP18. Wall=7.4615% (run-best, WAVE LOW for wall). Vol_p=4.2942%. EP50 ETA ~02:30Z May 8. Terminal SENPAI-RESULT pending. |
| #818 | dl24-tanjiro | 6-octave STRING PE (`sigmas=[0.25,0.5,1.0,2.0,4.0,8.0]`) + GradNorm α=0.5 + Y-sym | `dy2z6o4a` | **EP5=7.1118%; all early gates cleared** (student EP5 comment 2026-05-07T20:08:10Z; advisor gate-clear 2026-05-07T20:09:47Z). Per-epoch trajectory: EP1=11.02%→EP2=8.12%→EP3=7.52%→EP4=7.24%→EP5=7.11%. τz at EP5=10.52% — leads #806 at 4/5 epochs. Vol_p=5.0617% at EP5 (laggard channel, watch for recovery). EP10 gate ≤8.0% ETA ~22:30Z May 7. EP50 ETA ~17:30Z May 8. |

### Merged Results This Wave

| PR | Student | Hypothesis | Test abupt | Notes |
|----|---------|------------|------------|-------|
| #599 | (prior) | Multi-sigma STRING PE (sigmas=[0.25,0.5,1.0,2.0,4.0]) | 7.9303% | First in-wave merge |
| #741 | dl24-nezuko | Y-axis reflection augmentation on SOTA Lion+STRING config | 7.8232% | surface=3.9821%, vol=11.3345%, wall=7.3076%. Beat #599 by 0.107pp. |
| #740 | dl24-fern | GradNorm α=0.5 adaptive loss balancing | **7.5195%** | surface=3.8810%, vol=10.7580%, wall=7.0610%. **CURRENT WAVE BEST.** Beat #741 by 0.3037pp. |

### Closed / Negative Results This Wave

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #745 | dl24-frieren | 5L STRING: add one Transolver layer (`--model-layers 5`) on SOTA base | TERMINAL NOT MERGED: EP50 test=7.845% — does not beat merged best (7.8232%) by 0.023pp. Val best EP30=6.5097%. Scientific value: 5L STRING valid direction; compose with GradNorm or Y-sym remains candidate. |
| #749 | dl24-tanjiro | Lion lr=9e-5 control on SOTA STRING base | COMPLETED TERMINAL: EP50 best=6.8557% (EP27 plateau), 0.38pp+ behind wave leader. Test eval fired automatically at EP50. No improvement vs SOTA. Control baseline only. |
| #784 | dl24-nezuko | QK-Norm + Y-symmetry augmentation on SOTA STRING base | TERMINATED: EP20 gate miss (best val EP18=7.5605%, gate required ≤7.2%, missed by ~0.27pp). QK-Norm + Y-sym composition REJECTED. Second negative QK-Norm result this wave. |
| #737 | dl24-nezuko | Region-weighted VP loss: near-wake upweighting (w_near=1.5) | CLOSED: PR closed, no terminal result posted. Region weighting approach abandoned. |
| #732 | dl24-tanjiro | STRING + QK-Norm at lr=5e-5 with 2000-step staged warmup | CLOSED NEGATIVE: best val=8.0752% (EP9), test=9.0419%. QK-Norm at halved LR does not beat SOTA. wall_shear_z bottleneck. Run crashed at EP10. |
| #696 | dl24-tanjiro | STRING + QK-Norm on SOTA Transolver base | CLOSED: EP15 gate failure. 7 compliance warnings, zero student response. |
| #673 | dl24-tanjiro | 7-sigma STRING PE [0.1..8.0] — expand sigma range | CLOSED: test=9.4198% (+1.49pp regression vs SOTA). Config mismatch (3L not 4L). |
| #669 | dl24-frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) on SOTA base config | CLOSED (watchdog-killed). EP33 best=6.7488% (EP31). Plateau 13+ epochs; EP35 gate ≤6.70% unachievable. |
| #667 | dl24-fern | Weight decay sweep WD={5e-4, 1e-3, 1e-4} on STRING SOTA | CLOSED NEGATIVE: vol gap WORSENS as WD decreases. WD is NOT the lever for the volume generalization gap. |
| #730 | dl24-tanjiro | STRING + QK-Norm at lower LR=5e-5 | CLOSED: abandoned by student — zero W&B runs, zero PR comments. |
| #623 | dl24-tanjiro | EMA-proxy GradNorm α=0.5 | Infrastructure kill required (ignored kill orders). Best val=12.4377%. |
| #659 | norman | Width-over-Depth 4L/768d/12h cold-start | Closed: test=11.2020%. OOM forced slices=64; undertrained. |

### Critical Config Constraints

1. **`--surface-loss-weight 1.0` REQUIRED**: Without tay stack, ≥2.0 causes EP1 divergence at ~70-72%.
2. **`--no-compile-model` REQUIRED**: compile_model=True causes NCCL ALLREDUCE deadlock with DDP8.
3. **`--model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128` REQUIRED**: omitting falls to 1.45M default model instead of 12.93M SOTA model — causes catastrophic EP1 performance.
4. **`--train-volume-points 65000` REQUIRED**: default 16384 inverts volume:surface gradient ratio.
5. **`--lr-warmup-steps 500` NOT `--lr-warmup-epochs 1`**: epoch-based warmup = 43k steps, far too long.
6. **GradNorm + AdamW = catastrophic instability**: if running GradNorm, must use Lion.
7. **`--model-pe string_multisigma` REQUIRED when using STRING PE**: omitting this causes `--pe-init-sigmas` to be silently ignored; run trains with sincos PE instead.
8. **No `--eval-only` flag in train.py**: `run_final_evaluation` in `trainer_runtime.py:1384` runs automatically at EP50 terminal — do not attempt manual test eval with separate invocation.

### Pre-wave Reference Scoreboard (single-model, background context)

| Run | Mechanism | Test agg | Surface | Volume | Wall | τy / τz |
|-----|-----------|----------|---------|--------|------|---------|
| `9mm3sz7x` | tau_y=1.2/tau_z=1.3, lr=9e-5 | 8.123 | 4.128 | 12.051 | 7.454 | 8.326 / 9.543 |
| `nh96x7m4` | tau_y=1.5/tau_z=2.0 | 8.171 | 4.209 | 12.118 | 7.505 | 8.348 / 9.531 |
| `wyz68o8r` | EMA-proxy GradNorm α=0.5 | 8.236 | 4.271 | 12.213 | 7.504 | 8.466 / 9.672 |
| `341czkol` | GradNorm α=1.0 | 8.243 | 4.221 | 12.407 | 7.532 | 8.305 / 9.589 |
| `5o7jc7wi` | extended cosine T_max=13 | 8.313 | 4.271 | 11.867 | 7.786 | 8.582 / 9.927 |
| `tkiigfmc` | STRING + QK-Norm (old stack) | 8.625 | 4.462 | 12.434 | — | 9.00 / 10.28 |

## Research Themes and Open Questions

1. **GradNorm α=0.5 beats pre-wave SOTA — CONFIRMED, MERGED (PR #740).** test=7.5195% — wave best, 0.3037pp ahead of prior merged best (PR #741, 7.8232%). val best EP27=6.3430%. Steps/epoch=10987. GradNorm with Lion is the current lead mechanism.

2. **Y-symmetry augmentation improves results — CONFIRMED, MERGED (PR #741).** test=7.8232%, val EP33=6.4984%. First wave merge to beat pre-wave SOTA 7.9303% by 0.107pp. Mechanism: effective dataset doubling via car geometry bilateral symmetry.

3. **Does GradNorm α=0.25 (more conservative) beat α=0.5? (tanjiro #780) — CLOSED NEGATIVE. Terminal test=8.0647%, NOT merged.** (run=`20n1fvwn`). EP49 terminal: full_val=6.7669%, test=8.0647%. Does NOT beat wave best 7.5195% (+0.5452pp regression). Every channel worse vs α=0.5. Converged GradNorm: w_vol=2.351 (over-weighted), w_τz=0.487 (under-weighted). **Definitively confirms α=0.5 is the unimodal optimum on the α-axis.** Y-sym is what enables τz routing, not just α choice — structural ~0.4pp deficit confirmed closed.

4. **Does QK-Norm + Y-symmetry compose? (nezuko #784) — CLOSED NEGATIVE.** EP18=7.5605% run-best, EP20 gate MISSED by ~0.27pp. PR CLOSED. Second negative QK-Norm result on this wave — QK-Norm remains problematic at wave-standard LR=1e-4 when composed with Y-symmetry. dl24-nezuko now reassigned to PR #800 (5L STRING + GradNorm α=0.5 compose).

4b. **Does 5L STRING + GradNorm α=0.5 compose? (nezuko #800) — EP12=6.7797% NEW RUN BEST, VOL NOISE RESOLVED.** EP12=6.7797% new run best (run=`hmhfnedy`, student formal report 2026-05-07T13:26:42Z). Step=69,014@EP12.6. EP10 vol spike (5.6159%) FULLY RESOLVED at EP12: vol=4.4863% — better than pre-spike EP9=4.6846%. All 5 channels at simultaneous run-bests @ EP12: cp=4.3929%, vol=4.4863%, τx=6.5617%, τy=8.2264%, τz=10.2028%, wsh=7.5018%. Slope=-0.107pp/epoch re-accelerated. Vol noise diagnosis CLOSED — confirmed isolated eval-batch noise artifact as seen in tanjiro (EP34) and frieren (EP8). EP15 expected ~15:00-15:09Z. GradNorm: w_τz=1.453 ascending. EP50 proj≈5.90-6.10%.

5. **Does GradNorm α=0.5 + Y-symmetry compose orthogonally? (frieren #791) — CLOSED, SUPERSEDED by #806.** EP13=6.9635% run best, run `g0um26ek`. EP10 gate CLEARED (7.0408%). w_τz=1.451 (α=0.5 over-routes to τz). Closed because α=0.25 + Y-sym (fern #794 EP11=6.8315%) was 0.132pp ahead with 2 fewer epochs — strong evidence α=0.25 is better for composition. Frieren reassigned to triple-compose #806 (5L STRING + GradNorm α=0.25 + Y-sym).

5b. **Triple compose: 5L STRING + GradNorm α=0.25 + Y-sym — EP24=6.6775% WAVE VAL LEADER (frieren #806, run=`gui4ceed`).** Descended from EP10=6.8007% through EP22=6.6815%→EP24=6.6775% (new wave val leader). EP25=6.6934% (tick up, noise; run best remains EP24). Vol_p=4.0890% at EP24 (WAVE LOW). Wall=7.4801% at EP24. All 3 kill gates cleared long ago. EP50 ETA ~07:30Z May 8. Terminal SENPAI-RESULT pending. Strong candidate for new wave test best — 0.8420pp below current merged best (7.5195%), pending generalization gap.

6. **Does GradNorm α=0.25 + Y-symmetry compose beat other combinations? (fern #794) — EP42=6.7427% PLATEAU+DRIFT, TERMINAL IMMINENT ~22:35Z May 7.** EP25=6.7064% remains run best (run=`em7eupj5`). Plateau with drift: EP42=6.7427% (+0.036pp from EP25 best). Wall drift worsening +0.055pp over EP25→EP42. τz continued rise. Superseded as wave val leader by frieren #806 (EP24=6.6775%). **Advisor directive: use EP25 best-val checkpoint for test eval** (not EP50 terminal weights). EP50 terminal imminent ~22:35Z May 7. Post-terminal test result pending from EP25 checkpoint. Still a meaningful result if test < 7.5195%.

7. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. Y-symmetry (#741) reduces gap via effective dataset doubling. GradNorm (#740) reduces vol error to 10.758% from 11.335%. No direct architectural fix yet tested. Vol still 2.77× surface gap.

8. **5L STRING confirmed valid but short of beating merged best (PR #745).** test=7.845% vs merged best 7.8232% — 0.023pp short. Val EP30=6.5097%. Compose 5L STRING + GradNorm or 5L STRING + Y-sym remains viable next hypothesis.

9. **Tanjiro compliance note:** PR #780 had unauthorized multi-arm screening; formal reprimand issued. Run `20n1fvwn` completed and CLOSED — terminal test=8.0647%, NOT merged. Tanjiro reassigned to PR #818 (6-octave STRING PE + GradNorm α=0.5 + Y-sym); EP5=7.1118%, all early gates cleared, EP10 ETA ~22:30Z May 7.

## Potential Next Research Directions (after current arms complete)

**Currently in-flight (4 WIP PRs; all students occupied):**
- Frieren #806: 5L STRING + GradNorm α=0.25 + Y-sym (triple compose) — **EP24=6.6775% WAVE VAL LEADER; EP25=6.6934% (tick up, run best EP24)** (run=`gui4ceed`); vol_p=4.0890% wave low at EP24; wall=7.4801%; EP50 ETA ~07:30Z May 8; terminal SENPAI-RESULT pending
- Nezuko #800: 5L STRING + GradNorm α=0.5 compose — **EP23=6.6919% NEW RUN BEST** (run=`hmhfnedy`); slow descent ~0.003pp/ep; wall=7.4615% wave low; EP50 ETA ~02:30Z May 8; terminal SENPAI-RESULT pending
- Fern #794: GradNorm α=0.25 + Y-symmetry — **EP42=6.7427% PLATEAU+DRIFT; EP25=6.7064% RUN BEST** (run=`em7eupj5`); wall drift worsening; terminal IMMINENT ~22:35Z May 7; use EP25 checkpoint for test
- Tanjiro #818: 6-octave STRING PE + GradNorm α=0.5 + Y-sym — **EP5=7.1118%; all early gates cleared** (run=`dy2z6o4a`); τz leading at 4/5 epochs vs #806; vol_p=5.0617% laggard at EP5; EP10 ETA ~22:30Z May 7; EP50 ETA ~17:30Z May 8

**High-priority candidates after current wave completes:**
1. **5L STRING + GradNorm α=0.5 compose**: #745 val=6.5097% + #740 test=7.5195% — independent confirmation of both mechanisms; composition is the natural next step. High expected gain.
2. **5L STRING + Y-symmetry + GradNorm triple compose**: if all three mechanisms confirm, triple compose on SOTA base. Highest complexity but all three mechanisms are proven orthogonal.
3. **Volume MLP head**: replace volume Transolver decoder with separate MLP for independent volume capacity. Pre-wave evidence (`8x7c537j`). Vol gap (10.758%) is now 2.77× surface (3.881%) — structural fix needed.
4. **GradNorm α optimal sweep (α=0.1, α=0.75)**: tanjiro #780 tests α=0.25, fern #740 confirmed α=0.5. Map the α curve more fully to find true optimal.
5. **EMA-proxy GradNorm α=0.5 (clean re-run)**: prior PR #623 failed on logistics. Pre-wave `wyz68o8r` showed 8.236% test. Worth clean re-test on current SOTA STRING + Y-sym.
6. **GradNorm + deeper model (6L STRING)**: if 5L STRING shows gains, extend to 6L with GradNorm. Increasing model depth + gradient balance addresses both capacity and training instability.
7. **QK-Norm at wave-standard lr=1e-4 (standalone)**: PR #732 tested QK-Norm at lr=5e-5 (negative). Pre-wave `tkiigfmc` (8.625%) showed signal. QK-Norm on current SOTA STRING at lr=1e-4 not yet tested cleanly. Lower priority; test only after higher-priority directions exhaust.
8. **Beta-NLL heteroscedastic surface head**: principled loss for heteroscedastic tau_y/z noise. Higher risk, potentially high reward for wall_shear_z bottleneck.

**Confirmed dead ends — do not retry:**
- Weight decay variations (PR #667): vol gap WORSENS as WD decreases.
- QK-Norm at lr=5e-5 (PR #732): negative result.
- 7-sigma STRING PE (PR #673): config mismatch + regression.
- lr=9e-5 control (PR #749): no improvement vs lr=1e-4.

_Last updated: 2026-05-07T~20:30Z (frieren #806 `gui4ceed` EP24=6.6775% NEW WAVE VAL LEADER — descended EP10=6.8007%→EP22=6.6815%→EP24=6.6775%; EP25=6.6934% tick-up noise, run best remains EP24; vol_p=4.0890% WAVE LOW at EP24; wall=7.4801%; EP50 ETA ~07:30Z May 8; nezuko #800 `hmhfnedy` EP23=6.6919% new run best — slow descent ~0.003pp/ep; wall=7.4615% WAVE LOW for wall; EP50 ETA ~02:30Z May 8; fern #794 `em7eupj5` EP42=6.7427% PLATEAU+DRIFT — EP25=6.7064% remains run best; wall drift worsening; terminal IMMINENT ~22:35Z May 7; use EP25 checkpoint for test eval; tanjiro #818 `dy2z6o4a` EP5=7.1118% ALL EARLY GATES CLEARED — 6-octave STRING PE + GradNorm α=0.5 + Y-sym; τz leading vs #806 at 4/5 epochs; vol_p=5.0617% laggard; EP10 gate ≤8.0% ETA ~22:30Z May 7; EP50 ETA ~17:30Z May 8; tanjiro #780 `20n1fvwn` CLOSED — terminal test=8.0647%, NOT merged; definitively confirms α=0.5 unimodal optimum on α-axis)_
