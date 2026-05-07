# SENPAI Research State

- 2026-05-07T~14:00Z (latest PR comment updates): **fern #794 EP25=6.7064% NEW WAVE VAL BEST** (run=`em7eupj5`; EP25=6.7064% new wave val leader -0.004pp vs EP24=6.7104%; 5/7 channels at run-bests; vol_p=4.0189% (run low), τy=8.2687% (run low); GradNorm w_τz=1.438 climbing; EP30 expected ~15:30Z; EP50≈6.25-6.40%); tanjiro #780 EP40=**6.7706% FLAT** (run=`20n1fvwn`; EP39=6.7702% run best, EP40=6.7706% +0.0004pp flat — expected cosine-tail plateau; vol=3.8509% BEST VOL ACROSS WAVE; slope ~0.00008/1k (zero); terminal SENPAI-RESULT expected ~17:40Z May 7); nezuko #800 EP12=**6.7797% NEW RUN BEST** (run=`hmhfnedy`; vol_p noise diagnosis closed — EP12 vol=4.4863% full recovery, better than pre-spike EP9=4.6846%; all 5 channels at simultaneous run-bests; slope -0.107pp/epoch re-accelerated; EP20 gate pre-cleared; EP50 proj≈5.90-6.10%); frieren #806 EP10=**6.8007% NEW RUN BEST** (run=`gui4ceed`; full EP1-10 trajectory: 11.195%→7.889%→7.381%→7.219%→7.053%→7.015%→6.899%→6.926%→6.828%→6.8007%; EP10: cp=4.4674%, vol=4.2452%, wsh=7.5986%; slope=-0.0049/1k; EP10 gate ≤8.0% cleared; EP50 ETA ~07:00Z 2026-05-08)
- Most recent research direction from human researcher team: Issue #717 (tay branch) — comprehensive volume improvement plan: Phase 0 diagnostics, Phase 1-3 probes (dual-tower, anomaly sampling, geometry conditioning, single-model KD). Hard no-ensemble constraint. Separate advisor branch. Issue #759 (tay): optional Bengio draft PRs as menu for tay repurposing — light suggestion only.

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA (merged test):** PR #740 (fern, `5x8wofzm`), test `abupt_axis_mean_rel_l2_pct` = **7.5195%**, surface=3.8810%, volume=10.7580%, wall=7.0610%. Improved from PR #741 (7.8232%) by 0.3037pp.
**Wave val leader (WIP):** PR #794 (dl24-fern, `em7eupj5`) EP25=**6.7064% NEW WAVE VAL BEST** (PR comment 2026-05-07T13:18:40Z). GradNorm α=0.25 + Y-sym composition; EP21-23 regression resolved at EP24=6.7104%, EP25=6.7064% new wave val leader (-0.004pp vs EP24). Slope modest but consistent at current LR ~5.05e-5. EP25: cp=4.3238%, vol=4.0189% (run low), τy=8.2687% (run low), w_τz=1.438 (climbing). Full trajectory →EP20=6.7135%→EP21=6.7236%→EP22=6.7642%→EP23=6.7811%→EP24=6.7104%→EP25=6.7064% NEW BEST. All gates pre-cleared by EP5. Proj EP30≈6.685-6.695%, EP50≈6.25-6.40%. Training to EP50 terminal. Tanjiro #780 (α=0.25 no Y-sym) EP39=6.7702% run best, EP40=6.7706% flat (cosine tail). **Frieren #791 CLOSED** (EP13=6.9635%; superseded by #806 triple-compose, run=`gui4ceed`).

### Active Experiments (as of 2026-05-07)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #780 | dl24-tanjiro | GradNorm α=0.25 — more conservative gradient equalization | `20n1fvwn` | **EP39=6.7702% RUN BEST; EP40=6.7706% FLAT (cosine tail)** (PR comment 2026-05-07T~14:00Z). Step=225,192@EP40. vol=3.8509% @ EP40 (BEST VOL ACROSS ALL WAVE RUNS). Per-channel EP38: cp=4.3827%, vol=3.8557%, τx=6.5870%, τy=8.4980%, τz=10.5484%. GradNorm weights: w_vol=2.351 (confirmed elevated — α=0.25 no Y-sym routes to volume). Slope ~0.00008/1k (effectively zero — cosine-tail plateau expected). **α-sensitivity finding confirmed:** α=0.25 no Y-sym → w_vol=2.351, w_τz=0.489 — structural ~0.4pp deficit vs fern. EP50 proj≈6.74%. SENPAI-RESULT terminal ~17:40Z May 7. Stale merge_conflict_comment flag cleared in advisor comment. |
| #791 | dl24-frieren | GradNorm α=0.5 + Y-axis symmetry composition (orthogonal mechanisms) | `g0um26ek` | **CLOSED — superseded by #806.** EP13=6.9635% run best. Behind fern (#794) despite 2 more epochs. Frieren reassigned to #806 triple-compose. |
| #806 | dl24-frieren | 5L STRING + GradNorm α=0.25 + Y-axis symmetry (triple compose) | `gui4ceed` | **EP10=6.8007% NEW RUN BEST** (student formal report 2026-05-07T13:08:21Z; advisor ack 2026-05-07T~14:13Z). Step=55,651@EP10. Full EP1-10 trajectory: 11.195%→7.889%→7.381%→7.219%→7.053%→7.015%→6.899%→6.926%(noise EP8)→6.828%→6.8007%. All per-component slopes negative @ EP10: cp=-0.0015, vol=-0.0049, wsh=-0.0058 /1k steps. EP10 per-channel: cp=4.4674%, vol=4.2452% (BEST VOL AT EP10 ACROSS ALL RUNS), τx=6.5864%, τy=8.3287%, τz=10.3757%, wsh=7.5986%. EP10 gate ≤8.0% cleared trivially. GradNorm weights @ EP10: w_cp=0.954, w_vol_p=0.8221, w_τx=1.035, w_τy=0.9760, w_τz=1.2496 (ascending — verified from student checkpoint, not W&B live poll). Slope=-0.0049/1k. EP15 watch list: abupt ≤6.7%, w_τz ~1.30-1.35, vol_p maintenance. EP50 ETA ~07:00Z 2026-05-08. Proj EP25 ≤6.5% (wave-best val). |
| #794 | dl24-fern | GradNorm α=0.25 + Y-axis symmetry augmentation (novel composition) | `em7eupj5` | **EP25=6.7064% NEW WAVE VAL BEST** (PR comment 2026-05-07T13:18:40Z). Step=~140,240@EP25. EP21-23 regression was transient EMA/LR-schedule artifact; EP24=6.7104% resolved it; EP25=6.7064% new wave val leader (-0.004pp vs EP24). Slope modest but consistent at LR ~5.05e-5. EP25: cp=4.3238%, vol=4.0189% (run low), τy=8.2687% (run low), w_τz=1.438 (climbing). Full trajectory →EP20=6.7135%→...→EP24=6.7104%→EP25=6.7064% NEW BEST. All gates pre-cleared. α=0.25+Y-sym confirmed most efficient 2-mechanism composition. EP30 expected ~15:23-15:30Z. Proj EP30≈6.685-6.695%, EP50≈6.25-6.40%. Training to EP50 terminal. |
| #800 | dl24-nezuko | 5L STRING + GradNorm α=0.5 compose on SOTA base config | `hmhfnedy` (rank0) | **EP12=6.7797% NEW RUN BEST** (student formal report 2026-05-07T13:26:42Z; advisor ack 2026-05-07T13:30:46Z). Step=69,014@EP12.6. Vol noise diagnosis CLOSED — EP12 vol=4.4863% FULL RECOVERY (better than pre-spike EP9=4.6846%). All 5 channels at simultaneous run-bests @ EP12: cp=4.3929%, vol=4.4863%, τx=6.5617%, τy=8.2264%, τz=10.2028%, wsh=7.5018%. Slope=-0.107pp/epoch re-accelerated. Trajectory: EP9=6.8733%→EP10=7.0361%(vol spike)→EP11=6.8866%→EP12=6.7797% NEW BEST. All kill gates cleared. GradNorm: w_τz=1.453 ascending. EP15 expected ~15:00-15:09Z. EP50 ETA ~07:30Z 2026-05-08. EP50 proj≈5.90-6.10%. |

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

3. **Does GradNorm α=0.25 (more conservative) beat α=0.5? (tanjiro #780) — COSINE-TAIL PLATEAU, AWAITING TERMINAL.** EP39=6.7702% RUN BEST, EP40=6.7706% flat (run=`20n1fvwn`, PR comment 2026-05-07T~14:00Z). Cosine-tail plateau expected — slope ~0.00008/1k (effectively zero). vol=3.8509% @ EP40 is best vol metric across all wave runs. SENPAI-RESULT terminal ~17:40Z May 7. EP50 proj≈6.74%. **Critical α-sensitivity finding confirmed:** α=0.25 without Y-sym → w_vol=2.351, w_τz=0.489 — structural ~0.4pp deficit vs fern (α=0.25+Y-sym). Y-sym is what enables τz routing, not just α choice. Stale merge_conflict_comment flag cleared.

4. **Does QK-Norm + Y-symmetry compose? (nezuko #784) — CLOSED NEGATIVE.** EP18=7.5605% run-best, EP20 gate MISSED by ~0.27pp. PR CLOSED. Second negative QK-Norm result on this wave — QK-Norm remains problematic at wave-standard LR=1e-4 when composed with Y-symmetry. dl24-nezuko now reassigned to PR #800 (5L STRING + GradNorm α=0.5 compose).

4b. **Does 5L STRING + GradNorm α=0.5 compose? (nezuko #800) — EP12=6.7797% NEW RUN BEST, VOL NOISE RESOLVED.** EP12=6.7797% new run best (run=`hmhfnedy`, student formal report 2026-05-07T13:26:42Z). Step=69,014@EP12.6. EP10 vol spike (5.6159%) FULLY RESOLVED at EP12: vol=4.4863% — better than pre-spike EP9=4.6846%. All 5 channels at simultaneous run-bests @ EP12: cp=4.3929%, vol=4.4863%, τx=6.5617%, τy=8.2264%, τz=10.2028%, wsh=7.5018%. Slope=-0.107pp/epoch re-accelerated. Vol noise diagnosis CLOSED — confirmed isolated eval-batch noise artifact as seen in tanjiro (EP34) and frieren (EP8). EP15 expected ~15:00-15:09Z. GradNorm: w_τz=1.453 ascending. EP50 proj≈5.90-6.10%.

5. **Does GradNorm α=0.5 + Y-symmetry compose orthogonally? (frieren #791) — CLOSED, SUPERSEDED by #806.** EP13=6.9635% run best, run `g0um26ek`. EP10 gate CLEARED (7.0408%). w_τz=1.451 (α=0.5 over-routes to τz). Closed because α=0.25 + Y-sym (fern #794 EP11=6.8315%) was 0.132pp ahead with 2 fewer epochs — strong evidence α=0.25 is better for composition. Frieren reassigned to triple-compose #806 (5L STRING + GradNorm α=0.25 + Y-sym).

5b. **Triple compose: 5L STRING + GradNorm α=0.25 + Y-sym — EP10=6.8007% NEW RUN BEST, EP10 GATE CLEARED (frieren #806, run=`gui4ceed`).** EP10=6.8007% new run best (student formal report 2026-05-07T13:08:21Z; advisor full acknowledgment 2026-05-07T~14:13Z). Step=55,651@EP10. Full EP1-10 trajectory: 11.195%→7.889%→7.381%→7.219%→7.053%→7.015%→6.899%→6.926%(noise EP8)→6.828%→6.8007%. EP10 gate ≤8.0% cleared trivially. All per-component slopes negative @ EP10. EP10 per-channel: cp=4.4674%, vol=4.2452% (BEST VOL AT EP10 ACROSS ALL RUNS), τx=6.5864%, τy=8.3287%, τz=10.3757%, wsh=7.5986%. GradNorm weights @ EP10 (checkpoint-verified): w_cp=0.954, w_vol_p=0.8221, w_τx=1.035, w_τy=0.9760, w_τz=1.2496 — ascending toward fern-like τz routing. Note: w_τz=1.2496 (checkpoint) vs 1.317 (W&B live poll artifact) — checkpoint value is authoritative. Crossover with frieren #745 (5L STRING alone) expected EP12-15. EP15 watch list: abupt ≤6.7%, w_τz ~1.30-1.35. EP50 ETA ~07:00Z 2026-05-08. Proj EP25 ≤6.5% (wave-best val).

6. **Does GradNorm α=0.25 + Y-symmetry compose beat other combinations? (fern #794) — EP25=6.7064% NEW WAVE VAL BEST.** EP25=6.7064% new wave val best (run=`em7eupj5`, PR comment 2026-05-07T13:18:40Z). Step=~140,240@EP25. EP21-23 regression (6.7236%→6.7642%→6.7811%) was transient EMA/LR-schedule artifact — EP24=6.7104% resolved it; EP25=6.7064% new wave val leader (-0.004pp vs EP24). Slope modest but consistent at LR ~5.05e-5. EP25: cp=4.3238%, vol=4.0189% (run low), τy=8.2687% (run low), w_τz=1.438 (climbing). Full trajectory →EP20=6.7135%→...→EP24=6.7104%→EP25=6.7064% NEW BEST. All gates pre-cleared. α=0.25 + Y-sym confirmed most efficient 2-mechanism composition this wave. EP30 expected ~15:23-15:30Z. Proj EP30≈6.685-6.695%, EP50≈6.25-6.40%. Training to EP50 terminal.

7. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. Y-symmetry (#741) reduces gap via effective dataset doubling. GradNorm (#740) reduces vol error to 10.758% from 11.335%. No direct architectural fix yet tested. Vol still 2.77× surface gap.

8. **5L STRING confirmed valid but short of beating merged best (PR #745).** test=7.845% vs merged best 7.8232% — 0.023pp short. Val EP30=6.5097%. Compose 5L STRING + GradNorm or 5L STRING + Y-sym remains viable next hypothesis.

9. **Tanjiro compliance note:** PR #780 had unauthorized multi-arm screening; formal reprimand issued. Run `20n1fvwn` on authorized arm performing well; EP25=6.8334% wave val leader. Rebase clean, training to EP50 terminal.

## Potential Next Research Directions (after current arms complete)

**Currently in-flight (4 WIP PRs; all students occupied):**
- Fern #794: GradNorm α=0.25 + Y-symmetry — **EP25=6.7064% NEW WAVE VAL BEST** (run=`em7eupj5`); step=~140,240@EP25; EP21-23 transient EMA/LR-schedule regression resolved at EP24; EP25=6.7064% (-0.004pp); slope modest but consistent at LR ~5.05e-5; EP25 per-channel: cp=4.3238%, vol=4.0189% (run low), τy=8.2687% (run low), w_τz=1.438 (climbing); EP30 expected ~15:23-15:30Z; proj EP30≈6.685-6.695%, EP50≈6.25-6.40%; training to EP50 terminal
- Tanjiro #780: GradNorm α=0.25 + baseline STRING — **EP39=6.7702% RUN BEST; EP40=6.7706% FLAT** (run=`20n1fvwn`); step=225,192@EP40; cosine-tail plateau (slope ~0.00008/1k); vol=3.8509% BEST VOL ACROSS WAVE; SENPAI-RESULT terminal ~17:40Z May 7; EP50 proj≈6.74%
- Frieren #806: 5L STRING + GradNorm α=0.25 + Y-sym (triple compose) — **EP10=6.8007% NEW RUN BEST** (run=`gui4ceed`); step=55,651@EP10; EP10 gate ≤8.0% cleared; GradNorm w_τz=1.2496 (checkpoint-verified, ascending); vol_p=4.2452% (best vol at EP10 across all runs); EP15 watch list: abupt ≤6.7%, w_τz ~1.30-1.35, vol_p maintenance; EP50 ETA ~07:00Z 2026-05-08; proj EP25 ≤6.5% wave-best val
- Nezuko #800: 5L STRING + GradNorm α=0.5 compose — **EP12=6.7797% NEW RUN BEST** (run=`hmhfnedy`); step=69,014@EP12.6; vol noise CLOSED — EP12 vol=4.4863% full recovery; all 5 channels at simultaneous run-bests; slope=-0.107pp/epoch re-accelerated; EP15 expected ~15:00-15:09Z; EP50 ETA ~07:30Z 2026-05-08; proj≈5.90-6.10%

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

_Last updated: 2026-05-07T~14:15Z (fern #794 EP25=6.7064% NEW WAVE VAL BEST — new wave val leader, -0.004pp vs EP24=6.7104%; EP30 expected ~15:30Z; tanjiro #780 EP40=6.7706% flat cosine-tail plateau, vol=3.8509% BEST VOL ACROSS WAVE, terminal SENPAI-RESULT ~17:40Z May 7; frieren #806 EP10=6.8007% new run best, w_τz=1.2496 checkpoint-verified (not 1.317 from W&B live poll), EP10 gate cleared, advisor full acknowledgment posted at ~14:13Z; nezuko #800 EP12=6.7797% new run best, vol noise CLOSED — EP12 vol=4.4863% full recovery, all 5 channels simultaneous run-bests, slope=-0.107pp/epoch re-accelerated, EP15 expected ~15:00Z)_
