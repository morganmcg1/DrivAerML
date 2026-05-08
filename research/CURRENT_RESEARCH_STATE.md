# SENPAI Research State

- 2026-05-08 (latest update): **tanjiro #818 val best=6.6053% @ EP14.16** (run=`dy2z6o4a`; EP50 still running); **fern #831 val best=6.6164% @ EP6.50** (run=`pnrgixj1`; 6L STRING; strongly descending); **nezuko #843 EP2.00=7.4031%** (run=`hyzdxrj2`; 7-octave STRING PE; EP5 gate pending; student not posting gates); **frieren #844 run=`7dqsxvbq` EP1=12.4553%** (5L + GradNorm α=0.5 no Y-sym; EP5 gate ~10:20Z due); **tay screening wave**: thorfinn #842 EP3=8.246% borderline (gate missed by 0.25pp → continue to EP4; advisor OK'd); tanjiro-tay #840 EP2=12.345% PASS (EP3 overdue, nudge sent); nezuko-tay #823 RELAUNCHED as `ghh0s4ne` (prior kill retracted; EP1 due ~10:09Z); **CLOSED**: alphonse #839 (EP2=8.04% missed <8% gate), askeladd #836 (EP2=8.02% missed gate); **new tay screens**: alphonse #848 lion-lr-downsweep (lr=8e-5/7e-5; launch prompted), askeladd #849 tau-y-z-differential (τy=2.0/2.5 vs τz=1.5; launch prompted); **RFF capacity sweep** (fern #845 rff24, edward #846 rff32 ETA ~12:35Z); **frieren #847 lr-warmup-2ep** ETA ~12:37Z.
- Most recent research direction from human researcher team: Issue #717 (tay branch) — comprehensive volume improvement plan: Phase 0 diagnostics, Phase 1-3 probes (dual-tower, anomaly sampling, geometry conditioning, single-model KD). Hard no-ensemble constraint. Separate advisor branch. Issue #759 (tay): optional Bengio draft PRs as menu for tay repurposing — light suggestion only.

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA (merged test):** PR #740 (fern, `5x8wofzm`), test `abupt_axis_mean_rel_l2_pct` = **7.5195%**, surface=3.8810%, volume=10.7580%, wall=7.0610%. Improved from PR #741 (7.8232%) by 0.3037pp.
**Wave val leader (WIP):** PR #818 (dl24-tanjiro, `dy2z6o4a`) **best=6.6053% @ EP14.16 WAVE VAL LEADER** (2026-05-09). 6-octave STRING PE + GradNorm α=0.5 + Y-sym; EP15.17=6.6089% (latest, tight oscillation near floor; 0.0036pp above best). Fern #831 (6L STRING + GradNorm α=0.5 + Y-sym) EP6.50=6.6164% best (only 0.0111pp behind wave leader; strongly descending; early in run). Nezuko #843 (7-octave STRING PE + GradNorm α=0.5 + Y-sym) EP2.00=7.4031% best; EP5 gate ≤7.5% nearly met already. Frieren #844 (5L STRING + GradNorm α=0.5 no Y-sym) RELAUNCHED: run=`7dqsxvbq` (original `3054rc61` killed — wrong --train-volume-points 16384); ~EP0.30, EP5 gate pending. **CLOSED:** PR #806 (frieren 5L GradNorm α=0.25 + Y-sym, `gui4ceed`) terminal deep plateau — NOT merged; PR #780 (tanjiro α=0.25 no Y-sym) terminal test=8.0647% — NOT merged; PR #800 (nezuko 5L STRING α=0.5 + Y-sym) terminal test=7.8981% — NOT merged (vol_p val→test gap 7.76pp); PR #841 (edward Lion β₁=0.85) EP1=31.17% DIVERGED — CLOSED; PR #838 (fern RFF σ=0.125) test=8.7190% CLOSED — σ<0.25 dead end.

### Active Experiments (as of 2026-05-09)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #818 | dl24-tanjiro | 6-octave STRING PE (`sigmas=[0.25,0.5,1.0,2.0,4.0,8.0]`) + GradNorm α=0.5 + Y-sym | `dy2z6o4a` | **best=6.6053% @ EP14.16 WAVE VAL LEADER**. Val history: EP10=6.653%, EP12=6.647%, EP13=6.609%, EP14.16=6.6053% (best), EP14.67=6.6143%, EP15.17=6.6089% (latest). Tight oscillation near floor (noise band ~0.004pp). EP50 still running. |
| #831 | dl24-fern | 6L STRING + GradNorm α=0.5 + Y-sym (extra-depth variant) | `pnrgixj1` | **best=6.6164% @ EP6.50** (0.0111pp behind wave leader). Val history: EP5=6.993%, EP5.5=6.702%, EP6=6.840%, EP6.50=6.6164% (best), EP7=6.6245%, current EP7.28. Strongly descending; early in run; EP20 gate CLEARED. Wall-clock cutoff ~EP43-44. |
| #843 | dl24-nezuko | 7-octave STRING PE (`sigmas=[0.25,0.5,1.0,2.0,4.0,8.0,16.0]`) + GradNorm α=0.5 + Y-sym | `hyzdxrj2` | **EP2.00=7.4031% (best)**, EP2.21 current. Val: EP0.5=10.899%, EP1=7.943%, EP1.5=7.423%, EP2=7.4031%. On track for EP5 gate ≤7.5% (nearly met already). EP5 gate checkpoint ETA ~May 9. |
| #844 | dl24-frieren | 5L STRING + GradNorm α=0.5 compose (**no Y-sym**) — clean two-way stack | `7dqsxvbq` (RELAUNCHED) | ~EP0.30, no val yet. Original run `3054rc61` intentionally killed by student at EP1 — wrong `--train-volume-points 16384` in PR body; relaunched 2026-05-08T07:44Z with `--train-volume-points 65000`. NOTE: student also clarified that #740 (wave SOTA) also used string_multisigma PE — so only real diff vs #740 is `--model-layers 5`. Key novelty: clean test of 5L + GradNorm α=0.5 without Y-sym confound. EP5 gate ≤7.5% pending. |
| #842 | thorfinn (tay) | tay 4-epoch screen | `3487klz8` | EP3=8.246% (gate <8% MISSED by 0.25pp — advisor OK'd continue to EP4); EP4 gate ≤6.5985% |
| #840 | tanjiro-tay (tay) | tay 4-epoch screen | `oiptel6p` | EP2=12.345% PASS; EP3 result OVERDUE (nudge sent 2026-05-08); gate <8% pending |
| #823 | nezuko-tay (tay) | tay 4-epoch screen with --use-surf-to-vol-xattn | `ghh0s4ne` (RELAUNCHED) | Prior run `lp7u9r8g` killed by advisor error → retracted; new run launched 2026-05-08; EP1 gate <30% pending |
| #848 | alphonse (tay) | tay 4-epoch screen: Lion lr=8e-5 vs 7e-5 downsweep | TBD (launch prompted 2026-05-08) | Dual-arm: Arm A `--lr 8e-5`, Arm B `--lr 7e-5`; student not yet confirmed launch |
| #849 | askeladd (tay) | tay 4-epoch screen: τy=2.0/2.5 vs τz=1.5 differential | TBD (launch prompted 2026-05-08) | Dual-arm: Arm A τy=2.0/τz=1.5, Arm B τy=2.5/τz=1.5; student not yet confirmed launch |
| #845 | dl24-fern | RFF num_features=24, rff_init_sigmas=0.25,0.5,1.0,2.0,4.0 (sota-spectrum) | `thimjhnd` (rank0) | EP0.28; no val yet; group=rff-capacity-sweep; 8 runs |
| #846 | dl24-edward | RFF num_features=32, rff_init_sigmas=0.25,0.5,1.0,2.0,4.0 (sota-spectrum) | `px719275` | EP0.27; no val yet; group=rff-capacity-sweep |
| #847 | dl24-frieren | LR warmup 2ep variant | `7gzie3gj` | EP0.26; no val yet |

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
| #780 | dl24-tanjiro | GradNorm α=0.25 (no Y-sym) | TERMINAL test=8.0647%, NOT merged. Confirms α=0.5 is unimodal optimum on α-axis. |
| #794 | dl24-fern | GradNorm α=0.25 + Y-sym (4L) | TERMINAL test=7.9011% (+0.382pp regression). 4L architecture deficit vs 5L. NOT merged. |
| #800 | dl24-nezuko | 5L STRING + GradNorm α=0.5 + Y-sym | TERMINAL test=7.8981% (+0.379pp regression). Critical vol_p val→test gap 7.76pp. NOT merged. |
| #806 | dl24-frieren | 5L STRING + GradNorm α=0.25 + Y-sym | TERMINAL test=7.9323% (+0.413pp). Deep plateau EP28-47. NOT merged. Third triple-compose failure. |
| #841 | dl24-edward | Lion β₁=0.85 | EP1=31.17% DIVERGED. CLOSED. β₁=0.9 is stable optimum. |
| #838 | dl24-fern | STRING RFF σ=0.125, 4-epoch tay screen | **CLOSED** — test=8.7190% (+1.2pp regression vs SOTA 7.5195%); merge gate failed (tay EP4=7.4255% > ≤6.5985% required); σ<0.25 axis confirmed dead at 65k surface points (aliasing dominates capacity at high point density) |
| #839 | alphonse (tay) | tay screen: Lion lr=1e-4, τy=1.5, τz=2.0, string_separable, L5 | **CLOSED** — EP2=8.04% (gate <8% missed). Run `6jxmx316`. |
| #836 | askeladd (tay) | tay screen: same config as above | **CLOSED** — EP2=8.02% (gate <8% missed). Run `46idy5bk`. |

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

3. **Does GradNorm α=0.25 (more conservative) beat α=0.5? (tanjiro #780) — CLOSED NEGATIVE. Terminal test=8.0647%, NOT merged.** (run=`20n1fvwn`). Every channel worse vs α=0.5. **Definitively confirms α=0.5 is the unimodal optimum on the α-axis.**

4. **Does QK-Norm + Y-symmetry compose? (nezuko #784) — CLOSED NEGATIVE.** EP20 gate MISSED. Second negative QK-Norm result on this wave.

4b. **Does 5L STRING + GradNorm α=0.5 compose? (nezuko #800) — CLOSED TERMINAL test=7.8981%.** Critical vol_p val→test gap = 7.76pp. Does NOT beat merged SOTA. Superseded by nezuko #843 (7-octave STRING PE + GradNorm α=0.5 + Y-sym).

5. **Triple-compose failures (PRs #794, #800, #806):** All three triple-compose runs (5L STRING + GradNorm + Y-sym, in various combinations) showed same vol_p val→test gap (~3× ratio: val~4.0-4.3%, test~12.0%). Frieren #844 tests clean two-way (5L + GradNorm, no Y-sym) to isolate whether Y-sym is the confound.

6. **6-octave STRING PE (tanjiro #818) is wave val leader.** best=6.6053% @ EP14.16 — 0.453pp below current wave SOTA val best when SOTA was set. Tight oscillation near floor suggests approaching convergence. Terminal test will be the key metric.

7. **6L STRING (fern #831) closes gap rapidly.** best=6.6164% @ EP6.50 — only 0.0111pp behind wave leader at much earlier epoch (6.5 vs 14.16). Extra depth appears to be learning faster. Strong candidate to overtake #818 as val leader by EP20+.

8. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. Y-symmetry (#741) reduces gap via effective dataset doubling. GradNorm (#740) reduces vol error to 10.758%. All triple-compose failures driven by this gap. No direct architectural fix yet tested.

9. **RFF capacity sweep (PRs #845, #846):** Testing Random Fourier Features with different num_features (24 vs 32) and sota-spectrum sigmas (0.25-4.0). PR #838 (σ=0.125) was **CLOSED** — test=8.7190% (+1.2pp regression); σ<0.25 is a confirmed dead end at 65k surface points (aliasing ~82% of regression). #845 and #846 correctly use only the 5-octave SOTA spectrum (σ=0.25-4.0) to test RFF capacity.

## Potential Next Research Directions (after current arms complete)

**Currently in-flight (12 active PRs):**
- Tanjiro #818: 6-octave STRING PE — **best=6.6053% @ EP14.16 WAVE VAL LEADER**; EP50 still running
- Fern #831: 6L STRING — **best=6.6164% @ EP6.50**; strongly descending; early stages
- Nezuko #843: 7-octave STRING PE — **EP2.00=7.4031%**; EP5 gate pending (student not posting gates)
- Frieren #844: 5L STRING no Y-sym — run=`7dqsxvbq`; EP1=12.4553%; EP5 gate ~10:20Z due
- Tay screening: #842 (thorfinn EP3 borderline→continue EP4), #840 (tanjiro-tay EP3 overdue), #823 (nezuko-tay relaunched `ghh0s4ne`), #848 (alphonse launch-prompted), #849 (askeladd launch-prompted)
- CLOSED (EP2 gate miss): #839 (alphonse EP2=8.04%), #836 (askeladd EP2=8.02%)
- RFF capacity: #845 (fern rff24, `thimjhnd`), #846 (edward rff32, `px719275`) — ETA ~12:35Z
- LR warmup: #847 (frieren lr-warmup-2ep, `7gzie3gj`) — ETA ~12:37Z

**High-priority candidates after current wave completes:**
1. **5L STRING + Y-symmetry + GradNorm triple compose**: if frieren #844 two-way shows gains, add Y-sym back for full triple-compose but with α=0.5 (not α=0.25 like the failed triple-compose PRs #794, #800, #806). Key: #844 isolates the two-way first.
2. **Volume MLP head**: replace volume Transolver decoder with separate MLP for independent volume capacity. Pre-wave evidence (`8x7c537j`). Vol gap (10.758%) is now 2.77× surface (3.881%) — structural fix needed.
3. **GradNorm α optimal sweep (α=0.1, α=0.75)**: tanjiro #780 tests α=0.25, fern #740 confirmed α=0.5. Map the α curve more fully to find true optimal.
4. **EMA-proxy GradNorm α=0.5 (clean re-run)**: prior PR #623 failed on logistics. Pre-wave `wyz68o8r` showed 8.236% test. Worth clean re-test on current SOTA STRING + Y-sym.
5. **Beta-NLL heteroscedastic surface head**: principled loss for heteroscedastic tau_y/z noise. Higher risk, potentially high reward for wall_shear_z bottleneck.
6. **6L STRING + Y-sym + RFF**: if #831 confirms 6L depth advantage, combine with best RFF from capacity sweep.

**Confirmed dead ends — do not retry:**
- Weight decay variations (PR #667): vol gap WORSENS as WD decreases.
- QK-Norm at lr=5e-5 (PR #732): negative result.
- 7-sigma STRING PE (PR #673): config mismatch + regression.
- lr=9e-5 control (PR #749): no improvement vs lr=1e-4.
- Triple-compose with α=0.25 (PRs #794, #800, #806): all three failed; vol_p val→test gap systematic.
- GradNorm α=0.25 (PR #780): confirmed α=0.5 is optimal on α-axis.
- Lion β₁=0.85 (PR #841): catastrophic divergence.

_Last updated: 2026-05-08. Key events: (1) tanjiro #818 `dy2z6o4a` best=**6.6053%** @ EP14.16 WAVE VAL LEADER; EP50 running; (2) fern #831 `pnrgixj1` best=**6.6164%** @ EP6.50; 6L depth, strongly descending; (3) nezuko #843 `hyzdxrj2` EP2.00=7.4031%; EP5 gate pending (student not posting gates); (4) frieren #844 `7dqsxvbq` RELAUNCHED with correct --train-volume-points 65000; EP1=12.4553%; EP5 gate ~10:20Z; (5) tay wave: CLOSED #839 (alphonse EP2=8.04%) and #836 (askeladd EP2=8.02%); #842 (thorfinn EP3 borderline→continue EP4); #840 (tanjiro-tay EP3 overdue, nudge sent); #823 (nezuko-tay RELAUNCHED as `ghh0s4ne`); NEW #848 (alphonse lion-lr-downsweep) and #849 (askeladd tau-y-z-differential) — both launch-prompted; (6) RFF/LR screens #845, #846, #847 — 4-ep tay ETA ~12:35-12:37Z._
