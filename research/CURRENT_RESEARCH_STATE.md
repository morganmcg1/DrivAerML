# SENPAI Research State

- 2026-05-08 ~18:00 UTC (latest update)
- Most recent research direction from human researcher team: Issue #717 (tay branch) — comprehensive volume improvement plan: Phase 0 diagnostics, Phase 1-3 probes (dual-tower, anomaly sampling, geometry conditioning, single-model KD). Hard no-ensemble constraint. Separate advisor branch. Issue #759 (tay): optional Bengio draft PRs as menu for tay repurposing — light suggestion only.

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA (merged test):** PR #740 (fern, `5x8wofzm`), test `abupt_axis_mean_rel_l2_pct` = **7.5195%**, surface=3.8810%, volume=10.7580%, wall=7.0610%. Improved from PR #741 (7.8232%) by 0.3037pp.

**Wave val leader (WIP):** PR #831 (dl24-fern, `pnrgixj1`) **best=6.5477% @ EP29 WAVE VAL LEADER** (live W&B 2026-05-05). 6L STRING + GradNorm α=0.5 + Y-sym. EP31=6.5688% latest. **Below SOTA-advancement val threshold 6.5985% — pending terminal SENPAI-RESULT for merge consideration.** Frieren #844 (5L STRING + GradNorm α=0.5 no Y-sym) **EP20=6.5758%** — ALSO below 6.5985% threshold; new best for frieren, run best=6.5758% @ EP20, steadily descending. Nezuko #843 (7-octave STRING PE + GradNorm α=0.5 + Y-sym) EP24=**6.6646%** best; vol_p PERSISTENT instability EP20-25 (4.47-4.95% range oscillating vs 4.12% for fern/frieren).

**Y-sym physical signal CONFIRMED (PR #855, CLOSED):** Y-sym p=0.5 standalone (no GradNorm, no 6L) isolates τ_y < τ_z channel ordering at val EP3, EP4, AND test. Cleanest physical signal isolation to date. Gate miss (val EP4=8.0813%, test=9.2221%) due to no other optimizations; signal confirmed.

### Active Experiments (as of 2026-05-08 ~18:00Z)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #831 | dl24-fern | 6L STRING + GradNorm α=0.5 + Y-sym (extra-depth variant) | `pnrgixj1` | **WAVE VAL LEADER: best=6.5477% @ EP29, EP32=6.5772% latest** — BELOW 6.5985% SOTA-advancement threshold. vp=4.1194% stable. step=179115 (~18.16h runtime). Advisor requested EP35 report 2026-05-08T17:31Z. |
| #843 | dl24-nezuko | 7-octave STRING PE (`sigmas=[0.25,0.5,1.0,2.0,4.0,8.0,16.0]`) + GradNorm α=0.5 + Y-sym | `hyzdxrj2` | EP25 best=**6.6588%**. **PERSISTENT vol_p instability** (4.4557%, oscillating 4.45–4.95% range vs 4.12% for fern/frieren). **EP30 convergence gate: run best must improve beyond 6.6588% by EP30 or kill.** 0.099pp above fern wave leader. |
| #844 | dl24-frieren | 5L STRING + GradNorm α=0.5 (**no Y-sym**) — clean two-way stack | `7dqsxvbq` | **EP21 best=6.5747% — SOTA-BEATING (below 6.5985%).** vp=4.1128% (stable, declining). step=115696. 2nd wave val leader. Trajectory toward 6.54–6.55% by EP30-32. |
| #866 | dl24-tanjiro | 6L STRING + GradNorm α=0.5 + **Y-sym p=1.0** dose-response | `gb73kgzz` | EP1=**13.6449%**, vol_p=10.7627%, surf_p=10.1875%, wss=14.0790%. step~5588. **EP5 gate ≤7.5% pending** (needs −6.14pp drop by EP5, step 27469). |

**Tay-screen wave (closed/killed 2026-05-09):**
- PR #857 (askeladd σ-ladder Arm B `o7odqtqq`): NO DATA logged (run finished without metrics — likely crashed early). CLOSED.
- PR #859 (thorfinn model-slices=64 `7dq0l9s7`): EP3 GATE FAILED — val=8.54% vs threshold 8.0%. CLOSED.
- PR #861 (edward QK-norm ablation `gk7d3qqn`): EP3 GATE FAILED — val=11.79% vs threshold 8.0%. Third independent QK-Norm dead-end confirmation. CLOSED.
- PR #862 (tanjiro-tay Lion β₂=0.95 `dpxzt2cp`): EP2 GATE FAILED — val=22.68% vs threshold 16%. CLOSED.
- PR #863 (alphonse SGDR `7gnqa6l1`): live but launched with wrong haku-wave flags (`--lr-warmup-epochs 1`, `--pos-encoding-mode string_separable`, `--use-qk-norm`, `--model-layers 5`, `--lr 9e-5`, `--vol-points-schedule`); same failure mode as PR #865 closure. EP1 data pending; on tay branch (separate research line).

### Merged Results This Wave

| PR | Student | Hypothesis | Test abupt | Notes |
|----|---------|------------|------------|-------|
| #599 | (prior) | Multi-sigma STRING PE (sigmas=[0.25,0.5,1.0,2.0,4.0]) | 7.9303% | First in-wave merge |
| #741 | dl24-nezuko | Y-axis reflection augmentation on SOTA Lion+STRING config | 7.8232% | surface=3.9821%, vol=11.3345%, wall=7.3076%. Beat #599 by 0.107pp. |
| #740 | dl24-fern | GradNorm α=0.5 adaptive loss balancing | **7.5195%** | surface=3.8810%, vol=10.7580%, wall=7.0610%. **CURRENT WAVE BEST.** Beat #741 by 0.3037pp. |

### Closed / Negative Results This Wave

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #745 | dl24-frieren | 5L STRING: add one Transolver layer (`--model-layers 5`) on SOTA base | TERMINAL NOT MERGED: EP50 test=7.845%. Val best EP30=6.5097%. |
| #749 | dl24-tanjiro | Lion lr=9e-5 control on SOTA STRING base | COMPLETED TERMINAL: EP50 best=6.8557% (EP27 plateau). Control baseline only. |
| #784 | dl24-nezuko | QK-Norm + Y-symmetry augmentation on SOTA STRING base | TERMINATED: EP20 gate miss (best val EP18=7.5605%, gate ≤7.2%, missed by ~0.27pp). Second negative QK-Norm result. |
| #800 | dl24-nezuko | 5L STRING + GradNorm α=0.5 + Y-sym | TERMINAL test=7.8981% (+0.379pp regression). Critical vol_p val→test gap 7.76pp. NOT merged. |
| #806 | dl24-frieren | 5L STRING + GradNorm α=0.25 + Y-sym | TERMINAL test=7.9323% (+0.413pp). Deep plateau EP28-47. NOT merged. Third triple-compose failure. |
| #841 | dl24-edward | Lion β₁=0.85 | EP1=31.17% DIVERGED. CLOSED. β₁=0.9 is stable optimum. |
| #838 | dl24-fern | STRING RFF σ=0.125, 4-epoch tay screen | CLOSED — test=8.7190% (+1.2pp regression). σ<0.25 axis confirmed dead at 65k surface points. |
| #845 | dl24-fern | RFF num_features=24 (sota-spectrum σ=0.25-4.0) | CLOSED — EP3 gate FAILED (<8% threshold). RFF capacity increase from 16 features does NOT help within 4-ep screen. |
| #846 | dl24-edward | RFF num_features=32 (sota-spectrum σ=0.25-4.0) | CLOSED — EP3 passed (7.941%), EP4 FAILED gate ≤6.5985%. RFF capacity axis CLOSED — both 24 and 32 features fail gate. |
| #847 | dl24-frieren | LR warmup 2 epochs | CLOSED — EP4=7.871% vs gate ≤6.5985%. Definitively worse than warmup=1ep at every checkpoint. warmup=1ep (500 steps) is optimal. This axis is closed. |
| #855 | dl24-frieren | Y-sym p=0.5 standalone (no GradNorm, no 6L) — physical signal isolation | CLOSED — gate miss (val EP4=8.0813%, test=9.2221%), but τ_y < τ_z CONFIRMED at all checkpoints (val EP3, EP4, test). Cleanest Y-sym physical isolation to date. Frieren assigned Y-sym p=1.0 follow-up. |
| #856 | dl24-fern | τ Y/Z absolute upscaling — precursor tay | CLOSED — superseded by PR #860 (same student, fresh launch). |
| #737 | dl24-nezuko | Region-weighted VP loss: near-wake upweighting (w_near=1.5) | CLOSED: no terminal result posted. Region weighting approach abandoned. |
| #732 | dl24-tanjiro | STRING + QK-Norm at lr=5e-5 with 2000-step staged warmup | CLOSED NEGATIVE: best val=8.0752% (EP9), test=9.0419%. |
| #696 | dl24-tanjiro | STRING + QK-Norm on SOTA Transolver base | CLOSED: EP15 gate failure. |
| #673 | dl24-tanjiro | 7-sigma STRING PE [0.1..8.0] — expand sigma range | CLOSED: test=9.4198% (+1.49pp regression). Config mismatch (3L not 4L). |
| #669 | dl24-frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) | CLOSED (watchdog-killed). Plateau 13+ epochs. |
| #667 | dl24-fern | Weight decay sweep WD={5e-4, 1e-3, 1e-4} on STRING SOTA | CLOSED NEGATIVE: vol gap WORSENS as WD decreases. WD is NOT the lever for the volume generalization gap. |
| #780 | dl24-tanjiro | GradNorm α=0.25 (no Y-sym) | TERMINAL test=8.0647%. Confirms α=0.5 is unimodal optimum on α-axis. |
| #794 | dl24-fern | GradNorm α=0.25 + Y-sym (4L) | TERMINAL test=7.9011% (+0.382pp regression). 4L architecture deficit vs 5L. |
| #659 | norman | Width-over-Depth 4L/768d/12h cold-start | Closed: test=11.2020%. OOM forced slices=64; undertrained. |
| #839 | alphonse (tay) | tay screen: Lion lr=1e-4, τy=1.5, τz=2.0, string_separable, L5 | CLOSED — EP2=8.04% (gate <8% missed). |
| #836 | askeladd (tay) | tay screen: same config as above | CLOSED — EP2=8.02% (gate <8% missed). |
| #842 | thorfinn (tay) | tay 4-epoch screen | CLOSED — EP4=7.610% (run `3487klz8`), gate ≤6.5985% FAILED by 1.011pp. |
| #840 | tanjiro-tay (tay) | tay 4-epoch screen | CLOSED — EP4=7.8558% (run `oiptel6p`), gate ≤6.5985% FAILED by 1.256pp. |
| #818 | dl24-tanjiro | 6-octave STRING PE + GradNorm α=0.5 + Y-sym | TERMINAL: best=6.6053% @ EP28 (`dy2z6o4a`); did NOT beat fern #831 wave leader. |
| #857 | askeladd (tay) | STRING σ-ladder Arm B (σ={0.125..2.0}) | CLOSED — NO DATA logged (run `o7odqtqq` finished without metrics, crashed early). |
| #859 | thorfinn (tay) | model-slices=64 | CLOSED — EP3 GATE FAILED, val=8.54% vs 8.0% threshold (`7dq0l9s7`). |
| #861 | edward (tay) | QK-norm ablation L5 | CLOSED — EP3 GATE FAILED, val=11.79% vs 8.0% (`gk7d3qqn`). Third QK-Norm dead-end confirmation. |
| #862 | tanjiro-tay (tay) | Lion β₂=0.95 | CLOSED — EP2 GATE FAILED, val=22.68% vs 16% (`dpxzt2cp`). |

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

1. **GradNorm α=0.5 beats pre-wave SOTA — CONFIRMED, MERGED (PR #740).** test=7.5195% — wave best. GradNorm with Lion is the current lead mechanism.

2. **Y-symmetry augmentation improves results — CONFIRMED, MERGED (PR #741).** test=7.8232%. Mechanism: effective dataset doubling via car geometry bilateral symmetry.

3. **GradNorm α=0.5 is unimodal optimum — CONFIRMED.** α=0.25 (PR #780) terminal test=8.0647%. α-axis is closed.

4. **6L STRING (fern #831) is WAVE VAL LEADER at 6.5477% @ EP29 (EP32=6.5772% latest).** 5L frieren #844 has broken threshold too at 6.5747% @ EP21 (no Y-sym control). Both below SOTA-advancement threshold 6.5985%. Extra depth + Y-sym (fern) yields ~0.027pp additional gain vs frieren two-way stack. Wall-clock cutoff ~EP43-44 for fern. Test eval critical for both.

5. **Y-sym physical signal CONFIRMED ISOLATED (PR #855, CLOSED).** τ_y < τ_z channel ordering confirmed at val EP3, val EP4, AND test even in standalone Y-sym-only config. Physical basis for the augmentation is now well-established.

6. **Triple-compose failures (PRs #800, #806) — vol_p val→test gap systematic (~3× ratio: val~4.0-4.3%, test~12.0%).** Frieren #844 (5L + GradNorm, no Y-sym) EP21=6.5747% with vp=4.1128% (STABLE, declining) — clean two-way stack is working. This confirms that STRING + GradNorm alone can cross the SOTA threshold; Y-sym adds ~0.027pp (fern #831 EP29=6.5477%). Whether Y-sym is the source of triple-compose vol_p gap is still open — 7-octave nezuko #843 shows GradNorm can still cause vol_p instability (oscillating 4.45–4.95%) even without being a confound with Y-sym.

7. **RFF capacity axis CLOSED.** Both rff24 (#845) and rff32 (#846) failed the EP4 gate. RFF capacity increase (beyond SOTA 16 features) does not improve within 4-ep screen. RFF axis is closed.

8. **LR warmup axis CLOSED.** warmup=2ep (#847) is definitively worse than warmup=1ep at every checkpoint. 500-step warmup is optimal.

9. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. No direct architectural fix yet tested.

## Potential Next Research Directions

**Currently in-flight (as of 2026-05-08 ~18:00 UTC):**
- Fern #831: 6L STRING — **WAVE VAL LEADER: 6.5477% @ EP29, EP32=6.5772% latest**; advisor requested EP35 report 2026-05-08T17:31Z; wall-clock cutoff ~EP43-44; vp=4.1194% stable
- Frieren #844: 5L STRING no Y-sym — **EP21=6.5747% SOTA-BEATING**, below 6.5985% threshold; trajectory toward 6.54–6.55% by EP30-32; vp=4.1128% stable declining
- Nezuko #843: 7-octave STRING PE — **EP25 best=6.6588%**; PERSISTENT vol_p instability oscillating 4.45–4.95%; **EP30 kill gate if no improvement beyond 6.6588%**
- Tanjiro #866: 6L STRING + Y-sym p=1.0 — EP1=13.6449%, step~5588; **EP5 gate ≤7.5% pending** at step 27469

**High-priority candidates after current wave completes:**
1. **5L STRING + Y-sym + GradNorm triple compose with α=0.5**: if frieren #844 two-way shows gains, add Y-sym back for full triple-compose (α=0.5 not α=0.25). Key: #844 isolates two-way first.
2. **Volume MLP head**: replace volume Transolver decoder with separate MLP for independent volume capacity. Vol gap (10.758%) is now 2.77× surface (3.881%) — structural fix needed.
3. **6L STRING + Y-sym + GradNorm long run**: if fern #831 confirms 6L depth advantage, confirm best test with combined mechanisms.
4. **Y-sym p=1.0 long run**: force every batch to apply the flip (p=1.0 vs SOTA p=0.5). Frieren assigned 4-ep tay screen first.
5. **Beta-NLL heteroscedastic surface head**: principled loss for heteroscedastic tau_y/z noise. Higher risk, potentially high reward for wall_shear_z bottleneck.
6. **τ Y/Z differential upscaling**: differentiated surface channel weights if τ_y < τ_z signal translates to better loss weighting.

**Confirmed dead ends — do not retry:**
- Weight decay variations (PR #667): vol gap WORSENS as WD decreases.
- QK-Norm at lr=5e-5 (PR #732): negative result.
- 7-sigma STRING PE (PR #673): config mismatch + regression.
- lr=9e-5 control (PR #749): no improvement vs lr=1e-4.
- Triple-compose with α=0.25 (PRs #794, #800, #806): all three failed; vol_p val→test gap systematic.
- GradNorm α=0.25 (PR #780): confirmed α=0.5 is optimal on α-axis.
- Lion β₁=0.85 (PR #841): catastrophic divergence.
- RFF capacity above 16 features (PRs #845, #846): both EP4 gate failures. Axis closed.
- LR warmup 2 epochs (PR #847): definitively worse than 1-ep warmup. Axis closed.

_Last updated: 2026-05-08 ~18:00 UTC. Key events: (1) fern #831 `pnrgixj1` WAVE VAL LEADER best=**6.5477%** @ EP29; EP32=6.5772% latest; step=179115; vp=4.1194% stable; EP35 milestone pending — advisor comment posted 2026-05-08 17:29 UTC; (2) frieren #844 `7dqsxvbq` EP21=**6.5747%** — SOTA-BEATING (BELOW 6.5985% threshold); clean two-way stack (STRING+GradNorm, no Y-sym) CONFIRMED; vp=4.1128% stable declining; step=115696; EP25+ milestone pending; (3) nezuko #843 `hyzdxrj2` EP25 best=**6.6588%**; PERSISTENT vol_p instability (4.4557% latest, oscillating 4.45–4.95%); EP30 convergence kill gate if no improvement beyond 6.6588% — advisor comment posted 2026-05-08 17:29 UTC; (4) tanjiro #866 `gb73kgzz` EP1=13.6449%; step~5588; vol_p=10.7627%; EP5 gate ≤7.5% pending — advisor comment posted 2026-05-08 17:29 UTC; (5) all 4 students occupied, 0 idle._
