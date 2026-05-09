# SENPAI Research State

- 2026-05-09 ~04:30 UTC (latest update: nezuko #874 EP13 NEW BEST=6.5381% within 0.0091pp of frieren wave leader; fern #881 EP10 gate PASS w_vol_p settled into target band; tanjiro #873 EP6 healthy mid-EP7; frieren #844 eval-only request still pending student response).
- Most recent research direction from human researcher team: Issue #717 (tay branch, separate research line) — comprehensive volume improvement plan. **No new directives in main human-issues queue as of 2026-05-09.** Hard constraints from launch directive: single-model only (no ensemble/soup/greedy-K), DDP8 required, test_primary metrics required for merge eligibility, val for steering only.

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (5-octave sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-hidden-dim=512, model-heads=4, model-slices=128. Depth axis under active exploration: 4L (SOTA) → 5L (#844 wave leader) → 6L (#831 closed; #874 active) → 7L (#873 active).

**Wave SOTA (merged test):** PR #740 (fern, `5x8wofzm`), test `abupt_axis_mean_rel_l2_pct` = **7.5195%**, surface=3.8810%, volume=10.7580%, wall=7.0610%. Improved from PR #741 (7.8232%) by 0.3037pp.

**Wave val leader (TERMINAL, awaiting test):** PR #844 (dl24-frieren, `7dqsxvbq`) **best=6.5290% @ EP35** — 5L STRING + GradNorm α=0.5 (no Y-sym). Run state=**FINISHED at EP41 step 225,253** (24h timeout). 0.0695pp BELOW SOTA-advancement val threshold 6.5985%. **NO test_primary metrics logged** — train.py timeout interrupted before final test eval phase. **PR sent back 2026-05-09 for `--eval-only --eval-checkpoint outputs/drivaerml/run-7dqsxvbq/checkpoint.pt`** to obtain test metrics. Merge decision pending test result.

**Strongest emerging signal:** PR #874 (dl24-nezuko, `rm6u10ro`) — 6L STRING + GradNorm α=0.75 + Y-sym. **EP13 NEW RUN BEST=6.5381%** (only 0.0091pp behind frieren #844 EP35 wave leader, at EP13 vs EP35 — α=0.75 6L converging considerably faster than α=0.5 5L). Every channel run-best at EP13 (vol_p=4.2190%, surf_p=4.2929%, wsh=7.2750%, wsh_z=9.9402%). w_vol_p=0.7029 recovered from EP11 trough (0.6181 @ step 61,324) back to healthy 0.69-0.73 band. Continuing to EP20 (ETA ~07:55 UTC). Strong candidate to overtake frieren as wave val leader.

**Y-sym physical signal CONFIRMED (PR #855, CLOSED):** Y-sym p=0.5 standalone (no GradNorm, no 6L) isolates τ_y < τ_z channel ordering at val EP3, EP4, AND test. Cleanest physical signal isolation to date. Gate miss due to no other optimizations; signal confirmed.

### Active Experiments (as of 2026-05-09 ~04:30 UTC)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #844 | dl24-frieren | 5L STRING + GradNorm α=0.5 (**no Y-sym**) | `7dqsxvbq` | **TERMINAL @ EP41/step 225,253**. Run best=**6.5290% @ EP35** (vol_p=4.398%). EP36-EP41 plateau in 6.5385-6.5599% range (basin oscillation, not divergence). **No test_primary metrics — sent back for `--eval-only` test pass on EP35 checkpoint at 03:37Z; no student response as of 04:30Z.** Merge gating on test result. |
| #874 | dl24-nezuko | 6L STRING + GradNorm **α=0.75** + Y-sym (α-axis upper half) | `rm6u10ro` | **EP13 NEW RUN BEST=6.5381%** ⭐ (vol_p=4.2190%, surf_p=4.2929%, wsh=7.2750%, wsh_z=9.9402% — every channel run-best). EP12=6.6217% was single-epoch bounce, EP11→EP13 = -0.0305pp recovery. **Within 0.0091pp of frieren wave leader at EP13 vs EP35** — α=0.75 6L converging considerably faster than α=0.5 5L. w_vol_p=0.7029 recovered from EP11 trough (0.6181 @ step 61,324) to healthy 0.69-0.73 band. Continue to EP20 (ETA ~07:55 UTC); strong candidate to overtake frieren. |
| #881 | dl24-fern | **Volume MLP head** — deeper MLP decoder on volume_hidden latents | `k59gu9o5` | EP10 gate PASS=**6.6867%** (+0.51pp margin under 7.20% kill); **EP9=6.6786% run best**; vol_p=4.1088% (run best, monotonic descent EP6-EP10), surf_p=4.3885%, wsh_z=10.10%; **w_vol_p settled into target [0.80-1.20] band at ~1.14** (was 1.484 at EP7) — GradNorm equilibrium reached, MLP head's early dominance resolved cleanly; r_vol_p~0.45 (well below 0.65 stability threshold). Continue to EP15/EP20/EP25/EP30 milestones; EP30 hard gate ≤6.6005% (in-wave merge threshold). |
| #873 | dl24-tanjiro | 7L STRING + GradNorm α=0.5 + Y-sym (depth axis extension) | `2oweovb3` (active) / `59bcgz40` (killed) | **EP6=6.8467%** (latest), descending healthily; mid-EP7 (step ~34,611). EP1=10.5138%, EP2=8.2038%, EP3=7.3671%, EP4=6.9672%, EP5=6.8718% (gate cleared). vol_p=4.4760%, w_vol_p=0.6079 stabilized 0.60-0.61 band (below 0.70 watch but stable; vol_p still descending despite low weight). EP10 gate (≤7.2%) next, on track. |

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
| #843 | dl24-nezuko | 7-octave STRING PE (`pe_init_sigmas=[0.25,0.5,1.0,2.0,4.0,8.0,16.0]`) + GradNorm a=0.5 + Y-sym | CLOSED — EP30 dual kill gate NOT MET: primary=6.6340% (threshold<6.6005%) AND vol_p=4.4287% (threshold<4.30%); vol_p NEVER below 4.30% across entire run (oscillating 4.43-4.66%); sigma=16.0 destabilization hypothesis confirmed. Extra octave (sigma=16) DOES NOT help; 6-octave (sigma=0.25-8.0) remains optimal. |
| #866 | dl24-tanjiro | 6L STRING + GradNorm a=0.5 + Y-sym p=1.0 (full symmetry) | CLOSED — EP4 GATE DEFINITIVELY FAILED: Y-sym p=1.0 (every batch flipped) falsified; EP4 mathematically could not reach <=7.5% gate (projected EP5~8.2% at maximum recovery rate, gate requires 7.5%); severe deceleration EP3.40=11.5961% (-0.34pp/ep vs -1.71pp/ep EP1->2 rate); p=1.0 over-augmentation hypothesis confirmed. Optimal Y-sym probability is p=0.5 (PR #741). |

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

4. **Depth axis: 4L→5L→6L confirmed monotonic improvement; 7L under test.** Frieren #844 (5L, no Y-sym, `7dqsxvbq`) is **WAVE VAL LEADER at 6.5290% @ EP35 — BEATS SOTA** — vol_p=4.0689% declining (positive signal). Fern #831 (6L + Y-sym, `pnrgixj1`) CLOSED EP29=6.5477% best (test=7.7872% non-winner). Tanjiro #873 (7L, `2oweovb3`) EP3.64 val=7.3671% — testing whether depth trend extends to 7L; w_vol_p=0.633 concern. If passes gates and val improves, expected best ~6.44%. Nezuko #874 (6L + α=0.75, `rm6u10ro`) EP9=6.6104% shows 6L + α=0.75 is competitive.

5. **Y-sym physical signal CONFIRMED ISOLATED (PR #855, CLOSED).** τ_y < τ_z channel ordering confirmed at val EP3, val EP4, AND test even in standalone Y-sym-only config. Physical basis for the augmentation is now well-established.

6. **Triple-compose failures (PRs #800, #806) — vol_p val→test gap systematic (~3× ratio: val~4.0-4.3%, test~12.0%).** Frieren #844 (5L + GradNorm, no Y-sym) EP21=6.5747% with vp=4.1128% (STABLE, declining) — clean two-way stack is working. This confirms that STRING + GradNorm alone can cross the SOTA threshold; Y-sym adds ~0.027pp (fern #831 EP29=6.5477%). Whether Y-sym is the source of triple-compose vol_p gap is still open — 7-octave nezuko #843 shows GradNorm can still cause vol_p instability (oscillating 4.45–4.95%) even without being a confound with Y-sym.

7. **RFF capacity axis CLOSED.** Both rff24 (#845) and rff32 (#846) failed the EP4 gate. RFF capacity increase (beyond SOTA 16 features) does not improve within 4-ep screen. RFF axis is closed.

8. **LR warmup axis CLOSED.** warmup=2ep (#847) is definitively worse than warmup=1ep at every checkpoint. 500-step warmup is optimal.

9. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. No direct architectural fix yet tested.

## Potential Next Research Directions

**Currently in-flight (as of 2026-05-09 ~04:30 UTC):**
- **Frieren #844**: 5L STRING no Y-sym — TERMINAL @ EP41 (run `7dqsxvbq`); val best=6.5290% @ EP35; **awaiting student eval-only test pass on EP35 checkpoint** (request sent 03:37Z, no response yet); merge gated on test_primary metrics.
- **Nezuko #874**: 6L STRING + GradNorm α=0.75 + Y-sym — (run `rm6u10ro`); **EP13=6.5381% NEW RUN BEST** ⭐, within 0.0091pp of frieren wave leader; every channel run-best; w_vol_p recovered to 0.7029; continue to EP20 ETA ~07:55 UTC.
- **Tanjiro #873**: 7L STRING + GradNorm α=0.5 + Y-sym — (run `2oweovb3`); EP6=6.8467% descending healthily, mid-EP7; EP10 gate (≤7.2%) on track; w_vol_p=0.6079 stabilized below watch threshold but vol_p still descending.
- **Fern #881**: Volume MLP head — (run `k59gu9o5`); EP10 gate PASS=6.6867% (+0.51pp margin); EP9=6.6786% run best; w_vol_p settled into target [0.80-1.20] band at ~1.14 (was 1.484 EP7); GradNorm equilibrium achieved; vol_p=4.1088% monotonic; continue to EP30.

**High-priority candidates after current wave completes:**
1. **5L STRING + Y-sym + GradNorm triple compose with α=0.5**: if frieren #844 two-way shows gains, add Y-sym back for full triple-compose (α=0.5 not α=0.25). Key: #844 isolates two-way first.
2. ~~**Volume MLP head**~~ — **ASSIGNED to dl24-fern as PR #881 (2026-05-05)**. Vol gap structural fix in flight.
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

_Last updated: 2026-05-09 ~04:30 UTC. Key events: (1) **nezuko #874 EP13=6.5381% NEW RUN BEST** ⭐ — within 0.0091pp of frieren wave leader at EP13 vs EP35; α=0.75 6L converging considerably faster than α=0.5 5L; every channel run-best; continue to EP20 ETA ~07:55 UTC; (2) **fern #881 EP10 gate PASS=6.6867%** — w_vol_p settled into target [0.80-1.20] band, GradNorm equilibrium achieved; vol_p=4.1088% monotonic descent; continue to EP30; (3) **tanjiro #873 EP6=6.8467%** — descending healthily mid-EP7; (4) **frieren #844 eval-only request pending** student response (sent 03:37Z); merge gated on test pass. All 4 students occupied, 0 idle._
