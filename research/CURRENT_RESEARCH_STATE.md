# SENPAI Research State

- 2026-05-07T10:30Z (latest PR comment updates): **fern #794 EP16=6.7435% NEW RUN BEST, WAVE VAL LEADER** (run=`em7eupj5`, w_cp=0.789/w_τz=1.363/w_vol_p=0.998; all channels monotonically improving, EP17 in progress); tanjiro #780 EP33=**6.7970% NEW RUN BEST** (run=`20n1fvwn`; EP34=7.3507% eval noise outlier — uniform cross-channel jump, no NaN, GradNorm weights stable w_τz=0.49/w_vol=2.36; EP35 expected ~10:45Z); nezuko #800 EP5=**7.0322%** all gates cleared — fastest EP5 convergence in wave (run=`hmhfnedy`, w_τz=1.4926 mild pullback from 1.526); frieren #806 EP2=**7.8887%** triple-compose early breakout (run=`gui4ceed`, EP5+EP10 gates pre-cleared at EP2)
- Most recent research direction from human researcher team: Issue #717 (tay branch) — comprehensive volume improvement plan: Phase 0 diagnostics, Phase 1-3 probes (dual-tower, anomaly sampling, geometry conditioning, single-model KD). Hard no-ensemble constraint. Separate advisor branch. Issue #759 (tay): optional Bengio draft PRs as menu for tay repurposing — light suggestion only.

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA (merged test):** PR #740 (fern, `5x8wofzm`), test `abupt_axis_mean_rel_l2_pct` = **7.5195%**, surface=3.8810%, volume=10.7580%, wall=7.0610%. Improved from PR #741 (7.8232%) by 0.3037pp.
**Wave val leader (WIP):** PR #794 (dl24-fern, `em7eupj5`) EP~13-14 val=**6.7834% WAVE VAL LEADER** (W&B summary 2026-05-05). GradNorm α=0.25 + Y-sym composition; w_cp=0.789, w_τz=1.363, w_vol_p=0.998 at EP13-14 — intermediate routing (Y-sym relieves volume vs tanjiro w_vol=2.351). Fern converges fastest in wave; all gates pre-cleared by EP5. Training to EP50 terminal. Tanjiro #780 (α=0.25 no Y-sym) EP~30=6.8154% in plateau, proj EP50≈6.73%. **Frieren #791 CLOSED** (EP13=6.9635%; superseded by #806 triple-compose, run=`gui4ceed`, EP~1-2=11.195% very early stage).

### Active Experiments (as of 2026-05-07)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #780 | dl24-tanjiro | GradNorm α=0.25 — more conservative gradient equalization | `20n1fvwn` | **EP33=6.7970% NEW RUN BEST** (PR comment 2026-05-07T10:27Z). EP34=7.3507% eval noise outlier — uniform cross-channel jump (+0.34 cp, +0.39 vol, +0.99 τy, +0.64 τz), LR=2.4e-5 normal, GradNorm weights stable w_τz=0.49/w_vol=2.36 within 0.02 of EP30, no NaN/explosion. Per-channel EP33: cp=4.401%, vol=3.875%, τx=6.572%, τy=8.540%, τz=10.572%. GradNorm weights: w_cp=0.924, w_τx=0.979, w_τy=0.253, w_τz=0.493, **w_vol=2.351** (strongly elevated — volume routing without Y-sym). EP35 expected ~10:45Z to confirm noise diagnosis. SENPAI-RESULT terminal expected ~17:40Z May 7. Branch clean at 872b0df6 post-rebase. |
| #791 | dl24-frieren | GradNorm α=0.5 + Y-axis symmetry composition (orthogonal mechanisms) | `g0um26ek` | **CLOSED — superseded by #806.** EP13=6.9635% run best. Behind fern (#794) despite 2 more epochs. Frieren reassigned to #806 triple-compose. |
| #806 | dl24-frieren | 5L STRING + GradNorm α=0.25 + Y-axis symmetry (triple compose) | `gui4ceed` | **EP2=7.8887% EARLY BREAKOUT** (PR comment 2026-05-07T09:17Z). EP5+EP10 gates pre-cleared at EP2 (well below both ≤9.0% and ≤8.0% thresholds). GradNorm entering adaptation phase. Tied with nezuko at EP2. Highest-complexity composition: all three wave-positive mechanisms simultaneously. EP5 gate ≤9.0%: CLEAR AHEAD OF SCHEDULE. |
| #794 | dl24-fern | GradNorm α=0.25 + Y-axis symmetry augmentation (novel composition) | `em7eupj5` | **EP16=6.7435% NEW RUN BEST, WAVE VAL LEADER** (PR comment 2026-05-07T09:57Z). EP16=-0.032pp vs EP15=6.7750%. All channels improving monotonically: τy strongest gain (-0.045pp), τz second (-0.046pp). EP17 ~69% complete at reporting time. GradNorm weights: w_cp=0.789, w_τx=0.898, w_τy=0.952, w_τz=1.363, w_vol_p=0.998. Y-sym ABL gap vs tanjiro EP16≈0.165pp+. Full trajectory EP7=6.9907%→EP8=6.9228%→EP9=6.8943%→EP10=6.8631%→EP11=6.8315%→EP12=6.8132%→EP13=6.7834%→EP14=6.8195%(transient)→EP15=6.7750%→EP16=6.7435%. All gates pre-cleared. Projecting EP20≈6.64%, EP30≈6.37%, EP50≈6.1-6.3%. |
| #800 | dl24-nezuko | 5L STRING + GradNorm α=0.5 compose on SOTA base config | `hmhfnedy` (rank0) | **EP5=7.0322% ALL GATES CLEARED** (PR comment 2026-05-07T09:19Z). Fastest EP5 convergence in wave. GradNorm weights: w_cp=0.667, w_τx=0.800, w_τy=1.181, w_τz=1.4926 (mild pullback from 1.526), w_vol_p=0.825. All kill gates cleared. Sharp convergence ongoing. |

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

3. **Does GradNorm α=0.25 (more conservative) beat α=0.5? (tanjiro #780) — LATE-STAGE, STEADY DESCENT.** EP33=6.7970% NEW RUN BEST (run=`20n1fvwn`, PR comment 2026-05-07T10:27Z). EP34=7.3507% eval noise outlier confirmed (uniform cross-channel, LR normal, weights stable). EP35 expected ~10:45Z to confirm. SENPAI-RESULT terminal expected ~17:40Z May 7. Training to EP50, proj≈6.73-6.80%. **GradNorm routing:** α=0.25 without Y-sym routes heavily to volume (w_vol=2.351, w_τz=0.493) — compensating for volume gap via loss weighting. α=0.25 + Y-sym (fern) intermediate: w_vol=0.998, w_τz=1.363 — Y-sym relieves volume pressure. Fern #794 (α=0.25 + Y-sym) advancing faster than tanjiro standalone by 0.165pp+ at EP16. α=0.25 vs α=0.5 head-to-head answer awaits terminal.

4. **Does QK-Norm + Y-symmetry compose? (nezuko #784) — CLOSED NEGATIVE.** EP18=7.5605% run-best, EP20 gate MISSED by ~0.27pp. PR CLOSED. Second negative QK-Norm result on this wave — QK-Norm remains problematic at wave-standard LR=1e-4 when composed with Y-symmetry. dl24-nezuko now reassigned to PR #800 (5L STRING + GradNorm α=0.5 compose).

4b. **Does 5L STRING + GradNorm α=0.5 compose? (nezuko #800) — FASTEST WAVE CONVERGER AT EP5.** EP5=7.0322%, all gates cleared (run=`hmhfnedy`, PR comment 2026-05-07T09:19Z). GradNorm weights: w_cp=0.667, w_τx=0.800, w_τy=1.181, w_τz=1.4926 (mild pullback from 1.526 — τz routing stabilizing). Fastest EP5 convergence in this wave. Both mechanisms independently confirmed: 5L STRING (#745 val=6.5097%@EP30) and GradNorm α=0.5 (#740 test=7.5195% WAVE BEST). EP10 kill gate ≤8.0% is the next key checkpoint; if passes (very likely given EP5=7.03%), trajectory EP20 proj≈6.4-6.5%.

5. **Does GradNorm α=0.5 + Y-symmetry compose orthogonally? (frieren #791) — CLOSED, SUPERSEDED by #806.** EP13=6.9635% run best, run `g0um26ek`. EP10 gate CLEARED (7.0408%). w_τz=1.451 (α=0.5 over-routes to τz). Closed because α=0.25 + Y-sym (fern #794 EP11=6.8315%) was 0.132pp ahead with 2 fewer epochs — strong evidence α=0.25 is better for composition. Frieren reassigned to triple-compose #806 (5L STRING + GradNorm α=0.25 + Y-sym).

5b. **Triple compose: 5L STRING + GradNorm α=0.25 + Y-sym — EARLY BREAKOUT CONFIRMED (frieren #806, run=`gui4ceed`).** EP2=7.8887% (PR comment 2026-05-07T09:17Z). EP5+EP10 gates pre-cleared at EP2 (7.89% already below both ≤9.0% and ≤8.0% thresholds). Tied with nezuko at EP2 despite α=0.25 vs α=0.5 routing difference. GradNorm now entering adaptation phase (weights transitioning from near-uniform 0.88-1.18). Stacks all three wave-positive mechanisms: 5L STRING (val=6.5097%@EP30 standalone), Y-sym (test=7.8232%, reduces vol gap), GradNorm α=0.25 (wave val leader at EP15=6.7750% with Y-sym). Proj EP20-25 in 6.3-6.5% range if composition is orthogonal. Next key checkpoint: EP10 kill gate ≤8.0% (pre-cleared).

6. **Does GradNorm α=0.25 + Y-symmetry compose beat other combinations? (fern #794) — WAVE VAL LEADER.** EP16=6.7435% NEW RUN BEST, run `em7eupj5` (PR comment 2026-05-07T09:57Z). Monotonic channel improvement continues; τy and τz both -0.046pp strongest gains at EP16. WAVE VAL LEADER confirmed and extending lead. All gates pre-cleared by EP5. GradNorm weights: w_cp=0.789, w_τx=0.898, w_τy=0.952, w_τz=1.363, w_vol_p=0.998 — Y-sym relieves volume pressure vs tanjiro standalone (w_vol=2.351). α=0.25 + Y-sym confirmed most efficient 2-mechanism composition this wave. Full trajectory EP7→EP16 monotonically decreasing (except EP14 transient, resolved). Training to EP50 terminal; next checkpoint EP20.

7. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. Y-symmetry (#741) reduces gap via effective dataset doubling. GradNorm (#740) reduces vol error to 10.758% from 11.335%. No direct architectural fix yet tested. Vol still 2.77× surface gap.

8. **5L STRING confirmed valid but short of beating merged best (PR #745).** test=7.845% vs merged best 7.8232% — 0.023pp short. Val EP30=6.5097%. Compose 5L STRING + GradNorm or 5L STRING + Y-sym remains viable next hypothesis.

9. **Tanjiro compliance note:** PR #780 had unauthorized multi-arm screening; formal reprimand issued. Run `20n1fvwn` on authorized arm performing well; EP25=6.8334% wave val leader. Rebase clean, training to EP50 terminal.

## Potential Next Research Directions (after current arms complete)

**Currently in-flight (4 WIP PRs; all students occupied):**
- Fern #794: GradNorm α=0.25 + Y-symmetry — **EP16=6.7435% WAVE VAL LEADER** (run=`em7eupj5`), w_cp=0.789/w_τz=1.363/w_vol_p=0.998, all gates pre-cleared, EP20 proj≈6.64%, EP30 proj≈6.37%, training to EP50
- Tanjiro #780: GradNorm α=0.25 + baseline STRING — **EP33=6.7970% NEW RUN BEST** (run=`20n1fvwn`), EP34=7.3507% eval noise outlier, EP35 due ~10:45Z, SENPAI-RESULT terminal ~17:40Z, EP50 proj≈6.73%, w_vol=2.351 (heavy volume routing without Y-sym)
- Frieren #806: 5L STRING + GradNorm α=0.25 + Y-sym (triple compose) — **EP2=7.8887% EP5+EP10 GATES PRE-CLEARED** (run=`gui4ceed`), early breakout, tied with nezuko at EP2, GradNorm entering adaptation
- Nezuko #800: 5L STRING + GradNorm α=0.5 compose — **EP5=7.0322% ALL GATES CLEARED** (run=`hmhfnedy`), fastest EP5 in wave, w_τz=1.4926 stabilizing

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

_Last updated: 2026-05-07T10:30Z (fern #794 EP16=6.7435% NEW BEST WAVE VAL LEADER -0.032pp vs EP15, all channels monotonic; tanjiro #780 EP33=6.7970% NEW BEST, EP34=7.3507% eval noise outlier diagnosed, EP35 due ~10:45Z, terminal ~17:40Z; nezuko #800 EP5=7.0322% all gates cleared fastest EP5 in wave; frieren #806 EP2=7.8887% EP5+EP10 gates pre-cleared early breakout)_
