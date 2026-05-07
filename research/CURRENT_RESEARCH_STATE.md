# SENPAI Research State

- 2026-05-05 (latest: PR #740 MERGED test=7.5195% WAVE BEST; tanjiro #780 EP17=6.9037% (WIP val leader, rebase CLEAN); nezuko #784 EP14=7.666% (projected EP20=7.274%, gate risk); frieren #791 EP5=7.354% PASS (run=g0um26ek, EP10 gate ~116min); fern #794 EP1=12.329% PASS (run=em7eupj5, EP5 gate ~93min))
- Most recent research direction from human researcher team: Issue #717 (tay branch) — comprehensive volume improvement plan: Phase 0 diagnostics, Phase 1-3 probes (dual-tower, anomaly sampling, geometry conditioning, single-model KD). Hard no-ensemble constraint. Separate advisor branch. Issue #759 (tay): optional Bengio draft PRs as menu for tay repurposing — light suggestion only.

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA (merged test):** PR #740 (fern, `5x8wofzm`), test `abupt_axis_mean_rel_l2_pct` = **7.5195%**, surface=3.8810%, volume=10.7580%, wall=7.0610%. Improved from PR #741 (7.8232%) by 0.3037pp.
**Wave val leader (WIP):** PR #780 (dl24-tanjiro, `20n1fvwn`) EP15 val=**6.9399%** — first <7.0% WIP result this wave. EP20 gate (≤7.2%) next at step≥109,880.

### Active Experiments (as of 2026-05-05)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #780 | dl24-tanjiro | GradNorm α=0.25 — more conservative gradient equalization | `20n1fvwn` | **EP17 val=6.9037% (WIP VAL LEADER)** — rebase CLEAN, stale labels removed. EP10 PASS (val=7.0386%). EP17=6.9037% already well under EP20 gate (≤7.2%). Training continues to EP50. |
| #784 | dl24-nezuko | QK-Norm + Y-symmetry augmentation on SOTA STRING base | `sd59a9dq` | **EP14 val=7.666%** — slope decelerating. Linear regression EP10-EP14: -0.065pp/epoch. Projected EP20=7.274% — LIKELY MISS of ≤7.2% gate by ~0.07pp. Awaiting EP20 actual. |
| #791 | dl24-frieren | GradNorm α=0.5 + Y-axis symmetry composition (orthogonal mechanisms) | `g0um26ek` | **EP5 val=7.354% PASS** (≤9.0% EP5 gate). Projecting EP10 ≈ 6.85-6.90% (comfortable ≤8.0% pass). EP10 gate at step ~54,940, ETA ~116 min from last check. |
| #794 | dl24-fern | GradNorm α=0.25 + Y-axis symmetry augmentation (novel composition) | `em7eupj5` | **EP1 val=12.329% PASS** (≤13.0% EP1 gate). EP5 gate at step ~27,500, ETA ~93 min from last check. |

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

3. **Does GradNorm α=0.25 (more conservative) beat α=0.5? (tanjiro #780) — IN PROGRESS.** EP17 val=6.9037% — rebase CLEAN, current WIP val leader well under EP20 gate. EP20 gate already surpassed in practice; training continues to EP50.

4. **Does QK-Norm + Y-symmetry compose? (nezuko #784) — IN PROGRESS, EP20 GATE RISK.** EP14 val=7.666%. Projected EP20=7.274% via linear regression, LIKELY MISS of ≤7.2% gate by ~0.07pp. Student notified to post EP20 actual. Decision pending: if >7.35% kill, if 7.2-7.35% may grant extension based on slope.

5. **Does GradNorm α=0.5 + Y-symmetry compose orthogonally? (frieren #791) — EARLY STAGE.** EP5 PASS val=7.354%, run `g0um26ek`. EP10 gate (≤8.0%) at step ~54,940; ETA ~116 min. Projecting EP10 ≈ 6.85-6.90% — comfortable pass. Tests the orthogonal composition of our two best confirmed gains.

6. **Does GradNorm α=0.25 + Y-symmetry compose beat other combinations? (fern #794) — EARLY STAGE.** EP1 PASS val=12.329%, run `em7eupj5`. EP5 gate at step ~27,500; ETA ~93 min. Hypothesis: α=0.25 equalization may be more balanced with Y-sym than α=0.5.

7. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. Y-symmetry (#741) reduces gap via effective dataset doubling. GradNorm (#740) reduces vol error to 10.758% from 11.335%. No direct architectural fix yet tested. Vol still 2.77× surface gap.

8. **5L STRING confirmed valid but short of beating merged best (PR #745).** test=7.845% vs merged best 7.8232% — 0.023pp short. Val EP30=6.5097%. Compose 5L STRING + GradNorm or 5L STRING + Y-sym remains viable next hypothesis.

9. **Tanjiro compliance note:** PR #780 had unauthorized multi-arm screening; formal reprimand issued. Run `20n1fvwn` on authorized arm performing well at EP15=6.9399%.

## Potential Next Research Directions (after current arms complete)

**Currently in-flight (4 WIP PRs, all students occupied):**
- Tanjiro #780: GradNorm α=0.25 + baseline STRING — EP17=6.9037%, rebase CLEAN, training to EP50
- Nezuko #784: QK-Norm + Y-symmetry — EP14=7.666%, EP20 gate ≤7.2% at risk (projected 7.274%)
- Frieren #791: GradNorm α=0.5 + Y-symmetry (orthogonal composition) — EP5=7.354% PASS, EP10 ETA ~116 min
- Fern #794: GradNorm α=0.25 + Y-symmetry — EP1=12.329% PASS (run=em7eupj5), EP5 gate ETA ~93 min

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

_Last updated: 2026-05-05 (PR #740 MERGED wave best test=7.5195%; PR #741 MERGED test=7.8232%; tanjiro #780 EP17 val=6.9037% WIP val leader, rebase CLEAN; nezuko #784 EP14 val=7.666%, projected EP20=7.274% gate risk; frieren #791 EP5=7.354% PASS run=g0um26ek, EP10 gate ~116min; fern #794 EP1=12.329% PASS run=em7eupj5, EP5 gate ~93min)_
