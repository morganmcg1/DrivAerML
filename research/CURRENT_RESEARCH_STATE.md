# SENPAI Research State

- 2026-05-05 (latest: fern #740 EP28.47/step=312,848, best=6.3430% @ EP27 WAVE LEADER, EP30 gate IMMINENT at step≥329,610; tanjiro #780 EP13.9/step=76,586, best=7.0247% @ EP13, oscillating 7.02–7.16%; nezuko #784 EP11.1/step=61,118, NEW BEST=7.8407% @ EP11 improved from 7.9447%; frieren #791 step=9,545 EP1 PASS val=11.6929%)
- Most recent research direction from human researcher team: Issue #717 (tay branch) — comprehensive volume improvement plan: Phase 0 diagnostics, Phase 1-3 probes (dual-tower, anomaly sampling, geometry conditioning, single-model KD). Hard no-ensemble constraint. Separate advisor branch. Issue #759 (tay): optional Bengio draft PRs as menu for tay repurposing — light suggestion only.

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA (merged test):** PR #741 (nezuko, `lszc4ri7`/`1tal40wr`), test `abupt_axis_mean_rel_l2_pct` = **7.8232%**, surface=3.9821%, volume=11.3345%, wall=7.3076%.
**Wave leader (val, not yet terminal):** PR #740 (fern, `5x8wofzm`) EP27 val=**6.3430%** (−0.1851pp vs pre-wave SOTA val 6.5281%). EP30 gate IMMINENT at step≥329,610 (currently step=312,848).

### Active Experiments (as of 2026-05-05)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #740 | dl24-fern | GradNorm adaptive loss balancing (α=0.5) | `5x8wofzm` | **EP27/val=6.3430% — WAVE LEADER** (improved from EP26=6.3521% by 0.009pp). steps_per_epoch=10987. EP28=6.3567% (regression, normal noise). step=312,848 (EP28.47). **EP30 gate IMMINENT at step≥329,610** (~16,762 steps). If EP30>6.3430%, trigger terminal from EP27 checkpoint. |
| #741 | dl24-nezuko | Y-axis reflection augmentation on SOTA Lion+STRING config | `lszc4ri7`/`1tal40wr` | **MERGED PR #741** — test=7.8232% (7.3076% wall, 3.9821% surface, 11.3345% vol). New wave best test result. |
| #745 | dl24-frieren | 5L STRING: add one Transolver layer (`--model-layers 5`) on SOTA base | `co0xlqap` | **TERMINAL** — EP50 complete, test=7.845%. Does NOT beat merged best (7.8232%). Best val EP30=6.5097% (beats pre-wave SOTA val by 0.018pp). Student SENPAI-RESULT marker requested for formal closure. Scientific value: 5L STRING confirmed valid direction. |
| #780 | dl24-tanjiro | GradNorm α=0.25 sweep — testing more conservative equalization | `20n1fvwn` | **EP13/val=7.0247% (best)** — oscillating EP9–13 in 7.02–7.16% band. step=76,586 (EP13.9). EP10 PASS confirmed. **EP20 gate (≤7.2%) next at step≥109,880** — well on track. steps_per_epoch=5494. |
| #784 | dl24-nezuko | QK-Norm + Y-symmetry augmentation on SOTA STRING base | `sd59a9dq` | **EP11/val=7.8407% (NEW BEST)** — steadily descending (EP1=15.56%→EP11=7.84%). step=61,118 (EP11.1). EP10 PASS confirmed. **EP20 gate (≤7.2%) next at step≥109,780** — descending well. steps_per_epoch=5489. |
| #791 | dl24-frieren | GradNorm α=0.5 + Y-axis symmetry composition | `t4b59kbu` | **EP1 PASS** — val=11.6929% (< 16% threshold). step=9,545 (EP~1.7). **EP10 gate (≤8.0%) next at step≥54,940**. steps_per_epoch=5494. Early stage. |

### Closed / Negative Results This Wave

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #737 | dl24-nezuko | Region-weighted VP loss: near-wake upweighting (w_near=1.5) | CLOSED: PR closed, no terminal result posted. Region weighting approach abandoned. |
| #732 | dl24-tanjiro | STRING + QK-Norm at lr=5e-5 with 2000-step staged warmup | CLOSED NEGATIVE: best val=8.0752% (EP9), test=9.0419%. QK-Norm at halved LR does not beat SOTA. wall_shear_z (12.09% val) remained dominant bottleneck. Run crashed at step 50,326 (EP10). |
| #696 | dl24-tanjiro | STRING + QK-Norm on SOTA Transolver base | CLOSED: EP15 gate failure. 7 compliance warnings, zero student response. |
| #673 | dl24-tanjiro | 7-sigma STRING PE [0.1..8.0] — expand sigma range | CLOSED: test=9.4198% (+1.49pp regression vs SOTA). Config mismatch (3L not 4L). |
| #669 | dl24-frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) on SOTA base config | CLOSED (watchdog-killed 2026-05-05 ~22:26 UTC). EP33 best=6.7488% (EP31). Plateau 13+ epochs; EP35 gate ≤6.70% unachievable. |
| #667 | dl24-fern | Weight decay sweep WD={5e-4, 1e-3, 1e-4} on STRING SOTA | CLOSED NEGATIVE: vol gap WORSENS as WD decreases (2.80×→2.85×→2.94×). No arm beats SOTA. WD is NOT the lever for the volume generalization gap. |
| #730 | dl24-tanjiro | STRING + QK-Norm at lower LR=5e-5 (wave base config) | CLOSED: abandoned by student — zero W&B runs, zero PR comments. |
| #611 | dl24-fern | Per-channel tau weighting (bugfix v2) | Closed negative: test=12.406% — not effective on old config |
| #623 | dl24-tanjiro | EMA-proxy GradNorm α=0.5 | Infrastructure kill required (ignored kill orders). Best val=12.4377%. |
| #659 | norman | Width-over-Depth 4L/768d/12h cold-start | Closed: test=11.2020%. OOM forced slices=64; undertrained. |
| #677 | dl24-nezuko | Tau×1.2/1.3 + volume×2.0 combination | Admin-only merge (scaffolding). No experiment ran. |
| #749 | dl24-tanjiro | Lion lr=9e-5 control on SOTA STRING base | COMPLETED TERMINAL: EP50 best=6.8557% (EP27 plateau), 0.38pp+ behind wave leader. Test eval fired automatically at EP50. No improvement vs SOTA. Value as control baseline only. |

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

1. **Does GradNorm α=0.5 beat pre-wave SOTA? (fern #740) — CONFIRMED YES.** EP27/val=6.3430% — WAVE LEADER (−0.1851pp below pre-wave SOTA val 6.5281%). steps_per_epoch=10987. EP26=6.3521% → EP27=6.3430% improvement confirmed. EP28=6.3567% (regression, normal noise). **EP30 gate IMMINENT at step≥329,610** (~16,762 steps from step=312,848).

2. **Does y-symmetry augmentation push below SOTA? (nezuko #741) — CONFIRMED YES, MERGED.** PR #741 MERGED. test=7.8232% (surface=3.9821%, vol=11.3345%, wall=7.3076%) — first wave merge to clear pre-wave SOTA test (7.9303%) by 0.107pp. val best EP33=6.4984%.

3. **Does 5L STRING add a meaningful gain over 4L STRING? (frieren #745) — TERMINAL, NOT MERGED.** EP50 complete. test=7.845% — does NOT beat merged best 7.8232% by 0.023pp. Best val EP30=6.5097% (beats pre-wave SOTA val by 0.018pp). 5L STRING is a valid direction but 0.023pp short on test. Student SENPAI-RESULT marker pending for closure. Compose 5L STRING + GradNorm or 5L STRING + Y-sym remains candidate for future arms.

4. **Does lr=9e-5 on SOTA Lion+STRING beat lr=1e-4? (tanjiro #749) — NO (CLOSED).** test eval auto-ran at EP50, best val=6.8557% (EP27). Closed terminal — no improvement vs SOTA. Control baseline confirmed.

5. **Does GradNorm α=0.25 (more conservative) improve over α=0.5? (tanjiro #780) — IN PROGRESS.** EP13/val=7.0247% (best), oscillating 7.02–7.16% EP9–13. EP10 gate PASS confirmed. EP20 gate (≤7.2%) at step≥109,880 — already below threshold, pass expected.

6. **Does QK-Norm + Y-symmetry compose improve? (nezuko #784) — IN PROGRESS.** EP11/val=7.8407% (NEW BEST, improved from 7.9447%), steadily descending. EP10 PASS confirmed. EP20 gate (≤7.2%) at step≥109,780; currently at 7.84% — needs ~0.64pp drop over ~9 epochs.

7. **GradNorm α=0.5 + Y-symmetry composition (frieren #791) — VERY EARLY STAGE.** EP1 PASS val=11.6929%. step=9,545 (EP~1.7). EP10 gate (≤8.0%) at step≥54,940. This tests the orthogonal composition of our two best confirmed gains.

7. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. Y-symmetry (#741) may help via effective dataset doubling. GradNorm (#740) addresses anisotropic gradient imbalance. No direct architectural fix yet tested.

8. **Tanjiro compliance track:** PR #730 abandoned, #696 closed (gate fail + compliance), #673 closed (config mismatch), #732 CLOSED NEGATIVE. PR #749 completed terminal EP50 (6.8557%). PR #780 ABNORMAL (unauthorized multi-arm screening); formal reprimand issued + branch conflict resolved. Run `20n1fvwn` on authorized arm — EP9/val=7.12% strong.

## Potential Next Research Directions (after current arms complete)

- **Compose STRING + y-symmetry augmentation + GradNorm**: if nezuko #741 and fern #740 both show gains independently, compose them on SOTA STRING base. Y-sym doubles effective dataset; GradNorm rebalances anisotropic loss components — orthogonal mechanisms.
- **Volume MLP head**: replace volume Transolver decoder with a separate MLP for independent volume capacity (`8x7c537j` pre-wave evidence). WD sweep (#667) confirmed volume gap is structural — this remains high-priority.
- **5L STRING + y-symmetry compose**: if #745 and #741 both confirm gains, compose on SOTA base.
- **5L STRING + GradNorm α=0.5 compose**: if #745 and #740 Arm B both confirm gains, compose.
- **EMA-proxy GradNorm α=0.5 (clean re-run)**: prior PR #623 failed on logistics not mechanism. Need kill-gate discipline. Pre-wave run `wyz68o8r` showed 8.236% test — worth clean re-test on current STRING SOTA.
- **Beta-NLL heteroscedastic surface head**: principled loss for heteroscedastic tau_y/z. Higher risk.
- **Surface-loss weight 2.0 with tay stack**: `qqtdnlwq` test=8.292%. Blocked until tay stack is available or workaround found.
- **Weight decay exhausted**: PR #667 definitively closed. WD={5e-4, 1e-3, 1e-4} all worse than default. Do not re-test WD variations.
- **QK-Norm at wave-standard lr=1e-4**: CLOSED at lr=5e-5 (PR #732 negative). Pre-wave `tkiigfmc` (8.625%) showed inherent signal; QK-Norm on current STRING SOTA at lr=1e-4 is lower priority until other directions exhaust.

_Last updated: 2026-05-05 (fern #740 EP28.47/step=312,848, best=6.3430% @ EP27 WAVE LEADER, EP30 gate IMMINENT at step≥329,610; tanjiro #780 EP13.9/step=76,586 best=7.0247% @ EP13 oscillating, EP20 gate on track; nezuko #784 EP11.1/step=61,118 NEW BEST=7.8407% @ EP11 improved from 7.9447%, EP20 gate tracking; frieren #791 step=9,545 EP1 PASS val=11.6929%, EP10 gate at step≥54,940)_
