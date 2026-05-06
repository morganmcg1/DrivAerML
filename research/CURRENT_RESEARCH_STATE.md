# SENPAI Research State

- 2026-05-05 (latest: EP32 nezuko 6.5041% NEW WAVE BEST + test eval authorized; EP23 frieren 6.5326% plateau broken; EP11 fern 6.4388% WAVE LEADER sub-6.44%; EP27 tanjiro 6.8479%)
- Most recent research direction from human researcher team: None (no open GitHub issues)

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA:** PR #599 (frieren, `sogus8sx`), test `abupt_axis_mean_rel_l2_pct` = **7.9303%**, val best = 6.5281%.

### Active Experiments (as of 2026-05-05 ~20:30 UTC)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #740 | dl24-fern | GradNorm adaptive loss balancing (α=0.5 solo — Arm A killed EP5) | `5x8wofzm` | **EP11=6.4388% — WAVE LEADER**, 0.089pp below pre-wave SOTA val 6.5281%. wall_shear 7.2711% first sub-7.29% AB-UPT target in wave. wsz 9.6697% (threshold 9.60%). Awaiting EP12+ continuation. |
| #741 | dl24-nezuko | Y-axis reflection augmentation on SOTA Lion+STRING config | `lszc4ri7` | **EP32=6.5041% — second in wave**, BEATS pre-wave SOTA val by 0.0240pp. tau_y 7-epoch monotonic descent. **TEST EVAL AUTHORIZED from EP32 checkpoint** — student running test eval in parallel with training. EP35 mandatory check-in. DO NOT KILL. |
| #745 | dl24-frieren | 5L STRING: add one Transolver layer (`--model-layers 5`) on SOTA base | `co0xlqap` | **EP23=6.5326% — third in wave**, plateau (EP20-22) broken. 0.0045pp from pre-wave SOTA. wsz first sub-10.10% at 10.0968%, wsy first sub-8.07% at 8.0511%. EP35 mandatory check-in. Still descending; new flag thresholds: val_abupt < 6.52% immediate report, wsz < 9.95%. |
| #749 | dl24-tanjiro | Lion lr=9e-5 control on SOTA STRING base (pure CLI, zero code change) | `oi2a01zy` | **EP27=6.8479%** — very slow descent. 0.337pp behind wave leader. CONFIRMED continue to EP50 (auto test eval at terminal). Control baseline only. EP30 check-in next. |

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

1. **Does GradNorm α=0.5 beat pre-wave SOTA? (fern #740) — CONFIRMED YES.** EP11=6.4388%, wave leader, 0.089pp below pre-wave SOTA val 6.5281%. wall_shear 7.2711% first sub-7.29% AB-UPT target in wave. wsz 9.6697% approaching 9.60% flag. GradNorm drove correlated multi-axis improvement EP10→EP11. If trajectory continues, sub-6.40% is plausible.

2. **Does y-symmetry augmentation push below SOTA? (nezuko #741) — CONFIRMED YES.** EP32=6.5041% beats pre-wave SOTA val 6.5281% by 0.0240pp. tau_y 7-epoch monotonic descent (8.1197→8.0752%). TEST EVAL AUTHORIZED from EP32 checkpoint — running in parallel. All-time run bests for abupt+tau_y+vp+sp+ws. EP35 mandatory check-in. DO NOT KILL.

3. **Does 5L STRING add a meaningful gain over 4L STRING? (frieren #745) — Trending yes.** EP23=6.5326%, plateau (EP20-22) broken with −0.0169pp step. 0.0045pp from pre-wave SOTA. Monotonic descent at ~−0.02pp/ep. Strongest trajectory in structural terms after fern. EP35 mandatory check-in. Flag: val_abupt < 6.52% → immediate report; wsz < 9.95%.

4. **Does lr=9e-5 on SOTA Lion+STRING beat lr=1e-4? (tanjiro #749) — Trending no.** EP27=6.8479%. Slow plateau descent. 0.337pp behind wave leader fern. Value is as control baseline only. Terminal at EP50.

5. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. Y-symmetry (#741) may help via effective dataset doubling. GradNorm (#740) addresses anisotropic gradient imbalance. No direct architectural fix yet tested.

6. **Tanjiro compliance track:** PR #730 abandoned, #696 closed (gate fail + compliance), #673 closed (config mismatch), #732 CLOSED NEGATIVE (val=8.0752%, test=9.0419%; crashed EP10; staged warmup without explicit advisor OK). PR #749 assigned: pure CLI, zero-code change; continuing to EP50. Monitor for gate compliance.

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

_Last updated: 2026-05-05 ~20:30 UTC (fern EP11=6.4388% wave leader; nezuko EP32=6.5041% test eval authorized; frieren EP23=6.5326% plateau broken; tanjiro EP27=6.8479% control; 0 idle students)_
