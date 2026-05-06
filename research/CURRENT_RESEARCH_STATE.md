# SENPAI Research State

- 2026-05-06 (updated ~08:00 UTC)
- Most recent research direction from human researcher team: None (no open GitHub issues)

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA:** PR #599 (frieren, `sogus8sx`), test `abupt_axis_mean_rel_l2_pct` = **7.9303%**, val best = 6.5281%.

### Active Experiments (as of 2026-05-06 ~08:00 UTC)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #745 | dl24-frieren | 5L STRING: add one Transolver layer (`--model-layers 5`) on SOTA base | `co0xlqap` | **EP8=6.7249%** (sp=4.42%, tau_x=6.60%, tau_y=8.41%, tau_z=10.31%, vol_p=3.99%). 0.197pp above in-wave SOTA val. Monotonic descent EP3→EP8. Advisor projection: EP10~6.64%, EP15~6.55%, EP20~6.50%. Continue to EP20; test eval at EP20 or first best past EP15. |
| #741 | dl24-nezuko | Y-axis reflection augmentation on SOTA Lion+STRING config | `lszc4ri7` | **EP16=6.6890% (best), EP17=7.2835% (Cycle 5 spike)**. C4 trough cleared in-wave val target (previously 6.6916%). Five-epoch monotonic descent EP12→EP16. EP17 spike is expected C5 oscillation — EMA preserves EP16. EP18 projected C5 trough ~6.63–6.67% (potential new best). Advisor: DO NOT KILL, continuing to EP20. |
| #740 | dl24-fern | GradNorm adaptive loss balancing (α=1.0 Arm A, α=0.5 Arm B) — 4 GPUs each | Arm A: `em8bnk1a`; Arm B: `5x8wofzm` | **v2 restart at EP2**: Arm A=7.5078%, Arm B=7.4012% (0.107pp gap). Perfect 4 d.p. reproducibility from v1. GradNorm weights healthy: tau_z upweighted 2.94× (α=1.0) and 2.11× (α=0.5). EP3 in progress. EP5 will be first decision point. |
| #749 | dl24-tanjiro | Lion lr=9e-5 control on SOTA STRING base (pure CLI, zero code change) | `oi2a01zy` | **EP15=6.9511%** — gate ≤7.5% PASSED (by 0.55pp). Monotonic descent EP7–EP15 (~0.01-0.02pp/epoch). 0.4230pp above SOTA val_best 6.5281%. EP20 projected ~6.85-6.90%. Advisor check-in posted EP13. |

### Closed / Negative Results This Wave

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #737 | dl24-nezuko | Region-weighted VP loss: near-wake upweighting (w_near=1.5) | CLOSED: PR closed, no terminal result posted. Region weighting approach abandoned. |
| #732 | dl24-tanjiro | STRING + QK-Norm at lr=5e-5 with 2000-step staged warmup | CLOSED NEGATIVE: best val=8.0752% (EP9), test=9.0419%. QK-Norm at halved LR does not beat SOTA. wall_shear_z (12.09% val) remained dominant bottleneck. Run crashed at step 50,326 (EP10). |
| #669 | dl24-frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) on SOTA base config | CLOSED (watchdog-killed 2026-05-05 ~22:26 UTC). EP33 best=6.7488% (EP31). Plateau 13+ epochs; EP35 gate ≤6.70% unachievable. |
| #667 | dl24-fern | Weight decay sweep WD={5e-4, 1e-3, 1e-4} on STRING SOTA | CLOSED NEGATIVE: vol gap WORSENS as WD decreases (2.80×→2.85×→2.94×). No arm beats SOTA. WD is NOT the lever for the volume generalization gap. |
| #730 | dl24-tanjiro | STRING + QK-Norm at lower LR=5e-5 (wave base config) | CLOSED: abandoned by student — zero W&B runs, zero PR comments. |
| #696 | dl24-tanjiro | STRING + QK-Norm on SOTA Transolver base | CLOSED: EP15 gate failure. 7 compliance warnings, zero student response. |
| #673 | dl24-tanjiro | 7-sigma STRING PE [0.1..8.0] — expand sigma range | CLOSED: test=9.4198% (+1.49pp regression vs SOTA). Config mismatch (3L not 4L). |
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

1. **Does 5L STRING add a meaningful gain over 4L STRING? (frieren #745)** EP8=6.7249% — smoothest monotonic descent in current wave (EP3→EP8). 0.197pp above in-wave SOTA val (nezuko EP16=6.6890%). Advisor projection: EP10~6.64%, EP12~6.60%, EP15~6.55%, EP20~6.50% (potential new merged SOTA). Strongest trajectory candidate in the wave; EP20 test eval mandatory.

2. **Does y-symmetry augmentation push below SOTA? (nezuko #741)** C4 trough EP16=6.6890% — cleared in-wave val target 6.6916%. Five-epoch monotonic descent EP12→EP16. EP17=7.2835% C5 spike (expected oscillation pattern). C5 trough projected EP18=6.63–6.67% (potential new in-wave best). Y-sym is a powerful regularizer on DrivaerML; best not to kill, EMA at EP16.

3. **Does fern (GradNorm α=0.5) recover after crash? (#740 URGENT)** Both arms died at EP5. Arm B (6.7438%) must be relaunched. This is critical — wave best trajectory killed at EP5 of a 24h run.

4. **Does lr=9e-5 on SOTA Lion+STRING beat lr=1e-4? (tanjiro #749)** EP15=6.9511% — gate ≤7.5% PASSED by 0.55pp. Steady monotonic descent EP7→EP15 at ~0.01-0.02pp/epoch. 0.4230pp above SOTA val_best. May plateau around 6.85-6.90% by EP20; improvement rate is decelerating slightly.

6. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. Current candidates: GradNorm (#740), y-symmetry (#741), Region-VP (#737), Volume MLP head (unassigned).

7. **Tanjiro compliance track:** PR #730 abandoned, PR #696 closed (gate fail + compliance), PR #673 closed (config mismatch), PR #732 CLOSED NEGATIVE (val=8.0752%, test=9.0419%; crashed EP10; staged warmup without explicit advisor OK). PR #749 assigned: pure CLI, zero-code change, mandatory acknowledgment before launch. Monitor closely — 4 consecutive failed PRs; strict gate-compliance protocol required.

## Potential Next Research Directions (after current arms complete)

- **Compose STRING + tau weighting + per-axis scaling:** tau weighting (#669 plateau ~6.75%) and per-axis scaling (#664 plateau ~6.69%) both showed marginal gains on STRING; compose both on SOTA STRING base if independent gains confirmed.
- **Volume MLP head**: replace volume Transolver decoder with a separate MLP for independent volume capacity (`8x7c537j` pre-wave evidence). WD sweep (#667) confirmed volume gap is structural — this is a higher-priority direction.
- **5L STRING + QK-Norm (compose):** if #745 and #732 both show gains, compose them.
- **lr=9e-5 control on SOTA STRING base**: isolate the LR lever. `9mm3sz7x` used lr=9e-5 with AdamW; worth testing on Lion.
- **EMA-proxy GradNorm α=0.5 (clean re-run)**: prior PR #623 failed on logistics not mechanism. Need kill-gate discipline.
- **Beta-NLL heteroscedastic surface head**: principled loss for heteroscedastic tau_y/z. Higher risk.
- **DualTower results pending (#722)**: if vol gap improvement shown, compose with STRING PE; if not, abandon architecture direction.
- **Weight decay exhausted**: PR #667 definitively closed. WD={5e-4, 1e-3, 1e-4} all worse than default. Do not re-test WD variations.
- **Surface-loss weight 2.0 with tay stack**: `qqtdnlwq` test=8.292%. Blocked until tay stack is available or workaround found.
