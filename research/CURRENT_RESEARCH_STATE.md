# SENPAI Research State

- 2026-05-05 (updated ~22:30 UTC)
- Most recent research direction from human researcher team: None (no open GitHub issues)

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA:** PR #599 (frieren, `sogus8sx`), test `abupt_axis_mean_rel_l2_pct` = **7.9303%**, val best = 6.5281%.

### Active Experiments (as of 2026-05-05 ~22:30 UTC)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #732 | dl24-tanjiro | STRING + QK-Norm at lr=5e-5 with 2000-step staged warmup | `1b8ew6mq` | EP5 PASSED at 8.5612% (gate ≤10.0% ✓). Next gate EP10 ≤8.0%. Long run at step ~31,932. |
| #740 | dl24-fern | GradNorm adaptive loss balancing (α=1.0 Arm A, α=0.5 Arm B) on SOTA Lion+STRING | Arm A: `50tejga5`; Arm B: TBD | Arm A long run step ~2,302 (pre-EP1). Smoke EP1=11.7564% (PASSED). Arm B (α=0.5) instructed by advisor — not yet confirmed started. Kill gates: EP5 ≤9.0%, EP10 ≤8.0%, EP20 ≤7.2%. |
| #741 | dl24-nezuko | Y-axis reflection augmentation on SOTA Lion+STRING config | `lszc4ri7` | Long run step ~2,028 (pre-EP1). Smoke EP1=13.9983% (PASSED). Y-symmetry flip functional (50.32% rate, tau_y sign-flip correct). Kill gates: EP5 ≤9.0%, EP10 ≤8.0%, EP20 ≤7.2%. |
| #745 | dl24-frieren | 5L STRING: add one Transolver layer (`--model-layers 5`) on SOTA base — pure CLI, zero code change | `pwdrbqli` (smoke) | Smoke step ~2,627 (pre-EP1). Critical PE bug fixed in advisor comment (added `--model-pe string_multisigma`). Two advisor reminders sent; no student response. Kill gates (upper-bound): EP5 ≥8.5%, EP10 ≥7.5%, EP20 ≥7.0%. |

### Closed / Negative Results This Wave

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #669 | dl24-frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) on SOTA base config | CLOSED (watchdog-killed 2026-05-05 ~22:26 UTC). EP33 best=6.7488% (EP31). Plateau 13+ epochs; EP35 gate ≤6.70% unachievable. |
| #667 | dl24-fern | Weight decay sweep WD={5e-4, 1e-3, 1e-4} on STRING SOTA | CLOSED NEGATIVE: vol gap WORSENS as WD decreases (2.80×→2.85×→2.94×). No arm beats SOTA. WD is NOT the lever for the volume generalization gap. |
| #730 | dl24-tanjiro | STRING + QK-Norm at lower LR=5e-5 (wave base config) | CLOSED: abandoned by student — zero W&B runs, zero PR comments. Student pivoted to PR #722 without authorization. QK-Norm at LR=5e-5 hypothesis reassigned to PR #732. |
| #696 | dl24-tanjiro | STRING + QK-Norm on SOTA Transolver base — L2-normalize Q,K per head in TransolverAttention | CLOSED: EP15 gate failure expected at ~7.47% (gate ≤7.2%). 7 compliance warnings, zero student response. |
| #673 | dl24-tanjiro | 7-sigma STRING PE [0.1..8.0] — expand sigma range | CLOSED: test=9.4198% (+1.49pp regression vs SOTA). Config mismatch (3L not 4L) confounds result, but slope deceleration makes clean re-run unlikely to beat SOTA. |
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

1. **Does QK-Norm at lr=5e-5 with staged warmup improve on SOTA? (tanjiro #732)** EP5 PASSED at 8.5612%. Encouraging trajectory — pre-wave `tkiigfmc` reached 8.625% test with old stack; if lower LR plus better base config closes the gap further, this direction has merit. EP10 ≤8.0% is the next gate.

2. **Does GradNorm adaptive balancing reduce the chronic vol→test gap? (fern #740)** Mechanism confirmed functional in smoke. The 3× vol→test gap is structural (confirmed by WD sweep #667). GradNorm's per-task gradient equalization is theoretically well-motivated for this imbalance. α=1.0 (Arm A) and α=0.5 (Arm B) provide two operating points.

3. **Does y-symmetry augmentation improve volume generalization? (nezuko #741)** Physics-valid 2× effective training set. tau_y sign-flip confirmed correct. Expected to help on volume (most under-represented) rather than surface. Smoke functional; long run pre-EP1.

4. **Does 5L STRING add the same gain as 4L STRING did? (frieren #745)** Pre-wave `70lnb3dt` test=8.769%. 3→4L was +0.549pp. 4→5L pure CLI, zero code. Model grows ~12.93M→~16M params. Smoke running; critical PE flag bug fixed before launch.

5. **Volume val→test gap (3×) remains the central unsolved problem.** WD sweep (#667) definitively closed WD as a lever. Current candidates: GradNorm (#740), y-symmetry (#741), Volume MLP head (unassigned), DualTower (#722 closing).

6. **Tanjiro compliance track:** PR #730 abandoned, PR #696 closed, PR #673 closed, PR #732 showing inconsistent execution (staged warmup without explicit advisor OK). Monitor closely; PR #732 is allowed to run to completion given EP5 PASSED but further assignments require strict gate-compliance acknowledgment.

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
