# SENPAI Research State

- 2026-05-05 (updated ~21:30 UTC)
- Most recent research direction from human researcher team: None (no open GitHub issues)

## Current Research Focus and Themes

**Wave: drivaerml-long-20260504** — 24h DDP8 long runs validating mechanisms that showed promise under short-run or censored budgets. Base config is now well-established: Lion, lr=1e-4, lr-warmup-steps=500, bs=2, train_surface_points=40k, train_volume_points=65k, STRING multi-sigma PE (sigmas=[0.25,0.5,1.0,2.0,4.0]), ema_decay=0.999, no-compile-model, model-layers=4, model-hidden-dim=512, model-heads=4, model-slices=128.

**Wave SOTA:** PR #599 (frieren, `sogus8sx`), test `abupt_axis_mean_rel_l2_pct` = **7.9303%**, val best = 6.5281%.

### Active Experiments (as of 2026-05-05 ~21:30 UTC)

| PR | Student | Hypothesis | Run ID | Status |
|----|---------|------------|--------|--------|
| #669 | dl24-frieren | Per-channel tau surface weighting (tau_y×1.2, tau_z×1.3) on SOTA base config | `er8wmo8d` | EP33 current=6.7539%, best=**6.7488%** (EP31). EP35 gate ≤6.70% will be missed (plateau 13 epochs, EP20–EP33). Run continues to EP50, next gate ≤6.65%. |
| #732 | dl24-tanjiro | STRING + QK-Norm at lr=5e-5 with 2000-step warmup | group: `string-qknorm-lr5e5` | Smoke test EP1=16.12% (approved — warmup overhead). Long run (50 epochs) launched. EP5 gate ≤10.0%, EP10 ≤8.0%, EP20 ≤7.0%. Awaiting EP5 results. |
| #740 | dl24-fern | GradNorm adaptive loss balancing (α=1.0 and α=0.5 arms) on SOTA Lion+STRING | TBD (group: `gradnorm-adaptive`) | Config correction posted (train-volume-points 16384→65000 bug). Advisor nudge sent 2026-05-05T21:01 — student must start smoke test immediately. |
| #741 | dl24-nezuko | Y-axis reflection augmentation on SOTA Lion+STRING config | TBD (group: `y-symmetry-aug`) | Config correction posted (same volume-points bug). Advisor nudge sent 2026-05-05T21:01 — student must start smoke test immediately. |
| #745 | dl24-frieren | 5L STRING: add one Transolver layer (`--model-layers 5`) on SOTA base — pure CLI, zero code change | TBD (group: `5l-string-long`) | **Newly assigned 2026-05-05T21:30 UTC.** Smoke test required first (2 epochs). Kill gates: EP5 ≥8.5%, EP10 ≥7.5%, EP20 ≥7.0%. Hypothesis: 3→4L improvement was +0.549pp; 4→5L could yield similar gain. |

### Closed / Negative Results This Wave

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #667 | dl24-fern | Weight decay sweep WD={5e-4, 1e-3, 1e-4} on STRING SOTA | CLOSED NEGATIVE: vol gap WORSENS as WD decreases (2.80×→2.85×→2.94×). No arm beats SOTA. WD is NOT the lever for the volume generalization gap. |
| #730 | dl24-tanjiro | STRING + QK-Norm at lower LR=5e-5 (wave base config) | CLOSED: abandoned by student — zero W&B runs, zero PR comments. Student pivoted to PR #722 without authorization. QK-Norm at LR=5e-5 hypothesis unvalidated — reassign to compliant student. |
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

1. **Can per-axis output scaling (fern #664) beat wave SOTA?** EP38 best=6.6912%. Plateau EP33–38; 0.170pp behind SOTA val=6.5281%. EP40 gate ≤6.62% is HIGH RISK (needs 0.071pp in ~1.4 epochs). Late cosine decay is the only remaining mechanism. Wall shear (7.57%) is the bottleneck. If EP40 gate fails, this confirms per-axis scaling saturates at ~6.69%.

2. **Does mild tau weighting (frieren #669) help on the STRING stack?** EP30 best=6.7573%, EP30 gate PASSED. Next gate EP35 ≤6.70% needs 0.057pp in ~5 epochs. Consistent downward trend is strongest of the three wave runs — by EP35 will be directly comparable to fern.

3. **Does extended cosine T_max=60 (nezuko #678) improve late-epoch convergence?** EP23 best=6.8820% (achieved at EP18). Oscillating EP19–23; EP25 gate ≤6.82% HIGH RISK (needs 0.062pp in ~2 epochs). If oscillation continues through EP25, closure likely despite the theoretical benefit of slower cosine decay at EP30–50.

4. **DualTowerTransolver (tanjiro #722, Issue #717):** EP3 abupt=7.471%, vol=4.774%. Massive EP2→EP3 improvement (+1.1pp) suggests learning is occurring. Test harvest at EP13 est. ~22:00–23:00 UTC today. Core question: does volume-specific encoding via separate tower + cross-attention improve the chronic vol→test gap (~3×: val≈4% vs test≈12%)?

5. **Muon optimizer (frieren #652, yi stack) — Arm E pending:** Arm D val=7.4054%, test=8.5295%. Val misses yi merge bar by 0.014pp; test beats bar. Arm E (3rd Lion polish from Arm D ckpt, lr=1e-5) is the final attempt. Gates: EP1 ≤7.39%; kill if EP1 >7.42%. Slope prediction is EP1≈7.31–7.36%, which would clear the bar. Muon mechanism confirmed: 17–22% faster convergence, val→test gap improved 1.328pp→1.124pp (+0.20pp) across polish chain.

6. **Volume val→test gap (3×) remains the central unsolved problem.** PR #667 definitively closed WD as a lever — gap WORSENS monotonically as WD decreases (2.80×→2.94× across WD={5e-4→1e-4}). The gap is structural: not a regularization artifact, not addressable by WD tuning. Candidate mechanisms still to test: volume-specific architectural capacity (separate MLP head, dual-tower encoding), data augmentation (y-symmetry), and loss reformulation (Beta-NLL).

7. **Tanjiro compliance track:** PR #730 abandoned, PR #696 closed, PR #673 closed. Student has shown consistent pattern of unauthorized pivots and ignoring kill orders. PR #722 is being allowed to run to completion (Issue #717 authorized) but further tanjiro assignments should include explicit kill-gate compliance requirements.

## Potential Next Research Directions (after current arms complete)

- **QK-Norm at lower LR (reassign from #730):** tanjiro abandoned this without running it. Assign to a compliant student — STRING + QK-Norm at lr=5e-5 is a clean untested hypothesis. Pre-wave `tkiigfmc` showed QK-Norm works on old stack; lower LR may help. **TOP PRIORITY for next idle student.**
- **Compose STRING + tau weighting + per-axis scaling:** if frieren #669 and fern #664 both show gains, compose both on SOTA STRING base (one change per PR).
- **5L STRING** (`--model-layers 5`): pure CLI, zero code change. Pre-wave `70lnb3dt` test=8.769%. Could stack with QK-Norm.
- **lr=9e-5 control on SOTA STRING base**: isolate the LR lever. `9mm3sz7x` used lr=9e-5 with AdamW; worth testing on Lion.
- **EMA-proxy GradNorm α=0.5 (clean re-run)**: prior PR #623 failed on logistics not mechanism. Need kill-gate discipline.
- **Surface-loss weight 2.0 with tay stack**: `qqtdnlwq` test=8.292%. Blocked until tay stack is available or workaround found.
- **Volume MLP head**: replace volume Transolver decoder with a separate MLP for independent volume capacity (`8x7c537j` pre-wave evidence). WD sweep (#667) confirmed volume gap is structural — Volume MLP head is now a higher-priority direction.
- **Beta-NLL heteroscedastic surface head**: principled loss for heteroscedastic tau_y/z. Higher risk.
- **y-symmetry data augmentation**: physics-valid 2× training set. tau_y sign-flip required on flipped cases.
- **DualTower results pending**: if #722 shows vol gap improvement, compose with STRING PE on the tay branch; if not, abandon the architecture direction.
- **Weight decay exhausted**: PR #667 definitively closed. WD={5e-4, 1e-3, 1e-4} all worse than default. Do not re-test WD variations.
