# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 02:55 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Most recent direction from human team:** Issue #48 (2026-04-29 20:54 UTC) —
  "How's it going? We making progress?" — replied with full status. No new directives.
  Previous: Issue #18 from yi advisor: "be bolder; replace the backbone; mine noam/radford branches."

## Current SOTA — tanjiro arm B (no PR), 2026-04-30 02:44 UTC

| Lever | Student | test_abupt | Note |
|---|---|---:|---|
| #30 | alphonse | 19.81 | yi calibration config (4L/512d/8h) |
| #33 | fern | 17.77 | RFF coord features sigma=1.0, no-compile |
| #40 | alphonse | 17.25 | torch.compile fix |
| #39 | tanjiro | 15.43 | Lion paper config (lr=1.7e-5, wd=5e-3) |
| #46 | alphonse | 14.55 | AdamW + RFF + compile, epoch 16 |
| **arm B** | **tanjiro (no PR)** | **11.303** | **Lion lr=5e-5/wd=5e-4 — paper config was wrong** |

**W&B run:** `vnb7oheo` (rank 0) — group `tanjiro-lion-lr-sweep` — 290 min runtime, val
still descending at end. Run was a follow-up sweep launched by tanjiro's pod after PR #39
was reviewed, NOT advisor-assigned. BASELINE.md updated retroactively.

### CRITICAL FINDING: Lion paper config is wrong for this dataset

`lr=1.7e-5, wd=5e-3` (Chen et al. 2023, image classification) → test_abupt 15.43.
`lr=5e-5, wd=5e-4` (AdamW-equivalent translation) → test_abupt **11.30**. Same optimizer,
same code, same data — **−27% just from LR/WD constants**. Lion paper config is calibrated
to ImageNet-scale datasets; with 400 cars we need more aggressive per-step movement to
traverse the loss landscape inside a 270-min budget.

**All future Lion experiments must use `--lr 5e-5 --weight-decay 5e-4`.** Posted
notification on PRs #50, #51, #52, #54 to update before launch.

## Active assignments (8 students, all DDP8)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#42** | frieren | Lion + compile + squared_rel_l2 arm | Running run `24bdfcnz`, ~270 min budget. AdamW arm finished test 15.82 (didn't beat SOTA) |
| **#50** | nezuko | Lion (lr=5e-5) + compile (single delta) | Pod restarted; should now beat 11.30 by +compile epoch-16 bump |
| **#51** | edward | Lion (lr=5e-5) + RFF sigma=1.0 (single delta) | Pod restarted; expected ~10.5–10.8 |
| **#52** | tanjiro | Lion (lr=5e-5) + RFF + compile (full triple-stack) | Pod restarted; expected ~9.8–10.3 |
| **#54** | fern | Lion (lr=5e-5) + per-axis tau_y/tau_z loss weighting (2×/2×) | Targets binding 5×+ gap |
| **#55** | alphonse | RFF + compile + volume-loss-weight 3.0 (AdamW base) | LOWER PRIORITY now — vol_p regression dissolves under Lion |
| **#56** | thorfinn | AdamW + RFF + compile + cosine LR T_max=16 | Schedule miscalibration fix |
| **askeladd** | — | (just closed #49 +35% regression) | **IDLE — needs new round-3 hypothesis** |

## Key closed/merged experiments

| PR | Outcome | Why |
|---|---|---|
| **arm B** (no PR) | **NEW SOTA 11.30** | Lion lr=5e-5/wd=5e-4 — paper config was wrong by −27% |
| #46 alphonse | MERGED — 14.55 | AdamW + RFF + compile → epoch 16 (was tay SOTA before arm B) |
| #49 askeladd | CLOSED — +35% regression | Grad-clip 5.0 + AdamW; lever doesn't compose forward under Lion |
| #47 thorfinn | CLOSED — +36% regression | Bilateral train-aug pushes model to less-accurate point on real test |
| #43 fern | CLOSED — +15.7% regression | Multi-scale RFF redundant on `[−1,4]m` surface domain |
| #41 askeladd | CLOSED — +22.5% regression | Eval-time tangential projection destroys normal-component tau |
| #37 thorfinn | CLOSED — +14.2% regression | Bilateral TTA-eval only; per-axis weights never pushed |
| #35 nezuko | CLOSED — +13.7% regression | ANP uncompilable (inductor bug) |
| #44 edward | CLOSED — +18.8% regression | Cosine EMA over max_epochs=50 → only 18% of schedule run |
| #34 frieren | CLOSED Round-1 — +0.26% noise | FiLM didn't transfer at 9 uncompiled epochs |
| #32 edward | CLOSED Round-1 — regression | LR schedule over 50 epochs, same budget miscalibration |

## Critical learnings

1. **Lion paper config is wrong for this dataset** — `lr=5e-5, wd=5e-4` (AdamW-equivalent)
   crushes paper config by −27%. Small datasets need aggressive per-step movement.
2. **Lion sidesteps grad-clip compression** — sign-based updates make pre-clip grad-norm
   irrelevant; per-step movement bounded by `lr` directly. PR #49 confirmed AdamW + clip 5.0
   is uncompetitive vs Lion.
3. **Compile gives epoch 16** (vs 9 uncompiled). T_max for schedules should be 16, not 12.
4. **Volume_pressure regression from AdamW + compile** — Lion's sign-update normalizes
   per-channel gradient magnitude. With Lion lr=5e-5 we already get vol_p=12.76 < PR #39's
   13.83, so PR #55's vol_w=3.0 fix on AdamW base is now lower priority.
5. **RFF orthogonal to optimizer** — composition Lion+RFF (PR #51) is queued; expect ~10.5–10.8.
6. **Multi-scale RFF redundant** on DrivAerML surface domain — `[−1,4]m` already covered by σ=1.0.
7. **Cosine EMA needs `--ema-total-epochs 16`** (not 50 or 12). Edward #44 used 50 → 18% of schedule.
8. **ANP decoder uncompilable** — inductor dynamic-shapes bug. Dead end until patched.
9. **Eval-time tangential projection destroys normal-component tau** — closed door from yi #11.
10. **Bilateral aug pushes model off the asymmetric test distribution** — closed door from thorfinn #47.

## Next priority compositions (after Lion lr=5e-5 SOTA)

High priority (single-delta from new SOTA 11.30):
1. **Lion (5e-5) + compile** (PR #50 nezuko) — should land ~10.7–11.0 with epoch-16 capacity.
2. **Lion (5e-5) + RFF** (PR #51 edward) — RFF +0.88 at AdamW base; expect ~10.5–10.8.
3. **Lion (5e-5) + RFF + compile** (PR #52 tanjiro) — full triple-stack; expect ~9.8–10.3.
4. **Lion (5e-5) + per-axis tau_y/tau_z weights** (PR #54 fern) — targets the 3.8× / 3.9× gap.

Medium priority:
5. **PR #55 alphonse RFF + compile + vol_w=3.0** — vol_p regression dissolves under Lion;
   keep running but expect lower delta than originally projected.
6. **PR #56 thorfinn RFF + compile + cosine T_max=16** — schedule fix; orthogonal to Lion.
7. **PR #42 frieren Lion + compile + squared_rel_l2** — wait for run to finish.

To assign next (askeladd is now idle):
- **Lion (5e-5) + cosine EMA T_max=16** — yi's biggest non-arch lever, now correctly
  calibrated to the actual 16-epoch budget. Drop-in single delta on the new SOTA.
- **Lion (5e-5) + FiLM AdaLN-zero** — FiLM failed at uncompiled 9 epochs (frieren #34);
  Lion+compile gives FiLM another chance with full schedule.
- **Lion (5e-5) + width 768d µP-scaled LR** — bigger model under the winning optimizer.
- **Lion (5e-5) + extended budget** — `vnb7oheo` was still descending at 290 min; how much
  headroom is left at 360 min? (Caveat: this would require an env-var override, may not be
  acceptable under hard SENPAI_TIMEOUT_MINUTES policy.)

## Longer-horizon ideas (post-Lion composition saturation)

- **Backbone replacement** — Perceiver-IO, Mamba-2, Transolver-3, GINO on volume channel
- **Width increase** — 512d → 768d with µP-scaled LR
- **Deep supervision** — intermediate layer predictions
- **Uncertainty head** — NIG / evidential regression
- **Multi-resolution patches** — coarse + fine surface tokenization
- **Pretraining** — denoising on 50 unlabelled test geometries; MAE 75% masking

## Reference (vs new SOTA)

| Target | AB-UPT | tay SOTA | Gap |
|---|---:|---:|---:|
| `abupt` mean | — | **11.303** | — |
| `surface_pressure` | 3.82 | 6.216 | ×1.6 |
| `wall_shear` | 7.29 | 11.315 | ×1.6 |
| `volume_pressure` | 6.08 | 12.755 | ×2.1 |
| `tau_x` | 5.35 | 9.563 | ×1.8 |
| `tau_y` | 3.65 | 13.831 | ×3.8 |
| `tau_z` | 3.63 | 14.147 | ×3.9 |

Surface/wall_shear/tau_x gaps closed to ×1.6–1.8 — the project is well past the yi frontier.
**volume_pressure (×2.1) and tau_y/tau_z (×3.8/×3.9) are the binding gaps now.** Per-axis
weighting (PR #54 fern) and physical-symmetry priors (different from bilateral aug — maybe
SE(3)-equivariant feature attention?) are the most directly targeted levers.
