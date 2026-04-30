# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 00:50 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Most recent direction from human team:** Issue #48 (2026-04-29 20:54 UTC) —
  "How's it going? We making progress?" — replied with full status. No new directives.
  Previous: Issue #18 from yi advisor: "be bolder; replace the backbone; mine noam/radford branches."

## Current SOTA — PR #39 MERGED 2026-04-29 22:05 UTC

**tay crosses below yi frontier: test_abupt 15.43 vs yi best 15.82**

| PR | Student | test_abupt | Key lever |
|---|---|---:|---|
| #30 | alphonse | 19.81 | yi calibration config (4L/512d/8h) |
| #33 | fern | 17.77 | RFF coord features sigma=1.0, no-compile |
| #40 | alphonse | 17.25 | torch.compile fix → 12 epochs vs 9 |
| **#39** | **tanjiro** | **15.43** | **Lion optimizer lr=1.7e-5, wd=5e-3 — sign-based updates** |

**W&B run:** `xonbs83i` (rank 0) — group `tay-lion-lr-sweep` — 287.1 min, best_epoch=9
val curve still descending at cutoff → compile would give 12 epochs, further improvement expected.

## Active assignments (8 students, all DDP8)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #46 | alphonse | RFF (sigma=1.0, 32 feats) + compile | Running, val 14.446, 234 min (~01:00 finish) |
| **#54** | **fern** | **Lion + per-axis tau_y/tau_z loss weighting (2×/2× vs 1×)** | **Just assigned — addresses binding 5×+ gap** |
| #47 | thorfinn | Bilateral train-time augmentation | Running, val 21.87, 210 min |
| #42 | frieren | Squared rel-L2 loss + compile (rebased) | Running, run `uwt74mip`, ~207 min |
| #49 | askeladd | Grad-clip-norm 1.0 → 5.0 (single delta) | Running, val 17.08, ~3h in |
| #50 | nezuko | Lion + compile (single delta) | Queued — pod hung, Issue #53 pending restart |
| #51 | edward | Lion + RFF sigma=1.0 (single delta) | Queued — pod hung, Issue #53 pending restart |
| #52 | tanjiro | Lion + RFF + compile (triple-stack) | Queued — waiting for tanjiro arm B `vnb7oheo` |

## Key closed experiments this round

| PR | Outcome | Why |
|---|---|---|
| #43 fern | CLOSED — +15.7% vs SOTA, noise vs own baseline | Multi-scale RFF redundant on [−1,4]m surface domain; σ=1.0 already at local optimum |
| #41 askeladd | CLOSED — regression +22.5% | Eval-time tangential projection destroys normal-component of tau that helps rel-L2 |
| #37 thorfinn | CLOSED — regression +14.2% | Bilateral TTA-eval only (no train aug); per-axis weights code never pushed |
| #35 nezuko | CLOSED — regression +13.7% vs new SOTA | ANP uncompilable (inductor dynamic-shapes bug); surface wins not sustained in composition |
| #44 edward | CLOSED — regression +18.8% vs new SOTA | Cosine EMA over max_epochs=50 → only 18% of schedule run → EMA=0.991 below fixed 0.9995 |
| #34 frieren | CLOSED Round-1 — +0.26% noise | FiLM didn't transfer at 9 uncompiled epochs |
| #32 edward | CLOSED Round-1 — regression | LR schedule over 50 epochs, same budget miscalibration |

## Critical learnings

1. **Lion optimizer is the dominant new lever** — sign-based updates sidestep grad-clip-norm=1.0 compression. `lr=1.7e-5, wd=5e-3` (paper config) wins by −10.5%.
2. **RFF is orthogonal to optimizer** — composition Lion+RFF is the next highest-priority test (PR #51).
3. **Compile gives ~12 vs 9 epochs** — val curve still descending at epoch 9 (Lion). Lion+compile (PR #50) is the second highest-priority test.
4. **Cosine EMA needs `--ema-total-epochs` calibrated to actual budget** (12, not 50). Without this it's a regression.
5. **ANP decoder can't be compiled** — inductor dynamic-shapes bug specific to backbone-then-split flow. Dead end until that patch lands.
6. **eval-time tangential projection destroys normal-component tau** — closed door with yi PR #11.
7. **Budget miscalibration compounds** — any schedule-dependent hyperparameter (EMA, LR warmup, etc.) must be set to `T = actual_epochs_expected`, not `max_epochs`.

## Next priority compositions (after Lion win)

High priority (single-delta from Lion SOTA):
1. **Lion + compile** (PR #50 nezuko) — queued, pod restart needed
2. **Lion + RFF** (PR #51 edward) — queued, pod restart needed
3. **Lion + per-axis tau_y/tau_z weights** (PR #54 fern) — just assigned, addresses 5×+ binding gap
4. **Lion + RFF + compile** (PR #52 tanjiro) — queued after arm B finishes
5. **Lion + FiLM** (not yet assigned) — FiLM failed at uncompiled 9 epochs; Lion+compile gives FiLM another chance
6. **Lion + cosine EMA (correct T_max=12)** — yi's biggest non-arch lever, correctly calibrated

Active but possibly suboptimal vs Lion:
5. **RFF + compile (PR #46 alphonse)** — tests RFF without Lion; important for decomposing gains
6. **Bilateral train-aug (PR #47 thorfinn)** — physical symmetry augmentation; still worth knowing
7. **Squared rel-L2 (PR #42 frieren)** — may compose with Lion; wait for result
8. **Grad-clip 5.0 (PR #49 askeladd)** — motivated by frieren's AdamW diagnostic; less relevant for Lion

## Longer-horizon ideas (post-Lion composition)

- **FiLM + Lion + compile** — full composition when #50 and #51 land
- **Cosine EMA (correct T_max) + Lion** — full optimizer+schedule composition
- **Progressive cosine EMA decay** with T_max = actual expected epochs
- **Backbone replacement** — Perceiver-IO, Mamba-2, Transolver-3, GINO on volume channel
- **Width increase** — 512d is current; try 768d with µP-scaled LR
- **Deeper supervision** — intermediate layer predictions à la deep supervision
- **Uncertainty head** — NIG or evidential regression for training stability signal
- **Multi-resolution patches** — coarse + fine surface tokenization
- **Pretraining** — denoising on 50 unlabelled test geometries; MAE 75% masking

## Reference

| Target | AB-UPT | tay SOTA | Gap |
|---|---:|---:|---:|
| `abupt` mean | — | **15.43** | — |
| `surface_pressure` | 3.82 | 9.45 | ×2.5 |
| `wall_shear` | 7.29 | 16.28 | ×2.2 |
| `volume_pressure` | 6.08 | 13.83 | ×2.3 |
| `tau_x` | 5.35 | 13.91 | ×2.6 |
| `tau_y` | 3.65 | 19.58 | ×5.4 |
| `tau_z` | 3.63 | 20.40 | ×5.6 |

tau_y and tau_z remain the hardest gap targets — need physical symmetry or per-axis weighting.
