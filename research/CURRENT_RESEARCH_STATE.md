# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 16:30 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Most recent direction from human team:** Issue #48 (2026-04-29 20:54 UTC) —
  "How's it going? We making progress?" — replied with full status. No new directives.
  Previous: Issue #18 from yi advisor: "be bolder; replace the backbone; mine noam/radford branches."

## Current SOTA — nezuko PR #50 MERGED, test_abupt 11.208

| Lever | Student | test_abupt | Note |
|---|---|---:|---|
| #46 | alphonse | 14.55 | AdamW + RFF + compile, epoch 16 |
| arm B | tanjiro (no PR) | 11.303 | Lion lr=5e-5/wd=5e-4 |
| **#50** | **nezuko (MERGED)** | **11.208** | **Lion uncompiled — confirms arm B + marginally better** |

**W&B run:** `g2n4fyta` (rank 0) — 287 min runtime, 9 val epochs, best val 10.08.

## Active assignments (8 students, all DDP8)

| PR | Student | Hypothesis | Latest val | Status |
|---|---|---|---|---|
| **#91** | alphonse | Lion+RFF σ=2.0 — leads σ=1.0 by 1.8% but trails vanilla Lion ep8 | ep8 val 10.634 | Near budget (rt=270m), test eval imminent |
| **#93** | nezuko | Lion+cosine T_max=24 nocompile | ep5 val 14.587 | Running (rt=178m) |
| **#94** | askeladd | Lion+RFF σ=0.5 | ep5 val 13.834 | Running (rt=161m) |
| **#90** | tanjiro | Lion+RFF + EMA decay 0.9999 | ep8 val 32.053 | Running (rt=259m), won't beat baseline |
| **#109** | frieren | Lion uncompiled SOTA + 1-epoch warmup | — | Running (rt=20m) |
| **#110** | edward | Lion uncompiled SOTA + cosine T_max=50 (gentle schedule) | — | Newly assigned |
| **#72** | fern | AdamW+RFF+compile + per-axis tau_y/tau_z | — | Pod stuck (issue #53) |
| **#92** | thorfinn | AdamW+RFF+768d+compile | — | Pod stuck (issue #53) |

## CRITICAL HEAD-TO-HEAD: RFF sigma sweep vs vanilla Lion (uncompiled)

| Epoch | vanilla Lion SOTA | edward σ=1.0 | alphonse σ=2.0 | askeladd σ=0.5 |
|---|---|---|---|---|
| 1 | 80.7 | 73.0 | 72.1 | 75.0 |
| 4 | 19.74 | 15.79 | **15.43** | 15.88 |
| 5 | 14.25 | 13.53 | **13.25** | 13.83 |
| 6 | **12.29** | 12.31 | 12.04 | (pending) |
| 7 | **11.11** | 11.56 | 11.27 | (pending) |
| 8 | **10.38** | 11.01 | 10.63 | (pending) |
| 9 | **10.083** | 10.703 (FINAL) | (pending ~10.30) | (pending) |
| test | **11.208** | 11.741 (+4.7%) | proj ~11.4-11.5 (+2-3%) | proj? |

**KEY FINDING: RFF accelerates early-phase fitting (ep1-5) but vanilla Lion catches up by ep6 and DOMINATES from ep6 onward. RFF inductive bias interferes with finer convergence.**

- **σ=1.0 (edward, FINAL):** test 11.741, +4.7% regression. RFF closed-door at σ=1.0.
- **σ=2.0 (alphonse, ep8):** val 10.634 vs vanilla Lion ep8 10.38 (+2.5%). Will likely also regress, but smaller margin than σ=1.0.
- **σ=0.5 (askeladd, ep5):** val 13.83 — tracks σ=1.0/σ=2.0 closely; RFF basis less informative than σ=1/2.

## Key closed/merged experiments (full history)

| PR | Outcome | Why |
|---|---|---|
| **#50 nezuko** | **MERGED — SOTA 11.208** | Lion lr=5e-5/wd=5e-4 uncompiled |
| arm B (no PR) | SOTA 11.303 (prior) | Lion lr=5e-5/wd=5e-4 baseline |
| #46 alphonse | MERGED — 14.55 | AdamW + RFF + compile → epoch 16 |
| #51 edward | CLOSED — test 11.741 | Lion+RFF σ=1.0 reproducer: +4.7% — RFF doesn't compose with vanilla Lion uncompiled |
| #73 frieren | CLOSED — test 14.785 | 6L+compile budget-limited: 11ep vs 16ep for 4L. val still descending at cutoff |
| #57 askeladd | CLOSED — test 11.229 | Lion+cosine T_max=16 nocompile: stable but wash vs SOTA |
| #70 tanjiro | CLOSED — diverged | Lion+compile+half-LR: 9th Lion+modification failure |
| #69 thorfinn | CLOSED — test 12.351 | 768d uncompiled budget-limited (5ep vs 9) |
| #68 frieren | CLOSED — test 15.57 | Lion + vol_w=3: diverged ep6 |
| #55 alphonse | CLOSED — test 16.39 | AdamW+vol_w=3 regression |
| #54 fern | CLOSED — test 26.83 | Lion + per-axis tau weights: diverged ep4 |
| #52 tanjiro | CLOSED — test 13.20 | Lion+RFF+compile: compile diverges regardless |
| #56 thorfinn | CLOSED — +13% | Cosine T_max=16 on AdamW undercooked |
| #49 askeladd | CLOSED — +35% | Grad-clip 5.0 doesn't compose with Lion |

## Critical learnings

1. **Lion paper config is wrong** — lr=5e-5/wd=5e-4 (AdamW-equivalent) crushes paper config by −27%.
2. **Lion is fragile to ALL modifications involving compile** — 9 confirmed divergences. Stable: vanilla Lion uncompiled (PR #50) and Lion+RFF uncompiled (edward #51).
3. **Lion+compile diverges at any LR within 270min budget** — sign() + reduced gradient noise = biased signs.
4. **vol_w=3 closed-door at 4L/512d** — both AdamW (#55) and Lion (#68) fail.
5. **Budget-limited depth/width:** 768d uncompiled = 5ep (PR #69), 6L compile = 11ep (PR #73). Both fail vs 4L SOTA. Deeper/wider needs more epochs than 270m provides.
6. **Lion+cosine nocompile is STABLE but a wash** — T_max=16 wash; T_max=24 (nezuko) pending; T_max=50 (edward #110) gentle schedule trial.
7. **RFF closed-door on Lion uncompiled** — σ=1.0 (edward #51 FINAL) regresses test by 4.7%. σ=2.0 leads σ=1.0 by ~3% but trails vanilla Lion ep8 by 2.5%. Mechanism: RFF accelerates early-phase fitting (ep1-5) but inductive bias interferes with late convergence (ep6+).
8. **Warmup untested** — 1-epoch LR warmup running on frieren (#109), first time on Lion uncompiled.

## Current research focus (Round 7)

**Lion uncompiled (4L/512d) is the stable SOTA stack. RFF closed-door. Schedule sweep is the active lever.**

- **RFF sigma sweep — CLOSED-DOOR.** σ=1.0 regression confirmed (edward #51 test 11.741). σ=2.0 will likely also regress. RFF acceleration in ep1-5 doesn't survive late convergence.
- **Cosine schedule sweep** — nezuko #93 T_max=24 (running), edward #110 T_max=50 (just assigned). Together a 2-point line.
- **EMA sensitivity** — tanjiro #90 (0.9999) confirmed too slow, won't beat baseline.
- **Warmup** — frieren #109, single-variable test on SOTA stack (just started, rt=20m).
- **Drop-in next experiments after current round:**
  - **Larger batch + LR scaling** (compound capacity utilization)
  - **Mixup / data augmentation on geometry** — try noise injection on coords
  - **Per-task loss weighting** (volume_pressure ×2.1, tau_y/tau_z ×3.7/3.9 — these are the binding gaps)
  - **Architecture: Perceiver-IO backbone** (yi #18 directive, untouched)

**Two pod slots idle (issue #53 open, no human response since 13:33 UTC):**
- thorfinn (#92 AdamW+768d+compile) — GPU 0%, stuck on PR #69 cache
- fern (#72 AdamW+RFF+compile+tau weights) — stuck on PR #54 cache

## Next architecture experiments (if current levers plateau)

- **Perceiver-IO backbone** — yi #18 directive; yi branch has reference code (highest priority)
- **FiLM AdaLN-zero** — conditional on 768d or within 4L budget
- **Pretraining on 50 test geometries** — MAE 75% masking then finetune
- **Per-task loss weighting** — directly target the tau_y/tau_z gap (×3.7/3.9 vs reference) without touching backbone
- **Schedule + warmup compound** — combine winning T_max with frieren warmup
- **Larger batch + LR linear scaling** — capacity per step, fewer epochs needed

## Reference (vs current SOTA)

| Target | AB-UPT | tay SOTA (PR #50) | Gap |
|---|---:|---:|---:|
| `abupt` mean | — | **11.208** | — |
| `surface_pressure` | 3.82 | 6.193 | ×1.6 |
| `wall_shear` | 7.29 | 11.199 | ×1.5 |
| `volume_pressure` | 6.08 | 12.726 | ×2.1 |
| `tau_x` | 5.35 | 9.512 | ×1.8 |
| `tau_y` | 3.65 | 13.592 | ×3.7 |
| `tau_z` | 3.63 | 14.017 | ×3.9 |

**tau_y/tau_z (×3.7/×3.9) and volume_pressure (×2.1) remain the binding gaps.**
