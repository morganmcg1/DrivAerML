# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 15:40 UTC
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
| **#51** | edward | Lion+RFF σ=1.0 reproducer of ftg0ci0p | ep9 val 10.703 (FINAL — test eval running) | Near budget (rt=270m) |
| **#91** | alphonse | Lion+RFF σ=2.0 — LEADS σ=1.0 by 1.8% ep7 | ep7 val 11.273 | Running (rt=235m) |
| **#93** | nezuko | Lion+cosine T_max=24 nocompile | ep4 val 18.074 | Running (rt=143m) |
| **#94** | askeladd | Lion+RFF σ=0.5 | ep4 val 15.878 | Running (rt=127m) |
| **#90** | tanjiro | Lion+RFF + EMA decay 0.9999 | ep8 val 32.053 | Running (rt=259m), won't beat baseline |
| **#109** | frieren | Lion uncompiled SOTA + 1-epoch warmup | — | Newly assigned (PR just created) |
| **#72** | fern | AdamW+RFF+compile + per-axis tau_y/tau_z | — | Pod stuck (issue #53) |
| **#92** | thorfinn | AdamW+RFF+768d+compile | — | Pod stuck (issue #53) |

## CRITICAL HEAD-TO-HEAD: RFF sigma sweep (Lion uncompiled)

| Epoch | ftg0ci0p σ=1.0 (prior) | edward σ=1.0 (repro) | alphonse σ=2.0 | askeladd σ=0.5 |
|---|---|---|---|---|
| 1 | 66.2 | 73.0 | 72.1 | 75.0 |
| 2 | 34.8 | 35.1 | 33.0 | 36.1 |
| 3 | 20.3 | 20.6 | 21.2 | 21.2 |
| 4 | 15.47 | 15.79 | **15.43** | 15.88 |
| 5 | 13.46 | 13.53 | **13.25** | (pending) |
| 6 | 12.32 | 12.31 | **12.04** | (pending) |
| 7 | 11.48 | 11.56 | **11.27** | (pending) |
| 8 | 10.91 | 11.01 | (pending) | |
| 9 | 10.665 | **10.703 (FINAL)** | (pending) | |

**σ=2.0 leads σ=1.0 consistently from ep4 onward (1.8-2.3% advantage). σ=0.5 worse than both.**

**Edward σ=1.0 ep9 val 10.703** — closely matches ftg0ci0p 10.665. Test eval running; projection test_abupt ~11.0-11.1 (narrow SOTA improvement if confirmed).

**Alphonse σ=2.0 leading by ~1.8% at ep7** → projected ep9 val ~10.4-10.5 → test_abupt ~10.7-10.9 → **clear SOTA improvement if trend holds**.

## Key closed/merged experiments (full history)

| PR | Outcome | Why |
|---|---|---|
| **#50 nezuko** | **MERGED — SOTA 11.208** | Lion lr=5e-5/wd=5e-4 uncompiled |
| arm B (no PR) | SOTA 11.303 (prior) | Lion lr=5e-5/wd=5e-4 baseline |
| #46 alphonse | MERGED — 14.55 | AdamW + RFF + compile → epoch 16 |
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
6. **Lion+cosine nocompile is STABLE but a wash** — T_max=16 wash; T_max=24 (nezuko) pending.
7. **RFF σ=2.0 winning the freq sweep** — consistent 1.8-2.3% lead over σ=1.0 from ep4 onward. σ=0.5 worse than both.
8. **Warmup untested** — 1-epoch LR warmup assigned to frieren (#109), first time on Lion uncompiled.

## Current research focus (Round 7)

**Lion uncompiled (4L/512d) is the stable SOTA stack. Compound-lever hunt is active:**

- **RFF sigma sweep** — σ=2.0 is the current winner (ep4-7). σ=0.5 likely losing. σ=2.0 will compound.
- **Cosine schedule** — nezuko #93 T_max=24 slow in tail; T_max=50 is natural next step.
- **EMA sensitivity** — tanjiro #90 (0.9999) confirmed too slow, won't beat baseline.
- **Warmup** — frieren #109, single-variable test on SOTA stack.
- **σ=2.0 + warmup compound** — natural next assignment after frieren #109 result.

**Two pod slots idle (issue #53 open, no human response since 13:33 UTC):**
- thorfinn (#92 AdamW+768d+compile) — GPU 0%, stuck on PR #69 cache
- fern (#72 AdamW+RFF+compile+tau weights) — stuck on PR #54 cache

## Next architecture experiments (if current levers plateau)

- **Perceiver-IO backbone** — yi #18 directive; yi branch has reference code
- **FiLM AdaLN-zero** — conditional on 768d or within 4L budget
- **Pretraining on 50 test geometries** — MAE 75% masking then finetune
- **RFF σ=4.0** — extend the freq sweep above σ=2.0
- **σ=2.0 + cosine T_max=24** — compound σ winner with schedule winner (after both confirmed)
- **σ=2.0 + warmup** — compound after frieren #109 result (assign alphonse or nezuko round 8)

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
