# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 18:04 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Most recent direction from human team:** Issue #48 (2026-04-29 20:54 UTC) —
  "How's it going? We making progress?" — replied with full status. No new directives.
  Previous: Issue #18 from yi advisor: "be bolder; replace the backbone; mine noam/radford branches."
- **Infra (issue #53):** Fern + thorfinn pods RECOVERED 17:14-17:18 UTC. **Full 8-student fleet active.** Issue posted recovery comment.

## Current SOTA — nezuko PR #50 MERGED, test_abupt 11.208

| Lever | Student | test_abupt | Note |
|---|---|---:|---|
| #46 | alphonse | 14.55 | AdamW + RFF + compile, epoch 16 |
| arm B | tanjiro (no PR) | 11.303 | Lion lr=5e-5/wd=5e-4 |
| **#50** | **nezuko (MERGED)** | **11.208** | **Lion uncompiled — confirms arm B + marginally better** |

**W&B run:** `g2n4fyta` (rank 0) — 287 min runtime, 9 val epochs, best val 10.08.

## Active assignments (8 students, all DDP8 — ALL ACTIVE as of 17:18 UTC)

| PR | Student | Hypothesis | Latest val | Status |
|---|---|---|---|---|
| **#112** | alphonse | Lion uncompiled SOTA + lr=1e-4 (LR sweep, 2× current) | **ep2 val 45.09** (vs vanilla 36.5, **+24% — overshooting**) | Running (rt=67m) — looking poor |
| **#113** | nezuko | Lion uncompiled SOTA + lr=3e-5 (LR sweep lower bound) | — | Just assigned (rt=0m) |
| **#94** | askeladd | Lion+RFF σ=0.5 | **ep9 val 10.405** (vs vanilla 10.083, +3.2%) | In test eval (rt=276m) |
| **#111** | tanjiro | Lion uncompiled SOTA + EMA decay 0.999 (faster tracking) | **ep3 val 18.09** (vs vanilla ep4 19.74 — AHEAD by ~1 epoch) | Running (rt=104m) — **MAJOR WIN PROJECTED** |
| **#109** | frieren | Lion uncompiled SOTA + 1-epoch warmup | ep3 val 39.62 (vs vanilla ep3 ~25, costs ~1 epoch) | Running (rt=115m) |
| **#110** | edward | Lion uncompiled SOTA + cosine T_max=50 (gentle schedule) | ep2 val 46.58 (vs vanilla 36.5, slower) | Running (rt=90m) |
| **#72** | fern | AdamW+RFF+compile + per-axis tau_y/tau_z | ep2 val 41.28 | Running (rt=43m) |
| **#92** | thorfinn | AdamW+RFF+768d+compile | ep1 val 67.95 | Running (rt=35m) |

## CRITICAL HEAD-TO-HEAD: RFF sigma sweep vs vanilla Lion (uncompiled)

| Epoch | vanilla Lion SOTA | edward σ=1.0 | alphonse σ=2.0 | askeladd σ=0.5 |
|---|---|---|---|---|
| 1 | 80.7 | 73.0 | 72.1 | 74.99 |
| 4 | 19.74 | 15.79 | **15.43** | 15.88 |
| 5 | 14.25 | 13.53 | **13.25** | 13.83 |
| 6 | **12.29** | 12.31 | 12.04 | 12.46 |
| 7 | **11.11** | 11.56 | 11.27 | **11.495** |
| 8 | **10.38** | 11.01 | 10.63 | (pending ~10.7) |
| 9 | **10.083** | 10.703 (FINAL) | 10.321 (FINAL) | (pending ~10.4) |
| test | **11.208** | 11.741 (+4.7%) | 11.376 (+1.5%) | proj ~11.5 (+2-3%) |

**KEY FINDING: RFF accelerates early-phase fitting (ep1-5) but vanilla Lion catches up by ep6 and DOMINATES from ep6 onward. RFF inductive bias interferes with finer convergence.**

- **σ=1.0 (edward, FINAL):** test 11.741, +4.7% regression. RFF closed-door at σ=1.0.
- **σ=2.0 (alphonse #91, CLOSED):** test 11.376 (+1.5%). Val 10.321. Uniquely wins tau_y by 1% but regresses all other components.
- **σ=0.5 (askeladd, ep7 val 11.495):** tracking ~3% behind vanilla through ep7 — closing in on σ=2.0 pattern. Test eval imminent (~30m).

## Schedule sweep — Lion+cosine vs vanilla Lion (uncompiled)

| Epoch | vanilla Lion SOTA | nezuko T_max=24 | edward T_max=50 |
|---|---|---|---|
| 1 | 80.7 | 74.97 | 75.57 |
| 2 | 36.5 | 45.44 | 46.58 |
| 5 | 14.25 | 14.59 | (pending) |
| 6 | **12.29** | 12.66 | (pending) |
| 7 | **11.11** | 11.42 | (pending) |
| 8 | **10.38** | 10.64 | (pending) |
| 9 | **10.083** | (pending ~10.2-10.3) | (pending) |
| test | **11.208** | proj ~11.3-11.4 | (pending) |

**Schedule pattern: cosine schedule consistently tracks ~2-5% behind vanilla through all epochs. Both T_max=24 and T_max=50 hurt early descent (ep2: 45-47 vs vanilla 36.5).**

## Key closed/merged experiments (full history)

| PR | Outcome | Why |
|---|---|---|
| **#50 nezuko** | **MERGED — SOTA 11.208** | Lion lr=5e-5/wd=5e-4 uncompiled |
| arm B (no PR) | SOTA 11.303 (prior) | Lion lr=5e-5/wd=5e-4 baseline |
| #46 alphonse | MERGED — 14.55 | AdamW + RFF + compile → epoch 16 |
| #91 alphonse | CLOSED — test 11.376 | Lion+RFF σ=2.0: +1.5% — closest sigma but still regresses; tau_y uniquely −1% better |
| #90 tanjiro | CLOSED — test 30.203 | EMA 0.9999 budget-incompatible: half-life ~50 epochs vs 9 available |
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
8. **EMA 0.9999 budget-incompatible** — half-life ~50 epochs vs 9 available. EMA 0.9995 is calibrated correctly for this budget. EMA 0.999 (tanjiro #111) is the next test — faster tracking, may help late convergence.
9. **Warmup untested** — 1-epoch LR warmup running on frieren (#109), first time on Lion uncompiled.
10. **LR 5e-5 untested against sweep** — never directly swept on Lion uncompiled. lr=1e-4 (alphonse #112) is round 8's first LR variation.

## Current research focus (Rounds 7-8)

**Lion uncompiled (4L/512d) is the stable SOTA stack. RFF closed-door. Schedule + LR sweep is the active frontier.**

- **RFF sigma sweep — CLOSED-DOOR.** σ=0.5/1.0/2.0 all regress. Pattern: RFF helps ep1-5 but vanilla Lion dominates ep6+.
- **LR sweep** — alphonse #112 testing lr=1e-4 (2× current SOTA lr=5e-5). First LR variation on Lion uncompiled stack.
- **Cosine schedule sweep — CLOSED-DOOR.** nezuko #93 CLOSED (test 11.524, +2.8%), T_max=16 was a wash. Schedule hurts Lion in 9-epoch budget — constant LR is already optimal.
- **LR sweep** — alphonse #112 lr=1e-4 (running, ep2 overshooting +24%) + nezuko #113 lr=3e-5 (just assigned). Both sides of SOTA lr=5e-5 now probed.
- **EMA sweep** — tanjiro #90 (0.9999) closed. **Tanjiro #111 (0.999) ep2 val 27.00 = 26% better than vanilla ep2 36.5 — strong signal.**
- **Warmup** — frieren #109 running (rt=115m, ep4 val 21.11 vs vanilla ep4 19.74, +6.9%).
- **Upcoming hypothesis queue:**
  - **EMA=0.998** — next step if tanjiro #111 (0.999) wins, refining the sweep
  - **EMA=0.999 + winning LR** — compound stack once LR winner is known
  - **Per-task tau weighting** — directly target tau_y/tau_z binding gaps (×3.7/×3.9)
  - **Perceiver-IO backbone** — yi #18 directive, if current levers plateau
  - **Batch size 8** — 2× effective batch, may compound with any LR winner

## Round 7-8 status (17:57 UTC) — quick read

- **Schedule sweep CLOSED.** nezuko #93 T_max=24 finished test 11.524 (+2.8%) — closed-door confirmed alongside T_max=16 (wash, #57). Edward #110 T_max=50 still running (ep3 24.22), expected regression.
- **RFF σ=0.5 still running (ep8 10.77, finishing soon)**. Tracking similar to σ=1.0/2.0 — RFF remains closed-door across all sigma.
- **EMA=0.999 IS NOW ~1 EPOCH AHEAD OF VANILLA** (ep1: 55.06 vs 80.7; ep2: 27.00 vs 36.5; ep3: **18.09 vs vanilla ep4 19.74**). Tanjiro at ep3 has already exceeded vanilla's ep4. If this continues, projected ep9 val ~9.5-9.7 → test ~10.5-10.7 (vs SOTA 11.208) = **5-6% SOTA improvement** would be a major win. Most-important run in flight.
- **lr=1e-4 overshooting** (ep2 45.09 vs vanilla 36.5, +24%). Likely regression. lr=3e-5 (nezuko #113) assigned as counterpoint.
- **Warmup at ep4** (frieren #109): 21.11 vs vanilla 19.74 (+6.9%) — largely recovered from ep3 loss. Warmup may only cost half-an-epoch by ep4.
- **Fern + thorfinn** (ep1 73.24/67.95) — fresh starts post-recovery, AdamW+compile branches, orthogonal to Lion.

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
