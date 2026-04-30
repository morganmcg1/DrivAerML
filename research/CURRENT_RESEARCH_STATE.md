# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 13:00 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Most recent direction from human team:** Issue #48 (2026-04-29 20:54 UTC) —
  "How's it going? We making progress?" — replied with full status. No new directives.
  Previous: Issue #18 from yi advisor: "be bolder; replace the backbone; mine noam/radford branches."

## Current SOTA — nezuko PR #50 MERGED, 2026-04-30 12:56 UTC

| Lever | Student | test_abupt | Note |
|---|---|---:|---|
| #30 | alphonse | 19.81 | yi calibration config (4L/512d/8h) |
| #33 | fern | 17.77 | RFF coord features sigma=1.0, no-compile |
| #40 | alphonse | 17.25 | torch.compile fix |
| #39 | tanjiro | 15.43 | Lion paper config (lr=1.7e-5, wd=5e-3) |
| #46 | alphonse | 14.55 | AdamW + RFF + compile, epoch 16 |
| arm B | tanjiro (no PR) | 11.303 | Lion lr=5e-5/wd=5e-4 |
| **#50** | **nezuko (MERGED)** | **11.208** | **Lion uncompiled — confirms arm B + marginally better** |

**W&B run:** `g2n4fyta` (rank 0) — 287 min runtime, 9 val epochs, val 10.08 still descending.

### POTENTIAL SECOND SOTA UPDATE: askeladd #57 `jh1j9uq4` — val 10.13 ep9 (pending test)

Askeladd #57 (Lion+cosine T_max=16, nocompile) reached val 10.13 at ep9 (step 23434). Run past budget at rt=276min, currently in test eval. With val 10.13 ≈ nezuko val 10.08, expected test_abupt ~11.10-11.25. **Could compound on nezuko #50 SOTA immediately.**

## Active assignments (8 students, all DDP8)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#57** | askeladd | Lion+cosine T_max=16 nocompile — val 10.13 ep9 | In test eval (rt=276m) |
| **#91** | alphonse | Lion+RFF sigma=2.0 — higher-freq RFF | Running ~77min, val 32.97 ep2 |
| **#90** | tanjiro | Lion+RFF + EMA decay 0.9999 | Running ~88min, val 80.98 ep2 (EMA lag) |
| **#73** | frieren | AdamW + RFF + compile + depth 6L | Running ~138min, val 19.26 ep5 |
| **#72** | fern | AdamW + RFF + compile + per-axis tau_y/tau_z (2×) | Pod stuck (issue #53) |
| **#51** | edward | Lion+RFF sigma=1.0 — reproducer of ftg0ci0p | Running ~109min, val 20.58 ep3 |

**Students idle after round closures: thorfinn, nezuko — need new assignments.**

## Key closed/merged experiments (full history)

| PR | Outcome | Why |
|---|---|---|
| **#50 nezuko** | **MERGED — SOTA 11.208** | Lion lr=5e-5/wd=5e-4 uncompiled — clean SOTA confirm |
| arm B (no PR) | SOTA 11.303 (prior) | Lion lr=5e-5/wd=5e-4 — paper config wrong by −27% |
| #46 alphonse | MERGED — 14.55 | AdamW + RFF + compile → epoch 16 |
| #70 tanjiro | CLOSED — diverged | Lion+compile+half-LR(2.5e-5): 9th Lion+modification failure |
| #69 thorfinn | CLOSED — test 12.351 | 768d uncompiled budget-limited (5 epochs vs 9) |
| #68 frieren | CLOSED — test 15.57 | Lion + vol_w=3: diverged ep6 |
| #55 alphonse | CLOSED — test 16.39 | AdamW+vol_w=3 regression; vol_w=3 closed-door |
| #54 fern | CLOSED — test 26.83 | Lion + per-axis tau weights: diverged ep4 |
| #52 tanjiro | CLOSED — test 13.20 | Lion+RFF+compile: compile diverges regardless |
| #56 thorfinn | CLOSED — +13% | Cosine T_max=16 on AdamW undercooked |
| #49 askeladd | CLOSED — +35% | Grad-clip 5.0 doesn't compose with Lion |
| #47 thorfinn | CLOSED — +36% | Bilateral aug → off asymmetric test dist |
| #43 fern | CLOSED — +15.7% | Multi-scale RFF redundant |
| #41 askeladd | CLOSED — +22.5% | Eval tangential projection destroys tau |
| #35 nezuko | CLOSED — +13.7% | ANP uncompilable (inductor bug) |
| #44 edward | CLOSED — +18.8% | Cosine EMA over 50 epochs → 18% schedule |
| #42 frieren | CLOSED — +40% | Lion + sq_rel_l2: diverged |

## Critical learnings

1. **Lion paper config is wrong** — lr=5e-5/wd=5e-4 (AdamW-equivalent) crushes paper config by −27%.
2. **Lion is fragile to ALL non-trivial modifications** — 9 confirmed divergences. Stable: vanilla Lion uncompiled (PR #50, #57) and Lion+RFF uncompiled (PR #51 edward).
3. **Lion+compile diverges at any LR within 270min budget** — sign() + reduced gradient noise = biased signs. Half-LR delays (ep9-10) but does NOT prevent.
4. **vol_w=3 closed-door at 4L/512d** — both AdamW (#55) and Lion (#68) fail.
5. **768d uncompiled is budget-limited** — 5 epochs in 270min vs 9 for 512d. val 11.23 at ep5 ≈ 512d ep7. 768d+compile untested.
6. **Lion+cosine T_max=16 nocompile is STABLE** — askeladd #57 val 10.13 ep9. NEW FINDING this round. Cosine schedule doesn't fix Lion+compile, but with Lion nocompile it's stable and may improve final convergence.
7. **Compile gives epoch 16 in budget** (vs ~9 uncompiled). T_max for schedules = 16 if compiled.

## Current research focus (Round 6 → 7)

**Lion uncompiled confirmed as SOTA stack. Three stable Lion variants identified:**

1. Vanilla Lion uncompiled (PR #50 test 11.208) — SOTA
2. Lion+cosine T_max=16 nocompile (askeladd #57 val 10.13, test pending) — new stable variant
3. Lion+RFF sigma=1.0 nocompile (edward #51 ftg0ci0p val 10.665 ep9) — strongest val seen

**Active explorations:**
- Askeladd #57 test eval pending — if it beats 11.208, Lion+cosine is compounding lever
- Edward #51 ep3 val 20.58 — reproducing Lion+RFF. ep9 target ~10.5-11.0
- Tanjiro #90 (Lion+RFF+EMA 0.9999) ep2 val 80.98 — very slow start, needs ep5+
- Alphonse #91 (Lion+RFF sigma=2.0) ep2 val 32.97 — freq sweep
- Frieren #73 (6L compile AdamW+RFF) ep5 val 19.26 — architectural depth lever

**Next assignments for idle students (thorfinn, nezuko):**
- Thorfinn: 768d + compile (test if Lion+compile fragility extends to 768d or µP stabilizes it)
- Nezuko: Lion+cosine T_max=50 nocompile (longer schedule than T_max=16 — if cosine is a lever, T_max=50 gives more descent across 9 epochs)

## Next architecture experiments (if current round continues plateauing)

- **Perceiver-IO backbone** — yi #18 directive; yi branch has reference code
- **FiLM AdaLN-zero on 768d** — FiLM failed at 512d; 768d gives bigger capacity + more schedule
- **Pretraining on 50 unlabelled test geometries** — MAE 75% masking, then finetune
- **RFF sigma=0.5** — lower freq companion to sigma=2.0 (alphonse #91)
- **Cosine T_max=24 on AdamW** — T_max=16 too short, T_max=50 load-bearing, T_max=24 unexplored

## Reference (vs new SOTA)

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
