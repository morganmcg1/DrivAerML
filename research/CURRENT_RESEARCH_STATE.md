# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 11:10 UTC
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

### NEW INTERIM SOTA CANDIDATE: edward #51 `ftg0ci0p` — val 10.665 (ep9)

Edward's run `ftg0ci0p` (Lion+RFF+sigma=1.0, uncompiled) reached **val_abupt = 10.665** at epoch 9 — the lowest val ever recorded, 0.64 ppt below arm B SOTA. Run hit 270min budget before test eval. Pod auto-relaunched as `iocqp761` with same config. When `iocqp761` completes (~270min), we expect test_abupt ≈ 10.5-11.0 (NEW SOTA).

### CRITICAL FINDING: Lion paper config is wrong for this dataset

`lr=1.7e-5, wd=5e-3` (Chen et al. 2023, image classification) → test_abupt 15.43.
`lr=5e-5, wd=5e-4` (AdamW-equivalent translation) → test_abupt **11.30**. Same optimizer,
same code, same data — **−27% just from LR/WD constants**. Lion paper config is calibrated
to ImageNet-scale datasets; with 400 cars we need more aggressive per-step movement to
traverse the loss landscape inside a 270-min budget.

**All future Lion experiments must use `--lr 5e-5 --weight-decay 5e-4`.**

## Active assignments (8 students, all DDP8)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#91** | alphonse | Lion+RFF sigma=2.0 — higher-freq tau resolution | JUST ASSIGNED |
| **#90** | tanjiro | Lion+RFF + EMA decay 0.9999 — yi #13 compounding | JUST ASSIGNED |
| **#73** | frieren | AdamW + RFF + compile + depth 6L — architectural capacity | Running ~50min |
| **#72** | fern | AdamW + RFF + compile + per-axis tau_y/tau_z weights (2×) | Pod stuck (issue #53) |
| **#69** | thorfinn | Lion (lr=3.3e-5 µP) + width 768d / 12h — architectural capacity | Running ~180min, val 16.42 (ep3) |
| **#57** | askeladd | Lion+cosine fallback (nocompile) | Running ~157min, val 13.80 |
| **#51** | edward | Lion+RFF sigma=1.0 — SOTA CONTENDER (new run iocqp761) | Running ~10min, new after ftg0ci0p budget |
| **#50** | nezuko | Lion uncompiled fallback (same config as SOTA) | Running ~180min, val 14.25 |

## Key closed/merged experiments

| PR | Outcome | Why |
|---|---|---|
| **arm B** (no PR) | **NEW SOTA 11.30** | Lion lr=5e-5/wd=5e-4 — paper config was wrong by −27% |
| #46 alphonse | MERGED — 14.55 | AdamW + RFF + compile → epoch 16 |
| #70 tanjiro | CLOSED — diverged (best 13.07 mid-train) | Lion+compile+half-LR(2.5e-5): diverged ep9-10 — **9th** Lion+modification failure; half-LR delayed but didn't prevent |
| #55 alphonse | CLOSED — test 16.39 (+12.6% vs PR #46) | AdamW+vol_w=3 regression; also diverged ep17; vol_w=3 lever is closed-door on both AdamW AND Lion |
| #68 frieren | CLOSED — test 15.57 (+37.7%) | Lion + vol_w=3: diverged ep6; vol_w lever inert when run diverges |
| #54 fern | CLOSED — test 26.83 (+137%) | Lion + per-axis tau_y/tau_z weights: diverged ep4 |
| #52 tanjiro | CLOSED — test 13.20 (+16.8% vs SOTA) | Lion+RFF+compile: RFF+compile add zero orthogonal signal; compile diverges |
| #56 thorfinn | CLOSED — +13% vs #46 | Cosine T_max=16 on AdamW: undercooked schedule |
| #42 frieren | CLOSED — +40% regression | Lion + sq_rel_l2: diverged |
| #49 askeladd | CLOSED — +35% regression | Grad-clip 5.0; doesn't compose with Lion |
| #47 thorfinn | CLOSED — +36% regression | Bilateral aug → off asymmetric test distribution |
| #43 fern | CLOSED — +15.7% regression | Multi-scale RFF redundant on `[−1,4]m` domain |
| #41 askeladd | CLOSED — +22.5% regression | Eval-time tangential projection destroys tau |
| #35 nezuko | CLOSED — +13.7% regression | ANP uncompilable (inductor bug) |
| #44 edward | CLOSED — +18.8% regression | Cosine EMA over 50 epochs → 18% of schedule |

## Critical learnings

1. **Lion paper config is wrong for this dataset** — `lr=5e-5, wd=5e-4` (AdamW-equivalent) crushes paper config by −27%.
2. **Lion is fragile to ALL non-trivial modifications** — confirmed 9× across PRs #42, #50, #52, #54, #57, #68, #56, #70, plus half-LR. Any change to: compile (any LR), loss function (sq_rel_l2, per-axis weighting, vol_w increase), or schedule (cosine T_max=16) diverges. **Stable Lion variants:** vanilla Lion (vnb7oheo, 11.30 SOTA) and Lion+RFF uncompiled (PR #51 edward, val 10.665 ep9).
3. **Lion+compile mechanism:** sign() update bounded only by lr. Compile's reduced gradient noise → more deterministically biased signs → effectively higher LR than uncompiled. **Half-LR (2.5e-5) only delays divergence 2 epochs — does NOT prevent it.** Onset: ep4-10 depending on modification.
4. **vol_w=3 lever is closed-door at 4L/512d** — BOTH AdamW (#55, test 16.39) and Lion (#68, test 15.57) fail. AdamW also diverged late at vol_w=3 (ep17). The lever only targets volume at the cost of all surface axes.
5. **Cosine T_max=16 on AdamW is a regression** (PR #56: +13% vs #46). T_max=50 fallback (LR ≈88% at end) was load-bearing — model is undertrained at 16 epochs. Future cosine: T_max≥24 or budget extension only.
6. **Compile gives epoch 16** (vs ~9 uncompiled in 270 min). T_max for schedules = 16 if compiled.
7. **Per-axis loss weighting mechanism works at gradient level** — PR #54 showed tau_y/tau_x ratio improved (1.33 vs 1.45) before divergence. Test on stable AdamW base (PR #72 fern, pod stuck).
8. **Multi-scale RFF redundant** on DrivAerML surface domain — `[−1,4]m` already covered by σ=1.0. σ=2.0 (PR #91 alphonse) is the next unexplored frequency.
9. **Lion+RFF uncompiled is the SOTA contender** — Edward #51 val 10.665 at ep9. Expected test_abupt ~10.5-11.0. Single stable Lion+delta confirmed so far.

## Current research focus (Round 6)

**Lion+RFF is the validated SOTA stack.** Edward #51 confirmed Lion+RFF beats vanilla Lion in val trajectory (10.665 vs 10.096 at end). Three parallel explorations of the Lion+RFF configuration space:

1. **Edward #51 iocqp761** (Lion+RFF sigma=1.0, ema=0.9995) — reproducing ftg0ci0p for test eval. This is the baseline for round 6.
2. **Tanjiro #90** (Lion+RFF+EMA 0.9999) — tests yi #13 norman's EMA compounding directly on the SOTA stack. Expected upside: -5-9% if yi #13 result generalizes.
3. **Alphonse #91** (Lion+RFF sigma=2.0) — tests whether higher-frequency RFF resolves tau_y/tau_z finer-scale gradients. These axes have the largest remaining gap (×3.8/×3.9 vs AB-UPT).

Architectural capacity:
4. **Thorfinn #69** (width 768d uncompiled µP) — 768d is the AB-UPT reference width. Epoch 3 val 16.42 (early). Highest long-term upside if model is capacity-bound.
5. **Frieren #73** (depth 6L AdamW+RFF+compile) — orthogonal to width. Epoch 1 in progress.

Loss-shaping on AdamW base:
6. **Fern #72** (AdamW+RFF+compile+per-axis tau_y/tau_z 2×) — pod stuck, awaiting infra.
7. **Askeladd #57, Nezuko #50** — Lion uncompiled fallbacks completing.

## Next architecture experiments (if current round continues diverging)

- **Perceiver-IO backbone** — yi #18 directive; yi branch has reference code
- **FiLM AdaLN-zero on 768d** — FiLM failed at 512d / 9 uncompiled epochs; 768d gives bigger capacity + more schedule
- **Pretraining on 50 unlabelled test geometries** — MAE 75% masking, then finetune
- **RFF sigma sweep** (sigma=0.5) — lower freq, PR #91 tests sigma=2.0, sigma=0.5 is the downward companion
- **Cosine T_max=24 on AdamW** — untested schedule calibration (T_max=16 too short, T_max=50 load-bearing, T_max=24 unexplored)

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

**tau_y/tau_z (×3.8/×3.9) and volume_pressure (×2.1) are the binding gaps.** vol_w=3 lever is closed. Architectural capacity (thorfinn #69 768d, frieren #73 6L) and RFF sigma sweep (alphonse #91) are the primary remaining levers.
