# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 07:45 UTC
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

**All future Lion experiments must use `--lr 5e-5 --weight-decay 5e-4`.**

## Active assignments (8 students, all DDP8)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#69** | thorfinn | Lion (lr=3.3e-5 µP) + width 768d / 12h — architectural capacity uplift | JUST ASSIGNED |
| **#50** | nezuko | Lion (lr=5e-5) + compile (single delta) | Running `r6mn2x5c`, val 25% at epoch 3 |
| **#51** | edward | Lion (lr=5e-5) + RFF sigma=1.0 (single delta) | Running `ftg0ci0p`, val 35% at epoch 2 |
| **#52** | tanjiro | Lion (lr=5e-5) + RFF + compile (full triple-stack) | Running `5o1frm3u`, DIVERGED (best 12.22), awaiting test eval |
| **#54** | fern | Lion (lr=5e-5) + per-axis tau_y/tau_z loss weighting (2×/2×) | Running `8zhjetjt`, DIVERGED (best 26.38), awaiting test eval |
| **#55** | alphonse | RFF + compile + volume-loss-weight 3.0 (AdamW base) | Running `lahk19ws`, val 30% at epoch 3 |
| **#57** | askeladd | Lion (lr=5e-5) + compile + cosine T_max=16 | Running `zz07t1sh`, DIVERGED (best 42.58 at epoch 2 → 93% at epoch 4) |
| **#68** | frieren | Lion (lr=5e-5) + vol_w=3.0 | Running `daayn9kb`, DIVERGED (best 14.43 at epoch 5 → 88% at epoch 6) |

## Key closed/merged experiments

| PR | Outcome | Why |
|---|---|---|
| **arm B** (no PR) | **NEW SOTA 11.30** | Lion lr=5e-5/wd=5e-4 — paper config was wrong by −27% |
| #46 alphonse | MERGED — 14.55 | AdamW + RFF + compile → epoch 16 |
| #56 thorfinn | CLOSED — +13% vs #46 | Cosine T_max=16 on AdamW; schedule miscalib was load-bearing, not a bug |
| #42 frieren | CLOSED — +40% regression | Lion + sq_rel_l2: Lion sign-update + per-case ratio variance = divergence |
| #49 askeladd | CLOSED — +35% regression | Grad-clip 5.0; doesn't compose with Lion |
| #47 thorfinn | CLOSED — +36% regression | Bilateral aug → off asymmetric test distribution |
| #43 fern | CLOSED — +15.7% regression | Multi-scale RFF redundant on `[−1,4]m` domain |
| #41 askeladd | CLOSED — +22.5% regression | Eval-time tangential projection destroys tau |
| #37 thorfinn | CLOSED — +14.2% regression | Bilateral TTA-eval only |
| #35 nezuko | CLOSED — +13.7% regression | ANP uncompilable (inductor bug) |
| #44 edward | CLOSED — +18.8% regression | Cosine EMA over 50 epochs → 18% of schedule |
| #34 frieren | CLOSED Round-1 | FiLM didn't transfer at 9 uncompiled epochs |
| #32 edward | CLOSED Round-1 | LR schedule over 50 epochs, budget miscalibration |

## Critical learnings

1. **Lion paper config is wrong for this dataset** — `lr=5e-5, wd=5e-4` (AdamW-equivalent) crushes paper config by −27%.
2. **Lion+compile DIVERGES late-training** — observed across PRs #42, #52, #68, #57. Lion's sign-based update produces constant per-step movement bounded only by `lr`. With compile's precision, gradient noise is lower → signs more correlated across steps → effectively higher LR than uncompiled. Late-training divergence when loss gradients become small. **SOTA Lion (vnb7oheo, 11.30) was UNCOMPILED**. Compose Lion with architecture changes on uncompiled base first.
3. **Cosine T_max=16 on AdamW is a regression** (PR #56: +13% vs #46). The PR #46 "miscalibration" (T_max=50 over 16 epochs → LR still ~88% at end) was load-bearing. The model is UNDERTRAINED at 16 epochs with AdamW lr=5e-5, not overoptimized. Cosine should only be tried with T_max≥24 or budget extension.
4. **Compile gives epoch 16** (vs ~28 uncompiled in 290 min). T_max for schedules should be 16 if compile, not 50.
5. **Volume_pressure regression from AdamW + compile** — Lion's sign-update normalizes per-channel gradient; already gives vol_p=12.76 < PR #39's 13.83.
6. **RFF orthogonal to optimizer** — Lion+RFF (PR #51) in-flight; expect ~10.5–10.8.
7. **Multi-scale RFF redundant** on DrivAerML surface domain — `[−1,4]m` already covered by σ=1.0.
8. **Cosine EMA needs `--ema-total-epochs 16`** (not 50). Edward #44 used 50 → 18% of schedule.
9. **ANP decoder uncompilable** — inductor dynamic-shapes bug.
10. **Bilateral aug pushes model off the asymmetric test distribution** — closed door.
11. **Lion + per-case-normalized loss diverges** — sign-update + high-tail per-batch ratio variance permanently corrupts parameter state. Never compose sq_rel_l2 / rel_l2 / any 1/y² loss with Lion at lr ≥ 5e-5.

## Current research focus (Round 4)

**Lion+compile divergence is a dominant failure mode.** 4 of 8 active runs have diverged (PRs #52, #54, #57, #68). This is not random — Lion's sign-update + compile's precision = systematically higher effective LR. Near-term priority:

1. **PR #69 thorfinn** — width 768d (µP lr=3.3e-5), **uncompiled** (deliberately). Bold architectural lever targeting capacity-bound binding gaps (vol ×2.1, tau_y/tau_z ×3.8/3.9). AB-UPT reference itself uses 768d.
2. **PRs #50 #51 #55** (nezuko, edward, alphonse) — early-stage runs (50-67 min), val 25-35%. All on different subsets of the lever stack. Need to monitor whether Lion+compile (#50) diverges like #52/#57.
3. **Diverged runs #52 #54 #57 #68** — await test eval from best-val checkpoint. Predicted test from best-val: tanjiro #52 ~13.4%, fern #54 ~28%, askeladd #57 ~44%, frieren #68 ~16%.

## Next architecture experiments (if current round continues diverging)

- **Perceiver-IO backbone** — yi #18 directive; yi branch has reference code
- **Depth increase 4L → 6L** (orthogonal to width, cheaper)
- **FiLM AdaLN-zero on 768d** — FiLM failed at 512d / 9 uncompiled epochs; 768d gives bigger capacity + more schedule
- **Pretraining on 50 unlabelled test geometries** — MAE 75% masking, then finetune

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

Surface/wall_shear/tau_x gaps ×1.6–1.8. **volume_pressure (×2.1) and tau_y/tau_z (×3.8/×3.9) are the binding gaps.** Width 768d (PR #69) is the most direct next lever.
