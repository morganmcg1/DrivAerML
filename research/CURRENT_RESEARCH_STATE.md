# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 09:30 UTC
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
| **#71** | fern | AdamW + RFF + compile + per-axis tau_y/tau_z weights (2×) — stable base for per-axis mechanism | JUST ASSIGNED |
| **#70** | tanjiro | Lion (lr=2.5e-5) + RFF + compile — half-LR to suppress divergence | Running ~56min, val 22.84 (ep3), healthy descent |
| **#69** | thorfinn | Lion (lr=3.3e-5 µP) + width 768d / 12h — architectural capacity uplift | Running ~78min, val 71.04 (ep2), uncompiled |
| **#55** | alphonse | AdamW + RFF + compile + volume-loss-weight 3.0 | Running ~182min, val 16.56 (ep10), descending |
| **#51** | edward | Lion (lr=5e-5) + RFF sigma=1.0 (uncompiled) — **SOTA CONTENDER** | Running ~181min, val **13.46** (ep5), descending |
| **#68** | frieren | Lion (lr=5e-5) + vol_w=3.0 | Running ~229min, DIVERGED (best 14.43 ep5), finishing |
| **#57** | askeladd | Recovery nocompile run after #57 diverged | Running nocompile fallback ~56min |
| **#50** | nezuko | Recovery nocompile run after #50 Lion+compile diverged | Running nocompile fallback ~74min |

## Key closed/merged experiments

| PR | Outcome | Why |
|---|---|---|
| **arm B** (no PR) | **NEW SOTA 11.30** | Lion lr=5e-5/wd=5e-4 — paper config was wrong by −27% |
| #46 alphonse | MERGED — 14.55 | AdamW + RFF + compile → epoch 16 |
| #54 fern | CLOSED — +137% regression | Lion + per-axis tau_y/tau_z weights (2×): diverged ep4 test 26.83; mechanism works but Lion fragile |
| #52 tanjiro | CLOSED — +16.8% vs SOTA | Lion+RFF+compile: RFF+compile add zero orthogonal signal; compile diverges |
| #56 thorfinn | CLOSED — +13% vs #46 | Cosine T_max=16 on AdamW; schedule miscalib was load-bearing, not a bug |
| #42 frieren | CLOSED — +40% regression | Lion + sq_rel_l2: Lion sign-update + per-case ratio variance = divergence |
| #49 askeladd | CLOSED — +35% regression | Grad-clip 5.0; doesn't compose with Lion |
| #47 thorfinn | CLOSED — +36% regression | Bilateral aug → off asymmetric test distribution |
| #43 fern | CLOSED — +15.7% regression | Multi-scale RFF redundant on `[−1,4]m` domain |
| #41 askeladd | CLOSED — +22.5% regression | Eval-time tangential projection destroys tau |
| #35 nezuko | CLOSED — +13.7% regression | ANP uncompilable (inductor bug) |
| #44 edward | CLOSED — +18.8% regression | Cosine EMA over 50 epochs → 18% of schedule |

## Critical learnings

1. **Lion paper config is wrong for this dataset** — `lr=5e-5, wd=5e-4` (AdamW-equivalent) crushes paper config by −27%.
2. **Lion is fragile to ALL non-trivial modifications** — confirmed 7× across PRs #42, #50, #52, #54, #57, #68, #56. Any change to: compile, loss function (sq_rel_l2, per-axis weighting, vol_w increase), or schedule (cosine T_max=16) diverges. Stable Lion variants: vanilla Lion (vnb7oheo, 11.30) and Lion+RFF (PR #51 edward, descending at val 13.46).
3. **Lion+compile mechanism:** sign() update bounded only by lr. Compile's reduced gradient noise → more deterministically biased signs → effectively higher LR than uncompiled. Late-training divergence when loss gradients small. Onset: epoch 4-7 depending on modification.
4. **Cosine T_max=16 on AdamW is a regression** (PR #56: +13% vs #46). T_max=50 fallback (LR ≈88% at end) was load-bearing — model is undertrained at 16 epochs. Future cosine: T_max≥24 or budget extension only.
5. **Compile gives epoch 16** (vs ~8-9 uncompiled in 270 min). T_max for schedules = 16 if compiled.
6. **Per-axis loss weighting mechanism works at gradient level** — PR #54 showed tau_y/tau_x ratio improved (1.33 vs 1.45) before divergence. Test on stable AdamW base (PR #71 fern).
7. **Multi-scale RFF redundant** on DrivAerML surface domain — `[−1,4]m` already covered by σ=1.0.
8. **Cosine EMA needs `--ema-total-epochs 16`** (not 50).
9. **ANP decoder uncompilable** — inductor dynamic-shapes bug.
10. **Bilateral aug pushes model off asymmetric test distribution** — closed door.
11. **Lion + per-case-normalized loss** — never compose sq_rel_l2 / rel_l2 / any 1/y² loss with Lion at lr ≥ 5e-5.

## Current research focus (Round 4/5)

**Lion divergence landscape is fully mapped.** 7 independent observations confirm Lion is fragile to ANY non-trivial modification at lr=5e-5. Three parallel strategies now running:

1. **Edward #51** (Lion+RFF uncompiled, val 13.46% at ep5) — the only stable Lion+delta to date. This is the **SOTA contender** for the round. If it beats 11.30, we have a new SOTA from RFF alone on Lion.
2. **Tanjiro #70** (Lion+compile+half-LR 2.5e-5) — testing whether halving LR suppresses Lion+compile divergence. Stable at epoch 3 (val 22.84). If stable through epoch 6, this opens the Lion+compile frontier.
3. **Thorfinn #69** (width 768d uncompiled, µP lr=3.3e-5) — architectural capacity lever. Epoch 2 val 71% (expected, high early). 768d is the AB-UPT reference width. This is the highest-upside lever if model is capacity-bound.
4. **Fern #71** (AdamW+RFF+compile + per-axis tau_y/tau_z 2×) — tests per-axis weighting on stable base. Should beat PR #46 (14.55) if tau_y/tau_z gap is attentional.
5. **Alphonse #55** (AdamW+RFF+compile+vol_w=3, ep10 val 16.56%) — trajectory similar to PR #46, converging toward 14-15%.
6. **Frieren #68** (Lion+vol_w=3, diverged best 14.43) — awaiting test eval from best-val checkpoint, expected ~16%.
7. **Askeladd #57, Nezuko #50** — autonomous nocompile fallback recoveries in progress.

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
