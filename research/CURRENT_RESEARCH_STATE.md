# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 10:15 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #115 (thorfinn compound lr=1e-4 + EMA=0.999), test_abupt 10.580

**Verified SOTA config (W&B `d03oghpp`):**
- lr=1e-4, weight_decay=5e-4, optimizer=lion, lion_beta1=0.9, lion_beta2=0.99
- use_ema=True, ema_decay=0.999, **lr_cosine_t_max=0** (NO cosine), lr_warmup_epochs=0
- model_layers=4, model_hidden_dim=512, model_heads=8, **model_slices=128**, model_mlp_ratio=4
- **train/eval surface_points=65536, train/eval volume_points=65536** (NOT 40k defaults)
- volume_loss_weight=1, surface_loss_weight=1, batch_size=4, compile_model=False

| Metric | tay SOTA (PR #115) | AB-UPT | Gap |
|---|---:|---:|---:|
| `abupt` mean | **10.580** | — | — |
| `surface_pressure` | **5.690** | 3.82 | ×1.5 |
| `wall_shear` | **10.419** | 7.29 | ×1.4 |
| `volume_pressure` | 12.740 | 6.08 | **×2.1** |
| `tau_x` | **8.908** | 5.35 | ×1.7 |
| `tau_y` | **12.491** | 3.65 | **×3.4** |
| `tau_z` | **13.071** | 3.63 | **×3.6** |

W&B run `d03oghpp` — best val 9.484 (ep8). val→test ratio 1.115.
**SOTA val trajectory:** 53.75 / 24.15 / 16.51 / 13.47 / 11.83 / 10.88 / 10.16 / 9.73 / 9.48

## Round 11 in-flight status (updated 2026-05-01 ~09:00 UTC)

| PR | Student | Hypothesis | W&B run | Val (latest) | Status |
|---|---|---|---|---|---|
| **#157** | nezuko | mlp_ratio=6 (architecture) | xuppho03 | **12.308** (ep6+, 178min) | running — most advanced |
| **#158** | alphonse | vol_pts=96k (vol_p attack) | yfi14f1w | **15.841** (ep4, 145min) | running — early |
| **#159** | fern | Lion β1=0.95 (momentum) | rnmwwg6q | **12.610** (ep5-6, 160min) | running — competitive |
| **#161** | askeladd | lion_beta2=0.999 (single delta) | tfumujfi | 31.772 (ep1, 87min) | running — too early to judge |
| **#162** | edward | model_dropout=0.05 (single delta) | e5l1r38b | 25.505 (ep1, 87min) | running — too early |
| **#163** | thorfinn | weight_decay=1e-3 (single delta) | 7rp28zrm | 26.420 (ep1, 77min) | running — too early |
| **#173** | tanjiro | cosine T_max=50 on SOTA stack (single delta) | TBD | — | **JUST ASSIGNED** — single delta from PR #115: add --lr-cosine-t-max 50 |
| **#147** | frieren | compound + wd=2e-3 | NONE | — | **POD STUCK** — no W&B run ever started; restart requested on issue #48 |

SOTA val trajectory reference: ep0=53.75 / ep1=24.15 / ep2=16.51 / ep3=13.47 / ep4=11.83 / ep5=10.88 / ep6=10.16 / ep7=9.73 / ep8=9.484

## Round 11 closed (per-axis tau weights) — REGRESSION

| PR | Student | Test | vs SOTA | W&B run | Conclusion |
|---|---|---:|---:|---|---|
| **#149** | tanjiro | **11.022** | **+4.2%** | rtajk53c | Per-axis tau weights W_y=W_z=1.5 regressed ALL metrics. **Direction CLOSED.** |

**Why tau axis weighting fails with Lion:** Lion uses sign-based momentum updates, not raw gradients. Per-channel loss multipliers shift gradient magnitudes but NOT the sign (direction) of each parameter update. The axis ratio is preserved regardless of loss multiplier — weighting is effectively neutralized. Do NOT pursue W=2.0/2.5 follow-ups via this mechanism.

## Round 10 closeouts (test_abupt)

| PR | Student | Test | vs SOTA | Why |
|---|---|---:|---:|---|
| #138 | nezuko (5L depth) | 11.213 | +5.98% | per-epoch time same as 4L; budget closed |
| #139 | fern (slices=256) | 12.389 | +17.1% | 98.8% VRAM, slice attention quadratic |
| #141 | askeladd (sw+vw+T_max) | 10.605 | +0.23% | val win but test/val ratio diverged |
| #142 | thorfinn (vol_w=2.0) | 11.721 | +10.78% | vol_p flat; surface badly hurt; ratio imbalance |
| #146 | edward (6L/256d) | 12.662 | +19.7% | depth no faster than 4L; closed |
| #148 | alphonse (lr=1.5e-4) | early | +40% | LR ceiling confirmed at 1e-4 |

## Key learnings accumulated

1. **Depth-swap family CLOSED** — 5L (#138 +6%) and 6L (#146 +19.7%) both regressed. Per-epoch wall-clock unchanged (attention-dominated); 4L/512d is the right shape for 9-ep budget.
2. **Loss weight family CLOSED without ratio rebalancing** — vol_w=2.0 alone hurts surface (+18%), vol_p flat. Paired sw=2+vw=2 (#141) is net-zero (+0.23%) with val/test ratio divergence.
3. **LR ceiling confirmed at 1e-4** — lr=1.5e-4 failed (+40%); lr=1e-4 is SOTA.
4. **Slices=256 OOM** — 98.8% VRAM usage, slice attention quadratic. Default 128 is safe ceiling.
5. **Lion compile divergence** — 9/9 compile+Lion combinations diverged in earlier rounds.
6. **Per-axis tau weighting CLOSED** — Lion sign-based updates neutralize per-channel loss weighting. W_y=W_z=1.5 regressed all metrics +4.2% (PR #149). Do not pursue higher weights.
7. **Next approach for tau_yz gap**: Target normalization (asinh/log transform), surface-tangent-frame head, or decoupled magnitude+direction prediction head. Need code changes beyond current train.py flags.

## Optimizer hyperparam map (tay confirmed)

| Param | Min tested | SOTA | Max tested | In-flight |
|---|---|---|---|---|
| lr | 5e-5 | **1e-4** | 1.5e-4 ❌ | ceiling found |
| ema_decay | 0.9999 | **0.999** | 0.998 ❌ | sweet spot |
| lion_beta1 | — | **0.9** | 0.95 (#159) | in-flight |
| lion_beta2 | — | **0.99** | 0.999 (#161) | in-flight |
| model_dropout | — | **0.0** | 0.05 (#162) | in-flight |
| weight_decay | — | **5e-4** | 1e-3 (#163) | in-flight |
| tau_axis_weights | — | **1.0** | 1.5 ❌ | **CLOSED** — Lion sign mechanism neutralizes |

## Next research directions (priority order)

1. **Wait for fern #159 and nezuko #157** — both in ep5-6 range showing 12.3-12.6 val. Need ep8-9 to compare with SOTA val=9.484. These are the most advanced round 11 experiments.
2. **Alphonse #158 vol_pts=96k** — dominant vol_p binding gap attack (×2.1 vs AB-UPT). At ep4=15.84. Need ep8-9 final.
3. **Tanjiro #173 cosine T_max=50** — clean single-delta from PR #115 SOTA. Cosine was a winner on older stack (PR #110) but never tested on lr=1e-4+EMA=0.999. Good chance to stack another improvement.
4. **Frieren pod restart needed** — pod stuck at init since 23:14 UTC Apr 30. PR #147 (wd=2e-3) has NEVER started. Requires human `kubectl rollout restart`. Escalated on issue #48.
5. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Biggest untested architectural lever. Not yet assigned to any running pod.
6. **Volume loss weight isolation** — clean `--volume-loss-weight 1.5-2.0` experiment WITHOUT tau coupling side effects. Previous vol_w=2.0 hurt surface; hypothesis: lighter volume weight (1.5) with isolated testing might find a sweet spot.
7. **Tau_yz binding gap (code-change approach)** — must bypass Lion neutralization. Options: (a) asinh output normalization for tau_y/tau_z (normalizes small-magnitude components), (b) surface-tangent-frame prediction head (predicts in a frame where components are better conditioned), (c) decoupled magnitude+direction head. All require train.py modifications by a student.
8. **Observe round 11 askeladd/edward/thorfinn ep3+ signals** — lion_beta2, dropout, wd all at ep1 only. Need 2-3 more epochs to evaluate trajectories vs SOTA reference.
