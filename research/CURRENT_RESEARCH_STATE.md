# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 06:40 UTC
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

W&B run `d03oghpp` — best val 9.484 (ep9). val→test ratio 1.115.
**SOTA val trajectory:** 53.75 / 24.15 / 16.51 / 13.47 / 11.83 / 10.88 / 10.16 / 9.73 / 9.48

## Round 11 in-flight status

| PR | Student | Hypothesis | Status | Signal |
|---|---|---|---|---|
| **#161** | askeladd | lion_beta2=0.999 (single delta) | running, launched | no val yet |
| **#162** | edward | model_dropout=0.05 (single delta) | running, launched | no val yet |
| **#163** | thorfinn | weight_decay=1e-3 (single delta) | just assigned | — |
| #157 | nezuko | mlp_ratio=6 (architecture) | running step=7828 | val ep3=25.28 vs SOTA ep3=16.51 (+53%) |
| #158 | alphonse | vol_pts=96k (vol_p attack) | running step=4476 | val ep1=41.64 vs SOTA ep1=24.15 (+72%, slow/96k) |
| #159 | fern | Lion β1=0.95 (momentum) | running step=6759 | val ep2=23.82 vs SOTA ep2=16.51 (+44%) |
| #147 | frieren | compound + wd=2e-3 | **POD STUCK** (running old PR #134 round9) | no round11 W&B run |
| #149 | tanjiro | per-axis tau weights W=1.5 | **POD STUCK** (GPU 0%, iter 167 since 02:29) | no W&B run |

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
6. **Vol_p test improvement possible** — askeladd #141 achieved -0.60% on vol_p test with sw+vw stack; but tau_y regressed. Decoupling is needed.

## Optimizer hyperparam map (tay confirmed)

| Param | Min tested | SOTA | Max tested | In-flight |
|---|---|---|---|---|
| lr | 5e-5 | **1e-4** | 1.5e-4 ❌ | ceiling found |
| ema_decay | 0.9999 | **0.999** | 0.998 ❌ | sweet spot |
| lion_beta1 | — | **0.9** | 0.95 (#159) | partial |
| lion_beta2 | — | **0.99** | 0.999 (#161) | untested |
| model_dropout | — | **0.0** | 0.05 (#162) | untested |
| weight_decay | — | **5e-4** | 1e-3 (#163) | untested |

## Next research directions (priority order)

1. **Wait for alphonse #158 vol_pts=96k** — dominant binding gap attack (vol_p ×2.1). Just launched, early.
2. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Biggest untested architectural lever. Tanjiro/frieren pods stuck; reassign to working pods when available.
3. **Per-axis tau loss weights** — tau_y/tau_z are ×3.4-3.6 binding gaps. Requires code change (no train.py flag). Tanjiro #149 assigned but pod stuck.
4. **Observe round11 signals** — lion_beta2, dropout, wd all untested. Need ep3+ data.
5. **Human team action needed** — frieren and tanjiro pods stuck for 4+ hours. Cannot fix from advisor side.
