# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 (updated end of Round 12 wave assignment)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #115 (thorfinn compound lr=1e-4 + EMA=0.999), test_abupt 10.580

**Verified SOTA config (W&B `d03oghpp`):**
- lr=1e-4, weight_decay=5e-4, optimizer=lion, lion_beta1=0.9, lion_beta2=0.99
- use_ema=True, ema_decay=0.999, **lr_cosine_t_max=0** (fallback to `T_max=epochs`; with epochs=50 → cosine over 50, essentially flat over the actual 9-epoch run), lr_warmup_epochs=0
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

## In-flight (8/8 students running — all slots filled, Round 12)

Last updated: 2026-05-01 ~19:30 (end of this advisor session).

| PR | Student | Hypothesis | W&B group | Status |
|---|---|---|---|---|
| **#202** | tanjiro | lr_cosine_t_max=9 (genuine re-run, no confounds) | `tay-round12-cosine-tmax9` | running — was ep~2 at last check; NEGATIVE EXPECTED given edward #195 final result |
| **#203** | thorfinn | weight_decay=2.5e-4 | `tay-round12-wd-2p5e-4` | running — ep~3 at last check, val=19.327; wait for ep5+ |
| **#204** | frieren | vol_loss_weight=2.0 | `tay-round12-vol-loss-weight-2p0` | running — ep~3, val=10.969; wait for ep5+ |
| **#206** | alphonse | surface_points 64k→96k | `tay-round12-surface-pts-96k` | running — very early (ep~1-2) |
| **#222** | fern | lr_warmup=1ep (warmup from 0 over first epoch) | `tay-round12-lr-warmup-1ep` | running — ep~1, val=41.93%; very early |
| **#231** | nezuko | model_slices=64 (halve attention slices) | `tay-round12-model-slices-64` | just assigned this session |
| **#232** | askeladd | model_heads=4 (halve attention heads; NEVER TESTED) | `tay-round12-model-heads-4` | just assigned this session |
| **#233** | edward | model_layers=3 (reduce depth by 1; NEVER TESTED) | `tay-round12-model-layers-3` | just assigned this session |

## Round 12 — Closed / retired PRs

| PR | Student | Result | vs SOTA | Conclusion |
|---|---|---|---|---|
| **#195** | edward | test_abupt=10.809% | +2.16% | T_max=9 cosine **NEGATIVE** — LR fell near-zero by ep8-9 while model still converging. Val trajectory: 9.710% at ep9. 6/7 axes regressed. |
| **#194** | askeladd | test_abupt=11.619% | +9.8% | EMA=0.9995 **NEGATIVE** — slower averaging worse than 0.999. **EMA space fully closed.** |

**EMA space summary (all four values tested):**
| EMA | val_abupt | test_abupt | Outcome |
|---|---|---|---|
| 0.998 | ~10.5% | — | Negative |
| **0.999** | **9.484%** | **10.580%** | **SOTA** |
| 0.9995 | — | 11.619% | Negative |
| 0.9999 | — | ~11%+ | Negative |

**T_max space (cosine schedule):**
| T_max | val_abupt | Outcome |
|---|---|---|
| **50** (≈flat for 9ep) | **9.484%** | **SOTA** |
| 9 | 9.710% (edward #195) | Negative — LR collapses too early |

T_max=14-18 range untested; if tanjiro #202 also confirms T_max=9 negative, that entire faster-decay direction is closed. Gentle annealing (T_max=14-20) remains as a possible next probe.

## Round 11 — Closed PRs

| PR | Student | Test | vs SOTA | Conclusion |
|---|---|---:|---:|---|
| **#196** | frieren | running → now PR #204 | — | vol_loss_weight=2.0 (same hypothesis, fresh run) |
| **#189** | fern | closed | — | lion_beta1=0.8 negative — now on lr-warmup-1ep (#222) |
| **#187** | nezuko | closed | — | vol_loss_weight=1.5 final result; nezuko now on model-slices=64 (#231) |
| **#149** | tanjiro | 11.022 | +4.2% | Per-axis tau weights W_y=W_z=1.5 regressed all metrics. CLOSED. |

## Prior closed families (carried forward)

| Family | Status | Best tested | Note |
|---|---|---|---|
| Depth (model_layers) | **in-flight as #233** | 4L SOTA | 3L and 5L/6L untested vs new baseline |
| FFN width (mlp_ratio) | CLOSED | 4 SOTA | mlp_ratio=6 +6.4%; ceiling found |
| Attention heads | **in-flight as #232** | 8 SOTA | model_heads=4 **NEVER TESTED** |
| Attention slices | **in-flight as #231** | 128 SOTA | slices=256 OOM; slices=64 **NEVER TESTED** |
| LR | CLOSED | 1e-4 SOTA | lr=1.5e-4 +40%; ceiling confirmed |
| EMA decay | CLOSED | 0.999 SOTA | All four values tested; 0.999 confirmed optimal |
| lion_beta2 | CLOSED | 0.99 SOTA | 0.999 +18.7%; 100-step window optimal for 9ep |
| lion_beta1 | partially tested | 0.9 SOTA | 0.8 (fern #189) negative; 1.0 untested but default is 0.9 |
| weight_decay | **in-flight as #203** | 5e-4 SOTA | wd=1e-3 +4.5% (PR #163); now testing wd=2.5e-4 (down direction) |
| vol_loss_weight | **in-flight as #204** | 1.0 SOTA | 2.0 re-run via frieren; 1.5 was negative (PR #187) |
| vol_points | CLOSED | 65536 SOTA | 96k worse on every metric (PR #186) |
| surface_points | **in-flight as #206** | 65536 SOTA | 96k being tested (alphonse) |
| dropout | CLOSED | 0.0 SOTA | 0.05 +4.24%; model underfits, dropout harmful |
| tau_axis_weights | CLOSED | 1.0 SOTA | Lion sign mechanism neutralizes per-channel weighting |
| compile_model | CLOSED | False SOTA | compile+Lion diverged in all 9 tried combos |
| lr_cosine_t_max=9 | NEGATIVE (edward #195, tanjiro #202 pending) | 50 SOTA | LR collapses too early at T_max=9 |
| lr_warmup | **in-flight as #222** | 0 SOTA | 1ep warmup being tested |

## Key architectural levers — audit status (Round 12)

The three main untested architectural dimensions as of Round 12:
1. **model_heads** — `--model-heads 4` assigned to askeladd #232 (NEVER VARIED across ~60+ experiments)
2. **model_layers** — `--model-layers 3` assigned to edward #233 (NEVER VARIED in tay track)
3. **model_slices** — `--model-slices 64` assigned to nezuko #231 (halved from 128; 256 was OOM)

These are the highest-priority unexplored architectural levers after the FFN/depth/dropout/slices family sweeps.

## Next research directions (priority order after Round 12 closes)

1. **Compound stack** — once round 12 closes, stack all winners on SOTA #115 base (e.g., if model_heads=4 + surface_pts=96k both win, test them together).
2. **T_max gentle annealing (T_max=14-20)** — if tanjiro #202 confirms T_max=9 is negative, the T_max=9 exact value was too aggressive; a gentler T_max=14-18 might still find the sweet spot.
3. **lr_warmup tuning** — if fern #222 (warmup=1ep) wins, try warmup=0.5ep or 2ep to narrow down. If loses, warmup confirmed unhelpful.
4. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Biggest untested architectural lever. Reserve for idle slot after round 12 wave completes.
5. **Tau_yz binding gap (code-change approach)** — bypass Lion sign-neutralization: (a) asinh output normalization for tau_y/tau_z, (b) surface-tangent-frame prediction head, (c) decoupled magnitude+direction head. All require train.py modifications — student-side.
6. **Weight decay refinement** — if thorfinn #203 (wd=2.5e-4) wins, explore wd=1e-4. If loses, wd=5e-4 confirmed optimal direction confirmed closed both ways.
7. **lion_beta1 upper probe** — beta1=0.95 not tested; SOTA=0.9, 0.8 was negative. Could try 0.95 once beta1 space is clear.
8. **Plateau escalation option**: If Round 12 closes with no winners, consider **architecture overhaul** — move from current Transformer to a graph neural network backbone (e.g., PointGNN), physics-informed constraints, or multi-scale hierarchical attention. The conservative single-delta space is nearly exhausted.

## Key learnings (cumulative)

1. **Depth-swap family CLOSED** — 5L (#138 +6%) and 6L (#146 +19.7%) both regressed. Per-epoch wall-clock unchanged; 4L/512d is the right shape for 9-ep budget. 3L in-flight (#233).
2. **FFN-width family CLOSED** — mlp_ratio=6 (#157 +6.4%) regressed; capacity expansion via FFN width is saturated.
3. **LR ceiling confirmed at 1e-4** — lr=1.5e-4 failed (+40%); lr=1e-4 is SOTA.
4. **Slices=256 OOM** — 98.8% VRAM usage, slice attention quadratic. Default 128 is safe ceiling.
5. **Lion compile divergence** — 9/9 compile+Lion combinations diverged in earlier rounds.
6. **Per-axis tau weighting CLOSED** — Lion sign-based updates neutralize per-channel loss weighting. W_y=W_z=1.5 regressed all metrics +4.2% (PR #149). Do not pursue higher weights.
7. **`--lr-cosine-t-max 0` is a footgun** — fallback to `T_max=epochs` means with epochs=50 → essentially flat LR for 9 actual training epochs. Always specify `--lr-cosine-t-max 50` explicitly in 9-epoch single-deltas.
8. **Volume sampling density (vol_pts) CLOSED** — 96k worse on every metric (PR #186). 64k is at/near optimum.
9. **Dropout is a dead end** — model_dropout=0.05 (#162, +4.24%) WORSENED val/test ratio. The model is in underfitting regime at 9 epochs; adding regularization noise makes things worse.
10. **lion_beta2=0.999 is a dead end** — momentum window of 1000 effective steps is too wide for 9-ep training. beta2=0.99 (100-step effective window) is optimal for this budget.
11. **EMA decay fully closed** — 0.998, 0.9995, 0.9999 all negative; 0.999 is the confirmed optimum. No further EMA probes needed.
12. **T_max=9 cosine is negative** — LR falls to near-zero while model still actively converging. Confirmed by edward #195; tanjiro #202 running as variance probe.

## Optimizer hyperparam map (tay confirmed)

| Param | Min tested | SOTA | Max tested | Status |
|---|---|---|---|---|
| lr | 5e-5 | **1e-4** | 1.5e-4 ❌ | CLOSED — ceiling at 1e-4 |
| ema_decay | 0.9999 ❌ | **0.999** | 0.9995 ❌, 0.998 ❌ | **FULLY CLOSED** |
| lion_beta1 | 0.8 ❌ | **0.9** | — | partially closed (lower direction neg; upper untested) |
| lion_beta2 | — | **0.99** | 0.999 ❌ | **CLOSED** |
| model_dropout | — | **0.0** | 0.05 ❌ | **CLOSED** |
| weight_decay | 2.5e-4 (in-flight #203) | **5e-4** | 1e-3 ❌ | in-flight downward |
| volume_loss_weight | — | **1.0** | 1.5 ❌, 2.0 (in-flight #204) | 1.5 closed; 2.0 in-flight |
| volume_points | — | **65536** | 96000 ❌ | **CLOSED** |
| surface_points | — | **65536** | 96000 (in-flight #206) | in-flight |
| tau_axis_weights | — | **1.0** | 1.5 ❌ | **CLOSED** |
| mlp_ratio | — | **4** | 6 ❌ | **CLOSED** |
| model_layers | — | **4** | 3 (in-flight #233) | in-flight (lower direction) |
| model_heads | — | **8** | 4 (in-flight #232) | in-flight (lower direction) |
| model_slices | — | **128** | 64 (in-flight #231), 256 ❌ OOM | in-flight (lower direction) |
| lr_cosine_t_max | — | **50 (≈flat)** | 9 ❌ | negative at T_max=9; gentle range (14-20) untested |
| lr_warmup_epochs | — | **0** | 1 (in-flight #222) | in-flight |
