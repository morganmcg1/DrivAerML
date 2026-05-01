# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 (updated — Round 13 PRs added, closed PRs #202/#203/#204 retired)
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

## In-flight (8/8 students running — all slots filled, Round 12 + Round 13)

Last updated: 2026-05-01 ~20:00 (CURRENT_RESEARCH_STATE refresh after Round 13 assignment).

| PR | Round | Student | Hypothesis | W&B group | Status |
|---|---|---|---|---|---|
| **#206** | 12 | alphonse | surface_points 64k→96k | `tay-round12-surface-pts-96k` | running — ~68% complete (~step 16,662/24,480); val projected ~9.83% |
| **#222** | 12 | fern | lr_warmup=1ep (warmup from 0 over first epoch) | `tay-round12-lr-warmup-1ep` | running — ~50% complete; strongest slope of active runs |
| **#231** | 12 | nezuko | model_slices=64 (halve attention slices) | `tay-round12-model-slices-64` | running — wild-card; extreme recovery slope from high starting val |
| **#232** | 12 | askeladd | model_heads=4 (halve attention heads; NEVER TESTED) | `tay-round12-model-heads-4` | running |
| **#233** | 12 | edward | model_layers=3 (reduce depth by 1; NEVER TESTED) | `tay-round12-model-layers-3` | running |
| **#240** | 13 | frieren | wider FFN mlp_ratio=8 (vs. SOTA=4, 6 was negative) | `tay-round13-mlp-ratio-8` | running — pre-first-eval |
| **#241** | 13 | tanjiro | width scaling 512→768d with µP-scaled LR | `tay-round13-hidden-dim-768` | running — pre-first-eval |
| **#242** | 13 | thorfinn | dropout=0.1 re-test on SOTA stack (0.05 was negative) | `tay-round13-dropout-0p1` | running — pre-first-eval |

## Round 12 — Closed / retired PRs

| PR | Student | Result | vs SOTA | Conclusion |
|---|---|---|---|---|
| **#202** | tanjiro | val=9.710% (closed 2026-05-01) | NEGATIVE | T_max=9 cosine **CONFIRMED NEGATIVE** — variance probe agrees with edward #195. Cosine T_max=9 direction fully closed. |
| **#203** | thorfinn | val ~19.3% at ep3 (closed 2026-05-01) | NEGATIVE +11.9% | wd=2.5e-4 **NEGATIVE** — lower weight_decay regressed badly. wd=5e-4 confirmed as optimum both directions closed. |
| **#204** | frieren | val ~10.97% at ep3 (closed 2026-05-01) | NEGATIVE | vol_loss_weight=2.0 **NEGATIVE** re-run — 1.5 and 2.0 both worse. vol_loss_weight=1.0 SOTA confirmed. |
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

T_max=9 is now **fully confirmed negative** (edward #195 + tanjiro #202 both closed negative). Faster-decay direction is closed. Gentle annealing (T_max=14-20) remains as a possible next probe.

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
| weight_decay | **CLOSED** | 5e-4 SOTA | wd=2.5e-4 ❌ (#203) and wd=1e-3 ❌ (#163) both negative; 5e-4 confirmed optimum |
| vol_loss_weight | **CLOSED** | 1.0 SOTA | 1.5 ❌ (PR #187), 2.0 ❌ (#204) both negative; 1.0 confirmed optimum |
| vol_points | CLOSED | 65536 SOTA | 96k worse on every metric (PR #186) |
| surface_points | **in-flight as #206** | 65536 SOTA | 96k being tested (alphonse) |
| dropout | CLOSED | 0.0 SOTA | 0.05 +4.24%; model underfits, dropout harmful |
| tau_axis_weights | CLOSED | 1.0 SOTA | Lion sign mechanism neutralizes per-channel weighting |
| compile_model | CLOSED | False SOTA | compile+Lion diverged in all 9 tried combos |
| lr_cosine_t_max=9 | **CLOSED NEGATIVE** (edward #195 + tanjiro #202 both confirmed) | 50 SOTA | LR collapses too early at T_max=9; gentle range (14-20) still untested |
| lr_warmup | **in-flight as #222** | 0 SOTA | 1ep warmup being tested |

## Key architectural levers — audit status (Round 12)

The three main untested architectural dimensions as of Round 12:
1. **model_heads** — `--model-heads 4` assigned to askeladd #232 (NEVER VARIED across ~60+ experiments)
2. **model_layers** — `--model-layers 3` assigned to edward #233 (NEVER VARIED in tay track)
3. **model_slices** — `--model-slices 64` assigned to nezuko #231 (halved from 128; 256 was OOM)

These are the highest-priority unexplored architectural levers after the FFN/depth/dropout/slices family sweeps.

## Next research directions (priority order after Rounds 12+13 close)

**Round 13 currently testing:** mlp_ratio=8 (frieren #240), hidden_dim=768 (tanjiro #241), dropout=0.1 (thorfinn #242).

1. **Compound stack** — once round 12+13 close, identify all winning dimensions and stack them on SOTA #115 base. Architecture winners (e.g., model_heads=4 + surface_pts=96k) are often orthogonal and compound.
2. **T_max gentle annealing (T_max=14-20)** — T_max=9 is now fully confirmed negative; gentle decay midpoint is still untested. T_max=14 or T_max=18 could provide cosine benefit without premature LR collapse.
3. **lr_warmup tuning** — if fern #222 (warmup=1ep) wins, try warmup=0.5ep or 2ep. If loses, warmup direction is closed.
4. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Biggest untested architectural lever. Reserve for idle slot when Round 12/13 close.
5. **Tau_yz binding gap (code-change approach)** — bypass Lion sign-neutralization: (a) asinh output normalization for tau_y/tau_z, (b) surface-tangent-frame prediction head, (c) decoupled magnitude+direction head. All require train.py modifications — student-side.
6. **lion_beta1 upper probe** — beta1=0.95 not tested; SOTA=0.9, 0.8 was negative. Could try 0.95 once architectural space is better mapped.
7. **model_hidden_dim variants beyond 768** — if tanjiro #241 (768d) wins, explore 1024d. If loses, 512d confirmed optimal; capacity doesn't help at 9ep.
8. **Plateau escalation option**: If Rounds 12+13 close with no winners, consider **architecture overhaul** — move from current Transformer to a graph neural network backbone (e.g., PointGNN), physics-informed constraints, or multi-scale hierarchical attention. The conservative single-delta space is nearly exhausted. Invoke researcher-agent for fresh ideas.

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
12. **T_max=9 cosine is confirmed negative** — Confirmed by both edward #195 and tanjiro #202 (variance probe). LR collapses too early; T_max=50 (≈flat for 9ep) remains optimal. Gentle range T_max=14-20 still untested.
13. **weight_decay space CLOSED** — wd=2.5e-4 (PR #203, −11.9%) and wd=1e-3 (PR #163, +4.5%) both negative. wd=5e-4 confirmed optimum; both directions exhausted.
14. **vol_loss_weight space CLOSED** — 1.5 (PR #187) and 2.0 (PR #204) both worse than 1.0. Volume loss weighting doesn't help; loss balance between vol/surface at 1:1 is confirmed optimal.

## Optimizer hyperparam map (tay confirmed)

| Param | Min tested | SOTA | Max tested | Status |
|---|---|---|---|---|
| lr | 5e-5 | **1e-4** | 1.5e-4 ❌ | CLOSED — ceiling at 1e-4 |
| ema_decay | 0.9999 ❌ | **0.999** | 0.9995 ❌, 0.998 ❌ | **FULLY CLOSED** |
| lion_beta1 | 0.8 ❌ | **0.9** | — | partially closed (lower direction neg; upper untested) |
| lion_beta2 | — | **0.99** | 0.999 ❌ | **CLOSED** |
| model_dropout | — | **0.0** | 0.05 ❌ | **CLOSED** |
| weight_decay | 2.5e-4 ❌ (#203) | **5e-4** | 1e-3 ❌ | **CLOSED** — both directions tested negative |
| volume_loss_weight | — | **1.0** | 1.5 ❌, 2.0 ❌ (#204) | **CLOSED** — 1.5 and 2.0 both negative |
| volume_points | — | **65536** | 96000 ❌ | **CLOSED** |
| surface_points | — | **65536** | 96000 (in-flight #206) | in-flight |
| tau_axis_weights | — | **1.0** | 1.5 ❌ | **CLOSED** |
| mlp_ratio | — | **4** | 6 ❌ | **CLOSED** |
| model_layers | — | **4** | 3 (in-flight #233) | in-flight (lower direction) |
| model_heads | — | **8** | 4 (in-flight #232) | in-flight (lower direction) |
| model_slices | — | **128** | 64 (in-flight #231), 256 ❌ OOM | in-flight (lower direction) |
| lr_cosine_t_max | — | **50 (≈flat)** | 9 ❌ | negative at T_max=9; gentle range (14-20) untested |
| lr_warmup_epochs | — | **0** | 1 (in-flight #222) | in-flight |
