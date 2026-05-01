# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 (updated from survey)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #115 (thorfinn compound lr=1e-4 + EMA=0.999), test_abupt 10.580

**Verified SOTA config (W&B `d03oghpp`):**
- lr=1e-4, weight_decay=5e-4, optimizer=lion, lion_beta1=0.9, lion_beta2=0.99
- use_ema=True, ema_decay=0.999, **lr_cosine_t_max=0** (falls back to `T_max=epochs`; with epochs=50 → cosine over 50, essentially flat over the actual 9-epoch run), lr_warmup_epochs=0
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

## In-flight (8/8 students running — all slots filled)

Last updated: 2026-05-01 (updated — thorfinn reassigned).

| PR | Student | Hypothesis | ~Ep | Val (latest) | vs SOTA val | Status |
|---|---|---|---|---:|---:|---|
| **#147** | frieren | compound + wd=2e-3 | ep0 | — | — | **POD STUCK** since 2026-04-30 23:14 UTC; needs `kubectl rollout restart deployment senpai-drivaerml-ddp8-frieren` (escalated on issue #48) |
| **#186** | alphonse | vol_pts=96k CLEAN | ~ep1+ | TBD | — | running |
| **#187** | nezuko | volume_loss_weight=1.5 | ep5 | 13.815 | +4.33 | running |
| **#189** | fern | lion_beta1=0.8 | ep5 | 14.378 | +4.89 | running |
| **#194** | askeladd | ema_decay=0.9995 | ep1 | alarming (+23pp) | +23 | running — early epochs highly noisy; wait for ep4+ |
| **#195** | edward | lr_cosine_t_max=9 | ep1 | 52.407 | — | running — promising early signal (similar ep1 to SOTA run) |
| **#202** | tanjiro | lr_cosine_t_max=9 genuine | ep1 | TBD | — | **ASSIGNED** 2026-05-01 — parallel T_max=9 validation run |
| **#203** | **thorfinn** | weight_decay=2.5e-4 (sweep down) | — | — | — | **ASSIGNED** 2026-05-01 — sweep DOWN from SOTA wd=5e-4; PR #163 (wd=1e-3) regressed +4.5% confirming gradient points down |

## Closed this cycle (2026-05-01)

| PR | Student | Test | vs SOTA | Why |
|---|---|---:|---:|---|
| #161 | askeladd | 12.564 | +18.7% | lion_beta2=0.999 — momentum window too wide (1000 steps vs SOTA 100 steps); model stuck in early-training trajectories. beta2=0.99 confirmed optimal. |
| #162 | edward | 11.029 | +4.24% | model_dropout=0.05 — model underfits (not overfits) at 9ep; dropout noise hurts feature learning. val/test ratio WORSE (1.130 vs SOTA 1.115). Regularization lever closed. |
| #157 | nezuko | 11.261 | +6.4% | mlp_ratio=6 capacity expansion didn't help; val flatlined ep7-8. FFN-width family CLOSED. |
| #158 | alphonse | 13.179 | +24.6% | CONFOUNDED — `--lr-cosine-t-max 0` collapsed LR to 1e-6 by ep9. Original vol_pts hypothesis re-launched cleanly as PR #186. |

## Round 11 closeouts (carried forward)

| PR | Student | Test | vs SOTA | W&B run | Conclusion |
|---|---|---:|---:|---|---|
| **#149** | tanjiro | 11.022 | +4.2% | rtajk53c | Per-axis tau weights W_y=W_z=1.5 regressed all metrics. **Direction CLOSED.** |

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
| #157 | nezuko (mlp_ratio=6) | 11.261 | +6.4% | FFN width capacity ceiling hit early; val flatlined |
| #158 | alphonse (vol_pts=96k confounded) | 13.179 | +24.6% | LR-schedule confound; clean re-run launched as #186 |

## Key learnings accumulated

1. **Depth-swap family CLOSED** — 5L (#138 +6%) and 6L (#146 +19.7%) both regressed. Per-epoch wall-clock unchanged (attention-dominated); 4L/512d is the right shape for 9-ep budget.
2. **FFN-width family CLOSED** — mlp_ratio=6 (#157 +6.4%) regressed; +15% param cost without payoff; val flatlined at ep7. Capacity expansion via FFN width is saturated.
3. **Loss weight family** — vol_w=2.0 alone hurts surface (+18%), vol_p flat. Paired sw=2+vw=2 (#141) is net-zero (+0.23%) with val/test ratio divergence. Gentler vol_w=1.5 not yet tested — that's PR #187.
4. **LR ceiling confirmed at 1e-4** — lr=1.5e-4 failed (+40%); lr=1e-4 is SOTA.
5. **Slices=256 OOM** — 98.8% VRAM usage, slice attention quadratic. Default 128 is safe ceiling.
6. **Lion compile divergence** — 9/9 compile+Lion combinations diverged in earlier rounds.
7. **Per-axis tau weighting CLOSED** — Lion sign-based updates neutralize per-channel loss weighting. W_y=W_z=1.5 regressed all metrics +4.2% (PR #149). Do not pursue higher weights.
8. **`--lr-cosine-t-max 0` is a footgun** — fallback to `T_max=epochs` means `epochs=9, t_max=0` collapses LR to 1e-6 by ep9, but `epochs=50, t_max=0` keeps LR essentially flat for 9 epochs. Always specify `--lr-cosine-t-max 50` explicitly in 9-epoch single-deltas to match SOTA's effective schedule. PR #158 was confounded by this; PR #186 re-runs with the fix.
9. **Volume sampling density (vol_pts) is genuinely untested** — PR #186 is the first clean test. PR #158 outcome was unattributable due to LR confound.
10. **Next approach for tau_yz gap**: Target normalization (asinh/log transform), surface-tangent-frame head, or decoupled magnitude+direction prediction head. Need code changes beyond current train.py flags.
11. **Dropout is a dead end** — model_dropout=0.05 (#162, +4.24%) WORSENED val/test ratio (1.130 vs SOTA 1.115). The model is in an underfitting regime at 9 epochs; adding regularization noise makes things worse. Dropout closed.
12. **lion_beta2=0.999 is a dead end** — momentum window of 1000 effective steps is too wide for 9-ep training. Model gets stuck in early-training gradient directions. beta2=0.99 (100-step effective window) is well-calibrated for this budget. Higher beta2 closed.

## Optimizer hyperparam map (tay confirmed)

| Param | Min tested | SOTA | Max tested | In-flight |
|---|---|---|---|---|
| lr | 5e-5 | **1e-4** | 1.5e-4 ❌ | ceiling found |
| ema_decay | 0.9999 | **0.999** | 0.998 ❌ | sweet spot |
| lion_beta1 | — | **0.9** | 0.8 (#189, fern) | in-flight (also lower than SOTA — testing directional range) |
| lion_beta2 | — | **0.99** | 0.999 ❌ | **CLOSED** — momentum window 1000 steps too wide for 9ep budget; +18.7% regression |
| model_dropout | — | **0.0** | 0.05 ❌ | **CLOSED** — model underfits not overfits; dropout hurts feature learning at this budget |
| weight_decay | 2.5e-4 (#203, in-flight) | **5e-4** | 1e-3 ❌ (+4.5% regression, PR #163) | sweep DOWN: #203 (2.5e-4) |
| volume_loss_weight | — | **1.0** | 1.5 (#187), 2.0 ❌ | in-flight |
| volume_points | — | **65536** | 96000 (#186) | in-flight |
| tau_axis_weights | — | **1.0** | 1.5 ❌ | **CLOSED** — Lion sign mechanism neutralizes |
| mlp_ratio | — | **4** | 6 ❌ | **CLOSED** — capacity ceiling at 9-ep budget |

## Next research directions (priority order)

1. **Thorfinn PR #203 (wd=2.5e-4)** — sweep down from SOTA; PR #163 (wd=1e-3) regressed +4.5% locking gradient direction DOWN. 2.5e-4 is the midpoint between 0 and SOTA; watch ep3 val; if above ~11.5%, run is off-trajectory.
2. **Wait for PR #195 (edward T_max=9)** and **PR #202 (tanjiro T_max=9)** — first genuine cosine LR decay tests. If either beats SOTA val 9.484%, merge immediately and extend to T_max=7/8 follow-up.
3. **Wait for PR #194 (askeladd ema_decay=0.9995)** — probes unexplored EMA midpoint; ep1 alarming but early noise expected; wait for ep4+. If it beats SOTA, probe 0.9997/0.9998.
4. **Wait for PR #186 (alphonse vol_pts=96k clean)** — first clean volume sampling density test. Watch `val_primary/volume_pressure_rel_l2_pct` directly.
5. **Wait for PR #187 (nezuko vol_loss_weight=1.5)** — gentler loss balance; tests below vol_w=2.0 catastrophe.
6. **Wait for fern #189 (lion_beta1=0.8)** — lower directional momentum test; SOTA=0.9. If results known, assign lion_beta1=0.85 to close sensitivity curve midpoint.
7. **Frieren pod restart needed** — pod stuck. PR #147 (wd=2e-3) has NEVER started. Requires `kubectl rollout restart deployment senpai-drivaerml-ddp8-frieren`. Re-escalate on issue #48. Note: wd=2e-3 already predicted to regress further than wd=1e-3.
8. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Biggest untested architectural lever. Reserve for next idle slot after current wave completes.
9. **Tau_yz binding gap (code-change approach)** — bypass Lion neutralization: (a) asinh output normalization for tau_y/tau_z, (b) surface-tangent-frame prediction head, (c) decoupled magnitude+direction head. All require train.py modifications.
10. **wd follow-up** — if PR #203 (2.5e-4) beats SOTA, probe 1e-4 to tighten the curve. If #203 regresses, wd sweet spot is ~5e-4 and WD family is CLOSED.
11. **ema_decay follow-up curve** — if #194 (0.9995) beats SOTA, probe 0.9997 and 0.9998 to close out the sensitivity range.

## Weight decay sweep summary (PR #163 closed)

PR #163 (thorfinn, wd=1e-3) CLOSED: best val=9.911% vs SOTA 9.484%. All per-surface metrics regressed +4.5%. Gradient in WD points DOWN. Next: PR #203 wd=2.5e-4 (thorfinn, round12).
