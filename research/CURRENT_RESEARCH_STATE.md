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

## In-flight (8/8 students WIP — zero idle GPUs)

Last updated: 2026-05-01 survey. Steps measured at ~5000 steps/epoch (DDP8).

| PR | Student | Hypothesis | Step | ~Ep | Val (latest) | vs SOTA val | Status |
|---|---|---|---|---|---:|---:|---|
| **#147** | frieren | compound + wd=2e-3 | 0 | ep0 | — | — | **POD STUCK** since 2026-04-30 23:14 UTC; needs `kubectl rollout restart deployment senpai-drivaerml-ddp8-frieren` (escalated 5× on issue #48) |
| **#161** | askeladd | lion_beta2=0.999 | 21,700 | ~ep4.3 | 12.779 | +3.295 | running — POOR trajectory |
| **#162** | edward | model_dropout=0.05 | 21,462 | ~ep4.3 | 10.674 | +1.190 | running — **LEADING** |
| **#163** | thorfinn | weight_decay=1e-3 | 20,753 | ~ep4.2 | 10.767 | +1.283 | running — close 2nd |
| **#173** | tanjiro | cosine T_max=50 | 12,675 | ~ep2.5 | 13.858 | +4.374 | running — POOR (T_max too long for budget) |
| **#186** | alphonse | vol_pts=96k CLEAN | 1,903 | ~ep0.4 | 67.016 | — | running — too early (ep0 noise; SOTA ep0 val ~53-75 range) |
| **#187** | nezuko | volume_loss_weight=1.5 | 2,227 | ~ep0.4 | N/A | — | running — too early |
| **#189** | fern | lion_beta1=0.8 | 1,781 | ~ep0.4 | N/A | — | running — too early |

## Closed this cycle (2026-05-01)

| PR | Student | Test | vs SOTA | Why |
|---|---|---:|---:|---|
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

## Optimizer hyperparam map (tay confirmed)

| Param | Min tested | SOTA | Max tested | In-flight |
|---|---|---|---|---|
| lr | 5e-5 | **1e-4** | 1.5e-4 ❌ | ceiling found |
| ema_decay | 0.9999 | **0.999** | 0.998 ❌ | sweet spot |
| lion_beta1 | — | **0.9** | 0.8 (#189, fern) | in-flight (also lower than SOTA — testing directional range) |
| lion_beta2 | — | **0.99** | 0.999 (#161) | in-flight |
| model_dropout | — | **0.0** | 0.05 (#162) | in-flight |
| weight_decay | — | **5e-4** | 1e-3 (#163) | in-flight |
| volume_loss_weight | — | **1.0** | 1.5 (#187), 2.0 ❌ | in-flight |
| volume_points | — | **65536** | 96000 (#186) | in-flight |
| tau_axis_weights | — | **1.0** | 1.5 ❌ | **CLOSED** — Lion sign mechanism neutralizes |
| mlp_ratio | — | **4** | 6 ❌ | **CLOSED** — capacity ceiling at 9-ep budget |

## Next research directions (priority order)

1. **Wait for PR #173 (tanjiro replication)** — when done, close as SOTA noise-variance baseline and assign `--lr-cosine-t-max 9` follow-up (genuine LR-decay delta, not the null delta the harness ran).
2. **Wait for PR #186 (alphonse vol_pts=96k clean)** — first interpretable test of volume sampling density. Watch `val_primary/volume_pressure_rel_l2_pct` — direct read on the binding gap.
3. **Wait for PR #187 (nezuko vol_loss_weight=1.5)** — gentler than #142's 2.0; tests whether the loss-balance lever has a sweet spot below catastrophic surface degradation.
4. **Wait for fern #159, askeladd #161, edward #162, thorfinn #163** — round-11 single-delta sweep on optimizer/regularization hyperparams. Most are in early epochs.
5. **Frieren pod restart needed** — pod stuck at init since 23:14 UTC Apr 30. PR #147 (wd=2e-3) has NEVER started. Requires human `kubectl rollout restart deployment senpai-drivaerml-ddp8-frieren`. Re-escalate on issue #48 if not actioned by next survey.
6. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Biggest untested architectural lever. Not yet assigned. Reserve for next idle slot after current sweep results.
7. **Tau_yz binding gap (code-change approach)** — must bypass Lion neutralization. Options: (a) asinh output normalization for tau_y/tau_z, (b) surface-tangent-frame prediction head, (c) decoupled magnitude+direction head. All require train.py modifications by a student.
8. **`--lr-cosine-t-max 9` (genuine LR decay)** — would be the first test of an actually-decaying cosine over the 9-epoch budget. Reserve as tanjiro's follow-up after PR #173 closes.
