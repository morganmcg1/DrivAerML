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

Last updated: 2026-05-01 15:30 (fresh W&B poll; alphonse #186 closed and re-assigned as #206).

| PR | Student | Hypothesis | W&B run | ep config | Eps done | Latest Val | vs SOTA (ep-matched) | Status |
|---|---|---|---|---|---|---:|---:|---|
| **#206** | alphonse | surface_pts 64k→96k | (assigned, no W&B run yet) | 9 | 0 | — | — | newly assigned this cycle (replaces #186 closeout) |
| **#187** | nezuko | volume_loss_weight=1.5 | `osbmpkmq` (group `tay-round11-vol-loss-weight-1p5`) | 9 | 7/9 | 10.368 | +0.208 vs SOTA ep7=10.16 | running — 2 epochs remain; trajectory still approaching SOTA pace |
| **#189** | fern | lion_beta1=0.8 | `iukbkkgm` (group `round11-lion-beta1-0p8`) | 9 | 7/9 | 10.884 | +0.724 vs SOTA ep7=10.16 | running — 2 epochs remain; lagging SOTA pace by 0.7pp |
| **#194** | askeladd | ema_decay=0.9995 | `qvoj4lbs` (group `tay-round12-ema-decay-0p9995`) | 50 | 5 | 12.623 | +0.793 vs SOTA ep5=11.83 | running — lagging by 0.79pp; watching ep6+ |
| **#195** | edward | lr_cosine_t_max=9 | `0gftvkrc` (group `tay-round12-lr-cosine-tmax-9`) | 50 | **5** | **11.390** | **-0.440 vs SOTA ep5=11.83** | **AHEAD OF SOTA PACE 2 EPOCHS RUNNING** — strongest live signal; uniformly ahead on every sub-metric (surf=7.41<7.77, vol=6.95<7.12, wall=12.78<13.27) |
| **#196** | frieren | vol_loss_weight=2.0 | `qymdn7px` (group `tay-round12-vol-loss-weight-2p0`) | 9 | 2 | 25.452 | — (ep2 only) | running — too early to assess; wait for ep4+ |
| **#202** | tanjiro | lr_cosine_t_max=9 (replication) | `1wx7mfw6` (group `tay-round12-cosine-tmax9`) | 50 | 2 | 25.396 | — (ep2 only) | running — variance probe for edward #195; identical config |
| **#203** | thorfinn | weight_decay=2.5e-4 | `894ay3y1` (group `tay-round12-wd-2p5e-4`) | 50 | 3 | 19.327 | +2.813 vs SOTA ep3=16.51 | running — looking weak so far; wait for ep5+ |

**Key signals (2026-05-01 15:30):**
- **Edward #195 is the live SOTA-beat candidate** — ep5 abupt=11.390 vs SOTA 11.830 (-0.44pp), uniformly ahead on every sub-metric. If trend holds to ep9 the cosine_tmax=9 lever is a clear win and tanjiro #202 will give us a free run-to-run variance estimate.
- **Nezuko #187** at ep7=10.368% — 2 epochs remain; closest 9-epoch run to challenging SOTA but trajectory is +0.2pp behind ep-matched SOTA.
- **Fern #189** at ep7=10.884% — lagging; lion_beta1=0.8 looking like a clear loss vs SOTA 0.9.
- **Askeladd #194** at ep5 lags 0.79pp — ema_decay=0.9995 (slower averaging) appears worse than 0.999 at this point.
- **Alphonse #186 CLOSED** — vol_pts=96k clean re-run trailed SOTA on EVERY sub-metric at every epoch (ep9 abupt 10.068>9.484, surf 6.411>6.007, vol 6.358>5.896, wall 11.238>10.632). Vol_pts lever retired.
- (Earlier "vol_p=6.358% vs SOTA 12.740% = 50% improvement" claim was wrong — that compared val ep9 to *test* ep9 numbers, not val-to-val. Apples-to-apples val comparison shows alphonse vol_p IS worse than SOTA val by 0.46pp.)

## Closed this cycle (2026-05-01)

| PR | Student | Test | vs SOTA | Why |
|---|---|---:|---:|---|
| **#186** | **alphonse** | val 10.068 (ep9) | **+0.584 val** | **vol_pts=96k CLEAN re-run.** Trailed SOTA on every sub-metric at every epoch (surf 6.411>6.007, vol 6.358>5.896, wall 11.238>10.632). Vol_pts lever retired — 64k confirmed at/near optimum. |
| #161 | askeladd | 12.564 | +18.7% | lion_beta2=0.999 — momentum window too wide (1000 steps vs SOTA 100 steps); model stuck in early-training trajectories. beta2=0.99 confirmed optimal. |
| #162 | edward | 11.029 | +4.24% | model_dropout=0.05 — model underfits (not overfits) at 9ep; dropout noise hurts feature learning. val/test ratio WORSE (1.130 vs SOTA 1.115). Regularization lever closed. |
| #157 | nezuko | 11.261 | +6.4% | mlp_ratio=6 capacity expansion didn't help; val flatlined ep7-8. FFN-width family CLOSED. |
| #158 | alphonse | 13.179 | +24.6% | CONFOUNDED — `--lr-cosine-t-max 0` collapsed LR to 1e-6 by ep9. Original vol_pts hypothesis re-launched cleanly as PR #186 (now also closed, see above). |

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
| volume_loss_weight | — | **1.0** | 1.5 (#187 nezuko in-flight), 2.0 (#196 frieren in-flight) | in-flight (both) |
| volume_points | — | **65536** | 96000 ❌ (#186 closed: +0.46pp val on every metric) | **CLOSED** — 64k is at/near optimum |
| surface_points | — | **65536** | 96000 (#206, alphonse round12) | newly in-flight |
| tau_axis_weights | — | **1.0** | 1.5 ❌ | **CLOSED** — Lion sign mechanism neutralizes |
| mlp_ratio | — | **4** | 6 ❌ | **CLOSED** — capacity ceiling at 9-ep budget |

## Next research directions (priority order)

1. **Watch edward #195 (cosine T_max=9)** — ep5=11.390% vs SOTA ep5=11.830 (-0.44pp), uniformly ahead on every sub-metric. Highest-priority live signal for a SOTA beat; if trend holds to ep9 the lever is a clear win.
2. **Watch tanjiro #202 (cosine T_max=9 replication)** — variance probe in parallel with edward. If both beat SOTA, lever is robust; if one beats and one loses, run-to-run noise dominates and we need more replicates.
3. **Watch nezuko #187 (vol_loss_weight=1.5)** — ep7=10.368%, 2 epochs remaining; closest 9-ep run to SOTA. If ep9 finishes below 9.484%, merge immediately. Currently +0.2pp behind ep-matched SOTA — borderline.
4. **Watch fern #189 (lion_beta1=0.8)** — ep7=10.884%, lagging by 0.7pp. lion_beta1=0.8 appears suboptimal vs SOTA 0.9. If ep9 doesn't beat SOTA, close and confirm lion_beta1=0.9 as optimum (both sides tested: 0.8 here vs 0.9 SOTA, no upside seen).
5. **Watch askeladd #194 (ema_decay=0.9995)** — ep5=12.623% vs SOTA ep5=11.83 (+0.79pp lag). Watching ep6+. If gap doesn't close by ep7, ema_decay=0.999 confirmed optimal direction (slower averaging hurts).
6. **Watch alphonse #206 (surface_pts=96k, just assigned)** — clean parallel to retired vol_pts=96k probe. If wins, surface_pts is the lever; if loses, surface_pts at 64k is also confirmed optimal.
7. **Watch frieren #196 (vol_loss_weight=2.0)**, **thorfinn #203 (wd=2.5e-4)** — both at ep2/ep3, too early to assess; wait for ep4+.
8. **cosine T_max follow-up (T_max=7/8)** — if edward AND tanjiro both beat SOTA at T_max=9, probe shorter decay schedules to find the optimal annealing point.
9. **ema_decay follow-up curve** — if askeladd (0.9995) beats SOTA, probe 0.9997 and 0.9998. If loses, EMA family fully calibrated at 0.999.
10. **wd follow-up** — if thorfinn (2.5e-4) beats SOTA, probe 1e-4 next. If regresses, wd=5e-4 confirmed and WD family CLOSED both directions.
11. **Compound stack for next round** — once round 12 closes, stack the winners (likely cosine_tmax=9 + any other new winners) on top of SOTA #115 base.
12. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Biggest untested architectural lever. Reserve for idle slot after round 12 wave completes.
13. **Tau_yz binding gap (code-change approach)** — bypass Lion neutralization: (a) asinh output normalization for tau_y/tau_z, (b) surface-tangent-frame prediction head, (c) decoupled magnitude+direction head. All require train.py modifications.

## Weight decay sweep summary (PR #163 closed)

PR #163 (thorfinn, wd=1e-3) CLOSED: best val=9.911% vs SOTA 9.484%. All per-surface metrics regressed +4.5%. Gradient in WD points DOWN. Next: PR #203 wd=2.5e-4 (thorfinn, round12).
