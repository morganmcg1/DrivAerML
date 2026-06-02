# SENPAI Research State

**Updated**: 2026-06-02 04:15Z | Branch: `tay` | **SOTA: H342 3-cp output-avg ep13+ep14+ep15 × K=5 TTA — val_cal 5.8962 / test_cal 5.7357 (PR #1526 MERGED)**

---

## Current SOTA (H342 calibrated — PR #1526 MERGED 2026-06-01 19:15Z)

| Model | val_cal | test_cal | test_WSS_z | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| **H342 3-cp output-avg ep13+ep14+ep15 × K=4 ← CURRENT SOTA** | **5.8962%** | **5.7357%** | **8.6122%** | **3.3751%** | **3.6124%** | 3icmxaqe/qgw0ix77/ijadzof0 |
| H336 K=5+Student-t ν=4+8-res+cal (prior SOTA) | 5.8978% | 5.7379% | 8.6175% | 3.3735% | 3.6133% | 348i3z1v |
| Transolver-3 target | — | — | **< 5.850%** | ≤ 3.421% | ≤ 3.577% | — |

**Merge gate**: val_abupt_calibrated < **5.8962%** AND test_abupt_calibrated < **5.7357%** (AND-logic)

**WSS_z gap**: test_WSS_z 8.6122% → Transolver target 5.85% = **0.762pp remaining** (primary obstacle)

---

## Closed axes (do NOT revisit — 12 total)

| Axis | Finding | Closed by |
|---|---|---|
| TTA resolution R | `res-density-saturated-8res` | H267, H291, H331 |
| TTA K (Gaussian) | `K5-cal-redundant-at-h312-budget` | H330 |
| TTA noise-family ν | ν=4 Student-t optimal | H314 merged |
| TTA noise σ at ν=4 | `sigma-axis-closed-nu4` | H340 |
| Per-channel cal (affine) | OLS-MLE is global optimum | H316/H319/H323/H328/H329/H332/H333/H334 |
| Cal yield ceiling | `cal-cannot-rescue-train-raw-regression` — ~7-8bp | H343 |
| Weight-space soup | α-soup axis exhausted | H307 |
| WSS loss reweight (all scalar types) | `wz-reweight-monotone-nogate` × 3 axes | H339+H341+H346 |
| K-axis at ν=4 | `K6-vs-K5-noise-floor-tie` | H344 |
| Gradient conflict | `pcgrad-no-conflict-falsifies` — cos=+0.30 | H345 |
| SAM cosine-tail | `sam-flatness-pessimal-wssz` | H343 |
| FiLM decoder (both endpoints) | `filmdec-axis-fully-closed` + `wssz-gap-upstream-not-decoder` | H350+H354 |
| NGSB encoder-routing | `ngsb-normal-only-routing-pessimal` | H351 |
| BL derivative decoder | `bl-derivative-decoder-aux-neutral` | H355 |
| Surface curvature input features | `curvature-features-null` | H348 |
| SP-loss-reweight stacked | `sp-reweight-armC-x-h336-split-decision` | H338 |
| Target transform axis | `target-transform-axis-closed-wssz` (arcsinh + signed_power) | H349+H353 |
| GeoTransolver content embedding | `geotransolver-content-embedding-null` | H357 |
| **SWA within cosine-tail (Finding M)** | `swa-equivalent-to-ema-cosine-tail` — within-trajectory uniform SWA ≡ EMA endpoint, same cal coefs, same basin. Late-trajectory weight averaging cannot find meaningfully different point. **FastSWA cyclic-LR now needed (H365).** | **H352 CLOSED 04:35Z** |
| **Direction-magnitude decomposed WSS loss (Finding L)** | `wss-direction-magnitude-decomposed-loss-null` — cosine gradient rewards direction-matching at all magnitudes; rel-norm gradient over-weights small-magnitude points. Per-channel MSE IS the correct gradient object. **Discriminator outcome confirmed: WSS_z floor is NOT loss-geometry; it is representation.** | **H361 CLOSED 04:09Z** |

---

## Active Fleet (2026-06-02 04:15Z — 8/8 WIP students, fleet fully active)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#1548** | alphonse | **H356: 3-cp × K=5 output-avg — SOTA CANDIDATE** — val_raw 5.9206 (best pre-cal after frieren; projects val_cal ~5.84-5.85, LIKELY PASSES gate). Cal arm `0n1xkwic` running silently, ETA terminal ~04:25Z. test arm cascading. | 🔥 SOTA candidate, cal arm landing |
| **#1550** | frieren | **H358: tangent-basis residual head — BEST PRE-CAL val_raw 5.9191** — Phase 1 done EP16. Phase 2 TTA `6fcumd8m` running silently ~2.78h. Projects val_cal ~5.84, LIKELY PASSES. | 🔥 SOTA candidate, TTA+cal landing |
| **#1558** | thorfinn | **H365: FastSWA cyclic-LR cross-basin 3-cp output-avg** (JUST ASSIGNED 04:35Z) — 3 cyclic cosine restarts from EP13 (T_max=1 per epoch), output-average 3 cycle-end predictions with H336 TTA recipe. Cross-basin diversity vs H342's within-basin ep13/14/15. Student's own suggested follow-up from H352 close. | 🆕 just assigned |
| **#1538** | nezuko | H347: BL physics priors — Arm A val_cal 5.9253 FAILED gate. Arm B/C cascade auto-chaining. ETA terminal ~18-19Z June 2. | 🟡 WIP — long cascade running |
| **#1551** | askeladd | H359: Multi-scale surface kNN branch — Phase 1 done (val_raw ~6.013, 10bp behind SOTA). Single-cp TTA `kwe8tynw` triage running. If not approaching ~5.91 val_cal → close as null. | 🟡 WIP — TTA triage |
| **#1552** | fern | H360: LapPE-32 Laplacian eigenfunction PE — Phase 1 step ~4540 (55% done). Train_loss 0.006-0.010 healthy. ETA Phase 1 complete ~05:30Z. | 🟡 WIP — Phase 1 training |
| **#1556** | tanjiro | H363: Physics-regime MoE surface decoder (Morgan P4) — Arm A v2 N=2 RUNNING step 8459-8523 rt 32min, train_loss 0.014-0.017 HEALTHY. | 🟢 WIP — Phase 1 training |
| **#1557** | edward | **H364: Target-magnitude top-decile WSS hotspot reweight** (JUST ASSIGNED 04:10Z) — γ=3 Arm A / γ=5 Arm B conditional. Per-point weight based on ||target_wss|| top-decile. Orthogonal to all prior scalar reweights. Discriminator: is gradient mass concentration the bottleneck? | 🆕 just assigned |

---

## Current Research Focus: ENCODER/INPUT-SIDE + DECODER-CAPACITY + ALIGNED-LOSS ATTACKS

**12 closed axes** all point to same conclusion: WSS_z floor is representational/upstream. H361 was the definitive discriminator for loss-geometry; result confirmed representation.

**4 concurrent SOTA candidates** landing 04:00-05:30Z (frieren best at val_raw 5.9191):
- If ANY pass both gates (val_cal < 5.8962 AND test_cal < 5.7357) → MERGE-WINNER
- All 4 project val_cal ~5.84-5.85 from pre-cal val_raw after ~7-8bp cal yield

**Triangulation logic**:
- INPUT (global spectral): H360 LapPE-32 (fern) — ongoing
- INPUT (multi-scale local): H359 kNN branch (askeladd) — triage
- ENCODER: closed (H357 content embedding null, H351 NGSB null)
- OUTPUT BASIS: H358 tangent basis (frieren) — SOTA candidate
- DECODER CAPACITY: H363 MoE N=2 (tanjiro) — ongoing
- PHYSICS CONSTRAINT: H347 BL priors (nezuko) — long cascade
- LOSS (aligned objective): H364 hotspot reweight (edward) — just assigned
- TTA STACKING: H352 SWA (thorfinn), H356 3-cp K=5 (alphonse) — SOTA candidates

### Morgan directive queue (Issue #1056)
- P1 (BL derivative decoder): H355 closed as null — FALSIFIED
- P2 (tangent-basis output head): H358 frieren — in flight as SOTA candidate
- P3 (multi-scale kNN): H359 askeladd — in triage
- P4 (Physics-regime MoE): H363 tanjiro — Arm A running

**Next when students idle**: GeoTransolver-style geometric cross-attention (architectural escalation per Plateau Protocol).

---

## Findings bank highlights (WSS_z-relevant)

| Finding | Implication |
|---|---|
| `wz-reweight-monotone-nogate` | Scalar loss reweight CLOSED (3 axes: uniform, wz-only, per-vertex focal) |
| `K5-studentt-superadditive` | K=5 is optimal; K-axis saturated |
| `sigma-axis-closed-nu4` | TTA-noise family fully saturated |
| `cal-cannot-rescue-train-raw-regression` | Cal extracts ~7-8bp val; cannot close >10bp deficit |
| `sam-flatness-pessimal-wssz` | H336 basin already flat; SAM costs 20-50bp WSS_z |
| `decoder-pareto-optimal-at-h336-ep13` | Any decoder modification traces strictly worse |
| `wss-direction-magnitude-decomposed-loss-null` | Loss geometry NOT the bottleneck; representation IS (H361 discriminator) |
