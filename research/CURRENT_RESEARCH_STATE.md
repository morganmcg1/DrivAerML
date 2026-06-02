# SENPAI Research State

**Updated**: 2026-06-02 05:35Z | Branch: `tay` | **SOTA: H342 3-cp output-avg ep13+ep14+ep15 × K=5 TTA — val_cal 5.8962 / test_cal 5.7357 (PR #1526 MERGED)**

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

## Closed axes (do NOT revisit — 14 total)

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
| **MoE soft-mixture decoder (Finding N)** | `regime-moe-soft-decoder-redundant-residuals-null` — Two-expert soft-mixture MoE with switch-transformer load-balance aux. Router learned perfectly balanced 50/50 partition (entropy 0.686, load_bal 1.02). But soft mixture averaged experts back to single effective head — redundant residuals. WSS_z floor unmoved (9.20 ≡ H357 9.18). **14th decoder/output-side null axis.** | **H363 CLOSED 05:30Z** |

---

## Active Fleet (2026-06-02 05:35Z — 8/8 WIP students, fleet fully active)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#1548** | alphonse | **H356: 3-cp × K=5 output-avg — FAILED GATES** — Cal arm `0n1xkwic` finished: val_cal **5.9206** (>gate 5.8962, MISS 24bp) / test_cal **5.7668** (>gate 5.7357, MISS 31bp). Zero cal yield (already near-cal-stable). Awaiting terminal SENPAI-RESULT post then close. | ⏳ awaiting student terminal |
| **#1550** | frieren | **H358: tangent-basis residual head — LIKELY FAILED CAL** — Phase 2 TTA `6fcumd8m` still running rt 2.78h, current val_cal 5.9191 (>gate 5.8962, MISS 23bp). Same zero-cal-yield pattern as alphonse and thorfinn. Test_cal pending. | 🟡 WIP — likely close |
| **#1558** | thorfinn | **H365: FastSWA cyclic-LR cross-basin 3-cp output-avg** — 3 cyclic cosine restarts from EP13 (T_max=1 per epoch), output-average 3 cycle-end predictions with H336 TTA recipe. Cross-basin diversity vs H342's within-basin ep13/14/15. | 🟡 WIP — Phase 1 training |
| **#1538** | nezuko | H347: BL physics priors — Arm A val_cal 5.9253 FAILED gate. Arm B/C cascade auto-chaining. ETA terminal ~18-19Z June 2. | 🟡 WIP — long cascade running |
| **#1551** | askeladd | H359: Multi-scale surface kNN branch — Phase 1 done (val_raw ~6.013, 10bp behind SOTA). Single-cp TTA `kwe8tynw` triage running. If not approaching ~5.91 val_cal → close as null. | 🟡 WIP — TTA triage |
| **#1552** | fern | H360: LapPE-32 Laplacian eigenfunction PE — Phase 1 step ~4540+ (55%+ done). Train_loss 0.006-0.010 healthy. ETA Phase 1 complete ~05:30Z. | 🟡 WIP — Phase 1 training |
| **#1560** | tanjiro | **H366: Hierarchical kNN proximity attention bias** (JUST ASSIGNED 05:35Z) — Zero-param encoder-side: pre-softmax bias α_l·1[j∈kNN(i)] on slice-routing logits, per-layer learnable α init=0. Step-0 invariant. Orthogonal to H359 (feature aggregation) and H360 (global LapPE) — attention-pattern modification, not feature richness. Student's own B1 suggestion post-H363. | 🆕 just assigned |
| **#1557** | edward | **H364: Target-magnitude top-decile WSS hotspot reweight** — γ=3 Arm A / γ=5 Arm B conditional. Per-point weight based on ||target_wss|| top-decile. Orthogonal to all prior scalar reweights. Discriminator: is gradient mass concentration the bottleneck? | 🟡 WIP — Phase 1 running |

---

## Current Research Focus: ENCODER/REPRESENTATION TIER (DECODER FULLY SATURATED)

**14 closed axes** all point to the same conclusion: WSS_z floor is representational/upstream. H361 confirmed it is NOT loss-geometry; H363 confirmed decoder capacity reallocation is null. **The live tier is exclusively encoder/representation modifications.**

**Cal-arm failure pattern (2026-06-02 05:30Z)**:
- alphonse H356 (3-cp K=5): val_cal 5.9206 / test_cal 5.7668 — **FAILED both gates** by 24bp / 31bp
- frieren H358 (tangent basis): val_cal 5.9191 (running) — likely FAIL by 23bp
- thorfinn H352 (SWA): val_cal 5.8985 — **FAILED** by 0.23bp
- nezuko H347-A (BL priors): val_cal 5.9253 — FAILED
- **Pattern**: Cal yields drop to 0-2.4bp when the pre-cal output is already near cal-stable (3-cp averaging, SWA, antithetic K=5). H342 7-8bp cal yield is the exception, not the rule.
- **Implication**: Future candidates need to pass on val_raw alone OR demonstrate non-trivial cal headroom. The "project ~7bp cal yield" heuristic is dead for averaged/stacked outputs.

**Triangulation map (post-H363)**:
- INPUT (global spectral): H360 LapPE-32 (fern) — Phase 1 training
- INPUT (multi-scale local feature): H359 kNN branch (askeladd) — TTA triage
- ENCODER (attention-pattern modification): **H366 kNN proximity attention bias (tanjiro)** — just assigned — zero-param, orthogonal to H359 and H360
- ENCODER (content): closed (H357 content embedding null, H351 NGSB null)
- OUTPUT BASIS: H358 tangent basis (frieren) — TTA+cal likely failing
- DECODER CAPACITY: CLOSED (H363 MoE null, 14th axis)
- PHYSICS CONSTRAINT: H347 BL priors (nezuko) — long cascade running
- LOSS (aligned objective per-point): H364 hotspot reweight (edward) — Phase 1 running
- TTA (cross-basin): H365 FastSWA cyclic-LR (thorfinn) — Phase 1 training

### Morgan directive queue (Issue #1056)
- P1 (BL derivative decoder): H355 closed as null — FALSIFIED
- P2 (tangent-basis output head): H358 frieren — likely failing cal
- P3 (multi-scale kNN feature): H359 askeladd — TTA triage
- P4 (Physics-regime MoE decoder): H363 tanjiro — **CLOSED null (Finding N)**

### Next-tier hypotheses for idle students (per Plateau Protocol)

1. **Mesh-geodesic position encoding** — replace Euclidean Fourier features with surface-geodesic distance (computed via mesh adjacency). Tests whether the encoder's representation gap is using the wrong notion of "distance".
2. **Sparse + global attention pattern** — replace some slice-attention layers with sparse local (kNN) + global pool attention. Fundamentally changes how surface tokens interact.
3. **Optimal Transport (Sinkhorn) surface loss** — Earth-Mover-Distance regularizer on WSS predictions instead of pure MSE; captures spatial coherence.
4. **Curvature-MoE encoder routing** — gate encoder layer specialization by local geometric features (mean curvature, principal directions). Decoder MoE was null but encoder MoE may find different attention patterns per regime.
5. **Recurrent decoder refinement (test-time)** — apply decoder N times with cross-attention between iterations. Zero new params, decoder-side but iterative, not capacity expansion.

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
| `regime-moe-soft-decoder-redundant-residuals-null` | Decoder capacity reallocation null; 14th decoder/output axis closed (H363) |
| `cal-yield-collapses-on-averaged-outputs` | 3-cp/SWA/K=5 outputs already near-cal-stable; cal yield drops 7-8bp → 0-2.4bp |
