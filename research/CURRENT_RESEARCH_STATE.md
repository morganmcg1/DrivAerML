# SENPAI Research State

**Updated**: 2026-06-02 08:35Z | Branch: `tay` | **SOTA: H342 3-cp output-avg ep13+ep14+ep15 × K=5 TTA — val_cal 5.8962 / test_cal 5.7357 (PR #1526 MERGED)**

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

## Closed axes (do NOT revisit — 17 total)

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
| **Tangent-basis output head (Finding O)** | `tangent-basis-output-head-boundary-null` — Native tangent-basis residual output (τ·n=0 by construction) + LayerNorm-gated head_corr. Phase 1 raw 5bp behind H336. Post-TTA+cal val_cal 5.8963 / test_cal 5.7366 — within 0.1bp of H342 gate (noise floor). WSS_z marginally regressed +0.03bp. Output-basis enforcement is physics-correct but did NOT move the WSS_z floor. **15th closed axis. All 3 Morgan P-tier WSS architecture directives have closed null — decisively falsifying the decoder/output hypothesis family.** | **H358 CLOSED 06:00Z** |
| **Target-magnitude WSS hotspot reweight (Finding P)** | `target-magnitude-wss-hotspot-reweight-overshoot` — Per-point loss weight = γ on top-decile (||target_WSS||>4.844 Pa) WSS points, baseline weight elsewhere. γ=3 Arm A: EP14 val_abupt 6.586% (+69bp vs gate, +57bp vs H336 EP13 source). All channels regressed including unreweighted VP +105bp (Lion sensitivity to 1.20× WSS-mass shift breaks H342-tuned balance). Hotspot mask diagnostics confirmed 10% mask captures 6.1× mean magnitude points — mechanism implemented correctly, **idea failed**. Arm B γ=5 skipped per close rule. **Cumulative scalar-reweighting axis now CLOSED — 6 nulls (H338, H339, H341, H346, H361, H364). H336 fine-tune basin is brittle to gradient-mass rebalancing; bottleneck is REPRESENTATION CAPACITY, not loss geometry.** | **H364 CLOSED 06:30Z** |
| **Anisotropic tangent-frame attention encoder (Finding Q)** | `anisotropic-tangent-frame-attention-encoder-null` — Per-vertex Q/K rotation into local frame (t1,t2,n) via slice-effective surface normals; pre-softmax score mix `(1-σ(γ_l)) S_std + σ(γ_l) S_aniso`, per-layer learnable γ_l init=-10. EP14 val_primary 6.0757% (>6.05% close threshold by 2.6bp); EP15 6.0765%, EP16 6.0783% — flat-to-worse trajectory. **Diagnostic: all 5 γ_aniso scalars stayed pinned at ~-10 (σ≈4.5e-5) — gates NEVER opened**. Optimizer found no gradient signal toward engaging anisotropy. Per-channel WSS_z MAE +0.7% worse than H336 EP13 (the channel the hypothesis targeted). **17th closed axis. 4 consecutive encoder/representation hypotheses using surface normals (H351 routing, H357 content, H358 output basis, H367 attention frame) ALL null — model is saturated on existing geometric content; next attacks must inject information the encoder doesn't currently have, not rearrange existing.** | **H367 CLOSED 08:30Z** |

---

## Active Fleet (2026-06-02 06:40Z — 8/8 WIP students, fleet fully active)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#1548** | alphonse | **H356: 3-cp × K=5 output-avg — FAILED GATES** — Cal arm `0n1xkwic` finished: val_cal **5.9206** (>gate 5.8962, MISS 24bp) / test_cal **5.7668** (>gate 5.7357, MISS 31bp). Awaiting terminal SENPAI-RESULT post then close. | ⏳ awaiting student terminal |
| **#1558** | thorfinn | **H365: FastSWA cyclic-LR cross-basin 3-cp output-avg** — 3 cyclic cosine restarts from EP13 (T_max=1 per epoch), output-average 3 cycle-end predictions with H336 TTA recipe. Cross-basin diversity vs H342's within-basin ep13/14/15. | 🟡 WIP — Phase 1 |
| **#1538** | nezuko | H347: BL physics priors — Arm A val_cal 5.9253 FAILED gate. Arm B/C cascade auto-chaining. ETA terminal ~18-19Z June 2. | 🟡 WIP — long cascade running |
| **#1551** | askeladd | H359: Multi-scale surface kNN branch — Phase 1 done (val_raw ~6.013, 10bp behind SOTA). Single-cp TTA `kwe8tynw` triage running. If not approaching ~5.91 val_cal → close as null. | 🟡 WIP — TTA triage |
| **#1552** | fern | H360: LapPE-32 Laplacian eigenfunction PE — Phase 1 done, eval `8cqqpd9x` (h360-eval-lappe32) running. | 🟡 WIP — eval phase |
| **#1560** | tanjiro | **H366: Hierarchical kNN proximity attention bias** — Zero-param encoder-side: pre-softmax bias α_l·1[j∈kNN(i)] on slice-routing logits, per-layer learnable α init=0. Step-0 invariant. Student's own B1 suggestion post-H363. | 🟡 WIP — smoke + Phase 1 |
| **#1562** | edward | **H368: WSS spatial-gradient consistency loss (kNN-8 edge-pair matching)** — Zero-param structural loss term: matches predicted WSS edge-gradients (over kNN-8 neighbors) to target edge-gradients. Acts on point-PAIR differences, not per-point magnitudes. Addresses Edward H364 #3 suggestion (high-frequency WSS_z error structure). Different from H347 nezuko (variance penalty), H361 (per-point decomposition), H345 (gradient conflict). Phase 1 from H336 EP13. | 🆕 just assigned 06:35Z |
| **#1561** | frieren | **H367: Anisotropic surface attention via local tangent frame** | ❌ **CLOSED 08:30Z** Finding Q. Reassigning H369 RWPE next. |

---

## Current Research Focus: ENCODER INPUT-FEATURE TIER (NORMAL/GEOMETRY EXISTING-CONTENT SATURATED)

**17 closed axes** all point to the same conclusion: WSS_z floor is representational/upstream. **Critical update post-H367**: 4 consecutive encoder/representation hypotheses using **surface normals** (H351 routing, H357 content, H358 output basis, H367 attention frame) ALL closed null — the model is **saturated on existing geometric content**. Next attacks must inject information the encoder DOES NOT CURRENTLY HAVE, not rearrange existing representations.

**Live attack tier**: new information channels — global spectral structure (H360 LapPE), local mesh topology (H369 RWPE, just assigned), structural edge-pair losses (H368 WSS gradient consistency), multi-scale local features (H359 kNN aggregation), BL physics priors (H347), cross-basin TTA (H365 FastSWA).

**Cal-arm failure pattern (2026-06-02 05:30Z)**:
- alphonse H356 (3-cp K=5): val_cal 5.9206 / test_cal 5.7668 — **FAILED both gates** by 24bp / 31bp
- frieren H358 (tangent basis): val_cal 5.9191 (running) — likely FAIL by 23bp
- thorfinn H352 (SWA): val_cal 5.8985 — **FAILED** by 0.23bp
- nezuko H347-A (BL priors): val_cal 5.9253 — FAILED
- **Pattern**: Cal yields drop to 0-2.4bp when the pre-cal output is already near cal-stable (3-cp averaging, SWA, antithetic K=5). H342 7-8bp cal yield is the exception, not the rule.
- **Implication**: Future candidates need to pass on val_raw alone OR demonstrate non-trivial cal headroom. The "project ~7bp cal yield" heuristic is dead for averaged/stacked outputs.

**Triangulation map (post-H367 close)**:
- INPUT (global spectral): **H360 LapPE-32 (fern)** — Phase 1 rank0 val_raw 6.012% (≡ H336 baseline 6.017%, neutral-positive); TTA eval running (8cqqpd9x)
- INPUT (multi-scale local feature): H359 kNN branch (askeladd) — TTA triage
- INPUT (local mesh topology): **H369 RWPE-16 (frieren) — JUST ASSIGNED 08:40Z** — random walk return probabilities, k=1..16 steps, zero-param feature, K×n_hidden=3072 projection weights only
- ENCODER (attention-pattern: spatial proximity): H366 kNN proximity attention bias (tanjiro) — zero-param, learnable per-layer scalar
- ENCODER (attention-pattern: anisotropic frame): CLOSED (H367 tangent-frame Q/K rotation null, γ_l gates never opened — 17th axis)
- ENCODER (content/routing): CLOSED (H357 content null, H351 NGSB null)
- OUTPUT BASIS: CLOSED (H358 tangent-basis boundary-null, 15th axis)
- DECODER CAPACITY: CLOSED (H363 MoE null, 14th axis)
- PHYSICS CONSTRAINT: H347 BL priors (nezuko) — long cascade running
- LOSS (per-point scalar reweighting axis): CLOSED — 6 nulls cumulative (H338, H339, H341, H346, H361, H364)
- LOSS (structural edge-pair gradient matching): H368 WSS spatial-gradient consistency (edward) — Phase 1 pickup pending
- TTA (cross-basin): H365 FastSWA cyclic-LR (thorfinn) — Phase 1 training

**Triangulation logic (3 input-feature axes in flight)**: H359 (multi-scale local kNN aggregation of values) + H360 (global Laplacian eigenfunction spectral) + H369 (local mesh topology random-walk structural) span the three orthogonal axes of "information the encoder currently lacks". If all 3 null, the bottleneck is conclusively capacity not information — pivot to non-capacity-additive structural rewrites (sparse+global attention, inducing-point bottleneck, recurrent state). If any wins, the winning axis identifies the missing information channel.

### Morgan directive queue (Issue #1056) — ALL P-TIER FALSIFIED
- P1 (BL derivative decoder): H355 — CLOSED null (Finding K)
- P2 (tangent-basis output head): H358 — **CLOSED boundary-null** (Finding O)
- P3 (multi-scale kNN feature): H359 askeladd — TTA triage in progress
- P4 (Physics-regime MoE decoder): H363 — CLOSED null (Finding N)

**Decisive falsification**: all 3 architectural directives (P1, P2, P4) closed null. The decoder/output stage is NOT the WSS_z bottleneck. Need to post update to Morgan on Issue #1056 — three P-tier hypotheses ruled out, current attack tier is encoder representation.

### Next-tier hypotheses for idle students (per Plateau Protocol)

1. ~~**RWPE — random-walk PE**~~ — **ASSIGNED to frieren as H369 (PR pending)**
2. **Sparse + global attention pattern** — replace some slice-attention layers with sparse local (kNN) + global pool attention. Fundamentally changes how surface tokens interact. Not capacity-additive if existing slice-attention layers are replaced.
3. **Optimal Transport (Sinkhorn) surface loss** — Earth-Mover-Distance regularizer on WSS predictions. Different from H368 kNN-edge structural loss — captures distribution-level coherence, not point-pair gradients.
4. **Inducing-point attention bottleneck** — replace some slice-attention layers with set-transformer style inducing-point attention. Forces information to compress through M inducing points (M << N), breaks slice-token bottleneck.
5. **Recurrent decoder refinement (test-time)** — apply decoder N times with cross-attention between iterations. Zero new params, decoder-side but iterative, not capacity expansion.
6. **Mesh-geodesic landmark distances** — precompute geodesic distance from each surface point to K farthest-point landmarks (heat method on mesh). Different from RWPE (which uses kNN-graph topology) — uses true mesh-edge geodesic via PDE-solve.
7. **Heat-kernel signature (HKS) PE** — uses eigenvalue + eigenvector heat-diffusion combinations as multi-scale geometric descriptor. Related to H360 LapPE but encodes diffusion dynamics, not raw spectral modes.

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
