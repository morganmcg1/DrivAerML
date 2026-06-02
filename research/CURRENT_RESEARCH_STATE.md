# SENPAI Research State

**Updated**: 2026-06-02 13:15Z | Branch: `tay` | **SOTA: H342 3-cp output-avg ep13+ep14+ep15 × K=5 TTA — val_cal 5.8962 / test_cal 5.7357 (PR #1526 MERGED)**

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

## Closed axes (do NOT revisit — 23 total)

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
| **WSS spatial-gradient consistency loss (Finding R)** | `wss-spatial-gradient-consistency-loss-overshoot` — Per-edge L2 of (WSS_pred[j] - WSS_pred[i]) − (WSS_target[j] - WSS_target[i]) over kNN-8 surface graph, λ=0.3. EP14 val_primary **6.503%** (+49bp regression vs baseline 6.017%, blows close threshold by 45bp). ALL 5 channels regressed: WSS_z +38bp (the target channel got WORSE), WSS_y sizable, SP +32bp, VP **+100bp** (channel not touched by new loss term, regressed most). train/wss_grad_loss DID converge 0.026 → 0.015 — mechanism worked but destroyed point accuracy. **Cumulative loss-tier null pattern is now 7 axes (H338, H339, H341, H346, H361, H364, H368). H336/H342 loss formulation is at a tight Pareto-optimum on the EP13 fine-tune basin — any per-point reweighting OR per-edge structural addition breaks multi-channel balance via Lion-basin gradient-mass shift, regardless of weight or operator.** Decisive: edward is 4-for-4 nulls on loss-tier in a row (H338→H361→H364→H368) — pivoting to non-loss-tier next. | **H368 CLOSED 09:05Z** |
| **ISAB middle-layer REPLACE 3-of-5 layers (Finding S)** | `isab-middle-layer-replace-null` — Set Transformer ISAB (M=32 per-head inducing points `[H=4, M=32, dim_head=128]`) on layers idx 1,2,3 (60% of slice-mixing replaced). Warm-start H336 EP13, 3-epoch cosine tail, Lion. EP14 val_primary **7.0188%** (+1.00pp vs source, +1.12pp vs H342 gate — worst single-epoch regression of 5 recent edward closes). EVERY channel regressed +0.7–1.2pp — signature of "basin lost all calibrated slice-token interaction structure." Root cause: mixed-init pathology — cold ISAB layers 1/2/3 sandwiched between pretrained Transolver 0/4; 3-epoch cosine tail cannot recover from cold-init on 60% of slice-mixing under this LR budget. **19th closed axis.** Follow-up H371: single-layer probe at idx 2 only to bound operator family. | **H370 CLOSED 11:15Z** |
| **LapPE-32 surface Laplacian spectral input PE (Finding T)** | `lappe-spectral-input-null` — Top-32 eigenfunctions of normalized graph-Laplacian over kNN-8 surface mesh, injected as input PE via zero-pad surgery into project_surface_features (512,100→512,132). Wiring verified clean (lap_cols_norm=0.0e+00 post warm-load, no missing keys). Phase 1 raw NEUTRAL (-0.5bp vs source); TTA pre-cal val_RAW 5.9258 (+3bp before cal). Cal arm: **val_cal 5.9029 / test_cal 5.7427** — FAILS gate +6.7bp / +7.0bp. **Cal yield only 2.29bp** (4th cal-yield-collapse instance). val_WSS_z RAW TTA 9.0901% ≈ H336 raw — LapPE channels did not propagate signal to target channel through 3-epoch cosine tail. **20th closed axis.** | **H360 CLOSED 12:08Z** |
| **RWPE-K16 Random Walk Positional Encoding (Finding U)** | `rwpe-surface-topology-pe-null` — Random-walk return probabilities up to K=16 hops on kNN-16 surface graph, 32-channel projection with zero-init. EP14 val_primary **6.0742%** (+5.7bp vs source, +2.4bp past 6.05% close rule). Per-channel: SP +18bp, VP +35bp, WSS_z +14bp (target channel WORSE), WSS ≈flat. Crashed mid-EP15 at step=8943 before grace-period evidence; original close rule applies. Joint with H348/H359/H360: **4 of 4 input-feature hypotheses null — input-feature axis CONCLUSIVELY EXHAUSTED on EP13 fine-tune basin**. **21st closed axis.** | **H369 CLOSED 12:08Z** |
| **ISAB OPERATOR FAMILY CLOSED at single-layer bound (Finding V)** | `isab-single-layer-probe-null` — Single-layer ISAB at idx 2 (1-of-5 layers, M=32 inducing points). EP14 val_primary **6.7149%** (+70bp vs source, +66bp past 6.05% close rule). All 5 channels regressed broadly: SP +64bp, **VP +108bp (untouched-channel collapse)**, WSS +66bp, WSS_z +82bp. Same "basin lost calibrated slice-token interaction" signature as H370. **ISAB operator family CLOSED at the bound** — even one cold ISAB layer breaks token-count bijection enough to disrupt the pretrained surrounding layers under 3-epoch Lion cosine tail. **Implication: future operator replacements MUST preserve token-count bijection** — inducing-point/Perceiver-style bottleneck operators are empirically excluded from warm-start tier. **22nd closed axis.** | **H371 CLOSED 12:17Z** |
| **kNN spatial-proximity attention bias (Finding W)** | `knn-spatial-proximity-attn-bias-null` — Zero-param additive pre-softmax attention bias from surface kNN proximity (k=32). EP14 val_primary **6.0752%** (>6.05% close rule, +1.6bp). Pre-committed close rule fired immediately; Phase 2 TTA skipped. 8 DDP ranks finished cleanly (rt~3.83h). **Geometric attention-bias signal fails.** Discriminator status: sets up H376 (physics-informed WSS-gradient magnitude bias) to distinguish geometry-only vs. physics-informed attention steering. If H376 also nulls, attention-steering tier is fully closed. **23rd closed axis.** | **H366 CLOSED 12:55Z** |
| **Self-consistency τ_z-only vs EP13 EMA teacher (Finding X)** | `self-consistency-ep13-ema-teacher-null` — Frozen EP13 EMA teacher MSE consistency loss on τ_z channel only (λ=0.1). EP14 val_primary **6.0700%** (>6.00% kill gate). **ROOT CAUSE: teacher itself has ~9% WSS_z error** — Mean Teacher / NoRD prior requires teacher to have *lower* error than student, which is violated. Consistency penalty also propagated through shared backbone to other channels: WSS_y **+388bp** (worse than random initialization shift). Training-time τ_z regularization via bad teacher = noisy-target overfitting. **24th closed axis.** | **H374 PR #1569 CLOSED 13:00Z** |

---

## Active Fleet (2026-06-02 13:15Z — 8/8 WIP, **all students assigned**)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#1548** | alphonse | **H356: 3-cp × K=5 output-avg** — Arm A ep15 K=5 reproduced H336 exactly (val_cal 5.8978 / test_cal 5.7379). Arm B ep14 K=5 val_RAW 5.9293, test running. Arm C ep13 chained. Terminal ETA ~20Z; 3-cp average likely fails cal-yield-collapse pattern. | 🟡 WIP — chain ep14/ep13 |
| **#1558** | thorfinn | **H365: FastSWA cyclic-LR cross-basin 3-cp output-avg** — val_RAW 5.9273 single-cp, TTA cal+test running. SOTA candidate if cal yield ≥6bp. | 🟡 WIP — TTA cal+test |
| **#1538** | nezuko | **H347: BL physics priors** — Arm A FAILED gate (val_cal 5.9253). Arm B `65z21dv8` val_RAW 5.9224% (best of cluster), TTA running. ETA terminal ~15:30Z. | 🟡 WIP — Arm B/C cascade |
| **#1551** | askeladd | **H359: Multi-scale surface kNN** — val_RAW 5.9245% (kwe8tynw, -45bp vs H336 RAW). Test arm running (rt=22699s). Cal landing ETA ~17-18Z. **PRIMARY SOTA CANDIDATE** if cal yield ≥7bp. | 🟡 WIP — TTA test arm |
| **#1572** | tanjiro | **H376: WSS gradient-magnitude attention bias (EP13 EMA teacher, zero-param)** — Per-key additive attention bias proportional to ||∇τ_pred||₂ from frozen EP13 EMA teacher's WSS prediction over kNN-8 surface graph. Pre-compute once per sample; inject as static scalar into all slice-attention layers. Physics-informed successor to H366 (geometric kNN bias nulled as Finding W). Discriminator: if H376 also nulls, attention-steering tier CLOSED. Kill gate EP14 val>6.05%. ~14h+TTA. | 🟢 ASSIGNED 13:00Z |
| **#1570** | fern | **H375: Cross-Channel Decoder Query Tokens** — K=4 learnable query tokens, cross-attention from τ_x/τ_y context → τ_z decoder. +0.1% params (264K @ d_model=512). Channels verified: loader.py:42 [...,1]=τ_x, [...,2]=τ_y, [...,3]=τ_z. Replaces wrong-premise H372 (z-mirror, closed no-run). Kill gate EP14 val>6.05%. ~14h+TTA. | 🟢 ASSIGNED 12:45Z |
| **#1568** | frieren | **H373: Transolver++ Local Adaptive Slice Pooling** — Replace global slice softmax with k=16 kNN-constrained local pooling (temp=0.1). Zero params, preserves token-count bijection (post-H371 V constraint). Targets boundary-layer τ_z. Kill gate EP14 val>6.05%. ~14h+TTA. | 🟢 ASSIGNED 12:33Z |
| **#1574** | edward | **H377: Z-antisymmetry mirrored augmentation (H-A)** — τ_z has strict algebraic antisymmetry τ_z(x,y,−z)=−τ_z(x,y,z) under lateral mirror. Train with z-mirrored batch copies (negate z-coord, nz, τ_z targets; batch_size=2 so mirror doubles back to 4 eff). Average τ_z at inference: 0.5×(pred_orig + (−pred_mirror)). Highest-priority unattacked symmetry axis. Zero params, zero architecture change. Kill gate EP14 val>6.00%. ~10.5h+TTA. | 🟢 ASSIGNED 13:15Z |

---

## Current Research Focus: 24 CLOSED AXES — SYMMETRY AUGMENTATION NOW PRIMARY ATTACK (H377)

**24 closed axes**. H374 closed (Finding X): self-consistency vs EP13 EMA teacher null — teacher has ~9% WSS_z error itself, violating the Mean Teacher premise. H366 (Finding W): geometric kNN proximity attention bias null. H376 now tests physics-informed attention bias; H377 (edward) tests z-antisymmetry mirrored augmentation — HIGHEST-priority remaining unattacked axis. H375/H373 in live round testing cross-channel decoder conditioning and local slice pooling. Narrowed frontier: (1) physical symmetry augmentation / equivariance constraints, (2) physics-informed attention bias, (3) decoder cross-channel conditioning, (4) operator replacements preserving token-count bijection. Training-time regularizer, input-feature, loss, decoder-capacity, and ISAB-operator tiers all exhausted.

**SOTA CLUSTER UPDATE (post-H360 cal landing, 12:10Z):**
| Arm | val_RAW | val_CAL | test_CAL | Status |
|---|---:|---:|---:|---|
| H347 Arm B nezuko `65z21dv8` | **5.9224%** | TBD | TBD | test arm running, ETA ~15:30Z |
| H359 askeladd `kwe8tynw` | 5.9245% | TBD | TBD | test arm running, ETA ~15Z |
| ~~H360 fern `8cqqpd9x`~~ | 5.9258% | **5.9029** ❌ | **5.7427** ❌ | **CLOSED — cal yield only 2.29bp, fails gate +6.7bp/+7.0bp** |
| H365 thorfinn `d6zb0a18` | 5.9273% | TBD | TBD | val+test TTA running, ETA ~14Z |
H360 demonstrated cal-yield-collapse (2.29bp vs projected 7bp), so the 7bp projection for H347/H359/H365 may also collapse. Each is a single-cp K=4 arm (same predictor structure as H336) so 7-8bp cal is plausible, but TTA-averaged outputs across the cluster carry latent cal-yield-collapse risk. Watch carefully.

**Live attack tier (post-H360/H369/H371 close + 3 new assignments)**:
- **Input-feature axis**: CLOSED (H348/H360/H369 null; H359 cal pending likely null)
- **ISAB operator family**: CLOSED at single-layer bound (H370/H371; Finding V)
- **Cross-basin TTA**: H365 FastSWA cyclic-LR (thorfinn, val_RAW 5.9273 cal+test pending)
- **Encoder attention bias — physics-informed** (kNN gradient magnitude from EMA teacher): H376 (tanjiro, PR #1572, ASSIGNED 13:00Z)
- **Physics-prior cascade**: H347 BL priors (nezuko, val_RAW 5.9224 Arm B test arm)
- **Cross-channel decoder conditioning (NEW)**: H375 fern PR #1570 — K=4 cross-attention query tokens τ_x/τ_y → τ_z; +0.1% params; replaces wrong-premise H372 (z-axis is NOT bilateral axis; y is)
- **Local-adaptive slice pooling (NEW)**: H373 frieren PR #1568 — Transolver++ k=16 kNN-constrained slice softmax; zero params; preserves token-count bijection; targets boundary-layer τ_z
- **Self-consistency τ_z teacher (NEW)**: H374 edward PR #1569 — τ_z-only MSE consistency vs frozen EP13 EMA teacher (λ=0.1); zero params; training-time regularizer
- **Reserve tiers (still unattacked)**:
  - Output-space target transforms (log-rotated WSS basis, GLU output gating, learnable per-channel temperature)
  - Auxiliary-loss heads (predict skin-friction coefficient / vorticity / near-wall gradient as aux task)
  - Hard-example mining at case level (NOT per-vertex scalar reweight — already closed)
  - Test-time training (TTT — per-case backprop adaptation at test time)
  - Cross-channel decoder query tokens (H-D reserve, +0.1% params)
  - ~~WSS gradient magnitude attention bias (H-F)~~ — ASSIGNED as H376 to tanjiro (discriminator vs H366 geometric null)

**Loss-tier is FULLY CLOSED — 7 nulls cumulative**. Any future loss modification would be redundant. Decoder/output tier also FULLY CLOSED — 14 nulls. The live frontier is exclusively:
1. injecting new information into encoder input
2. replacing existing encoder architecture (not augmenting)
3. cross-basin TTA / weight-space

**Cal-arm failure pattern (2026-06-02 05:30Z)**:
- alphonse H356 (3-cp K=5): val_cal 5.9206 / test_cal 5.7668 — **FAILED both gates** by 24bp / 31bp
- frieren H358 (tangent basis): val_cal 5.9191 (running) — likely FAIL by 23bp
- thorfinn H352 (SWA): val_cal 5.8985 — **FAILED** by 0.23bp
- nezuko H347-A (BL priors): val_cal 5.9253 — FAILED
- **Pattern**: Cal yields drop to 0-2.4bp when the pre-cal output is already near cal-stable (3-cp averaging, SWA, antithetic K=5). H342 7-8bp cal yield is the exception, not the rule.
- **Implication**: Future candidates need to pass on val_raw alone OR demonstrate non-trivial cal headroom. The "project ~7bp cal yield" heuristic is dead for averaged/stacked outputs.

**Triangulation map (post-H360/H369 close — 21 axes closed)**:
- INPUT (global spectral): **CLOSED — H360 Finding T**
- INPUT (multi-scale local feature): H359 kNN branch (askeladd) — TTA test arm running, cal pending (likely null)
- INPUT (local mesh topology): **CLOSED — H369 Finding U** (crashed mid-EP15 after triggering close rule at EP14)
- INPUT (surface curvature): CLOSED — H348 curvature-features-null
- INPUT TIER: **3 of 4 closed null; 4th likely null at cal landing → tier essentially EXHAUSTED**
- OPERATOR (3-layer ISAB REPLACE): **CLOSED — H370 Finding S** (catastrophic +97bp mixed-init pathology)
- OPERATOR (1-layer ISAB probe): H371 edward — current architectural bound; EP14 val landing ~12:35Z
- ENCODER (attention-pattern: spatial proximity): **CLOSED — H366 Finding W** (geometric kNN bias null, EP14 6.0752%)
- ENCODER (attention-pattern: physics-informed gradient magnitude): **H376 tanjiro PR #1572** — discriminator; if null → attention-steering tier CLOSED
- ENCODER (attention-pattern: anisotropic frame): CLOSED (H367, 17th axis)
- ENCODER (content/routing): CLOSED (H357 + H351)
- OUTPUT BASIS: CLOSED (H358 tangent-basis boundary-null, 15th axis)
- DECODER CAPACITY: CLOSED (H363 MoE null, 14th axis)
- PHYSICS CONSTRAINT: H347 BL priors (nezuko) — Arm B test arm running
- LOSS axis: FULLY CLOSED — 7 nulls (H338, H339, H341, H346, H361, H364, H368)
- TTA (cross-basin): H365 FastSWA cyclic-LR (thorfinn) — TTA cal+test running

**Post-input-feature-exhaustion logic**: With 4 input-feature axes all null, the H336/H342 model has saturated information capacity from raw geometric inputs. Two paths remain: (A) operator-family rewrites under warm-start (H371 testing single-layer floor), or (B) different attack vector entirely — output-space transforms, auxiliary task heads, test-time training. The researcher-agent has been dispatched to generate hypotheses on path B for fern + frieren.

### Morgan directive queue (Issue #1056) — ALL P-TIER FALSIFIED
- P1 (BL derivative decoder): H355 — CLOSED null (Finding K)
- P2 (tangent-basis output head): H358 — **CLOSED boundary-null** (Finding O)
- P3 (multi-scale kNN feature): H359 askeladd — TTA triage in progress
- P4 (Physics-regime MoE decoder): H363 — CLOSED null (Finding N)

**Decisive falsification**: all 3 architectural directives (P1, P2, P4) closed null. The decoder/output stage is NOT the WSS_z bottleneck. Need to post update to Morgan on Issue #1056 — three P-tier hypotheses ruled out, current attack tier is encoder representation.

### Next-tier hypotheses for idle students (per Plateau Protocol, post-H360/H369 close)

**Input-feature tier essentially exhausted (3 of 4 closed; H359 cal pending)** — researcher-agent dispatched 12:11Z to generate fresh attack vectors in:

1. **Output-space target transforms** — log-rotated WSS basis, learnable per-channel temperature, GLU-gated outputs. WSS_z is currently predicted in raw Cartesian; predicting in a rotated/whitened basis might decorrelate it from WSS_x/y, letting the optimizer find a different WSS_z minimum.
2. **Auxiliary-loss heads** — predict skin-friction coefficient (Cf = ||WSS|| / (0.5·ρ·U²)), vorticity, tangential pressure gradient, or near-wall flow gradient as an aux task. Shared-trunk representation may improve WSS_z via implicit regularization.
3. **Self-distillation with EP13-as-teacher** — KL-to-teacher-logits + epsilon GT term. Sometimes gives 2-5bp on hard channels without changing architecture.
4. **Test-time training (TTT)** — per-case backprop adaptation at test time using an auxiliary self-supervised task (mask-prediction, neighbor-prediction, surface-continuity). 1-5 gradient steps per test case.
5. **Hard-example case-level mining** — re-weight CASES (not vertices — that's the closed scalar-reweight tier) by WSS_z difficulty during training. Different optimization mechanism than per-point reweight.
6. **Z-flip equivariance constraint** — force prediction to be exactly equivariant under z-flip via Siamese-pair loss term. WSS_z's z-asymmetry under mirror symmetry should make this a strong inductive constraint.
7. **Flow-token cross-attention decoder** — add a small set of learned "flow-mode" tokens that cross-attend to surface tokens (FlowFormer-style). Decoder-side, not stack-replace.
8. ~~RWPE — random-walk PE~~ — CLOSED H369
9. ~~LapPE-32~~ — CLOSED H360
10. ~~Curvature features~~ — CLOSED H348

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
| `cal-yield-collapses-on-averaged-outputs` | 3-cp/SWA/K=5 outputs already near-cal-stable; cal yield drops 7-8bp → 0-2.4bp (4 confirmations: H336/H342/H348/H360) |
| `lappe-spectral-input-null` | Surface Laplacian eigenfunction PE null on WSS_z under 3-epoch warm-start tail (H360 Finding T) |
| `rwpe-surface-topology-pe-null` | Random walk PE null + per-channel interference (+18bp SP, +35bp VP) on H336 basin (H369 Finding U) |
| **`input-feature-axis-tier-exhausted`** | **3 of 4 input-feature hypotheses null (H348/H360/H369); H359 cal pending. Zero-pad warm-start surgery doesn't propagate new input signal into WSS_z under 3-epoch cosine. Next attacks must target OPERATOR or OUTPUT-SPACE or AUXILIARY-TASK, not input.** |
