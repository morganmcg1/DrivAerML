# SENPAI Research State

**Updated**: 2026-05-31 04:40Z | Branch: `tay` | **SOTA: H312 H296+8-res+cal — MERGED ~02:30Z** | Round 4k+2

## 2026-05-31 SOTA progression (4 merges now):
1. H295 (K=5+6-res, 15:45Z) → test 5.7679
2. H296 (K=4+8-res, 17:05Z) → test 5.7678 (−0.15bp)
3. H300 (per-channel calibration, 18:48Z) → test_cal 5.7399 (−28bp)
4. **H312 (H296+8-res base+cal, 02:30Z) → test_cal 5.7388** (−1.1bp) ← **CURRENT SOTA**

**Finding 'better-raw-better-cal'**: H296 base (val_raw 5.9221) gives lower calibrated test than H285 base (val_raw 5.9275). H300's 28bp initial gain was from the calibration axis itself; H312 adds 1.1bp more by combining the better TTA base with calibration. Calibration on single-model is near saturation — next axes: cross-channel WSS mixing (H323), regional spatial calibration (H313 in-flight), quadratic terms (H317 in-flight).

---

## Current SOTA (H312 calibrated)

| Model | val_cal | test_cal | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| **H312 H296+8-res+cal ← CURRENT SOTA** | **5.8994%** | **5.7388%** | **6.6391%** | **3.3743%** | **3.6137%** | enf61qrr |
| H300 H285+cal | 5.9011% | 5.7399% | 6.6404% | 3.3763% | 3.6132% | 59r4noqh |
| H296 EP15+anti-K4+8-res+mirror (raw) | 5.9221% | 5.7678% | 6.6728% | 3.3763% | 3.6436% | at1jadnv |
| Transolver-3 target | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% | — |

**Merge gate (UPDATED)**: val_abupt_calibrated < **5.8994%** AND test_abupt_calibrated < **5.7388%**
**Paper floors**: test_VP 3.3743 < 3.421 ✓ | test_WSS 6.6391 < 6.727 ✓ | test_SP 3.6137 > 3.577 ✗ (3.7bp gap)
**WSS gap**: 6.6391 → 5.85 target = **0.789pp remaining** (effectively unchanged)

---

## Active Fleet (as of 02:45Z 2026-05-31 — 8 students active, all assigned)

| PR | Student | Hypothesis | val_raw | Status | ETA calibrated |
|---|---|---|---:|---|---|
| **#1508** | **edward** | **H323: Cross-channel WSS affine calibration (3×3 mixing matrix for τ_x/τ_y/τ_z)** | — | 🆕 assigned 02:45Z | ~09:30Z |
| **#1507** | **thorfinn** | **H307: Weight-space averaging of 2 EP15 seeds + H312 calibration** | — | 🟢 assigned 00:50Z, started ~01:00Z | ~08:00Z |
| **#1504** | **tanjiro** | **H317: Quadratic per-channel calibration (15 params vs H312's 16)** | — | 🟢 nld6viep running | ~03:30Z |
| **#1512** | **alphonse** | **H329: Abupt-weighted calibration (align fit objective with eval metric)** | — | 🆕 assigned 04:40Z (post-H316 close) | ~11:30Z |
| **#1501** | **nezuko** | **H315: TTA aggregation operator sweep — median & trimmed-mean** | 5.9264/5.9226% | 🟢 2wq13m13 — INTERIM: median worse, trim≈avg | ~02:30Z |
| **#1500** | **frieren** | **H314: Student-t weight-space noise (df=4)** | **5.9213%** ⭐ | 🟢 2scozlaf — INTERIM val −0.08bp vs H296 (positive) | ~03:00Z |
| **#1511** | **askeladd** | **H328: Ridge-regularized regional calibration (λ-sweep)** | — | 🆕 assigned 04:20Z (post-H313 close, rescues spatial signal via shrinkage to H300 global) | ~12:00Z |
| **#1509** | **fern** | **H319: Per-resolution H312 calibration (fit α/β per TTA resolution before mean aggregation)** | — | 🆕 assigned 03:00Z (after H306 close) | ~10:00Z |

**Merged this loop (02:30Z)**:
- **PR #1498 edward H312** — MERGED. **NEW SOTA** val_cal 5.8994 / test_cal 5.7388. Finding 'better-raw-better-cal' confirmed. Calibration near saturation for diagonal affine on single model.

**Closed this loop (03:00Z)**:
- **PR #1497 fern H306** — CLOSED. Finding 'conf-weighted-T1-equals-uniform-avg'. T=1.0 softmax over 64 TTA passes with similar per-pass loss collapses to uniform → null. Per-point inverse-variance TTA aggregation closed at T=1. → H319 per-resolution calibration assigned to fern.
- **PR #1499 askeladd H313** — CLOSED (04:15Z). Finding 'regional-cal-overfits-val'. 4z/6z regional affine cal passes val gate (5.8947/5.8938) but fails test by +2.0/+2.2bp. Per-zone α range 0.985–1.005 — real signal (rear zone α>1 vs everywhere else α<1) but too small to survive 34-car val→test gap. → H328 ridge-regularized λ-sweep assigned to askeladd (PR #1511) — rescues spatial signal with shrinkage to H300 global.
- **PR #1502 alphonse H316** — CLOSED (04:35Z). **Finding 'cal-scale-dominates-bias-null-on-test'**: ablation decomposed H300/H312's gain — α (scale) does 81%, β (bias) is 0% (+0.1bp). The huge β_VP ≈ -0.89 is the OLS intercept absorbing α≠1, not an independent bias correction. Model has WSS scale deficit (predictions over-large by 0.5-0.8% on τ channels). → H329 abupt-weighted OLS objective assigned to alphonse (PR #1512) — better α estimation via aligning fit objective with eval metric.

**Hot watch (03:00Z)**:
- **Frieren H314 2scozlaf** (~03:00Z) → val_raw 5.9213% leader. **If calibrated val < 5.8994, becomes SOTA candidate #1** against new tighter gate. This is the primary watch.
- **Tanjiro H317 nld6viep** (~03:30Z) → quadratic cal; must beat val_cal 5.8994/test_cal 5.7388 (harder now)
- **Alphonse H316 yg4sbrtw** (~03:00Z) → bias vs scale ablation; informs whether to pursue H312→H324 higher-K cal variants
- **Nezuko H315 2wq13m13** → **INTERIM: median aggregation +4.3bp worse, trim10 +0.5bp worse** — aggregation axis looks null
- **Askeladd H313 iya68eq8** (~02:30Z) → regional calibration; test pass nearly done
- **Fern H319 #1509** (~10:00Z) → per-resolution cal; tests resolution-dependent bias structure
- **Thorfinn H307** (~08:00Z) → model soup; uses new H312 SOTA as the seed-1 calibration baseline
- **Edward H323** (~09:30Z) → cross-channel WSS affine; tests whether τ_x/τ_y/τ_z have rotational coupling
- **Edward Arm B a8uyi3ev** (~08:30Z) → K=5+6-res+cal diagnostic; watch gate val_cal < 5.8994 / test_cal < 5.7388

**Queue for next idle students** (the calibration axis dominates):
- **H318**: Calibration on H299-style noise + H312 cal — does noise-family change the residual structure?
- **H320**: 3rd seed commission (if H307 model-soup wins) OR cubic calibration (if H317 quadratic gains ≥3bp)
- **H321**: Jackknife/LOO calibration variance on 34-car val (robustness check of 10-param fit on 34 cars)
- **H322**: Best-of-K aggregation under H312 calibration (per-case selection vs averaging)
- **H324**: K=5+6-res+cal confirmation (Arm B result will tell us if needed separately)
- **H325**: WLS calibration (minimize abupt-aligned objective per-car, vs OLS absolute-squared error)
- **H326**: Compose H323 (cross-channel WSS) + H319 (per-resolution) — both win → 8×{5×5+3} mixing matrix per resolution

---

## Recently Merged / Closed

| PR | Student | Finding | val | test |
|---|---|---|---|---|
| **#1484 fern H296** | EP15+anti-K4+8-res+mirror | **MERGED (17:05Z) — NEW SOTA** — Finding TTT: K=4+8-res Pareto-dominates K=5+6-res (res-axis compound > K-axis escalation at this operating point). 4/5 channels improve (VP dominant). New gate: val<5.9221, test<5.7678. | 5.9221 | 5.7678 |
| #1485 thorfinn H297 | EP15+anti-K4+6-res, σ_attn=0 | **CLOSED (17:05Z)** — Finding "per-layer-noise-attn0-mlp-null": K=4 TTA at EP15 is noise-routing-insensitive; attn vs mlp layer split has no effect on aggregate performance. Channels within 0.001 of H295 baseline. | 5.9237 | 5.7684 |
| **#1483 tanjiro H295** | EP15+anti-K5+6-res+mirror | **MERGED (15:45Z)** — Finding SSS: K-axis saturation curve at EP15 (K=4→5 slope 1/17 of K=3→4, ~0.04bp gain, 5/5 sign-consistent). | 5.9231 | 5.7679 |
| #1477 frieren H291 | EP15+anti-K3+8-res-LOWER {16K,24K,32K-131K}+mirror | **CLOSED (13:45Z)** — **Finding RRR**: 8-res lower fails both gates by ~1bp (val 5.9252 +1.7bp, test 5.7693 +1.0bp). Lower-res samples add Taylor variance without info gain. Combined with PPP: resolution axis fully closed asymmetrically below 32K and above 131K; 6-res {32K-131K} is global Pareto optimum. → H302 (channel-asymmetric resolution) assigned to frieren. | 5.9252 | 5.7693 |
| #1476 thorfinn H290 | EP15+anti-K3+multi-σ{3e-4,5e-4,7e-4}+6-res | **CLOSED (11:03Z)** — Finding LLL-multi-σ-diversity-null: val 5.9241/test 5.7689 fails H285 gate (+0.6bp both axes). σ-diversity subsumed by K-axis: K=3→K=4 at fixed σ gives 4–7× the gain vs σ-mixing at K=3. → H297 (per-layer noise stratification) assigned to thorfinn. | 5.9241 | 5.7689 |
| #1473 fern H288 | EP15+anti-K3+8-res densified | **CLOSED (09:56Z)** — Finding KKK-8res-K3-val-passes-test-marginal-fail: val 5.9229 (passes gate), test 5.7685 (fails +0.02bp). VP improves −2.1bp vs H275; SP degrades +1.1bp. K-axis and res-axis are orthogonal. → H296 (K=4+8-res compound) assigned to fern. | 5.9229 | 5.7685 |
| #1470 tanjiro H285 | EP15+anti-K4+6-res+mirror | **MERGED as NEW SOTA** — Finding JJJ-K4-K-axis-alive-at-EP15 (K=3→K=4 = −0.8bp val / −0.7bp test, sign-consistent 5/5 channels, slope diminishing) | 5.9235 | 5.7683 |
| #1472 askeladd H287 | EP14 checkpoint anti-K3 | **Finding III-EP-axis-fully-exhausted** (EP14 val 5.9326/test 5.7764 fails gate; EP13≈EP14<EP15(peak)>EP16; all accessible checkpoints below EP15 confirmed worse across all 3 channels) | 5.9326 | 5.7764 |
| #1460 nezuko H280 | EP13+Sobol-anti-K5+6-res | **Finding HHH-EP13-no-rescue-with-K5-Sobol** (EP13+K5+Sobol val 5.9313/test 5.7753 fails gate; all channels match-or-degrade vs SOTA; doubled-K Sobol diversity cannot compensate for EP13 vs EP15; EP-axis closed above AND below) | 5.9313 | 5.7753 |
| #1475 frieren H289 | EP16 checkpoint axis (closed w/o running) | **Finding FFF-EP-axis-closed-at-H275-recipe** (checkpoint EP12/EP16 unavailable; H244 3-res evals confirm EP15 peak, EP16 slight regression; EP-axis fully exhausted with available checkpoints) | n/a | n/a |
| #1466 thorfinn H283 | EP15+σ=3e-4+Sobol-K5+6-res+mirror | **Finding XX-Sobol-confirmed** (close, val 5.9274 +3.1bp / test 5.7725 +3.5bp fails H275 gate; σ-axis closed across all noise families at EP15) | 5.9274 | 5.7725 |
| #1459 frieren H279 | EP15+Sobol-K5+6-res+mirror | **Findings DDD-Sobol-EP15-super-additive + EEE-anti-dominates-Sobol-at-EP15** (close, val 5.9291 +4.8bp / test 5.7734 +4.4bp fails H275 gate; anti structurally dominates Sobol at both EPs) | 5.9291 | 5.7734 |
| #1461 fern H281 | EP13+σ=3e-4+anti-K=3 stacked | **Finding AAA confirmed at EP13** (close, σ=3e-4 worse than σ=5e-4 on EP13+anti too — σ-axis CLOSED for anti family on both EPs) | 5.9342 | 5.7785 |
| #1465 alphonse H282 | EP13+anti-K=2 stacked | **Finding BBB-K-curve-floor-EP13** (close, K=2 floor: +9.7bp val vs H275 SOTA) | 5.9340 | 5.7783 |
| #1458 askeladd H278 | EP13+anti-K=4 stacked | **Finding BBB-K-curve-saturates-at-K3-EP13** (close, K=4 only −0.6bp over K=3, both fail H275 gate) | 5.9316 | 5.7758 |
| #1457 tanjiro H277 | EP15+σ=3e-4+anti-K=3 stacked | **Finding AAA-sigma-flat-on-antithetic-EP15** (close, +1.2/+1.5bp worse than H275, fails new gate) | 5.9255 | 5.7705 |
| #1455 edward H275 | EP15+anti-K=3 stacked | **MERGED as SOTA — Finding ZZ-EP15-anti-super-additive** | 5.9243 | 5.7690 |
| #1456 thorfinn H276 | EP15+σ=3e-4+K=5 random stacked | **Finding XX-σ=3e-4-EP15-channel-asymmetric** (close, val -0.08bp, test +0.05bp, WSS +1.9bp) | 5.9314 | 5.7768 |
| #1448 alphonse H269 | EP13+K=10 random stacked | **Finding WW-antithetic-dominates-K-scaling** (close, +0.15bp gate) | 5.9330 | 5.7778 |
| #1454 fern H274 | EP13+anti-K=3 stacked | **MERGED as prior SOTA** | 5.9322 | 5.7763 |
| #1451 frieren H271 | EP13+Sobol K=5 stacked | **MERGED** (prior SOTA, now superseded by H274) | 5.9368 | 5.7797 |
| #1452 nezuko H272 | Hutchinson σ-scaling | **Finding TT-Hutchinson-shrinkage** | 5.9507 | 5.7944 |
| #1453 askeladd H273 | Taylor-2 correction | **Finding SS-Taylor-mixed-sign** | 5.9543 | 5.7973 |
| #1449 tanjiro H270 | EP13+σ=3e-4 stacked | **Finding RR-σ-EP13-stack** | 5.9394 | 5.7827 |
| #1447 edward H267 | EP15+full stack | **Prior SOTA** (now 2 generations back) | 5.9367 | 5.7825 |

---

## Findings Bank (48 banked — UUU and calibration-dominates-TTA added 18:55Z)

| ID | Source | Summary |
|---|---|---|
| **calibration-dominates-TTA-refinement** | H300 edward (merged 18:48Z) | Per-channel affine calibration (10 OLS params fit on 34-car val) gains 28bp test — more than H285→H296 TTA refinements combined. VP bias β≈−0.86 is model-intrinsic, not TTA-dependent. All 5 channels Pareto-improve. Calibration axis is orthogonal to TTA axis. |
| **UUU-embed-noise-null** | H299 askeladd (closed 18:48Z) | Restricting weight noise to embedding/pos-encoding layers only (inverse of H297 layer stratification): val=5.9244/test=5.7689, fails H296 gate by 2.3/1.1bp. Layer-stratification axis fully closed (H297 full-model = H299 embed-only = same aggregation). |
| **coord-noise-harmful** | H298B alphonse (znmtxzdk, val only) | Input-coord perturbation (σ=5e-4, K=4 anti + 6-res + mirror) → val=7.82% (+190bp). Geometry encoder is spatially sensitive — any coordinate noise destroys surface predictions. TTA noise budget must stay in weight-space. H298C σ=1e-3 expected to fail harder (monotone slope). |
| **TTT-K4+8res-Pareto-dominates** | H296 fern (merged 17:05Z) | K=4+8-res (test 5.7678) beats K=5+6-res (5.7679) on both gates. Res-axis compound > K-axis escalation at current operating point. Pareto frontier: (K=4, 8-res). |
| **per-layer-noise-attn0-mlp-null** | H297 thorfinn (closed 17:05Z) | Noise routed only to MLP layers (attn σ=0) at EP15+K=4 anti-thetic: channels within 0.001 of H295 baseline. Noise-routing-insensitive at this operating point. Layer stratification axis closed. |

## Prior Findings Bank (43 banked prior to today)

| ID | Source | Summary |
|---|---|---|
| **SSS-K-axis-saturation-at-K5** | H295 tanjiro (merged 15:45Z as NEW SOTA) | K=3→4 anti-thetic at EP15+6-res was −0.8bp val / −0.7bp test (Finding JJJ). K=4→5 is −0.04bp val / −0.04bp test (1/17th the slope). K_eff=10 at K=5 makes the slope diminish to near-noise-floor. All 5 channels still move in the right direction (signal is real). K=5 is the current optimum; K=6 would cost +20% compute for an estimated ≤−0.01bp gain. New gate: val<5.9231, test<5.7679. Next: H303 probes σ=3e-4 at K=5 (tests if tighter σ reduces 2nd-order residual better at higher K). |
| **RRR-8res-lower-degrades** | H291 frieren (closed 13:45Z) | 8-res lower {16K, 24K, 32K-131K} fails both gates by ~1bp (val +1.7bp, test +1.0bp). Lower-res samples add Taylor variance without info gain. Combined with PPP (8-res upper +11.8bp val): resolution axis fully closed asymmetrically — current 6-res {32K-131K} is global Pareto optimum in {n_res, resolution_range} space. Remaining unexplored direction: channel-asymmetric resolution selection. |
| **QQQ-student-t-null** | H294 askeladd (closed 12:26Z) | Student-t df=3 ≈ Gaussian at matched RMS σ=5e-4 (val 5.9248 +1.3bp). Combined with MMM: noise distribution family axis definitively closed. Leading-order anti-thetic cancellation is distribution-free. |
| **PPP-8res-upper-degrades** | H292 edward (closed 12:26Z) | 8-res upper {32K-192K} degrades val by 11.8bp vs H285 SOTA. Resolution ceiling confirmed at ≤131K. More high-res samples add variance without Taylor-cancellation benefit. |
| **OOO-aggregation-null** | H286 alphonse (closed 12:24Z) | All 5 homogeneous operators (mean, median, trimmed_mean_10/20, Huber) fail H285 gate. Best alternative (trimmed_mean_10) is 0.7bp worse than mean. Linear Taylor-cancellation already provides optimal pooling. → H301 tests heterogeneous per-channel selection. |
| **MMM-laplace-null** | H293 nezuko (closed 12:13Z) | Laplace noise ≈ Gaussian at matched RMS σ=5e-4 (val 5.9246/test 5.7694, both +1.1bp). Distribution shape doesn't matter — Gaussian, Laplace, Student-t all equivalent at this scale. Noise-family axis closed. |
| **LLL-multi-σ-diversity-null-at-EP15-anti-K3** | H290 thorfinn (closed 11:03Z) | At EP15+anti-K3+6-res, mixing σ∈{3e-4,5e-4,7e-4} gives val ~H275 (−0.02bp), all channels FAIL H285 gate (+0.6bp both axes). σ-diversity subsumed by K-axis: K=3→K=4 at fixed σ=5e-4 gives 4–7× more gain than σ-mixing at K=3. SP channel unimproved (3.6428 vs target 3.577). → H297 per-layer noise stratification assigned to thorfinn. |
| **KKK-8res-K3-val-passes-test-marginal-fail** | H288 fern (closed 09:56Z) | At EP15+anti-K3, 6→8 res (mid densification: +40960,+57344) gives val −1.4bp vs H275 but test only −0.05bp. Channel asymmetric: VP −2.1bp (large), SP +1.1bp (regression). K-axis and res-axis are **orthogonal** — distinct channel signatures → K=4+8-res compound (H296) should stack both gains additively. K=4+6-res (H285) beats K=3+8-res on test_abupt by 0.02bp within noise floor. |
| **JJJ-K4-K-axis-alive-at-EP15** | H285 tanjiro (merged 08:37Z as NEW SOTA) | K=3→K=4 anti-thetic at EP15+6-res gives val −0.8bp / test −0.7bp. Sign-consistent across all 5 paper-facing channels (5/5). K-axis still alive but slope diminishing: K=2→3 compound was super-additive (Finding ZZ); K=3→4 is weakly additive. K=5 next (H295 tanjiro) to close axis. Merge gate updated to val<5.9235, test<5.7683. |
| **III-EP-axis-fully-exhausted** | H287 askeladd (closed 08:10Z) | EP14+anti-K3+6-res val 5.9326/test 5.7764 fails gate. Combined with FFF (EP16), HHH (EP13), EP12 unavailable: EP-curve at H275 recipe fully closed as EP13≈EP14<EP15(peak)>EP16. No checkpoint variation can improve SOTA without new training. |
| **HHH-EP13-no-rescue-with-K5-Sobol** | H280 nezuko (closed 08:10Z) | EP13+Sobol-anti-K5+6-res val 5.9313/test 5.7753 fails gate. Doubled-K diversity + Sobol cannot rescue EP13 checkpoint. Combined with Finding FFF (EP16) and III (EP14): EP-axis fully exhausted on both sides of EP15 peak. |
| **GGG-Sobol-anti-non-stacking-at-K3-EP15** | H284 edward (closed 07:20Z) | Sobol×anti compound at K=3 EP15 is non-additive: val −0.012bp / test +0.065bp ≈ zero net movement vs H275. Anti-K=3 already kills the dominant linear Taylor variance; Sobol QMC targets the SAME variance budget. Both mechanisms not orthogonal at K=3. |
| **FFF-EP-axis-closed-at-H275-recipe** | H289 frieren (closed 06:05Z, no runs) | EP-axis exhausted at H275 recipe family. Only EP13/14/15 checkpoints available. H244 3-res evals show EP14 5.9613→EP15 5.9516 (peak)→EP16 5.9548 (regression). EP15 is unimodal peak. `best_epoch=15` in H244 config confirms. Do not test further EP variations at this recipe without new training. |
| **EEE-anti-dominates-Sobol-at-EP15** | H279 frieren (closed 05:00Z) vs H275 edward (merged) | Anti-K=3 beats Sobol-K=5 at EP15 by +4.8bp val / +4.4bp test (despite Sobol using 5 vs anti using 6 forwards). Anti's linear-cancellation structurally dominates Sobol's QMC coverage at BOTH EP13 (Finding VV/WW) and EP15. Anti is the right paradigm; Sobol is asymptotically suboptimal |
| **DDD-Sobol-EP15-super-additive** | H279 frieren (closed 05:00Z) vs H271 frieren (merged) | Sobol×EP15 nearly doubles benefit vs Sobol×EP13 (−9.1bp test vs −5.0bp). EP15 vs EP13 also amplifies under Sobol (−6.3bp test vs −2.2bp under random). EP15 flat minimum interacts super-additively with QMC coverage as well — mirroring Finding ZZ for anti |
| **BBB-K-curve-saturates-at-K3-EP13** | H282 alphonse + H278 askeladd (closed 03:35Z) | Full anti-thetic K-curve at EP13: K=2→K=3 is +1.8bp, K=3→K=4 is only −0.6bp (noise floor). K-axis closed at EP13. Linear Taylor cancellation captures most benefit at K=1 pair; further K adds diminishing K-averaging. Cross-EP: H275 K=3 EP15 beats H278 K=4 EP13 by 7.3bp → checkpoint axis dominates K-axis |
| **AAA-sigma-flat-antithetic** | H277 tanjiro + H281 fern (closed 02:50Z & 04:25Z) | σ=3e-4 uniformly slightly WORSE than σ=5e-4 in anti-thetic family on BOTH EP13 (+9.9bp val) and EP15 (+1.2bp val). σ axis fully closed for anti-thetic stack family — linear Taylor cancellation already exhausts the σ-dependent benefit. Bigger gap at EP13 due to less flat loss surface |
| **ZZ-EP15-anti-super-additive** | H275 edward (merged 01:17Z) | EP15×anti-K=3 is super-additive: +0.28bp val / +0.51bp test bonus vs pure-additive expectation. Flatter EP15 EMA minimum makes Taylor linear-term cancellation more meaningful. All channels improve (WSS −6.2bp, SP −8.8bp, VP −5.8bp) vs H274 |
| **XX-sigma3e4-EP15-channel-asymmetric** | H276 thorfinn (closed 00:40Z) | σ=3e-4+EP15+random K=5 gives SP/VP-favorable (−2.1/−2.3bp) but WSS-unfavorable (+1.9bp) vs SOTA. σ (global mag) ⊥ anti-thetic (pairing structure) at channel level |
| **WW-antithetic-dominates-K-scaling** | H269 alphonse (closed 00:02Z) | Random K=10 (10p) cannot match anti-K=3 (6p) on any channel in EP13+stack; 67% K-multiplication leaves +0.15bp test gap. Anti-thetic structural advantage > random K-scaling efficiency |
| **VV-antithetic-stacked** | H274 fern (merged 23:35Z) | EP13 anti-K=3 (72p) beats Sobol K=5 (60p) on all channels; −8.4bp test vs H253 random K=5; linear Taylor cancellation survives 12×(res,mirror) averaging |
| **UU-Sobol-QMC** | H271 frieren (merged 22:50Z) | Sobol K=5 QMC (proj_dim=1024) beats random K=5 uniform −0.5bp/ch; test_WSS −5.0bp |
| **TT-Hutchinson** | H272 nezuko | β=1e4 σ-shrinkage: 85% params at σ<0.5·σ_base, collapses diversity |
| **SS-Taylor-mixed** | H273 askeladd | EP13 Taylor-2 net 0.04bp, mixed sign across channels |
| **RR** | H270 tanjiro | σ=3e-4 on EP13+stack: −2.0bp test (WSS −3.2bp), 0.2bp from gate |
| **QQ** | H267 edward | EP15 stacks at ~55% additive rate (−5.1bp) |
| **PP** | H266 fern | 2³ ANOVA: Mirror×Noise −1.68bp interaction; Res most orthogonal axis |
| **NN** | H268 askeladd | Anti-thetic K=3 beats random K=5 by +1.3bp standalone |
| (17 more in prior rounds) | | |

---

## Next-Round Hypothesis Queue

AXES FULLY CLOSED (do not revisit without new training):
- **EP axis**: EP12 unavailable, EP13/EP14 fail gate, EP15=SOTA, EP16 hurts. Findings FFF+HHH+III close this completely.
- **σ axis (single-σ)**: σ=5e-4 optimal. Findings AAA+XX-Sobol close this.
- **σ-diversity axis (multi-σ at K=3)**: σ-mixing subsumed by K-axis. Finding LLL closes this.
- **Sobol×EP15 axis**: Anti beats Sobol at both EPs. Findings DDD+EEE close this.
- **Sobol×anti K=3 compound**: Non-additive. Finding GGG closes this.
- **Aggregation operator (homogeneous)**: mean/median/trimmed/Huber all null. Finding OOO closes this.
- **Noise distribution family**: Gaussian ≈ Laplace ≈ Student-t df=3 at matched RMS σ=5e-4. Findings MMM+QQQ close this.
- **Resolution axis (extent)**: 8-res upper (+160K,+192K) +11.8bp val; 8-res lower (+16K,+24K) +1.7bp val / +1.0bp test. Current 6-res {32K-131K} is Pareto optimum. Findings PPP+RRR close this asymmetrically.

IN-FLIGHT (do not duplicate):
- **K=4+8-res compound** (fern H296 PR #1484) — highest-EV K×res cell; Finding KKK orthogonality proven; val arm ~13:30Z
- **K=5 at EP15** (tanjiro H295) — closes K-axis; val arm ~12:00Z
- **Per-layer noise stratification σ_attn=0** (thorfinn H297 PR #1485) — next SP-targeted intervention; ~15:00Z
- **Resolution lower ladder** (frieren H291 PR #1477) — test relaunch in progress; val 5.9252 fails gate but want full result
- **Resolution upper ladder** (edward H292) — 8-res {32K-192K}; no val yet
- **Aggregation operators** (alphonse H286) — multi-mode sweep; val 5.9242 (fails gate), test arm pending
- **Laplace noise** (nezuko H293) — val 5.9246 (fails gate by +0.11bp), test pending
- **Student-t noise df=3** (askeladd H294) — val 5.9248 (fails gate by +0.13bp), test pending

PRIORITY NEXT ROUND (when students become idle):
1. Student-t df=1.5 (Cauchy-like, infinite variance but check stability) — if H294 wins
2. Per-layer noise stratification — target conv vs attention separately
3. Y-mirror augmentation (double-mirror x+y) — lateral symmetry exploitation
4. Best-of-K (min-loss per channel) selection — orthogonal to mean aggregation
5. Noise anisotropy — per-tensor σ proportional to gradient magnitude
6. SP floor (3.6427→3.577, 6.5bp gap) is the binding Morgan target; need wins on SP specifically

---

## Exhaustion Map — TTA Mechanisms (Current)

| Mechanism | Status |
|---|---|
| EP13 + anti-thetic K=3 + 6-res + mirror | ✓ H274 val 5.9322 / test 5.7763 (prior SOTA) |
| EP13 + Sobol QMC K=5 + 6-res + mirror | ✓ H271 val 5.9368 / test 5.7797 |
| EP15 + K=5 random + 6-res + mirror | ✓ H267 val 5.9367 / test 5.7825 |
| EP13 + K=5 random + 6-res + mirror | ✓ H253 val 5.9418 / test 5.7847 |
| EP13 + K=10 random + 6-res + mirror | ✗ H269 val 5.9330 / test 5.7778 — Finding WW (gates +0.08/+0.15bp) |
| EP13 + anti-thetic K=2 pairs (4p) + 6-res + mirror | ✗ H282 5.9340/5.7783 — Finding BBB-K-floor (fails H275 gate +9.7bp) |
| EP13 + anti-thetic K=4 pairs (8p) + 6-res + mirror | ✗ H278 5.9316/5.7758 — Finding BBB-K-saturates (K=4 only −0.6bp over K=3, fails H275 gate +7.3bp) |
| EP15 + anti-thetic K=3 + 6-res + mirror | ✓ H275 val 5.9243 / test 5.7690 ← **CURRENT SOTA** |
| EP15 + σ=3e-4 + K=5 random + 6-res + mirror | ✗ H276 5.9314/5.7768 — Finding XX (val passes, test +0.05bp, WSS+1.9bp) |
| EP15 + σ=3e-4 + Sobol K=5 + 6-res + mirror | ✗ H283 5.9274/5.7725 — Finding XX-Sobol-confirmed (fails H275 gate +3.1/+3.5bp; σ-axis closed across all noise families at EP15) |
| EP15 + σ=3e-4 + anti-thetic K=3 + 6-res + mirror | ✗ H277 5.9255/5.7705 — Finding AAA (σ=3e-4 flat/unfavorable under anti-thetic) |
| EP15 + anti-thetic K=4 pairs (8p) + 6-res + mirror | 🟡 H285 tanjiro (running) |
| EP15 + Sobol QMC K=5 + 6-res + mirror | ✗ H279 5.9291/5.7734 — Findings DDD-Sobol-EP15-super-additive + EEE-anti-dominates-Sobol-at-EP15 (fails H275 gate +4.8/+4.4bp; anti beats Sobol at EP15 too) |
| EP13 + Sobol × anti-thetic K=5 (10p) + 6-res + mirror | ✗ H280 5.9313/5.7753 — Finding HHH (EP13+K5+Sobol fails gate; EP-axis closed below too) |
| EP15 + Sobol × anti-thetic K=3 + 6-res + mirror | ✗ H284 5.9242/5.7697 — Finding GGG (Sobol×anti non-additive at K=3; anti exhausts linear variance) |
| EP13 + anti-thetic K=3 + σ=3e-4 + 6-res + mirror | ✗ H281 5.9342/5.7785 — Finding AAA confirmed at EP13 (+9.9bp val vs SOTA) |
| EP15 + aggregation operator sweep (mean/median/trimmed/Huber) | 🆕 H286 alphonse (assigned) |
| EP14 + anti-thetic K=3 + 6-res + mirror | ✗ H287 5.9326/5.7764 — Finding III (EP14 fails gate; EP-axis fully exhausted) |
| EP15 + Laplace noise + anti-K=3 + 6-res + mirror | 🆕 H293 nezuko (just assigned, PR #1481) |
| EP15 + Student-t df=3 noise + anti-K=3 + 6-res + mirror | 🆕 H294 askeladd (just assigned, PR #1482) |
| EP15 + anti-K=3 + 8-res densified (+40960+57344) + mirror | ✗ H288 val 5.9229/test 5.7685 — Finding KKK (val passes gate, test +0.02bp miss; VP−2.1bp gain, SP+1.1bp regression; K-axis⊥res-axis) |
| EP15 + anti-K=4 + 8-res densified (+40960+57344) + mirror | 🆕 H296 fern (PR #1484, just assigned) |
| EP15 + anti-K=3 + 8-res LOWER (+16384+24576) + mirror | ✗ H291 val 5.9252 / test 5.7693 — Finding RRR (fails both gates by ~1bp; res-axis closed below 32K) |
| EP15 + anti-K=3 + 8-res UPPER (+163840+196608) + mirror | 🆕 H292 edward (just assigned) |
| EP15 + Sobol × anti-thetic K=3 + 6-res + mirror | ✗ H284 5.9242/5.7697 — Finding GGG (Sobol×anti non-additive at K=3; anti exhausts linear variance) |
| EP15 + multi-σ{3e-4,5e-4,7e-4} + anti-K=3 + 6-res + mirror | ✗ H290 val 5.9241/test 5.7689 — Finding LLL (multi-σ null vs H285; σ-diversity subsumed by K-axis) |
| EP15 + anti-K=4 + 6-res + mirror (σ_attn=0, σ_mlp=5e-4) | 🆕 H297 thorfinn (PR #1485, just assigned) |
| EP16 checkpoint axis | ✗ H289 (closed, checkpoint deleted; FFF banked) |
| EP13 + Hutchinson curvature-σ | ✗ H272 Finding TT |
| EP13 + Taylor 2nd-order correction | ✗ H273 Finding SS |
| EP13 + σ=3e-4 random K=5 | H270 5.9394/5.7827 (0.2bp from old gate) |
| SWA adjacent-EP | ✗ Finding LL-SWA-null |
| EP16 + any TTA | ✗ Finding LL-EPchain |
| H183 + 6-res stacked | ✗ Finding HH-H183 |
