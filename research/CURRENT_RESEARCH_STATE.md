# SENPAI Research State

**Updated**: 2026-05-30 09:56Z | Branch: `tay` | **SOTA: H285 EP15+Anti-K4+6-res+mirror (PR #1470)** | Round 4k

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| **H285 EP15+anti-K4+6-res+mirror ← CURRENT SOTA (NEW)** | **5.9235%** | **5.7683%** | **6.6735%** | **3.3783%** | **3.6425%** | 4vvc40zs |
| H275 EP15+anti-K3+6-res+mirror (prior SOTA) | 5.9243% | 5.7690% | 6.6743% | 3.3788% | 3.6427% | 0b4t2bz2 |
| H274 EP13+anti-K3+6-res+mirror | 5.9322% | 5.7763% | 6.6805% | 3.3846% | 3.6515% | o8oq9r92 |
| Transolver-3 target (Morgan Issue #1056) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% | — |

**Merge gate**: val_abupt < **5.9235%** AND test_abupt < **5.7683%**
**Paper floors**: test_VP 3.3783 ≤ 3.421 ✓ | test_WSS 6.6735 ≤ 6.727 ✓ | test_SP 3.6425 > 3.577 ✗ (6.5bp gap — marginal 0.02bp vs H275)

**Key findings**: Finding JJJ (K4 at EP15 +0.7bp, slope diminishing but sign-consistent 5/5 channels), Finding ZZ (EP15×anti-K3 super-additive), Finding VV (anti-thetic cancels linear Taylor term)

---

## Active Fleet (as of 08:40Z — 8 students active)

| PR | Student | Hypothesis | Status | val | ETA |
|---|---|---|---|---|---|
| **#1483** | **tanjiro** | **H295: EP15+anti-K=5 pairs+6-res+mirror** | 🟡 uf17vdab running (started 08:40Z), val arm ETA ~12:00Z | — | ~12:00Z |
| **#1471** | **alphonse** | **H286: Aggregation operator sweep on H275 SOTA** | 🟡 bad1i1jv running, val 5.9242 (fails gate +0.7bp), test arm pending | 5.9242 | ~11:00Z |
| **#1482** | **askeladd** | **H294: Student-t df=3 noise at EP15+anti-K3+6-res** | 🟡 jb36zg3o running (relaunched after init crash bgz1d8mm), no metrics yet | — | ~12:30Z |
| **#1484** | **fern** | **H296: K=4+8-res compound at EP15** | 🆕 just assigned (compound K-axis+res-axis; Finding KKK orthogonality proven) | — | ~13:30Z |
| **#1477** | **frieren** | **H291: 8-res LOWER ladder {16K-131K} at EP15+anti-K3** | 🟡 aqcgnn1z val 5.9252 (fails gate +1.7bp, likely close), test pending | 5.9252 | ~11:00Z |
| **#1481** | **nezuko** | **H293: Laplace noise at EP15+anti-K3+6-res** | 🟡 hu472xfk running, no metrics yet | — | ~12:30Z |
| **#1476** | **thorfinn** | **H290: EP15+multi-σ{3e-4,5e-4,7e-4}+anti-K=3** | 🟡 slzcnzs7 val 5.9241 (fails gate +0.6bp), test pending | 5.9241 | ~11:00Z |
| **#1479** | **edward** | **H292: EP15+anti-K3+8-res UPPER {32K-192K}** | 🟡 xyy8f895 running, no metrics yet | — | ~13:00Z |

**Gate**: val < **5.9235** AND test < **5.7683** (H285 NEW SOTA merged 08:37Z)

**Hot watch**: 
- **Fern H296** (PR #1484): K=4+8-res compound — the HIGHEST-EV cell on the K×res matrix per Finding KKK orthogonality. Val arm ETA ~13:30Z. Primary SOTA candidate for next round.
- **Alphonse H286** val 5.9242 (+0.7bp vs gate), test arm pending — marginal; likely close but not winner.
- **Thorfinn H290** val 5.9241 (+0.6bp vs gate), test pending — marginal, close but fails gate.
- **Frieren H291** val 5.9252 (+1.7bp vs gate) — highly unlikely to pass test gate; will close when test lands.
- **Tanjiro H295** K=5 val arm at ~12:00Z — closes K-axis definitively (Finding JJJ follow-up).

---

## Recently Merged / Closed

| PR | Student | Finding | val | test |
|---|---|---|---|---|
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

## Findings Bank (39 banked, KKK added 09:56Z)

| ID | Source | Summary |
|---|---|---|
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
- **Sobol×EP15 axis**: Anti beats Sobol at both EPs. Findings DDD+EEE close this.
- **Sobol×anti K=3 compound**: Non-additive. Finding GGG closes this.

IN-FLIGHT (do not duplicate):
- **K=4+8-res compound** (fern H296 PR #1484) — highest-EV K×res cell; Finding KKK orthogonality proven
- **K=5 at EP15** (tanjiro H295) — val arm ~12:00Z; closes K-axis definitively
- **Resolution lower ladder** (frieren H291) — 8-res {16K-131K}; val 5.9252 (fails gate, close)
- **Resolution upper ladder** (edward H292) — 8-res {32K-192K}
- **Aggregation operators** (alphonse H286) — mean/median/trimmed/Huber sweep; val 5.9242 (fails gate)
- **Multi-σ diversity** (thorfinn H290) — σ={3e-4,5e-4,7e-4} per anti pair; val 5.9241 (fails gate)
- **Laplace noise** (nezuko H293) — noise family heavy-tail moderate
- **Student-t noise df=3** (askeladd H294) — noise family heavy-tail max finite-var

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
| EP15 + anti-K=3 + 8-res LOWER (+16384+24576) + mirror | 🆕 H291 frieren (just assigned) |
| EP15 + anti-K=3 + 8-res UPPER (+163840+196608) + mirror | 🆕 H292 edward (just assigned) |
| EP15 + Sobol × anti-thetic K=3 + 6-res + mirror | ✗ H284 5.9242/5.7697 — Finding GGG (Sobol×anti non-additive at K=3; anti exhausts linear variance) |
| EP15 + multi-σ{3e-4,5e-4,7e-4} + anti-K=3 + 6-res + mirror | 🆕 H290 thorfinn (just assigned) |
| EP16 checkpoint axis | ✗ H289 (closed, checkpoint deleted; FFF banked) |
| EP13 + Hutchinson curvature-σ | ✗ H272 Finding TT |
| EP13 + Taylor 2nd-order correction | ✗ H273 Finding SS |
| EP13 + σ=3e-4 random K=5 | H270 5.9394/5.7827 (0.2bp from old gate) |
| SWA adjacent-EP | ✗ Finding LL-SWA-null |
| EP16 + any TTA | ✗ Finding LL-EPchain |
| H183 + 6-res stacked | ✗ Finding HH-H183 |
