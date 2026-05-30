# SENPAI Research State

**Updated**: 2026-05-30 01:22Z | Branch: `tay` | **SOTA: H275 EP15+Anti-K3+6-res+mirror (PR #1455)** | Round 4k

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| H275 EP15+anti-K3+6-res+mirror ← **CURRENT SOTA** | **5.9243%** | **5.7690%** | **6.6743%** | **3.3788%** | **3.6427%** | 0b4t2bz2 |
| H274 EP13+anti-K3+6-res+mirror (prior) | 5.9322% | 5.7763% | 6.6805% | 3.3846% | 3.6515% | o8oq9r92 |
| H271 EP13+Sobol-K5+6-res+mirror | 5.9368% | 5.7797% | 6.6848% | 3.3864% | 3.6512% | o9hb87lt |
| H267 EP15+K=5 random+6-res+mirror | 5.9367% | 5.7825% | 6.6898% | 3.3850% | 3.6505% | snouz8zi |
| Transolver-3 target (Morgan Issue #1056) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% | — |

**Merge gate**: val_abupt < **5.9243%** AND test_abupt < **5.7690%**
**Paper floors**: test_VP 3.3788 ≤ 3.421 ✓ | test_WSS 6.6743 ≤ 6.727 ✓ | test_SP 3.6427 > 3.577 ✗ (6.5bp gap — improved 1.0bp vs H274)

**Key findings**: Finding ZZ (EP15×anti-K3 super-additive: +0.28bp val, +0.51bp test bonus vs pure-additive), Finding VV (anti-thetic cancels linear Taylor term), Finding UU (Sobol QMC uniform −0.5bp/ch over random)

---

## Active Fleet (as of 01:22Z — 8 students active)

| PR | Student | Hypothesis | Status | val | ETA |
|---|---|---|---|---|---|
| **#1467** | **edward** | **H284: EP15+Sobol-anti-K=3+6-res+mirror** | 🆕 assigned (triple-mechanism compound) | — | ~05:00Z |
| **#1457** | **tanjiro** | **H277: EP15+σ=3e-4+anti-K=3+6-res+mirror** | 🟡 val 5.9255 — test arm running (fails new H275 gate) | 5.9255 | test ~03:00Z |
| **#1458** | **askeladd** | **H278: EP13+anti-K=4 pairs (8p)+6-res+mirror** | 🟡 val 5.9316 — test arm running (fails new H275 gate) | 5.9316 | test ~04:30Z |
| **#1459** | **frieren** | **H279: EP15+Sobol-K5+6-res+mirror** | 🟠 crashed (NCCL timeout) + relaunched 01:35Z | — | ~04:40Z |
| **#1460** | **nezuko** | **H280: EP13+Sobol-anti-K5 (10p)+6-res+mirror** | 🟡 running | — | ~05:30Z |
| **#1461** | **fern** | **H281: EP13+anti-K3+σ=3e-4+6-res+mirror** | 🟡 relaunched after flag-fix | — | ~04:30Z |
| **#1465** | **alphonse** | **H282: EP13+anti-K=2 pairs (4p)+6-res+mirror** | 🟡 running (K-curve floor) | — | ~02:30Z |
| **#1466** | **thorfinn** | **H283: EP15+σ=3e-4+Sobol K=5+6-res+mirror** | 🟠 crashed (NCCL timeout) + relaunched 02:11Z | — | ~05:25Z |

**Gate**: val < **5.9243** AND test < **5.7690** (H275 SOTA merged 01:17Z)

**Hot watch**: 
- Tanjiro H277 (val 5.9255) and askeladd H278 (val 5.9316) both FAIL the new H275 val gate — their test arms will be informative for findings but unlikely to SOTA. Monitor for marginal K/σ channel insights.
- Edward H284 is the primary SOTA candidate — triple-mechanism compound (Sobol+anti+EP15). Needs Sobol integration into eval_tta_h252.py before run can start (~20-30 min dev).

---

## Recently Merged / Closed

| PR | Student | Finding | val | test |
|---|---|---|---|---|
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

## Findings Bank (28 banked)

| ID | Source | Summary |
|---|---|---|
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

After current round resolves (new gate val < 5.9243 / test < 5.7690):
1. **Sobol × anti × EP15**: H284 edward tests this directly (triple-mechanism compound, ETA ~05:00Z)
2. **σ=3e-4 × anti-thetic EP15**: H277 tanjiro (val 5.9255 — informative but likely fails H275 gate)
3. **Anti-thetic K-scaling at EP15**: H282 (K=2) + H274 (K=3) + H278 (K=4) → K-curve 
4. **SP floor**: 3.6427 → 3.577 = 6.5bp — primary paper-facing gap. H281 tests σ+anti; H283 tests σ+Sobol
5. **H285 candidate**: If Sobol-anti gives further −2-3bp, next: EP15+Sobol-anti K=4 or K=5

---

## Exhaustion Map — TTA Mechanisms (Current)

| Mechanism | Status |
|---|---|
| EP13 + anti-thetic K=3 + 6-res + mirror | ✓ H274 val 5.9322 / test 5.7763 (prior SOTA) |
| EP13 + Sobol QMC K=5 + 6-res + mirror | ✓ H271 val 5.9368 / test 5.7797 |
| EP15 + K=5 random + 6-res + mirror | ✓ H267 val 5.9367 / test 5.7825 |
| EP13 + K=5 random + 6-res + mirror | ✓ H253 val 5.9418 / test 5.7847 |
| EP13 + K=10 random + 6-res + mirror | ✗ H269 val 5.9330 / test 5.7778 — Finding WW (gates +0.08/+0.15bp) |
| EP13 + anti-thetic K=2 pairs (4p) + 6-res + mirror | 🆕 H282 alphonse (assigning) |
| EP15 + anti-thetic K=3 + 6-res + mirror | ✓ H275 val 5.9243 / test 5.7690 ← **CURRENT SOTA** |
| EP15 + σ=3e-4 + K=5 random + 6-res + mirror | ✗ H276 5.9314/5.7768 — Finding XX (val passes, test +0.05bp, WSS+1.9bp) |
| EP15 + σ=3e-4 + Sobol K=5 + 6-res + mirror | 🆕 H283 thorfinn (assigning) |
| EP15 + σ=3e-4 + anti-thetic K=3 + 6-res + mirror | 🟡 H277 tanjiro (running) |
| EP13 + anti-thetic K=4 pairs (8p) + 6-res + mirror | 🟡 H278 askeladd (running) |
| EP15 + Sobol QMC K=5 + 6-res + mirror | 🟡 H279 frieren (running) |
| EP13 + Sobol × anti-thetic K=5 (10p) + 6-res + mirror | 🟡 H280 nezuko (running) |
| EP15 + Sobol × anti-thetic K=3 + 6-res + mirror | 🆕 H284 edward (assigned) |
| EP13 + anti-thetic K=3 + σ=3e-4 + 6-res + mirror | 🟡 H281 fern (just assigned) |
| EP13 + Hutchinson curvature-σ | ✗ H272 Finding TT |
| EP13 + Taylor 2nd-order correction | ✗ H273 Finding SS |
| EP13 + σ=3e-4 random K=5 | H270 5.9394/5.7827 (0.2bp from old gate) |
| SWA adjacent-EP | ✗ Finding LL-SWA-null |
| EP16 + any TTA | ✗ Finding LL-EPchain |
| H183 + 6-res stacked | ✗ Finding HH-H183 |
