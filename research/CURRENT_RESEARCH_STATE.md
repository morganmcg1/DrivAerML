# SENPAI Research State

**Updated**: 2026-05-29 23:40Z | Branch: `tay` | **SOTA: H274 EP13+Anti-K3+6-res+mirror (PR #1454)** | Round 4k

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| H274 EP13+anti-K3+6-res+mirror ← **CURRENT SOTA** | **5.9322%** | **5.7763%** | **6.6805%** | **3.3846%** | **3.6515%** | o8oq9r92 |
| H271 EP13+Sobol-K5+6-res+mirror (prior) | 5.9368% | 5.7797% | 6.6848% | 3.3864% | 3.6512% | o9hb87lt |
| H267 EP15+K=5 random+6-res+mirror | 5.9367% | 5.7825% | 6.6898% | 3.3850% | 3.6505% | snouz8zi |
| Transolver-3 target (Morgan Issue #1056) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% | — |

**Merge gate**: val_abupt < **5.9322%** AND test_abupt < **5.7763%**
**Paper floors**: test_VP 3.3846 ≤ 3.421 ✓ | test_WSS 6.6805 ≤ 6.727 ✓ | test_SP 3.6515 > 3.577 ✗ (7.5bp gap)

**Key findings**: Finding VV (anti-thetic cancels linear Taylor term, dominates Sobol on all channels), Finding UU (Sobol QMC uniform -0.5bp/ch over random)

---

## Active Fleet (as of 23:40Z — 8 students all active)

| PR | Student | Hypothesis | Status | val | ETA |
|---|---|---|---|---|---|
| **#1448** | **alphonse** | **H269: EP13+K=10+6-res+mirror** | 🟢 test running | 5.9330 | ~00:15Z |
| **#1455** | **edward** | **H275: EP15+anti-K=3+6-res+mirror** | 🟡 running | — | ~01:00Z |
| **#1456** | **thorfinn** | **H276: EP15+σ=3e-4+K=5+6-res+mirror** | 🟡 running | — | ~01:20Z |
| **#1457** | **tanjiro** | **H277: EP15+σ=3e-4+anti-K=3+6-res+mirror** | 🟡 running | — | ~01:05Z |
| **#1458** | **askeladd** | **H278: EP13+anti-K=4 pairs (8p)+6-res+mirror** | 🟡 running | — | ~02:30Z |
| **#1459** | **frieren** | **H279: EP15+Sobol-K5+6-res+mirror** | 🟡 running | — | ~02:00Z |
| **#1460** | **nezuko** | **H280: EP13+Sobol-anti-K5 (10p)+6-res+mirror** | 🟡 running | — | ~05:30Z |
| **#1461** | **fern** | **H281: EP13+anti-K3+σ=3e-4+6-res+mirror** | 🟡 just assigned | — | ~03:40Z |

**Gate**: val < **5.9322** AND test < **5.7763**

**Note on alphonse H269**: val 5.9330 is ABOVE new gate (5.9322). Val won't clear. But test result is informative for the K=10 random K-curve (completing the picture of K-scaling in stacked context). Close if test misses gate; bank K=10 K-curve data.

---

## Recently Merged / Closed

| PR | Student | Finding | val | test |
|---|---|---|---|---|
| #1454 fern H274 | EP13+anti-K=3 stacked | **MERGED as SOTA** | 5.9322 | 5.7763 |
| #1451 frieren H271 | EP13+Sobol K=5 stacked | **MERGED** (prior SOTA, now superseded by H274) | 5.9368 | 5.7797 |
| #1452 nezuko H272 | Hutchinson σ-scaling | **Finding TT-Hutchinson-shrinkage** | 5.9507 | 5.7944 |
| #1453 askeladd H273 | Taylor-2 correction | **Finding SS-Taylor-mixed-sign** | 5.9543 | 5.7973 |
| #1449 tanjiro H270 | EP13+σ=3e-4 stacked | **Finding RR-σ-EP13-stack** | 5.9394 | 5.7827 |
| #1447 edward H267 | EP15+full stack | **Prior SOTA** (now 2 generations back) | 5.9367 | 5.7825 |

---

## Findings Bank (25 banked)

| ID | Source | Summary |
|---|---|---|
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

After current round resolves:
1. **Anti-thetic × EP15**: H275 edward tests this directly
2. **σ=3e-4 × anti-thetic EP13**: H281 fern tests this directly  
3. **σ=3e-4 × anti-thetic EP15**: H277 tanjiro tests this
4. **Sobol × anti-thetic**: H280 nezuko tests Sobol-anti-thetic K=5 (10p)
5. **Anti-thetic K-scaling**: H278 askeladd K=4 + H274 K=3 → K-curve for anti-thetic stacked
6. **SP floor**: 3.6515 → 3.577 = 7.5bp — biggest remaining paper-facing gap

---

## Exhaustion Map — TTA Mechanisms (Current)

| Mechanism | Status |
|---|---|
| EP13 + anti-thetic K=3 + 6-res + mirror ← **SOTA** | ✓ H274 val 5.9322 / test 5.7763 |
| EP13 + Sobol QMC K=5 + 6-res + mirror | ✓ H271 val 5.9368 / test 5.7797 |
| EP15 + K=5 random + 6-res + mirror | ✓ H267 val 5.9367 / test 5.7825 |
| EP13 + K=5 random + 6-res + mirror | ✓ H253 val 5.9418 / test 5.7847 |
| EP13 + K=10 random + 6-res + mirror | 🟢 H269 alphonse val 5.9330 (val above gate) |
| EP15 + anti-thetic K=3 + 6-res + mirror | 🟡 H275 edward (running) |
| EP15 + σ=3e-4 + K=5 random + 6-res + mirror | 🟡 H276 thorfinn (running) |
| EP15 + σ=3e-4 + anti-thetic K=3 + 6-res + mirror | 🟡 H277 tanjiro (running) |
| EP13 + anti-thetic K=4 pairs (8p) + 6-res + mirror | 🟡 H278 askeladd (running) |
| EP15 + Sobol QMC K=5 + 6-res + mirror | 🟡 H279 frieren (running) |
| EP13 + Sobol × anti-thetic K=5 (10p) + 6-res + mirror | 🟡 H280 nezuko (running) |
| EP13 + anti-thetic K=3 + σ=3e-4 + 6-res + mirror | 🟡 H281 fern (just assigned) |
| EP13 + Hutchinson curvature-σ | ✗ H272 Finding TT |
| EP13 + Taylor 2nd-order correction | ✗ H273 Finding SS |
| EP13 + σ=3e-4 random K=5 | H270 5.9394/5.7827 (0.2bp from old gate) |
| SWA adjacent-EP | ✗ Finding LL-SWA-null |
| EP16 + any TTA | ✗ Finding LL-EPchain |
| H183 + 6-res stacked | ✗ Finding HH-H183 |
