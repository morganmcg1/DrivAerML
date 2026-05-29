# SENPAI Research State

**Updated**: 2026-05-29 22:58Z | Branch: `tay` | **SOTA: H271 EP13+Sobol-K5+6-res+mirror (PR #1451)** | Round 4k: 8 active experiments

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| H271 EP13+Sobol-K5+6-res+mirror ← **CURRENT SOTA** | **5.9368%** | **5.7797%** | **6.6848%** | **3.3864%** | **3.6512%** | o9hb87lt |
| H267 EP15+K=5 random+6-res+mirror (prior SOTA) | 5.9367% | 5.7825% | 6.6898% | 3.3850% | 3.6505% | snouz8zi |
| H253 EP13+K=5 random+6-res+mirror | 5.9418% | 5.7847% | 6.6895% | 3.3891% | 3.6592% | qytjlv97 |
| Transolver-3 target (Morgan Issue #1056) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% | — |

**Merge gate**: val_abupt < **5.9368%** AND test_abupt < **5.7797%**
**Paper floors**: test_VP 3.3864 ≤ 3.421 ✓ | test_WSS 6.6848 ≤ 6.727 ✓ | test_SP 3.6512 > 3.577 ✗ (7.42bp gap)

**Finding UU-Sobol-QMC-coverage**: Sobol K=5 QMC (proj_dim=1024 scrambled) beats i.i.d. K=5 random by uniform −0.5bp/channel improvement. test_WSS −5.0bp (best in program). Mechanism: low-discrepancy coverage in perturbation space vs random clustering.

---

## Active Fleet (as of 22:58Z — 8 students all active)

| PR | Student | Hypothesis | Status | val | ETA |
|---|---|---|---|---|---|
| **#1454** | **fern** | **H274: EP13+anti-K=3+6-res+mirror** | 🟢 test running | **5.9322 ← BEST val** | ~23:30Z |
| **#1448** | **alphonse** | **H269: EP13+K=10+6-res+mirror** | 🟢 running | **5.9330** | ~23:30Z |
| **#1455** | **edward** | **H275: EP15+anti-K=3+6-res+mirror** | 🟡 running | — | ~01:00Z |
| **#1456** | **thorfinn** | **H276: EP15+σ=3e-4+K=5+6-res+mirror** | 🟡 running | — | ~01:20Z |
| **#1457** | **tanjiro** | **H277: EP15+σ=3e-4+anti-K=3+6-res+mirror** | 🟡 running | — | ~01:05Z |
| **#1458** | **askeladd** | **H278: EP13+anti-K=4 pairs (8p)+6-res+mirror** | 🟡 running | — | ~02:30Z |
| **#1459** | **frieren** | **H279: EP15+Sobol-K5+6-res+mirror** | 🟡 just assigned | — | ~02:00Z |
| **#1460** | **nezuko** | **H280: EP13+Sobol-anti-K5 (10p)+6-res+mirror** | 🟡 just assigned | — | ~05:30Z |

**Gate**: val < **5.9368** AND test < **5.7797**

**Active SOTA candidates** (val beats gate):
- fern H274: val 5.9322 (−4.6bp) — LEADING, test pending ~23:30Z. **If test < 5.7797, immediate SOTA.**
- alphonse H269: val 5.9330 (−3.8bp) — also beats gate, test pending ~23:30Z

**Just closed / just assigned**:
- frieren H271 ✓ MERGED as SOTA (PR #1451)
- nezuko H272 ✗ CLOSED Finding TT-Hutchinson-shrinkage

---

## Closed This Session

| PR | Student | Finding | val | test |
|---|---|---|---|---|
| #1451 frieren H271 | EP13+Sobol K=5 stacked | **MERGED as SOTA**: val 5.9368 / test 5.7797 | 5.9368 | 5.7797 |
| #1452 nezuko H272 | EP13+Hutchinson σ-scaling | **Finding TT-Hutchinson-shrinkage**: β=1e4 shrinks 85% params to σ<0.5·σ_base, collapses diversity | 5.9507 | 5.7944 |
| #1453 askeladd H273 | EP13+Taylor-2+λ-sweep | **Finding SS-Taylor-mixed-sign**: r_mean mixed sign across channels, 0.04bp net | 5.9543 | 5.7973 |
| #1449 tanjiro H270 | EP13+σ=3e-4 stacked | **Finding RR-σ-EP13-stack**: σ=3e-4 −2.0bp test, 0.2bp from gate | 5.9394 | 5.7827 |
| #1433 thorfinn H257 | EP13+σ-sweep | **Finding LL-noise-sweep-stacked**: flat [1e-4, 5e-4] at EP13 | 5.9417 | 5.7856 |
| #1447 edward H267 | EP15+full stack | **Prior SOTA**: val 5.9367 / test 5.7825 | 5.9367 | 5.7825 |

---

## Findings Bank (24 banked)

| ID | Source | Summary |
|---|---|---|
| **UU-Sobol-QMC** | H271 frieren (merged 22:50Z) | Sobol K=5 QMC (proj_dim=1024) beats random K=5 uniform −0.5bp/ch; test_WSS −5.0bp (best) |
| **TT-Hutchinson** | H272 nezuko (closed 22:48Z) | β=1e4 σ-shrinkage too aggressive: 85% params at σ<0.5·σ_base; collapses TTA diversity |
| **SS-Taylor-mixed** | H273 askeladd (closed 21:15Z) | EP13 Taylor-2 correction net 0.04bp; r_mean sign-incoherent across channels at σ=5e-4 |
| **RR** | H270 tanjiro | σ=3e-4 on EP13+stack: −2.0bp test (WSS −3.2bp), 0.2bp from gate |
| **QQ** | H267 edward | EP15 stacks on full recipe at ~55% additive rate (−5.1bp) |
| **PP** | H266 fern | 2³ ANOVA: Mirror×Noise −1.68bp interaction; Res most orthogonal axis |
| **NN** | H268 askeladd | Anti-thetic K=3 (6 passes) beats random K=5 by +1.3bp standalone |
| **KK-noise-sat** | H262 | K=5→10 stacked: +5.2bp |
| **JJ** | H253 | Noise+input-space TTA ~80% orthogonal; +1.3bp compounds |
| **LL-noise-sweep** | H257 thorfinn | EP13 stack σ-basin flat [1e-4, 5e-4], σ=1e-3 worse |
| **LL-EPchain** | H264 fern | EP13→EP15 optimum, EP16 regresses |
| **LL-SWA-null** | H263 frieren | Adjacent-EP SWA +6bp regression |
| **HH-H183** | H256 nezuko | H183+6-res stacked HURTS +70bp |
| (11 more in prior rounds) | | |

---

## Next-Round Hypothesis Queue (after current round)

Key open axes on the QMC frontier:
1. **H279 frieren (assigned)**: EP15 + Sobol K=5 — QMC×checkpoint compound (H271 EP13+Sobol just became SOTA; can EP15 add more?)
2. **H280 nezuko (assigned)**: EP13 + Sobol-anti-thetic K=5 (10p) — QMC×anti-thetic compound (most novel mechanism combination in queue)
3. **Fern H274 test pending**: If anti-thetic beats Sobol on test → H281 Sobol-anti-thetic at EP15 (triple compound)
4. **Anti-thetic K-curve**: H278 askeladd K=4 + H274 fern K=3 → map anti-thetic K scaling in stacked form
5. **SP-floor effort**: 3.6512 → 3.577 floor is 7.42bp — dedicated SP approach

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Status |
|---|---|
| EP13 + Sobol QMC K=5 + 6-res + mirror ← **CURRENT SOTA** | ✓ H271 val 5.9368 / test 5.7797 |
| EP15 + K=5 random + 6-res + mirror | ✓ H267 val 5.9367 / test 5.7825 |
| EP13 + K=5 random + 6-res + mirror | ✓ H253 val 5.9418 / test 5.7847 |
| EP13 + anti-thetic K=3 + 6-res + mirror | 🟢 H274 fern val 5.9322 (TEST PENDING) |
| EP13 + K=10 random + 6-res + mirror | 🟢 H269 alphonse val 5.9330 (TEST PENDING) |
| EP15 + anti-thetic K=3 + 6-res + mirror | 🟡 H275 edward (running) |
| EP15 + σ=3e-4 + K=5 random + 6-res + mirror | 🟡 H276 thorfinn (running) |
| EP15 + σ=3e-4 + anti-thetic K=3 + 6-res + mirror | 🟡 H277 tanjiro (running) |
| EP13 + anti-thetic K=4 pairs (8p) + 6-res + mirror | 🟡 H278 askeladd (running) |
| EP15 + Sobol QMC K=5 + 6-res + mirror | 🟡 H279 frieren (just assigned) |
| EP13 + Sobol × anti-thetic K=5 (10p) + 6-res + mirror | 🟡 H280 nezuko (just assigned) |
| EP13 + Hutchinson curvature-σ + K=5 + 6-res + mirror | ✗ H272 nezuko (Finding TT) |
| EP13 + Taylor 2nd-order correction | ✗ H273 askeladd (Finding SS) |
| EP13 + σ=3e-4 stacked | H270 tanjiro 5.9394/5.7827 (0.2bp from gate) |
| EP13 + σ=1e-4 stacked | H257 thorfinn 5.9417 (missed) |
| EP13 + σ=1e-3 stacked | H257 thorfinn 5.9626 (worse) |
| SWA adjacent-EP | ✗ Finding LL-SWA-null |
| EP16 + any TTA | ✗ Finding LL-EPchain |
| H183 + 6-res stacked | ✗ Finding HH-H183 |
