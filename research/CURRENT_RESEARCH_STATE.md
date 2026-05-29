# SENPAI Research State

**Updated**: 2026-05-29 21:18Z | Branch: `tay` | **SOTA: H267 EP15+full stack (PR #1447)** | Round 4k: 8 active experiments

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| H267 EP15+K=5+6-res+mirror ← **CURRENT SOTA** | **5.9367%** | **5.7825%** | **6.6898%** | **3.3850%** | **3.6505%** | snouz8zi |
| H253 EP13+K=5+6-res+mirror (prior SOTA) | 5.9418% | 5.7847% | 6.6895% | 3.3891% | 3.6592% | qytjlv97 |
| Transolver-3 target (Morgan Issue #1056) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% | — |

**Merge gate**: val_abupt < **5.9367%** AND test_abupt < **5.7825%**
**Paper floors**: test_VP 3.3850 ≤ 3.421 ✓ | test_WSS 6.6898 ≤ 6.727 ✓ | test_SP 3.6505 > 3.577 ✗ (7.35bp gap)

---

## Active Fleet (as of 21:18Z — 8 students all active)

| PR | Student | Hypothesis | Status | val | ETA |
|---|---|---|---|---|---|
| **#1454** | **fern** | **H274: EP13+anti-K=3+6-res+mirror** | 🟢 **test running** | **5.9322 ← BEST** | ~22:30Z |
| **#1448** | **alphonse** | **H269: EP13+K=10+6-res+mirror** | 🟢 test running | **5.9330** | ~23:00Z |
| **#1451** | **frieren** | **H271: EP13+Sobol+K=5+6-res+mirror** | 🟢 val complete | 5.9368 | ~22:30Z |
| **#1455** | **edward** | **H275: EP15+anti-K=3+6-res+mirror** | 🟡 running | — | ~01:00Z |
| **#1456** | **thorfinn** | **H276: EP15+σ=3e-4+K=5+6-res+mirror** | 🟡 running | — | ~01:20Z |
| **#1457** | **tanjiro** | **H277: EP15+σ=3e-4+anti-K=3+6-res+mirror** | 🟡 running | — | ~01:05Z |
| **#1452** | **nezuko** | **H272: EP13+Hutchinson-σ+K=5+6-res+mirror** | 🟢 val running | 5.9507 | ~22:30Z |
| **#1458** | **askeladd** | **H278: EP13+anti-K=4 pairs (8p)+6-res+mirror** | 🟡 just assigned | — | ~02:30Z |

**Gate**: val < **5.9367** AND test < **5.7825**

**Active SOTA candidates** (val beats gate):
- fern H274: val 5.9322 (−4.5bp) — **LEADING**, test pending ~22:30Z
- alphonse H269: val 5.9330 (−3.7bp) — test pending ~23:00Z
- frieren H271: val 5.9368 (+0.1bp, borderline) — test unlikely to clear without beating val gate

**Missed gate (running for mechanistic info)**:
- nezuko H272 val 5.9507 — Hutchinson σ scaling helps ~0bp, not worth chasing
- edward H275, thorfinn H276, tanjiro H277 — results ~01:00-01:20Z

---

## Closed This Session

| PR | Student | Finding | val | test |
|---|---|---|---|---|
| #1453 askeladd H273 | EP13+Taylor-2+λ-sweep | **Finding SS-Taylor-mixed-sign**: r_mean mixed sign across channels; net 0.04bp gain, Arm B not triggered | 5.9543 | 5.7973 |
| #1449 tanjiro H270 | EP13+σ=3e-4 stacked | **Finding RR-σ-EP13-stack**: σ=3e-4 gives −2.0bp test (test_WSS −3.2bp), 0.2bp from gate | 5.9394 | 5.7827 |
| #1433 thorfinn H257 | EP13+σ-sweep stacked | **Finding LL-noise-sweep-stacked**: σ-basin flat [1e-4, 5e-4] at EP13; σ=1e-3 worse | 5.9417 | 5.7856 |
| #1447 edward H267 | EP15+full stack | **MERGED as SOTA**: val 5.9367 / test 5.7825 | 5.9367 | 5.7825 |

---

## Findings Bank (22 banked)

| ID | Source | Summary |
|---|---|---|
| **SS-Taylor-mixed-sign** | H273 askeladd (closed 21:15Z) | EP13 Taylor-2 correction net 0.04bp; r_mean sign-incoherent across channels at σ=5e-4 |
| **RR** | H270 tanjiro (closed 21:00Z) | σ=3e-4 on EP13+stack: −2.0bp test vs σ=5e-4, 0.2bp short of gate; test_WSS −3.2bp |
| **QQ** | H267 edward (merged 20:29Z) | EP15 stacks on H253 full recipe at ~55% additive rate (−5.1bp); sub-additive but substantial |
| **PP** | H266 fern | 2³ ANOVA: Mirror×Noise −1.68bp interaction; Res most orthogonal axis |
| **NN** | H268 askeladd | Anti-thetic K=3 (6 passes) beats random K=5 (5 passes) by +1.3bp standalone |
| **KK-noise-sat** | H262 | K=5→10 stacked: +5.2bp (much larger than standalone +1.36bp) |
| **JJ** | H253 | Noise+input-space TTA ~80% orthogonal; residual +1.3bp compounds |
| **LL-noise-sweep** | H257 thorfinn (closed 21:02Z) | EP13 stack σ-basin flat [1e-4, 5e-4], σ=1e-3 worse |
| **LL-EPchain** | H264 fern | EP13→EP14→EP15→EP16: EP15 optimum, EP16 regresses |
| **LL-SWA-null** | H263 frieren | Adjacent-EP SWA +6bp regression |
| **HH-H183** | H256 nezuko | H183+6-res stacked HURTS +70bp |
| (11 more in prior rounds) | | |

---

## Next-Round Hypothesis Queue (for when students idle)

High-EV after current round concludes:
1. **H279 (next anti-thetic)**: EP15 + anti-thetic K=5 pairs (10 passes) — if H275/H278 both beat gate, combine best-K + best-checkpoint
2. **H280 σ=3e-4 EP15 test**: H276 directly tests this — if it wins, bank Finding QQ-σ-EP15
3. **Anti-thetic K-curve**: H274 K=3 + H278 K=4 + H269 K=10 random gives 3-point K-curve. Will clearly map the anti-thetic K scaling law.
4. **SP-floor effort**: SP 3.6505 → 3.577 floor is 7.35bp — dedicated SP-targeted approach
5. **WSS target**: 6.6898 → 5.850 is 84bp — requires fundamentally new mechanism or architecture

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Status |
|---|---|
| EP15 + K=5 random + 6-res + mirror (SOTA) | ✓ H267 val 5.9367 |
| EP13 + K=5 random + 6-res + mirror | ✓ H253 val 5.9418 |
| EP13 + anti-thetic K=3 + 6-res + mirror | 🟢 H274 fern val 5.9322 (TEST PENDING) |
| EP13 + K=10 random + 6-res + mirror | 🟢 H269 alphonse val 5.9330 (TEST PENDING) |
| EP15 + anti-thetic K=3 + 6-res + mirror | 🟡 H275 edward (running) |
| EP15 + σ=3e-4 + K=5 random + 6-res + mirror | 🟡 H276 thorfinn (running) |
| EP15 + σ=3e-4 + anti-thetic K=3 + 6-res + mirror | 🟡 H277 tanjiro (running) |
| EP13 + anti-thetic K=4 pairs (8p) + 6-res + mirror | 🟡 H278 askeladd (just assigned) |
| EP13 + Sobol QMC K=5 + 6-res + mirror | 🟢 H271 frieren val 5.9368 (borderline, TEST PENDING) |
| EP13 + Hutchinson curvature-σ + K=5 + 6-res + mirror | 🟢 H272 nezuko val 5.9507 (missed) |
| EP13 + Taylor 2nd-order correction | ✗ H273 askeladd val 5.9543/test 5.7973 (Finding SS) |
| EP13 + σ=3e-4 stacked | H270 tanjiro 5.9394/5.7827 (0.2bp from gate) |
| EP13 + σ=1e-4 stacked | H257 thorfinn 5.9417 (missed) |
| EP13 + σ=1e-3 stacked | H257 thorfinn 5.9626 (worse) |
| SWA adjacent-EP | ✗ Finding LL-SWA-null |
| EP16 + any TTA | ✗ Finding LL-EPchain |
| H183 + 6-res stacked | ✗ Finding HH-H183 |
