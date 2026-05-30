# SENPAI Research State

**Updated**: 2026-05-30 05:10Z | Branch: `tay` | **SOTA: H275 EP15+Anti-K3+6-res+mirror (PR #1455)** | Round 4k

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

## Active Fleet (as of 05:10Z — 8 students active)

| PR | Student | Hypothesis | Status | val | ETA |
|---|---|---|---|---|---|
| **#1467** | **edward** | **H284: EP15+Sobol-anti-K=3+6-res+mirror** | 🔥 val 5.9242 (TIED with SOTA gate, test pending) | **5.9242** | ~04:50Z |
| **#1470** | **tanjiro** | **H285: EP15+anti-K=4 pairs (8p)+6-res+mirror** | 🟡 running (K-scaling on SOTA) | — | ~07:10Z |
| **#1471** | **alphonse** | **H286: Aggregation operator sweep on H275 SOTA** | 🆕 assigned (mean/median/trimmed/Huber) | — | ~06:30Z |
| **#1472** | **askeladd** | **H287: EP14 checkpoint axis test** | 🆕 assigned (EP14 vs EP15 adjacent) | — | ~06:30Z |
| **#1473** | **fern** | **H288: 8-res densification on H275 SOTA** | 🆕 just assigned (8-res grid: +40960+57344) | — | ~07:30Z |
| **#1475** | **frieren** | **H289: EP12 checkpoint axis test** | 🆕 just assigned (EP12 vs EP15 — completes 4-pt EP-curve) | — | ~08:00Z |
| **#1460** | **nezuko** | **H280: EP13+Sobol-anti-K5 (10p)+6-res+mirror** | 🟠 pre-eval after 2 crashes (compute-heavy; alive, GPUs 100%) | — | ~06:30Z |
| **#1466** | **thorfinn** | **H283: EP15+σ=3e-4+Sobol K=5+6-res+mirror** | 🟡 running (NCCL-relaunched 02:11Z) | — | ~05:25Z |

**Gate**: val < **5.9243** AND test < **5.7690** (H275 SOTA merged 01:17Z)

**Hot watch**: 
- **Edward H284** is the PRIMARY SOTA candidate — val 5.9242 ties SOTA gate (within 0.01bp), Sobol×anti×EP15 triple compound. Test arm decides merge vs close. ETA ~04:50Z.
- Tanjiro H285 K=4 EP15 is the secondary candidate (next ~07:10Z).
- Nezuko H280 had 2 prior crashes (likely NCCL); current retry has been pre-eval for 2.5h. Worth monitoring closely if she crashes again.
- Frieren H279 CLOSED (Sobol K=5 EP15 val 5.9291 +4.8bp over SOTA gate) — banked Findings DDD-Sobol-EP15-super-additive and EEE-anti-dominates-Sobol-at-EP15.
- Fern H281 closed with Finding AAA confirmed at EP13 (σ=3e-4 worse than σ=5e-4 in anti-thetic on BOTH EP13 and EP15). σ-axis fully closed for anti-thetic stack.

---

## Recently Merged / Closed

| PR | Student | Finding | val | test |
|---|---|---|---|---|
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

## Findings Bank (33 banked, DDD+EEE added 05:10Z)

| ID | Source | Summary |
|---|---|---|
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

After current round resolves (new gate val < 5.9243 / test < 5.7690):
1. **Sobol × anti × EP15**: H284 edward (ETA ~04:50Z) — triple-mechanism compound, primary SOTA candidate
2. **Anti-thetic K-scaling at EP15**: H285 tanjiro K=4 (ETA ~07:10Z)
3. **Aggregation operator**: H286 alphonse (just assigned) — mean vs median vs trimmed vs Huber at H275 SOTA recipe; untested mechanism axis
4. **Checkpoint axis EP14**: H287 askeladd (just assigned) — EP14 adjacent to SOTA EP15; 3rd point on EP-curve
5. **8-res densification**: H288 fern (just assigned) — denser res grid (+40960+57344) at H275 SOTA recipe
6. **Checkpoint axis EP12**: H289 frieren (just assigned) — completes 4-pt EP-curve (EP12/EP13/EP14/EP15) for paper figure
7. **K-axis at EP13 CLOSED (Finding BBB)**: Do not assign further K variations at EP13 anti-thetic. K=3 is the sweet spot.
8. **σ axis CLOSED at EP15+anti (Finding AAA)**: σ=5e-4 is optimal; no further σ variations on anti-thetic stack.
9. **Sobol×EP15 CLOSED (Finding EEE)**: Anti-K=3 dominates Sobol-K=5 at both EPs; do not test more Sobol-only variants on EP15 stack.
10. **SP floor**: 3.6427 → 3.577 = 6.5bp gap — H283 thorfinn (σ+Sobol) likely closes σ at Sobol family too.

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
| EP15 + σ=3e-4 + Sobol K=5 + 6-res + mirror | 🟡 H283 thorfinn (running) |
| EP15 + σ=3e-4 + anti-thetic K=3 + 6-res + mirror | ✗ H277 5.9255/5.7705 — Finding AAA (σ=3e-4 flat/unfavorable under anti-thetic) |
| EP15 + anti-thetic K=4 pairs (8p) + 6-res + mirror | 🟡 H285 tanjiro (running) |
| EP15 + Sobol QMC K=5 + 6-res + mirror | ✗ H279 5.9291/5.7734 — Findings DDD-Sobol-EP15-super-additive + EEE-anti-dominates-Sobol-at-EP15 (fails H275 gate +4.8/+4.4bp; anti beats Sobol at EP15 too) |
| EP13 + Sobol × anti-thetic K=5 (10p) + 6-res + mirror | 🟡 H280 nezuko (running) |
| EP15 + Sobol × anti-thetic K=3 + 6-res + mirror | 🟡 H284 edward (running) |
| EP13 + anti-thetic K=3 + σ=3e-4 + 6-res + mirror | ✗ H281 5.9342/5.7785 — Finding AAA confirmed at EP13 (+9.9bp val vs SOTA) |
| EP15 + aggregation operator sweep (mean/median/trimmed/Huber) | 🆕 H286 alphonse (assigned) |
| EP14 + anti-thetic K=3 + 6-res + mirror | 🆕 H287 askeladd (assigned) |
| EP15 + anti-K=3 + 8-res densified (+40960+57344) + mirror | 🆕 H288 fern (just assigned) |
| EP13 + Hutchinson curvature-σ | ✗ H272 Finding TT |
| EP13 + Taylor 2nd-order correction | ✗ H273 Finding SS |
| EP13 + σ=3e-4 random K=5 | H270 5.9394/5.7827 (0.2bp from old gate) |
| SWA adjacent-EP | ✗ Finding LL-SWA-null |
| EP16 + any TTA | ✗ Finding LL-EPchain |
| H183 + 6-res stacked | ✗ Finding HH-H183 |
