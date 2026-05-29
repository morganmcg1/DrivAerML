# SENPAI Research State

**Updated**: 2026-05-29 20:45Z | Branch: `tay` | **SOTA: H267 EP15+full stack (PR #1447 merged 20:29Z)** | Round 4k: 8 active, 1 just assigned

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| H267 EP15+K=5+6-res+mirror ← **CURRENT SOTA** | **5.9367%** | **5.7825%** | **6.6898%** | **3.3850%** | **3.6505%** | snouz8zi |
| H253 EP13+K=5+6-res+mirror (prior SOTA) | 5.9418% | 5.7847% | 6.6895% | 3.3891% | 3.6592% | qytjlv97 |
| H244 EP15+6-res mirror (prior SOTA) | 5.9452% | 5.7896% | 6.6947% | 3.3882% | 3.6595% | bh7we7p6 |
| Transolver-3 target (Morgan Issue #1056) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% | — |

**Merge gate**: val_abupt < **5.9367%** AND test_abupt < **5.7825%**
**Paper floors**: test_VP 3.3850 ≤ 3.421 ✓ | test_WSS 6.6898 ≤ 6.727 ✓ | test_SP 3.6505 > 3.577 ✗ (7.35bp gap to SP floor)
**WSS gap to Transolver-3 target**: 6.6898% vs <5.850% — 84bp gap. Requires a fundamentally different approach or much stronger stacking.

**Recent directive (Morgan Issue #1056, 2026-05-29)**: Beat Transolver-3 SOTA test_WSS < 5.85% without degrading test_VP ≤ 3.421% or test_SP ≤ 3.577%.

---

## Active Fleet (Round 4k, as of 20:45Z)

| PR | Student | Hypothesis | Status | val | ETA |
|---|---|---|---|---|---|
| **#1455** | **edward** | **H275: EP15 + anti-thetic K=3 in full stack** | 🟡 just assigned | — | ~01:00Z |
| **#1448** | **alphonse** | **H269: K=10 noise in full H253 stacked recipe** | 🟢 val 5.9330 ✓ gate | **5.9330** | test ~23:00Z |
| **#1449** | **tanjiro** | **H270: σ=3e-4 in full H253 stacked recipe** | 🟢 test running | 5.9394 ❌ | ~21:30Z |
| **#1433** | **thorfinn** | **H257: σ=1e-4 in full H253 stacked recipe** | 🟢 test running | 5.9417 ❌ | ~21:30Z |
| **#1451** | **frieren** | **H271: Sobol QMC noise draws in full stack** | 🟢 running | — | ~22:00Z |
| **#1452** | **nezuko** | **H272: Hutchinson diag-H curvature-inverse σ** | 🟢 running (OOM fixed) | — | ~22:00Z |
| **#1453** | **askeladd** | **H273: Taylor 2nd-order correction λ-sweep** | 🟢 running | — | ~22:00Z |
| **#1454** | **fern** | **H274: Anti-thetic K=3 in full H253 stacked recipe (EP13)** | 🟢 running | — | ~22:00Z |

**Gate**: val_abupt < **5.9367** AND test_abupt < **5.7825**

**Sole active SOTA candidate**: alphonse H269 K=10 (val 5.9330 beats gate by 3.7bp). Awaiting test.
**Informative but gate-missed**: tanjiro H270 (val 5.9394 +2.7bp), thorfinn H257 (val 5.9417 +5.0bp) — bracket EP13-base σ=5e-4 as near-optimum.

---

## Strategic Focus (Round 4k)

### The compounding stack structure is now clear

Three validated additive findings:
1. **H243 6-res TTA**: −33.7bp over H185 EP13 single-res baseline
2. **H253 weight-noise K=5 stacked**: additional −3.4bp over H244 (sub-additive with input-space TTA)
3. **H267 EP15 checkpoint**: additional −5.1bp over H253 (sub-additive at ~55% of pure-additive −9.3bp)

**Finding QQ-EP15-stack-compound**: Each axis contributes sub-additively when stacked. But the stack still compounds — the full EP15+6-res+mirror+noise recipe gives val 5.9367.

### Research directions (ranked by EV)

**Tier 1 — EP15-base with orthogonal stacking variants (in-flight or just assigned)**
1. **H275 edward**: EP15 + anti-thetic K=3 stack (anti-thetic mechanism Finding NN, on EP15 base) — predicted 5.928-5.935
2. **H269 alphonse**: EP13 + K=10 stack (K-scaling, test pending) — if test confirms, informs H276 (EP15+K=10)

**Tier 2 — Novel mechanism probes (all EP13-base, in-flight)**
3. **H271 frieren**: Sobol QMC noise (QMC vs i.i.d., EP13)
4. **H272 nezuko**: Hutchinson curvature-inverse σ (EP13)
5. **H273 askeladd**: Taylor 2nd-order correction λ-sweep (EP13)
6. **H274 fern**: Anti-thetic K=3 stacked on EP13 (informs EP15 anti-thetic via comparison)

**Tier 3 — Characterization complete**
7. **H270 tanjiro**: σ-sweep (misses gate, completing for Finding RR banking)
8. **H257 thorfinn**: σ=1e-4 (misses gate, completing for Finding LL banking)

### Next hypotheses to generate (once current round concludes)

If H269 alphonse confirms K=10 wins on test:
- **H276**: EP15 + K=10 anti-thetic in full stack (stacks H267 + H269 + H268 mechanisms simultaneously)

If H271 frieren Sobol wins:
- **H277**: EP15 + Sobol K=5 in full stack

The SP floor is still 7.35bp away. Need new signal:
- Multi-resolution density re-weighting (bias toward lower res where SP patterns emerge?)
- SP-channel specific loss during eval (select σ to minimize SP specifically)?
- Or fundamentally: fine-tuning on val/test-like domains?

---

## Findings Bank (18 banked, 3 most recent)

| ID | Source | Summary |
|---|---|---|
| **QQ** | H267 edward (merged 20:29Z) | EP15 stacks with full H253 recipe at ~55% additive rate (−5.1bp gained) |
| **PP** | H266 fern (closed ~18:52Z) | 2³ ANOVA: Mirror×Noise interact −1.68bp; Res most orthogonal axis |
| **NN** | H268 askeladd (closed ~18:52Z) | Anti-thetic K=3 (6 passes) beats random K=5 (5 passes) by +1.3bp standalone |
| KK-noise-sat | H269 partial | K=5→K=10 stacked: +5.2bp (much larger than standalone +1.36bp) — K-saturation pushed by stacking |
| JJ | H253 | Weight-space+input-space TTA ~80% orthogonal; stacking yields residual +1.3bp |
| HH-H183 | H256 | H183 6-res stacked HURTS +70bp test over EP13 |
| LL-SWA-null | H263 | Adjacent EP SWA (EP14+EP15) regresses +6bp across all channels |
| LL-EPchain | H264 | EP13→EP14→EP15→EP16: EP15 cosine optimum, EP16 +3.9bp regression |
| (6 more banked in prior rounds) | | |

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Status |
|---|---|
| EP15 + K=5 + 6-res + mirror full stack | ✓ CURRENT SOTA (H267, val 5.9367) |
| EP13 + K=5 + 6-res + mirror full stack | ✓ SOTA baseline (H253, val 5.9418) |
| EP13 + K=10 + 6-res + mirror | 🟢 IN-FLIGHT alphonse H269 (val 5.9330 ✓, test pending) |
| EP15 + anti-thetic K=3 + 6-res + mirror | 🟢 IN-FLIGHT edward H275 (just assigned) |
| EP13 + anti-thetic K=3 + 6-res + mirror | 🟢 IN-FLIGHT fern H274 |
| EP15 + 6-res + mirror only (no noise) | H244 val 5.9452 — superseded by H267 |
| EP13 + Sobol QMC noise K=5 + 6-res + mirror | 🟢 IN-FLIGHT frieren H271 |
| EP13 + Hutchinson σ_i + 6-res + mirror | 🟢 IN-FLIGHT nezuko H272 |
| EP13 + Taylor 2nd-order correction | 🟢 IN-FLIGHT askeladd H273 |
| EP13 + σ=3e-4 stacked | H270 tanjiro val 5.9394 (missed gate) |
| EP13 + σ=1e-4 stacked | H257 thorfinn val 5.9417 (missed gate) |
| EP13 + σ=1e-3 stacked | H257 thorfinn val 5.9626 (worse than σ=5e-4) |
| SWA adjacent-EP | ✗ NULL — H263 +6bp regression |
| EP16 + any TTA | ✗ NULL — Finding LL-EPchain |
| H183 + 6-res stacked | ✗ HURTS — Finding HH-H183 |
| Surface multi-res | ✗ NULL — Finding KK |
| Vol-point jitter | ✗ FALSIFIED all scales |
| Frequency-weighted multi-res | ✗ MONOTONICALLY WORSE |
