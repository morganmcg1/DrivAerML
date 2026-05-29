# SENPAI Research State

**Updated**: 2026-05-29 11:35Z | Branch: `tay` | **NEW SOTA: H243 6-res TTA PR #1414** | Round 4i: 8 active | **alphonse H253 highest-EV (stacking arm)**

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| SOTA H185+TTA (PR #1382) | 5.9755% | 5.8221% | 6.7214% | 3.4400% | 3.6806% |
| H236 3-res mirror TTA (PR #1408) | 5.9613% | 5.8081% | 6.7130% | 3.4033% | 3.6759% |
| **H243 6-res mirror TTA (PR #1414)** | **5.9546%** | **5.7979%** | **6.7025%** | **3.3947%** | **3.6672%** |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Merge gate**: val_abupt < **5.9546%** AND test_abupt < **5.7979%**
**Paper floors**: test_VP 3.3947 ≤ 3.421 ✓ | test_WSS 6.7025 ≤ 6.727 ✓ | test_SP 3.6672 > 3.577 ✗

---

## Round 4i Active Fleet (as of 11:35Z)

| PR | Student | Hypothesis | Status | ETA |
|---|---|---|---|---|
| **#1413** | **tanjiro** | **H242 stacking arm: weight-noise+3-res (30 passes)** | 🟡 eval running (~150min) | ~12:00-12:30Z |
| **#1415** | **edward** | **H244: H185 EP14-16 retrain (TRAINING)** | 🟡 training | ~14:00Z |
| **#1428** | **alphonse** | **H253: weight-noise σ=5e-4 + H243 6-res (60 passes, HIGHEST EV)** | 🟢 running | ~12:30-13:00Z |
| **#1429** | **askeladd** | **H254: Surface-points multi-res (novel axis)** | 🟢 running | ~12:30-13:00Z |
| **#1431** | **fern** | **H255: 7-res {32k-164k} extension** | 🟢 running | ~12:30-13:00Z |
| **#1432** | **nezuko** | **H256: H183 + 6-res mirror + weight-noise stack** | 🟢 running | ~13:00-13:30Z |
| **#1433** | **thorfinn** | **H257: σ-sweep on H243 6-res stack {1e-4,2e-4,1e-3}** | 🟢 running | ~13:30-14:30Z |
| **#1434** | **frieren** | **H258: H148 EP13 + 6-res mirror multi-res TTA (flat-basin portability)** | 🟢 just assigned | ~12:00-12:30Z |

**Budget remaining**: ~4.2h (deadline ~15:45Z). edward H244 training uses ~2.7h more.

---

## Strategic Priorities (Round 4i)

### Tier 1 — Potential SOTA beats

1. **H253 alphonse** (HIGHEST EV): weight-noise σ=5e-4 × H243 6-res = 60 passes. Predicted val ~5.946/test ~5.791 if orthogonal. SOTA candidate.
2. **H258 frieren** (NEW, HIGH EV): H148 EP13 + 6-res mirror multi-res TTA. If Finding HH N=5, H148 base test 5.835 − 15bp = test ~5.73-5.80 → potential large SOTA push. If negative, confirms mirror-aug coupling (also valuable).
3. **H254 askeladd**: surface-points multi-res (novel axis). Unknown. Could add 3-7bp if positive.
4. **H255 fern**: 7-res {32k-164k}. Predicted +0.2-0.4bp. Potential tiny SOTA push.
5. **H257 thorfinn**: σ-sweep on 6-res stack. If σ=1e-3 still improves, new SOTA + basin flatness insight.

### Tier 2 — Informative / bank

6. **H242 tanjiro** (stacking on H236 3-res): predicted val ~5.953/test ~5.800. Probably misses H243 test gate (5.7979) by ~2bp; still validates stacking mechanism on the original 3-res recipe.
7. **H256 nezuko** (H183 full stack): confirms stack checkpoint-portability. Might hit test < 5.798 on H183.

### Tier 3 — Long horizon

8. **H244 edward** (~14:00Z): H185 EP14-16. If EP14/15 beats EP13 baseline: evaluate with H243-style 6-res TTA → potential big SOTA push. Highest-ceiling bet in the programme.

---

## Findings Banked This Round

| Finding | Source | Summary |
|---|---|---|
| LL | H249 fern | Tight-range multi-res WORSE than medium; wider is better (IID-proximity hypothesis falsified) |
| HH N=4 | H251 nezuko | Multi-res +12-15bp portable to H185/H183/H188 (N=4 confirmed) |
| FF generalized | H252 thorfinn | H148 flat basin confirmed; noise_only σ=5e-4 gives −39bp test (bigger than H185's −37bp) |
| DD-ext3 | H250 frieren | Frequency-weighted multi-res MONOTONICALLY WORSE; uniform blend is the global optimum |
| Test-floor convergence | H252 thorfinn | H148 weight-noise test 5.7978 ≈ H243 H185 multi-res test 5.7979 — approaching info ceiling |

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Result |
|---|---|
| Mirror-y | ✓ VALID — +5bp |
| Multi-res vol 3-res | ✓ VALID — +14bp (H236) |
| Multi-res vol 6-res | ✓ VALID — +21bp (H243 SOTA) |
| Multi-res vol 7-res | UNKNOWN — H255 fern testing |
| Surface-points multi-res | UNKNOWN — H254 askeladd testing |
| Weight-space noise σ=5e-4 | ✓ VALID — +8bp standalone |
| Noise + multi-res stacked (H185) | UNKNOWN — H253 alphonse testing (HIGHEST EV) |
| Noise + multi-res stacked (H183) | UNKNOWN — H256 nezuko testing |
| Noise + multi-res stacked (H242 3-res) | TESTING — tanjiro H242 arm |
| Multi-res portability H148 | UNKNOWN — H258 frieren testing |
| Tight-range multi-res | ✗ WORSE than medium (Finding LL) |
| Frequency-weighted multi-res | ✗ MONOTONICALLY WORSE (Finding DD-ext3, H250) |
| Per-channel/per-res α blending | ✗ ALL FALSIFIED (Findings DD, DD-ext, DD-ext2) |
| Point-position jitter | ✗ FALSIFIED (Finding EE) |
| Rotation θ≥0.1° | ✗ FALSIFIED |
| Coordinate scale ε=±2% | ✗ FALSIFIED |
| Mesh-subsample 80-95% | ✗ FALSIFIED |
| Gaussian input noise | ✗ FALSIFIED |
| Permutation | NULL |

---

## Test_WSS Gap Analysis

Test_WSS = 6.7025 vs target < 5.850 → gap **0.8525pp**. TTA alone cannot close this gap (ceiling ~10-15bp from all remaining stacking). The real WSS push will need architectural or loss changes from edward's H244 training path or a fundamentally new approach. The stacking experiments (H253/H256/H258) serve two purposes: (a) improve the overall gate metrics, (b) give fresh checkpoints to evaluate with improved training if H244 lands.
