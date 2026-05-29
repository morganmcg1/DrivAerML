# SENPAI Research State

**Updated**: 2026-05-29 17:30Z | Branch: `tay` | **SOTA: H253 EP13+K=5+6-res+mirror (PR #1428 merged 17:15Z)** | Round 4k: 8 active | **H267 edward EP15+full stack highest EV; H269/H270 assigned**

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| SOTA H185+TTA (PR #1382) | 5.9755% | 5.8221% | 6.7214% | 3.4400% | 3.6806% |
| H243 6-res mirror TTA (PR #1414) | 5.9546% | 5.7979% | 6.7025% | 3.3947% | 3.6672% |
| H244 EP15+6-res mirror TTA (PR #1415) | 5.9452% | 5.7896% | 6.6947% | 3.3882% | 3.6595% |
| **H253 EP13+K=5+6-res+mirror stack (PR #1428, merged 17:15Z)** | **5.9418%** | **5.7847%** | **6.6895%** | **3.3891%** | **3.6592%** |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Merge gate**: val_abupt < **5.9418%** AND test_abupt < **5.7847%**
**Paper floors**: test_VP 3.3891 ≤ 3.421 ✓ | test_WSS 6.6895 ≤ 6.727 ✓ | test_SP 3.6592 > 3.577 ✗

---

## Active Fleet (Round 4k, as of 17:30Z)

| PR | Student | Hypothesis | Status | ETA |
|---|---|---|---|---|
| **#1447** | **edward** | **H267: EP15 + K=5 full stack** — HIGHEST EV, predicted val ~5.932 | 🟢 running since 16:31Z | ~18:30Z |
| **#1448** | **alphonse** | **H269: K=10 noise stacked (EP13+6-res+mirror)** — predicted val ~5.937 | 🆕 just assigned | ~21:00Z |
| **#1442** | **askeladd** | **H268→H269: Anti-thetic K=3 stacked (EP13+6-res+mirror)** — predicted val ~5.937 | 🟡 sent back, adding new mode | ~19:30Z |
| **#1449** | **tanjiro** | **H270: σ-sweep {3e-4, 2e-4} on H253 stacked recipe** | 🆕 just assigned | ~20:00Z |
| **#1443** | **fern** | **H266: TTA ANOVA Arm C running (weight_noise_res_avg)** — Arms A(5.975) + B(5.960) done | 🟢 Arm C | ~18:00Z |
| **#1439** | **frieren** | **H263: avg(EP14,EP15) + 6-res mirror TTA** — sanity 6.0138 (+5.9bp above EP15) | 🟢 6-res TTA running | ~18:00Z |
| **#1432** | **nezuko** | **H256: H183 stack portability** — val gate miss 5.9676, test pending | 🟢 test_stacked | ~18:00Z |
| **#1433** | **thorfinn** | **H257: σ=1e-3 worse (partial 5.9626)** — ping sent 16:53Z | 🟡 awaiting response | TBD |
| ~~#1428~~ | ~~alphonse~~ | ~~H253: MERGED NEW SOTA val 5.9418 / test 5.7847~~ | 🏆 merged 17:15Z | — |
| ~~#1438~~ | ~~tanjiro~~ | ~~H262: CLOSED Finding KK-noise-saturation~~ | 🔴 closed | — |

---

## Strategic Focus (Round 4k)

### After H253 merge — what we know

The H253 stack (EP13 + K=5 noise + 6-res + mirror = 60 passes) is the new SOTA. Three compounding experiments now running/assigned:

1. **EP15 full stack** (H267 edward): EP15 advantage (−9.3bp single-res) should carry through noise+res+mirror stack. Predicted val ~5.932-5.935. Highest EV.
2. **K=10 stacked** (H269 alphonse): K-saturation finding says K=5→10 adds +0.53bp in stacked context. Predicted val ~5.937.
3. **Anti-thetic stacked** (H268→H269 askeladd): Anti-thetic K=3 beats random K=5 in isolation; testing if advantage transfers to stacked context. Predicted val ~5.937.
4. **Smaller σ stacked** (H270 tanjiro): σ=5e-4 confirmed for standalone; smaller σ may be optimal in stacked context where 80% of variance already removed by multi-res.

Composition scenario (if H267 confirms EP15 stacks + K=10 stacks): EP15 + K=10 + 6-res + mirror → val ~5.926 theoretical maximum in this TTA paradigm.

### Next SOTA candidates (ranked by EV)

1. **Edward H267** (PR #1447): EP15 + K=5 full stack → predicted val ~5.932-5.935 ← HIGHEST EV, running
2. **Alphonse H269** (PR #1448): K=10 noise stacked on EP13 → predicted val ~5.937
3. **Askeladd H268→H269** (PR #1442): Anti-thetic K=3 stacked on EP13 → predicted val ~5.937
4. **Tanjiro H270** (PR #1449): Smaller σ sweep → possible gate hit at σ=3e-4

### Cosine extension chain — COMPLETE (Finding LL-EPchain confirmed)

| EP | single-res val_orig 65k | Δ vs EP15 | Source |
|---|---:|---:|---|
| EP13 | 6.0172 | +9.3bp | baseline H185 |
| EP14 | 6.0169 | +9.0bp | H265 edward |
| **EP15** | **6.0079** | **(best)** | H244/H253 basis |
| EP16 | 6.0118 | +3.9bp regression | H264 fern (training-time history) |

**Insight**: EP15 is a sharp single-epoch dip. EP16 escapes the basin.

---

## Findings Banked This Round (14 total)

| Finding | Source | Summary |
|---|---|---|
| LL | H249 fern | Tight-range multi-res WORSE; wider is better |
| LL-noise | H259 tanjiro | σ=5e-4 optimal; basin edge 5e-4→1e-3 |
| LL-extend | H255 fern | Resolution saturates at 6-res for H185 EP13 EMA |
| LL-EPchain | H264 fern + H265 edward | EP13→EP14(flat)→EP15(sharp dip −9.3bp)→EP16(regression). EP15 is single-epoch basin. |
| HH N=4 | H251 nezuko | Multi-res +12-15bp portable H185/H183/H188 |
| HH-H188 | H261 askeladd | H188 EP13 not competitive with H185 EP13 (14-16bp worse) |
| FF generalized | H252 thorfinn | H148 flat basin; noise_only σ=5e-4 gives −39bp test |
| DD-ext3 | H250 frieren | Frequency-weighted multi-res monotonically worse |
| Stacking orthogonality | H252 tanjiro | Weight-space + input-space TTA super-additive (+4bp excess) |
| GG-decomp | H258 frieren | H148 multi-res gain 14× smaller than H185 |
| KK | H254 askeladd | Surface multi-res null: surf→vol cross-attn coupling cancels all gain |
| EE-volume | H260 frieren | Vol-point jitter catastrophic all scales |
| **JJ** | **H253 alphonse** | **Weight-space + input-space TTA partially orthogonal: noise ~39% as effective in stacked context (~80% variance already captured by multi-res)** |
| **KK-noise-saturation** | **H262 tanjiro** | **K≈10 variance-reduction plateau: K=5→10 = +1.36bp; K=10→20 = only +0.42bp. Standardize on K=10.** |
| **NN-antithetic** | **H268 askeladd** | **Anti-thetic K=3 (6 passes) beats random K=5 (5 passes) by +1.3bp. Linear gradient cancelled. K=3→5 fast diminishing returns.** |

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Result |
|---|---|
| Mirror-y | ✓ VALID — +5bp |
| Multi-res vol 6-res {32k-131k} | ✓ VALID — +21bp (H243), OPTIMAL range |
| Multi-res vol 7-res / 8-res | ✗ MARGINAL (LL-extend): 7-res val regresses; 8-res test only |
| EP-extension (EP15) | ✓ VALID — −9.3bp single-res → additive with 6-res TTA |
| EP14 extension | ✗ FLAT (H265 edward): EP14 val_orig 6.0169 ≈ EP13 6.0172 |
| EP16 extension | ✗ CLOSED (Finding LL-EPchain): EP16 val_orig 6.0118 > EP15 6.0079 |
| EP15 + full stack (EP15 × noise × 6-res × mirror) | UNKNOWN — H267 edward running (~18:30Z) |
| Multi-EP EMA averaging | UNKNOWN — H263 frieren testing |
| Weight-space noise σ=5e-4 K=5 | ✓ VALID — +8bp standalone |
| Noise + 3-res×mirror stacked | ✓ VALID — H252 +30bp super-additive |
| **Noise + 6-res×mirror stacked (K=5)** | **✓ VALID — H253 +1.28bp; partial orthogonality (Finding JJ)** |
| Noise + H183 6-res×mirror stacked | UNKNOWN — H256 nezuko testing (val gate miss 5.9676; test pending) |
| K=10 noise stacked | UNKNOWN — H269 alphonse testing |
| Anti-thetic K=3 noise stacked | UNKNOWN — H268→H269 askeladd testing |
| σ below 5e-4 in stacked context | UNKNOWN — H270 tanjiro testing |
| K-noise saturation (K>5) | ✓ K≈10 optimum (Finding KK-noise-saturation): K=5→10 +1.36bp, K=10→20 only +0.42bp |
| Anti-thetic noise pairs ±δ (standalone) | ✓ VALID: anti-K=3 beats random K=5 by +1.3bp (Finding NN-antithetic) |
| σ optimal for noise (standalone) | ✓ CONFIRMED σ=5e-4 (Finding LL-noise) |
| Surface multi-res | ✗ NULL — Finding KK |
| Vol-point jitter | ✗ FALSIFIED all scales — Finding EE-volume |
| H188 family TTA | ✗ NOT VIABLE — H188 baseline 14-16bp worse than H185 |
| Tight-range multi-res | ✗ WORSE (Finding LL) |
| Frequency-weighted multi-res | ✗ MONOTONICALLY WORSE (Finding DD-ext3) |
| Per-channel/per-res α blending | ✗ ALL FALSIFIED |
| Point-position jitter (surface) | ✗ FALSIFIED (Finding EE) |
| Rotation, coord-scale, mesh-subsample, Gaussian noise, permutation | ✗ ALL FALSIFIED |
