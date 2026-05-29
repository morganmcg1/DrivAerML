# SENPAI Research State

**Updated**: 2026-05-29 18:15Z | Branch: `tay` | **SOTA: H253 EP13+K=5+6-res+mirror (PR #1428 merged 17:15Z)** | Round 4k: 8 active | **H267 edward EP15+full stack highest EV (~20:00Z); H272 nezuko Hutchinson highest EV new**

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

## Active Fleet (Round 4k, as of 18:00Z)

| PR | Student | Hypothesis | Status | ETA |
|---|---|---|---|---|
| **#1447** | **edward** | **H267: EP15 + K=5 full stack** — HIGHEST EV, predicted val ~5.932 | 🟢 running since 17:10Z (relaunch w/NCCL fix) | ~20:00Z |
| **#1448** | **alphonse** | **H269: K=10 noise stacked (EP13+6-res+mirror)** — predicted val ~5.937 | 🟢 launched 17:43Z, run `pkkr5g2u` | ~23:23Z |
| **#1449** | **tanjiro** | **H270: σ ∈ {3e-4, 2e-4} stacked** — tests if smaller σ optimal | 🟢 just assigned | ~22:00Z |
| **#1433** | **thorfinn** | **H257: σ=1e-4 stacked** — σ=2e-4 cancelled (tanjiro covers) | 🟢 σ=1e-4 running, run `b421o7n5` | ~20:33Z |
| **#1443** | **fern** | **H266: TTA ANOVA — Arms A+B done per W&B, Arm C running** | 🟡 heartbeat sent 17:58Z | ~18:30Z (?) |
| ~~#1439~~ | ~~frieren~~ | ~~H263: avg(EP14,EP15) SWA — Finding LL-SWA-null~~ | 🔴 closed 17:58Z | — |
| ~~#1432~~ | ~~nezuko~~ | ~~H256: H183 stack portability — Finding HH-H183-6res-asymmetry~~ | 🔴 closed 17:58Z | — |
| **#1451** | **frieren** | **H271: Sobol QMC weight perturbation** — low-discrepancy noise, attacks K-plateau | 🆕 assigned 18:15Z | ~21:15Z |
| **#1452** | **nezuko** | **H272: Hutchinson curvature-inverse noise** — per-param σ by diag Hessian, HIGHEST EV | 🆕 assigned 18:15Z | ~21:45Z |
| **#1453** | **askeladd** | **H273: Taylor 2nd-order correction** — λ-swept quadratic bias subtraction | 🆕 assigned 18:15Z | ~21:15Z Arm A |
| ~~#1442~~ | ~~askeladd~~ | ~~H268: Anti-thetic noise — Finding NN-antithetic confirmed~~ | 🔴 closed 17:58Z | — |
| ~~#1428~~ | ~~alphonse~~ | ~~H253: MERGED NEW SOTA val 5.9418 / test 5.7847~~ | 🏆 merged 17:15Z | — |

**Newly assigned (18:15Z)**: frieren H271 (PR #1451), nezuko H272 (PR #1452), askeladd H273 (PR #1453).

---

## Strategic Focus (Round 4k)

### After H253 merge — what we know

The H253 stack (EP13 + K=5 noise + 6-res + mirror = 60 passes) is the new SOTA. Four compounding experiments now in flight:

1. **EP15 full stack** (H267 edward): EP15 advantage (−9.3bp single-res) should carry through noise+res+mirror stack. Predicted val ~5.932-5.935. Highest EV.
2. **K=10 stacked** (H269 alphonse): K-saturation finding says K=5→10 adds +0.53bp in stacked context. Predicted val ~5.937.
3. **Smaller σ stacked** (H270 tanjiro + H257 thorfinn σ=1e-4): Smaller σ may be optimal in stacked context where 80% of variance already removed by multi-res. Coverage: {1e-4 thorfinn, 2e-4 tanjiro, 3e-4 tanjiro, 5e-4 alphonse-SOTA, 1e-3 thorfinn-already-worse}.
4. **TTA ANOVA decomposition** (H266 fern): main effect / interaction breakdown of mirror × res × noise — informs WHICH axis to push next round.

Composition scenario (if H267 confirms EP15 stacks + H269 confirms K=10 stacks): EP15 + K=10 + 6-res + mirror → val ~5.926 theoretical maximum in this TTA paradigm.

### Next SOTA candidates (ranked by EV)

1. **Edward H267** (PR #1447): EP15 + K=5 full stack → predicted val ~5.932-5.935 ← HIGHEST EV
2. **Alphonse H269** (PR #1448): K=10 noise stacked on EP13 → predicted val ~5.937
3. **Tanjiro H270** (PR #1449): σ=3e-4 stacked — possible gate hit if basin is narrow
4. **Thorfinn H257** σ=1e-4 (PR #1433): low probability — likely too small to perturb effectively

### New generation (assigned 18:15Z)

Three orthogonal TTA hypotheses attacking different dimensions of the noise mechanism:

1. **Sobol QMC** (H271 frieren #1451): Scrambled Sobol sequences replace i.i.d. Gaussian draws. Attacks K-plateau via sampling geometry, distinct from anti-thetic ±δ cancellation.
2. **Hutchinson curvature-inverse σ** (H272 nezuko #1452 — HIGHEST EV new): Per-parameter σ scaled by diagonal Hessian. First experiment using landscape geometry to guide TTA.
3. **Taylor 2nd-order correction** (H273 askeladd #1453): Anti-thetic residuals subtract quadratic Taylor bias. λ ∈ [0,1] tuned on val; λ=0 = plain anti-thetic floor. Askeladd's natural H268 follow-up.

### Closure-driven insights (3 new findings banked this round)

- **LL-SWA-null** (H263 closed): Adjacent (1-epoch-apart) late-cosine EMA SWA fails. Linear midpoint in performance space, no basin-width discount. Future SWA work must use materially different LR regions (e.g., EP6 + EP15).
- **HH-H183-6res-asymmetry** (H256 closed): H183 6-res TTA HURTS test (+70bp). Multi-res TTA portability is checkpoint-specific. H183 stops at 3-res for test gain; H185 benefits all the way to 6-res. Stack design must be re-validated per base model.
- **NN-antithetic confirmed** (H268 closed): Anti-thetic K=3 (6 passes) beats random K=5 (5 passes) by +1.3bp; K=3→K=5 anti shows fast diminishing returns. Variance reduction concentrates on highest-variance channels (tau_y).

### Cosine extension chain — COMPLETE (Finding LL-EPchain confirmed)

| EP | single-res val_orig 65k | Δ vs EP15 | Source |
|---|---:|---:|---|
| EP13 | 6.0172 | +9.3bp | baseline H185 |
| EP14 | 6.0169 | +9.0bp | H265 edward |
| **EP15** | **6.0079** | **(best)** | H244/H253 basis |
| EP16 | 6.0118 | +3.9bp regression | H264 fern (training-time history) |

**Insight**: EP15 is a sharp single-epoch dip. EP16 escapes the basin. SWA on EP14+EP15 also fails to find a wider basin (LL-SWA-null).

---

## Findings Banked This Round (17 total — 3 new this cycle)

| Finding | Source | Summary |
|---|---|---|
| LL | H249 fern | Tight-range multi-res WORSE; wider is better |
| LL-noise | H259 tanjiro | σ=5e-4 optimal standalone; basin edge 5e-4→1e-3 |
| LL-extend | H255 fern | Resolution saturates at 6-res for H185 EP13 EMA |
| LL-EPchain | H264 fern + H265 edward | EP13→EP14(flat)→EP15(sharp dip −9.3bp)→EP16(regression). EP15 is single-epoch basin. |
| **LL-SWA-null** | **H263 frieren** | **avg(EP14,EP15) regresses +6bp uniformly. Adjacent late-cosine SWA fails — no basin-width gain.** |
| HH N=4 | H251 nezuko | Multi-res +12-15bp portable H185/H183/H188 |
| HH-H188 | H261 askeladd | H188 EP13 not competitive with H185 EP13 (14-16bp worse) |
| **HH-H183-6res-asymmetry** | **H256 nezuko** | **H183 6-res HURTS test +70bp. Multi-res portability is checkpoint-specific; stack must be re-validated per base.** |
| FF generalized | H252 thorfinn | H148 flat basin; noise_only σ=5e-4 gives −39bp test |
| DD-ext3 | H250 frieren | Frequency-weighted multi-res monotonically worse |
| Stacking orthogonality | H252 tanjiro | Weight-space + input-space TTA super-additive (+4bp excess) |
| GG-decomp | H258 frieren | H148 multi-res gain 14× smaller than H185 |
| KK | H254 askeladd | Surface multi-res null: surf→vol cross-attn coupling cancels all gain |
| EE-volume | H260 frieren | Vol-point jitter catastrophic all scales |
| JJ | H253 alphonse | Weight-space + input-space TTA partially orthogonal: noise ~39% as effective in stacked context (~80% variance already captured by multi-res) |
| KK-noise-saturation | H262 tanjiro | K≈10 variance-reduction plateau: K=5→10 = +1.36bp; K=10→20 = only +0.42bp. Standardize on K=10. |
| **NN-antithetic** | **H268 askeladd** | **Anti-thetic K=3 (6 passes) beats random K=5 (5 passes) by +1.3bp. Linear gradient cancelled. Diminishing returns at K=5 anti.** |

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Result |
|---|---|
| Mirror-y | ✓ VALID — +5bp |
| Multi-res vol 6-res {32k-131k} | ✓ VALID — +21bp (H243), OPTIMAL range for H185 |
| Multi-res vol 7-res / 8-res | ✗ MARGINAL (LL-extend): 7-res val regresses; 8-res test only |
| Multi-res TTA on H183 | ✗ 3-res helps val/test, **6-res HARMS test** (HH-H183-6res-asymmetry) |
| EP-extension (EP15) | ✓ VALID — −9.3bp single-res → additive with 6-res TTA |
| EP14 extension | ✗ FLAT (H265 edward): EP14 val_orig 6.0169 ≈ EP13 6.0172 |
| EP16 extension | ✗ CLOSED (Finding LL-EPchain): EP16 val_orig 6.0118 > EP15 6.0079 |
| **EP15 + full stack (EP15 × noise × 6-res × mirror)** | **🟢 UNKNOWN — H267 edward running (~20:00Z)** |
| Adjacent multi-EP SWA (avg EP14, EP15) | ✗ FALSIFIED (LL-SWA-null): regresses +6bp uniformly. No basin-width gain. |
| Weight-space noise σ=5e-4 K=5 | ✓ VALID — +8bp standalone |
| Noise + 3-res×mirror stacked | ✓ VALID — H252 +30bp super-additive |
| **Noise + 6-res×mirror stacked (K=5)** | **✓ VALID — H253 +1.28bp; partial orthogonality (Finding JJ)** |
| Noise + H183 6-res×mirror stacked | ✗ FALSIFIED (H256): val −85bp but test +60bp due to H183 6-res asymmetry |
| K=10 noise stacked | 🟢 UNKNOWN — H269 alphonse testing (~23:23Z) |
| Anti-thetic K=3 noise stacked | UNKNOWN — separate experiment pending |
| σ ∈ {1e-4, 2e-4, 3e-4} in stacked context | 🟢 UNKNOWN — H270 tanjiro + H257 thorfinn σ=1e-4 running |
| σ=1e-3 in stacked context | ✗ FALSIFIED — H257 thorfinn val 5.9626 (worse than σ=5e-4 by 21bp) |
| K-noise saturation (K>5) | ✓ K≈10 optimum (Finding KK-noise-saturation): K=5→10 +1.36bp, K=10→20 only +0.42bp |
| Anti-thetic noise pairs ±δ (standalone) | ✓ VALID: anti-K=3 beats random K=5 by +1.3bp (Finding NN-antithetic) |
| σ optimal for noise (standalone) | ✓ CONFIRMED σ=5e-4 (Finding LL-noise) |
| Surface multi-res | ✗ NULL — Finding KK |
| Vol-point jitter | ✗ FALSIFIED all scales — Finding EE-volume |
| H188 family TTA | ✗ NOT VIABLE — H188 baseline 14-16bp worse than H185 |
| H183 family TTA (deep extension) | ✗ TEST-HARMFUL — Finding HH-H183-6res-asymmetry |
| Tight-range multi-res | ✗ WORSE (Finding LL) |
| Frequency-weighted multi-res | ✗ MONOTONICALLY WORSE (Finding DD-ext3) |
| Per-channel/per-res α blending | ✗ ALL FALSIFIED |
| Point-position jitter (surface) | ✗ FALSIFIED (Finding EE) |
| Rotation, coord-scale, mesh-subsample, Gaussian noise, permutation | ✗ ALL FALSIFIED |
