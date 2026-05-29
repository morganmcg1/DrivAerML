# SENPAI Research State

**Updated**: 2026-05-29 17:00Z | Branch: `tay` | **SOTA: H244 EP15+6-res (PR #1415 merged 14:57Z)** | Round 4j: 8 active | **H253 val 5.9418 ✓; H268 askeladd done (antithetic marginal); H266 fern arms A/B done; H267 edward launched**

---

## Current SOTA

| Model | val_abupt | test_abupt | test_WSS | test_VP | test_SP |
|---|---:|---:|---:|---:|---:|
| SOTA H185+TTA (PR #1382) | 5.9755% | 5.8221% | 6.7214% | 3.4400% | 3.6806% |
| H236 3-res mirror TTA (PR #1408) | 5.9613% | 5.8081% | 6.7130% | 3.4033% | 3.6759% |
| H243 6-res mirror TTA (PR #1414) | 5.9546% | 5.7979% | 6.7025% | 3.3947% | 3.6672% |
| H252 stacked TTA: noise×3-res×mirror (PR #1413) | 5.9492% | 5.7975% | 6.7030% | 3.3996% | 3.6662% |
| **H244 EP15+6-res mirror TTA (PR #1415, merged 14:57Z)** | **5.9452%** | **5.7896%** | **6.6947%** | **3.3882%** | **3.6595%** |
| Transolver-3 target (Morgan) | — | — | **< 5.850%** | ≤ 3.643% | ≤ 3.577% |

**Merge gate**: val_abupt < **5.9452%** AND test_abupt < **5.7896%**
**Paper floors**: test_VP 3.3882 ≤ 3.421 ✓ | test_WSS 6.6947 ≤ 6.727 ✓ | test_SP 3.6595 > 3.577 ✗

---

## Active Fleet (Round 4j, as of 17:00Z)

| PR | Student | Hypothesis | Status | ETA |
|---|---|---|---|---|
| **#1428** | **alphonse** | **H253: EP13+full stack (val 5.9418 ✓)** | 🟢 test_stacked in progress | ~17:20Z |
| **#1432** | **nezuko** | **H256: H183 + 6-res + noise stack (val 5.9676 ❌ gate miss)** | 🟡 test_stacked in progress | ~17:50Z |
| **#1433** | **thorfinn** | **H257: σ-sweep on 6-res+noise stack** | 🟡 σ=1e-3 partial val 5.9626 worse, ping sent | TBD |
| **#1438** | **tanjiro** | **H262: K-saturation noise_only K=10 done (5.9709/5.8183) → K=20 running** | 🟢 K=10 partial | ~17:10Z |
| **#1439** | **frieren** | **H263: avg(EP14,EP15) sanity 6.0138 (+5.9bp vs EP15); 6-res mirror TTA running** | 🟢 6-res TTA in progress | ~17:30Z |
| **#1443** | **fern** | **H266: TTA ANOVA: Arm A val 5.975, Arm B val 5.960; Arm C running** | 🟢 Arm C (weight_noise_res_avg) | ~17:30Z |
| **#1442** | **askeladd** | **H268: Anti-thetic noise pairs — DONE, ping for SENPAI-RESULT** | 🟢 4 runs complete, ~1.5bp gain marginal | ping sent |
| **#1447** | **edward** | **H267: EP15 + full TTA stack (relaunch crashed Arm 2)** | 🟢 launched 16:31Z | ~18:30Z |
| ~~#1440~~ | ~~fern~~ | ~~H264: Finding LL-EPchain banked~~ | 🔴 closed | — |
| ~~#1441~~ | ~~edward~~ | ~~H265: EP14 + 6-res mirror gate miss; Finding LL-EPchain confirmed~~ | 🔴 closed | — |

---

## Strategic Focus (Round 4j)

### Critical insight after H244 merge

The EP15 checkpoint is the key differentiator. EP15 single-res val_orig 6.0079 (vs EP13 6.0172, −9.3bp) — this gain stacks additively with TTA. The research direction should now systematically explore:

1. **EP-extension chain**: Is EP16 > EP15? EP17? Or is EP15 the cosine floor?
2. **Multi-EP averaging**: Does avg(EP14,EP15) generalize better than either alone?
3. **Full stack on EP15**: EP15 + K=5 noise + 6-res + mirror (edward Arm 2, h1ae7x1j) — predicted val ~5.938-5.942
4. **K-optimum for noise**: Is K=5 the saturation point? (tanjiro H262, fast probe)

### Next SOTA candidates (ranked by EV)

1. **Alphonse H253** (`qytjlv97`): val_stacked **5.9418** ✓ clears val gate. Test arm pending ~17:20Z. **If test_stacked < 5.7896 → IMMEDIATE SOTA merge.**
2. **Edward H267** (PR #1447): EP15 + full stack (relaunch of crashed Arm 2). Predicted val ~5.935-5.940. ETA ~18:20Z.
3. **Frieren H263**: avg(EP14,EP15) + 6-res mirror. EP15 already in W&B; waiting on edward EP14 upload (~5 min).

### Cosine extension chain — COMPLETE (Finding LL-EPchain confirmed)

| EP | single-res val_orig 65k | Δ vs EP15 | Source |
|---|---:|---:|---|
| EP13 | 6.0172 | +9.3bp | baseline H185 |
| EP14 | 6.0169 | +9.0bp | H265 edward |
| **EP15** | **6.0079** | **(best)** | H244 SOTA basis |
| EP16 | 6.0118 | +3.9bp regression | H264 fern (training-time history) |

**Insight**: EP15 is a sharp single-epoch dip — not a smooth trend. EP14 is essentially flat vs EP13. The cosine schedule lands in a particular basin only at EP15, and EP16 escapes it. Has implications for paper narrative on cosine annealing sweet spots.

### Checkpoint accessibility (EP15 in W&B, EP14 pending edward upload)

- EP15 EMA = W&B artifact `model-edward-h244-h185-ep16-cosine-ext-0gjfv45i:v0` (alias `epoch-15`/`best`). Frieren can pull directly.
- EP14 — edward instructed to upload as part of H265 close (~5 min). Then frieren H263 unblocks.
- EP16 DEAD — Finding LL-EPchain. No upload needed.

---

## Findings Banked This Round (12 total)

| Finding | Source | Summary |
|---|---|---|
| LL | H249 fern | Tight-range multi-res WORSE; wider is better |
| LL-noise | H259 tanjiro | σ=5e-4 optimal; basin edge 5e-4→1e-3 |
| LL-extend | H255 fern | Resolution saturates at 6-res for H185 EP13 EMA |
| **LL-EPchain** | **H264 fern + H265 edward** | **EP-extension chain CONFIRMED: EP13(6.0172)→EP14(6.0169)→EP15(6.0079, deep dip)→EP16(6.0118). EP15 is a sharp single-epoch dip (not gradual trend). Cosine schedule lands in a particular basin only at EP15.** |
| HH N=4 | H251 nezuko | Multi-res +12-15bp portable H185/H183/H188 |
| **HH-H188** | **H261 askeladd** | **H188 EP13 not competitive with H185 EP13 (14-16bp worse absolute baseline)** |
| FF generalized | H252 thorfinn | H148 flat basin; noise_only σ=5e-4 gives −39bp test |
| DD-ext3 | H250 frieren | Frequency-weighted multi-res monotonically worse |
| Stacking orthogonality | H252 tanjiro | Weight-space + input-space TTA super-additive (+4bp excess) |
| GG-decomp | H258 frieren | H148 multi-res gain 14× smaller than H185 |
| KK | H254 askeladd | Surface multi-res null: surf→vol cross-attn coupling cancels all gain |
| EE-volume | H260 frieren | Vol-point jitter catastrophic all scales — closes point-position-jitter axis |

---

## Exhaustion Map — TTA Mechanisms

| Mechanism | Result |
|---|---|
| Mirror-y | ✓ VALID — +5bp |
| Multi-res vol 6-res {32k-131k} | ✓ VALID — +21bp (H243), OPTIMAL range |
| Multi-res vol 7-res / 8-res | ✗ MARGINAL (LL-extend): 7-res val regresses; 8-res test only |
| EP-extension (EP15) | ✓ VALID — −9.3bp single-res → additive with 6-res TTA |
| EP14 extension | ✗ FLAT (H265 edward): EP14 val_orig 6.0169 ≈ EP13 6.0172. No gain. EP15 dip not gradual. |
| EP16 extension | ✗ CLOSED (Finding LL-EPchain): EP16 val_orig 6.0118 > EP15 6.0079; regression +3.9bp. EP15 is sharp dip. |
| EP15 + full stack (EP15 × noise × 6-res × mirror) | UNKNOWN — H267 edward relaunching (Arm 2 crashed) |
| Multi-EP EMA averaging | UNKNOWN — H263 frieren testing |
| Weight-space noise σ=5e-4 K=5 | ✓ VALID — +8bp standalone |
| Noise + 3-res×mirror stacked | ✓ VALID — H252 +30bp super-additive |
| Noise + 6-res×mirror stacked | UNKNOWN — H253 alphonse testing (HIGHEST EV) |
| Noise + H183 6-res×mirror stacked | UNKNOWN — H256 nezuko testing |
| EP15 + 6-res×mirror+noise stacked | UNKNOWN — edward Arm 2 h1ae7x1j testing |
| K-noise saturation (K>5) | UNKNOWN — H262 tanjiro testing |
| Anti-thetic noise pairs ±δ | UNKNOWN — H268 askeladd testing |
| σ optimal for noise | ✓ CONFIRMED σ=5e-4 (Finding LL-noise) |
| Surface multi-res | ✗ NULL — Finding KK |
| Vol-point jitter | ✗ FALSIFIED all scales — Finding EE-volume |
| H188 family TTA | ✗ NOT VIABLE — H188 baseline 14-16bp worse than H185 |
| Tight-range multi-res | ✗ WORSE (Finding LL) |
| Frequency-weighted multi-res | ✗ MONOTONICALLY WORSE (Finding DD-ext3) |
| Per-channel/per-res α blending | ✗ ALL FALSIFIED |
| Point-position jitter (surface) | ✗ FALSIFIED (Finding EE) |
| Rotation, coord-scale, mesh-subsample, Gaussian noise, permutation | ✗ ALL FALSIFIED |
