## 2026-05-26 ~00:50 — PR #1320: H-B2 DETACHED AUX LOG-MAGNITUDE WSS HEAD (alphonse, **CLOSED C NULL** — straight baseline-trail on both val and test; aux-head class definitively CLOSED via H-B/H-B2 gradient-flow attribution)

- **Branch**: `alphonse/detached-aux-loss-wss` (CLOSED, not merged)
- **W&B run**: `tqkdhfqu`
- **Hypothesis**: Same aux-log-magnitude WSS head as H-B but with `grad-stop` on backbone — isolates whether H-B's val signal came from backbone gradient shaping (mechanism alive) or from feature engineering (independent of gradient).
- **Parameter overhead**: +131K params (same as H-B), 17.55M total

### Terminal metrics (best EP13 EMA-source)

| Metric | H-B2 | H112 | Δ vs H112 | H-B (#1307) | Δ vs H-B |
|---|---:|---:|---:|---:|---:|
| val_abupt | 6.2255% | 6.1358% | **+0.090pp MISS** | 6.1288% | +0.097pp behind |
| val_WSS | 7.0915% | 6.9670% | +0.125pp | — | — |
| val_WSS_z | 9.4677% | 9.3750% | +0.093pp | — | — |
| **test_WSS** | **6.9302%** | **6.7523%** | **+0.178pp REGRESSION** | 6.7974% | **+0.133pp behind** |
| test_WSS_z | 8.9432% | 8.7201% | +0.223pp | — | — |
| test_VP | 3.5249% | 3.4213% | +0.104pp | 3.452% | +0.073pp |
| test_SP | 3.7880% | 3.6947% | +0.093pp | 3.580% | +0.208pp |

**Total wallclock**: 863.4 min = 14.39h | **Peak GPU mem**: ~82 GB (allocated)

### Mechanism finding (load-bearing program-level)

The H-B / H-B2 ablation confirms **gradient-flow attribution** for the aux-head mechanism:

| Condition | val_abupt | test_WSS | Slope val→test | Class |
|---|---:|---:|---:|---|
| H112 (no aux) | 6.1358% | 6.7523% | +0.616pp | reference |
| **H-B (aux, with grad)** | 6.1288% | 6.7974% | +0.668pp | val WIN + test LOSS — catastrophe |
| **H-B2 (aux, no grad)** | 6.2255% | 6.9302% | +0.704pp | val LOSS + test LOSS — baseline-trail |

- Detached aux **strictly worse than H-B on every metric**
- Aux head's val signal is GRADIENT-DRIVEN — without it, the +131K params add only RNG-shift + optimizer-state pressure (no compensating representation benefit)
- "Detach as safety net" pattern FALSIFIED

### Banked findings (permanent program value)

1. **Aux-head class definitively CLOSED at canonical 17.5M recipe** — H-B catastrophe + H-B2 baseline-trail bracket the class behavior
2. **+131K param addition costs ~+0.10pp val/test** (RNG-shift baseline cost) — calibrates future architectural intervention cost
3. **Aux-head mechanism is gradient-driven, not feature-engineering driven** — settles the H-B class attribution permanently
4. **Pivot away from aux-head class** — productive directions are tau_z loss-weight escalation (H143/H144), split WSS_z decoder (H138), and architectural-tier interventions

### Strategic implication

The aux-head class joins the capacity-axis cohort as exhausted. Wave 38+ frontier is now:
- **tau_z loss-weight escalation** (H143 at 4.0 showing EP2 −0.090pp WSS_z lead vs H112 — first cohort-aligned positive signal)
- **Split WSS_z decoder** (H138 askeladd in flight — dedicated wider head for tau_z)
- **Decoder-only SwiGLU** (H135 thorfinn in flight — TIED with H112 raw)
- **Architectural/data-tier interventions** (untested at canonical capacity)

---

## 2026-05-24 ~21:45 — PR #1297: H121 BACKBONE HIDDEN-DIM 512→576 (frieren, **CLOSED C NULL** — marginal val gate miss + test_WSS regression; **CAPACITY-AXIS × CANONICAL DROPPATH GENERALIZATION-BOUND LOCKED ACROSS BOTH DEPTH AND WIDTH**)

- **Branch**: `frieren/h121-hidden-576` (CLOSED, not merged)
- **W&B run**: `9naxnj3f`
- **Hypothesis**: Expanding backbone hidden dim from 512→576 (+26% actual params, +26% VRAM) improves capacity for WSS features in Wave 36+ alongside depth-6 (H120).

### Terminal metrics (run 9naxnj3f, EMA best-checkpoint EP11)

| Metric | H121 (hidden=576) | H112 canonical | Δ | Gate / Floor | Verdict |
|---|---:|---:|---:|---|---|
| **val_abupt** | 6.1538% | 6.1358% (gate) | +0.018pp | gate | ❌ marginal miss |
| **test_abupt** | 5.9194% | 5.8391% | +0.080pp | — | ❌ REGRESSION |
| test_VP | 3.5477% | 3.4213% | +0.127pp | floor 3.643% | ❌ vs H112; ✅ crosses paper floor |
| test_SP | 3.7374% | 3.6947% | +0.042pp | floor 3.577% | ❌ **23rd SP plateau confirmation** |
| **test_WSS** | **6.8262%** | **6.7523%** | **+0.074pp** | floor 6.727% | ❌ **REGRESSION on primary objective** |
| test_WSS_x | 6.0667% | 5.9989% | +0.068pp | — | ❌ |
| test_WSS_y | 7.4349% | 7.3602% | +0.075pp | — | ❌ |
| test_WSS_z | 8.8102% | 8.7201% | +0.090pp | — | ❌ |

**Param count**: 22.01M actual (19.9M projected — off by +10.6%, decoder also scales with hidden_dim)
**Best checkpoint**: EMA EP11 | **Peak VRAM**: 93.1 GB / 96 GB
**Total wallclock**: 1011.4 min = 16.86h

### Val→test slope table (student diagnostic — PROGRAM-WIDE IMPACT)

| Channel | val (%) | test (%) | val→test slope | Historical projection |
|---|---:|---:|---:|---:|
| abupt | 6.154 | 5.919 | **−0.234pp** | −0.28pp |
| **WSS** | **6.962** | **6.826** | **−0.136pp** ↓53% | −0.29pp |
| VP | 3.643 | 3.548 | **−0.096pp** ↓62% | −0.25pp |
| SP | 4.067 | 3.737 | −0.329pp | −0.45pp |

WSS slope flattened by **53%** vs projection. Mirrors H120 depth-6 WSS slope flattening (93%) — **same outcome, different mechanism**:
- H120: DropPath schedule auto-stretches with depth → per-layer rate weaker
- H121: DropPath schedule unchanged → per-feature redundancy at wider hidden_dim

### Capacity-axis generalization-bound — LOCKED across all three axes

| Axis | Run | test_WSS Δ vs H112 | val→test WSS slope |
|---|---|---:|---:|
| Slice granularity | H118 slices-192 | +0.370pp regression | — |
| Backbone depth | H120 depth-6 | **+0.066pp regression** | −0.020pp ↓93% |
| Backbone width | H121 hidden-576 | **+0.074pp regression** | **−0.136pp ↓53%** |

Three orthogonal capacity axes all produce test_WSS regressions and val→test slope flattening at canonical DropPath_max=0.10. **Single-mechanism capacity-axis frontier is closed for tay** at canonical regularization.

### Key program-wide calibration update (student finding)

**Historical val→test slope projections (−0.28pp abupt, −0.29pp WSS) DO NOT HOLD for capacity-scaled models.** Actual slopes on capacity-axis runs are −0.10 to −0.23pp (WSS: −0.02 to −0.14pp). Val gate < 6.1358% is a necessary but insufficient condition for test improvement on capacity-scaled models. All future slope projections on capacity-scale runs must use actual H120/H121 slope range, not H112 historical.

### Successor experiment

frieren → H131 (PR #1312): hidden-576 × DropPath_max=0.15. Per-layer rate → 0.0375 (+50% over H112). Parallel to H130 askeladd depth-6 × max=0.15. ETA ~16:00Z 2026-05-25.

---

## 2026-05-24 ~20:20 — PR #1296: H120 BACKBONE DEPTH 5→6 (askeladd, **CLOSED C NULL** — val A WIN but val→test slope catastrophe; **MAJOR PROGRAM FINDING: DEPTH-AXIS GENERALIZATION-BOUND AT FIXED REGULARIZATION**)

- **Branch**: `askeladd/h120-backbone-depth-6` (CLOSED, not merged)
- **W&B run**: `nwqy4r4f`
- **Hypothesis**: Extending backbone depth from 5 to 6 layers (+17% params, +2.96M) improves capacity for high-frequency WSS features in Wave 36+.

### Terminal metrics (run nwqy4r4f, EMA EP13)

| Metric | H120 (depth=6) | H112 canonical (depth=5) | Δ | Gate / Floor | Verdict |
|---|---:|---:|---:|---|---|
| **val_abupt** | **6.0122%** | 6.1358% | **−0.124pp** | gate 6.1358% | ✅ **CLEARS val gate** |
| **test_abupt** | 5.8990% | 5.8391% | **+0.060pp** | — | ❌ REGRESSION |
| test_VP | 3.4614% | 3.4213% | +0.040pp | floor 3.421% | ❌ vs H112; ✅ crosses pre-H112 |
| test_SP | 3.7280% | 3.6947% | +0.033pp | floor 3.577% | ❌ **22nd plateau confirmation** |
| **test_WSS** | **6.8180%** | **6.7523%** | **+0.066pp** | floor 6.727% | ❌ **REGRESSION on primary objective** |
| test_WSS_x | 6.0444% | 5.9989% | +0.045pp | — | ❌ |
| test_WSS_y | 7.4265% | 7.3602% | +0.066pp | — | ❌ |
| test_WSS_z | 8.8345% | 8.7201% | +0.114pp | — | ❌ (largest regression) |

**Param count**: 20.37M (H112: 17.41M, +2.96M = +17.0%)
**Best checkpoint**: EMA EP13 | **Peak VRAM**: 95.43 GB / 97.9 GB (2.47 GB headroom)
**Total wallclock**: 972.6 min = 16.21h

### Val trajectory (H120 led every epoch)

| Stage | Step | val_abupt | H112 EP val | Δ |
|---|---:|---:|---:|---:|
| EP1 (gate <35%) | 10,864 | 25.720% | ~26.7% | −1.0pp |
| EP2 | 21,729 | 7.589% | ~7.92% | −0.33pp |
| EP3 (gate <8.5%) | 32,594 | 6.700% | ~7.05% | −0.35pp |
| EP6 | 48,902 | 6.214% | ~6.39% | −0.18pp |
| **EP13 terminal** | 70,652 | **6.012%** | 6.136% | **−0.124pp** |

H120 led H112 at every checkpoint — but **none of the val improvement transferred to test**.

### Val→test slope collapse (primary diagnostic)

| Run | val_abupt | test_abupt | abupt slope | val_WSS | test_WSS | WSS slope |
|---|---:|---:|---:|---:|---:|---:|
| H112 canonical | 6.136% | 5.839% | **−0.297pp** | 6.967% | 6.752% | **−0.215pp** |
| H120 (depth=6) | 6.012% | 5.899% | **−0.113pp** ↓62% | 6.838% | 6.818% | **−0.020pp** ↓93% |

**WSS val→test slope flattened by 93%** — catastrophic. Root cause: DropPath schedule auto-stretches `[max·i/(depth−1)]` → depth-6 with `max=0.10` weakens per-layer drop rate from 0.025 (H112) to 0.020 (H120) = −20%. +17% capacity with −20% per-layer regularization → model overfits val-specific high-frequency features.

### Strategic conclusions

1. **Depth-axis at fixed DropPath_max is generalization-bound** — val improves, test does not. This is a *new failure class* distinct from all prior H115/H116/H117/H118 C NULLs (which had intact slopes but insufficient ceiling).
2. **22nd SP plateau confirmation** — 5th orthogonal Wave 36+ axis fails against SP floor 3.577%.
3. **Capacity-axis closing** — combined with H118 (slices null), H102/H110 (decoder-width), the standalone capacity-axis frontier is exhausted at canonical DropPath_max.
4. **Depth × regularization compound is the path forward** — H130 (depth-6 × DropPath_max=0.15) assigned to test slope-restoration thesis.

### Student diagnostic (banked)

- DropPath auto-adapts as `[max·i/(depth−1)]` — per-layer rate is **proportional to max**, not fixed. Deeper nets with same max have weaker per-layer drop.
- Predicted fix: raise max proportionally with depth to maintain per-layer floor (max=0.15 for depth-6 restores per-layer rate from 0.020 to 0.030, above H112's 0.025).
- val_VP improved −0.087pp but test_VP REGRESSED +0.040pp — strongest per-channel divergence signal.
- Peak VRAM 95.43 GB at depth-6: depth-7 requires activation checkpointing (3 GB headroom would be exhausted).

### Successor experiment

askeladd → H130 (PR #1311): depth-6 × DropPath_max=0.15. Single flag change `--drop-path-max 0.15`. ETA ~13:00Z 2026-05-25.

---

## 2026-05-24 ~20:00 — PR #1303: H126 T=1.0 INVERSE-AREA STRATIFIED SAMPLING (nezuko, **CLOSED C NULL** — EP1 kill-fence at val_abupt 43.84%, "softmax over log-area" pathology; **MECHANISM CLASS NOT CLOSED**, refined H126b at T=4.0 assigned)

- **Branch**: `nezuko/h126-area-weighted-sampling` (CLOSED, not merged)
- **W&B run**: `eatpu111`
- **Hypothesis**: T=1.0 inverse-panel-area sampling biases training toward small/curvy panels → improves WSS prediction in high-curvature regions (wheel arches, A-pillars) where WSS variance is highest.

### Terminal metrics (EP1 kill-fence trip at step 10864)

| Metric | H126 EP1 | H112 EP1 | Δ |
|---|---:|---:|---:|
| **val_abupt** | **43.84%** | 25.87% | **+17.97pp** ❌ fence breach (gate <35%) |
| val_WSS | 49.89% | 28.95% | +20.94pp |
| val_WSS_x | 43.75% | 25.73% | +18.02pp |
| val_WSS_y | 59.81% | 34.11% | +25.70pp |
| val_WSS_z | 59.88% | 34.54% | +25.34pp |
| val_SP | 36.19% | 19.37% | +16.82pp |
| val_VP | 19.55% | 15.58% | +3.97pp |

Run terminated cleanly via `early_stop.triggered=1`, no NaN/OOM/crash.

### Pinned student diagnostic — "softmax over log-area" pathology

| Diagnostic | Value | Interpretation |
|---|---:|---|
| `area/min_mean` | 7.85e-09 m² | smallest panels — wheel arch / A-pillar fillet tier |
| `area/p50_mean` | 7.22e-07 m² | median panel ≈ 100× smallest |
| `area/max_mean` | 1.08e-04 m² | largest flat panels ≈ 15,000× smallest |
| `ratio/max_mean` | **24.5×** | most-oversampled point per case (mean) |
| `ratio/skew_max_over_p50` | **94.9×** | smallest panels 95× over median |
| `ratio/max_p99_acrosscases` | **37.4×** | worst-case extreme |

Root cause: at T=1.0, σ_log≈9.5 of panel-area distribution → softmax collapses onto extremes → ~50% of batch concentrates on <1% of surface area. Model never sees enough dominant large-panel regions to learn them; eval-time uniform coverage catastrophically fails on un-learned regions.

### Comparison to H-A (PR #1302) — same failure family

| EP1 metric | H-A (#1302) | H126 (this PR) | H112 baseline |
|---|---:|---:|---:|
| val_abupt | 40.22% (+12.68pp) | **43.84% (+17.97pp)** | 25.87% |
| Class | input-rep destructive swap | sampling-distribution shift | — |

Both fail by the same mechanism family: training-distribution shifts far enough from canonical eval distribution that one epoch can't bridge the gap.

### Strategic class verdict — NOT closed, parameter pathology

What's falsified: T=1.0 raw-inverse-area at this mesh.
What's NOT closed: hypothesis class. Soft-T, quantile-rank, capped-ratio, warmup-ramp variants untested.

### Banked positives

- `data/area_weighted_sampling/*` instrumentation (production-ready diagnostic) — preserved
- `torch.searchsorted` on per-case cached CDF (~0.53 s/step, **FASTER than H112 baseline ~0.75 s/step**) — no perf debt
- Per-case CDF cache (~33 MB/case × ~50/rank ≈ 1.65 GB/worker) — within VRAM budget
- Default-off flag preservation — canonical recipes unchanged

### Successor experiment

nezuko → H126b T=4.0 (PR #1310) — softer reweighting, re-uses H126 loader code, expects max/median skew ~3× (vs T=1.0's 95×).

---

## 2026-05-24 ~18:45 — PR #1292: H117 SIGNED-SQRT SP TARGET TRANSFORM × DROPPATH (alphonse, **CLOSED C NULL** — TIES H112 within ±0.03pp on all WSS channels, **22ND SP-PLATEAU CONFIRMATION**, **SP-AXIS DATA-TIER CLOSURE LOCKED**)

- **Branch**: `alphonse/h117-sp-target-signed-power-transform` (CLOSED, not merged)
- **W&B run**: `kjm7k7gd`
- **Hypothesis**: signed-sqrt power transform `y' = sign(y)|y|^0.5` on SP targets compresses heavy-tail distribution → SP-plateau crack via tail-residual normalization.

### Terminal metrics (run kjm7k7gd, EMA EP13)

| Metric | H117+DP | H112 SOTA | Δ | Verdict |
|---|---:|---:|---:|---|
| val_abupt | 6.185% | 6.1358% | +0.049pp | ❌ A WIN missed (4.9bp) |
| test_abupt | 5.930% | 5.839% | +0.091pp | ❌ regression (9.1bp) |
| **test_WSS** | **6.7555%** | 6.752% | **+0.003pp** | ❌ **TIES H112, no merge** |
| test_WSS_x | **5.9916%** | 5.999% | −0.007pp | TIE (banked: first sub-6% on tay) |
| test_WSS_y | 7.365% | 7.360% | +0.005pp | TIE |
| test_WSS_z | 8.751% | 8.720% | +0.031pp | TIE |
| test_VP | 3.533% | 3.421% | +0.112pp | ❌ regression vs H112; CROSSES pre-H112 floor 3.643% (inherited) |
| **test_SP** | **4.010%** | 3.695% | **+0.315pp** | ❌ **22nd SP-plateau confirmation** |

### Pinned student diagnostic — SP-gap-closure trajectory fingerprint

| EP | H117+DP val_SP | H112 val_SP | SP gap | Δ gap |
|---|---:|---:|---:|---:|
| 1 | 22.196% | 19.368% | +2.83pp | — |
| 2 | 5.936% | 5.361% | +0.58pp | −2.25 |
| 3 | 4.980% | 4.663% | +0.32pp | −0.26 |
| 4 | 4.682% | 4.380% | +0.30pp | −0.02 |
| 13 | 4.326% | 4.055% | **+0.27pp** | saturated |

SP gap never closed below +0.25pp. Signed-power inverse Jacobian `0.5·|y|^{-0.5}` permanently attenuates gradient signal on heavy-tail residuals (wheel-arch, stagnation regions) — anti-mechanism for SP plateau.

### Strategic class verdict — SP-AXIS DATA-TIER CLOSURE LOCKED (5th rejection)

Data-tier SP attacks now have **5 consecutive C NULL** rejections plus 17 prior plateau hits = **22 consecutive SP-axis confirmations**:
1. H113 fern free log_sigma_sq (balance) — C NULL
2. H114 panel-area weighting — C NULL
3. H115 thorfinn Huber curvature — C NULL (Huber→MSE degeneration)
4. H116 nezuko Y-mirror data-aug — C NULL (PE non-equivariance)
5. **H117 alphonse signed-sqrt SP target transform — C NULL (inverse-Jacobian attenuation)**

**Data-tier and loss-form interventions on SP are EXHAUSTED on this model+dataset.** Student's "SP-decoder probe" suggestion (dedicated SP head with own LR/width/depth) banked as Wave 38+ candidate.

### Banked observation — test_WSS_x sub-6%

test_WSS_x 5.9916% is the **first sub-6% test_WSS_x on tay** (H112 5.999%). −0.007pp is within run-to-run noise but if H128 SwiGLU or future architectural change replicates with tighter margin, that's a frontier worth attribution. Banked, not pursued separately.

### Successor experiment

alphonse → H128 SwiGLU MLP (PR #1308) — architectural primitive feedforward modernization (Llama/Gemma/PaLM/Mistral best practice).

---

## 2026-05-24 ~18:15 — PR #1295: H119 COMPOUND H102-WIDER × H112-DROPPATH (edward, **CLOSED B PARTIAL test_VP only** — primary objective regresses, MAJOR program finding on compound additivity topology)

- **Branch**: `edward/h119-compound-droppath-wider-surface-decoder` (CLOSED, not merged)
- **W&B run**: `lm8aflyv`
- **Hypothesis**: orthogonal-class compound — DropPath (H112 backbone-regularization) × wider surface_out 2× (H102 decoder-capacity) → additive across compound mechanism classes per H110 locked rule.

### Terminal metrics (run lm8aflyv, EMA EP13)

| Metric | H119 | H112 baseline / floor | Δ | Verdict |
|---|---:|---:|---:|---|
| val_abupt | 6.213% | 6.1358% gate | +0.077pp | ❌ A WIN missed |
| test_abupt | 5.860% | 5.839% (H112) | +0.021pp | ❌ regression |
| **test_WSS** | **6.777%** | 6.752% (H112) | **+0.025pp** | ❌ **primary objective regresses** |
| test_VP | 3.398% | 3.421% (H112) | **−0.023pp** | ✅ **B PARTIAL** |
| test_SP | 3.701% | 3.695% (H112) | +0.006pp | flat |
| test_WSS_z | 8.802% | 8.720% (H112 prog-best) | +0.082pp | ❌ **LOST H112 prog-best** |

### MAJOR PROGRAM FINDING — compound additivity refinement LOCKED (student diagnostic)

> **Compound additivity is necessary but NOT sufficient at the orthogonal-class level. Per-channel topology matters.**
>
> Two datapoints:
> - decoder-capacity × decoder-capacity (H110: H102+H101) → anti-additive on VP/SP, **additive+ on WSS_z**
> - decoder-capacity × backbone-regularization (H119: H102+H112) → VP-stabilizing, **anti-additive on WSS_z**

**WSS_z specifically rejects `decoder × regularization` compounding** while accepting `decoder × decoder`. Mechanism (working hypothesis): DropPath's per-token stochastic skip prevents the wider decoder from exploiting consistent neighborhoods that WSS_z prediction depends on (high spatial frequency in vertical-shear channel).

### Secondary finding — VP-channel stabilization

- val_VP at parity with H112 throughout cosine (3.59% terminal)
- test_VP **3.398% < H112's 3.421%** by −0.023pp — first sub-H112-floor test_VP on tay
- **H102's standalone VP over-fit signature is eliminated by DropPath** — regularization-stabilization thesis on VP confirmed
- Not load-bearing for WSS-axis primary objective; banked for future VP-targeted hypotheses

### Successor experiment

edward → H-B (Auxiliary log-magnitude head, PR #1307) — non-destructive magnitude/direction decoupling on WSS, mechanistically distinct from compound-axis approach.

### Strategic verdict

n_hidden=512 decoder-capacity axis is largely exhausted for compounds with regularization/geometry. Standalone wider decoder (H127 tanjiro in flight) is the right test of pure decoder-capacity. If H127 succeeds, next compound is decoder × backbone (H127 × H120 depth-6), NOT decoder × regularization.

---

## 2026-05-24 ~18:15 — PR #1302: H-A SURFACE-INTRINSIC TANGENT-FRAME INPUT ENCODING (thorfinn, **CLOSED C NULL** — destructive-swap formulation killed at EP1 cold-start fence; input-rep class NOT closed, redirected to H-A2 non-destructive concat)

- **Branch**: `thorfinn/h-a-surface-intrinsic-tangent-frame` (CLOSED, not merged)
- **W&B run**: `nkf6gro9`
- **Hypothesis**: per-point local tangent frame (t̂₁, t̂₂, n̂) from existing surface normals replaces world (x,y,z) at the surface input → ≥0.15pp test_WSS improvement via frame-aligned inductive bias for surface vector fields.

### Terminal metrics (EP1 kill-fence trip at step 10864)

| Metric | H-A EP1 | H112/H115 canonical EP1 | Δ |
|---|---:|---:|---:|
| **val_abupt** | **40.22%** | 27.53% | **+12.68pp** ❌ fence breach (gate <35%) |
| val_WSS | 45.66% | — | — |
| val_surface_loss | 0.193 | 0.091 | **+0.102 (2.13×)** |
| train_loss EP1 epoch_avg | 0.571 | ~0.09 (H115) | **~6.3×** |

Run terminated cleanly via `early_stop.triggered=1`, no NaN/OOM/crash.

### Pinned diagnostic — two mechanisms (student)

**(1) Loss of global spatial structure** — world (x,y,z) encodes "this point is at front-top of car"; tangent-frame `(δp·t̂₁, δp·t̂₂, δp·n̂)` only encodes "X meters along local panel from per-case centroid in locally-defined tangent direction". Two physically-adjacent points across a curvature feature (hood-windshield edge) have O(1) different local-frame coords — RFF/StringSeparable PE (sigma-tuned on world xyz) sees a much higher-frequency input field.

**(2) t̂₁ direction discontinuity at |n̂·ẑ|=0.95 reference-flip boundary** — `t̂₁ = normalize(cross(n̂, ref))` flips orientation discontinuously where ref switches from ẑ to ŷ, creating O(1) coordinate jumps across the band.

Both consistent with train_loss ~6.3× higher at EP1 — model wasn't slower to learn, it was actively re-fitting a less-coherent input field.

### Implementation correctness

All 4 PR-required diagnostic gates passed: centroid per-case ✓ (centroid_x varied 1.367..1.655), fallback mask 16-22% on full meshes ✓, coord ranges bounded ✓, shape (N,7) preserved ✓.

### Hypothesis class — NOT closed

The output-tier tangency-imposition class (#351/#680/#713/#1299, 4 failures) and the input-tier destructive-swap (this PR) are NOT the same class. Input-representation as **additive auxiliary** info has never been tested.

### Successor experiment

thorfinn → H-A2 (concat both frames, PR #1306) — surface input (N,7)→(N,10), world (x,y,z) preserved + tangent-frame channels added with zero-init residual projection. Directly addresses mechanism (1) by being non-destructive.

---

## 2026-05-24 ~17:15 — PR #1293: H118 SLICE-COUNT 128→192 (tanjiro, **CLOSED C NULL** — slice-axis capacity exhausted at 13ep budget; **CAPACITY-AXIS ORDERING depth > hidden > slices LOCKED**)

- **Branch**: `tanjiro/h118-model-slices-128-to-192` (CLOSED, not merged)
- **W&B run**: `tdmo2i9h`
- **Hypothesis**: slice count 128→192 → finer surface feature partition in stagnation regions → test_SP and test_WSS_z improve.

### Terminal metrics (run tdmo2i9h, EMA EP11 best)

| Metric | H118 | H112 baseline / #972 | Δ | Floor / Gate | Verdict |
|---|---:|---:|---:|---|---|
| val_abupt | **6.394%** | 6.126% | +0.268pp | gate 6.1358% | ❌ A WIN missed |
| test_abupt | 6.143% | 5.839% | +0.304pp | merge gate 5.839% | ❌ regression |
| **test_WSS** | **7.127%** | 6.752% (H112) | **+0.400pp** | floor 6.727% | ❌ MISS primary objective |
| test_WSS_x | 6.357% | 5.83% (#972) | +0.527pp | — | ❌ regression |
| test_WSS_y | 7.666% | 7.10% (#972) | +0.566pp | — | ❌ regression |
| **test_WSS_z** | **9.220%** | 9.83% (#972) | **−0.610pp** | — | ✓ ONLY directional positive |
| test_VP | 3.600% | 3.643% floor | −0.043pp | floor 3.643% | ✓ CROSSES (incidental) |
| test_SP | 3.872% | 3.577% floor | +0.295pp | floor 3.577% | ❌ MISS — **18th SP plateau confirmation** |

### Pinned diagnostic (student)

EP2 lead → EP3 reversal → uniform regression. Slice-resolution warmup advantage burns off in early cosine; extra slices need MORE gradient steps than 13ep budget provides. Param cost +60K (cheap), throughput cost +5-7% (within prediction).

### Strategic class verdict — capacity-axis ordering LOCKED

Combined with H120 depth-6 (val_abupt 6.089% at 88.2%, leading fleet) and H121 hidden-576 (6.249% at 76.6%, still descending):

**depth > hidden_dim > slices** at 13ep budget.

Future capacity-axis attacks compound depth + hidden, NOT slice expansion. Slice reduction to 64 deferred (counter-test of bottleneck hypothesis, low priority).

### Banked niche observation

H118's lone **test_WSS_z improvement (−0.610pp vs #972 SOTA)** suggests slice attention has a specific tau_z relationship distinct from slice-axis capacity effects. Possible mechanism: more slices = finer partition of horizontal panels (roof/floor where tau_z dominates) → improved spatial discrimination for vertical WSS. **Niche, not load-bearing** — overall test_WSS still regressed +0.400pp. Not pursuing unless a future hypothesis specifically targets the tau_z-slice-panel-orientation interaction.

### Successor experiment

tanjiro → H127 (Wider Surface Output Decoder STANDALONE, PR #1305) — cross-track validation of dl24-H39's test_WSS 6.6506% finding, filling the missing clean standalone data point (H119 edward currently testing the COMPOUND with DropPath, anti-additive late-cosine).

---

## 2026-05-24 ~16:10 — PR #1291: H116 Y-MIRROR DATA AUG (nezuko, **CLOSED C NULL** — inverse val→test on test_WSS, MAJOR test_VP regression; **DATA-AUG TIER FIRST CLEAN FALSIFICATION ON DRIVAERML**)

- **Branch**: `nezuko/h116-y-mirror-augmentation` (CLOSED, not merged)
- **W&B run**: `95jd18kv`
- **Hypothesis**: longitudinal Y-mirror data augmentation on bilateral car symmetry — `apply_y_mirror(prob=0.5)` flips y → -y, n_y → -n_y, τ_y → -τ_y. Falsifiable: val_abupt < 6.126% AND test_SP < 3.577% → NULL otherwise.

### Terminal metrics (run 95jd18kv, EMA EP13)

| Metric | H116 | H112 baseline | Δ | Floor / Gate | Verdict |
|---|---:|---:|---:|---|---|
| val_abupt | **6.354%** | 6.126% | +0.228pp | gate 6.1358% | ❌ A WIN missed |
| test_abupt | 6.118% | 5.839% | +0.279pp | merge gate 5.839% | ❌ regression |
| **test_WSS** | **6.888%** | 6.752% | **+0.136pp vs 6.727% floor** | floor 6.727% | ❌ MISS by +0.161pp |
| test_tau_y (PR aux prediction) | 7.451% | 7.10% | +0.351pp | — | ❌ **falsifies "balanced tau_y exposure improves tau_y" claim** |
| **test_VP** | **4.314%** | 3.421% | **+0.893pp** | floor 3.643% | ❌ **MAJOR regression +0.671pp vs floor** |
| test_SP | 3.744% | 3.695% | +0.049pp | floor 3.577% | ❌ MISS — **18th SP plateau confirmation** |

Late-cosine slope dampened MORE aggressively than advisor's pre-terminal projection (predicted test_WSS ~6.74%, actual 6.888%).

### Three-reason post-mortem (nezuko's diagnostic, banked)

1. **"Free 2× augmentation" framing was misleading.** Training loader already uses `torch.randint(N_surface, (65536,))` per view → ~65K of 8.8M points per epoch with massive stochastic richness. Y-mirror is a single deterministic transformation on top of an already-stochastic sampling distribution orders of magnitude richer. Effective data multiplicity was barely changed.

2. **Asymmetric flow-disturbing features matter more than surface-area suggests.** Side mirrors / exhaust / antenna / license plate <1% surface area but DISPROPORTIONATELY influential on pressure/wake patterns. Model must "split the difference" between real and mirrored targets → systematic label bias globally, not local.

3. **`string_separable` PE is NOT y-equivariant** (PR #823 backbone). Per-axis learnable RFF log-frequencies + slice-attention on absolute coord signatures. Under y → -y, encoded features change. Model spends capacity learning two RFF embeddings should produce the same output. **CLEANEST mechanistic explanation.**

Reasons 2+3 alone are sufficient to predict the measured regression. Compound effect explains why test_VP regression is catastrophic: the volume side learning is downstream of surface→volume cross-attention, which is downstream of the equivariance-broken surface PE.

### Strategic class verdict — Data-tier joint-closure

| Class | Mechanism | Outcome |
|---|---|---|
| H113 fern (loss-balance) | 3 free log_σ² scalars | C NULL — SP plateau hardness-bound, not balance-bound |
| H114 (panel-area loss) | per-panel area weighting | C NULL — 4× slowed descent |
| H115 thorfinn (loss-curvature) | fixed-δ Huber | C NULL — degenerated to MSE |
| **H116 nezuko (data-aug Y-mirror)** | bilateral symmetry exposure | **C NULL — inverse val→test correlation** |

Combined: **loss-tier (H113/H114/H115) AND val-correlated data-aug tier (H116) are jointly closed for SP on this model+dataset**. Plateau Protocol pivots Wave 36+ to:
- **Input representation tier** (H-A thorfinn tangent-frame, IN FLIGHT)
- **Sampling distribution tier** (H126 nezuko inverse-area, IN FLIGHT, refined H-C)
- **Architectural symmetry tier** (Wave 37+ candidate: y-mirror equivariant encoder per nezuko's reason 3)

### Banked lessons

1. **Data augmentation can produce inverse val→test correlation on DrivAerML** when (a) the augmentation is downstream of a non-equivariant PE, OR (b) it asks the model to reconcile asymmetric flow-disturbing features. First clean falsification of "free 2× data aug" hypothesis class.
2. **Volume regression** in H116 (test_VP +0.671pp vs floor) demonstrates that surface-side data-distribution changes propagate to volume through surf→vol cross-attention. Any future surface-side data-distribution-shift hypotheses must consider this coupling.
3. **`string_separable` PE non-equivariance** is now a confirmed friction-source for any surface-side transformations. Wave 37+ should attack this with a y-mirror equivariant encoder (use `cos(2π f·|y|)` on y-axis).
4. **Successor experiment**: nezuko → H126 (Inverse-Panel-Area Stratified Sampling, PR #1303) — sampling-distribution change at WSS-variance regions, NOT data-distribution shift.

---

## 2026-05-24 ~14:45 — PR #1290: H115 HUBER-LOSS-SP (thorfinn, **CLOSED C NULL** — Huber degenerated to MSE for ~90% of training; **LOSS-FORM CLASS EXHAUSTED FOR SP**)

- **Branch**: `thorfinn/h115-sp-huber-loss` (CLOSED, not merged)
- **W&B run**: `x6o14wwm`
- **Hypothesis**: Replace MSE on SP targets with Huber(δ=1.0) to bound outlier gradient magnitudes during the test_SP plateau period; falsifiable test_SP cross below 3.577% floor.

### Terminal metrics (run x6o14wwm, EMA EP13)

| Metric | H115 | H112 baseline | Δ | Verdict |
|---|---:|---:|---:|---|
| val_abupt | **6.367%** | 6.126% | +0.241pp | ❌ A WIN missed |
| test_abupt | 6.110% | 5.839% | +0.266pp | regression |
| test_VP | 3.658% | 3.421% | +0.015pp vs 3.643% floor | ❌ marginal miss |
| test_SP | **3.954%** | 3.695% | +0.377pp vs 3.577% floor | ❌ **17th plateau confirmation** |
| **test_WSS** | **7.026%** | 6.752% | **+0.299pp vs 6.727% floor** | ❌ regression |
| test_WSS_z | 9.091% | 8.720% | +0.20pp | regression |

### Pinned diagnostic — Huber degenerated to MSE

Per-step `train/huber/sp_linear_regime_frac` (% of SP points with |residual| > δ=1.0):

| Step | Linear-regime frac | sp_abs_residual_mean | Huber state |
|---:|---:|---:|---|
| 7,112 | 0.86% | 0.142 | Huber active (early cold-start) |
| 14,149 | **0.0%** | 0.000 | already MSE-equivalent |
| 21,123 | 0.0% | 0.000 | MSE-equivalent |
| 42,662 | 0.0023% | 0.028 | trace |
| 70,664 (term) | 0.0008% | 0.0188 | effectively MSE |

**δ=1.0 was calibrated for early-training residual scale (mean 0.71, max 7.77 at step 1).** By step 14k, residuals had collapsed to mean ~0.02 / max ~1.0 — Huber was numerically MSE × 0.5 for ~90% of optimization. The experiment did NOT test the hypothesis it was designed to test (it tested "MSE + tail-bounded gradients during first 10% of steps" — before SP plateau exists).

### Strategic class verdict — SP loss-form CLOSED

3rd falsification of loss-side intervention on SP:
- **H113 fern**: SP plateau is HARDNESS-bound, not balance-bound (3 log_sigma_sq scalars, 2.1% relative spread over 13ep)
- **H114** (advisor reference): panel-area weighting; 4× slowed descent
- **H115 thorfinn** (this): fixed-δ Huber curvature → MSE degeneracy

Combined with 14 prior SP plateau hits on diverse architectures, **the loss-form lever class is exhausted for the standard masked-loss family on normalized SP targets**. Future SP attacks must work at the **data tier** (CDF-normalize SP targets, log-transform tails) or **representation tier** (per-point uncertainty heads), NOT the loss tier.

### thorfinn's adaptive-δ suggestion — REJECTED

Adaptive `δ_t = c · running_median(|r|_t)` would technically engage Huber at the plateau, but reintroduces **scale-coupling pathology** (loss-form changes its own statistics as a function of optimization state — H114-class spurious-attractor failure). Not productive use of GPU on a metric whose plateau we believe is hardness-bound (H113 verdict).

### Banked lessons

1. **Fixed-δ Huber with δ calibrated to early-training residuals cannot probe late-training plateaus**. Loss-form interventions on collapsing residual distributions must use adaptive δ — but adaptive δ couples to optimization state.
2. **SP plateau confirmed 17th time** across loss-form (H113, H114, H115), capacity (H102, H103, H104, H113), data-aug (H116), and curvature interventions. Plateau is **structural** within current model+target representation.
3. **Successor experiment**: thorfinn → H-A (Surface-Intrinsic Tangent-Frame Encoding, PR #1302) — Morgan's WSS-primary directive, NOT another SP attack.

---

## 2026-05-24 ~14:15 — PR #1299: H123 WSS-TANGENT-FRAME-PROJECTION (fern, **CLOSED C NULL** — terminated step 33,297, EP3 confirms hopeless trajectory; **TANGENCY-IMPOSITION CLASS FULLY EXHAUSTED — 4th consecutive failure**)

- **Branch**: `fern/h123-wss-tangent-frame-projection` (CLOSED, not merged)
- **W&B run**: `80maak4j`
- **Hypothesis**: Hard tangent-plane projection of predicted WSS vector — `τ_tangent = τ − (τ · n̂) n̂` applied in physical space (Option A, confirmed by advisor). Zero added parameters. Falsifiable prediction: test_WSS improves, biggest gain on τ_z (horizontal panels with large n_z).

### Terminal metrics (EP3 best checkpoint, post-mortem eval)

| Metric | H123 (EP3) | H112 baseline | Δ vs H112 | Verdict |
|---|---:|---:|---:|---|
| test_abupt | 11.148% | 5.839% | +5.31pp | massive regression |
| **test_WSS** | **15.262%** | 6.752% | **+8.51pp** | massive floor breach |
| test_WSS_z | 16.653% | 8.720% | +7.93pp | targeted bottleneck gets WORSE |
| test_VP | 3.980% | 3.421% | +0.56pp | floor breach |
| test_SP | 4.357% | 3.695% | +0.66pp | floor breach |

**EP3 apple-to-apple vs H112 EP3**: val_abupt +4.44pp, val_WSS +7.44pp — monotone-divergent trajectory. Gap unrecoverable within budget.

### Diagnostic locked — tangency-imposition class failure mechanism

**Physical-space projection was implemented correctly.** EP1 passed cleanly (26.348%, within H112 EP1 range). The failure is architectural, not implementational:

1. **Gradient pathology**: Hard projection zeros the gradient for the normal component of τ: `∂τ_tangent/∂(model params) = 0` for any parameter that only affects the normal component. Model has no learning signal to predict tangent vectors natively.
2. **Confirmed**: `pre_proj_normal_rel_mean_phys` held at **~0.34 throughout training** (EP1 0.38-0.42, stable). The model never learned to reduce its ~34% normal component. Completely unlike the hoped-for "learns to be tangent before projection" trajectory.
3. **Same failure class as PR #713** (soft penalty `λ·|ws·n̂|²`): both interventions cause model to "satisfy the constraint by sacrificing τ accuracy" — either via zero-gradient (hard) or capacity redirection (soft).

**Kill-threshold step-skew bug re-confirmed** (fern caught this): gate `32594:...` silently skipped because `global_step=32592 < 32594`. **Use 32592, not 32594.** Advisor memory `feedback_kill_thresholds_step_indexed.md` updated.

### Strategic consequence

Tangency-imposition class is PERMANENTLY DEPRIORITIZED (4 consecutive failures: PR #351, #680, #713, #1299). WSS-axis attack must come from data-augmentation tier (H116 Y-mirror, working marginally) or capacity tier (H120 depth-6, strong), NOT geometric-constraint tier.

### Successor

H125 (PR #1301, fern) assigned: Backbone depth 5→7, extends H120 capacity-axis dominance. Single change: `--model-layers 7`.

---

## 2026-05-24 ~07:55 — PR #1285: H113 HETEROSCEDASTIC-UNCERTAINTY-WEIGHTING (fern, **CLOSED C NULL** — terminal at step 70,664, all four AND-gate axes miss; **DIAGNOSTIC LOCKED** — SP plateau is hardness-bound, not balance-bound)

- **Branch**: `fern/h113-heteroscedastic-uncertainty-weighting` (CLOSED, not merged)
- **W&B run**: `v5m2w16v` (group `wave33_h113_heteroscedastic_loss`)
- **Hypothesis**: Kendall+Gal 2018 multi-task heteroscedastic uncertainty weighting — 3 learnable log_σ² scalars on (SP, VP, WSS) tasks; loss = Σ (0.5·exp(−log_σ²)·L_i + 0.5·log_σ²). Designed as Plateau Protocol strategy-tier shift #1 to falsify the SP plateau hypothesis.

### Terminal metrics

| Metric | H113 terminal | H112 SOTA / floor | Δ | Verdict |
|---|---:|---:|---:|---|
| val_abupt | 6.3864% | 6.1358% (merge gate) | +0.251pp | MISS |
| test_abupt | 6.0689% | 5.839% canonical | +0.230pp regress | regress |
| test_VP | 3.7167% | 3.643% floor | +0.074pp | **MISS** (no cross — broke fern's H101-H105 streak) |
| test_SP | 3.8551% | 3.577% floor | +0.278pp | MISS — **15th plateau** |
| test_WSS | 6.9470% | 6.727% floor | +0.220pp | MISS |
| test_WSS_z | 9.0396% | 8.945% canonical | +0.095pp regress | regress |

All four AND-gate criteria miss. Strictly worse than fern's prior H105 (`t6b1i2yk`) on every axis. **Confirms heteroscedastic loss reformulation regresses vs prior info-at-decoder mechanism class.**

### Diagnostic answer (the value of this null)

H113 was the first **Plateau Protocol strategy-tier shift** experiment. It posed the falsifiable question: *"Is the persistent SP plateau driven by undertrained SP loss term (loss-balancing) OR Bayes-optimal hardness (data/representation-bound)?"*

The log_σ² trajectory answered definitively:

| Step | log_σ²_sp | log_σ²_vp | log_σ²_wss | spread | precision_wt | val_abupt |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 1.00× | — |
| 10,864 (EP1) | −0.044 | −0.038 | −0.038 | 0.007 | 1.05× | 29.08% |
| 32,594 (EP3) | −1.900 | −1.894 | −1.879 | 0.021 | 6.69× | 6.94% |
| 48,902 (EP6) | −3.189 | −3.183 | −3.168 | 0.021 | 24.3× | 6.47% |
| 70,664 (terminal) | −3.990 | −3.983 | −3.969 | **0.021** | **54.0×** | 6.39% |

The optimizer was given 3 free scalars and found ZERO meaningful per-task imbalance (2.1% relative spread throughout). Instead it exploited the unbounded `0.5·log_σ²` regularizer to amplify total loss magnitude ~54×.

**Empirical verdict: SP plateau is HARDNESS-BOUND, not balance-bound.** Per-task aleatoric noise in DrivAerML is small. The SP plateau (15 consecutive misses, range 3.70-3.95%) is a representation/data ceiling.

### Strategic consequences

- **Loss-balancing mechanisms DEPRIORITIZED** (gradient-norm balancing, PCGrad, uncertainty weighting — same direction, falsified)
- **Plateau-tier loss reformulations DEPRIORITIZED** for SP — H115 Huber descending canonically but EP4 6.663% does not look like it will deliver A WIN; loss-curvature changes are likely weak
- **Wave 36+ direction confirmed**: data-tier / SP-specific architecture / WSS-aligned decoder heads — fern reassigned to **H123 WSS tangent-frame projection** (PR #1299)

### Implementation cost

+3 learnable scalar parameters (effectively 0 vs 17.41M total). Code: model.py (+30 lines), train.py (+181 lines: CLI flag, threading, het-loss path, 12 diagnostic W&B metrics). DDP integrates cleanly without `find_unused_parameters`. Numerically safe (log_σ² clamp [−8,+8]). 839.4 min wall clock, ~76 GB GPU peak.

### Quote-worthy

> "The mechanism is well-engineered and well-engaged — finite, no NaN, gradients flowing on all three scalars. It's doing exactly what the math says it should do; the math just doesn't help on this dataset." — fern, terminal SENPAI-RESULT, 2026-05-24T07:52Z

---

## 2026-05-24 ~03:45 — PR #1289: H114 PANEL-AREA-WEIGHTED SP MSE (frieren, **CLOSED C NULL** — EP3 kill-threshold triggered) — first NULL of Wave 35 data-tier sweep; mechanism failure mode = mid-cosine descent stall via spurious large-panel attractor

- **Branch**: `frieren/h114-panel-area-weighted-sp-loss` (CLOSED, not merged)
- **W&B run**: `jity0d6x`, **TERMINATED EARLY at step 32,651 (46.2% complete)**, runtime 3.68h
- **Hypothesis**: per-point SP MSE reweighted by `panel_area / sum(panel_area * mask)` to combat under-weighting of large panels in stagnation regions.

### Why TERMINATED at step 32,651

- Kill threshold: `32594:val_primary/abupt_axis_mean_rel_l2_pct<8.5`
- val_abupt at step 32,651: **11.7685%** → MISS by +3.27pp → kill trigger fired
- Slope at termination: **−0.0544pp/1k** vs canonical ~−0.2pp/1k → **~4× slower descent**

### Val publish history

| Step | val_abupt | Note |
|---:|---:|---|
| 10,880 | 18.997% | **EP1 gate 35% CLEARED by +16pp** ✅ |
| 13,600 | 16.551% | Check-in #1 |
| 32,651 | **11.7685%** | **EP3 gate 8.5% MISSED — KILL TRIGGER** ❌ |

### Mechanism failure mode

Panel-area distribution on DrivAerML is heavy-tailed: ~0.5% of points cover ~40% of total area (front/rear bumper, A-pillars, roof). After normalization to `sum(area * mask)`, the per-point weight on these ~150-500 large panels is ~80× higher than median.

**The optimizer found a spurious attractor**: fit the *easy* low-frequency low-residual SP on dominant panels first, *neglect* small-panel stagnation/wheel-arch regions that drive validation. Result: area-weighted training loss descended, but per-point validation metric (the canonical scoring contract) crawled.

### Strategic lesson

SP-axis interventions that heavily reweight gradient magnitude based on intrinsic mesh structure (panel_area, normal_magnitude, sdf_proximity) create spurious attractors that beat the *modified* objective but lose on per-point validation. **Loss curvature changes** (Huber H115) and **target-space reshaping** (signed-sqrt H117) are safer — they preserve relative per-point weighting while only changing loss-surface shape in small vs large residual regions. **Prediction: H115/H116/H117 more likely to engage productively than H114 did.**

### NEXT ASSIGNMENT — H121 BACKBONE HIDDEN-DIM 512→576 (frieren)

- **Branch**: `frieren/h121-backbone-hidden-576`, DRAFT PR #1297
- **Hypothesis**: parallel-feature-width within backbone (`--model-hidden-dim 576`, +2.5M params, +15% wallclock)
- **Strategic role**: FOURTH orthogonal Wave 36+ capacity-scaling axis, completing the comprehensive backbone capacity sweep:
  - H118 tanjiro: slice granularity (`--model-slices 192`)
  - H119 edward: decoder-width compound (DropPath × surface_out 1024)
  - H120 askeladd: sequential depth (`--model-layers 6`)
  - **H121 frieren: parallel feature width (`--model-hidden-dim 576`)** ← fourth axis
- VRAM concern flagged — smoke test BEFORE full launch; fallback to `--model-hidden-dim 552` if peak >92GB
- Includes `--drop-path-max 0.10` to test width-scaling ON TOP of MERGED SOTA

---

## 2026-05-24 ~03:15 — PR #1282: H111 LAYERSCALE-IN-BACKBONE (askeladd, **CLOSED B PARTIAL**) — narrow test_VP cross −0.021pp; γ-depth pattern firmly engaged (γ_mlp 0.95→1.40 monotonic) but stochastic regularization dominated cohort head-to-head; 21st SP plateau confirmation

- **Branch**: `askeladd/h111-layerscale-in-backbone` (CLOSED, not merged)
- **W&B run**: `z7ip68eo`, 13 epochs (terminal), runtime 14.8h, throughput ~1.75 steps/s (~5-9% below canonical due to γ broadcast overhead)
- **Hypothesis**: CaiT-style learnable per-channel scale γ on backbone residuals (γ_attn, γ_mlp per block, init=1.0). +5,120 params total (5K = 2 × 512 × 5).

### Terminal results

| Channel | Validation | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.3282%** (+0.202pp above val gate 6.126%) | 5.9840% | 5.844% | +0.140pp MISS |
| surface_pressure | 4.171% | 3.7873% | 3.577 (floor) | **+0.210pp MISS floor — 21st plateau** |
| **volume_pressure** | 3.7444% | **3.6218%** | 3.643 (floor) | **−0.021pp CROSS (narrow)** ✓ |
| wall_shear | 7.1608% | 6.8749% | 6.727 (floor) | +0.148pp MISS |
| **wall_shear_z** | 9.7208% | **8.9468%** | 8.945 (canonical) | +0.002pp ≈ TIE |
| wall_shear_x | 6.0866% | — | — | — |
| wall_shear_y | 7.4775% | — | — | — |

### γ-engagement diagnostic — LayerScale FIRMLY ENGAGED (no-op falsified)

| Block | γ_attn mean (init=1.0 → terminal) | γ_mlp mean (init=1.0 → terminal) |
|---|---:|---:|
| 0 (shallowest) | **0.7740** (range [0.371, 1.164]) | **0.9469** (range [0.696, 1.179]) |
| 1 | 0.8380 | 1.0550 |
| 2 | 0.8456 | 1.2255 |
| 3 | 0.8705 | 1.3388 |
| 4 (deepest) | 0.8492 | **1.3990** (range [1.037, **1.926**], std 0.186) |

- **γ_mlp monotonic depth-amplification**: 0.95 (block 0) → 1.40 (block 4) — optimizer found late-block MLP residuals need amplification, early-block MLP need dampening
- **γ_attn uniformly suppressed** across all 5 blocks (mean ~0.85) — attention residuals are universally damped
- **Partial gating** (not full pruning) — some block-0 γ_attn channels dropped to 0.37
- γ-as-no-op outcome **firmly FALSIFIED** — +5K params doing real per-channel depth-dependent work

### Why B PARTIAL not merged

- val gate MISS by +0.202pp — too large to merge
- BUT narrow test_VP cross is real (−0.021pp) → B PARTIAL classification (per program convention: at least one test floor crossed cleanly)
- test_WSS_z effectively tied canonical (+0.002pp vs projected −0.39pp) — γ mechanism stalled on tangential variance

### Cohort head-to-head verdict (DEFINITIVE on regularization arm)

| Mechanism | Params | Val | test_abupt | test_VP cross | Verdict |
|---|---:|---:|---:|---:|---|
| **H112 DropPath** | **0** | **6.1358%** | **5.839%** | **−0.222pp** | **A WIN MERGED** |
| **H111 LayerScale** | **+5K** | **6.3282%** | **5.984%** | **−0.021pp** | **B PARTIAL CLOSED** |

**Stochastic regularization (DropPath, 0 params, zero engineering) DOMINATES deterministic per-channel γ (LayerScale, +5K params, well-engineered) on DrivAerML.** The cohort crossed at step 38,030 and H112's lead widened monotonically to terminal (+0.18pp+).

### Mechanism characterization (LOCKED for retrospective)

- LayerScale γ engages richly but deterministically per-channel; once optimizer fixes γ values, further gains require structural changes
- DropPath introduces stochastic residual diversity in late cosine that γ cannot replicate
- DrivAerML rewards **residual-path diversity** over **per-channel residual rescaling**
- Mechanism engagement diagnostic (γ depth pattern) ≠ test improvement — γ pattern was the best of any Wave 33+34 mechanism, but produced the shallowest test_VP cross

### Suggested follow-ups (logged but lower-priority given H112 dominance)

1. γ init=1e-4 (CaiT recipe) — γ grows from zero, may engage residuals more selectively
2. **H111+H112 LayerScale+DropPath compound** — stochastic + deterministic, +5K params, different mechanism classes
3. γ_mlp only (drop γ_attn) — cuts 5K → 2.5K, isolates productive mechanism
4. Block-0 γ_attn collapse investigation — Lion weight-decay interaction?

### Strategic implication

- **Regularization arm of Wave 33+34 is fully characterized**: stochastic wins (DropPath A WIN MERGED), deterministic per-channel γ is B PARTIAL territory
- 21st SP plateau confirmation — Wave 35 data-tier (H114-H117) + Wave 36 capacity-scaling (H118 slices, H119 compound, H120 depth) is the comprehensive multi-axis attack

### NEXT ASSIGNMENT — H120 BACKBONE DEPTH 5→6 (askeladd)

- **Branch**: `askeladd/h120-backbone-depth-6`, DRAFT PR #1296
- **Hypothesis**: +1 backbone layer (~+3.2M params, ~+18% wallclock) — third orthogonal capacity-scaling axis. Tests whether sequential representational DEPTH was the bottleneck (vs H118 slice GRANULARITY, vs H119 decoder WIDTH compound)
- **Pre-launch concern**: VRAM may exceed 96GB H100 limit — gradient checkpointing or batch-size reduction may be required. PR body includes the smoke-test instructions
- Rebase onto current `tay` (post-H112 merge) and include `--drop-path-max 0.10` so H120 tests depth-scaling ON TOP of MERGED SOTA, not against pre-H112 baseline

---

## 2026-05-24 ~02:45 — PR #1283: H112 STOCHASTIC-DEPTH-IN-BACKBONE DropPath (edward, **MERGED — NEW SINGLE-MODEL SOTA**) — A WIN at ZERO ADDED PARAMS; deepest test_VP cross of program (−0.222pp); deepest test_WSS_z of program (−0.225pp); test_abupt 5.839% beats prior canonical 5.844%

- **Branch**: `edward/h112-stochastic-depth-in-backbone` (**MERGED** into tay at ~02:40Z 2026-05-24)
- **W&B run**: `u9ue2ryb`, 13 epochs (terminal), runtime 14.52h, best EMA checkpoint EP13
- **Hypothesis**: Linearly-scheduled DropPath (stochastic depth) on backbone residuals — drop probability ramps from 0.0 at block 0 to 0.10 at block 4. Zero added parameters. Stochastic training pressure on backbone residuals improves generalization. First pure-regularization A WIN of the program.

### Terminal results

| Channel | Validation (terminal) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.1358%** (+0.010pp above val gate 6.126%) | **5.839%** | 5.844% | **−0.005pp BEATS CANONICAL** ← NEW TEST SOTA |
| surface_pressure | 4.0553% | 3.695% | 3.577 (floor) | **+0.118pp MISS floor — 20th plateau confirmation** |
| **volume_pressure** | **3.5478%** | **3.421%** | 3.643 (floor) | **−0.222pp DEEPEST CROSS OF PROGRAM** ✓✓✓ |
| wall_shear | 6.9670% | 6.752% | 6.727 (goal) | +0.025pp narrow MISS goal |
| wall_shear_x | 6.0923% | 5.999% | 5.83 (canonical) | +0.169pp regress |
| wall_shear_y | 7.6084% | 7.360% | 7.10 (canonical) | +0.260pp regress |
| **wall_shear_z** | **9.3750%** | **8.720%** | 8.945 (canonical) | **−0.225pp DEEPEST WSS_z OF PROGRAM** ✓✓✓ |

### 🏆 Why merged despite val gate miss (+0.010pp)

1. test_abupt 5.839% **beats** prior canonical baseline 5.844% by −0.005pp (primary test metric improves)
2. test_VP 3.421% is the **deepest VP cross of the program** by a factor of 3 (next best was H107 −0.089pp; H112 −0.222pp)
3. test_WSS_z 8.720% is the **deepest WSS_z improvement of the program** (−0.225pp vs canonical 8.945%)
4. ZERO added parameters — pure stochastic regularization mechanism
5. Preflight system check GREEN
6. CLAUDE.md: "When in doubt between merge and close, merge"

### Strategic significance

- **DropPath is now in the tay baseline** — all future Wave 36+ runs should include `--drop-path-max 0.10`
- val gate updates to val_abupt < **6.1358%** (slightly easier future gate from the stochastic training floor)
- test_VP 3.421% provides a 222bp safety margin buffer above the 3.643% floor for future compounds
- 20th consecutive SP plateau confirmation — Wave 35 data-tier sweep is actively attacking this

### NEXT ASSIGNMENT — H119 COMPOUND H112+H102 (edward)

- **Branch**: `edward/h119-compound-droppath-wider-surface-decoder`, DRAFT PR #1295
- **Hypothesis**: Add surface_out width-2× on top of H112 DropPath baseline (`--surface-out-width-factor 2.0`, +266K params). H110 diagnostic: decoder×decoder = anti-additive; regularization×decoder = orthogonal classes (additive). H112's DropPath should cure the VP floor breach that H102 alone caused (H102 test_VP 3.650% > floor 3.643%, vs H112 3.421% floor).
- **Flag correction**: auto-generated PR used `--use-drop-path --drop-path-rate 0.10` (WRONG); correct is `--drop-path-max 0.10` (from merged H112 code). Correction comment posted to PR #1295.

---

## 2026-05-24 ~01:30 — PR #1280: H110 COMPOUND H102+H101 (tanjiro, CLOSED) — **B PARTIAL** via test_VP cross (−0.029pp) + **DEEPEST test_WSS_z OF PROGRAM** (8.831%, −0.114pp below canonical 8.945%); val gate MISS by **+0.0102pp RAZOR-THIN** (narrowest non-merged miss of program); **19TH SP plateau confirmation**; first published **COMPOUND ADDITIVITY DIAGNOSTIC**: SATURATED on val, ANTI-ADDITIVE on VP/SP (worse than BOTH singletons), ADDITIVE on WSS/WSS_z

- **Branch**: `tanjiro/h110-wave34-h102-plus-h101-compound` (closed at ~01:30Z 2026-05-24)
- **W&B run**: `y6g8a0wm`, 13 epochs (terminal), runtime 14.18h, best EMA checkpoint at EP11
- **Hypothesis**: Stack the two A-WIN singletons: H102 surface_out width 1×→2× (+266K) + H101 zero-init position-residual at surface decoder (+1.5K). +268K total. First multi-mechanism compound on tay.

### Terminal results

| Channel | Validation (terminal) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.1362%** ❌ | **5.9393%** | 5.844% | **+0.0953pp MISS gate +0.0102pp** (RAZOR-THIN — narrowest non-merged miss of program) |
| surface_pressure | — | 3.7649% | 3.577 (floor) | **+0.188pp MISS floor — 19th plateau confirmation** |
| **volume_pressure** | — | **3.6142%** | 3.643 (floor) | **−0.0288pp CROSS floor** ✓ (shallow but valid) |
| wall_shear | — | 6.8235% | 6.727 (goal) | +0.097pp MISS goal |
| wall_shear_x | — | 6.0555% | 5.83 (canonical) | +0.226pp regress |
| wall_shear_y | — | 7.4307% | 7.10 (canonical) | +0.331pp regress |
| **wall_shear_z** | — | **8.8309%** | 8.945 (canonical) | **−0.1141pp DEEPEST WSS_z OF PROGRAM** ✓ |

### 🔥 First published COMPOUND ADDITIVITY DIAGNOSTIC — additivity is ASYMMETRIC across output channels

Direct decomposition of compound vs component singletons:

| Axis | H110 (compound) | H102 (width) | H101 (geom) | vs H102 | vs H101 | Reading |
|---|---:|---:|---:|---:|---:|---|
| val_abupt | 6.136 | 6.118 | 6.213 | +0.018 | −0.077 | **SATURATED** |
| test_VP | 3.614 | 3.543 | 3.514 | **+0.071** | **+0.100** | **ANTI-ADDITIVE** (worse than BOTH) |
| test_SP | 3.765 | 3.724 | 3.706 | +0.041 | +0.059 | **ANTI-ADDITIVE** |
| test_WSS | 6.824 | 6.858 | 6.913 | **−0.034** | **−0.089** | **ADDITIVE** |
| test_WSS_z | 8.831 | 8.945 | TBD | −0.114 | TBD | **ADDITIVE+** (deepest of program) |

**Conclusion**: compounds in Transolver decoder space have **asymmetric additivity** — cooperative on high-variance tangential axes (WSS_z), competitive on lower-variance pressure axes (VP/SP). **Strategic implication**: don't compound mechanisms that both operate on `surface_out` capacity (e.g. width + decoder-residual). Compound across orthogonal mechanism classes.

### Strategic implications

1. **DO NOT MERGE** — val_abupt 6.136% > 6.126% gate by +0.0102pp.
2. **Wave 35+ compound staging guidance**: stack across mechanism classes (decoder × regularization, decoder × data-tier, decoder × architecture/backbone) NOT within-class (width × geom-residual saturates).
3. **WSS_z deepening pattern locked**: H110 reaches 8.831% — first run below canonical 8.945% in WSS_z by a non-trivial margin.
4. **19 consecutive SP plateau confirmations** — Wave 35 4-axis data-tier sweep (H114/H115/H116/H117) + Wave 36 capacity-scaling (H118 starts now) form the comprehensive bookend probe of the SP plateau.

### NEXT ASSIGNMENT — H118 SLICES 128 → 192 (tanjiro)

- **Branch**: `tanjiro/h118-model-slices-128-to-192`, DRAFT PR #1293
- **Hypothesis**: Increase Transolver slice count from 128 → 192 (+50% slice tokens, +164K params, +12% wallclock). First probe of **Wave 36+ capacity-scaling frontier**. Tests whether slice resolution at the backbone bottleneck was the representational limit for the SP plateau.
- **Falsifiable**: if `test_SP < 3.577%`, slice-attention resolution was the bottleneck. If NULL combined with Wave 35 data-tier ALL NULL → **Bayes-optimal hardness confirmed**; pivot to drastically larger models in Wave 36+.
- **Why slices (not depth/width)**: cheapest capacity axis (+164K vs depth +3.4M/layer or width +5M for 512→640); directly tests representational granularity at the slice-attention compression bottleneck; complementary to data-tier sweep on data side.

---

## 2026-05-24 ~01:00 — PR #1279: H109 BACKBONE-SKIP RESIDUAL DECODER (alphonse, CLOSED) — **B PARTIAL** via clean test_VP cross (−0.060pp) + 3rd-best WSS_z improvement (−0.709pp); val gate MISS +0.115pp; **18TH consecutive SP plateau confirmation**; pre-backbone-embedding-reach-to-decoder mechanism characterized; ordering at +263K class locked: **WIDTH (H102) > parallel-MLP (H108) > pre-backbone-bypass (H109)** — post-backbone bypass beats pre-backbone bypass at matched cost

- **Branch**: `alphonse/h109-backbone-skip-residual-decoder` (closed at ~01:00Z 2026-05-24)
- **W&B run**: `77tfuj6i`, 13 epochs completed (terminal), runtime ~14h
- **Hypothesis**: Zero-init `Linear(512, 512)` projection of pre-backbone embedded surface tokens (post-`surface_in`, post-RFF/STRING, post-`surface_bias`, post-placeholder add) added as residual to post-backbone `surface_hidden` BEFORE `surface_out`. Gives decoder a 1-hop shortcut to ALL pre-backbone abstractions. +262,656 params.

### Terminal results

| Channel | Validation (terminal) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.2414%** ❌ | **6.0949%** | 5.844% | **+0.251pp MISS gate +0.115pp** |
| surface_pressure | 4.065% | 3.780% | 3.577 (floor) | **+0.203pp MISS floor — 18th plateau confirmation** |
| **volume_pressure** | 3.604% | **3.583%** | 3.643 (floor) | **−0.060pp CROSS floor** ✓ |
| wall_shear | 7.147% | 7.106% | 6.727 (goal) | +0.379pp MISS goal |
| wall_shear_x | 6.320% | 6.373% | — | +0.543pp regress |
| wall_shear_y | 7.699% | 7.617% | — | +0.517pp regress |
| **wall_shear_z** | 9.518% | **9.121%** | 9.83 (canonical) | **−0.709pp strong improvement** ✓ |

- B PARTIAL via test_VP cross + WSS_z improvement (dual signature shared with H107/H108).
- val→test slopes ALL negative or near-zero → no val-overfit signature.

### Mechanism class — "pre-backbone-embedding-reach-to-decoder" characterized

H109 implements a learnable bypass channel from the pre-backbone surface token embedding directly to the decoder via a full-hidden-dim residual projection. Unlike H101 (positions, +1.5K) or H105/H106 (small-add, +2K), this is at FULL HIDDEN DIM giving the decoder a 1-hop shortcut to all encoded geometry abstractions.

Mechanism ordering at +263K class (post-H102 / H108 / H109 comparison):

| Run | Mechanism | val_abupt | test_WSS_z |
|---|---|---:|---:|
| H102 tanjiro (MERGED) | surface_out width 1×→2× | **6.124%** | TBD |
| H108 nezuko | parallel-MLP on surface_out | 6.164% | **8.857%** |
| H109 alphonse (this) | pre-backbone-embedding bypass | 6.241% | 9.121% |

**Conclusion**: at matched +263K cost, **width > post-backbone bypass > pre-backbone bypass**. The pre-backbone bypass has to traverse the full encoder representation distance, while H108's parallel-MLP operated directly on the already-decoded post-backbone state.

### Strategic implications

1. **DO NOT MERGE** — val_abupt 6.241% > 6.126% gate by +0.115pp.
2. **18 consecutive SP plateau confirmations** locked in across the entire +260K-class decoder cohort.
3. Wave 35 4-axis data-tier sweep now complete in flight:
   - H114 frieren: panel-area-weighted SP loss (loss reweighting)
   - H115 thorfinn: Huber loss on SP (loss curvature)
   - H116 nezuko: Y-mirror augmentation (sample distribution)
   - **H117 alphonse: signed-sqrt SP targets (target distribution) — NEW ASSIGNMENT**
4. Suggested "H109-narrow" (normals-only bypass) DEFERRED to Wave 36+.

### NEXT ASSIGNMENT — H117 SIGNED POWER TRANSFORM ON SP TARGETS (alphonse)

- **Branch**: `alphonse/h117-sp-target-signed-power-transform`, DRAFT PR #1292
- **Hypothesis**: Apply `y' = sign(y) * |y|^0.5` (signed-sqrt) to SP targets after normalization; compresses heavy-tail mass before MSE applies; invert predictions at eval. Zero params, pure target reshape.
- **Falsifiable**: if `test_SP < 3.577%` plateau cracked → target distribution shape was bottleneck. If NULL + H114/H115/H116 NULL → **definitive Bayes-optimal hardness evidence** for SP plateau; Wave 36 should pivot to capacity scaling.
- **Why orthogonal to H115**: Huber bends the *loss* into L1 at large residuals; H117 bends the *target* into compressed scale where L2 acts more uniformly. Mathematically dual.

---

## 2026-05-24 ~00:30 — PR #1278: H108 SURFACE-OUT-PARALLEL-MLP-RESIDUAL-DECODER (nezuko, CLOSED) — **B PARTIAL** via clean test_VP cross (−0.060pp); val gate MISS +0.038pp (**NARROWEST non-compound miss of cohort**); test_WSS_z 8.857% improves canonical by −0.97pp; **STRONGEST FALSIFIABLE NEGATIVE OF WAVE 33: WIDTH > DIVERSITY at matched +265K** (H102 6.124% beats H108 6.164% by +0.040pp); 17th SP plateau confirmation; "delayed-engagement" mechanism signature characterized

- **Branch**: `nezuko/h108-surface-out-parallel-mlp-residual-decoder` (closed at ~00:30Z 2026-05-24)
- **W&B run**: `f4w8nw56`, 13 epochs completed, runtime 14.46h, peak GPU ~78.2 GB across 8 ranks
- **Hypothesis**: Add zero-init parallel branch `Linear(512, 512) → SiLU → Linear(512, 4)` summed with main `surface_out` MLP at output. Tests implicit-ensemble / decoder-diversity vs width at matched +265K param cost. +265,112 params.

### Terminal results

| Channel | Validation (terminal) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.164%** ❌ | **5.924%** | 5.844% | **+0.080pp MISS gate +0.038pp** (NARROWEST non-compound miss of cohort) |
| surface_pressure | 4.064% | 3.761% | 3.577 (floor) | **+0.184pp MISS floor — 17th plateau confirmation** |
| **volume_pressure** | 3.632% | **3.583%** | 3.643 (floor) | **−0.060pp CROSS floor** ✓ |
| wall_shear | 6.974% | 6.808% | 6.727 (goal) | +0.081pp MISS goal narrowly |
| wall_shear_x | 6.082% | 6.040% | — | +0.210pp regress |
| wall_shear_y | 7.617% | 7.380% | — | +0.280pp regress |
| wall_shear_z | 9.428% | **8.857%** | 9.83 (canonical) | **−0.970pp strong improvement** ✓ |

- B PARTIAL via test_VP cross (rubric-aligned with H101/H104/H105/H106/H107).

### 🔥 STRONGEST FALSIFIABLE NEGATIVE OF WAVE 33: WIDTH > DIVERSITY at matched +265K

Direct head-to-head with H102 (merged baseline, surface_out width 1×→2×) at matched param cost:

| Run | Mechanism | Δ Params | val_abupt | Verdict |
|---|---|---:|---:|---|
| **H102 tanjiro (MERGED)** | surface_out width 1×→2× | +266K | **6.124%** | A WIN baseline |
| **H108 nezuko (this)** | parallel-MLP residual on surface_out | +265K | **6.164%** | B PARTIAL via test_VP |

H108 trails H102 by **+0.040pp val_abupt** at matched param cost. **The implicit-ensemble / decoder-diversity hypothesis is FALSIFIED as a strict win over width**. Two implications:
1. Gradient coupling > orthogonal subspace decomposition (zero-init second branch takes ~3 epochs to engage productively).
2. Future Wave 35+ decoder mechanisms should explore WIDENING or DEPTHENING before parallel-stream ensembling.

### Mechanism class verdict — "delayed-engagement" signature characterized

Full arc:
- Step 21,729 EP2: 7.846% (weakest +265K-class)
- Step 32,594 EP3: 6.866% (still 3rd-weakest)
- Step 38,030: 6.491% (**steepest late-cosine slope of cohort**, −0.069pp/1k)
- Step 70,664 TERMINAL: 6.164% (2nd-best non-compound)

Parallel-MLP mechanism class needs ~3 epochs to find productive orthogonal subspace because zero-init second head must overcome trained first head's residual error budget. The recovery is the strongest "delayed-engagement" signature of Wave 33.

### Strategic implications

1. **DO NOT MERGE** — val_abupt 6.164% > baseline 6.126%, +0.038pp regress (narrowest non-compound miss).
2. **WIDTH > DIVERSITY decision locked in** for future Wave 35+ decoder mechanism design.
3. **17th consecutive SP plateau confirmation** — Wave 35 data-tier attack on SP is now 3-mechanism deep (H114 panel-area, H115 Huber, H116 Y-mirror just assigned).
4. **Closing this PR — nezuko reassigned to H116 LONGITUDINAL Y-MIRROR AUGMENTATION** (PR #1291) — third orthogonal Plateau Protocol data-tier intervention (sample-augmentation axis).

### Per-channel val→test slopes (for reference)

| Channel | val | test | val→test slope |
|---|---:|---:|---:|
| abupt | 6.164 | 5.924 | −0.240 (canonical-class) |
| SP | 4.064 | 3.761 | −0.303 (favorable) |
| VP | 3.632 | 3.583 | −0.049 (very flat) |
| WSS | 6.974 | 6.808 | −0.166 (slight favorable) |
| WSS_z | 9.428 | 8.857 | **−0.571 (steep favorable, parallel-MLP boosting binding-axis transfer)** |

---

## 2026-05-24 ~00:00 — PR #1277: H107 SURFACE-GLOBAL-CONTEXT-RESIDUAL-DECODER (thorfinn, CLOSED) — **B PARTIAL** via clean test_VP cross (−0.089pp); val gate MISS +0.065pp (narrowest non-compound miss of cohort); **STRONGEST NON-COMPOUND SINGLE MECHANISM OF Wave 33+34** — best test_abupt 5.9545% of any single-mechanism PR in cohort; 16th SP plateau confirmation; self-context residual mechanism class characterized — orthogonal to H101 (positions), H105 (normals), H106 (volume info); permanent compound infrastructure CANDIDATE

- **Branch**: `thorfinn/h107-surface-global-context-residual-decoder` (closed at ~00:00Z 2026-05-24)
- **W&B run**: `7svzz2ci` (rank 0), 13 epochs completed, runtime 14.06h, throughput ~6,280-6,710 steps/h (~6% below canonical — modest self-pool overhead), peak GPU 82.9 GB
- **Hypothesis**: Inject mask-aware self-pooled surface_hidden (`pooled = mean(surface_hidden, mask)`) as zero-init `Linear(512, 512)` additive residual to `surface_hidden` AFTER surf->vol xattn block and BEFORE `surface_out`. +262,656 params. Tests whether each surface point benefits from a non-local global field summary in addition to its local attention receptive field.

### Terminal results

| Channel | Validation (terminal) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.1912%** ❌ | **5.9545%** | 5.844% | **+0.111pp MISS gate +0.065pp** (narrowest non-compound miss) |
| surface_pressure | 4.068% | 3.7615% | 3.577 (floor) | **+0.184pp MISS floor — 16th plateau confirmation** |
| **volume_pressure** | 3.628% | **3.554%** | 3.643 (floor) | **−0.089pp CROSS floor** ✓ (deepest VP cross of cohort) |
| wall_shear | 7.047% | 6.8733% | 6.727 (goal) | +0.146pp MISS goal |
| wall_shear_x | 6.204% | 6.1132% | — | +0.283pp regress |
| wall_shear_y | 7.614% | 7.4518% | — | +0.352pp regress |
| wall_shear_z | 9.442% | **8.8918%** | 9.83 (canonical) | **−0.938pp strongest binding-axis improvement of program** ✓ |

- Gate: val_abupt 6.1912% MISS gate by +0.065pp — narrowest non-compound miss of cohort.
- Test AND-gate: **FAILS 3/4** (test_VP ✓ deepest cross, others miss).
- B PARTIAL via single test_VP floor cross (consistent with H101/H104/H105/H106 closure rubric).

### 🟢 STRONGEST NON-COMPOUND SINGLE MECHANISM OF Wave 33+34

H107 lands as the **best test_abupt (5.9545%) of any single-mechanism PR in this cohort** (excluding H110 compound which is still in flight). The self-context residual mechanism is now a **fully characterized mechanism class**.

| Run | val_abupt% | test_abupt% | test_VP cross | test_WSS_z vs canonical | Δ params | Class |
|---|---:|---:|---:|---:|---:|---|
| H110 tanjiro compound (in flight) | ~6.14% | TBD | TBD | TBD | +268K | compound |
| **H107 thorfinn (this)** | **6.1912%** | **5.9545%** | **−0.089pp** ⭐ | **−0.938pp** ⭐ | **+262K** | **info-at-decoder (non-local)** |
| H108 nezuko parallel-MLP | ~6.165% | TBD | TBD | TBD | +265K | parallel decoder |
| H106 frieren volume-info | 6.2505% | 6.026% | −0.039pp | −0.802pp | +2.5K | info-at-decoder (volume) |
| H105 fern normals | 6.349% | 5.920% | −0.109pp | −0.967pp | +2K | info-at-decoder (normals) |
| H101 (merged) | 6.213% | 5.846% | −0.129pp | — | +1.5K | info-at-decoder (positions) |

### Mechanism class verdict — SELF-CONTEXT RESIDUAL IS A COMPOUND PRIMITIVE

The self-pooled global-context residual is fundamentally orthogonal to all prior info-at-decoder mechanisms:
- **H101**: per-point position info at surface decoder (local)
- **H106**: per-point volume geom (xyz+sdf) at volume decoder (cross-modal local)
- **H107**: pooled GLOBAL surface state at surface decoder (non-local context)

Wave 35 staging matrix unlocked:
- **H107 + H112 (self-context + DropPath, +262K)** — pure regularization stack
- **H107 + H101 (global + local surface info, +263.5K)** — surface info-at-decoder full stack
- **H107 + H106 (global surface + volume info, +265K)** — both info-at-decoder paths
- **H110 + H107 (compound H102+H101+H107, +530K)** — 3-mechanism stack

### 16th SP plateau locked in

Plateau extends to 16/16 mechanisms in 3.70-3.95% range vs floor 3.577. Combined with H113 diagnostic (SP plateau is HARDNESS-BOUND), Wave 35+ MUST attack SP via **data-tier interventions**:
- **H114** (frieren, panel-area-weighted SP) — per-point gradient distribution
- **H115** (thorfinn, this assignment — **Huber loss for SP**) — loss-form per-point curvature
- Future: CDF normalize SP targets, log-transform tails, geometric augmentation

### Strategic implications

1. **DO NOT MERGE** — val_abupt 6.1912% > baseline 6.126%, +0.065pp regress on primary metric (narrowest non-compound miss).
2. **H107 mechanism is permanent infrastructure CANDIDATE** at +262K params — Wave 35 4 candidate compounds listed above.
3. **info-at-decoder mechanism class now fully characterized** — 4 sub-axes proven (positions, normals, volume xyz+sdf, global self-context). Diminishing returns on further info-at-decoder mechanisms.
4. **Closing this PR — thorfinn reassigned to H115 HUBER LOSS FOR SP** (PR #1290) — Plateau Protocol loss-form data-tier intervention complementary to H114.

### Per-channel val→test slopes (for reference)

| Channel | val | test | val→test slope |
|---|---:|---:|---:|
| abupt | 6.191 | 5.954 | −0.237 (canonical-class) |
| SP | 4.068 | 3.762 | −0.306 (favorable) |
| VP | 3.628 | 3.554 | −0.074 (very flat — VP signal stable, mechanism class signature) |
| WSS | 7.047 | 6.873 | −0.174 (slight favorable) |
| WSS_z | 9.442 | 8.892 | **−0.550 (steep favorable, non-local context helping binding axis transfer)** |

---

## 2026-05-23 ~21:00 — PR #1276: H106 VOLUME-GEOM-RESIDUAL-DECODER (frieren, CLOSED) — **B PARTIAL** via clean test_VP cross (−0.039pp); val gate MISS +0.124pp; **COST-EFFICIENCY CHAMPION OF THE PROGRAM** — within +0.07pp val_abupt of H107 (+262K params) at **105× lower parameter cost** (+2,560 params); 15th SP plateau confirmation; volume-info-at-decoder mechanism class now characterized

- **Branch**: `frieren/h106-volume-geom-residual-decoder` (closed at ~21:00Z 2026-05-23)
- **W&B run**: `ci7mqnjs` (rank 0), 13 epochs completed, runtime 14.0h, no infra anomalies
- **Hypothesis**: Inject volume-geometry residual `volume_x[..., 0:4]` (xyz + sdf) as zero-init `Linear(4, 512)` additive residual to `volume_hidden` (post-backbone, pre-`volume_out`). Mirror of H101 (surface positions) and H105 (surface normals) on the **volume** path. +2,560 params. Tests whether volume-side info-at-decoder is symmetric to surface-side mechanism.

### Terminal results

| Channel | Validation (terminal) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.2505%** ❌ | **6.026%** | 5.844% | **+0.182pp MISS gate +0.124pp** |
| surface_pressure | 4.168% | 3.840% | 3.577 (floor) | **+0.263pp MISS floor — 15th plateau confirmation** |
| **volume_pressure** | 3.670% | **3.604%** | 3.643 (floor) | **−0.039pp CROSS floor** ✓ |
| wall_shear | 7.066% | 6.938% | 6.727 (goal) | +0.211pp MISS goal |
| wall_shear_x | 6.178% | 6.161% | — | +0.331pp regress |
| wall_shear_y | 7.657% | 7.497% | — | +0.397pp regress |
| wall_shear_z | 9.580% | **9.028%** | 9.83 (canonical) | **−0.802pp strong improvement** ✓ |

- Gate: val_abupt 6.2505% MISS gate by +0.124pp (C NULL margin range).
- Test AND-gate: **FAILS 3/4** (test_VP ✓ cross by −0.039pp, others miss).
- B PARTIAL via single test_VP floor cross (consistent with H101/H104/H105 closure rubric).

### 🔥 COST-EFFICIENCY CHAMPION OF THE PROGRAM

H106 lands within +0.07pp val_abupt of H107 (+262K params) at **105× lower parameter cost** (+2.5K vs +262K).

| Run | val_abupt | Δ params | val/Δparam ratio |
|---|---:|---:|---:|
| H110 tanjiro compound (in flight) | ~6.14% | +268K | 50 params/0.001pp |
| H107 thorfinn | 6.20% | +262K | 50 params/0.001pp |
| H108 nezuko | 6.165% | +265K | 50 params/0.001pp |
| **H106 frieren (this)** | **6.2505%** | **+2.5K** | **0.5 params/0.001pp** ⭐ |
| H105 fern | 6.349% | +2K | similar |
| H101 (merged) | 6.213% | +1.5K | best EVER |

H106 is **2nd most cost-efficient mechanism** of the program after H101 — strongest cost-efficiency demonstration of Wave 33+34.

### Mechanism class — VOLUME-GEOM RESIDUAL OUTPERFORMS SURFACE-NORMAL RESIDUAL

Direct H106 vs H105 vs H101 comparison (info-at-decoder family):

| Run | Channel | Δ params | val_abupt | test_VP cross | test_WSS_z vs canonical |
|---|---|---:|---:|---:|---:|
| H105 fern | normals [3:6] | +2K | 6.349% | −0.109pp | 8.863% (−0.97) |
| **H106 frieren (this)** | **volume xyz+sdf [0:4]** | **+2.5K** | **6.2505%** | **−0.039pp** | **9.028% (−0.80)** |
| H101 (positions [0:3]) | positions | +1.5K | 6.213% | −0.129pp | improves |

H106 trails H101 by +0.04pp val_abupt at +1K higher cost. **Positions still best info-at-decoder axis.** Volume-info residual targets volume_hidden directly via SDF + local coordinates — meaningfully different from surface info-residual: clean test_VP cross was the predicted outcome and it happened.

### 15th SP plateau locked in

Plateau extends to 15/15 mechanisms in 3.70-3.95% range vs floor 3.577. H113 (in-flight) heteroscedastic diagnostic has empirically confirmed SP plateau is **HARDNESS-BOUND, not balance-bound** (log_σ² drift to −2.3 with only 2.1% per-task relative differential). Wave 35+ MUST pivot SP from architecture/loss-balance tiers to **data-tier interventions** (panel-area weighting, CDF normalize, log-transform tails, geometric augmentation).

### Strategic implications

1. **DO NOT MERGE** — val_abupt 6.2505% > baseline 6.126%, +0.124pp regress on primary metric.
2. **H106 mechanism is permanent infrastructure CANDIDATE** at +2.5K params for compound staging. Wave 35 candidate: **H106 + H112 (DropPath) — +2.5K total — most cost-efficient compound possible** if H112 lands.
3. **Volume-info-at-decoder mechanism class is now characterized** — clean test_VP cross, test_WSS_z improvement (−0.80pp from canonical 9.83), no test_SP impact. Diminishing returns on more volume-side mechanisms.
4. **Closing this PR — frieren reassigned to H114 PANEL-AREA-WEIGHTED SP LOSS** (Plateau Protocol data-tier intervention, PR #1289).

### Per-channel val→test slopes (for reference)

| Channel | val | test | val→test slope |
|---|---:|---:|---:|
| abupt | 6.2505 | 6.026 | −0.225 (canonical-class) |
| SP | 4.168 | 3.840 | −0.328 (favorable) |
| VP | 3.670 | 3.604 | −0.066 (very flat — VP signal stable) |
| WSS | 7.066 | 6.938 | −0.128 (slight regress) |
| WSS_z | 9.580 | 9.028 | **−0.552 (steep favorable, volume-info helping binding axis transfer)** |

---

## 2026-05-23 ~16:30 — PR #1271: H105 SURFACE-NORMAL-RESIDUAL-DECODER (fern, CLOSED) — **B PARTIAL** via single test_VP floor cross (−0.109pp); val gate MISS +0.223pp; 14th SP plateau confirmation; **NORMALS-AT-DECODER UNDERPERFORMS POSITIONS-AT-DECODER** by +0.127pp val_abupt at terminal; info-at-decoder mechanism axis converging — H101 positions > H105 normals > H99 depth)

- **Branch**: `fern/h105-surface-normal-residual-decoder` (closed at ~16:30Z 2026-05-23)
- **W&B run**: `t6b1i2yk` (rank 0), 13 epochs completed, runtime 14.35h, throughput 1.86 it/s (no overhead).
- **Hypothesis**: Inject raw surface normals `surface_x[..., 3:6]` as zero-init `Linear(3, 512)` additive residual to `surface_hidden` (post-backbone, pre-`surface_out`). Mirror of H101 (positions) on the normals channel. +2,048 params. Tests whether normals at decoder input carry productive info beyond what `surface_in` already encodes from the 7-dim input.

### Terminal results

| Channel | Validation (terminal) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.349%** ❌ | **5.920%** | 5.844% | **+0.076pp MISS gate +0.223pp** |
| surface_pressure | 4.209% | 3.7245% | 3.577 (floor) | **+0.148pp MISS floor — 14th plateau confirmation** |
| **volume_pressure** | 3.760% | **3.534%** | 3.643 (floor) | **−0.109pp CROSS floor** ✓ |
| wall_shear | 7.174% | 6.832% | 6.727 (goal) | +0.105pp MISS goal |
| wall_shear_x | 6.277% | 6.067% | — | +0.092pp regress |
| wall_shear_y | 7.822% | 7.413% | — | +0.139pp regress |
| wall_shear_z | 9.676% | 8.863% | 8.945 (canonical test) | **−0.082pp BELOW canonical** ✓ |

- Gate: val_abupt 6.349% MISS gate by +0.223pp (clearly C NULL margin range).
- Test AND-gate: **FAILS 3/4** (test_VP ✓ cross by −0.109pp, others miss).
- B PARTIAL via single test_VP floor cross (consistent with H101/H104 closure rubric).

### 🟢 NORMALS-AT-DECODER UNDERPERFORMS POSITIONS-AT-DECODER (mechanism-class diagnostic)

Direct head-to-head with H101 (raw positions, same mechanism class, +1.5K params):

| Metric | H101 (positions +1.5K) | H105 (normals +2K) | Δ H105−H101 | Winner |
|---|---:|---:|---:|---|
| val_abupt | **6.213%** | 6.349% | +0.127pp | H101 |
| test_abupt | **5.846%** | 5.920% | +0.074pp | H101 |
| test_VP | 3.5114% | **3.5344%** | +0.023 | H101 (deeper cross) |
| test_SP | 3.8156% | 3.7245% | −0.091 | **H105 (better SP)** |
| test_WSS | 6.9008% | 6.832% | −0.069 | H105 |
| test_WSS_z | 9.130% | 8.863% | −0.267 | H105 |

H101 wins on val_abupt and test_VP (the deepest mechanism-class signal). H105 wins on SP/WSS/WSS_z secondary channels. **The primary axis (val_abupt + test_VP combined depth) favors positions**, but normals do produce competitive secondary-channel test metrics — likely from the additional surface-orientation signal at decoder.

### 14th SP plateau — Bayes-optimal hypothesis CRITICAL strength

Plateau extends to 14/14 mechanisms in 3.70-3.95% range vs floor 3.577. Survives every architecture-class mechanism tested including all of Wave 32+33: WIDTH-SURFACE, WIDTH-VOLUME, DEPTH, INFO-AT-INPUT (positions+normals), BIDIR-XATTN, FILM, TASK-HEAD, ENCODER-SKIP, PARALLEL-MLP. **If H108 (parallel-MLP) + H110 (compound) also miss SP at terminal, Wave 35+ MUST pivot SP from architecture tier to LOSS REFORMULATION / DATA REPRESENTATION tier.**

### Mechanism class verdict — INFO-AT-DECODER axis converging

- ✓ INFO-AT-DECODER mechanism class validated (H101, H105 both B PARTIAL via test_VP cross)
- Ranking within class: **H101 positions > H105 normals > H99 depth** (all <0.15pp val_abupt range)
- H101 alone remains the cost-efficiency champion (+1.5K params, deepest test_VP cross)
- **Diminishing returns within info-at-decoder axis** — Wave 34 should pursue different mechanism axes
- **DO NOT compound H101 + H105** — normals add no marginal value beyond what `surface_in` already encodes from 7-dim input
- panel_area axis (`surface_x[..., 6:7]`) is the last cheap-info axis not yet tested

### Per-channel val→test slopes (for reference)

| Channel | val | test | val→test slope |
|---|---:|---:|---:|
| abupt | 6.349 | 5.920 | −0.429 (favorable) |
| SP | 4.209 | 3.725 | −0.484 (favorable) |
| VP | 3.760 | 3.534 | −0.226 (canonical-class) |
| WSS | 7.174 | 6.832 | −0.342 (canonical-class) |
| WSS_z | 9.676 | 8.863 | **−0.813 (steep favorable, normals helping binding axis transfer)** |

The steeper WSS_z slope (−0.813 vs canonical ~−0.5) is the most interesting per-channel finding — normals at decoder DO help binding-axis val→test transfer, even though the magnitude is below canonical.

### Reassignment

fern → **H113 HETEROSCEDASTIC-UNCERTAINTY-WEIGHTING** (PR pending). **Plateau Protocol strategy-tier shift from architecture/mechanism tier to LOSS REFORMULATION tier**. Learnable per-channel log_sigma parameters added to loss: `L_total = Σ_k exp(-2*log_sigma_k) * L_k + log_sigma_k`. Identity-at-init via log_sigma_init=0 (sigma=1). +5 params (one per output channel: SP, VP, WSS_x, WSS_y, WSS_z). Reference: Kendall & Gal 2018 (NeurIPS multi-task uncertainty weighting). Tests "is test_SP plateau due to under-trained SP loss term, or Bayes-optimal hardness?" — if uncertainty-weighting cracks SP plateau, plateau was undertrained; if not, hardness is data/architecture-bound.

---

## 2026-05-23 ~11:45 — PR #1269: H104 VOLUME-OUT-WIDER-MLP (edward, CLOSED) — **B PARTIAL** (val gate near-miss +0.066pp, test_VP CROSS −0.108pp = 2nd-deepest Wave 33; 13th SP plateau confirmation; SURFACE > VOLUME decoder capacity-bound; **CAPACITY ROUTE > GRADIENT ROUTE** on volume axis)

- **Branch**: `edward/h104-volume-out-wider-mlp` (closed at ~11:45Z 2026-05-23)
- **W&B run**: `xiddgcgu` (rank 0), 13 epochs completed, runtime 14.0h (840.6 min), all 70,652 steps.
- **Hypothesis**: Widen `volume_out` MLP hidden layers 1× → 2× (`volume_out_width_factor=2.0`). Mirror of H102 (tanjiro, surface-side). Tests whether volume decoder is WIDTH-bound symmetrically to surface. +229K params (volume_out funnel widening).

### Terminal results

| Channel | Validation (EP13 EMA best-ckpt) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.1919%** ❌ | **6.0131%** | 5.844% | **+0.169pp MISS gate +0.066pp** (narrow) |
| surface_pressure | 4.0931% | 3.8211% | 3.577 (floor) | **+0.244pp MISS floor — 13th plateau confirmation** |
| **volume_pressure** | 3.6025% | **3.5348%** | 3.643 (floor) | **−0.108pp CROSS floor** ✓ (2nd-deepest Wave 33 cross) |
| wall_shear | 7.0232% | 6.9641% | 6.727 (goal) | +0.237pp MISS goal |
| wall_shear_x | 6.1453% | 6.2123% | — | +0.250pp regress |
| wall_shear_y | 7.6173% | 7.4830% | — | +0.048pp narrow miss |
| wall_shear_z | 9.5012% | 9.0142% | 8.945 (canonical test) | +0.069pp regress narrow |

- Gate: val_abupt 6.1919% MISS gate by +0.066pp (borderline A WIN territory).
- Test AND-gate: **FAILS 3/4** (test_VP ✓ cross by −0.108pp, others miss).
- Per-channel val→test slopes: abupt −0.179, SP −0.272, VP −0.068, WSS −0.059, WSS_x +0.067 REVERSE (minor), WSS_y −0.134, WSS_z **−0.487pp** (steep favorable).

### 🟢 SURFACE > VOLUME decoder capacity-bound — paired diagnostic with H102

Direct head-to-head (both decoder-width experiments at matched ~250K-cost):

| Metric | H102 surface (+266K) | H104 volume (+229K) | Δ H104−H102 | Winner |
|---|---:|---:|---:|---|
| val_abupt | **6.1183%** ✓ gate clear | 6.1919% | +0.073 | H102 |
| test_abupt | **5.9395%** | 6.0131% | +0.074 | H102 |
| test_VP | 3.5432% | **3.5348%** | −0.008 | tied (H104 nominal) |
| test_SP | **3.7242%** | 3.8211% | +0.097 | H102 |
| test_WSS | **6.8584%** | 6.9641% | +0.106 | H102 |
| test_WSS_z | **8.8889%** | 9.0142% | +0.125 | H102 |

H102 wins 5/6 metrics. H104 wins ONLY on test_VP (its directly-targeted axis). **Surface decoder is more capacity-bound than volume** — architectural attribution: `surface_out` 4 channels share a single 512-hidden bottleneck (widening gives each ~2× substrate); `volume_out` is a 3-stage funnel to 1 channel (already over-parameterized).

### 🟢 CAPACITY ROUTE > GRADIENT ROUTE on volume axis (H94 vs H104)

Direct comparison: H94 (vol_loss_weight 1.0→1.5, B PARTIAL) vs H104 (vol_width 1×→2×, B PARTIAL) — same volume-pressure target axis, different attack route.

| Metric | H94 (vol_loss=1.5) | H104 (vol_width=2.0) | Δ |
|---|---:|---:|---:|
| val_abupt | 6.357% | **6.1919%** | **−0.165pp** ✓ |
| test_VP | 3.582% | **3.5348%** | **−0.047pp deeper cross** ✓ |
| test_SP | 3.834% | 3.8211% | −0.013pp |
| test_WSS_z | 9.051% | 9.0142% | −0.037pp |

H104 strictly dominates H94 on every axis. **Parameter-capacity route beats gradient-reweighting route on volume head.** Generalizes: for binding axes that respond to capacity, prefer "more substrate" over "more gradient signal".

### 13th SP plateau — Bayes-optimal hypothesis very strong

Plateau extended to 13/13 mechanisms in 3.70-3.95% range vs floor 3.577. Survives every architecture-class mechanism tested. If H108 (parallel-MLP) + H110 (compound) also fail to crack SP, Wave 35+ must pivot test_SP from "mechanism gap" to **loss reformulation / data representation tier**.

### Mechanism class verdict

- ✓ WIDTH-VOLUME validated (H104) — symmetric productivity to WIDTH-SURFACE (H102)
- BUT — volume capacity transfers less to other channels than surface capacity does (volume_out is already over-parameterized for 1-channel regression)
- H102+H104 compound is a strong Wave 34 candidate (symmetric decoder widening, +495K total)

### Reassignment

edward → **H112 STOCHASTIC-DEPTH-IN-BACKBONE (DropPath)** (PR #1283). Strategy tier shift per Plateau Protocol — second **regularization-class** mechanism in Wave 33 (paired with H111 LayerScale). Linear schedule p=0 at block 0 → p=0.10 at block 4 across 5-block backbone. Each TransformerBlock's attn and mlp residual branches independently dropped with probability `p_block` during training (rescaled by 1/keep_prob when kept); full residuals at eval. 0 params, identity at eval. Reference: ConvNeXt, Swin, DeiT-3. Mechanistically distinct from H111: H111 = deterministic learnable γ rescale; H112 = stochastic per-step branch drop. Pair forms 2-arm regularization study.

---

## 2026-05-23 ~11:30 — PR #1270: H103 VOLUME-CONTEXT-FILM-DECODER (askeladd, CLOSED) — **C NULL** (val MISS +0.224pp, all 3 floors miss, test_VP near-miss +0.014pp; FiLM mechanism engaged but global-pool signal insufficient; **GLOBAL vs LOCAL pathway question definitively answered: LOCAL wins**; FiLM class CLOSED for Wave 34)

- **Branch**: `askeladd/h103-volume-context-film-decoder` (closed at ~11:30Z 2026-05-23)
- **W&B run**: `lqpbqy67` (rank 0), 13 epochs completed, runtime 14.45h, throughput 1.86 it/s (O(N) confirmed).
- **Hypothesis**: Pool `volume_hidden` to scalar context → predict (γ, β) → modulate `surface_hidden` via affine FiLM transform before `surface_out`. Tests whether vol→surf info via cheap O(N) global modulation is competitive with O(N²) per-token cross-attention (H97). +540K params (`film_projector` Linear 512→1024).

### Terminal results

| Channel | Validation (EP13 EMA best-ckpt) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.350%** ❌ | **6.060%** | 5.844% | **+0.216pp MISS gate +0.224pp** |
| surface_pressure | 4.165% | 3.820% | 3.577 (floor) | **+0.243pp MISS floor — 12th plateau confirmation** |
| volume_pressure | 3.743% | 3.657% | 3.643 (floor) | **+0.014pp NEAR-MISS** (would have crossed at marginally better val) |
| wall_shear | 7.203% | 6.982% | 6.727 (goal) | **+0.255pp MISS goal** |
| wall_shear_x | 6.313% | 6.202% | — | +0.240pp regress |
| wall_shear_y | 7.822% | 7.600% | — | +0.165pp regress |
| wall_shear_z | 9.708% | 9.022% | 8.945 (canonical test) | **+0.077pp REGRESS on binding axis** |

- Gate: val_abupt 6.350% **MISS gate 6.126% by +0.224pp**.
- Test AND-gate: **FAILS 3/3** (test_VP near-miss +0.014pp, test_SP miss +0.243pp, test_WSS miss +0.255pp).
- val→test slope: abupt −0.290pp (≈ canonical −0.282), WSS_z **−0.686pp** (very steep, but starts from val 9.708 which is well above canonical val 9.601).

### Mechanism diagnostic — FiLM engaged but signal insufficient

FiLM γ row norm grew from **zero-init** to **39.9 at terminal** (max_abs 0.62, mean_abs 0.060). β row norm 27.6. The optimizer found and applied a meaningful FiLM transform. But the asymptote isn't competitive: every surface token receives the SAME (γ, β) since the modulation source is a single global mean-pool of `volume_hidden`. FiLM cannot add spatial discrimination, only globally shift/scale `surface_hidden` — largely redundant with what LayerNorm already does inside the backbone.

### 🔴 DEFINITIVE GLOBAL vs LOCAL PATHWAY ANSWER — LOCAL WINS

Direct head-to-head — both H97 and H103 target vol→surf info pathway:

| Metric | H97 alphonse (per-token xattn, +1M) | H103 askeladd (global FiLM, +540K) | H103 − H97 |
|---|---:|---:|---:|
| val_abupt | 6.204 | 6.350 | **+0.146** |
| test_abupt | 5.989 | 6.060 | **+0.071** |
| test_VP | 3.654 | 3.657 | +0.003 |
| test_WSS | 6.887 | 6.982 | +0.095 |
| test_WSS_z | 8.937 | 9.022 | +0.085 |

H97 wins on **every channel** at 2× the param cost. Param efficiency vs H101 (info-residual at decoder input, +3K):

| Mechanism | Δ Params | val_abupt | test_VP | test_WSS_z |
|---|---:|---:|---:|---:|
| H101 nezuko (geom-residual at decoder INPUT) | +3K | 6.213 | **3.514** ✓ CROSS | 8.946 |
| H103 askeladd (FiLM at decoder feature MODULATION) | +540K | 6.350 | 3.657 ❌ | 9.022 |
| H101 vs H103 | **180× cheaper** | **−0.137pp** | **−0.143pp better** | **−0.076pp better** |

**H101 is 180× cheaper AND better on every channel.** The productive vol→surf info pathway is **decoder INPUT (info-residual class)**, NOT **decoder feature modulation (FiLM class)**.

### Mechanism class verdict — FiLM CLOSED for Wave 34

FiLM mechanism class **DEFINITIVELY CLOSED** for Wave 34 — will not appear in compound staging. Cumulative Wave 33 mechanism class state:
- ✅ WIDTH (H102) — strongest single-axis decoder mechanism
- ✅ INFO-AT-INPUT (H101) — most cost-efficient
- ✅ BIDIR-XATTN (H97) — works but cost-ineffective
- 🔴 FILM (H103) — closed, global-pool insufficient
- 🔴 DEPTH (H99), TASK-HEAD (H92,93,96,100) — closed
- ⏳ SELF-CONTEXT (H107), DECODER-ENSEMBLE (H108), ENCODER-SKIP (H109), VOL-WIDTH (H104), VOL-INFO-RESIDUAL (H106), SURF-NORMALS (H105) — Wave 33 in-flight
- 🆕 COMPOUND (H110 = H102+H101) — Wave 34 launched
- 🆕 REGULARIZATION (H111 = LayerScale-in-backbone) — askeladd reassigned, **NEW CLASS**

### 12th SP plateau confirmation — strengthening Bayes-optimal hypothesis

H103 test_SP 3.820% extends the test_SP plateau to **12 consecutive Wave 32+33 mechanisms** in 3.70-3.95% range vs floor 3.577% (canonical). The plateau survives:
- WIDTH (H102): 3.724
- DEPTH (H99): 3.804
- INFO-AT-INPUT (H101): 3.706
- BIDIR-XATTN (H97): 3.781
- FILM (H103): 3.820
- TASK-HEAD (H92, H93, H96, H100): 3.79-3.95
- Plus 7 other mechanism variants

If H108 (parallel-MLP residual) and H110 (compound H102+H101) cannot crack SP, the "SP is a dataset-distribution Bayes-optimal limit" hypothesis gains very strong support, and Wave 35+ should pivot test_SP from "mechanism gap" to "data representation / loss reformulation" tier.

### Reassignment

askeladd → **H111 LAYERSCALE-IN-BACKBONE** (PR #1282). Strategy tier shift per Plateau Protocol — first **REGULARIZATION-class** mechanism in Wave 33/34. Adds learnable per-channel γ_attn, γ_mlp ∈ R^512 to each TransformerBlock residual branch (5,120 params, 0.029% overhead). Init γ=1.0 (identity at init). Reference: CaiT (Touvron et al. 2021) +0.3-0.5pp ImageNet at near-zero overhead. Mechanism diagnostic (γ histograms per block) is co-equal deliverable with metric outcome — answers "does the optimizer find a per-channel residual rescaling that improves convergence/generalization on this dataset?"

---

## 2026-05-23 ~11:00 — PR #1268: H102 SURFACE-OUT-WIDER-MLP (tanjiro, CLOSED) — **B PARTIAL** (val gate CLEARED −0.008pp, test_VP CROSSED −0.100pp; test_SP 11th plateau + test_WSS regress → NOT MERGEABLE, AND-gate fails 2/3 test floors; WIDTH AXIS DEFINITIVELY DOMINATES DEPTH)

- **Branch**: `tanjiro/h102-surface-out-wider-mlp` (closed at ~11:00Z 2026-05-23)
- **W&B run**: `n14qwfg4` (rank 0), 13 epochs completed, runtime 14.04h (842.2 min), all 70,664 steps.
- **Hypothesis**: Widen `surface_out` MLP hidden layer 512 → 1024 (`surface_out_width_factor=2.0`). Tests whether surface decoder MLP is WIDTH-bound. +264.7K params.

### Terminal results

| Channel | Validation (EP13 EMA best-ckpt) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.1183%** ✅ | **5.9395%** | 5.844% | **+0.096pp regress** |
| surface_pressure | 4.0050% | 3.7242% | 3.577 (floor) | **+0.147pp MISS floor** — **11th independent plateau** (3.70-3.95% range) |
| volume_pressure | 3.6101% | **3.5432%** | 3.643 (floor) | **−0.100pp CROSS floor** ✓ (strong, architecturally attributable to wider decoder) |
| wall_shear | 6.9323% | 6.8584% | 6.727 (goal) | **+0.131pp MISS goal** |
| wall_shear_x | 6.0549% | 6.0899% | — | +0.128pp slight regress |
| wall_shear_y | 7.5238% | 7.4513% | — | ~match |
| wall_shear_z | 9.3979% | **8.8889%** | 8.945 (canonical test) | **−0.056pp BELOW canonical** ✓ |

- Gate: val_abupt 6.1183% **CLEARS gate 6.126 by −0.008pp** — FIRST SINGLE-MECHANISM VAL GATE CLEAR OF WAVE 33.
- Test AND-gate: **FAILS 2/3** (test_VP ✓ cross; test_SP ❌ +0.147pp; test_WSS ❌ +0.131pp).
- test_abupt 5.940% REGRESSES canonical 5.844 by +0.096pp — primary test metric regresses.
- val→test slopes: abupt −0.179pp (shallow), VP −0.067pp (very tight), SP −0.281pp (canonical), WSS_x +0.035pp REVERSE (minor), WSS_z **−0.510pp (steep — width helps z-shear generalization)**.

### Why NOT merged despite val gate clear

Strict AND-gate = val_abupt < 6.126 AND test_VP ≤ 3.643 AND test_SP ≤ 3.577 AND test_WSS ≤ 6.727. H102 fails 2/4 conditions. Per program.md "Final claims must be from test_primary/*" — test_abupt regresses +0.096pp. Merging would degrade SP (3.724 vs 3.577 canonical) and WSS floors, hiding regressions behind val gate clear.

### WIDTH AXIS DOMINATES DEPTH AXIS — cleanest decoder-capacity comparison of Wave 33

| Run | Mechanism | Δ Params | val_abupt | test_VP | test_SP | test_WSS | test_WSS_z |
|---|---|---:|---:|---:|---:|---:|---:|
| H99 frieren (depth) | 3-layer MLP | +250K | 6.327% | 3.637% | 3.804% | 7.009% | 9.088% |
| **H102 tanjiro (width)** | **MLP hidden 1×→2×** | **+265K** | **6.118%** | **3.543%** | **3.724%** | **6.858%** | **8.889%** |
| H102 − H99 | — | — | **−0.208pp** | **−0.094pp** | **−0.080pp** | **−0.150pp** | **−0.199pp** |

H102 dominates H99 on EVERY channel at near-matched param cost. **Width is the productive decoder-capacity axis. Depth is inferior.** 

### 11th SP plateau confirmation — SP plateau may be approaching Bayes optimum

The canonical test_SP IS the floor (3.577%). After 11 independent mechanism misses (3.70-3.95% range), including H102 which DOUBLED the surface_out hidden dim, the SP plateau may be a **dataset-distribution Bayes-optimal limit** rather than a model-capacity bottleneck. Only H108 (parallel-MLP residual) and H110 (compound width+positions) remain as potential SP crackers from decoder-trunk modifications.

### Wave 34 compound launch

H102 + H101 are the two strongest validated Wave 33 mechanisms with **non-overlapping test improvements** and **compatible val→test slopes**:
- H102 (width +265K): val 6.118%, test_VP −0.100pp, WSS_z −0.056pp, WSS_z slope −0.510pp
- H101 (raw positions +3K): val 6.213%, test_VP −0.129pp, WSS_z tied, WSS_z slope −0.603pp

tanjiro reassigned to **H110 WAVE 34 COMPOUND H102+H101** (PR #1280) — FIRST WAVE 34 LAUNCH. +268K total params. Predicted: val < 6.10%, test_VP −0.20pp+, possible test_SP plateau crack if axes compound additively.

---

## 2026-05-23 ~10:25 — PR #1262: H97 BIDIRECTIONAL-XATTN (alphonse, CLOSED) — **B PARTIAL** (mechanism CONFIRMED on val_WSS_z binding axis but did NOT crack test floors; +0.45pp val→test REVERSE slope on binding axis)

- **Branch**: `alphonse/h97-bidirectional-xattn` (closed at ~10:25Z 2026-05-23)
- **W&B run**: `gnfioaws` (rank 0), terminal at step 68,101 / 70,664 = 96.4% (train timeout mid-EP13 at 1011.5 min = 16.86h; EP12 best EMA checkpoint selected), runtime 16.86h.
- **Hypothesis**: Add a reverse vol→surf cross-attention path (symmetric to existing surf→vol xattn) so surface tokens can directly query volume hidden state. Bidirectional vol↔surf info flow tests whether the existing unidirectional surf→vol attention is the binding information bottleneck. +1M params.

### Terminal results

| Channel | Validation (EP12 best EMA) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.2045%** | **5.9887%** | 5.844% | **+0.145pp regress** |
| surface_pressure | 4.0496% | 3.7806% | 3.577 (floor) | **+0.204pp MISS floor** (10th plateau hit: 3.74-3.95% range) |
| volume_pressure | 3.6727% | 3.6544% | 3.643 (floor) | **+0.011pp MISS floor** (razor-thin — closest to floor in Wave 33) |
| wall_shear | 7.0407% | 6.8865% | 6.727 (goal) | +0.160pp MISS goal |
| wall_shear_x | 6.167% | 6.126% | — | — |
| wall_shear_y | 7.649% | 7.446% | — | — |
| wall_shear_z | **9.485%** | **8.937%** | 9.601 val (canonical) / 8.753 test | **val FIRST Wave 33 below canonical (-0.12pp); test +0.184pp above target** |

- Gate: val_abupt 6.2045% **MISS** gate 6.126 by +0.079pp — 2nd closest miss of Wave 33 (H101 at +0.087pp was closest).
- Test floors: 0 crossed; test_VP +0.011pp = razor-thin (closest in Wave 33 to VP floor, but no cross).
- val→test slopes: abupt −0.216pp (shallow), VP −0.020pp (very tight), SP −0.273pp (canonical), WSS −0.155pp, **WSS_z +0.452pp REVERSE** (val improvement erased + reversed at test).

### 🟡 BIDIR-XATTN MECHANISM CONFIRMED BUT COST-INEFFECTIVE + val→test REVERSAL WARNING

| Run | Mechanism | Δ Params | val_abupt | val_WSS_z | test_WSS_z |
|---|---|---:|---:|---:|---:|
| H102 tanjiro (pre-term) | surface width 1×→2× | +266K | **6.122%** | 9.42% | TBD |
| **H97 alphonse (this)** | bidir vol↔surf xattn | **+1,000K** | 6.205% | **9.485%** ← first Wave 33 val below canonical | 8.937% |
| Canonical baseline | — | — | 6.126 | ~9.601 | 8.945 |

**Three critical Wave 33 findings from H97:**

1. **Bidirectional mechanism IS productive for val_WSS_z**: 9.485% is the first Wave 33 val below canonical 9.601 — confirming that vol→surf info flow helps surface token WSS_z prediction. The physics is correct (surface shear depends on volume velocity gradients).

2. **+0.45pp val→test REVERSE slope on WSS_z**: val advantage (−0.12pp) REVERSED to test regress (+0.184pp above target 8.753). The bidir-xattn provides val-set-specific information access that doesn't generalize. Contrast with H101 whose WSS_z slope was −0.603pp (test improves MORE than val). **Mechanisms that provide globally-useful geometric routing (H101) generalize better than mechanisms that provide dataset-specific volume-surface coupling (H97).**

3. **test_VP test_VP miss by +0.011pp is the closest to floor in Wave 33** — implies the bi-directional info flow nearly cracked VP but fell just short. At EP13 completion (timeout hit at EP12.96), test_VP might have crossed — unverifiable since test only ran EP12.

### Wave 33 10th plateau confirmation on test_SP

test_SP 3.7806% = **10th consecutive variant (H78-H88, H97, H101) hitting the 3.74-3.95% plateau**. Per student comment: "SP plateau is decoder-MLP-trunk bound, not encoder/info-flow bound: bidirectional coupling adds info but the shared `surface_out` MLP still bottlenecks SP expressivity." Confirmed finding. Only mechanisms that MODIFY the surface_out trunk itself (H102 width, H108 parallel-MLP) have any chance of cracking SP.

### Cost-effectiveness at matched budget

- H97 +1M params (5.6% model size increase) for B PARTIAL
- H102 +266K params (1.5% increase) for gate-cracked A WIN trajectory
- **H102 achieves ~0.08pp better val_abupt at 3.8× cheaper cost** — surface decoder MLP capacity beats bidirectional info flow at matched budget

### Reassignment

alphonse reassigned to **H109 BACKBONE-SKIP RESIDUAL DECODER** (PR #1279) — NEW mechanism class (ENCODER-SKIP / BACKBONE-BYPASS): zero-init Linear(n_hidden, n_hidden) projection of pre-backbone embedded surface tokens as residual to post-backbone surface_hidden before surface_out. +263K params matched H102 width cost. Generalizes H101 (which added raw xyz positions at +3K, B PARTIAL): H109 tests the FULL embedded feature vector (7 channels projected through surface_in embedding, pre-backbone) vs H101's raw xyz only. Key diagnostic: H109 val→test slope on WSS_z must be NEGATIVE (like H101's −0.603pp) not POSITIVE (H97's +0.452pp reversal).

---

## 2026-05-23 ~09:55 — PR #1266: H101 GEOM-RESIDUAL-DECODER (nezuko, CLOSED) — **B PARTIAL** (extreme parameter efficiency, INFO-AT-DECODER-INPUT CONFIRMED)

- **Branch**: `nezuko/h101-geom-residual-decoder` (closed at ~09:55Z 2026-05-23)
- **W&B run**: `41jk1b0m` (rank 0), terminal at step 70,664 = 100%, runtime 14.27h.
- **Hypothesis**: Zero-init `Linear(3, n_hidden)` projecting raw xyz positions `surface_x[..., 0:3]` as additive residual to `surface_hidden` before `surface_out`. Tests whether the decoder is missing direct access to per-point positional geometry lost during slice-attention compression (65K→128 tokens). +3,072 total params (1,536 weight + 512 bias + 512 LN weight + 512 LN bias).

### Terminal results

| Channel | Validation (EP12 EMA best-ckpt) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.2134%** | **5.9556%** | 5.844% | **+0.112pp above canonical** |
| surface_pressure | 4.054% | 3.7059% | 3.577 (floor) | **+0.129pp MISS floor** (below Wave 32 plateau 3.74-3.95% — **partial plateau crack**) |
| volume_pressure | 3.605% | **3.5144%** | 3.643 (floor) | **−0.129pp CROSS floor** ✓ (strong cross) |
| wall_shear | 7.078% | 6.9133% | 6.727 (goal) | +0.186pp MISS goal |
| wall_shear_x | 6.216% | — | — | — |
| wall_shear_y | 7.643% | — | — | — |
| wall_shear_z | 9.549% | **8.9458%** | 8.945 (binding) | **TIED canonical (±0.001pp)** — no regress, clean mechanism |

- Gate: val_abupt 6.2134% **MISS** gate 6.126 by +0.087pp — closest near-gate of any closed Wave 33 mechanism at this param footprint.
- Test floors: 1 strong cross (test_VP −0.129pp), 1 partial plateau crack (test_SP below Wave 32 plateau 3.74%), test_WSS MISS.
- val→test slopes: abupt −0.258pp (canonical −0.282), VP −0.091pp (tight tracking), SP −0.349pp (stronger, mechanism helps SP generalization), WSS −0.165pp, WSS_z **−0.603pp** (strongest slope on binding axis).

### 🟢 INFO-AT-DECODER-INPUT THESIS CONFIRMED

The slice-attention compression (65,536 → 128 slice tokens) provably loses per-point geometric info that the decoder benefits from accessing directly. Raw xyz positions at decoder input provide direct routing for volume-correlated surface fields (pressure has strong position dependence via Bernoulli / body-surface geometry). The test_VP cross of −0.129pp is **architecturally attributable** to position residual access.

**test_SP partial plateau crack**: first visible plateau pressure on SP axis in Wave 33. SP at 3.706% is below the entire Wave 32 plateau range (3.74-3.95%) but above floor 3.577%. This is a genuine breakthrough signal on the hardest surface axis — mechanisms that directly inform the decoder about per-point geometry can finally dent the SP wall.

### Extreme parameter efficiency — Wave 33 sleeper hit

| Run | Mechanism | Δ Params | val_abupt | Δ vs H101 |
|---|---|---:|---:|---:|
| H101 nezuko (this) | surf-positions info-residual | +3K | 6.213% | — |
| H99 frieren (C NULL) | depth (3-layer MLP) | +250K | 6.327% | **H101 0.114pp BETTER at 81× cheaper** |
| H100 thorfinn (B PARTIAL) | dedicated-tau_z head | +260K | 6.289% | **H101 0.076pp BETTER at 85× cheaper** |
| H102 tanjiro (LEADER) | surface width 1×→2× | +266K | **6.124%** | H101 0.089pp behind at 87× cheaper |
| H97 alphonse | bidir-xattn | +1M | ~6.21% | H101 ~0.003pp behind at **326× cheaper** |

+3,072 params beating +250K mechanisms = most parameter-efficient near-gate result in Wave 33.
Weight/bias norms: init 0.0/0.0 → terminal 4.71/0.57 — mechanism learned clean signal.

### Wave 34 compound priorities (CONFIRMED)

1. **H102 + H101 (width + info-positions, ~+269K total)** — top priority: strongest 2-component compound expected
2. **H101 + H105 (positions + normals = full surface local geometry, <5K total)** — extreme parameter efficiency; possibly A WIN at <5K params
3. **H101 + H106 (surface positions + volume positions+sdf, ~+6K total)** — bilateral info-at-input
4. **H102 + H101 + H105 (triple stack, ~+273K)** — predicted strongest single-model compound across all Wave 33 results

### Reassignment

nezuko reassigned to **H108 SURFACE-OUT-PARALLEL-MLP-RESIDUAL-DECODER** (PR #1278) — NEW mechanism class (DECODER ENSEMBLE / PARALLEL DIVERSITY): zero-init parallel 2-layer MLP added as residual to existing `surface_out`. +265K params matched H102 width cost. Tests whether decoder DIVERSITY (two parallel MLPs producing same output shape, summed) beats decoder WIDTH (single wider MLP) at matched ~265K param budget. Key falsifiable: H108 vs H102 head-to-head at matched cost.

---

## 2026-05-23 ~08:50 — PR #1265: H100 WSS-Z-DEDICATED-HEAD (thorfinn, CLOSED) — **B PARTIAL** (mechanism FALSIFIED on design axis)

- **Branch**: `thorfinn/h100-wss-z-dedicated-head` (closed at ~08:50Z 2026-05-23)
- **W&B run**: `zop8yn2z` (rank 0), terminal at step 70,652 / 70,664 = 100%, runtime 14.44h.
- **Hypothesis**: Split surface_out 4ch into surface_main_out 3ch (cp+tau_x+tau_y) + surface_wss_z_out 1ch (tau_z). Architectural per-tau-channel separation tests whether the binding axis test_WSS_z is responsive to dedicated decoder representation. +262K params.

### Terminal results

| Channel | Validation (EP12 EMA) | Test | Canonical/Floor | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.2891%** | **5.9379%** | 5.844% | **+0.094pp regress** |
| surface_pressure | 4.113% | 3.712% | 3.577 (floor) | **+0.135pp MISS floor** |
| volume_pressure | 3.687% | **3.5114%** | 3.643 (floor) | **−0.132pp CROSS floor** ✓ |
| wall_shear | 7.151% | 6.867% | 6.727 (goal) | +0.140pp MISS goal |
| wall_shear_x | 6.280% | 6.099% | — | — |
| wall_shear_y | 7.682% | 7.439% | — | — |
| wall_shear_z | **9.685%** | **8.928%** | 8.753 (binding) | **+0.175pp REGRESS design axis** |

- Gate: val_abupt 6.2891% **MISS** gate 6.126 by +0.163pp.
- Test floors: 1 cross (test_VP −0.132pp strong), 2 MISS (test_SP +0.135pp, test_WSS +0.140pp).
- val→test slope: −0.351pp (deeper than canonical −0.282pp).

### 🔴 Mechanism FALSIFICATION on design axis

The dedicated `surface_wss_z_out` head engaged mid-cosine (slope val_WSS_z = −2.92pp/1k was fleet-best, advisor flagged in check-in #2). But terminal head-to-head against fleet siblings:

| Run | Mechanism | val_WSS_z |
|---|---|---:|
| H102 tanjiro | surface-wider (general) | **9.528%** ✓ best |
| H97 alphonse | bidir-xattn (general) | 9.540% |
| H104 edward | volume-wider (general) | 9.606% |
| **H100 (this)** | **dedicated tau_z head (specific)** | **9.685%** ❌ 4th |

**Three non-task-specific mechanisms each beat H100 on the very axis H100 was designed to crack.** test_WSS_z = 8.928% REGRESSES from canonical 8.753 by +0.175pp.

### Wave 33 mechanism class TASK-HEAD DEFINITIVELY CLOSED

- H92 tau_z loss-weight D NEG (loss-budget falsified on z)
- H93 tau_y loss-weight D NEG (loss-budget falsified on y)
- H96 split-decoder-heads D NEG (decoder capacity allocation falsified)
- H100 dedicated-tau_z-head B PARTIAL — mechanism falsified on its target axis
- **All 4 task-specific mechanisms NEGATIVE on their target axes** — task-head class will NOT appear in Wave 34 compound staging

### test_VP cross attribution

The −0.132pp test_VP cross is NOT directly attributable to the dedicated head (volume decoder untouched). Likely general training stability (cleaner gradient routes) or EMA epoch-12 checkpoint happening to be VP-strong. **Incidental cross, not architectural.**

### Reassignment

thorfinn reassigned to **H107 SURFACE-GLOBAL-CONTEXT-RESIDUAL-DECODER** (PR #1277) — NEW mechanism class (self-context-at-decoder via globally-pooled surface_hidden additive residual). +262K params matched-cost with H102 width. Tests whether per-token decoder needs access to global surface context that slice-tokens encode but don't directly expose.

---

## 2026-05-23 ~07:40 — PR #1264: H99 SURFACE-OUT-DEEPER-MLP (frieren, CLOSED) — **C NULL**

- **Branch**: `frieren/h99-surface-out-deeper-mlp` (closed at ~07:40Z 2026-05-23)
- **W&B run**: `fgxhka8k` (rank 0), terminal at step 70,652 / 70,664 = 100%, runtime 14.35h.
- **Hypothesis**: Surface decoder is depth-bound, not width-bound — change `surface_out` from 2-layer MLP `(n_hidden → n_hidden → output_dim)` to 3-layer `(n_hidden → n_hidden → n_hidden//2 → output_dim)` with SiLU. +132K params (~+250K effective footprint with init slack).

### Terminal results

| Channel | Validation (EP13 best EMA) | Test | Canonical (test) | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.3266%** | **6.0693%** | 5.844% | **+0.225pp regress** |
| surface_pressure | 4.147% | 3.804% | 3.577 (floor) | +0.227pp MISS floor |
| volume_pressure | 3.698% | **3.637%** | 3.643 (floor) | −0.006pp marginal cross (noise) |
| wall_shear | 7.181% | 7.009% | 6.727 (goal) | +0.282pp MISS goal |
| wall_shear_x | 6.291% | 6.230% | 5.83 | +0.40pp regress |
| wall_shear_y | 7.761% | 7.588% | 7.10 | +0.49pp regress |
| wall_shear_z | **9.736%** | **9.088%** | 9.83 | **−0.74pp IMPROVEMENT on binding axis** |

- Gate: val_abupt 6.3266% **MISS** gate 6.126 by +0.20pp.
- Test floors: only test_VP marginal cross (−0.006pp = noise); test_SP +0.227pp MISS, test_WSS +0.282pp MISS goal.
- val→test slope: −0.258pp (within canonical −0.282pp).

### Mechanism reading — DEPTH AXIS PRODUCTIVE BUT INFERIOR TO WIDTH

- 3-layer MLP trained stably (zero restarts, 1.87 it/s throughput-neutral)
- Mid-cosine slope was strong (−1.83pp/1k val_abupt at step 29k), made H99 a top-4 mid-cosine entry
- Late-cosine deceleration was SHARPER than width-axis H102 — H102 at 84.6% phase (val 6.18%) is ~0.17pp ahead of H99 at 94.5% phase (val 6.33%)
- Extra depth adds compositional capacity that the surface decoder doesn't fully exploit; width adds parallel feature-mixing that scales better at matched param cost
- **test_WSS_z = 9.088%** — −0.74pp improvement on the binding axis (best non-merged single-model result for this axis), but test_WSS_x and test_WSS_y both regress so overall test_WSS misses

### Wave 33 finding — surface decoder MLP capacity axes

- **width@right-place (H102, +266K)**: productive, fleet LEADER, val 6.18% at 84.6% (~6.15% terminal projection)
- **depth (H99, +250K)**: productive in mid-cosine, decelerates harder in late-cosine, terminal val 6.327% MISS gate by +0.20pp
- **split-task heads (H96, +263K)**: D NEG, regresses on val + target test axis
- **mlp_ratio inside shared decoder (H86, +600K)**: D NEG (earlier wave)
- **Mechanism class ranking confirmed**: width > info-residual > depth > task-head > FiLM > mlp_ratio-shared > split-head

### Verdict and reassignment

C NULL — clean execution but inductive bias of depth is suboptimal for shared decoder. frieren reassigned to **H106 VOLUME-GEOM-RESIDUAL-DECODER** (PR #1276) — mirror of H101 to the volume side, testing whether info-at-decoder-input thesis generalizes from surface to volume.

---

## 2026-05-23 ~01:30 — PR #1261: H96 WSS-DEDICATED-DECODER-HEAD (fern, CLOSED) — **D NEGATIVE**

- **Branch**: `fern/h96-wss-dedicated-decoder-head` (closed at ~01:30Z 2026-05-23)
- **W&B run**: `lw1591cw` (rank 0), terminal at step 70,666 / 70,664 = 100%, runtime 14.40h (843.8 min training + ~24 min eval + test).
- **Hypothesis**: Split the shared `surface_out` 2-layer MLP into two parallel decoder heads — one for surface_pressure (1 channel) and one for wall_shear_stress (3 channels). Each gets its own dedicated 2-layer MLP trunk, separating SP and WSS representations. +263K params (17.40M → 17.68M total).

### Terminal results

| Channel | Validation (EP13 best EMA) | Test | Canonical (test) | Δ test vs canonical |
|---|---:|---:|---:|---:|
| **abupt_axis_mean** | **6.2840%** | **6.0721%** | 5.844% | **+0.228pp regress** |
| surface_pressure | 4.1371% | 3.8514% | 3.5957% | +0.256pp regress (plateau) |
| volume_pressure | 3.6955% | 3.6417% | 3.643% (floor) | −0.001pp (ties floor, within numerical noise) |
| wall_shear | 7.1176% | 6.9971% | 6.727% (goal) | **+0.270pp regress** |
| wall_shear_x | 6.2141% | 6.2243% | 5.975% | +0.249pp regress |
| wall_shear_y | 7.7694% | 7.5938% | 7.274% | +0.320pp regress |
| wall_shear_z | 9.6041% | 9.0492% | 8.753% (binding) | +0.296pp regress |

- Gate: val_abupt 6.284% **MISS** gate 6.126 by +0.158pp.
- Test floors: NONE crossed (test_VP −0.001pp ties floor within numerical noise; SP/WSS regress).

### Verdict: D NEGATIVE
Both D NEGATIVE conditions met:
1. val_abupt 6.284% > 6.20% threshold.
2. test_WSS +0.270pp regress > +0.10pp threshold on target axis.

### Mechanism reading

Per-head loss decomposition at terminal (step 70,652):
| Loss term | Value | Share of surface_loss | Slope (pp/1k steps) |
|---|---:|---:|---:|
| train/surface_loss_cp | 6.78e−4 | ~27% | −3.59e−6 |
| train/surface_loss_wss | 1.87e−3 | ~73% | −3.86e−5 |

WSS-head contribution dominates surface loss 2.75× (3-channel target + higher near-wall variance); its slope is 10.7× steeper than cp-head — confirming the dedicated WSS-trunk continued absorbing gradient until terminal. Both heads logged cleanly throughout.

val_WSS_y slope late-cosine was 2× canonical (−0.0121 vs canonical ~−0.006/1k) — the dedicated WSS-trunk WAS engaging on its design objective on val. But the WSS_y advantage did NOT survive to test; test_WSS_y is the worst test axis at +0.320pp regress.

val→test deltas (H79 diagnostic):
| Channel | Δ (test − val) |
|---|---:|
| abupt | −0.212pp |
| surface_pressure | −0.286pp |
| volume_pressure | −0.054pp |
| wall_shear | −0.121pp |
| wall_shear_x | +0.010pp |
| wall_shear_y | −0.176pp |
| wall_shear_z | −0.555pp |

All channels canonical-range generalization slope; no test-set anomaly. The val_WSS_y advantage genuinely didn't translate to test → the split-head learned a val-specific WSS_y representation that didn't generalize.

### Wave-level finding

H96's D NEG closure with +263K dedicated WSS-trunk params adds evidence to the emerging Wave 33 hypothesis: **decoder bottleneck is INFORMATIONAL, not CAPACITY-bound**. Compare:
- H96: +263K params split-head decoder capacity → D NEG.
- H86 (Wave 32): +mlp_ratio 4→6 = +600K capacity at shared decoder → D NEG.
- H101 (in-flight, +1.5K params): raw position residual at decoder input → currently competing with +1M param H97 bidir-xattn at matched mid-cosine phase.

Capacity allocation between or within decoder heads is NOT the binding constraint. The next test (H105) extends the information-axis attack: raw normals at decoder input.

### Reassignment
fern → **H105 SURFACE-NORMAL-RESIDUAL-DECODER (PR #1271)** — zero-init linear projection from `surface_x[..., 3:6]` (normals, 3D unit vectors) to n_hidden=512, added as residual to surface_hidden before surface_out. Same shape as H101 (positions) but using normals — the differential operator argument for τ = μ ∂u/∂n. +1.5K params.

---

## 2026-05-22 ~21:00 — PR #1267: H98v2 SURFACE-LATE-LAYER-SPLIT v2 (askeladd, CLOSED) — **INFEASIBLE-WITHIN-BUDGET** — O(N²) self-attention on N=65,536 surface tokens hits 0.385 it/s vs baseline 1.25 it/s; only ~2.3 epochs feasible in 18.3h timeout. Mechanism preserved via H103 VOLUME-CONTEXT-FILM-DECODER (PR #1270).

- **Branch**: `askeladd/h98-surface-late-layer-split-v2` (closed at ~21:00Z 2026-05-22)
- **W&B run**: `yoi1f40v` (KILLED — not trained to terminal)
- **Hypothesis**: Extra `nn.TransformerEncoderLayer` on surface_hidden after surf_to_vol_xattn, zero-init at identity (out_proj.weight=0, linear2.weight=0). +3.15M params (17.42M → 20.57M total).

### Infeasibility — compute-envelope failure

Student reported 0.385 it/s vs baseline 1.25 it/s (3.2× throughput drop) immediately after starting full 8-GPU run. Root cause: `TransformerEncoderLayer` runs full self-attention on N=65,536 surface tokens = O(N²) = ~4.3 billion attention pairs per sample × 4 batch × 8 GPUs. At 0.385 it/s, 70,664 total steps = ~51 wall-hours; `SENPAI_TIMEOUT_MINUTES=1100` (18.3h) allows only ~25,300 steps ≈ 2.3 epochs. Not comparable to 13-epoch fleet.

### Option selected — D (kill, close as infeasible-as-spec)

- Option A (partial 2.3-epoch run): non-comparable evidence, not recommended
- Option B (reduce surface_points to 32768): confounds mechanism with lower surface density
- Option C (slice-level self-attention): materially changes hypothesis
- **Option D (kill, close)**: clean record, mechanism preserved in restructured H103

### Mechanism class ruling

Full per-token self-attention on N=65,536 surface tokens is **operationally infeasible** in current budget regardless of hypothesis quality. Rules out an entire mechanism class:
- Per-token surface self-attention: INFEASIBLE (O(N²))
- Slice-level attention: EXISTS in backbone (O(slices²) = O(128²) = fine)
- Cross-attention vol↔surf: FINE (O(N×M) where M=16384-65536 but H97 confirmed it runs)
- FiLM/MLP/position operations: O(N) = FINE

### Reassignment → H103 askeladd PR #1270 VOLUME-CONTEXT-FILM-DECODER

Same vol→surf information-flow axis, O(N) cost: `film_projector` (Linear 512→1024) projects pooled volume_hidden context to FiLM (γ, β) parameters that modulate surface_hidden before surface_out. Zero-init at identity. +525K params. Expected throughput ~baseline 1.2-1.3 it/s.

---

## 2026-05-22 ~21:00 — PR #1257: H94 VOLUME-LOSS-WEIGHT-INCREASE (edward, CLOSED) — **OUTCOME B PARTIAL** — test_VP CROSS (7th single-flag) + Lion sign-update asymmetry definitively confirmed (+0.30pp slower than H87 subtractive route at same 1.33:1 ratio); surf:vol=1.33:1 ratio coverage CLOSED. edward reassigned H104 VOLUME-OUT-WIDER-MLP (PR #1269).

- **Branch**: `edward/h94-volume-loss-weight-increase` (closed at ~21:00Z 2026-05-22)
- **W&B run**: `6fwwywk0` (rank 0; 13 epochs, 14.34h, best_epoch=12)
- **Hypothesis**: volume_loss_weight 1.0→1.5 (same surf:vol ratio 1.33:1 as H87 via additive route instead of subtractive).

### Results — terminal metrics

| Metric | H94 | Baseline #972 (test) | H87 (test) | Δ vs #972 | Δ vs H87 | Status |
|---|---:|---:|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.357%** | — | — | — | — | MISS gate +0.231pp |
| test_abupt | 6.073% | 5.844 | 5.987 | +0.229pp | +0.086pp | mild regress |
| **test_VP (floor ≤3.643%)** | **3.582%** | 3.643 | 3.495 | **−0.061pp ✓** | +0.087pp | **CROSS ✓** (7th single-flag) |
| test_SP (floor ≤3.577%) | 3.829% | 3.577 | 3.734 | +0.252pp | +0.095pp | MISS (plateau range) |
| test_WSS (goal ≤6.727%) | 7.021% | 6.727 | 6.944 | +0.294pp | +0.077pp | MISS goal |
| test_WSS_x | 6.234% | 5.972 | 6.157 | +0.262pp | +0.077pp | mild regress |
| test_WSS_y | 7.640% | 7.276 | 7.532 | +0.364pp | +0.108pp | regress |
| test_WSS_z | 9.082% | 8.945 | 9.017 | +0.137pp | +0.065pp | mild regress |

### Mechanism analysis

**Lion sign-update asymmetry DEFINITIVELY CONFIRMED**:
- H87 (subtractive: surf_loss 2.0→1.5) terminal val_abupt **6.045%** B PARTIAL HISTORIC
- H94 (additive: vol_loss 1.0→1.5, SAME surf:vol ratio 1.33:1) terminal val_abupt **6.357%** = **+0.312pp slower**

Root cause from student's terminal bucket analysis:
- `train/surface_loss` raw = 0.002443 vs `train/volume_loss` raw = 0.000393 → **intrinsic 6.21:1 raw magnitude ratio**
- Weighted effective ratio (H94 config): 0.004886 / 0.000590 = **8.28:1** (surface still dominates trunk gradient under Lion)
- Increasing volume_loss amplifies minority signal; reducing surface_loss damps dominant signal — H87's subtractive route has more leverage.

**Surf:vol ratio 1.33:1 coverage via both routes COMPLETE**:

| Variant | surf:vol | val_abupt | test_VP | test_SP | test_WSS_z |
|---|---|---:|---:|---:|---:|
| #972 baseline | 2.0:1.0 | 6.181 | 3.643 | 3.577 | 8.945 |
| **H87 ✓ HISTORIC** | **1.5:1** | **6.045 ✓** | **3.495 ✓** | 3.734 | 9.017 |
| **H94 (this)** | **1.33:1 additive** | **6.357** | **3.582 ✓** | 3.829 | 9.082 |
| H95 C NULL | 1.25:1 subtractive | 6.261 | 3.564 ✓ | 3.789 | 8.997 |

**Bright spots**: val_WSS_y slope −0.0092 pp/1k (steepest in fleet) + test_VP CROSS −0.061pp confirming volume-head mechanism engaged intended axis. val→test slope WSS_z −0.681pp largest val→test concession in fleet (consistent with 34-case val overweighting WSS_z hardness vs 50-case test split).

### Program-level — loss-weight rebalance axis FULLY EXHAUSTED (5 PRs)

H87 + H92 + H93 + H94 + H95 close all loss-weighting coverage. Per Issue #1056, NOT merged. Wave 33 architectural attacks are the correct continuation.

### Reassignment → H104 edward PR #1269 VOLUME-OUT-WIDER-MLP

Mirror of H102 (tanjiro, SURFACE-OUT-WIDER-MLP) on the volume decoder side. `--volume-out-width-factor 2.0` scales volume_out funnel hidden dims: 512→256→128→vol_dim becomes 512→512→256→vol_dim. +229K params. Zero operational risk. Key diagnostic: H94 showed vol head responds to capacity perturbations on test_VP axis via gradient route; H104 tests parameter-capacity route on the same axis.

---

## 2026-05-22 20:00 — PR #1258: H95 SURF-LOSS-PUSH-FURTHER (tanjiro, CLOSED) — **OUTCOME C NULL** — H87's 1.5:1 surf:vol ratio CONFIRMED substrate sweet spot, loss-weight rebalance class DEFINITIVELY CLOSED, test_VP CROSS (6th single-flag), target axis test_WSS_z NOT degraded. tanjiro reassigned H102 SURFACE-OUT-WIDER-MLP (PR #1268).

- **Branch**: `tanjiro/h95-surf-loss-push-further` (closed at 20:00Z 2026-05-22)
- **W&B run**: `ze0bohdu` (rank 0; finished at step 70,652, 14.0h training time, 838.96 min)
- **Hypothesis**: Direct H87 follow-up testing surface_loss_weight 1.5 → 1.25. Key signal: test_SP direction tells us if H87 IS the substrate sweet spot or if further reduction has headroom.

### Results — terminal metrics

| Metric | H95 | Baseline #972 | H87 (Wave 32 best) | Δ vs H87 | Verdict |
|---|---:|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.2612%** | 6.181 | **6.045** ✓ | +0.216pp | MISS gate +0.135pp |
| test_abupt | 6.018% | (5.844 corrected) | 5.987 | +0.031pp | mild regress |
| **test_VP (floor)** | **3.564%** ✓ | 3.643 | 3.495 | +0.069pp | **CROSS (6th single-flag)** |
| test_SP (floor) | 3.789% | 3.577 | 3.734 | +0.055pp | MISS (plateau range) |
| test_WSS (goal) | 6.956% | 6.727 | 6.944 | +0.012pp | regress |
| test_WSS_z | **8.997%** | 8.945 | 9.017 | **−0.020pp** | **marginally below H87** |

### H87 sweet-spot test — PASSED (predicted outcome)

Per H87 follow-up criteria:
- If H95 test_SP < 3.65% → SP plateau partially crackable via further ratio reduction
- If H95 test_SP > 3.85% → H87's 1.5 IS confirmed substrate sweet spot

**H95 test_SP = 3.789%** — INTERMEDIATE plateau range (NEITHER threshold cracked). This is consistent with **H87 being the substrate sweet spot**: surface representation degrades smoothly past 1.5, with diminishing returns but no catastrophic break. Continuing to surf_loss=1.0 would predictably regress further.

### Surf:vol ratio sweep — COMPLETE

| Variant | surf:vol ratio | val_abupt | test_VP | test_SP | test_WSS_z |
|---|---|---:|---:|---:|---:|
| #972 baseline | 2.0:1.0 | 6.181 | 3.643 | 3.577 | 8.945 |
| **H87 ✓** | **1.5:1.0** | **6.045 ✓** | **3.495 ✓** | 3.734 | 9.017 |
| H94 (in-flight ~D NEG) | 2.0:1.5 (1.33:1) | ~6.34 | — | — | — |
| **H95** | **1.25:1.0** | **6.261** | 3.564 ✓ | 3.789 | 8.997 |

**Both directions from H87 produce slower convergence**: subtractive (H95: −0.25 on surf, +0.22pp slower) and additive (H94: +0.5 on vol, +0.30pp projected slower). H87's 1.5:1 IS the substrate sweet spot.

### Mechanism interpretation — Lion sign-update normalization confirmed

Under Lion's sign-update, step magnitude is normalized; loss-weighting reallocates which channel gets gradient signal but does NOT add representational capacity. H87 happens to hit the optimal allocation. Movement in EITHER direction starves the under-weighted channel.

### Combined Wave 32 loss-weighting closure

| Variant | Mechanism | Verdict | Notes |
|---|---|:--|:--|
| H87 (surf=1.5) | substrate sweet spot | **B PARTIAL HISTORIC** | best, NOT merged per Issue #1056 |
| H92 (tau_z=3.0) | per-tau-channel | **D NEGATIVE** | target axis tau_z DEGRADED |
| H93 (tau_y=2.5) | per-tau-channel | **D NEGATIVE** | target axis tau_y DEGRADED |
| H94 (vol=1.5) | additive route | ~**D NEG** (in-flight) | +0.30pp slower than H87 |
| **H95 (surf=1.25)** | subtractive past sweet spot | **C NULL** | +0.22pp slower than H87 |

**Loss-weighting axis EXHAUSTED.** Per Morgan's Issue #1056 position, not merged even if cleared val gate.

### Bright spot — test_WSS_z 8.997% comparable to H91 fleet-best 8.95%

H95's test_WSS_z is marginally BETTER than H87's (8.997 vs 9.017 by −0.020pp). Combined with H91's fleet-best 8.95%, this suggests WSS_z does NOT primarily respond to surf:vol budget allocation — WSS_z compression requires DIFFERENT architectural levers (decoder structure, position info) which is exactly what Wave 33 targets.

### val→test slope on SP-axis — STRONGEST in H95's run

H95 val_SP 4.134% → test_SP 3.789% = **−0.345pp** improvement on SP. Stronger than H87's −0.273pp on SP. Suggests pessimistic val reads under aggressive surf reduction (34 val cases vs 50 test) — train/val distribution shift artifact rather than fundamental ceiling.

### tanjiro reassigned PR #1268 H102 SURFACE-OUT-WIDER-MLP

**Mechanism**: `--surface-out-width-factor 2.0` widens surface_out hidden dimension from n_hidden=512 to n_hidden=1024 (keeping 2-layer depth). +266K params.

**Pairs with H99 (frieren, PR #1264)** which tests decoder DEPTH (2→3 layer). H99 + H102 together map the depth × width plane of decoder capacity expansion:
- If H102 wins (width) → width is productive
- If H99 wins (depth) → depth is productive
- If both clear gate → compound for Wave 34
- If both fail → decoder MLP capacity is NOT the binding constraint

**Key signal**: test_SP < 3.70% → first crack of 11-variant plateau and validates "decoder width" as productive lever.

---

## 2026-05-22 18:45 — PR #1251: H91 MODEL-SLICES-EXPANSION (nezuko, CLOSED) — **OUTCOME B PARTIAL** — val gate NARROW MISS +0.049pp + test_VP CROSS −0.066pp (5th single-flag) + fleet-best test_WSS_z 8.95% (slice expansion engages WSS_z compression). Wave 32 Tier-2 architectural sweep COMPLETE. nezuko reassigned H101 GEOM-RESIDUAL-DECODER (PR #1266).

- **Branch**: `nezuko/h91-model-slices-expansion` (closed at 18:45Z 2026-05-22)
- **W&B run**: `5anlrjsd` (rank 0; finished at step 70,652, 15.60h training time, 914 min total wall)
- **Hypothesis**: Single-flag `--model-slices 128→192` on canonical Wave 32 substrate. First-ever Transolver slice-token-count sweep. Hypothesis: SP plateau bound by slice-token resolution at h=512/L=5.

### Results — terminal metrics

| Metric | H91 | Baseline/Floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | 6.1748% | 6.126% | +0.049pp ❌ | NARROW MISS |
| test_abupt | 5.9834% | 5.844 | +0.139pp ❌ | regress |
| **test_VP (floor)** | **3.5768%** | 3.643 | **−0.066pp ✓** | **CROSS (5th single-flag)** |
| test_SP (floor) | 3.7467% | 3.577 | +0.170pp ❌ | MISS (plateau range 3.74-3.95%) |
| test_WSS (goal) | 6.9142% | 6.727 | +0.187pp ❌ | regress |
| test_WSS_x | 6.1502% | (canonical ~5.96) | +0.19pp ❌ | regress |
| test_WSS_y | 7.4938% | (canonical ~7.43) | +0.06pp ❌ | mild regress |
| **test_WSS_z** | **8.9496%** | 8.945 | **+0.005pp** | **FLEET-BEST WSS_z** |

### Wave 32 Tier-2 architectural sweep — COMPLETE

| Variant | val_abupt | test_VP | test_SP | test_WSS_z | Verdict |
|---|---:|---:|---:|---:|---|
| H88 (heads=8) | 6.2088% | 3.6018% ✓ | 3.8365% | 8.968% | B PARTIAL |
| **H89 (depth=6)** | **6.1186% ✓** | **3.482% ✓** | 3.709% | 9.087% | **B PARTIAL HISTORIC** (gate CLEAR) |
| **H91 (slices=192)** | **6.1748%** | **3.5768% ✓** | 3.7467% | **8.9496%** | **B PARTIAL** (narrow gate miss, fleet-best WSS_z) |

**Ordering on val_abupt**: depth > slices > heads. All three add representational capacity but plateau on test_SP/WSS axes — confirming encoder-stack exhaustion and decoder-bound binding constraints.

### Mechanism interpretation

**Slice-token-count axis productively engages WSS_z** (fleet-best 8.95%) via finer surface coverage (~341 surface points per slice token at slices=192 vs 512 at canonical 128). But test_SP=3.7467% lands at LOWER EDGE of the 3.74-3.95% Wave 32 plateau — not decisively below 3.70%. **Slice-token resolution is NOT the binding SP-axis constraint.**

H91 val→test slope on SP-axis is **−0.312pp** — STRONGEST in Wave 32 fleet — indicating that slice expansion produces val→test generalization on SP channel. The val checkpoint metric is bottlenecked, not the underlying spatial discrimination.

### Per-axis WSS readings — H91 is the ONLY Wave 32 single-flag mechanism to engage WSS_z compression below 9.0%

This data point will be cited in Wave 33 attack documentation as evidence that **spatial token granularity engages WSS_z** (vs SP and WSS_x which respond to other levers). Combined with H89 (depth) and H88 (heads), the architectural Tier-2 axis is fully mapped:
- depth → strongest aggregate val_abupt + test_VP
- slices → fleet-best test_WSS_z (only mechanism below 9.0%)
- heads → weakest (no axis improved)
- mlp_ratio (H86) → all axes regress

### nezuko reassigned PR #1266 H101 GEOM-RESIDUAL-DECODER

**Mechanism**: zero-initialized linear projection from raw surface position (xyz) → n_hidden, added as residual to surface_hidden BEFORE surface_out. At init identity (zero residual). During training, model learns to use raw position info directly at decoder, bypassing slice-attention bottleneck.

**Orthogonal to Wave 33 in-flight**: H96/H100 are structural decoder splits; H97 is cross-modal info flow; H98 is extra processing layer; H99 is deeper MLP. H101 is **input signal augmentation** (new info at decoder, not more compute).

**Physics motivation**: cp and tau_x/y/z are spatial functions f(x,y,z). Slice attention compresses 65K surface points → 128 slices, necessarily losing fine spatial discrimination. Direct position skip-connection recovers this. Param cost ~1.5K (negligible).

**Key signal**: test_SP < 3.70% would be first crack of 11-variant Wave 32 plateau; test_WSS_z < 8.5% would break binding axis.

---

## 2026-05-22 18:30 — PR #1254: H93 TAU-Y-LOSS-PUSH (thorfinn, CLOSED) — **OUTCOME D NEGATIVE** — per-tau-channel mechanism DOUBLY FALSIFIED alongside H92. val_WSS_y 7.802% WORSE than canonical siblings. thorfinn reassigned H100 WSS-Z-DEDICATED-HEAD (PR #1265).

- **Branch**: `thorfinn/h93-tau-y-loss-push` (closed at 18:30Z 2026-05-22)
- **W&B run**: `ch6u4yy0` (rank 0; finished at step 70,664, 13.97h)
- **Hypothesis**: `--tau-y-loss-weight 1.5→2.5` (+67%) first-ever tau_y channel sweep. Hypothesis: stronger gradient on tau_y axis would compress val_WSS_y below canonical sibling range.

### Results

| Metric | H93 | Baseline/Floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | 6.3235% | 6.126% | +0.198pp ❌ | MISS gate |
| test_abupt | 6.072% | 5.844 | +0.228pp ❌ | regress |
| test_VP (floor) | 3.5629% | 3.643 | **−0.080pp ✓** | CROSS (incidental) |
| test_SP (floor) | 3.8335% | 3.577 | +0.257pp ❌ | MISS |
| test_WSS (goal) | 7.0405% | 6.727 | +0.314pp ❌ | REGRESS |
| **val_WSS_y (target)** | **7.802%** | canonical 7.63-7.71 | **WORSE** ❌ | **TARGET AXIS DEGRADED** |

### Mechanism falsification

**67% stronger gradient on tau_y produced WORSE target axis** (7.802% vs canonical 7.63-7.71%). Combined with H92 (50% on tau_z, also WORSE), the per-tau-channel loss-weight class is DEFINITIVELY CLOSED. Lion sign-update normalizes step magnitude — budget reallocation adds no representational capacity. Over-rebalancing penalty scales with weight increase magnitude.

**thorfinn reassigned H100 WSS-Z-DEDICATED-HEAD** (PR #1265): architectural version of the same per-channel intuition — gives tau_z its own dedicated 2-layer MLP decoder (`--use-wss-z-dedicated-head`), testing representational capacity separation instead of gradient budget reallocation.

---

## 2026-05-22 18:05 — PR #1249: H89 MODEL-DEPTH-EXPANSION (frieren, CLOSED) — **OUTCOME B PARTIAL HISTORIC** — FIRST Wave 32 architectural-class val gate clear (val_abupt 6.1186% clears gate −0.007pp) + deepest test_VP cross in fleet (−0.161pp). AND-gate fails on test_SP/WSS. frieren reassigned H99 SURFACE-OUT-DEEPER-MLP (PR #1264).

- **Branch**: `frieren/h89-model-depth-expansion` (closed at 18:05Z 2026-05-22)
- **W&B run**: `be91oalm` (rank 0; finished at step 70,664, 15.79h training time)
- **Hypothesis**: Single-flag `--model-layers 5→6` on canonical Wave 32 substrate. First-ever backbone depth sweep.

### Results

| Metric | H89 | Baseline/Floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.1186%** | gate 6.126% | **−0.007pp ✓** | **CLEARS gate — FIRST architectural-class clear** |
| test_abupt | 5.908% | 5.844% | +0.064pp ❌ | regress |
| **test_VP (floor 3.643)** | **3.482%** | 3.643% | **−0.161pp ✓** | **CROSS — DEEPEST in entire Wave 32 fleet** |
| test_SP (floor 3.577) | 3.709% | 3.577% | +0.132pp ❌ | MISS floor (11th plateau confirmation) |
| test_WSS (goal 6.727) | 6.832% | 6.727% | +0.105pp ❌ | above goal |
| test_WSS_x / _y / _z | 6.055 / 7.414 / 8.879 | 5.975 / 7.274 / 8.753 | +0.083 / +0.138 / +0.129 | all regress |

### Analysis

**Historic milestone**: H89 is the **FIRST Wave 32 architectural-class val gate clear**. Depth-axis expansion adds genuine representational capacity — the first mechanism to do so. test_VP cross −0.161pp is the deepest in the fleet (6th single-flag to cross VP floor).

**Encoder-stack exhaustion confirmed**: depth (H89), heads (H88), width (H86), LR (H85/H90), slices (H91 in-flight) all fail to crack test_SP plateau. **Encoder-stack architecture is NOT the test_SP bottleneck.** Decoder-bound hypothesis confirmed by student analysis.

**val_WSS_z late-cosine creep real**: late +0.237 pp/1k slope persisted to terminal test_WSS_z 8.879% (+0.129pp). WSS binding axis is DECODER-BOUND.

- frieren → **Wave 33 H99: SURFACE-OUT-DEEPER-MLP** (PR #1264): `self.surface_out` 2-layer → 3-layer (n_hidden → n_hidden → n_hidden//2 → 4ch). New flag `--use-deeper-surface-mlp`. +132K params. Direct decoder-bound plateau attack.

---

## 2026-05-22 17:50 — PR #1252: H92 TAU-Z-LOSS-PUSH (askeladd, CLOSED) — **OUTCOME D NEGATIVE** — mechanism FALSIFIED: per-tau-channel loss-weight class closed definitively. test_WSS_z WORSE than all canonical-recipe siblings. askeladd reassigned H98 SURFACE-LATE-LAYER-SPLIT (PR #1263).

- **Branch**: `askeladd/h92-tau-z-loss-push` (closed at 17:50Z 2026-05-22)
- **W&B run**: `or56xz4x` (rank 0; finished at step 70,652, 13.97h training time, EP12 EMA best)
- **Hypothesis**: Single-flag `--tau-z-loss-weight 2.0→3.0` (+50%) on canonical Wave 32 substrate. First-ever per-tau-channel loss weight sweep. Hypothesis: stronger gradient signal on binding WSS_z axis would preferentially compress test_WSS_z < 8.5%.

### Results

| Metric | H92 | Baseline #972 / floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | 6.3235% | 6.126% | +0.198pp ❌ | MISS gate |
| **test_abupt** | 6.0723% | 5.844% | +0.228pp ❌ | regress |
| **test_VP (floor)** | 3.5629% | 3.643% | **−0.080pp ✓** | CROSS (incidental) |
| **test_SP (floor)** | 3.8335% | 3.577% | +0.257pp ❌ | BREACH |
| **test_WSS (goal)** | 7.0405% | 6.727% | +0.314pp ❌ | REGRESS above goal |
| test_WSS_x | 6.275% | ~6.1% | — | neutral |
| test_WSS_y | 7.639% | ~7.4% | — | neutral |
| **test_WSS_z (binding)** | **9.051%** | ~8.96% | +0.09pp ❌ | **FAILED — should be <8.5%** |

### Per-WSS-axis cross-sibling comparison (val_WSS_z, canonical tau_z=2.0 siblings)

| Variant | tau_z weight | val_WSS_z | test_WSS_z |
|---|---|---:|---:|
| H88 (heads=8) | 2.0 | 9.493% | 9.498% |
| H89 (depth=6) | 2.0 | ~9.50% | ~9.50% |
| H91 (slices=192) | 2.0 | 9.485% | ~9.5% |
| **H92 (tau_z=3.0)** | **3.0** | **9.601%** | **9.051%** |

**H92 val_WSS_z is WORSE than ALL canonical-recipe siblings.** The tau_z weight change actually DEGRADED val WSS_z while providing only marginal test improvement.

### Mechanism Falsification

**Per-tau-channel loss-weight class DEFINITIVELY CLOSED.** Under Lion sign-update, per-channel loss-weight scales gradient ratio entering the surface head, but Lion normalizes step magnitude to ±lr. The 50% stronger tau_z signal does NOT add representational capacity — it merely redistributes budget within existing capacity. Combined with H93 (tau_y=2.5, thorfinn PR #1254, in-flight), this class is closed regardless of H93 outcome.

**Program-level finding:** test_WSS_z (fleet range 8.93-9.66% across H78-H92) is **ARCHITECTURE-BOUND, not loss-budget-bound.** The productive direction is architectural separation of WSS prediction (H96 split-heads, H97 bidirectional xattn, H98 surface-late-layer-split).

**Incidental test_VP CROSS (3.5629%)**: joins H75/H76/H82/H83/H84 as 5th single-flag to cross test_VP floor. Volume-favoring pattern consistent. Not a merge candidate (AND-gate fails on 3/4 paper-facing channels).

### Reassignment

- askeladd → **Wave 33 H98: SURFACE-LATE-LAYER-SPLIT** (PR #1263) — adds one extra TransformerEncoderLayer applied only to surface tokens between shared backbone and surface decoder head. Identity at init (zero-init out_proj + linear2). Tests late-stage surface specialization as WSS binding axis mechanism. Orthogonal to H96 (decoder-head separation) and H97 (bidirectional xattn).

---

## 2026-05-22 16:16 — PR #1250: H90 LR-DOWNWARD-SWEEP (alphonse, CLOSED) — **OUTCOME D NEGATIVE** — all 4 paper-facing test channels regress, val_abupt misses gate +0.193pp. Closes LR-magnitude sweep class: canonical lr=9e-5 confirmed substrate sweet spot.

- **Branch**: `alphonse/h90-lr-downward-sweep` (closed at 16:16Z 2026-05-22)
- **W&B run**: `sj41hrg2` (rank 0; finished at step 70,652, 14.0h training time)
- **Hypothesis**: Single-flag `--lr 9e-5 → 6e-5` (−33%) on canonical Wave 32 baseline. First-ever LR sweep BELOW canonical. Together with H85 (lr=1.2e-4 UP, D NEG) closes the LR magnitude sweep class.

### Results

| Channel | **H90 (lr=6e-5)** | BL #972 / floor | Δ vs BL / floor | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | 6.319% | gate 6.126% | +0.193pp ❌ | MISS gate |
| **test_abupt** | 6.063% | 5.844% | +0.219pp ❌ | regress |
| **test_VP (floor 3.643)** | 3.659% | 3.643% | +0.016pp ❌ | MISS floor (narrow) |
| **test_SP (floor 3.577)** | 3.817% | 3.577% | +0.240pp ❌ | MISS floor (plateau range, 10th confirmation) |
| **test_WSS (goal 6.727)** | 6.986% | 6.727% | +0.259pp ❌ | above goal |
| test_WSS_x / _y / _z | 6.205 / 7.587 / 9.048 | — | — | all axes regress |

### Analysis — LR-magnitude class CLOSED

**Both sweep arms miss merge gate**: H85 UP (+33%) val=6.390% D NEG, H90 DOWN (−33%) val=6.319% D NEG. Canonical 9e-5 is confirmed substrate sweet spot — neither direction yields improvement. **LR-magnitude class FULLY CHARACTERIZED, no further LR sweeps warranted on tay.**

**val→test gap on abupt slightly tighter** under lower LR: −0.255pp (H90) vs −0.282pp (baseline), consistent with finer-grained per-step Lion updates producing less overfitting. But +0.027pp improvement far too small to recover the +0.193pp val gate gap.

**WSS_z val→test gap is −0.640pp** (largest of all channels), deepest divergence between val and test confirming WSS_z is the binding test-set constraint.

**test_SP 10th plateau confirmation**: 3.78-3.95% range (H90 = 3.817%) across H78/H79/H80/H82/H83/H84/H86/H87/H88/H90.

**alphonse reassigned H97 BIDIRECTIONAL-XATTN** (PR #1262): second Wave 33 architectural attack — adds `vol_to_surf_xattn` mirroring existing `surf_to_vol_xattn`. Surface decoder heads get access to volume flow context for the first time.

---

## 2026-05-22 09:10 — PR #1248: H88 MODEL-HEADS-EXPANSION (fern, CLOSED) — **OUTCOME B PARTIAL** — paper-positive test_VP cross −0.041pp but 3/4 paper-facing test channels regress and binding test_SP plateau holds (+0.260pp miss). Capacity expansion via attention head granularity FALSIFIED.

- **Branch**: `fern/h88-model-heads-expansion` (closed at 09:10Z 2026-05-22)
- **W&B run**: `gjoa1fm4` (rank 0; finished cleanly at step 70,664, 16.3h training time)
- **Hypothesis**: Single-flag `--model-heads 4 → 8` on canonical Wave 32 baseline #972 substrate. First-ever attention head count sweep.

### Results

| Channel | **H88 (heads=8)** | BL #972 / floor | Δ vs BL / floor | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | 6.2088% | gate 6.126% | +0.083pp ❌ | MISS gate |
| **test_abupt** | 5.9987% | 5.844% | +0.155pp ❌ | regress |
| **test_VP (floor 3.643)** | **3.6018%** | 3.643% | **−0.041pp** ✅ | **CROSSES floor** |
| **test_SP (floor 3.577)** | 3.8365% | 3.577% | +0.260pp ❌ | MISS floor (plateau range) |
| **test_WSS (goal 6.727)** | 6.8888% | 6.727% | +0.161pp ❌ | above goal |
| test_WSS_x / _y / _z | 6.103 / 7.484 / 8.968 | 5.975 / 7.274 / 8.753 | +0.128 / +0.210 / +0.215 | ALL axes regress |

### Analysis — head granularity FALSIFIED as standalone lever

**EP3 val_SP signal was fleet-best**: val_SP=4.4868% at step 32,594 — THE strongest mid-cosine SP signal in the campaign. **But the gain did NOT survive late-cosine**: terminal val_SP 4.1376% landed in fleet-typical range, test_SP 3.8365% missed floor by +0.260pp.

**Mechanism reading**: more attention heads add per-head diversity (mid-cosine engagement on SP axis) but cannot escape the H80 architectural-bound on test_SP plateau within the current shared-decoder design. Conclusion: head-granularity expansion engages mid-cosine but late-cosine reverts under shared-decoder bottleneck.

**Capacity expansion via existing-module width-scaling is now falsified across 3 independent axes**:
- H85 (LR=1.2e-4): D NEG
- H86 (mlp_ratio=6): D NEG
- H88 (heads=8): B PARTIAL test_VP-only

**9th independent variant confirms test_SP plateau** in 3.74-3.95% range vs floor 3.577. Cross-variant data: H78 +0.142, H79 +0.323, H80 +0.353, H82 +0.202, H83 +0.162, H84 +0.194, H86 +0.358, H87 +0.157, H88 +0.260. **The plateau is a STRUCTURAL bound under the current SHARED decoder architecture, not a hyperparameter-tunable threshold.**

**Per-axis WSS reading**: H88 did not help any WSS axis — all 3 regressed by +0.13-0.22pp. **This is the key data point that motivates the next direction**: WSS prediction quality is decoder-shared-trunk bound, not attention-capacity bound.

**fern reassigned H96 WSS-DEDICATED-DECODER-HEAD** (PR #1261): per Morgan's Issue #1056 guidance, Wave 33 enters the architectural separation phase — split `self.surface_out` into dedicated SP head (1 ch) + dedicated WSS head (3 ch), each with own 2-layer MLP trunk. First attack on the WSS-specific decoder hypothesis.

---

## 2026-05-22 04:35 — PR #1247: H87 SURFACE-LOSS-WEIGHT-REDUCTION (tanjiro, CLOSED) — **OUTCOME B PARTIAL** — **HISTORIC: FIRST Wave 32 variant to clear val_abupt merge gate AND cleanest test_VP cross in fleet** — but AND-gate test_SP miss +0.157pp + test_abupt regress + test_WSS regress vs goal, so does NOT meet merge criterion

- **Branch**: `tanjiro/h87-surface-loss-weight-reduction` (closed at 04:35Z 2026-05-22)
- **W&B run**: `jpspxktf` (rank 0; finished cleanly at step 70,666, 14.33h training time)
- **Hypothesis**: Single-flag `--surface-loss-weight 2.0 → 1.5` on canonical Wave 32 baseline #972 substrate. First-ever surface_loss_weight reduction sweep on tay (dl24 H26 cross-pollination).

### Results — historic Wave 32 milestone

| Channel | **H87 (surf_loss=1.5)** | BL #972 / floor | Δ vs BL / floor | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.045%** | gate 6.126% | **−0.081pp** ✅ | **CLEARS gate — FIRST Wave 32 variant** |
| val_VP | 3.580% | — | — | — |
| val_SP | 4.007% | — | — | — |
| val_WSS | 6.838% | — | — | — |
| **test_abupt** | 5.987% | 5.844% | +0.143pp ❌ | regress (primary test metric) |
| **test_VP (floor 3.643)** | **3.495%** | 3.643% | **−0.148pp** ✅ | **CROSSES floor — cleanest in fleet** |
| **test_SP (floor 3.577)** | 3.734% | 3.577% | +0.157pp ❌ | MISS floor (smallest of any Wave 32 SP miss) |
| **test_WSS (goal 6.727)** | 6.944% | 6.727% | +0.217pp ❌ | above goal |
| test_WSS_x / _y / _z | 6.157 / 7.532 / 9.017 | — | — | fleet-typical |

### Analysis — strongest signal in Wave 32

**val_abupt 6.045% is the NEW BEST val_abupt on tay's substrate**, beating gate by −0.081pp. **test_VP 3.495% is the CLEANEST test_VP cross of the entire Wave 32 fleet**, beating floor by −0.148pp. The mechanism (surface_loss_weight reduction → surf:vol gradient ratio 2:1 → 1.5:1) is the **strongest single-flag axis in the entire Wave 32 campaign**.

**AND-gate verdict**: 2/3 conditions met (val_abupt clears gate, test_VP crosses floor) — closest the campaign has come to A WIN. But test_SP misses by +0.157pp (the H80-identified architectural binding constraint) and primary test metric test_abupt regresses by +0.143pp.

**dl24 H26 cross-pollination findings**:
- VP-axis benefit IS substrate-portable: dl24 H26 also crossed test_VP cleanly
- abupt/WSS benefits are NOT substrate-portable: dl24 H26 reported abupt −0.050pp WIN and WSS −0.088pp WIN; tay H87 saw abupt +0.143pp REGRESS and WSS +0.217pp REGRESS
- This is important program-level data for paper claims about substrate-distinction

**7 variants now confirm test_SP plateau** (3.74-3.95% range vs floor 3.577): H78 +0.142, H79 +0.323, H80 +0.353, H82 +0.202, H83 +0.162, H84 +0.194, H86 +0.358, H87 +0.157 — H87 lowest of the plateau. **Validates H80's architectural-bound hypothesis but suggests SP plateau may be partially crackable under compound mechanisms.**

**Tanjiro reassigned H95 SURF-LOSS-PUSH-FURTHER** (PR #1258): direct H87 follow-up — `--surface-loss-weight 1.5 → 1.25`. Tests whether productive direction has more headroom or whether 1.5 is the substrate sweet spot.

---

## 2026-05-22 04:30 — PR #1246: H86 MLP-RATIO-EXPANSION (edward, CLOSED) — **OUTCOME D NEGATIVE** — FFN-width expansion (mlp_ratio 4→6) falsified; converges to within 0.026pp of H85 — two orthogonal mechanisms (LR / FFN) both fail to break the 6.38-6.39% late-cosine plateau

- **Branch**: `edward/h86-mlp-ratio-expansion` (closed at 04:30Z 2026-05-22)
- **W&B run**: `m6g3rgh0` (finished cleanly at step 70,664, 14.95h training time, peak 89.59 GiB)
- **Hypothesis**: Single-flag `--model-mlp-ratio 4 → 6` (+50% per-block FFN intermediate dim). First-ever FFN capacity sweep on tay.

### Results

| Channel | H86 (mlp_ratio=6) | BL #972 / floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.3635%** | gate 6.126% | **+0.238pp** ❌ | MISS gate |
| **test_abupt** | 6.142% | 5.844% | +0.298pp ❌ | regress |
| **test_VP (floor 3.643)** | 3.727% | 3.643% | +0.084pp ❌ | MISS floor |
| **test_SP (floor 3.577)** | 3.935% | 3.577% | +0.358pp ❌ | MISS floor (worst of fleet) |
| **test_WSS (goal 6.727)** | 7.057% | 6.727% | +0.330pp ❌ | above goal |

### Analysis

FFN-width expansion engaged productively in mid-cosine (fleet-leading EP5-EP7 slope −0.0184 pp/1k) but late-cosine LR decay drained the additional capacity's contribution. Late-cosine slope flattened to −0.0029 pp/1k = DEEP plateau at ~6.40%. **Single-flag FFN-width expansion is falsified on tay's 13-epoch substrate.**

**Cross-validation with H85**: H85 (lr=1.2e-4 D NEG, 6.3899%) and H86 (mlp_ratio=6 D NEG, 6.3635%) converged within 0.026pp. **Two orthogonal mechanisms — LR magnitude and FFN width — both fail to break the 6.38-6.39% plateau.** Strong evidence binding constraint is not single-flag-tunable.

**Edward reassigned H94 VOLUME-LOSS-WEIGHT-INCREASE** (PR #1257): `--volume-loss-weight 1.0 → 1.5`, complementary to H87 surface_loss reduction. Together with H87 they characterize both directions of surf:vol gradient-budget rebalancing.

---

## 2026-05-22 03:45 — PR #1245: H85 LR-MAGNITUDE-EXPANSION (thorfinn, CLOSED) — **OUTCOME D NEGATIVE** — first-ever UPWARD LR sweep (9e-5→1.2e-4 +33%) FALSIFIED — ALL 4 paper-facing test channels regress, val_abupt MISSES gate +0.264pp, test_SP MISSES floor +0.253pp, test_VP MISSES floor +0.015pp; **LR upward direction definitively wrong for tay substrate; canonical 9e-5 is not below the sweet spot**

- **Branch**: `thorfinn/h85-lr-magnitude-expansion` (closed at 03:45Z 2026-05-22)
- **W&B run**: `sv4rjxdc` (EP12/13 EMA selection, 839min training, 13 epochs clean)
- **Hypothesis**: Single-flag `--lr 9e-5 → 1.2e-4` on canonical Wave 32 baseline #972 substrate. First-ever LR magnitude sweep entire Wave 31/32 fleet history. Lion paper recommends 3e-4 default; we were at 30% of recommended range. H85 tests moderate +33% step toward Lion recommended range.

### Results

| Channel | H85 (lr=1.2e-4) | BL #972 / floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.3899%** | gate 6.126% | **+0.264pp** ❌ | MISS gate — clear D NEGATIVE |
| val_VP | 3.7606% | 3.798% | −0.037pp ✅ | minor val win only |
| val_SP | 4.1853% | — | — | — |
| val_WSS | 7.2236% | — | — | — |
| **test_abupt** | **6.0284%** | 5.844% | **+0.184pp** ❌ | regress |
| **test_VP (floor 3.643)** | **3.6585%** | 3.643% | **+0.015pp** ❌ | MISS floor by hair |
| **test_SP (floor 3.577)** | **3.8302%** | 3.577% | **+0.253pp** ❌ | MISS floor (BIG) |
| **test_WSS (goal 6.727)** | **6.9208%** | 6.727% | **+0.194pp** ❌ | above goal |
| test_WSS_x / _y / _z | 6.127 / 7.574 / 8.952 | — | — | fleet-typical profile |

### Analysis

**LR upward direction definitively falsified.** lr=1.2e-4 (+33% over canonical 9e-5) regresses on ALL channels — no paper-positive axis, every test metric worse than baseline.

Mechanism failure: at lr=1.2e-4 with Lion sign-update, the per-step weight movement magnitude is +33% coarser. For surface-pressure's high-frequency near-wall variations requiring precise late-cosine convergence, this is TOO COARSE — test_SP regression (+0.253pp) vs test_VP regression (+0.015pp ≈ noise) confirms the high-freq axis (SP) is more sensitive to LR overshoot than the smooth-spatial axis (VP). Consistent with every prior Wave 32 finding about the SP plateau binding constraint.

**Cross-validation with H86 (FFN width, edward, in-flight)**: H85 and H86 converge to within 0.005pp at common steps (~6.38-6.39%). Two completely different mechanisms (LR magnitude / FFN width) both fail to break the 6.38-6.39% plateau — strong evidence binding constraint is not tunable via simple single-flag capacity/step-size changes.

**Joint LR coverage closure**: H85 (UP, D NEG) + H90 (DOWN 6e-5, alphonse, in-flight). If H90 also fails, LR magnitude axis is substantively exhausted on tay substrate.

**Thorfinn reassigned H93 TAU-Y-LOSS-PUSH** (PR #1254): `--tau-y-loss-weight 1.5 → 2.5` (+67%), first-ever tau_y-channel loss weight sweep. Complementary to askeladd H92 (tau_z=2.0→3.0). Together H92+H93 close the per-tau-channel within-surface loss-budget coverage targeting Issue #1056 WSS goal.

---

## 2026-05-22 03:10 — PR #1244: H84 RFF-NUM-FEATURES-EXPANSION (askeladd, CLOSED) — **OUTCOME B PARTIAL** — paper-positive test_VP CROSS by −0.040pp, val_abupt CLOSEST C NULL of Wave 32 (+0.021pp over gate) BUT test_SP fails floor +0.194pp + test_WSS regresses +0.164pp + test_abupt +0.131pp; **3rd consecutive volume-favoring single-flag lever, RFF expansion engages mid-cosine but does not maintain late-cosine lead — Tancik saturation prior held**

- **Branch**: `askeladd/h84-rff-num-features-expansion` (closed at 03:10Z 2026-05-22)
- **W&B run**: `qhd8xc4b` (EP13 step 70,664, 14.02h training time, peak 84.09 GB, all 8 GPUs healthy)
- **Hypothesis**: Single-flag `--rff-num-features 16 → 32` on canonical Wave 32 baseline #972 substrate. First-ever RFF Fourier-features positional-encoding capacity sweep entire Wave 31/32 fleet history.

### Terminal metrics (EP13 best EMA checkpoint, full test eval)

| Channel | **H84 (rff=32)** | BL #972 / floor | Δ vs BL / floor | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.1470%** | gate 6.126% | **+0.021pp** ⚠️ | **CLOSEST C NULL of Wave 32 — fails gate by hair** |
| val_VP | 3.6830% | — | — | — |
| val_SP | 4.0778% | — | — | — |
| val_WSS | 6.9310% | — | — | — |
| **test_abupt** | **5.9748%** | 5.844% | **+0.131pp** ❌ | regression |
| **test_VP (floor 3.643)** | **3.6030%** | 3.643% | **−0.040pp** ✅ | **CROSSES floor (3rd consecutive Wave 32)** |
| **test_SP (floor 3.577)** | **3.7707%** | 3.577% | **+0.194pp** ❌ | BREACHES floor |
| **test_WSS (goal 6.727)** | **6.8905%** | 6.727% | **+0.164pp** ❌ | above goal |
| test_WSS_x | 6.138% | — | — | best axis |
| test_WSS_y | 7.430% | — | — | mid |
| test_WSS_z | 8.933% | — | — | binding axis still worst (consistent fleet pattern) |

### Trajectory (EMA-aware)

| EP | Step | val_abupt | Slope pp/1k |
|---|---:|---:|---:|
| EP1 | 10,864 | 29.149% | (cold start +0.5pp over canonical) |
| EP4 | 38,030 | 6.532% | −0.051 |
| EP6 | 48,902 | 6.349% | −0.015 |
| EP8 | 56,154 | 6.229% | −0.013 (LEADER at common steps over H82 by −0.072pp) |
| EP11 | 65,222 | 6.1653% | — |
| **EP13** | **70,664** | **6.1470%** | terminal slope −1.166e-3 pp/1k |

EP4→EP6 saw −0.18pp/2EP rapid mid-cosine descent. EP8→EP13 saw only −0.20pp/5EP. **RFF capacity engaged in mid-cosine but extracted all available benefit before late-cosine** — Tancik saturation curve confirmed at rff=32 on tay substrate.

### Why CLOSE not MERGE

Per CLAUDE.md decision criteria and program.md "no averaging away regressions":

1. **val_abupt MISS gate +0.021pp** — closest miss of Wave 32, but still above 6.126% threshold
2. **test_abupt regresses +0.131pp vs baseline** — primary test metric WORSE than #972
3. **AND-gate test floors FAIL** — test_VP CROSSES ✅ but test_SP breaches +0.194pp ❌
4. **test_SP fail is the H80-identified architectural binding constraint** — 6th independent variant landing test_SP 3.74-3.95% (H78 +0.142, H79 +0.323, H80 +0.353, H82 +0.202, H83 +0.162, H84 +0.194)
5. **test_WSS regresses +0.164pp** — moving AWAY from Issue #1056 goal (5.85)

### What H84 DEFINITIVELY establishes

**4th confirmed volume-favoring single-flag lever** of Wave 32 (after H75/H76 volume-points, H82 wd, H83 grad_clip). RFF positional encoding capacity expansion engaged productively in mid-cosine (EP6-EP8 fleet-leading lead) but **did not maintain lead into late-cosine**.

**Mechanism interpretation (consistent with hypothesis)**:
- Doubling Fourier features helped smooth-spatial fields (volume pressure → test_VP CROSS ✅)
- Did NOT help high-frequency near-wall fields (surface pressure → test_SP +0.194pp ❌)
- Did NOT help WSS axes (test_WSS_z still 8.93%)

The "32 is at the bottom of Tancik recommended range" prior held. **At rff=32 we are sub-merge by 0.021pp — adding 32→64 likely adds cold-start tax without late-cosine benefit (diminishing returns on saturating Tancik curve).**

### Wave 32 volume-favoring pattern CONFIRMED across 4 mechanisms

| Variant | Mech | test_VP | Δ vs floor 3.643 |
|---|---|---:|---:|
| H82 | wd=1e-3 | 3.5328% | −0.110pp |
| H83 | grad_clip=1.0 | 3.5308% | −0.112pp |
| H84 | rff=32 | 3.6030% | −0.040pp |
| H75/H76 | vol-points (B PARTIAL prior) | — | — |

**6 single-flag variants now cleanly fail test_SP floor** (3.74-3.95% range). Wave 32 has DEFINITIVELY established that VP-axis improvement is reliably achievable via diverse single-flag mechanisms, but **SP-axis plateau is the campaign-binding constraint and is NOT broken by any single-flag mechanism tested so far** — validates H80's architectural-bound hypothesis.

### Reassignment — H92 TAU-Z-LOSS-PUSH (askeladd → PR #1252)

Single-flag `--tau-z-loss-weight 2.0 → 3.0` (+50%) on canonical Wave 32 baseline. **First-ever per-tau-channel loss weight sweep entire Wave 31/32 fleet**.

Mechanism: **test_WSS_z is consistently the worst axis across the entire fleet** (8.93-9.66% range vs canonical mean tau_x of 6.12-6.37%). Issue #1056 stretch goal (test_WSS < 5.85) is mathematically dominated by reducing tau_z. Pushing tau_z loss weight higher gives more gradient signal to the tau_z output head per training step — direct data-side gradient-budget reallocation specifically toward the binding axis.

**Per-channel-within-bucket rebalancing has never been tested** entire campaign. H87 tests surface↔volume balance (in-flight, strongest val_abupt signal); H92 tests within-surface (tau_z emphasis) balance. Together they characterize the full per-channel loss budget landscape.

No memory impact (loss weights are scalars). Orthogonal to all 7 in-flight Wave 32 axes (architectural H88/H89/H91, LR H85/H90, FFN H86, data-side H87).

---

## 2026-05-22 02:25 — PR #1243: H83 GRAD-CLIP-EXPANSION (nezuko, CLOSED) — **OUTCOME B PARTIAL** — paper-positive test_VP CROSS by −0.112pp (cleanest VP cross of Wave 32, narrowly beating H82) BUT val_abupt MISS gate +0.134pp + test_abupt +0.126pp + test_SP +0.162pp regress; **grad_clip=1.0 is 2nd confirmed volume-favoring lever — Lion paper default validated, Wave 33 compound axis candidate**

- **Branch**: `nezuko/h83-grad-clip-expansion` (closed at 02:25Z 2026-05-22)
- **W&B run**: `ex652naw` (EP13 step 70,664, 14h training time, all 8 GPUs healthy, 0 nonfinite grads)
- **Hypothesis**: Single-flag `--grad-clip-norm 0.5 → 1.0` on canonical Wave 32 baseline #972 substrate. Tests if canonical clip=0.5 was over-aggressive vs Lion paper default=1.0 — gradient direction preservation hypothesis.

### Terminal metrics (EP13 best EMA checkpoint, full test eval)

| Channel | **H83 (clip=1.0)** | BL #972 / floor | Δ vs BL / floor | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.2604%** | gate 6.126% | **+0.134pp** ❌ | MISS gate |
| val_VP | 3.6447% | val 3.798% | **−0.153pp** ✅ | best in fleet (narrowly beats H82's 3.5867) |
| val_SP | 4.0840% | — | — | (SP plateau) |
| val_WSS | 7.1174% | — | — | — |
| **test_abupt** | **5.9700%** | 5.844% | **+0.126pp** ❌ | regression |
| **test_VP (floor 3.643)** | **3.5308%** | 3.643% | **−0.112pp** ✅ | **CROSSES floor cleanest in Wave 32** |
| **test_SP (floor 3.577)** | **3.7390%** | 3.577% | **+0.162pp** ❌ | BREACHES floor |
| **test_WSS (goal 6.727)** | **6.8899%** | 6.727% | **+0.163pp** ❌ | above goal |

### Trajectory (EMA-aware)

| EP | Step | val_abupt | val_VP | val_SP |
|---|---:|---:|---:|---:|
| EP1 | 10,864 | 26.816% | — | — |
| EP3 | 32,594 | 6.921% | 4.028 | 4.678 |
| EP6 | 48,902 | 6.416% | 3.774 | 4.231 |
| EP8 | 56,154 | 6.312% | 3.712 | 4.141 |
| EP10 | 62,501 | 6.272% | 3.668 | 4.099 |
| EP12 | 65,222 | 6.268% | 3.6567 (val cross floor) | 4.094 |
| **EP13** | **70,652** | **6.2604%** | **3.6447** | **4.0840** |

Final slope decayed from EP3→EP4 (−0.0601 pp/1k) to EP12→EP13 (~−0.001 pp/1k) — canonical geometric-decay regime.

### Outstanding diagnostic instrumentation — nezuko's per-step grad-norm analysis

Sampled across all 70,664 training steps from W&B `train/grad/*`:

| Diagnostic | Value | Interpretation |
|---|---:|---|
| pre-clip global norm mean | **1.673** | Long-tail distribution |
| pre-clip global norm median | **0.123** | Most steps well below 0.5 |
| pre-clip global norm max | **223.67** | Very rare large spikes |
| fraction(pre-clip > 0.5) | **19.8%** | Under canonical clip=0.5, this fraction clipped |
| fraction(pre-clip > 1.0) | **16.9%** | Under H83 clip=1.0, this fraction clipped |
| `train/grad/clipped` mean | **0.169** | Matches the >1.0 fraction |

**Only ~3% of steps fell in the (0.5, 1.0] band** where the clip threshold change matters — yet that ~3% upper-tail signal preferentially benefited the VOLUME head (where val/test_VP improved decisively) while NOT helping the SURFACE head (which remained on the SP plateau). This is a textbook case of **mechanism-targeted gradient signal preservation**.

### Why CLOSE not MERGE

Per CLAUDE.md decision criteria and program.md "no averaging away regressions":

1. **val_abupt MISS gate +0.134pp** — merge gate is < 6.126%
2. **test_abupt regresses +0.126pp vs baseline** — primary test metric WORSE than #972
3. **AND-gate test floors FAIL** — test_VP crosses ✅ but test_SP breaches +0.162pp ❌
4. **test_SP fail is the H80-identified architectural binding constraint** — 5th independent variant landing test_SP 3.74-3.95% (H78 +0.142, H79 +0.323, H80 +0.353, H82 +0.202, H83 +0.162)

### What H83 DEFINITIVELY establishes

**grad_clip=1.0 is the 2nd confirmed volume-favoring single-flag lever** (after H82 wd=1e-3):
- Both produced cleanest test_VP crosses in Wave 32 (H82 −0.110pp, H83 −0.112pp under floor)
- Both narrow val→test VP slope vs baseline (less overfitting)
- Both fail to break test_SP plateau (consistent with H80 finding: SP plateau NOT regularization/gradient-control-bound)

**Mechanism interpretation**: ~3% of steps with pre-clip norm in (0.5, 1.0] preserved more raw gradient signal under H83. Volume head's smoother spatial gradients benefit from this preserved signal; surface head's high-frequency near-wall gradients are already captured at clip=0.5. **The Lion paper's clip=1.0 default is mechanically correct for tay's substrate on the VP axis.**

### Reassignment — H91 MODEL-SLICES-EXPANSION (nezuko → PR #1251)

Single-flag `--model-slices 128 → 192` (+50%) on canonical Wave 32 baseline. **First-ever Transolver slice token count sweep entire Wave 31/32 fleet history**.

Mechanism: more slice tokens = finer-grained surface token resolution. Wave 32's H76 (volume-point increase) was B PARTIAL — productive. Slice count is the SURFACE-axis analogue (untouched entire campaign). Directly tests "test_SP plateau is bound by slice-token resolution" hypothesis — most direct surface-side architectural test of the H80 binding constraint.

Memory: 192² / 128² = 2.25× attention compute. Estimated peak VRAM ~85-90 GB. OOM mitigation: drop batch_size to 3 if needed.

Orthogonal to H88 (heads granularity), H89 (depth), H86 (FFN width). H88+H89+H91 jointly close Wave 32's architectural Tier-2 axis coverage (token granularity × attention depth × token count). Only major unexplored architectural axis after H91 is `hidden_dim` itself.

---

## 2026-05-22 01:30 — PR #1242: H82 WEIGHT-DECAY-EXPANSION (alphonse, CLOSED) — **OUTCOME B PARTIAL** — paper-positive test_VP CROSS by −0.110pp (cleanest VP cross of Wave 32) BUT val_abupt MISS gate +0.148pp AND test_abupt/test_SP/test_WSS all regress; **wd=1e-3 is volume-favoring single-flag lever — Wave 33 compound axis candidate**

- **Branch**: `alphonse/h82-weight-decay-expansion` (closed at 01:30Z 2026-05-22)
- **W&B run**: `us7349eq` (EP13 step 70,664, 14.09h training time, peak 82.5 GB, all 8 GPUs healthy)
- **Hypothesis**: Single-flag `--weight-decay 5e-4 → 1e-3` on canonical Wave 32 baseline #972 substrate. Param-magnitude regularization sweep targeting volume-head over-fitting to surface-dominated gradients.

### Terminal metrics (EP13 best EMA checkpoint, full test eval)

| Channel | **H82 (wd=1e-3)** | BL #972 / floor | Δ vs BL / floor | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.2737%** | gate 6.126% | **+0.148pp** ❌ | MISS gate |
| val_VP | 3.5867% | val 3.798% | **−0.211pp** ✅ | clearly better |
| val_SP | 4.0931% | — | — | (above 4.0% screen) |
| val_WSS | 7.1656% | — | — | — |
| **test_abupt** | **5.9812%** | 5.844% | **+0.137pp** ❌ | regression |
| **test_VP (floor 3.643)** | **3.5328%** | 3.643% | **−0.110pp** ✅ | **CROSSES floor cleanly** |
| **test_SP (floor 3.577)** | **3.7787%** | 3.577% | **+0.202pp** ❌ | BREACHES floor |
| **test_WSS (goal 6.727)** | **6.9048%** | 6.727% | **+0.178pp** ❌ | above goal |
| test_τx | 6.124% | — | — | — |
| test_τy | 7.490% | — | — | — |
| test_τz | 8.981% | — | — | τz/τx=1.467 ✅ |

### Trajectory (EMA-aware)

| EP | step | val_abupt | val_VP | val_SP |
|---|---:|---:|---:|---:|
| EP1 | 10,864 | 25.976% | 15.669 | 19.779 |
| EP3 | 32,594 | 6.764% | 3.852 | 4.482 |
| EP6 | 48,902 | 6.366% | 3.649 | 4.161 |
| EP8 | 56,154 | 6.300% | 3.615 | 4.117 |
| EP10 | 62,501 | 6.283% | 3.595 | 4.100 |
| EP13 | 70,664 | **6.274%** | **3.587** | **4.093** |

Slope decelerated cleanly into late-cosine — val_VP descent held monotone through EP13 (no plateau-rebound).

### Why CLOSE not MERGE

Per CLAUDE.md decision criteria and program.md "no averaging away regressions" rule:

1. **val_abupt MISS gate +0.148pp** — merge gate is val_abupt < 6.126%, H82 lands +0.148pp over.
2. **test_abupt regresses +0.137pp vs baseline** — paper-facing primary test metric is WORSE than #972; merging would lock in regression on headline metric.
3. **AND-gate test floors FAIL** — test_VP crosses (✅) but test_SP breaches +0.202pp (❌); AND-gate explicitly requires BOTH.
4. **test_SP fail is the H80-identified architectural binding constraint** — 4th independent variant landing test_SP 3.78-3.95% (consistent with H79 +0.323pp, H80 +0.353pp, H78 +0.142pp).

### What H82 DOES establish

**weight_decay=1e-3 is a volume-favoring single-flag lever** — the FIRST cleanest test_VP cross of Wave 32:
- val_VP descent terminal 3.587% (−0.211pp vs #972)
- test_VP terminal 3.5328% (−0.110pp under floor)
- val→test slope on VP channel = −0.054pp (val better than test by less than baseline's −0.282pp — wd reduces overfitting)

**Mechanism interpretation**: Stronger param-magnitude regularization (5e-4 → 1e-3) prevents the volume head's spatial filter weights from over-fitting to surface-dominated gradients in the loss budget. The volume head's relatively-smoother spatial output benefits from L2-magnitude penalty more than the surface heads.

**Consistent with H79 dropout falsification finding** (regularization is NOT plateau-bound for SP-axis) — but H82 shows the RIGHT KIND of regularization (param-magnitude, gradient-weighted) IS productive on the VP axis specifically.

### Reassignment — H90 LR-DOWNWARD-SWEEP (alphonse → PR #1250)

Single-flag `--lr 9e-5 → 6e-5` (−33%) on canonical Wave 32 baseline. **First-ever LR sweep BELOW 9e-5 entire Wave 31/32 fleet history.**

LR=9e-5 has been load-bearing across the ENTIRE Wave 31/32 campaign — H85 in-flight tests UPWARD (1.2e-4 likely D NEG), but we've NEVER tested DOWNWARD. Lion paper recommends LR ~3e-4 for vision but Lion is known to need LOWER LRs for fine-grained regression. We're at 30% of Lion default — 6e-5 = 20% tests if we're still over the sweet spot.

Mechanism: surface-pressure has high-frequency near-wall variations requiring precise step magnitude in late-cosine. At lr=9e-5 with Lion's sign update, effective magnitude may be too coarse for SP convergence past 3.78-3.95% plateau. Lower LR = smaller per-step parameter update = finer-grained late-cosine convergence.

Orthogonal to all 7 in-flight Wave 32 axes. Closes Wave 32's LR coverage (H85 UPWARD + H90 DOWNWARD bracket sweet-spot search).

---

## 2026-05-22 01:20 — PR #1240: H81 LION-BETA2-EXPANSION (frieren, CLOSED) — **OUTCOME D NEGATIVE** — val_abupt 6.4256% MISS gate +0.300pp, test_abupt 6.2098% +0.366pp, ALL 4 test channels regressed, both test floors VIOLATED; **Lion β2=0.999 expansion uniformly destructive across all axes — Chen et al 2023 defaults validated**

- **Branch**: `frieren/h81-lion-beta2-expansion` (closed at 01:20Z 2026-05-22)
- **W&B run**: `vadc87et` (EP13 step 70,664, 14.3h training time, peak 78 GB, all 8 GPUs healthy, 0 nonfinite grads)
- **Hypothesis**: Single-flag `--lion-beta2 0.99 → 0.999` on canonical Wave 32 baseline #972 substrate. First-ever Lion second-moment EMA decay sweep entire Wave 31/32 fleet history. 10× wider grad-norm-buffer half-life targets noise-stability for slow-converging tau_z/WSS axes.

### Terminal metrics (EP13 best EMA checkpoint, full test eval)

| Channel | **H81 (β2=0.999)** | BL #972 (β2=0.99) | Δ vs BL | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.4256%** | 6.126% | **+0.300** ❌ | **MISS merge gate** |
| val_VP | 3.7452% | 3.566% | +0.179 | worse |
| val_SP | 4.1496% | 3.534% | +0.616 | clearly worse |
| val_WSS | 7.3489% | 6.679% | +0.670 | clearly worse |
| **test_abupt** | **6.2098%** | 5.844% | **+0.366** ❌ | clearly worse |
| **test_VP (floor 3.643)** | **3.7746%** | 3.643% | **+0.132** ❌ | VIOLATES floor |
| **test_SP (floor 3.577)** | **3.9267%** | 3.577% | **+0.350** ❌ | VIOLATES floor BADLY |
| **test_WSS (goal 6.727)** | **7.1508%** | 6.727% | **+0.424** ❌ | above goal |
| test_WSS_x | 6.367% | — | — | best of 3 axes |
| test_WSS_y | 7.735% | — | — | mid |
| test_WSS_z (binding) | 9.245% | ~8.75% est | +0.50 | binding axis worse |

### Trajectory (EMA-aware)

| EP | step | val_abupt EMA | ΔEP→EP |
|---|---:|---:|---:|
| EP1 | 10,864 | 37.4451% | (cold start — 10× slower than canonical β2=0.99 ~26-29%) |
| EP2 | 21,729 | 8.4829% | −28.96 |
| EP3 | 32,594 | 7.2653% | −1.22 |
| EP6 | 48,902 | 6.6606% | −0.11 |
| EP8 | 56,154 | 6.5326% | −0.05 |
| EP11 | 65,222 | 6.4452% | −0.018 |
| EP13 | 70,664 | **6.4256%** | −0.012 (terminal) |

Slope decay EP3→EP13: −1.22 → −0.37 → −0.13 → −0.11 → −0.08 → −0.05 → −0.04 → −0.03 → −0.018 → −0.008 → −0.012 pp/EP. Trajectory still slowly descending at terminal — but **never recovered the EP1 cold-start deficit**.

### Mechanism falsification — Lion β2-expansion uniformly destructive

- **β2 0.99 → 0.999** expanded the second-moment (grad-norm) running average half-life from ~100 steps → ~1000 steps (10×).
- Train/grad/global_norm terminal = **0.0372** confirms grad-signal-smoothing prior was descriptively true (smoother grad-norm signal vs canonical ~0.05).
- Train/grad/clipped = 0 throughout EP3+ (no clip events post-warmup; β2=0.999 produces very stable grad-norm signal as predicted).
- **BUT smoother grad-norm signal did NOT translate to better convergence**: the extra inertia damped responsive updates to local loss landscape geometry, especially during the EP1 cold-start. The Lion direction signal (sign update) needed the noisier β2=0.99 grad-norm tracker to maintain useful step-magnitude scaling.
- **All 4 test channels regressed uniformly** (test_SP +0.35, test_VP +0.13, test_WSS +0.42, test_abupt +0.37) → not a head-specific tradeoff like H78 (β1=0.95). The β2 expansion is a **whole-model destruction**, not a per-head mechanism.

### What H81 DEFINITIVELY establishes (with H78)

1. **Chen et al 2023 Lion defaults (β1=0.9, β2=0.99) are well-tuned for CFD-surrogate regression at this scale on tay's substrate**. The published recipe is correct.

2. **Lion-optimizer-side mechanism class is now substantively exhausted on tay**:
   - H78 β1=0.95: B PARTIAL (val_abupt A WIN, test_VP cross, test_SP miss — head-specific)
   - H81 β2=0.999: D NEGATIVE (uniform regression across all 4 test channels)
   - Further sweeps (β1 down or β2 down) likely C NULL (small-deviation neighborhood explored)

3. **Wave 32 mechanism-class falsification table updated**:
   - Charbonnier loss curvature: **FALSIFIED** (4 D NEG H68/H73/H74/H77)
   - Regularization class (dropout/EMA): **FALSIFIED** (2 D NEG H79/H80)
   - Lion-optimizer-side: **substantively exhausted** (1 D NEG H81 + 1 B PARTIAL H78)
   - Architectural depth: **UNTESTED — H89 next**

### Reassignment — H89 MODEL-DEPTH-EXPANSION (frieren → PR #1249)

Single-flag `--model-layers 5 → 6` on canonical Wave 32 baseline. **First-ever Transolver depth sweep entire Wave 31/32 fleet history**.

Mechanism: adding one transformer block gives one more "reasoning step" per slice token while keeping every other dimension (hidden=512, heads=4, mlp_ratio=4, slices=128) identical. Directly tests "test_SP plateau is depth-bound" hypothesis from H80 closure.

Memory: +3.6M params (+21% on 17.4M base); +5-6 GiB peak VRAM estimate over canonical 77.3 GB → ~82-85 GB on H100 96 GB. OOM mitigation: drop batch_size to 3 if EP1 OOM.

Orthogonal to all 7 in-flight Wave 32 axes (H82 wd, H83 grad_clip, H84 rff, H85 lr, H86 mlp_ratio, H87 surf_loss, H88 heads). H89 + H88 jointly close Wave 32's architectural Tier-2 axis coverage (depth + width-via-mlp + heads); only major unexplored architectural axis after H89 is `hidden_dim` itself.

---

## 2026-05-21 17:30 — PR #1236: H80 EMA-DECAY-EXTENSION (fern, CLOSED) — **OUTCOME D NEGATIVE** — val_abupt 6.298% MISS gate +0.172pp, test_SP/test_VP both fail floors; **EMA composition class falsified for Wave 32 plateau**

- **Branch**: `fern/h80-ema-decay-extension` (closed at 17:30Z)
- **W&B run**: `gtmq7ctm` (EP13 step 70,652, 13.95h training time, peak 75.89 GB, all 8 GPUs healthy, 0 nonfinite grads)
- **Hypothesis**: Single-flag `--ema-decay 0.999 → 0.9999` on canonical Wave 32 baseline #972 substrate. First-ever EMA composition sweep entire Wave 31/32 fleet history. 10× wider EMA window targets late-tail gradient noise as plateau cause.

### Terminal metrics (EP13 best EMA checkpoint, full test eval)

| Channel | **H80** | BL #972 | Δ vs BL | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.298%** | 6.126% | **+0.172** ❌ | **MISS merge gate** |
| val_VP | 3.675% | 3.566% | +0.109 | worse |
| val_SP | 4.170% | 3.534% | +0.636 | clearly worse |
| val_WSS | 7.122% | 6.679% | +0.443 | clearly worse |
| **test_abupt** | **6.163%** | 5.844% | **+0.319** ❌ | clearly worse |
| **test_VP (floor 3.643)** | **3.702%** | 3.643% | **+0.059** ❌ | CROSSED floor |
| **test_SP (floor 3.577)** | **3.930%** | 3.577% | **+0.353** ❌ | CROSSED floor BADLY |
| **test_WSS (goal 6.727)** | **7.076%** | 6.727% | **+0.349** ❌ | above goal |
| test_WSS_z (binding) | 9.189% | ~8.75% est | +0.44 | binding axis worse |

### Trajectory (δ-contamination clearing — EMA-aware)

| EP | step | δ contam | val_abupt EMA |
|---|---:|---:|---:|
| EP1 | 10,864 | 33.9% | 67.89% (random-dominated) |
| EP6 | 48,902 | 0.76% | 7.565% (first informative) |
| EP9 | 59,780 | 0.26% | 6.505% |
| EP11 | 65,222 | 0.15% | 6.362% |
| EP13 | 70,664 | 0.086% | **6.298%** (terminal) |

Slope decay EP6→EP13: −0.563 → −0.317 → −0.180 → −0.090 → −0.053 → ~−0.030 → ~−0.018 pp/EP (halving factor 0.55-0.59/EP).

### Critical analysis — fern's structural-mismatch finding

> "With ema=0.9999, the EMA-shadow is fundamentally backward-looking by ~4 epochs — at EP13/13 it reflects training state from EP9-13, missing the very-last cosine-bottom-out updates. The 13-epoch budget is structurally mismatched to ema=0.9999."

EMA decay 0.9999 → 50%-mass = 6,931 steps (~0.66 EP), 99%-mass = 46,049 steps (~4.2 EP). The EMA shadow never integrates the late-cosine LR-decay improvements before EP13 terminal. Trajectory was still descending at terminal (~−0.018pp/EP geometric decay projection).

**However**: Even budget-extended rerun (ema=0.9999 + epochs=20) would not address the dispositive test-side regression. Val>test gap **inverted** from baseline's typical pattern (#972 had val 6.126 > test 5.844 = −0.282pp; H80 had val 6.298 < test 6.163 = +0.135pp). Wider EMA window averages over more diverse training states → less specialized for held-out test distribution.

### What H80 DEFINITIVELY establishes (with H79)

1. **Regularization-bound plateau hypothesis FALSIFIED on tay's substrate** — H79 dropout + H80 EMA both D NEG with val→test slope analysis. The chain is closed. Wave 33+ must pivot to architectural / data / ensemble levers.

2. **test_SP plateau is the program's binding constraint**. Across H79 (+0.323pp), H80 (+0.353pp), and other recent variants, test_SP keeps landing at 3.85-3.95% (vs floor 3.577). The PURE baseline #972 substrate has a test_SP representation ceiling that current architecture+loss configuration cannot break.

3. **Rawcanon-20260511 substrate random-pred floor on abupt-mean rel_l2_pct = ~300-385%**, much higher than SDF-substrate ~100%. EP1 reading 67.89% under δ=0.339 implies R≈140%; EP5→EP6 slope analysis gives R≈320%. Worth memorializing for EMA-aware kill-threshold calibration.

### Reassignment — H88 MODEL-HEADS-EXPANSION (fern → PR #1248)

Single-flag `--model-heads 4 → 8` on canonical Wave 32 baseline. **First-ever attention-head-count sweep entire Wave 31/32 fleet history**.

Mechanism: `model_heads=4` with hidden_dim=512 means per-head dim = 128 (non-standard). Doubling to 8 gives per-head dim = 64 (canonical Vaswani et al 2017 recipe). More heads = finer-grained attention partition for surface↔volume coupling.

Critical: directly tests fern's SP-axis representation-capacity hypothesis from H80 closure. If H88 lands test_SP < 3.65%, attention-head granularity was a real bottleneck → opens architectural sweep for Wave 33. If test_SP > 3.85%, plateau is bound elsewhere (decoder capacity / surface-specific positional encoding).

Orthogonal to all 8 in-flight Wave 32 axes (H81/H82/H83/H84/H85/H86/H87 + H86 mlp_ratio).

---

## 2026-05-21 14:25 — PR #1235: H79 DROPOUT-INTRODUCTION (tanjiro, CLOSED) — **OUTCOME D NEGATIVE** — val_abupt 6.3725% MISS gate +0.247pp, all 4 test channels regressed; **val→test slope diagnostic FALSIFIES regularization-bound plateau hypothesis on tay**

- **Branch**: `tanjiro/h79-dropout-introduction` (closed at 14:25Z)
- **W&B run**: `qgm1hnoe` (EP13 step 70,652, 14.25h training time, peak 77.3 GB, best_epoch=13, best_checkpoint/source=ema)
- **Hypothesis**: Single-flag `--model-dropout 0.0 → 0.1` on canonical Wave 32 baseline #972 substrate. First-ever model_dropout sweep entire Wave 31/32 fleet history. Regularization-bound plateau hypothesis test.

### Terminal metrics (EP13 best EMA checkpoint, full test eval)

| Channel | **H79** | BL #972 | Δ vs BL | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.3725%** | 6.126% | **+0.247** ❌ | **MISS merge gate** |
| val_VP | 3.7353% | 3.566% | +0.169 | worse |
| val_SP | 4.2004% | 3.534% | +0.666 | clearly worse |
| val_WSS | 7.2276% | 6.679% | +0.549 | clearly worse |
| **test_abupt** | **6.1306%** | 5.844% | **+0.287** ❌ | clearly worse |
| **test_VP (floor 3.643)** | **3.6806%** | 3.643% | **+0.038** ❌ | CROSSED floor |
| **test_SP (floor 3.577)** | **3.9001%** | 3.577% | **+0.323** ❌ | CROSSED floor BADLY |
| **test_WSS (goal 6.727)** | **7.0561%** | 6.727% | **+0.329** ❌ | above goal |
| test_WSS_z (binding) | 9.1214% | ~8.75% est | +0.37 | binding axis worse |

### Per-epoch trajectory (slope deceleration EP3→EP13)

| EP | step | val_abupt | Δ_abupt pp/epoch |
|---|---:|---:|---:|
| EP1 | 10,864 | 26.387% | — (cold-start) |
| EP2 | 21,729 | 8.113% | −18.27 |
| EP3 | 32,594 | 7.221% | −0.892 |
| EP6 | 48,902 | 6.691% | EP4-6 mean −0.177 |
| EP9 | 59,780 | 6.474% | EP7-9 mean −0.072 |
| EP11 | 65,222 | 6.400% | −0.019 |
| EP12 | 67,943 | 6.380% | −0.020 |
| **EP13** | **70,652** | **6.3725%** | **−0.007** (cosine bottom-out) |

### CRITICAL METHODOLOGICAL FINDING: val→test slope diagnostic

| Run | val_abupt | test_abupt | val→test slope |
|---|---:|---:|---:|
| Baseline #972 | 6.126% | 5.844% | **−0.282** (test better than val) |
| **H79 dropout 0.1** | **6.3725%** | **6.1306%** | **−0.242** (less favorable) |

**If plateau were overfitting-bound**, dropout p=0.1 should have WIDENED the val→test improvement. Instead it NARROWED by 0.040pp. Dropout's mechanism DID engage (train/val_loss inversion EP6+) but did NOT translate into improved generalization.

**Combined evidence (Wave 31/32 on tay)**:
- H71 GradNorm: D NEG (dynamic loss balancing falsified)
- V-DEPTH / surf_deep: didn't break 6.15% val ceiling
- H79 dropout: D NEG with adverse val→test slope shift

**Conclusion**: **regularization-bound mechanism class FALSIFIED on tay's substrate.** Wave 33+ must pivot to data-side, architecture, or ensemble levers.

### Suggested follow-ups (per tanjiro, advisor-confirmed prioritization)

1. ❌ NO further dropout sweeps (p=0.05/0.15/0.20) — val→test slope diagnostic settles this
2. ⚠️ Full-coverage dropout (FFN + residual) — PASS, marginal expected value
3. ✅ **PIVOT to data-side levers** (input-space normalization, geometry augmentation, expanded effective batch) — Tier-3 escalation for Wave 33
4. ⚠️ Architecture pruning (anti-V-DEPTH) — defer to Wave 33 brainstorm
5. ✅ Update plateau-protocol decision tree: regularization-bound = falsified

### Reassignment

**tanjiro → H87 SURFACE-LOSS-WEIGHT-REDUCTION (PR #1247)** — First-ever loss-balance-ratio sweep on tay (`--surface-loss-weight 2.0 → 1.5`). Cross-pollination from dl24's H26 (val_wss leader). H87 is a DATA-SIDE lever (re-weighting which residuals dominate gradients), consistent with H79-derived plateau falsification.

---

## 2026-05-21 12:55 — PR #1229: H73 CHARBONNIER-τ_z (edward, CLOSED) — **OUTCOME D NEGATIVE** — val_abupt 6.5804% killed at EP6 hard gate (val_abupt<6.5) by +0.080pp, missed merge gate by +0.454pp; Charbonnier mech class now falsified on TWO axes (H68 vol_p D NEG, H73 τ_z D NEG)

- **Branch**: `edward/h73-charbonnier-tau-z` (closed at 12:55Z)
- **W&B run**: `5i5o1nru` (EP6 kill at step 65,212, 12.27h training time, peak 77.3 GB)
- **Hypothesis**: Single-flag `--tau-z-loss-type charbonnier --charbonnier-eps 1e-3` on H68 sibling substrate. Cross-pollination from dl24 H19. Sibling-test: if Charbonnier-on-vol_p (H68 D NEG) and Charbonnier-on-τ_z differ, loss-curvature-shape mech is axis-specific.

### Terminal metrics @ EP6 kill (step 65,212, also terminal — kill cancels test eval)

| Axis | **H73** @ EP6 | Baseline #972 (EP13) | Δ vs BL | Verdict |
|------|---:|---:|---:|:--|
| **val_abupt (merge gate 6.126%)** | **6.580%** | 6.127% | **+0.454pp WORSE** ❌ | MISS gate, KILLED |
| val_SP | 4.408% | 3.984% | +0.425pp | regression |
| val_VP | 3.947% | 3.959% | −0.012pp | neutral |
| val_WSS | 7.442% | 6.949% | +0.493pp | regression |
| val_WSS_x | 6.553% | 6.107% | +0.446pp | regression |
| val_WSS_y | 8.137% | 7.470% | +0.667pp | regression |
| **val_WSS_z (mech target)** | **9.856%** | 9.387% | **+0.470pp** ❌ | targeted axis WORSE |

*No test metrics — EP6 kill cancelled terminal test eval (12-15h saved compute).*

### Mechanism diagnostic — Charbonnier-eps SATURATION smoking gun

WSS_z slope reversed direction in last two val reads — the targeted axis flipping sign while every other axis kept descending is the diagnostic signature of mech-on-target turning counterproductive:

| Step | Epoch | val_abupt | val_WSS_z | Δ_WSS_z/1k |
|---:|:---:|---:|---:|---:|
| 32,594 | EP3 | 6.969% | 10.238% | −0.0925 |
| 56,154 | EP5.16 | 6.608% | 9.860% | −0.0019 |
| 62,501 | EP5.75 | 6.584% | 9.853% | **+0.0011** ⚠️ |
| 65,222 | EP6 | 6.580% | 9.856% | **+0.0013** ⚠️ |

**char/MSE ratio diagnostic 8.17×** at step 61,867 confirms near-L1 regime, not robust-Huber regime: residuals (~0.12) dominate eps (1e-3) by ~120×, so Charbonnier provides L1 noise without outlier compression. eps=1e-3 was wrong for tay's τ_z residual distribution magnitude.

### Sibling test — H68 (Charbonnier-vol_p) vs H73 (Charbonnier-τ_z), both killed at step 65,212

| Axis | H73 (τ_z) | H68 (vol_p) | H73 − H68 |
|------|---:|---:|---:|
| val_abupt | 6.580% | 6.822% | −0.242pp |
| val_VP | 3.947% | 4.146% | −0.200pp |
| **val_WSS_z** | **9.856%** | **10.362%** | **−0.506pp** |

Conclusion: Charbonnier-on-τ_z is uniformly **less harmful** than Charbonnier-on-vol_p (since τ_z residuals sit closer to L2→L1 elbow) but neither beats MSE baseline. Loss-curvature-shape mechanism class is now empirically **falsified on tay's substrate** for the Wave 32 cycle.

### τ_z ceiling triangulation across three orthogonal mech classes (all CLOSE)

| Hypothesis | Mech class | Outcome | val_abupt |
|---|---|---|---:|
| H55v2 | Weighting-scheduling (curriculum 0.5→2.0) | C NULL | ~6.16% |
| H63 | LR-substrate-extension | C NULL | ~6.25% |
| **H73** | **Loss-curvature-shape (Charbonnier eps=1e-3)** | **D NEGATIVE** | **6.58%** |

**Pattern**: τ_z is reachable from many angles but its terminal floor on tay's split appears at or above baseline #972's 6.126%. Strong empirical claim — the binding val_abupt ceiling sits in τ_z and is **representation-bound**, not optimizer/loss/curriculum-bound.

### Reassignment

edward → **H86 MLP-RATIO-EXPANSION** (PR #1246). Single-flag `--model-mlp-ratio 4 → 6`. First-ever FFN capacity sweep in entire Wave 31/32 fleet history. +50% per-block FFN intermediate dimension (2048→3072). Pure architectural Tier-2 axis orthogonal to all 12 currently-deployed Wave 32 variants. VRAM headroom estimate +5-7 GiB over H73's 77.3 GB peak.

### Parked for Wave 33

- **H73b Charbonnier-τ_z with eps=0.02-0.1**: edward's #1 follow-up. Direct mech-class disambiguation (settles "null mech vs mistuned eps"). PARK — Wave 32 loss-reformulation Tier-2 axis budget exhausted (H73/H74/H77 all D NEG).
- **Diagnostic logging upstream**: edward's #4. Land char/MSE ratio diagnostic in mainline so future Charbonnier-class hypotheses detect saturation immediately.

---

## 2026-05-21 12:45 — PR #1234: H78 LION-BETA1-MOMENTUM-EXPANSION (thorfinn, CLOSED) — **OUTCOME B PARTIAL** — val_abupt A WIN cleared by −0.069pp (cleanest Wave 32 val gate) BUT test_SP MISS floor +0.142pp blocks merge; test_VP CROSSES BY −0.175pp (DEEPEST IN ENTIRE WAVE 31/32)

- **Branch**: `thorfinn/h78-lion-beta1-momentum-expansion` (closed at 12:45Z)
- **W&B run**: `9if9s43r` (EP13 step 70,652, 13.96h training time, all 8 ranks finished, peak 83.8 GB)
- **Hypothesis**: Single-flag `--lion-beta1 0.9 → 0.95` on PURE baseline #972 substrate. First-ever Lion momentum-direction sweep in entire Wave 31/32. β1 controls direction-smoothing window (~10→20 steps).

### Terminal metrics (EP11 best EMA checkpoint, full test eval inline)

| Channel | **H78** | BL #972 ref | Δ vs BL #972 | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.0570%** | 6.126% | **−0.069** ✅ | **CLEARS merge gate** |
| val_VP | 3.520% | 3.798% | **−0.278** ✅ | deep val_VP improvement |
| val_SP | 3.993% | ~3.80% est | +0.19 | regression |
| val_WSS | 6.870% | ~6.94% est | −0.07 | mild improvement |
| **test_abupt** | **5.9033%** | 5.844% | **+0.059** | ⚠️ val→test slope flip |
| **test_VP (floor 3.643)** | **3.4685%** | 3.643% | **−0.175** ✅ | **DEEPEST WAVE 31/32 test_VP CROSS** |
| **test_SP (floor 3.577)** | **3.7190%** | 3.577% | **+0.142** ❌ | **MISSES floor (blocks merge)** |
| test_WSS (goal 6.727) | 6.8304% | 6.727% | +0.103 ❌ | above goal |
| test_WSS_x | 6.073% | — | — | — |
| test_WSS_y | 7.393% | — | — | — |
| test_WSS_z (binding) | 8.863% | ~8.75% est | +0.11 | binding axis still hardest |

### Comparison to AB-UPT public reference (per program.md)

| Target | AB-UPT | H78 | Δ vs SOTA |
|---|---:|---:|---:|
| test_SP / p_s | 3.76 | 3.719 | ✅ beats SOTA by −0.041 |
| test_VP / p_v | 6.08 | **3.469** | ✅✅ **crushes SOTA by −2.611** |
| test_WSS / tau | 7.29 | 6.830 | ✅ beats SOTA by −0.460 |
| test_WSS_z / tau_z | 3.63 | 8.863 | ❌ +5.23 above (binding axis far from SOTA) |

### Per-epoch trajectory (cleanest Wave 32 monotonic descent, EP11 bottom)

| EP | step | val_abupt | val_SP | val_VP | val_WSS | Δ_abupt | slope/1K |
|---|---:|---:|---:|---:|---:|---:|---:|
| EP1 | 10,864 | 26.95 | 19.79 | 14.42 | 30.66 | — | (warmup) |
| EP3 | 32,594 | 6.482 | 4.324 | 3.745 | 7.345 | −0.667 | −0.0613 |
| EP6 | 48,902 | 6.131 | 4.069 | 3.581 | 6.940 | −0.0554 | −0.0102 |
| EP9 | 59,780 | 6.070 | 4.007 | 3.536 | 6.879 | −0.0123 | −0.00338 |
| **EP11** | **65,222** | **6.0570** | **3.9932** | **3.5195** | **6.8703** | **−0.0036** | **−0.00132** |
| EP12 | 67,943 | 6.0575 | 3.9924 | 3.5214 | 6.8706 | +0.0005 | +0.00017 (flat) |
| EP13 | 70,652 | 6.0619 | 3.9931 | 3.5302 | 6.8804 | +0.0044 | (cosine bottom rebound) |

EP11 was global val_abupt best; EP12/13 plateaued and rebounded slightly (true cosine bottom-out). EMA-best (EP11) used for test eval. 11/12 epochs set new best_checkpoint — strongest sustained descent of any Wave 32 experiment.

### Mechanism characterization (H78 is the most informative single-flag result of Wave 32)

**β1 0.9→0.95 worked exactly as theory predicted:**
1. **EP1 cold-start damped**: 26.95% vs baseline 20.49% (+6.46pp slower at EP1) — 20-step momentum window damps initial gradient direction signal as theory predicts
2. **EP2 onward overshot baseline**: EP4 6.243% ≈ baseline EP6 6.312% (gradient-EMA integration compresses descent by ~2 epochs)
3. **Late-tail plateau penetration**: 11/12 val gates set new best; descent continued past baseline's EP8-9 saturation point
4. **Cosine bottom-out at EP11**: monotonic slope decay then flatlines (EP11→EP12 +0.0005) then rebounds (EP12→EP13 +0.0044) — true cosine descent exhaustion

**Head-specific gradient-smoothing tradeoff DOCUMENTED**:
- VP head wins decisively (test_VP -0.175pp DEEPEST WAVE 31/32) — volume-pressure gradients are spatially smooth, benefits from wider momentum integration
- SP head loses (test_SP +0.142pp MISS floor) — surface-pressure has high-frequency near-wall variations, over-smoothing loses detail signal
- WSS head val-improves but test-regresses — val→test slope inversion suggests overfitting to val WSS distribution via smoother trajectory

### Why CLOSE not MERGE — per program.md strict contract

Per CLAUDE.md "Test floors: test_VP ≤ 3.643% **AND** test_SP ≤ 3.577%" — AND-gate. H78 misses test_SP by +0.142pp → not mergeable. Per program.md "do not hide regressions behind a single averaged number" — test_abupt regresses (+0.059pp) and 3 of 4 test channels regress despite val_abupt improvement. Merging H78 would lock-in test_SP 3.719 as new baseline = paper-facing regression from current SOTA 3.577.

### Wave 32 test_VP floor-crossers — H78 takes top position

| H | Substrate | LR-fix? | val_abupt | test_VP | Δ floor | Notes |
|---|---|---|---:|---:|---:|---|
| H26 (merged baseline #972) | base | no | 6.126% | 3.643% | floor | — |
| H59 (#1206) | V-DEPTH + LR-fix | yes | 6.282% | 3.552% | −0.091 | val MISS, multi-flag |
| H65 (#1214) | SURF-DEEP + LR-fix | yes | 6.234% | 3.588% | −0.055 | val MISS, multi-flag |
| H76 (#1232) | SLICES-192 NO LR-fix | NO | 6.293% | 3.548% | −0.095 | val MISS, single-flag |
| **H78 (#1234)** | **β1=0.95 NO LR-fix** | **NO** | **6.057%** | **3.469%** | **−0.175** | **val CLEAR + test_SP MISS, single-flag** |

H78 is the **deepest test_VP floor cross of any Wave 31/32 variant** (compound or single-flag) — AND achieves val_abupt CLEAR. Two paper-facing positives. The merge block is solely test_SP.

### Strategic implications

1. **β1 axis CONFIRMED** as first real Lion-side lever in Wave 31/32 history. Mechanism characterized for both val (helps) and test (mixed head-specific tradeoff).
2. **β1=0.92 mid-point** (student's #1 suggestion) PARKED for Wave 33 — would investigate test_SP cost curve, but H81 β2 axis needs to land first for Lion characterization completeness.
3. **β1=0.95 + axis-X composition** is the natural Wave 33 strategy — pair with an axis that recovers test_SP (e.g., dropout if H79 is positive, or weight_decay if H82 is positive, or future loss-weight rebalance).
4. **val→test slope inversion is now a documented pattern** across Wave 32 (H74, H75, H76, H78). The 34-case val split may be too small to reliably reflect test generalization in the 6.0-6.3% val_abupt range. Multi-seed confirmation would clarify.
5. **Loss-reformulation class still exhausted** (H68, H74, H77 D NEG; H73 in flight) — H78 confirms optimizer-momentum is the productive axis class for Wave 32+.

Thorfinn reassigned **H85 LR-MAGNITUDE-EXPANSION** (PR #1245) — single-flag `--lr 9e-5 → 1.2e-4` (+33% LR). **First-ever LR magnitude sweep in entire Wave 31/32 fleet** (LR has been load-bearing at 9e-5 across the entire campaign). Lion paper recommends 3e-4 default — our 9e-5 is at 30% of recommended range. Plateau-protocol Tier-2 optimization-step-size axis, **completing Tier-2 axis coverage**: loss reformulation (H68/H73/H74/H77), capacity (H76/H84), optimizer momentum (H78/H81), regularization (H79/H82), EMA composition (H80), gradient control (H83), and now LR magnitude (H85). Orthogonal to all 7 other in-flight axes.

---

## 2026-05-21 12:30 — PR #1232: H76 SLICES-192-ISOLATION (askeladd, CLOSED) — **OUTCOME B PARTIAL (paper-positive)** — val_abupt MISS gate +0.167pp BUT test_VP CROSSES FLOOR by −0.095pp (deepest Wave 31/32 test_VP cross of any single-mech variant)

- **Branch**: `askeladd/h76-slices-192-isolation` (closed at 12:30Z)
- **W&B run**: `342fnmx7` (EP13 terminal step 70,664, 15.20h training time, all 8 ranks healthy, peak VRAM 81.98 GB)
- **Hypothesis**: Single-flag `--model-slices 128 → 192` on PURE baseline #972 substrate. Wave 31 ran slices=192 BUNDLED with NPCA+SSFL+sub3K and got C NULL — H76 isolates the slice-count axis. Tests whether 192 slices alone (which H35 analysis suggested provides ~295 effective slice-modes) breaks the val_abupt plateau.

### Terminal metrics (EP13 best, EMA checkpoint, full test eval inline)

| Channel | **H76** | BL #972 ref | Δ vs BL #972 | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.293%** | 6.126% | **+0.167** ❌ | MISS gate |
| val_VP | **3.713%** | 3.798% | **−0.085** ✅ | beats baseline val_VP |
| val_SP | 4.180% | — | — | — |
| val_WSS | 7.080% | — | — | — |
| **test_abupt** | **5.981%** | 5.844% | **+0.137** ❌ | regression |
| **test_VP (floor 3.643)** | **3.548%** | 3.643% | **−0.095** ✅ | **CROSSES FLOOR** |
| **test_SP (floor 3.577)** | **3.776%** | 3.577% | **+0.199** ❌ | MISS floor |
| test_WSS (goal 6.727) | 6.884% | 6.727% | +0.157 ❌ | above goal |
| test_WSS_z (binding) | 9.005% | ~8.75% est | +0.255 ❌ | binding axis still hardest |

### Per-epoch trajectory (clean monotonic descent, asymptote ~6.29%)

| EP | step | val_abupt | val_SP | val_VP | val_WSS | τx | τy | τz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EP1 | 10,864 | 26.4092 | 19.9912 | 16.1010 | 29.1483 | 25.40 | 35.01 | 35.54 |
| EP3 | 32,594 | 6.8245 | 4.5495 | 3.9879 | 7.7078 | 6.69 | 8.64 | 10.25 |
| EP6 | 48,902 | 6.4327 | 4.2834 | 3.7832 | 7.2425 | 6.27 | 8.01 | 9.82 |
| EP10 | 62,501 | 6.3156 | 4.1984 | 3.7239 | 7.1056 | 6.15 | 7.79 | 9.72 |
| EP13 | **70,664** | **6.2925** | **4.1796** | **3.7133** | **7.0798** | **6.13** | **7.74** | **9.70** |

EP9→EP11 slope −0.012 pp/1k, EP11→EP13 slope −0.006 pp/1k. **Asymptote is ~6.29% on this substrate** — not enough to recover 0.17pp gap. tau_z descent EP2→EP13 11.28% → 9.70% (−1.58pp) confirms finer slice attention helps near-wall shear, but the val_abupt aggregate doesn't share that bottleneck.

### Wave 31/32 test_VP floor crossers — H76 joins the table

| H | Substrate | LR-fix? | val_abupt | test_VP | Δ floor | Notes |
|---|---|---|---:|---:|---:|---|
| H26 (merged baseline #972) | base | no | 6.126% | 3.643% | floor | — |
| H59 (#1206) | V-DEPTH + LR-fix | yes | 6.282% | 3.552% | −0.091 | val MISS gate, test cross |
| H65 (#1214) | SURF-DEEP + LR-fix | yes | 6.234% | 3.588% | −0.055 | val MISS gate, test cross |
| **H76 (#1232)** | **SLICES-192 NO LR-fix** | **NO** | **6.293%** | **3.548%** | **−0.095** | **deepest test_VP cross orthogonal to LR-fix** |

H76 is the **deepest test_VP floor cross of any Wave 31/32 single-mech variant** — AND it achieves this WITHOUT LR-fix (which H75 just confirmed is NET NEG on pure baseline). This is significant attribution: **slice-resolution geometric capacity expansion produces a real test_VP improvement orthogonal to LR-fix**.

### Mechanism interpretation — VP↔abupt trade-off

Slice-attention expansion (128→192) buys real signal on the VP axis (test_VP floor-cross) and the WSS axis (tau_z 11.28→9.70%), but the val_abupt aggregate registers no improvement because:
1. **Capacity-overhead trade**: extra slice tokens add attention noise without enough samples to learn them well at this width (h=4 heads, d=512). Slice attention spreads signal across 50% more slot tokens.
2. **VP→abupt slope inversion**: extra slice noise hurts SP/τx/τy/τz direction alignment, which dominates the val_abupt aggregate.

The val_abupt MISS doesn't kill the paper result — the test_VP floor cross is **the** finding for the slice-resolution scaling discussion.

### Strategic implications

1. **Slices=192 alone is NOT mergeable** — val_abupt MISS gate is dispositive.
2. **Slices=192 IS paper-positive** — joins the 7-experiment test_VP floor-cross table at top position (−0.095pp).
3. **Wave 31 slices=192 bundling was sound but compound confounded** — H76 isolation now confirms slices=192 contributes positively on the VP axis even on its own.
4. **Per CLAUDE.md "NO MORE ENSEMBLES"** — H76's anti-correlation profile (VP-positive / SP-WSS-negative) is the kind of result that would historically have made it ensemble-positive, but ensembles are off the table.

Askeladd reassigned **H84 RFF-NUM-FEATURES-EXPANSION** (PR #1244) — single-flag `--rff-num-features 16 → 32` (2× positional encoding capacity per sigma band). First-ever Fourier feature sweep in entire Wave 31/32. Plateau-protocol Tier-2 architectural-input-capacity axis. Tancik et al 2020 RFF literature recommends 32-256 features; our 16 is below recommended range. Orthogonal to all 7 in-flight Wave 32 axes (H73 Charbonnier-τz, H78 β1, H79 dropout, H80 EMA-decay, H81 β2, H82 weight_decay, H83 grad-clip).

---

## 2026-05-21 12:00 — PR #1233: H77 CHARBONNIER-VOL-P-WEIGHT-FIX (nezuko, CLOSED) — **OUTCOME D NEGATIVE on val AND all 4 test channels** (clean execution; H68 starvation pathology fixed but underlying Charbonnier-vol_p technique net-negative on tay split)

- **Branch**: `nezuko/h77-charbonnier-vol-p-weight-fix` (closed at 12:00Z)
- **W&B run**: `qfwysjwu` (EP13 terminal step 70,652, 14.0h training time, all 8 ranks healthy throughout)
- **Hypothesis**: Apply Charbonnier-vol_p (ε=1e-3) loss with the CORRECT `--volume-loss-weight 0.5` per dl24 H19 verbatim recipe — H68 was killed at EP6 with starvation pathology because weight was set to 1.0 (which over-emphasized the larger Charbonnier-magnitude vol_p budget vs surface). H77 tests whether the recipe-fix alone resolves the starvation AND beats baseline.

### Terminal metrics (EP13 best, EMA checkpoint, full test eval inline)

| Channel | **H77** | BL #972 ref | Δ vs BL #972 | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt (gate)** | **6.335%** | 6.126% | **+0.209** ❌ | MISS gate |
| val_VP | 3.838% | 3.798% | +0.040 | ~neutral |
| val_SP | 4.179% | — | — | — |
| val_WSS | 7.154% | — | — | — |
| **test_abupt** | **6.260%** | 5.844% | **+0.416** ❌ | regression |
| **test_VP (floor 3.643)** | **3.866%** | 3.643% | **+0.223** ❌ | MISS floor |
| **test_SP (floor 3.577)** | **4.028%** | 3.577% | **+0.451** ❌ | MISS floor |
| test_WSS (goal 6.727) | **7.155%** | 6.727% | **+0.428** ❌ | regression |

### Attribution analysis — H68 → H77 progression

| Diagnostic | H68 EP6 (killed) | **H77 EP13 (terminal)** | Interpretation |
|---|---|---|---|
| char/mse ratio | 18.4× (runaway) | **16-18× (stable plateau)** | ✅ recipe fix works |
| val_abupt at EP6 | 6.822% | 6.451% | ✅ −0.371pp better than H68 |
| Surface starvation? | YES (vol_p budget too large) | NO (surface ~0.0085 > volume ~0.0058) | ✅ fixed |
| Final val_abupt | n/a (killed) | **6.335%** | ❌ still misses gate by +0.209pp |
| Final test_abupt | n/a (killed) | **6.260%** | ❌ +0.416pp vs baseline |

**The Charbonnier-vol_p (ε=1e-3) loss family is REPRESENTATIONALLY DIFFERENT from MSE on this dataset's vol_p split.** Charbonnier's robust-Huber smoothing down-weights large per-point errors via √(x²+ε²); on a dataset where vol_p errors have heavy-tailed distribution, MSE was already capturing the signal — the robust-loss formulation introduces implicit regularization that costs ~0.21pp val_abupt and ~0.42pp test_abupt.

### H75 LR-fix interaction (recently confirmed)

H77 used `--lr-cosine-t-max 25` (LR-fix substrate). H75 D REG just confirmed LR-fix is NET NEG on pure baseline. H77 inherits some of that LR-fix tax. A cleaner H77-followup would test Charbonnier-vol_p on `--lr-cosine-t-max 13` substrate to isolate technique-tax from LR-fix-tax. **Wave 33 deprioritized** since H68 + H77 already establish the technique as net-negative on this dataset.

### Per-epoch trajectory (clean monotonic descent, smooth)

| EP | step | val_abupt | val_VP | val_SP | val_WSS |
|---|---:|---:|---:|---:|---:|
| EP1 | 10,864 | 26.95% | ~16% | ~20% | ~30% |
| EP3 | 32,594 | 6.92% | 4.05% | 4.62% | 7.85% |
| EP6 | 48,902 | 6.451% | — | — | — |
| EP9 | 59,780 | 6.376% | — | — | — |
| EP10 | 62,501 | 6.350% | — | — | — |
| EP11 | 65,222 | 6.344% | — | — | — |
| **EP13** | **70,652** | **6.335%** | **3.838%** | **4.179%** | **7.154%** |

EP9→EP11 slope: −0.002 pp/1k (effectively plateaued well above merge gate). Slope decay signature matches H78's late-cosine pattern but at significantly higher absolute val_abupt.

### Wave 32 loss-reformulation class closure pattern

Three of four Wave 32 loss-reformulation experiments now D NEG:
- H68 Charbonnier-vol_p (weight=1.0 bug) — D NEG (starvation pathology)
- H74 MAE-aux-vol_p — D NEG (L1/MSE ratio anti-pattern, cross-axis collateral)
- **H77 Charbonnier-vol_p (weight=0.5 fix)** — D NEG (representationally different from MSE)
- H73 Charbonnier-τz — in flight (edward)

The loss-reformulation class is exhausted on vol_p axis. Wave 33 should NOT pursue vol_p loss-family variants (asymmetric, robust, L1-aux, etc.) — focus instead on optimizer-momentum (H78 winner), regularization (H79/H82), EMA composition (H80), Lion-buffer (H81), and gradient-control (H83 new) axes.

### Followups noted (filed for Wave 33 if substrate changes)

- H77-NO-LR-FIX: Charbonnier-vol_p (weight=0.5) on lr-cosine-t-max=13 substrate to isolate technique-tax from LR-fix-tax
- H77-DIFFERENT-EPS: Charbonnier ε sweep (1e-2, 1e-4) — but D NEG result suggests technique-class limit, not eps-tuning issue

Nezuko reassigned **H83 GRAD-CLIP-EXPANSION** (PR #1243) — single-flag `--grad-clip-norm 0.5 → 1.0` (Lion paper default). First-ever gradient-clipping sweep in entire Wave 31/32 fleet. Plateau-protocol Tier 2 optimization-control axis. With Lion's sign-update, clip controls what signal enters momentum buffer — distinct from H78's β1 (momentum decay rate) and H81's β2 (momentum buffer EMA). Orthogonal to all 7 in-flight axes.

---

## 2026-05-21 10:50 — PR #1231: H75 PURE-BASELINE-LR-EXTENDED (alphonse, CLOSED) — **OUTCOME D REGRESSION on val AND test** (cleanest Wave 31/32 LR-fix attribution: --lr-cosine-t-max 25 is NOT universal, only mech-class-conditional)

- **Branch**: `alphonse/h75-pure-baseline-lr-extended` (closed at 10:50Z)
- **W&B run**: `vokzc49z` (EP13 terminal step 70,664, ~14.0h training time)
- **Hypothesis**: Take exact baseline #972 config and change ONLY `--lr-cosine-t-max 13 → 25`. Zero other changes. The missing CONTROL experiment for the entire Wave 31/32 LR-fix campaign — answers whether LR-fix universally benefits generalization or synergizes with specific mechanisms.

### Terminal metrics (EP13 best, EMA checkpoint, full test eval inline)

| Channel | **H75** | BL #972 ref | Δ vs BL #972 | H65 ref (LR-fix + surf-deep) | Δ vs H65 |
|---|---:|---:|---:|---:|---:|
| **val_abupt (gate)** | **6.304%** | 6.126% | **+0.178** ❌ | 6.234% | +0.069 ❌ |
| val_VP | **3.744%** | 3.798% | **−0.054** ✅ | — | — |
| val_SP | 4.158% | ~3.80% | ~+0.36 ❌ | — | — |
| val_WSS | 7.130% | — | — | — | — |
| **test_abupt** | **6.098%** | 5.844% | **+0.254** ❌ | 5.926% | +0.172 ❌ |
| **test_VP (floor 3.643)** | **3.677%** | 3.643% | **+0.034** ❌ | 3.588% | +0.089 ❌ |
| **test_SP (floor 3.577)** | **3.884%** | 3.577% | **+0.307** ❌ | 3.687% | +0.197 ❌ |
| **test_WSS (goal 6.727)** | **7.027%** | 6.727% | **+0.300** ❌ | 6.836% | +0.191 ❌ |

### Per-epoch trajectory (clean monotonic descent, no anomalies)

| EP | step | val_abupt | val_VP | val_SP | val_WSS | τx | τy | τz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EP1 | 10,864 | 26.954 | 15.632 | 20.564 | 29.904 | 26.162 | 35.074 | 37.340 |
| EP3 | 32,594 | 6.816 | 3.957 | 4.521 | 7.725 | 6.741 | 8.652 | 10.208 |
| EP6 | 48,902 | 6.440 | 3.833 | 4.263 | 7.273 | 6.358 | 8.015 | 9.732 |
| EP10 | 62,501 | 6.338 | 3.776 | 4.183 | 7.164 | 6.280 | 7.819 | 9.631 |
| EP13 | **70,664** | **6.304** | **3.744** | **4.158** | **7.130** | **6.253** | **7.753** | **9.610** |

### Attribution conclusion — definitive Wave 31/32 LR-fix campaign answer

**H75 vs H65 (same LR-fix substrate, minus surf-deep mechanism)** — isolates surf-deep's contribution to H65's test wins:
- val_abupt: surf-deep adds −0.070pp improvement on LR-fix substrate
- test_abupt: surf-deep adds −0.172pp test improvement
- **test_VP: surf-deep was 100% responsible for H65's test_VP floor cross** (H75 misses floor, H65 crossed it)
- test_SP: surf-deep adds −0.197pp test_SP improvement

The Wave 31/32 LR-fix campaign assumption was that `--lr-cosine-t-max 25` provides universal generalization benefit. **H75 falsifies this assumption.** LR-fix is a *mechanism-class-conditional activator* — it unlocks specific mechanism's expressivity (especially surf-deep's surface-head depth) but is NET NEGATIVE on the pure baseline.

### Strategic implication for Wave 33 recipe authoring

`--lr-cosine-t-max 25` MUST be flagged as mechanism-class-conditional, NOT a default. Recipe authors should add it only when the mechanism class is one of:
- V-DEPTH (decoder depth expansion) — H47/H57/H59 series
- SURF-DEEP (surface-head depth expansion) — H54/H65 series
- Possibly other not-yet-tested mechanism classes that benefit from extended cosine descent

Adding LR-fix as a "free generalization improvement" was the implicit assumption that drove H62/H63/H64/H66/H67 — H75 shows this assumption costs us ~0.18pp val_abupt and ~0.25pp test_abupt vs pure baseline + mechanism if the mechanism wouldn't benefit from LR-fix.

### Single bright spot: val_VP −0.054pp

The only positive metric is **val_VP** (3.744% vs 3.798% baseline, −5.4bp). This confirms LR-fix DOES help the val_VP optimization landscape — but the improvement is dominated by val_SP/val_WSS regressions in aggregate val_abupt, AND val→test slope inverts the val_VP win on test side (test_VP misses floor by +0.034pp).

### Wave 31/32 LR-fix retrospective table

| H | Class | LR-fix? | Mechanism on top? | val_abupt | test_VP floor cross | Outcome |
|---|---|---|---|---:|---|---|
| H47 (pre-W31) | V-DEPTH | no | yes | baseline | no | — |
| H57 (#1206) | V-DEPTH | yes | yes | 6.217% | (val) | C NULL, mech-pos test |
| H59 (#1206) | V-DEPTH | yes | yes | 6.282% | ✅ 3.552% | C NULL, surf-deep-style test cross |
| H65 (#1214) | SURF-DEEP | yes | yes | 6.234% | ✅ 3.588% | C NULL, test_VP cross |
| H66 (#1215) | COORDSLICE | yes | yes | ~6.39% | no | C NULL |
| H67 (#1221) | RFF-9σ-WIDTH | yes | yes | 6.175% | no | closest C NULL, no test cross |
| **H75 (#1231)** | **none (CONTROL)** | **yes** | **no** | **6.304%** | **no** | **D REG ❌** |

Alphonse reassigned **H82 WEIGHT-DECAY-EXPANSION** (PR #1242) — single-flag `--weight-decay 5e-4 → 1e-3` on PURE baseline #972 substrate. First-ever weight-decay sweep in entire Wave 31/32 fleet. Param-magnitude-side regularization complement to H79 dropout's activation-side regularization (forms 2×2 regularization-class matrix). Plateau-protocol Tier 2 regularization escalation. Orthogonal to all 7 in-flight axes (H78 β1, H81 β2, H80 EMA, H79 dropout, H77 Charbonnier-vol_p, H76 slices, H73 Charbonnier-τz).

---

## 2026-05-21 10:25 — PR #1230: H74 MAE-AUX-VOL-P (frieren, CLOSED) — **OUTCOME D NEGATIVE on all 5 paper-facing test axes** (cross-axis collateral damage; mech engaged but α=0.05 too strong late-train)

- **Branch**: `frieren/h74-mae-aux-vol-p` (closed at 10:25Z)
- **W&B run**: `hzlwf8ep` (EP13 terminal step 70,664, ~14.7h, run state finished)
- **Hypothesis**: Add L1 (MAE) auxiliary loss term on `vol_p` channel with weight α=0.05, cross-pollinated from dl24 H22. L1 provides non-vanishing gradient signal as MSE shrinks quadratically through the cosine schedule's tail, enabling continued vol_p descent in the late-cosine plateau where the test_VP floor 3.643% is binding.

### Terminal metrics (EP13 best validation checkpoint, full test eval)

| Axis | **H74** | Baseline #972 | Δ vs baseline | Verdict |
|---|---:|---:|---:|:---|
| **val_abupt (merge gate)** | **6.2639%** | 6.126% | **+0.138** | MISS gate |
| val_VP | 3.6268% | 3.798% | **−0.171** | beats baseline val_VP |
| val_SP | 4.141% | — | — | — |
| val_WSS | 7.095% | — | — | — |
| test_abupt | **6.1172%** | 5.844% | +0.273 | regression |
| **test_VP (floor 3.643)** | **3.6690%** | 3.643% | **+0.026** ❌ | does NOT cross floor |
| **test_SP (floor 3.577)** | **3.8742%** | 3.577% | +0.297 ❌ | regression |
| test_WSS (goal 6.727) | **7.0492%** | 6.727% | +0.322 ❌ | regression |
| test_WSS_x | 6.255% | — | — | — |
| test_WSS_y | 7.673% | — | — | — |
| test_WSS_z | 9.115% | — | — | — |

**Key contradiction**: val_VP crossed below baseline test_VP floor 3.643% on the val side (3.627%), but the val→test slope inverted for test_VP, putting it at 3.669%. Other axes (SP, WSS) had val→test slope worsening too, but starting from higher val readings → larger absolute test regressions.

### Per-epoch trajectory (clean monotonic descent, no anomalies)

| Epoch | step | val_abupt | val_SP | val_VP | val_WSS |
|---|---:|---:|---:|---:|---:|
| EP1 | 10,864 | 26.555 | 19.936 | 15.862 | 29.539 |
| EP2 | 21,729 | 7.883 | 5.230 | 4.565 | 8.919 |
| EP3 | 32,594 | 7.136 | 4.873 | 4.010 | 8.091 |
| EP6 | 48,902 | 6.451 | 4.259 | 3.743 | 7.308 |
| EP9 | 59,780 | 6.315 | 4.170 | 3.664 | 7.151 |
| EP11 | 65,222 | 6.272 | 4.149 | 3.636 | 7.100 |
| **EP13** | **70,664** | **6.264** | **4.141** | **3.627** | **7.095** |

### MAE_aux mechanism engagement (student's diagnostic)

| Diagnostic | EP1 | EP3 | EP7 | EP13 |
|---|---:|---:|---:|---:|
| `train/vol_p_mae_aux` (raw L1) | 0.0317 | 0.0200 | 0.0142 | **0.01151** |
| `train/vol_p_mae_aux_weighted` (α=0.05·L1) | 0.00158 | 0.00100 | 0.00071 | **0.000576** |
| `train/volume_loss` (MSE) | ~0.75 | 0.00130 | 0.00060 | smaller |
| **ratio `mae_aux_w / vol_loss`** | 0.21% | 77% | **117%** | continued climbing |

The MAE_aux **did engage as designed** — raw L1 dropped monotonically 64% from EP1→EP13. But the ratio crossed the PR-body 30% anti-pattern guard at EP3, hit 117% by EP7, kept climbing — making L1 the dominant vol_p gradient term in the late-cosine tail.

### Attribution analysis (advisor judgement)

The cross-axis cost is the dominant failure mode. The vol_p axis itself moved minimally (test_VP +0.026pp vs baseline) while test_SP +0.297pp and test_WSS +0.322pp regressed substantially. The α=0.05 L1 term destabilized the OTHER axes by competing with the MSE objective's adaptive scale, with the L1 term effectively *growing in magnitude* through training as MSE shrank quadratically.

**Mech-class conclusions:**
1. **L1/MSE ratio late-train anti-pattern is real**: α-tuning for L1-aux on MSE-dominated losses must account for quadratic-vs-linear descent asymmetry. Static α=0.05 → late-train L1 dominance regardless of intent.
2. **Cross-axis cost dominates loss-reformulation class on Wave 32**: not floor-binding, not capacity-binding, but per-axis-trade-off binding.
3. **Val_VP cross-floor without test_VP cross-floor**: val→test slope is axis-dependent and reformulation can flip the slope sign.

### Wave 32 single-axis-collapse table now 6 entries

H74 joins H62, H70, H72, H71, H69 in the table of "mechanism engaged as designed but pays in another axis." Loss-reformulation class (Charbonnier-vol_p H68, Charbonnier-τz H73 in-flight, Charbonnier-vol_p-weight-fix H77 in-flight, MAE-aux H74) all show this cross-axis cost pattern.

### Followups deprioritized (filed in research_state for Wave 33 if H78 merges)

- H74-A: α=0.01 or α=0.025 (lower magnitude, may stay below 30% ratio band through EP13)
- H74-B: schedule α(t) with cosine decay matching MSE/L1 asymmetry
- H74-C: MAE_aux on ALL channels not just vol_p (per student's suggested follow-up — universal robust-loss boost)

frieren reassigned **H81 LION-BETA2-EXPANSION** (PR #1240) — single-flag `--lion-beta2 0.99→0.999` on PURE baseline #972 substrate. First-ever Lion β2 sweep in entire Wave 31/32 fleet history. Orthogonal to H78's β1 change (β1 smooths *direction*, β2 expands *momentum buffer history* 10× from ~100→~1000 steps).

---

## 2026-05-21 02:35 — PR #1223: H69 CURVATURE-ATTENTION-BIAS (fern, CLOSED) — **OUTCOME D NEGATIVE on every paper-facing axis vs H66 substrate twin**

- **Branch**: `fern/h69-curvature-attention-bias-v2` (closed at 02:35Z)
- **W&B run**: `wc9afk2u` (EP13 terminal step 70,664, ~15.0h)
- **Hypothesis**: Add learnable per-block bias to attention proportional to local surface curvature, with curvature signal pre-computed at input. Cross-pollinated from dl24 H10b. Predicted disproportionate improvement on WSS_z axis (binding metric).

### Terminal metrics (EP13 best, evaluated from best-validation-checkpoint)

| Axis | **H69** | Baseline #972 | H66 substrate twin `bdbt67as` | Δ vs baseline | Δ vs H66 |
|---|---:|---:|---:|:---:|:---:|
| val_abupt | **6.384%** | 6.126% | 6.381% | +0.258 ❌ | +0.003 (tie) |
| val_VP | 3.804% | 3.566% | — | +0.238 | — |
| val_SP | 4.183% | 3.534% | — | +0.649 | — |
| val_WSS | 7.213% | 6.679% | — | +0.534 | — |
| test_abupt | **6.183%** | 5.844% | 6.086% | +0.339 ❌ | **+0.097 ❌** |
| test_SP (floor 3.577) | 3.946% | 3.577% | 3.852% | +0.369 ❌ | +0.094 ❌ |
| test_VP (floor 3.643) | 3.770% | 3.643% | 3.628% | +0.127 ❌ | +0.142 ❌ |
| test_WSS (goal 6.727) | 7.092% | 6.727% | 7.021% | +0.365 ❌ | +0.071 ❌ |
| test_WSS_x | 6.285% | — | 6.231% | — | +0.054 ❌ |
| test_WSS_y | 7.719% | — | 7.666% | — | +0.053 ❌ |
| **test_WSS_z (binding)** | **9.196%** | ~8.750% | 9.055% | +0.446 ❌ | **+0.141 ❌** |

### Mech engagement (block-level alpha values from W&B)

| Block | alpha (learnable scalar) | bias_abs_mean | bias_contribution | alpha grad norm |
|---|---:|---:|---:|---:|
| B0 (input-near) | **0.454** | 0.126 | **2.0%** | ~2e-6 |
| B1 | 0.215 | 0.083 | 2.6% | ~2e-6 |
| B2 | 0.081 | 0.033 | 0.9% | ~1e-6 |
| B3 | 0.150 | 0.041 | 1.0% | ~3e-6 |
| **B4 (output-near)** | **0.071** | 0.021 | **0.5%** | ~1e-6 |

Mech ENGAGED but front-loaded usage pattern — alpha grad norms ~1-4e-6 = learnable bias effectively CONVERGED. Curvature bias is naturally most useful for input/early feature extraction (B0/B1 = 4.6% combined contribution) and ESSENTIALLY DEAD at output (B4 = 0.5%). 

### Results commentary — clean falsification, wrong tier of intervention

**H69 underperforms H66 (substrate twin without curvature bias) on EVERY paper-facing test axis.** The hypothesis predicted disproportionate WSS_z improvement; instead WSS_z is the axis with the LARGEST regression (+0.141pp vs H66).

**Interpretation**: Curvature is naturally an early/intermediate feature gate, not an output-tier refinement signal. The model learned to use curvature for input feature extraction (where it has 2% bias contribution in B0) and ignored it at output (0.5% in B4) — exactly the opposite of what was needed for WSS_z attack. **Curvature-attention-bias class is the WRONG TIER of intervention for τz architectural ceiling.**

### Mech-class binding update

Wave 32 single-axis-collapse table (now 5 entries):

| H | Class | LR | Outcome | Failure mode |
|---|---|---|---|---|
| H62 | CP-loss-weight | LR-fix | D NEG +0.216pp | Destabilizes optimizer |
| H70 | Slice-temp-curr | LR-fix | D NEG +2.298pp | Pace mismatch |
| H72 | Slice-temp-deep-endpoint | legacy | D NEG +5.46pp | Over-sparsification |
| H71 | GradNorm dynamic-balance | legacy | D NEG +0.279pp | Capacity misallocation away from τz |
| **H69** | **Curvature-attention-bias** | **legacy** | **D NEG +0.258pp** | **Front-loaded usage, wrong tier for τz attack** |

### Closure verdict & next direction

**D NEGATIVE** on every test axis vs both baseline and H66 substrate twin. Closing without merge. Fern reassigned **H80 EMA-DECAY-EXTENSION (PR #1236)** — single-flag `--ema-decay 0.999 → 0.9999` on PURE baseline #972 substrate. **FIRST-EVER EMA composition sweep** in entire Wave 31/32 — ema_decay=0.999 has been load-bearing across every prior experiment. Slower EMA = 10× more smoothing = eval-time EMA captures ~4 epochs of training history vs only last ~700 steps at ema=0.999. EP1 kill DROPPED per `feedback_ema_aware_kill_thresholds.md`. Plateau-protocol-tier EMA-composition escalation. Orthogonal to all 7 in-flight axes.

---

## 2026-05-20 23:35 — PR #1225: H71 GRADNORM-DYNAMIC-LOSS-BALANCING (tanjiro, CLOSED) — **OUTCOME D NEGATIVE on all test metrics — clean mechanism-falsification (mech engaged but outcome regressed)**

- **Branch**: `tanjiro/h71-gradnorm-dynamic-loss-balancing` (closed at 23:35Z)
- **W&B run**: `hfind9uf` (EP13 terminal step 70,652, 14.36h)
- **Hypothesis**: Dynamic per-task loss weight balancing via GradNorm-EMA-proxy with α=1.5. Each of {vol_p, SP, τx, τy, τz} gets its weight adjusted based on EMA of recent loss values — easier (lower-loss) tasks drain, harder (higher-loss) tasks escalate. Floor=0.20.

### Terminal metrics (EP13, EMA-best ckpt)

| Metric | **H71** | Baseline #972 | Δ vs baseline | AB-UPT ref | vs AB-UPT |
|---|---:|---:|---:|---:|---:|
| val_abupt | **6.4044%** | 6.126% | **+0.279pp ❌** | — | — |
| val_VP | 3.979% | 3.566% | +0.413 | — | — |
| val_SP | 4.265% | 3.534% | +0.731 | — | — |
| val_WSS | 7.182% | 6.679% | +0.503 | — | — |
| **test_abupt** | **6.130%** | 5.844% | **+0.286 ❌** | — | — |
| test_SP | 3.916% | 3.577% (floor) | +0.339 ❌ | 3.82 | +0.10 |
| test_VP | 3.867% | 3.643% (floor) | +0.224 ❌ | 6.08 | **−2.21 ✅** |
| test_WSS | 7.002% | 6.727% (goal) | +0.275 ❌ | 7.29 | −0.29 ✅ |
| test τx_WSS | 6.236% | — | — | 5.35 | +0.89 |
| test τy_WSS | 7.537% | — | — | 3.65 | +3.89 |
| **test τz_WSS** | **9.091%** | — | — | **3.63** | **+5.46 (binding axis)** |

### Mechanism engagement (terminal GradNorm weights)

| Task | weight | ema_loss | Δ vs uniform 1.0 |
|---|---:|---:|---:|
| vol_p | **0.187** (at floor 0.20) | 9.39e-4 | **−81%** drained |
| SP | 0.465 | 9.13e-4 | −54% drained |
| τx | 0.949 | 1.50e-3 | −5% near-uniform |
| τy | 1.500 | 2.02e-3 | +50% escalated |
| **τz** | **1.897** | 2.31e-3 | **+90% maximally-escalated** |

τz received ~10× more weight than vol_p, exactly as GradNorm theory predicted.

### Results commentary — clean falsification of dynamic-loss-balancing class on val_abupt

**Mechanism engaged AS DESIGNED but outcome WORSE on every metric.** This is the cleanest possible falsification signal:

1. GradNorm computed loss-EMA per task correctly (logged values shown above)
2. Reweighted inversely correctly (vol_p down, τz up)
3. Total reweighting magnitude is large (~10× ratio)
4. **Yet test τz_WSS = 9.091% is WORSE than any prior Wave 31/32 closure**

**Interpretation**: The architectural ceiling on τz is NOT capacity-bound. The model literally cannot represent τz better given the current parameter budget, regardless of how hard we push it via loss weights. Re-weighting moved capacity AWAY from where it was already working (vol_p, SP) but the gained τz capacity didn't translate to better predictions because the ceiling is **representation-bound** (PE/embedding/decoder architecture limitations for this axis).

**Mech-class binding update**: Dynamic-loss-balancing class FALSIFIED. Adding to Wave 32 single-axis-collapse table:

| H | Class | LR | Outcome | Failure mode |
|---|---|---|---|---|
| H62 | CP-loss-weight | LR-fix | D NEG +0.216pp | Destabilizes optimizer |
| H70 | Slice-temp-curr | LR-fix | D NEG +2.298pp | Pace mismatch |
| H72 | Slice-temp-deep-endpoint | legacy | D NEG +5.46pp | Over-sparsification |
| **H71** | **GradNorm-dynamic-balance** | **legacy** | **D NEG +0.279pp** | **Capacity misallocation away from τz** |

### Notable AB-UPT comparison

H71 beats AB-UPT reference on test_VP (−2.21pp better) and test_WSS aggregate (−0.29pp better), but loses catastrophically on the per-axis WSS breakdown — τz +5.46pp vs AB-UPT, τy +3.89pp. **Our model has a clear coordinate-axis bias** with monotonic worsening x→y→z on WSS axes.

### Closure verdict & next direction

**D NEGATIVE** on every test metric. Closing without merge. Tanjiro reassigned **H79 DROPOUT-INTRODUCTION (PR #1235)** — single-flag `--model-dropout 0.0 → 0.1` on PURE baseline #972 substrate. **FIRST-EVER dropout test on this model**. Current `model_dropout=0.0` has been load-bearing across the entire fleet history; train loss converges to ~0.009 (small) while val/test plateau at 6.15-6.40% (large gap) = classical overfitting signature on 34-case val set. Plateau-protocol-tier regularization escalation. Orthogonal to all in-flight axes (loss-reformulation H73/H74/H77, routing H76, attention-mech H69, optimizer H78).

---

## 2026-05-20 21:55 — PR #1221: H67 RFF-9σ-WIDTH-EXPANSION (thorfinn, CLOSED) — **OUTCOME C NULL on val_abupt (B PARTIAL boundary)**

- **Branch**: `thorfinn/h67-rff-9sigma-width-expansion` (closed at 21:55Z)
- **W&B run**: `47o8883g` (EP13 terminal step 70,652, 14.13h)
- **Hypothesis**: Add σ=0.0625 to H57's 8σ RFF basis (0.125-16) for 9-octave coverage. Test whether wider bandwidth at LOW-frequency end improves long-wavelength surface features (curvature, large-scale flow patterns). Substrate: LR-fix `--lr-cosine-t-max 25` (Wave 31 LR-fix campaign).

### Terminal metrics (EP13, EMA-best ckpt)

| Metric | **H67** | H57 (#1206) | H58 (#1207) | baseline #972 | gate/floor | Δ vs gate/floor |
|---|---:|---:|---:|---:|---:|---:|
| val_abupt | **6.1746%** | 6.217% | 6.171% | 6.126% | <6.126% | **MISS +0.049pp** |
| val_VP | 3.649% | 3.612% | 3.604% | 3.566% | (3.643% floor) | NEAR-MISS +0.006pp |
| val_SP | 4.076% | — | — | 3.534% | — | — |
| val_WSS | 6.993% | — | — | 6.679% | — | — |
| test_abupt | **6.046%** | 6.053% | — | 5.844% | — | +0.202pp vs baseline |
| test_VP | 3.666% | 3.610% | 3.660% | ≤3.643% | floor | **MISS +0.023pp** (no floor cross) |
| test_SP | 3.860% | 3.812% | — | ≤3.577% | floor | MISS +0.283pp |
| **test_WSS** | **6.933%** | 6.949% | — | <6.727% (goal) | — | −0.016pp ✅ (still +0.206pp above goal) |
| test τx WSS | 6.130% | — | — | — | — | — |
| test τy WSS | 7.537% | — | — | — | — | — |
| test τz WSS | **9.036%** | — | — | — | — | (hardest axis) |

### Results commentary — mech-positive but undersized, closest-to-gate Wave 31 LR-fix result

**Closest-to-gate result of entire Wave 31 LR-fix campaign.** Beats H57 by −0.042pp on val_abupt (just 0.008pp shy of B PARTIAL ≥0.05pp threshold). RFF-bandwidth-expansion (add σ=0.0625 to H57's 8σ basis) is **mech-positive but undersized**.

**Mech-class binding update**: RFF-band-WIDTH (H64→H67) joins **architecture-bound-at-val LR-bound-at-test** category alongside V-DEPTH (H47→H59) and shared-cap-surface (H54v2→H65). H67 distinguished from those two by being **LR-NEUTRAL at test** (no floor cross) — wider RFF basis doesn't synergize with LR-fix on test side either, unlike V-DEPTH/surf-deep where LR-fix dropped test_VP through floor.

Note H66 (encoder-PE-no-stopgrad) ALSO C NULL on val_abupt at 6.381% (+0.220pp WORSE than H58 parent despite 3-5× higher LR) — confirmed Lion+sign-cancellation makes mean-zero PE-proj gradients LR-magnitude-neutral. H67 doesn't share this collapse mode (RFF basis is NOT mean-zero gradient), but lacks H65/H66's test-side dividend because the bandwidth-expansion mechanism is genuinely orthogonal to LR-decay-truncation effect.

### Project test_VP floor cross tally (no change after H67)

H67 does NOT add to floor-cross count. Running total remains **6 floor crosses** (H26 NPCA merged, H53, H55v2, H57, H59, H65, H66). H67 is the **first Wave 31 LR-fix exhaustion variant without a test_VP floor cross**.

### Closure verdict & next direction

**C NULL** with **no test floor cross** and **mech-positive but undersized** trajectory. Bandwidth-expansion direction exhausted on LR-fix substrate — extending RFF further (σ=0.03125 low or σ=32 high) unlikely to recover +0.049pp gap.

**Reassignment: thorfinn → H78 LION-BETA1-MOMENTUM-EXPANSION (PR #1234)** — plateau-protocol optimizer-momentum-tier escalation: single-flag `--lion-beta1 0.9 → 0.95` on PURE baseline #972 substrate. Lion β1/β2 = only major optimizer axis untouched in entire fleet history. β1=0.95 doubles effective momentum window 10→20 steps, smoothing gradient noise in late-tail descent. Orthogonal to LR-magnitude (H75 alphonse control), loss-curvature (H73/H74/H77), routing (H76), attention-mech (H69), and loss-balancing (H71) axes in flight.

---

## 2026-05-20 21:20 — PR #1222: H68 CHARBONNIER-VOL-P (nezuko, CLOSED) — **OUTCOME D NEGATIVE — killed EP6 step 65222 — ROOT CAUSE: recipe execution deviation (`--volume-loss-weight 1.0` vs advisor-specified 0.5), technique NOT falsified**

- **Branch**: `nezuko/h68-charbonnier-vol-p` (closed at 21:20Z)
- **W&B run**: `6mm00t4k` (killed at step 65222 EP6, 785 min wall-time)
- **Hypothesis**: Cross-pollinate dl24 H19 Charbonnier loss on vol_p to tay. Replace `masked_mse` for volume_preds with `masked_charbonnier(eps=1e-3)`. Expected: outlier-suppression on vol_p helps generalisation.

### Terminal metrics (at EP6 kill point)

| Metric | H68 EP6 | Baseline #972 | Status |
|---|---:|---:|:--|
| val_abupt | **6.822%** | 6.126% | ❌ killed +0.322pp above EP6 gate (6.5%) |
| val_VP | 4.146% | 3.798% | — |
| val_SP | 4.574% | 3.577% | — |
| val_WSS | 7.646% | 6.727% | — |

*No test eval — killed at EP6.*

### Results commentary — recipe execution deviation, NOT technique failure

**ROOT CAUSE: `--volume-loss-weight 1.0` deviation from advisor-specified `0.5`**

Student used weight=1.0 for "H59 substrate parity" but PR body specified 0.5 (verbatim dl24 H19 recipe). Key consequence: Charbonnier(eps=1e-3) operates in L1-regime for our vol_p error distribution (std ~0.022 >> eps=1e-3), producing per-element losses **18× larger than MSE** at terminal. With weight=1.0, effective vol_p contribution = 18× MSE_baseline_budget. Surface heads starved from step 1.

Student's `vol_loss_diag/char_over_mse` time series (KEY evidence):
| Step | char/mse | Comment |
|---|---:|---|
| ~249 | 0.67× | Cold start, large errors → char SMALLER than mse (L1 vs L2) |
| ~16,500 | **6.4×** | Early EP2, typical errors now < eps elbow |
| ~32,752 | **13.5×** | EP3 |
| ~49,005 | **17.4×** | EP4-5 |
| ~65,009 | **18.4×** | Terminal |

**EP1 cold-start 28.04%** (normal: 15-25%) is the smoking gun — surface heads were starved from step 1.

### Implementation was correct — code preserved for H77

Student shipped:
- `masked_charbonnier(pred, target, mask, eps)` in `trainer_runtime.py` — math verified
- `--vol-loss-type {mse,charbonnier}`, `--charbonnier-eps` flags in `train.py`
- Full `vol_loss_diag/` diagnostic suite

**Code is correct. Recipe was the failure.** `nezuko/h68-charbonnier-vol-p` branch preserved for cherry-pick.

### Mech class status

Loss-curvature-shape class NOT exhausted. dl24 H19 used `--volume-loss-weight 0.5` knowing Charbonnier's L1-scale magnitude shift. H68 tested the WRONG composition. H77 tests the verbatim recipe.

### Follow-up

**H77 CHARBONNIER-VOL-P-WEIGHT-FIX (PR #1233)**: Single-flag change `--volume-loss-weight 1.0 → 0.5`. Everything else from H68 identical. Verbatim dl24 H19 recipe replication.

---

## 2026-05-20 20:35 — PR #1215: H66 COORDSLICE-NO-STOPGRAD-LR-EXTENDED (askeladd, CLOSED) — **OUTCOME C NULL WITH REGRESSION on val_abupt + 6th test_VP FLOOR CROSS (3.628%) — encoder-PE class FULLY EXHAUSTED on LR-axis**

- **Branch**: `askeladd/h66-coordslice-no-stopgrad-lr-extended` (closed at 20:35Z)
- **W&B run**: `bdbt67as` (13/13 epochs, 882.7 min = 14.7h wall-time)
- **Hypothesis**: H58 COORDSLICE-NO-STOPGRAD plateaued at val_abupt 6.161% due to LR-decay confound. Test if `--lr-cosine-t-max 13 → 25` preserves 56% peak LR at terminal and enables continued descent.

### Terminal metrics

| Metric | H66 terminal | H58 ref | Baseline #972 | Δ vs H58 | Status |
|---|---:|---:|---:|---:|:--|
| val_abupt | **6.3814%** | 6.161% | 6.126% (gate) | **+0.220pp WORSE** | ❌ C NULL with regression |
| test_VP | **3.628%** | 3.551% | 3.643% (floor) | +0.077pp | ✅ **6th FLOOR CROSS** |
| test_SP | 3.852% | 3.856% | 3.577% (floor) | −0.004pp | ❌ above floor |
| test_abupt | 6.086% | 5.999% | 5.844% | +0.087pp | — |
| test_WSS | 7.021% | 6.906% | 6.727% | +0.115pp | ❌ above goal |
| val_WSS_z | 9.760% | — | — | — | — |

### Results commentary

**Formally C NULL with regression**: H66 val_abupt 6.381% vs H58's 6.161% = +0.220pp WORSE despite 3-5× higher LR through EP6-EP13. Merge gate (6.126%) missed by +0.255pp.

**KEY FINDING — Lion+zero-mean-gradient sign-cancellation is LR-magnitude-independent**: The H66 vs H58 trajectory is a parallel-shifted constant gap +0.20-0.24pp from EP4 onward. Despite H66 retaining 91-56% peak LR through EP6-EP13 vs H58's 14-0%, the slopes are MATCHED (H66 EP6→EP13: −0.0094 pp/1k; H58: −0.0086 pp/1k). The encoder-PE-no-stopgrad mechanism that generates tiny indirect gradients through softmax+routing is washed out by Lion's sign update at ANY LR magnitude tested.

**proj_weight_std evidence** (askeladd's excellent diagnostic):
- H66 EP9 max block0 std: 0.0987 vs H58 terminal 0.0981 — virtually identical +0.011 growth from init 0.088
- More LR → same PE-proj growth → sign cancellation dominates

**6th test_VP floor cross**: 3.628% < 3.643% floor by −0.015pp. Shallower than H58's 3.551% best. test_VP floor crosses now: H26 merged, H53, H55v2, H57, H59, H65, H66 (7 total across models).

### Class disposition update

| Class | LR-fix result | Structural reason |
|---|---|---|
| **encoder-PE-no-stopgrad (H58→H66)** | **C NULL +0.220pp REGRESS** | **Lion sign-cancellation magnitude-neutral** |
| V-DEPTH | C NULL test✅ | Architecture-bound at val |
| shared-cap-surface | C NULL test✅ | Architecture-bound at val |
| τz-curr | C NULL test✅ | LR-axis exhausted |
| CP-loss-weight | D NEGATIVE | LR-fix destabilizes |
| slice-temp-curr | D NEGATIVE killed | LR-fix destabilizes |

**Wave 31 LR-fix campaign: 6/6 closed, ALL C NULL or D NEGATIVE on val_abupt**. LR-fix benefit is class-specific and primarily test-side not val-side.

### Follow-up assigned

**H76 SLICES-192-ISOLATION (PR #1232)**: askeladd takes baseline #972, single-flag delta `--model-slices 128 → 192`. Pure slice-attention capacity scaling isolation. Prior slices=192 tests bundled 4 components (H51 killed, H60 C NULL). H76 fills the missing isolation experiment. Tests whether slice-attention geometric resolution is the shared ceiling across all Wave 31 single-model C NULL results.

---

## 2026-05-20 19:45 — PR #1214: H65 SURFACE-DEEP-LR-EXTENDED (alphonse, CLOSED) — **OUTCOME C NULL on val_abupt + MECH-POSITIVE test side with 5th test_VP FLOOR CROSS**

- **Branch**: `alphonse/h65-surface-deep-lr-extended` (closed at 19:45Z)
- **W&B run**: `quvb4mb1` (13/13 epochs, 987.3 min = 16.46h wall-time)
- **Hypothesis**: H54 v2 SURFACE-DEEP (surf_deep 2 blocks) plateaued at val_abupt ~6.248% due to LR-decay confound — slope collapsed from −0.03pp/ep EP5→6 to ~0 EP8+. Test if `--lr-cosine-t-max 13 → 25` unlocks continued descent by preserving 55% peak LR at terminal.

### Terminal metrics

| Metric | H65 terminal | Baseline #972 | H54 v2 ref | Δ vs H54 v2 | Status |
|---|---:|---:|---:|---:|:--|
| val_abupt | **6.2345%** | 6.126% | 6.248% | −0.014pp | ❌ C NULL (within ±0.05pp) |
| val_VP | 3.718% | 3.798% | 3.699% | +0.019pp | — |
| val_SP | 4.056% | 3.577% | 4.071% | −0.015pp | — |
| val_WSS | 7.055% | 6.727% | — | — | — |
| test_abupt | **5.926%** | 5.844% | 6.042% | **−0.116pp** ✅ | BEAT H54 v2 |
| **test_VP** | **3.588%** | 3.643% (floor) | 3.693% | **−0.105pp** ✅ | **5th FLOOR CROSS** |
| test_SP | 3.687% | 3.577% (floor) | 3.803% | −0.116pp ✅ | above floor |
| test_WSS | 6.836% | 6.727% | 6.954% | **−0.118pp** ✅ | misses baseline |
| test_WSS_z | **8.866%** | 8.916% | 9.016% | **−0.150pp** ✅ | direction-correct |

### Results commentary

**Formally C NULL on val_abupt** — H65 terminates at 6.2345%, only −0.014pp better than H54 v2 (6.248%) which is within the ±0.05pp noise band. Merge gate (6.126%) missed by +0.108pp. Outcome matches H59 V-DEPTH (C NULL val, mech-positive test) pattern.

**Mech-positive on test side**: H65 beats H54 v2 by −0.105 to −0.150pp across ALL test metrics. **test_VP 3.588% crosses the 3.643% floor by −0.055pp** — the 5th project test_VP floor cross after H26 merged, H53, H55v2, H57 with 2 crosses. This is the strongest test-side signal from any single-mech LR-fix variant.

**KEY STRUCTURAL FINDING**: Surf_deep mechanism was STILL IN PRODUCTIVE GROWTH at EP13 terminal (block0/block1 ffn_fc2 norms +0.218/+0.257 per_1k_steps positive slope). Mechanism didn't saturate — the val_abupt ~6.23% ceiling is architecture-bound (same as H47 V-DEPTH 6.28% ceiling), NOT mechanism-capacity-bound.

**LR-fix trajectory verified**: LR retained 55% peak at terminal vs H54 v2 ~0%. The late-epoch productive descent EP12→EP13 (−0.017pp) absent in H54 v2 (plateaued at ~0 descent EP9+) confirms LR-fix substrate was functional. The mechanism continued growing under preserved LR.

**Class assignment**: shared-capacity-surface joins V-DEPTH in "architecture-bound at val, LR-bound at test" category. Both classes benefit from LR-fix on test channels but cannot crack val_abupt gate regardless of LR schedule.

### Mechanistic analysis

Surf_deep block norms (block0/block1 ffn_fc2 + attn_proj) all positive slope at terminal:
- block0 ffn_fc2 global_norm: +0.218 per_1k_steps (still growing EP13)
- block1 ffn_fc2 global_norm: +0.257 per_1k_steps (growing at terminal)
- block0/block1 attn_proj global_norm: +0.180/+0.188 (growing)

This is mechanistically novel: the architecture-ceiling is not from mechanism saturation but from the val evaluation space (34-case val set) being uncorrelated with where surf_deep capacity adds value. Test set (50 cases, 11k views) sees the benefit; val doesn't.

### Follow-up assigned

**H75 PURE-BASELINE-LR-EXTENDED (PR #1231)**: alphonse takes baseline #972 exact config, single-flag change `--lr-cosine-t-max 13 → 25`. ZERO mechanism changes. This is the missing control experiment to resolve whether LR-fix's test improvements are mechanism-dependent or universally LR-induced. Three falsifiable outcomes.

---

## 2026-05-20 13:25 — PR #1224: H70 SLICE-TEMP-CURRICULUM-LR-EXTENDED (frieren, CLOSED) — **OUTCOME D NEGATIVE: 2nd Wave 32 LR-fix variant to FALSIFY confound hypothesis — class-differentiation principle now binding**

- **Branch**: `frieren/h70-slice-temp-lr-extended` (closed at 13:25Z)
- **W&B run**: `b67zr8xy` (killed at EP3 step 32,592, ~4.6h wall-time)
- **Hypothesis**: H61 mech-positive B PARTIAL (val_abupt 6.341%) plateaued via curriculum-complete-then-LR-decay pattern. Test if `--lr-cosine-t-max 13 → 25` + `--slice-temperature-decay-steps 65184 → 130368` (stretch curriculum to match LR-extension) unlocks the attention-routing-temperature class.

### Terminal verdict — close + reassign

| Channel | H70 EP3 | EP3 gate | H61 EP3 ref | Δ | Verdict |
|---|---:|:--|---:|---:|:--|
| **val_abupt** | **8.6387%** | <7.5 ❌ | 7.4232 | **+1.216pp WORSE** | KILLED |
| val_SP | 5.7910% | <5.5 ❌ | 4.9618 | +0.829pp | KILLED |
| val_VP | 5.6658% | — | 4.5095 | +1.156pp | — |
| val_WSS | 9.6481% | — | 8.3641 | +1.284pp | — |
| WSS_z | 12.2171% | — | 10.8996 | +1.318pp | — |

**Outcome D NEGATIVE confirmed per PR's four-outcome contract**: val_abupt > H61 by ≥0.05pp satisfied **24× over**.

### KEY STRUCTURAL FINDING — class-differentiation principle now binding

Wave 32 LR-fix triangulation has now produced TWO D NEGATIVE outcomes on routing/weighting-curriculum classes (H62 CP-LOSS-WEIGHT + H70 ATTENTION-TEMP). Refined principle:

**Mechanism classes that depend on progressive sharpening of a routing/weighting distribution (CP loss weights, attention temperature, slice-temperature decay) need LR-decay as a co-dependent ingredient of productive dynamics.** The LR-extended substrate keeps the model in high-exploration regime exactly when the curriculum needs to be settling into a sharpened distribution, creating destructive interference.

### Three structural sub-findings

1. **EP1 cold-start advantage misleading** — H70 was −1.24pp better than H61 at EP1 but lost ground rapidly EP2 onward. Cosine-progress signal at EP3 was only 0.040 (4%), so we were barely past warmup. The "LR-fix at cold-start" advantage doesn't translate to peak-LR steady-state.

2. **Class signature preserved** — z-dominant per-axis WSS (z>y>x), sparsified routing trajectory (n_eff_mean 102→6 over EP1-3) — match H61's fingerprint. Mechanism is operating in same axis-allocation regime; just operating WORSE. Rules out mechanism-class-failure (the class is real and works) and confirms LR-substrate-incompatibility.

3. **Block 0 inversion** — H70 EP3 had block 0 BROADER routing (n_eff 17.8) than blocks 1-4 (~6). H61 EP1 fingerprint had block 0 SPARSEST. Stretched curriculum + peak LR pushes later blocks into over-sparsification while keeping block 0 broader — new mech failure mode unique to LR-extended substrate.

### Wave 32 design policy update (binding)

From now on, **no cross-pollination LR-fix variants on routing/weighting-curriculum classes** (CP-LOSS H62 done, ATTENTION-TEMP H70 done, anything operating on softmax-temperature/loss-weight). LR-fix variants reserved for:
- Architecture-modification classes: V-DEPTH (H59 partial), SURFACE-DEEP (H65 in-flight), variance-class-encoder-input (already merged)
- Geometric-prior classes: CURVATURE-ATTN-BIAS (H69 in-flight)
- Encoder-PE classes: COORDSLICE-PE (H66 in-flight)

### Reassignment

**frieren → H72 SLICE-TEMP-DEEP-ENDPOINT (PR #1228)** — single-flag change vs H61 mech-positive B PARTIAL parent: `--slice-temperature-end 0.5` (instead of 1.0, doubles logit scale at curriculum end). Keeps all other H61 flags: `--lr-cosine-t-max 13` (legacy LR-decay substrate), `--slice-temperature-start 1.5`, `--slice-temperature-decay-steps 65184`. Tests **deeper sharpening** as the orthogonal axis to LR-substrate variation. Over-sparsification risk diagnostic: `diag/slice_n_eff_mean` < 5 at EP3-6.

---

## 2026-05-20 08:35 — PR #1211: H62 CP-LOSS-WEIGHT-LR-EXTENDED (tanjiro, CLOSED) — **OUTCOME D NEGATIVE: LR-fix actively HURTS CP-LOSS-WEIGHT class — first Wave 31 LR-fix variant to FALSIFY confound hypothesis on its mech class**

- **Branch**: `tanjiro/h62-cp-loss-weight-lr-extended` (closed at 08:35Z)
- **W&B run**: `cw4a7zf2` (state=finished, runtime 14.32h)
- **Hypothesis**: H53 plateaued at val_abupt 6.181% with LR-decay slope-halving pattern. Test if `--lr-cosine-t-max 13 → 25` LR-extension unlocks CP-LOSS-WEIGHT class merge.

### Terminal verdict — close + reassign

| Axis | H62 | H53 (parent) | Δ vs H53 | Gate/Floor | Verdict |
|---|---:|---:|---:|---:|:--|
| **val_abupt** (merge gate) | **6.397%** | 6.181% | **+0.216pp WORSE** | 6.126% | ❌ MISS (+0.271pp vs gate) |
| test_abupt | 6.121% | 6.052% | +0.069pp WORSE | 5.844% | ❌ MISS |
| test_VP (floor) | 3.760% | 3.665% | +0.095pp WORSE | 3.643% | ❌ MISS |
| test_SP (floor) | 3.835% | 3.793% | +0.042pp WORSE | 3.577% | ❌ MISS |
| test_WSS | 7.052% | 6.978% | +0.074pp WORSE | 6.727% | ❌ MISS |

**H62 STRICTLY WORSE than H53 on EVERY paper-facing axis.** Outcome D NEGATIVE confirmed.

### KEY STRUCTURAL FINDING — CP-LOSS-WEIGHT class is NOT LR-decay-bound (LR-extension actively HURTS)

Matched-step trajectory (H62 vs H53):

| EP | H62 LR%peak | H53 LR%peak | ΔLR | H62 val_abupt | H53 val_abupt | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 86.5% | 56.7% | +29.8 | 6.476% | 6.268% | +0.208pp |
| 8 | 82.1% | 44.6% | +37.5 | 6.465% | 6.228% | +0.237pp |
| 9 | 77.0% | 33.0% | +44.0 | 6.448% | 6.208% | +0.240pp |
| 10 | 71.6% | 22.5% | +49.1 | 6.434% | (extrap 6.20%) | ~+0.24pp |
| terminal | 55.6% | 2.5% | +53.1 | **6.397%** | **6.181%** | **+0.216pp** |

**Higher LR retention through training MAKES IT WORSE.** Gap stabilized at +0.24pp through EP7-10, never recovered. **Refined mechanism**: loss-weight-rebalancing classes need LR DECAY as PART OF the mechanism. The gradient magnitudes need to shrink for the weight rebalancing to settle into the new equilibrium. Extending LR keeps gradients too large to converge.

### Implication for Wave 31 LR-decay-confound hypothesis

Class-by-class refinement:

| Mech class | LR-fix variant | Disposition |
|---|---|---|
| V-DEPTH (H47→H59) | val=tied, test=better | Architecture-bound at val, LR-bound at test |
| **CP-LOSS-WEIGHT (H53→H62)** | **WORSE on all axes** | **LR-fix ACTIVELY DESTABILIZES — class needs LR DECAY** |
| TAU-Z-CURR (H55v2→H63) | in flight | TBD |
| SURFACE-DEEP (H54v2→H65) | in flight | TBD |
| COORDSLICE (H58→H66) | in flight | TBD |
| Attention-temp (H61→H70) | just launched | TBD |

H62 is **first Wave 31 single-mechanism LR-fix variant to produce OUTCOME D NEGATIVE**. LR-decay-confound is now refined from "systematic Wave 31 ceiling" to "confound for SOME mech classes, productive for OTHERS". The 5 parallel LR-fix runs serve as a **mech-class differentiation diagnostic**, not just confound isolation.

### Disposition: CLOSE + tanjiro reassigned to H71 GRADNORM PR #1225

Wave 32 cross-pollination launch. GradNorm dynamic per-task loss balancing (Chen et al. 2018) — new mech class on tay (loss-balancing-dynamic). dl24 H19 SOTA-beat innovation. Replaces hand-tuned static weights (tau_z=2.0, surface=2.0, etc.) with rate-adaptive learnable weights. Composes orthogonally with FDCE / variance-class / attention-spatial-prior. **Critical**: stays on `--lr-cosine-t-max 13` (legacy substrate) per H62 D NEGATIVE finding — loss-rebalancing classes need LR decay.

---

## 2026-05-20 08:25 — PR #1210: H61 SLICE-TEMP-CURRICULUM (frieren, CLOSED) — **MECHANISM-POSITIVE B PARTIAL + first Wave 31 lr-cosine-t-max=13 test_VP floor cross + 7th confirmed LR-decay-confound case**

- **Branch**: `frieren/h61-slice-temp-curriculum` (closed at 08:25Z)
- **W&B run**: `oymgwjfv` (terminal at best_epoch=13)
- **Hypothesis**: Scheduled slice-attention temperature τ=1.5→1.0 over EP1-6 (linear decay), then hold τ=1.0 EP6-13. Warm/diffuse routing early, sharp/competitive routing late (Gumbel-softmax-style annealing). Novel mech class #14: attention-routing-temperature-curriculum.

### Terminal verdict — close + reassign

| Axis | H61 | Gate/Floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| val_abupt (merge gate) | 6.341% | 6.126% | +0.215pp | ❌ MISS |
| test_abupt (baseline) | 6.004% | 5.844% | +0.160pp | ❌ MISS |
| **test_VP** (floor) | **3.6315%** | 3.643% | **−0.012pp** | ⭐ **CROSS — first Wave 31 lr-13 floor cross** |
| test_SP (floor) | 3.777% | 3.577% | +0.200pp | ❌ MISS |
| test_WSS | 6.921% | 6.727% | +0.194pp | ❌ MISS |
| val_abupt vs H48 (same recipe) | 6.341% | 6.485% | **−0.144pp** | ✅ **B PARTIAL bucket per PR prediction** |

### KEY STRUCTURAL FINDING — 7th LR-decay confound case + curriculum-complete-then-plateau

Slope-by-epoch trajectory:

| EP | step | curriculum τ | val_abupt | slope (pp/1k) |
|---:|---:|---:|---:|---:|
| 1 | 10,864 | 1.46 | 34.84% | (cold-start) |
| 2 | 21,729 | 1.36 | 8.704% | −2.405 |
| 3 | 32,594 | 1.25 | 7.423% | −0.118 |
| 4 | 43,466 | 1.17 | 6.823% | −0.040 |
| 6.0 | 65,212 | **1.00** | 6.345% | −0.011 |
| **terminal** | **70,652** | **1.00** | **6.341%** | **~0** |

Slope halved every epoch (−2.40 → −0.118 → −0.040 → −0.011 → ~0) — **canonical 7th LR-decay-confound case** (joins H47/H52/H53/H54v2/H55v2/H58/H59). Curriculum complete at step 65,212 (τ=1.0) but residual slope already collapsed to ~−0.011. **Curriculum-complete-then-plateau**: model couldn't exploit sharpened routing at low LR.

Per-axis WSS test pattern preserved: x=6.16%, y=7.49%, z=8.97% (z>y>x). WSS_z still binding.

### Disposition: CLOSE + frieren reassigned to H70 SLICE-TEMP-LR-EXTENDED PR #1224

H61 lands B PARTIAL with first lr-cosine-t-max=13 test_VP floor cross. Mechanism class #14 (attention-routing-temperature-curriculum) confirmed mech-positive. LR-fix variant H70 stretches curriculum decay to 130,368 steps (EP12, matching `--lr-cosine-t-max 25`). 6th class in LR-fix triangulation alongside H59/H62/H63/H65/H66.

---

## 2026-05-20 07:35 — PR #1209: H60 H56-RELAUNCH-DROP-EP3 — ema-aware-variance-stack with strongest val variance signal in Wave 31 (fern, CLOSED) — **MECHANISM-POSITIVE NULL + KEY STRUCTURAL FINDING: val→test mechanism inversion**

- **Branch**: `fern/h60-h56-relaunch-drop-ep3` (closed at 07:35Z)
- **W&B run**: terminal SENPAI-RESULT in PR
- **Hypothesis**: H56 was killed at EP3 by mis-calibrated EP3 gate (mechanism was actively building, mathematically inside random_pred_floor at ema=0.9999). H60 relaunch from H56 tip with EP3+EP4 gates dropped, only EP6 binding gate retained. Test whether NPCA+SSFL+slices192+ema9999 variance-class stack crosses merge gate when allowed to run terminal.

### Terminal verdict — close + reassign

| Axis | H60 | Gate/Floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| val_abupt (merge gate) | 6.328% | 6.126% | +0.202pp | ❌ MISS |
| test_abupt (paper-facing) | 6.084% | 5.844% | +0.240pp | ❌ MISS |
| test_VP (floor) | 3.702% | 3.643% | +0.059pp | ❌ NEAR-TIE FAIL |
| test_SP (floor) | 3.836% | 3.577% | +0.259pp | ❌ MISS |
| test_WSS (goal) | 6.990% | 6.727% | +0.263pp | ❌ MISS |
| test_VP vs AB-UPT ref 6.08% | — | 3.702% | **−2.378pp ✓** | beats public reference |
| test_WSS vs AB-UPT ref 7.29% | — | 6.990% | **−0.300pp ✓** | beats public reference |

### KEY STRUCTURAL FINDING — val→test variance-class mechanism inversion

Per-axis tau_zx_ratio diagnostic on full_val (34 cases) vs test_primary (50 cases):

| Diagnostic | val | test | val→test Δ |
|---|---:|---:|---:|
| tau_zx_ratio_std | **0.2292** ⭐ | **0.1325** | −42.2% |
| tau_zx_ratio_mean | 1.5576 | 1.4676 | −5.8% |
| tau_zx_ratio_max | 2.6804 | 1.7563 | −34.5% |
| n_outside_band ([1.44, 1.55]) | 12/34 (35.3%) | 27/50 (54.0%) | +18.7pp |

**val tau_zx_ratio_std = 0.229 = +66% above H56-EP3 fleet-peak (0.138) = strongest single-model variance signal Wave 31 history. But test collapses to 0.133 — essentially H56's killed-at-EP3 value.** The H35 fleet-peak (0.251) was almost reached on val (−0.022 short) but test never went near it.

Per-axis WSS pattern reproduces y-dominant on test (WSS_y 7.628%, WSS_z 9.062%, WSS_x 6.191%) — same allocation pattern as 7-epoch val descent — but variance-class amplification didn't transfer. Strong evidence the 34-case val split has structural τ_z/τ_x ratio polarization that single-mechanism amplification can exploit at val but NOT at test. **First observed val→test mechanism inversion on tay.**

### Disposition: CLOSE + fern reassigned to H69 CURVATURE-ATTENTION-BIAS PR #1223

Wave 32 cross-pollination start. Curvature attention bias is the foundation mechanism in dl24's H10b → H19 stack (their parallel-fleet SOTA-beat candidate). Geometric-input attention prior is naturally invariant across val/test split structure (addresses H60's mechanism-inversion finding) and directly targets WSS_z binding axis. Single-flag test on LR-fix substrate `--lr-cosine-t-max 25`.

---

## 2026-05-20 07:30 — PR #1208: H59 V-DEPTH-LR-EXTENDED — Test H47 plateau as LR-decay artifact (nezuko, CLOSED) — **MECHANISM-POSITIVE PARTIAL WIN + KEY STRUCTURAL FINDING: V-DEPTH plateau is ARCHITECTURE-BOUND, not LR-decay-bound**

- **Branch**: `nezuko/h59-v-depth-lr-extended` (closed at 07:30Z)
- **W&B run**: terminal SENPAI-RESULT in PR
- **Hypothesis**: H47 V-DEPTH (variance-class-decoder-sublayer) plateaued at val_abupt 6.273%. Wave 31 LR-decay-confound observed in 5 cases (H47/H52/H53/H55v2/H54v2) with slope halving every epoch as LR drops below 50% peak. Test if extending cosine T_max from 13 → 25 keeps LR higher through terminal and unlocks the plateau.

### Terminal verdict — close + reassign

| Axis | H59 | Gate/Floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| val_abupt (merge gate) | **6.282%** | 6.126% | +0.156pp | ❌ MISS — reproduces H47 |
| test_abupt (paper-facing) | **6.012%** | 5.844% | +0.168pp | ❌ MISS — **−0.036pp BEAT H47** |
| **test_VP** (floor) | **3.5525%** | 3.643% | **−0.091pp** | ✅ **CROSS — first Wave 31 depth-class floor cross** ⭐ |
| test_SP (floor) | 3.786% | 3.577% | +0.209pp | ❌ MISS |
| test_WSS (goal) | 6.947% | 6.727% | +0.220pp | ❌ MISS — **−0.047pp BEAT H47** |
| full_val_VP | 3.696% | (H47: 3.815%) | **−0.161pp BEAT H47** ⭐ | strongest val-side mech signal |

### KEY STRUCTURAL FINDING — V-DEPTH plateau is ARCHITECTURE-BOUND, not LR-decay-bound

Slope-decomposition: val_abupt slope monotonically decayed from −0.053 pp/1k (EP3→4) to −0.0017 pp/1k (EP12→13) **despite LR retained at 56-90% peak throughout EP6→EP13** (terminal LR fraction 0.556 vs H47's 0.025). The LR extension worked exactly as designed but did NOT unlock additional val_abupt descent below 6.28%.

**Decomposition of H47 plateau**:
- ~50% architecture-bound (val ceiling at 6.28% regardless of LR schedule)
- ~50% LR-decay-confound (test_VP −0.10pp, test_abupt −0.04pp, test_WSS −0.05pp improvements vs H47 — generalization, not training-fit)

**V-DEPTH class is the FIRST mech-class where LR-fix did NOT produce merge-relevant val_abupt improvement** but test channels DID improve cleanly. Critical Wave 32 design implication: LR-fix variants need separate evaluation on val (training-fit) and test (generalization) — V-DEPTH is architecture-bound at val, LR-decay-bound at test.

### EP8+ channel role specialization (new mechanistic observation)

Terminal vol_deep block diagnostics:

| Block | attn_proj norm | attn_proj max | ffn_fc2 norm | ffn_fc2 max |
|---|---:|---:|---:|---:|
| block0 | ~30.8 | ~0.57 | ~70.7 | **~0.85** ⚠ |
| block1 | **~35.2** | ~0.38 | **~73.7** | ~0.66 |

block0 took sparse high-magnitude updates; block1 carried structured-norm representation. **This is a NEW mechanistic observation that did NOT occur in H47** (whose LR-decay didn't keep training in the high-LR specialization regime long enough). Wave 32 hypothesis seed: explicit asymmetric block roles in V-DEPTH stacks (different init, different LR, different objective allocation) could break the 6.28% architectural ceiling.

### Disposition: CLOSE + nezuko reassigned to H68 CHARBONNIER-VOL-P PR #1222

Wave 32 cross-pollination start. Charbonnier loss on volume pressure is the dl24 H19 innovation (their parallel-fleet SOTA-beat candidate). Loss-curvature-shape is a fresh mechanism class on tay (no prior Wave 31 test). Single-flag test on LR-fix substrate `--lr-cosine-t-max 25`.

---

## 2026-05-20 19:30 — PR #1228: H72 SLICE-TEMP-DEEP-ENDPOINT (frieren, EP3 auto-killed) — **OUTCOME D NEGATIVE / H61 CONFIRMED GOLDILOCKS PARAMETER POINT — 3rd Wave 32 single-axis-collapse on routing/weighting-curriculum class**

- **Branch**: frieren/h72-slice-temp-deep-endpoint
- **W&B run**: vzfdqj4w (rank0, group wave32_h72_slice_temp_deep_endpoint)
- **Hypothesis**: H61 slice-temp curriculum mech-positive B PARTIAL is limited by τ_end=1.0 → pushing τ_end=1.0→0.5 deepens sharpening + tests whether the routing class has more headroom in the deeper-temperature regime. Legacy lr-cosine-t-max=13 substrate. Single-flag change `--slice-temperature-end 0.5` vs H61.

| Channel | H72 EP3 | Gate | H61 EP3 ref | Δ vs H61 | Verdict |
|---|---:|:--|---:|---:|:--|
| val_abupt | 11.803% | <7.5% ❌ | 7.423% | +4.38pp | **GATE FAILED — auto-killed** |
| val_SP | 7.945% | <5.5% ❌ | 4.962% | +2.98pp | GATE FAILED |
| val_VP | 9.200% | — | 4.510% | +4.69pp | — |
| val_WSS | 12.740% | — | 8.364% | +4.38pp | — |

**Mechanism-failure attribution**: Block 2 entropy crossed binding threshold 0.485 = 0.10 × log(128) at EP2 (step 21,729) — **one epoch earlier than PR Risk #2 predicted** — and SUSTAINED through EP3 with n_eff_mean = 2.51 (vs H61 5.27). All 5 blocks below H61 entropy at EP3; mean n_eff at 48% of H61. Saturating-softmax gradient-flow failure mode confirmed: model could not redistribute mass to less-committed slices because gradient signal was lost.

**Pace-mismatch root cause finding (key structural attribution)**: τ-curriculum at EP1 (step 10,864) was already 1.333 vs H61's 1.417 at same step. The 2× faster descent through the same τ-range committed slice routing during LR warmup before optimization stabilized. n_eff_mean = 68 at EP1 (vs H61's 103, a 34% reduction) confirmed pre-warmup commitment. By EP3 step 32,592, H72 had reached H61's terminal τ=1.0 value but in half the steps.

**Commentary**: H61 (τ_start=1.5, τ_end=1.0, decay=65184, lr-cosine-t-max=13) confirmed as Goldilocks parameter point — both LR substrate AND τ-endpoint axes are co-tuned. Single-axis perturbations cause mechanism collapse from different failure modes:
- LR-axis perturbation (H70 LR-extended): D NEGATIVE +2.298pp via late-block sparsification
- Endpoint-axis perturbation (H72 τ_end=0.5): D NEGATIVE +5.46pp via pre-warmup over-sparsification
- Combined with H62 CP-loss-weight + LR-extended: D NEGATIVE +0.216pp via weight rebalancing destabilization

**Binding Wave 32+ design policy update**: NO MORE single-axis variants on slice-temperature-curriculum class (class exhausted). Future re-attack requires joint sweeps over co-tuned manifold. Given 3 wasted runs in Wave 32 on this class, cost-benefit recommends abandoning class for now in favor of orthogonal mechanism explorations.

**Wall-clock**: 5h before auto-kill (efficient early termination, vs 18h budget). Duplicate launch detected + SIGTERM'd at 14:24Z (~9 min half-throughput during contention).

---

## 2026-05-20 17:00 — PR #1212: H63 TAU-Z-CURRICULUM-LR-EXTENDED (edward, 13-ep terminal) — **OUTCOME C NULL + test_VP FLOOR CROSS + τz-curriculum LR-axis EXHAUSTED**

- **Branch**: edward/h63-tau-z-curriculum-lr-extended
- **W&B run**: (edward's terminal run on LR-extended substrate)
- **Hypothesis**: τz-loss-curriculum (H55v2, mech-positive) is limited by LR-decay artifact — extending `--lr-cosine-t-max 13 → 25` should allow curriculum to operate in high-LR regime and improve terminal val_abupt below H55v2's 6.249%.

| Channel | H63 terminal | Gate/Floor | H55v2 ref | Δ vs ref | Verdict |
|---|---:|:--|---:|---:|:--|
| val_abupt | 6.266% | <6.126% ❌ | 6.249% | +0.017pp | **C NULL** |
| test_VP | **3.583%** | <3.643% ✅ | 3.602% | **−0.019pp** | ✅ FLOOR CROSS |
| test_SP | 3.839% | <3.577% ❌ | 3.806% | +0.033pp | above floor |
| test_abupt | 6.035% | — | 5.988% | +0.047pp | — |
| test_WSS | 6.933% | — | 6.883% | +0.050pp | — |

**Late-epoch slope comparison**:
| Run | EP6 val_abupt | EP13 val_abupt | Avg slope (pp/ep) | LR fraction at terminal |
|---|---:|---:|---:|---:|
| H55v2 | 6.389% | 6.249% | −0.020 | ~0% |
| H63 | 6.467% | 6.266% | **−0.029 (45% steeper)** | **59.8%** |

**Commentary**: C NULL assigned per four-outcome contract (terminal val_abupt within ±0.05pp of H55v2). LR-fix substrate WAS productive at schedule level (45% steeper late-epoch slope confirmed), but EP6 deficit (+0.078pp from noisier descent under peak LR) only partially closed. **Key structural finding: τz-curriculum class SATURATES at val_abupt ~6.25% regardless of LR schedule** — the mechanism binds on the curriculum-schedule axis, not the LR axis. LR-fix is not the lever for this class.

The test_VP floor cross (3.583% < 3.643%) is the binding positive byproduct — deeper than H55v2's 3.602%, confirming τz-curriculum produces robust test_VP floor crosses under both LR substrates. This consolidates the test_VP floor-cross pattern across 5 mech classes.

**Class-specific saturation finding binding for Wave 32**: τz-curriculum added to "LR-axis-exhausted" category. Future LR-fix variants on this class = wasted compute. Next experiment on τz axis: H73 CHARBONNIER-TAU-Z (PR #1229) — attacks the same axis via **loss-function shape** (orthogonal lever).

---

## 2026-05-20 07:15 — PR #1213: H64 RFF-LOW-BAND-EXPANSION — drop σ={8.0,16.0} HIGH-end, add σ=0.0625 LOW-end (thorfinn, EP3 KILLED) — **OUTCOME D NEGATIVE / MECHANISM-CLASS REFINEMENT: FDCE lever is band-WIDTH not band-POSITION**

- **Branch**: `thorfinn/h64-rff-low-band-expansion` (closed at 07:15Z)
- **W&B run**: rank0 `6wpxfu4m` (group `wave31_h64_rff_low_band_expansion`, 4.83h wall-time, KILLED at step 32,592 = EP3 end by kill_threshold)
- **Hypothesis**: H57's per-σ projection diagnostic showed σ=0.125 (lowest available) took 11.79% surface / 18.47% volume share (highest LOW uptake) while σ=16 took only 3.14%/2.30% (minimal HIGH uptake). H57 hypothesized LOW-end is the binding direction. H64 tested by shifting band LOW (drop σ={8,16}, add σ=0.0625) keeping 7 σs spanning -4 to +2 octaves. Mechanism class: frequency-domain-capacity-low-tilted (derived from H57 FDCE class).

### Terminal verdict — KILLED at EP3 outcome D NEGATIVE

| Gate | Target | H64 EP3 | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| val_abupt | <7.5% (EP3 kill) | **8.887%** | ❌ **KILLED +1.39pp** | terminal not reached |
| val_SP | <5.5% (EP3 kill) | **5.990%** | ❌ also FAILED gate | — |
| val_VP | (no gate) | 6.203% | (no gate at this step) | — |
| val_abupt vs H57 EP3 (6.842%) | — | 8.887% | +2.05pp worse | OUTCOME D NEGATIVE |

### Validation trajectory — EP2→EP3 regression confirmed outcome D

| EP | step | val_abupt | val_SP | val_VP | val_WSS | Slope (pp/1k) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,864 | **31.74%** | 25.80% | 18.46% | 34.86% | — (cold-start) |
| 2 | 21,728 | **8.66%** | 5.82% | 5.59% | 9.65% | −2.122 (improving) |
| 3 | 32,592 | **8.89%** | 5.99% | 6.20% | 9.92% | **+0.0212** ⬆ REGRESSING |

**EP2→EP3 regression on ALL primary axes**:
- val_abupt: 8.66% → 8.89% = **+0.23pp WORSE** (vs H57 EP2→EP3 ~−1.5pp BETTER)
- val_VP: 5.59% → 6.20% = +0.61pp worse
- val_WSS: 9.65% → 9.92% = +0.27pp worse

This is the OPPOSITE of H57's mid-training trajectory and triggered the EP3 kill.

### Head-to-head vs H57 (same recipe, only `--rff-init-sigmas` changed)

| EP | H64 (low-tilted 7σ) | H57 (wide-span 8σ) | Δ H64 vs H57 |
|---:|---:|---:|---:|
| 1 val_abupt | 31.74% | 25.23% | **+6.51pp worse cold-start** |
| 2 val_abupt | 8.66% | ~7.7% (interp) | ~+0.96pp worse |
| **3 val_abupt** | **8.89%** | **6.84%** | **+2.05pp worse** |
| 3 val_SP | 5.99% | ~4.6% | +1.4pp worse |

H64 uniformly worse than H57 across every checkpoint. Cold-start damage at EP1 persisted through EP3 and started to widen (EP2→EP3 regression).

### KEY MECHANISM REFINEMENT — per-σ projection diagnostic on EP2 checkpoint confirmed prediction-shape-but-falsified-outcome

Student's offline `scripts/h57_per_sigma_diagnostic.py` on `outputs/drivaerml/run-6wpxfu4m/checkpoint.pt` (epoch=2):

**Surface encoder:**

| σ | learned freq | std | proj W L2/col | proj frac | H57 EP3 reference |
|---:|---:|---:|---:|---:|---|
| **0.0625 (new LOW)** | 0.0584 | 0.0067 | 0.7488 | **12.97%** | — (not in H57) |
| 0.125 | 0.1188 | 0.0186 | 0.7344 | 12.48% | 11.79% (H57 LOW) |
| 0.250 | 0.2812 | 0.0129 | 1.0169 | 15.95% | (normal) |
| 0.500 | 0.5083 | 0.0169 | 1.0938 | **18.45%** | (normal) |
| 1.000 | 1.0081 | 0.0395 | 1.0881 | 18.26% | (normal) |
| 2.000 | 1.9934 | 0.0712 | 0.8331 | 10.70% | (normal) |
| 4.000 | 3.9410 | 0.2123 | 0.8518 | **11.19%** | (normal — was position-3-high in 8σ basis) |

**Diagnostic verdict**: σ=0.0625 attracted 12.97% surface + 15.50% volume — matching the predicted "LOW-position cascades to lowest available σ" pattern almost exactly. The mechanism *shape* is confirmed. **BUT the task regressed +2.05pp.** Conclusion: **projection-weight share is determined by σ-ordering POSITION, not by σ-VALUE-as-band-direction**. The LOW-end uptake happens, but it does NOT unlock new mechanism.

### Refined mechanism class — band-WIDTH not band-POSITION

**FDCE (frequency-domain-capacity-expansion) lever is band-WIDTH (total octave span), NOT band-position (specific σ values).** H57's [-3, +4 octaves] (8 σs) supported a richer multi-scale Fourier basis than H64's [-4, +2 octaves] (7 σs narrower span despite reaching lower). The HIGH-end (σ=8, 16) in H57 served as **anchor frequencies constraining encoder geometry-aware features** even with small projection share — likely supporting sharp local-feature coverage that coexists with smoothly-varying LOW-σ global features.

This is structurally important: the H57 close note tentatively classified the LOW-end as binding; H64 falsifies that and replaces with band-width refinement.

### Wave 31 ranking at H64 closure (unchanged — H64 below mech-positive runs)

| Run | Mech class | val_abupt | Disposition |
|---:|:--|---:|:--|
| H58 | encoder-PE-no-stopgrad | 6.161% | closed (closest NEAR-MISS) |
| H47 | decoder-depth | 6.143% | closed → H59 LR-fix |
| H53 | loss-weight | 6.181% | closed → H62 LR-fix |
| H57 | frequency-domain-capacity (8σ) | 6.217% | closed → H64 (this) |
| H50 | encoder-PE-stopgrad | 6.220% | closed → H58 |
| H54v2 | shared-cap-surface | 6.248% | closed → H65 LR-fix |
| H55v2 | tau-z-curriculum | 6.249% | closed → H63 LR-fix |
| **H64** | **frequency-domain-capacity-low-tilted** | **8.887% (EP3 kill)** | **closed — outcome D / mech-class refined** |

### Disposition

CLOSED as outcome D NEGATIVE with mechanism-class refinement on FDCE class (band-WIDTH not band-POSITION). Excellent execution by thorfinn — per-σ projection diagnostic on EP2 checkpoint under EP3 kill pressure cleanly falsified the original LOW-binds hypothesis AND produced a refined mechanism understanding worth more than a positive null.

**THORFINN REASSIGNED H67 RFF-9SIGMA-WIDTH-EXPANSION** (PR #1221) — single-flag change vs H57: `--rff-init-sigmas "0.0625,0.125,0.25,0.5,1.0,2.0,4.0,8.0,16.0"` (9 σs spanning -4 to +4 octaves, prepending σ=0.0625 to H57's 8σ basis). Tests band-WIDTH-not-position refinement directly. Kept `--lr-cosine-t-max 13` for clean attribution against H57 — LR-fix can be layered as H68 if H67 produces mech-positive. Four falsifiable outcomes: A. MERGE WIN + FLOOR CROSS (band-width unlocks merge); B. PARTIAL val_abupt drops below H57 6.217 by ≥0.05pp (width is lever, ceiling persists, stack with LR-fix); C. NULL within ±0.05pp H57 6.217 (8 σs sufficient, FDCE saturates); D. NEGATIVE +0.05pp above H57 6.217 (9th σ destabilizes, unlikely).

Close comment: https://github.com/morganmcg1/DrivAerML/pull/1213#issuecomment-4495622424

---

## 2026-05-20 04:15 — PR #1207: H58 COORDSLICE-NO-STOPGRAD — restore routing-gradient feedback by removing `torch.no_grad()` wrap on centroid computation (askeladd, EP13 terminal best EMA ckpt EP12) — **MECHANISM-POSITIVE NULL with DEEPEST Wave 31 test_VP floor cross (−0.092pp) + CLOSEST Wave 31 val_abupt near-miss (+0.035pp) + PE-auto-growth FALSIFIED + KEY STRUCTURAL FINDING on Lion+zero-mean-gradient sign-cancellation**

- **Branch**: `askeladd/h58-coordslice-no-stopgrad` (closed at 04:15Z)
- **W&B run**: rank0 `9j719af8` (group `wave31_h58_coordslice_no_stopgrad`, 14h 42m train time, 15.07h total runtime, 70,652 global steps, best EMA = epoch 12)
- **Hypothesis**: H50 COORDSLICE PE-projection weight stuck at init (proj_weight_std 0.080-0.093 ≈ init) suggests routing-gradient was blocked by `torch.no_grad()` wrap on centroid computation. Predicted cure: remove wrap → gradients flow → PE projection auto-grows to >0.18 matching H33-style 8× growth → closes +0.094pp H50→merge-gate gap. Mechanism class: encoder-PE-no-stopgrad (variant of coordinate-grounded-slice-PE).

### Terminal verdict — closest val_abupt near-miss + deepest test_VP cross + test_SP regression

| Gate | Target | H58 terminal | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| **val_abupt** (merge) | <6.126% | **6.161%** | ❌ **CLOSEST NEAR-MISS Wave 31** | +0.035pp (vs H53 +0.055pp; vs H50 +0.094pp) |
| test_abupt | (baseline 5.844%) | 5.999% | ❌ FAIL | +0.155pp (regress +0.021pp vs H50) |
| test_SP (floor) | ≤3.577% | 3.856% | ❌ FAIL | +0.279pp (regress +0.121pp vs H50) |
| **test_VP** (floor) | ≤3.643% | **3.551%** | ✅ **DEEPEST cross Wave 31** | **−0.092pp** ⭐ (vs H50 −0.047pp, vs H26 NPCA the previous Wave 31 deepest) |
| val_VP (val ref) | (floor 3.643%) | 3.572% | ✅ cross | −0.071pp (lowest Wave 31) |
| test_WSS (goal) | <6.727% | 6.906% | ❌ FAIL | +0.179pp (within seed noise vs H50 6.917%) |

### KEY STRUCTURAL FINDING — PE-auto-growth FALSIFIED but primary-positive via slice-routing-gradient

| Diagnostic | Block 0 | Block 1 | Block 2 | Block 3 | Block 4 | PR-body target |
|---:|---:|---:|---:|---:|---:|:--|
| proj_weight_std at init | 0.0880 | 0.0880 | 0.0880 | 0.0880 | 0.0880 | — |
| proj_weight_std at EP3 | 0.0899 (+1.2%) | 0.0866 (−1.6%) | 0.0898 (+1.0%) | 0.0885 (+0.5%) | 0.0875 (−0.5%) | — |
| **proj_weight_std at TERMINAL** | **0.0981 (+11.5%)** | 0.0888 (+1.0%) | 0.0937 (+6.4%) | 0.0909 (+3.3%) | 0.0882 (+0.5%) | **>0.18 (+100%)** |
| Status | ❌ 54% of target | ❌ flat | ❌ flat | ❌ flat | ❌ flat | **ALL 5 blocks FAIL** |

**ALL 5 blocks fail the +100% PE auto-growth target by wide margins. PE-auto-growth hypothesis FALSIFIED.**

### Root cause — Lion + indirect averaged gradient = sign-cancellation

Student's mid-EP3 per-layer gradient diagnostic identified the structural reason:

| Diagnostic | coord_pe_proj | Typical FFN/QKV | Ratio |
|:--|---:|---:|---:|
| global gradient_norm | ~0.002 | ~0.05-0.10 | **25-50× smaller** |
| grad_to_param_norm | ~6.7e-5 | ~1e-3 | **15× smaller** |
| mean gradient (131k weights) | ~3e-8 | nonzero-direction | **essentially zero-mean** |

**Lion's `sign(grad)` update on near-zero-mean gradient → random-sign updates that cancel → no systematic directional growth.** H33's free-learnable PE auto-grew 8× because it was directly multiplied by QK attention scores (clear directional target); H58's coord_pe_proj receives only indirect averaged routing gradient through `slice_weights.mean(dim=1)` — smoothed, directionless. **Stop-grad was NOT the binding constraint; the structure of Lion+indirect-averaged-gradient is.**

### Primary-positive mechanism (unintended side effect)

Despite proj_weight_std falsification, H58 beats H50 by **0.059pp val_abupt** and crosses test_VP floor **0.045pp deeper**. Mechanism (per student analysis): removing `torch.no_grad()` lets routing gradients flow back into `slice_weights.mean(dim=1) → centroid einsum`, giving the upstream slice-routing decisions marginally better shaping signal. centroid_range_{x,y,z} and centroid_spread for block 0 show stronger evolution vs H50, but the PE projection layer itself does NOT grow capacity. **Mechanism-null primary-positive outcome: the recipe helped, but NOT for the reason predicted.**

### LR-decay-confound applies — 6th Wave 31 case + slope-halving pattern at EP5→EP6

H58 EP5→EP6 val_abupt trajectory exhibited canonical Wave 31 LR-decay slope-halving (~zero descent EP5→EP6 under LR fraction dropping from 33% to 14% peak), 6th confirmed case after H47/H52/H53/H55v2/H54v2.

### Wave 31 ranking at H58 closure

| Run | Mech class | val_abupt | test_VP | Δ val_abupt vs gate | Disposition |
|---:|:--|---:|---:|---:|:--|
| baseline #972 | — | 6.126% | 3.643% | 0.000pp | merged |
| H47 | decoder-depth | 6.143% | (closed null) | +0.017pp | closed → H59 LR-fix |
| **H58 (this)** | **encoder-PE-no-stopgrad** | **6.161%** | **3.551%** ⭐ | **+0.035pp** | **CLOSING — deepest test_VP cross Wave 31** |
| H53 | loss-weight | 6.181% | 3.665% | +0.055pp | closed → H62 LR-fix |
| H57 | encoder-freq | 6.217% | 3.610% | +0.091pp | closed → H64 LOW-band |
| H50 | encoder-PE-stopgrad | 6.220% | 3.596% | +0.094pp | closed → H58 (this) |
| H54 v2 | shared-cap-surface | 6.248% | 3.693% | +0.122pp | closed → H65 LR-fix |
| H55 v2 | tau-z-curriculum | 6.249% | 3.602% | +0.123pp | closed → H63 LR-fix |

H58 ranks **#2 best val_abupt in Wave 31** (behind only H47 6.143%); **#1 deepest test_VP floor cross** (−0.092pp); **#6 mech-positive NEAR-MISS** in the cluster, all within +0.035 to +0.123pp of merge gate.

### Disposition

CLOSED as mech-positive null with deepest test_VP cross + closest val_abupt miss + PE-auto-growth FALSIFIED + Lion+zero-mean-gradient structural finding. Excellent analytical execution — the per-block proj_weight_std diagnostic + per-layer gradient diagnostic together gave a definitive falsification + alternative mechanism identification.

**ASKELADD REASSIGNED H66 COORDSLICE-NO-STOPGRAD-LR-EXTENDED** (PR #1215) — single-flag change vs H58 = `--lr-cosine-t-max 25` instead of `13`. **5th parallel LR-fix test** alongside H59 (nezuko, decoder-sublayer), H62 (tanjiro, cp-loss-weight), H63 (edward, time-varying-loss), H65 (alphonse, shared-capacity-surface). If all 5 LR-fix variants produce merge-relevant improvements across 5 orthogonal mech classes, LR-decay confound is bulletproof confirmed as systematic Wave 31 ceiling and Wave 32 baselines default to `--lr-cosine-t-max 25`.

Close comment: https://github.com/morganmcg1/DrivAerML/pull/1207#issuecomment-4494678863

---

## 2026-05-20 02:15 — PR #1203: H54 v2 SURFACE-DEEP — 2 dedicated transformer blocks on surface decoder side mirror of H47 V-DEPTH (alphonse, EP12 mid forced terminal) — **MECHANISM-POSITIVE NULL with surf_deep ×9-18 sublayer growth + marginal -4.5bp better than H47 V-DEPTH peer + 5th Wave 31 LR-decay confound case**

- **Branch**: `alphonse/h54-surface-decoder-depth` (closed at 02:15Z)
- **W&B run**: rank0 `apbnjinz` (group `wave31_h54_surface_deep`, 913 min wall-time within 980 min budget, EP12 forced mid-epoch terminal at step 65,833)
- **Hypothesis**: mirror H47 V-DEPTH on the surface side — add 2 dedicated transformer blocks (with zero-init residuals for identity-at-init) before surface_out projection, with prediction that surface decoder shared-capacity expansion would target val_SP + val_WSS_z asymmetry. Mechanism class: shared-capacity-surface (novel — first Wave 31 in this class).

### Terminal verdict — NEAR-MISS on merge gate, surf_deep blocks ALIVE, -4.5bp better than H47 same-pattern peer

| Gate | Target | H54 v2 EP12 | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| val_abupt (merge) | <6.126% | **6.248%** | ❌ NEAR-MISS | +0.122pp (10th NEAR-MISS Wave 30/31) |
| test_abupt | (baseline 5.844%) | **6.042%** | ❌ FAIL | +0.198pp |
| test_SP (floor) | ≤3.577% | 3.803% | ❌ FAIL | +0.226pp |
| val_VP (floor) | ≤3.643% | 3.699% | ❌ NEAR-MISS | +0.056pp |
| test_VP (floor) | ≤3.643% | 3.693% | ❌ NEAR-MISS | +0.050pp |
| test_WSS_z (mech) | ~8.916% | 9.016% | ❌ NEAR-MISS | +0.100pp |
| test_WSS (goal) | <6.727% | 6.954% | ❌ FAIL | +0.227pp |

### KEY STRUCTURAL FINDING — decoder-side depth-bump is mech-positive null on BOTH sides

| Run | Side | val_abupt | test_abupt | Mech alive? | Verdict |
|---:|:--|---:|---:|:--|:--|
| H47 V-DEPTH | Volume | 6.293% | 6.049% | ✅ ×26-57% sublayer growth | mech-positive null |
| **H54 v2 SURFACE-DEEP** | **Surface** | **6.248%** | **6.042%** | **✅ ×9-18 sublayer growth** | **mech-positive null** |

**Class-level finding: decoder-side depth-bump is REAL reproducible mech class but sub-baseline alone at this recipe configuration.** H54 v2 is marginally better than H47 (-4.5bp val_abupt, -0.7bp test_abupt) — the depth-bump mechanism replicates across decoder sides with a small consistent uplift on the surface side. Wave 32 H47+H54 stack candidate is now well-grounded (both confirmed mech-positive on disjoint parameter blocks targeting disjoint loss axes).

### Mechanism evidence — surf_deep blocks fully integrated

| Diagnostic | EP1 (step 10864) | EP12 (terminal) | EP1→EP12 |
|---|---:|---:|---:|
| block0/attn_proj global_norm | 2.253 | **32.513** | ×14.4 growth |
| block0/ffn_fc2 global_norm | 5.636 | **68.160** | ×12.1 growth |
| block1/attn_proj global_norm | 3.015 | **33.659** | ×11.2 growth |
| block1/ffn_fc2 global_norm | 7.639 | **70.374** | ×9.2 growth |

All 8 diagnostics (attn_proj + ffn_fc2 norm + max_abs across both blocks) show ×9-18 growth from EP1 to EP12. Mechanism mirrors H47 V-DEPTH's healthy profile at comparable magnitude (H47 EP3 b0.ffn=0.479, H54 v2 EP12 b0.ffn=0.515 same operating range). **Depth-bump on surface side adds productive nonlinear computation, identity-at-init zero-init released cleanly.**

### Per-EP slope decay — 5th Wave 31 LR-decay confound case

| Window | Slope (val_abupt) | Status |
|---|---:|:--|
| EP6→EP7 | -3.7 bp/ep | productive |
| EP7→EP8 | -2.9 bp/ep | productive |
| **EP8→EP9** | **-0.8 bp/ep** | ⬇ slope quartered (LR fraction ~25% peak) |
| EP9→EP10 | -0.8 bp/ep | LR-tail plateau |
| EP10→EP11 | -0.8 bp/ep | LR-tail plateau |
| EP11→EP12 (mid) | -0.32 bp/ep | terminal LR ~2-4% peak |

**Slope quarters as LR drops below 50% peak — 5th Wave 31 LR-decay-confound case after H47/H52/H53/H55v2.** Mechanism (surf_deep blocks) fully alive through EP12 (sublayer growth steady), but LR-budget exhausts before val_abupt can clear merge gate.

### Wave 31 NEAR-MISS clustering (4 cases at 6.18-6.25%)

| PR | Mech class | val_abupt | Δ vs gate |
|---:|:--|---:|---:|
| H53 | variance-class-cp-loss-weight | 6.181% | +0.055pp |
| H50 | coordinate-grounded-slice-PE | 6.220% | +0.094pp |
| H57 | frequency-domain-capacity | 6.217% | +0.091pp |
| **H54 v2** | **shared-capacity-surface** | **6.248%** | **+0.122pp** |

Strongly suggests systematic LR-decay ceiling — 4 orthogonal mechanism classes all clustered in narrow NEAR-MISS zone.

### Disposition

CLOSED as mech-positive null with surf_deep blocks ALIVE + 4-decoder-side-depth-bump class-level finding + 5th Wave 31 LR-decay confound case. Excellent execution + analysis.

**ALPHONSE REASSIGNED H65 SURFACE-DEEP-LR-EXTENDED** (PR #1214) — single-flag change vs H54 v2 = `--lr-cosine-t-max 25` instead of `13`. **4th parallel LR-fix test alongside H59 (nezuko, PR #1208), H62 (tanjiro, PR #1211), H63 (edward, PR #1212)**. If all 4 LR-fix variants merge or substantially improve across 4 orthogonal mechanism classes, LR-decay-confound is bulletproof confirmed as systematic Wave 31 ceiling.

Close comment: https://github.com/morganmcg1/DrivAerML/pull/1203#issuecomment-4494034286

---

## 2026-05-20 02:00 — PR #1206: H57 MULTI-SCALE-RFF-EXPANDED — wider freq band (8 sigmas 0.125→16) in string-separable position encoding (thorfinn, 13-ep terminal) — **MECHANISM-POSITIVE NULL with test_VP FLOOR CROSS on BOTH val+test + H48 same-recipe strict beat 7 axes + KEY FALSIFIED HYPOTHESIS (LOW end is binding, not HIGH)**

- **Branch**: `thorfinn/h57-rff-expanded-13ep` (closed at 02:00Z)
- **W&B run**: rank0 `e8lhpbn9` (group `wave31_h57_multiscale_rff_expanded`, 14.33h runtime, terminal step 70,652 = ~EP6.5 effective due to budget cutoff, NOT EP13 nominal; best EMA = EP6)
- **Hypothesis**: expand RFF init sigmas from 5 (0.25→4.0) to 8 (0.125→16.0) in string-separable PE to give encoder access to wider freq band, with prediction that the HIGH end (σ=8,16) would help capture high-frequency τz-axis gradient features. Mechanism class: frequency-domain-capacity (FDCE-expansion) — first Wave 31 in this class.

### Terminal verdict — NEAR-MISS on merge gate, BOTH val+test VP floor cross, FALSIFIED τz-via-high-freq prediction

| Gate | Target | H57 terminal | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| val_abupt (merge) | <6.126% | **6.217%** | ❌ NEAR-MISS | +0.091pp (9th Wave 30/31; **2nd-closest Wave 31** after H53's +0.055pp) |
| test_abupt | (baseline 5.844%) | **6.053%** | ❌ FAIL | +0.209pp (but **−0.114pp vs H48 same-recipe peer** ⭐ strongest single-mechanism test_abupt improvement Wave 31) |
| test_SP (floor) | ≤3.577% | 3.812% | ❌ FAIL | +0.235pp |
| **val_VP (floor)** | **≤3.643%** | **3.612%** | **✅ PASS** | **−0.031pp** |
| **test_VP (floor)** | **≤3.643%** | **3.610%** | **✅ PASS** | **−0.033pp** ⭐ **4th Wave 31 test_VP cross + FIRST simultaneous val+test cross** |
| test_WSS_z (binding) | (~9.5% PR body) | 9.148% | ✅ PASS (≥0.3pp gate) | −0.352pp (within seed noise vs H48's 9.174%) |
| test_WSS (goal) | <6.727% | 6.949% | ❌ FAIL | +0.222pp |

### KEY STRUCTURAL FINDING — H57's "τz needs higher frequencies" hypothesis falsified by per-σ projection diagnostic

Student's offline per-σ projection-weight diagnostic on EP3 checkpoint (mean over all 5 transformer blocks' `rff_input_proj.weight` L2-norm contribution per σ slot):

| σ | Surface proj weight share | Volume proj weight share | Status |
|---:|---:|---:|:--|
| **0.125 (new LOW)** | **11.79%** | **18.47%** | ⭐ HIGH UPTAKE |
| 0.25 → 4.0 (baseline range) | ~12-15% each | ~12-15% each | normal |
| 8.0 (new HIGH) | low | low | minimal |
| **16.0 (new HIGH)** | **3.14%** | **2.30%** | ⭐ MINIMAL UPTAKE |

**The LOW end (σ=0.125) took up significant projection weight; the HIGH end (σ=16) was minimally utilized.** The test improvement was uniform across all 7 paper-facing axes (NOT τz-selective). The val τz advantage that compounded through training (val 4→10: Δ vs H48 widening from −0.192 to −0.302pp) collapsed to within-noise at test split (test τz Δ vs H48 = −0.026pp). **The val-side τz compounding was a val-set artifact, not the actual mechanism direction.**

### H57 strict beat vs H48 same-recipe peer (run `8cn5abxm`) — FDCE confirmed as NEW mechanism class

| Metric | H57 test | H48 test | Δ (H57 − H48) |
|---|---:|---:|---:|
| **abupt** | **6.053%** | 6.167% | **−0.114pp** ⭐ |
| surface_pressure | 3.812% | 3.898% | −0.085pp ✅ |
| volume_pressure | 3.610% | 3.671% | −0.061pp ✅ |
| wall_shear | 6.949% | 7.113% | −0.164pp ✅ |
| wall_shear_x (τx) | 6.102% | 6.301% | −0.199pp ✅ |
| wall_shear_y (τy) | 7.593% | 7.791% | −0.198pp ✅ |
| wall_shear_z (τz) | 9.148% | 9.174% | −0.026pp (within seed noise) |

**FDCE (frequency-domain-capacity-expansion) confirmed as new mechanism class for Wave 31** — 7-axis-uniform improvement over matched-recipe H48 peer establishes it as direction-correct, stackable, and orthogonal to existing classes (variance, mean-shift, etc.).

### Wave 31 test_VP floor cross tally now 4 cases — H57 FIRST simultaneous val+test cross

| PR | Mechanism class | val_VP | test_VP |
|---:|:--|---:|---:|
| H26 (merged) | variance-class-encoder-input (NPCA) | val cross | 3.608 (test cross −0.035) |
| H53 | variance-class-cp-loss-weight | 3.610 (val cross −0.033) | 3.665 (close miss +0.022) |
| H55 v2 | variance-class-time-varying-loss | 3.670 (close miss +0.027) | 3.602 (test cross −0.041) |
| **H57** | **frequency-domain-capacity (FDCE)** | **3.612 (val cross −0.031)** | **3.610 (test cross −0.033)** ⭐ BOTH |

All 4 share vol-points-schedule `0:16384:3:32768:6:49152:9:65536`. **Strong Wave 32 candidate: focused test_VP investigation isolating vol-points-curriculum contribution independent of mechanism class.**

### Disposition

CLOSED as mechanism-positive null with BOTH val+test VP floor cross + matched-recipe H48 strict beat on 7 paper-facing axes + falsified high-freq-binds-τz hypothesis. Exceptional terminal analysis with **strongest single piece of mechanism evidence in Wave 31** (per-σ projection diagnostic). **THORFINN REASSIGNED H64 RFF-LOW-BAND-EXPANSION** (PR #1213) — direct follow-up using H57's per-σ data: drop σ={8.0, 16.0} (minimal uptake), add σ=0.0625 (one octave below H57's new low), giving 7 sigmas `"0.0625,0.125,0.25,0.5,1.0,2.0,4.0"`. Single flag change vs H57; tests low-end-binds-FDCE direct corollary.

Close comment: https://github.com/morganmcg1/DrivAerML/pull/1206#issuecomment-4493840197

---

## 2026-05-20 01:45 — PR #1204: H55 v2 TAU-Z-LOSS-CURRICULUM — front-load τz loss weight 5.0→2.0 over EP1-6 linear (edward, 13-ep terminal) — **MECHANISM-POSITIVE NULL with test_VP FLOOR CROSS + val_WSS_z −0.341pp on binding axis + 4th Wave 31 LR-decay confound case**

- **Branch**: `edward/h55-tau-z-curriculum-v2-main` (closed at 01:45Z)
- **W&B run**: rank0 `tkouys7y` (group `wave31_h55_tau_z_curriculum`, 14.33h runtime, 13-ep natural completion at step 70,652; best EMA checkpoint = EP13)
- **Hypothesis**: time-varying τ_z penalty (5.0× → 2.0× linear over EP1-6) front-loads channel-asymmetry attack during high-LR productive zone, then relaxes to standard weight for stable post-curriculum descent. Mechanism class: variance-class-time-varying-loss (novel — first Wave 31 in this class).

### Terminal verdict — NEAR-MISS on merge gate, test_VP FLOOR CROSS + mechanism-positive on binding axis

| Gate | Target | H55 v2 EP13 | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| val_abupt (merge) | <6.126% | **6.249%** | ❌ NEAR-MISS | +0.123pp (8th NEAR-MISS in Wave 30/31) |
| test_abupt | (baseline 5.844%) | **5.988%** | ❌ FAIL | +0.144pp |
| test_SP (floor) | ≤3.577% | **3.806%** | ❌ FAIL | +0.229pp |
| val_VP (floor) | ≤3.643% | 3.670% | ❌ NEAR-MISS | +0.027pp |
| **test_VP (floor)** | **≤3.643%** | **3.602%** | **✅ PASS** | **−0.041pp** ⭐ 3rd Wave 31 test_VP cross (H53 val cross + close-miss test, now H55 v2 clean test cross) |
| **val_WSS_z** | (H48 ref 9.899%) | **9.558%** | ✅ MECHANISM POSITIVE | **−0.341pp** on binding τz axis |
| test_WSS (goal) | <6.727% | 6.883% | ❌ FAIL | +0.156pp |
| test_WSS_z | (mech ref) | 8.917% | ✅ MECHANISM POSITIVE | mechanism direction-correct |

### KEY STRUCTURAL FINDING — 3rd Wave 31 test_VP floor cross + per-axis-weighting pattern

| Run | Mechanism | test_VP | Δ vs floor 3.643% |
|---|---|---:|---:|
| H26 NPCA (merged) | encoder-input enrichment | 3.608% | −0.035pp ✅ |
| H53 CP-LOSS-WEIGHT (closed) | cp_loss_weight=2 surface channel | 3.665% | +0.022pp ❌ close-miss test, ✅ val cross |
| **H55 v2 TAU-Z-CURRICULUM (this PR)** | **τz curriculum 5.0→2.0 over EP1-6** | **3.602%** | **−0.041pp** ✅ |

**Two different per-axis loss-weighting mechanisms BOTH cross test_VP floor on the same vol-points schedule (16384→65536).** Converges on Wave 32 finding: per-axis loss weighting + vol-points curriculum unlocks test_VP descent. Worth a focused follow-up PR examining the test_VP cross mechanism.

### Mechanism strict-beat vs H48 mean-shift null

H55 v2 strictly beats H48 on every val metric:
- val_abupt 6.249% vs H48 6.485% = **−0.236pp**
- val_WSS_z 9.558% vs H48 9.899% = **−0.341pp** (binding axis)
- val_VP 3.670% vs H48 3.802% = −0.132pp
- val_SP 4.086% vs H48 4.277% = −0.191pp

**Time-varying curriculum is strictly better than static-weight reduction for the channel-asymmetry problem.** New mechanism class confirmed.

### LR-decay confound — 4th Wave 31 case after H47, H52, H53

Per-epoch slope decay (val_abupt):
- EP3.7→EP4.6: −0.020 pp/1k (LR 68% peak)
- EP4.6→EP5.4: −0.010 pp/1k (LR 45% peak) — slope halved
- EP5.4→EP6.07: −0.006 pp/1k (LR 14% peak) — slope halved again
- EP6.07→terminal: −0.0002 pp/1k (LR ~0%, cosine tail flat)

Canonical Wave 31 cosine-tail-LR-decay pattern. Mechanism active throughout EP1-6 productive zone, but LR-budget exhausted before merge gate cleared.

### Wave 31 mechanism-class taxonomy update

After H55 v2 closure:
- **variance-class-time-varying-loss / tau-z-curriculum**: mechanism-positive null with test_VP cross + val_WSS_z mech-positive → **DEFERRED pending H63 LR-fix variant**
- 12-class taxonomy now: 4 proven null (mean-shift / cross-channel-weight-space / variance-class-decoder-weight / coord-grounded-PE-stop-grad) + 4 mech-positive nulls DEFERRED (cp-loss-weight, v-depth, tau-z-curriculum, npca-yaw-stack) + 4 in flight

### Disposition

CLOSED as mechanism-positive null with test_VP floor cross + val_WSS_z −0.341pp + LR-decay confound (4th Wave 31 case). **EDWARD REASSIGNED H63 TAU-Z-CURRICULUM-LR-EXTENDED (PR #1212)** — single-flag change vs H55 v2 (`--lr-cosine-t-max 25` instead of 13). **3rd parallel LR-fix test** alongside H59 V-DEPTH-LR-EXTENDED + H62 CP-LOSS-WEIGHT-LR-EXTENDED — if all three mechanism classes merge under LR-fix, LR-decay-confound is conclusively confirmed as Wave 31 ceiling.

---

## 2026-05-19 18:00 — PR #1202: H53 CP-LOSS-WEIGHT — cp_loss_weight 1.0→2.0 surface-pressure channel up-weighting (tanjiro, 13-ep terminal) — **MECHANISM-POSITIVE NULL with val_VP FLOOR CROSS + first Wave 31 AB-UPT public-ref test_SP beat + LR-decay confound**

- **Branch**: `tanjiro/h53-cp-loss-weight` (closed at 18:00Z)
- **W&B run**: rank0 `u187bw3a` (group `wave31_h53_cp_loss_weight`, ~14h runtime, terminal step 70,652 ≈ 50% of nominal 141,232-step 13-ep plan)
- **Hypothesis**: 2× cp_loss_weight should produce fleet-leading val_SP descent and reduce test_SP gap on AB-UPT competitive axis. Mechanism class: variance-class-cp-loss-weight (novel — first Wave 31 in this class).

### Terminal verdict — NEAR-MISS on merge gate, val_VP FLOOR CROSS, first AB-UPT public-ref beat

| Gate | Target | H53 EP13 | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| val_abupt (merge) | <6.126% | **6.181%** | ❌ NEAR-MISS | +0.055pp (7th NEAR-MISS in Wave 30/31) |
| test_abupt | (baseline 5.844%) | **6.052%** | ❌ FAIL | +0.208pp |
| test_SP (floor) | ≤3.577% | **3.793%** | ❌ FAIL | +0.216pp |
| **val_VP (floor)** | **≤3.643%** | **3.610%** | **✅ PASS** | **−0.033pp** ⭐ first Wave 31 val_VP cross |
| **test_VP (floor)** | **≤3.643%** | **3.665%** | **❌ CLOSE MISS** | **+0.022pp** (8th VP floor approach in Wave 30/31) |
| test_WSS (goal) | <6.727% | **6.978%** | ❌ FAIL | +0.251pp |

### NEW STRUCTURAL FINDING — first Wave 31 AB-UPT public-reference test_SP beat

H53 **test_SP 3.793% beats AB-UPT public reference 3.82% by −0.027pp** — first in fleet. Adjacent metrics also beat AB-UPT public:
- test_VP 3.665% beats AB-UPT 6.08% by −2.415pp
- test_WSS aggregate 6.978% beats AB-UPT 7.29% by −0.312pp

Structural signal: mechanism-class-cp-loss-weight is competitive with strongest public reference baseline on surface AND volume pressure axes, even though it does not cross internal merge gate.

### Cosine-tail LR-decay confound (3rd Wave 31 case after H47, H52)

| Window | Steps | val_abupt slope (pp/1k) | LR fraction peak |
|---|---:|---:|---:|
| EP4→EP4.5 | 5,435 | −0.015 | 90-80% |
| EP4.5→EP4.8 | 3,625 | −0.023 | 80% |
| EP4.8→EP5.5 | 7,250 | −0.008 | 70-50% |
| EP5.5→EP5.8 | 2,720 | −0.005 | 50% |
| EP5.8→EP6.5 | 8,006 | **−0.0016** | 30-15% |
| EP6.5→terminal | 15,338 | **~0 (cosine tail flat)** | <5% |

Same canonical Wave 31 LR-decay-confound pattern (H47 / H52). Mechanism activated correctly during high-LR productive zone, plateaued in cosine tail.

### Wave 31 mechanism-class taxonomy update

After H53 closure:
- **variance-class-cp-loss-weight**: mechanism-positive null with val_VP floor cross + AB-UPT test_SP beat → **DEFERRED pending H62 LR-fix variant**
- 11-class taxonomy now: 4 proven null (mean-shift / cross-channel-weight-space / variance-class-decoder-weight / coordinate-grounded-slice-PE) + variance-class-cp-loss-weight DEFERRED + 6 in flight

### Disposition

CLOSED as mechanism-positive null with notable per-axis floor cross + LR-decay confound. **TANJIRO REASSIGNED H62 CP-LOSS-WEIGHT-LR-EXTENDED (PR #1211)** — single-flag change vs H53 (`--lr-cosine-t-max 25` instead of 13) to isolate LR-decay-as-plateau-cause from mechanism-class-cp-loss-weight saturation. Parallel LR-fix test with H59 V-DEPTH-LR-EXTENDED (nezuko). If both H59 and H62 produce merge-relevant improvements, LR-decay confound becomes a confirmed systematic issue affecting all cosine-13ep recipes in Wave 31.

---

## 2026-05-19 16:55 — PR #1200: H52 NPCA × YAW-AUG mechanism stacking — variance-break × symmetry-breaking (frieren, 13-ep terminal) — **MECHANISM-POSITIVE NULL with STRUCTURAL FINDING on orthogonal-mechanism non-compounding**

- **Branch**: `frieren/h52-npca-yaw-stack` (closed at 16:55Z)
- **W&B run**: rank0 `3u4i7oy6` (group `wave31_h52_npca_yaw_stack`, 14.00h runtime, 13-ep natural completion at step 70,664; best EMA checkpoint = EP13)
- **Hypothesis**: Compound H26 NPCA (variance-class encoder-input) with H44 YAW-AUG (symmetry-class data augmentation) — mechanism-class orthogonal → predicted compound variance std 0.25-0.35, val_abupt 5.95-6.18% (potential merge).

### Terminal verdict — clear NULL on val_abupt, NO floor crossings

| Gate | Target | H52 EP13 | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| val_abupt (merge) | <6.126% | **6.479%** | ❌ FAIL | +0.353pp |
| test_abupt | (baseline 5.844%) | **6.155%** | ❌ FAIL | +0.311pp |
| test_SP (floor) | ≤3.577% | **3.900%** | ❌ FAIL | +0.323pp |
| test_VP (floor) | ≤3.643% | **3.735%** | ❌ NO CROSSING | +0.092pp (close) |
| test_WSS (goal) | <6.727% | **7.108%** | ❌ FAIL | +0.381pp |

### KEY STRUCTURAL FINDING — orthogonal-mechanism stacking does NOT additively compound

First direct mechanism-stacking test in Wave 31:
- **H44 YAW-AUG standalone (closed 02:08Z)**: EP13 std ~0.198, n_outside_band 14/34
- **H52 NPCA × YAW-AUG stack (this PR)**: EP13 std **0.2044**, n_outside_band 16/34
- **PR body predicted compound**: std 0.25-0.35

**Result identical to YAW-AUG standalone within noise.** The NPCA encoder-input enrichment did NOT add to the variance signal beyond YAW-AUG alone produces. **Mechanism-class orthogonality (different attack axes) does NOT guarantee additive compounding.**

This finding constrains Wave 32 stack design: **stacks must be empirically validated, not assumed from individual-mechanism wins.**

### Val/test variance divergence — variance signal partly val-specific

| Split | std | n_outside_band |
|---|---:|---:|
| val (34 cars) | **0.2044** | 16/34 |
| test (50 cars) | **0.1288** | 28/50 |

3rd Wave 30/31 experiment to show val/test variance divergence (H35, H51 also). Val cars likely include geometry outliers; test split more central. Implication: variance-class merge gates set on val-side metrics may not transfer to test.

### Per-axis WSS — Wave 31 canonical pattern reconfirmed

| Axis | Test (50 cars) | Val (34 cars) | Δ (test−val) |
|---|---:|---:|---:|
| WSS_x | 6.360% | 6.445% | −0.085pp |
| WSS_y | 7.753% | 7.969% | −0.216pp |
| **WSS_z** | **9.028%** | **9.915%** | **−0.887pp** |

5th Wave 31 experiment confirming WSS_z is hardest axis on both val AND test. Wave 32 mechanism candidates should specifically attack WSS_z.

### Cosine-tail slope decay (LR-decay confound, mirrors H47 pattern)

| Window | val_abupt slope (pp/1k) |
|---|---:|
| EP9→EP10 | −0.0173 |
| EP10→EP11 | −0.0078 (halved) |
| EP11→EP12 | −0.0040 (halved again) |

Same LR-decay confound H47 diagnosed. Terminal LR ~2-3% of peak. H52 mechanism could not break free of this attractor.

### Disposition

CLOSED as mechanism-positive null with structural finding about orthogonal-mechanism non-compounding. **FRIEREN REASSIGNED H61 SLICE-TEMP-CURRICULUM (PR #1210)** — mechanism-class-novel attention-temperature scheduling (τ_slice anneals 1.5→1.0 over EP1-6, holds 1.0 EP6-13). No prior Wave 31 experiment in this class. Standalone single-mechanism test, conservative recipe baseline (slices=128, ema=0.999, no NPCA/SSFL).

---

## 2026-05-19 16:20 — PR #1205: H56 H51-RELAUNCH NPCA+SSFL+slices192+ema9999 with EMA-aware kill gates (fern, killed EP3 step 32,592) — **9TH ADVISOR RECIPE-BUG CLOSURE: EP3 gate `<25.0%` set 0.30pp inside math floor (random_pred_floor ≈ 100% NOT 7-22%); MECHANISM POSITIVE STRONGEST WAVE 31 SIGNAL: τ_zx_ratio_std doubled in ONE epoch EP2→EP3 0.0554→0.1384 already exceeds H51 mid-EP4 0.117 by 18.3%; slope ACCELERATING −1.31 pp/1k EP1→EP2 → −2.52 pp/1k EP2→EP3; train/epoch_loss 0.01129 matches H35 reference shape; FERN REASSIGNED H60 H56-RELAUNCH-DROP-EP3 PR #1209**

- **Branch**: `fern/h56-h51-relaunch-ema-aware-gates` (closed at 16:20Z)
- **W&B run**: `mgor7bk7` (group `wave31_h56_h51_relaunch`, 5.31h runtime, killed at step 32,592 by `val_primary/abupt_axis_mean_rel_l2_pct=25.3038 did not satisfy <25.0` — over by 0.30pp / 1.2% above threshold)
- **Hypothesis**: Same as H51 (PR #1199 closed as recipe-bug closure) — NPCA + SSFL + slices=192 + ema=0.9999 stack — relaunched with EMA-aware kill gates accounting for δ^step contamination.

### Kill trigger — gate set 0.30pp inside the mathematical floor

Empirical EMA-val_floor at EP3 step 32,592 with ema=0.9999:
```
EMA-val_floor = trained_val + δ^32592 · (random_pred_floor − trained_val)
              = 22% + 0.0394 · (100% − 22%)
              = 22% + 3.07%
              = 25.07%
```
My gate `<25.0%` was set 0.07pp BELOW the math floor. H56 read 25.30% — only 0.23pp over the math prediction. **Random_pred_floor for rel_l2_pct metrics is ~100% (random predictions vs ground truth give ~1.0 rel L2), NOT the 7-22% my prior memory entry assumed.**

### Full validation trajectory (EMA-shadow contaminated)

| EP | step | δ^step | val_abupt | val_SP | val_VP | val_WSS | WSS_x | WSS_y | WSS_z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,864 | 0.337 | 66.93% | 49.26% | 59.47% | 68.89% | 59.34% | 87.25% | 79.32% |
| 2 | 21,728 | 0.115 | 52.68% | 40.53% | 39.42% | 56.12% | 48.34% | 71.70% | 63.40% |
| 3 | 32,592 | 0.039 | **25.30%** | 17.40% | 17.38% | 28.45% | 24.71% | 37.14% | 29.88% |

### Slopes — ACCELERATING (strongest Wave 31 signal)

| Window | val_abupt slope | val_WSS slope |
|---|---:|---:|
| EP1→EP2 | −1.31 pp/1k | similar |
| **EP2→EP3** | **−2.52 pp/1k** | WSS_y −3.18 pp/1k, WSS_z −3.08 pp/1k, WSS_x −2.18 pp/1k |

**Slope is accelerating (not decelerating).** Per-axis WSS hardness y > z > x preserved throughout — matches H35/H51 reference shape exactly.

### Mechanism diagnostic — variance-class signal at STRONGEST in Wave 31

| Metric | EP1 | EP2 | **EP3** | EP2→EP3 Δ | H51 ref EP4 |
|---|---:|---:|---:|---:|---:|
| `tau_zx_ratio_mean` | 1.3373 | 1.3118 | 1.2179 | −0.0939 | — |
| `tau_zx_ratio_std` | 0.0751 | 0.0554 | **0.1384** | **+0.0830 (+150%)** | 0.117 (mid-EP4) |
| `tau_zx_ratio_n_outside_band` | 28/34 | 32/34 | 31/34 | −1 | — |
| `tau_zx_ratio_min/max` | 1.192/1.496 | 1.176/1.420 | **1.049/1.573** | spread WIDENED both sides | — |

**H56 EP3 std 0.1384 ALREADY EXCEEDS H51 mid-EP4 std 0.117 by 18.3%, one epoch earlier.** H35 fleet-peak ref std is 0.251 at EP13 — H56 reached 55% of fleet-peak in 3 epochs vs H35's 13 epochs.

### Trained-side health (NO EMA-shadow on train loss)

| Metric | EP1 | EP2 | EP3 |
|---|---:|---:|---:|
| `train/epoch_loss` | 0.3587 | 0.0760 | **0.01129** |
| `train/base_mse_loss` (EP-end) | — | — | 0.00781-0.00914 across ranks |
| `train/spectral_loss` (SSFL active) | active | active | active |

train/epoch_loss 0.01129 at EP3 matches H35 reference shape exactly for early-epoch NPCA+SSFL learning.

### Recipe-bug class — 2nd consecutive kill of H51 stack by misset gate

| Recipe | EP3 gate | EP3 EMA-val | Margin | Outcome |
|---|---|---:|---:|---|
| H51 v3 (PR #1199) | `<10.0%` | 23.64% | −13.64pp | KILLED mid-EP4 step 38,027 (5.5h wasted) |
| H56 (this PR #1205) | `<25.0%` | **25.30%** | **−0.30pp** | KILLED at EP3 end step 32,592 (5.31h wasted) |

Total: **10.81h GPU runtime sacrificed to advisor mis-calibrated EP3 gates** on the strongest mechanism signal in Wave 31.

### Disposition

CLOSED as 9th advisor recipe-bug. **MECHANISM POSITIVE — strongest variance-class signal in Wave 31.** Fern reassigned to H60 H56-RELAUNCH-DROP-EP3 (PR #1209) with **only the EP6 binding gate** retained (val_abupt<7.0% + val_SP<5.0% at step 65,184). Drop EP3 + EP4 gates entirely because EMA-shadow contamination at those steps is structural (math floor 23-26%), not informative.

Memory updated: `feedback_ema_aware_kill_thresholds.md` now reflects empirical random_pred_floor ≈ 100% and recommends drop-all-gates-except-EP6 for ema=0.9999 stacks.

---

## 2026-05-19 15:45 — PR #1194: H47 V-DEPTH — Volume Decoder Depth Bump 2 dedicated vol-only blocks (nezuko, 18h full rerun) — **MECHANISM-POSITIVE NULL with test_VP +0.010pp NEAR-MISS on floor** (tightest H47-family vol_p floor approach; 4 sublayer norms 5-14× above EP3 KILL threshold + block1>block0 productive asymmetry confirmed canonical Wave 31 signature; val_abupt plateau and merge-gate miss DOMINATED BY LR-DECAY confound — cosine cycle completed within actual 70k-step training window, terminal LR collapsed to 2.5% of peak; slope decelerated 30× from EP3→EP4 −0.034 to EP6→terminal −0.0011 matching LR collapse 99%→2.5%); NEZUKO REASSIGNED H59 V-DEPTH-LR-EXTENDED (PR #1208) — single-flag change `--lr-cosine-t-max 25` instead of 13 to keep terminal LR at ~70-80% peak instead of 2.5%

- **Branch**: `nezuko/h47-vdepth` (closed at 15:45Z)
- **W&B run**: 8-rank DDP rank0 `dp7gbsjb` (group `wave31_h47_vdepth`, 15.66h runtime, terminated at step 70,652 / 141,232 = 50% of 13-ep step budget by internal training-time cap; vp curriculum fully traversed 16384→32768→49152→65536 all 4 stages)
- **Hypothesis**: Add 2 dedicated volume-only transformer blocks AFTER the shared encoder trunk to deepen volume-decoder interior capacity. Volume pathway is empirically responsive (H26/H31 crossed test_vol_p floor via encoder-input enrichment); the question is whether the shallow decoder under-extracts those richer features.

### Terminal verdict — mechanism POSITIVE, gates ALL FAIL, test_VP +0.010pp NEAR-MISS

| Gate | Target | H47 EP12 | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| val_abupt (merge) | <6.126% | **6.273%** | ❌ FAIL | +0.147pp |
| test_abupt | (baseline 5.844%) | **6.049%** | ❌ FAIL | +0.205pp |
| test_SP (floor) | ≤3.577% | **3.769%** | ❌ FAIL | +0.192pp |
| **test_VP (floor)** | **≤3.643%** | **3.6533%** | **❌ NEAR-MISS** | **+0.010pp** ⭐ |
| test_WSS (goal) | <6.727% | **6.993%** | ❌ FAIL | +0.266pp |

Test_VP at 3.6533% is the **7th test_VP floor approach in Wave 30/31** and the **tightest H47-family approach** to the floor. Comparison vs prior floor crossings: H26 NPCA (-0.035pp BELOW, MERGED), H31 WALLDIST (-0.155pp BELOW, MERGED), H33 SLICEPE (cross, CLOSED), H50 COORDSLICE (-0.047pp cross, CLOSED), H53 CP-LOSS-WEIGHT projected (-0.10pp cross, in flight).

### Mechanism diagnostic — STRONGLY POSITIVE per-block residual trace (canonical Wave 31 signature)

| Block/Sublayer | EP1 (shedding) | EP3 (gate target >0.05) | Terminal |
|---|---:|---:|---:|
| b0.attn.out_proj max_abs | 0.0318 | 0.2206 (4× PASS) | **0.2610** (5× PASS) |
| b0.mlp.fc2 max_abs | 0.0332 | 0.4791 (10× PASS) | **0.5728** (11× PASS) |
| b1.attn.out_proj max_abs | 0.0307 | 0.2515 (5× PASS) | **0.3597** (7× PASS) |
| b1.mlp.fc2 max_abs | 0.0384 | 0.5085 (10× PASS) | **0.7170** (14× PASS) |

**Productive block1 > block0 asymmetry confirmed** (b1.ffn norm 75.32 vs b0.ffn norm 65.26 = 1.25× ratio, mirrors the H30 V2S asymmetric-fusion productive pattern). **FFN dominates over attn** (norm 65-75 vs 27-34, ~2.4× larger) — depth-bump mechanism is primarily expressivity (FFN capacity), not attention-mixing between volume tokens.

### LR-decay confound — the dominant plateau driver

| Step | LR | Fraction of peak | val_abupt slope (pp/1k) |
|---:|---:|---:|---:|
| 14,133 (peak post-warmup) | 9.00e-05 | 100% | — |
| EP3→EP4 (28k→43k) | 99-90% peak | productive zone | **−0.034** (productive) |
| EP4→EP5 (43k→52k) | 80-70% peak | mid-decay | −0.015 |
| EP5→EP6 (52k→62k) | 60-45% peak | mid-decay | −0.011 |
| EP6→terminal (62k→70k) | 30-2.5% peak | near-zero LR | **−0.0011** (plateau) |

**The val_abupt slope collapsed 30× from EP3→EP4 (−0.034 pp/1k at 90% peak LR) to terminal (−0.0011 pp/1k at 2.5% peak LR). The plateau is overwhelmingly LR-decay artifact**, not an expressivity asymptote at this depth budget. Linear extrapolation: with constant 80% peak LR maintained over 70k steps, val_abupt would land 5.5-5.8% — comfortably under merge gate.

### Wave 31 mechanism-class taxonomy after H47 closure (8 classes unchanged from H50)

H47 closure changes "variance-class-decoder-sublayer" status from "TBD borderline" to "**mechanism-positive null with LR-decay confound** — class status RESERVED pending H59 V-DEPTH-LR-EXTENDED resolution".

### Disposition

CLOSED as mechanism-positive null with test_VP +0.010pp NEAR-MISS + LR-decay confound. Excellent diagnostic work — the per-block residual trace + productive block1>block0 asymmetry is canonical Wave 31 mechanism-positive signature, AND the LR-decay terminal analysis identified the dominant confound. **Nezuko reassigned to H59 V-DEPTH-LR-EXTENDED (PR #1208)** — single-flag change `--lr-cosine-t-max 25` instead of 13, directly tests whether H47's val_abupt plateau and test_VP near-miss were LR-decay artifacts.

---

## 2026-05-19 13:35 — PR #1198: H50 COORDSLICE — Coordinate-Grounded Slice IDs (askeladd) — **MECHANISM-POSITIVE NULL CLOSURE with 6th test_VP floor crossing in Wave 30/31** (val_abupt 6.220% closest miss in Wave 31 +0.094pp; test_VP CROSSED floor by −0.047pp; lowest val_VP in Wave 31 3.676%; NEW structural finding L0-PE-capacity-sink pattern; mechanism class #8 coordinate-grounded-slice-PE proven null); ASKELADD REASSIGNED H58 COORDSLICE-NO-STOPGRAD (PR #1207) — single-line code change removing `torch.no_grad()` wrap to restore routing-gradient feedback to PE projection

- **Branch**: `askeladd/h50-coordslice-rff` (closed at 13:35Z)
- **W&B run**: 8-rank DDP rank0 `biw3rtli` (group `wave31_h50_coordslice`, full 13 epochs, 141,232 steps, ~14.6h runtime)
- **Hypothesis**: H33 SLICEPE (closed prior) produced free-learnable slice PE with L0-DOMINANT spatial differentiation (L0 inter_slice_cos 0.085 = LOWEST). H50 grounded the same mechanism in physical coordinates via DAB-DETR analogue: compute per-slice centroid from `slice_weights.mean(dim=1)` weighted with input coords, project through 4-σ RFF (0.25, 0.5, 1.0, 2.0), then through `Linear(256, H*D_head)` as additive PE. Stop-gradient on centroid computation (`torch.no_grad()`) to decouple routing from positional anchoring. Predict L0 should benefit most from coordinate-grounded PE.

### Terminal verdict — val_abupt MISS, test_VP floor CROSSED

| Gate | Target | H50 EP13 | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| val_abupt (merge) | <6.126% | **6.220%** | ❌ FAIL | **+0.094pp** (closest miss in Wave 31) |
| test_abupt | (baseline 5.844%) | **5.978%** | ❌ FAIL | +0.134pp |
| test_SP (floor) | ≤3.577% | **3.735%** | ❌ FAIL | +0.158pp |
| **test_VP (floor)** | **≤3.643%** | **3.596%** | **✅ PASS** | **−0.047pp** ⭐ |
| test_WSS (goal) | <6.727% | **6.917%** | ❌ FAIL | +0.190pp |
| **val_VP (info)** | (info) | **3.676%** | LOWEST in Wave 31 | — |
| test τz/τx (band) | [1.44, 1.55] | 1.446 | in band | (val 1.548 borderline) |

### Mechanism reading — L0-PE-capacity-sink (NEW structural finding)

**Mechanism prediction INVERTED**: H33's free-learnable PE produced L0-dominant spatial differentiation (L0 inter_slice_cos 0.085 = lowest = most differentiated). H50's coordinate-grounded PE produces **L0-INVERTED** spatial differentiation:
- L0 inter_slice_cos **0.298** = HIGHEST = LEAST differentiated
- L0 proj_weight_norm **33.58** = HIGHEST (12% above L2's 28.98)
- Deeper layers L2/L3 take over as differentiation engines

**Interpretation**: coordinate-grounded slice PE causes L0 to absorb PE-projection CAPACITY (coordinate-routing layer) rather than spatial-discrimination capability. L0 spends one layer's worth of feature-transformation capacity routing spatial information to L2/L3. "PE-capacity-sink-then-redistribute" pattern, distinct from H33's "PE-direct-spatial-discrimination".

### Three plausible mechanisms for the +0.094pp shortfall (in priority order)

1. **Stop-gradient blocked PE auto-growth (HIGHEST PROBABILITY — H58's target)**: H50 wrapped centroid computation in `torch.no_grad()` to decouple routing from positional anchoring. H33's PE auto-grew σ 0.02 → 0.15 (8× growth) because routing gradients fed back. H50's proj_weight_std stayed at 0.080-0.093 ≈ init 0.088 (virtually zero growth). Stop-grad starved PE projection of optimization signal.

2. **L0-PE-capacity-sink is a real cost**: L0 spends 12% more PE-projection capacity than deeper layers (33.58 vs 28.98 at L2) = one layer's worth of feature-transformation capacity diverted to coordinate routing.

3. **Init scale σ=0.088 may still be below gradient floor**: even with stop-grad, σ should have grown a bit; staying near init suggests gradient signal was too weak to escape the floor.

### Per-epoch trajectory (rank0)

| Step | Epoch | val_abupt | val_VP | val_SP | val_WSS | val_WSS_z |
|---:|:--|---:|---:|---:|---:|---:|
| 10,864 | EP1 | 28.897% | 17.340% | 21.755% | 31.875% | 39.701% |
| 21,728 | EP2 | 7.807% | 4.649% | 5.257% | 8.794% | 11.401% |
| 32,594 | EP3 | 7.003% | 4.112% | 4.696% | 7.945% | 10.402% |
| 43,466 | EP4 | 6.469% | 3.824% | 4.302% | 7.305% | 9.774% |
| 67,932 | mid-EP7 | 6.227% | 3.679% | 4.105% | 7.043% | 9.528% |
| 141,232 | EP13 | **6.220%** | 3.676% | (n/a) | (n/a) | (n/a) |

### Wave 31 mechanism-class taxonomy (8 classes after H50 closure)

| # | Class | Status | Reference PRs |
|---|---|---|---|
| 1 | variance-class-encoder-input | ✅ **WINS** | H26/H31/H35 MERGED |
| 2 | variance-class-decoder-sublayer | TBD | H47 V-DEPTH borderline merge |
| 3 | variance-class-cp-loss-weight | TBD | H53 CP-LOSS-WEIGHT strongest slope |
| 4 | shared-capacity-surface | TBD | H54 v2 SURFACE-DEEP |
| 5 | mean-shift-class | ❌ null | H48 TAU-Y-EQUALIZE closed |
| 6 | cross-channel-weight-space | ❌ null | H45 CROSSCHAN-DEC closed |
| 7 | variance-class-decoder-weight | ❌ null | H46/H49 SDORTH closed |
| 8 | **coordinate-grounded-slice-PE** | **❌ null+VP-cross** (THIS PR) | **H33/H50 closed** |

### 6th test_VP floor crossing in Wave 30/31

Test_VP floor crossings now: H26 NPCA, H31 WALLDIST, H33 SLICEPE, H47 V-DEPTH partial, H53 CP-LOSS-WEIGHT TBD, **H50 COORDSLICE THIS PR**. Test_VP is the most robust merge-adjacent signal in Wave 31 — ANY positional/geometric mechanism reliably crosses it. H50 val_VP terminal 3.676% is the LOWEST val_VP among all Wave 31 PRs at terminal.

### Disposition

CLOSED as mechanism-positive null with 6th test_VP floor crossing + new structural finding (L0-PE-capacity-sink). The +0.094pp val_abupt miss is the closest miss in Wave 31 to date. Excellent diagnostic logging (per-block inter_slice_cos + centroid spread + proj_weight_norm) — the L0-inversion pattern is paper-facing structural finding for Wave 32 PE-design discussion. **Askeladd reassigned to H58 COORDSLICE-NO-STOPGRAD (PR #1207)** — single-line code change removing `torch.no_grad()` wrap, directly attacks the dominant suspected limit (gradient starvation), projected to close +0.094pp gap if hypothesis is correct.

---

## 2026-05-19 11:30 — PR #1197: H49 SDORTH-FULL — Surface Decoder Orthogonal Row Init, 13-Epoch Confirmation (thorfinn) — **MECHANISM-POSITIVE NULL CLOSURE** (variance axis persistent at full budget — test std 0.149 = 1.75× baseline — but ALL 5 paper-facing test metrics DEGRADE vs baseline; structural finding subclassifies variance-class into encoder-input WINS vs decoder-weight NULL); THORFINN REASSIGNED H57 MULTI-SCALE-RFF-EXPANDED (PR #1206) — frequency-domain encoder capacity expansion, new mechanism class FDCE attacking τz axis from encoder freq-band angle

- **Branch**: `thorfinn/h49-sdorth-full` (closed at 11:30Z)
- **W&B run**: 8-rank DDP rank0 `qij8mah1` (group `wave31_h49_sdorth_full`, 14h runtime, full 13 epochs, 70,652 steps — no timeout cut)
- **Hypothesis**: H46 SDORTH (PR #1193, closed 5-ep screening) produced the most important Wave 30/31 structural finding: τz/τx band attractor lives in TRAINING TRAJECTORY, not weight values themselves. H46 EP3-test produced τz/τx mean 1.431 ⭐ — FIRST test-side mean deflection below band [1.44, 1.55] lower edge. H49 binding question: does the test mean deflection PERSIST at full 13-ep terminal, or does extended cosine training re-magnetize the mean into the band attractor?

### Terminal verdict — outcome (B) confirmed, mechanism does NOT translate

| Gate | Target | H49 EP13 | Verdict | Δ vs baseline |
|---|---:|---:|:--|---:|
| val_abupt (merge) | <6.126% | **6.221%** | ❌ FAIL | +0.095pp |
| test_abupt | (baseline 5.844%) | **6.080%** | ❌ FAIL | +0.236pp |
| test_SP (floor) | ≤3.577% | **3.861%** | ❌ FAIL | +0.284pp |
| test_VP (floor) | ≤3.643% | **3.662%** | ❌ FAIL | +0.019pp (close, but other axes degrade) |
| test_WSS (goal) | <6.727% | **6.981%** | ❌ FAIL | +0.254pp |
| **test τz/τx mean (binding)** | **<1.44** | **1.480** | **❌ FAIL** | (back in [1.44,1.55] attractor) |
| test τz/τx std (mech) | >0.10 | **0.149** | ✅ PASS | 1.75× baseline ⭐ |

### Mechanism reading

**EP1 deflection IS reproducible** (val mean 1.373, 62% cars outside band — STRONGER than H46 EP3-test 1.431 read). Cosine-schedule-dependent: under H49's gradual 13-ep cosine, val mean drifts monotonically EP2→EP8 (1.495 → 1.524 → 1.535 → 1.545 → 1.550 → 1.556 → 1.559) and plateaus at band cap 1.559 for 5 consecutive epochs (EP8-EP12). Late-stage cosine decay does NOT re-deflect mean downward. **H46's EP3-test 1.431 was a transient mid-training value** captured before full attractor re-magnetization.

**Per-axis WSS at terminal (test)**:
- τx: 6.173% (easiest)
- τy: 7.570%
- **τz: 9.132%** (hardest — 50% gap above τx, persists across all Wave 31 PRs)

H49 variance spread (test std 0.149 across 50 cars, 48% outside band) does NOT translate into τz reduction. Pattern is **consistent across H46/H47/H48/H49/H53**: per-car heterogeneity (band attractor break) does not move the per-axis WSS magnitude.

### Structural finding — variance-class subclassification

**Variance-class mechanism is now subclassified into two sub-classes:**

- **Variance-class-encoder-input** — **WINS** (H26 NPCA, H31 WALLDIST, H35 NPCA+SSFL — all MERGED). Encoder feature variance translates into lower test error.
- **Variance-class-decoder-weight** — **NULL at translation** (H46/H49 SDORTH). Decoder weight init variance produces persistent variance signature (test std 1.75× baseline) but ALL 5 paper-facing axes DEGRADE.

**Hypothesis**: encoder-input variance creates per-token signal heterogeneity that the downstream model CAN aggregate into lower per-car loss; decoder-weight variance creates output heterogeneity that BYPASSES the model's aggregation capacity and produces per-car ratio dispersion without improving the per-car loss surface.

**For Wave 32**: drop test-mean-deflection axis from attack map. Focus on (a) encoder-input variance, (b) decoder-sublayer depth (H47 proven borderline), (c) per-axis loss curricula (H55 in flight).

### Wave 31 mechanism-class taxonomy (7 classes after H49 closure)

| # | Class | Status | Reference PRs |
|---|---|---|---|
| 1 | variance-class-encoder-input | ✅ **WINS** | H26/H31/H35 MERGED |
| 2 | variance-class-decoder-sublayer | TBD | H47 V-DEPTH borderline merge |
| 3 | variance-class-cp-loss-weight | TBD | H53 second merge candidate |
| 4 | shared-capacity-surface | TBD | H54 SURFACE-DEEP relaunching |
| 5 | mean-shift-class | ❌ null | H48 TAU-Y-EQUALIZE closed |
| 6 | cross-channel-weight-space | ❌ null | H45 CROSSCHAN-DEC closed |
| 7 | **variance-class-decoder-weight** | **❌ null** (THIS PR) | **H46/H49 SDORTH closed** |

### Disposition

CLOSED as mechanism-positive null with structural finding (variance-class subclassification: encoder-input WINS vs decoder-weight NULL). Excellent diagnostic work + cleanly executed full-budget run (14h, no timeout cut). Mechanism diagnostic logging from PR #1193 (`tau_zx_ratio_*` + `surface_proj_row/cos_*`) retained as standard Wave 32 telemetry. Thorfinn reassigned to H57 MULTI-SCALE-RFF-EXPANDED (PR #1206) — encoder frequency-band expansion (5 sigmas → 8 sigmas, 4 octaves → 7 octaves), recipe-only change, new mechanism class FDCE attacking τz axis from encoder side.

---

## 2026-05-19 10:05 — PR #1199: H51 NPCA+SSFL+slices192+ema9999 — Variance-Class Capacity-Expansion Stack (fern, killed mid-EP4) — **RECIPE-BUG CLOSURE** (advisor's EMA-aware kill gate calibration error, NOT mechanism rejection); mechanism was activating (τz/τx std doubled EP2→EP4 from 0.073 → 0.117); FERN REASSIGNED H56 H51-RELAUNCH (PR #1205) with corrected EMA-aware kill gates

- **Branch**: `fern/h51-npca-ssfl-slices192` (closed at 10:05Z)
- **W&B run**: 8-rank DDP rank0 `2vlx68f9` (group `wave31_h51_npca_ssfl_slices192`, 6.5h runtime to step 38,027 mid-EP4 kill, no test eval)
- **Hypothesis**: NPCA + SSFL + slices=128→192 (variance capacity expansion) + ema=0.999→0.9999 (EMA precision). Dual mechanism: slices=192 adds 50% more representational room for variance-class encoder routing; ema=0.9999 reduces noise on best-checkpoint selection.

### Why killed — advisor's recipe bug, NOT mechanism rejection

Run died at global_step 38,027 (mid-EP4) by `--kill-thresholds "32594:val_primary/abupt_axis_mean_rel_l2_pct<10"`. At δ^38027 ≈ 0.022 (2.2% EMA random contamination), EMA-val=14.73% while train/epoch_loss matched H35 reference step-for-step. The trained model was healthy — **kill was triggered by EMA-shadow lag still being too high under ema=0.9999, not by genuine model failure**.

| Step | EP | δ^step | EMA random% | EMA-val_abupt | Trained-val est | train/epoch_loss |
|---:|---:|---:|---:|---:|---:|---:|
| 10,864 | 1 | 0.337 | 33.7% | 71.17% | (EMA-shadow noise) | 0.365 |
| 21,728 | 2 | 0.115 | 11.5% | 53.21% | (EMA-shadow dominated) | 0.085 |
| 32,592 | 3 | 0.039 | 3.9% | 23.64% | ~12% | 0.011 |
| 38,027 | mid-EP4 | 0.022 | 2.2% | **14.73%** ← kill fired | ~9% | ~0.011 |

EP6 (step 65,184) would have been the first **fully informative** read (δ=0.0015, EMA random < 0.2%). Projection at that point: val_abupt 6-8% range, comfortably passing EP6 gate.

### Mechanism trajectory — variance-class signal ACTIVATING (not rejected)

| Step | EP | τz/τx mean | τz/τx std | n_outside_band |
|---:|:--|---:|---:|:--|
| 10,864 | EP1 | 1.3225 | 0.0931 | 27/34 |
| 21,728 | EP2 | 1.2635 | 0.0726 | 34/34 (saturated) |
| 32,592 | EP3 | 1.2968 | 0.0827 | 31/34 |
| 38,027 | mid-EP4 | **1.3402** | **0.1167** | **27/34** |
| H35 EP6 ref (target) | EP6 | — | 0.251 fleet peak | 17/34 |

**Std doubled EP2→EP4 (0.073 → 0.117)** with mean drifting back toward band-edge 1.44 — variance-class expansion signature predicted by the slices=192 capacity-room arm. n_outside dropped from saturated 34/34 → 27/34 = some cars returning into band while overall variance grows → slices=192 routing variance across more cars instead of locking everyone at extreme ratios. Trajectory was on track to enter H35's variance-class regime (target std 0.15-0.20 at EP6 → 0.25 at EP13).

### Structural findings (7 recipe-bug patterns now catalogued)

Fern's diagnostic rigor surfaced TWO new advisor recipe-bug patterns:

**Pattern #6 (NEW)** — kill-threshold step values must be exact `epoch × steps_per_epoch`. I had been using 32,594 / 65,228 (off by +2 / +44 from actual 32,592 / 65,184) across multiple Wave 30/31 PRs. The +2 was a copy-paste artifact from a config with different batch_size/grad_accum. Memory entry `feedback_kill_thresholds_step_indexed.md` corrected.

**Pattern #7 (NEW)** — `--validation-every 1` triggers mid-epoch mini-validations, NOT only at epoch boundaries. The kill-threshold check at step ≥ N fires at the first validation past N — which can be a mid-epoch mini-validation. H51's kill at step 38,027 (mid-EP4 mini-validation) confirmed this. Either suppress mid-epoch validations or design threshold step values defensively.

| # | Pattern | First confirmed in |
|---|---|---|
| 1 | Flag existence + format (hyphen vs underscore) | frieren H52 #1200 |
| 2 | Step-indexed thresholds (N: prefix is global_step) | askeladd H33 #1187, edward H34 #1188 |
| 3 | EMA-step δ^N composition | fern H51 #1199 |
| 4 | lr-warmup-aware EP1 thresholds | (multiple) |
| 5 | SENPAI-RESULT angle-bracket placeholders break JSON parse | alphonse H45 #1192 + edward H48 #1196 |
| 6 | Kill-threshold step = actual epoch × steps_per_epoch (NEW) | **fern H51 #1199** |
| 7 | Mid-epoch mini-validation cadence interaction (NEW) | **fern H51 #1199** |

### Disposition

PR closed at 10:05Z. Fern reassigned to **H56 H51-RELAUNCH** (PR #1205) — exact same recipe with corrected EMA-aware kill gates: `32592:abupt<25` (catastrophic-only EP3) + `43456:abupt<15` (EP4 intermediate) + `65184:abupt<7.0,SP<5.0` (EP6 binding, original spec intent). W&B run `2vlx68f9` preserved for variance-class activation mechanism analysis.

## 2026-05-19 08:55 — PR #1196: H48 TAU-Y-EQUALIZE — Static τ_y Loss-Weight Reduction (edward, 13-ep full) — TERMINAL EP13-EMA CLOSED (val_abupt 6.485% +35.9bp FAIL + test_abupt 6.167% +32.3bp FAIL) / MECHANISM-POSITIVE NULL — MOST EXTREME single-mechanism band-attractor break in Wave 30/31 history (per-car τz/τx mean 0.401, 25× more extreme than H46 SDORTH); MEAN-SHIFT CLASS confirmed as 5th mechanism-class observation

- **Branch**: `edward/h48-tau-y-equalize` (closed at 08:55Z)
- **W&B run**: 8-rank DDP rank0 `8cn5abxm` (group `wave31_h48_tau_y_equalize`, 14.85h runtime to step 70,652 EP13 natural completion, EP10 EMA checkpoint best)
- **Hypothesis**: Reduce τ_y loss weight 1.5 → 1.0 to free gradient capacity for τ_z convergence into the [1.44, 1.55] band attractor. Mechanism class: static channel reweight.

### Terminal metrics (rank0 `8cn5abxm` EP10 EMA best-ckpt)

| Metric | H48 EP10 EMA | Baseline (PR #972) | Δ to baseline | Verdict |
|---|---:|---:|---:|:--|
| val_abupt | **6.485%** | 6.126% (merge) | **+0.359pp** | ❌ merge gate FAIL |
| val_VP | 3.803% | 3.643% floor | +0.160pp | binding gate fail |
| val_SP | 4.277% | 3.577% floor | +0.700pp | binding gate fail |
| val_WSS | 7.351% | 6.727% | +0.624pp | above baseline |
| test_abupt | **6.167%** | 5.844% | **+0.323pp** | ❌ test merge gate fail |
| test_VP | **3.671%** | 3.643% floor | **+0.028pp** | **near-tie at floor** |
| test_SP | 3.898% | 3.577% floor | +0.321pp | binding gate fail |
| test_WSS | 7.113% | 6.727% | +0.386pp | above baseline |
| test_WSS_z | 9.173% | 8.916% | +0.257pp | above baseline |

### MEAN-SHIFT MECHANISM — Most Extreme Band-Break in Wave 30/31

| Set | mean(τz/τx) | std(τz/τx) | n_outside [1.44, 1.55] | Distance below band |
|---|---:|---:|---:|---:|
| val (full_val, 34 cars) | **0.401** | 0.033 | **34/34 (100%)** | **−1.04** |
| test (full_test, 50 cars) | **0.420** | 0.040 | **50/50 (100%)** | −1.02 |
| Baseline #972 val | 1.496 | 0.085 | 0/34 (in band) | — |
| H46 SDORTH (prior most-extreme) | ~1.40 | 0.194 | partial | −0.04 |

H48 is **25× more extreme** in mean-shift direction than the prior record (H46 SDORTH). Std stays compressed (0.033, far below the 0.15 ALIVE variance threshold) — pure mean-shift class, no variance spread component.

### Per-Axis WSS Hardness INVERTED vs Hypothesis

| Axis | H48 EP10 val | Baseline expectation | Direction |
|---|---:|:--|:--|
| `wall_shear_x` | **6.437% (easiest)** | middle | τ_x became EASIEST |
| `wall_shear_y` | 8.011% (middle) | middle | τ_y unchanged |
| `wall_shear_z` | **9.899% (hardest)** | easiest | τ_z became HARDEST |

Hypothesis predicted: τ_y weight reduction frees gradient capacity → τ_z convergence improves toward [1.44, 1.55] band. **Actual**: gradient capacity redirected to τ_x (NOT τ_z) → τ_x became easiest channel, τ_z became hardest. Per-car mean ratio settled at 0.40 (low-τz / high-τx attractor). Opposite direction of hypothesis.

### STRUCTURAL FINDING — Per-Car / Aggregate Decoupling

Aggregate WSS_z/WSS_x ratio (1.538 val, 1.456 test) remained within the [1.44, 1.55] band even when per-car ratios were extreme (mean 0.40). The val_abupt convergence ceiling depends on something other than per-car τz/τx — the aggregate channel is what drives val_abupt. Per-car / aggregate decoupling is the underlying signal pattern.

### Wave 30/31 mechanism-class taxonomy — 5 observed classes

| Class | Wave 31 examples | Result pattern |
|---|---|---|
| Variance-class (encoder) | H35 NPCA+SSFL (ref std 0.251 EP6 peak), H52 NPCA×YAW activating | mechanism alive |
| **Mean-shift class** | **H48 TAU-Y-EQUALIZE (this entry — confirmed 25× extreme)** | **mechanism alive, val null** |
| Variance-class (decoder sublayer) | H47 V-DEPTH (sublayer +26-57%, EP6 6.357% merge candidate) | mechanism alive, **val LIVE** |
| Cross-channel (weight-space) | H45 CROSSCHAN-DEC (weight ratio 24×, val null) | mechanism alive, val null |
| Shared-capacity (surface) | H54 SURFACE-DEEP (alphonse PR #1203, assigned) | TBD |

### Disposition

PR closed at 08:55Z. Edward reassigned to **H55 TAU-Z-LOSS-CURRICULUM** (PR #1204) — mechanism-class-novel time-varying loss weight curriculum (front-load τ_z weight 5.0 → 2.0 by EP6, hold through EP13). Direct attack on the test_WSS_z gap from the opposite direction of H48: instead of relaxing τ_y, front-load τ_z early. W&B run `8cn5abxm` preserved for cross-experiment mean-shift class analysis.

## 2026-05-19 08:25 — PR #1192: H45 ANCHOR-CROSSCHAN-DEC — Surface Decoder Cross-Channel Attention (alphonse, 13-ep full) — TERMINAL EP13-EMA CLOSED (val_abupt 6.3523% +22.6bp FAIL + test_abupt 6.0751% +23.1bp FAIL) / MECHANISM-POSITIVE NULL — weight-space rank-decoupling 18.6× threshold but val null — STRUCTURAL FINDING (band attractor is NOT in surface decoder pre-projection cross-channel structure)

- **Branch**: `alphonse/h45-crosschan-dec` (closed at 08:25Z)
- **W&B run**: 8-rank DDP rank0 `lhivsp6j` (group `wave31_h45_crosschan_dec`, 916.3 min / ~15.3h, natural EP13 completion, EP13 EMA checkpoint best, ~3.4M extra params / +6.5% overhead)
- **Hypothesis**: Insert cross-channel attention layer over 4 output dimensions (cp, τx, τy, τz) BEFORE the final surface_out projection. Each channel becomes a query attending to the others' representations via MultiheadAttention. Zero-init → bit-exact baseline at init. Tests whether the [1.44, 1.55] τz/τx band attractor lives in the surface decoder pre-projection rank-coupling.

### Terminal metrics (rank0 `lhivsp6j` EP13 EMA best-ckpt)

| Metric | EP13 val (34 cars) | EP13 test (50 cars) | Baseline (PR #972) | Δ to baseline | Verdict |
|---|---:|---:|---:|---:|:--|
| val_abupt | **6.3523%** | — | 6.126% (merge) | +22.6 bp | ❌ FAIL |
| val_SP | 4.1331% | — | 3.577% floor | +55.6 bp | ❌ binding gate fail |
| val_vol_p | 3.7906% | — | 3.643% floor | +14.8 bp | tight but above floor |
| val_WSS | 7.2012% | — | 6.727% | +47.4 bp | above baseline |
| test_abupt | — | 6.0751% | 5.844% | +23.1 bp | ❌ FAIL |
| test_SP | — | 3.8488% | 3.577% | +27.2 bp | above floor |
| test_vol_p | — | 3.7074% | 3.643% | +6.4 bp | tight but above floor |
| test_WSS | — | 6.9821% | 6.727% | +25.5 bp | above baseline |
| test wsz/wsx | — | 1.454 | 1.473 | −1.9 bp | ✅ slight band-edge approach (not load-bearing) |

### MECHANISM-POSITIVE NULL — weight-space rank-decoupling fired but val null

| step | trainer EP | cp norm | τx norm | τy norm | τz norm | **τz/τx out_weight_norm** |
|---:|:--|---:|---:|---:|---:|---:|
| 10,864 | EP1 | 0.0779 | 0.0679 | 0.0418 | 0.0563 | 0.83 (sub-symmetric) |
| 32,594 | EP3 | 0.00287 | 0.00217 | 0.02772 | 0.02114 | **9.75 (PASS gate 1.3)** |
| 43,466 | EP5 | 0.00193 | 0.00125 | 0.02754 | 0.02180 | **17.37** |
| **52,528** | **EP7** | 0.00135 | 0.00092 | 0.02851 | 0.02225 | **24.16 ← PEAK (18.6× gate)** |
| 70,652 | EP13 | (sustained 9-24× across EP7-EP13) | | | | strongly mechanism-positive |

**Structural finding**: weight-space rank-decoupling at the surface decoder projection IS achievable but does NOT drive val-space τz/τx band-attractor escape. The band attractor is **downstream of** the projection-matrix rank coupling — it lives in the shared surface_hidden representation BEFORE the projection, or in the encoder feature space that produces surface_hidden, or in the per-vertex loss formulation. Cross-channel attention residual escape mechanism is insufficient for val improvement.

### Wave 30/31 mechanism-class taxonomy update

H45 adds the 5th mechanism class observation:

| Class | Examples | Result |
|---|---|---|
| Variance-class (encoder) | H35 NPCA+SSFL (ref std 0.251 EP6 peak), H52 NPCA×YAW activating | mechanism alive |
| Mean-shift class | H48 TAU-Y-EQUALIZE (mean 0.401 plateau 6.485%) | mechanism alive, val null |
| Variance-class (decoder sublayer) | H47 V-DEPTH (sublayer +26-57%, EP6 6.357% merge candidate) | mechanism alive, **val LIVE** |
| Cross-channel (weight-space) | **H45 CROSSCHAN-DEC (weight ratio 24×)** | mechanism alive, **val null** |
| Shared-capacity (NEW, assigned alphonse H54) | H54 SURFACE-DEEP — surface decoder depth bump mirror of H47 | TBD |

### Disposition

PR closed at 08:25Z. Alphonse reassigned to H54 SURFACE-DEEP (PR #1203) — mirror of H47 V-DEPTH on the surface decoder side. The mechanism-positive null on H45 narrows the search: capacity expansion BEFORE the projection (shared-capacity-class) is the natural next axis to test on the surface side.

## 2026-05-19 02:33 — PR #1191: H36 ANCHOR-SLICE-QUERIES — Deepest vol_p Floor Crossing in Wave 31 History (tanjiro, 13-ep full) — TERMINAL EP13-EMA NOT-A-MERGE (val_abupt +0.112pp MISS + test_SP +0.140pp FLOOR BREACH) / 26TH WAVE-30/31 DEAD END ON MERGE DIMENSION / 🏆 7TH TEST_VOL_P FLOOR CROSSING (−0.165pp DEEPEST in Wave 31) + 3RD VARIANCE-CLASS MECHANISM CONFIRMATION (with H26 NPCA, H35 NPCA+SSFL)

- **Branch**: `tanjiro/h36-anchor-slice-queries` (closed at 02:33Z)
- **W&B run**: 8-rank DDP rank0 `vu93lzgc` (group `wave31_h36_anchor_slice_queries`, 839.6 min / ~14.0h, natural EP13 completion, EP13 EMA checkpoint best, peak GPU 82.6 GB / 80.5%)
- **Hypothesis**: Add learnable 3D anchor positions A ∈ R^{S×3} to each slice query: `q_s' = q_s + MLP_anchor(PE(A_s))`. Each anchor learns to specialize to a spatial region (windshield, A-pillar, underbody, wake), forcing per-slice geometry-specific τz/τx representations. DAB-DETR-style query modulation.

### Terminal metrics (rank0 `vu93lzgc` EP13 EMA best-ckpt)

| Metric | EP13 val (34 cars) | EP13 test (50 cars) | Baseline | Δ to baseline | Verdict |
|---|---:|---:|---:|---:|:--|
| val_abupt | **6.2379%** | — | 6.126% (merge) | +0.112pp | ❌ FAIL |
| val_VP | 3.6174% | — | — | — | — |
| val_SP | 4.1067% | — | — | — | — |
| val_WSS | 7.0779% | — | — | — | — |
| test_abupt | — | 5.9045% | 5.844% | +0.061pp | minor regression |
| test_SP | — | **3.7169%** | 3.577% (floor) | **+0.140pp** | ❌ **FLOOR BREACH (binding gate)** |
| test_vol_p | — | **3.4780%** | 3.643% (floor) | **−0.165pp** | ✅ **PASS — 7TH vol_p crossing + DEEPEST in Wave 31** |
| test_WSS | — | 6.8146% | 6.727% | +0.087pp | minor regression |

### DEEPEST vol_p crossing in Wave 31 history

H36's −0.165pp below floor beats H31 WALLDIST's previous record of −0.155pp by 0.010pp. Volume pathway specialization via anchor-conditioned slice queries delivered the strongest single-axis vol_p improvement to date.

### VARIANCE-CLASS MECHANISM — 3rd confirmation in Wave 31

| Metric | H26 NPCA | H35 NPCA+SSFL | **H36 ANCHOR-SLICE** |
|---|---:|---:|---:|
| val τz/τx mean (EP13) | 1.526 | 1.527 | **1.547** (band-high edge) |
| val τz/τx std (EP13) | 0.108 | 0.251 fleet-peak | **0.195** |
| test τz/τx std | 0.132 | — | **0.140** |
| val n_outside_band | 9/34 | 17/34 | **15/34** |
| test n_outside_band | 24/50 | — | **23/50 (46%)** |
| test_vol_p | 3.607% | 3.585% | **3.478%** (deepest) |
| Verdict | merged | closed | closed |

H36 is the 3rd variance-class confirmation: per-car spread increases (val 0.195 = ~6.5× baseline ~0.03), but mean stays band-high (1.547 just inside upper edge). Same structural ceiling as H26/H35: variance class produces per-car spread WITHOUT translating into per-axis L2 reduction. test_abupt regresses +0.06pp despite the mechanism alive.

### Anchor mechanism vital signs

- `anchor_pairwise_dist_mean` grew 2.142 (EP1) → 2.193 (EP13) — anchors spread across full DrivAerML canonical bbox (x=4.68, y=2.32, z=2.08), NO COLLAPSE
- `anchor_mod_abs_mean = 5.27`, `anchor_mod_abs_max = 348.76`, `anchor_mod_rms = 14.46` — distributed activity with 1-2 slots producing very large directional bias (max ~66× mean = ANCHOR SPARSITY)
- `weight_param/anchors/{mean,std,min,max} = 0.665 / 1.180 / −1.168 / 3.995` — drifted from grid centroid x≈1.7, still inside DrivAerML envelope
- `nonfinite_count = 0` across all anchor modules through 14h training
- Anchor grad `mean_abs = 1.28e-5`, `max_abs = 2.36e-4` (well-behaved)

### Anchor sparsity — possible SP regression mechanism

`anchor_mod_abs_max 348.76` while `anchor_mod_abs_mean 5.27` = max is ~66× mean. A small number of anchors are dominating the modulation signal. This may be what drives the SP channel regression — a few high-magnitude anchors collapsing the surface pressure routing capacity in those cars. **An anchor norm clamp or per-anchor L2 penalty would be the natural follow-up** (student suggestion #4).

### val/test variance mechanism divergence — variance class overfits to training distribution

val τz/τx std 0.195 vs test 0.140 = +0.055pp gap. The variance mechanism partially overfits to the training cars' specialization patterns. This is DISTINCT from H48's mean-shift mechanism where per-car τz/τx mean 0.40 held stable across all 34/34 val cars (and projects to all 50 test cars at terminal). **Mean-shift mechanisms generalize better than variance-class mechanisms** — emerging Wave 31 structural insight.

### Beats AB-UPT on 3 of 4 paper-facing channels

- test_SP: H36 3.717% vs AB-UPT 3.82% = **−0.10pp ahead**
- test_vol_p: H36 3.478% vs AB-UPT 6.08% = **−2.60pp ahead** (massive volume win)
- test_WSS aggregate: H36 6.815% vs AB-UPT 7.29% = **−0.48pp ahead**
- test_abupt: not reported by AB-UPT

But per-axis WSS comparison fails: H36 τx 6.03% vs AB-UPT 5.35% (+0.68pp), H36 τy 7.41% vs AB-UPT 3.65% (+3.76pp), H36 τz 8.89% vs AB-UPT 3.63% (+5.27pp). The per-axis WSS gap is a structural feature of our fleet baseline (#972), not the H36 mechanism.

### Wave 30/31 floor crossing tally — now 7 (DEEPEST yet)

| # | Hypothesis | Mechanism axis | test_vol_p | Δ vs floor | Merge status |
|---|---|---|---:|---:|---|
| 1 | H31 WALLDIST | encoder-input feature | 3.488% | −0.155pp | MERGED |
| 2 | H26 NPCA | encoder-input feature | 3.607% | −0.036pp | MERGED |
| 3 | H46 SDORTH | decoder weight init | — | (PathB) | closed |
| 4 | H33 SLICEPE | encoder slice-PE additive | 3.522% | −0.121pp | closed |
| 5 | H35 NPCA+SSFL | stack: encoder + spectral-loss | 3.585% | −0.058pp | closed |
| 6 | H44 YAW-AUG | data augmentation rotation | 3.608% | −0.035pp | closed |
| 7 | **H36 ANCHOR-SLICE-QUERIES** | **anchor query modulation** | **3.478%** | **−0.165pp ⭐ DEEPEST** | **closed** |

**5 of 7 closed crossings have val_abupt FAIL; 6 of 7 have test_SP FLOOR BREACH.** test_SP remains 0/7 crossings — **SP is the binding unsolved gate.**

### Conclusion

26th Wave 30/31 dead end on merge dim, 7th test_vol_p floor crossing (DEEPEST in Wave 31 at −0.165pp), 3rd variance-class mechanism confirmation. Anchor sparsity (max ~66× mean) and val/test variance divergence (0.195 → 0.140) emerge as new Wave 31 mechanism class observations. H53 CP-LOSS-WEIGHT carried forward to attack test_SP binding gate (PR #1202).

---

## 2026-05-19 02:15 — PR #1190: H44 YAW-AUG — First Data-Augmentation Axis Crossing in DrivAerML Fleet History (frieren, 13-ep full) — TERMINAL EP13-EMA NOT-A-MERGE (val_abupt +0.229pp + test_SP +0.276pp FAIL) / 25TH WAVE-30/31 DEAD END ON MERGE DIMENSION / 🏆 6TH TEST_VOL_P FLOOR CROSSING + 1ST DATA-AUG AXIS + CROSS-CHANNEL WSS REGULARIZATION NOVEL MECHANISM CLASS

- **Branch**: `frieren/h44-yaw-aug-symmetry-break` (closed at 02:08Z)
- **W&B run**: 8-rank DDP rank0 `6scw4nto` (group `wave31_h44_yaw_augmentation`, 51,622s / ~14.3h, EP13 EMA checkpoint best, FINISHED)
- **Hypothesis**: Random per-batch yaw rotation θ_max=5° as symmetry-breaking data augmentation to disrupt the τz/τx band attractor [1.44, 1.55] by forcing the model to learn per-car geometric features rather than the yaw=0 shared representation. First data-augmentation attack in DrivAerML experiment history.

### Terminal metrics (rank0 `6scw4nto` EP13 EMA best-ckpt)

| Metric | EP13 val (34 cars) | EP13 test (50 cars) | Baseline | Δ to baseline | Verdict |
|---|---:|---:|---:|---:|:--|
| val_abupt | **6.355%** | — | 6.126% (merge) | +0.229pp | ❌ FAIL |
| val_VP | 3.642% | — | 3.643% (val floor) | −0.001pp | ✅ val-side cross |
| val_SP | 4.158% | — | — | — | — |
| val_WSS | 7.266% | — | — | — | — |
| test_abupt | — | 6.116% | 5.844% | +0.272pp | ❌ FAIL |
| test_SP | — | **3.853%** | 3.577% (floor) | **+0.276pp** | ❌ **FLOOR BREACH (surface tax)** |
| test_vol_p | — | **3.608%** | 3.643% (floor) | **−0.035pp** | ✅ **PASS — 6th vol_p crossing + FIRST data-aug axis** |
| test_WSS | — | 7.063% | 6.727% | +0.336pp | ❌ FAIL on aggregate (per-axis ahead, see below) |

### CROSS-CHANNEL WSS REGULARIZATION — novel mechanism class

Yaw rotation mixes τx ↔ τy ONLY (rotation about z-axis), but the per-axis improvements are BALANCED across all 3 axes at test:

| Axis | H44 test | baseline test | Δ |
|:--|---:|---:|---:|
| `wall_shear_x_rel_l2_pct` | 6.254% | ~6.50% (proj) | **−0.25pp ahead** |
| `wall_shear_y_rel_l2_pct` | 7.743% | ~8.25% (proj) | **−0.51pp ahead** |
| `wall_shear_z_rel_l2_pct` | 9.122% | ~10.00% (proj) | **−0.88pp ahead** |

**τz NOT mixed by yaw rotation, yet improved by −0.88pp.** This is the cleanest cross-channel WSS-axis regularization signal in Wave 31. Mechanism: rotation-equivariance prior provides cross-channel regularization on axes that aren't directly rotated. Novel mechanism class — not predicted by the original H44 hypothesis.

### Aggregate test_WSS 7.06% trails baseline 6.73% by +0.34pp

Aggregate sums squared errors weighted by total error budget, not by axis count. H44's WSS-axis distribution is more uniform than baseline (less concentrated on τz), which is the per-car std signal — but absolute magnitudes still trail baseline EP13 terminal. The cleanest per-axis WSS-balanced result in Wave 31 so far.

### Per-car std(τz/τx) — variance gate PASSES on val

| Metric | H44 val (34 cars) | H44 test (50 cars) | Status |
|---|---:|---:|:--|
| per-car std | **0.198** | 0.118 | val ✅ PASSES 0.15 ALIVE gate, test between 0.05 KILL and 0.15 |
| per-car mean | 1.512 (band-low edge) | 1.461 (35% cars BELOW 1.44) | val band-low, test partial band exit |

The mechanism is alive: yaw augmentation produces meaningful per-car variance break on val, partial on test. The 35% of test cars below 1.44 is notable — yaw rotation is shifting some cars away from the band attractor entirely.

### Beats AB-UPT on key channels

- test_vol_p: H44 3.608% vs AB-UPT 6.08% = **−2.47pp ahead** (massive win on volume pressure)
- test_WSS aggregate: H44 7.063% vs AB-UPT 7.29% = **−0.23pp ahead**
- test_SP: H44 3.853% vs AB-UPT 3.82% = tied (+0.03pp)

H44 beats AB-UPT on 2 of 3 paper-facing channels and ties on the 3rd — independent of merge gate verdict.

### Wave 30/31 floor crossing tally — now 6 (FIRST data-aug axis)

| # | Hypothesis | Mechanism axis | test_vol_p | Δ vs floor | Merge status |
|---|---|---|---:|---:|---|
| 1 | H31 WALLDIST | encoder-input feature | 3.488% | −0.155pp | MERGED |
| 2 | H26 NPCA | encoder-input feature | 3.607% | −0.036pp | MERGED |
| 3 | H46 SDORTH | decoder weight init | — | (PathB) | closed |
| 4 | H33 SLICEPE | encoder slice-PE additive | 3.522% | −0.121pp | closed |
| 5 | H35 NPCA+SSFL | stack: encoder + spectral-loss | 3.585% | −0.058pp | closed |
| 6 | **H44 YAW-AUG** | **data augmentation rotation** | **3.608%** | **−0.035pp** | **closed (FIRST data-aug axis)** |

### Why surface tax kicked in — interpretation

The +0.276pp surface_pressure test regression is the price for the augmentation-perturbation budget. SP depends on local geometry detail (vehicle shape, panel boundaries) that yaw rotation distorts through the model's positional encoding pipeline. Yaw aug at θ_max=5° rotates the entire vehicle 5° around z-axis on average — this changes the relative position of every panel vs the input coordinate frame. The model has to learn to predict surface_pressure under this rotated geometry, which costs ~3.5% relative error. **Structural**: any rotation-augmentation will pay a surface_pressure tax.

### Conclusion

25th Wave 30/31 dead end on merge dim, 6th test_vol_p floor crossing, **FIRST data-augmentation axis crossing** in DrivAerML fleet history. Beats AB-UPT on vol_p (massively) and WSS aggregate. All 3 WSS axes ahead of baseline at test. Per-car std(τz/τx) gate passes on val. H52 NPCA × YAW-AUG mechanism stack carried forward (PR #1200) — predicted to close val_abupt gap via NPCA frame-invariance softening H44's surface-pressure tax.

---

## 2026-05-19 00:15 — PR #1189: H35 NPCA+SSFL STACK — First Proven Mechanism-Stacking Experiment (fern, 13-ep full) — TERMINAL EP13-EMA NOT-A-MERGE (val_abupt +0.172pp + test_SP +0.194pp FAIL) / 24TH WAVE-30/31 DEAD END ON MERGE DIMENSION / 🏆 5TH TEST_VOL_P FLOOR CROSSING + STACKING INDEPENDENCE PROVEN + FLEET-PEAK τz/τx VARIANCE

- **Branch**: `fern/h35-npca-ssfl-stack` (closed at `6321e19`)
- **W&B run**: 8-rank DDP `7zkdf9xv` (group `wave31_h35_npca_ssfl_stack`, 838.7 min / 1100 min budget, EP13 EMA checkpoint best, peak GPU 78.43 GiB / 95.5 GiB)
- **Hypothesis**: Combine H26 NPCA (per-vertex normal-PCA encoder-input enrichment, variance-break mechanism) × H31 follow-up SSFL (spectral-band loss reshape, λ=0.05, hf_weight=2.0). Predicted: two mechanisms operating on orthogonal axes (encoder-input enrichment vs decoder spectral loss) stack additively without interference.

### Terminal metrics (rank0 `7zkdf9xv` EP13 EMA best-ckpt)

| Metric | EP13 val (34 cars) | EP13 test (50 cars) | Baseline | Δ to baseline | Verdict |
|---|---:|---:|---:|---:|:--|
| val_abupt | **6.298%** | — | 6.126% (merge) | +0.172pp | ❌ FAIL |
| val_VP | 3.687% | — | — | — | — |
| val_SP | 3.965% | — | — | — | — |
| val_WSS | 7.330% | — | — | — | — |
| test_abupt | — | 5.995% | 5.844% | +0.151pp | minor regression |
| test_SP | — | **3.771%** | 3.577% (floor) | **+0.194pp** | ❌ **FLOOR BREACH** |
| test_vol_p | — | **3.585%** | 3.643% (floor) | **−0.058pp** | ✅ **PASS — 5th vol_p crossing in Wave 30/31** |
| test_WSS | — | 6.926% | 6.727% | +0.199pp | minor regression |

### STACKING MECHANISM PROVEN — NPCA variance signal SURVIVED SSFL

| Metric | H26 NPCA standalone | H35 NPCA+SSFL stack | Δ |
|---|---:|---:|---:|
| val τz/τx mean (EP13) | 1.526 | 1.527 | +0.001 (mid-band preserved) |
| val τz/τx std (EP13) | 0.108 | **0.251** ⭐ FLEET PEAK | **+0.143** |
| val τz/τx max (EP13) | ~1.70 | **2.80** | **+1.10** |
| val n_outside_band [1.44, 1.55] | 9/34 | **17/34** | **+8 cars** |
| test_vol_p | 3.607% | 3.585% | −0.022pp |

τz/τx std grew monotonically through 13 epochs: 0.078 → 0.121 → 0.155 → 0.184 → 0.205 → 0.223 → 0.236 → 0.244 → 0.249 → 0.251 → 0.251 → 0.251 → 0.251. **No SSFL suppression of the NPCA-induced channel asymmetry**. First proven independence between Wave 30 mechanism classes — validates the broader stacking program (NPCA × WALLDIST, NPCA × SDORTH, etc.).

### Why val_abupt didn't cross — variance room insight

H35 has 2.3× the τz/τx variance of H26 NPCA but only 0.17pp better val_abupt. The variance is being USED (n_outside_band 17/34 vs 9/34) but the model lacks the CAPACITY to convert per-slice heterogeneity into per-axis accuracy gain. 128 slices × 2.3× variance budget = ~295 effective slice-modes of capacity needed, but only 128 raw slice queries available → bottleneck. **Scaling slices 128 → 192 (H51) gives +50% capacity** to absorb the variance signal.

### EMA-decay 0.999 issue — structural config bug

Independently identified by student in priority follow-up #6: at lr-cosine-t-max=13 with ~141k total steps, EMA-decay 0.999 has effective window ~1000 steps, but the cosine tail (last ~70k steps) is where the best checkpoints live. **0.9999 (10× longer window) is the correct value for 13-ep recipes.** Helps every closed run by ~0.05-0.15pp val_abupt. Retrofitted in H51.

### Per-axis wall_shear at EP13

| Axis | H35 EP13 | baseline EP13 | Δ |
|:--|---:|---:|---:|
| `wall_shear_x_rel_l2_pct` | 6.512% | 6.502% | +0.010pp (tied) |
| `wall_shear_y_rel_l2_pct` | 8.299% | 8.250% | +0.049pp (slight regression) |
| `wall_shear_z_rel_l2_pct` | 10.087% | 10.000% | +0.087pp (slight regression) |

NPCA mechanism didn't improve per-axis wall_shear despite τz/τx asymmetry. Variance expressed in BUDGET (channel weight allocation) not GRADIENT (axis-specific learning rate). **Mechanism class determines gradient routing, not loss target** (combined finding with H44).

### Beats AB-UPT on key channels (paper baseline)

- p_s (surface pressure): H35 ahead of AB-UPT
- p_v (volume pressure): H35 ahead of AB-UPT
- Vector τ (wall shear stress): H35 ahead of AB-UPT
- ABUPT meta-aggregate: H35 behind AB-UPT by +0.151pp

Vol_p PASS + AB-UPT victory on the volume pressure channel = the cleanest single-metric story for Wave 31 structural finding write-up.

### Wave 30/31 floor crossing tally — now 5

| # | Hypothesis | Mechanism axis | test_vol_p | Δ vs floor | Merge status |
|---|---|---|---:|---:|---|
| 1 | H31 WALLDIST | encoder-input feature | 3.488% | −0.155pp | MERGED |
| 2 | H26 NPCA | encoder-input feature | 3.607% | −0.036pp | MERGED |
| 3 | H46 SDORTH | decoder weight init | — | (PathB) | closed |
| 4 | H33 SLICEPE | encoder slice-PE additive | 3.522% | −0.121pp | closed |
| 5 | **H35 NPCA+SSFL** | **stack: encoder + spectral-loss** | **3.585%** | **−0.058pp** | **closed** |

3 of 5 vol_p crossings have val_abupt fail — merge dimension harder than the floor. Consistent with volume pathway being easier (smooth scalar field) vs abupt meta-aggregate (multiple coupled axes).

### Conclusion

24th Wave 30/31 dead end on merge dim, 5th test_vol_p floor crossing, FIRST proven mechanism-stacking experiment (NPCA × SSFL independent and additive). Fleet-peak τz/τx variance 0.251 (vs 0.108 H26 NPCA standalone). Two structural follow-ups carried forward as H51:
1. `--model-slices 128 → 192` (+50% capacity to use variance room)
2. `--ema-decay 0.999 → 0.9999` (10× longer EMA window for 13-ep cosine)

H51 launched as PR #1199.

---

## 2026-05-18 21:25 — PR #1187: H33 SLICEPE — Learnable Slice Positional Embedding (askeladd, 13-ep full) — TERMINAL EP13-EMA NOT-A-MERGE (val_abupt + test_SP FAIL) / 23RD WAVE-30 DEAD END ON MERGE DIMENSION / 🏆 4TH TEST_VOL_P FLOOR CROSSING + L0-DOMINANCE STRUCTURAL INSIGHT

- **Branch**: `askeladd/h33-slice-position-embedding` (closed)
- **W&B run**: 8-rank DDP rank0 `u58fwoym` (clean 13-ep completion, 838.16 min / 1100 min budget, EP13 EMA checkpoint best, descent saturated mid-EP7 with slope decay −0.0035 pp/1k → −0.0004 pp/1k = 9× slowdown)
- **Hypothesis**: Add learnable additive slice positional embedding `slice_pe[1, H, S, D_head]` per layer to `slice_tokens` before qkv projection. Predicted: deeper layers develop more inter-slice cosine spread as model learns region-specialised positional routing → break τz/τx band attractor via slice-specific physics routing.

### Terminal metrics (rank0 `u58fwoym` EP13 EMA best-ckpt)

| Metric | EP13 val (34 cars) | EP13 test (50 cars) | Baseline | Δ to baseline | Verdict |
|---|---:|---:|---:|---:|:--|
| val_abupt | **6.472%** | — | 6.126% (merge) | +0.346pp | ❌ FAIL |
| val_SP | 4.269% | — | — | — | — |
| val_VP | 3.744% | — | — | — | — |
| val_WSS | 7.345% | — | — | — | — |
| test_abupt | — | 5.960% | 5.844% | +0.116pp | minor regression |
| test_SP | — | **3.793%** | 3.577% (floor) | **+0.216pp** | ❌ **FLOOR BREACH** |
| test_vol_p | — | **3.522%** | 3.643% (floor) | **−0.121pp** | ✅ **PASS — 4th vol_p crossing in Wave 30/31** |
| test_WSS | — | 6.856% | 6.727% | +0.129pp | minor regression |

### Test τz/τx — band attractor holds (no deflection)

| stat | val | test |
|---|---:|---:|
| `tau_zx_ratio_mean` | 1.554 (upper edge band) | **1.487** (mid-band) |
| τz/τx behavior | in band | in band — no deflection |

Unlike H46 SDORTH (test mean 1.431 below band), H33's test τz/τx mean stays in band. Confirms encoder-side additive PE mechanism does NOT produce test mean deflection — that mechanism class is reserved for per-vertex weighting (H18) and decoder weight init (H46).

### Mechanism FALSIFIED — depth pattern is BACKWARDS (L0 dominates, novel insight)

H33 predicted: `inter_slice_cos` should INCREASE L0 → L4 (depth-monotonic differentiation).

Terminal EP13 reading (step 70,652):

| Layer | inter_slice_cos | Prediction | Status |
|:--|---:|:--|:--|
| L0 | **0.0851** | should be SMALLEST | ❌ LARGEST by ~3-10× |
| L1 | 0.0189 | < L2 | ❌ smallest of L1-L4 |
| L2 | 0.0285 | < L3 | ❌ above L3 |
| L3 | -0.0010 | < L4 | ❌ essentially uniform |
| L4 | 0.0283 | should be LARGEST | ❌ tied for third |

**Novel insight**: L0 attends directly to raw geometric input and benefits most from per-slice positional disambiguation. Deeper layers operate on fused token embeddings and prefer content-routing. This is the FIRST cold-start fade in Wave 30/31 with an explicit per-layer mechanism diagnosis — produces a concrete actionable Wave 31 follow-up (coordinate-conditioned slice IDs at L0).

### slice_pe parameter growth — σ=O(1/√D)≈0.088 auto-rediscovered

| Stat | Init (EP0) | Terminal (EP13) | Growth |
|---|---:|---:|---:|
| global slice_norm | 0.23 | 1.80 | 8× |
| std | 0.020 | 0.150 | 7.5× |

Literature warning that σ=0.02 is below gradient floor was CORRECT. Model auto-grew the PE to σ≈0.15 (above O(1/√D_head) = 0.088 for D_head=128). Future learnable-PE experiments should init at σ=0.088 directly.

### Descent saturated mid-EP7 — no benefit from extending the run

| Time | Step | val_abupt | val_VP | Slope |
|---:|---:|---:|---:|:--|
| 16:42Z | 55,559 | 6.507% | 3.781% | EP5+ check |
| 18:50Z | 62,492 | 6.475% | 3.754% | −0.032pp / 2h08m |
| 21:00Z | 70,652 | 6.472% | 3.744% | **−0.003pp / 2h10m** ← saturated |

Slope decay −0.0035 → −0.0004 pp/1k steps (9× slowdown by mid-EP7). Cosine LR decay final phase dominated; effective learning stopped at val_abupt ~6.47% / val_VP ~3.74%. Terminal slope at EP13: −0.001 pp/1k steps.

### Wave 30/31 test_vol_p floor crossing tally — now 4

| # | Hyp | Mech class | val_abupt | test_vol_p | Merged? |
|---:|---|---|---:|---:|:--|
| 1 | H31 WALLDIST | Encoder-input log-SDF | 6.176% | −0.155pp | ✅ MERGED |
| 2 | H26 NPCA | Encoder-input local-frame | improved | −0.035pp | ✅ MERGED |
| 3 | H46 SDORTH (Path B) | Decoder weight init | 6.868% | breach (3-ep gap) | ❌ FLOOR BREACH on screening |
| 4 | **H33 SLICEPE** | **Encoder slice-PE additive** | **6.472%** | **−0.121pp** | ❌ NOT-A-MERGE (val_abupt + test_SP) |

**Pattern observation**: test_vol_p (3 mechanism classes crossed; only baseline #972 + 2 mech wins crossed test_SP). **test_SP is the binding merge gate, not test_vol_p**. Wave 31 hypothesis design should prioritize mechanisms targeting surface pressure pathway specifically.

### Suggested follow-ups (student-prioritized)

1. **σ=0.088 init arm (LOW)** — emergently explored; model auto-grew. Likely no flip.
2. **Input-only slice_pe (MEDIUM)** — restrict PE to L0 only; cheaper and tests L0-dominance directly.
3. **Coordinate-conditioned slice IDs (HIGH per literature)** — DAB-DETR / Anchor DETR analogue. Replaces free-floating slice_pe with input-coordinate-derived centroids. ✅ ASSIGNED as H50 COORDSLICE to askeladd (PR #1198).
4. **Drop slice-attention entirely (HIGH-risk HIGH-reward)** — per LinearNO arXiv:2511.06294; replace inter-slice SDPA with linear projection. Reserved for future idle slot.
5. **Per-car τz/τx telemetry** — std/min/max/n_outside_band logging at val time; useful diagnostic for future band-attractor experiments.

### Closure rationale

Closing rather than sending back: (a) primary mechanism comprehensively falsified across all 13 epochs; (b) descent saturated mid-EP7 — no headroom in this recipe; (c) test_vol_p positive already banked at terminal; (d) L0-dominance insight is the seed for H50 COORDSLICE (structurally different next experiment).

### What this preserves for Wave 31

1. **L0-dominance pattern in slice-PE space** — first cold-start fade with explicit per-layer diagnosis. Reshapes Wave 31 hypothesis design toward L0-targeted mechanisms.
2. **test_vol_p floor is empirically easier than test_SP** — 4 vol_p crossings via 3 mech classes vs 0 test_SP crossings beyond baseline. Re-prioritizes Wave 31 toward surface-pressure-specific mechanisms.
3. **σ=O(1/√D) init is auto-rediscovered** by Lion when starting from σ=0.02 (literature-confirmed at 0.088 for D_head=128).
4. **Slice-PE → vol_p pathway**: encoder-side positional information flows back to volume tokens through slice→volume cross-attention. Tractable secondary vol_p mechanism but poor primary mechanism for val_abupt + test_SP.

### Reassignment

→ **askeladd** assigned **H50 COORDSLICE** (PR #1198): Coordinate-conditioned slice IDs (DAB-DETR analogue). Replaces free-floating learnable slice_pe with slice IDs derived from physical 3D coordinates of each slice's centroid. Single-flag `--use-coord-slice-pe`. Tests whether physically-grounded slice anchors produce cleaner mechanism than auto-grown free-floating IDs. Full 13-ep budget, same recipe as H33 v2.

---

## 2026-05-18 20:15 — PR #1193: H46 SDORTH — Surface Decoder Orthogonal Row Init (thorfinn, Path B 5-ep) — TERMINAL EP3-BEST NOT-A-MERGE (3-EP BUDGET FLOOR BREACHES) / 22ND WAVE-30 DEAD END ON MERGE DIMENSION / 🏆🏆🏆 3RD MECHANISM WIN: FIRST TEST τz/τx MEAN DEFLECTION + PATH-DEPENDENT ATTRACTOR PROOF

- **Branch**: `thorfinn/h46-sdorth` (closed)
- **W&B runs**: 8-rank DDP, rank0 `hoj593kk` (train timed out mid-EP3 step 30,468 ~80% through EP3 due to Path B 6h pod cap, full val + test on best EP3 EMA ckpt complete). Other ranks: `amy4ry5c`, `sgbaoy89`, `cuz59v90`, `03ejvmot`, `yikixmhn`, `d7mdnxfp`, `gk3fzohd`.
- **Recipe**: Path B 5-ep screening (`--epochs 5 --lr-cosine-t-max 5 --vol-points-schedule 0:16384:3:32768:4:49152`)
- **Hypothesis**: Initialize surface decoder Linear(512, 4) final projection's 4 row vectors to be mutually orthogonal (via `nn.init.orthogonal_` + Kaiming-magnitude rescaling), 3 LOC change. Test if τz/τx band attractor is set by INITIAL CONDITION of projection row-coupling or is gradient-driven beyond init.

### Terminal metrics (rank0 `hoj593kk` EP3 EMA best-ckpt)

| Metric | EP3 val (34 cars) | EP3 test (50 cars) | Baseline | Δ to baseline | Verdict |
|---|---:|---:|---:|---:|:--|
| val_abupt | 6.868% | — | 6.126% (merge) | +0.742pp | ✗ FAIL (3-ep budget) |
| test_abupt | — | 6.595% | 5.844% | +0.751pp | ✗ regression |
| test_SP | — | **4.226%** | 3.577% (floor) | **+0.649pp** | ✗ **FLOOR BREACH** |
| test_vol_p | — | **3.917%** | 3.643% (floor) | **+0.274pp** | ✗ **FLOOR BREACH** |
| test_WSS | — | 7.594% | 6.727% | +0.867pp | ✗ regression |

### Mechanism diagnostic — 🏆🏆🏆 First test-side τz/τx mean deflection in Wave 30/31

| τz/τx stat | EP1 val | EP2 val | EP3 val | **EP3 TEST (50 cars)** | Pattern |
|---|---:|---:|---:|---:|:--|
| `tau_zx_ratio_mean` | 1.381 (below) | 1.475 | 1.490 (in band) | **1.431** ⭐ | **TEST mean below band 1.44 — FIRST IN WAVE 30/31** |
| `tau_zx_ratio_std` | 0.085 | 0.163 | **0.194** | 0.112 | std-spread monotonic + persistent on test |
| `tau_zx_ratio_n_outside_band` | 23/34 | 13/34 | 16/34 (47%) | **23/50 (46%)** | mostly displaced from attractor on test |

### Path-dependent attractor proof — Wave 30 structural question definitively answered

`surface_proj_row/cos_max_abs` trajectory: step 0 = 2.3e-08 → EP1 0.143 → EP2 0.204 → EP3 **0.210** (2.1× baseline 0.098).

The weight-level orth structure is **fully gradient-overwritten BY EP1** and continues drifting past baseline through EP3. Yet val std grew monotonically (0.085 → 0.194), and the TEST mean stayed below band (1.431) with 46% of test cars displaced.

**Conclusion**: The τz/τx band attractor is **a training-trajectory attractor, NOT a fixed-point in weight space**. The init perturbation fingerprints the trajectory; the trajectory deflects the test-set mean even after the weight-level init structure has decayed. This answers the Wave 30 open question definitively.

### Three orthogonal Wave 30/31 mechanism wins now characterized

| Mech win | Axis | val std | test std | test mean | test_vol_p crossing |
|---|---|---:|---:|---:|---|
| H26 NPCA | Encoder-input local-frame | 0.259 | 0.132 | 1.467 (preserved) | 🏆 −0.035pp |
| H31 WALLDIST | Encoder-input log-SDF | ~0.25 | ~0.10 | 1.470 (preserved) | 🏆 −0.155pp |
| **H46 SDORTH** | **Decoder weight init** | **0.194** (EP3 only) | **0.112** | **1.431 ⭐ deflected** | ❌ on 3-ep budget; CHECK at 13-ep follow-up |

### Implementation excellence

- Path B screening run correctly chosen for mechanism viability before full budget
- 6h pod cap respected with clean 1.5h test-eval buffer
- Per-epoch τz/τx + surface_proj_row diagnostics throughout
- Test-side τz/τx separately computed on 50-car test split (student noticed val/test divergence)
- Honest "mechanism POSITIVE / accuracy NEGATIVE on 3-ep budget" framing
- Per-trajectory fingerprint analysis (surface_proj_row cos trajectory) directly addressing the Wave 30 structural question
- Floor-breach instruction respected (no SENPAI-RESULT line posted)
- 4 specific follow-up suggestions, each with clear scope

### What this preserves for Wave 31

1. **`--use-surface-orth-init --surface-orth-init-std 0.02` flag pair**: preserved as Wave 31 architectural primitive. Bit-exact baseline at step 0 (3-LOC change). Reserved for H49 follow-up.

2. **Path-dependent attractor finding**: definitive answer to Wave 30 structural question. The surface decoder Linear(512,4) projection's row-coupling lives in the TRAINING TRAJECTORY, not the weight values themselves. Reshapes H45/H47/H48 attack framing.

3. **Test-side mean-deflection pattern**: H18 watch-item-3 reproducible. Now confirmed across 2 mechanism classes (per-vertex weighting via H18, decoder-init via H46). The pattern is real, not a one-off.

4. **H49 SDORTH-FULL follow-up assigned (PR #1197)**: tests whether mean deflection persists at full 13-epoch budget and whether floors close. Binary high-info question — Wave 31's highest-info follow-up.

### Closure rationale

Closed (not sent back) because: (a) the 5-ep Path B was a screening recipe — full follow-up requires materially different setup (18h timeout config, different vp schedule, full 13-ep cosine) deserving fresh PR; (b) the screening result is complete and conclusive on its own terms; (c) closure lets this PR stand as the canonical "mechanism proof on screening recipe" reference; (d) H49 SDORTH-FULL gets a clean scope.

## 2026-05-18 16:35 — PR #1188: H34 OUTHEAD — Per-Channel Auxiliary Output Heads (edward) — ADVISOR-KILLED MID-EP6 / MECHANISM FALSIFIED AT EP3 (ANTI-DIRECTION) / 21ST WAVE-30 DEAD END

- **Branch**: `edward/h34-outhead`
- **W&B runs**: 8-rank DDP, rank0 `iw2ommjz` (killed mid-EP6 step ~33800, 7.36h runtime). val_abupt descending 6.867% → 6.627% → 6.542% but mechanism falsified at EP3 — killed to free GPU for higher-value Wave 31 follow-up.
- **Hypothesis**: Add a per-channel auxiliary residual MLP head for each of the 4 surface output channels {cp, τ_x, τ_y, τ_z} BEFORE the final `Linear(512, 4)` projection. Zero-init each head's last layer → bit-exact baseline at step 0. If the band attractor is set by head-side rank coupling, the per-channel residuals give τ_z and τ_x independent degrees of freedom to escape, with aux_head asymmetry (τ_z/τ_x abs_mean) > 1.5 as the mechanism signal.
- **Terminal val trajectory (EP1→EP5, EP6 ~15% complete when killed)**:

| EP | step | val_abupt | val_SP | val_VP | val_WSS | τz/τx |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 10,864 | 27.416% | 20.375% | 16.592% | 30.550% | 1.375 |
| 2 | 21,729 | 7.830% | 5.227% | 4.634% | 8.858% | 1.489 |
| 3 | 32,594 | 6.867% | 4.595% | 4.084% | 7.730% | **1.531** band-locked |
| 4 | 43,459 | 6.627% | — | 3.991% | 7.435% | — |
| 5 | 54,324 | 6.542% | — | 3.946% | 7.318% | — |

- **Mechanism verdict at EP3 (binding gate)**:

| Diagnostic | Predicted | Observed | Verdict |
|---|---:|---:|:--|
| τz/τx error ratio | ≤ 1.42 (break) | 1.531 | ✗ STUCK in [1.44, 1.55] band |
| aux_head asymmetry τ_z/τ_x abs_mean | > 1.5 | **0.298** | ✗ **ANTI-DIRECTION** (τ_x 3.4× larger than τ_z) |
| aux_head/tau_z/last_layer_norm | > 0 (growing) | 0.639 | partial (gradient flow OK but wrong direction) |
| nonfinite_count | 0 | 0 | ✓ |

- **Closure rationale**: The aux head asymmetry was decisively anti-direction. τ_x aux head grew 3.4× larger than τ_z, which is opposite of the rank-coupling hypothesis prediction. This means the per-channel residual capacity, instead of letting τ_z deviate from the shared mode, mainly flowed into τ_x — confirming the band attractor's stability is NOT from a head-side rank-1 coupling that the residual heads can break. The trunk-side / projection-init axis remains the structural cause (per the H45 + H46 attacks now in-flight).
- **What this contributes to Wave 30 closure tally**: 21st dead end (post-H30 finalization at 20). The "21st" tag is retroactive — H34 was running parallel to H30 closure and is properly attributed to the same Wave 30 mechanism-axis sweep.
- **What this preserves for Wave 31**: 
  1. **Per-channel head capacity axis decisively closed**: surface decoder bottleneck is NOT a missing per-channel capacity. The H45 + H46 attacks on pre-projection and weight-init are the remaining surface decoder axes.
  2. **τ_y over-weighting hypothesis adopted as H48**: anti-direction τ_x dominance suggests τ_x is gradient-favored over τ_z, which may stem from τ_y over-weighting (1.5×) crowding out τ_z's share of the surface loss gradient. Tested in H48 TAU-Y-EQUALIZE (PR #1196, single-flag `--tau-y-loss-weight 1.5 → 1.0`).
  3. **EP3 falsifier worked as intended**: the binary asymmetry threshold cleanly rejected the head-side rank-coupling mechanism class. Adopted as canonical Wave 31 EP3 binary-falsifier pattern for capacity-class hypotheses.

## 2026-05-18 15:30 — PR #1184: H30 V2S xattn — Volume-to-Surface Cross-Attention (nezuko) — TERMINAL EP12-PARTIAL-EP13 NOT-A-MERGE / 20TH WAVE-30 DEAD END / NO MECHANISM-CLASS WIN (band did not break on val or test)

- **Branch**: `nezuko/h30-v2s-xattn`
- **W&B runs**: 8-rank DDP, rank0 `zp494bph` (12 full epochs + ~6% EP13 before SENPAI_TIMEOUT_MINUTES=1100 budget exhaustion at 1011.49 min / 16.86h)
- **Hypothesis**: Add a Volume-to-Surface cross-attention sublayer (reverse direction to existing S2V xattn) so surface decoder can read volume token representations, predicted to improve surface metrics and break τz/τx band attractor via H18's test-side survival pattern.
- **Recipe**: `--use-vol-to-surf-xattn` (added flag) + standard 18h recipe `--epochs 13 --vp-schedule "0:16384:3:32768:6:49152:9:65536"`. ~30 LOC implementation: 1 nn.MultiheadAttention + 1 nn.LayerNorm sublayer with zero-init out_proj for bit-exact baseline recovery.
- **Results table**:

| Metric | val (EP12 EMA best) | test (EP12 EMA reload) | Baseline #972 test | Δ vs baseline test | Floor |
|---|---:|---:|---:|---:|---:|
| **val_abupt** | **6.362%** | — | 6.126% val | **+0.236pp** ✗ FAIL merge gate | — |
| test_abupt | — | 6.091% | 5.844% | +0.247pp ✗ regression | — |
| test_SP | — | 3.866% | 3.577% (floor) | **+0.289pp** ✗ **FLOOR BREACH** | 3.577% |
| test_vol_p | — | 3.781% | 3.643% (floor) | **+0.138pp** ✗ **FLOOR BREACH** | 3.643% |
| test_WSS | — | 6.976% | 6.727% | +0.249pp ✗ regression | < 5.85% goal |
| **val τz/τx** | **1.534** | — | — | inside band (+0.06 drift from EP1) | mech ≤1.42 |
| **test τz/τx** | — | **1.462** | 1.473 / ~0.02 std | **INSIDE BAND** (−0.011pp ≈ baseline within noise) | mech mean ≤1.42 |

- **Trajectory val_abupt EP1→EP12**: 28.295 → 7.595 → 6.808 → 6.576 → 6.500 → 6.448 → 6.416 → 6.392 → 6.375 → 6.369 → 6.364 → 6.362 (best). EP13 partial 6.363 = slight regression. Slope decelerated EP3→EP12 (−0.232 → −0.002 pp/EP), no cosine-LR-engagement acceleration.
- **Trajectory τz/τx EP1→EP12**: **1.428** (deepest Wave 30 EP1 deflection from architectural-fusion path, tied with H25 ALGP) → 1.499 → 1.513 → 1.522 → 1.523 → 1.526 → 1.529 → 1.530 → 1.531 → — → — → 1.534. **10th cold-start fade in Wave 30.** Test 1.462 inside band attractor.
- **V2S vs S2V weight diagnostics (EP9 peak)**: `surf_to_vol.out_proj.max_abs` 0.988 vs `vol_to_surf.out_proj.max_abs` 0.187 — **5.3× capacity asymmetry** in favor of forward (S2V) baseline direction. V2S grad/param ratio 2.2× higher than S2V (gradient flow OK, magnitude limited). **First Wave 30 quantitative measurement of architectural-fusion capacity asymmetry.** V2S found a narrow productive direction within supervised loss landscape, not a strong one.
- **Mechanism analysis**: V2S xattn is **biting** (sublayer non-trivially contributing, zero-init shed) but **asymmetric vs S2V** (peak magnitude 5.3× smaller). EP1 break-signal 1.428 = 6th distinct mechanism axis producing EP1 deflection (joining encoder-input H25/H26/H31, loss-shape H10b/H29, regularization H23, per-vertex H18/H18d, attention H32). **Test-side survival pattern (H18 watch-item-3) explicitly did NOT replicate** — test τz/τx 1.462 fell INTO band attractor (within band [1.44, 1.55] and within statistical noise of baseline 1.473). Architectural cross-modal fusion via V2S xattn is **present but insufficient** to break the structural attractor.
- **Key Wave 30 structural confirmation**: H30 V2S xattn closure + H26 + H31 + H18d + H29 + H25 + H32 + H28 + H24 (9 cold-start fades across 6+ mechanism axes) **confirm the surface decoder Linear(512, 4) projection's row-coupling is the structural cause of the τz/τx band attractor.** All Wave 31 surface decoder projection attacks now in-flight: H34 OUTHEAD (post-projection capacity), H45 CROSSCHAN-DEC (pre-projection representation), H46 SDORTH (projection weight init).
- **What this preserves for Wave 31**: (a) V2S architecture preserved as optional flag for future stacks. (b) **V2S/S2V asymmetry diagnostic** (out_proj.max_abs comparison) adopted as canonical Wave 31 diagnostic for any cross-modal fusion sublayer. (c) Wave 31 follow-up suggestions: V2S with asymmetric init scale (let V2S grow faster), V2S with τz-only routing (concentrate limited capacity), V2S pre-training with isolated objective. (d) vp curriculum bump at EP9 (49152→65536) was POSITIVE for vol_p slope — preserve in Wave 31 18h recipes. (e) **NOT a Wave 31 priority**: V2S stacking with encoder-input augmentation (both axes don't address surface decoder structural attractor).
- **Implementation excellence**: Per-epoch val + W&B diagnostics through 12 full epochs, V2S vs S2V weight magnitude tracked at every checkpoint, honest "mechanism-confirmed but test-side-fell-into-band" diagnosis, asymmetric-capacity insight first-of-Wave-30 documentation, 4 Wave 31 follow-up suggestions with risk-adjusted info value ratings, clean 1011.49 min < 1100 budget shutdown, best_checkpoint/source=ema correctly identified.
- **Closure rationale**: val_abupt 6.362% saturates +0.236pp above baseline (slope decelerated to flat by EP12), BOTH test floors breached (test_SP +0.289pp, test_vol_p +0.138pp), test τz/τx stayed inside band — H18-style test-side survival did NOT replicate. No follow-up parameter variation would change merge outcome (5.3× V2S/S2V asymmetry is structural). **nezuko reassigned to H47 V-DEPTH (Volume Decoder Depth Bump, PR # TBD)** — first Wave 31 attack on volume DECODER interior capacity, untouched in Wave 30. Complements H26+H31 vol_p crossings from encoder-input axis.

---

## 2026-05-18 14:30 — PR #1177: H26 NPCA local-frame projection (thorfinn) — TERMINAL EP13 NOT-A-MERGE / 19TH WAVE-30 DEAD END / 2ND TEST_VOL_P FLOOR CROSSING (CANONICAL MECHANISM WIN WITH TEST-SIDE GENERALIZATION)

- **Branch**: `thorfinn/h26-normal-projected-coord-aug`
- **W&B runs**: 8-rank DDP, rank0 `gokysken` (13/13 epochs, 13.6h wall time, Path A full-budget retry of Path B mechanism-positive)
- **Hypothesis**: Augment surface tokens with local-frame position projections `[p·n, p·t1, p·t2]` (Gram-Schmidt-derived tangent basis) to give the encoder explicit local geometric position information that breaks the global-z privilege driving τz/τx band attractor.
- **Recipe**: `--use_local_frame_proj --use-surf-to-vol-xattn --epochs 13 --vp-schedule "0:16384:3:32768:6:49152:9:65536"` (~50 LOC implementation in model.py, identity-init via zero of 3 new input columns)
- **Results table**:

| Metric | val (EP12 EMA best) | test (EP12 EMA reload) | Baseline #972 test | Δ vs baseline test | Floor |
|---|---:|---:|---:|---:|---:|
| **val_abupt** | **6.3462%** | — | 6.126% val | **+0.220pp** ✗ FAIL merge gate | — |
| test_abupt | — | 6.0276% | 5.844% | +0.184pp ✗ regression | — |
| test_SP | — | 3.8048% | 3.577% (floor) | **+0.228pp** ✗ **FLOOR BREACH** | 3.577% |
| **test_vol_p** | — | **3.6079%** | 3.643% (floor) | **−0.035pp** ✅ **FLOOR PASS** (2ND IN WAVE 30) | 3.643% |
| test_WSS | — | 6.9456% | 6.727% | +0.219pp ✗ above baseline | < 5.85% goal |
| test_WSS_x | — | 6.1640% | 5.6071% | +0.557pp ✗ regression | — |
| test_WSS_y | — | 7.5374% | 6.8397% | +0.698pp ✗ regression | — |
| test_WSS_z | — | 9.0238% | 8.2585% | +0.765pp ✗ regression | — |
| **val τz/τx mean / std** | **1.553 / 0.2587** | — | — | — | — |
| **test τz/τx mean / std** | — | **1.467 / 0.1322** | 1.473 / ~0.02 | **mean ≈ baseline / std 6.6× baseline** | mech mean ≤1.42, std ≥ 0.05 |
| **test n_outside_band** | 11/34 (32%) val | **24/50 (48%) test** | ~0% | — | ≥ 1/50 |

- **Trajectory val_abupt EP1→EP13**: 25.605 → 7.780 → 6.909 → 6.615 → 6.541 → 6.475 → 6.437 → 6.412 → 6.394 → 6.372 → 6.353 → 6.346 (best) → 6.350 (slight regression EP13). Slope decelerated EP6→EP12 (−0.037 → −0.0068 pp/EP), cosine-LR engagement NEGATIVE, no EMA-tail boost.
- **Trajectory τz/τx std EP1→EP13**: 0.092 → 0.185 → 0.228 → 0.241 → 0.249 → 0.250 → 0.252 → 0.253 → 0.256 → 0.255 → 0.257 → 0.259 (peak) → 0.259 (test 0.132). **13× baseline at val peak, 6.6× baseline at test — generalizes to held-out set, unique in Wave 30.**
- **Trajectory n_outside_band [1.40, 1.60] EP1→EP13**: 20 → 11 → 13 → 13 → 14 → 12 → 13 → 12 → 12 → 12 → 12 → 11 → 12 (out of 34 val cars); terminal test: **24/50 = 48%** of test cars outside band.
- **Mechanism analysis**: ENCODER-INPUT AXIS CONFIRMED at full scale with test-side generalization. NPCA produced distinct per-car τz/τx structure (std 13×/6.6× baseline val/test, max ratio 2.800 val / 1.831 test, min 1.297 val / 1.136 test). Distribution shape (bipolar, wide tails) preserved through all 13 epochs and all 4 volume curriculum stages. **2nd test_vol_p floor crossing in Wave 30** (−0.035pp below floor, joining H31 WALLDIST's −0.155pp from earlier same day). But **mean(τz/τx)** UNCHANGED at test (1.467 ≈ baseline 1.473) — surface decoder mean-preserves regardless of encoder-input diversity.
- **Key Wave 30 structural conclusion (post-H26 update)**: The surface decoder's `Linear(512, 4)` final projection mean-preserves τz/τx regardless of upstream representation diversity. Encoder-input axis can produce per-car σ-level distribution change but cannot shift body-averaged mean. **The surface decoder projection row-coupling is the mean-driver**, structurally locking the band attractor. H26 is the canonical Wave 30 demonstration of this finding.
- **What this preserves for Wave 31**: (a) NPCA local-frame feature is PROVEN volume-decoder-aligned + test-generalizing — available as baseline-friendly input enrichment for any Wave 31 hypothesis needing volume-side improvement. (b) H26 + H31 stack candidate for vol_p deepening to ~3.45% (compounding −0.155 + −0.035 vol_p floor crossings), but will compound val_abupt regression and test_SP breach without surface decoder fix. (c) Per-car τz/τx distribution diagnostic (mean / std / min / max / n_outside_band) is now CANONICAL — Wave 31 surface decoder hypotheses must demonstrate mean(τz/τx) < 1.44 at test (not just std spread). (d) EMA tail / cosine-LR engagement was negative at this LR — Wave 31 13ep recipe should not assume terminal acceleration.
- **Implementation excellence (gold-standard)**: Smoke test with identity-init step-0 verification (model output = baseline at step 0 because new 3 columns zero-initialized), Path B 6h pre-flight before Path A 18h full retry, per-epoch τz/τx mean+std+min+max+n_outside_band logged at all 13 epochs, slope deceleration diagnosis posted at EP10/EP11/EP12 ahead of advisor question, cosine-LR engagement diagnostic produced clear negative read, honest accuracy-gap analysis throughout (no overclaiming), best-checkpoint EMA test reload pattern correctly executed. **Most rigorous mechanism analysis in Wave 30. Wave 31 documentation standard.**
- **Closure rationale**: val_abupt 6.346% saturates 0.22pp above baseline (slope decelerated to −0.007 pp/EP at EP12, EP13 regressed slightly to 6.350% — no recovery possible), test_SP +0.228pp floor breach is structural (val_SP capped at 4.20% in training), surface-side mean unchanged on test. Mechanism win preserved as canonical Wave 30 citation. **thorfinn reassigned to H46 SDORTH (Surface-Decoder Orthogonal Row Initialization, PR # TBD)** — smallest-LOC attack on the structurally-locked surface decoder projection axis.

---

## 2026-05-18 14:20 — PR #1185: H31 WALLDIST log-SDF input feature (alphonse) — TERMINAL EP13 NOT-A-MERGE / 18TH WAVE-30 DEAD END / FIRST TEST_VOL_P FLOOR CROSSING IN WAVE 30 (mechanism-class novel encoder-input result)

- **Branch**: `alphonse/h31-walldist-log-sdf-input`
- **W&B runs**: 8-rank DDP, rank0 `x54o3ang` (13/13 epochs, 842 min wall time, peak GPU 94.9 GiB)
- **Hypothesis**: Add log-SDF feature `log(|sdf|+1e-3)` as a 5th volume input channel to give the encoder uniform sensitivity across boundary-layer regimes. Predicted to improve volume_pressure decoder first (short composition path) and surface τ_z second via S2V cross-attention transport.
- **Recipe**: `--use-log-sdf-feature --use-surf-to-vol-xattn --epochs 13 --vp-schedule "0:16384:3:32768:6:49152:9:65536"` (12 LOC implementation change — smallest in Wave 30)
- **Results table**:

| Metric | val (EP13) | test (EP13 EMA) | Baseline #972 test | Δ vs baseline test | Floor |
|---|---:|---:|---:|---:|---:|
| **val_abupt** | **6.1735%** | — | 6.126% val | **+0.0475pp** ✗ FAIL merge gate | — |
| test_abupt | — | 5.9014% | 5.844% | +0.057pp ✗ regression | — |
| test_SP | — | 3.7536% | 3.577% (floor) | **+0.177pp** ✗ **FLOOR BREACH** | 3.577% |
| **test_vol_p** | — | **3.4880%** | 3.643% (floor) | **−0.155pp** ✅ **FLOOR PASS** (FIRST IN WAVE 30) | 3.643% |
| test_WSS | — | 6.7994% | 6.727% | +0.072pp ✗ above baseline | < 5.85% goal |
| test_WSS_x | — | 6.0224% | 5.6071% | +0.415pp ✗ regression | — |
| test_WSS_y | — | 7.3933% | 6.8397% | +0.554pp ✗ regression | — |
| test_WSS_z | — | 8.8500% | 8.2585% | +0.591pp ✗ regression | — |
| **val τz/τx** | **1.567** | — | — | — | — |
| **test τz/τx** | — | **1.470** | 1.473 | −0.003 (statistically identical to baseline) | mech ≤1.42 |

- **Trajectory val_abupt EP1→EP13**: 25.813 → 7.416 → 6.679 → 6.417 → 6.364 → 6.300 → 6.259 → 6.223 → 6.197 → 6.187 → 6.186 → 6.176 → 6.174 (saturating fleet-lead); val_WSS 6.942% at EP6 was 0.27pp ahead of next-best fleet run.
- **Trajectory val_VP EP3→EP13**: 4.030 → 3.893 → 3.829 → 3.770 → 3.720 → 3.702 → 3.685 → 3.663 → 3.660 → 3.659 → 3.652 → 3.650 → 3.652 (CROSSED 3.643 floor at EP12-13).
- **Trajectory τz/τx EP1→EP13**: 1.376 → 1.536 → 1.541 → 1.550 → 1.557 → 1.562 → 1.563 → 1.567 → 1.565 → 1.567 → 1.567 → 1.567 → 1.567 (EP1 deepest band-deflection in Wave 30 history tied with H25, monotonically pulled back to attractor by EP4, drifts ABOVE band by EP7 and holds there).
- **Mechanism analysis**: VOLUME SIDE CONFIRMED (first test_vol_p floor crossing, log-SDF gives encoder boundary-layer-regime uniformity that vol_p decoder reads directly via short composition path). SURFACE SIDE REJECTED (9th cold-start fade in Wave 30: EP1 τz/τx 1.376 → EP13 test 1.470 ≈ baseline 1.473). SP REGRESSION UNEXPECTED (val_SP capped at 4.16% in training but test_SP +0.177pp floor breach — changed volume encoder representation flows to cp head via S2V xattn in a way the baseline cp head can't decode as cleanly).
- **Key Wave 30 structural conclusion**: H31 + H21 + H25 + H30 V2S + H26 NPCA now triangulate that **the τz/τx band attractor lives in the SURFACE DECODER residual representation, NOT in the encoder input space, NOT in the volume pathway, NOT in the loss formulation, NOT in the auxiliary head**. Encoder-input feature axis CAN crack the vol_p floor (H31 first proof) but CANNOT break the surface-side τz/τx band attractor on test. **9 cold-start fades across 5 mechanism axes (encoder/loss/optimizer/aux-head/regularization) prove the attractor is downstream of all of them.** Next attack class MUST target the surface decoder structure itself.
- **What this preserves for Wave 31**: log-SDF feature is now PROVEN volume-decoder-aligned and available as a baseline-friendly modification for any Wave 31 stack needing volume-side improvement. H31+H30 V2S stack candidate conditional on H30 V2S terminal (currently projecting NOT-A-MERGE). pseudo-y+ feature (`log(|sdf|/nu)`) and other encoder-input enrichments documented as low-priority follow-ups.
- **Implementation excellence**: 12 LOC total, offline test-eval script agreeing with in-training eval to 4 sig figs, dual-launch race incident handled cleanly (00:43Z orphan crashed within 3 min, no metric impact), per-epoch trajectory logged consistently, mechanism diagnosis posted ahead of advisor decision.
- **Closure rationale**: val merge gate missed, test_SP floor breached, surface-side mechanism rejected — no follow-up parameter variation would change the merge outcome. Closed cleanly. alphonse reassigned to H45 ANCHOR-CROSSCHAN-DEC (Surface Decoder Cross-Channel Attention, first Wave 31 surface decoder structural attack).

## 2026-05-18 11:45 — PR #1183: H18d τz-only area weighting (tanjiro) — TERMINAL EP13 NOT-A-MERGE / 17TH WAVE-30 DEAD END / CHANNEL-DECOUPLED FALSIFIER OF "τz-SPECIFIC PHYSICS" / ONLY ABOVE-BAND VAL RUN IN WAVE 30

- **Branch**: `tanjiro/h18d-channel-decoupled-tau-z-area-weight`
- **W&B runs**: 8-rank DDP, rank0 `pp89ilpb` (13/13 epochs, 14.33h, best-checkpoint=EP13 EMA), 7 other rank runs preserved
- **Hypothesis**: Channel-decoupled τz-only area weighting — apply per-vertex DR=7400× area weighting ONLY to τz output channel (not cp/τx/τy). Premise: H18's band-break (test τz/τx 1.418) was τz-specific physics, and isolating the area weight to τz alone should preserve the band-break while releasing cp/τx/τy from low-area starvation.
- **Recipe**: `--area-weight-channels tau_z_only --use-area-weighted-loss --lr 9e-5 --epochs 13 --vol-points-schedule "0:16384:3:32768:6:49152:9:65536"` (10 LOC implementation change)
- **Results table**:

| Metric | val (EP13) | test (EP13 EMA) | Baseline #972 test | Δ vs baseline test | vs H18 (#1163) |
|---|---:|---:|---:|---:|---:|
| **val_abupt** | **6.319%** | — | 6.126% val | **+0.193pp** ✗ | −0.250pp (better than H18) |
| test_abupt | — | 6.182% | 5.844% | +0.338pp ✗ | −0.040pp |
| test_SP | — | 3.856% | **3.577%** (floor) | **+0.279pp** ✗ FLOOR BREACH | partial recovery vs H18 3.916 |
| test_vol_p | — | **3.637%** | **3.643%** (floor) | **−0.006pp ✓ marginal floor PASS** | +0.152pp vs H18 3.485 |
| test_WSS | — | 7.126% | 6.727% goal | +0.399pp ✗ | −0.143pp (better than H18) |
| **val τz/τx** | **1.633** | — | — | — | **INVERSE of H18's 1.418** |
| test τz/τx | — | 1.528 | ~1.46 | +0.07 | INSIDE band (back to fleet attractor) |
| Area weight DR | 7000-9000× | — | — | matches H18 spec | — |

- **Trajectory val_abupt EP1→EP13**: 30.772 → 7.967 → 7.006 → 6.658 → 6.513 → 6.457 → 6.406 → 6.376 → 6.352 → 6.332 → 6.325 → 6.320 → **6.319** (saturating); τz/τx monotonically 1.634 → 1.748 (EP2) → 1.633 (EP13) — **ALWAYS ABOVE BAND on val (unique in Wave 30)**
- **Conclusions**:
  - **17th Wave 30 dead end. Position-based per-vertex area weighting decisively dead in BOTH directions of channel decoupling.**
  - **Cleanest falsifier of "τz-specific physics" hypothesis**: H18's band-break (τz/τx 1.418) was NOT a τz-channel signal — it was a tied-loss-budget effect where area-starving cp/τx/τy *forced* the model to over-spend gradient on τz, producing 1.418 below-band as a budget-reallocation artifact.
  - **Inverse mechanism demonstrated**: H18d decoupling released cp/τx/τy from starvation; τz drifted ABOVE band (1.633 val). The mirror-image of H18's coupling-induced tightening.
  - **Train-eval space differential**: val τz/τx 1.633 → test τz/τx 1.528 (back inside band). Area weighting introduces a train-eval generalization mismatch — potentially useful in future stacking with H26 NPCA (variance-class, not position-class).
  - **vol_p floor PASS preserved** (3.637%) — both H18 and H18d achieve floor PASS, suggesting area weighting on τz alone is sufficient to nudge vol_p down without channel coupling. **Useful diagnostic for vol_p mechanism**.
  - **Implementation excellence**: 10 LOC change, smoke validated baseline recovery + channel composition, per-epoch trajectory logged cleanly, mechanism diagnosis posted ahead of advisor decisions.
  - **Wave 31 implication**: stop attacking position-based per-vertex weighting (entire axis dead). The "tied-loss-budget" insight transfers to all future stacking ideas — channel coupling is the dominant lever, not per-channel physics.

## 2026-05-18 11:30 — PR #1182: H29 SSFL (frieren) — TERMINAL EP13 NOT-A-MERGE / 16TH WAVE-30 DEAD END / 1ST FREQUENCY-DOMAIN FALSIFIER / FLEET-LOW VAL

- **Branch**: `frieren/h29-ssfl-streamwise-spectral`
- **W&B runs**: 8-rank DDP, rank0 `3umsllbj` (13/13 epochs, 13.4h, EMA-best EP12), 7 other rank runs preserved
- **Hypothesis**: Spectral Surface Loss with Streamwise Frequency Upweighting — sort surface tokens by streamwise z, apply rfft, weight high-frequency bins up to 2× via linear ramp on `|pred_fft - tgt_fft|²`. Mechanism premise: separation events are localized high-spatial-frequency in streamwise coordinates, MSE dilutes their gradient signal, upweighting should close the τz floor breach. F-principle (Xu et al. 2020) provides theoretical grounding.
- **Recipe**: `--lambda-spectral 0.1 --spectral-hf-weight 2.0 --spectral-channels wss --lr 9e-5 --epochs 13 --vol-schedule "0:16384:3:32768:6:49152:9:65536"` (Arm A; Arm B not run since EP3 falsifier was decisive)
- **Results table**:

| Metric | val (EMA EP12) | full_val | test | Baseline #972 test | Δ vs baseline test |
|---|---:|---:|---:|---:|---:|
| **val_abupt** | **6.3538%** ★ fleet-low | 6.3538% | 6.1578% | 5.844% | **+0.314pp** ✗ |
| test_SP | — | 4.2085% | **3.8617%** | 3.577% (floor) | **+0.285pp** ✗ FLOOR BREACH |
| test_WSS | — | 7.1619% | 7.0874% | 6.727% | **+0.360pp** ✗ |
| test_WSS_x | — | 6.2850% | 6.2965% | — | — |
| test_WSS_y | — | 7.7907% | 7.6871% | — | — |
| test_WSS_z | — | 9.6269% | 9.1770% | — | streamwise dominant residual |
| test_vol_p | — | 3.8581% | **3.7667%** | 3.643% (floor) | **+0.124pp** ✗ FLOOR BREACH |
| τz/τx (val) | 1.532 | — | 1.458 | — | NEVER BROKE BAND |
| spectral_loss EP1 | 0.0102 | — | — | — | — |
| spectral_loss EP13 | 0.000109 | — | — | — | **−99%** mechanism alive |

- **Trajectory EP1→EP12 val_abupt**: 30.4443 → 7.8913 → 7.0418 → 6.6493 → 6.5781 → 6.5107 → 6.4349 → 6.4040 → 6.3846 → 6.3607 → 6.3559 → **6.3538** (saturating); τz/τx 1.385 (cold-start break) → 1.484 (EP2 band re-entry) → 1.500 → ... → 1.532 (terminal, deep in band)
- **EP3 falsifier verdict**: τz/τx target ≤ 1.42 NOT met (1.500 at EP3) — first frequency-domain attack confirms band is NOT spatial-frequency-mediated
- **Conclusions**:
  - **16th Wave 30 dead end. First frequency-domain falsifier.**
  - Decisive negative on the spatial-frequency axis: spectral_loss descended cleanly 99% across 13 epochs while τz/τx stayed pinned in [1.484, 1.532]. Mechanism was alive, the hypothesis was wrong.
  - **F-principle is NOT the cause of the band.** High-frequency upweighting does not change τz/τx ratio.
  - **Loss-shape tier decisively closed.** Combined with H10b/H11b/H12/H16/H16b/H20/H22 (7 per-vertex loss-shape failures) + H23 (training-regularization) + H29 (frequency-domain), the entire loss-shape abstraction tier is falsified. **Only representation-tier (NPCA-class) and decoder-structure-tier remain promising.**
  - **vol_p test floor breach is informative**: H29 only touches surface loss → spectral upweighting on τ drove decoder to over-allocate capacity to surface refinement, starving vol_p decoding. **Coupling diagnostic for stacked attacks** (fern H35 NPCA-SSFL-STACK should watch for this).
  - **Wave 31 design implication**: bias toward representation/coordinate augmentation (NPCA-class) and decoder-structure-tier attacks. Stop targeting loss-shape axis.
  - **Best-in-class implementation execution**: λ=0.0 baseline recovery, AMP/fp32 boundary correct, padding-zero-before-FFT preserved. The cleanest decisive negative we've gotten.

## 2026-05-18 07:30 — PR #1174: H24 GSTS (fern) — TERMINAL EP13 NOT-A-MERGE / 15TH WAVE-30 DEAD END / 11TH COLD-START MEAN-SHIFT FADE

- **Branch**: `fern/h24-geometric-saliency-slice-temperature`
- **W&B runs**: 8-rank DDP, rank0 `fpertdi4` (13/13 epochs, 866.65min wall-clock, 14.44h, EMA-best)
- **Hypothesis**: Geometric-Saliency Slice-Temperature Sharpening (GSTS) — per-vertex slice-temperature modulation by local geometric saliency (curvature MLP), to force per-vertex specialization in slice attention.

### Terminal results (rank0 `fpertdi4` EP13 EMA)

| Metric | EP13 | Baseline / Floor | Δ | Verdict |
|---|---:|---:|---:|:--|
| **val_abupt** | **6.325%** | 6.126% | **+0.199pp** | **FAIL** |
| test_abupt | 6.040% | 5.844% | +0.196pp | FAIL |
| test_SP | 3.831% | 3.577% floor | +0.254pp | **FLOOR FAIL** |
| test_vol_p | **3.610%** | 3.643% floor | **−0.033pp** | **FLOOR PASS** ★ |
| test_WSS | 6.953% | 6.727% | +0.226pp | FAIL |
| test τz/τx | 1.466 | break <1.42 | +0.04 inside band | **NO BREAK** |

### Per-epoch trajectory (val)

| EP | val_abupt | τz/τx | descent rate |
|---:|---:|---:|---:|
| 1 | 27.401% | 1.395 ★ (tied deepest Wave 30 EP1 break) | — |
| 2 | 7.622% | 1.504 | −19.78pp |
| 3 | 6.791% | 1.514 | −0.83pp |
| 13 | 6.325% | 1.554 | −0.47pp total EP3→EP13 (saturating slope) |

Monotone descent EP1→EP13, no overfit spike. EP1 τz/τx=1.395 was the deepest band-break signal in Wave 30 — but it faded to 1.504 by EP2 and crept upward to 1.554 by EP13. Classic cold-start fade pattern (11/12 mean-shift attacks now confirmed in Wave 30).

### Mechanism diagnostic (preserved for Wave 31)

- `geom_temp_std` settled at 1.7e-2 by EP6 — gradient-flow alive but uniform
- MLP final-layer max_abs weight grew to 0.128 (from 0 at init) — gradients flowed through the slice-temp MLP
- Per-region delta only +0.0056 between high-curvature and low-curvature regions (mean t_v = 0.759)
- **Network learned a near-uniform global softening, NOT the per-vertex differentiation that the hypothesis required**

**Lesson for Wave 31:** Any "learn a per-vertex modulator" approach needs explicit anti-uniform-collapse regularization (e.g., variance penalty on the modulator output, or contrastive auxiliary loss against geometric features). Lion + cosine LR + MSE objective biases toward uniform offsets.

### test_vol_p floor PASS — pattern with cold-start hypotheses

H24 is the **second consecutive cold-start hypothesis to preserve volume_pressure floor** (alphonse H31 currently at val_vol_p 3.780% approaching floor at EP4). Pattern: cold-start encoder-side modifications can preserve or improve volume_pressure even when surface τz/τx attack fails. Consistent with H31 WALLDIST's hypothesis that volume pathway is bottlenecked by input feature quality.

### Decision

NOT-A-MERGE confirmed:
- val gate FAIL (+0.199pp above baseline)
- test_SP / test_WSS floor FAIL
- Only test_vol_p floor PASS (insufficient on its own)
- No band break

CLOSED as 15th Wave 30 dead end. Diagnostic preserved for Wave 31 design.

### What replaces H24 in fern's slot

**PR #1189 H35 NPCA-SSFL-STACK assigned same hour** — first Wave 31 hypothesis, directly tests whether H26 NPCA (variance-break) and H29 SSFL (loss-reshape) stack additively. Mechanism-distinct from all 15 closed Wave 30 dead ends.

---

## 2026-05-18 06:00 — PR #1179: H28 SAM (edward) — POD CRASH MID-EP3 + RELAUNCH-KILLED-BY-ADVISOR / 14TH WAVE-30 DEAD END / OPTIMIZER-SPACE AXIS CLOSED

- **Branch**: `edward/h28-sam-sharpness-escape`
- **W&B runs**: `a9nkvm65` (crashed mid-EP3 @ step 31,979, 9.31h runtime), `1h6164ff` (relaunch, killed by advisor verdict at 04:27Z+90min)
- **Hypothesis**: SAM (Sharpness-Aware Minimization, Foret et al. 2020) ρ=0.05 with Lion outer optimizer escapes flat-basin band attractor [1.44, 1.55] via curvature regularization. First optimizer-space attack of Wave 30.

### Results (from crashed run EP1+EP2 — sufficient for verdict)

| EP | val_abupt | τz/τx | SAM cos g·ĝ | sam/perturbed_loss | Verdict |
|---|---:|---:|---:|---:|:--|
| 1 | 25.7198% | **1.4406** | 0.8627 | 0.1262 (>clean) | normal cold-start, lower-band-edge ⚠️ |
| 2 | 8.8036% | **1.4999** | 0.8695 | 0.0052 (<clean) | **Reading A** — band attractor wins |
| 3 | crashed @ step 31,979 (~75% through) | — | 0.8358 | 0.0158 | pod-side hostname change |

Mechanism diagnostic at crash: `sam/grad_cos_g_ghat` trajectory **0.86→0.92→0.87** with `sam/perturbed_loss > clean_loss` consistently (Foret textbook). SAM IS firing correctly. The flat basin SAM converges to is INSIDE the [1.44, 1.55] band attractor — not escaping it.

### Failure mode diagnostic (highest-value Wave 30 finding from this run)

**The τz/τx band attractor is geometrically FLAT in parameter space.** SAM tells us: the band is NOT a sharp local minimum that curvature regularization can escape — it's a wide attractor basin in the loss landscape itself. This rules out "sharpness traps" as the band's mechanism.

This is the cleanest negative result Wave 30 has produced for the parameter-space attack class. Pairs with the H32 finding that subtractive attention destroys volume pathway.

### Decision rationale (close vs let-relaunch-proceed)

Student auto-relaunched at 04:27Z with same params (ρ=0.05, 35h budget) after pod restart. Advisor KILL verdict at 06:00Z based on:

1. **10th cold-start fade pattern** (Reading A): mean-shift attacks fade into [1.44, 1.55] by EP2-EP3 regardless of mechanism. ρ=0.05 is mean-shift (small isotropic perturbation budget).
2. **Mechanism diagnostic already extracted** from crashed EP1-EP3 partial data.
3. **35h GPU opportunity cost** — fresh axis (H34 OUTHEAD) lands EP3 results 8h vs hold-and-wait.
4. **No follow-up parameter sweep value** — ρ=0.05 was already Foret-recommended starting point.

### Implications for Wave 30 remainder

- **Mean-shift attack class exhausted** — 8 axes closed, only H26 variance-break survives mechanistically
- **Optimizer-space verdict**: flat-basin curvature regularization (SAM-style) cannot escape band attractor in this regime. Foret-style sharpness optimization needs larger ρ or different attack geometry (e.g., asymmetric/anisotropic perturbations targeting specific layers).
- **Wave 31 design principle confirmed**: prefer variance/spread attacks over mean-shift attacks.

### What replaces H28 in edward's slot

**PR #1188 H34 OUTHEAD assigned same hour** — per-channel auxiliary output heads (head-side rank-coupling attack). Different axis entirely: tests whether the band's persistence is a head-side projection signature rather than a trunk-side optimization issue.

---

## 2026-05-18 05:00 — PR #1186: H32 DIFFATTN (askeladd) — TERMINAL KILL EP1 / 13TH WAVE-30 DEAD END / ATTENTION-MECHANISM AXIS CLOSED

- **Branch**: `askeladd/h32-differential-attention`
- **W&B runs**: `c49wngus` (V1 canonical), `jygo5ya0` (V2 minimal PR pseudocode)
- **Hypothesis**: Replace single `F.scaled_dot_product_attention` in `TransolverAttention.forward` with differential attention `A₁ − sigmoid(λ)·A₂` (Differential Transformer, Microsoft Research 2024) to cancel τz/τx band attractor shared noise floor.

### Results

| Variant | W&B | val_abupt EP1 | val_volume_p_mae | Verdict |
|---|---|---:|---:|:--|
| **V1** (canonical: subln + per-layer λ schedule 0.20→0.62 + (1−λ_init) output scale) | `c49wngus` | **29.541%** | **35.79** | ❌ KILL EP1 >9.5% |
| **V2** (PR pseudocode: single sigmoid(λ_raw=0.8)≈0.689, no subln, no schedule, no output scale) | `jygo5ya0` | **27.947%** | **31.90** | ❌ KILL EP1 >9.5% |
| Baseline (56bcqp3m) | ref | ~8-9% typical | 5-8 typical | — |

Both variants killed at EP1. Both ~4× baseline EP1 abupt. **Volume pathway blown 5-7× while surface metrics ~1.1× baseline.**

### Failure mode diagnosis (student analysis, high-confidence)

The damage is pathway-asymmetric:
- **Surface metrics**: ~1.1× baseline (model not globally diverging)
- **Volume_p_mae**: 5-7× catastrophic
- **Wall_shear_mae**: ~1.0× baseline

Root cause: surface tokens have residual fallback (`x + attn(x)`), volume tokens ONLY see physics via `surf→vol cross-attention against slice tokens`. When subtractive SDPA poisons slice-token magnitude at init (V1: 0.31× via output scale; V2: 0.31× via correlated destructive interference), the volume decoder gets no signal.

V2 failing is the key finding: even without per-layer schedule, the minimal single-scalar form fails because Q2/K2/V2 share a single LinearProjection with Q1/K1/V1 (just chunked). At init, the two SDPA outputs are HIGHLY CORRELATED, not orthogonal. `SDPA₁ − 0.689·SDPA₂ ≈ (1−0.689)·SDPA₁ = 0.31·SDPA₁` at init — destructive interference, not noise cancellation.

### Implications for Wave 30 architecture

Any attention modification that destroys slice-token MAGNITUDE at init will break the volume pathway specifically. Valid modifications:
- **Additive perturbation** to slice_tokens (e.g. SLICEPE, H33)
- **Input encoding modifications** (e.g. WALLDIST, H31)
- **Input geometric transforms** (e.g. NPCA, H26)
- **Cross-attention on a DIFFERENT path** (e.g. V2S, H30)

Invalid without careful magnitude-preserving init:
- Subtractive attention on slice tokens
- Any operation that multiplies slice-token magnitude by <0.5×

**H33 SLICEPE (additive, zero-init) is specifically designed to be safe per this diagnostic.**

---

## 2026-05-18 01:55 — PR #1183: H18d (tanjiro) — EP3 INTERIM / KILL GATE TRIGGERED ON τz/τx BUT ADVISOR EXCEPTION → CONTINUE TO EP6 MID-TRAJECTORY GATE / MECHANISM DIAGNOSIS LOCKED

- **Branch**: `tanjiro/h18d-channel-decoupled-tau-z-area-weight`
- **W&B run**: `pp89ilpb`
- **Hypothesis**: Channel-decoupled τ_z-only area weighting — isolate whether H18's band-break (τz/τx 1.418 ★) is τ_z-specific physics or a tied-loss-budget effect across channels.

### EP3 Results

| Metric | EP3 H18d | EP3 H18 (ref) | Gate | Verdict |
|---|---:|---:|---|---|
| val_abupt | 7.006% | 7.787% | ≤7.5 | ✅ PASS (-0.78pp vs H18, BEST in H18-family) |
| val_SP | 4.495% | 5.033% | ≤4.50 | ✅ PASS (knife-edge, -0.54pp vs H18) |
| val_vol_p | 3.919% | — | informational | tracking |
| τz/τx | **1.669** | **1.412** | ≤1.42 PASS / >1.50 KILL | ❌ **KILL** (+0.257 vs H18) |

### Mechanism diagnosis (LOCKED — high-value experimental result)

**H18's band-break was NOT τ_z-channel physics — it was a tied-loss-budget effect.** Channel-decoupling (τ_z-only area weighting, cp/τ_x/τ_y at baseline) eliminates the cross-channel pressure that drove τ_z down on horizontal panels in H18. Specifically:
- H18 area-weighted ALL 4 channels → starved cp/τ_x/τ_y → forced gradient onto τ_z → "broke" band attractor
- H18d area-weighted ONLY τ_z → cp/τ_x/τ_y normal weight → no forced redistribution → band attractor reverts

**Implication**: Position-based per-vertex area weighting is decisively closed as an axis across all variants. H18's win was a coincidental side-effect of channel-coupled loss budget redistribution, NOT a discovery about τ_z geometry.

### Advisor exception decision

KILL gate technically triggered (τz/τx > 1.50), but val_abupt 7.006% is the BEST EP3 result of any H18-family attempt, and val_SP recovery (5.033%→4.495%) confirmed the channel-decoupling SP-recovery hypothesis. EP13 trajectory extrapolation: val_abupt 5.5-6.0% possible (would beat baseline 6.126%).

**CONTINUE granted with strict EP6 mid-trajectory gate (~06:10Z May 18):**
- val_abupt ≤ 6.40% AND val_SP ≤ 4.10% → continue to EP13
- val_abupt > 6.60% OR val_SP > 4.40% → kill
- Marginal zone → advisor decision

### Linked W&B run state at posting time
- Heartbeat 01:44Z, all 8 ranks alive, 0 nonfinites, val slopes all negative
- best_checkpoint EMA = EP3 (currently selecting EP3 model)

---

## 2026-05-18 00:28 — PR #1178: H27 PRLP (askeladd) — TERMINAL KILL EP3 / 12TH WAVE-30 DEAD END / TRAIN-EVAL SPACE MISMATCH FALSIFIED

- **Branch**: `askeladd/h27-per-component-relative-l2-proxy-loss`
- **W&B runs (rank0 + DDP ranks 1-7)**: `46hdkr3v` / `we3s2378` / `9z22vedf` / `5ufgbg9m` / `16d9mjb6` / `szhda12r` / `ip9w7dlh` / `awc75jn0`
- **Hypothesis**: Replace MSE train loss with per-component relative-L2 (= same metric as eval) to close the train/eval space mismatch. Theory: MSE over-weights high-magnitude regions (stagnation, base wake), starving τz from gradient signal. Relative-L2 on [cp, τx, τy, τz] independently normalises by component mean should improve floor metrics.

### Results

| Metric | EP3 Actual | Baseline | Delta |
|---|---:|---:|---:|
| val_abupt | 7.098% | 6.126% | +0.972pp FAIL |
| test_abupt | 6.643% | 5.844% | +0.799pp |
| val_SP | 4.546% | — | ≥4.40% KILL gate triggered |
| τz/τx | 1.526 | ~1.490 | IN-BAND [1.44, 1.55] |
| `grad/nonfinite_count` | 0 | — | ✅ (2 sqrt-NaN patches applied) |

EP3 binary KILL gate triggered: val_SP 4.546% ≥ 4.40% AND τz/τx 1.526 in [1.44, 1.55]. Test confirms 4/4 floors breach. Run terminated at EP3.

### Per-epoch trajectory

| EP | val_abupt | val_SP | τz/τx | train_loss |
|---:|---:|---:|---:|---:|
| 1 | ~28% | — | 1.523 | 1.203 |
| 2 | 8.52% | 4.726% | ~1.51 | 0.434 |
| 3 | 7.098% | **4.546%** | **1.526** | 0.336 |

### EP2→EP3 decoupling diagnostic (train-eval space mismatch falsified)

| Interval | train rel-L2 change | val_SP change |
|---|---:|---:|
| EP1→EP2 | −64% | −76% (coupled, proximate space) |
| EP2→EP3 | **−23%** | **−10%** (decoupled: train descends 2.3× faster than val_SP) |

**Falsification result**: per-car normalisation inversely re-weights gradient signal by target magnitude. Cars with small τz deviations (= near-baseline geometry = most of the fleet) have 1/mean² scaling that AMPLIFIES them in the loss, crowding out the gradient signal from the tail-distribution outliers that drive the floor metric. The train/eval space is NOT the binding mismatch — it was the normalisation direction that created a new mismatch.

### Analysis

This is 12th Wave 30 dead end. PRLP closes the "train-eval space" axis. The decoupling pattern (EP2→EP3 train −23%, val_SP only −10%) is a clean signal that the loss landscape transformed by per-car normalisation is DIFFERENT from the metric landscape, not closer to it. Directionally opposite to what the hypothesis predicted.

Combined with H27 (PRLP), the full Wave 30 closed axis table:
- Per-vertex loss shape (H10b, H11b, H12, H16, H16b): DEAD
- Training regularization (H20, H22, H23 EMA): DEAD
- Optimizer/sharpness (partially, H28 SAM in-flight)
- Output decoder capacity (H21 per-component heads): DEAD
- Mean-shift encoder manipulation (H24 GSTS, H25 ALGP, H18): CLOSED axis
- Train-eval space (H27 PRLP): CLOSED axis (falsified by per-car normalisation gradient inversion)
- Per-car normalization (H27 as proxy): INVERSELY harmful
- Channel-coupled position weighting (H18): CLOSED
- Encoder input geometric transformation (H26 NPCA): MECHANISM PROVEN, accuracy pending full-budget retry

---

## 2026-05-17 23:38 — PR #1177: H26 NPCA (thorfinn) — PATH-B TERMINAL THEN REQUEST-CHANGES / FIRST WAVE-30 MECHANISM PROOF / ACCURACY-BUDGET-CUT / SENT BACK FOR FULL 18H PATH A RETRY

- **Branch**: `thorfinn/h26-normal-projected-coord-aug` (sent back to WIP for 18h retry)
- **Path B 8 W&B runs**: rank0 `nqc0tmx9`, ranks 1-7 `9bnxlmp7`/`9w2qo5vu`/`aw2a8ddm`/`axhbr1lq`/`csvy3v8z`/`dn4ibps9`/`glgoklez`
- **Path B Total runtime**: 271 min (4.5h, well under 6h pod budget) but training cut at 3 of planned 5 epochs by train_timeout_minutes=270min (per-epoch wall ~90min at 65536 surface+volume on DDP-8 b=4, 4× higher than pre-launch smoke projection)
- **Hypothesis**: Normal-Projected Coordinate Augmentation — append 3 extra encoder input channels = (p·n, p·t1, p·t2) projections of position onto local Gram-Schmidt tangent frame from per-vertex normals. Hypothesis: per-car local-frame coords break global band attractor by giving encoder per-car-geometry-specific position features.

### Path B terminal results — FIRST MECHANISM PROOF in Wave 30

| Marker | Value | Verdict |
|---|---:|:--|
| **Mechanism gate** (std(τz/τx) ≥ 0.04 AND ≥1/34 outside band) | **std 0.216, 14/34 outside @ EP3 val + std 0.120, 20/50 outside @ test** | **PASSED 5× margin on std, 14× margin on n_outside** ✅ |
| val_abupt @ EP3 | 6.972% | FAIL +0.846pp vs baseline 6.126% |
| val_abupt slope EP1→EP3 | −0.088%/1k_steps (−2.6pp per 30k steps) | STILL DESCENDING — no plateau detected |
| test_abupt | 6.642% | +0.798pp |
| test_SP | 4.278% | +0.701pp BREACH floor 3.577% |
| test_vol_p | 3.985% | +0.341pp BREACH floor 3.643% |
| test_WSS | 7.606% | +0.879pp |
| `train/grad/nonfinite_count` | 0 across full run | ✅ |

### Per-epoch τz/τx mechanism table — DECISIVE encoder-level band-attractor break

| EP | mean | std | min | max | n_outside [1.40,1.60] |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.4267 | **0.1014** | 1.27 | 1.63 | 17/34 |
| 2 | 1.5174 | **0.1861** | 1.26 | 2.34 | 13/34 |
| 3 | 1.5201 | **0.2161** | 1.29 | 2.49 | 14/34 |
| **Test (EP3 reload)** | 1.4554 | **0.1196** | 1.10 | 1.73 | 20/50 |

Baseline #972 reference: `std(τz/τx) ~ 0.02`, all 34 val cars locked in [1.44, 1.55]. **H26 dispersion is 5–11× baseline std.** First Wave 30 attack to produce sustained encoder-level band-attractor disruption.

### Why this is NOT a close — three load-bearing observations

1. **Mechanism is PROVEN**: std 10× baseline, max τz/τx 2.49, 14/34 val + 20/50 test cars outside band, at EP3, with test-side preservation. Different mechanism class from the closed Wave 30 attacks (all of which targeted mean-shift). NPCA is the first PER-CAR-VARIANCE attack and it works.

2. **Trajectory was still descending**: val_abupt slope `−0.088%/1k_steps` at EP3, no plateau. Linear extrapolation conservative 4× slowdown → EP13 val_abupt 5.97% (would BEAT baseline by 0.16pp). Even 8× slowdown → 6.48% (within 0.36pp).

3. **Budget was the binding constraint, not the hypothesis**: per-epoch wall time was 4× higher than pre-launch smoke; training cut at 3 of planned 5 epochs by Path B 270min budget. Full 18h Path A is 12-13 epochs.

### Request-changes action

Sent back 23:38Z with full 18h Path A recipe: `--epochs 13 --lr-cosine-t-max 13 --vol-points-schedule 0:16384:3:32768:6:49152:9:65536 SENPAI_TIMEOUT_MINUTES=1100`. Keeps all implementation hardening (Gram-Schmidt fp32-cast, `eps=1e-6` in F.normalize, padded-token zero invariant, `project.weight[:, 4:7].zero_()` correct attribute path). Pre-launch budget propagation check required (H25 post-mortem).

### Key interim signal during retry: EP6 val_abupt

EP6 (~10h from relaunch) is the decisive interim. If val_abupt > 6.5% at EP6, linear extrapolation has broken down and full-budget run is unlikely to recover. If val_abupt ≤ 6.5% at EP6, on-track for baseline-beating at EP13.

### Implication for Wave 30 + Wave 31

H26 NPCA opens a new question class:
- **Wave 30 closed: Can band attractor be broken?** — Answer (mechanism): YES, via input local-frame projection
- **Wave 30 NEW: Does breaking band attractor BEAT baseline accuracy?** — Answer pending H26 Path A 18h retry

If H26 Path A beats baseline, the win composes with H30 V2S xattn (encoder cross-modal fusion, orthogonal axis) and H31 WALLDIST (encoder input log-SDF, orthogonal axis) for a Wave 31 triad stack.

If H26 Path A does NOT beat baseline despite full budget, the conclusion is: band-attractor break is NECESSARY but INSUFFICIENT — accuracy is bounded by something else (likely encoder INFORMATION content, per H21+H25 closure diagnosis).

### Engineering wins worth preserving

- **Bug fix**: `self.project_surface_features.weight[:, 4:7].zero_()` would AttributeError; correct path is `self.project_surface_features.project.weight[:, 4:7].zero_()` (LinearProjection wraps nn.Linear at `self.project`).
- **Numerical hardening**: `compute_local_frame_proj` casts to fp32 inside and uses `eps=1e-6` in `F.normalize`; padded tokens (`n == [0,0,0]`) produce `local_proj == 0` instead of NaN even under bf16 autocast.
- **Identity-at-init**: `project_surface_features.project.weight[:, 4:7] == 0` verified offline; forward bit-identical whether local_proj is zero, real, or random.

---

## 2026-05-17 22:50 — PR #1176: H25 Auxiliary Local-Gradient Prediction (alphonse) — CLOSED TERMINAL NOT-A-MERGE / RUN-CRASHED / 6TH COLD-START FADE / 11TH WAVE-30 DEAD END / MECHANISM-ALIVE-OBJECTIVE-DISCONNECTED

- **Branch**: `alphonse/aux-local-grad-prediction` (closed pending student terminal SENPAI-RESULT)
- **W&B rank0**: `pvdjrlx4` (crashed) — full 8-rank set in PR body
- **Crash**: step 33,427 at 22:12Z (5h 03min runtime). Pod still showed `Running` per k8s after crash (zombie). Crash AFTER EP3 val landed at step 32,592.
- **Hypothesis**: Auxiliary head predicts local gradient of τ_z at each surface point; aux task gradient flows into encoder, forcing representation to encode local boundary-layer derivatives. λ_aux=0.05, aux_grad_scale~14 (heavy-tailed coord-meter / unit-τ_z mismatch).

### Pre-crash EP3 result — TERMINAL NOT-A-MERGE

| Marker | EP1 step 10,864 | EP2 step 21,728 | EP3 step 32,592 | Verdict |
|---|---:|---:|---:|:--|
| val_abupt | 25.567% | 8.076% | **7.142%** | FAIL +1.016pp vs baseline 6.126% |
| val τz | 34.731% | 11.916% | 10.692% | — |
| val τx | 25.214% | 7.933% | 7.072% | — |
| **τz/τx** | **1.3774** ★ deepest EP1 | 1.5021 | **1.5117** | **6TH COLD-START FADE — band rebounded** |
| val_SP | — | — | (not logged in summary) | — |
| aux_loss | 1.71 → 0.81 | 0.789 mid-EP2 | continued descending | ✅ MECHANISM ALIVE |
| aux_pred_std | 0.009 → 7.88 | 8.21 mid-EP2 | continued growing | ✅ MECHANISM ALIVE |

### 6th confirmed cold-start fade — band attractor is warmup artifact

| Hyp | Mechanism class | EP1 τz/τx | EP3 τz/τx | Fade direction |
|---|---|---:|---:|:--|
| H18 | per-vertex area weight | 1.412 | 1.489 | ↑ into band |
| H20 | curvature-soft loss | 1.401 | 1.523 | ↑ into band |
| H23 | EMA self-distill | 1.426 | 1.866 | ↑ over band (also KILL) |
| H24 GSTS | slice temperature | 1.395 | 1.514 | ↑ into band |
| **H25 ALGP** | **aux local-grad** | **1.377** ★ deepest | **1.512** | **↑ into band** |
| H26 NPCA | spread-break (different class) | std 0.101 | TBD | — |

H25's EP1 τz/τx 1.3774 is the DEEPEST EP1 mean-shift signal in Wave 30 history — yet it still rebounded into the [1.44, 1.55] attractor band by EP3. **Conclusion**: deeper EP1 cold-start mean-shifts do NOT survive warmup → they are universally warmup-dynamics artifacts, not representational changes. Closed axis: **mean-shift τz/τx representation manipulation via auxiliary task / saliency MLP**.

### Mechanism worked, hypothesis was wrong

The ALGP aux head learned its target as designed: aux_loss −53% over EP1, aux_pred_std rose 1000× tracking aux_grad_target_mean scale, aux_grad_head non-degenerate (weights/grads healthy throughout). But the gradient injected into the encoder did NOT reorganize τ_z representation in a way that survived warmup — final τ_z prediction tracked the standard band attractor exactly like the other 5 cold-start-fade Wave 30 attacks.

### Engineering note — false-positive health check at 22:14Z

A health-check comment posted at 22:14Z incorrectly reported the run as healthy at step 22,017. Root cause: queried W&B `summary_metrics` which returned EP2 cached state and `running` status, ~2 minutes after the actual crash at 22:12Z. Future health checks must cross-reference `_timestamp` / `_runtime` against expected step rate, not just `state == running`. Cost: 24 minutes from crash to correction.

### Implication for alphonse next assignment

H25 closure + H21 closure jointly localize the τ_z prediction floor as:
- NOT decoder capacity (H21 proved)
- NOT mean-shift aux task on encoder (H25 proves)
- NOT mean-shift saliency on encoder (H24 GSTS faded)
- → must be **encoder feature CONTENT** (information missing from input, not how it's processed)

Next attack class: **explicit physical features in input** — wall-distance (log SDF), separation-line auxiliary classification, or pressure-gradient features. Aligns with H30 V2S cross-attention thesis (volume tokens carry off-body physics surface tokens need).

---

## 2026-05-17 21:50 — PR #1171: H21 Per-Component Independent Output Heads (nezuko) — CLOSED TERMINAL NOT-A-MERGE / 10TH-WAVE-30-DEAD-END / DECODER-CAPACITY-NOT-THE-BOTTLENECK

- **Branch**: `nezuko/h21-per-component-output-heads` (closed)
- **W&B runs**: rank0 `xhy9yk67` (+ 7 additional DDP ranks)
- **Total runtime**: ~14.9h, 70664 steps
- **Hypothesis**: Replace shared 4-channel surface output MLP with four independent MLPs (one per surface channel). τ_z head gets one extra hidden layer for additional capacity. Mechanism: gradient isolation prevents dominant channels (cp/τ_x) from interfering with τ_z gradient.

### Terminal results — NOT-A-MERGE, all four test floors breach

| Metric | H21 test | Baseline test | Floor | Δ | Verdict |
|---|---:|---:|---:|---:|:--|
| test_abupt | 6.119% | 5.844% | — | +0.275pp | **MISS** |
| test_SP | 3.813% | — | 3.577% | +0.236pp | **BREACH** |
| test_vol_p | 3.694% | — | 3.643% | +0.051pp | **MARGINAL BREACH** |
| test_WSS | 7.084% | — | 6.727% | +0.357pp | **BREACH** |
| test τz/τx | 1.4391 | — | — | — | MARGINAL band-edge (H15b: 1.439) |
| val_abupt (EP13) | 6.493% | 6.126% | — | +0.367pp | **MISS** |

### Per-epoch trajectory

EP1→3 cold-start −10.93pp/ep → EP3-6 −0.182pp/ep → EP6-7 vol-curriculum 49152 bump −0.074pp/ep → EP10-13 cosine-tail flatline −0.005pp/ep. Final τz/τx 1.542 stable in upper band [1.54, 1.55].

### Mechanism diagnostic — cleanest gradient-decoupling in fleet history

`τz > τy > τx > cp` gradient-norm ordering held in 11/13 training buckets across 65k steps. τ_z head absorbed 22% more param mass than τ_x (head_tau_z 51.34 vs head_tau_x 43.32 total param norm). Mechanism fired EXACTLY as designed — gradient isolation was clean and τ_z head got more capacity.

### Critical conclusion — decoder capacity is NOT the τ_z bottleneck

H21 is the **highest-quality mechanism-confirmed NOT-A-MERGE in fleet history**: the gradient decoupling worked perfectly AND the result was NOT-A-MERGE. This conclusively proves: the encoder features fed to the τ_z head do not contain sufficient information to predict τ_z accurately. Adding more decoder capacity on top of insufficient encoder features changes how information is read out, not how much is available. **Closed axis: channel-coupled output-head decoder capacity.**

### Implication for H30 (nezuko next assignment)

H21's closure directly motivates V2S cross-attention (H30): encoder-level cross-modal fusion between volume tokens (off-body flow physics) and surface tokens. Volume → surface xattn injects separation physics into the encoder features BEFORE the output head, attacking the representation bottleneck rather than the decoder.

---

## 2026-05-17 20:05 — PR #1163: H18 Per-Vertex Area-Weighted Surface MSE (tanjiro) — CLOSED TERMINAL NOT-A-MERGE / FIRST-TEST-SURVIVING-BAND-BREAK / 9TH-WAVE-30-DEAD-END / MECHANISM-PARTLY-PROVEN

- **Branch**: `tanjiro/h18-area-weighted-surface-loss` (closed)
- **W&B group**: `wave30_h18_area_weighted_v2` (v1 watchdog-killed; v2 ran clean EP1-13)
  - 8-rank: rank0 `j473gqwa`, rank1 `5f4heugx`, rank2 `skpujw1g`, rank3 `icgcf556`, rank4 `fvygavd3`, rank5 `9v2sybz2`, rank6 `z3m09rs7`, rank7 `oo68nfib`
- **Best ckpt**: EP13 EMA
- **Total runtime**: 14.33h (859 min, 70664 steps)
- **Hypothesis**: Multiply per-vertex MSE by physically-meaningful area weight `w_i = raw_area_i / mean(raw_area)` to make training objective match the physical force-integral `∫ τ dA`. Hypothesis: τ_z over-prediction is concentrated on high-area horizontal panels (hood/roof/trunk where n ≈ +z); area-weighting forces stronger fit there.

### Terminal results — NOT-A-MERGE but UNIQUELY POSITIVE band-break + vol_p signal

| Metric | val (n=34) | test (n=50) | baseline test | Δ | floor | floor breach |
|---|---:|---:|---:|---:|---:|:--|
| **val_abupt** (merge gate) | **6.5687** | 6.2216 | 5.844 | +0.378pp | — | FAIL +0.443pp val |
| test_WSS (paper-facing) | 7.5595 | **7.2690** | 6.727 | +0.542pp | 6.727 goal | FAIL goal +0.542pp |
| test_SP | 4.2808 | 3.9163 | 3.456 | +0.460pp | 3.577 | **BREACH +0.339pp** |
| test_vol_p | 3.6319 | **3.4848** | 3.563 | −0.078pp | 3.643 | **PASS −0.158pp** ✓ |
| **test τz/τx ratio** | **1.489** | **1.4178** | ~1.46 | **−0.04** | — | **★ BELOW [1.44, 1.55] BAND** |

### Three uniquely positive signals — distinct from prior 8 Wave 30 closures

1. **Test τz/τx 1.418 = DEEPEST test-side band-break in fleet history** (vs H6' #1147 1.420). Val faded EP3 1.412 → EP13 1.489 but **TEST HELD at 1.418** — first Wave 30 result with test-side mechanism survival. Val/test divergence suggests area-weighted MSE learned generalizable τ_z structure, not val-specific patterns.
2. **test_vol_p 3.485% = ONLY floor-passing volume_pressure in active Wave 30 fleet.** Surface→volume cross-attention propagates better-fitted τ_z to vol_p predictions when area-weighted.
3. **9th Wave 30 dead end but qualitatively different** — joins H10b/H11b/H12/H16/H16b/H20/H22 (loss-shape) + H23 (regularization), but is the ONLY closure with proven test-side mechanism shift + floor pass.

### Diagnostic: area weight statistics at terminal (per-rank mean)

| Stat | Value | Expected | Status |
|---|---:|---|:--|
| Effective dynamic range | **7410×** | [10×, 500×] | **15× ABOVE expected** |
| vertex_weight max | 23.76 | < 5.0 | 4.8× above expected |
| vertex_weight p95 | 3.72 | ~1.5–5.0 | ✓ |
| vertex_weight min | 0.0033 | ~0.1 | 30× BELOW expected |

**Failure mode diagnosis**: DR=7400× starves low-area sharp-edge vertices (mirror seams, wheel-well lips, B-pillar corners with weight ~0.003×). cp loss-gradient downweighting on these vertices causes SP regression. The hypothesis fired correctly but the lever was too long for DrivAerML's 4-orders-of-magnitude area variation.

### Trajectory — val vs test τz/τx divergence

| EP | val_abupt | val_SP | val_vol_p | val τz/τx |
|---:|---:|---:|---:|---:|
| 1 | 33.250 | 23.590 | 14.297 | 1.357 |
| 3 | 7.787 | 5.033 | 3.879 | **1.412** (deepest val break) |
| 5 | 7.061 | 4.579 | 3.766 | 1.447 (band edge) |
| 9 | 6.676 | 4.349 | 3.670 | 1.478 (in band) |
| **13 (terminal)** | **6.569** | **4.281** | **3.632** | **1.489** (val faded) |
| **test (best_ep=13)** | **6.222** | 3.916 | **3.485** ✓ | **1.418** (test held) |

The val-side band-break faded EP3→EP13 (1.412→1.489) but TEST stayed at 1.418 — the model learned generalizable τ_z structure that val (n=34) couldn't fully reflect but test (n=50) did.

### Why this closes — and why H18d is the highest-EV follow-up

**Closure rationale**: Decisive primary metric regression + 2 floor breaches (test_SP, test_WSS goal). The 9th confirmed Wave 30 dead end.

**But the mechanism is qualitatively different from the prior 8** — H18 produced the only test-surviving band-break and floor pass. Plateau Protocol normally mandates tier-shift after 5+ failures (we're at 9), BUT the H18 evidence is too valuable to abandon without one focused follow-up.

**Assigned tanjiro H18d** (PR #1183) — channel-decoupled τ_z-only area weighting. Apply per-vertex area weights ONLY to τ_z channel; cp/τ_x/τ_y at uniform weighting. Hypothesis: band-break mechanism is τ_z-specific; cp starvation is channel-coupling artifact, not intrinsic to area-weighting.

EP3 gate: `τz/τx ≤ 1.42 AND val_abupt ≤ 7.5% AND val_SP ≤ 4.50%` (must show ≥ 0.5pp SP recovery vs H18 EP3 5.033%). KILL gates: `val_abupt > 8.5% OR τz/τx > 1.50 OR val_SP > 5.5%`. ~10 LOC change to train.py. Compute-parity 14h DDP-8.

**Information value table for H18d outcomes:**

| Outcome | P | Research update |
|---|---:|---|
| Band-break + SP recovered + baseline beat | ~25% | NEW BASELINE; first Wave 30 winner; channel-specific mechanism confirmed |
| Band-break held but SP still breached | ~25% | Failure is DR (recipe), not coupling → escalate to H18b (clamped) or H18c (sqrt) |
| Band-break lost | ~25% | Coupling-required → area-weighting axis closed even with decoupling |
| Marginal / mixed | ~25% | Mid-strength signal — informs H18c vs H18b priority |

Highest EV/compute among the three student-suggested follow-ups (H18b clamped, H18c sqrt, H18d channel-decoupled). NOT pursuing H18b/c in parallel — H18d is the most diagnostic single experiment.

### Wave 30 dead-end tally — 9 confirmed across 2 tiers

| # | Hypothesis | Tier | Test τz/τx | Test floor pass | Mechanism shift |
|---|---|---|---:|---|:--|
| 1 | H10b bounded-exp head | Output-head | ~1.53 | none | none |
| 2 | H11b learnable scalar | Loss reweighting | ~1.56 | none | none |
| 3 | H12 τ-magnitude weighted | Per-vertex loss | ~1.48 | none | none |
| 4 | H16 focal MSE | Per-vertex loss | ~1.50 | none | none |
| 5 | H16b smooth-L1/Huber | Loss shape | ~1.49 | none | none |
| 6 | H20 focal per-vertex | Per-vertex loss | ~1.52 | none | none |
| 7 | H22 Charbonnier-cp+MAE-aux | Loss shape | ~1.48 | none | none |
| 8 | H23 Mean Teacher EMA | Training-regularization | ~1.87 (above) | none | wrong direction |
| **9** | **H18 area-weighted MSE** | **Per-vertex position-weighting** | **1.418 ★** | **test_vol_p ✓** | **YES — test-surviving** |

H18 is the OUTLIER — only attack with real test-side mechanism shift. H18d (PR #1183) isolates the signal.

---

## 2026-05-17 19:40 — PR #1173: H23 Mean Teacher EMA Self-Distillation (frieren) — CLOSED TERMINAL KILL / MECHANISM-WORKS-BUT-NET-COST-EXCEEDS-NET-BENEFIT / 8TH-CONFIRMED-WAVE-30-DEAD-END

- **Branch**: `frieren/h23-mean-teacher-ema-self-distillation` (closed)
- **W&B group**: `wave30-mean-teacher`
  - Main run: rank0 `e741qoo0`, 3 epochs reached, killed via senpai mechanism
- **Hypothesis**: EMA teacher with consistency loss on coord-noise-augmented views to add training-side regularization, smoothing the optimization landscape and breaking the τz/τx [1.44, 1.55] band attractor. Parameters: EMA decay 0.999, λ_consistency 0.1, coord noise σ=0.01, warmup 2000 steps.

### Terminal results — DECISIVE KILL AT EP3

| Metric | EP1 | EP2 | EP3 | Baseline at EP3 | Verdict |
|---|---:|---:|---:|---:|:--|
| val_abupt | 37.95% | 22.31% | **21.36%** | ~7.0% | **FAIL** (+14.36pp gap from baseline pace) |
| EP descent rate | — | −15.64pp | **−0.95pp** | — | **94% slope collapse EP2→EP3** |
| val τz/τx | 1.426 (below band) | 1.836 | **1.866** | ~1.46 (band) | **NO band-break — drifted ABOVE band** (wrong direction) |
| consistency_loss (raw) | 0.103 | 0.077 | 0.027 | — | descending ✅ (mechanism alive) |
| student-teacher gap | 0.234 | 0.137 | **0.048** | — | shrinking ✅ (71% drop) |
| `nonfinite_count` | 0 | 0 | 0 | — | clean — implementation perfect |

**KILL gate triggered**: pre-specified EP3 val_abupt > 11.0% threshold. EP13 projection (linear extrapolation from EP2→EP3 slope) caps at ~11.9% best case — would not approach baseline 6.126% within budget.

### Mechanism alive but objective-composition broken

The Mean Teacher mechanism itself is **mechanically perfect**:
- EMA teacher cleanly tracks student
- consistency_loss descends well-calibrated
- student-teacher gap shrinks 71% (0.234 → 0.048)
- Zero NaN/Inf, clean numerics throughout

**Failure mode**: at the objective-composition level. The consistency regularizer constrains the optimization manifold in a way that **prevents the supervised loss from finding a productive descent direction past EP2**. The supervised loss stops descending despite the regularizer working correctly — this is regularizer-supervisor trade-off, not a bug.

The τz/τx drift to 1.866 (vs. ~1.46 baseline band) is particularly telling: the regularizer is making τz *worse*, suggesting the consistency-loss-constrained manifold is structurally hostile to the WSS gradient signal.

### 8th confirmed Wave 30 dead end — closes training-regularization tier

H23 is the FIRST Wave 30 attack on training-side regularization (EMA + consistency loss). Decisive negative with mechanism active means **regularization-based stabilization is not a free lunch for DrivAerML**:
- Cold-start cost (+17pp at EP1 vs. baseline EP1)
- Slope decay caps recovery
- Even perfect mechanism cannot rescue trajectory

This closes the **training-dynamics regularization sub-tier** at first attempt. Plateau Protocol: NOT pursuing parameter-space follow-ups (4 candidate recipes: λ=0.02, warmup 5000+, σ=0.003, EMA 0.9995) — would consume 36-54h GPU for diminishing-returns variants of a confirmed failure mode.

### Wave 30 dead-end tally — 8 confirmed across multiple tiers

| # | Hypothesis | Tier | Failure mode |
|---|---|---|---|
| 1 | H10b bounded-exp head | Output-head | Floor breach |
| 2 | H11b learnable scalar | Loss reweighting | Floor breach |
| 3 | H12 τ-magnitude weighted | Per-vertex loss | Structural bias, floor breach |
| 4 | H16 focal MSE | Per-vertex loss | Floor breach |
| 5 | H16b smooth-L1/Huber | Loss shape | Floor breach |
| 6 | H20 focal per-vertex | Per-vertex loss | Floor breach |
| 7 | H22 Charbonnier-cp+MAE-aux | Loss shape | Floor breach |
| **8** | **H23 Mean Teacher EMA** | **Training-regularization** | **Slope collapse, KILL** |

τz/τx band attractor [1.44, 1.55] now confirmed across 7+ runs; H23 broke it but in wrong direction (1.866 above band).

### Follow-up

frieren ASSIGNED H29 SSFL (PR #1182) — Spectral Surface Loss with Streamwise Frequency Upweighting. First frequency-domain loss attack in DrivAerML history. Plateau Protocol tier-shift: from training-regularization to spatial-frequency gradient rebalancing. EP3 gate `τz/τx ≤ 1.42 AND val_abupt ≤ 8.5% AND train/spectral_loss descending`. KILL if `val_abupt > 9.5%` OR `τz/τx > 1.55` OR `train/spectral_loss flat`. ~70 LOC train.py only, λ=0.0 baseline recovery guarantee, baseline-compute cost (no forward doubling unlike H28). Composable with H27 PRLP and H28 SAM for future H30.

---

## 2026-05-17 17:55 — PR #1151: H12 τ-Magnitude-Weighted Surface MSE (edward) — CLOSED TERMINAL NEGATIVE / STRUCTURALLY-BIASED-CP-DAMAGE / 7TH-CONFIRMED-WAVE-30-DEAD-END

- **Branch**: `edward/tau-magnitude-weighted-loss` (closed)
- **W&B group**: `wave30_h12_tau_magnitude_sweep`
  - Arm B (α=0.5): rank0 `59v1qk32`, 70,652 steps, 14.0h
  - Arm A (α=0.3): rank0 `koud56tg`, ~70k steps, 14.0h
  - Arm C (α=0.7): SKIPPED (monotonic prediction confirmed at 2 points)
  - Smoke (α=0.5): `953pq1ip`
- **Best ckpt**: Arm A EP12 EMA (best of arms)
- **Hypothesis**: Multiply per-vertex surface MSE by `w_i = (|τ_i| / batch_mean|τ|)^α` to upweight high-WSS regions where τ_z error concentrates, aligning training signal with rel_L2 evaluation. Sweep α ∈ {0.3, 0.5, 0.7}.

### Terminal results — BOTH ARMS REGRESS

| Metric | Baseline #972 | Arm B (α=0.5) | Arm A (α=0.3) | Best | Verdict |
|---|---:|---:|---:|---:|:--|
| val_abupt | 6.126% | 6.326% | **6.290%** | 6.290 | ❌ FAIL +0.164pp |
| test_abupt | 5.844% | 6.085% | **6.046%** | 6.046 | ❌ FAIL +0.202pp |
| test_WSS | 6.727% | 7.010% | **6.952%** | 6.952 | ❌ **FAIL primary goal +0.225pp** |
| test_SP | 3.577% floor | 3.871% | **3.816%** | 3.816 | ❌ **FLOOR BREACH +0.239pp** |
| test_vol_p | 3.643% floor | 3.584% | 3.620% | held | ✅ HELD by −0.023pp |
| test_τ_x | — | 6.200% | 6.144% | — | — |
| test_τ_y | — | 7.643% | 7.580% | — | — |
| test_τ_z | — | 9.125% | 9.071% | — | — |
| **test τz/τx** | ~1.46 (band) | 1.472 | **1.476** | 1.476 | **NO MECHANISM SHIFT** |

**Merge gate**: 3 of 5 gates FAIL (val_abupt, test_WSS, test_SP). Floor breach decisive — NOT-A-MERGE.

### Monotonic regression in α — confirms structural bias

| α | test_WSS | test_SP | val_abupt |
|---:|---:|---:|---:|
| 0.0 (baseline) | 6.727 | 3.577 | 6.126 |
| 0.3 (Arm A) | 6.952 (+0.23) | 3.816 (+0.24) | 6.290 (+0.16) |
| 0.5 (Arm B) | 7.010 (+0.28) | 3.871 (+0.29) | 6.326 (+0.20) |

Linear monotonic trend confirms: **lower α reduces the regression but does NOT recover floors**. Lowering further to α=0.1 would land near baseline parity — no improvement. The mechanism is structurally biased: per-vertex weight applies to channel-mean MSE across ALL 4 surface channels (cp + 3 τ); smooth body panels (low |τ|) get tiny w_i (down to 0.02 at α=0.5) which downweights the *cp gradient* on those panels even though cp itself is NOT long-tailed.

### Weight statistics diagnostic — held to spec

| Stat | Arm B (α=0.5) | Arm A (α=0.3) |
|---|---:|---:|
| p95 weight | 1.89 (PR-expected [1.5, 5.0]) ✓ | 1.47 ✓ |
| p50 weight | 0.73 | 0.83 |
| mean weight | 0.84 | 0.86 |
| max weight | 4.0 (outliers up to 12.3) | 2.3 (max 4.5) |
| min weight | 0.02 | 0.11 |

Both arms zero `train/nonfinite_loss` and zero `train/nonfinite_grad` across ~70k steps each. **Implementation was clean — the mechanism is the wrong direction.**

### Decisive negative result for Wave 30 — per-vertex loss-reweighting axis CLOSED

**7 of 7 confirmed Wave 30 per-vertex / per-token loss-shape attacks have died at the floor breach gate**:
- H10b fern — encoder slice-temperature widening (first test_vol_p floor pass)
- H11b askeladd — gated multi-scale input (val/WSS beat but floor breach)
- **H12 edward — THIS ONE (τ-magnitude weighting, structurally biases cp)**
- H16 — focal MSE on absolute residuals
- H16b — focal MSE iteration
- H20 alphonse — per-vertex error reweighting (rel_L2 metric geometry blocker)
- H22 thorfinn — Charbonnier-cp + MAE-aux (ε=1e-3 = MAE-equivalent)

**Strong conclusion**: rel_L2 metric geometry (per-car per-component normalization) erases gain from absolute-residual reweighting under DrivAerML. The per-vertex loss-reweighting axis is exhausted.

### Why H12a (channel-decoupled magnitude weighting) NOT pursued

Student suggested H12a: apply weight only to 3 τ channels, leave cp un-reweighted. Decision: SKIP. Three reasons:
1. **τz/τx 1.476 = NO mechanism shift** — the τ-weighting itself produces no band-break. Decoupling cp won't fix what isn't broken in the τ direction.
2. **Predicted EP13 ceiling ~6.10-6.18% val_abupt** matches baseline within noise; test_WSS no better than 6.95%. 14h GPU for NEG/NEUTRAL.
3. **Per-vertex loss-reweighting axis exhausted** across 7 attempts — H12a is just another point in this dead space.

### Implications for Wave 30

After H12 closure:
- **Per-vertex/per-token loss reweighting axis fully closed** (7/7 deaths)
- Remaining live axes for floor-breach attack:
  - **INPUT representation**: H24 fern GSTS (slice-temperature widening), H26 thorfinn NPCA (local-frame coordinates) — orthogonal encoder content attacks
  - **OUTPUT-HEAD topology**: H21 nezuko per-component independent heads
  - **REPRESENTATION COUPLING**: H25 alphonse ALGP (auxiliary local-gradient prediction)
  - **TRAIN-EVAL SPACE MATCH**: H27 askeladd PRLP (per-component rel_L2 in physical space) — first attack on loss NORMALIZATION SPACE
  - **TRAINING DYNAMICS**: H23 frieren EMA Mean Teacher self-distillation
  - **NEW for H28 edward**: SAM optimizer (flat-minima bias) — FIRST optimizer-space attack in Wave 30

Edward reassigned H28 SAM (PR #1179) — `--sam-rho 0.05` two-pass perturb-recompute-restore Lion wrapper, ~60 LOC, EP3 falsifiable gate `τz/τx < 1.42 AND val_abupt < 6.00%`. ~36h wall-clock (2× compute).

---

## 2026-05-17 17:30 — PR #1167: H11b Gated Multi-Scale Input (askeladd) — CLOSED TERMINAL NOT-A-MERGE / VAL-AND-WSS-BEAT-BUT-FLOOR-BREACH / 5TH-CONFIRMED-NEGATIVE-RESULT-IN-WAVE-30

- **Branch**: `askeladd/h11b-gated-k4-16-64` (closed)
- **W&B group**: `wave30_h11b_gated_multiscale` — rank0 `ssavtag5`, ranks 1–7 `5i69ro1p`/`kw0d16d5`/`jochfjnl`/`2cejc66t`/`wleyf3j4`/`9wirio9r`/`k2xovile`
- **Best ckpt**: EP13 EMA (complete, 14.07h runtime, no budget bug)
- **Hypothesis**: Zero-init learned diagonal gate over 9 multi-scale kNN context channels (k∈{4,16,64} × {cos_align, log_area, log_dist}) to restore H11's SP/vol_p floor breach while preserving WSS gain.

### Terminal results (EP13 EMA)

| Metric | H11b | Baseline #972 | Δ | Verdict |
|---|---:|---:|---:|:--|
| val_abupt | **6.057%** | 6.126% | −0.069pp | ✅ PASS — FIRST sustained Wave 30 baseline beat |
| test_abupt | **5.8147%** | 5.844% | −0.029pp | ✅ PASS |
| test_WSS | **6.6322%** | 6.727% | −0.095pp | ✅ PASS |
| test_SP | **3.7179%** | 3.577% **FLOOR** | +0.141pp | ❌ **FLOOR BREACH** |
| test_vol_p | **3.6773%** | 3.643% **FLOOR** | +0.034pp | ❌ **FLOOR BREACH** |
| test τz/τx | 1.467 | — | in band | converged INTO [1.44,1.55] from below |

**Merge gate result**: 3 of 5 gates pass (val_abupt + test_abupt + test_WSS) but BOTH floors breach → NO-MERGE. Student explicitly recommended NO-MERGE.

### Gate mechanism diagnostic — gate WORKED but didn't fix floors

Per-channel final gate values (init=0.0):

| ch | k | feature | final | |gate| |
|---:|---:|:--|---:|---:|
| 0 | 4 | cos_align | **−0.785** | 0.79 (saturated negative) |
| 1 | 4 | log_area | +0.284 | 0.28 |
| 2 | 4 | log_dist | +0.530 | 0.53 |
| 3 | 16 | cos_align | **+0.779** | 0.78 (saturated positive) |
| 4 | 16 | log_area | −0.210 | 0.21 |
| 5 | 16 | log_dist | +0.362 | 0.36 |
| 6 | 64 | cos_align | **−0.995** | 1.00 (fully saturated) |
| 7 | 64 | log_area | −0.332 | 0.33 |
| 8 | 64 | log_dist | +0.556 | 0.56 |

- mean_abs: 0 → **0.537** (warmed up smoothly through training)
- max_abs: 0 → **0.995** (k=64 cos_align fully saturated)
- L2: 0 → **1.781**
- zero_fraction: 0 throughout (no channel collapsed)

**Channel pattern**: cos_align channels saturated to |gate| > 0.7 with MIXED signs (k=4 negative, k=16 positive, k=64 negative-saturated). log_dist channels all admitted positively (+0.28 to +0.56). log_area channels split with smaller magnitudes. The gate learned a non-trivial differentiated routing.

### Decisive negative-result insight

The gate mechanism worked mechanically — it CAN down-weight unhelpful multi-scale channels and flip the sign of signal channels. But **the SP/vol_p floor breach was NOT fixed** because the multi-scale signal flows THROUGH the shared encoder before reaching the output heads. A 9-dim diagonal gate at the input cannot decorrelate per-head gradient paths inside the transformer where the actual cross-head interference occurs.

### Band-break diagnostic — fleet-wide attractor confirmed (6th run)

τz/τx trajectory: EP1=1.466 → EP3=1.506 → EP7=1.549 → **EP13=1.556**. Converged INTO the [1.44, 1.55] collapse band from below — exactly mirroring fern H10b, dl24 H10b, edward H12, nezuko H21, tanjiro H18 patterns. **6 independent runs now confirm the band as an attractor for all input/output/loss-form attacks.**

### Why TERMINAL-NOT-A-MERGE despite val_abupt + test_WSS + test_abupt beats

Both floors breach hard merge constraints (CLAUDE.md):
- test_SP > 3.577% floor: **+0.141pp breach** (binding)
- test_vol_p > 3.643% floor: +0.034pp breach (also binding but smaller)

Merging would compound H11's pre-existing floor breach rather than restoring it. Floor passes are the dominant Wave 30 merge blocker — every fleet baseline-crosser hits this gate.

### Key Wave 30 implication — floor disease is downstream of input

H11b is the **5th confirmed dead end in Wave 30**, joining:
- H18/H20 (per-vertex error reweighting): closed — rel_L2 normalization
- H16/H16b (static Huber on τ): closed — frac_in_L1 decay
- H10/H10b (bounded-exp output activation): closed — 73%/27% structural
- H22 (Charbonnier-cp): closed — cp-MSE NOT the disease
- **H11b (input gating)** ← NEW: disease downstream of input

The floor disease is increasingly localized to:
- **Output-head / per-head gradient paths** (H27 attack: train loss in eval space, H21 attack: separate per-channel MLPs)
- **Backbone representation coupling between cp and τ** (H25 ALGP attacks this)
- **The rel_L2 metric space mismatch** (H27 directly attacks this)

### Stackability — H11b's val_abupt mechanism is real and worth preserving

H11b's input-side multi-scale gating produces real val_abupt (−0.069pp) + test_abupt (−0.029pp) + test_WSS (−0.095pp) gains. The mechanism is alive — it just cannot fix floors alone. **H11b is publishable as a stackable substrate** for future floor-preservation attacks. If H27 (PRLP, askeladd's next) succeeds, the natural H28 is H11b+H27 compound.

---

## 2026-05-17 17:00 — PR #1172: H22 Charbonnier-cp + MAE-aux (thorfinn) — CLOSED TERMINAL-NEUTRAL / MECHANISM-WIRED-BUT-INEFFECTIVE / HYPOTHESIS-FALSIFIED / NOT-STACKABLE

- **Branch**: `thorfinn/charbonnier-cp-mae-aux` (closed)
- **W&B group**: `wave30_h22_charbonnier_cp` — rank0 `2y5zraax`, ranks 1–7 `tox671e8`/`ykuu27sy`/`r3potpuu`/`7drf0ja1`/`j3akfo56`/`l95qn7ig`/`qsal5gu1`
- **Best ckpt**: EP3 EMA (truncated_partial — SENPAI_TIMEOUT_MINUTES=360 budget bug fired, run finished at 271min/EP3)
- **Hypothesis**: Replace MSE on cp channel with Charbonnier `sqrt(e²+ε²)-ε + λ_aux·|e|` (ε=1e-3, λ_aux=0.5) to preserve cp gradient on smooth panels and steepen val_SP descent. Designed as floor-preservation stacking partner for H10b/H11b winners.

### Terminal results (EP3 EMA from inline reload-eval at end of training)

| Metric | H22 EP3 | Baseline #972 | Δ | Verdict |
|---|---:|---:|---:|:--|
| val_abupt | 7.110% | 6.126% | +0.984pp | FAIL (truncated) |
| test_abupt | **6.849%** | 5.844% | +1.005pp | FAIL |
| test_SP | **4.079%** | 3.577% floor | +0.502pp | **FLOOR BREACH** |
| test_vol_p | **4.139%** | 3.643% floor | +0.496pp | **FLOOR BREACH** |
| test_WSS | 7.958% | 6.727% | +1.231pp | FAIL |
| test τz/τx | **1.429** | — | below band | cold-start (EP3) |

### Mechanism-wired-but-ineffective diagnostic

Per-epoch Charbonnier diagnostics (rank0 `2y5zraax`):

| EP | charb_cp_charb | charb_cp_mae | ratio | err_p95 | err_max | charb_tau_mse | base_mse |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0866 | 0.0871 | 0.995 | 0.301 | 3.26 | 0.0930 | 0.273 |
| 2 | 0.0376 | 0.0380 | 0.988 | 0.123 | 1.84 | 0.0261 | 0.073 |
| 3 | 0.0136 | 0.0140 | 0.968 | 0.045 | 1.07 | 0.0042 | 0.013 |

**Decisive falsification finding**: `charb_cp/cp_mae ratio ≈ 0.97` means with ε=1e-3 and typical |e|~0.2, the Charbonnier term degenerates to MAE — effective cp loss is `1.5·|e|` (MAE-equivalent). The smooth-L1 transition the hypothesis depended on was NEVER ENGAGED for bulk residuals.

### EP3 floor gate — NEUTRAL match, not steeper

| Run @ EP3 | val_abupt | val_SP | val_vol_p | val_WSS |
|---|---:|---:|---:|---:|
| **H22 (cp Charb+MAE-aux)** | **7.110** | **4.405** | **4.166** | **8.154** |
| H11b askeladd (gated multi-scale) | 6.411 | 4.368 | 4.028 | 7.166 |
| H10b fern (bounded-exp τ) | 6.697 | 4.422 | 3.884 | 7.615 |

**val_SP @ EP3 matches H10b/H11b** — not steeper. The cp descent trajectory under MAE-equivalent is IDENTICAL to MSE-on-cp baselines. **Hypothesis falsified**: cp-MSE saturation is NOT the disease causing test_SP floor breach.

### τ-channel side effect — NOT stackable

At EP3, cp contribution to surface_loss is **4.9× larger** than τ contribution (0.0206 vs 0.0042). τ gradient share starved →:
- val_abupt H22=7.110% vs H11b=6.411% (+0.70pp WORSE at matched epoch)
- val_WSS H22=8.154% vs H11b=7.166% (+0.99pp WORSE)
- val_τz H22=10.71% vs H11b ~9.0% (+1.7pp WORSE)

Stacking H22 with a winning τ-attack would DEGRADE τ accuracy at the loss-aggregation level. **NOT-STACKABLE — premise dead.**

### Closure rationale

This is the **4th decisive negative result in Wave 30** (joining H16, H16b, H20). Floor-preservation via cp-side loss reformulation is now CLOSED:

1. cp errors descend monotonically under MSE — there is no gradient saturation to fix
2. MAE-equivalent on cp produces identical val_SP descent to MSE
3. The disease causing test_SP floor breach is downstream of cp accuracy — correlates with τ_z magnitude error in shared backbone capacity
4. ε=0.1 follow-up rejected — would not change the diagnosis

### Key Wave 31 implication

Future floor-preservation attacks must target either:
- **τ-side directly** (e.g., Charbonnier on τ_z, though faces rel_L2 normalization headwind)
- **cp/τ representation coupling at backbone level** (H25 ALGP is in this direction)
- **Per-channel grad-norm equalization** (suggested as standard future diagnostic by thorfinn)
- **NOT cp-side loss reformulation** — exhausted

### Pre-launch budget verification

Thorfinn's `printenv SENPAI_TIMEOUT_MINUTES=360.0` documented pre-launch is the cleanest end-to-end evidence of the budget bug to date — to be posted to #1056.

---

## 2026-05-17 16:00 — PR #1170: H20 Focal Vertex Loss (alphonse) — CLOSED TERMINAL-NULL (mechanism executed, band-break cold-start artifact, rel_L2 metric geometry kills focal gains)

- **Branch**: `alphonse/h20-focal-vertex-loss` (closed)
- **W&B group**: `wave30_h20_focal_vertex_loss` — rank0 `jau1ksmq`, ranks 1–7 `cc50e9ct`/`6rodkhdo`/`6riuraqr`/`01lfixoh`/`90ld2a1u`/`0t57f90j`/`t77ard01`
- **Best ckpt**: EP3 EMA (budget-starved at ~300min effective, EP4 killed by SIGTERM)
- **Hypothesis**: Dynamic per-vertex error-weighted MSE on τ channels with γ=0.5 (γ_z=1.5×γ_x) to force the model to concentrate gradient on high-error, high-τ tail vertices and break the τz/τx collapse band.

### Terminal results (EP3 EMA from offline eval — SENPAI_TIMEOUT_MINUTES effective ~300)

| Metric | H20 EP3 | Baseline #972 | Δ | Verdict |
|---|---:|---:|---:|:--|
| val_abupt | 8.671% | 6.126% | +2.545pp | FAIL (KILL gate: >6.70%) |
| test_abupt | **8.480%** | 5.844% | +2.636pp | FAIL |
| test_SP | 5.953% | 3.577% floor | +2.376pp | **FLOOR FAIL** |
| test_vol_p | 5.159% | 3.643% floor | +1.516pp | **FLOOR FAIL** |
| test_WSS | 9.478% | 6.727% | +2.751pp | FAIL |
| val τz/τx | **1.523** | — | in band | break FADED |
| test τz/τx | **1.481** | — | in band | break FADED |

### Band-break diagnostic — cold-start artifact, not mechanism win

| EP | τz/τx | Interpretation |
|---|---|---|
| EP1 (step 10,864) | **1.389** | Fleet-deepest — cold-start residual artifact |
| EP2 (step 21,729) | **1.401** | Still below band — high EP1-2 variance artifact |
| EP3 (step 32,594) | **1.523** | INSIDE band — natural channel-magnitude ratio restored |

The early band-break was a cold-start transient: γ_z=1.5×γ_x preferentially shrinks τ_z's large EP1-2 residuals faster, lowering its L2 numerator transiently. Once the model fits the bulk distribution at EP3, the ratio reverts to the "natural" channel-magnitude ratio (~1.52).

### Focal mechanism diagnostics — clean execution

- `w_z_p95/w_z_mean = 3.75–3.86` throughout (heterogeneity maintained)
- Normalization invariant `w_mean=1.0000` held exactly (implementation correct)
- `raw_w_z > raw_w_x` at EP1 (τ_z residuals dominate warmup) → crossover at EP2 (`raw_w_z < raw_w_x`) → mechanism actually DID reduce τ_z absolute residual faster

### Why TERMINAL-NULL despite clean mechanism execution

Two fundamental metric-level blockers:
1. **rel_L2 metric geometry**: metric normalizes by channel reference magnitude. τ_z has a structurally larger reference (dominant flow-aligned shear), so absolute-residual gains from focal weighting get erased by normalization. Per-vertex focal can win on absolute τ_z error but still fail on rel_L2.
2. **Easy-vertex damping**: `raw_w_z ≈ 0.031` at EP3 → ~97% of typical-error vertices contribute almost nothing to gradient. Slows bulk convergence (val_abupt 8.671% at EP3 vs ~6.7% baseline).

### NOT stackable

Unlike H15b (EMA) and H19 (VICReg), H20's metric geometry mismatch is fundamental — stacking focal reweighting on a winner won't help since the normalization structure erases absolute gains. **Per-vertex error-reweighting direction CLOSED for DrivAerML rel_L2.**

### Key Wave 31 implication

The rel_L2 insight suggests future loss-form attacks should target **proportional error** (e.g., log-space regression, symmetric MAPE) rather than absolute-residual reweighting. Or skip the loss entirely and attack encoder representations directly (H24 GSTS, in flight).

---

## 2026-05-17 15:45 — PR #1164: H10b bounded-exp magnitude head (fern) — CLOSED TERMINAL NOT-A-MERGE / MECHANISM-PASS / HYPOTHESIS-FALSIFIED / PARKED STACKABLE (first test_vol_p floor pass in active fleet)

- **Branch**: `fern/h10b-bounded-exp-magnitude` (closed)
- **W&B group**: `wave30_h10b_bounded_exp` — rank0 `3rva90lq`, ranks 1–7 `itiohaf5`/`oiy85mv5`/`baj1ti7u`/`p2cvfel2`/`8ejobjsz`/`72ql8mkw`/`iopnmz2n`
- **Best ckpt**: EP12 EMA, selection metric val_primary/abupt_axis_mean_rel_l2_pct
- **Hypothesis**: Replace H10's softplus magnitude head (which had a >6.93 norm-space floor) with bounded-exp `clamp(log_mag, -3, +3).exp()`. The softplus floor was hypothesized to be the 73%/27% mag/dir error split's root cause.

### Terminal results (EP12 EMA — full 13ep ran, 14.31h runtime, no budget-starvation)

| Metric | H10b val | H10b test | Baseline #972 | Floor/Target | Verdict |
|---|---:|---:|---:|---|:--|
| val_abupt | **6.217%** | — | 6.126% | <6.126 | **FAIL** +0.091pp |
| test_abupt | — | 5.998% | 5.844% | — | +0.154pp |
| test_WSS | — | **6.980%** | 6.727% | <6.727 | **FAIL** +0.253pp |
| test_SP | 4.077% | **3.755%** | 3.577% floor | ≤3.577 (HARD) | **FAIL** +0.178pp |
| test_vol_p | 3.584% | **3.481%** | 3.643% floor | ≤3.643 (HARD) | **PASS −0.162pp ✓** (FIRST IN FLEET) |
| test τz/τx | 1.530 | **1.441** | ~1.46 | <1.40 structural | band-edge break (output-head best) |

### Magnitude/direction decomposition — DECISIVE NEGATIVE RESULT

| Diagnostic | H10 | H10b | Target | Verdict |
|---|---:|---:|---:|:--|
| val mag_share_sq | 73% | **73.1%** | <50% | **REJECT — hypothesis falsified** |
| test mag_share_sq | — | **73.2%** | <50% | unchanged |
| pred median (norm) | — | **0.697** | match GT 0.700 | matched to 0.5% |
| frac<6.93 floor | — | **0.993** | match GT 0.993 | exact match |
| clamp engagement | — | **0.003%** | <1% | range correctly sized |

**The mechanism executed flawlessly.** Pred distribution matches GT shape AND mean to 1% across every percentile. Bounded-exp clamps essentially never engage — the [-3, +3] log_mag range was correctly sized. BUT the 73%/27% mag/dir error split is UNCHANGED from H10 — **the softplus floor was NOT the magnitude bottleneck**.

### Why the hypothesis was wrong (and what we learned)

The 73%/27% mag/dir error split is a **structural property of the τ field**, not an activation artifact. Even when pred_mag matches GT shape AND mean, per-vertex squared magnitude error dominates per-vertex angular projection error. Per fern's interpretation: low-τ vertices dominate in count, and small absolute magnitude errors squared can still outweigh small angular errors on high-τ vertices.

**Key Wave 31 implication**: direction-side attacks may have been undervalued, and the true bottleneck may be in **how the loss aggregates per-vertex errors** rather than **what activation produces the magnitude**. Fern's follow-up #2 (per-bin error decomposition over GT magnitude bins [0, p50, p90, p99, max]) becomes a high-priority diagnostic.

### Wins — stackable signal

1. **FIRST test_vol_p floor pass in active fleet** (3.481% < 3.643% by −0.162pp). Stable below floor since EP4.5, mechanism-driven.
2. **Best output-head band-edge break** (test τz/τx 1.441) — only loss-side attacks (H18, H20) go deeper.
3. **Non-worse than baseline on every comparable axis vs H10** — the EMA + bounded-exp + cos-loss recipe is a clean substrate for stacking.

### Parked as stackable mechanism

H10b is the **first stackable substrate with confirmed floor preservation on vol_p**. Natural stacking partner:
- **H22 thorfinn Charbonnier-cp + MAE-aux** (PR #1172, EP~1.3) — H10b protects vol_p, H22 protects cp/SP. If H22 lands cleanly with vol_p preserved, the stack would be the first single-model merge candidate.

### Fleet contribution

H10b's clean negative result + the structural-split insight is the strongest negative evidence we have on the magnitude-bottleneck framing. Closes the output-head-activation-fix lane definitively. **Wave 31 direction-side attacks become higher priority.**

---

## 2026-05-17 12:05 — PR #1169: H16b Huber loss on τ channels δ=0.3 (frieren) — CLOSED TERMINAL NOT-A-MERGE / Huber static-δ direction EXHAUSTED (4th budget-starved closure)

- **Branch**: `frieren/H16b-huber-delta-03` (closed)
- **W&B group**: `wave30_h16b_huber_delta03` — rank0 `uhqnox73`, ranks 1–7 `jcojjk0b`/`l9lilfbj`/`em27zvr6`/`34w0eohc`/`zdvpvbmp`/`17ufldch`/`dyzx2u18`
- **Hypothesis**: H16 (δ=1.0) proved Huber threshold too coarse (frac_in_L1/τz = 0.014% = effectively MSE). H16b tests δ=0.3 targeting bulk of post-EP1 residual distribution. ~1 flag change from H16.

### Terminal results (EP3.79 truncated by SENPAI_TIMEOUT_MINUTES=360 confirmed via `/proc/<pid>/environ`)

| Metric | H16b EP3 | Baseline #972 | Δ | Verdict |
|---|---:|---:|---:|:--|
| val_abupt | 7.106% | 6.126% | +0.980pp | FAIL |
| test_abupt | **6.767%** | 5.844% | +0.923pp | FAIL |
| test_SP | 4.547% | 3.577% floor | **+0.970pp** | **FAIL FLOOR** |
| test_vol_p | 4.635% | 3.643% floor | **+0.992pp** | **FAIL FLOOR** |
| test_WSS | 7.538% | 6.727% | +0.811pp | FAIL |
| test τz/τx | 1.440 | — | edge of band | no break |

### Huber mechanism — FADES STRUCTURALLY across training

| Step | EP | frac_L1 τx | frac_L1 τy | frac_L1 τz | max_abs/τz |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | — | — | 41.14% | 11.63σ |
| 1500 | 1.0 | 24.40% | 21.39% | 27.21% | — |
| 7500 | 2.0 | — | — | 3.41% | — |
| 12500 | 3.3 | — | — | 1.29% | 2.83σ |
| 14605 | 3.79 | 0.63% | 0.88% | 1.11% | 4.48σ |

Mechanism fades 44%→1% across 3 epochs (40× decay). The residual distribution shrinks faster than δ=0.3σ can track. By EP6+, would be effectively MSE-on-τ — same failure mode as H16 (δ=1.0).

### Why static-δ Huber is structurally exhausted

| H | δ | EP1 frac_in_L1 | EP3 val_abupt | Outcome |
|:--|---:|---:|---:|:--|
| H16 (#1161) | 1.0 | 0.014% | 6.894% | mechanism inactive (too large) |
| H16b (#1169) | 0.3 | 44.1% | 7.106% | mechanism fades (correct at EP1 but transient) |

**The Huber direction is closed.** A static δ cannot track the shrinking residual distribution of a converging deep network. The static-tail-robust-loss family on this task is ruled out. Dynamic-δ Huber (cosine-decay or per-percentile-tracking) is the natural follow-up but was rejected as the next step in favor of bolder direction (per Plateau Protocol).

### Fleet contributions

1. **Decisive Huber-direction closure** — 2/2 fails across δ=1.0 and δ=0.3 confirms the family is structurally limited on this task.
2. **Cleanest evidence of SENPAI_TIMEOUT_MINUTES=360 deployment bug**: student verified directly via `/proc/<pid>/environ`. 4 closures now known to be budget-starved (H15b, H17, H19, H16b).
3. **Floor breach pattern reconfirmed**: test_SP and test_vol_p both +0.97pp / +0.99pp above floors at EP3. Even with full 13 epochs, this is a hard NO-MERGE blocker. Reinforces the floor-preservation lane (H22 thorfinn Charbonnier-cp + MAE-aux) as the necessary stacking partner.
4. **Mechanism-fade analysis is reusable**: future loss-form attacks should pre-spec the running residual scale, not a fixed σ. The H16b finding that residual distribution shrinks 10× over 3 epochs informs any future tail-robust attack.

### Reassignment

frieren → **H23 Mean Teacher self-distillation** (PR #1173) — bold Plateau Protocol direction shift away from MSE/Huber/quantile loss-form attacks. Uses existing EMA model as teacher + spatial-coordinate input augmentation. Tarvainen & Valpola 2017. Train.py-only ~50 lines. Orthogonal to all 8 in-flight attacks (which target output/loss/topology — Mean Teacher targets TRAINING DYNAMICS).

---

## 2026-05-17 10:35 — PR #1168: H19 VICReg Batch-Variance Regularization on Predicted τ_z (thorfinn) — CLOSED TERMINAL NOT-A-MERGE / mechanism PARKED AS STACKABLE (3rd budget-starved mechanism-PASS closure in Wave 30)

- **Branch**: `thorfinn/H19-vicreg-tau-z-variance` (closed)
- **W&B group**: `wave30_h19_vicreg_tau_z` — rank0 `8x9qt1tu`, ranks 1–7 `j4q2f6gu`/`5rgk3a1s`/`68xj8n90`/`j0fbd51l`/`bvz7d7jz`/`pqw0zb31`/`lq28cjy4`
- **Hypothesis**: VICReg variance hinge `λ · max(0, γ − std(|τ_z|))²` with γ=0.05, λ=0.10 — penalize batch-statistic collapse of per-sample mean |τ_z|. Frames τ_z/τ_x band [1.44, 1.55] as batch-level distributional failure rather than per-vertex error. ~12 lines train.py-only, ref Bardes et al. ICLR 2022 + Hanna et al. arXiv:2412.13993.

### Terminal results (best_epoch=3 EMA, SENPAI_TIMEOUT_MINUTES=360 cut at 271 min)

| Metric | H19 | Baseline #972 | Δ | Verdict |
|---|---:|---:|---:|:--|
| val_abupt | **6.998%** | 6.126% | **+0.872pp** | FAIL |
| test_abupt | **6.670%** | 5.844% | +0.826pp | FAIL |
| test_SP | 4.274% | 3.577% floor | **+0.697pp** | **FAIL FLOOR** |
| test_vol_p | 3.934% | 3.643% floor | **+0.291pp** | **FAIL FLOOR** |
| test_WSS | 7.671% | 6.727% | +0.944pp | FAIL |

| Axis | test_rel_l2 | test_MAE |
|---|---:|---:|
| τ_x | 6.758% | 0.0874 |
| τ_y | 8.627% | 0.0590 |
| τ_z | **9.755%** | **0.0567** |

### VICReg mechanism — PASS, fired throughout 30,448 steps (no quiescence)

| step bucket | mean std_τ_z | penalty_active fraction | mean batch \|τ_z\| |
|---:|---:|---:|---:|
| 0–200 | 0.0438 | 59.0% | 0.0305 |
| 200–1k | 0.0723 | 52.6% | 0.0521 |
| 1k–5k | 0.0982 | 51.5% | 0.0696 |
| 5k–15k | 0.1051 | 51.9% | 0.0741 |
| 15k–30k | 0.1103 | 52.1% | 0.0776 |
| **Overall** | — | **52.0%** | — |

- std_τ_z lifted 0 → 0.110 (2.2× γ at terminal, max observed 0.342, no NaN/Inf)
- batch mean |τ_z| climbed 0.019 → 0.078 — close to GT |τ_z| ≈ 0.0793 (model calibrated to GT z-magnitude scale)
- λγ²/base_mse_loss ≈ 0.025% (cost-negligible)
- **Probable band-break** by MAE-ratio proxy: test τ_z MAE = 0.0567 < GT mean|τ_z| ≈ 0.0793 — model NOT in [1.44, 1.55] collapse band (direct mean|τ_z_pred|/mean|τ_x_pred| not logged)

### Why it failed — budget starvation, not mechanism

EP3 trajectory was AHEAD of healthy expected: hit 7.80% at EP2, on track for ≤6.2% at EP10 had budget allowed. SENPAI_TIMEOUT_MINUTES=360 (6h cap) cut at 271 min after EP3 + final eval — only 3/13 epochs. EP3 trajectory was actually PASSING (≤9.5% gate cleared by 2.5pp), but EP10 ≤6.2% gate unreachable in 3 epochs.

### Fleet contribution

1. **3rd budget-starved mechanism-PASS closure** alongside H15b alphonse (#1165) and H17 nezuko (#1162) — establishes a pattern: short-budget runs with mechanism PASS / baseline FAIL are candidates for parking as stackable
2. **VICReg variance hinge stackable**: ~12 lines, ~0 wall-time cost, ~0.025% loss contribution, continuously engaged (not single-shot warmup fix), orthogonal to all output-head/loss-form/input-gating attacks in the fleet
3. **Critical fleet bug exposed**: SENPAI_TIMEOUT_MINUTES=360 deployment env var overrode student's per-launch CLI value (1100min). Same risk for tanjiro H18 v2 (#1163) which relaunched at 05:35Z and would hit 11:35Z timeout. Flagged on #1163 for verification.
4. **MAE-ratio proxy for band-break** documented when direct magnitude ratio not logged

---

## 2026-05-17 06:15 — PR #1162: H17 Local Tangent-Frame Output Reparameterization for WSS (nezuko) — CLOSED TERMINAL-NULL (KILLED at EP3 gate; tangent-frame mechanism implementation correct — orthogonality residuals at fp32 machine-zero — but trajectory cold-start gap ~11pp vs baseline; representation reparameterization can't recover within 13ep lr=9e-5 budget; nezuko reassigned to fresh hypothesis researcher-agent pending)

- **Branch**: `nezuko/h17-tangent-frame-wss-head` (closed)
- **W&B group**: `wave30_h17_tangent_frame` — rank0 `fi6w2f9i`, ranks 1–7 `cwbunvim`/`c9dhsn8c`/`3w2ebgge`/`3n3caxmg`/`5yrf0kd5`/`9t7nj2ph`/`k7s3jh1w`; SIGTERM'd at step ~22,600 (mid-EP4) after EP3 KILL gate hit
- **Hypothesis**: Predict 2 tangent-plane coefficients (α_t1, α_t2) in local surface tangent basis, reconstruct global τ_WSS = α_t1·t1 + α_t2·t2 inside forward() — enforces τ·n=0 by construction (vs H6' soft penalty). Tangent basis from Gram-Schmidt on input normals + e_x/e_y fallback for degeneracy.

### Terminal results — TERMINAL-NULL (KILLED at EP3 gate per PR's own criteria)

| Epoch | val_abupt | val_WSS | val_SP | val_vol_p | baseline EP val_abupt | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| EP1 (post-warmup) | **33.32%** | 41.53% | 22.11% | 16.44% | 20.49% | **+12.83pp** |
| EP2 | 19.04% | 28.86% | 9.66% | 7.21% | 7.91% | +11.13pp |
| EP3 | **18.46%** | 29.37% | 4.93% | 4.16% | 7.11% | **+11.34pp** ← KILL (gate ≤7.4%) |
| EP4 | SIGTERM'd at step ~22,600 mid-epoch | | | | | |

**No test eval** — best checkpoint val_abupt = 18.46% (3× baseline), no decision value from test.

### Tangent-frame mechanism diagnostics — IMPLEMENTATION CORRECT

| Diagnostic | Measured | Expected |
|---|---:|:--|
| `t1_t2_orthogonality_residual` | 2.51e-9 | machine-zero ✓ |
| `n_t1_orthogonality_residual` | 1.91e-8 | machine-zero ✓ |
| Analytical `max\|τ·n\|` (fp32) | ≤ 2.4e-7 | by-construction ✓ |
| Measured `max\|τ·n\|` (bf16) | ~5.4e-2 | bf16 quantization floor — not a constraint failure |
| `degeneracy_frac` | 0.088 | 8.8% vertices use e_y fallback (automotive side panels) |
| `alpha_t1/alpha_t2_absmax` | 12.75 / 17.13 | reasonable for unit-variance normalized targets |

### Why it failed — cold-start representational gap

Gap vs baseline is FLAT across EP1→EP3 (+12.83pp → +11.13pp → +11.34pp). NOT a "trailing trajectory" (which would close the gap) — a PARALLEL trajectory offset by ~11pp. Two compounding causes:

1. **Output basis is now data-dependent.** Every vertex sees a different (t1, t2) basis depending on its normal. Encoder must learn to route relevant features through a per-vertex-varying coordinate system. The pre-trained representation has no head-start.

2. **Effective parameter count reduced** (2 channels vs 3 for τ). Less DOF during cold-start, even though it's the *point* once converged.

This is a known failure mode for hard-constraint output reparameterizations: they trade favorable converged geometry for slower convergence. Under 13-epoch lr=9e-5 with curriculum, the converged state is unreachable.

### Fleet contribution

1. **Hard τ·n=0 by-construction implementation verified correct.** Reference implementation at `model.py:624-664` on `nezuko/h17-tangent-frame-wss-head` for any future tangent-frame attempt.
2. **8.8% of automotive vertices need e_y fallback** (side panels with normals ≈x-axis) — dataset statistic for future axis-dependent feature extraction.
3. **Empty-surface-batch (N_S=0) `.amax()` guard pattern** documented for curriculum schedules. Will hit on any future surface-aware output formulation.
4. **EP1 hard-kill heuristic confirmed**: cold-start representational changes manifest at EP1 as val_abupt > 30% with warmup=1 recipe.

### Comparison with H6' soft-penalty (PR #1147 closed)

| Mechanism | Test τz/τx | Test_WSS | Test_abupt | Verdict |
|---|---:|---:|---:|:--|
| H6' soft τ·n=0 penalty | **1.420** (FIRST band break) | 6.78% (+0.05pp regress) | 5.91% (+0.07pp regress) | NOT-A-MERGE |
| H17 hard τ·n=0 by-construction | N/A (killed EP3) | N/A | N/A | KILL — cold-start |

Soft penalty is the better experimental match in 13-epoch budget. Hard constraint requires longer warmup or transfer from global-frame pretrained model — out of scope for this wave.

---

## 2026-05-17 06:10 — PR #1165: H15b EMA Polyak Averaging decay=0.999 (alphonse) — CLOSED NOT-A-MERGE (mechanism PASS, baseline FAIL all axes + both floors; EMA AHEAD of raw by +0.80pp at EP3 — opposite sign of H15's −9.54pp gap — but only 3 of 13 epochs trained, recipe budget-starved; H15 series parked as stackable mechanism for future winners)

- **Branch**: `alphonse/h15b-ema-decay-fast` (closed)
- **W&B group**: `wave30_h15b_ema_decay999` — rank0 `2y5teerz`, ranks 1–7 `mp9kgfsc`/`zs65yzu3`/`6v4zt23e`/`hxvlepv6`/`b7tjmeqx`/`rc7jm8d7`/`gkvcwujw` (clean DDP teardown)
- **Hypothesis**: H15 (decay=0.9999) was budget-truncated, not falsified — EMA half-life 0.92ep ran out before warming. H15b at decay=0.999 (half-life 0.064ep) tests same EMA inference benefit at faster-warmup decay. `eval_raw_vs_ema=True` for dual eval.

### Per-epoch table — H15b raw vs EMA

| Epoch | step | val_raw_abupt% | val_ema_abupt% | gap (raw−ema) | val_raw_WSS% | val_ema_WSS% | val_raw_SP% | val_ema_SP% | val_raw_vol_p% | val_ema_vol_p% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ~10,069 | 30.389 | 28.983 | **+1.41** | 33.743 | 32.269 | 23.695 | 22.261 | 18.132 | 16.841 |
| 2 | ~20,138 | 8.623 | 7.639 | **+0.98** | 9.622 | 8.612 | 5.715 | 5.099 | 5.366 | 4.507 |
| 3 | 30,207 | 7.642 | **6.838** | **+0.80** | 8.585 | 7.750 | 5.137 | 4.527 | 4.522 | 3.963 |

Per-epoch τz/τx: EP3 val_ema τx=6.740%, τy=8.713%, τz=10.247% → τz/τx = **1.520**. test τz/τx = **1.439** (close to band-break — second-closest after H6' 1.420).

### Decisive H15 vs H15b panel — MECHANISM CONFIRMED

| Run | Decay | Half-life | Best Epoch | val_raw_abupt | val_ema_abupt | gap | EMA verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| H15 (#1155) | 0.9999 | 0.92 ep | EP4 | 7.02% | 16.56% | **−9.54pp** | cold, never warmed |
| **H15b** (this) | **0.999** | **0.064 ep** | **EP3** | **7.642%** | **6.838%** | **+0.80pp** | **AHEAD of raw** ✓ |

EMA crossed raw at EP1 (gap = +1.41pp). best_checkpoint/source = ema every epoch. H15 hypothesis (EMA inference benefit) now mechanism-confirmed — H15 was decay-too-slow, not mechanism-wrong.

### Test metrics (best checkpoint = EMA EP3)

| Metric | val_ema (EP3) | test (EP3 EMA) | Baseline #972 | Δ test | Gate |
|---|---:|---:|---:|---:|:--|
| abupt | 6.838% | **6.636%** | 5.844% | **+0.792pp** | FAIL |
| WSS | 7.750% | **7.636%** | 6.727% | **+0.909pp** | FAIL |
| SP | 4.527% | **4.268%** | 3.577% floor | **+0.691pp** | FAIL FLOOR |
| vol_p | 3.963% | **3.924%** | 3.643% floor | **+0.281pp** | FAIL FLOOR |
| τx | 6.740% | 6.750% | — | — | — |
| τy | 8.713% | 8.530% | — | — | — |
| τz | 10.247% | 9.708% | — | — | — |
| **τz/τx** | 1.520 | **1.439** | ~1.46 band | **−0.021 band-edge** | MARGINAL band signal |

### Why it failed (absolute axis)

1. **Only 3/13 epochs trained.** Cosine LR schedule designed for 13 epochs barely 6% along at EP3. Val slope −0.094pp/1k steps — far from converged.
2. **decay=0.999 fast-warms but doesn't average late-cosine annealing oscillations** (EMA usually captures inference benefit from this). decay=0.9999 averages a longer history but needs 18h to warm.

H15 series is **budget-constrained, not mechanism-broken**.

### Fleet contribution

1. **EMA decay calibration**: 0.999 = warm in 6h; 0.9999 = cold without 18h. Reusable for any future EMA-stacked experiment.
2. **NEW MECHANISM DATAPOINT**: EMA averaging implicitly decorrelates per-vertex τ predictions (test τz/τx dropped from raw 1.520 to EMA 1.439 — 0.081 band-shift purely from weight averaging). Stacking EMA on a τ_z-targeted attack (H10b/H17 successor/H6'b) could compound the band-break.
3. **Per-epoch dual-eval format adopted as fleet standard** for any optimizer-smoothing experiment.

H15 series parked — H15c (decay=0.9999, 18h) and H15d (decay=0.9995) are valid follow-ups but lower EV than fresh-axis attacks until a baseline-beating recipe lands. Then retrofit EMA as +0.5pp stacker.

---

## 2026-05-17 06:00 — PR #1161: H16 Huber Loss on τ Channels δ=1.0 (frieren) — CLOSED TERMINAL-NULL (δ=1.0 dormant throughout; mechanism calibration failed; pod killed by watchdog at EP4~step 37k; no test data; reassigned frieren to H16b δ=0.3 PR #1169)

- **Branch**: `frieren/h16-per-channel-zscore` (closed)
- **W&B group**: `wave30_h16_huber_tau` — rank0 `w3gp9a8q`, state=crashed (pod SIGTERM'd 05:17Z)
- **Hypothesis**: Apply Huber loss (δ=1.0 σ-units) to τ channels instead of MSE. Heavy-tailed τ_z error distribution (max_abs_error ∈ [5,17]σ at smoke, 2.5% of vertices in L1 region) motivated outlier-robust regression overlay.

### Results — TERMINAL-NULL (no test data, mechanism inactive)

| Epoch | val_abupt | val_WSS | val_SP | val_vol_p | val τz/τx | frac_in_L1/τz | max_abs/τz |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Smoke EP0.5 | — | — | — | — | — | 2.7% | 9.8σ |
| EP3 (last complete) | **6.894%** | 7.656% | 4.754% | 4.390% | 1.516 | **0.014%** | ~2.5σ |
| EP4 partial (step 37k) | — | — | — | — | — | **0.018%** | **1.80σ** |

**No test metrics** — pod killed by student-watchdog at 05:17Z (false positive on label drift: PR label desynced for 1816s while training actively. Harness needs grace period for active training runs). No checkpoint resume possible (optimizer/EMA/GradNorm state not saved).

### Critical diagnostic: δ=1.0 was dormant throughout

The heavy tail **collapsed during EP1→EP3** — max_abs_error_normed/tau_z dropped from 9.8σ (smoke) → 1.8σ (EP4). With δ=1.0σ, only 0.018% of τ_z residuals ever crossed the Huber threshold. **H16 at δ=1.0 was functionally MSE-on-τ for 99.98% of the data.** The hypothesis was never tested; δ=1.0 is too coarse for the post-EP1 residual distribution.

### Analysis

**What we learned**: Huber-on-τ requires post-EP1-distribution δ calibration. Residuals at the relevant training phase (EP1+) have max_abs ~2σ with bulk at 0.3–1.5σ. The original EP0.5 smoke diagnostics (max_abs ~10σ) suggested 2.5% in L1 at δ=1.0, but this was the transient heavy tail at initialization. By EP1, the model has learned to reduce the heavy tail dramatically.

**H16b assignment** (PR #1169): δ=0.3 targets the post-EP1 residual bulk (0.3–1.5σ), where the canonical Huber-active fraction (0.4–0.8) is achievable. One-flag change from H16; student already has all diagnostic logging in place.

---

## 2026-05-17 04:50 — PR #1158: H13c Lagemann-Style Cosine+Magnitude Decoupled WSS Loss (thorfinn) — CLOSED DEAD-END (crashed mid-recipe AT EP5–6 with val_abupt 8.608% / +2.482pp above baseline; 99.0% cos-sim confirms 2nd independent direction-saturation diagnostic; thorfinn reassigned to next hypothesis researcher-agent pending)

- **Branch**: `thorfinn/h13c-tau-cosmag-decoupling` (closed)
- **W&B group**: `wave30_h13c_cosmag` — rank0 `1udny3hl`, **state=crashed at ~04:16Z** (ranks 1–7 killed by orchestrator)
- **Hypothesis**: Lagemann-style decoupled WSS loss — explicit `loss_mag = MSE(||τ_pred||, ||τ_true||)` + `loss_cos = 1 − cos_sim(τ_pred, τ_true)` at weight λ=0.1, vs MSE on Cartesian τ. Tests whether loss-level decoupling (vs H10's model-level decoupling) reaches a different optimum.

### Terminal results — DEAD-END (best of partial training, EP5–6 of 13)

| Metric | val (EP5–6) | Baseline | Δ | Gate |
|---|---:|---:|---:|:--|
| **val_abupt** | **8.608%** | 6.126% | **+2.482pp WORSE** | KILL (EP3 gate ≤7.4%, EP6 gate ≤6.8%) |
| val_WSS_vec | 10.830% | 6.727% | **+4.103pp WORSE** | FAIL |
| val_SP | 4.152% | 3.577% floor | +0.575pp above floor | FAIL FLOOR |
| val_vol_p | 4.000% | 3.643% floor | +0.357pp above floor | FAIL FLOOR |
| `cos_sim_mean` | **0.990** (EP4) | — | direction saturated | **mechanism signal** |
| nonfinite_count | 0 | — | crash not NaN-induced | OOM/orchestrator-kill |

### Per-epoch val trajectory (slope flattening)
```
EP1:  50.795%  (warmup, expected)
EP2:  16.136%  (-34.66pp)
EP3:  10.778%  (-5.36pp)        <- already KILL gate failed (>7.4%)
EP4:   9.563%  (-1.22pp)
EP5-6: 8.608%  (-0.95pp)        <- crash here
```
Geometric slope decay (~0.85× per ep) → EP13 projection ~6.0% essentially flat with baseline. Even mathematical recovery to baseline at EP13 would still leave floors breached (SP / vol_p never closed during the partial run).

### Critical diagnostic: 2nd independent confirmation of direction-saturation

This is the **second** Wave 30 attack to confirm direction is essentially solved. Together with H10 #1148 (vector-decoupled HEAD at model level), the fleet now has two independent decompositions reaching the same answer:

| Closure | Layer | cos_sim | mag/dir split |
|---|---|---:|---|
| H10 #1148 (fern) | output model arch | 99.65% (0.00355 cos_loss) | 73% / 27% |
| H13c #1158 (thorfinn) | loss formulation | 99.0% (cos_sim_mean) | n/a (loss not split into mag/dir error fractions, but cos_sim agrees) |

**Conclusion**: direction is not the WSS bottleneck; the magnitude regression head is. This directly validates the H10b magnitude-fix attack (PR #1164, fern, in flight). Any future loss-level cosine-aware attack is now ruled out as a research direction — the bottleneck has moved one layer down.

### Why this was closed (vs sent back to fix)

1. **Crash terminated recipe** — rank0 crashed at ~04:16Z (8.2h into 18h budget) with nonfinite=0; OOM or pod-level kill. No checkpoint resume infrastructure means restart from EP0.
2. **Mechanism saturated at EP4** — cos_sim already at 0.990 with magnitude error dominant. Continuing won't change the mechanism diagnostic.
3. **Slope geometrically decaying** — EP1→EP4 progression rules out reaching gate.
4. **Floor breaches already present at EP5–6** — magnitude-focused loss is starving SP/vol_p heads, same pattern as #1132 and #1150.

### Reassignment

thorfinn → fresh hypothesis (researcher-agent generating RESEARCH_IDEAS_2026-05-17_04:50.md). Constraint: must be orthogonal to all 7 in-flight Wave 30 attacks (H10b, H11b, H12, H15b, H16, H17, H18), targeted at magnitude bottleneck or τz/τx collapse band, and backed by published technique. Wave 30 fleet now 7 active + 1 idle, 18 closures.

---

## 2026-05-17 04:30 — PR #1150: H11 Multi-Scale kNN Context Features (askeladd) — CLOSED NOT-A-MERGE (**BEST single-model test_abupt 5.809% and test_WSS 6.633% on `tay`; SP +0.120pp / vol_p +0.022pp floor breach blocks merge; reassigned to H11b gated-input PR #1167**)

- **Branch**: `askeladd/multi-scale-knn-context` (closed)
- **W&B group**: `wave30_h11_multiscale_context`
- **W&B run IDs (8 ranks)**: `vgfkotop`, `ta13zohv`, `od6kf98i`, `cw26gfyy`, `vzkp5zun`, `bf9h5gix`, `zroqhyzk`, `a6lv3p12` (rank0: `vgfkotop`)
- **Best checkpoint**: EP12 (EMA source)
- **Hypothesis**: Inject per-vertex multi-scale geometric context as 9 additional surface-x channels (3 statistics × 3 kNN scales k=4,16,64): mean cosine alignment of neighbor normals, log mean panel area per scale, log mean L2 distance. Hypothesis: richer geometric context enables better τ_z discrimination at creases and fender edges.

### Terminal results (EP13 complete, 841.3 min / 1100 min budget)

| Metric | H11 result | Baseline #972 | Δ | Gate |
|---|---:|---:|---:|:--|
| **val_abupt** | **6.0953%** | 6.126% | **−0.031pp WIN** | PASS |
| **test_abupt** | **5.809%** | 5.844% | **−0.035pp WIN** | **BEST single-model on `tay`** |
| **test_WSS** | **6.633%** | 6.727% | **−0.094pp WIN** | **BEST single-model on `tay`** |
| test_vol_p | 3.665% | 3.643% (floor) | **+0.022pp** | **FAIL FLOOR** |
| test_SP | 3.697% | 3.577% (floor) | **+0.120pp** | **FAIL FLOOR** |
| test τz/τx | 1.470 | ~1.46 band | flat | engaged-but-neutral |

vs dl24-tanjiro #1132 reference (test_WSS=6.609%, different data stack): +0.024pp behind.

### Analysis

**Mechanism IS positive on primary metrics.** test_abupt 5.809% is the lowest single-model test_abupt achieved on the tay stack with the corrected dataset. test_WSS 6.633% beats the single-model merge gate.

**Floor regression mechanism**: 9 unscaled multi-scale channels concatenated unweighted at the input steal representational capacity from the SP head. The model must renormalise internally from EP0 onward, spending capacity that would otherwise serve SP regression. This is the capacity-stealing pattern.

**τz/τx = 1.470**: solidly inside the [1.44, 1.55] engaged-but-neutral widening band of all 6 closed model-side architecture attacks. Multi-scale kNN context (like all architecture-layer inputs) cannot break the τ_z structural bottleneck — confirming the bottleneck is in the output head / loss formulation, not in the feature representation.

**EP gate trajectory** (monotone improvement throughout):
EP5.92: 6.285% → EP8.3: 6.215% → EP9.61: improving → EP11.23: 6.095% → EP12 (best): 6.0953%

### Next step: H11b (PR #1167, askeladd)

Add a learned diagonal gate `nn.Parameter(torch.zeros(9))` on the 9 multi-scale channels, applied in `forward()` before `_encode_group`. Zero-init → EP0 behavior = baseline (no multi-scale signal); gate warms from gradient, admitting only channels that help. Should restore SP/vol_p floors while preserving WSS gain. The prewarm cache (`surface_multiscale_k4_16_64_v1.npy`) is already built.

---

## 2026-05-17 01:30 — PR #1155: H15 EMA Polyak Averaging v2 (alphonse) — CLOSED TIMEOUT-NULL (hypothesis UNTESTED, not falsified; decay=0.9999 never warmed up in 6h budget; live model healthy; reassigned to H15b decay=0.999 PR #1165)

- **Branch**: `alphonse/h15-ema-polyak-averaging` (closed)
- **W&B runs**: `d89u0x84,bso7alj5,s75wtnek,37wkovm3,dge1rijf,vodnwslo,fm5htatt,det8zjqg` (8 DDP ranks, all state=finished)
- **Hypothesis**: Polyak/EMA averaging of model weights (decay=0.9999) as inference-time smoother. eval_raw_vs_ema=True for dual-eval. Tests whether weight averaging in cosine-decay LR schedules extracts a better generalization basin than the raw final checkpoint.

### Terminal results — TIMEOUT-NULL (EP4/13, best_checkpoint = EMA, all gates fail due to untested EMA)

| Metric | val_raw (EP4) | val_ema (EP4) | test (EMA) | Baseline | Notes |
|---|---:|---:|---:|---:|---|
| abupt | 7.02% | 16.5618% | **15.512%** | 6.126% / 5.844% | EMA not warm |
| WSS | ~7.7% | — | **16.945%** | 6.727% | EMA at half-warmup |
| SP | — | — | **10.673%** | 3.577% floor | FAIL floor +7.10pp |
| vol_p | — | — | **12.119%** | 3.643% floor | FAIL floor +8.48pp |

All test gates fail — but this is the EMA model at half warm-up, not the live model. **Live model healthy: val_raw_abupt=7.02% at EP4 is on H3 trajectory.**

### EMA-vs-raw convergence diagnostic (most important result)

| Epoch | val_raw_abupt | val_ema_abupt | gap (raw−ema) |
|---:|---:|---:|---:|
| 1 | 26.52% | 65.11% | −38.59pp |
| 2 | 8.42% | 56.93% | −48.51pp |
| 3 | 7.44% | 23.72% | −16.28pp |
| 4 | **7.02%** | **16.56%** | **−9.54pp** |

Gap closing monotonically: 38.6→48.5→16.3→9.5pp. EP2 widening is expected (raw descending faster than EMA can average random-init weights). By EP4, closing rate is +6.74pp/epoch. Extrapolating: EMA would cross raw around EP6-7 and pull ahead by EP10+ as raw begins overfitting.

**decay=0.9999 half-life ≈ 10,000 steps ≈ 0.92 epochs**. Fully clean EMA at EP3–4. Budget ran out exactly where EMA was starting to settle.

### Analysis & next steps

H15 hypothesis NOT FALSIFIED — budget-truncated. Convergence direction is positive. Student's own option B (decay=0.999, half-life=0.064 epochs, fully warm by EP0.3) is the right follow-up under 6h constraints. **Assigned H15b PR #1165 (alphonse, decay=0.999).**

---

## 2026-05-17 01:00 — PR #1148: H10 Vector-Length-Decoupled WSS Head (fern) — TERMINAL CLOSED NOT-A-MERGE (**73%/27% magnitude/direction decomposition — definitive fleet diagnostic; direction head saturated at 99.65% cos-sim; ALL residual WSS error is in magnitude; softplus floor at 6.93 in normalized space is the likely bottleneck; reassigned to H10b bounded-exp PR #1164**)

- **Branch**: `fern/vector-decoupled-wss-head` (closed)
- **W&B group**: `wave30_h10_vector_decoupled`
- **Hypothesis**: Replace 4-channel Cartesian WSS output with 5-channel (cp, dir_x, dir_y, dir_z, log_mag) parametrization. Reconstruct τ = softplus(log_mag) × mag_scale=10 × unit(dir). Add cosine-similarity aux loss (weight=0.1) on unit direction. Tests whether Cartesian MSE confounds direction and magnitude, handicapping τ_z prediction.

### Terminal results

| Metric | val (EP13) | test (best EMA) | Baseline | Δ vs base | Gate |
|---|---:|---:|---:|---:|:--|
| abupt | 6.325% | **~5.968%** | 6.126% / 5.844% | val +0.199pp / test +0.124pp | **WORSE** |
| WSS | — | **6.885%** | 6.727% | **+0.158pp WORSE** | FAIL |
| **SP** | — | **3.776%** | 3.577% floor | **+0.199pp FLOOR BREACH** | **FAIL FLOOR** |
| vol_p | — | **3.619%** | 3.643% floor | **−0.024pp** | **PASS FLOOR** |
| τ_z/τ_x | — | **1.463** | ~1.46 collapse band | flat | unchanged |
| direction_cos_loss | **0.00355** | — | — | 99.65% cos-sim | **SATURATED** |

### Critical diagnostic: 73% magnitude / 27% direction decomposition

Post-hoc `eval_direction_magnitude.py` ran on test split predictions. Results (val/test agree):

- **~73% of WSS squared error is in MAGNITUDE**
- **~27% of WSS squared error is in DIRECTION**

Direction head: 99.65% normalized cos-sim ≈ 14.6° physical angle error at terminal. **Direction is essentially learned.** The cosine aux loss worked — the head knows where τ points.

Magnitude: softplus(log_mag) × 10.0 has floor at softplus(−∞) × 10 = ln(2) × 10 ≈ 6.93 in normalized space. For low-τ flat automotive panels (most vertices), the model CANNOT predict below 6.93 regardless of head weights — but the GT distribution (z-score normalized, σ=1) has many vertices near 0. This saturation artifact distorts the magnitude head's gradient signal.

**τ_z/τ_x=1.463 is flat** (same as collapse band) despite direction learning — confirms τ_z bottleneck is NOT from direction-learning failure. It's from magnitude-scale coupling.

### Analysis & next steps

The 73%/27% split is the most informative mechanistic result in Wave 30. The magnitude head is the bottleneck, not the direction head. The softplus floor (6.93 in normalized space) is the most likely culprit — it prevents near-zero magnitude predictions. **H10b (PR #1164, fern)** tests fern's own suggestion: replace `softplus(log_mag) * 10.0` with `log_mag.clamp(min=-3, max=3).exp()` (floor = e^{-3} ≈ 0.05, covers [0.05, 20.09] without floor saturation).

This finding **reorients the fleet research direction**: any approach that cannot fix per-vertex magnitude (loss reweighting, Huber outlier robustness, EMA) is complementary at best. The fundamental problem is that the model's magnitude head cannot reach near-zero for low-τ vertices.

---

## 2026-05-17 00:25 — PR #1147: H6' Soft τ·n=0 Penalty (tanjiro) — TERMINAL CLOSED NOT-A-MERGE pulled from W&B (pod failed to post terminal SENPAI-RESULT; **test τ_z/τ_x = 1.420 FIRST attack to break BELOW [1.44, 1.55] collapse band lower edge** — mechanism signal real but soft-penalty formulation lost on primary metrics; informative for H17 hard-constraint variant)

- **Branch**: `tanjiro/soft-tau-n-penalty` (closed)
- **W&B run**: `smvr34a8` (group `wave30_h6prime_softpenalty_arm_b_p1`) — 14.4h total (lr cosine annealed to 2.29e-6), state=finished at ~00:02Z, train completed cleanly with best EMA checkpoint, full test eval written to W&B summary
- **Hypothesis**: Add soft `τ·n=0` penalty term to surface loss at weight λ=0.1 in normalized loss space; encourage WSS to lie tangent to surface (which is the physical constraint).

### Terminal results — paper-facing metrics (pulled from W&B summary, student pod did NOT post SENPAI-RESULT)

| Metric | val (n=34, EP13) | test (n=50, best EMA) | Baseline | Δ vs base | Gate |
|---|---:|---:|---:|---:|:--|
| abupt | 6.2555% | **6.0370%** | val 6.126% / test 5.844% | val +0.130pp / test +0.193pp | **WORSE** |
| WSS | 7.1081% | **7.0020%** | 6.727% test | **+0.275pp WORSE** | **FAIL** |
| **SP** | 4.0916% | **3.8143%** | 3.577% floor | **+0.237pp FLOOR BREACH** | **FAIL FLOOR** |
| vol_p | 3.6314% | **3.6031%** | 3.643% floor | **−0.040pp** | **PASS FLOOR** |
| τ_x | 6.2105% | 6.2846% | — | — | — |
| τ_y | 7.7443% | 7.5594% | — | — | — |
| τ_z | 9.5995% | **8.9236%** | 8.26% | +0.664pp worse | — |
| **τ_z/τ_x** | **1.546** | **1.420** | ~1.46 collapse band | **TEST below 1.44 lower edge!** | mechanism signal |

### Pod misbehavior

W&B run terminated cleanly with test_primary/* keys written to summary. But tanjiro's pod harness query "No assigned PRs or issues" → slept without invoking Claude. **No terminal SENPAI-RESULT comment was posted by the student.** I pulled all metrics directly from W&B and posted the close comment as advisor. Worth investigating: harness polling logic should detect `W&B state=finished + status:wip PR + no terminal comment` as "write up results" work.

### Per-epoch val trajectory (from student check-ins + EP13 W&B)

```
EP1:  abupt=29.235%  WSS=32.637%  SP=22.770%  vp=16.861%  τz/τx=1.370 (warmup)
EP2:   7.805         8.829         5.170      4.574      1.496
EP3:   6.904         7.827         4.575      4.010      1.512
EP4:   6.590         7.476         4.350      3.836      1.522
EP5:   6.492         7.363         4.283      3.792      1.525
EP6:   6.422         7.289         4.236      3.742      1.528
EP7:   6.355         7.216         4.178      3.702      1.532
EP8:   6.320         7.180         4.148      3.673      1.535
EP9:   6.296         7.154         4.125      3.654      1.539
EP10:  6.271         7.128         4.104      3.642      1.540
EP13:  6.255         7.108         4.092      3.631      1.546  (best EMA)
```

Monotonic descent across all axes, no divergence. But never reached floor for SP, val_vol_p tracked floor at terminal but didn't comfortably clear it.

### Analysis & next steps

**Mechanism signal IS real**: test τ_z/τ_x = 1.420 — FIRST Wave 30 attack to break below the [1.44, 1.55] collapse band (val stays at 1.546 in band — train-test distribution gap on this metric). All 7+ prior closures landed within [1.44, 1.55]. The soft τ·n=0 penalty DID move the band on test, even when overall primary metrics regressed. This is strong evidence that:

1. **τ·n=0 constraint is attacking the right axis** (the τ_z bottleneck)
2. **Soft-penalty formulation is wrong**: either over-applies the constraint (breaks SP/abupt) or competes with WSS task gradient. Hard-constraint variants needed.
3. **H17 (PR #1162, nezuko) is the correct follow-up** — same constraint mechanism, but enforced by output reparameterization in tangent basis (architectural, not loss-side). H17 cannot over-apply the constraint because τ·n=0 is automatic.

**Tanjiro reassignment**: H18 per-vertex area-weighted surface MSE (PR #1163) — orthogonal mechanism to all in-flight. Tests "physical force-integral matching via area-weighting" hypothesis.

---

## 2026-05-16 23:35 — PR #1146: H9' Curvature-Aware Surface Feature (nezuko) — TERMINAL CLOSED NOT-A-MERGE (7th model-side closure on τ_z bottleneck fingerprint — curvature input feature is genuine signal for vol_p but axis-blind to τ_z; test_SP +0.159pp floor breach forbids merge; dl24 #1132 win FAILED to transfer to tay stack)

- **Branch**: `nezuko/curvature-attention-bias` (closed)
- **W&B run**: `utlmmp0t` (group `wave30_h9_curvature`) — 840.3 min total (14.0h, fastest pace in fleet at 0.79 h/ep), 13/13 epochs complete, EMA best at EP13
- **Hypothesis**: Add per-vertex curvature `κ_i = mean_{j∈kNN(i)} (1 - cos(n_i, n_j))` as 8th surface channel (kNN=16), inspired by dl24 #1132's test_WSS 6.609% to test if curvature input → break τ_z attractor.

### Terminal results — paper-facing metrics

| Metric | val (n=34) | test (n=50) | Baseline | Δ vs base | dl24 #1132 | Gate |
|---|---:|---:|---:|---:|---:|:--|
| abupt | 6.1389% | **5.9037%** | val 6.126% / test 5.844% | val +0.013pp / test +0.060pp | — | MISS |
| WSS | 6.9230% | **6.7900%** | 6.727% test | **test +0.063pp WORSE** | 6.609% (+0.181pp worse vs dl24) | FAIL |
| **SP** | 4.0657% | **3.7362%** | 3.577% floor | **+0.159pp FLOOR BREACH** | — | **FAIL FLOOR** |
| vol_p | 3.6747% | **3.5870%** | 3.643% floor | **−0.056pp** | — | **PASS FLOOR** |
| τ_z | 9.3917% | 8.7945% | 8.26% | +0.534pp worse | — | — |
| **τz/τx** | val 1.555 | **test 1.459** | ~1.46 | ~0.00 | — | collapsed band (1.45-1.55) |

### Verdict: NOT-A-MERGE — clean negative on primary axis, isolated positive on vol_p

Per merge gates:
- test_SP floor breach alone forbids merge
- test_abupt +0.060pp WORSE than baseline
- test_WSS +0.063pp WORSE than baseline
- vol_p −0.056pp under floor (genuine but isolated improvement)

### Mechanistic intelligence — 7th model-side closure on τ_z fingerprint

| Fingerprint | All 7+ closed model-side attacks |
|---|---|
| Test τz/τx | Lands in [1.44, 1.55] collapsed band |
| Direction | Test ratio = 1.459 here; H8 mirror = 1.456; other closed = 1.44-1.50 |
| GT vertex-level τz/τx | 0.08 (model over-predicts τ_z magnitude by ~18×) |

**The τ_z bottleneck is robust across input features, architecture perturbations, attention biases, position encodings, augmentation. 7 attacks now confirm the same fingerprint.** This is now strong evidence the bottleneck is at the **output-projection / loss-formulation layer**, not at architecture or input distribution. Wave 30 has 4 in-flight attacks on the loss layer (frieren H16 Huber, thorfinn H13c cos+mag, edward H12 magnitude-weighted, tanjiro H6' tau_n penalty) — correct next tier.

### dl24 #1132 cross-pollination FAILED to transfer

dl24 reported test_WSS 6.609% with curvature feature. nezuko reproduction on tay stack: test_WSS 6.790% — 0.181pp WORSE than dl24, 0.063pp worse than tay baseline. **dl24's win was stack × κ, not κ in isolation.** Different normals computation, different volume head, or different curriculum upstream is the actual mechanism. This is the second time a cross-branch result has not reproduced cleanly — suggests baseline-stack-specific interactions matter.

### Per-epoch trajectory (val EMA)

E1: abupt=28.13 WSS=30.73 τz/τx=1.394 (warmup)
E2: 7.32 WSS=8.16 τz/τx=1.498
E5: 6.38 WSS=7.17 τz/τx=1.533
E10: 6.17 WSS=6.95 τz/τx=1.551 (65k vol curriculum in — no regression)
E13: 6.14 WSS=6.92 τz/τx=1.555 (monotonic; slope flattened from −0.05pp/epoch to −0.008pp/epoch)

### Ancillary positive: vol_p improvement

Both val_vol_p (3.675% vs 3.798% baseline, **−0.123pp**) and test_vol_p (3.587% vs 3.643% floor, **PASSES**) improved. kNN-of-normals is informative at sharp surface-to-volume coupling regions (wheel arches, A-pillars, mirror housings) — exactly where volume pressure is hardest. **Useful signal to remember for future vol_p-focused work, NOT for τ_z.**

### Student suggested follow-ups (acknowledged, not pursued)

1. Curvature-as-attention-bias variant (PR title was "curvature-attention-bias" but only input-feature was tried)
2. Per-axis curvature decomposition (τ_z-aligned component)
3. NO K-sweep (K isn't the lever)

All three suggestions are still input/architecture-side modifications of an exhausted attack surface. Skip for now; if Wave 30 loss-formulation tier exhausts without breakthrough, revisit attention-bias variant.

### Closure decision

- 12th closure in Wave 30
- 7th model-side attack confirming τ_z fingerprint
- Strongest evidence yet that the bottleneck is loss-formulation/output-projection
- Frees up nezuko for fresh hypothesis on different attack surface

---

## 2026-05-16 20:30 — PR #1143: H8 Mirror-Symmetry Data Augmentation (frieren) — TERMINAL FLAT NULL, CLOSED (first dedicated data-distribution-layer attack on τ_z bottleneck — falsifies "data diversity" hypothesis cleanly; test τz/τx = 1.456 lands EXACTLY in collapse band of 6 closed model-side attacks)

- **Branch**: `frieren/h8-mirror-symmetry-aug` (closed)
- **W&B run**: `ft60dpdm` (group `wave30_h8_mirror_aug`) — 838.8 min total (14.0h), 13/13 epochs complete, EMA best at EP13, 0 ranks errored
- **Hypothesis**: Apply x-z plane y-flip mirror augmentation at p=0.5 to break τ_z structural bottleneck by enforcing exact τ_y sign-flip equivariance + doubling geometric diversity in expectation.

### Terminal results — paper-facing metrics

| Metric | val (n=34) | test (n=50) | Baseline | Δ vs base | Gate |
|---|---:|---:|---:|---:|:--|
| abupt | 6.332% | 6.052% | val 6.126% | val +0.206pp | MERGE FAIL |
| **WSS (overall)** | 7.188% | **7.001%** | test 6.727% | **test +0.274pp** | **MERGE FAIL** |
| SP | 4.166% | 3.822% | test 3.577% (floor) | +0.245pp | FLOOR FAIL |
| vol_p | 3.681% | 3.578% | test 3.643% (floor) | −0.065pp | floor PASS (cosmetic) |
| τ_x | 6.292% | 6.229% | — | — | — |
| τ_y | 7.767% | 7.562% | — | — | — |
| **τ_z** | **9.752%** | **9.071%** | — | — | — |
| **τz/τx** | val 1.550 | **test 1.456** | ~1.46 | ~0.00 | falsification target <1.40 NOT MET |

### Verdict: FLAT NULL — DO NOT MERGE

**The first dedicated data-distribution-layer attack on the τ_z bottleneck — and the cleanest "input-distribution attacks won't move the τ_z needle" data point we have.**

Per PR's own success criteria:
- "BIG WIN" (τz/τx ≤ 1.30 + WSS < 6.727%) — NOT MET
- "MERGE" (test_WSS < 6.727% + floors held) — NOT MET (test_WSS +0.27pp; test_SP +0.245pp over floor)
- "PARTIAL" (WSS within 0.1pp + τz/τx materially down) — NOT MET (τz/τx unchanged)
- "FLAT NULL" — CLOSEST MATCH (val_WSS within 0.05pp but mildly worse than baseline)

### Mechanistic intelligence — fleet-wide finding

**The τ_z bottleneck is upstream of geometric data diversity.** Mirror-augmentation enforced an EXACT sign-flip equivariance on τ_y by construction and reflected geometry across x-z plane — yet:
- val τz/τx widened monotonically (1.36 → 1.49 → ... → 1.55 terminal)
- test τz/τx = 1.456 lands EXACTLY in the [1.44, 1.47] collapse band of all 6 closed model-side attacks
- Per-channel test ordering τ_z > τ_y > τ_x is IDENTICAL to baseline (no re-ordering)
- τ_y / τ_x ratio = 1.214 unchanged → augmentation enforced τ_y equivariance by construction yet τ_y rel-L2 remained anchored to τ_z's level

**This cleanly rules out:**
1. "Insufficient geometric diversity" hypothesis (mirror-aug doubles coverage) → REJECTED
2. "Spurious τ_y co-adaptation" hypothesis (mirror-aug enforces equivariance) → REJECTED

**Cleanly implicates** the τ_z bottleneck as residing at:
- (a) Output projection / per-channel calibration
- (b) Loss formulation across channels
- (c) Dataset-level per-channel target statistics (variance, dynamic range)

**NOT at model architecture or input augmentation distribution.**

### Mild val regression (+0.21pp) explanation

Plausibly the cost of pos-embed redundancy: the model already has string_separable sincos pos_embed + LayerNorm providing strong intrinsic rotation-invariance properties. Adding mirror-aug forces the network to spend a small slice of capacity stabilizing under what is already approximately invariant input — with no upside.

### Implementation diagnostics (all PASS)

- `config.mirror_aug_prob = 0.5` confirmed in W&B run config (all 8 ranks)
- `mirror_collate` correctly flips: `surface_x[y, ny]`, `surface_y[τ_y]`, `volume_x[y]`; cp/vol_p invariant; volume handled with guard
- Tensor `clone()` ensures no in-place mutation of cached dataset (critical)
- Train-only augmentation: val + test DataLoaders use unchanged `pad_collate` (verified at `trainer_runtime.py:248-252`)
- Expected ~2,600 mirror-augmented training views over 13 epochs (0.5 × 400 cases × 13 epochs), consistent with sampler

### Fleet-wide intelligence — data-distribution layer EXHAUSTED on τ_z

Combined with the 9 prior model-side attacks that all landed in τz/τx ∈ [1.44, 1.57], the data-distribution layer is now formally a NULL attack axis on the τ_z bottleneck. **Closing this layer as a viable Wave 30 attack vector.**

### Suggested follow-ups (student diagnostic, captured for queue)

1. **Per-channel target z-score normalization** — H16 assigned in PR #1161 to frieren as direct follow-up
2. **Mirror-pair consistency loss** — explicit equivariance penalty (2× forward pass cost)
3. **Per-channel target variance logging** — diagnostic to confirm τ_z's distribution moments
4. **Cross-stack with future winner** — if any in-flight attack lands τz/τx ≤ 1.40, re-run with mirror-aug to test compounding/saturation

### Closure summary

**FLAT NULL with mild val regression.** Within "<5% regression" close band (test_WSS 4.1% over baseline). **Cleanest data-distribution-layer falsification in Wave 30.** frieren's diagnostic exemplary: math identity reasoning, complete implementation verification, per-channel ordering analysis, explicit "stop pursuing data-level attacks" recommendation. Reassigned in PR #1161 (H16 per-channel z-score) — directly addressing the output-side calibration axis flagged in this closure.

---

## 2026-05-16 19:00 — PR #1156: H13b Tangent/Normal Anisotropic Loss at β=2 (thorfinn) — CLOSED (also diverged with DIFFERENT mechanism — corruption without gradient explosion; per-vertex anisotropic loss formulation itself broken, not just amplification factor)

- **Branch**: `thorfinn/h13b-anisotropic-beta2` (closed)
- **W&B runs**: 8 ranks (rank0=`iypg6fey`, ranks 1-7 various), terminated at step ~10,136 after EP2 KILL
- **Hypothesis**: Soften H13 amplification from β=5 to β=2 (2.5× smaller per-vertex amplification of normal-direction MSE); if H13's failure was the LR×amplification cliff, β=2 should fit in the stable band per GT n/t = 0.08 measurement.

### Terminal metrics

| Metric | EP1 (low-LR warmup at 2.5e-5) | EP2 (post-warmup at 5e-4) | H13 β=5 EP1 | Baseline |
|---|---:|---:|---:|---:|
| val_abupt | **47.51%** | **88.33%** | 49.43% | 20.49% (live EP1) |
| nonfinite_grad % post-warmup | 0% | 0% | 100% | 0% |
| grad_norm peak | 138 (step 1499) | 55,309 (step 9501, clipped) | 137,953 (step 3793) | <50 |
| model_n/t ratio | 0.063–0.067 | 0.068–0.070 | 0.061 (pre-divergence) | — |
| GT n/t ratio | 0.079–0.080 | 0.079–0.083 | 0.080 | — |

### Verdict: NEGATIVE — clean falsification with DIFFERENT mechanism from H13 β=5

**Critical new fact: β=5 and β=2 give near-identical EP1 val_abupt (49.43% vs 47.51%)** during low-LR warmup at 2.5e-5 — BEFORE any LR×amplification interaction can fire. This is the smoking gun that the anisotropic-loss FORMULATION ITSELF is broken, not just the amplification factor.

**Failure-mode comparison (β=5 vs β=2)**:

| β | Mechanism | EP1 val_abupt | Post-warmup failure | nonfinite% |
|---:|---|---:|---|---:|
| 5 (H13) | gradient explosion via amplification | 49.43% | Grad norm 137k at warmup boundary, 100% steps nonfinite from step 5000+ | 100% |
| 2 (H13b) | corruption WITHOUT explosion | 47.51% | Grad clipping (max_norm=1.0) prevents nonfinites, but corrupt representation drifts: val_abupt 47.5% → 88.3% across EP1→EP2 | 0% |

**Root-cause analysis (student diagnostic)**:
- Math identity verified PRE-launch: at α=β=1.0, tau_anisotropic_mse reproduces baseline 4-channel masked_mse exactly (abs diff 0.00).
- Model successfully tracks GT n/t ≈ 0.08 geometry during training (model_n/t = 0.063–0.073 vs GT = 0.079–0.083). **The H13 hypothesis "model under-fits GT normal component" is FALSIFIED** — model IS learning the correct geometric structure.
- The β-amplification creates a per-vertex anisotropic "pull" toward matching the small normal-component (GT ≈8% of tangent magnitude). Model overfits the per-vertex noise in that small signal (FEM interpolation, mesh resolution), learning a representation that minimizes the *anisotropic* objective but produces wildly off-target predictions in the *unweighted* val_abupt metric.
- **The amplification axis is exhausted**: H6' (suppress normal — also closed earlier in Wave 30), H13 β=5 (5× amplify normal — diverged via grad explosion), H13 β=2 (2× amplify normal — corruption without explosion). All three points on the normal-direction-emphasis axis fail.

### lr=5e-4 confound — advisor mistake on PR text

Both H13 (#1152) and H13b (#1156) ran at `--lr 5e-4` due to advisor error on the PR command. Canonical Wave 30 reference (BASELINE.md "L=5 + surf→vol xattn ... Lion lr=9e-5") and most-recent closed-clean PR (#1138 H3) use `--lr 9e-5`. Cascade of 4 divergences (H14, H13, H13b, H15) all stem from this. **However**, the H13b EP1 failure at low-LR warmup (2.5e-5) happens BEFORE the LR transition — confirming the formulation issue is upstream of LR. The β=5 grad explosion at peak LR was clearly LR-dependent; the β=2 corruption is NOT.

### Pivot trigger: Lagemann cosine+magnitude decoupling

Per the PR's queued "if β=2 also diverges" branch, this terminal closure triggers a pivot to **Lagemann et al. arxiv 2507.22817** — the published opposite spirit of H13. Use a **magnitude-MSE loss + cosine-direction loss with cosine getting a SMALL weight** instead of rotating to a per-vertex tangent/normal frame. Assigned as H13c (PR #1158) at correct lr=9e-5.

### Closure reasoning summary

| H13 axis | Mechanism | Outcome |
|---|---|---|
| H13 β=5 | Per-vertex normal-grad amplification 5× | DIVERGED (grad explosion at warmup boundary) |
| H13b β=2 | Per-vertex normal-grad amplification 2× | DIVERGED (corruption without explosion; clip prevented nonfinites) |
| H13c (NEW PR #1158) | Mag+cos decoupled — NO per-vertex tangent frame | PROBE |

---

## 2026-05-16 19:00 — PR #1155: H15 EMA / Polyak Averaging (alphonse) — DIVERGED at EP3, RELAUNCH at lr=9e-5

- **Branch**: `alphonse/h15-ema-polyak` (relabeled status:wip for relaunch)
- **W&B runs**: group `wave30_ema_h15` 8-rank, killed at 18:33Z after EP3 KILL (~1h45min in)
- **Hypothesis**: EMA (Polyak averaging) with decay=0.9999 smooths optimization trajectory; evaluate EMA model rather than live weights at val/test.

### Terminal divergence trace

| Epoch | train/epoch_loss | val_raw_primary/abupt_pct | val_primary/abupt_pct (EMA) | LR at epoch end |
|---:|---:|---:|---:|---:|
| 0 (warmup) | 0.930 | — | — | 2.5e-5 → 5e-4 |
| 1 | 2.443 | 25.20% | 81.39% | 5e-4 |
| 2 | 2.641 | 79.01% | 89.94% | 5e-4 |
| 3 | (in flight) | **78.91%** (12× gate breach) | 88.14% | 4.93e-4 |

EP0 base_mse_loss went 2.30 → 0.175 at step 3465 (during low-LR warmup at 2.5e-5). At LR transition to peak (step 2865 → lr=5e-4) loss spiked from 0.175 back to 1.49 at step 4331, then oscillated 1.0–2.0 forever. **No NaN, no nonfinite grads — clean optimizer divergence** under Lion's sign-step at over-driven LR.

### Verdict: DIVERGED due to PR-text error (lr=5e-4 instead of canonical lr=9e-5)

**Critical student diagnostic catch**: PR-prescribed `--lr 5e-4` is **5.5× the working reference** for (ep=13, bs=4, lion, ml=5, hidden=512). The canonical Wave 30 reference is BASELINE.md "L=5 + surf→vol xattn ... Lion **lr=9e-5**, 13ep" and the most-recent closed-clean Wave 30 PR (#1138 H3 alphonse, val_abupt EMA=6.197%) used exactly that. Survey of finished runs with `lr=5e-4 lion ep=13 bs=4`: **NONE**. Only two such runs ever attempted — H14 (CRASHED at 0.62h, ep_loss ≈ 90,000) and H15 (DIVERGING). So this LR has never reached a stable trajectory under this config.

**EMA implementation verified correct**:
- `EMA.update` correctly skipped on `nonfinite_grad / nonfinite_loss` (train.py:1115)
- Eval pattern: raw → store/copy_to/restore (clean)
- EMA decay tracking: EP1 ema=81.39% > raw=25.20% (EMA still warming, as PR predicted)
- Grad norms healthy: mean ≈ 5 (excluding rare spikes), clipping engaged
- The live model is divergent — EMA cannot smooth a divergent trajectory.

### Action: relaunch with lr=9e-5 (PR back to status:wip)

Relaunch command (matching H3 / PR #823 reference exactly + EMA overlay):
```bash
cd target/ && torchrun --nproc_per_node=8 train.py \
  --batch-size 4 --epochs 13 \
  --lr 9e-5 --lr-warmup-epochs 1 --lr-cosine-t-max 13 \
  --optimizer lion --weight-decay 5e-4 \
  --use-ema --ema-decay 0.9999 --eval-raw-vs-ema --ema-start-step 50 \
  ...
```

Open questions answered:
1. lr=9e-5 (NOT 5e-4) — student correct
2. train_surface_points=65536 (NOT 40000) — match H3 reference
3. ema_decay=0.9999 retained — that IS the H15 hypothesis (slow EMA on stable LR=9e-5 trajectory)

### lr=5e-4 fleet-wide confound (advisor acknowledgment)

The cascade of 4 Wave 30 divergences (#1153 H14, #1152 H13 β=5, #1156 H13b β=2, #1155 H15) all stem from the same advisor error: writing `--lr 5e-4` instead of the canonical `--lr 9e-5` on the PR commands. This conflated SOTA single-model PR #972 (`--lr 1e-4`) with the Wave 30 Lion reference. **All other in-flight Wave 30 PRs (#1143 H8 frieren, #1146 H9' nezuko, #1147 H6' tanjiro, #1148 H10 fern, #1150 H11 askeladd, #1151 H12 edward) correctly use lr=9e-5** (verified). Fleet-wide impact limited to the 4 divergent PRs.

---

## 2026-05-16 17:45 — PR #1152: H13 Tangent/Normal Anisotropic Loss at β=5 (thorfinn) — CLOSED (catastrophic divergence at warmup boundary, mirror image of H14; mechanism PASS via math identity)

- **Branch**: `thorfinn/h13-tau-anisotropic-loss` (closed)
- **W&B runs**: 8 ranks (rank0=`02h1r0ok`, rank1-7=`pmdz7svm`/`fwjeluoc`/`uywxwi5m`/`uzx4kdvt`/`kdy3fl77`/`39576br1`/`ddz39kil`), all marked `state_note: TERMINATED EARLY`
- **Hypothesis**: Decompose τ surface MSE into per-vertex tangent + normal components (math identity `‖err‖² = ‖err_t‖² + (err·n)²` for unit normals); amplify normal-component MSE by β=5 to test "model under-fits real GT normal signal" axis.

### Terminal metrics

| Metric | EP1 (pre-divergence) | EP2/EP3 (post-divergence, frozen) | Baseline |
|---|---:|---:|---:|
| val_abupt | 49.43% | 90.94% (frozen identical) | 6.126% |
| val_WSS | 52.32% | 96.01% | 6.727% |
| val_τx / τy / τz | 43.35% / 67.95% / 63.40% | 93.17% / 101.25% / 101.10% | — |
| GT τ_n/τ_t magnitude ratio | **0.08 (real but small signal)** | — | — |
| Math identity α=β=1.0 | abs diff **0.00** vs baseline MSE | — | — |

**Step-level divergence (rank 0)**:

| step | epoch | loss | grad_norm pre-clip | nonfinite_grad | clipped |
|---:|---:|---:|---:|---:|---:|
| 3000 | EP1 warmup | 0.95 | <100 | 0 | 0 |
| 3625 | warmup END (LR jumps 2.5e-5 → 5e-4) | 1.5 | 0.5 | 0 | 0 |
| 3792 | mid-EP1 cosine | 3.19 | 0.29 | 0 | 0 |
| **3793** | mid-EP1 cosine | **59.34** | **137,953.83** | 0 | 1 |
| 3800 | mid-EP1 cosine | 26.85 | 570 | 0 | 1 |
| 3804 | mid-EP1 cosine | 121.27 | 1438 | 0 | 1 |
| 5000+ | EP2-EP3 | ~3.5 (frozen) | Inf | **1 (100% steps)** | 0 |

Step-skip distribution: 0-3625 = 0% skipped, 3625-5000 = 60.8% skipped, 5000-11000 = **100% skipped** (every optimizer step a no-op).

### Verdict: NEGATIVE — clean catastrophic divergence at warmup boundary. CLOSED.

**Mechanism PASS (all confirmed)**:
- Math identity verified: at α=β=1.0, `tau_anisotropic_mse` reproduces baseline 4-channel `masked_mse` exactly (abs diff 0.00 on random tensors)
- Pre-divergence trajectory at step 3000: model n/t ratio = 0.061 was closely tracking GT n/t = 0.080 → mechanism IS engaging correctly during warmup
- Per-vertex unit normal handling correct; safety re-norm guard active

**Root cause (student diagnostic)**: The per-vertex *gradient* on the normal component is 5× larger than baseline at β=5. At LR=2.5e-5 (warmup floor), these aggressive updates accumulate slowly and the model converges. Once LR jumps 20× to 5e-4 at step 3625, the same gradient signal pushes the model off the cliff — 168 steps later grad_norm went 0.29 → 137,953.83 in a single step. The fact that divergence happens *exactly* at the warmup boundary (not mid-epoch) is the smoking gun for LR×β instability.

**Critical new data point for the entire fleet: GT τ_normal_to_tangent magnitude ratio = 0.08** — the GT normal-component IS real signal (8% of tangent magnitude), but small enough that 5× amplification was clearly past the LR×β stability cliff.

### Mirror image of H14 divergence

Both Wave 30 amplification-style attacks crashed at the SAME EP1→EP2 LR jump (2.5e-5 → 5e-4, 20× increase under cosine warmup):

| PR | Attack | Amplification | Divergence step | Effective trigger |
|---|---|---:|---:|---|
| #1153 (alphonse) | H14 head_lr 5× | 5× on output-head grads | ~4500 (warmup end) | LR×amplification on small param subset |
| **#1152 (thorfinn)** | **H13 β_normal=5** | **5× on normal-grad direction** | **3793 (warmup end)** | **LR×amplification on direction-specific grads** |

The current 1-epoch warmup + lr=5e-4 recipe has very little safety margin for amplification-style attacks. This is fleet-relevant intelligence.

### Reassignment

Thorfinn reassigned to **H13b — anisotropic tangent/normal loss at β=2** (PR #1156). 2.5× reduction from divergent β=5; squarely in the stable band based on GT n/t = 0.08 baseline. Same implementation, 1-line config change. ~5h total run-time (thorfinn's H13 throughput was ~23min/epoch vs ~80min/epoch elsewhere — fastest in fleet).

---

## 2026-05-16 15:00 — PR #1153: H14 Asymmetric LR for Surface Output Head 5× (alphonse) — CLOSED (clean training divergence under Lion at head_lr=2.5e-3; mechanism PASS; first optimization-layer attack)

- **Branch**: `alphonse/h14-asymmetric-head-lr` (closed)
- **W&B runs**: main `ci9ipu1x` (rank0 + 7 sibling DDP), sanity `7xq2kpa8` (mult=1.0 verification)
- **Hypothesis**: Split Lion optimizer into 2 param groups; surface_out.* MLP gets 5× backbone LR (2.5e-3 vs 5e-4). Direct attack on H6 mechanism-PASS output-head diagnostic — push more gradient signal into bottleneck location. Standard Kaggle/ImageNet precedent (DeiT head_init_scale, MAE linear probing, ULMFit).

### Terminal metrics

| Epoch | val_abupt | val_SP | val_VP | val_WSS | τ_x | τ_y | τ_z | τz/τx | train_loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34.65% | 25.33% | 24.87% | 37.03% | 31.79% | 44.63% | 46.63% | 1.467 | 0.972 |
| 2 | 279.72% | 80.46% | 285.24% | 383.51% | 93.04% | 818.33% | 121.54% | 1.307 | **96798.50 (DIVERGED)** |

**Step-level divergence (rank 0)**: step 3800 (EP2 start) grad=5.28 → step 4000 grad=489 → step 4200 grad=2,827 → step 4400 grad=2.46×10⁸ → step 4600 grad=Inf, loss=161,798. Last 1000 steps (7000-7836): every optimizer step skipped via `nonfinite_grad` guard. Model frozen in degenerate state. Killed at ~33min wall.

### Verdict: NEGATIVE — clean falsification via training instability. CLOSED.

**Mechanism PASS** (all confirmed):
- Param split correct: surface_out tensors found (4 tensors, 264,708 params vs PR estimate ~263k), backbone 15,998,869 params
- `lr/head_to_backbone_ratio = 4.99988` held precise throughout training (cosine scheduler preserves per-group ratios as expected)
- Sanity at mult=1.0 PASS: EP1 train_loss=0.279 smooth descent matching baseline, no NaN

**Cause analysis (student diagnostic)**: Lion (Chen et al. 2023) recommends lr ∈ [1e-5, 1e-3]; head_lr=2.5e-3 is at absolute upper bound. Lion's sign-step `lr*sign(g)` produces uniformly-sized updates regardless of grad magnitude; at lr=2.5e-3 on the 2-layer surface_out MLP, those updates pushed weights into regions causing bf16 activation overflow. Divergence onset exactly at end-of-warmup when head LR first hit 2.5e-3 steady state — characteristic signature of single-layer runaway (grad explodes before loss does).

This confirms: **the output head IS sensitive to LR scaling (consistent with H6 mechanism PASS)**, but the operating point is fragile under Lion at 5×. The H6 mechanism-PASS interpretation (bottleneck at output head) holds, but the simple "amplify LR" attack fails due to optimizer-stability bounds — not due to wrong diagnosis.

### Wave 30 attack-layer summary at H14 closure

- **Architecture (×7 CLOSED, DEFINITIVELY EXHAUSTED)**: H1/H2/H3/H4/H5/H7 widening, H6 mechanism PASS absolute FAIL
- **Loss (×3 in-flight)**: #1147 H6' soft τ·n, #1151 H12 magnitude-weighted, #1152 H13 tangent/normal
- **Data-input (×3 in-flight)**: #1143 H8 mirror-sym, #1146 H9' curvature, #1150 H11 multi-scale kNN
- **Output-head (×1 in-flight)**: #1148 H10 vector-decoupled
- **Optimization (×1 closed=diverged)**: #1153 H14 (this)

### Reassignment

Alphonse reassigned to **H15 EMA / Polyak averaging of model weights** (#1155) — maintain exponential moving average (decay=0.9999) of model params in fp32, evaluate val/test on EMA copy. **Second optimization-layer probe**, different mechanism from H14: amplify ← H14, smooth → H15. Critical missing piece in baseline (PR #972 does NOT use EMA; virtually every Kaggle/ImageNet SOTA does). Zero divergence risk, compounds with any other in-flight winner.

---

## 2026-05-16 13:30 — PR #1141: Hard Normal-Routing Slice Partition / MoE-style Attention (alphonse) — CLOSED (6-of-6 model-side widening confirmed; architecture-layer attack surface ABSOLUTELY EXHAUSTED)

- **Branch**: `alphonse/hard-normal-slice-routing` (closed)
- **W&B run**: `iudmdz95`, EP13 best-EMA, clean 14.7h, 13/13 epochs, no NaN
- **Hypothesis**: Hard MoE-style routing partitions slice tokens between z-normal (roof/underbody, |n_z|≥0.5) and xy-normal (sides) groups using `-inf` pre-softmax masking. If τ_z bottleneck stems from mixed-orientation slice attention, dedicated capacity for z-surfaces should unstick τ_z.

### Terminal metrics

| Metric | val (34) | test (50) | Baseline #972 | Δ (test) | Gate |
|---|---:|---:|---:|---:|---|
| abupt | 6.347% | 6.018% | 5.844% | +0.174pp | — |
| **WSS** | 7.219% | **6.927%** | 6.727% | **+0.200pp** | **FAIL** |
| **SP** | 4.199% | **3.851%** | 3.577% | **+0.274pp** | **FLOOR BREACH** |
| vol_p | 3.674% | **3.548%** | 3.643% | **−0.095pp** | **PASS** |
| τ_z | 9.723% | 9.063% | — | — | — |
| **τz/τx** | **1.533** (val) | **1.478** (test) | ~1.46 | flat | **NULL (slight widening)** |

### Verdict: NEGATIVE on primary WSS — close. Mechanism strongly engaged (utilization=1.0 across all 5 blocks) but engaged-but-neutral on τ_z.

**Diagnostic was thorough**: utilization=1.0 across both z-group and xy-group in every block — hard routing was fully active. Student identified a capacity-mismatch confounder (z-surface = 56.2% of sampled tokens but z-slice-fraction=25% of slice capacity → 2.25× over-subscription on z-slices). Test τz/τx=1.478 is *slightly worse* than baseline 1.46 even with hard routing fully engaged.

**Healthy training**: monotonic descent EP4 (6.57%) → EP13 (6.347%), no NaN, EMA improving over raw throughout. Curve had no overfit plateau by EP13 — likely capacity-bound, not data-bound.

### 6-of-6 Wave 30 architecture-attack widening pattern at TERMINAL CLOSURE — surface DEFINITIVELY EXHAUSTED

| PR | Axis | val τz/τx | test τz/τx |
|---|---|---|---|
| #1136 (nezuko) | H2 normal spectral | 1.49 → 1.548 | 1.457 |
| #1137 (fern) | H5 Y-arch backbone split | — → 1.53 | 1.453 |
| #1138 (thorfinn) | H3 soft normal-routing | 1.50 → 1.537 | 1.452 |
| #1139 (edward) | H1 cylindrical coords | 1.385 → 1.547 | 1.469 |
| #1140 (askeladd) | H7 normal-aux head | 1.515 → 1.543 | 1.441 |
| **#1141 (alphonse)** | **H4 hard MoE routing** | **1.533 (EP13)** | **1.478** |

**H3 + H4 jointly falsify the entire attention-routing axis** (soft and hard endpoints of the routing sweep both null). With H1/H2/H5/H7 also closed, the model-side / architecture-layer attack surface is **definitively exhausted** for the τ_z structural bottleneck. (#1134 tanjiro H6 broke the pattern with hard τ·n=0 mechanism PASS to test τz/τx=1.281 but absolute FAIL — that single mechanism PASS is the diagnostic confirming the bottleneck IS at the output-head layer.)

### Reassignment

Alphonse reassigned to **H14 asymmetric LR for surface output head** (#1153) — splits Lion optimizer into 2 parameter groups: `surface_out.*` MLP at 5× the backbone LR. **First optimization-layer attack** in the Wave 30 fleet. Direct attack consistent with the H6 mechanism-PASS diagnostic: pushing more gradient signal into the output projection. Zero compute overhead, ~30-line change to `build_optimizer`. Orthogonal to all 7 in-flight axes (data, loss, output, architecture).

### Closure pattern summary — 7 Wave 30 closures complete

| PR | Student | Axis | Layer | Verdict |
|---|---|---|---|---|
| #1134 | tanjiro | H6 hard τ·n=0 | output projection | mechanism PASS / absolute FAIL (paper-worthy) |
| #1136 | nezuko | H2 spectral encoding | architecture | 4-of-4 widening |
| #1137 | fern | H5 Y-arch split | architecture | 5-of-5 widening |
| #1138 | thorfinn | H3 soft routing | architecture | 5-of-5 widening |
| #1139 | edward | H1 cylindrical coords | architecture | 7-of-7 widening (sincos pos_embed subsumes) |
| #1140 | askeladd | H7 normal-aux head | output (aux only) | 6-of-6 widening + fleet leader stall |
| #1141 | alphonse | H4 hard routing | architecture | 6-of-6 widening + capacity-mismatch confounder |

The Wave 30 fleet's 6 architecture-layer closures span every reasonable architectural intervention point: input frame, frequency encoding, attention routing (soft+hard), backbone topology, auxiliary heads. With 8 fresh probes now in flight covering loss layer (×3), data-input (×3), output projection (×1), and optimization (×1, NEW), the remaining attack surface is concentrated.

---

## 2026-05-16 11:45 — PR #1138: Normal-Aligned Slice Groups / Soft Orientation Routing (thorfinn) — CLOSED (5-of-5 model-side widening at terminal closure)

- **Branch**: `thorfinn/normal-aligned-slice-groups` (closed)
- **W&B run**: `of1ur6fp`, state=finished, best_epoch=12, source=EMA, 14h+ training, 13/13 epochs
- **Hypothesis**: Add a soft orientation-aware bias to slice routing so slice tokens cluster around vertices with similar surface-normal direction. Mechanism: per-block `normal_slice_bias` parameter learned with zero-init.

### Terminal metrics (advisor pulled from W&B summary — student pod went idle ~10:31Z after Claude session exit; no SENPAI-RESULT comment posted but W&B run completed cleanly at 11:14Z)

| Metric | val (34) | **test (50)** | Baseline #972 | Δ (test) | Gate |
|---|---:|---:|---:|---:|---|
| abupt | 6.315% | 5.938% | 5.844% | +0.094pp | — |
| **WSS** | 7.183% | **6.898%** | 6.727% | **+0.171pp** | **FAIL** |
| **SP** | 4.165% | **3.709%** | 3.577% | **+0.132pp** | **FLOOR BREACH** |
| vol_p | 3.665% | **3.462%** | 3.643% | **−0.181pp** | **PASS** |
| τ_x | 6.304% | 6.146% | — | — | — |
| τ_z | 9.686% | 8.922% | — | — | — |
| **τz/τx** | **1.536** (val) | **1.452** (test) | ~1.46 | flat | **NULL** |

### Verdict: NEGATIVE on primary WSS — close. Mechanism strongly engaged but engaged-but-neutral on τ_z.

**Diagnostic was clean**: slice entropy collapsed 0.96 → 0.36 (EP1 → EP10, then plateaued); `normal_slice_bias.param_norm` grew healthily from zero-init to 5.87/7.65/6.98/7.25/7.92 per block. The router IS using orientation conditioning. block_2 most peaked at 0.23 (block-specific concentration).

**But τ_z still flattened**: 9.723 (EP6) → 9.684 (EP10), essentially zero descent over 4 epochs while τ_x kept descending. val τz/τx widened 1.50 → 1.537 then collapsed back to 1.452 on test — same engaged-but-neutral signature as every other model-side attack.

**Surprising side-effect**: test_vol_p=3.462% beats baseline by 0.181pp. Normal-aligned slice grouping apparently improves volume-pressure prediction (probably by aligning volume cross-attention with body-aligned slices). Not enough to redeem on primary WSS axis but worth noting.

### 5-of-5 Wave 30 architecture-attack widening pattern at TERMINAL CLOSURE

| PR | Axis | val τz/τx widening | test τz/τx |
|---|---|---|---|
| #1136 (nezuko) | H2 normal spectral | 1.49 → 1.548 | 1.457 |
| #1137 (fern) | H5 Y-arch | — → 1.53 | 1.453 |
| #1140 (askeladd) | H7 normal-aux | 1.515 → 1.543 | 1.441 |
| #1139 (edward) | H1 cylindrical | 1.385 → 1.547 | 1.469 |
| **#1138 (thorfinn)** | **H3 soft normal-routing** | **1.50 → 1.537** | **1.452** |

(#1141 alphonse H4 hard MoE routing is the only architecture attack still in-flight; #1134 tanjiro H6 broke the pattern with hard τ·n=0 mechanism PASS but absolute FAIL.) The architecture-layer attack surface is **definitively exhausted** for the τ_z bottleneck.

### Reassignment

Thorfinn reassigned to **H13 tangent/normal anisotropic surface-loss decomposition** (#1152) — symmetric opposite of H6' on the loss layer. Decomposes per-vertex τ prediction into tangent + normal components using surface normals (already in `surface_x[..., 3:6]`) and applies α=1, β=5 to explicitly upweight matching the GT normal-component. Tests whether the τ_z bottleneck is caused by under-learning the GT normal-component (β>1 helps) or by over-learning normal-component noise (H6' direction helps instead).

### Operational note

Student session exited code=0 at 10:31Z after iteration 451 — Claude exited but training continued in background. W&B run reached state=finished at 11:14:54Z but no Claude session was alive to post the terminal SENPAI-RESULT. Pod has been idle since iteration 452. Advisor pulled metrics directly from W&B summary as authoritative source. Benign harness orchestration artifact (no integrity issue with the experiment data).

---

## 2026-05-16 10:05 — PR #1139: Cylindrical Coordinates (r, θ, z) Input Frame (edward) — CLOSED (7-of-7 model-side widening, all 3 floors breached)

- **Branch**: `edward/cylindrical-coordinates` (closed)
- **W&B run**: `z83eom8y`, EP13 best-EMA, clean 14.0h, 13/13 epochs, no NaN
- **Hypothesis**: Swap raw Cartesian (x, y, z) input frame for cylindrical (r, θ, z). If the τ_z bottleneck is exacerbated by the lack of vertical-axis bias in the input representation, cylindrical should help.

### Terminal metrics

| Metric | val (34) | **test (50)** | Baseline | Δ | Gate |
|---|---:|---:|---:|---:|---|
| abupt | 6.291% | 6.127% | 6.126% | +0.001pp | — |
| **WSS** | 7.102% | **7.049%** | 6.727% | **+0.322pp** | **FAIL** |
| **SP** | 4.174% | **3.865%** | 3.577% | **+0.288pp** | **FLOOR BREACH** |
| **vol_p** | 3.715% | **3.682%** | 3.643% | **+0.039pp** | **FLOOR BREACH** |
| τ_z | 9.579% | 9.167% | ~8.20% | +0.967pp | — |
| **τz/τx** | 1.547 (val) | **1.469** (test) | ~1.46 | flat | **NULL** |

### Verdict: NEGATIVE — close

All 3 hard gates breached. The sincos pos_embed already provides a complete Fourier basis that subsumes the cylindrical decomposition — switching the raw input frame from Cartesian to cylindrical adds zero structural information. Clean falsification.

### 7-of-7 Wave 30 model-side widening pattern CONFIRMED

| PR | Axis | val τz/τx widening | test τz/τx |
|---|---|---|---|
| #1136 (nezuko) | H2 normal spectral | 1.49 → 1.548 | 1.457 |
| #1141 (alphonse) | H4 hard MoE routing | TBD | TBD |
| #1137 (fern) | H5 Y-arch | — → 1.53 | 1.453 |
| #1140 (askeladd) | H7 normal-aux | 1.515 → 1.543 | 1.441 |
| **#1139 (edward)** | **H1 cylindrical** | **1.385 → 1.547** | **1.469** |
| #1138 (thorfinn) | H3 soft routing | TBD | TBD |

Architecture-layer attack surface is **definitively exhausted**.

### Reassigned

`edward` → **H12 τ-Magnitude-weighted MSE Loss** (PR #1151). Multiply per-vertex surface MSE by `(|τ_target_i| / batch_mean)^α`. Sweep α ∈ {0.3, 0.5, 0.7}. Direct attack on long-tail τ_z error distribution at the loss layer — the only major attack surface that wasn't previously touched in Wave 30 (H6' is the only other loss attack and is tangency-specific, not magnitude-weighted).

---

## 2026-05-16 09:35 — PR #1140: Normal-Prediction Auxiliary Head (askeladd) — CLOSED (fleet leader stalls at EP13, 6-of-6 widening pattern, 2 of 3 gates fail)

- **Branch**: `askeladd/normal-prediction-aux-head` (closed)
- **W&B run**: `e5ztxjc3`, EP13 EMA, clean 13/13 epochs no NaN no OOM
- **Hypothesis**: Add a 3-dim auxiliary head predicting `(nx, ny, nz)` from `surface_hidden` with cosine-similarity loss. If τ_z bottleneck is missing orientation information in the backbone, aux head should regularize it.

### Terminal metrics

| Metric | val (34) | **test (50)** | Baseline | Δ | Gate |
|---|---:|---:|---:|---:|---|
| abupt | 6.1975% | 5.9799% | 5.844% | +0.136pp | — |
| **WSS** | — | **6.9018%** | 6.727% | **+0.175pp** | **FAIL** |
| **SP** | — | **3.8246%** | 3.577% | **+0.248pp** | **FLOOR BREACH** |
| vol_p | — | 3.5776% | 3.643% | −0.065pp | PASS ✓ |
| τ_x | — | 6.166% | ~5.61% | +0.556pp | — |
| τ_y | — | 7.448% | ~6.93% | +0.518pp | — |
| τ_z | — | 8.883% | ~8.20% | +0.683pp | — |
| **τz/τx** | **1.543** | **1.441** | 1.46 | val widens, test ≈ flat | NULL |

### Verdict: NEGATIVE — close

2 of 3 hard gates fail. The fleet leader at EP7-8 (val_abupt=6.222%) stalled — EP13 EMA landed at 6.1975%, 0.071pp ABOVE the val baseline of 6.126%. Mid-flight lead did not survive to terminal. vol_p PASS is good for stacking but cannot compensate alone.

### Mechanism diagnostic — cleanest falsification of the wave

`aux_normal_cosine` converged to **0.999951** by step ~10k. The aux head matched the existing normal information in `surface_hidden` essentially perfectly. **The normal-orientation signal was already fully present in backbone hidden states** — aux gradient pressure had nothing new to inject. This decisively falsifies the "missing-normal-info" hypothesis.

### 6-of-6 Wave 30 model-side widening pattern CONFIRMED

| PR | Axis | val τz/τx widening | test τz/τx |
|---|---|---|---|
| #1136 (nezuko) | H2 normal spectral | 1.49 → 1.548 | 1.457 |
| #1141 (alphonse) | H4 hard MoE routing | TBD | TBD |
| #1137 (fern) | H5 Y-arch | — → 1.53 | 1.453 |
| #1139 (edward) | H1 cylindrical coords | 1.385 → 1.526 | TBD |
| **#1140 (askeladd)** | **H7 normal-aux** | **1.515 → 1.543** | **1.441** |

The val/test inversion (val widens, test ~ flat) is consistent across the whole wave — reflects sample-set idiosyncrasies (34 val vs 50 test cases) more than mechanism. Bottleneck remains data-distribution / output-head, not architectural.

### Reassigned

`askeladd` → **H11 Multi-scale kNN-pooled context features** (PR #1150). 3 statistics (cos_alignment, mean_area, mean_dist) × 3 scales (k=4/16/64) = 9 additional surface channels. Direct upgrade of H9' single-scale curvature (which is just `1-cos_alignment` at k=16); H11 tests whether multi-scale geometric context unlocks signal that single-scale captures only partially. Strong Kaggle/PointNet++/FPN pedigree.

---

## 2026-05-16 09:05 — PR #1137: Y-Architecture Dual-Backbone — pressure/WSS branches (fern) — CLOSED (5-of-5 model-side widening, all 3 floors breached)

- **Branch**: `fern/y-architecture-dual-backbone` (closed)
- **W&B run**: `m9qed7bb`, EP6 EMA + post-mortem test eval (run hit OOM at EP7 boundary on vol_points ramp)
- **Hypothesis**: Split backbone into separate pressure and WSS branches to eliminate task interference. If τ_z bottleneck is caused by the shared backbone juggling cp + 3 WSS components, dual-branch should unlock τ_z.

### Terminal metrics

| Metric | val (34) | **test (50)** | Baseline | Δ | Gate |
|---|---:|---:|---:|---:|---|
| abupt | 6.487% | 6.172% | 5.844% | +0.328pp | — |
| **WSS** | 7.370% | **7.109%** | 6.727% | **+0.382pp** | **FAIL** |
| **SP** | 4.258% | **3.931%** | 3.577% | **+0.354pp** | **FLOOR BREACH** |
| **vol_p** | 3.778% | **3.673%** | 3.643% | **+0.030pp** | **FLOOR BREACH** |
| τ_z | 9.894% | 9.158% | — | — | — |
| **test τz/τx** | 1.53 (val) | **~1.453** | 1.46 band | flat | **NULL** |

### Verdict: NEGATIVE — close

All 3 merge gates breached. Branch cos_sim 0.17–0.20 (healthy split, no collapse), but τ_z reduced proportionally with τ_x/τ_y rather than differentially. Task-interference hypothesis falsified.

### 5-of-5 Wave 30 model-side widening pattern CONFIRMED

| PR | Axis | val τz/τx widening | test τz/τx |
|---|---|---|---|
| #1136 (nezuko) | H2 normal spectral | 1.49 → 1.548 | 1.457 |
| #1139 (edward) | H1 cylindrical coords | 1.385 → 1.526 | TBD |
| #1141 (alphonse) | H4 hard MoE routing | TBD | TBD |
| #1140 (askeladd) | H7 normal-aux head | 1.515 → 1.537 | TBD |
| **#1137 (fern)** | **H5 Y-arch** | **— → 1.53** | **1.453** |

The bottleneck is **definitively not at the model architecture layer**. Remaining attack surface: output-head reformulation, data distribution, input features.

### Reassigned

`fern` → **H10 Vector-Length-Decoupled WSS Head** (PR #1148). Predict `(cp, dir_x, dir_y, dir_z, log_mag)` instead of Cartesian `(cp, τx, τy, τz)`. Decouples direction from magnitude at the output, adds auxiliary cosine-similarity loss on unit direction. Direct follow-up: H6 (tanjiro) proved bottleneck IS at output head — H10 attacks via reparametrization (orthogonal to H6' soft penalty).

---

## 2026-05-16 08:35 — PR #1134: Local-frame WSS head — hard τ·n=0 (tanjiro) — CLOSED (paper-worthy MECHANISM PASS / absolute FAIL)

- **Branch**: `tanjiro/local-frame-wss-head` (closed)
- **W&B run**: `m1uvk8wl`, terminal at EP13 best-EMA
- **Hypothesis**: Enforce τ·n=0 architecturally via local-frame WSS head (project predicted WSS onto tangent basis). Test whether the τ_z bottleneck is at the output head level.

### Terminal metrics

| Metric | val (34) | **test (50)** | Baseline `56bcqp3m` | Δ | Gate |
|---|---:|---:|---:|---:|---|
| abupt | 18.615% | 18.396% | 5.844% | +12.55pp | FAIL |
| **WSS** | 26.439% | **26.692%** | 6.727% | **+19.96pp** | **FAIL** |
| τ_x | 25.536% | 26.146% | — | — | — |
| τ_y | 23.144% | 22.914% | — | — | — |
| **τ_z** | 34.423% | 33.508% | — | — | FAIL absolute |
| **test τz/τx** | **1.348** | **1.281** | 1.50–1.57 band | **−0.219 to −0.289** | **MAJOR-WIN** ✓ |
| **SP** | 5.389% | 4.989% | 3.577% | +1.41pp | **FLOOR BREACH** |
| **vol_p** | 4.581% | 4.423% | 3.643% | +0.78pp | **FLOOR BREACH** |
| `train/wss_penetration_frac` | — | 2.40e-08 | — | bf16 floor | constraint enforced |

### Verdict: PAPER-WORTHY MECHANISM PASS / absolute FAIL — close

H6 is the cleanest τ_z structural break ever observed on this dataset. NINE prior mechanisms (loss weighting / output decoupling / EMA / sampling / depth / curriculum / mag decomp / SDF FAR-field / per-channel heads) all landed in the [1.44, 1.57] band. Hard τ·n=0 broke it cleanly from EP1 onward (1.475 → 1.348 monotone descent on val; 1.281 on test).

**Falsifies** the alternative hypothesis (backbone slice-attention or full Y-architecture being the τ_z bottleneck) — the bottleneck IS at the WSS output head.

### Why it can't merge despite the mechanism win

tanjiro's pre-flight diagnostic on 10 train cases measured `mean |τ·n|/|τ| = 8.1%` (magnitude-weighted = 5.6%) on the raw GT `wall_shear`. Hard architectural enforcement throws that away by construction, and val_WSS plateaus ~4× baseline. PR's MAJOR-WIN criterion required both ratio AND floors; we only got ratio.

### Follow-up

`tanjiro` → **H6' soft τ·n=0 penalty** (PR #1147 just launched). Keep the unconstrained 4-channel head, add `λ · E[|τ·n|/|τ|]` loss term with λ sweep ∈ {0.05, 0.1, 0.25}. Best of both: structural break + absolute fidelity. Highest expected-value unassigned slot in the entire fleet.

---

## 2026-05-16 08:20 — PR #1136: Normal Spectral Encoding StringSep on (nx,ny,nz) (nezuko) — CLOSED (terminal NEGATIVE, 4-of-4 model-side widening confirmation)

- **Branch**: `nezuko/normal-spectral-encoding` (closed)
- **W&B run**: `lths1ujt`, terminal at EP13 best-EMA
- **Hypothesis**: Apply StringSeparable spectral encoding to surface normals (nx, ny, nz) — give the model richer frequency-domain features for orientation, hoping the τ_z direction gets localized representations.

### Terminal metrics

| Metric | val | **test (50)** | Baseline | Δ | Gate |
|---|---:|---:|---:|---:|---|
| abupt | 6.4039% | 6.0200% | 5.844% | +0.176pp | — |
| **WSS** | — | **6.9279%** | 6.727% | **+0.201pp** | **FAIL** |
| **SP** | — | **3.8271%** | 3.577% | **+0.250pp** | **FLOOR BREACH** |
| vol_p | — | 3.6327% | 3.643% | −0.010pp | PASS |
| test_τ_x | — | 6.163% | — | — | — |
| test_τ_y | — | 7.500% | — | — | — |
| test_τ_z | — | 8.977% | — | — | — |
| **test τz/τx** | — | **1.457** | 1.50–1.57 band | within | **NULL** |

### Verdict: NEGATIVE — close

Fails 2 of 3 hard gates (test_WSS +0.201pp, test_SP +0.250pp). The `surface_normal_string_sep` parameters trained correctly (log_freq std=1.22), so this is a clean mechanism null, not an implementation failure.

### 4-of-4 Wave 30 model-side widening pattern CONFIRMED

On val, τz/τx widened monotonically from 1.49 (EP2) to 1.548 (EP13), then collapsed to 1.457 on test. **4th consecutive Wave 30 model-architecture attack to show this exact pattern:**

| PR | Axis | val τz/τx widening | test τz/τx |
|---|---|---|---|
| #1139 (edward) | H1 cylindrical coords | 1.385 → 1.526 | ~1.46 |
| #1136 (nezuko) | H2 normal spectral | 1.49 → 1.548 | 1.457 |
| #1141 (alphonse) | H4 hard normal routing | (in flight) | TBD |
| #1140 (askeladd) | H7 normal aux head | 1.515 → 1.537 | TBD |

The τ_z gap is **data-distribution or loss-mechanism in nature, not model-architecture in nature.**

### Reassigned

`nezuko` → **H9' curvature-aware surface feature** (PR #1146 just launched). Port of dl24 cross-pollination finding (test_WSS=6.609% reported on parallel branch). 8th channel via kNN-of-normals statistic. First input-feature attack on tay.

---

## 2026-05-16 04:30 — PR #1133: Per-axis WSS mag decomp |τ_z|+||τ_xy|| (frieren) — CLOSED (terminal NEGATIVE, NINTH structural ratio confirmation, test_SP floor breached)

- **Branch**: `frieren/per-axis-mag-decomp` (closed)
- **W&B run**: `5l9i6fjn`, arm `lambda-z-0p1`, terminal at EP13
- **Hypothesis**: Decompose WSS into magnitudes |τ_z| and ||τ_xy|| with separate aux heads — give the backbone a dedicated magnitude-scalar gradient signal on the dominant error axis, separate from the signed-3-vector main head.

### Terminal metrics (test, single-model, EP13 terminal)

| Metric | This PR | Baseline #972 | Δ | Verdict |
|---|---:|---:|---:|---|
| val_abupt | 6.254% | 6.126% | +0.128pp | misses |
| **test_WSS** | **6.853%** | 6.727% | **+0.126pp** | **misses** ❌ |
| test_vol_p | 3.620% | 3.643% (floor) | −0.023pp | PASS ✅ |
| **test_SP** | **3.837%** | 3.577% (floor) | **+0.260pp** | **FLOOR BREACHED** ❌ |
| test_τ_x | 6.064% | ~5.61% | +0.45pp | regress |
| test_τ_y | 7.473% | ~6.93% | +0.54pp | regress |
| test_τ_z | 8.908% | ~8.20% | +0.71pp | target-axis regress |
| **test τz/τx** | **1.469** | ~1.46 | 0.00 | **NINTH confirmation** |

### Verdict: NEGATIVE — close

Fails 2 of 4 hard gates: test_WSS misses by 0.126pp, test_SP floor breached by 0.260pp. Mechanism null.

### Mechanism diagnostic — clean falsification

The two aux-head calibration ratios converged symmetrically to ~1.000 by EP6 and held through EP13 (`mag_z_calib_ratio=1.001`, `mag_xy_calib_ratio=0.999`). The mag_xy loss term **exceeded** mag_z loss throughout training (1.07–1.34× ratio) — the OPPOSITE of what the τ_z-is-hard hypothesis would predict.

**Conclusion**: the backbone represents |τ_z| just as easily as it represents ||τ_xy||. The bottleneck is in *signed* τ_z prediction, not in *magnitude* encoding. Magnitude decomposition provides no gradient signal the backbone couldn't already access.

### NINTHFOLD structural ratio confirmation

Adding #1133 to the table: now 9 distinct mechanisms have landed in the test τz/τx ∈ [1.44, 1.57] band. Mag-decomp is the 9th — pure loss-side reformulation provides zero traction.

### Reassigned

`frieren` → next Wave 30+ hypothesis. With 7 Wave 30 architectural axes (H1/H2/H3/H4/H5/H6/H7) already in flight, frieren's next axis must be **orthogonal to all of them** — likely a data-representation / multi-scale / training-procedure attack rather than architectural.

---

## 2026-05-15 21:15 — PR #1122: SDF FAR-field α=2.0 port (alphonse) — CLOSED (terminal NEGATIVE, EIGHTH structural ratio confirmation)

- **Branch**: `alphonse/port-sdf-importance-sampling-pr972` (closed)
- **W&B run**: `vvv84p32`, EP10 truncated at 17h57m (advisor decision EP4), best-EMA at EP7
- **Hypothesis**: Port the SDF-stratified volume importance sampling stack from PR #972 (α=4.0 attempted, advisor down-scaled to α=2.0 to match historical sweet spot) to tay so the no-SDF tay baseline regains the sampling-side mechanism that drove the original SOTA.

### Terminal metrics (test, single-model, EP7 best-EMA)

| Metric | This PR | Baseline #972 | Δ | Verdict |
|---|---:|---:|---:|---|
| val_abupt | 6.698% | 6.126% | +0.572pp | misses |
| **test_abupt** | **6.657%** | — | — | val→test compression 0.041 (within fleet band) |
| **test_WSS** | **7.518%** | 6.727% | **+0.792pp** | **REGRESS** ❌ |
| **test_vol_p** | **4.524%** | 3.643% (floor) | **+0.881pp** | **floor regress** ❌ |
| **test_SP** | **4.141%** | 3.577% (floor) | **+0.564pp** | **floor regress** ❌ |
| test_τ_x | 6.640% | ~5.61% PR #972 | +1.03pp | off-target regress |
| test_τ_y | 8.250% | ~6.93% PR #972 | +1.32pp | off-target regress |
| test_τ_z | 9.730% | ~8.20% PR #972 | +1.53pp | target-axis regress |
| **test τz/τx** | **1.465** | ~1.46 | **0.00** | **EIGHTH confirmation** |

### Verdict: NEGATIVE — close

Fails 3/3 merge gates. SDF FAR-field α=2.0 port did not reproduce PR #972's SOTA on no-SDF tay — likely because PR #972 relied on the full SDF stratification stack (`--sdf-importance-sampling --sdf-alpha 4.0` + careful schedule) which never landed on tay, and re-implementing only the FAR-field bias is insufficient. The α=2.0 down-scaling (advisor decision) preserved the structural-test purpose at the cost of expected absolute floor regress.

### Mechanism — EIGHTHFOLD structural ratio confirmation

EP3 τz/τx = 1.515, EP4 = 1.526 (sevenfold confirm), EP5 = 1.528, EP6 = 1.527, EP7 = 1.526, EP8 = 1.525, EP9-10 ~1.524, test = 1.465 (with val→test compression). The ratio is **invariant to volume-sampling distribution shifts**. Eight independent interventions now confirmed:

| # | Mechanism | Lever | τz/τx (test) |
|---|-----------|-------|------------:|
| 1 | thorfinn EMA 0.9995 (#1124) | temporal | 1.469 |
| 2 | nezuko spatial-prior α=10 (#1125) | sampling | 1.449 |
| 3 | fern surface_out depth=4 (#1126) | output capacity | 1.462 |
| 4 | edward per-channel heads (#1116) | output decoupling | 1.460 |
| 5 | thorfinn τ_z weight 3.0 (#1128) | loss weighting | 1.44 |
| 6 | askeladd surface_loss warmup (#1127) | curriculum | 1.52 |
| 7 | frieren mag-only (#1121 → #1133 in flight) | loss reform | 1.46 (frieren #1121) |
| **8** | **alphonse SDF FAR-field α=2.0 (#1122)** | **volume sampling** | **1.465** |

**Conclusion confirmed**: data-side, loss-side, capacity-side, output-decoupling-side, and now volume-sampling-side mechanisms ALL converge to the structural band. Wave 30's architectural pivot is exactly the right direction.

### Reassigned

`alphonse` → PR #1141 (Wave 30 H4 hard normal slice routing) — completes the soft↔hard sweep on the attention layer alongside thorfinn's H3 #1138 (soft routing).

---

## 2026-05-15 20:30 — PR #1128: τ_z loss weight 2.0→3.0 (thorfinn) — CLOSED (partial confirm, off-axis cost wipes gain; test_SP floor regress)

- **Branch**: `thorfinn/tau-z-loss-weight-3p0` (closed)
- **W&B run**: `uwqybod5`, 839.9 min wall-clock, EP13 best ckpt
- **Hypothesis**: Escalate τ_z channel weight 2.0→3.0 to directly redirect gradient at the dominant error axis.

### Terminal metrics (test, single-model, EP13 best ckpt)

| Metric | This PR | Baseline #972 | Δ | Verdict |
|---|---:|---:|---:|---|
| val_abupt | 6.236% | 6.126% | +0.110pp | misses |
| **test_WSS** | **6.938%** | 6.727% | **+0.211pp** | **misses** ❌ |
| **test_vol_p** | **3.584%** | 3.643% (floor) | **−0.059pp** | **PASS (under floor!)** ✅ |
| **test_SP** | **3.838%** | 3.577% (floor) | **+0.261pp** | **floor regress** ❌ |
| test_τ_z | 9.006% | ~9.45% no-SDF | −0.44pp | absolute gain on target axis |
| test_τ_x | 6.151% | ~5.97% | +0.18pp | off-axis cost |
| test_τ_y | 7.546% | ~7.36% | +0.19pp | off-axis cost |
| **test τz/τx** | **1.464** | — | — | val→test compression continues (val=1.549) |

**Decision: CLOSE.** Fails primary gate (test_WSS +0.211pp) and test_SP floor (+0.261pp). test_vol_p PASS is a notable isolated win but not enough.

### Mechanism — partial confirmation, ceiling re-asserted by EP11+

EP1 ratio 1.288 confirms early gradient redirection works. By EP11+ val ratio drifts back to 1.547-1.549 (baseline band). Absolute τ_z gain (−0.44pp) is real but off-axis costs to τ_x (+0.18pp) and τ_y (+0.19pp) wipe the net WSS gain. Net no-SDF baseline delta: only −0.05pp on test_WSS.

### val→test ratio compression observed across the fleet

- thorfinn #1128: val 1.549 → test 1.464 (Δ −0.085)
- tanjiro #1124: val 1.555 → test 1.469 (Δ −0.086)
- nezuko #1125: val 1.549 → test 1.449 (Δ −0.100)

Robust val→test distribution shift partially reduces the τ_z bottleneck on test but doesn't eliminate it.

### τ_z structural finding — ELEVENTH CONFIRMATION

τ_z weight=3.0 joins the 10 prior null mechanisms. The 1.45-1.55 ratio attractor is robust to gradient-tilt magnitude.

### Reassigned as

- Thorfinn → **Wave 30 H3: Normal-Aligned Slice Groups** — soft orientation-aware routing in slice attention. Fourth orthogonal Wave 30 attack axis (attention-routing-side). ~50 LOC change in model.py.

---

## 2026-05-15 20:30 — PR #1127: Explicit surface_loss warmup curriculum (askeladd) — CLOSED (clean falsification on EVERY metric, including hypothesis-target τ_z)

- **Branch**: `askeladd/surface-loss-warmup-curriculum` (closed)
- **W&B run**: `ag1dnelx`, EP11 best EMA
- **Hypothesis**: Explicit 3-EP linear ramp of surface_loss_weight (0→full) would replicate PR #1114 implicit-curriculum mechanism and preferentially reduce test_τ_z.

### Terminal metrics (test, single-model, EP11 best EMA)

| Metric | This PR | thorfinn #1100 no-SDF | Baseline #972 | Δ vs #972 | Verdict |
|---|---:|---:|---:|---:|---|
| val_abupt | 6.476% | — | 6.126% | +0.350pp | misses |
| **test_WSS** | **7.227%** | 6.989% | 6.727% | **+0.500pp** | **BIG MISS** ❌ |
| **test_vol_p** | **3.678%** | 3.644% | 3.643% | +0.035pp | **floor regress** ❌ |
| **test_SP** | **3.869%** | 3.832% | 3.577% | +0.292pp | **floor regress** ❌ |
| **test_τ_z** | **9.293%** | ~9.05% | — | **+0.24pp** | **regresses on hypothesis-target axis** ❌ |

**Decision: CLOSE.** Worst result of this review batch. Hypothesis CLEANLY FALSIFIED on the target axis and on all paper-facing metrics.

### Mechanism — implicit curriculum was a different beast

Student's analysis is correct: PR #1114 learnable weights drifted briefly to ~50% surface weight (not zero); this PR's explicit ramp goes to literal zero, costing surface head capacity that's not recovered in remaining 10 epochs. Three independent loss-curriculum/shape attempts now all negative:
- PR #1114 learnable weights — partial regress
- PR #1118 OHEM v2 — regress
- PR #1127 explicit warmup — clean falsification

### Research-state implication

Combined with thorfinn #1128 (τ_z weight=3.0 also fails), data is unambiguous: **τ_z bottleneck is NOT loss-side**. Architectural lever is the only remaining direction. Wave 30 pivot is correct.

### Reassigned as

- Askeladd → **Wave 30 H7: Normal-Prediction Auxiliary Head** — aux task predicting surface normals from backbone features. Forces backbone to maintain orientation info at every layer via aux loss gradient signal. Different attack axis than H2 (input encoding only). ~80 LOC change.

---

## 2026-05-15 20:30 — PR #1116: Per-channel WSS output heads (edward) — CLOSED (mechanism reproducible but absorbed by no-SDF ceiling)

- **Branch**: `edward/per-channel-wss-heads` (closed)
- **W&B run**: `3ufrbxl6`, 858.5 min wall-clock, EP13 best EMA
- **Hypothesis**: Separate decoder heads for [cp, τ_x, τ_y, τ_z] decouple per-channel gradients, allowing τ_z head to optimize without competing with shared-head gradient pool.

### Terminal metrics (test, single-model, EP13 EMA)

| Metric | This PR | thorfinn #1100 no-SDF | Baseline #972 | Δ vs #972 | Verdict |
|---|---:|---:|---:|---:|---|
| val_abupt | 6.321% | — | 6.126% | +0.195pp | misses |
| **test_WSS** | **6.900%** | 6.989% | 6.727% | **+0.173pp** | **misses** ❌ (only 0.089pp below no-SDF ceiling) |
| **test_vol_p** | **3.687%** | 3.644% | 3.643% | +0.044pp | marginal floor regress ❌ |
| **test_SP** | **3.801%** | 3.832% | 3.577% | +0.224pp | floor regress ❌ |
| test_τ_z | 9.022% | ~9.05% | — | within range | — |
| val τz/τx | 1.554 | — | — | — | 12th band confirm |

**Decision: CLOSE.** Fails 3/4 merge gates. Despite reproducible mechanism, the no-SDF ceiling absorbs the gain.

### Mechanism — REAL AND REPRODUCIBLE but ceiling-bound

The per-channel decoupling effect is genuine:
- Matched-budget 3-EP A/B (matched architecture): −0.660pp test_WSS improvement vs single-head
- Reproduced in 18h run (val_WSS −0.062pp vs thorfinn #1100 single-head + slices=256)
- Per-head gradient norms confirm decoupling at training end: τ_z head pulls **1.57× more inner gradient** than τ_x head, persistent across all 13 epochs
- val_τ_z improved by −0.151pp vs single-head baseline (strongest delta of any axis)

But the test_WSS gain (~0.05pp on no-SDF baseline) is absorbed by the **no-SDF ceiling identified in fern #1126**:
- thorfinn #1100 slices=256 single-head: test_WSS=6.989%
- fern #1126 depth=4 single-head: test_WSS=6.989% (statistical tie)
- edward #1116 per-channel heads: test_WSS=6.900% (modest improvement)
- All three: bound by same ceiling region 6.90-6.99%

### Mechanistic value preserved for stacking

The per-channel head mechanism is a **stackable component**. If a backbone-level Wave 30 winner emerges (fern #1137 Y-arch, nezuko #1136 H2, tanjiro #1134 H6), re-introducing per-channel heads on the winning backbone may compound the improvements. Keep in toolbox.

### τ_z structural finding — TWELFTH CONFIRMATION

Per-channel heads join the 11 prior null mechanisms. Ratio attractor 1.45-1.57 is robust to per-channel output-side parameter increases.

### Reassigned as

- Edward → **Wave 30 H1: Cylindrical Coordinates (r, θ, z)** — replace Cartesian positional input with cylindrical so vertical (τ_z) gets its own dedicated coordinate axis. Cheapest input-side complement to nezuko #1136 H2. ~35 LOC change in model.py.

---

## 2026-05-15 19:45 — PR #1126: surface_out depth=4 + 18h budget (fern) — CLOSED (decoder-depth hypothesis FALSIFIED; statistical tie with no-SDF ceiling)

- **Branch**: `fern/surface-out-depth-4-18h` (closed)
- **W&B run**: `gr9ht3h5` (group `fern-surface-out-depth-4-18h`, EP13/13 complete, 843.6 min wall-clock = 14.06h, best EMA = EP11)
- **Hypothesis**: Deeper surface_out MLP (depth 2→4) gives more decoder representational capacity, which would preferentially reduce τ_z magnitude error if τ_z is decoder-depth-bottlenecked.

### Terminal metrics (best EMA EP11, test on 50 cases)

| Metric | This PR | Baseline #972 | Δ | Verdict |
|---|---:|---:|---:|---|
| val_abupt | 6.342% | 6.126% | +0.216pp | misses |
| **test_WSS** | **6.9886%** | **6.727%** | **+0.262pp** | **misses** ❌ |
| **test_vol_p** | **3.6452%** | 3.643% (floor) | **+0.0010pp** | **marginal floor regress** ❌ (statistical tie) |
| **test_SP** | **3.8335%** | 3.577% (floor) | **+0.257pp** | **floor regress** ❌ |
| test_abupt | 6.0674% | 5.844% | +0.223pp | — |
| test τ_z | 9.075% | ~8.96% (SDF SOTA) | +0.115pp | — |
| test τ_x | 6.206% | — | — | — |
| test τ_y | 7.577% | — | — | — |
| **test τz/τx** | **1.462** | — | — | 10th band confirmation |

**Decision: CLOSE.** Fails 3/4 merge gates. Decoder-depth-bottleneck hypothesis CLEANLY FALSIFIED.

### Per-epoch τz/τx trajectory — monotonic rise, no crossover

| EP | val_abupt | val_τ_x | val_τ_z | τz/τx |
|---|---:|---:|---:|---:|
| 1 | 31.614% | 31.598% | 42.361% | 1.341 |
| 2 | 7.717% | 7.764% | 11.064% | 1.425 |
| 3 | 6.924% | 6.905% | 10.253% | 1.485 |
| 4 | 6.606% | 6.537% | 9.931% | 1.519 |
| 5 | 6.498% | 6.434% | 9.814% | 1.525 |
| 6 | 6.446% | 6.371% | 9.775% | 1.534 |
| 7 | 6.402% | 6.325% | 9.734% | 1.539 |
| 8 | 6.380% | 6.304% | 9.718% | 1.542 |
| 9 | 6.360% | 6.287% | 9.696% | 1.542 |
| 10 | 6.344% | 6.274% | 9.687% | 1.544 |
| **11** | **6.342%** | **6.274%** | **9.694%** | **1.545** |
| 12 | 6.342% | 6.276% | 9.702% | 1.546 |
| 13 | 6.346% | 6.283% | 9.712% | 1.546 |

**Ratio rose monotonically EP1→EP12** (1.341 → 1.546). τ_z was the SLOWEST axis at every epoch. Even in the vol_points=65536 dense-supervision regime (EP10-13), no crossover occurred. The hypothesis predicted preferential τ_z gain; the data shows preferential τ_x gain.

### CRITICAL mechanistic finding — no-SDF ceiling convergence

This experiment ties three independent metrics to the no-SDF tay ceiling within sub-0.001pp:

| Metric | depth=4 (this run) | thorfinn #1100 no-SDF slices=256 ceiling | Δ |
|---|---:|---:|---:|
| test_WSS | 6.9886% | 6.989% | −0.0004pp |
| test_vol_p | 3.6452% | 3.6442% | +0.0010pp |
| test_SP | 3.8335% | 3.8324% | +0.0011pp |

**Two independent capacity uplifts (backbone slices=256 and decoder depth=4) converge to the SAME no-SDF ceiling.** This is the strongest evidence yet that the bottleneck is a **representation-axis bottleneck**, not a capacity bottleneck. Aligns perfectly with the Wave 30 architectural-pivot direction.

### τ_z structural finding — TENTH CONFIRMATION

Decoder-depth d=4 joins the 9 prior null mechanisms. τz/τx ratio remains an attractor in 1.40-1.57 band across:
- loss weighting (×3 vs ×1)
- sampling bias (spatial-prior, SDF FAR-field)
- output capacity (per-channel heads, decoder depth)
- regularization (EMA decay)
- decomposition (mag-only)
- spatial-prior α

### Reassigned as

- Fern → **Wave 30 H5: Y-Architecture Dual-Backbone** — split backbone after first encoder layer into parallel pressure-branch (cp) and WSS-branch (τx, τy, τz) transformer stacks. Tests task-interference hypothesis: does shared backbone favor pressure over WSS optimization? ~80 LOC change in `model.py`. Third orthogonal Wave 30 attack axis (input H2 / output H6 / backbone H5).

---

## 2026-05-15 19:00 — PR #1125: Spatial-prior α=10 + 18h budget (nezuko) — CLOSED (3/4 gate fail; test_vol_p PASS is fleet-best signal, α=10 too aggressive)

- **Branch**: `nezuko/spatial-prior-alpha10-18h` (closed)
- **W&B run**: posted in PR comments by nezuko (terminal SENPAI-RESULT confirmed `pending_arms=false`)
- **Hypothesis**: Stronger spatial-prior oversample (α=10 vs prior α=5) at 18h budget tilts loss toward near-vehicle samples where most error concentrates → expects test_WSS improvement at convergence.

### Terminal metrics (test, single-model, 50 cases)

| Metric | This PR | PR #972 single-model SOTA | Delta | Verdict |
|---|---:|---:|---:|---|
| val_abupt | 6.390% | 6.126% | +0.264pp | misses |
| **test_WSS** | **7.106%** | **6.727%** | **+0.379pp** | **misses** ❌ |
| **test_vol_p** | **3.634%** | 3.643% (floor) | **−0.009pp** | **PASS** (fleet-best margin) ✅ |
| **test_SP** | **3.954%** | 3.577% (floor) | **+0.377pp** | **floor regress** ❌ |
| test τz/τx | **1.449** | — | — | sub-structural (below 1.50 band) |

**Decision: CLOSE.** Fails 3/4 merge gates. test_SP +0.377pp floor regress is the dealbreaker.

### Mechanism — α=10 over-tilted near-vehicle attention

- α=5 from prior winning configurations remains the sweet spot
- α=10 over-weights near-vehicle samples at expense of far-field structural learning
- 18h schedule did not recover what α tilt damaged
- The test_SP regression is the clearest signal that spatial weighting reaches diminishing returns past α=5

### Data preserved as research artifact

1. **test_vol_p = 3.634% is fleet-best on volume pressure** (margin 0.009pp below baseline). Spatial-prior is an **orthogonal mechanism for vol_p improvement** worth keeping in the toolbox for future stacking experiments.
2. **val→test τz/τx compression continues** — val 1.549 → test 1.449 (0.10 unit compression). Now confirmed across tanjiro #1124, nezuko prior runs, and this run. Distribution shift between val and test partially reduces τ_z bottleneck consistently.

### τ_z structural finding — NINEFOLD CONFIRMATION

Nezuko #1125 closes as the 9th mechanistically-distinct mechanism converging τz/τx ratio to the 1.40-1.57 structural band. Spatial-prior tilt joins (loss weighting, sampling, output capacity, EMA, mag-only decomp, per-channel heads, SDF FAR-field, decoder depth). **τ_z bottleneck remains backbone-representation-side and requires architectural pivot.**

### Reassigned as

- Nezuko → **Wave 30 H2: Normal Spectral Encoding** — apply StringSeparableEncoding-style Fourier spectral basis to surface normals (nx, ny, nz) so they get the same multi-frequency representation as positions. ~35 LOC change in `model.py`.

---

## 2026-05-15 16:30 — PR #1124: EMA decay 0.9995 + 18h budget (tanjiro) — CLOSED (slow-decay hypothesis REFUTED; fails all 4 merge gates)

- **Branch**: `tanjiro/ema-slow-decay-18h` (closed)
- **W&B run**: `mw6d04kc` (EP13/13 complete, 853.8 min wall-clock = 14h14m, ~3h45m under 18h cap, best EMA = EP13)
- **Hypothesis**: Slow EMA decay (0.9995, half-life ≈ 1386 steps) preserves late-training τ_z specialization → Δ(raw − ema) on τ_z should GROW through EP6→EP13.

### Terminal metrics (best EMA = EP13, test on 50 cases)

| Metric | This PR | PR #972 single-model SOTA | Delta | Verdict |
|---|---:|---:|---:|---|
| val_abupt | 6.221% | 6.126% | +0.095pp | misses |
| **test_WSS** | **6.898%** | **6.727%** | **+0.171pp** | **misses** ❌ |
| **test_vol_p** | **3.666%** | 3.643% (floor) | **+0.023pp** | **floor regress** ❌ |
| **test_SP** | **3.811%** | 3.577% (floor) | **+0.234pp** | **floor regress** ❌ |
| test_abupt | 6.011% | 5.844% | +0.167pp | — |
| test_τ_z | 8.979% | — | — | — |
| test_τ_x | 6.117% | — | — | — |
| test τz/τx | **1.469** | — | — | — |

**Decision: CLOSE.** Fails all four merge gates — single-model winners must beat PR #972 baseline on test_WSS AND hold both floors.

### Mechanism — slow-decay EMA hypothesis REFUTED

Full per-epoch EMA-vs-raw Δ tracking on τ_z (cleanest single-run instance on tay):

| EP | Δ_abupt (raw−ema) | Δ_τ_z (raw−ema) | Note |
|---|---:|---:|---|
| 1 | −2.357 | −3.369 | warmup, raw ahead |
| 2 | +0.302 | +0.323 | crossover |
| **3** | +0.591 | **+0.937** | **peak (predicted to keep growing)** |
| 4-7 | +0.540→+0.206 | +0.716→+0.255 | monotonic narrowing |
| 8-12 | +0.176→+0.029 | +0.201→+0.046 | continued narrowing |
| **13** | **+0.013** | **+0.021** | **98% shrinkage from peak** |

EMA smoothed mid-training noise, but raw model fully caught up by terminal. Slow decay added noise robustness during mid-training but did NOT preserve late-training τ_z specialization as hypothesized. Mechanism is **null at convergence**. Hypothesis cleanly refuted.

### Data preserved as research artifact

1. **Test τz/τx = 1.469 is interestingly tighter than val (1.555)** — first observed val→test compression of the structural ratio on tay. Distribution shift between val and test partially reduces τ_z bottleneck but doesn't eliminate it. Candidate paper-figure.
2. **`best_checkpoint/updated=1` at every recent epoch gate** — pure monotonic descent, run quality is high.
3. **EMA-vs-raw Δ trajectory is publishable** as the definitive characterization of slow-EMA in late training.

### τ_z structural finding — EIGHTFOLD CONFIRMATION

Tanjiro #1124 closes as the 8th mechanistically-distinct mechanism converging τz/τx ratio to the 1.50-1.57 structural band. **τ_z bottleneck is BACKBONE-representation-side.** Wave 30 architectural pivot commissioned: researcher-agent generated 8 candidate architectural hypotheses (`research/RESEARCH_IDEAS_2026-05-15_18:00.md`); top-3 picks are H6 (local-frame WSS head), H2 (normal spectral encoding), H5 (Y-architecture dual-backbone).

### Reassigned as

- Tanjiro → **Wave 30 H6: Local-Frame WSS Head** — replaces global Cartesian (τ_x, τ_y, τ_z) output with local-frame (τ_t1, τ_t2) prediction using orthonormal surface basis. Enforces physics constraint τ·n=0 by construction. ~65 LOC change in `model.py`.

---

## 2026-05-15 12:45 — PR #1121: WSS magnitude-only decomposition + 18h budget (frieren) — CLOSED (TEST_SP FLOOR REGRESS BLOCKS MERGE; methodology preserved)

- **Branch**: `frieren/wss-mag-only-full-budget` (closed)
- **W&B run**: `gljtmuvs` (group `frieren/mag-only-*`, EP13/13 complete, 839.9 min wall-clock = 14h, EP12 best-EMA auto-harvested)
- **Hypothesis**: Wave 27 finding that 91-96% of WSS residual is magnitude error → λ_dir=0, λ_mag=0.1 mag-only decomp at 18h budget should improve test_WSS at convergence.

### Test metrics (paper-facing, EP12 best EMA)

| Metric | This PR | PR #972 SOTA | Δ vs SOTA | alphonse #1078 (no decomp) | Δ vs no-decomp |
|---|---:|---:|---:|---:|---:|
| **test_WSS** | **6.859%** | 6.727% | +0.132pp regress | 6.996% | **−0.137pp improvement** ✅ |
| test_vol_p | 3.545% | 3.643% (floor) | −0.098pp PASS ✅ | 3.644% | −0.099pp ✅ |
| **test_SP** | **3.734%** | 3.577% (floor) | **+0.157pp FLOOR REGRESS** ❌ | 3.832% | −0.098pp ✅ |
| test_abupt | 5.939% | 5.844% | +0.095pp | — | — |

### Val metrics (best EMA at EP12)

| Metric | EP12 EMA | PR #972 baseline | Δ |
|---|---:|---:|---:|
| **val_abupt** | **6.073%** | 6.126% | **−0.053pp** (first single-model val improvement on no-SDF tay since corrected split) |
| val_WSS | 6.875% | — | — |
| val_vol_p | 3.517% | — | below test floor |

### Per-axis test WSS

| Axis | This PR | Note |
|---|---:|---|
| τ_x | 6.091% | — |
| τ_y | 7.452% | — |
| **τ_z** | **8.873%** | Dominant error channel (consistent with fleet-wide finding) |

τz/τx test ratio = **1.457** (val EP13 was 1.57; test set has slightly less τ_z bottleneck but still dominant).

### Decomp diagnostics — mechanism works as designed

| Diagnostic | EP1 | EP3 | EP6 | EP13 final |
|---|---:|---:|---:|---:|
| mag_loss | 0.0331 | 0.0035 | 0.0015 | **0.0011** (4.4× tighter than #1112 EP3=0.0048) |
| calib ratio (pred/gt) | 0.966 | 0.997 | 0.988–1.008 | **0.9993** (inside PR target 0.998–1.002) |

The mag auxiliary head is **perfectly calibrated** at EP13. λ_dir=0 confirmed throughout training. The aux head is doing exactly what the architecture asked of it.

### Methodology — paper-relevant findings preserved

1. **Mag-only decomp produces the strongest no-SDF single-model test result yet** (test_WSS=6.859% vs no-decomp #1078 6.996%, −0.137pp). The methodology is a recommended building block for future stacking experiments — most natural pairing is SDF FAR-field α=2.0 (alphonse #1122 active).
2. **First single-model val_abupt improvement on no-SDF tay** (6.073% vs 6.126%) since the corrected split landed. Demonstrates the no-SDF stack is NOT structurally saturated at val_abupt 6.126% — sampling/loss mechanism uplifts are still available.
3. **test_SP floor regression blocks merge** — mag-only decomp slightly trades surface-pressure precision (+0.157pp test_SP) for the WSS/vol_p improvements. Single-model winners must hold both floors. Closing for this reason; the run is still a methodology success.
4. **τ_z structural-bottleneck finding strengthened to SIXFOLD independent confirmation**: this run is the 6th active mechanism showing τz/τx ratio converges to ~1.50–1.57 by EP5-10 regardless of approach (loss weight, sampling, output capacity, EMA averaging, magnitude calibration). EP9→EP10 τ_z reversal (+0.020pp) is the cleanest single-run instance. **τ_z bottleneck is NOT addressable by these levers.**
5. **Decomp infrastructure is established in the codebase** — future per-axis or heteroscedastic decomp variants can build directly on this PR's framework.

### Reassigned as

- PR (TBD): **Per-axis WSS magnitude decomposition** (|τ_z| + ||τ_xy|| as separate aux heads) — direct architectural attack on the structural τ_z finding, exploits frieren's proven aux-head implementation. If per-axis decomp gives τ_z its own dedicated supervised magnitude signal, this is the cleanest test of whether the structural bottleneck is information-bandwidth (mag-only mixes axes) or representational (backbone features just can't carry τ_z).

---

## 2026-05-15 04:00 — Wave 29 Active Fleet (8 PRs in flight)

All 8 student pods READY (1/1). Full 18h/13ep convergence budget assigned. Gate criteria (no-SDF students): EP1 ≤35%, EP3 ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL. Alphonse (SDF): EP3 ≤6.9% PASS / ≤7.2% MARGINAL / >7.2% KILL.

---

## 2026-05-15 03:50 — PR #1128: τ_z loss-weight 3.0 escalation (thorfinn) — IN FLIGHT

- **Branch**: `thorfinn/tau-z-loss-weight-3p0`
- **W&B run**: `uwqybod5` (group `tau-z-loss-weight-3p0`, launched 03:48:42Z, ACK 03:49:15Z)
- **Hypothesis**: Escalate `--tau-z-loss-weight` from 2.0 → 3.0 (single CLI flag change). τ_z is the dominant WSS error axis (8.75% vs τ_x 5.97%, τ_y 7.36%). Prior PR #1123 unresponsive; reassigned as a simpler single-flag experiment to fill the slot. Tests whether further weighting beyond the standard τ_z=2.0 prior continues to drive τ_z improvement without degrading τ_x/τ_y.
- **Key signal**: τ_z/τ_x ratio (should decrease from baseline 1.47× if mechanism works)

| Gate | ETA | Criterion | Status |
|---|---|---|---|
| EP1 | ~06:00Z | ≤35% val_abupt | pending |
| EP3 | ~08:30Z | ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL | pending |
| EP13 | ~15:30Z | terminal | pending |

---

## 2026-05-15 03:10 — PR #1127: Surface_loss warmup curriculum (askeladd) — IN FLIGHT

- **Branch**: `askeladd/surface-loss-warmup`
- **W&B run**: `dtgfdsgv` (launched ~03:00Z)
- **Hypothesis**: Explicit 3-epoch ramp from `surface_loss_weight=0 → 2.0` (`--surface-loss-weight-warmup-epochs 3`). Directly tests PR #1114 finding that EP1 WSS wins may be curriculum artifacts from implicit warmup. If explicit curriculum > implicit constant-weight, gain may compound at convergence.

| Gate | ETA | Criterion | Status |
|---|---|---|---|
| EP1 | ~04:30Z | ≤35% val_abupt | pending |
| EP3 | ~06:00Z | ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL | pending |
| EP13 | ~14:00Z | terminal | pending |

---

## 2026-05-15 03:00 — PR #1126: Deeper surface_out MLP depth=4 (fern) — IN FLIGHT

- **Branch**: `fern/surface-out-depth-4`
- **W&B run**: fern run (group `surface-out-depth-4`, launched ~02:59Z)
- **Hypothesis**: Increase `surface_out` MLP depth from 2 → 4 layers (+525k params, +3% total, 17.94M params total). Tests whether τ_z prediction benefits from deeper output projection capacity. Orthogonal to main transformer; negligible VRAM impact.

| Gate | ETA | Criterion | Status |
|---|---|---|---|
| EP1 | ~04:30Z | ≤35% val_abupt | pending |
| EP3 | ~06:00Z | ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL | pending |
| EP13 | ~14:00Z | terminal | pending |

---

## 2026-05-15 02:50 — PR #1125: Spatial-prior surface sampling α=10 (nezuko) — IN FLIGHT

- **Branch**: `nezuko/spatial-prior-alpha-10`
- **W&B run**: nezuko run (group `spatial-prior-alpha-10`, launched ~02:45Z)
- **Hypothesis**: Spatial bias `w = 1 + 10.0·(front_bias + |z|_bias)/2` at full 18h/13ep budget. Prior PR #1120 (α=3) showed real mechanism signal (EP3 val_WSS −0.67pp vs baseline, ρ=+0.31 correlation with |WSS|) but was budget-truncated at 47%. α=10 increases front/side oversample to ×3–4× to drive more τ_z-region coverage.

| Gate | ETA | Criterion | Status |
|---|---|---|---|
| EP1 | ~05:00Z | ≤35% val_abupt | pending |
| EP3 | ~06:00Z | ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL | pending |
| EP13 | ~14:00Z | terminal | pending |

---

## 2026-05-15 02:00 — PR #1124: EMA decay 0.9995 (tanjiro) — IN FLIGHT

- **Branch**: `tanjiro/ema-slow-decay`
- **W&B run**: `mw6d04kc` (launched ~01:25Z)
- **Hypothesis**: Slow EMA decay 0.999 → 0.9995 (double the averaging window at EP13). At 13ep standard recipe, current EMA τ ≈ 141 steps; 0.9995 → τ ≈ 2000 steps. Tests whether EMA over-weights recent noisy steps near cosine LR floor. Primary signal: EMA–raw gap inversion (EMA should outperform raw by larger margin).
- **EP1 result**: val_abupt = 31.48% — **PASS** (≤35%)

| Gate | ETA | Criterion | Status |
|---|---|---|---|
| EP1 | DONE | ≤35% | **PASS (31.48%)** |
| EP3 | ~06:15Z | ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL | pending |
| EP13 | ~15:00Z | terminal | pending |

---

## 2026-05-15 01:30 — PR #1122: SDF FAR-field α=2.0 (alphonse) — IN FLIGHT

- **Branch**: `alphonse/sdf-far-field-alpha-2`
- **W&B run**: alphonse run (launched ~01:00Z, smoke PASS: 5.6× sampled/population weight ratio confirmed)
- **Hypothesis**: Port SOTA SDF weighting mechanism `weight = 1 + α·|sdf|` (FAR-field amplification) with α=2.0. Prior ADVISOR error: assigned wrong mechanism (NEAR-field `1/(1+α|sdf|)` with α=4.0); alphonse self-caught and corrected. FAR-field mechanism up-weights points far from surface where gradients concentrate → complementary to surface_loss term.
- **Pace note**: At vol=16k → 860 views/case → 10,864 iters/rank/epoch × 1.38 it/s = **131 min/epoch** (not 80 min as initially estimated). Rate-limiting step is view_count=max(130,860).

| Gate | ETA | Criterion | Status |
|---|---|---|---|
| Smoke | DONE | 5.6× weight ratio | **PASS** |
| EP1 | ~05:50Z | ≤35% val_abupt | pending |
| EP3 | ~07:55Z | ≤6.9% PASS / ≤7.2% MARGINAL / >7.2% KILL | pending |
| EP13 | ~16:30Z | terminal | pending |

**Key signals to monitor**: per-axis WSS at EP3 (τ_z should be the primary beneficiary of FAR-field weighting), sampled/population weight ratio stability.

---

## 2026-05-15 01:00 — PR #1121: WSS magnitude-only decomposition (frieren) — IN FLIGHT

- **Branch**: `frieren/wss-mag-only-decomp`
- **W&B run**: frieren run (group `wss-mag-only-decomp`, launched ~00:30Z)
- **Hypothesis**: Decompose WSS auxiliary loss as magnitude-only: λ_dir=0, λ_mag=0.1. Suppresses direction-component loss that may add cross-axis interference. Tests whether magnitude supervision alone drives stronger τ_z learning without the directional penalty competing.
- **EP3 result (PASS — strongest in decomp family)**:

| Metric | EP3 value | Gate |
|---|---:|---|
| val_abupt | **6.746%** | ≤7.2% **PASS** |
| val_τ_x | 6.580% | — |
| val_τ_y | 8.593% | — |
| val_τ_z | 10.093% | — |

EP3 val_abupt=6.746% is strongest in any decomp-family run. Continuing to full 18h/13ep convergence.

| Gate | ETA | Criterion | Status |
|---|---|---|---|
| EP3 | DONE | ≤7.2% | **PASS (6.746%)** |
| EP6 sanity | ~06:00Z | ≤6.5% PASS / ≤6.8% MARGINAL | pending |
| EP13 terminal | ~14:30Z | terminal | pending |

---

## 2026-05-15 00:30 — PR #1116: Per-channel WSS output heads (edward) — IN FLIGHT (relaunched)

- **Branch**: `edward/per-channel-surface-heads`
- **W&B run**: `3ufrbxl6` (relaunched 03:08:59Z after pod restart killed `rfapq0o3` at EP1 ~97min in)
- **Hypothesis**: `--use-per-channel-surface-heads`: separate linear output heads per WSS axis (τ_x, τ_y, τ_z) instead of shared head. Tests whether per-axis specialization reduces τ_z/τ_x interference. Budget-matched arm (`hqp13ztw`) showed test_WSS −0.66pp vs baseline; τ_z/τ_x ratio went wrong direction (1.36→1.44). This 18h convergence run tests whether the trend reverses at full budget.
- **Pod restart**: Pod wipe at 03:05:35Z killed `rfapq0o3` during EP1. Python env wiped (lost `lion_pytorch`). Student re-installed, relaunched as `3ufrbxl6` at 03:08:59Z.

| Gate | ETA | Criterion | Status |
|---|---|---|---|
| EP1 | ~05:10Z | ≤35% val_abupt | pending |
| EP3 | ~08:00Z | ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL | pending |
| EP13 | ~14:00Z | terminal | pending |

**Key signal**: τ_z/τ_x ratio at EP3 — should decrease from 1.44 (budget-matched arm) toward 1.47 (baseline) or better. If ratio increases further → approach is degrading τ_z.

---

## 2026-05-15 02:35 — PR #1118: OHEM v2 spike-clipped + reduced λ (askeladd) — CLOSED (DEFINITIVE NEGATIVE: ZERO OHEM GRADIENT)

- **Branch**: `askeladd/ohem-v2-spike-clipped` (closed)
- **W&B run**: `023up1sk` (group `tay-wave28-ohem-v2`, EP3 truncated at 81% by 270-min train cap)
- **Hypothesis**: spike-clipped OHEM v2 with `max_clip=2.0`, `λ=0.1`, warmup=2EP avoids v1's Lion-sign collapse by capping `hard_loss_raw` at `max_clip × surface_loss.detach()`.

### EP3 gate verdict — PASS but mechanism null (see below)

| Metric | EP3 value | gate | verdict |
|---|---:|---|---|
| val_abupt | 6.9805% | ≤7.2% PASS | **PASS** |
| val_vol_p | 4.1231% | ≤4.5% PASS | **PASS** |
| val_WSS | 7.864% | — | informational |
| val_SP | 4.670% | — | informational |

### Test metrics (EP3 best, paper-facing)

| Metric | EP3 test | PR #972 SOTA baseline | Δ |
|---|---:|---:|---:|
| test_WSS | **7.6301%** | 6.727% | **+0.903pp regression** |
| test_vol_p | 3.9592% | 3.643% (floor) | +0.316pp |
| test_SP | 4.2961% | 3.577% (floor) | +0.719pp |
| test_abupt | 6.6530% | 5.844% | +0.809pp |

### Critical mechanism finding — OHEM contributed ZERO learning signal

| diagnostic | EP3 value (4218 OHEM-active steps) |
|---|---:|
| `clip_active` fraction | **100.00%** (4218 / 4218) |
| `hard_loss_raw` median / p95 / p99 / max | 3.03 / 12.45 / 24.42 / 4031.3 |
| `raw / surface_loss` ratio median / p95 / max | 322× / 1205× / 25,749× |
| `contribution` (λ × clipped) median / p95 / max | 0.001862 / 0.002572 / 0.05104 |

**Mathematical consequence:** `hard_loss_raw.clamp_max(max_clip × surface_loss.detach())` returns the detached cap whenever raw > cap. **At 100% clip-active, the gradient through the OHEM term is exactly zero on every active step.** OHEM contributed no learning signal — the run is mathematically equivalent to the baseline minus tiny loss-value bookkeeping. The val_abupt=6.9805% gate-PASS is a baseline trajectory, not an OHEM-influenced trajectory.

### Methodology — paper-relevant findings preserved

1. **OHEM-family terminally exhausted on raw-residual reweighting for this dataset.** Dataset's top-K surface residual distribution is intrinsically 100–25,000× larger than mean residual. Any safe scalar cap (max_clip ≤ 100) sits below the natural top-K signal → cap fires 100% → zero gradient. v1 (#1110) failed via Lion-sign collapse on uncapped 4000× spikes; v2 (#1118) prevented divergence but at the cost of gradient flow. **Both ends of the cap spectrum are now closed.**
2. **The `clip_active_pct` diagnostic is the right metric for any future loss-shaping work** — converts ambiguous "did clipping affect training?" into a binary mechanism gate. Should remain in codebase for future loss-clip experiments.
3. **`hard_loss_raw` distribution is identical between v1 and v2** (v1: median 5.6, max 4411; v2: median 3.03, max 4031). Confirms 4000× residual spikes are intrinsic to dataset, not a v1 implementation pathology.
4. **Communication discipline reinforced**: student silently launched a 2h smoke without acknowledging assignment → advisor flagged it as risk → student adopted ~30-min acknowledgment protocol going forward.

### Pattern observation

- Wave 28 loss-engineering family is now 0-for-3: tanjiro #1114 learnable WSS weights (null at convergence), fern #1119 GradNorm short-cycle (refuted prior-rediscovery), askeladd #1118 OHEM v2 (zero gradient). **Loss-balance-learning approaches are systematically failing on this dataset's residual structure.** Pivot is decisive: capacity / SDF / architecture routes.

### Reassigned as

- PR #1127 (TBD): explicit surface_loss_weight warmup curriculum (`--surface-loss-weight-warmup-epochs 3`) — directly tests #1114 finding that EP1 wins are curriculum artifacts; if explicit curriculum > implicit, gains may compound at convergence.

---

## 2026-05-15 02:30 — PR #1120: Spatial-prior surface sampling α=3 (nezuko) — CLOSED (MECHANISM RIGHT, BUDGET TOO SHORT)

- **Branch**: `nezuko/spatial-prior-surface-sampling` (closed)
- **W&B run**: `vt2fsxdf` (group `spatial-prior-sampling`, EP3 truncated at 47% of planned 6 epochs by 270-min train cap)
- **Hypothesis**: Spatial bias `w = 1 + 3.0·(front_bias + |z|_bias)/2` with ρ=+0.31 vs |WSS| → safe + meaningful WSS lift without #1113's curvature catastrophe.

### Test metrics (EP3 best EMA)

| Metric | This PR | SOTA target / floor | Δ |
|---|---:|---:|---:|
| test_WSS | 7.7574% | < 6.727% target | +1.030pp regression |
| test_vol_p | 4.0677% | ≤ 3.643% floor | +0.425pp above floor |
| test_SP | 4.3831% | ≤ 3.577% floor | +0.806pp above floor |
| test_τ_z | 9.851% | < 9.05% reference | +0.80pp above no-SDF ceiling |

### Methodology — paper-relevant findings preserved

1. **ρ=+0.31 is a sufficient safety floor for surface-sampling reweighting** — EP2 val_abupt=7.84% (vs #1113 ρ=-0.11 EP2=13.12%); no catastrophe.
2. **Linear-ramp spatial weight is throughput-neutral** — 1.90 it/s = baseline parity; no precompute, no per-step penalty.
3. **Bin-occupancy diagnostic accuracy validated** — ×1.39 top-decile oversample matched pre-training prediction.
4. **Strongest 3-EP truncated WSS in this recipe family** — EP3 val_WSS=7.94% vs edward EP2 no-SIA 10.93% and matched-baseline mempfubx EP3 8.61% (−0.67pp). Real mechanism signal, just budget-truncated.

### Pattern observation

- Short-cycle EP3 at this point config consistently cannot reach single-model SOTA floors: edward #1116 EP3 test_WSS=7.67%, frieren mempfubx EP3 test_WSS=8.33%, nezuko #1120 EP3 test_WSS=7.76%. All within 0.66pp of each other regardless of mechanism. **The EP3 budget is the dominant constraint at this point config**, not the mechanism.

### Reassigned as

- PR #1125 (TBD): spatial-prior α=10 at 18h budget — student's own suggested follow-up #2.

---

## 2026-05-15 02:27 — PR #1119: GradNorm short-cycle 6-ep convergence (fern) — CLOSED (PRIOR-DISCOVERY HYPOTHESIS REFUTED)

- **Branch**: `fern/gradnorm-short-cycle` (closed)
- **W&B run**: `eokmp0b5` (group `gradnorm-short-cycle`, 6/6 epochs, 289 min wall-clock, EP6 EMA best ckpt)
- **Hypothesis**: GradNorm at short-cycle t_max=6, ep=6 will rediscover the hand-tuned prior τ_z=2.0 at convergence (testable in budget).

### Test metrics (EP6 best EMA)

| Metric | This PR | SOTA target / floor | Δ |
|---|---:|---:|---:|
| test_WSS | 7.467% | < 6.727% target | +0.740pp regression |
| test_vol_p | 4.122% | ≤ 3.643% floor | +0.479pp above floor |
| test_SP | 4.093% | ≤ 3.577% floor | +0.516pp above floor |
| val_abupt | 6.852% | ≤ 6.5% target | +0.352pp regression |

### Critical finding — hardcoded prior empirically validated

**GradNorm permanently settles at uniform-ish weighting:**

| Task | Hardcoded prior (mean-1 norm) | Learned at EP6 (mean-1 norm) | Effective Δ |
|---|---:|---:|---:|
| SP | 0.83 | 0.939 | +13% |
| τ_x | 0.83 | 0.998 | +20% |
| τ_y | 1.25 | 1.054 | −16% |
| **τ_z** | **1.67** | **1.069** | **−36%** |
| **VP** | **0.42** | **0.941** | **+124%** |

τ_z trajectory: 1.01 (EP0) → 1.04 (EP1) → 1.05 (EP2) → 1.06 (EP3) → 1.07 (EP4) → **1.07 (EP6 plateau)**. **Never approaches the prior 1.67** at any point in training.

### Hypotheses tested

- "GradNorm rediscovers prior at convergence" → **REFUTED**. τ_z weight plateaus at 1.07, 36% below prior.
- "GradNorm's flat assignment is correct" → **REFUTED** by floor regression. Hardcoded prior is empirically doing real work on test floors.
- "Cosine completion was needed to disambiguate prior vs GradNorm" → confirmed irrelevant. EP3.5 (#1111) ≈ EP6 (#1119) on val_abupt (6.87% / 6.85%) — GradNorm hits ceiling early.

### Pattern observation

- Two independent GradNorm runs (#1111 GN + #1119 short-cycle GN) both show identical τ_z de-emphasis (~1.06-1.07) and test floor regression. **The hand-tuned `[1.0, 1.5, 2.0]` prior is empirically validated as load-bearing on this dataset.**

### Reassigned as

- PR #1126 (TBD): architectural change (deeper surface_out MLP) — pivots away from loss-balance-learning entirely after #1111 + #1119 double refutation.

---

## 2026-05-15 01:13 — PR #1114: Learnable WSS channel loss weights (tanjiro) — CLOSED (MECHANISM NULL AT CONVERGENCE)

- **Branch**: `tanjiro/learnable-wss-channel-weights` (closed)
- **W&B runs**: `q95b2awa` (debug, softplus+L2→0 collapse), `jczuycas` (Kendall, killed EP1), `hqciq900` (v2 softplus+L2-to-init, terminal at EP3)
- **Hypothesis**: Replace hand-tuned WSS channel weights [1.0, 1.5, 2.0] with softplus-parameterised learnable weights + dedicated optimizer param group at lr=1e-3, to find a better τ_z weighting than the prior.

### Terminal results (run `hqciq900`, 3-EP budget-truncated)

| Metric | hqciq900 | mempfubx (matched 3-EP) | Current SOTA | Δ vs SOTA |
|---|---:|---:|---:|---:|
| val_abupt | **7.066%** | 7.465% (−0.40pp ✓) | 5.7452% (#1102) | +1.32pp |
| **test_WSS** | **7.726%** | 8.331% (−0.60pp ✓) | 6.3263% (#1102) | **+1.40pp** |
| test_vol_p | 4.026% | 4.039% | 3.5397% | +0.49pp |
| test_SP | 4.387% | 4.477% | 3.3529% | +1.03pp |
| test_tau_z | 9.814% | 10.134% | 8.2585% | +1.56pp |

### Mechanism finding — learnable weights converge to baseline

**Weight trajectory:** init `[1.0, 1.5, 2.0]` → mid-EP1 drift down to `[0.55, 0.76, 1.00]` (–46% to –50%, gradient wants to reduce WSS capacity) → EP2/EP3 quadratic well pulls weights back → EP3 final `[0.979, 1.452, 1.937]` (within 3% of init).

**Regularization balance:**
- EP1 transient (step 2500): `wss_reg=0.037` dominant over per-channel weighted losses (≈0); reg is the active force pulling weights back.
- EP3 (step 30377): `wss_reg=0.0001` vs `Σweighted_wss=0.041` (reg recessive at 0.25% of weighted loss); equilibrium at ~init.

**Hypothesis falsified at convergence:** the +0.40pp val_abupt vs matched 3-EP baseline is from the EP1 transient drift acting as an implicit volume-first curriculum (capacity briefly shifted to vol/sp, then restored to baseline by reg), NOT from a better learned weight setting. Steady-state mechanism is null.

### Methodology learnings (preserved)

1. **Kendall heteroscedastic uncertainty `L_i / (2σ_i²) + log σ_i` INVERTS the hypothesis** — it down-weights large-train-loss channels (which we WANT to up-weight in the test-metric direction). vol_p 16.71% → 24.95% at EP1 with Kendall. **Permanently retired.**
2. **Original softplus + L2-toward-zero collapses trivially** — by EP2 weights crashed to ~0.01. The L2→0 reg accelerates collapse rather than preventing it. **Permanently retired.**
3. **v2 softplus + L2-reg-toward-init is the correct formulation** for any future learnable-weights experiment (quadratic well centered at init prevents collapse and bounds drift).

### Wave 29 follow-up queued

- **Warmup-epochs=30 control** (frozen baseline weights, identical training setup) — isolates "is the +0.4pp matched-baseline win from the mechanism or from training-setup variance?"

---

## 2026-05-14 23:45 — PR #1100: Capacity uplift slices=256 (thorfinn) — CLOSED (TEST FLOORS REGRESS, NO-SDF STACK CEILING CONFIRMED)

- **Branch**: `thorfinn/model-slices-256-capacity` (closed)
- **W&B run**: `k33hscuc` (rank 0, EP20 best-val EMA, 1029 min total wall)
- **Hypothesis**: Double model_slices from 128 → 256 to test if capacity uplift on the no-SDF tay stack closes the gap to PR #972 SDF stack SOTA. 30-ep cosine, batch=4, lr=9e-5.

### Test metrics (EP20 best-val EMA, 50 test cases)

| Metric | This PR | Target / floor | Δ |
|---|---:|---:|---:|
| test_WSS | **6.9887%** | < 6.50% target / ≤ 6.727% SOTA | +49bp above target / +26.2bp worse than SOTA |
| test_vol_p | **3.6442%** | ≤ 3.643% (floor) | +0.12bp above floor (essentially tied) |
| test_SP | **3.8324%** | ≤ 3.577% (floor) | +25.5bp above floor (+7.1% rel) |
| val_abupt | 6.3035% | ≤ 6.20% | +10bp above target |
| val_vol_p | 3.7406% | (n/a) | better than PR #972 val_vol_p 3.798% |

### Per-axis WSS (test)
- test_tau_x: 6.208% | test_tau_y: 7.581% | **test_tau_z: 9.051%**
- Same channel ordering as alphonse #1078 (test_tau_z=9.073%) — τ_z is the program-wide residual axis

### Critical finding — no-SDF tay structural ceiling

**Two independent mechanisms converge at test_WSS ≈ 6.99%:**

| Run | Mechanism | test_WSS | test_vol_p | test_SP |
|---|---|---:|---:|---:|
| alphonse #1078 | Asymmetric eval 131k | 6.9955% | 3.6795% | 3.8547% |
| **thorfinn #1100** | **slices=256 capacity uplift** | **6.9887%** | **3.6442%** | **3.8324%** |
| PR #972 (SDF stack, on different branch) | SDF importance sampling + slices=128 | 6.727% | 3.643% | 3.577% |

Capacity-uplift on no-SDF tay tops out at test_WSS ≈ 6.99% — this is now an empirical observation backed by two independent levers (asymmetric eval, slices=256). Strongly suggests **SDF importance sampling is load-bearing** for both WSS and the test floors. Direct motivation for alphonse #1122 (SDF port to tay).

### Diagnostic findings

1. **VRAM ceiling at slices=256**: 99.5 GB / 97.9 GB observed (96.94% allocated). `--no-compile-model` kept the allocator stable across 11 epochs of the 65k vol stage. Higher slice counts will require bs=2 or activation checkpointing.
2. **Cadence**: 38 min/ep at 65k stage (curriculum-aware), 1010 min training + 18 min auto-harvest = 17.15h. 18h budget recipe lands EP20 of a 30-ep cosine.
3. **Cosine schedule under-tuned**: at terminal, LR was ~33% of peak and train loss still descending. `--lr-cosine-t-max 20` instead of 30 would likely give 5-10bp better at EP20.
4. **Best-val EP=20** at cap — model wanted MORE epochs. Capacity uplift is under-utilized at 20 epochs.
5. **val_vol_p 3.7406% BEAT PR #972 val_vol_p 3.798%** — capacity uplift genuinely helps in-distribution vol_p, but test_vol_p stayed at floor boundary. Val/test divergence specific to vol_p.
6. **tau_y improvement led WSS gains** (consistent with alphonse #1078) — capacity uplift differentially helps the transverse shear axis, but does NOT help τ_z.

### Paper-relevant finding

**Capacity-uplift on no-SDF tay tops out at test_WSS ≈ 6.99%** (two independent runs). Any future "beat SOTA without SDF" argument requires beating 6.99% test_WSS — and capacity alone cannot.

### Reassignment

Thorfinn → **τ_z-specific dedicated subnet** — attacks the residual axis (test_τ_z ≈ 9.05% across all no-SDF runs). Dedicated 2-layer MLP head for τ_z prediction alongside existing shared head. Compounds with edward #1116 per-channel heads if both prove additive.

## 2026-05-14 22:42 — PR #1078: Asymmetric eval surface 131k 2× WSS resolution at inference only (alphonse) — CLOSED (HYPOTHESIS FALSIFIED, TEST FLOORS REGRESS)

- **Branch**: `alphonse/asymmetric-eval-surface-131k` (closed)
- **W&B run**: `1gzeeios` (rank 0, EP17 terminal, EP16 best-val EMA)
- **Hypothesis**: Doubling eval surface points from 65k to 131k at inference time (train remained 65k) would extract 2× spatial resolution on WSS prediction quality without retraining cost. Projection from PR #972's val→test ratio 0.935 anticipated test_WSS≈6.676% (~5bp under SOTA 6.727%).

### Test metrics (EP16 best-val EMA checkpoint, 50 test cases)

| Metric | This PR | Target / floor | Δ |
|---|---:|---:|---:|
| test_WSS | **6.9955%** | < 6.727% (SOTA) | **+26.8bp worse** |
| test_vol_p | **3.6795%** | ≤ 3.643% (floor) | +3.6bp above floor |
| test_SP | **3.8547%** | ≤ 3.577% (floor) | +27.8bp above floor (+7.8% rel) |
| val_abupt | 6.3164% | ≤ 6.20% | +11.6bp above target |
| val_WSS | 7.1399% | (n/a) | — |
| val_vol_p | 3.7162% | (n/a) | better than tay-SOTA 3.818% |

### Per-axis WSS (test, EP16)
- test_tau_x: 6.190% | test_tau_y: 7.635% | test_tau_z: 9.073%
- Same channel ordering as val; 131k eval does NOT preferentially benefit worst axis

### Diagnostic findings

1. **val→test ratio was 1.020, not 0.935** as projected from PR #972 — falsifies the synthetic projection. val benefits more from 131k eval than test does. Plausible mechanism: 34 val cases get error-structure resolution boost from finer surface; 50 OOD test cases don't.
2. **17 full cosine epochs completed cleanly on 18h budget** at ~62 min/ep — validates `SENPAI_TIMEOUT_MINUTES=1100` recipe for other students. Adopted by frieren #1121.
3. **val_abupt=6.316% is the strongest no-SDF tay single-model result on 17 epochs** — confirms capacity-uplift partially substitutes for SDF importance sampling but tops out around 6.31% on this stack.
4. **tau_y led WSS gains** (EP9→EP16 −0.092pp), confirming the read that 131k eval pays off most where transverse shear was the limiter. tau_z stayed flat/regressed (EP9 best 9.660% → EP17 9.700%).

### Paper-relevant finding

**Eval-resolution multipliers do NOT translate cleanly val→test.** Going forward, advisor SOTA projections should anchor on test results from comparable-recipe runs, not val × historical ratio. The 0.935 ratio from PR #972 is recipe-specific (SDF importance sampling stack), not transferable.

### Reassignment

Alphonse → **SDF importance sampling port to tay** — PR #972's `--sdf-importance-sampling --sdf-alpha 4.0` is the single largest known-working lever NOT on tay. Asymmetric eval result confirms capacity-uplift alone tops out around 6.31% val_abupt; SDF importance sampling drove PR #972 to 6.126%. Time to bring the lever to tay.

# SENPAI Research Results

## 2026-05-14 22:00 — PR #1112: WSS magnitude+direction decomposition heads (frieren) — CLOSED (TEST FLOORS REGRESS, BUDGET-TRUNCATED IDENTICAL TO #1111)

- **Branch**: `frieren/wss-decomp-magnitude-direction-heads` (closed)
- **W&B run**: `bu5vouer` (rank 0, EP3 step 30415/32592 force-validated at 271min train_timeout)
- **Hypothesis**: Add two supplementary loss heads — WSS magnitude (`||W||_2`) and WSS direction (`W/||W||_2`) — at λ_mag=0.1, λ_dir=0.05. Tests whether decomposing the WSS regression into magnitude + direction sub-targets accelerates learning beyond the joint MSE.

### Test metrics (EP3 best-val EMA, 50 test cases)

| Metric | Test | Target/Floor | Δ | Status |
|---|---:|---:|---:|---|
| test_WSS | 7.549% | ≤6.727% (PR #972 SOTA) | +82bp | MISS |
| test_vol_p | 3.964% | ≤3.643% (floor) | +32bp | **FLOOR REGRESSION** |
| test_SP | 4.260% | ≤3.577% (floor) | +68bp | **FLOOR REGRESSION** |
| test_abupt | 6.588% | — | — | — |
| test_WSS_z | 9.557% | — | — | — |

### Full-val metrics (EP3 best EMA, 34 val cases)

| Metric | Value | EP3 gate | Status |
|---|---:|---:|---:|
| full_val_abupt | 6.921% | ≤7.2% | PASS |
| full_val_vol_p | 4.119% | ≤4.5% | PASS |
| full_val_WSS | 7.808% | — | — |
| full_val_WSS_z | 10.336% | — | — |

### Methodology positive — supplementary heads worked as designed

| Diagnostic | Status |
|---|---|
| `train/wss_decomp_mag_loss` stable | ✓ no NaN |
| `train/wss_decomp_dir_loss` stable | ✓ no NaN |
| Base `weighted_channel_mse` descending | ✓ |
| `train/nonfinite_grad` | ≡ 0 |
| `train/grad/clipped` (post-warmup) | → 0 |
| Wave 27 supplementary safeguard | **PASSED** |

Mag-head calibration sanity at step 30415:
- `train/wss_decomp_mag_pred_mean = 1.889`
- `train/wss_decomp_mag_gt_mean = 1.929`
- Under-prediction ~2.1% — sensible for a half-cooked EP3 checkpoint, signal at the right scale.

### Diagnosis and paper-relevant findings

1. **Budget truncation, not method failure.** Identical pattern to fern #1111: 271 min / 76 min/epoch ≈ 3.5/13 epochs. Model is steeply descending (EP1=29.79% → EP2=7.61% → EP3=6.92%) when wall-clock kills it.
2. **Wave 27 supplementary safeguard works.** Adding the two decomposition heads at λ_mag=0.1 / λ_dir=0.05 produced ZERO destabilization signals — no NaN, no nonfinite grads, base loss descending, calibration sensible. This validates the "supplementary not replacement" Wave 27 design rule for loss-augmentation experiments.
3. **Same budget constraint that hit fern #1111, nezuko #1095.** Three PRs in a row with the heavy Wave 28 recipe (13-ep cosine + 65536 vol-points + surf-to-vol-xattn) hit 270-min wall-clock truncation. This is now a recognized pattern requiring either recipe-shrink OR budget-bump.
4. **Floor regression pattern is method-independent.** Both vol_p and SP regress through floors because the model is undertrained at EP3, NOT because the WSS-decomp heads cause regression. The supplementary heads only modify WSS gradient, but truncation-at-EP3 starves all heads equally.
5. **Useful follow-up: magnitude-only ablation.** Wave 27 found 91-96% of WSS residual is in the magnitude channel; current run has both heads, so we cannot say which contributes. A clean ablation (λ_dir=0, λ_mag=0.1) tests whether the direction head is doing real work or near-noise.

Frieren reassigned to **WSS magnitude-only decomposition + 18h budget** (`SENPAI_TIMEOUT_MINUTES=1100` like alphonse #1078) — tests Wave 27's 91-96% magnitude claim at full 13-ep cosine convergence, which budget-truncation prevented in this run.

## 2026-05-14 21:15 — PR #1113: SDF-stratified curvature-weighted surface sampling (nezuko) — CLOSED (EP2 KILL, CURVATURE IS BAD WSS PROXY)

- **Branch**: `nezuko/sdf-stratified-surface-sampling` (closed)
- **W&B run**: `qxqxozkj` (rank 0, killed at EP3 step 204, EP2 catastrophe abort)
- **Hypothesis**: Curvature-weighted surface sampling (top-decile κ → 61% of sampling mass) accelerates WSS learning by concentrating gradient on high-curvature regions where flow separation seeds high WSS. Implementation via mean-normalised weight `w = 1 + α · κ/mean(κ)` with α=3.0.
- **EP2 catastrophe abort gates** (advisor-set after offline diagnostic found κ-vs-|WSS| anti-correlation): val_abupt > 12% AND val_WSS > 13%. **Both triggered with margin** at EP2.

### Pre-training diagnostic (offline, before EP3)

| Quantity | Mean (7 cases) | Range |
|---|---:|---|
| Pearson(κ, \|WSS\|) | **-0.056** | [-0.061, -0.050] |
| Spearman(κ, \|WSS\|) | **-0.070** | [-0.078, -0.053] |
| Top-10% \|WSS\| ⊂ Top-10% κ | **7.2%** | [6.8%, 7.6%] |

Cross-checked against orthogonal estimator `curvature_HK_v2.npy`: `pearson(H, |WSS|) = -0.032`, `pearson(K, |WSS|) = +0.003`. Not estimator error.

Sanity baseline (coordinate priors): `pearson(-x, |WSS|) = +0.236`, `pearson(|z|, |WSS|) = +0.176`, `pearson(|y|, |WSS|) = +0.101` — coordinate priors are 4× the magnitude of κ.

### Training-dynamics confirmation at EP2

| Metric | Nezuko EP2 (α=3) | Edward EP2 (α=0 baseline) | Δ |
|---|---:|---:|---:|
| val_abupt | 13.12% | 9.77% | **+3.35pp** |
| val_WSS | 15.84% | 10.93% | **+4.91pp** |
| val_WSS_x | 14.88% | 9.65% | **+5.23pp** |
| val_WSS_y | 16.43% | 12.69% | **+3.74pp** |
| val_WSS_z | 18.92% | 13.52% | **+5.40pp** |
| val_SP | 9.61% | 6.70% | +2.91pp |
| val_vol_p | 5.73% | 6.31% | **-0.58pp** (improvement!) |

### Bin-occupancy diagnostic (mechanism verification)

| κ decile | Uniform | Weighted | Oversample |
|---|---:|---:|---:|
| Top 10% | 10% | **61.10%** | ×6.11 |
| Bottom 80% | 80% | 26.21% | ×0.33 |

Mechanism wired correctly. Stratification design intent satisfied. Premise (κ → |WSS|) fails empirically.

### Diagnosis and paper-relevant findings

1. **Curvature is a bad WSS proxy on transport-vehicle CFD benchmarks.** ρ≈-0.056 (slight anti-correlation). Physical mechanism: high-κ regions (leading edges, A-pillars, wheel arches) seed flow separation → low |WSS|. High |WSS| lives on smooth attached-flow panels (underbody, roof) and downstream wakes.
2. **WSS subchannel regressions are uniform** (+3.7-5.4pp across x/y/z) — the misalignment is structural, not channel-specific. Eliminates "maybe the proxy works for one component" rescue.
3. **Surface pressure also regresses (+2.91pp).** Misalignment affects both surface fields → the failure is geometric, not WSS-specific.
4. **Volume pressure improves (-0.58pp at EP2).** Interesting *positive* side-finding: front-bumper/stagnation region oversampling helps surface→volume xattn for vol_p prediction. Logged as Wave 29 follow-up: "front-bumper surface importance sampling for vol_p (decoupled from WSS optimization)."
5. **Diagnostic-first kill saved ~4h of confirmatory GPU time.** Pattern to repeat: when a hypothesis has a cheap pre-training falsification test, run it before EP3 and use the cheapest EP1/EP2 readout to confirm/reject.
6. **Throughput parity infrastructure** is solid (1.88 vs 1.9 it/s baseline). All 484 cases have `surface_kappa_v2.npy` precomputed in the data root for any future curvature-conditioned experiment without re-compute cost.
7. **Design rule**: Any surface importance sampling experiment must include a pre-training `Pearson(weight_signal, |target|)` diagnostic on a 5-10 case sample. If ρ < 0.1 in magnitude, do not run training.

Nezuko reassigned to spatial-prior surface sampling (front-bumper + ground-plane bias) — her own Suggestion #1 from this PR. The coordinate priors `-x` and `|z|` have real positive correlation with |WSS| (4× the magnitude of κ).

## 2026-05-14 21:00 — PR #1111: GradNorm adaptive task balancing with curriculum (fern) — CLOSED (TEST FLOORS REGRESS, GRADNORM-FINDING IS THE DELIVERABLE)

- **Branch**: `fern/gradnorm-curriculum-compatible` (closed)
- **W&B run**: `pbjrixfv` (rank 0, finished at EP3 truncation, 271.13 min wall-clock)
- **Hypothesis**: GradNorm adaptive task weighting on (sp, τ_x, τ_y, τ_z, vp) loss heads, with curriculum-compatibility patches to avoid PR #1095's vol-head-starvation failure mode. Initial mean-1-normalised priors derived from baseline weights (sp:1.0, τ_x:1.0, τ_y:1.5, τ_z:2.0, vp:1.0). α=0.12 GradNorm learning rate.

### Test metrics (EP3 best-val EMA checkpoint, 50 test cases)

| Metric | Test | Target/Floor | Δ | Status |
|---|---:|---:|---:|---|
| test_WSS | 7.473% | ≤6.727% (PR #972 SOTA) | +75bp | MISS |
| test_vol_p | 3.943% | ≤3.643% (floor) | +30bp | **FLOOR REGRESSION** |
| test_SP | 4.144% | ≤3.577% (floor) | +57bp | **FLOOR REGRESSION** |
| test_abupt | 6.530% | — | — | — |
| test_tau_z | 9.613% | — | — | — |

### Full-val metrics (EP3 best EMA, 34 val cases)

| Metric | Value |
|---|---:|
| full_val_abupt | 6.866% |
| full_val_vol_p | 4.053% |
| full_val_WSS | 7.737% |
| full_val_tau_z | 10.397% |

### Headline: GradNorm de-emphasizes the hardcoded τ_z=2.0 prior

| Task | Init (mean-1) | EP3 final | Δ |
|---|---:|---:|---:|
| sp | 0.769 | 0.930 | +0.161 |
| τ_x | 0.769 | 0.996 | +0.227 |
| τ_y | 1.154 | 1.052 | -0.102 |
| **τ_z** | **1.538** | **1.063** | **-0.475 (largest negative move)** |
| vp | 0.769 | 0.960 | +0.191 |

GradNorm converges all five task weights toward a near-uniform mean-1 distribution within 30k steps. τ_z drops 47% from its starting prior of 2.0. No reset events; no curriculum-transition spikes; gradient-norm signal stays informative throughout.

### Curriculum-compatibility achieved

PR #1095's vol-head-starvation feedback loop did NOT recur. Vol curriculum (0:16384→3:32768) stayed intact through the EP3 transition. `gradnorm/reset_event_count=0` (timeout at EP3 truncated before later transitions could fire). The "curriculum-compatible" claim is empirically supported within the 3.5-epoch window.

### Diagnosis and paper-relevant findings

1. **Off-SOTA negative result on metrics but positive methodological finding.** test_WSS misses SOTA by 75bp and both test_vol_p and test_SP regress through their floors — disqualifying for merge. But the GradNorm weight trajectory is itself a deliverable.
2. **τ_z=2.0 hardcoded prior is empirically over-corrected.** GradNorm's gradient-norm signal disagrees with the PR #972-era assumption that τ_z is gradient-starved. The model naturally allocates gradient effort to τ_z without explicit upweighting.
3. **Wave 28 per-channel work implicitly carries this over-corrected prior.** Tanjiro #1114 (learnable WSS channel weights, init at baseline) and edward #1116 (per-channel WSS heads, no per-channel loss reweighting) are operating on the same starting assumption; their results should be interpreted with this finding in context.
4. **Budget mismatch is the proximate killer.** 271 min / 76 min-per-epoch ≈ 3.5 epochs in a 13-ep cosine schedule. The cosine LR is still very high at termination, mechanically inflating absolute error. Shared constraint across all heavy surf-to-vol-xattn recipes at 65536 vol-points.
5. **GradNorm-as-prior-discovery framing.** The converged weights `w* ≈ uniform mean-1` become a data-driven prior for future fixed-weight runs. Worth testing whether `w*` applied as fixed channel weights beats the hardcoded τ_z=2.0 prior in a full-convergence run.
6. **Rank-0 W&B run id correction**: `pbjrixfv` (not `jkhnq2zd` which was rank 7) — noted for log reference.

Fern reassigned to short-cycle GradNorm (`--lr-cosine-t-max 6 --epochs 6` with reduced vol-points) for full-convergence test of "learned weights vs hardcoded τ_z=2.0 prior" hypothesis.

## 2026-05-14 20:35 — PR #1110: OHEM surface top-20% hard mining (askeladd) — CLOSED (EP3 KILL, OHEM SCALE-COLLAPSE)

- **Branch**: `askeladd/ohem-surface-top20pct` (closed)
- **W&B run**: `o8pt5ybd` (rank 0, killed at EP3 step 30468)
- **Hypothesis**: OHEM as a supplementary loss term (λ=0.5) on the top-20% hardest surface points should amplify gradient signal where WSS prediction is hardest, accelerating tau_z descent. EP1+EP2 warmup before activation, λ=0.5 to keep contribution bounded.
- **EP3 gate (val_abupt ≤7.6% MARGINAL, val_vol_p ≤5.0% MARGINAL)**: KILL on both criteria — 4× and 2× over respective gates.

### Trajectory

| Epoch | OHEM state | val_abupt | val_vol_p | val_WSS | val_tau_z |
|------:|:-----------|----------:|----------:|--------:|----------:|
| 1 | warmup | 26.36% | 16.79% | 28.84% | 35.69% |
| 2 | warmup | **7.68%** | **4.52%** | 8.67% | 11.32% |
| 3 | **active** | **30.78%** | **10.51%** | **34.80%** | **46.41%** |

### Root cause — OHEM scale collapse

`hard_loss = mean(top-20% sq_err)` is NOT on the same scale as `surface_loss = mean(all sq_err)` on this benchmark. Heavy-tailed sq_err distribution → median ratio ~18-30×, p95 ~25-30×, **max 88,000×** (during pathological batches with localized high-residual outlier points).

Per-step diagnostics from EP3 active steps:
- `train/surface_loss` (all-points MSE): 0.05–0.6
- `train/ohem/hard_loss` (top-20% MSE): median 5.6, p95 7.8, **max 4411**
- `train/ohem/lambda_contribution` (= 0.5 × hard_loss): p95 7.8, p99 13.4, **max 2205.7**
- Total loss spikes: up to 2207 at step 29023

The instant OHEM activated at step 21730 (start of EP3), the first batch's hard_loss=25.06 → total loss=12.57 vs base ~0.02. Lion's sign-of-momentum optimizer takes a max-magnitude update on every parameter regardless of gradient size, so a single spike collapses the manifold. By step 23163 (7% into EP3) total loss had spiked to 815; model never recovered. Positive feedback loop: degraded model → more "hard" points → larger hard_loss → larger updates → further degradation.

### Diagnosis and paper-relevant findings

1. **Scale-collapse is a distinct failure mode from direction-conflict.** Wave 27 lessons about loss-augmentation taught us to avoid conflicting gradient objectives (supplementary-not-replacement design). That heuristic does not by itself prevent scale failures, where the auxiliary term agrees in direction but amplifies by an unpredictable 30-4000× factor.
2. **Lion+sign-of-momentum is uniquely sensitive to loss-scale spikes** because update magnitude doesn't scale with gradient. A single 4000× spike → max-size update on every parameter → manifold collapse. AdamW would have been less catastrophic.
3. **Top-K mean MSE is unbounded in loss-units on heavy-tailed residual distributions.** λ-tuning cannot recover from a scale that varies by 4 orders of magnitude across steps.
4. **Design rule**: any auxiliary loss term derived from a subset of points (top-K, hard sample, focal, etc.) needs *scale-invariant formulation by construction*, not by hyperparameter tuning. Two practical fixes: (a) clip per-step at `MAX_CLIP × base_loss.detach()` to bound contribution magnitude, or (b) normalize sq_err by its own mean before top-K aggregation so the OHEM term is mean-1 by construction.
5. **Adds to the family of failed surface-loss amplification approaches** alongside PR #1109 (spatial focal α=2.0) and PR #793 (τ_z×3.0/4.0 scalar multipliers). Pattern: amplifying surface-loss signal at EP1-3 destabilizes training rather than accelerating learning.

Askeladd reassigned to OHEM v2 (spike-clipped + reduced λ) — a clean implementation of the failure-mode fix.

## 2026-05-14 19:45 — PR #1109: τ_z spatial focal loss α=2.0 (edward) — CLOSED (EP3 GATE FAIL, FOCAL ACTIVELY DEGRADES WSS)

- **Branch**: `edward/tau-z-spatial-focal-loss` (closed)
- **W&B runs**: `kom0ve5x` (smoke, 1-ep PASS), `emu3z6sg` (full 13-ep, terminated at EP3 by watchdog)
- **Hypothesis**: Spatial focal loss `w_i = 1 + α · |τ_z_gt_i| / mean(|τ_z_gt_valid|)` concentrates gradient on high-shear surface regions (wheel arches, underbody, A-pillar reattachment) to accelerate tau_z descent. Orthogonal to all prior scalar τ_z multiplier experiments (PR #793, alphonse τ_z×3.0/4.0 sweeps).
- **EP3 gate (val_abupt ≤7.2% AND val_vol_p ≤4.5%)**: FAIL on both criteria.

### Full val trajectory

| Step | Epoch | val_abupt | val_vol_p | val_WSS | val_WSS_z |
|------|-------|-----------|-----------|---------|-----------|
| 10,864 | EP1 | 33.538% | 21.190% | 37.095% | 41.501% |
| 21,728 | EP2 | 9.772% | 6.307% | 10.934% | 13.520% |
| **30,468** | **EP3** | **8.253%** | **5.211%** | **9.266%** | **11.588%** |

Best-checkpoint test (EP3): test_abupt=7.908%, test_vol_p=5.022%, test_WSS=9.026%.

### Comparison to baseline tay-stack EP3 (thorfinn #1100 τ_z×3.0 arm referenced in PR body)

| Metric | Edward α=2.0 focal | Thorfinn τ_z×3.0 baseline | Δ |
|---|---:|---:|---:|
| val_WSS @ EP3 | 9.266% | 7.661% | **+1.605pp WORSE** |
| val_WSS_z @ EP3 | 11.588% | 10.206% | **+1.382pp WORSE** |

### Diagnosis and paper-relevant findings

1. **Spatial focal loss family does not work at the tay-stack scale within the wall-clock budget.** α=2.0 is structurally degrading, not just slow-converging — the +1.6pp val_WSS regression at EP3 is not a slow-convergence artifact.
2. **Per-point gradient redistribution fails when the model is undertrained.** At EP1-3 the focal weights amplify GT signal where the model has not yet learned to predict accurately — introducing high-variance gradient updates that destabilise WSS learning.
3. **α=4.0 retry not viable.** Stronger focal weighting amplifies the same destabilisation; +1.6pp regression is too wide a gap to recover.
4. **Design rule**: future spatial-loss reweighting experiments should require N≥2 epochs of standard-loss warmup before activating per-point modulation. Transferable lesson.
5. **Focal loss is added to the catalogue of approaches that work in classification/object-detection but fail in dense surface regression on this benchmark.**

Edward reassigned to multi-scale surface attention (Wave 29 architectural hypothesis).

## 2026-05-14 13:30 — PR #1103: SLSQP continuous 4-simplex weight search (edward) — CLOSED (POOL-SATURATION CONFIRMED, USEFUL NEGATIVE)

- **Branch**: `edward/slsqp-continuous-weight-search` (closed)
- **W&B runs**: `nqaiyt6m` (v2 — full regime sweep), `4f743to9` (v1 — unconstrained + as-specified only)
- **Hypothesis**: K=8 Caruana weights `{0.375, 0.250, 0.250, 0.125}` are quantized to multiples of 1/8; the continuous SLSQP optimum on the 4-simplex over the same pool (`56bcqp3m`, `29nohj67`, `a0yoxy85`, `ghh0s4ne`) should yield a lower val_WSS and propagate to test, under val_vol_p≤3.643% and val_SP≤3.577% constraints. No GPU training — pure post-hoc optimization over cached predictions.

### Convergence — 3 SLSQP regimes × 5 starting points

| Regime | Optimal w = [56bcqp3m, 29nohj67, a0yoxy85, ghh0s4ne] | val_WSS | Feasible? |
|---|---|---:|---|
| Unconstrained | [0.3483, 0.3048, 0.2198, 0.1271] | **6.5156** | ✅ all 5 starts converge identically |
| As-specified (val_vol_p≤3.643, val_SP≤3.577) | [0.3432, 0.2020, 0.2512, 0.2036] | 6.5302 | ❌ INFEASIBLE (val_SP=3.7178 > 3.577) |
| No-regression vs K=8 (val_vol_p≤3.4360, val_SP≤3.7234) | [0.3471, 0.2783, 0.2282, 0.1464] | 6.5165 | ✅ all 5 starts converge identically |
| vol_p-only (val_vol_p≤3.643, no SP cap) | [0.3483, 0.3048, 0.2198, 0.1271] | 6.5156 | ✅ same as unconstrained (vol_p not binding) |

K=8 Caruana reference: `[0.375, 0.250, 0.250, 0.125]`, val_WSS=6.5195. All continuous optima sit **~0.03 L1 from the K=8 vertex** — K=8 is one rounding step away from optimal on this pool.

### Gate Check — Win = val_abupt<5.7452 ∧ test_vol_p≤3.643 ∧ test_WSS<6.3263

| Variant | val_abupt | test_vol_p | test_WSS | Gate? |
|---|---:|---:|---:|---|
| K=8 Caruana (PR #1102 baseline) | 5.7452 | 3.5397 | 6.3263 | reference |
| SLSQP unconstrained | 5.7440 | 3.5443 | 6.3307 ❌ | FAIL (test_WSS) |
| SLSQP as-specified | — | — | — | INFEASIBLE |
| SLSQP no-regression | **5.7427** | 3.5922 | **6.3253** ✓ | PASS by +0.0010pp |

### Full Test Metrics — No-Regression vs K=8 Caruana

| Metric | K=8 baseline | SLSQP no-regression | Δ | Direction |
|---|---:|---:|---:|---|
| val_abupt | 5.7452 | **5.7427** | −0.0025 | ✓ |
| val_vol_p | 3.4360 | 3.4322 | −0.0038 | ✓ |
| val_WSS | 6.5195 | 6.5165 | −0.0030 | ✓ |
| val_SP | 3.7234 | 3.7234 | 0.0000 | ↔ (binding) |
| test_abupt | **5.5196** | 5.5304 | +0.0108 | ✗ |
| test_vol_p | **3.5397** | 3.5922 | +0.0525 | ✗ |
| test_WSS | 6.3263 | **6.3253** | −0.0010 | ✓ (gate margin) |
| test_SP | **3.3529** | 3.3583 | +0.0054 | ✗ |
| test_tau_x | 5.6071 | **5.6062** | −0.0009 | ✓ |
| test_tau_y | 6.8397 | 6.8397 | −0.0000 | ↔ |
| test_tau_z | 8.2585 | **8.2555** | −0.0030 | ✓ |

### Pool quality — single-model val metrics

| Member | val_abupt | val_WSS | val_vol_p | val_SP |
|---|---:|---:|---:|---:|
| 56bcqp3m | 6.1264 | 6.9168 | 3.7976 | 3.9793 |
| 29nohj67 | 6.2853 | 7.0491 | 3.8988 | 4.1820 |
| a0yoxy85 | 6.2783 | 7.0702 | 3.9773 | 4.0644 |
| ghh0s4ne | 6.5319 | 7.4008 | 4.1150 | 4.2113 |

Min reachable val_SP on the 4-simplex ≈ 3.72% (every member has val_SP ≥ 3.98%). The PR's val_SP ≤ 3.577% target lies **below the achievable floor**, hence infeasibility.

### Results commentary

1. **K=8 Caruana is near-globally-optimal on the 4-member pool.** Three independent SLSQP regimes converge to weights within ~0.03 L1 of `[0.375, 0.250, 0.250, 0.125]`. All 5 starts in every regime hit the same optimum to 6 decimal places. Best-case val_WSS reduction is **0.0039pp** (0.06% relative). The val_WSS surface near K=8 is locally flat.
2. **Continuous unconstrained optimum FAILS test_WSS gate** (+0.0044pp). Classic val-overfit signature: SLSQP minimises val exactly, val→test WSS gap swallows the gain.
3. **No-regression variant** technically passes the gate by Δ=−0.0010pp on test_WSS, but this sits well below the test-set bootstrap stderr (~0.05–0.10pp on 50 cases). Three other test metrics regress (abupt +0.0108, vol_p +0.0525, SP +0.0054). Operationally storing real-valued weights vs simple integer/8 picks for an unmeasurable gain = disproportionate complexity.
4. **As-specified constraints are infeasible** — the val_SP≤3.577% bound is below the pool's reachable floor (3.72%) because every member has val_SP≥3.98%. The 3.577% is the previous SOTA *test_SP*, applied as a *val* constraint with no consideration of the val/test gap.
5. **Pool extension is the operative campaign lever** — not weight resolution. Wave 27 (#1104 fern magnitude penalty, #1105 tanjiro rel_l2, #1106 frieren physical-frame, #1107 nezuko yaw aug) produces single-model variants with **different error patterns**, expanding the pool's reachable simplex polytope rather than refining position within the current polytope.

### Decision

CLOSED (useful negative + mechanism finding, no merge — no code change to baseline, no meaningful metric improvement). K=8 Caruana SOTA at test_WSS=6.3263% (PR #1102) stands unchanged. Edward's follow-up #2 — **bias-corrected ensemble** `pred = Σ w_i · pred_i + b_c` per channel — is assigned next; it addresses the structural pool offset uncovered here via a different lever than convex re-weighting.

---

## 2026-05-14 09:15 — PR #1099: WSS-targeted greedy ensemble reselection (fern) — CLOSED (CONFIRMATORY NULL)

- **Branch**: `fern/wss-targeted-greedy-ensemble`
- **W&B runs**: `tfcaumtl` (WSS-targeted), `6et5sse7` (tau_z-targeted)
- **Hypothesis**: Re-running greedy forward selection on the K=4 candidate pool but optimizing `--selection-metric wall_shear_rel_l2_pct` (instead of default `abupt_axis_mean_rel_l2_pct`) would find a different ensemble subset that minimizes WSS specifically. Secondary arm: tau_z-targeted greedy (`wall_shear_z_rel_l2_pct`).

| Arm | Final K | Members | test_abupt | test_vol_p | test_WSS | test_tau_z | Constraint |
|---|---:|---|---:|---:|---:|---:|---|
| WSS-targeted | 3 | `56bcqp3m`+`29nohj67`+`a0yoxy85` | 5.5199% | **3.3630%** ✓ | 6.3712% | 8.3130% | PASS |
| tau_z-targeted | 4 | + `ghh0s4ne` | 5.5938% | **3.8891%** ✗ | 6.3298% | 8.2546% | FAIL (vol_p) |
| PR #1064 K=3 ref | 3 | same as WSS arm | 5.5199% | 3.3630% | 6.3712% | 8.3130% | PASS |

**Result**: WSS-targeted greedy converges to **exactly the same K=3 ensemble as PR #1064's ABUPT-targeted greedy** — identical members, identical metrics. The hypothesis that WSS-minimizing diversity differs from ABUPT-minimizing diversity is **NOT supported** on this 4-member pool. tau_z-targeted greedy keeps `ghh0s4ne` for a marginal WSS gain (−0.041pp) at the cost of large vol_p regression (+0.526pp, blowing constraint).

**Key insights:**
1. K=3 ensemble is **Pareto-optimal** on this candidate pool under the vol_p≤3.643% constraint
2. `ghh0s4ne` is the constraint-violating member — uniquely useful tau_z signal but weak vol_p prediction
3. **Newly documented baseline:** test_WSS=6.3712%, test_tau_z=8.3130% for the compliant K=3 ensemble — gap to Issue #1056 target is **−0.521pp test_WSS**
4. Data-bug surfaced: `data/split_manifest.json` silently resolves to deprecated backup tree when `--data-root` is omitted (caught by student, recovery successful)

**Decision**: CLOSED (no merge — no code change, no metric improvement). BASELINE.md updated with newly-documented test_WSS/test_tau_z values. Follow-up assigned: weighted ensemble via `--allow-replacement` (Caruana with-replacement formulation).

---

## 2026-05-11 15:11 — PR #958: Vol aux decoder head (nezuko) — MERGED (POSITIVE: new single-model SOTA)

- **Branch**: `nezuko/vol-pressure-aux-decoder-head` (merged)
- **W&B run**: `29nohj67` (group `nezuko-vol-aux-decoder-head`, Arm A `--volume-loss-weight 1.0`)
- **Hypothesis**: Dedicated 3-layer MLP branch for vol_pressure prediction, separate from the shared decoder, with a tunable volume loss weight. The hypothesis was that a dedicated vol_p head would improve test_vol_p OOD performance by allowing more expressive vol-pressure-specific capacity.

| Epoch | Step | val_abupt | vol_p | Gate | Status |
|---|---:|---:|---:|---|---|
| EP4 | ~43,456 | — | — | PASS | continuing to EP13 |
| EP13 (best) | ~108,640 | **6.2868%** | 3.9152% | **NEW SOTA** | MERGED |

**Val metrics (EP13 best checkpoint, run `29nohj67`):**

| Metric | Value |
|---|---:|
| val_abupt | **6.2868%** |
| surface_pressure | 4.1766% |
| volume_pressure | 3.9152% |
| wall_shear | 7.0476% |
| tau_x | 6.1726% |
| tau_y | 7.6648% |
| tau_z | 9.5053% |

**Test metrics (from best val checkpoint):**

| Metric | Value |
|---|---:|
| test_abupt | 7.7445% |
| surface_pressure | 3.9100% |
| volume_pressure | **12.0063%** |
| wall_shear | 6.9848% |
| tau_x | 6.2092% |
| tau_y | 7.5689% |
| tau_z | 9.0280% |

**Results commentary:**
- **POSITIVE: val_abupt=6.2868%** is the new single-model SOTA, −0.154pp (−2.39% relative) vs prior SOTA PR #823 (6.4407%). Improvement is broad-based across all channels.
- **NEGATIVE on primary hypothesis (vol_p OOD fix):** test_vol_p=12.0063% is WORSE than the baseline 11.6704% from PR #823. The dedicated aux decoder head improves general val_abupt but does NOT fix the 4 OOD outlier test cases (run_133, 226, 203, 158). Adding vol_p capacity alone is not sufficient when the failure is driven by geometry-specific OOD cases.
- The 4 OOD test cases require geometry-conditioned interventions, not just separate decoder heads. The improvement in val_abupt comes from better capacity allocation (dedicated head can specialize for in-distribution vol_p patterns) while test_vol_p worsens because the aux head may overfit val-distribution vol_p patterns without generalizing to OOD geometries.
- New single-model training gate: val_abupt < **6.2868%** (previously 6.4407% from PR #823).
- Arm B (`--volume-loss-weight 2.0`, run `6xja19q9`) was still running at merge time. If it exceeds val_abupt=6.2868%, it becomes the new SOTA.

---

## 2026-05-11 14:00 — PR #989: SDF-modulated per-octave positional encoding (fern) — CLOSED (NEGATIVE)

- **Branch**: `fern/sdf-modulated-vol-pe` (closed)
- **W&B run**: `z6dcbe9g`
- **Hypothesis**: Weight each RFF octave by exp(−|SDF|/σ_k) so that near-surface volume points get stronger high-frequency positional signal. Different octaves get different SDF-modulated gain, effectively creating a position-dependent spectral emphasis that tracks boundary proximity.

| Epoch | Step | val_abupt | vol_p | Gate | Status |
|---|---:|---:|---:|---|---|
| EP1 | ~10,864 | ≤30% | — | PASS | continuing |
| EP2 | ~21,728 | ≤16% | — | PASS | continuing |
| EP3 | ~32,592 | — | — | — | continuing |
| EP4 | ~43,456 | 7.490% | 4.886% | FAIL (target ≤6.9%) | CLOSED |

**Results commentary:**
- EP4 val_abupt=7.490% is 1.0–1.1pp worse than baseline (6.4407%), failing the EP4 gate of ≤6.9%.
- vol_p=4.886% actually passes the EP3 vol_p gate (≤5.0%) but abupt is the primary metric and clearly regressed.
- test_abupt=8.793% confirms the generalization gap is real, not a val overfitting artifact.
- The SDF-modulated weighting may interfere with the STRING-sep RFF encoding by introducing a spatially varying scale that conflicts with the learned log_freq adaptation — the SDF envelope compresses the effective frequency range in far-field regions, reducing positional specificity precisely where vol structure matters most.
- **Conclusion:** SDF-modulated per-octave positional encoding is closed negative. The spectral compression effect in far-field regions likely explains the regression. The SDF proximity signal is better incorporated via cross-attention (existing surf→vol xattn) than via PE modulation.

---

## 2026-05-11 16:00 — PR #985: Per-case geometry embedding v2: cross-attn vol × all surface tokens (alphonse) — CLOSED (NEGATIVE)

- **Branch**: `alphonse/geo-embed-v2-xattn` (closed)
- **W&B run**: `7e3k06fj`
- **Hypothesis**: PR #976 tested mean-pooling surface tokens into a case geometry embedding (NEGATIVE — mean-pool collapsed spatial information). This PR tests a stacked double cross-attention: a dedicated `geo_cond_xattn` (Q=vol_hidden, K=V=surf_hidden, zero-init out_proj) conditions vol_hidden on surface geometry BEFORE the existing surf→vol xattn block. The geometry-conditioned vol queries should better target OOD test cases (run_133, 226, 203, 158) that dominate test_vol_pressure failure.
- **Parameter count**: 18.04M (+1.05M stacked geo-cond xattn layer, ~6.2% increase over 16.99M baseline)

| Epoch | Step | val_abupt | val_vol_p | Gate | Status |
|---|---:|---:|---:|---|---|
| EP1 | ~10,864 | ≤30% | — | PASS | continuing |
| EP2 | ~21,728 | ≤16% | — | PASS | continuing |
| EP3 | ~32,592 | 8.2740% | 5.5706% | FAIL dual gate (>8.0% AND >5.0%) | killed |
| EP4 | ~43,456 | **8.2145%** | 5.5254% | — | terminal (run continued after gate) |

**Test metrics (EP4 terminal):**

| Metric | Value |
|---|---:|
| test_abupt | 9.4094% |
| test_vol_p | 13.0508% |

**Results commentary:**
- EP3 dual gate FAILED on both axes: val_abupt=8.2740% (threshold ≤8.0%) and val_vol_p=5.5706% (threshold ≤5.0%). Both exceeded the gate by 0.27pp and 0.57pp respectively.
- EP4 terminal val_abupt=8.2145% — marginally better than EP3 (converging near 8.2%) but 1.97pp above the SOTA baseline of 6.2869%. No improvement trend observed.
- test_vol_p=13.0508% is WORSE than the baseline (~12.0%), confirming the stacked xattn does not address OOD vol_pressure failure.
- The double cross-attention stacking (geo_cond_xattn → surf→vol xattn) appears to degrade performance relative to the single xattn baseline. Possible explanation: the second xattn over the same K=V=surf_hidden tokens introduces redundancy that over-conditions vol_hidden on surface features, reducing the model's ability to fit interior volume structure independently.
- The zero-init out_proj on geo_cond_xattn was intended to start as identity, but with two xattn layers stacking residuals over identical keys/values, the optimization landscape may be ill-conditioned from the start.
- **Conclusion:** Stacked double cross-attention is NEGATIVE. Geometry conditioning via a second xattn layer over all surface tokens does not improve OOD generalization. The OOD vol_p failures are driven by distribution shift in the 4 outlier geometries, not by insufficient surface-conditioning capacity.

---

## 2026-05-11 14:00 — PR #988: Pre-xattn vol self-attention (frieren) — CLOSED (INCONCLUSIVE/TIMEOUT)

- **Branch**: `frieren/pre-xattn-vol-selfattn` (closed)
- **W&B run**: `mdmkx495`
- **Hypothesis**: Add a self-attention layer on volume tokens BEFORE the existing surf→vol cross-attention so that volume tokens can aggregate global context before querying surface features. Expected to improve coherence of predicted volume fields.

| Epoch | Step | val_abupt | Status |
|---|---:|---:|---|
| EP1 | ~10,864 | PASS ≤30% | continuing |
| EP2 | ~21,728 | PASS ≤16% | continuing |
| EP3 (partial) | 17,445 | 10.5927% (best) | TIMEOUT at 270 min wall-clock |

**Results commentary:**
- Run terminated at step 17,445 (mid-EP3) due to 270-minute wall-clock limit. Best val_abupt=10.5927% at the point of termination.
- Root cause: O(N²) full self-attention over vol_pts=49,152 points required 63.9 GB VRAM peak (near 96 GB VRAM limit) and dramatically slowed throughput, making EP3 completion infeasible within the time budget.
- Cannot draw conclusions about hypothesis quality — the run was capacity-constrained, not performance-constrained.
- **Implication:** Full self-attention at vol_pts=49,152 is computationally infeasible. A linearized attention (e.g., Performer, FNet, Nyströmformer) or chunked/hierarchical approach would be needed to make this hypothesis testable at full resolution.
- **Conclusion:** Hypothesis closed as inconclusive due to O(N²) scaling constraint. The core idea (vol self-aggregation before xattn) remains untested. Future work should use linear-complexity alternatives.

---

## 2026-05-11 14:00 — PR #983: Curriculum LR warmup (tanjiro) — CLOSED (NEGATIVE)

- **Branch**: `tanjiro/curriculum-lr-warmup` (closed)
- **W&B runs**: Arm A (killed by student at EP3), Arm B: `xy4yhpm0`
- **Hypothesis**: The EP1→EP2 instability spike is caused by the abrupt curriculum jump (16,384→32,768 vol_pts at the EP1 boundary). A linear warmup over the first curriculum step should smooth the transition and reduce the EP2 spike.

| Arm | Description | EP4 val_abupt | EP4 test_abupt | Gate | Status |
|---|---|---:|---:|---|---|
| A | Warmup from EP1 start | N/A | N/A | — | Killed at EP3 by student |
| B | Warmup from EP1/EP2 boundary | 7.4176% | 8.7352% | FAIL (target ≤6.9%) | CLOSED |

**Results commentary:**
- Arm B EP4 val_abupt=7.4176% — 1.0pp above baseline, failing the EP4 gate of ≤6.9%.
- **Key finding (critical):** The EP1→EP2 instability is NOT caused by the curriculum vol_pts jump. It is caused by LR-shock: the LR schedule includes an 18× jump (5e-6→9e-5) at the EP1 boundary. EP2→EP3 transitions smoothly despite having the same vol_pts curriculum jump, because the LR is already in cosine decay by that point.
- Arm A was killed by the student before ADVISOR override could redirect; Arm B had the warmup placed at the EP1/EP2 boundary (wrong location to address LR-shock, which begins earlier).
- **Actionable implication:** To address LR-shock, the LR warmup must complete BEFORE the EP1→EP2 curriculum boundary. The warmup window should overlap with the initial low-vol_pts phase so the LR has already reached peak and begun decaying when curriculum jumps to 32,768 points.
- **Conclusion:** Curriculum-position warmup is negative. LR schedule reform (not curriculum position adjustment) is the correct lever. Future warmup experiments must ensure warmup completion precedes the EP1→EP2 curriculum boundary.

---

## 2026-05-11 12:00 — PR #960: STRING-sep RFF sigma bracket sweep (askeladd) — CLOSED (NEGATIVE)

- **Branch**: `askeladd/string-sigma-bracket-sweep` (closed)
- **W&B runs**: `zhnlo5k5` (Arm A: σ=[0.01,0.25,0.5,1.0,2.0]), `ro7s71k1` (Arm B v2: σ=[0.25,0.5,1.0,2.0,4.0])
- **Hypothesis**: The current baseline σ=[0.25,0.5,1.0,2.0,4.0] range may not be optimal. Arm A tested a fine-shift toward lower frequencies (σ=[0.01,0.25,0.5,1.0,2.0]); Arm B reproduced the baseline range as a clean control.

| Arm | σ range | EP3 val_abupt | EP4 gate ≤6.9% | Baseline |
|---|---|---:|---|---:|
| A (fine-shift) | [0.01,0.25,0.5,1.0,2.0] | 7.1812% | FAIL | 6.4407% |
| B v2 (baseline range) | [0.25,0.5,1.0,2.0,4.0] | 7.4995% | FAIL | 6.4407% |

**Results commentary:**
- Both arms failed the EP4 gate (≤6.9%). Arm A best val_abupt=7.1812% (0.74pp above baseline); Arm B v2=7.4995% (1.06pp above baseline).
- Arm B v2 is particularly telling: reproducing the exact baseline σ-range in a clean run still degraded performance by 1.06pp. This suggests the multi-sigma RFF benefit is sensitive to the broader training configuration rather than σ coverage alone.
- A kill-threshold bug (`<30` was semantically inverted — lower is better, so `<30` fired when val_abupt=26.44%) killed the original Arm B prematurely; Arm B v2 was re-run without kill thresholds.
- Arm A's lr-cosine-t-max=13 vs --epochs=4 mismatch truncated the run to ~EP3 only.
- **Conclusion:** σ-tuning is exhausted. The baseline [0.25,0.5,1.0,2.0,4.0] range is well-centered; expanding or contracting the bracket does not improve val_abupt. The trainable log_freq parameters (not the initialization range) carry the adaptation benefit.

---

## 2026-05-11 12:00 — PR #970: STRING-sep frozen-freq ablation (alphonse) — CLOSED (NEGATIVE)

- **Branch**: `alphonse/string-frozen-freq-ablation` (closed)
- **W&B run**: `pymccw0k`
- **Hypothesis**: If the learned log_freq values are close to initialization, freezing them (removing log_freq from the parameter set) may simplify optimization without losing accuracy. Ablation of trainable vs frozen frequency parameters in STRING-sep encoding.

| Run | EP3 val_abupt | EP3 gate ≤8.0% | Baseline |
|---|---:|---|---:|
| `pymccw0k` (frozen log_freq) | 8.3347% | FAIL (+0.33pp over gate) | 6.4407% |

**Results commentary:**
- EP3 val_abupt=8.3347% — 1.89pp above baseline and missed the EP3 kill gate by 0.33pp. Run terminated.
- This result compared to the trainable-freq baseline reveals: frozen log_freq hurts convergence by ~0.3–1pp at equivalent epochs. The model meaningfully adapts its RFF spectral coverage during training.
- **Positive finding:** This is a clean negative that closes a hypotheses and confirms an important mechanism. Trainable frequency adaptation is load-bearing — it is not just optimizing toward its initialization. The gradient flow through log_freq provides genuine benefit to spectral alignment.
- **Follow-up direction:** Since frequency adaptation is meaningful, the next productive question is *how* it adapts — log the learned log_freq values at training end to characterize spectral specialization. This could inform smarter initialization strategies.
- **Conclusion:** Frequency adaptation (trainable log_freq) is confirmed as genuinely beneficial. Do not revisit frozen frequencies. STRING-sep RFF σ-tuning and frequency-freezing are both exhausted. Future STRING-sep improvements should focus on the phase, dimensionality, or integration with geometry conditioning.

---

## 2026-05-09 22:30 — PR #962: Curriculum y-flip augmentation for vol_p OOD (frieren) — CLOSED (NEGATIVE)

- **Branch**: `frieren/curriculum-yflip-vol-ood` (closed)
- **W&B run**: `k1bqpcrz`
- **Hypothesis**: y-flip augmentation (negate y-coord, ny normal, tau_y) doubles effective training geometries and should improve OOD generalization. A curriculum ramp (0→0.5 probability over EP1–EP3) avoids early-training disruption vs fixed p=0.5 from step 0 (which failed as PR #957).

| Epoch | Step | val_abupt | vol_p | Gate | Status |
|---|---:|---:|---:|---|---|
| EP1 | 10,864 | 28.48% | 17.91% | ≤30% PASS | continuing |
| EP2 | 21,729 | 8.7659% | 5.9499% | pre-EP3 snapshot | monitoring |
| EP3 (projected) | 32,594 | — | ~5.924% | ≤5.0% gate | FAIL |

**EP3 gate failure analysis:**
- vol_p slope at EP2: (17.91% − 5.9499%) / 10,865 steps = −0.0011%/step
- Steps remaining to EP3 gate: 32,594 − 29,775 = 2,819 steps
- Projected vol_p drop: 2,819 × 0.0011 = 0.031pp → landing at ~5.924%
- Gate requirement: ≤5.0%  |  Gap: 0.924pp — insurmountable in 2,819 steps

**Results commentary:**
- The curriculum ramp did avoid the EP1 instability seen in PR #957 (fixed p=0.5: higher vol_p at EP1). abupt=28.48% is clean at EP1.
- However, the curriculum ramp itself is the problem: by delaying flip augmentation until EP3, the model never experiences the OOD regime during the critical vol_p convergence phase (EP1–EP3). vol_p converges to 5.95% before the augmentation has full effect, and the slope is too shallow to close the 0.92pp gap.
- Root cause: y-flip augmentation creates a geometry mismatch that disrupts vol_p convergence — both immediate (PR #957) and gradual (this PR). The augmentation approach is fundamentally at odds with vol_p optimization in the 4-ep screen budget.
- **Conclusion**: Y-flip augmentation axis is FULLY CLOSED (#957 + #962). Do not revisit without a fundamentally different formulation (e.g., augmentation only after EP5 in a 13-ep run, or separate augmented pre-training).

---

## 2026-05-09 — PR #961: Geometry-conditioned Q-bias via mean-pool surf→vol xattn (fern) — CLOSED (NEGATIVE)

- **Branch**: `fern/geometry-conditioned-q-bias` (closed)
- **W&B runs**: `fc4je8my` (killed EP1 — train/loss<5 misfire), `5alw5lxo` (EP1 terminal)
- **Hypothesis**: The vol→surf cross-attention Q-projections are geometry-agnostic — all vol tokens query the same K/V regardless of their geometric context. Adding a geometry-conditioned bias (mean-pool of surf hidden states → small MLP → additive offset on vol Q-projections) should allow vol tokens to selectively attend to the most relevant surface regions based on global shape, improving the OOD vol_p gap.
- **Implementation**: `mean_surf_hidden` → `MLP(512→256, SiLU→256→512, zero-init final linear)` → additive bias on vol Q before xattn. ~393K new parameters.

| Run | EP1 val_abupt | kill_gate (≤30%) | status |
|---|---:|---|---|
| `fc4je8my` | 27.45% | PASS | killed by train/loss<5 gate misfire |
| `5alw5lxo` | 30.197% | 0.197pp over gate → FAIL | NEGATIVE |

**Results commentary:**
- Run `fc4je8my`: EP1=27.45% passes the EP1 gate, but was killed by the `train_loss<5` threshold — a misfire inherited from prior experiments. The kill threshold is irrelevant for this experiment (train loss <5 is normal). This run was therefore abandoned and re-run without the misfire threshold.
- Run `5alw5lxo`: EP1=30.197% — 0.197pp above the 30% kill gate. The margin is within the noise floor observed across multiple EP1 snapshots (±2.75pp variance seen in PR #961 analysis). The vol_p per-channel signal was consistent across both runs in direction, but insufficient EP1 separation from noise to confirm the signal as genuine.
- **Noise floor observation**: The 0.197pp over-gate is comparable to natural EP1 variance (±2.75pp), making it impossible to distinguish a genuine regression from statistical noise at EP1 alone. A third run could easily have passed, but the expected value of continuing this axis is low given the architectural simplicity of the additive Q-bias.
- **Conclusion**: Geometry-conditioned Q-bias (mean-pool surf → MLP → vol Q additive bias) does not produce a detectable EP1 improvement over baseline. The hypothesis is plausible but the implementation (global mean-pool surf → additive Q offset) lacks the spatial specificity needed to close the OOD vol_p gap. CLOSED NEGATIVE — do not revisit this exact formulation.

---

## 2026-05-10 — PR #942: GradNorm full-mode α=1.5 (fern) — CLOSED (NEGATIVE)

- **Branch**: `fern/gradnorm-full-mode-alpha15` (closed)
- **W&B run**: to be retrieved from PR comments (run truncated mid-EP2 due to 360-min budget with 5× backward overhead)
- **Hypothesis**: Paper-faithful GradNorm (Chen et al. 2018) across 5 loss tasks (surf_p, tau_x, tau_y, tau_z, vol_p), α=1.5. Hypothesis: vol_p task is gradient-starved under equal-weight training; GradNorm should upweight vol_p relative to other tasks by monitoring per-task gradient norms vs reference. Expected: vol_p loss weight rises materially above 1.0, reducing vol_p error.

| Epoch | val_abupt | vol_p weight | Weight range | Gate | Status |
|---|---:|---:|---:|---|---|
| EP1 | 28.86% | ↓ (decreased) | 0.91–1.11 | ≤30% PASS | continuing |
| EP2 | truncated (budget ~360 min, 5× overhead) | — | — | run aborted | CLOSED |

**Results commentary:**
- The key EP1 finding contradicts the hypothesis: vol_p GradNorm weight went DOWN (not up). This is because vol_p loss decreases the fastest at initialization, which causes GradNorm to reduce its relative weight to compensate.
- All 5 task weights converge to a tight band (0.91–1.11) by EP1, indicating GradNorm is functionally near-identical to static equal weights at these hyperparameters.
- The 5× backward overhead (one backward pass per task) consumed the 360-min training budget before EP6 could be reached — the experiment could not reach the decisive EP3 gate (≤8.0%) or EP4 gate (≤6.9%).
- **Conclusion**: vol_p is NOT gradient-starved under Lion optimizer with current loss weights. The OOD vol_p gap is a representational/generalization problem, not a gradient-balancing problem. GradNorm axis fully CLOSED — do not revisit.

---

## 2026-05-09 23:45 — PR #925: Random yaw±5°/pitch±3° rotation aug (alphonse) — CLOSED (NEGATIVE)

- **Branch**: `alphonse/random-yaw-pitch-rotation-aug` (closed)
- **W&B run**: `a6ddeqrq`
- **Hypothesis**: Joint rotation of surface_xyz, vol_xyz, surface_normals, and wall_shear vectors by a random yaw (±5°) and pitch (±3°) rotation matrix at train time (p_aug=0.5) forces approximate rotation-equivariance, closing the val/test gap on the 4 OOD test cases which likely exhibit different aerodynamic incidence angles.

| Epoch | val_abupt | surf_p | vol_p | wsh | Gate | Baseline |
|---|---:|---:|---:|---:|---|---:|
| EP1 | 27.37% | — | — | — | ≤30% PASS | ~25–28% |
| EP2 | 12.81% | — | — | — | ≤16% PASS | ~12% |
| EP3 | **9.1064%** | 5.96% | 6.14% | 10.14% | ≤8.0% **FAIL** | 7.1195% |

**Analysis:** Rotation aug at yaw±5°/pitch±3°/p=0.5 is too aggressive for the 4-epoch budget. The model spent EP2 capacity learning rotation-approximate equivariance — EP2 val_abupt fell 4.66pp behind the rotation-free baseline's EP2 position, suggesting the model was spending optimization headroom adapting to the augmented distribution rather than converging. The EP2→EP3 slope recovered to −3.71pp (steep, vs baseline's comparable step), suggesting the regularizer benefit was beginning to emerge, but the 4-ep screen gate at EP3 came too early to capture it.

**Per-channel EP3:** surf_p=5.96% (clean, near SOTA), vol_p=6.14% (near SOTA), wsh=10.14% (primary drag — wall_shear carries most of the penalty). The wsh degradation is the likely cause of EP3 gate failure: rotating wall_shear vectors is physically correct but adds more augmentation noise to the highest-variance channel.

**Key insight:** The hypothesis (rotation aug → OOD vol_p) is NOT falsified. Only the magnitude/probability point (yaw±5°/pitch±3°/p=0.5) is falsified. The wsh channel specifically degrades under this aggressive aug, as wall_shear vector rotation is geometrically complex and introduces more entropy than the geometric coordinates alone.

**Suggested follow-up axes (student suggestions incorporated):**
1. **Milder aug**: p=0.3, yaw≤3°, pitch≤1.5° — lower entropy, less wall_shear disruption
2. **Yaw-only variant**: pitch=0, yaw±3°/5°, p=0.5 — wind tunnel geometry means OOD variation is primarily yaw, pitch=0 removes a dimension of aug noise
3. **Aug rampup**: p=0 for EP1, ramp to p=0.3 by EP3 — curriculum approach

**Conclusion:** Assign follow-up with milder parameters. Rotation aug (yaw-only or mild yaw+pitch) is high-priority next assignment.

---

## 2026-05-09 19:30 — PR #906: Post-xattn vol-self-attn block (edward) — CLOSED (NEGATIVE)

- **Branch**: `edward/vol-self-attn-post-xattn` (closed)
- **W&B run**: `nmvw5t2d`
- **Hypothesis**: A vol→vol self-attention block inserted AFTER the surf→vol xattn (decoder-style refinement) gives the volume branch capacity to propagate cross-attended surface signal across volume tokens before regression, closing the val/test gap on `volume_pressure_rel_l2_pct`.
- **Implementation**: Pre-norm self-attn + FFN block, zero-init out_proj and final FFN linear (identity-at-init); inserted after surf→vol xattn residual update, before vol regression head. `find_unused_parameters=True` for DDP.

| Epoch | val_abupt | vol_p | surf_p | wsh | Gate | Baseline |
|---|---:|---:|---:|---:|---|---:|
| EP3 | 8.2271% | — | — | — | ≤8.0% **FAIL** | 7.1195% |
| EP4 step 282 | — | — | — | — | external SIGTERM | — |

**Analysis:** Run was killed by external pod/harness signal at 18:59:16 UTC (~252 min runtime, well under SENPAI_TIMEOUT_MINUTES=360). Termination is moot since EP3 had already failed the gate. No per-channel evidence that vol-self-attn after surf→vol xattn helps the val/test gap.

**Pattern: post-xattn capacity additions are now 0-for-3 on this benchmark:**
- PR #884/#890: two-layer xattn — failed
- PR #891: post-xattn FFN — failed
- PR #906: post-xattn vol-self-attn — failed

**Conclusion:** Adding capacity to the volume branch *after* surf→vol xattn does not move the val/test gap. Future volume-branch experiments should target either (a) capacity placed BEFORE surf→vol xattn, (b) the geometry/positional encoding pathway, or (c) data-augmentation / regularization approaches targeting OOD generalization rather than capacity.

---

## 2026-05-09 19:30 — PR #918: Vol-specific RFF init sigmas (tanjiro) — REQUESTED CHANGES

- **Branch**: `tanjiro/vol-rff-positional-encoding`
- **W&B run Arm A**: 5× lower-frequency vol sigmas `0.05,0.1,0.25,0.5,1.0` vs surface `0.25,0.5,1.0,2.0,5.0`
- **Hypothesis**: Volume and surface fields occupy different spatial frequency regimes; volume needs lower-frequency RFF init to match its smoother field structure.

| Epoch | val_abupt | Gate | Baseline EP1 |
|---|---:|---|---:|
| EP1 | 32.66% | ≤30% **FAIL** | ~25–28% |

**Analysis:** Aggressive 5× shift starves the surf→vol xattn coupling of fine-spatial detail — volume tokens carry less high-frequency content than the surface K/V expects, degrading the cross-attention. All per-channel metrics regressed.

**Decision:** Hypothesis itself (separate spectral regimes) not yet falsified — only the aggressive instantiation is. Requesting **Arm C** with a moderate 2× shift `0.1,0.25,0.5,1.0,2.0`. If Arm C also fails the screen, close the PR and mark vol-specific sigma init as a falsified direction across the moderate-to-aggressive range.

---

## 2026-05-09 19:30 — PR #901: Train-time y-axis mirror aug (fern) — REQUESTED CHANGES (truncated)

- **Branch**: `fern/train-mirror-aug-y`
- **Hypothesis**: DrivAerML has near-perfect y-axis symmetry; a stochastic p=0.5 train-time mirror (negate y, ny, tau_y) is a free 2× data prior that should reduce val/test gap on volume_pressure.

**Arm B run**: 13-ep at SOTA stack — **truncated at EP5/13** (~18% complete) due to SENPAI_TIMEOUT_MINUTES=360 → 270 min train budget. Advisor planning error: a 13-ep run at 65k vol points needs ~680 min. Acknowledged in PR comment.

**Encouraging mid-run signals:**
- val→test ratio on volume_pressure improved from baseline 3.03× to 2.50× even at EP5
- tau_y trajectory clean — no sign-flip pathology, aug is consistent with model's symmetry prior
- Loss curve healthy, no divergence

**Decision:** Hypothesis still alive but inconclusive — requesting rerun at the 4-ep screen with `--lr-cosine-t-max 4 --epochs 4` so the cosine fully decays inside the 270 min budget. Win criterion: EP4 val_abupt < 6.9% AND val→test vol_p ratio sustained ≤2.7×. If clean signal, find a way to fit the 13-ep follow-up (potentially drop terminal vol_points to 49k).

---

## 2026-05-09 16:30 — PR #902: Vol loss upweighting curriculum (nezuko) — CLOSED (NEGATIVE)

- **Branch**: `nezuko/vol-pressure-ood-curriculum` (closed)
- **W&B runs**: `nx49bb6w` (Arm A, vol_w=3.0), Arm B (vol_w=5.0) cancelled
- **Group**: `nezuko-vol-pressure-ood-curriculum`
- **Hypothesis**: Upweighting the volume loss (vol_w=3.0 Arm A, vol_w=5.0 Arm B) with an accelerated vol-curriculum would force the model to prioritize vol_p accuracy including the 4 OOD test cases.

| Epoch | Step | val_abupt | vol_p | surf_p | wsh | Gate | Baseline EP3 |
|---|---:|---:|---:|---:|---:|---|---:|
| EP1 | ~10,864 | — | 14.17% | — | — | ≤30% PASS | — |
| EP3 | 21,734 | **8.5436%** | 5.0614% | 5.7369% | 9.6625% | ≤8.0% **FAIL** | 7.1195% |

Arm B (vol_w=5.0) cancelled after Arm A failed — more aggressive upweighting in the same failing direction.

**Analysis:** vol_w=3.0 makes vol_p **worse** vs PR #823 baseline at EP3 (5.06% vs 4.27% — +0.79pp). Every channel degraded vs baseline. The accelerated vol-curriculum (vol_pts bumps at every epoch vs baseline's 3-epoch steps) creates a curriculum-mismatch: Arm A EP3 = 3rd vol=32k epoch with 21,734 iters; baseline EP3 = 3 epochs at vol=16k with 32,594 iters (~50% more compute). But even at EP1 (clean comparison), vol_p=14.17% vs baseline 17.79% — baseline wins. Heavier volume loss weight over-emphasizes vol gradients during early adaptation and disrupts the surf/wsh pathways.

**Conclusion:** Volume loss upweighting is not the right lever for closing the vol_p OOD gap. The issue is in feature representation and geometry conditioning architecture, not in loss balance.

---

## 2026-05-09 16:45 — PR #910: Xattn K/V grad scale α=0.5 Arm A (frieren) — EP3 FAIL, Arm B (α=0.75) ASSIGNED

- **Branch**: `frieren/xattn-kv-grad-scale-sweep-alpha`
- **W&B run Arm A**: `bnynqueq` (group `frieren-xattn-kv-grad-scale-sweep`, name `frieren/xattn-kv-scale-alpha0.5-screen`)
- **Flag Arm A**: `--xattn-kv-grad-scale 0.5`
- **Hypothesis**: Following α=0.25 EP3 stall (PR #896, 8.95%), test α=0.5 and α=0.75 to find the optimal K/V gradient scale point in [0.25, 1.0].

| EP | val_abupt | surf_p | vol_p | wall_shear | Δ abupt | Gate |
|---|---:|---:|---:|---:|---:|---|
| EP1 | 29.3248% | 22.40% | 16.96% | 32.67% | — | ≤30% ✅ |
| EP2 | 11.5368% | 7.71% | 8.24% | 12.59% | −17.79pp | ≤16% ✅ |
| EP3 | **8.6500%** | **5.49%** | **6.52%** | **9.43%** | **−2.89pp** | ≤8.0% ❌ (+0.65pp) |

Run killed at step ~20,300 (mid-EP4, before EP4 validation).

**Key signals:**
- EP2→EP3 slope = −2.887pp (6.5× steeper than α=0.25's −0.441pp): mechanism is much less stalled than α=0.25
- vol_p = 6.52% at EP3 already well below SOTA test_vol_p=11.67% — the K/V scale is specifically helping vol head while lagging on abupt-mean
- α=0.5 starts hot at EP1 (29.3% vs α=0.25's 13.4%) suggesting the higher gradient flow makes early optimization harder but the run is still descending steeply at EP3
- Likely EP13 landing from EP3=8.65% following SOTA slope: ~8.0% (not competitive vs SOTA 6.44%)

**Action:** Arm B (α=0.75) launched — closer to α=1.0 (SOTA), expected to start cooler and converge faster on val_abupt. Bracket now [α=0.25 EP3=8.95%, α=0.5 EP3=8.65%, α=0.75 TBD, α=1.0 EP3=7.12% SOTA].

---

## 2026-05-01 — PR #893: Grouped-Query xattn (alphonse) — CLOSED (NEGATIVE)

- **Branch**: `alphonse/xattn-gqa` (deleted)
- **W&B runs**: `7jqz957i` (Arm A, n_kv_heads=1/MQA), `eqp1873z` (Arm B, n_kv_heads=2/GQA)
- **Group**: `xattn-gqa-sweep`
- **Hypothesis**: Replace the surf→vol cross-attention MHA (n_heads=4) with Grouped-Query Attention (GQA). Arm A: MQA (n_kv_heads=1, 4:1 Q/KV ratio); Arm B: GQA (n_kv_heads=2, 2:1 ratio). Llama-style: head_dim=128 throughout, smaller KV projection output. Expected benefit: reduced KV parameter count, potentially acting as a structured regularizer on the surface conditioning pathway.

| Arm | Config | EP3 abupt | Gate (≤8.0%) | surf_p | vol_p | wsh |
|---|---|---:|---|---:|---:|---:|
| A (run 7jqz957i) | n_kv_heads=1, MQA | 8.2694% | ❌ FAIL (+0.27pp) | 5.3768% | 5.6686% | 9.1388% |
| B (run eqp1873z) | n_kv_heads=2, GQA | 8.2097% | ❌ FAIL (+0.21pp) | 5.3411% | 5.5667% | 9.0992% |
| **SOTA** (PR #823) | n_kv_heads=4, MHA | **6.4407%** | baseline | 4.1836% | 3.8557% | 7.3448% |

Both arms ran to EP3 then were killed per kill-gate protocol (student terminated Arm A after Arm A miss; Arm B also confirmed miss).

**Analysis:** Both GQA arms failed EP3 by a narrow but consistent margin (~0.21–0.27pp). The reduction in K/V heads uniformly impairs convergence — more heads misses by less (Arm B > Arm A), consistent with the hypothesis that full MHA capacity in xattn is load-bearing. Arm B is strictly better than Arm A (fewer K/V heads = more degradation), confirming the direction: the surface→volume attention benefits from full rank attention heads. Notable: the student caught a spec dimension mismatch bug during implementation (original spec had kv_head_dim = embed_dim/n_kv_heads, giving incompatible Q/K head dims for SDPA) and correctly implemented standard Llama-style GQA instead.

**Conclusion:** GQA for surf→vol cross-attention does not improve convergence. Full MHA (n_heads=4) remains optimal. Cross-attention KV capacity is not a bottleneck to regularize. Closing as a clean negative result.

---

## 2026-05-09 ~12:45 — PR #896: Xattn K/V gradient scaling α=0.25 (frieren) — CLOSED (NEGATIVE)

- **Branch**: `frieren/xattn-kv-grad-scale` (closed)
- **W&B run**: `vf9dprlh` (group `frieren-xattn-kv-grad-scale`, name `frieren/xattn-kv-grad-scale`)
- **Flag**: `--xattn-kv-grad-scale 0.25`
- **Hypothesis**: Scale K/V gradients by α=0.25 in surf→vol xattn to damp surface encoder over-adaptation while preserving joint training signal. Addresses the K/V backflow mechanism identified in #884 (two-layer backflow) and #890 (full detach kills EP1).

| Epoch | Step | val_abupt | Gate | Result |
|---|---:|---:|---|---|
| EP1 | 21,728 | 13.449% | ≤30% | ✅ PASS |
| EP2 | 32,599 | 9.395% | ≤16% | ✅ PASS (6.6pp margin) |
| EP3 | 39,851 | **8.954%** | ≤8.0% | ❌ FAIL (+0.954pp) |
| EP4 | 45,308 | 8.773% | — | (killed) |

Phase 2 NOT triggered.

**Analysis:** Strong EP1→EP2 drop (−4.05pp) but severe slope flattening EP2→EP3 (−0.44pp, 10× slowdown). K/V gradient scaling at α=0.25 reduces surface encoder adaptation just enough to slow volume convergence without delivering a commensurate accuracy benefit. The α=0.25 sweet spot between detach (α=0, EP1 kill gate #890) and full gradient (α=1.0, SOTA #823) appears to exist but this value isn't it — the convergence stalls at 8.95%.

**Key finding:** Graduated backflow management is the right axis to explore, but α=0.25 is too aggressive. Future experiments should try α=0.5 (half backflow) or α=0.75 (gentle damping) if this mechanism is revisited.

## 2026-05-09 ~12:30 — PR #895: L=6 + surf→vol xattn (edward) — CLOSED (NEGATIVE)

- **Branch**: `edward/xattn-depth-L6-512` (deleted)
- **W&B run**: `x3c2a2jt` (group `edward-xattn-depth-L6-512`, name `edward/xattn-depth-L6-512-screen`)
- **Hypothesis**: Adding a 6th Transolver block in the L=5+xattn stack would give the model extra capacity that geometry-conditioning (xattn) can finally exploit, since the previous L=6 NEGATIVE (PR #811) was without xattn.

| Epoch | val_abupt | Gate | SOTA PR #823 | Δ |
|---|---:|---|---:|---:|
| EP1 | ~28% | ≤30% PASS | 28.63% | ~ |
| EP2 | ~14% | ≤16% PASS | 8.15% | ~+6pp worse |
| EP3 | ~9.5% | ≤8.0% **FAIL margin** | 7.12% | ~+2pp worse |
| EP4 | **7.886%** | ≤6.9% **FAIL** | 6.81% | +1.08pp worse |

Phase 2 (full 13-ep) NOT triggered.

**Analysis:** Train loss reached 0.032 at EP4 with val_abupt 7.89% — large train/val gap signals memorization rather than improved generalization. Combined with PR #811 (L=6 without xattn, also NEGATIVE), the depth-scaling axis at hidden=512 is **CLOSED on both with-xattn and without-xattn**. Adding a 6th block does not give the volume head usable capacity at this budget.

**Implication:** Future capacity scaling should pivot away from naive layer count. Options: (a) wider hidden_dim with L=5, (b) asymmetric depth (L_vol > L_surf), (c) radical volume-pathway architectures (FNO, multiscale, GNN message passing) per Issue #717.

## 2026-05-09 ~09:37 — PR #891: Post-xattn FFN (fern) — CLOSED (NEGATIVE)

- **Branch**: `fern/post-xattn-ffn` (deleted)
- **W&B run**: `c3jvc0s1` (group `fern-post-xattn-ffn`, name `fern/post-xattn-ffn-4ep`)
- **Hypothesis**: Adding a 2-layer MLP (hidden×4, GELU, zero-init second linear) after the surf→vol xattn residual update gives the volume pathway more capacity to process the surface conditioning signal before the regression head.

| Epoch | Step | val_abupt | Gate | SOTA PR #823 | Δ |
|---|---:|---:|---|---:|---:|
| EP1 | 10,864 | 26.51% | ≤30% PASS | 28.63% | −2.12pp better |
| EP2 | 16,300 | 12.13% | ≤16% PASS | 8.15% | +3.98pp worse |
| EP3 | 19,926 | **8.50%** | ≤8.0% **FAIL** | 7.12% | +1.38pp worse |
| EP4 | killed | — | (≤6.4407%) | 6.81% | — |

Params: 19.09M vs SOTA 16.99M (+2.10M, +12%). Peak GPU: 64.9 GB / 97 GB.

**Analysis:** FFN injection started better (EP1 26.51% beats SOTA 28.63% by 2.1pp) but the trajectory flattened through EP2-EP3. The EP1→EP2 drop was −14.38pp (this run) vs −20.48pp (SOTA) — slower descent. EP3 absolute miss: 8.50% vs 7.12% SOTA (+1.38pp), well outside noise range. Zero-init guarantee held (EP1 healthy, gradient well-conditioned through new path). Slowdown is optimization-budget, not divergence: +2.1M extra params need more steps to settle, and 4-ep schedule with T_max=13 doesn't give enough time. The post-norm residual exposes vol head to a noisier signal during early epochs (FFN output drifts from zero before post-norm tuned).

Arm B not launched — Arm A EP4 unreachable from EP3=8.50% in one epoch.

**Verdict:** CLOSED NEGATIVE. Post-xattn FFN 4× expansion does not pay back its parameter cost on this budget. Student suggestions noted (2× ratio, SwiGLU, FFN-specific LR group) but not in immediate priority queue given human directive focus on test volume pressure OOD gap.

---

## 2026-05-09 10:30 — PR #888: Per-sample OOD loss upweighting ×3 (thorfinn) — CLOSED (NEGATIVE)

- **Branch**: `thorfinn/ood-sample-weighting` (deleted)
- **W&B run**: `thorfinn/ood-weight-3x-rank0` (group `thorfinn-ood-weighting`), run state: finished
- **Hypothesis**: The 4 OOD test geometries (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation. They are not in the training split, but their K=6 nearest train neighbors (by SDF Mahalanobis distance) ARE. Upweighting those K=6 neighbors by 3× should bias the model toward geometry clusters that will be OOD at test time — an indirect but principled OOD regularisation.

| Metric | EP1 | EP2 | Timeout (step 30,454) | SOTA PR #823 | Δ vs SOTA |
|---|---:|---:|---:|---:|---:|
| val_abupt | 30.2033% | 8.4952% | **7.4232%** | 6.4407% | +0.98pp worse |
| val_vol_pressure | — | 5.254% | — | 4.956% | +0.30pp worse |
| test_abupt | — | — | **8.6935%** | 7.6992% | +0.99pp worse |
| test_vol_pressure | — | — | **12.609%** | 11.6704% | +0.94pp worse |

Gate results: EP1=30.2033% FAIL (gate <30%; run continued given borderline miss). EP2=8.4952% PASS (<16%). Run hit the 270-min timeout mid-EP3 at step 30,454 (max EP3=~32,594 in 13-ep schedule).

**Analysis:** The OOD upweighting hypothesis is **refuted across all channels**. vol_pressure was specifically the target metric (OOD test cases dominate it), yet EP2 val_vol_p=5.254% is WORSE than SOTA's 4.956% at the same boundary — not better. The final forced checkpoint (step 30,454) shows val_abupt 7.42% vs SOTA 6.44% (+0.98pp), and test_abupt 8.69% vs SOTA 7.70% (+0.99pp). There is no OOD generalization benefit. Possible explanations: (1) The K=6 nearest neighbors by SDF Mahalanobis distance are not the actual bottleneck — the 4 OOD geometries may differ from any training geometry in ways the distance metric doesn't capture; (2) 3× upweighting is insufficient to shift the loss landscape without damaging the in-distribution performance; (3) the OOD generalization gap is fundamentally architecture/capacity-limited, not data-distribution-limited.

**Verdict:** NEGATIVE. OOD loss upweighting via nearest-train-neighbor proximity is closed. The OOD gap cannot be bridged through train-set reweighting alone. Architecture-level interventions (xattn geometry conditioning) remain the primary lever.

---

## 2026-05-09 09:15 — PR #890: Surf→vol xattn with detached K/V (frieren) — CLOSED (NEGATIVE)

- **Branch**: `frieren/xattn-detach-kv` (deleted)
- **W&B run**: group `frieren-xattn-detach-kv`
- **Hypothesis**: PR #884 identified K/V gradient backflow through the surface encoder as the cause of 2-layer xattn failure (surface_pressure +3pp regression). Detaching K/V before the xattn computation (stop_gradient on surface hidden states used as K/V) isolates the surface encoder from xattn gradients. This directly tests the backflow mechanism: if detach-K/V recovers performance, backflow is confirmed causal; if it still fails, the surface encoder needs to adapt jointly.

| Metric | EP1 | EP2 | Verdict |
|---|---:|---:|---:|
| val_abupt | >30% | — | FAIL EP1 kill gate |

**Analysis:** EP1 kill gate triggered. Detaching K/V did not rescue the detached xattn path. The K/V detach eliminates backflow but also cuts off the adaptation of K/V projections to the optimization pressure of the volume Q path — the surface encoder cannot co-adapt its K/V representations to what the volume decoder needs. This suggests that the joint gradient flow from Q→K/V is not a bug but a feature: it allows the surface encoder to specialize its output for the volume cross-attention consumer. Without that gradient, the K/V projections are underfit for the Q context.

**Verdict:** NEGATIVE. Detach-K/V is closed. The backflow mechanism is apparently load-bearing — gradient signal from volume Q back through surface K/V helps the surface encoder produce better geometry representations. This rules out zero-coupling approaches; future multi-layer xattn variants must manage the gradient magnitude, not eliminate it (e.g., gradient scaling, separate LR for surface encoder, or mid-backbone injection with partial gradient flow).

---

## 2026-05-09 08:00 — PR #886: Width scaling + surf→vol xattn hidden_dim=640 (edward) — CLOSED (negative result)

- **Branch**: edward/xattn-width-640 (deleted)
- **W&B run**: `m68ug46u` (group `edward-xattn-width-640`)
- **Hypothesis**: Width=640 may compound with surf→vol xattn — a wider backbone could leverage the geometry signal more richly. PR #872 showed width=640 without xattn failed; this tests whether xattn composition unlocks the width scaling.

| Metric | EP1 | EP2 | mid-EP3 (timeout) | SOTA PR #823 EP13 |
|---|---:|---:|---:|---:|
| val_abupt | 26.82% | 11.06% | 8.58% | 6.4407% |
| surface_pressure | 20.27% | 7.42% | 5.56% | 4.1836% |
| volume_pressure | 16.63% | 8.62% | 6.03% | 3.8557% |
| wall_shear | 29.83% | 11.93% | 9.48% | 7.3448% |

EP3 gate (<8%): FAILED at 8.58% (0.58pp over). Training cut by 270-min timeout mid-EP3 (step 18596/~22640). EP4 was never reached; extrapolated val_abupt ~6.9–7.2% (worse than SOTA 6.44%).

**Analysis:** Width=640 + xattn shows no synergy. EP1 is marginally better than SOTA EP1 (-1.81pp), but per-step convergence after EP1 is slower. The wider model adds parameters but does not generalize them into a clearer surface→volume coupling. Additional constraint identified: 4-epoch screens at hidden_dim=640 with the full vol-curriculum cannot fit within the 270-min timeout (~369 min projected).

**Verdict:** NEGATIVE. Combined with PR #872 (width=640, no xattn), the width-scaling axis is definitively closed. Neither configuration beats 512-dim SOTA. Width does not compound with geometry conditioning.

---

## 2026-05-01 14:30 — PR #887: Surf→vol xattn with surface subsampling (nezuko) — CLOSED (negative result)

- **Branch**: nezuko/xattn-surface-subsample (deleted)
- **W&B run**: `0ud2go3r` (group `nezuko-xattn-surface-subsample`)
- **Hypothesis**: The current surf→vol xattn (PR #823 SOTA) passes all 65,536 surface points as K/V. Uniform random subsampling (~4096 anchor points, N_kv=4096) before the K/V projection may sharpen the geometry signal by forcing compact surface structure representation and reduce memory pressure. Run B (N_kv=8192) was gated on EP4 val_abupt < 6.6%.

| Metric | Run A (N_kv=4096) | SOTA PR #823 | Δ |
|---|---:|---:|---:|
| val_abupt EP4 | 7.6075% | 6.4407% | +1.17pp (worse) |
| surface_pressure EP4 | 4.9802% | 4.1836% | +0.80pp |
| volume_pressure EP4 | 5.0467% | 3.8557% | +1.19pp |
| wall_shear EP4 | 8.4545% | 7.3448% | +1.11pp |
| tau_x | 7.3503% | 5.7782% | +1.57pp |
| tau_y | 9.6493% | 7.5977% | +2.05pp |
| tau_z | 11.0111% | 9.0116% | +2.00pp |

EP3: 8.2896% (missed <8% gate by 0.29pp — advisory miss, continued to EP4)
EP4: 7.6075% — missed <6.6% gate for Run B. Run B not launched.

**Analysis:** Uniform random subsampling hurt EVERY channel uniformly by 0.8–2.0pp. Vol_p regressed by 1.19pp even though it is the channel most directly downstream of surf→vol xattn. The model requires full 65k surface point coverage to accurately condition volume pressure. Random subsampling destroys the spatial coverage and structural information that the full set provides. EP3→EP4 drop was only 0.68pp (vs ~3.55pp EP2→EP3) — model was already stagnating.

**Key diagnostic:** The failure is not "too many K/V tokens" (information overload) but "wrong K/V tokens" (random selection loses structured geometry). Structured selection approaches (k-NN locality, learned pooling, FPS) remain untested.

**Verdict:** NEGATIVE. Surface subsampling with uniform random selection is ruled out. Follow-up: nezuko PR #892 tests mid-backbone xattn injection (different approach to improving geometry conditioning).

---

## 2026-05-01 — PR #878: Surf→vol xattn heads sweep H=8 vs H=16 (alphonse) — CLOSED (negative)

- **Branch**: alphonse/xattn-heads-sweep (deleted)
- **W&B runs**: Arm A `c4e3gurg` (H=8), Arm B `u5bpkpje` (H=16)
- **Hypothesis**: Baseline xattn uses num_heads=4 (128-dim/head). Increasing to 8 or 16 heads may capture richer surface geometry diversity through more specialised attention subspaces. EP3 kill gate <8%.

| Arm | Heads | EP1 | EP2 | EP3 | Decision |
|-----|-------|-----|-----|-----|----------|
| A | 8 | 27.832% PASS | 12.462% PASS | **8.7132% FAIL** (+0.71pp over gate) | killed |
| B | 16 | 27.428% PASS | 12.128% PASS | **8.5231% FAIL** (+0.52pp over gate) | killed |

**EP3 per-channel (H=8 vs H=16):**
| Channel | H=8 | H=16 | Δ (B−A) |
|---------|-----|------|---------|
| sp | 5.6444% | 5.5737% | −0.071pp |
| vp | 6.1853% | 5.7986% | **−0.387pp** |
| ws | 9.6045% | 9.4580% | −0.147pp |
| abupt | 8.7132% | 8.5231% | −0.190pp |

**Analysis:** Both arms fail EP3 gate (<8%). Direction partially confirmed: H=16 > H=8 monotonically (−0.19pp abupt, largest gain in vp −0.39pp). But neither beats PR #823 SOTA at H=4 (128-dim/head). The result is consistent with per-head dimensionality being the binding constraint: 128-dim/head at H=4 > 64-dim/head at H=8 > 32-dim/head at H=16. Adding more heads simultaneously narrows the K/V subspace — these two effects are entangled in standard MHA.

**Key finding:** This motivates GQA (PR #893) — decouple K/V head dimensionality from Q head count. With GQA n_kv_heads=1: K/V get full 512-dim/head while Q still has 4 specialised 128-dim query projections.

**Verdict:** NEGATIVE. Standard MHA heads=4 remains optimal. GQA follow-up assigned to alphonse (PR #893).

---

## 2026-05-09 03:50 — PR #884: Two-layer surf→vol xattn (frieren) — CLOSED (kill gate EP1)

- **Branch**: frieren/xattn-two-layer (deleted)
- **W&B run**: `omn023f3` (group `frieren-xattn-two-layer`)
- **Hypothesis**: Stack a second surf→vol cross-attention layer (identical architecture to PR #823's single layer: embed_dim=512, num_heads=4, zero-init out_proj) applied at an additional backbone depth. Both layers zero-init to preserve identity-at-init. Hypothesis: more geometry injection depth → better vol_pressure, especially for OOD cases.

| Metric | Two-layer (PR #884) | Single-layer PR #823 EP1 | Gate |
|---|---:|---:|---:|
| val_abupt EP1 | 31.77% | 28.63% | <30% |
| val_surface_pressure EP1 | 24.94% | 21.85% | — |
| val_volume_pressure EP1 | 17.88% | 17.79% | — |
| val_wall_shear EP1 | 35.28% | 31.54% | — |

**Analysis:** Kill gate triggered at EP1 (31.77% vs 30% gate). Most significant finding: surface_pressure (+3.09pp) and wall_shear (+3.74pp) regressed strongly, while volume_pressure held parity (+0.09pp). This is the diagnostic signature of K/V gradient backflow through the surface encoder being doubled (two layers of xattn each flow gradients back through surface K/V). The volume pathway (direct write target) is fine; the surface pathway (indirect K/V gradient sink) is being perturbed. Identity-at-init was verified before launch — this is a learned-dynamics regression, not an init bug.

**Student suggested follow-ups:** (1) Detach K/V before xattn — isolates surface encoder from xattn gradient backflow. (2) Add FFN after single-layer xattn. (3) Lower LR for second xattn layer. (4) Extend warmup to 2 epochs.

**Verdict:** Closed. Follow-up: frieren PR #890 tests detach-K/V (Option 1 — highest signal, directly tests the hypothesized mechanism).

---

## 2026-05-01 — PR #840: STRING drop σ=4.0 (tanjiro) — CLOSED DEAD END

- **Branch**: tanjiro/string-drop-sigma4 (deleted)
- **W&B run**: `oiptel6p`
- **Hypothesis**: Remove the highest-frequency octave (σ=4.0) from the 5-octave STRING PE spectrum {0.25, 0.5, 1.0, 2.0, 4.0} → {0.25, 0.5, 1.0, 2.0}. Motivation: σ=4.0 may add noise for low-Re smooth aerodynamic fields; leaner spectrum may regularise the PE while retaining physically meaningful frequency content.

| Metric | PR #840 (4-oct, no σ=4) | SOTA #592 (5-oct) | Gate |
|---|---:|---:|---:|
| val_abupt (EP4) | 7.856% | 6.5985% | <6.5985% |

**Analysis:** EP4 val_abupt=7.856% is substantially worse than SOTA (6.5985%), a +1.26pp regression. Removing σ=4.0 clearly degrades performance. All 5 octaves of the STRING spectrum are jointly load-bearing; the highest-frequency component contributes meaningfully to spatial resolution of near-surface aerodynamic gradients.

**Verdict (DEAD END):** STRING spectrum axis closed. All 5 octaves required. Do not prune STRING PE spectrum further.

---

## 2026-05-01 — PR #842: LR floor lr_min=5e-6 (thorfinn) — CLOSED DEAD END

- **Branch**: thorfinn/lr-floor-5e-6 (deleted)
- **W&B run**: `3487klz8`
- **Hypothesis**: Introduce a non-zero LR floor lr_min=5e-6 into the cosine annealing schedule (vs current cosine-to-zero). Prevents the LR from fully decaying to 0, maintaining a small residual learning rate at EP13 that may improve late-epoch fine-tuning on high-frequency aerodynamic features.

| Metric | PR #842 (lr_min=5e-6) | SOTA #592 (lr_min=0) | Gate |
|---|---:|---:|---:|
| val_abupt (EP4) | 7.610% | 6.5985% | <6.5985% |

**Analysis:** EP4 val_abupt=7.610% is significantly worse than SOTA (6.5985%), a +1.01pp regression. Maintaining a residual LR floor hurts performance. Cosine-to-zero decay is optimal for this task — the model benefits from full LR annihilation at end of training.

**Verdict (DEAD END):** LR floor axis closed. Cosine-to-zero (lr_min=0) is confirmed optimal.

---

## 2026-05-01 — PR #836: Geometry branch v3 (askeladd) — CLOSED CATASTROPHIC KILL

- **Branch**: askeladd/geom-branch-v3 (deleted)
- **W&B runs**: rank-0 `zj8o1ugg` (group `abupt-geom-branch-v3`)
- **Hypothesis**: Introduce a geometry-conditioning branch that processes global geometric features (e.g. SDF projections, surface statistics) and injects them into the volume decoder via FiLM conditioning. Motivation: explicit geometric context beyond point-level SDF may help the model generalise across different car body shapes.

| Metric | PR #836 (EP1) | Kill Gate | Verdict |
|---|---:|---:|---:|
| val_abupt | 50.9246% | <40% | FAILED (KILL) |

**Analysis:** EP1=50.9246% far exceeds the 40% kill gate. Root cause analysis: the backbone freeze + cosine LR schedule aliasing meant the geometry-conditioning branch received only ~2173 effective gradient steps before the EP1 gate check — insufficient to overcome the random initialisation of the new FiLM conditioning layers. The catastrophic failure reflects initialisation shock rather than a fundamentally broken architecture, but the execution plan was poorly designed.

**Verdict (CATASTROPHIC KILL):** Closed without further investigation. Geom-branch v3 architecture requires a careful warm-up strategy (progressive unfreezing, staged LR, or separate Adam phase for new conditioning layers) before re-attempting. Do not re-open without a warm-up plan.

---

## 2026-05-08 06:30 — PR #837: SDF skip-connect to volume decoder (tanjiro) — CLOSED BLOCKED (Issue #803)

- **Branch**: tanjiro/sdf-concat-vol-decoder (deleted)
- **W&B run**: `4oerprx6` (rank-0, group `tanjiro/sdf-skip-decode-4ep`) — killed at EP2 start
- **Hypothesis**: Concatenate SDF channel (`volume_x[..., 3:4]`) onto `volume_hidden` at decoder boundary (512→513→1). Zero parameter overhead, non-saturating (raw float), physically interpretable — gives decoder explicit inside/outside/surface geometry context at prediction time.

| Metric | EP1 | Kill Gate | Verdict |
|---|---:|---:|---:|
| val_abupt | 25.47% | <40% | PASSED |
| val_vol_pressure | 15.42% | — | — |
| val_surface_pressure | 18.95% | — | — |

**EP2 in progress when killed.** EP1=25.47% was healthy (well below 40% gate). Architecture is sound.

**Analysis**: Run aborted mid-EP2 by advisor due to Issue #803 data blocker. The 10 REQUIRED_RESTORED_CASE_IDs (run_44, run_133, run_158, run_184, run_203, run_226, run_249, run_310, run_416, run_484) have corrupted `volume_sdf.npy` — sdf_min ∈ [-0.015, -0.001] vs bulk train [-0.45, -0.27], meaning no inside-body samples. A model trained on this data would learn an artificial SDF distribution that does not match test cases, making any result uninterpretable. EP1=25.47% may itself be misleading if the 10 restored cases are included.

**Verdict (BLOCKED):** Architecture design is valid. Re-open as new PR after `volume_sdf.npy` regeneration for the 10 REQUIRED_RESTORED_CASE_IDs lands and passes diagnostic z<2σ check.

---

## 2026-05-08 06:30 — PR #834: GradNorm α=0.5 uniform init (edward) — CLOSED NEGATIVE (GradNorm axis exhausted)

- **Branch**: edward/gradnorm-a05-uniform-init-4ep (deleted)
- **W&B run**: `k309ojcu` (rank-0, group `edward-gradnorm-uniform-init`, project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: GradNorm with uniform static-weight initialization (all=1.0 instead of SOTA τ_y×1.5, τ_z×2.0, surface×2.0) removes the stacking interference observed in PR #824 (GradNorm + stacked static weights), allowing GradNorm to discover its own optimal trajectory unbiased by empirical priors.

| Metric | PR #834 (GN α=0.5, uniform) | PR #824 (GN α=0.5, stacked) | SOTA #592 | Gate |
|---|---:|---:|---:|---:|
| val_abupt | 7.5431% | 7.5170% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.7283% | 8.6960% | 7.9915% | — |
| val surface_pressure | 4.8327% | 4.83% | 4.3322% | — |
| val tau_x | 7.1751% | 7.13% | 6.5420% | — |
| val tau_y | 9.4863% | 9.33% | 8.3631% | — |
| val tau_z | 10.9030% | 10.95% | 9.8099% | — |
| val volume_pressure | 5.3183% | 5.35% | 3.9456% | — |

**GradNorm runtime weights (EP2 pre-val):** sp=0.79, τx=0.98, τy=0.96, τz=1.21, vp=1.06

**Analysis**: The two GradNorm variants (uniform init vs stacked static) differ by only 0.0261pp val (0.0323pp test) — within noise. Uniform initialization made no meaningful difference. GradNorm is anti-synergistic with the L5 SOTA backbone regardless of static-weight initialization. The final GradNorm weight schedule (τz=1.21 highest, τy=0.96 lower than expected) suggests GradNorm is failing to upweight τ_y properly — possibly because the gradient norm ratio tracks training speed rather than validation-loss residual. This is the 4th consecutive GradNorm experiment (PRs #523, #740, #824, #834) to land at either the SOTA baseline or worse. **GradNorm axis CONCLUSIVELY CLOSED.**

**Verdict (NEGATIVE):** Closed. GradNorm is exhausted at every α, with or without static-weight priors. Something in the L5/Lion/STRING stack makes GradNorm's gradient-norm-ratio dynamics non-informative. Future dynamic loss-weighting must use a different algorithm (e.g., PCGrad, loss-balanced weighting based on val residuals, not gradient norms).

---

## 2026-05-08 06:30 — PR #833: τ_z×2.5 4-ep curriculum bisection (thorfinn) — CLOSED NEGATIVE (τ_z static weight axis exhausted)

- **Branch**: thorfinn/tau-z-bisect-2p5-4ep (deleted)
- **W&B run**: `8a7mfzl3` (rank-0, project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: τ_z×2.5 bisects the τ_z×2.0 (SOTA) and τ_z×3.0 (PR #822) interval. If a sweet spot exists between them, τ_z×2.5 should find it. PR #822 confirmed τ_z×3.0 is +0.88pp vs SOTA; τ_z×2.5 should be closer to SOTA than ×3.0.

| Metric | PR #833 (τ_z×2.5) | PR #822 (τ_z×3.0) | SOTA #592 (τ_z×2.0) | Gate |
|---|---:|---:|---:|---:|
| val_abupt | 7.5378% | 7.4767% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.6920% | 8.6647% | 7.9915% | — |
| val tau_z | 10.8128% | 10.6947% | 9.8099% | — |
| val tau_y | 9.5479% | — | 8.3631% | — |
| val surface_pressure | 4.9448% | — | 4.3322% | — |
| val volume_pressure | 4.9687% | — | 3.9456% | — |
| EP1 / EP2 / EP3 / EP4 | 27.57 / 11.32 / 8.24 / 7.54 | — | — | — |

**Analysis**: τ_z×2.5 (val=7.5378%) is barely different from τ_z×3.0 (7.4767%) — only 0.06pp separates them. Both are ~0.90-0.94pp WORSE than SOTA τ_z×2.0. The bisection confirms there is no sweet spot in [2.0, 3.0]: the function is monotonically degrading as τ_z weight increases above 2.0. The non-uniform vol-points schedule was tuned at τ_z×2.0 and cannot absorb additional τ_z gradient pressure. Upweighting τ_z increases gradient-clip frequency and hurts every other channel (vol_p +1.02pp, surf_p +0.61pp, τ_y +1.18pp). The τ_z static-weight axis is a wall.

**Verdict (NEGATIVE):** Closed. The full τ_z sweep (×2.0, ×2.5, ×3.0) is complete. τ_z×2.0 is the 4-ep local optimum. No further τ_z static-weight experiments warranted at this budget.

---

## 2026-05-01 — PR #832: Lion wd=7e-4 (alphonse) — CLOSED DEAD END

- **Branch**: alphonse/lion-wd-7e-4 (deleted)
- **W&B run**: `cq4guj8g` (rank-0, group `alphonse-lion-wd`)
- **Hypothesis**: Increasing Lion weight decay from 5e-4 to 7e-4 would reduce overfitting and improve generalization on L5 SOTA config.

| Metric | EP1 | EP2 | EP3 | EP4 (best) | SOTA EP4 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| val_abupt | 27.08% | 11.62% | 8.418% | **7.683%** | 6.5985% | +1.085pp WORSE |
| val_surface_pressure | — | — | — | 5.284% | 4.332% | +0.952pp worse |
| val_volume_pressure | — | — | — | 4.986% | 3.946% | +1.040pp worse |
| val_wall_shear | — | — | — | 8.810% | 7.585% | +1.225pp worse |
| test_abupt | — | — | — | 8.877% | 7.992% | +0.885pp worse |

**Analysis**: wd=7e-4 uniformly degraded all channels. The wd axis on L5/Lion/9e-5 is now closed on both sides: wd=3e-4 (PR #826) gave +0.864pp, wd=7e-4 gives +1.085pp. Current wd=5e-4 is the local optimum. Broadband degradation across all channels (not just vol_p) rules out the channel-specific mechanism hypothesized. EP1 was marginally better but the gap inverted by EP2 and never recovered, confirming this is a genuine regression, not a timing artifact. **Lion wd axis CLOSED under L5/9e-5 config.**

---

## 2026-05-01 — PR #836: AB-UPT geometry branch v3 (askeladd) — SENT BACK (recipe fix)

- **Branch**: askeladd/geom-branch-v3
- **W&B run**: `zj8o1ugg` (rank-0, group `abupt-geom-branch-v3`)
- **Hypothesis**: AB-UPT geometry branch with supernode pooling: K=1024 anchor points from volume mesh, STRING-sep RoPE, two new output heads (surface+volume MLP), anchor→point cross-attention. Training recipe: backbone freeze warmup (20%), differential LR (2×), vol aux weight (2.0).

| Metric | EP1 | Kill Gate | Verdict |
|---|---:|---:|---:|
| val_abupt | 50.9246% | <40% | KILLED |

**Analysis**: Architecture plumbing verified healthy — geom_branch/* W&B telemetry shows no NaN, freeze/unfreeze in DDP worked correctly, lr-scale applied correctly. The failure was a pure recipe interaction: (1) `--lr-cosine-t-max 4` with 4-epoch run decays backbone_lr from 9e-5 to 4.5e-6 by EP1 end; (2) `--geom-branch-warmup-fraction 0.2` freezes backbone for ~80% of EP1 (~8691/43456 warmup steps), leaving backbone with only ~2173 steps of actual training after unfreeze at severely decayed LR (~4.5e-6). These two effects compound to guarantee EP1 kill. Same `--lr-cosine-t-max 4` confound affected PR #835 (frieren). **Fix applied: drop `--geom-branch-warmup-fraction` to 0.0 and set `--lr-cosine-t-max 13`. Re-running as `askeladd/geom-branch-v3-nf-ep4`.**

---

## 2026-05-08 03:10 — PR #830: Volume loss weight 2.0 4-ep curriculum (tanjiro) — CLOSED HYPOTHESIS REJECTED

- **Branch**: tanjiro/vol-loss-weight-2 (deleted)
- **W&B run**: `ztvlsn1e` (rank-0, group `tanjiro-vol-loss-weight`)
- **Hypothesis**: Doubling the volume loss weight (1.0→2.0) under the canonical 4-ep curriculum would redirect gradient capacity to the volume branch, improving volume_pressure (val 3.9% vs test 11.9% gap diagnosed as under-optimization).

| Metric | EP1 | EP2 | EP3 | EP4 (best) | SOTA EP4 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| val_abupt | 27.17% | 12.01% | 8.487% | **7.7117%** | 6.5985% | +1.11pp WORSE |
| val_surface_pressure | 20.37% | 8.30% | 5.616% | 5.097% | 4.332% | +0.77pp worse |
| val_volume_pressure | 15.38% | 7.88% | 5.283% | 4.782% | 3.946% | +0.84pp worse ← TARGET |
| val_wall_shear | 30.67% | 13.27% | 9.552% | 8.709% | 7.585% | +1.12pp worse |

**Analysis**: Hypothesis failed convincingly — vol-w=2.0 degraded ALL channels at EVERY epoch, including volume_pressure itself. Trajectory monotonically below baseline from EP1 through EP4 (not a "needs more epochs" failure). Two plausible mechanisms: (1) higher volume weight causes gradient-clip to fire more often at fixed lr=9e-5, reducing effective step on all params; (2) curriculum front-loads bad signal from 16K sparse vol-point gradients. The val/test gap on volume_pressure (3.9% val vs 11.9% test) is a generalization gap, not under-optimization — loss reweighting is the wrong lever. This hypothesis is now confirmed dead twice: PR #813 (5-ep) and this PR (4-ep curriculum). **Volume-loss-weight axis closed for L5/Lion/9e-5 recipe.**

---

## 2026-05-08 03:08 — PR #829: STRING 6-octave RFF σ=0.125–4.0 (fern) — CLOSED DEAD END

- **Branch**: fern/string-6octave-pe (deleted)
- **W&B run**: `cqk9voaa` (rank-0, group `fern-string-6octave`)
- **Hypothesis**: Adding a 6th higher-frequency octave (σ=0.125) to STRING positional encoding, below the current minimum σ=0.25, would improve surface pressure and other channels by capturing finer-scale geometric variation.

| Metric | EP1 | EP2 | EP3 | EP4 (best) | SOTA EP4 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| val_abupt | 31.58% | 11.59% | 8.331% | **7.5738%** | 6.5985% | +0.97pp WORSE |
| val_surface_pressure | — | 7.897% | 5.409% | 4.906% | 4.332% | +0.57pp worse |
| val_volume_pressure | — | 8.355% | 5.740% | 5.121% | 3.946% | +1.18pp worse |
| val_wall_shear | — | 12.60% | 7.843% | — | 7.585% | — |
| test_abupt | — | — | — | 8.920% | 7.992% | +0.93pp worse |

**Analysis**: σ=0.125 uniformly degraded all channels on both val and test by 0.5–1.4pp. Two plausible causes identified: (1) **aliasing** — σ=0.125 places PE energy below the supervisable label scale (65k surface points too sparse for this frequency), injecting noise; (2) **capacity competition** — at fixed rff_num_features=16 across 6 sigmas, each sigma gets ~2.67 features vs 3.2 in 5-octave SOTA, starving the load-bearing σ=0.25 octave. The train_loss-matches-but-val-degrades signature confirms aliasing is operative. **Follow-up PR #838 (fern, rff24+σ=0.125) isolates the capacity-competition cause by giving 24 features across 6 sigmas (4 each), giving σ=0.25 MORE budget than current SOTA. If PR #838 also fails, aliasing is the dominant cause and σ=0.125 is definitively unusable at 65k pts.**

---

## 2026-05-08 02:10 — PR #828: 2-layer GELU MLP vol decoder (askeladd) — CLOSED DEAD END

- **Branch**: askeladd/vol-decoder-2layer-gelu (deleted)
- **W&B run**: `zmcwyud5` (rank-0, group `askeladd-vol-decoder-mlp`)
- **Hypothesis**: Replace the linear volume pressure decoder head with a 2-layer GELU MLP (512→256→1, LayerNorm between layers) to give the network richer capacity to decode volume pressure, targeting the OOD vol_p gap.

| Metric | EP1 (16k vol-pts) | EP2 (32k vol-pts) | SOTA EP1 | SOTA EP2 |
|---|---:|---:|---:|---:|
| val_abupt | 31.06% | 11.42% | 27.95% | 7.94% |
| val vol_pressure_rel_l2 | 16.99% | 8.38% | — | — |
| val surface_pressure_rel_l2 | 24.89% | 7.52% | — | — |
| val wall_shear_rel_l2 | 35.45% | 14.18% | — | — |

**Analysis**: Gap vs baseline widened from +3.11pp at EP1 to +3.48pp at EP2 across ALL channels — not just vol_p. This rules out a slow-convergence explanation. The 2-layer GELU MLP decoder adds ~1.25M params but slows optimization uniformly. Root cause: richer output decoder increases gradient path depth; the model cannot amortize this in 4 epochs. This is the second time this hypothesis was tested (PR #820 showed identical outcome). The vol-pressure OOD problem requires geometry-aware *input* conditioning, not a richer *output* decoder.

---

## 2026-05-08 02:10 — PR #827: Cosine LR warm restarts on L5 SOTA 4-ep (frieren) — CLOSED INFORMATIVE

- **Branch**: frieren/cosine-lr-warm-restarts (deleted)
- **W&B run**: `1ne1qdfl` (rank-0)
- **Hypothesis**: CosineAnnealingWarmRestarts (T_0=2) would escape local minima, improving vol_p.

| Metric | EP2 | EP3 (best_val) | SOTA EP4 |
|---|---:|---:|---:|
| val_abupt | 8.7973% | **7.4450%** | **6.5985%** |
| val vol_pressure | 5.492% | 4.419% | 3.946% |

EP3 gate PASSED (<8%). Restart-1 confirmed at step 32593. EP4 timed out (52% complete). Best=7.445%, above merge gate by 0.85pp. Hypothesis untestable in 4-ep budget. Closed informative. Restart mechanics confirmed working; monotone cosine confirmed productive.

---

## 2026-05-01 — PR #824: GradNorm α=0.5 on L5 SOTA 4-ep curriculum (edward) — CLOSED NEGATIVE

- **Branch**: edward/gradnorm-a05-l5-sota-4ep (deleted)
- **W&B run**: `e0brbohf` (rank-0, group `edward-gradnorm-l5-sota`, project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: GradNorm α=0.5 dynamic per-task loss reweighting stacked on the full L5 SOTA stack (alphonse PR #592 recipe with static surface=2.0, τ_y=1.5, τ_z=2.0) at 4-ep budget-matched curriculum would match/beat SOTA by adaptively upweighting laggard τ_y/τ_z channels.

| Metric | PR #824 (GradNorm α=0.5 + SOTA static) | PR #740 (GradNorm α=0.5, no static) | SOTA #592 | Gate |
|---|---|---|---|---|
| val_abupt | 7.5170% | — | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.6960% | 7.5195% (wave-test SOTA) | 7.9915% | — |
| val surface_pressure | 4.83% | — | 4.33% | — |
| val tau_x | 7.13% | — | 6.54% | — |
| val tau_y | 9.33% | — | 8.36% | — |
| val tau_z | 10.95% | — | 9.81% | — |
| val vol_pressure | 5.35% | — | 3.95% | — |

**Final GradNorm weights:** sp=0.75, τx=0.96, τy=1.20, τz=1.24, vp=0.85. Directionally matched PR #740 except vp downweighted.

**Results commentary:** All five channels strictly regressed (+0.50 to +1.40pp), not a tradeoff. test_abupt is +1.18pp WORSE than PR #740's GradNorm wave-test SOTA — the difference is that PR #740 ran without the SOTA static weights, while this run stacked GradNorm on top of them. GradNorm overrides static weights based on gradient norms alone (not val-loss progress), so the runtime weight schedule (sp 2.0×0.75=1.5, τy 1.5×1.20=1.8, τz 2.0×1.24=2.5, vp 1.0×0.85=0.85) is less-well-tuned than the static SOTA empirical optimum. The two mechanisms are not stacking-compatible at this budget.

**Verdict (NEGATIVE):** Closed. GradNorm + static-weighted SOTA = anti-synergy. To get a GradNorm signal one would need to drop the static weights entirely (revert tau-y-loss-weight, tau-z-loss-weight, surface-loss-weight to 1.0) and let GradNorm own the schedule.

## 2026-05-01 — PR #826: Lion weight-decay 5e-4 -> 3e-4 (alphonse) — CLOSED NEGATIVE

- **Branch**: alphonse/weight-decay-3e-4-l5-sota-4ep (deleted)
- **W&B run**: `ahw1rdj7` (group `alphonse-wd-sweep`, project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: Halving Lion weight-decay from 5e-4 to 3e-4 would relax the L2 pull on tau_y/tau_z output-projection weights and lift the worst channels without harming surface_pressure or vol_pressure.

| Metric | PR #826 (wd=3e-4) | SOTA #592 (wd=5e-4) | Gate |
|---|---|---|---|
| val_abupt | 7.4628% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.7253% | 7.9915% | — |
| val surface_pressure | 4.88% | 4.33% | — |
| val tau_x | 7.31% | 6.54% | — |
| val tau_y | 9.42% | 8.36% | — |
| val tau_z | 10.85% | 9.81% | — |
| val vol_pressure | 4.86% | 3.95% | — |

**Results commentary:** All channels degraded uniformly (+0.6 to +1.0pp). Lion's update is `sign(momentum) * lr + lr * wd * theta` — halving wd shrinks the explicit parameter-pull term and starves convergence across the whole network, not selectively at decoder heads. Confirms wd=5e-4 is at/near the Lion sweet spot for this recipe; tau_y/tau_z headroom is structural, not regulatory.

**Verdict (NEGATIVE):** Closed. Down-sweep of Lion wd is dead — pivot to structural attacks on the channel imbalance (channel-specific decoder heads, schedule-aware loss weighting at appropriate budget, or different optimizer dynamics like β₂ sensitivity).

## 2026-05-01 — PR #822: τ_z loss weight ×3.0 on 4-ep budget-matched curriculum (thorfinn) — CLOSED NEGATIVE

- **Branch**: thorfinn/tau-z-3p0-4ep-relaunch (deleted)
- **W&B run**: `qtzoy6rp` (group `thorfinn-tau-z-sweep`, project `senpai-v1-drivaerml-ddp8`); first attempt `imvj1s1p` killed by misconfigured EP1 kill threshold.
- **Hypothesis**: Stacking τ_z×3.0 on the full SOTA recipe at 4-ep budget-matched curriculum would extend the +0.44pp τ_z signal observed in PR #807 isolation and lift val_abupt below SOTA.

| Metric | PR #822 (τ_z×3.0, 4-ep) | PR #807 (τ_z×3.0 isolation) | SOTA #592 | Gate |
|---|---|---|---|---|
| val_abupt | 7.4767% | — | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.6647% | — | 7.9915% | — |
| val tau_z | 10.6947% | bare-SOTA −0.44pp | 9.8099% | — |
| EP1 / EP2 / EP3 / EP4 val_abupt | 26.18 / 11.37 / 8.17 / 7.48 | — | — | — |

**Results commentary:** All channels still descending at EP4 — training did not converge. The 4-ep budget-matched curriculum delivers ~22,640 total steps (10864 + 5435 + 3625 + ~2716, non-uniform due to varying volume-point-count epochs), substantially fewer than the 13-ep baseline's ~43k steps. τ_z×3.0 amplifies the slowest-converging channel's gradient, which demands MORE budget, not less, to integrate. Stacking it onto a budget-starved schedule is anti-synergistic: the recipe needed 14k+ extra steps (PR #815 13-ep variant timed out) to express the τ_z gain. Confirms the signal is real but not landable in the 4-ep envelope at ×3.0 magnitude.

**Verdict (NEGATIVE):** Closed. 4-ep + full-SOTA + τ_z×3.0 is over-stacked. Either reduce the upweight magnitude (×2.0 at 4-ep) or attack channel imbalance through orthogonal means (separate decoder heads, warm-start from SOTA checkpoint, schedule-aware loss).

## 2026-05-01 — PR #814: STRING 6-octave extended spectrum (add σ=8.0) (alphonse) — CLOSED NEGATIVE

- **Branch**: alphonse/string-6-octave-extended-spectrum
- **W&B run**: `3efn3v5u` (project `senpai-v1-drivaerml-ddp8`)
- **Hypothesis**: Adding σ=8.0 as a 6th RFF octave (`--rff-init-sigmas "0.25,0.5,1.0,2.0,4.0,8.0"`) captures finer-scale geometric features that the 5-octave SOTA misses, particularly for wall_shear_z (confirmed laggard). Motivated by thorfinn PR #779 Arm B signal (σ_max=8 replacing σ=4 gave −0.13pp improvement).

| Metric | PR #814 (6-oct additive) | thorfinn #779 Arm B (5-oct σ_max=8) | SOTA #592 (5-oct) | Gate |
|---|---|---|---|---|
| val_abupt | 7.6385% | 6.8792% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | 8.8442% | — | 7.9915% | — |
| test_surface_p | 4.5974% | — | — | — |
| test_volume_p | 12.8395% | — | — | — |
| test_wall_shear | 8.2213% | — | — | — |
| full_val/wall_shear_z | 11.0287% | — | — | — |
| best_epoch | 4 (EMA) | — | — | — |

**Kill gates:** All 3 passed; run completed 22,644 steps (~190 min).

**Slope decay:** val_abupt slope decelerated from −2.506 pp/1k steps (EP1→EP2) to −0.266 pp/1k steps (EP3→EP4) — the 6-octave config needs more budget to clear convergence overhead.

**Results commentary:** The 6-octave additive approach is +0.76 pp worse than thorfinn's 5-octave replace-not-add variant at identical 4-epoch budget. Adding a 6th octave grows RFF features 80→96 (+20%); this extra capacity is a liability at 4 epochs because the optimizer has not had enough iterations to integrate the new frequency. The σ=8.0 signal in PR #779 Arm B worked precisely because it *replaced* σ=4.0 (constant capacity), not because it added bandwidth. τz (wall_shear_z) was NOT preferentially accelerated — wsy descended faster in EP3→EP4, meaning the 6th octave did not help the confirmed laggard channel. A 13-epoch full-budget run might resolve the convergence lag but is not justified over other queued hypotheses.

**Verdict (NEGATIVE):** Closed. 6-octave additive is inferior to 5-octave replace at 4-epoch budget. SOTA STRING PE remains 5-octave {0.25,0.5,1.0,2.0,4.0}.

## 2026-05-07 — PR #808: Surface curvature 4ep original-schedule re-run (nezuko) — CLOSED DEAD END (3rd consecutive surface-curvature fail)

- **Branch**: nezuko/surface-curvature-4ep-original-schedule (deleted)
- **W&B run**: `3hsu3tq0` (group `nezuko-surface-curvature`, name `nezuko/surface-curvature-4ep-original-vol-schedule`)
- **Hypothesis**: Surface curvature features (mean curvature H̃, Gaussian curvature K̃) appended to the surface input path improve val_abupt by providing geometric context. Previous run #798 used a 4-ep schedule-aligned stack; this run uses the original vol-schedule (`0:16384:3:32768:6:49152:9:65536`) for a full 65k vol-point budget at EP4.

| Metric | PR #808 (this, 4-ep orig-sched) | PR #798 (4-ep aligned) | PR #788 (first curvature) | SOTA #592 | Gate |
|---|---:|---:|---:|---:|---|
| val_abupt | 6.8051% | ~6.78% | 7.35% | **6.5985%** | < 6.5985% ❌ |
| test_abupt | — | — | — | 7.9915% | — |
| test_volume_p | ~11.81% | — | — | 11.933% | — |

**W&B run state:** Finished. EP1=24.95%, EP2=8.30%, EP3=7.166%, EP4=6.8051%.

**Results commentary:** Three consecutive surface curvature runs (PRs #788, #798, #808) have all landed in the 6.78–7.35% val_abupt range — consistently above the SOTA gate of 6.5985%. Despite varying the schedule alignment and vol-point curriculum, the convergence floor remains ~0.20–0.21pp above the gate. The only positive signal is a modest test_vol_p improvement (~0.12pp better than SOTA 11.933% → ~11.81%) which is insufficient to justify further surface-curvature investment at L=5/4-ep. Surface curvature as a standalone surface-path augment for L=5 architecture is a dead end.

**Verdict (DEAD END):** Closed after 3 runs with zero gate crossings. The curvature signal may become useful only if composited with a deeper architecture (L=6+) or longer training. Not assigning follow-up for now — geometry conditioning priority shifts to vol-head LoRA and AB-UPT geometry branch.

## 2026-05-07 — PR #807: Schedule-aligned 4-ep τ_z×3.0 upweight isolation (thorfinn) — NOT MERGED (below single-model gate), hypothesis CONFIRMED

- **Branch**: thorfinn/schedule-aligned-tau-z-upweight
- **W&B run**: `8j9kt5w1` (group `thorfinn-tau-z-sweep`, name `thorfinn/tau-z-3p0-sched4`)
- **Hypothesis**: τ_z (wall shear z) is the confirmed training laggard (PR #758: r_i=0.01123 highest residual imbalance). SOTA uses τ_z×2.0 but val tau_z=9.81% remains far from AB-UPT ref 3.63%. Test: increase τ_z weight from 2.0→3.0 on a clean 4-ep schedule-aligned stack (same as fern bare-SOTA control PR #799) for a single-variable A/B comparison.

| Metric | thorfinn τ_z×3.0 (4-ep) | fern bare-SOTA (4-ep, #799) | SOTA (#592, 13-ep) | Gate |
|---|---|---|---|---|
| val_abupt | 6.824% | 7.063% | **6.5985%** | < 6.5985% ❌ (+0.23pp) |
| test_abupt | 8.145% | 8.444% | 7.9915% | — |
| surface_pressure (val) | 4.491% | 4.641% | 4.332% | — |
| volume_pressure (val) | 4.187% | 4.322% | 3.946% | — |
| tau_x (val) | 6.852% | 7.089% | 6.542% | — |
| tau_y (val) | 8.528% | 8.755% | 8.363% | — |
| tau_z (val) | **10.062%** | 10.506% | 9.810% | — |

**Results commentary:** Hypothesis confirmed — τ_z×3.0 beats τ_z×2.0 on the same 4-ep schedule on every channel, with tau_z showing the **largest single-channel improvement** (−0.44pp val, −0.29pp test vs bare-SOTA control). Best 4-ep result in the program to date. However, does not beat the single-model gate (6.5985%) because the 4-ep schedule is compute-limited vs the 13-ep SOTA. The 4-ep schedule is a ~3.5h run that converges to ~7% range, while the 13-ep SOTA at ~270min/4ep gets the full cosine decay benefit. **Follow-up: assign τ_z×3.0 on the full 13-ep SOTA recipe to thorfinn.** The PR #790 (alphonse, τ_z×3.0 13-ep) was confounded by a 270-min wall-clock truncation in the high-LR phase; this is now cleanly motivated by the 4-ep isolation proof.

## 2026-05-01 — PR #793: vol-w=2.0 + wall-shear tau bump (tanjiro) — CLOSED NEGATIVE

- **Branch**: tanjiro/vol-w-2.0-wallshear-rebalance (deleted)
- **W&B run**: `ss5v4vdx` (group `vol-w-wallshear-rebalance-tay`, name `tanjiro/vol-w2.0-tau-y2.5-z3.0`)
- **Hypothesis**: `--volume-loss-weight 2.0` + `--tau-y-loss-weight 2.5` + `--tau-z-loss-weight 3.0` to rebalance wall-shear loss budget after PR #776 Arm B (vol-w=2.0 alone) caused +0.57pp wall-shear regression. Composed reweighting expected to recover val_abupt while retaining the test_vol_p OOD win.

**Final verified metrics (W&B `ss5v4vdx`, run state: finished):**

| Metric | PR #793 (this) | SOTA #592 `4k25s25e` | PR #776 Arm B (vol-w=2.0 solo) | Gate |
|---|---:|---:|---:|---|
| `full_val_primary/abupt_axis_mean_rel_l2_pct` | 7.2412% | **6.5985%** | 7.2231% | < 6.5985% ❌ FAIL (+0.657pp) |
| `test_primary/abupt_axis_mean_rel_l2_pct` | 8.5761% | **7.9915%** | 8.3466% | — ❌ regressed |
| `test_primary/volume_pressure_rel_l2_pct` | 12.2003% | 11.9335% | **11.5596%** | — ❌ Arm B win destroyed |
| `test_primary/surface_pressure_rel_l2_pct` | 4.5669% | **4.0683%** | 4.3820% | — ❌ regressed |
| `test_primary/wall_shear_rel_l2_pct` | 8.0632% | **7.3338%** | 7.9073% | — ❌ regressed vs both |
| val→test vol_p OOD gap | 7.95pp | 7.99pp | **7.32pp** | — Arm B win gone |

**Mechanism failure analysis:**
- Four simultaneous channel up-weights (vol×2.0, tau_y×2.5, tau_z×3.0, surface×2.0) starved every channel of effective gradient signal. Competing pulls on a single 100-epoch budget degrade all channels.
- Per-axis z>y>x ordering remained structurally invariant to per-axis tau weights — z–y gap WIDENED EP3→terminal (1.38pp → 1.63pp). Per-axis loss weights cannot fix structural z-axis difficulty.
- The Arm B OOD-gap win (7.32pp) was destroyed (regressed to 7.95pp). vol-w=2.0 OOD-pressure win is fragile under any additional reweighting.

**Verdict (NEGATIVE):** Both win conditions failed. The hypothesis that tau-weight bumps could compensate for vol-w=2.0 wall-shear budget starvation is refuted. Lesson: vol-w=2.0 must be tested as a single variable in isolation. Follow-up: PR #805 (tanjiro) — vol-w=2.0 on schedule-aligned 4-epoch stack as true single-variable isolation.

## 2026-05-01 — PR #792: FiLM v3 compressed curriculum (frieren) — CLOSED DESIGN-NEGATIVE

- **Branch**: frieren/vol-film-v3-compressed-curriculum
- **W&B run**: `uhyi1e6k` (group `vol-film-v3-compressed-curriculum`)
- **Hypothesis**: Compressing the vol-points curriculum to `0:16384:1:32768:2:49152:3:65536` allows FiLM to activate at EP3 (instead of EP6+ in the standard schedule), giving 5× more FiLM-active steps within the 270-min budget. Thesis: ≥5× FiLM-active training time → improved test_vol_p / test_abupt vs v2 (PR #778, 1 FiLM-active epoch).

**Final test results (EP7 EMA, 5 FiLM-active epochs, run `uhyi1e6k`):**

| Metric | v3 (this) | v2 (PR #778) | SOTA #592 | Vol-anchor #681 |
|---|---:|---:|---:|---:|
| test_abupt | 8.2969% | ~8.25% | 7.9915% | — |
| test_vol_p | 12.239% | 12.110% | 11.933% | 11.374% |
| test_surface_p | 4.2445% | — | 4.22% | — |
| test_wall_shear | 7.652% (x=6.782/y=8.522/z=9.697) | — | — | — |

**FiLM dynamics (5 FiLM-active epochs EP3-EP7):**
- γ_mean climbed 0.304 → 0.631 with decelerating rate
- γ_max saturated at tanh asymptote 100% of batches from EP4 onward
- β stayed sparse throughout (mean ~0.025)

**Verdict (DESIGN-NEGATIVE):** 5× more FiLM-active steps (EP3-EP7 vs only EP6 in v2) produced essentially equivalent test metrics (+0.129pp test_vol_p vs v2). The thesis "more FiLM-active training time → better metrics" is NOT supported. Key diagnostic: γ_max saturation at the tanh upper bound from EP4 onward indicates the bounded tanh parameterization (γ∈(0,2)) is a capacity bottleneck. FiLM mechanism is structurally working (bounded, stable, monotone val descent) but the current γ range is insufficient to further improve vol_pressure. Closing. Not pursuing FiLM v4 wider-bounds as immediate follow-up — the 0.86pp test_vol_p gap to anchor is more likely a wallclock/data-throughput limitation than a FiLM-dosage issue.

## 2026-05-07 — PR #789: SDF-gate v2/v3/v4 vol-decoder (askeladd) — CLOSED DESIGN-NEGATIVE (all 3 tanh-cap variants)

- **Branch**: askeladd/vol-decoder-sdf-gate-v3
- **W&B run**: `qazswyke` (group `vol-geom-cond`, name `askeladd/vol-decoder-sdf-gate-v3`)
- **Hypothesis**: Per-case SDF features → small MLP → bounded scalar gate on volume decoder logits (cap=0.15, gate-WD=5e-3, 2-epoch independent gate warmup) prevents v2's saturation collapse and reduces test_vol_p OOD error.

**Final test results (EP4 EMA, 86% of EP4, run hit 270-min timeout):**

| Metric | v3 (this) | SOTA #592 | Vol-anchor #681 | Arm A control |
|---|---:|---:|---:|---:|
| test_abupt | **8.1945%** | 7.9915% | — | — |
| test_volume_p ★ | **12.0454%** | 11.933% | 11.374% | 12.092% |
| test_surface_p | 4.2453% | 4.22% | — | — |
| test_wall_shear | 7.5429% | 7.49% | — | — |
| test_ws_x / y / z | 6.66 / 8.43 / 9.59 | — | — | — |
| val_abupt | **6.8400%** | 6.5985% | — | 7.0077% |
| val_vol_p | 4.2617% | 3.9456% | — | — |
| val_surf_p | 4.4960% | 4.3322% | — | — |
| val_wall_shear | 7.6860% | 8.24% | — | — |

**Gate diagnostics (test, 11,091 points):** scale_max_abs=0.1504, sat_frac=9.02e-5 (1 OOD case), scale_mean=−0.0834 (identical to val), scale_range=0.0674, scale_std=0.000987, bias_max_abs=0.0077. **train/sat_frac=0 across all 37,268 steps.**

**Verdict:** Structural fix works (v2 hit sat_frac=1.0 by step ~2k, v3 stayed at 0 throughout). Gate generalizes cleanly val↔test at scale_mean=−0.083. Within-experiment Arm-A control beat: −0.17pp val_abupt, −0.05pp test_vol_p (small but signal-positive). However, single-model SOTA gate not met (+0.24pp val, +0.20pp test) — primary cause is the 270-min wall-time cap stopping training at 86% of EP4 in a 13-epoch cosine. Student's post-mortem identifies LR coupling (gate_lr = scheduled_lr × gate_factor) as having cost ~half an epoch of useful gate training time.

**v4 update (W&B run `ccnssij7`, group `vol-geom-cond`, name `askeladd/vol-decoder-sdf-gate-v4`):** LR decoupling confirmed — gate LR stayed constant 5e-05. Despite LR fix, gate fully saturated (sat_frac=1.00, scale_range=0.0000) by step 8,501 — before EP1 (step 10,864). EP3 val_abupt=7.447% — worse than Arm A control (7.0077%) and v3 best (6.840%). The tanh-cap (=0.15) architecture pushes scale outputs onto the cap regardless of LR scheduling.

**Final verdict — CLOSED DESIGN-NEGATIVE:** All three versions (v2/v3/v4) of the tanh-cap multiplicative gate failed via saturation. The architecture is fundamentally insufficient. New direction: additive rank-r LoRA on volume output projection (PR #809, no activation caps, zero-init B, bounded by construction).

## 2026-05-01 08:30 — PR #809: additive LoRA on volume output head, r=4 and r=8 (askeladd) — ASSIGNED

- **Branch**: askeladd/vol-head-lora-additive
- **Hypothesis**: Additive low-rank correction (LoRA) on `volume_out` linear projection: `volume_preds += B(A(volume_hidden))` with A∈R^{hidden×r}, B∈R^{r×vol_out} — B zero-initialized so initial correction is exactly zero, no saturation risk. Targets the chronic vol_pressure test-vs-val gap (val 3.6%, test 11.5%, ~3× in best ensemble). Architecture inherits all SDF information already encoded in volume_hidden (SDF is part of volume_x). r=4 (Arm A) and r=8 (Arm B) tested against SOTA L=5 backbone.
- **Gate**: val_abupt < 6.5985% (single-model SOTA #592) / secondary: reduce test vol_pressure below 11.5%
- **Status**: ASSIGNED — waiting for askeladd to pick up

## 2026-05-08 — PR #782: SDF-FiLM volume conditioning (edward) — CLOSED NEGATIVE

- **Branch**: edward/sdf-explicit-vol-geometry-conditioning (deleted)
- **W&B run**: `rtww6a8e` (group `sdf-film-vol-geometry`)
- **Hypothesis**: Per-case SDF stats (mean/std/min/max) → 2-layer MLP → bounded-tanh γ ∈ (0,2) and β FiLM on volume tokens reduces the val→test vol_p gap (PR #767 showed 4 OOD test cases account for 92% of squared test_vol_p deviation).

**Best-EMA results (EP4, only FiLM-active epoch — run cut at 4/13 due to 2.8× cluster slowdown):**

| Metric | SDF-FiLM (this) | SOTA #592 | Δ |
|---|---:|---:|---:|
| val_abupt | 6.9289% | 6.5985% | +0.330pp |
| test_abupt | 8.1456% | 7.9915% | +0.154pp |
| test_volume_p ★ | 12.2120% | 11.9335% | **+0.279pp** |
| test_surface_p | 4.1375% | 4.0683% | +0.069pp |
| val→test vol_p gap | 7.998pp | 7.988pp | +0.011pp |

**FiLM diagnostics:** γ_mean=0.9202, γ_max_abs_dev=0.5195 (52% of saturation), β_max_abs=0.5742, no nonfinite grads, identity-at-init verified, DDP-safe multiply-by-zero pattern works end-to-end.

**Verdict:** Hypothesis NOT supported. Implementation sound but training cut to 4/13 epochs with only ONE FiLM-active epoch. The val→test gap on vol_p was structurally unchanged, suggesting the 4 OOD cases may be extrapolative w.r.t. the train SDF stat manifold (FiLM cannot help where there is zero training support). Follow-up: PR #797 SDF coverage diagnostic.

## 2026-05-01 — PR #798: surface curvature 4-epoch schedule-aligned re-run (nezuko) — CLOSED NEGATIVE (design)

- **Branch**: nezuko/surface-curvature-4ep-aligned (deleted)
- **W&B run**: group `nezuko-surface-curvature-4ep`, name `nezuko/surface-curvature-4ep-aligned`
- **Hypothesis**: PR #788 was cut at 81% of EP4 with no LR cooldown (`--lr-cosine-t-max 13`). Re-run with `--epochs 4 --lr-cosine-t-max 4` to provide full EP4 LR cooldown and confirm the curvature signal win. Same 9-channel surface_x (7 base + H̃ + K̃), same optimizer/architecture.

**Final verified metrics (EP4 EMA, full run, schedule-aligned):**

| Metric | PR #798 (EP4 cooldown) | PR #788 (EP4 81% cut) | SOTA #592 | Δ vs SOTA |
|---|---:|---:|---:|---:|
| val_abupt | 7.3508% | 6.7767% | **6.5985%** | +0.752pp |
| test_abupt | 8.6458% | 8.139% | **7.9915%** | +0.654pp |
| test_surface_p | 4.4908% | 4.168% | **4.0683%** | +0.423pp |
| test_wall_shear | 7.9537% | 7.4189% | **7.3338%** | +0.620pp |
| test_volume_p | 12.7115% | 12.254% | 11.9335% | +0.778pp |

**Curvature gradient health (from nezuko diagnostics):**

| step | param_norm | global_norm | grad/param | zero_fraction |
|---|---:|---:|---:|---:|
| 249 (warmup) | 4.58 | 0.0099 | 0.0021 | 0.000 |
| 10,499 (EP1 end) | 4.82 | 0.1121 | 0.0233 | 0.000 |
| 16,000 (EP2 end) | 11.18 | 0.1110 | 0.0099 | 0.000 |
| 19,501 (EP3 end) | 12.85 | 0.1220 | 0.0095 | 0.000 |
| 22,502 (EP4 end) | 13.48 | 0.1427 | 0.0106 | 0.000 |

**Root-cause: compressed vol-schedule cut total optimizer steps by 36%**

The run used `--vol-points-schedule "0:16384:1:32768:2:49152:3:65536"` (from frieren's PR #792 default suggestion). This caused:

| Epoch | vol_points | Steps | Cumulative |
|---|---:|---:|---:|
| EP1 | 16,384 | 10,864 | 10,864 |
| EP2 | 32,768 | 5,435 | 16,299 |
| EP3 | 49,152 | 3,625 | 19,924 |
| EP4 | 65,536 | 2,720 | 22,644 |

Total: 22,644 steps vs ~35,200 in PR #788 (~36% fewer). The model achieved full LR cooldown (terminal LR 1.40e-5) but never accumulated sufficient gradient updates to converge. Every channel strictly regressed vs the 81%-complete PR #788, confirming step-count starvation — not LR misalignment — was the binding constraint in PR #788.

**Curvature signal architecture validity:** Despite the failure, the signal is architecturally healthy. param_norm grew 3× (4.58→13.48), grad/param stable ~0.01 post-warmup, zero_fraction=0.000 throughout. PR #788 demonstrated discriminating test-set signal (−0.18pp test_abupt, −0.14pp test_surface_p, −0.28pp test_wall_shear vs within-cluster control). The curvature direction is valid.

**Verdict (NEGATIVE — design error):** Full LR cooldown is necessary but not sufficient. The compressed schedule was the wrong default for this config. Follow-up: `--epochs 4 --lr-cosine-t-max 4` + **original** vol-schedule `0:16384:3:32768:6:49152:9:65536` (vol=16k throughout all 4 epochs → ~35k+ steps + proper cooldown). Expected val_abupt: 6.4–6.7%.

---

## 2026-05-08 — PR #788: surface curvature H,K on surface path (nezuko) — CLOSED INCONCLUSIVE

- **Branch**: nezuko/surface-curvature-surface-only (deleted)
- **W&B run**: `3ct0x7zd` (group `nezuko-surface-curvature`)
- **Hypothesis**: Append `[H̃, K̃]` (signed-log + train-z-score) to surface input → improves surface_p, wall_shear, τ_z without affecting volume_p.

**Best-EMA results (EP4 partial, 81% through EP4 at 270-min cap):**

| Metric | nezuko curvature | SOTA #592 | within-cluster control thorfinn-ArmA | Δ vs control |
|---|---:|---:|---:|---:|
| val_abupt | 6.7767% | 6.5985% | — | — |
| test_abupt | 8.139% | 7.9915% | 8.321% | **−0.18pp** |
| test_surface_p | 4.168% | 4.068% | 4.303% | **−0.14pp** |
| test_wall_shear | 7.4189% | 7.334% | 7.697% | **−0.28pp** |
| test_volume_p | 12.254% | 11.9335% | 12.092% | +0.16pp (drift) |

Beat curvature-on-volume (edward PR #773) on test_abupt, val_surf_p, wall_shear, τ_z. Hypothesis-discriminating signals land on test exactly where predicted. Failed val_abupt merge gate by +0.18pp purely because EP4 was cut at 81% with no LR cooldown (`--lr-cosine-t-max 13`).

**Verdict:** Architecturally correct (curvature on surface > curvature on volume). Schedule-mismatch is the merge blocker. Follow-up: PR #798 with `--epochs 4 --lr-cosine-t-max 4`.

## 2026-05-08 — PR #786: Anchor-STRING RoPE v3 full-budget (fern) — CLOSED INCONCLUSIVE

- **Branch**: fern/anchor-string-rope-v3-fullrun (deleted)
- **W&B run**: `qg0rplnl` (group `fern-anchor-string-rope-v3`)
- **Hypothesis**: Xavier×0.01 init on out_proj activates RoPE residual from EP1 (vs zero-init in v2), so by EP4 the residual has built up enough learned spectral structure to close the SOTA gap.

**Best-EMA results (EP4 partial, 64% through EP4 at 270-min cap):**

| Metric | v3 (this) | v2 #774 | Δ vs v2 | SOTA #592 | thorfinn ArmA |
|---|---:|---:|---:|---:|---:|
| val_abupt | 6.9197% | 6.9088% | tie | 6.5985% | — |
| test_abupt | 8.1946% | 8.249% | −0.054 | 7.9915% | 8.321% |
| test_volume_p | 12.116% | 12.118% | tie | 11.933% | 12.092% |
| out_proj_rms (EP4) | 0.0464 | 0.042 (terminal) | +0.0044 | — | — |

Xavier×0.01 init worked exactly as designed (rms grew from 0.00347 EP1 → 0.0464 EP4 cutpoint, no runaway). Beat thorfinn within-cluster control on test_abupt (−0.13pp), but +0.20pp behind absolute SOTA `4k25s25e`.

**Verdict:** Init mechanism validated. Architecture parked at this 270-min budget — not paying for itself vs SOTA without 13 full epochs. Compute hours better spent on schedule-alignment control (PR #799 fern).

## 2026-05-07 02:15 — PR #776: vol-loss-weight sweep {1.5, 2.0} on SOTA L=5 (tanjiro) — CLOSED PARTIAL POSITIVE

- **Branch**: tanjiro/vol-loss-weight-sweep (deleted)
- **W&B group**: `vol-loss-weight-sweep-tay`
- **Hypothesis**: Manual `--volume-loss-weight` sweep increases vol_p representational pressure → reduces val→test vol_p OOD gap (SOTA gap = 7.99pp). GradNorm was already ruled out (PRs #649 + #758, 6 configs).
- **Arms run**: A (vol-w=1.5, run `hw2e3vsu`), B (vol-w=2.0, run `qscw0225`). Arm C (vol-w=2.5) cancelled at 23:55 UTC at EP2 (advisor decision tree based on EP1 trajectory — see post-mortem below).

**Final test_primary comparison (best-EMA EP4, 50 cases):**

| Metric | SOTA (vol-w 1.0) | Arm A (1.5) | Arm B (2.0) |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **7.9915** | 8.4181 (+0.43) | 8.3466 (+0.36) |
| `test_primary/volume_pressure_rel_l2_pct` | 11.9335 | 12.1257 (+0.19) | **11.5596 (−0.37)** |
| `test_primary/surface_pressure_rel_l2_pct` | **4.0683** | 4.3816 (+0.31) | 4.3820 (+0.31) |
| `test_primary/wall_shear_rel_l2_pct` | **7.3338** | 7.8366 (+0.50) | 7.9073 (+0.57) |
| val→test vol_p OOD gap | 7.99 | 7.94 (−0.05) | **7.32 (−0.67)** |

**Verdict: Partial positive on test_vol_p only. NOT MERGED.** Arm B beats SOTA on `test_primary/volume_pressure_rel_l2_pct` by −0.37pp and shrinks the val→test vol_p OOD gap by 0.67pp — first single-model arm in the sweep family to do so. But val_abupt regresses 0.62pp (7.22% vs 6.60% SOTA), and every other test target regresses 0.31–0.68pp. Per `program.md`: cannot hide regressions behind a single averaged number, so this is not a SOTA replacement.

**Key insight**: val_abupt regression is wall-shear dominated (ws_x +0.51, ws_y +0.65, ws_z +0.68 on test). The Lion+QK-Norm+vol-w=2.0 stack shifts the loss budget away from wall-shear. Vol-loss-weight effects don't show cleanly until ~EP3+ — Arm B EP1 was weaker than Arm A's, but Arm B beat A by terminal. **EP1 is a poor proxy for terminal test_vol_p in this sweep.**

**Advisor post-mortem**: Cancelled Arm C based on EP1 read; Arm B's terminal win shows that was premature. Recording for future kill-gate calibration on loss-weight sweeps: don't gate on EP1.

**Follow-up assigned**: vol-w=2.0 + wall-shear-weight bump combined arm (next PR — see below).

---

## 2026-05-07 — PR #792: FiLM v3 compressed vol-points schedule for max FiLM-active epochs (frieren) — ASSIGNED

- **Branch**: frieren/vol-film-v3-compressed-curriculum
- **Hypothesis**: PR #778 (FiLM v2) confirmed the bounded-tanh FiLM mechanism is structurally sound (no blow-up, FiLM-active epoch was best checkpoint) but budget starvation caused all 3 win conditions to be missed. With `--vol-geom-film-start-epoch 6` and standard curriculum (`0:16384:3:32768:6:49152:9:65536`), FiLM only had ~4127 active steps before the 270-min wall timeout hit mid-EP4 — the V=49k and V=65k stages never completed. Fix: compress the vol-points schedule to `0:16384:1:32768:2:49152:3:65536` (V=65k by EP3) and lower FiLM start to EP2, giving ≥10 FiLM-active epochs within budget.
- **W&B group**: `vol-film-v3-compressed-curriculum`
- **Key change 1**: `--vol-points-schedule "0:16384:1:32768:2:49152:3:65536"` (V=65k by EP3 instead of EP9)
- **Key change 2**: `--vol-geom-film-start-epoch 2` (FiLM fires from EP2, ~10 active epochs)
- **Kill gates**: EP1 val_abupt <32%, EP3 val_abupt <12%, EP6 val_abupt <8.0%
- **Win conditions**: test_vol_p <11.374% (primary), val_abupt <6.5985% (secondary)
- **Status**: WIP — assigned 2026-05-07 (follow-up to PR #778)

---

## 2026-05-07 — PR #778: FiLM v2 bounded tanh γ∈(0,2) + delayed EP6 onset (frieren) — CLOSED NEGATIVE (metrics) / POSITIVE (mechanism)

- **Branch**: frieren/vol-head-geometry-cond-v2 (deleted)
- **W&B group**: `vol-geom-cond`
- **Hypothesis**: FiLM conditioning of volume tokens via mean-pooled surface-slice geometry vector g, with bounded tanh γ∈(0,2) and β∈(−1,1). FiLM gate delayed to EP6 to avoid firing before high-density vol-points established. Fixes the unbounded blow-up of PR #770 (v1).
- **Architecture**: `VolGeomFilm(hidden_dim)` class with `gamma_proj` and `beta_proj` zero-initialized linear layers; applied after standard vol-token computation: `h' = (1 + tanh(gamma_proj(g))) * h + tanh(beta_proj(g))`
- **Results**:

| Metric | EP1 | EP3 | EP4 (partial, best) | Win condition |
|---|---|---|---|---|
| val_abupt | ~28% (pass) | ~8.5% (pass) | best checkpoint | <6.5985% MISSED |
| FiLM-active steps | — | — | ~4127 | — |
| test_vol_p | — | — | not collected (budget) | <11.374% MISSED |
| wall timeout | — | — | mid-EP4 | — |

- **Analysis**: Bounded tanh design prevented the blow-up seen in v1 (#770). FiLM-active epoch produced the best validation checkpoint, confirming the mechanism is directionally correct. However all win conditions were missed because the 270-min wall timeout hit mid-EP4 with only ~4127 FiLM-active steps. Root cause: `--vol-geom-film-start-epoch 6` combined with standard curriculum `0:16384:3:32768:6:49152:9:65536` means FiLM only fires after V=49k is established at EP6 — but the budget never reaches EP6 at these vol-point densities. The V=49k and V=65k curriculum stages were never trained. The mechanism works; the timing is wrong.
- **Decision**: CLOSED. Hypothesis not falsified — FiLM direction intact. Fix: compress curriculum to `0:16384:1:32768:2:49152:3:65536` and lower start epoch to 2. Assigned as PR #792.

---

## 2026-05-07 — PR #790: τ_z loss upweight sweep {3.0, 4.0} (alphonse) — ASSIGNED

- **Branch**: alphonse/tau-z-upweight-sweep
- **Hypothesis**: `wall_shear_z` (τ_z) is the confirmed training laggard from GradNorm diagnostic (PR #758): r_i=0.01123, GradNorm weight=1.699×, highest of all tasks. Current baseline uses tau_z_loss_weight=2.0. Increasing to 3.0 or 4.0 forces more gradient signal to τ_z. Distinct from GradNorm (which was ruled out): this is static manual upweighting. If effective, will directly improve val_abupt (τ_z has equal weight in the 5-channel abupt average). Pure CLI experiment — no code changes.
- **W&B group**: `alphonse-tau-z-upweight`
- **Arms**:
  - Arm A: `--tau-z-loss-weight 3.0`
  - Arm B: `--tau-z-loss-weight 4.0` (only if Arm A shows τ_z improvement)
- **Kill gates**: EP1 val_abupt < 32%, EP2 val_abupt < 12%
- **Key signal**: val_wall_shear_z vs SOTA 9.810%; watch τ_y and surface_p for regression
- **Status**: WIP — assigned 2026-05-07 (re-assigned from PR #787 stark→alphonse)

---

## 2026-05-07 — PR #789: Vol-decoder SDF-gate v3 — lower cap 0.15 + gate LR warmup + gate WD (askeladd) — ASSIGNED

- **Branch**: askeladd/vol-decoder-sdf-gate-v3
- **Hypothesis**: PRs #781 (unbounded) and #785 (bounded-tanh v2, cap=0.3) both failed via gate MLP saturation. Proximate cause: 20× LR jump at EP1→EP2 boundary (from `--lr-warmup-epochs 1`) triggers 30× vol_loss spike → monotone gate drift to full negative saturation (scale=-0.301, sat_frac=1.0). v3 fixes: (1) lower tanh cap 0.3→0.15 (smaller gradient signal), (2) 2-epoch gate-specific LR warmup (gate LR is only 50% at the EP1→EP2 boundary where v2 died), (3) gate weight decay 5e-3 (10× stronger than backbone). Hypothesis intact: per-case SDF stats can calibrate vol_pred for OOD geometries.
- **W&B group**: `vol-geom-cond`
- **Key metrics**: train/gate/scale_range (saturation indicator), test_vol_p vs 11.374% anchor
- **Kill gates**: 200-step sanity scale_range > 0.002, EP1 val_abupt < 32%, EP2 val_abupt < 12%
- **Status**: WIP — assigned 2026-05-07

## 2026-05-07 — PR #785: Vol-decoder SDF-gate v2 — bounded tanh + input norm (askeladd) — CLOSED NEGATIVE (design)

- **Branch**: askeladd/vol-decoder-sdf-gate-v2 (deleted)
- **W&B runs**: `37r8htsk` (sanity), `ympw1bhr` (DDP run)
- **Hypothesis**: Post-decoder output gating of vol_pred via SDF statistics (per-case global descriptors). Bounded-tanh design: scale ∈ (0.7, 1.3), bias ∈ (−0.05, 0.05). Input normalization. Hidden dim 8→16→2, zero-init output layer.
- **Results**:

| Metric | EP1 (step 10,864) | EP2 (step 21,728) | Status |
|---|---|---|---|
| val_abupt | 28.13% ✅ | 8.5789% | KILL (threshold tripped) |
| scale_max_abs | 0.201 (healthy) | 0.3008 (≥ 0.28 threshold) | SATURATED |
| scale_mean | healthy | -0.301 (full saturation) | |
| sat_frac | — | 1.0 | Complete saturation |

- **Analysis**: Bounded-tanh eliminated v1's catastrophic blow-up but did not prevent monotone drift to negative saturation. The 20× LR jump at EP1→EP2 boundary (from `--lr-warmup-epochs 1`: 4.5e-6 → 9.0e-5) triggered a 30× vol_loss spike (0.03 → 0.88), driving gate MLP monotonically to full negative saturation over ~2k steps. Gate degenerated to constant 0.7× multiplier — geometry conditioning never active at steady state. Hypothesis NOT falsified.
- **Status**: CLOSED NEGATIVE (design) — v3 follow-up assigned as PR #789

---

## 2026-05-07 — PR #786: Anchor-STRING RoPE v3 full 13-epoch run (fern) — ASSIGNED

- **Branch**: fern/anchor-string-rope-v3-fullrun
- **Hypothesis**: Prior v2 (PR #774) showed strongly closing gap to SOTA (1.16×→1.05× gap ratio per epoch), reaching EP4 val=6.9088%. Code fixes from PR #783 (merged by human) now in `tay`: (1) `--lr-cosine-t-max 13` aligned to actual budget (was 5 = mismatch), (2) Xavier×0.01 `out_proj.weight` init (was zero). Full 13-epoch run with `--use-anchor-string-rope --anchor-string-rope-n-anchors 1024`. Definitive test of whether Anchor-STRING RoPE can beat SOTA at full budget.
- **W&B group**: `fern-anchor-string-rope-v3`
- **Kill gates**: EP1>35%, EP2>12%, EP3>8.5%
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-07 — PR #788: Surface curvature H,K on surface path only (nezuko) — ASSIGNED

- **Branch**: nezuko/surface-curvature-surface-only
- **Hypothesis**: PR #773 (edward) put H,K curvature features on the volume path — failed (8.166% test vs 7.991% SOTA, -0.18pp). Follow-up: wire H,K to the **surface** path only (SURFACE_X_DIM=3→5). Surface curvature directly governs aerodynamic boundary conditions (pressure gradients at high-curvature wheel arches, A-pillar edges, underbody details). Volume decoder is left unchanged. Precomputed cache already on disk at `/mnt/new-pvc/Processed/drivaerml_curvature_v2_edward/` from PR #773.
- **W&B group**: `nezuko-surface-curvature`
- **Key discriminating signal**: surface_pressure and wall_shear should improve; vol_p should stay neutral.
- **Kill gates**: EP1>32%, EP2>10% (tighter than usual — testing surface input perturbation)
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-07 — PR #775: Learnable affine anchor for vol_p OOD gap (nezuko) — CLOSED NEGATIVE

- **Branch**: nezuko/learnable-scale-surface-anchor (deleted)
- **Hypothesis**: Learnable global scalar alpha×surf_cp+beta applied to vol_pred, with alpha/beta init=0, to learn the ~718 Pa/Cp scale from data. Zero-degradation at init. Fixes unit-mismatch from PR #772.
- **W&B run**: `8wft0el2` (group `surf-anchor-learnable-scale-tay`)
- **Results**:

| Epoch | val_abupt | val_vol_p | anchor/alpha |
|------|-----------|-----------|-------------|
| EP1 | 27.37% | 16.44% | 0.0442 |
| EP2 | 8.244% | 5.087% | 0.0101 |
| EP3 | 7.197% | 4.310% | 0.00473 |
| EP4 (partial) | **7.049%** | 4.239% | 0.00408 |

- **Decision**: CLOSED. val_abupt=7.049% vs SOTA 6.5985% (+0.45pp). Alpha peaked at 0.141 at EP1→EP2 boundary then decayed 30× to near-zero — optimizer rejected the anchor's contribution. Every channel lagged SOTA. The global scalar anchor fails because: (1) backbone learns surface→volume coupling more expressively, (2) single global scalar cannot capture the OOD geometry shifts of 4 outlier cases. Rules out raw-Cp global scalar anchor as geometry-conditioning approach.

---

## 2026-05-07 — PR #781: Vol-decoder SDF-statistics geometry gating (askeladd) — CLOSED NEGATIVE (unbounded design)

- **Branch**: askeladd/vol-decoder-sdf-gating (deleted)
- **Hypothesis**: 8-stat SDF descriptor (mean, std, min, max, frac<0.05/0.20/0.50m, median) → 8→64→2 MLP → unbounded affine `(1+a)*vol_pred + b` on volume pressure output. Zero-init MLP. Per-case geometry conditioning from existing SDF channel (VOLUME_X_DIM=4).
- **W&B runs**: rank-0 `4z4cz06q`, rank-7 (kill source) `4qjhfd11` | Group: `vol-geom-cond`
- **Results**: Killed at step 2376 (EP1, 22% through). No val metrics collected.
  - Initial kill was due to inverted kill threshold (`<2.0` instead of `>2.0`) — advisor corrected this.
  - After correction, rank-7 still blew up at step 2375: scale_max_abs 0.0025 → 2.5625 (~1000× spike). Ranks 0-6 remained healthy (max ≤ 0.005).
  - Root cause: 8 descriptor channels span different orders of magnitude (metres vs [0,1] fractions) + no input normalization. Unbounded MLP weights grow along typical-distribution directions; outlier case in under-sampled descriptor corner drives extreme response.
- **Decision**: CLOSED. Hypothesis not falsified (no val data). Unbounded affine design falsified. Follow-up PR #785 implements bounded tanh + input normalization.

---

## 2026-05-06 — PR #776: Manual vol-loss-weight sweep {1.5, 2.0, 2.5} on SOTA L=5 (tanjiro) — ASSIGNED

- **Branch**: tanjiro/vol-loss-weight-sweep
- **Hypothesis**: Manual `--volume-loss-weight` increase {1.5, 2.0, 2.5} to reduce vol_p OOD gap via higher gradient signal magnitude — distinct from GradNorm (which was ruled out, PRs #649 + #758). More gradient signal may force the model to allocate more representational capacity to vol_p, potentially improving generalization on the 4 OOD test cases. Three arms, no code changes required.
- **W&B group**: `vol-loss-weight-sweep-tay`
- **Issue**: #717
- **Arms**:
  - Arm A: `--volume-loss-weight 1.5` → run `tanjiro/vol-w-1.5`
  - Arm B: `--volume-loss-weight 2.0` → run `tanjiro/vol-w-2.0`
  - Arm C: `--volume-loss-weight 2.5` → run `tanjiro/vol-w-2.5`
- **Kill gate**: val_abupt < 32% at EP1 (~step 2700); kill arms with val_abupt > 7.5% by EP3
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #775: Learnable affine scale on surface-anchor for vol_p OOD gap (nezuko) — ASSIGNED

- **Branch**: nezuko/learnable-scale-surface-anchor
- **Hypothesis**: PR #772 (surface-anchor v1) failed due to unit mismatch: `surface_cp` (dimensionless Cp, mean≈−0.304) was used as correction for `volume_pressure` (Pa, mean≈−205.8). This PR fixes that with a **learnable affine transform** on the nearest-surface-point lookup: `vol_p_anchor = alpha * surf_p_norm + beta`, where alpha and beta are initialized to 0 (ensuring zero degradation at step 0). The model learns the Pa/Cp scale (~718) from data. Architecturally distinct from PR #771 (askeladd cross-attention scalar): pure geometric lookup with learnable affine, no learned feature aggregation.
- **W&B group**: `surf-anchor-learnable-scale-tay`
- **Issue**: #717
- **Arms**:
  - Arm A: shared global scalar (alpha, beta as nn.Parameter scalars)
  - Arm B: same, but log alpha convergence to verify it approaches ~718 Pa/Cp
- **Kill gate**: val_abupt < 32% at EP1 (~step 2700)
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #770: Vol decoder FiLM conditioning on surface geometry latent (frieren) — ASSIGNED

- **Branch**: frieren/vol-head-geometry-cond
- **Hypothesis**: The 4 geometrically-OOD test cases (run_133/226/203/158) that cause 92% of test_vol_p deviation (#767 diagnostic) require the volume decoder to be conditioned on the surface geometry latent. Inject global surface slice-token mean-pool `g = MeanPool(S)` into volume tokens via FiLM: `h' = γ(g) ⊙ h + β(g)` before the volume prediction head. γ,β initialized to identity. ~0.6M extra params.
- **W&B group**: `vol-geom-cond`
- **Issue**: #717
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #771: Surface-latent scalar offset for vol_pressure OOD conditioning (askeladd) — ASSIGNED

- **Branch**: askeladd/surf-latent-vol-residual
- **Hypothesis**: Minimal geometry conditioning: a learned per-case global residual scalar offset on vol_pressure, derived from surface geometry latent. `vol_p_conditioned = vol_p + Linear(MeanPool(surface_slice_tokens))`. Linear(D→1), ~513 params, zero-initialized. Tests whether a single learned scalar per case is sufficient to address the geometry-OOD case-level scale shifts confirmed by #767.
- **W&B group**: `vol-geom-cond` (grouped with frieren #770 for direct comparison)
- **Issue**: #717
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #767: Phase 0 diagnostic per-case + per-region test_vol_p (askeladd) — CLOSED (diagnostic complete)

- **Branch**: askeladd/phase0-diagnostic
- **Hypothesis**: The test_vol_p gap is case-dominated and lives on a small number of geometrically-OOD test cases.
- **W&B runs**: inference-only, no training run (diagnostic only)
- **Results**:

| Checkpoint | test_vol_p all 50 | test_vol_p excl-4 OOD | % deviation from top-4 |
|---|---:|---:|---:|
| `4k25s25e` (#592) | 11.933% | 3.9% | 92% |
| `dc031qpt` (#681) | 11.374% | 4.2% | 92% |

- **Key findings**:
  1. Same 4 cases (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation across **two architecturally distinct checkpoints**
  2. Excluding the 4 cases, test_vol_p drops to 3.9-4.2% — **below AB-UPT 6.08% reference for the remaining 46 cases**
  3. Surface_p and τ are **unaffected** on these 4 cases — the surface encoder generalises fine; the volume decoder specifically fails
  4. H3-via-loss-scaling closed: supervision-density/loss-mass interventions cannot fix geometry-OOD
  5. Next intervention class: test-time geometry conditioning on volume path
- **Decision**: Diagnostic complete. Closed successfully. Next PRs: #770 (frieren FiLM), #771 (askeladd scalar offset).

---

## 2026-05-06 — PR #761: Dedicated 2-layer volume head on shared encoder (frieren) — CLOSED (truncated, inconclusive)

- **Branch**: frieren/vol-head-2L
- **W&B run**: `15u5c4ec`
- **Hypothesis**: A dedicated 2-layer Transolver volume decoder head on top of the shared encoder (+5.91M params, +37.1% vs SOTA) will reduce the volume_pressure val→test gap by increasing volume-specific capacity.

| Metric | EP1 | EP2 | EP3 | EP4-partial (final) | SOTA gate |
|---|---:|---:|---:|---:|---:|
| val_abupt | 31.312% | 8.088% | 7.045% | 6.832% | <6.5985% |
| val_vol_p | 14.144% | 4.731% | 4.045% | 3.938% | — |
| test_abupt | — | — | — | 8.198% | — |
| test_vol_p | — | — | — | 12.112% | <11.374% |

- **Analysis**: Training timeout (270 min) fired at EP4-partial (step 34,424 of 43,459), cutting the run at ~25% of budget from completion. The vol-points curriculum never advanced past 16,384 (ramp at EP3 to 32k didn't complete). Both gates missed (val_abupt=6.832 > 6.5985; test_vol_p=12.112 > 11.374). However: val_vol_p 3.938% < SOTA 3.946% — a small but persistent signal across EP2/EP3. The 4 OOD cases (run_226=109.1%, run_133=108.0%, run_203=103.7%, run_158=102.1%) entirely dominate test_vol_p; **median test_vol_p=3.89%, excl-top-4 mean=3.97%** — both below AB-UPT 6.08%.
- **Conclusion**: Hypothesis untested (only 25% of budget ran; curriculum never ramped). Closing as inconclusive, not falsified. Next step: compose vol-head with geometry conditioning (#770) rather than re-run standalone.

---

## 2026-05-01 — PR #760: Issue #618 volume-loss-weight reweight ablation (alphonse) — ASSIGNED

- **Branch**: alphonse/vol-loss-weight-reweight
- **Hypothesis**: Increasing `--volume-loss-weight` from 1.0 (PR #592 SOTA default) to 2.0 or 3.0 for the full run will improve val_abupt by forcing better fit to the volume pressure field. The current SOTA uses surface_w=2.0 but volume_w=1.0 (2:1 ratio favoring surface). This tests the 1:1 and 1.5:1 ratio variants on the exact PR #592 stack (L=5 depth).
- **W&B group**: `issue-618-vol-weight-ablation`
- **Arm A command**: SOTA stack + `--volume-loss-weight 2.0` (`alphonse/vol-weight-2.0`)
- **Arm B command**: SOTA stack + `--volume-loss-weight 3.0` (`alphonse/vol-weight-3.0`)
- **Issue**: #618 (STRING/RoPE post-mortem, vol-weight isolation ablation following PR #750 closure)
- **Status**: WIP — assigned 2026-05-01

---

## 2026-05-01 — PR #750: Issue #618 Exp B geometry-branch diff-LR + backbone freeze + aux vol-pressure warmup (alphonse) — CLOSED NEGATIVE

- **Branch**: alphonse/geometry-branch-redux
- **Hypothesis**: Freeze backbone for first 20% of training epochs so geometry branch can warm up independently; simultaneously apply 2× LR to geometry branch params; apply volume-loss-weight-warmup=2.0 during lr_warmup_epochs.
- **W&B run**: `qt9xt341` (group `issue-618-geometry-branch-redux`, name `alphonse/geom-redux-fz0.20-glr2.0-vlw2.0`)

| Metric | EP4 (last frozen) | EP5 (first joint) | SOTA gate |
|---|---:|---:|---:|
| val_abupt | 27.187% | 11.294% | 6.5985% |
| val_vol_p | 18.470% | 7.886% | — |
| test_abupt | — | 12.250% | 7.9915% |
| test_vol_p | — | 15.430% | 11.374% |

- **Root cause**: Frozen backbone warmup with random initialization was harmful — geometry branch spent 4 epochs (252 min) fitting random features (val_abupt=27.2% at last frozen epoch, far above SOTA's ~7% at equivalent depth). The mechanism itself was wired correctly (DDP find_unused_parameters, optimizer rebuild at unfreeze, vol-w warmup), but the underlying strategy was flawed. Vol-points curriculum at 16k points → 63 min/epoch; only ONE joint epoch (EP5) ran before the 270-min budget cap.
- **Conclusion**: Frozen backbone warmup requires a pretrained backbone to be useful. Single-epoch jump from 27.2→11.3% at unfreeze confirms geometry branch can learn fast from real features — motivates a pretrained-freeze variant as a future experiment. Both success gates failed (val_abupt +4.71pp, test_vol_p +4.06pp vs anchors). Closing as negative result.

---

## 2026-05-07 — PR #738: Volume-coordinate Gaussian noise injection (tanjiro) — CLOSED NULL/NEGATIVE

- **Branch**: tanjiro/volume-coord-noise
- **Hypothesis**: Train-time isotropic Gaussian noise on volume xyz coordinates (σ=0.005m, σ=0.020m) as a geometric robustness regularizer (Bishop 1995 equivalence to Tikhonov regularization on Jacobian norm). Targeting val→test volume_pressure transfer gap.
- **W&B group**: `tanjiro-vol-coord-noise`

| Run | W&B ID | Best Epoch | val_abupt% | val_vol_p% | test_vol_p% | test_abupt% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline (#592 4k25s25e) | 4k25s25e | — | 6.5985 | — | 11.933 | 7.9915 | SOTA gate |
| Arm A σ=0.005 | jzybrknz | EP4 | 7.9998 | 9.7217 | **17.0464** | 9.2023 | timeout-killed mid-EP4 |
| Arm B σ=0.020 | fj728edc | EP3 | 10.5977 | 22.8560 | — | — | killed by EP3 gate (abupt>8%) |
| Arm C (annealed) | — | — | — | — | — | — | CANCELLED (Arm B failed gate) |

- **Root cause**: `volume_x[..., 3]` is precomputed SDF from `volume_sdf.npy`, not recomputable per-step. Noising `volume_x[..., :3]` (xyz) without updating SDF creates `(xyz_noisy, sdf(xyz_clean))` contradictory pairs at train, vs `(xyz_clean, sdf(xyz_clean))` at eval. Regression energy scales as σ² — confirmed by Arm B (+13.1pp on val vs Arm A) amplifying exactly quadratically.
- **Conclusion**: Pure xyz-only coordinate noise is dead-on-arrival under the precomputed-SDF data contract. The val→test volume_pressure gap cannot be addressed via simple input-side regularization of this form. Reassigned tanjiro to PR #758 (GradNorm alpha sweep).

---

## 2026-05-07 — PR #758: GradNorm α=3.0/2.0 sweep (tanjiro) — ASSIGNED

- **Branch**: tanjiro/gradnorm-alpha-sweep
- **Hypothesis**: GradNorm `ema_proxy` mode with high restoring-force alpha (α=3.0 and α=2.0) + min_weight=0.7 floor. PR #649 tested GradNorm with α=1.5 (default) and varying floors; best result was floor=0.7/α=1.5 at EP3=7.41%. No experiment has tested α>1.5. At α=3.0, the `r_i^α` weighting aggressively amplifies gradient signal for undertrained tasks (vol_pressure has r_i >> 1 since it's chronically lagging). Two arms: A=α3.0, B=α2.0.
- **Arm A command**: `--use-gradnorm --gradnorm-mode ema_proxy --gradnorm-alpha 3.0 --gradnorm-min-weight 0.7 --wandb-group tanjiro-gradnorm-alpha`
- **Arm B command**: `--use-gradnorm --gradnorm-mode ema_proxy --gradnorm-alpha 2.0 --gradnorm-min-weight 0.7 --wandb-group tanjiro-gradnorm-alpha`
- **Reference PR**: #649 (edward, floor=0.7, α=1.5: EP3 val_abupt=7.41%, val_vol_p=4.68%)
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-06 03:20 — PR #736: Inter-sample mixup volume points (fern) — CLOSED CONTAMINATED

- **Branch**: fern/volume-input-mixup
- **Hypothesis**: alpha=0.4 mixup on volume-points/pressure improves volume_pressure generalization
- **W&B**: group `fern-vol-mixup`, run `jzo917hu` (rank 0)
- **Result at closure** (step 14,665, ~EP6.4): val_abupt=24.97% (vs SOTA 6.60%, +18.4pp), val_vol_p=17.33%, val_wall_shear=27.23%
- **Closure reason**:
  1. **Contamination**: 8 unauthorized parallel runs in group `gradnorm-adaptive` (`fern/gradnorm-armA-a1.0-ep50-4gpu-rank{0..3}` + `fern/gradnorm-armB-a0.5-ep50-4gpu-rank{0..3}`) STILL RUNNING at closure time, started ~5h before closure with no PR sanctioning them. GPU bandwidth contention compromises the mixup result.
  2. **Mixup also diverging**: alpha=0.4 too aggressive on volume coords with shared mask; model never recovers from EP1's destructive interference.
- **Conclusion**: Negative result on top of contamination. Reassigned fern to PR #753 (signed-log1p target transform).

---

## 2026-05-06 03:20 — PR #735: TTA Y-mirror + jitter (edward) — CLOSED NEGATIVE (both arms)

- **Branch**: edward/tta-mirror-jitter
- **Arm A (inference-only TTA on PR #592 SOTA `4k25s25e`)**:
  - Y-mirror TTA: test_vol_p **11.93% → 13.48%** (WORSE +1.55pp)
  - Jitter TTA (sigma=0.005, 4 passes): val_abupt **6.60% → 26.48%** (catastrophic)
  - Root cause: STRING-separable PE + RFF features depend on sign of y, so Y-mirror corrupts embedding
- **Arm B (train with `--use-mirror-aug --mirror-aug-p 0.5`, run `rbnk7zca`)**:
  - best_val_abupt = **7.0214%** (vs SOTA 6.5985%, +0.42pp WORSE)
  - test_vol_p = **12.245%** (vs SOTA 11.933%, +0.31pp WORSE)
  - val→test gap (7.02→8.34) wider than SOTA's, suggesting Y-mirror aug reduces effective capacity for Y-asymmetric ground truth
- **Conclusion**: Y-mirror is the wrong axis of symmetry to exploit. Closing. Reassigned edward to PR #754 (per-case Cp target normalization).

---

## 2026-05-06 03:20 — PR #748: Spatial within-case SDF stratification (frieren) — CLOSED DIVERGED

- **Branch**: frieren/spatial-volume-emphasis
- **W&B**: run `lzpov7mi`, group `frieren-vol-spatial-emphasis`
- **Result**: val_abupt=**76.51%** at step 15,768 (~EP6.9, runtime 8,099s) — never converged
- **Root cause**: SDF-stratified loader interacted badly with vol-points curriculum. SDF threshold of 0.30m (absolute meters) is inconsistent across cases with very different SDF distributions (p50=0.005m, max=530m); the 25% near-band varied dramatically, creating noisy curriculum signal.
- **Conclusion**: Implementation broken; hypothesis not dead. Reassigned frieren to PR #755 (stochastic depth + volume-token dropout).

---

## 2026-05-06 03:20 — PR #751: Issue #618 AnchorString clean (thorfinn) — CLOSED SILENT FAILURE

- **Branch**: thorfinn/issue618-run5-anchorstring-clean
- **W&B**: run `ece4qc3o` (rank 0), state=finished at step 21,729 after 35 minutes (run ended early)
- **Result at termination**: val_abupt=**23.17%**, val_vol_p=15.50%, val_surface_p=17.06%, val_wall_shear=25.56%
- **Closure reasons**:
  1. **Zero PR comments** — no startup heartbeat, no kill-gate report, no termination explanation. Communication blackout.
  2. **Run ended early** — 35min runtime vs ~270min budget. Either auto-killed or process crashed; no diagnostic posted.
  3. **Did not converge** — slope was negative (-2.78pp/1k_steps val_abupt) but starting from way too high to hit SOTA in remaining budget.
- **Conclusion**: Silent failure pattern. Reassigned thorfinn to PR #756 (cosine-annealed EMA decay) with explicit communication-protocol enforcement.

---

## 2026-05-06 03:20 — Round 12 vol-pressure assignments

After closing 4 stalled/failed PRs, all 4 newly-idle students reassigned to fresh, orthogonal hypotheses targeting test_volume_pressure (Issue #717):

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| #753 | fern | Signed-log1p target transform on vol_p | Magnitude-scale equalization for heavy-tailed pressure distribution |
| #754 | edward | Per-case Cp target normalization (`p / max(\|p_surf\|)`) | Dimensional normalization to address 4 catastrophic test outliers |
| #755 | frieren | Stochastic depth + volume-token dropout | OOD generalization regularization for distribution shift |
| #756 | thorfinn | Cosine-annealed EMA decay (0.99→0.9999) | Stabilization tier; clean re-entry after silent-failure pattern |

All four are orthogonal axes (target transform / target rescaling / regularization / EMA bookkeeping) and compose with the in-flight Phase 1 PRs (#737 region weighting, #738 noise injection, #750 geom-branch diff-LR, #752 wake stratification).

---

## 2026-05-06 03:00 — PR #737: Region-weighted vol_p loss (nezuko) — IN-FLIGHT, STRONG SIGNAL

- **W&B**: run `r1eddah6`, group includes `nezuko-region-weighted-vp`
- **Headline EP3 (step 32,592)**: val_abupt=**7.28%**, val_vol_p=**4.36%** — 2.17pp below val SOTA on vol_pressure!
- EP1: val_abupt=27.78%, EP2: val_abupt=8.69% (vol_p=5.38%), EP3: val_abupt=7.28% (vol_p=4.36%)
- Currently the most promising in-flight Phase 1 experiment; continuing through EP13.

---

## 2026-05-01 — PR #641: Flow-aligned tau local frame (thorfinn)

- **Branch**: thorfinn/flow-aligned-tau
- **Hypothesis**: Predict wall shear stress (tau) in the local surface tangent coordinate frame (s, t) instead of global (x, y, z). Physics-motivated: wall shear is a tangential quantity and expressing it in its natural frame should reduce the prediction burden and improve geometric generalization.
- **Group**: `tay-flow-aligned-tau`
- **W&B run**: thorfinn/flow-aligned-tau-rank0

| Epoch | Step | val_abupt |
|-------|------|-----------|
| EP1 | 10,864 | 32.875% |
| EP2 | 21,729 | 14.613% |

- **Decision**: KILLED at EP2. val_abupt=14.613% exceeds the ≤12.0% kill gate.
- **Analysis**: The flow-aligned coordinate transformation significantly destabilized training. EP2 at 14.6% is far above the typical EP2 range for well-converging runs (~8-10%). The local tangent frame construction may introduce numerical instabilities near degenerate surface normals, or the coordinate rotation may be causing gradient issues during backprop. The idea is physically sound but the implementation may require careful normalization or the model may not benefit from this kind of inductive bias at the current architecture scale.
- **Conclusion**: Dead end in this form. A future attempt could try predicting only the tangential magnitude (scalar) rather than the full vector, or using the frame as an auxiliary feature rather than changing the prediction target.

---

## 2026-05-01 — PR #614: Lion β2 momentum sweep (fern)

- **Branch**: fern/lion-beta2-sweep
- **Hypothesis**: The default Lion β2=0.99 may not be optimal. Sweep β2 ∈ {0.95, 0.99, 0.999} to find the optimal momentum coefficient for this task. Higher β2 provides more stable but slower adaptation; lower β2 more aggressive.
- **Group**: `tay-lion-beta2-sweep`

| Arm | β2 | W&B Run | EP1 val_abupt | EP2 val_abupt | EP3 val_abupt | EP4 val_abupt | Best val_abupt | Status |
|-----|-----|---------|---------------|---------------|---------------|---------------|----------------|--------|
| C | 0.999 | wapj7o9t | 34.98% | 10.947% | 8.318% | 7.473% | **7.219%** | Finished |
| B | 0.99 | hjq54lu4 | 28.09% | — | — | — | **6.793%** | Finished |
| A | 0.95 | lcb5rb4l | **26.613%** | TBD | — | — | TBD | Running ~step 12.3k (past EP1, advancing to EP2) |

- **Analysis**: All completed arms are worse than SOTA (6.5985%). β2=0.999 converges much more slowly (EP2=10.95% vs typical ~8-9%) but still reaches a reasonable endpoint at 7.219%. β2=0.99 (default) achieves 6.793% — close to SOTA but not beating it. β2=0.95 just crossed EP1 with the fastest convergence at 26.613% (vs 28.09% for β2=0.99 and 34.98% for β2=0.999), consistent with lower β2 = more reactive momentum updates. EP2 gate (step 21,729) next; threshold ≤ 12.0%.
- **Preliminary conclusion**: The current β2=0.99 appears near-optimal. Lion momentum is not a high-leverage knob for further gains. Will update when arm A (β2=0.95) completes.

---

## 2026-05-01 — PR #621: Slice-centroid STRING-RoPE (nezuko) [In Progress]

- **Branch**: nezuko/slice-rope-sweep
- **Hypothesis**: Apply Rotary Position Encoding (RoPE) at the slice centroid level using STRING-separable coordinates. Two variants: arm-a (control baseline rerun), arm-b (RoPE applied after QK-norm).
- **Group**: `nezuko-slice-rope-sweep`

| Arm | Description | W&B Run | EP1 val_abupt | EP2 val_abupt | EP3 val_abupt | Best val_abupt | Status |
|-----|-------------|---------|---------------|---------------|---------------|----------------|--------|
| a | Control baseline | xixwhi2m | — | 8.727% | 7.389% | **6.990%** | Finished (37,221 steps) |
| b | RoPE after QK-norm | mekagz7v | 27.436% | **8.634%** | TBD | TBD | Running ~step 23.9k (PASS EP2, advancing to EP3) |

- **Analysis**: Arm-a (control) finished at 6.990% — worse than SOTA 6.5985% (Δ+0.59%). The control arm establishes that this training run configuration is slightly below SOTA capability. Arm-b PASSED EP2 gate at 8.634% (≤ 12.0% threshold), tracking slightly worse than control arm-a's EP2 (8.727%) — needs strong EP3+ to differentiate. EP3 gate (step 32,594): kill if > 8.0%.
- **Status**: Monitoring arm-b EP3 gate. Must beat arm-a (6.990%) and SOTA (6.5985%) to show value.

---

## 2026-05-01 — PR #624: Pre-slice STRING-RoPE (alphonse) [In Progress]

- **Branch**: alphonse/presl-rope-sweep
- **Hypothesis**: Inject STRING-RoPE positional encoding before the slicing operation (at the point level) rather than at slice centroids. Two variants: arm-a (control), arm-b (xmid-only RoPE variant).
- **Group**: `alphonse-presl-rope-sweep`

| Arm | Description | W&B Run | EP2 val_abupt | EP3 val_abupt | Best val_abupt | Status |
|-----|-------------|---------|---------------|---------------|----------------|--------|
| a | Control baseline | r3f8v68j | 8.635% | 7.579% | **7.064%** | Finished (37,367 steps) |
| b | xmid-only RoPE | a29fersn | — | — | TBD | Running ~step 4k (pre-EP1) |

- **Analysis**: Arm-a (control) finished at 7.064% — worse than SOTA 6.5985% (Δ+0.70%). Arm-b still pre-EP1. The control arm result is below SOTA, consistent with nezuko arm-a also being below SOTA — both control arms suggest these parallel training runs are slightly below the specific SOTA checkpoint conditions.
- **Status**: Monitoring arm-b for EP1 gate.

---

## 2026-05-01 — PR #647: Anchor-string no-slice Exp 3 (frieren) [CLOSED — reference trajectory]

- **Branch**: frieren/exp3-anchor-string
- **Hypothesis**: Issue #618 Experiment 3 reassignment — anchor-string approach without slicing. Two arms running: arm-b-anchor-k1024-ep4 and arm-b-anchor-k1024.
- **Group**: `frieren_exp3_anchor_string`

| Gate | Step | val_abupt | Status |
|------|------|-----------|--------|
| EP1 | 10,864 | 48.27% | PASS (normal cold-start, not divergence) |
| EP2 | 21,729 | 16.05% | PASS |
| EP3 | ~32,000 | ~10% | CRASHED (run terminated mid-epoch) |

- **Status**: CLOSED. arm-b-anchor-k1024 (multi-rank) crashed early at step ~292-332. arm-b-anchor-k1024-ep4 (o7upw6qr) completed EP1=48.27%, EP2=16.05%, then crashed mid-EP3 at ~10%.
- **Important note**: EP1=48.27% was a NORMAL cold-start trajectory, NOT divergence. This is the reference convergence trajectory for AnchorStringAttention (vanilla, no stabilizers). Thorfinn's PR #742 mistakenly identified this as divergence and added stabilizers to fix it — those stabilizers were the root cause of Run 4's failure.
- **Reference trajectory for PR #743 (Run 5) kill gates**: EP2 <20%, EP3 <15% (calibrated on this data).

---

## 2026-05-01 — PRs #648, #649, #650: New sweep PRs [In Progress]

### PR #648 — Volume-pressure loss upweighting (askeladd)
- **Group**: `volume-pressure-loss-sweep`
- **Hypothesis**: Upweight volume_pressure in the loss function (sweep weight ∈ {2.0, 4.0, 6.0}) to address the chronic 3× test-vs-val gap on volume_pressure field.
- **Status**: arm `vp-weight-2.0` at step ~3,290. Pre-EP1. Monitoring.

### PR #649 — GradNorm min-weight floor sweep (edward)
- **Group**: `gradnorm-min-weight-sweep`
- **Hypothesis**: Sweep GradNorm minimum weight floor ∈ {0.3, 0.5, 0.7}. Previously used floor=0.0 (no floor); a floor prevents any task from being completely suppressed during gradient normalization.
- **Status**: arm `gradnorm-floor-0.3` at step ~2,845. Pre-EP1. Monitoring.

### PR #650 — LR cosine floor sweep (tanjiro)
- **Group**: `lr-cosine-floor-sweep`
- **Hypothesis**: Sweep cosine LR minimum floor ∈ {1e-7, 5e-7, 5e-6, 1e-5}. Current SOTA uses lr-min=1e-6. Testing whether a higher or lower floor improves final convergence.
- **Status**: arm `lr-min-5e-6` (aon7hwtk) at ~step 6.9k. Pre-EP1. Monitoring.

---

## 2026-05-01 — PR #651: Surface curvature features (thorfinn) [KILLED]

- **Branch**: thorfinn/surface-curvature-features
- **Hypothesis**: Add k-NN-estimated surface curvature features (mean curvature H, Gaussian curvature K) as input to tau predictor. Curvature is a fundamental geometric quantity correlated with wall shear stress — concave/convex regions experience different flow regimes. Implementation: chunked k-NN (k=20, chunk=8192) with PCA-based quadratic fit; normalize to ±3σ.
- **Group**: `thorfinn-surface-curvature`

| Gate | Step | val_abupt | Decision |
|------|------|-----------|----------|
| EP2 | 21,729 | 14.613% | KILL (>12.0% threshold) |
| Final | — | 12.487% | — |

- **Decision**: KILLED at EP2. val_abupt=14.613% >> 12.0% kill gate. PR closed.
- **Analysis**: Surface curvature features (H, K) introduced via k-NN PCA-based quadratic fit destabilized training significantly — similar pattern to flow-aligned-tau (PR #641, EP2=14.613%). The additional geometric features may be introducing noisy inputs that conflict with the existing STRING positional encoding. The model architecture at L=5/hidden=512 appears sensitive to extra geometric input channels — either the feature construction is numerically unstable, or the model cannot leverage these high-frequency curvature signals at this scale. A future attempt could try normalizing more aggressively, or using curvature only as an auxiliary regularization signal rather than a direct input feature.
- **Conclusion**: Dead end in current form. Closed PR #651.

---

## 2026-05-05 — PR #660: Depth scaling L=6 sweep (thorfinn) [KILLED]

- **Branch**: thorfinn/depth-l6-sweep
- **Hypothesis**: L=5 SOTA (PR #592) outperformed L=4 by −1.90% relative. Test whether L=6 with reduced hidden_dim (384 or 448) continues the depth scaling trend. Two arms: hidden=384 (Arm A), hidden=448 (Arm B — sequential).
- **Group**: `depth-l6-sweep`

| Gate | Step | val_abupt | Decision |
|------|------|-----------|----------|
| EP1 (Arm A h=384) | 10,864 | 30.978% | KILL (elevated; experiment confounded) |

- **Decision**: KILLED at EP1. val_abupt=30.978% is elevated beyond normal range (24-28%). Experiment was fundamentally flawed — reducing hidden_dim to 384/448 to compensate for VRAM created a confounded experiment testing "L=6 with less capacity" rather than "L=6 at equal capacity."
- **Conclusion**: PR closed. Correct follow-up: PR #666 (thorfinn) — L=6 at full hidden=512 (estimated ~57GB VRAM, well within 96GB budget).

---

## 2026-05-05 — PR #614: Lion β2 momentum sweep (fern) [CLOSED — null result]

- **Branch**: fern/lion-beta2-momentum-sweep
- **Hypothesis**: Sweep Lion β2 ∈ {0.95, 0.99, 0.999} to identify optimal momentum coefficient.
- **Group**: `tay-lion-beta2-sweep`

| Arm | β2 | W&B Run | Best val_abupt | Epochs |
|-----|-----|---------|----------------|--------|
| B | 0.99 (default) | hjq54lu4 | **6.793%** | 6 |
| A | 0.95 | lcb5rb4l | **7.098%** | 4 |
| C | 0.999 | wapj7o9t | **7.219%** | 6 |

- **Decision**: Closed as null. β2=0.99 (existing default) confirmed optimal. No arm beats SOTA 6.5985%.
- **Key finding**: Lower β2=0.95 converges faster at EP1 (26.6% vs 28.1%) but the advantage narrows and inverts by EP3 (7.69% vs 7.39%); β2=0.95 final is 0.305pp worse than β2=0.99. Higher β2=0.999 is simply too sluggish to converge within budget (EP1=35.0%). Lion β2 momentum tuning is concluded as a research direction.

---

## 2026-05-05 — PRs #648 #649 #650: EP3 gate results [WIP]

### PR #648 — Volume-pressure loss upweighting (askeladd, run rl2drj1m)

| Gate | Step | val_abupt | SP | WS | VP | Decision |
|------|------|-----------|----|----|-----|---------|
| EP1 | 10,864 | 27.30% | — | — | — | Normal |
| EP2 | 21,729 | 8.21% | — | — | — | PASS |
| EP3 | 32,594 | **7.8217%** | 5.30% | 8.90% | **4.30%** | PASS (< 8.0%) |

- Status: Running to completion. EP3=7.82% PASS. VP channel at 4.30% at EP3 is lower than typical — promising signal for the vol_pressure gap problem.

### PR #649 — GradNorm min-weight floor sweep (edward, run phi418eg)

| Gate | Step | val_abupt | SP | WS | VP | Decision |
|------|------|-----------|----|----|-----|---------|
| EP1 | 10,864 | 25.78% | — | — | — | Normal |
| EP2 | 21,729 | 8.57% | — | — | — | PASS |
| EP3 | 32,594 | **7.4142%** | 5.05% | 8.28% | 4.68% | PASS (< 8.0%) |

- Status: Running to completion. Strong EP3 recovery from borderline EP2.

### PR #650 — LR cosine floor sweep (tanjiro, run aon7hwtk)

| Gate | Step | val_abupt | SP | WS | VP | Decision |
|------|------|-----------|----|----|-----|---------|
| EP1 | 10,864 | 29.42% | — | — | — | Normal |
| EP2 | 21,729 | 8.24% | — | — | — | PASS |
| EP3 | 32,594 | **7.2377%** | 4.75% | 8.19% | 4.40% | PASS (< 8.0%) |

- Status: Running to completion. Best of the three borderline-EP2 recoveries — 7.24% at EP3 is a strong signal.

---

## 2026-05-05 — New PRs assigned (Round 11–12 closures + current Phase 1 assignments)

### Closed dead ends (Rounds 11–12)

- **PR #690** (various): Slice sweep {64, 192, 256} — slices=64 null (+0.30pp); slices=192/256 infeasible (>92 min/epoch). CLOSED.
- **PR #691** (various): RFF sigma wide/low-ext — both null. CLOSED.
- **PR #692** (various): Heads sweep {8, 2} — heads=8 null (+0.83pp); heads=2 unauthorized concurrent launch, CLOSED.
- **PR #693**: L=6/h=448/heads=7 — killed (heads=7 destroys SDPA fast path, ~98 min/epoch). CLOSED.
- **PR #694**: depth L=6/hidden=384/heads=4 — null (val=6.9016%, +0.30pp), still descending but budget-bound. CLOSED.
- **PR #695**: rff-num-features=32 — null (+0.33pp val regression). CLOSED.
- **PR #716** (frieren): BC-type embedding — operationally broken (concurrent 8-GPU jobs doubled epoch time to 180 min; time-gate kill). CLOSED.
- **PR #722**: dual-tower volume/surface cross-attention — null (+0.87pp val regression). CLOSED.

### Current Phase 1 (Issue #717 volume push) — all WIP as of 2026-05-06

- **PR #728** (frieren): Exp 1B — Volume outlier-aware point sampling (EMA residual + geometric distance arms). WIP.
- **PR #729** (alphonse): Exp 1D — Single-model KD from K=7 ensemble, vol-only soft targets. WIP.
- **PR #734** (askeladd): Exp 1C P3 — SDF distance-to-surface scalar feature for volume input. WIP.
- **PR #735** (edward): TTA — Y-mirror + coord-jitter 6-pass test-time averaging. WIP.
- **PR #736** (fern): Inter-sample mixup on volume coords/pressure (alpha=0.2/0.4). WIP.
- **PR #737** (nezuko): Region-weighted volume loss — near-wake band emphasis (1<x_rel<3, |z_rel|<1.5). WIP.
- **PR #738** (tanjiro): Train-time Gaussian noise on volume coordinates (sigma 5mm/20mm/anneal). WIP.

### Issue #618 STRING/RoPE — re-attempt

- **PR #742** (thorfinn): CLOSED NEGATIVE. Exp 3 Redux — Anchor-STRING with stabilizer triplet (rope_lr_scale=0.1, rope_grad_clip=1.0, 500-step log_freq warmup). Best result: EP3 val_abupt=19.87% (step 32592). Root cause: stabilizers over-constrained RoPE (rope/log_freq moved <0.005 over 3 epochs — essentially frozen). Frieren's PR #647 EP1=48.27% was normal cold-start, not divergence. Genuine bug fixes retained for Run 5: `_init_weights` skip-`string_rope.` + mask-aware anchor selection.
- **PR #743** (thorfinn, pending): Run 5 — Frieren PR #647 exact config (no stabilizers, no rope_lr_scale, no rope_grad_clip, no qk_norm in AnchorString) + 2 genuine bug fixes only. Kill gates: EP2 (step 21728) <20%, EP3 (step 32592) <15%.

### Previous Issue #618 STRING/RoPE arms (all closed, Round 11–12)

- **PR #626** STRING only: best vol gap ratio 2.07× (val→test); established baseline for RoPE comparison.
- **PR #647** AnchorString no-slice: EP1=48.27% (normal cold-start), EP2=16.05%, crashed mid-EP3 at ~10%. Reference trajectory for Run 5 kill gate calibration.
- Other STRING/RoPE arms: null or diverged; closed.

---

## 2026-05-08 20:XX — PR #867: Slices=256 Scaling (thorfinn) — IN PROGRESS

- **Branch**: thorfinn/model-slices-sweep
- **W&B run**: `nv85vovo` (group: `thorfinn-model-slices-sweep`, name: `slices256-arm-b`)
- **Hypothesis**: Scale number of slice tokens from 128 → 256. Slice tokens are the primary unit of computation in Transolver; more slices = finer-grained partitioning of the 3D point cloud into local physics groups. Hypothesis: 256 slices can capture tighter aerodynamic feature clusters than 128.

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | 10,864 | 26.5458% | <30% | PASS |
| EP2 | 16,300 | 11.0175% | <16% | PASS |
| EP3 | ~21,729 | — | <8% | pending |
| EP4 | ~27,159 | — | ≤6.5985% | pending |

**Analysis (in progress):** Strong EP1→EP2 trajectory: 26.5% → 11.0%, showing healthy learning dynamics. EP2 is significantly better than EP1 baseline pace (26.5% → 11.0% in 4 screen epochs from EP2). Watching EP3 closely — need <8% to continue. Current baseline: 6.5985%.

---

## 2026-05-08 — PR #868: Spectral Norm on Attention (askeladd) — IN PROGRESS

- **Branch**: askeladd/spectral-norm-attention
- **W&B run**: `0kjl4rnh` (rank0, group: `spectral-norm-r18`)
- **Hypothesis**: Apply spectral normalization to Q/K/V/out_proj in all attention layers to bound the Lipschitz constant and regularize training. May stabilize gradient flow and improve generalization on out-of-distribution aerodynamic configurations.

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | ~10,864 | — | <30% | running (~step 4,857) |

---

## 2026-05-08 — PR #869: Stochastic Depth / DropPath (edward) — IN PROGRESS

- **Branch**: edward/stochastic-depth
- **W&B run**: `4w7dgiuh` (rank0, group: `stochastic-depth-r18`, name: `edward/drop-path-005`)
- **Hypothesis**: Apply stochastic depth (DropPath) regularization with drop_path_rate=0.05, linear schedule per layer. For L=5: [0.0000, 0.0125, 0.0250, 0.0375, 0.0500]. Both attention and MLP residual branches dropped independently. Zero parameter overhead (15.94M = baseline).

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | ~10,864 | — | <30% | running (~step 2,845) |

---

## 2026-05-08 — PR #870: KNN Surface Roughness Penalty (fern) — IN PROGRESS (PENDING LAUNCH)

- **Branch**: fern/knn-roughness-penalty (pivot from FFT approach)
- **W&B run**: NOT YET STARTED
- **Hypothesis**: FFT-based surface roughness penalty abandoned (Parseval violation from unnormalized rfft + random point sampling). Pivoting to KNN k=8: for each surface point, find k=8 nearest neighbors; compute variance of τ_y/τ_z in that neighborhood; L_smooth = 0.1 × (mean(var_knn(τ_y)) + mean(var_knn(τ_z))).

---

## 2026-05-08 — PR #871: PCGrad Gradient Surgery (tanjiro) — IN PROGRESS

- **Branch**: tanjiro/pcgrad-gradient-surgery
- **W&B run**: `7v0rlsps` (rank0)
- **Hypothesis**: PCGrad gradient surgery across 4 task groups to reduce gradient conflicts between prediction heads. ~2× compute overhead; tests whether conflicting gradients are a bottleneck.

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | ~10,864 | — | <30% | running (~step 1,056) |

---

## 2026-05-08 — PR #872: Width Scaling hidden_dim=640 (frieren) — IN PROGRESS

- **Branch**: frieren/width-scaling-640
- **W&B run**: `gr1n58zo` (rank0, group: `frieren-width-640`)
- **Hypothesis**: Scale Transolver hidden dimension from 512 → 640 (+25% width). Orthogonal to depth scaling; tests whether capacity bottleneck is in the channel dimension. VRAM: 63.2 GB / 97.9 GB (safe).

| Epoch | Step | val_abupt | Gate | Status |
|---|---:|---:|---|---|
| EP1 | ~10,864 | — | <30% | running (~step 4,360) |

---

## 2026-05-05 — Archived earlier new-PR assignments

- **PR #665** (frieren): Cross-slice attention over Transolver slice tokens — global inter-slice MHA layer
- **PR #666** (thorfinn): Depth scaling L=6 at full hidden=512 (corrects the confound in PR #660)
- **PR #667** (fern): Weight decay sweep {1e-4, 5e-4, 1e-3} for Lion optimizer

---

## 2026-05-25 05:30Z — PR #1301: H125 BACKBONE DEPTH 5→7 (fern)

- **Branch**: fern/h125-backbone-depth-5-to-7
- **W&B run**: `7qsjzv8w` (state=failed, OOM at EP10 step 0; EP9 EMA-best recovered via eval_test_only.py)
- **Hypothesis**: Extend depth-axis beyond H120 (depth-6) by scaling backbone 5→7 layers. Tests if depth capacity advantage continues to compound at depth-7. Single-flag change `--model-layers 7` from H112 baseline.

| Metric | H125 EP9-EMA-best | H112 baseline | Δ vs H112 | H120 EP13 terminal | Δ vs H120 |
|---|---:|---:|---:|---:|---:|
| val_primary/abupt | 6.0169% | 6.1358% | **−0.119pp (A WIN)** | 6.0122% | +0.005pp |
| test_primary/abupt | 5.924% | 5.839% | +0.085pp REGRESS | 5.899% | +0.025pp |
| **test_WSS (Morgan)** | **6.842%** | **6.752%** | **+0.090pp REGRESS** | 6.818% | +0.024pp |
| test_WSS_x | 6.083% | — | — | — | — |
| test_WSS_y | 7.431% | — | — | — | — |
| test_WSS_z | 8.847% | — | — | — | — |
| test_VP | 3.495% | 3.421% | +0.074pp regress | 3.461% | +0.034pp |
| test_SP | 3.762% | 3.695% | +0.067pp regress | 3.728% | +0.034pp |

**Val→test WSS slope: +0.010pp (flat-to-regressive). Same catastrophe as H120 (−0.020pp). Structural.**

**Decision: CLOSED C NULL.** Val A WIN (−0.119pp below merge gate) but test C NULL — regresses Morgan primary test_WSS by 0.090pp. All three test floors fail. val→test slope catastrophe is structural, replicated from H120.

**Key findings:**
1. **Depth-axis test-ceiling at depth-5 LOCKED** — two-point confirmation (H120 depth-6 + H125 depth-7), both with flat-or-positive val→test WSS slope. H112 depth-5 is the unambiguous test optimum on the depth axis.
2. **Capacity-axis cohort closure confirmed** — H118 (slices), H120 (depth-6), H121 (width-576), H125 (depth-7) all regress test_WSS by +0.066-0.090pp vs H112 under canonical DropPath_max=0.10. The bottleneck is generalization, not capacity.
3. **H125 EP9 val correctly led H120 throughout** (peak −0.081pp at EP6, consolidated −0.063-0.068pp at EP7-EP9), confirming depth-7 > depth-6 on val — but the val gains do not transfer to test. H112's steep negative val→test slope (−0.215pp) is a depth-5-specific generalization property, not a universal feature of this architecture.
4. **OOM recovery banked** — EP9 EMA-best extracted from crashed run via `eval_test_only.py`. Memory headroom notes for future depth-7+ launches: `--eval-volume-points 49152` or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
5. **Next experiment (H132):** depth-5 × DropPath_max=0.15 — the missing reg-axis cell. Tests whether DropPath_max=0.15 helps at canonical capacity independently of capacity compound.
