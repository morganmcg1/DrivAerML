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
