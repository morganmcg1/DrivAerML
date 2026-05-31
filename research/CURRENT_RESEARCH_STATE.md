# SENPAI Research State

**Updated**: 2026-05-31 22:01Z | Branch: `tay` | **SOTA: H314 Student-t ν=4 — MERGED 17:45Z** | Round 4k+3 | **All inference axes closed (TTA K+R, cal 7-fold null after H333+H334, noise-family ν-sweep, weight-space soup α∈[0.5,0.85] H307 monotone-no-cross). Open frontier: (a) training-time SP-floor gap H338/edward; (b) training-time WSS uniform reweight H339/frieren; (c) training-time WSS wz-only reweight H341/fern (per-axis decomposition); (d) composition K×noise-family H336/nezuko (rebase pending); (e) σ-axis at ν=4 H340/tanjiro; (f) multi-checkpoint output-averaging H342/alphonse (output-space ensemble); (g) **thorfinn — pending NEW hypothesis assignment after H307 closure (22:01Z)**.

**Partial result this loop (08:11Z)**: H307 Arm B α=0.5 (model soup of two EP15 seeds + H300 cal) → **test_cal 5.7380 BEATS gate (−0.8bp)**, val_cal 5.9017 misses by +2.3bp. Soup extracts −30.4bp on test (vs ~28bp on single-seed H312) — **mechanistic finding: weight-averaging produces more affine residual structure, so diagonal cal catches more error post-soup**. Thorfinn launched Arm C α=0.75 at 08:14Z (ETA ~14:59Z). α_τz on soup = 0.99160 vs H312 ref 0.99170 — preliminary cross-axis confirmation that α is model-perturbation-stable.

## 2026-05-31 SOTA progression (5 merges now):
1. H295 (K=5+6-res, 15:45Z) → test 5.7679
2. H296 (K=4+8-res, 17:05Z) → test 5.7678 (−0.15bp)
3. H300 (per-channel calibration, 18:48Z) → test_cal 5.7399 (−28bp)
4. H312 (H296+8-res base+cal, 02:30Z) → test_cal 5.7388 (−1.1bp)
5. **H314 (Student-t ν=4 weight noise, 17:45Z) → test_cal 5.7387** (−0.1bp) ← **CURRENT SOTA**

**Finding 'better-raw-better-cal'**: H296 base (val_raw 5.9221) gives lower calibrated test than H285 base (val_raw 5.9275). H300's 28bp initial gain was from the calibration axis itself; H312 adds 1.1bp more by combining the better TTA base with calibration. Calibration on single-model is near saturation — next axes: cross-channel WSS mixing (H323), regional spatial calibration (H313 in-flight), quadratic terms (H317 in-flight).

---

## Current SOTA (H314 calibrated)

| Model | val_cal | test_cal | test_WSS | test_VP | test_SP | W&B |
|---|---:|---:|---:|---:|---:|---|
| **H314 Student-t ν=4 + 8-res+cal ← CURRENT SOTA** | **5.8987%** | **5.7387%** | **6.6390%** | **3.3739%** | **3.6136%** | 2scozlaf |
| H312 H296+8-res+cal | 5.8994% | 5.7388% | 6.6391% | 3.3743% | 3.6137% | enf61qrr |
| H300 H285+cal | 5.9011% | 5.7399% | 6.6404% | 3.3763% | 3.6132% | 59r4noqh |
| Transolver-3 target | — | — | **< 5.850%** | ≤ 3.421% | ≤ 3.577% | — |

**Merge gate (UPDATED)**: val_abupt_calibrated < **5.8987%** AND test_abupt_calibrated < **5.7387%**
**Paper floors**: test_VP 3.3739 < 3.421 ✓ | test_WSS 6.6390 < 6.727 ✓ | test_SP 3.6136 > 3.577 ✗ (3.6bp gap, H338 in flight)
**WSS gap**: 6.6390 → 5.85 target = **0.789pp remaining** (H339 frieren targeting this)

---

## Active Fleet (as of 22:01Z 2026-05-31 — 7 students active + 1 idle [thorfinn] post-H307 close)

| PR | Student | Hypothesis | val_raw | Status | ETA calibrated |
|---|---|---|---:|---|---|
| **#1522** | **edward** | **H338: SP-targeted loss reweighting cosine-tail (3-arm SP-weight sweep {1.05, 1.10, 1.20} to close paper SP floor gap +3.4bp)** | — | 🆕 assigned 15:45Z (post-H332 close) | ~01:00Z+1 (3 arms × 3h sequential) |
| **#1524** | **tanjiro** | **H340: σ-sweep at Student-t ν=4 — last inference-axis question. 3 arms σ ∈ {2.5e-4, 5e-4 ref, 1e-3} on H185 ep15. Arm B reproduces H314 (sanity).** | — | 🆕 assigned 18:55Z (post-H337 close, post-H314 merge) | ~14:00Z+1 (~18-21h chained eval) |
| **#1526** | **alphonse** | **H342: Multi-checkpoint output-averaging TTA (ep14+ep15+ep16 EMA from H312 SOTA cosine-tail run-enf61qrr) — 3 evals + 4 post-hoc arms (ep15 alone control, ep14+ep15, ep15+ep16, ep14+ep15+ep16). Orthogonal output-space ensemble vs H307 weight-space.** | — | 🆕 assigned 19:58Z (post-H334 close) | ~13:00Z+1 (3 evals × ~6-7h + post-hoc avg) |
| **#1520** | **nezuko** | **H336: Compose K=5 + Student-t ν=4 + 8-res + mirror + cal (K × noise-family interaction at cal-extracted regime)** | — | 🟡 in-flight 348i3z1v ETA ~23:30-00:00Z; **PR sent back 18:15Z for rebase** (H314 merge conflict on eval_tta_h252.py) — rebase after terminal | ~00:00Z+1 |
| **#1523** | **frieren** | **H339: WSS-targeted loss reweighting cosine-tail (3-arm sweep {1.5, 2.0, 3.0} on WSS loss multiplier — targets 0.789pp WSS gap)** | — | 🆕 assigned 17:45Z (post-H314 merge) | ~24:00Z+1 (training ~9h + eval ~21h, chained multi-session) |
| **#1519** | **askeladd** | **H335: Per-resolution K allocation under fixed H312-budget (3 arms: low-res-heavy / high-res-heavy / uniform-control)** | 5.9221 (Arm C) | 🟡 Arm C TTA in-flight ~19:30Z; **PR sent back 18:51Z for rebase** (H314 merge conflict on eval_tta_h252.py) — rebase before Arms A/B launch | ~22:00Z+ |
| **#1525** | **fern** | **H341: wz-only WSS loss reweight cosine-tail (3 arms wz × {2.0, 3.0, 5.0} — per-axis decomposition of H339 targeting bottleneck wz=8.62%)** | — | 🆕 assigned 19:25Z (post-H333 close, post-H314 merge) | ~25:00Z+1 (training ~9h + eval ~21h, chained multi-session) |

**Merged this loop**:
- **PR #1500 frieren H314** — MERGED (17:45Z). **NEW SOTA** val_cal 5.8987 / test_cal 5.7387. Student-t ν=4 beats H312 by 0.7bp/0.1bp. ν-sweep parabolic (ν=4 minimum), gain uniform across channels, α ν-invariant. Merge gate tightened.
- **PR #1498 edward H312** — MERGED (02:30Z). **PRIOR SOTA** val_cal 5.8994 / test_cal 5.7388. Finding 'better-raw-better-cal' confirmed. Calibration near saturation for diagonal affine on single model.

**Closed this loop (22:01Z)**:
- **PR #1507 thorfinn H307** — CLOSED (22:01Z). **Weight-space model soup α-sweep NULL on H314 gate** (3 findings banked): 3-arm α∈{0.5, 0.75, 0.85} sweep of seed-1+seed-2 EP15 EMA mix with full H312 cal. Terminal Arm D α=0.85 `v4ye4e4k`: val_cal **5.8990** (+0.27bp MISS H314 gate) / test_cal **5.7380** (−0.7bp BEAT). AND-fails by 3 milli-bp on val. **3 findings**: (a) **`soup-alpha-val-monotone-no-cross`** — val_cal monotone 5.9017→5.8999→5.8990 across α∈{0.5, 0.75, 0.85} approaching seed-1 from below; diminishing returns geometry → linear extrapolation from D→1.0 closes only ~0.13bp, seed-1-alone val_cal estimated ~5.8977; pure-seed-1 IS the val_raw optimum at val_N=34; soup cannot improve val. test_cal α-invariant at ~5.7380 across all 3 arms (within 0.2bp noise floor). (b) **`soup-cal-extracts-more`** — α_cp shift across arms Δ=3.75e-4 vs H332 cross-α invariance threshold ±2.6e-4 — soup flattens residual structure, letting diagonal-affine cal catch slightly more error. The 2.7bp val_cal closure across arms is entirely cal-extraction, not raw improvement. (c) **`soup-wz-improves-mildly`** — test_WSS_z 8.617 vs H314 8.6195 (−0.22bp) only channel-level improvement; insufficient alone but third corroborating signal that wz is the bottleneck (motivates H341/fern). **Structural verdict**: α-soup axis exhausted at this datapool. Response curve is smooth, monotone, lands ~0.3bp below H314 gate at most favorable α (0.85). No room without (a) larger seed-pool ensembling [Morgan blocked 2026-05-28] or (b) different perturbation geometry (H342 alphonse output-space). → **thorfinn now idle — next hypothesis assignment pending**.
- **PR #1518 alphonse H334** — CLOSED (19:58Z). **Brute-force per-channel α grid search NULL on gate** (3 findings banked): 41-point α-grid sweep × 5 channels + 5-iteration joint greedy descent → optimal α floor at val_cal 5.9015 (regresses H314 gate by +2.8bp), test_cal 5.7368 (beats H314 test gate by 1.9bp) but AND-fails. The 3 findings: (a) **val-mle-is-h312-ols** — joint-greedy converges to H312 OLS α within 1e-4 across all 5 channels in 2 iterations (zero curvature on val loss in 0.001 neighborhood of OLS — confirms H329 WLS=OLS at val_N=34 directly via grid); (b) **cal-val-test-mismatch-15milli** — test set prefers uniformly Δα=−0.0015 across 4/5 channels giving −0.20bp test improvement vs +0.21bp val regression (below the 0.5bp noise floor; same magnitude as H333 Bootstrap test gain — both methods extracting the same val-test asymmetry from different angles); (c) **tau-z-cal-saturated** — wz channel α uniquely flat over ±0.01 grid range (curvature 0 to 4 dp), confirms wz error is structural not calibratable — must be attacked at training time (motivates H341 fern). Structural verdict: **diagonal-affine cal axis is closed for H185 EP15 at the OLS-MLE; remaining test-side asymmetry is val_N=34 measurement noise floor, not exploitable via val-based fitting**. → H342 (PR #1526) multi-checkpoint output-averaging assigned to alphonse — orthogonal output-space ensemble: same recipe, average across consecutive EP14+EP15+EP16 EMA snapshots. Complementary to H307 (weight-space averaging across seeds).
- **PR #1517 fern H333** — CLOSED (19:25Z). **3-fold cal-stability verification, NULL on gate**: All 5 cal arms (OLS/LOO/Bootstrap/L1/Huber) tie at val_cal=5.8994 (fails H314 val gate by 0.7bp). Bootstrap test_cal 5.7386 marginally beats H314 test gate by 0.1bp but cannot rescue the val miss. Findings banked: `cal-coefficients-loo-stable` (max LOO range/α=0.0944%), `cal-robust-equivalent-to-OLS` (L1/Huber β_VP shifts 6.4% but no test gain — residual is homoscedastic, confirms H329), `cal-bootstrap-marginal-test-gain` (Bootstrap −0.2bp test vs OLS, below noise floor). Structural verdict: **post-hoc calibration axis is closed for diagonal-affine on H185 EP15**. → H341 (PR #1525) wz-only WSS reweight assigned to fern.
- **PR #1521 tanjiro H337** — CLOSED (18:55Z). **Commissioning success**: seed-3 EP15 EMA artifact uploaded (W&B `yteak7pd`, mirror val=6.0097% well within ≤6.10% gate, no divergence; EP16 ema also saved). 3-way (s1, s2, s3) model-soup α-search now unblocked. Test_abupt at EP16 train-loop=5.8549% (raw single-seed without TTA, as expected). **CLOSED not MERGED** — H337 was structured as artifact-production, not SOTA candidate. → H340 (PR #1524) σ-sweep at ν=4 assigned to tanjiro.

**Closed this loop (03:00Z)**:
- **PR #1497 fern H306** — CLOSED. Finding 'conf-weighted-T1-equals-uniform-avg'. T=1.0 softmax over 64 TTA passes with similar per-pass loss collapses to uniform → null. Per-point inverse-variance TTA aggregation closed at T=1. → H319 per-resolution calibration assigned to fern.
- **PR #1499 askeladd H313** — CLOSED (04:15Z). Finding 'regional-cal-overfits-val'. 4z/6z regional affine cal passes val gate (5.8947/5.8938) but fails test by +2.0/+2.2bp. Per-zone α range 0.985–1.005 — real signal (rear zone α>1 vs everywhere else α<1) but too small to survive 34-car val→test gap. → H328 ridge-regularized λ-sweep assigned to askeladd (PR #1511) — rescues spatial signal with shrinkage to H300 global.
- **PR #1502 alphonse H316** — CLOSED (04:35Z). **Finding 'cal-scale-dominates-bias-null-on-test'**: ablation decomposed H300/H312's gain — α (scale) does 81%, β (bias) is 0% (+0.1bp). The huge β_VP ≈ -0.89 is the OLS intercept absorbing α≠1, not an independent bias correction. Model has WSS scale deficit (predictions over-large by 0.5-0.8% on τ channels). → H329 abupt-weighted OLS objective assigned to alphonse (PR #1512) — better α estimation via aligning fit objective with eval metric.
- **PR #1501 nezuko H315** — CLOSED (05:00Z). **Finding 'agg-operator-mean-is-blue'**: 3-arm aggregator sweep (mean/median/trim10) — mean wins uniformly. Median +4.0bp val / +6.4bp test, trim10 +0.3bp val / +1.8bp test. Per-channel α/β agree to 1e-5 across arms. TTA noise distribution at σ=5e-4 K=4-anti is approximately Gaussian-symmetric → mean is BLUE. Mean arm reproduces H312 (val 5.8994 / test 5.7388) exactly — confirms H312 SOTA is point-stable, not lucky. → H330 K=5+8-res+H312-cal compositional test assigned to nezuko (PR #1514).
- **PR #1504 tanjiro H317** — CLOSED (05:35Z). **Finding 'quadratic-cal-memorizes-val'**: 15-param quadratic per-channel cal fails both gates (val_cal 5.9069 vs 5.8994, test_cal 5.7420 vs 5.7388). Fitted a₂ coefficients are 2-7 orders of magnitude smaller than a₁; VP — dominant cal channel — has a₂=+3e-7 (zero curvature). Confirms H316 prediction (val improves, test regresses). Per-channel DOF axis saturated at affine 10-param under 34-car val. → H331 (PR #1515) 10-res gap-fill assigned to tanjiro — pivots from DOF allocation to TTA budget allocation (R-axis vs H330's K-axis at equal 80-forwards/case compute).
- **PR #1509 fern H319** — CLOSED (10:45Z). **Finding 'per-resolution-cal-invariant'**: 8-resolution × 5-channel cal (80 params) passes H312 gates by 0.0001bp (val 5.899394 / test 5.738691) — improvement is below noise floor. max(α_range) = 0.002023 (τ_z) < 0.005 across all 5 channels at val_N=34 → α is resolution-invariant. Only signal: β_VP monotone-in-res (range 0.271, 36% swing) but doesn't propagate to abupt. 8× parameter count for ~0bp gain triggers the disproportionate-complexity exception. **Cal axis structural sweep now exhausted**: per-channel (H316), cross-channel (H323), per-region (H313), per-resolution (H319) all null beyond diagonal-affine. → H333 (PR #1517) assigned to fern — cal-coefficient stability under LOO-CV/bootstrap/L1/Huber.
- **PR #1512 alphonse H329** — CLOSED (11:39Z). **Finding 'abupt-weighting-irrelevant-at-this-scale'**: WLS cal with `w_i ∝ 1/||truth_i||²` converges to unweighted OLS at val_N=34 — α agrees to 1e-4 across all 5 channels; β within 1e-3. val_cal 5.899385 / test_cal 5.738560 — gates pass by 0.015bp / 0.24bp, both below noise floor. Per-car residual histogram is approximately homoscedastic (no outliers, no fat tails) → OLS is BLUE, any weighted variant is equal-or-worse. → H334 (PR #1518) assigned to alphonse — brute-force per-channel α grid search (41 points × 5 channels + 5-iteration joint greedy descent) verifies whether H312 sits at the global per-channel optimum within ±0.001.
- **PR #1516 edward H332** — CLOSED (15:40Z). **Findings 'α-seed-invariant' + 'cal-transfer-lossless-h312-to-seed2' + 'seed-2-val-deficit-test-parity'**: val_cal misses H312 gate by +7.1bp (5.9065), test_cal beats gate by 1.8bp (5.7370) but AND-fails. The 3 findings: (a) per-channel α agrees across seed-1/seed-2 within max |Δα|=2.6e-4 (4× tighter than ±1e-3 threshold) — H312 α is model-class invariant; (b) Arm B (transfer) ≡ Arm C (refit) to 4 dp → soup cal can use hardcoded H312 α (saves ~30s/eval, no refit needed); (c) seed-2 +7bp WORSE val_raw but +0.14bp BETTER test_raw than seed-1 → two seeds occupy different basins of equal-ish test quality (precondition for H307 soup variance-reduction). **Paper-floor structural observation**: test_SP_cal 3.6116 vs floor 3.577 = +3.4bp gap is the ONLY remaining paper-floor-eligible channel — must come from training. → H338 (PR #1522) assigned to edward — SP-targeted loss reweighting cosine-tail extension, 3-arm sweep {1.05, 1.10, 1.20} on SP loss multiplier (~9h total).
- **PR #1515 tanjiro H331** — CLOSED (14:55Z). **Finding 'res-density-saturated-8res'** (+ joint cross-fleet finding **'K-R-tta-axes-saturated-jointly'** with H330): val_cal 5.8992 passes (-0.02bp), test_cal 5.7392 FAILS strict gate (+0.04bp). Null within ±1bp band. **The H330 ↔ H331 val_raw bit-identity at 5.9217 is the structural headline**: at +25% compute beyond H312, neither K-axis nor R-axis extracts new raw signal — the TTA inference-set axes are saturated in both sub-axes simultaneously. Resolution axis now fully closed across all 3 directions (below-32K H291, above-131K H267/H292, within-density H331). Per-channel cal coefficients identical to H312 (α shifts ≤1.1e-4). → H337 (PR #1521) assigned to tanjiro — 3rd EP15 seed commissioning via H310-style cosine extension at `--seed=2027` (~3h artifact production, unblocks 3-way model soup α-search since H307 2-way soup α=0.5 already beat test gate by 0.8bp at +2.3bp val miss).
- **PR #1514 nezuko H330** — CLOSED (14:12Z). **Finding 'K5-cal-redundant-at-h312-budget'** (+ sub-findings **'β_VP-K-sensitive'**, **'K×res-cal-additive-at-K=4-5'**): Both gates pass strict by 0.03-0.04bp (val 5.8990 / test 5.7385) — technically a 2-gate SOTA. **CLOSED not MERGED** invoking CLAUDE.md disproportionate-complexity exception: 0.03bp test_cal at +25% PERMANENT compute on every future eval = ~80× worse cost/benefit than H312 cal win. Pending H314 Arm A (frieren PR #1500) is 3× bigger test effect at SAME compute (Student-t ν=4 replaces Gaussian) and is the better candidate — merging H330 would invalidate H314 Arm A's test gate. β_VP shifts −1.17e-2 (1.4% rel) between K=4/K=5 — VP is uniquely K-sensitive (largest dynamic range), but rel_l2 absorbs the constant offset. → H336 (PR #1520) assigned to nezuko — compose K=5 + Student-t ν=4 + 8-res + mirror + cal. Tests whether K-axis (this PR) and noise-family-axis (H314 Arm A) cal-orthogonal improvements are ADDITIVE (predicted ~0.13bp test_cal SOTA) or REDUNDANT (K-axis absorbed by Student-t). Either outcome closes the K × noise-family interaction.
- **PR #1511 askeladd H328** — CLOSED (12:00Z). **Finding 'regional-cal-irrecoverable'** (+ sub-finding **'cal-val-test-disagreement-on-shrinkage'**): 12-arm λ sweep over ridge-regularized 6-zone regional cal. Test_abupt is **monotonically decreasing in λ** with min at λ=∞ (= H300 global). No interior λ where test improves over H300. Best λ by val gives test=5.7410 (+2.1bp regression vs H300) — val and test DISAGREE on optimal shrinkage at val_N=34 (val wants λ→0, test wants λ→∞). The rear-zone α>1 vs everywhere-α<1 signal H313 surfaced is a coincidence of the 34-car val sample, NOT a structural model property. → H335 (PR #1519) assigned to askeladd — per-resolution K allocation under fixed H312-budget (3 arms; tests whether K-uniform-across-resolutions is Pareto-optimal at H312's compute). **Cal axis now has 5 independent structural nulls confirming H312 α is at the diagonal-affine MLE**: H316 (β-null) + H319 (per-res-invariant) + H323 (cross-channel-diagonal) + H328 (regional-cal-irrecoverable) + H329 (abupt-weighting-irrelevant). H334 verifies whether OLS-MLE is also the global test-loss minimum.
- **PR #1508 edward H323** — CLOSED (07:35Z) on smoke evidence. **Finding 'cross-channel-wss-diagonal'**: 3×3 cross-channel WSS A matrix is effectively diagonal at val-set scale — off-diag/|diag| ≤ 3.04e-4 (16× below 5e-3 close threshold, 33× below original 1e-2 "meaningful" threshold). Cross == diag to <0.01bp on every WSS channel and abupt. By construction of per-point pooled OLS, off-diagonals can only deviate by fitting per-point val noise. Excellent student diagnosis of smoke v1's per-car-mean vs per-point pooled OLS objective mismatch. 7h GPU slot reclaimed. → H332 (PR #1516) seed-2 α robustness diagnostic assigned to edward — tests whether H312 α transfers cleanly to seed-2 (informs H307 soup cal) and whether seed-2 is competitive standalone.

**Hot watch (03:00Z)**:
- **Frieren H314 2scozlaf** (~03:00Z) → val_raw 5.9213% leader. **If calibrated val < 5.8994, becomes SOTA candidate #1** against new tighter gate. This is the primary watch.
- **Tanjiro H317 nld6viep** (~03:30Z) → quadratic cal; must beat val_cal 5.8994/test_cal 5.7388 (harder now)
- **Alphonse H316 yg4sbrtw** (~03:00Z) → bias vs scale ablation; informs whether to pursue H312→H324 higher-K cal variants
- **Nezuko H315 2wq13m13** → **INTERIM: median aggregation +4.3bp worse, trim10 +0.5bp worse** — aggregation axis looks null
- **Askeladd H313 iya68eq8** (~02:30Z) → regional calibration; test pass nearly done
- **Fern H319 #1509** (~10:00Z) → per-resolution cal; tests resolution-dependent bias structure
- **Thorfinn H307** (~08:00Z) → model soup; uses new H312 SOTA as the seed-1 calibration baseline
- **Edward H323** (~09:30Z) → cross-channel WSS affine; tests whether τ_x/τ_y/τ_z have rotational coupling
- **Frieren H314 Arm B (ν=3) `fk8kd4nl`** (~10:44Z) → val_raw 5.9222 tied H296 baseline, slightly worse than Arm A (ν=4) 5.9213 — ν=4 sweet-spot signal strengthening. Arm C (ν=8) auto-chained, ETA ~17:25Z
- **Thorfinn H307 Arm C α=0.75 `<pending>`** (~14:59Z) → primary watch — if val_cal < 5.8994 AND test_cal < 5.7388, **MERGE as new SOTA via soup**; α=0.5 already beat test gate, α=0.75 (seed-1-favoring) is the likely best-by-val
- ~~Edward Arm B a8uyi3ev~~ — killed at 08:55Z by edward (orphan from prior cycle, hogged GPUs alongside H332)

**Queue for next idle students** (the calibration axis dominates):
- **H318**: Calibration on H299-style noise + H312 cal — does noise-family change the residual structure?
- **H320**: 3rd seed commission (if H307 model-soup wins) OR cubic calibration (if H317 quadratic gains ≥3bp)
- **H321**: Jackknife/LOO calibration variance on 34-car val (robustness check of 10-param fit on 34 cars)
- **H322**: Best-of-K aggregation under H312 calibration (per-case selection vs averaging)
- **H324**: K=5+6-res+cal confirmation (Arm B result will tell us if needed separately)
- **H325**: WLS calibration (minimize abupt-aligned objective per-car, vs OLS absolute-squared error)
- **H326**: Compose H323 (cross-channel WSS) + H319 (per-resolution) — both win → 8×{5×5+3} mixing matrix per resolution

---

## Recently Merged / Closed

| PR | Student | Finding | val | test |
|---|---|---|---|---|
| **#1484 fern H296** | EP15+anti-K4+8-res+mirror | **MERGED (17:05Z) — NEW SOTA** — Finding TTT: K=4+8-res Pareto-dominates K=5+6-res (res-axis compound > K-axis escalation at this operating point). 4/5 channels improve (VP dominant). New gate: val<5.9221, test<5.7678. | 5.9221 | 5.7678 |
| #1485 thorfinn H297 | EP15+anti-K4+6-res, σ_attn=0 | **CLOSED (17:05Z)** — Finding "per-layer-noise-attn0-mlp-null": K=4 TTA at EP15 is noise-routing-insensitive; attn vs mlp layer split has no effect on aggregate performance. Channels within 0.001 of H295 baseline. | 5.9237 | 5.7684 |
| **#1483 tanjiro H295** | EP15+anti-K5+6-res+mirror | **MERGED (15:45Z)** — Finding SSS: K-axis saturation curve at EP15 (K=4→5 slope 1/17 of K=3→4, ~0.04bp gain, 5/5 sign-consistent). | 5.9231 | 5.7679 |
| #1477 frieren H291 | EP15+anti-K3+8-res-LOWER {16K,24K,32K-131K}+mirror | **CLOSED (13:45Z)** — **Finding RRR**: 8-res lower fails both gates by ~1bp (val 5.9252 +1.7bp, test 5.7693 +1.0bp). Lower-res samples add Taylor variance without info gain. Combined with PPP: resolution axis fully closed asymmetrically below 32K and above 131K; 6-res {32K-131K} is global Pareto optimum. → H302 (channel-asymmetric resolution) assigned to frieren. | 5.9252 | 5.7693 |
| #1476 thorfinn H290 | EP15+anti-K3+multi-σ{3e-4,5e-4,7e-4}+6-res | **CLOSED (11:03Z)** — Finding LLL-multi-σ-diversity-null: val 5.9241/test 5.7689 fails H285 gate (+0.6bp both axes). σ-diversity subsumed by K-axis: K=3→K=4 at fixed σ gives 4–7× the gain vs σ-mixing at K=3. → H297 (per-layer noise stratification) assigned to thorfinn. | 5.9241 | 5.7689 |
| #1473 fern H288 | EP15+anti-K3+8-res densified | **CLOSED (09:56Z)** — Finding KKK-8res-K3-val-passes-test-marginal-fail: val 5.9229 (passes gate), test 5.7685 (fails +0.02bp). VP improves −2.1bp vs H275; SP degrades +1.1bp. K-axis and res-axis are orthogonal. → H296 (K=4+8-res compound) assigned to fern. | 5.9229 | 5.7685 |
| #1470 tanjiro H285 | EP15+anti-K4+6-res+mirror | **MERGED as NEW SOTA** — Finding JJJ-K4-K-axis-alive-at-EP15 (K=3→K=4 = −0.8bp val / −0.7bp test, sign-consistent 5/5 channels, slope diminishing) | 5.9235 | 5.7683 |
| #1472 askeladd H287 | EP14 checkpoint anti-K3 | **Finding III-EP-axis-fully-exhausted** (EP14 val 5.9326/test 5.7764 fails gate; EP13≈EP14<EP15(peak)>EP16; all accessible checkpoints below EP15 confirmed worse across all 3 channels) | 5.9326 | 5.7764 |
| #1460 nezuko H280 | EP13+Sobol-anti-K5+6-res | **Finding HHH-EP13-no-rescue-with-K5-Sobol** (EP13+K5+Sobol val 5.9313/test 5.7753 fails gate; all channels match-or-degrade vs SOTA; doubled-K Sobol diversity cannot compensate for EP13 vs EP15; EP-axis closed above AND below) | 5.9313 | 5.7753 |
| #1475 frieren H289 | EP16 checkpoint axis (closed w/o running) | **Finding FFF-EP-axis-closed-at-H275-recipe** (checkpoint EP12/EP16 unavailable; H244 3-res evals confirm EP15 peak, EP16 slight regression; EP-axis fully exhausted with available checkpoints) | n/a | n/a |
| #1466 thorfinn H283 | EP15+σ=3e-4+Sobol-K5+6-res+mirror | **Finding XX-Sobol-confirmed** (close, val 5.9274 +3.1bp / test 5.7725 +3.5bp fails H275 gate; σ-axis closed across all noise families at EP15) | 5.9274 | 5.7725 |
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

## Findings Bank (48 banked — UUU and calibration-dominates-TTA added 18:55Z)

| ID | Source | Summary |
|---|---|---|
| **calibration-dominates-TTA-refinement** | H300 edward (merged 18:48Z) | Per-channel affine calibration (10 OLS params fit on 34-car val) gains 28bp test — more than H285→H296 TTA refinements combined. VP bias β≈−0.86 is model-intrinsic, not TTA-dependent. All 5 channels Pareto-improve. Calibration axis is orthogonal to TTA axis. |
| **UUU-embed-noise-null** | H299 askeladd (closed 18:48Z) | Restricting weight noise to embedding/pos-encoding layers only (inverse of H297 layer stratification): val=5.9244/test=5.7689, fails H296 gate by 2.3/1.1bp. Layer-stratification axis fully closed (H297 full-model = H299 embed-only = same aggregation). |
| **coord-noise-harmful** | H298B alphonse (znmtxzdk, val only) | Input-coord perturbation (σ=5e-4, K=4 anti + 6-res + mirror) → val=7.82% (+190bp). Geometry encoder is spatially sensitive — any coordinate noise destroys surface predictions. TTA noise budget must stay in weight-space. H298C σ=1e-3 expected to fail harder (monotone slope). |
| **TTT-K4+8res-Pareto-dominates** | H296 fern (merged 17:05Z) | K=4+8-res (test 5.7678) beats K=5+6-res (5.7679) on both gates. Res-axis compound > K-axis escalation at current operating point. Pareto frontier: (K=4, 8-res). |
| **per-layer-noise-attn0-mlp-null** | H297 thorfinn (closed 17:05Z) | Noise routed only to MLP layers (attn σ=0) at EP15+K=4 anti-thetic: channels within 0.001 of H295 baseline. Noise-routing-insensitive at this operating point. Layer stratification axis closed. |

## Prior Findings Bank (43 banked prior to today)

| ID | Source | Summary |
|---|---|---|
| **SSS-K-axis-saturation-at-K5** | H295 tanjiro (merged 15:45Z as NEW SOTA) | K=3→4 anti-thetic at EP15+6-res was −0.8bp val / −0.7bp test (Finding JJJ). K=4→5 is −0.04bp val / −0.04bp test (1/17th the slope). K_eff=10 at K=5 makes the slope diminish to near-noise-floor. All 5 channels still move in the right direction (signal is real). K=5 is the current optimum; K=6 would cost +20% compute for an estimated ≤−0.01bp gain. New gate: val<5.9231, test<5.7679. Next: H303 probes σ=3e-4 at K=5 (tests if tighter σ reduces 2nd-order residual better at higher K). |
| **RRR-8res-lower-degrades** | H291 frieren (closed 13:45Z) | 8-res lower {16K, 24K, 32K-131K} fails both gates by ~1bp (val +1.7bp, test +1.0bp). Lower-res samples add Taylor variance without info gain. Combined with PPP (8-res upper +11.8bp val): resolution axis fully closed asymmetrically — current 6-res {32K-131K} is global Pareto optimum in {n_res, resolution_range} space. Remaining unexplored direction: channel-asymmetric resolution selection. |
| **QQQ-student-t-null** | H294 askeladd (closed 12:26Z) | Student-t df=3 ≈ Gaussian at matched RMS σ=5e-4 (val 5.9248 +1.3bp). Combined with MMM: noise distribution family axis definitively closed. Leading-order anti-thetic cancellation is distribution-free. |
| **PPP-8res-upper-degrades** | H292 edward (closed 12:26Z) | 8-res upper {32K-192K} degrades val by 11.8bp vs H285 SOTA. Resolution ceiling confirmed at ≤131K. More high-res samples add variance without Taylor-cancellation benefit. |
| **OOO-aggregation-null** | H286 alphonse (closed 12:24Z) | All 5 homogeneous operators (mean, median, trimmed_mean_10/20, Huber) fail H285 gate. Best alternative (trimmed_mean_10) is 0.7bp worse than mean. Linear Taylor-cancellation already provides optimal pooling. → H301 tests heterogeneous per-channel selection. |
| **MMM-laplace-null** | H293 nezuko (closed 12:13Z) | Laplace noise ≈ Gaussian at matched RMS σ=5e-4 (val 5.9246/test 5.7694, both +1.1bp). Distribution shape doesn't matter — Gaussian, Laplace, Student-t all equivalent at this scale. Noise-family axis closed. |
| **LLL-multi-σ-diversity-null-at-EP15-anti-K3** | H290 thorfinn (closed 11:03Z) | At EP15+anti-K3+6-res, mixing σ∈{3e-4,5e-4,7e-4} gives val ~H275 (−0.02bp), all channels FAIL H285 gate (+0.6bp both axes). σ-diversity subsumed by K-axis: K=3→K=4 at fixed σ=5e-4 gives 4–7× more gain than σ-mixing at K=3. SP channel unimproved (3.6428 vs target 3.577). → H297 per-layer noise stratification assigned to thorfinn. |
| **KKK-8res-K3-val-passes-test-marginal-fail** | H288 fern (closed 09:56Z) | At EP15+anti-K3, 6→8 res (mid densification: +40960,+57344) gives val −1.4bp vs H275 but test only −0.05bp. Channel asymmetric: VP −2.1bp (large), SP +1.1bp (regression). K-axis and res-axis are **orthogonal** — distinct channel signatures → K=4+8-res compound (H296) should stack both gains additively. K=4+6-res (H285) beats K=3+8-res on test_abupt by 0.02bp within noise floor. |
| **JJJ-K4-K-axis-alive-at-EP15** | H285 tanjiro (merged 08:37Z as NEW SOTA) | K=3→K=4 anti-thetic at EP15+6-res gives val −0.8bp / test −0.7bp. Sign-consistent across all 5 paper-facing channels (5/5). K-axis still alive but slope diminishing: K=2→3 compound was super-additive (Finding ZZ); K=3→4 is weakly additive. K=5 next (H295 tanjiro) to close axis. Merge gate updated to val<5.9235, test<5.7683. |
| **III-EP-axis-fully-exhausted** | H287 askeladd (closed 08:10Z) | EP14+anti-K3+6-res val 5.9326/test 5.7764 fails gate. Combined with FFF (EP16), HHH (EP13), EP12 unavailable: EP-curve at H275 recipe fully closed as EP13≈EP14<EP15(peak)>EP16. No checkpoint variation can improve SOTA without new training. |
| **HHH-EP13-no-rescue-with-K5-Sobol** | H280 nezuko (closed 08:10Z) | EP13+Sobol-anti-K5+6-res val 5.9313/test 5.7753 fails gate. Doubled-K diversity + Sobol cannot rescue EP13 checkpoint. Combined with Finding FFF (EP16) and III (EP14): EP-axis fully exhausted on both sides of EP15 peak. |
| **GGG-Sobol-anti-non-stacking-at-K3-EP15** | H284 edward (closed 07:20Z) | Sobol×anti compound at K=3 EP15 is non-additive: val −0.012bp / test +0.065bp ≈ zero net movement vs H275. Anti-K=3 already kills the dominant linear Taylor variance; Sobol QMC targets the SAME variance budget. Both mechanisms not orthogonal at K=3. |
| **FFF-EP-axis-closed-at-H275-recipe** | H289 frieren (closed 06:05Z, no runs) | EP-axis exhausted at H275 recipe family. Only EP13/14/15 checkpoints available. H244 3-res evals show EP14 5.9613→EP15 5.9516 (peak)→EP16 5.9548 (regression). EP15 is unimodal peak. `best_epoch=15` in H244 config confirms. Do not test further EP variations at this recipe without new training. |
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

AXES FULLY CLOSED (do not revisit without new training):
- **EP axis**: EP12 unavailable, EP13/EP14 fail gate, EP15=SOTA, EP16 hurts. Findings FFF+HHH+III close this completely.
- **σ axis (single-σ)**: σ=5e-4 optimal. Findings AAA+XX-Sobol close this.
- **σ-diversity axis (multi-σ at K=3)**: σ-mixing subsumed by K-axis. Finding LLL closes this.
- **Sobol×EP15 axis**: Anti beats Sobol at both EPs. Findings DDD+EEE close this.
- **Sobol×anti K=3 compound**: Non-additive. Finding GGG closes this.
- **Aggregation operator (homogeneous)**: mean/median/trimmed/Huber all null. Finding OOO closes this.
- **Noise distribution family**: Gaussian ≈ Laplace ≈ Student-t df=3 at matched RMS σ=5e-4. Findings MMM+QQQ close this.
- **Resolution axis (extent)**: 8-res upper (+160K,+192K) +11.8bp val; 8-res lower (+16K,+24K) +1.7bp val / +1.0bp test. Current 6-res {32K-131K} is Pareto optimum. Findings PPP+RRR close this asymmetrically.

IN-FLIGHT (do not duplicate):
- **K=4+8-res compound** (fern H296 PR #1484) — highest-EV K×res cell; Finding KKK orthogonality proven; val arm ~13:30Z
- **K=5 at EP15** (tanjiro H295) — closes K-axis; val arm ~12:00Z
- **Per-layer noise stratification σ_attn=0** (thorfinn H297 PR #1485) — next SP-targeted intervention; ~15:00Z
- **Resolution lower ladder** (frieren H291 PR #1477) — test relaunch in progress; val 5.9252 fails gate but want full result
- **Resolution upper ladder** (edward H292) — 8-res {32K-192K}; no val yet
- **Aggregation operators** (alphonse H286) — multi-mode sweep; val 5.9242 (fails gate), test arm pending
- **Laplace noise** (nezuko H293) — val 5.9246 (fails gate by +0.11bp), test pending
- **Student-t noise df=3** (askeladd H294) — val 5.9248 (fails gate by +0.13bp), test pending

PRIORITY NEXT ROUND (when students become idle):
1. Student-t df=1.5 (Cauchy-like, infinite variance but check stability) — if H294 wins
2. Per-layer noise stratification — target conv vs attention separately
3. Y-mirror augmentation (double-mirror x+y) — lateral symmetry exploitation
4. Best-of-K (min-loss per channel) selection — orthogonal to mean aggregation
5. Noise anisotropy — per-tensor σ proportional to gradient magnitude
6. SP floor (3.6427→3.577, 6.5bp gap) is the binding Morgan target; need wins on SP specifically

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
| EP15 + σ=3e-4 + Sobol K=5 + 6-res + mirror | ✗ H283 5.9274/5.7725 — Finding XX-Sobol-confirmed (fails H275 gate +3.1/+3.5bp; σ-axis closed across all noise families at EP15) |
| EP15 + σ=3e-4 + anti-thetic K=3 + 6-res + mirror | ✗ H277 5.9255/5.7705 — Finding AAA (σ=3e-4 flat/unfavorable under anti-thetic) |
| EP15 + anti-thetic K=4 pairs (8p) + 6-res + mirror | 🟡 H285 tanjiro (running) |
| EP15 + Sobol QMC K=5 + 6-res + mirror | ✗ H279 5.9291/5.7734 — Findings DDD-Sobol-EP15-super-additive + EEE-anti-dominates-Sobol-at-EP15 (fails H275 gate +4.8/+4.4bp; anti beats Sobol at EP15 too) |
| EP13 + Sobol × anti-thetic K=5 (10p) + 6-res + mirror | ✗ H280 5.9313/5.7753 — Finding HHH (EP13+K5+Sobol fails gate; EP-axis closed below too) |
| EP15 + Sobol × anti-thetic K=3 + 6-res + mirror | ✗ H284 5.9242/5.7697 — Finding GGG (Sobol×anti non-additive at K=3; anti exhausts linear variance) |
| EP13 + anti-thetic K=3 + σ=3e-4 + 6-res + mirror | ✗ H281 5.9342/5.7785 — Finding AAA confirmed at EP13 (+9.9bp val vs SOTA) |
| EP15 + aggregation operator sweep (mean/median/trimmed/Huber) | 🆕 H286 alphonse (assigned) |
| EP14 + anti-thetic K=3 + 6-res + mirror | ✗ H287 5.9326/5.7764 — Finding III (EP14 fails gate; EP-axis fully exhausted) |
| EP15 + Laplace noise + anti-K=3 + 6-res + mirror | 🆕 H293 nezuko (just assigned, PR #1481) |
| EP15 + Student-t df=3 noise + anti-K=3 + 6-res + mirror | 🆕 H294 askeladd (just assigned, PR #1482) |
| EP15 + anti-K=3 + 8-res densified (+40960+57344) + mirror | ✗ H288 val 5.9229/test 5.7685 — Finding KKK (val passes gate, test +0.02bp miss; VP−2.1bp gain, SP+1.1bp regression; K-axis⊥res-axis) |
| EP15 + anti-K=4 + 8-res densified (+40960+57344) + mirror | 🆕 H296 fern (PR #1484, just assigned) |
| EP15 + anti-K=3 + 8-res LOWER (+16384+24576) + mirror | ✗ H291 val 5.9252 / test 5.7693 — Finding RRR (fails both gates by ~1bp; res-axis closed below 32K) |
| EP15 + anti-K=3 + 8-res UPPER (+163840+196608) + mirror | 🆕 H292 edward (just assigned) |
| EP15 + Sobol × anti-thetic K=3 + 6-res + mirror | ✗ H284 5.9242/5.7697 — Finding GGG (Sobol×anti non-additive at K=3; anti exhausts linear variance) |
| EP15 + multi-σ{3e-4,5e-4,7e-4} + anti-K=3 + 6-res + mirror | ✗ H290 val 5.9241/test 5.7689 — Finding LLL (multi-σ null vs H285; σ-diversity subsumed by K-axis) |
| EP15 + anti-K=4 + 6-res + mirror (σ_attn=0, σ_mlp=5e-4) | 🆕 H297 thorfinn (PR #1485, just assigned) |
| EP16 checkpoint axis | ✗ H289 (closed, checkpoint deleted; FFF banked) |
| EP13 + Hutchinson curvature-σ | ✗ H272 Finding TT |
| EP13 + Taylor 2nd-order correction | ✗ H273 Finding SS |
| EP13 + σ=3e-4 random K=5 | H270 5.9394/5.7827 (0.2bp from old gate) |
| SWA adjacent-EP | ✗ Finding LL-SWA-null |
| EP16 + any TTA | ✗ Finding LL-EPchain |
| H183 + 6-res stacked | ✗ Finding HH-H183 |
