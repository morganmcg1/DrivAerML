# SENPAI Research State

_Last updated: 2026-05-29 10:30Z._

**CRITICAL — BASELINE.md correction (commit ea99dda was wrong):** nezuko discovered in PR #1360 that the 2026-05-28 H147 SOTA section copy-pasted H39's `yym5oa8x` test_VP/test_SP into the H147 row. Verified by W&B query of `k6q4c3on`. Correct H147 metrics: **test_VP=3.4014%** (clears 3.643% floor by 0.242pp) and **test_SP=3.5634%** (clears 3.577% floor by 0.014pp). **H147 actually CLEARS all 4 floors**, not 3-of-4. Patched in BASELINE.md this session.

**Consequence — β-decoupling finding I banked from H150 is RETRACTED.** Under the corrected H147 baseline, H150 (β1=0.97/β2=0.985) regressed on ALL 4 metrics (WSS +0.024pp, VP +0.10pp, SP +0.045pp and crosses the 3.577% cap by +0.032pp, ABUPT +0.054pp). The "pressure heads prefer β2↑" mechanism is FALSE. Simpler truth: **β1=0.95, β2=0.98 is the joint optimum on every test metric.** Wave 41 β-grid closure remains the wave's finding but is now stronger (single optimum, not a head-axis tradeoff). Corrigendum posted on closed PR #1359.

**Session events 09:50Z → 10:30Z:**
- **H151 TERMINAL CLOSED PR #1360 (no merge):** test_WSS=6.5439% (+0.003pp — noise tie); regressed VP/SP/ABUPT; crossed SP cap by +0.035pp. Headline finding from nezuko: **best-val=EP25 in BOTH H151 (T_max=45) and H147 (T_max=30)** — H147 recipe at convergence ceiling under same recipe; "more epochs" axis is exhausted.
- **H150 TERMINAL CLOSED PR #1359 (no merge):** verdict unchanged but reasoning corrected (see retraction above).
- **H158 ABORTED EP1 PR #1420 CLOSED:** advisor error stripped 4 H147 SOTA defaults; corrected stack used for H159+H160.
- **H159 (vol_p_charbonnier=0.3 on CORRECTED H147 stack)** assigned to frieren PR #1423 — tests heavier VP regularization to free GradNorm head-weight for WSS. 8-EP T_max=8.
- **H160 (β1=0.95, β2=0.985 — last untested β-grid cell)** assigned to tanjiro PR #1424 — isolates β2-only mover. Under corrected baseline, H160 closes the β-grid cleanly in all 3 outcomes.
- **H161 (cosine restart / 2-cycle on H147 stack)** to be assigned to nezuko next — natural follow-up to her own convergence-ceiling finding.
- **H157b EP9.28=6.5085% val_WSS** running on fern PR #1385 — descent rate -0.103pp/EP; below H147 final val by 0.08pp; projected test_WSS 6.41-6.46% — **leading current merge candidate**. EP15 realistic terminal ~14:46Z. Watch test_SP (H150-β stack may push SP over 3.577% cap, like H150 did).
- **Bug-fix commit e31fd60** (precompute_curvature_proxy.py + curvature_proxy_stats_k16_v1.json) — cherry-pick pending; needed for any future advisor pulling the H147 stack from cold env.

**HUMAN DEADLINE 15:45Z (5h15m remaining)** — H157b EP15 terminal ~14:46Z; H159 8-EP terminal ~15:25Z; H160 8-EP terminal ~15:30Z; H161 8-EP terminal ~15:30Z.

---

## Current research focus: Wave 41 — closing β-grid + tier-shift to tau_z and lr-axis

**Primary target:** Beat test_WSS < 5.85% (Transolver-3 SOTA) without degrading test_VP ≤ 3.643% or test_SP ≤ 3.577%.  
**Current single-model SOTA:** PR #1344 (H147) — test_WSS = **6.5409%** — gap to target = −0.69pp.  
**Hard constraint (Issue #1056):** WSS is the main focus. NO ENSEMBLES. ~1 day until shutdown (2026-05-29).

---

## Most recent human research directive (Issue #1056)

- "test_wss is the main focus now!!" (comment #92)
- "NO MORE ENSEMBLES! Its the lazy route to better results" (comment #17)
- "there are only 2 days left before we shut down this experiment" (~2026-05-29)
- Test VP and SP floors from PR #972 must NOT be regressed: test_VP ≤ 3.643%, test_SP ≤ 3.577%

---

## Wave 41: β-grid disentanglement — COMPLETE (2026-05-28 18:30Z)

### Final β-grid verdict

```
        β1=0.93    β1=0.95    β1=0.97
β2=0.97  H149⛔    H152⛔     —
β2=0.98            H147⭐     H153⛔
β2=0.985           —          H150⏳
```

| Direction | Outcome | Evidence |
|-----------|---------|----------|
| β1↓ + β2↓ | ⛔ falsifying | H149 aborted EP3 val_WSS=7.083% |
| β2↓ pure | ⛔ falsifying | H152 aborted EP5.4 val_WSS=7.113% (β1=0.95, β2=0.97) |
| β1↑ pure | ⛔ destabilizes EP1 | H153 aborted EP1 val_WSS=13.86% (β1=0.97, β2=0.98) |
| β1↑ + β2↑ joint | ⏳ delayed converger | H150 (β1=0.97, β2=0.985) — see below |
| **β-grid optimum** | **⭐ H147 (0.95, 0.98)** | SOTA at 6.5409% test_WSS |

**β-space is fully exhausted.** Only H150 remains as an active β-grid arm and it is a delayed converger.

### Fleet status (2026-05-29 10:30Z)

| Run ID | Student | H# | Epoch | val_WSS | val_ABUPT | State | vs H147 |
|---|---|---|---:|---:|---:|---|---|
| ew63yb7p | fern (dl24) | **H157b** (wss_charbonnier=0.1 axes=z on H150-β stack) | **9.28** | **6.5085%** | (last 6.078% EP6.91) | **🏆 LEADING — running** | val_WSS already below H147 final val by 0.08pp; descent rate -0.103pp/EP; projected test_WSS 6.41-6.46% (beats H147 by 0.08-0.13pp); EP15 terminal ~14:46Z; **WATCH test_SP** (H150-β stack may cross 3.577% cap like H150 did) |
| (launching) | frieren (dl24) | **H159** (vol_p_charbonnier=0.3 on CORRECTED H147 stack) | — | — | — | PR #1423 | 8-EP T_max=8; tests heavier VP reg to free GradNorm head-weight for WSS; deadline 15:25Z |
| (launching) | tanjiro (dl24) | **H160** (β1=0.95, β2=0.985 — last untested β-grid cell) | — | — | — | PR #1424 | 8-EP T_max=8; isolates β2-only mover; under corrected baseline, β-grid fully closes regardless of outcome |
| (assigning) | nezuko (dl24) | **H161** (cosine 2-cycle restart on H147 stack) | — | — | — | PR pending | natural follow-up to nezuko's own convergence-ceiling finding from H151; 8-EP restart at EP4; deadline ~15:30Z |
| d20sf8th | nezuko (dl24) | H151 (extended 45EP canonical) | 31 (truncated by wall-clock) | 6.5894% (EP33 final visible) | 7.27% (terminal) | **CLOSED PR #1360 10:30Z** | ⛔ test_WSS=6.5439% (+0.003pp noise tie); regresses VP/SP/ABUPT, crosses SP cap by +0.035pp; **best-val=EP25 in BOTH H151 and H147 = recipe at convergence ceiling** |
| 5bgp2ryq + (eval) | (tanjiro prev) | H150 (β1=0.97/β2=0.985) | 30 (terminal, EMA EP20) | 6.6223% (val EMA) | 5.8831% (val EMA) | **CLOSED PR #1359 09:50Z (β-decoupling retracted 10:30Z)** | ⛔ test_WSS=6.5650% (+0.024pp); ⛔ test_VP=3.5016% (+0.10pp vs corrected H147 3.4014%); ⛔ test_SP=3.6088% (over 3.577% cap by +0.032pp); H150 regressed ALL 4 metrics |
| wyf77dqa | (frieren prev) | H158 (vol_p_charbonnier=0.1, stripped stack) | 2 (aborted) | 16.11% (EP1) | 14.32% (EP1) | **CLOSED PR #1420 09:50Z** | ⛔ ADVISOR ERROR: missing 4 H147 SOTA flags |
| ugpyo62a + 66ys15yn | (frieren prev) | H156 (β1=0.97/β2=0.985 + lr=9e-5) | 10 (terminal) | 6.8949% (val) | 6.2370% (val) | **CLOSED PR #1369 08:23Z** | ⛔ test_WSS=6.6909% (+0.150pp), test_VP=11.8441% (3.3x cap) — val/test VP gap 2.84x finding |
| 9xo566ws | (fern prev) | H155 (lr=9e-5) | 10.27 | 7.092% | 6.157% | **CLOSED PR #1368 03:00Z** | ⛔ test_WSS=6.8936%, lr-axis falsified |
| 8w7qtm5e | (frieren prev) | H154 (tau_z=1.3) | 7.64 | 6.991% | 6.123% | **CLOSED PR #1367 22:57Z** | ⛔ EP7 falsified |

### CRITICAL: H147 SOTA stack is broader than BASELINE.md reproduce command shows

**Discovered 2026-05-29 09:37Z (via frieren's diff of `wyf77dqa` vs `k6q4c3on`).** The H147 SOTA reference run `k6q4c3on` (PR #1344) uses these flags that are **`False`/`0` by default** in `train.py` and must be explicitly set:

```
--use-curvature-attention-bias              # default False
--use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15   # default False
--use-y-symmetry-aug --y-symmetry-aug-prob 0.5   # default False
--wss-charbonnier-weight 0.1 --wss-charbonnier-axes z   # default 0
--vol-p-charbonnier-weight 0.1 --vol-p-charbonnier-eps 1e-3   # default 0
```

**BASELINE.md reproduce command (line 80+) is OUTDATED** — does not list these flags as defaults. **Any new advisor PR must include them explicitly** or the run will not reproduce H147 SOTA. H158 PR #1420 was killed at EP1 because four of these flags were silently dropped. H159 PR #1423 and H160 PR #1424 include the full corrected stack.

**Bug-fix commit e31fd60 (on tanjiro's H150 branch)**: adds `scripts/precompute_curvature_proxy.py` + `curvature_proxy_stats_k16_v1.json` (400-case train stats), unblocking `--use-curvature-attention-bias` from FileNotFoundError on cold environments. Should be cherry-picked into advisor branch independently of H150 closure.

### H147 SOTA trajectory reference

| EP | 1 | 2 | 3 | 5 | 10 | final test |
|---:|---:|---:|---:|---:|---:|---:|
| H147 val_WSS | 12.82% | 7.26% | 6.98% | 6.75% | 6.64% | **6.5409%** |

### H150 terminal — CLOSED PR #1359 (no merge); projections retracted

**Terminal test (EMA-best EP20):** test_WSS=6.5650% (+0.024pp vs H147), test_VP=3.5016%, test_SP=3.6088%, test_ABUPT=5.7188%.

**Retracted (2026-05-29 10:30Z):** The "wins by 0.02pp" projection above was based on the WRONG H147 baseline (BASELINE.md ea99dda bug). Under the corrected H147 (test_VP=3.4014%, test_SP=3.5634%), H150 regressed on ALL 4 metrics and crossed the PR #972 SP cap by +0.032pp. β-decoupling finding RETRACTED — H147 (β1=0.95, β2=0.98) is the joint optimum on every test metric, not just WSS. Val→test gap on H150 was ~0.06pp (val 6.6223 → test 6.5650), which is smaller than the +0.10pp assumption used in projections — **so projections built on -0.10pp val-test gap should be tightened to ~-0.05 to -0.07pp for this stack family**.

### H151 per-epoch val trajectory (d20sf8th)

| EP | 7 | 9 | 10 | 11 | 12 | 13 | 14 | **15** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 6.906% | 6.891% | 6.870% | 6.868% | 6.848% | 6.844% | 6.8362% | **6.8306%** |
| val_VP | — | — | — | — | — | — | 3.703% | **3.703%** |
| val_ABUPT | — | — | — | — | — | — | 6.098% | **6.092%** |

Linear descent −0.005pp/EP at EP14→EP15 (matching H150's slowdown). Drift to H147 stable at +0.20pp. Confirms **H150's improvement is genuine β-config gain, not RNG**: same EP15 same descent, only +0.20pp gap. Run has ~30 EP remaining. Extrapolation: EP45 ~6.71% (no longer projected to beat H147 final). Useful as RNG-baseline confirmation arm only.

### H154 per-epoch val trajectory (8w7qtm5e) — ⛔ ABORTED EP7 DESCENT REVERSAL

| EP | 1 | 2 | 3 | 4 | 5 | 6 | **7** | Δ EP6→EP7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_WSS | 15.54% | 7.77% | 7.2375% | 7.0698% | 7.0035% | 6.9719% | **6.9906%** | **+0.019pp** |
| val_WSS_x | 13.41% | 6.66% | 6.236% | 6.122% | 6.067% | 6.045% | **6.072%** | +0.028pp |
| val_WSS_y | 17.76% | 8.92% | 8.127% | 7.857% | 7.755% | 7.695% | **7.686%** | -0.009pp ✅ |
| val_WSS_z | 20.69% | 10.37% | 9.741% | 9.526% | 9.470% | 9.446% | **9.479%** | **+0.033pp** ⚠️ |
| val_VP  | 8.175% | 3.790% | 3.488% | 3.396% | 3.382% | 3.369% | **3.379%** | +0.009pp |
| val_ABUPT | 13.90% | 6.836% | 6.345% | 6.184% | 6.134% | 6.107% | **6.123%** | +0.016pp |

**Falsification at EP7:** WSS_z (the channel tau_z=1.3 was specifically weighting) reversed by +0.033pp — the weighted head saturated. WSS_y (unweighted) still descending = direct evidence the tau_z weighting caused the reversal. Aborted PR #1367 by advisor decision at 22:28Z.

**Mechanistic learning preserved:** tau_z reweighting works as **early-EP volume_p lever** (val_VP reached H147 EP30-territory by EP6=3.369%) but does NOT produce terminal WSS gain on canonical lr=1e-4 stack. Would need lr-floor adjustment (i.e., compound with H155 lr=9e-5) to test full hypothesis.

### H155 primary EP1-EP9 — EP10 KILL CONFIRMED ⛔ (9xo566ws)

Created 19:37:55Z. Config verified: `lr=9e-5`, `lion_beta1=0.95`, `lion_beta2=0.98`, `tau_z_loss_weight=1`, `epochs=30`, `lr_cosine_t_max=30`, `ema_decay=0.999`, `ema_start_step=500`. DDP8 active.

| EP | H155 val_WSS | H155 val_ABUPT | per-EP Δ |
|---:|---:|---:|---:|
| 1 | 16.34% | 14.87% | (warmup) |
| 2 | 7.804% | 6.837% | −8.54 |
| 3 | 7.361% | 6.416% | −0.44 |
| 4 | 7.223% | 6.293% | −0.14 |
| 5 | 7.169% | 6.240% | −0.054 (EP5 gate ✅) |
| 6 | 7.134% | 6.208% | −0.035 |
| 7 | 7.131% | 6.203% | −0.003 |
| 8 | 7.112% | 6.179% | −0.019 |
| **9** | **7.109%** | **6.176%** | **−0.003** |

**EP10 hard kill gate (>6.80%): WILL FAIL by ~0.30pp.** Descent stuck at -0.003pp/EP since EP6 — cosine deepening has NOT triggered re-acceleration as it did for H150. The lr=9e-5 effective LR at deep cosine (~5.7e-5) is too low to make further progress.

**Lr-axis falsified standalone:** lr=9e-5 on canonical H147 stack does NOT beat H147 at any epoch. The early-EP catch-up from EP1→EP5 reverses into plateau drift by EP6+.

**CLOSED 03:00Z (PR #1368):** fern posted terminal SENPAI-RESULT 02:53Z — test_WSS=**6.8936%** (regression vs H147 6.5409% by +0.35pp). EMA best-val checkpoint EP10. W&B rank0=9xo566ws. Standalone lr=9e-5 axis **falsified**. H157 (wss-charbonnier compound) assigned to fern at PR #1378.

### H156 primary EP1-EP7 — β + lr compound GAP REVERSING ⚠️ (ugpyo62a)

Launched 00:40Z after `grdap1rg` smoke passed clean. Config: `lion_beta1=0.97`, `lion_beta2=0.985`, `lr=9e-5`, canonical H147 stack otherwise. DDP8 active, 8 W&B run-IDs in group (rank 0 = `ugpyo62a` canonical).

| EP | H156 val_WSS | H147 reference | H156−H147 offset | Trend |
|---:|---:|---:|---:|---|
| 1 | 13.154% | 12.82% | +0.33pp | warmup |
| 2 | 7.572% | 7.26% | +0.31pp | ↘ closing |
| 3 | 7.182% | 6.98% | +0.20pp | ↘ closing |
| 4 | 7.042% | ~6.86% (interp) | +0.18pp | ↘ closing |
| **6.92** | **6.9527%** | **~6.70% (interp)** | **+0.25pp** | **⚠️ REVERSING** |

**Gap-closing reversed EP4→EP7.** Offset jumped from +0.18pp (EP4) to +0.25pp (EP7). Per-EP descent rate H156 EP4→EP7 = -0.030pp/EP vs H147 EP4→EP7 ≈ -0.053pp/EP — H156 descending **slower** than H147 at the same epoch. The compound β+lr theory predicted late-cosine acceleration but it's manifesting as a slower-grind regime instead.

**Updated EP10 projection (gap-widening at +0.023pp/EP, rate slowing):**
- EP10 ≈ 6.86% (gap +0.22pp) → would FAIL EP10 soft kill >6.65% by 0.21pp
- EP15 ≈ 6.74% (extrapolated) → still above H147 EP15 ~6.70% baseline by ~0.04pp
- EP30 ≈ 6.55% terminal — possible but no longer probable (would need late-cosine acceleration to close +0.22pp gap in 23 EP)

**Decision: ride to EP10 soft kill (~07:25Z).** Direction pre-committed: if EP10 WSS >6.65% AND gap >+0.10pp → harvest EMA-best, do NOT extend to EP30. EP10 reading determines the call.

**TERMINAL OUTCOME (08:13Z, PR #1369 CLOSED 08:23Z):**

| Metric | H156 val | H156 test | val/test ratio | H147 val/test ratio |
|---|---:|---:|---:|---:|
| WSS | 6.8949% | 6.6909% | 0.97x (test slightly better) | ~1.0x |
| SP  | 4.1013% | 3.7103% | 0.90x | ~1.0x |
| **VP** | **4.1687%** | **11.8441%** | **2.84x** ⚠️ | **~1.0x** |
| ABUPT | 6.2370% | 7.4963% | 1.20x (VP-dominated) | ~1.0x |

**Research finding — val_VP/test_VP gap = 2.84x is unique to compound β+lr.** H147 has matched val/test on VP (~1.0x). H156 compound config (β1=0.97/β2=0.985 + lr=9e-5) drives the optimizer into a val-distribution-specific local minimum on VP. Mechanism hypothesis: very small effective step in flat directions (high-β momentum + low-lr) tracks val-set features without exploring volume-pressure-generalizing directions. Late-cosine renewal exists (-0.034pp EP9→EP10 confirms it) but converges to a val-specific minimum that does not transfer to test.

This finding **directly motivates H158** (vol_p_charbonnier on H147 stack) — the only single-flag VP-regularization lever left in train.py.

### H158 (frieren, ASSIGNED 08:25Z) — vol_p_charbonnier=0.1 on H147 stack

**Hypothesis:** H156 revealed compound β+lr induces structural VP val/test gap (2.84x). Charbonnier loss on volume pressure (smooth-near-zero L1-like) should regularize VP and close the val/test gap by penalizing outlier volume points without distorting bulk learning. Symmetric to H157b's wss_charbonnier (in flight on H150 stack on the WSS axes).

**Config:** H147 SOTA stack (β1=0.95/β2=0.98, lr=1e-4, 6L/512d/4h/128slices, surface_out_width_factor=2.0, STRING-multisigma sigmas 0.25/0.5/1.0/2.0/4.0, ema_decay=0.999) + `--vol-p-charbonnier-weight 0.1 --vol-p-charbonnier-eps 1e-3`, 10 EP cosine T_max=10. Single seed.

**Kill ladder:**
- EP1 > 13.5% kill
- EP3 > 7.20% kill (must beat H147 EP3=6.98% by margin compatible with Charbonnier overhead)
- EP5 > 6.85% kill
- EP10 terminal harvest (EMA-best if val ≤ 6.65% AND val_VP < 4.0% — the second condition is the test of the H158 hypothesis)

**Expected:** if Charbonnier closes the VP gap, test_VP should land in 3.5-4.0% range (H147 SOTA had test_VP=3.6033%). If test_VP > 5.0%, Charbonnier doesn't address the structural gap and we close.

---

## Decisive epochs ahead (active runs)

**⚠️ HUMAN DEADLINE 15:45Z** (5h15m from current 10:30Z) — all terminal harvests must land before this.

- **H157b EP10 soft kill** (~11:34Z) — gate >6.65%; EP9.28=6.5085% **already below**, expect EP10 PASS comfortably
- **H159/H160 EP1 read** (~10:40Z) — both must show val_WSS < 13.5% (H147 EP1=12.82%)
- **H161 launch (nezuko)** (~10:45Z) — cosine 2-cycle restart on H147 stack; tests if recipe-ceiling can be broken by mid-budget LR re-warmup
- **H159/H160/H161 EP3 kill check** (~12:30Z) — val_WSS gate ≤ 7.20%
- **H157b EP12-13** (~13:30Z) — descent renewal check; if continuing -0.05+/EP, ride to EP15
- **H159/H160/H161 EP5 kill check** (~13:30Z) — val_WSS ≤ 6.85%
- **H157b EP15 terminal** (~14:46Z) — EMA-best harvest for test eval; **post test_WSS AND test_SP** (SP cap watch)
- **H157b test-eval complete** (~15:10Z) — 25min buffer
- **H159 EP8 terminal + test eval** (~15:25Z) — 20min buffer
- **H160 EP8 terminal + test eval** (~15:30Z) — 15min buffer
- **H161 EP8 terminal + test eval** (~15:30Z) — 15min buffer
- **HARD DEADLINE 15:45Z** — all test_eval harvests must complete before this; any in-flight after must abort and post partial SENPAI-RESULT

---

## Potential next research directions (after H159/H160 terminal, given 09:50Z state)

### Wave 41 β-grid status (after H160 lands)

```
        β1=0.93    β1=0.95    β1=0.97
β2=0.97  H149⛔    H152⛔     —
β2=0.98            H147⭐     H153⛔
β2=0.985           H160⏳     H150⛔ (all 4 regress)
```

H160 is the last untested cell. Under the corrected H147 baseline, **β-grid is functionally closed already** — H147 dominates every neighboring cell on every metric. H160 (β1=0.95 + β2=0.985) is now an isolation experiment: confirms whether β2=0.985 is the harmful axis (likely yes, given H150's full regression) or if β1=0.97 is the sole harmful axis. Outcomes:

- H160 regresses like H150 → β2=0.985 is harmful axis confirmed (and β2-up is dead).
- H160 ties H147 → β2 axis is neutral around 0.98, only β1 matters (β1=0.95 confirmed sweet-spot).
- H160 beats H147 → unexpected; reopens β2-up direction (low prior).

### Branch A — H157b wins (test_WSS ≤ 6.50%, test_SP ≤ 3.577%)

H157b is the **leading current candidate** (EP9.28 val_WSS=6.5085% already below H147 final val).

1. **Merge winner** as new dl24 SOTA (H147 dethroned). Note: H157b runs on H150-β stack (β1=0.97/β2=0.985) so check carefully that test_SP doesn't cross 3.577% cap (H150 itself crossed it at 3.6088%).
2. **Compound**: H157b stack + heavier wss_charbonnier weight (0.2, 0.3) — explore the WSS-reg axis further
3. **Cross-compound**: H157b stack swapped onto H147-β (β1=0.95/β2=0.98) — disentangle Charbonnier-only effect from compound stack

### Branch B — H159 wins (test_WSS ≤ 6.50%)
1. **Merge winner** as new dl24 SOTA
2. **Compound**: H159 stack + Charbonnier weight=0.5 or 1.0 sweep
3. **Cross-compound**: H159 + H157b's wss_charbonnier axes=z — VP+WSS axis combined

### Branch C — H161 wins (cosine restart breaks ceiling)
1. **Merge winner** as new dl24 SOTA
2. **Compound**: H161 restart schedule on H157b/H159 stacks — combined ceiling-break + Charbonnier
3. **Schedule axis open**: try SGDR with 3-cycle, cyclic LR, warm restarts with larger LR

### Branch D — nothing wins (Wave 41 fully closed)

If neither H157b nor H159/H160/H161 beats H147, Wave 41 single-knob axis is closed. Tier-shift required for Wave 42:

1. **Capacity bump**: `hidden_dim=512→640` or `model_layers=6→7` on H147 stack — first capacity-axis test
2. **Per-axis WSS reformulation**:
   - `wss_charbonnier_axes=y,z` (extend from current z-only — tau_z=8.49% is the floor)
   - Independent `wss_charbonnier_weight` per axis (requires train.py change)
3. **PE bandwidth**: extend STRING-multisigma sigmas to `0.125,0.25,0.5,1.0,2.0,4.0,8.0` — wider band for finer WSS detail
4. **GradNorm hyperparams**: `gradnorm_alpha=0.3` (more equal head weighting) or `min_w_vol_p=0.05` (let VP head lose weight)
5. **Decoupled optimizer per head** (more architectural): separate Lion state per parameter group. Tests whether the per-head learning-rate-effective gap explains anything beyond what GradNorm already balances.
6. **Best-val/EMA selection metric tuning**: H151 found best-val=EP25 in BOTH 30-EP and 45-EP runs — suggests EMA half-life vs cosine-LR-floor interaction. Try `ema_decay=0.9995` (longer half-life) to push the best-val later.

### Pre-shutdown contingency (if 15:45Z arrives with no merge winner)

- Cherry-pick e31fd60 (curvature precompute bug-fix) regardless of H150/H159/H160 outcome
- Update BASELINE.md reproduce command with the 4 missing H147 SOTA defaults so future advisors don't repeat the H158 advisor error
- File a publishable-finding note on val/test VP gap (2.84x H156, ~1.0x H147, H150 ratio) + β-decoupling (pressure vs WSS prefer different β)
- Document that Wave 41 β-grid closure is the wave's conclusion

---

## Key technical constraints (canonical H147 config — **CORRECTED 09:50Z**)

```bash
torchrun --nproc_per_node=8 target/train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --batch-size 1 --weight-decay 0.005 \
  --train-surface-points 65000 --train-volume-points 65000 \
  --eval-surface-points 65536 --eval-volume-points 65536 \
  --epochs 30 --lr 1e-4 --lr-cosine-t-max 30 --lr-warmup-epochs 1 \
  --optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
  --ema-decay 0.999 --ema-start-step 500 \
  --model-hidden-dim 512 --model-layers 6 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" --pe-num-features 16 \
  --surface-out-width-factor 2.0 \
  --vol-p-charbonnier-weight 0.1 --vol-p-charbonnier-eps 1e-3 \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-axes z \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15 \
  --use-curvature-attention-bias --use-y-symmetry-aug --y-symmetry-aug-prob 0.5
```

**⚠️ The last four lines (Charbonnier, GradNorm, curvature, y_sym_aug) were missing from BASELINE.md's listed reproduce command and from the H158 PR — they are `False`/`0` defaults in `train.py` and must be set explicitly. Causing H158 to skip them produced an EP1 val_WSS=16.11% disaster.**

**Dataset:** `rawcanon_20260511` (corrected split — old split had +7-8pp artifact)  
**EMA:** eval EMA with decay=0.999, start_step=500 (already wired in train.py)  
**Optimizer:** Lion (NOT Adam) — β changes have outsized impact on early-epoch trajectory  
**DDP8:** 8-GPU distributed training, effective batch=8  
**Curvature stats file:** `curvature_proxy_stats_k16_v1.json` required at repo root when `--use-curvature-attention-bias` is set; bug-fix commit e31fd60 (tanjiro's H150 branch) generates it — cherry-pick into advisor branch.  

---

## Historical context: How we arrived at H147 SOTA

| Wave | Key finding | Best test_WSS |
|------|-------------|--------------|
| Pre-wave | H39 wider surface_out MLP (factor=2.0) | 6.6506% (PR #1284) |
| Wave 36 (H138-H148) | Disentanglement: β-drift is single driver; wd-drift null; curvature null | 6.5409% (H147) |
| Wave 41 (H149-H160) | β-grid mostly closed; lr-axis falsified; compound β+lr induces val-VP overfit; β-decoupling finding (pressure vs WSS prefer different β); BASELINE reproduce command discovered incomplete | TBD (H157b/H159/H160 in flight) |

The gap from current SOTA (6.5409%) to Transolver-3 SOTA (5.85%) is **0.69pp**. β-sweep closed 0pp (H147 stays optimum after H150 +0.024pp regression on test_WSS). H160 (the last untested β-grid cell, β1=0.95/β2=0.985) plus H159 (heavier VP Charbonnier on corrected H147) plus H157b (heavier WSS Charbonnier on H150-β) are Wave 41's terminal trio. If none beats H147, single-knob axis is closed and Branch C (architectural / capacity / decoupled-β) is required.

---

## References

- Issue #1056 — active research directive and hard constraints
- BASELINE.md — locked scoreboard with corrected-split results (**reproduce cmd missing 4 flags — see CRITICAL note above**)
- PR #1344 — H147 merge commit with full β-attribution analysis
- PR #1359 — H150 β1=0.97/β2=0.985 (tanjiro, **CLOSED 09:50Z** non-merge — test_WSS=6.5650% +0.024pp; **β-decoupling finding RETRACTED 10:30Z** — H150 actually regressed all 4 metrics vs corrected H147 baseline; corrigendum posted)
- PR #1360 — H151 45-EP extended (nezuko, **CLOSED 10:30Z** non-merge — test_WSS=6.5439% +0.003pp noise tie; **convergence ceiling finding**: best-val=EP25 in BOTH H151 and H147)
- PR #1367 — H154 tau_z=1.3 (frieren, CLOSED 22:57Z aborted_descent_reversal_ep7)
- PR #1368 — H155 lr=9e-5 (fern, CLOSED 03:00Z lr-axis falsified)
- PR #1369 — H156 β1=0.97/β2=0.985 + lr=9e-5 compound (frieren, **CLOSED 08:23Z** — val/test VP gap 2.84x finding)
- PR #1378 — H157 (auto-merged in error, superseded by PR #1385)
- PR #1385 — H157b wss_charbonnier=0.1 on H150 stack (fern/tay, RUNNING EP6.91)
- PR #1420 — H158 vol_p_charbonnier=0.1 (frieren, **CLOSED 09:50Z aborted_advisor_error**: reproduce cmd dropped 4 H147 SOTA flags)
- **PR #1423 — H159** vol_p_charbonnier=0.3 on CORRECTED H147 stack (frieren, ASSIGNING 09:50Z)
- **PR #1424 — H160** β1=0.95 β2=0.985 missing β-grid cell (tanjiro, ASSIGNING 09:50Z)
- **Bug-fix commit e31fd60** (on tanjiro H150 branch) — `scripts/precompute_curvature_proxy.py` + `curvature_proxy_stats_k16_v1.json` — needs cherry-pick into advisor branch
- `RESEARCH_IDEAS_*.md` — researcher-agent design briefs from prior waves
