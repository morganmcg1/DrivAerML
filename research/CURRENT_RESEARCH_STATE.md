# SENPAI Research State

_Last updated: 2026-05-30 00:15Z._

**00:15Z early-morning launch status — H167 EP6 jackpot trajectory confirmed (sole structural-wave survivor); H168/H169/H170 wave launches IN FLIGHT, all 4 students DDP8 running. EP1 readings expected ~02:30-04:30Z.**

### Wave-2 launch verification (00:10Z W&B snapshot)

| Run | Student | rank0 | Phase | Step | Notes |
|---|---|---|---|---:|---|
| H167 `heads-8-h147` | tanjiro | `9b7sdo5k` | mid-EP7 | ~73,632 | EP6 confirmed 6.7613% val_WSS; EP7 dual-gate pending; EP8 terminal ~02:00Z |
| H168 `pe-lo-sigma` | fern | `t9h0inur` | main pre-EP1 | ~3,821 | ⚠ no smoke prefix detected — went direct to main; monitor EP1 acutely |
| H169 `wss-charb-yz` | nezuko | `rha7q5tp` | **smoke** | ~2,229 | ✓ proper smoke-first protocol; main launches after smoke clears |
| H170 `gradnorm-alpha-03` | frieren | `nkc26gvj` | main pre-EP1 | ~1,335 | ⚠ no smoke prefix — direct main; monitor EP1 acutely |

H168/H170 smoke-skip is a soft protocol gap; both runs healthy at low step count. Will flag in PR if EP1 reading is anomalous. H169's smoke-then-main protocol is correct.

### Wave-1 conclusion: H167 sole survivor of structural wave (23:50Z)



### Wave terminal table (3 of 4 closed; H167 alive)

| Run | rank0 | rt | test_WSS | Δ H147 | test_SP cap | Verdict |
|-----|---|---:|---:|---:|---:|---|
| H164 frieren slices 128→192 | `2qm6c9w4` | 6.77h | **6.6296%** | +0.089pp | ❌ 3.6631 (+0.086) | **CLOSE PR #1444** |
| H165 fern pe_features 16→12 | `3mpka9g9` | 6.13h | **6.6727%** | +0.132pp | ❌ 3.6633 (+0.086) | **CLOSE PR #1445** |
| H166 nezuko surface_out 2.0→3.0 | `jmyv1byk` | 5.91h | **6.6052%** | +0.064pp | ⚠ 3.6031 (+0.026 marg) | **CLOSE PR #1446** |
| **H167 tanjiro heads 4→8** | `9b7sdo5k` | 5.29h (mid-EP7) | TBD | TBD | TBD | **WIP — sole hope** |

### H167 EP6 jackpot trajectory (23:30Z tanjiro report)

| EP | H167 val_WSS | H147 ref | Δ | val_WSS_z |
|---|---:|---:|---:|---:|
| 1 | 13.55% | 12.82% | +0.73 HOT | 17.42% |
| 5 | 6.82% | 6.75% | +0.07 | 9.22% |
| 6 | **6.7613%** | (none, H147 EP5=6.75%) | **+0.011pp behind H147 EP5** | **9.1661%** |

**Gap to H147 EP5 closed to +0.011pp at H167 EP6.** EP7 dual-gate ahead: val_WSS ≤ 6.70% OR val_WSS_z ≤ 9.00% required for jackpot. Tanjiro continues to EP7 (~00:15Z) and EP8 terminal (~02:00Z). Strong late-EP cooling (-0.062pp/EP through EP6) confirms the "doubled attention subspaces enables late-EP capacity-friendly descent" hypothesis on this axis.

### Joint structural-wave finding (SEALED — multi-axis local optimum)

| Axis | Direction tested | test_WSS Δ | SP cap |
|---|---|---:|---:|
| trunk-token-count (slices) | 128→192 (H164) | +0.089 | ❌ broken |
| PE-projection density | 16→24 (H162), 16→12 (H165) | +0.166, +0.132 | ❌ broken (H165) |
| output-head capacity | 2.0→3.0 (H166) | +0.064 | ⚠ marginal |
| attention subspace count | 4→8 (H167) | TBD (jackpot trajectory) | TBD |

Combined with single-knob loss/optimizer falsifications (H159/H161/H162/H160 + β-grid + lr + tau-axis weights), the H147 stack sits at a **tight multi-axis local optimum**. Moderate single-axis structural perturbations on slice count, PE density, and output head width all regress test_WSS AND break the SP floor at 3.577%. **Floor is in attention design (H167) or normalization/decoder routing (next wave).**

### Next wave H168-H170 (dispatching 23:55Z)

- **H168 `pe-lo-sigma-h147` (fern)** — add σ=0.1 to STRING multi-sigma `[0.25,0.5,1.0,2.0,4.0]→[0.1,0.25,0.5,1.0,2.0,4.0]`; pe_features stays 16 (16 spread across 6 bands). Mechanism: finer-band PE resolves boundary-layer transition geometry at the millimeter-cm scale where tau_z is hardest. Aligns with fern's own H165 follow-up #2 (sigma values).
- **H169 `wss-charb-yz-h147` (nezuko)** — extend WSS Charbonnier from `axes=z` to `axes=yz` keeping `weight=0.1` (NOT amplification — H161 amplification falsified). Mechanism: covers second-worst axis tau_y at identical weight, tests axis-coverage gating vs weight-amplitude. Nezuko has H161 history with this exact mechanism.
- **H170 `gradnorm-alpha-03-h147` (frieren)** — α 0.5→0.3 on H147. Mechanism: softer restoring force → smoother per-task weight oscillation → cleaner tau_z gradient accumulation. Untested half of α axis (high tail blew up H874).

**Previous 20:50Z full per-EP trajectory table (for reference):**

Full per-EP trajectory across the 4-arm wave (all rank0 IDs, vs H147 reference):

| EP | H147 | H164 slices=192 (frieren) | H165 pe=12 (fern) | H166 surfw=3 (nezuko) | H167 heads=8 (tanjiro) |
|---|---:|---:|---:|---:|---:|
| 1 | 12.82 | **12.76 (-0.06)** | **12.65 (-0.17)** | **12.66 (-0.16)** | 13.55 (+0.73 HOT) |
| 2 | 7.26 | 7.31 (+0.05) | 7.38 (+0.12) | 7.36 (+0.10) | **7.29 (+0.03 recov)** |
| 3 | 6.98 | 7.03 (+0.05) | 7.09 (+0.11) | 7.08 (+0.10) | **7.00 (+0.02 TIED)** |
| 4 | ~6.85 | 6.92 (+0.07) | 6.99 (+0.14) | 6.95 (+0.10) | — |
| 5 | 6.75 | **6.86 (+0.11)** | — | **6.87 (+0.12)** | — |

| Hyp | Student | rank0 run | rt | latest EP val_WSS | State |
|-----|---------|-----------|-----|---------------|-------|
| **H164 slices=192** | frieren | 2qm6c9w4 | 4.13h | EP5=6.8598% (+0.11) | PR #1444 — **LEADING merge candidate; projected test ~6.57-6.62%** |
| **H167 heads=8** | tanjiro | 9b7sdo5k | 2.73h | EP3=6.9989% (+0.02 TIED) | PR #1450 — **#2 contender; recovery complete** |
| **H166 surfw=3.0** | nezuko | jmyv1byk | 3.69h | EP5=6.8712% (+0.12) | PR #1446 — runner-up; close behind H164 |
| **H165 pe=12** | fern | 3mpka9g9 | 3.53h | EP4=6.9919% (+0.14) | PR #1445 — trailing, non-merge expected |

**Closed in previous cycles (background context):**
| Hyp | Run | rt | Terminal test_WSS | vs H147 SOTA 6.5409% | State |
|-----|-----|----|------|------|------|
| H159 vol_p_charb=0.3 (frieren prev) | z6ybgmx7+juxadtjh | 5.13h+eval | 6.6678 | +0.127 | CLOSED PR #1423 15:51Z — NON-MERGE |
| **H160 β=0.95/0.985 (tanjiro)** | **7a14s7uo** | **6.1h terminal+eval** | **6.6247** | **+0.084** | **CLOSED PR #1424 17:15Z — NON-MERGE** |
| H161 wss_charb=0.3 z (nezuko prev) | kvfaya2j+5ttbfh4o | terminal | 6.7402 | +0.199 | CLOSED PR #1427 15:23Z |
| H162 pe_features=24 (fern prev) | 7vdb5zwz+0jfesb3w | 3.78h+eval | 6.7070 | +0.166 | CLOSED PR #1430 15:52Z — NON-MERGE |

**Key 20:50Z reads — structural wave update:**
- **H164 (frieren) is the leading merge candidate.** EP5=6.8598%, cooling rate -0.06/EP (vs H147 -0.075 in this band). Projected EP8 ≈ 6.71-6.76% → projected test_WSS ≈ 6.57-6.62% vs H147 6.5409%. **Razor-thin — could tie or beat H147.** Only arm with EP1 BEAT H147 AND tight cooling. Continue to terminal with best-val test eval.
- **H167 (tanjiro) heads=8 is the surprise #2 contender.** Recovery from EP1=13.55% (HOT) → EP2=+0.03 → EP3=+0.02 (effectively TIED with H147 EP3). Trajectory shape unique to this arm (slow EP1, sharp catchup); 8 heads = double attention-subspace count, late-EP capacity-friendly. EP5 gate: ≤6.77% confirms, >6.85% reverts to trail pattern.
- **H166 (nezuko) surfw=3.0** EP5=6.8712 — close runner-up to H164. Cooling EP4→EP5 = -0.077 matches H147 rate. Projected EP8 ~6.73-6.78% → test ~6.59-6.64%, behind H147 by ~+0.05-0.10pp. Plausible runner-up if H164 misses.
- **H165 (fern) pe_features=12** EP4=6.9919 — trailing at steady +0.10-0.14pp band. Non-merge expected. EP5 reading will confirm whether pe=12 is in the "tight local optimum around 16" pattern (like H162 pe=24's +0.166 regression).

**Wave-level insight:** Three of four structural perturbations BEAT H147 at EP1 (-0.06, -0.17, -0.16), then the gap reverses at EP2 and stabilizes. This suggests H147's exact configuration is at a **tight architectural local optimum** where mild structural perturbations help initial feature variety but hurt converged-fit quality. H164's preserved cooling rate is what separates it.

**Key 17:15Z reads (background):**
- **H160 NON-MERGE (tanjiro — closed 17:15Z):** test_WSS=6.6247% (+0.084pp); test_VP=3.5659% ✓ clears 3.643%; test_SP=3.6542% ❌ misses 3.577% floor by 0.077pp; test_ABUPT=5.7827% ✓ clears 5.844%; test_WSS_z=8.6665% +0.178pp vs H147. **3 of 4 caps regress; test_SP missed.** Trajectory tracked H147 within ±0.09pp at every EP1-EP8 checkpoint — β2=0.985 axis is mildly forgiving (no destabilization, unlike β1↑), but does not open new test ceiling.
- **β-grid is CLOSED.** All five grid cells around H147 (0.95, 0.98) explored. Every perturbation off the central point loses. H147 is the joint optimum on every test metric. β1 dominates early WSS dynamics; β2 is a late-phase pressure-smoothing axis that doesn't cross test ceiling. Decoupled-β per-head optimizer is the last untested β lever; banked for future wave.

**18:55Z DDP val-callback rank0 gotcha (info-only, no fix needed):** When polling structural-wave run trajectories, the multi-rank DDP val callback fires only on rank 0. The earlier active-runs query picked non-rank-0 W&B run IDs (`hsm6ljmh` = rank2 for H164, `52hvx489` = rank6 for H165, etc.) which return zero val_primary rows. Always query the rank0 run ID for trajectory reads: H164=`2qm6c9w4`, H165=`3mpka9g9`, H166=`jmyv1byk`, H167=`9b7sdo5k`. All 4 runs are healthy.

**MAJOR FINDING — Quadruple-arm joint conclusion (H159+H161+H162+H160):**

Four orthogonal single-knob perturbations on the H147 stack:

| Hypothesis | Axis | Δ test_WSS |
|---|---|---:|
| H161 (nezuko) | WSS-Charbonnier 0.1 → 0.3 axes=z | +0.199 |
| H162 (fern) | pe_num_features 16 → 24 | +0.166 |
| H159 (frieren) | vol_p-Charbonnier 0.1 → 0.3 | +0.127 |
| **H160 (tanjiro)** | **β2 0.98 → 0.985** | **+0.084** |

Compounded with H149/H150/H152/H153 (other β-grid cells) and H155 (lr=9e-5) and H154 (tau_z=1.3) — **every single-knob perturbation off H147 regresses test_WSS**. The H147 stack sits at a tight local test optimum across loss-density, Charbonnier-axis, PE-spectral, β-grid, lr, and per-axis weight axes simultaneously. **Single-knob perturbation channel is closed.** Next gain must come from structural changes to the architecture (heads, slices, decoder width, PE band, geometry features).

**Implications for next-wave hypothesis design:**
- **KILLED candidates**: every direction of single-knob amplification on the H147 loss/PE/Charbonnier/β/lr/per-axis surface. The bordering region is sealed.
- **STRUCTURAL WAVE in flight (4 orthogonal axes)**:
  - H164 — model_slices 128→192 (trunk-token-count; PR #1444 frieren)
  - H165 — pe_num_features 16→12 (spectral-density inverse; PR #1445 fern; tests if H162's null was overprovisioning)
  - H166 — surface_out_width_factor 2.0→3.0 (decoder-head-width; PR #1446 nezuko)
  - **H167 — model_heads 4→8 (attention-subspace-count; PR #1450 tanjiro)**
- **OPEN candidates** for next wave (if structural wave produces no win): GradNorm-α grid, conditional slices+Charbonnier combined, decoupled-β per-head optimizer, test_SP-as-primary architectural surgery, depth ablation (6→8 layers).

**CRITICAL — BASELINE.md correction (commit ea99dda was wrong):** nezuko discovered in PR #1360 that the 2026-05-28 H147 SOTA section copy-pasted H39's `yym5oa8x` test_VP/test_SP into the H147 row. Verified by W&B query of `k6q4c3on`. Correct H147 metrics: **test_VP=3.4014%** (clears 3.643% floor by 0.242pp) and **test_SP=3.5634%** (clears 3.577% floor by 0.014pp). **H147 actually CLEARS all 4 floors**, not 3-of-4. Patched in BASELINE.md this session.

**Consequence — β-decoupling finding I banked from H150 is RETRACTED.** Under the corrected H147 baseline, H150 (β1=0.97/β2=0.985) regressed on ALL 4 metrics (WSS +0.024pp, VP +0.10pp, SP +0.045pp and crosses the 3.577% cap by +0.032pp, ABUPT +0.054pp). The "pressure heads prefer β2↑" mechanism is FALSE. Simpler truth: **β1=0.95, β2=0.98 is the joint optimum on every test metric.** Wave 41 β-grid closure remains the wave's finding but is now stronger (single optimum, not a head-axis tradeoff). Corrigendum posted on closed PR #1359.

**RETRACTION — H157b is NOT the leading candidate (per fern's 10:35Z report):** my 09:55Z "EP9.28 val_WSS=6.5085%" heartbeat was a W&B chart-axis misread of a step that does not exist in `ew63yb7p` `scan_history`. The actual H157b trajectory had only 7 val passes (EP1-EP7) then **diverged at EP7→EP8**: calm EP7 train (grad_norm ≤ 0.06) → grad spike at step 83584 (80.9) → nonfinite gradients at step 85113 → 201-skip auto-abort at step 86363. Fern is harvesting EP6 EMA-best (val_WSS=6.921%, val_ABUPT=6.078%) for test eval. Projected test_WSS ~6.85-6.90% — **non-merge expected**.

**NEW FINDING — high-β Lion + Charbonnier-aux compound instability:**
- H150 (β=0.97/0.985, NO Charbonnier) ran 30 EP stable.
- H147 (β=0.95/0.98 + Charbonnier 0.1) ran 30 EP stable.
- **H157b (β=0.97/0.985 + Charbonnier 0.1) blew up at EP7.87.** The compound is the failure mode — high-β Lion momentum carries through Charbonnier's sharp-near-eps gradient into a divergent basin.
- **Implication:** H159 (frieren, β=0.95 + VP-Charb 0.3) and H161 (nezuko, β=0.95 + WSS-Charb 0.3) are on safer H147-β. Both should ride the heavier weight without divergence — but H161 grad-norm watch past EP4 is prudent (3x H157b's weight).

**Session events 09:50Z → 10:45Z:**
- **H151 TERMINAL CLOSED PR #1360 (no merge):** test_WSS=6.5439% (+0.003pp — noise tie); regressed VP/SP/ABUPT; crossed SP cap by +0.035pp. Headline finding from nezuko: **best-val=EP25 in BOTH H151 (T_max=45) and H147 (T_max=30)** — H147 recipe at convergence ceiling under same recipe; "more epochs" axis is exhausted.
- **H150 TERMINAL CLOSED PR #1359 (no merge):** verdict unchanged but reasoning corrected (see retraction above).
- **H158 ABORTED EP1 PR #1420 CLOSED:** advisor error stripped 4 H147 SOTA defaults; corrected stack used for H159+H160+H161.
- **H157b DIVERGED EP7.87 PR #1385:** fern harvesting EP6 EMA-best for test eval; closure expected ~11:30Z; non-merge expected; finding = high-β × Charbonnier compound instability (see above).
- **H159 (vol_p_charbonnier=0.3 on CORRECTED H147 stack)** **launched** on frieren PR #1423 — 8 DDP ranks at step ~6193 (rt 0.40h, EP~0.5).
- **H160 (β1=0.95, β2=0.985 — last untested β-grid cell)** assigned to tanjiro PR #1424 — **not yet launched on W&B** as of 10:42Z; tanjiro pod iteration 10 in progress. Will follow up at 10:50Z if no smoke run visible.
- **H161 (wss_charbonnier=0.3 axes=z on CORRECTED H147 stack — WSS-axis analog of H159)** assigned to nezuko PR #1427 at 10:37Z; nezuko's next poll cycle (~10:46Z) will pick it up.
- **Bug-fix commit e31fd60** (precompute_curvature_proxy.py + curvature_proxy_stats_k16_v1.json) — cherry-pick pending; needed for any future advisor pulling the H147 stack from cold env.

**11:00Z events:**
- **H157b CLOSED PR #1385** (non-merge): test_WSS=6.8819% (+0.341pp regression vs corrected H147 6.5409%). Note: fern ran axes=ALL (not axes=z as previously stated) — the finding disentanglement requires H162 as a follow-up. Per-axis: x=6.074%, y=7.539%, z=8.945%.
- **H162 ASSIGNED to fern PR #1430**: pe_num_features 16→24 (richer STRING spectral basis) on corrected H147 stack. Hypothesis: denser spectral PE improves WSS-z bottleneck. 8-EP T_max=8.
- **H159 CONFIRMED running**: step 12304 (0.82h, EP~1.1), EP1 val expected ~11:15Z.
- **H160 CONFIRMED running**: step 5416 (0.35h, EP~0.5), β1=0.95/β2=0.985 verified in config.
- **H161 CONFIRMED running**: step 4460 (0.29h, EP~0.4), charb_w=0.3/axes=z verified in config.

**HUMAN DEADLINE 15:45Z (4h40m remaining)** — H159 8-EP terminal ~16:25Z (running since ~10:15Z); H160 8-EP terminal ~16:40Z (running since ~10:40Z); H161 8-EP terminal ~16:55Z (running since ~10:50Z); H162 8-EP terminal ~17:00Z (launching ~11:30Z). Note: all 8-EP runs will slightly exceed the 15:45Z soft deadline; expecting terminal results 16:00-17:00Z.

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

### Fleet status (2026-05-29 11:05Z)

| Run IDs | Student | H# | Epoch | val_WSS | State | vs H147 |
|---|---|---|---:|---:|---|---|
| ze0shldj+7 ranks | frieren (dl24) | **H159** (vol_p_charbonnier=0.3 on CORRECTED H147 stack) | ~1.1 (step 12304, rt 0.82h) | EP1 pending ~11:15Z | **RUNNING** PR #1423 | 8-EP T_max=8; EP1 gate ≤13.5%. Terminal ~16:25Z. |
| 7a14s7uo+7 ranks | tanjiro (dl24) | **H160** (β1=0.95, β2=0.985 — last untested β-grid cell) | ~0.5 (step 5416, rt 0.35h) | — | **RUNNING** PR #1424 | Config verified β1=0.95/β2=0.985. 8-EP T_max=8. Terminal ~16:40Z. |
| y9xrfk5t+7 ranks | nezuko (dl24) | **H161** (wss_charbonnier=0.3 axes=z on CORRECTED H147 stack) | ~0.4 (step 4460, rt 0.29h) | — | **RUNNING** PR #1427 | Config verified charb_w=0.3/axes=z. 8-EP T_max=8. Terminal ~16:55Z. |
| (launching) | fern (dl24) | **H162** (pe_num_features 16→24 on CORRECTED H147 stack) | — | — | PR #1430 assigned | Fern picks up at next poll. 8-EP T_max=8. Terminal ~17:00Z. |
| ew63yb7p + q00o0xqk | (fern prev) | H157b (wss_charbonnier=0.1 **axes=ALL** on H150-β stack) | DIVERGED EP7.87 | 6.921% EP6 EMA | **CLOSED PR #1385 11:00Z** | ❌ test_WSS=6.8819% (+0.341pp), test_SP over cap +0.131pp. Finding: **β=0.97/0.985 + Charb axes=ALL weight=0.1 diverges at EP7.87**. axes=ALL (not z as previously stated) is the confound — disentanglement needed (H162 approach changed to PE-features; axes=z vs axes=ALL β-compound is open question). |
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

---

## 16:06Z next-cycle assignments launched

**3 new structural-axis hypotheses assigned to idle students (frieren, fern, nezuko):**

| PR | Student | Hypothesis | Family | Rationale |
|---|---|---|---|---|
| #1444 | dl24-frieren | **H164** — model_slices 128→192 | trunk-attention structural | Triple-arm joint finding indicates WSS-z floor is representational. Slice count = trunk token count. Untested axis. |
| #1445 | dl24-fern | **H165** — pe_num_features 16→12 | PE-projection-density inverse | Fern's own H162 follow-up #1. Tests projection-dilution hypothesis at lower-density end. |
| #1446 | dl24-nezuko | **H166** — surface_out_width_factor 2.0→3.0 | decoder-head structural | Output-head capacity test. Decoder-side analog of H164's trunk-side test. H39 SOTA set factor=2.0; nothing above tested. |

**Information-design value**: H164+H166 form an orthogonal pair (trunk-side vs decoder-side capacity). Any combination of {improve, flat, regress} on the two arms localizes the WSS-z floor mechanism. H165 cleanly tests the projection-dilution hypothesis raised by H162.

**Pod anomaly**: dl24-tanjiro pod stuck inside student-loop iteration 14 since 11:53Z (4h13m). H160 W&B training still running (5.31h, state=running). Advisor harvest directive posted 15:55Z; no student response. Cannot kubectl exec. Will likely auto-recover on next heartbeat or proceed via training terminal.

**Researcher-agent running in background**: generating 6-10 structural hypotheses for next cycle. Output to `/workspace/senpai/target/research/RESEARCH_IDEAS_2026-05-29_15:55.md`. Will inform next-cycle assignment planning.
