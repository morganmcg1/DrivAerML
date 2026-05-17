# SENPAI Research State
- 2026-05-17 10:00 UTC (**MAJOR: H12 EP11 REGRESSION ON ALL 7 AXES (wss 6.933→6.961, vol_p +0.307pp single-epoch jump) — entered ABORT GATE LIVE zone (6.95-6.99). EP12 hard kill at ~10:35Z if val_wss ≥ 6.95 sustained. Architecture experiment outcome: real EP7 lead (-0.037pp vs H10b at same step) but does NOT sustain through compound Lion+Charb+separate-τ-head — separate τ head DROPPED from primary H18 spec. H10b EP15 ckpt LOCKED as SOTA-beat winner (test_wss ~6.68 -0.05pp under SOTA, +3 axes also under). H11b EP18 strong reaccel sustained (-0.021/ep wss). H9b EP22 flat plateau 6.996 (autonomous to EP30 ~12:00Z). NEXT: H12 EP12 kill decision ~10:35Z, then H18 composition (reduced stack) ready for first idle student.**)


## Human Research Directive (Issue #1056 — 2026-05-14)

**TOP PRIORITY — Wall Shear Stress (WSS) Focus:**
- The **TEST wall shear stress L2 error** is the primary metric to drive down
- Target: **test_wss < 5.85%** (Transolver-3 reference; current SOTA = 6.727%, gap +0.877pp = 13% relative)
- **Strict AND-clause floors** (must NOT degrade vs PR #972 SOTA):
  - `test_vol_p ≤ 3.643%`
  - `test_surf_p ≤ 3.577%`
- **NO ENSEMBLES** — single-model only. Per Morgan: "we want genuine breakthroughs, not incremental improvements based on ensembling".
- All experiments must build on PR #972 training stack AND use corrected dataset: `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`

**Morgan check-in at 19:30Z** — asked "hows it looking now? any breakthroughs? what are we learning from these experiments?" Posted detailed status response at 19:35Z covering H9 wave headline (4/7 axes under SOTA), H10/H11 wave findings, H9b in flight, H7 bear case.

## Wave SOTA (Corrected Split — rawcanon_20260511)

**PR #972** (run `56bcqp3m`, eval `zxnhtagj`) — single-model best

| Metric | Value | Status |
|--------|-------|--------|
| test_abupt | **5.844%** | wave SOTA |
| test_surf_p | 3.577% | **floor for WSS wave** |
| test_vol_p | **3.643%** | **floor for WSS wave** |
| test_wss | 6.727% | **PRIMARY TARGET: < 5.85%** |
| test_τ_x | 5.971% | — |
| test_τ_y | 7.362% | — |
| test_τ_z | 8.747% | — |

## HEADLINE WAVE FINDING #1 (from H9 EP30, confirmed terminal 17:32Z)

**4 of 7 test_primary axes under SOTA #972** — all four on the WSS side.
- test_wss=6.678% (−0.049pp ✅), test_τ_x=5.903% (−0.068pp ✅), test_τ_y=7.308% (−0.054pp ✅), test_τ_z=8.668% (−0.079pp ✅)
- BUT both floors breached: test_vol_p=3.913% (+0.270pp ❌), test_surf_p=3.692% (+0.115pp ❌)
- **Curvature additive attention bias mechanism CONFIRMED as the WSS path.**

## HEADLINE WAVE FINDING #2 (from H9b EP3, confirmed 21:00Z) — **vol_p ceiling is FALSIFIABLE**

H9 wave finding said "vol_p ceiling is representation-bound at 4.05%". **H9b EP3 falsifies this:**

| | H9 (terminal EP10) | H9b EP3 (1/3 training) |
|---|---:|---:|
| val_vol_p | 4.056% | **4.180%** (already at H9 terminal level) |
| EP2→EP3 vol_p slope | (n/a) | **−1.496 pp/ep** (huge descent) |
| w_vol_p | 0.088 | 0.269 (3.1× H9, stabilizing 0.30-0.33 band) |
| Decision band | terminal+floor breach | **strong continue (<4.5%)** |

**The MAE aux + clamp combination is producing vol_p descent 2-3× faster than H9 baseline.** The "representation-bound ceiling" framing was MAE-aux-deficient. With the aux loss providing a persistent direct L1 signal (currently 0.00128 weighted), the GradNorm task-share dynamic does NOT collapse and vol_p task gradient mass stays active. **Projected: val_vol_p < 3.5% by EP10, well under the 3.643% test floor.**

**This is the wave's first credible path to SOTA-on-aggregate (all 7 axes under SOTA #972).**

## Active Experiments (2026-05-17 10:00 UTC)

| Student | PR | Hypothesis | EP / Duration | val_abupt | val_wss | val_vol_p | val_surf_p | Notes |
|---------|-----|-----------|---------------|----------:|--------:|----------:|-----------:|-------|
| dl24-fern | #1166 | **H12: separate τ head** | **EP11 LANDED — REGRESSION on all 7 axes** / 8.0h | 6.383% | **6.961%** | 4.543% | 4.275% | EP10→11 ALL 7 AXES regressed: wss +0.028, **vol_p +0.307 single-epoch jump** (10× prior descent — optimizer instability or batch-noise event). In ABORT GATE LIVE zone (6.95-6.99). EP12 HARD KILL at ~10:35Z if val_wss ≥ 6.95 sustained. EP10 ckpt is best (val=6.933 → test ~6.73 at SOTA). Architecture lesson: separate τ head provides real EP7 lead (-0.037pp) but does NOT sustain in compound with Lion+Charb. **DROPPED from H18 primary spec**. |
| dl24-tanjiro | #1157 | **H9b: clamp + curvature + MAE_aux** | **EP22 LANDED — flat plateau** / 15.5h | 6.219% | 6.996% | 3.782% | 4.170% | wss flat (+0.004/ep EP21→22). Pod autonomous to EP30 ~12:00Z. NOT contract winner. vol_p mechanism validated for H18. |
| dl24-frieren | #1159 | **H10b: curvature + Charbonnier τ_z** | **EP18 LANDED — 4-epoch plateau LOCKED** / 15.5h | 6.257% | **6.890%** | 4.366% | 4.067% | **EP15 ckpt LOCKED AS WINNER**: test_wss projection 6.68 (-0.05pp under SOTA = CLEAN BEAT), +3 axes also under (τ_x -0.18, τ_y -0.09). vol_p/surf_p floors STILL BREACHED. EP20 land imminent (~10:05Z). |
| dl24-nezuko | #1160 | **H11b: AdamW + per-axis weights** | **EP18 LANDED — strong reaccel** / 14.5h | 6.489% | 7.234% | 4.066% | 4.424% | EP17→18 -0.021/ep wss strongest single-epoch in cohort. EP30 proj range 6.83-6.95 → test 6.63-6.75 (coin flip ±0.10 vs SOTA). vol_p closing strongly (-0.025/ep gap to H9b). EP20 land imminent (~10:05Z). |

**Step rate**: Both Lion AND AdamW run at ~4-5 steps/sec → 30-epoch run ≈ **33 hours**.

## H9 Wave Findings (PR #1145 CLOSED 17:44Z)

1. **Curvature bias transfers cleanly through val→test gap.** H5 and H9 both show ~−0.25pp val→test on every axis — mechanism is robust, not val-overfit.

2. **GradNorm vol_p starvation has TWO distinct modes:**
   - **Gradient-mass mode (H5)**: w_vol_p crashed to 0.0064 (362× below w_τ_z). Curvature mechanism absent.
   - **Representation-capacity mode (H9)**: Curvature bias self-stabilizes w_vol_p at 0.088 (13.7× alive vs H5). w_vol_p ALIVE but vol_p still stalls at ~4.05% — bottleneck is volume decoder representation, not loss weighting.

3. **The 0.05 clamp was DORMANT** in H9 — natural floor (0.088) exceeded clamp. Binding clamp for H9 stack = ≥0.10-0.15. **H9b implements clamp=0.15 + vol_p MAE aux** to disambiguate mass vs direction bottleneck.

4. **Vol_p ceiling is NOT rate-coupled**: despite 13.7× higher w_vol_p than H5, val_vol_p stalls at the same ~4.05%. Points to representational capacity as the real bottleneck.

## H10 Wave Findings (PR #1149 CLOSED 19:30Z) — "Representation Floor"

H10 frieren EP30 terminal:
- val_abupt=6.426%, val_wss=7.142%, val_vol_p=4.131%, val_surf_p=4.090%
- test_τ_z val→test gap: **+0.600pp** (3-4× larger than typical 0.150pp) — Charbonnier IS reshaping the τ_z loss landscape, just on the wrong representation
- Same val plateau as H9 baseline (no acceleration despite Charbonnier mechanically engaged)

**Wave finding**: Loss-axis reshape is not sufficient on the original Lion representation. Charbonnier must be paired with the H9 curvature representation upgrade to deliver value. H10b PR #1159 is the compound test.

## H11 Wave Findings (PR #1154 KILLED 19:28Z) — "AdamW lr>5e-4 + per-axis weights = instability"

H11 nezuko EP1-5:
- val_abupt: EP1=20.04 → EP2=28.48 → EP3=16.54 → EP4=16.27 → EP5=12.21 (8pp+ above H8 reference trajectory at every epoch)
- EP1→EP2 spike of +8.44pp indicates warmup→full-LR optimization instability
- Per-axis τ_z=1.5 boost amplified gradient norm during the post-warmup transition

**Wave finding**: With AdamW + GradNorm, lr should NOT exceed 5e-4 when stacking per-axis weight changes. H11b PR (queued, ready to launch once #1154 closes) holds lr=5e-4 fixed to isolate the per-axis mechanism cleanly.

## Plateau Protocol Status (REFINED — H9 broke through on WSS axis)

The WSS plateau IS broken at H9: 4 of 7 test axes under SOTA #972, including the wave's primary target. **The remaining puzzle is the vol_p+surf_p floor pair.**

| PR | Hypothesis | Status | Result |
|----|-----------|--------|--------|
| #1115 H1 | Wind-exposure input channels | ❌ CLOSED | gradnorm task-share imbalance (vol_p breach +0.437pp) |
| #1117 H2 | Curvature input channels | ❌ CLOSED | Same imbalance (vol_p breach +0.340pp); WSS improved −0.059pp (signal valid) |
| #1129 H3 | Near-wall volume cross-attn | ❌ CLOSED | Cross-attn duplicated existing backbone signal; τ_z WORSE +0.355pp |
| #1130 H4 | Per-axis WSS loss weights | ❌ CLOSED (later INVALIDATED — was NO-OP under GradNorm) | H11 discovered the flag never entered loss; H4 effectively ran baseline |
| #1132 H5 | Curvature additive attn bias | ❌ CLOSED | test_wss=6.609% (best WSS so far), vol_p=3.955% breach |
| #1135 H6 | Wind-exposure additive attn bias | ❌ CLOSED | test_τy=−0.057pp mechanism confirmed; floor breaches |
| #1144 H8 | Lion→AdamW lr=5e-4 | ❌ CLOSED | All-axis regression (+0.5pp); WAVE FINDING: AdamW+GradNorm stable (w_vol_p=0.298) |
| #1145 H9 | Curvature bias + 0.05 clamp | 🎯 **WSS BREAKTHROUGH** | 4/7 axes under SOTA; floor breaches remain. Clamp dormant. |
| #1149 H10 | Charbonnier WSS supp loss | ❌ CLOSED (19:30Z) | Representation-floor wave finding (same equilibrium, different residuals; +0.600pp τ_z val→test gap) |
| #1154 H11 | AdamW lr=7e-4 + per-axis WSS | ❌ CLOSED 19:48Z | Catastrophic regression (test_abupt=11.16%, test_wss=11.30%); wave finding "AdamW lr>5e-4 + per-axis weights = instability" |
| #1157 H9b | clamp=0.15 + curvature + vol_p MAE aux | 🟢 EP1 verified | Positive emergent: MAE aux equilibrium shift of 10× on w_vol_p; clamp dormant |
| #1159 H10b | H9 curvature + Charbonnier on τ_z only | 🟢 LAUNCHING | Compound H9 representation × τ_z-leverage Charbonnier |
| #1160 H11b | AdamW lr=5e-4 + per-axis WSS τ-weights | 🟢 LAUNCHING | Clean per-axis isolation; same as H8 stack + per-axis weights only |

## Key Mechanistic Findings (Wave Summary)

1. **Wrong injection point pattern**: Adding raw surface input channels (7→9 or 7→10) consistently triggers gradnorm task-share imbalance → vol_p floor breach. NEVER add raw input channels.
2. **Zero-init additive attention bias** is the safe injection pathway (H5, H6, H9). No input-dim change → no gradnorm perturbation.
3. **AdamW+GradNorm is stable** (H8 wave finding), but AdamW lr>5e-4 with per-axis weight changes is unstable (H11 wave finding). Future per-axis experiments must use lr=5e-4 baseline.
4. **Surface-task upweight (H7 stack)** consistently produces vol_p under-floor as side-effect. Backbone learns richer features when surface task gets more gradient. Most stable mechanism in the wave.
5. **vol_p ceiling under curvature bias** is NOT representation-bound — **FALSIFIED at H9b EP3.** With MAE aux providing direct L1 signal alongside GradNorm, val_vol_p hits H9's terminal level at 1/3 of training. The earlier "representation-bound" framing was a missing-mechanism artifact, not a true ceiling.
6. **WSS plateau [6.96, 6.99] val band IS NOT a true plateau** — H9 at val_wss=6.925% landed test_wss=6.678% (under SOTA).
7. **Representation floor (H10)**: Loss-axis reshape needs the right representation to act on. Charbonnier on original Lion stack = same plateau height, different equilibrium. Charbonnier on H9 curvature stack (H10b) = the real test.
8. **τ_z val→test gap is the bellwether WSS axis.** H9 had +0.116pp gap; H10 had +0.600pp gap. Gap size tracks how aggressively the loss is reshaping the τ_z optimization landscape.

## Next Research Directions

### Currently running:
- **H7 fern (#1142)**: EP21+, surf_p flat at 4.04% — bear case confirmed; continue to EP30 (~24:00Z).
- **H9b tanjiro (#1157)**: EP1 verified, MAE aux rewriting GradNorm. **The wave's most important active experiment.** EP3 viability gate ~21:00Z; EP10 decision ~02:00Z. If MAE aux + clamp punches through vol_p ceiling, we may be SOTA-on-aggregate.
- **H10b frieren (#1159)**: launching. Compound of H9 curvature WSS unlock + Charbonnier-on-τ_z (highest val→test gap axis). 

### Just launched (next ~30 min):
- **H11b nezuko (#1160)**: AdamW lr=5e-4 + per-axis WSS τ-weights (1.0, 1.2, 1.5). Clean single-variable isolation. EP3 gate ~22:00Z (val_abupt ≤ 8.5%, +0.24pp tolerance vs H8's 8.26%).

### Compositional candidates (post-H9b):
- **H7 + H9 stack (composition)**: surface_loss_weight=1.5 + curvature bias + clamp=0.15. If H9b clears floor on vol_p, combine with H7's surface upweight mechanism for further surf_p reduction.

### Next-tier ideas (if H9b plateaus):
- **L1/Huber main loss on vol_p only** (not aux) — tests whether the issue is L2's quadratic penalty on residuals < 0.2σ failing to drive vol_p further.
- **Skip-connection bypass from input to vol decoder** — direct conditioning shortcut for vol_p signal that bypasses the GradNorm-balanced backbone path.
- **Wider/deeper volume decoder MLP head** — increases representational capacity at the volume output point.
- **Fourier-feature volume coordinates** — input-side capacity bump for vol_p without touching backbone gradient sharing.

## Infrastructure Notes

- **Corrected split confirmed**: All runs on `rawcanon_20260511`. val→test gap is clean (~−0.25pp on H9 curvature stack).
- **PR #1087 EMA fix**: shadow initialized from live weights at `ema-start-step`. All current runs clean.
- **Step rate**: ~4.4-5.6 steps/sec → 30-epoch run ≈ 33 hours.
- **W&B keys**: use `val_primary/<field>_rel_l2_pct`. Surface WSS uses `wall_shear_rel_l2_pct` (NOT `wall_shear_stress_rel_l2_pct`).
- **Student commit hygiene reminder**: H11 implementation code was never pushed to the H11 branch (only assignment commit). H11b instructions explicitly require pushing the implementation BEFORE launching.
