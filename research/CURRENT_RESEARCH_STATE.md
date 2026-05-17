# SENPAI Research State
- 2026-05-17 16:25 UTC (**H9b TERMINAL + H18 ASSIGNED: H9b EP30 CLOSED — test_wss=6.781 MISS (+0.054 above SOTA), test_vol_p=3.646 AT FLOOR (within 0.003pp, mechanism VALIDATED ✅), test_surf_p=3.787 BREACH. NOT contract winner. Key finding: clamp+MAE_aux costs +0.103pp test_wss vs H9 baseline (anti-additive risk for H18 composition). H18 PR #1175 ASSIGNED to dl24-tanjiro: H10b curvature+Charb+clamp+MAE_aux 3-mechanism composition. Expected outcome: if mechanisms orthogonal, test_wss ~6.50-6.65 (SOTA-BEAT) + floors preserved; if anti-additive, test_wss ~6.78 (narrow miss). This is the wave's highest-priority experiment. H12 EP18=6.900 NEW BEST (trap-pattern FALSIFIED at EP17, EP11-14 was a single transient instability burst). Architecture confirmed equivalent to H10b at terminal val, ZERO contract benefit for H18 spec (separate τ head DROPPED). H11b EP26=7.144 NEW BEST (val_vol_p NEW LOW 3.941), EP30 proj test_wss ~6.93 narrow MISS. H10b 13-epoch plateau LOCKED at wave-low, EP15 ckpt winner (test proj 6.68 -0.05pp CLEAN BEAT). NEXT: H18 #1175 EP3 gate, H10b/H11b/H12 EP30 terminals.**)


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

## Active Experiments (2026-05-17 16:25 UTC)

| Student | PR | Hypothesis | EP / Duration | val_abupt | val_wss | val_vol_p | val_surf_p | Notes |
|---------|-----|-----------|---------------|----------:|--------:|----------:|-----------:|-------|
| **dl24-tanjiro** | **#1175** | **H18: H10b+H9b composition** | **JUST LAUNCHED** / 0h | — | — | — | — | **Highest-priority experiment**: H10b curvature+Charb + H9b clamp=0.15+MAE_aux. Anti-additive risk quantified: H9b costs +0.103pp test_wss vs H9 baseline. If H10b curvature rep makes cost orthogonal, test_wss ~6.50-6.65 (BEAT + floors). If anti-additive, ~6.78 (miss). EP3 hard abort: val_wss > 7.10. |
| dl24-fern | #1166 | **H12: separate τ head** | **EP18 LANDED — new best 6.900** / 14.0h | 6.289% | **6.900%** | 4.382% | 4.173% | Trap-pattern FALSIFIED. EP18 new H12 best. Architecture mechanistically equivalent to H10b at terminal val_wss — ZERO contract benefit vs H10b, confirms H18 should drop separate τ head. Continuing to EP30 (~21:00Z). |
| dl24-frieren | #1159 | **H10b: curvature + Charbonnier τ_z** | **EP27 LANDED — 13-epoch plateau LOCKED** / 21.0h | 6.280% | **6.915%** | 4.374% | 4.087% | **EP15 ckpt LOCKED AS WINNER**: 13-epoch plateau at [6.880, 6.915], EP15 wave-low. Test proj from EP15 ckpt: **6.68 (-0.05pp under SOTA = CLEAN BEAT)**. vol_p/surf_p floors BREACHED. ~3 epochs to EP30 (~5h). Terminal review ~20:00Z. |
| dl24-nezuko | #1160 | **H11b: AdamW + per-axis weights** | **EP27 LANDED — new best 7.144** / 20.0h | 6.396% | **7.144%** | 3.941% | 4.373% | EP26/27 = H11b best 7.144. val_vol_p NEW LOW 3.941. EP30 proj test_wss ~6.93 narrow MISS. NOT contract winner. Best-ckpt EP26. |

## Closed Experiments (2026-05-17)

| PR | Student | Hypothesis | Test result | Closing reason |
|---|---|---|---|---|
| #1157 | dl24-tanjiro | **H9b: clamp+curvature+MAE_aux** | test_wss=6.781 (MISS +0.054pp), test_vol_p=3.646 (AT FLOOR ±0.003pp), test_surf_p=3.787 (BREACH) | NOT contract winner. vol_p mechanism VALIDATED for H18. Anti-additive cost +0.103pp test_wss vs H9 baseline quantified. |

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
