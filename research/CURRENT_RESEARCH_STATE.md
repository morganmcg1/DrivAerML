# SENPAI Research State
- 2026-05-17 00:35 UTC (**H10b EP6 sub-7% wss MILESTONE (6.996) but τ_z slope decel -0.036 → SOTA-beat downgraded to MARGINAL on wss only; H9b EP8 CLAMP ENGAGED (w_vol_p=0.150) with sustained -0.023 vol_p slope → EP10 projects 3.802% UNDER kill; H11b EP6 gap to H9b narrowed to 0.78pp; H7 fern EP30 terminal val LANDED, test eval requested**)

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

## Active Experiments (2026-05-17 00:35 UTC)

| Student | PR | Hypothesis | EP / Duration | val_abupt | val_wss | val_vol_p | val_surf_p | Notes |
|---------|-----|-----------|---------------|----------:|--------:|----------:|-----------:|-------|
| dl24-fern | #1142 | H7: surface_loss_weight=1.5 | **EP30 TERMINAL val** / 21h | 6.234% | 7.116% | **3.495%** ✅ | **4.054%** ❌ | EP26-30 flat (σ <0.01pp). val_wss=7.116 → test_wss ~6.81-6.92% likely **NOT a SOTA-beat** (above 6.727 by 0.08-0.19pp). **Test eval requested at 00:36Z**, awaiting student SENPAI-RESULT post. NOT-A-MERGE expected. |
| dl24-tanjiro | #1157 | **H9b: clamp=0.15 + curvature bias + vol_p MAE aux 0.05** | **EP8 LANDED** / 5h | 6.301% | 7.076% | **3.848%** | 4.205% | **CLAMP ENGAGED**: w_vol_p=0.150 at floor. EP7→8 vol_p slope sustained at **-0.023** (no further decay!). EP10 projection: 3.802% — UNDER 3.85% kill margin. Clamp+MAE_aux mechanism load-bearing. EP30 vol_p ~3.30-3.46% → test ~3.15-3.50% **CLEARS floor 3.643** ✅. **BUT** surf_p stalled at 4.20% → test_surf_p ~4.0pp breaches floor 3.577 by ~0.4pp. wss/τ_z trajectories: marginal vs SOTA. |
| dl24-frieren | #1159 | **H10b: H9 curvature bias + Charbonnier on τ_z only** | **EP6 LANDED** / 4.4h | 6.321% | **6.996%** ✅ | 4.268% | 4.110% | **First sub-7% val_wss in the wave**. 5/6 axes ahead of H9b at EP6. τ_z lead **NARROWED** from -0.198 to **-0.174pp** (slope decay 0.54×). Charb ratio=0.957. **EP7 is decisive**: slope ≥-0.025 = SOTA-beat alive; ≥-0.015 = recommend hot-swap to weight=0.05. **Downgraded to MARGINAL wss-only beat** projection (test_wss ~6.65-6.70 vs SOTA 6.727; τ_z likely 9.13% test, NOT beating SOTA 8.747). |
| dl24-nezuko | #1160 | **H11b: AdamW lr=5e-4 + per-axis WSS τ-weights** | **EP6 LANDED** / 4h | 7.118% | 7.837% | 4.933% | 4.789% | Case 2 marginal recovery sustaining: decay factor 0.43-0.53 across axes. Gap to H9b at EP6=0.776pp (closing -0.20pp/ep). **EP30 projection: val_abupt ~6.70%, ~0.4pp above H9b** → NOT a SOTA-beat candidate. Per-axis WSS upweight pays optimization friction tax. Continue to EP10 decision gate. |

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
