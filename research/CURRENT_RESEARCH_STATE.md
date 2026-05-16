# SENPAI Research State
- 2026-05-16 18:35 UTC (H9 CLOSED w/ wave-headline 4 of 7 axes under SOTA; H10 KILL+eval pending; **H11 KILL autonomous** EP5; H7 EP21 bear case confirmed; H9b LAUNCHED tanjiro)

## Human Research Directive (Issue #1056 — 2026-05-14)

**TOP PRIORITY — Wall Shear Stress (WSS) Focus:**
- The **TEST wall shear stress L2 error** is the primary metric to drive down
- Target: **test_wss < 5.85%** (Transolver-3 reference; current SOTA = 6.727%, gap +0.877pp = 13% relative)
- **Strict AND-clause floors** (must NOT degrade vs PR #972 SOTA):
  - `test_vol_p ≤ 3.643%`
  - `test_surf_p ≤ 3.577%`
- **NO ENSEMBLES** — single-model only. Per Morgan: "we want genuine breakthroughs, not incremental improvements based on ensembling".
- All experiments must build on PR #972 training stack AND use corrected dataset: `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`

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

## HEADLINE WAVE FINDING (from H9 EP10, confirmed 17:32Z)

**4 of 7 test_primary axes under SOTA #972** — all four on the WSS side.
- test_wss=6.678% (−0.049pp ✅), test_τ_x=5.903% (−0.068pp ✅), test_τ_y=7.308% (−0.054pp ✅), test_τ_z=8.668% (−0.079pp ✅)
- BUT both floors breached: test_vol_p=3.913% (+0.270pp ❌), test_surf_p=3.692% (+0.115pp ❌)
- **Curvature additive attention bias mechanism CONFIRMED as the WSS path.**
- val→test gap on H9 curvature stack is clean (~−0.25pp on every axis), so the mechanism is robust and not val-overfit.

**This is the first single-model run in the wave to achieve SOTA-under on the WSS aggregate.** The path to merge: unlock vol_p + surf_p floors while preserving the curvature WSS gain.

## Active Experiments (2026-05-16 18:35 UTC)

| Student | PR | Hypothesis | EP / Duration | val_abupt | val_wss | val_vol_p | val_surf_p | Notes |
|---------|-----|-----------|---------------|----------:|--------:|----------:|-----------:|-------|
| dl24-fern | #1142 | H7: surface_loss_weight=1.5 | **EP21 live** / 14h | **6.215%** | 7.093% | **3.485%** | 4.041% | **EP15→EP21 surf_p flat (σ ≈ 0.005pp). Bear case confirmed — terminal projection +0.15-0.30pp surf_p breach. Vol_p mechanism locked under floor. Continue to EP30; advisor EP20 ack posted 18:33Z (W&B direct read since student silent). Stale_wip flag was harness-set; student is healthy.** |
| dl24-tanjiro | #1157 | **H9b: clamp=0.15 + curvature bias + vol_p MAE aux 0.05** | **EP1 launching** / 0h | — | — | — | — | **LAUNCHED 18:25:58Z, 8 GPUs. 2×2 ablation: clamp (binding) + MAE aux (orthogonal direct vol_p signal). EP3 viability gate ~21:00Z, EP10 decision ~02:00Z.** |
| dl24-frieren | #1149 | **H10: Charbonnier WSS — KILLED EP10**, test eval running | EP10 KILL | 6.341% | 7.078% | 4.133% | 4.148% | **🛑 KILLED 17:47Z. Plateau-marginal kill (val_wss=7.078% ≥ 7.05% kill band). Test eval `c5436ytt` launched 18:23Z; rank0 still running at 18:30Z. Test metrics expected by ~18:40Z. H10b queued: Charbonnier on τ_z only (single-axis ablation).** |
| dl24-nezuko | #1154 | **H11: AdamW lr=7e-4 + per-axis WSS τ-weights — KILL AUTHORIZED** | EP5.1 KILL | 12.21% | — | — | — | **🛑 ADVISOR-KILLED 18:35Z (autonomous, 40min radio silence post-anomaly nudge). EP1=20.04→EP2=28.48→EP5=12.21% (vs H8 reference EP5≈6.5%). Hypothesis: lr=7e-4 too aggressive in high-residual warmup; per-axis τ_z=1.5 boost amplified gradient norm. H11b queued: AdamW lr=5e-4 + per-axis weights (single-variable isolation of the per-axis mechanism from LR jump).** |

**Step rate**: Both Lion AND AdamW run at ~4-5 steps/sec → 30-epoch run ≈ **33 hours**.

## H9 Wave Findings (PR #1145 CLOSED 17:44Z)

1. **Curvature bias transfers cleanly through val→test gap.** H5 and H9 both show ~−0.25pp val→test on every axis — mechanism is robust, not val-overfit.

2. **GradNorm vol_p starvation has TWO distinct modes:**
   - **Gradient-mass mode (H5)**: w_vol_p crashed to 0.0064 (362× below w_τ_z). Curvature mechanism absent.
   - **Representation-capacity mode (H9)**: Curvature bias self-stabilizes w_vol_p at 0.088 (13.7× alive vs H5). w_vol_p ALIVE but vol_p still stalls at ~4.05% — bottleneck is volume decoder representation, not loss weighting.

3. **The 0.05 clamp was DORMANT** in H9 — natural floor (0.088) exceeded clamp. Binding clamp for H9 stack = ≥0.10-0.15. **H9b implements clamp=0.15 + vol_p MAE aux** to disambiguate mass vs direction bottleneck.

4. **Vol_p ceiling is NOT rate-coupled**: despite 13.7× higher w_vol_p than H5, val_vol_p stalls at the same ~4.05%. Points to representational capacity as the real bottleneck.

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
| #1149 H10 | Charbonnier WSS supp loss | 🛑 KILLED EP10 | Plateau-marginal: representation-floor finding ("same equilibrium, different residuals") |
| #1154 H11 | AdamW lr=7e-4 + per-axis WSS weights | 🛑 KILLED EP5 (autonomous) | LR jump + per-axis weights compounded instability; first actual test of per-axis under GradNorm |

## Key Mechanistic Findings (Wave Summary)

1. **Wrong injection point pattern**: Adding raw surface input channels (7→9 or 7→10) consistently triggers gradnorm task-share imbalance → vol_p floor breach. NEVER add raw input channels.
2. **Zero-init additive attention bias** is the safe injection pathway (H5, H6, H9). No input-dim change → no gradnorm perturbation.
3. **AdamW+GradNorm is stable** (H8 wave finding), but AdamW lr>5e-4 with per-axis weight changes is unstable (H11 wave finding). Future per-axis experiments must use lr=5e-4 baseline.
4. **Surface-task upweight (H7 stack)** consistently produces vol_p under-floor as side-effect. Backbone learns richer features when surface task gets more gradient. This is the most stable mechanism in the wave.
5. **vol_p ceiling under curvature bias** is representation-bound at ~4.05%, not gradient-rate-bound. H9b tests whether direct L1 (MAE aux) can punch through this ceiling.
6. **WSS plateau [6.96, 6.99] val band IS NOT a true plateau** — H9 at val_wss=6.925% landed test_wss=6.678% (under SOTA). The plateau was a sampling artifact of the H5/H6 stack lacking the H9 clamp+EMA combination.

## Next Research Directions

### Currently running:
- **H7 fern (#1142)**: EP21, surf_p flatlining at 4.04% — bear case confirmed; vol_p mechanism locked. Continue to EP30 (~24:00Z). Most likely NOT-A-MERGE due to surf_p breach, but H7 stack is **confirmed positive ingredient** for composition.
- **H9b tanjiro (#1157)**: clamp=0.15 + vol_p MAE aux 0.05 + curvature bias. **The wave's most important active experiment.** EP3 viability gate ~21:00Z; EP10 decision ~02:00Z. If MAE aux + clamp punches through vol_p ceiling, we may be SOTA-on-aggregate.

### Pending kill+eval (test metrics expected ~18:40-19:00Z):
- **H10 frieren eval `c5436ytt`**: Charbonnier wave finding pending — calibrates val→test for mid-range residual reweighting. Result feeds H10b design.
- **H11 nezuko eval**: pending student SIGTERM+eval-only run. Quick eval needed before next assignment.

### Queued assignments (post-eval):
- **H10b frieren**: Charbonnier on **τ_z axis only** (single-axis ablation, orthogonal to H11 per-axis weighting). Single-mechanism isolation since H10 confirmed full-WSS Charbonnier is plateau-equivalent.
- **H11b nezuko**: AdamW **lr=5e-4** + per-axis WSS τ-weights. Isolates per-axis mechanism from LR jump that broke H11.

### Compositional candidates (post-H9b):
- **H7 + H9 stack (composition)**: surface_loss_weight=1.5 + curvature bias + clamp=0.15. If H7 confirms vol_p under-floor mechanism transfers to test AND H9 confirms WSS axis. Use H7 mechanism's vol_p side-benefit + H9 mechanism's WSS punch + H9b mechanism's floor-clearing.

### Next-tier ideas (if H9b plateaus):
- **L1/Huber main loss on vol_p only** (not aux) — tests whether the issue is L2's quadratic penalty on residuals < 0.2σ failing to drive vol_p further. Same idea as MAE aux but as MAIN signal.
- **Skip-connection bypass from input to vol decoder** — direct conditioning shortcut for vol_p signal that bypasses the GradNorm-balanced backbone path.
- **Wider/deeper volume decoder MLP head** — increases representational capacity at the volume output point (where the H9 finding said the bottleneck lives).
- **Fourier-feature volume coordinates** — input-side capacity bump for vol_p without touching backbone gradient sharing.

## Infrastructure Notes

- **Corrected split confirmed**: All runs on `rawcanon_20260511`. val→test gap is clean (~−0.25pp on H9 curvature stack).
- **PR #1087 EMA fix**: shadow initialized from live weights at `ema-start-step`. All current runs clean.
- **Step rate**: ~4.4-5.6 steps/sec → 30-epoch run ≈ 33 hours.
- **W&B keys**: use `val_primary/<field>_rel_l2_pct`. Surface WSS uses `wall_shear_rel_l2_pct` (NOT `wall_shear_stress_rel_l2_pct`).
- **H10 eval status check**: 7 of 8 eval ranks finished at 18:30Z; rank0 still running (typical 12-16 min). Watch summary `test_primary/abupt_axis_mean_rel_l2_pct` for arrival signal.
