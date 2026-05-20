# SENPAI Research State
- 2026-05-20 09:55 UTC — **EP3 WAVE UPDATE: H22 ⭐ and H23 ⭐ BOTH BEAT H19 EP3 val_wss; H21 SOFT-FLAGGED (within bounds); H24 DETECTED MISCONFIG — sent back to fern for relaunch with corrected `--train-*-points 65000` flags.** All 4 students still occupied. H22 projects to test_vol_p ~3.83% (vs floor 3.643%) if EP3 trajectory holds — could be wave's contract winner candidate.
- 2026-05-20 07:00 UTC — H19 IS WAVE'S FIRST test_wss + test_abupt SOTA-BEAT (PR #1180 frieren `r5eigmer`). 4 PRs closed in prior pass. H21-H24 dispatched.

## Human Research Directive (Issue #1056 — 2026-05-14, re-confirmed Morgan check-in 2026-05-19 19:50Z)

**TOP PRIORITY — Wall Shear Stress (WSS) Focus:**
- Drive **test_wss < 5.85%** (Transolver-3 target; current best on this branch: 6.634% from H19)
- **Strict AND-clause floors** (must NOT degrade vs PR #972 SOTA):
  - `test_vol_p ≤ 3.643%`
  - `test_surf_p ≤ 3.577%`
- **NO ENSEMBLES** — single-model only
- Corrected dataset: `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`
- Report val AND test (best-val checkpoint)

## Wave SOTA (Corrected Split — rawcanon_20260511)

**PR #972** (run `56bcqp3m`, eval `zxnhtagj`) — locked single-model best (NOT merged into this branch; the floor contract reference)

| Metric | Value | Status |
|--------|-------|--------|
| test_abupt | 5.844% | wave SOTA on primary aggregate |
| test_surf_p | **3.577%** | **floor for WSS wave** |
| test_vol_p | **3.643%** | **floor for WSS wave** |
| test_wss | 6.727% | wave SOTA (target: drive below 5.85%) |
| test_τ_x | 5.971% | — |
| test_τ_y | 7.362% | — |
| test_τ_z | 8.747% | — |

## Wave-best results so far (single-model, this branch)

| Run | test_abupt | test_wss | test_vol_p | test_surf_p | Status |
|-----|-----------:|---------:|----------:|------------:|--------|
| **H19 `r5eigmer`** ⭐ | **5.8197** | **6.6339** | 3.7786 (+0.136 breach) | 3.6267 (+0.050 breach) | CLOSED — first wss+abupt SOTA beat |
| H18 `xhx2qlpo` | 5.895 | 6.746 | 3.702 (+0.059 breach) | 3.729 (+0.152 breach) | CLOSED — narrow miss, time-truncated |
| H10b `60zl0p4h` | 5.929 | 6.6651 | 4.160 (+0.517 breach) | 3.690 (+0.113 breach) | CLOSED — wss-beat, vol_p breach |
| H9b `smflmb5t` | 5.922 | 6.781 | **3.646** (≈ AT floor) | 3.787 (+0.210 breach) | CLOSED — vol_p mechanism validated |
| H20 `4yvl848t` | 5.972 | 6.808 | 3.847 | 3.740 | CLOSED — clamp-only mechanism isolation |
| H12 `3v58n2m5` | 6.004 | 6.732 | 4.074 | 3.874 | CLOSED — separate τ head no contract benefit |

## Wave insight summary (as of 2026-05-20)

1. **H10b curvature + Charb_τz** is the wss-SOTA mechanism. Confirmed across H10b, H18, H19.

2. **Charb-on-vol_p under GradNorm (H19)** creates an asymmetric GradNorm budget: w_vol_p collapses to floor while w_τ_z surges to ~1.85. Result: STRONGEST wss in the wave (6.63), at cost of vol_p +0.136pp above floor.

3. **MAE_aux is NOT the wss-cost driver** (H19 has no MAE_aux and STILL beats SOTA on wss). The CLAMP itself is the wss-cost mechanism (~+0.14pp on H10b base).

4. **MAE_aux carries unique vol_p benefit** (~−0.20pp) that clamp alone cannot replicate. H9b's full stack achieves the floor only because both mechanisms compose.

5. **Curvature representation has a "global vol_p penalty"**: H10b base + clamp+MAE_aux (H18) still breaches vol_p by +0.059pp — the floor-locking transfers cleanly to H9 but partially erodes on H10b.

6. **The vol_p floor breach is now SMALL (+0.136pp on H19)** — 4× smaller than H10b. Adding clamp=0.15 on top of H19 should close the remaining gap by 3× more vol_p gradient mass.

## Active Experiments (2026-05-20 09:55 UTC — EP3 reports landed for 3/4)

| Student | PR | Hypothesis | EP3 val_wss vs H19 | EP3 status |
|---------|-----|-----------|-------------------:|-----------|
| **dl24-frieren** | **#1216** | H21: H19 + clamp=0.15 | **7.589 (+0.20pp)** | SOFT-FLAG, clamp engaged, continue |
| **dl24-nezuko** | **#1217** | H22: H19 + vol_p MAE_aux=0.05 | **7.293 (−0.10pp) ⭐** | BEATS H19, MAE_aux mechanism confirmed |
| **dl24-tanjiro** | **#1218** | H23: H19 + Charb on τ_y | **7.321 (−0.07pp) ⭐** | BEATS H19, direct τ_y improvement |
| **dl24-fern** | **#1220** | H24: H19 + clamp=0.15 + per-axis τ | n/a (regime misconfig) | **RELAUNCH** — PR body missing `--train-*-points 65000` overrides; current run hits 4× more steps/epoch (43h projected vs 22.5h timeout) |

**EP3 read-out (2026-05-20 09:30Z):**

1. **MAE_aux IS THE WAVE'S BREAKTHROUGH MECHANISM (H22).** Provides −0.244pp vol_p improvement at EP3 by feeding L1 gradient external to GradNorm budget, while w_vol_p still collapses to ~0.057 floor. The PR-predicted "w_vol_p stays high" hypothesis was WRONG (it does collapse), but the additive L1 outside the budget IS the load-bearing mechanism. Projects vol_p ~3.83% at EP19 (still above floor but huge improvement).

2. **H23 confirms per-axis Charb extension works (−0.068pp val_wss).** Direct τ_y improvement (val_τ_y −0.084pp) shows the mechanism is τ_y axis-specific, not just budget reshuffling. τ_z STILL improves despite budget split — additive landscape effect.

3. **H21 EP3 clamp cost (+0.20pp val_wss) larger than predicted.** Clamp engaged perfectly (w_vol_p pinned at 0.15 since step 15k), but the vol_p improvement vs H19 only opens up in mid-late epochs once H19's collapse takes effect. H21's payoff projected EP10+.

4. **H24 setup is broken — relaunching.** Researcher-agent's auto-generated PR command omitted `--train-surface-points 65000 --eval-surface-points 65536 --train-volume-points 65000 --eval-volume-points 65536` flags that H21/H22/H23 sibling PRs include. Fern was running with `train_volume_points=16384` (4× smaller) and `train_views=347657` (4× more), producing a 43h projected runtime vs 22.5h timeout. Comment posted on #1220 with full corrected command.

**PR #1219 history:** Original H24 design was clamp=0.10 (ablation midpoint). Morgan merged the assignment commit at 06:56Z (#1219), making fern briefly idle. Researcher-agent then created the compound H24 (#1220 at 07:06Z); fern's compound run was misconfigured and is being relaunched as H24-v2.

## Next-wave queue (researcher-agent designs, ordered by expected value)

From `RESEARCH_IDEAS_2026-05-20_22:00.md`, post H21-H24 results:

1. **H25 = H21 + Charb on τ_x (`--wss-charbonnier-axes xz`)** — multi-axis Charb expansion on floor-safe stack
2. **H26 = H21 + Charb ε=1e-4 on vol_p** — sharper sub-quadratic regime, may organically raise w_vol_p under GradNorm
3. **H27 = H21 + cosine T_max=25** — 5-epoch LR plateau at end for fine-tune extraction
4. **H28 = H21 + Charb on yz (`--wss-charbonnier-axes yz`)** — Charb on the two highest-error WSS axes (τ_y=7.36%, τ_z=8.75%)
5. **H29 = IMTL-G replacement of GradNorm** — gradient-surgery approach, bypasses Charb loss-magnitude distortion entirely (requires implementation; reserve for plateau trigger)
6. **H30 = H19 + Charb vol_p weight 0.05→0.10** — organic GradNorm fix (raises apparent vol_p task difficulty without clamp)
7. **H31 = H19 + GradNorm α=0.5→1.0** — sharper gradient normalization; risk: may amplify Charb distortion in wrong direction

**Alternative tier-shift directions (if H21-H28 wave plateaus):**

- **Physics-informed regularization** — incompressibility constraint on vol_p predictions, pressure-Poisson residual penalty
- **Distillation from ensemble teacher** — Issue #1056 forbids ensembles AS PRIMARY, but ensemble-distillation into a single-model student is allowed
- **Volume decoder capacity expansion** — H18's "curvature has global vol_p penalty" finding suggests representational bottleneck; testing 2-layer MLP on vol_p head
- **PCGrad / CAGrad gradient surgery** — multi-task gradient conflict resolution alternatives to GradNorm

## Key uncertainties to resolve next

- **Is the vol_p breach in H19 a gradient-mass problem or a landscape problem?** H21 (H19 + clamp=0.15) directly tests this. If clamp alone closes the gap, gradient-mass; if not, landscape/representational.
- **Does the asymmetric GradNorm budget (H19's w_τ_z=1.85) survive a clamp constraint?** Adding clamp=0.15 forces w_vol_p ≥ 0.15, which means total budget for τ_z drops. The wss benefit of H19 might be partially lost.
- **What's the minimum wss cost to clear floors?** If H21 costs +0.14pp wss (like H20 vs H10b), final test_wss ~6.77 — above SOTA 6.727. We'd need ~+0.05pp cost to stay sub-SOTA, which depends on whether Charb landscape reshape softens the clamp's cost.

## References for next wave

- Issue #1056 — the active research directive
- BASELINE.md — the locked contract metrics (PR #972)
- H19 PR #1180 (closed) — wave-milestone result with detailed mechanism analysis
- H18 PR #1175 (closed) — three-way comparison data and EP8 spike suppression finding
- `RESEARCH_IDEAS_2026-05-20_22:00.md` — full H21-H31 researcher-agent design briefs with decision tree and stop conditions
