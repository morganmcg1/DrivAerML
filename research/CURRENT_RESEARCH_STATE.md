# SENPAI Research State
- 2026-05-20 22:00 UTC — **⭐ H19 IS WAVE'S FIRST test_wss + test_abupt SOTA-BEAT (PR #1180 frieren `r5eigmer`). 4 PRs closed in this pass (H18 yesterday, H12/H19/H20 today). 4 students idle and awaiting next-wave assignments. Researcher-agent dispatched for fresh hypotheses; H21 = H19 + clamp=0.15 is the immediate priority follow-up. Posted milestone update to Issue #1056.**

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

## Active Experiments (2026-05-20 22:30 UTC — all 4 students dispatched)

| Student | PR | Hypothesis | Key change vs H19 | Priority |
|---------|-----|-----------|-------------------|----------|
| **dl24-frieren** | **#1216** | **H21: H19 + clamp=0.15** | `--gradnorm-min-w-vol-p 0.15` | **HIGHEST** — direct vol_p floor fix |
| **dl24-nezuko** | **#1217** | **H22: H19 + vol_p MAE_aux=0.05** | `--vol-p-aux-mae-weight 0.05` | HIGH — orthogonal floor-fix via L1 |
| **dl24-tanjiro** | **#1218** | **H23: H19 + Charb on τ_y** | `--wss-charbonnier-axes y,z` | HIGH — wss amplification |
| **dl24-fern** | **#1219** | **H24: H19 + clamp=0.10** | `--gradnorm-min-w-vol-p 0.10` | MEDIUM — ablation midpoint |

**Design logic:** H21-H24 form a systematic ablation of the two dimensions blocking H19 from being a contract winner:
- **Axis 1 — vol_p gradient mass**: H24 (clamp=0.10) < H22 (MAE_aux) < H21 (clamp=0.15)
- **Axis 2 — wss amplification**: H23 (Charb_τy in addition to τz)

H21 is the most direct path to a contract winner (H9b's clamp at 3× gradient mass closes the 0.136pp breach).

## Potential follow-on directions (post H21 evaluation)

1. **H22 = H19 + lighter clamp=0.10** — if H21 costs too much wss, softer clamp may keep more budget on wss while still pulling vol_p down partway
2. **H23 = H19 + Charb on τ_y AND τ_z** — extends the wss-axis mechanism (student's H18 follow-up suggestion); may compound wss benefit
3. **H24 = H19 + vol_p decoder LR multiplier** — orthogonal: decouple encoder gradient budget from decoder fit (student's #2 suggestion)
4. **H25 = pure additive Charb-on-vol_p (no GradNorm task-signal swap)** — diagnostic to test whether H19's win came from "Charb under GradNorm" (load-bearing) or "additive Charb on vol_p" (diagnostic-only)
5. **LR cosine extension** — tay fleet noticed `--lr-cosine-t-max 25` instead of 13 may stretch the productive learning window; worth testing on top of the H21 stack if H21 lands close
6. **PCGrad / CAGrad gradient surgery** — explicit multi-task gradient conflict resolution as alternative to GradNorm
7. **Physics-informed regularization** — incompressibility constraint on vol_p predictions, pressure-Poisson residual penalty
8. **Distillation from an ensemble teacher** — Issue #1056 forbids ensembles AS PRIMARY, but distilling ensemble outputs into a single-model student is allowed and could close the floor gap
9. **Volume decoder capacity expansion** — H18's "curvature has global vol_p penalty" finding suggests representational bottleneck; testing 2-layer MLP on vol_p head could help

## Key uncertainties to resolve next

- **Is the vol_p breach in H19 a gradient-mass problem or a landscape problem?** H21 (H19 + clamp=0.15) directly tests this. If clamp alone closes the gap, gradient-mass; if not, landscape/representational.
- **Does the asymmetric GradNorm budget (H19's w_τ_z=1.85) survive a clamp constraint?** Adding clamp=0.15 forces w_vol_p ≥ 0.15, which means total budget for τ_z drops. The wss benefit of H19 might be partially lost.
- **What's the minimum wss cost to clear floors?** If H21 costs +0.14pp wss (like H20 vs H10b), final test_wss ~6.77 — above SOTA 6.727. We'd need ~+0.05pp cost to stay sub-SOTA, which depends on whether Charb landscape reshape softens the clamp's cost.

## References for next wave

- Issue #1056 — the active research directive
- BASELINE.md — the locked contract metrics (PR #972)
- H19 PR #1180 (closed) — wave-milestone result with detailed mechanism analysis
- H18 PR #1175 (closed) — three-way comparison data and EP8 spike suppression finding
- Researcher-agent output (running, ETA ~15min) — `/workspace/senpai/target/research/RESEARCH_IDEAS_2026-05-20_22:00.md`
