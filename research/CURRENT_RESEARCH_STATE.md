# SENPAI Research State
- 2026-05-20 11:55 UTC — **EP6 WAVE READOUT: H22 ⭐ and H23 ⭐ confirmed beating H19 (val_wss −0.006pp / −0.040pp), H21 vol_p mechanism dominant (−0.382pp val_vol_p vs H19) at +0.107pp wss cost. PR #1220 (H24-OLD misconfigured) CLOSED; H24-v2 reassigned as PR #1226 — fern will pick up after killing run `5dp7s3nz`.** All 4 students remain occupied through wave completion.
- 2026-05-20 09:55 UTC — EP3 wave update: H22/H23 both beat H19 EP3 val_wss; H21 soft-flagged but clamp engaged. H24-OLD detected misconfigured (4× more views/epoch).
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

## Active Experiments (2026-05-20 11:55 UTC — EP6 readout; EP10 ETA ~14:30Z)

| Student | PR | Hypothesis | EP6 val_wss vs H19 EP6 (7.152) | EP6 val_vol_p vs H19 EP6 (4.415) | Status |
|---------|-----|-----------|------------------------------:|---------------------------------:|--------|
| **dl24-frieren** | **#1216** | H21: H19 + clamp=0.15 | 7.259 (+0.107pp) | **4.033 (−0.382pp) ⭐⭐** | Clamp dominant, vol_p crushing H19, wss cost large but expected |
| **dl24-nezuko** | **#1217** | H22: H19 + vol_p MAE_aux=0.05 | **7.146 (−0.006pp) ⭐** | **4.214 (−0.201pp) ⭐** | BOTH wss AND vol_p beat H19 — best dual mechanism |
| **dl24-tanjiro** | **#1218** | H23: H19 + Charb on τ_y | **7.112 (−0.040pp) ⭐** | 4.490 (+0.075pp) | wss-direct via τ_y, vol_p slight regression |
| **dl24-fern** | **#1226** (was #1220) | H24-v2: H19 + clamp=0.15 + per-axis τ "1.0,1.2,1.5" | n/a — relaunch pending | n/a | PR #1220 CLOSED 11:30Z (misconfig + stalled relaunch); fresh #1226 with corrected CLI; old run `5dp7s3nz` to be killed by fern student loop |

**EP6 read-out (2026-05-20 11:55Z) — wave now well past EP3 with clearer trajectories:**

1. **H21 (clamp=0.15) is the vol_p MECHANISM WINNER.** At EP6 val_vol_p=4.033% vs H19=4.415% (−0.382pp = 8.7% relative improvement). The clamp is producing predicted asymmetric tradeoff: w_vol_p pinned at 0.15 (vs collapse to 0.057 in H19), τ_z weight =1.74 (lower than H19's 1.85), wss cost +0.107pp. Vol_p val→test gap is ~+0.42pp on H19 → H21 projects test_vol_p ≈ 3.6%, AT OR UNDER the 3.643% floor. If wss cost stays at +0.1pp, terminal test_wss ≈ 6.73% (≈ tied with SOTA 6.727, NOT improving on H19's wss). **Sole-mechanism contract winner trajectory.**

2. **H22 (MAE_aux=0.05) is the DUAL-METRIC WINNER.** EP6 val_wss=7.146 (−0.006pp vs H19, essentially break-even) AND val_vol_p=4.214 (−0.201pp). w_vol_p still collapsed to ~0.057 floor (PR-predicted "w stays high" hypothesis WRONG) — the L1 aux gradient is **additive outside the GradNorm budget**. This is the wave's first "free improvement" — beats H19 on BOTH primary metrics simultaneously.

3. **H23 (Charb on τ_y) is the WSS-DIRECT WINNER.** EP6 val_wss=7.112 (−0.040pp), driven by val_τ_y=7.748 (−0.038pp vs H19 7.786). Compared to H10b's curvature+Charb_τz mechanism, H23 extends per-axis Charbonnier to a second highest-error axis. Vol_p regression +0.075pp (mild). If H23's mechanism stacks cleanly with H21's clamp, H25-H28 (Charb on multiple axes) become valuable.

4. **H24-v2 (#1226) relaunched cleanly.** PR #1220 closed 11:30Z (fern's student loop did not act on the 10:05Z relaunch comment after 1.5h; old run `5dp7s3nz` was still active at step 126k/1303k, 43h projected). Fresh PR #1226 with corrected CLI (4 `--*-surface/volume-points` flags added) + new wandb-group `H24-clamp015-peraxis-v2`. Fern's student loop should detect closed PR → mark idle → pick up #1226 on next poll. Compound mechanism still high-VP slot.

**PR #1219 history:** Original H24 design was clamp=0.10 (ablation midpoint). Morgan merged the assignment commit at 06:56Z (#1219), making fern briefly idle. Researcher-agent then created the compound H24 (#1220 at 07:06Z); fern's compound run was misconfigured. PR #1220 closed 2026-05-20T11:30Z and replaced by #1226 (compound H24-v2, same hypothesis, corrected CLI).

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
