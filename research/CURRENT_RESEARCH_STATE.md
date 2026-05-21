# SENPAI Research State
- 2026-05-21 16:35 UTC — **EP15 CLUSTER — H26 vol_p drift now +0.010pp ⭐⭐⭐ — almost matched to H21.** W&B summary @ 16:34Z: **H26 tanjiro (`apgpxli8`) EP15.35** val_wss=**6.8111** (continuing slope, −0.015pp from EP13.6), val_vol_p=**3.8470** (drift narrowed from +0.022pp at EP13.6 to **+0.010pp at EP15** vs H21 EP15 ~3.837), val_surf_p=**4.003** (slightly worse than H21 EP15 ~3.85 but continuing to fall at ~0.012/EP), val_abupt=**6.090**, w_τ_z=1.722 (vs H21 terminal 1.99), r_vol_p=**5.00** (clamp engaged ✓). **H26 leads all 4 val metrics — wave-leader unchanged.** **H25 nezuko (`xjbz1v84`) EP15.85**: val_wss=7.091 (mechanism falsified +0.014pp vs H21 EP15), val_vol_p=3.855, val_surf_p=4.167, val_abupt=6.295, **w_τ_z=1.57 (LOWER than H21 terminal 1.99 despite Charb-τz weight 2× boosted)** — confirms falsification: static loss boost makes GradNorm see "louder" axis as needing LESS push (same shape as H24-v2). Continuing for vol_p signal. **H27 frieren (`yyo3q1xb`) EP15.13**: val_wss=6.898, val_vol_p=3.884, val_surf_p=4.097, val_abupt=6.185, r_vol_p=3.98 (clamp_active=0; below 5.0 threshold; lighter 0.10 clamp running freer). **H28 fern (`83iayezy`) EP7.99**: val_wss=7.025 (Plateau Protocol on H19; +0.10 vs H21 EP10 = trailing wave but vol_p still rolling down 4.45 from 4.81@EP6.55). EP10 gate ~18:00Z. **Next milestones**: H28 EP10 ~18:00Z; H25/H26/H27 EP20 cluster ~19:30Z; H25/H26/H27 EP30 terminals ~02:00-04:00Z 2026-05-22; H28 EP30 ~08:00Z 2026-05-22.
- 2026-05-21 15:05 UTC — **H26 vol_p DRIFT NARROWING ⭐⭐ — full contract winner now plausible.** EP13.6 summary: H26 (`apgpxli8`) val_wss=**6.8266** (was 6.890 at EP10, −0.063pp), val_vol_p=**3.8584** (was 3.946, **−0.088pp** drift narrowing back toward H21 EP10=3.837 reference; only +0.022pp gap now vs +0.109pp at EP10), val_surf_p=**4.012** (was 4.055, −0.043pp), val_abupt=**6.107** (was 6.107, flat). H26 leads ALL 4 metrics. Projection at EP30 if H21-shape val drift continues (-0.008pp/EP vol_p): EP30 val_vol_p ~3.728 → test ~3.633 (UNDER FLOOR 3.643 −0.010pp ⭐). H26 projected test_wss ~6.46 (BEATS SOTA -0.27pp ⭐), test_surf_p ~3.56 (UNDER FLOOR -0.017pp ⭐), test_abupt ~5.73 (BEATS SOTA -0.11pp ⭐). **H26 could clear ALL 4 contract conditions — first wave arm to potentially satisfy the AND-clause.** Wave at 15:05Z: H25 EP14.1 val_wss=7.097 (still trailing, vol_p mechanism healthy 3.862), H27 EP13.4 val_wss=6.914, H28 EP6.2 val_wss=6.994 (Plateau Protocol arm; already at H19 EP10-quality at EP6 — accelerated convergence). H28 EP3 gate PASSED at 13:34Z (val_wss=7.286 vs target 7.39).
- 2026-05-21 13:08 UTC — **EP10 CLUSTER LANDED — H26 IS WAVE LEADER ALL 4 VAL METRICS** ⭐⭐⭐ H26 tanjiro (`apgpxli8`, surface-loss-weight 1.5) EP10: val_wss=**6.890** (H21 EP10=7.090, −0.200pp), val_vol_p=3.946 (+0.109pp tradeoff vs H21), val_surf_p=4.055 (−0.092pp vs H21), val_abupt=6.107 (−0.191pp), val_wssz=9.448 (−0.18pp). **H26 EP10 val_wss=6.890 already beats H21 TERMINAL (7.077) by −0.187pp** — surface re-weighting is the wss lever. Projected test_wss (using H21 val→test gap −0.347pp): **~6.543 — NEW WSS SOTA candidate −0.184pp vs PR #972 6.727** ⭐. **H27 frieren (`yyo3q1xb`, clamp=0.10) EP10**: val_wss=6.955, val_vol_p=3.968, val_surf_p=4.131 (predicted target <4.05 missed; lighter clamp dividend smaller than expected — w_τ_x=0.911 (+0.16 vs H21), w_cp=0.602 (−0.04 vs H21) → wss gain bigger than surf_p gain). **H25 nezuko (EP10.78)**: val_wss=7.132 (falsified +0.055pp), val_vol_p=3.911 (vol_p mechanism intact, clamp_active=1, on track for floor); ADVISOR replied 12:30Z to continue to EP30 for vol_p signal. **H28 fern (EP3.0)**: val_wss=7.59 EP2 (vs H19 EP2=7.804, −0.21pp early); EP3 gate post pending (next iter ~13:10Z). Watch H26 EP20 (~19:30Z) & EP30 (~02:20Z 2026-05-22) and vol_p drift — current vol_p+0.109pp trajectory clears floor only if late-epoch clamp engages.
- 2026-05-21 10:30 UTC — **H24-v2 CLOSED + Fern reassigned H28** (Plateau Protocol on H19): test_wss=6.81 / test_vol_p=3.71 / test_surf_p=3.73 / test_abupt=5.94 — regresses ALL 4 metrics vs SOTA. test_τ_z=**8.88** (worse than H19's 8.747, OPPOSITE of intent) — per-axis τ weights backfire under GradNorm (static pre-weight makes GradNorm see "louder" axis as needing LESS boost; CANCELS or INVERTS the intended push). clamp=0.15 vol_p mechanism survives (test_vol_p 3.71 still beats H19's 3.779). Per Morgan's Issue #1056 Plateau Protocol directive, **H28 = H19 reference + extended cosine LR tail** (lr-decay-iters set to 2× train horizon to keep meaningful LR through EP6-30; current cosine bottoms out at EP6 leaving EP7-30 in near-frozen learning zone). **Wave update**: H26 (tanjiro) is the strongest in-flight candidate — EP6 val_wss=**6.981** (beats H19 EP6=7.146 AND H21 EP6=7.259); projected test_wss ~6.5-6.6 if val→test gap holds. Watch H26 EP10 closely.
- 2026-05-21 05:30 UTC — **H21 CLOSED + Frieren reassigned H27** (PR #1239): H21 terminal canonical posted by frieren (vol_p=3.579 SUB-FLOOR first in wave, but surf_p=3.679 breaches; not contract winner). **Critical mechanism diagnostic**: terminal GradNorm weights show clamp's vol_p budget came from **τ_x (w=0.75) and cp (w=0.64)**, NOT τ_z (w=1.99, +0.14 vs H19). Charb-on-vol_p+curvature-attention reshapes the loss-ratio landscape; GradNorm's revealed-preference for off-loading is cp+τ_x, not τ_z. **H27 = H21 + clamp 0.15→0.10** — lighter clamp predicted to preserve most vol_p win (val_vol_p flat EP15→EP30) while restoring surf_p floor + wss SOTA-tie. Wave matrix now: H21 closed, H24-v2 fern in flight EP24, H25 nezuko Charb-τz 2×, H26 tanjiro surface_loss_weight 1.5, H27 frieren clamp=0.10.
- 2026-05-21 05:07 UTC — **H21 TERMINAL — clamp mechanism validated; near-miss contract** (xcj9749y): test_abupt=**5.832** (BEATS SOTA 5.844 by −0.012pp ⭐), test_vol_p=**3.579** (CLEARS floor 3.643 by −0.064pp ⭐⭐ FIRST IN WAVE), test_wss=6.730 (tied with SOTA 6.727 +0.003pp), test_surf_p=**3.679** (BREACH floor 3.577 by +0.102pp ❌). H21 is the **first wave run to clear vol_p floor** — clamp=0.15 mechanism validated for vol_p. But Issue #1056 AND-clause requires BOTH vol_p AND surf_p under floor; surf_p still breaches. test_τ_z=8.630 = H19's value, confirms clamp doesn't degrade wss. Awaiting frieren SENPAI-RESULT (last PR comment EP5 10:53Z; student loop slow but pod alive on iter 285).
- 2026-05-21 05:00 UTC — **H23 CLOSED + Tanjiro reassigned H26** (PR #1238): Charb-τ_y on H19 base falsified (all 4 primary metrics regress vs SOTA AND vs H19). Saturation insight retained: Charb-under-GradNorm saturates at 1 surface wss axis; adding τ_y starves w_cp + w_τ_x. **H26 = H21 + surface_loss_weight 1.0→1.5** — surface re-weighting (tanjiro's own suggested non-Charb path). Parallel arm to nezuko's H25 (Charb-τz weight 0.1→0.2); orthogonal mechanisms (gradient mass vs loss-shape).
- 2026-05-21 04:50 UTC — **H23 TERMINAL (zq1czmdu) — NOT a contract winner**: test_wss=6.774 (+0.047pp regression vs SOTA), test_vol_p=3.909 (+0.266pp breach), test_surf_p=3.689 (+0.112pp breach), test_abupt=5.933 (+0.089pp regression). Charb-τy mechanism alone regresses ALL metrics vs SOTA — confirmed dead end. Waiting for tanjiro SENPAI-RESULT post. **H21 in terminal eval** (EP30 step 329279, 6+ min stalled — eval in flight; val_vol_p=3.674 wave-best); test_vol_p projected ~3.37 if H19 val→test gap holds.
- 2026-05-21 04:45 UTC — **H22 CLOSED** (PR #1217): test_wss=6.681 (−0.046pp, mild improvement) BUT vol_p=3.800 (+0.157pp breach) AND surf_p=3.736 (+0.159pp breach). MAE_aux+Charb falsified — near-identical L1 signals, don't compose. **Nezuko assigned H25** (PR #1237): H19+clamp + boosted Charb-τz weight 0.1→0.2 — single-parameter push on worst WSS axis (test_wss_z=8.65% in H22, 5pp gap vs AB-UPT bench 3.63%).
- 2026-05-21 04:38 UTC — **H22 TERMINAL (rlgxm0r3) — NOT a contract winner**: test_wss=**6.681** (−0.046pp vs PR #972 6.727, mild wss improvement ⭐) BUT test_vol_p=**3.800** (+0.157pp breach vs 3.643 floor ❌) AND test_surf_p=**3.736** (+0.159pp breach vs 3.577 floor ❌); test_abupt=5.872 (+0.028pp regression vs 5.844). Per Issue #1056 strict AND-floor clause, H22 cannot merge — vol_p+surf_p both regress. MAE_aux mechanism alone is insufficient. Awaiting student EP30 canonical report. H23 in terminal eval (step 329280, 5+ min stalled — eval in flight). H21 at EP29.88, H24-v2 EP22.72.
- 2026-05-20 19:54 UTC — **H24-v2 EP10 LEADS WAVE ON BOTH AXES** ⭐⭐: `j2pvm44m` posted EP10 readout — val_wss=**6.982** (wave-best by −0.130pp vs H23) AND val_vol_p=**3.920** (wave-best by −0.113pp vs H21). First wave run leading both wss AND vol_p simultaneously. Clamp engagement 35.2% of steps; w_τ_y=1.594 (highest, per-axis weights composing with H23 evidence). EP30 ETA ~09:15Z 2026-05-21.
- 2026-05-20 15:55 UTC — **EP12 wave snapshot (W&B)**: H21 `xcj9749y` val_wss=7.060 val_vol_p=**3.784** (within +0.14pp of 3.643 floor — wave's contract-winner trajectory) ⭐; H23 `zq1czmdu` val_wss=**7.048** (wave-best wss) val_vol_p=4.275; H22 `rlgxm0r3` val_wss=7.076 val_vol_p=4.067 (already posted canonical EP10 at 14:20Z); H24-v2 `j2pvm44m` EP5.2 val_wss=7.053 val_vol_p=4.150 (early but tracking strong). H21 + H23 student loops have not posted canonical EP10 reports despite being past EP12 — per project memory, not reassigning on stale_wip; runs healthy.
- 2026-05-20 12:18 UTC — **H24-v2 RELAUNCHED CLEAN**: fern launched run `j2pvm44m` at 12:14Z on PR #1226 with corrected CLI; config verified (65000/65536 train/eval points, all H24 mechanism flags set). Old misconfig `5dp7s3nz` crashed. Issue #1227 (infra escalation) closed — fern loop revived itself; polling cadence just longer than peers. All 4 students productive.
- 2026-05-20 11:55 UTC — EP6 wave readout: H22 ⭐ and H23 ⭐ confirmed beating H19 (val_wss −0.006pp / −0.040pp), H21 vol_p mechanism dominant (−0.382pp val_vol_p vs H19) at +0.107pp wss cost. PR #1220 (H24-OLD misconfigured) CLOSED; H24-v2 reassigned as PR #1226.
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
| **dl24-fern** | **#1226** (was #1220) | H24-v2: H19 + clamp=0.15 + per-axis τ "1.0,1.2,1.5" | n/a — EP3 ETA ~14:25Z | n/a | Run `j2pvm44m` LAUNCHED 12:14Z with corrected CLI ✓; old `5dp7s3nz` crashed. Config verified: all 4 point overrides correct, mechanism flags match design |

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
