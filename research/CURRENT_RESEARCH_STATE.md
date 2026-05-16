# SENPAI Research State
- 2026-05-16 14:35 UTC (H8 CLOSED, H11 assigned to nezuko; H7 EP14 surf_p re-engaged; H9 EP6 strong; H10 EP4 on plan)

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

## Active Experiments (2026-05-16 13:45 UTC)

| Student | PR | Hypothesis | EP / Duration | val_abupt | val_wss | val_vol_p | val_surf_p | Notes |
|---------|-----|-----------|---------------|----------:|--------:|----------:|-----------:|-------|
| dl24-fern | #1142 | H7: surface_loss_weight=1.5 | **EP14 live** / 10h | 6.246% | 7.127% | **3.500%** | 4.051% | **Surf_p slope re-engaged EP13→14 (−0.009pp vs EP10→13 mean −0.004). vol_p UNDER FLOOR by 0.143pp. EP15 student post expected ~14:20Z. Decision: val_surf_p ≤3.95% bull / 3.95-4.05% central / >4.05% bear. Currently 4.051% = bear edge but slope re-engaged. Continue to EP30 (~21:30Z).** |
| **dl24-tanjiro** | **#1145** | **H9: curvature bias + w_vol_p clamp ≥0.05** | EP6 mid-EP7 / 6h | 6.307% | **7.041%** | **4.055%** | 4.102% | **STRONG: EP6 tied with H5 on WSS (7.04 vs 7.00), BETTER on vol_p (4.05 vs 4.18) BEFORE clamp activation. w_vol_p=0.147 declining toward 0.05 clamp (EP15-20 activation projected). Slope EP4→6 −0.058/ep — projects EP10 wss ≈ 6.81% (ahead of H5 EP10). Student posted EP3+6 combined at 13:33Z. EP10 next gate ~16:00Z (≤6.5/≤7.0/<4.0%).** |
| **dl24-nezuko** | **#1154** | **H11: AdamW lr=7e-4 + per-axis WSS τ-weights (τx=1.0, τy=1.2, τz=1.5)** | EP0 assigned 14:35Z | — | — | — | — | **NEW ASSIGNMENT. Compounds H8 wave finding (AdamW=safe+GradNorm stable) with H4-revisited (per-axis weights, Lion-noise root cause now neutralized). lr=7e-4 (40% above H8 to fix convergence rate), T_max=25 cosine. EP6 bellwether kill: val_τz > 9.80%.** |
| dl24-frieren | #1149 | H10: Charbonnier supplementary WSS loss | EP4 live / 5h | 6.516% | 7.280% | 4.209% | 4.231% | **ON TRACK: τz=9.76% at EP4 already at H5/H6 EP6 reference (2 epochs ahead). Charb/MSE weighted share stabilizing 44-46% (squarely in 20-60% healthy band). Slope τz −0.19/ep projects EP6=9.4% (clears ≤9.5% bellwether). val_wss projection EP10 ≈ 6.3% would map to test_wss ~5.9-6.0% (in #1056 target band). EP6 gate ~14:40Z.** |

**Step rate**: Both Lion AND AdamW run at ~4-5 steps/sec → 30-epoch run ≈ **33 hours**.

**Wave summary (13:45Z):**
- **H5 tanjiro** CLOSED (#1132) — test_wss=6.609% (BEST WSS IN WAVE, −0.118pp under SOTA) BUT both floors breached. GradNorm starvation root cause identified.
- **H6 frieren** CLOSED (#1135) — test_wss=6.770% tied, both floors breached. τy −0.057pp mechanism confirmed.
- **H7 fern** EP14 — val_surf_p 4.051% (slope re-engaged from EP10→13 plateau), val_vol_p 3.500% deep UNDER FLOOR. Bear-edge but slope re-acceleration could carry surf_p below floor at terminal. Continue to EP30.
- **H8 nezuko** EP10 **KILL AUTHORIZED** — τz=10.300% in 9.5+% kill range, hypothesis NOT validated. AdamW+GradNorm sanity is the wave finding. SIGTERM/eval/SENPAI-RESULT in progress. 8 GPUs freeing for H11.
- **H9 tanjiro** EP6 STRONG — val_wss 7.041 tied with H5, val_vol_p 4.055 BETTER than H5 BEFORE clamp activation. Slope projects EP10 wss ≈ 6.81%. Direct SOTA-chase trajectory ACTIVE.
- **H10 frieren** EP4 ON TRACK — τz=9.76% at EP4 already at H5/H6 EP6 reference. Charbonnier share 44-46% healthy. Floor-safe by construction. Bellwether EP6 gate ~14:40Z.

## Critical New Finding: GradNorm Vol_p Starvation Root Cause

**Discovered via H5 tanjiro terminal (PR #1132, 08:00Z)**

GradNorm's relative-rate balancing crushes `w_vol_p` because vol_p has the lowest task-loss slope (volume pressure is more linear / easier to fit than τ_z/τ_y). Final H5 weights: `w_vol_p=0.0064` vs `w_τ_z=2.318` (362× lower). This mechanism explains **every** vol_p floor breach in the wave (H1 +0.437pp, H2 +0.340pp, H3 +0.211pp, H5 +0.312pp) — all caused by the same GradNorm starvation regardless of injection point.

**Implication**: Any experiment that improves WSS (by increasing surface-task gradient signal) will also trigger GradNorm to crush `w_vol_p` further. The fix is a hard floor on `w_vol_p` (H9).

## Plateau Protocol Status (ACTIVE)

**4 consecutive closed failures + 2 plateauing experiments = Plateau Protocol triggered.**

| PR | Hypothesis | Status | Key failure mode |
|----|-----------|--------|-----------------|
| #1115 H1 | Wind-exposure input channels | ❌ CLOSED | gradnorm task-share imbalance (vol_p breach +0.437pp) |
| #1117 H2 | Curvature input channels | ❌ CLOSED | Same imbalance (vol_p breach +0.340pp); WSS improved −0.059pp (signal valid) |
| #1129 H3 | Near-wall volume cross-attn | ❌ CLOSED (06:25Z today) | Cross-attn duplicated existing backbone signal; τ_z WORSE +0.355pp |
| #1130 H4 | Per-axis WSS loss weights | ❌ CLOSED (02:33Z today) | Lion-noise on weighted axes; surf_p floor breach |
| #1132 H5 | Curvature additive attn bias | ❌ CLOSED (08:00Z) | test_wss=6.609% BEST in wave, but vol_p=3.955% breach. GradNorm starvation root cause identified. |
| #1135 H6 | Wind-exposure additive attn bias | ❌ CLOSED (09:30Z) | test: wss=6.770% tied, vol_p +0.314 breach, surf_p +0.092 breach. τy −0.057pp mechanism confirmed. 5-exp GradNorm starvation pattern confirmed. |

**Tier change response (H8):** Optimizer regime swap (Lion → AdamW) attacks update geometry rather than representation. All prior failures involved Lion's sign-based update amplifying per-axis noise.

## Key Mechanistic Findings (Wave Summary)

1. **Wrong injection point pattern**: Adding raw surface input channels (7→9 or 7→10) consistently triggers gradnorm task-share imbalance → vol_p floor breach. NEVER add raw input channels.
2. **Zero-init additive attention bias** is the safe injection pathway (H5, H6). No input-dim change → no gradnorm perturbation.
3. **Lion-noise contamination**: Lion `sign(momentum)` amplifies per-axis noise. Per-axis WSS loss weights (H4) made ALL τ axes worse. τ_z (lowest SNR, 8.747%) is most vulnerable.
4. **Surface-task upweight is a confirmed positive mechanism**: Both H3 (cross-attn expansion) and H4 (mean weight = 1.5×) produced vol_p under-floor improvement as a side-effect. Backbone learns richer features when surface task gets more gradient. H7 isolates this cleanly.
5. **vol_p side-effect is a 2-experiment pattern**: Any time surface gradient is boosted, vol_p improves. This has been replicated 2× and is >3× seed variance.

## Next Research Directions

### Current running (may win):
- **H9 tanjiro (#1145)**: curvature bias + w_vol_p clamp ≥0.05. **Strongest active candidate** — EP6 beats H5 on vol_p preemptively, ties on WSS, slope projects EP10 wss ahead of H5. Clamp activation EP15-20.
- **H7 fern (#1142)**: surface_loss_weight=1.5 — EP14 surf_p re-engaged after EP10-13 plateau. Bear-edge but trajectory still viable. EP15 decision metric ~14:20Z.
- **H10 frieren (#1149)**: Charbonnier WSS loss — EP4 2 epochs ahead of H5/H6 reference on τz. Charb share 44-46% healthy. Floor-safe by construction. EP6 bellwether ~14:40Z.

### Running + newly assigned (4 students active):
- **H11 nezuko (#1154)**: AdamW lr=7e-4 + per-axis WSS τ-magnitude weighting (τ_x=1.0, τ_y=1.2, τ_z=1.5). Compounds H8 wave finding (AdamW=safe optimizer) with H4-revisited (per-axis weighting, Lion-noise root cause now neutralized). lr bumped 40% + T_max=25 to address H8 convergence-rate deficit. **EP6 kill criterion: val_τ_z > 9.80%.**

### Closed this wave (H8 nezuko):
- **H8 nezuko (#1144 CLOSED)**: AdamW lr=5e-4 optimizer swap. REJECTED — test_wss +0.537pp regression, both floors breached. Convergence-rate deficit was the failure mode. WAVE FINDING: AdamW+GradNorm = stable (w_vol_p=0.298 vs Lion 0.0064 = 68× saner ratio). Prior "AdamW+GradNorm instability" warning falsified at lr=5e-4+cosine+1ep-warmup.

### Researcher-agent (background, agentId: a08af1e87b171e527):
- Running since ~06:26Z. Output: `/workspace/senpai/target/research/RESEARCH_IDEAS_2026-05-16_<time>.md`
- Prompt: fresh ideas attacking from optimizer, architecture, physics-informed loss, neural operator, curriculum learning, multi-resolution angles.

### Next-wave candidates (if H5-H8 all fail):
- **AdamW + surface_loss_weight=1.5 composition** (if H7 wins but H8 also wins independently)
- **Lower AdamW LR (1e-4 or 2e-4)** if H8 shows LR=5e-4 too aggressive
- **Fourier/Neural Operator architecture** (FNO, DeepONet) — completely different backbone
- **Multi-resolution training** — coarse→fine curriculum
- **Physical constraint in loss**: curl(τ)=0 or τ·n=0 enforcement (but careful of gradient explosion risk per Wave 27 lesson)

## Infrastructure Notes

- **Corrected split confirmed**: All runs on `rawcanon_20260511`. val→test gap is clean.
- **PR #1087 EMA fix**: shadow initialized from live weights at `ema-start-step`. All current runs clean.
- **Step rate**: ~4.4-5.6 steps/sec → 30-epoch run ≈ 21 hours.
- **W&B keys**: use `val_primary/<field>_rel_l2_pct` (NOT flat `val_*` which don't exist in this stack).
