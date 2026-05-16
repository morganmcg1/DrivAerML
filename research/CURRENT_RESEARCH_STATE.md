# SENPAI Research State
- 2026-05-16 15:50 UTC (H7 EP15 surf_p borderline-bear, H9 EP10 imminent strong trajectory, H10 EP6 marginal slope-flattened, H11 EP1 launched + **H4 no-op finding**)

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
| dl24-fern | #1142 | H7: surface_loss_weight=1.5 | **EP15 live** / 11h | 6.231% | 7.107% | **3.488%** | **4.047%** | **EP15 surf_p at top edge of central band [3.95, 4.05]%, slope decelerating −0.004pp/ep (mirrors EP5→6 pattern). vol_p UNDER FLOOR by 0.155pp. WSS projection 6.50-6.65% test = beats SOTA. **Most likely terminal: surf_p breach 0.10-0.25pp**; bull case needs late-cosine LR re-engagement EP18-22. Continue to EP30. EP20 check ~17:50Z.** |
| **dl24-tanjiro** | **#1145** | **H9: curvature bias + w_vol_p clamp ≥0.05** | **EP9.5 live W&B** / 7h | **6.232%** | **6.938%** | **4.051%** | 4.075% | **STRONG (live W&B step 105,914, EP10 imminent): val_wss=6.938 (down −0.103 from EP6), val_τz=9.403 (deceleration into target band), vol_p FLAT 4.051→4.055 EP6→9.5 (clamp positioning effect). w_vol_p=0.092 approaching 0.05 clamp (1-3 EP to engagement). Student EP10 post expected 16:05-16:15Z.** |
| **dl24-nezuko** | **#1154** | **H11: AdamW lr=7e-4 + per-axis WSS τ-weights (τx=1.0, τy=1.2, τz=1.5)** | **EP1 mid live** / 1h | — | — | — | — | **LAUNCHED 14:55Z W&B `kukjenp5`. EP1 partial: w_vol_p=0.451 (well above 0.15 floor, even above H8 terminal 0.298). GradNorm balanced. CRITICAL FINDING from student: H4 PR #1130's wss_axis_loss_weights was a NO-OP under GradNorm — never entered loss. H4 effectively ran baseline τ-weighting. H11 is the FIRST actual test of per-axis weighting under GradNorm. EP3 gate ~16:55Z.** |
| dl24-frieren | #1149 | H10: Charbonnier supplementary WSS loss | EP6 live / 7h | 6.423% | 7.192% | 4.090% | 4.204% | **MARGINAL: slope flattened sharply at EP4 (−0.168 → −0.033 → −0.055/ep), EP6 wss=7.192 (behind H9 EP6 7.041 by 0.15pp), τz=9.662 in continue-with-watch band [9.5, 9.7]. Charb share creeping 44→48%. EP10 decision logic locked: ≤6.85=STRONG, 6.85-6.99=OK, 6.99-7.05=marginal, ≥7.05=KILL. EP10 expected ~17:30Z.** |

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
3. **Lion-noise contamination (DOWNGRADED 15:50Z)**: Originally attributed H4 failure to Lion `sign(momentum)` amplifying per-axis noise. **H11 student discovered H4's `wss_axis_loss_weights` flag was a NO-OP under GradNorm** — never entered loss when gradnorm_weights set. H4 effectively ran baseline τ-weighting. The "Lion-noise on per-axis weights" finding from H4 is invalidated. H11 is the first actual test of the per-axis WSS mechanism under GradNorm. Lion-noise on τ_z (lowest SNR) remains plausible but UNCONFIRMED.
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
