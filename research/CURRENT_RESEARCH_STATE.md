# SENPAI Research State
- 2026-05-16 13:25 UTC (H7 EP10 PASS, H9 EP6 strong, H8 EP10 marginal, H10 EP2 launched)

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

## Active Experiments (2026-05-16 13:25 UTC)

| Student | PR | Hypothesis | EP / Duration | val_abupt | val_wss | val_vol_p | val_surf_p | Notes |
|---------|-----|-----------|---------------|----------:|--------:|----------:|-----------:|-------|
| dl24-fern | #1142 | H7: surface_loss_weight=1.5 | **EP10 PASS** / 7h | **6.275%** | 7.153% | **3.524%** | 4.073% | **EP10 gate PASS: surf_p ≤4.10% threshold cleared by 0.027pp. vol_p UNDER FLOOR by 0.119pp. Cosine LR re-engaged surf_p slope post EP6. Continue to EP30 (~21:30Z). Terminal call surf_p center 3.55-3.65% — borderline.** |
| **dl24-tanjiro** | **#1145** | **H9: curvature bias + w_vol_p clamp ≥0.05** | EP6 live / 5h | 6.307% | **7.041%** | 4.055% | 4.102% | **STRONG: EP6 tied with H5 on WSS (7.04 vs 7.00), BETTER on vol_p (4.05 vs 4.18). w_vol_p=0.148 declining toward 0.05 clamp (active in 3-5 EP). Student hasn't posted EP3/EP6 — nudged at 13:25Z.** |
| dl24-nezuko | #1144 | H8: Lion → AdamW lr=5e-4 | ~EP9.5 / 7h | 6.903% | 7.689% | 4.446% | 4.561% | **EP10 marginal: wss 7.69% well over 7.1% gate. τz=10.38% still 1pp above H5 ref — AdamW NOT helping low-SNR axis as hypothesized. Vol_p discipline intact (no GradNorm starvation; w_vol_p 0.31). Likely close at terminal unless EP15+ surprises.** |
| dl24-frieren | #1149 | H10: Charbonnier supplementary WSS loss | EP2 / 2h | 20.14% | 20.88% | 16.21% | 14.60% | **Launched 10:28Z. Student caught my Charb/MSE ratio prediction error — corrected at 13:25Z. EP3 gate (~14:00Z). Floor-safe by construction. τz=27.9% (EP1 noise). Bellwether EP6 τz target ≤9.5%.** |

**Step rate correction**: Both Lion AND AdamW run at ~4-5 steps/sec → 30-epoch run ≈ **33 hours**.

**Wave summary (13:25Z):**
- **H5 tanjiro** CLOSED (#1132) — test_wss=6.609% (BEST WSS IN WAVE, −0.118pp under SOTA) BUT both floors breached. GradNorm starvation root cause identified.
- **H6 frieren** CLOSED (#1135) — test_wss=6.770% tied, both floors breached. τy −0.057pp mechanism confirmed.
- **H7 fern** EP10 PASS — val_surf_p 4.073% (under 4.10% threshold), val_vol_p 3.524% UNDER FLOOR. Slope re-engaged post EP5→6 transient flatline. Terminal surf_p center 3.55-3.65% (borderline). Continue to EP30.
- **H8 nezuko** EP10 marginal — wss 7.69% over 7.1% gate, τz=10.38% over H5 ref by 1pp. AdamW NOT delivering predicted low-SNR τz gain. Vol_p discipline intact (no GradNorm starvation, w_vol_p 0.31). Likely close at terminal.
- **H9 tanjiro** EP6 STRONG — val_wss 7.041 tied with H5, val_vol_p 4.055 BETTER than H5. w_vol_p declining to clamp; clamp activation expected EP9-12. Direct SOTA-chase trajectory ACTIVE.
- **H10 frieren** EP2 — Charbonnier supplementary WSS loss. Student caught my Charb/MSE ratio prediction error (Charb/MSE rises as |δ| falls, not falls). Corrected guidance posted. Bellwether EP6 τz target ≤9.5%.

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
- **H6 frieren (#1135)**: Best remaining candidate. wss<7.0% at EP10, vol_p stable. Watch EP15-20 gate closely.
- **H7 fern (#1142)**: surface_loss_weight=1.5 — clean isolation of the only positive H4 effect. High confidence. Watch EP3-EP10 closely for vol_p under-floor + surf_p not breaching.
- **H8 nezuko (#1144)**: AdamW lr=5e-4 — tier-change sweep. Addresses Lion-noise root cause. Bold swing.

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
