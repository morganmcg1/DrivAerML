# SENPAI Research State
- 2026-05-16 06:28 UTC (post-H3 nezuko terminal, H8 assigned)

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

## Active Experiments (2026-05-16 06:28 UTC)

| Student | PR | Hypothesis | Latest EP | val_abupt | val_wss | val_vol_p | Status |
|---------|-----|-----------|----------:|----------:|--------:|----------:|--------|
| **dl24-nezuko** | **#1144** | **H8: Lion → AdamW (lr=5e-4)** | **JUST LAUNCHED** | — | — | — | **NEW 06:28Z. Plateau Protocol tier change.** |
| dl24-fern | #1142 | H7: surface_loss_weight=1.5 | RUNNING (EP0-2 est.) | — | — | — | Launched 04:25Z Option B. EP3 gate ~06:30Z. |
| dl24-frieren | #1135 | H6: wind-exposure additive attn bias | EP14 best-val est. | ~6.28% | ~6.99% | ~4.11% | EP15 gate ~07:30Z. Terminal ~14:40Z. |
| dl24-tanjiro | #1132 | H5: curvature additive attn bias | EP26 est. | ~6.18% | ~6.84% | ~4.31% | Plateau confirmed EP21. vol_p drifting up. Terminal ~07:00-08:00Z. Likely CLOSE. |

## Plateau Protocol Status (ACTIVE)

**4 consecutive closed failures + 2 plateauing experiments = Plateau Protocol triggered.**

| PR | Hypothesis | Status | Key failure mode |
|----|-----------|--------|-----------------|
| #1115 H1 | Wind-exposure input channels | ❌ CLOSED | gradnorm task-share imbalance (vol_p breach +0.437pp) |
| #1117 H2 | Curvature input channels | ❌ CLOSED | Same imbalance (vol_p breach +0.340pp); WSS improved −0.059pp (signal valid) |
| #1129 H3 | Near-wall volume cross-attn | ❌ CLOSED (06:25Z today) | Cross-attn duplicated existing backbone signal; τ_z WORSE +0.355pp |
| #1130 H4 | Per-axis WSS loss weights | ❌ CLOSED (02:33Z today) | Lion-noise on weighted axes; surf_p floor breach |
| #1132 H5 | Curvature additive attn bias | ⏳ Plateauing EP21, vol_p drifting | WSS stalled at 6.84% (worse than SOTA); vol_p drifting up |
| #1135 H6 | Wind-exposure additive attn bias | ⏳ Running, EP15 gate pending | wss<7.0% milestone hit EP10; vol_p stable band |

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
