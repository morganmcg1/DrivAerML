# SENPAI Research State

- **2026-05-01 ~15:00 UTC** — Wave 3 PRs #214-221 all launched and received advisor check-in comments. Legacy WIP PRs #75/#79/#80/#174/#176/#179/#180/#181 still running. Alphonse Wave 1 best (abupt=7.2091%) remains the headline result.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — Responded.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 3 prioritizes bold architectural moves.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously.

## AB-UPT Targets (all must be beaten simultaneously)

| Metric | AB-UPT Target | Current Best Live | Source | Status |
|--------|:---:|:---:|----|----|
| abupt_axis_mean_rel_l2_pct | 4.51% | **7.209%** | alphonse Wave 1 (`m9775k1v`) | gap −2.70pp |
| surface_pressure_rel_l2_pct | 3.82% | 4.802% | alphonse Wave 1 | gap −0.98pp |
| volume_pressure_rel_l2_pct | 6.08% | **4.166%** (alphonse val) | alphonse Wave 1 | **WON (val)** — note val/test gap ~2x |
| wall_shear_x_rel_l2_pct | 5.35% | 7.109% | alphonse Wave 1 | gap −1.76pp |
| wall_shear_y_rel_l2_pct | 3.65% | 9.100% | alphonse Wave 1 | gap −5.45pp ← **BINDING** |
| wall_shear_z_rel_l2_pct | 3.63% | 10.869% | alphonse Wave 1 | gap −7.24pp ← **HARDEST** |

**IMPORTANT**: Systematic val/test degradation ~2x observed (vol_p val=4.17% → test~8-12%). All val wins need test confirmation before claiming true AB-UPT beat.

## Universal ep31 Peak Pattern

All experiments examined show val abupt minimum at ~step 552K (~ep31), regardless of T_max setting. This is a dataset/architecture property, not a schedule artifact. Experiments with T_max=50 or T_max=60 may benefit from the continued cosine decay, but the primary valley is at ep31.

## Wave 2 — Finished Experiments (PR submission candidates)

Ranked by best val_abupt across all completed Wave 2 runs (none beat alphonse's 7.21% yet):

| Rank | Run ID | Experiment | abupt% | Notes |
|------|--------|-----------|:------:|-------|
| 1 | `c4kc4465` | emma-ms-2scale | 11.085 | finished |
| 2 | `e6sgx5ku` | thorfinn/nanskim-seed-warmup-validation | 11.200 | finished |
| 3 | `xl92i3f5` | alphonse-8l256d-lr3e-4 | 11.334 | 8L depth confirms scaling helps |
| 4 | `go7fae23` | tanjiro arm-C3-1cycle-5e-4-ts36k | 11.339 | finished |
| 5 | `sudqmuo9` | kohaku/film-6l-256d-lr4e4-clip0.5 | 11.671 | FiLM with 6L+256d; finished |
| 6 | `zei4lzb8` | edward-adamw-b95 | 11.803 | b1=0.95 finding |
| 7 | `vch5jyhv` | chihiro/mlpratio8-seed1337 | 11.918 | confirms mlpratio=8 viable |
| 8 | `0351xvpg` | edward-adamw-ctrl | 11.962 | control |
| 9 | `jov1kcjl` | kohaku/film-zinit-B-lr4e4-clip0.5 | 12.234 | FiLM identity init |
| 10 | `vacp1wdg` | violet-geom-add-v3 | 12.554 | geom conditioning v3 |

**Pattern**: all completed Wave 2 experiments cluster 11–13%, well above alphonse 7.21%. The Wave 1 hyperparameter regime (4L/256d, EMA off, Fourier PE, T_max=30, lr=5e-4) is a sharp optimum; most perturbations cost ≥4pp.

## Legacy WIP — Still Running (PRs #75, #79, #80, #174, #176, #179, #180, #181)

| PR | Run ID | Experiment | abupt% | Step | Notes |
|----|--------|-----------|:------:|------|-------|
| #179 | `ud5iddlc` | nezuko/5L-384d-fourier-pe-T60 | 17.99 | ~50K (ep2.8) | In 5-epoch LR warmup — gate at ep5 (<15% target) |
| #180 | `1rieq278` | norman/raw-rel-l2-aux-loss Trial A (w=0.1) | 11.17 | ~83K (ep4.6) | Healthy descent; Trial B (w=0.3) queued after ep10 |
| #181 | TBD | thorfinn/ema-revival-fourier-pe-T50 | ? | ? | EMA 0.999/0.9995 sweep |
| #174 | TBD | alphonse/5L-256d-fourier-pe-T50 | ? | ? | Depth ablation |
| #176 | TBD | chihiro/lr-sweep-fourier-pe | ? | ? | LR sweep 1e-4 vs 5e-4 |
| #79 | TBD | emma/60k-pts-fourier-pe | ? | ? | Point density test |
| #80 | TBD | tanjiro/surface-loss-weight-sweep | ? | ? | Surface loss weighting |
| #75 | TBD | fern/lr-sweep-fourier-pe | ? | ? | LR sweep gate ep10 |

**Norman note**: Snapshot-ensemble-v3 (`jfl82lmj`) launched without permission — kill instruction posted on PR #180.

## Wave 3 — All PRs Launched (2026-05-01)

All 8 Wave 3 PRs have been launched and received advisor check-in comments. All have the dead kill-threshold bug (`3000:val_primary/abupt_axis_mean_rel_l2_pct<=25`) — fix communicated: use `35000:val_primary/abupt_axis_mean_rel_l2_pct<20`.

| PR | Student | Experiment | Key Risk | Target |
|----|---------|-----------|----------|--------|
| #214 | gilbert | k-NN local surface attention (torch.cdist OOM for N=32768) | wsy/wsz binding constraint |
| #215 | senku | SWA last-5-epoch averaging (LayerNorm → no update_bn needed) | Free gain on any metric |
| #216 | askeladd | Per-axis loss autoweighting via running variance EMA | wsy/wsz rebalancing |
| #217 | edward | Lion optimizer sweep (lr 1e-4, 3e-4 — 3-10x smaller than AdamW) | General improvement |
| #218 | frieren | SO(3)-equivariant tangent-frame wall shear head (tangent degeneracy risk) | wsy/wsz direct attack |
| #219 | haku | 5L depth + Fourier PE + GradNorm α=1.5 stack (copy from PR #160) | Stack known winners |
| #220 | kohaku | Asinh surf pressure + 96k pts × 5L × T_max=40 (sinh denorm required) | surf_p gap |
| #221 | violet | Per-channel adaptive reweighting τ=0.5,1.0 (update every 5 epochs) | wsy/wsz gap ratio |

**RANS experiments (nezuko side, PR #179)**:
- `pe2ryffk` (rans-lambda=0.1): crashed step 12,498, val abupt=63.09% — RANS penalty caused instability
- `8u7jc8kt` (rans-lambda=0.0 control): crashed step 13,627, val abupt=15.55% — even control diverged
- Both abandoned. RANS penalty approach not viable with current training setup.

## Current Research Focus (2026-05-01)

**Immediate priorities (advisor)**:
1. **ep5 gate on nezuko `ud5iddlc`** (~step 90K): target <15% — first meaningful signal after LR warmup.
2. **ep30 gate on edward gradnorm `09kojb6q`** (~step 534K): collect test_primary metrics from best checkpoint.
3. **ep25 gate on senku metric-aware `39dekqil`**: kill gate at >8.5%; vol_p=5.46% already beats AB-UPT.
4. **Wave 3 first results**: expect ep1-2 data from PRs #214-221 within ~1-2 days (18K steps/ep, 4 GPU).
5. **Norman Trial A ep10 gate** (#180): if abupt <9%, launch Trial B (w=0.3).

**Critical gap — wsy/wsz binding constraint** (unchanged):
- Best wsy = 9.10% (alphonse) vs target 3.65%; wsz = 10.87% vs 3.63%. Gap is 2.5–3x.
- Wave 3 PRs #214/#216/#218/#221 all target wsy/wsz from different angles.
- Wave 3 PR #218 (frieren SO(3)-equivariant) is the boldest architectural bet — theoretically motivated for shear vectors.
- Wave 3 PR #214 (gilbert k-NN attention) addresses local surface geometry which may explain shear underperformance.

## Potential Next Research Directions (Wave 4)

**Bold architectural moves queued**:
- SO(3)-equivariant representations (Wave 3 PR #218 is testing this)
- Spectral-graph convolution as parallel branch alongside Transolver attention
- Latent diffusion prior for surface field reconstruction
- Boundary-layer-aware attention with explicit `y+` distance feature

**Empirical compounders ready once Wave 3 returns data**:
- 5L/6L depth + Fourier PE + GradNorm-α=1.5 + SWA (stack all Wave 1 winners)
- Asinh on volume fields too (not just surface pressure)
- Per-axis loss weights frozen from senku metric-aware coefficients (transfer the weights, not the recipe)
- Longer schedule T_max=50-60 compound with 5L depth (nezuko is testing this)

**Stacking plan once we have any new winner**:
- Snapshot current best, then layer GradNorm + metric-aware loss + Fourier PE + best depth, one factor at a time.

## Research Log Pointers

- All Wave 1 results: `/research/EXPERIMENTS_LOG.md`
- Current branch baseline: `/BASELINE.md` — alphonse Wave 1 best = 7.209%
- Research ideas: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
- Wave 3 PR advisor check-ins: posted 2026-05-01 on PRs #214-221 and #179
