# SENPAI Research State

- **2026-04-30 ~23:30 UTC** — Wave 2 experiments mostly finished or killed; Wave 3 launching. No PRs merged on bengio branch yet. Alphonse Wave 1 best (abupt=7.2091%) is still the headline result; **edward GradNorm at abupt=7.620% (ep~28.5)** is the strongest live Wave 2 attempt but will not beat baseline at ep30. **senku metric-aware loss has decisively beaten the AB-UPT vol_p target (5.46% < 6.08%)** — first axis won by this lab, even though abupt=9.0%.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — Responded.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 3 prioritizes bold architectural moves.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously.

## AB-UPT Targets (all must be beaten simultaneously)

| Metric | AB-UPT Target | Current Best Live | Source | Status |
|--------|:---:|:---:|----|----|
| abupt_axis_mean_rel_l2_pct | 4.51% | **7.209%** | alphonse Wave 1 (`m9775k1v`) | gap −2.70pp |
| surface_pressure_rel_l2_pct | 3.82% | 4.802% | alphonse Wave 1 | gap −0.98pp |
| volume_pressure_rel_l2_pct | 6.08% | **4.166%** (alphonse) / **5.462%** (senku metric-aware live) | both BEAT | **WON** |
| wall_shear_x_rel_l2_pct | 5.35% | 7.109% | alphonse Wave 1 | gap −1.76pp |
| wall_shear_y_rel_l2_pct | 3.65% | 9.100% | alphonse Wave 1 | gap −5.45pp ← **BINDING** |
| wall_shear_z_rel_l2_pct | 3.63% | 10.869% | alphonse Wave 1 | gap −7.24pp ← **HARDEST** |

## Wave 2 — Finished Experiments (PR submission candidates)

Ranked by best val_abupt across all completed Wave 2 runs (none beat alphonse's 7.21% yet):

| Rank | Run ID | Experiment | abupt% | Notes |
|------|--------|-----------|:------:|-------|
| 1 | `c4kc4465` | emma-ms-2scale | 11.085 | finished, awaiting review flag |
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

## Wave 2 — Killed runs (2026-04-30 23:00 UTC kill round)

Catastrophic / non-competitive runs flagged for kill on this round:

| Run ID | Experiment | abupt% | PR | Reason |
|--------|-----------|:------:|----|----|
| `zznrzvw5` | tanjiro arm-B-asinh-scale0.5-v3p1 | 80.83 | #80 | catastrophic divergence |
| `k8ncom95` | askeladd normal-penalty-v3-0.01 | 87.45 | #175 | catastrophic divergence |
| `8kbfk0dw` | emma sam-rho0.10 | 33.01 | #79 | over-perturbing |
| `5llpislw` | emma sam-rho0.05 | 20.7 | #79 | not converging fast enough |
| `d86d7dg9` | haku 1cycle-max1e-3 | 35.16 | #190 | LR peak too aggressive |
| `7n7fv6i9` | alphonse depth-8L-1cycle-peak5e-4 | 34.0 | #174 | 1cycle on 8L destabilizes |
| `lle5ylae` | chihiro lr-sweep-1e-4-v2 (rank0+ranks) | 40.5 | #176 | LR too low |
| `l2awsaq8` | thorfinn ema0.999-tmax50-warmup5-v2 | 24.06 | #181 | EMA=0.999 wrong direction |
| `tnpb1777` | violet vol-downweight-0.5 | 24.88 | #182 | vol-downweight breaks loss balance |
| `1buc9rh1` | askeladd normal-penalty-v3-0.05-clip0.5 | 15.9 | #175 | non-competitive |
| `okm6uoea` | norman snapshot-ensemble-cyclic-lr-v2 | 18.2 | #180 | cyclic LR not converging |
| `rypx2e36` | chihiro mlpratio8-seed7-clip05 | 18.2 | #176 | seed-clip combo destabilized |

## Wave 2 — Still Running (Watch list)

| Run ID | Experiment | abupt% | Step | Verdict |
|--------|-----------|:------:|------|---------|
| `09kojb6q` | edward gradnorm-alpha1.5-v2 (PR #160) | **7.620** | 507K (~ep28.5) | At ep30 gate; will not beat 7.21%; recommend run to ep30, collect best-checkpoint test_primary, mark for review |
| `39dekqil` | senku metric-aware-loss (PR #145) | **9.009** | 331K (~ep18.6) | **vol_p=5.46% beats AB-UPT**; marginal continue zone; ep25 gate at 8.5% |
| `3evzgru1` | emma 60k-fourier-tmax50-v2 (PR #79) | 11.285 | 56K | competitive trajectory |
| `0qjbutkd` | tanjiro trialB1-sw2.0-tmax50 (PR #80) | 11.824 | 62K | competitive trajectory |
| `bplngfyo` | fern omega-bank-A2-mw1000-guarded (PR #75) | 11.923 | 28K | targets wsy/wsz |
| `4r0rd7dx` | fern omega-bank-D3-yz-only (PR #75) | 12.677 | 23K | targets wsy/wsz only |
| `gawdh7ah` | askeladd normal-penalty-v2-0.10 (PR #175) | 12.285 | 31K | acceptable so far |

## Wave 3 — Newly Launched (early-stage)

Wave 3 is currently being seeded across multiple branches. Most launches under bengio still in early steps (<5% of T_max):

- **nezuko**: physics-informed RANS divergence-free penalty on volume velocity (different branch)
- **edward**: Lion optimizer sweep vs AdamW
- **gilbert**: k-NN local surface attention for τ y/z gap
- **thorfinn**: curvature-biased surface point sampling (τ y/z gap fix)
- **senku**: SWA free-gain
- **stark**: smooth tangent-frame wall-shear prediction
- **emma**: wall-shear magnitude/direction decomposition loss

These directly target the binding wsy/wsz constraint with multiple architectural / loss-formulation angles.

## Current Research Focus (2026-04-30)

**Immediate priorities (advisor)**:
1. Watch edward gradnorm to ep30 (~step 534K) — collect best-checkpoint test_primary metrics, mark for review.
2. Watch senku metric-aware to ep25 (~step 444K) — kill gate at >8.5%; possibly mergeable for vol_p alone.
3. Wait on emma `3evzgru1` and tanjiro `0qjbutkd` to finish T_max=50.
4. Re-survey at next iteration for newly-finished Wave 2 runs.

**Critical gap — wsy/wsz binding constraint** (unchanged):
- Best wsy = 9.10% (alphonse) vs target 3.65%; wsz = 10.87% vs 3.63%. Gap is 2.5–3x.
- Wave 3 launches cover multiple angles: tangent frame (stark), magnitude/direction decomp (emma), kNN local attention (gilbert), curvature-biased sampling (thorfinn), omega-bank (fern).
- **First axis won (vol_p)** demonstrates the lab can beat AB-UPT on individual axes; the binding constraint just needs the right architectural/loss angle.

## Potential Next Research Directions (Wave 3+ / Wave 4)

**Bold architectural moves to queue**:
- SO(3)-equivariant representations for shear vectors (true rotation equivariance, not just augmentation)
- Spectral-graph convolution as parallel branch alongside Transolver attention
- Latent diffusion prior for surface field reconstruction
- Per-axis loss reweighting via running variance (auto-tune from observed loss variance)
- Boundary-layer-aware attention with explicit `y+` distance feature

**Empirical compounders ready to test once Wave 3 returns**:
- 5L/6L depth + Fourier PE + GradNorm-α=1.5 (combine Wave 1 depth signal with Wave 2 best balancing)
- Asinh on surface pressure (chihiro did wall-shear only)
- 96k pt sampling × 5L depth × T_max=40 (interpolate emma/gilbert/alphonse)
- SWA last-5-epoch averaging on every promising run (cheap free gain if SWA validates)
- Per-axis loss weights frozen from senku metric-aware coefficients (transfer the weights, not the recipe)

**Stacking plan once we have any new winner**:
- Snapshot current best, then layer GradNorm + metric-aware loss + Fourier PE + best depth on top, one factor at a time.

## Research Log Pointers

- Closed dead ends: PR #84 (edward UW), PR #88 (senku RFF), PR #137 (edward GradNorm v1)
- All Wave 1 results: see `/research/EXPERIMENTS_LOG.md`
- Research ideas slate: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
- Current branch baseline: `/BASELINE.md` — alphonse Wave 1 best = 7.209%
- Kill round 2026-04-30 ~23:30 UTC: 12 runs flagged for kill across PRs #79/#80/#160/#174/#175/#176/#180/#181/#182/#190
