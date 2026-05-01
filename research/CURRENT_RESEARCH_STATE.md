# SENPAI Research State

- **2026-05-01 14:30 UTC** — Wave 3 launched but students haven't published comments yet. Wave 2 still has several active runs. **Headline**: alphonse Wave 1 best (abupt=7.2091%) remains the leaderboard. **vol_p axis WON** at 4.166% (alphonse) vs AB-UPT 6.08%. **Strong new signal**: fern Trial B (`uz4em31o`, lr=5e-4 + Fourier PE) at ep5 = 9.90% with -2.20pp on wsy and -2.18pp on wsz vs Trial A — wall-shear gap is closing under higher LR.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — Responded.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 3 prioritizes bold architectural moves including SO(3)-equivariant heads and tangent-frame decomposition.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously.

## AB-UPT Targets (all must be beaten simultaneously)

| Metric | AB-UPT Target | Current Best Live | Source | Status |
|--------|:---:|:---:|----|----|
| abupt_axis_mean_rel_l2_pct | 4.51% | **7.209%** | alphonse Wave 1 (`m9775k1v`) | gap −2.70pp |
| surface_pressure_rel_l2_pct | 3.82% | 4.802% | alphonse Wave 1 | gap −0.98pp |
| volume_pressure_rel_l2_pct | 6.08% | **4.166%** | alphonse Wave 1 | **WON** |
| wall_shear_x_rel_l2_pct | 5.35% | 7.109% | alphonse Wave 1 | gap −1.76pp |
| wall_shear_y_rel_l2_pct | 3.65% | 9.100% | alphonse Wave 1 | gap −5.45pp ← **BINDING** |
| wall_shear_z_rel_l2_pct | 3.63% | 10.869% | alphonse Wave 1 | gap −7.24pp ← **HARDEST** |

## Wave 2 — Closed since last update

| PR | Run ID | Reason | Final abupt% |
|----|--------|--------|:------:|
| #145 | `39dekqil` | senku metric-aware loss — killed at ep20 gate (>8.5% threshold) | 8.891% (vol_p WON: 5.46% < AB-UPT 6.08%) |
| #160 | `09kojb6q` | edward gradnorm-α=1.5 — does not beat alphonse | 7.5986% |

## Wave 2 — Still Running (Watch list, 2026-05-01 14:30 UTC)

| Run ID | Experiment | Latest abupt% | Step | Status / Gates |
|--------|-----------|:------:|------|---------|
| `uz4em31o` | fern lr-sweep Trial B (lr=5e-4) PR #75 | **9.897% @ ep5** (ep6 bounce 13.62%, ep7 in progress) | ~115K | Strong wsy/wsz delta vs Trial A; ep10 gate <8.0% wanted |
| `0qjbutkd` | tanjiro Trial B1 (SW=2.0, T_max=50) PR #80 | 11.82% @ ~62K | ~62K | Trial A done: test 9.697%; B2/B3 queued |
| `3evzgru1` | emma 60k-fourier-tmax50-v2 PR #79 | 10.799% @ ep5 | ~56K | ep15<9.0%, ep20<8.0%, ep30<7.5%, ep50 terminal |
| `bplngfyo` | fern omega-bank-A2-mw1000-guarded PR #75 | 11.923% | 28K | targets wsy/wsz |
| `4r0rd7dx` | fern omega-bank-D3-yz-only PR #75 | 12.677% | 23K | targets wsy/wsz only |
| `gawdh7ah` | askeladd normal-penalty-v2-0.10 PR #175 | 12.285% | 31K | acceptable |
| `scefipy4` | thorfinn no-EMA + b1=0.95 + T_max=50 + warmup=5 PR #181 | running | early | Wave 1 lessons + edward b95 stack |
| `ld3ff1gs` | chihiro lr-sweep-5e-4 Trial B PR #176 | early | early | mirroring fern Trial B; Trial A killed |
| `ud5iddlc` | nezuko 5L/384d Fourier PE T_max=60 PR #179 | early | early | wide+deep ~45h ETA |

## Wave 2 — Already finished, awaiting potential PR submission

| Run ID | Experiment | abupt% | Notes |
|--------|-----------|:------:|-------|
| `c4kc4465` | emma-ms-2scale | 11.085 | finished |
| `e6sgx5ku` | thorfinn nanskim-seed-warmup-validation | 11.200 | finished |
| `xl92i3f5` | alphonse-8l256d-lr3e-4 | 11.334 | 8L depth confirms scaling helps |
| `go7fae23` | tanjiro arm-C3-1cycle-5e-4-ts36k | 11.339 | finished |
| `sudqmuo9` | kohaku/film-6l-256d-lr4e4-clip0.5 | 11.671 | FiLM with 6L+256d |
| `zei4lzb8` | edward-adamw-b95 | 11.803 | b1=0.95 finding |
| `vch5jyhv` | chihiro/mlpratio8-seed1337 | 11.918 | mlpratio=8 viable |
| `0351xvpg` | edward-adamw-ctrl | 11.962 | control |
| `vacp1wdg` | violet-geom-add-v3 | 12.554 | geom conditioning v3 |

## Wave 3 — Active PRs (launched 2026-04-30, not yet acknowledged by students)

All 8 students assigned. Zero comments on any PR yet — students are presumably setting up commands or already running silently.

| PR | Student | Experiment | Hypothesis |
|----|---------|-----------|-----------|
| #214 | gilbert | knn-local-surface-attention-wsy-wsz | k-NN local attention post-backbone for tangential shear |
| #215 | senku | swa-free-gain | SWA last-5-epoch averaging — free generalization gain |
| #216 | askeladd | per-axis-loss-autoweighting | Dynamic per-axis weights from running variance (Kendall-style) |
| #217 | edward | lion-optimizer-sweep | Lion optimizer (sign-based) vs AdamW |
| #218 | frieren | so3-equivariant-wall-shear | Tangent-frame head: predict shear in (e_t1, e_t2) frame |
| #219 | haku | depth5-fourier-gradnorm-stack | 5L + GradNorm α=1.5 compound stack |
| #220 | kohaku | asinh-surf-pressure-96k-pts | Asinh surface pressure + 96k pts × 5L × T_max=40 |
| #221 | violet | per-channel-adaptive-reweighting | Dynamic channel weights from live gap to AB-UPT targets |

## Current Research Focus (2026-05-01 14:30 UTC)

**Immediate priorities**:
1. **Watch fern Trial B `uz4em31o`** — strongest live signal. ep5=9.90%, biggest deltas on exactly the binding wsy/wsz axes. ep10 expected ~8.0% if trajectory holds. ETA finish ~10:30 UTC May 2.
2. **Watch tanjiro B1 `0qjbutkd`** — T_max=50 + SW=2.0 finishing; B2/B3 queued. SW=2.0 trial A test=9.697% does not beat alphonse, prognosis poor.
3. **Watch emma Trial B v2 `3evzgru1`** — ep5=10.80%; advisor gates at ep15/20/30/50.
4. **Watch thorfinn `scefipy4`** — clean Wave 1 + b1=0.95 stack; could be a strong recipe.
5. **Watch nezuko `ud5iddlc`** — 5L/384d wide+deep with Fourier PE.
6. **Wave 3 PRs (#214–221)** — first val signals expected within ~24h. None acknowledged yet.

**Critical gap — wsy/wsz binding constraint**:
- Best wsy = 9.10% vs target 3.65% (gap 2.5×); wsz = 10.87% vs 3.63% (gap 3×).
- fern Trial B is the **first concrete sign** that higher LR (5e-4) preferentially helps wsy/wsz over the easier channels.
- Wave 3 attacks this from: tangent-frame SO(3) head (frieren), kNN local attention (gilbert), curvature-biased sampling (thorfinn), omega-bank (fern), adaptive reweighting (violet/askeladd).
- **First axis won (vol_p)** — alphonse 4.166% < AB-UPT 6.08%.

## Session findings (2026-05-01)

- fern Trial B confirms **lr=5e-4 + Fourier PE** is the right neighborhood: ep5 is 1.6pp better than Trial A on abupt and gains 2.2pp on wsy/wsz specifically.
- Tanjiro surface-loss-weighting (SW=2.0) is **NOT the right lever**: test_primary 9.697% vs alphonse 8.48%. Surface weighting shifts capacity away from the wall-shear gap.
- val/test gap on vol_p is dataset-level (not model-specific): val ~5.2-5.9% → test ~12.9% across multiple recipes. Holdout test contains harder volume-pressure cases.
- Cosine T_max=30 + 50 epochs creates a bounce at ep30; T_max=50 schedules eliminate this.
- All Wave 1 hyperparameter regime conclusions hold: 4L/256d, EMA off, Fourier PE, T_max=30 (or 50 for longer schedules), lr=5e-4.

## Potential Next Research Directions (Wave 4+)

**Bold architectural moves**:
- Spectral-graph convolution as parallel branch alongside Transolver attention
- Latent diffusion prior for surface field reconstruction
- Boundary-layer-aware attention with explicit `y+` distance feature
- Message-passing GNN with geometric edge features (surface curvature, principal directions)
- Transformer with geometric algebra tokens (rotor-equivariant attention)

**Empirical compounders ready once Wave 3 returns**:
- 5L/6L depth + Fourier PE + GradNorm-α=1.5 + SWA (compound haku + senku findings)
- Per-axis frozen weights from senku metric-aware run (transfer weights, not the recipe)
- Ensemble: average top-3 Wave 1 checkpoints (different seeds, same recipe)
- AdamW β1=0.95 + GradNorm + T_max=50 (combining edward Wave 2, thorfinn current, axis balancing)
- **lr=5e-4 + 5L/256d + Fourier PE + warmup=5 + T_max=50** (fern Trial B + nezuko depth + thorfinn schedule)

**Stacking plan once any Wave 3 experiment wins**:
- Snapshot current best, then layer in GradNorm + metric-aware loss + Fourier PE + best depth, one factor at a time.

## Research Log Pointers

- Closed dead ends: PR #84 (edward UW), PR #88 (senku RFF), PR #137 (edward GradNorm v1), PR #145 (senku metric-aware), PR #160 (edward GradNorm v2), PR #182 (violet vol-downweight)
- All Wave 1 results: see `/research/EXPERIMENTS_LOG.md`
- Research ideas slate: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
- Current branch baseline: `/BASELINE.md` — alphonse Wave 1 best = 7.209%
- Kill round 2026-04-30 ~23:30 UTC: 12 runs flagged for kill across PRs #79/#80/#160/#174/#175/#176/#180/#181/#182/#190
