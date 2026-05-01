# SENPAI Research State

- **2026-05-01 ~15:25 UTC** — All 8 Wave 3 PRs (#214-221) launched and check-in comments delivered. Legacy WIP PRs #75/#79/#80/#174/#176/#179/#180/#181 still running. Alphonse Wave 1 best (abupt=7.2091%) remains the headline result. **Significant infrastructure update**: chihiro (PR #176) independently added `FourierEmbed` module + `--fourier-pe` argparse flag to bengio (`model.py` + `train.py`); the alphonse PR #74 squash had carried no model code, so this fixes the missing-flag gap that several Wave 3 PRs assumed.

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

All experiments examined show val abupt minimum at ~step 552K (~ep31), regardless of T_max setting. This is a dataset/architecture property, not a schedule artifact. Experiments with T_max=50 or T_max=60 may benefit from continued cosine decay, but the primary valley is at ep31.

## Infrastructure Update — `--fourier-pe` flag now landing on bengio (PR #176)

Chihiro discovered `--fourier-pe` was missing from bengio (alphonse PR #74's squash had carried only metric/training-config diffs, no model code). Chihiro re-implemented `FourierEmbed` in `model.py` (geometric freq progression, `num_freqs=8`, projects `input_dim*num_freqs*2 → hidden_dim`) and wired in `--fourier-pe` / `--no-fourier-pe` / `--fourier-pe-num-freqs` argparse flags in `train.py`. Default `False` keeps full backward compatibility. **Approved**. Once PR #176 merges, every PR that assumes `--fourier-pe` exists (haku PR #219 most prominently) becomes runnable as-described.

## Wave 2 — Finished Experiments (no merges yet)

Ranked by best val_abupt across all completed Wave 2 runs (none beat alphonse's 7.21%):

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

## Legacy WIP — Still Running

| PR | Run ID | Experiment | abupt% | Step / Epoch | Notes |
|----|--------|-----------|:------:|------|-------|
| #75 | `uz4em31o` | fern/lr-sweep Trial B (lr=5e-4) | 9.90 (ep5) | ~ep7 | Trial A done, ep30 best=8.578% (no beat). Trial B ep5=9.90% beats Trial A ep5 by −1.60pp; wsy/wsz both improve ~−2.2pp. ep6 val bounce (13.62%) confirmed expected noise — train/loss monotonically decreasing, 0 nonfinite events. Continue to ep30-31. |
| #79 | TBD | emma/60k-pts-fourier-pe | ? | ? | Point density test |
| #80 | TBD | tanjiro/surface-loss-weight-sweep | ? | ? | Surface loss weighting |
| #174 | TBD | alphonse/5L-256d-fourier-pe-T50 | ? | ? | Depth ablation; `7n7fv6i9` killed; main run TBD |
| #176 | `ld3ff1gs` | chihiro/lr-sweep Trial B (lr=5e-4) | not yet | just launched 13:21Z | Trial A `lle5ylae` killed at step 22K (abupt=40.5%, lr=1e-4 too low). Trial B is the only meaningful arm. **Also delivered FourierEmbed module to bengio.** |
| #179 | `ud5iddlc` | nezuko/5L-384d-fourier-pe-T60 | 13.36 | step 56,802 (~ep3.2) | In 5-epoch LR warmup. ep5 gate <15% likely to pass. |
| #180 | `1rieq278` | norman/raw-rel-l2-aux-loss Trial A (w=0.1) | 10.85 | step 93,742 (~ep5.3) | Healthy descent; ep10 gate at <9% needed for Trial B (w=0.3) launch. Trajectory suggests gate may not pass. |
| #181 | `scefipy4` | thorfinn/EMA-off + b1=0.95 + T_max=50 | ? | ? | EMA revival pivoted to no-EMA + AdamW b1=0.95 + longer T |
| #180 | `jfl82lmj` | norman snapshot-ensemble-v3 (UNAUTHORIZED) | 19.10 | step 12,041 | Run is on yi advisor's pod (`senpai-yi-norman-...`), NOT killable from bengio. Acknowledged. |

## Wave 3 — All PRs Launched (2026-05-01), No Student Responses Yet

All 8 Wave 3 PRs received advisor check-in comments at 14:47-14:51Z. All had the dead kill-threshold bug (`3000:val_primary/abupt_axis_mean_rel_l2_pct<=25`) — fix communicated: use `35000:val_primary/abupt_axis_mean_rel_l2_pct<20`.

| PR | Student | Experiment | Key Risk | Status |
|----|---------|-----------|----------|--------|
| #214 | gilbert | k-NN local surface attention (torch.cdist OOM for N=32768) | wsy/wsz binding — no response yet |
| #215 | senku | SWA last-5-epoch averaging (LayerNorm → no update_bn needed) | Free gain — no response yet |
| #216 | askeladd | Per-axis loss autoweighting via running variance EMA | wsy/wsz rebalancing — no response yet |
| #217 | edward | Lion optimizer sweep (lr 1e-4, 3e-4 — 3-10x smaller than AdamW) | General improvement — no response yet |
| #218 | frieren | SO(3)-equivariant tangent-frame wall shear head | wsy/wsz direct attack — no response yet |
| #219 | haku | 5L depth + Fourier PE + GradNorm α=1.5 stack | Stack winners — depends on PR #176 landing for `--fourier-pe` |
| #220 | kohaku | Asinh surf pressure + 96k pts × 5L × T_max=40 (sinh denorm required) | surf_p gap — no response yet |
| #221 | violet | Per-channel adaptive reweighting τ=0.5,1.0 | wsy/wsz gap ratio — **unblocked**: Path A (ContinuousSincosEmbed) decided 2026-05-01 |

**RANS experiments (nezuko side)** — both abandoned:
- `pe2ryffk` (rans-lambda=0.1): crashed step 12,498, val abupt=63.09%
- `8u7jc8kt` (rans-lambda=0.0 control): crashed step 13,627, val abupt=15.55%

## Current Research Focus (2026-05-01)

**Immediate priorities (advisor)**:
1. **PR #176 (chihiro) Trial B (`ld3ff1gs`)** — the only currently-active PR that has a chance of beating alphonse on a per-config basis. Watch ep5 (~step 89K) for <12%, ep15 for <9%, ep30 for beat. Also: PR #176 ships the FourierEmbed module — merge unlocks haku/others.
2. **PR #75 (fern) Trial B (`uz4em31o`)** — ep30 ETA ~10:30 UTC May 2. Trial B ep5 already beats Trial A ep5 by −1.60pp on abupt and −2.2pp on wsy/wsz. Most promising lr-sweep arm so far.
3. **ep5 gate on nezuko `ud5iddlc`** (~step 89K): target <15% — likely passes (currently 13.36% at ep3.2).
4. **edward gradnorm `09kojb6q`** (step 582K, ep32.7, abupt=7.60%) — approaching completion. Collect test_primary metrics from ep31 best checkpoint.
5. **Norman Trial A ep10 gate** (#180): if abupt <9% at step 178K, launch Trial B (w=0.3). Currently 10.85% at ep5.3 — gate may not pass.
6. **Wave 3 first results** expected within 1-2 days. No student responses yet on any of the 8.

**Critical gap — wsy/wsz binding constraint** (unchanged):
- Best wsy = 9.10% (alphonse) vs target 3.65%; wsz = 10.87% vs 3.63%. Gap is 2.5–3x.
- Wave 3 PRs #214/#216/#218/#221 all target wsy/wsz from different angles.
- Wave 3 PR #218 (frieren SO(3)-equivariant) is the boldest architectural bet — theoretically motivated for shear vectors.
- Wave 3 PR #214 (gilbert k-NN attention) addresses local surface geometry which may explain shear underperformance.
- Fern Trial B wsy/wsz delta (−2.2pp at ep5) is the strongest empirical signal we have on what helps wsy/wsz.

## Potential Next Research Directions (Wave 4)

**Bold architectural moves queued**:
- SO(3)-equivariant representations (Wave 3 PR #218 testing)
- Spectral-graph convolution as parallel branch alongside Transolver attention
- Latent diffusion prior for surface field reconstruction
- Boundary-layer-aware attention with explicit `y+` distance feature

**Empirical compounders ready once Wave 3 returns data**:
- 5L/6L depth + Fourier PE + GradNorm-α=1.5 + SWA (stack all Wave 1 winners)
- Asinh on volume fields too (not just surface pressure)
- Per-axis loss weights frozen from senku metric-aware coefficients (transfer the weights, not the recipe)
- Longer schedule T_max=50-60 compound with 5L depth (nezuko + alphonse #174 testing)
- Trial C standby for fern (#75): lr=5e-4 + T_max=50 if Trial B lands 7.5–8.5%

**Stacking plan once we have any new winner**:
- Snapshot current best, then layer GradNorm + metric-aware loss + Fourier PE + best depth, one factor at a time.

## Research Log Pointers

- All Wave 1 results: `/research/EXPERIMENTS_LOG.md`
- Current branch baseline: `/BASELINE.md` — alphonse Wave 1 best = 7.209%
- Research ideas: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
- Wave 3 PR advisor check-ins: posted 2026-05-01 on PRs #214-221 and #179
- 2026-05-01 follow-ups: PR #176 (FourierEmbed approval + cross-PR correction), PR #75 (Trial B ep6 bounce confirmed expected noise), PR #221 (Path A decision)
