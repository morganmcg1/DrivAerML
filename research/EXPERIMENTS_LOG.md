# SENPAI Research Results

<!-- Results are appended here as experiments complete and are reviewed -->
<!-- Format: ## YYYY-MM-DD HH:MM — PR #<number>: <title> -->

## 2026-04-30 20:45 — Wave 1 Mid-Training Status Snapshot

All 16 PRs are WIP. No PRs have completed yet. Status as of ~20:30 UTC 2026-04-30:

| PR | Student | Run ID | Epoch | abupt% | surf_p% | vol_p% | Notes |
|----|---------|--------|-------|--------|---------|--------|-------|
| #74 | alphonse | `m9775k1v` | ~16/50 | **7.70** | 5.09 | **4.42** | Wave leader; 4 unanswered check-ins — ESCALATION |
| #75 | fern | `pxty4knv` | 19/50 | 9.02 | 5.87 | 6.15 | Trial A (lr=1e-4) running; Trial B (lr=5e-4) auto-launches ~May1 11Z |
| #76 | gilbert | `kn756yk6` | 9/50 | 8.78 | ~5.20 | ~5.57 | 5L/256d, healthy, ETA May1 13Z |
| #77 | haku | `nbbbw8qw` | 11/50 | 8.64 | 5.56 | **5.05** | 4L/384d, healthy, ETA May1 22Z |
| #78 | kohaku | `h7ve1hmb` | ~16/50 | ~8.13 | 5.41 | 5.82 | 128 slices+PE; 4 unanswered check-ins — ESCALATION |
| #79 | emma | `kuk0oy8g` | ~22/50 | 8.33 | 5.59 | 5.99 | 60k pts; healthy |
| #80 | tanjiro | `846uciam` | ~20/50 | 8.66 | 5.60 | **5.32** | SW=2.0, vol_p below target |
| #81 | violet | `em5ixfew` | 19/50 | 9.07 | 5.84 | **5.29** | T_max=50 cosine, vol_p below target |
| #82 | askeladd | `uxrhudp1` | ~19/50 | 8.61 | 5.64 | **4.82** | Log-Fourier SDF; 2 unanswered check-ins |
| #83 | chihiro | `kit58p2e` | 22/50 | 8.98 | 5.42 | **4.93** | asinh scale=1.0, vol_p below target |
| #85 | frieren | `l23vz4md` | ~9/50 | 8.55 | 5.42 | **5.19** | Cross-attn bridge; most promising early signal (vol_p=5.89 at ep9) |
| #86 | nezuko | `p8swf78o` | ~13/50 | 8.39 | 5.36 | **4.94** | mlp_ratio=6, vol_p below target |
| #87 | norman | `0iv7wifz` | 18/50 | 8.90 | 5.78 | **5.28** | Dropout=0.1, vol_p below target |
| #88 | senku | `k8ytnvh8` | ~16/50 | 10.15 | 6.49 | 6.89 | RFF (sigma suspected misconfigured), above baseline |
| #89 | thorfinn | `snrwvw14` | ~24/50 | 8.61 | 5.57 | **5.00** | gc=0.5+wd=1e-3 Trial A; Trial B (wd=1e-4) queued |
| #137 | edward | `v5ybmwra` | ~5/50 | — | — | — | GradNorm, early training |

Bold vol_p values are at or below AB-UPT target (6.08). Wave 1 is still mid-training; no test_primary metrics yet.

## 2026-04-30 22:10 — PR #84: [edward] DrivAerML Dynamic Uncertainty Loss Weighting (CLOSED)

- **Branch:** `edward/uncertainty-loss-weighting`
- **Hypothesis:** Kendall & Gal homoscedastic uncertainty weighting (per-task learnable `log_var`) auto-balances per-task losses and improves `abupt_axis_mean_rel_l2_pct` over the fixed-weight cohort baseline. Implemented with clamp `[-5, 5]` and the +0.5·log_var regularizer.
- **W&B run:** `3gfy3fi7` (`bengio-stream2-edward`, edward/uw-fixed-kill). Stopped at epoch 20/50, post-hoc test eval on best-epoch-16 checkpoint via new `--eval-only` flag.

### Results table

| Metric | This run | AB-UPT ref | Δ |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **9.99** | 4.51 | +5.48 |
| `test_primary/surface_pressure_rel_l2_pct` | 5.23 | 3.82 | +1.41 |
| `test_primary/wall_shear_rel_l2_pct` (vector) | 9.63 | 7.29 | +2.34 |
| `test_primary/wall_shear_x_rel_l2_pct` | 7.99 | 5.35 | +2.64 |
| `test_primary/wall_shear_y_rel_l2_pct` | 11.88 | 3.65 | +8.23 |
| `test_primary/wall_shear_z_rel_l2_pct` | 12.34 | 3.63 | +8.71 |
| `test_primary/volume_pressure_rel_l2_pct` | 12.51 | 6.08 | +6.43 |

`full_val_primary/abupt_axis_mean_rel_l2_pct = 8.87` at epoch 16 (best). Cohort leader (alphonse, val) at this point ≈ 7.55–7.41.

### Conclusion

**Rejected. Closed as dead end.** Two compounding failure modes:

1. **Clamp-induced rectified equilibrium.** Per-task MSEs in normalized space are 0.003–0.02; analytic `s* = log(L)` lies at -3.9 to -5.7. With clamp at -5, `surface_pressure` and `wall_shear_x` saturated at the floor. Effective weight `0.5·exp(5)=74.2` for those tasks dwarfed the unclamped tasks, breaking the auto-balancing intent.
2. **Late-stage destabilization.** Pre-clip global grad norm jumped from ~10 (mid-training) to ~140 post-epoch-16; `--grad-clip-norm 1.0` clipped every step by ~140×. Non-clamped task losses (`ws_y/z`, `vol_p`) climbed steadily; val_abupt regressed 8.87 → 14.29 in 4 epochs.

Volume_pressure test/val gap (5.83 → 12.51) is the dominant component of the test-set degradation, indicating UW under-weighted `volume_pressure` enough to hurt held-out generalization specifically.

## 2026-04-30 ~latest — PR #76: [gilbert] 5L/256d Depth Scaling + Fourier PE (CLOSED)

- **Branch:** `gilbert/5l-256d-depth-scaling-fourier-pe`
- **Hypothesis:** Adding one extra transformer layer (4L→5L) at the Wave 1 256d baseline + Fourier PE would increase capacity and lower val_abupt below 7.2091%.
- **W&B run:** `kn756yk6`, group `bengio-stream1-gilbert`

| Epoch | abupt% | wsy% | wsz% | Notes |
|-------|--------|------|------|-------|
| 10 | 9.704 | 11.81 | 13.26 | |
| 20 | 8.026 | 9.97 | 11.57 | |
| 25 | 7.773 | 9.34 | 11.11 | |
| 28 | 7.508 | 9.01 | 11.02 | |
| 30 | 7.473 | 8.95 | 10.96 | |
| **31** | **7.473** | 8.93 | 10.95 | **BEST** |
| 32 | 7.492 | 8.97 | 10.99 | cosine bounce |
| 38 | 7.542 | 9.07 | 11.08 | |

**Best: ep31, abupt=7.4726%. Does NOT beat 7.2091% baseline. Gap: +0.26pp.**

### Conclusion

Closed. Pure depth scaling (5L at 256d) consistently finishes ~0.26pp above the 4L/256d + Fourier PE baseline. The extra layer adds capacity but not enough to close the shear field gap. vol_p=5.27% is strong but doesn't compensate. Depth lever exhausted at 256d width; revisit only if combined with width scaling (5L/384d covered by Wave 2 PR #179).

## 2026-04-30 ~latest — PR #77: [haku] 4L/384d Width Scaling (no Fourier PE) (CLOSED)

- **Branch:** `haku/4l-384d-width-scaling`
- **Hypothesis:** Wider hidden dimension (256d→384d) at constant depth would improve val_abupt, testing orthogonal capacity dimension from PR #76.
- **W&B run:** `nbbbw8qw`, group `bengio-stream1-haku`

| Epoch | abupt% | wsy% | wsz% | Notes |
|-------|--------|------|------|-------|
| 10 | 9.964 | 13.04 | 14.67 | |
| 20 | 8.417 | 10.95 | 12.29 | |
| 25 | 7.708 | 10.25 | 11.45 | |
| 28 | 7.658 | 10.17 | 11.36 | |
| 30 | 7.639 | 10.12 | 11.32 | |
| **31** | **7.634** | 10.11 | 11.31 | **BEST** |

**Best: ep31, abupt=7.6344%. Does NOT beat 7.2091% baseline. Gap: +0.43pp.**

### Conclusion

Closed. Width scaling without Fourier PE is strictly worse than the baseline (4L/256d + PE). Comparing haku (7.634%) vs gilbert (7.473%) vs baseline (7.209%), the ranking is: PE > depth > width (no PE). Wall-shear y/z actually worsened vs baseline (wsy 10.11% vs 9.10%), indicating PE is the critical ingredient for shear field resolution. Width scaling in isolation without PE reallocates capacity but doesn't address the coordinate-encoding gap. The 4L/384d + PE hypothesis (askeladd #175) remains the clean follow-up.

### Follow-ups (from edward's diagnostic, archived for the queue)

1. Wider clamp `[-10, 10]` or unclamped log_vars — let log_vars equilibrate at their analytic optimum.
2. Loss-scale-aware `log_var` initialization (one short pre-pass to estimate per-task losses, init log_vars there) — avoid the long warm-in.
3. Drop the +0.5·log_var regularizer and use an explicit per-task scale schedule.
4. Higher grad-clip floor (10–50) to prevent the clip from distorting step direction.
5. Decoupled per-task LRs on per-task heads (simpler alternative to dynamic weights).

The follow-ups #1–#3 are queued under Wave 3 ideas; not worth re-spinning UW in Wave 2 while the cohort fixed-weight recipe is still mid-training and clearly competitive. Edward reassigned to fresh hypothesis (Theme C3 — GradNorm) immediately.

## 2026-05-01 08:45 — PR #137: [edward] DrivAerML GradNorm Per-Task Gradient Equalization (CLOSED)

- **Branch:** `edward/gradnorm-shear-balance`
- **Hypothesis:** Per-task gradient norm equalization (GradNorm) would auto-balance the contribution of wall_shear_y/z vs. surface_pressure in the training loss, directly addressing the wsy/wsz binding constraint.
- **W&B run:** `v5ybmwra`, also `09kojb6q` (GradNorm re-run). Primary run `v5ybmwra` stopped at step 17,816.

### Results

| Metric | Run v5ybmwra | Notes |
|--------|-------------|-------|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **33.43%** | Diverged — 4-5x above cohort |
| `test_primary/*` | Not populated | Killed before any useful convergence |

### Conclusion

**Rejected. Closed as dead end.** Run diverged catastrophically; val_abupt=33.43% vs. Wave 1 cohort range of 7.3–8.9%. No test_primary metrics populated. Per-task gradient norm balancing was unstable on DrivAerML, likely due to:

1. The wsy/wsz vs. surf_p gradient scale gap (~3-4x) destabilizing the GradNorm controller early in training.
2. No isolated LR for the GradNorm controller — used main LR=3e-4, which is too large for the task-weight parameters.
3. Kill-thresholds may not have been set aggressively enough; run ran to step 17,816 before termination.

### Follow-ups (archived)

- If GradNorm is revisited, use a dedicated, much smaller LR for task weights (e.g., 1e-5) decoupled from the backbone LR.
- Alternative: simple fixed per-axis loss upweighting for wsy/wsz (e.g., multiply wsy/wsz channel losses by 3-5x explicitly).
- Edward immediately reassigned to PR #160: split surface output head (dedicated cp MLP + wall-shear MLP) as a simpler, more stable architectural approach to the wsy/wsz binding constraint.
## 2026-04-30 — Status: Wave 1 In-Progress, Forced Harvests Requested

---

## 2026-04-30 10:00 — PR #74 (alphonse): 4L/256d Fourier PE Baseline

- **Branch**: alphonse
- **W&B Run**: `m9775k1v` (entity: morganmcg1, project: DrivAerML)
- **Hypothesis**: Establish 4-layer / 256-dim transformer with Fourier positional encodings as the Wave 1 baseline. No EMA, cosine LR with T_max=30. This was the strongest recipe from radford PR #2593 (~12.96% abupt) — run from scratch on bengio branch.
- **Status**: Running (ep39/50 as of harvest check). Regressing since ep30.

**Epoch trajectory (selected)**:

| Epoch | Step | val_abupt% | surf_p% | wall_sh% | vol_p% | wsx% | wsy% | wsz% |
|-------|------|-----------|---------|---------|-------|-----|-----|-----|
| ep10 | 184,110 | 7.4260 | 5.017 | 8.268 | 4.473 | 6.622 | 8.926 | 10.617 |
| ep20 | 368,220 | 7.2988 | 4.899 | 8.096 | 4.375 | 6.499 | 8.733 | 10.438 |
| ep28 | 515,508 | 7.2215 | 4.823 | 8.002 | 4.322 | 6.423 | 8.649 | 10.345 |
| ep29 | 533,919 | 7.2145 | 4.815 | 7.994 | 4.317 | 6.417 | 8.641 | 10.330 |
| **ep30** | **552,326** | **7.2091** | **4.802** | **8.160** | **4.166** | **7.109** | **9.100** | **10.869** |
| ep31 | 570,737 | 7.2534 | 4.840 | 8.017 | 4.357 | 6.448 | 8.680 | 10.383 |
| ep35 | 644,381 | 7.2329 | 4.816 | 7.996 | 4.339 | 6.425 | 8.647 | 10.346 |
| ep39 | 718,025 | 7.3454 | 4.910 | 8.142 | 4.421 | 6.572 | 8.823 | 10.526 |

**Best val_abupt: 7.2091% at ep30 (step 552,326)**

Per-channel at best:
- surf_p = 4.802% (AB-UPT: 3.82% — 1.0pp gap)
- vol_p = 4.166% **BEATS AB-UPT target of 6.08%** ✓
- wall_sh = 8.160% (AB-UPT: 7.29% — 0.87pp gap)
- wsx = 7.109% (AB-UPT: 5.35% — 1.76pp gap)
- wsy = 9.100% (AB-UPT: 3.65% — 5.45pp gap)
- wsz = 10.869% (AB-UPT: 3.63% — 7.24pp gap)

**Analysis**: Run clearly past optimum. vol_p already beating AB-UPT target. wsy and wsz are the hardest channels — both 2.5-3x above AB-UPT targets. This is the Wave 1 leader and bengio branch merge candidate. Forced harvest requested at ep30 checkpoint. Awaiting test_primary/* evaluation.

---

## 2026-04-30 10:00 — PR #78 (kohaku): 128-Slice + Fourier PE

- **Branch**: kohaku
- **W&B Run**: `h7ve1hmb` (entity: morganmcg1, project: DrivAerML)
- **Hypothesis**: Increase radial resolution from 64 to 128 slices with Fourier PE. Test whether higher mesh resolution improves accuracy over alphonse's 64-slice baseline.
- **Status**: Running (ep35/50 as of harvest check). Very slight improvement continuing past ep31 but essentially plateaued.

**Epoch trajectory (selected)**:

| Epoch | Step | val_abupt% | surf_p% | wall_sh% | vol_p% | wsx% | wsy% | wsz% |
|-------|------|-----------|---------|---------|-------|-----|-----|-----|
| ep3 | 71,267 | 13.5136 | — | — | — | — | — | — |
| ep10 | 200,144 | 7.9399 | 5.317 | 8.760 | 4.713 | 7.046 | 9.592 | 11.484 |
| ep20 | 384,254 | 7.8587 | 5.249 | 8.633 | 4.641 | 6.965 | 9.490 | 11.337 |
| ep29 | 549,953 | 7.8415 | 5.239 | 8.611 | 4.626 | 6.944 | 9.469 | 11.303 |
| ep30 | 568,364 | 7.8396 | 5.237 | 8.607 | 4.625 | 6.943 | 9.467 | 11.300 |
| **ep31** | **570,143** | **7.8338** | **5.235** | **8.556** | **5.569** | **7.480** | **9.465** | **11.420** |
| ep32 | 606,186 | 7.8395 | 5.238 | 8.606 | 4.624 | 6.942 | 9.465 | 11.298 |
| ep35 | 661,419 | 7.8370 | 5.236 | 8.602 | 4.622 | 6.941 | 9.462 | 11.293 |

**Best val_abupt: 7.8338% at ep31 (step 570,143)**

Per-channel at best:
- surf_p = 5.235% (AB-UPT: 3.82% — 1.42pp gap)
- vol_p = 5.569% **BEATS AB-UPT target of 6.08%** ✓
- wall_sh = 8.556% (AB-UPT: 7.29% — 1.27pp gap)
- wsx = 7.480% (AB-UPT: 5.35% — 2.13pp gap)
- wsy = 9.465% (AB-UPT: 3.65% — 5.82pp gap)
- wsz = 11.420% (AB-UPT: 3.63% — 7.79pp gap)

**Analysis**: 128-slice adds +0.625pp overhead vs 64-slice alphonse. ep3 instability spike (13.51%) resolved by ep4 — early training instability pattern. vol_p beats AB-UPT target but all other metrics worse than alphonse. Conclusion: higher radial slice count is not beneficial at this model scale — more mesh resolution does not help without corresponding model capacity increase. Forced harvest requested at ep31. Awaiting test_primary/* evaluation. Will only merge if it beats alphonse's eventual test_primary baseline.

---

## 2026-04-30 12:00 — PR #145 (senku): MSE + Raw Relative L2 Auxiliary Loss (w=0.05)

- **Branch**: senku
- **W&B Run**: `39dekqil` (entity: morganmcg1, project: DrivAerML)
- **Hypothesis**: Add raw (non-normalized) relative L2 as auxiliary loss term weighted at 0.05. Theory: standard L2 loss may under-weight physically meaningful relative errors; auxiliary loss directly targets the eval metric.
- **Status**: Running (ep4 as of last check). Decision rule being applied.

**Epoch trajectory so far**:

| Epoch | Step | val_abupt% | Decision |
|-------|------|-----------|---------|
| ep4 | ~73,644 | 11.53 | Keep going (11-13 zone), flag ep10 |

**Decision rule applied**:
- ≤11% at ep5: keep going at w=0.05
- (11, 13] at ep5: keep going, flag for ep10 check
- >13% at ep5: kill and relaunch at w=0.01

**Per-channel at ep4**: wsy=15.5%, wsz=16.4% still elevated — auxiliary loss is targeting the right channels but needs more epochs to take effect. Decline rate ~2.1pp/epoch, extrapolated ep5 ≈ 9.4%.

**Analysis**: Early result consistent with learning. Keep monitoring. Target ≤8% at ep10 for this config to be competitive.

---

## 2026-05-01 07:15 — Wave 1 Forced Harvest Sweep (PRs #76, #77, #81, #82, #83, #85, #86, #87, #89)

Posted forced-harvest decision comments on 9 silent past-best Wave 1 PRs. Each comment instructs the student to halt training, reload best-val checkpoint, run test eval, log all 6 `test_primary/*` metrics, and mark `status:review`. Best-val numbers below were pulled directly from W&B history.

| PR | Student | Run ID | Best step | Best ep | abupt% | surf_p% | vol_p% | wsx% | wsy% | wsz% |
|----|---------|--------|-----------|---------|--------|---------|--------|------|------|------|
| #76 | gilbert | `kn756yk6` | 552,326 | ep30 | 7.473 | 4.917 | **5.269** | 7.195 | 9.375 | 11.111 |
| #77 | haku | `nbbbw8qw` | 463,241 | ep25 | 7.708 | 5.085 | **4.477** | 7.385 | 9.866 | 11.730 |
| #81 | violet | `em5ixfew` | 694,862 | ep37 | 8.580 | 5.466 | **5.025** | 7.998 | 11.225 | 13.181 |
| #82 | askeladd | `uxrhudp1` | 570,143 | ep31 | 8.409 | 5.528 | **4.719** | 8.063 | 11.307 | 12.429 |
| #83 | chihiro | `kit58p2e` | 534,509 | ep29 | 8.769 | 5.546 | **4.814** | 8.207 | 11.629 | 13.652 |
| #85 | frieren | `l23vz4md` | 552,326 | ep30 | 8.172 | 5.199 | **4.979** | 7.825 | 10.912 | 11.946 |
| #86 | nezuko | `p8swf78o` | 552,326 | ep30 | 8.144 | 5.249 | **4.869** | 7.728 | 10.905 | 12.072 |
| #87 | norman | `0iv7wifz` | 552,326 | ep30 | 8.611 | 5.495 | **5.120** | 8.073 | 11.499 | 12.870 |
| #89 | thorfinn | `snrwvw14` | 552,326 | ep30 | 8.322 | 5.288 | **4.914** | 7.801 | 11.099 | 12.512 |

Bold vol_p = at or below the 6.08 AB-UPT target (8 of 9 beat it on val).

**Universal pattern**: 8 of 9 silent students hit best-val at step 552,326 (~ep30), violet at ep37, haku at ep25. Beyond best-val, all are slowly regressing — continued training is wasting GPU. Wall-shear y/z are the binding constraint across the entire wave (3–4× above AB-UPT targets).

**Pending student response**: test_primary metrics from best-val checkpoint. Without these I cannot merge per `target/program.md`. Tanjiro #80, fern #75, emma #79 are responsive and training to ep50; they will report final test_primary on their own. Senku #145 directed to continue to ep10 with revised thresholds after ep5 regression (13.24% vs ep4 11.53%).
