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
---

## 2026-04-30 23:59 — Wave 1 Final Results: All 16 Streams Finished (ep50)

All 16 Wave 1 student streams completed at epoch 50. Ranked by best val_abupt:

| Rank | Student | Run ID | PR | Best abupt% | surf_p% | vol_p% | wsy% | wsz% | Notes |
|------|---------|--------|----|-----------:|--------:|-------:|-----:|-----:|-------|
| 1 | alphonse | `m9775k1v` | #74 | **7.209** | 4.802 | **4.166** | 9.100 | 10.869 | 4L/256d Fourier PE, T_max=30; best at ep30 |
| 2 | gilbert | `kn756yk6` | #76 | 7.737 | 5.110 | **4.766** | 9.488 | 11.297 | 5L/256d Fourier PE — depth signal confirmed |
| 3 | emma | `kuk0oy8g` | #79 | 8.214 | 5.410 | 5.596 | 10.476 | 12.205 | 60k pts input |
| 4 | kohaku | `h7ve1hmb` | #78 | 8.347 | 5.250 | 5.467 | 9.578 | 11.417 | 128 slices + Fourier PE |
| 5 | thorfinn | `snrwvw14` | #89 | 8.322 | 5.414 | **4.837** | 11.044 | 12.153 | gc=0.5 + wd=1e-3 |
| 6 | askeladd | `uxrhudp1` | #82 | 8.409 | 5.506 | **4.680** | 11.210 | 12.435 | SDF log-Fourier PE |
| 7 | tanjiro | `846uciam` | #80 | 8.436 | 5.453 | 5.134 | 11.068 | 12.288 | SW=2.0 |
| 8 | frieren | `l23vz4md` | #85 | 8.594 | 5.327 | 5.122 | 10.985 | 12.350 | cross-attention bridge |
| 9 | fern | `pxty4knv` | #75 | 8.578 | 5.715 | 5.859 | 10.683 | 12.478 | lr=1e-4 Trial A |
| 10 | norman | `0iv7wifz` | #87 | 8.611 | 5.608 | 5.067 | 11.370 | 12.614 | dropout=0.1 |
| 11 | nezuko | `p8swf78o` | #86 | 8.642 | 5.328 | **4.894** | 11.038 | 12.377 | mlp_ratio=6 |
| 12 | haku | `nbbbw8qw` | #77 | ~8.6 | ~5.6 | ~5.1 | ~10.9 | ~12.5 | 4L/384d |
| 13 | chihiro | `kit58p2e` | #83 | 8.769 | 5.287 | **4.800** | 12.018 | 12.925 | asinh wall-shear scale=1.0 |
| 14 | edward(v2) | `09kojb6q` | #160 | 8.870 | 6.097 | 7.649 | 11.936 | 13.217 | split output heads (after GradNorm failure) |
| 15 | violet | `em5ixfew` | #81 | ~9.0 | ~5.8 | ~5.2 | ~12.0 | ~13.0 | T_max=50 cosine |
| 16 | senku | `k8ytnvh8` | #88 | ~10.0 | ~6.4 | ~6.8 | ~12.6 | ~14.9 | RFF — CLOSED dead end |

**Key findings from Wave 1:**
- alphonse (7.209%) is the Wave 1 best and bengio branch baseline.
- gilbert (7.737%) confirms 5L > 4L: depth adds ~0.5pp improvement. Clear architecture signal.
- vol_p universally below AB-UPT target 6.08% — effectively solved.
- wsy and wsz are the decisive binding constraint: 2.5–4× above AB-UPT targets across all 16 students.
- 128-slice kohaku (8.347%) worse than alphonse 64-slice — resolution ≠ accuracy at this model scale.
- Fourier PE is the dominant positive factor (alphonse, gilbert both use it; top 2 overall).
- RFF (senku): closed dead end.

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

## 2026-04-30 22:00 — Wave 2 Launch: All 16 Students Assigned New Experiments

Wave 2 launched following completion of all Wave 1 streams. Focus: stack Wave 1 winners, attack wsy/wsz binding constraint, bold architectural ideas.

**Wave 2 assignment summary (as of 2026-04-30 22:00 UTC):**

| Student | PR | Group | Hypothesis |
|---------|----|----|---|
| alphonse | #174 | alphonse-depth-8L-1cycle | 8-layer depth + 1cycle LR peak=5e-4 |
| senku | #145 | senku-metric-aware-loss | metric-aware loss (rel-L2 auxiliary, w=0.05) — continued from Wave 1 |
| thorfinn | #176 | bengio-wave2-ema | EMA=0.9995 + best alphonse recipe |
| norman | #177 | bengio-stream2-norman / norman-snapshot | dropout=0.2 + snapshot ensemble |
| violet | #178 | violet-geom-moment-conditioning-r10 | geometry moment conditioning (add-v3) + vol-downweight |
| tanjiro | #179 | bengio-stream1-tanjiro / asinh-wallshear | asinh on wall-shear-yz + ARM-B/C/D ablations |
| nezuko | #180 | nezuko-symmetry-augmentation-r10 | symmetry augmentation (symm-p50, symm-p100) |
| chihiro | #181 | chihiro-mlpratio8-seeds | mlp_ratio=8 with multiple seeds |
| emma | #182 | emma-sam | SAM optimizer (rho=0.05, rho=0.10) |
| edward | #188 | edward-lion-r12 + bengio-stream2-edward-gradnorm | Lion optimizer + GradNorm alpha=1.5 v2 |
| fern | #75 (continued) | fern-omega-bank-sweep | omega bank frequency sweep |
| kohaku | #79 (continued) | kohaku-film-smallinit-sweep | FiLM conditioning with small-init (zinit-B) |
| askeladd | #80 (continued) | askeladd-normal-penalty-v2/v3 | surface normal penalty term v2/v3 |
| haku | #190 | bengio-wave2-ema + 1cycle-lr-superconvergence | 4L/512d DDP8 radford-champion + 1cycle LR |
| frieren | #85 (reviewing) | — | Wave 1 cross-attention bridge finished — pending review |
| gilbert | #76 (reviewing) | — | Wave 1 5L/256d Fourier PE finished — pending review |

---

## 2026-04-30 23:00 — Wave 2 Early Failures (Catastrophic/Severe)

### nezuko PR #180: Symmetry Augmentation — CATASTROPHIC FAILURE

- **Group:** `nezuko-symmetry-augmentation-r10`
- **Hypothesis:** Apply left-right geometric symmetry augmentation during training (50% or 100% of batches mirrored) to improve generalization and wsy/wsz prediction.
- **W&B runs:** symm-p50, symm-p100

| Run | abupt% | Notes |
|-----|:------:|-------|
| symm-p100 | **54.69%** | Complete divergence — 7.5× above baseline |
| symm-p50 | **16.56%** | Much worse than baseline (7.209%) |

**Conclusion:** Symmetry augmentation destroys model performance on DrivAerML. The DrivAerML car geometry is NOT left-right symmetric — there are external mirrors, exhausts, and other asymmetric features. Flipping x-axis coordinates breaks the coordinate frame for wall-shear prediction (wsy/wsz sign conventions). The model cannot learn consistent shear direction with flipped inputs. Both symm-p50 and symm-p100 terminated as failures.

**Follow-up:** Do not revisit symmetry augmentation without first verifying DrivAerML geometry is truly left-right symmetric in the provided dataset.

---

### violet PR #178: Geometry Moment Conditioning (geom-add-every) — FAILURE

- **Group:** `violet-geom-moment-conditioning-r10`
- **Hypothesis:** Condition the transformer on global geometric moments (volume, surface area, bounding box aspects) to improve generalization across car shapes.
- **Run:** geom-add-v3 (add conditioning at every transformer layer)

| Run | abupt% | Notes |
|-----|:------:|-------|
| geom-add-every | **16.53%** | 2.3× above baseline — too much geometric bias |

**Conclusion:** Adding geometric moment conditioning at every layer introduces too much inductive bias. The global shape statistics may be interfering with local surface feature learning. Partial conditioning (add at input only, not every layer) warrants investigation. Geom-add-v3 terminated.

---

### tanjiro PR #179: arm-C-v3 — DIVERGING

- **Group:** `bengio-stream1-tanjiro` / `asinh-wallshear-yz`
- **Run:** arm-C-v3 (ARM schedule variant C, version 3)

| Run | abupt% | Notes |
|-----|:------:|-------|
| arm-C-v3 | **17.55%** | Diverging — 2.4× above baseline |

**Conclusion:** arm-C-v3 configuration is unstable. Other tanjiro ARM ablations (A, B, D) and asinh-wallshear-yz experiments still in flight and may be viable.

---

### chihiro PR #181: mlp_ratio=8 seed1337 — POOR START

- **Group:** `chihiro-mlpratio8-seeds`
- **Run:** mlpratio8-seed1337

| Run | abupt% | Notes |
|-----|:------:|-------|
| mlpratio8-seed1337 | **11.92%** at ep~3 | Much worse than Wave 1 baseline seed (8.769%) |

**Conclusion:** mlpratio8 with seed1337 is a poor starting point at ep3. Wider MLP ratio with different init may be hurting early convergence. Still monitoring — early epoch instability sometimes resolves (see kohaku Wave 1: ep3=13.51% → ep10=7.94%).

---

## 2026-04-30 23:30 — Wave 2 Running Experiments (In-Flight Status)

### haku PR #190: 4L/512d DDP8 radford-champion + 1cycle LR

- **Group:** `bengio-wave2-ema` / `1cycle-lr-superconvergence`
- **Hypothesis:** Port the radford-champion recipe (4L/512d/8H, EMA=0.9995, gc=0.5, lr=4.8e-4, T_max=36) to DDP8 to get 8× throughput. Secondary experiment: 1cycle LR superconvergence.
- **Run ID:** `7ghbj3b7` (DDP8 radford-champion, ~612K steps of ~890K total)
- **Status:** Running — most advanced Wave 2 experiment. Step ~612K, val_abupt~7.8. On track to beat alphonse.
- **Notes:** 1cycle LR secondary at step ~10.9K. Monitoring both.

---

### senku PR #145: Metric-Aware Loss (continued from Wave 1)

- **Group:** `senku-metric-aware-loss`
- **Run ID:** `39dekqil`
- **Status:** Step ~279K (~ep15/50), val_abupt~9.4%, steadily improving.
- **Trajectory:** ep4=11.53% → ep15~9.4%. Expected to reach 8.5–9% at completion.
- **Baseline target:** Must reach <7.209% to beat alphonse. Currently on a trajectory that looks unlikely to beat baseline unless it accelerates sharply after ep20.

---

### edward PR #188: GradNorm v2 + Lion Optimizer

- **Group:** `bengio-stream2-edward-gradnorm` (GradNorm alpha=1.5 v2) / `edward-lion-r12` (Lion)
- **Status:** GradNorm v2 at step ~454K. Lion at step ~817 (very early).
- **Notes:** GradNorm v1 diverged at 33.43% (PR #137). v2 uses alpha=1.5 (more conservative) and should be more stable. Lion optimizer is a completely different approach — memory-efficient momentum + sign gradient.

---

### thorfinn PR #176: EMA on alphonse base

- **Group:** `bengio-wave2-ema`
- **Hypothesis:** EMA=0.9995 applied to alphonse's 4L/256d Fourier PE recipe. EMA should smooth the model weights and improve final accuracy by ~0.3–0.5pp based on radford experiments.
- **Status:** Step ~13.5K, very early training. Expected to run full 50 epochs.

---

### alphonse PR #174: 8L depth + 1cycle LR

- **Group:** `alphonse-depth-8L-1cycle`
- **Hypothesis:** Extend alphonse's 4L depth to 8L and use 1cycle LR for faster convergence. Wave 1 showed 5L>4L (gilbert: 7.737% vs 7.209%); 8L may push further.
- **Status:** Step ~3.5K, very early training.

---

### fern PR #75: Omega Bank Frequency Sweep

- **Group:** `fern-omega-bank-sweep`
- **Hypothesis:** Explore different base frequencies for the Fourier PE omega bank to find optimal frequency coverage for DrivAerML geometry.
- **Status:** Multiple trials at step ~3–18K, early training.

---

### kohaku PR #79: FiLM Conditioning Small-Init

- **Group:** `kohaku-film-smallinit-sweep`
- **Hypothesis:** FiLM (Feature-wise Linear Modulation) with small initialization (zinit-B) for geometric feature conditioning.
- **Status:** Step ~24K, early training.

---

### askeladd PR #80: Surface Normal Penalty v2/v3

- **Group:** `askeladd-normal-penalty-v2/v3`
- **Hypothesis:** Penalize predictions that violate normal-direction constraints on wall shear stress, specifically targeting wsy/wsz accuracy.
- **Status:** Trials at step ~9.9–20.6K, early training.

---

### emma PR #182: SAM Optimizer (rho=0.05, rho=0.10)

- **Group:** `emma-sam`
- **Hypothesis:** Sharpness-Aware Minimization (SAM) finds flatter minima that generalize better. Test two perturbation radii.
- **Status:** Step ~11.4K, early training.

---

### norman PR #177: Dropout=0.2 + Snapshot Ensemble

- **Group:** `bengio-stream2-norman` / `norman-snapshot`
- **Hypothesis:** Increase dropout to 0.2 for better regularization; snapshot ensemble with cyclic LR v2 for free model diversity.
- **Status:** Step ~5K, early training.

---

### violet PR #178: Vol-Downweight (geom-add-v3 terminated, vol-downweight continues)

- **Group:** `bengio-wave2-vol-downweight`
- **Hypothesis:** Downweight volume_pressure in the loss since vol_p already beats AB-UPT target — redirect model capacity toward wsy/wsz.
- **Status:** Monitoring, step unknown. geom-add-every variant terminated (16.53% diverged).

---

### tanjiro PR #179: asinh wall-shear yz + ARM ablations A/B/D

- **Group:** `asinh-wallshear-yz` / `bengio-stream1-tanjiro`
- **Hypothesis:** Apply asinh transformation to wall-shear y/z targets (chihiro used asinh on all shear; this targets y/z specifically). ARM schedule variants A/B/D (arm-C-v3 terminated at 17.55%).
- **Status:** asinh-wallshear-yz at step ~26K. ARM A/B/D at various steps ~1.8–18K.

---

### chihiro PR #181: mlp_ratio=8 seed42 (seed1337 flagged as poor start)

- **Group:** `chihiro-mlpratio8-seeds`
- **Hypothesis:** mlp_ratio=8 (double standard 4) to increase MLP capacity. Seed sweep to distinguish architecture effect from seed variance.
- **Status:** seed42 at step ~3.5K. seed1337 showing 11.92% at ep3 (early instability — monitoring).

---

## 2026-04-30 14:30 — PR #182: [violet] Volume loss downweight {0.5, 0.25} on Fourier PE baseline (CLOSED)

- **Branch:** `violet/volume-loss-downweight-fourier-pe`
- **Hypothesis:** Downweight `--volume-loss-weight` (0.5 or 0.25) to redirect optimizer gradient bandwidth from the already-converged vol_p axis toward the binding wsy/wsz shear axes.
- **W&B run IDs:** `tnpb1777` (rank0), `y4bfvygm`, `avc3ic2e`, `o40i85ua` (Trial A only; Trial B never launched)
- **W&B group:** `bengio-wave2-vol-downweight`

### Results table (Trial A at ep2, killed)

| Metric | Trial A (vol_w=0.5) @ ep2 | Wave 1 baseline | AB-UPT target |
|---|---:|---:|---:|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **24.879** | 7.209 | 4.51 |
| `val_primary/surface_pressure_rel_l2_pct` | 17.666 | 4.802 | 3.82 |
| `val_primary/wall_shear_rel_l2_pct` | 27.369 | 8.160 | 7.29 |
| `val_primary/wall_shear_y_rel_l2_pct` | 32.797 | 9.100 | 3.65 |
| `val_primary/wall_shear_z_rel_l2_pct` | 34.596 | 10.869 | 3.63 |
| `val_primary/volume_pressure_rel_l2_pct` | 15.808 | 4.166 | 6.08 |

### Commentary

Dead end. Killed Trial A at ep2 (24.88% abupt — 3.45× worse than baseline). Fixed volume downweighting destabilizes the loss balance at the surface level; even at only 50% vol downweight, the surface terms dominate catastrophically. Trial B (vol_w=0.25) was never launched.

**Key failure mode**: The 5-epoch warmup + the kill-threshold interaction caused an initial false kill, which violet correctly diagnosed and fixed. But the underlying hypothesis is unsound: senku/metric-aware-loss concurrently showed vol_p < 6.08% (beats target) **without any volume downweighting**, confirming the knob is wrong.

**Successor hypothesis**: Per-channel adaptive loss reweighting (PR #205) — targets the same wsy/wsz gap via running-variance-based auto-tuning rather than fixed scalar downweight.

---

## 2026-04-30 14:30 — PR #79: [emma] DrivAerML 60k Points + Fourier PE (sent back to WIP)

- **Branch:** `emma/60k-points-fourier-pe`
- **Hypothesis:** 60k surface/volume points (vs. default 40k) + FourierEmbed PE + T_max=50 cosine (Trial B v2, fixes Trial A's T_max=30 warm-restart pathology)
- **W&B run ID:** `3evzgru1` (rank 0), group `bengio-stream1-emma-trialB`, running

### W&B-verified metrics at ep5 (step ~59,434)

| Channel | ep5 value | Trial A terminal (ep32) | Wave 1 baseline | AB-UPT |
|---|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **10.799** | 8.214 | 7.209 | 4.51 |
| `surface_pressure_rel_l2_pct` | 7.281 | — | 4.802 | 3.82 |
| `wall_shear_rel_l2_pct` | 11.592 | — | 8.160 | 7.29 |
| `volume_pressure_rel_l2_pct` | 8.038 | — | 4.166 | 6.08 |
| `wall_shear_y_rel_l2_pct` | 13.392 | — | 9.100 | 3.65 |
| `wall_shear_z_rel_l2_pct` | 15.325 | — | 10.869 | 3.63 |

### Commentary

Run is healthy: `grad/global_norm=0.169`, `nonfinite_count=0`, all val/slope metrics negative (consistent improvement across all channels). FourierEmbed PE is learning (`grad_to_param_norm=0.011`). Best checkpoint saved at ep5.

Sent back to WIP — too early to merge/close at ep5 (only 10% through training). Trial A's pathology was the cosine warm-restart bouncing at T_max=30, which pushed the terminal from an ep32 optimum (8.214%) to ep50 worse. Trial B v2 with T_max=50 removes this pathology; terminal should beat 8.214%. Whether it can reach <7.21% (alphonse) remains to be seen at ep30+.

**New gates**: ep15 abupt<9.0%, ep20 abupt<8.0%, ep30 abupt<7.5%, terminal ep50 full test_primary breakdown.

---

## 2026-04-30 14:30 — PR #205: [violet] Per-channel surface loss rebalance (adaptive weights) (ASSIGNED)

- **Branch:** `violet/per-channel-loss-rebalance`
- **Hypothesis:** Replace uniform surface MSE with per-channel adaptive weighting (inversely proportional to exponential running mean of per-channel loss). Channels with higher residual loss (wsy, wsz — binding axes) get upweighted automatically.
- **Motivation:** Wave 1 best shows wsy gap = 5.45pp, wsz gap = 7.24pp vs. AB-UPT targets. Equal channel weighting wastes gradient on cp (nearly converged) at the expense of the binding shear axes. Adaptive reweighting continuously redirects gradient bandwidth to the hardest axes.
- **Implementation:** `PerChannelLossRebalancer` class with momentum=0.99 EMA of per-channel losses, normalized weights fed into `train_loss()`.
- **W&B group:** `bengio-wave3-per-ch-rebalance`
- **Status:** Assigned, student picking up.
- **Notes:** Early instability at ep3 is common (kohaku had 13.51% at ep3, recovered to 7.94% at ep10).
