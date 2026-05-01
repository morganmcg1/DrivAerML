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
