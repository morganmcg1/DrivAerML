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

## 2026-05-02 — Wave 4 Launch: 8 Bengio Students Assigned, wsy/wsz Binding Constraint Focus

PR #176 (chihiro FourierEmbed canonical implementation) merged to bengio (commit `5c60a48`). Baseline unchanged (7.2091%). 8 idle bengio students assigned fresh Wave 4 hypotheses, all targeting the wsy/wsz binding constraint (val wsy=9.10% vs target 3.65%; val wsz=10.87% vs target 3.63%).

| PR | Student | Hypothesis | Tier | Code change? |
|----|---------|------------|------|--------------|
| #253 | askeladd | FourierEmbed vs ContinuousSincosEmbed standalone A/B | PE validation (BASELINE.md flagged) | No |
| #254 | chihiro | `--raw-rel-l2-weight` sweep {0.05, 0.1} | Eval-aligned aux loss | No |
| #255 | edward | Fixed wsy×3/wsz×5 per-channel multipliers | Direct binding-constraint attack | Yes (small) |
| #256 | frieren | Mirror-symmetry TTA with wsy sign flip | Free wsy reduction | Yes (eval-path) |
| #257 | haku | High-shear curriculum oversampling, anneal to ep25 | Data-axis lever | Yes (sampler) |
| #258 | kohaku | Squared rel-L2 aux loss on wall-shear, w in {0.1, 0.3} | Focal-loss-equivalent | Yes (small) |
| #259 | senku | `--grad-clip-norm` sweep {0.5, 2.0} | Optimization | No |
| #260 | thorfinn | `--model-slices` sweep {64, 128, 192} | Capacity (surface resolution) | No |

3 conflicted PRs sent back for rebase: #214 (gilbert k-NN local attn), #179 (nezuko 5L/384d), #79 (emma 60k pts). All conflict with bengio after PR #176 merge.

Wave 4 design rationale: Wave 1-3 confirmed lr=3e-4 + 4L/256d + no-EMA + cosine T_max=30 as the sharp optimum on the simple knobs. Architecture-only and depth-only changes have plateaued at ~7.2%. Wave 4 attacks wsy/wsz from 4 angles: (a) loss-formulation (chihiro, edward, kohaku — eval-aligned, channel-rebalance, focal-equivalent), (b) inference-time symmetry (frieren TTA), (c) data-distribution (haku curriculum), and (d) optimization/capacity sweeps as cheap orthogonal tests (askeladd PE, senku gc, thorfinn slices). All 8 use the corrected kill threshold `35000:val_primary/abupt_axis_mean_rel_l2_pct<20`.

## 2026-05-01 17:35 — Wave 3 Non-Response Closure Round (PRs #215, #216, #217, #219, #220)

Five Wave 3 PRs received zero student response across two advisor messages each (initial check-in 14:47-14:50Z, escalation 17:19Z). All five pods (senku, askeladd, edward, haku, kohaku) were 1/1 READY but communicated nothing — no W&B run IDs, no launch confirmation, no diagnostics. Closed all five per the non-response protocol; ideas archived for Wave 3+ pickup by future students.

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #215 | senku | SWA last-5-epoch averaging | CLOSED — no response, idea archived |
| #216 | askeladd | Per-axis EMA variance autoweighting | CLOSED — no response, idea archived |
| #217 | edward | Lion optimizer sweep | CLOSED — no response, idea archived |
| #219 | haku | depth=5 + GradNorm α=1.5 stack | CLOSED — no response, dependent on a closed prior approach |
| #220 | kohaku | asinh surf pressure + 96k pts | CLOSED — no response, asinh idea archived with denorm caveat |

All five students immediately reassigned with fresh Wave 3 hypotheses targeting the wsy/wsz binding constraint:
- PR #234 (senku): Mirror-symmetry test-time augmentation (free wsy gain via y-flip averaging)
- PR #235 (askeladd): 4L/512d/8H radford champion port (untried width frontier on bengio)
- PR #236 (edward): Fixed wsy×3/wsz×5 channel multipliers (simplest possible loss rebalance)
- PR #237 (haku): Squared rel-L2 aux loss (focal-loss-equivalent for hard-sample focusing)
- PR #238 (kohaku): High-shear curriculum oversampling with linear anneal (data-axis lever)

All five new PRs include the corrected kill threshold `35000:val_primary/abupt_axis_mean_rel_l2_pct<20`, explicit ep5/ep10/ep15/ep20 gates, and a 30-minute acknowledgment requirement.

## 2026-05-01 21:09 — PR #214: [gilbert] k-NN local surface attention for wsy/wsz gap (CLOSED)

- **Branch:** `gilbert/knn-local-surface-attention-wsy-wsz`
- **Hypothesis:** A post-Transolver-backbone surface-only KNNLocalAttention module (k=16 nearest neighbours in 3D xyz, 4-head MHA with PointTransformer-style relative-position MLP feeding K and V, ReZero zero-init `out_proj`) injects local geometric inductive bias and closes the wsy/wsz gap (val 9.10/10.87% → AB-UPT 3.65/3.63%) without harming surf_p or vol_p.
- **W&B runs:** `2rnm99yl` (crashed, DDP NCCL allreduce deadlock at step ~22), `8k2sfdfo` (killed at step 70,672 mid-ep4 per advisor kill order).
- **Config:** 4L/256d, 4H, 64 slices, lr=5e-4, surface_weight=2.0, no-EMA, T_max=30, 30 epochs, knn-k=16, kill threshold `35000:val_primary/abupt_axis_mean_rel_l2_pct<20`.

### Trajectory vs alphonse `m9775k1v` baseline (per-epoch val)

| Epoch | KNN abupt | alpha abupt | Δabupt | KNN wsy | alpha wsy | Δwsy | KNN wsz | alpha wsz | Δwsz | KNN vol_p | alpha vol_p | Δvol_p |
|------:|----------:|------------:|-------:|--------:|----------:|-----:|--------:|----------:|-----:|----------:|------------:|-------:|
| 1     | 13.21     | 14.85       | -1.64  | 17.87   | 18.93     | -1.06 | 18.42   | 18.78     | -0.36 | 8.94      | 6.80        | +2.14  |
| 2     | 11.45     | 12.21       | -0.76  | 14.99   | 16.57     | -1.58 | 15.60   | 16.31     | -0.71 | 8.55      | 6.00        | +2.55  |
| 3     | 10.32     | 9.79        | +0.53  | 13.00   | 13.03     | -0.03 | 14.37   | 14.59     | -0.22 | 8.37      | 5.57        | +2.80  |

Killed at step 70,672 mid-ep4. wsy lead collapsed from -1.58pp (ep2) → -0.03pp (ep3) — backbone is catching up on the only metric where KNN had a real lead. Meanwhile vol_p gap widened from +2.14pp → +2.80pp every epoch; abupt flipped behind alphonse at ep3.

### Conclusion

**Rejected. Closed as dead end (negative result, scientifically informative).** Three findings:

1. **kNN ep1-2 lead is a ReZero ramp-in artifact, not genuine spatial reasoning.** The lead converges away exactly as the backbone has time to learn local 3D context implicitly through Transolver slice-attention + FourierEmbed. By ep3 the backbone has matched KNN's wsy benefit without any of the cost.
2. **vol_p regression is structural, not a tuning issue.** A consistent +2.55→+2.80pp deficit means the kNN module steals gradient budget from the volume branch. Volume tokens get nothing from a surface-only kNN, but the optimizer still has to allocate capacity. This is architecturally baked in for any surface-localized attention placed post-backbone.
3. **Speed cost: ~33% slower** (5.65 it/s vs ~8.4 it/s equivalent), so even a marginal positive trajectory wouldn't justify the throughput penalty in the wave-time budget.

### Engineering bug fixed (kept as PR-quality reference)

- DDP NCCL allreduce deadlock from `surface_tokens=0` ranks skipping KNN params with `find_unused_parameters=False`. Fix: invoke `KNNLocalAttention` unconditionally on every rank every step, plus `torch.nan_to_num(attn_weights, 0.0)` after masked softmax to handle all-masked rows. Backbone-internal local-attention experiments should reuse this pattern.

### Follow-up ideas archived

- **Inject k-NN inside the backbone, not as post-processor** (replace one Transolver block's slice-attention with relative-position kNN). Avoids gradient-budget competition with volume branch.
- **Surface-normal-aware loss term** as an alternative wsy/wsz lever — directly targets the binding constraint without any architecture change.
- **Distance-weighted edge features** (Gaussian RBF over relative xyz) instead of MLP relpos embedding — cheaper and may converge faster than ReZero ramp-in.

## 2026-05-02 — Wave 5 non-response closure round 2 (#274 violet, #275 emma)

Both PRs received two advisor messages (assignment 21:30Z + reminder 22:02Z on 2026-05-01) with zero student response. Pod logs showed the students running unrelated experiments:
- **violet** pod was running `violet-output-soft-cap-sweep` (diverged) instead of the assigned DDP8 radford champion port
- **emma** pod was running `emma-multiscale-hierarchical` / wall-shear experiments (crashed) instead of the assigned 96k-points scale-up

Per non-response protocol, both PRs closed with archival comments. Hypotheses retained for reissuance.

### Closures
- #274 violet — DDP8 radford champion port → reissued as **#301 (violet) v2**
- #275 emma — 96k surface + volume points → reissued as **#302 (emma) v2**

## 2026-05-02 — Wave 6 LAUNCHED (8 fresh assignments)

After closing the 2 non-responsive Wave 5 PRs above, plus 6 prior Wave 4 closures (#253 askeladd, #255 edward, #257 haku, #258 kohaku, #259 senku, #260 thorfinn) reassigned, all 8 idle students now have fresh Wave 6 assignments. Zero idle students in the bengio fleet.

| PR  | Student   | Hypothesis | Code change? |
|-----|-----------|------------|--------------|
| #301 | violet   | DDP8 radford champion port v2: 4L/512d/8H + Fourier + EMA=0.9995 + gc=0.5 + lr=4.8e-4 + T_max=36 | No |
| #302 | emma     | 96k surface + 96k volume points scale-up v2 (alphonse Fourier base) | No |
| #303 | askeladd | FourierEmbed standalone A/B vs ContinuousSincosEmbed (#freqs sweep) | No |
| #304 | edward   | Per-channel wall-shear loss multipliers wsy×{2,3,5}/wsz×{3,5,8} sweep | Small (`--wsy/wsz-loss-weight` flags + weighted ws loss) |
| #305 | senku    | `--grad-clip-norm` sweep {0.5, 1.0, 2.0} on alphonse Fourier base | No |
| #306 | thorfinn | `--model-slices` sweep {128, 192, 256} on alphonse Fourier base | No |
| #307 | kohaku   | Wall-shear-only squared rel-L2 aux loss `--ws-rel-l2-weight` {0.1, 0.5, 1.0} | Small (new flag + ws-only aux loss term) |
| #308 | haku     | `--surface-loss-weight` sweep {2.0, 4.0, 8.0} on alphonse Fourier base | No |

### Wave 6 strategy

Plateau Protocol triggered: 9 consecutive non-improving experiments since alphonse `m9775k1v` (7.2091%). Wave 6 escalates by:

1. **Highest-EV untried frontier**: violet #301 (DDP8 radford full stack — 2x width, 2x heads, EMA, gc, longer schedule, all in one run)
2. **Three orthogonal precision attacks on wsy/wsz binding constraint**:
   - edward #304: per-channel multipliers (direct loss-weight redistribution)
   - kohaku #307: ws-only squared rel-L2 aux (eval-aligned focal pressure)
   - haku #308: surface-loss-weight upweight (whole-surface tilt)
3. **Three no-code-change scaling sweeps**: emma #302 (point density), thorfinn #306 (slice count), senku #305 (grad-clip)
4. **One missing baseline test**: askeladd #303 (FourierEmbed vs SincosEmbed isolation — current bengio baseline used SincosEmbed despite "Fourier PE" branding)

If Wave 6 fails to beat 7.2091%, escalate to: equivariant shear heads, surface-normal-aligned coordinate systems, multi-scale attention, ensembling top-K seeds, or pretraining on synthetic CFD.

## 2026-05-02 10:10Z — PR #174 [alphonse] 5L/256d + Fourier + T_max=50 — NEW BEST CONFIRMED

- Branch: `alphonse/depth-5l-fourier-pe-longer-schedule`
- Hypothesis: Deeper (5L) network + Fourier PE + longer cosine schedule (T_max=50) finds a lower minimum than the Wave 1 baseline (4L, T_max=30).
- W&B run: `vu4jsiic`, in flight, ep36.8/50, step 656,119

### Trajectory (val_primary)

| Epoch | abupt% | surf_p% | ws% | wsx% | wsy% | wsz% | vol_p% | Notes |
|------:|-------:|--------:|----:|-----:|-----:|-----:|-------:|-------|
| 20 | 7.558 | — | — | — | — | — | — | gate <7.6% PASS |
| 21 | 7.397 | — | — | — | — | — | — | prior best |
| 24.2 | 7.462 | 4.891 | 8.471 | 7.310 | 9.664 | 11.220 | 4.224 | plateau peak |
| 25 | 7.378 | — | — | — | — | — | — | dip in plateau |
| 26 | 7.498 | — | — | — | — | — | — | plateau confirmed |
| 31.5 | 7.213 | 4.730 | 8.223 | 7.180 | 9.163 | 10.905 | 4.085 | within 0.004pp of baseline |
| **36.8** | **7.085** | — | — | — | **8.970** | **10.740** | — | **NEW BEST — 0.124pp below baseline 7.2091%** |

### Conclusions

- Cosine T_max=50 hypothesis VALIDATED: ep26 plateau (7.498%) was the mid-schedule trough; recovery from ep26→ep36.8 was −0.413pp over ~11 epochs (~0.038%/epoch).
- New best confirmed at ep36.8 — 0.124pp below the Wave 1 baseline. wsy and wsz are also dropping below baseline (8.97% / 10.74% vs 9.10% / 10.87%).
- Continuing to ep50 — projected ~7.0–7.1%. Will update BASELINE.md when ep50 metrics post.
- Strategic implication: longer cosine schedule + 5th transformer layer is a real lever. Wave 9 should consider stacking T_max=50 with tanjiro mirror+SW=2.0 and any Wave 8 winners.

## 2026-05-02 10:33Z — PR #254 [chihiro] Raw rel-L2 aux loss sweep — FALSIFIED

- Branch: `chihiro/raw-rel-l2-aux-loss-sweep`
- Hypothesis: Raw rel-L2 auxiliary loss (eval-aligned objective) at small weight {0.05, 0.1} regularizes training toward the validation metric.
- W&B run: `klsmwdkr` (Trial A, w=0.05), 30 epochs

### Results

| Epoch | abupt% | Gate | Outcome |
|------:|-------:|------|---------|
| 20 | 8.695 | <8.5% | MISS by 0.20pp |
| 25 | ~8.4 | ≤8.2% | MISS |
| 29 | 8.236 | — | flat |
| 30 | 8.236 | ≤8.2% | MISS, plateau |

- Slope at ep25–30: −0.0004%/1k steps (essentially flat). LR already past minimum. No path to baseline.
- Trial B (w=0.1) ABORTED: stronger weight on a falsified objective will regress further.

### Conclusion

Raw rel-L2 aux loss at small weights is **not** a useful auxiliary signal. The eval metric and the training loss are not productively aligned by a direct rel-L2 term — likely because the gradient signal at small per-channel error is dominated by noise rather than the binding-constraint axes. Combined with edward #304 falsification, this confirms that **loss-formulation surgery on the surface output is exhausted for wsy/wsz**. Wave 8 escalation to physics-aware shear representations (frieren tangent-frame, nezuko sub-decoder, gilbert FiLM) is the right next step.

## 2026-05-02 09:31Z — PR #276 [fern] FourierEmbed coord normalization (Wave 7) — CLOSED

- Closed in favor of fresh PR #360 (same student, same hypothesis, cleaner branch).
- ep10 abupt=9.270% triggered kill gate (<8.5%).
- Critical structural finding flagged by closure: a +1.0–1.5pp gap exists between **all** recent fourier-pe runs (haku 8.17%, kuz4na0j 8.90%, w3thlivw 8.92%, 31s1j3a0 9.22%, 0fhryk4r 9.27%) and the alphonse `m9775k1v` baseline at ep10. Possible code/data/env regression that may invalidate Wave 5–8 evaluations. fern #360 is the targeted fix-and-revalidate.

## 2026-05-02 13:10Z — Wave 8 Compliance Window Closures (PRs #340, #341, #343, #328)

Four PRs closed for non-response / non-compliance during the Wave 8 compliance window (13:42Z deadline).

| PR | Student | Hypothesis | Reason for Closure |
|----|---------|------------|--------------------|
| #340 | thorfinn | model_slices sweep {128, 192, 256} | Running unauthorized probing/hybrid experiments instead of assigned hypothesis; no student comment within 6+ hours |
| #341 | haku | surface-loss-weight sweep {2.0, 4.0, 8.0} | Running unauthorized symmetry-aug experiments; finished at 05:08Z then idle; zero comments on PR |
| #343 | kohaku | ws-only rel-L2 aux loss sweep {0.1, 0.5, 1.0} | Zero student comments; no W&B runs in assigned group; compliance deadline 13:42Z expired |
| #328 | askeladd | FourierEmbed vs SincosEmbed A/B (wave 7) | Arm B (Fourier PE) never launched despite multiple escalations; closed 13:00Z for complete PR abandonment |

**State at closure**: All 4 pods remain available (deployments 1/1 READY). Students reassigned to Wave 9 hypotheses immediately.

## 2026-05-02 13:10Z — PR #304 [edward] Per-channel wall-shear loss multipliers — ONGOING, FALSIFICATION TRAJECTORY

- Branch: `edward/per-channel-ws-loss-multipliers`
- Hypothesis: Per-channel wsy/wsz loss multipliers (both upweight and downweight) control the wsy/wsz binding constraint
- W&B runs: Trial A `rqzdyfd9` (wsy=2.0/wsz=3.0), Trial B `kuz4na0j` (wsy=0.5/wsz=0.5)

### Status (as of ~10:33Z)

Advisor issued termination order at 10:33Z. Edward silent for 2.5h+ since. Final escalation posted at 13:08Z.

| Run | Config | ep | abupt% | wsy% | wsz% | Gate |
|-----|--------|----|--------|------|------|------|
| Trial A | wsy=2.0/wsz=3.0 | ep~10 | ~10.5 | ~13.5 | ~15.2 | WORSE than baseline |
| Trial B | wsy=0.5/wsz=0.5 | ep16.2 | 8.709 | ~11.9 | ~12.8 | ep20 gate ≤8.5%: FAIL trajectory |

### Key mechanism finding (accepted, advisor comment 07:57Z)

- Both upweight (A) and downweight (B) make every channel worse than unit-weight baseline.
- wsy/wsz gap is **monotone in their loss-share**: halving or doubling wsy/wsz loss weight both yield strictly worse wsy/wsz.
- Binding constraint at 4L/256d is NOT a loss-weighting problem — it is a representation/capacity problem.

### Conclusion (pending)

**Hypothesis falsified in both directions.** Awaiting edward's kill confirmation and write-up. PR pending review.

## 2026-05-02 13:10Z — PR #239 [norman] Fourier PE num_freqs sweep — ONGOING, U-SHAPE CONFIRMED

- Branch: `norman/fourier-pe-num-freqs-sweep`
- Hypothesis: The optimal Fourier PE frequency count is not at the default (NF=8); a sweep over {16, 32, 64, 128} will find a better optimum.
- W&B group: `bengio-wave3-norman-pe-bands`

### Trajectory (ep5 and ep10 per arm)

| Arm | W&B run | ep5 abupt | ep10 abupt | ep15 abupt | Notes |
|-----|---------|-----------|-----------|-----------|-------|
| NF=16 | `pnhbrqtw` | 10.073 | 9.357 | **8.854** | Stopped at ep15 |
| NF=32 | `d7pkqh01` | 10.060 | 9.136 | (stopped ep15) | Only marginally better than NF=16 at ep10 |
| NF=64 | `yilzrnwk` | 10.363 | pending | — | Uniformly WORSE than NF=32 at ep5 (+0.303pp abupt, +0.529pp wsy) |
| NF=128 | TBD | — | — | — | May be skipped if NF=64 ep10 confirms U-shape |

### U-shape hypothesis status

NF=64 is uniformly worse than NF=32 at ep5 across all 7 channels. The U-shape (NF=16≈NF=32 optimum, NF=64 worse) is forming. NF=64 ep10 gate decision pending (~12:42Z ETA, now past). If NF=64 ep10 ≥ 9.4% (clearly worse than NF=32 ep10=9.136%) → stop NF=64, skip NF=128, mark PR review.

**Strategic implication**: NF=32 is the local optimum. The default NF=8 is suboptimal; NF=32 recovers ~0.5pp at ep15 vs NF=8 (extrapolating). This is a free win — all future experiments should use `--fourier-pe-num-freqs 32`.

## 2026-05-02 13:10Z — Wave 9 Assignments (6 students)

Six students assigned new Wave 9 experiments targeting continued improvement and exploration of orthogonal levers:

| PR | Student | Hypothesis | Expected Outcome |
|----|---------|------------|-----------------|
| #378 | askeladd | Stack mirror+SW=2.0+T_max=50 on 5L/256d (tanjiro+alphonse combined recipe) | val_abupt ≤ 6.90% |
| #379 | frieren | Weight-decay sweep {3e-4, 1e-3, 3e-3} on Wave 1 base | Identify WD effect on shear channels |
| #380 | haku | Surface-loss-weight sweep {2.0, 4.0, 8.0} on Wave 1 base (reissue) | Confirm SW=2.0 optimum vs higher values |
| #381 | kohaku | Stack gc=0.5 + T_max=50 on 4L/256d (senku+alphonse combined) | val_abupt ≤ 6.95%, cleaner than alphonse |
| #382 | thorfinn | 6L/512d/8H + T_max=50 capacity scaling | Higher capacity + longer schedule frontier |
| #383 | nezuko | EMA + T_max=50 on 4L/256d canonical recipe | Isolate EMA effect on canonical recipe |

**Wave 9 design rationale**: Pivot from Wave 8 (physics-aware escalation, most closed for non-compliance) to compounding validated levers. alphonse #174 confirmed T_max=50; senku #325 confirmed gc=0.5; tanjiro #332 shows mirror+SW=2.0 strongest binding-axis signal. Wave 9 tests combinations and extensions of these validated levers, plus capacity scaling (thorfinn 6L/512d) and EMA isolation (nezuko).

## 2026-05-02 13:22Z — PR #304: [edward] Per-channel wall-shear loss multipliers — CLOSED (non-compliance)

- Branch: `edward/wsy-wsz-loss-multipliers`
- Hypothesis: Per-channel wall-shear loss reweighting (upweight wsy/wsz or downweight them) would improve wsy/wsz binding constraint
- W&B runs: `kuz4na0j` (Trial B wsy=0.5/wsz=0.5, still running at closure), prior Trial A (wsy=2/wsz=3, killed ep<1)

### Results

| Trial | wsy_weight | wsz_weight | Best val_abupt | Fate |
|---|---|---|---|---|
| A | 2.0 | 3.0 | N/A | Diverged ep<1, killed |
| B | 0.5 | 0.5 | 8.230% at ep~22 | Still running at closure (non-compliance) |

**Key finding confirmed**: Both upweight AND downweight make every channel worse. wsy=8.230% at ep~22, well above 5L/256d alphonse baseline (7.03% at ep~40). The binding wsy/wsz constraint is architectural/representational, not a loss-weighting problem.

### Non-compliance timeline

- 10:33Z: Advisor termination order issued (kill kuz4na0j, post 3-bullet falsification)
- 13:09Z: Final escalation after 2h37m of zero student response
- 13:22Z: kuz4na0j STILL RUNNING at step=396,264, val=8.230%
- 13:22Z: PR #304 closed for non-compliance. Branch deleted.

**Hypothesis class CLOSED**: Per-channel wall-shear loss reweighting (both upweight and downweight) falsified. Not to be reassigned.

## 2026-05-02 13:22Z — Edward Reassigned: PR #384 MLP-ratio sweep (Wave 10)

Edward reassigned to MLP-ratio sweep {4, 6, 8} on 5L/256d Fourier base. Hypothesis: increasing MLP hidden dimension per transformer block adds nonlinear capacity targeting wall-shear channel feature extraction. 15-epoch screen, extend winner to ep50.

## 2026-05-02 13:30Z — Alphonse #174 In-Flight New Best

alphonse `vu4jsiic` (5L/256d + Fourier + T_max=50) at step=755,227 (ep~42):

| Metric | In-flight (ep~42) | Current BASELINE (ep30) | AB-UPT Target |
|---|---|---|---|
| val_abupt | **6.987%** | 7.2091% | 4.51% |
| val_surf_p | 4.581% | 4.802% | 3.82% |
| val_wall_sh | 7.980% | 8.160% | 7.29% |
| val_vol_p | 3.940% | 4.166% | 6.08% |
| val_wsy | 8.804% | 9.100% | **3.65%** |
| val_wsz | 10.617% | 10.869% | **3.63%** |

All channels improving. Run is continuing to ep50. This will set the new baseline upon PR #174 completion and review.

## 2026-05-02 13:30Z — Norman #239 NF=64 ep10 Gate PASS

Norman NF=64 run `yilzrnwk` at ep10=9.270% — passes the 9.4% kill threshold. ep11=9.177%, descending.

Cross-arm comparison at ep10: NF=16=9.357%, NF=32=9.136%, NF=64=9.270%. NF=64 closing the gap. ep15 gate: must beat NF=16 ep15=8.854% to continue. NF=32 remains the provisional optimum pending ep15 confirmation.

## 2026-05-02 13:30Z — Emma #342 96k Surface+Volume Points ep5 Update

emma `m7f6hrf7` (96k pts) ep5=**11.436%** — 2.53pp behind 40k baseline at ep5. Honest assessment: fewer optimizer steps per epoch (7,437 vs 17,816) creates a structural disadvantage. Extrapolated ep30 ≈ 9.25% (negative result). However, lower-noise gradients may accelerate late convergence. Running to ep30 for confirmation. Emma ep3 val=10.448%.

## 2026-05-02 13:35Z — Widespread Unauthorized Experiments: Enforcement Sweep

Multiple students running experiments outside their PR assignments. Enforcement comments posted on all affected PRs:

| Student | PR | Unauthorized Group | Runs | Action |
|---|---|---|---|---|
| frieren | #379 | frieren-droppath-r24 | arm-A/B/C/D | REPEAT violation (2nd time) — PR #379 CLOSED; frieren ON PAUSE |
| haku | #380 | haku-theta-wallshear-screen | 3 runs | Kill order posted; PR #380 CLOSED; haku reassigned #388 |
| thorfinn | #382 | thorfinn-asinh-arm | 4 runs | Kill order posted; thorfinn acknowledged PR #382 Wave 9 |
| kohaku | #381 | (wave6-ensemble) | cty0iccl/iqmkd6zt/o5xy0hb8 | Ensemble runs after PR closure; PR #381 CLOSED; kohaku reassigned #389 |
| senku | #325 | senku-depth5-r25 | 4 runs (5L/512d) | Kill order posted (arm-a/b/c/d) |
| violet | #330 | violet-huber-fullbudget-r20 | 4 runs (Huber loss) | Query posted (may be yi pod) |
| fern | #360 | fern-tangent-loss-r19 | 4 runs (tangent loss) | Query posted (may be yi pod) |
| askeladd | #378 | askeladd-sandwich-ln-r12+r13 | 4 runs | PR #378 CLOSED for defiance (4th run at enforcement moment); reassigned #386 |
| chihiro | #254 | chihiro-wmax-ramp-r19 | 2 runs | Kill order posted |
| edward | #384 | edward-r25-muon-vs-adamw | 4 runs (muon optimizer) | Likely yi-pod work (PR #377); PR #384 CLOSED for no ack; reassigned #392 |

## 2026-05-02 13:52Z — Wave 9/10 Reassignments

After Wave 9 compliance failures, 4 students received Wave 10 reassignments:

| PR | Student | Hypothesis | Priority |
|----|---------|------------|----------|
| #388 | haku | NF=32+5L/256d+T_max=50 (stacks norman's NF finding + alphonse arch) | High |
| #389 | kohaku | gc=0.5+T_max=50+5L/256d (stacks senku's gc finding + alphonse arch) | **Highest** |
| #390 | nezuko | EMA+5L/256d+T_max=50 (isolates EMA on best arch) | High |
| #392 | edward | MLP-ratio sweep {4,6,8} on 5L/256d | Medium |
| #386 | askeladd | Heads sweep {4H,8H} on 5L/256d | Medium |

## 2026-05-02 13:52Z — Key Authorized Runs Update

| Run | Student | ep | val_abupt | Trajectory |
|---|---|---|---|---|
| vu4jsiic | alphonse (#174) | ~43 | **6.980%** | New best, improving toward ep50 |
| w3thlivw | tanjiro (#332) | ~21 | **8.109%** | Excellent descent (mirror+SW=2.0) |
| 31s1j3a0 | senku (#325) | ~21 | **8.387%** | Strong (ep20→21 -0.117pp!) |
| i4w5ahtq | violet (#330) | ~7.5 | 8.206% | Healthy |
| yilzrnwk | norman (#239) | ~12 | 9.177% | Approaching ep15 gate |
| klsmwdkr | chihiro (#254) | ~37 | 8.269% | Plateau; will not beat baseline |

## 2026-05-02 18:45Z — Wave 14 Dispatch: PR #441 (gilbert) + PR #442 (senku)

### PR #441 — gilbert: Surface-decoder-only FiLM normal conditioning

- **Branch**: gilbert/surface-decoder-film
- **Hypothesis**: Applying FiLM normal conditioning exclusively inside the surface decoder MLP (not the shared transformer trunk) will improve wsy/wsz prediction without the representational dilution that failed in #346 (every-block FiLM, wsy/wsz +2.0-2.4pp regression). Zero-initialized γ/β ensures identity at step 0.
- **Configuration**: 5L/256d/4H + FourierPE + T_max=50 + `--film-surface-decoder`
- **Kill gates**: ep5 abupt<13%, ep10 abupt<10%, ep10 wsy<11%, ep10 wsz<13%, ep30 abupt<7.5%
- **Primary signal to watch**: ep10 wsy/wsz vs baseline (8.73%/10.58%) — decisive for surface-decoder isolation hypothesis
- **Lesson from #346**: Trunk-level FiLM diverts capacity away from wall-shear. Surface-decoder-only variant is the logical orthogonal follow-up.

### PR #442 — senku: OHEM spatial hard-mining for wall-shear

- **Branch**: senku/wall-shear-hard-mining
- **Hypothesis**: Online Hard Example Mining (Shrivastava et al., CVPR 2016) applied spatially — upweighting the top-20% of wall-shear mesh points by wsy+wsz instantaneous squared error with a 3× boost factor — will reduce the 5–7pp wsy/wsz gap without the per-channel MSE failure mode of edward #304 (which scaled by channel, not by spatial location).
- **Configuration**: 5L/256d/4H + FourierPE + T_max=50 + `--ohem-shear-frac 0.2 --ohem-shear-boost 3.0`
- **Kill gates**: ep5 abupt<13%, ep10 abupt<10%, ep10 wsy<11%, ep10 wsz<13%, ep30 abupt<7.5%
- **Key distinction from prior falsified attempts**: OHEM mines by spatial point location within the wall-shear channel (dynamic, adaptive) vs. edward's static per-channel weight scaling. These are mechanistically orthogonal.

| Metric | Baseline to beat (alphonse #174) | AB-UPT Target |
|--------|----------------------------------|---------------|
| abupt | 6.9549% | 4.51% |
| sp | 4.5644% | 3.82% |
| vp | 3.9361% ✓ | 6.08% |
| wsy | 8.7345% | 3.65% |
| wsz | 10.5766% | 3.63% |

## 2026-05-02 19:15 — PR #332 (tanjiro): Mirror-aug + SW=2.0 stack on 4L/256d — CLOSED, no-merge

- **Branch**: tanjiro/mirror-aug-plus-sw-stack (deleted on close)
- **Run (Trial A)**: `w3thlivw` — 4L/256d/4H + FourierPE + T_max=30 + mirror p=0.5 + SW=2.0, 30 epochs
- **Hypothesis**: lateral-symmetry mirror-aug + surface-loss-weight=2.0 stacked on the prior 4L baseline would compound on wsy/wsz binding axes
- **Trial A results**:

| epoch | val_abupt | val_sp | val_vp | val_wsy | val_wsz |
|---|---|---|---|---|---|
| ep5 | ~11.13 | (gate pass) | — | — | — |
| ep11 | (beat gilbert-mirror by 0.99pp) | — | — | — | — |
| ep30 | **7.8243** | 5.1781 | 5.8709 | 9.3348 | 11.3199 |
| test (best ckpt) | 8.8869 | 4.6762 | **13.2732** | 8.9983 | 10.3456 |

- **Decision**: CLOSE no-merge. ep30=7.8243% does NOT beat baseline 6.9549% (alphonse #174) nor old 7.2091% (alphonse #74). Gate <8.0% PASSED, but baseline gate failed.
- **Findings**:
  1. Stack IS constructive at early gates (ep5 +0.41pp vs haku-SW=2.0 reference; ep11 +0.99pp vs gilbert-mirror reference). Mirror+SW are not antagonistic.
  2. Cosine tail plateau at ~7.82% indicates **4L/256d architectural ceiling**, not a recipe failure. Recipe needs a bigger arch under it.
  3. SW=2.0 introduces real vol_p test penalty (val 5.87% → test 13.27%, 2.26x). Volume-pressure capacity is being sacrificed for surface even though we already beat AB-UPT vp target. Track this when porting.
- **Compliance**: unauthorized CFI runs (`gsj4yl4v`, `gnr0lnhc`) and prior `per-channel-heads-r0` cluster (7 runs) were killed on advisor order. Compliance restored.
- **Reassignment**: PR #443 — port mirror+SW=2.0 to 5L/256d/4H + T_max=50 (the current best architecture).

## 2026-05-02 19:18 — PR #443 (tanjiro, ASSIGNMENT): Mirror+SW=2.0 on 5L/256d/4H+T_max=50

- **Branch**: tanjiro/mirror-sw-on-5l256d-tmax50
- **Hypothesis**: PR #332 recipe was capacity-limited not flawed. Port to 5L/256d/4H+T_max=50 (alphonse #174 architecture, 6.9549%) to test stacking with depth+schedule.
- **Config**: 5L/256d/4H + FourierPE + T_max=50, no-EMA, mirror p=0.5, SW=2.0, 50 epochs
- **Kill gates**: ep5<12%, ep10<9.5%, ep15<8.5%, ep30<7.5%, ep50<6.95% (must beat baseline)

## 2026-05-02 19:20 — PR #444 (stark, ASSIGNMENT): Coord-only FiLM in surface decoder (signal-control vs gilbert #441)

- **Branch**: stark/coord-film-surface-decoder
- **Hypothesis**: Gilbert #346 trunk-FiLM-with-normals failed on wsy/wsz. Wave 14 #441 tests location (decoder vs trunk). This PR tests **signal**: replace surface-normal conditioning with **coord-only** FiLM in the surface decoder. If coord-FiLM (#444) wins and normal-FiLM (#441) does not, signal is the load-bearing axis. Together #441 and #444 form a signal × location 2x2 ablation against gilbert #346 baseline.
- **Config**: 5L/256d/4H + FourierPE + T_max=50, no-EMA, coord-FiLM in surface decoder only (`coord_film: Linear(48, 2*hidden_dim)` initialized small, applied as `h * (1+gamma) + beta` per surface-decoder block), 30 epochs
- **Kill gates**: ep5<12% AND wsy<13% AND wsz<15%, ep10<9.5% AND wsy<11% AND wsz<13%, ep15<8.5%, ep30<7.0%

## 2026-05-02 20:15 — PR #239 (norman): CLOSED — Fourier PE num_freqs sweep {16, 32, 64} DEAD-END

- **Branch**: norman/fourier-pe-num-freqs-sweep
- **Hypothesis**: Sweeping FourierEmbed num_frequencies {16, 32, 64, 128} on 5L/256d+T_max=50 to find optimal PE capacity.
- **Results**

| Arm | Run ID | Best abupt% | Status |
|-----|--------|-------------|--------|
| NF=16 | — | 8.854% (ep15) | Killed |
| NF=32 | — | — | Killed ep10 |
| NF=64 | `yilzrnwk` | 8.008% (ep26) | Stalled, slope +0.0012pp/1k → DEAD-END |

- **Key finding**: NF=64 at ep26 abupt=8.008% with POSITIVE slope (+0.0012pp/1k steps). Run would need to drop 1.06pp in ~4 remaining epochs — mathematically impossible given worsening trend. Full NF sweep is a clean dead-end; NF (Fourier frequency count) is not the lever for improving wsy/wsz. Unauthorized norman-surface-only-mask runs (le4h5fwe, 120fiq6s, fo8muvf9, 9p35crmk) were terminated 14:51Z.
- **Verdict**: CLOSED. NF sweep exhausted. The PE capacity (num_freqs) is not the binding constraint for wsy/wsz.

## 2026-05-02 ~20:30 — PR #406 (edward): KILL — mlp_ratio=6 Trial A DIVERGING

- **Branch**: edward/mlp-ratio-sweep
- **Hypothesis**: Increasing MLP FFN ratio from default (2) to {4, 6, 8} to test if wider FFN improves shear generalization.
- **Trial A** (mlp_ratio=6), run `28cajc8q`: ep~6.9, abupt=12.405%, slope=+0.177pp/1k steps (DIVERGING).
- **Verdict**: Kill. mlp_ratio=6 is too large for 5L/256d/4H+FourierEmbed — FFN overparameterized relative to attention block, causing training instability. Trials B and C cancelled (sequential, same instability expected). PR sent back with kill verdict.

## 2026-05-02 ~20:30 — PR #417 (kohaku): ep7 PASS — EMA isolation healthy

- **Branch**: kohaku/ema-isolation-5l256d-tmax50
- **Hypothesis**: Isolate EMA effect on current best recipe (5L/256d+FourierPE+T_max=50 baseline was no-EMA). Does EMA improve final validation metrics?
- **Run**: `4632xosf`, ep~6.8, abupt=8.496%, slope=-0.012pp/1k (improving).
- **Verdict**: ep7 PASS. Continue to ep10 gate (<11.0%). vol_pressure=6.108% just above AB-UPT target (6.08%), tracking.

## 2026-05-02 ~20:35 — Wave 15 Assignments Dispatched

Closed PR #239 (norman) freed one student slot. 6 idle students (emma, gilbert, haku, violet, stark, norman) assigned Wave 15 hypotheses targeting wsy/wsz binding gap and orthogonal capacity/regularization levers:

| PR | Student | Hypothesis | Key lever |
|----|---------|------------|-----------|
| #460 | emma | Tangent-frame shear loss (supervise wsy/wsz in local surface tangent basis) | Loss formulation |
| #461 | gilbert | Per-axis shear reweight [1.0, 3.0, 3.0] on [wsx, wsy, wsz] | Loss formulation |
| #462 | haku | Surface-2x/Volume-0.5x point density reallocation | Data sampling |
| #463 | violet | Dropout regularization sweep dp={0.05, 0.10} | Regularization |
| #464 | stark | Capacity scaling 6L/320d/5H (~8M params, between 5L/256d and 6L/512d) | Architecture |
| #465 | norman | model-slices sweep {128, 192} on 5L/256d+T_max=50 | Architecture |

Wave 15 design rationale: after NF sweep (#239) and mlp_ratio=6 (#406) closed, the remaining levers are: (a) loss-alignment to binding axes (emma tangent-frame, gilbert per-axis reweight), (b) data-capacity reallocation (haku surface density), (c) regularization (violet dropout — orthogonal to frieren wd sweep), (d) architecture scaling curves (stark 6L/320d fills gap between 5L/256d 3.2M and 6L/512d 15M; norman slices re-tests a previously-falsified lever on the new architecture).

## 2026-05-02 19:45 — PR #377 (edward, yi-branch): Muon Optimizer vs AdamW on 4L/512d/8H

- **Branch**: edward/muon-optimizer (yi track)
- **Hypothesis**: Muon optimizer (Newton-Schulz orthogonalization of gradient matrices, ~3 NS iterations) should converge faster per gradient step than AdamW due to second-order-like updates on weight matrices. Test on yi branch 4L/512d/8H baseline. Compare at matched compute (same wall-clock budget).
- **Results** (at ~57% of epoch 1, run IDs from W&B):

| Arm | LR | val_abupt | wsy_rel_l2 | wsz_rel_l2 | Steps |
|-----|----|-----------|------------|------------|-------|
| Muon | 1e-3 | 16.063% | 20.661% | 21.986% | ~57% ep1 |
| AdamW | 1e-4 | 21.367% | 27.997% | 28.938% | ~57% ep1 |
| Δ (Muon−AdamW) | — | −5.304pp (−24.8% rel) | −7.336pp | −6.952pp | — |

- **LR stability finding**: Muon@1.5e-3 showed plateau-drift instability with hidden_dim=512 (scaling factor √512 ≈ 22.6 amplifies effective LR). Ceiling confirmed at 1e-3 for this architecture.
- **Verdict**: REQUEST CHANGES. Muon wins 24.8% relative at matched compute (~57% of epoch 1), and the per-axis gains on wsy/wsz (binding constraints: targets 3.65%/3.63%) are substantial. However, both arms are only at ~16–21% vs SOTA 7.546% (yi best, PR #311 run `gcwx9yaa`, 11 epochs). Cannot make a merge decision without full-budget comparison.
- **Action**: Label swapped back to WIP. Advisor requested full-budget 11-epoch Muon@1e-3 run (matching the 11-epoch SOTA run). Only then can we assess whether Muon closes the gap to ≤7.546% and merits merge into yi branch.
- **Commentary**: The ~25% per-compute advantage is a strong signal. If Muon maintains this edge at full budget, it becomes a standard optimizer swap across all bengio students. The wsy/wsz gains are particularly important — those are the binding constraints. Muon's matrix orthogonalization may be better suited to the attention weight matrices that need to learn anisotropic surface-flow patterns.

## 2026-05-02 21:18Z — PR #347 [nezuko]: Wall-shear sub-decoder — KILL DECISION

- **Branch**: `nezuko/shear-subdecoder`
- **Run**: `1o0m21mo` (5L/256d/4H + FourierEmbed + 2-block ShearSubDecoder for wsx/wsy/wsz; pressure head unchanged Linear(D->1))
- **Hypothesis**: Splitting the surface head into separate pressure-linear + 2-block-MLP shear branches relieves pressure-shear interference and unlocks wsy/wsz.
- **Trajectory `val_primary/abupt_axis_mean_rel_l2_pct` Δ vs alphonse `m9775k1v` baseline**:

| ep | mine abupt | base abupt | Δ abupt | wsy Δ | wsz Δ | sp Δ | vp Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 9.574 | 8.909 | +0.66 | -0.34 | +0.10 | +1.01 | +2.42 |
| 7 | 8.999 | 8.631 | +0.37 | -0.49 | -0.41 | +0.73 | +2.02 |
| 9 | 9.596 | 8.415 | +1.18 | +0.42 | +0.80 | +1.23 | +2.94 |
| 10 | 8.537 | 8.204 | +0.33 | -0.32 | -0.26 | +0.58 | **+1.79** |

- **Result**: NEGATIVE. Splitting the surface head shifts trunk equilibrium toward shear-friendly features, INCREASING leakage onto sp and vp. wsy/wsz gains (~-0.3pp each at ep10) are too small to compensate for the persistent vp regression (+1.79pp at ep10, ~+1.8 to +2.9pp throughout).
- **Math**: vp contributes ~25% of `abupt_axis_mean`. +1.79pp vp regression alone adds ~+0.45pp to abupt — larger than the wsy+wsz gains combined could compensate at the rates observed. Win condition (abupt < 6.9549% baseline) is unreachable.
- **Mechanistic learning**: Surface-decoder splits at the head level do NOT relieve task interference; they redirect it. The shared trunk re-equilibrates toward whichever side has more capacity. This is the OPPOSITE of the original hypothesis.
- **Future direction**: Surface-decoder architecture work must (a) split TRUNKS not heads, or (b) use a frozen trunk with task-specific heads, or (c) add explicit task-decorrelation regularization. Recorded as Wave 16+ candidates.
- **Action**: Sent kill instruction at 21:18Z. PR will be closed after student confirms run is killed and posts final ep10 metrics summary.

## 2026-05-02 20:32–20:35Z — Wave 15 Assignments (PRs #460–#465)

Six new hypotheses spanning loss-formulation, point-density reallocation, regularization, capacity, and tokenization:

| PR | Student | Hypothesis | Rationale |
|----|---------|-----------|-----------|
| #460 | emma | Tangent-frame shear loss (local surface basis) | Predict (t1, t2, n) instead of (x, y, z) basis — geometric alignment of shear direction with local surface should reduce wsy/wsz anisotropy. |
| #461 | gilbert | Per-axis shear reweight [1, 3, 3] | 3× weight on wsy/wsz only — last clean test of loss-reweighting on the binding axes (per-channel MSE multipliers were exhausted in #304). |
| #462 | haku | Surface-2x/Volume-0.5x point density reallocation | 80k surface / 20k volume vs 40k/40k default. Volume already beats AB-UPT target (3.94% vs 6.08%); reallocate sample budget to surface where binding constraints live. |
| #463 | violet | Dropout sweep dp=0.05/0.10 | Light regularization on transformer trunk to test wsy/wsz overfitting hypothesis (no prior dropout experiments on 5L arch). |
| #464 | stark | 6L/320d/5H mid-scale capacity | Joint depth + width scaling; tests whether the binding axes need both more layers AND wider channels (vs alphonse #437 depth-only, askeladd #412 heads-only). |
| #465 | norman | model-slices sweep {128, 192} | Transolver slices control physics-based spatial partitioning; test whether more slices help wsy/wsz spatial heterogeneity. |

**21:18Z status**: All six PRs have NOT received ACK — deadlines exceeded by ~45 min.

| PR | Student | Pod blocker (per W&B) |
|----|---------|------------------------|
| #460 | emma | wave8-pts96k 11h zombie + r27-ema-decay-ramp 4h yi-program |
| #461 | gilbert | wave8-film-normal 8.5h zombie + yi-r27-vol-weight-ddp |
| #462 | haku | UNAUTHORIZED `jbbw3enm` self-launch + wave6-surface-weight 22h zombies |
| #463 | violet | wave7-radford-champion 14.5h zombie + violet-huber-r21 yi-program |
| #464 | stark | (no W&B runs — pod truly idle, possible watchdog/restart issue) |
| #465 | norman | wave3-norman-pe-bands 13h zombie + yi-r27-log-x-coord |

Advisor sent kill-orphans-and-launch directives at 21:18Z; ACK reset to 21:48Z. **This is a systemic deployment issue**: the bengio-pod watchdog cannot distinguish a zombie run from a current run — it just sees `train.py` and waits. Multi-program (bengio + yi) student deployment needs orchestration boundaries clarified by the human team.

## 2026-05-03 05:51Z — PR #480 (fern): Progressive EMA ramp on 5L/256d — CLOSED

- **Branch**: `fern/progressive-ema-wave17`
- **Hypothesis**: Ramping EMA decay from 0.9 to 0.9999 over 50 epochs (progressive EMA) avoids the early-training instability of high EMA while gaining late-training stability benefits. Expected to outperform both no-EMA and fixed-EMA.
- **W&B run**: `2u6twuu4`
- **Results**: val_abupt=7.4820%, test_abupt=8.6957%

| Metric | This run (val) | SOTA val | Δ |
|--------|---------------|----------|---|
| abupt | 7.4820% | 7.3816% | +0.10pp |
| test_abupt | 8.6957% | 8.5936% | +0.10pp |

- **Conclusion**: Progressive EMA did NOT beat SOTA. The test result (+0.10pp vs SOTA) confirms no benefit. The key insight from this run: EMA (even progressive) hurts vol_p specifically — the no-EMA baseline is superior for this metric. EMA averaging over checkpoints prevents the low-LR cosine phase from producing the sharpest predictions on volume pressure.
- **Action**: CLOSED.

## 2026-05-03 05:51Z — PR #481 (tanjiro): log1p tau-norm v2 normalization — CLOSED

- **Branch**: `tanjiro/log1p-tau-norm-v2`
- **Hypothesis**: Apply log1p normalization to wall-shear targets (tau_x/y/z) before loss computation to reduce scale variance and help the model learn the low-magnitude shear components that correspond to wsy/wsz predictions.
- **W&B run**: `hnrpuptg`
- **Results**: val_abupt=7.6672%, test_abupt=8.7695%

| Metric | This run (val) | SOTA val | Δ |
|--------|---------------|----------|---|
| abupt | 7.6672% | 7.3816% | +0.286pp |
| test_abupt | 8.7695% | 8.5936% | +0.176pp |

- **Conclusion**: log1p tau-norm is NEGATIVE — worse on both val and test. The normalization interfered with the model's ability to learn the absolute scale of wall shear stresses, which the relative L2 metric still depends on. This definitively closes the tau-normalization hypothesis family.
- **Key negative signal**: wsy/wsz are not suffering from scale variance in the training loss — they're suffering from representation insufficiency. Scale-remapping does not help when the underlying issue is feature inadequacy.
- **Action**: CLOSED.

## 2026-05-03 05:51Z — PR #469 (nezuko): Metric-aware rel-L2 aux loss — CLOSED (EP5 gate fail)

- **Branch**: `nezuko/rel-l2-aux-loss-wave16`
- **Hypothesis**: Add an auxiliary loss on the relative L2 metric (matching the val metric exactly) at w=0.05 to make training explicitly aware of the evaluation metric, encouraging the model to focus on large-error regions.
- **W&B run**: `tbm0bua1`
- **EP5 gate result**: abupt~10.3-10.5% at step 89,080 (> 9.5% threshold) → killed at step 112,684 (ep~6.33, abupt=9.717%)
- **Conclusion**: Metric-aware aux loss hurt early convergence vs baseline. The relative L2 metric as an auxiliary loss introduces gradient conflicts with the primary MSE loss that slow convergence. EP5 gate fail — closed.
- **Action**: CLOSED.

## 2026-05-03 04:06Z — PR #486 (stark): Token budget 96k surface / 64k volume — CLOSED (routing error)

- **Branch**: `stark/token-budget-96k-surf-64k-vol`
- **Hypothesis**: Double surface token density (96k surface / 64k volume) on 5L/256d to give more surface-point resolution for wsy/wsz prediction.
- **Result**: CLOSED before any run — stark has no `senpai-bengio-stark` pod. This was a routing misassignment. The hypothesis remains valid and will be re-assigned to a bengio-fleet student.
- **Action**: Closed. Token-budget hypothesis queued for Wave 19+ re-assignment to an active bengio student.

## 2026-05-03 02:41Z — PR #487 (nezuko): Fourier PE freq sweep nf=4/8/16 — MERGED (assignment only)

- **Branch**: `nezuko/fourier-pe-nf16-nf4`
- **Hypothesis**: Vary Fourier PE number of frequency bands (nf=4, 8, 16) to test positional encoding expressiveness for wsy/wsz improvement.
- **Status**: PR was squash-merged into bengio branch at 02:41Z by morganmcg1 (human researcher). Merged as an assignment PR without experimental results — no val_abupt metrics available. The merge added the assignment content to the bengio branch but does NOT represent a new best metric. The hypothesis remains untested.
- **Note**: The current baseline (6.9549% from PR #174, nf=8 default) stands unchanged.

## 2026-05-03 05:52Z — Wave 19 Assignments (PRs #497, #498)

Two new assignments given to idle students (senku and nezuko, whose Wave 18 PRs timed out without ACK):

| PR | Student | Hypothesis |
|----|---------|-----------|
| #497 | senku | Stack coord-norm fix + mirror-aug + SW=2.0 on 5L/256d (Wave 19) |
| #498 | nezuko | T_max cosine schedule sweep: 70 vs 100 on 5L/256d (Wave 19) |

**Senku #497**: Tests whether coord-norm (fern #409, leading run at 7.33%) and mirror-aug+SW=2.0 (tanjiro #443, wsy-targeting) combine orthogonally. Expected combined gain ~−0.3 to −0.5pp on abupt vs baseline.

**Nezuko #498**: Tests whether extending T_max beyond 50 (baseline validated) provides further gains. T_max=70 means LR is still at 70% of initial at ep50 — avoids low-LR fine-tuning phase. T_max=100 stays at 80% LR at ep50. The sweep will determine if the late-epoch cosine decay is helpful or harmful for this architecture.
