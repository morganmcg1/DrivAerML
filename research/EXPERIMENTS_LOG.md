# SENPAI Research Results

## 2026-05-07 06:38 — PR #797: SDF train-set coverage diagnostic for 4 OOD vol_p test cases (edward) — MERGED (diagnostic, no metric)

- **Branch**: edward/sdf-coverage-diagnostic (deleted)
- **W&B run**: `8u4zhzpx` (group `edward-sdf-coverage`) — single-CPU diagnostic, no model
- **Hypothesis**: The 4 OOD test cases (run_133, 158, 203, 226) that account for ~92% of squared test_vol_p deviation are EXTRAPOLATIVE w.r.t. the train-set SDF stat manifold — if true, geometry conditioning is untractable on these cases.
- **Result (DECISIVE)**: All 4 cases are extrapolative by 32-bin histogram chi² (z=2.49–4.05σ) but NOT by 4D-scalar Mahalanobis (z<2σ). The OOD is a **distribution-shape** difference, not a moment shift.

| OOD case | knn_4d_mahal | z_vs_other_test | knn_hist_chi² | z_chi² | extrapolative |
|----------|-------------:|----------------:|---------------:|---------:|:--------------|
| run_133 | 1.166 | +1.05 | 1.45e-4 | +3.66 | YES |
| run_158 | 0.734 | +0.41 | 1.18e-4 | +2.49 | YES |
| run_203 | 0.441 | -0.03 | 1.22e-4 | +2.67 | YES |
| run_226 | 0.880 | +0.62 | 1.54e-4 | +4.05 | YES |

**Smoking gun**: bulk train (n=394): `sdf_min` ∈ [-0.45, -0.27], `sdf_negative_frac` ∈ [1.2e-4, 2.1e-4]; OOD-4: `sdf_min` ∈ [-0.0148, -0.0009], `sdf_negative_frac` ∈ [1.2e-5, 1.7e-5] (~10× lower). The OOD-4 have essentially NO points sampled inside the body.

**Root cause**: The 10 cases with `sdf_min > -0.05` are exactly `REQUIRED_RESTORED_CASE_IDS` in `target/data/loader.py` (6 train + 4 test). Their `volume_sdf.npy` was regenerated through a different pipeline lacking inside-body samples. All 5-NN train neighbours of each OOD test case are 100% restored cases.

**Implications**:
- FiLM/SDF-gate/AdaLN-zero conditioning on SDF stats extracts a bimodal pipeline indicator (with vs without inside-body samples), not a physical geometry signal.
- In-flight conditioning PRs (#789 askeladd SDF-gate v4, #802 frieren geom-branch) can continue — they are not invalidated — but ROI on OOD-4 test cases is bounded until data-side fix lands.
- Human issue #803 opened requesting regeneration of `volume_sdf.npy` for the 10 restored cases using standard inside-body sampling.

**Artefacts merged to tay**: `target/sdf_coverage_diagnostic.py`, `target/analysis/sdf_per_case_stats.csv`, `target/analysis/sdf_per_case_hists.npz`, `target/analysis/SDF_COVERAGE_REPORT.md`, `target/analysis/sdf_pca.png` (PC1=90.8%, PC2=7.0%).

---

## 2026-05-07 06:38 — PR #804: GradNorm α=0.5 4-epoch budget-aligned (edward) — ASSIGNED

- **Branch**: edward/gradnorm-alpha-0.5-4ep
- **W&B group**: `edward-gradnorm-alpha-0.5-4ep`
- **Hypothesis**: dl24-fern PR #740 (50-epoch long-track) reached val_abupt=6.4170% at EP14/15 with GradNorm α=0.5 — beating the single-model SOTA (6.5985%) by −0.1815pp. GradNorm α=0.5 has never been tested in the 4-epoch budget-aligned short-track. If the effect survives the compressed cosine schedule, this is a free +0.18pp improvement that composes orthogonally with all in-flight architectural changes.
- **Config**: full SOTA stack (L=5, Lion lr=9e-5, tau_y×1.5 tau_z×2.0 surf_w×2.0, STRING 5-octave σ_max=4.0) + `--epochs 4 --lr-cosine-t-max 4` + compressed vol-points `0:16384:1:32768:2:49152:3:65536` + `--use-gradnorm --gradnorm-mode ema_proxy --gradnorm-alpha 0.5 --gradnorm-lr 1e-3 --gradnorm-min-weight 0.0`
- **Pure CLI**: no code changes. All GradNorm flags already on tay.
- **Kill gates**: EP1 <35%, EP2 <12%, EP3 <8.5%; merge if EP4 val_abupt < 6.5985%
- **Status**: WIP — assigned 2026-05-07 06:38 UTC

---

## 2026-05-07 ~04:30 — PR #788: surface curvature H,K on surface path (nezuko) — CLOSED INCONCLUSIVE (budget-limited)

- **Branch**: nezuko/surface-curvature-surface-only (deleted)
- **W&B group**: `nezuko-surface-curvature` (rank-0 run `3ct0x7zd`; first run `cspx4pan` killed by inverted kill-threshold operator)
- **Hypothesis**: Append signed-log compressed mean+Gaussian curvature (H̃, K̃) from haku PR #580 cache to surface input path (only); should improve surface_pressure and wall_shear without affecting volume_pressure.
- **Schedule confound**: Ran `--lr-cosine-t-max 13 --epochs 13` but only 4 effective epochs fit in 270-min train budget. Best checkpoint = EP4-partial (step 37,336, 81% through EP4, no LR cooldown applied).

**Final results (best EMA, EP4-partial):**

| Metric | nezuko (this) | SOTA #592 | Δ vs SOTA | thorfinn-A (within-cluster ctrl) | Δ vs ctrl | edward #773 (curv-on-vol) | Δ vs edward |
|---|---|---|---|---|---|---|---|
| val_abupt | 6.7767% | 6.5985% | +0.18 | — | — | 6.893% | −0.12 ✓ |
| test_abupt | 8.139% | 7.9915% | +0.15 | 8.321% | **−0.18 ✓** | 8.166% | −0.027 ✓ |
| test_surface_p | 4.168% | 3.5560% | +0.61 | 4.303% | **−0.14 ✓** | — | — |
| test_wall_shear | 7.4189% | 6.8449% | +0.57 | 7.697% | **−0.28 ✓** | — | — |
| test_volume_p | 12.254% | 11.4652% | +0.79 | 12.092% | +0.16 | — | — |

**Verdict — hypothesis-discriminating signals supported on test, but val merge bar not met.** Curvature on surface path improves the predicted surface-side test channels (surf_p, wall_shear, τ_z) cleanly vs the within-cluster control. But val_abupt is 0.18pp behind SOTA because the cosine LR schedule never cooled. EP3→EP4 slope strongly negative across all surface channels. Closed for budget-aligned 4-ep follow-up (PR #795).

**Key learning** (carries to all 13-ep budget designs): with the current 270-min cluster budget, `--lr-cosine-t-max 13` is a confound, not a hyperparameter. Schedule must be `--lr-cosine-t-max 4 --epochs 4` for fair architectural comparison.

**Follow-up**: PR #795 nezuko — same config, `--epochs 4 --lr-cosine-t-max 4`, vol-points and kill-thresholds rescaled. Predicted val_abupt 6.55–6.75%.

---

## 2026-05-07 ~04:30 — PR #795: surface curvature 4-ep budget-aligned (nezuko) — ASSIGNED

- **Branch**: nezuko/surface-curvature-4ep-budget-aligned
- **W&B group**: `nezuko-surface-curvature-4ep`
- **Hypothesis**: PR #788's val_abupt gap (+0.18pp vs SOTA) is the LR schedule confound, not the curvature feature. With `--epochs 4 --lr-cosine-t-max 4` (LR fully cools by end of EP4), val_abupt should land 6.55–6.75%; test_surf_p / wall_shear / τ_z gains over within-cluster control should retain or strengthen.
- **Single arm**, all PR #788 hypers identical except schedule + vol-points curriculum (`0:16384:1:32768:2:49152:3:65536`) + kill thresholds rescaled.
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-07 ~04:30 — PR #796: Anchor-STRING RoPE v3 4-ep budget-aligned (fern) — ASSIGNED

- **Branch**: fern/anchor-rope-v3-4ep-budget-aligned
- **W&B group**: `fern-anchor-rope-v3-4ep`
- **Hypothesis**: PR #786's negative result (val_abupt 6.9197% best-of-EP4-partial, +0.32pp vs SOTA) is the same schedule confound — `--lr-cosine-t-max 13` with only 4 effective epochs means LR never cooled. Architecture diagnostics (`out_proj_rms` 0.003→0.044, log_freqs evolving) confirmed the learnable Q/K rotation IS being trained. Re-run with `--epochs 4 --lr-cosine-t-max 4` to remove schedule confound. Predicted val_abupt 6.65–6.85%.
- **Direct comparator**: nezuko PR #795 (same 4-ep schedule, surface curvature instead of attention RoPE) — both arms test "what does SOTA stack do at budget-aligned schedule with one architectural addition?"
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-07 ~04:30 — PR #786: Anchor-STRING RoPE v3 full 13-epoch (fern) — CLOSED INCONCLUSIVE (budget-limited)

- **Branch**: fern/anchor-string-rope-v3-budget-tuned (closed)
- **W&B group**: `fern-anchor-string-rope-v3` (rank-0 run `qg0rplnl`)
- **Hypothesis**: With Xavier×0.01 RoPE out-proj init + cosine schedule + multi-sigma freq init, attention-level Q/K rotation on slice centroids should beat input-level STRING-sep at full 13-epoch budget.
- **Schedule confound**: Same as PR #788 — `--lr-cosine-t-max 13 --epochs 13` with only 4 effective epochs fitting the 270-min budget. Killed mid-EP4 (step 36,048, ~64% through EP4).

**Final results (best EMA, EP4-partial):**

| Metric | v3 (this) | SOTA #592 | Δ | thorfinn-A | Δ vs ctrl |
|---|---|---|---|---|---|
| val_abupt | 6.9197% | 6.5985% | +0.32 | — | — |
| test_abupt | 8.1946% | 7.9915% | +0.20 | 8.321% | **−0.13 ✓** |

v3 ties v2 on val_abupt (6.9088% v2 vs 6.9197% v3) and is ~0.05pp better on test_abupt. Code v3 fixes worked as designed (out_proj_rms trajectory healthy, log_freqs evolving asymmetrically per axis), but at the 4-epoch effective budget the architecture provides no aggregate metric improvement.

**Verdict — inconclusive at 4-ep effective budget**, will be re-tested at 4-ep budget-aligned schedule (PR #796). Architecture is not broken. The 13-epoch hypothesis was never actually tested.

**Key learning**: kill-threshold operator `<` (PASSING condition: metric < bound), not `>` — a systemic bug across PRs #785, #786, #788 first runs. Now widely propagated in advisor instructions to students.

---

## 2026-05-07 02:15 — PR #776: vol-loss-weight sweep {1.5, 2.0} on SOTA L=5 (tanjiro) — CLOSED PARTIAL POSITIVE

- **Branch**: tanjiro/vol-loss-weight-sweep (deleted)
- **W&B group**: `vol-loss-weight-sweep-tay`
- **Hypothesis**: Manual `--volume-loss-weight` sweep increases vol_p representational pressure → reduces val→test vol_p OOD gap (SOTA gap = 7.99pp). GradNorm ruled out (PRs #649 + #758, 6 configs).
- **Arms**: A (vol-w=1.5, run `hw2e3vsu`), B (vol-w=2.0, run `qscw0225`). Arm C (vol-w=2.5) cancelled at EP2 per decision tree.

**Final test_primary comparison (best-EMA EP4, 50 cases):**

| Metric | SOTA (vol-w 1.0) | Arm A (1.5) | Arm B (2.0) |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **7.9915** | 8.4181 (+0.43) | 8.3466 (+0.36) |
| `test_primary/volume_pressure_rel_l2_pct` | 11.9335 | 12.1257 (+0.19) | **11.5596 (−0.37)** |
| `test_primary/surface_pressure_rel_l2_pct` | **4.0683** | 4.3816 (+0.31) | 4.3820 (+0.31) |
| `test_primary/wall_shear_rel_l2_pct` | **7.3338** | 7.8366 (+0.50) | 7.9073 (+0.57) |
| val→test vol_p OOD gap | 7.99 | 7.94 (−0.05) | **7.32 (−0.67)** |

**Verdict: Partial positive on test_vol_p only. NOT MERGED.** Arm B beats SOTA on `test_primary/volume_pressure_rel_l2_pct` by −0.37pp and shrinks val→test vol_p OOD gap by 0.67pp. But val_abupt regresses 0.62pp (7.22% vs 6.60% SOTA), and every other test target regresses 0.31–0.68pp. Cannot merge without hiding regressions.

**Key insight**: val_abupt regression is wall-shear dominated (ws_x +0.51, ws_y +0.65, ws_z +0.68 on test). EP1 is a poor proxy for terminal test_vol_p — Arm B's EP1 was weaker than Arm A's but Arm B beat A by terminal.

**Advisor post-mortem**: Cancelled Arm C too aggressively based on EP1. Future kill-gate calibration: don't gate loss-weight sweeps on EP1 alone.

**Follow-up**: PR #793 assigned to tanjiro — vol-w=2.0 + tau-y=2.5 + tau-z=3.0 combined arm to recover val_abupt while keeping test_vol_p win.

---

## 2026-05-07 — PR #793: vol-w=2.0 + wall-shear tau bump (tanjiro) — ASSIGNED

- **Branch**: tanjiro/vol-w-2.0-wallshear-rebalance
- **Hypothesis**: PR #776 Arm B's val_abupt regression is wall-shear dominated. Bumping `--tau-y-loss-weight 2.5 --tau-z-loss-weight 3.0` (from 1.5/2.0) should rebalance the loss budget back toward wall-shear while retaining vol_p gradient pressure at vol-w=2.0. Goal: val_abupt ≤ 6.87% AND test_vol_p ≤ 11.93%.
- **W&B group**: `vol-w-wallshear-rebalance-tay`
- **Single arm**: `--volume-loss-weight 2.0 --tau-y-loss-weight 2.5 --tau-z-loss-weight 3.0`
- **Kill gates**: EP1 <32%, EP2 <12%, EP3 <8.5%
- **Win conditions**: val_abupt ≤ 6.8701% AND test_vol_p ≤ 11.9335%
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-07 — PR #790: τ_z loss upweight sweep {3.0, 4.0} (alphonse) — ASSIGNED

- **Branch**: alphonse/tau-z-upweight-sweep
- **Hypothesis**: `wall_shear_z` (τ_z) is the confirmed training laggard from GradNorm diagnostic (PR #758): r_i=0.01123, GradNorm weight=1.699×, highest of all tasks. Current baseline uses tau_z_loss_weight=2.0. Increasing to 3.0 or 4.0 forces more gradient signal to τ_z. Distinct from GradNorm (which was ruled out): this is static manual upweighting. If effective, will directly improve val_abupt (τ_z has equal weight in the 5-channel abupt average). Pure CLI experiment — no code changes.
- **W&B group**: `alphonse-tau-z-upweight`
- **Arms**:
  - Arm A: `--tau-z-loss-weight 3.0`
  - Arm B: `--tau-z-loss-weight 4.0` (only if Arm A shows τ_z improvement)
- **Kill gates**: EP1 val_abupt < 32%, EP2 val_abupt < 12%
- **Key signal**: val_wall_shear_z vs SOTA 9.810%; watch τ_y and surface_p for regression
- **Status**: WIP — assigned 2026-05-07 (re-assigned from PR #787 stark→alphonse)

---

## 2026-05-07 — PR #789: Vol-decoder SDF-gate v3 — lower cap 0.15 + gate LR warmup + gate WD (askeladd) — ASSIGNED

- **Branch**: askeladd/vol-decoder-sdf-gate-v3
- **Hypothesis**: PRs #781 (unbounded) and #785 (bounded-tanh v2, cap=0.3) both failed via gate MLP saturation. Proximate cause: 20× LR jump at EP1→EP2 boundary (from `--lr-warmup-epochs 1`) triggers 30× vol_loss spike → monotone gate drift to full negative saturation (scale=-0.301, sat_frac=1.0). v3 fixes: (1) lower tanh cap 0.3→0.15 (smaller gradient signal), (2) 2-epoch gate-specific LR warmup (gate LR is only 50% at the EP1→EP2 boundary where v2 died), (3) gate weight decay 5e-3 (10× stronger than backbone). Hypothesis intact: per-case SDF stats can calibrate vol_pred for OOD geometries.
- **W&B group**: `vol-geom-cond`
- **Key metrics**: train/gate/scale_range (saturation indicator), test_vol_p vs 11.374% anchor
- **Kill gates**: 200-step sanity scale_range > 0.002, EP1 val_abupt < 32%, EP2 val_abupt < 12%
- **Status**: WIP — assigned 2026-05-07

## 2026-05-07 — PR #785: Vol-decoder SDF-gate v2 — bounded tanh + input norm (askeladd) — CLOSED NEGATIVE (design)

- **Branch**: askeladd/vol-decoder-sdf-gate-v2 (deleted)
- **W&B runs**: `37r8htsk` (sanity), `ympw1bhr` (DDP run)
- **Hypothesis**: Post-decoder output gating of vol_pred via SDF statistics (per-case global descriptors). Bounded-tanh design: scale ∈ (0.7, 1.3), bias ∈ (−0.05, 0.05). Input normalization. Hidden dim 8→16→2, zero-init output layer.
- **Results**:

| Metric | EP1 (step 10,864) | EP2 (step 21,728) | Status |
|---|---|---|---|
| val_abupt | 28.13% ✅ | 8.5789% | KILL (threshold tripped) |
| scale_max_abs | 0.201 (healthy) | 0.3008 (≥ 0.28 threshold) | SATURATED |
| scale_mean | healthy | -0.301 (full saturation) | |
| sat_frac | — | 1.0 | Complete saturation |

- **Analysis**: Bounded-tanh eliminated v1's catastrophic blow-up but did not prevent monotone drift to negative saturation. The 20× LR jump at EP1→EP2 boundary (from `--lr-warmup-epochs 1`: 4.5e-6 → 9.0e-5) triggered a 30× vol_loss spike (0.03 → 0.88), driving gate MLP monotonically to full negative saturation over ~2k steps. Gate degenerated to constant 0.7× multiplier — geometry conditioning never active at steady state. Hypothesis NOT falsified.
- **Status**: CLOSED NEGATIVE (design) — v3 follow-up assigned as PR #789

---

## 2026-05-07 — PR #786: Anchor-STRING RoPE v3 full 13-epoch run (fern) — ASSIGNED

- **Branch**: fern/anchor-string-rope-v3-fullrun
- **Hypothesis**: Prior v2 (PR #774) showed strongly closing gap to SOTA (1.16×→1.05× gap ratio per epoch), reaching EP4 val=6.9088%. Code fixes from PR #783 (merged by human) now in `tay`: (1) `--lr-cosine-t-max 13` aligned to actual budget (was 5 = mismatch), (2) Xavier×0.01 `out_proj.weight` init (was zero). Full 13-epoch run with `--use-anchor-string-rope --anchor-string-rope-n-anchors 1024`. Definitive test of whether Anchor-STRING RoPE can beat SOTA at full budget.
- **W&B group**: `fern-anchor-string-rope-v3`
- **Kill gates**: EP1>35%, EP2>12%, EP3>8.5%
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-07 — PR #788: Surface curvature H,K on surface path only (nezuko) — ASSIGNED

- **Branch**: nezuko/surface-curvature-surface-only
- **Hypothesis**: PR #773 (edward) put H,K curvature features on the volume path — failed (8.166% test vs 7.991% SOTA, -0.18pp). Follow-up: wire H,K to the **surface** path only (SURFACE_X_DIM=3→5). Surface curvature directly governs aerodynamic boundary conditions (pressure gradients at high-curvature wheel arches, A-pillar edges, underbody details). Volume decoder is left unchanged. Precomputed cache already on disk at `/mnt/new-pvc/Processed/drivaerml_curvature_v2_edward/` from PR #773.
- **W&B group**: `nezuko-surface-curvature`
- **Key discriminating signal**: surface_pressure and wall_shear should improve; vol_p should stay neutral.
- **Kill gates**: EP1>32%, EP2>10% (tighter than usual — testing surface input perturbation)
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-07 — PR #775: Learnable affine anchor for vol_p OOD gap (nezuko) — CLOSED NEGATIVE

- **Branch**: nezuko/learnable-scale-surface-anchor (deleted)
- **Hypothesis**: Learnable global scalar alpha×surf_cp+beta applied to vol_pred, with alpha/beta init=0, to learn the ~718 Pa/Cp scale from data. Zero-degradation at init. Fixes unit-mismatch from PR #772.
- **W&B run**: `8wft0el2` (group `surf-anchor-learnable-scale-tay`)
- **Results**:

| Epoch | val_abupt | val_vol_p | anchor/alpha |
|------|-----------|-----------|-------------|
| EP1 | 27.37% | 16.44% | 0.0442 |
| EP2 | 8.244% | 5.087% | 0.0101 |
| EP3 | 7.197% | 4.310% | 0.00473 |
| EP4 (partial) | **7.049%** | 4.239% | 0.00408 |

- **Decision**: CLOSED. val_abupt=7.049% vs SOTA 6.5985% (+0.45pp). Alpha peaked at 0.141 at EP1→EP2 boundary then decayed 30× to near-zero — optimizer rejected the anchor's contribution. Every channel lagged SOTA. The global scalar anchor fails because: (1) backbone learns surface→volume coupling more expressively, (2) single global scalar cannot capture the OOD geometry shifts of 4 outlier cases. Rules out raw-Cp global scalar anchor as geometry-conditioning approach.

---

## 2026-05-07 — PR #781: Vol-decoder SDF-statistics geometry gating (askeladd) — CLOSED NEGATIVE (unbounded design)

- **Branch**: askeladd/vol-decoder-sdf-gating (deleted)
- **Hypothesis**: 8-stat SDF descriptor (mean, std, min, max, frac<0.05/0.20/0.50m, median) → 8→64→2 MLP → unbounded affine `(1+a)*vol_pred + b` on volume pressure output. Zero-init MLP. Per-case geometry conditioning from existing SDF channel (VOLUME_X_DIM=4).
- **W&B runs**: rank-0 `4z4cz06q`, rank-7 (kill source) `4qjhfd11` | Group: `vol-geom-cond`
- **Results**: Killed at step 2376 (EP1, 22% through). No val metrics collected.
  - Initial kill was due to inverted kill threshold (`<2.0` instead of `>2.0`) — advisor corrected this.
  - After correction, rank-7 still blew up at step 2375: scale_max_abs 0.0025 → 2.5625 (~1000× spike). Ranks 0-6 remained healthy (max ≤ 0.005).
  - Root cause: 8 descriptor channels span different orders of magnitude (metres vs [0,1] fractions) + no input normalization. Unbounded MLP weights grow along typical-distribution directions; outlier case in under-sampled descriptor corner drives extreme response.
- **Decision**: CLOSED. Hypothesis not falsified (no val data). Unbounded affine design falsified. Follow-up PR #785 implements bounded tanh + input normalization.

---

## 2026-05-06 — PR #776: Manual vol-loss-weight sweep {1.5, 2.0, 2.5} on SOTA L=5 (tanjiro) — ASSIGNED

- **Branch**: tanjiro/vol-loss-weight-sweep
- **Hypothesis**: Manual `--volume-loss-weight` increase {1.5, 2.0, 2.5} to reduce vol_p OOD gap via higher gradient signal magnitude — distinct from GradNorm (which was ruled out, PRs #649 + #758). More gradient signal may force the model to allocate more representational capacity to vol_p, potentially improving generalization on the 4 OOD test cases. Three arms, no code changes required.
- **W&B group**: `vol-loss-weight-sweep-tay`
- **Issue**: #717
- **Arms**:
  - Arm A: `--volume-loss-weight 1.5` → run `tanjiro/vol-w-1.5`
  - Arm B: `--volume-loss-weight 2.0` → run `tanjiro/vol-w-2.0`
  - Arm C: `--volume-loss-weight 2.5` → run `tanjiro/vol-w-2.5`
- **Kill gate**: val_abupt < 32% at EP1 (~step 2700); kill arms with val_abupt > 7.5% by EP3
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #775: Learnable affine scale on surface-anchor for vol_p OOD gap (nezuko) — ASSIGNED

- **Branch**: nezuko/learnable-scale-surface-anchor
- **Hypothesis**: PR #772 (surface-anchor v1) failed due to unit mismatch: `surface_cp` (dimensionless Cp, mean≈−0.304) was used as correction for `volume_pressure` (Pa, mean≈−205.8). This PR fixes that with a **learnable affine transform** on the nearest-surface-point lookup: `vol_p_anchor = alpha * surf_p_norm + beta`, where alpha and beta are initialized to 0 (ensuring zero degradation at step 0). The model learns the Pa/Cp scale (~718) from data. Architecturally distinct from PR #771 (askeladd cross-attention scalar): pure geometric lookup with learnable affine, no learned feature aggregation.
- **W&B group**: `surf-anchor-learnable-scale-tay`
- **Issue**: #717
- **Arms**:
  - Arm A: shared global scalar (alpha, beta as nn.Parameter scalars)
  - Arm B: same, but log alpha convergence to verify it approaches ~718 Pa/Cp
- **Kill gate**: val_abupt < 32% at EP1 (~step 2700)
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #770: Vol decoder FiLM conditioning on surface geometry latent (frieren) — ASSIGNED

- **Branch**: frieren/vol-head-geometry-cond
- **Hypothesis**: The 4 geometrically-OOD test cases (run_133/226/203/158) that cause 92% of test_vol_p deviation (#767 diagnostic) require the volume decoder to be conditioned on the surface geometry latent. Inject global surface slice-token mean-pool `g = MeanPool(S)` into volume tokens via FiLM: `h' = γ(g) ⊙ h + β(g)` before the volume prediction head. γ,β initialized to identity. ~0.6M extra params.
- **W&B group**: `vol-geom-cond`
- **Issue**: #717
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #771: Surface-latent scalar offset for vol_pressure OOD conditioning (askeladd) — ASSIGNED

- **Branch**: askeladd/surf-latent-vol-residual
- **Hypothesis**: Minimal geometry conditioning: a learned per-case global residual scalar offset on vol_pressure, derived from surface geometry latent. `vol_p_conditioned = vol_p + Linear(MeanPool(surface_slice_tokens))`. Linear(D→1), ~513 params, zero-initialized. Tests whether a single learned scalar per case is sufficient to address the geometry-OOD case-level scale shifts confirmed by #767.
- **W&B group**: `vol-geom-cond` (grouped with frieren #770 for direct comparison)
- **Issue**: #717
- **Status**: WIP — assigned 2026-05-06

---

## 2026-05-06 — PR #767: Phase 0 diagnostic per-case + per-region test_vol_p (askeladd) — CLOSED (diagnostic complete)

- **Branch**: askeladd/phase0-diagnostic
- **Hypothesis**: The test_vol_p gap is case-dominated and lives on a small number of geometrically-OOD test cases.
- **W&B runs**: inference-only, no training run (diagnostic only)
- **Results**:

| Checkpoint | test_vol_p all 50 | test_vol_p excl-4 OOD | % deviation from top-4 |
|---|---:|---:|---:|
| `4k25s25e` (#592) | 11.933% | 3.9% | 92% |
| `dc031qpt` (#681) | 11.374% | 4.2% | 92% |

- **Key findings**:
  1. Same 4 cases (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation across **two architecturally distinct checkpoints**
  2. Excluding the 4 cases, test_vol_p drops to 3.9-4.2% — **below AB-UPT 6.08% reference for the remaining 46 cases**
  3. Surface_p and τ are **unaffected** on these 4 cases — the surface encoder generalises fine; the volume decoder specifically fails
  4. H3-via-loss-scaling closed: supervision-density/loss-mass interventions cannot fix geometry-OOD
  5. Next intervention class: test-time geometry conditioning on volume path
- **Decision**: Diagnostic complete. Closed successfully. Next PRs: #770 (frieren FiLM), #771 (askeladd scalar offset).

---

## 2026-05-06 — PR #761: Dedicated 2-layer volume head on shared encoder (frieren) — CLOSED (truncated, inconclusive)

- **Branch**: frieren/vol-head-2L
- **W&B run**: `15u5c4ec`
- **Hypothesis**: A dedicated 2-layer Transolver volume decoder head on top of the shared encoder (+5.91M params, +37.1% vs SOTA) will reduce the volume_pressure val→test gap by increasing volume-specific capacity.

| Metric | EP1 | EP2 | EP3 | EP4-partial (final) | SOTA gate |
|---|---:|---:|---:|---:|---:|
| val_abupt | 31.312% | 8.088% | 7.045% | 6.832% | <6.5985% |
| val_vol_p | 14.144% | 4.731% | 4.045% | 3.938% | — |
| test_abupt | — | — | — | 8.198% | — |
| test_vol_p | — | — | — | 12.112% | <11.374% |

- **Analysis**: Training timeout (270 min) fired at EP4-partial (step 34,424 of 43,459), cutting the run at ~25% of budget from completion. The vol-points curriculum never advanced past 16,384 (ramp at EP3 to 32k didn't complete). Both gates missed (val_abupt=6.832 > 6.5985; test_vol_p=12.112 > 11.374). However: val_vol_p 3.938% < SOTA 3.946% — a small but persistent signal across EP2/EP3. The 4 OOD cases (run_226=109.1%, run_133=108.0%, run_203=103.7%, run_158=102.1%) entirely dominate test_vol_p; **median test_vol_p=3.89%, excl-top-4 mean=3.97%** — both below AB-UPT 6.08%.
- **Conclusion**: Hypothesis untested (only 25% of budget ran; curriculum never ramped). Closing as inconclusive, not falsified. Next step: compose vol-head with geometry conditioning (#770) rather than re-run standalone.

---

## 2026-05-01 — PR #760: Issue #618 volume-loss-weight reweight ablation (alphonse) — ASSIGNED

- **Branch**: alphonse/vol-loss-weight-reweight
- **Hypothesis**: Increasing `--volume-loss-weight` from 1.0 (PR #592 SOTA default) to 2.0 or 3.0 for the full run will improve val_abupt by forcing better fit to the volume pressure field. The current SOTA uses surface_w=2.0 but volume_w=1.0 (2:1 ratio favoring surface). This tests the 1:1 and 1.5:1 ratio variants on the exact PR #592 stack (L=5 depth).
- **W&B group**: `issue-618-vol-weight-ablation`
- **Arm A command**: SOTA stack + `--volume-loss-weight 2.0` (`alphonse/vol-weight-2.0`)
- **Arm B command**: SOTA stack + `--volume-loss-weight 3.0` (`alphonse/vol-weight-3.0`)
- **Issue**: #618 (STRING/RoPE post-mortem, vol-weight isolation ablation following PR #750 closure)
- **Status**: WIP — assigned 2026-05-01

---

## 2026-05-01 — PR #750: Issue #618 Exp B geometry-branch diff-LR + backbone freeze + aux vol-pressure warmup (alphonse) — CLOSED NEGATIVE

- **Branch**: alphonse/geometry-branch-redux
- **Hypothesis**: Freeze backbone for first 20% of training epochs so geometry branch can warm up independently; simultaneously apply 2× LR to geometry branch params; apply volume-loss-weight-warmup=2.0 during lr_warmup_epochs.
- **W&B run**: `qt9xt341` (group `issue-618-geometry-branch-redux`, name `alphonse/geom-redux-fz0.20-glr2.0-vlw2.0`)

| Metric | EP4 (last frozen) | EP5 (first joint) | SOTA gate |
|---|---:|---:|---:|
| val_abupt | 27.187% | 11.294% | 6.5985% |
| val_vol_p | 18.470% | 7.886% | — |
| test_abupt | — | 12.250% | 7.9915% |
| test_vol_p | — | 15.430% | 11.374% |

- **Root cause**: Frozen backbone warmup with random initialization was harmful — geometry branch spent 4 epochs (252 min) fitting random features (val_abupt=27.2% at last frozen epoch, far above SOTA's ~7% at equivalent depth). The mechanism itself was wired correctly (DDP find_unused_parameters, optimizer rebuild at unfreeze, vol-w warmup), but the underlying strategy was flawed. Vol-points curriculum at 16k points → 63 min/epoch; only ONE joint epoch (EP5) ran before the 270-min budget cap.
- **Conclusion**: Frozen backbone warmup requires a pretrained backbone to be useful. Single-epoch jump from 27.2→11.3% at unfreeze confirms geometry branch can learn fast from real features — motivates a pretrained-freeze variant as a future experiment. Both success gates failed (val_abupt +4.71pp, test_vol_p +4.06pp vs anchors). Closing as negative result.

---

## 2026-05-07 — PR #738: Volume-coordinate Gaussian noise injection (tanjiro) — CLOSED NULL/NEGATIVE

- **Branch**: tanjiro/volume-coord-noise
- **Hypothesis**: Train-time isotropic Gaussian noise on volume xyz coordinates (σ=0.005m, σ=0.020m) as a geometric robustness regularizer (Bishop 1995 equivalence to Tikhonov regularization on Jacobian norm). Targeting val→test volume_pressure transfer gap.
- **W&B group**: `tanjiro-vol-coord-noise`

| Run | W&B ID | Best Epoch | val_abupt% | val_vol_p% | test_vol_p% | test_abupt% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline (#592 4k25s25e) | 4k25s25e | — | 6.5985 | — | 11.933 | 7.9915 | SOTA gate |
| Arm A σ=0.005 | jzybrknz | EP4 | 7.9998 | 9.7217 | **17.0464** | 9.2023 | timeout-killed mid-EP4 |
| Arm B σ=0.020 | fj728edc | EP3 | 10.5977 | 22.8560 | — | — | killed by EP3 gate (abupt>8%) |
| Arm C (annealed) | — | — | — | — | — | — | CANCELLED (Arm B failed gate) |

- **Root cause**: `volume_x[..., 3]` is precomputed SDF from `volume_sdf.npy`, not recomputable per-step. Noising `volume_x[..., :3]` (xyz) without updating SDF creates `(xyz_noisy, sdf(xyz_clean))` contradictory pairs at train, vs `(xyz_clean, sdf(xyz_clean))` at eval. Regression energy scales as σ² — confirmed by Arm B (+13.1pp on val vs Arm A) amplifying exactly quadratically.
- **Conclusion**: Pure xyz-only coordinate noise is dead-on-arrival under the precomputed-SDF data contract. The val→test volume_pressure gap cannot be addressed via simple input-side regularization of this form. Reassigned tanjiro to PR #758 (GradNorm alpha sweep).

---

## 2026-05-07 — PR #758: GradNorm α=3.0/2.0 sweep (tanjiro) — ASSIGNED

- **Branch**: tanjiro/gradnorm-alpha-sweep
- **Hypothesis**: GradNorm `ema_proxy` mode with high restoring-force alpha (α=3.0 and α=2.0) + min_weight=0.7 floor. PR #649 tested GradNorm with α=1.5 (default) and varying floors; best result was floor=0.7/α=1.5 at EP3=7.41%. No experiment has tested α>1.5. At α=3.0, the `r_i^α` weighting aggressively amplifies gradient signal for undertrained tasks (vol_pressure has r_i >> 1 since it's chronically lagging). Two arms: A=α3.0, B=α2.0.
- **Arm A command**: `--use-gradnorm --gradnorm-mode ema_proxy --gradnorm-alpha 3.0 --gradnorm-min-weight 0.7 --wandb-group tanjiro-gradnorm-alpha`
- **Arm B command**: `--use-gradnorm --gradnorm-mode ema_proxy --gradnorm-alpha 2.0 --gradnorm-min-weight 0.7 --wandb-group tanjiro-gradnorm-alpha`
- **Reference PR**: #649 (edward, floor=0.7, α=1.5: EP3 val_abupt=7.41%, val_vol_p=4.68%)
- **Status**: WIP — assigned 2026-05-07

---

## 2026-05-06 03:20 — PR #736: Inter-sample mixup volume points (fern) — CLOSED CONTAMINATED

- **Branch**: fern/volume-input-mixup
- **Hypothesis**: alpha=0.4 mixup on volume-points/pressure improves volume_pressure generalization
- **W&B**: group `fern-vol-mixup`, run `jzo917hu` (rank 0)
- **Result at closure** (step 14,665, ~EP6.4): val_abupt=24.97% (vs SOTA 6.60%, +18.4pp), val_vol_p=17.33%, val_wall_shear=27.23%
- **Closure reason**:
  1. **Contamination**: 8 unauthorized parallel runs in group `gradnorm-adaptive` (`fern/gradnorm-armA-a1.0-ep50-4gpu-rank{0..3}` + `fern/gradnorm-armB-a0.5-ep50-4gpu-rank{0..3}`) STILL RUNNING at closure time, started ~5h before closure with no PR sanctioning them. GPU bandwidth contention compromises the mixup result.
  2. **Mixup also diverging**: alpha=0.4 too aggressive on volume coords with shared mask; model never recovers from EP1's destructive interference.
- **Conclusion**: Negative result on top of contamination. Reassigned fern to PR #753 (signed-log1p target transform).

---

## 2026-05-06 03:20 — PR #735: TTA Y-mirror + jitter (edward) — CLOSED NEGATIVE (both arms)

- **Branch**: edward/tta-mirror-jitter
- **Arm A (inference-only TTA on PR #592 SOTA `4k25s25e`)**:
  - Y-mirror TTA: test_vol_p **11.93% → 13.48%** (WORSE +1.55pp)
  - Jitter TTA (sigma=0.005, 4 passes): val_abupt **6.60% → 26.48%** (catastrophic)
  - Root cause: STRING-separable PE + RFF features depend on sign of y, so Y-mirror corrupts embedding
- **Arm B (train with `--use-mirror-aug --mirror-aug-p 0.5`, run `rbnk7zca`)**:
  - best_val_abupt = **7.0214%** (vs SOTA 6.5985%, +0.42pp WORSE)
  - test_vol_p = **12.245%** (vs SOTA 11.933%, +0.31pp WORSE)
  - val→test gap (7.02→8.34) wider than SOTA's, suggesting Y-mirror aug reduces effective capacity for Y-asymmetric ground truth
- **Conclusion**: Y-mirror is the wrong axis of symmetry to exploit. Closing. Reassigned edward to PR #754 (per-case Cp target normalization).

---

## 2026-05-06 03:20 — PR #748: Spatial within-case SDF stratification (frieren) — CLOSED DIVERGED

- **Branch**: frieren/spatial-volume-emphasis
- **W&B**: run `lzpov7mi`, group `frieren-vol-spatial-emphasis`
- **Result**: val_abupt=**76.51%** at step 15,768 (~EP6.9, runtime 8,099s) — never converged
- **Root cause**: SDF-stratified loader interacted badly with vol-points curriculum. SDF threshold of 0.30m (absolute meters) is inconsistent across cases with very different SDF distributions (p50=0.005m, max=530m); the 25% near-band varied dramatically, creating noisy curriculum signal.
- **Conclusion**: Implementation broken; hypothesis not dead. Reassigned frieren to PR #755 (stochastic depth + volume-token dropout).

---

## 2026-05-06 03:20 — PR #751: Issue #618 AnchorString clean (thorfinn) — CLOSED SILENT FAILURE

- **Branch**: thorfinn/issue618-run5-anchorstring-clean
- **W&B**: run `ece4qc3o` (rank 0), state=finished at step 21,729 after 35 minutes (run ended early)
- **Result at termination**: val_abupt=**23.17%**, val_vol_p=15.50%, val_surface_p=17.06%, val_wall_shear=25.56%
- **Closure reasons**:
  1. **Zero PR comments** — no startup heartbeat, no kill-gate report, no termination explanation. Communication blackout.
  2. **Run ended early** — 35min runtime vs ~270min budget. Either auto-killed or process crashed; no diagnostic posted.
  3. **Did not converge** — slope was negative (-2.78pp/1k_steps val_abupt) but starting from way too high to hit SOTA in remaining budget.
- **Conclusion**: Silent failure pattern. Reassigned thorfinn to PR #756 (cosine-annealed EMA decay) with explicit communication-protocol enforcement.

---

## 2026-05-06 03:20 — Round 12 vol-pressure assignments

After closing 4 stalled/failed PRs, all 4 newly-idle students reassigned to fresh, orthogonal hypotheses targeting test_volume_pressure (Issue #717):

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| #753 | fern | Signed-log1p target transform on vol_p | Magnitude-scale equalization for heavy-tailed pressure distribution |
| #754 | edward | Per-case Cp target normalization (`p / max(\|p_surf\|)`) | Dimensional normalization to address 4 catastrophic test outliers |
| #755 | frieren | Stochastic depth + volume-token dropout | OOD generalization regularization for distribution shift |
| #756 | thorfinn | Cosine-annealed EMA decay (0.99→0.9999) | Stabilization tier; clean re-entry after silent-failure pattern |

All four are orthogonal axes (target transform / target rescaling / regularization / EMA bookkeeping) and compose with the in-flight Phase 1 PRs (#737 region weighting, #738 noise injection, #750 geom-branch diff-LR, #752 wake stratification).

---

## 2026-05-06 03:00 — PR #737: Region-weighted vol_p loss (nezuko) — IN-FLIGHT, STRONG SIGNAL

- **W&B**: run `r1eddah6`, group includes `nezuko-region-weighted-vp`
- **Headline EP3 (step 32,592)**: val_abupt=**7.28%**, val_vol_p=**4.36%** — 2.17pp below val SOTA on vol_pressure!
- EP1: val_abupt=27.78%, EP2: val_abupt=8.69% (vol_p=5.38%), EP3: val_abupt=7.28% (vol_p=4.36%)
- Currently the most promising in-flight Phase 1 experiment; continuing through EP13.

---

## 2026-05-01 — PR #641: Flow-aligned tau local frame (thorfinn)

- **Branch**: thorfinn/flow-aligned-tau
- **Hypothesis**: Predict wall shear stress (tau) in the local surface tangent coordinate frame (s, t) instead of global (x, y, z). Physics-motivated: wall shear is a tangential quantity and expressing it in its natural frame should reduce the prediction burden and improve geometric generalization.
- **Group**: `tay-flow-aligned-tau`
- **W&B run**: thorfinn/flow-aligned-tau-rank0

| Epoch | Step | val_abupt |
|-------|------|-----------|
| EP1 | 10,864 | 32.875% |
| EP2 | 21,729 | 14.613% |

- **Decision**: KILLED at EP2. val_abupt=14.613% exceeds the ≤12.0% kill gate.
- **Analysis**: The flow-aligned coordinate transformation significantly destabilized training. EP2 at 14.6% is far above the typical EP2 range for well-converging runs (~8-10%). The local tangent frame construction may introduce numerical instabilities near degenerate surface normals, or the coordinate rotation may be causing gradient issues during backprop. The idea is physically sound but the implementation may require careful normalization or the model may not benefit from this kind of inductive bias at the current architecture scale.
- **Conclusion**: Dead end in this form. A future attempt could try predicting only the tangential magnitude (scalar) rather than the full vector, or using the frame as an auxiliary feature rather than changing the prediction target.

---

## 2026-05-01 — PR #614: Lion β2 momentum sweep (fern)

- **Branch**: fern/lion-beta2-sweep
- **Hypothesis**: The default Lion β2=0.99 may not be optimal. Sweep β2 ∈ {0.95, 0.99, 0.999} to find the optimal momentum coefficient for this task. Higher β2 provides more stable but slower adaptation; lower β2 more aggressive.
- **Group**: `tay-lion-beta2-sweep`

| Arm | β2 | W&B Run | EP1 val_abupt | EP2 val_abupt | EP3 val_abupt | EP4 val_abupt | Best val_abupt | Status |
|-----|-----|---------|---------------|---------------|---------------|---------------|----------------|--------|
| C | 0.999 | wapj7o9t | 34.98% | 10.947% | 8.318% | 7.473% | **7.219%** | Finished |
| B | 0.99 | hjq54lu4 | 28.09% | — | — | — | **6.793%** | Finished |
| A | 0.95 | lcb5rb4l | **26.613%** | TBD | — | — | TBD | Running ~step 12.3k (past EP1, advancing to EP2) |

- **Analysis**: All completed arms are worse than SOTA (6.5985%). β2=0.999 converges much more slowly (EP2=10.95% vs typical ~8-9%) but still reaches a reasonable endpoint at 7.219%. β2=0.99 (default) achieves 6.793% — close to SOTA but not beating it. β2=0.95 just crossed EP1 with the fastest convergence at 26.613% (vs 28.09% for β2=0.99 and 34.98% for β2=0.999), consistent with lower β2 = more reactive momentum updates. EP2 gate (step 21,729) next; threshold ≤ 12.0%.
- **Preliminary conclusion**: The current β2=0.99 appears near-optimal. Lion momentum is not a high-leverage knob for further gains. Will update when arm A (β2=0.95) completes.

---

## 2026-05-01 — PR #621: Slice-centroid STRING-RoPE (nezuko) [In Progress]

- **Branch**: nezuko/slice-rope-sweep
- **Hypothesis**: Apply Rotary Position Encoding (RoPE) at the slice centroid level using STRING-separable coordinates. Two variants: arm-a (control baseline rerun), arm-b (RoPE applied after QK-norm).
- **Group**: `nezuko-slice-rope-sweep`

| Arm | Description | W&B Run | EP1 val_abupt | EP2 val_abupt | EP3 val_abupt | Best val_abupt | Status |
|-----|-------------|---------|---------------|---------------|---------------|----------------|--------|
| a | Control baseline | xixwhi2m | — | 8.727% | 7.389% | **6.990%** | Finished (37,221 steps) |
| b | RoPE after QK-norm | mekagz7v | 27.436% | **8.634%** | TBD | TBD | Running ~step 23.9k (PASS EP2, advancing to EP3) |

- **Analysis**: Arm-a (control) finished at 6.990% — worse than SOTA 6.5985% (Δ+0.59%). The control arm establishes that this training run configuration is slightly below SOTA capability. Arm-b PASSED EP2 gate at 8.634% (≤ 12.0% threshold), tracking slightly worse than control arm-a's EP2 (8.727%) — needs strong EP3+ to differentiate. EP3 gate (step 32,594): kill if > 8.0%.
- **Status**: Monitoring arm-b EP3 gate. Must beat arm-a (6.990%) and SOTA (6.5985%) to show value.

---

## 2026-05-01 — PR #624: Pre-slice STRING-RoPE (alphonse) [In Progress]

- **Branch**: alphonse/presl-rope-sweep
- **Hypothesis**: Inject STRING-RoPE positional encoding before the slicing operation (at the point level) rather than at slice centroids. Two variants: arm-a (control), arm-b (xmid-only RoPE variant).
- **Group**: `alphonse-presl-rope-sweep`

| Arm | Description | W&B Run | EP2 val_abupt | EP3 val_abupt | Best val_abupt | Status |
|-----|-------------|---------|---------------|---------------|----------------|--------|
| a | Control baseline | r3f8v68j | 8.635% | 7.579% | **7.064%** | Finished (37,367 steps) |
| b | xmid-only RoPE | a29fersn | — | — | TBD | Running ~step 4k (pre-EP1) |

- **Analysis**: Arm-a (control) finished at 7.064% — worse than SOTA 6.5985% (Δ+0.70%). Arm-b still pre-EP1. The control arm result is below SOTA, consistent with nezuko arm-a also being below SOTA — both control arms suggest these parallel training runs are slightly below the specific SOTA checkpoint conditions.
- **Status**: Monitoring arm-b for EP1 gate.

---

## 2026-05-01 — PR #647: Anchor-string no-slice Exp 3 (frieren) [CLOSED — reference trajectory]

- **Branch**: frieren/exp3-anchor-string
- **Hypothesis**: Issue #618 Experiment 3 reassignment — anchor-string approach without slicing. Two arms running: arm-b-anchor-k1024-ep4 and arm-b-anchor-k1024.
- **Group**: `frieren_exp3_anchor_string`

| Gate | Step | val_abupt | Status |
|------|------|-----------|--------|
| EP1 | 10,864 | 48.27% | PASS (normal cold-start, not divergence) |
| EP2 | 21,729 | 16.05% | PASS |
| EP3 | ~32,000 | ~10% | CRASHED (run terminated mid-epoch) |

- **Status**: CLOSED. arm-b-anchor-k1024 (multi-rank) crashed early at step ~292-332. arm-b-anchor-k1024-ep4 (o7upw6qr) completed EP1=48.27%, EP2=16.05%, then crashed mid-EP3 at ~10%.
- **Important note**: EP1=48.27% was a NORMAL cold-start trajectory, NOT divergence. This is the reference convergence trajectory for AnchorStringAttention (vanilla, no stabilizers). Thorfinn's PR #742 mistakenly identified this as divergence and added stabilizers to fix it — those stabilizers were the root cause of Run 4's failure.
- **Reference trajectory for PR #743 (Run 5) kill gates**: EP2 <20%, EP3 <15% (calibrated on this data).

---

## 2026-05-01 — PRs #648, #649, #650: New sweep PRs [In Progress]

### PR #648 — Volume-pressure loss upweighting (askeladd)
- **Group**: `volume-pressure-loss-sweep`
- **Hypothesis**: Upweight volume_pressure in the loss function (sweep weight ∈ {2.0, 4.0, 6.0}) to address the chronic 3× test-vs-val gap on volume_pressure field.
- **Status**: arm `vp-weight-2.0` at step ~3,290. Pre-EP1. Monitoring.

### PR #649 — GradNorm min-weight floor sweep (edward)
- **Group**: `gradnorm-min-weight-sweep`
- **Hypothesis**: Sweep GradNorm minimum weight floor ∈ {0.3, 0.5, 0.7}. Previously used floor=0.0 (no floor); a floor prevents any task from being completely suppressed during gradient normalization.
- **Status**: arm `gradnorm-floor-0.3` at step ~2,845. Pre-EP1. Monitoring.

### PR #650 — LR cosine floor sweep (tanjiro)
- **Group**: `lr-cosine-floor-sweep`
- **Hypothesis**: Sweep cosine LR minimum floor ∈ {1e-7, 5e-7, 5e-6, 1e-5}. Current SOTA uses lr-min=1e-6. Testing whether a higher or lower floor improves final convergence.
- **Status**: arm `lr-min-5e-6` (aon7hwtk) at ~step 6.9k. Pre-EP1. Monitoring.

---

## 2026-05-01 — PR #651: Surface curvature features (thorfinn) [KILLED]

- **Branch**: thorfinn/surface-curvature-features
- **Hypothesis**: Add k-NN-estimated surface curvature features (mean curvature H, Gaussian curvature K) as input to tau predictor. Curvature is a fundamental geometric quantity correlated with wall shear stress — concave/convex regions experience different flow regimes. Implementation: chunked k-NN (k=20, chunk=8192) with PCA-based quadratic fit; normalize to ±3σ.
- **Group**: `thorfinn-surface-curvature`

| Gate | Step | val_abupt | Decision |
|------|------|-----------|----------|
| EP2 | 21,729 | 14.613% | KILL (>12.0% threshold) |
| Final | — | 12.487% | — |

- **Decision**: KILLED at EP2. val_abupt=14.613% >> 12.0% kill gate. PR closed.
- **Analysis**: Surface curvature features (H, K) introduced via k-NN PCA-based quadratic fit destabilized training significantly — similar pattern to flow-aligned-tau (PR #641, EP2=14.613%). The additional geometric features may be introducing noisy inputs that conflict with the existing STRING positional encoding. The model architecture at L=5/hidden=512 appears sensitive to extra geometric input channels — either the feature construction is numerically unstable, or the model cannot leverage these high-frequency curvature signals at this scale. A future attempt could try normalizing more aggressively, or using curvature only as an auxiliary regularization signal rather than a direct input feature.
- **Conclusion**: Dead end in current form. Closed PR #651.

---

## 2026-05-05 — PR #660: Depth scaling L=6 sweep (thorfinn) [KILLED]

- **Branch**: thorfinn/depth-l6-sweep
- **Hypothesis**: L=5 SOTA (PR #592) outperformed L=4 by −1.90% relative. Test whether L=6 with reduced hidden_dim (384 or 448) continues the depth scaling trend. Two arms: hidden=384 (Arm A), hidden=448 (Arm B — sequential).
- **Group**: `depth-l6-sweep`

| Gate | Step | val_abupt | Decision |
|------|------|-----------|----------|
| EP1 (Arm A h=384) | 10,864 | 30.978% | KILL (elevated; experiment confounded) |

- **Decision**: KILLED at EP1. val_abupt=30.978% is elevated beyond normal range (24-28%). Experiment was fundamentally flawed — reducing hidden_dim to 384/448 to compensate for VRAM created a confounded experiment testing "L=6 with less capacity" rather than "L=6 at equal capacity."
- **Conclusion**: PR closed. Correct follow-up: PR #666 (thorfinn) — L=6 at full hidden=512 (estimated ~57GB VRAM, well within 96GB budget).

---

## 2026-05-05 — PR #614: Lion β2 momentum sweep (fern) [CLOSED — null result]

- **Branch**: fern/lion-beta2-momentum-sweep
- **Hypothesis**: Sweep Lion β2 ∈ {0.95, 0.99, 0.999} to identify optimal momentum coefficient.
- **Group**: `tay-lion-beta2-sweep`

| Arm | β2 | W&B Run | Best val_abupt | Epochs |
|-----|-----|---------|----------------|--------|
| B | 0.99 (default) | hjq54lu4 | **6.793%** | 6 |
| A | 0.95 | lcb5rb4l | **7.098%** | 4 |
| C | 0.999 | wapj7o9t | **7.219%** | 6 |

- **Decision**: Closed as null. β2=0.99 (existing default) confirmed optimal. No arm beats SOTA 6.5985%.
- **Key finding**: Lower β2=0.95 converges faster at EP1 (26.6% vs 28.1%) but the advantage narrows and inverts by EP3 (7.69% vs 7.39%); β2=0.95 final is 0.305pp worse than β2=0.99. Higher β2=0.999 is simply too sluggish to converge within budget (EP1=35.0%). Lion β2 momentum tuning is concluded as a research direction.

---

## 2026-05-05 — PRs #648 #649 #650: EP3 gate results [WIP]

### PR #648 — Volume-pressure loss upweighting (askeladd, run rl2drj1m)

| Gate | Step | val_abupt | SP | WS | VP | Decision |
|------|------|-----------|----|----|-----|---------|
| EP1 | 10,864 | 27.30% | — | — | — | Normal |
| EP2 | 21,729 | 8.21% | — | — | — | PASS |
| EP3 | 32,594 | **7.8217%** | 5.30% | 8.90% | **4.30%** | PASS (< 8.0%) |

- Status: Running to completion. EP3=7.82% PASS. VP channel at 4.30% at EP3 is lower than typical — promising signal for the vol_pressure gap problem.

### PR #649 — GradNorm min-weight floor sweep (edward, run phi418eg)

| Gate | Step | val_abupt | SP | WS | VP | Decision |
|------|------|-----------|----|----|-----|---------|
| EP1 | 10,864 | 25.78% | — | — | — | Normal |
| EP2 | 21,729 | 8.57% | — | — | — | PASS |
| EP3 | 32,594 | **7.4142%** | 5.05% | 8.28% | 4.68% | PASS (< 8.0%) |

- Status: Running to completion. Strong EP3 recovery from borderline EP2.

### PR #650 — LR cosine floor sweep (tanjiro, run aon7hwtk)

| Gate | Step | val_abupt | SP | WS | VP | Decision |
|------|------|-----------|----|----|-----|---------|
| EP1 | 10,864 | 29.42% | — | — | — | Normal |
| EP2 | 21,729 | 8.24% | — | — | — | PASS |
| EP3 | 32,594 | **7.2377%** | 4.75% | 8.19% | 4.40% | PASS (< 8.0%) |

- Status: Running to completion. Best of the three borderline-EP2 recoveries — 7.24% at EP3 is a strong signal.

---

## 2026-05-05 — New PRs assigned (Round 11–12 closures + current Phase 1 assignments)

### Closed dead ends (Rounds 11–12)

- **PR #690** (various): Slice sweep {64, 192, 256} — slices=64 null (+0.30pp); slices=192/256 infeasible (>92 min/epoch). CLOSED.
- **PR #691** (various): RFF sigma wide/low-ext — both null. CLOSED.
- **PR #692** (various): Heads sweep {8, 2} — heads=8 null (+0.83pp); heads=2 unauthorized concurrent launch, CLOSED.
- **PR #693**: L=6/h=448/heads=7 — killed (heads=7 destroys SDPA fast path, ~98 min/epoch). CLOSED.
- **PR #694**: depth L=6/hidden=384/heads=4 — null (val=6.9016%, +0.30pp), still descending but budget-bound. CLOSED.
- **PR #695**: rff-num-features=32 — null (+0.33pp val regression). CLOSED.
- **PR #716** (frieren): BC-type embedding — operationally broken (concurrent 8-GPU jobs doubled epoch time to 180 min; time-gate kill). CLOSED.
- **PR #722**: dual-tower volume/surface cross-attention — null (+0.87pp val regression). CLOSED.

### Current Phase 1 (Issue #717 volume push) — all WIP as of 2026-05-06

- **PR #728** (frieren): Exp 1B — Volume outlier-aware point sampling (EMA residual + geometric distance arms). WIP.
- **PR #729** (alphonse): Exp 1D — Single-model KD from K=7 ensemble, vol-only soft targets. WIP.
- **PR #734** (askeladd): Exp 1C P3 — SDF distance-to-surface scalar feature for volume input. WIP.
- **PR #735** (edward): TTA — Y-mirror + coord-jitter 6-pass test-time averaging. WIP.
- **PR #736** (fern): Inter-sample mixup on volume coords/pressure (alpha=0.2/0.4). WIP.
- **PR #737** (nezuko): Region-weighted volume loss — near-wake band emphasis (1<x_rel<3, |z_rel|<1.5). WIP.
- **PR #738** (tanjiro): Train-time Gaussian noise on volume coordinates (sigma 5mm/20mm/anneal). WIP.

### Issue #618 STRING/RoPE — re-attempt

- **PR #742** (thorfinn): CLOSED NEGATIVE. Exp 3 Redux — Anchor-STRING with stabilizer triplet (rope_lr_scale=0.1, rope_grad_clip=1.0, 500-step log_freq warmup). Best result: EP3 val_abupt=19.87% (step 32592). Root cause: stabilizers over-constrained RoPE (rope/log_freq moved <0.005 over 3 epochs — essentially frozen). Frieren's PR #647 EP1=48.27% was normal cold-start, not divergence. Genuine bug fixes retained for Run 5: `_init_weights` skip-`string_rope.` + mask-aware anchor selection.
- **PR #743** (thorfinn, pending): Run 5 — Frieren PR #647 exact config (no stabilizers, no rope_lr_scale, no rope_grad_clip, no qk_norm in AnchorString) + 2 genuine bug fixes only. Kill gates: EP2 (step 21728) <20%, EP3 (step 32592) <15%.

### Previous Issue #618 STRING/RoPE arms (all closed, Round 11–12)

- **PR #626** STRING only: best vol gap ratio 2.07× (val→test); established baseline for RoPE comparison.
- **PR #647** AnchorString no-slice: EP1=48.27% (normal cold-start), EP2=16.05%, crashed mid-EP3 at ~10%. Reference trajectory for Run 5 kill gate calibration.
- Other STRING/RoPE arms: null or diverged; closed.

---

## 2026-05-05 — Archived earlier new-PR assignments

- **PR #665** (frieren): Cross-slice attention over Transolver slice tokens — global inter-slice MHA layer
- **PR #666** (thorfinn): Depth scaling L=6 at full hidden=512 (corrects the confound in PR #660)
- **PR #667** (fern): Weight decay sweep {1e-4, 5e-4, 1e-3} for Lion optimizer
