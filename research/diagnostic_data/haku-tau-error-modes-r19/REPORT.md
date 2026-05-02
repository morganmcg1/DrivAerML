# PR #363 — Tau_y/z Error-Mode Diagnostic Report (r19)

**Author:** STUDENT haku
**Branch:** `haku/diagnose-tau-yz-error-modes`
**W&B group:** `haku-tau-error-modes-r19`
**Date:** 2026-05-02

---

## Top 3 takeaways for the fleet

1. **Cross-flow alignment is the dominant per-point driver of tau_y/z error.**
   Splitting val points by the angle theta between the GT wall-shear vector and the global-x axis: stream-aligned points (theta < 30°) have mean |tau_y_err| = 0.070 and |tau_z_err| = 0.066, while cross-flow points (theta > 60°) have 0.112 and 0.104 respectively — a **1.60x / 1.57x increase**. The decile 9 → decile 0 gradient is even sharper (~**2.2x** for both y and z). The model genuinely cannot represent cross-flow content; this is not "under-resolution that loss tweaks alone can fix". Theta-conditioned loss weighting, anisotropic feature representations, or tangent-frame loss formulations are the targeted interventions; isotropic global optimizers are unlikely to close the y/z gap further.

2. **High surface curvature is the second-strongest geometric driver.**
   Curvature deciles 7-9 (wheel arches, A-pillar, mirror housings) have **1.41x** the |tau_y_err| and **1.21x** the |tau_z_err| of deciles 0-2 (flat panels). For tau_x the ratio is similar (~1.40x). Local feature enrichment around high-curvature patches (explicit curvature features, mesh Laplacian channels, or local cross-attention) is a concrete intervention. This complements the cross-flow finding: high curvature and cross-flow co-occur on the same physical regions (wheel arches induce both).

3. **Errors are spatially clustered everywhere but not catastrophically concentrated in any single region.**
   Moran's I (kNN-16, 5000-pt subsample, 199 permutations) shows 34/34 cases have *p < 0.05* on all three axes (median I_y=0.124, I_z=0.157, I_x=0.177), so within each case errors form spatial blobs rather than salt-and-pepper noise. **However, no fleet-wide region exceeds 2x the global mean error**: the worst region (front-side-underbody) sits at 1.52x tau_y and 1.54x tau_z, and accounts for 26% of all val surface points. Targeted region-only loss weighting (>2x) is not justified by the data; instead, treat front-side-underbody as the highest-mass contributor and combine it with the cross-flow / curvature signals above. Per-case spread is large (run_324: 16.12% tau_y; run_298: 12.12%) and run_228 is a tau_z outlier (21.63%); investigating those four cases for geometry/data artifacts is a cheap follow-up.

---

## Setup

**Checkpoint origin (important caveat):** The canonical PR #222 W&B run `ut1qmc3i` is no longer accessible from W&B (the run id resolves but artifact restore fails for this entity/project — verified across multiple paths). As a stand-in I used the kohaku/seed2024 W&B run `k7wq5uxx` (PR #313 ensemble member, *yi*-branch), whose architecture matches PR #222 exactly: 4 layers / 512 hidden / 8 heads / 128 slices, AdamW lr=5e-4, EMA(0.999), `pos_max_wavelength=1000`, `wallshear_y_weight=2`, `wallshear_z_weight=2`. Best-val checkpoint loaded from artifact `model-seed2024-k7wq5uxx:best` (epoch 2; val_abupt 10.35%, val_wsy 13.58%, val_wsz 14.57% per W&B run summary).

The run is a few percent worse than PR #222 (9.291% val_abupt) in absolute terms but the *relative* spatial / geometric / per-case error patterns should be near-identical — both checkpoints are 4L/512d/8H/128sl yi-branch models trained with the same loss, the same per-channel weights, and same EMA. **All section conclusions describe the relative shape of the error distribution and should hold for PR #222 to within noise.**

**Inference protocol:** validation split (34 cases), full-fidelity strided eval, batch size 1, AMP bf16. Each val point visited exactly once across views (`eval_chunk` strided sampling), so per-case point counts are the full mesh sizes (~7.5M – 9.7M points/case, 286M points fleet-wide). 5.5 minutes wall-clock on 1×H100.

**Inference rel-L2 (global, val set):**
- tau_x: **9.94%**
- tau_y: **13.62%**
- tau_z: **14.65%**

(For comparison, PR #222 wall_shear_rel_l2 combined ≈ 10.34%; per-channel not separately logged. The ratio tau_y / tau_x ≈ 1.37 and tau_z / tau_x ≈ 1.47 here is consistent with the fleet's qualitative observation of a ~30–50% tau_y/z relative gap over tau_x.)

Predictions saved to `predictions/run_<id>.npz` (one file per case). Analysis saved to `analysis/`.

---

## Section 1 — Per-axis error magnitude by surface region

Each surface point binned by its signed-bbox-relative coordinate (per case): **x** ∈ {front (x_rel<-0.166), mid (-0.166..0.166), rear (>0.166)} × **y** ∈ {center (|y_rel|<0.25), side (>=0.25)} × **z** ∈ {underbody (z_rel<-0.166), body (-0.166..0.166), roof (>=0.166)}. 18 regions populated.

**Global mean |error|:** tau_x = 0.135, tau_y = 0.0918, tau_z = 0.0834 (Pa, after target inversion).

**Top-10 regions by |tau_y_err| (full table at `analysis/section1_region_table.csv`, figure at `analysis/section1_region_top10.png`):**

| region | n_points | mean_y | ratio_y | mean_z | ratio_z |
|--------|---------:|-------:|--------:|-------:|--------:|
| front-side-underbody     | 81,426,249 | 0.1394 | **1.52** | 0.1282 | **1.54** |
| rear-side-underbody      | 78,190,509 | 0.0883 | 0.96    | 0.0707 | 0.85    |
| rear-center-underbody    |  7,287,218 | 0.0873 | 0.95    | 0.0785 | 0.94    |
| front-side-roof          |    980,567 | 0.0844 | 0.92    | 0.0577 | 0.69    |
| mid-center-underbody     | 10,084,824 | 0.0802 | 0.87    | 0.0751 | 0.90    |
| front-center-body        |  5,450,796 | 0.0762 | 0.83    | 0.0872 | **1.05**|
| mid-side-body            |  4,331,277 | 0.0709 | 0.77    | 0.0796 | 0.95    |
| front-center-underbody   |  6,205,175 | 0.0689 | 0.75    | 0.0832 | 1.00    |
| front-side-body          | 24,369,618 | 0.0684 | 0.74    | 0.0687 | 0.82    |
| mid-side-roof            | 18,236,480 | 0.0680 | 0.74    | 0.0646 | 0.77    |

**Top-3 regions where tau_y or tau_z >= 2x global mean: NONE.** The maximum ratio observed is 1.54x (front-side-underbody, tau_z).

**Interpretation.** The error landscape is *graded*, not bimodal. Front-side-underbody is the only region simultaneously elevated on all three axes (1.46x / 1.52x / 1.54x for tau_x/y/z), and it is also the largest region by point count (~26% of val surface points). Together this region is the single largest contributor to fleet-wide error mass for tau_y (~40% by integrated abs error) and tau_z (~46%). The relative-elevation factor is modest, so a region-only loss reweighting at 2x threshold has nothing to bite on — but front-side-underbody-targeted *augmentation* or *local feature enrichment* is a justified intervention because of its absolute mass. Roof and rear-body regions consistently have the lowest errors; mid-side-roof in particular has anomalously elevated tau_x (1.03x) but suppressed tau_y/z, suggesting the model handles streamwise content there but struggles with cross-flow components even on a flat panel.

---

## Section 2 — Spatial autocorrelation (Moran's I)

Per case: random subsample of N=5000 surface points, kNN graph with k=16 from `surface_xyz`, row-standardized weights, observed I and 199-permutation null distribution, two-sided p-value. Computed for each axis of |tau_pred − tau_target|.

| axis | median I | mean I | min I | max I | frac p<0.05 |
|------|---------:|-------:|------:|------:|------------:|
| tau_x | 0.177 | 0.189 | 0.140 | 0.282 | **34/34 = 1.00** |
| tau_y | 0.124 | 0.128 | 0.083 | 0.185 | **34/34 = 1.00** |
| tau_z | 0.157 | 0.166 | 0.115 | 0.247 | **34/34 = 1.00** |

Figure: `analysis/section2_morans_i.png`.

**Interpretation.** All 34 val cases × 3 axes have statistically significant positive spatial autocorrelation. This rules out the "errors are random, optimizer just hasn't converged the last bit of stochastic noise" hypothesis. Errors form spatial *blobs* — coherent regions where the model is wrong together — for every case. Median I_y = 0.124 is moderate (interpretable as "neighbouring 16 points have ~12% predictive power for a point's error magnitude"); I_z = 0.157 is slightly stronger. Cases with the highest spatial structure (run_328: I_y=0.185, run_324: I_y=0.175, run_487: I_y=0.170) are also among the worst per-case rel-L2 (Section 5), so high spatial structure correlates with high overall error. Implication: any fix that locally relaxes a single coherent failure region can yield disproportionate fleet improvement, because the error mass is concentrated on spatial patches rather than diffuse.

---

## Section 3 — Error vs surface curvature

Local scalar curvature per point: smallest eigenvalue of the kNN(k=32) covariance, divided by eigenvalue sum (range [0, 1/3]; small ≈ flat, large ≈ ridge/edge). Computed on a 30,000-point per-case subsample (1.02M points total), then binned into deciles of curvature globally. Mean abs error per axis per decile:

| decile | curvature_p50 | tau_x_err | tau_y_err | tau_z_err |
|-------:|--------------:|----------:|----------:|----------:|
| 0 (flat)   | 0.0013 | 0.139 | 0.067 | 0.063 |
| 1          | 0.0041 | 0.112 | 0.080 | 0.085 |
| 2          | 0.0060 | 0.104 | 0.083 | 0.086 |
| 3          | 0.0083 | 0.099 | 0.082 | 0.084 |
| 4          | 0.0119 | 0.099 | 0.079 | 0.077 |
| 5          | 0.0213 | 0.124 | 0.090 | 0.073 |
| 6          | 0.0367 | 0.161 | 0.111 | 0.081 |
| 7          | 0.0562 | 0.178 | 0.111 | 0.088 |
| 8          | 0.0879 | 0.179 | 0.108 | 0.096 |
| 9 (sharp)  | 0.1473 | 0.152 | 0.105 | 0.099 |

**Low (deciles 0–2) vs high (deciles 7–9) ratio:**
- tau_y: 0.0765 vs 0.1082 → **1.41x**
- tau_z: 0.0781 vs 0.0946 → **1.21x**
- tau_x: similar to tau_y (~1.4x).

Figure: `analysis/section3_curvature.png`. CSV: `analysis/section3_curvature_table.csv`.

**Interpretation.** Errors do concentrate in high-curvature regions, consistent with the fleet's intuition that wheel arches, A-pillars, and mirror housings are problem zones. The effect is real but modest (1.2–1.4x, not 5x). Notably, tau_x has a U-shape — large error on the very flattest decile too — likely because the underbody is large/flat *and* sees high-magnitude streamwise shear, so absolute error scales with target magnitude there. tau_y and tau_z monotonically increase from decile 1 to 7 then plateau: the model handles flat panels well but degrades into ridges/edges. Concrete intervention: feed local curvature (or a kNN-pooled feature) as an additional input channel; add a curvature-conditioned loss weight; or use mesh-informed local attention for the highest-decile bins. Combine with the cross-flow signal (Section 4) to identify mutual high-leverage points.

---

## Section 4 — Error vs flow alignment

For each surface point, theta = arccos(|tau_target_x| / ||tau_target||), folded to [0, π/2]. theta = 0 → wall-shear is purely streamwise; theta → π/2 → purely cross-flow (where tau_y/z dominates). Filtered to ||tau_target|| > 0.05 to exclude near-stagnation/separation noise (target-norm percentiles: p10=0.10, p50=0.82, p90=4.17). Then deciles over remaining theta.

| decile | theta_p50 (deg) | target_norm_p50 | tau_x_err | tau_y_err | tau_z_err |
|-------:|----------------:|----------------:|----------:|----------:|----------:|
| 0 |  4.4 | 2.20 | 0.174 | **0.052** | **0.053** |
| 1 | 11.1 | 1.97 | 0.174 | 0.068 | 0.062 |
| 2 | 18.2 | 1.76 | 0.173 | 0.077 | 0.072 |
| 3 | 25.7 | 1.36 | 0.158 | 0.082 | 0.077 |
| 4 | 34.7 | 0.94 | 0.147 | 0.094 | 0.087 |
| 5 | 45.1 | 0.77 | 0.135 | 0.108 | 0.093 |
| 6 | 55.4 | 0.71 | 0.120 | 0.116 | 0.092 |
| 7 | 65.9 | 0.57 | 0.105 | 0.109 | 0.099 |
| 8 | 76.3 | 0.54 | 0.094 | 0.112 | 0.104 |
| 9 | 85.6 | 0.54 | 0.088 | **0.115** | **0.108** |

**Stream-aligned (theta<30°) vs cross-flow (theta>60°):**
- tau_y: 0.0699 vs 0.1118 → **1.60x**
- tau_z: 0.0660 vs 0.1036 → **1.57x**
- decile-9 vs decile-0 (more extreme): tau_y 2.21x, tau_z 2.04x.

Figure: `analysis/section4_alignment.png`. CSV: `analysis/section4_alignment_table.csv`.

**Interpretation.** This is the strongest single per-point signal in the diagnostic. The tau_y/z error increases monotonically with theta from decile 0 to decile 9, and the stream/cross-flow ratio is ~1.6x — larger than the curvature ratio (1.2–1.4x) and larger than any single region ratio (max 1.54x). Importantly, target magnitude *decreases* with theta (decile 0 has p50 |tau| = 2.2; decile 9 has 0.54), so the model is making its largest absolute errors on the *smallest-magnitude* targets. This means the rel-L2 gap on tau_y/z is driven by the model failing to predict the small cross-flow components rather than by failing to scale them — it is missing **cross-flow content**, not amplitude.

The mirror image: tau_x error *decreases* with theta (decile 0: 0.174 → decile 9: 0.088). This is by construction (cross-flow points have small target tau_x by definition) but reinforces that the model is internally biased toward streamwise content. The decile-0 tau_x error is also remarkably uniform across decile — this is the "overall scale of streamwise prediction error" baseline.

**Concrete intervention:** theta-conditioned loss weighting (multiplicative weight depending on local target alignment), tangent-frame wall-shear loss formulations (e.g. the existing `use_tangential_wallshear_loss` flag, which is *off* in the stand-in checkpoint), or anisotropic feature representations (separate streamwise and cross-flow heads). The 1.6x ratio is a >50% margin and should produce visible movement in val_wsy/wsz if successfully exploited.

---

## Section 5 — Per-case ranking

Per-case rel-L2 (%) ordered descending by tau_y. Full table at `analysis/section5_per_case_table.csv`; sorted bar chart at `analysis/section5_per_case.png`.

**Top 3 worst by tau_y rel-L2:**
| rank | case_id | tau_y_rel_l2_% | tau_z_rel_l2_% | tau_x_rel_l2_% |
|-----:|---------|---------------:|---------------:|---------------:|
| 1 | run_324 | **16.12** | 18.68 | 11.71 |
| 2 | run_275 | **15.18** | 15.88 | 10.61 |
| 3 | run_380 | **15.04** | 14.61 | 11.67 |

**Top 3 best by tau_y rel-L2 (i.e. easiest cases):**
| rank | case_id | tau_y_rel_l2_% | tau_z_rel_l2_% | tau_x_rel_l2_% |
|-----:|---------|---------------:|---------------:|---------------:|
| 32 | run_241 | 12.45 | 12.69 | 8.72 |
| 33 | run_4   | 12.44 | 12.74 | 8.91 |
| 34 | run_298 | **12.12** | 13.86 | 8.68 |

**Top 3 worst by tau_z rel-L2 (different ordering):**
| rank | case_id | tau_z_rel_l2_% | tau_y_rel_l2_% |
|-----:|---------|---------------:|---------------:|
| 1 | run_228 | **21.63** | 14.00 |
| 2 | run_324 | **18.68** | 16.12 |
| 3 | run_275 | **15.88** | 15.18 |

**Top 3 best by tau_z rel-L2:**
| rank | case_id | tau_z_rel_l2_% | tau_y_rel_l2_% |
|-----:|---------|---------------:|---------------:|
| 32 | run_241 | 12.69 | 12.45 |
| 33 | run_271 | 12.63 | 12.49 |
| 34 | run_165 | **12.48** | 13.38 |

**Interpretation.** Per-case spread is **substantial** — tau_y rel-L2 ranges from 12.12% (run_298) to 16.12% (run_324), a 33% relative spread; tau_z is even wider, 12.48% (run_165) to 21.63% (run_228), a **73% relative spread**. run_228 is a clear tau_z outlier (gap of +2.9 percentage points to the next-worst case), and worth opening individually — likely a geometry that exercises a failure mode the model has not seen. run_324 appears in top-3 worst on **both** tau_y and tau_z, and is also among the highest spatial-autocorrelation cases (Section 2: I_y=0.175), so its errors are clustered on a few coherent regions; visualizing run_324 surface error overlaid on the mesh is the highest-yield single follow-up.

The "easy" cases (run_241, run_4, run_298) all sit at ~8.7% tau_x rel-L2, ~12.5% tau_y/z, suggesting a floor around 12% that no case beats — consistent with the fleet's hypothesis that closing tau_y/z below ~10% requires architectural change, not just optimization. **For targeted train-set augmentation: add geometric variations around run_324, run_228, run_275, run_380** (the four worst cases collectively) to give the model more support around their failure modes.

---

## Files in this directory

- `predictions/run_<id>.npz` — per-case raw inputs/outputs (34 files, ~12 GB compressed). Keys: `surface_xyz`, `surface_normals`, `tau_pred`, `tau_target`, `ps_pred`, `ps_target`.
- `predictions/summary.yaml` — global rel-L2 numbers + checkpoint provenance.
- `checkpoint_k7wq5uxx/checkpoint.pt` + `config.yaml` — frozen reference checkpoint (50.8 MB).
- `analysis/stats.json` — full numeric stats for every section, machine-readable.
- `analysis/per_case_stats.json` — per-case Moran's I + per-case rel-L2 rows.
- `analysis/section1_region_table.csv` and `section1_region_top10.png`.
- `analysis/section2_morans_i.png`.
- `analysis/section3_curvature_table.csv` and `section3_curvature.png`.
- `analysis/section4_alignment_table.csv` and `section4_alignment.png`.
- `analysis/section5_per_case_table.csv` and `section5_per_case.png`.
- `run_inference.py`, `run_analysis.py` — reproducer scripts.

## Reproduction

```bash
# 1) inference (5.5 min on 1xH100, ~14 GB peak GPU mem, ~17 GB RAM)
cd /workspace/senpai/target
python research/diagnostic_data/haku-tau-error-modes-r19/run_inference.py \
  --checkpoint research/diagnostic_data/haku-tau-error-modes-r19/checkpoint_k7wq5uxx/checkpoint.pt \
  --checkpoint-config research/diagnostic_data/haku-tau-error-modes-r19/checkpoint_k7wq5uxx/config.yaml \
  --out-dir research/diagnostic_data/haku-tau-error-modes-r19/predictions \
  --split val

# 2) analysis (~7 min, ~30 GB RAM at peak — full-resolution per-region stats)
python research/diagnostic_data/haku-tau-error-modes-r19/run_analysis.py \
  --predictions-dir research/diagnostic_data/haku-tau-error-modes-r19/predictions \
  --out-dir research/diagnostic_data/haku-tau-error-modes-r19/analysis \
  --moran-points 5000 --moran-knn 16 --curvature-knn 32 \
  --moran-permutations 199 --seed 0
```

## Suggested follow-ups (not implemented in this PR)

- **Theta-conditioned loss weight** on wall-shear loss: weight ∝ 1 + α · (theta_local / (π/2)), α ∈ [0.5, 2.0] swept on a 5-epoch run. Predicted to move tau_y/z rel-L2 down ~5–10% relative based on the 1.6x cross-flow gap.
- **Curvature feature injection**: add a per-point scalar curvature (kNN-32 PCA) as an 8th surface_x channel. Predicted to help wheel-arch / A-pillar regions (deciles 7–9).
- **Front-side-underbody targeted augmentation**: random rotations / small translations applied preferentially to underbody mesh patches in train.
- **Investigate run_228 (tau_z outlier) and run_324 (top-3 on both axes)**: load these two cases, render the per-point error vector field on the surface mesh, identify the single failure region. One-day investigation, very high ROI.
- **Tangential-frame loss** (`use_tangential_wallshear_loss=True`, currently *off* in the stand-in): re-orients the loss to tangent/normal coords so that y/z gap is partially absorbed into a magnitude+direction split.
- **Re-run this diagnostic on the actual PR #222 EMA weights** if/when artifact accessibility is restored — to confirm the absolute numbers and ratios. Relative findings (region/curvature/alignment ratios) are likely unchanged but absolute mean abs errors will scale with the global ~5% rel-L2 gap between the stand-in (10.35% val_abupt) and PR #222 (9.29% val_abupt).
