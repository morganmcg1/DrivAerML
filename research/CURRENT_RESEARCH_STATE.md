# SENPAI Research State

- **Date:** 2026-05-06 14:10 UTC (Round 13 cycle — PR #763 closed, PR #772 assigned, 7 PRs WIP, 0 idle)
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

---

## Latest Human Researcher Directives

**Issue #717 (DrivAerML Single-Model Volume Pressure Plan, 2026-05-05).** Active. The chronic ~3x volume_pressure test-vs-val gap (val~3.6%, test~11.5%) is the primary systematic problem. Ensembles banned for the volume push; only single-model artifacts at inference. Phase 1 = short probes (3-6h) of dual-tower / outlier-sampling / geometry-conditioning / single-model-KD; Phase 2 = combinations of mechanisms that moved single-model test volume; Phase 3 = long verification. All PRs in this wave must use the 9-column reporting table and compare against the three frozen anchors `sogus8sx` (#599), `4k25s25e` (#592), `dc031qpt` (#681).

**Issue #618 (STRING/RoPE follow-ups, 2026-05-04).** Directive: "assign at least 2 students" to STRING/RoPE when free slots open. SATISFIED — fern #769 (slice-centroid STRING-RoPE, Exp 1) and thorfinn #764 (STRING spectral 8-octave, Exp 2) are both active.

---

## Current Baselines

| Tier | val_abupt | test_abupt | PR | Notes |
|---|---|---|---|---|
| Ensemble SOTA | **6.1751%** | 7.5347% | #612 (nezuko) | K=7 greedy pool-24; volume_p test still 11.47% |
| Single-model SOTA | **6.5985%** | 7.9915% | #592 (alphonse) | depth-L5, run `4k25s25e`, EP4 |
| Single-model gate | < 6.5985% | — | — | Must beat to update single-model SOTA |
| Issue #717 vol-test ladder | weak < 11.0%, solid <= 10.0%, major <= 8.5%, target <= 6.08% | — | — | Promotion thresholds for new vol mechanisms |

**Issue #717 frozen anchors** (every new vol-push PR reports against these):

| Run | PR | Test volume_p | Test wall_shear | Test tau_y | Test tau_z |
|---|---:|---:|---:|---:|---:|
| `sogus8sx` | #599 | 11.694 | 7.299 | 7.941 | 9.535 |
| `4k25s25e` | #592 | 11.933 | 7.334 | 8.145 | 9.298 |
| `dc031qpt` | #681 | 11.374 | 8.321 | 9.596 | 10.738 |

---

## Current Research Themes

### 1. Volume_pressure test-transfer (Issue #717) — primary focus

**Converging negatives on loss-mass arm (CLOSED):**
- **PR #752** (askeladd) near-wake x-slab oversampling — null (test_vol_p 12.49%/12.41%)
- **PR #737** (alphonse-track) near-wake loss weight — null
- **PR #763** (nezuko) upstream-region loss weight w=1.5 — null/negative (test_vol_p 12.027%, upstream val→test ratio 2.879× worse than #737's 2.76×)
- **PR #760** (alphonse) whole-volume loss weight vol_w=2.0 — null (val_abupt 6.98%, regressed)

**Confirmed root cause (PR #767 Phase 0 diagnostic):**
- Top-4 OOD test cases (run_133, run_226, run_203, run_158) account for ~92% of squared test_vol_p deviation
- Surface_p and wall_shear transfer with val→test ratio <1× — only vol_p suffers
- Gap is geometry-OOD specific, NOT addressable by point-density or loss-mass manipulation
- Volume branch is **capacity-limited**, not feature-overfit (PR #755 regularization → made vol_p worse)

**In-flight geometry-conditioning approaches (correct frontier):**
- **PR #771 (askeladd)** — Global scalar surface-latent offset per case (minimal geometry conditioning, 513 params)
- **PR #770 (frieren)** — FiLM block-wise conditioning on surface geometry latent (full per-block modulation)
- **PR #766 (alphonse)** — Offline k-NN vol_pressure grad-consistency aux loss (physics-motivated ∇p consistency)
- **PR #762 (edward)** — Surface curvature (H, K) from local PCA propagated to volume points
- **PR #772 (nezuko) NEW** — Per-point surface-anchored residual decoder (`vol_p_pred = surf_p_anchor(nearest_surf_pt) + Δ`)

**Other in-flight:**
- **PR #758 (tanjiro)** — GradNorm ema_proxy α=3.0/2.0 sweep (vol_p not primary target)

### 2. STRING/RoPE hypothesis (Issue #618)

**Directive satisfied — 2 students assigned:**
- **PR #769 (fern)** — Slice-centroid Learnable-STRING-RoPE on Transolver Q/K (Exp 1). 4 arms (control / after-QKnorm 0.5×LR / before-QKnorm 0.5×LR / no-diff-LR). Supersedes closed #765 (Exp 3 anchor-STRING, K=1024 too sparse + 0.1×LR froze RoPE).
- **PR #764 (thorfinn)** — STRING spectral budget expansion (8-octave, 24 RFF features, Exp 2).

**Key lessons from closed PR #765 (anchor-STRING):**
1. K=1024 random anchors = 0.78% of points — bimodal train loss, gradient signal washed out
2. 0.1× LR rail froze RoPE (log_freq barely moved); use 0.5× minimum
3. `LearnableCoordinateRoPE` module is sound — reused in #769 at slice centroids

### 3. Established architectural decisions (do NOT re-litigate)

- Transolver **L=5, hidden=512, heads=4, slices=128** (SOTA config).
- STRING-separable PE, RFF=16, sigmas {0.25, 0.5, 1.0, 2.0, 4.0}.
- Lion lr=9e-5, β2=0.99, weight-decay=5e-4.
- EMA decay=0.999. Loss weights: tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0.
- Heads MUST be power-of-2 for SDPA fast-path (heads=7 kills throughput).
- **NEVER run two `torchrun --nproc_per_node=8` jobs concurrently on the same 8 GPUs.**

---

## Active Fleet Status (2026-05-06 14:10 UTC)

7 students active, 0 idle:

| Student | PR | Hypothesis | Theme |
|---|---|---|---|
| alphonse | **#766** | Offline k-NN vol_pressure grad-consistency aux loss | Issue #717 geometry-OOD |
| askeladd | **#771** | Surface-latent global scalar residual offset per case | Issue #717 geometry-OOD |
| edward | **#762** | Surface curvature (H, K) propagated to volume points | Issue #717 geometry-OOD |
| fern | **#769** | Slice-centroid Learnable-STRING-RoPE on Q/K (Exp 1) | Issue #618 STRING/RoPE |
| frieren | **#770** | FiLM block-wise conditioning on surface geometry latent | Issue #717 geometry-OOD |
| nezuko | **#772** | Per-point surface-anchored residual vol_p decoder | Issue #717 geometry-OOD |
| tanjiro | **#758** | GradNorm ema_proxy α=3.0/2.0 sweep | GradNorm tuning |
| thorfinn | **#764** | STRING spectral budget expansion (8-octave, RFF=24) | Issue #618 STRING/RoPE |

---

## Potential Next Research Directions

### If vol_p conditioning experiments succeed (Phase 2)

1. **Global + local conditioning combined**: If askeladd #771 (global scalar) + frieren #770 (FiLM) both show signal, combine them on the best single-model base.
2. **Per-point surface-anchor + FiLM**: If nezuko #772 shows the physical anchor works, combine with frieren's block-wise FiLM.
3. **Long-budget verification (Phase 3)**: Two seeds, 5+ EMA epochs, multi-checkpoint reporting, for mechanisms beating 11.0% test_vol_p.

### If vol_p conditioning experiments all null (plateau protocol)

4. **Test-time adaptation (geometry-aware)**: Learn a lightweight adapter on each test case using surface-only predictions as self-supervised signal.
5. **Sub-bucketing upstream into 3–4 sub-zones**: Localize error within the upstream region to identify which spatial band drives the gap.
6. **Geometry-OOD train/test distribution analysis**: Quantify how much the 4 outlier cases differ geometrically from the training distribution (PCA of bounding-box features / shape descriptors).
7. **Implicit neural surface representation**: Replace the point-cloud surface encoder with a signed-distance implicit field that provides more geometry-discriminative features.
8. **Multi-resolution volume representation**: Voxel-grid + point hybrid with octree-style attention.

### STRING/RoPE follow-ups (after #769/#764 close)

9. **Learnable per-axis sigma in STRING**: Make the RFF sigma parameters learnable, optimizing the frequency basis for DrivAerML geometry scale.
10. **AB-UPT-style geometry branch (full)**: Revisit #626 (AB-UPT-style, which showed 35% vol_p gap compression) with longer training + backbone freeze warmup, now that the baseline is stronger.

---

## Key Operational Notes

- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Training budget:** SENPAI_TIMEOUT_MINUTES=360 (with ~90 min val reserve = ~270 min training).
- **Reproduce SOTA stack:** see `BASELINE.md` PR #592 block (Lion 9e-5, L=5, hidden=512, heads=4, slices=128, RFF=16, sigmas 0.25-4.0, vol-points-curriculum 16k->65k, EMA=0.999, grad-clip=0.5, lr-warmup-1ep, cosine T_max=13).
- **Required reporting on every Issue #717 PR:** 9-column table, three anchors (#599/#592/#681), per-case top-10 worst test volume, per-region test volume breakdown (upstream/near/far), val→test transfer ratio statement.
