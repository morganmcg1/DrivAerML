# SENPAI Research State

- **Date:** 2026-05-05 (post-Round-12 idle assignment refresh; thorfinn re-routed to Issue #618 Exp 3 Redux)
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

---

## Latest Human Researcher Directives

**Issue #717 (DrivAerML Single-Model Volume Pressure Plan, 2026-05-05).** Active. The chronic ~3x volume_pressure test-vs-val gap (val~3.6%, test~11.5%) is the primary systematic problem. Ensembles banned for the volume push; only single-model artifacts at inference. Phase 1 = short probes (3-6h) of dual-tower / outlier-sampling / geometry-conditioning / single-model-KD; Phase 2 = combinations of mechanisms that moved single-model test volume; Phase 3 = long verification. All PRs in this wave must use the 9-column reporting table and compare against the three frozen anchors `sogus8sx` (#599), `4k25s25e` (#592), `dc031qpt` (#681).

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

**In flight (2 PRs):**
- **PR #729 (alphonse)** — Exp 1D: single-model knowledge distillation from K=7 ensemble, soft-target loss on volume.
- **PR #728 (frieren)** — Exp 1B: volume anomaly/outlier-aware sampling (per-point residual EMA + geometric distance arms).

**Just assigned (5 PRs, 2026-05-05):**
- **PR #734 (askeladd)** — Exp 1C P3: soft distance-to-surface scalar feature (4-channel kernel encoding via cdist) concatenated to volume input.
- **PR #735 (edward)** — TTA: Y-mirror + 5mm coord jitter, 6-pass averaging at inference. Two arms: existing-checkpoint TTA + train-with-mirror-aug + TTA.
- **PR #736 (fern)** — Inter-sample mixup on volume coords + volume_pressure targets only (alpha=0.2 / 0.4).
- **PR #737 (nezuko)** — Region-weighted volume loss: near-wake band emphasis (1<x_rel<3, |z_rel|<1.5) at 3 weight ratios.
- **PR #738 (tanjiro)** — Train-time Gaussian noise on volume coordinates (sigma 5mm/20mm/anneal).

**Issue #618 STRING/RoPE re-attempt (1 PR, 2026-05-05):**
- **PR #742 (thorfinn)** — Exp 3 Redux: Anchor-STRING stabilized. Fixes PR #647's EP1=48.27% divergence with (1) differential LR 0.1× for RoPE freq params, (2) RoPE-path grad-clip=1.0 (separate from global 0.5), (3) conservative log-spaced freq init in [0.1, 10.0]. Targets vol_pressure val→test gap compression toward PR #626's 2.07× (vs SOTA 3.17×).

### 2. Closed dead ends from prior round (Issue #717-aligned closures)

- **PR #722** dual-tower volume/surface cross-attention — null (+0.87pp val regression).
- **PR #716** BC-type embedding (frieren) — operationally broken (concurrent 8-GPU jobs doubled epoch time; lesson learned).
- **PR #695** rff-num-features=32 — null (+0.33pp val regression).
- **PR #694** depth L=6/hidden=384/heads=4 — null (val=6.9016%, +0.30pp), still descending but budget-bound.
- **PR #693** L=6/h=448/heads=7 — killed (heads=7 destroys SDPA fast path, ~98 min/epoch).
- **PR #692** heads sweep {8, 2} — heads=8 null (+0.83pp); heads=2 unauthorized launch, closed.
- **PR #691** RFF sigma {wide, low-ext} — both arms null.
- **PR #690** slice sweep {64, 192, 256} — slices=64 null (+0.30pp); slices=192/256 infeasible (>92 min/epoch).
- **PR #667** weight-decay sweep {1e-4, 5e-4, 1e-3} — all 3 null on test, weakest WD wins on val only.

### 3. Established architectural decisions (do NOT re-litigate)

- Transolver **L=5, hidden=512, heads=4, slices=128** (SOTA config).
- STRING-separable PE, RFF=16, sigmas {0.25, 0.5, 1.0, 2.0, 4.0}.
- Lion lr=9e-5, β2=0.99, weight-decay=5e-4.
- EMA decay=0.999. Loss weights: tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0.
- Heads MUST be power-of-2 for SDPA fast-path (heads=7 kills throughput).
- **NEVER run two `torchrun --nproc_per_node=8` jobs concurrently on the same 8 GPUs** (PR #716 lesson — doubled epoch time to 180 min, time-gate kill).

---

## Active Fleet Status

All 8 students running on Issue #717 volume push:

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| alphonse | #729 | Single-model KD from K=7 ensemble (Exp 1D) | WIP |
| frieren | #728 | Volume outlier-aware sampling (Exp 1B) | WIP |
| thorfinn | **#742** | Anchor-STRING stabilized (Issue #618 Exp 3 Redux) | Just assigned |
| askeladd | **#734** | SDF distance-to-surface feature (Exp 1C P3) | Just assigned |
| edward | **#735** | TTA Y-mirror + jitter (orthogonal to Exp 1*) | Just assigned |
| fern | **#736** | Volume input-mixup regularizer | Just assigned |
| nezuko | **#737** | Region-weighted volume loss (near-wake) | Just assigned |
| tanjiro | **#738** | Train-time volume coord-noise (Bishop-style) | Just assigned |

**Zero idle students. Zero idle GPUs.**

**Upcoming gate actions (6 newly assigned PRs incl. #742):**
- EP1 informational + time gate kill if epoch_time > 80 min.
- EP2 (step ~21,729): kill if val_abupt > 12.0%.
- EP3 (step ~32,594): kill if val_abupt > 8.0%.
- Final SENPAI-RESULT must include 9-column Issue #717 table + per-region/per-case test volume diagnostics.

---

## Potential Next Research Directions (post-current-round)

### Phase 2 (combinations) — only after Phase 1 produces test-volume movers

1. **Best-volume mechanism on #599 base** (Issue #717 Exp 2A). Combine the single most-effective Phase 1 lever with the wall-shear-clean #599 anchor.
2. **Best-volume + mild tau stabilization** (Exp 2B). Only if a tau-stabilizing mechanism (#669 family) clears clean test evidence.
3. **L5/RFF capacity variant** (Exp 2C). Re-stack mild RFF capacity on L=5 if Phase 1 finds the volume mechanism is capacity-bound.

### Mechanism follow-ups to consider if Phase 1 partially succeeds

4. **Curvature features** (Issue #717 1C P4) on top of distance-to-surface. Soft H/K from local point-cloud PCA. Lower priority than P3.
5. **Spectral pre-bias**: If geometry-conditioning helps but is noisy, learn the sigma kernel widths end-to-end (PR #691 RFF sigma proved fixed-set is brittle but learning could change that).
6. **Depth-wise volume head**: a dedicated 2-layer Transolver head exclusively for volume queries, branching off the shared encoder (different from the failed dual-tower #722 by sharing encoder, splitting only at the head).
7. **Long-budget verification (Phase 3)**: Two seeds, 5+ EMA epochs, multi-checkpoint reporting, only for mechanisms that beat 11.0% test volume in short probes.

### Plateau protocol candidates if 5 in-flight + 5 new all null

8. **Switch to a Markov / learned sampler over volume points** (RL or top-K residual-driven): more aggressive than #728's EMA reweighting.
9. **Volume-only flow-matching / diffusion-style refinement head** on top of the deterministic prediction.
10. **Multi-resolution volume representation**: voxel-grid + point hybrid with octree-style attention.

---

## Key Operational Notes

- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Training budget:** SENPAI_TIMEOUT_MINUTES=360 (with ~90 min val reserve = ~270 min training).
- **Reproduce SOTA stack:** see `BASELINE.md` PR #592 block (Lion 9e-5, L=5, hidden=512, heads=4, slices=128, RFF=16, sigmas 0.25-4.0, vol-points-curriculum 16k->65k, EMA=0.999, grad-clip=0.5, lr-warmup-1ep, cosine T_max=13).
- **Required reporting on every Issue #717 PR:** 9-column table, three anchors (#599/#592/#681), per-case top-10 worst test volume, per-region test volume breakdown, val→test transfer ratio statement.
