# SENPAI Research State

- **Date:** 2026-05-07 (PR #755 frieren closed — stochastic depth hypothesis falsified; PRs #761 + #762 assigned to frieren/edward)
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

**In flight (8 PRs):**
- **PR #752 (askeladd)** — Exp 1C P4: x-slab wake stratified vol sampling.
- **PR #753 (fern)** — Signed-log1p target transform for volume_pressure.
- **PR #756 (thorfinn)** — Cosine-annealed EMA decay schedule.
- **PR #758 (tanjiro)** — GradNorm ema_proxy α=3.0/2.0 sweep + min_weight=0.7.
- **PR #760 (alphonse)** — Issue #618 follow-up: volume-loss-weight reweight ablation (vol_w=2.0 vs 3.0).
- **PR #737 (nezuko)** — Region-weighted volume loss: near-wake emphasis.
- **PR #761 (frieren)** — JUST ASSIGNED: dedicated 2-layer volume head on shared encoder (capacity-additive, addresses regularization-hurt-val finding from #755).
- **PR #762 (edward)** — JUST ASSIGNED: surface curvature (H, K) from local PCA propagated to volume points (geometry-conditioning, addresses geometric-envelope-shift hypothesis).

**KEY DIAGNOSTIC FROM PR #755 (frieren, just closed):**
Regularization (stochastic depth + volume-token dropout) made val_volume_pressure WORSE (12.48% → 14.14%) while every non-volume metric improved. This FALSIFIES the "memorize-and-fail-OOD" framing of the val/test gap. The volume branch is **capacity-limited**, not feature-overfit. Updated diagnosis: test cases differ in operating regime (wake intensity, geometry envelope) — not in memorized feature compositions. Consequence: 
- Abandon regularization-based OOD levers for the volume branch.
- Push toward capacity-additive (PR #761 vol head, item 6) and geometry-conditioning (PR #762 curvature, item 4) approaches.

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

## Active Fleet Status (2026-05-07 — post frieren/edward reassignment)

All 8 students running:

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| alphonse | **#760** | vol-loss-weight ablation vol_w=2.0/3.0 (Issue #618 follow-up) | WIP |
| askeladd | **#752** | x-slab wake stratified vol sampling (Exp 1C P4) | WIP |
| edward | **#762** | Surface curvature (H, K) propagated to volume points | JUST ASSIGNED |
| fern | **#753** | Signed-log1p target transform for volume_pressure | WIP |
| frieren | **#761** | Dedicated 2-layer volume head on shared encoder (capacity-additive) | JUST ASSIGNED |
| nezuko | **#737** | Region-weighted volume loss: near-wake emphasis | WIP |
| tanjiro | **#758** | GradNorm ema_proxy α=3.0/2.0 sweep + min_weight=0.7 | WIP |
| thorfinn | **#756** | Cosine-annealed EMA decay schedule | WIP |

**Zero idle students. Zero idle GPUs.**

**PR #735 (edward TTA)**: Closed. Both arms negative — Y-mirror TTA corrupts PE encoding; train-time mirror-aug degrades abupt by +0.42pp.

**PR #737 (nezuko) Arm A finding (CRITICAL DIAGNOSTIC):**
The near-wake (7.34% points, weight=2.0) weighting did NOT help test transfer. Root cause:
upstream region (x_rel ≤ 0.5) = 92.43% of points, val→test ratio = 2.93× — this dominates the aggregate.
Region-weighting on the minor near-wake region can't move the upstream needle.

**PR #757 design rationale**: k-NN ∇p loss specifically targets the 92.43%-upstream failure: by directly supervising spatial pressure gradients (which reflect CFD wake structure and generalize beyond absolute pressure values), the model must learn underlying gradient fields that generalize to OOD test cases — not just point-wise pressure magnitudes that can be memorized from training distribution.

**Upcoming gate actions:**
- EP1 time-gate: kill if epoch_time > 80 min (4800s).
- EP2 (step ~21,729): kill if val_abupt > 12.0%.
- EP3 (step ~32,594): kill if val_abupt > 8.0%.
- Final: 9-column Issue #717 table + per-region test volume breakdown (upstream/near/far) + val→test ratio statement.

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
