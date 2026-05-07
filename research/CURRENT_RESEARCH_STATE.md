# SENPAI Research State
- **Date:** 2026-05-07 (updated)
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current Baselines

| Tier | val_abupt | test_abupt | test_vol_pressure | PR | Notes |
|---|---|---|---|---|---|
| **Ensemble SOTA** | **6.1751%** | **7.5347%** | 11.4652% | #612 (nezuko) | K=7 greedy pool-24 |
| **Single-model SOTA** | **6.5985%** | **7.9915%** | 11.933% | #592 (alphonse) | depth-L5, EP4, run `4k25s25e` |
| **Vol-pressure best anchor** | — | — | 11.374% | #681 (dc031qpt) | Issue #717 reference |

---

## Latest Human Researcher Directives

- **Issue #618** (STRING/RoPE): Architecture experiments concluded. All RoPE/Anchor variants closed negative. Direction exhausted at single-model level.
- **Issue #717** (vol_pressure gap): vol_p OOD gap is a data distribution-shift problem (confirmed by L6 depth scale PR #811 and geom-branch PR #802). The gap (val≈4%, test≈12%, ~3×) is structurally identical across L5/L6/GradNorm/geom-branch. Primary lever: Issue #803 SDF regeneration for 10 REQUIRED_RESTORED_CASE_IDs (blocking — human team). Secondary: data-side augmentation.
- **Issue #803** (volume_sdf.npy): Awaiting human team delivery of regenerated SDF for 10 REQUIRED_RESTORED_CASE_IDs. BLOCKING for geometry-conditioning experiments.
- **Issue #759** (Bengio backlog): Reserved.

---

## Round Summary (Just Closed)

### Negative results this round
- **PR #817 (edward, τ_y×2.0)**: CLOSED NEGATIVE — τ_y loss-weight axis CLOSED at L5/Lion/9e-5. Manual rebalancing broke τ_y:τ_z ratio, all channels regressed. τ_y gap is architectural, not supervision-strength.
- **PR #811 (fern, L6 depth scale)**: CLOSED NEGATIVE — vol_p OOD gap structurally identical across L5/L6/GradNorm (~7.5pp). Depth-scaling axis CLOSED. Data distribution-shift is the root cause.
- **PR #802 (frieren, AB-UPT geom branch v3)**: CLOSED — architecturally validated (F1/F2/F3 all working; vol_p gap compressed 3.17×→2.225×) but NOT competitive on val_abupt (8.563% vs SOTA 6.598%). Geometry branch competes with backbone for representation budget. FiLM-style geometry conditioning may be lighter-weight alternative.

### PR returned for relaunch
- **PR #822 (thorfinn, τ_z×3.0 4-ep)**: SENT BACK — advisor kill-gate configuration bug caused false EP1 kill (EP1 val=26.61%, gate was `<12` instead of `<30`). Corrected gates: EP1<30%, EP2<10%, EP3<8%. Relaunch in progress.

---

## Active PRs (6 WIP)

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| askeladd | #820 | 2-layer GELU MLP vol decoder (512→256→GELU→1) | WIP |
| tanjiro | #821 | vol-loss-weight=2.0 budget-matched 5-ep cosine | WIP |
| nezuko | #823 | Surface→volume cross-attention for geometry-aware vol_p | WIP |
| thorfinn | #822 | τ_z×3.0 4-ep relaunch (corrected kill gates) | WIP — relaunching |
| edward | #824 | GradNorm α=0.5 on L5 SOTA 4-ep curriculum | **NEWLY ASSIGNED** |
| fern | #825 | Lion β₂=0.999 on L5 SOTA 4-ep curriculum | **NEWLY ASSIGNED** |
| alphonse | #826 | Lion wd=3e-4 on L5 SOTA 4-ep curriculum | **NEWLY ASSIGNED** |

---

## Current Research Focus

### Theme 1: Optimizer Axis Exploration
After τ_y and depth-scale experiments closed negative, the research focus shifts to optimizer hyperparameters — the one axis that has NOT been explored at the L5 SOTA stack.

- **Lion β₂=0.999** (fern, #825): Higher momentum EMA β₂ (0.99→0.999) to smooth signed-update noise on hard τ_y/τ_z channels. The PR #817 failure showed Lion signed-momentum is sensitive to gradient landscape changes — smoother β₂ may stabilize wall-shear channels.
- **Lion wd=3e-4** (alphonse, #826): Reduce L2 penalty on output-projection weights for τ_y/τ_z channels that may be penalized away by aggressive regularization. Clean single-variable test — wd=5e-4 has been constant since SOTA was established.
- **GradNorm α=0.5 on L5 SOTA** (edward, #824): Dynamic per-task loss reweighting on the correct base (L5 SOTA + 4-ep curriculum). Prior GradNorm runs showed correct weight adaptation (τ_y/τz upweighted, vol_p downweighted). Composition with SOTA recipe not yet tested.

### Theme 2: Wall-Shear τ_z Recovery
- **τ_z×3.0 4-ep relaunch** (thorfinn, #822): In-flight relaunch after kill-gate bug. Prior runs `y862359i` and `imvj1s1p` both confirmed τ_z×3.0 upweights τ_z channel appropriately. The full 4-ep budget comparison vs SOTA is pending.

### Theme 3: Volume Architecture
- **2-layer GELU MLP vol decoder** (askeladd, #820): Replaces linear `volume_out` (512→1) with 512→256→GELU→1 MLP. Clean non-linearity test at vol decoding stage.
- **Surface→volume cross-attention** (nezuko, #823): Geometry-aware vol_p decoding via surface→volume attention.
- **vol-loss-weight=2.0 budget-matched** (tanjiro, #821): vol_p weighting with correct 5-ep budget.

---

## Key Architectural Constraints Established

- **SOTA config (PR #592):** L=5, hidden=512, heads=4, slices=128, Lion lr=9e-5, wd=5e-4, β₁=0.9, β₂=0.99, EMA=0.999, STRING-sep RFF σ∈{0.25,0.5,1.0,2.0,4.0}
- **Correct training unit:** 4-epoch curriculum `--vol-points-schedule "0:16384:1:32768:2:49152:3:65536"` — 13-ep schedule CANNOT fit in 270-min budget
- **Kill gate polarity:** `<X` means "must stay below X, kill if not" — NOT `>X`
- **vol_p OOD gap:** ~7.5pp val→test gap is DATA DISTRIBUTION-SHIFT, not model capacity (confirmed L5/L6/GradNorm all identical)
- **τ_y gap:** architectural/representation issue, NOT supervision-strength (τ_y loss-weight axis CLOSED)
- **RFF spectrum {0.25,0.5,1.0,2.0,4.0}:** optimal at 4-ep budget — σ=8.0 additions hurt consistently (PRs #814, #819)
- **L6 depth:** worse than L5 at 4-ep budget (under-convergence)
- **Geometry branch:** works architecturally but competes with backbone — val 8.56% vs SOTA 6.60%, too costly for composition

---

## Potential Next Research Directions

### Optimizer / Regularization (current focus)
1. If wd=3e-4 (alphonse #826) wins → try wd=1e-4 sweep
2. If β₂=0.999 (fern #825) wins → try β₁=0.95 (higher momentum) compose
3. If GradNorm α=0.5 (edward #824) wins → try α=1.0, α=0.25 sweep

### Data-side vol_p OOD fix (highest priority, pending Issue #803)
4. Once Issue #803 SDF regeneration lands: test geometry-conditioning with correct SDF inputs for 10 REQUIRED_RESTORED_CASE_IDs
5. Volume-side coordinate noise augmentation: add Gaussian noise to vol query coordinates during training
6. Test-time augmentation (TTA) on vol coordinates: average predictions across ±ε perturbations

### Ensemble refresh
7. After new single-model candidates emerge from current round, re-run greedy pool-25 selection (nezuko)

### Architecture (lower priority)
8. FiLM modulation (lightweight γ/β from surface pooling) — lower overhead than full geom branch
9. Per-channel output projection (separate vol head per channel — sp/τx/τy/τz/vp have different spatial statistics)
