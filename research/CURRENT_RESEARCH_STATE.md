# SENPAI Research State
- **Date:** 2026-05-08 22:25 (Round 19 ACTIVE — 8 tay-track WIP PRs after PR #868 askeladd CLOSED FALSIFIED. nezuko #823 leading at EP11=6.4521% (BEATS SOTA -2.07%). New askeladd assignment: mixup/geometric-interpolation augmentation for OOD vol_p.)
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current Baselines

| Tier | val_abupt | test_abupt | test_vol_pressure | PR | Notes |
|---|---|---|---|---|---|
| **Ensemble SOTA** | **6.1751%** | **7.5347%** | 11.4652% | #612 (nezuko) | K=7 greedy pool-24 |
| **Wave-test SOTA** | — | **7.5195%** | — | #740 (`5x8wofzm`) | GradNorm α=0.5 winner |
| **Single-model SOTA** | **6.5985%** | **7.9915%** | 11.933% | #592 (alphonse) | depth-L5, EP4, run `4k25s25e` |
| **Vol-pressure best anchor** | — | — | 11.374% | #681 (dc031qpt) | Issue #717 reference |

**Advancement gate:** EP4 val_abupt ≤ 6.5985% on a 4-ep tay screen → advance to 13-ep full run.

**Key finding from #767:** 4 OOD test cases (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation. Excluding them, test_vol_p = 3.9–4.2% (below AB-UPT 6.08%). Geometry conditioning of the volume decoder is the highest-priority intervention.

**Vol-pressure promotion ladder (test_vol_pressure target):**
- Weak: ≤11.0% | Solid: ≤10.0% | Major: ≤8.5% | Target: ≤6.08% (AB-UPT reference)

---

## Latest Human Researcher Directives

- **Issue #618** (STRING/RoPE): Architecture experiments concluded. All STRING axes FULLY CLOSED.
- **Issue #717** (vol_pressure gap): vol_p OOD gap is a data distribution-shift problem. Primary lever: Issue #803 SDF regeneration (blocking — human team). Active attack: nezuko #823 (surface→vol cross-attention, EP7=6.5335% BEATS SOTA).
- **Issue #803** (volume_sdf.npy): Awaiting human team delivery of regenerated SDF. BLOCKING for SDF-architecture experiments.
- No new directives pending.

---

## Round 17 Closeouts (ALL CLOSED)

| PR | Student | Hypothesis | Result | Verdict |
|----|---------|-----------|--------|---------|
| #855 | frieren | Y-sym augmentation p=0.5 | EP4=8.08% | CONFOUNDED (wrong LR flag; y-sym flags absent from train.py) |
| #856 | fern | τ Y/Z absolute upscaling | No W&B activity | CLOSED STALLED |
| #857 | askeladd | σ-ladder sweep (ascending/descending) | Both FAIL >8% | CLOSED — σ-ladder axis FULLY CLOSED |
| #858 | alphonse | Lion wd sweep {1e-4, 1e-3} | Arm A FAIL, Arm B killed | wd=5e-4 CONFIRMED OPTIMAL |
| #859 | thorfinn | slices=64 vs 256 vs 128 | Arm A FAIL; Arm B never launched | slices=64 FAIL; slices=256 → Round 18 |
| #860 | fern | τ absolute upscaling escalation | FAIL >8% | τ_y=1.5/τ_z=2.0 CONFIRMED OPTIMAL |
| #861 | edward | QK-norm ablation | EP2=26.35% CATASTROPHIC | QK-norm CONFIRMED NECESSARY |
| #862 | tanjiro | Lion β₂ sweep (β₂=0.95) | FAIL >16% | β₂=0.99 CONFIRMED OPTIMAL |

**Round 17 summary:** All optimizer hyperparameter axes (β₁, β₂, wd, lr) confirmed optimal. All STRING axes exhausted. QK-norm locked as required architectural decision.

---

## Round 18 Closeouts (Partial)

| PR | Student | Hypothesis | Result | Verdict |
|----|---------|-----------|--------|---------|
| #863 | alphonse | SGDR warm restarts within 4-ep budget | EP4=7.6208% | FAILED — LR peaks at epoch boundaries, not troughs. SGDR axis CLOSED. |
| #867 | thorfinn | slices=256 (only untested arm from Round 17 #859 Arm B) | EP3=8.1599% | FAILED — fewer points per slice degrades long-range spatial context. slices axis CLOSED (64/128/256 all tested). |

---

## Active PRs (Round 19, all RUNNING)

| PR | Student | Hypothesis | W&B run | Status |
|---|---|---|---|---|
| #823 | nezuko | Surface→vol cross-attention (13-ep full run) | `ghh0s4ne` | **EP11=6.4521% BEATS SOTA (-2.07%)**. Trajectory: EP6=6.590 → EP11=6.452 still improving. Expect sub-6.4 by EP13. |
| #868 | askeladd | Spectral normalization on attention projection layers | `0kjl4rnh` | **CLOSED FALSIFIED** EP4 val=7.6778%, test=8.9345%, test_vol_p=12.777% (OOD widened). |
| (new) | askeladd | Mixup / geometric-interpolation augmentation (OOD vol_p) | — | Assigning 2026-05-08 22:25. |
| #869 | edward | Stochastic depth (drop_path={0.05, 0.10}) | — | EP1=29.245% [OK]. Awaiting EP2 (gate ≤16.0). |
| #870 | fern | KNN smoothness penalty on τ_y/τ_z (pivoted from FFT, λ=0.1, k=8) | `d0echeyh` | EP1=30.32% borderline (gate 30.0 — barely missed). Watch EP2 closely. |
| #871 | tanjiro | PCGrad gradient surgery across 4 task groups | — | Pre-EP1 @ step=6155. |
| #872 | frieren | hidden_dim=640 width scaling | `gr1n58zo` (v2) | EP1=27.62% [OK]. VRAM 63.2 / 97.9 GB safe. |
| #873 | dl24-tanjiro | 7L STRING + GradNorm α=0.5 + Y-sym (long-track) | — | Pre-EP1 @ step=4610. |
| #875 | alphonse | AdamW vs Lion direct comparison at SOTA config | — | Pre-EP1 @ step=4708. |
| #876 | thorfinn | Huber loss δ=0.5 and δ=1.0 (two-arm) | — | Pre-EP1 @ step=3660. |

**Long-track WIP (DDP8):** #831 dl24-fern, #843 dl24-nezuko, #844 dl24-frieren.

---

## Current Research Focus

### Theme 1: Cross-Attention Geometry Conditioning (CRITICAL — Issue #717)
- **Surface→vol cross-attention** (nezuko #823, ACTIVE): EP11=6.4521% — BEATS SOTA 6.5985% by -0.146pp (-2.07%). Run `ghh0s4ne`. Trajectory still descending (EP6→EP11 = -0.138pp). **Highest-priority active experiment; expected new single-model SOTA at EP13.** Cross-attn learning confirmed (out_proj.weight 0.0→4.99). OOD geometry compression confirmed.
- **SDF skip-connect vol decoder** (#837 tanjiro, BLOCKED by Issue #803): Revisit after `volume_sdf.npy` regeneration.

### Theme 2: Optimizer Exploration (post-Lion-confirmation)
- All Lion axes now exhaustively confirmed (β₁=0.9, β₂=0.99, wd=5e-4, lr=9e-5). **No more Lion tuning.**
- **SGDR warm restarts CLOSED** (#863, EP4=7.6208%): LR at epoch boundaries coincides with cycle peaks. SGDR axis closed.
- **AdamW vs Lion comparison** (alphonse #875, ACTIVE, Round 19): Lion selected early; direct comparison at optimal configs never run. lr=9e-4 (10× translation), wd=0.01. EP4 gate ≤6.5985%.
- **Layer-wise LR decay (LLRD)**: Higher lr on later transformer layers. Never tested. Next after #875 resolves.

### Theme 3: Architecture Exploration (post-STRING-exhaustion)
- All STRING axes CLOSED. All positional encoding variants CLOSED.
- **slices axis FULLY CLOSED**: slices=64 FAIL (Round 17 #859 Arm A), slices=128 OPTIMAL (SOTA), slices=256 FAIL (#867 EP3=8.1599%). No more slices experiments.
- **Spectral normalization on attention layers** (askeladd #868, ACTIVE): EP1=25.97%, EP2=11.79% — healthy. Stability regularization orthogonal to QK-norm.
- **Stochastic depth / layer drop** (edward #869, ACTIVE): Drop random transformer layers during training; implicit ensembling at inference. EP1=29.245% [OK].
- **hidden_dim=640 width scaling** (frieren #872, ACTIVE): EP1=27.62% [OK]. Run `gr1n58zo` (v2). VRAM 63.2/97.9 GB safe.
- **Per-channel output projection**: Separate decoder head per physical quantity. Never tested.

### Theme 4: Loss Reformulation
- τ_y=1.5/τ_z=2.0 OPTIMAL, surface×2.0, volume×1.0 CONFIRMED.
- **KNN smoothness penalty** (fern #870, ACTIVE, pivoted from FFT): k=8 neighbors, var-mode, λ=0.1. EP1=30.32% borderline (just above 30.0 gate). Run `d0echeyh`.
- **PCGrad gradient surgery** (tanjiro #871, ACTIVE): Gradient conflict resolution across 4 task groups. Pre-EP1.
- **Huber loss vs MSE** (thorfinn #876, ACTIVE, Round 19): δ=0.5 and δ=1.0. Directly targets 4 OOD cases driving 92% of vol_p squared error.

---

## Key Confirmed Architectural Decisions (LOCKED)

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128 (~15.9M params)
- **QK-norm:** REQUIRED — removing causes catastrophic failure (EP2=26.35%, PR #861)
- **Positional encoding:** STRING-separable, rff16, σ={0.25,0.5,1.0,2.0,4.0} — ALL axes CLOSED
- **Optimizer:** Lion, lr=9e-5, β₁=0.9, β₂=0.99, wd=5e-4 — ALL axes CONFIRMED OPTIMAL
- **Loss weights:** τ_y=1.5, τ_z=2.0, surface=2.0, volume=1.0 — CONFIRMED OPTIMAL
- **EMA:** 0.999 (axis EXHAUSTED)
- **Training schedule:** `--lr-cosine-t-max 13 --epochs 4` (NEVER `--lr-cosine-t-max 4`)
- **Vol curriculum:** `0:16384:1:32768:2:49152:3:65536`
- **GradNorm:** CONCLUSIVELY CLOSED — 5 failures across all α (0.1, 0.25, 0.5, 0.75, 1.0)
- **Y-sym flags absent from train.py** — re-implementation required before retry
- **`--no-compile-model` required** — torch.compile + DDP NCCL deadlock at step 1
- **heads must be power-of-2** — for SDPA/Triton fast paths

---

## Kill Gates (4-ep tay screen)

| Epoch | Gate |
|-------|------|
| EP1 | <30% |
| EP2 | <16% |
| EP3 | <8% |
| EP4 | ≤6.5985% (SOTA gate) |

---

## Key Diagnostic Findings Established

- **Wall shear z is confirmed training laggard** (#758): r_i=0.01123, weight=1.699. Vol_p is NOT undertrained — gap is OOD generalization.
- **4 OOD test cases dominate vol_pressure** (#767): 92% of squared deviation. Geometry conditioning is the right lever.
- **AB-UPT geometry branch compresses OOD gap** (#626, #802): v2 achieved 3.17×→2.225× OOD gap compression (30%) even at val_abupt=8.563%.
- **FiLM mechanism saturates** (#792): γ saturates at tanh bound 100% from EP4.
- **Depth-scaling CLOSED** (#811): L6 underperforms L5 SOTA by 0.62–0.67pp.
- **Schedule alignment is a confounder** (#805): vol-w=2.0 regression collapses from +1.79pp at EP1 to +0.035pp at EP3.
- **STRING σ<0.25 definitively closed** (#829, #838): σ=0.125 aliases at 65k surface density.
- **4-ep schedule confound**: Use `--lr-cosine-t-max 13 --epochs 4`. NEVER `--lr-cosine-t-max 4`.
- **SGDR incompatible with 4-ep budget** (#863): LR at epoch boundaries coincides with cycle peaks, not cosine troughs. EP4=7.6208%. SGDR axis CLOSED.
- **slices=256 FAIL** (#867): EP3=8.1599%. Fewer points per slice degrades long-range spatial context. All slices tested (64/128/256): 128 OPTIMAL. Axis fully CLOSED.

---

## Potential Next Research Directions (post Round 18)

### Currently in flight (do not re-assign)
- **Surface→vol cross-attention (13-ep full run)** → nezuko #823 (LEADING, EP11=6.4521%)
- **Spectral normalization on attention** → askeladd #868 (EP2=11.79%, healthy)
- **Stochastic depth** → edward #869 (EP1=29.245%)
- **KNN smoothness penalty** → fern #870 (EP1=30.32% borderline)
- **PCGrad gradient projection** → tanjiro #871 (pre-EP1)
- **hidden_dim=640 width scaling** → frieren #872 (EP1=27.62%)
- **AdamW benchmark** → alphonse #875 (pre-EP1)
- **Huber loss δ=0.5/1.0** → thorfinn #876 (pre-EP1)
- **dl24-tanjiro 7L STRING + GradNorm + Y-sym** → #873 (long-track, pre-EP1)

### Next assignments (when students become idle)
1. **Merge nezuko #823** when EP13 finishes — will become new single-model SOTA.
2. **Cross-attention follow-ups**: Variations on #823 (different attention configs, larger context windows, multi-head conditioning).
3. **Compose cross-attention with other improvements** (PCGrad/drop-path/Huber): Once primary approaches confirmed.
4. **Ensemble pool-25 refresh**: After nezuko #823 full 13-ep completes — add to greedy ensemble and re-run greedy selection.

### Medium priority (Round 20+)
3. **Layer-wise LR decay (LLRD)**: Larger lr on later transformer layers. Never tested. Next after #875 AdamW resolves.
4. **Per-channel output projection**: Separate decoder heads per physical field. Never tested.
5. **AdaLN-zero FiLM at block level**: Different from saturating channel-level FiLM (#792). Conditioning on surface latents.

### Bold / plateau-protocol ideas
8. **Geometry hash-encoding input**: Replace STRING with instant-NGP style multi-resolution hash grid encoding.
9. **Test-time ensembling via dropout**: MC-dropout at inference for uncertainty-weighted ensemble.
10. **Cross-geometry pretraining then fine-tune**: Initialize from a broader geometry distribution to reduce OOD sensitivity.

### Blocked (awaiting Issue #803 — human team)
11. **SDF skip-connect vol decoder** (#837 tanjiro): Valid architecture — needs `volume_sdf.npy` regeneration.
12. **Y-symmetry augmentation re-attempt**: Flags (`--use-y-symmetry-aug`, `--y-symmetry-aug-prob`) absent from current `train.py`. Requires re-implementation.
