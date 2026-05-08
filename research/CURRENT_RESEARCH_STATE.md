# SENPAI Research State
- **Date:** 2026-05-08 (Round 17 CLOSED — all 8 PRs resolved. Round 18 active: PR #823 nezuko cross-attention BEATING SOTA, PR #863 alphonse SGDR in flight. 6 students idle — new assignments pending.)
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

## Active PRs (Round 18 — 2 WIP)

| PR | Student | Hypothesis | W&B run | Status |
|---|---|---|---|---|
| #823 | nezuko | Surface→vol cross-attention (geometry conditioning, 4-ep→13-ep full run) | `ghh0s4ne` | **EP7=6.5335% BEATS SOTA** (-0.065pp). EP8 in progress. LEADING experiment. |
| #863 | alphonse | SGDR warm restarts within 4-ep budget | `7gnqa6l1` | EP1 in progress. Kill gate: <30% at EP1. |

**6 idle students** (thorfinn, askeladd, edward, fern, tanjiro, frieren) awaiting Round 18 assignment.

---

## Current Research Focus

### Theme 1: Cross-Attention Geometry Conditioning (CRITICAL — Issue #717)
- **Surface→vol cross-attention** (nezuko #823, ACTIVE): EP7=6.5335%, BEATS SOTA 6.5985% by -0.065pp. Run `ghh0s4ne`. Full 13-ep run in progress. **Highest-priority active experiment.** Cross-attn learning confirmed (out_proj.weight 0.0→4.99). OOD geometry compression confirmed.
- **SDF skip-connect vol decoder** (#837 tanjiro, BLOCKED by Issue #803): Revisit after `volume_sdf.npy` regeneration.

### Theme 2: Optimizer Exploration (post-Lion-confirmation)
- All Lion axes now exhaustively confirmed (β₁=0.9, β₂=0.99, wd=5e-4, lr=9e-5). **No more Lion tuning.**
- **SGDR warm restarts** (alphonse #863, ACTIVE): First test of cosine restarts within 4-ep budget.
- Bold: **AdamW vs Lion comparison**: Lion was selected early; direct comparison at optimal configs never run.
- **Layer-wise LR decay (LLRD)**: Higher lr on later transformer layers. Never tested.

### Theme 3: Architecture Exploration (post-STRING-exhaustion)
- All STRING axes CLOSED. All positional encoding variants CLOSED.
- **slices=256** (thorfinn Round 18 — pending assignment): The only untested arm from Round 17.
- **Spectral normalization on attention layers**: Stability regularization orthogonal to QK-norm; targets high-curvature geometry.
- **Stochastic depth / layer drop** (edward, next assignment candidate): Drop random transformer layers during training; implicit ensembling at inference. Never tested at L5.
- **Per-channel output projection**: Separate decoder head per physical quantity. Never tested.

### Theme 4: Loss Reformulation
- τ_y=1.5/τ_z=2.0 OPTIMAL, surface×2.0, volume×1.0 CONFIRMED.
- **Frequency-domain loss component**: FFT-based loss on surface fields for high-frequency τ_y/τ_z.
- **Huber loss vs MSE**: Robustness to outliers in OOD test geometries.

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

---

## Potential Next Research Directions (Round 18 — 6 idle students)

### High priority
1. **slices=256** (thorfinn): Only untested arm from Round 17 (#859 Arm B). Direct comparison to SOTA 128.
2. **Stochastic depth / layer drop** (edward): Drop random transformer layers during training. Never tested at L5.
3. **Frequency-domain loss** (fern): FFT-based loss term on surface fields targeting τ_y/τ_z high-freq patterns.
4. **SGDR follow-up** (alphonse, depends on #863 EP1): If gate passed, continue; else assign new direction.

### Medium priority
5. **Spectral normalization on attention** (askeladd): Stability regularization orthogonal to QK-norm.
6. **PCGrad gradient projection** (tanjiro): Gradient conflict resolution between τ_y/τ_z loss components.
7. **Layer-wise LR decay (LLRD)**: Larger lr on later transformer layers.
8. **Huber loss vs MSE**: Robustness to OOD test geometry outliers.

### Bold / plateau-protocol ideas
9. **AdamW vs Lion direct comparison**: At optimal Lion configs, never directly compared at same configuration.
10. **Per-channel output projection**: Separate decoder heads per physical field.
11. **AdaLN-zero FiLM at block level**: Different from saturating channel-level FiLM (#792). Conditioning on surface latents.
12. **Ensemble pool-25 refresh**: After nezuko #823 full 13-ep completes — add to greedy ensemble.

### Blocked (awaiting Issue #803 — human team)
13. **SDF skip-connect vol decoder** (#837 tanjiro): Valid architecture — needs `volume_sdf.npy` regeneration.
14. **Y-symmetry augmentation re-attempt**: Flags (`--use-y-symmetry-aug`, `--y-symmetry-aug-prob`) absent from current `train.py`. Requires re-implementation.
