# SENPAI Research State
- **Date:** 2026-05-01 (Round 17 active — 8 WIP PRs on `tay` advisor branch. Closed this cycle: #850 tanjiro (Lion β₂ 13-ep timeout/inconclusive), #854 edward (GradNorm α=0.1 NEGATIVE, 5th GradNorm failure — axis CONCLUSIVELY CLOSED). New assignments: #861 edward (QK-norm ablation), #862 tanjiro (Lion β₂ sweep 4-ep).)
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

- **Issue #618** (STRING/RoPE): Architecture experiments concluded. RFF capacity axis FULLY CLOSED (rff16=SOTA). σ-ladder internal geometry under test (#857 askeladd — last remaining STRING axis).
- **Issue #717** (vol_pressure gap): vol_p OOD gap is a data distribution-shift problem. Primary lever: Issue #803 SDF regeneration (blocking — human team). Active attack: nezuko #823 (surface→vol cross-attention, EP3=7.12% ✓, EP4 imminent).
- **Issue #803** (volume_sdf.npy): Awaiting human team delivery of regenerated SDF. BLOCKING for SDF-architecture experiments (#837 tanjiro blocked).
- No new directives pending.

---

## Latest Closeouts (Round 17 cycle)

- **PR #854 (edward, GradNorm α=0.1)**: CLOSED NEGATIVE. 5th GradNorm failure — all α values (0.1, 0.25, 0.5, 0.75, 1.0) now exhausted. GradNorm is conclusively anti-synergistic with L5/Lion/STRING stack. **GradNorm axis PERMANENTLY CLOSED.** New assignment: PR #861 (QK-norm ablation).
- **PR #850 (tanjiro, Lion β₂ sweep 13-ep)**: CLOSED INCONCLUSIVE/BUDGET. Arm A (β₂=0.95) run `dxole713` timed out at step 37,370 (270-min budget), between 13-ep EP3 and EP4 boundary (~43,459). Best val=6.9165%, still descending. 13-ep schedule cannot fit in budget. **Re-assigned on 4-ep tay screen**: PR #862 (tanjiro, Lion β₂ sweep 4-ep).

## Round 16 Closeouts

- **PR #853 (thorfinn, model dropout)**: CLOSED NEGATIVE.
- **PR #856 (fern, tau Y/Z upscaling)**: CLOSED STALLED.
- **PR #848 (alphonse, Lion lr down-sweep 8e-5/7e-5)**: CLOSED. New assignment: PR #858 wd sweep.
- **PR #849 (askeladd, differential τ reweight)**: CLOSED NEGATIVE. τ_y:τ_z ratio inversion harmful. New assignment: PR #857 σ-ladder.
- **PR #846 (edward, STRING rff32)**: CLOSED NEGATIVE. RFF capacity axis FULLY CLOSED.
- **PR #847 (frieren, Lion LR warmup=2ep)**: CLOSED NEGATIVE. LR warmup axis CLOSED.

---

## Active PRs (8 WIP — Round 17)

| PR | Student | Hypothesis | W&B run | Status |
|---|---|---|---|---|
| #823 | nezuko | Surface→vol cross-attention (geometry conditioning, 4-ep→13-ep confirm) | `ghh0s4ne` | EP3=7.12% ✓; step ~37,904, EP4 boundary ~43,459. **Leading geometry experiment.** |
| #855 | frieren | Y-symmetry standalone augmentation — clean 4-ep ablation | `tzfpf31d` | Step ~19,360, EP2=13.6% ✓. **CRITICAL — EP3 boundary ~19,921 imminent (~560 steps away), gate <8%** |
| #857 | askeladd | STRING σ-ladder geometry: Arm A {0.25,0.5,0.75,1.0,2.0} vs Arm B {0.125,0.25,0.5,1.0,2.0} | `qoe1yfm2` | Step ~13,761, EP1=28.1% ✓. EP2 boundary ~16,300 approaching (~2,539 steps) |
| #858 | alphonse | Lion wd sweep: wd=1e-4 (Arm A) vs wd=1e-3 (Arm B, sequential) | `bpq271la` | Step ~9,665, pre-EP1 (boundary 10,864). No val yet. |
| #859 | thorfinn | Model slices sweep: slices=64 (Arm A) vs slices=256 (Arm B) vs SOTA 128 | `thorfinn-slices64-arm-a` | Step ~1,671, pre-EP1. |
| #860 | fern | Tau Y/Z absolute upscaling (preserved ratio): tau_y=2.0/tau_z=2.5 | `dfk2kxtf` | Step ~14,444, approaching EP1. |
| #861 | edward | QK-norm ablation: remove --use-qk-norm from L5 SOTA (never ablated) | — | NEW — awaiting student pick-up |
| #862 | tanjiro | Lion β₂ sweep 4-ep: β₂=0.95 (Arm A) vs β₂=0.999 (Arm B) | — | NEW — awaiting student pick-up |

---

## Current Research Focus

### Theme 1: Geometry Conditioning on Volume Decoder (Issue #717 — highest priority)
- **Surface→vol cross-attention** (nezuko #823, IN FLIGHT): EP3=7.12% ✓. EP4 boundary ~43,459 imminent. **LEADING geometry-conditioning experiment.** Cross-attn learning confirmed (out_proj.weight 0.0→4.99). High potential for EP4 win.
- **SDF skip-connect vol decoder** (#837 tanjiro, BLOCKED by Issue #803): Valid architecture — revisit after `volume_sdf.npy` regeneration.
- ~~**2-layer GELU MLP vol decoder**~~: CLOSED DEAD END (PRs #820 + #828).

### Theme 2: STRING Positional Encoding — σ-ladder (last open STRING axis)
- **σ-ladder geometry sweep** (askeladd #857, IN FLIGHT): Arm A {0.25,0.5,0.75,1.0,2.0} (compressed upper); Arm B {0.125,0.25,0.5,1.0,2.0} (extended lower). First test of σ-ladder internal geometry since SOTA.
- ~~**RFF capacity axis**~~: FULLY CLOSED. rff16=SOTA. rff24/rff32 both fail gate.
- ~~**STRING spectrum pruning**~~: CLOSED. All 5 octaves jointly load-bearing.
- ~~**STRING σ<0.25 axis**~~: CLOSED. σ=0.125 aliases at 65k density.
- ~~**STRING σ-shift up-sweep**~~: CLOSED. σ_max=8.0 consistently hurts.

### Theme 3: Optimizer / Training Dynamics
- **Lion wd sweep** (alphonse #858, IN FLIGHT): wd=1e-4 Arm A approaching EP1. wd=5e-4 is SOTA.
- **Y-symmetry standalone** (frieren #855, IN FLIGHT): **EP3 gate imminent — ~560 steps from boundary ~19,921. Gate: <8%.**
- **Model slices sweep** (thorfinn #859, IN FLIGHT): slices=64 vs 256 vs SOTA 128. Tests attention capacity axis.
- **Tau Y/Z absolute upscaling** (fern #860, IN FLIGHT): tau_y=2.0/tau_z=2.5 (preserved ratio from SOTA 1.5:2.0).
- **QK-norm ablation** (edward #861, NEW): Remove `--use-qk-norm` — never ablated. Cleanest 1-flag delta on L5 SOTA.
- **Lion β₂ sweep 4-ep** (tanjiro #862, NEW): β₂=0.95 vs β₂=0.999. Re-run of closed #850 on correct 4-ep budget.
- ~~**GradNorm**~~: **CONCLUSIVELY CLOSED** — 5 failures across all α (0.1, 0.25, 0.5, 0.75, 1.0). Do not revisit ever.
- ~~**Lion β₁ axis**~~: CLOSED. β₁=0.9 confirmed optimal.
- ~~**EMA decay axis**~~: FULLY EXHAUSTED (#851, EP1=69.852%).
- ~~**LR floor axis**~~: CLOSED (#842).
- ~~**τ_z static-weight axis**~~: CLOSED (#833).
- ~~**Volume-loss-weight axis**~~: CLOSED (#830).
- ~~**τ differential-ratio axis**~~: CLOSED (#849).

---

## Key Diagnostic Findings Established

- **Wall shear z is confirmed training laggard** (#758): r_i=0.01123, weight=1.699. Vol_p is NOT undertrained — gap is OOD generalization.
- **4 OOD test cases dominate vol_pressure** (#767): 92% of squared deviation. Geometry conditioning is the right lever.
- **AB-UPT geometry branch compresses OOD gap** (#626, #802): v2 achieved 3.17×→2.225× OOD gap compression (30% improvement) even at val_abupt=8.563%.
- **FiLM mechanism saturates** (#792): γ saturates at tanh bound 100% from EP4.
- **Depth-scaling CLOSED** (#811): L6 underperforms L5 SOTA by 0.62–0.67pp.
- **τ_y loss-weight axis CLOSED** (#817): τ_y×2.0 made τ_y itself worse.
- **STRING σ=0.25 is load-bearing** (#819): σ=0.25 encodes panel-scale surface detail.
- **Schedule alignment is a confounder** (#805): vol-w=2.0 regression collapses from +1.79pp at EP1 to +0.035pp at EP3.
- **STRING σ<0.25 definitively closed** (#829, #838): σ=0.125 aliases at 65k surface density.
- **Lion β₁=0.9 confirmed optimal** (#841, #839): β₁=0.85 catastrophic.
- **4-ep schedule confound**: Use `--lr-cosine-t-max 13 --epochs 4`. NEVER `--lr-cosine-t-max 4`.
- **13-ep schedule cannot fit in 270-min budget**: Exits at ~step 37k, EP4 boundary ~43k. All tay-track PRs MUST use 4-ep screen.
- **GradNorm CONCLUSIVELY CLOSED**: 5 consecutive failures across all α values. Anti-synergistic with L5/Lion/STRING stack.

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128 (~15.9M params, SOTA config)
- **Depth:** L=5 is optimal. L=6 underperforms. Depth-scaling axis CLOSED.
- **Positional encoding:** STRING-separable (rff_num_features=16, sigmas {0.25,0.5,1.0,2.0,4.0}). All axes CLOSED except σ-ladder internal geometry (#857 in flight).
- **Optimizer:** Lion, lr=9e-5, β₁=0.9 (CLOSED), β₂=0.99 (testing in #862 on 4-ep), wd=5e-4 (testing wd=1e-4 in #858 Arm A)
- **QK-norm:** enabled (ablation in progress: #861 edward)
- **Loss weights:** tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0. All individual axes CLOSED.
- **EMA:** 0.999 (axis FULLY EXHAUSTED)
- **Training budget:** ~270 min. 13-ep schedule CANNOT complete EP4. All tay-track PRs MUST use `--epochs 4 --lr-cosine-t-max 13`.
- **4-ep screen schedule:** `--lr-cosine-t-max 13 --epochs 4` with vol-curriculum `0:16384:1:32768:2:49152:3:65536`
- **VOLUME_X_DIM=4:** (x, y, z, sdf). Channel 4 is precomputed SDF.
- **Heads constraint:** heads MUST be power-of-2 for SDPA/Triton fast paths.
- **`--no-compile-model` required:** torch.compile + DDP NCCL deadlock at step 1.
- **SDF pipeline freeze:** Issue #803 hold — no changes to SDF architecture.

---

## Potential Next Research Directions

### Geometry conditioning (highest priority — Issue #717)

1. **Surface→vol cross-attention** (#823, IN FLIGHT): EP4 imminent. Leading experiment.
2. **SDF skip-connect vol decoder** (#837, BLOCKED): Revisit after Issue #803 resolution.
3. **AdaLN-zero FiLM at block level**: Different from channel-level FiLM (#792 closed). Worth revisiting after Issue #803.
4. **Hierarchical vol decoder** (3-layer MLP with skip): Only if cross-attention (#823) fails.

### Optimizer / training dynamics

5. **QK-norm ablation** (#861 edward, NEW): First clean test of this SOTA flag.
6. **Lion β₂ sweep 4-ep** (#862 tanjiro, NEW): β₂=0.95 vs β₂=0.999 — clean 4-ep budget.
7. **Lion wd sweep** (#858 alphonse, IN FLIGHT): wd=1e-4 approaching EP1.
8. **Y-symmetry standalone** (#855 frieren, IN FLIGHT): EP3 boundary ~imminent.
9. **PCGrad or gradient projection**: If β₂ sweep and dropout both fail, try gradient conflict resolution for τ_y/τ_z. NOTE: NOT implemented — requires code change.
10. **Model slices sweep** (#859 thorfinn, IN FLIGHT): slices=64 vs 256 vs SOTA 128.
11. **Tau Y/Z absolute upscaling** (#860 fern, IN FLIGHT): tau_y=2.0/tau_z=2.5.
12. **Layer-wise LR decay (LLRD)**: Higher lr on later transformer layers. Never tested.
13. **Warm restart cycles within 4-ep budget**: Multiple short cosine cycles. Informative evidence from #827 (frieren) shows current monotone cosine is productive in decay phase — any restart must run a full 13-ep to be comparable.

### STRING — σ-ladder (last remaining axis)

14. **σ-ladder geometry sweep** (#857 askeladd, IN FLIGHT): Arm A {0.25,0.5,0.75,1.0,2.0} vs Arm B {0.125,0.25,0.5,1.0,2.0}. Last untested STRING axis.
15. **Learnable σ (meta-learned RFF sigmas)**: Only if #857 shows positive signal without clear optimum. Requires code change.

### Ensemble refresh

16. **Ensemble pool-25**: After new single-model candidates emerge from current round.

### Bold / plateau-protocol ideas (if Round 17 stalls)

17. **Spectral normalization on attention layers**: Stability regularization orthogonal to QK-norm; targets high-curvature geometry regions.
18. **Frequency-domain loss component**: FFT-based loss term on surface fields — captures aliasing-prone high-frequency τ_y/τ_z patterns.
19. **Stochastic depth / layer drop**: Drop random transformer layers during training; implicit ensembling at inference. Never tested at L5.
20. **Per-channel output projection**: Separate decoder head per physical quantity (surface_p, tau_x, tau_y, tau_z, vol_p). Untested — targets channel-specific bottlenecks.
