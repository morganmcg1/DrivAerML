# SENPAI Research State
- **Date:** 2026-05-08 11:55 UTC (Round in progress — 8 WIP PRs on `tay` advisor branch)
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

---

## Latest Human Researcher Directives

- **Issue #618** (STRING/RoPE): Architecture experiments concluded. All RoPE/Anchor variants closed negative. Direction exhausted at single-model level.
- **Issue #717** (vol_pressure gap): vol_p OOD gap is a data distribution-shift problem. Primary lever: Issue #803 SDF regeneration (blocking — human team). Secondary: data-side augmentation.
- **Issue #803** (volume_sdf.npy): Awaiting human team delivery of regenerated SDF. BLOCKING for geometry-conditioning experiments.

---

## Active PRs (8 WIP — current round)

| PR | Student | Hypothesis | W&B run | Status |
|---|---|---|---|---|
| #823 | nezuko | Surface→volume cross-attention (13-ep confirm) | `ghh0s4ne` | EP2 PASS 8.148% (vs SOTA 7.94% +0.21pp) — EP3 in progress |
| #848 | alphonse | Lion lr down-sweep (8e-5 / 7e-5) | Arm A `hauwt4fv` | Arm A killed at EP3=8.36% (gate miss); Arm B (lr=7e-5) launching |
| #849 | askeladd | Differential τy/τz (2.0/1.5; 2.5/1.5) | Arm A `cww5o53v` | Arm A EP3=8.29% (gate miss by 0.29%); EP4 imminent |
| #850 | tanjiro | Lion β₂ sweep (0.95 vs 0.999) | Arm A `dxole713` | EP1 PASS 25.49%; EP2 imminent (~step 21729) |
| #852 | fern | GradNorm α=0.75 on L5 SOTA tay screen | — | **CONFLICTING** — sent back for rebase onto origin/tay |
| #853 | thorfinn | Model dropout sweep (0.05 / 0.10) | Arm A `baxqh2ok` | Arm A launched ~11:16Z; EP1 ETA ~12:33Z |
| #854 | edward | GradNorm α=0.1 on L5 SOTA tay screen | `f1l3m752` | Launched; ~24% through EP1 |
| #855 | frieren | Y-symmetry augmentation standalone | — | y-sym ported from drivaerml-long-20260504; launched |

---

## Current Research Focus

### Theme 1: Optimizer Axis Exploration
- **Lion lr down-sweep** (alphonse #848): lr=8e-5 missed EP3 gate (8.36%); falling back to lr=7e-5 in Arm B.
- **Lion β₂** (tanjiro #850): β₂=0.95 vs 0.999 — momentum smoothing axis. EP1 clean.

### Theme 2: GradNorm α Sweep
- **α=0.1** (edward #854): aggressive gradient balancing extreme.
- **α=0.75** (fern #852): conservative tuning around α=0.5 SOTA. **Currently blocked on rebase.**
- α=1.0 already known to crash; α=0.25 worse than α=0.5.

### Theme 3: Wall-Shear τ Reweighting
- **τy=2.0 / τz=1.5** (askeladd #849 Arm A): differential reweight; EP3 close to gate, EP4 will tell.
- **τy=2.5 / τz=1.5** (askeladd #849 Arm B): queued.

### Theme 4: Volume / Geometry
- **Surface→volume cross-attention 13-ep** (nezuko #823): on-pace, +0.21pp behind SOTA at EP2.

### Theme 5: Regularization
- **Model dropout sweep** (thorfinn #853): 0.05 / 0.10 — capacity-vs-overfit lever, untested at L5 SOTA.
- **Y-symmetry augmentation** (frieren #855): bilateral symmetry prior; previously baked into long runs only, now standalone.

---

## Key Architectural Constraints Established

- **SOTA config (PR #592):** L=5, hidden=512, heads=4, slices=128, Lion lr=9e-5, wd=5e-4, β₁=0.9, β₂=0.99, EMA=0.999, STRING-sep RFF σ∈{0.25,0.5,1.0,2.0,4.0}
- **Correct training unit:** 4-epoch curriculum `--vol-points-schedule "0:16384:1:32768:2:49152:3:65536"` — 13-ep schedule CANNOT fit in 270-min budget
- **Kill gates:** EP1 <30%, EP2 <16%, EP3 <8%, EP4 ≤6.5985%
- **Tay-stack flags** (only on `tay`, not `drivaerml-long-20260504`): `--pos-encoding-mode string_separable`, `--use-qk-norm`, `--rff-num-features`, `--rff-init-sigmas`, `--tau-y-loss-weight`, `--tau-z-loss-weight`, `--vol-points-schedule`, `--gradnorm-mode ema_proxy`, `--gradnorm-min-weight`
- **Y-symmetry flags** (`--use-y-symmetry-aug`, `--y-symmetry-aug-prob`): originally on `drivaerml-long-20260504` only; ported to `tay` on commit `87a178e` (frieren PR #855).
- **vol_p OOD gap:** ~7.5pp val→test gap is DATA DISTRIBUTION-SHIFT, not model capacity
- **τ_y gap:** architectural, NOT supervision-strength
- **GradNorm:** α=0.5 SOTA (PR #740 wave test 7.5195%); α=0.25 worse; α=1.0 crashes

---

## Potential Next Research Directions

### Optimizer
1. If lr=7e-5 (alphonse #848 Arm B) passes → close lr down-sweep and push lr=1e-4 / 1.2e-4 up-sweep
2. If β₂=0.95 (tanjiro #850) wins → try β₂=0.9; if loses → β₂=0.99 confirmed optimal
3. Lion β₂=0.999 still pending revisit at L5 SOTA stack

### GradNorm α
4. α=0.1 + α=0.75 combined with current α=0.5 SOTA give a 4-point parabola — fit and propose optimum
5. GradNorm warmup: static for EP1, GradNorm from EP2 onward

### Wall-shear τ
6. If τy=2.0/τz=1.5 (askeladd) marginal → try Arm B τy=2.5/τz=1.5
7. If both fail → close differential τ axis, escalate to per-channel decoder head

### Volume / Geometry (Issue #717, highest EV)
8. Pending Issue #803 SDF regeneration → resume geometry-branch + FiLM modulation experiments
9. If nezuko xattn 13-ep beats SOTA → escalate to per-channel xattn module
10. Per-channel output projection: separate decoder head per physical quantity

### Data augmentation
11. If frieren y-sym standalone wins → composite with rotation, jitter, noise
12. ShapeNet-style point dropout / occlusion augmentation

### Regularization
13. If thorfinn dropout wins → DropPath scheduling; stochastic depth
14. Layer-wise lr decay (LLRD): higher lr on later layers

### Plateau watch
- 5+ consecutive negative results would trigger plateau protocol — current round still has live arms.
