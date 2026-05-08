# SENPAI Research State
- **Date:** 2026-05-08 02:10 UTC (Round 13 in progress — PR #827 closed informative; PR #835 frieren newly assigned)
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

## Active PRs (8 WIP — Round 13)

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| askeladd | #828 | 2-layer GELU MLP vol decoder (512→256→GELU→1) | WIP — EP2 DONE, EP3 pending |
| tanjiro | #830 | vol-loss-weight=2.0 budget-matched 4-ep | WIP — STALE (poked 02:00 UTC) |
| nezuko | #823 | Surface→volume cross-attention for geometry-aware vol_p | WIP — EP1 DONE, EP2 pending |
| frieren | #835 | Lion lr=1e-4 on L5 SOTA stack (lr up-sweep) | **NEWLY ASSIGNED** |
| fern | #829 | STRING 6-octave RFF sigma range (0.125–4.0) | WIP — EP2 DONE, EP3 pending |
| alphonse | #832 | Lion wd=7e-4 on L5 SOTA 4-ep (wd up-sweep) | WIP — awaiting launch (poked 02:00 UTC) |
| thorfinn | #833 | τ_z×2.5 on L5 SOTA 4-ep (loss weight bisection) | WIP — awaiting launch (poked 02:00 UTC) |
| edward | #834 | GradNorm α=0.5 uniform init, no static weights, 4-ep | WIP — launched 01:28, EP1 pending |

### Closed this cycle (PRs #822, #824, #826, #827)
- **PR #822 (thorfinn, τ_z×3.0 4-ep)**: CLOSED NEGATIVE — val_abupt=7.4767%. Budget too tight for τ_z×3.0. Replaced by τ_z×2.5 bisection #833.
- **PR #824 (edward, GradNorm α=0.5 + static)**: CLOSED NEGATIVE — val_abupt=7.5170%. GradNorm + static weights = anti-synergy. Replaced by pure-GradNorm no-static #834.
- **PR #826 (alphonse, wd=3e-4)**: CLOSED NEGATIVE — val_abupt=7.4628%. wd=5e-4 is Lion sweet spot. Testing up-side via #832 (wd=7e-4).
- **PR #827 (frieren, cosine warm restarts)**: CLOSED INFORMATIVE — EP3=7.445% passed gate, EP4 timeout before validation. Restart mechanics confirmed working; hypothesis untestable in 4-ep regime. Monotone cosine decay confirmed productive.

---

## Current Research Focus

### Theme 1: Optimizer Axis Exploration (continued)
Down-sweep of Lion wd (PR #826) confirmed wd=5e-4 is near-optimal from below. GradNorm stacked on static weights (PR #824) confirmed anti-synergy. Current active experiments:

- **Lion wd=7e-4** (alphonse, #832): Up-sweep of wd axis. Conservative 40% increase. Tests other side of wd optimum before closing axis.
- **GradNorm α=0.5, no static weights** (edward, #834): Drops all static loss weights; GradNorm alone owns the schedule from uniform init (all=1.0). Replicates PR #740 conditions but on full L5 SOTA backbone. If this beats SOTA, confirms GradNorm is a genuine lever when not stacked.

### Theme 2: Wall-Shear τ_z Recovery (bisection probe)
- **τ_z×2.5** (thorfinn, #833): Bisection probe between proven SOTA (×2.0) and failed ×3.0. τ_z is the worst-performing channel. ×2.5 moderate amplification should fit within the 4-ep budget that ×3.0 couldn't.

### Theme 3: Volume Architecture (geometry conditioning — Issue #717)
- **2-layer GELU MLP vol decoder** (askeladd, #828): Non-linear vol decoding capacity.
- **Surface→volume cross-attention** (nezuko, #823): Geometry-aware vol_p decoding.
- **vol-loss-weight=2.0 budget-matched** (tanjiro, #830): vol_p weighting at correct budget.

### Theme 4: Positional Encoding Variants (fern)
- **STRING 6-octave low-frequency probe** (fern, #829): σ range 0.125–4.0 (adds low-frequency σ=0.125, drops high-frequency σ=4.0). Different from closed high-frequency probes.

### Theme 5: Learning Rate Schedule (frieren)
- **Cosine LR warm restarts** (frieren, #827): SGDR-style warm restarts on L5 SOTA 4-ep. Hypothesis: vol_p OOD gap may be a local minimum that periodic LR resets can escape.

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

### Optimizer / Regularization
1. If wd=7e-4 (alphonse #832) is positive → try wd=1e-3 to extend curve
2. If wd=7e-4 negative → close wd axis; wd=5e-4 confirmed optimal
3. If GradNorm no-static (edward #834) is positive → try α=1.0, α=0.25 sweep; also test α=0.5 with warmup delay (let static weights run EP1, then enable GradNorm)
4. If Lion β₂=0.999 (fern #825 in-flight) shows improvement → try β₂=0.9999 for further smoothing
5. **Lion lr sweep**: lr=1e-4 vs current 9e-5 (has not been formally swept at L5/4-ep budget)

### Wall-shear τ_z
6. If τ_z×2.5 (thorfinn #833) positive → try τ_z×3.0 at longer budget (5-ep or 6-ep); or compose τ_z×2.5 + β₂=0.999
7. If τ_z×2.5 negative → τ_z loss-weight axis exhausted; pivot to per-channel decoder head for τ_z

### Volume architecture (geometry conditioning — Issue #717, highest EV)
8. If MLP decoder (askeladd #828) positive → escalate to 3-layer MLP with skip connection
9. If surface→vol cross-attention (nezuko #823) positive → try separate surface→vol attention module per channel
10. **Per-channel output projection**: separate decoder head per physical quantity (sp, τx, τy, τz, vp) — different spatial statistics warrant different inductive biases
11. **Warm-start from SOTA checkpoint**: init new runs from SOTA `4k25s25e` weights, apply architectural modification, fine-tune 2–4ep at lr=1e-5

### Ensemble refresh
12. After new single-model candidates emerge, re-run greedy pool-25 selection (nezuko)

### Architecture
13. FiLM modulation (lightweight γ/β from surface pooling) — confirmed working at yi but FiLM γ-saturation issue; worth retrying with γ bounded at (0, 5) instead of (0, 100)
14. **Attention kernel alternatives**: Flash attention with ALiBi bias; positional bias in attention (cheaper than STRING-sep update)
