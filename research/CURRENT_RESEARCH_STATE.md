# SENPAI Research State
- 2026-05-04 (Round 37 — advisor cycle check, updated: 2026-05-04 20:30 UTC — all 16 students active, 19 WIP PRs [#656 closed → replaced by #674], no PRs ready for review, no human issues)
- **CURRENT SOTA (yi branch): PR #637 (fern, extended low-LR training at lr=1e-5 from t4qaysur checkpoint) — val_abupt 7.5373%**. Active yi merge bar: **7.5373%** (run `vzprvtaw`).
- **PR #576 (frieren STRING-sep learnable PE + Lion) MERGED to yi.** The `--learnable-pe` flag is available in yi `train.py`. Requires `--no-compile-model` (torch.compile inductor broadcast bug).
- **PR #637 (fern extended low-LR continuation) MERGED to yi.** New SOTA 7.5373%.
- **Tay SOTA (reference track, not yi): PR #511 (edward) — val_abupt 7.013% / test_abupt 8.313%** (tay branch only).

## Most Recent Research Direction from Human Researcher Team

**Issue #18** (open, standing directive):
> "Stop incremental tuning. Be bold with architecture. Empower students to replace the model backbone while maintaining logging/validation/checkpointing."

## Current Baseline: PR #637 (fern) — yi — val_abupt 7.5373%

**Active yi merge bar: val_abupt < 7.5373%** (run `vzprvtaw`)

| Metric | yi bar (PR #637) | Tay SOTA ref (PR #511 test) | AB-UPT target | Gap |
|---|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | 7.5373 | 8.313 | — | — |
| `wall_shear_y_rel_l2_pct` | 9.8691 | 8.582 | 3.65 | **2.05×** |
| `wall_shear_z_rel_l2_pct` | 11.2477 | 9.927 | 3.63 | **2.45×** |
| `volume_pressure_rel_l2_pct` | — | 11.867 | 6.08 | **~1.78×** |

Dominant open gaps: τ_z (2.45×), τ_y (2.05×), vol_p (~1.78×).

## Active WIP PRs (Round 37 — 19 PRs as of 2026-05-04 20:30 UTC)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #616 | chihiro | Perceiver-IO backbone replacement (bold arch) | Running EP1 |
| #622 | thorfinn | SDF-proximity volume loss weighting (α=2.0, σ=0.1) | Running |
| #628 | edward | τ_y/τ_z loss weight sweep: W=4 and W=6 arms | WIP |
| #635 | kohaku | RANS Laplace regularizer on far-field volume pressure | WIP |
| #636 | gilbert | Inverse local point density loss weighting | WIP |
| #638 | tanjiro | Model dropout regularization sweep (p=0.05 vs p=0.1) | WIP |
| #661 | haku | RFF surface geometry lift — isolated single-variable test (dim=64,128) | WIP (reassigned from closed #642) |
| #645 | norman | 6L/512d clean depth ablation — isolated (no k1_k2, no β-NLL, lr-warmup-epochs=2) | WIP |
| #646 | alphonse | Asymmetric τ_y/τ_z weighting + curvature-focal loss | WIP |
| #652 | frieren | Muon optimizer on full yi stack (STRING-sep + grad-ema + β-NLL + k1_k2) | WIP |
| #653 | askeladd | 1-cycle LR with 1e-3 peak (OneCycleLR) | WIP |
| #654 | emma | Dual-Tower surface+volume cross-attention encoder | WIP |
| #655 | senku | 6L/512d depth scale-up on full yi stack | WIP |
| #674 | violet | Surface normal vectors (nₓ,nᵧ,n_z) as surface input features (τ_y/τ_z attack) | WIP |
| #657 | fern | Ultra-low LR 1e-6 continuation from PR #637 yi best checkpoint | WIP |
| #658 | nezuko | SWA staged trajectory: resume PR #637 SOTA, 2ep at lr=5e-6, swa-start-fraction=0.0 | WIP |
| #659 | norman | Width-over-depth 4L/768d/12h bold capacity increase (Issue #18) | WIP |
| #662 | chihiro | Compose surface curvature features (κ₁/κ₂) with full yi SOTA stack | WIP |
| #663 | kohaku | Stochastic depth (DropPath) sweep: prob=0.05 vs 0.10 on yi SOTA | WIP |

### Recently closed this cycle:
| PR | Student | Result |
|---|---|---|
| #642 | haku | CLOSED — no training activity after 5 escalation nudges; implementation complete; hypothesis reassigned to PR #661 |
| #631 | violet | CLOSED — CosineAnnealingWarmRestarts null in cold-start: val_abupt=10.74% (+30% vs SOTA). Within-cycle LR decay dominates 3-epoch budget. |
| #634 | nezuko | CLOSED positive mechanism, below bar — SWA improved ALL metrics vs best-ckpt by 0.61pp (τ_y +0.87pp, τ_z +0.79pp). Needs staged trajectory to beat bar. |
| #656 | violet | CLOSED — Multi-EMA ensemble (3 shadows at decays 0.9995/0.999/0.995): val_abupt=9.6146% (+2.077pp vs bar). Stale early-epoch weights at <9 half-life budget drag ensemble down. τ_y +2.77pp, τ_z +2.67pp. Reassigned to PR #674 (surface normals). |
| #633 | norman | CLOSED — 6L/512d depth scale-up: EP1 abort (18.9% >> 15% gate). Reassigned to PR #645. |

## Research Themes (Round 37 priority order)

1. **Lock in STRING-sep + Lion combination** (baseline from PR #576/637) — anchor run combining both known wins
2. **τ_y/τ_z structural attack (multi-prong):**
   - Asymmetric τ_y/τ_z weighting + curvature-focal loss (PR #646 alphonse)
   - τ_y/τ_z loss weight sweep: W=4 and W=6 (PR #628 edward)
   - Surface normal vectors as surface input channels (PR #674 violet) — directional geometry for τ_y/τ_z decomposition
   - RFF surface geometry lift isolated test (PR #661 haku)
   - Compose surface curvature features κ₁/κ₂ (PR #662 chihiro)
3. **Volume pressure gap attack:**
   - SDF-proximity volume loss weighting (PR #622 thorfinn) — near-surface upweight
   - RANS Laplace regularizer (PR #635 kohaku) — physics-informed far-field ∇²p ≈ 0
   - Inverse local point density loss weighting (PR #636 gilbert) — sparse region emphasis

4. **Training dynamics / optimization:**
   - Ultra-low LR 1e-6 continuation (PR #657 fern) — pushing further from current SOTA
   - 1-cycle LR with 1e-3 peak (PR #653 askeladd) — aggressive warm-up
   - SWA staged trajectory (PR #658 nezuko) — warm restart from SOTA checkpoint
   - Muon optimizer (PR #652 frieren) — second-order-ish update rule

5. **Architecture (bold, Issue #18):**
   - Perceiver-IO backbone replacement (PR #616 chihiro)
   - Width-over-depth 4L/768d/12h (PR #659 norman)
   - Dual-Tower cross-attention (PR #654 emma)
   - 6L/512d depth scale-up full stack (PR #655 senku)

6. **Pending follow-up ideas (from closed PRs):**
   - **Warm restarts as fine-tuning** (from PR #631 violet): T_0=1, lr=1e-5, resume from SOTA checkpoint. Different hypothesis from cold-start. (unassigned)

## Known Null Results (do not repeat)

- Volume coord transforms: signed-log, asinh on volume stream input — flat (#520)
- BF16→FP32 final epoch — slight regression at <3 epoch horizon (#522)
- BERT-direction LLRD (decay < 1 on later layers) — hurts from-scratch training (#519 Arm A 11.17%)
- Model-only checkpoint resume — EMA/optimizer reset undoes gains (#472)
- Per-axis static loss weight sweeps (#244, #454)
- Uncertainty weighting Kendall (#496) — inverts gradient for lagging tasks
- LLRD: tentatively null for all 3 arms if B and C also miss by >10%
- Coordinate frame transforms on vol stream (#520): signed-log, asinh flat
- FP32 final-epoch upcast (#522): slight regression
- Flow-aligned tau v1 (PR #613) — null result, no improvement
- Principal surface curvatures k1_k2 (PR #580) — infrastructure-only (plumbing merged, metric bar unchanged at 3-epoch budget)
- ADOPT optimizer (PR #588) — completed but below bar
- SDF gradient direction as aux input (PR #589 tanjiro) — above bar
- CosineAnnealingWarmRestarts cold-start (PR #631 violet) — null in ≤3 epoch budget: LR decay penalty dominates
- SWA from-scratch 3-epoch (PR #634 nezuko) — positive mechanism but wrong regime; floor too high
- Multi-shadow EMA averaging at <20 half-life budget (PR #656 violet) — stale early-training weights dragged all metrics; τ_y +2.77pp, τ_z +2.67pp worst regressions

## Fleet Stability Constants

- Lion optimizer: lr=1e-4, warmup=1 epoch, clip=0.5, wd=5e-4
- 4×H100/H200 DDP: ~1.25 s/it × 5442 steps/epoch → ~113–191 min/epoch → ~2–3 epochs in 6h budget
- `--no-compile-model` required when `--learnable-pe` is active (torch.compile inductor bug with multi-axis broadcast in ContinuousSincosEmbed)
- Flag name: `--clip-grad-norm 0.5` (NOT `--grad-clip-norm`) — verified via `python train.py --help`
- OOM mitigation: chunked cdist (chunk_size=2048–4096) for pairwise distance computations over 65k-point clouds

## Recommended Base Config (from PR #576 + #637 SOTA)

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --agent <student> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --clip-grad-norm 0.5 --no-compile-model --learnable-pe \
  --grad-ema-alpha 0.5 \
  --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --model-slices 128 --ema-decay 0.999 --lr-warmup-epochs 1
```
