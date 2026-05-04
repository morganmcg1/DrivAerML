# SENPAI Research State
- 2026-05-04 23:30 UTC (Round 36 — advisor cycle, updated)
- **CURRENT SOTA (yi branch): PR #576 (frieren, STRING-sep PE + Lion lr=1e-4 clip=0.5) — val_abupt 8.2528%**. Active yi merge bar: **8.2528%** (run `t4qaysur`).
- **PR #576 (frieren STRING-sep learnable PE + Lion) MERGED to yi.** The `--learnable-pe` flag is available in yi `train.py`. Requires `--no-compile-model` (torch.compile inductor broadcast bug).
- **Tay SOTA (reference track, not yi): PR #511 (edward) — val_abupt 7.013% / test_abupt 8.313%** (tay branch only).

## Most Recent Research Direction from Human Researcher Team

**Issue #18** (open, standing directive):
> "Stop incremental tuning. Be bold with architecture. Empower students to replace the model backbone while maintaining logging/validation/checkpointing."

## Current Baseline: PR #576 (frieren) — yi — val_abupt 8.2528%

**Active yi merge bar: val_abupt < 8.2528%** (run `t4qaysur`)

| Metric | yi bar (val, PR #576) | Tay SOTA ref (PR #511 test) | AB-UPT target | Gap |
|---|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | 8.2528 | 8.313 | — | — |
| `surface_pressure_rel_l2_pct` | — | 4.271 | 3.82 | 1.12× |
| `wall_shear_rel_l2_pct` | — | 7.786 | 7.29 | 1.07× |
| `volume_pressure_rel_l2_pct` | — | 11.867 | 6.08 | **1.95×** |
| `wall_shear_y_rel_l2_pct` | — | 8.582 | 3.65 | **2.35×** |
| `wall_shear_z_rel_l2_pct` | — | 9.927 | 3.63 | **2.73×** |

Dominant open gaps: τ_z (2.73×), τ_y (2.35×), vol_p (1.95×).

## Active WIP PRs (Round 36 — as of 2026-05-04 22:00 UTC)

### In-flight experiments (confirmed W&B runs running):
| PR | Student | Hypothesis | W&B Run | Status |
|---|---|---|---|---|
| #622 | thorfinn | SDF-proximity volume loss weighting (α=2.0, σ=0.1) | w51b83zx | Running EP2 (97% done) |
| #627 | frieren | Tangent-frame wall-shear head | qrdhyxbs | Running EP1 (65%) |
| #616 | chihiro | Perceiver-IO backbone replacement (bold arch) | achht6dr | Running EP1 (72%) |

### Advisor-prompted PRs (WIP, awaiting results):
| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #628 | edward | τ_y/τ_z loss weight sweep: W=4 and W=6 arms | WIP |
| #629 | askeladd | asinh target normalization for wall shear τ_y/τ_z heavy tails | WIP |
| #630 | emma | Lookahead wrapper around Lion optimizer | WIP |
| #631 | violet | Cosine annealing warm restarts (Arm C: T_0=2, T_mult=2) | WIP — Arm C pending |
| #632 | senku | Wake-region volume pressure upweighting (2× downstream) | WIP |
| #634 | nezuko | Stochastic Weight Averaging (SWA) over last 30% | WIP |
| #635 | kohaku | RANS Laplace regularizer on far-field volume pressure | WIP |
| #636 | gilbert | Inverse local point density loss weighting | WIP |
| #637 | fern | Extended low-LR continued training from yi best checkpoint | WIP |
| #638 | tanjiro | Model dropout regularization sweep (p=0.05 vs p=0.1) | WIP |
| #639 | alphonse | 4L/768d width scale-up | WIP |
| #640 | nezuko | NorMuon optimizer on L=5 SOTA stack | WIP |
| #641 | askeladd | Flow-aligned tau: predict wall shear in local surface tangent frame | WIP |
| #642 | haku | Fourier surface geometry features for tau_y/z gap (RFF input lift) | WIP |
| #624 | alphonse | Point-level pre-slice STRING rotation in Transolver | WIP |
| #625 | askeladd | No-slice Anchor-STRING / AB-UPT-lite (Issue #618 Exp 3) | WIP |
| #626 | frieren | Full AB-UPT-style geometry branch (Issue #618 Exp 4) | WIP |
| #621 | nezuko | Slice-centroid STRING-RoPE for Transolver attention | WIP |
| #623 | tanjiro | Stronger bounded tau weights τ_y×1.5 / τ_z×2.0 | WIP |

### Recently closed this cycle:
| PR | Student | Result |
|---|---|---|
| #633 | norman | CLOSED — 6L/512d depth scale-up: EP1 abort (18.9% >> 15% gate). Reassigned to PR #645. |

### Norman reassigned:
| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #645 | norman | 6L/512d clean depth ablation — isolated (no k1_k2, no β-NLL, lr-warmup-epochs=2) | WIP |

## Research Themes (Round 36 priority order)

1. **Bold architecture replacement** (Issue #18 mandate):
   - Perceiver-IO backbone (PR #616 chihiro) — full backbone swap, 32.8M params
   - AB-UPT-style geometry branch (PR #626 frieren) — Issue #618 Exp 4
   - No-slice Anchor-STRING (PR #625 askeladd) — Issue #618 Exp 3
   - NorMuon optimizer (PR #640 nezuko) — second-order orthogonalization
   
2. **τ_y/τ_z structural attack (multi-prong):**
   - Tangent-frame wall-shear head (PR #627 frieren) — coordinate frame decomposition
   - Flow-aligned tau frame (PR #641 askeladd) — local tangent frame prediction
   - asinh target normalization (PR #629 alphonse) — heavy-tail compression
   - τ_y/τ_z loss weight sweep W=4,6 (PR #628 thorfinn) — direct upweighting
   - Stronger bounded tau weights τ_z×2.0 (PR #623 tanjiro) — aggressive weighting

3. **Volume pressure gap attack:**
   - SDF-proximity volume loss weighting (PR #622 thorfinn) — near-surface upweight
   - Wake-region upweighting 2× downstream (PR #632 fern) — physics-informed spatial weight
   - RANS Laplace regularizer (PR #635 senku) — physics-informed far-field ∇²p ≈ 0

4. **Architecture/input features:**
   - 6L/512d depth scale-up (PR #633 gilbert) — capacity increase
   - 4L/768d width scale-up (PR #639 haku) — hidden dim expansion
   - Fourier surface geometry RFF features (PR #642 haku) — input lift
   - Point-level pre-slice STRING rotation (PR #624 alphonse) — Transolver PE
   - Slice-centroid STRING-RoPE (PR #621 nezuko) — attention geometry

5. **Training dynamics:**
   - Extended low-LR training from yi checkpoint (PR #637 edward) — LR floor exploitation
   - Cosine annealing warm restarts T_0=3 (PR #631 emma) — LR schedule
   - Lookahead + Lion (PR #630 violet) — optimizer wrapper
   - SWA over last 30% (PR #634 tanjiro) — weight averaging
   - Model dropout p=0.05/0.1 (PR #638 kohaku) — regularization

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
- ADOPT optimizer (PR #588) — completed but below bar (status pending review)
- SDF gradient direction as aux input (PR #589 tanjiro) — EP2 24.9506%, compare to grad-ema control EP2 26.99% → marginal improvement, still above bar

## Fleet Stability Constants

- Lion optimizer: lr=1e-4, warmup=1 epoch, clip=0.5, wd=5e-4
- 4×H100/H200 DDP: ~1.25 s/it × 5442 steps/epoch → ~113–191 min/epoch → ~2–3 epochs in 6h budget
- `--no-compile-model` required when `--learnable-pe` is active (torch.compile inductor bug with multi-axis broadcast in ContinuousSincosEmbed)
- Flag name: `--clip-grad-norm 0.5` (NOT `--grad-clip-norm`) — verified via `python train.py --help`
- OOM mitigation: chunked cdist (chunk_size=2048–4096) for pairwise distance computations over 65k-point clouds
- EP1 gate for bold arch experiments: ≤15% = continue, 15–25% = marginal, >25% = close immediately
- EP2 kill gate for standard experiments: ≤12% = continue to full run, >12% = kill arm

## Recommended Base Config (from PR #576 = STRING-sep + Lion + grad-ema)

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

Optional composition flags (validated on yi, available but not required):
- `--surface-curvature-features k1_k2` — adds κ₁, κ₂ as 9-channel surface input
- `--beta-nll-beta 0.5` — heteroscedastic NLL loss (gains came primarily from extra training)
