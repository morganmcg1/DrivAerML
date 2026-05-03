# SENPAI Research State
- 2026-05-03 22:00 UTC (Round 34 — advisor cycle)
- **CURRENT SOTA (yi branch): PR #517 (askeladd, Lion lr=1e-4 clip=0.5) — val_abupt 9.032%**. Active yi merge bar: **9.032%** (run `brat65z4`).
- **PR #490 (frieren STRING-sep learnable PE) MERGED to yi at 15:48 UTC 2026-05-03.** The `--learnable-pe` flag is now available in yi `train.py`. PR #517 was launched BEFORE #490 merged, so the current baseline does NOT yet include STRING-sep PE. The next highest-priority merge is a from-scratch run combining STRING-sep + Lion lr=1e-4 clip=0.5 (PR #539 frieren, just assigned).
- **Tay SOTA (reference track, not yi): PR #511 (edward) — val_abupt 7.013% / test_abupt 8.313%** (tay branch only, not reproducible on yi standalone).

## Key Merges Since Round 30

| PR | Student | What | Impact |
|---|---|---|---|
| #490 | frieren | STRING-sep learnable PE port to yi | Mechanism now on yi; from-scratch combination pending |
| #517 | askeladd | Lion lr=1e-4 clip=0.5 (new yi bar) | val_abupt 9.032% (from 9.039%) |

## Most Recent Research Direction from Human Researcher Team

**Issue #18** (open, standing directive):
> "Stop incremental tuning. Be bold with architecture. Empower students to replace the model backbone while maintaining logging/validation/checkpointing. Mine noam/radford branches."

**Issue #252** (open): "Get inspired by Modded-NanoGPT" — already addressed Rounds 15-25. Muon optimizer (edward #472), sigmoid slice attention (senku #521), LLRD (gilbert #519) are the current modded-NanoGPT follow-ons.

## Current Baseline: PR #517 (askeladd) — yi — val_abupt 9.032%

**Active yi merge bar: val_abupt < 9.032%** (run `brat65z4`)

| Metric | yi bar (test, PR #517) | Tay SOTA ref (PR #511 test) | AB-UPT target |
|---|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | 10.119 | 8.313 | — |
| `surface_pressure_rel_l2_pct` | — | 4.271 | 3.82 (1.12×) |
| `wall_shear_rel_l2_pct` | — | 7.786 | 7.29 (1.07×) |
| `volume_pressure_rel_l2_pct` | — | 11.867 | 6.08 (**1.95×**) |
| `wall_shear_x_rel_l2_pct` | — | 6.918 | 5.35 (1.29×) |
| `wall_shear_y_rel_l2_pct` | — | 8.582 | 3.65 (**2.35×**) |
| `wall_shear_z_rel_l2_pct` | — | 9.927 | 3.63 (**2.73×**) |

Dominant open gaps: τ_z (2.73×), τ_y (2.35×), vol_p (1.95×).

## Architecture Configuration (Current Best)

```bash
torchrun --standalone --nproc_per_node=4 target/train.py \
  --learnable-pe \          # PR #490, now on yi
  --optimizer lion \
  --lr 1e-4 \
  --grad-clip-norm 0.5 \
  --weight-decay 5e-4 \
  --lr-warmup-epochs 1 \
  --ema-decay 0.999 \
  --model-layers 4 \
  --model-hidden-dim 512 \
  --model-heads 8 \
  --model-slices 128 \
  --batch-size 4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --epochs 50
```

Note: `--learnable-pe` is available on yi (PR #490) but not yet confirmed in a clean from-scratch run at 9.032% config level. PR #539 (frieren) will lock this in.

## Active WIP PRs (Round 33/34 — 16 PRs)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #547 | tanjiro | Joint 2D Cholesky NLL head for τ_y/τ_z | NEW Round 33 |
| #546 | violet | Online hard example mining for τ_y/τ_z | NEW Round 33 |
| #545 | senku | RoPE on Transolver slice tokens (3D centroid keys) | NEW Round 33 |
| #544 | fern | y-symmetry paired loss + test-time mirror ensemble | NEW Round 33 |
| #543 | haku | Principal curvatures (H,K) as surface features | NEW Round 33 |
| #542 | alphonse | Laplace far-field soft penalty on vol pressure | NEW Round 33 |
| #541 | askeladd | Streamline-aligned wall-shear target frame | NEW Round 33 |
| #540 | emma | GradNorm adaptive τ_y/τ_z up-weighting | NEW Round 33 |
| #539 | frieren | STRING-sep + Lion lr=1e-4 clip=0.5 from scratch | NEW Round 33 |
| #528 | nezuko | 1-cycle LR pct_start=0.05 vs cosine ep3 | WIP |
| #522 | thorfinn | BF16→FP32 last-2-epoch precision | WIP |
| #520 | norman | Volume-only coord transform (signed-log/asinh) | WIP |
| #519 | gilbert | LLRD sweep (decay=0.75/0.95) | WIP |
| #518 | kohaku | Loss-side asinh wallshear (delta sweep) | WIP |
| #478 | chihiro | Per-step cosine LR schedule | WIP |
| #548 | edward | **Muon@1e-3 + STRING-sep PE composition (both wins from scratch)** | **NEW — just assigned** |

## Key Closed Dead Ends (summary — do not repeat)

- Optimizer: Adam/AdamW (replaced by Lion). SAM. AGC. β2 sweep (null). Muon single-GPU (promising → promoted to PR #472).
- Architecture: Perceiver-IO replacement (PR #122), Mamba-2 SSM (PR #45), K-NN local attention (PR #197), dual-stream cross-attention (PR #421), multi-scale radius pooling.
- Loss formulation: Asinh wall-shear scale sweep (PR #485, 3 scales, null), Huber δ sweep no-tangential (PR #440, negative), magnitude+direction decomposition (PR #200), uncertainty weighting homoscedastic Kendall (PR #496, mechanism inverted — DOWN-weights lagging tasks).
- Input features: Surface-tangent wall-shear input (PR #312/419, closed), NIG evidential regression (PR #38).
- LR: 1-cycle 1e-3 (PR #125, closed), WSD linear warmdown (PR #262, null), cosine T_max mismatch (PR #478 active — per-step variant).
- Data: Coord jitter (PR #314, null), surface-loss-weight static sweeps (PRs #244, #454, null).
- Regularization: SWA (PR #527 fern, closed), DropPath (PR #338, frieren, null), stochastic depth+dropout (PR #479, thorfinn).
- EMA: Decay ramp 0.99→0.9999 (PR #430, null). Model soup (PR #491, frieren, null).
- Grad clip: Structural sweep (PR #431, null — only tighter clip than 0.5 hurts).
- Augmentation: Random mirror aug train-time (PR #225, haku, null).
- PE: Omega-bank sweep (PR #183, done → PR #490 STRING-sep is the follow-through). Per-axis coord normalization (PR #473, alphonse).

## Research Themes (Round 33 priority order)

1. **Lock in STRING-sep + Lion combination** (PR #539 frieren) — highest confidence, lowest risk, drops bar to ~8.1%.
2. **Structural τ_y/τ_z gap attack:**
   - Streamline frame target rotation (PR #541 askeladd) — coordinate-frame fix
   - GradNorm adaptive task weighting (PR #540 emma) — principled up-weighting
   - y-symmetry paired loss + TTA (PR #544 fern) — equivariance enforcement
   - PR #530 tanjiro (tangent-frame head — currently running, converging)
   - PR #529 violet (stream-normal ws weight — running)
3. **Volume pressure gap attack:**
   - Laplace far-field penalty (PR #542 alphonse) — physics-informed regularizer
   - PR #520 norman (coord transform — running)
4. **Architecture improvements:**
   - PR #521 senku (sigmoid-l1 slice gating — running)
   - PR #543 haku (surface curvature features — running)
5. **Optimizer/LR confirmations:**
   - PR #548 edward (Muon@1e-3 + STRING-sep PE composition — **highest priority combination run**)
   - PR #519 gilbert (LLRD)
   - PR #528 nezuko (1-cycle)
   - PR #478 chihiro (per-step cosine)

## Post-#490 Rebase Required (4 PRs)

Branches predating the 15:48 UTC PR #490 merge to yi have train.py conflicts (PE/embedding section). These students have been instructed to finish their in-flight runs first, then rebase before re-marking ready for review:

- PR #522 (thorfinn) — needs_rebase + active Arm B rerun (`y11sr80t`)
- PR #528 (nezuko) — needs_rebase + Arm A `x3qlnjjm` running
- PR #519 (gilbert) — needs_rebase + Arm C running, Arm B to relaunch
- PR #478 (chihiro) — needs_rebase + r30 rematch Arm A `vyeq1ggj` running

## Fleet Stability Constraints

- Lion optimizer stable at lr=1e-4 with warmup=1 epoch, clip=0.5, wd=5e-4
- AdamW stable max lr=5e-4 with warmup=500 steps
- Muon@1e-3 validated on full-budget 4-GPU (PR #472, closed): val_abupt 11.349% (~43.8% relative gain over AdamW@1e-4). Compute-limited at 3 epochs/4.5h. Composition with STRING-sep now in-flight PR #548.
- `SENPAI_TIMEOUT_MINUTES=360` → ~5–6 epochs at DDP-4 (4L/512d, 65k pts)
- VRAM: 4× H100 96GB; typical usage 68-74 GB / 96 GB at DDP-4 batch=4
- Gnorm kill threshold: anything consistently > 300 (Lion stable < 50)
