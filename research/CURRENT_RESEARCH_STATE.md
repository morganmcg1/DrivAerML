# SENPAI Research State
- 2026-05-03 18:38 UTC (Round 33 — advisor cycle)
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

## Active WIP PRs (Round 33 — 16 PRs)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #530 | tanjiro | Tangent-frame output head (τ→{t1,t2,n}) | Running ep4 (val 9.794%, converging) |
| #529 | violet | Stream-normal per-point ws loss weight (γ=1/3 sweep) | Running (arm B gamma=3 val 11.5% at step 10886) |
| #528 | nezuko | 1-cycle LR pct_start=0.05 vs cosine ep3 | Arm A done (10.502%), arm B cosine-ctrl running |
| #522 | thorfinn | BF16→FP32 last-2-epoch precision | Arm A done (10.923%), arm B rerun running |
| #521 | senku | Sigmoid-l1 slice attention vs softmax | Arm B running (live 13.4% at step 10886) |
| #520 | norman | Volume-only coord transform (signed-log/asinh) | Arm A done (10.964%), arm C asinh started |
| #519 | gilbert | LLRD sweep (decay=0.75/0.95) | Arm A done (11.168%), arm C running |
| #518 | kohaku | Loss-side asinh wallshear (delta sweep) | Arm d=2.0 running (28min) |
| #478 | chihiro | Per-step cosine LR schedule | Arm B step-cosine running (live 25.0%) |
| #472 | edward | Muon@1e-3 vs AdamW full-budget 4-GPU | Arm C muon running (live 10.1% at step 10884) |
| #539 | frieren | STRING-sep + Lion lr=1e-4 clip=0.5 from scratch | **NEW — just assigned** |
| #540 | emma | GradNorm adaptive τ_y/τ_z up-weighting | **NEW — just assigned** |
| #541 | askeladd | Streamline-aligned wall-shear target frame | **NEW — just assigned** |
| #542 | alphonse | Laplace far-field soft penalty on vol pressure | **NEW — just assigned** |
| #543 | haku | Principal curvatures (H,K) as surface features | **NEW — just assigned** |
| #544 | fern | y-symmetry paired loss + test-time mirror ensemble | **NEW — just assigned** |

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
   - PR #472 edward (Muon@1e-3 — live 10.1%, very promising if it holds)
   - PR #519 gilbert (LLRD)
   - PR #528 nezuko (1-cycle)
   - PR #478 chihiro (per-step cosine)

## Fleet Stability Constraints

- Lion optimizer stable at lr=1e-4 with warmup=1 epoch, clip=0.5, wd=5e-4
- AdamW stable max lr=5e-4 with warmup=500 steps
- Muon tested at 1e-3 (edward) — single-GPU validated −24.8% relative, full-budget pending
- `SENPAI_TIMEOUT_MINUTES=360` → ~5–6 epochs at DDP-4 (4L/512d, 65k pts)
- VRAM: 4× H100 96GB; typical usage 68-74 GB / 96 GB at DDP-4 batch=4
- Gnorm kill threshold: anything consistently > 300 (Lion stable < 50)
