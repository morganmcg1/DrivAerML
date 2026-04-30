# SENPAI Research State
- 2026-04-30 (Round-5 bold experiments fanning out — all 14 yi students running)

## Most Recent Research Direction from Human Researcher Team

From Issue #18 (open, Morgan — latest message 2026-04-30T20:29:19Z):

**Overarching directive:** Stop incremental tuning. Rip out the model architecture and try completely new approaches. Students can handle radical departures from the reference train.py as long as logging/validation/checkpointing are maintained.

**Round-5 bold experiment priorities (Morgan's ordered list by impact):**
1. **Surface-tangent frame wall-shear prediction** — The 4× wall shear y/z error is a coordinate frame mismatch, not a hyperparameter problem. Predict tau in local surface-tangent frame (tau_normal=0, tau_t1, tau_t2), rotate back to global. Requires per-point surface normal computation + rotation head.
2. **Perceiver-IO backbone replacement** — Replace Transolver entirely. Perceiver-IO uses learned latent queries for unstructured CFD meshes, ~3× faster per epoch = more epochs within budget.
3. **asinh/log target normalization** — Wall shear spans 4 decades. Predict asinh(tau) to fix MSE over-weighting high-magnitude patches. 10-line change but changes the loss landscape entirely.
4. **Physics-informed RANS constraint** — Add soft divergence-free penalty (∇·u=0) on predicted velocity at volume points. Pure loss change, no architecture touch.
5. **1-cycle LR schedule with higher peak** — Warmup to 1e-3, cosine decay to 1e-6, ~20% warmup of total steps. Fern's lr=5e-4 was still converging at epoch cutoff.

**Additional guidance from Morgan:**
- Use W&B gradient norms, weight histograms, and loss slopes to identify epoch-limited runs (healthy slope at cutoff = still converging, worth follow-up)
- Before finalizing hypotheses, scan PRs from `noam` and `radford` branches for prior art inspiration
- Gradient metric failure flags: spikes >10 or flat <0.01 = fundamental failure; healthy slope at cutoff = epoch-limited

**Advisor status:** All 5 Morgan directives are actively running (PRs #121–125). Additional complementary Round-5 experiments also in flight (#126–132).

## Current Baseline: PR #99 (fern) — abupt 10.69 — 2026-04-29

**Compounded wins on `yi` so far:**
1. PR #11 kohaku — tangential wall-shear projection loss code
2. PR #9 gilbert — protocol fixes (bs=8, vol_w=2.0, validation-every=1)
3. PR #4 chihiro — width scale-up to 512d/8h
4. PR #14 senku — depth scale-up to 6L/256d (21% improvement: 16.64 → 13.15)
5. PR #58 alphonse — NaN-safe checkpoint guard (bugfix)
6. PR #66 thorfinn — per-axis tau_y/z loss upweighting W_y=2, W_z=2 (3.1%: 13.15 → 12.74)
7. PR #99 fern — LR peak 5e-4 (16.1%: 12.74 → 10.69)

**Current best metrics (PR #99, W&B run `3hljb0mg`):**

| Metric | yi best | AB-UPT | Ratio |
|---|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **10.69** | — | — |
| `surface_pressure_rel_l2_pct` | **6.97** | 3.82 | 1.8× |
| `wall_shear_rel_l2_pct` | **11.69** | 7.29 | 1.6× |
| `volume_pressure_rel_l2_pct` | **7.85** | 6.08 | 1.3× |
| `wall_shear_x_rel_l2_pct` | **10.17** | 5.35 | 1.9× |
| `wall_shear_y_rel_l2_pct` | **13.73** | 3.65 | 3.8× |
| `wall_shear_z_rel_l2_pct` | **14.73** | 3.63 | 4.1× |

**Key structural observations:**
- Depth (6L/256d, 4.73M params) is far more parameter-efficient than width (4L/512d, 12.7M params)
- Wall_shear_y and wall_shear_z remain the largest gaps (~4× AB-UPT) despite thorfinn's 2× upweighting + fern's lr boost
- Volume pressure gap almost closed (1.3× AB-UPT) — sp and ws_x are next priority
- LR=5e-4 at 5× base dramatically accelerated convergence within epoch budget
- Both 5L and 6L runs were still descending at timeout — epoch budget is tight

**Standard base config (PR #99 winning arm):**
```bash
python train.py \
  --volume-loss-weight 2.0 --batch-size 8 --validation-every 1 \
  --lr 5e-4 --weight-decay 5e-4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 6 --model-hidden-dim 256 --model-heads 4 --model-slices 128 \
  --ema-decay 0.9995 --clip-grad-norm 1.0 \
  --wallshear-y-weight 2.0 --wallshear-z-weight 2.0 \
  --gradient-log-every 100 --weight-log-every 100 --no-log-gradient-histograms
```

## Active WIP PRs (experiments in flight on PR #99 baseline)

### Round-4 incremental PRs
| PR | Student | Branch | Hypothesis |
|---|---|---|---|
| #116 | fern | `fern/wallshear-axis-upweight-sweep` | Higher tau_y/z weights on lr=5e-4 base |
| #117 | alphonse | `alphonse/width-384d-sweep` | 6L/384d width expansion test |
| #118 | chihiro | `chihiro/mlp-ratio-sweep-r4` | MLP-ratio sweep on PR #99 base |
| #119 | edward | `edward/rff-coordinate-encoding` | Random Fourier Feature coordinate encoding |

### Round-5 bold PRs (Morgan Issue #18 directives + complementary explorations)
| PR | Student | Branch | Hypothesis |
|---|---|---|---|
| #121 | askeladd | `askeladd/surface-tangent-frame-wallshear` | Predict tau in local {t1, t2, n} frame, rotate back |
| #122 | emma | `emma/perceiver-io-backbone` | Perceiver-IO backbone replacing Transolver |
| #123 | frieren | `frieren/asinh-log-target-normalization` | asinh wall-shear target normalization |
| #124 | gilbert | `gilbert/rans-divergence-constraint` | Soft div(u)=0 RANS penalty on volume points |
| #125 | haku | `haku/onecycle-lr-peak-1e3` | 1cycle LR with max=1e-3 peak |
| #126 | kohaku | `kohaku/film-conditioning-6l-256d` | FiLM geometry conditioning on PR #99 base |
| #127 | nezuko | `nezuko/stochastic-depth-regularization` | Stochastic-depth sweep (0.05/0.1/0.2) |
| #128 | norman | `norman/ema-decay-warmup-schedule` | EMA decay warmup schedule (0.99 -> 0.9999) |
| #129 | senku | `senku/surface-loss-upweight-sweep` | Surface loss weight sweep (1.5/2.0/3.0) |
| #130 | tanjiro | `tanjiro/curriculum-tau-yz-weighting` | Curriculum tau_y/z weighting (start=1, ramp to 3-4) |
| #131 | thorfinn | `thorfinn/log-magnitude-wallshear-targets` | Log-magnitude wall-shear target normalization |
| #132 | violet | `violet/wallshear-magnitude-direction-decoupled` | Decoupled |tau| + direction (cosine loss) heads |

## Current Research Themes

### Theme 1: Closing wall_shear_y/z gap (4× AB-UPT — HIGHEST PRIORITY)
The single biggest lever still not pulled: **why do y/z shear components fail 4× harder than AB-UPT?**
- #116 fern — higher tau_y/z weights on new lr=5e-4 base
- #121 askeladd — surface-tangent frame prediction (Morgan #1 priority — coord frame mismatch hypothesis)
- #130 tanjiro — curriculum tau_y/z weighting (start=1, ramp to 3-4)
- #132 violet — decoupled |tau| + direction (cosine loss) heads

### Theme 2: Convergence speed within epoch budget
Epoch budget is tight (~3-4 epochs). Every schedule/LR decision matters hugely.
- #99 fern (merged) — lr=5e-4 gave 16.1% win; 5× acceleration effect
- #125 haku — 1cycle LR max=1e-3 (Morgan #5 priority — warmup to 1e-3, cosine anneal to 1e-6)
- #128 norman — EMA decay warmup schedule (0.99 → 0.9999 over training)

### Theme 3: Architecture — Backbone replacement and variants
Currently on 6L/256d Transolver. Bold replacement being tested.
- #117 alphonse — 6L/384d width expansion
- #118 chihiro — MLP ratio sweep (6/8)
- #119 edward — RFF coordinate encoding
- #122 emma — Perceiver-IO backbone replacing Transolver (Morgan #2 priority — 3× faster per epoch)
- #126 kohaku — FiLM geometry conditioning on PR #99 6L/256d base

### Theme 4: Target normalization
Wall shear spans 4 decades of magnitude. MSE on raw values over-weights large signals.
- #123 frieren — asinh/log wall-shear target normalization (Morgan #3 priority)
- #131 thorfinn — log-magnitude wall-shear target normalization (complementary approach)

### Theme 5: Physics-informed constraints and loss engineering
- #124 gilbert — RANS div(u)=0 penalty on volume points (Morgan #4 priority)
- #129 senku — surface loss weight sweep (1.5/2.0/3.0)
- #127 nezuko — stochastic depth regularization sweep

## Closed Dead Ends (prior architecture experiments)
- SE(3) equivariant coordinates (#25, #28) — CLOSED: didn't converge, gradient collapse
- ANP cross-attention decoder (#26, #35) — CLOSED: unstable, no improvement
- Mamba-2 SSM decoder (#45) — CLOSED: unstable, diverged
- SDF-gated volume attention (#15, #36) — CLOSED: no improvement vs vanilla
- AdaLN FiLM (#34) — CLOSED: marginal, superseded by FiLM on 6L (#62 pending)
- Area-weighted loss v1 (#7, #17) — CLOSED: area-weighted non-viable on prior baselines; retesting on PR #99 base (#95)

## Key Constraints
- Training budget: ~270 min training + ~90 min val/test = 360 min total (~3-4 epochs at 6L/256d)
- VRAM: 96 GB per GPU; 6L/256d at bs=8 uses 75.5 GB
- Epoch budget: ~3-4 epochs at 6L/256d throughput (~2.1 it/s)
- Gradient clipping: clip_grad_norm=1.0 is standard (anything without it is unstable)
- Students have 4 GPUs each (but run single-GPU experiments; DDP available for bold architecture tests)

## Next Research Directions (Round-5 Priorities)

1. **Surface-tangent frame wall-shear prediction** — predict tau in local geometric frame, rotate back; directly targets 4× wsy/wsz gap
2. **Perceiver-IO backbone** — replace Transolver; faster per-epoch enables more epochs within budget
3. **asinh/log target normalization** — normalize wall shear before loss; heavy-tail problem hypothesis
4. **1cycle LR with higher peak** (1e-3 max) — squeeze more convergence from limited epochs
5. **Physics-informed RANS constraint** — div-free volume pressure soft penalty
6. **Curriculum tau_y/z weighting** — start at W_y=W_z=1, ramp to 3-4 over training
