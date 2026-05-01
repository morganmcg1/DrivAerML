# SENPAI Research State
- 2026-05-01 (Round-6 launched — 9 new student assignments after Round-5 all-negative batch)

## Most Recent Research Direction from Human Researcher Team

From Issue #18 (open, Morgan — latest message 2026-04-30T20:29:19Z):

**Overarching directive:** Stop incremental tuning. Rip out the model architecture and try completely new approaches. Students can handle radical departures from the reference train.py as long as logging/validation/checkpointing are maintained.

**Round-5 bold experiment priorities (Morgan's ordered list by impact):**
1. **Surface-tangent frame wall-shear prediction** — 4× wall shear y/z error is a coordinate frame mismatch, not a hyperparameter problem
2. **Perceiver-IO backbone replacement** — Replace Transolver entirely; ~3× faster per epoch = more epochs within budget
3. **asinh/log target normalization** — Wall shear spans 4 decades; predict asinh(tau) to fix MSE over-weighting
4. **Physics-informed RANS constraint** — Soft divergence-free penalty on predicted velocity at volume points
5. **1-cycle LR schedule with higher peak** — Warmup to 1e-3, cosine decay to 1e-6

**Advisor status:** Issue #18 acknowledged and responded. No new messages since 2026-04-30T20:42Z.

## Current Baseline: PR #99 (fern) — abupt 10.69 — 2026-04-29

Baseline unchanged — all 7 Round-5 PRs were NEGATIVE.

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

## Round-5 Closed Results (all NEGATIVE — 2026-05-01)

Critical lessons learned from 7 failed experiments:

| PR | Student | Hypothesis | Outcome | Key Finding |
|---|---|---|---|---|
| #131 | thorfinn | logmag transform (eps=0.01-1.0) | NEGATIVE abupt=11.03 | Gradient of log1p near 0 is ~1/eps; eps≤0.10 caused 2M+ pre_clip_norm spikes. NaN-skip safeguard (commit 2a8f7e4) is a keeper. |
| #130 | tanjiro | Curriculum W_y 1→3 ramp | NEGATIVE 6/6 diverged | Adam m/v desync around W_y≈2.7; static weights needed, not curriculum |
| #124 | gilbert | Laplacian pressure constraint | NEGATIVE wrong physics | ∇²p≈0 is creeping flow; real RANS has advective terms. kNN Laplacian implementation reusable. |
| #121 | askeladd | Tangent-frame wall-shear | NEGATIVE worse than Cartesian | Duff ONB discontinuous at t1.x sign-flip; non-gauge-equivariant model can't learn it. Channel coupling in Adam. |
| #118 | chihiro | mlp_ratio sweep 6/8 | AMBIGUOUS 12/16 diverged | Trend toward mlp_ratio=8 but seed-dependent instability prevented convergent comparison. Needs warmup + seed flags. |
| #129 | senku | Uniform surface_sw sweep | NEGATIVE 7/8 diverged | Uniform sw amplifies all surface incl. already-upweighted W_y=W_z=2. Monotone instability with sw. Per-component is right knob. |
| #117 | alphonse | 6L/384d + 8L/256d depth | NEGATIVE no merger | 8L/256d time-limited (extrapolated to cross baseline 11 min after timeout). 384d unstable in bf16 at all LRs. Depth (not width) is viable scale-up. |

## Active WIP PRs (in flight)

### Still-running from Round-5 launch batch
| PR | Student | Branch | Hypothesis |
|---|---|---|---|
| #156 | levi | `levi/ohem-hard-case-mining` | OHEM top-25% hard case mining |
| #155 | armin | `armin/checkpoint-ensemble` | Top-3 checkpoint ensemble |
| #154 | mikasa | `mikasa/gradient-accumulation` | Grad accum eff-bs=32 |
| #153 | mob | `mob/lookahead-optimizer` | Lookahead optimizer k=5/10 |
| #152 | violet | `violet/geom-moment-conditioning` | 14-dim analytic geometry conditioning |
| #151 | nezuko | `nezuko/symmetry-augmentation` | L/R symmetry augmentation for tau_y gap |
| #150 | emma | `emma/multi-scale-hierarchy` | Multi-scale point hierarchy (2/3 scales) |
| #144 | edward | `edward/adamw-beta2-sweep` | AdamW beta2 sweep (0.95 vs 0.999) |
| #143 | fern | `fern/coord-normalization-sweep` | Coordinate normalization fix for sincos anisotropy |
| #126 | kohaku | `kohaku/film-conditioning-6l-256d` | FiLM geometry conditioning on PR #99 base |
| #125 | haku | `haku/onecycle-lr-peak-1e3` | 1cycle LR max=1e-3 |
| #123 | frieren | `frieren/asinh-log-target-normalization` | asinh wall-shear target normalization |

### Round-6 newly assigned (2026-05-01)
| PR | Student | Branch | Hypothesis |
|---|---|---|---|
| #164 | alphonse | `alphonse/depth-8L-1cycle-recovery` | 8L/256d depth + OneCycleLR (time-limited recovery) |
| #165 | chihiro | `chihiro/mlp-ratio-8-hardened` | mlp_ratio=8 + 1k warmup + seed=42/1337/7 (3-arm) |
| #166 | senku | `senku/per-component-wallshear-yz-3` | Static W_y=W_z=3.0 + 500-step LR warmup |
| #167 | tanjiro | `tanjiro/static-wyz-35-warmup` | Static W_y=W_z=3.5 + 1k LR warmup |
| #168 | askeladd | `askeladd/normal-penalty-wallshear-yz` | Normal-consistency soft penalty λ∈{0.01,0.05,0.10} |
| #169 | thorfinn | `thorfinn/nan-skip-utility-cherry-pick` | NaN-skip + seed + LR warmup utility infra |
| #170 | gilbert | `gilbert/width-384d-qknorm-fp32attn` | 384d + QK-norm + fp32-attention (stability fix) |
| #171 | norman | `norman/snapshot-ensemble-cyclic-lr` | Snapshot ensemble via cyclic LR (3 ckpts avg) |
| #172 | stark | `stark/adamw-eps-sweep` | AdamW eps sweep 1e-8/7/6/5 (gradient stability) |

## Current Research Themes

### Theme 1: Closing wall_shear_y/z gap (4× AB-UPT — HIGHEST PRIORITY)
The single biggest lever: **why do y/z shear components fail 4× harder than AB-UPT?**
- #166 senku — per-component W_y=W_z=3.0, static (correct knob vs #129's failed uniform sw)
- #167 tanjiro — per-component W_y=W_z=3.5, static (complement to senku's 3.0, maps stability ceiling)
- #168 askeladd — soft normal-consistency penalty λ∈{0.01,0.05,0.10} (physics without frame discontinuity)
- #123 frieren (in flight) — asinh target normalization (heavy-tail hypothesis)
- #151 nezuko (in flight) — L/R symmetry augmentation for tau_y gap

### Theme 2: Convergence / Architecture scaling
Epoch budget is tight (~3-4 epochs). Depth beats width. 8L/256d was time-limited.
- #164 alphonse — 8L/256d + 1cycle LR (super-convergence to beat 10.69 within budget)
- #165 chihiro — mlp_ratio=8 + stability hardening (seed + warmup)
- #170 gilbert — 384d width + QK-norm + fp32 attention (stability for width scale-up)
- #125 haku (in flight) — 1cycle LR max=1e-3

### Theme 3: Optimizer stability
Pervasive Round-5 divergences motivate systematic optimizer investigation.
- #172 stark — AdamW eps sweep (1e-8/7/6/5, maps denominator floor effect)
- #169 thorfinn — NaN-skip + seed + LR warmup utility PR (infra for all future experiments)
- #144 edward (in flight) — AdamW beta2 sweep (0.95 vs 0.999)

### Theme 4: Post-training / test-time gain
- #171 norman — snapshot ensemble with cyclic LR (free gain from averaging 3 cycle ckpts)
- #155 armin (in flight) — top-3 checkpoint ensemble

### Theme 5: Data representation and augmentation
- #143 fern (in flight) — coordinate normalization sweep (sincos anisotropy fix)
- #151 nezuko (in flight) — L/R symmetry augmentation
- #152 violet (in flight) — analytic geometry moment conditioning (14-dim)

## Key Structural Findings Accumulated

- **Depth >> Width:** 6L/256d (4.73M) dominates 4L/512d (12.7M) in param efficiency. 8L/256d promising but time-limited.
- **Adam m/v coupling:** Mid-run weight schedule changes (curriculum, EMA warmup) cause second-moment desynchronization. Always initialize Adam with the training-time weights.
- **Uniform surface_sw is the wrong knob:** Amplifies already-upweighted W_y/W_z. Per-component --wallshear-y/z-weight is correct.
- **384d in bf16 is unstable:** d_head=96 causes pre-softmax logit variance overflow. QK-norm or fp32 attention required.
- **logmag transform gradient:** gradient of sign(x)*log1p(|x|/eps) is ~1/eps near 0; eps≤0.10 caused 2M+ pre_clip_norm.
- **Duff ONB discontinuity:** Branchless ONB has sign-flip discontinuity incompatible with non-gauge-equivariant Transolver.
- **Δp≈0 is wrong RANS physics:** Laplacian pressure constraint only valid for Stokes (creeping) flow.
- **Volume pressure nearly converged:** 1.3× AB-UPT at baseline — wall_shear_y/z is the main gap.
- **LR warmup guards:** 500-1000 step linear warmup from 1e-5 is now standard practice for experiments with elevated initial gradients.

## Key Constraints
- Training budget: ~270 min training + ~90 min val/test = 360 min total (~3-4 epochs at 6L/256d)
- VRAM: 96 GB per GPU; 6L/256d at bs=8 uses ~75 GB; 384d requires bs=4
- Gradient clipping: clip_grad_norm=1.0 is standard
- Students have 4 GPUs each (DDP available for large architectures)
