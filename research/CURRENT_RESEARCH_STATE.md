# SENPAI Research State
- 2026-04-30 15:25 UTC (Round-3 running, Round-4 first assignment)

## Most Recent Research Direction from Human Researcher Team

From Issue #18 (open):
- Use ALL W&B logged metrics — gradient norms, weight histograms, loss slope — not just final val loss
- Flag epoch-limited runs (still on a downward trajectory at the epoch cap) as promising, not closed
- Differentiate failure modes: flat/diverging gradients early = fundamental; healthy slope at cutoff = epoch-limited
- Prioritize convergence speed in new hypotheses (warmup schedules, better init, fast-converging architectures)
- Bold architecture changes permitted — students may completely replace the model backbone as long as logging/validation/checkpointing are preserved
- Scan noam and radford branches for prior techniques and inspiration
- Use gradient norms, weight histograms, loss slopes to identify epoch-limited runs for follow-up

## Current Research Focus and Themes

### Yi Best: 12.74 abupt (PR #66, thorfinn per-axis tau_y/z W_y=W_z=2, 2026-04-30)

**Baseline config (PR #66 winning arm, W&B run gvigs86q):**
```
6L/256d/4h, lr=2e-4, ema-decay-start=0.99/end=0.9999, vol_w=2.0, bs=8, clip=1.0,
wallshear-y-weight=2.0, wallshear-z-weight=2.0
```

**Compounding wins on yi:**
1. PR #11 kohaku — tangential wall-shear projection loss code
2. PR #9 gilbert — protocol fixes (bs=8, vol_w=2.0, validation-every=1)
3. PR #4 chihiro — width scale-up to 512d/8h
4. PR #14 senku — depth scale-up to 6L/256d (21% gain)
5. PR #58 alphonse — NaN-safe checkpoint guard (bugfix)
6. PR #66 thorfinn — per-axis tau_y/z loss upweighting W_y=W_z=2 (3.1% gain)

**Distance to AB-UPT (current best, PR #66):**

| Metric | yi best | AB-UPT | Ratio |
|---|---:|---:|---:|
| surface_pressure | 7.64 | 3.82 | 2.0x |
| wall_shear | 12.86 | 7.29 | 1.8x |
| volume_pressure | 13.14 | 6.08 | 2.2x |
| wall_shear_x | 11.29 | 5.35 | 2.1x |
| wall_shear_y | 15.15 | 3.65 | 4.2x |
| wall_shear_z | 15.05 | 3.63 | 4.1x |

**Key diagnostic findings:**
- Volume pressure val→test gap: val ~6.9 (≈AB-UPT) vs test 13.14 — overfitting not capacity
- Wall-shear y/z (~4x AB-UPT) remain the largest gap; per-axis upweighting helps but not enough
- Epoch-limited: baseline trajectory 22.78→15.89→13.30→12.36 still descending at epoch 4
- FiLM on 6L: 12.905 abupt (norman PR #62 run phfo03pc) — beat old baseline but not new 12.74

## Active WIP PRs (Round-3 — running)

| PR | Student | Hypothesis |
|---|---|---|
| #107 | violet | vol_w decay schedule (4→1.5, 3→1.5) |
| #106 | thorfinn | Finer tau_y/z weight sweep (2.0/2.5/3.0 + asym) |
| #105 | tanjiro | Huber surface loss (delta=0.05/0.10) |
| #104 | senku | EMA decay sweep (0.999/0.9995/0.9997/0.9999) |
| #103 | kohaku | vol→surf cross-attention for volume pressure |
| #102 | haku | Attention dropout sweep (0/0.05/0.10/0.20) |
| #101 | gilbert | Larger point budget 131072 vs 65536 |
| #100 | frieren | cp-channel upweight (1.5/2.0/3.0) + tau_y/z=2 |
| #99 | fern | LR peak sweep (1e-4/2e-4/3e-4/5e-4) on thorfinn base |
| #98 | emma | Weight-decay sweep (1e-4/5e-4/2e-3/5e-3) |
| #97 | edward | Slice-count sweep 128→192→256 |
| #96 | chihiro | MLP-ratio sweep (4 vs 6 vs 8) |
| #95 | alphonse | Area-weighted surface MSE loss |

## Active WIP PRs (Older — action needed)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #62 | norman | FiLM conditioning on 6L | Needs rebase; advisor gave feedback to rebase + compose with tau_y/z=2 base |
| #63 | askeladd | Squared rel-L2 aux loss on 6L | WIP, Round-2 |

## Round-4 Assignments

| PR | Student | Hypothesis |
|---|---|---|
| #108 | nezuko | Point-cloud xyz jitter augmentation sigma sweep (0/0.001/0.003/0.005) |

## Closed / Resolved This Session

| PR | Reason |
|---|---|
| #67 | kafka LR warmup — stale orphan, no kafka pod, superseded by PR #99 (fern LR sweep) |

## Potential Next Research Directions

### Priority 1: Generalization — close the val→test gap (biggest leverage remaining)
- Point cloud jitter augmentation (nezuko PR #108, just assigned) — sigma sweep on 6L base
- FiLM on 6L with thorfinn base — norman PR #62 needs rebase; FiLM showed 12.905 vs old baseline 13.15; if it beats new 12.74 it merges
- Surface-volume consistency aux loss (enforce p_s = p_v at SDF~0) — not yet assigned
- SAM optimizer (Sharpness-Aware Minimization) — explicitly targets flat minima / generalization

### Priority 2: Bold architecture changes (human team directive)
- Separate surface/volume backbone branches — specialized per modality
- Multi-scale point cloud attention (PointTransformer-style setabstraction)
- Mixture-of-Experts routing per surface region (wheel arch vs roof vs underbody)

### Priority 3: Wall-shear y/z targeted attacks (4x AB-UPT gap)
- Finer tau_y/z sweep (thorfinn PR #106 running)
- Asymmetric tau weights (harder axes upweighted more aggressively)
- Physics-aware: divergence-free constraint on wall-shear (not yet tried)
- Bernoulli-consistency loss in potential flow regions

### Priority 4: Composition after Round-3 resolves
- Best loss weights + best EMA decay + best LR + jitter on 6L (wait for winners first)
- FiLM + tau_y/z + best lr/ema composition

### Priority 5: Epoch-limited revisits (from human team issue #18 directive)
- Flag any Round-3 run with downward slope at epoch limit as a candidate for longer run
- Currently: baseline 6L was still descending at epoch 4 (12.36 partial) — 6L with more time is high-value

## Key Constraints
- Training budget: ~270 min training + ~90 min val/test = 360 min total
- VRAM: 96 GB per GPU; 6L/256d at bs=8 uses 75.5 GB
- Epoch budget: ~3-4 epochs at 6L/256d throughput (~2.1 it/s)
- Gradient clipping: clip_grad_norm=1.0 is standard
- Single-delta principle: one change per PR; bundle only when compound effect is the hypothesis
