# SENPAI Research State
- 2026-04-29 (Round-2 assignments complete)

## Most Recent Research Direction from Human Researcher Team

From Issue #18 (open):
- Use ALL W&B logged metrics — gradient norms, weight histograms, loss slope — not just final val loss
- Flag epoch-limited runs (still on a downward trajectory at the epoch cap) as promising, not closed
- Differentiate failure modes: flat/diverging gradients early = fundamental; healthy slope at cutoff = epoch-limited
- Prioritize convergence speed in new hypotheses (warmup schedules, better init, fast-converging architectures)

## Current Research Focus and Themes

### Yi Best: 13.15 abupt (PR #14, senku 6L/256d, 2026-04-29)

**Breakthrough:** Depth is more parameter-efficient than width at this scale.
6L/256d (4.73M params) crushed 4L/512d (12.7M params) by 21%. Both 5L and 6L
were still descending at timeout — significant untapped improvement available with
longer training.

**Key findings from Round-1/2 reviews:**
1. Depth scaling law: 4L→5L = −18.7%, 5L→6L = −2.7% (diminishing returns, but 7L/8L not yet tested)
2. Cosine EMA (PR #13) adds 9% orthogonally — now standard on yi
3. Gradient clipping clip=1.0 is now standard and essential for stability at 6L
4. FiLM geometry conditioning: 46% relative improvement at 1 epoch on 4L (frieren PR #8 pending rebase)
5. Volume pressure shows val→test gap (6.93 val ≈ AB-UPT, 13.58 test = 2×) — generalization problem
6. Wall-shear y/z remain the largest gap (4.4-4.6× AB-UPT) despite 60-70% improvement from Round-1 start

**Distance to AB-UPT (current best):**

| Metric | yi best | AB-UPT | Ratio |
|---|---:|---:|---:|
| surface_pressure | 7.64 | 3.82 | 2.0× |
| wall_shear | 13.47 | 7.29 | 1.8× |
| volume_pressure | 13.58 | 6.08 | 2.2× |
| wall_shear_x | 11.53 | 5.35 | 2.2× |
| wall_shear_y | 16.23 | 3.65 | 4.4× |
| wall_shear_z | 16.75 | 3.63 | 4.6× |

## Active WIP PRs (Round-2)

| PR | Student | Hypothesis |
|---|---|---|
| #58 | alphonse | NaN checkpoint guard bugfix (correctness) |
| #59 | senku | 7L/8L depth sweep beyond 6L win |
| #60 | chihiro | 6L/512d depth × width composition |
| #61 | gilbert | Tangential wall-shear projection on 6L |
| #62 | norman | FiLM geometry conditioning on 6L |
| #63 | askeladd | Squared rel-L2 aux loss on 6L (w∈{0.1,0.5,1.0}) |
| #64 | fern | Stochastic depth regularization (p∈{0.05,0.1,0.2}) |
| #65 | violet | Volume loss weight sweep (1.5/2.0/3.0/4.0) |
| #66 | thorfinn | Per-axis tau_y/z loss upweighting |
| #67 | kafka | LR warmup + cosine decay schedule |
| #21 | kohaku | Normal-suppression rerun on 6L (WIP — sent back) |
| #15 | tanjiro | SDF-gated volume attention (sigma=0.005, sent back) |

## Pending Code Merges

| PR | Student | Code | Status |
|---|---|---|---|
| #8 | frieren | FiLM geometry conditioning code | Needs rebase |
| #24 | emma | Squared rel-L2 aux loss code | Needs rebase |
| #28 | norman | SE(3) local-frame features (A02) | Draft, no-status |
| #23 | frieren | FiLM+projection+protocol composition | Draft, no-status |

## Potential Next Research Directions

### Priority 1: Compositional wins on 6L base (highest expected value)
- FiLM + 6L (norman PR #62) — if FiLM's 46% gain at 1-epoch composes with depth, expect abupt < 11
- Tangential projection + 6L (gilbert PR #61) — projection may finally work stably with clip=1.0
- 7L/8L depth (senku PR #59) — depth scaling law may still hold; both 5L/6L descending at timeout
- 6L/512d (chihiro PR #60) — test if width+depth is additive (hypothesis: abupt ~11-12)

### Priority 2: Loss formulation and training dynamics
- Squared rel-L2 aux loss on 6L (askeladd PR #63) — emma's w=0.5 showed +11% on 4L
- Per-axis tau_y/z loss weighting (thorfinn PR #66) — direct attack on largest remaining gap
- Volume loss weight sweep (violet PR #65) — vw=3.0 now safe with clip=1.0
- LR warmup + cosine decay (kafka PR #67) — convergence speed is rate-limiting

### Priority 3: Generalization (volume pressure val→test gap)
- Stochastic depth regularization (fern PR #64) — drop path targets the generalization gap
- SDF-gated volume attention with sigma=0.005 (tanjiro PR #15) — near-wall focus
- Data augmentation: point cloud jitter, geometry reflection during training (not yet tested)

### Priority 4: Architecture exploration (longer horizon)
- SE(3) local-frame coordinate features (norman PR #28 draft) — equivariant features
- Longer training runs (10-12h) on 6L or 7L — both still descending at 4.5h timeout
- 6L with all current wins composed (projection + FiLM + EMA + clip + aux_loss)

## Known Correctness Issues
1. **NaN checkpoint guard bug**: when EMA becomes NaN, `_finite_mean()` returns 0.0
   which incorrectly passes the `< best_val` check and overwrites a valid checkpoint.
   Fix: `improved = math.isfinite(primary_val) and primary_val > 0.0 and primary_val < best_val`.
   Assigned to alphonse PR #58.

## Key Constraints
- Training budget: ~270 min training + ~90 min val/test = 360 min total
- VRAM: 96 GB per GPU; 6L/256d at bs=8 uses 75.5 GB; 6L/512d at bs=4 estimated ~80-90 GB
- Epoch budget: ~3-4 epochs at 6L/256d throughput (~2.1 it/s)
- Gradient clipping: clip_grad_norm=1.0 is now standard (anything without it is unstable)
- Baseline: 6L/256d, lr=2e-4, ema-decay-start=0.99/end=0.9999, vol_w=2.0, bs=8, clip=1.0
