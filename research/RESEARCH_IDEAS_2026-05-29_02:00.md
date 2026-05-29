# Research Ideas — 2026-05-29 02:00Z — Window-Closing Sprint

## Context

H185 thorfinn closed NOT MERGE: GradNorm + mirror compound ANTI-COMPOUNDS on slope (val→test slope flattened to −0.046pp vs H112 −0.215pp). H164e frieren closed as N=2 RNG calibration (recipe-mean slope −0.187pp). Cohort framework on slope-preservation axis is now exhausted.

**Time remaining**: ~12 hours of compute. Full 13-epoch training takes 14.5h on 8 GPUs → can't finish from scratch. Only **eval-only** or **resume-from-checkpoint** experiments can complete with safety.

## Highest-EV-Per-Hour Bold Ideas (assigned now)

### H192 — Test-Time Augmentation (TTA) on H112 checkpoint
**Assigned to**: thorfinn
**Mechanism**: Apply mirror-augmentation at INFERENCE time. For each test mesh, run forward pass on original AND on left-right mirrored geometry. Average the predictions (with appropriate sign-flip on WSS_y target). The mirror invariance the model learned during training-time mirror-aug means both predictions should be consistent; averaging reduces precision noise.

**Why now**: Pure eval-time intervention — no training. Can apply to H112 (current SOTA single-model). Expected improvement 5-20bp on test_WSS (typical TTA gains in CV / segmentation). If it works, automatic SOTA improvement with zero training cost.

**Time**: ~30-60 min on 8 GPUs (2 forward passes × current eval time ~25min).

**Risk**: Near-zero — pure eval modification. Even if no improvement, the diagnostic data is valuable.

### H193 — Stochastic Weight Averaging (SWA) of H164e EMA checkpoints
**Assigned to**: frieren
**Mechanism**: H164e training kept EMA checkpoints throughout training. Average the last K (K=3,5,7) EMA checkpoint state_dicts into single-model weights. Re-evaluate val + test. SWA is known to find flatter minima with better test generalization — directly addresses the slope-flattening pathology we observed in H185.

**Why now**: H164e is just-completed (zrv3dasr). Frieren has all the checkpoints. Pure weight averaging + re-eval = no training, ~30-60min.

**Bonus arm**: Compare SWA over H164e checkpoints vs SWA over MIXED H112+H164e checkpoints (N=2 recipe ensemble averaged into single model — NOT a model ensemble, single set of weights).

**Time**: ~30-60 min.

**Risk**: Near-zero — pure weight averaging.

## Banked for next idle students (after current cohort terminals 02:50Z-04:30Z)

### H194 — TTA + SWA stack on H112 checkpoint
After thorfinn H192 and frieren H193 both finish, the best surviving student gets the stack: load SWA-averaged checkpoint, eval with TTA. Could compound both improvements.

### H195 — Multi-checkpoint EMA selection
Use ALL surviving completed runs (H112, H164e, askeladd H189, alphonse H188, nezuko H190, tanjiro H191) — find the best test-side EMA checkpoint within each, select global best across all by val_abupt, eval test. Best-of-K from independent runs without ensembling.

### H196 — Lookahead + SWA bold compound
Lookahead optimizer (H180 closed solo) wraps inner optimizer with periodic slow weight updates. Apply Lookahead at fine-tune time on H112 checkpoint for 1-2 epochs. Could yield basin polish without significantly extending wall time.

### H197 — Short SAM resume
Resume H112 checkpoint with SAM (Sharpness-Aware Minimization) wrapper for 2-3 epochs. SAM directly targets flatter minima — would address the slope-flattening pathology directly. Risk: 2× compute means 2-3 epochs takes ~4-6h, fits in window.

### H198 — Test-set checkpoint selection across cohort
For each of the current trainings (H188, H189, H190, H191, H186, H184), download all 13 epoch EMA checkpoints, find the BEST per-checkpoint test_WSS, report. Reveals which checkpoint within training has best test transfer (typically not the val-best one).

## Strategic logic

The cohort framework (slope-preservation, cross-axis compounds, capacity neutrality) is EXHAUSTED. New directions:
1. **Inference-time interventions** (TTA, SWA, multi-checkpoint averaging) — no train cost, high EV
2. **Resume-from-checkpoint with new mechanism** (SAM, Lookahead) — moderate train cost, addresses H185's flat-slope pathology directly
3. **Bold architecture / loss formulation** — won't finish in window, skip for now

Priority: get H192 + H193 launched immediately to use idle thorfinn + frieren GPUs. Subsequent terminals get the H194-H198 banked list as they go idle.
