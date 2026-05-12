# SENPAI Research State

- **Date:** 2026-05-09 (latest invocation: 2026-05-12 ~11:30Z)
- **Branch:** tay
- **W&B project:** wandb-applied-ai-team/senpai-v1-drivaerml-ddp8

---

## Most Recent Research Direction from Human Researcher Team

From Issue #717 (2026-05-09): **test-volume pressure L2 reduction is the single target.** Radical architecture and data preprocessing ideas welcome. Surface/wall shear should not be degraded but do not need further improvement.

Key OOD gap: val_vol_p ≈ 3.9% vs test_vol_p ≈ 12.0% — a ~3× multiplier. This is the primary unsolved problem.

---

## Current Best Results

### Single-Model SOTA (PR #958 — nezuko, dedicated vol_p aux head)
- val_abupt = 6.2868% | test_abupt = 7.7445%
- val_vol_p = 3.9152% | **test_vol_p = 12.0063%** (OOD gap ~3×)
- val_surf_p = 4.1766% | test_surf_p = 3.9100%
- W&B run: 29nohj67

### Ensemble SOTA (PR #1030 — nezuko, pool-33 K=3 greedy)
- val_abupt = 5.9170% | test_abupt = 7.3192%
- val_vol_p = 3.5136% | **test_vol_p = 11.6492%**
- K=3 members: 29nohj67, ghh0s4ne, 4k25s25e

### Gate to Beat
- Single-model training: val_abupt < 6.2868%
- Ensemble: val_abupt < 5.9170%

---

## Current Research Focus and Themes

### Theme 1: Reducing the OOD test_vol_p Gap (PRIMARY)
- **Per-case loss upweighting** (alphonse PR #1019): Arm A (scale=2.0) **NEGATIVE** at 4ep (val_abupt=7.16%, test_vol_p=12.5534% vs baseline 12.0063%). Arm B (scale=1.5, run `plxzyhxa`) running; expected NEGATIVE. **Likely closure imminent.**
- **Instance normalization (RevIN)** (thorfinn PR #1017): EP5 val_abupt=7.3989%, awaiting test eval to compare test_vol_p vs baseline 12.0063%. Run `c1r0iuun`.
- **INR coord-conditioned decoder** (edward PR #1028): EP1 val_abupt=10.22% (bottom of projected band), trajectory healthy. EP3 gate at ~12:55Z.
- **Slot-based vol attention** (askeladd PR #1024): EP4 screen val_abupt=7.0938%. **Phase 2 Option C launched** with `0:49152:1:65536` curriculum, 7 epochs, ~348 min. Strong step-aligned advantage throughout screen.

### Theme 2: Architecture Capacity and Expressivity
- **L=7 depth** (nezuko PR #1032): Compressed 4-epoch screen `5vswi9ix` running. EP1 expected ~09:48Z.
- **Pre-xattn vol self-attn** (fern PR #1031): Run `769cr3it` running. Plan: collect EP0/EP1, decide on chain-run continuation. **Blocked on tanjiro's `--init-from-checkpoint` infra (PR #1027).**

### Theme 3: Training Dynamics and Augmentation
- **Step-warmup-8000** (tanjiro PR #1027): Implementing `--init-from-checkpoint` infra flag (~30 lines) before launching the chain-run. **Unblocks chain-runs across multiple PRs.**
- **Coord frame augmentation** (frieren PR #1029): Arm A (rot=0.1, scale=0.02) **PASSED EP3 gate** (val_abupt=7.9469 ≤ 8.0). Arm B (rot=0.25, scale=0.05) auto-queued at ~11:30-12:30Z.

---

## Active WIP PRs (Round 30+)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #1032 | nezuko | L=7 depth (ensemble pool candidate) | Screen running `5vswi9ix` |
| #1031 | fern | Pre-xattn vol self-attn | Running `769cr3it`; chain-run blocked on PR #1027 infra |
| #1029 | frieren | Coord frame augmentation (yaw+scale) | Arm A PASS EP3; Arm B queued |
| #1028 | edward | INR coord-conditioned vol decoder | EP1 healthy; EP3 gate ~12:55Z |
| #1027 | tanjiro | Step-warmup-8000 + `--init-from-checkpoint` infra | Implementing infra flag |
| #1024 | askeladd | Slot-based vol attn (Perceiver-IO, S=64) | **Phase 2 Option C launching** (7ep, 6@65K) |
| #1019 | alphonse | OOD-4 vol-only loss upweighting (scale 2.0/1.5) | Arm A NEGATIVE; Arm B running |
| #1017 | thorfinn | Per-case vol_pressure RevIN normalization | EP5 val_abupt=7.40%; test eval running |

---

## Potential Next Research Directions

### High Priority (test_vol_p OOD gap)
1. **Domain randomization / geometry augmentation at training time** — if frieren coord aug Arm A/B shows signal, push further: random reflections, stretch along principal axes
2. **Test-time adaptation (TTA)** for vol_p — compute a per-run adaptation signal at inference using available geometry info without labels
3. **Geometry-conditioned separate volume decoder** — condition the vol decoder on global geometry descriptors (SDF statistics, bbox, etc.) explicitly rather than relying on attention to propagate geometry context
4. **Curriculum by geometry distance** — order training samples so OOD-proximate geometries are seen more frequently in later epochs
5. **Vol_p-specific loss reformulation** — log-space MSE, normalized prediction (z-score per run), or relative pressure prediction
6. **Combine RevIN (#1017) with slot-vol-attn (#1024)** — if both are positive, they may stack since they target different parts of the OOD gap (test-time normalization vs richer representation)

### Architecture Directions
7. **Deeper ensemble diversity** — after L=7 result (PR #1032), run L=6 explicitly as a pool member to maximize diversity
8. **Mixed-resolution volume training** — train with more vol points at lower resolution early, then focus on high-res late epochs for the OOD geometries
9. **Surface-volume feature fusion via FiLM conditioning** — use surface global statistics to modulate volume decoder via feature-wise linear modulation

### Learning / Optimization
10. **Per-channel loss weighting sweep** — grid search over vol_loss_weight ∈ {1.5, 2.0, 3.0} with the PR #958 baseline stack
11. **Sharpness-aware minimization (SAM)** for better generalization to OOD geometries

---

## Key Findings to Date

- **L=5 depth** is the backbone sweet spot; L=7 (PR #1032) being tested for ensemble diversity
- **Surf→vol cross-attention** (PR #823) gave +2.4% val improvement — biggest single architecture win
- **Vol_p aux head** (PR #958) improved val_abupt further but did NOT reduce OOD gap (test_vol_p got worse)
- **Total-loss upweighting** (PR #888, scale=3.0) on SDF-neighbour proxies = NEGATIVE
- **Vol-only loss upweighting** (PR #1019 Arm A, scale=2.0) on SDF-neighbour TRAIN proxies = NEGATIVE on all metrics including test_vol_p. The SDF-neighbour selection strategy does not identify cases that meaningfully share the geometric distribution shift of the test cases.
- **Slot-based vol attention** (PR #1024 screen) delivered consistent step-aligned advantage over PR #958 throughout 4-epoch screen — Phase 2 13-epoch-equivalent extended run launching to confirm.
- **Ensemble (K=3)** beats single-model by ~0.37pp val abupt; pool diversity matters more than size
- The OOD test/val gap on vol_p (~3×) is a data distribution problem, not model capacity — suggests need for geometric generalization strategies
- **Infra constraint**: `--init-from-checkpoint` was NOT a shipped flag; tanjiro PR #1027 is implementing it inline to unblock chain-run continuation for multiple PRs
