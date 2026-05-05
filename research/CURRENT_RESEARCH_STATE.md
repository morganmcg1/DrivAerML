# SENPAI Research State — yi branch (DrivAerML)

- **Date:** 2026-05-05 16:30 UTC
- **Advisor branch:** yi
- **Active students:** 16 (all GPUs occupied, zero idle)
- **Current merge bar:** val_abupt = **7.3767%**, test_abupt = **8.7015%** (PR #681, nezuko, terminal LR polish lr=3e-7, W&B run `dc031qpt`)
- **Aspirational target:** val_abupt ~7.0% (tay branch SOTA PR #511)

---

## Latest Human Research Directives (from Issue #18)

- **Bold architecture changes**: Don't rely on incremental tweaks — completely replace model architectures. Only constraint is maintaining strong logging, validation, and checkpointing.
- **Cross-branch inspiration**: Before finalizing new hypothesis assignments, scan PRs from `noam` and `radford` branches in wandb/senpai for prior art.
- **Epoch-limited signal detection**: Use gradient norms, weight histograms, and loss slopes from W&B to identify runs that were training well but hit the epoch cap.
- **Wall shear structural fix**: Surface-tangent frame wall-shear head (highest priority), Perceiver-IO backbone, asinh/log normalization, RANS divergence-free constraint, 1-cycle LR.

No new human messages since last check (2026-05-04).

---

## Current Research Focus and Themes

### Primary Gap: Wall Shear τ_y/τ_z (still ~2.6× and ~3.0× above AB-UPT)

Current yi SOTA per-axis (val, PR #681):
- τ_y: 9.5832% val vs AB-UPT 3.65% → ~2.6× gap
- τ_z: 11.0377% val vs AB-UPT 3.63% → ~3.0× gap
- surface_p: 4.8515% val vs AB-UPT 3.82% → 1.27× gap

### Secondary Gap: Volume Pressure val/test anomaly

- vol_pressure: 4.31% val vs 11.46% test (2.7× ratio) — the largest unexplained residual in current results

---

## Round 40 — New Assignments (2026-05-05)

| PR | Student | Hypothesis | Priority |
|----|---------|------------|----------|
| #718 | alphonse | Selective τ_y TTA at inference (zero training cost, +2.79% τ_y shown in PR #286) | HIGH |
| #719 | kohaku | SDF-stratified volume norm (diagnose + fix 4.3% val / 11.5% test vp gap) | HIGH |
| #720 | nezuko | Surface-tangent frame τ targets (remove Cartesian τ_y/τ_z entanglement) | MEDIUM |
| #721 | thorfinn | CRPS/MAE loss for wall-shear only (vs β-NLL over-smoothing hypothesis) | MEDIUM |

---

## Active WIP PRs — Round 40

| PR | Student | Hypothesis | W&B / State |
|----|---------|------------|-------------|
| #718 | alphonse | Selective TTA τ_y at inference | Round 40 — freshly assigned |
| #719 | kohaku | SDF-stratified volume pressure normalization | Round 40 — freshly assigned |
| #720 | nezuko | Surface-tangent frame τ targets | Round 40 — freshly assigned |
| #721 | thorfinn | CRPS/MAE loss for wall-shear channels | Round 40 — freshly assigned |
| #715 | askeladd | Annealed per-axis wallshear weighting | Running: `1qpqhyrt`, step 2417 |
| #714 | senku | 6L/512d depth retry (900min budget) | Running: `0zr5g357`, step 1260 |
| #713 | fern | Normal-penalty wallshear tangency regularizer | Running: `m8fq2dvb`, step 5028 |
| #675 | norman | Perceiver-IO backbone replacement | Running: `yorhbhi9`, val 29.69% |
| #674 | violet | Surface normal RFF (dim=128, σ=4) | Running: `09qsbtgo`, step 5161 |
| #672 | edward | Decoupled τ_y/τ_z MLP head — sent back for SENPAI-RESULT + polish | Run `o5nplmj9` finished 10.03%; pivot to polish from pxsnrw36 |
| #671 | tanjiro | y-symmetry pair loss (long run) | Running: `wbjsawz7`, EP1=18.12% |
| #668 | gilbert | asinh wall-shear target normalization | Running: `z02089nc`, step 7449 |
| #662 | chihiro | k1_k2 curvature cold-start ablation (Arm A control, 720min) | Running: `4abva8us`, 9.13%; let Arm A finish, skip Arm B |
| #661 | haku | Surface position RFF dim=64/128 EP1 resume | Running at 100% GPU; sent check-in ping for new run IDs |
| #654 | emma | DualTowerTransolver (hottest lead — EP2=8.57%) | Running: `sjq4wvg1`, EP3 expected ~18:00 UTC |
| #652 | frieren | Muon optimizer on full yi stack | Running: `jh3e3r5d`, step 4149 |

---

## Closed This Round (R37–40)

| PR | Student | Outcome |
|----|---------|---------|
| #622 | thorfinn | CLOSED — SDF-proximity volume loss weighting null |
| #628 | edward | CLOSED — symmetric τ-weight sweep zero-sum |
| #636 | gilbert | CLOSED — inverse density weighting null |
| #638 | tanjiro | CLOSED — dropout null |
| #646 | alphonse | CLOSED — asymmetric W_y/W_z + curvature-focal both regression |
| #656 | violet | CLOSED — multi-EMA ensemble null |
| #659 | norman | CLOSED — 4L/768d cold-start undertrained |

---

## Merged (running SOTA chain on yi)

| PR | Student | Result | Key finding |
|----|---------|--------|-------------|
| #681 | nezuko | **7.3767% val / 8.7015% test** ← current bar | lr=3e-7 still extracts gains from near-converged SOTA |
| #658 | nezuko | 7.3914% val / 8.7189% test | SWA EMA best-ckpt; EMA dominates SWA in flat-basin |
| #657 | fern | 7.4861% val / 8.8110% test | lr=1e-6 continuation; diminishing returns |
| #637 | fern | 7.5373% val / 8.8533% test | Extended training at lr=1e-5 from STRING-sep SOTA |
| #576 | frieren | 8.2528% val | STRING-sep PE + Lion + grad-EMA cold-start |
| #590 | thorfinn | 8.686% val | grad-EMA α=0.5 |
| #583 | edward | 8.861% val | β-NLL β=0.5 |
| #517 | askeladd | 9.032% val | Lion lr=1e-4 clip=0.5 confirmed optimal |
| #580 | haku | Infrastructure: κ₁/κ₂ curvature feature flag | |
| #490 | frieren | Infrastructure: STRING-sep learnable PE | |

---

## Potential Next Research Directions (Round 41+)

### High priority (queued for next idle slots)
- **Multigrid hierarchical volume attention** — two-resolution coarse-to-fine for elliptic vol_p (Card 5 from round-40 research file)
- **Residual correction MLP on frozen SOTA** — lightweight bias correction (Card 6)
- **SAM optimizer** — sharpness-aware fine-tuning from yi SOTA (Card 7)
- **Geometry-aware mixup** — kNN-constrained mixup on similar vehicle pairs (Card 8)

### Architecture
- **8L/512d ultra-deep** — push beyond 6L if senku's 6L retry shows improvement
- **DualTower continuation** — emma #654 EP2=8.57%; if EP3 trajectory continues → long-run candidate
- **4L/640d/10h width** — fair width test (follow-up to closed PR #659)

### Physics-informed
- **RANS divergence-free constraint** — soft physics loss on volume velocity field
- **Boundary layer thickness feature** — estimate δ from geometry as surface input
- **SDF-augmented features** — `[sdf^2, log|sdf|]` appended to volume input (Card 2 Phase 2)
