# SENPAI Research State — yi branch (DrivAerML)

- **Date:** 2026-05-05 23:10 UTC
- **Advisor branch:** yi
- **Active students:** 16 (all GPUs occupied, zero idle)
- **Last triage cycle (23:10 UTC):** 2 PRs merged. **Norman #724 MERGED** → new bar 7.3588%/8.6884% (residual correction MLP, `u7obwlh7`). **Senku #743 awaiting terminal marker** (K=2 ensemble = 7.2733%, −0.086pp vs new bar, will merge as soon as terminal posted). **Norman reassigned → PR #747** (Stage 2: correction MLP + unfreeze last Transolver layer from u7obwlh7). **Haku #727 closed** (geometry-aware mixup +17pp regression), **haku reassigned → PR #746** (y-flip training aug). **Edward #672** at EP2 val 7.3660%, projecting to **~7.28% by EP4** at current slope (−0.0079%/1k) — potential major win; EP3 val expected ~23:55 UTC.
- **Current merge bar:** val_abupt = **7.3588%**, test_abupt = **8.6884%** (PR #724, norman, residual correction MLP, W&B run `u7obwlh7`)
- **Aspirational target:** val_abupt ~7.0% (tay branch SOTA PR #511, `5o7jc7wi`)
- **Next likely merge:** Senku #743 K=2 ensemble (7.2733%) once terminal marker posted; edward #672 EP4 (~01:55 UTC) if trajectory holds

---

## Latest Human Research Directives (from Issue #18)

- **Bold architecture changes**: Don't rely on incremental tweaks — completely replace model architectures (Perceiver-IO, neural operators, equivariant networks). Only constraint is maintaining strong logging, validation, and checkpointing.
- **Cross-branch inspiration**: Before finalizing hypothesis assignments, scan PRs from `noam` and `radford` branches in wandb/senpai for prior art.
- **Epoch-limited signal detection**: Use gradient norms, weight histograms, and loss slopes from W&B to identify gradient-healthy runs hitting the epoch cap.
- **Wall shear structural fix**: Surface-tangent frame wall-shear head (highest priority), Perceiver-IO backbone, asinh/log normalization, RANS divergence-free constraint, 1-cycle LR.

No new human messages since last check (most recent human comment 2026-04-29).

---

## Current Research Focus and Themes

### Primary Gap: Wall Shear τ_y/τ_z (~2.6× and ~3.0× above AB-UPT)

Current yi SOTA per-axis (val/test, PR #724 norman, u7obwlh7):
- τ_y: 9.5185% val / 9.5287% test vs AB-UPT 3.65% → still ~2.6× gap
- τ_z: 11.0188% val / 10.7254% test vs AB-UPT 3.63% → still ~3.0× gap
- surface_p: 4.8440% val / 4.6156% test
- vol_p: 4.3156% val / 11.4062% test

### Secondary Gap: Volume Pressure val/test anomaly

- vol_pressure: 4.31% val vs 11.37% test — **structurally explained by 4 "restored" CFD test cases** (run_133/226/203/158, kohaku #719 Phase 1b). These cases have pathological SDF distributions from a different mesh-generation pipeline. Excluding them: test vol_p ≈ 4.07% ≈ val 4.15%.

### Key Structural Findings

1. **Residual correction MLP works** (norman #724 MERGED): tiny 37.7k-param per-point MLP on frozen backbone gets −0.018pp; largest gain on τ_y. Stage 2 (partial unfreeze) is next.
2. **Multi-checkpoint ensemble works** (senku #743, pending merge): K=2 uniform avg of dc031qpt+pxsnrw36 → 7.2733%, −0.086pp vs new bar. Pearson r≈0.999 but residual errors still decorrelate usefully.
3. **Decoupled τ_y/τ_z head works** (edward #672, running): 10× head LR vs trunk LR gives clean gradient separation; EP2 at 7.3660% with strong slope, projecting ~7.28% by EP4.
4. **SOTA is NOT y-equivariant**: confirmed by alphonse #718 + haku #727. Training-time y-flip aug (haku #746) is the structural fix.
5. **Surface input feature saturation**: RFF on normals/xyz = NULL.
6. **vol_p val/test gap is data quality**: 4 restored test cases, not model failure.

---

## Round 42 — Active Assignments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #747 | norman | Stage 2: correction MLP + unfreeze last Transolver layer from u7obwlh7 | Assigned 2026-05-05 23:05 UTC |
| #746 | haku | Y-flip training augmentation (50% prob, cold-start, 30 epochs) | Assigned 2026-05-05 22:10 UTC |
| #743 | senku | Multi-checkpoint inference ensemble (K=2, pending terminal marker for merge) | **val 7.2733% — AWAITING TERMINAL MARKER** |

---

## Round 41 — Active WIP PRs

| PR | Student | Hypothesis | Last Update / State |
|----|---------|------------|---------------------|
| #672 | edward | Decoupled τ_y/τ_z MLP head polish (head_lr=1e-5, trunk_lr=1e-6) | **EP2 val 7.3660% (step 11129), projecting ~7.28% by EP4; EP3 expected ~23:55 UTC** |
| #744 | tanjiro | Per-case hard-mining polish from SOTA (β-sweep 0.5/1.0, 5 epochs) | Running (assigned 21:38 UTC) |
| #739 | chihiro | Curvature-weighted loss polish from SOTA (α=0.5/1.5 sweep) | Running |
| #733 | emma | Polish-on-SOTA dual-tower bridge (graft cross-attn onto dc031qpt) | Architecture deviations approved (−curvature features, −β-NLL); run started |
| #731 | alphonse | EMA snapshot ensemble TTA (K=5 via --snapshot-save-every-steps 2000) | `y2xnzk6w`, restarted ~21:29 UTC; no val yet |
| #726 | gilbert | SAM optimizer polish (ρ sweep 0.02/0.05/0.10) | Arm A `0z86xbcu` at step ~5936 (4h in, EP1 due ~22:09 UTC) |
| #725 | violet | Multigrid hierarchical volume attention (cold-start, 8 epochs) | `7dr0vcvh` at step 7129, val 22.64% — cold-start descent; pod alive |
| #721 | thorfinn | CRPS/MAE loss for τ_y/τ_z (corrected arm: dropped β-NLL confound) | `kn3nne8i` running, no val yet |
| #720 | nezuko | Surface-tangent frame τ targets (Option B v2) | `8w7f1b4e` running, no val yet |
| #719 | kohaku | SDF augment Phase 2 smoke run | `th6fnceg` val 7.3763% (stalled since step 5442 — may be between val checkpoints) |
| #715 | askeladd | Annealed per-axis wallshear loss weighting | `1qpqhyrt` at step 19,575, val 8.714%, plateaued; sent back for per-axis diagnosis |
| #713 | fern | Normal-penalty wallshear tangency regularizer | Arm A finished (13.882%, regression); Arm B not launched; sent back for status |
| #652 | frieren | Muon optimizer + Lion polish chain | Running |

---

## Hottest Leads This Round

**Senku #743 (Multi-checkpoint K=2 ensemble):** K=2 ensemble of dc031qpt+pxsnrw36 = **val 7.2733%** (−0.086pp vs new bar). All channels improved. Test 8.5989%. Awaiting terminal SENPAI-RESULT to merge. **Once merged, will become new bar; senku to be reassigned to K=3 (add u7obwlh7).**

**Edward #672 (Decoupled τ_y/τ_z MLP head):** EP2 = **7.3660%** at step 11129. Slope −0.0079%/1k, τ_y slope −0.0189%/1k. **Projected EP4 ≈ 7.28%** (linear). If confirmed at EP3, will be a clean merge candidate. τ_y 9.5531% → projected ~9.34% by EP4.

**Norman #747 (Stage 2 correction MLP + partial unfreeze):** New assignment from u7obwlh7. Expected to add another 0.02–0.08pp on top of Stage 1 by letting the top Transolver layer re-tune to the correction target.

---

## Merged This Round (cumulative)

| PR | Student | Result |
|----|---------|--------|
| #724 | norman | MERGED (val 7.3588%, test 8.6884%, new bar −0.018pp vs prior) |
| #681 | nezuko | (prior SOTA, val 7.3767%) |

---

## Closed This Round

| PR | Student | Outcome |
|----|---------|---------|
| #727 | haku | CLOSED — geometry-aware mixup +17.1pp regression, EP1 gate |
| #714 | senku | CLOSED — 6L depth dead end |
| #671 | tanjiro | CLOSED — y-symmetry pair loss diverged |
| #668 | gilbert | CLOSED — asinh wallshear norm null |
| (more in EXPERIMENTS_LOG.md) | | |

---

## Potential Next Research Directions (Round 42+)

### High priority (active or just assigned)
- **Y-flip training augmentation** ← haku #746 ACTIVE
- **Stage 2 correction MLP + unfreeze** ← norman #747 ACTIVE
- **K=3 ensemble** (add u7obwlh7 to K=2) ← senku assignment-on-deck
- **Y-symmetry pair loss stabilized** — cosine LR 1e-4→1e-7, clip 0.25, cold-start 30ep (researcher Card 1)
- **Cosine LR warm-restart on Lion** — escape shallow basins (researcher Card 2)

### Architecture (bold)
- **Cholesky correlated NLL for τ_y/τ_z** — joint covariance β-NLL (researcher Card 3)
- **δ_99 boundary layer thickness feature** — SDF-gradient walk estimate (researcher Card 4)
- **Slice-token RoPE** — 3D RoPE on Transolver centroids (researcher Card 5, parameter-free)
- **SE(3)-equivariant backbone** — architecture-level symmetry (highest potential)
- **4L/640d/10h width** — unexplored width dimension

### Physics-informed
- **RANS divergence-free constraint** — soft physics loss on volume velocity
- **SDF-augmented volume features Phase 2** — append [sdf², log|sdf|] to volume input
