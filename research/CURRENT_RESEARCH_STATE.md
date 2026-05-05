# SENPAI Research State — yi branch (DrivAerML)

- **Date:** 2026-05-05 21:11 UTC
- **Advisor branch:** yi
- **Active students:** 16 (all GPUs occupied, zero idle)
- **Last triage cycle (21:11 UTC):** 0 review-ready, 0 idle. Two strong polish trajectories surfacing: **norman #724 EP3 val_abupt = 7.3610% BELOW SOTA 7.3767%** (-0.0157pp, residual correction MLP, monotonically descending; identity-init verified at step 0 = 0 correction). **edward #672 polish-on-SOTA EP1 = 7.4088%** (head still warming on lr=1e-5 trunk lr=1e-6 split, all components dropping vs SOTA). Both runs continuing to EP4-5; will assess at terminal SENPAI-RESULT. **emma #733 EP0 sanity confirmed val=7.3767% baseline match**; full training launched as `yf1twmyu` at 20:56 UTC (step 775, 17min in). **tanjiro #671 still diverged** at step 24192, 9.0h runtime, val slope +0.018/1k steps positive (worsening); third escalation kill instruction issued at 21:11 UTC. **senku #714 EP1 = 21.524% (cold-start 6L), EP2 hard-kill gate at step 11000 (val ≤9%) will fire automatically**; trust the kill threshold to handle. alphonse #731, violet #725, askeladd #715, fern #713 all running healthily.
- **Current merge bar:** val_abupt = **7.3767%**, test_abupt = **8.7015%** (PR #681, nezuko, terminal LR polish lr=3e-7, W&B run `dc031qpt`)
- **Aspirational target:** val_abupt ~7.0% (tay branch SOTA PR #511, `5o7jc7wi`)

---

## Latest Human Research Directives (from Issue #18)

- **Bold architecture changes**: Don't rely on incremental tweaks — completely replace model architectures (Perceiver-IO, neural operators, equivariant networks). Only constraint is maintaining strong logging, validation, and checkpointing.
- **Cross-branch inspiration**: Before finalizing hypothesis assignments, scan PRs from `noam` and `radford` branches in wandb/senpai for prior art.
- **Epoch-limited signal detection**: Use gradient norms, weight histograms, and loss slopes from W&B to identify gradient-healthy runs hitting the epoch cap.
- **Wall shear structural fix**: Surface-tangent frame wall-shear head (highest priority), Perceiver-IO backbone, asinh/log normalization, RANS divergence-free constraint, 1-cycle LR.

No new human messages since last check (2026-05-04).

---

## Current Research Focus and Themes

### Primary Gap: Wall Shear τ_y/τ_z (~2.6× and ~3.0× above AB-UPT)

Current yi SOTA per-axis (val/test, PR #681 nezuko):
- τ_y: 9.5832% val / 9.5964% test vs AB-UPT 3.65% → ~2.6× gap
- τ_z: 11.0377% val / 10.7383% test vs AB-UPT 3.63% → ~3.0× gap
- surface_p: 4.8515% val / 4.6236% test vs AB-UPT 3.82% → 1.27× gap

### Secondary Gap: Volume Pressure val/test anomaly

- vol_pressure: 4.31% val vs 11.37% test (2.64× ratio) — biggest unexplained residual. Two attacks in flight: SDF-stratified norm (kohaku #719, normalization approach) and multigrid volume attention (violet #725, architecture approach).

### Key Structural Finding This Round (PR #674)

Surface input feature saturation confirmed: RFF on normals was a null result — τ_y/τ_z regressed +0.73/+0.78pp. The SOTA is not bottlenecked by missing normal frequency content. LinearProjection of existing [nx, ny, nz, area] channels is sufficient. Future input-feature experiments need a strong mechanistic differentiation.

---

## Round 41 — Active Assignments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #724 | norman | Residual correction MLP on frozen SOTA (τ_y/τ_z bias fix via pxsnrw36 features) | **EP3 val 7.3610% BELOW SOTA (-0.016pp); monotonically descending; merge-eligible at terminal** |
| #725 | violet | Multigrid hierarchical volume attention (target: vol_p 11.37% test → ≤9.0%) | Assigned 2026-05-05 18:18 UTC |
| #726 | gilbert | SAM optimizer polish from yi SOTA (ρ sweep 0.02/0.05/0.10) | Assigned 2026-05-05 18:35 UTC |
| #727 | haku | Geometry-aware mixup (kNN surface pairs, α=0.2/0.4) | Assigned 2026-05-05 18:38 UTC |

---

## Round 40 — Active WIP PRs

| PR | Student | Hypothesis | Last Update / State |
|----|---------|------------|---------------------|
| #721 | thorfinn | CRPS/MAE loss for τ_y/τ_z (replace β-NLL on wall-shear only) | Running |
| #720 | nezuko | Surface-tangent frame τ targets (remove Cartesian τ_y/τ_z entanglement) | Running |
| #719 | kohaku | SDF-stratified volume norm (diagnose + fix vol_p val/test gap) | Running |
| #731 | alphonse | EMA snapshot ensemble TTA (K=1/3/5, variance reduction for τ_y/τ_z) | Assigned 2026-05-05 19:50 UTC |
| #715 | askeladd | Annealed per-axis wallshear loss weighting | Running (corrected launch 15:19 UTC) |
| #714 | senku | 6L/512d depth retry (900-min budget, run `0zr5g357`) | Running (launched 15:33 UTC) |
| #713 | fern | Normal-penalty wallshear tangency regularizer (λ·|ws·n̂|²) | Running |
| #672 | edward | Decoupled τ_y/τ_z MLP head (polish-on-SOTA z7724dbt) | EP1 7.4088% (head warming, all components ↓ vs SOTA); 3 epochs to go |
| #671 | tanjiro | O(2) y-symmetry pair loss (50-epoch long run) | **DIVERGED step 22616+ (val 17.9%, slope +0.018/1k); 3rd kill escalation 21:11 UTC** |
| #668 | gilbert | asinh wall-shear target normalization | Arm C running (~19:21 UTC terminal) |
| #739 | chihiro | Curvature-weighted loss polish from SOTA (α=0.5/1.5 sweep) | Assigned 2026-05-05 22:00 UTC |
| #661 | haku | Surface RFF (dim=64/128 resume, both arms passing gates) | Running — Arm A ahead at EP2 |
| #733 | emma | Polish-on-SOTA dual-tower bridge (graft cross-attn onto dc031qpt) | Assigned 2026-05-05 21:30 UTC |
| #652 | frieren | Muon optimizer + Lion polish chain (Arm D `jh3e3r5d`) | Running |

---

## Hottest Leads This Round

**Norman #724 (Residual Correction MLP, hidden_dim=64):** EP3 val_abupt = **7.3610%** vs SOTA 7.3767% (-0.0157pp). Identity-init verified (correction abs_mean=0 at step 0 = exact baseline). All channels improved vs SOTA at EP3: surface_p -0.007, τ_y -0.059 (largest gain), τ_z -0.014, vol_p +0.009 (within noise; surface-only correction). Run in EP4/5 with ~80 min remaining. Already merge-eligible if terminal SENPAI-RESULT confirms.

**Edward #672 (Decoupled τ_y/τ_z head, polish):** EP1 of `z7724dbt` = 7.4088% (+0.017pp vs SOTA, head still warming on param-group split: head lr=1e-5, trunk lr=1e-6). All components dropping vs SOTA at EP1. Three more epochs queued; expected to descend below SOTA by EP2-3.

**Tanjiro #671 (y-symmetry pair loss):** DIVERGED — peak val ≈8.17% at EP2 then climbed to 17.929% at step 22616, still rising. Run currently at step 24192, 9.0h runtime; third escalation kill instruction issued at 21:11 UTC.

---

## Closed This Round

| PR | Student | Outcome |
|----|---------|---------|
| #668 | gilbert | CLOSED — asinh wallshear norm null; chain-rule inversion → weaker not stronger gradient on tails |
| #661 | haku | CLOSED — RFF surface-xyz null (+ structural confirmation with #674); input features saturated |
| #674 | violet | CLOSED — surface normal RFF null; τ_y/τ_z regressed most, input features saturated |
| #718 | alphonse | CLOSED — selective y-flip TTA null (+12.3% τ_y regression); SOTA not y-equivariant (no aug training) |
| #654 | emma | CLOSED — DualTowerTransolver cold-start null vs SOTA (EP4 val 7.90% / test 9.04%); strong vol_p val signal (10.60% → 5.17%) but didn't generalize to test |
| #662 | chihiro | CLOSED — k1_k2 curvature features do not compound with SOTA stack (Arm A val 7.91% / test 9.04%); EP1 advantage collapsed by EP2; PR #580 win was artifact of weaker baseline |
| #697 | alphonse | CLOSED — fourier surface-RFF duplicate of #674 |
| #707 | nezuko | CLOSED — full-mesh volume density duplicate of #719 |
| #675 | norman | CLOSED — Perceiver-IO backbone undertrained (val 29.69%) |
| #659 | norman | CLOSED — 4L/768d cold-start undertrained |
| #646 | alphonse | CLOSED — asymmetric W_y/W_z + curvature-focal both regression |
| #638 | tanjiro | CLOSED — dropout null |
| #636 | gilbert | CLOSED — inverse density weighting null |
| #628 | edward | CLOSED — symmetric τ-weight sweep zero-sum |
| #622 | thorfinn | CLOSED — SDF-proximity volume loss weighting null |

---

## Potential Next Research Directions (Round 42+)

### High priority
- **8L/512d ultra-deep** — if senku's 6L retry (#714) shows gains, push to 8L next; if not, 6L stays dead
- **DualTower continuation run** — if emma #654 beats baseline, immediately queue lr=1e-5 or lower continuation from best ckpt
- **RANS divergence-free constraint** — soft physics loss on volume velocity; targets vol_pressure generalization via physical constraint

### Architecture (bold)
- **Geometry-aware mixup** — kNN-constrained mixup on geometrically similar vehicle pairs (Card 8 from round-40 research file)
- **Equivariant backbone** — E(3)/SE(3)-equivariant architecture for τ_y/τ_z; symmetry-guaranteed (distinct from data augmentation)
- **4L/640d/10h width** — isolated width test beyond 512d

### Physics-informed
- **Boundary layer thickness feature** — estimate δ_99 from geometry as surface input
- **SDF-augmented volume features** — append `[sdf², log|sdf|]` to volume input (Phase 2 after #719 diagnostic)
