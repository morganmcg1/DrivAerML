# SENPAI Research State — yi branch (DrivAerML)

- **Date:** 2026-05-05 21:38 UTC
- **Advisor branch:** yi
- **Active students:** 16 (all GPUs occupied, zero idle)
- **Last triage cycle (21:38 UTC):** 0 review-ready, 0 idle. **senku #714 CLOSED** — depth dead end (6L EP2=12.18% vs gate ≤9%, trajectory decelerating). **tanjiro #671 CLOSED** — y-symmetry pair loss diverged; torchrun orphaned, pod recycled via kubectl delete at 21:35 UTC. Both students reassigned: **senku → PR #743 multi-checkpoint inference ensemble** (dc031qpt + pxsnrw36 averaging), **tanjiro → PR #744 per-case hard-mining polish from SOTA** (β-sweep 0.5/1.0). **Norman #724 now at val 7.3592% (-0.0175pp vs SOTA, still training)** — new leader. **Alphonse #731 at step 4282 = 7.3758%, kohaku #719 at step 7816 = 7.3763%** — both near SOTA very early. **Violet #725 diverging at 22.63%, haku #727 at 24.48%** — both cold-start architectures not converging. **Thorfinn #721 relaunched** after CRPS+beta-NLL confound; corrected arm running. **Senku #714 EP2 kill gate fired correctly.** Emma #733 new training `yf1twmyu` step 1641, healthy.
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
| #724 | norman | Residual correction MLP on frozen SOTA (τ_y/τ_z bias fix via pxsnrw36 features) | **val 7.3592% at step 26657 (LEADS SOTA by 0.0175pp); still training EP4/5** |
| #743 | senku | Multi-checkpoint inference ensemble (dc031qpt + pxsnrw36 uniform averaging) | Assigned 2026-05-05 21:35 UTC |
| #744 | tanjiro | Per-case hard-mining polish from SOTA (β-sweep 0.5/1.0, 5 epochs) | Assigned 2026-05-05 21:38 UTC |
| #725 | violet | Multigrid hierarchical volume attention (target: vol_p 11.37% test → ≤9.0%) | Diverging at 22.63% (step 5558) — cold-start architecture struggling |
| #726 | gilbert | SAM optimizer polish from yi SOTA (ρ sweep 0.02/0.05/0.10) | Running (Arm A started 19:13 UTC, SAM ~2h56m/epoch) |
| #727 | haku | Geometry-aware mixup (kNN surface pairs, α=0.2/0.4) | Diverging at 24.48% (step 5839) — cold-start mixup struggling |

---

## Round 40 — Active WIP PRs

| PR | Student | Hypothesis | Last Update / State |
|----|---------|------------|---------------------|
| #721 | thorfinn | CRPS/MAE loss for τ_y/τ_z (correct arm: beta-nll-beta -1.0, MSE cp/vp + MAE ws) | Running after CRPS+beta-NLL confound fix; pod GPUs at 100% 21:12-21:32 UTC |
| #720 | nezuko | Surface-tangent frame τ targets (Option B: no mask, predict all 3 tangent components) | Run `8w7f1b4e` relaunched after premature kill; EP2 val expected ~23:30 UTC |
| #719 | kohaku | SDF-stratified volume norm (Phase 1b done: gap = 4 restored test cases, not SDF) | Phase 2 smoke run `th6fnceg` at step 7816 = **7.3763% (near SOTA)**; healthy |
| #731 | alphonse | EMA snapshot ensemble TTA (K=1/3/5, variance reduction) | Step 4282 = **7.3758% (near SOTA)** — very close to SOTA at only 1.59h |
| #715 | askeladd | Annealed per-axis wallshear loss weighting | Step 16698 = 8.714%, above SOTA; plateaued |
| #713 | fern | Normal-penalty wallshear tangency regularizer — Arm A done (14.0%), Arm B running | Arm A `m8fq2dvb` finished at 14.0%; pod GPUs at 100% = Arm B in flight |
| #672 | edward | Decoupled τ_y/τ_z MLP head polish `z7724dbt` | EP1 7.4088% (head warming; all components ↓ vs SOTA); 3 epochs to go |
| #739 | chihiro | Curvature-weighted loss polish from SOTA (α=0.5/1.5 sweep) | Assigned 2026-05-05 22:00 UTC |
| #733 | emma | Polish-on-SOTA dual-tower bridge (graft cross-attn onto dc031qpt) | New run `yf1twmyu` started 20:56 UTC, step 1641, healthy |
| #652 | frieren | Muon optimizer + Lion polish chain | Running |

---

## Hottest Leads This Round

**Norman #724 (Residual Correction MLP, hidden_dim=64):** val_abupt = **7.3592%** at step 26657 (LEADS SOTA by 0.0175pp). Trajectory: EP1 7.3790% → EP2 7.3662% → EP3 7.3610% → current 7.3592% (monotonically descending). Identity-init verified; all channels improving (τ_y largest at -0.059pp). Still training — will hit terminal SENPAI-RESULT in ~60-80 min. **Merge-eligible once terminal result posted.**

**Alphonse #731 (EMA snapshot ensemble):** val_abupt = **7.3758%** at step 4282 (1.59h in) — within 0.001pp of SOTA after only 1.59h, pre-val. If this trajectory holds into val checkpoints, alphonse may also land near/below SOTA purely from EMA stabilization.

**Kohaku #719 (SDF augment smoke run):** val_abupt = **7.3763%** at step 7816 — near SOTA. Phase 1b showed val/test gap is NOT SDF-related (4 outlier "restored" test cases). SDF augment smoke run surprisingly healthy. **Major structural finding from kohaku: vol_p val/test gap is a data-quality issue (4 restored CFD cases in test), not a model failure.**

**Edward #672 (Decoupled τ_y/τ_z head, polish):** EP1 `z7724dbt` = 7.4088% at step 9335 (head warming on param-group split: head lr=1e-5, trunk lr=1e-6). All components improving. 3+ more epochs remaining.

---

## Closed This Round

| PR | Student | Outcome |
|----|---------|---------|
| #714 | senku | CLOSED — 6L/512d depth dead end; EP2=12.18% (gate ≤9% fired); trajectory decelerating; 4L SOTA out-trains 6L in early epochs |
| #671 | tanjiro | CLOSED — y-symmetry pair loss diverged (val 17.93% slope +0.018/1k at step 25k); peak val ~8.17% at EP2 shows mechanism works but needs cosine decay + tighter clip_grad_norm |
| #668 | gilbert | CLOSED — asinh wallshear norm null; chain-rule inversion → weaker not stronger gradient on tails |
| #661 | haku | CLOSED — RFF surface-xyz null (+ structural confirmation with #674); input features saturated |
| #674 | violet | CLOSED — surface normal RFF null; τ_y/τ_z regressed most, input features saturated |
| #718 | alphonse | CLOSED — selective y-flip TTA null (+12.3% τ_y regression); SOTA not y-equivariant (no aug training) |
| #654 | emma | CLOSED — DualTowerTransolver cold-start null vs SOTA; strong vol_p val signal but didn't generalize to test |
| #662 | chihiro | CLOSED — k1_k2 curvature features do not compound with SOTA stack; PR #580 win was artifact of weaker baseline |
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
- **Y-flip training augmentation (train-time data aug, not pair loss)** — alphonse #718 proved SOTA isn't y-equivariant. Training with 50% y-flip augmentation should close that gap with stable long-run. Assign when a student becomes idle.
- **Y-symmetry pair loss revisited** — tanjiro's #671 peak val ~8.17% at EP2 confirms mechanism works; needs cosine LR decay (1e-4→1e-7 over 30 epochs) + clip_grad_norm=0.25 + hard kill gate val>11% from step 10k.
- **Stacking ensemble K=3** — if senku #743 (K=2) wins, immediately add norman's terminal checkpoint as K=3.
- **RANS divergence-free constraint** — soft physics loss on volume velocity; targets vol_pressure generalization via physical constraint

### Architecture (bold)
- **Geometry-aware mixup** — kNN-constrained mixup on geometrically similar vehicle pairs (Card 8 from round-40 research file)
- **Equivariant backbone** — E(3)/SE(3)-equivariant architecture for τ_y/τ_z; symmetry-guaranteed (distinct from data augmentation)
- **4L/640d/10h width** — isolated width test beyond 512d

### Physics-informed
- **Boundary layer thickness feature** — estimate δ_99 from geometry as surface input
- **SDF-augmented volume features** — append `[sdf², log|sdf|]` to volume input (Phase 2 after #719 diagnostic)
