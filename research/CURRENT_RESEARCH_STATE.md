# SENPAI Research State — yi branch (DrivAerML)

- **Date:** 2026-05-04 (last update: 2026-05-04)
- **Advisor branch:** yi
- **Active students:** 16 (all GPUs occupied, zero idle)
- **Current merge bar:** val_abupt = **7.3914%**, test_abupt = **8.7189%** (PR #658, nezuko, EMA best-ckpt EP2 from lr=5e-6 SWA-staged continuation, run `pxsnrw36`)
- **Single-model SOTA (drivaerml-long branch):** val_abupt = **6.5281%**, test_abupt = **7.9303%** (frieren `sogus8sx`, PR #592)
- **Aspirational target:** val_abupt ~7.0% (tay SOTA PR #511, `5o7jc7wi`)
- **Round 38 start:** PR #672 (edward, decoupled τ_y/τ_z MLP head — STALLED, 2 escalations posted)
- **Round 39 start:** PRs #679–#682 (alphonse curvature-focal, fern tangential-wallshear, nezuko terminal LR polish, thorfinn L=6/h=448 budget-fit depth)
- **Round 38 new assignments (2026-05-04):** PR #713 (fern, normal-penalty wallshear tangency), PR #714 (senku, 6L depth retry 900min), PR #715 (askeladd, annealed τ_y/τ_z axis weighting)

---

## Latest Human Research Directives (from Issue #18)

- **Bold architecture changes**: Don't be afraid to completely replace the model backbone. Students can handle radical departures from reference train.py as long as logging, validation, and checkpointing are maintained.
- **Cross-branch inspiration**: Scan PRs from `noam` and `radford` branches in wandb/senpai for prior art on similar techniques before finalizing new hypothesis assignments.
- **Epoch-limited signal detection**: Use gradient norms, weight histograms, and loss slopes from W&B to identify runs that were training well but hit the epoch cap — these deserve follow-up experiments.
- **Wall shear structural fix**: The 4x tau_y/z gap vs AB-UPT is likely a coordinate frame issue. Priority experiments: surface-tangent frame wall-shear head, Perceiver-IO backbone replacement, asinh/log target normalization, RANS divergence-free constraint, 1-cycle LR with 1e-3 peak.
- **Empower students**: Frame assignments to give students latitude to make big changes rather than conservative tweaks.

---

## Current Research Focus and Themes

### Primary Gap: Wall Shear tau_y/tau_z (~2.6x and ~3.0x above AB-UPT)

The dominant remaining failure mode is the anisotropic wall shear components (PR #658 EMA best-ckpt val):
- τ_y: 9.6123% val (test 9.x%) vs AB-UPT 3.65% → ~2.6× gap
- τ_z: 11.0573% val (test 10.x%) vs AB-UPT 3.63% → ~3.0× gap

Confirmed null this round on the τ_y/τ_z gap:
- **Symmetric loss reweighting** (W_y=W_z=4 or 6, PR #628): zero-sum, regresses other channels
- **Asymmetric loss reweighting** (W_y=3, W_z=5 + curvature focal γ=0, PR #646 Arm A): EP3-partial=9.181% — regression on baseline
- **Multi-EMA ensemble** (PR #656): closed — slow shadows stale at 12k steps
- **Standard activation dropout** (PR #638): closed — Arm A EP3=11.16%, Arm B EP3=10.30%, no train↔val gap closure
- **Inverse local point density loss weighting** (PR #636): closed — subsampling noise dominates density signal
- **SDF-proximity volume loss weighting** (PR #622): closed across 3 arms — volume pressure improves on val but does not generalize to test

Hypotheses still active to close the τ_y/τ_z gap:
1. **Decoupled τ_y/τ_z prediction head** (edward, PR #672) — STALLED (2 escalations sent)
2. **DropPath stochastic depth** (kohaku, PR #663) — v2 arms relaunched; EP1 ETA ~12:50 UTC
3. **Surface normal RFF lift** (violet, PR #674) — implementing
4. **Surface position RFF lift** (haku, PR #661) — relaunch from correct GPU pod, results pending
5. **asinh wall-shear normalization** (gilbert, PR #668) — Arm A scale=0.1 EP1=25.88% (FAIL); awaiting Arm B scale=0.5
6. **Cross-slice attention** (frieren, PR #665) — Arm A control EP2=8.70%; Arm B with cross-slice not yet launched
7. **Per-axis output scaling** (fern, PR #664, drivaerml-long branch) — EP8=7.0915%, on track for merge candidate target ≤7.0% at EP10
8. **6L depth scale-up** (thorfinn, PR #666, drivaerml-long branch) — EP2=8.47%, tracking ~0.3pp below L=5 SOTA
9. **Curvature-focal γ=1 + W_y=2/W_z=2** (alphonse, PR #679) — Round 39 cold-start (requires train.py code impl); replaces PR #646
10. **Tangential wallshear loss cold-start** (fern, PR #680) — Round 39 full yi SOTA stack + `--use-tangential-wallshear-loss`
11. **Knowledge distillation from K=7 teacher ensemble** (nezuko, PR #676) — UNSTARTED, 1 escalation sent
12. **Terminal LR polish lr=3e-7** (nezuko, PR #681) — Round 39 resume from PR #658 checkpoint (pxsnrw36)
13. **Y-symmetry augmentation** (tanjiro, PR #671) — alternating mirror, training EP1
14. **7-sigma denser STRING PE** (tanjiro, PR #673, drivaerml-long branch) — EP5=8.88%, passes EP5 gate
15. **Weight decay sweep** (fern, PR #667, drivaerml-long branch) — Arm A control wd=5e-4 EP3=7.30%; Arm B wd=1e-3 queued
16. **Extended cosine T_max=60** (tanjiro, PR #670) — smoke training, awaiting EP1
17. **Tau per-channel weighting v3 long run** (frieren, PR #669) — STALLED, awaiting v3 W&B run ID after 30+ min
18. **Perceiver-IO backbone** (norman, PR #675) — UNSTARTED, no comments yet
19. **L=6/hidden=448/heads=7 budget-fit depth** (thorfinn, PR #682) — Round 39 fresh cold-start architecture test

### Secondary Gap: Volume Pressure (~1.88x above AB-UPT on test)

- vol_pressure: 11.46% test vs AB-UPT 6.08% → 1.88× gap on test (anomalous 4.41% val vs 11.46% test)
- **Bold architecture: DualTowerTransolver** (emma, PR #654) — EP3 cold-start=10.69%, request changes posted: resume from checkpoint at 32k points, 12h budget, target EP6 < 9.0%
- **Muon optimizer** (frieren, PR #652) — Arm A EP3-partial=8.45% cold-start (faster than Lion EP2=11.20%), continuation arm at lr=1e-5 from checkpoint authorized

### Closed This Round (R37–39)

| PR | Student | Outcome |
|----|---------|---------|
| #622 | thorfinn | CLOSED — SDF-proximity volume loss weighting null across 3 arms (best test=12.07%, vol_p val/test gap reveals near-surface overfitting on training geometries) |
| #628 | edward | CLOSED — symmetric τ-weight sweep zero-sum, replaced by PR #672 |
| #636 | gilbert | CLOSED — inverse density weighting fails because random subsampling destroys density signal; future fix would compute density on full mesh and propagate weights |
| #638 | tanjiro | CLOSED — dropout p=0.05/0.10 both above 9% gate; β-NLL+Lion+grad-EMA already provides sufficient regularization |
| #646 | alphonse | CLOSED — Arm A asymmetric W_y=3/W_z=5 EP3=9.181% REGRESSION; Arm B curvature-focal γ=1 EP3=8.9488% REGRESSION vs 7.3914% bar; follow-on PR #679 issued |
| #656 | violet | CLOSED — multi-EMA ensemble null (slow shadow stale at 12k steps) |
| #659 | norman | CLOSED — 4L/768d cold-start undertrained (EP2=10.03% in 3-epoch budget); width hypothesis not falsified, suggested 4L/640d/10h fits within slices=128 for fair retest |

### Recently Merged (R37)

- **PR #657** (fern, lr=1e-6 ultra-low LR continuation from PR #637): val_abupt 7.5373% → 7.4861%, test 8.8533% → 8.8110%
- **PR #658** (nezuko, EMA best-ckpt from lr=5e-6 SWA-staged): val_abupt 7.4861% → **7.3914%** (current bar), test 8.8110% → **8.7189%**
  - Key lesson: EMA decay=0.999 dominates SWA uniform average in flat-basin near-converged regime; SWA only beneficial in cold-start exploring regime

---

## Active Experiments (WIP PRs, Round 37–39)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #715 | askeladd | Annealed per-axis wallshear weighting W_target=3/6, τ_y/τ_z | Round 38 — freshly assigned, cold-start |
| #714 | senku | 6L/512d depth retry with SENPAI_TIMEOUT=900 + --lr-warmup-epochs 0 | Round 38 — freshly assigned, cold-start |
| #713 | fern | Normal-penalty wallshear tangency regularizer λ=0.1/1.0 | Round 38 — freshly assigned, cold-start |
| #682 | thorfinn | L=6/hidden=448/heads=7 budget-fit depth | Round 39 — cold-start, awaiting EP1 |
| #681 | nezuko | Terminal LR polish lr=3e-7 from PR #658 ckpt | Round 39 — resume started |
| #680 | fern | Tangential wallshear loss cold-start | Round 39 — cold-start, awaiting EP1 |
| #679 | alphonse | Curvature-focal γ=1 + W_y=2/W_z=2 (code impl required) | Round 39 — awaiting train.py implementation |
| #676 | nezuko | Knowledge distillation from K=7 teacher ensemble | UNSTARTED — 1 escalation sent |
| #675 | norman | Perceiver-IO backbone replacement | UNSTARTED |
| #674 | violet | Surface normal RFF (Option A, σ=4) | Implementation in progress |
| #673 | tanjiro | 7-sigma STRING PE (drivaerml-long branch) | EP5=8.88% PASS; long run to EP50 |
| #671 | tanjiro | y-symmetry augmentation (alternating mirror) | EP1 training ~22% |
| #670 | tanjiro | Cosine T_max=60 SOTA continuation | Smoke launched, no EP1 yet |
| #669 | frieren | Tau-pc τ_y×1.2/τ_z×1.3 long-run v3 | STALLED — no W&B run ID after 30+ min |
| #668 | gilbert | asinh wall-shear normalization | Arm A scale=0.1 EP1=25.88% FAIL; Arm B scale=0.5 pending |
| #667 | fern | Weight decay sweep (drivaerml-long) | Arm A wd=5e-4 EP3=7.30% PASS; Arm B wd=1e-3 queued |
| #666 | thorfinn | 6L depth (drivaerml-long) | EP2=8.47% PASS, ~0.3pp below L=5 SOTA |
| #665 | frieren | Cross-slice attention | Arm A control EP2=8.70% PASS |
| #664 | fern | Per-axis output scaling (drivaerml-long) | EP8=7.0915% — close to merge target ≤7.0% |
| #663 | kohaku | DropPath p=0.05/0.10 sweep | v1 killed by inverted threshold; v2 EP1 ETA 12:50 UTC |
| #662 | chihiro | k1_k2 curvature compose with yi SOTA stack | Cold-start hit timeout; request changes posted (control + treatment ablation) |
| #661 | haku | Surface position RFF lift dim=64/128 σ=10 | Re-launching from correct GPU pod after NCCL bug |
| #657 | fern | (already merged) | — |
| #658 | nezuko | (already merged) | — |
| #655 | senku | 6L/512d depth (yi stack) | Arm A timed out EP2=9.76% PASS; Arm B 192-slices needs bs=2 + expandable_segments |
| #654 | emma | DualTowerTransolver | EP3 cold-start=10.69%; request changes — resume from ckpt, 32k points, 12h |
| #653 | askeladd | OneCycleLR 1-cycle | Arm A max_lr=1e-3 EP1=106.37% FAIL; Arm B max_lr=5e-4 launched |
| #652 | frieren | Muon optimizer | Arm A EP3-partial=8.45% cold-start; Arm B lr=1e-3 launched; lr=1e-5 continuation arm authorized |
| #646 | alphonse | Asymmetric W_y=3 W_z=5 + curvature focal | CLOSED — Arm A γ=0 EP3=9.181% REGRESSION; Arm B γ=1 EP3=8.9488% REGRESSION vs 7.3914% bar |

---

## Potential Next Research Directions

### High Priority (τ_y/τ_z gap closure)

- **Coordinate frame reformulation**: Decompose τ into surface-tangent frame (PR #627 frieren follow-up) — physics-correct
- **Streamline-aligned coordinate frame**: Rotate τ targets into flow-aligned frame before prediction
- **Density-on-full-mesh**: Recompute local point density on the full surface mesh and propagate per-point weights through the 65k subsampling pipeline (revives PR #636 hypothesis with the bias removed)
- **Annealed feature warmup**: Ramp new loss weights from 0 → target over EP1–EP5 to avoid cold-start destabilization
- **Surface-attention pre-pass**: Apply attention over surface-only tokens before main cross-attention for wall-adjacent geometry
- **Full GNN backbone replacement**: Message passing over surface mesh neighborhoods (vs only as a pre-encoder)

### Architecture Experiments

- **4L/640d/10h ultra-wide configuration** (norman PR #659 follow-up): fits within slices=128 budget for fair test
- **8L/512d ultra-deep configuration**: if 6L (PR #666) shows improvement over 4L, push to 8L
- **U-Net style skip connections in Transolver**: long-range residuals between early and late layers

### Training and Optimization

- **lr=3e-7 terminal polish from PR #658 checkpoint**: IN FLIGHT PR #681 (nezuko). If successful, continue chain with lr=1e-7.
- **Muon + lr=1e-5 polishing continuation arm** (PR #652 follow-up authorized)
- **Mixed-resolution training**: 65536 surface points training, 16384-point val pass every epoch
- **Curriculum learning**: low-curvature → high-curvature samples
- **Knowledge distillation** (PR #676 in flight) — soft-label averaging from K=7 teacher runs

### Physics-Informed

- **Continuity equation soft constraint**: divergence-free velocity field as volume regularizer
- **Bernoulli pressure constraint**: soft surface-pressure ↔ freestream linkage
- **Boundary layer thickness feature**: estimate δ from geometry as surface feature
- **Volume pressure anomaly investigation**: understand why vol_pressure val (4.41%) diverges from test (11.46%) — distribution shift, not loss form
