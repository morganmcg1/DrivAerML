# SENPAI Research State — yi branch (DrivAerML)

- **Date:** 2026-05-05 22:10 UTC
- **Advisor branch:** yi
- **Active students:** 16 (all GPUs occupied, zero idle)
- **Last triage cycle (22:10 UTC):** 0 review-ready, 0 idle after haku reassigned. **Haku #727 CLOSED** — geometry-aware mixup fundamental mechanism failure (val 24.476%, +17pp regression, EP1 gate failed). **Haku reassigned → PR #746 y-flip training augmentation (50% prob, cold-start).** Norman #724 (residual MLP polish) finished training at val **7.3588%** (beats SOTA by -0.0179pp, test 8.6884%); nudged for terminal SENPAI-RESULT. **Edward #672 (decoupled τ_y/τ_z head) now at 7.3660% at step 11129 — beats SOTA by -0.011pp**, still training. Both norman+edward are merge candidates once terminal posted.
- **Current merge bar:** val_abupt = **7.3767%**, test_abupt = **8.7015%** (PR #681, nezuko, terminal LR polish lr=3e-7, W&B run `dc031qpt`)
- **Aspirational target:** val_abupt ~7.0% (tay branch SOTA PR #511, `5o7jc7wi`)

---

## Latest Human Research Directives (from Issue #18)

- **Bold architecture changes**: Don't rely on incremental tweaks — completely replace model architectures (Perceiver-IO, neural operators, equivariant networks). Only constraint is maintaining strong logging, validation, and checkpointing.
- **Cross-branch inspiration**: Before finalizing hypothesis assignments, scan PRs from `noam` and `radford` branches in wandb/senpai for prior art.
- **Epoch-limited signal detection**: Use gradient norms, weight histograms, and loss slopes from W&B to identify gradient-healthy runs hitting the epoch cap.
- **Wall shear structural fix**: Surface-tangent frame wall-shear head (highest priority), Perceiver-IO backbone, asinh/log normalization, RANS divergence-free constraint, 1-cycle LR.

No new human messages since last check (2026-05-05 22:10 UTC; most recent human comment 2026-04-29).

---

## Current Research Focus and Themes

### Primary Gap: Wall Shear τ_y/τ_z (~2.6× and ~3.0× above AB-UPT)

Current yi SOTA per-axis (val/test, PR #681 nezuko):
- τ_y: 9.5832% val / 9.5964% test vs AB-UPT 3.65% → ~2.6× gap
- τ_z: 11.0377% val / 10.7383% test vs AB-UPT 3.63% → ~3.0× gap
- surface_p: 4.8515% val / 4.6236% test vs AB-UPT 3.82% → 1.27× gap

### Secondary Gap: Volume Pressure val/test anomaly

- vol_pressure: 4.31% val vs 11.37% test (2.64× ratio) — **structurally explained by 4 "restored" CFD test cases** (run_133/226/203/158, kohaku #719 Phase 1b finding). These cases have pathological SDF distributions (no negative-SDF points inside car body) from a different mesh-generation pipeline. Excluding them: test vol_p ≈ 4.07% ≈ val 4.15%. The gap is largely data-quality, not model failure.

### Key Structural Findings

1. **Surface input feature saturation**: RFF on normals/xyz = NULL (#674, #661). LinearProjection of [nx,ny,nz,area] + STRING-sep PE is sufficient. Future input-feature experiments need strong mechanistic differentiation.
2. **SOTA is NOT y-equivariant**: alphonse #718 (inference TTA) proved this. Training-time y-flip aug (haku #746) is the structural fix.
3. **Geometry-aware mixup ≠ physics augmentation**: kNN body-style pairing too coarse for per-point field data. Cross-stream trunk corruption via shared parameters.
4. **vol_p val/test gap is data-quality**: 4 restored test cases, not model failure.
5. **Residual correction MLP works**: Norman #724 at 7.3588% via identity-init MLP head on frozen SOTA, attributing gain to τ_y correction (-0.065pp, largest channel).

---

## Round 41 — Active Assignments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #724 | norman | Residual correction MLP on frozen SOTA (τ_y/τ_z bias fix) | **FINISHED: val 7.3588%, test 8.6884% — BEATS SOTA; awaiting terminal SENPAI-RESULT** |
| #672 | edward | Decoupled τ_y/τ_z MLP head polish (head_lr=1e-5, trunk_lr=1e-6) | **val 7.3660% at step 11129 — BEATS SOTA by -0.011pp; still training EP2** |
| #746 | haku | Y-flip training augmentation (50% prob, cold-start, 30 epochs) | Assigned 2026-05-05 22:10 UTC |
| #743 | senku | Multi-checkpoint inference ensemble (dc031qpt + pxsnrw36 averaging) | Running (assigned 21:35 UTC) |
| #744 | tanjiro | Per-case hard-mining polish from SOTA (β-sweep 0.5/1.0, 5 epochs) | Running (assigned 21:38 UTC) |
| #725 | violet | Multigrid hierarchical volume attention (cold-start, 8 epochs) | At 22.64% step 7129 — cold-start descent expected, let run |
| #726 | gilbert | SAM optimizer polish (ρ sweep 0.02/0.05/0.10) | Arm A `0z86xbcu` at step 5936 (3h17m, no val yet; EP1 ~22:09 UTC) |
| #739 | chihiro | Curvature-weighted loss polish from SOTA (α=0.5/1.5 sweep) | Running |

---

## Round 40 — Active WIP PRs

| PR | Student | Hypothesis | Last Update / State |
|----|---------|------------|---------------------|
| #731 | alphonse | EMA snapshot ensemble TTA (K=5 via --snapshot-save-every-steps 2000) | Restarted ~21:29 UTC (`y2xnzk6w`), 1h02m in at step 2939, no val yet |
| #721 | thorfinn | CRPS/MAE loss for τ_y/τ_z (corrected arm: no β-NLL confound) | `kn3nne8i` at step 8096, 2h55m in, no val yet |
| #720 | nezuko | Surface-tangent frame τ targets (Option B v2, relaunched after premature kill) | `8w7f1b4e` at step 7060, 2h36m in, no val yet |
| #719 | kohaku | SDF augment Phase 2 smoke run | `th6fnceg` at step 9916, val 7.3763% (stalled since step 5442 — may be between val checkpoints) |
| #715 | askeladd | Annealed per-axis wallshear loss weighting | `1qpqhyrt` at step 19,575, val 8.714%, slope -0.79%/1k; plateau; sent back for per-axis diagnosis |
| #713 | fern | Normal-penalty wallshear tangency regularizer | Arm A `m8fq2dvb` finished at val 13.882%/test 15.02%; Arm B NOT LAUNCHED; sent back for status |
| #733 | emma | Polish-on-SOTA dual-tower bridge (graft cross-attn onto dc031qpt) | `yf1twmyu` at step 3597, no val yet |
| #652 | frieren | Muon optimizer + Lion polish chain | Running |

---

## Hottest Leads This Round

**Norman #724 (Residual Correction MLP, hidden_dim=64, frozen SOTA):** val_abupt = **7.3588%** at step 27216 (EP5 terminal, all 5 epochs complete). Trajectory: EP1 7.3790% → EP2 7.3662% → EP3 7.3610% → terminal 7.3588%. **Beats SOTA by -0.0179pp.** Test 8.6884% (beats test SOTA 8.7015% by -0.013pp). τ_y gain is largest (-0.065pp), validating the hypothesis. **Merge-eligible once terminal SENPAI-RESULT posted.** Nudged at 22:05 UTC.

**Edward #672 (Decoupled τ_y/τ_z MLP head, head_lr=1e-5, trunk_lr=1e-6):** val_abupt = **7.3660%** at step 11129 (EP2). Trajectory: EP1 7.4088% → EP2 7.3660% (-0.043pp, still descending). **Beats SOTA by -0.011pp**. Per-axis: τ_y 9.5531%, τ_z 11.0230%, sp 4.8510%, vp 4.3022%. Still training — EP3+ likely to improve further. Could compound with norman if both merge.

**Alphonse #731 (EMA snapshot ensemble):** Restarted after 2-epoch plan with `--snapshot-save-every-steps 2000`. Step 2939, ~1h02m in. Prior run had step 4282 = 7.3758% (near SOTA). New run health unknown pending first val.

**Kohaku #719 (SDF smoke, structural finding):** val 7.3763% since step 5442. Phase 1b finding (vol_p gap = data quality) is the headline contribution regardless of whether the smoke run improves. If the smoke run stays near SOTA through the full budget, the SDF augment may be a free near-SOTA training variant.

---

## Closed This Round

| PR | Student | Outcome |
|----|---------|---------|
| #727 | haku | CLOSED — geometry-aware mixup fundamental failure (val 24.476%, +17.1pp, EP1 gate); mechanism: chimeric targets + cross-stream trunk corruption |
| #714 | senku | CLOSED — 6L/512d depth dead end; EP2=12.18% (gate ≤9% fired); trajectory decelerating |
| #671 | tanjiro | CLOSED — y-symmetry pair loss diverged (val 17.93% slope +0.018/1k at step 25k); peak val ~8.17% at EP2 shows mechanism works but needs cosine decay + tighter clip_grad_norm |
| #668 | gilbert | CLOSED — asinh wallshear norm null; chain-rule inversion → weaker gradient on tails |
| #661 | haku | CLOSED — RFF surface-xyz null (+ structural confirmation with #674); input features saturated |
| #674 | violet | CLOSED — surface normal RFF null; τ_y/τ_z regressed most, input features saturated |
| #718 | alphonse | CLOSED — selective y-flip TTA null (+12.3% τ_y regression); SOTA not y-equivariant (no aug training) |
| #654 | emma | CLOSED — DualTowerTransolver cold-start null vs SOTA |
| #662 | chihiro | CLOSED — k1_k2 curvature features do not compound with SOTA stack |
| #697 | alphonse | CLOSED — fourier surface-RFF duplicate of #674 |
| #707 | nezuko | CLOSED — full-mesh volume density duplicate of #719 |
| #675 | norman | CLOSED — Perceiver-IO backbone undertrained (val 29.69%) |
| #659 | norman | CLOSED — 4L/768d cold-start undertrained |

---

## Potential Next Research Directions (Round 42+)

### High priority (active or just assigned)
- **Y-flip training augmentation** ← haku #746 ACTIVE — training-time bilateral symmetry enforcement; structural fix for the SOTA's y-non-equivariance
- **Y-symmetry pair loss revisited** — tanjiro's #671 peak val ~8.17% at EP2 confirms mechanism works; needs cosine LR decay (1e-4→1e-7 over 30 epochs) + clip_grad_norm=0.25 + kill gate val>11% from step 10k. Could be more powerful than y-flip aug if stable.
- **Stacking ensemble K=3** — if senku #743 (K=2) wins, immediately add norman's terminal checkpoint as K=3.
- **RANS divergence-free constraint** — soft physics loss on volume velocity; targets vol_pressure generalization via physical constraint

### Architecture (bold)
- **SE(3)-equivariant backbone** — E(3)/SE(3)-equivariant Transolver variant for τ_y/τ_z; symmetry-guaranteed at architecture level (distinct from data augmentation); highest potential upside
- **4L/640d/10h width** — isolated width test beyond 512d; search vs depth has not been fully explored at larger widths
- **Boundary layer thickness feature** — estimate δ_99 from geometry as surface input; physics-grounded spatial conditioning for wall-shear magnitude

### Physics-informed
- **SDF-augmented volume features Phase 2** — append `[sdf², log|sdf|]` to volume input (build on kohaku #719 Phase 2 smoke)
- **Turbulence intensity as surface feature** — estimated local Re_theta from flow separation geometry; boundary layer transition feature for τ_y/τ_z magnitude
- **Pressure Poisson loss regularizer** — soft constraint on volume pressure satisfying ∇²p ≈ -ρ∇·(u·∇u); orthogonal to RANS div-free

### Training/optimization
- **Cosine decay LR scheduler on Lion (1e-4→1e-7 over 30 epochs)** — may prevent the late-run Lion divergence seen in tanjiro #671 and askeladd #715 plateau; composable with current SOTA stack
- **Mixup at decoder-embedding level** — manifold mixup (Verma 2019) on slice tokens after transformer layers; addresses the surface-level chimera problem from #727
- **Per-epoch checkpoint voting** — ensemble multiple epoch checkpoints from a single run using majority-vote on per-point predictions (different from EMA averaging)
