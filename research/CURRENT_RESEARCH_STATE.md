# SENPAI Research State

- **Date:** 2026-05-05 (updated 17:30 UTC)
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

---

## Latest Human Researcher Directives

**Issue #717 (Volume Pressure Improvement Plan)**: Human team has set an explicit multi-phase programme to reduce `test_primary/volume_pressure_rel_l2_pct` from ~11.5% toward the AB-UPT reference of 6.08%.

**Hard constraint (Issue #717):** Zero model ensembling. All reported metrics must be single-model only.

**Baseline anchors for all VP experiments:**
- `sogus8sx` (PR #599): test vol=11.694%, test wall=7.299%, test tau_y=7.941%, test tau_z=9.535%
- `4k25s25e` (PR #592): test vol=11.933%, test wall=7.334%, test tau_y=8.145%, test tau_z=9.298%

**Issue #717 VP promotion ladder:**
- Weak win: test vol_pressure ≤ 10.8%
- Solid win: test vol_pressure ≤ 10.0%
- Major win: test vol_pressure ≤ 9.5%
- Target: test vol_pressure ≤ 6.08% (AB-UPT reference)

---

## Current Baselines

| Tier | val_abupt | PR | Notes |
|---|---|---|---|
| **Ensemble SOTA** | **6.1751%** | #612 (nezuko) | K=7 greedy pool-24; test=7.5347% |
| **Single-model SOTA** | **6.5985%** | #592 (alphonse) | depth-L5, EP4 step 43,459 |
| **Single-model prior** | 6.7258% | #594 (askeladd) | rff32 |
| **Ensemble gate** | < 6.1751% | — | Must beat to update ensemble SOTA |
| **Single-model gate** | < 6.5985% | — | Must beat to update single-model SOTA |

**Key finding:** Chronic volume_pressure test-vs-val gap (val≈3.6%, test≈11.5%, ~3×) remains the primary systematic issue to resolve.

---

## Current Research Themes

### 1. Volume Pressure Improvement (Issue #717 — Phase 1 Probes)

Active experiments targeting the chronic 3× vol_pressure test-vs-val gap:

- **PR #722** (tanjiro): **Exp 1A** — Dual-tower volume/surface cross-attention. Separate 3L surface encoder + 3L volume encoder + cross-attention bridge. Running — EP1 PASSED (val=8.57%, vol_pressure=6.47%, epoch_time=37.6min ✅).

- **PR #723** (thorfinn): **Exp 1C** — Coordinate normalization geometry conditioning. Per-sample centroid-subtraction + bounding-box scale normalization before RFF encoding. Running — EP1 in progress.

- **PR #728** (frieren): **Exp 1B** — Volume outlier-aware point sampling. Two arms: (A) EMA-residual case-level reweighted DataLoader, (B) geometric-distance stratified near/far volume points (3× up-weight far-wake). Just assigned — pre-EP1.

- **PR #729** (alphonse): **Exp 1D** — Single-model KD from K=7 ensemble (PR #612). Pre-cache ensemble predictions; soft-target loss (alpha=0.5) on vol_pressure only. Tests whether ensemble knowledge can transfer to single model. Just assigned — pre-EP1.

**Closed (null) — do not repeat:**
- BC-type feature embedding (PR #716): KILLED — arm-a EP2 val=26.61% >> 12%; arm-b EP1 val=27.12%, epoch_time=180min >> 80min gate
- Slice count sweep (PR #690): CLOSED — slices=64 val=6.896% (+0.30pp miss); slices=192 epoch_time=92min (13ep infeasible); slices=128 confirmed near-optimal
- Vol-pressure loss upweighting (PR #648): closed null, no arm beat SOTA
- Abrupt VP curriculum, strong tau weights, hard tangency/normal penalties: failed
- Flow-aligned tau (PR #641): EP2 KILL
- Surface curvature features (PR #651): EP2 KILL
- Y-symmetry pair loss: failed

**Open question:** Can architectural separation (Exp 1A dual-tower) or point-sampling emphasis (Exp 1B) or coordinate-space normalization (Exp 1C) or ensemble KD (Exp 1D) reduce the 3× vol_pressure test-vs-val gap?

### 2. Positional Encoding / Geometry Representation

- **PR #691** (askeladd): RFF sigma range expansion — arm-a: 7-sigma wide {0.125,0.25,0.5,1.0,2.0,4.0,8.0}; arm-b: 6-sigma low-ext {0.125,0.25,0.5,1.0,2.0,4.0}. Running — val=8.47%, vol_pressure=5.35% (below AB-UPT reference on val!), step=23332.

**Closed (null):**
- PR #621 (nezuko): Slice-centroid STRING-RoPE — EP3 killed.
- PR #624 (alphonse): Point-level pre-slice STRING rotation — EP3 killed.
- PR #625 (askeladd): No-slice Anchor-STRING — pod hung, no results.
- PR #626 (frieren): AB-UPT geometry branch — val=9.12%, vol_pressure gap reduced 3.17×→2.07× (key finding).
- PR #647 (frieren): Anchor-string Exp 3 — EP2 KILL (16.046%), coordinate embedding bug.

**Open question:** Does widening RFF sigma coverage improve geometry representation, particularly for vol_pressure?

### 3. Optimizer / Training Dynamics

- **PR #667** (fern): Weight decay sweep {5e-4 ctrl=6.959% test=8.135% vol-ratio 2.80×, 1e-3 final=6.913% test=8.097% vol-ratio 2.85×, 1e-4 running}. **Arm B FINAL — WD scientifically null on volume val→test gap.** Test_vp flat across WD values; Arm C running.
- **PR #695** (edward): RFF num_features=32 — val=8.53%, vol_pressure=5.37% (below AB-UPT reference on val!). Critical EP3 gate (must drop below 8.0%).

**Closed (null):**
- PR #603 (tanjiro): EMA decay sweep.
- PR #614 (fern): Lion β2 sweep — β2=0.99 confirmed optimal.
- PR #640 (edward): NorMuon optimizer — diverged.
- PR #649 (edward): GradNorm min-weight floor sweep — closed null.
- PR #650 (tanjiro): LR cosine floor sweep — closed null.

### 4. Knowledge Distillation

- **PR #676** (nezuko): K=7 ensemble KD, Arm A (kd=0.7) final val=7.0153% — misses bar, budget-limited (2.37 it/s vs ~5 it/s normal → only 4 epochs). **Volume val→test ratio narrowed 3.17×→2.78%** — first lever to move vol gap. Arm B (kd=0.5) running. val=8.78%, approaching EP3 gate (8.0%).
- **PR #729** (alphonse): New Exp 1D — single-model KD from K=7 ensemble on vol_pressure only (channel-selective). Learning from PR #676 budget issue.

### 5. Architecture / Attention

- **PR #692** (tanjiro): NOW SUPERSEDED by PR #722 (tanjiro → Exp 1A). Closed null (heads=8 EP2 PASS 8.55%, final result below SOTA; heads=2 not tested). Heads=4 confirmed optimal.
- **PR #694** (thorfinn): L=6/hidden=384/heads=4 — NOW SUPERSEDED by PR #723 (thorfinn → Exp 1C). L=6/h=384 closed null.

**Established:**
- L=5 > L=4 by −1.90% relative gain; L=6 variants all infeasible or null within budget.
- Heads=4 confirmed optimal; power-of-2 constraint for SDPA.
- Slices=128 confirmed near-optimal (PR #690 closed).

### 5b. Physics-Informed Loss

**Closed (null):**
- PR #641, #648, #651 — VP upweighting, flow-aligned tau, surface curvature features all null.

---

## Active Fleet Status

All 8 tay-branch students running:

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| tanjiro | #722 | Exp 1A: Dual-tower vol/surface cross-attn (Issue #717) | Running — EP1 PASS (val=8.57%, vol_vp=6.47%, 37.6min/ep) |
| thorfinn | #723 | Exp 1C: Coordinate normalization geometry conditioning | Running — EP1 in progress |
| frieren | #728 | Exp 1B: Volume outlier-aware point sampling (Issue #717) | Just assigned — pre-EP1 |
| askeladd | #691 | RFF sigma expansion {7-wide, 6-low-ext} | Running — val=8.47%, vol_pressure=5.35%, step=23332 |
| alphonse | #729 | Exp 1D: Single-model KD from K=7 ensemble (Issue #717) | Just assigned — pre-EP1 |
| edward | #695 | rff-num-features=32 (doubled RFF encoding) | Running — val=8.53%, vol_pressure=5.37% **EP3 CRITICAL** |
| fern | #667 | Weight decay sweep {1e-4, 5e-4, 1e-3} | Running — Arm C (wd=1e-4) in progress; WD null on vol gap |
| nezuko | #676 | Ensemble distillation K=7 teacher → L=5 student | Running — val=8.78%, Arm B (kd=0.5) approaching EP3 |

**Zero idle students. Zero idle GPUs.**

**Kills executed this cycle:**
- PR #716 (frieren, BC-type): KILLED — arm-a EP2 val=26.61% >> 12%; arm-b EP1 val=27.12% + epoch_time=180min >> 80min gate
- PR #690 (alphonse, slice-count): CLOSED — slices=64 val=6.896% miss; slices=192 epoch_time=92min infeasible; slices=128 confirmed optimal

**Upcoming gate actions:**
- PR #722 (tanjiro): EP2 KILL if > 12.0%; EP3 KILL if > 8.0%
- PR #723 (thorfinn): EP1 epoch-time gate (kill if > 80min); EP2 KILL if > 12.0%; EP3 KILL if > 8.0%
- PR #728 (frieren): EP1 epoch-time gate; EP2 KILL if > 12.0%; EP3 KILL if > 8.0%
- PR #729 (alphonse): EP1 epoch-time gate; EP2 KILL if > 12.0%; EP3 KILL if > 8.0%
- PR #695 (edward): EP3 CRITICAL — val=8.53%, must drop below 8.0% at EP3 or kill
- PR #691 (askeladd): EP3 gate approach — val=8.47%
- PR #667 (fern): Arm C (wd=1e-4) running; report final result
- PR #676 (nezuko): Arm B EP3 approach — val=8.78%

---

## Potential Next Research Directions

### High Priority (Issue #717 Phase 1 — when students become idle)

1. **Exp 1E: Scheduled KD alpha (0.9→0.1)** — If PR #676 Arm B (kd=0.5) shows that KD helps vol gap but val_abupt is too high from ensemble label noise on surface/tau, try dynamically decaying kd_alpha from 0.9 (early epochs, ensemble teaches overall distribution) to 0.1 (late epochs, GT supervision dominates for refinement).

2. **Exp 1F: Batched KD with faster inference** — PR #676 showed KD at 2.37 it/s vs ~5 it/s normal (2× slowdown). If Exp 1D (#729) shows improvement, implement a batch-precomputed KD cache on GPU to halve per-step overhead and allow more epochs.

3. **Ensemble pool expansion**: When new single-model winners emerge (val_abupt < 6.5985%), run greedy ensemble (pool-25) immediately.

### Medium Priority

4. **RFF sigma learning**: If sigma expansion (PR #691) shows one direction better, consider learning the sigmas end-to-end.

5. **Depth × coord-norm cross**: If coord normalization (PR #723) improves vol_pressure, combine with L=5 + expanded RFF.

6. **Weight decay × VP upweighting**: WD direction confirmed null on vol gap (PR #667). VP upweighting (PR #648) showed VP channel 4.30% — consider coupling with Exp 1B/1D findings.

### Lower Priority (longer-horizon)

7. **Multi-scale spatial sampling**: Progressive or multi-resolution surface point sampling during training.

8. **Geometry-aware positional encoding**: Following PR #626 finding (vol_pressure gap 3.17→2.07× with AB-UPT geometry features), revisit with a cleaner implementation that avoids the coordinate embedding bug that killed PR #647.

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128 (SOTA config)
- **Positional encoding:** STRING-separable (rff_num_features=16, sigmas 0.25-4.0)
- **Optimizer:** Lion, lr=9e-5, β2=0.99 (confirmed optimal, PR #614)
- **Weight decay:** 5e-4 (under investigation in PR #667)
- **QK-norm:** enabled
- **Loss weights:** tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0
- **EMA:** 0.999 (confirmed optimal, PR #603)
- **Training budget:** ~270 min (SENPAI_TIMEOUT_MINUTES=360 with 90 min val reserve)
- **Depth scaling:** L=4→L=5 gave −1.90% relative gain; L=6/hidden=512 infeasible (~88.6 min/epoch); L=6/h=448/heads=7 infeasible (~98 min/epoch, heads=7 kills SDPA); L=6/h=384/heads=4 (PR #694) closed null.
- **Heads constraint:** heads MUST be power-of-2 for SDPA/Triton fast paths. heads=4 confirmed optimal.
- **Slice count:** slices=128 confirmed near-optimal (PR #690 closed — slices=64 misses, slices=192/256 infeasible).
