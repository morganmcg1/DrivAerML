# SENPAI Research State

- **Date:** 2026-05-05
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

---

## Latest Human Researcher Directives

No new directives from the human research team. Issue #618 (STRING/RoPE queue) all 4 experiments complete (PRs #621, #624, #625/647, #626).

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

### 1. Positional Encoding / Geometry Representation (1 PR in-flight)

- **PR #691** (askeladd): RFF sigma range expansion — arm-a: 7-sigma wide {0.125,0.25,0.5,1.0,2.0,4.0,8.0}; arm-b: 6-sigma low-ext {0.125,0.25,0.5,1.0,2.0,4.0}. Just assigned, pre-EP1.

**Closed (null):**
- PR #621 (nezuko): Slice-centroid STRING-RoPE — EP3 killed.
- PR #624 (alphonse): Point-level pre-slice STRING rotation — EP3 killed.
- PR #625 (askeladd): No-slice Anchor-STRING — pod hung, no results.
- PR #626 (frieren): AB-UPT geometry branch — val=9.12%, vol_pressure gap reduced 3.17×→2.07× (key finding).
- PR #647 (frieren): Anchor-string Exp 3 — EP2 KILL (16.046%), coordinate embedding bug.

**Open question:** Does widening RFF sigma coverage (lower and/or upper frequencies) improve geometry representation, particularly for the vol_pressure channel?

### 2. Optimizer / Training Dynamics (2 PRs in-flight)

- **PR #649** (edward): GradNorm min-weight floor sweep — running, arms B/C pending completion; EP3 PASS on best arm.
- **PR #667** (fern): Weight decay sweep {1e-4, 5e-4, 1e-3} — pre-EP1.

**Closed (null):**
- PR #603 (tanjiro): EMA decay sweep — {0.9993, 0.9997, 0.9999} all no improvement.
- PR #614 (fern): Lion β2 sweep — β2=0.99 (default) confirmed optimal; no arm beats SOTA.
- PR #640 (edward): NorMuon optimizer — diverged (val=69.15%), dead end.
- PR #650 (tanjiro): LR cosine floor sweep — closed null, no arm beat SOTA.

**Open question:** Can weight decay tuning narrow the vol_pressure val→test gap via stronger regularization?

### 3. Attention Mechanism (2 PRs in-flight)

- **PR #692** (tanjiro): Attention heads sweep — arm-a: heads=8 (dim_head=64); arm-b: heads=2 (dim_head=256). Just assigned, pre-EP1.
- **PR #665** (frieren): Cross-slice attention — adds global inter-slice MHA layer to Transolver blocks. Pre-EP1.

**Open question:** Does more diverse (heads=8) or higher-capacity (heads=2) attention improve over SOTA heads=4? Does inter-slice attention capture global geometry correlations?

### 4. Architecture Scaling (2 PRs in-flight)

- **PR #694** (thorfinn): Depth L=6 hidden=384 heads=4 (dim_head=96) — budget-safe follow-up to closed PR #693. PR #693 (L=6/h=448/heads=7) killed at EP1 because heads=7 (non-power-of-2) destroyed SDPA/Triton fast paths → 98 min/epoch, only ~2.7 epochs feasible. This PR uses heads=4 (PoW2, preserves fast paths), hidden=384 (dim_head=96, PoW2). Estimated ~57 min/epoch → ~4-5 epochs feasible. Just assigned, pre-EP1.
- **PR #690** (alphonse): Slice count sweep — arm-a: slices=64; arm-b: slices=192; arm-c: slices=256. Just assigned, pre-EP1.

**Established:**
- L=5 > L=4 by −1.90% relative gain.
- L=6 at full hidden=512 (PR #666): EP2=8.47% PASS but ~88.6 min/epoch → only 3/13 epochs feasible — computationally infeasible within budget.
- L=6/hidden=448/heads=7 (PR #693): CLOSED — 98 min/epoch (heads=7 kills SDPA fast path). Key lesson: heads must be power-of-2 for budget-feasible training.

**Open question:** Does L=6/hidden=384/heads=4 (PR #694) continue the depth scaling trend? Is slices=128 optimal or does smaller/larger over-fragment or under-communicate geometry (PR #690)?

### 5. Physics-Informed Loss / Output Formulation

**Closed (null):**
- PR #641 (thorfinn): Flow-aligned tau — EP2 KILL (14.613%).
- PR #648 (askeladd): Volume-pressure loss upweighting {2.0, 4.0, 6.0} — closed null, no arm beat SOTA.
- PR #651 (thorfinn): Surface curvature features (H, K) — EP2 KILL (14.613%).

**Key finding from PR #648:** VP channel at best arm was 4.30%, promising direction but did not translate to overall metric gain.

---

## Active Fleet Status

All 8 students running:

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| alphonse | #690 | Slice count sweep {64, 192, 256} vs SOTA 128 | Just assigned — pre-EP1 |
| askeladd | #691 | RFF sigma expansion {7-wide, 6-low-ext} | Just assigned — pre-EP1 |
| tanjiro | #692 | Heads sweep {8, 2} vs SOTA heads=4 | Just assigned — pre-EP1 |
| thorfinn | #694 | Depth L=6 hidden=384 heads=4 (budget-safe) | Just assigned — pre-EP1 |
| edward | #649 | GradNorm min-weight floor sweep | Arms B/C running, EP3 PASS on best arm |
| fern | #667 | Weight decay sweep {1e-4, 5e-4, 1e-3} | Pre-EP1 |
| frieren | #665 | Cross-slice attention over Transolver slices | Pre-EP1 |
| nezuko | #676 | (running experiment) | Watch EP gates |

**Zero idle students. Zero idle GPUs.**

**Upcoming gate actions:**
- All new PRs (#690, #691, #692, #694): EP1 informational (+ epoch-time gate for #694: kill if > 75 min/epoch), EP2 gate at step ~21,729 — KILL if > 12.0%; EP3 gate — KILL if > 8.0%
- Edward PR #649: post SENPAI-RESULT when done; run greedy ensemble (pool-25) if any arm beats 6.5985%
- Fern PR #667: EP1/EP2 gates — KILL if > 12.0% at EP2
- Frieren PR #665: EP1/EP2 gates — KILL if > 12.0% at EP2

---

## Potential Next Research Directions

### High Priority (target primary val_abupt metric directly)

1. **Ensemble pool expansion**: When new single-model winners emerge, immediately run greedy ensemble selection (pool-25) to check if any new run improves the ensemble.

2. **Depth scaling continuation**: If L=6 hidden=384 (PR #694) beats SOTA, try L=6/hidden=448/heads=4 (dim_head=112, not PoW2 — risky) or L=6/hidden=384 + wider heads=8 (dim_head=48). If PR #692 (heads sweep) finds heads ≠ 4 wins, cross with L=6.

3. **Volume-pressure gap follow-up**: PR #648 showed VP channel at 4.30% with upweighting. If weight decay (PR #667) reduces the test-vs-val gap, combine optimal wd with VP upweighting in a targeted follow-up.

### Medium Priority (architecture exploration)

4. **RFF sigma learning**: If sigma expansion (PR #691) shows one direction better, consider learning the sigmas end-to-end rather than fixing them.

5. **Boundary condition explicit encoding**: Encode inlet/outlet/wall BC type as a feature input. May help model distinguish surface types and reduce wall_shear error.

6. **Multi-scale spatial sampling**: Progressive or multi-resolution surface point sampling during training.

### Lower Priority (longer-horizon)

7. **Slice count follow-up**: If PR #690 finds slices ≠ 128 is better, run a finer sweep around the winning value.

8. **Heads × depth co-sweep**: Once optimal heads are known from PR #692, re-run depth scaling with that heads setting.

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
- **Depth scaling:** L=4→L=5 gave −1.90% relative gain; L=6/hidden=512 infeasible (~88.6 min/epoch); L=6/h=448/heads=7 infeasible (~98 min/epoch, heads=7 kills SDPA); L=6/h=384/heads=4 under investigation (PR #694, estimated ~57 min/epoch)
- **Heads constraint:** heads MUST be power-of-2 for SDPA/Triton fast paths. heads=7 causes ~50% epoch-time regression.
