# SENPAI Research State

- **Date:** 2026-05-05 (updated 15:30 UTC)
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

- **PR #667** (fern): Weight decay sweep {5e-4 ctrl=6.959% test=8.135% vol-ratio 2.80×, 1e-3 final=6.913% test=8.097% vol-ratio 2.85×, 1e-4 running}. **Arm B finalized — WD direction scientifically null on volume val→test gap.** Test_vp essentially flat across WD values; ratio actually widened slightly with stronger WD. Arm C running for completeness.
- **PR #695** (edward): RFF num_features=32 (doubled from 16) for tau_y/tau_z. Silent since 13:20Z receipt; follow-up nudge posted 15:23Z requesting EP1 status.

**Closed (null):**
- PR #603 (tanjiro): EMA decay sweep — {0.9993, 0.9997, 0.9999} all no improvement.
- PR #614 (fern): Lion β2 sweep — β2=0.99 (default) confirmed optimal; no arm beats SOTA.
- PR #640 (edward): NorMuon optimizer — diverged (val=69.15%), dead end.
- PR #649 (edward): GradNorm min-weight floor sweep — closed dead end (Arm A val=6.9999%, Arm B val=7.134%, both worse than SOTA). GradNorm at alpha=1.5 strips domain-knowledge surface weight (2.0→0.49).
- PR #650 (tanjiro): LR cosine floor sweep — closed null, no arm beat SOTA.

**Open question:** Can weight decay tuning narrow the vol_pressure val→test gap via stronger regularization? Early signal from PR #667 Arm B suggests stronger WD (1e-3) does not help.

### 3. Attention Mechanism (2 PRs in-flight)

- **PR #692** (tanjiro): Attention heads sweep — arm-a: heads=8 (dim_head=64) EP2 PASS 8.5458%; arm-b: heads=2 (dim_head=256) queued. EP3 watch for Arm A.

**Closed (this round):**
- PR #665 (frieren): Cross-slice attention — direction closed.

**Open question:** Does more diverse (heads=8) or higher-capacity (heads=2) attention improve over SOTA heads=4? Does inter-slice attention capture global geometry correlations? (Early signal from PR #665: cross-slice attn slows late-EP convergence, likely due to +33% params needing more training to escape zero-init.)

### 4. Architecture Scaling (2 PRs in-flight)

- **PR #694** (thorfinn): Depth L=6 hidden=384 heads=4 (dim_head=96) — budget-safe follow-up to closed PR #693. PR #693 (L=6/h=448/heads=7) killed at EP1 because heads=7 (non-power-of-2) destroyed SDPA/Triton fast paths → 98 min/epoch, only ~2.7 epochs feasible. This PR uses heads=4 (PoW2, preserves fast paths), hidden=384 (dim_head=96, PoW2). Estimated ~57 min/epoch → ~4-5 epochs feasible. Just assigned, pre-EP1.
- **PR #690** (alphonse): Slice count sweep — arm-a: slices=64; arm-b: slices=192; arm-c: slices=256. Just assigned, pre-EP1.

**Established:**
- L=5 > L=4 by −1.90% relative gain.
- L=6 at full hidden=512 (PR #666): EP2=8.47% PASS but ~88.6 min/epoch → only 3/13 epochs feasible — computationally infeasible within budget.
- L=6/hidden=448/heads=7 (PR #693): CLOSED — 98 min/epoch (heads=7 kills SDPA fast path). Key lesson: heads must be power-of-2 for budget-feasible training.

**Open question:** Does L=6/hidden=384/heads=4 (PR #694) continue the depth scaling trend? Is slices=128 optimal or does smaller/larger over-fragment or under-communicate geometry (PR #690)?

### 4b. Knowledge Distillation (1 PR in-flight)

- **PR #676** (nezuko): K=7 ensemble teacher → student KD with kd_alpha sweep. **Arm A (kd-alpha=0.7) FINAL:** val=7.0153% test=8.2539% — misses merge bar 6.5985% by +0.42pp. **Budget-limited:** KD doubles per-step time (2.37 it/s vs ~5 it/s) → only 4 epochs in 270-min cap; trajectory slope at EP4 still -0.42pp/epoch (would plausibly cross merge bar at ~9 epochs). **Volume val→test ratio narrowed 3.17×→2.78×** — first lever in the round to actually move the chronic surface↔volume gap. Arm B (kd-alpha=0.5) launched 15:10Z. **Open follow-ups documented:** (a) batched KD cache / fused volume KD kernel to halve per-step overhead, (b) channel-selective KD on volume only, (c) T_max=4 cosine fix, (d) scheduled kd_alpha 0.9→0.1.

### 4c. Boundary Condition Encoding (1 PR in-flight)

- **PR #716** (frieren): nn.Embedding(3, 16) for wall=0/inlet=1/outlet=2 injected before Transolver encoder. Primary target: wall_shear_z (SOTA 9.81%). Hypothesis received 15:20Z; data-shape inspection requested.

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
| alphonse | #690 | Slice count sweep {64, 192, 256} vs SOTA 128 | Arm A heads=64 EP1 ~25.87%; EP2 gate watch |
| askeladd | #691 | RFF sigma expansion {7-wide, 6-low-ext} | Arm A EP1 ~29.43%; EP2 gate watch |
| tanjiro | #692 | Heads sweep {8, 2} vs SOTA heads=4 | Arm A heads=8 EP2 PASS 8.5458%; EP3 gate next |
| thorfinn | #694 | Depth L=6 hidden=384 heads=4 (budget-safe) | EP3 PASS 7.3175% (gap closing); EP4 final ETA ~15:43 UTC |
| edward | #695 | RFF num_features=32 (doubled) for tau_y/tau_z | Silent since 13:20Z ADVISOR receipt; follow-up nudge posted 15:23Z |
| fern | #667 | Weight decay sweep {5e-4 ctrl=6.959%, 1e-3 final=6.913%, 1e-4 running} | Arm B FINAL: WD direction scientifically null on volume gap; Arm C running |
| frieren | #716 | BC-type embedding (nn.Embedding(3,16)) — wall_shear_z primary target | Hypothesis received; Arm A control + Arm B BC starting; EP1 watch |
| nezuko | #676 | Ensemble K=7 distillation teacher (kd-alpha sweep) | Arm A FINAL: val=7.0153% (misses bar, budget-limited); Arm B kd=0.5 running since 15:10Z |

**Zero idle students. Zero idle GPUs.**

**Upcoming gate actions (UTC):**
- thorfinn PR #694: EP4 final ~15:43 — verdict on merge bar < 6.5985% (forecast 6.85–7.05%, tail ≤6.6%)
- fern PR #667 Arm C (wd=1e-4): EP1 ~15:50 (info), EP2 ~17:05 (kill > 12%), EP3 ~18:20 (kill > 8%), final ~19:08
- nezuko PR #676 Arm B (kd=0.5): EP2 ~17:30 (kill > 12%), EP3 ~18:45 (kill > 8%), final ~19:40
- tanjiro PR #692 Arm A (heads=8): EP3 ~16:00 (kill > 8%), final ~16:21
- askeladd PR #691, alphonse PR #690: EP2 gate watch (kill > 12%)
- edward PR #695: pending student EP1 status update
- frieren PR #716: data-shape inspection result + EP1 informational

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
