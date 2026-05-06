# SENPAI Research State

- **Date:** 2026-05-06 13:30 UTC (Round 13 mid-flight — 8 PRs WIP, 0 review-ready, 0 idle; fern #765 closed → #769 assigned)
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

---

## Latest Human Researcher Directives

**Issue #717 (DrivAerML Single-Model Volume Pressure Plan, 2026-05-05).** Active. The chronic ~3x volume_pressure test-vs-val gap (val~3.6%, test~11.5%) is the primary systematic problem. Ensembles banned for the volume push; only single-model artifacts at inference. Phase 1 = short probes (3-6h) of dual-tower / outlier-sampling / geometry-conditioning / single-model-KD; Phase 2 = combinations of mechanisms that moved single-model test volume; Phase 3 = long verification. All PRs in this wave must use the 9-column reporting table and compare against the three frozen anchors `sogus8sx` (#599), `4k25s25e` (#592), `dc031qpt` (#681).

---

## Current Baselines

| Tier | val_abupt | test_abupt | PR | Notes |
|---|---|---|---|---|
| Ensemble SOTA | **6.1751%** | 7.5347% | #612 (nezuko) | K=7 greedy pool-24; volume_p test still 11.47% |
| Single-model SOTA | **6.5985%** | 7.9915% | #592 (alphonse) | depth-L5, run `4k25s25e`, EP4 |
| Single-model gate | < 6.5985% | — | — | Must beat to update single-model SOTA |
| Issue #717 vol-test ladder | weak < 11.0%, solid <= 10.0%, major <= 8.5%, target <= 6.08% | — | — | Promotion thresholds for new vol mechanisms |

**Issue #717 frozen anchors** (every new vol-push PR reports against these):

| Run | PR | Test volume_p | Test wall_shear | Test tau_y | Test tau_z |
|---|---:|---:|---:|---:|---:|
| `sogus8sx` | #599 | 11.694 | 7.299 | 7.941 | 9.535 |
| `4k25s25e` | #592 | 11.933 | 7.334 | 8.145 | 9.298 |
| `dc031qpt` | #681 | 11.374 | 8.321 | 9.596 | 10.738 |

---

## Current Research Themes

### 1. Volume_pressure test-transfer (Issue #717) — primary focus

**In flight (8 PRs):**
- **PR #752 (askeladd)** — Exp 1C P4: x-slab wake stratified vol sampling. Arm A (far-wake) closed neg test_vol_p=12.49%; Arm B (near-wake [1m,4m), factor=3.0) launched 07:45 UTC, run `jc2t6sxa`, EP1 due ~09:01 UTC.
- **PR #758 (tanjiro)** — GradNorm ema_proxy α=3.0/2.0 sweep + min_weight=0.7. Arm A COMPLETE val_abupt=7.0798%, test_vol_p=12.38% (worse than SOTA). Arm B (`1bmbfu30`, α=2.0, floor=0.7) launched.
- **PR #760 (alphonse)** — CLOSED NEGATIVE. vol_w=2.0 regression across all metrics (val_abupt=6.9810%). Vol-loss reweighting ruled out as a lever.
- **PR #761 (frieren)** — Dedicated 2-layer volume head on shared encoder (capacity-additive). EP2 val_abupt=8.088%; EP3 in flight.
- **PR #762 (edward)** — Surface curvature (H, K) from local PCA propagated to volume points. EP2 val_abupt=8.58%, mid-EP3.
- **PR #763 (nezuko)** — Upstream-region supervised attention (x_rel ≤ 0.5, w_upstream ∈ {1.5, 2.0, 3.0} sweep). ~EP2 boundary.
- **PR #764 (thorfinn)** — STRING spectral budget expansion (sigmas to 8-octave; rff-num-features 16→24). EP2 val_abupt=8.477% passes gate.
- **PR #765 (fern) CLOSED NEGATIVE** — No-slice Anchor-STRING transformer (Issue #618 Exp 3). Two runs failed (`klw97qgk` EP1=54.89% per-layer-resample bug; `muthl81j` EP1=45.09% post-fix, kill-gate exceeded). Root causes: K=1024 random anchors too sparse (0.78% of points, bimodal training loss), 0.1× LR rail froze RoPE (`log_freq` norm 7.996→7.996). Fern reassigned to #769 below.
- **PR #769 (fern) NEW** — Slice-centroid Learnable-STRING-RoPE on Transolver Q/K (Issue #618 Exp 1). Reuses `LearnableCoordinateRoPE` from #765 but rotates slice-token Q/K with `slice_weights @ point_coords` centroids instead of random anchors. 4 arms (control / B after-QK-norm 0.5×LR / C before-QK-norm 0.5×LR / D no-diff-LR), `--wandb_group fern-slice-string-rope-v1`. Decision rule: merge if any arm beats single-model SOTA val_abupt < 6.5985%.
- **PR #766 (alphonse) NEW** — Offline k-NN vol_pressure gradient-consistency aux loss. Revives PR #757 (edward, killed ~2.9× epoch overhead) with offline precomputed k-NN graph to achieve ~1.1–1.3× overhead. Physics-motivated: supervises ∇p field consistency, not just point-wise MSE. Applies only at epoch ≥ 9 (V=65k) to avoid k-NN remapping. λ sweep 0.05 (Arm A) / 0.01 (Arm B, if needed).

**KEY DIAGNOSTIC FROM PR #737 (nezuko, just closed 2026-05-01):**

The chronic vol_p test-vs-val gap is empirically owned by the **upstream region** (x_rel ≤ 0.5). Per-region split (Arm B):
- upstream: 92.43% pts, val=4.32%, test=11.93%, val→test=2.76× — owns ~4× more L2 mass than near-wake
- near: 7.34% pts, val=13.81%, test=21.58%, val→test=1.56× (mechanism works locally but can't move aggregate)
- far: 0.23% pts, val=70%, test=79%, val→test=1.12×

Static near-wake upweighting cannot move the aggregate; counter-intuitively Arm B's marginal test_vol_p improvement (-0.27pp vs Arm A) came from improved upstream val→test transfer (2.76× vs 2.93×), not from near-wake gain. Arm B's near-wake test was actually worse than Arm A's (21.58% vs 20.88%).

**Consequence**: Three converging interventions now attack the upstream-dominant vol_p:
1. PR #763 (nezuko): direct upstream-region per-point loss reweighting
2. PR #760 (alphonse): whole-volume loss reweighting (uniform amplification, less targeted)
3. PR #761 (frieren): dedicated volume head (capacity-additive — matches #755 finding that volume branch is capacity-limited, not feature-overfit)

**KEY DIAGNOSTIC FROM PR #755 (frieren):**
Regularization (stochastic depth + volume-token dropout) made val_volume_pressure WORSE (12.48% → 14.14%) while every non-volume metric improved. This FALSIFIES the "memorize-and-fail-OOD" framing of the val/test gap. The volume branch is **capacity-limited**, not feature-overfit. Updated diagnosis: test cases differ in operating regime (wake intensity, geometry envelope) — not in memorized feature compositions. Consequence: 
- Abandon regularization-based OOD levers for the volume branch.
- Push toward capacity-additive (PR #761 vol head, item 6) and geometry-conditioning (PR #762 curvature, item 4) approaches.

### 2. Closed dead ends from prior round (Issue #717-aligned closures)

- **PR #722** dual-tower volume/surface cross-attention — null (+0.87pp val regression).
- **PR #716** BC-type embedding (frieren) — operationally broken (concurrent 8-GPU jobs doubled epoch time; lesson learned).
- **PR #695** rff-num-features=32 — null (+0.33pp val regression).
- **PR #694** depth L=6/hidden=384/heads=4 — null (val=6.9016%, +0.30pp), still descending but budget-bound.
- **PR #693** L=6/h=448/heads=7 — killed (heads=7 destroys SDPA fast path, ~98 min/epoch).
- **PR #692** heads sweep {8, 2} — heads=8 null (+0.83pp); heads=2 unauthorized launch, closed.
- **PR #691** RFF sigma {wide, low-ext} — both arms null.
- **PR #690** slice sweep {64, 192, 256} — slices=64 null (+0.30pp); slices=192/256 infeasible (>92 min/epoch).
- **PR #667** weight-decay sweep {1e-4, 5e-4, 1e-3} — all 3 null on test, weakest WD wins on val only.

### 3. Established architectural decisions (do NOT re-litigate)

- Transolver **L=5, hidden=512, heads=4, slices=128** (SOTA config).
- STRING-separable PE, RFF=16, sigmas {0.25, 0.5, 1.0, 2.0, 4.0}.
- Lion lr=9e-5, β2=0.99, weight-decay=5e-4.
- EMA decay=0.999. Loss weights: tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0.
- Heads MUST be power-of-2 for SDPA fast-path (heads=7 kills throughput).
- **NEVER run two `torchrun --nproc_per_node=8` jobs concurrently on the same 8 GPUs** (PR #716 lesson — doubled epoch time to 180 min, time-gate kill).

---

## Active Fleet Status (2026-05-06 11:55 UTC — Round 13 in flight)

All 8 students assigned (alphonse reassigned from #760→#766 this cycle):

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| alphonse | **#760** | vol-loss-weight ablation vol_w=2.0/3.0 | WIP, Arm A `1gv5s938` EP4 landed: val_abupt=**6.98%**, vol_p=4.10% (still below SOTA 6.5985%). Run still going at step 36799/270min. Arm B halted; awaiting student final + test report. |
| askeladd | **#752** | x-slab wake stratified vol sampling (Issue #717) | WIP, Arm B `jc2t6sxa` cleared EP3 gate: val_abupt=**7.44%**, vol_p=4.86% (passes <8% gate). Run at step 33001. Arm A closed neg test_vol_p=12.49%. |
| edward | **#762** | Surface curvature (H, K) propagated to volume points | WIP, EP3 territory (step 21786). EP1 val_abupt=30.32% (anomalous), EP2=**8.58%, vol_p=5.20%** (passes EP2 gate cleanly). Student silent 2.5+ hr; advisor nudge posted demanding EP1/EP2 retrospective + EP3 watch. |
| fern | **#769** | Slice-centroid Learnable-STRING-RoPE on Transolver Q/K (Issue #618 Exp 1) | WIP, fresh assignment 13:30 UTC. Supersedes closed #765 (Exp 3 anchor-STRING; both runs failed kill gate). Reuses validated `LearnableCoordinateRoPE` module; rotates slice Q/K with differentiable `slice_weights @ point_coords` centroids; 0.5× LR rail (vs 0.1× that froze #765); 4 arms incl. control / after-QKnorm / before-QKnorm / no-diff-LR. |
| frieren | **#761** | Dedicated 2-layer volume head (capacity-additive) | WIP, mid-EP3 (step 25743). EP2 val_abupt=8.088% (+0.15pp vs SOTA EP2 7.940%); EP3 gate <8% is tight but plausible. |
| nezuko | **#763** | Upstream-region supervised attention (w_upstream=1.5) | WIP, ~EP2 boundary (step 21591/21729). EP1 val_abupt=27.57%; per-region: upstream 92.42% pts at 15.59% rel_l2, near 7.34% at 39.84%, far 0.23% at 203.41%. |
| tanjiro | **#758** | GradNorm ema_proxy α=3.0/2.0 sweep | WIP, Arm A COMPLETE val_abupt=7.0798%, **test_vol_p=12.38% (worse than SOTA 11.93%)** — hypothesis NOT confirmed (tau_z is laggard, not vol_p). Arm B (`1bmbfu30`, α=2.0, floor=0.7) launched 11:11 UTC, mid-EP1 (step 1210). |
| thorfinn | **#764** | STRING spectral budget expansion (8-octave, 24 RFF features) — Issue #618 | WIP, just past EP2 gate (step 21970), val_abupt=**8.477% (passes <9% gate)**. EP1=27.564% beat 7-sigma SOTA-stack. Awaiting student EP2 comment. |

**Zero idle students. Zero idle GPUs.**

**Open advisor actions (this cycle):**
- alphonse #766: New assignment — offline k-NN grad-consistency. Student must implement precompute_vol_knn.py + new CLI flags.
- edward #762: Advisor nudge posted at 11:42 UTC demanding retrospective EP1/EP2 reports + EP3 status.
- fern #769: New assignment — slice-centroid STRING-RoPE (Issue #618 Exp 1). Supersedes closed #765.
- All other PRs: gate-watching at EP2/EP3 thresholds; passive.

**Issue #717 status:** No arm has yet beaten the weak win gate `test_vol_p < 11.374%`. Tanjiro Arm A (12.38%), askeladd Arm A (12.49%), and alphonse #760 (12.20%) all regress. The remaining six in-flight arms plus new #766 are the live attempt set.

**Upcoming gate actions for active runs:**
- EP1 time-gate: kill if epoch_time > 80 min (4800s).
- EP2 (step ~21,729): kill if val_abupt > 12.0%.
- EP3 (step ~32,594): kill if val_abupt > 8.0%.
- Final: 9-column Issue #717 table + per-region test volume breakdown (upstream/near/far) + val→test ratio statement.

---

## Potential Next Research Directions (post-current-round)

### Phase 2 (combinations) — only after Phase 1 produces test-volume movers

1. **Best-volume mechanism on #599 base** (Issue #717 Exp 2A). Combine the single most-effective Phase 1 lever with the wall-shear-clean #599 anchor.
2. **Best-volume + mild tau stabilization** (Exp 2B). Only if a tau-stabilizing mechanism (#669 family) clears clean test evidence.
3. **L5/RFF capacity variant** (Exp 2C). Re-stack mild RFF capacity on L=5 if Phase 1 finds the volume mechanism is capacity-bound.

### Mechanism follow-ups to consider if Phase 1 partially succeeds

4. **Curvature features** (Issue #717 1C P4) on top of distance-to-surface. Soft H/K from local point-cloud PCA. Lower priority than P3.
5. **Spectral pre-bias**: If geometry-conditioning helps but is noisy, learn the sigma kernel widths end-to-end (PR #691 RFF sigma proved fixed-set is brittle but learning could change that).
6. **Depth-wise volume head**: a dedicated 2-layer Transolver head exclusively for volume queries, branching off the shared encoder (different from the failed dual-tower #722 by sharing encoder, splitting only at the head).
7. **Long-budget verification (Phase 3)**: Two seeds, 5+ EMA epochs, multi-checkpoint reporting, only for mechanisms that beat 11.0% test volume in short probes.

### Plateau protocol candidates if 5 in-flight + 5 new all null

8. **Switch to a Markov / learned sampler over volume points** (RL or top-K residual-driven): more aggressive than #728's EMA reweighting.
9. **Volume-only flow-matching / diffusion-style refinement head** on top of the deterministic prediction.
10. **Multi-resolution volume representation**: voxel-grid + point hybrid with octree-style attention.

---

## Key Operational Notes

- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Training budget:** SENPAI_TIMEOUT_MINUTES=360 (with ~90 min val reserve = ~270 min training).
- **Reproduce SOTA stack:** see `BASELINE.md` PR #592 block (Lion 9e-5, L=5, hidden=512, heads=4, slices=128, RFF=16, sigmas 0.25-4.0, vol-points-curriculum 16k->65k, EMA=0.999, grad-clip=0.5, lr-warmup-1ep, cosine T_max=13).
- **Required reporting on every Issue #717 PR:** 9-column table, three anchors (#599/#592/#681), per-case top-10 worst test volume, per-region test volume breakdown, val→test transfer ratio statement.
