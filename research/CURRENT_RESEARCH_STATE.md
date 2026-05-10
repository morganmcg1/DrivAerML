# SENPAI Research State
- **Date:** 2026-05-09 23:50 — Round 24 close-out. PR #925 alphonse rotation-aug NEGATIVE (EP3=9.1064%). Assigning alphonse milder rotation aug follow-up. 7→8 active students.
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current Baselines

| Tier | val_abupt | test_abupt | test_vol_pressure | PR | Notes |
|---|---|---|---|---|---|
| **Ensemble SOTA** | **6.0289%** | **7.3693%** | — | #880 (nezuko) | K=6 greedy pool-32 |
| **Single-model SOTA** | **6.4407%** | **7.6992%** | 11.6704% | #823 (nezuko) | surf→vol xattn, EP13, run `ghh0s4ne` |
| **Vol-pressure best anchor** | — | — | 11.374% | #681 (dc031qpt) | Issue #717 reference |

**Key win from #823:** surf→vol xattn adds one `nn.MultiheadAttention` (embed_dim=512, num_heads=4) after post-backbone LayerNorm. Q=vol_hidden, K=V=surf_hidden, zero-init out_proj. Improvement is broad-based (~−2.4% relative val, −3.6% test). The OOD test/val ratio is UNCHANGED (3.027× vs 3.025×) — xattn is a general capacity boost, NOT a targeted OOD fix. The 4 outlier test cases remain the dominant vol_pressure failure mode.

**Key finding from #767:** 4 OOD test cases (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation. Excluding them, test_vol_p = 3.9–4.2% (below AB-UPT 6.08%). Geometry conditioning of the volume decoder is the highest-priority intervention.

**Vol-pressure promotion ladder (test_vol_pressure target):**
- Weak: ≤11.0% | Solid: ≤10.0% | Major: ≤8.5% | Target: ≤6.08% (AB-UPT reference)

---

## Latest Human Researcher Directives

- **Issue #618** (STRING/RoPE): **FULLY CLOSED AXIS.** All STRING axes exhausted. σ=0.25 load-bearing. **No further STRING experiments planned.**
- **Issue #717** (vol_pressure gap): Phase 2 active — building on xattn win from #823. Post-xattn capacity axis fully closed (0-for-3: #891 FFN, #893 k-NN graph, #906 vol-self-attn). Now exploring pre-xattn capacity (#929), OOD augmentation (#925 rotation, #921 mixup, #901 mirror), vol PE (#918), smoothness regularization (#927), and TTA (#928).
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- **Issue #803** (SDF freeze): No new SDF/geometry-conditioning assignments until REQUIRED_RESTORED_CASE_IDs confirmed.
- No new directives pending.

---

## Active PRs (Round 24)

### tay-track (8 WIP, fully occupied)

| Student | PR | Hypothesis | Branch | Status |
|---|---|---|---|---|
| edward | #929 | **NEW:** Pre-xattn vol self-attention — single MHA block on vol tokens immediately BEFORE surf→vol xattn. Allows vol tokens to build spatial-neighborhood context before forming xattn queries. Symmetric counter to closed post-xattn 0-for-3 pattern. 4-ep screen with `--lr-cosine-t-max 4`. | `edward/pre-xattn-vol-self-attn` | Assigned |
| nezuko | #928 | TTA y-mirror inference ensemble on SOTA checkpoint `ghh0s4ne`. Pure inference-side intervention — negate y-coord + n_y + un-mirror tau_y, average predictions. No training required. Key signal: val→test vol_p ratio change vs 3.027× baseline. | `nezuko/tta-y-mirror-sota` | Assigned |
| alphonse | **#930** | **NEW:** Mild yaw-only rotation aug — yaw±3°, pitch=0°, p=0.3. Follow-up to #925 (yaw±5°/pitch±3°/p=0.5 failed EP3 gate at 9.11%). Yaw-only removes wall_shear vector-rotation complexity; p=0.3 reduces entropy cost. 4-ep screen. | `alphonse/mild-yaw-only-rotation-aug` | Assigning |
| askeladd | #926 | Vol geo features: centroid+bbox dist scalar features appended to vol_x. Gives vol tokens car-level geometry context (global bounding box distance). 4-ep screen. | `askeladd/vol-geo-features-centroid-bbox` | Assigned |
| frieren | #927 | Surface cp k-NN Laplacian smoothness aux loss (λ=0.05, k=8): penalise spatially abrupt cp predictions between surface neighbors. Physical constraint: cp must be continuous. | `frieren/surface-cp-laplacian-aux-loss` | Assigned |
| thorfinn | #921 | Geometric mixup: interpolate between training car geometries (weighted avg of surface + vol point clouds + targets) at train time. Arm A α_param=0.5, Arm B α_param=0.3 on 4 GPUs each. | `thorfinn/geometric-mixup-aug` | Assigned |
| tanjiro | #918 | Vol-specific RFF init sigmas — Arm C (moderate 2× lower-freq for vol: 0.1,0.25,0.5,1.0,2.0). Arm A (5× shift) killed at EP1=32.66%. Arm C EP1 PASS 28.35% — running. | `tanjiro/vol-rff-arm-c` | In progress (Arm C running) |
| fern | #901 | Train-time y-mirror aug — Arm C 4-ep screen with `--lr-cosine-t-max 4`. Arm B (13-ep) truncated at EP5 due to budget. EP1 PASS 27.43%. Running. | `fern/train-mirror-aug-4ep` | In progress (Arm C EP1 passed) |

---

## Closed Axes (Round 22–24)

### Post-xattn capacity: FULLY CLOSED (0-for-3)
- PR #891 (fern post-xattn FFN): EP3=8.50% NEGATIVE — no spatial communication
- PR #893 (askeladd vol k-NN graph attn): NEGATIVE — same post-xattn failure mode
- PR #906 (edward post-xattn vol self-attn): EP3=8.23% NEGATIVE — same post-xattn failure mode

### All prior closed axes (do not revisit)
- **Depth scaling**: L=6 CLOSED (both with-xattn #895 and without-xattn #811 NEGATIVE)
- **Width scaling**: hidden=640 CLOSED (#886 NEGATIVE)
- **Two-layer xattn**: #884 CLOSED (EP1 kill gate)
- **Detach-K/V xattn**: #890 CLOSED (K/V gradient backflow load-bearing)
- **GQA/MQA xattn**: #893 CLOSED (both n_kv_heads=1 MQA and 2 GQA fail; full MHA n_heads=4 is optimal)
- **MLP ratio=2.0**: #897 CLOSED (ratio=4.0 is optimal)
- **Learned surf pool (Perceiver)**: #894 CLOSED (destroys spatial locality)
- **Pos-encoding bias on xattn queries (static)**: #883 CLOSED (timeout, worse on all channels)
- **OOD static loss upweighting**: #888 CLOSED (gap is not data-distribution-limited)
- **Vol loss upweighting (curriculum)**: #902 CLOSED (Arm A vol_w=3.0 degraded all channels)
- **Mid-backbone surf→vol xattn (#917)**: NEGATIVE (Round 24 closeout)
- **STRING, GradNorm, EMA, LoRA, FiLM, 2-layer MLP decoder, Huber loss, Lion β2 sweep, BC-type embedding, dual-tower xattn, spectral-norm attn, stochastic depth, PCGrad, surface curvature features**: ALL CLOSED

---

## Current Research Focus

**Primary mandate (Issue #717):** Reduce test_vol_pressure below 11.0% → 10.0% → 8.5% → 6.08% (AB-UPT). All channels must not degrade.

### Theme 1: OOD Augmentation (Round 24 majority focus)
Building on the finding that the 4 OOD test cases account for 92% of squared vol_p test deviation. Round 24 tests multiple orthogonal OOD-attack angles:

- **Y-mirror aug** (#901 fern, Arm C running): 50% stochastic y-flip. EP1 PASS 27.43%. Best EP1 of any mirror-aug arm. Physical symmetry prior.
- **Rotation aug aggressive (#925, CLOSED NEGATIVE)**: yaw±5°/pitch±3°/p=0.5 — EP3=9.11%, wsh=10.14% drag. Too aggressive for 4-ep budget.
- **Mild yaw-only rotation aug** (#930 alphonse, ASSIGNING): yaw±3°/pitch=0/p=0.3. Targeted follow-up: removes wall_shear rotation complexity, lower entropy cost.
- **Geometric mixup** (#921 thorfinn, ASSIGNED): interpolate between training car geometries. Creates virtual training examples between training distribution and OOD region.
- **TTA y-mirror** (#928 nezuko, ASSIGNED): inference-only, no retraining. Upper bound on what train-time aug can achieve.

### Theme 2: Vol Input Enrichment / PE
- **Vol geo features** (#926 askeladd, ASSIGNED): centroid+bbox dist gives vol tokens global geometry context before backbone.
- **Vol-specific RFF sigmas** (#918 tanjiro, Arm C running): moderate 2× frequency shift for vol PE. EP1 PASS 28.35%.

### Theme 3: Architecture (pre-xattn)
- **Pre-xattn vol self-attn** (#929 edward, ASSIGNED): vol-vol MHA before surf→vol xattn. Symmetric counter to closed post-xattn 0-for-3 pattern. Key informative result either way.

### Theme 4: Smoothness Regularization
- **Surface cp Laplacian aux loss** (#927 frieren, ASSIGNED): k-NN smoothness penalty on predicted cp. Physical constraint.

---

## Potential Next Research Directions (post-Round 24)

### If Round 24 aug experiments win
1. **Compose aug winners**: y-mirror + rotation + geometric-mixup likely orthogonal; test 2-way and 3-way compositions.
2. **Augmented ensemble**: if TTA (#928) improves single-model, re-run greedy ensemble with TTA-augmented predictions from all pool members.
3. **Stronger augmentation (only if mild succeeds)**: extend rotation range (yaw±10°, pitch±5°); test random scale ±5% (size variation). NOTE: aggressive aug (yaw±5°/pitch±3°/p=0.5) already falsified — must nail mild version first.

### If Round 24 architecture experiments win (#929 edward pre-xattn)
4. **SDF-modulated vol PE (H2)**: after pre-xattn win, SDF-conditioned per-octave RFF scaling. ~100 params, model.py only.
5. **Learned geometry Q-bias (H3)**: mean-pooled surface global descriptor modulates vol Q before xattn. ~394K params.
6. **Pre-xattn + best-aug composition**: pre-xattn capacity + mirror aug are orthogonal; compose if both win.

### If Round 24 vol PE experiments win (#918 tanjiro Arm C)
7. **Multi-sigma vol PE**: test vol-specific sigma grid (e.g., 0.1,0.5,2.5,12.5 for longer-range vol features).

### Radical escalation (if plateau continues after Round 24)
8. **Point Transformer V3 vol head**: replace Transolver vol head with PTv3 architecture (Hilbert-curve serialized attention).
9. **GNN vol pathway**: replace Transolver vol attention with k-NN GNN for vol-vol communication (k=8-16).
10. **Manifold Mixup (H4)**: interpolate backbone hidden states (complementary to coordinate-level geometric mixup #921).

### Closed axes — do not revisit
See "Closed Axes" section above. Particularly: all post-xattn capacity, depth/width scaling, STRING, GradNorm, vol loss upweighting.

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128, surf→vol xattn (1 layer, 4 heads, ~1.05M params) → ~16.99M total
- **Depth:** L=5 optimal. FULLY CLOSED.
- **Positional encoding:** STRING-separable (rff=16, sigmas {0.25,0.5,1.0,2.0,4.0}). ALL STRING axes CLOSED.
- **Optimizer:** Lion, lr=9e-5, β₁=0.9, β₂=0.99, wd=5e-4
- **QK-norm:** enabled
- **Loss weights:** tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0. All loss-weight axes CLOSED.
- **EMA:** 0.999
- **4-ep screen schedule:** `--lr-cosine-t-max 4 --epochs 4` with vol-curriculum `0:16384:1:32768:2:49152:3:65536`
- **`--no-compile-model` required:** torch.compile + DDP NCCL deadlock at step 1
- **`find_unused_parameters=True` required for DDP when using conditional modules**

## Key Diagnostic Findings Established

- **4 OOD test cases dominate vol_pressure** (#767): 92% of squared deviation. Excluding them, test_vol_p = 3.9–4.2% (below AB-UPT). Geometry conditioning is the right lever.
- **Surf→vol xattn is broad-spectrum win, not OOD fix** (#823): OOD test/val ratio unchanged (3.027×). The 4 outlier cases still drive vol_p gap.
- **Post-xattn capacity is a dead end** (0-for-3: #891, #893, #906): all capacity added AFTER surf→vol xattn fails to improve. The bottleneck is in the xattn query quality or the K/V representation, not post-xattn processing.
- **STRING σ=0.25 is load-bearing** (#819): σ-shift/ladder failed. Encodes panel-scale surface detail critical for L5/4-ep budget.
- **K/V gradient backflow is load-bearing** (#890): detach-K/V failed. Two-layer xattn also killed by backflow. Full MHA gradient is necessary.
- **Full MHA n_heads=4 is optimal** (#893): both GQA (n_kv_heads=2) and MQA (n_kv_heads=1) fail EP3 gate.
