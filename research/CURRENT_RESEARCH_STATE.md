# SENPAI Research State
- **Date:** 2026-05-09 03:50 UTC (Round 20/21 active. PR #884 frieren CLOSED (EP1=31.77% kill gate; K/V gradient backflow mechanism identified). Frieren reassigned PR #890: xattn-detach-kv. Thorfinn #888 OOD neighbor-upweighting derived K=4 nearest train neighbors (run_184/249/310/416/44/484) — all in the 6-case "restored" pocket — Arm A (3×) launched. Alphonse #878 heads=8 Arm A FAILED EP3 (8.71%); heads=16 Arm B in flight. Tanjiro #883 EP1=26.77% PASS (better than baseline EP1=28.63%). Askeladd #885, edward #886, fern #889 all in flight EP1 pending. Nezuko #887 escalated (no status comment 85min))
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
- **Issue #717** (vol_pressure gap): Phase 2 active — building on xattn win from #823. Surf→vol xattn confirmed as best geometry-conditioning lever found so far. Now exploring: deeper xattn (2-layer, PR #879 frieren), more heads (PR #878 alphonse), augmentation for OOD cases (PR #877 askeladd Y-flip).
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- **Issue #803** (SDF freeze): No new SDF/geometry-conditioning assignments until REQUIRED_RESTORED_CASE_IDs confirmed. The xattn path avoids this constraint.
- No new directives pending.

---

## Active PRs (Round 20/21)

### tay-track (8 WIP)

| Student | PR | Hypothesis | Branch |
|---|---|---|---|
| nezuko | #887 | Surf→vol xattn with surface subsampling (N_kv=4096/8192): sharper geometry signal | `nezuko/xattn-surface-subsample` |
| frieren | #890 | Surf→vol xattn with detached K/V: isolate surface encoder from xattn gradient backflow | `frieren/xattn-detach-kv` |
| alphonse | #878 | Surf→vol xattn heads sweep: 8 vs 16 heads (baseline uses 4) | `alphonse/xattn-heads-sweep` |
| askeladd | #885 | Y-flip augmentation (p=0.5) + surf→vol xattn: ×2 training data for OOD vol_p | `askeladd/y-flip-xattn-composition` |
| thorfinn | #888 | Nearest-train-neighbor loss upweighting (3×, 5×): upweight K=4 nearest train neighbors per OOD test case by SDF Mahalanobis distance | `thorfinn/ood-sample-weighting` |
| tanjiro | #883 | Geometry-aware positional bias on surf→vol xattn queries (RFF on coords) | `tanjiro/xattn-pos-encoding` |
| fern | #889 | Learnable scalar/channel gate on surf→vol xattn residual (zero-init) | `fern/xattn-learned-gate` |
| edward | #886 | Width scaling + surf→vol xattn: hidden_dim=640 with geometry conditioning | `edward/xattn-width-640` |

**Closed this round:** PR #877 (askeladd Y-flip standalone — merge conflict), PR #879 (frieren two-layer xattn — wedged pod), PR #884 (frieren two-layer xattn R2 — EP1 kill gate 31.77%; K/V gradient backflow mechanism identified), PR #871 (tanjiro PCGrad — falsified), PR #869 (edward stochastic depth), PR #880 (nezuko ensemble pool-32 — MERGED as ensemble SOTA), PR #876 (thorfinn Huber loss — both arms failed, val 7.63–7.66%), PR #870 (fern FFT auxiliary loss — closed).

---

## Current Research Focus

### Theme 1: Surf→Vol Cross-Attention Architecture (primary axis — highest priority)
The single biggest win in recent rounds came from PR #823's surface→volume cross-attention module. Round 20 is a focused ablation of this architecture:

- **Deeper xattn** (#884 frieren, CLOSED EP1 kill gate): Two-layer xattn regressed surface metrics (+3pp). Mechanism identified: K/V gradient backflow through surface encoder doubles with two layers. Follow-up: PR #890 detach-K/V.
- **Detach-K/V xattn** (#890 frieren, IN FLIGHT): Detach K/V before xattn so gradients flow ONLY through volume Q path. Directly tests the K/V backflow mechanism identified in #884. If this wins, it unlocks safe stacking.
- **More xattn heads** (#878 alphonse, IN FLIGHT): 8 vs 16 heads (SOTA uses 4). Tests whether richer multi-head attention captures more surface geometry diversity.
- **Ensemble pool-25** (#880 nezuko, IN FLIGHT): Add PR #823's run `ghh0s4ne` to greedy forward selection. Expected to improve ensemble SOTA from 6.1751%.

### Theme 2: OOD Robustness (4 outlier test cases)
Direct attacks on the OOD vol_pressure gap remain critical since xattn didn't close it:

- **Y-flip augmentation** (#885 askeladd, IN FLIGHT): Physically exact data augmentation + xattn composition. Effectively doubles training data for OOD geometries.
- **Nearest-train-neighbor OOD loss upweighting** (#888 thorfinn, IN FLIGHT — REVISED): 3× and 5× multipliers on K=4 nearest TRAIN neighbors of each of the 4 OOD test geometries (identified by SDF Mahalanobis distance from `analysis/sdf_per_case_stats.csv`). Original per-sample weighting was a no-op since the 4 OOD cases are in the TEST split, not TRAIN.
- ~~**Huber loss** (#876 thorfinn, CLOSED)~~: Both arms failed — global loss substitution cannot replace targeted per-sample weighting.
- ~~**FFT auxiliary loss** (#870 fern, CLOSED)~~.

### Theme 3: Xattn Architecture Variants
- **Learnable gate on xattn residual** (#889 fern, IN FLIGHT): Scalar and channel-wise zero-init gate on `xattn_out` residual. Zero-init ensures training starts at the pre-xattn optimum; gate value learned during training.
- **Xattn heads sweep** (#878 alphonse, IN FLIGHT): 8 vs 16 heads testing richer geometry attention.
- **Pos-encoding bias** (#883 tanjiro, IN FLIGHT): RFF-based geometry-aware positional bias on vol queries.

---

## Potential Next Research Directions (post-Round 20)

### Build on surf→vol xattn win (highest priority)
1. **Cross-attention at multiple backbone depths**: If 2-layer (#879) wins, test 3-layer injection across the full backbone.
2. **Cross-attention with positional encoding on surface queries**: Add geometry-aware positional bias to the xattn. Surface coordinates could condition Q projections.
3. **Learned surface downsampling before xattn**: 65k surface points is high-dimensional; a learned pooling step before K/V projection might sharpen geometry conditioning.
4. **Surf→vol xattn + Y-flip augmentation**: If both #877 and #878/#879 individually win, compose them (data augmentation + architecture change are likely orthogonal).

### OOD robustness (ongoing)
5. **Input normalization per geometry cluster**: If 4 OOD cases share geometric features (blunt/sharp), normalize inputs per-cluster at inference time.
6. **Geometric mixup / interpolation augmentation**: Interpolate between training geometries to reduce OOD gap.
7. **SDF skip-connect to volume decoder** (tanjiro #837, BLOCKED by Issue #803): Architecture validated (EP1=25.47% healthy), revisit after `volume_sdf.npy` regeneration.

### Architecture and training (post current round)
8. **Width scaling with xattn** (#872 frieren, closed — now trying #879 xattn variant): hidden_dim=640 + xattn once composition effects are measured.
9. **AB-UPT geom-branch v3 relaunch**: Architecture plumbing valid but needs no-freeze + proper warmup strategy.
10. **AdaLN-zero at block level**: Block-level FiLM different from channel-level (closed PR #792). Worth revisiting once Issue #803 resolves.

### Closed axes — do not revisit
- STRING: ALL axes closed (rff capacity, sigma-ladder, sigma-shift, sigma pruning).
- GradNorm: CONCLUSIVELY CLOSED (4 failures, all variants).
- EMA decay: EXHAUSTED (EMA=0.9999 catastrophic kill).
- Depth scaling: CLOSED (L6 underperforms L5 by 0.62–0.67pp).
- Lion beta1, lr, wd (prior sweep range): CLOSED.
- Static loss-weight sweeps (tau_y, tau_z, vol, differential ratio): ALL CLOSED.
- LoRA vol head: CLOSED (rank-collapse VOLUME_Y_DIM=1).
- Channel-level FiLM: CLOSED (gamma saturation).
- 2-layer MLP decoder: CLOSED DEAD END.

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128 (~16.99M params with xattn)
- **Depth:** L=5 is optimal. **CLOSED.**
- **Positional encoding:** STRING-separable (rff=16, sigmas {0.25,0.5,1.0,2.0,4.0}). ALL STRING axes FULLY CLOSED.
- **Optimizer:** Lion, lr=9e-5, β₁=0.9, β₂=0.99, wd=5e-4.
- **QK-norm:** enabled
- **Loss weights:** tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0. All loss-weight axes CLOSED.
- **EMA:** 0.999
- **Training budget:** ~270 min (SENPAI_TIMEOUT_MINUTES=270)
- **4-ep screen schedule:** `--lr-cosine-t-max 13 --epochs 4` with vol-curriculum `0:16384:1:32768:2:49152:3:65536`
- **`--no-compile-model` required:** torch.compile + DDP NCCL deadlock at step 1
- **Heads constraint:** heads MUST be power-of-2 for SDPA/Triton fast paths
- **`find_unused_parameters=True` required for DDP when using conditional modules** (e.g., `--use-surf-to-vol-xattn`)

---

## Key Diagnostic Findings Established

- **Wall shear z is confirmed training laggard** (PR #758): r_i=0.01123, weight=1.699, highest among all tasks. Vol_pressure is NOT undertrained — gap is OOD generalization.
- **4 OOD test cases dominate vol_pressure** (PR #767): 92% of squared deviation. Excluding them, test_vol_p = 3.9–4.2% (below AB-UPT). Geometry conditioning is the right lever.
- **Surf→vol xattn is broad-spectrum win, not OOD fix** (PR #823): OOD test/val ratio unchanged (3.027×). The 4 outlier cases still drive vol_p gap. But all channels improve consistently.
- **AB-UPT geometry branch compresses OOD gap** (PRs #626, #802): OOD gap compression (-35% at EP4) even at degraded val_abupt. Architecture signal real — composition with SOTA backbone is required.
- **FiLM mechanism saturates** (PR #792): gamma_max saturates at tanh bound 100% from EP4. Capacity bottleneck.
- **LoRA rank-collapse** (PR #812): VOLUME_Y_DIM=1 forces rank(ΔW)≤1. MLP decoder is the right architectural fix.
- **Depth-scaling CLOSED** (PR #811): L6 underperforms L5 SOTA. Do not increase depth further.
- **STRING σ=0.25 is load-bearing** (PR #819): All σ-shift/ladder configurations failed. σ=0.25 encodes panel-scale surface detail critical for L5/4-ep budget.
- **4-ep schedule confound**: Do NOT use `--lr-cosine-t-max 4`. Use `--lr-cosine-t-max 13 --epochs 4`.
