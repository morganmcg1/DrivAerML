# SENPAI Research State
- **Date:** 2026-05-09 (PR #902 nezuko vol-loss-upweighting CLOSED NEGATIVE — Arm A vol_w=3.0 degraded all channels vs baseline; Arm B vol_w=5.0 cancelled. Assigned PR #917 nezuko: mid-backbone surf→vol xattn injection (after L=3). PR #910 frieren Arm A α=0.5 EP3 FAIL at 8.65% — Arm B α=0.75 launched.)
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

## Active PRs (Round 22)

### tay-track (8 WIP)

| Student | PR | Hypothesis | Branch | Status |
|---|---|---|---|---|
| nezuko | #917 | **NEW:** Mid-backbone surf→vol xattn injection: second xattn sublayer inserted after backbone L=3 (mid-point), so geometry conditioning compounds through L4+L5. Arm A standalone; Arm B composed with post-backbone xattn. | `nezuko/mid-backbone-xattn-injection` | Assigned |
| alphonse | #915 | Surface jitter augmentation: Gaussian noise on surface xyz at train time (Arm A σ=0.001m, Arm B σ=0.003m). OOD geometry robustness via input-noise regularization. | `alphonse/surface-jitter-aug` | Assigned |
| fern | #901 | Train-time y-axis mirror augmentation: 50% random lateral flip. EP1 PASS: 29.56%. | `fern/train-mirror-aug-y` | In progress |
| frieren | #910 | Xattn K/V grad scale sweep α=0.5 (Arm A FAIL EP3=8.65%) and α=0.75 (Arm B, now running). Follow-up from #896 (α=0.25 stalled EP3=8.95%). | `frieren/xattn-kv-grad-scale-sweep-alpha` | Arm B running |
| edward | #906 | Vol→vol self-attention block post surf→vol xattn: Transformer decoder pattern gives vol tokens spatial communication after surface conditioning. | `edward/vol-self-attn-post-xattn` | Assigned |
| askeladd | #916 | Vol k-NN graph attention post-xattn: each vol token attends to k=8 spatial nearest neighbors. Locality-preserving vol-vol communication (k-NN vs O(N²) full self-attn). Memory-efficient chunked kNN implementation. | `askeladd/vol-knn-graph-attn` | Assigned |
| tanjiro | #908 | 1D spectral mixing (FNO-style) on vol tokens post-xattn: learned complex mixing in Fourier domain over N_vol tokens for global vol_p structure. | `tanjiro/vol-spectral-mixing-1d-fno` | Assigned |
| thorfinn | #909 | Vol-head SWA (stochastic weight averaging on vol_head + xattn only): smoother vol sub-module for OOD generalization. | `thorfinn/vol-head-swa` | Assigned (STALE — check pod) |

**Closed this round (Round 22):** PR #902 (nezuko vol-loss-upweighting — NEGATIVE: Arm A vol_w=3.0 degraded all channels vs baseline at EP3=8.54%; Arm B vol_w=5.0 cancelled. Vol loss upweighting axis CLOSED for this mechanism). PR #893 (alphonse GQA xattn — NEGATIVE: both n_kv_heads=1 MQA EP3=8.27% and n_kv_heads=2 GQA EP3=8.21% fail gate; full MHA capacity load-bearing; GQA axis CLOSED). PR #907 (askeladd vol-approx-sdf — NEGATIVE: EP1 32.20% > 30% kill gate; kNN SDF feature failed EP1 screen; on-the-fly approx-SDF axis CLOSED for now).

**Closed in Round 21/22 transition:** PR #894 (askeladd learned surf pool — NEGATIVE: EP4=7.73%, Perceiver pooling destroys spatial locality), PR #883 (tanjiro xattn pos-encoding — NEGATIVE: 3/13 ep at 7.4879%, timeout, worse on all channels than SOTA), PR #897 (thorfinn mlp-ratio=2.0 — NEGATIVE: EP3=11.18%, halved FFN capacity kills convergence), **PR #895 (edward L=6+xattn — NEGATIVE: EP4=7.886%, deep train/val gap, memorization > generalization; depth axis FULLY CLOSED both with and without xattn)**, PR #891 (fern post-xattn FFN — NEGATIVE: EP3=8.50%, FFN-only vol processing lacks spatial communication), **PR #896 (frieren K/V grad scale α=0.25 — NEGATIVE: EP3=8.954%, slope stalled EP2→EP3; α=0.25 too aggressive; follow-up α=0.5/α=0.75 in PR #910)**.

**Closed in Round 20/21:** PR #878 (alphonse heads-sweep — NEGATIVE), PR #887 (nezuko surface-subsample — negative), PR #877 (askeladd Y-flip standalone — merge conflict), PR #879 (frieren two-layer xattn — wedged pod), PR #884 (frieren two-layer xattn R2 — EP1 kill gate), PR #871 (tanjiro PCGrad — falsified), PR #869 (edward stochastic depth), PR #880 (nezuko ensemble pool-32 — MERGED as ensemble SOTA), PR #876 (thorfinn Huber loss — both arms failed), PR #870 (fern FFT auxiliary loss — closed), PR #888 (thorfinn OOD upweighting ×3 — NEGATIVE), PR #890 (frieren detach-K/V xattn — NEGATIVE: EP1 kill gate, K/V gradient backflow load-bearing), PR #886 (edward xattn-width-640 — NEGATIVE).

---

## Current Research Focus

**Primary mandate (Issue #717):** Reduce test_vol_pressure below 11.0% (weak), 10.0% (solid), 8.5% (major), targeting ≤6.08% (AB-UPT). All other metrics must not degrade. Surface/wall-shear are secondary.

### Theme 1: Post-xattn Volume Pathway Enrichment (Round 22 focus)
Building on the PR #823 xattn win, the vol tokens after xattn only receive surface conditioning but cannot communicate with each other. Round 22 tests radical vol-pathway enrichments:

- **Vol→vol self-attention post-xattn** (#906 edward, ASSIGNED): Full Transformer decoder block (vol self-attention + FFN) after surf→vol xattn. Adds vol-spatial communication that closed post-xattn FFN (#891) lacked. Zero-init out_proj for identity-at-init.
- **Vol k-NN graph attention post-xattn** (#916 askeladd, ASSIGNED): k=8 nearest-neighbor graph attention on vol tokens after surf→vol xattn. Gives vol tokens locality-preserving spatial communication: each vol token attends to 8 spatial neighbors instead of all N (O(N·k) vs O(N²)). Chunked kNN distance computation avoids OOM. Zero-init out_proj for identity-at-init. Physics motivation: CFD pressure coupling is local (BL gradients, wake interactions). The 4 OOD outlier cases have anomalous near-wake geometry.
- **1D spectral mixing (FNO-style)** (#908 tanjiro, ASSIGNED): Learned complex linear mixing in Fourier domain along N_vol token dimension, post-xattn. Captures global spectral structure of volume pressure (wake, stagnation zones). FNO-motivated; zero-init for identity-at-init.
- **Vol-head SWA** (#909 thorfinn, ASSIGNED): Stochastic weight averaging on vol_head + xattn submodule only (starting from EP3/EP9). Smoother vol head for OOD generalization without disrupting backbone. Applied to live weights (not EMA).

### Theme 2: Xattn Architecture Variants (in-flight)
- **Mid-backbone xattn injection** (#917 nezuko, ASSIGNED): Insert surf→vol xattn after backbone L=3 (mid-point). Arm A standalone (mid-backbone only); Arm B composed with post-backbone xattn (#823). Geometry conditioning compounds through L4+L5. Requires code change to split backbone execution.
- **K/V gradient scaling sweep α=0.75** (#910 frieren, ARM B RUNNING): Arm A (α=0.5) EP3 FAIL at 8.65% — steep slope (−2.89pp EP2→EP3, 6.5× better than α=0.25) but still missed gate. Arm B (α=0.75) now running. Bracket [α=0.25 EP3=8.95%, α=0.5 EP3=8.65%, α=0.75 TBD, α=1.0 SOTA EP3=7.12%]. If α=0.75 also fails, K/V grad-scale axis is likely closed — any dampening degrades abupt-mean even when vol_p benefits.
- **GQA xattn** (#893 alphonse, CLOSED NEGATIVE): Both n_kv_heads=1 (MQA) and n_kv_heads=2 (GQA) fail EP3 gate. Full MHA capacity is load-bearing. Axis closed.

### Theme 3: OOD Augmentation (in-flight)
- **Train-time y-axis mirror aug** (#901 fern, IN FLIGHT — EP1 PASS 29.56%): 50% stochastic y-flip. Directly attacks the ~3× val/test volume pressure gap via learned symmetry prior.
- **Surface jitter aug** (#915 alphonse, ASSIGNED): Gaussian noise on surface xyz at train time (σ=0.001m Arm A, σ=0.003m Arm B). Forces surface encoder OOD robustness via input-noise regularization (Bishop 1995). Orthogonal to fern's symmetry prior.

### Closed Axes (do not revisit)
- **Depth scaling**: L=6 CLOSED (both with-xattn #895 NEGATIVE and without-xattn #811 NEGATIVE)
- **Width scaling**: hidden=640 CLOSED (#886 NEGATIVE)
- **Post-xattn FFN only**: #891 CLOSED (lacks vol-vol spatial communication, too few params)
- **Learned surf pooling**: #894 CLOSED (Perceiver 65k→256 destroys spatial locality)
- **Pos-encoding bias on xattn queries**: #883 CLOSED (timeout at EP3=7.49%, worse than SOTA on all channels)
- **MLP ratio sweep**: #897 CLOSED (ratio=2.0 EP3 FAIL, ratio=4.0 is optimal)
- **Detach-K/V xattn**: #890 CLOSED (K/V gradient backflow is load-bearing)
- **Two-layer xattn**: #884 CLOSED (EP1 kill gate)
- **GQA/MQA xattn**: #893 CLOSED (both n_kv_heads=1 MQA EP3=8.27% and n_kv_heads=2 GQA EP3=8.21% fail gate; full MHA n_heads=4 is optimal)
- **OOD static loss upweighting**: #888 CLOSED (gap is not data-distribution-limited)
- **STRING, GradNorm, EMA, LoRA, FiLM, 2-layer MLP decoder, Huber loss**: ALL CLOSED

---

## Potential Next Research Directions (post-Round 22)

### If Round 22 vol-pathway enrichment experiments win
1. **Compose winners**: Vol-self-attn + approx-SDF + spectral-mixing are likely orthogonal; if any 2 win individually, test composition.
2. **Deeper spectral mixing**: If 1D FNO wins, test multi-layer spectral mixing (2-3 spectral blocks) on the vol pathway.
3. **Learned SDF vs approx SDF**: If approx SDF feature wins, replace nearest-neighbor approximation with a learned SDF encoder (small MLP on surface points → vol SDF).

### If Round 22 xattn architecture experiments win (frieren #910, nezuko #917)
4. **K/V grad scale + mid-backbone xattn composition**: If both #910 (α=0.75 scale) and #917 (mid-backbone injection) win independently, compose them.
5. **Layer-split optimization for mid-backbone**: If #917 wins, test after L=2 vs L=3 injection point (current hypothesis: after L=2, 0-indexed = 3rd block). Also test 3 injection points (after L=1, L=3, post-backbone).
6. **Surface jitter + vol-self-attn composition**: If alphonse #915 surface jitter wins and edward #906 vol-self-attn wins, combine (data augmentation + architecture enrichment are orthogonal).
   — **GQA axis fully closed** (#893 NEGATIVE): do not revisit.

### Radical architecture changes (escalation if plateau continues)
6. **Point Transformer V3 (PTv3) volume head**: Replace Transolver vol head with PTv3 architecture; PTv3 uses serialized attention with hilbert-curve ordering and is SOTA for 3D point clouds.
7. **Graph Neural Network vol pathway**: Replace Transolver vol attention with a k-NN GNN (k=8-16 nearest neighbors) for the volume pathway. GNNs are natural for unstructured point clouds and can propagate signals across the OOD geometry more robustly.
8. **Ensemble with vol-SWA members**: If thorfinn #909 vol-SWA wins on single model, re-run all SOTA single-model runs with SWA and add to ensemble pool.

### OOD robustness (ongoing)
9. **Mirror aug + all architecture wins**: If fern #901 mirror aug wins, compose with best architecture variant.
10. **Geometric mixup**: Interpolate between training car geometries (weighted average of surface point clouds + vol points) to generate virtual training cases near the OOD test cases.

### Closed axes — do not revisit
- STRING: ALL axes closed (rff capacity, sigma-ladder, sigma-shift, sigma pruning).
- GradNorm: CONCLUSIVELY CLOSED (4 failures, all variants).
- EMA decay: EXHAUSTED (EMA=0.9999 catastrophic kill).
- Depth scaling: **CLOSED with and without xattn** (L=6 without #811 NEGATIVE; L=6 with #895 NEGATIVE).
- Width scaling: CLOSED (#886 xattn-width-640 NEGATIVE EP3=8.58%).
- Lion beta1, lr, wd (prior sweep range): CLOSED.
- Static loss-weight sweeps (tau_y, tau_z, vol, differential ratio): ALL CLOSED.
- LoRA vol head: CLOSED (rank-collapse VOLUME_Y_DIM=1).
- Channel-level FiLM: CLOSED (gamma saturation).
- 2-layer MLP decoder: CLOSED DEAD END.
- Post-xattn FFN only (no vol self-attention): CLOSED (#891 NEGATIVE, lacks spatial communication).
- Learned surf pool (Perceiver 65k→256): CLOSED (#894 NEGATIVE, destroys spatial locality).
- Xattn positional bias on queries: CLOSED (#883 NEGATIVE, timeout, worse on all channels).
- MLP ratio=2.0 (halved FFN): CLOSED (#897 NEGATIVE, EP3=11.18%).
- Detach-K/V: CLOSED (#890 NEGATIVE, K/V backflow load-bearing).
- Two-layer xattn: CLOSED (#884 R1/#884 R2, EP1 kill gates, K/V backflow too strong).
- GQA/MQA xattn: CLOSED (#893 NEGATIVE, both n_kv_heads=1 and 2 fail EP3; full MHA n_heads=4 is optimal).
- OOD static loss upweighting: CLOSED (#888 NEGATIVE, gap is not data-distribution-limited).

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128 (~16.99M params with xattn)
- **Depth:** L=5 is optimal. **FULLY CLOSED** (L=6 without xattn #811 NEGATIVE, L=6 with xattn #895 NEGATIVE).
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
- **Depth-scaling FULLY CLOSED** (PR #811 without xattn; PR #895 with xattn): L=6 underperforms L=5 in both architectures. Depth does not compound with xattn. Do not increase depth.
- **STRING σ=0.25 is load-bearing** (PR #819): All σ-shift/ladder configurations failed. σ=0.25 encodes panel-scale surface detail critical for L5/4-ep budget.
- **4-ep schedule confound**: Do NOT use `--lr-cosine-t-max 4`. Use `--lr-cosine-t-max 13 --epochs 4`.
