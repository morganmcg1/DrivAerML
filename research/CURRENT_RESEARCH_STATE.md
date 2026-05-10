# SENPAI Research State
- **Date:** 2026-05-10 — Round 25 active. All 8 students running. PR #942 fern (GradNorm full-mode α=1.5) closed NEGATIVE. New assignment: PR #952 fern (Manifold Mixup hidden-state regularization). Active WIP: #947 askeladd, #948 frieren, #949 thorfinn, #950 alphonse, #941 edward, #935 tanjiro, #930 nezuko, #952 fern.
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
- **Issue #717** (vol_pressure gap): Phase 2 active — building on xattn win from #823. Post-xattn capacity axis fully closed (0-for-3: #891 FFN, #893 k-NN graph, #906 vol-self-attn). Now exploring loss stratification (#930), geometry-conditioned queries (#950), surface loss annealing (#947), vol-pressure aux head (#948), surface-points curriculum (#949), SDF-modulated vol PE (#935).
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- **Issue #803** (SDF fix): Edward (#941) actively regenerating corrupted volume_sdf.npy for 10 REQUIRED_RESTORED cases; full retrain with fixed SDF data underway (W&B: wtxiaqk0). No new SDF/geometry-conditioning assignments until edward's results are known.
- No new directives pending.

---

## Active PRs (Round 25)

### tay-track (8 WIP, fully occupied)

| Student | PR | Hypothesis | Branch | Status |
|---|---|---|---|---|
| alphonse | #950 | Learned geometry Q-bias: mean-pool surf_hidden → [B,1,512] global descriptor → zero-init Linear(512,512) → add as bias to ALL vol_hidden before xattn Q-projection. ~394K params, zero-init. Flag: `--use-geom-q-bias`. EP3 gate: val_abupt ≤8.0% AND val_vol_p ≤3.86%. | `alphonse/learned-geometry-q-bias` | Assigned, awaiting implementation |
| thorfinn | #949 | Surface-points curriculum: ramp surface tokens from 16K→65K over training (mirror of vol-points-schedule). Flag: `--surface-points-schedule`. Arm A symmetric (0:16384:3:32778), Arm B gentler (0:32778:3:65536). EP4 gate: val_abupt ≤6.44%. | `thorfinn/surface-points-curriculum` | Assigned, awaiting implementation |
| frieren | #948 | Vol-pressure dedicated aux decoder head: `nn.Sequential(Linear(512,256), GELU, Linear(256,1))` with zero-init final layer. Flags: `--vol-pressure-aux-head`, `--vol-pressure-aux-weight`. Arms: weight=1.0 and weight=2.0. EP4 gate: val_abupt ≤6.44%. | `frieren/vol-pressure-aux-head` | Assigned, awaiting implementation |
| askeladd | #947 | Adaptive surface loss annealing: cosine decay from high start weight to low end weight over training. Flags: `--surface-loss-weight` (start) and `--surface-loss-weight-final`. Arm A: 3.0→0.5, Arm B: 2.5→1.0. EP4 gate: val_abupt ≤6.44%. | `askeladd/adaptive-surface-loss-weighting` | Assigned, awaiting implementation |
| fern | #952 | Manifold Mixup hidden-state regularization: Beta(α=0.2,0.2) convex interpolation of two samples' backbone transformer hidden states at random layer k∈[0,L-1], p_mix=0.5. Targets val→test generalization gap (vol_p 3.86%→11.67%, 3×). Flags: `--use-manifold-mixup`, `--manifold-mixup-alpha`, `--manifold-mixup-prob`. 4-ep screen first. | `fern/manifold-mixup-backbone` | Assigned — awaiting implementation |
| edward | #941 | SDF data fix: regenerate volume_sdf.npy for 10 corrupted REQUIRED_RESTORED cases (run_44, 133, 158, 184, 203, 226, 249, 310, 416, 484). Synthesis script in `synthesize_inside_body.py`. Full SOTA retrain with fixed data now running (W&B: `wtxiaqk0`, group: edward-sdf-fix). EP1 kill gate: val_abupt < 30% at step 10,864. | `edward/sdf-data-fix-regenerate-corrupted-sdfs` | In progress — SOTA retrain running |
| tanjiro | #935 | SDF-modulated vol PE (per-octave scaler, ~61 params). Identity-init retry: final Linear bias=+4.0 → sigmoid output starts 0.982 (near-transparent at init). Prior run had half-strength PE init (bias=0), causing EP1 regression. Now running with `--epochs 41` (T_max=41) to avoid premature LR decay before budget timeout. W&B: `ixrg3mg1`, group: tanjiro-sdf-vol-pe-identity-init. EP1 ETA ~07:42 UTC. | `tanjiro/sdf-modulated-vol-pe` | In progress — EP1 running |
| nezuko | #930 | SDF-stratified vol loss: weight vol cells by exp(-SDF/λ), λ controls near-surface emphasis. Arm A (λ=0.10) FAIL EP3: abupt=7.34% (pass) but vol_p=4.564% (fail ≤3.86%); far-field under-fit. Arm B (λ=0.30) launched at 04:09 UTC, EP3 gate ETA ~05:41 UTC. Two-consecutive-failure kill rule: if Arm B also fails, close PR. | `nezuko/volume-ood-domain-adapter` | In progress — Arm B EP3 pending |

---

## Round 24 Outcomes (Closed)

| Student | PR | Result |
|---|---|---|
| edward | #929 | Pre-xattn vol self-attn — CLOSED NEGATIVE (0-for-3 pattern holds for pre-xattn too; Round 25 PRs assigned) |
| nezuko | #928 | TTA y-mirror — CLOSED (merged or closed based on results; see experiments log) |
| alphonse | #937 | Mild yaw-only rotation aug — CLOSED (see experiments log) |
| askeladd | #926 | Vol geo features centroid+bbox — CLOSED (see experiments log) |
| frieren | #927 | Surface cp Laplacian aux loss — CLOSED (see experiments log) |
| thorfinn | #921 | Geometric mixup aug — CLOSED (see experiments log) |
| tanjiro | #918 | Vol-specific RFF Arm C — CLOSED; follow-up: SDF-modulated vol PE #935 |
| fern | #901 | Train-time y-mirror aug Arm C — CLOSED; follow-up: GradNorm full-mode #942 |

---

## Current Research Focus

**Primary mandate (Issue #717):** Reduce test_vol_pressure below 11.0% → 10.0% → 8.5% → 6.08% (AB-UPT). All channels must not degrade.

### Theme 1: SDF Data Quality (edward #941)
The 10 REQUIRED_RESTORED cases have corrupted volume_sdf.npy (inside-body cells with SDF=0 or negative). Edward has synthesized correct SDF values using a point-in-mesh check + nearest-boundary interpolation. Full SOTA retrain now running. This is the highest-priority upstream fix — if corrupted SDF data degrades model performance, fixing it may materially improve vol_pressure especially on the OOD test cases (run_133, run_226, run_203, run_158 are in the REQUIRED_RESTORED set).

### Theme 2: Geometry-Conditioned Xattn Queries (alphonse #950)
Hypothesis: providing vol tokens with a global car-geometry descriptor before forming xattn queries allows the model to condition its surface-lookup on car shape. Zero-init ensures identical-to-baseline at epoch 0. Most promising pre-xattn intervention.

### Theme 3: Loss Formulation (askeladd #947, frieren #948, nezuko #930)
Three parallel loss experiments:
- **Surface loss annealing** (#947): start with high surface weight (strong early supervision) then decay to normal — prevents surface-overfitting while maintaining surface quality
- **Vol-pressure aux head** (#948): dedicated capacity for vol_p prediction; the shared head must simultaneously decode sp, tau, and vp from the same hidden state
- **SDF-stratified vol loss** (#930): emphasize near-surface vol cells where OOD cases differ most from training distribution

### Theme 4: Curriculum Learning (thorfinn #949)
Surface-points curriculum mirrors the successful vol-points-schedule: start with fewer surface tokens, ramp up. Allows earlier-epoch training to capture coarser surface structure before refinement.

### Theme 5: Vol PE Quality (tanjiro #935)
SDF-modulated vol PE: learn per-octave attention scaling based on distance-from-surface. Near-surface vol cells need panel-scale encoding; far-field cells need domain-scale. The identity-init (bias=+4.0) ensures the scaler starts transparent and learns to selectively attenuate.

### Theme 6: Manifold Mixup (fern #952)
Hidden-state interpolation between paired training samples at a randomly selected Transolver backbone layer. Targets the val→test vol_p generalization gap (3.86% → 11.67%, 3×). With only 400 training cars, augmenting in hidden space rather than input space should produce smoother interpolations — and is orthogonal to all current experiments. GradNorm full-mode (PR #942) closed NEGATIVE: EP2 metrics were ~15.7% abupt, but vp weight went DOWN at EP1 (not up), confirming vol_p is not gradient-starved under Lion optimizer. GradNorm axis fully closed.

---

## Potential Next Research Directions (post-Round 25)

### If geometry-conditioning wins (#950 alphonse, #935 tanjiro)
1. **SDF-conditioned Q-bias composition**: combine geom-Q-bias (#950) + SDF-modulated PE (#935) — orthogonal geometry signals
2. **Per-case geometry embedding**: learn a geometry code per training car via message passing, use as xattn conditioning
3. **Point Transformer V3 vol head**: replace Transolver vol attention with PTv3 (Hilbert-curve serialized attention)

### If SDF fix (#941) reveals data corruption impact
4. **Re-run OOD aug experiments with clean SDF**: mirror aug, rotation aug may have been trained on corrupted data — retest strongest aug variant
5. **Re-run ensemble with clean-SDF checkpoint**: if edward's retrain improves test_vol_p significantly, rebuild greedy ensemble

### If loss experiments win (#947, #948, #930)
6. **Compose loss winners**: surface loss annealing + vol-pressure aux head are orthogonal — compose if both win
7. **OOD-targeted aux head**: if vol-pressure aux head works, add OOD-specific branch trained only on worst-case validation set

### If surface curriculum wins (#949)
8. **Joint surface+vol curriculum**: ramp both surface and vol points simultaneously; test symmetric vs asymmetric schedules

### Radical escalation (if plateau continues)
9. **Point Transformer V3 vol head**: replace Transolver vol attention with PTv3 architecture
10. **GNN vol pathway**: replace Transolver vol attention with k-NN GNN for vol-vol communication (k=8-16)
11. **Manifold Mixup**: interpolate backbone hidden states — ASSIGNED (#952 fern)
12. **Fourier Neural Operator vol decoder**: replace MLP vol decoder with FNO for better frequency resolution

### Closed axes — do not revisit
See "Closed Axes" section below.

---

## Closed Axes (Rounds 22–25)

### Post-xattn capacity: FULLY CLOSED (0-for-3)
- PR #891 (fern post-xattn FFN): EP3=8.50% NEGATIVE
- PR #893 (askeladd vol k-NN graph attn): NEGATIVE
- PR #906 (edward post-xattn vol self-attn): EP3=8.23% NEGATIVE

### Pre-xattn capacity: CLOSED (0-for-1)
- PR #929 (edward pre-xattn vol self-attn): CLOSED NEGATIVE (see experiments log)

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
- **Vol loss upweighting (curriculum)**: #902 CLOSED (vol_w=3.0 degraded all channels)
- **Mid-backbone surf→vol xattn**: #917 CLOSED NEGATIVE (Round 24)
- **Rotation aug aggressive (yaw±5°/pitch±3°/p=0.5)**: #925 CLOSED NEGATIVE (EP3=9.11%)
- **Geometric mixup aug**: #921 thorfinn CLOSED (Round 24, see experiments log)
- **Vol geo features centroid+bbox**: #926 askeladd CLOSED (Round 24)
- **Surface cp Laplacian aux loss**: #927 frieren CLOSED (Round 24)
- **Vol-specific RFF sigmas (Arm A/B/C)**: #918 tanjiro CLOSED (follow-up is SDF-modulated PE)
- **Train-time y-mirror aug**: #901 fern CLOSED (Round 24 Arm C, see experiments log)
- **TTA y-mirror**: #928 nezuko CLOSED (Round 24)
- **Mild yaw-only rotation aug**: #937 alphonse CLOSED (Round 24)
- **GradNorm full-mode α=1.5** (#942 fern): CLOSED NEGATIVE — vp weight decreased at EP1 (not increased), all 5 weights converge to 0.91–1.11 band; vol_p is not gradient-starved under Lion; 5× backward overhead consumed budget before EP6 gate
- **STRING, GradNorm ema_proxy, EMA, LoRA, FiLM, 2-layer MLP decoder, Huber loss, Lion β2 sweep, BC-type embedding, dual-tower xattn, spectral-norm attn, stochastic depth, PCGrad, surface curvature features**: ALL CLOSED

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128, surf→vol xattn (1 layer, 4 heads, ~1.05M params) → ~16.99M total
- **Depth:** L=5 optimal. FULLY CLOSED.
- **Positional encoding:** STRING-separable (rff=16, sigmas {0.25,0.5,1.0,2.0,4.0}). ALL STRING axes CLOSED. σ=0.25 load-bearing.
- **Optimizer:** Lion, lr=9e-5, β₁=0.9, β₂=0.99, wd=5e-4
- **QK-norm:** enabled
- **Loss weights:** tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0. All static loss-weight axes CLOSED.
- **EMA:** 0.999
- **4-ep screen schedule:** `--lr-cosine-t-max 4 --epochs 4` with vol-curriculum `0:16384:1:32768:2:49152:3:65536`
- **`--no-compile-model` required:** torch.compile + DDP NCCL deadlock at step 1; also required for `--gradnorm-mode full`
- **`find_unused_parameters=True` required for DDP when using conditional modules**

## Key Diagnostic Findings Established

- **4 OOD test cases dominate vol_pressure** (#767): 92% of squared deviation. Excluding them, test_vol_p = 3.9–4.2% (below AB-UPT). Geometry conditioning is the right lever.
- **Surf→vol xattn is broad-spectrum win, not OOD fix** (#823): OOD test/val ratio unchanged (3.027×). The 4 outlier cases still drive vol_p gap.
- **Post-xattn capacity is a dead end** (0-for-3: #891, #893, #906): all capacity added AFTER surf→vol xattn fails to improve. The bottleneck is in the xattn query quality or the K/V representation, not post-xattn processing.
- **Pre-xattn vol self-attention also failed** (#929): symmetry does not hold; pre-xattn capacity not useful either.
- **STRING σ=0.25 is load-bearing** (#819): σ-shift/ladder failed. Encodes panel-scale surface detail critical for L5/4-ep budget.
- **K/V gradient backflow is load-bearing** (#890): detach-K/V failed. Two-layer xattn also killed by backflow. Full MHA gradient is necessary.
- **Full MHA n_heads=4 is optimal** (#893): both GQA (n_kv_heads=2) and MQA (n_kv_heads=1) fail EP3 gate.
- **GradNorm full-mode CLOSED NEGATIVE** (#942): vp weight goes DOWN (not up) because vp loss decreases fastest; all 5 weights converge to 0.91–1.11 band at EP1. GradNorm axis fully closed — vol_p is not gradient-starved under Lion optimizer.
- **10 REQUIRED_RESTORED cases have corrupted SDF data** (#941): inside-body cells had SDF=0 or negative; synthesis fix applied. Run_133, 226, 203, 158 (OOD outliers) are in this set — SDF corruption may explain part of the vol_p OOD gap.
