# SENPAI Research State
- **Date:** 2026-05-09 — Round 26 active. All 8 students running. Prior Round 25 PRs #947 askeladd, #948 frieren, #950 alphonse, #930 nezuko replaced by new Round 26 assignments. Active WIP: #955 alphonse, #956 askeladd, #957 frieren, #958 nezuko, #941 edward, #935 tanjiro, #949 thorfinn, #952 fern.
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

- **Issue #618** (STRING/RoPE): **Reopened for targeted exploration.** Prior STRING axes (sigma shifts, global ladder variants) closed. New dimension: (a) slice-centroid local RoPE coordinates (alphonse #955) and (b) sigma-ladder octave range sweep — 7-octave fine/coarse and all-fine 5-octave (askeladd #956). σ=0.25 still assumed load-bearing; hypothesis is that adding sub-panel or super-car scale octaves can reduce vol_p OOD gap.
- **Issue #717** (vol_pressure gap): Phase 2 active — building on xattn win from #823. Post-xattn capacity axis fully closed (0-for-3: #891 FFN, #893 k-NN graph, #906 vol-self-attn). Now exploring loss stratification (#930), geometry-conditioned queries (#950), surface loss annealing (#947), vol-pressure aux head (#948), surface-points curriculum (#949), SDF-modulated vol PE (#935).
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- **Issue #803** (SDF fix): Edward (#941) actively regenerating corrupted volume_sdf.npy for 10 REQUIRED_RESTORED cases; full retrain with fixed SDF data underway (W&B: wtxiaqk0). No new SDF/geometry-conditioning assignments until edward's results are known.
- No new directives pending.

---

## Active PRs (Round 26)

### tay-track (8 WIP, fully occupied)

| Student | PR | Hypothesis | Branch | Status |
|---|---|---|---|---|
| alphonse | #955 | STRING RoPE: slice-centroid local coords as RoPE anchor. Compute per-slice centroids via soft assignment weights; encode (x−cx, y−cy, z−cz) instead of global coords. Arm A: `--rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"` (global scale); Arm B: `--rff-init-sigmas "0.05,0.1,0.25,0.5,1.0"` (local scale). EP3 gate: val_abupt ≤8.0%. `--wandb-group alphonse-string-slice-centroid-rope` | `alphonse/string-rope-slice-centroid-learnable` | Assigned — awaiting implementation |
| askeladd | #956 | STRING σ-ladder geometry sweep: test unexplored octave ranges. Arm A (7-oct fine): `0.03,0.06,0.125,0.25,0.5,1.0,2.0`; Arm B (7-oct coarse): `0.25,0.5,1.0,2.0,4.0,8.0,16.0`; Arm C (5-oct all-fine): `0.03,0.06,0.125,0.25,0.5`. Kill any arm at EP3 if val_abupt > 7.5%. `--wandb-group askeladd-string-sigma-sweep` | `askeladd/string-sigma-ladder-geometry-sweep` | Assigned — awaiting implementation |
| frieren | #957 | Y-flip augmentation for vol_p OOD: y→−y with p=0.5 (negates y coord, ny normal, tau_y; SDF/cp/tau_x/tau_z invariant). Doubles effective training geometries. Arm A: flip only (p=0.5); Arm B: flip + `--volume-loss-weight 2.0`. EP3 gate: val_abupt ≤8.0% AND val_vol_p ≤4.2%. `--wandb-group frieren-yflip-vol-ood` | `frieren/yflip-augmentation-vol-ood` | Assigned — awaiting implementation |
| nezuko | #958 | Dedicated vol_p aux decoder head: independent 3-layer MLP branch (Linear(512,256)→SiLU→Linear(256,64)→SiLU→Linear(64,1)) separate from the shared 4-output surface head. Zero-init final layer. Arm A: `--volume-loss-weight 1.0`; Arm B: `--volume-loss-weight 2.0`. EP3 gate: val_abupt ≤8.0%. `--wandb-group nezuko-vol-aux-decoder-head` | `nezuko/vol-pressure-aux-decoder-head` | Assigned — awaiting implementation |
| thorfinn | #949 | Surface-points curriculum: ramp surface tokens from 16K→65K over training (mirror of vol-points-schedule). Flag: `--surface-points-schedule`. Arm A symmetric (0:16384:3:32778), Arm B gentler (0:32778:3:65536). EP4 gate: val_abupt ≤6.44%. | `thorfinn/surface-points-curriculum` | Assigned — awaiting implementation |
| fern | #952 | Manifold Mixup hidden-state regularization: Beta(α=0.2,0.2) convex interpolation of two samples' backbone transformer hidden states at random layer k∈[0,L-1], p_mix=0.5. Targets val→test generalization gap (vol_p 3.86%→11.67%, 3×). Flags: `--use-manifold-mixup`, `--manifold-mixup-alpha`, `--manifold-mixup-prob`. 4-ep screen first. | `fern/manifold-mixup-backbone` | Assigned — awaiting implementation |
| edward | #941 | SDF data fix: regenerate volume_sdf.npy for 10 corrupted REQUIRED_RESTORED cases (run_44, 133, 158, 184, 203, 226, 249, 310, 416, 484). Synthesis script in `synthesize_inside_body.py`. Full SOTA retrain with fixed data now running (W&B: `wtxiaqk0`, group: edward-sdf-fix). EP1 kill gate: val_abupt < 30% at step 10,864. | `edward/sdf-data-fix-regenerate-corrupted-sdfs` | In progress — SOTA retrain running |
| tanjiro | #935 | SDF-modulated vol PE (per-octave scaler, ~61 params). Identity-init retry: final Linear bias=+4.0 → sigmoid output starts 0.982 (near-transparent at init). Prior run had half-strength PE init (bias=0), causing EP1 regression. Now running with `--epochs 41` (T_max=41). W&B: `ixrg3mg1`, group: tanjiro-sdf-vol-pe-identity-init. | `tanjiro/sdf-modulated-vol-pe` | In progress — EP1 running |

---

## Round 25 Outcomes (Closed / Superseded)

| Student | PR | Result |
|---|---|---|
| alphonse | #950 | Learned geometry Q-bias — SUPERSEDED by Round 26 assignment (#955) |
| askeladd | #947 | Adaptive surface loss annealing — SUPERSEDED by Round 26 assignment (#956) |
| frieren | #948 | Vol-pressure aux decoder head (v1) — SUPERSEDED by Round 26 assignment (#957; different direction: Y-flip aug) |
| nezuko | #930 | SDF-stratified vol loss — SUPERSEDED by Round 26 assignment (#958); Arm B EP3 result unknown, PR closed |
| thorfinn | #949 | Surface-points curriculum — CARRIED OVER to Round 26 (still WIP) |
| fern | #952 | Manifold Mixup — CARRIED OVER to Round 26 (still WIP) |
| edward | #941 | SDF data fix — CARRIED OVER to Round 26 (retrain running) |
| tanjiro | #935 | SDF-modulated vol PE — CARRIED OVER to Round 26 (EP1 running) |

## Round 24 Outcomes (Closed)

| Student | PR | Result |
|---|---|---|
| edward | #929 | Pre-xattn vol self-attn — CLOSED NEGATIVE |
| nezuko | #928 | TTA y-mirror — CLOSED (see experiments log) |
| alphonse | #937 | Mild yaw-only rotation aug — CLOSED (see experiments log) |
| askeladd | #926 | Vol geo features centroid+bbox — CLOSED (see experiments log) |
| frieren | #927 | Surface cp Laplacian aux loss — CLOSED (see experiments log) |
| thorfinn | #921 | Geometric mixup aug — CLOSED (see experiments log) |
| tanjiro | #918 | Vol-specific RFF Arm C — CLOSED; follow-up: SDF-modulated vol PE #935 |
| fern | #901 | Train-time y-mirror aug Arm C — CLOSED; follow-up: GradNorm full-mode #942 |

---

## Current Research Focus (Round 26)

**Primary mandate (Issue #717):** Reduce test_vol_pressure below 11.0% → 10.0% → 8.5% → 6.08% (AB-UPT). All channels must not degrade.

### Theme 1: SDF Data Quality (edward #941)
The 10 REQUIRED_RESTORED cases have corrupted volume_sdf.npy (inside-body cells with SDF=0 or negative). Edward has synthesized correct SDF values using a point-in-mesh check + nearest-boundary interpolation. Full SOTA retrain now running (W&B: `wtxiaqk0`). Highest-priority upstream fix — run_133, 226, 203, 158 (OOD outliers) are in this set. No new SDF/geometry-conditioning assignments until edward's results are known (Issue #803 blocker).

### Theme 2: STRING Positional Encoding (alphonse #955, askeladd #956)
Two new STRING RoPE directions re-opened per Issue #618 mandate:
- **Slice-centroid local RoPE** (#955 alphonse): replace global (x,y,z) anchor with per-slice centroid; tests whether local slice-relative coordinates improve feature encoding
- **Sigma-ladder octave sweep** (#956 askeladd): test 7-octave fine (down to σ=0.03) and 7-octave coarse (up to σ=16.0) vs. SOTA 5-octave; critical test of whether sub-panel or super-car frequencies help

### Theme 3: Data Augmentation for OOD (frieren #957)
Y-flip augmentation: y→−y with p=0.5, negating y coord, ny normal, tau_y while leaving SDF/cp/tau_x/tau_z invariant. Doubles effective training geometries for OOD generalization. Arm B also tests with vol_loss_weight=2.0 for combined augmentation+loss emphasis.

### Theme 4: Dedicated Vol Pressure Decoder (nezuko #958)
Independent 3-layer MLP branch for vol_p prediction, separate from the shared 4-output surface head. Hypothesis: the shared decoder must encode too many competing signals; a dedicated path lets it specialize on vol_p geometry.

### Theme 5: Vol PE Quality (tanjiro #935)
SDF-modulated vol PE: per-octave scaler learned from distance-from-surface. Identity-init (bias=+4.0) ensures transparent start. Rerun with `--epochs 41` to avoid premature LR decay.

### Theme 6: Curriculum Learning (thorfinn #949)
Surface-points curriculum mirrors the successful vol-points-schedule: start with fewer surface tokens, ramp up. Allows earlier-epoch training to capture coarser structure before refinement.

### Theme 7: Manifold Mixup (fern #952)
Hidden-state interpolation between paired training samples at a randomly selected backbone layer. Targets the val→test vol_p generalization gap (3.86% → 11.67%, 3×). With only 400 training cars, hidden-space augmentation should produce smoother interpolations than input-space augmentation.

---

## Potential Next Research Directions (post-Round 26)

### If STRING positional encoding wins (#955 alphonse, #956 askeladd)
1. **Compose slice-centroid RoPE + winning sigma-ladder**: orthogonal within STRING framework — can stack if both win
2. **Learned sigma values via gradient descent**: if fixed ladder wins, allow sigmas to be learned parameters (constrained positive)
3. **Per-axis sigmas**: different sigma ladders for x, y, z axes (z=height may need finer encoding)

### If geometry-conditioning wins (#935 tanjiro)
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
- **GradNorm ema_proxy, EMA, LoRA, FiLM, 2-layer MLP decoder, Huber loss, Lion β2 sweep, BC-type embedding, dual-tower xattn, spectral-norm attn, stochastic depth, PCGrad, surface curvature features**: ALL CLOSED
- **Prior STRING axes**: sigma shift, all-fine/all-coarse ladder variants, vol-specific RFF (#918) — CLOSED. New targeted STRING axes (#955, #956) are live in Round 26.

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
