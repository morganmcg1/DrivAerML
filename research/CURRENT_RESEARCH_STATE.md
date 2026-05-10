# SENPAI Research State
- **Date:** 2026-05-10 — Round 27 active. All 8 students running. Round 26 PRs #935 (tanjiro), #949 (thorfinn), #956 (askeladd) closed/superseded. New assignments: #953 (thorfinn), #959 (tanjiro), #960 (askeladd), #961 (fern geom-q-bias replaces #952 Manifold Mixup CLOSED NEGATIVE). PR #957 (frieren fixed-p yflip) CLOSED NEGATIVE; replaced by #962 curriculum yflip warmup. PR #953 (thorfinn sdf-zone p_max=0.15) CLOSED — false-kill confirmed (loss spike was transient augmentation artefact); replaced by #963 sdf-zone relaunch with relaxed loss threshold.
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

- **Issue #618** (STRING/RoPE): **Reopened for targeted exploration.** New dimension: (a) slice-centroid local RoPE coordinates (alphonse #955) and (b) sigma-bracket sweep — fine-shift σ=0.125–2.0 and coarse-shift σ=0.5–8.0 (askeladd #960). Prior STRING axes (sigma shifts, global ladder variants, 7-octave mega-sweep #956) closed. σ=0.25 still assumed load-bearing; bracket sweep tests whether shifting the 5-sigma window ±1 octave from SOTA improves or degrades.
- **Issue #717** (vol_pressure gap): Phase 2 active — building on xattn win from #823. Post-xattn capacity fully closed (0-for-3: #891 FFN, #893 k-NN graph, #906 vol-self-attn). Now exploring loss stratification, geometry-conditioned queries, surface loss annealing, vol-pressure aux head, SDF-modulated vol PE, coordinate-conditioned vol output head (#959 tanjiro).
- **Issue #803** (SDF fix): Edward (#941) actively running full SOTA retrain with fixed SDF data (W&B: `wtxiaqk0`). No new SDF/geometry-conditioning assignments until edward's results are known.
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- No new directives pending.

---

## Active PRs (Round 27)

### tay-track (8 WIP, fully occupied)

| Student | PR | Hypothesis | Branch | Status |
|---|---|---|---|---|
| alphonse | #955 | STRING RoPE: slice-centroid local coords as RoPE anchor. Compute per-slice centroids via soft assignment weights; encode (x−cx, y−cy, z−cz) instead of global coords. Arm A: `--rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"` (global scale); Arm B: `--rff-init-sigmas "0.05,0.1,0.25,0.5,1.0"` (local scale). EP3 gate: val_abupt ≤8.0%. `--wandb-group alphonse-string-slice-centroid-rope` | `alphonse/string-rope-slice-centroid-learnable` | WIP — running |
| askeladd | #960 | STRING σ-bracket sweep: fine-shift σ=0.125,0.25,0.5,1.0,2.0 (Arm A) vs coarse-shift σ=0.5,1.0,2.0,4.0,8.0 (Arm B). Tests whether the 5-sigma SOTA window is optimally placed or should shift ±1 octave. Arm A (fine-shift) running (W&B: `zhnlo5k5`); EP1 expected ~14:28 UTC. Arm B launches after Arm A EP1. EP4 gate: val_abupt ≤7.5%. `--wandb-group askeladd-string-sigma-bracket` | `askeladd/string-sigma-bracket-sweep` | WIP — Arm A EP1 in progress (run `zhnlo5k5`) |
| frieren | #962 | Curriculum y-flip augmentation for vol_p OOD: linearly ramp yflip probability 0→0.5 over first 3 epochs (`--yflip-prob 0.5 --yflip-warmup-epochs 3`). Avoids early-training instability from immediate 50% flip rate. Kill threshold relaxed: `"2000:val_primary/abupt_axis_mean_rel_l2_pct<30"` (no loss gate — yflip does not cause loss spike). EP3 gate: val_abupt ≤8.0% AND val_vol_p ≤5.0%. EP13 target: ≤6.44%. `--wandb-group frieren-curriculum-yflip` | `frieren/curriculum-yflip-vol-ood` | WIP — assigned |
| nezuko | #958 | Dedicated vol_p aux decoder head: independent 3-layer MLP branch (Linear(512,256)→SiLU→Linear(256,64)→SiLU→Linear(64,1)) separate from the shared 4-output surface head. Zero-init final layer. Arm A: `--volume-loss-weight 1.0`; Arm B: `--volume-loss-weight 2.0`. EP3 gate: val_abupt ≤8.0%. `--wandb-group nezuko-vol-aux-decoder-head` | `nezuko/vol-pressure-aux-decoder-head` | WIP — running |
| thorfinn | #963 | SDF-zone vol token masking Arm-C relaunch (fixed kill threshold): same p_max=0.15 / anneal=6ep design as #953 but with relaxed kill threshold `500:train/loss<10` (not `<5`). PR #953 was false-killed — loss spike at step 500 was a transient augmentation artefact (baseline itself peaks at 3.72 with p_max masking causing ≤6.14 transiently). The `<5` threshold was over-sensitive; `<10` retains genuine divergence detection while surviving transient spikes. EP1 gate: val_abupt ≤30%. EP5 ≤7.5%. EP13 ≤6.44%. `--wandb-group thorfinn-sdf-zone-mask-relaunch` | `thorfinn/sdf-zone-mask-pmax015-relaunch` | WIP — assigned |
| fern | #961 | Geometry-conditioned Q-bias: mean-pool surf hidden state → MLP(512→256→512) zero-init final layer → additive bias on vol Q-projections immediately before surf→vol xattn. ~394K params. New flag `--use-geom-q-bias`. 4-ep screen first; EP4 ≤6.9% gate before 13-ep confirm. `--wandb-group fern-geom-q-bias` | `fern/geom-q-bias` | WIP — assigned |
| edward | #941 | SDF data fix: regenerate volume_sdf.npy for 10 corrupted REQUIRED_RESTORED cases (run_44, 133, 158, 184, 203, 226, 249, 310, 416, 484). Full SOTA retrain with fixed data running (W&B: `wtxiaqk0`, group: edward-sdf-fix). EP1 kill gate: val_abupt < 30% at step 10,864. | `edward/sdf-data-fix-regenerate-corrupted-sdfs` | WIP — SOTA retrain running (W&B: `wtxiaqk0`) |
| tanjiro | #959 | Coordinate-conditioned volume output MLP head: `CoordVolHead` takes [hidden(512) ∥ xyz(3)] → 256 → 64 → 1 (SiLU, zero-init final layer). Hypothesis: vol_p is spatially structured (stagnation nose, suction roof, wake); explicit spatial prior frees backbone capacity. W&B: `ivyxtt02`. **EP1 PASSED (28.39%, ≤30% gate)**. EP2 gate: val_vol_p ≤16% AND val_abupt ≤16%. `--wandb-group tanjiro-coord-vol-head` | `tanjiro/coord-vol-output-head` | WIP — EP1 passed, EP2 in progress (run `ivyxtt02`) |

---

## Round 26 Outcomes (Closed / Superseded)

| Student | PR | Result |
|---|---|---|
| askeladd | #956 | STRING σ-ladder geometry sweep (7-octave fine/coarse/all-fine) — CLOSED NEGATIVE. Arm A (7-oct fine, 0.03–2.0) failed EP1 at 30.27%; too wide a range degrades early convergence. Superseded by bracket sweep #960 |
| thorfinn | #949 | Surface-points curriculum — CLOSED. Results ambiguous; superseded by SDF-zone masking #953 |
| tanjiro | #935 | SDF-modulated vol PE (per-octave scaler, identity-init retry) — CLOSED. Superseded by coordinate-conditioned vol output head #959 |
| alphonse | #955 | STRING RoPE slice-centroid — CARRIED OVER to Round 27 (still WIP) |
| frieren | #957 | Y-flip augmentation (fixed p=0.5) — CLOSED NEGATIVE. No improvement at EP1; curriculum ramp version assigned as #962. |
| nezuko | #958 | Dedicated vol_p aux decoder head — CARRIED OVER to Round 27 (still WIP) |
| thorfinn | #953 | SDF-zone vol masking p_max=0.15 — CLOSED (false-kill; `train/loss<5` threshold fired on transient augmentation spike, not genuine divergence). Relaunched as #963 with relaxed `<10` threshold. |
| fern | #952 | Manifold Mixup — CLOSED NEGATIVE (Round 27); reassigned #961 geom-q-bias |
| edward | #941 | SDF data fix — CARRIED OVER to Round 27 (retrain running) |

## Round 25 Outcomes (Closed)

| Student | PR | Result |
|---|---|---|
| alphonse | #950 | Learned geometry Q-bias — CLOSED (see experiments log) |
| askeladd | #947 | Adaptive surface loss annealing — CLOSED (see experiments log) |
| frieren | #948 | Vol-pressure aux decoder head (v1) — CLOSED (see experiments log) |
| nezuko | #930 | SDF-stratified vol loss — CLOSED (see experiments log) |

---

## Current Research Focus (Round 27)

**Primary mandate (Issue #717):** Reduce test_vol_pressure below 11.0% → 10.0% → 8.5% → 6.08% (AB-UPT). All channels must not degrade.

### Theme 1: SDF Data Quality (edward #941)
The 10 REQUIRED_RESTORED cases have corrupted volume_sdf.npy (inside-body cells with SDF=0 or negative). Edward has synthesized correct SDF values using a point-in-mesh check + nearest-boundary interpolation. Full SOTA retrain now running (W&B: `wtxiaqk0`). Highest-priority upstream fix — run_133, 226, 203, 158 (OOD outliers) are in this set. No new SDF/geometry-conditioning assignments until edward's results are known (Issue #803 blocker).

### Theme 2: STRING Positional Encoding (alphonse #955, askeladd #960)
Two STRING directions live:
- **Slice-centroid local RoPE** (#955 alphonse): replace global (x,y,z) anchor with per-slice centroid; tests whether local slice-relative coordinates improve feature encoding
- **Sigma-bracket sweep** (#960 askeladd): test fine-shift (0.125–2.0) and coarse-shift (0.5–8.0) bracketing the SOTA window (0.25–4.0); Arm A running (W&B: `zhnlo5k5`). Prior 7-octave mega-sweep (#956) failed EP1 — bracket approach is more conservative.

### Theme 3: Data Augmentation for OOD (frieren #962)
Curriculum y-flip augmentation: ramp yflip probability from 0→0.5 over first 3 epochs (`--yflip-prob 0.5 --yflip-warmup-epochs 3`). Negates y coord, ny normal, tau_y while leaving SDF/cp/tau_x/tau_z invariant. Doubles effective training geometries for OOD generalization. Prior PR #957 (fixed p=0.5 from epoch 0) CLOSED NEGATIVE — no val_abupt/vol_p improvement at EP1. Curriculum version avoids early-training disruption from immediate 50% flip rate. Kill threshold uses only the ABUPT EP2 gate (no loss threshold) since yflip does not cause loss spikes.

### Theme 4: Dedicated Vol Pressure Decoder (nezuko #958)
Independent 3-layer MLP branch for vol_p prediction, separate from the shared 4-output surface head. Hypothesis: the shared decoder must encode too many competing signals; a dedicated path lets it specialize on vol_p geometry.

### Theme 5: Coordinate-Conditioned Vol Output Head (tanjiro #959)
`CoordVolHead` MLP concatenates [hidden(512) ∥ xyz(3)] and decodes through 256→64→1. Zero-init final layer ensures transparent start. EP1 passed (28.39%). Now at EP2 gate (val_vol_p ≤16% AND val_abupt ≤16%). The hypothesis is that volume pressure is strongly spatially structured (stagnation at nose, suction at roof, wake behind car), and an explicit spatial prior frees backbone capacity for flow-feature learning. W&B: `ivyxtt02`.

### Theme 6: SDF-Zone Vol Token Masking (thorfinn #963)
Annealed dropout of vol tokens in the near-surface SDF zone [-0.3, 0.05]m. History: p_max=0.30 showed high run-to-run variance at EP1 (±4.2pp); p_max=0.15 on PR #953 was false-killed at step 500 by the over-sensitive `train/loss<5` threshold (training loss transiently hit ~6.14 due to augmentation noise, but this is not a genuine divergence). PR #963 relaunches p_max=0.15 with relaxed `500:train/loss<10` kill threshold. If EP1 val_abupt ≤30% this time, the approach remains viable. If it again fails EP1 on abupt, the masking is too disruptive regardless of loss threshold.

### Theme 7: Geometry-Conditioned Q-Bias (fern #961)
Mean-pool surf hidden state to a global geometry descriptor (512-dim), pass through tiny MLP (512→256→512, zero-init final layer), add as bias to vol Q-projections immediately before surf→vol xattn. ~394K params. Targets the OOD vol_p failure mode (run_133/226/203/158 contribute 92% of squared test deviation): two cars with very different geometries currently produce identical Q distributions from a given vol point. Replaces fern's Manifold Mixup #952 (CLOSED NEGATIVE).

---

## Potential Next Research Directions (post-Round 27)

### If SDF fix (#941 edward) shows vol_p improvement
1. **Re-run ensemble with clean-SDF checkpoint**: if edward's retrain improves test_vol_p significantly, rebuild greedy ensemble
2. **Re-run OOD aug experiments with clean SDF**: mirror aug, rotation aug may have been trained on corrupted data — retest strongest aug variant

### If coordinate-conditioned head (#959) wins
3. **Compose CoordVolHead + SDF-modulated scale**: orthogonal geometry signals; SDF gives distance-from-surface while xyz gives absolute position
4. **Per-octave coordinate conditioning**: different spatial priors for different frequency bands

### If STRING sigma bracket (#960) wins
5. **Compose slice-centroid RoPE (#955) + winning sigma bracket (#960)**: orthogonal within STRING framework
6. **Learned sigma values via gradient descent**: if fixed bracket wins, allow sigmas to be learned parameters (constrained positive)

### If geometry-conditioning wins broadly
7. **Per-case geometry embedding**: learn a geometry code per training car via message passing, use as xattn conditioning
8. **Point Transformer V3 vol head**: replace Transolver vol attention with PTv3 (Hilbert-curve serialized attention)

### Radical escalation (if plateau continues)
9. **GNN vol pathway**: replace Transolver vol attention with k-NN GNN for vol-vol communication (k=8-16)
10. **Fourier Neural Operator vol decoder**: replace MLP vol decoder with FNO for better frequency resolution
11. **Attention-free SSM decoder for vol**: S4/Mamba-based sequence model for vol token decoding

### Closed axes — do not revisit
See "Closed Axes" section below.

---

## Closed Axes (Rounds 22–27)

### Post-xattn capacity: FULLY CLOSED (0-for-3)
- PR #891 (fern post-xattn FFN): EP3=8.50% NEGATIVE
- PR #893 (askeladd vol k-NN graph attn): NEGATIVE
- PR #906 (edward post-xattn vol self-attn): EP3=8.23% NEGATIVE

### Pre-xattn capacity: CLOSED (0-for-1)
- PR #929 (edward pre-xattn vol self-attn): CLOSED NEGATIVE

### STRING axes closed
- PR #956 askeladd: 7-octave mega-sweep (0.0625–8.0) — CLOSED NEGATIVE (EP1=30.27%)
- Prior STRING axes: sigma shift, all-fine/all-coarse ladder variants, vol-specific RFF (#918) — ALL CLOSED
- σ=0.25 confirmed load-bearing (#819); σ-shift/ladder failed

### All other closed axes (do not revisit)
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
- **Geometric mixup aug**: #921 thorfinn CLOSED (Round 24)
- **Vol geo features centroid+bbox**: #926 askeladd CLOSED (Round 24)
- **Surface cp Laplacian aux loss**: #927 frieren CLOSED (Round 24)
- **Vol-specific RFF sigmas (Arm A/B/C)**: #918 tanjiro CLOSED
- **Train-time y-mirror aug**: #901 fern CLOSED (Round 24 Arm C)
- **TTA y-mirror**: #928 nezuko CLOSED (Round 24)
- **Mild yaw-only rotation aug**: #937 alphonse CLOSED (Round 24)
- **GradNorm full-mode α=1.5** (#942 fern): CLOSED NEGATIVE — vp weight decreased at EP1, all weights converge to 0.91–1.11 band; vol_p is not gradient-starved under Lion; 5× backward overhead consumed budget before EP6 gate
- **GradNorm ema_proxy, EMA, LoRA, FiLM, 2-layer MLP decoder, Huber loss, Lion β2 sweep, BC-type embedding, dual-tower xattn, spectral-norm attn, stochastic depth, PCGrad, surface curvature features**: ALL CLOSED
- **SDF-modulated vol PE (identity-init)**: #935 tanjiro CLOSED; superseded by coord-conditioned vol head #959
- **Surface-points curriculum**: #949 thorfinn CLOSED; superseded by SDF-zone masking #953
- **Learned geometry Q-bias (static)**: #950 alphonse CLOSED

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128, surf→vol xattn (1 layer, 4 heads, ~1.05M params) → ~16.99M total
- **Depth:** L=5 optimal. FULLY CLOSED.
- **Positional encoding:** STRING-separable (rff=16, sigmas {0.25,0.5,1.0,2.0,4.0}). σ=0.25 load-bearing.
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
- **GradNorm full-mode CLOSED NEGATIVE** (#942): vp weight goes DOWN (not up) because vp loss decreases fastest; all 5 weights converge to 0.91–1.11 band at EP1. GradNorm axis fully closed.
- **SDF-zone masking at p_max=0.30 is unstable** (#953): two identical runs produced EP1 val_abupt=28.89% and 32.84% (±4.2pp). Stochastic BL masking disrupts whole backbone globally, not just vol_p channel. p_max=0.15 is being tested as a lower-risk variant.
- **Coord-conditioned vol head EP1 passes cleanly** (#959, run `ivyxtt02`): val_abupt=28.39% (vs baseline 28.63%), val_vol_p=17.06%. Promising early signal.
- **10 REQUIRED_RESTORED cases have corrupted SDF data** (#941): inside-body cells had SDF=0 or negative; synthesis fix applied. Run_133, 226, 203, 158 (OOD outliers) are in this set — SDF corruption may explain part of the vol_p OOD gap.
