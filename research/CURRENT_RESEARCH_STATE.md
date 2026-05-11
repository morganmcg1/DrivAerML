# SENPAI Research State
- **Date:** 2026-05-09 — Round 27 active (8 WIP). PR #966 (tanjiro SDF scalar vol input feature) CLOSED NEGATIVE — EP4 val_abupt=7.6359% (gate ≤6.9%, delta +1.20pp vs SOTA). Root cause: raw `(sdf − μ) / σ` normalization created +20,000σ outliers (SDF max ~36,000 m, σ=1.681517 m) that dominated the vol input projection Linear(98→512), masking all other geometric features. New assignment: tanjiro #973 (bounded SDF encoding — tanh/asinh normalization, 2-arm sweep, scale=2.0 m). PR #959 (tanjiro CoordVolHead) CLOSED NEGATIVE — failed EP4 gate (7.428% vs 6.9%); STRING-sep RoPE + surf→vol xattn already encode xyz-awareness, raw-coord injection at output head redundant. PR #963 (thorfinn sdf-zone relaunch) CLOSED ABANDONED — no replicable positive signal above variance floor (~2.26pp run-to-run). New assignments: tanjiro #966 (SDF scalar vol input feature), thorfinn #967 (SDF-FiLM vol conditioning), fern #969 (SDF-PE octave scaling H2). PR #962 (frieren curriculum-yflip) CLOSED NEGATIVE — EP3 vol_p gate FAIL: vol_p=5.9499% at EP2 with slope −0.0011%/step; projected landing ~5.924% at EP3 gate, cannot reach ≤5.0%. Curriculum ramp slows vol_p convergence — augmentation approach on this axis is closed. frieren now IDLE — new assignment needed.
- **CRITICAL UPDATE (2026-05-09):** Nezuko PR #958 run `29nohj67` (vol aux decoder head) has reached **EP7 val_abupt=6.3885%**, beating single-model SOTA of 6.4407% (delta −0.052pp). Currently in EP8 (step 54,610). Trajectory healthy: EP5=6.45%→EP6=6.45%→EP7=6.39%. If EP8–12 continue improving, this will be a clear merge winner. **Monitoring closely.**
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
- **Issue #803** (SDF fix): Edward (#941) actively running full SOTA retrain with fixed SDF data (W&B: `2ub8dmy7`). EP4 gate PASSED: val_abupt=6.8533% (gate ≤7.5%). Continuing to EP7 (val_abupt ≤6.9%). No new SDF/geometry-conditioning assignments until edward's results are known.
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- No new directives pending.

---

## Active PRs (Round 27)

### tay-track (8 WIP, fully occupied — tanjiro re-assigned #973 after #966 CLOSED NEGATIVE)

| Student | PR | Hypothesis | Branch | Status |
|---|---|---|---|---|
| alphonse | #970 | STRING-sep frozen-freq ablation: register `log_freq` as non-trainable buffer (frozen at SOTA octave init {0.25,0.5,1.0,2.0,4.0}); keep only `phase` as `nn.Parameter`. Tests whether gradient-based freq adaptation helps or the octave init is doing all the work. Flag: `--rff-freeze-freqs`. 4-ep screen; EP3 gate val_abupt ≤8.0% AND vol_p ≤5.0%. `--wandb-group alphonse-string-frozen-freq-ablation` | `alphonse/string-frozen-freq-ablation` | WIP — assigned |
| askeladd | #960 | STRING σ-bracket sweep: fine-shift σ=0.125,0.25,0.5,1.0,2.0 (Arm A) vs coarse-shift σ=0.5,1.0,2.0,4.0,8.0 (Arm B). Tests whether the 5-sigma SOTA window is optimally placed or should shift ±1 octave. Arm A (fine-shift) running (W&B: `zhnlo5k5`); EP1 expected ~14:28 UTC. Arm B launches after Arm A EP1. EP4 gate: val_abupt ≤7.5%. `--wandb-group askeladd-string-sigma-bracket` | `askeladd/string-sigma-bracket-sweep` | WIP — Arm A EP1 in progress (run `zhnlo5k5`) |
| frieren | #971 | Learned distance RPB on surf→vol cross-attention: add per-pair (vol_i, surf_j) Euclidean distance bias to xattn logits: `A_ij = (Q_i·K_j)/√d + f(dist_ij)`. MLP: `Linear(1→16)→SiLU→Linear(16→1)`, ~49 params, zero-init output layer. Flag: `--use-xattn-distance-rpe`. 4-ep screen; EP3 gate ≤8.0% AND vol_p ≤5.0%, EP4 ≤7.5%. `--wandb-group frieren-xattn-distance-rpe` | `frieren/xattn-distance-rpe` | WIP — assigned |
| nezuko | #958 | Dedicated vol_p aux decoder head: independent 3-layer MLP branch (Linear(512,256)→SiLU→Linear(256,64)→SiLU→Linear(64,1)) separate from the shared 4-output surface head. Zero-init final layer. Arm A: `--volume-loss-weight 1.0`; Arm B: `--volume-loss-weight 2.0`. EP3 gate: val_abupt ≤8.0%. `--wandb-group nezuko-vol-aux-decoder-head` | `nezuko/vol-pressure-aux-decoder-head` | WIP — running |
| thorfinn | #974 | SDF-conditioned attention temperature on surf→vol xattn: tiny MLP (Linear(1→16)→SiLU→Linear(16→1)→Sigmoid, ~49 params, zero-init final layer) reads per-vol-token SDF value, outputs τ∈(0,1); scale=0.5+τ∈(0.5,1.5) multiplies per-token logits. Near-surface tokens → sharper attention; far-field tokens → broader attention. Custom manual xattn forward extracts Q/K/V from in_proj_weight. Identity-at-init (scale=1.0 at init). Flag: `--use-sdf-xattn-temperature`. EP3 gate ≤8.0% AND vol_p ≤5.0%, EP4 ≤6.9%. `--wandb-group thorfinn-sdf-xattn-temperature` | `thorfinn/sdf-xattn-temperature` | WIP — assigned (PR #974) |
| fern | #968 | SDF-modulated vol PE: per-octave spectral scaling of STRING-sep RFF features conditioned on per-token SDF distance. Tiny MLP (Linear(1→16)→SiLU→Linear(16→5)→Softplus, ~112 params, identity-init) reads SDF value and outputs 5 octave weight scalars. Each STRING-sep sigma group scaled per-token. Flag: `--use-sdf-pe-scaling`. EP4 ≤6.9% gate. `--wandb-group fern-sdf-pe-octave-scaling` | `fern/sdf-pe-octave-scaling` | WIP — assigned |
| edward | #941 | SDF data fix: regenerate volume_sdf.npy for 10 corrupted REQUIRED_RESTORED cases (run_44, 133, 158, 184, 203, 226, 249, 310, 416, 484). Full SOTA retrain with fixed data running (W&B: `2ub8dmy7`, group: edward-sdf-fix). EP4 gate PASSED: val_abupt=6.8533% (≤7.5%). Next gate: EP7 val_abupt ≤6.9%. | `edward/sdf-data-fix-regenerate-corrupted-sdfs` | WIP — EP4 PASSED (W&B: `2ub8dmy7`), continuing to EP7 |
| tanjiro | #973 | Bounded SDF encoding: replace raw `(sdf−μ)/σ` normalization (+20,000σ outliers) with `tanh(sdf_centered/scale)` or `asinh(sdf_centered/scale)`, scale=2.0 m. Two-arm sweep: Arm A (tanh) vs Arm B (asinh). New flags `--sdf-vol-encoding {linear,tanh,asinh}` and `--sdf-vol-scale`. Scale 2.0 m maps near-surface cells (BL, 0.01–0.1 m) to (0, 0.76) with full resolution; saturates far-field wake (1–5 m+) to ≈±1. Centers by mean (0.222682 m), does NOT divide by σ (1.681517 m) — σ-scaling is what caused outliers. Tightened EP2 gate: ≤10.0% (reference: PR #966 unhealthy hit 11.47%); EP3 ≤8.0% AND vol_p ≤5.0%; EP4 ≤6.9%. `--wandb-group tanjiro-bounded-sdf-encoding` | `tanjiro/bounded-sdf-encoding` | WIP — assigned (PR #973) |

---

## Round 26 Outcomes (Closed / Superseded)

| Student | PR | Result |
|---|---|---|
| askeladd | #956 | STRING σ-ladder geometry sweep (7-octave fine/coarse/all-fine) — CLOSED NEGATIVE. Arm A (7-oct fine, 0.03–2.0) failed EP1 at 30.27%; too wide a range degrades early convergence. Superseded by bracket sweep #960 |
| thorfinn | #949 | Surface-points curriculum — CLOSED. Results ambiguous; superseded by SDF-zone masking #953 |
| tanjiro | #966 | SDF scalar vol input feature (raw linear norm) — CLOSED NEGATIVE. EP4 val_abupt=7.6359% (gate ≤6.9%, delta +1.20pp vs SOTA). Root cause: `(sdf−μ)/σ` normalization — train-set SDF std=1.681517 m, max ~36,000 m → +20,000σ outliers dominated vol input projection. Bounded encoding follow-up: #973 (tanh/asinh, scale=2.0 m). |
| tanjiro | #935 | SDF-modulated vol PE (per-octave scaler, identity-init retry) — CLOSED. Superseded by coordinate-conditioned vol output head #959 |
| alphonse | #955 | STRING RoPE slice-centroid — CLOSED NEGATIVE. EP3 gate failed: val_abupt=8.1833% (gate ≤8.0%), vol_p=5.9055% (gate ≤5.0%). Local centroid coords did not improve over global. New assignment: #970 frozen-freq ablation. |
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
The 10 REQUIRED_RESTORED cases have corrupted volume_sdf.npy (inside-body cells with SDF=0 or negative). Edward has synthesized correct SDF values using STL rejection sampling + pyvista `compute_implicit_distance`. Full SOTA retrain running (W&B: `2ub8dmy7`, group: `edward-sdf-fix`). **EP4 gate PASSED: val_abupt=6.8533% (gate ≤7.5%).** Trajectory EP1→EP4: 27.47% → 8.37% → 7.32% → 6.85% — healthy convergence. Per-case EP3 diagnostic confirmed OOD-4 still at ~102% mean vol_p (expected — 16,384 vol_points yields ~2–3 inside-body samples/batch; SDF fix won't express until EP9+ at 65,536 vol_points). Next gate: EP7 val_abupt ≤6.9%. Key test: EP10 val_vol_p ≤8.0%. No new SDF/geometry-conditioning assignments until edward's EP13 results are known (Issue #803 blocker).

### Theme 2: STRING Positional Encoding (alphonse #970, askeladd #960)
Two STRING directions live:
- **Frozen-freq ablation** (#970 alphonse): freeze `log_freq` as non-trainable buffer (octave init), keep only `phase` as `nn.Parameter`. Tests whether gradient-based frequency adaptation is necessary or the multi-sigma init is sufficient. #955 CLOSED NEGATIVE — slice-centroid local RoPE failed EP3 gate (abupt=8.1833%, vol_p=5.9055%).
- **Sigma-bracket sweep** (#960 askeladd): test fine-shift (0.125–2.0) and coarse-shift (0.5–8.0) bracketing the SOTA window (0.25–4.0); Arm A running (W&B: `zhnlo5k5`). Prior 7-octave mega-sweep (#956) failed EP1 — bracket approach is more conservative.

### Theme 3: Data Augmentation for OOD — FULLY CLOSED
Both y-flip augmentation approaches have been tested and failed:
- **PR #957** (fixed p=0.5 from EP0): CLOSED NEGATIVE — no val improvement at EP1, higher vol_p
- **PR #962** (curriculum ramp 0→0.5 over 3ep): CLOSED NEGATIVE — EP3 vol_p gate FAIL (5.9499% → projected ~5.924% vs ≤5.0% gate). Curriculum ramp slows vol_p convergence; insufficient headroom to pass gate.
y-flip augmentation axis is FULLY CLOSED.

### Theme 8: Learned Relative Position Bias on surf→vol xattn (frieren #971)
Per-pair (vol_i, surf_j) distance-based additive bias to cross-attention logits: `A_ij = (Q_i·K_j)/√d + f(dist_ij)`. Tiny MLP (`Linear(1→16)→SiLU→Linear(16→1)`, ~49 params) learns distance-to-bias mapping; zero-init output layer preserves baseline convergence at step 0. Geometric analogue of ALiBi (Press et al., ICLR 2022). Key distinction from all closed experiments: operates on logits per-pair (not global Q-bias like #961, not static PE offset like #883). Inductive prior: closer surface points should dominate; model currently learns locality from data alone. 4-ep screen, standard gates.

### Theme 4: Dedicated Vol Pressure Decoder (nezuko #958)
Independent 3-layer MLP branch for vol_p prediction, separate from the shared 4-output surface head. Hypothesis: the shared decoder must encode too many competing signals; a dedicated path lets it specialize on vol_p geometry.

### Theme 5: Bounded SDF Encoding for Vol-Token Input (tanjiro #973)
PR #966 (raw `(sdf−μ)/σ` SDF concat) CLOSED NEGATIVE — EP4 val_abupt=7.6359% (gate ≤6.9%). Root cause diagnosis: the training-set SDF has mean=0.222682 m, std=1.681517 m, max ~36,000 m. Linear normalization created +20,000σ outliers that dominated the vol input projection Linear(98→512) throughout the early curriculum, masking all other geometric features. The direction (SDF as vol-token input feature) is retained because the mechanistic failure is well-understood and fixable.

PR #973 (bounded SDF encoding): replace linear normalization with `tanh(sdf_centered/scale)` or `asinh(sdf_centered/scale)`, scale=2.0 m. Two-arm sweep: Arm A (tanh) vs Arm B (asinh). Center by mean; do NOT divide by σ. Scale 2.0 m maps near-surface cells (BL, 0.01–0.1 m) to (0, 0.76) with full resolution; saturates far-field wake (1–5 m+) to ≈±1. Tightened EP2 gate to ≤10.0% (reference: PR #966 unhealthy hit 11.47%, healthy should be ~7.8–8.1%). New flags: `--sdf-vol-encoding {linear,tanh,asinh}` and `--sdf-vol-scale`.

If tanh/asinh wins: next step is composing with SDF-FiLM (#967) — input-level vs. hidden-state-level SDF conditioning are orthogonal.

### Theme 6: SDF-Conditioned Attention Temperature on surf→vol xattn (thorfinn #974)
SDF-FiLM (#967) CLOSED NEGATIVE — EP3 val_abupt failed gate. New direction: per-vol-token scaling of surf→vol xattn logit temperature as a function of SDF distance. Near-surface tokens (small SDF) → τ→0 → scale→0.5 → sharper attention; far-field tokens (large SDF) → τ→1 → scale→1.5 → broader attention. Tiny MLP (Linear(1→16)→SiLU→Linear(16→1)→Sigmoid, ~49 params) outputs τ∈(0,1). Applied as `logits[i] = (Q_i @ K.T) * (0.5 + τ_i) / sqrt(d_head)`. Identity-at-init (sigmoid(0)=0.5 → scale=1.0). Custom manual attention forward required (nn.MultiheadAttention doesn't expose per-token temperature). Flag: `--use-sdf-xattn-temperature`. EP3 gate ≤8.0% AND vol_p ≤5.0%, EP4 ≤6.9%. Physically motivated: near-surface vol tokens encode pressure gradients for which surface-to-volume information is highly localized; far-field tokens draw from broader surface context.

### Theme 7: SDF-Modulated Vol PE via Per-Octave Spectral Scaling (fern #968)
Tiny MLP (Linear(1→16)→SiLU→Linear(16→5)→Softplus, ~112 params) reads each vol token's SDF distance value and outputs 5 scalar octave weights — one per STRING-sep sigma group (0.25, 0.5, 1.0, 2.0, 4.0). The weights are applied to the corresponding frequency group within the STRING-sep RFF encoding before vol-token projection. Identity-init via final bias ≈ 0.541 (Softplus(0.541) ≈ 1.0). Hypothesis: vol tokens near the surface (small SDF) should emphasize fine-scale (low-sigma) frequencies; far-field tokens should emphasize coarse-scale (high-sigma) frequencies. This SDF-conditioned spectral emphasis is a physically motivated prior: boundary layer dynamics are dominated by fine-scale pressure variations, while bulk flow variation is captured by coarse-scale features. Replaces fern's closed geom-Q-bias (#961 NEGATIVE).

---

## Potential Next Research Directions (post-Round 27)

### If SDF fix (#941 edward) shows vol_p improvement
1. **Re-run ensemble with clean-SDF checkpoint**: if edward's retrain improves test_vol_p significantly, rebuild greedy ensemble
2. **Re-run OOD aug experiments with clean SDF**: mirror aug, rotation aug may have been trained on corrupted data — retest strongest aug variant

### If SDF vol input feature (#966) or SDF-FiLM (#967) wins
3. **Compose SDF input feature + SDF-FiLM**: orthogonal conditioning mechanisms — input-level vs. hidden-state-level SDF signal
4. **Multi-channel SDF conditioning**: use both SDF scalar and xyz coordinates as input features; test whether coordinate info adds beyond SDF alone

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
- PR #955 alphonse: slice-centroid local RoPE — CLOSED NEGATIVE (EP3 gate: abupt=8.1833%, vol_p=5.9055%)
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
- **Y-flip augmentation fixed p=0.5**: #957 frieren CLOSED NEGATIVE (no EP1 improvement)
- **Y-flip augmentation curriculum ramp 0→0.5**: #962 frieren CLOSED NEGATIVE (EP3 vol_p gate: 5.9499% projected 5.924% vs ≤5.0%). frieren re-assigned #971 xattn-distance-rpe.
- **Mild yaw-only rotation aug**: #937 alphonse CLOSED (Round 24)
- **GradNorm full-mode α=1.5** (#942 fern): CLOSED NEGATIVE — vp weight decreased at EP1, all weights converge to 0.91–1.11 band; vol_p is not gradient-starved under Lion; 5× backward overhead consumed budget before EP6 gate
- **GradNorm ema_proxy, EMA, LoRA, FiLM, 2-layer MLP decoder, Huber loss, Lion β2 sweep, BC-type embedding, dual-tower xattn, spectral-norm attn, stochastic depth, PCGrad, surface curvature features**: ALL CLOSED
- **SDF-modulated vol PE (identity-init)**: #935 tanjiro CLOSED; superseded by coord-conditioned vol head #959
- **Surface-points curriculum**: #949 thorfinn CLOSED; superseded by SDF-zone masking #953
- **Learned geometry Q-bias (static)**: #950 alphonse CLOSED
- **CoordVolHead (raw xyz at output MLP)**: #959 tanjiro CLOSED NEGATIVE — EP4 val_abupt=7.428% (gate 6.9%); redundant with STRING-sep RoPE + surf→vol xattn backbone xyz encoding
- **SDF-zone vol token masking (p_max=0.30 and p_max=0.15)**: #953, #963 thorfinn FULLY CLOSED — high run-to-run variance (±4.2pp at p_max=0.30), no replicable signal above variance floor at p_max=0.15; near-surface dropout disrupts backbone globally
- **Raw SDF linear normalization as vol input feature** (#966 tanjiro): CLOSED NEGATIVE — EP4 val_abupt=7.6359% (gate ≤6.9%). `(sdf−μ)/σ` with σ=1.681517 m creates +20,000σ outliers (SDF max ~36,000 m) that dominate the vol input projection; raw linear normalization for SDF vol features is a closed axis. Bounded transforms (tanh/asinh) still live as #973.

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
- **CoordVolHead (#959) CLOSED NEGATIVE**: Failed EP4 gate (val_abupt=7.428% vs gate 6.9%). Root cause: STRING-sep RoPE + surf→vol xattn already encode xyz-awareness throughout the backbone; raw-coord injection at the output head is redundant. Xyz conditioning at the output is a closed axis.
- **SDF-zone masking fully CLOSED** (#953, #963): p_max=0.30 had ±4.2pp run-to-run variance; p_max=0.15 relaunch (#963) abandoned — no replicable positive signal above variance floor (~2.26pp). Near-surface token dropout is a dead end at current scale.
- **Geometry-conditioned Q-bias** (#961 fern, CLOSED NEGATIVE): mean-pool surf → MLP(512→256→512, zero-init) → additive bias on vol Q-projections before xattn. EP1=30.197% (0.197pp over 30% gate); natural EP1 variance ±2.75pp makes this uninterpretable but expected value is low. Global mean-pool surf loses spatial specificity needed to close OOD vol_p gap.
- **SDF-FiLM vol hidden-state conditioning** (#967 thorfinn, CLOSED NEGATIVE): FiLM modulation (γ, β) of vol hidden states after each TransolverBlock using SDF distance as conditioning signal. EP3 gate FAILED — thorfinn superseded by SDF-conditioned attention temperature #974.
- **10 REQUIRED_RESTORED cases have corrupted SDF data** (#941): inside-body cells had SDF=0 or negative; synthesis fix applied. Run_133, 226, 203, 158 (OOD outliers) are in this set — SDF corruption may explain part of the vol_p OOD gap.
