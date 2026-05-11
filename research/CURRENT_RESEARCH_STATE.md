# SENPAI Research State
- **Date:** 2026-05-09 — Round 27 active (8 WIP, 0 idle). STRING fully exhausted. SDF data fix (edward #941) CLOSED NEGATIVE — test_vol_p=11.9618% (worse than baseline 11.6704%); SDF corruption fix did not help OOD vol_p. Issue #803 blocker dissolved. frieren #971 (xattn-distance-rpe) CLOSED — SM 12.0 OOM on [B=4, H=4, N_vol=16384, N_surf=65536] attention tensor. Frieren assigned #981 (SDF-stratified vol sampling); edward assigned #980 (per-case bbox geometry conditioning). Thorfinn #974 (SDF-conditioned xattn temperature MLP) CLOSED NEGATIVE — MLP collapsed to global scale=0.515, EP3 dual gate miss. Thorfinn assigned #984 (constant xattn temp scale=0.5 zero-param control).
- **Advisor branch:** `tay`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current Baselines

| Tier | val_abupt | test_abupt | test_vol_pressure | PR | Notes |
|---|---|---|---|---|---|
| **Ensemble SOTA** | **6.0289%** | **7.3693%** | — | #880 (nezuko) | K=6 greedy pool-32 |
| **Single-model SOTA** | **6.2869%** | — | 12.0063% (NEGATIVE) | #958 Arm A (nezuko) | vol aux decoder head, EP13, run `29nohj67`, best_epoch=13 |
| **Prior single-model SOTA** | 6.4407% | 7.6992% | 11.6704% | #823 (nezuko) | surf→vol xattn, EP13, run `ghh0s4ne` |
| **Vol-pressure best anchor** | — | — | 11.374% | #681 (dc031qpt) | Issue #717 reference |

**Key win from #958 Arm A (nezuko):** vol aux decoder head (`--volume-loss-weight 1.0`) achieves val_abupt=6.2869% at best_epoch=13 (+0.154pp improvement over prior SOTA 6.4407%). test_vol_p=12.0063% is NEGATIVE (worse than 11.6704% baseline) — the dedicated aux head improves val_abupt generally but does NOT fix the 4 OOD test vol_p cases.

**Key win from #823:** surf→vol xattn adds one `nn.MultiheadAttention` (embed_dim=512, num_heads=4) after post-backbone LayerNorm. Q=vol_hidden, K=V=surf_hidden, zero-init out_proj. Improvement is broad-based (~−2.4% relative val, −3.6% test). The OOD test/val ratio is UNCHANGED (3.027× vs 3.025×) — xattn is a general capacity boost, NOT a targeted OOD fix. The 4 outlier test cases remain the dominant vol_pressure failure mode.

**Key finding from #767:** 4 OOD test cases (run_133, run_226, run_203, run_158) account for 92% of squared test_vol_p deviation. Excluding them, test_vol_p = 3.9–4.2% (below AB-UPT 6.08%). Geometry conditioning of the volume decoder is the highest-priority intervention.

**Vol-pressure promotion ladder (test_vol_pressure target):**
- Weak: ≤11.0% | Solid: ≤10.0% | Major: ≤8.5% | Target: ≤6.08% (AB-UPT reference)

---

## Latest Human Researcher Directives

- **Issue #618** (STRING/RoPE): **FULLY CLOSED — all axes exhausted.** Final closed experiments: #960 (askeladd σ-bracket sweep NEGATIVE) and #970 (alphonse frozen-freq ablation NEGATIVE). All STRING dimensions — sigma shifts, ladder variants, 7-octave mega-sweep, slice-centroid RoPE, frozen frequencies — have been tested and closed. STRING positional encoding is confirmed at its optimum: trainable `log_freq` with octave init {0.25,0.5,1.0,2.0,4.0}, trainable `phase`, QK-norm enabled. No new STRING experiments.
- **Issue #717** (vol_pressure gap): Phase 2 active — building on xattn win from #823. Post-xattn capacity fully closed (0-for-3: #891 FFN, #893 k-NN graph, #906 vol-self-attn). Now exploring loss stratification, geometry-conditioned queries, surface loss annealing, vol-pressure aux head, SDF-modulated vol PE, SDF-conditioned attention temperature, bounded SDF vol input encoding, per-case geometry embedding, surface-head-only isolation.
- **Issue #803** (SDF fix): **DISSOLVED.** Edward (#941) CLOSED NEGATIVE — full SOTA retrain with fixed SDF data (W&B: `2ub8dmy7`) completed to EP13: test_vol_p=11.9618% (worse than baseline 11.6704%). The 4 OOD outlier cases (run_133, 226, 203, 158) remain dominant failure mode even with corrected SDF values. SDF corruption is not the root cause of the vol_p OOD gap.
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- No new directives pending.

---

## Active PRs (Round 27)

### tay-track (8 WIP, 0 idle)

| Student | PR | Hypothesis | Branch | Status |
|---|---|---|---|---|
| alphonse | #976 | Per-case geometry embedding: mean-pool masked surface hidden states (post-backbone, post-norm) → geometry code vector [B, n_hidden]; project via zero-init `nn.Linear(n_hidden, n_hidden)`; add to volume hidden states before surf→vol xattn. Zero-init preserves baseline behavior at epoch 0. Generalizes to OOD geometries — no lookup table. Flag: `--use-geo-embed`. 4-ep screen; EP3 gate val_abupt ≤8.0% AND vol_p ≤5.0%; EP4 ≤6.9%. `--wandb-group alphonse-geo-embed` | `alphonse/per-case-geometry-embedding` | WIP — run `6vzsw2ec` launched |
| askeladd | #975 | Surface-head-only isolation: ablate surf→vol xattn, keep only the surface Transolver head as output for surface-only predictions. Tests whether the surface head alone can match the full model's surface metrics, isolating the contribution of xattn. 4-ep screen; EP3 gate val_abupt ≤8.0% AND vol_p ≤5.0%. `--wandb-group askeladd-surface-head-only` | `askeladd/surface-head-only-isolation` | WIP — run `sdjkbztj` launched |
| nezuko | #958 | Dedicated vol_p aux decoder head: independent 3-layer MLP branch (Linear(512,256)→SiLU→Linear(256,64)→SiLU→Linear(64,1)) separate from the shared 4-output surface head. Arm A: COMPLETE — new SOTA val_abupt=6.2869%, test_vol_p NEGATIVE. Arm B: `--volume-loss-weight 2.0`, run `6xja19q9`, in progress. | `nezuko/vol-pressure-aux-decoder-head` | WIP — Arm A DONE (SOTA); Arm B running |
| thorfinn | #984 | Constant xattn temp scale=0.5 control (zero-param): hardcode `scale=0.5` on Q (volume_hidden) before surf→vol xattn. Zero learnable params. Isolates global attention sharpening signal from per-token boundary locality hypothesis — PR #974 MLP collapsed to scale_mean=0.515 (τ_std=0.00177, 3 orders below global drift). Plain float multiply, no gradient, identity at scale=1.0. Flag: `--xattn-temp-scale 0.5`. EP3 dual gate val_abupt ≤8.0% AND vol_p ≤5.0%. | `thorfinn/constant-attn-temp-scale` | WIP — assigned, awaiting run launch |
| fern | #982 | No-SDF-PE-scaling control (fast schedule): ablation of SDF-PE spectral scaling to verify baseline under fast curriculum `0:16384:1:32768:2:49152:3:65536`. | `fern/no-sdf-pe-scaling-control` | WIP — assigned, awaiting run launch |
| tanjiro | #983 | Curriculum warmup-ramp: implement `--vol-points-warmup-steps` to smoothly ramp vol points per epoch rather than step jumps. | `tanjiro/curriculum-warmup-ramp` | WIP — assigned, awaiting run launch |
| frieren | #981 | SDF-stratified vol sampling: sample vol tokens proportionally to proximity to OOD-4 surface zones rather than uniform; near-surface tokens upweighted during vol curriculum. | `frieren/sdf-stratified-vol-sampling` | WIP — assigned, awaiting run launch |
| edward | #980 | Per-case bbox geometry conditioning: extract per-case macroscopic shape statistics (frontal area, aspect ratio, height/width statistics) from surface point cloud; concat as extra conditioning to vol decoder. | `edward/per-case-bbox-geometry-conditioning` | WIP — assigned, awaiting run launch |

### IDLE students (need new assignments)
- **frieren**: PR #971 (xattn-distance-rpe) CLOSED — SM 12.0 OOM on [B=4, H=4, N_vol=16384, N_surf=65536] attention tensor materialization (~32 GiB bf16). Needs new tay assignment.
- **edward**: PR #941 (SDF data fix) CLOSED NEGATIVE — test_vol_p=11.9618% worse than baseline. Needs new tay assignment.

---

## Round 27 Closures

| Student | PR | Result |
|---|---|---|
| frieren | #971 | Learned distance RPB on surf→vol xattn — CLOSED. SM 12.0 additive attention bias OOM: [B=4, H=4, N_vol=16384, N_surf=65536] bf16 materialization ~32 GiB. Flash SDPA avoids materialization but doesn't support custom additive bias per-pair without memory-inefficient workaround. |
| edward | #941 | SDF data fix retrain — CLOSED NEGATIVE. EP13 test_vol_p=11.9618% (gate: ≤11.6704% baseline). Full SOTA retrain with corrected SDF (W&B: `2ub8dmy7`) failed to improve OOD vol_p. SDF corruption is not the root cause of the vol_p OOD gap; Issue #803 blocker dissolved. |
| thorfinn | #967 | SDF-FiLM vol conditioning — CLOSED NEGATIVE. EP3 val_abupt gate FAILED. FiLM modulation (γ, β) of vol hidden states after each TransolverBlock using SDF distance did not help. Superseded by SDF-conditioned attention temperature #974. |
| thorfinn | #974 | SDF-conditioned xattn temperature MLP — CLOSED NEGATIVE. MLP (1→16→1, 33 params) collapsed to global constant: scale_mean=0.515, τ_std=0.00177 (3 orders of magnitude below global drift). EP3: val_abupt=8.148%, vol_p=5.350% — narrowly missed dual gate (miss by 0.148pp and 0.350pp). MLP was NOT learning per-token boundary differentiation; it discovered global attention sharpening (scale≈0.5). Zero-param control #984 assigned to isolate this signal. W&B run `vcs1ymnx`. |
| fern | #968 | SDF-PE octave scaling (original schedule) — SUPERSEDED by #977 (fast schedule). |
| tanjiro | #973 Arm A | tanh bounded SDF encoding — KILLED. tanh saturation degeneracy: scale=2.0m → ~99% vol tokens saturate tanh≈1.0 (gradient≈1e-8). Catastrophic loss spike at 16K→32K vol curriculum transition (EP2 step 3 loss=10.6274). Arm B (asinh) continues. |

---

## Round 26 Outcomes (Closed / Superseded)

| Student | PR | Result |
|---|---|---|
| alphonse | #970 | STRING-sep frozen-freq ablation — CLOSED NEGATIVE. EP3=8.3347% (gate ≤8.0%, missed by 0.33pp). Trainable `log_freq` confirmed load-bearing; frequency adaptation via gradient descent is genuinely beneficial. New assignment: #976 per-case geometry embedding. |
| askeladd | #960 | STRING σ-bracket sweep — CLOSED NEGATIVE. Arm A (fine-shift [0.01,0.25,0.5,1.0,2.0]) EP3=7.1812%; Arm B v2 (baseline [0.25,0.5,1.0,2.0,4.0]) EP4=7.4995%. Neither bracket improved over SOTA. σ-tuning axis exhausted; STRING fully closed. New assignment: #975 surface-head-only isolation. |
| askeladd | #956 | STRING σ-ladder geometry sweep (7-octave fine/coarse/all-fine) — CLOSED NEGATIVE. Arm A (7-oct fine, 0.03–2.0) failed EP1 at 30.27%; too wide a range degrades early convergence. Superseded by bracket sweep #960 |
| thorfinn | #949 | Surface-points curriculum — CLOSED. Results ambiguous; superseded by SDF-zone masking #953 |
| tanjiro | #966 | SDF scalar vol input feature (raw linear norm) — CLOSED NEGATIVE. EP4 val_abupt=7.6359% (gate ≤6.9%, delta +1.20pp vs SOTA). Root cause: `(sdf−μ)/σ` normalization — train-set SDF std=1.681517 m, max ~36,000 m → +20,000σ outliers dominated vol input projection. Bounded encoding follow-up: #973 (tanh/asinh, scale=2.0 m). |
| tanjiro | #935 | SDF-modulated vol PE (per-octave scaler, identity-init retry) — CLOSED. Superseded by coordinate-conditioned vol head #959 |
| alphonse | #955 | STRING RoPE slice-centroid — CLOSED NEGATIVE. EP3 gate failed: val_abupt=8.1833% (gate ≤8.0%), vol_p=5.9055% (gate ≤5.0%). Local centroid coords did not improve over global. New assignment: #970 frozen-freq ablation. |
| frieren | #957 | Y-flip augmentation (fixed p=0.5) — CLOSED NEGATIVE. No improvement at EP1; curriculum ramp version assigned as #962. |
| nezuko | #958 | Dedicated vol_p aux decoder head — CARRIED OVER / PARTIALLY COMPLETE (Arm A DONE, new SOTA; Arm B continuing) |
| thorfinn | #953 | SDF-zone vol masking p_max=0.15 — CLOSED (false-kill; `train/loss<5` threshold fired on transient augmentation spike, not genuine divergence). Relaunched as #963 with relaxed `<10` threshold. |
| fern | #952 | Manifold Mixup — CLOSED NEGATIVE (Round 27); reassigned #961 geom-q-bias |
| edward | #941 | SDF data fix — CLOSED NEGATIVE (see Round 27 Closures above) |

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

### Theme 1: SDF Data Quality — CLOSED NEGATIVE
edward #941 CLOSED: test_vol_p=11.9618% worse than baseline 11.6704%. SDF corruption is NOT the root cause of the OOD vol_p gap. The 4 outlier test cases (run_133, 226, 203, 158) remain dominant failure mode regardless of SDF data quality. Issue #803 dissolved.

### Theme 2: STRING Positional Encoding — FULLY CLOSED
All STRING axes have now been tested and closed:
- #956 askeladd: 7-octave mega-sweep (0.0625–8.0) — CLOSED NEGATIVE (EP1=30.27%)
- #955 alphonse: slice-centroid local RoPE — CLOSED NEGATIVE (EP3 gate: abupt=8.1833%, vol_p=5.9055%)
- #970 alphonse: frozen-freq ablation — CLOSED NEGATIVE (EP3=8.3347%, gate ≤8.0%). Trainable `log_freq` confirmed load-bearing.
- #960 askeladd: sigma-bracket sweep — CLOSED NEGATIVE (Arm A EP3=7.1812%; Arm B v2 EP4=7.4995%). σ-tuning axis exhausted.
- Prior axes: sigma shift, all-fine/all-coarse ladder variants, vol-specific RFF (#918) — ALL CLOSED.

**STRING is fully exhausted. No new STRING experiments.**

### Theme 2b: Geometry Conditioning (alphonse #976, askeladd #975)
Replacing the closed STRING theme with two geometry-conditioning approaches:
- **Per-case geometry embedding** (#976 alphonse): mean-pool masked surface hidden states (post-backbone) → geometry code vector; zero-init projection; add to vol hidden states before surf→vol xattn. Preserves baseline at init. Generalizes to OOD geometries without lookup table. Run `6vzsw2ec` in flight.
- **Surface-head-only isolation** (#975 askeladd): ablate surf→vol xattn, keep only surface Transolver head for surface-only predictions. Diagnostic ablation to quantify xattn's contribution to surface metrics. Run `sdjkbztj` in flight.

### Theme 3: Data Augmentation for OOD — FULLY CLOSED
Both y-flip augmentation approaches have been tested and failed:
- **PR #957** (fixed p=0.5 from EP0): CLOSED NEGATIVE — no val improvement at EP1, higher vol_p
- **PR #962** (curriculum ramp 0→0.5 over 3ep): CLOSED NEGATIVE — EP3 vol_p gate FAIL (5.9499% projected 5.924% vs ≤5.0%). Curriculum ramp slows vol_p convergence; insufficient headroom to pass gate.
y-flip augmentation axis is FULLY CLOSED.

### Theme 4: Dedicated Vol Pressure Decoder (nezuko #958)
Independent 3-layer MLP branch for vol_p prediction, separate from the shared 4-output surface head. Hypothesis: the shared decoder must encode too many competing signals; a dedicated path lets it specialize on vol_p geometry.
- **Arm A** (vol_loss_weight=1.0): COMPLETE — new single-model SOTA val_abupt=6.2869% (run `29nohj67`, best_epoch=13). test_vol_p=12.0063% NEGATIVE (worse than baseline). val_abupt improvement is genuine and broad-based; test_vol_p gap persists — OOD problem is not addressed by the aux head alone.
- **Arm B** (vol_loss_weight=2.0): run `6xja19q9`, EP1 PASS (29.43%), EP2 strong, EP3 gate ETA ~04:30Z. Hypothesis: higher vol loss weight may help the OOD cases more.

### Theme 5: Bounded SDF Encoding for Vol-Token Input (tanjiro #973)
- **Arm A (tanh)**: KILLED — tanh saturation degeneracy at scale=2.0m (gradient≈1e-8, catastrophic EP2 step 3 loss=10.6274). Well-understood mechanistic failure.
- **Arm B (asinh)**: run `lj2hsiui` launched 02:19Z. asinh avoids saturation via graded log compression (no hard ceiling). EP1 ETA ~03:54Z, gate ≤30%.

### Theme 6: SDF-Conditioned Attention Temperature (thorfinn #984)
SDF-FiLM (#967) CLOSED NEGATIVE. SDF-conditioned xattn temperature MLP (#974) CLOSED NEGATIVE — MLP collapsed to global constant scale_mean=0.515, narrowly missed EP3 dual gate (val_abupt=8.148%, vol_p=5.350%). Key finding: MLP was not learning per-token differentiation; it was discovering that global attention sharpening (scale≈0.5) is beneficial. Follow-up: #984 (thorfinn) hardcodes `scale=0.5` with zero learnable parameters — plain float multiply on Q before `nn.MultiheadAttention`. Cleanly isolates global sharpening from per-token locality hypothesis. Flag: `--xattn-temp-scale 0.5`. Awaiting run launch.

### Theme 7: SDF-Modulated Vol PE via Per-Octave Spectral Scaling (fern #977)
Supersedes #968 (original schedule) with accelerated vol-curriculum. SDF-conditioned spectral emphasis of STRING-sep RFF frequency groups. Tiny MLP (Linear(1→16)→SiLU→Linear(16→5)→Softplus). EP1 PASS (27.355%, run `q1tw64wi`). EP2 gate ≤16%, ETA ~02:50Z.

### Theme 8: Learned Distance RPB on surf→vol xattn — CLOSED
frieren #971 CLOSED — SM 12.0 OOM. [B=4, H=4, N_vol=16384, N_surf=65536] bf16 attention tensor materialization ~32 GiB. Flash SDPA supports custom bias only via block-sparse masks or scaled_dot_product_attention workarounds that lose precision or add overhead. This approach requires a memory-efficient reformulation to be viable.

---

## Potential Next Research Directions (post-Round 27)

### If SDF bounded encoding (#973 Arm B asinh) wins
1. **Compose asinh SDF + SDF-conditioned attn temperature**: orthogonal input-level vs. attention-level SDF conditioning
2. **Scale sweep**: test asinh scale in {1.0, 2.0, 4.0} m to find optimal near-surface resolution tradeoff

### If geometry embedding (#976) wins
3. **Deeper geometry code**: 2-layer geometry encoder (512→256→512) with residual connection
4. **Multi-scale geometry pooling**: pool at multiple layers (early, mid, late backbone) and combine

### If SDF attn temperature (#974) wins
5. **Compose with bounded SDF input feature**: temperature + asinh encoding are orthogonal mechanisms
6. **Learnable temperature range**: make the 0.5 offset and 1.0 scale range learnable parameters

### If nezuko Arm B (vol_loss=2.0) beats Arm A on vol_p
7. **Vol loss weight sweep**: {1.5, 2.0, 3.0} with the aux head on top of the new SOTA checkpoint
8. **Per-case vol loss weighting**: upweight OOD-4 cases (run_133, 226, 203, 158) by 2× in vol aux head loss

### Radical escalation (if plateau continues past Round 28)
9. **GNN vol pathway**: replace Transolver vol attention with k-NN GNN for vol-vol communication (k=8-16)
10. **Fourier Neural Operator vol decoder**: replace MLP vol decoder with FNO for better frequency resolution
11. **Attention-free SSM decoder for vol**: S4/Mamba-based sequence model for vol token decoding
12. **Point Transformer V3 vol head**: replace Transolver vol attention with PTv3 (Hilbert-curve serialized attention)

### Closed axes — do not revisit
See "Closed Axes" section below.

---

## Closed Axes (Rounds 22–27)

### SDF data quality: CLOSED NEGATIVE
- **#941 edward**: SDF data fix — test_vol_p=11.9618% (worse than 11.6704% baseline). SDF corruption is NOT the root cause.

### Post-xattn capacity: FULLY CLOSED (0-for-3)
- PR #891 (fern post-xattn FFN): EP3=8.50% NEGATIVE
- PR #893 (askeladd vol k-NN graph attn): NEGATIVE
- PR #906 (edward post-xattn vol self-attn): EP3=8.23% NEGATIVE

### Pre-xattn capacity: CLOSED (0-for-1)
- PR #929 (edward pre-xattn vol self-attn): CLOSED NEGATIVE

### Learned distance RPB on xattn: CLOSED (OOM)
- PR #971 (frieren xattn-distance-rpe): CLOSED — SM 12.0 OOM on full attention tensor; needs memory-efficient reformulation.

### SDF-FiLM hidden-state conditioning: CLOSED NEGATIVE
- PR #967 (thorfinn): FiLM modulation (γ, β) of vol hidden states after each TransolverBlock — EP3 gate FAILED.

### STRING axes closed — FULLY EXHAUSTED
- PR #956 askeladd: 7-octave mega-sweep (0.0625–8.0) — CLOSED NEGATIVE (EP1=30.27%)
- PR #955 alphonse: slice-centroid local RoPE — CLOSED NEGATIVE (EP3 gate: abupt=8.1833%, vol_p=5.9055%)
- PR #970 alphonse: frozen-freq ablation — CLOSED NEGATIVE (EP3=8.3347%, gate ≤8.0%). Trainable `log_freq` confirmed load-bearing; frequency adaptation is genuinely beneficial.
- PR #960 askeladd: sigma-bracket sweep (fine-shift [0.01,0.25,0.5,1.0,2.0] Arm A; baseline [0.25,0.5,1.0,2.0,4.0] Arm B v2) — CLOSED NEGATIVE (Arm A EP3=7.1812%; Arm B v2 EP4=7.4995%). σ-tuning axis exhausted.
- Prior STRING axes: sigma shift, all-fine/all-coarse ladder variants, vol-specific RFF (#918) — ALL CLOSED
- σ=0.25 confirmed load-bearing (#819); σ-shift/ladder failed
- **No further STRING experiments permitted.**

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
- **Geometry-conditioned Q-bias** (#961 fern, CLOSED NEGATIVE): mean-pool surf → MLP(512→256→512, zero-init) → additive bias on vol Q-projections before xattn. EP1=30.197% (0.197pp over 30% gate); global mean-pool surf loses spatial specificity.
- **tanh bounded SDF encoding** (#973 tanjiro Arm A): KILLED — saturation degeneracy at scale=2.0m (~99% tokens saturate, gradient≈1e-8), catastrophic curriculum-transition loss spike.

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
- **SDF data quality is NOT the root cause** (#941 edward): Full SOTA retrain with corrected SDF (fixed inside-body zero/negative SDF values) resulted in test_vol_p=11.9618% — WORSE than baseline 11.6704%. The OOD vol_p gap is a model capacity/generalization problem, not a data quality artifact.
- **Post-xattn capacity is a dead end** (0-for-3: #891, #893, #906): all capacity added AFTER surf→vol xattn fails to improve. The bottleneck is in the xattn query quality or the K/V representation, not post-xattn processing.
- **Pre-xattn vol self-attention also failed** (#929): symmetry does not hold; pre-xattn capacity not useful either.
- **STRING σ=0.25 is load-bearing** (#819): σ-shift/ladder failed. Encodes panel-scale surface detail critical for L5/4-ep budget.
- **K/V gradient backflow is load-bearing** (#890): detach-K/V failed. Two-layer xattn also killed by backflow. Full MHA gradient is necessary.
- **Full MHA n_heads=4 is optimal** (#893): both GQA (n_kv_heads=2) and MQA (n_kv_heads=1) fail EP3 gate.
- **GradNorm full-mode CLOSED NEGATIVE** (#942): vp weight goes DOWN (not up) because vp loss decreases fastest; all 5 weights converge to 0.91–1.11 band at EP1. GradNorm axis fully closed.
- **SDF-zone masking at p_max=0.30 is unstable** (#953): two identical runs produced EP1 val_abupt=28.89% and 32.84% (±4.2pp). Stochastic BL masking disrupts whole backbone globally, not just vol_p channel. p_max=0.15 also closed (no replicable signal).
- **CoordVolHead (#959) CLOSED NEGATIVE**: Failed EP4 gate (val_abupt=7.428% vs gate 6.9%). Root cause: STRING-sep RoPE + surf→vol xattn already encode xyz-awareness throughout the backbone; raw-coord injection at the output head is redundant. Xyz conditioning at the output is a closed axis.
- **Geometry-conditioned Q-bias** (#961 fern, CLOSED NEGATIVE): mean-pool surf → MLP(512→256→512, zero-init) → additive bias on vol Q-projections before xattn. EP1=30.197% (0.197pp over 30% gate); natural EP1 variance ±2.75pp makes this uninterpretable but expected value is low. Global mean-pool surf loses spatial specificity needed to close OOD vol_p gap.
- **SDF-FiLM vol hidden-state conditioning** (#967 thorfinn, CLOSED NEGATIVE): FiLM modulation (γ, β) of vol hidden states after each TransolverBlock using SDF distance as conditioning signal. EP3 gate FAILED — thorfinn superseded by SDF-conditioned attention temperature #974.
- **tanh saturation degeneracy** (#973 tanjiro Arm A): at scale=2.0m, ~99% of vol tokens saturate tanh≈1.0 (gradient≈1e-8). Causes catastrophic loss spike at curriculum transition (16K→32K vol tokens). asinh avoids this via graded log compression.
- **Vol aux decoder head improves val_abupt but not test_vol_p** (#958 nezuko Arm A): new single-model SOTA val_abupt=6.2869%, but test_vol_p=12.0063% is worse than baseline 11.6704%. The OOD vol_p gap requires geometry-specific conditioning beyond a dedicated decoder head.
- **Flash SDPA required for large xattn tensors**: [B=4, H=4, N_vol=16384, N_surf=65536] bf16 materialization = ~32 GiB — OOM without Flash SDPA. Thorfinn confirmed and fixed this. Custom per-token attention bias (distance RPB, temperature) requires manual Q/K/V extraction from in_proj_weight to apply Flash SDPA per-token correctly.
