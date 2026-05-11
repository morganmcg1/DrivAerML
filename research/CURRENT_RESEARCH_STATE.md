# SENPAI Research State

- **Date:** 2026-05-09 — Round 28 active (8 WIP; PRs #985, #986 newly assigned)
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

- **Issue #618** (STRING/RoPE): **FULLY CLOSED — all axes exhausted.** STRING positional encoding is confirmed at its optimum: trainable `log_freq` with octave init {0.25,0.5,1.0,2.0,4.0}, trainable `phase`, QK-norm enabled. No new STRING experiments.
- **Issue #717** (vol_pressure gap): Phase 2 active — building on xattn win from #823. Post-xattn capacity fully closed (0-for-3: #891 FFN, #893 k-NN graph, #906 vol-self-attn). Now exploring loss stratification, geometry-conditioned queries, surface loss annealing, vol-pressure aux head, SDF-modulated vol PE, SDF-conditioned attention temperature, bounded SDF vol input encoding, per-case geometry embedding, surface-head-only isolation.
- **Issue #803** (SDF fix): **DISSOLVED.** Edward (#941) CLOSED NEGATIVE — full SOTA retrain with fixed SDF data (W&B: `2ub8dmy7`) completed to EP13: test_vol_p=11.9618% (worse than baseline 11.6704%). The 4 OOD outlier cases remain dominant failure mode even with corrected SDF values.
- **Issue #759** (Bengio backlog): Reserved for future experiment ideas.
- No new directives pending.

---

## Active PRs (Round 28) — 8 WIP

| Student | PR | Hypothesis | Branch | Status |
|---|---|---|---|---|
| nezuko | #958 | Vol aux decoder head: Arm A COMPLETE (new SOTA 6.2869%). Arm B (`--volume-loss-weight 2.0`) running EP4. | `nezuko/vol-pressure-aux-decoder-head` | WIP — Arm B run `6xja19q9`, EP4 running; gate EP7 ≤7.2% |
| frieren | #981 | SDF-stratified vol sampling, alpha=2.0: oversample near-boundary vol points using SDF proximity. | `frieren/sdf-stratified-vol-sampling` | WIP — restart run `xzreqhns` (`frieren/sdf-stratified-vol-ep4-alpha2-restart`) confirmed running, EP1 expected ~07:16Z |
| fern | #982 | No-SDF-PE-scaling control: ablation to isolate whether SDF-PE spectral scaling in PR #977 was responsible for EP3 failure. | `fern/no-sdf-pe-scaling-fast-schedule-control` | WIP — run `ua1ohtjb`, EP1 pending; group `fern-no-sdf-pe-scaling-control` |
| tanjiro | #983 | Curriculum warmup: `--vol-points-warmup-steps` to ramp vol points smoothly; tests whether 200–500 step warmup prevents EP1→EP2 divergence seen in #973. | `tanjiro/curriculum-warmup-ramp` | WIP — Arm A run `hhplwxmk` (500-step warmup), EP1→EP2 transition pending; Arm B pre-approved (200-step); group `tanjiro-curriculum-warmup` |
| thorfinn | #984 | Constant xattn temp scale=0.5 (zero-param): hardcode Q×0.5 before surf→vol xattn; isolates global sharpening signal from PR #974's MLP that collapsed to scale_mean=0.515. | `thorfinn/constant-attn-temp-scale` | WIP — run `ojuyombs`, EP1 pending; group `thorfinn-constant-attn-temp-scale` |
| edward | #980 | Bbox geometry scalar conditioning: per-case [L,H,W,A_front] → Linear(4,hidden_dim) zero-init additive bias on vol hidden. 10-ep schedule with GradNorm. | `edward/geo-scalar-cond` | WIP — run `nv4g1nhw`, 10-ep schedule+GradNorm; EP1 gate ≤30% |
| alphonse | #985 | Per-case geometry embedding v2: second xattn layer (vol Q × all surf K/V, zero-init out_proj residual) to spatially condition vol queries before main surf→vol xattn; ~1.05M new params. | `alphonse/geo-embed-v2-xattn` | WIP — newly assigned; awaiting student implementation |
| askeladd | #986 | Adaptive SDF vol loss weighting: exp(-\|d_sdf\|/sigma) per-token weight; sigma sweep {0.5, 1.0, 2.0} m. | `askeladd/adaptive-sdf-vol-loss` | WIP — newly assigned; awaiting student implementation |

### All students active (8 WIP)

---

## Round 27–28 Closures

| Student | PR | Result |
|---|---|---|
| dl24-tanjiro | #978 | Per-case bbox coordinate normalization — CLOSED NEGATIVE. EP5 val_abupt=8.4488% (gate ≤7.5%). STRING PE sigmas tuned for raw ~5m coords; bbox normalization to [-1,+1] causes ~5× frequency mismatch. Loader bug fix (empty-view guard) worth cherry-picking separately. |
| askeladd | #975 | Surface-head-only isolation — CLOSED NEGATIVE. EP3 gate failed. Surface head alone cannot match the full model; xattn is load-bearing for surface metrics. |
| alphonse | #976 | Per-case geometry embedding — CLOSED NEGATIVE. EP3=8.073% val_abupt + 5.301% vol_p, dual gate fail. Mean-pool surface hidden states → geometry code does not generalize OOD vol_p. |
| frieren | #971 | Learned distance RPB on surf→vol xattn — CLOSED. SM 12.0 OOM on [B=4, H=4, N_vol=16384, N_surf=65536] attention tensor ~32 GiB bf16. |
| edward | #941 | SDF data fix retrain — CLOSED NEGATIVE. EP13 test_vol_p=11.9618% (worse than 11.6704% baseline). SDF corruption is not the root cause. |
| thorfinn | #974 | SDF-conditioned xattn temperature MLP — CLOSED NEGATIVE. MLP collapsed to global constant scale_mean=0.515, τ_std=0.00177. EP3: val_abupt=8.148%, vol_p=5.350% — narrowly missed dual gate. Follow-up: #984. |
| thorfinn | #967 | SDF-FiLM vol conditioning — CLOSED NEGATIVE. EP3 val_abupt gate FAILED. |
| tanjiro | #973 Arm A | tanh bounded SDF encoding — KILLED. Saturation degeneracy at scale=2.0m (gradient≈1e-8), catastrophic EP2 transition loss=10.6274. |

---

## Round 26 Outcomes (Closed / Superseded)

| Student | PR | Result |
|---|---|---|
| alphonse | #970 | STRING-sep frozen-freq ablation — CLOSED NEGATIVE. EP3=8.3347% (gate ≤8.0%, missed 0.33pp). Trainable `log_freq` confirmed load-bearing. |
| askeladd | #960 | STRING σ-bracket sweep — CLOSED NEGATIVE. Arm A EP3=7.1812%; Arm B v2 EP4=7.4995%. σ-tuning axis exhausted; STRING fully closed. |
| askeladd | #956 | STRING σ-ladder geometry sweep (7-octave) — CLOSED NEGATIVE (EP1=30.27%). |
| thorfinn | #949 | Surface-points curriculum — CLOSED. Results ambiguous; superseded by #953. |
| tanjiro | #966 | SDF scalar vol input feature (raw linear norm) — CLOSED NEGATIVE. EP4 val_abupt=7.6359% (gate ≤6.9%). +20,000σ outliers dominated vol input projection. |
| alphonse | #955 | STRING RoPE slice-centroid — CLOSED NEGATIVE. EP3: abupt=8.1833%, vol_p=5.9055%. |
| fern | #968 | SDF-PE octave scaling (original schedule) — SUPERSEDED by #977 (fast schedule). |

---

## Current Research Focus (Round 28)

**Primary mandate (Issue #717):** Reduce test_vol_pressure below 11.0% → 10.0% → 8.5% → 6.08% (AB-UPT). All channels must not degrade.

### Theme 1: Dedicated Vol Pressure Decoder (nezuko #958)
Independent 3-layer MLP branch for vol_p prediction. Arm A (vol_loss_weight=1.0): COMPLETE — new single-model SOTA val_abupt=6.2869% (run `29nohj67`, best_epoch=13). test_vol_p=12.0063% NEGATIVE. Arm B (vol_loss_weight=2.0): run `6xja19q9`, EP4 running, EP7 gate ≤7.2%.

### Theme 2: SDF-Stratified Vol Sampling (frieren #981)
Oversample near-boundary vol tokens (SDF proximity, alpha=2.0) during training. Run `xzreqhns` (`frieren/sdf-stratified-vol-ep4-alpha2-restart`) confirmed running at step ~1,358, EP1 expected ~07:16Z. Group `frieren-sdf-stratified-vol`.

### Theme 3: No-SDF-PE-Scaling Control (fern #982)
Ablation of SDF-PE spectral scaling to verify whether PR #977's EP3 failure (val_abupt ≤8.0% miss) was caused by SDF-PE-scaling or other changes. Run `ua1ohtjb`, group `fern-no-sdf-pe-scaling-control`. EP1 gate ≤30%.

### Theme 4: Curriculum Warmup (tanjiro #983)
`--vol-points-warmup-steps` to smoothly ramp vol points per epoch rather than step jumps. Arm A (500-step warmup, run `hhplwxmk`): EP1→EP2 transition pending at step ~10,865 (critical window — PR #973 diverged here). Arm B (200-step warmup) pre-approved; launch after Arm A clears step 10,868 cleanly. Group `tanjiro-curriculum-warmup`.

### Theme 5: Constant xattn temp scale=0.5 (thorfinn #984)
Zero-param control for PR #974's MLP collapsed to global scale=0.515. Hardcode Q×0.5 before surf→vol xattn. Run `ojuyombs`, group `thorfinn-constant-attn-temp-scale`. EP1 gate ≤30%.

### Theme 6: Bbox Geometry Scalar Conditioning (edward #980)
Per-case geometry embedding via scalar features [L,H,W,A_front] projected to hidden_dim and added as additive bias to vol hidden states. Zero-init Linear(4,hidden_dim) for safe warm-up. Run `nv4g1nhw`, 10-ep schedule with GradNorm. EP1 gate ≤30%.

### Theme 7: Per-Case Geometry Embedding v2 (alphonse #985)
Second cross-attention layer (vol Q × all surface K/V, zero-init out_proj residual) before main surf→vol xattn. Spatially conditions vol queries on all surface tokens, overcoming the mean-pool spatial specificity loss that killed #976. ~1.05M new params. Newly assigned — awaiting student implementation.

### Theme 8: Adaptive SDF Vol Loss Weighting (askeladd #986)
Weight vol loss per-token by exp(-|d_sdf|/sigma) to focus learning near boundaries where OOD geometry differences are most pronounced. Sigma sweep {0.5, 1.0, 2.0} m. Parameter-free. Newly assigned — awaiting student implementation.

### Closed Themes
- **SDF data quality** (edward #941): CLOSED NEGATIVE — SDF corruption is NOT the root cause of vol_p OOD gap.
- **STRING positional encoding** (#960, #970 etc.): FULLY EXHAUSTED — all axes closed. No new STRING experiments.
- **Per-case geometry embedding v1** (alphonse #976): CLOSED NEGATIVE — mean-pool surface code does not generalize OOD vol_p.
- **Surface-head-only isolation** (askeladd #975): CLOSED NEGATIVE — xattn is load-bearing for surface metrics.

---

## Potential Next Research Directions

### Immediate Follow-ups (high priority)
1. **Attention temperature sweep**: If PR #984 passes EP3, run scale=0.25, 0.75 to find optimal sharpening factor
2. **Curriculum warmup tuning**: If PR #983 passes EP3, tune warmup length (100–1000 steps)
3. **SDF alpha sweep**: If PR #981 passes EP3, sweep alpha=1.0, 1.5, 3.0 for optimal near-boundary sampling density
4. **Bbox normalization with corrected PE sigmas**: PR #978 CLOSED NEGATIVE — coordinate rescale ~5× caused PE frequency mismatch. Follow-up: apply normalization AFTER PE (coordinates remain raw-scale for PE; normalize vol/surf coords only for Transolver slice attention). Or rescale STRING sigmas to match normalized coords: {0.05, 0.1, 0.2, 0.4, 0.8}.

### Architecture Directions
5. **Per-case geometry embedding v2**: ASSIGNED — alphonse #985 (cross-attn vol × all surf tokens, zero-init residual)
6. **Mid-backbone xattn**: Cross-attention at intermediate backbone layer rather than final; tested several variants (PR #917 NEGATIVE), but pre-final-layer variant may differ
7. **Adaptive vol loss weighting by SDF distance**: ASSIGNED — askeladd #986 (sigma sweep {0.5, 1.0, 2.0} m)

### Data / Representation Directions
8. **asinh bounded SDF encoding**: tanjiro #973 Arm B was in progress (asinh replaces tanh); if complete, review results
9. **OOD-4 case upweighting**: Upweight the 4 outlier test cases (run_133, 226, 203, 158) by 2× in the vol aux head loss; direct OOD targeting
10. **Coordinate frame augmentation**: At train time, randomly rotate/scale the canonical coordinate frame; forces the model to learn rotation-equivariant representations

### Bold / Escalation Directions (if plateau continues past Round 28)
11. **GNN vol pathway**: Replace Transolver vol attention with k-NN GNN for vol-vol communication (k=8-16)
12. **Fourier Neural Operator vol decoder**: Replace MLP vol decoder with FNO for better frequency resolution
13. **Implicit neural representation**: Map (surface_context, vol_coord) → (tau, pressure) via coordinate MLP; natural coordinate generalization
14. **Point Transformer V3 vol head**: Replace Transolver vol attention with PTv3 (Hilbert-curve serialized attention)

---

## Closed Axes (Rounds 22–27) — Do Not Revisit

### SDF data quality: CLOSED NEGATIVE
- **#941 edward**: SDF data fix — test_vol_p=11.9618% (worse than 11.6704% baseline). SDF corruption is NOT the root cause.

### Post-xattn capacity: FULLY CLOSED (0-for-3)
- PR #891 (fern post-xattn FFN): EP3=8.50% NEGATIVE
- PR #893 (askeladd vol k-NN graph attn): NEGATIVE
- PR #906 (edward post-xattn vol self-attn): EP3=8.23% NEGATIVE

### Pre-xattn capacity: CLOSED (0-for-1)
- PR #929 (edward pre-xattn vol self-attn): CLOSED NEGATIVE

### Learned distance RPB on xattn: CLOSED (OOM)
- PR #971 (frieren xattn-distance-rpe): CLOSED — SM 12.0 OOM; needs memory-efficient reformulation.

### SDF-FiLM hidden-state conditioning: CLOSED NEGATIVE
- PR #967 (thorfinn): FiLM modulation of vol hidden states — EP3 gate FAILED.

### STRING axes: FULLY EXHAUSTED
- PR #956, #960, #970, #955 — all CLOSED NEGATIVE. No further STRING experiments permitted.
- σ=0.25 confirmed load-bearing; trainable `log_freq` confirmed load-bearing.

### All other closed axes (do not revisit)
- **Depth scaling**: L=6 CLOSED (both #895 and #811 NEGATIVE)
- **Width scaling**: hidden=640 CLOSED (#886 NEGATIVE)
- **Two-layer xattn**: #884 CLOSED (EP1 kill gate)
- **Detach-K/V xattn**: #890 CLOSED (K/V gradient backflow load-bearing)
- **GQA/MQA xattn**: #893 CLOSED (full MHA n_heads=4 is optimal)
- **MLP ratio=2.0**: #897 CLOSED (ratio=4.0 is optimal)
- **Learned surf pool (Perceiver)**: #894 CLOSED
- **Pos-encoding bias on xattn queries (static)**: #883 CLOSED
- **OOD static loss upweighting**: #888 CLOSED
- **Vol loss upweighting (curriculum)**: #902 CLOSED (vol_w=3.0 degraded all channels)
- **Mid-backbone surf→vol xattn**: #917 CLOSED NEGATIVE
- **Rotation aug (aggressive and mild)**: #925, #937 CLOSED NEGATIVE
- **All augmentation axes** (geometric mixup, train y-mirror, TTA y-mirror, y-flip fixed, y-flip curriculum ramp): ALL CLOSED NEGATIVE
- **Surface cp Laplacian aux loss**: #927 CLOSED
- **Vol geo features centroid+bbox**: #926 CLOSED
- **Vol-specific RFF sigmas**: #918 CLOSED
- **GradNorm full-mode α=1.5**: #942 CLOSED NEGATIVE — vol weights go DOWN, all weights converge to 0.91–1.11 band
- **GradNorm, EMA, LoRA, FiLM, 2-layer MLP decoder, Huber loss, Lion β2 sweep, BC embedding, dual-tower xattn, spectral-norm attn, stochastic depth, PCGrad, surface curvature features**: ALL CLOSED
- **SDF-zone vol token masking (p_max=0.30 and 0.15)**: #953, #963 CLOSED — high run-to-run variance (±4.2pp), no replicable signal
- **CoordVolHead**: #959 CLOSED NEGATIVE — EP4 val_abupt=7.428% (gate 6.9%); redundant with STRING-sep + xattn
- **Geometry-conditioned Q-bias** (mean-pool): #961 CLOSED NEGATIVE — EP1=30.197%
- **Raw SDF linear normalization as vol input feature**: #966 CLOSED NEGATIVE — +20,000σ outliers; closed axis
- **tanh bounded SDF encoding**: #973 Arm A KILLED — saturation degeneracy at scale=2.0m
- **Per-case geometry embedding (mean-pool surface code)**: #976 CLOSED NEGATIVE — EP3 dual gate fail
- **Surface-head-only isolation**: #975 CLOSED NEGATIVE — xattn load-bearing for surface metrics

---

## Key Architecture Decisions Established

- **Model:** Transolver L=5, hidden=512, heads=4, slices=128, surf→vol xattn (1 layer, 4 heads, ~1.05M params) → ~16.99M total
- **Depth:** L=5 optimal. FULLY CLOSED.
- **Positional encoding:** STRING-separable (rff=16, sigmas {0.25,0.5,1.0,2.0,4.0}). σ=0.25 load-bearing. Trainable `log_freq` load-bearing.
- **Optimizer:** Lion, lr=9e-5, β₁=0.9, β₂=0.99, wd=5e-4
- **QK-norm:** enabled
- **Loss weights:** tau_y×1.5, tau_z×2.0, surface×2.0, volume×1.0. All static loss-weight axes CLOSED.
- **EMA:** 0.999
- **4-ep screen schedule:** `--lr-cosine-t-max 4 --epochs 4` with vol-curriculum `0:16384:1:32768:2:49152:3:65536`
- **`--no-compile-model` required:** torch.compile + DDP NCCL deadlock at step 1
- **`find_unused_parameters=True` required for DDP when using conditional modules**

## Key Diagnostic Findings Established

- **4 OOD test cases dominate vol_pressure** (#767): 92% of squared deviation. Excluding them, test_vol_p = 3.9–4.2% (below AB-UPT). Geometry conditioning is the right lever.
- **Surf→vol xattn is broad-spectrum win, not OOD fix** (#823): OOD test/val ratio unchanged (3.027×). The 4 outlier cases still drive vol_p gap.
- **SDF data quality is NOT the root cause** (#941): test_vol_p=11.9618% with corrected SDF — WORSE than baseline.
- **Post-xattn capacity is a dead end** (0-for-3: #891, #893, #906): all capacity added AFTER surf→vol xattn fails.
- **Pre-xattn vol self-attention also failed** (#929): symmetry does not hold.
- **STRING σ=0.25 is load-bearing** (#819). **Trainable log_freq is load-bearing** (#970).
- **K/V gradient backflow is load-bearing** (#890): detach-K/V failed.
- **Full MHA n_heads=4 is optimal** (#893).
- **GradNorm full-mode CLOSED NEGATIVE** (#942): vol pressure weight goes DOWN under GradNorm (vol_p loss decreases fastest).
- **SDF-zone masking at p_max=0.30 is unstable** (#953): ±4.2pp run-to-run variance.
- **CoordVolHead CLOSED NEGATIVE** (#959): Failed EP4 gate (7.428% vs gate 6.9%). STRING-sep + xattn already encode xyz-awareness.
- **Geometry-conditioned Q-bias CLOSED NEGATIVE** (#961): EP1=30.197% (0.197pp over 30% gate). Mean-pool surf loses spatial specificity.
- **SDF-FiLM CLOSED NEGATIVE** (#967): FiLM modulation of vol hidden states fails EP3 gate.
- **tanh saturation degeneracy** (#973 Arm A): at scale=2.0m, ~99% vol tokens saturate (gradient≈1e-8), catastrophic curriculum-transition loss spike.
- **Vol aux decoder head improves val_abupt but not test_vol_p** (#958 Arm A): new single-model SOTA val_abupt=6.2869%, but test_vol_p=12.0063% WORSE than baseline. OOD vol_p requires geometry-specific conditioning beyond a dedicated decoder head.
- **Flash SDPA required for large xattn tensors**: [B=4, H=4, N_vol=16384, N_surf=65536] bf16 ~32 GiB — OOM without Flash SDPA.
- **Mean-pool geometry embedding fails OOD vol_p** (#976 alphonse): global mean-pool surf hidden states lose spatial specificity needed for OOD geometry differentiation.
- **xattn is load-bearing for surface metrics** (#975 askeladd): surface-head-only ablation fails — xattn contributes to surface metrics not just vol_p.
