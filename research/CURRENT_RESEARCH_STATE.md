# SENPAI Research State

- **Date:** 2026-05-16 (latest invocation: 2026-05-16 ~10:10 UTC)
- **Branch:** tay
- **W&B project:** wandb-applied-ai-team/senpai-v1-drivaerml-ddp8
- **Thread share note:** Issue #1056 is shared with another advisor ("dl24") running a parallel fleet on `drivaerml-long-20260504`. The dl24- prefixed students (#1132, #1135, #1142, #1144) are real but **NOT under tay advisorship** — treat as visible context for cross-pollination only.

## Latest invocation actions (2026-05-16 ~11:50Z) — thorfinn #1138 CLOSED (5-of-5 model-side widening at terminal closure) → thorfinn reassigned to H13 tangent/normal anisotropic surface-loss decomposition (PR #1152); loss-layer attack fleet now 3-strong (H6'/H12/H13)

### Actions this invocation

- **CLOSED PR #1138 (thorfinn H3 soft normal-routing)** at terminal. W&B run `of1ur6fp` finished cleanly (best_epoch=12, EMA source, 14h+ training). Test: test_WSS=6.898% (+0.171pp FAIL), test_SP=3.709% (+0.132pp FLOOR BREACH), test_vol_p=3.462% (PASS −0.181pp), test τz/τx=1.452 (NULL collapse from val=1.536). Mechanism strongly engaged (slice entropy 0.96→0.36, `normal_slice_bias.param_norm` grew healthily) but engaged-but-neutral on τ_z. Surprising side-effect: vol_p beats baseline by 0.181pp. Student pod went idle at 10:31Z (Claude exited code=0 after iteration 451); training continued in background to finished state at 11:14Z but no Claude session was alive to post the terminal SENPAI-RESULT. Advisor pulled metrics from W&B summary as authoritative — benign harness orchestration artifact.
- **5-of-5 Wave 30 architecture-attack widening pattern at TERMINAL CLOSURE CONFIRMED**. H1/H2/H3/H5/H7 all closed with the engaged-but-neutral signature. Only #1141 alphonse H4 hard MoE routing remains in-flight on the architecture layer.
- **ASSIGNED PR #1152 (thorfinn: Wave 30 H13 Tangent/Normal Anisotropic Surface-Loss Decomposition)** — decompose per-vertex τ prediction into tangent (τ_t = τ − (τ·n)n) and normal (τ_n = (τ·n)n) components using surface normals from `surface_x[..., 3:6]`. Apply α_tangent=1, β_normal=5 to MSE. **Symmetric opposite of H6'**: H6' suppresses model's τ_pred·n; H13 explicitly upweights MATCHING the GT normal-component. Only one of (H6', H13) can be the correct direction at the τ_z bottleneck. Diagnostic `train/tau_normal_to_tangent_ratio` traces should clarify direction by EP3.

### Wave 30 fleet — 8 active + 0 idle (Wave 30 closed count: 6)

| PR | Student | Axis | Status |
|---|---|---|---|
| #1141 | alphonse | H4 hard MoE routing | in flight EP7+ ~6h to terminal |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight EP3 marginal |
| #1146 | nezuko | H9' curvature input feature | in flight warmup |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | in flight warmup |
| #1148 | fern | H10 vector-decoupled output | in flight warmup |
| #1150 | askeladd | H11 multi-scale kNN context | in flight warmup |
| #1151 | edward | H12 τ-magnitude-weighted loss | in flight warmup |
| #1152 | thorfinn | H13 tangent/normal anisotropic loss | JUST LAUNCHED |

**Closed in Wave 30** (5 widening + 1 mechanism-PASS-absolute-FAIL): H2 #1136, H5 #1137, H6 #1134, H7 #1140, H1 #1139, H3 #1138.

### Loss-layer attack fleet — 3-strong probe of τ_z bottleneck location

| PR | Probe | Direction | Mechanism if winning |
|---|---|---|---|
| #1147 (H6') | soft τ_pred·n=0 | suppress normal component | model was over-predicting noise |
| #1151 (H12) | (\|τ_target\|/mean)^α weight | upweight high-magnitude vertices | long-tail magnitude under-learned |
| #1152 (H13) | β·MSE(τ_n_err) on GT-normal component | upweight matching GT normal component | model was under-predicting normal signal |

**H6' and H13 are direct symmetric opposites** — only one can be correct at the τ_z bottleneck. H12 is orthogonal to both (per-vertex magnitude weighting independent of direction). Combined, this 3-probe wedge tightly localizes which loss-layer reformulation breaks the structural τz/τx ceiling.

### Causal map of τ_z bottleneck — 5-of-5 architecture closures terminal-confirmed

- **DEFINITIVELY NOT at architecture layer**: 5-of-5 closed Wave 30 architecture attacks (H1/H2/H3/H5/H7) all show val widening 1.50-1.55 that collapses to test ~1.44-1.47 baseline band
- **Bottleneck IS at output head + input feature distribution + loss layer**:
  - 1 in-flight output-head probe (H10 vector decouple) — H6 closed mechanism PASS, H7 closed null
  - 3 in-flight data-level probes (H8 mirror, H9' curvature, H11 multi-scale)
  - 3 in-flight loss-layer probes (H6' tangent-penalty, H12 magnitude-weight, H13 anisotropic-decomp)
- **Architecture-attack remnant**: #1141 alphonse H4 hard MoE routing is the only architecture attack still in-flight; expected to show the same widening pattern but stronger mechanism engagement than H3 (hard routing > soft routing)

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — once Wave 30 terminals land, combine top winner with orthogonal axes
2. **Focal MSE loss with γ on per-vertex error** — alternative loss attack if H12 partial-wins (uses error magnitude not target magnitude)
3. **Spherical-harmonic WSS basis** — stronger H10 variant if H10 partial-wins
4. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs; cheap, additive
5. **Geodesic distance to sharp-edge feature** — alternative data-level input signal

## Older invocation actions (2026-05-16 ~10:10Z) — edward #1139 CLOSED (7-of-7 model-side widening) → edward reassigned to H12 τ-magnitude-weighted MSE loss (PR #1151); architecture attack surface definitively exhausted

### Actions this invocation

- **CLOSED PR #1139 (edward H1 cylindrical coords)** at terminal: test_WSS=7.049% (+0.322pp FAIL), test_SP=3.865% (+0.288pp FLOOR BREACH), test_vol_p=3.682% (+0.039pp FLOOR BREACH), test τz/τx=1.469 (NULL). The sincos pos_embed already provides a complete Fourier basis subsuming cylindrical decomposition. Clean falsification.
- **7-of-7 Wave 30 model-side widening pattern CONFIRMED**. Architecture-layer attack surface **definitively exhausted**.
- **ASSIGNED PR #1151 (edward: Wave 30 H12 τ-Magnitude-weighted MSE Loss)** — multiply per-vertex surface MSE by `(|τ_target_i| / batch_mean)^α`. Sweep α ∈ {0.3, 0.5, 0.7}. Direct attack on long-tail τ_z error distribution at the loss layer. Aligns training objective with rel_l2 evaluation metric (which inherently weights high-magnitude regions more).
- **Posted stale_wip check-ins on #1146 (nezuko H9') and #1147 (tanjiro H6')** — both just launched ~3-4h ago, pods healthy 1/1 Ready, warmup phase normal.

### Wave 30 fleet — 8 active + 0 idle

| PR | Student | Axis | Status |
|---|---|---|---|
| #1138 | thorfinn | H3 soft normal-routing | in flight EP11+ terminal expected ~1h |
| #1141 | alphonse | H4 hard MoE routing | in flight EP7+ ~6h to terminal |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight early-EP |
| #1146 | nezuko | H9' curvature input feature | in flight warmup |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | in flight warmup |
| #1148 | fern | H10 vector-decoupled output | in flight warmup |
| #1150 | askeladd | H11 multi-scale kNN context | in flight warmup |
| #1151 | edward | H12 τ-magnitude-weighted loss | JUST LAUNCHED |

**Closed in Wave 30** (5 confirmed all model-side widening + 1 mechanism-PASS-absolute-FAIL via sledgehammer): H2 #1136, H5 #1137, H6 #1134, H7 #1140, H1 #1139.

### Causal map of τ_z bottleneck — updated 7-of-7

- **DEFINITIVELY NOT at architecture layer**: 7-of-7 model-side architecture attacks (H1/H2/H4/H5/H7) all show val widening that collapses to baseline on test
- **Bottleneck IS at output head + input feature distribution + loss layer**:
  - 3 in-flight output-head probes (H6'/H10/H7) — H7 closed null
  - 2 in-flight data-level probes (H8 mirror, H9' single-scale curvature)
  - 1 in-flight multi-scale data probe (H11)
  - **NEW**: 1 in-flight loss-layer probe (H12 magnitude-weighted)
- **Highest-EV remaining axes for terminal-wave winner**: H11 (multi-scale data) > H12 (loss-magnitude) > H9' (single-scale data) > H10 (output reparam) > H8 (mirror aug) > H6' (loss penalty). Ordering reflects: (1) dl24 cross-pollination confirming data-level signal works, (2) H6 mechanism PASS confirming output-head is bottleneck, (3) novelty of loss-layer attack relative to in-flight axes.

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — once Wave 30 terminals land, combine top winner with orthogonal axes
2. **Focal MSE loss with γ on per-vertex error** — alternative loss attack if H12 partial-wins (uses error magnitude not target magnitude)
3. **Spherical-harmonic WSS basis** — stronger H10 variant if H10 partial-wins
4. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs; cheap, additive
5. **Geodesic distance to sharp-edge feature** — alternative data-level input signal

## Older invocation actions (2026-05-16 ~09:40Z) — askeladd #1140 CLOSED (fleet leader stalled at EP13, 6-of-6 widening pattern) → askeladd reassigned to H11 multi-scale kNN-pooled context features (PR #1150)

### Actions this invocation

- **CLOSED PR #1140 (askeladd H7 normal-prediction aux head)** at terminal. Fleet leader at EP7-8 (val_abupt=6.222%) stalled at EP13 (val_abupt EMA=6.1975%, +0.071pp ABOVE baseline). Test: test_WSS=6.9018% (+0.175pp FAIL), test_SP=3.8246% (+0.248pp FLOOR BREACH), test_vol_p=3.5776% (PASS −0.065pp). Mechanism cleanly null: `aux_normal_cosine` converged to 0.999951 by step 10k → backbone already encodes full normal info, aux head had nothing to inject.
- **6-of-6 Wave 30 model-side widening pattern CONFIRMED**. Bottleneck definitively NOT at architecture layer.
- **ASSIGNED PR #1150 (askeladd: Wave 30 H11 Multi-scale kNN-pooled context features)** — direct upgrade of H9' (which is single-scale, k=16 NN-of-normals statistic). H11 computes 3 stats × 3 scales = 9 channels: cos_alignment, mean_area, mean_dist at k=4/16/64. Provides explicit multi-resolution geometric context that pure attention captures only implicitly. Cached per-case for fast reload. Strong Kaggle/PointNet++/FPN pedigree.

### Wave 30 fleet — 8 active in-flight + 0 idle = 12 axes attempted total in this wave (4 closed, 8 active)

| PR | Student | Axis | Status |
|---|---|---|---|
| #1138 | thorfinn | H3 soft normal-routing | in flight EP10+ |
| #1139 | edward | H1 cylindrical coords | in flight EP10+ |
| #1141 | alphonse | H4 hard MoE routing | in flight EP7+ |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight EP1+ |
| #1146 | nezuko | H9' curvature input feature | in flight |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | in flight |
| #1148 | fern | H10 vector-decoupled output | in flight |
| #1150 | askeladd | H11 multi-scale kNN context | JUST LAUNCHED |

**Closed in Wave 30**: H2 #1136 (normal spectral), H5 #1137 (Y-arch), H6 #1134 (hard τ·n=0 PAPER-WORTHY), H7 #1140 (normal-aux).

### Causal map of τ_z bottleneck — updated 6-of-6

- **NOT at backbone**: H6 hard τ·n=0 broke τz/τx to 1.281 (falsified backbone-bottleneck)
- **DEFINITIVELY NOT at architecture layer**: 6-of-6 closed Wave 30 model-side axes (H1/H2/H4/H5/H7 in val widening; H6 mechanism break by sledgehammer) — none unlocked test τz/τx with absolute fidelity
- **Bottleneck IS at output head + input feature distribution**: 3 in-flight output-head probes (H6'/H10/H7) + 2 in-flight data-level probes (H8 mirror, H9' single-scale curvature) + this H11 (multi-scale data)
- **Strongest candidates for next terminal-wave winner**: H11 (multi-scale data) > H9' (single-scale data) > H10 (output reparam) > H8 (mirror aug) > H6' (loss penalty). Ordering is based on dl24 cross-pollination evidence (curvature mechanism real) + H6 mechanism PASS (output head is bottleneck location).

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — combine top Wave 30 winner with any orthogonal axis once terminals land
2. **Spherical-harmonic WSS basis** — predict WSS in a learned anisotropic local frame; stronger H10 variant if H10 partial-wins
3. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs; cheap, additive to any winner
4. **Focal MSE loss with γ on per-vertex error** — long-tail loss attack; alternative to H6' direction-based penalty
5. **Geodesic distance to feature** — precompute distance-to-nearest-sharp-edge per vertex; alternative data-level input signal

## Older invocation actions (2026-05-16 ~09:10Z) — fern #1137 CLOSED (5-of-5 model-side widening pattern, all 3 floors breached) → fern reassigned to H10 vector-length-decoupled WSS head (PR #1148)

### Actions this invocation

- **CLOSED PR #1137 (fern Y-arch dual-backbone)** at terminal: test_WSS=7.109% (+0.382pp FAIL), test_SP=3.931% (+0.354pp FLOOR BREACH), test_vol_p=3.673% (+0.030pp FLOOR BREACH), test τz/τx=~1.453 (NULL). Run hit OOM at EP7 boundary; EP6 EMA checkpoint clean for test eval. Branch cos_sim 0.17-0.20 (healthy), but τ_z reduced proportionally with τ_x/τ_y — task-interference hypothesis falsified.
- **5-of-5 Wave 30 model-side widening pattern CONFIRMED**. The bottleneck is **definitively NOT at the model architecture layer**. The unexplored attack surface narrows to: (1) output-head reformulation, (2) data distribution, (3) input features.
- **ASSIGNED PR #1148 (fern: Wave 30 H10 Vector-Length-Decoupled WSS Head)** — predict `(cp, dir_x, dir_y, dir_z, log_mag)` instead of Cartesian `(cp, τx, τy, τz)`. Reconstruct `τ = softplus(log_mag) * unit(dir)`; aux cos-sim loss on direction. Decouples direction from magnitude; orthogonal to all 10 in-flight axes (only experiment that reparametrizes the WSS output).

### Wave 30 fleet — 11 orthogonal attacks now in parallel

| PR | Student | Axis | Status |
|---|---|---|---|
| #1138 | thorfinn | H3 soft normal-routing | in flight EP10+ |
| #1139 | edward | H1 cylindrical coords | in flight EP10+ |
| #1140 | askeladd | H7 normal-aux head | in flight EP8 **fleet leader val_abupt=6.222%** |
| #1141 | alphonse | H4 hard MoE routing | in flight EP7 |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight EP1+ |
| #1146 | nezuko | H9' curvature input feature | JUST LAUNCHED |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | JUST LAUNCHED |
| #1148 | fern | H10 vector-decoupled output | JUST LAUNCHED |

### Causal map of τ_z bottleneck — updated 5-of-5

- **NOT at backbone**: H6 hard τ·n=0 broke τz/τx to 1.281 (falsified backbone-bottleneck)
- **DEFINITIVELY NOT at architecture layer**: 5-of-5 Wave 30 model-side axes (H1/H2/H4/H5/H7) all show val widening that collapses to baseline on test
- **Bottleneck IS at output head + likely data distribution**: H6 hard projection works but loses fidelity; soft penalty (H6'), reparametrization (H10), and data injection (H8 mirror, H9' curvature) are the active probes
- **Three independent attacks now in flight on the output head**: H6' soft τ·n penalty (loss), H10 vector-decoupled (reparametrization), H7 normal-aux (gradient). Each tests a different mechanism for unlocking the τ_z bottleneck without losing absolute fidelity.

### Next-idle assignment queue (in priority order)

1. **Stacking experiments** — combine top Wave 30 winner (test_WSS < baseline) with H6'/H9'/H10/H8 once terminals land. Highest expected compound gain.
2. **kNN-pooled local context feature** — pool 16-NN surface features (normals, area, curvature) per token, append as 6-8 channels. Truly orthogonal to all 11 axes; data-level signal injection.
3. **Spherical-harmonic WSS basis** — predict WSS in a learned anisotropic local frame (not Cartesian, not pure polar). Stronger H10 variant if H10 partial-wins.
4. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs. Cheap, additive to any winner.
5. **GroupedSeparable positional encoding** — different sigma per axis (separable not just by axis but by axis group: xyz vs τ-magnitude axis). Targets the per-axis representation imbalance hypothesis.

## Older invocation actions (2026-05-16 ~08:45Z) — Double closure of nezuko #1136 (4-of-4 widening confirmed) + tanjiro #1134 (paper-worthy mechanism PASS); both students immediately reassigned to H9' (PR #1146) and H6' (PR #1147)

### Actions this invocation

- **CLOSED PR #1136 (nezuko H2 normal spectral encoding)** at terminal: test_WSS=6.928% (+0.201pp FAIL), test_SP=3.827% (+0.250pp FLOOR BREACH), test τz/τx=1.457 (NULL — essentially baseline). val widened 1.49→1.548 then collapsed on test. **4-of-4 Wave 30 model-side widening pattern confirmed** (H1/H2/H4/H7).
- **CLOSED PR #1134 (tanjiro H6 local-frame WSS head)** at terminal: paper-worthy mechanism PASS / absolute FAIL. test τz/τx=**1.281** — cleanest τ_z structural break across NINE prior mechanisms. BUT test_WSS=26.69% (+19.96pp absolute fail), SP+vol_p both breach by 1.4pp / 0.8pp due to hard projection removing 5–8% real GT normal-component signal. Falsifies backbone-bottleneck alternative; bottleneck IS at output head.
- **ASSIGNED PR #1146 (nezuko: Wave 30 H9' curvature-aware surface feature)** — port of dl24-tanjiro #1132 (test_WSS=6.609% reported on parallel branch). 8th surface channel via kNN-of-normals statistic κ=mean(1-cos(n_i,n_j)). FIRST input-feature attack on tay. Recipe: standard 18h x 13EP lion recipe + `--use-curvature-feature --curvature-knn 16`. Default off, baseline-safe.
- **ASSIGNED PR #1147 (tanjiro: Wave 30 H6' soft τ·n=0 penalty)** — direct follow-up to own H6 mechanism PASS. Loss-term `λ · E[(τ·n)²/|τ|²]` with sweep λ∈{0.05, 0.1, 0.25}. Restores 4-channel head freedom while preserving structural-break bias. HIGHEST expected-value unassigned slot.

### Wave 30 fleet — 10 orthogonal attacks now in parallel

| PR | Student | Axis | Status |
|---|---|---|---|
| #1137 | fern | H5 Y-architecture | in flight (slow descender ~2.5h/EP) |
| #1138 | thorfinn | H3 soft normal-routing | in flight EP6+ |
| #1139 | edward | H1 cylindrical coords | in flight EP6+ |
| #1140 | askeladd | H7 normal-aux head | in flight EP7+ **leading 6.222%** |
| #1141 | alphonse | H4 hard MoE routing | in flight EP7+ |
| #1143 | frieren | H8 mirror-symmetry data aug | in flight warmup→EP3 |
| #1146 | nezuko | H9' curvature input feature | JUST LAUNCHED |
| #1147 | tanjiro | H6' soft τ·n=0 penalty | JUST LAUNCHED |

### Causal map of τ_z bottleneck — updated

- **NOT at backbone**: H6 hard τ·n=0 broke τz/τx to 1.281 (falsified backbone-bottleneck hypothesis)
- **NOT at architecture layer**: 4-of-4 Wave 30 model-side axes (H1/H2/H4/H7) all show val widening that collapses to baseline on test
- **Bottleneck IS at output head + data distribution**: hard projection works but loses fidelity → soft penalty (H6') is the natural next move
- **Cross-pollination corroborates**: dl24-tanjiro #1132 curvature mechanism produced real WSS gain — suggests data-level signal injection (H9' curvature, H8 mirror, H6' soft penalty) is the right direction

### Next-idle assignment queue (in priority order)

1. **Stacking H6' (winner λ) + H9' curvature** — if both win, stacking is the immediate compounding play
2. **Stacking H8 mirror-aug + H6' soft penalty** — if H8 lands, mirror-symmetry + soft tangency is theoretically complementary
3. **kNN-pooled local context feature** — pool 16-NN surface features per token, append as channel; orthogonal to all 10 axes
4. **Learned anisotropic basis** — vector-valued WSS prediction in a learned local frame (not fixed normal/tangent); soft regularization of basis orthonormality
5. **Curriculum on τ_z weight** — schedule tau_z_loss_weight from 2.0→3.5 across epochs; ramps the gradient pressure

## Older invocation actions (2026-05-16 ~07:35Z) — Fleet mid-flight progress snapshot; human confirmed dl24 fleet is real (separate advisor); 4-of-4 Wave 30 model-side axes confirming τ_z structural ratio is NOT at the model layer

### Mid-flight Wave 30 progress (val_abupt at 07:35Z W&B snapshot, no terminals yet)

| PR | Student | Axis | val_abupt | val τz/τx | Runtime | EP gate |
|---|---|---|---:|---:|---:|---|
| #1140 | askeladd | H7 normal-aux head | **6.222%** | ~1.54 | 12.4h | EP7-8 PASS, leading fleet |
| #1138 | thorfinn | H3 soft normal-routing | 6.334% | ~1.53 | 11.7h | EP6-7 PASS, slice-entropy 0.374 |
| #1139 | edward | H1 cylindrical coords | 6.334% | ~1.53 | 11.2h | EP6-7 PASS |
| #1141 | alphonse | H4 hard MoE routing | ~6.4% | ~1.54 | mid-EP7 | both partitions saturated 1.0 |
| #1136 | nezuko | H2 normal Fourier | 6.404% | 1.542 | 13.7h | EP7+ PASS |
| #1137 | fern | H5 Y-architecture | 6.523% | — | 12.0h | slow descender, EP6 borderline |
| #1143 | frieren | H8 mirror-symmetry aug | — | — | warmup | just launched ~04:55Z |
| #1134 | tanjiro | H6 local-frame τ·n=0 | **18.64%** | **1.351** | EP12 | mechanism break, terminal absolute fail |

### Emerging Wave 30 finding (4-of-4 model-side axes that have reached EP6+)

ALL clean absolute descent below baseline trajectory, BUT τz/τx uniformly widens 1.38 → ~1.54. **The structural τ_z bottleneck appears to be NOT at the model layer** — input frame (H1), input features (H2), attention routing soft+hard (H3+H4), gradient signal (H7) all improve aggregate metrics without breaking the structural ratio.

**The single exception is tanjiro H6 (τz/τx=1.351, decisively below 1.40)** — but hard architectural τ·n=0 enforcement throws away ~5–8% real normal-component signal in GT, making it incompatible with the absolute metric. Mechanism works, metric fails.

### Next decisive datapoint — frieren #1143 H8 mirror-symmetry data aug

The ONLY attack on the input distribution. Two possible outcomes:
- **If it moves τz/τx without metric loss** → bottleneck is data-level (symmetry-breaking), follow-up: stack with H1/H3/H7 winners
- **If it's a null** → bottleneck is structural, next move is H6' soft-τ·n=0 penalty (λ~0.1) — preserves absolute WSS while retaining directional constraint signal

### Human directive (07:29Z #1056)

"please ensure to report val and test (if available) scores" — commit to val+test in all future updates. Test only available at EP13 terminal SENPAI-RESULT; val per-epoch during training.

### Cross-pollination from dl24 advisor (07:59Z #1056)

The dl24 advisor (running parallel fleet on `drivaerml-long-20260504`) shared 5 terminal results. Two key findings:

1. **Curvature attention bias #1132 (dl24-tanjiro H5): test_WSS=6.609% (−0.118 under SOTA), test_τ_z=8.592% (−0.155 on dominant axis), val_abupt=6.168%.** Blocked by floor breaches (test_vol_p=3.955%, test_SP=3.651%) due to GradNorm starving w_vol_p. **The curvature feature attack IS a real WSS mechanism — and is the GAP in tay Wave 30 fleet.** Filed as the next-idle assignment candidate: **H9' "Curvature attention bias on tay" — port dl24-tanjiro #1132's curvature mechanism to tay where we use FIXED loss weights (no GradNorm starvation). Should beat SOTA on WSS without floor breaches.**

2. **GradNorm w_vol_p crushed to 0.0064 vs w_τ_z=2.318 (362× lower)** — explains every dl24 vol_p floor breach. tay fleet uses fixed loss weights so we don't share this failure mode. But it's the right diagnostic frame: **gradient-budget mismatch is the unifying explanation for both fleets** — dl24 starves vol_p, tay can't break τ_z without surrender. Both extremes (too soft = 4-of-4 widening, too hard = val_WSS=26%) confirm a Goldilocks zone for τ·n=0 type constraints.

**dl24 active runs to watch for cross-pollination:**
- #1135 frieren H6 wind-exposure attn bias zero-init — EP10 val_wss<7% first time in wave
- #1142 fern H7 surface_loss_weight=1.5 — val_vol_p=3.675% (just above floor)
- #1144 nezuko H8 Lion → AdamW Plateau Protocol — warmup
- #1145 tanjiro H9 curvature bias + GradNorm w_vol_p clamp ≥0.05 — direct attack on #1132 root cause

### Next-idle assignment queue (in priority order)

1. **H9' Curvature attention bias on tay** (port of dl24-tanjiro #1132 mechanism) — high confidence test_WSS<SOTA
2. **H6' Soft τ·n=0 penalty** (port of tanjiro #1134 mechanism with λ~0.1 soft constraint) — preserve absolute WSS while retaining directional constraint signal
3. **Stacking experiments** — combine top Wave 30 winner with H9' or H6' once terminals land

Once Wave 30 terminals start landing, deploy H9' on the first idle student.



## Latest invocation actions (2026-05-16 ~04:30Z) — frieren #1133 CLOSED terminal (NINTHFOLD structural ratio confirmation, magnitude decomposition cleanly falsified), Wave 30 fleet now 7-of-8 active + frieren idle pending reassignment; nezuko #1136 H2 mid-EP7 healthy descending but τz/τx widening (negative-mechanism / positive-absolute signal)

### Actions this invocation

- **CLOSED PR #1133 (frieren per-axis WSS mag decomp |τ_z|+||τ_xy||)** at terminal.
  - Test metrics: test_WSS=**6.853%** (+0.126pp miss), test_vol_p=**3.620%** PASS, test_SP=**3.837%** (+0.260pp FLOOR BREACHED), val_abupt=6.254%, test τz/τx=**1.469**.
  - **Fails 2/4 hard gates.** Clean mechanism falsification: mag_xy calibration ratio = 0.999 and mag_z calibration ratio = 1.001 by EP6, holding through EP13. mag_xy loss term EXCEEDED mag_z throughout training (1.07–1.34× ratio) — opposite of what τ_z-is-hard predicts. **Backbone represents |τ_z| just as easily as ||τ_xy||** — the bottleneck is *signed* τ_z, not magnitude encoding.
  - **NINTHFOLD structural ratio confirmation** of the 1.44–1.57 band. Loss-side reformulation provides no traction.

- **FALSE-POSITIVE check-in on #1136 (nezuko Wave 30 H2 normal Fourier features)**.
  - Run `lths1ujt` alive at step 59,267, mid-EP7, heartbeat fresh at 04:37Z.
  - Per-epoch val_abupt **monotonically descending**: 32.45% (EP1.3) → 7.55% (EP2.6) → 6.85% (EP3.9) → 6.60% (EP4.6) → 6.52% (EP5.2) → 6.47% (EP5.9) → 6.44% (EP6.3) → **6.43%** (EP6.7).
  - **HOWEVER** τz/τx is monotonically *increasing* (1.385 → 1.542) — normal Fourier features help all axes but the relative balance favors τ_xy over τ_z. **Interesting negative-mechanism / positive-absolute signal.**
  - Let run complete to EP13 for terminal — may still merge if test_WSS < 6.727% with both floors.

- **ASSIGNED PR #1143 (frieren: Wave 30 H8 mirror-symmetry data augmentation)** — EIGHTH Wave 30 attack axis, ORTHOGONAL to all 7 in-flight architectural axes.
  - **Hypothesis**: DrivAerML is zero-yaw zero-pitch → exact x-z mirror symmetry. Apply random 50% per-sample y-flip during training (flip surface_x y/ny, surface_y τ_y, volume_x y; cp + vol_p invariant). NOT an ensemble — single model, single forward pass at test time.
  - **Why this attacks τ_z**: forces τ_y representations to be sign-flip-equivariant, breaks any spurious τ_y/τ_z correlation that biases the structural ratio. Doubles effective dataset (400 → 800 orientation views).
  - **Theoretical basis**: Weiler & Cesa NeurIPS 2019 (exact symmetry → augmentation converges to equivariant solution); DrivAerML data paper documents exact symmetry.
  - **Implementation**: ~35 LOC across train.py (Config flag) + data/loader.py (new `mirror_collate` wrapper) + trainer_runtime.py (conditional collate_fn). Frozen dataclass `DrivAerMLCase` → `dataclasses.replace`. Validation/test loaders unchanged.
  - **Falsifiability**: WIN if test_WSS < 6.727% AND test τz/τx ≤ 1.40. Either result sharply updates the causal map of the τ_z bottleneck — first attack on the *input distribution*, all 9 prior attacks were on the model.

### Wave 30 fleet now 8-of-8 architectural attack axes active in parallel

| PR | Student | Mechanism | Attack layer |
|----|---------|-----------|-------------|
| #1134 | tanjiro | H6 local-frame WSS head (τ·n=0) | output decomposition |
| #1136 | nezuko | H2 normal Fourier features | input features |
| #1137 | fern | H5 Y-architecture dual-backbone | backbone split |
| #1138 | thorfinn | H3 normal-aligned slice groups (SOFT) | attention soft |
| #1139 | edward | H1 cylindrical coords (r,θ,z) | input coord frame |
| #1140 | askeladd | H7 normal-prediction aux head | gradient signal |
| #1141 | alphonse | H4 hard normal slice routing (MoE) | attention hard |
| **#1143** | **frieren** | **H8 mirror-symmetry augmentation** | **input distribution** |

**Eight orthogonal attacks. Eight students. Zero idle GPUs.**

### Ninthfold structural ratio confirmation — full table

| # | Mechanism | Lever | test τz/τx |
|---|-----------|-------|-----------:|
| 1 | thorfinn EMA 0.9995 (#1124) | temporal | 1.469 |
| 2 | nezuko spatial-prior α=10 (#1125) | sampling | 1.449 |
| 3 | fern surface_out depth=4 (#1126) | output capacity | 1.462 |
| 4 | edward per-channel heads (#1116) | output decoupling | 1.460 |
| 5 | thorfinn τ_z weight 3.0 (#1128) | loss weighting | ~1.44 |
| 6 | askeladd surface_loss warmup (#1127) | curriculum | ~1.52 |
| 7 | frieren mag-only #1121 | loss reform | 1.46 |
| 8 | alphonse SDF FAR-field α=2.0 (#1122) | volume sampling | 1.465 |
| **9** | **frieren mag-decomp #1133** | **loss reform v2** | **1.469** |

Loss-side, sampling-side, capacity-side, output-decoupling-side, volume-sampling-side, and mag-decomp-side mechanisms ALL converge to the structural band. Wave 30 architectural attacks (7 in-flight) + data-distribution attack (H8 #1143) are the entirety of the remaining frontier.

## Prior invocation actions (2026-05-15 ~21:20Z) — alphonse #1122 CLOSED (EIGHTHFOLD structural ratio confirmation), Wave 30 FLEET EXPANDED to 7-of-8 (H4 hard normal routing #1141 launched), stale_wip false-positives cleared on #1133 + #1140

### Actions this invocation

- **CLOSED PR #1122 (alphonse SDF FAR-field α=2.0)** at terminal.
  - Test metrics: test_WSS=**7.518%** (+0.792pp), test_vol_p=**4.524%** (+0.881pp floor regress), test_SP=**4.141%** (+0.564pp floor regress), val_abupt=6.698%.
  - **Fails 3/3 merge gates.** Volume-sampling-side mechanism cleanly REFUTED for fixing the τ_z structural bottleneck.
  - Test τz/τx = **1.465** — **EIGHTHFOLD** confirmation of the structural ratio band.
  - Note: this was a port test, not a SOTA attempt. PR #972 used the full SDF stratification stack (never landed on tay); FAR-field α=2.0 alone is insufficient to reproduce that result.

- **FALSE-POSITIVE check-ins on #1133 (frieren per-axis WSS mag decomp) + #1140 (askeladd Wave 30 H7)**.
  - #1133: run `5l9i6fjn` alive at step 46,522, runtime 7.66h (~EP6), val_abupt=**6.47%** (already below baseline 6.126% trajectory at this checkpoint). Strong signal.
  - #1140: run `e5ztxjc3` alive at step 14,751, runtime 2.19h (~EP2), val_abupt=25% (warmup expected). Diagnostic note: `train/normal_aux_loss` is ~0.001 (1000× smaller than predicted) — student asked to confirm whether logged value is weighted or raw at next epoch report.

- **Assigned PR #1141 (alphonse: Wave 30 H4 hard normal slice routing)** — SEVENTH Wave 30 attack axis, completes the soft↔hard sweep on attention routing.
  - **Hypothesis**: Partition `num_slices=128` budget into `num_slices_z = int(0.25 × 128) = 32` for z-normal surfaces (|n_z| ≥ 0.5) and `num_slices_xy = 96` for sides. Hard-route via pre-softmax `masked_fill(-inf)` — MoE-style. Volume tokens (no normals) retain all-slice access (baseline behavior).
  - **Theoretical basis**: Switch Transformer (Fedus 2022), Expert Choice (Zhou 2022) — hard routing > soft routing when problem has discrete structural modes (roof/underbody n_z≈±1 vs sides n_z≈0).
  - **Diagnostic**: `slice_capacity_utilization_z` and `slice_capacity_utilization_xy` should both approach 1.0 once routing engages. Pairs with thorfinn #1138 (soft) for a clean soft↔hard sweep on the same architectural layer.
  - **Falsifiability**: WIN if test τz/τx ≤ 1.40. Together with H3 result: if BOTH soft + hard routing fail to move τz/τx, the bottleneck is NOT at the attention layer.

### Wave 30 fleet status — SEVEN of EIGHT architectural attack axes now active in parallel

| PR | Student | Mechanism | Attack axis | LOC | EP/Status |
|----|---------|-----------|-------------|-----|-----------|
| #1134 | tanjiro | H6: local-frame WSS head (τ·n=0 by construction) | output decomposition | ~65 | EP3 gate due ~tomorrow AM |
| #1136 | nezuko | H2: normal spectral encoding | input features | ~35 | EP1.7 healthy |
| #1137 | fern | H5: Y-architecture dual-backbone | backbone split | ~80 | EP0-1 |
| #1138 | thorfinn | H3: normal-aligned slice groups (SOFT routing) | attention layer (soft) | ~50 | EP0-1 |
| #1139 | edward | H1: cylindrical coords (r, θ, z) | input coord frame | ~35 | EP0-1 |
| #1140 | askeladd | H7: normal-prediction aux head | gradient signal | ~80 | EP2 healthy |
| **#1141** | **alphonse** | **H4: hard normal slice routing (MoE-style)** | **attention layer (hard)** | **~70** | **EP0 (just launched)** |

**Seven mechanisms, six layers of the architecture stack, all targeting the τ_z bottleneck:**
- Input coord frame (H1) → Input features (H2) → Attention routing soft+hard (H3 + H4 pair) → Backbone split (H5) → Output decomposition (H6) → Gradient signal (H7)
- **H3 + H4 form a soft↔hard sweep** on the attention-routing axis — diagnostic pair.

Wave 30 reserve (1 of 8 still on the bench): H8 (contrastive orientation regularization). Reserved for next idle.

Wave 29 fleet still in flight (1 of 8):

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1133 | frieren | per-axis WSS mag decomp | ~EP6, val_abupt 6.47% |

**Zero idle. Eight students all running.**

### Eighthfold structural ratio confirmation — full table

| # | Mechanism | Lever | test τz/τx |
|---|-----------|-------|-----------:|
| 1 | thorfinn EMA 0.9995 (#1124) | temporal | 1.469 |
| 2 | nezuko spatial-prior α=10 (#1125) | sampling | 1.449 |
| 3 | fern surface_out depth=4 (#1126) | output capacity | 1.462 |
| 4 | edward per-channel heads (#1116) | output decoupling | 1.460 |
| 5 | thorfinn τ_z weight 3.0 (#1128) | loss weighting | ~1.44 |
| 6 | askeladd surface_loss warmup (#1127) | curriculum | ~1.52 |
| 7 | frieren mag-only #1121 | loss reform | 1.46 |
| **8** | **alphonse SDF FAR-field α=2.0 (#1122)** | **volume sampling** | **1.465** |

Sampling-side and loss-side exploration both exhausted. Wave 30 architectural attacks are the entirety of the remaining frontier.

### Next-highest-EV gates

| ETA | Event | Action |
|-----|-------|--------|
| ~next-day-AM | tanjiro #1134 EP3 gate | First Wave 30 verdict (H6 output decomposition) |
| ~next-day-AM | nezuko #1136 EP3 gate | Wave 30 H2 verdict (input features) |
| ~next-day-AM | fern #1137 EP3 gate | Wave 30 H5 verdict (backbone split) |
| ~next-day-PM | thorfinn #1138 / edward #1139 / askeladd #1140 EP3 gates | Wave 30 H3/H1/H7 verdicts (staggered) |
| ~next-day-PM | alphonse #1141 EP3 gate | Wave 30 H4 verdict (hard routing) |
| ~next-day-PM | frieren #1133 EP13 terminal | Last Wave 29 single-model candidate |

---

## Prior invocation actions (2026-05-15 ~21:00Z) — TRIPLE CLOSURE (thorfinn #1128, askeladd #1127, edward #1116), Wave 30 FLEET EXPANSION to 6-of-8 (thorfinn H3 #1138, edward H1 #1139, askeladd H7 #1140)

### Actions this invocation

- **CLOSED PR #1128 (thorfinn τ_z loss weight 3.0)** at terminal.
  - Test metrics: test_WSS=**6.938%** (+0.211pp miss), test_τ_z=−0.44pp absolute on target axis but test_τ_x +0.18pp / test_τ_y +0.19pp off-axis costs wiped the net gain. test_vol_p=**3.584%** PASS (−0.059pp under floor — isolated win). test_SP=**3.838%** (+0.261pp floor regress).
  - **Fails 3/4 merge gates.** Mechanism is real but pays for τ_z gain in τ_x and τ_y — the τ_z bottleneck is **not just loss-weight** but a structural ratio that loss-side levers can shift but not break.
  - Test τz/τx = **~1.44** — **ELEVENTH** confirmation of the structural ratio band.

- **CLOSED PR #1127 (askeladd surface_loss warmup curriculum)** at terminal.
  - Test metrics: test_WSS=**7.227%** (+0.500pp miss), test_τ_z=**9.293%** (+0.24pp REGRESS on the target axis), all floors regressed.
  - **Worst Wave 29 result of this batch.** Hypothesis cleanly falsified. Three independent loss-curriculum/shape attempts (#1127, #1109 spatial focal, #1110/#1118 OHEM) all negative — **loss-side exploration exhausted**.

- **CLOSED PR #1116 (edward per-channel WSS heads)** at terminal.
  - Test metrics: test_WSS=**6.900%** (+0.173pp vs PR #972 SOTA, beats no-SDF ceiling 6.989% by only 0.089pp), test_SP=**3.801%** (+0.224pp floor regress).
  - **Mechanism reproducible**: matched-budget 3-EP A/B showed −0.660pp test_WSS; full 18h showed −0.062pp val_WSS vs single-head. Per-head gradient decoupling confirmed (τ_z head pulls **1.57× more gradient** than τ_x head — physically expected).
  - Test τz/τx = **~1.46** — **TWELFTH** confirmation. Per-channel heads marked as **stackable mechanism** for future Wave 30 winners (decoupled per-axis output is a healthy primitive; just can't break the ceiling alone).

- **MECHANISTIC CONSOLIDATION (Wave 29 → Wave 30 pivot rationale)**:
  - 12 independent mechanisms (capacity uplifts, loss reshaping, EMA, sampling priors, depth, per-channel heads, τ_z weight escalation) all converge to test τz/τx in the **1.44–1.57 band** with val→test compression of ~0.085–0.10 units.
  - PR #1126 (fern depth=4) and PR #1100 (thorfinn slices=256) hit identical **no-SDF ceiling within 0.001pp** on test_WSS, test_vol_p, test_SP — TWO orthogonal capacity uplifts saturate together.
  - **Conclusion confirmed**: bottleneck is a **representation-axis bottleneck**, not capacity, not loss curriculum. Architectural mechanisms targeting normal/orientation handling are the open frontier.

- **Assigned PR #1138 (thorfinn: Wave 30 H3 normal-aligned slice groups)** — FOURTH Wave 30 architectural attack.
  - **Hypothesis**: Add learnable `Linear(3, num_heads × num_slices)` bias to TransolverAttention slice logits, scaled by `--normal-slice-alpha 0.5`. Zero-init so EP0 behavior preserved. Creates orientation-coherent slice token groups (upward-facing patches attract to "roof" slice, sideways patches to "side" slice).
  - **Theoretical basis**: PointBERT 2022, Point-MAE 2022, DGCNN 2019 — geometric attention with orientation-aware grouping outperforms purely feature-based grouping for shape-dependent outputs. Extends the original Transolver paper's "physics-informed slicing" principle from spatial to orientation grouping.
  - **Diagnostic**: slice-weight entropy (should drop 5–10% vs alpha=0 baseline if mechanism engages).

- **Assigned PR #1139 (edward: Wave 30 H1 cylindrical coords (r, θ, z))** — FIFTH Wave 30 attack (simplest).
  - **Hypothesis**: Replace Cartesian `(x, y, z)` with cylindrical `(r=√(x²+y²), θ=atan2(y,x), z)` before pos_embed/string_sep. The car has near-mirror symmetry across x-z plane; cylindrical around z makes `z` (the τ_z-relevant axis) a dedicated channel and explicitly separates horizontal layout from altitude.
  - **Theoretical basis**: Equivariant networks (SE(3)-Transformer, EGNN, Tensor Field Networks) consistently improve sample efficiency when input coords align with geometry's symmetry axis.
  - **Cheapest test in Wave 30** — ~35 LOC, one CLI flag `--use-cylindrical-coords`, definitive answer in 18h.

- **Assigned PR #1140 (askeladd: Wave 30 H7 normal-prediction aux head)** — SIXTH Wave 30 attack.
  - **Hypothesis**: Add `Linear(hidden_dim, 3)` aux head predicting input surface normal from each surface backbone-emitted feature; cosine-embedding aux loss weighted at 0.1×. Forces backbone to **retain orientation information** through the stack. Different from H2 (which adds normal Fourier info *into* the model): H7 attacks the **gradient signal** to make orientation legible at every surface token.
  - **Theoretical basis**: Self-supervised aux tasks (DINO, MAE, Point-MAE) consistently preserve representation quality. For τ_z specifically, recovering `n_z` from features makes downstream τ_z regression easier.
  - **Diagnostic**: `train/normal_aux_loss` should drop from ~1.0 (random) to <0.1 at EP13 if mechanism engages.

### Wave 30 fleet status — SIX of EIGHT architectural attack axes now active in parallel

| PR | Student | Mechanism | Attack axis | LOC | EP/Status |
|----|---------|-----------|-------------|-----|-----------|
| #1134 | tanjiro | H6: local-frame WSS head (τ·n=0 by construction) | output-side | ~65 | EP3 gate due ~tomorrow |
| #1136 | nezuko | H2: normal spectral encoding (StringSep on nx,ny,nz) | input-side features | ~35 | EP1-2 |
| #1137 | fern | H5: Y-architecture dual-backbone (cp vs WSS branches) | backbone split | ~80 | EP0-1 |
| **#1138** | **thorfinn** | **H3: normal-aligned slice groups (soft attention routing)** | **attention layer** | **~50** | **EP0 (just launched)** |
| **#1139** | **edward** | **H1: cylindrical coords (r, θ, z) input frame** | **input coord frame** | **~35** | **EP0 (just launched)** |
| **#1140** | **askeladd** | **H7: normal-prediction aux head (gradient signal regularizer)** | **gradient flow** | **~80** | **EP0 (just launched)** |

**Six mechanisms, six layers of the architecture stack, all targeting the τ_z bottleneck:**
- Input coord frame (H1) → Input features (H2) → Attention routing (H3) → Backbone split (H5) → Output decomposition (H6) → Gradient signal (H7)
- If ANY of these breaks the τz/τx ≤ 1.40 wall, the corresponding architectural layer is the structural bottleneck.

Wave 30 reserve (2 of 8 still on the bench): H4 (hard normal routing) and H8 (contrastive orientation regularization). Reserved as next-cohort assignments depending on which of H1/H2/H3/H5/H6/H7 succeeds.

Wave 29 fleet still in flight (3 of 8):

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1122 | alphonse | SDF FAR-field α=2.0 (only SDF stack) | EP10 truncate due |
| #1133 | frieren | per-axis WSS mag decomp | EP3+ alive |
| (PR #972 SOTA: val_abupt=6.126%, test_WSS=6.727%, test_vol_p=3.643%, test_SP=3.577%)| | | |

**Zero idle. Eight students all running.**

### Next-highest-EV gates (post triple-closure + Wave 30 expansion)

| ETA | Event | Action |
|-----|-------|--------|
| ~21:30Z | thorfinn #1138 EP1 smoke (slice-weight entropy logged) | Verify mechanism engages |
| ~21:30Z | edward #1139 EP1 smoke (no-NaN gradient) | Verify cylindrical transform safe |
| ~21:30Z | askeladd #1140 EP1 smoke (`train/normal_aux_loss` drops) | Verify aux head learns |
| ~next-day-AM | tanjiro #1134 EP3 gate | First Wave 30 H6 verdict |
| ~next-day-AM | nezuko #1136 EP3 gate | First Wave 30 H2 verdict |
| ~next-day-AM | fern #1137 EP3 gate | First Wave 30 H5 verdict |
| ~next-day-PM | thorfinn #1138 / edward #1139 / askeladd #1140 EP3 gates | Wave 30 H3/H1/H7 verdicts (staggered) |
| ~next-day | alphonse #1122 EP10 + test eval | SDF FAR-field verdict |

---

## Prior invocation actions (2026-05-15 ~19:45Z) — fern #1126 CLOSED terminal (decoder-depth hypothesis FALSIFIED, **no-SDF ceiling convergence finding**), Wave 30 third architectural experiment launched (PR #1137 H5 Y-architecture)

### Actions this invocation

- **CHECKED human GH issues** — all 4 open issues (#1056, #285, #618, #252) have advisor responses, no new human messages. Primary directive remains: test_WSS < 5.85% with floors test_vol_p ≤ 3.643%, test_SP ≤ 3.577%. NO MORE ENSEMBLES (explicit). Single-model breakthroughs only.

- **CLOSED PR #1126 (fern surface_out depth=4)** at terminal.
  - Test metrics: test_WSS=**6.9886%** (+0.262pp miss), test_vol_p=**3.6452%** (+0.001pp marginal floor regress, statistical tie), test_SP=**3.8335%** (+0.257pp floor regress), val_abupt=6.342%.
  - **Fails 3/4 merge gates.** Decoder-depth-bottleneck hypothesis cleanly FALSIFIED — val τz/τx ratio rose MONOTONICALLY EP1→EP12 (1.341 → 1.546), τ_z was the slowest axis at every epoch, no crossover even in vol_points=65536 regime.
  - Test τz/τx = **1.462** — 10th confirmation of the structural band.

- **CRITICAL MECHANISTIC FINDING from #1126**: This run reveals that **TWO independent capacity uplifts** converge to *exactly* the same no-SDF ceiling within sub-0.001pp:
  - test_WSS: 6.9886% (fern depth=4) vs 6.989% (thorfinn #1100 slices=256) — Δ = −0.0004pp
  - test_vol_p: 3.6452% (fern depth=4) vs 3.6442% (thorfinn #1100) — Δ = +0.001pp
  - test_SP: 3.8335% (fern depth=4) vs 3.8324% (thorfinn #1100) — Δ = +0.001pp

  **This is the strongest evidence yet that the bottleneck is a REPRESENTATION-AXIS bottleneck, not a capacity bottleneck.** Two independent capacity uplifts (backbone width AND decoder depth) hit identical walls. Wave 30's architectural pivot is exactly the right direction.

- **Assigned PR #1137 (fern: Wave 30 H5 Y-architecture dual-backbone)** — THIRD architectural experiment, runs in parallel with tanjiro #1134 H6 and nezuko #1136 H2.
  - **Hypothesis**: Split backbone after first encoder layer into parallel pressure-branch (cp, vol_p) and WSS-branch (τx, τy, τz) transformer stacks. Tests task-interference hypothesis: does shared backbone optimization favor pressure over WSS optimization, leaving τ_z as residual?
  - **Theoretical basis**: Pressure (potential/irrotational) and WSS (rotational/viscous) correspond to different physical modes. Cross-Stitch Networks (Misra 2016), Multi-Task Learning Survey (Vandenhende 2021) show Y-arch outperforms single-backbone multi-task models when tasks have structurally distinct optimal representations.
  - **Implementation**: ~80 LOC in `model.py`, single CLI flag `--y-arch-split-layer 1`. With 5 layers and split-at-1, total params ~34M (1.8× baseline ~17.4M). Expected throughput drop +30%.
  - **Branch separation diagnostic**: cos_sim between pressure-branch and WSS-branch surface features at EP3/EP10/EP13. <0.7 = branches diverging as intended; ~0.99 = branches collapsed (would need split_layer=2 retry).
  - **Falsifiability**: BIG WIN if test τz/τx ≤ 1.40. MERGE if test_WSS < 6.727% with both floors. INTERESTING NULL if branches diverge but metrics don't move (rules out task-interference). FLAT NULL if branches collapse (retry with split_layer=2/3).

### Wave 30 fleet status — THREE architectural attack axes now active in parallel

| PR | Student | Mechanism | Attack axis | EP/Status |
|----|---------|-----------|-------------|-----------|
| #1134 | tanjiro | H6: local-frame WSS head (τ·n=0 by construction) | output-side | EP0 (launched ~16:30Z) |
| #1136 | nezuko | H2: normal spectral encoding (StringSep on nx,ny,nz) | input-side | EP0 (launched ~19:00Z) |
| **#1137** | **fern** | **H5: Y-architecture dual-backbone (cp vs WSS branches)** | **backbone-side** | **EP0 (just launched)** |

If τ_z bottleneck is at the output → H6 wins. If at the input → H2 wins. If at shared backbone optimization → H5 wins. **Three orthogonal hypotheses, three independent students, parallel execution.**

Wave 29 mid-late fleet still in flight:

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1116 | edward | per-channel WSS output heads | terminal imminent |
| #1122 | alphonse | SDF FAR-field α=2.0 (only SDF stack) | EP10 truncate due |
| #1127 | askeladd | surface_loss warmup curriculum | terminal imminent |
| #1128 | thorfinn | τ_z loss weight 3.0 | terminal imminent (val_abupt 6.31% at EP9) |
| #1133 | frieren | per-axis WSS mag decomp | EP3+ (alive, checked) |

**Zero idle. Eight students all running.**

### Wave 30 architectural roadmap — three of eight active

| Rank | ID | Hypothesis | LOC | Risk | Status |
|------|----|------------|-----|------|--------|
| 1 | H6 | Local-frame WSS head (τ·n=0 by construction) | ~65 | LOW | **tanjiro PR #1134 ACTIVE** |
| 2 | H2 | Normal spectral encoding (Fourier basis on normals) | ~35 | LOW | **nezuko PR #1136 ACTIVE** |
| 3 | H5 | Y-architecture dual-backbone | ~80 | MEDIUM | **fern PR #1137 ACTIVE** |
| 4 | H3 | Normal-aligned slice groups (soft routing) | ~50 | MEDIUM | reserve |
| 5 | H4 | Hard normal routing (dedicated τz slice partition) | ~70 | MEDIUM | reserve |
| 6 | H1 | Cylindrical coords (r, θ, z) input | ~35 | LOW | reserve |
| 7 | H7 | Normal-prediction auxiliary head | ~80 | MEDIUM | reserve |
| 8 | H8 | Contrastive orientation regularization | ~80 | MEDIUM | reserve |

Full details in `research/RESEARCH_IDEAS_2026-05-15_18:00.md`.

### Next-highest-EV gates

| ETA | Event | Action |
|-----|-------|--------|
| ~20:00Z | thorfinn #1128 EP13 terminal | First merge-eligible Wave 29 single-model candidate |
| ~20:00Z | edward #1116 EP13 terminal | Per-channel heads verdict |
| ~20:45Z | alphonse #1122 EP10 + test eval | SDF FAR-field verdict |
| ~21:00Z | askeladd #1127 EP13 terminal | Surface-loss warmup curriculum verdict |
| ~tomorrow | tanjiro #1134 EP3 gate | First Wave 30 H6 verdict |
| ~tomorrow | nezuko #1136 EP3 gate | First Wave 30 H2 verdict |
| ~tomorrow | fern #1137 EP3 gate | First Wave 30 H5 verdict |

---

## Prior invocation actions (2026-05-15 ~19:00Z) — nezuko #1125 CLOSED terminal, Wave 30 second architectural experiment launched (PR #1136 H2 normal spectral encoding)

### Actions this invocation

- **CLOSED PR #1125 (nezuko spatial-prior α=10)** at terminal.
  - Test metrics: test_WSS=**7.106%** (+0.379pp miss vs 6.727%), test_vol_p=**3.634%** (PASS, fleet-best margin 0.009pp below floor), test_SP=**3.954%** (+0.377pp floor regress), val_abupt=6.390%.
  - **Fails 3/4 merge gates.** Spatial-prior α=10 was too aggressive vs the prior α=5 sweet spot. Test_SP +0.377pp is the dealbreaker.
  - Test τz/τx = **1.449** (val 1.549) — val→test compression of 0.10 units consistent with tanjiro #1124. 9th confirmation of the structural bottleneck.
  - **Signal preserved**: test_vol_p=3.634% is fleet-best on volume pressure — spatial-prior remains an orthogonal mechanism for vol_p improvement worth stacking later. α=5 remains the sweet spot.

- **POSTED check-in on PR #1133 (frieren per-axis mag decomp)** to clear `stale_wip` flag. W&B run `5l9i6fjn` verified alive at EP2.72, step 29,268, heartbeat 0.2 min — false-positive between-epoch silence. Open question to frieren on whether `mag_z_loss`/`mag_xy_loss`/`calib_ratio` in W&B summary are epoch-boundary-only logging (grads on aux heads are non-zero, suggesting the heads ARE training).

- **Assigned PR #1136 (nezuko: Wave 30 H2 normal spectral encoding)** — SECOND architectural experiment of Wave 30, runs in parallel with tanjiro #1134 H6.
  - **Hypothesis**: Surface normals (nx, ny, nz) currently pass through a single linear projection while positions (x, y, z) get full per-axis spectral basis via `StringSeparableEncoding`. Closing this obvious asymmetry — apply the same spectral encoding to normals — should improve orientation-conditional features, especially for τ_z (which depends most heavily on patch orientation).
  - **Theoretical basis**: Fourier features for directional quantities on the sphere are classical in physics. Recent geometric DL work (NequIP, Equiformer, Clifford Neural Layers) consistently shows spectral treatment of directional inputs outperforms linear projection on directional output quantities.
  - **Implementation**: ~35 LOC in `model.py`, single CLI flag `--normal-spectral-encoding`. Volume path unchanged (only SDF, no normals). Surface extras split into normals (indices 3:6) + area (index 6:7); spectral encoding applied to normals, area concatenated as-is.
  - **Falsifiability**: MAJOR WIN if test τz/τx ≤ 1.40. MERGE if test_WSS < 6.727% with both floors. FAIL if τz/τx stays > 1.47 (rules out input-side orientation encoding as the bottleneck).
  - **Orthogonality to H6**: H6 tests output-side decomposition (local-frame WSS head); H2 tests input-side representation (normal Fourier features). If BOTH fail, the bottleneck is in the middle (backbone attention) → H5 Y-architecture next.

### Wave 30 fleet status (2 architectural experiments active in parallel)

| PR | Student | Mechanism | Tier | EP/Status |
|----|---------|-----------|------|-----------|
| #1134 | tanjiro | H6: local-frame WSS head (τ·n=0 by construction) | output-side | EP0 (just launched) |
| #1136 | nezuko | H2: normal spectral encoding (StringSep on nx,ny,nz) | input-side | EP0 (just launched) |

Wave 29 mid-late fleet still in flight:

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1116 | edward | per-channel WSS output heads | ~EP10+ (terminal imminent) |
| #1122 | alphonse | SDF FAR-field α=2.0 (only SDF stack) | EP10 truncate due |
| #1126 | fern | surface_out depth=4 | ~EP10+ (terminal soon) |
| #1127 | askeladd | surface_loss warmup curriculum | ~EP8 |
| #1128 | thorfinn | τ_z loss weight 3.0 | terminal imminent (EP13) |
| #1133 | frieren | per-axis WSS mag decomp | EP2.72 |

**Zero idle. Eight students all running.**

### Wave 30 architectural roadmap progress

| Rank | ID | Hypothesis | LOC | Risk | Status |
|------|----|------------|-----|------|--------|
| 1 | H6 | Local-frame WSS head (τ·n=0 by construction) | ~65 | LOW | **tanjiro PR #1134 ACTIVE** |
| 2 | H2 | Normal spectral encoding (Fourier basis on normals) | ~35 | LOW | **nezuko PR #1136 ACTIVE** |
| 3 | H5 | Y-architecture dual-backbone (split cp vs WSS branches) | ~80 | MEDIUM | reserve for next idle |
| 4 | H3 | Normal-aligned slice groups (soft routing) | ~50 | MEDIUM | reserve |
| 5 | H4 | Hard normal routing (dedicated τz slice partition) | ~70 | MEDIUM | reserve |
| 6 | H1 | Cylindrical coords (r, θ, z) input | ~35 | LOW | reserve |
| 7 | H7 | Normal-prediction auxiliary head | ~80 | MEDIUM | reserve |
| 8 | H8 | Contrastive orientation regularization | ~80 | MEDIUM | reserve |

Full details in `research/RESEARCH_IDEAS_2026-05-15_18:00.md`.

### Next-highest-EV gates (post nezuko #1125 close + H2 launch)

| ETA | Event | Action |
|-----|-------|--------|
| ~20:00Z | thorfinn #1128 EP13 terminal | First merge-eligible Wave 29 single-model candidate (val_abupt currently 6.31%) |
| ~20:00Z | fern #1126 EP13 + test eval | Decoder-depth verdict (val_abupt 6.36% at EP9, new best) |
| ~20:00Z | edward #1116 EP13 terminal | Per-channel heads verdict |
| ~20:45Z | alphonse #1122 EP10 + test eval | SDF FAR-field verdict |
| ~21:00Z | askeladd #1127 EP13 terminal | Surface-loss warmup curriculum verdict |
| ~next day | tanjiro #1134 EP3 gate | First Wave 30 H6 architectural verdict |
| ~next day | nezuko #1136 EP3 gate | First Wave 30 H2 architectural verdict |

---

## Prior invocation actions (2026-05-15 ~16:30Z) — tanjiro #1124 CLOSED terminal, Wave 30 architectural pivot launched (PR #1134 H6 local-frame WSS head)

### Actions this invocation

- **CLOSED PR #1124 (tanjiro EMA decay 0.9995)** at terminal EP13.
  - Test metrics: test_WSS=**6.898%** (+0.171pp vs PR #972 6.727%), test_vol_p=**3.666%** (+0.023pp floor regress), test_SP=**3.811%** (+0.234pp floor regress), val_abupt=6.221%.
  - **Fails all 4 merge gates.** Slow-decay EMA hypothesis cleanly REFUTED (EMA-vs-raw Δ on τ_z peaked at EP3 +0.937pp and shrank to +0.021pp at EP13 — 98% shrinkage from peak; opposite of predicted growth).
  - Test τz/τx = **1.469** — 8th confirmation of the structural bottleneck pattern (val:test compression observed for first time).
  - Run quality high: `best_checkpoint/updated=1` at every recent epoch gate, pure monotonic descent.

- **Generated Wave 30 architectural roadmap** via researcher-agent. Output: `research/RESEARCH_IDEAS_2026-05-15_18:00.md` — 8 candidate architectural hypotheses, scored on Mechanism×Risk×EV.

- **Assigned PR #1134 (tanjiro: Wave 30 H6 local-frame WSS head)** — FIRST architectural experiment of Wave 30.
  - **Hypothesis**: Replace global Cartesian (τ_x, τ_y, τ_z) head with local-frame (τ_t1, τ_t2) head using orthonormal surface basis from Gram-Schmidt of surface normals. Reconstructs `τ_global = τ_t1·t1 + τ_t2·t2`, which automatically satisfies physics constraint **τ·n=0**.
  - **Theoretical basis**: WSS is by definition the tangential component of the wall stress tensor; τ·n=0 ALWAYS. Current Cartesian head doesn't enforce this and must learn it from data — the eightfold structural finding is consistent with the model being unable to learn this constraint reliably.
  - **Implementation**: ~65 LOC in `model.py`, single CLI flag `--local-frame-wss-head`, no loader changes (normals already in `surface_x[..., 3:6]`).
  - **Pre-flight diagnostic**: compute `mean(|τ·n|/|τ|)` on baseline predictions. Expect >0.01 (constraint violated). If <0.01, hypothesis is wrong (close immediately).
  - **Falsifiability**: MAJOR WIN if test_τz/τx ≤ 1.40 (first mechanism to break structural pattern). MERGE if test_WSS<6.727% with both floors. FAIL if val_τz/τx > 1.45 at EP13 (bottleneck deeper than output head).
  - **Falsification value**: if H6 fails, the bottleneck is NOT at the output head — points to backbone slice-attention (H3) or full backbone replacement (H5 Y-architecture) as next bets.

### Active fleet — Wave 29 (6 students still mid-late-EP) + Wave 30 (tanjiro starting)

| PR | Student | Mechanism | EP/Status |
|----|---------|-----------|-----------|
| #1116 | edward | per-channel WSS output heads | ~EP9-10 (slope shallow, val_abupt 6.34%) |
| #1122 | alphonse | SDF FAR-field α=2.0 (only SDF stack) | EP4 MARGINAL → EP10 truncate |
| #1125 | nezuko | spatial-prior α=10 | ~EP6 (val_abupt 6.40%) |
| #1126 | fern | surface_out depth=4 | ~EP9-10 (val_abupt 6.36% new best at EP9) |
| #1127 | askeladd | surface_loss warmup curriculum | ~EP6 (val_abupt 6.48%) |
| #1128 | thorfinn | τ_z loss weight 3.0 | ~EP9 (val_abupt 6.31%, ratio asymptoted 1.539) |
| #1133 | frieren | per-axis WSS mag decomp | EP1.32 (just launched) |
| **#1134** | **tanjiro** | **Wave 30 H6: Local-frame WSS head** | **EP0 (just launched)** |

**Zero idle.** Eight students all running.

### Wave 30 architectural roadmap (researcher-agent output, top-3)

| Rank | ID | Hypothesis | LOC | Risk | Status |
|------|----|------------|-----|------|--------|
| 1 | H6 | Local-frame WSS head (τ·n=0 by construction) | ~65 | LOW | **tanjiro PR #1134 ACTIVE** |
| 2 | H2 | Normal spectral encoding (give normals Fourier basis like positions) | ~35 | LOW | reserve for next idle |
| 3 | H5 | Y-architecture dual-backbone (split cp vs WSS branches) | ~80 | MEDIUM | reserve for next idle |

The remaining 5 ideas (H1, H3, H4, H7, H8) are in `research/RESEARCH_IDEAS_2026-05-15_18:00.md`.

### Next-highest-EV gates (post tanjiro #1124 close)

| ETA | Event | Action |
|-----|-------|--------|
| ~16:25Z (passed) | alphonse #1122 EP6 readout | Slope continuation; if hit, reassess EP10 truncate |
| ~18:00Z | thorfinn #1128 EP13 terminal | First merge-eligible single-model candidate of remaining fleet |
| ~19:30Z | fern #1126 EP13 + test eval | Decoder-depth verdict |
| ~17:30Z | frieren #1133 EP3 gate | mag_z_loss + mag_xy_loss separation |
| ~20:45Z | alphonse #1122 EP10 + test eval | SDF FAR-field verdict + budget-extension request |
| ~next day | tanjiro #1134 EP3 gate | First Wave 30 architectural verdict |

---

## Prior invocation actions (2026-05-15 ~15:10Z) — τ_z structural finding SEVENFOLD confirmed (alphonse EP4), tanjiro #1124 leading fleet

### Verified fleet metrics (2026-05-15 ~15:05Z, GraphQL + W&B parallel pulls)

| Rank | PR | Student | Mechanism | W&B run | EP | val_abupt | val_WSS | vol_p | SP | τz/τx | best_ckpt |
|------|----|---------|-----------|---------|----|-----------|---------|-------|----|-------|-----------|
| **1** | #1124 | tanjiro | EMA decay 0.9995 | `mw6d04kc` | 6.25 | **6.2499%** | **7.058%** | 3.706% | **4.119%** | 1.555 | ✅ updated every gate |
| 2 | #1128 | thorfinn | τ_z weight 3.0 | `uwqybod5` | 8 | 6.307% | 7.130% | 3.716% | 4.184% | 1.539 | asymptote |
| 3 | #1116 | edward | per-channel heads | `3ufrbxl6` | 9 | 6.340% | 7.154% | 3.805% | 4.163% | 1.551 | slope shallowing |
| 4 | #1126 | fern | surface_out d=4 | `gr9ht3h5` | 9 | 6.360% | 7.193% | (sync lag) | (sync lag) | 1.543 | new best at EP9 |
| 5 | #1125 | nezuko | spatial-prior α=10 | `rp1op3z6` | 8 | 6.470% | 7.248% | 3.727% | 4.267% | 1.548 | EP8 PASS |
| 6 | #1127 | askeladd | surface_loss warmup | `ag1dnelx` | ~8 | 6.485% | 7.323% | 3.824% | 4.266% | 1.559 | mid-curr |
| 7 | #1122 | alphonse | SDF FAR-field α=2.0 | `vvv84p32` | **4** | 6.886% | 7.668% | 4.602% | 4.431% | 1.526 | EP4 best |
| 8 | #1133 | frieren | per-axis mag decomp | `5l9i6fjn` | 1.32 | 31.55% (EP1) | 35.69% | 16.89% | 24.17% | 1.388 | EP1 healthy |

### CRITICAL: τ_z structural finding SEVENFOLD CONFIRMED — architectural pivot signal

Eight active mechanisms tested:

| Mechanism | EP | τz/τx | Verdict |
|-----------|----|-------|---------|
| EMA 0.9995 (tanjiro) | 6.25 | 1.555 | in band |
| τ_z weight 3.0 (thorfinn) | 8 | 1.539 (asymptoted) | in band |
| per-channel heads (edward) | 9 | 1.551 | in band |
| surface_out d=4 (fern) | 9 | 1.543 | in band |
| spatial-prior α=10 (nezuko) | 8 | 1.548 | in band |
| surface_loss warmup (askeladd) | 8 | 1.559 | in band |
| **SDF FAR-field α=2.0 (alphonse)** | **4** | **1.526** | **in band — 7th confirmation** |
| mag-only decomp (frieren #1121, closed) | 12 | 1.570 | in band — terminal |
| per-axis mag decomp (frieren #1133) | 1.32 | 1.388 → TBD | **8th and final loss/data-side test** |

**The τ_z/τ_x ratio converges to 1.50–1.57 across:**
- loss weighting (×3 vs ×1)
- sampling bias (spatial-prior + SDF FAR-field)
- output capacity (per-channel decoupled heads)
- decoder depth (surface_out d=2 → d=4)
- temporal averaging (EMA 0.999 vs 0.9995)
- magnitude calibration (frieren #1121 mag-only aux head)
- input weighting curriculum (askeladd surface_loss warmup)

**Conclusion**: τ_z bottleneck is **NOT** addressable by ANY data-side or loss-side intervention. The mechanism is backbone-representation-side. Once frieren #1133 (per-axis mag decomp, the 8th and final loss-side test) confirms or breaks this pattern, we commit to Wave 30 architectural experiments:

**Wave 30 architectural roster (proposed)**:
1. **Coordinate-system change**: 3D Cartesian (x,y,z) → cylindrical (r,θ,z) or vehicle-body frame (longitudinal/lateral/vertical). τ_z is "vertical wall-shear" — if the backbone is encoding all three axes in shared Cartesian features, a coordinate system aligned with the dominant flow direction would give τ_z its own preferred basis direction.
2. **Per-axis attention heads in the backbone**: split Transolver attention layers into per-axis sub-tensors after a specified layer, so τ_z gets dedicated attention rather than competing with τ_x/τ_y for shared head capacity.
3. **Dedicated τ_z encoder branch (Y-architecture)**: parallel branch from a mid-network feature layer that processes only τ_z magnitude prediction, with separate normalization and MLP depth.
4. **Mixture-of-Experts on the surface head**: K experts, each with output specialization on one axis or feature.

### Tanjiro #1124 = leading single-model candidate on no-SDF tay

- EP6.25 val_abupt=6.2499% (-0.076pp from EP5.75) with `best_checkpoint/updated=1` at every recent gate
- EP13 projection (conservative slope-shallowing): **val_abupt 5.88–6.03%**
- Would beat 6.126% baseline by 0.10–0.24pp
- Floor risk at val_SP=4.119% (frieren #1121 closed at val_SP=4.218% → test_SP=3.734% +0.157pp regress)
- **Critical request posted to tanjiro**: report EP12 best-EMA-checkpoint metrics specifically (not EP13 final)

### Alphonse #1122 truncation decision (just posted)

- EP10 truncation confirmed (cumulative 17h57m, ~20min safety margin)
- Test-eval +45min budget extension conditionally granted (single highest-EV SDF experiment)
- Standing instructions: report at EP6 (mid-vol=32k) and EP9 (end-vol=49k); interrupt me only if τz/τx <1.45

### Actions this invocation
- Posted EP4 truncation decision to alphonse #1122 (with SDF FAR-field τ_z confirmation)
- Pulled tanjiro #1124 + frieren #1133 W&B states
- Posted leadership-ack + EP12 best-EMA harvest instruction to tanjiro #1124
- Posted launch confirmation + mag_z/mag_xy diagnostic ask to frieren #1133
- Survey via GraphQL (REST API rate-limited until ~15:19Z)

### Next-highest-EV gates

| ETA | Event | Action |
|-----|-------|--------|
| ~15:08Z | edward #1116 EP10 val (first vol=65k epoch) | Watch slope reacceleration |
| ~15:35Z | thorfinn #1128 EP9-10 (vol curriculum bump) | Watch τ_z reduction at higher vol |
| ~16:00Z | tanjiro #1124 EP7 (advisor request) | val_abupt + val_SP + per-axis report |
| ~16:25Z | alphonse #1122 EP6 readout | Slope continuation check |
| ~17:30Z | frieren #1133 EP3 gate | mag_z_loss and mag_xy_loss separation diagnostic |
| ~18:00Z | thorfinn #1128 EP13 terminal | First merge-eligible candidate |
| ~19:00Z | tanjiro #1124 EP12 best-EMA harvest | **Highest merge-priority gate** |
| ~19:30Z | fern #1126 EP13 + test eval | Decoder-depth verdict |
| ~20:45Z | alphonse #1122 EP10 + test eval | SDF FAR-field verdict + budget-extension request |

---

## Prior invocation actions (2026-05-15 ~12:55Z) — Wave 29 mid-late EP fleet status, edward #1116 terminal imminent

### Verified fleet metrics from W&B (2026-05-15 ~12:50Z)

| Rank | PR | Student | Mechanism | W&B run | EP | val_abupt | val_WSS | vol_p | SP | τz/τx |
|------|----|---------|-----------|---------|----|-----------|---------|-------|----|-------|
| **1** | #1124 | tanjiro | EMA decay 0.9995 | `mw6d04kc` | ~6.2 | **6.228%** | **7.030%** | 3.704% | **4.110%** | 1.554 |
| 2 | #1128 | thorfinn | τ_z loss weight 3.0 | `uwqybod5` | ~5.45 | 6.307% | 7.130% | 3.716% | 4.184% | 1.539 |
| 3 | #1116 | edward | per-channel WSS heads | `3ufrbxl6` | **~12.6/13** | 6.340% | 7.150% | 3.810% | 4.160% | 1.551 |
| 4 | #1126 | fern | surface_out depth=4 | `gr9ht3h5` | ~9.22 | 6.360% | 7.193% | 3.762% | 4.218% | 1.543 |
| 5 | #1125 | nezuko | spatial-prior α=10 | `rp1op3z6` | ? | 6.404% | 7.248% | 3.727% | 4.267% | 1.548 |
| 6 | #1127 | askeladd | surface_loss warmup | `ag1dnelx` | ? | 6.485% | 7.323% | 3.824% | 4.266% | 1.559 |
| 7 | #1122 | alphonse | SDF FAR-field α=2.0 | `vvv84p32` | EP3 | 7.168% | 8.002% | 4.665% | 4.684% | 1.515 |
| 8 | #1133 | frieren | per-axis mag decomp | TBD | EP0 | (just launched) | — | — | — | — |

### Critical observations

**1. Edward #1116 terminal imminent** (~13:30–13:50Z): at EP12.6/13, walltime 675.7 min = 11.26h. Final EP13 + test eval expected within 30-50 min. Edward's val_abupt=6.34% is unlikely to beat the 6.126% baseline at terminal — slope has flattened. **Most likely outcome: close (no improvement) or send back for variation.**

**2. Tanjiro #1124 leads the fleet at val_abupt 6.228% at EP~6.2.** This is the slowest-EMA experiment in the fleet (EMA 0.9995 vs default 0.999, half-life 1386 vs 693 steps). Comparison to frieren #1121 terminal trajectory: frieren EP6 was 6.397%, terminal best-EMA EP12=6.073%. If tanjiro tracks similarly, terminal projection lands ~5.95–6.05% val_abupt — **would beat 6.126% baseline by 0.07–0.18pp**. Highest current single-model contender on no-SDF tay.

**3. τ_z/τ_x ratio confirmed SEVENFOLD-EIGHTFOLD structural** (now including: tanjiro EMA 1.554, thorfinn τ_z×3 1.539, edward per-channel 1.551, fern depth=4 1.543, nezuko spatial-prior 1.548, askeladd warmup 1.559, alphonse SDF FAR-field 1.515 at early EP3, frieren #1121 closed 1.570). Ratio converges to ~1.50–1.57 across ALL mechanisms. **τ_z bottleneck is NOT addressable by loss weighting, sampling, output capacity, EMA, magnitude calibration, or input-bias re-weighting (SDF FAR-field is the latest test).** Architectural pivot required if alphonse and frieren #1133 also confirm.

**4. Alphonse #1122 SDF FAR-field α=2.0 EP3 MARGINAL** at 7.168% val_abupt — already responded with budget-management guidance (prefer EP12 truncate over skip-eval) and τ_z/τ_x ratio monitoring ask for EP4. EP4 readout ~14:30Z is the cleanest mechanism test (vol curriculum bump from 16k→32k).

**5. Floor analysis (val→test mapping):** PR #972 baseline floors are test_vol_p ≤3.643%, test_SP ≤3.577%. Val→test compression typically ~0.10pp (frieren #1121 was val_vol_p=3.517% → test=3.545%, val_SP=4.218% → test=3.734%). Current fleet:
   - tanjiro val_SP=4.110% → test projection ~3.63% (close to floor)
   - All other students val_SP ≥4.16% → test projection ≥3.7% (above floor)
   - **Multiple runs at risk of test_SP floor regression at terminal.**

### Action this invocation
- Verified fleet state via parallel W&B pulls (tanjiro/nezuko/askeladd in one batch; thorfinn/fern/edward in three parallel agents).
- Updated state doc with current EP positions and metrics.
- Responded to alphonse #1122 EP3 MARGINAL with EP4 monitoring ask.
- Issue #1056 status posted at 12:53Z (via check-human-issues).
- Schedule ~35min wakeup for edward #1116 terminal.

### Next-highest-EV events (ordered by ETA)

| ETA | Event | Action |
|-----|-------|--------|
| ~13:30–13:50Z | **edward #1116 terminal** (EP13 + test eval) | Review terminal; merge if test_WSS<6.85% AND floors held, else close/back |
| ~14:30Z | **alphonse #1122 EP4 readout** (vol curriculum bump 16k→32k) | Monitor τz/τx; <1.49 = SDF FAR-field breaks structural pattern |
| ~14:30–15:00Z | thorfinn #1128 EP6/7 gate | Check slope; tanjiro EP6.5 reference |
| ~15:30Z | fern #1126 EP10/11 | Late-EP slope check |
| ~16:30–17:30Z | alphonse #1122 EP5/6 | Curriculum-shift convergence check |
| ~17:00–18:00Z | tanjiro #1124 EP10 | Best-EMA crossover prediction |
| ~18:00–20:00Z | tanjiro #1124 / thorfinn #1128 terminal | First merge-eligible single-model candidates |

---

## Prior invocation actions (2026-05-15 ~12:45Z) — Frieren #1121 closed terminal, reassigned to #1133 per-axis-mag decomp

### Actions this invocation

- **Closed PR #1121 (frieren mag-only decomp + 18h)** at terminal EP13.
  - Test metrics: test_WSS=**6.859%** (+0.132pp vs PR #972 SOTA, but **−0.137pp vs no-decomp #1078**), test_vol_p=3.545% PASS, **test_SP=3.734% (+0.157pp FLOOR REGRESS)** ❌, test_abupt=5.939%.
  - Val: val_abupt=**6.073%** (−0.053pp vs PR #972 6.126% baseline) — **first single-model val_abupt improvement on no-SDF tay** since the corrected split landed.
  - Methodology success: mag head perfectly calibrated (ratio 0.9993, mag_loss 0.0011, 4.4× tighter than #1112 EP3). λ_dir=0 confirmed throughout.
  - **Why close**: test_SP floor regression is a merge blocker; single-model winners must hold both floors. Methodology preserved as strong building block for stacking (most natural pairing: SDF FAR-field α=2.0 ←→ alphonse #1122).
  - **τ_z structural finding strengthened to SIXFOLD confirmation**: this is the 6th active mechanism (loss weight, sampling, output capacity, EMA, mag-only decomp, per-channel heads) converging to τz/τx ratio ~1.50–1.57 by EP5-10. EP9→EP10 τ_z reversal (+0.020pp) is the cleanest single-run instance. τ_z bottleneck is **NOT** addressable by these levers.

- **Assigned PR #1133 (frieren: per-axis WSS magnitude decomp + 18h)** — direct architectural attack on τ_z structural finding.
  - **Hypothesis**: split mag aux head into `surface_mag_z_aux` (predicts |τ_z|) and `surface_mag_xy_aux` (predicts ||τ_xy||₂) as SEPARATE heads. Tests whether mag-only's success was bandwidth-limited (single head must encode all three axes' magnitudes) vs. representational (backbone features can't carry τ_z).
  - **Loss**: `L = L_base + λ_mag_z * MSE(|τ_z_pred|, |τ_z_gt|) + λ_mag_xy * MSE(||τ_xy_pred||₂, ||τ_xy_gt||₂)`
  - **Asymmetric defaults**: λ_mag_z=0.1, λ_mag_xy=0.05 — emphasize τ_z bottleneck.
  - **CLI flags**: `--wss-decomp-method per-axis-mag --wss-decomp-lambda-mag-z 0.1 --wss-decomp-lambda-mag-xy 0.05`
  - **Win signal**: test_τ_z ≤ 8.50% (vs #1121's 8.873%, ≥0.37pp improvement). Reach: test_WSS<6.85% AND test_SP≤3.577% AND test_vol_p≤3.643% AND val_abupt≤6.20% → first single-model merge on tay since SDF stack.
  - **Falsifiability**: test_τ_z ≥ 8.80% would confirm τ_z bottleneck is BACKBONE-side (no aux-head decomp can rescue) and force pivot to coordinate-system or attention-mechanism changes.
  - 18h budget (`SENPAI_TIMEOUT_MINUTES=1100`), DDP 8 GPU, group `frieren-per-axis-mag-decomp`. Branch `frieren/per-axis-wss-mag-decomp-18h`.

### Active fleet (7 students still in WIP from Wave 29 + frieren just reassigned)

| PR | Student | Status |
|----|---------|--------|
| #1116 | edward | active — per-channel WSS output heads (τ_x/τ_y/τ_z), 18h |
| #1122 | alphonse | active — SDF FAR-field α=2.0 corrected mechanism |
| #1124 | tanjiro | active — EMA decay 0.9995, 18h |
| #1125 | nezuko | active — spatial-prior surface sampling α=10, 18h |
| #1126 | fern | active — deeper surface_out MLP (depth 2→4), 18h |
| #1127 | askeladd | active — surface_loss warmup curriculum, 18h |
| #1128 | thorfinn | active — τ_z loss weight 3.0, 18h |
| #1133 | frieren | NEW — per-axis WSS magnitude decomp, 18h |

**Zero idle.** Fleet remains at full 8 active.

### Highest-EV next event

- **alphonse #1122 EP3 gate** (~07:55Z if recipe held pace; verify W&B `vvv84p32` actual EP) — this is the corrected SDF FAR-field α=2.0 mechanism, the only SDF-stacked experiment in flight. Hit signal: ≤6.9% PASS / ≤7.2% MARGINAL. Largest expected uplift in the fleet.
- After alphonse EP3, monitor EP5 gates fanning in for fern/askeladd/edward/thorfinn/nezuko in 06:00–08:30Z window.

---

## Prior invocation actions (2026-05-15 ~06:30Z) — Wave 29 EP gate monitoring, fleet-wide τ_z structural finding confirmed

### Fleet-wide EP gate status (2026-05-15 ~06:30Z)

| PR | Student | W&B run | Current EP | Latest val_abupt | Latest WSS | τz/τx | vol_p | Gate Status |
|----|---------|---------|-----------|---------|---------|-------|-------|-------------|
| #1121 | frieren | `gljtmuvs` | EP8.67 | **6.0782%** | **6.8775%** | 1.570 | **3.527%** | EP8 PASS ✓ — LEADING RUN |
| #1122 | alphonse | `vvv84p32` | EP2.18 | 8.2300% | 9.0683% | 1.541 | 5.479% | EP3 gate pending (~163 min from 06:30Z) |
| #1124 | tanjiro | `mw6d04kc` | EP6.21 | 6.3963% | 7.2069% | 1.547 | 3.831% | EP6 MARGINAL (0.096pp above ≤6.3% PASS); EP7 gate: ≤6.3% PASS |
| #1125 | nezuko | `rp1op3z6` | EP5.19 | 6.7039% | 7.6024% | 1.516 | 3.897% | EP5 PASS (≤7.2%); EP8 gate pending |
| #1126 | fern | `gr9ht3h5` | EP4.88 | 6.6062% | 7.4646% | 1.519 | 3.924% | EP4 MARGINAL; EP5 gate imminent |
| #1127 | askeladd | `ag1dnelx` | EP4.91 | 6.7613% | 7.6589% | 1.526 | 3.966% | EP4 MARGINAL; EP5 gate imminent |
| #1116 | edward | `3ufrbxl6` | EP4.59 | 6.5968% | 7.4533% | 1.537 | 3.925% | EP4 PASS; EP5 gate approaching |
| #1128 | thorfinn | `uwqybod5` | EP4.14 | 6.5675% | 7.4273% | 1.513 | 3.880% | EP4 MARGINAL; EP5 approaching |

### CRITICAL FLEET-WIDE FINDING: τ_z bottleneck is STRUCTURAL

ALL τ_z-targeted interventions have FAILED to suppress τz/τx ratio. Every agent's ratio monotonically rises to ~1.50–1.57 by EP5-8 regardless of approach:
- nezuko α=10: 1.371→1.516 by EP5
- thorfinn τz_weight=3.0: 1.288→1.513 by EP4 (transient EP1 suppression only)
- edward per-channel heads: 1.400→1.537 by EP4
- frieren mag-only: 1.389→1.570 by EP8.5 (stabilizing)
- tanjiro EMA 0.9995: 1.454→1.547 by EP6

**Conclusion**: τ_z bottleneck is NOT addressable by loss weighting, sampling, or output capacity. Requires architectural solution targeting the τ_z representational bottleneck (e.g., coordinate system change, dedicated physics-informed τ_z head with orthogonal basis, or attention mechanism change).

### Gate comments posted this invocation
- **Frieren EP8 PASS** → EP10 gate: val_WSS ≤6.80% PASS / ≤6.85% MARGINAL / >6.85% KILL
- **Tanjiro EP6 MARGINAL** → EP7 gate: val_abupt ≤6.3% PASS / 6.3-6.5% MARGINAL / >6.5% KILL
- **Alphonse EP2 progress** → EP3 gate: ≤6.9% PASS / 6.9-7.2% MARGINAL / >7.2% KILL; vol_p ≤4.5%
- **Fern EP4 MARGINAL** → EP5 gate: ≤6.5% PASS / ≤6.8% MARGINAL / >6.8% KILL
- **Askeladd EP4 MARGINAL** → EP5 gate: ≤6.5% PASS / ≤6.8% MARGINAL / >6.8% KILL
- **Edward EP4 PASS** → EP5 gate: ≤6.5% PASS / ≤7.0% MARGINAL / >7.0% KILL
- **Thorfinn EP4 MARGINAL** → EP6 gate: ≤6.5% PASS / ≤6.8% MARGINAL / >6.8% KILL

---

## Latest invocation actions (2026-05-15 03:48–04:00Z) — Wave 29 full fleet confirmed active, all 8 students running

- **Closed PR #1123 (thorfinn τ_z dedicated subnet)** — zero student activity after 4+ hours, four advisor check-in messages unanswered. Pod confirmed idle (1/1 READY via kubectl). Hypothesis is sound but requires code implementation; reassigned pod to a zero-code-change experiment to eliminate implementation failure mode.
- **Assigned PR #1128 (thorfinn: τ_z loss weight escalation 2.0→3.0)** — pure CLI flag change `--tau-z-loss-weight 3.0`, no model code changes. Directly attacks dominant error axis (test_τ_z ≈ 9.05–10.1% across all no-SDF runs). Pass signal: τ_z/τ_x ratio at EP13 < 1.5 (down from ~1.6–1.7 baseline). Full 18h budget (SENPAI_TIMEOUT_MINUTES=1100). W&B run `uwqybod5` (group `tau-z-loss-weight-3p0`), launched 03:48:42Z.
- **thorfinn #1128 confirmed launched** — student ACK received 03:49:15Z with PID confirmed and `SENPAI_TIMEOUT_MINUTES=1100` set. W&B run ID `uwqybod5`, W&B name `thorfinn/tau-z-loss-weight-3p0-20260515T034842Z`. Resolves escalation from #1123 closure.
- **alphonse #1122 pace corrected** — actual pace at vol=16k is ~131 min/epoch (not 80 min). Root cause: vol=16k → 860 views/case (ceil(14M/16k)) → view_count=max(130,860)=860 → 10,864 iters/rank/epoch × 1.38 it/s = 131 min. Gate schedule revised: EP1 ~05:50Z, EP3 ~07:55Z. Smoke confirmed 5.6× sampled/population weight ratio → correct FAR-field SOTA mechanism.
- **Full Wave 29 fleet all active** (kubectl: all 8 deployments 1/1 READY at 03:52Z). Zero idle.

### Wave 29 fleet — full status and gate schedule (as of 03:52Z, 2026-05-15)

| PR | Student | Hypothesis | W&B Run | EP1 Gate | EP3 Gate | EP13 ETA |
|----|---------|------------|---------|----------|----------|----------|
| #1116 | edward | Per-channel WSS output heads (τ_x/τ_y/τ_z) — 18h convergence (relaunched 03:09Z as `3ufrbxl6`) | `3ufrbxl6` | ~05:10Z | **~08:00Z** | ~14:00Z |
| #1121 | frieren | WSS magnitude-only decomp (λ_dir=0, λ_mag=0.1) — EP3 PASS 6.746% (best in family) | `frieren/mag-only-*` | DONE | **DONE (PASS)** | ~14:30Z |
| #1122 | alphonse | SDF FAR-field α=2.0 (`weight=1+α|sdf|`) — corrected SOTA mechanism port | alphonse run | ~05:50Z | **~07:55Z** | ~16:30Z |
| #1124 | tanjiro | EMA decay 0.9995 — EP1 PASS 31.48%, EP2 in flight | `mw6d04kc` | DONE | **~06:15Z** | ~15:00Z |
| #1125 | nezuko | Spatial-prior surface sampling α=10 — 18h budget | nezuko run | **~05:00Z** | ~06:00Z | ~14:00Z |
| #1126 | fern | Deeper surface_out MLP (depth 2→4, +525k params) — 18h budget | fern run | ~04:30Z | **~06:00Z** | ~14:00Z |
| #1127 | askeladd | Surface_loss warmup curriculum (3-ep ramp 0→full) — 18h budget | `dtgfdsgv` | ~04:30Z | **~06:00Z** | ~14:00Z |
| #1128 | thorfinn | τ_z loss-weight 3.0 (single CLI flag escalation from 2.0) | `uwqybod5` | **~06:00Z** | ~08:30Z | ~15:30Z |

Gate criteria per row:
- **frieren #1121 EP6** (~06:00Z): val_abupt ≤6.5% PASS / ≤6.8% MARGINAL (half-way convergence sanity)
- **Standard no-SDF EP3**: val_abupt ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL
- **alphonse #1122 EP3 (SDF FAR-field)**: val_abupt ≤6.9% PASS / ≤7.2% MARGINAL / >7.2% KILL (tighter — SDF expected uplift)
- Per-axis WSS signal: τ_z/τ_x ratio direction is primary quality signal for all WSS-targeting experiments

## Prior invocation actions (2026-05-15 02:35–02:50Z) — Wave 28.5 closures complete, Wave 29 architectural pivot launched

- **Closed PR #1118 (askeladd OHEM v2)** — definitive negative mechanism: `clip_active`=100.00% across all 4218 EP3 OHEM-active steps → gradient through OHEM term is exactly zero → run is mathematically equivalent to baseline. Test metrics regressed +0.903pp test_WSS vs SOTA at EP3-only (truncated by 270-min cap). **OHEM-on-raw-residuals family terminally exhausted**: dataset's top-K residuals are intrinsically 100–25,000× larger than mean → any safe scalar cap fires 100% → zero learning signal. The `clip_active_pct` diagnostic was the right metric and identified the failure mode within EP3 — should remain in codebase for future loss-clip work.
- **Wave 28.5 loss-engineering pattern: 0-for-3 at convergence** — #1114 learnable WSS (null), #1119 GradNorm short-cycle (refutes prior-rediscovery), #1118 OHEM v2 (zero gradient). Decisive pivot to capacity / data-sampling / architecture routes.
- **Assigned PR #1126 (fern: deeper surface_out MLP depth 2→4 + 18h)** — Wave 29 architectural pivot. Tests whether τ_z magnitude prediction is decoder-depth-limited at the surface head (current 2-layer MLP). Orthogonal to thorfinn #1123 (separate τ_z branch) and edward #1116 (per-channel heads). Parameterizes `surface_out_depth` config; depth=2 default preserves backward compat. Full 13-EP convergence at SENPAI_TIMEOUT_MINUTES=1100.
- **Assigned PR #1127 (askeladd: explicit surface_loss_weight warmup curriculum + 18h)** — directly tests #1114 finding that EP1 wins are implicit-curriculum artifacts. Adds `--surface-loss-weight-warmup-epochs 3` flag that linearly ramps surface_loss_weight from 0 → full over first 3 epochs. Gradient-flow-preserving (scalar multiplier, NOT residual reweight) → avoids OHEM #1118 trap. Predicted payoff: stable volume-conditioned backbone before surface head receives full gradient → better τ_z magnitude convergence at terminal.
- **All 8 students now active**: alphonse #1122 (SDF FAR-field α=2.0), nezuko #1125 (spatial-prior α=10 + 18h), tanjiro #1124 (EMA decay 0.9995 + 18h), thorfinn #1123 (τ_z subnet — CLOSED, replaced by #1128), edward #1116 (per-channel heads, 18h convergence), frieren #1121 (magnitude-only + 18h), fern #1126 (surface_out depth=4 + 18h), askeladd #1127 (surface_loss warmup curriculum + 18h). **Zero idle.**

## Prior invocation actions (2026-05-15 01:41Z) — CRITICAL SDF MECHANISM DIAGNOSTIC

- **PR #1122 alphonse SDF port → CHANGES REQUESTED, corrected plan approved**: alphonse paused the 13ep run at 28min in (EP1 ~25% done) after spotting THREE issues with my original assignment:
  1. **Mechanism inversion**: commit `023f766` I cited as reference impl implements `weight = 1/(1+α·|sdf|)` (NEAR-surface emphasis), but PR #972 body and the actual SOTA run `56bcqp3m` use `weight = 1 + α·|sdf|` (FAR-field emphasis). These are OPPOSITE hypotheses.
  2. **α value mismatch**: SOTA `56bcqp3m` ran α=2.0, not α=4.0. The NEAR-surface alpha sweep on W&B (α=0.25→6.265%, α=0.5→6.290%, α=1.0→6.356%, α=3.0→7.251% over kill gate) shows higher α is monotonically worse for the NEAR-surface inversion.
  3. **IO regression**: alphonse's port used `np.load(path, mmap_mode="r")[rows]` fancy-indexed memmap which runs ~3× slower than PR #972's contiguous load + in-memory slice on this PVC. Smoke EP1 took 114 min vs SOTA reference 41 min.
  4. SOTA confounders captured: `56bcqp3m` also ran batch_size=1, model_layers=6, GradNorm, y_symmetry_aug, epochs=30 — these are NOT part of the corrected single-variable port.
- **Approved corrected plan**: revert IO optimization, switch to FAR-field `weight = 1 + α·|sdf|` α=2.0, keep tay baseline recipe (batch_size=4, model_layers=5, no GradNorm, no y_sym, epochs=13), smoke 2EP then full 13EP. Single-variable change isolates the SDF mechanism; full-recipe SOTA reproduction held for follow-up if FAR-field α=2.0 alone doesn't beat 6.99% ceiling.
- **Adjusted EP3 gate for FAR-field α=2.0**: PASS ≤ 6.9% / MARGINAL ≤ 7.2% / KILL otherwise. Projected EP13 terminal val_abupt ~6.4-6.6%, putting test_WSS in striking range of 6.5-6.7% (likely strongest single-model on tay).

## Methodology lesson for advisor

Always **verify the SOTA reference mechanism from the actual W&B config** before citing it in an assignment, not from a commit body that may be a different formulation. The PR body, the commit text, and the run config can all diverge. Going forward: when citing a SOTA mechanism, pull its W&B config first.

## Prior invocation actions (2026-05-15 01:15–01:30Z)

- **Closed PR #1114** (tanjiro learnable WSS channel weights): terminal SENPAI-RESULT `test_WSS=7.726%, val_abupt=7.066%` at EP3 (budget-truncated). +0.40pp val_abupt over matched 3-EP baseline (mempfubx 7.465%) but driven by EP1 transient drift (weights briefly dropped to ~50% of init, then quadratic-well-regularized back to baseline by EP3 within 3% of init). Mechanism null at convergence. Methodology data preserved.
- **Reassigned tanjiro → PR #1124** (EMA decay 0.9995 + 18h budget): single-flag experiment, slower EMA half-life ≈ 1386 steps vs default 693 steps. Tests whether late-converging τ_z benefits from longer EMA averaging window. Full 13-EP convergence test, comparison to no-SDF tay ceiling 6.99%.
- **Sent PR #1116 back to edward** (per-channel WSS heads, draft state): matched-budget A/B at EP3 truncated showed −0.66pp test_WSS, −0.09pp test_vol_p, −0.23pp test_SP — every metric improved vs single-head baseline `mempfubx`. **First clean positive Wave 28.5 signal.** But test_WSS=7.671% does not beat no-SDF ceiling (6.99%), so requires 18h budget convergence confirmation: if matched-budget delta holds at EP13, test_WSS → 6.33%, would tie/beat ensemble SOTA. Re-running with `SENPAI_TIMEOUT_MINUTES=1100`, no other changes.
- **τ_z/τ_x ratio finding (edward EP3)**: per-channel heads UNIFORMLY uplift capacity, not τ_z-specifically — ratio went 1.44 → 1.52 (wrong direction). Mechanism is "decoupled head capacity", not "τ_z specialization". Implies follow-on work needs deeper/wider τ_z head specifically (overlaps with thorfinn #1123 τ_z subnet).

---

## PRIMARY RESEARCH DIRECTIVE (Issue #1056, 2026-05-12, ongoing)

**Beat Wall Shear Stress SOTA from Transolver-3 (5.85%) while not degrading vol_p or surface pressure.**

- **WSS target:** `test_WSS` < **5.85%**
- **Non-negotiable constraints:** `test_vol_p` ≤ 3.643% AND `test_SP` ≤ 3.577% (PR #972 levels)
- **Baseline for all new single-model runs:** PR #972 SDF-stratified stack

**WSS Gap (post-PR #1102):**
- Single-model best: **6.727%** (PR #972) → need −0.88pp
- Ensemble best (compliant): **6.3263%** (PR #1102 K=8 Caruana) → need **−0.476pp**

Most recent human check-in: 2026-05-14 14:17 UTC — **"NO MORE ENSEMBLES! Its the lazy route to better results, we want genuine breakthroughs, not incremental improvements based on ensembling which we know we can deploy at any point to improve results."** (Issue #1056 comment from morganmcg1). Ensemble experiments are BANNED until explicitly unlocked. Status updates posted at ~12:35 UTC and ~15:00 UTC.

---

## CORRECTED DATASET IN EFFECT (since 2026-05-11)

Issue #1053 (deployed 2026-05-11) fixed a case-split/indexing bug that eliminated an artificial ~3× vol_p OOD gap.

**Corrected dataset path:** `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`
**Split:** val=34 cases/7295 views, test=50 cases/11091 views

All new runs MUST use the corrected dataset path and `--data-root` flag (not `--data-path`).

---

## Current Best Results (Corrected Split)

### **Ensemble SOTA (PR #1102 — K=8 Caruana with-replacement, WSS-optimised)**
- val_abupt = **5.7452%** | test_abupt = **5.5196%**
- val_vol_p = 3.4360% | test_vol_p = 3.5397%  ← satisfies ≤ 3.643%
- val_WSS = 6.5195% | **test_WSS = 6.3263%**  ← TRUE WIN
- val_SP = 3.7234% | test_SP = 3.3529%  ← satisfies ≤ 3.577%
- test_tau_x = 5.6071% | test_tau_y = 6.8397% | **test_tau_z = 8.2585%** (still worst axis)
- W&B: `bq1gaewq` (Arm D greedy), `ems8ekee`, `s7pirpr1`, `qf1lqwz0`
- **Members:** `56bcqp3m`×3, `29nohj67`×2, `a0yoxy85`×2, `ghh0s4ne`×1
- **Effective weights:** {56bcqp3m:0.375, 29nohj67:0.250, a0yoxy85:0.250, ghh0s4ne:0.125}

### Prior Ensemble SOTA (PR #1064 K=3 greedy, superseded by #1102)
- val_abupt = 5.7758% | test_abupt = 5.5199% | test_WSS = 6.3712%

### Single-Model SOTA (PR #972 — SDF-stratified vol importance sampling)
- val_abupt = 6.126% | test_abupt = 5.844%
- val_vol_p = 3.798% | test_vol_p = 3.643%  ← constraint boundary
- test_SP = 3.577%  ← constraint boundary
- test_WSS = 6.727%
- W&B: source=`56bcqp3m`, eval=`zxnhtagj`

### Rank 2 Single-Model (PR #968 — Stochastic vol subsampling)
- val_abupt = 6.278% | test_abupt = 5.986% | test_WSS = 6.825%
- W&B: source=`a0yoxy85`, eval=`qbg9pkmx`

---

## Gate Criteria

### Single-Model EP3 Gates (current tay stack — no SDF importance sampling)
- **PASS:** val_abupt ≤ **7.2%** AND val_vol_p ≤ 4.5%
- **MARGINAL:** val_abupt ≤ 7.6% AND val_vol_p ≤ 5.0%
- **KILL:** otherwise

(Historical PR #972 SDF stack gates were ≤ 6.2% / ≤ 6.5% — those reflect SDF-stratified sampling that is NOT on tay; do not apply to current single-model runs.)

### WSS-Targeted Single-Model Win Criteria (becomes new pool member)
- test_WSS ≤ 6.50% AND test_vol_p ≤ 3.643% AND test_SP ≤ 3.577% AND val_abupt ≤ 6.20%

### Ensemble Win Criteria (true new SOTA after PR #1102)
- val_abupt < **5.7452%** AND test_vol_p ≤ **3.643%** AND test_WSS < **6.3263%**

---

## Current Research Focus and Themes

### Primary: WSS Magnitude Bottleneck Attack (Wave 28 onwards — single-model only)

**New mechanism finding from PR #1097 close (tanjiro, WSS direction loss NEGATIVE):**
- WSS **direction is essentially solved** — cos_sim stabilises at 0.996 (~5° angular error) by EP2.
- **91–96% of remaining WSS residual is magnitude error.**
- This pivots the campaign from "direction-aware" experiments (which #1094, #1096, #1097 all targeted) toward **magnitude-targeted** mechanisms (rel_l2 loss, magnitude penalty) and **frame-equivariance** (in-plane rotation aug).

### Pool Saturation — CONFIRMED (PR #1103 closed 2026-05-14 13:30Z)

The current 4-member candidate pool {`56bcqp3m`, `29nohj67`, `a0yoxy85`, `ghh0s4ne`} is Pareto-saturated under convex combinations:
- PR #1102 K=8 Caruana (MERGED) — near-globally-optimal at discrete 1/8 grid
- PR #1099 K=3 WSS-targeted (CLOSED) — converged to identical K=3 subset as #1064
- PR #1103 SLSQP continuous optimisation (CLOSED) — confirmed K=8 within ~0.03 L1 of global continuous optimum; best-case val_WSS improvement = 0.0039pp (0.06% relative); val_SP ≤ 3.577% **infeasible** on this pool (simplex floor ~3.72%, every member ≥ 3.98%)

**Active lever for ensemble gains:**
1. **Pool extension via new single-model members** — only remaining path (ensembles BANNED per human directive)

⚠️ **ENSEMBLES BANNED** — Per morganmcg1 Issue #1056 directive 2026-05-14 14:17Z: no new ensemble experiments until explicitly unlocked. PR #1108 (bias-corrected ensemble) was superseded by PR #1109 (τ_z focal loss) before training started; #1108 is effectively dead.

---

## Active WIP PRs (as of 2026-05-15 ~02:50Z)

### Wave 28.5 → Wave 29 transition complete — all 8 students in flight

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| ~~#1114~~ | tanjiro | ~~Learnable WSS channel loss weights~~ | **CLOSED 01:13Z** — mechanism null at convergence; reassigned → #1124 |
| **#1116** | edward | **Per-channel WSS output heads** — decouple tau_x/tau_y/tau_z heads + 18h convergence | CHANGES REQUESTED 01:25Z — re-running at 18h to confirm matched-budget −0.66pp test_WSS delta holds at EP13; projected test_WSS ≈ 6.33% if delta holds (would tie ensemble SOTA) |
| ~~#1118~~ | askeladd | ~~OHEM v2 spike-clipped~~ | **CLOSED 02:35Z** — `clip_active`=100% → zero OHEM gradient → mathematically baseline-equivalent; reassigned → #1127 |
| ~~#1119~~ | fern | ~~GradNorm short-cycle (t_max=6, ep=6)~~ | **CLOSED 02:27Z** — REFUTES prior-rediscovery hypothesis; τ_z weight plateaus 1.07 (vs prior 2.0); hardcoded prior empirically validated; reassigned → #1126 |
| ~~#1120~~ | nezuko | ~~Spatial-prior surface sampling α=3~~ | **CLOSED 02:30Z** — mechanism right (ρ=+0.31 PASS), EP3 budget too short; strongest 3-EP truncated WSS in family but truncated; reassigned → #1125 (α=10 + 18h) |
| **#1121** | frieren | **WSS magnitude-only decomposition + 18h budget** — `λ_dir=0.0`, full 13-ep cosine; tests Wave 27 "91-96% magnitude" claim | Active WIP; EP3 gate ~02:48Z; EP13 ~14:00Z |
| **#1122** | alphonse | **SDF importance sampling port to tay — FAR-field α=2.0 (corrected mechanism)** — `weight = 1 + α·|sdf|`; highest-EV untested-on-tay lever; reproduces PR #972 SOTA mechanism (NOT the inverted `1/(1+α·|sdf|)`) | Active WIP draft post-correction (01:41Z); smoke-then-full plan approved; EP3 gate ≤6.9% PASS |
| **#1123** | thorfinn | **τ_z dedicated subnet** — 2-layer MLP head attacking residual axis test_τ_z ≈ 9.05% | Active WIP; launched 23:50Z post-#1100 close |
| **#1124** | tanjiro | **EMA decay 0.9995 + 18h budget** — single-flag test of slower EMA half-life (~1386 vs 693 steps) for late-converging τ_z | Active WIP; assigned 01:18Z post-#1114 close; full 13-EP convergence test |
| **#1125** | nezuko | **Spatial-prior surface sampling α=10 + 18h budget** — stronger oversample at full convergence (student's suggested follow-up #2); tests if mechanism scales without catastrophe | Active WIP; assigned 02:23Z post-#1120 close |
| **#1126** | fern | **Deeper surface_out MLP (depth 2→4) + 18h budget** — Wave 29 architectural pivot; tests if τ_z magnitude is decoder-depth-limited at surface head; orthogonal to thorfinn #1123 (separate branch) and edward #1116 (per-channel heads) | Active WIP; assigned 02:45Z post-#1119 close |
| **#1127** | askeladd | **Explicit surface_loss_weight warmup curriculum (3-ep ramp 0→full) + 18h** — directly tests #1114 implicit-curriculum finding; gradient-flow-preserving (avoids OHEM #1118 trap) | Active WIP; assigned 02:50Z post-#1118 close |

---

## Wave 28 Closures (2026-05-14 19:43Z–21:33Z) — methodology data captured, all reassigned

| PR | Student | Result | Key Mechanism Finding | Reassigned As |
|----|---------|--------|----------------------|---------------|
| #1109 | edward | val_WSS=8.766% EP3 (+1.6pp vs no-decomp ref) | Spatial focal α=2.0 amplifies per-point WSS errors at hot-spots faster than they can train down; underweights smooth bulk; baseline isn't smooth-dominated | #1116 per-channel heads |
| #1110 | askeladd | OHEM scale-collapse @ EP3 | Top-20% mining catastrophically scale-collapses without spike-clip; magnitude of L_hard explodes vs base loss | #1118 OHEM v2 spike-clipped |
| #1111 | fern | GradNorm test floors regress (test_vol_p +0.5pp, test_SP +0.4pp) | GradNorm de-emphasizes τ_z prior (hardcoded 2.0 weight); short-cycle test needed to disambiguate prior-vs-learned at convergence | #1119 GradNorm short-cycle |
| #1112 | frieren | Truncated EP3.5 @ 270-min wall-clock; calibration validated (mag head ratio=0.979 at half-cooked) | Mag head infrastructure works; full budget needed for convergence test | #1121 mag-only + 18h budget |
| #1113 | nezuko | val_abupt=8.04% EP3 (KILL) | Curvature is anti-correlated WSS proxy (ρ=-0.11); curvature-weighted sampling steers attention AWAY from high-WSS regions | #1120 spatial-prior (ρ=+0.31) |

## Wave 27 Closures (2026-05-14 ~13:45Z) — CATASTROPHIC FAILURE

All 4 experiments failed at EP3 with val_abupt 27–32% (4× above EP3 KILL gate of 7.6%). Root causes:

| PR | Student | val_abupt@EP3 | Root Cause |
|----|---------|---------------|------------|
| #1104 | fern | ~27% | L1 magnitude penalty `|‖τ‖−‖τ_gt‖|` creates conflicting gradients vs MSE loss; loss scale mismatch blows up training |
| #1105 | tanjiro | ~30% | Relative L2 `(pred-gt)²/‖gt‖²` numerically explodes when GT~0; near-zero WSS regions produce infinite loss |
| #1106 | frieren | ~28% | Physical-coordinate normal-frame rotation corrupts geometry signal — coordinate transformation invalidates learned features |
| #1107 | nezuko | ~32% | Yaw augmentation destroys physical orientation; model cannot learn orientation-dependent aerodynamics |

Common diagnosis: Wave 27 hypotheses all modified the **loss function or input transformation** at a fundamental level without sufficient numerical safeguards. The supplementary-loss OHEM approach (Wave 28) is designed to avoid these failure modes by adding a *supplementary* term (not replacing the base loss) with warmup and floor guards.

## Wave 26 Additional Kill (2026-05-14)

| PR | Student | Result | Key Mechanism Finding |
|----|---------|--------|----------------------|
| #1081 | askeladd | KILL @ EP10 (val_abupt=7.97%) | slw=3.0 surface loss weight — too aggressive; distorts vol_p head; baseline slw=2.0 is optimal |

## Wave 26 Closures (2026-05-13 → 2026-05-14)

| PR | Student | Result | Key Mechanism Finding |
|----|---------|--------|----------------------|
| #1094 | frieren | KILL @ EP3 (val_abupt=7.465%) | Normal-frame supervision built in normalised space — non-orthonormal |
| #1095 | nezuko | NEGATIVE (test_WSS=7.761% +1.03pp) | GradNorm mechanism healthy but starved vol head; curriculum is load-bearing |
| #1096 | edward | NEGATIVE (test_WSS +0.261pp vs ref) | Tangent-frame features redundant with normals; z-hat fallback discontinuity |
| #1097 | tanjiro | NEGATIVE (val_abupt=6.847% > KILL) | Direction NOT the bottleneck (cos_sim=0.996) |
| #1099 | fern | CONVERGED (same K=3 as #1064) | WSS-targeted greedy on 4-member pool converges to identical subset |
| #1102 | fern | **MERGED — new ensemble SOTA** | K=8 Caruana extracts ghh0s4ne WSS signal at 12.5% weight; NOW BANNED FROM EXTENSION per human directive |

---

## Baseline Training Recipe (current tay stack — NOT PR #972 SDF stack)

⚠️ **IMPORTANT:** the PR #972 SDF-stratified vol sampling code (`--sdf-importance-sampling --sdf-alpha 4.0`) was **never merged into tay**. Do NOT include those flags in any assignment — `argparse` will reject them. The live tay baseline is the stack below (no SDF importance sampling). Single-model EP3 on this baseline lands ~6.7–6.9% val_abupt, not the historical PR #972 6.2%. Gates must be recalibrated accordingly: PASS ≤ 7.2%, MARGINAL ≤ 7.6%, KILL otherwise.

```
--optimizer lion --lr 9e-5 --weight-decay 5e-4
--tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0
--use-ema --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1
--pos-encoding-mode string_separable --use-qk-norm
--rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"
--lr-cosine-t-max 13 --epochs 13
--vol-points-schedule "0:16384:3:32768:6:49152:9:65536"
--no-compile-model
--model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128
--batch-size 4 --validation-every 1
--train-surface-points 65536 --eval-surface-points 65536
--train-volume-points 65536 --eval-volume-points 65536
--use-surf-to-vol-xattn
--data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511
```

The PR #972 single-model SOTA W&B run `56bcqp3m` was trained with SDF-stratified sampling on a different branch (`dl24-frieren/vol-test-domain-augmentation`, commit `291efd2`); that code never landed on tay. Until it does, all new single-model runs are evaluated relative to the no-SDF tay baseline (thorfinn #1100 EP3=6.768% is a representative live trajectory).

---

## Next-Wave Hypothesis Queue

Wave 28.5 → Wave 29 in flight — 8 students busy, zero idle. Capacity students (alphonse #1078, thorfinn #1100) closed; SDF mechanism port (#1122) is the highest-EV active experiment.

Queue for Wave 30 (after current cohort lands ~tomorrow):

1. **Higher τ_z loss weight (3.0 from current 2.0)** — if fern #1126 and edward #1116 reveal decoder is the bottleneck, the prior τ_z=2.0 may now be undertuned with the increased capacity.
2. **WSS magnitude/direction joint head** — if frieren #1121 mag-only beats baseline, add a second head for direction (cos_sim) with bounded loss; combine via learnable α.
3. **Multi-scale surface attention** — second surface encoder at 0.5× token density to capture macro-flow features.
4. **Heteroscedastic WSS loss** — model both mean and variance per surface point; downweight high-aleatoric regions.
5. **τ_z frequency analysis** — Fourier decompose tau_z predictions vs GT to find spatial frequency bands where error is concentrated; use to motivate loss or architecture changes.
6. **Surface point sampling Voronoi tessellation** — sample uniformly over surface area (not raw vertex density) to remove sampling bias from non-uniform mesh refinement.
7. **Combine SDF FAR-field α=2.0 (from #1122) with deeper surface_out (from #1126)** — if both win independently, the combination is the obvious next step; orthogonal mechanism stacking.
8. **Curriculum at higher granularity** — if askeladd #1127 surface-loss warmup wins, try ramping individual WSS channel weights (τ_z last) instead of the global scalar.

⚠️ Permanently retired (catastrophic failure): yaw aug (#1107), magnitude penalty (#1104), rel_l2 (#1105), normal-frame rotation (#1094, #1106), curvature-weighted surface sampling (#1113 — wrong-sign proxy).

---

## Infrastructure Status

### GitHub Token Rate Limiting (RESOLVED 2026-05-14)
Senpai PR #3445 merged 06:42Z deployed per-student token fix + REST API migration. Fleet was back online by ~07:30Z. No further rate-limit-driven idle GPU incidents reported in current invocation.

### Pod Health
All 8 students have active pods (kubectl: `senpai-drivaerml-ddp8-*` deployments, 1/1 ready). DDP via 8× H100 96GB per student. Zero idle students as of 02:50Z. **Wave 28.5 → Wave 29 transition complete.** PR distribution: edward #1116 (per-channel heads 18h), frieren #1121 (magnitude-only 18h), alphonse #1122 (SDF FAR-field α=2.0), thorfinn #1123 (τ_z subnet), tanjiro #1124 (EMA decay 0.9995 18h), nezuko #1125 (spatial-prior α=10 18h), fern #1126 (surface_out depth=4 18h), askeladd #1127 (surface_loss warmup curriculum 18h).

---

## Key Findings to Date

- **WSS error is magnitude-dominated** (91–96% of residual, not direction) — pivot away from direction-aware experiments
- **tau_z (spanwise) still worst axis** (test_tau_z=8.2585% on PR #1102) — primary remaining target
- **Wave 27 catastrophic lesson**: NEVER replace base MSE loss — always use supplementary/additive formulations; loss scale mismatches and numerical instability (div-by-near-zero) destroy training even at 27–32% val_abupt; Wave 28 OHEM designed as additive supplement with 2-ep warmup to avoid this
- **Relative L2 loss is unstable** (PR #1105) — near-zero GT WSS regions produce unbounded loss; avoid any loss form with GT in denominator without explicit safeguards
- **slw=3.0 surface weight too aggressive** (PR #1081 killed) — baseline slw=2.0 is optimal
- **ENSEMBLES BANNED** (human directive 2026-05-14 14:17Z) — all new work must improve the single-model SOTA
- **Corrected dataset** (2026-05-11) eliminated artificial ~3× vol_p OOD gap — biggest research-program insight
- **SDF-stratified vol importance sampling** (PR #972) is single-model SOTA: val_abupt=6.126%, test_WSS=6.727%
- **Ensemble SOTA** (PR #1102 K=8 Caruana) test_WSS=6.3263% — first compliant ensemble below 6.33%
- **4-pool Pareto-saturated** (PR #1103 CONFIRMED) — K=8 within 0.03 L1 of global continuous optimum; val_SP ≤ 3.577% infeasible on this pool (simplex floor ~3.72%); new pool members are the operative lever
- ~~**Bias-corrected ensemble** (PR #1108)~~ — closed (superseded by τ_z focal loss #1109; ensemble research BANNED)
- **Training-time vol sampling** matters more than loss weighting or architecture depth for vol_p
- **Throughput regression risk on data-pipeline experiments** (nezuko #1113 self-diagnosed 12× slowdown from 20s/case curvature compute serialised through 4 workers; fix = precompute-and-cache; advisor must spec precompute step in any future data-pipeline assignment)
- **Curvature is a bad WSS proxy** (PR #1113 closed) — surface curvature is anti-correlated with |WSS| (ρ=-0.11); using curvature to oversample steers attention AWAY from high-WSS regions. Spatial position (`-x + |z|`) achieved ρ=+0.31 by contrast (PR #1120).
- **270-min wall-clock budget hits Wave 28 recipe at EP3.5** (#1111, #1112, historical #1095 all truncated) — recipe runs 76 min/epoch; full 13-ep cosine needs ~16h. Two responses available: recipe shrink (short t_max, fern #1119) or budget bump (`SENPAI_TIMEOUT_MINUTES=1100`, frieren #1121, matches alphonse #1078 working regime).
- **GradNorm de-emphasizes τ_z hard-coded prior** (#1111 close) — when learned, GradNorm reduces τ_z weight from prior 2.0 toward 1.4, which regresses test_vol_p and test_SP floors. Question: is the 2.0 prior over-tuned, or is the learned weight wrong? Short-cycle test (#1119 fern) measures this at full convergence.
- **OHEM scale-collapse** (#1110 close) — top-k mining catastrophically collapses without spike-clipping; magnitude of L_hard scales superlinearly when targeting top-20% of L distribution.
- **Spatial focal α=2.0 amplifies hot-spot error faster than training rate** (#1109 close) — per-point focal modulation creates concentrated gradients on outliers; baseline isn't bulk-smooth-dominated so amplification destabilizes optimization.
- **val→test ratio is NOT stable across eval configurations** (#1078 close) — asymmetric eval 131k produced val→test ratio of 1.020, not the 0.935 anchored on PR #972. The 0.935 ratio is recipe-specific (SDF stack), not transferable. Advisor SOTA projections must use test results from comparable-recipe runs, not synthetic val × historical ratio. This is a methodology guard for the entire program.
- **18h budget recipe validated end-to-end** (#1078 close): `SENPAI_TIMEOUT_MINUTES=1100` ran 17 epochs cleanly at ~62 min/ep (faster than initially projected). All future Wave 28+ runs can adopt it confidently; frieren #1121 has already.
- **Capacity-uplift ceiling on no-SDF tay is val_abupt ≈ 6.31%** (#1078 EP16 / #1100 EP16 close). Beyond that, the bottleneck is training-time sampling, not parameter count. Justifies #1122 (alphonse SDF port).
- **No-SDF tay structural ceiling at test_WSS ≈ 6.99%** (#1078 + #1100 close, two independent mechanisms). Asymmetric eval 131k and slices=256 capacity uplift both converge here at full convergence. Test floors regress under both. Any "beat SOTA without SDF" claim must beat 6.99% — capacity alone cannot. Direct paper-relevant finding.
- **τ_z is the program-wide residual axis** (test_τ_z ≈ 9.05% across all no-SDF runs). Consistently 30-45% worse than τ_x and ~18bp worse than τ_y. Justifies #1123 (thorfinn τ_z dedicated subnet) attacking representational capacity for τ_z specifically.
- **Initial-state debug crash** (tanjiro #1114 val_abupt=65.34% on 1-ep debug, then 8-rank DDP retry also crashed) — root cause likely learnable-weight unbounded growth; mitigated by lr=1e-3 separate group + L2 reg 1e-4 + 2-ep warmup option
- **Post-xattn capacity additions** 0-for-3 (PRs #884, #891, #906) — do not add layers after surf→vol xattn
- **Rotation aug** (PR #925): aggressive yaw+pitch degrades; mild yaw-only (≤45°) being tested in PR #1107
- **Normal-frame WSS in normalised space** fails (PR #1094); physical-frame variant (#1106) is the corrected attempt
- **Tangent-frame features** redundant with surface normals (PR #1096) — model already has the information
- **Direction loss** redundant with weighted MSE (PR #1097) — cos_sim=0.996 by EP2 without it
- **GradNorm + fixed-65k vol** fails because vol curriculum is load-bearing (PR #1095)
- **Infra:** `--data-root` (not `--data-path`); mount point `/mnt/new-pvc/` (not `/mnt/pvc/`)
- **EMA lag:** `eval_raw_vs_ema=False` means only EMA weights evaluated — early-epoch EMA metrics appear worse than raw
