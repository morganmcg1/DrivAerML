# Tanjiro Next Experiment — H351: Normal-Relative Geometric Slice Bias (NGSB)
## Generated: 2026-06-01 15:55Z | Advisor: tay

---

## Context: Why Now

The WSS_z bottleneck (test_WSS_z = 8.6175% vs Transolver-3 target 5.85%) has been confirmed **representational, not gradient-budget** (H345: cos(g_wss, g_pres) = +0.30, 99.55% positive; H341/H346: all loss-reweight arms regress monotonically). The 4-axis in-flight attack (H347 physics priors, H348 curvature inputs, H349 arcsinh transform, H350 FiLM decoder) covers INPUT, OUTPUT-TRANSFORM, OUTPUT-DECODER, and PHYSICS-CONSTRAINT axes.

One axis remains entirely unattacked: the **attention mechanism's physics-state assignment**. The Transolver's `slice_logits` are computed purely from learned content projections — there is no geometric structural prior influencing which tokens co-cluster into the same "physics state." Surface normals (channels 3:6 of SURFACE_X_DIM=7) are available as input features but are consumed only through the flat `LinearProjection` embedding before attention; they are never used as pairwise or per-token relational geometry signals inside `TransolverAttention`.

---

## H351: Normal-Relative Geometric Slice Bias (NGSB)

### One-Line Summary

Add a tiny learned linear projection from per-token surface normals to a per-head additive bias on the Transolver's slice-assignment logits, so tokens with similar surface orientation preferentially cluster into the same physics-state slice.

---

### Mechanism

**Current slice-assignment path (read-only):**

```python
# TransolverAttention.create_slices(), line 292
x_mid = self.in_project_x(x)           # x is already embedded; normals are one of 7 input features
slice_logits = self.in_project_slice(x_mid) / self.temperature
slice_weights = F.softmax(slice_logits, dim=-1)   # shape: [B, H, N, num_slices]
```

`x` at this point is the hidden representation after the input embedding; the geometric normal signal is already mixed with xyz and area into a single vector. The slice assignment is purely content-driven.

**Proposed modification:**

Pass the raw surface normals `n_i ∈ ℝ³` (per-token unit normal, available as raw geometry features before the embedding) as an auxiliary structural tensor alongside the hidden state `x`. In `TransolverAttention.__init__` add:

```python
self.normal_slice_bias = nn.Linear(3, num_heads, bias=False)
# Weight init: zeros (residual/additive, no effect at init)
nn.init.zeros_(self.normal_slice_bias.weight)
```

In `create_slices`, modify the logit computation:

```python
# normals: [B, N, 3] — raw surface normals, unit-vector, NOT the embedded x
normal_bias = self.normal_slice_bias(normals)         # [B, N, num_heads]
normal_bias = normal_bias.permute(0, 2, 1).unsqueeze(-1)  # [B, num_heads, N, 1]
# Broadcast across num_slices: each head gets a token-specific additive offset
slice_logits = self.in_project_slice(x_mid) / self.temperature + normal_bias
slice_weights = F.softmax(slice_logits, dim=-1)
```

**Parameter overhead**: `3 × num_heads` = 3 × H weights (typically 3×8=24 or 3×16=48 parameters). Verified <<+1% of model capacity.

**Zero-initialisation guarantees**: at init, `normal_bias = 0`, so the model starts identical to H336 SOTA. The bias grows only if it reduces training loss — making this a residual structural prior rather than a hard geometric constraint.

### Why Surface Normals Target WSS_z Specifically

WSS_z = τ_z = the z-component of wall shear stress. For a surface element with normal `n`:
- By definition, τ = ∥viscous_stress_tensor · n − (n · viscous_stress_tensor · n)n∥ (shear tangent to surface)
- The z-component τ_z is maximally sensitive to surface elements where the normal has a large x or y component (faces oriented toward/away from the z-axis freestream)
- These surface elements are geometrically coherent — they form blade-like strakes, door shutlines, underfloor channels: regions where normal direction is the primary discriminator of aerodynamic loading

Current physics-state clustering is content-based: the model must infer from the embedded feature representation alone which tokens are "side-surface shear" tokens vs "underbody shear" vs "rooftop" tokens. A geometric normal bias lets the model **enforce** that tokens facing the same geometric direction cluster together, giving τ_z a more homogeneous representational group to predict from.

This is directly analogous to the GALE mechanism in GeoTransolver (NVIDIA, arxiv 2512.20399, Dec 2025): that work injects multi-scale ball-query geometric context into every Transolver block as cross-attention, reporting ~4.9% WSS relative L1 on DrivAerML test vs our current 6.6%. The NGSB is architecturally lighter (no cross-attention, no kNN ball queries, ~24 parameters vs thousands) but targets the same structural gap: Transolver's content-driven slice assignment ignores geometry.

UniAero (OpenReview ICLR 2026) similarly motivates Geometry-aware Position Encoding (GPE) for automotive WSS prediction, noting that geometry-blind attention creates systematic errors on high-curvature and geometrically-distinctive regions — exactly the WSS_z pathology pattern.

---

### Implementation Details

**Where to pass normals:**

The `SurfaceTransolver.forward()` path currently constructs `x = self.input_embedding(surface_x)` where `surface_x[:, :, 3:6]` are the surface normals. The cleanest implementation passes `surface_x[:, :, 3:6]` (or a pre-normalized copy) as an auxiliary tensor through `TransformerBlock.forward()` → `TransolverAttention.create_slices()`. 

Alternatively — and even simpler to implement — project the raw normals *before* embedding to the normal bias and cache it at the start of `SurfaceTransolver.forward()`, then pass it as a static per-layer bias tensor into each `TransformerBlock`. Since `nn.Linear(3, num_heads, bias=False)` is shared across all layers, this needs only one matrix multiply per forward pass.

**Shared vs per-layer bias projection:**

- **Ablation A (recommended first):** single shared `nn.Linear(3, num_heads)` for all layers. 24 parameters. Tests the mechanism with minimal parameter overhead.
- **Ablation B:** per-layer `nn.Linear(3, num_heads)` (one per TransolverAttention block, depth × 24 params). If depth=12, that is 288 parameters. Still <<+1%.

Start with Ablation A as the diagnostic gate.

**Hyperparameters to match H336:**

All training hyperparameters should be identical to H336 (the cosine-tail finetune that produced the SOTA). The only change is the addition of `normal_slice_bias` weights. Because they are zero-initialized, the model starts at H336's initial checkpoint and can be loaded via standard checkpoint resume without weight mismatch (add the new key with zeros to the checkpoint dict before loading, or handle via `strict=False`).

**Checkpoint loading:**
```python
checkpoint = torch.load(h336_checkpoint_path)
# Insert zero-initialised normal_slice_bias weights
for layer_idx in range(depth):
    key = f"transformer.blocks.{layer_idx}.attention.normal_slice_bias.weight"
    if key not in checkpoint["model_state_dict"]:
        checkpoint["model_state_dict"][key] = torch.zeros(num_heads, 3)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
```

**Diagnostic gate (4–6h first):**

Train for ~4h on a reduced budget (e.g. 3 epochs of cosine tail) starting from H336 checkpoint with the normal bias inserted. Check:
1. `train/loss_wss_z` curve vs baseline — does it decrease faster or reach a lower plateau?
2. Slice-weight entropy per-head — do any heads show structurally lower entropy (more deterministic assignment) on WSS_z-heavy faces?
3. Val metrics at the diagnostic checkpoint vs H336 at same epoch count.

If no movement in WSS_z train loss at the diagnostic gate → mechanism is not alive at this budget → close.

---

### What Makes This Orthogonal to H347–H350

| Axis | Mechanism | What It Modifies |
|---|---|---|
| H347 nezuko | Physics constraint (τ⊥n + kNN smoothness loss) | Training objective / auxiliary loss |
| H348 fern | Curvature input features (H, K, k1, k2) | Input channels (new geometric features) |
| H349 frieren | arcsinh target transform | Output-space regression target |
| H350 askeladd | FiLM per-axis decoder conditioning | Output decoder capacity/conditioning |
| **H351 (proposed)** | Normal-relative slice bias | **Attention mechanism: slice assignment logits** |

H347 adds a training-time auxiliary loss (does not change the forward pass at inference). H348 adds new input channels. H349 transforms the regression target. H350 modifies the output decoder. H351 is the only proposal that touches the **internal attention mechanism** — specifically the physics-state clustering that controls which tokens inform each other during self-attention. These axes are structurally non-redundant.

---

### Falsification Conditions

**Mechanism is dead** if either:
1. At the 4–6h diagnostic gate, `train/loss_wss_z` does not decrease faster than the H336 baseline trajectory. This would indicate the model's content representation already captures normal-direction information implicitly and the explicit bias adds no signal.
2. `slice_weight_entropy` per-head stays approximately constant across surface-normal orientations (bias learns near-zero weights) — the model rejects the structural prior.

**Mechanism succeeds** if:
1. Diagnostic gate shows WSS_z train loss improvement → proceed to full 20-24h finetune.
2. Full finetune achieves val_cal < 5.8978% AND test_WSS_z improvement vs H336's 8.6175%.

---

### Expected Upside and Risk

**Expected upside:**
- GeoTransolver (full geometry cross-attention, orders of magnitude more parameters) achieves test_WSS ~4.9% vs our 6.6% — a 170bp gap. The NGSB is a far simpler geometric prior but addresses the same structural weakness. A 50–100bp improvement in test_WSS_z seems plausible if the mechanism is alive, which would translate to ~10–20bp val_cal improvement (based on H336's WSS_z fraction of the composite metric).
- If the 277bp WSS_z gap vs Transolver-3 target (8.6175% → 5.85%) is even partially representational-attention-resident, this is the minimally invasive fix.

**Risk profile:** Low. Zero-init guarantees no regression at epoch 0. The diagnostic gate costs only 4–6h GPU time before committing to the full 20-24h finetune. Param overhead is negligible (24 parameters). The mechanism is mechanistically grounded in the GeoTransolver and UniAero literature.

---

### GPU Budget

| Phase | Duration | Wall Clock |
|---|---|---|
| Phase A: diagnostic gate | 4–6h | EP13 cosine-tail, 3 epochs |
| Phase B: full finetune | 20–24h | EP13–EP15 standard budget |

DDP: `torchrun --nproc-per-node=8`
Total if Phase A passes: ~26–30h

---

### Key External Evidence

1. **GeoTransolver** (NVIDIA, arxiv:2512.20399, Dec 2025) — "Geometry-Aware Long-range Efficient Transformer." Injects multi-scale ball-query geometric context into every Transolver block as cross-attention. Benchmarked on DrivAerML: test_WSS ~4.9% relative L1. Shows geometry injection into Transolver's internal mechanism (not input features or loss) closes WSS error by a large margin. https://arxiv.org/abs/2512.20399

2. **UniAero** (OpenReview ICLR 2026) — Geometry-aware Position Encoding (GPE) for automotive aerodynamics. Notes geometry-blind attention creates systematic WSS prediction errors on geometrically-distinctive regions. Withdrawn but architecture insight stands. https://openreview.net/forum?id=... (see ICLR 2026 proceedings)

3. **Transolver** (Wu et al., 2024, NeurIPS) — original physics-informed slice-based transformer. Content-driven slice assignment by design; authors note this learns physics-state clusters implicitly from data. The NGSB extends this with a structural geometric prior that is complementary to, not competitive with, the content learning.

---

### Summary for Assignment

**Hypothesis**: Adding a 24-parameter learned linear map from surface normals to per-head additive slice-assignment logit bias (zero-initialized, additive) gives the Transolver's physics-state clustering a structural geometric prior, enabling τ_z tokens with similar surface orientation to co-cluster and share representational capacity.

**Why orthogonal to H347–H350**: Touches the attention mechanism's internal slice assignment — not input features (H348), not training losses (H347), not output targets (H349), not output decoder (H350).

**Falsifier**: WSS_z train loss unchanged at 4–6h diagnostic gate → mechanism not alive, close immediately.

**Expected upside**: ~50–100bp test_WSS_z if mechanism lives (10–20bp val_cal). GeoTransolver external evidence strongly supports geometric attention injection closing the WSS_z gap on DrivAerML.

**Budget**: 4–6h diagnostic gate + 20–24h full finetune if gate passes. 8×GPU DDP standard config.
