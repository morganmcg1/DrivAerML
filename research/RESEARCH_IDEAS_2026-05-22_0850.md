# Research Ideas — 2026-05-22 08:50
# Architectural Proposals for Breaking the surf_p Floor + WSS Target

**Context:** Scalar/loss-weight space (H19–H32) is exhausted. The surf_p floor (3.577%)
has never been cleared; best WSS remains 6.634% (H19). The root cause is the
shared-encoder architecture: GradNorm budget allocated to any task propagates
through the shared encoder to all others, making scalar fixes self-defeating
(H30 falsified this conclusively at EP6). Morgan's directive: move to
architecture — specifically WSS-dedicated head with cross-attention to surf_p
and geometry features.

**Constraints (hard):**
- NO ENSEMBLES, single-model only
- PR #972 AND-contract: test_vol_p ≤ 3.643% AND test_surf_p ≤ 3.577%
- WSS target: test_wss < 5.85% (Transolver-3)
- 24h wall-clock, DDP8, 8×H100 96GB, 30 epochs typical

---

## Idea A — GALE-Transolver: Persistent Geometry Cross-Attention at Every Layer

### What it is
Inject a shared geometry context bank into every Transolver block via
cross-attention, following the GeoTransolver GALE (Geometry-Aware Latent
Embeddings) design. Instead of one-shot geometry encoding at the input
projection, geometry information re-enters the Physics-Attention computation
at every layer through a side-channel cross-attention.

### Why it addresses the bottleneck
The coupling problem is that surf_p and WSS share encoder slices, so gradient
budgets compete. Persistent geometry conditioning creates an independent
pathway for geometry signal that does not route through the contested shared
physical states. Each layer can query the geometry bank without consuming
shared-state capacity.

GeoTransolver on DrivAerML achieves **surf_p=2.86%, wss=4.90%, vol_p=3.09%**
— clearing ALL three floor constraints simultaneously. This is the only
known configuration that does so on this exact benchmark split.

### Architecture change
Within each Transolver block, after the Physics-Attention step, add a
cross-attention sublayer:

```python
# Geometry context bank: encode surface normals, area, curvature at multiple scales
# Computed once before the forward pass, shared across all layers
geo_ctx = GeometryContextBank(surface_x, radii=[0.01, 0.05, 0.25, 1.0, 2.5, 5.0])
# shape: [B, K, D] where K = number of geometry anchors (e.g., 512–2048)

# Inside each Transolver block:
h_after_sa = physics_attention(h)         # existing self-attention on slices
h_geo = cross_attn(h_after_sa, geo_ctx)   # NEW: cross-attn to geometry bank
# Adaptive gate (learned per layer):
alpha = sigmoid(linear(cat(pool(h_after_sa), pool(h_geo))))
h_out = alpha * h_geo + (1 - alpha) * h_after_sa
```

Multi-scale ball queries at 6 radii (0.01, 0.05, 0.25, 1.0, 2.5, 5.0) sample
geometry anchors from surface mesh points using `torch_cluster` KNN or
radius_graph. The geometry bank is encoded once per forward pass from surface
normals, area, curvature (channels already available in surface_x).

### Editable files
`model.py` (add GeometryContextBank class, modify each TransolverBlock),
`train.py` (no change). Estimated ~200–300 LoC, 4–6 hours.

### Implementation notes
- Geometry bank size K=512 or 1024 to stay within memory budget. With B=1 on
  DDP8, total surface points ~200k, so K=1024 ball-query anchors is feasible.
- Cross-attention heads: 4 (matching existing model heads), key/value dim 64.
- The adaptive gate alpha prevents geometry signal from dominating when not
  needed (e.g., volume decoder does not query geometry bank).
- Surface decoder and WSS decoder both receive geometry-conditioned features;
  volume decoder optionally queries only at low weight.
- Watch for memory: 6 scales × 1024 anchors × 512 dim adds ~3.1M parameters
  and ~4GB activation overhead per GPU at 65k surface points.

### Predicted metric deltas
Based on GeoTransolver's DrivAerML numbers relative to Transolver-3 baseline:
- surf_p: −0.72pp (from ~3.58% to ~2.86%) — expected to clear floor
- wss: −0.95pp (from ~5.85% to ~4.90%)
- vol_p: −0.55pp improvement

### Falsifiable predictions
1. If geometry xattn is working: val_surf_p should drop below 3.4% by EP6.
   If still above 3.5% at EP6, geometry conditioning is not reaching the surface decoder.
2. Ablation signal: removing geometry xattn from surface-only layers should
   degrade surf_p by at least 0.2pp while leaving vol_p unchanged.
3. If alpha gates collapse to 0 everywhere, geometry signal is not flowing —
   monitor alpha histograms in W&B.

### References
- GeoTransolver (arxiv 2505.12558, 2025): Geometry-Aware Latent Embeddings +
  multi-scale cross-attention for CFD mesh surrogates. DrivAerML results:
  surf_p=2.86%, wss=4.90%, vol_p=3.09%.
- Transolver (arxiv 2402.02366, Wu et al., NeurIPS 2024): base architecture.

### Taste score
- Mechanistic grounding: 4 (GeoTransolver's DrivAerML numbers are direct
  evidence in the same benchmark; mechanism targets the shared-encoder
  coupling bottleneck precisely)
- Research-state value: 4 (would either break surf_p floor — first time in the
  wave — or definitively show GeoTransolver's DrivAerML results don't transfer
  to this codebase, which itself updates the map sharply)
- Execution value: 3 (200–300 LoC change, staged EP6 gate feasible, but memory
  pressure requires a dry-run check before committing 30-epoch slot)

---

## Idea B — Ada-Temp Slices: Per-Point Learned Softmax Temperature

### What it is
Replace the fixed softmax temperature in Transolver's Physics-Attention slice
assignment with a per-point learned temperature:
`τ_i = τ₀ · exp(MLP_τ(x_i))` where `x_i` is the point's spatial/feature
embedding. Add Gumbel-Softmax reparameterization during training to prevent
slice collapse (Transolver++ design, arxiv 2502.02414).

### Why it addresses the bottleneck
State homogenization is a known failure mode of Physics-Attention on large
meshes: all points in a region collapse to one physical state, losing boundary
layer resolution. WSS is a near-wall quantity requiring fine-grained
differentiation between the boundary layer and the freestream. A fixed τ
cannot simultaneously handle both scales. Per-point τ allows near-wall
points to use low temperature (sharp, local slice assignment) while far-field
points use high temperature (diffuse, global averaging).

Transolver++ reports 13% relative gain across PDE benchmarks without
architectural surgery to the decoder. It is a targeted fix for the slice
homogenization that degrades boundary-layer WSS prediction.

### Architecture change
In the Physics-Attention module:

```python
# Before (fixed temperature):
slice_weights = softmax(query @ key.T / tau)  # tau is scalar hyperparameter

# After (Ada-Temp):
tau_raw = self.tau_head(x_embed)  # x_embed: [B, N, D], tau_head is Linear(D, 1)
tau_i = self.tau0 * torch.exp(tau_raw).clamp(0.1, 10.0)  # per-point, [B, N, 1]

# During training: Gumbel-Softmax for discrete slice assignment
if self.training:
    slice_weights = gumbel_softmax(query @ key.T / tau_i, hard=False, tau=tau_i)
else:
    slice_weights = softmax(query @ key.T / tau_i)
```

τ₀ initialized to current fixed-temperature value. tau_head is a single
Linear layer (D→1) per Transolver block; no additional non-linearity needed
per Transolver++ ablations.

### Editable files
`model.py` only (modify PhysicsAttention class). Estimated ~60–80 LoC, 1–2 hours.

### Implementation notes
- τ clamp [0.1, 10.0] is critical: without lower bound, temperature collapses
  to 0 and causes NaN in softmax; without upper bound, all slices become
  uniform (no differentiation).
- Gumbel noise scale should match τ magnitude: `gumbel_scale = 0.1 * τ₀`.
- Monitor: log `tau_i.mean()` and `tau_i.std()` per layer in W&B to confirm
  spatial differentiation is forming (near-wall vs far-field τ divergence).
- Per Transolver++ ablation: Gumbel-Softmax adds 1.2% overhead. The per-point
  tau_head adds ~6k parameters per layer (trivial).
- This does NOT change the decoder or loss — it is a pure encoder change.
  Expected to improve WSS most; surf_p benefit secondary.

### Predicted metric deltas
Based on Transolver++ 13% relative gain extrapolated to current wss=6.634%:
- wss: ~−0.86pp (to ~5.77%) — expected to clear Transolver-3 target
- surf_p: ~−0.2pp secondary benefit from better boundary-layer encoding
- vol_p: neutral (far-field states unaffected)

### Falsifiable predictions
1. Near-wall points (SDF < 0.01) should have lower mean τ than far-field
   points (SDF > 1.0) by EP4 if the mechanism is working.
2. If τ_i.std() ≈ 0 across layers (all temperatures converge), the tau_head
   is not learning spatial differentiation — check for gradient flow.
3. If wss does not improve >0.3pp vs baseline by EP10, spatial temperature
   differentiation is not the limiting factor for WSS on this dataset.

### References
- Transolver++ (arxiv 2502.02414, 2025): Ada-Temp + Gumbel-Softmax on large
  PDE meshes. 13% relative gain reported; DrivAerNet++ results cited.
- Transolver (arxiv 2402.02366): base architecture reference.

### Taste score
- Mechanistic grounding: 3 (mechanism is precise and targets known
  homogenization failure mode; external evidence from same benchmark family;
  DrivAerML-specific validation is indirect)
- Research-state value: 3 (negative result would confirm slice homogenization
  is NOT the WSS bottleneck, which is still useful; positive would compound
  with geometry xattn)
- Execution value: 4 (60–80 LoC, 1–2 hours implementation, cheapest
  diagnostic in this list, staged EP6 gate, direct WSS metric)

---

## Idea C — WSS↔surf_p Bidirectional Cross-Attention Decoder

### What it is
After the final Transolver encoder layer, split into dedicated WSS and surf_p
decoder heads, then add a single bidirectional cross-attention step between
the two heads before their final projection layers. WSS attends to surf_p
features (to inherit pressure-gradient topology) and surf_p attends to WSS
features (to inherit boundary-layer shear topology). This is Morgan's explicit
architectural suggestion.

### Why it addresses the bottleneck
WSS and surf_p are physically coupled through the no-slip condition:
τ_w = μ · (∂u/∂n)|_wall and ∇p|_surface drives ∂u/∂n. A model that predicts
them from independent heads without coupling will learn them as uncorrelated
signals, missing this physical relationship. H12 (PR #1166, `3v58n2m5`)
showed a separate τ head with no coupling did NOT help — confirming that
separation alone is insufficient. The cross-attention coupling is the key
architectural element absent from H12.

Historical xattn experiments (PRs #883–#896) were volume→surface or used
a different architecture wave; they did not test WSS↔surf_p cross-attention
in the current Transolver-with-curvature-bias codebase.

### Architecture change
After the final TransolverBlock, before surface projection:

```python
# Existing: shared encoder output h_surface [B, N_s, D]
# New: split into two parallel streams

h_wss = self.wss_norm(h_surface)    # [B, N_s, D]
h_cp  = self.cp_norm(h_surface)     # [B, N_s, D]

# Bidirectional cross-attention (single head, lightweight)
# WSS attends to surf_p features
h_wss_aug = h_wss + self.wss_to_cp_xattn(
    query=h_wss, key=h_cp, value=h_cp
)  # WSS inherits pressure-gradient topology

# surf_p attends to WSS features
h_cp_aug = h_cp + self.cp_to_wss_xattn(
    query=h_cp, key=h_wss, value=h_wss
)  # surf_p inherits boundary-layer shear topology

# Independent projections
wss_pred = self.wss_head(h_wss_aug)  # → channels 1-3 (wss_x/y/z)
cp_pred  = self.cp_head(h_cp_aug)    # → channel 0 (cp/surf_p)
```

Cross-attention: 4 heads, key dim 64, single layer (no MLP between). The two
xattn modules share no parameters (asymmetric coupling allowed).

### Editable files
`model.py` (add wss_norm, cp_norm, two cross-attention modules, replace final
surface projection with split heads), `train.py` (no change, model contract
preserved — still outputs surface_preds [B,N,4]). Estimated ~120–150 LoC,
2–3 hours.

### Implementation notes
- The model contract (surface_preds [B,N,4]) is unchanged: wss_pred fills
  channels 1-3, cp_pred fills channel 0, then cat along dim -1.
- Cross-attention scale: use standard sqrt(d_k) scaling. Do NOT add MLP
  after; residual add is sufficient for a single coupling layer.
- This is NOT H12: H12 was a separate τ-head (no cross-attention, no coupling
  between heads). H3 (PR #1129) was volume→surface, not WSS↔surf_p.
- Monitor: log cosine similarity between h_wss_aug and h_cp_aug during
  training. If similarity collapses to 1.0, the heads are not differentiating.
- GradNorm still applies per-loss-term; having separate heads does not require
  changing the weighting scheme, though it creates new gradient paths.
- Risk: if the cross-attention collapses (all-uniform attention weights), it
  reduces to Idea H12 and should show no gain. The falsifiable test is whether
  attention weights show meaningful structure over the surface.

### Predicted metric deltas
This is the most speculative prediction (no direct DrivAerML evidence):
- wss: −0.3 to −0.6pp (from ~6.63% toward 6.0–6.3%)
- surf_p: −0.1 to −0.3pp (benefit from WSS→surf_p pressure coupling)
- vol_p: neutral (volume head unchanged)

### Falsifiable predictions
1. Attention weight entropy for wss→cp cross-attention should be significantly
   lower than random (concentrated on pressure-gradient regions) by EP6.
2. If wss gain is <0.2pp with meaningful attention weight structure, the
   coupling exists but is insufficient alone — combine with Idea B.
3. Cosine similarity between wss_aug and cp_aug features should diverge from
   1.0 as training progresses (if heads remain identical, coupling adds nothing).

### References
- Morgan's explicit architectural directive (CURRENT_RESEARCH_STATE.md,
  2026-05-22): "head specifically for WSS" + "cross attention between that
  head and others"
- GeoTransolver (arxiv 2505.12558): provides cross-attention design patterns
  in this architecture class
- H12 (PR #1166, run 3v58n2m5): negative result — separate head WITHOUT
  cross-attention; establishes that separation alone does not help

### Taste score
- Mechanistic grounding: 3 (physically motivated coupling; H12 negative
  provides useful negative anchor; no DrivAerML numbers from exact method)
- Research-state value: 4 (Morgan's explicit directive; if positive, opens
  cross-attention decoder as new compoundable direction; if negative,
  confirms that decoder coupling is not the bottleneck, pointing squarely
  at encoder geometry conditioning — Idea A)
- Execution value: 3 (120–150 LoC, 2–3 hours, staged EP6 gate on WSS;
  the bidirectional design directly tests Morgan's hypothesis vs. H12)

---

## Idea D — Curvature-Enhanced Surface Features + Second-Order Geometry Encoding

### What it is
Augment the 7-channel surface input `[x,y,z,nx,ny,nz,area]` with 4 additional
curvature-derived features: mean curvature H, Gaussian curvature K, principal
curvature directions (κ₁, κ₂). These are computable from the existing mesh
normals via discrete exterior calculus (libigl / pytorch3d) and cached at
preprocessing time.

### Why it addresses the bottleneck
PR #1131 (curvature additive attention bias) was merged as a positive result,
demonstrating that curvature carries information the model cannot extract from
normals alone. The additive bias mechanism is a weak signal path — injecting
curvature as input features gives the model direct access to it from the first
projection layer, not just as a residual attention bias. Near-wall WSS is
controlled by curvature of the boundary layer (high κ → strong adverse
pressure gradient → early separation → large WSS deviation). Mean curvature
is particularly informative for surf_p (stagnation zones have κ ≫ 0).

### Implementation notes
- Preprocessing script (outside `model.py`) computes H, K, κ₁, κ₂ from mesh
  normals using cotan-weight discrete Laplacian. Cache alongside existing
  features in processed dataset.
- Update `surface_in_channels` from 7 to 11 in `model.py` input projection.
- Clamp/normalize curvature values: mesh curvature can be numerically extreme
  at sharp edges; use tanh(c · κ) with c=0.1.
- Monitor: if curvature features have near-zero gradient in the first
  projection layer, the model is ignoring them — check by zeroing them at
  inference to measure drop.

### Predicted metric deltas
- surf_p: −0.15 to −0.30pp (stagnation detection)
- wss: −0.20 to −0.40pp (separation zone detection)
- vol_p: neutral

### References
- PR #1131: curvature additive attention bias — merged positive result
  (same codebase, confirms curvature signal is useful)
- "DrivAerNet++: A Large-Scale Multimodal Car Dataset" (arxiv 2406.09624):
  uses curvature as an input feature in several baseline models

### Taste score
- Mechanistic grounding: 3 (PR #1131 provides local codebase evidence;
  mechanism is clear; the question is magnitude of gain)
- Research-state value: 2 (positive: adds one more compoundable improvement;
  negative: shows PR #1131's attention-bias form captured all available
  curvature signal, which is a useful distinction)
- Execution value: 3 (preprocessing change + 1-line model edit; low risk;
  combines naturally with Ideas A/B/C as a +0 complexity addition)

---

## Idea E — Slice-Aware Volume-Surface Information Routing via Shared Physical States

### What it is
Force the Transolver encoder to maintain a shared physical state bank whose
surface-attending slices and volume-attending slices partially overlap, then
add an explicit routing module that reads the overlap states and writes them
to both surface and volume decoders. This is a within-model multi-task
information sharing mechanism that does NOT require ensemble routing.

### Why it addresses the bottleneck
The current architecture has a shared encoder but SEPARATE decoders with no
coupling after the encoder. Volume pressure (vol_p) and surface pressure
(surf_p) are physically related through boundary conditions — yet the model
has no pathway for the vol_p decoder to inform the surf_p decoder (or vice
versa). A shared physical state bank that explicitly routes vol_p-relevant
states to the surf_p decoder could break the surf_p floor by giving surf_p
access to volume boundary-layer information.

This is distinct from GradNorm, which only adjusts gradient magnitudes — it
adds an actual information pathway.

### Architecture change
After the final Transolver layer, compute per-state relevance scores for
surface vs. volume tasks, then route states by relevance:

```python
# Physical states: [B, M, D] where M = num_slices (e.g., 128)
# Relevance scores per state:
surf_rel = sigmoid(self.surf_router(states))   # [B, M, 1]
vol_rel  = sigmoid(self.vol_router(states))    # [B, M, 1]

# Weighted readout for each decoder
surf_states = (states * surf_rel).sum(dim=1)  # [B, D]
vol_states  = (states * vol_rel).sum(dim=1)   # [B, D]

# Cross-routing: add top-k vol_rel states to surface decoder
k = 16
top_vol_states = states[..., topk(vol_rel, k), :]  # [B, k, D]
surf_ctx = self.vol_to_surf_xattn(surf_queries, top_vol_states)
```

### Predicted metric deltas
Speculative (no direct analogues): surf_p −0.1 to −0.2pp, wss neutral to −0.1pp.
This idea is primarily a surf_p fix, not a WSS fix.

### Taste score
- Mechanistic grounding: 2 (plausible physical motivation; no external
  evidence for this exact mechanism; more speculative than A–C)
- Research-state value: 3 (tests a specific hypothesis about vol_p→surf_p
  routing that no prior experiment in the wave has touched)
- Execution value: 2 (more complex than Ideas B/C; should be deprioritized
  until A–C are tested)

---

## Recommended Execution Order

Given the evidence base and Morgan's directive, the recommended order is:

1. **Idea A (GALE-Transolver)** — strongest external evidence (GeoTransolver
   DrivAerML numbers), addresses the root cause (shared encoder coupling),
   expected to clear both surf_p floor and WSS target simultaneously.
   Run first; positive EP6 gate → full 30-epoch run.

2. **Idea B (Ada-Temp)** — cheapest implementation (~60 LoC), orthogonal to
   Idea A, can run in parallel. Positive result compounds with A.
   If A is in progress, assign B to a second student simultaneously.

3. **Idea C (WSS↔surf_p xattn)** — Morgan's explicit hypothesis, directly
   tests the decoder coupling mechanism H12 did not test. Run after or
   in parallel with A; if A succeeds, combine A+C for next wave.

4. **Idea D (curvature features)** — lowest risk, highest stackability.
   Assign as a complement to whichever of A/B/C shows early promise.

5. **Idea E (slice routing)** — most speculative; defer until A–C have results.

### Staged EP gate protocol (applies to all ideas)
- EP6 gate: check `val_surf_p`. If still above 3.5% with no downward trend,
  send back to student for diagnosis before consuming remaining epochs.
- EP15 gate: if primary metric has not improved >0.2pp vs baseline, close.
- EP30 terminal: full contract check (vol_p ≤ 3.643% AND surf_p ≤ 3.577% AND
  WSS target considered).

---

## Stop Condition

If Idea A (GALE-Transolver) fails to clear surf_p < 3.5% at EP6 after correct
implementation verification, this indicates one of:
(a) the geometry context bank is not reaching the surface decoder (check alpha gates)
(b) the DrivAerML dataset in this split/normalization differs materially from
    GeoTransolver's training distribution
(c) the curvature bias (PR #1131) already captured the available geometry signal

In that case: run GeoTransolver in reference configuration on this exact
dataset as a diagnostic oracle before investing in further architecture work.
