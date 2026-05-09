# Research Ideas — 2026-05-09 22:00
# Edward — Next Hypotheses After PR #906 (Post-xattn Vol Self-Attn NEGATIVE)

## Context

**Constraint from closed experiment history:**
Post-xattn capacity additions are 0-for-3 on this benchmark:
- PR #891 (fern): post-xattn FFN, EP3=8.50%, NEGATIVE
- PR #906 (edward): post-xattn vol self-attn, EP3=8.23%, NEGATIVE
- PR #893 (Round 22): vol kNN-graph-attn post-xattn, NEGATIVE

This closes the post-xattn capacity axis. Future volume-branch experiments must target:
(a) capacity placed BEFORE the surf→vol xattn,
(b) the geometry/positional encoding pathway,
(c) data-augmentation / regularization approaches targeting OOD generalization.

**Round 24 in-flight (do not duplicate):**
- #901 fern: y-mirror aug (REQUESTED CHANGES)
- #918 tanjiro: vol RFF sigma shift (REQUESTED CHANGES)
- #921: geometric mixup (coordinate-level interpolation)
- #925: surface aug variant
- #926: vol geo features (centroid+bbox coarse shape context)
- #927: aux physics loss
- #928: TTA (test-time augmentation)

**Primary SOTA single model (PR #823):** val_abupt=6.4407%, test_abupt=7.6992%, vol_p_val=3.8557%, vol_p_test=11.6704%

**AB-UPT targets:** surface_pressure=3.82%, wall_shear=7.29%, volume_pressure=6.08%

**OOD bottleneck:** 4 test cases (run_133, run_226, run_203, run_158) account for 92% of squared vol_p test deviation. Excluding them, test_vol_p ≈ 3.9–4.2% (already near AB-UPT target). The gap is geometric/architectural, not training-distribution.

---

## H1 [Category a — Pre-xattn vol self-attention] — TOP PICK

### What it is
A single `nn.MultiheadAttention` (MHA) block with zero-initialized out_proj, inserted in the volume branch immediately BEFORE the surf→vol cross-attention, so that vol tokens can communicate with spatial neighbors before being used as cross-attention queries.

### Why it might help here
Post-xattn vol-vol capacity adds communication AFTER the surface conditioning step, but the cross-attention Q-projection already has no spatial context at that point — each vol token asks the surface independently, ignoring what neighboring vol tokens have already inferred. Pre-xattn vol self-attention gives each vol token awareness of its spatial neighborhood before it forms its query. This "query refinement" mechanism is the direct architectural complement to the closed post-xattn pattern and has not been tested.

The mechanism is well-validated in the Perceiver IO family: Perceiver IO (Jaegle et al., 2022) uses alternating cross-attention + self-attention within the latent array, where the self-attention steps refine latent representations between cross-attention read-outs. The same principle applies here: vol self-attention before surf→vol cross-attention makes the queries geometrically informed. Stratified Transformer (Lai et al., 2022 CVPR) further shows that combining local (dense) + long-range (sparse) self-attention before cross-modal attention improves 3D point cloud processing.

The key distinguishing feature: this is pre-xattn (new position, never tested) vs. post-xattn (0-for-3 closed). The zero-init guarantees identity-at-init so the 4-ep screen starts from the same effective weight state as the SOTA checkpoint.

### Key papers
- Perceiver IO (Jaegle et al., 2022, NeurIPS): iterative latent self-attn between cross-attn steps. https://arxiv.org/abs/2107.14795
- Stratified Transformer (Lai et al., 2022, CVPR): local+long-range self-attn before cross-modal for 3D point clouds. https://arxiv.org/abs/2203.14508
- Point Transformer (Zhao et al., 2021, ICCV): self-attn in 3D point clouds before aggregation. https://arxiv.org/abs/2012.09164

### Implementation notes
In `model.py`, locate the surf→vol xattn call (after backbone, around the `cross_attn(Q=vol_h, K=V=surf_h)` line). Add immediately before it:

```python
# Pre-xattn vol self-attention (H1)
if self.use_pre_xattn_vol_self_attn:
    vol_h_res = vol_h
    vol_h = self.pre_xattn_vol_ln(vol_h)  # optional pre-norm
    vol_h_out, _ = self.pre_xattn_vol_mha(vol_h, vol_h, vol_h,
                                            key_padding_mask=~volume_mask)
    vol_h = vol_h_res + vol_h_out  # residual
```

Add in `__init__`:
```python
if use_pre_xattn_vol_self_attn:
    self.pre_xattn_vol_ln = nn.LayerNorm(hidden_dim)
    self.pre_xattn_vol_mha = nn.MultiheadAttention(
        embed_dim=hidden_dim,  # 512
        num_heads=num_heads,   # 4
        batch_first=True,
        dropout=0.0,
    )
    nn.init.zeros_(self.pre_xattn_vol_mha.out_proj.weight)
    nn.init.zeros_(self.pre_xattn_vol_mha.out_proj.bias)
```

Add `--use-pre-xattn-vol-self-attn` flag. Set `find_unused_parameters=True` in DDP init. Use `--no-compile-model`. Parameters: ~2.1M (same budget as closed post-xattn FFN but at a different architectural position).

### Suggested experiment design
4-ep screen with standard schedule:
```
--epochs 4 --lr-cosine-t-max 13 --use-pre-xattn-vol-self-attn
--vol-n-points 16384 --vol-curriculum 0:16384:1:32768:2:49152:3:65536
```
Kill gate: EP1 val_abupt > 12% (same as SOTA trajectory). EP3 kill gate: > 8.0%. If EP3 < 8.0%, run full 13-ep confirmation.

### Taste rubric
- Research mode: **frontier refinement** (targeting an untested architectural position adjacent to a known winner)
- Mechanistic grounding: **4** — mechanism is precise (query refinement before cross-attention), falsifiable (EP3 result vs. 8.0% kill gate), tied to closed post-xattn pattern and Perceiver IO literature
- Research-state value: **4** — distinguishes whether the pre-xattn position vs. post-xattn position is the key variable; result sharpens the research map either way
- Execution value: **4** — ~15 lines of code, model.py only, proven init pattern, 4-ep screen, no trainer changes

---

## H2 [Category b — SDF-Modulated Vol PE: Per-Token Spectral Scaling]

### What it is
A tiny MLP (3 layers, ~1024 parameters total) that reads each volume token's SDF value and outputs 5 scalar weights, one per STRING RFF octave. These weights multiply the per-octave RFF features before concatenation, giving each volume token physics-informed, spatially-adaptive positional resolution.

### Why it might help here
The SDF feature (vol_x[:,3]) encodes signed distance to the car surface. Near-surface vol tokens (|sdf|<0.1m) live in the boundary layer — high pressure gradient, high spatial frequency, needing the high-σ STRING octaves. Far-field tokens (|sdf|>2m) are freestream — slowly varying, benefiting from low-σ octaves. The current STRING PE treats all vol tokens identically. An SDF-conditioned per-token octave weighting MLP replicates the physics: "this token is in the boundary layer, so emphasize fine-grained spatial encoding." This is exactly the mechanism behind Adaptive Frequency Net (AFNet, Zhang et al. 2023), which uses a hypernetwork to learn local texture frequency encoding — adapted here to use domain physics (SDF) instead of learned local statistics.

The 4 OOD test cases have cars with different body-to-wake SDF distributions than the training set. By making PE resolution a function of the SDF value (a geometry-invariant physical signal), the model adapts its positional encoding to geometry OOD shift automatically.

### Key papers
- NeRF (Mildenhall et al., 2020): frequency-dependent positional encoding sensitivity to local signal complexity. https://arxiv.org/abs/2003.08934
- AFNet (Zhang et al., 2023, CVPR): hypernetwork-driven adaptive frequency modulation for point cloud rendering. https://arxiv.org/abs/2303.07596
- Instant-NGP (Müller et al., 2022): hash-based adaptive spatial encoding. https://arxiv.org/abs/2201.05989

### Implementation notes
In `model.py` STRING PE computation for volume tokens, locate the RFF section. After computing RFF features for each octave, apply:

```python
# SDF-modulated octave scaling (H2)
if self.use_sdf_pe_scaling:
    sdf_vals = vol_x[:, :, 3:4]  # [B, N_vol, 1] — already in vol_x
    octave_scales = self.sdf_pe_mlp(sdf_vals)  # [B, N_vol, 5] — softplus output
    # octave_scales has shape [B, N_vol, 5]; RFF_stack is [B, N_vol, 5, 2*d_rff_per_octave]
    rff_scaled = RFF_stack * octave_scales.unsqueeze(-1)  # broadcast
    vol_pe = rff_scaled.flatten(-2)  # [B, N_vol, 5*2*d_rff_per_octave]
```

MLP: `Linear(1→16) → SiLU → Linear(16→5) → Softplus` (softplus ensures positive scaling, initialized near 1.0 by setting final bias to log(exp(1)-1) ≈ 0.541). Params: ~112. Entire change is in model.py.

### Suggested experiment design
```
--epochs 4 --lr-cosine-t-max 13 --use-sdf-pe-scaling
--vol-curriculum 0:16384:1:32768:2:49152:3:65536
```
Diagnostic: check EP1 vol_p vs. SOTA (should be ≤ 3.9%). If worse, the SDF feature may need normalization (clip to [-5, 5]m before MLP input).

### Taste rubric
- Research mode: **frontier refinement** (geometry/PE pathway, new mechanism)
- Mechanistic grounding: **3** — physics motivation is clear (boundary layer = high spatial frequency), but the link between per-octave scaling and the 4 OOD cases is somewhat speculative
- Research-state value: **3** — separates geometry-adaptive PE from global PE axis; if it works, opens a whole direction of physics-informed PE
- Execution value: **4** — ~20 lines, model.py only, trivially reversible, minimal compute overhead

---

## H3 [Category b — Learned Surface-Geometry-Conditioned Q-Bias for xattn]

### What it is
A small MLP that reads the mean-pooled surface hidden state (a 512-dim global geometry descriptor) and outputs an additive bias on vol Q-projections immediately before surf→vol cross-attention. This allows the model to shift "where" each vol token queries the surface as a function of the overall car geometry.

### Why it might help here
The 4 OOD test cases have car geometries outside the training distribution. The vol Q-projections for cross-attention are computed from per-token vol hidden states that have no explicit global geometry context. A global geometry bias on Q makes the query direction geometry-conditional: for a fastback body, the near-wake Q-bias differs from a squareback body. This is a dynamic analog to the fixed cross-attention positional bias tested in PR #883 (CLOSED, used static RFF-based PE) — here the bias is LEARNED from the actual input surface geometry of each example.

Equivariant Neural Fields (Wessels et al., 2025, ICLR) show that geometry-informed cross-attention conditioning significantly improves generalization to unseen geometries. The key difference from closed #883: that PR used a fixed global PE that cannot adapt to per-example geometry; this uses a learned, input-conditioned global descriptor.

### Key papers
- Equivariant Neural Fields (Wessels et al., 2025, ICLR): geometry-informed cross-attention for conditional neural fields. https://arxiv.org/abs/2406.05416
- Flamingo (Alayrac et al., 2022): tanh-gated cross-attention bias for multi-modal conditioning. https://arxiv.org/abs/2204.14198
- PointMAE (Pang et al., 2022): global geometry tokens for 3D transformers. https://arxiv.org/abs/2203.06604

### Implementation notes
In `model.py`, before the surf→vol xattn call:

```python
# Geometry-conditioned Q-bias (H3)
if self.use_geom_q_bias:
    geom_desc = surf_h.mean(dim=1)  # [B, 512] — mean-pool surface hidden state
    q_bias = self.geom_q_mlp(geom_desc)  # [B, 512]
    vol_h_q = vol_h + q_bias.unsqueeze(1)  # broadcast to [B, N_vol, 512]
    # Use vol_h_q as Q in xattn, vol_h unchanged for residual
```

MLP: `Linear(512→256) → GELU → Linear(256→512)` — ~394K parameters. Initialize final layer to near-zero (normal init with std=0.01). This only affects Q, not K/V, so no K/V backflow concerns (distinguishing from closed #890 detach-K/V, which restricted gradient flow differently).

Risk: 400 training examples may overfit to a global geometry descriptor. Mitigate with weight decay (current wd=5e-4 may need raising to 1e-3 for this MLP) or by using a lower-dim bottleneck (256→64→512).

### Suggested experiment design
```
--epochs 4 --lr-cosine-t-max 13 --use-geom-q-bias
--vol-curriculum 0:16384:1:32768:2:49152:3:65536
```
Ablation signal: monitor EP1 vol_p_val closely. If > 4.5% (regression from SOTA 3.86%), kill early — overfit risk is materializing. If < 4.0%, continue.

### Taste rubric
- Research mode: **diagnostic + frontier refinement** (tests whether the OOD gap is PE/geometry-representation-limited vs. architecture-limited)
- Mechanistic grounding: **3** — well-motivated, clearly distinguishable from #883 (dynamic vs. fixed bias), but the 400-example overfit risk is a live concern
- Research-state value: **3** — result distinguishes whether global geometry conditioning matters for xattn quality
- Execution value: **3** — clean model.py change, but ~394K params needs careful weight decay tuning; consider running two arms (wd=5e-4 and wd=1e-3)

---

## H4 [Category c — Manifold Mixup on Backbone Hidden States]

### What it is
Interpolation of surface and volume hidden representations between two training examples (in the latent space, after the backbone, before the heads and xattn) with mixed targets. This is Manifold Mixup (Verma et al., 2019) applied at the representation level rather than input level.

### Why it might help here
Geometric Mixup (#921, in-flight) interpolates at the coordinate level. Manifold Mixup interpolates at the hidden representation level, which is the manifold the model has actually learned. This creates denser coverage of the learned geometry manifold in hidden space, directly targeting the OOD gap: the 4 OOD test cases are far from the training distribution in geometry space, but may be interpolable in hidden representation space after backbone pre-training.

Manifold Mixup (Verma et al., 2019, ICML) consistently improves calibration and OOD robustness on image classification and regression by "flattening" the hidden representation surface between training examples. The mixing in hidden space is more structured than coordinate-space mixing because the backbone has already factored out the geometric complexity.

### Key papers
- Manifold Mixup (Verma et al., 2019, ICML): interpolate hidden representations for regularization. https://arxiv.org/abs/1806.05236
- On Mixup Regularization (Carratino et al., 2022, JMLR): theoretical analysis of Mixup regularization effects. https://www.jmlr.org/papers/volume23/20-1385/20-1385.pdf
- Mixing between worlds (2022): Mixup for OOD generalization in regression settings.

### Implementation notes
In `trainer_runtime.py`, after backbone forward pass and before xattn + head forward pass:

```python
# Manifold Mixup (H4) — applied with probability p_mix=0.5 per batch
if training and random.random() < 0.5:
    lam = np.random.beta(0.2, 0.2)
    idx = torch.randperm(B, device=surf_h.device)
    surf_h_mix = lam * surf_h + (1 - lam) * surf_h[idx]
    vol_h_mix  = lam * vol_h  + (1 - lam) * vol_h[idx]
    surf_targets_mix = lam * surf_targets + (1 - lam) * surf_targets[idx]
    vol_targets_mix  = lam * vol_targets  + (1 - lam) * vol_targets[idx]
    # Use *_mix for forward pass through heads + xattn + loss
```

Key considerations:
- Point cloud masks must be handled carefully: after mixing, use the intersection of masks (`mask_mix = mask & mask[idx]`) or the union (more conservative).
- Batch size must be >= 2 per GPU (confirm this holds under vol-curriculum sampling).
- `lam ~ Beta(0.2, 0.2)` encourages strong mixing (values near 0 or 1 more likely than 0.5); use `max(lam, 1-lam)` if you want to avoid inverting supervision.
- Incompatible with naive DDP pair-sampling (pairs must be formed within-rank, not across ranks). Simplify by pairing within the per-GPU batch.

This requires trainer changes — scope is slightly larger than H1/H2.

### Suggested experiment design
```
--epochs 4 --lr-cosine-t-max 13 --use-manifold-mixup --mixup-alpha 0.2
--vol-curriculum 0:16384:1:32768:2:49152:3:65536
```
Check EP1 vol_p_val. Manifold Mixup often slightly slows convergence in EP1 due to mixed supervision, so a slightly looser EP1 kill gate (13.5% instead of 12%) may be warranted. If EP3 < 8.2%, continue.

### Taste rubric
- Research mode: **OOD regularization** (category c)
- Mechanistic grounding: **3** — proven OOD regularizer in image settings, plausible mechanism for geometry OOD; slightly weaker than H1 because the CFD geometry manifold may not be interpolable in hidden space at this training size (400 examples)
- Research-state value: **3** — distinguishes manifold-mixing from coordinate-mixing (#921 in-flight); if both work, the level of mixup matters; if this works and #921 doesn't, representational mixing is the key
- Execution value: **2** — requires trainer changes, mask handling complexity, slightly higher implementation risk than model.py-only changes

---

## H5 [Category c — SDF-Range Stochastic Vol Token Masking (Physics-Zone Dropout)]

### What it is
During training, randomly mask vol tokens that fall within a specific SDF range (e.g., near-surface boundary layer: sdf ∈ [-0.3, 0.05]m) with a per-epoch annealed dropout probability, forcing the model to predict vol_p under missing near-surface coverage.

### Why it might help here
The 4 OOD test cases have geometrically unusual near-wake regions — the SDF distribution in the boundary layer zone is shifted relative to training cases. By randomly dropping near-surface vol tokens during training, force the model to predict vol_p without relying on uniform spatial coverage. This mimics the coverage gaps that arise in OOD geometries and should increase robustness to near-surface geometry variation.

This is analogous to DropPath / structured dropout in vision transformers, but applied in the physics space (SDF bins) rather than the network depth dimension. The near-surface zone (|sdf| < 0.1m) accounts for the highest-stakes vol_p predictions — getting these right in OOD geometries is the core problem.

### Implementation notes
In `model.py`, vol forward pass before xattn:

```python
# Physics-zone vol token masking (H5)
if training and self.use_sdf_zone_masking:
    sdf = vol_x[:, :, 3]  # [B, N_vol]
    near_surf_mask = (sdf > -0.3) & (sdf < 0.05)
    drop_mask = near_surf_mask & (torch.rand_like(sdf) < self.sdf_zone_p_drop)
    vol_h = vol_h.masked_fill(drop_mask.unsqueeze(-1), 0.0)
    volume_mask = volume_mask & ~drop_mask
```

Annealed schedule: start `p_drop=0.2` at EP1, linearly decay to 0.05 by EP4. This avoids early convergence disruption.

Risk: near-surface tokens carry high-gradient vol_p signals; dropping them in training may slow convergence noticeably at EP1. Kill gate should allow up to 14% at EP1.

### Suggested experiment design
```
--epochs 4 --lr-cosine-t-max 13 --use-sdf-zone-masking --sdf-zone-p-drop 0.15
--vol-curriculum 0:16384:1:32768:2:49152:3:65536
```

### Taste rubric
- Research mode: **diagnostic / OOD regularization** (category c)
- Mechanistic grounding: **2** — plausible mechanism but the link between near-surface training dropout and OOD geometric shifts is indirect; the 4 OOD cases have different body geometry, not just different coverage
- Research-state value: **2** — result is hard to interpret cleanly because the dropout zone and the OOD geometry shift may not be aligned
- Execution value: **3** — model.py only, ~15 lines, fast to implement

---

## Ranking Summary

| Rank | ID | Category | Mechanism | Risk | Params | Code scope |
|------|----|----------|-----------|------|--------|------------|
| 1 | **H1** | pre-xattn capacity | Query refinement vol self-attn before xattn | Low | +2.1M | model.py ~15 lines |
| 2 | H2 | geometry/PE | SDF-conditioned per-octave RFF scaling | Low | +112 | model.py ~20 lines |
| 3 | H3 | geometry/PE | Learned global geom Q-bias for xattn | Medium (overfit) | +394K | model.py ~10 lines |
| 4 | H4 | OOD aug | Manifold Mixup on backbone hidden states | Medium (convergence) | 0 | trainer_runtime.py |
| 5 | H5 | OOD aug | SDF-range stochastic vol token masking | Medium-low | 0 | model.py ~15 lines |

**TOP PICK: H1 — Pre-xattn vol self-attention block.**

Reasons: (1) directly tests the one untested architectural position relative to a known 0-for-3 closed pattern; (2) mechanism is precise and falsifiable; (3) zero-init preserves convergence trajectory; (4) model.py only, ~15 lines, no trainer changes needed; (5) 4-ep screen will definitively answer whether pre-xattn vol-vol communication improves query quality.

If H1 wins, follow with H2 (SDF-modulated PE) as the natural next hypothesis — both target the pre-xattn vol representation, and their effects are orthogonal (H1 adds spatial communication, H2 adds physics-informed frequency adaptation).

---

## Research State Update

**Current best explanation for vol_p test gap:** 4 OOD test cases have car geometries outside the training distribution. The model's vol Q-projections for surf→vol xattn are formed without spatial neighborhood context (no vol-vol communication pre-xattn) and without geometry-adaptive positional encoding. Both of these make the model's cross-attention read-out from the surface sensitive to geometric OOD shift.

**Ruled out:** post-xattn vol capacity (0-for-3), depth scaling, width scaling, loss reweighting, n_heads != 4, detach K/V, multi-layer xattn, fixed cross-attention positional bias.

**Open uncertainties:**
1. Is the OOD gap recoverable with single-model architectural changes, or does it require ensemble diversity?
2. Does pre-xattn vol-vol context improve xattn quality enough to matter at the 4-case OOD level?
3. Are the 4 OOD cases fundamentally out-of-distribution for the training geometry manifold, making regularization approaches ineffective without additional training data?

**Stop condition for H1:** EP3 val_abupt > 8.0% → close, vol-vol pre-xattn is not the bottleneck. Move to H2.
**Stop condition for H2:** EP3 val_abupt > 8.0% → close, SDF-conditioned PE is not providing additional signal. Move to H3.
**Stop condition overall:** If H1+H2+H3 all fail, the OOD gap is likely not recoverable by single-model architectural changes. Escalate to ensemble-diversity approaches or data augmentation targeting the specific 4-case geometry distribution.
