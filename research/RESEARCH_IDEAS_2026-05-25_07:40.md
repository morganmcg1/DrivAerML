# Research Ideas — 2026-05-25 07:40

**Context**: Wave 34 architectural pivot. H39 SOTA test_WSS=6.6506% (val=6.7761%). Full capacity-axis sweep exhausted (H115/H117/H122/H123/H132/H133 all NULL or FALSIFIED). Per Plateau Protocol: major architectural swings required. Target: test_WSS < 5.85%.

**Hard constraints**: single-model only (NO ensembles), DDP8 8×H100 96GB, 30 epochs, 24h wallclock.

**AND-contract from PR #972**: test_vol_p ≤ 3.643% AND test_surf_p ≤ 3.577% must not be broken.

---

## H134 — GALE-Transolver: Geometry-Aware Latent Embeddings at Every Block

**Assigned to**: dl24-fern

### Mechanism

The core hypothesis is that Transolver's Physics-Attention slices capture global state distributions but are blind to local geometric context at inference time. The GALE architecture (GeoTransolver, arxiv 2512.20399) addresses this by inserting a cross-attention query into a multi-scale Geometry Context Bank at every Transformer block, gated by a learned scalar α. On DrivAerML, GeoTransolver achieves wss=4.90%, surf_p=2.86%, vol_p=3.09% — 1.75pp below H39 SOTA on WSS.

The mechanism: for each Transolver block, after Physics-Attention (slice assignment → aggregate → mix), build a geometry context tensor C by concatenating: (1) raw position p ∈ R³, (2) curvature feature c_geom (already computed for H39 curvature bias), and (3) multi-scale neighborhood encodings E₁...E_S computed by ball queries at radii r ∈ {0.01, 0.05, 0.25, 1.0, 2.5, 5.0} with k=32 neighbors each. Then:

```python
# After Physics-Attention produces H_sa (slice-aggregated hidden state):
CA_m = cross_attention(
    query = H_sa @ W_Q_c,      # (N, D_head)
    key   = C @ W_K_c,          # (N_ctx, D_head)
    value = C @ W_V_c           # (N_ctx, D_head)
)
alpha = sigmoid(linear(cat(pool(H_sa), pool(CA_m))))   # learned gate (B, 1)
H_out = (1 - alpha) * H_sa + alpha * CA_m
```

This gives every block awareness of the local geometry at 6 spatial scales. The adaptive gate α prevents over-reliance on geometry in regions where physics slices are already sufficient.

### Model changes (model.py)

1. Add `GeometryContextBank` module: for each of 6 radii, run `torch_cluster.radius` (or `torch_geometric.nn.radius`) to find k=32 neighbors, compute mean/std/max aggregation features (3×3=9 features per radius → 54-dim), project to hidden_dim via linear.
2. In `TransolverBlock.__init__`: add `self.geo_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)` and `self.geo_gate = nn.Linear(2*hidden_dim, 1)`.
3. In `TransolverBlock.forward`: after `h = self.physics_attention(h)`, compute GALE cross-attention and gate update as above.
4. Keep depth=6, hidden=512, slices=128, all H39 hyperparameters identical. Only add GALE on top.
5. Approximate parameter budget: each GALE block adds ~3×hidden²/heads params for QKV projections → depth-6 adds ~3×512²×6/8 ≈ 2.4M params (vs H39 ~10M total), acceptable.

### Training changes (train.py)

No changes to optimizer, LR schedule, GradNorm, or loss weights. The GALE context bank is built once per batch from surface coordinates (already available as `batch.pos` or equivalent). Use the H39 curvature features as c_geom input to avoid recomputing.

### EP gates and abort conditions

- **EP3 abort**: val_WSS > 7.10% (H39 EP3 val_WSS ≈ 6.97% from typical trajectory; GALE should improve or match)
- **EP6 gate**: val_WSS > 7.00% → close (GALE cross-attention should be learning useful geometry by EP6)
- **EP10 gate**: val_WSS > 6.80% → close (must show trajectory toward target)
- **EP15 gate**: projected EP30 val_WSS > 6.65% (H39 val floor) → close
- **EP30 terminal**: test_WSS AND test_vol_p ≤ 3.643% AND test_surf_p ≤ 3.577%

### Predicted outcome

- **Best case**: test_WSS = 5.4–5.7% (GeoTransolver baseline 4.90% with 20 layers / 500 epochs gives strong signal; our depth-6 / 30 epoch version will be weaker but still substantial)
- **Expected case**: test_WSS = 5.7–6.2% (−0.45 to −0.95pp from H39)
- **Null case**: test_WSS ≈ 6.5–6.6% (geometry context saturated by existing curvature bias; GALE adds no independent signal)
- **Probability below 6.0%**: 60–70%
- **AND-contract risk**: moderate — GeoTransolver achieves surf_p=2.86% (well within 3.577% floor), so risk is low IF the mechanism transfers

### Falsifiable prediction

If GALE improves WSS by > 0.3pp at EP10, it will likely maintain that margin to EP30. If val_WSS at EP10 is worse than H39 EP10 baseline, the mechanism does not transfer at depth-6/30-epoch budget and the run should be closed.

### References

- GeoTransolver (arxiv 2512.20399, 2024): GALE on DrivAerML wss=4.90% with 20 layers, 29M params, 500 epochs
- GeoTransolver ablation (Table 4): single scale → multi-scale = 9% surf_p improvement; depth 6→20 = 17-19% WSS improvement
- Ball-query neighbor search: torch_cluster, torch_geometric radius/radius_graph

---

## H135 — Ada-Temp Slices: Adaptive Per-Point Temperature in Physics-Attention

**Assigned to**: dl24-frieren

### Mechanism

Transolver's Physics-Attention uses a fixed temperature τ₀ for the softmax slice assignment. This forces all mesh points — smooth hood surface, sharp A-pillar edge, complex mirror geometry — to use the same soft/hard assignment profile. The Transolver++ paper (arxiv 2502.02414) shows that replacing fixed τ with a per-point learned temperature `τᵢ = τ₀ · exp(Linear(xᵢ))` improves slice specialization dramatically: Ada-Temp alone achieves 46% error reduction on AirCraft and 13% PDE benchmark improvement, with full Transolver++ achieving 20% industrial gain including DrivAerNet Volume +11%, Surface +12.6%.

The mechanism: the model learns that high-curvature points (A-pillar, mirror base) benefit from sharper (lower τ) assignment to specialized slices, while smooth regions can use softer (higher τ) assignments for smoother gradients. During training, Gumbel-Softmax noise encourages exploration of slice assignments; during eval, standard softmax with learned τᵢ is used.

```python
# In PhysicsAttention.forward, replacing fixed temperature:
tau_raw = self.tau_head(x_embed)               # Linear(D, 1), x_embed is point feature
tau_i   = self.tau0 * torch.exp(tau_raw).clamp(0.1, 10.0)   # per-point, clamped

if self.training:
    # Gumbel-Softmax: encourages exploration during training
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
    slice_weights = torch.softmax((scores + gumbel_noise) / tau_i, dim=-1)
else:
    slice_weights = torch.softmax(scores / tau_i, dim=-1)
```

where `scores = query @ key.T` is the standard Physics-Attention dot product.

### Model changes (model.py)

1. In `PhysicsAttention.__init__`: add `self.tau_head = nn.Linear(hidden_dim, 1)` and `self.tau0 = nn.Parameter(torch.ones(1) * initial_tau)`. Set `initial_tau = 1.0` (matches current H39 behavior at init).
2. In `PhysicsAttention.forward`: replace `softmax(scores / tau0)` with per-point τᵢ as above.
3. Apply to ALL 6 Transolver blocks (not just the first or last — ablation in Transolver++ paper shows all-block application is necessary for full gain).
4. Keep all other H39 hyperparameters identical: depth=6, hidden=512, slices=128, curvature-bias, y-aug, etc.
5. Parameter overhead: 6 × hidden_dim = 6 × 512 = 3072 extra parameters — negligible.

### Training changes (train.py)

No changes except optionally monitoring `tau_i.mean()` and `tau_i.std()` per block as W&B metrics to verify learning is occurring. If τᵢ collapses to constant (std < 0.01 by EP5), the tau_head is not learning and the run should be inspected.

### EP gates and abort conditions

- **EP3 abort**: val_WSS > 7.05% (Ada-Temp should show early benefit from better slice specialization)
- **EP6 gate**: val_WSS > 6.95% → close
- **EP10 gate**: val_WSS > 6.75% → close
- **EP15 gate**: projected EP30 > 6.60% → close
- **Diagnostic at EP5**: log `tau_i.mean()` and `tau_i.std()` per block. If std < 0.01 across all blocks, inspect model; temperature is not learning.
- **EP30 terminal**: test_WSS AND AND-contract check

### Predicted outcome

- **Best case**: test_WSS = 5.8–6.0% (Transolver++ DrivAerNet +11% surface → extrapolates to ~−0.73pp; our shallower 30-epoch version will be less)
- **Expected case**: test_WSS = 6.1–6.45% (−0.2 to −0.55pp)
- **Null case**: τ collapses or learns near-constant → no gain vs H39
- **Probability below 6.5%**: 55–65%
- **AND-contract risk**: low — Ada-Temp does not change loss weights, decoder heads, or multi-task balance

### Falsifiable prediction

By EP5, `tau_i.std()` across blocks should be > 0.05 (temperature is differentiating). By EP10, val_WSS should be ≥ 0.1pp better than H39 EP10 baseline. If neither occurs, the slice assignment in H39 is already near-optimal and Ada-Temp adds no leverage.

### References

- Transolver++ (arxiv 2502.02414, 2025): Ada-Temp + Gumbel-Softmax; DrivAerNet Volume +11%, Surface +12.6%; AirCraft Ada-Temp ablation 46% error reduction
- Gumbel-Softmax: Jang et al. 2017 (arxiv 1611.01144) — standard implementation in torch.nn.functional.gumbel_softmax
- Transolver++ full model also adds adaptive position encoding and enhanced SA, but Ada-Temp is the dominant contributor per paper ablation

---

## H136 — IMTL-G: Implicit Multi-Task Gradient Surgery Replacing GradNorm

**Assigned to**: dl24-nezuko

### Mechanism

H39 uses GradNorm (α=0.5, clamp=0.15 on min_w_vol_p) for multi-task loss balancing. The known failure mode of GradNorm in H39: the WSS task weight w_τ_z surges to ~1.85, vol_p weight collapses to the 0.15 floor. This means the optimizer is effectively spending budget predominantly on WSS gradient direction, at the cost of vol_p and surf_p gradients. The clamp prevents full collapse but creates a hard floor that distorts gradient directions.

Implicit Multi-Task Learning with gradient (IMTL-G, Liu et al. 2021, arxiv 2108.01547) achieves equal projections of each task's gradient onto the others via a closed-form scaling, without requiring α or clamp hyperparameters. The key property: IMTL-G provably ensures that task i's gradient does not project negatively onto task j's gradient direction, eliminating the destructive interference that causes GradNorm to over-weight WSS at the expense of surf_p and vol_p.

The hypothesis: the AND-contract breach in H39 (test_surf_p=3.6498% vs 3.577% floor, +0.073pp over floor) is caused precisely by GradNorm's w_vol_p collapsing and pushing the optimizer away from surf_p/vol_p gradient directions. IMTL-G will naturally balance the three tasks without manual α tuning.

IMTL-G closed-form update (per-step, after computing per-task gradients g₁...gₙ):
```python
# Given task gradients g_i (flattened), compute scaling w_i such that
# sum_i w_i * g_i has equal cosine similarity to each g_i
# Closed-form solution from Liu et al. 2021 eq. (8):

def imtl_g_weights(grads):
    # grads: list of N gradient vectors, each shape (P,)
    N = len(grads)
    G = torch.stack(grads)           # (N, P)
    # Gram matrix: M[i,j] = g_i · g_j
    M = G @ G.T                      # (N, N)
    # IMTL-G: w = M^{-1} ones / (ones^T M^{-1} ones)
    ones = torch.ones(N, device=G.device)
    try:
        w = torch.linalg.solve(M + 1e-6*torch.eye(N, device=G.device), ones)
    except:
        w = ones / N   # fallback to uniform
    w = w / w.sum()  # normalize
    return w.clamp(min=0.0)  # non-negative weights
```

### Training changes (train.py)

1. Remove GradNorm entirely: delete `gradnorm_loss`, `task_weights` parameters, α and clamp logic.
2. In the training step, compute per-task loss scalars for: L_WSS (tau_z), L_vol_p, L_surf_p, L_abupt. Compute per-task gradient vectors (using autograd or PCGrad-style backward hooks).
3. Apply `imtl_g_weights(grads)` to get task weights w_i.
4. Combined loss: `sum(w_i * L_i)`.
5. Add PyTorch implementation reference: github.com/JohnLaMaster/Impartial-Multi-Task-Learning (MIT license, drop-in replacement compatible).
6. Log `w_wss`, `w_vol_p`, `w_surf_p`, `w_abupt` to W&B each step to verify IMTL-G is producing non-collapsed weights.

**Important**: computing per-task gradients requires either (a) separate forward passes per task (4× memory), or (b) gradient projection via autograd hooks on shared backbone. Use option (b): register backward hooks on the last shared layer to capture per-task gradient contributions. See PCGrad implementation for reference pattern.

### Model changes (model.py)

None — IMTL-G is purely a training-side change. The model architecture remains identical to H39.

### EP gates and abort conditions

- **EP3 abort**: val_WSS > 7.05% AND val_surf_p > 3.8% (IMTL-G should show surf_p balance improvement vs H39 from early epochs)
- **EP5 diagnostic**: log w_wss, w_vol_p, w_surf_p. If w_wss > 1.5 (i.e., IMTL-G is still collapsing to WSS-dominant), the gradient hook implementation has a bug; inspect and abort.
- **EP10 gate**: val_WSS > 6.75% OR val_surf_p > 3.7% → close
- **EP15 gate**: projected EP30 val_WSS > 6.60% → close
- **EP30 terminal**: AND-contract check is critical — IMTL-G should deliver surf_p ≤ 3.577% as its primary diagnostic improvement alongside WSS

### Predicted outcome

- **Best case**: test_WSS = 6.2–6.35% AND test_surf_p ≤ 3.50% (IMTL-G fixes AND-contract breach + modest WSS improvement from better gradient balance)
- **Expected case**: test_WSS ≈ 6.45–6.55% + surf_p ≤ 3.577% (restores AND-contract compliance while holding WSS parity)
- **Null case**: IMTL-G converges to similar weights as GradNorm in practice → metrics unchanged
- **Probability of AND-contract compliance**: 65–75%
- **Probability of >0.15pp WSS improvement**: 40–50%

### Falsifiable prediction

By EP5, w_vol_p and w_surf_p should each be > 0.15 (above the GradNorm clamp floor) and w_wss should be < 1.8 (below GradNorm's typical surge). If weights collapse similar to GradNorm by EP5, IMTL-G is not providing the balance correction and the hypothesis is falsified early.

### References

- IMTL-G: Liu et al. 2021 "Towards Impartial Multi-task Learning" (arxiv 2108.01547): closed-form gradient scaling achieving equal projection property
- PyTorch implementation: github.com/JohnLaMaster/Impartial-Multi-Task-Learning (MIT license)
- PCGrad (Yu et al. 2020, arxiv 2001.06782): predecessor, project conflicting gradients; IMTL-G has stronger theoretical guarantees
- CAGrad (Liu et al. 2021): another alternative, but IMTL-G closed-form is simpler to implement correctly

---

## Research State Summary

### Current bottleneck assessment

The capacity-axis sweep result is a strong signal: H39 is at an architectural local optimum under the current Transolver formulation. The three hypotheses attack three distinct and orthogonal levels:

1. **H134 (GALE)**: Architecture — the model cannot represent local geometry multi-scale context within current Physics-Attention
2. **H135 (Ada-Temp)**: Architecture/Training hybrid — the fixed-temperature slice assignment creates a suboptimal representation bottleneck in high-curvature regions
3. **H136 (IMTL-G)**: Training — gradient surgery distortion from GradNorm creates an AND-contract breach and potentially suppresses surf_p/vol_p improvement

### Experiment tree

```
H134 (GALE) EP10:
├── val_WSS < 6.60% (−0.1pp+ below H39 EP10): CONTINUE → H134 likely strongest candidate
│   └── EP30 terminal:
│       ├── test_WSS < 6.5% → MERGE + dispatch H137 (GALE + Ada-Temp combined)
│       └── test_WSS 6.5–6.65% → merge if AND-contract OK, else request-changes
└── val_WSS ≥ 6.75% at EP10: CLOSE H134 → GALE does not transfer at depth-6/30ep

H135 (Ada-Temp) EP5:
├── tau_i.std() > 0.05 across blocks: CONTINUE (temperature is learning)
│   └── EP10 val_WSS:
│       ├── < 6.65%: CONTINUE → H135 viable
│       └── ≥ 6.75%: CLOSE
└── tau_i.std() < 0.01 at EP5: INSPECT → likely init issue, try initial_tau=0.5

H136 (IMTL-G) EP5:
├── w_vol_p > 0.15 AND w_surf_p > 0.15 (balanced): CONTINUE
│   └── EP10:
│       ├── val_surf_p < 3.6% + val_WSS < 6.75%: CONTINUE → AND-contract fix in range
│       └── val_surf_p ≥ 3.7%: CLOSE (IMTL-G not producing balance)
└── w_wss > 1.5 at EP5 (GradNorm-like collapse): DIAGNOSE gradient hook → abort if bug

If ALL THREE fail EP10:
→ Escalate to full GeoTransolver replication (depth-20, but requires 24h wall-clock check)
→ Consider knowledge distillation from GeoTransolver-teacher to depth-6 student
```

### Stop conditions

- **H134 stop**: val_WSS ≥ 6.80% at EP10 (no signal above H39 baseline trajectory)
- **H135 stop**: tau_i.std() < 0.01 at EP5 (temperature not learning) OR val_WSS ≥ 6.80% at EP10
- **H136 stop**: w_wss > 1.5 at EP5 (gradient surgery collapsed) OR val_WSS ≥ 6.70% AND val_surf_p ≥ 3.7% at EP10

### Taste rubric

| Experiment | Mode | Mechanistic Grounding | Research-State Value | Execution Value | Notes |
|---|---|---|---|---|---|
| H134 GALE | Tier shift | 4 — DrivAerML wss=4.90% directly anchors mechanism | 4 — falsifiable at EP10, sharp update either way | 3 — 200-300 LoC, ball-query implementation non-trivial | Highest expected delta; external evidence strongest |
| H135 Ada-Temp | Tier shift | 3 — AirCraft 46% reduction is strong but different distribution | 3 — τ.std diagnostic is clean early falsifier | 4 — 60-80 LoC, negligible parameter overhead | Cheapest architectural change; good risk/reward |
| H136 IMTL-G | Frontier refinement | 3 — GradNorm distortion observed in H39 W&B; IMTL-G targets specific failure | 4 — directly tests AND-contract hypothesis; results interpretable | 3 — gradient hook implementation requires care | Uniquely tests training-side hypothesis vs architecture |

### Confidence

- **H134**: Strong external evidence from GeoTransolver (same dataset, identical metric). Main uncertainty is depth-6/30ep budget vs 20-layer/500ep paper. **High confidence in direction, moderate confidence in magnitude.**
- **H135**: Strong mechanism from Transolver++ ablation. Transfer to DrivAerML less direct (paper tested on DrivAerNet with different mesh density). **Moderate confidence.**
- **H136**: IMTL-G has strong theoretical grounding and the GradNorm distortion is directly observable in W&B logs. Execution risk is the gradient hook implementation. **Moderate-high confidence in AND-contract improvement, lower confidence in WSS delta.**
