# Wave-5 Hypothesis Catalog — 2026-05-30 21:30Z

## Context

- Plateau Protocol triggered: H167-H180 = 14+ consecutive NON-MERGEs
- dl24 SOTA: H147 test_WSS=6.5409%, test_VP=3.4014%, test_SP=3.5634%, test_ABUPT=5.6648%
- Paper SOTA target: Transolver-3 test_WSS < 5.85% (gap = 0.69pp)
- Exhausted search space: Lion β-grid, GradNorm α-grid, EMA decay variants, architecture (layers/heads/slices/hidden_dim), Charbonnier axes, PE sigmas, LR/WD sweep, Y-sym aug
- Primary bottleneck: τ_y=7.056% vs target 3.65%; τ_z=8.488% vs target 3.63% — 3-5pp gap
- GradNorm terminal weight profile: w_τ_z=1.749 (highest), w_τ_y=1.155 → confirms τ_z is the dominant bottleneck

## Constraints

- Single-model DDP8 only
- 24h wall-clock max (~30 epochs at current throughput)
- All 8 GPUs must be used (no split arms)
- Ensembles BANNED; model-soup / NNLS BANNED
- Must register improvement on primary metric: test_WSS (lower is better)

---

## H-W5-1: WSD Learning Rate Schedule

**Priority: HIGH**

**Short title:** Warmup-Stable-Decay (WSD) schedule to prevent premature LR collapse

**Mechanism:**
Replace cosine annealing (T_max=30, lr_min≈0) with a three-phase schedule:
- Phase 1 (warmup): epochs 0-1, linear ramp 0→1e-4 (identical to current)
- Phase 2 (stable): epochs 1-18 (~60%), constant LR=1e-4; model sees full gradient signal longer
- Phase 3 (decay): epochs 18-30 (~40%), cosine decay to LR_min=1e-6

The EMA experiments (H172, H181) confirmed the best per-epoch validation improvements coincide with the active cosine descent phase. The current T_max=30 cosine schedule begins decaying LR immediately from epoch 1, so the model spends most of training in a sub-optimal LR regime. WSD extends the flat-LR phase where gradient signal is richest, then applies a clean final descent. This is the schedule used by Mistral, Llama-3, and DeepSeek training runs that outperformed cosine across long sequences. It directly addresses the EMA finding without adding any model parameters.

**Falsifying signal:**
- EP30 val_WSS ≥ 6.5451 (H147 terminal val) → schedule change irrelevant
- val_WSS begins rising after stable→decay transition → optimal stable fraction too long

**Expected gain:** −0.05 to −0.15pp test_WSS; conservative because schedule is the only change

**Implementation cost:** LOW
- Add `--lr-schedule wsd --lr-stable-frac 0.6 --lr-warmup-epochs 1` to train.py LR scheduler logic
- No model changes; no new dependencies
- Ablate stable-frac in {0.4, 0.5, 0.6} — start with 0.6

**Key reference:** Hu et al. "MiniCPM" (2024); Defazio et al. "Schedule Free" (2024); standard practice in frontier LLM training

**Exact suggested command (delta from H147):**
```
# Replace: --lr-cosine-t-max 30 --lr-warmup-epochs 1
# Add:     --lr-schedule wsd --lr-stable-frac 0.6 --lr-warmup-epochs 1
```

---

## H-W5-2: Per-Channel Decoder Heads (Split surface_out MLP)

**Priority: HIGH**

**Short title:** Split surface MLP into 4 per-channel heads (cp, τ_x, τ_y, τ_z)

**Mechanism:**
The current architecture routes all 4 surface outputs through a single shared MLP:
`surface_out: [512] → [1024] → [4]` (with `surface_out_width_factor=2.0`)

Replace with 4 independent per-channel MLPs, each:
`head_k: [512] → [1024] → GELU → [1024] → [1]`

This gives τ_y and τ_z dedicated nonlinear pathways without requiring shared capacity to simultaneously represent cp (which is already well-fit at 3.5%) and τ_z (which is 8.5%). The shared head creates a capacity bottleneck: a single set of weights must simultaneously model near-zero cp-dominated features and high-gradient τ_z features in the wake region. H39's own follow-up suggestion explicitly flagged "per-channel decoder heads" as an open frontier.

The total parameter count increases by ~3× in the decoder only (from 4×512×2=4M to 4×(512×1024+1024×1)=~2M per head × 4 = ~8M extra), which is a small fraction of the full Transolver encoder.

**Falsifying signal:**
- EP15 val_WSS ≥ H147 EP15 val_WSS (6.5466%) → dedicated heads add no signal at mid-training
- val_tau_z does not improve relative to H147 → capacity was not the bottleneck

**Expected gain:** −0.1 to −0.3pp test_WSS; highest upside of any wave-5 idea because it directly targets the 3-5pp τ_y/τ_z gap

**Implementation cost:** MEDIUM
- Modify `model.py`: replace final linear layer with `nn.ModuleList([MLP(hidden, 1) for _ in range(4)])`
- No change to encoder or physics-attention block
- Optional: also try width_factor=3× or 4× on the per-channel heads

**Exact suggested command (delta from H147):**
```
# Replace: --surface-out-width-factor 2.0
# Add:     --per-channel-surface-heads --per-channel-head-width 1024
# (or equivalent flag names as implemented)
```

---

## H-W5-3: τ_y GradNorm Floor (w_τ_y ≥ 0.30)

**Priority: MEDIUM**

**Short title:** Pin GradNorm τ_y weight floor at 0.30 to prevent starvation during cooldown

**Mechanism:**
H176 (trace confirmed) showed that when GradNorm clamps vol_p weight at floor=0.15, the freed budget redistributes primarily to τ_x and τ_y — but during cosine cooldown, τ_y weight subsequently collapses toward 1.0/N=0.25. The vol_p floor (merged in H147) prevents vol_p starvation; the same mechanism applies to τ_y.

Terminal GradNorm weight profile from H147: w_τ_y=1.155, which is above parity but near the theoretical unattended floor. During the final 5 epochs when cosine LR is near-zero, GradNorm updates slow dramatically and w_τ_y may drift down without a floor.

The τ_y test gap (7.056% vs target 3.65%) is 2× larger than τ_z and both are 3-5pp above paper target. A floor at 0.30 (vs vol_p floor at 0.15) biases the loss budget proportionally toward τ_y.

**Falsifying signal:**
- EP30 logged w_τ_y never hits floor (always > 0.30) → mechanism was not active, floor had no effect
- val_VP degrades significantly (>0.1pp) while val_WSS improves → trade-off too steep

**Expected gain:** −0.05 to −0.10pp test_WSS; −0.10 to −0.20pp test per-axis τ_y specifically

**Implementation cost:** LOW
- One-line addition to GradNorm clamp logic: `--gradnorm-min-w-tau-y 0.30`
- Can stack with current vol_p floor=0.15

**Exact suggested command (delta from H147):**
```
# Add: --gradnorm-min-w-tau-y 0.30
# Keep: --gradnorm-min-w-vol-p 0.15 --gradnorm-alpha 0.5
```

---

## H-W5-4: Muon / NorMuon Optimizer

**Priority: MEDIUM**

**Short title:** Replace Lion with Muon (Newton-Schulz orthogonalized gradient updates)

**Mechanism:**
Lion optimizer was merged in H147 and the β-grid (β1∈{0.93,0.95,0.97}×β2∈{0.97,0.98,0.985,0.99}) is exhausted. Lion's update rule is `sign(β1·m + (1-β1)·g)` — it uses gradient sign only. Muon (Momentum + Newton-Schulz orthogonalization) maintains a momentum buffer and applies a Newton-Schulz iteration to orthogonalize the gradient matrix before applying it. This approximates the whitened steepest-descent direction for weight matrices, which is known to work especially well for attention weight matrices.

Recent empirical results (Kosson et al. "Muon", Jordan et al. 2024): Muon matched or beat AdamW on GPT-2 scale transformers at lower LR (0.02 vs 6e-4). The "NorMuon" variant (Bernstein 2024) applies the same idea with spectral normalization for stability. For the Transolver physics-attention heads, which have learned a strong geometry-encoding inductive bias, a curvature-aware update rule could escape local optima that sign-based Lion cannot.

The LR scale for Muon is different: typically 0.01-0.05 for transformer weights vs 1e-4 for Lion. This requires a LR sweep, but the mechanism is orthogonal to all explored directions.

**Falsifying signal:**
- EP3 val_WSS ≥ H147 EP3 (6.9754%) → Muon provides no early-epoch advantage; abort
- Training loss diverges in EP1 → LR too high, reduce by 10×

**Expected gain:** −0.05 to −0.20pp test_WSS (high uncertainty; completely unexplored optimizer family here)

**Implementation cost:** MEDIUM
- Install `muon` package (open-source: https://github.com/KellerJordan/Muon) or implement Newton-Schulz in ~15 lines
- Apply Muon only to attention+MLP matrices; use AdamW for embeddings and biases (standard practice)
- Tune LR: start at 0.02, weight_decay=0.0
- EMA and GradNorm remain unchanged

**Key reference:**
- Jordan et al. "Muon: An optimizer for hidden layers in neural networks" (2024) https://github.com/KellerJordan/Muon
- Bernstein et al. "Old Optimizer, New Norm" (NorMuon, 2024)

**Exact suggested command (delta from H147):**
```
# Replace: --optimizer lion --lr 1e-4 --weight-decay 0.005 --lion-beta1 0.95 --lion-beta2 0.98
# Add:     --optimizer muon --lr 0.02 --muon-momentum 0.95 --weight-decay 0.0
# (embeddings/biases use AdamW lr=1e-3 via param-group split)
```

---

## H-W5-5: Sobolev Auxiliary Loss on WSS Spatial Gradient

**Priority: LOW-MEDIUM**

**Short title:** Add surface-gradient coherence auxiliary loss on WSS prediction

**Mechanism:**
All 196 prior PRs optimize per-point L1/L2/Charbonnier losses. None penalize spatial incoherence. The τ_z residuals (8.5% MAPE) are concentrated at sharp geometry transitions (A-pillar, mirror base, wheel arch), where the predicted WSS field has abrupt discontinuities relative to the reference CFD solution. A Sobolev-norm auxiliary loss penalizes:

`L_sob = λ · mean(|∇_surface WSS_pred − ∇_surface WSS_true|²)`

where surface gradients are estimated via kNN finite difference over the mesh topology. This regularizes the output to be spatially smooth in the same way the CFD solution is, targeting the high-frequency per-voxel error that per-point losses cannot capture.

Related work: Rixner & Šarić "Physics-informed Gaussian process regression" (2021); Horie & Mitsume "Physics-embedded neural networks" (2021); standard in PDE-constrained optimization literature.

**Falsifying signal:**
- Training loss oscillates or diverges → gradient-of-gradient is numerically unstable on sparse mesh
- val_WSS ≥ H147 at EP30 → spatial coherence is not the bottleneck

**Expected gain:** −0.05 to −0.15pp test_WSS on τ_z/τ_y axes; higher if geometry-transition regions drive the error floor

**Implementation cost:** HIGH
- Requires precomputing kNN neighbor indices for surface mesh (~1h offline)
- New loss term in `train.py`: finite-difference WSS gradient over k=8 nearest neighbors
- Tune λ ∈ {0.01, 0.05, 0.1}; risk of instability requires gradient clipping
- Not recommended as first wave-5 experiment; best run after H-W5-1 and H-W5-2 results are in

**Exact suggested command (delta from H147):**
```
# Add: --sobolev-wss-weight 0.05 --sobolev-knn 8
```

---

## Priority Order Summary

| Rank | ID | Title | Priority | Cost | Expected gain |
|------|----|-------|----------|------|---------------|
| 1 | H-W5-1 | WSD LR Schedule | HIGH | LOW | −0.05 to −0.15pp WSS |
| 2 | H-W5-2 | Per-Channel Decoder Heads | HIGH | MEDIUM | −0.1 to −0.3pp WSS |
| 3 | H-W5-3 | τ_y GradNorm Floor | MEDIUM | LOW | −0.05 to −0.10pp WSS |
| 4 | H-W5-4 | Muon Optimizer | MEDIUM | MEDIUM | −0.05 to −0.20pp WSS |
| 5 | H-W5-5 | Sobolev Auxiliary Loss | LOW-MEDIUM | HIGH | −0.05 to −0.15pp WSS |

## Recommended First Assignments

Given 3 active WIP PRs (H172/tanjiro, H181/frieren, H178/fern) and current plateau:

- **When next student becomes idle:** Assign H-W5-1 (WSD schedule) — lowest risk, directly motivated by EMA finding
- **Second idle student:** Assign H-W5-2 (per-channel heads) — highest upside, explicitly flagged by H39 follow-up
- **Third idle student:** Assign H-W5-3 (τ_y pin) — lowest implementation cost, stackable with current stack

## Decision Tree

```
H-W5-1 (WSD) →
  SUCCESS (val_WSS < H147): merge, then assign H-W5-2 on top of WSD winner
  FAILURE (val_WSS ≥ H147): assign H-W5-2 in parallel; WSD schedule is ruled out
    H-W5-2 (per-channel heads) →
      SUCCESS: merge; then assign H-W5-3 as stacking experiment
      FAILURE: assign H-W5-4 (Muon) — different mechanism entirely
        H-W5-4 (Muon) →
          SUCCESS: merge; re-examine schedule and head experiments on top
          FAILURE: assign H-W5-5 (Sobolev) as last-resort high-cost idea
                   or escalate to full architecture replacement (Performer, GNO)
```

## Research State Note

The wave-5 catalog is motivated by the Plateau Protocol: 14+ consecutive NON-MERGEs spanning the full local hyperparameter neighborhood. The ideas above represent a deliberate shift from:
- hyperparameter tuning (LR, WD, β) → done
- adaptive loss weights (GradNorm α, floors) → done
- augmentation (Y-sym, SDF stratification) → done

To genuinely new mechanisms:
- Training schedule family (WSD vs cosine)
- Decoder architecture restructuring (per-channel heads)
- Loss weight targeting (τ_y pin)
- Optimizer family replacement (Muon)
- Loss formulation augmentation (Sobolev)

The τ_y/τ_z gap (3-5pp above paper target) is the primary bottleneck. Any experiment that does not address this gap directly is lower priority regardless of other considerations.
