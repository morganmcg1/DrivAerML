# Research Ideas — 2026-05-16 22:35

## H17 — Local Tangent-Frame Output Reparameterization for WSS

### One-sentence summary

Instead of predicting τ_WSS in global (x, y, z) coordinates, predict two scalar coefficients (α_t1, α_t2) in the local surface tangent basis and reconstruct the global-frame WSS vector inside `forward()` — enforcing τ·n=0 by construction rather than as a soft penalty.

---

### Why this might help here

Every closed Wave 30 experiment that attacks τ_z from the model side lands in the same collapse band: τ_z/τ_x ∈ [1.44, 1.55] versus the ground-truth ratio of 0.08. Fern H10 EP8 shows that WSS direction is already learned to 99.65% cosine similarity; ALL residual error is in magnitude reconstruction. This narrows the bottleneck to a representational problem in how the model parameterizes the WSS vector magnitude.

The likely mechanism is a spurious geometric correlation in the global frame: on horizontal automotive surfaces (hood, roof, trunk — high-area, high-importance panels), the surface normal n points approximately in the +z direction. This means τ_z in the global frame partially collapses onto the normal direction, which has zero physical content for WSS (τ·n ≡ 0 by definition). The model must learn to suppress τ_z on these panels — an implicit constraint that the global-frame output space does not express naturally. In the local tangent basis (t1, t2 orthogonal to n), this constraint is eliminated by construction: the model only predicts tangential components, and τ_z emerges from the basis rotation rather than being predicted directly.

This is qualitatively different from H6' (tanjiro, in-flight), which adds a soft τ·n=0 penalty after global-frame prediction. H17 changes the OUTPUT REPRESENTATION so the constraint cannot be violated — analogous to the difference between penalizing a neural net to predict positive values versus using a softplus output activation.

---

### Theoretical grounding

**Gao et al. 2024, "An Intrinsic Vector Heat Network"** (ICML 2024 Spotlight, arxiv:2406.09648)
Treats tangent vector fields as intrinsic to the manifold. The core finding: scalar-channel decomposition of surface vectors in global frames fails to preserve intrinsic geometric properties. A local-frame representation is required for the network to learn the correct equivariance structure. Directly supports H17.

**Hendriks et al. 2020, "Linearly Constrained Neural Networks"** (arxiv:2002.01600)
Models output as a linear transformation of an unconstrained latent to guarantee linear constraints. The τ·n=0 constraint is a linear constraint on the output vector; the tangent-frame parameterization is exactly the minimum-rank reparameterization that satisfies it everywhere with zero overhead.

---

### What is NOT being changed

- Loss function (`train_loss`, `per_task_train_losses`) — unchanged, operates on global-xyz `out["surface_preds"]` [B, N, 4]
- Eval code (`EvalAccumulator`, `trainer_runtime.py`) — unchanged, indexes channel 1/2/3 for τ_x/τ_y/τ_z in global xyz
- Optimizer, LR, schedule — canonical Wave 30: Lion lr=9e-5, 13 epochs, batch 4 DDP-8
- All other hyperparameters — baseline config

The ONLY change is in `model.py`.

---

### Implementation — model.py only

All edits are confined to the `SurfaceTransolver` class.

#### 1. Constructor `__init__` — change output dimension from 4 to 3

**File**: `/workspace/senpai/target/model.py`

Locate the `surface_out` definition (around line 484, under `use_aux_decoder_heads=True` path):

```python
# CURRENT:
self.surface_out = nn.Sequential(
    nn.Linear(n_hidden, n_hidden),
    nn.SiLU(),
    nn.Linear(n_hidden, self.surface_output_dim),  # 4 channels: cp, τx, τy, τz
)

# CHANGE TO:
self.surface_out = nn.Sequential(
    nn.Linear(n_hidden, n_hidden),
    nn.SiLU(),
    nn.Linear(n_hidden, 3),  # 3 channels: cp, α_t1, α_t2 (tangent coefficients)
)
```

Note: `self.surface_output_dim` (= `SURFACE_Y_DIM` = 4) is still used by volume/eval paths — do NOT change the attribute. Only change the final Linear layer argument.

#### 2. `forward()` — add tangent-frame reconstruction after surface_out call

**File**: `/workspace/senpai/target/model.py`

Locate the surface prediction block (around line 624):

```python
# CURRENT:
if surface_x is not None:
    surface_preds = self.surface_out(surface_hidden) * surface_mask.unsqueeze(-1)
```

Replace with:

```python
# REPLACEMENT:
if surface_x is not None:
    raw = self.surface_out(surface_hidden)  # [B, N, 3]: (cp, α_t1, α_t2)

    # Build orthonormal tangent frame from surface normals
    # surface_x[:, :, 3:6] contains (nx, ny, nz) — already unit normals in dataset
    n = F.normalize(surface_x[:, :, 3:6], dim=-1)  # [B, N, 3], unit normals

    # Choose reference vector to avoid degeneracy: use e_x unless |n · e_x| > 0.9
    # then fall back to e_y
    e_x = torch.zeros_like(n)
    e_x[..., 0] = 1.0
    e_y = torch.zeros_like(n)
    e_y[..., 1] = 1.0
    dot_x = (n * e_x).sum(dim=-1, keepdim=True).abs()  # [B, N, 1]
    ref = torch.where(dot_x < 0.9, e_x, e_y)  # [B, N, 3]

    # Gram-Schmidt to get t1 perpendicular to n
    t1 = ref - (ref * n).sum(dim=-1, keepdim=True) * n
    t1 = F.normalize(t1, dim=-1)  # [B, N, 3]

    # t2 = n × t1, already unit and perpendicular to both
    t2 = torch.linalg.cross(n, t1, dim=-1)  # [B, N, 3]

    # Reconstruct WSS vector from tangent coefficients
    alpha_t1 = raw[..., 1:2]  # [B, N, 1]
    alpha_t2 = raw[..., 2:3]  # [B, N, 1]
    tau_wss = alpha_t1 * t1 + alpha_t2 * t2  # [B, N, 3], globally-framed, n-orthogonal by construction

    # Concatenate cp + reconstructed WSS → [B, N, 4] in global xyz (same contract as before)
    surface_preds = torch.cat([raw[..., 0:1], tau_wss], dim=-1)
    surface_preds = surface_preds * surface_mask.unsqueeze(-1)
```

Add `import torch.nn.functional as F` at the top of `model.py` if not already present (it likely is).

#### 3. No other changes

No changes to `train.py`, `trainer_runtime.py`, `data/loader.py`, or any eval/metric code.

---

### Configuration / CLI command

Canonical Wave 30 config — no new flags needed:

```bash
python train.py \
  --dataset_path /mnt/pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --optimizer lion \
  --lr 9e-5 \
  --lion_beta1 0.9 \
  --lion_beta2 0.99 \
  --batch_size 4 \
  --epochs 13 \
  --n_hidden 512 \
  --n_heads 4 \
  --n_layers 5 \
  --n_slice 128 \
  --amp_mode bf16 \
  --grad_clip_norm 1.0 \
  --wandb_project senpai-v1-drivaerml-ddp8 \
  --wandb_group H17-tangent-frame \
  --use_ema \
  --ema_decay 0.999
```

**CRITICAL**: lr=9e-5 (NOT 5e-4 — this caused 4 Wave 30 divergences).

---

### Expected runtime

13 epochs, DDP-8 × A100 96 GB, batch 4 (same as Wave 30 baseline). Estimated 14–16 hours, well within the 18h budget. The tangent-frame reconstruction adds O(N) operations per forward pass — negligible versus attention cost.

---

### EP gate (kill criterion)

Check val_abupt at EP1 (after warmup). Expected range for a healthy run: ~30–35%. Kill if val_abupt > 48% at EP1 (same criterion used for H13b/H14 divergence detection).

At EP5 (mid-training), expect val_abupt ≤ 8.5% for a run that will beat the baseline at EP13. Kill if val_abupt > 12% at EP5.

---

### Success criteria

**Primary**: val_abupt < 5.7452% (ensemble gate) at best checkpoint.

**Mechanistic probe** (examine even if val_abupt does not beat baseline):
- τ_z/τ_x ratio at test time should drop from the collapse band [1.44–1.55] toward the GT ratio of 0.08. Any significant movement (e.g., below 1.3) confirms the mechanism is alive.
- direction_cos_loss should stay ≥ 0.9965 (should not regress from fern H10 EP8 level).

**What success looks like**: test_WSS < 6.3263% (ensemble SOTA), test_SP < 3.3529%, τ_z/τ_x ratio materially below 1.44.

**Falsifying result**: val_abupt matches baseline AND τ_z/τ_x ratio is unchanged from the collapse band. This would imply the implicit constraint enforcement does not address the over-prediction cause — pointing instead toward a normalisation or data-distribution mechanism.

---

### Implementation gotchas

1. **Degeneracy at near-vertical normals**: The reference vector fallback (e_x → e_y when |n·e_x| > 0.9) handles panels where n is nearly aligned with x. Do not use a single fixed reference across the whole mesh.

2. **Gradient flow through t1, t2**: t1 and t2 are computed from `surface_x[:, :, 3:6]` which is the INPUT feature tensor (not a parameter). The tangent basis is constant per sample during forward/backward — gradients flow only through (α_t1, α_t2), not through the basis vectors. This is correct behavior: the basis is a fixed geometric frame, not a learned frame.

3. **Normalized target matching**: The loss compares `surface_preds` to `surface_target = transform.apply_surface(batch.surface_y)`. The normalization (`TargetTransform`) is applied in global xyz space before the comparison. Since H17 outputs global-xyz τ_WSS, the normalized comparison is correct — no changes needed to `TargetTransform`.

4. **surface_output_dim attribute**: The attribute `self.surface_output_dim = SURFACE_Y_DIM = 4` is used elsewhere in the model (e.g., volume head initialization or checkpoint compatibility). Do NOT change this attribute; only change the final `nn.Linear` argument in `surface_out`.

5. **Checkpoint compatibility**: This is a new architecture; no pretrained checkpoint loading from prior runs. Start from scratch — same as all Wave 30 experiments.

6. **`torch.linalg.cross` signature**: In PyTorch ≥ 1.10, use `torch.linalg.cross(a, b, dim=-1)`. The `dim` argument is required to specify the 3D axis when tensors have batch dimensions.

---

### Research state update

**Current best explanation for τ_z bottleneck**: The global-frame output space forces the model to implicitly learn the τ·n=0 constraint. On horizontal automotive surfaces (hood, roof, trunk — dominant by area), the surface normal is approximately +z, creating a spurious correlation between the "normal direction" and the "τ_z channel" in the network's output space. The model learns to predict large τ_z on these surfaces, which are precisely the surfaces where τ_z should be near zero by physics.

**Evidence**: (a) τ_z/τ_x collapse band [1.44–1.55] is independent of loss reweighting, architecture width, multi-scale kNN aggregation, data augmentation, and EMA — all model-side attacks closed without movement. (b) Direction is learned correctly (fern H10 EP8 direction_cos_loss = 0.00355); the problem is magnitude in the output parameterization. (c) The over-predicted τ_z is most visible on automotive surface patches where n ≈ +z.

**Ruled out**: Loss-reweighting axis (4 variants closed), architecture widening (6 variants closed), data augmentation (mirror-aug H8 closed), EMA/Polyak smoothing (H15 in-flight, mechanism distinct), multi-scale aggregation (H11 in-flight, mechanism distinct).

**Open uncertainties**: (1) Whether the collapse is in the normalized prediction space or the raw prediction space — tangent-frame operates post-normalization so this matters for whether the mechanism fires. (2) Whether the tangent basis degeneracy on near-vertical panels (car body sides) is numerically stable with the proposed Gram-Schmidt fallback. (3) Whether the constraint enforcement interacts with per-channel z-score normalization in H16 (frieren, in-flight) — H17 is orthogonal to H16 and the two could be combined in a later wave if both succeed individually.

**Confidence**: Strong mechanistic grounding (external analogue: Gao et al. ICML 2024), zero implementation risk (all changes inside `forward()`), directly targets the observed failure mode. The only uncertainty is whether the geometric hypothesis (spurious n·e_z correlation) is the dominant contributor to the collapse band or a secondary effect.
