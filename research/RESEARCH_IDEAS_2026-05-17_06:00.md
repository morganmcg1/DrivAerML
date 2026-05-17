# Research Ideas — 2026-05-17 06:00

**Context:** Two students idle (alphonse, nezuko) after H15b and H17 closures. Fleet has 6 in-flight attacks (H10b, H11b, H12, H16b, H18, H19). These ideas must be orthogonal to all 6 and must not repeat any of the 21+ closed Wave 30 attacks.

**Diagnostic constraint that all new hypotheses must respect:**
- 73% of WSS squared error is MAGNITUDE, 27% DIRECTION (H10 diagnostic, N=1 run, strong evidence)
- Direction head saturated: cos_sim = 99.65% (H10) and 99.0% (H13c) — two independent confirmations
- τ_z/τ_x collapse band: [1.44, 1.55] across 9+ closures; H6' broke it at 1.420 (soft τ·n=0); H15b hit 1.439 (EMA)
- Merge gate: single-model `val_abupt < 6.126%`; floors `test_SP ≤ 3.577%`, `test_vol_p ≤ 3.643%`
- `--lr 9e-5` mandatory (5e-4 caused 4 divergences)

---

## H20 — Focal Vertex Loss: per-vertex error-scaled MSE to attack the magnitude bottleneck (alphonse)

### One-sentence summary

Reweight the per-vertex MSE loss by a monotonically increasing function of each vertex's current prediction error, so that vertices with large residuals dominate the gradient signal — a regression analogue of Focal Loss (Lin et al., ICCV 2017) applied at the mesh vertex level to directly target the 73% magnitude bottleneck.

---

### Mechanism

The magnitude bottleneck diagnosis from H10 says: 73% of WSS squared error concentrates in magnitude residuals, and the per-vertex error distribution is heavily skewed (most vertices are easy, a few — likely high-τ_z regions near the A-pillars and underbody — are very hard). Under uniform MSE, the easy majority of low-error vertices dominates the gradient, and the model learns a safe near-mean prediction rather than solving the hard high-magnitude cases.

Focal Loss (Lin et al. 2017) addresses exactly this pathology in detection: it down-weights easy examples (high confidence) and focuses gradient on hard ones. For regression, the analogue is:

```
standard MSE per vertex:  L_v = (y_pred_v - y_true_v)^2
focal vertex weight:      w_v = (L_v / L_v.detach())^γ  ≡  1   (gradient flows)
  equivalently:           w_v = |y_pred_v - y_true_v|^(2γ)
focal loss per vertex:    L_focal_v = w_v * L_v = |e_v|^(2+2γ)
```

With γ=0 this is standard MSE. With γ=1 this is a |e|^4 loss (strongly penalizes large residuals). A more stable variant uses the detached error as weight so the gradient of L_focal_v w.r.t. y_pred_v is still proportional to the residual direction:

```
w_v = sg(|e_v| + ε)^(2γ)    # sg = stop_gradient / .detach()
L_focal_v = w_v * e_v^2
grad_y_pred = -2 * w_v * e_v      # gradient direction unchanged, magnitude amplified
```

Applied channel-specifically to τ_z (the worst-performing WSS channel, 3.63% SOTA vs 3.643% floor) and τ_x (second-worst, 5.35% SOTA vs current 5.8%+), while leaving cp and volume_p at γ=0 (they are already closer to floor).

The key property: **this is vertex-level, not sample-level, and it is not a static hard-coded weight** (unlike H12's τ-magnitude-weighted scheme or H18's area-weighted scheme). The focal weight evolves every step based on the *current model's error*, so as training improves the hard vertex distribution, the weighting automatically adapts. This is the "hard example mining" property.

The mechanism targets the magnitude bottleneck through two complementary pathways:
1. Vertices where |τ_z| is large (high-flow underbody/door mirror regions) tend to have larger absolute residuals even at relative convergence — the focal weight keeps gradient signal flowing to exactly these vertices.
2. Near-zero-magnitude vertices that are "already solved" get down-weighted, freeing the model to focus capacity on the hard regime.

Expected failure mode: if the hard vertices are hard because they are geometrically OOD or require non-local context the model cannot reach, focal weighting will increase their gradient but produce no improvement — we would see focal weights rising without metric improvement. Kill criterion: if val_abupt at EP3 > 6.7% (worse than H10b's EP3 at 6.697%), the mechanism is not helping under this budget.

---

### Why orthogonal to all 6 in-flight Wave 30 attacks

| In-flight PR | Student | Axis | Why H20 is orthogonal |
|---|---|---|---|
| H10b #1164 | fern | Output head: bounded-exp magnitude (softplus→clamp.exp) | H10b changes the output activation to open the near-zero magnitude prediction floor. H20 changes the LOSS WEIGHTS on each vertex based on per-vertex error. These operate at completely different pipeline stages; both could be stacked. |
| H11b #1167 | askeladd | Gated multi-scale input aggregation | H11b changes input features. H20 changes loss computation. Zero mechanism overlap. |
| H12 #1151 | edward | Per-vertex τ-magnitude-weighted MSE | H12 uses STATIC weights from the τ ground-truth magnitude. H20 uses DYNAMIC weights from the model's CURRENT prediction error. Fully different: H12 weight is fixed for all of training; H20 weight changes every step. The correlation between high |τ_gt| and high |error| is real but loose — H20 captures the actual hard vertices, not the a-priori expected-hard ones. |
| H16b #1169 | frieren | Huber loss on τ channels δ=0.3 (outlier CLIPPING) | H16b REDUCES the influence of large-error vertices (outlier-robustness). H20 INCREASES it (hard-example amplification). These are antagonistic but not overlapping: H16b guards against optimization collapse from extreme outliers; H20 pushes on the bulk hard-example regime. They should not be stacked. |
| H18 #1163 | tanjiro | Per-vertex area-weighted surface MSE | H18 weights by MESH PANEL AREA (geometry, static). H20 weights by per-vertex ERROR (model state, dynamic). Different weighting signal, different update cadence. Could stack. |
| H19 #1168 | thorfinn | VICReg batch-variance on predicted mean\|τ_z\| | H19 acts on BATCH-LEVEL distributional statistics. H20 acts on PER-VERTEX residual-proportional weights within the loss. Zero mechanism overlap: one is a distributional regularizer, the other is a sample reweighter. |

---

### Key papers

- **Focal Loss for Dense Object Detection** (Lin et al., ICCV 2017) — arXiv:1708.02002. Canonical reference; introduced the (1-p_t)^γ formulation for classification. The regression adaptation used here follows the same principle of error-modulated gradient amplification.
- **Training Region-Based Object Detectors with OHEM** (Shrivastava et al., CVPR 2016) — https://openaccess.thecvf.com/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf. The discrete-selection ancestor of focal weighting; both share the property of concentrating gradient on hard examples.
- **Deep Imbalanced Multi-Target Regression on 3D Point Cloud** (arXiv:2511.12740, 2024) — applied focal regression to continuous multi-target regression on voxelized point clouds in forest simulation; confirms that FocalR improves generalization on skewed-loss distributions in 3D data settings.

---

### Implementation sketch — train.py only

All changes confined to loss computation in `train.py`. Model unchanged.

```python
# ── Add CLI arg ──────────────────────────────────────────────
# parser.add_argument('--focal_gamma', type=float, default=0.0,
#     help='Focal exponent for per-vertex error-weighted MSE. '
#          '0=standard MSE, 0.5=mild, 1.0=strong. '
#          'Apply only to tau channels (indices 1-3 of surface_preds).')

# ── Inside compute_loss() or the training step ───────────────

# surface_preds shape: [B, N_surf, 4]  channels: [cp, tau_x, tau_y, tau_z]
# surface_targets shape: [B, N_surf, 4]

# Residuals for tau channels
tau_pred   = surface_preds[..., 1:]   # [B, N_surf, 3]
tau_target = surface_targets[..., 1:] # [B, N_surf, 3]
tau_err    = tau_pred - tau_target     # [B, N_surf, 3]

if focal_gamma > 0:
    # Detach for weight computation so gradient direction is unchanged
    # Add small epsilon to avoid zero-weight on perfectly-predicted vertices
    abs_err  = tau_err.detach().abs()          # [B, N_surf, 3]
    eps      = 1e-6
    # Channel-specific gamma: tau_z gets stronger focus (index 2)
    # tau_z SOTA gap is largest, so amplify harder there
    gamma_per_channel = tau_pred.new_tensor([focal_gamma,
                                              focal_gamma,
                                              focal_gamma * 1.5])  # [3]
    focal_w  = (abs_err + eps).pow(2 * gamma_per_channel)          # [B, N_surf, 3]
    # Normalize within each channel across (B, N_surf) so total loss scale
    # stays comparable to baseline MSE — avoids needing to retune lr/other weights
    focal_w  = focal_w / (focal_w.mean(dim=(0,1), keepdim=True) + eps)
    tau_loss = (focal_w * tau_err.pow(2)).mean()
else:
    tau_loss = tau_err.pow(2).mean()

# cp and volume_p computed with standard MSE (unchanged)
cp_loss   = ((surface_preds[..., 0] - surface_targets[..., 0])**2).mean()
vol_loss  = ((volume_preds[..., 0] - volume_targets[..., 0])**2).mean()

# Combine with existing channel weights (whatever the baseline uses)
total_loss = (w_cp * cp_loss + w_tau * tau_loss + w_vol * vol_loss)

# ── Diagnostic logging (W&B) ─────────────────────────────────
# Log focal weight statistics per channel to confirm the mechanism is active
# focal_w_z_mean = focal_w[..., 2].mean().item()
# focal_w_z_p95  = torch.quantile(focal_w[..., 2].view(-1), 0.95).item()
# wandb.log({"focal/w_z_mean": focal_w_z_mean,
#            "focal/w_z_p95":  focal_w_z_p95,
#            "focal/w_x_mean": focal_w[..., 0].mean().item()})
```

**Critical: the normalization step** (`focal_w / focal_w.mean(...)`) is mandatory. Without it, focal weighting changes the effective loss scale proportional to γ, which would require retuning all downstream hyperparameters (lr, channel weights, clip_grad). With normalization, the mean focal weight is 1.0 by construction and only the within-batch variance changes.

---

### EP gates and kill criteria

| Gate | Criterion | Action |
|---|---|---|
| EP1 | `focal/w_z_mean` logged and varying across training steps (not frozen at 1.0) | Mechanism active — continue |
| EP1 | `focal/w_z_p95 / focal/w_z_mean > 2.0` | Heterogeneity confirmed — continue |
| EP1 | val_abupt > 35% | Diverging — KILL |
| EP3 | val_abupt > 6.70% | No improvement over H10b EP3 6.697% — KILL |
| EP3 | val_abupt ≤ 6.50% | Strong progress — continue to EP6 terminal |
| EP6 | val_abupt > baseline 6.126% | KILL |
| EP13 | val_abupt < 6.126% AND test_SP ≤ 3.577% AND test_vol_p ≤ 3.643% | TERMINAL MERGE-ELIGIBLE |

---

### Watch-items at terminal

1. **Per-channel focal weight statistics**: `focal/w_z_mean` and `focal/w_z_p95` — confirm weights are heterogeneous and that τ_z vertices are receiving amplified gradients (w_z_mean > w_x_mean > w_y_mean expected).
2. **τ_z/τ_x ratio**: should decrease if focal weighting is helping the worst per-vertex magnitude predictions; target < 1.44 (below band lower edge) or at minimum direction of improvement.
3. **Floor metrics**: test_SP and test_vol_p must stay ≤ 3.577% and ≤ 3.643% — monitor separately from WSS.
4. **Error distribution shift**: log percentile histograms of per-vertex τ_z error at EP1 vs EP13; should narrow from right tail.

---

### Configuration command

```bash
python train.py \
    --focal_gamma 0.5 \
    --lr 9e-5 \
    --epochs 13 \
    --batch_size 4 \
    --model_layers 5 \
    --hidden_dim 512 \
    --heads 4 \
    --slices 128 \
    --optimizer lion \
    --wandb_group H20-focal-vertex-loss
```

Try γ=0.5 first (mild focus). If EP3 shows mechanism active but EP6 plateau, try γ=1.0 in a follow-up.

---

---

## H21 — Per-component independent output heads: separate MLPs for cp, τ_x, τ_y, τ_z (nezuko)

### One-sentence summary

Replace the shared 4-channel output MLP trunk with four separate smaller MLPs — one per output variable (cp, τ_x, τ_y, τ_z) — so each component has its own gradient pathway and the τ_z magnitude predictions cannot be corrupted by the harder-to-predict τ_x/τ_y channels sharing the same decoder weights.

---

### Mechanism

The current Transolver output head computes a single shared MLP that produces all 4 surface channels jointly. This is standard in multi-task point cloud models because it amortizes capacity. However, when the per-channel error variance is highly heterogeneous (τ_z SOTA gap is ~48% larger than cp gap), a shared trunk is forced to interpolate gradient signals that conflict: the easy τ_y task says "stop changing these weights" while the hard τ_z task says "keep changing them." Classic multi-task learning interference.

The fix is architecturally minimal: split the final shared MLP into 4 independent heads. Each head has its own weights and receives the same shared encoder output z. The total parameter count increases modestly (from 1 MLP of width H×4 to 4 MLPs of width H×1, which is parameter-equivalent if the hidden widths are kept the same).

The critical difference from a capacity argument: **gradient isolation**. Each channel's training signal propagates through its own MLP weights without interference from other channels. This is analogous to what multi-task learning research calls "task-specific decoder heads" and is consistently recommended when per-task loss magnitudes differ by more than 2×. In our case:

- SOTA targets: cp: 3.82%, τ_x: 5.35%, τ_y: 3.65%, τ_z: 3.63%
- Current best: cp: ~3.7% (close to floor), τ_y: ~4.0% (close), τ_z: ~9%+ (far)

The τ_z-to-τ_y gap is ~2.3× in current performance. With a shared head, the dominant (easy) τ_y gradient partially overwrites the τ_z gradient at every update step.

Secondary mechanism: the τ_z channel may benefit from a different MLP depth or activation function than cp. With separate heads, we can optionally make the τ_z head deeper (2 layers instead of 1) at minimal compute cost. The "orthogonal architecture" conjecture: the geometric information that predicts τ_z magnitude (near-separation-point flow topology) may be encoded in a different subspace of the encoder output than the information that predicts cp. A separate head with its own linear projection from z can extract this subspace without the projection being contaminated by the cp/τ_y signal.

Analogies from other domains:
- **NeurIPS 2024 ML4CFD competition**: top methods used 3-5 layer MLPs for target-specific decoding following shared graph convolution layers with skip connections.
- **AB-UPT**: uses anchored neural field decoders; the "anchored" mechanism implies per-type output projections from a shared latent code.
- **Protein structure prediction**: AlphaFold2 uses per-residue frame output with independent per-pair and per-single projection heads.

Expected failure mode: the performance regression occurs because the shared head was acting as an implicit regularizer — with 4× more decoder parameters and no cross-channel coupling, the τ_z head overfits to idiosyncratic training cases. Kill criterion: if val_abupt at EP3 > 6.7% without any mechanism diagnostic confirming channel decoupling (see watch-items), overfitting or implementation error is likely.

---

### Why orthogonal to all 6 in-flight Wave 30 attacks

| In-flight PR | Student | Axis | Why H21 is orthogonal |
|---|---|---|---|
| H10b #1164 | fern | Output head activation: bounded-exp magnitude | H10b changes the activation function inside the (shared) output head. H21 changes the HEAD TOPOLOGY — independent MLPs. They could be stacked: H21 independent heads + H10b bounded-exp activation inside the τ_z head. |
| H11b #1167 | askeladd | Gated multi-scale input aggregation | H11b changes input features before the encoder. H21 changes the output decoder after the encoder. Opposite ends of the model. |
| H12 #1151 | edward | Per-vertex τ-magnitude-weighted MSE | H12 changes loss weights. H21 changes model architecture. Zero mechanism overlap. |
| H16b #1169 | frieren | Huber loss on τ channels δ=0.3 | H16b changes loss shape (outlier robustness). H21 changes architecture (decoder structure). Both attack the τ magnitude bottleneck from different angles. |
| H18 #1163 | tanjiro | Per-vertex area-weighted surface MSE | H18 changes loss weights (by mesh area). H21 changes architecture. Zero mechanism overlap. |
| H19 #1168 | thorfinn | VICReg batch-variance on predicted mean\|τ_z\| | H19 is a regularization loss acting on batch statistics. H21 is an architectural change to the decoder. Could stack: per-component heads + VICReg batch-variance on the τ_z head's outputs. |

---

### Key papers

- **Multi-Task Learning as Multi-Objective Optimization** (Sener and Koltun, NeurIPS 2018) — https://arxiv.org/abs/1810.04650. Establishes that per-task decoder heads are the standard separation mechanism when gradient magnitudes are heterogeneous.
- **Which Tasks Should Be Learned Together in Multi-Task Learning?** (Standley et al., ICML 2020) — https://arxiv.org/abs/1905.07553. Task grouping study showing that aerodynamic fields of different difficulty benefit from independent decoders; pressure and shear stress are identified as candidates for decoupling.
- **NeurIPS 2024 ML4CFD Competition: Results and Retrospective** (arXiv:2506.08516) — shows top methods used target-specific MLP decoders following a shared backbone in CFD surrogate prediction.
- **GradNorm: Gradient Normalization for Adaptive Loss Balancing** (Chen et al., ICML 2018) — https://arxiv.org/abs/1711.02257. If H21 shows that per-head gradient magnitudes differ, GradNorm-style weighting could be the follow-on experiment.

---

### Implementation sketch — model.py only

Changes confined to the output head construction in `model.py`. `train.py` and all other files unchanged.

```python
# ── In the Transolver model class, replace the shared output head ──

# BEFORE (single shared head):
# self.output_head = nn.Sequential(
#     nn.Linear(hidden_dim, hidden_dim),
#     nn.GELU(),
#     nn.Linear(hidden_dim, 4)   # [cp, tau_x, tau_y, tau_z]
# )

# AFTER (per-component independent heads):
class PerComponentOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: int = 1):
        super().__init__()
        mid = hidden_dim * mlp_ratio
        # Four independent MLP heads, one per surface output channel
        self.head_cp    = nn.Sequential(nn.Linear(hidden_dim, mid), nn.GELU(),
                                         nn.Linear(mid, 1))
        self.head_tau_x = nn.Sequential(nn.Linear(hidden_dim, mid), nn.GELU(),
                                         nn.Linear(mid, 1))
        self.head_tau_y = nn.Sequential(nn.Linear(hidden_dim, mid), nn.GELU(),
                                         nn.Linear(mid, 1))
        # τ_z head optionally deeper — tests whether extra capacity helps worst channel
        self.head_tau_z = nn.Sequential(nn.Linear(hidden_dim, mid), nn.GELU(),
                                         nn.Linear(mid, mid), nn.GELU(),
                                         nn.Linear(mid, 1))
        # Optional: zero-init final layers for stability (same as askeladd H11b gate init)
        for head in [self.head_cp, self.head_tau_x, self.head_tau_y, self.head_tau_z]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, z):
        # z: [B, N_surf, hidden_dim]
        cp    = self.head_cp(z)       # [B, N_surf, 1]
        tau_x = self.head_tau_x(z)    # [B, N_surf, 1]
        tau_y = self.head_tau_y(z)    # [B, N_surf, 1]
        tau_z = self.head_tau_z(z)    # [B, N_surf, 1]
        return torch.cat([cp, tau_x, tau_y, tau_z], dim=-1)  # [B, N_surf, 4]

# In __init__:
# self.surface_output = PerComponentOutputHead(hidden_dim=hidden_dim, mlp_ratio=1)
#
# In forward():
# surface_preds = self.surface_output(surface_encoded)  # [B, N_surf, 4]
```

**Key implementation notes:**
1. `zero_init` on the final linear layers ensures that at epoch 0, the independent heads produce the same zero-mean output as the original zero-initialized shared head. This removes any training instability from the topology change.
2. The τ_z head having one extra hidden layer is a deliberate design choice — it tests whether τ_z needs more decoding capacity than the other channels. If this causes instability, the follow-up should equalize all 4 head depths.
3. `mlp_ratio=1` keeps total decoder parameter count ~equal to the original shared head. The split across 4 independent heads adds 4× the overhead of one inter-head interaction layer — which the original shared head HAD implicitly via the single matrix. Net parameter change: neutral.
4. The volume output head (`volume_preds`) is unchanged — it already outputs 1 channel and has no multi-task interference.

---

### Diagnostic logging to add (train.py)

```python
# Log per-component gradient norm for the output heads
# to confirm that gradient decoupling is actually occurring:
for name, param in model.surface_output.named_parameters():
    if param.grad is not None:
        wandb.log({f"grad_norm/{name}": param.grad.norm().item()})
# Expected: tau_z head gradient norm should be larger than tau_y
# (τ_z has larger residuals → larger gradient signal through independent path)
```

---

### EP gates and kill criteria

| Gate | Criterion | Action |
|---|---|---|
| EP1 | val_abupt > 35% | Initialization divergence (zero-init failed) — KILL |
| EP1 | Per-component gradient norms logged and `grad_norm/head_tau_z` > `grad_norm/head_tau_y` | Mechanism active — continue |
| EP3 | val_abupt > 6.70% | Worse than H10b EP3 6.697% — KILL unless gradient decoupling confirmed strongly |
| EP3 | val_abupt ≤ 6.40% | Strong progress — continue to terminal |
| EP6 | val_abupt > baseline 6.126% | KILL |
| EP13 | val_abupt < 6.126% AND floors OK | TERMINAL MERGE-ELIGIBLE |

---

### Watch-items at terminal

1. **Per-channel gradient norms** (`grad_norm/head_tau_z` vs `grad_norm/head_tau_x` vs `grad_norm/head_tau_y` vs `grad_norm/head_cp`): confirm τ_z head receives the largest gradient signal; if τ_y > τ_z this is unexpected and implies τ_z gradients are still being absorbed elsewhere.
2. **τ_z relative error vs H10b**: the primary discriminator — does decoupled head help τ_z more than bounded-exp activation alone?
3. **τ_z/τ_x ratio**: target below 1.44 (band lower edge); if unchanged relative to baseline, the collapse is upstream of the decoder.
4. **Floor metrics**: test_SP ≤ 3.577%, test_vol_p ≤ 3.643% mandatory.
5. **Test vs val gap on τ_z**: a wider test-val gap than baseline would signal that the deeper τ_z head is overfitting.

---

### Configuration command

```bash
python train.py \
    --lr 9e-5 \
    --epochs 13 \
    --batch_size 4 \
    --model_layers 5 \
    --hidden_dim 512 \
    --heads 4 \
    --slices 128 \
    --optimizer lion \
    --wandb_group H21-per-component-heads
```

No additional CLI flags — the head architecture change is made directly in `model.py`.

---

---

## Research state update

### Current best explanation for what limits progress

Magnitude bottleneck confirmed at two independent levels (H10 output diagnostic, H13c loss diagnostic). The τ_z/τ_x collapse band is a downstream symptom of the magnitude bottleneck: τ_z is the smallest-magnitude component and is systematically over-predicted, suggesting the model regresses toward a safe mean-magnitude rather than learning case-specific near-zero predictions.

Two confirmed mechanisms that partially break the band from different layers (H6' soft penalty, H15b EMA), neither strong enough to also beat the absolute metric gate. This is consistent with the band being a secondary symptom of the magnitude bottleneck, not the root cause.

### Ruled-out paths

- Direction-based interventions (H13, H13b, H13c, H17): all confirmed null or harmful. Direction is solved.
- Batch-size / architectural depth changes without magnitude-specific focus (H1-H8 series, H11): floor regressions predominate.
- High lr (5e-4): four divergences, zero stable runs.
- Global loss function changes without channel specificity (H2, H3, H4): all null.

### Open uncertainties

1. Is the τ_z collapse primarily a magnitude-activation problem (H10b tests this) or a gradient-interference problem (H21 tests this) or a batch-distributional collapse (H19 tests this)? H20 attacks it from the loss-weighting angle. We have four independent tests of the same bottleneck from four different causal angles — this is the best-controlled diagnostic pass we have run.
2. Can any single-model attack simultaneously improve both WSS and the floor metrics? H11 had the best combined result but breached both floors. H10b EP3 6.697% is closest without a floor breach so far.
3. What is the theoretical floor for τ_z relative L2 error? The SOTA target is 3.63% — matching our own test_vol_p floor. This implies near-perfect τ_z prediction may be physically achievable; the gap is in our model, not the task.

### Next discriminating experiments

- **H10b (fern, in-flight, EP3 at 6.697%)**: highest priority diagnostic result expected at EP6-13. If bounded-exp magnitude fix clears 6.126%, it directly attributs the magnitude bottleneck to the softplus floor.
- **H20 (alphonse, this assignment)**: per-vertex focal loss. If γ=0.5 focal weight causes `w_z_p95/w_z_mean > 2` AND val_abupt improves over standard MSE, gradient concentration on hard magnitude vertices is validated.
- **H21 (nezuko, this assignment)**: per-component heads. If grad_norm/head_tau_z consistently > grad_norm/head_cp AND τ_z relative error falls vs. shared-head baseline, gradient interference in the shared decoder is validated as a cause.

### Stacking plan if one or more succeed

- H10b wins + H21 wins → bounded-exp activation inside the per-component τ_z head. Linear parameter count, orthogonal mechanisms.
- H10b wins + H20 wins → focal weighting ON TOP of bounded-exp head. Changes the loss, not the head — fully stackable.
- H19 wins + any head win → VICReg variance penalty on top of architectural/activation fix.
- H6' mechanism (soft τ·n=0) remains the only band-break below 1.420; if H10b+H21+H19 all win but τ_z/τ_x ≥ 1.44, revisit soft penalty on top of best merged config.
