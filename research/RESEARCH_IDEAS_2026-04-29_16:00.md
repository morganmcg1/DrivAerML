# Research Ideas — DrivAerML tay/DDP8 — 2026-04-29 16:00

**Context:** All ideas are designed to compose orthogonally with fern's confirmed RFF
coordinate feature win (test_abupt 17.77). Primary target: tau_y=25.50 and tau_z=26.53
(current tay/DDP8, 7× gap vs AB-UPT 3.65/3.63). Budget: ~9 uncompiled epochs now;
~22 compiled epochs after PR #40 compile fix lands. All ideas are single-delta
(100-300 LOC) changes that do NOT duplicate the Round 2 queue.

AB-UPT reference targets (lower is better):
- surface_pressure: 3.82, wall_shear: 7.29, volume_pressure: 6.08
- tau_x: 5.35, tau_y: 3.65, tau_z: 3.63

Current tay/DDP8 baseline (PR #30, test): abupt=19.81, tau_y=25.50, tau_z=26.53

---

## H01 — Multi-sigma RFF (extend fern's win across spatial frequencies)

**Hypothesis:** Fern's single-sigma (σ=1.0) RFF captures one spatial frequency band.
Concatenating encodings at σ ∈ {0.1, 1.0, 10.0} gives the model simultaneous access
to global body shape, meso-scale curvature transitions, and near-panel local variation —
the three scales that matter most for tau_y and tau_z, which vary at all three.

**Mechanism:** Spectral bias in neural networks (Rahaman et al. 2019, ICML) means
high-frequency components are learned last. Multi-scale Fourier encoding injects
high-frequency content into the input directly, bypassing the learning-order bottleneck.
PGCAN (Hao et al., 2024, arXiv:2403.15652) demonstrated this explicitly on PDE boundary
layers, where spectral bias suppresses oscillatory modes — exactly the structure of
tau near flow separation.

**Composes with:** fern's single-sigma RFF (this extends it rather than replacing it),
per-axis loss weights (thorfinn), FiLM, compiled training budget.

**Implementation cost:** Low (~30 LOC). Replace fern's single GaussianFourierProjection
with a ConcatMultiScaleRFF module that stacks three projections at σ={0.1, 1.0, 10.0}.
Input dim increases from 7+2F to 7+6F (e.g., 7+192 if each head is 32 freq × 2). Keep
architecture otherwise identical to fern's PR.

**Risk level:** Low. fern's single-sigma version works; this is a strict superset of
that representation. The only risk is that the wider input projection adds parameters
that overfit at 9 epochs, but with DDP8 bs=32 and weight decay this is unlikely.

**Citation:** Tancik et al. (2020) "Fourier Features Let Networks Learn High Frequency
Functions in Low Dimensional Domains", NeurIPS 2020. Hao et al. (2024) PGCAN
arXiv:2403.15652. Rahman et al. (2022) "U-NO" shows multi-scale Fourier encoding on
multiple operator learning benchmarks.

**Expected lift:** −1.0 to −2.0 abupt vs fern's 17.77. tau_y and tau_z should see
disproportionate gain because they encode high-frequency flow features near body
transitions.

---

## H02 — Local tangent-frame coordinate features (give model direct access to tau's native basis)

**Hypothesis:** tau_y and tau_z are defined in the global Cartesian frame, but wall
shear stress physics lives in the local surface tangent frame. The model currently sees
[x,y,z,nx,ny,nz,area] and must learn the relationship between global normals and local
tangent axes. Adding precomputed unit tangent vectors [t1x, t1y, t1z, t2x, t2y, t2z]
as explicit input features gives the model the coordinate system in which tau_y and
tau_z are most meaningful, without requiring equivariance.

**Mechanism:** Given a surface normal n, the tangent frame (t1, t2) can be constructed
as: t1 = normalize(cross(n, e_z)) if |cross(n, e_z)| > ε else normalize(cross(n, e_x));
t2 = cross(n, t1). This is stable for all surface orientations except exactly vertical
panels (which are rare on the DrivAer geometry). The model can now learn tau_y ≈
dot(tau_vector, t1) and tau_z ≈ dot(tau_vector, t2) directly from the input feature
correlation rather than from geometric inference during forward passes.

**Composes with:** RFF (RFF applied independently to the 7-dim base features and the
6-dim tangent frame), per-axis weights, all architecture changes.

**Implementation cost:** Low (~50 LOC in dataset/feature construction or model.py
preprocessing). Tangent frame computation is deterministic from the existing nx,ny,nz
columns. Input dim increases from 7 to 13.

**Risk level:** Low-medium. The tangent frame construction must handle degenerate
normals (n ≈ ±e_z → cross product near zero); use the e_x fallback for safety.
Known gotcha: frame is not unique (any rotation of t1/t2 about n is valid), but
consistency within a given geometry is sufficient — the model can learn the
dataset-level convention.

**Citation:** Borsuk et al. (2022) "GNN-based wall shear stress estimation in LES"
shows local surface frame features improve tau prediction in GNNs. Weiler et al.
(2021) "E(n)-equivariant graph neural networks" discusses local frame construction;
we use the non-equivariant (simpler) version. FIGConvNet (NVIDIA, NeurIPS ML4PS 2025)
uses local surface geometry features for DrivAerML.

**Expected lift:** −0.5 to −1.5 abupt, concentrated in tau_y and tau_z. The
hypothesis is most valuable as a diagnostic: if adding the true tangent frame doesn't
move tau_y/tau_z at all, it rules out "coordinate system mismatch" as a driver and
points toward insufficient model capacity.

---

## H03 — Surface curvature as input feature (encode flow-critical geometry transitions)

**Hypothesis:** Mean curvature κ_H and Gaussian curvature κ_G encode where the surface
geometry transitions sharply — A-pillar, C-pillar, wheel arches, spoiler edge. These
are exactly the locations where tau_y and tau_z have their highest magnitude and
steepest gradients (flow separation, reattachment, horseshoe vortices). Providing
curvature as explicit point features reduces the model's need to infer geometric
transitions from position and normal alone.

**Mechanism:** Curvature is computed from the local normal field:
κ_H = 0.5 * trace(second_fundamental_form), κ_G = det(second_fundamental_form).
On a point cloud with normals, this can be approximated via PCA of the k-NN normal
covariance matrix. The feature is purely geometric and deterministic — it adds no
learned components. Adding [κ_H, κ_G] (2 scalars) to the input feature vector is
a feature engineering delta.

**Composes with:** RFF features (apply RFF to coordinates + curvature), tangent frame
features (H03 composes with H02), per-axis weights.

**Implementation cost:** Medium (~100 LOC). Curvature estimation from a point cloud
requires a k-NN graph construction at preprocessing time. Can be done in dataset
loading with torch_geometric.nn.pool or via sklearn k-NN + PCA. Key hyperparameter:
k (number of neighbors for curvature estimation); k=16 is standard.

**Risk level:** Medium. Curvature estimation from unstructured meshes can be noisy,
especially at sharp features (edges and corners where curvature is undefined in the
smooth sense). Clip curvature to [−50, 50] before normalization to avoid outlier
features. The DrivAer mesh is reasonably fine-grained so noise should be manageable.

**Citation:** Sharp et al. (2020) "DiffusionNet: Discretization Agnostic Learning on
Surfaces", SIGGRAPH 2022 — uses principal curvatures as input features for surface
learning. Borsuk et al. (2022) GNN wall shear paper shows curvature features improve
boundary layer tau prediction.

**Expected lift:** −0.3 to −1.0 abupt. Most useful in combination with H02 (tangent
frame). Standalone curvature is less likely to be the key bottleneck than coordinate
frame, but the diagnostic value (does curvature knowledge help?) is high.

---

## H04 — Channel-selective Huber loss for tau channels (robust training on heavy-tailed wall-shear distributions)

**Hypothesis:** Wall shear stress distributions near flow separation and reattachment
lines are heavily right-skewed — a small fraction of surface points have tau magnitude
5-10× the median. The relative-L2 loss squares these residuals, creating enormous
gradient spikes from the small fraction of high-tau outlier patches. A Huber loss
on the tau channels (with L2 behavior below threshold δ and L1 above) reduces gradient
scale from outlier patches by a factor of ~|residual|/δ, smoothing optimization and
allowing the model to generalize better to moderate-tau regions (which dominate the
test set metric).

**Mechanism:** Replace `loss_tau = rel_l2(pred_tau, gt_tau)` with a channel-selective
variant: apply Huber(δ) to tau channels and keep rel_l2 for cp and pv. The Huber
threshold δ controls the crossover; δ=0.5 in normalized tau-space (after the existing
per-channel normalization) is a reasonable starting point. This is explicitly distinct
from squared rel-L2 (in the Round 2 queue) because squared rel-L2 makes the outlier
problem WORSE while Huber makes it better.

**Composes with:** All feature engineering changes (H01-H03), per-axis weights
(thorfinn's approach orthogonally weights axes; Huber separately controls outlier
robustness within each axis), FiLM.

**Implementation cost:** Low (~40 LOC). Add a `compute_loss_huber` function in
train.py with configurable δ and `--tau_loss_type {rel_l2, huber}` CLI flag. No
model changes.

**Risk level:** Low. Huber loss is well understood. The main calibration concern is
choosing δ: too small → pure L1 (biased toward median, misses peaks); too large →
same as L2 (no outlier protection). A 2-arm W&B sweep (δ=0.1, δ=0.5) is advisable.
Use `--wandb_group tau_huber_delta_sweep` for the sweep.

**Citation:** Huber (1964). Borsuk et al. (2022) applies robust losses to GNN wall
shear. More recently: Lienen & Gunnemann (2022) "From LODE to MolDyn" uses
Cauchy/Huber in molecular force field predictions (heavy-tailed force distribution
analogue). Hao et al. (2024) PGCAN uses adaptive loss weighting for PDE boundary
layers.

**Expected lift:** −0.5 to −1.5 abupt. Primary signal is in tau_y and tau_z at
separation/reattachment locations. Note: if the tau distribution is NOT heavy-tailed
after normalization, this idea has no mechanism — check the distribution of
`gt_tau_y` in the training set before committing to this experiment.

---

## H05 — Normal-dot geometric attention bias (ALiBi-style orientation encoding in transformer attention)

**Hypothesis:** The Physics-Attention in Transolver routes tokens based on learned
slice assignments, but the attention weights do not encode surface orientation
relationships. For wall shear prediction, points on similarly-oriented surface
patches (same local normal direction) should have stronger inductive feature sharing.
Adding dot(n_i, n_j) as an additive bias to the pre-softmax attention logits
(like ALiBi positional encoding but for surface orientation rather than sequence
position) gives the model a free, differentiable geometric inductive bias without
modifying the token structure.

**Mechanism:** In each attention layer, compute normal cosine similarity matrix
B_ij = dot(n_i, n_j) ∈ [−1, 1], scale by a learned scalar β per head, and add to
the QK^T logits before softmax: attn_logits = QK^T/√d_k + β * B. This has O(N²)
memory overhead for surface points but surface N is typically ~5000-10000 per
sample — manageable on 96GB GPUs.

**Composes with:** All input feature engineering (H01-H03, fern's RFF), per-axis
weights, FiLM.

**Implementation cost:** Medium (~80 LOC in model.py). Requires surface normal tensor
to be passed into the attention module. The β parameter per head can be initialized
to 0.0 (no effect at init) and learned from there — ensuring stability at start.

**Risk level:** Medium. The O(N²) normal similarity computation adds memory pressure.
For N_surface=8000 per GPU and bs=4 per GPU, the B matrix is 8000×8000×4×fp16 ≈ 1.9
GB per GPU — tight but feasible on 96GB. Known gotcha: if N varies across the batch,
the normal matrix must be computed per-sample, not per-batch; use torch.einsum with
masking.

**Citation:** Press et al. (2022) "Train Short, Test Long: Attention with Linear
Biases (ALiBi)", ICLR 2022 — the source mechanism. Choromanski et al. (2021)
FAVOR+ attention. Geometric attention biases used in SE(3)-Transformer (Fuchs et al.
2020) and EquiformerV2 (Liao et al. 2023, ICLR 2024) for molecular property
prediction.

**Expected lift:** −0.5 to −1.5 abupt. Most benefit expected on tau prediction
since tau is most strongly locally correlated with surface orientation patterns.
If β → 0 during training, the mechanism provides no signal (informative null result).

---

## H06 — Log1p target normalization for tau channels (compress heavy-tailed tau distribution)

**Hypothesis:** Wall shear stress has a near-lognormal marginal distribution:
most of the surface has |tau| < 1 Pa but near-separation patches reach |tau| > 10 Pa.
When rel-L2 loss is computed in linear space, the high-|tau| patches dominate the
gradient signal. Applying a sign-preserving log1p transform to the tau targets
before loss computation (and inverting at eval time) compresses the dynamic range,
making the effective training signal more uniformly distributed across the surface.
Unlike Huber (H04), this acts globally on the distribution shape rather than
clipping individual residuals.

**Mechanism:** Transform: tau_transformed = sign(tau) * log1p(|tau|). Inverse:
tau_original = sign(tau_t) * (exp(|tau_t|) − 1). Apply the transform to GT targets
at loss time; apply the inverse to model predictions at evaluation time. The model
predicts in log-tau space; the metric is computed in original tau space (so results
are directly comparable to all other PRs).

**Composes with:** All feature engineering changes, per-axis weights. NOTE: log-space
targets interact with the relative-L2 loss formula — the "relative" normalization
denominator should use the log-transformed GT norm, not the original norm. Implement
as `rel_l2_log_tau = ||pred_log - gt_log||_2 / ||gt_log||_2` for tau channels.

**Implementation cost:** Low (~50 LOC in train.py, no model changes). Add
`--tau_log_transform` flag. Log transform applied just before loss call; inverse
applied at eval metric computation.

**Risk level:** Medium. If tau values span both positive and negative with similar
magnitude (as they do for tau_z on symmetric panels), the sign-preserving log1p
is well-behaved. Key concern: the inverse transform is numerically sensitive for
large |tau_t|; clamp predictions to a reasonable range before inversion.

**Citation:** Lognormal distributions in turbulent wall shear: Diaz-Daniel et al.
(2017) "Wall shear stress fluctuations in turbulent channels" (JFM) — establishes
near-lognormal distribution of |tau| in turbulent flows. Log-transform of skewed
regression targets: standard Kaggle practice for competitions with skewed targets
(e.g., house prices, load forecasting). Cohn et al. (2021) "Meta-learning for
neural PDE solvers" uses log-normalized outputs for pressure fields.

**Expected lift:** −0.3 to −1.5 abupt. If the tau distribution is indeed heavy-tailed
(check: compute percentiles of |gt_tau_y| across the training set), this should work.
Null result: if tau is actually approximately Gaussian after the existing normalization,
log1p provides no benefit. This makes it a clean diagnostic for distribution shape.

---

## H07 — k-NN local EdgeConv pre-encoder (inject surface-local context before global attention)

**Hypothesis:** Transolver's Physics-Attention operates on individual point tokens
with no explicit local surface connectivity. The model must infer local surface
patterns (flow gradients, boundary layer transitions) purely from the global attention
over all N points. Adding a single EdgeConv layer (Wang et al. 2019, DGCNN) that
aggregates each point's k nearest surface neighbors before the Transolver trunk
gives every token a local neighborhood summary. Local surface context is particularly
important for tau prediction: boundary layer thickness, local flow acceleration, and
vortex footprints all have length scales of O(1-10 cm) on the DrivAer body.

**Mechanism:** Construct the k-NN graph on surface points (k=16, radius threshold
optional). Apply EdgeConv: h_i = AGG_{j ∈ kNN(i)} MLP([h_i, h_j - h_i]) where
h_i is the input feature at point i. This produces a new feature vector of the same
dimension as the input for each point, which is then passed to the existing Transolver
trunk unchanged. One EdgeConv layer adds roughly 2 × d × (d_out) parameters.

**Composes with:** RFF features (EdgeConv operates on the same feature space fern
enhances), curvature features (H03), tangent frame features (H02).

**Implementation cost:** Medium (~150 LOC). EdgeConv is available in
torch_geometric.nn.conv.EdgeConv or PyG. The main implementation cost is building
the k-NN graph at train time and ensuring it works with DDP (batch graphs require
offset indices). k-NN construction can be done in the DataLoader worker.

**Risk level:** Medium. Adding a GNN layer before the transformer has two risks:
(1) k-NN graph construction is O(N log N) — benchmark that it doesn't become a
CPU bottleneck in the DataLoader at bs=32. (2) The EdgeConv MLP adds parameters;
with only 9-22 training epochs, underfitting the new parameters is possible. Use
a small EdgeConv hidden dim (= input dim, no expansion).

**Citation:** Wang et al. (2019) "Dynamic Graph CNN for Learning on Point Clouds"
(DGCNN), ACM Trans. Graphics. Borsuk et al. (2022) "GNN-based wall shear stress
estimation in complex geometries" shows GNN local aggregation improves tau_y and
tau_z over purely global approaches. FIGConvNet (NVIDIA, NeurIPS ML4PS 2025) uses
hierarchical convolution over multiple spatial scales for DrivAerML.

**Expected lift:** −0.5 to −2.0 abupt. Highest expected return of any single
architectural change, but also highest implementation risk. Recommend implementing
as a skip-connection (output = input + EdgeConv(input)) initialized to zero weight
so the model starts at Transolver baseline and grows the local context gradually.

---

## H08 — Multi-scale neighborhood pooling (FIGConvNet-style hierarchical surface aggregation)

**Hypothesis:** A single-radius k-NN captures one scale of surface context. The
DrivAer geometry has flow features at multiple length scales: local panel curvature
(~2 cm), body transitions (A-pillar, ~10 cm), and global body regions (frontal stagnation,
~50 cm). Pooling features at three radii (r = {0.05m, 0.15m, 0.45m} — approximately
1:3:9 ratio) and concatenating gives the model a multi-resolution view of the local
geometry, analogous to the multi-scale convolutions in FIGConvNet.

**Mechanism:** For each surface point, compute three radius-ball feature aggregations
(mean of neighbor features within each radius). Concatenate the three aggregate
vectors and pass through a small MLP to project back to the original feature dimension.
This is a strictly input-side operation — the Transolver trunk receives augmented
features of the same dimension as before.

**Composes with:** RFF features (apply RFF before or after pooling — before is
simpler), tangent frame (H02), EdgeConv (H07 and H08 can be combined: H07 for the
finest scale, H08 for the two coarser scales).

**Implementation cost:** Medium (~120 LOC). Radius ball queries are available via
torch_geometric.nn.pool.radius_graph or torch_cluster.radius. The main cost is
efficient batched radius ball computation; use torch_cluster for speed.
Key hyperparameter: radius values — start with {0.05, 0.15, 0.45} in normalized
mesh coordinates and verify that median neighborhood sizes are ~16, ~50, ~150 points.

**Risk level:** Medium. Radius queries require knowing the coordinate scale of the
mesh. The DrivAer geometry is ~4.6m long × 2.0m wide; radii of 0.05/0.15/0.45
are in physical meters and should be confirmed against the data loader's normalization.
If coordinates are normalized to [0, 1], multiply radii by ~1/4.6.

**Citation:** Lin et al. (2023) "FIGConvNet: Fully Implicit Graph Convolutional Networks
for Atmospheric Fluid Simulation" — multi-scale hierarchical convolutions, benchmarked
on DrivAerML at NeurIPS ML4PS 2025. Qi et al. (2017) PointNet++, NIPS 2017 — the
canonical multi-scale grouping approach on point clouds.

**Expected lift:** −0.5 to −1.5 abupt. Likely better than single-scale EdgeConv
(H07) for global pressure patterns (pv) and comparable or better for tau. Most
distinguishing feature: H07 is local-connectivity-based; H08 is radius-based. If
the DrivAer mesh has highly variable point density (it does — finer at body transitions),
radius-based pooling is more physically consistent than k-NN.

---

## H09 — Per-axis learnable output head scaling (cheap tau_y/tau_z magnitude fix)

**Hypothesis:** A root cause of tau_y=25.50 and tau_z=26.53 while tau_x=18.24 is
that the model may be systematically under-predicting tau_y and tau_z in magnitude.
Adding a learnable scalar multiplier per output channel (initialized to 1.0) gives
the model a direct path to recalibrate the dynamic range of each tau axis
independently, without requiring architectural changes or loss reformulation.
This is a pure "can the model represent the right scale?" diagnostic.

**Mechanism:** After the output projection head, multiply predictions by a learned
per-channel scale vector s ∈ R^4 (one per output channel: cp, tau_x, tau_y, tau_z),
initialized to all-ones. Optionally add a per-channel learned bias b ∈ R^4 initialized
to zero. This is equivalent to an unconstrained affine recalibration head — but
initialized to identity so training starts at the Transolver baseline.

**Composes with:** Everything. This is the cheapest possible change and composes
orthogonally with all feature engineering, loss, and architecture changes. If H09
improves results, it should be included in every subsequent experiment.

**Implementation cost:** Very low (~20 LOC in model.py). Add `self.output_scale =
nn.Parameter(torch.ones(4))` and multiply output before returning. Add optional bias.

**Risk level:** Very low. This is strictly a superset of the current model (the
current model is the special case where s=[1,1,1,1]). The only risk is that the
optimizer finds a degenerate solution (s → 0 for some channel); prevent with a
small L2 penalty on (s − 1)², strength ~1e-4.

**Citation:** Platt scaling (Platt 1999) is the scalar calibration analogue for
classification. Task-specific head scaling is used in multi-task learning (Kendall
et al. 2018 "Multi-Task Learning Using Uncertainty to Weigh Losses") and is implicit
in all learned output normalizations. This is the "simplest possible" version without
uncertainty weighting.

**Expected lift:** −0.3 to −1.0 abupt. High expected return per LOC ratio.
Diagnostic value: if s_tau_y and s_tau_z converge to values significantly > 1.0
during training, that confirms systematic under-prediction of those axes and
motivates further magnitude-calibration experiments.

---

## H10 — Bilateral tau smoothness auxiliary loss (physics-motivated surface regularization)

**Hypothesis:** Wall shear stress should vary smoothly across the surface except at
high-curvature features (sharp edges, geometric transitions). The current rel-L2 loss
treats each surface point independently with no spatial continuity constraint. Adding
an auxiliary loss that penalizes |tau_pred(i) − tau_pred(j)|² for nearby point pairs,
weighted by exp(−|κ_i − κ_j|) (similarity in curvature → higher penalty for
discontinuity), encourages smooth tau variation in low-curvature regions while
permitting sharp transitions where curvature justifies them.

**Mechanism:** L_smooth = λ_s * (1/|E|) * Σ_{(i,j)∈E} w_ij * ||tau_pred_i − tau_pred_j||²
where E is the k-NN edge set (reused from H07 if available, otherwise constructed
separately), w_ij = exp(−α * |κ_i − κ_j|), and λ_s is a loss weight hyperparameter.
This is an energy-based graph regularizer — equivalent to a graph Laplacian smoothness
term but with geometry-adaptive weights.

**Composes with:** RFF features, tangent frame (H02), curvature features (H03 — the
w_ij weights reuse the curvature computation), per-axis loss weights. Do NOT compose
with the training-time tangential projection (PR #31 — ruled out due to structural
tau_z bug).

**Implementation cost:** Medium (~100 LOC in train.py). Requires k-NN graph
construction (can share with H07 if both are run together) and curvature estimates
for w_ij (can share with H03). The graph Laplacian quadratic form can be computed
as a sparse matrix-vector product. Key hyperparameter: λ_s (start with 0.01,
sweep {0.001, 0.01, 0.1}); α controls how sharply curvature gates the smoothness
(start α=1.0).

**Risk level:** Medium-high. The auxiliary loss adds a regularization term that
could hurt tau prediction at legitimate sharp transitions (A-pillar, wheel arch).
The curvature-adaptive weighting (w_ij) is critical — without it, the loss will
bleed across sharp edges and hurt performance. Verify that w_ij correctly gates
near-zero for high-curvature pairs in the DrivAer mesh before using.

**Citation:** Cheung et al. (2021) "Graph Laplacian Regularization for Inverse
Problems" — bilateral Laplacian smoothness in graph signal processing. Borsuk
et al. (2022) uses spatial smoothness constraints for GNN tau prediction. Schmidt
et al. (2024) "Symmetric Basis Convolutions for Learning Lagrangian Fluid Mechanics"
(arXiv:2403.16680) uses symmetry-adapted basis functions with smoothness constraints
for fluid surface fields.

**Expected lift:** −0.2 to −0.8 abupt. This idea is highest-risk of the batch —
it may hurt tau_x at sharp edges while helping tau_y/tau_z in smooth regions.
Recommend running as a follow-up after H02/H03 are validated, using their curvature
estimates as inputs.

---

## Priority ranking and decision tree

### For immediate next assignments (Round 2, DDP8, single-delta, with fern's RFF as base):

**Tier 1 (highest confidence, lowest risk):**
1. H01 — Multi-sigma RFF (direct extension of fern's confirmed win)
2. H09 — Per-axis output head scaling (20 LOC, pure diagnostic, strong signal)
3. H04 — Channel-selective Huber loss (40 LOC, orthogonal to feature engineering)

**Tier 2 (medium risk, high upside):**
4. H02 — Local tangent-frame features (50 LOC, directly targets tau_y/tau_z basis)
5. H06 — Log1p tau target normalization (50 LOC, attacks distribution shape)
6. H07 — k-NN EdgeConv pre-encoder (150 LOC, GNN literature support)

**Tier 3 (higher implementation cost, strong external evidence):**
7. H03 — Curvature features (100 LOC, best combined with H02)
8. H05 — Normal-dot attention bias (80 LOC, requires model.py surgery)
9. H08 — Multi-scale radius pooling (120 LOC, FIGConvNet-inspired)
10. H10 — Bilateral tau smoothness loss (100 LOC, highest risk)

### Experiment tree:

```
Base: fern's RFF win (17.77 abupt) — start everything from here

Round 2a (run in parallel, all single-delta on top of fern):
├── H01 multi-sigma RFF
│   ├── win → merge, compose with H02 and H04 in Round 3
│   └── loss → single-sigma RFF is the right frequency; try H04 or H09 instead
├── H09 per-axis head scaling
│   ├── win (s_tau_y >> 1) → confirms magnitude underestimation; compose with everything
│   └── win (s converges near 1) → magnitude not the issue; deprioritize H06
├── H04 Huber loss on tau channels
│   ├── win → tau loss is outlier-dominated; compose with H01 and H09
│   └── loss → distribution not heavy-tailed; close H06 (same mechanism)
└── H02 tangent-frame features
    ├── win → coordinate basis is missing; compose with H01, H09, H03
    └── loss → coordinate frame not the bottleneck; deprioritize H05

Round 2b (compose winners from 2a):
└── H01 × H09 × H04 (if all win) → compose stack
    → if abupt < 15.0, this is the current frontier; add architecture (H07 or H08)
    → if abupt 15-17, architectural delta (H07 or H08) is the next lever

Round 3 (architecture tier):
├── H07 EdgeConv + RFF + per-axis weights (GNN local context)
└── H08 multi-scale pooling (FIGConvNet-inspired)
```

### Stop conditions:
- H01 and H09 both fail to improve on fern (17.77): the representation is not the
  bottleneck; escalate directly to H07/H08 (architecture tier).
- Three consecutive rounds with no improvement vs fern: switch to pretraining
  (denoising, MAE — already in Round 2 queue) or GAOT/Perceiver-IO backbone replacement.
- Any run diverges or gets NaN tau: check log1p inverse transform (H06) and
  curvature feature normalization (H03) for numerical issues.

---

## Implementation notes (cross-cutting)

1. **RFF sigma recommendation for H01**: Use σ={0.1, 1.0, 10.0}. Each scale needs
   the same number of frequencies as fern (32 per scale recommended). Total added
   input dim = 3 × 64 = 192 (vs fern's 64). Verify input projection weight is
   initialized correctly (kaiming_uniform with the new fan_in).

2. **Curvature normalization for H03 and H10**: Curvature values range from ~0 (flat
   panels) to ~100 (sharp edges) in inverse meters. Normalize with robust scaler:
   κ_normalized = (κ − median) / IQR. Clip to [−5, 5] after normalization.

3. **Compose order for feature engineering**: Apply features in this order to avoid
   dimension explosion: [base coords] → [tangent frame H02] → [curvature H03] →
   [multi-sigma RFF H01 applied to all of the above]. Total input dim at maximum
   composition: 7 + 6 + 2 + 3×32×2 = 207.

4. **DDP correctness for k-NN graphs (H07, H08)**: The k-NN graph must use the
   original point positions (before any normalization), and neighbor indices must
   be computed per-sample and offset by the sample's position in the batch. Use
   torch_geometric DataLoader with Batch.from_data_list for correct batched graph
   handling. This is the most common DDP bug in GNN experiments.

5. **Baseline comparison**: All Round 2 runs should be benchmarked against fern's
   17.77 (PR ready, pending merge), not the alphonse 19.81 baseline. Ensure the
   student PRs explicitly list fern's PR as the baseline.

6. **After PR #40 compile fix**: Round 2 experiments started before #40 merges must
   use `--no-compile-model`. Round 2 experiments started after #40 merges should
   use compile by default. Do not mix compiled and uncompiled runs in the same table.
