# RESEARCH IDEAS — 2026-05-17 22:40Z
# Wave 30 — alphonse next assignment after H25 ALGP closure

## Context

alphonse H25 ALGP: TERMINAL NOT-A-MERGE. Run crashed; EP3 val_abupt 7.142% FAIL (+1.016pp),
τz/τx 1.5117 — 6th confirmed cold-start fade, rebounding into band attractor [1.44, 1.55].
11 closures in Wave 30. Plateau Protocol applies. Bold, high-variance ideas are appropriate.

**Baseline (PR #972, run 56bcqp3m):**
- val_abupt = 6.126%, test_SP = 3.577%, test_vol_p = 3.643%, test_WSS = 6.727%, test_abupt = 5.844%

**Hard constraints:**
- Lion optimizer lr=9e-5, hidden_dim=512, layers=5, slices=128, heads=4, batch_size=4
- 18h budget: SENPAI_TIMEOUT_MINUTES=1100, 13 epochs, vol schedule `0:16384:3:32768:6:49152:9:65536`
- DDP-8, branch `tay`, NO ensembles

**Closed axes (must not reopen):**
1. Per-vertex loss reweighting (all variants, H10b/H11b/H12/H16/H16b/H20/H22/H18/H18d)
2. EMA self-distillation
3. Channel-coupled decoder MLP capacity
4. Position-density area weight (all variants)
5. Mean-shift saliency manipulation
6. Mean-shift auxiliary gradient prediction (H25 ALGP)

**Active experiments (must not overlap):**
- H24 fern: geometric saliency slice temperature (encoder)
- H26 thorfinn: normal-projected coord augmentation (input)
- H27 askeladd: per-component rel-L2 proxy loss (loss space)
- H28 edward: SAM sharpness-aware minimization (optimizer)
- H29 frieren: spectral surface loss streamwise upweighting (loss frequency domain)
- H30 nezuko: volume-to-surface cross-attention (encoder cross-modal fusion)
- H18d tanjiro: channel-decoupled τz-only area weight (loss diagnostic)

---

## Hypothesis Table

| Slug | Family | One-line hypothesis | Why for alphonse | LOC | Risk/reward | Orthogonal |
|------|--------|---------------------|------------------|-----|-------------|------------|
| WALLDIST | Input Feature / Physics | Add `log(|sdf|+1e-3)` as a 5th volume input channel to give the encoder explicit boundary-layer distance information | H21 proved encoder representation gap — encoder lacked the info to distinguish near-wall τz; WALLDIST injects that physics directly | ~25 | Low risk, high physics justification, easy to falsify | Yes — H26 changes coord frame; WALLDIST adds a new physics channel |
| DIFFATTN | Backbone Architecture | Replace `F.scaled_dot_product_attention` with differential attention (`A₁ − λA₂`) to cancel attention noise and amplify τz-carrying slice signals | Attention noise aliasing in slice tokens is a plausible cause of the τz/τx band attractor; differential attention provably cancels this class of noise | ~80 | Medium risk, strong theoretical basis (MSFT 2024), confined to TransolverAttention class | Yes — H24 only modulates temperature scalar within existing sdpa |
| RESIDCFD | Training Objective / Output Head | Predict Δ = y − μ_train (residual over dataset mean field) instead of raw field values to reduce output dynamic range and focus capacity on car-specific variation | Band attractor [1.44, 1.55] may reflect the model collapsing toward a dataset mean τz/τx ratio; residual prediction removes the mean and forces car-specific fitting | ~50 | Low-medium risk, training objective axis untested in Wave 30 | Yes — H27 changes loss normalization space; RESIDCFD changes what is predicted |
| MSMHA | Backbone Architecture | Run parallel coarse (32 slices) and fine (128 slices) slice pools within each TransformerBlock and merge, capturing both global pressure and local separation structure | τz band attractor may reflect the 128-slice resolution failing to separate near-separation from smooth-surface tokens; coarse track provides global context | ~100 | Medium-high risk, but architecturally orthogonal to all active | Yes — H24 modulates single-scale temperature; MSMHA adds a parallel scale |
| WALLAUX | Auxiliary Task / Output Head | Add a binary auxiliary classification head predicting τz sign-change regions (flow separation pseudo-labels) to force the encoder to learn separation-aware representations | H21 showed gradient decoupling worked perfectly but encoder lacked τz-resolving info; forcing the encoder to classify separation should build that representation organically | ~90 | Medium risk, supervised signal grounded in known physics failure mode, distinct from H25 ALGP (mean-shift local gradients) | Yes — H25 ALGP used mean-shift local gradient prediction; WALLAUX uses sign-change binary classification |

---

## Implementation Sketches

### WALLDIST — Log-SDF Boundary-Layer Input Feature

In `trainer_runtime.py` (or `train.py`), after loading `volume_x` shaped [B, N, 4] with features
[x, y, z, sdf], compute `log_sdf = torch.log(volume_x[..., 3:4].abs() + 1e-3)` and concatenate
along the feature dim to get [B, N, 5]. Pass `volume_input_dim=5` to `SurfaceTransolver`
constructor; `project_volume_features` (a `LinearProjection(volume_input_dim - 3, hidden_dim)`)
handles the expanded dim automatically since `volume_extra_dim` becomes 2. No changes to
`data/loader.py` (read-only) and no changes to `model.py` are needed.

**Mechanism:** The SDF value already in volume_x is linear near the wall, but boundary layer
thickness scales logarithmically with wall distance (y+ ~ 5–300 viscous sublayer). Log-SDF gives
the model a feature that is roughly uniform across the boundary layer, enabling the backbone to
attend meaningfully to near-wall vs. far-field volume tokens and propagate that distinction to the
surface τz head via surf_to_vol_xattn.

**EP3 falsifiable gate:** τz/τx ≤ 1.42 AND val_abupt ≤ 6.5% AND val_vol_p ≤ 4.8%.
KILL: val_abupt > 8.0% OR τz/τx > 1.55 at EP3.

---

### DIFFATTN — Differential Attention in Transolver Slice Layers

In `TransolverAttention.__init__` (model.py line ~238), replace `self.qkv = LinearProjection(dim_head,
dim_head * 3, bias=False)` with `self.qkv = LinearProjection(dim_head, dim_head * 6, bias=False)`
and add `self.diff_lambda = nn.Parameter(torch.tensor(0.8))`. In `forward`, split the 6-chunk
qkv into `(q1, k1, v1, q2, k2, v2)`, compute
`out_slice = F.scaled_dot_product_attention(q1, k1, v1) - self.diff_lambda.sigmoid() * F.scaled_dot_product_attention(q2, k2, v2)`,
then apply layer norm as in the MSFT paper. Add `use_diff_attn: bool = False` flag to both
`TransolverAttention` and `SurfaceTransolver` constructors so `use_diff_attn=False` recovers
exact baseline.

**Mechanism:** In the Transolver slice mechanism, softmax attention over 128 slice tokens creates
a distributed but noisy assignment — each slice aggregates a weighted mean of all points. The
differential term `−λA₂` cancels systematic attention noise (the "background leakage" shared
across both attention maps), isolating the genuinely informative slice tokens that carry τz
information. This directly targets the τz/τx aliasing hypothesis.

**EP3 falsifiable gate:** τz/τx ≤ 1.42 AND val_abupt ≤ 6.5% AND val_vol_p ≤ 4.8%.
KILL: val_abupt > 9.0% OR τz/τx > 1.55 at EP3.

---

### RESIDCFD — Residual Prediction over Dataset Mean Field

In `trainer_runtime.py`, precompute `μ_train` (shape [4] for surface channels, [1] for volume)
as the training-set per-channel mean of raw (denormalized) field values once at epoch 0, register
as a non-trainable buffer. Before loss computation, subtract `μ_train` from normalized targets;
before metric evaluation, add `μ_train` back to model outputs. The normalization pipeline and
model architecture are unchanged — the model just learns `f(x) = y − μ` instead of `y`.

**Mechanism:** The band attractor τz/τx ∈ [1.44, 1.55] is suspiciously close to a dataset-mean
ratio — the model may be collapsing toward a prior rather than fitting the car-specific field.
Subtracting the mean forces the model to predict zero in the trivial case and allocates all model
capacity toward the car-shape-dependent residual. The vol_p floor disease (H18 unique pass) may
also reflect mean-field collapse in the volume head.

**EP3 falsifiable gate:** τz/τx ≤ 1.42 AND val_abupt ≤ 6.5% AND val_vol_p ≤ 4.5%.
KILL: val_abupt > 8.5% OR τz/τx > 1.55 at EP3.

---

### MSMHA — Multi-Scale Slice Attention (128 ∥ 32)

In `TransformerBlock.__init__`, add `self.attention_coarse = TransolverAttention(hidden_dim,
num_heads, num_slices=32)` alongside the existing `self.attention` (128 slices). In `forward`,
run both: `fine = self.attention(x)`, `coarse = self.attention_coarse(x)`, then apply a learned
merge: `self.merge = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)` (zero-initialized) so
`out = x + self.merge(torch.cat([fine, coarse], dim=-1))`. Add `use_multiscale: bool = False`
flag to `SurfaceTransolver` to gate the coarse branch and allow baseline recovery.

**Mechanism:** 128 slices at dim_head=128 may be too fine-grained to capture global pressure
structures that couple to τz (e.g., the separation-reattachment bubble on the roof). The coarse
32-slice track aggregates larger patches, providing the global context the fine track lacks.
The merge is zero-initialized, so at epoch 0 the model is identical to baseline; the coarse
branch engages only as the merge weight grows.

**EP3 falsifiable gate:** τz/τx ≤ 1.42 AND val_abupt ≤ 6.5%.
KILL: val_abupt > 9.5% OR τz/τx > 1.60 at EP3 (coarse branch destabilization risk).

---

### WALLAUX — Separation-Line Auxiliary Binary Classification

In `model.py`, after `surface_out`, add `self.sep_head = nn.Sequential(nn.Linear(n_hidden, n_hidden // 4), nn.SiLU(), nn.Linear(n_hidden // 4, 1))` with the final linear zero-initialized. In
`trainer_runtime.py`, precompute per-point pseudo-labels from training τz targets: a point is
labeled "separation" if its τz sign differs from the majority of its K-nearest neighbors (e.g.,
K=8 in normalized coord space, precomputed once). Add weighted BCE:
`loss = main_loss + λ_sep * bce_loss(sep_head(surface_hidden).squeeze(-1), pseudo_labels)` with
`λ_sep=0.05`. Zero-init ensures exact baseline at `λ_sep=0.0`.

**Mechanism:** H21 proved that decoder-level gradient decoupling worked perfectly, but failed
because the encoder backbone did not have a τz-resolving representation to decouple from.
WALLAUX directly forces the backbone to build a representation that localizes sign changes in τz,
which is precisely the information H21 needed and could not find. Unlike H25 ALGP (mean-shift
local gradient prediction in continuous space), this is a binary classification task over
topological flow features, so pseudo-labels are sparse and high-signal.

**EP3 falsifiable gate:** τz/τx ≤ 1.42 AND val_abupt ≤ 6.5% AND sep_loss DESCENDING.
KILL: val_abupt > 9.0% OR sep_loss diverging at EP3.

---

## Top Picks for alphonse

### Pick 1: WALLDIST

**Why first:** ~25 LOC change, physically motivated, directly addresses the encoder representation
gap that H21 proved exists, and uses SDF information already sitting in volume_x[..., 3]. The
log transform is the key insight — raw SDF is nearly zero at the wall and grows linearly, which
is the wrong scale for boundary-layer physics. Log-SDF is roughly uniform across the y+ range
where τz is determined (viscous sublayer 5 < y+ < 300). This gives the backbone a feature it
currently cannot derive from positional embeddings alone. Falsifiable at EP3 without any
architectural risk.

### Pick 2: DIFFATTN

**Why second:** The τz/τx band attractor is suspiciously stable across 10+ independent runs.
A stable attractor in loss space usually means a basin the optimizer cannot escape — SAM (H28)
is attacking this from the optimizer side, but differential attention attacks it from the
attention mechanism side. If slice token noise is the cause of the attractor, differential
attention directly removes that noise at the source. The Microsoft Research 2024 paper shows
consistent gains on token-level tasks and suppression of attention sink pathology — both
plausible descriptions of the τz aliasing problem. The change is confined to TransolverAttention
and recovers exact baseline with `use_diff_attn=False`. Orthogonal to H28 SAM (optimizer) and
H24 (temperature only).

**Assignment recommendation:** WALLDIST is the highest EV per unit compute. DIFFATTN is the
bolder bet. If alphonse is the next idle student, assign WALLDIST. If a second slot opens
simultaneously, DIFFATTN is the natural second choice.
