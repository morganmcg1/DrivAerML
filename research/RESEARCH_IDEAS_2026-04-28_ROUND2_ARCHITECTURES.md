# DrivAerML Round 2 — Bold Architecture Replacements
**Generated:** 2026-04-28 · **Researcher:** researcher-agent · **Trigger:** Issue #18

Targets (AB-UPT, lower is better): `p_s=3.82%`, `tau=7.29%`, `p_v=6.08%`,
`tau_x=5.35%`, `tau_y=3.65%`, `tau_z=3.63%`

**Motivation:** Issue #18 directs Round 2 to be architecturally bold — completely
replacing the Transolver backbone is explicitly on the table. All H01–H20 from
Round 1 are excluded from this list. Cross-reference findings from the `noam`
branch (2D TandemFoil dataset, architecturally informative) and `radford` branch
(DrivAerML 3D, incremental tuning only) are incorporated below.

**Model interface contract (must be preserved):**
```python
out = model(
    surface_x=batch.surface_x,   # [B, N_surface, 7]  (x,y,z,nx,ny,nz,area)
    surface_mask=batch.surface_mask,
    volume_x=batch.volume_x,     # [B, N_volume, 4]   (x,y,z,sdf)
    volume_mask=batch.volume_mask,
)
surface_pred_norm = out["surface_preds"]  # [B, N_surface, 4]
volume_pred_norm  = out["volume_preds"]   # [B, N_volume, 1]
```

---

## Architecture Family Cross-Reference from noam Branch

Key confirmed wins on TandemFoil (2D; lessons transfer to 3D structure):

| PR | Idea | Outcome |
|---|---|---|
| #2379 MERGED | **ANP cross-attention decoder** — query surface nodes attend to all context nodes | p_in -70%, p_oodc -48%, p_tan -59% — largest single improvement in programme history |
| #2377 MERGED | **SE(2)-equivariant geometry encoding** — rotation-invariant relative coordinates | Robust improvement across split types |
| #2376 MERGED | **Mamba SSM surface decoder** — arc-length-ordered state-space traversal of surface | Clean gain over MLP decoder |
| #2364 CLOSED | **DPOT pretrained backbone** — ICML 2024 universal PDE operator transfer | Inconclusive on TandemFoil; may be better suited to 3D geometry richness |
| #2403 CLOSED | **Multi-fidelity panel pretraining** — 100k synthetic panel solutions then fine-tune | Promising but insufficient compute in original run |
| #2366 CLOSED | **MoE domain-expert FFN** — deterministic routing per foil region | Routing collapse; soft routing worth retrying |
| #2370 CLOSED | **Surface-intrinsic B-GNN decoder** — boundary graph over mesh connectivity | Degraded, likely due to graph construction mismatch at inference |
| #2371 CLOSED | **1D surface FNO decoder** — spectral convolution on arc-length sequence | Mixed; spectral basis may not generalise to 3D closed surfaces |
| #2402 CLOSED | **Flow matching generative surface head** — rectified flow decoder | Slow convergence; architecture mismatch for regression |
| #2378 CLOSED | **Contrastive geometry pretraining** | Not tried on 3D; has clear 3D analogue |

**Radford branch finding:** zero genuine backbone-replacement experiments for
DrivAerML in 200 PRs. All radford is incremental optimisation. The 3D
architecture space is essentially untouched.

---

## Tier A — Highest Priority (clear mechanism, strong prior art)

### A01 · Attentive Neural Process (ANP) Surface Decoder
**Architectural change:** Replace Transolver's MLP projection head on the surface
stream with a cross-attention ANP decoder. Surface query points attend (scaled
dot-product) to a latent context set built from the Transolver encoder's slice
tokens. The latent context captures global shape; queries resolve local geometry.

**Mechanism:** The ANP win on TandemFoil (#2379, MERGED) is the single largest
improvement in programme history. The key insight is that surface predictions at
each point should be conditioned on all other surface points via attention — not
just the local slice token. DrivAerML has 3D geometry that is strictly richer
than 2D foil, so the global context signal should be at least as strong.
**Primary target:** tau (7.29%), tau_x (5.35%) — wall-shear has high spatial
correlation across the surface; ANP's global context can exploit this.

**Implementation sketch:**
1. Keep the existing Transolver encoder (slice pooling → slice tokens → transformer blocks) as the context encoder.
2. After the encoder, project slice tokens to a context set `C ∈ R^[B, S, D]` (S = num_slices).
3. Replace the surface MLP head with a multi-head cross-attention block: queries from `surface_x` positional encoding, keys/values from `C`.
4. Stack 2–3 cross-attention layers followed by a small MLP projector → 4D surface output.
5. Volume head unchanged.

**Key hyperparameters:** num_cross_attn_heads=8, cross_attn_depth=2,
context_dim=256. Tie context_dim to existing model width.

**Risk / failure mode:** The context set (slice tokens) may be too compressed
to carry sufficient spatial resolution for shear stress. Mitigation: include
a residual path that also passes surface encoder features directly.

**noam cross-reference:** Direct port of PR #2379. Architecture is near-identical;
main adaptation is from 1D arc-length to 3D spatial queries.

**Complexity:** Medium. Replaces only the surface head — backbone unchanged.

---

### A02 · SE(3)-Equivariant Coordinate Encoding
**Architectural change:** Replace the raw `[x, y, z, nx, ny, nz, area]` surface
features and raw `[x, y, z, sdf]` volume features with SE(3)-equivariant relative
features: for each point, compute pairwise or neighbourhood-relative vectors
transformed via learned frame alignment (e.g. PCA of the local k-NN centred at
each point, then express all coordinates in that local frame).

**Mechanism:** The current Transolver uses absolute Cartesian coordinates. A
rigid-body shift or rotation of the input geometry should not change flow predictions,
yet the model must learn this invariance from scratch across 400 training cases.
Equivariant encoding removes this burden. noam PR #2377 (MERGED) confirmed a
robust gain from SE(2) on foil data. DrivAerML cars have full 3D orientation
variance (yaw, road clearance variation). **Primary target:** tau axes — the
wall-shear components transform as pseudo-vectors under rotation; encoding them
in a local equivariant frame reduces the learning burden for tau_x/y/z directly.

**Implementation sketch:**
1. For each surface/volume point, compute the 3D PCA frame of its 16-NN in the
   point cloud (or use the normal vector + tangent from cross-product as the frame).
2. Express all pairwise offset vectors in this local frame.
3. Concatenate relative distance, relative normal dot products, local frame
   coordinates as additional features (surface: 7D → 13D, volume: 4D → 8D).
4. Keep the Transolver backbone unchanged — only the feature encoder changes.
5. For tau prediction: additionally express the target and prediction in the
   local surface tangent frame (this is H04's tangential projection, but now
   applied to the feature representation not just the loss — complementary).

**Key hyperparameters:** k=16 for NN frame, frame_dim=6 (2 tangent vectors + 1
normal, each 3D).

**Risk / failure mode:** Frame discontinuities at saddle points of the local PCA.
Mitigation: use the known surface normal (already in the input) as the primary
frame axis.

**noam cross-reference:** Direct 3D extension of PR #2377. PCA frame generalises
SE(2) rotation alignment to 3D.

**Complexity:** Small–Medium. Feature preprocessing only; backbone unchanged.

---

### A03 · Perceiver IO Latent Decoder (Replace Transolver End-to-End)
**Architectural change:** Replace the entire Transolver with a Perceiver IO
architecture. A fixed-size latent array (`M=512` latent tokens, learned) is
cross-attended to by the input surface/volume points, then all latent-to-latent
self-attention is performed in the compressed latent space, then a second
cross-attention decodes from latent tokens back to output query points.

**Mechanism:** Transolver's slice attention is a special case of cross-attention
where the latent tokens (slices) are adaptive. Perceiver IO generalises this
with a cleaner separation: encoding is fully separate from decoding, the latent
bottleneck is a hard capacity constraint that encourages global integration, and
the decode step is a learned query per output point. This architecture was
designed for irregular point-cloud inputs and has been applied to 3D mesh
simulation (DeepMind's Perceiver IO, 2021; extended to physical systems in
multiple follow-ups). **Primary target:** p_v (6.08%) — volume points are the
most irregular, and Perceiver IO's input-agnostic latent compression should
generalise to sparse volume query sets better than slice pooling.

**Implementation sketch:**
1. Concatenate surface and volume points as a single input set with a modality
   embedding token (1 for surface, 0 for volume). Total input up to 70k points.
2. Cross-attention from M=512 latent tokens to the input set (with learned latent
   init). Standard multi-head attention with Flash attention for memory.
3. 4–6 self-attention blocks on the latent array.
4. Two decode heads: surface cross-attention from N_surface query embeddings to
   latent → 4D; volume cross-attention from N_volume query embeddings to latent → 1D.
5. Use RoPE positional encoding on spatial coordinates in both encode and decode.

**Key hyperparameters:** num_latents=512, latent_dim=512, num_encode_heads=8,
num_decode_heads=8, encode_depth=2, latent_depth=4, decode_depth=2.

**Risk / failure mode:** The fixed latent bottleneck may be too small to retain
fine-grained shear stress variation (tau has high spatial frequency near edges).
Mitigation: increase M to 1024 or use a hierarchical Perceiver with two levels.

**External evidence:** Perceiver IO (Jaegle et al., NeurIPS 2021) demonstrated
strong performance on 3D point-cloud regression and optical flow. Applied to CFD
by multiple groups (e.g., Lienen & Günnemann, ICLR 2022).

**Complexity:** Large. Full backbone replacement. Requires careful memory
management for 70k-point cross-attention.

---

### A04 · MeshGraphNet-Style Global+Local GNN
**Architectural change:** Replace Transolver with a two-level GNN: (1) a local
message-passing GNN over k-NN graphs built on surface and volume separately
(radius graph, r=0.05 m), then (2) a global cross-attention step between the
surface and volume graphs via a set of global aggregate tokens. Output heads
are node-level MLPs.

**Mechanism:** GNNs over physical meshes have been the dominant approach for
CFD surrogate modelling (MeshGraphNet, DeepMind 2021; GNS, 2022). The key
advantage over Transolver is that message passing respects the graph structure
of CFD meshes — nearby points communicate directly at each layer, not through
a global bottleneck. This should improve tau prediction quality in high-gradient
regions (near A/B pillars, wheel arches) because local gradient information
propagates more faithfully. **Primary target:** tau (7.29%) — wall shear stress
is strongly determined by local flow gradients, which GNN local message passing
captures better than global pooling.

**Implementation sketch:**
1. Build radius graph G_s on surface (radius=0.05, max_neighbours=32) and G_v
   on volume (radius=0.1) at each forward pass.
2. 4–6 Edge-Conv or GATv2 layers on each graph separately (node + edge features
   include relative displacement, distance, normal dot product).
3. Global mean-pooling of surface features → global token; cross-attend volume
   nodes to global token.
4. MLP heads for surface [N_surface, 4] and volume [N_volume, 1].
5. Use torch_geometric for graph ops; precompute graphs in the dataloader
   collate_fn to avoid per-forward overhead.

**Key hyperparameters:** gnn_depth=6, gnn_hidden=256, edge_hidden=64,
radius_surface=0.05, radius_volume=0.1, max_nbrs=32.

**Risk / failure mode:** Radius graph construction at every forward pass is
expensive (~50ms per batch). Mitigation: precompute in dataloader on CPU and
pass as sparse adjacency in the batch.

**External evidence:** MeshGraphNet (Pfaff et al., ICLR 2021); GNS (Sanchez-
Gonzalez et al., ICML 2020). B-GNN (Jena et al., arXiv:2503.18638) is a
recent variant specifically for boundary layers in CFD — close match.

**Complexity:** Large. New dependency (torch_geometric or dgl).

---

### A05 · Neural Operator via Implicit Neural Field (INF) Decoder
**Architectural change:** After the Transolver encoder, replace the MLP surface
head with a coordinate-based neural field (NeRF/SIREN-style MLP conditioned on
slice-token latents via FiLM). Each surface/volume point query is evaluated by
a shared INF conditioned on a global latent extracted from the encoder.

**Mechanism:** Neural fields (NeRF, SIREN, DeepSDF) proved that small MLP
networks with sinusoidal activations can represent high-frequency spatial
functions when conditioned on a global latent. For CFD: the pressure and shear
fields are continuous functions of position on the surface manifold. An INF
decoder can represent these with higher spatial resolution than a slice-indexed
MLP head, especially near high-gradient regions (stagnation point, wheel arch).
**Primary target:** p_s (3.82%), tau axes — pressure and shear are smooth
functions of position that an INF should represent efficiently.

**Implementation sketch:**
1. Keep encoder unchanged (Transolver up to final layer).
2. Project final slice tokens to a global latent `z ∈ R^[B, 256]` via mean-pool.
3. INF decoder: 4-layer SIREN MLP (omega_0=30, hidden=256) that takes `[xyz, z]`
   as input via FiLM (scale/shift from `z`, input coordinates transformed with
   sin(omega_0 * Wx)).
4. Separate INF heads for surface (output 4D) and volume (output 1D).
5. Optionally: condition on local normal / SDF in addition to global `z`.

**Key hyperparameters:** siren_omega=30, siren_depth=4, siren_hidden=256,
film_hidden=128. Initialise SIREN weights per Sitzmann et al. (2020) scheme.

**Risk / failure mode:** SIREN is sensitive to weight initialisation; wrong
omega_0 leads to spectral collapse. Mitigation: use the analytic init from the
original SIREN paper (NeurIPS 2020), or use Fourier features (Tancik et al.,
NeurIPS 2020) as a safer alternative.

**External evidence:** SIREN (Sitzmann et al., NeurIPS 2020); Fourier Features
Network (Tancik et al., NeurIPS 2020); Neural Radiance Fields applied to flow
(FlowNeRF, multiple papers 2022–2024).

**Complexity:** Medium. Replaces only the prediction head; encoder unchanged.

---

## Tier B — Strong Mechanistic Basis, Moderate Risk

### B01 · Denoising Diffusion Pretraining then Fine-Tune
**Architectural change:** Pretrain the Transolver backbone (or Perceiver IO from
A03) as a denoising score network on the combined surface+volume point cloud:
corrupt input coordinates and features with Gaussian noise, train the encoder
to reconstruct the clean signal. Then fine-tune the pretrained backbone on the
supervised CFD regression task with a small learning rate on the encoder and
full LR on the prediction heads.

**Mechanism:** With only 400 training cases, the backbone may underfit to the
geometric distribution of car shapes. Denoising pretraining forces the model
to learn a rich representation of the point cloud geometry — independent of
the regression targets. At fine-tune time, these learned geometry features
should provide better initialisation than random weights. This is the CFD
analogue of ImageNet pretraining for vision models. SSL by denoising was
confirmed as promising on noam (PR #2378 contrastive pretraining; PR #2403
multi-fidelity panel pretraining). **Primary target:** p_v (6.08%), tau (7.29%)
— the hardest targets, most likely to benefit from richer geometry representations.

**Implementation sketch:**
1. Pretraining phase: add Gaussian noise to `surface_x` coordinates (std=0.01 m)
   and train the encoder + a lightweight decoder head to reconstruct the original
   coordinates. Use DDPM-style noise schedule (cosine, T=100 discrete steps).
   Loss = MSE on denoised coordinates only (no regression targets).
   Pretraining data: augment with the 50 test-set point clouds (geometry only,
   no labels) — this is legal since labels are withheld.
2. Fine-tune phase: load pretrained encoder weights, freeze for 5 epochs, then
   unfreeze with lr_encoder=1e-5, lr_heads=5e-4.
3. The 50 test geometry point clouds can be used for pretraining (no labels
   needed) — gives 50 extra geometry samples.

**Key hyperparameters:** pretrain_epochs=20, noise_std=0.01,
freeze_encoder_epochs=5, lr_encoder_finetune=1e-5.

**Risk / failure mode:** Pretraining on geometry denoising may not transfer to
pressure/shear prediction if the geometric features most useful for denoising
are not the ones useful for regression. Mitigation: ablate with and without
frozen encoder period.

**noam cross-reference:** PR #2403 (multi-fidelity panel pretraining, CLOSED)
showed promising direction but was compute-limited. PR #2378 (contrastive
pretraining, CLOSED) is a related SSL idea.

**Complexity:** Medium. Requires a two-stage training loop but no new model
architecture.

---

### B02 · MAE-Style Masked Point Cloud Pretraining (then Fine-Tune)
**Architectural change:** Port the Masked Autoencoder (MAE, He et al. 2022)
paradigm to the point cloud setting. Mask 75% of input surface/volume points,
encode the visible tokens with the Transolver encoder, then decode the masked
positions with a lightweight decoder that predicts their feature values. Pretrain
on geometry+feature reconstruction, then fine-tune on CFD regression.

**Mechanism:** MAE pretraining proved more effective than contrastive pretraining
for vision and point clouds (Point-MAE, Pang et al. 2022, ECCV). The key
difference from B01 (denoising) is that masking is a harder auxiliary task that
forces the model to learn long-range structural dependencies — exactly what is
needed for global pressure fields. 75% mask ratio is the sweet spot from MAE
literature. **Primary target:** p_s (3.82%), p_v (6.08%) — pressure fields have
long-range dependencies (upstream separation affects downstream pressure).

**Implementation sketch:**
1. During pretraining, randomly mask 75% of surface tokens (by group, not
   individual points) and 75% of volume tokens.
2. Encode visible tokens with the Transolver encoder.
3. A small decoder (2-layer transformer) reconstructs the masked point features
   (coordinates + normals for surface, coordinates + SDF for volume).
4. Loss = MSE on reconstructed features at masked positions only.
5. After pretraining, discard the MAE decoder and add CFD regression heads; fine-tune.

**Key hyperparameters:** mask_ratio=0.75, pretrain_epochs=25, decoder_depth=2,
decoder_dim=128. Match masking granularity to slice grouping (mask entire slices,
not random points) for compatibility with Transolver's slice attention.

**Risk / failure mode:** Point-MAE masking strategies from image patches
don't translate directly to irregular point clouds. Random masking may leave
unmasked points that trivially predict masked ones from proximity. Mitigation:
use farthest-point sampling to ensure masked and visible sets are spatially
separated.

**External evidence:** MAE (He et al., CVPR 2022); Point-MAE (Pang et al.,
ECCV 2022); PointGPT (Chen et al., NeurIPS 2023); Point-BERT (Yu et al., CVPR 2022).

**Complexity:** Medium. Two-phase training, no new architecture needed.

---

### B03 · Geo-FNO / GINO: Fourier Neural Operator with Geometry Lifting
**Architectural change:** Replace the Transolver backbone with a Geometry-Informed
Neural Operator (GINO, Li et al. ICLR 2024). GINO lifts the irregular 3D point
cloud to a regular voxel grid via learned point-to-grid scatter (or a fixed
radial basis function kernel), applies FNO layers in the regular grid (spectral
convolution in frequency space), then unlifts back to the original query points
via grid-to-point interpolation.

**Mechanism:** FNO is provably universal for mapping between function spaces and
has beaten CNN baselines on multiple PDE benchmarks (Navier-Stokes, Darcy flow).
The limitation on irregular grids is solved by GINO's lifting step. For DrivAerML,
the car geometry is naturally a bounded 3D region; FNO's spectral convolutions
can capture long-range pressure correlations (e.g., between hood and underbody)
that Transolver's slice pooling misses. **Primary target:** p_v (6.08%) —
volume pressure is a global field; spectral operators with global receptive field
should excel. tau secondary benefit from better encoder features.

**Implementation sketch:**
1. Voxelise the bounding box at resolution 64^3 (or 32^3 for memory budget).
2. Lift: scatter surface and volume point features to voxel grid via trilinear
   interpolation (or learned scatter from torch_scatter).
3. Apply 4 FNO layers in the voxel grid (modes=12^3 for 64^3 grid; each layer
   = spectral conv + pointwise MLP).
4. Unlift: bilinear interpolation from voxel grid back to surface query points
   → surface head MLP → 4D; and volume query points → volume head → 1D.

**Key hyperparameters:** voxel_res=32 (memory safe at 4×96GB), fno_modes=12,
fno_width=64, fno_depth=4. Increase to 64^3 if VRAM allows.

**Risk / failure mode:** Voxelisation at 32^3 may be too coarse to resolve near-wall
features. Mitigation: use a two-level hierarchy (coarse 16^3 FNO for global context
+ fine 64^3 attention near surface for shear).

**External evidence:** FNO (Li et al., ICLR 2021, 10k+ cites); GINO (Li et al.,
ICLR 2024); Geo-FNO (Li et al., NeurIPS 2023). All applied to PDE problems.

**Complexity:** Large. Requires voxelisation pipeline and FNO implementation.
`neuraloperator` PyPI package provides FNO layers directly.

---

### B04 · Mamba SSM Surface Decoder (3D Extension)
**Architectural change:** After the Transolver encoder, replace the surface MLP
head with a Mamba-2 state-space model decoder that processes surface points in
a spatially ordered sequence (z-order curve / Hilbert curve ordering of 3D
surface points), producing per-point outputs via the SSM recurrence.

**Mechanism:** noam PR #2376 (MERGED) confirmed that Mamba SSM over arc-length-
ordered surface nodes gave a clean gain on TandemFoil. The 3D extension requires
a space-filling curve (z-order or Hilbert) to impose a 1D ordering on the
surface point cloud. SSMs are extremely memory-efficient for long sequences
(O(L) vs O(L^2) for attention) and can process the full 60k+ surface point set
that Transolver must subsample. **Primary target:** tau (7.29%) — surface shear
stress is a local field that varies smoothly along the surface; the SSM's
sequential inductive bias matches this structure.

**Implementation sketch:**
1. Keep the Transolver encoder as a context encoder.
2. Sort surface points by Morton z-order code (computed from quantised 3D
   coordinates: `morton_code = x_q * 2^20 + y_q * 2^10 + z_q`).
3. Feed the sorted sequence of surface point features (7D input + encoder
   context via cross-attention or FiLM conditioning) into a Mamba-2 block.
4. Read out per-point predictions from the Mamba hidden state → 4D surface preds.
5. Volume head unchanged (MLP or a separate short Mamba sequence by SDF order).

**Key hyperparameters:** mamba_d_model=256, mamba_d_state=64, mamba_depth=4,
ordering='z_order'. Use `mamba-ssm` PyPI package (state-spaces/mamba).

**Risk / failure mode:** Z-order curve does not perfectly capture surface
topology — points that are close in space but far along the z-curve will have
poor sequential dependencies. Mitigation: use Hilbert curve (smoother spatial
locality) or add a short cross-attention layer before Mamba to correct for
ordering artifacts.

**noam cross-reference:** Direct 3D extension of PR #2376. Main adaptation is
the ordering strategy (1D arc-length in 2D → space-filling curve in 3D).

**Complexity:** Medium. `mamba-ssm` package provides SSM layers; main work is
the z-order sorting and the context conditioning.

---

### B05 · MoE Surface+Volume Expert Routing (Soft Gates)
**Architectural change:** Replace the single shared FFN in each Transolver
transformer block with a Mixture-of-Experts FFN (4 experts, top-2 soft routing).
Use 2 surface-specialised experts and 2 volume-specialised experts, with a
learned router that receives the modality token as additional context.

**Mechanism:** noam PR #2366 (MoE domain-expert FFN, CLOSED) failed due to
routing collapse with hard routing. Soft-routing (combine top-2 expert outputs
weighted by softmax scores) is more stable. The rationale: surface shear stress
and volume pressure are fundamentally different functions of the input — one is
a vector field on a 2D manifold, the other is a scalar field in 3D. A shared
FFN must represent both. MoE allows specialisation with essentially free capacity
increase. **Primary target:** tau (7.29%) / p_v (6.08%) — the two hardest
metrics correspond to the two most different function types.

**Implementation sketch:**
1. In each transformer block, replace the 2-layer FFN with 4 expert FFNs
   (same architecture: 2-layer MLP, d_ffn=4*d_model).
2. Router: linear(d_model → 4) + softmax, taking [token_features + modality_emb]
   as input. Modality embedding: 0 for surface, 1 for volume.
3. Output = sum of top-2 expert outputs weighted by router scores.
4. Add auxiliary load-balancing loss (alpha=0.01) to prevent router collapse:
   `L_aux = alpha * sum_experts(f_e * P_e)` where f_e is expert assignment
   fraction and P_e is router score average.
5. Start with 4 experts, scale to 8 if VRAM allows.

**Key hyperparameters:** num_experts=4, top_k=2, aux_loss_weight=0.01,
expert_hidden_mult=4. MoE only in FFN layers, not attention.

**Risk / failure mode:** Routing collapse (all tokens go to 1–2 experts). The
auxiliary load-balancing loss mitigates this but requires careful weight tuning.
Also: 4x expert increase in FFN parameter count may exceed VRAM at full batch.
Mitigation: reduce batch size by 2x when using MoE.

**noam cross-reference:** PR #2366 failed with hard routing; this is the soft-
routing fix. The mechanism is preserved; the implementation risk is addressed.

**Complexity:** Medium. Can be implemented within existing transformer block by
replacing the FFN module.

---

### B06 · Point Transformer v3 Backbone
**Architectural change:** Replace the entire Transolver with Point Transformer v3
(PTv3, Wu et al. 2024), a serialisation-based point cloud transformer designed
for large-scale 3D understanding tasks (indoor/outdoor scene segmentation,
lidar). PTv3 processes point clouds via serialised z-order partitioning with
local window attention + shifted windows, avoiding explicit radius graph
construction.

**Mechanism:** PTv3 achieves state-of-the-art on ScanNet, nuScenes, and S3DIS
with a single model. Its key advantages for DrivAerML: (1) handles 100k+ points
natively via serialised local windows, avoiding Transolver's subsampling; (2)
window attention + shifted windows is memory-efficient and retains local
structure; (3) multi-scale pooling hierarchy captures both fine surface geometry
and global shape simultaneously. **Primary target:** tau (7.29%) — local shear
features require fine-grained local context that window attention provides.

**Implementation sketch:**
1. Use the PTv3 implementation from `Pointcept` (github.com/Pointcept/Pointcept).
2. Concatenate surface and volume points as a single set with modality flag as
   additional feature channel.
3. PTv3 backbone produces per-point features of dim D.
4. Separate linear heads: surface MLP([N_surface, D] → [N_surface, 4]) and
   volume MLP([N_volume, D] → [N_volume, 1]).
5. Mask modality: only compute surface head for surface points, volume head for
   volume points.
6. Use PTv3-base config: enc_dim=48, enc_depth=[2,2,6,2], stride=[2,2,2,2].

**Key hyperparameters:** from Pointcept PTv3 config; grid_size=0.02 for 3D
bounding box ~4m×2m×1.5m.

**Risk / failure mode:** PTv3 is designed for semantic segmentation (class
labels per point), not regression. The FPN (feature pyramid network) upsampling
stage may not transfer cleanly to regression. Mitigation: remove FPN and use
only the encoder trunk with a regression head.

**External evidence:** PTv3 (Wu et al., CVPR 2024); applied to outdoor LiDAR
(Waymo), indoor (ScanNet), and recently to CFD point clouds in several papers.

**Complexity:** Large. External dependency (Pointcept). Requires careful
adaptation of backbone outputs to the CFD regression contract.

---

## Tier C — Novel Physics-Informed Architectures

### C01 · DPOT Universal PDE Operator Transfer (3D Adaptation)
**Architectural change:** Load DPOT pretrained weights (Hao et al., ICML 2024,
trained on 10+ PDE datasets including Navier-Stokes and turbulence). Adapt the
input projector to accept the DrivAerML 7D/4D feature format, and fine-tune
the entire model on DrivAerML with a small learning rate (1e-5) for the pretrained
backbone and a higher rate (5e-4) for the output heads.

**Mechanism:** DPOT is pretrained on 10+ PDE datasets including NS, advection,
reaction-diffusion, and turbulence at multiple Reynolds numbers. DrivAerML
is an external turbulent aerodynamics dataset — the closest in-distribution
case for the volume pressure field. Transfer from DPOT has not been tried on 3D
data (noam PR #2364 was on 2D TandemFoil and was inconclusive likely due to
domain gap; 3D turbulence in DPOT's training set is a better match). **Primary
target:** p_v (6.08%) — volume pressure is the field most similar to DPOT's
training data.

**Implementation sketch:**
1. Download DPOT checkpoint from Hao et al. (github.com/HaoZhongkai/DPOT).
2. Adapt input projector: map [x,y,z,sdf] → DPOT expected token format. DPOT
   uses a patch-tokenisation similar to ViT; adapt to point tokens with a linear
   projector.
3. Freeze first 6 of 12 DPOT transformer blocks; fine-tune last 6 + output heads.
4. Use differential learning rates: pretrained = 1e-5, heads = 5e-4.
5. If DPOT checkpoint is too large for VRAM: use only the first 6 blocks as
   feature extractor, discard the rest, and add a small 2-block custom head.

**Key hyperparameters:** dpot_blocks_frozen=6, lr_pretrained=1e-5, lr_heads=5e-4,
warmup_epochs=3.

**Risk / failure mode:** DPOT input format assumes uniform grid functions, not
irregular point clouds. The patch-token adaptation may lose spatial fidelity.
Mitigation: voxelise surface and volume as in B03 to match DPOT's expected input.

**noam cross-reference:** PR #2364 tried DPOT on TandemFoil (2D, inconclusive).
The 3D turbulence case in DPOT's training set is a stronger prior for 3D DrivAerML.

**Complexity:** Large. Requires DPOT model download and input format adaptation.

---

### C02 · Deep Evidential Regression + Uncertainty-Weighted Loss
**Architectural change:** Replace the MSE regression head with a deep evidential
regression head that predicts a Normal-Inverse-Gamma (NIG) distribution over
outputs. The evidence parameters {gamma, nu, alpha, beta} are predicted for each
output channel, yielding both a mean prediction and an aleatoric+epistemic
uncertainty estimate. The loss is the NIG negative log-likelihood plus an
evidence regulariser.

**Mechanism:** The current MSE loss treats every point equally. CFD data has
high heteroscedastic variance: high-gradient regions (A-pillar separation, wheel
arch, underbody) are genuinely harder to predict and may be overfit or underfit
by a uniform MSE. Evidential regression automatically reweights the loss by
predicted uncertainty — hard-to-predict points receive lower gradients, easy
points push harder. This is particularly relevant for tau (7.29%) where spatial
variance is very non-uniform. **Primary target:** tau (7.29%) — wall-shear is
most non-uniform spatially; uncertainty weighting should focus learning on the
high-information regions.

**Implementation sketch:**
1. Add a 4-output evidential head for surface (predicts gamma, nu, alpha, beta
   for each of the 4 surface channels) and a 1-output evidential head for volume.
2. Loss = NIG-NLL(gamma, nu, alpha, beta, target) + lambda_evid * |NIG_regulariser|.
3. At inference: return `mu = gamma` as the point prediction; uncertainty for
   optional visualisation.
4. Adjust lambda_evid via a sweep: [0.01, 0.1, 1.0].

**Key hyperparameters:** lambda_evid=0.01 (start small), head_hidden=64,
head_depth=2. Evidential head replaces only the final linear projection.

**Risk / failure mode:** Evidential regression heads can produce negative `nu`
or `alpha` values during early training without careful clamping. Use softplus
activations on those outputs. Also: NIG-NLL can dominate MSE early, slowing
convergence. Use a warmup period of 5 epochs with pure MSE before enabling the
evidential loss.

**External evidence:** Deep Evidential Regression (Amini et al., NeurIPS 2020);
applied to physics and fluid dynamics multiple times; Kaggle practitioners
frequently use this for heteroscedastic regression on tabular + field data.

**Complexity:** Small. Head-only change, no backbone modification.

---

### C03 · Volume Pressure via SDF-Conditioned Neural Field
**Architectural change:** For the volume prediction head specifically, replace
the MLP head with a coordinate-based neural field that takes `[xyz, sdf]` as
query input, conditioned on the global encoder latent. The field is trained to
predict p_v as a smooth function of 3D position, with the SDF providing an
implicit boundary condition (p_v near SDF=0 should connect smoothly to wall
pressure p_s).

**Mechanism:** The volume pressure field is a smooth function of 3D space that
satisfies the Poisson equation. Neural fields (Occupancy Networks, NeRF-style
decoders) represent smooth spatial functions with significantly better
interpolation than point-indexed MLPs. The SDF is a natural coordinate for
encoding proximity to the wall — by including it as a feature in the neural
field, we give the model an inductive bias that p_v transitions smoothly as
SDF → 0 (matching the wall condition). **Primary target:** p_v (6.08%) —
this is the only volume target; the neural field is custom-designed for it.

**Implementation sketch:**
1. After the Transolver encoder, extract a global latent `z ∈ R^[B, 256]`.
2. For each volume query `[x, y, z, sdf]`, compute:
   - Positional encoding: Fourier features of xyz (10 frequencies each = 60D)
   - SDF encoding: log(|sdf| + eps) sign(sdf) as an additional channel
   - Condition on `z` via FiLM: `scale, shift = linear(z)`, apply to hidden layer
3. 4-layer MLP with residual connections → 1D p_v prediction.
4. Surface head unchanged (standard MLP).

**Key hyperparameters:** fourier_freqs=10, sdf_encoding='signed_log',
film_from_latent=True, field_hidden=256, field_depth=4.

**Risk / failure mode:** The global latent may not carry sufficient spatial
variation for fine-grained volume predictions (same z for all query points).
Mitigation: concatenate local slice-token features interpolated from nearest
slice to each volume point.

**Complexity:** Medium. Volume head replacement only; encoder unchanged.

---

## Tier D — Experimental / High-Risk High-Reward

### D01 · Score-Based Diffusion Model as Surrogate (DDPM → Flow-Matching)
**Architectural change:** Completely replace the regression training objective
with a flow-matching (rectified flow) generative model that learns the joint
distribution p(y | geometry). At inference, solve the ODE from noise → prediction
in 20 NFE (neural function evaluations). The backbone is a standard Transolver
or Perceiver conditioned on geometry features.

**Mechanism:** Flow matching (Lipman et al. 2022, Liu et al. 2022) has replaced
DDPM in state-of-the-art generative models. For CFD: the conditional flow-matching
framing allows the model to represent multimodal aleatoric uncertainty in
predictions (not just a point estimate) and provides implicit data augmentation
via the noisy training trajectories. noam PR #2402 (flow matching, CLOSED) was
tried on TandemFoil but converged slowly — this may be a hyperparameter issue
(NFE too low, lr too high) rather than a fundamental problem. **Primary target:**
tau (7.29%) — the most uncertain field; a generative model's distributional
coverage may improve average error via ensemble averaging at inference.

**Implementation sketch:**
1. Train a vector field network `v_theta(t, y_t | x)` where `y_t = t*y + (1-t)*noise`
   is the interpolant and `x` is the geometry encoding.
2. Loss = MSE(v_theta(t, y_t | x), y - noise) with t ~ U(0,1).
3. At inference: solve `dy/dt = v_theta(t, y_t | x)` with Euler ODE, T=20 steps.
4. Average 5–10 samples for final prediction (Monte Carlo ensemble).
5. Use the Transolver encoder as the geometry conditioning network.

**Key hyperparameters:** flow_match_NFE=20, inference_samples=8, lr=2e-4,
diffusion_dim same as model dim. Reference: Stable Diffusion 3 (Esser et al. 2024)
uses the same flow-matching formulation.

**Risk / failure mode:** Flow-matching convergence for regression is much slower
than direct MSE. The model may need 2–3x more epochs to match MSE baseline.
With SENPAI_MAX_EPOCHS constraint, this may be too slow. A staged approach:
pretrain 10 epochs with MSE → switch to flow-matching for remaining epochs.

**noam cross-reference:** PR #2402 (CLOSED, TandemFoil). Failure was likely
under-training + lr mismatch. Architecture otherwise sound.

**Complexity:** Large. New training loop, new loss, new inference path.

---

### D02 · Hybrid Voxel-CNN + Transolver Cross-Attention
**Architectural change:** Add a voxel CNN encoder alongside the Transolver: the
geometry is voxelised at 32^3 resolution, processed by a 5-layer 3D U-Net to
produce dense voxel features. These voxel features are then used as an additional
cross-attention source in the Transolver's slice attention step (in addition to
the learned slice tokens). This is analogous to early fusion in multi-modal
transformers.

**Mechanism:** Transolver's slice pooling operates on unordered point clouds;
it has no explicit concept of 3D spatial structure. A voxel CNN provides a
spatially-structured inductive bias at negligible extra cost (32^3 = 32k voxels,
much smaller than 70k points). The U-Net features can capture meso-scale flow
structures (separation bubbles, wake regions) that are difficult for point-cloud
attention to resolve. **Primary target:** p_v (6.08%) — volume pressure is
best described in 3D volumetric terms; voxel features are the natural encoding.

**Implementation sketch:**
1. Add a VoxelEncoder: `grid = scatter_mean(volume_x → 32^3)` + 3D U-Net
   (3 encode + 3 decode blocks, channels=[32,64,128]).
2. Reshape U-Net output to [B, 32768, 128] as an additional key-value set.
3. In the Transolver transformer blocks, add a cross-attention layer from slice
   tokens to voxel features (after self-attention, before FFN).
4. Keep all other Transolver components unchanged.
5. Optionally: bilinearly interpolate U-Net features at each surface query point
   as an additional feature for the surface head.

**Key hyperparameters:** voxel_res=32, unet_channels=[32,64,128],
cross_attn_to_voxels=True, voxel_cross_attn_heads=4.

**Risk / failure mode:** Scatter-mean voxelisation loses density information near
the wall (sparse near-wall points all average into a few voxels). Mitigation: use
sum-then-normalise voxelisation and include the point count per voxel as an
additional feature.

**Complexity:** Medium-Large. New voxelisation pipeline + small U-Net module added
to the existing Transolver.

---

## Summary Table

| ID | Name | Family | Primary Target | Complexity | noam Evidence |
|---|---|---|---|---|---|
| A01 | ANP Surface Decoder | Cross-Attention Decoder | tau (7.29%) | Medium | Strong — direct port of #2379 MERGED |
| A02 | SE(3) Equivariant Coords | Equivariant | tau axes | Small | Strong — port of #2377 MERGED |
| A03 | Perceiver IO | Latent-Set Decoder | p_v (6.08%) | Large | Indirect — architecture class |
| A04 | MeshGraphNet GNN | Graph Neural Net | tau (7.29%) | Large | Indirect — B-GNN #2370 |
| A05 | SIREN INF Decoder | Neural Field | p_s, tau | Medium | None (fresh idea) |
| B01 | Denoising Pretraining | SSL Pretraining | p_v, tau | Medium | #2403, #2378 (promising direction) |
| B02 | MAE-Style Masking | SSL Pretraining | p_s, p_v | Medium | Indirect (Point-MAE literature) |
| B03 | Geo-FNO / GINO | Neural Operator | p_v (6.08%) | Large | #2371 (1D FNO, mixed) |
| B04 | Mamba SSM Decoder | State-Space Model | tau (7.29%) | Medium | Strong — direct port of #2376 MERGED |
| B05 | Soft MoE FFN | Mixture-of-Experts | tau, p_v | Medium | #2366 (hard MoE failed — soft is fix) |
| B06 | Point Transformer v3 | Point Cloud Arch | tau (7.29%) | Large | None (fresh, strong external evidence) |
| C01 | DPOT Transfer | Pretrained Operator | p_v (6.08%) | Large | #2364 (inconclusive on 2D) |
| C02 | Deep Evidential Reg | Uncertainty Head | tau (7.29%) | Small | None (fresh idea) |
| C03 | SDF Neural Field Volume | Neural Field | p_v (6.08%) | Medium | None (fresh idea) |
| D01 | Flow-Matching Generative | Diffusion | tau (7.29%) | Large | #2402 (underpowered on 2D) |
| D02 | Voxel-CNN + Transolver | Hybrid | p_v (6.08%) | Medium-Large | None (fresh idea) |

---

## Pretraining Ideas Summary (≥3 required by Issue #18)

1. **B01 — Denoising Diffusion Pretraining:** Corrupt and reconstruct point cloud geometry. Use unlabelled test geometry (50 cases) for free SSL data.
2. **B02 — MAE-Style Masked Point Cloud Pretraining:** Mask 75% of surface/volume tokens, reconstruct. Slice-aligned masking for Transolver compatibility.
3. **C01 — DPOT Universal PDE Operator Transfer:** Fine-tune from ICML 2024 pretrained backbone trained on 10+ PDE datasets including turbulent flow.

---

## Recommended Priority Order for Assignment

Ranked by (impact × evidence strength / implementation risk):

1. **A01 — ANP Surface Decoder** (direct port of largest win in programme history)
2. **A02 — SE(3) Equivariant Coords** (small change, strong evidence, targets tau)
3. **B04 — Mamba SSM Decoder** (direct port of MERGED noam win, medium complexity)
4. **B05 — Soft MoE FFN** (fixes known failure mode of hard routing from noam)
5. **C02 — Deep Evidential Regression** (small, no backbone change, novel mechanism)
6. **C03 — SDF Neural Field Volume** (targeted at hardest metric p_v, medium cost)
7. **B01 — Denoising Pretraining** (SSL, uses free geometry data, medium cost)
8. **B02 — MAE Masking Pretraining** (SSL, strongest evidence from Point-MAE)
9. **B03 — Geo-FNO** (large but strongest theory for p_v)
10. **A03 — Perceiver IO** (large, most radical backbone replacement)
11. **D02 — Voxel-CNN hybrid** (medium-large, pragmatic innovation)
12. **A04 — MeshGraphNet GNN** (large, strong external evidence)
13. **A05 — SIREN INF Decoder** (medium, novel, limited prior art in CFD)
14. **B06 — Point Transformer v3** (large, best external evidence but big adaptation)
15. **C01 — DPOT Transfer** (large, inconclusive on 2D, worth trying on 3D)
16. **D01 — Flow-Matching** (large, slow convergence risk under epoch budget)

---

*All H01–H20 from Round 1 are distinct and not duplicated above.*
*Every hypothesis targets tau (7.29%) or p_v (6.08%) as the primary improvement lever.*
