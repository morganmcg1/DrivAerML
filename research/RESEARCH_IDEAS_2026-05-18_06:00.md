# Wave 31 Research Hypotheses
Generated: 2026-05-18 06:00Z
Researcher-agent: claude-sonnet-4-6

## Context and Constraints

**Current single-model baseline:** val_abupt 6.126%
**Primary goal:** Break the τz/τx [1.44, 1.55] band attractor AND beat val_abupt 6.126%
**Architecture:** Transolver (S=128 slices, H=4 heads, hidden=512, L=5 layers, DDP-8, Lion lr=9e-5)
**Training budget:** 18h, cosine LR, 1-ep warmup

### Wave 30 Mechanism Map (what is ruled out)

**DEAD (10 confirmed mean-shift fades):**
- Loss-shape / per-vertex: H10b, H11b, H12, H16, H16b, H20, H22 (fades EP2-3)
- Mean Teacher EMA: H23 (KILL 21.36% EP3)
- Area-weighted MSE: H18 (outlier — test τz/τx 1.418 but misses val/SP)
- Per-component heads: H21 (DEAD — capacity not bottleneck)
- Aux local-grad (ALGP): H25 (KILL EP3 FADE — objective-disconnected)
- Per-component rel-L2 loss (PRLP): H27 (KILL EP3 — per-car normalisation INVERSELY re-weights gradient)
- DIFFATTN subtractive SDPA: H32 (KILL EP1 — destroys slice-token magnitude 0.31×, vol_p_mae ~31-36)
- SAM optimizer: H28 (KILL EP2 — band attractor is GEOMETRICALLY FLAT, not a sharpness trap)

**SURVIVING:**
- Variance/spread attack: H26 NPCA (std(τz/τx) 0.092→0.228, 13/34 cars outside band at EP3 — MECHANISM GATE PASSED)
- Loss-reshape absolute accuracy: H29 SSFL (EP7 val_abupt 6.4349% = Wave 30 fleet-low, achieves accuracy WITHIN band)

**IN-FLIGHT:**
- H31 WALLDIST: volume pathway boost via wall-distance feature
- H30 V2S xattn: volume-to-surface cross-attention
- H33 SLICEPE: positional encoding on slice queries
- H34 OUTHEAD: per-channel auxiliary MLP output heads

### Critical Design Rules (enforced by H32 closure)
- ANY modification multiplying slice-token magnitude by <0.5× at init will break the volume pathway
- Subtractive SDPA at init → 0.31× baseline magnitude → vol_p_mae ~31-36 vs normal 5-10
- Additive or multiplicative modifications are safe; subtractive architectural changes are forbidden without compensating residual

### Wave 31 Design Principles
1. Bias toward variance/spread attacks over mean-shift attacks
2. Combine surviving mechanisms (H26 NPCA + H29 SSFL stacking)
3. Frequency-domain and physics-informed losses as next major axis
4. Anchor-conditioned slice queries (DAB-DETR/SpiderSolver line)
5. No magnitude-destructive architectural changes
6. Output-head explorations only if H34 OUTHEAD lands signal

---

## H35: NPCA + SSFL Stack (mechanism combination)

**Slug:** `npca-ssfl-stack`

**Hypothesis:** Local-frame coordinate input (H26 NPCA) and spectral surface-field loss (H29 SSFL) target orthogonal failure modes — the first breaks the τz/τx attractor via variance/spread, the second improves absolute accuracy within whatever spread is achieved. Combining both in a single run should yield a model that both escapes the attractor AND achieves lower val_abupt than either mechanism alone.

**Mechanism class:** Variance/spread + loss-reshape (orthogonal combination)

**Implementation:** Add the SSFL spectral loss term (λ_ssfl from H29, applied to surface fields) on top of the H26 NPCA branch. Both modifications are in separate code paths (coordinate preprocessing vs. loss computation) and require no architectural changes. Preserve H26's local-frame rotation (normal-aligned coordinate basis) and add H29's spectral harmonic decomposition loss.

**LOC estimate:** ~40 LOC — concat NPCA coordinate transform from H26 branch + spectral loss term from H29 branch. Both are already implemented in their respective branches.

**Risk class:** Low. Both mechanisms are individually validated. The only risk is interaction: if SSFL's gradient signal conflicts with NPCA's coordinate-induced variance signal, the combination could average toward band center. The EP3 gate detects this: if std(τz/τx) drops below 0.15, the SSFL loss is dominating and suppressing variance. In that case, reduce λ_ssfl by 0.5× and recheck at EP5.

**EP3 gate (concrete falsifiable thresholds):**
- std(τz/τx) per-car ≥ 0.15 (gate: variance mechanism survives stacking)
- val_abupt ≤ 6.50% (gate: SSFL not hurting convergence)
- val_vol_p_mae ≤ 6.0 (gate: volume pathway not degraded)
- KILL if: std(τz/τx) < 0.10 AND val_abupt > 6.60% (both mechanisms suppressed each other)
- CONTINUE if: std(τz/τx) ≥ 0.15 AND val_abupt ≤ 6.50% (promising combination)
- SUCCESS at EP8+ if: val_abupt < 6.126% AND std(τz/τx) ≥ 0.15

**Why mechanistically distinct from 14+ closures:**
- Not a mean-shift attack (not targeting τz/τx ratio mean)
- Not a loss-shape-only change (previous loss changes H10b-H22 all failed because mean-shift suppresses variance)
- Combines two independently validated mechanisms that address DIFFERENT parts of the failure mode
- H27 PRLP is the closest comparison — it was a loss-space change that inversely re-weighted gradients. SSFL is a frequency-domain reshaping that adds high-frequency signal rather than re-weighting per-car norms. Distinct mechanism, distinct failure mode.

---

## H36: Anchor-Conditioned Slice Queries (DAB-DETR/SpiderSolver line)

**Slug:** `anchor-slice-queries`

**Hypothesis:** Transolver's current slice queries are learned but position-unaware — they pool surface tokens into S=128 context slices without any spatial anchor. Injecting learned 3D anchor positions into each slice query (as in DAB-DETR's anchor-modulated cross-attention and SpiderSolver's anchor-node design) will force slices to specialize by spatial region, breaking the translational invariance that allows the model to learn a single τz/τx ratio regardless of car geometry.

**Mechanism class:** Variance/spread via spatial query specialization

**Implementation:**
1. Add a learned anchor matrix A ∈ R^{S×3} (S=128 anchors in normalized [0,1]^3 car space)
2. At each slice-attention layer, modulate slice query q_s by positional encoding PE(A_s): q_s' = q_s + MLP_anchor(PE(A_s)) where PE is the existing STRING-sep σ-ladder encoding
3. Anchors are shared across the batch but updated via gradient — they learn to spread over car geometry
4. NO change to key/value computation — only query modulation, so magnitude is preserved (H32 safe)

**LOC estimate:** ~60 LOC. Anchor matrix init (uniform grid over [0,1]^3), PE lookup, MLP_anchor (2-layer, hidden=64), query addition. All additive — no subtractive paths.

**Risk class:** Medium. Anchor positions must be in the same coordinate frame as surface_x. If surface_x is in raw CFD space (not normalized), anchors in [0,1]^3 will mismatch. Check coordinate ranges from data. Also, if MLP_anchor is too large, it will dominate q_s and destroy the learned slice structure. Initialize MLP_anchor output at near-zero (final layer zero-init).

**EP3 gate:**
- std(τz/τx) per-car ≥ 0.15 (anchor specialization must show spatial spread)
- Anchor diversity metric: mean pairwise anchor distance ≥ 0.10 (anchors not collapsed)
- val_abupt ≤ 6.60%
- KILL if: all anchors collapse to mean car position (diversity < 0.05) AND std(τz/τx) < 0.10
- SUCCESS signal: std(τz/τx) ≥ 0.20 AND val_abupt < 6.30%

**Why mechanistically distinct:**
- The 14 closures include no spatial anchor mechanism — all slice queries are positionally anonymous
- H21 (per-component heads) proved decoder capacity is not the bottleneck; this attacks query specialization, not output capacity
- H33 SLICEPE (in-flight) adds positional encoding to slices but with fixed σ=0.02 and no learned anchors — this is strictly more expressive: learned anchors + PE modulation vs. fixed PE only
- SpiderSolver (ICLR 2025) uses this exact principle on 2D PDEs; DrivAerML 3D automotive geometry is the natural extension

---

## H37: Wavelet Surface Loss (frequency-domain variance attack)

**Slug:** `wavelet-surface-loss`

**Hypothesis:** The τz/τx attractor is a low-frequency failure — the model learns a spatially smooth ratio because MSE loss heavily weights low-frequency global patterns. Adding a wavelet decomposition loss that explicitly penalizes high-frequency reconstruction error on τx and τz surfaces independently will force the model to resolve local geometry-driven variations, increasing per-car variance in τz/τx without attacking the mean.

**Mechanism class:** Variance/spread via frequency-domain loss

**Implementation:**
1. Apply 1D Haar wavelet decomposition along the sorted surface mesh (sorted by surface_x position or by angular coordinate on car surface)
2. Compute L2 loss on detail coefficients (high-frequency bands d1, d2, d3) separately for τx and τz channels
3. Loss = MSE_base + λ_wvl * (L2(d_τx) + L2(d_τz)) where λ_wvl ∈ {0.1, 0.3, 1.0}
4. Surface points must be consistently ordered across the batch — use sorted z-coordinate (lengthwise car axis) as ordering key
5. Haar DWT is differentiable and adds < 5ms per step

**LOC estimate:** ~50 LOC. Haar DWT kernel (or import pywt and wrap), ordering logic, loss term addition. No architectural changes.

**Risk class:** Medium. Surface point ordering must be consistent across batches — if ordering is noisy, wavelet coefficients will be meaningless. Validate by checking that two runs of the same car produce the same wavelet spectrum. Also, λ_wvl must be tuned: too high → τx/τz dominate and surface pressure degrades.

**EP3 gate:**
- std(τz/τx) per-car ≥ 0.15
- val_tau_z_rel_l2 ≤ 5.0% (wavelet loss must not degrade absolute τz accuracy)
- val_SP_rel_l2 ≤ 5.0% (surface pressure must not suffer from τ-biased gradients)
- KILL if: val_SP_rel_l2 > 6.5% (wavelet loss stealing gradient from pressure)
- Try λ_wvl ∈ {0.1, 0.3} before 1.0 — start conservative

**Why mechanistically distinct:**
- H29 SSFL uses spectral decomposition via spherical harmonics on the global surface field; this uses wavelet decomposition on INDIVIDUAL channels (τx, τz independently) targeting per-channel spatial frequency
- H10b-H22 all used global reweighting (per-vertex, per-component norms, area weights) — none decomposed the signal in frequency space
- H27 PRLP failed because per-car normalisation re-weighted ACROSS cars; wavelet loss is within-car frequency decomposition, not cross-car normalisation

---

## H38: Turbulent Boundary Layer Physics Loss (physics-informed)

**Slug:** `tbl-physics-loss`

**Hypothesis:** For attached turbulent flow over an automotive body, τz (streamwise wall shear) and the streamwise pressure gradient ∂p/∂z are coupled via the TBL momentum equation: τ_w ≈ μ ∂u/∂y ∝ -(∂p/∂z). Adding a physics-informed loss that penalizes violations of this zero-pressure-gradient TBL relationship on the hood and roof (regions of attached flow) will force τz predictions to track local pressure gradient predictions, breaking the constant-ratio attractor by making τz geometry-specific.

**Mechanism class:** Variance/spread via physics-informed constraint

**Implementation:**
1. Identify attached-flow surface regions by SDF proximity and surface normal orientation: select surface points where |n·z_hat| < 0.3 (near-horizontal surfaces: hood, roof, trunk)
2. Compute discrete streamwise pressure gradient: Δp_z = (p_{i+z} - p_{i-z}) / (2Δz) using k-NN on predicted surface pressure
3. Add loss: L_tbl = λ_tbl * mean( (τz_pred + C * Δp_z_pred)^2 ) over attached-flow points
4. C is a learned scalar (initialized at 1.0, updated by gradient) that absorbs Reynolds number and viscosity
5. λ_tbl ∈ {0.05, 0.1, 0.2}

**LOC estimate:** ~80 LOC. Surface region selector (mask by normal), k-NN gradient estimator (~20 LOC), TBL loss term, learned C scalar. No architectural changes.

**Risk class:** Medium-high. The TBL coupling only holds in attached flow — separated regions (underbody, wheel arches) will corrupt the signal if not masked correctly. Must verify that the attached-flow mask excludes >90% of separated-flow surface area. Also, k-NN gradient on unstructured mesh is noisy — need at least k=8 neighbors and weighted averaging. This is more complex than H37 but addresses a different part of the physics.

**EP3 gate:**
- std(τz/τx) per-car ≥ 0.15
- TBL constraint residual on attached surfaces ≤ 0.5× baseline MSE (physics coupling tightening)
- val_abupt ≤ 6.60%
- KILL if: val_SP degrades > 10% (physics loss capturing spurious coupling with pressure)

**Why mechanistically distinct:**
- No prior hypothesis has used physics-informed loss. The 14 closures are all data-driven. This directly encodes domain knowledge about turbulent boundary layer momentum balance.
- H25 ALGP added local gradient auxiliary loss for geometry; this adds physical constraint on the relationship between predicted fields (τz vs. ∂p/∂z), which is a different mechanism: physics coupling vs. geometry signal
- The AB-UPT reference shows τz and τx targets are achievable (3.63%, 5.35%) — the gap suggests the model is not exploiting available physics

---

## H39: Mixture-of-Experts Slice Router (MoE)

**Slug:** `moe-slice-router`

**Hypothesis:** A single shared Transolver backbone assigns all S=128 slices to the same MLP experts in the FFN layers, which forces all car-region representations through the same transformation. Replacing the FFN in the final 2 Transolver layers with a sparse MoE (E=8 experts, top-K=2 routing) will allow the model to develop specialized computational pathways for different flow regions (attached flow, separated flow, underbody), increasing per-slice diversity and thus per-car τz/τx variance.

**Mechanism class:** Variance/spread via capacity specialization

**Implementation:**
1. Replace the FFN in Transolver layers 4 and 5 (the final 2 of 5 layers) with MoE FFN
2. E=8 experts, each is a standard 2-layer MLP (same dim as original FFN), top-K=2 routing
3. Router: linear(hidden=512) → softmax → top-2 selection with auxiliary load-balancing loss
4. Load-balancing coefficient λ_lb = 0.01 (standard from Switch Transformer)
5. At init, all experts are identical copies of the original FFN weights — guarantees no magnitude change at step 0 (H32 safe)
6. Total parameter increase: ~4× FFN params for last 2 layers ≈ 2.5M extra params (negligible vs. 35M model)

**LOC estimate:** ~120 LOC. MoE FFN module, router, load-balancing loss, weight init from existing FFN, layer swap. Several open-source MoE implementations are < 100 LOC (e.g., Mixtral-style sparse router).

**Risk class:** Medium. Expert collapse is a known failure mode: all tokens route to 1-2 experts, destroying specialization. Load-balancing loss must be active from step 0. Monitor expert utilization variance in W&B — if top-1 expert receives > 60% of tokens by EP2, increase λ_lb to 0.05. Also, MoE training can be slower due to dynamic routing — verify throughput is within 10% of baseline before committing to full run.

**EP3 gate:**
- Expert utilization entropy ≥ 1.5 nats (experts are not collapsed)
- std(τz/τx) per-car ≥ 0.12 (some variance escape vs. 0.092 baseline)
- val_abupt ≤ 6.50%
- KILL if: expert entropy < 0.5 nats (complete collapse) at EP2

**Why mechanistically distinct:**
- H21 (per-component heads) added capacity in the output layer — this adds capacity in the intermediate processing layer with dynamic routing
- None of the 14 closures used MoE or any expert-mixture mechanism
- The theoretical motivation is specific: band attractor may be caused by a single shared FFN mapping all car geometries to the same feature manifold; MoE breaks this by learning geometry-conditional transformations

---

## H40: Curvature-Conditioned Volume Decoder (geometry-aware vol pathway)

**Slug:** `curv-vol-decoder`

**Hypothesis:** The volume pressure decoder currently receives only (x,y,z,SDF) as geometric input. The SDF alone cannot distinguish between high-curvature regions (wheel arches, mirror housings) where strong pressure gradients exist and low-curvature flat regions. Augmenting volume tokens with mean curvature κ (computed from the SDF Hessian) will give the decoder geometry-aware conditioning, improving vol_p predictions in high-gradient regions and potentially reducing the τz/τx attractor indirectly via the surface-vol cross-attention pathway.

**Mechanism class:** Volume pathway boost + geometric conditioning

**Implementation:**
1. Compute mean curvature κ = (1/2) * tr(∇²SDF) / |∇SDF| from the SDF field using finite differences on the volume grid
2. Add κ as a 5th input feature to volume_x: volume_x ∈ R^{N_v×5} (x,y,z,SDF,κ)
3. Re-initialize the first linear layer of the volume branch to accept 5 features (the extra column initialized at 0 to preserve initial output magnitudes)
4. κ should be clipped to [-κ_max, κ_max] with κ_max = 5th percentile of |κ| distribution to prevent outlier contamination

**LOC estimate:** ~50 LOC. Curvature computation (SDF Hessian via torch.autograd or finite differences), feature concat, layer reinit. The 0-init column guarantee is critical for H32 safety.

**Risk class:** Low-medium. The main risk is κ computation from discrete SDF — on coarse volume grids, the Hessian estimate is noisy. Validate by checking that computed κ values are spatially smooth (variance on flat regions < 0.1) before training. Also, H31 WALLDIST is in-flight and adds a different geometric feature (wall distance) — the two are complementary but should not be run simultaneously to avoid confounding.

**EP3 gate:**
- val_vol_p_mae ≤ 3.8 (must improve on current fleet-best vol pathway)
- val_abupt ≤ 6.40%
- Curvature gradient magnitude in high-κ regions ≥ 5× flat regions (conditioning is active)
- KILL if: val_vol_p_mae > 5.0 (curvature feature confusing volume decoder)

**Why mechanistically distinct:**
- H31 WALLDIST adds wall-distance as a proxy for flow regime (near-wall vs. freestream); this adds direct geometric curvature which is distinct — curvature predicts pressure gradient magnitude while wall-distance predicts turbulence regime
- H30 V2S xattn adds volume-to-surface cross-attention for information flow; this improves the input representation geometry, not the attention architecture
- None of the 14 closures targeted the volume decoder's geometric input features beyond SDF

---

## H41: Per-Car Normal-Aligned Local Frame + Increased Slice Capacity (NPCA-192)

**Slug:** `npca-192-slices`

**Hypothesis:** H26 NPCA proved that local-frame coordinate input breaks the attractor (std 0.228, 13/34 cars outside band at EP3). The residual 21/34 cars inside the band suggests that slice capacity (S=128) is limiting — each slice must represent more spatial regions than it can disambiguate. Increasing slice count from 128 to 192 while keeping the H26 NPCA coordinate input will give the variance mechanism more representational headroom to map distinct geometries to distinct slices.

**Mechanism class:** Variance/spread + capacity (token budget)

**Implementation:**
1. Start from H26 NPCA branch exactly (local-frame coordinate input, all else baseline)
2. Change --num_slices 128 → 192
3. Memory: 192 slices × 512 hidden × 5 layers = ~50% more slice-level activations. At DDP-8, check OOM. If needed, reduce batch size from default by 1 per GPU (usually 4→3).
4. Initialize the extra 64 slice queries as copies of the median 64 existing slice queries (to prevent cold start) or as random perturbations of existing queries

**LOC estimate:** ~5 LOC (single hyperparameter change + batch size guard). The NPCA coordinate preprocessing is already in the H26 branch.

**Risk class:** Low. This is a direct capacity extension of a validated mechanism. The main risk is OOM — needs batch size check. Secondary risk: with more slices, each slice represents fewer surface tokens, potentially causing attention sparsity in the low-density underbody region.

**EP3 gate:**
- std(τz/τx) per-car ≥ 0.25 (more slices should give HIGHER variance than H26's 0.228 at EP3)
- Cars outside band: ≥ 15/34 (vs. H26's 13/34 at EP3)
- val_abupt ≤ 6.60% (same ceiling as H26 — 192 slices should not hurt convergence)
- KILL if: std(τz/τx) ≤ 0.20 at EP3 (slice increase not helping vs. H26 baseline at same epoch)

**Why mechanistically distinct:**
- This is a confirmed-mechanism scaling experiment, not a new hypothesis exploration
- The specific falsifying test (std(τz/τx) ≤ 0.20 at EP3 when H26 already reached 0.228) is strong: if 192 slices don't improve over 128 slices in the same mechanism, it means slice count is not the limiting factor for variance, pointing toward a different diagnosis (anchoring, query diversity, or head count)
- H33 SLICEPE (in-flight) tests PE on slices at S=128; this tests capacity at S=192 with NPCA — these are orthogonal axes

---

## H42: Warm-Start from H26 NPCA Checkpoint (late-epoch variance consolidation)

**Slug:** `npca-warmstart`

**Hypothesis:** H26 NPCA achieves strong variance signal at EP3 (std 0.228) but val_abupt 6.91% is still above baseline. The model has broken the attractor geometrically but has not yet translated that geometric spread into accuracy gains. Warm-starting a new run from the H26 EP6 or EP8 checkpoint with the SSFL spectral loss added (λ_ssfl=0.3) and a reduced LR (lr=3e-5) will accelerate convergence from the escaped attractor state toward lower val_abupt without the model re-converging into the band from a random init.

**Mechanism class:** Variance/spread consolidation + loss-reshape (checkpoint warm-start)

**Implementation:**
1. Obtain H26 NPCA checkpoint at the epoch where std(τz/τx) is highest AND val_abupt is still descending (likely EP6-EP8 based on EP3 trajectory)
2. Load checkpoint with --resume_from path/to/npca_epN.pt
3. Set lr=3e-5 (1/3× training LR — fine-tuning regime)
4. Set lr_warmup_epochs=0 (no warmup — already warmed up)
5. Add SSFL spectral loss with λ_ssfl=0.3
6. Train for remaining budget (up to 18h from warm-start launch)

**LOC estimate:** ~20 LOC. Checkpoint loading is already supported (--resume_from). LR and loss flags are existing CLI arguments. The only new code is SSFL loss injection (if not already merged from H29).

**Risk class:** Low-medium. Main risk: if the H26 checkpoint is already descending val_abupt rapidly without SSFL, adding SSFL mid-training may interfere with the trajectory. Diagnostic: run EP1 of warm-start and check that val_abupt is lower than the loaded checkpoint's val_abupt within the first 500 steps. If val_abupt rises, SSFL is conflicting — reduce λ_ssfl to 0.1.

**EP3 gate (from warm-start epoch 0):**
- val_abupt ≤ 6.40% by warm-start EP3 (must be converging toward target faster than cold-start H26)
- std(τz/τx) per-car ≥ 0.18 (variance must be preserved — not re-collapsing into band)
- KILL if: std(τz/τx) < 0.10 at warm-start EP2 (SSFL re-collapsed the attractor)
- SUCCESS: val_abupt < 6.126% with std(τz/τx) ≥ 0.15

**Why mechanistically distinct:**
- This is not re-running H26 or H29 — it tests whether the escaped-attractor state (H26's high-variance checkpoint) can be leveraged as a starting point for the SSFL accuracy mechanism
- No prior experiment has used warm-starting from a partially-converged mechanism-gate checkpoint
- The hypothesis is falsifiable: if warm-starting from an escaped attractor and then adding SSFL does NOT improve over cold-start SSFL, it means the band escape and accuracy optimization are truly separable and must be achieved simultaneously, not sequentially

---

## H43: Head-Doubled Slice Attention (H=4→8 heads)

**Slug:** `heads-doubled`

**Hypothesis:** Transolver uses H=4 attention heads across S=128 slices. With hidden=512, each head has dimension 128 — sufficient for generic patterns but potentially insufficient for the 5 distinct surface fields (p_s, τx, τy, τz, p_v). Doubling to H=8 heads (dim=64 per head) increases multi-scale pattern capture in slice attention and may allow some heads to specialize on τz vs. τx independently, reducing the cross-field coupling that sustains the attractor.

**Mechanism class:** Variance/spread via attention capacity

**Implementation:**
1. Change --num_heads 4 → 8 (single flag change in Transolver config)
2. All head projection matrices at init: existing 4-head weights copied to heads 1-4; heads 5-8 initialized as random perturbations (std=0.01) of the mean head projection — near-identity at init, no magnitude change
3. Memory: head count doubling adds negligible parameter count (only QKV projections scale with H, not the FFN)
4. Verify that hidden=512 divides evenly by 8 (it does: 512/8=64)

**LOC estimate:** ~5 LOC (flag change + head init logic). Zero architectural change to model structure.

**Risk class:** Low. Head count is a well-understood hyperparameter. Main risk: with head_dim=64, attention patterns may be too narrow to capture long-range surface correlations. H=8, dim=64 is standard in BERT-base — well within normal operating range.

**EP3 gate:**
- std(τz/τx) per-car ≥ 0.12 (head specialization must show some variance)
- val_abupt ≤ 6.40%
- KILL if: val_abupt > 6.70% at EP3 (heads increase hurting convergence)

**Why mechanistically distinct:**
- H21 (per-component heads) added separate output heads for each field — capacity at decode time
- This adds attention heads at the slice representation level — capacity at encode/aggregate time
- These are different mechanisms: H21's failure means "separate output pathways don't help"; H43's hypothesis is "more diverse attention patterns in the slice aggregation step help"
- H33 SLICEPE (in-flight) adds positional encoding to slice queries; this adds head diversity — orthogonal axes

---

## H44: Random Yaw Augmentation in Normal-Aligned Frame (data augmentation)

**Slug:** `yaw-augmentation-npca`

**Hypothesis:** All DrivAerML cars are yaw=0 (symmetric alignment along the z-axis). The model learns τz/τx patterns from 400 symmetric cases and may be exploiting this symmetry assumption to short-circuit the geometry-specific computation. Applying random yaw rotations (θ ∈ [-15°, +15°]) during training in the normal-aligned NPCA coordinate frame (so the local frame rotates with the car) will disrupt the model's ability to memorize the yaw=0 symmetry and force geometry-specific τz/τx predictions.

**Mechanism class:** Variance/spread via data augmentation (symmetry breaking)

**Implementation:**
1. Apply random yaw rotation Ry(θ) to all input coordinates (surface_x[:,:3] and volume_x[:,:3]) at each training step, with θ ~ U[-15°, +15°]
2. Apply the SAME rotation to all surface normal vectors (surface_x[:,3:6]): n' = Ry(θ) · n
3. Do NOT rotate the target fields — targets are defined in the fixed CFD frame. Rotate the INPUTS only, forcing the model to learn rotation-equivariant τz/τx representations.
4. At test time: apply no augmentation (standard θ=0 inference)
5. If combined with H26 NPCA: apply yaw rotation BEFORE the normal-aligned frame computation

**LOC estimate:** ~40 LOC. Rotation matrix (3×3), application to point cloud and normals, augmentation flag. No architectural changes.

**Risk class:** Medium. This is a strong regularizer — too large a rotation range may degrade absolute accuracy if the model cannot learn rotation equivariance from 400 training cases. Start with [-5°, +5°] and monitor val_abupt at EP2. If val_abupt > 6.50% at EP2 with [-15°, +15°], roll back to [-5°, +5°]. Also, yaw rotation changes the SDF field — the volume SDF must also be rotated consistently with the surface.

**EP3 gate:**
- std(τz/τx) per-car ≥ 0.15 (yaw augmentation must increase per-car variance — the geometric signal of each car's asymmetry becomes more discriminative)
- val_abupt ≤ 6.50%
- KILL if: val_abupt > 6.80% at EP2 (augmentation too aggressive — overly destroying the training signal)
- Try θ_max=5° before 15°

**Why mechanistically distinct:**
- PR #1107 (yaw rotation augmentation) is in the "Never ran" list — this idea was proposed but never executed
- None of the 14 closures used data augmentation — all were loss or architecture changes
- Yaw augmentation attacks symmetry-exploitation, not gradient weighting or architecture capacity
- The combination with NPCA (local frame) is specifically designed: if the model learns in a geometry-relative frame, random yaw forces it to generalize the local-frame representation across orientations

---

## H45: GeoTransolver: Normal-Aligned Slice Groups + Anchor PE (combined geometry)

**Slug:** `geo-transolver`

**Hypothesis:** GeoTransolver (Chen et al., 2024) achieves state-of-the-art PDE surrogate results by combining geometric slice grouping (grouping tokens by local normal direction) with anchor-conditioned slice queries. The DrivAerML Transolver uses learnable slice queries without geometric constraints. Implementing a GeoTransolver-style slice grouping where slices 0-63 are assigned to the upper body (n·y > 0) and slices 64-127 to the lower body/underbody (n·y ≤ 0), with the NPCA local frame coordinates, will force geometric specialization at the slice allocation level rather than leaving it to be learned from scratch.

**Mechanism class:** Variance/spread via explicit geometric slice specialization

**Implementation:**
1. At each Transolver slice-attention layer, compute a soft-assignment score for each surface token to each slice: a_{is} = softmax(q_s · f_i / sqrt(d)) where f_i is the token embedding
2. Add a geometric prior: a_{is} += λ_geo * exp(-||n_i - d_s||^2 / 2σ²) where d_s is the assigned normal direction for slice s and n_i is surface point normal
3. Slice normal directions d_s: first 64 slices assigned to upper-body normal directions (evenly sampled from n·y > 0 hemisphere), last 64 to lower-body/underbody
4. λ_geo = 0.1 (soft constraint — allows deviation from rigid assignment), σ = 0.5 (normal direction bandwidth)
5. This is additive to attention logits — no magnitude destruction at init

**LOC estimate:** ~70 LOC. Slice normal direction precomputation, geometric prior computation, addition to attention logits. Soft assignment preserves gradient flow through all tokens.

**Risk class:** Medium. The upper/lower split is a strong geometric inductive bias that may fail for complex underbody geometries (wheel arches, exhaust tunnels) where normals are highly variable. If the geometric prior overwhelms the learned attention (λ_geo too high), slices will not be able to attend to geometrically-distant but fluid-dynamically-coupled regions (e.g., roof → underbody through wake pressure). Start with λ_geo=0.05 and monitor attention entropy.

**EP3 gate:**
- Upper-body slice entropy ≥ 1.0 (slices are not collapsed to single regions)
- std(τz/τx) per-car ≥ 0.15
- val_abupt ≤ 6.60%
- KILL if: attention entropy < 0.3 (geometric prior over-constraining slices to rigid assignment)

**Why mechanistically distinct:**
- Distinct from H36 (ANCHOR-SLICE-QUERIES): H36 uses learned spatial anchor positions; H45 uses a fixed geometric prior (surface normals) that encodes domain knowledge about upper/lower body flow separation
- GeoTransolver (2024) applies this on fluid simulation meshes — DrivAerML automotive surfaces are the natural analogue with the additional challenge of complex underbody geometry
- H33 SLICEPE (in-flight) uses fixed σ=0.02 positional encoding — this uses dynamic normal-direction geometric grouping, which is conceptually and implementationally different

---

## Priority Ordering and Assignment Recommendation

**Immediate (4 students available now):**
1. **H35 NPCA-SSFL-STACK** (lowest risk, highest expected return — combine two validated mechanisms)
2. **H42 NPCA-WARMSTART** (low risk, tests checkpoint-continuation hypothesis cleanly)
3. **H41 NPCA-192-SLICES** (very low LOC, direct scaling of validated mechanism)
4. **H44 YAW-AUGMENTATION-NPCA** (novel axis, never tried, medium risk)

**Next wave (2-3 more students):**
5. **H36 ANCHOR-SLICE-QUERIES** (medium risk, strong theoretical basis from SpiderSolver/DAB-DETR)
6. **H43 HEADS-DOUBLED** (very low LOC, orthogonal to H33 SLICEPE in-flight)
7. **H37 WAVELET-SURFACE-LOSS** (medium risk, clean frequency-domain mechanism)

**Later wave (after H36/H37 results):**
8. **H39 MOE-SLICE-ROUTER** (medium LOC, most novel mechanism, wait for anchor/wavelet results first)
9. **H40 CURV-VOL-DECODER** (wait for H31 WALLDIST results to avoid confounding)
10. **H38 TBL-PHYSICS-LOSS** (highest LOC/complexity — reserve for plateau if H35-H44 all fail)
11. **H45 GEO-TRANSOLVER** (highest conceptual risk, run after H36 anchor-query results)

---

## Decision Tree

```
H35 NPCA-SSFL-STACK (EP3)
├── std(τz/τx) ≥ 0.15 AND val_abupt ≤ 6.50%:
│   → CONTINUE TO EP8+; target val_abupt < 6.126% 
│   → If succeeds: MERGE, then assign H41 (NPCA-192) + H42 (WARMSTART) from this checkpoint
│   → If reaches EP8 at 6.10-6.12%: launch H41-192-slices for final push
└── std(τz/τx) < 0.10 AND val_abupt > 6.60%:
    → SSFL suppressed NPCA variance — KILL
    → Lesson: SSFL loss frequency and NPCA variance mechanism conflict at λ_ssfl=0.3
    → Next: Try H37 WAVELET-SURFACE-LOSS (different frequency attack, τ-channel specific)

H42 NPCA-WARMSTART (warm-start EP3)
├── std(τz/τx) ≥ 0.18 AND val_abupt ≤ 6.40%:
│   → Band escape preserved + accuracy improving; CONTINUE
│   → This is the fastest path to val_abupt < 6.126% if H26 converges to ~6.5% standalone
└── std(τz/τx) < 0.10:
    → SSFL re-collapsed attractor; KILL
    → Lesson: Once in escaped state, SSFL loss is too aggressive for attractor stability
    → Next: Try warm-start with λ_ssfl=0.1 (gentler) OR warm-start with yaw-augmentation only

H41 NPCA-192-SLICES (EP3)
├── std(τz/τx) ≥ 0.25 AND cars-outside-band ≥ 15/34:
│   → More slices = more variance; CONTINUE
│   → This confirms slice capacity was the limiting factor for H26
└── std(τz/τx) ≤ 0.20:
    → Slice count not the limiting factor; KILL
    → Lesson: The 21/34 cars inside band at H26 EP3 are not limited by slice capacity
    → Next: Assign H36 ANCHOR-SLICE-QUERIES (spatial anchoring may be the missing ingredient)

H44 YAW-AUGMENTATION-NPCA (EP2-3)
├── std(τz/τx) ≥ 0.15 AND val_abupt ≤ 6.50%:
│   → Symmetry-breaking is contributing; CONTINUE
└── val_abupt > 6.80%:
    → Augmentation too aggressive; try θ_max=5° in follow-up
    → Lesson: 400-car dataset may be insufficient for rotation equivariance learning at 15°
```

---

## Taste Rubric

| Hypothesis | Mode | Mechanistic Grounding | Research-State Value | Execution Value | Notes |
|---|---|---|---|---|---|
| H35 NPCA-SSFL-STACK | Frontier refinement | 4 | 4 | 4 | Both mechanisms validated independently; combination is cleanest possible next step |
| H42 NPCA-WARMSTART | Diagnostic | 3 | 4 | 4 | Directly tests sequential vs. simultaneous mechanism combination — high falsifiability |
| H41 NPCA-192-SLICES | Diagnostic | 4 | 4 | 4 | Single-LOC capacity scaling of validated mechanism; tells us if slice count limits H26 |
| H44 YAW-AUGMENTATION | Tier shift (data aug) | 3 | 3 | 3 | Novel axis (never tried), addresses symmetry bias, but unknown yield on 400 cases |
| H36 ANCHOR-SLICE-QUERIES | Tier shift (architecture) | 3 | 3 | 3 | Strong theory from SpiderSolver, but coordinate frame matching adds risk |
| H43 HEADS-DOUBLED | Frontier refinement | 2 | 3 | 4 | Very cheap, but H21 negative result weakens the "capacity at output" story; heads are mid-stream |
| H37 WAVELET-SURFACE-LOSS | Frontier refinement | 3 | 3 | 3 | Clean frequency-domain mechanism, distinct from SSFL, but surface ordering adds complexity |
| H39 MOE-SLICE-ROUTER | Tier shift (architecture) | 2 | 3 | 2 | Strong theory but expert collapse risk + higher implementation cost; defer until anchoring tested |
| H40 CURV-VOL-DECODER | Frontier refinement | 3 | 3 | 3 | Clean volume pathway improvement; wait for H31 WALLDIST to avoid confounding |
| H38 TBL-PHYSICS-LOSS | Tier shift (physics) | 2 | 3 | 2 | Highest complexity; reserve for plateau if simpler mechanisms exhausted |
| H45 GEO-TRANSOLVER | Tier shift (architecture) | 2 | 3 | 2 | Most speculative; run after H36 anchor results confirm the spatial-query axis |
