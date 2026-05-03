# Round 33 Research Ideas — 2026-05-03 (yi branch)

## State at idea-generation time

- **Active yi merge bar:** val_abupt 9.032% (PR #517 askeladd Lion lr=1e-4 clip=0.5). Run `brat65z4`.
- **Critical**: PR #490 (frieren STRING-sep learnable PE) MERGED to yi at 15:48 UTC. The `--learnable-pe` flag is now available on yi. PR #517 was launched BEFORE #490 merged, so the merged baseline does NOT use STRING-sep PE. Frieren resumed-from-checkpoint run reached 8.0867% with `--learnable-pe`, but a clean from-scratch combination has not been logged.
- **10 active WIP PRs** — see CURRENT_RESEARCH_STATE.md. Themes covered already: optimizer/LR (Muon, 1-cycle, LLRD, per-step cosine), wall-shear loss reformulations (tangent-frame head, stream-normal weight, loss-side asinh), volume coord transform, slice attention gating, fp32 last-2-epoch.
- **Largest open gaps** (from tay SOTA PR #511 reference, yi-equivalent gaps similar):
  - wall_shear_y: 8.58 vs 3.65 → **2.35×**
  - wall_shear_z: 9.93 vs 3.63 → **2.73×**
  - volume_pressure: 11.87 vs 6.08 → **1.95×**

## Idea 1 — frieren: STRING-sep + Lion lr=1e-4 clip=0.5 from-scratch (anchor)

**Hypothesis.** Combining the merged STRING-sep PE (PR #490) with the merged Lion lr=1e-4 clip=0.5 hyperparameters (PR #517) in a single from-scratch run on the post-merge yi codebase will reach val_abupt ≤ 8.5%, locking in a clean new baseline below the current 9.032% bar.

**Mechanism.** The 8.0867% frieren resumed run proved the combination works, but resume-from-checkpoint is non-canonical. A from-scratch confirmation run is the cheapest, highest-confidence merge available right now. Both ingredients are already on yi; this is an integration test, not a new research direction.

**Code change.** None — yi `train.py` already has both. Simply launch with `--learnable-pe --optimizer lion --lr 1e-4 --grad-clip-norm 0.5 --weight-decay 5e-4 --lr-warmup-epochs 1 --ema-decay 0.999`.

**Sweep plan (1 arm, 4-GPU DDP, full budget).**
- Single arm A: `--learnable-pe`, all other PR #517 baseline hyperparameters intact.
- Optional Arm B: same + `lr_warmup_epochs=0` (matched to frieren resumed run).

**Win criterion.** val_abupt < 9.032 (merge). Stretch: < 8.5%. Test_abupt should drop with it.

## Idea 2 — emma: GradNorm dynamic multi-task balancing

**Hypothesis.** Per-axis static loss weights have been tried (PR #454 frieren, closed) and uncertainty weighting de-emphasized lagging axes (PR #496 tanjiro, closed because mechanism is *aleatoric down-weighting*). The principled UP-weighting algorithm is GradNorm (Chen et al. 2018) — it normalizes per-task gradient magnitudes against their inverse training rates, dynamically increasing weight on tasks that lag (high L_i/L_i_init). Applied to DrivAerML's 5 axes (sp, τ_x, τ_y, τ_z, vol_p), GradNorm should auto-discover that τ_y/τ_z need 2-3× weight and apply it without manual tuning.

**Mechanism.** At each step, compute |∇W_shared L_i| for each task i, target `r_i^α = (L_i(t)/L_i(0))^α / mean_i(...)` for α ∈ [0.5, 1.5]. Weights w_i tracked as `nn.Parameter`s, updated via auxiliary loss `sum_i |G_i - target_G_i|` on backward shared. EXP-MOV-AVG smooth task losses to avoid step noise. Normalize w_i to sum to T (number of tasks).

**Code change.** Add `--gradnorm-alpha <float>` flag and `--gradnorm-init-anchor` (uses initial losses or first 10 steps). Compute per-task |∇W_last_shared L_i| via separate `loss.backward(retain_graph=True)` with task masks; alternatively use the elegant `torch.autograd.grad` per-task path. Set `w_i.requires_grad=True` and update via a dedicated AdamW(lr=1e-3) for the weights. Log per-task `w_i` and `r_i` to W&B every step.

**Sweep plan (4-GPU DDP, 4 arms).**
- A: α=0.0 (control, equal weights — equivalent to baseline)
- B: α=0.5 (mild rebalancing)
- C: α=1.0 (canonical)
- D: α=1.5 (aggressive)
- All on PR #517 stack + `--learnable-pe` (post-#490).

**Win criterion.** Best arm beats val_abupt 9.032 AND τ_y, τ_z components show ≥10% relative improvement vs surface_pressure (asymmetric improvement = mechanism working as intended). If all arms regress, GradNorm is the wrong tool; close.

## Idea 3 — askeladd: streamline-aligned wall-shear target frame

**Hypothesis.** The 2.7× τ_z gap is structural: in a Cartesian frame, τ_y and τ_z are tiny on most surface points (the dominant component is τ_x along streamwise) and the model has near-zero signal/noise on them. Rotating the ws targets per-point into a local streamline frame `(s, n_t1, n_t2)` where `s` is parallel to the surface tangent of the streamline projection and `n_t1, n_t2` span the tangent-plane perpendicular gives the model a representation where the "small" components are physically meaningful — the cross-flow secondary vortices and separation footprint.

**Mechanism.** Use the surface normal `n` (already in `surface_x[..., 3:6]`) and a global freestream direction `e_x = (1, 0, 0)` to construct a per-point local frame: `t1 = normalize(e_x - (e_x·n)n)` (streamwise tangent), `t2 = n × t1` (cross-tangent). Rotate target `(τ_x, τ_y, τ_z)` into `(τ_t1, τ_t2, τ_n)` where τ_n should be ≈0 (ws is tangential by definition). Predict in this frame; rotate back for inverse transform during validation. The ws_n component becomes a free regularizer (its target is 0).

**Code change.** Add `--wallshear-frame {cartesian, streamline}` flag. In dataset/loader, compute `(t1, t2)` per surface point and store as 6 extra channels (the frame basis). In loss, rotate target. In validation, rotate prediction back to Cartesian for AB-UPT metric computation. Both training and inference paths must use the same rotation. Add `train/loss_ws_t1`, `train/loss_ws_t2`, `train/loss_ws_n` (the latter should decay to ≈0).

**Sweep plan (4-GPU DDP, 2 arms).**
- A: streamline frame (Arm B: include ws_n=0 as auxiliary regularizer with weight=0.1).
- Control: cartesian (current). Same hyperparameters: PR #517 + `--learnable-pe`.

**Win criterion.** val_abupt < 9.032 AND τ_y or τ_z (in Cartesian-decoded metric) show >10% relative improvement. Different from PR #312/419 (which used surface tangent vectors as INPUT features); this rotates TARGETS.

## Idea 4 — alphonse: mass-conservation / divergence-free soft penalty on volume head

**Hypothesis.** Volume-pressure has a 1.95× test gap with high-error outliers (test vol_p MAE ~28 vs val ~8). The volume_pressure target is a scalar field whose gradient should be physically consistent with a divergence-free velocity field (incompressible RANS). We don't have velocity targets, but we can add a soft penalty that the *predicted pressure gradient* obeys a known structure — specifically, that ∇²p ≈ 0 in regions far from the body (Laplacian penalty on volume points where SDF > threshold). This is a physics-informed regularizer that should suppress test-time outliers because the spurious extreme predictions in heavy-tail outlier vehicles violate Laplace.

**Mechanism.** Sample K=512 per-batch volume points where SDF > 0.5m (far-field), compute ∇²p = ∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z² via autograd (predict p as function of input coordinates with `coords.requires_grad_(True)`, then second-derivative). Penalty: `lambda * mean(∇²p_far²)`. λ swept.

**Code change.** Add `--laplace-penalty-lambda <float>`, `--laplace-far-sdf-threshold <float>`, `--laplace-num-points <int>`. In the volume head forward, when laplace-penalty > 0, set `coords.requires_grad_(True)` for the K sampled far-field points; compute `grad(p, coords)` (first), then `grad(grad_p[:,i], coords)[:,i]` for i ∈ {x,y,z} and sum (Laplacian). Penalty added to total loss. Log `train/loss_laplace_far` and the actual mean |∇²p|.

**Sweep plan (4-GPU DDP, 3 arms).**
- A: λ=0 (control) — baseline post-#490 with `--learnable-pe`
- B: λ=1e-3
- C: λ=1e-2
- All other hyperparameters PR #517.

**Win criterion.** val_abupt < 9.032 AND volume_pressure component improves ≥5% relative. test vol_p outlier reduction (max-residual decrease) is a strong secondary signal.

## Idea 5 — haku: principal surface curvatures (κ_1, κ_2) as input features

**Hypothesis.** The current surface input features don't include any second-order geometric information. The cross-flow shear regions correspond to high-curvature surface zones (A-pillar, side-mirror wake foot, wheel-arch lip) where boundary-layer separation strongly modulates τ_y/τ_z. Adding per-vertex principal curvatures κ_1, κ_2 (or mean H = (κ_1+κ_2)/2 and Gauss K = κ_1·κ_2) as additional surface input features gives the model an explicit local-geometry signal that current 6-channel `(xyz, normals)` representation lacks.

**Mechanism.** Compute per-surface-point κ_1, κ_2 via local fitting: for each surface point, gather its k=8 nearest neighbors, fit a local quadratic patch in the tangent plane (z = ax² + bxy + cy²), and extract eigenvalues of the Hessian. Precompute once per case at dataset construction (cheap; ~1s per case). Append (κ_1, κ_2) or (H, K) as 2 extra channels to surface input.

**Code change.** Modify dataset loader to compute and cache curvatures (NumPy/SciPy `KDTree` + local fit). Add `--surface-curvature-features {none, h_k, k1_k2}` flag. Update model's surface input projection to consume 8 channels instead of 6. Document the feature in `train/surface_input_dim` config dump.

**Sweep plan (4-GPU DDP, 3 arms).**
- A: none (control, post-#490 + STRING-sep + Lion)
- B: H, K (mean + Gauss)
- C: κ_1, κ_2 (principal magnitudes + sign info)
- PR #517 base config + `--learnable-pe`.

**Win criterion.** val_abupt < 9.032 AND τ_y or τ_z component improves; ideally improvement is concentrated in high-curvature surface zones (require haku to log per-decile curvature error breakdown). Different from PR #314 (coord jitter) and PR #225 (mirror aug).

## Idea 6 — fern: O(2)-equivariant rotation-augmented training in y-z plane (aerodynamic symmetry)

**Hypothesis.** Cars are rigorously y-symmetric about the x-z mid-plane in DrivAerML (the dataset is left-right mirrored), and the wall-shear field obeys: `τ(reflect_y(x)) = R · τ(x)` where R = diag(1, -1, 1) flips τ_y. PR #225 (haku) tested *training-time* mirror augmentation and closed it. A different angle: replace train-time augmentation with a **frame-averaged** training objective — for each batch, present BOTH x and reflect_y(x), then minimize the symmetrized loss `0.5·[L(f(x), y) + L(reflect(f(reflect(x))), y)]`. This forces the predictions to be exactly equivariant *as a learned property*, not just approximately via aug.

**Mechanism.** This is a soft-equivariance regularizer akin to Cohen-Welling group-convolutions but applied at the data/loss level. Each batch's effective gradient is the average over the symmetric pair, encouraging the model's f to obey the symmetry exactly. Combined with test-time mirror+average ensemble (free at inference), this should compound. Differs from PR #225 because PR #225 augmented (random flip) rather than always training on the symmetric pair simultaneously.

**Code change.** Add `--symm-pair-loss <bool>` flag. In training step, when enabled, build a 2x batch by concatenating each input with its y-mirror. Forward both. Compute `L = 0.5·MSE(pred_orig, target_orig) + 0.5·MSE(R · pred_mirror, target_orig)` where R flips τ_y in the prediction (equivalent to flipping target's τ_y). Add a new metric `train/symm_consistency_pp = mean(|pred_orig - R·pred_mirror|^2 / |pred_orig|^2)` to track convergence to symmetry.

At eval/test time always evaluate as `0.5·(model(x) + R·model(reflect(x)))` (free 2x ensemble; ~5min per epoch overhead).

**Sweep plan (4-GPU DDP, 2 arms).**
- A: control — PR #517 + `--learnable-pe`
- B: `--symm-pair-loss` + test-time mirror ensemble
- Same hyperparameters.

**Win criterion.** val_abupt < 9.032 AND `train/symm_consistency_pp` → 0 by epoch 3; AND τ_y component improves on test split. If symm-consistency is already near zero in baseline (model auto-learns it), the train-time arm can be closed; the test-time ensemble alone may still help.

## Routing summary

| Student   | PR target          | Theme                              | Risk    |
|-----------|--------------------|------------------------------------|---------|
| frieren   | STRING-sep+Lion    | Anchor / merge-bar lock-in         | Low     |
| emma      | GradNorm           | Multi-task balancing               | Medium  |
| askeladd  | Streamline frame   | Coordinate-frame target rotation   | Medium  |
| alphonse  | Laplace penalty    | Physics-informed regularizer       | Medium  |
| haku      | Curvature features | Geometric input features           | Low     |
| fern      | y-symm pair loss   | Equivariance via paired loss       | Medium  |

All six are orthogonal to each other and to the 10 active WIP PRs.
