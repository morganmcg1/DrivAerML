# SENPAI Research Ideas — 2026-04-30 14:40

Generated after Round-2 review. All ideas are for the yi branch DrivAerML target.
Current baseline: abupt=12.74 (PR #66 thorfinn, 6L/256d + W_y=W_z=2).

## Ideas Assigned in Round-3

| PR | Student | Hypothesis |
|---|---|---|
| #95 | alphonse | Area-weighted surface MSE loss |
| #96 | chihiro | MLP ratio sweep (4 vs 6 vs 8) |
| #97 | edward | Slice count sweep (128 vs 192 vs 256) |
| #98 | emma | Weight decay sweep (1e-4/5e-4/2e-3/5e-3) |
| #99 | fern | LR peak sweep (1e-4/2e-4/3e-4/5e-4) |
| #100 | frieren | cp-channel upweight (1.5/2.0/3.0) |
| #101 | gilbert | Larger point budget (131072 vs 65536) |
| #102 | haku | Dropout sweep (0/0.05/0.10/0.20) |
| #103 | kohaku | Volume-surface cross-attention |
| #104 | senku | EMA decay sweep (0.999/0.9995/0.9997/0.9999) |
| #105 | tanjiro | Huber surface loss (delta=0.05/0.10) |
| #106 | thorfinn | Finer tau_y/z weight sweep (2.0/2.5/3.0 + asym) |
| #107 | violet | Volume loss decay schedule (4->1.5, 3->1.5) |

## Ideas for Round-4

### Architecture Ideas (Bold Swings)

1. **Separate surface/volume backbone branches** — Dedicated 6L/256d surface backbone + 3L/128d volume backbone, specialized per modality.

2. **Point cloud hierarchy (multi-scale attention)** — Group 65536 points into coarse clusters, apply SetAbstraction-style attention at multiple scales. Better spatial resolution for tau_y/z.

3. **Fourier position encodings** — sin/cos at multiple frequencies added to xyz coords, giving explicit spatial frequency decomposition.

4. **Mixture-of-Experts per surface region** — Route 4-8 experts based on point position (wheel arch, underbody, roof, stagnation zone have different flow regimes).

### Loss / Training Ideas

5. **Laplacian smoothness regularization** — Penalize discontinuities in volume pressure gradient fields.

6. **Surface-volume consistency aux loss** — Enforce p_s = p_v at the boundary (SDF~0 volume points).

7. **Adversarial domain adaptation** — Gradient-reversal layer to make backbone invariant to train/test geometry distribution shift, targeting the structural val-test gap.

8. **Online hard example mining (OHEM)** — Upweight hard cases in batch sampling based on per-case rolling val L2 error.

### Data / Augmentation Ideas

9. **Point cloud jitter augmentation** — Small Gaussian noise (sigma=0.001 in normalized coords) on surface xyz during training.

10. **Radial feature augmentation** — Add |r| and theta (polar angle around car axis) as additional input features.

11. **Geometric moment conditioning** — Per-case bounding box, surface area, volume as global conditioning signals (cheaper than FiLM neural encoder).

### Optimization Ideas

12. **SAM (Sharpness-Aware Minimization)** — Seek flatter minima via weight perturbation + gradient at perturbed point. Directly targets val->test gap.

13. **Gradient accumulation with effective bs=32** — 4 accumulation steps at bs=8, better gradient estimates without VRAM increase.

14. **AdaGrad layer-wise LR (discriminative fine-tuning)** — Earlier layers get lower LR (carry learned features), later layers get higher LR.

### Ensemble / Post-processing Ideas

15. **Multi-checkpoint ensemble** — Average best checkpoints from epochs N-2, N-1, N post-hoc.

16. **TTA on mature model** — Test-time bilateral symmetry averaging. Failed at epoch 1 (PR #16) but may help on the better-trained 12.74 model.

### Physics-Informed Ideas

17. **Divergence-free constraint on wall-shear** — Surface divergence of wall shear stress is constrained by the pressure Laplacian; add soft regularization.

18. **Bernoulli-consistency aux loss** — In potential flow regions, enforce approximate p + 0.5*rho*|v|^2 = const.
