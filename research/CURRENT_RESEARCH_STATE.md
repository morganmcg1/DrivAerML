# SENPAI Research State

- 2026-04-30 (latest check) — Wave 1 in flight (16 PRs #74–#89), all 16 bengio student pods 1/1 READY
- All 16 student pods Running and healthy; no PRs review-ready yet; no human issues

## Most Recent Human Researcher Direction

No human researcher issues found. Mission: crush DrivAerML AB-UPT public reference metrics.

AB-UPT targets to beat:
- surface_pressure_rel_l2_pct < 3.82
- wall_shear_rel_l2_pct < 7.29
- volume_pressure_rel_l2_pct < 6.08
- wall_shear_x_rel_l2_pct < 5.35
- wall_shear_y_rel_l2_pct < 3.65
- wall_shear_z_rel_l2_pct < 3.63
- abupt_axis_mean_rel_l2_pct ~ 4.51 (mean of 5 axis metrics)

## Current Research Focus and Themes

**Wave 1: Two parallel streams to maximize coverage**

### Stream 1 — Exploit radford DrivAerML History

The strongest known prior result (radford PR #2593) used:
- 4L/256d architecture (model_layers=4, model_hidden_dim=256)
- No EMA (--no-use-ema)
- Fourier positional encoding (requires model.py change)
- Cosine LR schedule T_max=30
- lr=3e-4, weight_decay=1e-4, batch_size=2

Experiments in this stream build on that recipe:
1. alphonse: Replicate 4L/256d + no-EMA + Fourier + T_max=30 as bengio baseline
2. fern: LR sweep (1e-4, 5e-4) with the confirmed 4L/256d+no-EMA+Fourier+T_max=30 recipe
3. gilbert: 5L/256d depth scaling with no-EMA + Fourier + T_max=30
4. haku: 4L/384d width scaling with no-EMA + T_max=30
5. kohaku: Higher slice count (128 slices) with 4L/256d recipe
6. emma: More surface/volume train points (60k) with confirmed recipe
7. tanjiro: Surface/volume loss weight tuning (surface_weight=2.0)
8. violet: Longer cosine schedule T_max=50 + lr=2e-4

### Stream 2 — Fresh High-Variance Ideas

9. askeladd: SDF-conditioned volume encoding — inject SDF as extra MLP feature before attention
10. chihiro: asinh target normalization for wall shear to compress heavy-tail distribution
11. edward: Per-target loss weighting via dynamic uncertainty weighting (homoscedastic uncertainty)
12. frieren: Cross-attention bridge between surface and volume tokens (separate backbones per modality)
13. nezuko: MLP-ratio=6 or 8 in transformer blocks to increase expressivity without depth
14. norman: Dropout regularization sweep (dropout=0.1) with cosine schedule
15. senku: RFF (Random Fourier Features) positional encoding with larger bandwidth
16. thorfinn: Gradient clipping sweep (clip_norm=0.5) + weight_decay=5e-4 for tighter regularization

## Potential Next Research Directions

- Physics-informed loss terms (enforce continuity/momentum constraints)
- Equivariant representations using SO(3)/SE(3) encodings for normals and shear vectors
- Multi-resolution tokenization: coarse global + fine local attention
- Latent diffusion conditioning for geometric priors
- DeepSpeed ZeRO / larger model once baseline recipe confirmed
- Ensemble of best models from different random seeds
- Curriculum learning: easier samples first, then hard aerodynamic extremes
- Pretraining on synthetic/simplified CFD data then fine-tuning
- Graph neural network hybrid with Transolver backbone
