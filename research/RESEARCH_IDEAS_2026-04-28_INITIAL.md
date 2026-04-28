# DrivAerML Round 1 — Hypothesis Pool
**Generated:** 2026-04-28 · **Researcher:** researcher-agent

Targets (AB-UPT, lower is better): `p_s=3.82%`, `tau=7.29%`, `p_v=6.08%`,
`tau_x=5.35%`, `tau_y=3.65%`, `tau_z=3.63%`

Ranked by expected impact / implementation cost ratio.
Bias toward `wall_shear` and `volume_pressure` (hardest metrics).

---

## Tier 1 — high impact, low complexity

| # | Name | Summary | Target | PR |
|---|---|---|---|---|
| H01 | Per-Axis Wall-Shear Loss Weights | Upweight tau_x/y/z channels (2–4x) vs cp; MSE blind to 2x error gap | tau (7.29%) | #10 (haku) |
| H02 | Volume Loss Upweighting | Single `--volume-loss-weight 2–3` flag; volume head undertrained vs surface | p_v (6.08%) | #9 (gilbert) |
| H04 | Tangential Wall-Shear Projection | Project predicted/target wall-shear onto tangent plane; removes unphysical normal component | tau axes | #11 (kohaku) |
| H07 | Scale slices+hidden | slices 96→192, hidden 192→256; capacity-limited architecture | all | (askeladd #3 covers this) |
| H08 | Deeper Model 5L | model_layers 3→5; depth for multi-scale flow features | all | #14 (senku) |

## Tier 2 — medium impact, targeted mechanism

| # | Name | Summary | Target | PR |
|---|---|---|---|---|
| H06 | SDF-Gated Volume Attention | Bias slice routing toward near-wall volume points (small |SDF|) | p_v (6.08%) | #15 (tanjiro) |
| H09 | Progressive EMA Decay | Anneal EMA 0.99→0.9999 cosine; diffusion model recipe for sharper final ckpt | test ckpt | #13 (norman) |
| H15 | Stochastic Depth | Linear DropPath 0–10%; regularizer + ~10% throughput gain | generalization | #12 (nezuko) |
| H18 | TTA Bilateral Symmetry | Mirror xz-plane at inference, average; free ensemble at zero training cost | tau_y | #16 (thorfinn) |
| H10 | Dropout Reg | `model_dropout=0.05`; small-data transformer standard | generalization | (reserved) |

## Tier 3 — novel physics-informed ideas

| # | Name | Summary | Target | Complexity |
|---|---|---|---|---|
| H03 | Joint H01+H02 | Combined channel rebalancing (stack H01 and H02 wins) | all | Small |
| H05 | Spectral Gradient Loss | Penalize spatial smoothness errors via finite-diff gradient on surface k-NN | tau | Medium |
| H11 | Divergence-Free Wall-Shear | Penalize surface divergence of predicted wall-shear (should → 0 at no-slip wall) | tau | Medium-Large |
| H12 | Curvature Feature Augment | Append mean/Gaussian curvature to surface_x (7D→9D); explicit shape signal | p_s, tau | Large |
| H13 | Geometry Complexity Curriculum | Train on simpler cars first; only useful if hard-case gap is the bottleneck | p_s, tau | Medium |
| H14 | Multi-Resolution Volume | SDF-stratified near-wall oversampling or hierarchical cross-attention | p_v | Medium-Large |
| H16 | Delta Learning from Prior | Predict RANS residual from inviscid prior; reduces target dynamic range 50-80% | all | Large |
| H17 | SO(3) Frame Normalization | Canonicalize vehicle orientation; reduces orientation variance in xyz features | all | Medium |
| H19 | Weight Decay Sweep | AdamW wd sweep [0.001, 0.01, 0.05]; standard small-data regularization | all | Small |
| H20 | K-Means Slice Init | Seed slice tokens from K-Means on training geometry; warm-start routing | all | Medium |

---

## Round 1 coverage

All Tier 1 and most Tier 2 ideas are covered by the 16 Round-1 PRs (#2–#17).
Tier 3 and H03 (combination) reserved for Round 2 once baseline is established.
