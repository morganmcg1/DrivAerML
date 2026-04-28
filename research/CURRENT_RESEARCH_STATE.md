# SENPAI Research State

- **Date:** 2026-04-28
- **Branch:** `yi`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B:** `wandb-applied-ai-team/senpai-v1-drivaerml`
- **Most recent direction from human team:** none received

## Current research focus

Round 1 calibration on a clean slate. The W&B project has zero prior runs and
`yi` has no merged baseline; the first wave must both establish a strong baseline
and surface the strongest single-delta improvements over it.

## Known prior art (from outside the `yi` W&B project)

- `codex/optimized-lineage` branch ships a heavier baseline: 4L/256d/4h/128sl,
  lr 2e-4, wd 5e-4, 65k points, EMA 0.9995. Treated as the proven floor.
- `wandb/senpai` `radford` branch converged on a 4L/512d/8h champion with
  fourier features, cosine-T_max=36 LR schedule, EMA 0.9995, metric-aware
  rel-L2 auxiliary loss, and DomainLayerNorm. Best surface val ≈ 3.6%.
  The hardest known levers were `wall_shear` and `volume_pressure`.

## Round 1 — active PRs (all 16 students assigned 2026-04-28)

### Stream 1 — exploit existing evidence

| PR | Student | Hypothesis |
|---|---|---|
| #2 | alphonse | Stock defaults baseline — calibration floor |
| #3 | askeladd | codex/optimized-lineage config (4L/256d/4h, 65k pts, lr=2e-4) |
| #4 | chihiro | Large model 4L/512d/8h — radford champion scale-up |
| #5 | edward | Cosine LR + 5% warmup (proven radford winner family) |
| #6 | emma | Metric-aware MSE + rel-L2 aux loss (proven 2nd radford winner) |

### Stream 2 — fresh targeted ideas

| PR | Student | Hypothesis | Primary target |
|---|---|---|---|
| #7 | fern | Gaussian random Fourier features for coordinates | p_s / tau |
| #8 | frieren | Per-case geometry FiLM conditioning | all |
| #9 | gilbert | Volume loss weight sweep 2.0x vs 3.0x | p_v (6.08%) |
| #10 | haku | Per-axis wall-shear channel loss weights (2x vs 3x) | tau (7.29%) |
| #11 | kohaku | Tangential wall-shear projection loss (physics-aware) | tau axes |
| #12 | nezuko | Stochastic depth / DropPath regularization | generalization |
| #13 | norman | Progressive EMA decay anneal 0.99→0.9999 | test checkpoint |
| #14 | senku | Deeper model 5L/256d/4h (depth ablation) | all |
| #15 | tanjiro | SDF-gated volume attention bias (near-wall emphasis) | p_v (6.08%) |
| #16 | thorfinn | Test-time bilateral symmetry TTA (xz-plane) | tau_y esp. |
| #17 | violet | Surface-area-weighted MSE loss (physics-consistent) | p_s / tau |

## Candidate next directions (post round 1)

- Geometry-aware conditioning (per-case mesh encoder + FiLM/AdaLN)
- Equivariant / vector-neuron variants of slice attention for vector targets
- Latent neural-operator decoders for full-resolution surface/volume queries
- Physics-residual auxiliary losses (continuity, smoothness, log-magnitude
  wall-shear targets)
- Attention scaling: linear / Performer / grouped-query for full-resolution
- Curriculum sampling and hard-example mining once a stable baseline exists
- Pretraining via self-supervised denoising on volume/surface points

## Constraints

- `SENPAI_MAX_EPOCHS` and `SENPAI_TIMEOUT_MINUTES` are fixed by harness — do
  not override. Throughput improvements (compile, AMP, point-count tuning,
  attention scaling) are the main lever for "more update budget".
- Read-only files: `data/*`, `pyproject.toml`, `instructions/*`. All edits
  go in `train.py`.
