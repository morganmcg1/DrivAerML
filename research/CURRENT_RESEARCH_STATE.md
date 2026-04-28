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

## Round 1 — two parallel streams

### Stream 1 — exploit existing evidence (5 students)

Pin the floor with multiple known-good baselines and proven-additive deltas
(LR schedule, metric-aware aux loss, scale-up). Provides reliable comparison
points for everything else.

### Stream 2 — fresh high-variance ideas (11 students)

Bias toward `wall_shear` (7.29%) and `volume_pressure` (6.08%) — the two
hardest AB-UPT targets. Researcher-agent generates a ranked hypothesis pool;
each student tests a single mechanistically distinct delta against the same
strong base config.

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
