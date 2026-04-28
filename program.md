<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# DrivAerML Research Target

Research target for CFD surrogate modelling on DrivAerML. Given vehicle surface points and volume points, predict three target families:

- `surface_pressure`: surface pressure coefficient `cp`
- `wall_shear`: 3-channel surface friction / wall shear vector
- `volume_pressure`: scalar pressure at volume points

The baseline is a plain grouped Transolver with one shared backbone and separate surface/volume heads. Keep the vanilla path simple; more opinionated variants should be explored as separate experiment arms.

## Codebase

- `train.py` — trainer, model, training loop, sparse-cadence full-fidelity validation, end-of-run full-fidelity test evaluation, W&B artifact upload. **Primary editable entrypoint.**
- `data/loader.py` — PVC-backed DrivAerML case store, point-view sampling, batching, target stats. **Read-only during normal experiment PRs.**
- `data/generate_manifest.py` — regenerates `data/split_manifest.json` from the processed PVC manifests. **Read-only.**
- `data/preload.py` — validates packaged arrays and writes point counts. **Read-only.**
- `data/split_manifest.json` — pinned public processed split. **Read-only.**
- `instructions/prompt-advisor.md`, `instructions/prompt-student.md` — senpai role prompts.
- `pyproject.toml` — runtime deps. Add any new package in the same PR that uses it.

## Data

Processed samples live on the PVC at `/mnt/pvc/Processed/drivaerml_processed` or `/mnt/new-pvc/Processed/drivaerml_processed`.

Each case directory must contain:

```
surface_xyz.npy        # [N_surface, 3]
surface_normals.npy    # [N_surface, 3]
surface_area.npy       # [N_surface] or [N_surface, 1]
surface_cp.npy         # [N_surface] or [N_surface, 1]
surface_friction.npy   # [N_surface, 3]
volume_xyz.npy         # [N_volume, 3]
volume_sdf.npy         # [N_volume] or [N_volume, 1]
volume_pressure.npy    # [N_volume] or [N_volume, 1]
```

The loader concatenates surface features into `[x, y, z, nx, ny, nz, area]` and volume features into `[x, y, z, sdf]`.

Targets:

| Tensor | Channels | Description |
|--------|----------|-------------|
| `surface_y` | 0 | `surface_pressure` / `surface_cp` |
| `surface_y` | 1-3 | `wall_shear_x`, `wall_shear_y`, `wall_shear_z` from `surface_friction` |
| `volume_y` | 0 | `volume_pressure` |

`normalizers.json` must provide `surface_cp`, `surface_friction`, and `volume_pressure` stats. Losses are computed in normalized space; all MAE and relative-L2 metrics are computed after denormalization.

## Splits

The pinned split follows the packaged public processed DrivAerML manifest:

| Split | Cases |
|-------|------:|
| train | 400 |
| val | 34 |
| test | 50 |

The old failed-case gap has been repaired in this packaged split. `data/loader.py` validates that the split is exactly `400 / 34 / 50`, has no overlap, and includes the restored public case IDs.

## Point-View Sampling

Full surface and volume cases can be large, so the trainer uses point-limited views.

- Training with `--train-surface-points N --train-volume-points M`: each case is repeated enough times to cover the larger surface/volume view count; each modality participates in its own required number of views and does not get silently duplicated after it is covered.
- Validation/test with `--eval-surface-points N --eval-volume-points M`: each case is split into deterministic strided chunks. This is full-fidelity evaluation: every loaded surface point and every loaded volume point is evaluated exactly once, then reaggregated by case.
- Set a point limit to `0` only when you intentionally want full-case loading in one batch and have checked memory.

Validation is intentionally sparse by default: `train.py` validates at epoch 1, every `--validation-every 10` epochs, and the final epoch. After loading the best checkpoint, it runs and logs `full_val/*` before `test/*`. Do not replace final validation/test with a random point sample.

## Model Contract

The baseline model interface is:

```python
out = model(
    surface_x=batch.surface_x,
    surface_mask=batch.surface_mask,
    volume_x=batch.volume_x,
    volume_mask=batch.volume_mask,
)
surface_pred_norm = out["surface_preds"]  # [B, N_surface, 4]
volume_pred_norm = out["volume_preds"]    # [B, N_volume, 1]
```

`batch.surface_mask` and `batch.volume_mask` are required because cases are padded independently. Do not compute loss or metrics on padding tokens.

The legacy `out["preds"]` alias still points to `surface_preds` for compatibility, but new work should use the explicit keys.

## Gradient And Slope Telemetry

The trainer intentionally logs high-fidelity gradient telemetry on every optimizer update by default. Future agents must preserve this unless the advisor explicitly asks to change the logging contract.

The W&B stream includes:

- `train/grad/*` — aggregate gradient health: global norm, RMS, mean absolute gradient, max absolute gradient, zero fraction, non-finite count, parameter norm, and grad-to-parameter norm ratio.
- `train/grad_type/<LayerType>/*` — the same statistics grouped by module class, for example `LinearProjection`, `TransolverAttention`, `LayerNorm`, or `TransformerBlock`.
- `train/grad_module/<LayerType>/<module_path>/*` — layer-by-layer statistics for every named module with trainable parameters.
- `train/grad_param/<LayerType>/<parameter_path>/*` — per-parameter statistics for every trainable tensor.
- `train/grad_hist/all` and `train/grad_hist_param/<LayerType>/<parameter_path>` — gradient histograms for distribution drift, spikes, saturation, dead layers, and collapse detection.

Keep gradient logging close to `loss.backward()` and before `optimizer.step()` so it represents the update that is about to be applied. When adding new model blocks, make sure their parameters remain visible through `named_modules()` / `named_parameters()` so the layer-type and layer-path logging stays useful.

Slope telemetry is also part of the contract. Every `--slope-log-fraction 0.05` of the estimated optimizer-step budget, `train.py` logs slopes for key curves under `train/slope/*`: losses, grad norms, grad-to-param ratio, RMS gradients, max gradients, MAE curves, and relative-L2 curves. Validation slopes are logged under `val/slope/*` at validation events; final validation and test use `full_val/slope/*` and `test/slope/*` when enough history exists. If you rename metric keys, preserve or document equivalent slope curves.

## Metrics

Checkpoint selection uses the mean relative L2 across the three target families:

```
target_mean_rel_l2_pct = mean(
    surface_pressure_rel_l2_pct,
    wall_shear_rel_l2_pct,
    volume_pressure_rel_l2_pct,
)
```

Primary logged metrics:

- `val_primary/target_mean_rel_l2_pct` — checkpoint selection metric
- `full_val_primary/target_mean_rel_l2_pct` — best-checkpoint validation metric after training
- `test_primary/target_mean_rel_l2_pct` — final held-out result
- `*_primary/surface_pressure_mae`, `*_primary/wall_shear_mae`, `*_primary/volume_pressure_mae` — target-family MAE diagnostics
- `*_primary/surface_pressure_rel_l2_pct`, `*_primary/wall_shear_rel_l2_pct`, `*_primary/volume_pressure_rel_l2_pct` — target-family relative-L2 diagnostics

Lower is better. For paper-facing reporting, use the final `test_primary/*` metrics and include all three MAEs plus all three relative-L2 percentages. The old `surface_rel_l2_pct` key remains as an alias for `surface_pressure_rel_l2_pct`, not the full problem score.

## Experiment Length

This target trains slower than TandemFoilSet. Use a mix of shorter and longer experiments:

- Short runs are for viability: does the idea compile, stay stable, produce sane gradients, and improve early validation trends?
- Longer runs are for confirmation: does a stable idea keep improving once the model has enough update budget to learn surface and volume structure?

Do not run only short experiments; they can discard ideas before the model has started learning. Do not run only long experiments; they burn the time budget before enough hypotheses are screened. Use the slope and gradient logs to decide which short runs deserve a longer confirmation run.

## Research Workflow

For substantial architecture or training-strategy changes, preserve main context by using research subagents before implementation. A good advisor pass should split the work like this:

- Architecture research agent: compare vanilla Transolver, grouped surface/volume variants, UPT-style ideas, geometric encodings, and memory-efficient attention choices.
- Experiment-history research agent: review prior DrivAerML PRs, W&B runs, failures, and merged wins so the next hypothesis is not a duplicate.
- Optimization/data research agent: inspect batch-size, compile, gradient health, validation cadence, target weighting, and preprocessing implications.
- Synthesis research agent: read the prior research-agent outputs, do any missing checks, and choose the final experiment set.

Use subagents when the task is broad enough to justify them; include their conclusions in the PR body. Keep the train script's logging fidelity intact while iterating.

## Lineages

`main` is the vanilla baseline lineage: simple grouped Transolver, AdamW, EMA, compile enabled, and explicit metrics for all targets.

An optimized lineage should live on branch `codex/optimized-lineage`. It may explore more opinionated model and optimizer choices, but it must keep the same data targets, final full-fidelity validation/test metrics, gradient telemetry, and slope telemetry. Advisor programs can assign a subset of students to that branch while others continue vanilla-lineage exploration.

## Reference Targets

There is no exact published number for this repo's packaged three-target trainer path. Closest public DrivAerML surface-pressure references are low-single-digit relative-L2 percent:

- AB-UPT: `p_s = 3.82`
- Transolver baseline reported by AB-UPT: `p_s = 4.81`
- Transolver-3: `p_s = 3.71`
- Transolver++ row inside Transolver-3: `p_s = 4.12`

Treat these as external target bands for surface pressure only, not apples-to-apples multi-target baselines.

## Constraints

- GPUs have 96 GB. Prefer larger batch sizes when they use GPU capacity well, but compare gradient health across batch sizes because larger batches can degrade final quality.
- `torch.compile` is enabled by default to improve throughput. If you disable it for an experiment, explain why.
- Respect `SENPAI_TIMEOUT_MINUTES`; do not override it in experiments.
- Simpler is better. A small improvement with unreadable complexity is not worth it.
- Data loaders and split files are read-only for normal research PRs.
- No new packages outside `pyproject.toml`.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. GitHub Issues are used for communication with the human researcher team. See `instructions/prompt-advisor.md` and `instructions/prompt-student.md`.
