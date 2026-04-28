<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# DrivAerML Research Target

Research target for CFD surrogate modelling on DrivAerML. Given vehicle surface points and volume points, predict three target families:

- `surface_pressure`: surface pressure coefficient `cp`
- `wall_shear`: 3-channel surface wall-shear-stress vector
- `volume_pressure`: scalar pressure at volume points

## Mission

The research goal is to find the strongest DrivAerML model we can, with particular focus on `volume_pressure` and `wall_shear`. Success is measured on the held-out test metrics logged by `train.py` after the best validation checkpoint is reloaded. Validation metrics are useful for steering, but final claims must be made from `test_primary/*`.

The target is not merely to match the current public reference: the goal is to beat it decisively. Agents should relentlessly search for ways to drive down `test_primary/volume_pressure_rel_l2_pct`, `test_primary/wall_shear_rel_l2_pct`, the per-axis wall-shear metrics, and the aggregate `test_primary/abupt_axis_mean_rel_l2_pct`, without hiding regressions behind a single averaged number.

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
surface_wallshearstress.npy  # [N_surface, 3]
volume_xyz.npy         # [N_volume, 3]
volume_sdf.npy         # [N_volume] or [N_volume, 1]
volume_pressure.npy    # [N_volume] or [N_volume, 1]
```

The loader concatenates surface features into `[x, y, z, nx, ny, nz, area]` and volume features into `[x, y, z, sdf]`.

Targets:

| Tensor | Channels | Description |
|--------|----------|-------------|
| `surface_y` | 0 | `surface_pressure` / `surface_cp` |
| `surface_y` | 1-3 | `wall_shear_x`, `wall_shear_y`, `wall_shear_z` from `surface_wallshearstress` |
| `volume_y` | 0 | `volume_pressure` |

`normalizers.json` must provide `surface_cp`, `surface_wallshearstress`, and `volume_pressure` stats. Losses are computed in normalized space; all MAE and relative-L2 metrics are computed after denormalization.

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

- Training with `--train-surface-points N --train-volume-points M`: for each case and modality, `view_count = ceil(total_points / points_per_view)` when point-limited. Each view draws `points_per_view` rows uniformly with replacement. Across one epoch the loader draws about `total_points` rows per case per modality, but duplicates and missed points are expected. For example, one epoch of replacement sampling at one full equivalent draw sees about 63% unique points in expectation; subsequent epochs compound coverage.
- When surface and volume need different numbers of views, the case is repeated enough times to cover the larger count. The smaller modality returns empty tensors after its own views are covered instead of silently repeating full data.
- Validation/test with `--eval-surface-points N --eval-volume-points M`: each case is split into deterministic strided chunks with `torch.arange(view_index, total_points, view_count)`. This is full-fidelity evaluation: every loaded surface point and every loaded volume point is evaluated exactly once, then reaggregated by case.
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

The Transolver backbone must also keep padding masked internally. Slice-attention pooling, residual blocks, final normalization, and output heads should never let padded zero rows contribute to slice tokens or produce nonzero hidden states.

## Gradient, Weight, And Slope Telemetry

The trainer intentionally logs high-fidelity gradient and weight telemetry on every optimizer update by default. Future agents must preserve this unless the advisor explicitly asks to change the logging contract.

The W&B stream includes:

- `train/grad/*` — aggregate gradient health: global norm, RMS, mean absolute gradient, max absolute gradient, zero fraction, non-finite count, parameter norm, and grad-to-parameter norm ratio.
- `train/grad_type/<LayerType>/*` — the same statistics grouped by module class, for example `LinearProjection`, `TransolverAttention`, `LayerNorm`, or `TransformerBlock`.
- `train/grad_module/<LayerType>/<module_path>/*` — layer-by-layer statistics for every named module with trainable parameters.
- `train/grad_param/<LayerType>/<parameter_path>/*` — per-parameter statistics for every trainable tensor.
- `train/grad_hist/all` and `train/grad_hist_param/<LayerType>/<parameter_path>` — gradient histograms for distribution drift, spikes, saturation, dead layers, and collapse detection.
- `train/weight/*` — aggregate parameter health after each optimizer update: norm, mean, mean absolute value, RMS, standard deviation, min/max, max absolute value, zero fraction, non-finite count, and trainable/frozen tensor counts.
- `train/weight_type/<LayerType>/*`, `train/weight_module/<LayerType>/<module_path>/*`, and `train/weight_param/<LayerType>/<parameter_path>/*` — the same parameter statistics grouped by module class, layer path, and parameter tensor.
- `train/weight_hist/all` and `train/weight_hist_param/<LayerType>/<parameter_path>` — optional parameter histograms controlled by `--log-weight-histograms`.

Keep gradient logging close to `loss.backward()` and before `optimizer.step()` so it represents the update that is about to be applied. Keep weight logging after `optimizer.step()` so it represents the model state after the update. When adding new model blocks, make sure their parameters remain visible through `named_modules()` / `named_parameters()` so the layer-type and layer-path logging stays useful.

Slope telemetry is also part of the contract. Every `--slope-log-fraction 0.05` of the estimated optimizer-step budget, `train.py` logs slopes for key curves under `train/slope/*`: losses, grad norms, grad-to-param ratio, RMS gradients, max gradients, MAE curves, and relative-L2 curves. Validation slopes are logged under `val/slope/*` at validation events; final validation and test use `full_val/slope/*` and `test/slope/*` when enough history exists. If you rename metric keys, preserve or document equivalent slope curves.

Optional early-stop kill thresholds are available through `--kill-thresholds`. The format is a comma- or semicolon-separated list of `STEP:metric<value`, `STEP:metric<=value`, `STEP:metric>value`, or `STEP:metric>=value` checks, for example:

```
--kill-thresholds "500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25"
```

Each check is evaluated only when that metric is logged at or after the requested global optimizer step. If the condition fails, the run logs `early_stop/*`, finishes W&B, and skips expensive final validation/test. Use this to discard clearly unstable or hopeless short runs; do not use it to hide final metrics for serious confirmation runs.

The parser intentionally validates the full format before training starts. Accepted operators are `<`, `<=`, `>`, and `>=`; steps must be positive integers; values must be finite numbers; metric keys must be spelled exactly as logged.

## Metrics

Checkpoint selection uses an internal scalar mean over the AB-UPT-aligned scalar relative-L2 columns:

```
abupt_axis_mean_rel_l2_pct = mean(
    surface_pressure_rel_l2_pct,
    wall_shear_x_rel_l2_pct,
    wall_shear_y_rel_l2_pct,
    wall_shear_z_rel_l2_pct,
    volume_pressure_rel_l2_pct,
)
```

Primary logged metrics:

- `val_primary/abupt_axis_mean_rel_l2_pct` — checkpoint selection metric
- `full_val_primary/abupt_axis_mean_rel_l2_pct` — best-checkpoint validation metric after training
- `test_primary/abupt_axis_mean_rel_l2_pct` — final held-out scalar used for quick run triage
- `*_primary/surface_pressure_mae`, `*_primary/wall_shear_mae`, `*_primary/volume_pressure_mae` — target-family MAE diagnostics
- `*_primary/surface_pressure_rel_l2_pct`, `*_primary/wall_shear_rel_l2_pct`, `*_primary/volume_pressure_rel_l2_pct` — target-family relative-L2 diagnostics
- `*_primary/wall_shear_x_rel_l2_pct`, `*_primary/wall_shear_y_rel_l2_pct`, `*_primary/wall_shear_z_rel_l2_pct` — per-axis wall-shear relative-L2 diagnostics

Lower is better. For paper-facing reporting, use the final `test_primary/*` metrics and include the target-family MAEs plus `surface_pressure`, vector `wall_shear`, per-axis wall-shear, and `volume_pressure` relative-L2 percentages.

### AB-UPT Metric Alignment

AB-UPT reports relative L2 error in percent, averaged per sample across the split. The paper defines the relative L2 over all points and output dimensions for a target vector, then averages those per-sample values across test cases.

For DrivAerML, the main paper table reports `p_s`, `u`, and `omega`; Appendix A reports `p_s`, vector wall shear `tau`, and `p_v`; Appendix B also reports the DoMINO comparison with wall shear split into `tau_x`, `tau_y`, and `tau_z`. This repo logs the matching columns it can compute:

- `surface_pressure_rel_l2_pct` maps to AB-UPT `p_s`.
- `wall_shear_rel_l2_pct` maps to vector `tau`.
- `wall_shear_x_rel_l2_pct`, `wall_shear_y_rel_l2_pct`, `wall_shear_z_rel_l2_pct` map to the per-axis wall-shear columns.
- `volume_pressure_rel_l2_pct` maps to AB-UPT `p_v`.

`abupt_axis_mean_rel_l2_pct` is a convenience aggregate for checkpointing and triage. It is not a standalone AB-UPT benchmark column, so paper-facing comparisons should quote the individual test fields above.

### State-Of-The-Art Targets

AB-UPT is the current public DrivAerML reference to beat. Its reported DrivAerML values are relative L2 error in percent, averaged per test case; lower is better. Use these as external SOTA targets, while still reporting this repo's exact held-out `test_primary/*` metrics.

| Target | This repo metric | AB-UPT reference |
|--------|------------------|-----------------:|
| Surface pressure `p_s` | `test_primary/surface_pressure_rel_l2_pct` | `3.82` |
| Vector wall shear `tau` | `test_primary/wall_shear_rel_l2_pct` | `7.29` |
| Volume pressure `p_v` | `test_primary/volume_pressure_rel_l2_pct` | `6.08` |
| Wall shear `tau_x` | `test_primary/wall_shear_x_rel_l2_pct` | `5.35` |
| Wall shear `tau_y` | `test_primary/wall_shear_y_rel_l2_pct` | `3.65` |
| Wall shear `tau_z` | `test_primary/wall_shear_z_rel_l2_pct` | `3.63` |

The per-axis wall-shear table reports `p_s = 3.76` and `p_v = 6.29` in the same AB-UPT/DoMINO comparison setting. Treat the stricter of the relevant published values as the target band when deciding whether a result is exciting. A run that only improves the internal aggregate but misses `p_v` or wall-shear targets has not solved the problem.

### Target Scaling

The baseline normalizes each target channel with train-split mean/std only. `surface_cp` is already a pressure coefficient, and the packaged loader does not expose verified per-case freestream or Reynolds metadata for volume pressure or wall shear. Do not add guessed per-case `p / Re^2` or Cp-style rescaling unless that metadata is explicitly plumbed through and final validation/test metrics are still computed on the original target units for AB-UPT alignment.

## Experiment Length

This target trains slower than TandemFoilSet. Use a mix of shorter and longer experiments:

- Short runs are for viability: does the idea compile, stay stable, produce sane gradients, and improve early validation trends?
- Longer runs are for confirmation: does a stable idea keep improving once the model has enough update budget to learn surface and volume structure?

Do not run only short experiments; they can discard ideas before the model has started learning. Do not run only long experiments; they burn the time budget before enough hypotheses are screened. Use the slope, gradient, and weight logs to decide which short runs deserve a longer confirmation run.

## Research Workflow

For substantial architecture or training-strategy changes, preserve main context by using research subagents before implementation. The advisor should deliberately run two complementary streams instead of only local hill-climbing.

### Stream 1: Exploit Existing Evidence

Mine the existing DrivAerML history before assigning a wave of experiments:

- Search this target repo's prior branches and PRs, including `main`, `codex/optimized-lineage`, and any previous DrivAerML experiment branches.
- Search the `wandb/senpai` `radford` branch, related DrivAerML PRs, W&B runs, analysis files, and advisor notes for mechanisms that already worked or failed.
- Assign a subset of students to build on the strongest past ideas rather than rediscovering them: useful preprocessing, target transforms, batching choices, normalization, loss weighting, architecture deltas, and optimizer schedules.
- When reusing a past idea, state the source PR/run/branch in the assignment and explain what is being preserved versus changed.

### Stream 2: Generate New High-Variance Ideas

Reserve another subset of students for fresh, aggressive ideas that may not be present in the history. Search broadly across CFD surrogate literature and adjacent ML:

- DrivAerML and AI-for-CFD work: AB-UPT, UPT, Transolver, Transolver-3, GeoTransolver, DoMINO, GINO/FNO-style operators, graph/mesh transformers, point-cloud networks, geometric encodings, and multi-resolution tokenization.
- AI-for-science ideas from physics, chemistry, and biology: equivariant or invariant representations, denoising/pretraining, multiscale message passing, latent neural operators, simulator residual modelling, and uncertainty-aware losses.
- Modern transformer and LLM architecture ideas available at run time: DeepSeek, Kimi, GLM, and other recent papers for MoE routing, latent/linear attention, token compression, normalization, optimizer, curriculum, and distillation tricks that can plausibly transfer to point/field regression.

Wild ideas are welcome, but they must still honor the data split, full-fidelity test evaluation, and telemetry contract. A clever idea that cannot be measured on `test_primary/*` is not useful for this sprint.

### Suggested Subagents

A good advisor pass should split the work like this:

- Architecture research agent: compare vanilla Transolver, grouped surface/volume variants, UPT-style ideas, geometric encodings, and memory-efficient attention choices.
- Experiment-history research agent: review prior DrivAerML PRs, W&B runs, failures, and merged wins so the next hypothesis is not a duplicate.
- Frontier-ideas research agent: search outside the repo for recent AI-for-science and transformer/LLM mechanisms that might transfer to DrivAerML.
- Optimization/data research agent: inspect batch-size, compile, gradient and weight health, validation cadence, target weighting, and preprocessing implications.
- Synthesis research agent: read the prior research-agent outputs, do any missing checks, and choose the final experiment set across both streams.

Use subagents when the task is broad enough to justify them; include their conclusions in the PR body. Keep the train script's logging fidelity intact while iterating.

## Lineages

`main` is the vanilla baseline lineage: simple grouped Transolver, AdamW, EMA, compile enabled, and explicit metrics for all targets.

This branch is `codex/optimized-lineage`: a higher-capacity starting point with 4 layers, 256 hidden width, 128 slices, larger point chunks, and slower EMA decay. It may explore more opinionated model and optimizer choices, but it must keep the same data targets, final full-fidelity validation/test metrics, gradient telemetry, weight telemetry, and slope telemetry. Advisor programs can assign a subset of students here while others continue vanilla-lineage exploration on `main`.

## Reference Targets

There is no exact published number for this repo's packaged three-target trainer path. Closest public DrivAerML surface-pressure references are low-single-digit relative-L2 percent:

- AB-UPT: `p_s = 3.82`
- Transolver baseline reported by AB-UPT: `p_s = 4.81`
- Transolver-3: `p_s = 3.71`
- Transolver++ row inside Transolver-3: `p_s = 4.12`

Treat these as external target bands for surface pressure only, not apples-to-apples multi-target baselines.

## Constraints

- GPUs have 96 GB. Prefer larger batch sizes when they use GPU capacity well, but compare gradient and weight health across batch sizes because larger batches can degrade final quality.
- `torch.compile` is enabled by default to improve throughput. If you disable it for an experiment, explain why.
- Respect `SENPAI_TIMEOUT_MINUTES`; do not override it in experiments.
- Simpler is better. A small improvement with unreadable complexity is not worth it.
- Data loaders and split files are read-only for normal research PRs.
- No new packages outside `pyproject.toml`.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. GitHub Issues are used for communication with the human researcher team. See `instructions/prompt-advisor.md` and `instructions/prompt-student.md`.
