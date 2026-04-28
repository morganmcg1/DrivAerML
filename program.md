<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# DrivAerML Research Target

Research target for CFD surrogate modelling on DrivAerML. Given a 3D vehicle surface point cloud with normals and panel area, predict the surface pressure coefficient `cp`.

The baseline is a surface-only Transolver with physics-aware attention over irregular point clouds. Beat it.

## Codebase

- `train.py` — trainer, model, training loop, validation, end-of-run test evaluation, W&B artifact upload. **Primary editable entrypoint.**
- `data/loader.py` — PVC-backed DrivAerML case store, point-view sampling, batching, target stats. **Read-only during normal experiment PRs.**
- `data/generate_manifest.py` — regenerates `data/split_manifest.json` from the processed PVC manifests. **Read-only.**
- `data/preload.py` — validates the packaged arrays and writes point counts. **Read-only.**
- `data/split_manifest.json` — pinned public processed split. **Read-only.**
- `instructions/prompt-advisor.md`, `instructions/prompt-student.md` — senpai role prompts.
- `pyproject.toml` — runtime deps. Add any new package in the same PR that uses it.

## Data

Processed samples live on the PVC at `/mnt/pvc/Processed/drivaerml_processed` or `/mnt/new-pvc/Processed/drivaerml_processed`.

Each case directory contains:

```
surface_xyz.npy       # [N, 3]
surface_normals.npy   # [N, 3]
surface_area.npy      # [N] or [N, 1]
surface_cp.npy        # [N] or [N, 1]
```

The loader concatenates surface features into:

| Dims | Feature |
|------|---------|
| 0-2 | Surface coordinates `(x, y, z)` |
| 3-5 | Surface normal vector `(nx, ny, nz)` |
| 6 | Surface panel area |

Target:

| Channel | Description |
|---------|-------------|
| 0 | `cp` — surface pressure coefficient |

`normalizers.json` supplies the target mean and standard deviation used for training loss normalization. Metrics are always computed after denormalization.

## Splits

The pinned split follows the packaged public processed DrivAerML manifest:

| Split | Cases |
|-------|------:|
| train | 400 |
| val | 34 |
| test | 50 |

The old failed-case gap has been repaired in this packaged split. `data/loader.py` validates that the split is exactly `400 / 34 / 50`, has no overlap, and includes the restored public case IDs.

## Point-View Sampling

Full surface cases can be large, so the trainer defaults to point-limited views.

- Training with `--train-surface-points N`: each case is repeated `ceil(num_points / N)` times per epoch; each view draws `N` random surface rows with replacement.
- Validation/test with `--eval-surface-points N`: each case is split into deterministic strided views so every point is evaluated exactly once.
- Metric aggregation reassembles chunked views by case before computing relative L2.

Set either point limit to `0` to load complete cases.

## Model Contract

The baseline `train.py` model interface is:

```python
out = model(x=batch.x, mask=batch.mask)
pred_norm = out["preds"]  # [B, N, 1], normalized cp space
```

`batch.mask` is required because cases are padded to the largest point count in the batch. Do not compute loss or metrics on padding tokens.

The model predicts normalized targets:

```python
y_norm = (y - y_mean) / y_std
pred_cp = pred_norm * y_std + y_mean
```

Keep this contract intact unless your PR deliberately changes both training and evaluation.

## Gradient Telemetry

The trainer intentionally logs high-fidelity gradient telemetry on every optimizer update by default. Future agents must preserve this unless the advisor explicitly asks to change the logging contract.

The W&B stream includes:

- `train/grad/*` — aggregate gradient health: global norm, RMS, mean absolute gradient, max absolute gradient, zero fraction, non-finite count, parameter norm, and grad-to-parameter norm ratio.
- `train/grad_type/<LayerType>/*` — the same statistics grouped by module class, for example `LinearProjection`, `TransolverAttention`, `LayerNorm`, or `TransformerBlock`.
- `train/grad_module/<LayerType>/<module_path>/*` — layer-by-layer statistics for every named module with trainable parameters.
- `train/grad_param/<LayerType>/<parameter_path>/*` — per-parameter statistics for every trainable tensor.
- `train/grad_hist/all` and `train/grad_hist_param/<LayerType>/<parameter_path>` — gradient histograms for distribution drift, spikes, saturation, dead layers, and collapse detection.

Keep gradient logging close to `loss.backward()` and before `optimizer.step()` so it represents the update that is about to be applied. When adding new model blocks, make sure their parameters remain visible through `named_modules()` / `named_parameters()` so the layer-type and layer-path logging stays useful. If you rename metric keys, document the migration in the PR because downstream agents compare these histories over time.

## Metrics

Primary metric is the AB-UPT-style per-case relative L2 on unnormalized `surface_cp`, percent-scaled:

```
rel_l2_case = 100 * ||pred_cp - target_cp||_2 / ||target_cp||_2
surface_rel_l2_pct = mean(rel_l2_case over cases)
```

Logged metrics:

- `val_primary/surface_rel_l2_pct` — checkpoint selection metric
- `test_primary/surface_rel_l2_pct` — final held-out result
- `val/val_surface/surface_rel_l2_pct`, `test/test_surface/surface_rel_l2_pct` — split diagnostics
- `surface_rel_l2` — same metric before percent scaling
- `loss` — normalized-space MSE, useful for optimization debugging only

Lower is better. For paper-facing reporting, use `test_primary/surface_rel_l2_pct`.

## Reference Targets

There is no exact published number for this repo's packaged `surface_cp` target and trainer path. Closest public DrivAerML surface-pressure references are low-single-digit relative-L2 percent:

- AB-UPT: `p_s = 3.82`
- Transolver baseline reported by AB-UPT: `p_s = 4.81`
- Transolver-3: `p_s = 3.71`
- Transolver++ row inside Transolver-3: `p_s = 4.12`

Treat these as external target bands, not apples-to-apples baselines.

## Constraints

- GPUs have 96 GB. Avoid OOM; point clouds are large.
- Respect `SENPAI_TIMEOUT_MINUTES`; do not override it in experiments.
- Simpler is better. A small improvement with unreadable complexity is not worth it.
- Data loaders and split files are read-only for normal research PRs.
- No new packages outside `pyproject.toml`.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. GitHub Issues are used for communication with the human researcher team. See `instructions/prompt-advisor.md` and `instructions/prompt-student.md`.
