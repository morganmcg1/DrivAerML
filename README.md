<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# DrivAerML

This repo contains a self-contained DrivAerML benchmark package for the [senpai](https://github.com/wandb/senpai) autoresearch harness.

It is seeded from the DrivAerML target inside [`morganmcg1/icml2026`](https://github.com/morganmcg1/icml2026), but trimmed down to the same kind of shape as [`morganmcg1/TandemFoilSet-Balanced`](https://github.com/morganmcg1/TandemFoilSet-Balanced): one main `train.py`, a small `data/` package, pinned split metadata, and agent-facing instructions.

## Benchmark

DrivAerML is an automotive CFD surrogate dataset. The packaged target predicts surface and volume quantities:

- Surface input: `[x, y, z, nx, ny, nz, area]`
- Volume input: `[x, y, z, sdf]`
- Surface targets: `surface_cp` plus 3-channel `surface_wallshearstress`
- Volume target: `volume_pressure`
- Split: public processed `400 train / 34 val / 50 test`
- Internal checkpoint scalar: mean per-case relative L2 across the AB-UPT-aligned scalar columns `p_s`, `tau_x`, `tau_y`, `tau_z`, and `p_v`, logged as `abupt_axis_mean_rel_l2_pct`

The validation scalar used for checkpoint selection is:

```
val_primary/abupt_axis_mean_rel_l2_pct
```

The final held-out test scalar is:

```
test_primary/abupt_axis_mean_rel_l2_pct
```

Lower is better. Final reports should include the individual AB-UPT comparison columns and the target MAEs, not only this aggregate checkpoint scalar.

## Files Layout

```
.
├── README.md
├── program.md                 # research contract; read this first
├── train.py                   # grouped surface/volume Transolver trainer
├── instructions/
│   ├── prompt-advisor.md
│   └── prompt-student.md
└── data/
    ├── __init__.py
    ├── loader.py              # DrivAerMLCaseStore, DrivAerMLSurfaceDataset, pad_collate
    ├── generate_manifest.py   # regenerate split_manifest.json from PVC manifests
    ├── preload.py             # validate arrays and write point-count cache
    ├── split_manifest.json    # pinned split definition
    └── split_utils.py
```

## Data

The processed DrivAerML arrays are expected on the PVC at one of:

```
/mnt/pvc/Processed/drivaerml_processed
/mnt/new-pvc/Processed/drivaerml_processed
```

You can also point to another copy:

```
python train.py --data-root /path/to/drivaerml_processed
```

The processed root should contain case directories such as `run_1/`, each with:

```
surface_xyz.npy
surface_normals.npy
surface_area.npy
surface_cp.npy
surface_wallshearstress.npy
volume_xyz.npy
volume_sdf.npy
volume_pressure.npy
normalizers.json
```

Regenerate the split manifest from the PVC manifests:

```
python data/generate_manifest.py
```

Validate all case arrays and write point counts:

```
python data/preload.py
```

If the PVC is mounted somewhere other than `/mnt/pvc` or `/mnt/new-pvc`, set:

```
export PVC_MOUNT_PATH=/your/mount
```

## Training

```
python train.py --epochs 50 --agent <name> --wandb_name "<name>/<experiment>"
```

Defaults are point-limited for memory while still full-fidelity for validation/test:

- `--train-surface-points 40000` and `--train-volume-points 40000` sample random points per training view
- `--eval-surface-points 40000` and `--eval-volume-points 40000` evaluate deterministic strided chunks that cover every point exactly once
- `--validation-every 10` validates sparsely, with an extra epoch-1 and final-epoch validation
- `--compile-model` is on by default
- `--slope-log-fraction 0.05` logs key curve slopes every 5% of the estimated update budget
- `--kill-thresholds "500:train/loss<5,2000:val_primary/abupt_axis_mean_rel_l2_pct<25"` stops a poor run early once a logged metric misses a step-gated threshold

Gradient telemetry is intentionally high fidelity: aggregate norms, layer/type/parameter stats, and histograms are logged around every optimizer update by default.

Training sampling is with replacement inside each random view. For a case with `N` points and `K` points per view, an epoch creates `ceil(N / K)` views and draws `K` random rows for each view, so it draws about `N` rows per case per epoch but may contain duplicates and omissions. Validation and test use deterministic chunks without replacement across views, so a full eval covers every available point exactly once.

AB-UPT comparison metrics are logged separately. Use `test_primary/surface_pressure_rel_l2_pct`, `test_primary/wall_shear_rel_l2_pct`, `test_primary/wall_shear_x_rel_l2_pct`, `test_primary/wall_shear_y_rel_l2_pct`, `test_primary/wall_shear_z_rel_l2_pct`, and `test_primary/volume_pressure_rel_l2_pct` for paper-aligned comparisons. `abupt_axis_mean_rel_l2_pct` is only a checkpointing/triage scalar, not a published AB-UPT table column.

The baseline keeps target normalization to train-split mean/std. `surface_cp` is already nondimensionalized; do not add guessed per-case `p / Re^2` rescaling unless verified per-case freestream/Reynolds metadata is plumbed through and the final AB-UPT-style metrics are still computed on the original target units.

Environment:

- `SENPAI_TIMEOUT_MINUTES` — wall-clock cap, default `30`
- `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE` — W&B routing

## Citation

For the underlying dataset, cite the official DrivAerML dataset paper:
<https://arxiv.org/abs/2408.11969>
