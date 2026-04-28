<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# DrivAerML

This repo contains a self-contained DrivAerML surface-pressure benchmark package for the [senpai](https://github.com/wandb/senpai) autoresearch harness.

It is seeded from the DrivAerML target inside [`morganmcg1/icml2026`](https://github.com/morganmcg1/icml2026), but trimmed down to the same kind of shape as [`morganmcg1/TandemFoilSet-Balanced`](https://github.com/morganmcg1/TandemFoilSet-Balanced): one main `train.py`, a small `data/` package, pinned split metadata, and agent-facing instructions.

## Benchmark

DrivAerML is an automotive CFD surrogate dataset. The packaged target here is surface-first:

- Input: surface point cloud features `[x, y, z, nx, ny, nz, area]`
- Target: `surface_cp`
- Split: public processed `400 train / 34 val / 50 test`
- Primary metric: mean per-case full-surface relative L2 on unnormalized `surface_cp`, logged as `surface_rel_l2_pct`

The validation scalar used for checkpoint selection is:

```
val_primary/surface_rel_l2_pct
```

The final held-out test scalar is:

```
test_primary/surface_rel_l2_pct
```

Lower is better.

## Files Layout

```
.
├── README.md
├── program.md                 # research contract; read this first
├── train.py                   # surface Transolver trainer; primary editable entrypoint
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

Defaults are point-limited for practicality:

- `--train-surface-points 40000` samples random surface points per view during training
- `--eval-surface-points 40000` evaluates deterministic strided views that cover every point exactly once

Set either value to `0` for full-case loading.

Environment:

- `SENPAI_TIMEOUT_MINUTES` — wall-clock cap, default `30`
- `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE` — W&B routing

## Citation

For the underlying dataset, cite the official DrivAerML dataset paper:
<https://arxiv.org/abs/2408.11969>
