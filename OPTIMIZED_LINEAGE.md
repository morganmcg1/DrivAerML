<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Optimized Lineage

This branch is the more opinionated DrivAerML arm. It keeps the same data contract, targets, validation/test metrics, gradient telemetry, and slope telemetry as `main`, but starts students from stronger defaults:

- 4 Transolver layers instead of 3
- 256 hidden width and 4 heads
- 128 slices
- 65,536 surface and volume points per train/eval chunk
- `torch.compile` enabled
- AdamW with lower learning rate, higher weight decay, and slower EMA decay

Use this branch when the advisor wants students to iterate from a higher-capacity baseline instead of the vanilla baseline. Do not remove `surface_pressure`, `wall_shear`, or `volume_pressure` targets, and do not weaken the full-fidelity `full_val/*` and `test/*` evaluation.

Suggested first experiments:

- Compare batch sizes 1, 2, and 3 while inspecting `train/grad/*` and `train/slope/*`.
- Compare 128 vs 96 slices to separate accuracy from throughput.
- Try target-family loss weights, but report all three MAEs and all three relative-L2 percentages.
- Keep at least one short stability run and one longer confirmation run for each serious idea.
