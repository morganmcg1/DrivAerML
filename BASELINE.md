# DrivAerML Baseline Metrics

## Current Best: AB-UPT Public Reference (no experiment merged yet)

No experiments have been merged on the bengio branch yet. The baseline to beat is the AB-UPT public reference.

| Metric | Value | AB-UPT Target |
|--------|-------|--------------|
| `test_primary/surface_pressure_rel_l2_pct` | — | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | — | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | — | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | — | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | — | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | — | 3.63 |
| `test_primary/abupt_axis_mean_rel_l2_pct` | — | ~4.51 (mean of 5 axis metrics) |

## Historical Best from radford branch (PR #2593)

The strongest known DrivAerML result from prior research is the 4L/256d + no-EMA + Fourier features + T_max=30 recipe from radford PR #2593, which achieved approximately 12.96% on `abupt_axis_mean_rel_l2_pct`.

Note: The radford PR result still needs to beat the AB-UPT targets above. These reference the bengio branch's quest to decisively beat those targets.

## Update Log

- 2026-04-30: Branch initialized. No experiments merged yet. AB-UPT reference values are the targets.
