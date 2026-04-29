# SENPAI Research Results — DrivAerML (`yi`)

W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml`.
Targets to beat (lower is better): `surface_pressure 3.82`, `wall_shear 7.29`,
`volume_pressure 6.08`, `tau_x 5.35`, `tau_y 3.65`, `tau_z 3.63`.

## Round 1 — opened 2026-04-28

Round 1 launches 16 parallel experiments: 5 known-good baselines / proven-additive
deltas (Stream 1) and 11 fresh single-delta hypotheses (Stream 2).

## 2026-04-29 02:30 — PR #12: stochastic depth / DropPath p=0.1 (nezuko) — CLOSED

- Branch: `nezuko/round1-stochastic-depth`
- Hypothesis: linear-schedule DropPath (max p=0.1 at deepest layer) regularizes the
  Transolver and gives ~10% throughput from skipped residual branches.

| Metric | nezuko (DropPath p=0.1, `mdo2p8q7`) | norman (no DropPath, `akbdunir`) | AB-UPT target |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **81.21** | 64.66 | — |
| `test_primary/surface_pressure_rel_l2_pct` | 66.49 | 48.43 | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | 84.27 | 66.89 | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | 69.42 | 55.54 | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | 75.40 | 55.54 | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | 102.42 | 90.15 | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | 92.32 | 73.66 | 3.63 |

**Result: rejected (+16.5 pp worse on abupt_axis_mean than no-DropPath).**

**Analysis:** both nezuko and norman finished with `best_epoch=1`. Train loss
keeps falling, EMA-val degrades from epoch 1 onward — the runs are firmly in
the underfitting regime, not the overfitting regime where regularization helps.
Stochastic depth adds noise to the residual signal without addressing the
binding constraint (insufficient optimizer steps to convergence at this
4L/256d/4h/128sl + 65k-points config inside the 6 h timeout).

**Important byproduct: per-step timeout fix.** nezuko shipped a `train.py` fix
(commit `1ab3a9b`) that adds a per-step wall-clock timeout check, reserves
`SENPAI_VAL_BUDGET_MINUTES` (default 90), and forces a final validation when
mid-epoch timeout fires. Cherry-picked into `yi` as commit `af92e9a` and
broadcast to all active Round-1 PRs. This unblocks every 65k-points run from
the silent "epoch longer than timeout → no test_primary" failure mode that
trapped the prior `u38zaxeg` attempt.

**Round-1 follow-ups triggered:**
- Recommend `--validation-every 1` (or 2) for all Round-1 runs.
- Flagged the train→val divergence pattern across runs for all students to
  report and investigate.
