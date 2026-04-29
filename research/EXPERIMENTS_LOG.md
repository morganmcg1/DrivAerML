# SENPAI Research Results — DrivAerML (`yi`)

W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml`.
Targets to beat (lower is better): `surface_pressure 3.82`, `wall_shear 7.29`,
`volume_pressure 6.08`, `tau_x 5.35`, `tau_y 3.65`, `tau_z 3.63`.

## Round 1 — opened 2026-04-28

Round 1 launches 16 parallel experiments: 5 known-good baselines / proven-additive
deltas (Stream 1) and 11 fresh single-delta hypotheses (Stream 2).

## 2026-04-29 03:13 — PR #11: tangential wall-shear projection loss (kohaku) — MERGED, FIRST yi BASELINE

- Branch: `kohaku/round1-tangential-wallshear-loss`
- Hypothesis: project predicted/target wall-shear onto surface tangent plane
  before MSE — physics says wall shear has zero normal component on a no-slip
  wall, so penalising the normal component is unphysical noise.
- W&B run: `uy0ds6iz` (state=finished, 1 full epoch reached, run pre-dated
  the per-step timeout fix so timed out at the inter-epoch check).

| Metric | kohaku (PR #11) | norman (akbdunir, no-projection comparator) | nezuko (mdo2p8q7, DropPath) | AB-UPT |
|---|---:|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **35.12** | 64.66 | 81.21 | — |
| `test_primary/surface_pressure_rel_l2_pct` | 10.07 | 48.43 | 66.49 | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | 43.05 | 66.89 | 84.27 | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | 14.99 | 55.54 | 69.42 | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | 30.85 | 55.54 | 75.40 | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | 42.06 | 90.15 | 102.42 | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | 77.65 | 73.66 | 92.32 | 3.63 |

**Result: merged (~46% reduction on `abupt_axis_mean` vs the closest comparator).**

**Key wins:**
- First yi baseline established. All future PRs measured against PR #11.
- kohaku's deviation from the PR pseudocode was correct: PR text projected in
  normalized space, but per-axis wall-shear stds are non-uniform
  ([2.08, 1.36, 1.11]), so true tangential projection requires
  denormalize → project → renormalize. Physically motivated and analytically
  rigorous.
- New diagnostic `train/wallshear_pred_normal_rms` instruments the predicted
  normal component — confirmed it grows ~2.4× during a single epoch
  (0.52 Pa → 1.21 Pa), validating the predicted failure mode.

**Caveats:**
- Only 1 epoch reached (run pre-dated the per-step timeout fix). Subsequent
  PRs with the fix + `--validation-every 1` should reach 4–5 epochs.
- All wall-shear axes still 5–21× from AB-UPT targets — most headroom is in
  the wall-shear regression, especially `tau_z` (77.65% vs target 3.63%).

**Round-1 follow-up assigned to kohaku (PR #21):** sweep
`λ * mean((ws_pred · n_hat)^2)` regularizer on top of projection — directly
addresses the failure mode the diagnostic exposed. Also serves as the first
multi-epoch run with projection on (the λ=0 arm).

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
