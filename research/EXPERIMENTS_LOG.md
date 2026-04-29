# SENPAI Research Results — DrivAerML (`yi`)

W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml`.
Targets to beat (lower is better): `surface_pressure 3.82`, `wall_shear 7.29`,
`volume_pressure 6.08`, `tau_x 5.35`, `tau_y 3.65`, `tau_z 3.63`.

## Round 1 — opened 2026-04-28

Round 1 launches 16 parallel experiments: 5 known-good baselines / proven-additive
deltas (Stream 1) and 11 fresh single-delta hypotheses (Stream 2).

## 2026-04-29 03:57 — PR #9: volume loss weight sweep (gilbert) — MERGED, NEW yi BASELINE

- Branch: `gilbert/round1-volume-loss-reweight`
- Hypothesis: upweight volume loss to 2.0–3.0 to focus gradient budget on
  the hardest target (`volume_pressure`).
- Run A (vol_w=2.0): `y2gigs61`, state=finished, 6 epochs reached, best_epoch=3.
- Run B (vol_w=3.0): `s45dwv6i`, state=finished, 6 epochs reached, best_epoch=1.

**test_primary/* (Run A new yi best vs prior PR #11 baseline):**

| Metric | Run A (vol_w=2.0) | PR #11 (kohaku, prior) | Δ | AB-UPT |
|---|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **17.39** | 35.12 | **−50.5%** | — |
| `surface_pressure_rel_l2_pct` | 11.07 | 10.07 | +9.9% | 3.82 |
| `wall_shear_rel_l2_pct` | **18.32** | 43.05 | **−57.4%** | 7.29 |
| `volume_pressure_rel_l2_pct` | 15.21 | 14.99 | +1.5% | 6.08 |
| `wall_shear_x_rel_l2_pct` | **15.65** | 30.85 | **−49.3%** | 5.35 |
| `wall_shear_y_rel_l2_pct` | **21.86** | 42.06 | **−48.0%** | 3.65 |
| `wall_shear_z_rel_l2_pct` | **23.18** | 77.65 | **−70.1%** | 3.63 |

Run B (vol_w=3.0): `abupt=30.08`, diverged at epoch 2 (best_epoch=1).
**vol_w=3.0 strictly worse than vol_w=2.0**, confirming the PR's question.

**The big confound:** gilbert's run did **not** include
`--use-tangential-wallshear-loss` (kohaku's projection code is on yi but
default off). Yet still beat kohaku's projection-loss run by 50%. The bulk
of the win came from the **protocol fixes**:

- `--batch-size 8` (vs default 2)
- `--validation-every 1` (vs default 10)
- `--gradient-log-every 100 --weight-log-every 100` (Issue #19 throughput)

vol_w=2.0 vs vol_w=1.0 single-delta is therefore untested, but vol_w=2.0
appears at worst neutral. Combining gilbert's config with kohaku's
projection should compose for further gains.

**Critical bug uncovered (gilbert PR comment):** `train.py` has no gradient
clipping. Run B and several other Round-1 runs (chihiro, emma, fern, haku)
diverged on the exact same mechanism. **Round-2 follow-up PR #22 (gilbert)
adds `torch.nn.utils.clip_grad_norm_` + sweeps clip values.**

**Round-2 follow-ups triggered:**
- PR #22 (gilbert): add gradient clipping to `train.py` — infrastructure
  win blocking high-LR / high-weight / high-batch sweeps.
- BASELINE.md: new winning reproduce config recorded with all four protocol
  flags + vol_w=2.0.

## 2026-04-29 03:13 — PR #11: tangential wall-shear projection loss (kohaku) — MERGED, prior baseline (superseded by PR #9)

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
