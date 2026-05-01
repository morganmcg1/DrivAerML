# SENPAI Research State

- **2026-04-30 ~latest** — Closed PR #76 (gilbert 5L/256d, best ep31 abupt=7.4726%) and PR #77 (haku 4L/384d, best ep31 abupt=7.6344%) — both confirmed via full W&B scan, neither beat 7.2091% baseline. gilbert reassigned → PR #188 (slices=192 + Fourier PE, orthogonal to all Wave 2). Wave 2 pods are waiting on Wave 1 train.py completion; all 9 PRs #174-#182 have 0 comments (not yet picked up by students).
- **2026-05-01 ~12:30Z (Wave 2 LAUNCHED)** — All 9 idle bengio students assigned Wave 2 PRs (#174-#182), all draft, awaiting student pickup. PR #74 (alphonse Wave 1) merged as wave leader; current baseline val_abupt=7.2091% (vol_p=4.166% beats AB-UPT target). Wave 2 cohort built on Fourier PE foundation (Wave 1's strongest signal). Test_primary eval for PR #74 (run `m9775k1v` ep30 ckpt) still pending — not blocking.
- **2026-05-01 ~10:30Z** — Wave 1 wrap: 16 bengio runs healthy. alphonse leader (val_abupt=7.296 at ep39), peaked ep31 at 7.209. PR #74 merged. Wave 2 ready.
- **2026-05-01 ~08:45Z** — PR #137 (edward GradNorm) closed: diverged val_abupt=33.43%. edward → PR #160 split-output-heads.
- torch.compile bug: PyTorch 2.x Inductor `tiling_utils.get_pw_red_splits` crashes at first validation; all bengio students use `--no-compile-model`.

## Wave 2 Cohort (LAUNCHED 2026-05-01 ~12:30Z)

All 9 PRs are built on the Wave 1 winning recipe (4L/256d + Fourier PE + no-EMA + cosine LR + lr=3e-4 + warmup=5 + DDP4) with one orthogonal change each. Goal: close the 2.79pp abupt gap (7.21% → 4.51%) and address the wsy/wsz binding constraint.

| PR | Student | Hypothesis | Tier |
|----|---------|------------|------|
| #174 | alphonse | 5L/256d + PE + T_max=50 (depth + longer schedule) | Capacity scaling |
| #175 | askeladd | 4L/384d + PE + T_max=50 (width scaling) | Capacity scaling |
| #176 | chihiro | LR sweep {1e-4, 5e-4} on 4L/256d + PE + T_max=50 | Optimization |
| #177 | frieren | Surface loss upweight {3.0, 5.0} (target wsy/wsz) | Loss rebalance |
| #178 | kohaku | 6L/256d + PE + T_max=60 (deeper depth scaling) | Capacity scaling |
| #179 | nezuko | 5L/384d + PE + lr=2e-4 + T_max=60 (joint width+depth) | Capacity scaling |
| #180 | norman | Raw rel-L2 aux loss {0.1, 0.3} (eval-aligned loss) | Loss formulation |
| #181 | thorfinn | EMA revival {0.999, 0.9995} on PE + T_max=50 | Optimization |
| #182 | violet | Volume loss downweight {0.5, 0.25} (capacity → shear) | Loss rebalance |

**Coverage map (depth × width grid)**:
- 4L/256d (Wave 1 best, baseline) — alphonse #74 merged
- 5L/256d — alphonse #174
- 6L/256d — kohaku #178
- 4L/384d — askeladd #175
- 5L/384d — nezuko #179 (joint frontier)

**Loss-rebalance pair targeting wsy/wsz binding constraint**:
- frieren #177: increase surface loss weight (push surface gradients up)
- violet #182: decrease volume loss weight (vol_p already < target; reallocate capacity to surface)

**Optimization tier**:
- chihiro #176: LR sweep
- thorfinn #181: EMA revival on longer schedule
- norman #180: raw rel-L2 aux loss (eval-metric-aligned)

**Other in-flight bengio PRs**:
- PR #145 (senku): metric-aware loss `mse_plus_raw_rel_l2` w=0.05 — running, ep~10, abupt=9.75% (Good zone, continue to ep20)
- PR #160 (edward): split surface output head (cp MLP + wall-shear MLP) — running, ep~20, abupt~8.4% and improving
- PR #188 (gilbert): 4L/256d + Fourier PE + slices=192 (new, Wave 2+ slice-count scaling; orthogonal to Wave 2 cohort)
- PRs #76/#77 (gilbert/haku): CLOSED — neither beat baseline. Full scan confirmed best 7.47%/7.63% respectively.
- PRs #75, #79-83, #85-87 (Wave 1): still running on old Wave 1 pods; watchdog waiting for train.py to complete before picking up Wave 2 assignments

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — ADVISOR responded; PR #74 merged as Wave 1 leader.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 2 includes capacity scaling, loss rebalance, EMA revival, and raw rel-L2 alignment as orthogonal directions.

AB-UPT targets to beat:
- surface_pressure_rel_l2_pct < 3.82
- wall_shear_rel_l2_pct < 7.29
- volume_pressure_rel_l2_pct < 6.08 ✓ (alphonse 4.166)
- wall_shear_x_rel_l2_pct < 5.35
- wall_shear_y_rel_l2_pct < 3.65 (5.45pp gap — binding constraint)
- wall_shear_z_rel_l2_pct < 3.63 (7.24pp gap — hardest binding constraint)
- abupt_axis_mean_rel_l2_pct ~ 4.51 (mean of 5 axis metrics)

Current bengio best: PR #74 (alphonse) val_abupt=7.2091%, 2.7pp from AB-UPT target.

## Current Research Focus and Themes

**Wave 2 (in flight) — Three strategy tiers on the Fourier PE baseline**:
- **Capacity scaling**: 5L, 6L, 384d, joint 5L/384d depth × width frontier
- **Loss rebalance**: surface upweight, volume downweight to break wsy/wsz binding constraint
- **Optimization**: LR sweep, EMA revival, raw rel-L2 aux loss

**CRITICAL GAP**: Wall-shear y/z axes are 3-4× above AB-UPT targets across ALL Wave 1 runs (best wsy=8.94, target 3.65; best wsz=10.95, target 3.63). Wave 2 frieren #177 + violet #182 target this directly via loss rebalance; Wave 3 may need physics-aware approaches if rebalance alone is insufficient.

**Physical explanation for wsy/wsz gap**: Car geometry has dominant side-flow vs. axial-flow structure. Wall-shear y (lateral) and z (vertical/normal) capture cross-flow vortices and boundary-layer separation that are harder to predict without explicit physics encoding. wsx (streamwise) aligns with primary flow direction and is easier to learn.

## Potential Next Research Directions (Wave 3+)

- **Physics-aware shear prediction**: Surface-normal-aligned coordinate systems for shear vectors; boundary-layer-aware attention with finer slices near no-slip walls.
- **Equivariant heads**: SO(3)/SE(3) heads for shear vector prediction (the coordinate frame shouldn't matter for the physics, but does for the model).
- **Loss-formulation v2**: Squared rel-L2 (penalizes large errors more), per-axis adaptive weighting, mixup on surface fields.
- **Architecture v2**: DomainLayerNorm, FiLM conditioning, multi-scale attention, dual-resolution heads (coarse for vol, fine for surf).
- **Data/training v2**: Curriculum (easy cars first, then aero-extreme cases), 96k pts, mirror TTA, SWA, ensembling top-K seeds.
- **Pretraining**: Synthetic/simplified CFD pretraining then DrivAerML fine-tune.
- **DDP8 pool**: 4L/512d/8H + EMA=0.9995 + gc=0.5 + lr=4.8e-4 + T_max=36 (radford champion port).
- **Stacking winners**: If Wave 2 confirms 5L + 384d + PE separately, Wave 3 launches the joint stack with 60k pts and 128 slices.
- **UW follow-ups (edward PR #84)**: wider clamp [-10,10], loss-scale-aware log_var init, drop regularizer.
- **Combined recipe stacking**: Wave 2 winners merged sequentially; small per-PR gains compound.
