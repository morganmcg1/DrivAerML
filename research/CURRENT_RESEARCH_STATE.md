# SENPAI Research State

- **2026-05-01 17:40Z (Wave 3 reassignment round)** — After two escalations with zero student response on PRs #215 senku, #216 askeladd, #217 edward, #219 haku, #220 kohaku — all five closed. Five fresh Wave 3 hypotheses assigned: PR #234 senku (mirror-symmetry TTA — free wsy gain via y-flip averaging), PR #235 askeladd (4L/512d/8H radford champion port — width frontier untried on bengio), PR #236 edward (fixed wsy×3/wsz×5 channel multipliers — simplest possible wsy/wsz attack), PR #237 haku (squared rel-L2 aux loss — focal-loss-equivalent for hard-sample focusing), PR #238 kohaku (high-shear curriculum oversampling with linear anneal — orthogonal data-axis lever). All five include the corrected kill threshold `35000:...<20` and explicit gates. 11 active bengio PRs continue (Wave 2 + Wave 3 originals + Wave 1 leftovers).
- **2026-05-01 16:14Z (Active monitoring)** — 16 bengio WIP PRs, 0 review-ready, 0 idle. **PR #174 (alphonse) escalated TWICE** — 13:12Z kill instruction unanswered, 15:36Z first escalation unanswered, 16:05Z second escalation posted (default replacement: 8L/256d + T_max=30 + EMA off + b1=0.95; 30min response deadline = 16:35Z). Wave 2 active runs healthy: chihiro `ld3ff1gs` (lr=5e-4, ep5=11.05%), nezuko `ud5iddlc` (5L/384d, ep4=11.21%), norman `1rieq278` (raw rel-L2 w=0.1, ep5=10.85%), thorfinn `scefipy4` (EMA off + b1=0.95 + T_max=50, launched 13:21Z), emma `3evzgru1` (60k pts + T_max=50, ep5=10.799%, sent back with new gates ep15<9.0/ep20<8.0/ep30<7.5). Wave 3 students (#214-220) all received advisor check-ins 14:47-14:53Z with kill-threshold fix; #218 frieren launched 15:22Z (TangentFrameHead w/ Frisvad-Duff), #221 violet Run A (tau=1.0) launched 15:38Z ep1=15.36%. #214/215/217/219/220 not yet confirmed launched (~85min since check-in, borderline). **Critical baseline correction (frieren PR #218 audit)**: alphonse Wave 1 winner `m9775k1v` actually used `ContinuousSincosEmbed`, NOT FourierEmbed — PR #74 merged the assignment commit only, no model code. Chihiro PR #176 implemented canonical FourierEmbed; advisor approved 15:21Z; askeladd/nezuko cherry-picked. So Wave 1 best (7.21%) was 4L/256d + SincosEmbed + no-EMA + T_max=30 — not Fourier. Edward PR #160 split-heads run `09kojb6q` reached 7.620% at ep28.5 — best non-alphonse Wave 1/2 result.
- **2026-05-01 10:30Z (Active monitoring)** — 16 bengio WIP PRs all queued or running; 0 review-ready, 0 idle. Stream-1 finishers (emma #79 Trial A, fern #75 Trial A, tanjiro #80 SW=2.0) all hitting cosine warm-restart bounce — best-val checkpoints locked at ep30-ep32. Wave-2 PRs #174-#182, #188, #190 queued behind active runs (watchdog correctly waits for current train.py to exit). Senku #145 metric-aware aux loss at ep10=9.75% with **vol_p=5.97% — first cohort axis to clear AB-UPT target (6.08%)**. Edward #160 split-heads at ep20=7.961%, projected ep30 ~7.30% (borderline vs alphonse 7.21% baseline). Emma Trial B (`nlrs16c6`, T_max=50 monotone) launched 10:24Z. Fern Trial B fires ~10:50Z via auto-launcher.
- **2026-05-01 ~07:50Z (Wave 2 LAUNCHED)** — All 9 idle bengio students assigned Wave 2 PRs (#174-#182), draft, queued behind stream-1. Plus #188 gilbert (slices=192) and #190 haku (radford champion 4L/512d/8H DDP8). PR #74 (alphonse Wave 1) merged as wave leader; current baseline val_abupt=7.2091% (vol_p=4.166% beats AB-UPT target). Test_primary eval for PR #74 (run `m9775k1v` ep30 ckpt) still pending — not blocking.
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

**Other in-flight bengio PRs (Wave 1 leftovers / re-assignments)**:
- PR #145 (senku): metric-aware loss `mse_plus_raw_rel_l2` w=0.05 — running, ep~4
- PR #160 (edward): split surface output head (cp MLP + wall-shear MLP) — running
- PR #75-77, #79-83, #85-87 (Wave 1, status:wip but should be reviewed once they hit ep50 / they're nearing completion)

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
