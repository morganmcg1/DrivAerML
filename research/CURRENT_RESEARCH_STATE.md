# SENPAI Research State

- **2026-05-03 06:30Z** — Session status update. Gate comments posted on fern #409 (EP25 continue, best ep24=7.16%), tanjiro #443 (EP20 plateau watch, best=8.00%), haku #462 (EP15 kill warning, projection ~9.5%), norman #465 (EP15 borderline kill warning, ep12=8.84%, plateau on wsy/wsz). chihiro #407 force-closed (ep24=7.69%, ceiling at ~7.59%, will not beat baseline). 4 idle students need assignment: chihiro (freshly vacated), emma, gilbert, violet (zombie cleanup pending). Stark still has no pod (infra blocker).
- **2026-05-03 05:52Z** — Session status update. 2 new Wave 19 PRs assigned (senku #497, nezuko #498). BASELINE.md n_params corrected. Key leads: fern #409 (7.45% ep18), thorfinn #382 (7.59% ep11), edward #483 cross-attn (8.007% ep7, accelerating).
- **Most recent human researcher direction**: Issue #466 still tracking zombie pod SIGKILL needs (emma/gilbert/violet blocked). No new research directives from human team since Issue #18 (yi): "Ensure you're really pushing hard on new ideas".

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.
**Current best (MERGED baseline)**: alphonse PR #174, val_abupt = **6.9549%** (run `vu4jsiic`, ep~45.3, step 807,025). 5L/256d + FourierEmbed + T_max=50. n_params=3,992,313.

**Binding unsolved constraints**: wsy=8.7345% (target 3.65%: **5.08pp gap**), wsz=10.5766% (target 3.63%: **6.95pp gap**).

## Active PRs (in flight, 13 active)

| PR | Student | Wave | W&B run | abupt now | Notes |
|----|---------|------|--------|----------:|-------|
| #409 | fern | W11 | `hph6eaky` | **7.16 ep24** | LEADING. coord-norm fix on 5L/256d+FourierPE+T_max=50. ep25=7.29 (uptick). Best=ep24=7.1624%, gap to baseline=0.21pp. Continue to ep50 — still on convergence trajectory. |
| #382 | thorfinn | W9 | `5ifnf1wc` | **7.54 ep12** | 6L/512d/8H. ep12=7.5445%, monotonic descent (ep5=8.25→ep12=7.54). EP15 imminent. |
| #417 | kohaku | W11 | `4632xosf` | ~7.60 | EMA on 5L/256d. ep~20. Unlikely to beat baseline at ep25 (projected 7.3%). |
| ~~#407~~ | chihiro | W14 | `jmbe8hys` | **CLOSED** | Stacked recipe. ep24=7.69%, best ep23=7.59%. Force-closed at 06:30Z — will not beat baseline. |
| #437 | alphonse | W13 | `0xi2n4oo` | ~8.15 est | 6L/256d. ep~19 (no update since ep15=8.35%). EP25 ETA ~11:00Z. |
| #443 | tanjiro | W14 | `vyhpqruv` | **8.00 ep20** | mirror-aug+SW=2.0. Best ep20=7.998%, plateau. EP25 hard gate: <7.5% to continue. |
| #465 | norman | W15 | `o6zxx2uq` | **8.84 ep12** | model-slices=128. EP10 best=8.78. Plateau ep10-12. EP15 borderline kill (threshold 8.5%). Trial B (slices=192) queued. |
| #454 | frieren | W17 | `l8nu1ajz` | 7.921 ep9 | Per-axis tau_yz weight=1.5. Running to EP12. Unlikely to beat SOTA. |
| #462 | haku | W15 | `f9nkv7p9` | **9.86 ep12** | Surface-2x/Volume-0.5x. EP15 KILL likely (projection ~9.5–9.7% vs threshold 8.5%). |
| #468 | edward | W16 | `6b08y222` LR=5e-4 | ep~0.9 (no val) | Muon optimizer retry. Too early for gate. EP5 gate ETA ~3-4h. |
| #483 | edward | W17 | `ok98szul` | 8.007 ep7 | Cross-attn bridge v4. EP7 slope ACCELERATED (−0.646/ep). Projected EP12 at SOTA boundary (~7.40-7.55%). |
| #488 | alphonse | W17 | `ki2q9ko9` | 7.978 ep7 | Multi-sigma STRING-sep freq init. Tau_y/z improving fastest. Projected EP12 ~7.40-7.55% (SOTA boundary). |
| #489 | thorfinn | W17 | `r5rw40rn` | 30.2 ep1 | Volume curriculum 16k→65k. Very early — coarse stage, high val expected. |
| #495 | askeladd | W18 | `ky4rf6g5` | 15.30 ep1 | CoordConv dist-to-surface feature. ep1 elevated vs fern/tanjiro (~14%). EP5 watch. |
| #361 | frieren | W9 | Trial B `e1kxrd6b` | 14.998 ep1 | wd=1e-3 sweep. ep~1.3, only one val checkpoint. EP5 gate first. |
| #497 | senku | W19 | — | — | Stacked coord-norm+mirror-aug+SW=2.0. Assigned 05:52Z. ACK pending (34 min into 60 min window). |
| #498 | nezuko | W19 | — | — | T_max sweep 70 vs 100. Assigned 05:52Z. ACK pending (34 min into 60 min window). |

**Idle students (need fresh assignments next loop):**
- chihiro: freshly vacated after #407 force-close (06:30Z).
- emma, gilbert, violet: previously zombie-blocked. Pods 1/1 ready per kubectl. Pre-staged hypotheses: deep supervision (emma), best-ckpt EMA-soup (gilbert), trunk-split decoder (violet). Reassign in next cycle.
- stark: NO POD exists in kubectl deployments. Infrastructure blocker; human team must provision.

## Key Insights (consolidated)

1. **Coord-norm fix** (fern #409) is the strongest single architectural lever yet — currently leading all active runs at 7.45% ep18.
2. **6L/512d/8H capacity** (thorfinn #382) is the strongest architecture change — 7.59% ep11 with wsy dropping 0.40pp/ep (binding axis).
3. **T_max=50** validated as +0.25pp vs T_max=30 (baseline PR #174). T_max=70/100 sweep now testing.
4. **Mirror-aug + SW=2.0** directly moves wsy — it's the best targeted binding-axis lever.
5. **Stacking coord-norm + mirror-aug + SW=2.0** (senku #497) is the next logical compound step.
6. **EMA is inferior to no-EMA** on this architecture — kohaku #417 confirms EMA hurts vol_p by 1.0pp vs coord-norm baseline at ep17.
7. **Progressive EMA** (fern #480) CLOSED — does not beat no-EMA.
8. **log1p tau-norm** (tanjiro #481) CLOSED NEGATIVE — scale-remapping does not help wsy/wsz; the issue is representation, not scale variance.
9. **vol_p has been solved** (fern ep18: 4.29%, well below AB-UPT target 6.08%). wsy/wsz is the universal binding bottleneck.
10. **Cross-attention bridge** (edward #483) showing unexpected acceleration at EP7 (8.007%, slope −0.65pp/ep) — high-potential trajectory, monitoring closely.
11. **Muon optimizer** at LR=1e-3 is too aggressive for 5L/256d. Retry at LR=5e-4 ongoing.
12. **6L/256d depth extension** (alphonse #437) is slower to converge than 5L. At ep15=8.35%, it trails both coord-norm-5L and 6L-512d-8H.
13. **Weight decay** sweep (frieren #361): Trial A (wd=3e-4, ep30=7.833%) well above baseline. Trial B (wd=1e-3) running.
14. **Cross-pod contamination** from yi-track is a recurring false-positive source. Always verify `metadata.host` before flagging.

## Immediate Gate Decisions Pending

| PR | Student | Gate | ETA | Action needed |
|----|---------|------|-----|---------------|
| #462 | haku | EP15 kill (<8.5%) | ~09:00Z | EP12=9.86%; KILL likely. Pre-staged. |
| #465 | norman | EP15 borderline kill (<8.5%) | ~07:30Z | EP12=8.84%; plateau. Trial B (slices=192) queued. |
| #443 | tanjiro | EP25 hard gate (<7.5%→continue) | ~10:30Z | EP20=8.00%, plateau. |
| #495 | askeladd | EP5 (<13%) | ~08:30Z | EP1=15.30% (slightly elevated). |
| #409 | fern | EP30+ (continue trajectory) | ~10:00Z | Best=7.16% ep24, continue to ep50. |
| #382 | thorfinn | EP15 (<8.5%) | ~10:35Z | EP12=7.54% — easy pass. |
| #468 | edward | EP5 Muon retry (<9.5%→kill) | ~12:00Z | ep~0.9, no val data yet. |
| #361 | frieren | EP5 wd=1e-3 (<13%) | ~08:30Z | ep1=14.998%. |
| #417 | kohaku | EP25 (<7.0%→kill,<6.9549%→review) | ~12:00Z | Monitor |
| #437 | alphonse | EP25 (<8.0%→continue,<6.9549%→review) | ~11:00Z | Monitor |

## Potential Next Research Directions (Wave 19/20+)

1. **Stacked best-of-wave recipe**: coord-norm + mirror-aug + SW=2.0 (senku #497, assigned) — if wins, then add top Wave 17 finding.
2. **Full-budget 6L/512d/8H replay**: thorfinn #382 is the leading architecture. If EP15 projection holds (~6.92%), schedule a 50-epoch full-budget DDP8 run to confirm.
3. **Equivariant shear heads**: SE(3)/SO(3) equivariant outputs for wsy/wsz — the most physics-motivated path to closing the 5-7pp gap.
4. **Trunk-split surface decoder**: split TRUNK not head for surface vs volume prediction — nezuko #347 showed head-split backfires; trunk-split is the mechanistically-motivated fix.
5. **Test-time augmentation**: mirror + rotation averaging at inference. Zero training cost; wsy flip is sign-invariant under y-mirror, wsz is invariant. Free ensemble signal.
6. **Best-checkpoint ensemble**: average late-epoch checkpoints from fern/thorfinn/kohaku — negligible effort, potential +0.2pp.
7. **OOD geometry test sweep**: confirm val→test gap on all top-5 val runs before claiming AB-UPT wins.
8. **Researcher-agent sweep**: invoke to generate Wave 20 candidates targeting the wsy/wsz 5-7pp binding gap with architectural + loss + data solutions.
9. **Deep supervision**: auxiliary wsy/wsz prediction losses at intermediate transformer layers — Schmidhuber-style gradient injection at layers 2, 3, 4.
10. **Larger batch size / gradient accumulation**: test if current 4-GPU DDP underfits on complex geometries.

## Targets

| Metric | Current Best (val) | AB-UPT Target | Gap |
|--------|--------------------|---------------|-----|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **6.9549** (alphonse PR #174) | 4.51 | 2.44pp |
| `val_primary/surface_pressure_rel_l2_pct` | 4.5644 | 3.82 | 0.74pp |
| `val_primary/volume_pressure_rel_l2_pct` | 3.9361 ✓ | 6.08 | beats target |
| `val_primary/wall_shear_y_rel_l2_pct` | 8.7345 | 3.65 | **5.08pp** |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.5766 | 3.63 | **6.95pp** |

**val/test gap warning**: ~2x degradation on vol_p confirmed. Test confirmation required before claiming AB-UPT wins.

## Constraints (hard)

- `--no-compile-model`: Mandatory (PyTorch 2.x Inductor crash at validation)
- `--fourier-pe`: Mandatory for comparability (n_params=3,992,313 for 5L/256d)
- `--coord-norm`: Required for all coord-norm experiments (fern baseline family)
- Kill-threshold operator: `< VALUE` means kill if metric NOT below VALUE (≥ VALUE)
- Standard gate schedule: ep5, ep10, ep15, ep25, ep50 (epoch budget per assignment varies)
- Correct grad-clip flag: `--grad-clip-norm` (NOT `--grad-clip`)

## Compliance Watch

| Student | Status |
|---------|--------|
| haku | 3rd off-script offense (jbbw3enm, cross-pod contamination confirmed). Current PR #462 on-script. |
| frieren #361 | Trial A killed at ep30 (CosineAnnealingLR oscillation found); Trial B running. Technically compliant — early stop was the right call. |
| All prior "off-script" flags | **WITHDRAWN** — confirmed cross-pod yi contamination via metadata.host checks. |
