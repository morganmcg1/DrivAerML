# SENPAI Research State

- **2026-05-03 05:52Z** — Session status update. 2 new Wave 19 PRs assigned (senku #497, nezuko #498). BASELINE.md n_params corrected. Key leads: fern #409 (7.45% ep18), thorfinn #382 (7.59% ep11), edward #483 cross-attn (8.007% ep7, accelerating).
- **Most recent human researcher direction**: Issue #466 still tracking zombie pod SIGKILL needs (emma/gilbert/violet blocked). No new research directives from human team since Issue #18 (yi): "Ensure you're really pushing hard on new ideas".

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.
**Current best (MERGED baseline)**: alphonse PR #174, val_abupt = **6.9549%** (run `vu4jsiic`, ep~45.3, step 807,025). 5L/256d + FourierEmbed + T_max=50. n_params=3,992,313.

**Binding unsolved constraints**: wsy=8.7345% (target 3.65%: **5.08pp gap**), wsz=10.5766% (target 3.63%: **6.95pp gap**).

## Active PRs (in flight, 13 active)

| PR | Student | Wave | W&B run | abupt now | Notes |
|----|---------|------|--------|----------:|-------|
| #409 | fern | W11 | `hph6eaky` | **7.452** | LEADING. coord-norm fix on 5L/256d+FourierPE+T_max=50. ep18 (step 320k). EP25 ETA ~08:00Z, projected 6.88% — baseline beat likely. |
| #382 | thorfinn | W9 | `5ifnf1wc` | **7.590** | 6L/512d/8H. ep11. EP15 ETA ~10:35Z, projected ~6.92% at ep15 — may beat baseline. Strongest wsy signal (-0.40pp/ep). |
| #417 | kohaku | W11 | `4632xosf` | ~7.60 | EMA on 5L/256d. ep~20. Unlikely to beat baseline at ep25 (projected 7.3%). Cosine late-kick could help. |
| #407 | chihiro | W14 | `jmbe8hys` | ~7.9 est | Stacked recipe. ep~19-20 (no update since ep10=8.42%). Nudge sent. |
| #437 | alphonse | W13 | `0xi2n4oo` | ~8.15 est | 6L/256d. ep~19 (no update since ep15=8.35%). EP25 ETA ~11:00Z. |
| #443 | tanjiro | W14 | `vyhpqruv` | ~7.8 est | mirror-aug+SW=2.0. ep~20 (EP18=8.0075% last confirmed). EP20 gate overdue — nudge sent. |
| #465 | norman | W15 | `o6zxx2uq` | 8.78 | model-slices=128. ep10 PASS. Running to ep30. Trial B (slices=192) launches after Trial A. |
| #454 | frieren | W17 | `l8nu1ajz` | 7.921 ep9 | Per-axis tau_yz weight=1.5. Running to EP12. Unlikely to beat SOTA. |
| #462 | haku | W15 | `f9nkv7p9` | ~9.5 est | Surface-2x/Volume-0.5x. ep~9. EP5 gate overdue — nudge sent. |
| #468 | edward | W16 | retry `LR=5e-4` | early | Muon optimizer. LR=1e-3 failed at EP5 gate (11.37%). Retry at LR=5e-4 launched. |
| #483 | edward | W17 | `ok98szul` | 8.007 ep7 | Cross-attn bridge v4. EP7 slope ACCELERATED (−0.646/ep). Projected EP12 at SOTA boundary (~7.40-7.55%). |
| #488 | alphonse | W17 | `ki2q9ko9` | 7.978 ep7 | Multi-sigma STRING-sep freq init. Tau_y/z improving fastest. Projected EP12 ~7.40-7.55% (SOTA boundary). |
| #489 | thorfinn | W17 | `r5rw40rn` | 30.2 ep1 | Volume curriculum 16k→65k. Very early — coarse stage, high val expected. |
| #495 | askeladd | W18 | `ky4rf6g5` | early | CoordConv dist-to-surface feature. Launched 05:30Z. EP5 ETA ~08:00Z. |
| #361 | frieren | W9 | `totote1p`→Trial B `e1kxrd6b` | 7.833 Trial A ep30 | Weight-decay sweep. Trial A done (wd=3e-4, not better than baseline). Trial B (wd=1e-3) launched at 05:52Z. ETA ~19:50Z. Trial C (wd=3e-3) queued. |
| #497 | senku | W19 | — | — | Stacked coord-norm+mirror-aug+SW=2.0. Assigned 05:52Z. ACK pending. |
| #498 | nezuko | W19 | — | — | T_max sweep 70 vs 100. Assigned 05:52Z. ACK pending. |

**Zombie-blocked (awaiting human SIGKILL on Issue #466):**
- emma, gilbert, violet: pods occupied by zombie runs from Wave 7/8. Cannot be assigned until human team kills the processes.

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
| #462 | haku | EP5 (abupt>13%→kill) | OVERDUE (3h) | Check student update; kill if >13% |
| #443 | tanjiro | EP20 (abupt≤8.0%) | OVERDUE (30min) | Check ep19/20 metrics |
| #495 | askeladd | EP5 (abupt>13%→kill) | ~08:00Z | Monitor |
| #409 | fern | EP25 (<6.9549%→review) | ~08:00Z | Monitor; MERGE if beats baseline |
| #382 | thorfinn | EP15 (<8.5%) | ~10:35Z | Already at 7.59% ep11 — likely pass |
| #437 | alphonse | EP25 (<8.0%→continue,<6.9549%→review) | ~11:00Z | Monitor |
| #417 | kohaku | EP25 (<7.0%→kill,<6.9549%→review) | ~12:00Z | Monitor |

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
