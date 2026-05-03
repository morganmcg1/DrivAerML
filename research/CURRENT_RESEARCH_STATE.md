# SENPAI Research State

- **2026-05-03 04:35Z — All 16 students busy, 0 idle, 0 PRs ready for review.** Wave 17 (#454,#458,#471,#480,#481,#483,#488,#489) and Wave 18 (#493,#494,#495) all in flight. Latest live W&B snapshot below.
- **2026-05-03 04:35Z — Live cohort snapshot (W&B `senpai-v1-drivaerml`, val_primary keys)**:

| Run | PR | Student | Step | abupt | sp | vp | wsy | wsz | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `hph6eaky` | #409 | fern | 392k | **7.327** | 4.855 | 4.283 | 9.469 | 10.993 | LEADING. coord-norm fix on 5L/256d+FourierPE+T_max=50. ep~22 ($\approx 17.8$k steps/ep). |
| `5ifnf1wc` | #382 | thorfinn | 197k | 7.590 | 4.836 | 5.211 | 9.452 | 11.201 | 6L/512d/8H. Strongest capacity-scale run. ep~11. |
| `4632xosf` | #417 | kohaku | 370k | 7.605 | 5.054 | 5.300 | 9.301 | 11.117 | EMA on 5L/256d+FourierPE+T_max=50. |
| `jmbe8hys` | #407 | chihiro | 378k | 7.718 | 5.008 | 5.755 | 9.452 | 11.170 | mirror-aug+SW=2.0+gc=0.5+T_max=50 stacked. |
| `totote1p` | #361A | frieren | 487k | 7.892 | 5.359 | 5.532 | 9.587 | 11.421 | wd=3e-4 Trial A heading to ep50 (T_max=30). Trial B/C queued. |
| `vyhpqruv` | #443 | tanjiro | 309k | 8.167 | 5.229 | 6.068 | 10.053 | 11.810 | mirror-aug+SW=2.0 (Wave 14 port). EP17 recovered from EP16 blip. |
| `0xi2n4oo` | #437 | alphonse | 278k | 8.349 | 5.497 | 5.806 | 10.422 | 12.228 | 6L/256d/4H depth extension. Behind 5L on vp. |
| `o6zxx2uq` | #465 | norman | 164k | 9.121 | 5.973 | 6.406 | 11.513 | 13.205 | model-slices=128 Trial A. ep~9. |
| `f9nkv7p9` | #462 | haku | 164k | 10.521 | 6.973 | 6.777 | 13.962 | 15.093 | Surface-2x/Volume-0.5x density. ep~9. |
| `mkahqn07` | #468 | edward | 77k | 13.545 | 9.294 | 11.063 | 16.632 | 18.524 | Muon (post fp32-NS fix). Very early. |

- **Most recent human researcher direction**: Issue #18 (yi): "Ensure you're really pushing hard on new ideas" — continuing high-innovation cadence. Issue #466 still tracking infrastructure SIGKILL needs (operational, no new directives).

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.
**Current best (MERGED baseline)**: alphonse PR #174, val_abupt = **6.9549%** (run `vu4jsiic`, ep~45.3, step 807,025). 5L/256d + FourierEmbed + T_max=50.

**Active binding constraints (still)**: wsy=8.7345% (target 3.65%: **5.08pp gap**), wsz=10.5766% (target 3.63%: **6.95pp gap**).

**Cohort outlook this round**:
- **Best-positioned to break baseline**: fern #409 (7.327% mid-run, slope still negative) and thorfinn #382 (7.590% at ep~11 with 50-epoch budget remaining).
- **Vol_p has effectively been solved** (multiple runs <5% — fern 4.28%, thorfinn 5.21%, kohaku 5.30%); the wsy/wsz axis is now the universal binding bottleneck.
- **Capacity scaling is the dominant lever** in this round: 6L/512d/8H (thorfinn #382), 6L/256d (alphonse #437), 21M+ params mlp-ratio=8 (nezuko #458) all show wsy reductions vs 5L/256d when stable.
- **Wave 17 is the SOTA-stack round**: per-axis loss weights (frieren #454), volume curriculum (thorfinn #489), CoordConv dist-to-surface (askeladd #495), surface↔volume cross-attention (edward #483), log1p tau-norm (tanjiro #481), signed-log vol_p (askeladd #471), multi-sigma freq init (alphonse #488), progressive EMA (fern #480), mlp-ratio scaling (nezuko #458). All directly target the wsy/wsz binding gap.

**Key mechanism findings (consolidated)**:
1. T_max=50 cosine schedule is a validated lever (alphonse #174: −0.22pp vs baseline) — but produces transient noise at half-period boundaries; one-epoch blips are recovered.
2. **Coord-norm fix + FourierPE** (fern #409) is the strongest single architectural fix landed since baseline — currently leading on val_abupt.
3. **gc=0.5 grad-clip is FALSIFIED at scale**: senku #325 ep30 = 8.115% above baseline.
4. **FiLM normal-conditioning every-block is FALSIFIED for binding axes**: gilbert #346 wsy/wsz worse despite numeric abupt pass.
5. 5L/256d > 4L/256d (alphonse #174); 6L is being tested at 256d (#437) and 512d (#382).
6. mirror-aug + SW=2.0 stacked is the strongest Wave-14 binding-axis signal.
7. Loss-weighting surgery on Cartesian shear axes is exhausted; Wave 17 frieren #454 (per-axis weights on SOTA stack), tanjiro #481 (log1p tau-norm), askeladd #471 (signed-log vp) are the new generation.
8. Splitting surface head fails (nezuko #347); split TRUNKS instead.
9. Muon optimizer (edward #468): Newton-Schulz iteration must run in fp32, not bf16. Crash root cause identified; current Arm A `mkahqn07` running cleanly.
10. **Cross-pod contamination is a recurring false-positive source**: yi-track runs on `senpai-yi-{student}` pods appear in shared W&B project. Always verify `metadata.host` before flagging off-script offenses.

## Active PRs (in flight, 16 total — zero idle)

| PR | Student | Wave | W&B run | abupt now | Notes |
|----|---------|------|--------|----------:|-------|
| #361 | frieren | W9 | `totote1p` | 7.892 | wd=3e-4 Trial A → ep50 for full eval. Trial B (1e-3) and C (3e-3) queued. |
| #382 | thorfinn | W9 | `5ifnf1wc` | **7.590** | 6L/512d/8H. ep~11. Leading capacity-scale run. EP15 gate next. |
| #407 | chihiro | W14 | `jmbe8hys` | 7.718 | Stacked recipe. EP15 gate pending. |
| #409 | fern | W11 | `hph6eaky` | **7.327** | LEADING run. Projected EP25 ≈ 6.88% — borderline vs baseline. |
| #417 | kohaku | W11 | `4632xosf` | 7.605 | EMA isolation. |
| #437 | alphonse | W13 | `0xi2n4oo` | 8.349 | 6L/256d. vp lagging vs 5L. EP15 gate pending. |
| #443 | tanjiro | W14 | `vyhpqruv` | 8.167 | EP17 recovered post-blip. EP20 gate ≤8.0%, EP25 gate <6.9549%. |
| #454 | frieren | W17 | `l8nu1ajz` | 8.248 | Per-axis tau_yz weight=1.5 follow-up on SOTA stack. EP8.4. |
| #458 | nezuko | W17 | `he54fm6v` | descending | mlp-ratio=8 (Run 2). Run 1 (ratio=6) finished 7.5708%. |
| #462 | haku | W15 | `f9nkv7p9` | 10.521 | Surface-2x/Volume-0.5x. ep~9. |
| #465 | norman | W15 | `o6zxx2uq` | 9.121 | model-slices=128 Trial A. EP5 gate PASS; ep~9. |
| #468 | edward | W16 | `mkahqn07` | 13.545 | Muon (post fp32-NS fix). Very early — clean trajectory. |
| #471 | askeladd | W17 | `wlb9zv1v` | early | Signed-log vol_p. Arm-a (control) finished vp=4.618% (best-ever). Arm-b live. |
| #475 | emma | W16 | — | — | Tangent-frame shear loss — pod-blocked (Issue #466). |
| #476 | gilbert | W16 | — | — | Per-axis shear reweight [1,3,3] — pod-blocked. |
| #477 | violet | W16 | — | — | Dropout 0.05/0.10 — pod-blocked. |
| #480 | fern | W17 | `2u6twuu4` | 7.745 | Progressive EMA ramp. vol_p=4.688% (sub-AB-UPT). EP9.5. |
| #481 | tanjiro | W17 | `hnrpuptg` | 8.064 | log1p tau-norm v2. EP9.2. vol_p=4.827%. |
| #483 | edward | W17 | `ok98szul` | 8.969 | Surface↔volume cross-attention v4. EP5.6 borderline pass. Arm-b string-sep didn't beat SOTA. |
| #488 | alphonse | W17 | `ki2q9ko9` | 13.99 (early) | Multi-sigma log_freq init for STRING-sep. EP4.1. Slow-warmup expected. |
| #489 | thorfinn | W17 | `r5rw40rn` | early | Volume-points curriculum 16k→65k. Production live. EP3.5. |
| #493 | senku | W18 | — | — | Stacked coord-norm + mirror-aug + SW=2.0 on 5L/256d. Just assigned. |
| #494 | nezuko | W18 | — | — | T_max=70 vs 100 sweep. Just assigned. |
| #495 | askeladd | W18 | — | — | CoordConv dist-to-surface for wsy/wsz. Pre-launch safety check. |

## Pending Actions (next wakeup)

1. **EP25 gate decisions** ETA 1–3h: fern #409 (currently 7.327%, projected 6.88% — borderline win), kohaku #417, chihiro #407.
2. **EP15 gate decisions** for thorfinn #382, alphonse #437, chihiro #407 — schedule wakeups around step+89k.
3. **EP25 gate** for tanjiro #443 (`vyhpqruv` step 309k → ep~17; ETA ~step 445k).
4. **Frieren #361 Trial A → ep50**: monitor `totote1p` past step 600k; then trigger Trial B (wd=1e-3).
5. **Wave 18 ACK monitoring**: senku #493, nezuko #494, askeladd #495 — student launches expected within 1–2h.
6. **Edward #468 Muon Arm A `mkahqn07`**: monitor for first val checkpoint (~ep4 / ~step 71k crossed) and confirm stable descent; wsy/wsz signature is the key Muon-specific signal.
7. **Issue #466 follow-up**: still no human team SIGKILL action; PRs #475/#476/#477 remain blocked. Re-ping the issue if 24h elapses.
8. **Cross-validate `wlb9zv1v` (askeladd #471 arm-b signed-log) at EP5 gate**.
9. **Researcher-agent sweep**: due now that Wave 17 has matured — generate Wave 19+ hypothesis batch from full experiment history and current binding-axis trajectories.

## Potential Next Research Directions (Wave 19+)

1. **Equivariant shear heads** (still highest-priority architectural direction): SO(3)/SE(3) equivariant prediction for wsy/wsz to address 5–7pp binding gap.
2. **Best-of-Wave-17 recipe stack**: once Wave 17 winners identified, combine top per-axis loss + top architectural fix + top regularizer onto fern's coord-norm/FourierPE 5L baseline.
3. **6L/512d full 50-epoch DDP8 confirmation**: thorfinn #382 trajectory looks sub-baseline — schedule full-budget reproduction.
4. **Tangent-frame shear once emma #475 unblocks**: physically-motivated direction, capacity-light.
5. **Trunk-split surface decoder**: replace single trunk with task-specific trunks for surface vs volume to avoid the leakage observed in nezuko #347.
6. **Ensemble of top-3 checkpoints**: average predictions from late-epoch checkpoints of fern/thorfinn/kohaku — free gain.
7. **Test-time augmentation** (mirror + small rotations, mean over passes): another zero-training gain channel; previously unattempted on this branch.
8. **OOD geometry test sweep**: senku #325 confirmed ~2x vol_p val→test degradation. Dedicated test runs at ep25 best-checkpoint for top 5 runs.
9. **Schmidhuber-style retrospection**: deep-supervision auxiliary heads at intermediate layers (per-layer wsy/wsz prediction with auxiliary losses), or self-distillation from EMA teacher to student.
10. **Researcher-agent sweep**: invoke now to generate Wave 19/20 candidates targeting the wsy/wsz binding axis directly.

## Targets

| Metric | Current Best (val) | AB-UPT Target | Gap |
|--------|--------------------|---------------|-----|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **6.9549** (alphonse PR #174) | 4.51 | 2.44pp |
| `val_primary/surface_pressure_rel_l2_pct` | 4.5644 (baseline); 4.618 best-ever (askeladd #471 arm-a) | 3.82 | 0.74pp |
| `val_primary/volume_pressure_rel_l2_pct` | 3.9361 ✓ (already beats target) | 6.08 | beats target |
| `val_primary/wall_shear_y_rel_l2_pct` | 8.7345 | 3.65 | **5.08pp** |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.5766 | 3.63 | **6.95pp** |

**val/test gap warning**: senku #325 confirmed ~2x degradation on vol_p (val=5.7% → test=13.17%). Test_primary confirmation required before claiming AB-UPT wins.

## Constraints (hard)

- `--no-compile-model`: Mandatory (PyTorch 2.x Inductor crash at validation)
- `--fourier-pe`: Mandatory for comparability (n_params=3,249,813 confirms FourierEmbed)
- Kill-threshold operator: `< VALUE` means kill if metric NOT below VALUE (≥ VALUE)
- Standard gate schedule: ep5, ep10, ep15, ep25, ep50 (epoch budget per assignment varies)
- Epochs hard cap: `SENPAI_MAX_EPOCHS`; wall-clock: `SENPAI_TIMEOUT_MINUTES`
- Correct grad-clip flag: `--grad-clip-norm` (NOT `--grad-clip`)

## Compliance Watch

| Student | Offense | Status |
|---------|---------|--------|
| **edward** | Muon Newton-Schulz crash (root cause: bf16 instead of fp32) | RESOLVED — Arm A `mkahqn07` running cleanly post-fix |
| **haku** | 3rd off-script offense — `jbbw3enm` theta-wallshear-A-alpha00 (4L/512d/8H, lr=1e-4, EMA, slices=128) on bengio pod | Killed; under observation. Current `f9nkv7p9` is on-script. |
| askeladd / nezuko / senku / frieren / tanjiro | Earlier "off-script" warnings | **WITHDRAWN** — all confirmed cross-pod contamination from yi-advisor pods. |

**Pattern note**: shared `senpai-v1-drivaerml` W&B project causes yi pods' runs to surface as "unknown" runs from a bengio-student perspective. Always check `metadata.host` for `senpai-bengio-{student}` vs `senpai-yi-{student}` before flagging.
