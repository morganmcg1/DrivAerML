# SENPAI Research State

- **2026-05-03 09:00Z** — Gate update. 16 active WIP PRs across Waves 9–20. 0 review-ready. 0 idle students. All 16 pods 1/1 Ready. Stark pod still absent. Wave 19/20 watchdog blocked on prior train.py — students will resume after prior runs finish.
  - **askeladd #495** EP5 PASS: abupt=9.6641% @ ep5.06 (run `ky4rf6g5`).
  - **edward #468** pre-EP5: abupt=8.7703% @ ep3.87 (run `6b08y222`), clean monotonic descent.
  - **frieren #361** EP5 PASS: abupt=10.1743% @ ep5.40 (run `e1kxrd6b`), EP10 watch on track.
  - **alphonse #513** running: run `hdzyr4fl` (rank0), step 3,966 (~ep0.22), 6L/512d/8H + T_max=70.
  - **emma #502** running: run `3y93q5h7` just started (step 10, 08:27Z) — IGNORE this run id, it's actually a yi-pod contamination (host=`senpai-yi-emma-...`). Real bengio run still pending.
  - **haku #507** running: run `zvros0ej` (1708 steps, ~ep0.1).
- **2026-05-03 08:30Z** — Full session update. 16 active WIP PRs across Waves 9–20. 0 review-ready. 0 idle students. All 16 pods 1/1 Ready. Stark pod still absent (infra blocker). Wave 19 (5 PRs) and Wave 20 (3 PRs) students pinged to confirm run starts and post W&B IDs.
- **Most recent human researcher direction**: Issue #466 tracking zombie pod SIGKILL, stark pod provisioning, and cross-track W&B tagging — no new directives since Issue #18 (yi): "Ensure you're really pushing hard on new ideas".

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.
**Current best (MERGED baseline)**: alphonse PR #174, val_abupt = **6.9549%** (run `vu4jsiic`, ep~45.3, step 807,025). 5L/256d + FourierEmbed + T_max=50. n_params=3,992,313.

**Binding unsolved constraints**: wsy=8.7345% (target 3.65%: **5.08pp gap**), wsz=10.5766% (target 3.63%: **6.95pp gap**).

## Active PRs — All 16 In-Flight

| PR | Student | Wave | W&B run | Last known abupt | Notes |
|----|---------|------|---------|----------------:|-------|
| #443 | tanjiro | W14 | `vyhpqruv` | **7.843% ep22.7** | Mirror-aug+SW=2.0. Slope −0.01030%/k. Continue to ep50. |
| #382 | thorfinn | W9 | `5ifnf1wc` | **7.506% ep13.3** | 6L/512d/8H. Slope −0.002189%/k. Continue to ep50. |
| #361 | frieren | W9 | `e1kxrd6b` | **10.174% ep5.40** | wd=1e-3 sweep Trial B. EP5 PASS. Slope -0.0190%/k. EP10 watch (<10.5%, ETA step ~178k). |
| #495 | askeladd | W18 | `ky4rf6g5` | **9.664% ep5.06** | CoordConv dist-to-surface. EP5 PASS (gate <13%). Slope -0.0239%/k. EP10 watch (<10.5%). |
| #468 | edward | W16 | `6b08y222` | **8.770% ep3.87** | Muon LR=5e-4 retry. Clean monotonic descent. Slope -0.0235%/k. Approaching EP5 gate (<9.5%). |
| #508 | kohaku | W12 | `5v1mjka1` | — ep0 | SWA over cosine tail ep46-50 on 5L/256d baseline. Just launched. |
| #509 | norman | W12 | `wb2ww9a2` | — ep0 | OHEM top-50% hard-mining for wall-shear. Just launched. |
| #513 | alphonse | W20 | `hdzyr4fl` | — ep0.22 | 6L/512d/8H + T_max=70. Launched 08:10Z. Step 3,966 / 17,816 (ep1). |
| #498 | nezuko | W19 | — | — | T_max sweep 70 vs 100. Pinged 08:30Z — awaiting run ID. |
| #502 | emma | W19 | — | — | Deep supervision aux wsy/wsz losses at layers 2,3,4. Pinged 08:30Z. |
| #504 | violet | W19 | — | — | Trunk-split decoder: specialized surface/volume transformer. Pinged 08:30Z. |
| #505 | chihiro | W19 | — | — | SO(3)-equivariant shear head tangent-frame prediction. Pinged 08:30Z. |
| #507 | haku | W19 | `zvros0ej` | — ep0.1 | Per-axis shear loss reweight [wsy=3, wsz=3]. Started 07:52Z. |
| #512 | gilbert | W20 | — | — | Tangent-frame wall-shear loss. Pinged 08:30Z — awaiting ACK. |
| #514 | fern | W20 | — | — | TTA y-mirror averaging for wsy symmetry. Pinged 08:30Z — awaiting ACK. |
| #515 | senku | W20 | — | — | Coord-norm + TTA y-mirror stacked. Pinged 08:30Z — awaiting ACK. |

## Key Insights (consolidated)

1. **Coord-norm fix** (fern #409, merged) is the strongest single architectural lever — drove fern to 7.16% ep24 before merging. All coord-norm family runs now benefit.
2. **6L/512d/8H capacity** (thorfinn #382) is the strongest architecture at 7.506% ep13.3 (descending).
3. **T_max=50** validated as +0.25pp vs T_max=30. T_max=70/100 sweep (alphonse #513, nezuko #498) now testing.
4. **Mirror-aug + SW=2.0** directly moves wsy — best targeted binding-axis lever (tanjiro #443 at 7.843% ep22).
5. **EMA is inferior to no-EMA** on this architecture — kohaku #417 confirmed EMA hurts vol_p by 1.0pp.
6. **vol_p has been solved** (well below AB-UPT target 6.08%). wsy/wsz is the universal binding bottleneck.
7. **Muon LR=1e-3** too aggressive for 5L/256d (oscillating, failed EP5). Retry at LR=5e-4 (edward #468) showing clean descent.
8. **Cross-attention bridge** (edward #483) closed without beating baseline, but showed promising early acceleration pattern.
9. **6L/256d** (alphonse #437) slower than 5L to converge and did not beat baseline. 6L+512d+8H (thorfinn) is the better architecture upgrade.
10. **Weight decay** wd=3e-4 Trial A confirmed worse than baseline (7.833% ep30). Trial B wd=1e-3 running.
11. **Cross-pod yi contamination** is a recurring false-positive: always verify `metadata.host` before flagging.

## Upcoming EP Gates

| PR | Student | Gate | Action needed |
|----|---------|------|---------------|
| #443 | tanjiro | Continue to ep50 | EP22.7=7.843%. On trajectory, watch for convergence near baseline. |
| #382 | thorfinn | Continue to ep50 | EP13.3=7.506%. On track but slow convergence. |
| #361 | frieren | EP10 (<10.5%) | EP5.40=10.174% PASS. Slope projects ep10 ≈ 8.6%. Monitor. |
| #495 | askeladd | EP10 (<10.5%) | EP5.06=9.664% PASS. Strong neg slope. Monitor. |
| #468 | edward | EP5 (<9.5%) | EP3.87=8.770%, projects ep5 ≈ 8.30%. On track. |
| #508 | kohaku | EP5 (<13%) | Just launched SWA tail run. |
| #509 | norman | EP5 (<13%) | Just launched OHEM run. |
| #513 | alphonse | EP5 (<13%) | Just launched 6L/512d/8H T_max=70. |
| W19/W20 | 8 students | EP5 (<13%) | Awaiting run starts and ep1 metrics. |

## Potential Next Research Directions (Wave 21+)

1. **Compound best-of-wave**: coord-norm + mirror-aug + SW=2.0 + 6L/512d/8H — if thorfinn and tanjiro both beat baseline, combine them.
2. **Test-time augmentation on best checkpoint**: mirror averaging at inference, free ensemble signal for wsy (fern #514 testing).
3. **Equivariant geometry heads**: SE(3)/SO(3) equivariant outputs for wsy/wsz — physics-motivated path for 5-7pp gap (chihiro #505 testing tangent-frame).
4. **OOD geometry test sweep**: confirm val→test gap on all top-5 val runs before claiming AB-UPT wins.
5. **Best-checkpoint ensemble**: average late-epoch checkpoints from thorfinn/tanjiro/alphonse.
6. **Researcher-agent sweep**: generate Wave 21+ candidates targeting wsy/wsz 5-7pp gap via loss + data + architecture.
7. **Fourier feature ablation**: systematically test Fourier embedding frequencies on wall-shear axes.
8. **Multi-resolution point sampling**: coarse+fine hierarchical sampling targeting high-gradient boundary regions.

## Targets

| Metric | Current Best (val) | AB-UPT Target | Gap |
|--------|--------------------|---------------|-----|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **6.9549** (alphonse PR #174) | 4.51 | 2.44pp |
| `val_primary/surface_pressure_rel_l2_pct` | 4.5644 | 3.82 | 0.74pp |
| `val_primary/volume_pressure_rel_l2_pct` | 3.9361 ✓ | 6.08 | beats target |
| `val_primary/wall_shear_y_rel_l2_pct` | 8.7345 | 3.65 | **5.08pp** |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.5766 | 3.63 | **6.95pp** |

**val/test gap warning**: ~2x degradation on vol_p confirmed on test set. Test confirmation required before claiming AB-UPT wins.

## Constraints (hard)

- `--no-compile-model`: Mandatory (PyTorch 2.x Inductor crash at validation)
- `--fourier-pe`: Mandatory for comparability (n_params=3,992,313 for 5L/256d)
- `--coord-norm`: Required for all coord-norm experiments
- Kill-threshold operator: `< VALUE` means kill if metric NOT below VALUE (≥ VALUE)
- Standard gate schedule: ep5, ep10, ep15, ep20/25, ep50 (varies by assignment)
- Correct grad-clip flag: `--grad-clip-norm` (NOT `--grad-clip`)

## Compliance Watch

| Student | Status |
|---------|--------|
| All prior "off-script" flags | **WITHDRAWN** — confirmed cross-pod yi contamination via metadata.host checks. |
| haku #507 | Per-axis reweight [wsy=3, wsz=3] — on script, clean assignment. Monitor. |
| edward #468 | Muon LR=1e-3 self-killed and retried at LR=5e-4 per fallback plan. Compliant. |
| frieren #361 | Trial A ep30 self-stopped at CosineAnnealingLR boundary — correct call. |
