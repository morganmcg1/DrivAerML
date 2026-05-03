# SENPAI Research State

- **2026-05-03 10:00Z** — Wave 21 assigned (PRs #524–526: tanjiro Laplacian curvature, nezuko yaw-TTA, stark Fourier-freq sweep). PR #443 (tanjiro W14 mirror+SW) and PR #498 (nezuko T_max sweep) CLOSED. Active WIP count: **17 PRs** across Waves 9–21. Stark pod still absent (infra blocker — stark PR #526 DRAFT pending pod). Emma compliance corrected: run `ww9cxr3h` confirmed with proper flags + deep supervision.
  - **askeladd #495** EP5 PASS: abupt=9.6641% @ ep5.06 (run `ky4rf6g5`). EP10 watch active (<10.5%).
  - **edward #468** pre-EP5: abupt=8.7703% @ ep3.87 (run `6b08y222`), projecting ep5≈8.30%. On track.
  - **frieren #361** EP5 PASS: abupt=10.1743% @ ep5.40 (run `e1kxrd6b`). EP10 watch active (<10.5%).
  - **thorfinn #382** EP15 PASS: abupt≈7.5445% @ ep~12.8 (run `5ifnf1wc`). Slow convergence. Continue to ep50.
  - **alphonse #513** running: run group `bengio-wave20`, prefix `alphonse/6l-512d-tmax70`, W&B ID TBD (explicit ID not yet posted). Step ~3,966 (~ep0.22) as of 09:00Z.
  - **emma #502** running: run `ww9cxr3h` (compliant relaunch with `--fourier-pe --no-use-ema --lr 3e-4 --lr-cosine-t-max 50`; deep supervision AuxShearHead at layers 2,3,4). EP gate watch pending ep1.
  - **haku #507** running: run `zvros0ej` (~ep0.1 at 08:30Z). EP gate watch pending ep1.
  - **kohaku #508** running: run `5v1mjka1` (~ep0). SWA tail ep46-50.
  - **norman #509** running: run `wb2ww9a2` (~ep0). OHEM top-50%.
- **2026-05-03 09:00Z** — Previous gate update (see above for updates since then).
- **Most recent human researcher direction**: Issue #466 tracking zombie pod SIGKILL, stark pod provisioning, and cross-track W&B tagging — no new directives since Issue #18 (yi): "Ensure you're really pushing hard on new ideas".

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.
**Current best (MERGED baseline)**: alphonse PR #174, val_abupt = **6.9549%** (run `vu4jsiic`, ep~45.3, step 807,025). 5L/256d + FourierEmbed + T_max=50. n_params=3,992,313.

**Binding unsolved constraints**: wsy=8.7345% (target 3.65%: **5.08pp gap**), wsz=10.5766% (target 3.63%: **6.95pp gap**).

## Active PRs — 17 In-Flight

| PR | Student | Wave | W&B run | Last known abupt | Notes |
|----|---------|------|---------|----------------:|-------|
| #382 | thorfinn | W9 | `5ifnf1wc` | **7.544% ep~12.8** | 6L/512d/8H. EP15 PASS. Slow convergence. Continue to ep50. |
| #361 | frieren | W9 | `e1kxrd6b` | **10.174% ep5.40** | wd=1e-3 sweep Trial B. EP5 PASS. Slope -0.0190%/k. EP10 watch (<10.5%). |
| #495 | askeladd | W18 | `ky4rf6g5` | **9.664% ep5.06** | CoordConv dist-to-surface. EP5 PASS (gate <13%). Slope -0.0239%/k. EP10 watch (<10.5%). |
| #468 | edward | W16 | `6b08y222` | **8.770% ep3.87** | Muon LR=5e-4 retry. Clean monotonic descent. Slope -0.0235%/k. Approaching EP5 gate (<9.5%). |
| #508 | kohaku | W12 | `5v1mjka1` | — ep0 | SWA over cosine tail ep46-50 on 5L/256d baseline. Just launched. |
| #509 | norman | W12 | `wb2ww9a2` | — ep0 | OHEM top-50% hard-mining for wall-shear. Just launched. |
| #513 | alphonse | W20 | TBD (group: `bengio-wave20`, prefix `alphonse/6l-512d-tmax70`) | — ep0.22 | 6L/512d/8H + T_max=70. Launched 08:10Z. Step ~3,966. Run ID not yet explicitly confirmed in PR. |
| #502 | emma | W19 | `ww9cxr3h` | — ep0 | Deep supervision aux wsy/wsz losses at layers 2,3,4 (AuxShearHead). Compliant relaunch confirmed. |
| #504 | violet | W19 | — | — | Trunk-split decoder: specialized surface/volume transformer. Pinged 08:30Z — awaiting ACK. |
| #505 | chihiro | W19 | — | — | SO(3)-equivariant shear head tangent-frame prediction. Pinged 08:30Z — awaiting ACK. |
| #507 | haku | W19 | `zvros0ej` | — ep0.1 | Per-axis shear loss reweight [wsy=3, wsz=3]. Started 07:52Z. |
| #512 | gilbert | W20 | — | — | Tangent-frame wall-shear loss. Pinged 08:30Z — awaiting ACK. |
| #514 | fern | W20 | — | — | TTA y-mirror averaging for wsy symmetry. Pinged 08:30Z — awaiting ACK. |
| #515 | senku | W20 | — | — | Coord-norm + TTA y-mirror stacked. Pinged 08:30Z — awaiting ACK. |
| #524 | tanjiro | W21 | — | — | Laplacian curvature (κ_H, κ_G) as surface input features. DRAFT — awaiting launch. |
| #525 | nezuko | W21 | — | — | Yaw-rotation TTA ±2° eval-only on vu4jsiic checkpoint. DRAFT — awaiting launch. |
| #526 | stark | W21 | — | — | Fourier PE frequency sweep: n=6 vs n=16 (baseline n=8). DRAFT — blocked on pod provisioning. |

**Non-responsive (unconfirmed run IDs)**: violet #504, chihiro #505, gilbert #512, fern #514, senku #515 — all Wave 19/20, pinged 08:30Z. Second ping may be needed.

**CLOSED this session**:
- #443 (tanjiro W14 mirror+SW=2.0): Plateaued ~7.8-8.0% at ep22.7 (run `vyhpqruv`). EP25 gate failed. CLOSED.
- #498 (nezuko T_max sweep): Killed at EP20 gate — run `tbm0bua1` ep18=8.157%, projected ep20~8.07-8.10%, gate requires <8.0%. T_max=70/100 does NOT provide sufficient headroom over T_max=50. CLOSED.

## Key Insights (consolidated)

1. **Coord-norm fix** (fern #409, merged) is the strongest single architectural lever — drove fern to 7.16% ep24 before merging. All coord-norm family runs now benefit.
2. **6L/512d/8H capacity** (thorfinn #382) is the strongest architecture at 7.544% ep~12.8 (still descending slowly — hold to ep50).
3. **T_max=50 is the validated optimum** — T_max=70/100 does NOT improve over T_max=50 (nezuko #498 closed at EP20 with T_max sweep failing gate). Alphonse #513 still testing T_max=70 with 6L/512d/8H.
4. **Mirror-aug + SW=2.0** (tanjiro #443): CLOSED at EP25. Did not beat baseline. wsy-targeting via data augmentation alone insufficient.
5. **EMA is inferior to no-EMA** on this architecture — kohaku #417 confirmed EMA hurts vol_p by 1.0pp.
6. **vol_p has been solved** (well below AB-UPT target 6.08%). wsy/wsz is the universal binding bottleneck.
7. **Muon LR=1e-3** too aggressive for 5L/256d (oscillating, failed EP5). Retry at LR=5e-4 (edward #468) showing clean descent.
8. **Cross-attention bridge** (edward #483) closed without beating baseline.
9. **6L/256d** (alphonse #437) slower than 5L to converge and did not beat baseline. 6L+512d+8H (thorfinn) is the better architecture upgrade.
10. **Weight decay** wd=3e-4 Trial A confirmed worse than baseline (7.833% ep30). Trial B wd=1e-3 running (frieren #361).
11. **Cross-pod yi contamination** is a recurring false-positive: always verify `metadata.host` before flagging.
12. **Deep supervision** for wsy/wsz (emma #502): AuxShearHead at transformer layers 2,3,4. Compliant run `ww9cxr3h` just launched — watch for EP5 gate.

## Upcoming EP Gates

| PR | Student | Gate | Action needed |
|----|---------|------|---------------|
| #382 | thorfinn | Continue to ep50 | EP15=7.544%. Slow convergence. Continue. |
| #361 | frieren | EP10 (<10.5%) | EP5.40=10.174% PASS. Slope projects ep10≈8.6%. Monitor. |
| #495 | askeladd | EP10 (<10.5%) | EP5.06=9.664% PASS. Strong neg slope. Monitor. |
| #468 | edward | EP5 (<9.5%) | EP3.87=8.770%, projects ep5≈8.30%. On track. |
| #508 | kohaku | EP5 (<13%) | SWA tail run just launched. |
| #509 | norman | EP5 (<13%) | OHEM run just launched. |
| #513 | alphonse | EP5 (<13%) | 6L/512d/8H T_max=70. ~ep0.22. |
| #502 | emma | EP5 (<13%) | Deep supervision compliant relaunch. ~ep0. |
| #507 | haku | EP5 (<13%) | Per-axis shear reweight. ~ep0.1. |
| W19/W20 | 5 students | EP5 (<13%) | gilbert/violet/chihiro/fern/senku: awaiting run confirms. |
| W21 | tanjiro/nezuko/stark | — | DRAFT — awaiting pod/launch. |

## Potential Next Research Directions (Wave 22+)

1. **Compound best-of-wave**: coord-norm + 6L/512d/8H — if thorfinn beats baseline, combine with other winning techniques.
2. **Equivariant geometry heads**: SE(3)/SO(3) equivariant outputs for wsy/wsz — physics-motivated path for 5-7pp gap (chihiro #505 testing).
3. **OOD geometry test sweep**: confirm val→test gap on all top-5 val runs before claiming AB-UPT wins (~2x degradation on vol_p confirmed).
4. **Best-checkpoint ensemble**: average late-epoch checkpoints from thorfinn/alphonse/other top runs.
5. **Multi-resolution point sampling**: coarse+fine hierarchical sampling targeting high-gradient boundary regions.
6. **Fourier feature ablation results**: stark #526 sweep (n=6 vs n=16 vs n=8 baseline) will inform PE capacity recommendations.
7. **Laplacian curvature features** (tanjiro #524): If κ_H/κ_G improve wsy/wsz, generalize to full curvature tensor family.
8. **Yaw-rotation TTA results** (nezuko #525): Free ensemble signal; if effective, combine with other TTA strategies.
9. **Researcher-agent sweep**: Generate Wave 22+ candidates targeting wsy/wsz 5-7pp gap via loss + data + architecture angles not yet explored.

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
- `--no-use-ema`: Mandatory (EMA confirmed inferior on this architecture)
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
| emma #502 | **CORRECTED**: Initial run `zg3ukcex` had wrong flags (ema=True, no fourier_pe, lr=1e-4, no cosine). Advisor flagged; emma relaunched as `ww9cxr3h` with canonical flags + deep supervision. Now compliant. |
