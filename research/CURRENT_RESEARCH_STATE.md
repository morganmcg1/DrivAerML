# SENPAI Research State

- **2026-05-03 09:30Z** — Mid-flight gate sweep complete. Three advisor comments posted (thorfinn #382, askeladd #495, frieren #361) reflecting ep6/7/14 progress checks. **Critical infra issue**: 6 student pods (haku, gilbert, senku, violet, chihiro, fern) stuck on stale train.py for 19–50 hours; status update posted to issue #466 escalating to human team for manual SIGKILL or `kubectl rollout restart`. **24 GPU-slots idle on infrastructure block**.
  - **edward #468** EP5 PASS: abupt=8.402% @ ep5 (run `6b08y222`). EP10 watch active (kill if ≥8.5% — TIGHT margin, only 0.098pp headroom). Slope healthy: ep1=10.00 → ep5=8.40 (-1.60pp over 4 epochs).
  - **askeladd #495** EP6: abupt=9.603% (run `ky4rf6g5`). **Slope decelerating sharply** — ep5→ep6 only -0.061pp drop. Projecting ep10 ≈ 9.36% (clears EP10 gate by 1.1pp but won't beat baseline). Decision rule posted: kill at ep10 if abupt > 9.0%.
  - **frieren #361** EP7: abupt=9.745% (run `e1kxrd6b`). One stochastic ep6 bounce (+0.136pp), recovered ep7 (-0.565pp). On-track for EP10 gate (<10.5%).
  - **thorfinn #382** EP14: abupt=7.508% (run `5ifnf1wc`). **Stalled ep13→ep14 (+0.003pp regress)** at 7.5055→7.5084. May be hitting plateau before cosine LR drop activates. Continue running; watch ep15 closely.
  - **emma #502** running: run `ww9cxr3h`, step 7,494 (~ep0.42). Compliant (FourierPE + no-EMA + cosine T_max=50 + deep-supervision aux). EP1 val pending.
  - **kohaku #508** ep2: abupt=16.39%, descending normally (5L/256d baseline reproduce for SWA experiment).
  - **norman #509** ep2: abupt=16.79%, descending normally (5L/256d + OHEM ratio=0.5 with 5-epoch warmup).
  - **alphonse #513** running: run group `bengio-wave20`, prefix `alphonse/6l-512d-tmax70`, W&B ID still TBD (explicit ID not yet confirmed in PR comments).
- **Most recent human researcher direction**: Issue #466 tracking zombie pod SIGKILL, stark pod provisioning, and cross-track W&B tagging. No new directives since Issue #18 (yi): "Ensure you're really pushing hard on new ideas".

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.
**Current best (MERGED baseline)**: alphonse PR #174, val_abupt = **6.9549%** (run `vu4jsiic`, ep~45.3, step 807,025). 5L/256d + FourierEmbed + T_max=50. n_params=3,992,313.

**Binding unsolved constraints**: wsy=8.7345% (target 3.65%: **5.08pp gap**), wsz=10.5766% (target 3.63%: **6.95pp gap**).

**Strongest active candidate to beat baseline**: thorfinn #382 (`5ifnf1wc`, 6L/512d/8H + T_max=50) at **7.508% ep14** — needs ~0.55pp more drop in remaining 36 epochs of cosine schedule. Recent ep13→ep14 stall is the key risk indicator.

## Active PRs — 17 In-Flight

| PR | Student | Wave | W&B run | Last known abupt | Notes |
|----|---------|------|---------|----------------:|-------|
| #382 | thorfinn | W9 | `5ifnf1wc` | **7.508% ep14** | 6L/512d/8H. EP15 PASS (ep12=7.5445). ep13→ep14 stalled (+0.003pp). Slope flattening, watching for ep15. |
| #361 | frieren | W9 | `e1kxrd6b` | **9.745% ep7** | wd=1e-3 sweep Trial B. ep6 bounce (+0.136), ep7 recovered. EP10 watch (<10.5%). |
| #495 | askeladd | W18 | `ky4rf6g5` | **9.603% ep6** | CoordConv dist-to-surface. **Slope decelerating** (ep5→ep6 only -0.061pp). EP10 watch — kill at ep10 if >9.0%. |
| #468 | edward | W16 | `6b08y222` | **8.402% ep5** | Muon LR=5e-4. EP5 PASS. EP10 gate at 8.5% is **TIGHT (0.098pp headroom)**. |
| #508 | kohaku | W12 | `5v1mjka1` | 16.39% ep2 | SWA over cosine tail ep46-50 on 5L/256d baseline. Healthy descent. |
| #509 | norman | W12 | `wb2ww9a2` | 16.79% ep2 | OHEM top-50% wall-shear hard-mining + 5-epoch warmup. Healthy descent. |
| #513 | alphonse | W20 | TBD | — ~ep0.22 | 6L/512d/8H + T_max=70. Run ID still not posted explicitly. |
| #502 | emma | W19 | `ww9cxr3h` | — ~ep0.42 | Deep supervision aux wsy/wsz losses at layers 2,3,4 (AuxShearHead). Compliant relaunch. EP1 pending. |
| #504 | violet | W19 | — | — | **POD STUCK** since 2026-05-02 14:04Z (19h). Trunk-split decoder. |
| #505 | chihiro | W19 | — | — | **POD STUCK** since 2026-05-02 14:03Z (19h). SO(3)-equivariant shear head. |
| #507 | haku | W19 | — (zvros0ej assigned, never advanced) | — | **POD STUCK** since 2026-05-01 07:42Z (49.7h). Per-axis shear reweight. |
| #512 | gilbert | W20 | — | — | **POD STUCK** since 2026-05-01 07:43Z (49.7h). Tangent-frame wall-shear. |
| #514 | fern | W20 | — | — | **POD STUCK** since 2026-05-02 14:04Z (19h). TTA y-mirror averaging. |
| #515 | senku | W20 | — | — | **POD STUCK** since 2026-05-01 17:29Z (39.9h). Coord-norm + TTA stacked. |
| #524 | tanjiro | W21 | — | — | Laplacian curvature (κ_H, κ_G) as surface input features. Sign-off given 09:16Z. |
| #525 | nezuko | W21 | — | — | Yaw-rotation TTA ±2° eval-only on vu4jsiic checkpoint. DRAFT — awaiting launch. |
| #526 | stark | W21 | — | — | Fourier PE frequency sweep: n=6 vs n=16 (baseline n=8). DRAFT — **blocked on missing pod**. |

**Infrastructure block**: 6 pods (haku, gilbert, senku, violet, chihiro, fern) stuck on stale train.py — `kubectl exec` is forbidden for orchestrator service account; **HUMAN TEAM ESCALATION REQUIRED**. Issue #466 has full context. Effective lost capacity = 24 GPU-slots × 19–50 hours.

**CLOSED in recent session**:
- #443 (tanjiro W14 mirror+SW=2.0): Plateaued ~7.8-8.0% at ep22.7 (run `vyhpqruv`). EP25 gate failed.
- #498 (nezuko T_max sweep): Killed at EP20 gate — T_max=70/100 does NOT improve over T_max=50.

## Key Insights (consolidated)

1. **Coord-norm fix** (fern #409, merged) is the strongest single architectural lever — drove fern to 7.16% ep24 before merging.
2. **6L/512d/8H capacity** (thorfinn #382) is the strongest active architecture at 7.508% ep14 (still descending slowly — hold to ep50). Stall at ep13→ep14 is the key risk indicator.
3. **T_max=50 is the validated optimum** — T_max=70/100 does NOT improve over T_max=50 (nezuko #498 closed at EP20). Alphonse #513 still testing T_max=70 with 6L/512d/8H separately.
4. **Mirror-aug + SW=2.0** (tanjiro #443): Did not beat baseline. wsy-targeting via data augmentation alone insufficient.
5. **EMA is inferior to no-EMA** on this architecture (kohaku #417 confirmed EMA hurts vol_p by 1.0pp).
6. **vol_p has been solved** (well below AB-UPT target 6.08%). wsy/wsz is the universal binding bottleneck.
7. **Muon LR=1e-3** too aggressive for 5L/256d. Retry at LR=5e-4 (edward #468) showing clean monotonic descent ep1→ep5: 10.00 → 8.40 (-1.60pp).
8. **Cross-attention bridge** (edward #483) closed without beating baseline.
9. **6L/256d** (alphonse #437) slower than 5L to converge and did not beat baseline.
10. **Weight decay** wd=3e-4 Trial A confirmed worse than baseline (7.833% ep30). Trial B wd=1e-3 (frieren #361) tracking similar to non-WD baseline at ep7 — suggesting WD provides limited net benefit.
11. **Cross-pod yi contamination** is a recurring false-positive: always verify `metadata.host` before flagging.
12. **Deep supervision** for wsy/wsz (emma #502): AuxShearHead at transformer layers 2,3,4. Compliant run `ww9cxr3h` — EP1 pending.
13. **CoordConv distance-to-surface** (askeladd #495): showing weak signal (decelerating after ep5). Provides uniform improvement, not specialization on wall-shear.
14. **Critical infra failure mode**: senpai watchdog cannot SIGKILL stale train.py if the previous PR was closed without proper run termination. 6 pods affected; orchestrator service account lacks `pods/exec` permission to manually kill from advisor side.

## Upcoming EP Gates

| PR | Student | Gate | Status / Action |
|----|---------|------|------|
| #382 | thorfinn | Continue to ep50 | EP14=7.508% (stalled ep13→14). Watch ep15 for plateau confirmation. |
| #361 | frieren | EP10 (<10.5%) | EP7=9.745%. On track. |
| #495 | askeladd | EP10 (<10.5%) | EP6=9.603%. Decelerating. **Kill at ep10 if >9.0%.** |
| #468 | edward | EP10 (<8.5%) | EP5=8.402%. **TIGHT — 0.098pp headroom**. |
| #508 | kohaku | EP5 (<13%) | EP2=16.39%. Healthy descent. |
| #509 | norman | EP5 (<13%) | EP2=16.79%. Healthy descent. |
| #513 | alphonse | EP5 (<13%) | ~ep0.22. Run ID still TBD. |
| #502 | emma | EP5 (<13%) | ~ep0.42. EP1 pending. |
| #504/505/507/512/514/515 | 6 students | — | **POD STUCK — escalated to human team via issue #466.** |
| #524/525 | tanjiro/nezuko | — | Awaiting launch. |
| #526 | stark | — | Awaiting pod provisioning. |

## Potential Next Research Directions (Wave 22+)

1. **Compound best-of-wave**: coord-norm + 6L/512d/8H — if thorfinn beats baseline, combine with other winning techniques.
2. **Equivariant geometry heads**: SE(3)/SO(3) equivariant outputs for wsy/wsz — physics-motivated path for 5-7pp gap (chihiro #505 testing once unblocked).
3. **OOD geometry test sweep**: confirm val→test gap on all top-5 val runs before claiming AB-UPT wins (~2x degradation on vol_p confirmed).
4. **Best-checkpoint ensemble**: average late-epoch checkpoints from thorfinn/alphonse/other top runs.
5. **Multi-resolution point sampling**: coarse+fine hierarchical sampling targeting high-gradient boundary regions.
6. **Fourier feature ablation results**: stark #526 sweep (n=6 vs n=16 vs n=8 baseline) will inform PE capacity recommendations once unblocked.
7. **Laplacian curvature features** (tanjiro #524): If κ_H/κ_G improve wsy/wsz, generalize to full curvature tensor family.
8. **Yaw-rotation TTA results** (nezuko #525): Free ensemble signal; if effective, combine with other TTA strategies.
9. **If thorfinn plateaus at ~7.5%**: pivot to architectural changes — cross-attention surface↔volume coupling at higher capacity, or mesh-aware geometric kernels.
10. **Researcher-agent sweep**: Generate Wave 22+ candidates targeting wsy/wsz 5-7pp gap via loss + data + architecture angles not yet explored.

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
| haku #507 | **POD STUCK** since 2026-05-01. Cannot ACK. Escalated to human team. |
| edward #468 | Muon LR=1e-3 self-killed and retried at LR=5e-4 per fallback plan. Compliant. |
| frieren #361 | Trial A ep30 self-stopped at CosineAnnealingLR boundary — correct call. |
| emma #502 | **CORRECTED**: Initial run `zg3ukcex` had wrong flags. Relaunched as `ww9cxr3h` with canonical flags + deep supervision. Now compliant. |
| 6 stuck-pod students | violet, chihiro, gilbert, fern, senku — non-responsive due to infra block, not non-compliance. |

## Infrastructure Status

- **Stuck pods**: 6 (haku 49.7h, gilbert 49.7h, senku 39.9h, violet 19.4h, chihiro 19.4h, fern 19.4h)
- **stark pod**: Not provisioned (PR #526 blocked).
- **`kubectl exec` from orchestrator**: Forbidden by RBAC — advisor cannot kill stale processes.
- **`senpai_track` W&B tag enforcement**: Open ask for cross-track filtering.
- **Closed-PR auto-kill**: Open ask for deployment-level SIGKILL when PR transitions to closed.
- **Combined idle GPU-time**: ≥150 GPU-hours lost on stuck pods alone (24 slots × ~6h average since the human team last checked).
