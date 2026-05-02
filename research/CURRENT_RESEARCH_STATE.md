# SENPAI Research State

- **2026-05-02 19:21Z** — Wave 15 launched; 3 new assignments posted (haku #455 SW=2.0 isolation, violet #456 warmup=3ep, gilbert #457 dropout=0.1). Wave 14 non-compliance: haku #445, violet #446, gilbert #447 all closed for missed ACK deadlines (haku/violet 2nd consecutive miss, gilbert 2nd consecutive miss). stark #448 ACK deadline 19:29Z — monitoring.
- **Current best**: alphonse PR #174, val_abupt=**6.9549%**, run `vu4jsiic`, ep~45.3, step 807,025 (5L/256d/4H + FourierPE + T_max=50, no-EMA).
- **Most recent human researcher direction**: Issue #18 (yi): "Ensure you're really pushing hard on new ideas" — continuing high-innovation cadence.

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.  
**Current best**: alphonse PR #174, val_abupt=**6.9549%** (merged, run `vu4jsiic`, ep~45.3). 2.44pp gap to AB-UPT target remains.

**Active binding constraints**: wsy=8.7345% (target 3.65%: **5.08pp gap**), wsz=10.5766% (target 3.63%: **6.95pp gap**).

**Key mechanism findings (consolidated)**:
1. T_max=50 cosine schedule is a validated real lever (alphonse #174: −0.22pp vs baseline)
2. gc=0.5 grad-clip is a validated real lever (senku #325: ep20=8.504%, strong descent toward 7.x%)
3. 5L/256d architecture > 4L/256d: alphonse #174 6.987% at ep42 vs 7.2091% baseline 4L/256d
4. mirror-aug + SW=2.0 stacked is the strongest binding-axis signal (tanjiro #332: wsy=10.5% at ep20)
5. Loss-weighting surgery is exhausted: per-channel MSE multipliers (edward #304) falsified BOTH directions; rel-L2 aux loss (chihiro #254) falsified
6. NF=32 Fourier PE freqs is local optimum (NF=16≈NF=32, NF=64 converging toward NF=32 at ep11) — norman #239
7. FourierEmbed coordinate normalization needs fix (fern #360: ep5=9.03%, verifying structural fix)
8. EMA effect on standard recipe: NOT yet isolated cleanly (nezuko #383 Wave 9)

## Active PRs (Wave 14-15 fleet — 2026-05-02 19:21Z)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #239 | norman | NF sweep (16,32,64,128) on FourierPE | WIP |
| #342 | emma | 96k surface+volume scale-up | WIP (stale) |
| #347 | nezuko | Dedicated 2-block wall-shear sub-decoder | WIP |
| #361 | frieren | Weight-decay sweep {3e-4,1e-3,3e-3} | WIP |
| #382 | thorfinn | 6L/512d/8H + T_max=50 capacity scaling | WIP — ep4=8.510%, ep5 gate ETA ~20:10Z |
| #406 | edward | mlp_ratio={4,6,8} screen on 5L/256d | WIP (stale) |
| #407 | chihiro | mirror-aug+SW=2.0+gc=0.5 stacked on 5L/256d | WIP |
| #409 | fern | Coord-norm fix on 5L/256d + T_max=50 | WIP — ep5=8.9515% PASS; ep10 gate ETA ~21:50Z |
| #412 | askeladd | Heads sweep 4H vs 8H on 5L/256d | WIP — Trial A(8H) ep3=10.352%; ep5 ETA ~20:10Z |
| #417 | kohaku | EMA vs no-EMA on 5L/256d | WIP (stale) |
| #437 | alphonse | 6L/256d depth + T_max=50 | WIP — launched ~17:58Z; ep5 ETA ~22:10Z |
| #442 | senku | OHEM spatial hard-mining wsy/wsz | WIP |
| #443 | tanjiro | mirror-aug+SW=2.0 on 5L/256d + T_max=50 | WIP |
| #448 | stark | coord-only FiLM in surface decoder | WIP — ACK deadline 19:29Z |
| #455 | haku | SW=2.0 isolation on 5L/256d + T_max=50 | **NEW (Wave 15)** — ACK deadline ~19:55Z |
| #456 | violet | LR warmup=3ep on 5L/256d + T_max=50 | **NEW (Wave 15)** — ACK deadline ~19:55Z |
| #457 | gilbert | Dropout=0.1 on 5L/256d + T_max=50 | **NEW (Wave 15)** — ACK deadline ~19:55Z |

## Recent Closures (Wave 14)

| PR | Student | Reason |
|----|---------|--------|
| #445 | haku | ACK deadline 19:13Z missed (2nd consecutive miss) |
| #446 | violet | ACK deadline 19:14Z missed (2nd consecutive miss) |
| #447 | gilbert | ACK deadline 19:19Z missed (2nd consecutive miss) |

---

## Wave 10 — ALL CLOSED for non-compliance (2026-05-02 14:49Z)

| PR | Student | Hypothesis | Reason closed |
|----|---------|------------|---------------|
| #386 | askeladd | Heads sweep {4H,8H} on 5L/256d | Zero ACK + UNAUTH askeladd-sandwich-ln-r13 |
| #388 | haku | NF=32 + 5L/256d + T_max=50 | Zero ACK; deadline expired |
| #389 | kohaku | gc=0.5 + T_max=50 stack | Zero ACK + UNAUTH kohaku-ensemble-r19-adamw |
| #390 | nezuko | EMA + 5L/256d + T_max=50 | Zero ACK; deadline expired |
| #392 | edward | MLP-ratio sweep {4,6,8} | Zero ACK + UNAUTH edward-r25-muon-vs-adamw (6+ runs) |
| #397 | chihiro | Full stacked recipe | Zero ACK + UNAUTH klsmwdkr (PR #254) still running |
| #398 | fern | Coord-norm fix replay | Zero ACK; deadline expired |
| #399 | violet | DDP8 radford-champion | Zero ACK + UNAUTH i4w5ahtq (PR #330) still running |

**All Wave 10 hypotheses remain valid and may be reassigned once compliance returns.**

## Closed Wave 9/10 PRs (non-compliance)

| PR | Student | Reason |
|----|---------|--------|
| #378 | askeladd | 4th unauthorized sandwich-LN run after enforcement |
| #379 | frieren | REPEAT droppath violation; 30min deadline expired |
| #380 | haku | 30-min deadline expired; theta-wallshear runs still active |
| #381 | kohaku | 30-min deadline expired; ensemble runs still active |
| #383 | nezuko | 30-min deadline expired; zero ack |
| #254 | chihiro | wmax-ramp/raw-rel-l2 unauthorized; klsmwdkr still running |
| #330 | violet | 4 huber runs still running |
| #360 | fern | 4 tangent-loss runs still running |
| #386 | askeladd | Wave 10 zero-ACK + sandwich-LN unauth |
| #388 | haku | Wave 10 zero-ACK |
| #389 | kohaku | Wave 10 zero-ACK + ensemble-adamw unauth |
| #390 | nezuko | Wave 10 zero-ACK |
| #392 | edward | Wave 10 zero-ACK + 6 muon-vs-adamw unauth |
| #397 | chihiro | Wave 10 zero-ACK + klsmwdkr still running |
| #398 | fern | Wave 10 zero-ACK |
| #399 | violet | Wave 10 zero-ACK + i4w5ahtq still running |
| #402 | frieren | On pause; opened anyway with droppath-r24 unauth |

## Compliance Watch

- **Frieren (ON PAUSE)**: 2 consecutive droppath violations (#361, #379, #402). frieren-droppath-r24 still running 4 unauthorized runs.
- **Repeat-defiance pattern (PR closed AND unauth runs continuing)**: askeladd, kohaku, edward, chihiro, violet, fern, haku, nezuko. Multiple students have unauthorized side experiments (alphonse-string-multiscale, norman-surface-only-mask, thorfinn-asinh-r25, nezuko-warmdown-3ep) that are running on top of legitimate assigned runs.
- **Compliant (legit run + no obvious unauth)**: senku (`31s1j3a0`), tanjiro (`w3thlivw`), gilbert (`qiah2plu`), emma (`m7f6hrf7`), thorfinn (`5ifnf1wc` legit, but ALSO running thorfinn-asinh-r25 unauthorized).
- **Until compliance returns**, no new Wave 10 reassignments. Existing legit runs (#174, #239, #325, #332, #342, #346, #382) continue.

## Recently Closed (all waves)

| PR | Student | Reason |
|----|---------|--------|
| #360 | fern | 30min unauthorized enforcement deadline expired, tangent-loss runs still running |
| #330 | violet | 30min unauthorized enforcement deadline expired, huber runs still running |
| #254 | chihiro | 30min unauthorized enforcement deadline expired, wmax-ramp runs still running |
| #378 | askeladd | Defiance: 4th unauthorized sandwich-LN at enforcement moment |
| #379 | frieren | REPEAT droppath violation, 30min deadline expired |
| #380 | haku | 30min deadline expired, theta-wallshear still running |
| #381 | kohaku | 30min deadline expired, ensemble still running |
| #383 | nezuko | 30min deadline expired |
| #304 | edward | kuz4na0j running 2h49m after termination order; per-channel loss weights falsified |
| #361 | frieren | Ran droppath instead of assigned weight-decay (Wave 8) |
| #340 | thorfinn | Unauthorized experiments + idle |
| #341 | haku | Unauthorized experiments + 8h idle |
| #343 | kohaku | Zero comments; no W&B runs |
| #328 | askeladd | Arm B never launched; escalations ignored |

## Potential Next Research Directions (Wave 16+)

1. **Larger T_max** (T_max=100): The 5L/256d run hit its minimum at ep~45 out of 50 — trying T_max=100 could let the cosine schedule find a deeper minimum with the same initial LR.
2. **Equivariant shear heads**: SO(3)/SE(3) equivariant prediction for wsy/wsz — fundamental approach to binding constraint gap (5–7pp on wall-shear).
3. **Multi-scale attention / hierarchical heads**: Dual-resolution heads — coarse for volume, fine for surface boundary-layer regions where wall-shear is concentrated.
4. **Adaptive surface point sampling**: Oversample near-wall/boundary regions for surface training; reduces effective smoothing on wall-shear gradients.
5. **GNN backbone (replace Transformer)**: Message-passing on surface mesh graph — natural inductive bias for surface geometry.
6. **Physics-informed loss terms**: Continuity equation residual as auxiliary loss — particularly relevant for vol_p; may provide regularization on surface.
7. **NF=32 once norman #239 complete**: If NF=32 shows improvement over NF=16, update as a free ~0.3pp gain to stack on future experiments.
8. **Separate val/test gap investigation**: The ~2x vol_p degradation (val~4% → test~8-12%) suggests test distribution differs from val. Understanding this gap is a research priority before claiming AB-UPT wins.

## Targets

| Metric | Current Best (val, PR #174) | AB-UPT Target | Gap |
|--------|----------------------------|---------------|-----|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **6.9549%** | 4.51% | 2.44pp |
| `val_primary/surface_pressure_rel_l2_pct` | 4.5644% | 3.82% | 0.74pp |
| `val_primary/volume_pressure_rel_l2_pct` | 3.9361% ✓ | 6.08% | beats target |
| `val_primary/wall_shear_y_rel_l2_pct` | 8.7345% | 3.65% | **5.08pp** |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.5766% | 3.63% | **6.95pp** |

**val/test gap warning**: ~2x degradation on vol_p (val~4% → test~8-12%). Test_primary confirmation required before claiming AB-UPT wins.

## Constraints (hard)

- `--no-compile-model`: Mandatory (PyTorch 2.x Inductor crash at validation)
- `--fourier-pe`: Mandatory for comparability (n_params=3,249,813 confirms FourierEmbed)
- Kill-threshold operator: Use `<VALUE` (kill if metric NOT below VALUE)
- Epochs hard cap: `SENPAI_MAX_EPOCHS`; wall-clock: `SENPAI_TIMEOUT_MINUTES`
