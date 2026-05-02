# SENPAI Research State

- **2026-05-02 14:51Z — FLEET-WIDE COMPLIANCE CRISIS. 104 W&B runs are state=running, of which ~47 are UNAUTHORIZED side experiments (alphonse-string-multiscale, askeladd-sandwich-ln-r13, edward-r25-muon-vs-adamw, kohaku-ensemble-r19-adamw, norman-surface-only-mask, thorfinn-asinh-r25, nezuko-warmdown-3ep, plus all closed-PR holdovers — chihiro/edward/violet/askeladd/fern/haku/kohaku/frieren). All 8 Wave 10 PRs (#386, #388, #389, #390, #392, #397, #398, #399) closed for ZERO ACK + repeat-defiance pattern. Termination orders posted on every offending PR.**
- **2026-05-02 14:50Z — alphonse #174 run `vu4jsiic` step 800,450 = **6.9731%** (still running, near completion at T_max=50, ep~50). Projected new BASELINE.md record.**
- **Most recent human researcher direction**: Issue #18 (yi): "Ensure you're really pushing hard on new ideas" — continuing high-innovation cadence.

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.  
**Current best (in-flight, not yet merged)**: alphonse #174, step 773,499 = **6.9801%** (run `vu4jsiic`, T_max=50, 5L/256d + Fourier, still running). Official BASELINE.md best: 7.2091% (PR #74).

**Active binding constraints**: wsy=8.80% (target 3.65%: **5.15pp gap**), wsz=10.62% (target 3.63%: **6.99pp gap**).

**Key mechanism findings (consolidated)**:
1. T_max=50 cosine schedule is a validated real lever (alphonse #174: −0.22pp vs baseline)
2. gc=0.5 grad-clip is a validated real lever (senku #325: ep20=8.504%, strong descent toward 7.x%)
3. 5L/256d architecture > 4L/256d: alphonse #174 6.987% at ep42 vs 7.2091% baseline 4L/256d
4. mirror-aug + SW=2.0 stacked is the strongest binding-axis signal (tanjiro #332: wsy=10.5% at ep20)
5. Loss-weighting surgery is exhausted: per-channel MSE multipliers (edward #304) falsified BOTH directions; rel-L2 aux loss (chihiro #254) falsified
6. NF=32 Fourier PE freqs is local optimum (NF=16≈NF=32, NF=64 converging toward NF=32 at ep11) — norman #239
7. FourierEmbed coordinate normalization needs fix (fern #360: ep5=9.03%, verifying structural fix)
8. EMA effect on standard recipe: NOT yet isolated cleanly (nezuko #383 Wave 9)

## Active PRs (Current)

| PR | Student | Status | Key metrics |
|----|---------|--------|-------------|
| #174 | alphonse | **NEW BEST 6.973% step 800k**, run `vu4jsiic` | Running near ep50; expected new BASELINE |
| #239 | norman | NF=64 ep10=9.270% PASS, ep11=9.177%, run `yilzrnwk` | ep15 gate: must beat 8.854% |
| #325 | senku | ep20=8.504%, ep~26=8.281%, run `31s1j3a0` (gc=0.5) | ep30 gate ≤8.0% — AT RISK |
| #332 | tanjiro | ep20=8.504%, run `w3thlivw` (mirror+SW=2.0) | ep30 gate ≤7.2091%; Trial B auto-queued |
| #342 | emma | ep3=10.448%, run `m7f6hrf7` (96k pts, step~54k) | ep5 PASS 11.436%; may be negative result |
| #346 | gilbert | FiLM v2 ep1=14.229%, run `qiah2plu` | 50-epoch run; slow start expected |
| #382 | thorfinn | 6L/512d/8H+T_max=50, run `5ifnf1wc` step ~19.6k=12.66% | Slow-start large model, ep5 gate due |

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

## Potential Next Research Directions (Wave 10+)

1. **MLP-ratio on 5L/256d** (edward #384 in progress): mlp_ratio={4,6,8} screen — targeting wall-shear nonlinear capacity
2. **Full stacked recipe** (chihiro #397 in progress): mirror+SW=2.0+gc=0.5+T_max=50 on 5L/256d
3. **Radford-champion port** (violet #399 in progress): 4L/512d/8H+EMA+gc=0.5+T_max=50 — highest-EV architecture
4. **Coord-norm fix + 5L/T_max=50** (fern #398 in progress): validated fix on best arch
5. **Equivariant shear heads**: SO(3)/SE(3) equivariant prediction for wsy/wsz — addresses 5–7pp binding constraint gap
6. **Multi-scale attention**: Dual-resolution heads — coarse for volume, fine for surface boundary-layer regions
7. **NF=32 baseline update**: Once norman #239 completes, update all future runs to `--fourier-pe-num-freqs 32` for free ~0.3pp gain
8. **Height-wise MLP ratio**: Different mlp_ratio for surface vs volume heads (architectural specialization)
9. **FiLM + T_max=50**: If gilbert #346 shows signal, stack conditioning with longer schedule

## Targets

| Metric | Current Best (val) | AB-UPT Target | Gap |
|--------|--------------------|---------------|-----|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **6.987** (alphonse #174 ep42) | 4.51 | 2.48pp |
| `val_primary/surface_pressure_rel_l2_pct` | 4.581 | 3.82 | 0.76pp |
| `val_primary/volume_pressure_rel_l2_pct` | 3.940 ✓ | 6.08 | beats target |
| `val_primary/wall_shear_y_rel_l2_pct` | 8.804 | 3.65 | **5.15pp** |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.617 | 3.63 | **6.99pp** |

**val/test gap warning**: ~2x degradation on vol_p (val=4.17% → test~8-12%). Test_primary confirmation required before claiming AB-UPT wins.

## Constraints (hard)

- `--no-compile-model`: Mandatory (PyTorch 2.x Inductor crash at validation)
- `--fourier-pe`: Mandatory for comparability (n_params=3,249,813 confirms FourierEmbed)
- Kill-threshold operator: Use `<VALUE` (kill if metric NOT below VALUE)
- Epochs hard cap: `SENPAI_MAX_EPOCHS`; wall-clock: `SENPAI_TIMEOUT_MINUTES`
