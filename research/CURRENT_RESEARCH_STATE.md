# SENPAI Research State

- **2026-05-02 13:15Z — Wave 9 launched. 6 fresh assignments. Wave 8 compliance window closed.**
- **Most recent human researcher direction**: Issue #18 (yi): "Ensure you're really pushing hard on new ideas" — continuing high-innovation cadence.

## Current Research Focus

**Primary goal**: Bring `val_primary/abupt_axis_mean_rel_l2_pct` below the AB-UPT target of 4.51%.  
**Current best (in-flight, not yet merged)**: alphonse #174, ep~40 = **7.0283%** (run `vu4jsiic`, T_max=50, 5L/256d + Fourier). Official BASELINE.md best: 7.2091% (PR #74).

**Active binding constraints**: wsy=8.87% (target 3.65%: **5.22pp gap**), wsz=10.65% (target 3.63%: **7.02pp gap**).

**Key mechanism findings (consolidated)**:
1. T_max=50 cosine schedule is a validated real lever (alphonse #174: −0.181pp)
2. gc=0.5 grad-clip is a validated real lever (senku #325: ep18=8.586%, strong descent)
3. mirror-aug + SW=2.0 stacked is the strongest binding-axis signal (tanjiro #332: wsy=11.16% at ep10)
4. Loss-weighting surgery is exhausted: per-channel MSE multipliers (edward #304) and rel-L2 aux loss (chihiro #254) both falsified
5. NF=32 Fourier PE freqs is local optimum (NF=64 uniformly worse — norman #239 U-shape)
6. FourierEmbed coordinate normalization needs fix (fern #360 verifying — ep5 PASS 9.03%)

## Active PRs (Wave 7/8)

| PR | Student | Status | Key metrics / gates |
|----|---------|--------|---------------------|
| #174 | alphonse | **NEW BEST 7.0283% at ep~40**, run `vu4jsiic` | Still running to ep50; projected ~6.95–7.05% |
| #239 | norman | NF=64 ep5=10.363% PASS (worse than NF=32), run `yilzrnwk` | ep10 gate pending; U-shape confirmed; NF=32 optimum |
| #254 | chihiro | FALSIFIED + ABORTED, run `klsmwdkr` | ep30=8.236%; run to ep50 doc; awaiting write-up |
| #304 | edward | Trial B ep16.2=8.709%, run `kuz4na0j` | TERMINATION ORDER (10:33Z); edward 2.5h+ silent; escalation posted |
| #325 | senku | **STRONG**: ep18=8.586%, run `31s1j3a0` (gc=0.5) | ep30 projected ~7.4–7.8%; ep20 report due |
| #330 | violet | **STRONG**: ep5=8.716%, run `i4w5ahtq` (4L/512d/8H+EMA+gc=0.5) | ep10 gate <8.0% next; trajectory ~7.0–7.5% |
| #332 | tanjiro | ep17=8.37%, run `w3thlivw` (mirror+SW=2.0) | ep30 gate ≤8.0%; Trial B queued; CFI runs killed per compliance |
| #342 | emma | ep5=11.44% PASS, run `m7f6hrf7` (96k pts) | 25.3GB VRAM OK; ep10 gate <11% |
| #346 | gilbert | FiLM v2 run `qiah2plu` (DDP fix, find_unused_params=True) | Ep1+ running; +137k params (+4.23%) |
| #360 | fern | ep5=9.03% PASS (1.5pp headroom), run `q40rez85` | Coord-norm fix validated; ep10 gate |

## Wave 9 PRs (just assigned, not yet started)

| PR | Student | Hypothesis | Priority |
|----|---------|------------|----------|
| #378 | askeladd | Stack mirror+SW=2.0+T_max=50 on 5L/256d | **Highest** — stacks 3 validated levers |
| #379 | frieren | Weight-decay sweep {3e-4, 1e-3, 3e-3} | Medium — diagnostic sweep |
| #380 | haku | Surface-loss-weight sweep {2.0, 4.0, 8.0} | Medium — SW=2.0 confirmed by tanjiro; find optimum |
| #381 | kohaku | Stack gc=0.5 + T_max=50 on 4L/256d | **High** — two validated levers, shortest path to new best |
| #382 | thorfinn | 6L/512d/8H + T_max=50 capacity scaling | High — capacity frontier |
| #383 | nezuko | EMA + T_max=50 on 4L/256d canonical recipe | Medium — EMA isolation test |

## Closed in Wave 8 compliance window

| PR | Student | Reason |
|----|---------|--------|
| #340 | thorfinn | Unauthorized experiments; no compliance |
| #341 | haku | Unauthorized experiments; 8h idle after finish |
| #343 | kohaku | Zero comments; no W&B runs in group; deadline expired |
| #328 | askeladd | Arm B never launched; 2+ escalations ignored |

## Targets

| Metric | Current Best (val) | AB-UPT Target | Gap |
|--------|--------------------|---------------|-----|
| `val_primary/abupt_axis_mean_rel_l2_pct` | **7.0283** (alphonse #174 ep40) | 4.51 | 2.52pp |
| `val_primary/surface_pressure_rel_l2_pct` | 4.62 | 3.82 | 0.80pp |
| `val_primary/volume_pressure_rel_l2_pct` | 3.97 ✓ | 6.08 | beats target |
| `val_primary/wall_shear_y_rel_l2_pct` | 8.87 | 3.65 | **5.22pp** |
| `val_primary/wall_shear_z_rel_l2_pct` | 10.65 | 3.63 | **7.02pp** |

**val/test gap warning**: ~2x degradation on vol_p (val=4.17% → test~8-12%). Test_primary confirmation required before claiming AB-UPT wins.

## Potential Next Research Directions (Wave 10+)

1. **Stacked recipe T_max=70**: If askeladd #378 (mirror+SW=2.0+T_max=50) shows continued descent at ep50, extend to T_max=70 with SGDR restarts
2. **gc=0.5 + mirror + SW=2.0 + T_max=50 full stack**: If kohaku #381 and askeladd #378 both win, their best configs can be combined
3. **Equivariant shear heads**: SO(3)/SE(3) equivariant prediction for wsy/wsz (not yet attempted)
4. **Multi-scale attention**: Dual-resolution heads — coarse for volume, fine for surface boundary-layer regions
5. **Ensembling**: Top-K seed averaging; SWA over cosine restarts
6. **NF=32 everywhere**: Norman #239 result (once confirmed) should trigger updating all future baselines to use `--fourier-pe-num-freqs 32`
7. **Coordinate normalization audit**: If fern #360 confirms the +1.0–1.5pp structural gap, replay Wave 5–8 winners with the fix
8. **FiLM + longer schedule**: If gilbert #346 FiLM shows signal, stack with T_max=50

## Constraints (hard)

- `--no-compile-model`: Mandatory (PyTorch 2.x Inductor crash at validation)
- `--fourier-pe`: Mandatory for comparability (n_params=3,249,813 confirms FourierEmbed)
- Kill-threshold operator: Use `<VALUE` (kill if metric NOT below VALUE)
- Epochs hard cap: `SENPAI_MAX_EPOCHS`; wall-clock: `SENPAI_TIMEOUT_MINUTES`
