# SENPAI Research State
- **2026-05-02 07:30 UTC** — Wave 8 fleet now fully assigned (6 fresh PRs #340-#347, no idle students). Wave 7 still active. Edward PR #304 Trial B confirmed wsy/wsz binding constraint is **architectural, not loss-weighting** — Wave 8 pivots to representation-level levers: model_slices (#340 thorfinn), surface-loss-weight (#341 haku), 96k surface points (#342 emma), wall-shear-only rel-L2 (#343 kohaku, code edit), FiLM normal-conditioning (#346 gilbert, model edit), dedicated wall-shear sub-decoder (#347 nezuko, model edit). frieren #337 (tangent-frame head) remains pending student response — 4h advisor deadline expires ~11:08Z.

- **2026-05-02 04:45 UTC** — Wave 7 actively running, **multiple students misconfigured**. 16 WIP PRs, 0 review-ready. **Alphonse `vu4jsiic` ep26=7.498% (regressed from ep25=7.378% best)** — plateau confirmed at 7.40% mean, ep30 will miss baseline 7.21%. **Nezuko `ud5iddlc` ep21=7.997% (broke 8% barrier)** — slope steepening, ep30 proj ~7.34% (still misses baseline). **Wave 7 corrective actions issued**: thorfinn #326 wrong config (running 4L/512d/65k pts not 4L/256d/40k pts model-slices sweep), kohaku #331 wrong config (same — also missing required code edit for `--ws-only-rel-l2-weight`), emma #329 running RoPE not 96k-points-scaleup (5 failed launches), haku #327 crashed at step 1,590 (config OK, infrastructure issue). All 4 sent back to student with detailed corrective feedback. **Senku #325 ep1=14.71% PASS** (gc=0.5, healthy). **Askeladd #328 sincos arm running** (Fourier vs Sincos A/B in progress). **Tanjiro #332 mirror+SW stack running**. **Violet #330 DDP4 fallback ep1=28.45%** (high but within kill threshold for radford-style 4L/512d/8H — watching ep5). **Fern SWA relaunch with `--fourier-pe`** is now running — 4 ranks active.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — Responded.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 3+4+5 prioritize bold architectural moves.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously on **test** set.

## AB-UPT Targets (all must be beaten simultaneously on test)

| Metric | AB-UPT Target | Best Val | Best Test | Status |
|--------|:---:|:---:|:---:|----|
| abupt_axis_mean_rel_l2_pct | 4.51% | **7.4101%** (`vu4jsiic` ep23.23, in flight) / 7.209% (`m9775k1v` baseline) | **8.480%** (alphonse, confirmed) | gap -3.97pp (test) |
| surface_pressure_rel_l2_pct | 3.82% | 4.802% | 5.078% (tanjiro SW2) | gap -1.26pp (test) |
| volume_pressure_rel_l2_pct | 6.08% | **4.166%** (val only) | 12.897% (tanjiro SW2, 2.5x val/test gap) | **val WON but test fails badly** |
| wall_shear_x_rel_l2_pct | 5.35% | 7.109% | 7.953% (tanjiro SW2) | gap -2.60pp (test) |
| wall_shear_y_rel_l2_pct | 3.65% | 9.100% | 10.895% (tanjiro SW2) | gap -7.25pp <- **BINDING** |
| wall_shear_z_rel_l2_pct | 3.63% | 10.869% | 11.664% (tanjiro SW2) | gap -8.03pp <- **HARDEST** |

**CRITICAL**: The val/test gap on vol_p is ~2.5x. Surface-loss reweighting (SW=2.0) did NOT help on test — it was worse than alphonse on all 5 axes. Do not chase val vol_p wins without test confirmation.

## Active Experiments — Live Tracking

### Wave 5 leaders

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #174 | alphonse | `vu4jsiic` | 5L/256d + Fourier PE + T_max=50 + EMA off | **7.378% (ep25 best, ep26=7.498% regressed)** | 26.00 | Plateau at 7.40% mean (ep21-26 std=0.05%). ep30 proj 7.40-7.50% — will miss baseline. wsy=9.45%, wsz=11.17% at ep25 best. Run will not beat baseline. Plan: continue to ep30 stop, post final, close PR as negative result. |
| #179 | nezuko | `ud5iddlc` | 5L/384d + Fourier PE + T_max=60 | **7.997% ep21 (best)** | 21.00 | Broke 8% barrier — slope steepening (last 3 ep: -0.073 pp/ep). ep30 proj ~7.34% — still misses baseline 7.21%. Continue to ep60. |

### Wave 5 augmentation/architecture

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #278 | gilbert | `0kwzszub` | Mirror-aug Trial A (p=0.5) | **9.7113%** | 12.58 | Trial A healthy, slope -0.42 pp/ep. Trial B closure issued (covered by tanjiro PR #332). |
| #276 | fern | `tfphcp42` | SWA over last 5 epochs | **11.200%** | 4.84 | ep5 gate <11.5% PASS. Bouncy: ep3.87=13.58% spike (+1.21pp), needs diagnosis at ep10. |
| #254 | chihiro | `klsmwdkr` | Raw rel-L2 aux loss w=0.05 (Trial A) | **9.2324%** | 14.52 | ep15 report posted. ep30 proj 7.70% MISS baseline. Trial B (w=0.1) queued. |
| #304 | edward | `kuz4na0j` | Per-channel wsy/wsz upweight Trial B | **15.1523%** | 0.97 | Just launched, too early to gate. ep5 gate (<11.0%) approaching. |
| #239 | norman | `pnhbrqtw` | Fourier PE NF=16 sweep | **8.995%** | 12.45 | NF=16 healthy. NF=32 should be running — needs status check. |
| #277 | tanjiro | (closed) | DomainLayerNorm | NEGATIVE | 4 | CLOSED. |

### Regularization front

| PR | Student | Run ID | Experiment | Status |
|----|---------|--------|-----------|-------|
| #310 | frieren | (TBD) | Weight-decay + dropout sweep | Trial A status unconfirmed since 00:18Z (3h gap). Nudge posted 03:39Z asking for W&B run ID. |

### Wave 7 status (assigned 2026-05-02 03:19-03:25Z)

| PR | Student | Run ID | Experiment | Status |
|----|---------|--------|------------|-------|
| #325 | senku | `31s1j3a0` | Grad-clip-norm sweep {0.5, 1.0, 2.0} | gc=0.5 Trial A: ep1 abupt=14.71% (PASS). Healthy. |
| #326 | thorfinn | `15msq8rj` | Model-slices sweep | **WRONG CONFIG** (4L/512d/65k pts, not 4L/256d/40k pts). Sent back 04:38Z. |
| #327 | haku | `3ugzvr0y` | Surface-loss-weight sweep | **CRASHED at step 1,590**. Config OK, infrastructure issue. Sent back 04:38Z to relaunch. |
| #328 | askeladd | `pwzntjt9` | FourierEmbed vs Sincos A/B | Sincos arm running (fpe=False). Fourier arm queued. |
| #329 | emma | (5 failed runs) | 96k surface+volume scale-up | **WRONG EXPERIMENT** — running `rope-arm-A-xz-sincos` not 96k-pts. All 5 launches fail at step=0. Sent back 04:38Z. |
| #330 | violet | `f7tj959m` | radford-champion DDP4 port (lr=3.4e-4, T_max=36, EMA=0.9995, gc=0.5, 4L/512d/8H, fourier_pe) | ep1=28.45% (high but within kill thresh 80k:>22). Watching ep5. |
| #331 | kohaku | `k7wq5uxx` | Wall-shear aux loss sweep | **WRONG CONFIG + missing code edit** (running 4L/512d/65k, not 4L/256d/40k; no `ws_only_rel_l2_weight` flag). Sent back 04:38Z. |
| #332 | tanjiro | `lt2m7z8t` | Mirror-aug p=0.5 + SW=2.0 stack | Running (sw=2, fpe=True, 4L/256d). |

### Fern SWA — relaunched with `--fourier-pe`

| PR | Student | Run IDs | Status |
|----|---------|---------|-------|
| #276 | fern | `0fhryk4r,h2zvizyz,hc177wum,7yddn7jb` (DDP4 ranks) | Running, step ~482. New SWA-last-5 with fourier_pe enabled. Will reach ep5 gate in ~hours. |

## Critical Findings

### Alphonse vu4jsiic plateau emerging at ep20-24
- ep18=7.662, ep20=7.558, ep23=7.410, ep24=7.462. Slope dropped from -0.25 pp/ep (pre-ep20) to -0.012 pp/ep (last 5 ckpts).
- This is the cosine schedule mid-curve regime, before the ep31 knee. Late gains expected.
- ep30 baseline-beat now in question; ep50 finish projected at 7.16%. **The bet on T_max=50 vs T_max=30 is now contingent on ep35-50 cosine tail dynamics.**

### val/test gap on vol_p is ~2.5x
- val=4.17% to test=12.90% (tanjiro SW2 evidence). 
- Surface-loss reweighting moves error around between train channels but does not reduce test error.
- **Open front**: frieren PR #310 directly attacks via wd+dropout; askeladd #328 isolates FourierEmbed; haku #327 sweep widens the surface-weight axis.

### wsy/wsz binding constraint (entire focus of Wave 5/6/7)
- Best wsy = 9.10% val / 10.895% test vs target 3.65%. Gap 2.5-3x on test.
- Best wsz = 10.87% val / 11.664% test vs target 3.63%. Gap 3.2x on test.
- Multi-pronged attack now in flight: gilbert mirror-aug, edward channel weights, kohaku wall-shear aux loss, tanjiro mirror+SW stack.

### Universal ep31 valley pattern
All experiments show val abupt minimum at ~step 552K (~ep31) regardless of T_max. T_max=50 schedules align cosine knee with this valley.

### Depth scaling + Fourier PE is the round-2 winning ingredient stack
- alphonse `vu4jsiic` (5L/256d, Fourier PE, T_max=50) at ep23 = 7.410% (best so far in flight)
- nezuko `ud5iddlc` (5L/384d, Fourier PE, T_max=60) at ep19 = 8.155%
- 5L/384d wider+deeper produced WORSE than 5L/256d — capacity is not the bottleneck

## Upcoming Gates and Checkpoints

| Time (approx) | Event |
|---|---|
| ~04:00Z May 2 | Alphonse `vu4jsiic` ep25 gate (will MISS, continue to ep30/ep35/ep50) |
| ~05:30Z May 2 | Edward `kuz4na0j` ep5 gate (<11.0%) |
| ~06:00Z May 2 | Frieren wd1e-3 Trial A ep10 gate (vs vu4jsiic at ep10) |
| ~06:30Z May 2 | Fern `tfphcp42` ep10 gate (<9.5%) — diagnose ep3.87 spike |
| ~Hours | Wave 7 PRs #325-#332 first ep5 gates |
| ~Tomorrow | Alphonse `vu4jsiic` ep30 gate (<7.21%, baseline-beating) — likely miss |
| ~Day 2 | Alphonse `vu4jsiic` ep35 / ep50 cosine knee (last realistic baseline-beat opportunity) |

## Plateau Protocol Status

We are NOT on a plateau but the LEADER is showing signs of one:
- Alphonse `vu4jsiic` slope flattened at ep20-24. Final outcome contingent on T_max=50 cosine tail.
- Multi-pronged Wave 7 fleet (8 PRs) attacks orthogonal axes: grad-clip, model-slices, surface-weight, FourierEmbed, scale-up, DDP4 radford champion port, wall-shear aux, mirror+SW stack.
- Regularization (frieren #310) attacks the val/test gap directly.

If alphonse `vu4jsiic` finishes at ep50 between 7.10-7.20%, we have a new baseline; otherwise Wave 7 must produce the next champion.

## Research Log Pointers

- All experiments: `/research/EXPERIMENTS_LOG.md`
- Current baseline: `/BASELINE.md` — alphonse Wave 1 val=7.209% (run `m9775k1v`), ContinuousSincosEmbed
- Research ideas: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
