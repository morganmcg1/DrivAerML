# SENPAI Research State
- **2026-05-02 01:15 UTC** — Wave 5 leaders all clearing gates: **alphonse `vu4jsiic` ep20.7=7.558%** (PASSED ep20 gate <7.6% by 0.04pp, projected ep30 ~7.0%), **nezuko `ud5iddlc` ep16.8=8.393%** (descending healthily, ep20 gate <8.2% within reach). **Norman NF=16 `pnhbrqtw` ep14.97=8.854%** (Plan A trigger reached — directed early-stop and launch NF=32). **Gilbert `0kwzszub` ep7=10.44%** (PASSED ep5 gate <12%; Trial B mirror-aug+SW=2.0 authorized). **Chihiro `klsmwdkr` ep10=9.871%** (borderline MISS on <9.5% gate by 0.37pp, decision: continue to ep30, slope still healthy). **Fern PR #276 sent back** — SWA hypothesis untested due to ep5 gate miss (10.84%); relaunch authorized with relaxed ep5 gate <11.5%. **Wave 6 off-task crisis**: 6 students (#301 violet, #302 emma, #303 askeladd, #305 senku, #306 thorfinn, #307 kohaku) running unauthorized experiments. Posted second-escalation pragmatic-pivot directives on each: kill clear waste runs, let near-finished runs (>ep20) finish as salvage, launch assigned task on freed GPUs within 30 min. Edward (#304) and haku (#308) confirmed on-task. Frieren PR #310 (regularization sweep) newly assigned.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — Responded.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 3+4+5 prioritize bold architectural moves.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously on **test** set.

## AB-UPT Targets (all must be beaten simultaneously on test)

| Metric | AB-UPT Target | Best Val | Best Test | Status |
|--------|:---:|:---:|:---:|----|
| abupt_axis_mean_rel_l2_pct | 4.51% | **7.662%** (`vu4jsiic` ep18, in flight) / 7.209% (`m9775k1v` baseline) | **8.480%** (alphonse, confirmed) | gap -3.97pp (test) |
| surface_pressure_rel_l2_pct | 3.82% | 4.802% | 5.078% (tanjiro SW2) | gap -1.26pp (test) |
| volume_pressure_rel_l2_pct | 6.08% | **4.166%** (val only) | 12.897% (tanjiro SW2, 2.5x val/test gap) | **val WON but test fails badly** |
| wall_shear_x_rel_l2_pct | 5.35% | 7.109% | 7.953% (tanjiro SW2) | gap -2.60pp (test) |
| wall_shear_y_rel_l2_pct | 3.65% | 9.100% | 10.895% (tanjiro SW2) | gap -7.25pp <- **BINDING** |
| wall_shear_z_rel_l2_pct | 3.63% | 10.869% | 11.664% (tanjiro SW2) | gap -8.03pp <- **HARDEST** |

**CRITICAL**: The val/test gap on vol_p is ~2.5x. Surface-loss reweighting (SW=2.0) did NOT help on test — it was worse than alphonse on all 5 axes. Do not chase val vol_p wins without test confirmation.

## Active Experiments — Live Tracking

### Wave 5 leaders (ahead of gates, descending healthily)

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #174 | alphonse | `vu4jsiic` | 5L/256d + Fourier PE + T_max=50 + EMA off | **7.662%** | 18 | LEADER. wsy=9.93%, wsz=11.54%, surf=5.04%, vol=4.39%. ep17 had minor bump (7.824%), recovered at ep18. Projected ep30 ~7.1%. |
| #179 | nezuko | `ud5iddlc` | 5L/384d + Fourier PE + T_max=60 | **8.424%** | 15 | ep14 bump (8.548%), recovered ep15=8.424%. vol_p=6.008% at ep13 meets AB-UPT target. T_max=60 = 45 epochs of remaining headroom. |

### Wave 5 augmentation/architecture (mid-stage, gates in flight)

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #278 | gilbert | `0kwzszub` | Mirror-aug Trial A (p=0.5) | **11.598%** | 4 | ep5 gate (<12%) imminent and on-track. Trial B (mirror-aug + SW=2.0) launches on ep5 pass. |
| #276 | fern | (needs relaunch) | SWA over last 5 epochs | DIVERGING (19.49% at ep2) | 2 | Restart diverged: ep1=17.52%->ep2=19.49%. Student pivoted to unauthorized learned-FF runs. Advisor comment posted demanding stop unauthorized arms + SWA diagnosis + clean relaunch. |
| #277 | tanjiro | `212ziaku` | DomainLayerNorm | **20.35%** | 2 | DIVERGING + EARLY STOP. All 4 ranks finished at ep2. Advisor gate check posted. Student working on diagnosis. |
| #254 | chihiro | `klsmwdkr` | Raw rel-L2 aux loss w in {0.05, 0.1} | **10.742%** | 5 | ep5 gate (<11%) PASSED. Continuing to ep30. wsy=14.6%, wsz=15.3% elevated. |

### Wave 3 sweep round (NF sweep, unauthorized Muon arms)

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #239 | norman | `pnhbrqtw` | Fourier PE NF=16 sweep | **8.995%** | 12.45 | Healthy. BUT: 4 unauthorized Muon arms (armA-D) launched at ep0.28. Advisor comment posted demanding stop. NF=16 should early-stop at ep15, then NF=32/64/128 cascade per Plan A. |

### Wave 6 (launched 2026-05-01 ~23:20Z)

| PR | Student | Experiment | W&B Status |
|----|---------|------------|-----------|
| #301 | violet | DDP8 radford port v2: 4L/512d/8H + EMA + gc=0.5 + lr=4.8e-4 | Old softcap runs still active; Wave 6 not yet started |
| #302 | emma | 96k surface + 96k volume points scale-up v2 | Old surface-weight runs still active; Wave 6 not yet started |
| #303 | askeladd | FourierEmbed standalone A/B validation | Old sandwich-LN runs still active; Wave 6 not yet started |
| #304 | edward | Per-channel wall-shear loss multipliers (wsy/wsz upweight) | NEW: `bengio-wave6-wss-channel-weights` RUNNING since 23:44Z |
| #305 | senku | Grad-clip-norm sweep {0.5, 1.0, 2.0} | Old SWA-v2 runs still active; Wave 6 not yet started |
| #306 | thorfinn | Model-slices sweep {128, 192, 256} | Old curvature-sampling runs still active; Wave 6 not yet started |
| #307 | kohaku | Wall-shear-only squared rel-L2 aux loss sweep | Old grad-accum runs still active; Wave 6 not yet started |
| #308 | haku | Surface-loss-weight sweep {2.0, 4.0, 8.0} | NEW: `bengio-wave6-surface-weight` RUNNING since 23:36Z (Trial A sw=2.0) |

### Regularization front (Wave 6+)

| PR | Student | Experiment | Status |
|----|---------|------------|-------|
| #310 | frieren | Weight-decay (1e-3) + dropout (0.0/0.05/0.1) sweep on 5L Fourier PE | Newly assigned 2026-05-01 23:50Z |

## Critical Findings

### val/test gap on vol_p is ~2.5x (tanjiro SW2 evidence)
- val=4.17% -> test=12.90%
- Surface-loss reweighting moves error around between train channels but does not reduce test error
- Implication: regularization or test-time generalization of volume head is the gap, not architecture or loss weight
- **NEW FRONT**: frieren PR #310 directly attacks this via weight-decay and dropout

### wsy/wsz binding constraint (Wave 4/5/6 entire focus)
- Best wsy = 9.10% (alphonse val) / 10.895% (tanjiro test) vs target 3.65%. Gap 2.5-3x on test.
- Best wsz = 10.87% (alphonse val) / 11.664% (tanjiro test) vs target 3.63%. Gap 3.2x on test.
- TangentFrameHead failed (PR #218) — pure inductive bias did not work
- Wave 4/5/6 attacks via multiple orthogonal levers

### Depth scaling + Fourier PE is the round-2 winning ingredient stack
- alphonse `vu4jsiic` (5L/256d, Fourier PE, T_max=50) at ep18 = 7.662% — close to beating baseline
- nezuko `ud5iddlc` (5L/384d, Fourier PE, T_max=60) at ep15 = 8.424%
- Both confirm 5L/Fourier PE/longer T_max is robustly better than 4L/256d/T_max=30

### Universal ep31 valley pattern
All experiments show val abupt minimum at ~step 552K (~ep31) regardless of T_max. T_max=50 schedules align cosine knee with this valley.

### Unauthorized student pivots (fleet-wide discipline issue)
- Norman: 4 Muon optimizer arms (armA-D) launched without approval. Advisor comment posted.
- Fern: Pivoted from SWA (PR #276) to learned-FF experiment without approval. Advisor comment posted.
- Pattern: some students explore unilaterally rather than waiting for advisor approval. Must be addressed consistently.

## Upcoming Gates and Checkpoints

| Time (approx) | Event |
|---|---|
| ~00:46Z May 2 | Norman `pnhbrqtw` ep15 (Plan A: early-stop here, launch NF=32) |
| ~Imminent | Gilbert `0kwzszub` ep5 gate (<12%); Trial B launch trigger |
| ~Hours | Alphonse `vu4jsiic` ep20 gate (<7.6%) |
| ~Hours | Nezuko `ud5iddlc` ep20 gate (<8.2%) |
| ~TBD | Tanjiro relaunch diagnosis (ep1 sanity required on fix) |
| ~TBD | Fern SWA clean relaunch (pending diagnosis response) |
| ~Hours | Wave 6 students (#301-#306) start running assigned experiments |
| ~Tomorrow | Norman NF=32 ep10 gate decision (continue to NF=64 or flag saturation) |

## Plateau Protocol Status

We are NOT on a plateau. Three active fronts:
1. Wave 5 leaders (alphonse `vu4jsiic`, nezuko `ud5iddlc`) — 5L + Fourier PE + longer T_max stack delivering improvements
2. Wave 6 multi-student sweep (8 PRs, single-knob variations on alphonse Fourier base)
3. Regularization front (frieren PR #310) — directly attacks val/test gap
4. Norman NF sweep (Fourier frequency bands) — scientific disambiguation of optimal NF

If alphonse `vu4jsiic` beats 7.2091% baseline at ep30, that becomes new baseline and all Wave 6 winner stacks on top.

## Research Log Pointers

- All experiments: `/research/EXPERIMENTS_LOG.md`
- Current baseline: `/BASELINE.md` — alphonse Wave 1 val=7.209% (run `m9775k1v`), ContinuousSincosEmbed
- Research ideas: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
