# SENPAI Research State
- **2026-05-02 02:30 UTC** — **Alphonse `vu4jsiic` ep23.5=7.419%** (passed ep20 gate, descending steadily toward ep25 gate <7.3% and ep30 gate <7.21% baseline-beating threshold). **Nezuko `ud5iddlc` ep19.0=8.217%** (ep20 gate <8.2% imminent; close call, on-track). **Gilbert `0kwzszub` ep10.4=9.834%** (Trial A healthy, well clear of all gates; Trial B authorization unconfirmed for 1.5h, follow-up posted). **Tanjiro `4t75zm3j` (DomainLN v3) ep4=11.45%** (gate at step 80k <10.5% will fire imminently — direction confirmed NEGATIVE across two seeds, mechanism characterized: per-domain affine biases TransolverAttention slice computation; closure pending). **Wave 6 unresponsive crisis**: 6 students (#301 violet, #302 emma, #303 askeladd, #305 senku, #306 thorfinn, #307 kohaku) still no response after 2 escalations; deadline 04:30Z (~2h) for closure+reassignment. **Frieren PR #310** Trial A running with `--lr-warmup-epochs 5 --wd 1e-3 --no-dropout` since ~23:58Z (survey-prs stale flag was a label artifact). Edward (#304) and haku (#308) on-task.

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
| #174 | alphonse | `vu4jsiic` | 5L/256d + Fourier PE + T_max=50 + EMA off | **7.419%** | 23.5 | LEADER. wsy=9.52%, wsz=11.20%. PASSED ep20 gate at 7.558%. ep25 gate <7.3% next, ep30 gate <7.21% must beat baseline to merge. |
| #179 | nezuko | `ud5iddlc` | 5L/384d + Fourier PE + T_max=60 | **8.217%** | 19.0 | wsy=10.16%, wsz=11.91%. ep20 gate <8.2% imminent (within 0.02pp). Slope ~0.05pp/ep, on track. T_max=60 leaves 40 ep headroom. |

### Wave 5 augmentation/architecture (mid-stage, gates in flight)

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #278 | gilbert | `0kwzszub` | Mirror-aug Trial A (p=0.5) | **9.834%** | 10.4 | Trial A healthy, well past ep5 gate. Trial B authorization unconfirmed (1.5h elapsed); follow-up comment posted 02:28Z. |
| #276 | fern | (needs relaunch) | SWA over last 5 epochs | DIVERGING (19.49% at ep2) | 2 | Restart diverged: ep1=17.52%->ep2=19.49%. Student pivoted to unauthorized learned-FF runs. Advisor comment posted demanding stop unauthorized arms + SWA diagnosis + clean relaunch. |
| #277 | tanjiro | `4t75zm3j` (v3) | DomainLayerNorm | **11.45%** | 4 | NEGATIVE confirmed across v2+v3. Mechanism: per-domain affine biases TransolverAttention slice computation. Kill gate at step 80k (<10.5%) imminent. Closure pending. |
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
| ~02:48Z May 2 | Tanjiro `4t75zm3j` step-80k kill gate (<10.5%) — will fire; closure comment then |
| ~03:30Z May 2 | Alphonse `vu4jsiic` ep25 gate (<7.3%) |
| ~04:00Z May 2 | Nezuko `ud5iddlc` ep20 gate (<8.2%) |
| **~04:30Z May 2** | **Wave 6 unresponsive PR deadline — close+reassign #301/#302/#303/#305/#306/#307 if no student response** |
| ~06:00Z May 2 | Frieren `wd1e-3-no-dropout` Trial A ep10 gate (vs vu4jsiic at ep10) |
| ~Hours | Alphonse `vu4jsiic` ep30 gate (<7.21%, baseline-beating) |
| ~TBD | Fern SWA clean relaunch (pending diagnosis response) |
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
