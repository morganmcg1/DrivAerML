# SENPAI Research State

- **2026-05-02 ~late (Wave 5 mid-flight, sweep round)** — Wave 5 is now the dominant front: PR #174 (alphonse 5L/256d + Fourier PE + T_max=50, run `vu4jsiic`) and PR #179 (nezuko 5L/384d + Fourier PE + T_max=60, run `ud5iddlc`) are both pre-clearing their gate thresholds with abupt trajectories below the 7.21% baseline path. PR #239 (norman Fourier PE num_freqs sweep) is healthy at ep9, NF=32 arm pending launch. PR #276 (fern SWA last-5) and PR #278 (gilbert mirror-aug) are early-stage with kill gates active. FourierEmbed (PR #176) merged to bengio; lr=3e-4 confirmed optimal.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — Responded.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 3+4+5 prioritize bold architectural moves.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously on **test** set.

## AB-UPT Targets (all must be beaten simultaneously on test)

| Metric | AB-UPT Target | Best Val | Best Test | Status |
|--------|:---:|:---:|:---:|----|
| abupt_axis_mean_rel_l2_pct | 4.51% | **7.667%** (`vu4jsiic` ep16, in flight) / 7.209% (`m9775k1v` baseline) | **8.480%** (alphonse, confirmed) | gap −3.97pp (test) |
| surface_pressure_rel_l2_pct | 3.82% | 4.802% | 5.078% (tanjiro SW2) | gap −1.26pp (test) |
| volume_pressure_rel_l2_pct | 6.08% | **4.166%** (val only) | 12.897% (tanjiro SW2, 2.5× val/test gap) | **val WON but test fails badly** |
| wall_shear_x_rel_l2_pct | 5.35% | 7.109% | 7.953% (tanjiro SW2) | gap −2.60pp (test) |
| wall_shear_y_rel_l2_pct | 3.65% | 9.100% | 10.895% (tanjiro SW2) | gap −7.25pp ← **BINDING** |
| wall_shear_z_rel_l2_pct | 3.63% | 10.869% | 11.664% (tanjiro SW2) | gap −8.03pp ← **HARDEST** |

**CRITICAL**: The val/test gap on vol_p is ~2.5×. Surface-loss reweighting (SW=2.0) did NOT help on test — it was worse than alphonse on all 5 axes. Do not chase val vol_p wins without test confirmation.

## Active Experiments — Live Tracking

### Wave 5 leaders (ahead of gates)

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #174 | alphonse | `vu4jsiic` | 5L/256d + Fourier PE + T_max=50 + EMA off | **7.667%** | 16 | LEADER. ep20 gate <7.8% PRE-CLEARED. vol_p=4.39% (well under target). Plateau ep10-13 cleared at ep14. ep30 projected <7.2%. |
| #179 | nezuko | `ud5iddlc` | 5L/384d + Fourier PE + T_max=60 | **8.410%** | 13 | ep15 gate <8.5% PRE-CLEARED. **vol_p=6.008% MEETS AB-UPT target** (6.08%). T_max=60 means 47 epochs of remaining headroom. |

### Wave 3 sweep round (in flight)

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #239 | norman | `pnhbrqtw` | Fourier PE num_freqs sweep, NF=16 arm | **descending** | 9 | All gates pass. ep6 minor bump (10.12% vs ep5 10.07%) noted. vol_p=6.855% approaching target. NF=32 arm to launch at ep15. |

### Wave 4/5 augmentation experiments (early-stage)

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #276 | fern | `ep4dl3uw` | SWA over last 5 epochs (26-30) | 14.436% | 2.13 | ep5 kill gate <10.5% borderline. Verification checklist posted (BN update, eval path consistency). |
| #278 | gilbert | `0kwzszub` | Mirror-aug (p=0.5 y-flip, negate wsy) | 17.910% | 1.25 | step-35k kill gate <20% will pass. Trial B (SW=2.0 + mirror-aug) to launch after Trial A ep5. |

### Wave 2/3 leftovers

| PR | Student | Run ID | Status |
|----|---------|--------|--------|
| #75 | fern | `uz4em31o` | T_max=30 + lr=5e-4 + Fourier PE Trial B — terminal soon |
| #79 | emma | `3evzgru1` | 60k pts + Fourier PE + T_max=50 Trial B v2 — descending; merge conflict |
| #80 | tanjiro | `0qjbutkd` | SW=2.0 + T_max=50 — gates passing |
| #221 | violet | ? | Per-channel adaptive loss reweighting — needs check-in |
| #214 | gilbert | `2rnm99yl` | k-NN local attention — merge conflict, superseded by #278 |

### Wave 4 (launched 2026-04-30) — wsy/wsz binding-constraint attack

| PR | Student | Hypothesis | Tier |
|----|---------|------------|------|
| #253 | askeladd | FourierEmbed vs ContinuousSincosEmbed standalone A/B | Embedding family disambiguation |
| #254 | chihiro | Raw rel-L2 auxiliary loss sweep w in {0.05, 0.1} | Loss aux term |
| #255 | edward | Fixed per-channel wsy/wsz loss multipliers | Simplest loss rebalance |
| #256 | frieren | Mirror-symmetry TTA for wsy reduction | Inference-time / free gain |
| #257 | haku | High-shear curriculum oversampling with linear anneal | Data sampling reweight |
| #258 | kohaku | Squared rel-L2 aux loss on wall-shear (focal-loss-equivalent) | Loss formulation |
| #259 | senku | grad-clip-norm sweep {0.5, 2.0} on baseline | Optimization stability |
| #260 | thorfinn | model-slices sweep {64, 128, 192} on baseline | Architecture scaling |

### Recently merged / closed

- **PR #176 chihiro FourierEmbed lr sweep** — MERGED (2026-05-02). lr=3e-4 confirmed optimal among {1e-4, 3e-4, 5e-4} on bengio. FourierEmbed standalone validation vs ContinuousSincosEmbed remains a Wave 4 target via PR #253.
- **PR #218 frieren TangentFrameHead** — closed at ep11=13.195%; clean negative result; reassigned to PR #256.
- **PR #75 fern** Trial A `pxty4knv` finished ep50=9.0433% (non-competitive); Trial B `uz4em31o` is the productive replacement.

## Critical Findings

### val/test gap on vol_p is ~2.5× (tanjiro SW2 evidence)
- val=4.17% → test=12.90%
- Surface-loss reweighting moves error around between train channels but does not reduce test error
- Implication: regularization or test-time generalization of volume head is the gap, not architecture or loss weight

### wsy/wsz binding constraint (Wave 4/5 entire focus)
- Best wsy = 9.10% (alphonse val) / 10.895% (tanjiro test) vs target 3.65%. Gap 2.5–3× on test.
- Best wsz = 10.87% (alphonse val) / 11.664% (tanjiro test) vs target 3.63%. Gap 3.2× on test.
- TangentFrameHead failed (PR #218) — pure inductive bias did not work
- Wave 4 attacks via 8 orthogonal levers; Wave 5 adds depth scaling (5L) and SWA/mirror-aug

### Depth scaling + Fourier PE is the round-2 winning ingredient stack
- alphonse `vu4jsiic` (5L/256d, Fourier PE, T_max=50) at ep16 = 7.667% — already below baseline trajectory
- nezuko `ud5iddlc` (5L/384d, Fourier PE, T_max=60) at ep13 = 8.410% — same recipe with 50% wider hidden
- Both confirm 5L/Fourier PE/longer T_max is robustly better than 4L/256d + ContinuousSincos baseline at matched epoch
- vol_p trajectories on these runs are <5% (alphonse) / 6.0% (nezuko), already at AB-UPT target territory

### Universal ep31 valley pattern
All experiments show val abupt minimum at ~step 552K (~ep31) regardless of T_max. T_max=50 schedules align cosine knee with this valley. Runs with T_max=50/60 may extract more after ep31 but the primary descent is always near ep31.

### lr=5e-4 unlocks vol_p capacity at 60k pts (fern Trial B)
fern Trial B at ep18 hit vol_p=6.07% — at AB-UPT target — while still 12 epochs from cosine knee. Confirms 60k pts + Fourier PE + lr=5e-4 is a valid recipe ingredient. Need to test on test set.

## Potential Next Research Directions (Wave 6 prep)

**If alphonse/nezuko Wave 5 leaders land <7.0% val with healthy wsy/wsz**:
- Run test_primary eval from best val ckpt — the val/test gap on vol_p must be confirmed before submission
- Stack winners: best Wave-4 wsy/wsz lever + 5L/256d-or-384d + Fourier PE + T_max=50/60 + EMA off
- Test-set re-evaluation campaign on all val-winners

**If wsy/wsz remain stuck >9% on test even after Wave 4/5**:
- Architecture pivot: spectral graph convolution branch parallel to Transolver attention
- Boundary-layer-aware attention with explicit y+ distance feature
- Graph neural network on surface mesh (explicit topology vs point cloud)
- Boundary-layer physics loss (eddy-viscosity-aware, log-law inspired)
- Stronger regularization specific to volume decoder (target 2.5× val/test gap)
- TangentFrameHead followups: PCA-of-kNN-normals basis, soft loss term, warm start from Cartesian

**Already merged / validated ingredients to compose**:
- FourierEmbed (PR #176, lr=3e-4 optimal) — but standalone validation vs ContinuousSincos still pending PR #253
- 60k points (emma confirmed lr=5e-4 unlocks vol_p)
- T_max=50/60 cosine schedule
- EMA off (alphonse `vu4jsiic` leading without EMA)
- 5L depth (both Wave 5 leaders use 5L)

## Upcoming Gates and Checkpoints

| Time (approx) | Event |
|---|---|
| Next ~hours | norman `pnhbrqtw` ep10/ep15 gate |
| Next ~hours | fern `ep4dl3uw` ep5 kill gate <10.5% (borderline) |
| Next ~hours | gilbert `0kwzszub` step-35k kill gate <20% (will pass) |
| Next ~6h | nezuko `ud5iddlc` ep15 gate (already cleared at ep13) |
| Next ~8h | alphonse `vu4jsiic` ep20 gate (already cleared at ep16) |
| Next ~24h | alphonse `vu4jsiic` ep30 (projected <7.2%; possible new baseline) |
| Next ~24h | nezuko `ud5iddlc` ep30 |
| Wave 4 PRs #253-260 | first ep5/ep10 reports rolling in |

## Plateau Protocol Status

We are NOT on a plateau. Two strong fronts:
1. Wave 5 leaders (alphonse `vu4jsiic`, nezuko `ud5iddlc`) — 5L + Fourier PE + longer T_max stack is delivering improvements over baseline trajectory
2. Wave 4 wsy/wsz attack waves still in flight on the binding constraint
3. norman NF sweep should disambiguate the optimal Fourier frequency band

If alphonse `vu4jsiic` lands <7.0% terminal val with confirmed test improvement, that becomes the new baseline and Wave 4 wsy/wsz winners stack on top.

## Discipline Note

- alphonse `vu4jsiic` is healthy and beating gates; no further escalations needed.
- PRs needing rebase post-#176 merge: #79 emma. Others (#214 gilbert, #179 nezuko) appear to have rebased or moved on.

## Research Log Pointers

- All experiments: `/research/EXPERIMENTS_LOG.md`
- Current baseline: `/BASELINE.md` — alphonse Wave 1 val=7.209% (run `m9775k1v`), ContinuousSincosEmbed
- Research ideas: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
