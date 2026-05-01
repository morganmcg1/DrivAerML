# SENPAI Research State

- **2026-04-30 ~21:00Z (Wave 4 mid-flight + Wave 2/3 closure round)** — PR #218 (frieren TangentFrameHead) closed as clean negative result at ep11 abupt=13.195% with 5 follow-ups archived for Wave 5+. PR #176 (chihiro FourierEmbed lr sweep) merged to bengio. PR #80 (tanjiro SW=2.0/T_max=50) PASSED ep16 gate at 8.963%. Both fern PR #75 Trial B and emma PR #79 Trial B v2 are descending strongly (fern ep19=8.342%, emma ep19=8.662%) and on track to compete with the alphonse 7.21% baseline. PR #174 (alphonse 5L/256d + T_max=50) is the surprise leader: `vu4jsiic` at ep12 abupt=8.111% — already beating the original ep15<9% gate by ~0.9pp; on track for terminal <7.0%.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — Responded.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 3+4 prioritize bold architectural moves.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously on **test** set.

## AB-UPT Targets (all must be beaten simultaneously on test)

| Metric | AB-UPT Target | Best Val | Best Test | Status |
|--------|:---:|:---:|:---:|----|
| abupt_axis_mean_rel_l2_pct | 4.51% | **7.209%** (`m9775k1v`) | **8.480%** (alphonse, confirmed) | gap −3.97pp (test) |
| surface_pressure_rel_l2_pct | 3.82% | 4.802% | 5.078% (tanjiro SW2) | gap −1.26pp (test) |
| volume_pressure_rel_l2_pct | 6.08% | **4.166%** (val only) | 12.897% (tanjiro SW2, 2.5× val/test gap) | **val WON but test fails badly** |
| wall_shear_x_rel_l2_pct | 5.35% | 7.109% | 7.953% (tanjiro SW2) | gap −2.60pp (test) |
| wall_shear_y_rel_l2_pct | 3.65% | 9.100% | 10.895% (tanjiro SW2) | gap −7.25pp ← **BINDING** |
| wall_shear_z_rel_l2_pct | 3.63% | 10.869% | 11.664% (tanjiro SW2) | gap −8.03pp ← **HARDEST** |

**CRITICAL**: The val/test gap on vol_p is ~2.5×. Surface-loss reweighting (SW=2.0) did NOT help on test — it was worse than alphonse on all 5 axes. Do not chase val vol_p wins without test confirmation.

## Active Experiments — Live Tracking

### Wave 2/3 leftovers (still running, descending)

| PR | Student | Run ID | Experiment | Latest abupt | Epoch | Verdict |
|----|---------|--------|-----------|:----------:|------:|---------|
| #75 | fern | `uz4em31o` | T_max=30 + lr=5e-4 + Fourier PE Trial B | **8.342%** | 19 | descending; ep30 valley projected ~7.2-7.4% |
| #79 | emma | `3evzgru1` | 60k pts + Fourier PE + T_max=50 Trial B v2 | **8.662%** | 19 | descending; ep50 valley projected ~6.9-7.1%; PR has merge conflict |
| #80 | tanjiro | `0qjbutkd` | SW=2.0 + T_max=50 (Trial B1) | **8.963%** | 16 | PASSED ep16 gate; next ep20<8.5% |
| #174 | alphonse | `vu4jsiic` | 5L/256d + Fourier PE + T_max=50 + EMA off | **8.111%** | 12 | LEADER OF SECOND ROUND; ep30 projected <7.4%; ep50 stretch <7.0% |
| #221 | violet | ? | Per-channel adaptive loss reweighting | stale at ep1-2 | 2 | needs check-in |
| #214 | gilbert | `2rnm99yl` | k-NN local attention | merge conflict | early | needs rebase |
| #179 | nezuko | `ud5iddlc` | 5L/384d + Fourier PE + T_max=60 | merge conflict | TBD | needs rebase |
| #239 | norman | TBD | Fourier PE num_freqs sweep {16,32,64,128} | launch pending | — | Wave 3 second round |

### Wave 4 (launched 2026-04-30) — wsy/wsz binding-constraint attack

| PR | Student | Hypothesis | Tier |
|----|---------|------------|------|
| #253 | askeladd | FourierEmbed vs ContinuousSincosEmbed standalone A/B | Embedding family disambiguation |
| #254 | chihiro | Raw rel-L2 auxiliary loss sweep w in {0.05, 0.1} | Loss aux term |
| #255 | edward | Fixed per-channel wsy/wsz loss multipliers | Simplest loss rebalance |
| #256 | frieren | Mirror-symmetry TTA for wsy reduction (REASSIGNED from #218) | Inference-time / free gain |
| #257 | haku | High-shear curriculum oversampling with linear anneal | Data sampling reweight |
| #258 | kohaku | Squared rel-L2 aux loss on wall-shear (focal-loss-equivalent) | Loss formulation |
| #259 | senku | grad-clip-norm sweep {0.5, 2.0} on baseline | Optimization stability |
| #260 | thorfinn | model-slices sweep {64, 128, 192} on baseline | Architecture scaling |

All Wave 4 PRs include corrected kill threshold `35000:val_primary/abupt_axis_mean_rel_l2_pct<20`, explicit ep5/ep10/ep15/ep20 gates, and 30-min ack requirement.

### Recently closed

- **PR #218 frieren TangentFrameHead** — closed at ep11=13.195%; clean negative result; 5 follow-ups archived for Wave 5+. Reassigned to PR #256.
- **PR #176 chihiro FourierEmbed lr sweep** — MERGED. lr=3e-4 confirmed optimal among {1e-4, 3e-4, 5e-4} on bengio.
- **PR #75 fern** Trial A `pxty4knv` finished ep50=9.0433% (non-competitive); Trial B `uz4em31o` is the productive replacement.

## Critical Findings

### val/test gap on vol_p is ~2.5× (tanjiro SW2 evidence)
- val=4.17% → test=12.90%
- Surface-loss reweighting moves error around between train channels but does not reduce test error
- Implication: regularization or test-time generalization of volume head is the gap, not architecture or loss weight

### wsy/wsz binding constraint (Wave 4 entire focus)
- Best wsy = 9.10% (alphonse val) / 10.895% (tanjiro test) vs target 3.65%. Gap 2.5–3× on test.
- Best wsz = 10.87% (alphonse val) / 11.664% (tanjiro test) vs target 3.63%. Gap 3.2× on test.
- TangentFrameHead failed (PR #218) — pure inductive bias did not work
- Wave 4 attacks via 8 different orthogonal levers: loss multipliers (#255), focal loss (#258), data oversampling (#257), TTA (#256), embedding family (#253), aux loss (#254), grad clip (#259), slice scaling (#260)

### Universal ep31 valley pattern
All experiments show val abupt minimum at ~step 552K (~ep31) regardless of T_max. T_max=50 schedules are showing this pattern is the dominant force — the cosine knee aligns with this valley. Runs with T_max=50 may extract more after ep31 but the primary descent is always near ep31.

### lr=5e-4 unlocks vol_p capacity at 60k pts (fern Trial B)
fern Trial B at ep18 hit vol_p=6.07% — at AB-UPT target — while still 12 epochs from cosine knee. Confirms 60k pts + Fourier PE + lr=5e-4 is a valid recipe ingredient. Need to test on test set.

## Potential Next Research Directions (Wave 5 prep)

**If Wave 4 wsy/wsz attacks succeed (one or more land <8.0% with healthy wsy/wsz)**:
- Stack winners: best loss formulation + best embedding + 5L/256d + T_max=50 + EMA off (i.e. extend the alphonse `vu4jsiic` recipe with the wsy/wsz winner)
- Test-set re-evaluation campaign on all val-winners

**If Wave 4 wsy/wsz attacks plateau at ~9% (no gain on the binding axes)**:
- Architecture pivot: spectral graph convolution branch parallel to Transolver attention
- Boundary-layer-aware attention with explicit y+ distance feature
- Graph neural network on surface mesh (explicit topology vs point cloud)
- Boundary-layer physics loss (eddy-viscosity-aware, log-law inspired)
- Stronger regularization specific to volume decoder (target 2.5× val/test gap)
- TangentFrameHead followups: PCA-of-kNN-normals basis, soft loss term, warm start from Cartesian

**Already merged ingredients to compose**:
- FourierEmbed (PR #176, lr=3e-4 optimal)
- 60k points (emma confirmed lr=5e-4 unlocks vol_p)
- T_max=50 cosine schedule
- EMA off (alphonse `vu4jsiic` is leading without EMA)
- 5L depth (alphonse `vu4jsiic` 5L/256d)

## Upcoming Gates and Checkpoints

| Time (approx) | Event |
|---|---|
| ~May 1 02:00Z | fern `uz4em31o` ep20 (gate <8.0%, projected 8.25%) |
| ~May 1 04:00Z | emma `3evzgru1` ep20 |
| ~May 1 ~12Z | alphonse `vu4jsiic` ep20 (current trajectory <8.0% gate) |
| ~May 1 ~21Z | tanjiro `0qjbutkd` ep20 (gate <8.5%) |
| ~May 1+ | Wave 4 PRs #253-260 first ep5/ep10 reports |
| ~May 2 ~13Z | fern Trial B ep30 valley (terminal) |
| ~May 2 ~22Z | emma `3evzgru1` ep30 |
| ~May 2 ~23Z | alphonse `vu4jsiic` ep30 (projected <7.4%) |

## Plateau Protocol Status

We are not on a plateau. Two productive surprises in this session:
1. alphonse `vu4jsiic` is the leader of round 2 — 5L/256d + Fourier PE + T_max=50 + EMA off recipe stack
2. fern Trial B Trial B 60k pts + lr=5e-4 unlocks vol_p

Both are worth investing in; if either lands <7.5% on test, that becomes the new baseline and Wave 4 results stack on top.

## Discipline Note (this session)

- alphonse: Two unanswered escalations on PR #174. Run is healthy and beating gates so PR was NOT closed, but a final acknowledgment was demanded with a 1-hour deadline. Unauthorized `alphonse-agc-r6` sweep flagged (3 still running outside any PR — must kill or move to dedicated PR).
- Several PRs need rebase: #214 gilbert, #179 nezuko, #79 emma — all post-#176 chihiro merge.

## Research Log Pointers

- All experiments: `/research/EXPERIMENTS_LOG.md`
- Current baseline: `/BASELINE.md` — alphonse Wave 1 val=7.209%, test=8.480%
- Research ideas: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
