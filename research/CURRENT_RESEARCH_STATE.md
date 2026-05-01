# SENPAI Research State

- **2026-05-01 17:40Z (Wave 3 reassignment round)** — After two escalations with zero student response on PRs #215 senku, #216 askeladd, #217 edward, #219 haku, #220 kohaku — all five closed per non-response protocol, branches deleted. Five fresh Wave 3 hypotheses assigned, all targeting the wsy/wsz binding constraint via different levers: PR #234 senku (mirror-symmetry TTA — free wsy gain via y-flip averaging at inference), PR #235 askeladd (4L/512d/8H radford champion port — width frontier untried on bengio), PR #236 edward (fixed wsy×3/wsz×5 channel multipliers — simplest possible wsy/wsz attack after UW #84 and GradNorm #137 both closed), PR #237 haku (squared rel-L2 aux loss — focal-loss-equivalent for hard-sample focusing), PR #238 kohaku (high-shear curriculum oversampling with linear anneal — orthogonal data-axis lever). All five include the corrected kill threshold `35000:val_primary/abupt_axis_mean_rel_l2_pct<20`, explicit ep5/ep10/ep15/ep20 gates, and a 30-minute acknowledgment requirement. 11 active Wave 2/Wave 3 PRs continue; 16 bengio WIP total after reassignment.


- **2026-05-01 ~17:00Z** — Comprehensive PR audit complete. 16 WIP, 0 review-ready, 0 idle. Key update: tanjiro SW=2.0 T_max=30 full test results in (test abupt=9.697% — lost to alphonse 8.480%). First confirmed alphonse test baseline: **8.480%**. val/test gap confirmed at ~2.5× on vol_p (val=5.19% → test=12.90%). Wave 3 PRs #214 (gilbert) and #218 (frieren) launched with results incoming. Stale PRs: #215 (senku), #216 (askeladd), #217 (edward), #219 (haku), #220 (kohaku) — no student responses to advisor check-ins.

## Most Recent Human Researcher Direction

- **Issue #48 (tay/morganmcg1)**: "Hows it going? we making progress?" — Responded.
- **Issue #18 (yi)**: "Ensure you're really pushing hard on new ideas" — Wave 3 prioritizes bold architectural moves.
- Mission: crush DrivAerML AB-UPT public reference metrics across all 6 axis metrics simultaneously on **test** set.

## AB-UPT Targets (all must be beaten simultaneously on test)

| Metric | AB-UPT Target | Best Val | Best Test | Status |
|--------|:---:|:---:|:---:|----|
| abupt_axis_mean_rel_l2_pct | 4.51% | **7.209%** (`m9775k1v`) | **8.480%** (alphonse, confirmed by tanjiro) | gap −3.97pp (test) |
| surface_pressure_rel_l2_pct | 3.82% | 4.802% | 5.078% (tanjiro SW2) | gap −1.26pp (test) |
| volume_pressure_rel_l2_pct | 6.08% | **4.166%** (val only) | 12.897% (tanjiro SW2, 2.5× val/test gap) | **val WON but test fails badly** |
| wall_shear_x_rel_l2_pct | 5.35% | 7.109% | 7.953% (tanjiro SW2) | gap −2.60pp (test) |
| wall_shear_y_rel_l2_pct | 3.65% | 9.100% | 10.895% (tanjiro SW2) | gap −7.25pp ← **BINDING** |
| wall_shear_z_rel_l2_pct | 3.63% | 10.869% | 11.664% (tanjiro SW2) | gap −8.03pp ← **HARDEST** |

**CRITICAL**: The val/test gap on vol_p is ~2.5×. Surface-loss reweighting (SW=2.0) did NOT help on test — it was worse than alphonse on all 5 axes. Do not chase val vol_p wins without test confirmation.

**Alphonse test baseline confirmed**: abupt=8.480% (5 axis mean, tanjiro PR #80 reported on 2026-05-01). Previously only val=7.209% was known.

## Baseline Correction (2026-05-01, frieren PR #218 audit)

- alphonse Wave 1 winner `m9775k1v` used **`ContinuousSincosEmbed`** NOT FourierEmbed
- PR #74 was a squash merge of assignment commit only — no model code landed
- FourierEmbed added to bengio later by chihiro PR #176 (not yet merged)
- All students using `--fourier-pe` are cherry-picking askeladd commit `d97c19d` from PR #175

## Universal ep31 Peak Pattern

All experiments show val abupt minimum at ~step 552K (~ep31) regardless of T_max. This is a dataset/architecture property. Experiments with T_max=50+ may benefit from continued cosine decay but the primary valley is always near ep31.

## Active Experiments — Live Tracking

### Legacy Wave 2 (still running)

| PR | Student | Run ID | Experiment | Best abupt | Epoch | Gate | ETA |
|----|---------|--------|-----------|:----------:|-------|------|-----|
| #75 | fern | `uz4em31o` | lr=5e-4 Trial B | ~9.40% | ep9 | ep15 <9% | ep31 ~10:30Z May 2 |
| #79 | emma | `3evzgru1` | 60k pts + Fourier PE + T_max=50 Trial B v2 | 9.214% | ep12 | ep15 <9% (projected ~8.82% PASS) | ep31 ~TBD |
| #80 | tanjiro | `0qjbutkd` | SW=2.0 + T_max=50 (Trial B1) | 9.562% | ep10 | ep15 <9%, ep25 <8%, ep50 <7% | ep50 ~May 2 10:30Z |
| #174 | alphonse | `vu4jsiic` | 5L/256d + T_max=50 Trial B v2 | 9.917% | ep10 | ep15 <9% | TBD |
| #176 | chihiro | `ld3ff1gs` | lr=5e-4 Trial B (FourierEmbed) | ~11% | ep5+ | ep15 <9% | TBD |
| #179 | nezuko | `ud5iddlc` | 5L/384d + Fourier PE + T_max=60 | 10.51% | ep5 | ep5 gate PASSED | ep31 ~May 2 11:00Z |
| #180 | norman | `1rieq278` | raw rel-L2 aux loss w=0.1 Trial A | 10.85% | ep5 | ep10 <9% (may fail) | TBD |
| #181 | thorfinn | `scefipy4` | no-EMA + b1=0.95 + T_max=50 | ? | ep1+ | ep10-15 report pending | TBD |

**Tanjiro Trial B matrix auto-launch**: B2 (SW=3.0/T_max=30) and B3 (SW=3.0/T_max=50) queued via PID 188667 script, fire after B1 ep31.

**Fern Trial C**: `auto_kc_trialC.sh` queued to fire after Trial B ep31 (~May 2 01:30Z). Trial C = T_max=50 + lr=5e-4.

### Wave 3 (launched 2026-05-01)

| PR | Student | Run ID | Experiment | Last Known | Kill Threshold Fixed? | Notes |
|----|---------|--------|-----------|:----------:|-----------------------|-------|
| #214 | gilbert | `2rnm99yl` | k-NN local attention (PointTransformer-style rel PE, zero-init out_proj) | launched 16:42Z | YES → `35000:<20` | k=16, chunk=4096, ~8.4 it/s; ~21.5h ETA for ep1; no ep val yet |
| #215 | senku | ? | SWA last-5-epoch averaging | no response | NO | Stale — only advisor check-in |
| #216 | askeladd | ? | Per-axis EMA variance autoweighting | no response | NO | Stale — only advisor check-in |
| #217 | edward | ? | Lion optimizer sweep (lr=1e-4, 3e-4) | no response | NO | Stale — only advisor check-in |
| #218 | frieren | ? (no ID posted) | TangentFrameHead w/ Frisvad-Duff basis (τ = α·e_t1 + β·e_t2) | launched, no W&B ID in thread | YES → `35000:<20` | Physically motivated; `max|τ·n|<1e-6` by construction; stats buffered before torch.compile |
| #219 | haku | ? | 5L depth + Fourier PE + GradNorm α=1.5 stack | no response | NO | Stale — depends on PR #176 landing |
| #220 | kohaku | ? | Asinh surf pressure + 96k pts | no response | NO | Stale — only advisor check-in |
| #221 | violet | ? | Adaptive loss reweighting (gap-ratio softmax, τ=1.0, weights every 5 ep) | ep1-2 launched | YES | Run A (ContinuousSincosEmbed) — no recent update |

**Dead RANS experiments** (nezuko): `pe2ryffk` (λ=0.1) crashed step 12K; `8u7jc8kt` (control) crashed step 13K. Direction closed.

**Symmetry augmentation** (nezuko Wave 2): symm-p50=16.564%, symm-p100=54.686%. Direction closed — breaks coordinate encoding.

## Stale PR Watch List (need follow-up)

PRs with only advisor check-in, no student response — may need escalation or reassignment:
- **#215 (senku)**: SWA
- **#216 (askeladd)**: Per-axis EMA variance autoweighting
- **#217 (edward)**: Lion optimizer
- **#219 (haku)**: GradNorm stack
- **#220 (kohaku)**: Asinh surf pressure

All share the dead kill threshold bug (`3000:val_primary/abupt_axis_mean_rel_l2_pct<=25`). Fix communicated in advisor check-in comments: use `35000:val_primary/abupt_axis_mean_rel_l2_pct<20`.

## Critical Research Finding: Surface-Loss Reweighting Does Not Help on Test

tanjiro PR #80 provides the definitive result:
- SW=2.0 T_max=30 (`846uciam`): test abupt=9.697% vs alphonse test=8.480% — **lost on all 5 axes**
- The "vol_p beats AB-UPT" result (val=4.17%) is a **val artifact** — test vol_p=12.897% (2.5× degradation)
- Surface loss reweighting moves error around between train channels but does not reduce test error
- SW=2.0 T_max=50 (B1 `0qjbutkd`) currently testing whether longer schedule recovers SW benefits

## Critical Gap — wsy/wsz Binding Constraint

Best wsy = 9.10% (alphonse val) / 10.895% (tanjiro test) vs target 3.65%. Gap is 2.5–3× on test.  
Best wsz = 10.87% (alphonse val) / 11.664% (tanjiro test) vs target 3.63%. Gap is 3.2× on test.

Wave 3 bets targeting this:
- **#218 frieren** (TangentFrameHead): theoretically motivated — shear lives in tangent plane, predict α/β scalars → reconstruct τ. Strongest inductive bias bet.
- **#214 gilbert** (k-NN local attention): local surface geometry may explain shear underperformance
- **#216 askeladd** (EMA variance autoweighting): upweights harder axes dynamically
- **#221 violet** (gap-ratio softmax): explicitly targets axes furthest from AB-UPT target

Empirical signal: fern Trial B wsy delta = −2.2pp at ep5 vs Trial A — strongest wsy/wsz improvement yet from lr=5e-4.

## Potential Next Research Directions (Wave 4)

**Bold architectural moves**:
- SO(3)-equivariant representations (Wave 3 PR #218 testing first flavor)
- Spectral-graph convolution as parallel branch alongside Transolver attention
- Latent diffusion prior for surface field reconstruction
- Boundary-layer-aware attention with explicit `y+` distance feature
- Graph neural network on surface mesh (explicit topology vs point cloud)

**Empirical compounders** (ready once Wave 3 data returns):
- Stack all winning ingredients: 5L depth + Fourier PE + GradNorm α=1.5 + T_max=50 + SWA
- Asinh on volume fields too (not just surface pressure)
- Per-axis loss weights from senku metric-aware coefficients (transfer optimal weights)
- Trial C for fern: lr=5e-4 + T_max=50 if Trial B lands 7.5–8.5%

**Test-focused strategy**:
- The val/test vol_p gap (2.5×) is the biggest blocker. Hypothesis: overfitting to training distribution on volume pressure. Try stronger regularization (higher dropout, weight decay) specifically for volume decoder.
- Consider test_primary eval of all completed val-winners before claiming any axis beat.

**Plateau protocol**: 5+ consecutive experiments with no test improvement → escalate to architecture-level changes. We are at ~2 rounds of improvements post-Wave-1 with no test beat. Wave 3 results are the next decision point.

## Upcoming Gates and Checkpoints

| Time (approx) | Event |
|---|---|
| ~May 1 18:00Z | Emma `3evzgru1` ep15 (projected 8.82%, PASS) |
| ~May 1 19:00Z | Chihiro `ld3ff1gs` ep15 gate (<9%) |
| ~May 1 20:00Z | Gilbert `2rnm99yl` ep1 val (first wsy/wsz vs baseline) |
| ~May 2 01:30Z | Fern Trial B ep31 → auto-fire Trial C |
| ~May 2 10:30Z | Tanjiro B1 `0qjbutkd` ep50 → auto-fire B2 (SW=3.0/T_max=30) |
| ~May 2 11:00Z | Nezuko `ud5iddlc` ep31 (5L/384d valley) |
| ~May 2 11:00Z | Alphonse `vu4jsiic` ep31 (5L/256d + T_max=50) |
| ~May 2 TBD | Frieren first ep1-5 results (TangentFrameHead) |

## Research Log Pointers

- All experiments: `/research/EXPERIMENTS_LOG.md`
- Current baseline: `/BASELINE.md` — alphonse Wave 1 val=7.209%, test=8.480%
- Research ideas: `/research/RESEARCH_IDEAS_2026-04-30_15:34.md`
- Wave 3 check-ins: posted 2026-05-01 on PRs #214-221
