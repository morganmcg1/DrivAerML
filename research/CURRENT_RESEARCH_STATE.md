# SENPAI Research State

- **Date:** 2026-05-15 (latest invocation: 2026-05-15 ~12:45 UTC)
- **Branch:** tay
- **W&B project:** wandb-applied-ai-team/senpai-v1-drivaerml-ddp8

## Latest invocation actions (2026-05-15 ~15:10Z) — τ_z structural finding SEVENFOLD confirmed (alphonse EP4), tanjiro #1124 leading fleet

### Verified fleet metrics (2026-05-15 ~15:05Z, GraphQL + W&B parallel pulls)

| Rank | PR | Student | Mechanism | W&B run | EP | val_abupt | val_WSS | vol_p | SP | τz/τx | best_ckpt |
|------|----|---------|-----------|---------|----|-----------|---------|-------|----|-------|-----------|
| **1** | #1124 | tanjiro | EMA decay 0.9995 | `mw6d04kc` | 6.25 | **6.2499%** | **7.058%** | 3.706% | **4.119%** | 1.555 | ✅ updated every gate |
| 2 | #1128 | thorfinn | τ_z weight 3.0 | `uwqybod5` | 8 | 6.307% | 7.130% | 3.716% | 4.184% | 1.539 | asymptote |
| 3 | #1116 | edward | per-channel heads | `3ufrbxl6` | 9 | 6.340% | 7.154% | 3.805% | 4.163% | 1.551 | slope shallowing |
| 4 | #1126 | fern | surface_out d=4 | `gr9ht3h5` | 9 | 6.360% | 7.193% | (sync lag) | (sync lag) | 1.543 | new best at EP9 |
| 5 | #1125 | nezuko | spatial-prior α=10 | `rp1op3z6` | 8 | 6.470% | 7.248% | 3.727% | 4.267% | 1.548 | EP8 PASS |
| 6 | #1127 | askeladd | surface_loss warmup | `ag1dnelx` | ~8 | 6.485% | 7.323% | 3.824% | 4.266% | 1.559 | mid-curr |
| 7 | #1122 | alphonse | SDF FAR-field α=2.0 | `vvv84p32` | **4** | 6.886% | 7.668% | 4.602% | 4.431% | 1.526 | EP4 best |
| 8 | #1133 | frieren | per-axis mag decomp | `5l9i6fjn` | 1.32 | 31.55% (EP1) | 35.69% | 16.89% | 24.17% | 1.388 | EP1 healthy |

### CRITICAL: τ_z structural finding SEVENFOLD CONFIRMED — architectural pivot signal

Eight active mechanisms tested:

| Mechanism | EP | τz/τx | Verdict |
|-----------|----|-------|---------|
| EMA 0.9995 (tanjiro) | 6.25 | 1.555 | in band |
| τ_z weight 3.0 (thorfinn) | 8 | 1.539 (asymptoted) | in band |
| per-channel heads (edward) | 9 | 1.551 | in band |
| surface_out d=4 (fern) | 9 | 1.543 | in band |
| spatial-prior α=10 (nezuko) | 8 | 1.548 | in band |
| surface_loss warmup (askeladd) | 8 | 1.559 | in band |
| **SDF FAR-field α=2.0 (alphonse)** | **4** | **1.526** | **in band — 7th confirmation** |
| mag-only decomp (frieren #1121, closed) | 12 | 1.570 | in band — terminal |
| per-axis mag decomp (frieren #1133) | 1.32 | 1.388 → TBD | **8th and final loss/data-side test** |

**The τ_z/τ_x ratio converges to 1.50–1.57 across:**
- loss weighting (×3 vs ×1)
- sampling bias (spatial-prior + SDF FAR-field)
- output capacity (per-channel decoupled heads)
- decoder depth (surface_out d=2 → d=4)
- temporal averaging (EMA 0.999 vs 0.9995)
- magnitude calibration (frieren #1121 mag-only aux head)
- input weighting curriculum (askeladd surface_loss warmup)

**Conclusion**: τ_z bottleneck is **NOT** addressable by ANY data-side or loss-side intervention. The mechanism is backbone-representation-side. Once frieren #1133 (per-axis mag decomp, the 8th and final loss-side test) confirms or breaks this pattern, we commit to Wave 30 architectural experiments:

**Wave 30 architectural roster (proposed)**:
1. **Coordinate-system change**: 3D Cartesian (x,y,z) → cylindrical (r,θ,z) or vehicle-body frame (longitudinal/lateral/vertical). τ_z is "vertical wall-shear" — if the backbone is encoding all three axes in shared Cartesian features, a coordinate system aligned with the dominant flow direction would give τ_z its own preferred basis direction.
2. **Per-axis attention heads in the backbone**: split Transolver attention layers into per-axis sub-tensors after a specified layer, so τ_z gets dedicated attention rather than competing with τ_x/τ_y for shared head capacity.
3. **Dedicated τ_z encoder branch (Y-architecture)**: parallel branch from a mid-network feature layer that processes only τ_z magnitude prediction, with separate normalization and MLP depth.
4. **Mixture-of-Experts on the surface head**: K experts, each with output specialization on one axis or feature.

### Tanjiro #1124 = leading single-model candidate on no-SDF tay

- EP6.25 val_abupt=6.2499% (-0.076pp from EP5.75) with `best_checkpoint/updated=1` at every recent gate
- EP13 projection (conservative slope-shallowing): **val_abupt 5.88–6.03%**
- Would beat 6.126% baseline by 0.10–0.24pp
- Floor risk at val_SP=4.119% (frieren #1121 closed at val_SP=4.218% → test_SP=3.734% +0.157pp regress)
- **Critical request posted to tanjiro**: report EP12 best-EMA-checkpoint metrics specifically (not EP13 final)

### Alphonse #1122 truncation decision (just posted)

- EP10 truncation confirmed (cumulative 17h57m, ~20min safety margin)
- Test-eval +45min budget extension conditionally granted (single highest-EV SDF experiment)
- Standing instructions: report at EP6 (mid-vol=32k) and EP9 (end-vol=49k); interrupt me only if τz/τx <1.45

### Actions this invocation
- Posted EP4 truncation decision to alphonse #1122 (with SDF FAR-field τ_z confirmation)
- Pulled tanjiro #1124 + frieren #1133 W&B states
- Posted leadership-ack + EP12 best-EMA harvest instruction to tanjiro #1124
- Posted launch confirmation + mag_z/mag_xy diagnostic ask to frieren #1133
- Survey via GraphQL (REST API rate-limited until ~15:19Z)

### Next-highest-EV gates

| ETA | Event | Action |
|-----|-------|--------|
| ~15:08Z | edward #1116 EP10 val (first vol=65k epoch) | Watch slope reacceleration |
| ~15:35Z | thorfinn #1128 EP9-10 (vol curriculum bump) | Watch τ_z reduction at higher vol |
| ~16:00Z | tanjiro #1124 EP7 (advisor request) | val_abupt + val_SP + per-axis report |
| ~16:25Z | alphonse #1122 EP6 readout | Slope continuation check |
| ~17:30Z | frieren #1133 EP3 gate | mag_z_loss and mag_xy_loss separation diagnostic |
| ~18:00Z | thorfinn #1128 EP13 terminal | First merge-eligible candidate |
| ~19:00Z | tanjiro #1124 EP12 best-EMA harvest | **Highest merge-priority gate** |
| ~19:30Z | fern #1126 EP13 + test eval | Decoder-depth verdict |
| ~20:45Z | alphonse #1122 EP10 + test eval | SDF FAR-field verdict + budget-extension request |

---

## Prior invocation actions (2026-05-15 ~12:55Z) — Wave 29 mid-late EP fleet status, edward #1116 terminal imminent

### Verified fleet metrics from W&B (2026-05-15 ~12:50Z)

| Rank | PR | Student | Mechanism | W&B run | EP | val_abupt | val_WSS | vol_p | SP | τz/τx |
|------|----|---------|-----------|---------|----|-----------|---------|-------|----|-------|
| **1** | #1124 | tanjiro | EMA decay 0.9995 | `mw6d04kc` | ~6.2 | **6.228%** | **7.030%** | 3.704% | **4.110%** | 1.554 |
| 2 | #1128 | thorfinn | τ_z loss weight 3.0 | `uwqybod5` | ~5.45 | 6.307% | 7.130% | 3.716% | 4.184% | 1.539 |
| 3 | #1116 | edward | per-channel WSS heads | `3ufrbxl6` | **~12.6/13** | 6.340% | 7.150% | 3.810% | 4.160% | 1.551 |
| 4 | #1126 | fern | surface_out depth=4 | `gr9ht3h5` | ~9.22 | 6.360% | 7.193% | 3.762% | 4.218% | 1.543 |
| 5 | #1125 | nezuko | spatial-prior α=10 | `rp1op3z6` | ? | 6.404% | 7.248% | 3.727% | 4.267% | 1.548 |
| 6 | #1127 | askeladd | surface_loss warmup | `ag1dnelx` | ? | 6.485% | 7.323% | 3.824% | 4.266% | 1.559 |
| 7 | #1122 | alphonse | SDF FAR-field α=2.0 | `vvv84p32` | EP3 | 7.168% | 8.002% | 4.665% | 4.684% | 1.515 |
| 8 | #1133 | frieren | per-axis mag decomp | TBD | EP0 | (just launched) | — | — | — | — |

### Critical observations

**1. Edward #1116 terminal imminent** (~13:30–13:50Z): at EP12.6/13, walltime 675.7 min = 11.26h. Final EP13 + test eval expected within 30-50 min. Edward's val_abupt=6.34% is unlikely to beat the 6.126% baseline at terminal — slope has flattened. **Most likely outcome: close (no improvement) or send back for variation.**

**2. Tanjiro #1124 leads the fleet at val_abupt 6.228% at EP~6.2.** This is the slowest-EMA experiment in the fleet (EMA 0.9995 vs default 0.999, half-life 1386 vs 693 steps). Comparison to frieren #1121 terminal trajectory: frieren EP6 was 6.397%, terminal best-EMA EP12=6.073%. If tanjiro tracks similarly, terminal projection lands ~5.95–6.05% val_abupt — **would beat 6.126% baseline by 0.07–0.18pp**. Highest current single-model contender on no-SDF tay.

**3. τ_z/τ_x ratio confirmed SEVENFOLD-EIGHTFOLD structural** (now including: tanjiro EMA 1.554, thorfinn τ_z×3 1.539, edward per-channel 1.551, fern depth=4 1.543, nezuko spatial-prior 1.548, askeladd warmup 1.559, alphonse SDF FAR-field 1.515 at early EP3, frieren #1121 closed 1.570). Ratio converges to ~1.50–1.57 across ALL mechanisms. **τ_z bottleneck is NOT addressable by loss weighting, sampling, output capacity, EMA, magnitude calibration, or input-bias re-weighting (SDF FAR-field is the latest test).** Architectural pivot required if alphonse and frieren #1133 also confirm.

**4. Alphonse #1122 SDF FAR-field α=2.0 EP3 MARGINAL** at 7.168% val_abupt — already responded with budget-management guidance (prefer EP12 truncate over skip-eval) and τ_z/τ_x ratio monitoring ask for EP4. EP4 readout ~14:30Z is the cleanest mechanism test (vol curriculum bump from 16k→32k).

**5. Floor analysis (val→test mapping):** PR #972 baseline floors are test_vol_p ≤3.643%, test_SP ≤3.577%. Val→test compression typically ~0.10pp (frieren #1121 was val_vol_p=3.517% → test=3.545%, val_SP=4.218% → test=3.734%). Current fleet:
   - tanjiro val_SP=4.110% → test projection ~3.63% (close to floor)
   - All other students val_SP ≥4.16% → test projection ≥3.7% (above floor)
   - **Multiple runs at risk of test_SP floor regression at terminal.**

### Action this invocation
- Verified fleet state via parallel W&B pulls (tanjiro/nezuko/askeladd in one batch; thorfinn/fern/edward in three parallel agents).
- Updated state doc with current EP positions and metrics.
- Responded to alphonse #1122 EP3 MARGINAL with EP4 monitoring ask.
- Issue #1056 status posted at 12:53Z (via check-human-issues).
- Schedule ~35min wakeup for edward #1116 terminal.

### Next-highest-EV events (ordered by ETA)

| ETA | Event | Action |
|-----|-------|--------|
| ~13:30–13:50Z | **edward #1116 terminal** (EP13 + test eval) | Review terminal; merge if test_WSS<6.85% AND floors held, else close/back |
| ~14:30Z | **alphonse #1122 EP4 readout** (vol curriculum bump 16k→32k) | Monitor τz/τx; <1.49 = SDF FAR-field breaks structural pattern |
| ~14:30–15:00Z | thorfinn #1128 EP6/7 gate | Check slope; tanjiro EP6.5 reference |
| ~15:30Z | fern #1126 EP10/11 | Late-EP slope check |
| ~16:30–17:30Z | alphonse #1122 EP5/6 | Curriculum-shift convergence check |
| ~17:00–18:00Z | tanjiro #1124 EP10 | Best-EMA crossover prediction |
| ~18:00–20:00Z | tanjiro #1124 / thorfinn #1128 terminal | First merge-eligible single-model candidates |

---

## Prior invocation actions (2026-05-15 ~12:45Z) — Frieren #1121 closed terminal, reassigned to #1133 per-axis-mag decomp

### Actions this invocation

- **Closed PR #1121 (frieren mag-only decomp + 18h)** at terminal EP13.
  - Test metrics: test_WSS=**6.859%** (+0.132pp vs PR #972 SOTA, but **−0.137pp vs no-decomp #1078**), test_vol_p=3.545% PASS, **test_SP=3.734% (+0.157pp FLOOR REGRESS)** ❌, test_abupt=5.939%.
  - Val: val_abupt=**6.073%** (−0.053pp vs PR #972 6.126% baseline) — **first single-model val_abupt improvement on no-SDF tay** since the corrected split landed.
  - Methodology success: mag head perfectly calibrated (ratio 0.9993, mag_loss 0.0011, 4.4× tighter than #1112 EP3). λ_dir=0 confirmed throughout.
  - **Why close**: test_SP floor regression is a merge blocker; single-model winners must hold both floors. Methodology preserved as strong building block for stacking (most natural pairing: SDF FAR-field α=2.0 ←→ alphonse #1122).
  - **τ_z structural finding strengthened to SIXFOLD confirmation**: this is the 6th active mechanism (loss weight, sampling, output capacity, EMA, mag-only decomp, per-channel heads) converging to τz/τx ratio ~1.50–1.57 by EP5-10. EP9→EP10 τ_z reversal (+0.020pp) is the cleanest single-run instance. τ_z bottleneck is **NOT** addressable by these levers.

- **Assigned PR #1133 (frieren: per-axis WSS magnitude decomp + 18h)** — direct architectural attack on τ_z structural finding.
  - **Hypothesis**: split mag aux head into `surface_mag_z_aux` (predicts |τ_z|) and `surface_mag_xy_aux` (predicts ||τ_xy||₂) as SEPARATE heads. Tests whether mag-only's success was bandwidth-limited (single head must encode all three axes' magnitudes) vs. representational (backbone features can't carry τ_z).
  - **Loss**: `L = L_base + λ_mag_z * MSE(|τ_z_pred|, |τ_z_gt|) + λ_mag_xy * MSE(||τ_xy_pred||₂, ||τ_xy_gt||₂)`
  - **Asymmetric defaults**: λ_mag_z=0.1, λ_mag_xy=0.05 — emphasize τ_z bottleneck.
  - **CLI flags**: `--wss-decomp-method per-axis-mag --wss-decomp-lambda-mag-z 0.1 --wss-decomp-lambda-mag-xy 0.05`
  - **Win signal**: test_τ_z ≤ 8.50% (vs #1121's 8.873%, ≥0.37pp improvement). Reach: test_WSS<6.85% AND test_SP≤3.577% AND test_vol_p≤3.643% AND val_abupt≤6.20% → first single-model merge on tay since SDF stack.
  - **Falsifiability**: test_τ_z ≥ 8.80% would confirm τ_z bottleneck is BACKBONE-side (no aux-head decomp can rescue) and force pivot to coordinate-system or attention-mechanism changes.
  - 18h budget (`SENPAI_TIMEOUT_MINUTES=1100`), DDP 8 GPU, group `frieren-per-axis-mag-decomp`. Branch `frieren/per-axis-wss-mag-decomp-18h`.

### Active fleet (7 students still in WIP from Wave 29 + frieren just reassigned)

| PR | Student | Status |
|----|---------|--------|
| #1116 | edward | active — per-channel WSS output heads (τ_x/τ_y/τ_z), 18h |
| #1122 | alphonse | active — SDF FAR-field α=2.0 corrected mechanism |
| #1124 | tanjiro | active — EMA decay 0.9995, 18h |
| #1125 | nezuko | active — spatial-prior surface sampling α=10, 18h |
| #1126 | fern | active — deeper surface_out MLP (depth 2→4), 18h |
| #1127 | askeladd | active — surface_loss warmup curriculum, 18h |
| #1128 | thorfinn | active — τ_z loss weight 3.0, 18h |
| #1133 | frieren | NEW — per-axis WSS magnitude decomp, 18h |

**Zero idle.** Fleet remains at full 8 active.

### Highest-EV next event

- **alphonse #1122 EP3 gate** (~07:55Z if recipe held pace; verify W&B `vvv84p32` actual EP) — this is the corrected SDF FAR-field α=2.0 mechanism, the only SDF-stacked experiment in flight. Hit signal: ≤6.9% PASS / ≤7.2% MARGINAL. Largest expected uplift in the fleet.
- After alphonse EP3, monitor EP5 gates fanning in for fern/askeladd/edward/thorfinn/nezuko in 06:00–08:30Z window.

---

## Prior invocation actions (2026-05-15 ~06:30Z) — Wave 29 EP gate monitoring, fleet-wide τ_z structural finding confirmed

### Fleet-wide EP gate status (2026-05-15 ~06:30Z)

| PR | Student | W&B run | Current EP | Latest val_abupt | Latest WSS | τz/τx | vol_p | Gate Status |
|----|---------|---------|-----------|---------|---------|-------|-------|-------------|
| #1121 | frieren | `gljtmuvs` | EP8.67 | **6.0782%** | **6.8775%** | 1.570 | **3.527%** | EP8 PASS ✓ — LEADING RUN |
| #1122 | alphonse | `vvv84p32` | EP2.18 | 8.2300% | 9.0683% | 1.541 | 5.479% | EP3 gate pending (~163 min from 06:30Z) |
| #1124 | tanjiro | `mw6d04kc` | EP6.21 | 6.3963% | 7.2069% | 1.547 | 3.831% | EP6 MARGINAL (0.096pp above ≤6.3% PASS); EP7 gate: ≤6.3% PASS |
| #1125 | nezuko | `rp1op3z6` | EP5.19 | 6.7039% | 7.6024% | 1.516 | 3.897% | EP5 PASS (≤7.2%); EP8 gate pending |
| #1126 | fern | `gr9ht3h5` | EP4.88 | 6.6062% | 7.4646% | 1.519 | 3.924% | EP4 MARGINAL; EP5 gate imminent |
| #1127 | askeladd | `ag1dnelx` | EP4.91 | 6.7613% | 7.6589% | 1.526 | 3.966% | EP4 MARGINAL; EP5 gate imminent |
| #1116 | edward | `3ufrbxl6` | EP4.59 | 6.5968% | 7.4533% | 1.537 | 3.925% | EP4 PASS; EP5 gate approaching |
| #1128 | thorfinn | `uwqybod5` | EP4.14 | 6.5675% | 7.4273% | 1.513 | 3.880% | EP4 MARGINAL; EP5 approaching |

### CRITICAL FLEET-WIDE FINDING: τ_z bottleneck is STRUCTURAL

ALL τ_z-targeted interventions have FAILED to suppress τz/τx ratio. Every agent's ratio monotonically rises to ~1.50–1.57 by EP5-8 regardless of approach:
- nezuko α=10: 1.371→1.516 by EP5
- thorfinn τz_weight=3.0: 1.288→1.513 by EP4 (transient EP1 suppression only)
- edward per-channel heads: 1.400→1.537 by EP4
- frieren mag-only: 1.389→1.570 by EP8.5 (stabilizing)
- tanjiro EMA 0.9995: 1.454→1.547 by EP6

**Conclusion**: τ_z bottleneck is NOT addressable by loss weighting, sampling, or output capacity. Requires architectural solution targeting the τ_z representational bottleneck (e.g., coordinate system change, dedicated physics-informed τ_z head with orthogonal basis, or attention mechanism change).

### Gate comments posted this invocation
- **Frieren EP8 PASS** → EP10 gate: val_WSS ≤6.80% PASS / ≤6.85% MARGINAL / >6.85% KILL
- **Tanjiro EP6 MARGINAL** → EP7 gate: val_abupt ≤6.3% PASS / 6.3-6.5% MARGINAL / >6.5% KILL
- **Alphonse EP2 progress** → EP3 gate: ≤6.9% PASS / 6.9-7.2% MARGINAL / >7.2% KILL; vol_p ≤4.5%
- **Fern EP4 MARGINAL** → EP5 gate: ≤6.5% PASS / ≤6.8% MARGINAL / >6.8% KILL
- **Askeladd EP4 MARGINAL** → EP5 gate: ≤6.5% PASS / ≤6.8% MARGINAL / >6.8% KILL
- **Edward EP4 PASS** → EP5 gate: ≤6.5% PASS / ≤7.0% MARGINAL / >7.0% KILL
- **Thorfinn EP4 MARGINAL** → EP6 gate: ≤6.5% PASS / ≤6.8% MARGINAL / >6.8% KILL

---

## Latest invocation actions (2026-05-15 03:48–04:00Z) — Wave 29 full fleet confirmed active, all 8 students running

- **Closed PR #1123 (thorfinn τ_z dedicated subnet)** — zero student activity after 4+ hours, four advisor check-in messages unanswered. Pod confirmed idle (1/1 READY via kubectl). Hypothesis is sound but requires code implementation; reassigned pod to a zero-code-change experiment to eliminate implementation failure mode.
- **Assigned PR #1128 (thorfinn: τ_z loss weight escalation 2.0→3.0)** — pure CLI flag change `--tau-z-loss-weight 3.0`, no model code changes. Directly attacks dominant error axis (test_τ_z ≈ 9.05–10.1% across all no-SDF runs). Pass signal: τ_z/τ_x ratio at EP13 < 1.5 (down from ~1.6–1.7 baseline). Full 18h budget (SENPAI_TIMEOUT_MINUTES=1100). W&B run `uwqybod5` (group `tau-z-loss-weight-3p0`), launched 03:48:42Z.
- **thorfinn #1128 confirmed launched** — student ACK received 03:49:15Z with PID confirmed and `SENPAI_TIMEOUT_MINUTES=1100` set. W&B run ID `uwqybod5`, W&B name `thorfinn/tau-z-loss-weight-3p0-20260515T034842Z`. Resolves escalation from #1123 closure.
- **alphonse #1122 pace corrected** — actual pace at vol=16k is ~131 min/epoch (not 80 min). Root cause: vol=16k → 860 views/case (ceil(14M/16k)) → view_count=max(130,860)=860 → 10,864 iters/rank/epoch × 1.38 it/s = 131 min. Gate schedule revised: EP1 ~05:50Z, EP3 ~07:55Z. Smoke confirmed 5.6× sampled/population weight ratio → correct FAR-field SOTA mechanism.
- **Full Wave 29 fleet all active** (kubectl: all 8 deployments 1/1 READY at 03:52Z). Zero idle.

### Wave 29 fleet — full status and gate schedule (as of 03:52Z, 2026-05-15)

| PR | Student | Hypothesis | W&B Run | EP1 Gate | EP3 Gate | EP13 ETA |
|----|---------|------------|---------|----------|----------|----------|
| #1116 | edward | Per-channel WSS output heads (τ_x/τ_y/τ_z) — 18h convergence (relaunched 03:09Z as `3ufrbxl6`) | `3ufrbxl6` | ~05:10Z | **~08:00Z** | ~14:00Z |
| #1121 | frieren | WSS magnitude-only decomp (λ_dir=0, λ_mag=0.1) — EP3 PASS 6.746% (best in family) | `frieren/mag-only-*` | DONE | **DONE (PASS)** | ~14:30Z |
| #1122 | alphonse | SDF FAR-field α=2.0 (`weight=1+α|sdf|`) — corrected SOTA mechanism port | alphonse run | ~05:50Z | **~07:55Z** | ~16:30Z |
| #1124 | tanjiro | EMA decay 0.9995 — EP1 PASS 31.48%, EP2 in flight | `mw6d04kc` | DONE | **~06:15Z** | ~15:00Z |
| #1125 | nezuko | Spatial-prior surface sampling α=10 — 18h budget | nezuko run | **~05:00Z** | ~06:00Z | ~14:00Z |
| #1126 | fern | Deeper surface_out MLP (depth 2→4, +525k params) — 18h budget | fern run | ~04:30Z | **~06:00Z** | ~14:00Z |
| #1127 | askeladd | Surface_loss warmup curriculum (3-ep ramp 0→full) — 18h budget | `dtgfdsgv` | ~04:30Z | **~06:00Z** | ~14:00Z |
| #1128 | thorfinn | τ_z loss-weight 3.0 (single CLI flag escalation from 2.0) | `uwqybod5` | **~06:00Z** | ~08:30Z | ~15:30Z |

Gate criteria per row:
- **frieren #1121 EP6** (~06:00Z): val_abupt ≤6.5% PASS / ≤6.8% MARGINAL (half-way convergence sanity)
- **Standard no-SDF EP3**: val_abupt ≤7.2% PASS / ≤7.6% MARGINAL / >7.6% KILL
- **alphonse #1122 EP3 (SDF FAR-field)**: val_abupt ≤6.9% PASS / ≤7.2% MARGINAL / >7.2% KILL (tighter — SDF expected uplift)
- Per-axis WSS signal: τ_z/τ_x ratio direction is primary quality signal for all WSS-targeting experiments

## Prior invocation actions (2026-05-15 02:35–02:50Z) — Wave 28.5 closures complete, Wave 29 architectural pivot launched

- **Closed PR #1118 (askeladd OHEM v2)** — definitive negative mechanism: `clip_active`=100.00% across all 4218 EP3 OHEM-active steps → gradient through OHEM term is exactly zero → run is mathematically equivalent to baseline. Test metrics regressed +0.903pp test_WSS vs SOTA at EP3-only (truncated by 270-min cap). **OHEM-on-raw-residuals family terminally exhausted**: dataset's top-K residuals are intrinsically 100–25,000× larger than mean → any safe scalar cap fires 100% → zero learning signal. The `clip_active_pct` diagnostic was the right metric and identified the failure mode within EP3 — should remain in codebase for future loss-clip work.
- **Wave 28.5 loss-engineering pattern: 0-for-3 at convergence** — #1114 learnable WSS (null), #1119 GradNorm short-cycle (refutes prior-rediscovery), #1118 OHEM v2 (zero gradient). Decisive pivot to capacity / data-sampling / architecture routes.
- **Assigned PR #1126 (fern: deeper surface_out MLP depth 2→4 + 18h)** — Wave 29 architectural pivot. Tests whether τ_z magnitude prediction is decoder-depth-limited at the surface head (current 2-layer MLP). Orthogonal to thorfinn #1123 (separate τ_z branch) and edward #1116 (per-channel heads). Parameterizes `surface_out_depth` config; depth=2 default preserves backward compat. Full 13-EP convergence at SENPAI_TIMEOUT_MINUTES=1100.
- **Assigned PR #1127 (askeladd: explicit surface_loss_weight warmup curriculum + 18h)** — directly tests #1114 finding that EP1 wins are implicit-curriculum artifacts. Adds `--surface-loss-weight-warmup-epochs 3` flag that linearly ramps surface_loss_weight from 0 → full over first 3 epochs. Gradient-flow-preserving (scalar multiplier, NOT residual reweight) → avoids OHEM #1118 trap. Predicted payoff: stable volume-conditioned backbone before surface head receives full gradient → better τ_z magnitude convergence at terminal.
- **All 8 students now active**: alphonse #1122 (SDF FAR-field α=2.0), nezuko #1125 (spatial-prior α=10 + 18h), tanjiro #1124 (EMA decay 0.9995 + 18h), thorfinn #1123 (τ_z subnet — CLOSED, replaced by #1128), edward #1116 (per-channel heads, 18h convergence), frieren #1121 (magnitude-only + 18h), fern #1126 (surface_out depth=4 + 18h), askeladd #1127 (surface_loss warmup curriculum + 18h). **Zero idle.**

## Prior invocation actions (2026-05-15 01:41Z) — CRITICAL SDF MECHANISM DIAGNOSTIC

- **PR #1122 alphonse SDF port → CHANGES REQUESTED, corrected plan approved**: alphonse paused the 13ep run at 28min in (EP1 ~25% done) after spotting THREE issues with my original assignment:
  1. **Mechanism inversion**: commit `023f766` I cited as reference impl implements `weight = 1/(1+α·|sdf|)` (NEAR-surface emphasis), but PR #972 body and the actual SOTA run `56bcqp3m` use `weight = 1 + α·|sdf|` (FAR-field emphasis). These are OPPOSITE hypotheses.
  2. **α value mismatch**: SOTA `56bcqp3m` ran α=2.0, not α=4.0. The NEAR-surface alpha sweep on W&B (α=0.25→6.265%, α=0.5→6.290%, α=1.0→6.356%, α=3.0→7.251% over kill gate) shows higher α is monotonically worse for the NEAR-surface inversion.
  3. **IO regression**: alphonse's port used `np.load(path, mmap_mode="r")[rows]` fancy-indexed memmap which runs ~3× slower than PR #972's contiguous load + in-memory slice on this PVC. Smoke EP1 took 114 min vs SOTA reference 41 min.
  4. SOTA confounders captured: `56bcqp3m` also ran batch_size=1, model_layers=6, GradNorm, y_symmetry_aug, epochs=30 — these are NOT part of the corrected single-variable port.
- **Approved corrected plan**: revert IO optimization, switch to FAR-field `weight = 1 + α·|sdf|` α=2.0, keep tay baseline recipe (batch_size=4, model_layers=5, no GradNorm, no y_sym, epochs=13), smoke 2EP then full 13EP. Single-variable change isolates the SDF mechanism; full-recipe SOTA reproduction held for follow-up if FAR-field α=2.0 alone doesn't beat 6.99% ceiling.
- **Adjusted EP3 gate for FAR-field α=2.0**: PASS ≤ 6.9% / MARGINAL ≤ 7.2% / KILL otherwise. Projected EP13 terminal val_abupt ~6.4-6.6%, putting test_WSS in striking range of 6.5-6.7% (likely strongest single-model on tay).

## Methodology lesson for advisor

Always **verify the SOTA reference mechanism from the actual W&B config** before citing it in an assignment, not from a commit body that may be a different formulation. The PR body, the commit text, and the run config can all diverge. Going forward: when citing a SOTA mechanism, pull its W&B config first.

## Prior invocation actions (2026-05-15 01:15–01:30Z)

- **Closed PR #1114** (tanjiro learnable WSS channel weights): terminal SENPAI-RESULT `test_WSS=7.726%, val_abupt=7.066%` at EP3 (budget-truncated). +0.40pp val_abupt over matched 3-EP baseline (mempfubx 7.465%) but driven by EP1 transient drift (weights briefly dropped to ~50% of init, then quadratic-well-regularized back to baseline by EP3 within 3% of init). Mechanism null at convergence. Methodology data preserved.
- **Reassigned tanjiro → PR #1124** (EMA decay 0.9995 + 18h budget): single-flag experiment, slower EMA half-life ≈ 1386 steps vs default 693 steps. Tests whether late-converging τ_z benefits from longer EMA averaging window. Full 13-EP convergence test, comparison to no-SDF tay ceiling 6.99%.
- **Sent PR #1116 back to edward** (per-channel WSS heads, draft state): matched-budget A/B at EP3 truncated showed −0.66pp test_WSS, −0.09pp test_vol_p, −0.23pp test_SP — every metric improved vs single-head baseline `mempfubx`. **First clean positive Wave 28.5 signal.** But test_WSS=7.671% does not beat no-SDF ceiling (6.99%), so requires 18h budget convergence confirmation: if matched-budget delta holds at EP13, test_WSS → 6.33%, would tie/beat ensemble SOTA. Re-running with `SENPAI_TIMEOUT_MINUTES=1100`, no other changes.
- **τ_z/τ_x ratio finding (edward EP3)**: per-channel heads UNIFORMLY uplift capacity, not τ_z-specifically — ratio went 1.44 → 1.52 (wrong direction). Mechanism is "decoupled head capacity", not "τ_z specialization". Implies follow-on work needs deeper/wider τ_z head specifically (overlaps with thorfinn #1123 τ_z subnet).

---

## PRIMARY RESEARCH DIRECTIVE (Issue #1056, 2026-05-12, ongoing)

**Beat Wall Shear Stress SOTA from Transolver-3 (5.85%) while not degrading vol_p or surface pressure.**

- **WSS target:** `test_WSS` < **5.85%**
- **Non-negotiable constraints:** `test_vol_p` ≤ 3.643% AND `test_SP` ≤ 3.577% (PR #972 levels)
- **Baseline for all new single-model runs:** PR #972 SDF-stratified stack

**WSS Gap (post-PR #1102):**
- Single-model best: **6.727%** (PR #972) → need −0.88pp
- Ensemble best (compliant): **6.3263%** (PR #1102 K=8 Caruana) → need **−0.476pp**

Most recent human check-in: 2026-05-14 14:17 UTC — **"NO MORE ENSEMBLES! Its the lazy route to better results, we want genuine breakthroughs, not incremental improvements based on ensembling which we know we can deploy at any point to improve results."** (Issue #1056 comment from morganmcg1). Ensemble experiments are BANNED until explicitly unlocked. Status updates posted at ~12:35 UTC and ~15:00 UTC.

---

## CORRECTED DATASET IN EFFECT (since 2026-05-11)

Issue #1053 (deployed 2026-05-11) fixed a case-split/indexing bug that eliminated an artificial ~3× vol_p OOD gap.

**Corrected dataset path:** `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`
**Split:** val=34 cases/7295 views, test=50 cases/11091 views

All new runs MUST use the corrected dataset path and `--data-root` flag (not `--data-path`).

---

## Current Best Results (Corrected Split)

### **Ensemble SOTA (PR #1102 — K=8 Caruana with-replacement, WSS-optimised)**
- val_abupt = **5.7452%** | test_abupt = **5.5196%**
- val_vol_p = 3.4360% | test_vol_p = 3.5397%  ← satisfies ≤ 3.643%
- val_WSS = 6.5195% | **test_WSS = 6.3263%**  ← TRUE WIN
- val_SP = 3.7234% | test_SP = 3.3529%  ← satisfies ≤ 3.577%
- test_tau_x = 5.6071% | test_tau_y = 6.8397% | **test_tau_z = 8.2585%** (still worst axis)
- W&B: `bq1gaewq` (Arm D greedy), `ems8ekee`, `s7pirpr1`, `qf1lqwz0`
- **Members:** `56bcqp3m`×3, `29nohj67`×2, `a0yoxy85`×2, `ghh0s4ne`×1
- **Effective weights:** {56bcqp3m:0.375, 29nohj67:0.250, a0yoxy85:0.250, ghh0s4ne:0.125}

### Prior Ensemble SOTA (PR #1064 K=3 greedy, superseded by #1102)
- val_abupt = 5.7758% | test_abupt = 5.5199% | test_WSS = 6.3712%

### Single-Model SOTA (PR #972 — SDF-stratified vol importance sampling)
- val_abupt = 6.126% | test_abupt = 5.844%
- val_vol_p = 3.798% | test_vol_p = 3.643%  ← constraint boundary
- test_SP = 3.577%  ← constraint boundary
- test_WSS = 6.727%
- W&B: source=`56bcqp3m`, eval=`zxnhtagj`

### Rank 2 Single-Model (PR #968 — Stochastic vol subsampling)
- val_abupt = 6.278% | test_abupt = 5.986% | test_WSS = 6.825%
- W&B: source=`a0yoxy85`, eval=`qbg9pkmx`

---

## Gate Criteria

### Single-Model EP3 Gates (current tay stack — no SDF importance sampling)
- **PASS:** val_abupt ≤ **7.2%** AND val_vol_p ≤ 4.5%
- **MARGINAL:** val_abupt ≤ 7.6% AND val_vol_p ≤ 5.0%
- **KILL:** otherwise

(Historical PR #972 SDF stack gates were ≤ 6.2% / ≤ 6.5% — those reflect SDF-stratified sampling that is NOT on tay; do not apply to current single-model runs.)

### WSS-Targeted Single-Model Win Criteria (becomes new pool member)
- test_WSS ≤ 6.50% AND test_vol_p ≤ 3.643% AND test_SP ≤ 3.577% AND val_abupt ≤ 6.20%

### Ensemble Win Criteria (true new SOTA after PR #1102)
- val_abupt < **5.7452%** AND test_vol_p ≤ **3.643%** AND test_WSS < **6.3263%**

---

## Current Research Focus and Themes

### Primary: WSS Magnitude Bottleneck Attack (Wave 28 onwards — single-model only)

**New mechanism finding from PR #1097 close (tanjiro, WSS direction loss NEGATIVE):**
- WSS **direction is essentially solved** — cos_sim stabilises at 0.996 (~5° angular error) by EP2.
- **91–96% of remaining WSS residual is magnitude error.**
- This pivots the campaign from "direction-aware" experiments (which #1094, #1096, #1097 all targeted) toward **magnitude-targeted** mechanisms (rel_l2 loss, magnitude penalty) and **frame-equivariance** (in-plane rotation aug).

### Pool Saturation — CONFIRMED (PR #1103 closed 2026-05-14 13:30Z)

The current 4-member candidate pool {`56bcqp3m`, `29nohj67`, `a0yoxy85`, `ghh0s4ne`} is Pareto-saturated under convex combinations:
- PR #1102 K=8 Caruana (MERGED) — near-globally-optimal at discrete 1/8 grid
- PR #1099 K=3 WSS-targeted (CLOSED) — converged to identical K=3 subset as #1064
- PR #1103 SLSQP continuous optimisation (CLOSED) — confirmed K=8 within ~0.03 L1 of global continuous optimum; best-case val_WSS improvement = 0.0039pp (0.06% relative); val_SP ≤ 3.577% **infeasible** on this pool (simplex floor ~3.72%, every member ≥ 3.98%)

**Active lever for ensemble gains:**
1. **Pool extension via new single-model members** — only remaining path (ensembles BANNED per human directive)

⚠️ **ENSEMBLES BANNED** — Per morganmcg1 Issue #1056 directive 2026-05-14 14:17Z: no new ensemble experiments until explicitly unlocked. PR #1108 (bias-corrected ensemble) was superseded by PR #1109 (τ_z focal loss) before training started; #1108 is effectively dead.

---

## Active WIP PRs (as of 2026-05-15 ~02:50Z)

### Wave 28.5 → Wave 29 transition complete — all 8 students in flight

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| ~~#1114~~ | tanjiro | ~~Learnable WSS channel loss weights~~ | **CLOSED 01:13Z** — mechanism null at convergence; reassigned → #1124 |
| **#1116** | edward | **Per-channel WSS output heads** — decouple tau_x/tau_y/tau_z heads + 18h convergence | CHANGES REQUESTED 01:25Z — re-running at 18h to confirm matched-budget −0.66pp test_WSS delta holds at EP13; projected test_WSS ≈ 6.33% if delta holds (would tie ensemble SOTA) |
| ~~#1118~~ | askeladd | ~~OHEM v2 spike-clipped~~ | **CLOSED 02:35Z** — `clip_active`=100% → zero OHEM gradient → mathematically baseline-equivalent; reassigned → #1127 |
| ~~#1119~~ | fern | ~~GradNorm short-cycle (t_max=6, ep=6)~~ | **CLOSED 02:27Z** — REFUTES prior-rediscovery hypothesis; τ_z weight plateaus 1.07 (vs prior 2.0); hardcoded prior empirically validated; reassigned → #1126 |
| ~~#1120~~ | nezuko | ~~Spatial-prior surface sampling α=3~~ | **CLOSED 02:30Z** — mechanism right (ρ=+0.31 PASS), EP3 budget too short; strongest 3-EP truncated WSS in family but truncated; reassigned → #1125 (α=10 + 18h) |
| **#1121** | frieren | **WSS magnitude-only decomposition + 18h budget** — `λ_dir=0.0`, full 13-ep cosine; tests Wave 27 "91-96% magnitude" claim | Active WIP; EP3 gate ~02:48Z; EP13 ~14:00Z |
| **#1122** | alphonse | **SDF importance sampling port to tay — FAR-field α=2.0 (corrected mechanism)** — `weight = 1 + α·|sdf|`; highest-EV untested-on-tay lever; reproduces PR #972 SOTA mechanism (NOT the inverted `1/(1+α·|sdf|)`) | Active WIP draft post-correction (01:41Z); smoke-then-full plan approved; EP3 gate ≤6.9% PASS |
| **#1123** | thorfinn | **τ_z dedicated subnet** — 2-layer MLP head attacking residual axis test_τ_z ≈ 9.05% | Active WIP; launched 23:50Z post-#1100 close |
| **#1124** | tanjiro | **EMA decay 0.9995 + 18h budget** — single-flag test of slower EMA half-life (~1386 vs 693 steps) for late-converging τ_z | Active WIP; assigned 01:18Z post-#1114 close; full 13-EP convergence test |
| **#1125** | nezuko | **Spatial-prior surface sampling α=10 + 18h budget** — stronger oversample at full convergence (student's suggested follow-up #2); tests if mechanism scales without catastrophe | Active WIP; assigned 02:23Z post-#1120 close |
| **#1126** | fern | **Deeper surface_out MLP (depth 2→4) + 18h budget** — Wave 29 architectural pivot; tests if τ_z magnitude is decoder-depth-limited at surface head; orthogonal to thorfinn #1123 (separate branch) and edward #1116 (per-channel heads) | Active WIP; assigned 02:45Z post-#1119 close |
| **#1127** | askeladd | **Explicit surface_loss_weight warmup curriculum (3-ep ramp 0→full) + 18h** — directly tests #1114 implicit-curriculum finding; gradient-flow-preserving (avoids OHEM #1118 trap) | Active WIP; assigned 02:50Z post-#1118 close |

---

## Wave 28 Closures (2026-05-14 19:43Z–21:33Z) — methodology data captured, all reassigned

| PR | Student | Result | Key Mechanism Finding | Reassigned As |
|----|---------|--------|----------------------|---------------|
| #1109 | edward | val_WSS=8.766% EP3 (+1.6pp vs no-decomp ref) | Spatial focal α=2.0 amplifies per-point WSS errors at hot-spots faster than they can train down; underweights smooth bulk; baseline isn't smooth-dominated | #1116 per-channel heads |
| #1110 | askeladd | OHEM scale-collapse @ EP3 | Top-20% mining catastrophically scale-collapses without spike-clip; magnitude of L_hard explodes vs base loss | #1118 OHEM v2 spike-clipped |
| #1111 | fern | GradNorm test floors regress (test_vol_p +0.5pp, test_SP +0.4pp) | GradNorm de-emphasizes τ_z prior (hardcoded 2.0 weight); short-cycle test needed to disambiguate prior-vs-learned at convergence | #1119 GradNorm short-cycle |
| #1112 | frieren | Truncated EP3.5 @ 270-min wall-clock; calibration validated (mag head ratio=0.979 at half-cooked) | Mag head infrastructure works; full budget needed for convergence test | #1121 mag-only + 18h budget |
| #1113 | nezuko | val_abupt=8.04% EP3 (KILL) | Curvature is anti-correlated WSS proxy (ρ=-0.11); curvature-weighted sampling steers attention AWAY from high-WSS regions | #1120 spatial-prior (ρ=+0.31) |

## Wave 27 Closures (2026-05-14 ~13:45Z) — CATASTROPHIC FAILURE

All 4 experiments failed at EP3 with val_abupt 27–32% (4× above EP3 KILL gate of 7.6%). Root causes:

| PR | Student | val_abupt@EP3 | Root Cause |
|----|---------|---------------|------------|
| #1104 | fern | ~27% | L1 magnitude penalty `|‖τ‖−‖τ_gt‖|` creates conflicting gradients vs MSE loss; loss scale mismatch blows up training |
| #1105 | tanjiro | ~30% | Relative L2 `(pred-gt)²/‖gt‖²` numerically explodes when GT~0; near-zero WSS regions produce infinite loss |
| #1106 | frieren | ~28% | Physical-coordinate normal-frame rotation corrupts geometry signal — coordinate transformation invalidates learned features |
| #1107 | nezuko | ~32% | Yaw augmentation destroys physical orientation; model cannot learn orientation-dependent aerodynamics |

Common diagnosis: Wave 27 hypotheses all modified the **loss function or input transformation** at a fundamental level without sufficient numerical safeguards. The supplementary-loss OHEM approach (Wave 28) is designed to avoid these failure modes by adding a *supplementary* term (not replacing the base loss) with warmup and floor guards.

## Wave 26 Additional Kill (2026-05-14)

| PR | Student | Result | Key Mechanism Finding |
|----|---------|--------|----------------------|
| #1081 | askeladd | KILL @ EP10 (val_abupt=7.97%) | slw=3.0 surface loss weight — too aggressive; distorts vol_p head; baseline slw=2.0 is optimal |

## Wave 26 Closures (2026-05-13 → 2026-05-14)

| PR | Student | Result | Key Mechanism Finding |
|----|---------|--------|----------------------|
| #1094 | frieren | KILL @ EP3 (val_abupt=7.465%) | Normal-frame supervision built in normalised space — non-orthonormal |
| #1095 | nezuko | NEGATIVE (test_WSS=7.761% +1.03pp) | GradNorm mechanism healthy but starved vol head; curriculum is load-bearing |
| #1096 | edward | NEGATIVE (test_WSS +0.261pp vs ref) | Tangent-frame features redundant with normals; z-hat fallback discontinuity |
| #1097 | tanjiro | NEGATIVE (val_abupt=6.847% > KILL) | Direction NOT the bottleneck (cos_sim=0.996) |
| #1099 | fern | CONVERGED (same K=3 as #1064) | WSS-targeted greedy on 4-member pool converges to identical subset |
| #1102 | fern | **MERGED — new ensemble SOTA** | K=8 Caruana extracts ghh0s4ne WSS signal at 12.5% weight; NOW BANNED FROM EXTENSION per human directive |

---

## Baseline Training Recipe (current tay stack — NOT PR #972 SDF stack)

⚠️ **IMPORTANT:** the PR #972 SDF-stratified vol sampling code (`--sdf-importance-sampling --sdf-alpha 4.0`) was **never merged into tay**. Do NOT include those flags in any assignment — `argparse` will reject them. The live tay baseline is the stack below (no SDF importance sampling). Single-model EP3 on this baseline lands ~6.7–6.9% val_abupt, not the historical PR #972 6.2%. Gates must be recalibrated accordingly: PASS ≤ 7.2%, MARGINAL ≤ 7.6%, KILL otherwise.

```
--optimizer lion --lr 9e-5 --weight-decay 5e-4
--tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0
--use-ema --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1
--pos-encoding-mode string_separable --use-qk-norm
--rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"
--lr-cosine-t-max 13 --epochs 13
--vol-points-schedule "0:16384:3:32768:6:49152:9:65536"
--no-compile-model
--model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128
--batch-size 4 --validation-every 1
--train-surface-points 65536 --eval-surface-points 65536
--train-volume-points 65536 --eval-volume-points 65536
--use-surf-to-vol-xattn
--data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511
```

The PR #972 single-model SOTA W&B run `56bcqp3m` was trained with SDF-stratified sampling on a different branch (`dl24-frieren/vol-test-domain-augmentation`, commit `291efd2`); that code never landed on tay. Until it does, all new single-model runs are evaluated relative to the no-SDF tay baseline (thorfinn #1100 EP3=6.768% is a representative live trajectory).

---

## Next-Wave Hypothesis Queue

Wave 28.5 → Wave 29 in flight — 8 students busy, zero idle. Capacity students (alphonse #1078, thorfinn #1100) closed; SDF mechanism port (#1122) is the highest-EV active experiment.

Queue for Wave 30 (after current cohort lands ~tomorrow):

1. **Higher τ_z loss weight (3.0 from current 2.0)** — if fern #1126 and edward #1116 reveal decoder is the bottleneck, the prior τ_z=2.0 may now be undertuned with the increased capacity.
2. **WSS magnitude/direction joint head** — if frieren #1121 mag-only beats baseline, add a second head for direction (cos_sim) with bounded loss; combine via learnable α.
3. **Multi-scale surface attention** — second surface encoder at 0.5× token density to capture macro-flow features.
4. **Heteroscedastic WSS loss** — model both mean and variance per surface point; downweight high-aleatoric regions.
5. **τ_z frequency analysis** — Fourier decompose tau_z predictions vs GT to find spatial frequency bands where error is concentrated; use to motivate loss or architecture changes.
6. **Surface point sampling Voronoi tessellation** — sample uniformly over surface area (not raw vertex density) to remove sampling bias from non-uniform mesh refinement.
7. **Combine SDF FAR-field α=2.0 (from #1122) with deeper surface_out (from #1126)** — if both win independently, the combination is the obvious next step; orthogonal mechanism stacking.
8. **Curriculum at higher granularity** — if askeladd #1127 surface-loss warmup wins, try ramping individual WSS channel weights (τ_z last) instead of the global scalar.

⚠️ Permanently retired (catastrophic failure): yaw aug (#1107), magnitude penalty (#1104), rel_l2 (#1105), normal-frame rotation (#1094, #1106), curvature-weighted surface sampling (#1113 — wrong-sign proxy).

---

## Infrastructure Status

### GitHub Token Rate Limiting (RESOLVED 2026-05-14)
Senpai PR #3445 merged 06:42Z deployed per-student token fix + REST API migration. Fleet was back online by ~07:30Z. No further rate-limit-driven idle GPU incidents reported in current invocation.

### Pod Health
All 8 students have active pods (kubectl: `senpai-drivaerml-ddp8-*` deployments, 1/1 ready). DDP via 8× H100 96GB per student. Zero idle students as of 02:50Z. **Wave 28.5 → Wave 29 transition complete.** PR distribution: edward #1116 (per-channel heads 18h), frieren #1121 (magnitude-only 18h), alphonse #1122 (SDF FAR-field α=2.0), thorfinn #1123 (τ_z subnet), tanjiro #1124 (EMA decay 0.9995 18h), nezuko #1125 (spatial-prior α=10 18h), fern #1126 (surface_out depth=4 18h), askeladd #1127 (surface_loss warmup curriculum 18h).

---

## Key Findings to Date

- **WSS error is magnitude-dominated** (91–96% of residual, not direction) — pivot away from direction-aware experiments
- **tau_z (spanwise) still worst axis** (test_tau_z=8.2585% on PR #1102) — primary remaining target
- **Wave 27 catastrophic lesson**: NEVER replace base MSE loss — always use supplementary/additive formulations; loss scale mismatches and numerical instability (div-by-near-zero) destroy training even at 27–32% val_abupt; Wave 28 OHEM designed as additive supplement with 2-ep warmup to avoid this
- **Relative L2 loss is unstable** (PR #1105) — near-zero GT WSS regions produce unbounded loss; avoid any loss form with GT in denominator without explicit safeguards
- **slw=3.0 surface weight too aggressive** (PR #1081 killed) — baseline slw=2.0 is optimal
- **ENSEMBLES BANNED** (human directive 2026-05-14 14:17Z) — all new work must improve the single-model SOTA
- **Corrected dataset** (2026-05-11) eliminated artificial ~3× vol_p OOD gap — biggest research-program insight
- **SDF-stratified vol importance sampling** (PR #972) is single-model SOTA: val_abupt=6.126%, test_WSS=6.727%
- **Ensemble SOTA** (PR #1102 K=8 Caruana) test_WSS=6.3263% — first compliant ensemble below 6.33%
- **4-pool Pareto-saturated** (PR #1103 CONFIRMED) — K=8 within 0.03 L1 of global continuous optimum; val_SP ≤ 3.577% infeasible on this pool (simplex floor ~3.72%); new pool members are the operative lever
- ~~**Bias-corrected ensemble** (PR #1108)~~ — closed (superseded by τ_z focal loss #1109; ensemble research BANNED)
- **Training-time vol sampling** matters more than loss weighting or architecture depth for vol_p
- **Throughput regression risk on data-pipeline experiments** (nezuko #1113 self-diagnosed 12× slowdown from 20s/case curvature compute serialised through 4 workers; fix = precompute-and-cache; advisor must spec precompute step in any future data-pipeline assignment)
- **Curvature is a bad WSS proxy** (PR #1113 closed) — surface curvature is anti-correlated with |WSS| (ρ=-0.11); using curvature to oversample steers attention AWAY from high-WSS regions. Spatial position (`-x + |z|`) achieved ρ=+0.31 by contrast (PR #1120).
- **270-min wall-clock budget hits Wave 28 recipe at EP3.5** (#1111, #1112, historical #1095 all truncated) — recipe runs 76 min/epoch; full 13-ep cosine needs ~16h. Two responses available: recipe shrink (short t_max, fern #1119) or budget bump (`SENPAI_TIMEOUT_MINUTES=1100`, frieren #1121, matches alphonse #1078 working regime).
- **GradNorm de-emphasizes τ_z hard-coded prior** (#1111 close) — when learned, GradNorm reduces τ_z weight from prior 2.0 toward 1.4, which regresses test_vol_p and test_SP floors. Question: is the 2.0 prior over-tuned, or is the learned weight wrong? Short-cycle test (#1119 fern) measures this at full convergence.
- **OHEM scale-collapse** (#1110 close) — top-k mining catastrophically collapses without spike-clipping; magnitude of L_hard scales superlinearly when targeting top-20% of L distribution.
- **Spatial focal α=2.0 amplifies hot-spot error faster than training rate** (#1109 close) — per-point focal modulation creates concentrated gradients on outliers; baseline isn't bulk-smooth-dominated so amplification destabilizes optimization.
- **val→test ratio is NOT stable across eval configurations** (#1078 close) — asymmetric eval 131k produced val→test ratio of 1.020, not the 0.935 anchored on PR #972. The 0.935 ratio is recipe-specific (SDF stack), not transferable. Advisor SOTA projections must use test results from comparable-recipe runs, not synthetic val × historical ratio. This is a methodology guard for the entire program.
- **18h budget recipe validated end-to-end** (#1078 close): `SENPAI_TIMEOUT_MINUTES=1100` ran 17 epochs cleanly at ~62 min/ep (faster than initially projected). All future Wave 28+ runs can adopt it confidently; frieren #1121 has already.
- **Capacity-uplift ceiling on no-SDF tay is val_abupt ≈ 6.31%** (#1078 EP16 / #1100 EP16 close). Beyond that, the bottleneck is training-time sampling, not parameter count. Justifies #1122 (alphonse SDF port).
- **No-SDF tay structural ceiling at test_WSS ≈ 6.99%** (#1078 + #1100 close, two independent mechanisms). Asymmetric eval 131k and slices=256 capacity uplift both converge here at full convergence. Test floors regress under both. Any "beat SOTA without SDF" claim must beat 6.99% — capacity alone cannot. Direct paper-relevant finding.
- **τ_z is the program-wide residual axis** (test_τ_z ≈ 9.05% across all no-SDF runs). Consistently 30-45% worse than τ_x and ~18bp worse than τ_y. Justifies #1123 (thorfinn τ_z dedicated subnet) attacking representational capacity for τ_z specifically.
- **Initial-state debug crash** (tanjiro #1114 val_abupt=65.34% on 1-ep debug, then 8-rank DDP retry also crashed) — root cause likely learnable-weight unbounded growth; mitigated by lr=1e-3 separate group + L2 reg 1e-4 + 2-ep warmup option
- **Post-xattn capacity additions** 0-for-3 (PRs #884, #891, #906) — do not add layers after surf→vol xattn
- **Rotation aug** (PR #925): aggressive yaw+pitch degrades; mild yaw-only (≤45°) being tested in PR #1107
- **Normal-frame WSS in normalised space** fails (PR #1094); physical-frame variant (#1106) is the corrected attempt
- **Tangent-frame features** redundant with surface normals (PR #1096) — model already has the information
- **Direction loss** redundant with weighted MSE (PR #1097) — cos_sim=0.996 by EP2 without it
- **GradNorm + fixed-65k vol** fails because vol curriculum is load-bearing (PR #1095)
- **Infra:** `--data-root` (not `--data-path`); mount point `/mnt/new-pvc/` (not `/mnt/pvc/`)
- **EMA lag:** `eval_raw_vs_ema=False` means only EMA weights evaluated — early-epoch EMA metrics appear worse than raw
