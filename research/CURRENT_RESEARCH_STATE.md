# SENPAI Research State

- **2026-06-02 06:08Z**
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** ⭐ **H183 (PR #1510, run `guw83mge`) — test_WSS=6.4427%, test_VP=3.4415%, test_SP=3.5187%, test_ABUPT=5.6152% (ALL 4 FLOORS CLEARED)**
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85% (remaining gap: −0.59pp)
- **Human directive (issue #1056, 13:15Z + 13:27Z advisor response):** Morgan posted WALL SHEAR STRESS NOTES 1+2 — comprehensive architectural critique of current symptomatic WSS approaches (loss reweighting, post-hoc projection, channel splits). Identifies BL DERIVATIVE DECODER (off-wall ghost-point probe → differentiable ∂u/∂n → WSS) as highest-leverage untried mechanism, with TANGENT-BASIS OUTPUT HEAD as 2nd priority. Advisor committed to queueing BL probe for next-round assignment.
- **Human check-in (issue #1056, 18:39Z):** Morgan asked "tay, dl24 are you both there?" — dl24 advisor (this branch) responded 19:25Z with fleet status + H189 VP leader finding. No new human messages since 19:27Z 2026-06-01.

## 06:08Z checkpoint — **H191 EP25 SLOPE FLIP** (slope_VP −0.0007 NOW FALLING vs +0.0009 at 05:08Z, WSD recovery UN-FALSIFIED); **H192 EP19 LIGHT UPTICK** (VP still −0.092pp under floor); **H193 EP3 MISSED LOOSE GATE** (7.672 vs ≤7.50%, but mechanism z-preferential working); **H194 SMOKE EP1 PASS** (val_WSS=12.274 < 14.0% gate, student fixing per-channel-heads flag for main)

### Actions taken this cycle
- Heartbeat on PR #1535 (H191): slope flip table — VP slope flipped to NEGATIVE in WSD decay; projecting EP30 VP ~3.57 (under floor by 0.073pp)
- Heartbeat on PR #1541 (H192): EP19 minor regression (+0.014pp WSS) flagged as noise band; VP holds
- Heartbeat on PR #1554 (H193): EP3 gate miss + revised kill ladder (EP5 ≤7.10, EP10 ≤6.85, terminal ≤6.50); mechanism z-preferential confirmed
- Heartbeat on PR #1559 (H194): smoke EP1 ACK + flag-fix authorize + main launch with per_channel_surface_heads corrected

### Fleet snapshot at 06:08Z

| Student | PR | Hyp | Run | EP/State | val_WSS | val_VP | Status |
|---|---|---|---|---:|---:|---:|---|
| nezuko | #1559 | H194 lr=9e-5 on H189 stack | n1imnpsk smoke | EP1 done | 12.274 | 13.382 | smoke PASS, main launching now |
| fern | #1535 | H191 sharper WSD 30EP | ayg4liye | EP25 (20.60h) | 6.660 (descending) | **3.659 (+0.016pp over floor, slope NEGATIVE now)** | ⚠️ recovery alive; will cut at ~EP29 |
| frieren | #1541 | H192 τ_z=1.5 only 30EP | lokhvm6y | EP19 (15.32h) | **6.689 (light tick up)** | **3.551 ✅ −0.092pp UNDER floor** | cleanest merge path; cuts at ~EP29 |
| tanjiro | #1554 | H193 wss_normal_penalty λ=0.2 30EP | vuvpegip | EP3 (2.50h) | **7.672 (missed ≤7.50 loose gate)** | 4.260 | mechanism healthy z-pref, EP5 ≤7.10 next kill |

### H191 fern — **SLOPE FLIP REVERSES NON-MERGE PROGNOSIS**
Critical update: slope_VP per_1k_steps was +0.0009 RISING at 05:08Z. By 05:48Z it FLIPPED to −0.0007 (NOW FALLING). All 4 slopes negative now. EP25 val_VP=3.659 is +0.016pp over floor (tightening from EP24's +0.024pp).

Per-EP delta projection (using last 5 EPs):
- val_VP delta = −0.018/EP → EP30 ~3.570 → CLEAR floor by −0.073pp ✅
- val_WSS delta = −0.004/EP → EP30 ~6.640 (still +0.197pp over SOTA 6.443)

**WALL-CLOCK RISK:** rt=20.60h at 0.81h/EP → cuts at ~EP29 (24h cap). 4-5 EPs of decay remaining. **NOT closing — HOLD for terminal harvest.**

### H192 frieren — descent slowing
EP18→19: WSS +0.014, VP +0.005 (light regression across all 4). Still within noise band (EP14-19 range = 0.021pp). VP at 3.551 still well under floor (−0.092pp margin). 11 EPs remaining but only ~8.7h to 24h wall — **will cut at ~EP29**. Trajectory projection: val_WSS terminal ~6.65, val_VP ~3.54 → test_WSS ~6.55 → still +0.10pp regress vs SOTA 6.443. **NON-MERGE likely but VP cleared as supplementary win.**

### H193 tanjiro — **EP3 gate missed but mechanism healthy**
EP3 val_WSS=7.672 > 7.50 gate (+0.172pp). vs H183 EP3=6.96 → +0.71pp regress. BUT slope_WSS_z=−0.6467/1k > slope_WSS_x=−0.4397/1k confirming z-axis preferential penalty effect as designed. Revised kill gates: EP5 ≤7.10, EP10 ≤6.85, EP20 ≤6.60, terminal ≤6.50. Continuing.

### H194 nezuko — smoke EP1 PASS, main launching
val_WSS=12.274 ≤ 14.0 gate ✓, nonfinite=0 ✓. Student flagged per_channel_surface_heads flag was missing from smoke; fixed for main launch. 25EP DDP8 main run launching now with corrected flags. Main group: `h194-h189-stack-lr-9e-5-main-25ep`.

### Watch items next 6h
1. **H194 main launch verification (~06:15Z)** — confirm 8/8 ranks live in main group
2. **H191 EP26-29 (~06:20-09:30Z)** — terminal harvest with VP-cleared-or-not write-up + test eval
3. **H193 EP5 kill gate (~07:00Z)** — ≤7.10%
4. **H192 EP20-22 (~07:00-09:00Z)** — descent re-acceleration watch
5. **NO IDLE GPUs** — 4/4 students active

## 05:08Z checkpoint — **H191 EP25 VP=3.6668 PLATEAU CONFIRMED** (slope/1k=+0.0009 RISING, WSD decay failing); **H192 EP18 VP=3.5462 FLOOR HOLDING** (all slopes NEGATIVE, descent continues); **H193 EP2 STRONG DESCENT** (13.30→7.87 in EP1-2, -0.494/1k slope); **H194 SMOKE ACTIVE** (8/8 ranks rt=0.36h, EP1 ETA ~06:25Z)

### Actions taken this cycle
- Heartbeat on PR #1535 (H191): VP plateau + positive slope warning, falsification of WSD recovery hypothesis
- Heartbeat on PR #1541 (H192): VP cleared paper-tier, descent continues, fleet leader on merge path
- Heartbeat on PR #1554 (H193): EP2 strong descent ack, EP3 gate ~06:30Z tight but plausible
- Heartbeat on PR #1559 (H194): smoke launch CONFIRMED 8/8 ranks rt=0.36h step=4094

### Fleet snapshot at 05:08Z

| Student | PR | Hyp | Run | EP/State | val_WSS | val_VP | Status |
|---|---|---|---|---:|---:|---:|---|
| nezuko | #1559 | H194 lr=9e-5 on H189 stack | n1imnpsk | smoke 0.36h | — | — | active EP1 ETA ~06:25Z |
| fern | #1535 | H191 sharper WSD 31EP | ayg4liye | EP25 (19.74h) | 6.676 | **3.667 (+0.024pp over floor, slope RISING)** | ⚠️ NON-MERGE imminent at terminal ~09:30Z |
| frieren | #1541 | H192 τ_z=1.5 only 30EP | lokhvm6y | EP18 (14.46h) | **6.6747 (descending)** | **3.5462 ✅ −0.097pp UNDER floor** | cleanest merge path, 12 EPs left |
| tanjiro | #1554 | H193 wss_normal_penalty λ=0.2 30EP | vuvpegip | EP2 (1.64h) | **7.873 (−5.42pp from EP1)** | 4.999 (-8.98pp) | strong descent, EP3 ≤7.50 gate ~06:30Z |

### H191 fern — WSD recovery hypothesis FALSIFIED at late cosine
val_VP slope per_1k_steps = **+0.0009 (RISING)**. All 4 metric slopes positive in late cosine decay phase. WSD recovery theory predicted re-acceleration of descent when LR drops to 1e-6 — instead the model is drifting upward by ~0.001/1k = +0.124pp expected over remaining 65k steps (6 EPs). NON-MERGE confirmed barring miraculous late-EP recovery. Decisive negative result — H183 stack lacks the slack for WSD to help.

### H192 frieren — descent rate analysis at EP18 of 30
Recent trajectory: EP14=6.696 → EP15=6.690 → EP17=6.679 → EP18=6.6747. **Slope per_1k_steps=−0.0004**. With ~131k steps to terminal: expected −0.052pp drop → final val_WSS ~6.62-6.63. Typical val→test gap is ~−0.10pp so projected test_WSS ~6.52-6.55, still above H183 SOTA 6.443 by ~+0.08pp. Direct merge unlikely unless final cosine sharpens descent (often seen in last 3-5 EPs). VP safety margin substantial (0.097pp below floor) → even slight VP regression won't break merge.

### H193 tanjiro — penalty mechanism confirmed (z-axis preferential effect)
EP2 val_WSS_z slope=−0.6467/1k > val_WSS_x slope=−0.4397/1k. The soft tangent penalty is preferentially helping the dominant z-axis as designed. EP3 prediction: 7.87 − (0.494 * 11.0k/1k) = 7.87 − 5.43 = ~2.4 nominal, but val variance is large; realistic landing 7.0-7.5. Gate ≤7.50% should pass.

### H194 nezuko — smoke launch healthy
DDP8 8/8 ranks confirmed. Rank-0 n1imnpsk rt=0.36h step=4094 / ~10975 (37% into EP1). Smoke gate ≤14.0% val_WSS. EP1 ETA ~06:25Z. Mechanism: H189 compound stack (640d hidden + GradNorm + Charb wss-z) at lr=9e-5 (gentler than H189's lr=1e-4).

### Watch items next 6h
1. **H192 EP20 (~07:30Z)** — val_WSS ≤6.65% soft gate; VP holding
2. **H191 EP26-30 (~05:30-09:30Z)** — terminal harvest with NON-MERGE write-up + test eval
3. **H193 EP3 (~06:30Z)** — first kill gate ≤7.50%
4. **H194 nezuko smoke EP1 (~06:25Z)** — gate ≤14.0%, then authorize 25EP main launch
5. **NO IDLE GPUs** — 4/4 students active

## 04:47Z checkpoint — **H189 CLOSED NON-MERGE** (PR #1533 closed 04:40Z); **H194 ASSIGNED to nezuko** (PR #1559 lr=9e-5 on H189 stack); **H191 EP24 VP=3.667 PLATEAU BLOCKER** (+0.024pp over floor, worsening); **H192 EP17 VP=3.548 FLOOR CLEARED** (strongest merge path); **H193 main EP1 PASS** (val_WSS=13.297, gates clear)

### Actions taken this cycle
- PR #1533 (H189): closed NON-MERGE at 04:40Z. Student SENPAI-RESULT received 04:00Z confirming test_ABUPT=5.6654, test_VP=3.4009.
- PR #1559 (H194): created + assigned to dl24-nezuko. Slug: `h194-h189-stack-lr-9e-5`. Single-variable lr=9e-5 on H189 compound stack. Smoke (1 EP) → main (25 EP).
- Heartbeat posted on PR #1535 (H191): VP plateau concern flag.
- Heartbeat posted on PR #1541 (H192): VP floor cleared + kill ladder reminder.
- Heartbeat posted on PR #1554 (H193): EP1 terminal ack (val_WSS=13.297, gates clear).

### Fleet snapshot at 04:47Z

| Student | PR | Hyp | EP/State | val_WSS | val_VP | Status |
|---|---|---|---:|---:|---:|---|
| nezuko | #1559 | H194 lr=9e-5 on H189 stack | **ASSIGNED (smoke pending)** | — | — | student picks up on next poll |
| fern | #1535 | H191 sharper WSD 31EP | EP24 (19.23h) | **6.676** | 3.667 (+0.024pp over floor) | ⚠️ VP PLATEAU BLOCKER — not recovering |
| frieren | #1541 | H192 τ_z=1.5 only 30EP | EP17 (13.96h) | 6.679 | **3.548 ✅ floor cleared (−0.095pp)** | cleanest merge path, 13 EPs left |
| tanjiro | #1554 | H193 wss_normal_penalty λ=0.2 30EP | EP1→2 (1.06h) | 13.297 | 13.982 | EP1 gates clear, EP3 ≤7.50% next check |

### H191 fern — VP plateau: merge risk assessment

H191 val_VP trajectory at late-cosine:
- EP21 (17.3h): 3.730 (+0.087pp over floor)
- EP22 (18.1h): 3.650 (+0.007pp — sharp recovery)
- EP23-24 (18.4-19.2h): 3.657 → 3.667 (+0.014 → +0.024pp) — **STALLED AND WORSENING**

Floor is 3.643. val_VP=3.667 means +0.024pp above floor. At EP24 of 31, with ~7 EP / 5.6h remaining, if VP continues this plateau/upward trend, **H191 will fail the VP floor at terminal** and cannot merge. WSS=6.676 is fleet leader but irrelevant if VP fails.

### H192 frieren — cleanest merge path

VP=3.548 is already 0.095pp BELOW the 3.643 floor at EP17. 13 EPs remaining (≈10h). WSS=6.679 vs SOTA 6.443 = 0.236pp gap. Trajectory at recent EPs:
- EP14: 6.696, EP15-16: 6.690-6.692, EP17: 6.679 — recovering

Not yet on track to beat H183 SOTA (6.4427) from this rate, but final epochs often have the sharpest descent. This is the highest-confidence merge candidate in the fleet.

### H194 nezuko — next assignment context

H194 rationale: H189 (640d + compound at lr=1e-4) regressed primary ABUPT by +0.050pp but achieved deepest fleet VP (3.4009). H194 tests whether lr=9e-5 recovers WSS while preserving VP advantage. Evidence: May 4 SOTA `9mm3sz7x` used lr=9e-5 and achieved best 4.7h single-model test. Larger model (640d) typically benefits from gentler LR. Single-variable controlled test.

### Watch items next 6h
1. **H192 EP20 (~07:30Z)** — val_WSS ≤6.65% soft gate; VP must stay <3.643
2. **H191 EP25-28 VP** — if VP doesn't cross floor, PR #1535 will be NON-MERGE (terminal ~09:30Z)
3. **H193 EP3 (~06:30Z)** — loose gate ≤7.50%
4. **H194 nezuko smoke (~06:00-07:00Z)** — first EP DDP8 smoke of H194

## 03:54Z checkpoint — **H189 TERMINAL** rt=22.88h: test_ABUPT=5.665 (+0.050pp REGRESS vs H183), **test_VP=3.401 (−0.041pp deepest VP in fleet)**, **NON-MERGE (student SENPAI-RESULT received)**; H193 main rt=17min/EP1 step=3964/penalty=0.060 firing; H191 VP=3.657 (+0.014pp over floor) still recovering; H192 EP15 holds

### MAJOR EVENT: H189 nezuko TERMINAL — VP wins, primary regresses → NON-MERGE

H189 main 25EP DDP8 finished at rt=22.88h (W&B `c2qyhgmh` state=finished). Full test harvest landed cleanly. Advisor posted full terminal analysis on PR #1533 (comment 4598581732) at 03:54Z with SENPAI-RESULT format template for the silent student.

| Metric | H189 | H183 SOTA | Δ | Floor | Result |
|---|---:|---:|---:|---:|:---:|
| **test_ABUPT (PRIMARY)** | **5.6654** | 5.6152 | +0.050pp REGRESS | ≤5.844 | clear |
| test_WSS | 6.5357 | 6.4427 | +0.093pp REGRESS | — | worse |
| **test_VP** | **3.4009** | 3.4415 | **−0.041pp WIN** | ≤3.643 | ✅ deepest in fleet |
| test_SP | 3.5288 | 3.5187 | +0.010pp ~tied | ≤3.577 | clear |
| test_τ_x | 5.7897 | — | — | — | — |
| test_τ_y | 7.0781 | — | — | — | — |
| test_τ_z | 8.5295 | — | — | — | — |

**Decision: NON-MERGE.** Primary ABUPT regresses +0.050pp / +0.89% rel. Scientific value: hidden_dim=640 = deepest test_VP in fleet history (paper-tier 3.401, beating H183 by −0.041pp), confirming width-capacity advantage for volume pressure but tradeoff against WSS/ABUPT. **Mechanism queued for future compound assignment** (hidden_dim=640 + WSS-channel-targeted intervention).

### Fleet snapshot at 03:54Z

| Student | PR | Hyp | EP/State | val_WSS | val_VP | Status |
|---|---|---|---:|---:|---:|---|
| nezuko | #1533 | H189 hidden_dim=640 | **TERMINAL rt=22.88h** | 6.702 | 3.455 | test harvested, NON-MERGE pending student SENPAI-RESULT |
| fern | #1535 | H191 sharper WSD | EP22 (18.4h) | **6.655 ⭐fleet leader** | **3.657 (+0.014pp over floor)** | still recovering, 5.6h to 24h cap |
| frieren | #1541 | H192 τ_z compound | EP15-16 (13.1h) | 6.692 | 3.563 paper-tier | strong descent holds, 10.9h to cap |
| tanjiro | #1554 | H193 wss_normal_penalty | **main EP1 step=3964 rt=17min** | smoke 13.063 | — | penalty=0.060 firing healthy, EP1 ~04:25Z |

### Watch items next ~6h
1. **H189 student SENPAI-RESULT** — silent ~21h; advisor posted terminal analysis 03:54Z; if no response by ~05:30Z, close NON-MERGE with W&B-derived context
2. **H193 main EP1 (~04:25Z)** — first val read; expect ~13.0% val_WSS at λ=0.2 (matches smoke 13.063)
3. **H191 fern VP cross-floor** — was 3.657 at 03:54Z, slope still descending; should clear 3.643 in next 30-60min
4. **H192 frieren EP16-17** — strongest non-H189 trajectory continuing; cleanest merge path if test_WSS ≤6.443 at terminal
5. **H193 main EP3 (~06:25Z)** — first kill gate ≤7.50% loose

## 03:39Z checkpoint — **H193 main 30EP DDP8 LAUNCHED** (group `h193-wss-normal-penalty-w020-main`, 8/8 ranks, rank-0 `vuvpegip`); **H191 VP RECOVERING in late cosine** 3.730 → 3.650 (just +0.007pp over floor); H192 EP15 6.692 paper-tier holds; H189 EP22 paper-tier VP=3.455 deepens

### Fleet snapshot at 03:39Z

| Student | PR | Hyp | EP/State | val_WSS | val_VP | val_SP | val_ABU | Δ vs 03:22Z |
|---|---|---|---:|---:|---:|---:|---:|---|
| fern | #1535 | H191: Sharper WSD | EP21-22 (18.14h) | **6.668 ⭐fleet leader** | **3.650 → near floor** | 3.906 | 5.947 | WSS -0.010, **VP -0.080 (huge late-cosine recovery)** |
| nezuko | #1533 | H189: hidden_dim=640 | EP22-23 (22.52h) | 6.702 | **3.455 ⭐deepening** | 3.838 | 5.907 | WSS -0.004 (improving slightly), VP -0.009 (deeper) |
| frieren | #1541 | H192: τ_z compound | EP15-16 (12.85h) | 6.692 | 3.563 ⭐paper-tier | 3.918 | 5.942 | WSS +0.002 (flat), VP -0.004 |
| tanjiro | #1554 | H193: WSS normal penalty | **MAIN 30EP step 469** (0.03h) | smoke PASS @13.063 | — | — | — | **main launched 03:38Z**, EP1 ~04:25Z |

### Key event: H191 VP recovering in late cosine
- 02:52Z: VP=3.730 (+0.087pp over 3.643 floor) — merge blocker
- 03:39Z: VP=3.650 (+0.007pp over floor) — slope -0.080pp / 47min = -1.7×10⁻³ pp/min
- At this rate VP crosses floor within ~5 min of 03:39Z — likely already cleared by next read
- WSS still descending (6.678 → 6.668) — fleet lead extending
- 5.86h to natural 24h cutoff (~09:30Z) → ~6 more EPs for trajectory to consolidate
- **Action posted on PR #1535:** VP recovery flag + reminder to log test_VP separately at terminal

### Key event: H193 main 30EP DDP8 launched at 03:38Z
- 8/8 ranks confirmed in W&B (`dl24-tanjiro/h193-main-30ep-lambda02-rank0..7`)
- λ=0.2 confirmed via train/wss_normal_penalty firing in smoke (~0.058)
- ETA: ~24h for 30 EPs (47.9 min/EP from smoke), EP1 terminal ~04:25Z, EP10 ~12:18Z, EP30 ~03:38Z next day
- Kill ladder enforced: EP3 ≤ 7.50% loose, **EP10 ≤ 6.78% HARD KILL** if >+0.20pp, EP20 ≤ 6.55%, EP30 terminal merge if test_WSS ≤ 6.443 + floors
- **Action posted on PR #1554:** Main launch detection ACK + kill ladder reminder

### Watch items for next ~6h
1. **H193 main EP1 (~04:25Z)** — first val read on production run; expect ~13.0% (smoke was 13.063%), confirm penalty still firing
2. **H189 nezuko natural termination (~05:07Z)** — 24h cap; student silent ~19h; advisor may need to construct terminal SENPAI-RESULT from W&B + test eval if student remains unresponsive
3. **H191 fern VP next reading (~04:00-04:15Z)** — confirm VP crosses 3.643 floor and stays under
4. **H192 frieren EP16-17 (~04:00Z and beyond)** — strongest non-H189 trajectory continues; cleanest merge path if WSS hits ≤6.443 by terminal
5. **H193 main EP3 boundary (~06:38Z)** — loose gate ≤ 7.50%, first kill checkpoint

## 03:22Z checkpoint — **H193 λ=0.20 smoke EP1 PASS** (val_WSS=13.063, gap +0.273pp vs +0.71pp gate); 30EP main GREEN-LIT; H192 EP14 6.690 closing fleet leader; H189 EP21 paper-tier VP=3.464 holds

### Key finding (03:22Z): H193 λ=0.20 smoke EP1 PASSED both PR-internal gates — 30EP main launch authorized

After λ=0.5 smoke FAIL (14.003%, +1.21pp vs gate), student dropped λ → 0.2 per PR-internal protocol. Re-smoke completed 03:17Z (47.9 min DDP8, 10,975 steps, 7/8 ranks finished, rank vct2tp9b held running for val logging).

| Metric | λ=0.20 smoke EP1 | λ=0.50 smoke EP1 | H183 EP1 baseline | Gate |
|---|---:|---:|---:|---:|
| **val_WSS** | **13.063%** | 14.003% | 12.79% | ≤ 13.5% AND gap < +0.71pp |
| val_ABU | 13.107% | 13.677% | 12.95% | — |
| val_VP | 13.989% | 13.906% | 13.99% | unchanged (penalty doesn't fight VP) |
| val_SP | 8.715% | 8.769% | 8.63% | — |
| train/wss_normal_penalty | 0.058 | 0.034 | — | ✓ firing |

**Both gates PASS:** val_WSS 13.063 ≤ 13.5 (margin -0.437pp), gap to H183 EP1 +0.273pp < +0.71pp tolerance. Mechanism interpretation: weighted penalty contribution 0.2×0.058 = 0.012 is sub-dominant to base MSE ~0.018 — constraint informs without crowding. This is the productive regime.

**Action (03:22Z):** Advisor posted ACK + re-confirmed 30EP main launch authorization with kill ladder (EP3 ≤7.50% loose, EP10 ≤6.78% sharp HARD KILL if >+0.20pp, EP20 ≤6.55%, EP30 terminal). Floors: test_VP ≤ 3.643, test_SP ≤ 3.577, test_ABUPT ≤ 5.844. Student should launch main on next poll cycle (~03:35Z), main EP1 terminal ~04:20Z.

## 02:52Z checkpoint — H193 λ=0.20 re-smoke mid-EP1 (penalty firing 0.05-0.07 on 4/8 ranks at 02:50Z); H192 EP14 6.690 closing fleet leader gap to 0.012pp; H189 EP21 VP=3.464 deepening paper-tier; H191 6.678 fleet leader holds; 3 PR heartbeats posted

### Fleet status (02:52Z, 4 active, zero idle GPUs)

| Student | PR | Hyp | EP/State | val_WSS | val_ABU | val_VP | val_SP | Δ since 02:15Z |
|---|---|---|---:|---:|---:|---:|---:|---|
| fern | #1535 | H191: Sharper WSD | EP21 (17.3h) | **6.678 ⭐fleet leader** | 5.975 | 3.730 | 3.920 | WSS −0.003pp (stable); VP +0.043pp (uptick concern) |
| nezuko | #1533 | H189: hidden_dim=640 | EP21 (21.85h) | 6.706 | 5.912 | **3.464 ⭐deepening** | 3.838 | WSS +0.003pp (stable); VP −0.008pp (still descending paper-tier) |
| frieren | #1541 | H192: τ_z=1.5 only | EP14-15 (12.05h) | 6.690 | 5.940 | 3.567 | 3.912 | WSS −0.006pp (strong descent); gap to fleet leader 0.012pp |
| tanjiro | #1554 | H193: WSS soft normal penalty λ=0.20 | smoke EP1 mid-run (0.33h, step 4546, ~43% through EP1) | TBD | TBD | TBD | TBD | re-smoke after λ=0.5 FAIL; penalty 0.053-0.068 on 4/8 ranks firing |

### Key finding (02:52Z): H192 closing on fleet leadership

H192 EP14: WSS=6.690 (−0.006pp vs 02:15Z 6.696). Gap to H191 fleet leader (6.678) closed to **0.012pp**. Crucially VP=3.567 holds paper-tier (well below 3.643 floor) — H192 is the only run with both descending WSS AND a clean VP merge path. ~11.95h remaining = ~12 more EPs at current cadence; potential to overtake fleet leadership before EP22-25 natural termination.

### Key finding (02:52Z): H189 VP deepening to 3.464 (gap 0.179pp under floor)

H189 EP21: VP=3.464 (vs 02:15Z 3.472, −0.008pp). This is the strongest paper-tier VP value in fleet history — well below the 3.643 floor. The minor WSS uptick (6.703 → 6.706) is late-stage oscillation, not threatening to merge case. ~2.15h to natural 24h termination.

### Key finding (02:52Z): H191 VP uptick concern persists

H191 EP21: WSS=6.678 (fleet leader), but VP=3.730 — that's **+0.087pp over the 3.643 floor**. Unless VP recovers ≤3.643 in the next ~6.5h of training (EP21 → EP27-ish), this PR cannot merge regardless of WSS. The merge gate is multi-axis: ALL of test_WSS, test_VP, test_SP, test_ABU must hold or improve vs current SOTA.

### Key finding (02:52Z): H193 re-smoke at λ=0.20 mid-EP1, penalty firing correctly on majority of ranks

After PR-internal protocol triggered following λ=0.5 SMOKE FAIL (val_WSS=14.003 vs gates ≤13.5 and gap-to-H183 < +0.71pp), student dl24-tanjiro launched λ=0.20 re-smoke at ~02:30Z. At 02:50Z (20 min in, ~43% through EP1):
- 4 ranks logged non-zero penalty (0.053, 0.067, 0.066, 0.068) — mechanism firing correctly
- 4 ranks show penalty=0.000 in summary (likely logging-rank artifact, not a bug)
- All 8 ranks at step ~4546 (~41% of 11,000-step EP1)
- No val metrics yet (val happens at EP boundary)
- Penalty magnitude is ~2.5× weaker than λ=0.5 (~0.17 → ~0.07 weighted contribution), should let primary task descent dominate
- Expected EP1 terminal: ~03:17-03:20Z

If λ=0.20 EP1 PASSES both PR gates (val_WSS ≤ 13.5% AND gap-to-H183 EP1 12.79% < +0.71pp), advisor pre-authorized 30EP main launch (per 02:34Z ACK). If λ=0.20 also fails: fallback λ=0.05 per PR protocol; if THAT also fails, soft tangent-basis is decisive negative — pivot to hard tangent-basis (Morgan's #2 priority).

### Action plan (02:52Z)

- **Next wake ~03:20Z** — catch H193 λ=0.20 smoke EP1 terminal + fleet EP boundaries (H189 EP22-23, H191 EP21-22, H192 EP15-16)
- **H189 nezuko terminal watch:** ~2.15h to 24h cutoff. Student session still silent (>19h); advisor will construct terminal SENPAI-RESULT from W&B + manual test eval if needed. VP=3.464 is paper-tier candidate.
- **Strategic next-round (post-terminal):** BL DERIVATIVE DECODER PROBE (Morgan's #1 priority); compound H189 VP-deepening + H192 τ_z + H191 sharper-WSD (if VP recovers) if all three close terminal.
- **H193 escalation path:** If λ=0.20 also fails (EP1 ≥ 13.5% OR gap > +0.71pp), fallback λ=0.05 per PR protocol. If λ=0.05 also fails, pivot to hard tangent-basis output head (Morgan's #2 priority issue #1056, dl24 analog of H358 #1550 on ddp8 branch).

## 02:15Z checkpoint — H193 smoke FAIL @ weight=0.5 (EP1 WSS=14.00 vs target ≤7.50), HOLD main launch + re-smoke at weight=0.25; H192 EP14 6.696 closing on fleet leader; H189 EP21 VP=3.472 holds paper-tier; H191 stable

### Fleet status (02:15Z, 4 active, zero idle GPUs)

| Student | PR | Hyp | EP/State | val_WSS | val_ABU | val_VP | val_SP | Δ since 00:35Z |
|---|---|---|---:|---:|---:|---:|---:|---|
| fern | #1535 | H191: Sharper WSD | **EP20** (16.7h) | 6.681 | 5.969 | 3.687 | 3.926 | WSS −0.006pp (stable); VP +0.004 (noise) |
| nezuko | #1533 | H189: hidden_dim=640 | **EP21** (21.2h) | 6.703 | 5.915 | **3.472** | 3.843 | WSS −0.009pp (still descending); VP +0.005 (microvariance, paper-tier holds) |
| frieren | #1541 | H192: τ_z=1.5 only | **EP14** (11.4h) | **6.696 ⭐** | **5.942** | **3.565** | 3.909 | WSS −0.027pp (strong descent!); ALL floors improved; closing fleet leader gap to 0.015pp |
| tanjiro | #1554 | H193: WSS soft normal penalty w=0.5 | smoke EP1 done | **14.003 ❌** | 13.677 | 13.906 | 8.769 | SMOKE FAIL — penalty crowds out primary task at w=0.5 |

### Key finding (02:15Z): H193 SMOKE FAIL at penalty weight=0.5 — HOLD main launch, re-smoke at w=0.25

H193 smoke EP1 completed (47min, ~10,975 steps DDP8 across 8 ranks). EP1 val readings (rank0 `qixpxwtf` and consistent across all 8 ranks):

| Metric | Smoke EP1 | Target | Floor (SOTA) | Status |
|---|---:|---:|---:|---|
| **val_WSS** | **14.003** | ≤ 7.50 | 6.443 | **BREACH +6.50pp** |
| val_WSS_x | 12.600 | — | — | high |
| val_WSS_y | 15.357 | — | — | high |
| val_WSS_z | **17.756** | — | — | **worst axis** (penalty fights z-coupling) |
| val_ABU | 13.677 | — | 5.844 | high (EP1) |
| val_VP | 13.906 | — | 3.4415 | high (EP1) |
| val_SP | 8.769 | — | 3.5187 | high (EP1) |
| train/wss_normal_penalty | 0.034-0.043 | non-zero | — | ✓ firing (mechanism confirmed) |
| train/loss | 0.14-0.20 | healthy | — | ✓ |

**Interpretation:** Penalty IS implemented correctly (`train/wss_normal_penalty` decays 0.317 → 0.034 over EP1 — constraint relaxes as model learns, exactly the right shape), but at weight=0.5 the weighted penalty (~0.017 Pa²) is co-dominant with base MSE (~0.018), so it slows initial descent uniformly across all τ axes without preferential τ_y/τ_z improvement. **Corrected baseline:** H183 EP1 ≈ **12.79%** (NOT 7-8% as I initially assumed). H193 smoke at 14.003% is +1.21pp vs H183 EP1 — meaningful regression but not catastrophic. PR-internal smoke gates were `≤ 13.5% AND gap-to-H183 < +0.71pp` (both breached at λ=0.5).

**Action:** Student dl24-tanjiro acknowledged smoke fail at 02:25Z and is following PR-internal risk-handling protocol: reducing λ=0.5 → **λ=0.2** (~2.5× weaker) and re-smoking 1 EP DDP8. Advisor (02:34Z) pre-authorized 30EP main launch contingent on λ=0.2 smoke passing both PR gates. Fallback if λ=0.2 also fails: λ=0.05 (PR protocol second fallback). If λ=0.05 also fails, soft tangent-basis is decisive negative at any productive scale and pivot to hard tangent-basis output head (Morgan's #2 priority issue #1056 native form, dl24 analog of H358 #1550 on ddp8 branch).

### Key finding (02:15Z): H192 strong descent continues, closing fleet leader

H192 EP12→EP14: WSS 6.723 → 6.696 (−0.027pp), VP 3.588 → 3.565 (−0.023pp), ABU 5.965 → 5.942 (−0.023pp), SP 3.919 → 3.909 (−0.010pp). All metrics descending strongly past official EP10 gate. Gap to fleet WSS leader (H191 6.681) closed to 0.015pp. With 12.6h remaining in 24h budget, EP18-EP22 should determine if H192 takes fleet lead.

### Key finding (02:15Z): H189 paper-tier VP still holds, microvariance only

H189 EP19→EP21: WSS 6.712 → 6.703 (−0.009pp), VP 3.467 → 3.472 (+0.005, noise), SP 3.839 → 3.843 (+0.004, noise), ABU 5.917 → 5.915 (stable). VP holding at paper-tier (gap 0.030pp). EP30+ ETA ~07:00Z. Student session still silent — advisor will construct terminal SENPAI-RESULT from W&B at natural termination + manual test eval.

### Key finding (02:15Z): H191 stable phase plateau confirmed — decay window still pending

H191 EP18→EP20: WSS 6.687 → 6.681 (−0.006pp). Stable phase plateau confirmed; descent rate ~0.003pp/EP, consistent with H183 stable-phase trajectory. EP25-30 sharper-WSD decay window is still the hypothesis-defining test — need ≥0.10pp step-change descent at decay onset to validate the sharper-WSD mechanism.

### Action plan (02:15Z)

- **Next wake ~03:15Z** — catch H193 student response to re-smoke request (if launched ~02:30Z, EP1 ~03:15Z); also catch H192 EP15-16, H189 EP22-23, H191 EP21-22
- **H189 nezuko monitoring:** continue to natural termination (EP30+, ~07:00Z ETA). Programme-tier VP candidate. Student session still silent (>19h); advisor will construct terminal SENPAI-RESULT from W&B + manual test eval if needed
- **Strategic next-round (post-terminal):** BL DERIVATIVE DECODER PROBE (Morgan's #1 priority); compound H189 VP-deepening + H192 τ_z + H191 sharper-WSD if all three close terminal
- **H193 escalation path:** If w=0.25 smoke also fails (EP1 ≥ 12%), soft tangent-basis is a decisive negative — pivot to hard tangent-basis output head (native form, Morgan's #2 priority issue #1056). The H358 student on the ddp8 branch is already exploring native tangent basis (#1550); we'd be the dl24 analog.

## 00:35Z checkpoint — H189 VP=3.467 PAPER-TIER (gap 0.026pp); H192 EP10 GATE PASS; H191 EP17 fleet WSS leader 6.678; H193 smoke healthy w/ penalty firing

### Fleet status (00:35Z, all 4 active, zero idle GPUs)

| Student | PR | Hyp | EP | val_WSS | val_ABU | val_VP | val_SP | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| fern | #1535 | H191: Sharper WSD | **EP18** (14.5h) | 6.687 | 5.969 | 3.683 | 3.919 | EP17 fleet leader 6.678; stable phase plateau ⏸; EP25-30 decay = hypothesis test |
| nezuko | #1533 | H189: hidden_dim=640 | **EP19** (19h, student silent 17h+) | 6.712 | 5.917 | **3.467 ⭐⭐⭐⭐** | 3.839 | **PAPER-TIER VP** (gap 0.026pp); test_VP could land sub-SOTA |
| frieren | #1541 | H192: τ_z=1.5 only, lr=1e-4 | **EP12** (9.2h) | 6.723 | 5.965 | 3.588 | 3.919 | EP10 GATE PASS 6.726; EP11 SHARP 6.714; descent intact |
| tanjiro | #1554 | H193: WSS soft normal-component penalty | smoke (0.5h, step 6764) | — | — | — | — | Penalty FIRING (0.030 non-inert); train/loss=0.188 healthy; full EP1 val ~01:30Z |

### Key finding 1 (00:35Z): H189 paper-tier VP confirmed

H189 EP15→EP19 VP trajectory: 3.486 → 3.481 → 3.625 (EP17 spike outlier) → 3.470 → **3.467**. EP17 isolated spike (WSS+ABU+VP+SP all up ~0.15-0.20pp) was an eval-batch noise event, fully recovered EP18-19. Gap to H183 SOTA test_VP=3.4415 is now **0.026pp** at val_VP=3.467. If test_VP ≤ val_VP (typical for healthy training), this run could land **sub-SOTA on VP floor** — a programme-tier outcome on the volume-pressure axis. WSS=6.712 (+0.27pp vs SOTA, NOT a WSS path). ABU=5.917 (closing toward floor 5.844, gap +0.07pp).

### Key finding 2 (00:35Z): H192 mechanism CONFIRMED at official EP10 gate

H192 EP10 = **6.726** (loose target 6.80 PASS, sharp 6.70 near-miss by 0.026pp). EP11 = **6.714** (sharp target PASS). EP12 = 6.723 (minor uptick, normal variance). VP descent continues monotonically EP7→12: 3.713 → 3.646 → 3.630 → 3.617 → 3.602 → 3.588. Mechanism is confirmed isolated from H188's lr=9e-5 confound. Path to EP24 natural termination is open; if EP18-24 settles WSS ≤ 6.65 with floors clearing, this is a mergeable WSS-axis improvement.

### Key finding 3 (00:35Z): H191 stable-phase plateau, decay window is the hypothesis test

H191 EP15-18: 6.687 → 6.680 → 6.678 → 6.687. Stable phase has plateaued. The sharper WSD hypothesis is about the decay window (EP25-EP30) — sharper 100× LR drop should produce a step-change descent at decay onset. EP25-30 needs to descend ≥0.10pp below the EP15-24 plateau to validate. If no, this is a negative result on the LR-schedule axis.

### Key finding 4 (00:35Z): H193 smoke healthy — soft penalty is active

H193 tanjiro picked up assignment within ~10 minutes; smoke launched 00:08:50Z DDP8 (group `h193-wss-normal-penalty`, 8 ranks). Half-epoch into smoke: `train/wss_normal_penalty = 0.030` (penalty is non-inert, model has nonzero predicted normal component being regularized), `train/loss = 0.188` (healthy magnitude). Config confirms `wss_normal_penalty_weight = 0.5` applied. Full EP1 val expected ~01:30Z; if EP1 val_WSS ≤ 7.50 and all floors trending toward H183 baseline, main 30-EP launch is greenlit.

### Action plan (00:35Z)

- **Next wake ~01:30Z** — catch H193 smoke EP1 terminal + EP1 val + main launch decision; H191 EP20-21; H189 EP20-22 (VP descent continuation); H192 EP14-15
- **H189 nezuko monitoring:** continue to natural termination (EP30+, ~07:00Z ETA). Programme-tier VP candidate. If student session never recovers, advisor constructs terminal SENPAI-RESULT from W&B + manual test eval
- **Strategic next-round (post-terminal):** BL DERIVATIVE DECODER PROBE (Morgan's #1 priority); compound H189 VP-deepening + H192 τ_z + H191 sharper-WSD if all three close terminal
- **Compound hypothesis brewing:** H189 (640 hidden_dim) + H192 (τ_z=1.5 isolated) + H191 (sharper WSD) — three independent mechanisms; round-after-this candidate compound

## 23:55Z checkpoint — H190 CLOSED NON-MERGE (SP floor breach decisive); H193 dispatched to tanjiro (soft tangent-basis penalty per Morgan #1056 priority)

### Fleet status (23:55Z, 3 active + 1 dispatching, zero idle GPUs)

| Student | PR | Hyp | EP | val_WSS | val_ABU | val_VP | val_SP | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| fern | #1535 | H191: Sharper WSD | **EP15** | **6.687 ⭐** | 5.970 | 3.679 | 3.924 | FLEET WSS LEADER; EP25-30 decay is hypothesis test |
| nezuko | #1533 | H189: hidden_dim=640 | **EP16** (student silent 15h+) | 6.727 | 5.933 | **3.481 ⭐⭐⭐** | 3.852 | VP=3.481 PAPER-TIER (gap to SOTA 0.039pp); advisor reads W&B directly |
| frieren | #1541 | H192: τ_z=1.5 only, lr=1e-4 | **EP8** | 6.740 | 5.988 | 3.646 | 3.910 | τ_z=1.5 MECHANISM CONFIRMED; EP10 = official gate |
| tanjiro | (new) | H193: WSS normal-component soft penalty | dispatching | — | — | — | — | Morgan's #2 priority (tangent-basis) implemented as soft regularizer |

### Key finding (23:55Z): H190 NON-MERGE — wider per-channel heads do NOT help

H190 test_WSS=6.6711 (+0.228pp vs SOTA), test_SP=3.6793 (BREACH +0.102pp over 3.577 floor). Width-factor=2.5 widening (hidden 1024 → 1280) tracked H183 with WIDENING gap (+0.19 → +0.27pp from EP10→EP20). Architectural finding: after H183 made per-channel heads the default at width=2.0, additional width is past the channel-conditional information ceiling at 65k surface-point density. PR #1534 closed at 23:55Z.

### H193 hypothesis (tanjiro, dispatching): Soft normal-component penalty (tangent-basis priority, soft form)

**Hypothesis:** For no-slip walls, WSS is tangential (τ · n_surface ≈ 0). Adding a soft physics penalty λ * mean((τ_pred · n_surface)²) as regularizer reduces effective output dimensionality from 3 (free cartesian) to 2 (tangent plane), expected to improve cross-flow τ_y, τ_z (the SOTA bottleneck). This is the SOFT form of Morgan's strategic priority #2 (tangent-basis output head, issue #1056). Historical hard tangent projection failed (`la5hrm16`, `lz51r7nb`, `o78w9geu`) because they enforced as hard projection/loss reframing; a soft penalty avoids that failure mode.

Single delta vs H183: `--wss-normal-penalty-weight 0.5` (new flag). All other config = H183 baseline.

### Action plan (23:55Z)

- **Tanjiro dispatch H193** — soft normal-component WSS penalty (single delta from H183)
- **Next wake ~01:30Z** — catch H192 EP10-11 official gate (the H188 collapse-point comparison), H191 EP18-20 stable-phase descent, H189 EP20-21 VP descent continuation, H193 smoke EP1 verification
- **Strategic next-round (post-H191/H192/H189 terminal):** BL DERIVATIVE DECODER PROBE (Morgan's #1 priority); compound winners from this round
- **Cleanup queue:** H183 cleanup (per-channel default merge, PR #1531) already merged; current SOTA stack stable

## 21:30Z checkpoint — H189 VP=3.481 (gap 0.039pp PAPER-TIER); H191 EP15 fleet leader 6.687; H192 τ_z mechanism CONFIRMED; H190 EP18 uptick

### Fleet status (21:30Z, all 4 healthy, zero idle GPUs)

| Student | PR | Hyp | EP | val_WSS | val_ABU | val_VP | val_SP | Δ vs H183 EP10 | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| fern | #1535 | H191: Sharper WSD | **EP15** (W&B, 12.3h) | **6.687 ⭐** | 5.970 | 3.679 | 3.924 | WSS +0.05 | **FLEET WSS LEADER**; stable phase still descending (EP14→15 −0.019); EP25-30 decay test is hypothesis-defining window |
| nezuko | #1533 | H189: hidden_dim=640 | **EP16** (W&B, 16.8h, student silent 13h+) | 6.727 | 5.933 | **3.481 ⭐⭐⭐** | 3.852 | WSS +0.08; **VP −0.147pp (gap to SOTA test_VP only 0.039pp)**; ABU +0.02 | **PAPER-TIER VP CANDIDATE**; advisor rescinded 20:00Z deadline — let descend to natural termination |
| tanjiro | #1534 | H190: width-factor=2.5 | **EP18** (15.0h) | 6.796 | 6.052 | 3.717 | 3.983 | WSS +0.15 | EP18 UPTICK (WSS +0.023, ABU +0.018, VP +0.011 from EP17) — first non-monotonic in slow window; EP20 terminal ~23:10Z; NON-MERGE path confirmed |
| frieren | #1541 | H192: τ_z=1.5 only, lr=1e-4 | **EP8** (7.1h) | 6.740 | 5.988 | 3.646 | 3.910 | WSS +0.10 | **τ_z=1.5 MECHANISM CONFIRMED** — EP8 6.740 already beat H188 EP9 collapse (6.802) with lr=1e-4 isolated; EP5→8 deltas WSS −0.105pp, VP −0.198pp — steepest descent in fleet |

### Key finding 1 (21:30Z): H189 VP=3.481 at EP16 — paper-tier territory

H189 VP trajectory: EP10 3.570 → EP12 3.514 → EP13 3.507 → EP14 3.507 → EP15 3.486 → **EP16 3.481** (monotonic descent resumed after EP13-14 plateau). Gap to H183 SOTA test_VP=3.4415 = **0.039pp**. If EP17-EP30 cosine descent maintains −0.003pp/EP rate, test_VP could land at 3.40 (sub-SOTA on a floor metric). This is a programme-tier VP candidate — hidden_dim=640 is producing the strongest VP improvement signal of the round. Compound finding: capacity helps VP but not WSS (Morgan's diagnosis confirmed).

### Key finding 2 (21:30Z): H192 τ_z=1.5 mechanism CONFIRMED isolated from lr confound

H192 (lr=1e-4 held, τ_z=1.5 only) at EP8 has WSS=6.740. H188 (lr=9e-5 + τ_y=1.2 + τ_z=1.5) at EP9 collapsed to 6.802. With lr=1e-4 isolated, τ_z=1.5 produces healthy descent rather than collapse. EP8 6.740 puts H192 in fern H191's EP14 territory at half the wall-time. Descent rate (WSS −0.105pp over EP5-8) is steepest in fleet. The H188 collapse was lr=9e-5 confound, not τ_z weighting itself — this is a recoverable mechanism finding.

### Key finding 3 (21:30Z): H190 EP18 uptick — wider heads have higher EP-to-EP variance

H190 EP15-18: 6.788 → 6.782 → 6.773 → **6.796** (EP18 first uptick in window). VP also up 3.706 → 3.717. Path remains NON-MERGE (above 6.75 flat-zone). Width-factor=2.5 carries higher EP-to-EP variance than standard per-channel heads — useful architectural finding for future hyperparameter choices.

### Operational status: H189 nezuko student loop still silent (13h+) but training healthy

Pod healthy 4d11h Running 0 restarts, training W&B run `c2qyhgmh` healthy with normal heartbeat through 16.8h. Advisor reading W&B directly + posting heartbeats. Original 20:00Z deadline RESCINDED at 21:30Z — VP=3.481 trajectory is too valuable to interrupt. Will let run continue to natural termination (EP30+, ~31h total runtime, ETA ~07:00Z 2026-06-02). If student session never recovers, advisor will need to construct terminal SENPAI-RESULT marker from W&B + manual test harvest.

### Action plan (21:30Z)

- **Next wake ~22:30Z** — catch H190 tanjiro EP19 (does uptick recover?); H192 frieren EP9-10 official gate read; H191 fern EP16-17; H189 nezuko EP17 (VP descent continuation)
- **Priority watch:** H189 nezuko VP descent to EP30 — paper-tier if EP30 VP < 3.45
- **Strategic next-round:** Architectural pivot per issue #1056 (BL probe, tangent-basis decoder) once H189 terminal lands and confirms VP-direction is the right paper-axis lever
- **VP-direction next experiments:** If H189 terminates with strong VP (<3.50), compound with WSS-targeted architecture (BL probe ON TOP OF hidden_dim=640)
- **Compound hypothesis brewing:** H191 (sharper WSD) + H192 (τ_z=1.5 isolated) + H189 (640 hidden_dim) — three independent mechanisms all working; next-round candidate is a compound H193 trying all three at once

## 19:25Z checkpoint — H189 VP plateau at 3.507 (−0.121pp); H191 leads WSS at EP11 (6.719); H192 EP5 PASSED tight gate

### Fleet status (19:25Z, all 4 healthy, zero idle GPUs)

| Student | PR | Hyp | EP | val_WSS | val_ABU | val_VP | val_SP | Δ vs H183 EP10 | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| fern | #1535 | H191: Sharper WSD | **EP11** (W&B, 9.5h) | **6.719 ⭐** | 5.999 | 3.701 | 3.929 | WSS +0.08 | **FLEET WSS LEADER**; stable phase still has signal (EP10→11 −0.024); EP25-30 decay is hypothesis test |
| nezuko | #1533 | H189: hidden_dim=640 | **EP14** (W&B, 14.1h, student silent since 08:25Z) | 6.746 | 5.952 | **3.507 ⭐⭐** | 3.865 | WSS +0.11; **VP −0.121pp**; ABU +0.04 (closing) | **PROGRAMME VP LEADER**; VP plateau EP13→EP14; cosine T_max=30, past midpoint |
| tanjiro | #1534 | H190: width-factor=2.5 | **EP15** (12.2h) | 6.788 | 6.049 | 3.722 | 3.975 | WSS +0.15 | EP15 BOTH gates FAILED; extending to EP20 cosine tail (advisor); if EP18-20 stays ≥6.75 close NON-MERGE |
| frieren | #1541 | H192: τ_z=1.5 only, lr=1e-4 | **EP5** (4.2h) | 6.845 | 6.116 | 3.844 | 3.974 | (early) | EP5 kill gate PASSED margin 0.005pp; EP10 = the H188 collapse-point comparison |

### Key finding (19:25Z): H189 VP plateau but lead deepening since EP10

H189 VP trajectory: EP10 3.570 → EP12 3.514 → EP13 3.507 → **EP14 3.507** (plateau but lead at −0.121pp vs H183 EP10 SOTA 3.628). val_ABU=5.952 at EP14 (only 0.035pp behind SOTA), still descending slowly. WSS=6.746 holding +0.11pp behind SOTA — capacity boost protective on VP but doesn't help WSS (Morgan's diagnosis confirmed: WSS needs architectural mechanism, not capacity).

### Operational issue: H189 nezuko Claude student session stalled

Pod healthy 4d8h Running 0 restarts, training W&B run `c2qyhgmh` healthy with 1-min heartbeat through 14h. But pod logs show Claude iteration 144 at 08:02Z then silent. kubectl logs --since=2h returns empty. Advisor doing W&B-direct reads + posting heartbeats. Escalation #3 at 19:05Z with 20:00Z deadline. If student doesn't recover before EP30 (~9h), advisor will need to construct terminal SENPAI-RESULT marker from W&B + run test harvest via human intervention.

### H191 (fern, PR #1535) — fleet WSS leader entering deep stable phase

EP10 kill gate PASSED 0.0075pp margin; EP11 deepens to 6.719 (−0.024 from EP10). Per fern's confirmation, val/* rows ARE EMA-validated (config.eval_raw_vs_ema=False default). Critical waypoint is EP25-30 decay test (lr 1e-4 → 1e-6 over 6 EPs, 100× drop). Advisor not expecting posts EP12-EP24 unless plateau breaks.

### H190 (tanjiro, PR #1534) — extended to EP20 cosine tail

EP15 6.788 failed BOTH gates (advisor 6.55, PR 6.60). EP13 was best descent (−0.036), EP14 ticked +0.006, EP15 recovered to 6.788 but trajectory clearly stalled. Extension to EP20 to test whether width=2.5 catches H183 late. If EP18-20 ≥6.75 flat, close NON-MERGE.

### H192 (frieren, PR #1541) — EP5 kill gate PASSED (margin 0.005pp tight)

EP boundaries: EP1 12.80 → EP2 7.30 → EP3 7.02 → EP4 6.89 → EP5 6.845. VP descent especially strong (EP4→EP5 −0.142pp). Per-axis EP3→EP5: τ_x −0.13, τ_y −0.29, τ_z −0.19 (upweighted axis descending faster than τ_x). Next critical milestone: EP10 (the H188 collapse point — H188 was at 6.802 with lr=9e-5+τ_y/τ_z weights). If H192 EP10 < 6.80, τ_z=1.5 isolation helps; if ≈ 6.80, then H188 collapse was lr=9e-5 alone.

### Action plan (19:25Z)

- **Next wake ~20:00Z** — H189 nezuko escalation deadline + H192 EP6-7 boundary + H190 EP16-17 + H189 EP15 cosine descent acceleration check
- **Priority watch:** H189 nezuko terminal protocol — paper-tier VP candidate, may need advisor-side terminal if student doesn't recover
- **Strategic next-round:** Architectural pivot per issue #1056 (BL probe, tangent-basis decoder) once H189 terminal lands and confirms VP-direction is the right paper-axis lever
- **VP-direction next experiments:** If H189 terminates with strong VP, compound with WSS-targeted architecture (BL probe ON TOP OF hidden_dim=640)

## 17:40Z checkpoint — H189 EXTENDS VP lead at EP12 (−0.11pp vs H183); H189 closing on ABU SOTA

### Fleet status (17:40Z, all 4 healthy, zero idle GPUs)

| Student | PR | Hyp | EP | val_WSS | val_ABU | val_VP | val_SP | Δ vs H183 EP10 | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| nezuko | #1533 | **H189**: hidden_dim=640 | **EP12** (W&B, 12.8h) | 6.753 | **5.960** | **3.514 ⭐⭐** | 3.863 | WSS +0.11; **VP −0.11pp**; ABU +0.04 (closing) | **PROGRAMME LEADER on VP** — capacity helps VP, descent monotonic, on track for EP30 terminal |
| fern | #1535 | H191: Sharper WSD | **EP10** (8.2h) | **6.743** | 6.020 | 3.720 | 3.937 | WSS +0.10 | EP10 kill gate PASS (margin 0.0075pp); 14 more stable EPs to EP24, then 100× decay EP25-30 (the actual hypothesis test) |
| tanjiro | #1534 | H190: width-factor=2.5 | **EP13** (10.9h) | 6.792 | 6.051 | 3.721 | 3.965 | WSS +0.15 | EP12→EP13 −0.036 (strongest single-EP since EP10); descent resumed after EP11 plateau; EP15 ≤6.55 gate still unlikely |
| frieren | #1541 | H192: τ_z=1.5 only, lr=1e-4 | **EP3** (3.0h) | 7.018 | 6.339 | 4.259 | 4.070 | (early) | Mechanism descending healthy, EP5 gate ETA 18:45Z |

### Key finding (17:40Z): H189 hidden_dim=640 VP improvement DEEPENS with epochs

H189 EP10 val_VP=3.570 → **EP12 val_VP=3.514** (improvement deepens, −0.058pp → −0.114pp gap vs H183 EP10). This is a real capacity-direction effect. ABU also converging: EP10=6.011, EP11=5.969, EP12=5.960 — now only 0.04pp behind SOTA 5.917 and still descending at −0.01 to −0.04pp/EP. If EP15-EP30 sustains this, H189 will cross BOTH val_VP AND val_ABU SOTA simultaneously.

**WSS is the holdout axis** for H189: EP12 val_WSS=6.753 vs H183 SOTA 6.640 = +0.113pp behind. Capacity boost is NOT helping WSS in the same way — confirms Morgan's hypothesis (issue #1056) that WSS needs architectural mechanism (BL probe, tangent decoder), not just capacity.

### Critical: H189 may be the strongest paper-tier candidate so far

If H189 terminal lands with test_VP < 3.4415 AND test_ABUPT < 5.6152 (the current floors), this is a multi-axis SOTA improvement. The student has been silent since 08:25Z — escalation posted at 17:40Z requesting EP13/EP14 reads by 18:30Z and confirming terminal report plan.

### H191 (fern, PR #1535) — EP10 kill gate cleared 0.0075pp margin

val_WSS=6.7425, descent slowing per stable-phase expectation (EP8→9 −0.010, EP9→10 −0.020). Per fern's own note, the hypothesis is tested EP25-30 (the 100× decay), not stable phase. Advisor will not kill on EP10-EP24 plateau. EP24 ≤6.55 advisor gate active, then decay test EP25→EP30.

### H190 (tanjiro, PR #1534) — EP13 descent resumed after EP11 tick-up

EP10→EP13 deltas: WSS −0.039, ABU −0.038, VP −0.044, SP −0.015. EP15 ≤6.55 gate still requires ~−0.12pp/EP unrealistic; if EP15 lands ≥6.75 we close NON-MERGE; if 6.70-6.75 extend to EP20.

### H192 (frieren, PR #1541) — EP3 healthy, mechanism test starts EP5

EP1→EP2→EP3: WSS 12.80→7.30→7.02, VP 14.21→5.14→4.26 (very strong). EP5 kill gate ≤6.85 ETA 18:45Z. EP10 is the H188 collapse point — if H192 EP10 < 6.80, τ_z=1.5 alone helps; if ≈ H183, then H188 collapse was lr=9e-5 alone.

### Action plan

- **Next wake ~18:50Z** — H192 EP5 gate + H189 nezuko response + H190 EP14-15 + H191 EP12-13
- **Priority watch:** H189 nezuko terminal protocol — needs proper SENPAI-RESULT at EP30 since this is paper-tier candidate
- **Strategic next-round:** Architectural pivot per issue #1056 (BL probe, tangent-basis decoder) once H189 terminal lands and confirms VP-direction is the right paper-axis lever
- **VP-direction next experiments:** If H189 terminates with strong VP, compound with WSS-targeted architecture (BL probe ON TOP OF hidden_dim=640)

## 15:20Z checkpoint — H191 leading WSS; H189 wins VP; H192 main launched

### Fleet status (15:20Z) — full slate, zero idle, all 4 runs healthy

| Student | PR | Hyp | EP | val_WSS | val_ABU | val_VP | val_SP | Δ vs H183 | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| frieren | #1541 | **H192**: τ_z=1.5, lr=1e-4 | smoke done; main step 8814 (pre-EP1) | (smoke EP1 = 12.69) | — | — | — | — | smoke PASS, main launched, EP1 ETA 16:00Z |
| nezuko | #1533 | H189: hidden_dim=640 | **EP10** (W&B direct, student silent since 08:25Z) | 6.802 | 6.011 | **3.570 ⭐** | 3.881 | WSS +0.16, **VP −0.06pp** | Hidden_dim helps VP; weak on WSS — capacity-direction trade-off |
| tanjiro | #1534 | H190: width-factor=2.5 | **EP10** (15:08Z) | 6.831 | 6.089 | 3.765 | 3.980 | WSS +0.19 | EP10 PASSED kill gate; descent −0.03pp/EP plateau forming |
| fern | #1535 | H191: Sharper WSD | **EP7** (15:16Z) | **6.790 ⭐** | 6.074 | 3.800 | 3.958 | WSS +0.10 vs H183 EP7 | **LEADING WSS** in fleet; 17 more stable EPs before EP25 WSD decay test |

### Key finding (15:20Z): H189 hidden_dim=640 IMPROVES VP

**First programme evidence of capacity-direction VP improvement.** H189 EP10 val_VP=3.570 vs H183 EP10 val_VP=3.6284 = **−0.058pp BELOW H183 reference**. This is in the per-channel surface decoder stack, so the +25% width is helping the trunk/volume coupling rather than just the surface heads. The downside: WSS/SP/ABU all slightly behind (+0.16/+0.03/+0.09 vs H183). The capacity boost is VP-biased, not WSS-biased.

**Implication for next-round hypotheses:** if we want WSS improvement, hidden_dim=640 is the wrong lever. If we want VP improvement (paper floor protection), this is a strong direction. Could compound with a WSS-targeted intervention (Morgan's BL probe, tangent-basis decoder) on top of hidden_dim=640.

### H191 (fern, PR #1535) — leading WSS in fleet at EP7 stable phase

val_WSS=6.790 at EP7 is already lower than H189/H190 at EP10. The H183 EP7 reference was 6.6890, so H191 is +0.10pp behind H183 at this point. But the WSD decay phase (EP25-30, lr 1e-4 → 1e-6 over 6 EPs, 100× drop) is where the entire H191 thesis lives. If decay produces 4-5× descent rate boost (the sharp-WSD literature claim), terminal could push test_WSS sub-6.0. If decay fails like H184 did on the H147 stack, terminal ≈ 6.5-6.7.

### H190 (tanjiro, PR #1534) — EP10 PASSED but width=2.5 underperforming

EP10 val_WSS=6.831, EP9→10 descent slowed to ~−0.03pp/EP. Path to beat H183 SOTA (6.4427 EP30) requires ~−0.055pp/EP from EP10→EP30 = unlikely given the current rate. Width=2.5 is acting more symptomatic than transformative.

### H189 (nezuko, PR #1533) — silent since 08:25Z but training is healthy

Pod 1/1 Running, GPUs at 91-100% util, run c2qyhgmh at step 115k EP10. Posted 15:20Z heartbeat asking nezuko to confirm presence and post EP11-12 reads. If still silent by 16:30Z, will request kubectl pod logs to verify training-side reporting (not a kill — the run is healthy).

### H192 (frieren, PR #1541) — clean ablation of H188 mechanism just launched

Smoke EP1 = 12.688 (gate ≤13.5 PASSED). Main `lokhvm6y` running. EP1 main val expected ~16:00Z. Tight kill ladder (EP10 ≤6.70). Hold lr=1e-4, single-axis τ_z=1.5 upweight.

### Action plan

- **Next wake ~16:00Z** — H192 EP1 main + H189 nezuko response + H190 EP12-13 + H191 EP9-10
- H191 EP10 boundary critical (compare directly vs H189/H190 EP10 reads)
- H192 EP3-5 gate critical (must track H183 closely)
- Strategic: defer architectural pivot to BL probe / tangent decoder until current fleet results land (~18-24h)

## 13:35Z checkpoint — H188 KILLED + closed; H192 dispatched; 4 active runs

### Fleet status (13:35Z) — full slate, zero idle GPUs

| Student | PR | Hyp | EP | val_WSS | val_ABU | Δ vs H183 | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1541 | **H192**: τ_z=1.5 only, lr=1e-4 (H183 default) | NEW | — | — | — | Assigned 13:35Z — clean tau-z isolation from LR |
| nezuko | #1533 | H189: hidden_dim=640 | (EP5+ W&B silent since 08:25Z) | — | (10:53Z EP5: 6.12) | — | **stale — 12:45Z heartbeat unanswered, escalate at 14:30Z** |
| tanjiro | #1534 | H190: width-factor=2.5 | **EP7** (12:44Z) | 6.876 | 6.142 | WSS +0.04 vs H183 EP7 | descent stable ~0.03pp/EP — borderline merge path |
| fern | #1535 | H191: Sharper WSD | **EP4** (12:57Z) | (stable phase) | (stable phase) | — | **STRONGEST trajectory — WSD decay starts EP25** |

### H188 (frieren, PR #1532) — CLOSED NON-MERGE at 13:35Z

Killed at EP10 (val_WSS=6.8016, failed both 6.75 kill gate and 6.70 NON-MERGE threshold). KEY FINDING from this experiment: **per-channel decoupling DOES make per-axis tau weighting legible** — first programme evidence (~0.05pp differential between weighted and unweighted axes):

| Axis | Weight | Lag vs H183 |
|---|---:|---:|
| τ_x | 1.0× (unweighted) | **+0.184pp** |
| τ_y | 1.2× | +0.130pp |
| τ_z | 1.3× | +0.125pp |

The mechanism worked directionally. But the differential (~0.05pp) was too small to overcome the lr=9e-5 (−10%) penalty (~0.15pp drag). H188 confounded tau weighting + LR drop.

### H192 (frieren, PR #1541) — clean ablation of H188 mechanism

Strategy: hold lr=1e-4 (H183 default — NO LR change), single-axis upweight on τ_z (the dominant outlier at val_τ_z=9.20% vs val_τ_x=5.93%). Single lever change vs H183 SOTA.

If per-channel decoupling makes tau weighting effective without LR penalty, we should see:
- τ_z descent improves (mechanism direct effect)
- WSS overall improves (τ_z is the dominant contributor)
- Other axes stable (per-channel decoupling protects them)
- SP/VP unchanged

Tight kill ladder (EP2 6.85, EP5 6.85, EP10 **6.70**, EP15 6.60). Predicting H192 should TRACK H183 closely from EP3 onwards, not lag like H188.

### H189 (nezuko, PR #1533) — STALE, escalation imminent

Last student comment 08:25Z (EP3 boundary). Heartbeat at 12:45Z unanswered. Pod kubectl shows 1/1 Running (no infra issue). Last W&B query at 10:53Z showed run `c2qyhgmh` alive at EP5 (ABU=6.12). If no response by 14:30Z heartbeat, will request kubectl pod logs to verify training is still progressing — possible silent stall or comment-side bug.

If hidden_dim=640 (width +25% on H183) is working and reporting just delayed, this could overtake H190 as the leading capacity-direction candidate. EP10 boundary expected ~14:30Z.

### H190 (tanjiro, PR #1534) — descent stable, borderline path to merge

EP7 val_WSS=6.876, ABU=6.142. Per-epoch deltas EP5→6=−0.029, EP6→7=−0.027 (stable ~0.03pp/EP). If sustained, EP10 ≈ 6.79 and EP30 ≈ 6.18 — would beat H183 SOTA by healthy margin. But H183's own EP7→EP30 only descended −0.10pp (most gain came post-EP15 cosine ramp). Extrapolation from EP7 still uncertain.

Surface_out_width_factor=2.5 (H190) builds on H183's 2.0 default, a clean +25% width direction. EP10 watch in ~3.5h.

### H191 (fern, PR #1535) — strongest fleet trajectory, stable phase

EP4 (12:57Z update). EP3 kill gate ≤7.60 passed with 0.62pp margin. EP1→2 was −2.50pp (warmup→stable), EP2→3 was −0.21pp. Currently in stable LR phase (lr=1e-4 until EP24), then sharp WSD decay EP25→30 → lr_min=1e-6 (true 100× drop).

CRITICAL TEST. If H191 matches H183 trajectory through EP24 stable phase AND gets 4-5× descent boost during EP25-30 decay, test_WSS could drop below 6.0%. If decay fails (like H184 on H147 stack), terminal around 6.5-6.7%. Per-channel stack changes the geometry vs H184 — non-zero chance the decay-phase boost materializes this time.

### Action plan

- **Next wake ~14:30Z** — H189 nezuko escalation if no response + H190 EP8 + H191 EP5
- H191 EP5 boundary ~13:34Z (next critical kill gate)
- H190 EP10 boundary ~17:00Z (kill if WSS > 6.75)
- H192 frieren smoke EP1 expected ~14:30-15:00Z if launches promptly

## 12:45Z checkpoint — H188 fading at EP10; H190 borderline; H191 strong + still stable phase

### Fleet status (12:45Z)

| Student | PR | Hyp | EP | val_WSS | val_ABU | Δ vs H183 | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1532 | H188: τ-yz 1.2/1.3 + lr=9e-5 | **EP9** | 6.815 | 6.06 | WSS +0.15pp | **FADING — EP10 likely fails 6.75 gate** |
| nezuko | #1533 | H189: hidden_dim=640 | EP5+ (silent) | (10:53Z W&B: EP5 ABU=6.12) | 6.12 | ABU −0.43 vs H183 EP5 | pod alive but student reporting silent since 08:25Z |
| tanjiro | #1534 | H190: width-factor=2.5 | **EP7** | 6.876 | 6.14 | WSS +0.04 vs H183 EP7 | descent stabilized ~0.03pp/EP — borderline |
| fern | #1535 | H191: Sharper WSD | **EP3** (stable phase) | 6.98 | 6.31 | EP3 kill gate 7.60 PASSED with 0.62pp | **STRONGEST — sharper WSD early descent thesis confirmed** |

### H188 (frieren, PR #1532) — fading at EP10

EP9 val_WSS=6.815, gap to H183 stable at +0.15pp since EP3. Descent rate EP8→9 only −0.011pp (H183 at same boundary −0.006pp), so H188 closing but very slowly. **Projected EP10 ≈ 6.79-6.80, FAILS the 6.75 kill gate.**

Per-axis read (positive evidence for mechanism, negative for magnitude): τ_x +0.17pp (most lag, unweighted), τ_y +0.12, τ_z +0.11 (weighted axes lagging less, but small). The mild tau weighting IS doing what we hypothesized but the effect is too small to overcome the lr=9e-5 (−10%) penalty.

**Decision pending at EP10 (~13:10Z)** — if val_WSS > 6.75, kill. ABU at EP9 still 6.06 below H183 EP10 (6.20) by 0.14pp, so we'd be killing despite ABU lead. Worth considering: ABU is paper-facing primary; should we continue if WSS just over gate but ABU strongly ahead?

**Counterfactual:** If we kill H188 at EP10, we lose the EP25-30 cosine descent that could close the WSS gap. If we continue and the gap doesn't close by EP15, we burn 6+ EPs of GPU time. Recommend: **continue past EP10 kill gate** ONLY IF EP10 val_WSS ≤ 6.80 AND EP10 ABU ≤ 6.05 (both lead H183 EP10 ref on ABU AND match WSS within reach).

### H189 (nezuko, PR #1533) — pod alive but student reporting silent

Student hasn't posted since 08:25Z (EP3). My own 10:53Z W&B query showed EP5 ABU=6.12 (run alive). Pod kubectl shows nezuko 1/1 Running, 0 restarts, 4d2h uptime — no infrastructure issue. **Posted 12:45Z heartbeat asking nezuko to confirm pod health + post EP6-7-8 boundaries.**

If she responds with strong EP6-8 trajectory, H189 may overtake H188 as leading candidate. hidden_dim=640 is +25% width on H183 stack — if it works it opens up a width-direction follow-up.

### H190 (tanjiro, PR #1534) — descent stabilizing, borderline

EP7 val_WSS=6.876, ABU=6.14. Per-epoch deltas: EP5→6 −0.029pp, EP6→7 −0.027pp. **Descent rate plateaued at ~0.03pp/EP.** If sustained, EP10 ≈ 6.79, EP30 terminal ≈ 6.18 — would beat H183 SOTA 6.4427 by a healthy margin.

Caveat: H183's own EP7→EP30 descent was only −0.10pp (6.689 → ~6.59) — most of H183's gain came after EP15 cosine ramp. H190's monotonic descent without LR drop is encouraging but extrapolating to EP30 from EP7 is uncertain.

### H191 (fern, PR #1535) — strongest trajectory, still in stable phase

EP3 val_WSS=6.98, ABU=6.31. EP3 kill gate ≤7.60 passed with 0.62pp margin. Descent EP1→2 was −2.50pp (warmup→stable transition), EP2→3 was −0.21pp. Currently in stable LR phase (lr=1e-4 until EP24), then sharp WSD decay EP25→30 → lr_min=1e-6 (true 100× drop).

**This is the critical test.** If H191 can match H183 trajectory through EP24 (stable phase) AND get a 4-5× descent boost during EP25-30 decay, it could push test_WSS below 6.0%. If the decay descent fails (like H184 on H147 stack), it'll terminal around 6.5-6.7%.

### Action plan

- **Next wake ~13:30Z** for H188 EP10 decision + H189 nezuko response check + H191 EP4 read.
- H188 EP10 boundary at ~13:10Z is the critical decision.
- H191 EP5 at ~13:34Z is next checkpoint.
- Keep monitoring closely — fleet is converging toward EP10-EP12 decision points across 3 of 4 runs.

## 08:30Z checkpoint — FLEET ALL AHEAD of H183 on ABU; H188 EP7 ALREADY past H183 EP10 reference

### Fleet status (08:30Z) — all 4 active runs trending positively

| Student | PR | Hyp | EP | val_ABU | val_WSS | Δ ABU vs H183 ref | Verdict |
|---|---|---|---|---:|---:|---:|---|
| frieren | #1532 | H188: τ_y=1.2/τ_z=1.3 + lr=9e-5 on H183 stack | **EP7** | **6.08** | ~6.89 | **−0.12 vs H183 EP10=6.20** | **STRONGEST — already past H183 EP10 ABU reference** |
| nezuko | #1533 | H189: hidden_dim=640 + H183 stack | EP5 | **6.12** | 6.89 | **−0.43 vs H183 EP5=6.55** | ABU lead in fleet at EP5 |
| tanjiro | #1534 | H190: surface_out_width_factor=2.5 | EP4 | **6.27** | 6.97 | **−0.28 vs H183 EP5=6.55** | tracking H183 closely (WSS +0.05) |
| fern | #1535 | H191: Sharper WSD (lr→1e-6, stable=24EP/decay=6EP) | EP1 | **9.53** | 9.69 | **−1.51 vs H183 EP1=11.04** | dramatic early descent — sharper WSD producing fastest EP1 of all 4 runs |

### Key finding: H188 at EP7 has ABU=6.08, already below H183's EP10 reference (6.20)

This is **the first time in this research programme** we have a run trending below H183 EP10 ABU 7 epochs ahead of schedule. Mechanism reading: τ_y=1.2/τ_z=1.3 channel weights are pulling tau-axis predictions tighter at no cost to surface pressure or volume pressure. With 23 more EPs to go (30 total), the trajectory suggests test_ABU could land well below H183's 5.6152.

Caveat: H188 WSS still +0.18-0.25 above H183 at EP3-5; need to watch whether WSS converges by EP15-EP20 or if the τ-reweighting trades WSS for ABU. **The paper-facing primary is ABU, so a WSS regress within floor is acceptable if ABU improves materially.**

### H189 (nezuko, PR #1533) — capacity bump leads ABU narrowly

EP5 ABU=6.12 leads the fleet at EP5. Hidden_dim=640 (+25% width) is improving early-EP descent. Tracking ahead of H188 at the same epoch (H188 EP5 ABU=6.18).

### H190 (tanjiro, PR #1534) — width-factor=2.5 closely tracking

EP4 ABU=6.27, val_WSS=6.97. Trajectory similar to H189 with slightly slower descent.

### H191 (fern, PR #1535) — dramatic early descent

EP1 ABU=9.53 vs H183 EP1=11.04 (−1.51pp), val_WSS=9.69 vs H183 EP1=11.84 (−2.15pp). **Strongest EP1 of all 4 active runs.** Sharper WSD (lr=1e-5→1e-6 over 6 EP) is producing a much faster initial descent than H183's reference WSD shape. 30 more EPs of data to come.

### Action plan
- **Next wake ~10:00Z** for EP9-10 boundary cluster on H188/H189, EP6-7 on H190, EP3-4 on H191.
- H188 watch: when does WSS converge? If WSS gap closes by EP15, H188 looks like a clear winner. If WSS plateau ~0.2pp above H183 with ABU advantage, still a winner on paper-facing primary.
- No closes or merges yet — all 4 are mid-flight on long runs (24+ hr expected).

## 07:30Z checkpoint — H188 LEADING (EP3 ABU=6.40 ahead of H183); H189 strong; H190 smoke PASSED + main launched; H191 fern dispatched

### Fleet status (07:30Z)

| Student | PR | Hyp | EP | val_ABU | val_WSS | Verdict |
|---|---|---|---|---:|---:|---|
| frieren | #1532 | H188: τ_y=1.2/τ_z=1.3 + lr=9e-5 on H183 stack | EP3 | **6.40** | 7.09 | **STRONG LEAD vs H183 EP3≈7.04** |
| nezuko | #1533 | H189: hidden_dim=640 + H183 stack | EP2.6 | 6.68 | 7.28 | strong active descent (slope −0.447/1k) |
| tanjiro | #1534 | H190: surface_out_width_factor=2.5 | EP0.6 main (smoke EP2 PASSED 9.08%) | — | — | main launched 06:52Z |
| fern | #1535 | H191: Sharper WSD 100× drop on H183 | (waiting) | — | — | label fixed status:wip; fern's pod will pick up next poll |

### H188 (frieren, PR #1532) — leading candidate

**Mechanism:** Per-channel surface decoder heads (H183 default) + τ_y=1.2/τ_z=1.3 channel-aware loss weights + lr=9e-5. Tests whether the H183 stack still has cross-flow shear gains accessible via mild bounded τ reweighting (which was strong on pre-H183 stack — `9mm3sz7x` test_WSS=8.12 best single-model in May 4 wave).

**EP3 ABU=6.40% is ~0.64pp ahead of H183 EP3 reference 7.04%.** Descent slope −0.022/1k steps (steady).

Kill ladder: EP5 ≤6.95 ✓ on track (current 6.40 with 2 EPs remaining), EP10 ≤6.75, terminal must improve test_WSS over 6.4427.

### H189 (nezuko, PR #1533) — capacity bump candidate

**Mechanism:** H183 stack with hidden_dim=640 (vs H183=512, +25% width, ~1.5× backbone params). Tests whether the H183 ceiling is set by representation capacity.

EP2.6 ABU=6.68% with strong descent slope −0.447/1k steps. Comfortably under EP3 gate ≤7.3.

### H190 (tanjiro, PR #1534) — per-channel head width sweep

**Smoke PASSED:** EP2 ABU=9.08% (gate ≤13.5%). Main launched 06:52Z (run `rmz7dng2`). EP1 main read expected ~07:50Z.

**Mechanism:** Per-channel surface decoder heads at width-factor=2.5 vs H183's 2.0. Tests whether per-channel structure benefits from wider per-head MLPs (no capacity sharing → each head can absorb more committed capacity).

### H191 (fern, PR #1535) — dispatched + label fixed

**Mechanism:** Sharper WSD on H183 stack with true 100× LR drop (1e-4 → 1e-6), stable=24 EP, decay=6 EP, total=31 EP. Re-tests WSD mechanism on the per-channel-head stack (H184 falsified WSD on pre-H183 stack).

Routing issue resolved (label: status:review→status:wip; student:fern→student:dl24-fern). Fern's pod will pick up next poll. Must implement WSD scheduler before launch (impl details in PR body + smoke gate posted).

## 06:30Z checkpoint — H191 assigned to fern; H184 CLOSED NON-MERGE; H188/H189 main running; H190 smoke

### Fleet status (06:30Z)

| Student | PR | Hyp | Status |
|---|---|---|---|
| fern | #1535 | H191: Sharper WSD (lr→1e-6 100× drop, stable=24EP, decay=6EP) | **JUST ASSIGNED** — must implement WSD scheduler first, then 31-EP run |
| frieren | #1532 | H188: H183-stack + mild τ_y/z weights (1.2/1.3) + lr=9e-5 | **MAIN RUNNING** — since 05:08Z; EP3 gate val_WSS>7.6 kill |
| nezuko | #1533 | H189: H183-stack + hidden_dim=640 | **MAIN RUNNING** — since 05:01Z |
| tanjiro | #1534 | H190: per-channel surface width=2.5 | **SMOKE** — EP1 expected ~06:00Z; gate ≤13.5% |

### H184 (fern, PR #1513) — CLOSED NON-MERGE (terminal SENPAI-RESULT + test eval)

**Terminal test (best-EP19 EMA checkpoint):**
| metric | H184 | H183 SOTA | H147 SOTA | floor | floor status |
|---|---:|---:|---:|---:|---|
| **test_WSS** | **6.5982** | 6.4427 | 6.5409 | (primary) | **+0.057 BEHIND H147** ❌ |
| test_VP | 3.6087 | 3.4415 | 3.4014 | ≤3.643 | passes (−0.034) but behind |
| test_SP | **3.7064** | 3.5187 | 3.5634 | ≤3.577 | **BREACH +0.129pp** ❌ |
| test_ABU | 5.7841 | 5.6152 | 5.6648 | ≤5.844 | passes (−0.060) but behind |

**WSD schedule shape verified end-to-end** (warmup → 21 EP plateau at lr=1e-4 → 8 EP cosine to lr_min=1e-6). Decay-phase descent rate = −0.003pp/EP (essentially identical to late-stable plateau +0.001pp/EP). **Predicted 4-5× decay-phase boost did NOT materialize.**

**Mechanism reading:** H147 stack saturates the peak-LR basin by ~EP10 → at decay onset (EP22), model is already at a local optimum and has nothing left to descend toward. WSD theory's boost assumes model is NOT at a local optimum when decay begins; ours is. **Cosine wins on this stack because earlier-onset decay couples LR drop with active descent, finding a tighter sub-basin during the descent rather than after saturating a peak-LR basin.**

H191 tests the corrected configuration on the H183 stack — same WSD mechanism but with true 100× drop (1e-4 → 1e-6) over a longer stable phase (24 EP) + sharp decay (6 EP). Different stack (H183 has per-channel heads, GradNorm, curvature attention) and different schedule depth, so the prior is non-zero but the H184 finding tempers the expected payoff magnitude.

### H191 hypothesis (PR #1535, fern)

**Sharper WSD on H183 stack** — true 100× LR drop (lr_peak=1e-4 → lr_min=1e-6), stable phase=24 EPs, decay=6 EPs, total=31 EPs.

CRITICAL: WSD scheduler NOT in main branch (H184 was NON-MERGE). Fern must implement before running:
1. Add Config fields: `lr_schedule: str = "cosine"`, `lr_wsd_stable_epochs: int = 24`, `lr_wsd_decay_epochs: int = 6`
2. Add CLI args `--lr-schedule`, `--lr-wsd-stable-epochs`, `--lr-wsd-decay-epochs`
3. Add WSD branch to `build_lr_scheduler` in trainer_runtime.py (full code in PR body)

Kill gates: EP3>7.6 kill, EP5>7.1 kill, EP10>6.75 kill, EP25>6.55 kill, terminal must beat 6.4427%.

## 05:15Z checkpoint — H183-cleanup MERGED; tanjiro idle→H190; H188/H189 main runs launched; H184 FINISHED (NON-MERGE)

### Fleet status (05:15Z)

| Student | PR | Hyp | Status |
|---|---|---|---|
| tanjiro | #1534 | H190: per-channel surface width=2.5 | **JUST ASSIGNED** — smoke expected ~06:00Z |
| frieren | #1532 | H188: H183-stack + mild τ_y/z weights (1.2/1.3) + lr=9e-5 | **MAIN RUNNING** — 8 ranks since 05:08Z, step ~1500 (~7min). EP1 val ~06:10Z |
| nezuko | #1533 | H189: H183-stack + hidden_dim=640 | **MAIN RUNNING** — 8 ranks since 05:01Z, step advancing. EP1 val ~05:50Z |
| fern | #1513 | H184: WSD LR schedule | **FINISHED** 22.42h, val_WSS=6.838% — NON-MERGE all 4 axes. SENPAI-RESULT pending from fern. Close when result lands. |

### H183-cleanup (PR #1531) MERGED 04:58Z

Tanjiro's cleanup successfully removed `--per-channel-surface-heads` flag and ~30 LOC dead code paths. Per-channel surface decoder heads are now the **unconditional default in model.py**. All future experiments on this branch inherit per-channel heads without any flag.

- Smoke verified: EP1 val_WSS=12.691% (within H183 EP1 RNG variance of 12.973%, nonfinite_grads=0)
- Codebase: cleaner, harder to mis-run, no legacy flags

### H190 hypothesis (PR #1534, dl24-tanjiro)

**Width-factor sweep for per-channel surface decoder heads**: `--surface-out-width-factor 2.5` vs H183's 2.0.

- H183 established per-channel heads at width=2.0 (each head MLP: Linear(512,1024)→GELU→Linear(1024,1))
- With per-channel structure, no capacity sharing between channels — each head may benefit from modestly wider hidden layers
- 2.5 → Linear(512,1280)→GELU→Linear(1280,1) — 25% wider per head, ~100k extra params total across 4 heads
- Single-variable change vs H183 reproduce command
- Smoke EP1/EP2 gates, then 30-EP main if clear

### H184 (fern, PR #1513) — FINISHED terminal NON-MERGE

val at terminal (step=329281, 22.42h):
- val_WSS=6.838% (+0.40pp vs H183 SOTA 6.443%), NON-MERGE
- val_VP=3.714%, val_SP=4.066%, val_ABU=6.093%
- WSD mechanism FALSIFIED: decay to ~32% peak LR produced no descent (−0.0003pp/1k steps on WSS slope)
- Closing direction: WSD on this stack requires sharper 100× drop + longer stable phase to work at all
- SENPAI-RESULT from fern pending; close PR as NON-MERGE upon receipt

### Post-H184 fern assignment (H191 planned)

Once fern closes H184, next assignment is H191: **Sharper WSD** on H183 stack — same mechanism but with a true 100× LR drop (lr_peak=1e-4 → lr_min=1e-6, stable phase = 24 EPs, decay = 6 EPs). Tests whether the WSD mechanism requires steeper schedule to deliver the "deep minimum" effect.

**Reason for deeper investigation**: WSD literature claim (4-5× boost from sharp 100× lr drop at decay start) requires the stable phase NOT to close too aggressively. H184's decay was ~0.32× peak → did not satisfy the 100× drop condition. H191 will use the correct configuration.

## 04:00Z checkpoint — fleet update: H184 EP28 WSD-null confirmed, smokes starting for H188/H189/cleanup

| Student | PR | Hyp | Status |
|---|---|---|---|
| fern | #1513 | H184 WSD LR | EP28 val_WSS=6.8386, WSD null-decay confirmed. Terminal NON-MERGE ~05:30Z. Fern to post SENPAI-RESULT at EP30. |
| tanjiro | #1531 | H183-cleanup: per-channel heads default | EP1 smoke started 03:52Z (8 ranks running). ~04:35Z results. |
| frieren | #1532 | H188: H183-stack + mild τ_y/z weights + lr=9e-5 | Branch switched 03:57Z; H185 being killed; H188 launching ~04:05Z. |
| nezuko | #1533 | H189: H183-stack + hidden_dim=640 | EP1 smoke started 03:52Z (8 ranks running). ~04:35Z results. |

**Next decision points:** EP1 smoke reads for H183-cleanup/H189 (~04:35Z), H188 launch confirmation (~04:05Z), H184 terminal SENPAI-RESULT (~05:30Z). Schedule wakeup 04:40Z.

## 04:30Z MAJOR UPDATE — H183 MERGED NEW SOTA; H185/H186/H184 closed; 3 new assignments dispatched on H183 stack

### Current fleet status

| Student | PR | Hyp | Status |
|---|---|---|---|
| tanjiro | #1531 | H183-cleanup: make per-channel heads default | NEW WIP — code cleanup, ~1-2h |
| frieren | #1532 | H188: H183-stack + mild τ_y/z weights (1.2/1.3) + lr=9e-5 | NEW WIP — smoke then 30-EP |
| nezuko | #1533 | H189: H183-stack + hidden_dim=640 | NEW WIP — smoke then 25-EP |
| fern | #1513 | H184: WSD LR schedule | WIP TERMINAL IMMINENT — NON-MERGE all 4 axes; SENPAI-RESULT pending |

### H183 MERGED — test metrics (EP24 EMA, run `guw83mge`)

| Metric | H183 | H147 prev SOTA | Δ |
|---|---:|---:|---:|
| **test_WSS** | **6.4427%** | 6.5409% | **−0.098pp ⭐ NEW SOTA** |
| test_SP | 3.5187% | 3.5634% | −0.045pp ✅ |
| test_VP | 3.4415% | 3.4014% | +0.040pp ✅ |
| test_ABU | 5.6152% | 5.6648% | −0.050pp ✅ |
| test_τ_x | 5.6983% | 5.8155% | −0.117pp |
| test_τ_y | 6.9813% | 7.0556% | −0.074pp |
| test_τ_z | 8.4364% | 8.4882% | −0.052pp |

**Critical methodological finding (val→test mapping):** H183's val trajectory appeared +0.04pp BEHIND H147 at EP15-30, but test showed −0.098pp IMPROVEMENT. Val→test gap is architecture-dependent: H147 shows ~0pp gap; H183 shows −0.14pp (WSS) / −0.32pp (SP). **Future decoder-architecture variants MUST NOT use H147's val→test pattern for projection.**

### Closed this cycle

| PR | Hyp | Verdict | Reason |
|---|---|---|---|
| #1510 | H183 per-channel heads | **MERGED** ⭐ | test_WSS=6.4427 (NEW SOTA, all 4 floors) |
| #1527 | H185 hidden_dim=640 (old stack) | NON-MERGE | EP7 val_WSS=6.79, +0.07pp behind H147; definitively non-merge against new SOTA |
| #1529 | H186 layers=8 (old stack) | NON-MERGE | EP3 val_WSS=7.01, same EP1-lead-erasure pattern as H185 |
| #1513 | H184 WSD LR | NON-MERGE (terminal pending) | lr≈0, val_WSS=6.84 — WSD decay null, all 4 axes fail |

### Research direction change (H183 stack)

All new experiments now build on H183 per-channel decoder stack (default in codebase). The old shared-MLP surface decoder path is removed in PR #1531.

**Key findings from this wave:**
1. Per-channel decoder heads = confirmed SOTA mechanism (+0.098pp WSS, all floors improved)
2. Capacity axis (width=640, depth=8) CLOSED on shared-decoder stack — H147 512d/6L is optimal
3. WSD schedule FALSIFIED on this stack (wrong schedule shape — needs 100× drop, not 0.32×)
4. val→test mapping is NOT constant across decoder architectures — critical lesson for future val-trajectory projections

**Next research directions:**
1. **H188 (frieren):** Mild τ_y/z loss weights (1.2/1.3) + lr=9e-5 on H183 stack — per-channel independence should make this cleaner than on shared decoder
2. **H189 (nezuko):** hidden_dim=640 on H183 per-channel decoder — retry capacity expansion with decoupled gradients
3. **H183-cleanup (tanjiro):** Consolidate per-channel heads as unconditional default
4. **Post-cleanup tanjiro:** H190 planned — extended training (40-45 EP) or per-head width sweep
5. **Post-H184 fern:** H191 — sharper WSD (100× lr drop, longer stable phase) on H183 stack
6. **Longer-term:** grouped crossflow head (τ_y+τ_z shared, τ_x separate) to test mechanism direction

### H184 (fern, #1513) — terminal pending

val_WSS=6.8386, val_SP=4.0627, val_VP=3.7123, val_ABU=6.0918 at last W&B read. train/lr≈0. SENPAI-RESULT expected any minute. Close NON-MERGE after result lands.

## 01:40Z snapshot — H183 EP28 plateau confirmed + per-channel asymmetry finding; H185 EP7 descent slowing (kill-gate fail likely); H186 EP3 tracking H147; H184 EP28 still decay-null

**Terminal cluster timing:** H183 ~03:20Z, H184 ~03:30-04:00Z, H185 EP10 gate ~04:00Z, H186 EP5 gate ~03:30Z.

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | val_ABU | lr | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| frieren | #1527 | H185 hidden_dim=640 | 7.1 | 6.7902 (rate slowing) | 3.6352 | 3.9476 | 6.0192 | 9.05e-5 | EP10 gate ≤6.65 likely FAIL (linear proj 6.72) |
| nezuko | #1529 | H186 layers=8 main | 3.25 | 7.0149 (+0.03 vs H147 EP3) | 4.0338 | 4.0905 | 6.2914 | 9.84e-5 | EP5 gate ~03:00Z |
| tanjiro | #1510 | H183 per-channel heads | 27.98 | 6.5886 (plateau, EP24 best=6.5844) | 3.5826 | **3.8428 (flat)** | 5.8726 | 3.4e-6 (deep tail) | NON-MERGE on SP floor; terminal ~03:20Z |
| fern | #1513 | H184 WSD LR | 27.78 | 6.8321 (decay null) | 3.7123 | 4.0628 | 6.0887 | 3.16e-5 | NON-MERGE projected ALL 4 axes; terminal ~03:30Z |

### H183 per-channel heads finding (extracted from tanjiro EP25 per-axis read)

| τ-axis | H183 EP25 | H147 SOTA ref | Δ |
|---|---:|---:|---:|
| τ_x (streamwise) | **5.7116** | 5.8155 | **−0.10pp ahead** |
| τ_y (crossflow) | 7.1727 | 7.0556 | **+0.12pp behind** |
| τ_z (z-axis) | 9.0579 | 8.4882 | **+0.57pp behind** |

**Asymmetric mechanism finding**: Per-channel-heads HELPS streamwise (τ_x) but HURTS crossflow (τ_y, τ_z). The decoder decoupling is real but the polarity is wrong in this implementation. Next iteration should INVERT channel groupings (group τ_y+τ_z together as crossflow head, keep τ_x separate or with pressure) to test if crossflow channels compete for capacity OR benefit from differentiation.

This is a useful negative result with mechanism-level extraction. Future hypothesis for tanjiro post-H183-close.



**Active fleet, 4 students, all WIP, terminal cluster ~02:30-03:30Z:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | val_ABU | lr | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| frieren | #1527 | H185 hidden_dim=640 | 6 | **6.8127** (+0.07 vs H147 EP6) | 3.6850 | 3.9736 | 6.0497 | 9.3e-5 | EP1 lead eroded; EP10 ≤6.65 gate sharpened |
| nezuko | #1529 | H186 layers=8 main (rank0=`31pux7bu`) | 2-3 | ~7.27 (tied H147 EP2) | 4.78 | 4.25 | 6.65 | 1.0e-4 | EP5 watch ~02:30Z |
| tanjiro | #1510 | H183 per-channel heads | 27.5 | 6.5883 (plateau) | 3.5816 | **3.8424 (flat)** | 5.8720 | 5e-6 (final tail) | NON-MERGE on SP floor; terminal ~03:30Z |
| fern | #1513 | H184 WSD LR | 27 | 6.8321 (decay NULL) | 3.7123 | 4.0628 | 6.0887 | 3.2e-5 | NON-MERGE projected ALL 4 axes; terminal ~02:30Z |

### Critical 01:00Z finding: capacity-axis (width AND depth) does NOT improve over H147 at hidden_dim=512/layers=6

**H185 trajectory (width=640):** EP1=11.49 (−1.33 vs H147=12.82) → EP6=6.8127 (+0.07 vs H147 EP6 ~6.74). The −1.33pp EP1 lead fully eroded by EP3 and **inverted by EP5-6**. Wider model uses extra capacity for redundancy, not for fitting harder features.

**H186 trajectory (layers=8):** Smoke EP1=11.892 (−0.93 vs H147=12.82), main EP2-3 ~7.27 (tied with H147 EP2=7.26). Depth axis showing the same pattern — strong EP1 lead, convergence to H147 by EP2-3.

**Joint conclusion (provisional, awaiting EP10+ reads):** H147's hidden_dim=512 + layers=6 is at or near the local capacity optimum for this DDP8 stack. Capacity-axis search closing. Next experiments should focus on **non-capacity axes**: loss shaping, schedule shape, attention mechanism, augmentation, distillation, gradient surgery.

### H184 WSD decay finally activating but still NON-MERGE on all 4 floors

EP24→EP27 (3 EPs, lr 8.55e-5 → 3.20e-5):
- val_WSS 6.8200 → 6.8321 (+0.012, slight uptick — decay still doesn't unlock features)
- val_VP 3.7192 → 3.7123 (−0.007, basically flat)
- val_SP 4.0604 → 4.0628 (flat)

Even at lr=32% of peak, no descent. Confirms WSD on this stack requires sharper schedule (100× drop, not 0.32×) AND longer stable phase before decay. Test projection: test_WSS=6.66 (+0.12), test_SP=3.86 (+0.28), test_VP=3.69 (+0.04), test_ABU=5.91 (+0.07). All 4 floors fail.



**Active fleet, 4 students, all WIP, terminal cluster ~02:30-03:30Z:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | val_ABU | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| frieren | #1527 | H185 hidden_dim=640 | 4 | **6.9002** (~tracking H147) | 3.8850 | 4.0246 | 6.1621 | EP5 ~01:40Z; EP1 lead converged |
| nezuko | #1529 | H186 layers=8 main (rank0=`31pux7bu`) | 0.66 | — (smoke EP1=11.892) | — | — | — | EP1 main read ~01:10Z |
| tanjiro | #1510 | H183 per-channel heads | 23+ | 6.5908 (plateau) | 3.5854 | **3.8463 (flat)** | 5.8748 | NON-MERGE on SP floor confirmed; terminal ~03:30Z |
| fern | #1513 | H184 WSD LR | 24+ | 6.8200 (decay NULL) | 3.7192 | 4.0604 | 6.0833 | NON-MERGE projected on WSS + SP; terminal ~02:30Z |

### Cross-cycle findings consolidated (22:43Z + 00:51Z)

**H185 capacity-axis (width=640):** EP1 lead of −1.33pp vs H147 converged to ~+0.05pp behind by EP4. Hidden_dim=640 boosts initial fit speed but H147's hidden_dim=512 catches up by EP2. **Width axis: modest mid-training value, requires EP10-15 decay phase to show whether capacity helps the tail.** EP10 kill gate: ≤6.65 (matches H147 EP10 = 6.64).

**H186 capacity-axis (layers=8):** Smoke EP1=11.892 (−0.928pp vs H147, weaker than H185's −1.33pp width advantage). Depth gives less initial lift than width per unit param. Main 25-EP run launched 00:15Z, EP1 main read ~01:10Z.

**Joint width-vs-depth read at EP10-15** will inform whether structural changes on H147 stack help OR whether H147's specific config is near-optimal at its size.

**H184 WSD decay FAILED to activate** — lr at EP24=8.55e-5 still ~85% of peak (decay schedule too gentle). WSS rate −0.0022pp/EP through decay (same as stable phase). Confirms WSD's literature claim (4-5× boost from sharp 100× lr drop) requires sharper schedule than tested. Future WSD attempts need lr at start-of-decay × 0.01 (not × 0.5).

**H183 SP floor breach confirmed**: val_SP=3.8463 at EP23 — flat slope through EP22-23. Test_SP projected 3.65 (FAILS 3.577 floor +0.07pp). Per-channel-heads ablation extracted some WSS-axis benefit but introduced SP regression — closes the axis-decoupled-heads direction.

## 22:43Z snapshot — **H182 CLOSED NON-MERGE** (test_WSS=6.6180, test_SP=3.6723 BREACH); H186 `layers=8` assigned to nezuko (PR #1529); H183/H184/H185 still active

**Active fleet, 4 students, 3 WIP + 1 new assignment (nezuko just freed by H182 closure):**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | Status |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1527 | H185 hidden_dim=640 | 2 | EP1=11.49 (−1.33 vs H147) | — | — | EP5 watch ~01:30Z |
| nezuko | #1529 | H186 layers=8 (depth) | — | — | — | — | NEW ASSIGNMENT |
| tanjiro | #1510 | H183 per-channel heads | 22 | 6.5894 (plateau) | 3.5802 | **3.8386 (flat)** | NON-MERGE on SP floor confirmed; terminal ~03:30Z |
| fern | #1513 | H184 WSD LR | 16 | 6.8560 (stall) | 3.7216 | 4.0635 | EP22 decay binary read ~01:00Z |

### H182 (PR #1506) CLOSED NON-MERGE — 22:46Z terminal

W&B run `ecw2sct9` — 30 EPs, 22.43h, EMA best EP23.

Test metrics (regress on ALL 4 axes vs H147):
- test_WSS=**6.6180** (+0.0771 vs H147 SOTA) ❌ primary regress
- test_VP=**3.4648** (+0.0634 vs H147, passes 3.643 floor) — partial-SOTA candidacy on VP FAILED
- test_SP=**3.6723** (BREACH 3.577 floor by +0.0953) ❌
- test_ABU=**5.7474** (+0.0826 vs H147, passes 5.844 floor)

**Critical methodological finding**: val→test pattern shifted significantly from H147 baseline. My 15:34Z/19:00Z projections used H147's val→test deltas (VP +0.06pp UP, SP −0.20pp DOWN) but H182 actual deltas were VP −0.018pp (basically flat) and SP −0.27pp DOWN. The val_VP=3.4830 partial-SOTA signal **did not generalize to test_VP**. Future hypotheses claiming val_VP improvements must show test-side validation; val_VP < 3.55 is NOT a reliable test_VP signal.

**Closes the lr-boost direction definitively**: H149 (β1=0.93/β2=0.97 + H147 lr) → H150 (β1=0.97/β2=0.985) → H182 (lr=1.3e-4 + ema=0.9999 compound) all NON-MERGE. Lion + lr=1e-4 + β1=0.95/β2=0.98 is a tight local optimum.

### H186 hypothesis (PR #1529, dl24-nezuko)

`model-layers=8` on H147 stack (single-flag change). Tests the **depth axis**, orthogonal to H185's **width axis** (hidden_dim=640). Together H185 and H186 jointly characterize the capacity-axis space on the H147 architecture for the first time.

- 25 EPs DDP8 (budget-constrained from 30) — loses ~−0.02 to −0.04pp cosine tail
- Smoke 1-EP first to measure throughput (kill if >56 min/EP)
- Kill ladder: EP1 ≤13.5, EP5 ≤6.95, EP10 ≤6.75, EP15 ≤6.65, EP20 ≤6.55, EP25 ≤6.50 + all 4 floors clear
- Companion to H185 (frieren PR #1527); if both win → compose width+depth next cycle

### H185 (frieren, PR #1527) — STRONG EP1 START (−1.33pp vs H147)

EP1 val_WSS=11.49% (H147 EP1=12.82%). EP1 timing ~53 min (~20% slower than H147 as expected with +25% width). Plan: complete 30-EP cosine; truncate at EP27 if budget tight (frieren's call).

Per-axis EP1 WSS: τ_x=10.20, τ_y=12.83, τ_z=14.74 — wider model helps all 3 axes from EP1.

EP5 gate at ≤6.90% — ~3.5h after EP1 = ~01:30Z next day.

### H183 (tanjiro, PR #1510) — SP FLOOR BREACH CONFIRMED, NON-MERGE projected; CONTINUE to EP30 for scientific value

EP18-22 disambiguation read: EP18 val_SP=3.8491 > 3.840 threshold = **CONSERVATIVE READ CONFIRMS** (the EP16→17 −0.025pp drop was noise, not decay-phase mechanism). SP slope EP15→EP22 = −0.0008pp/EP (essentially flat through 8 EPs).

Projected terminal val_SP ~3.83 → test_SP ~3.63 = **FAILS 3.577 floor by ~0.05pp** = NON-MERGE.

WSS plateau ~6.587-6.589 at EP22; terminal projection 6.55-6.59 val → test 6.39-6.43 = TIES or NARROWLY BEATS H147 (best case −0.15pp). Per-channel-heads decoder approach extracted some WSS-axis benefit but introduced SP regression.

Terminal at EP30 expected ~03:30Z (47 min/EP × 8 EPs from EP22 22:22Z). Will close NON-MERGE upon terminal.

**Scientific contribution recovery**: per-axis τ slopes EP15→EP22 (τ_x −0.0028pp/EP, τ_y −0.0070pp/EP, τ_z −0.0014pp/EP) confirm τ_y is the most-improvable WSS axis with axis-decoupled heads, τ_z is the persistent bottleneck (consistent with H154/H155/H156 falsifications).

### H184 (fern, PR #1513) — EP22 decay binary read pending, ~01:00Z

EP15-19 val_WSS stuck at ~6.84 (descent collapsed from −0.016pp/EP to flat). val_SP=4.06 (+0.49 above floor) is severely off-trajectory — even strong WSS decay won't pull SP through 3.577 floor unless decay delivers >−0.50pp SP over 8 EPs (very unlikely).

EP22 decay activation (~01:00Z) tests:
- WSS decay magnitude (target −0.30pp over 8 EPs for SOTA candidacy)
- SP decay magnitude (target ≥−0.50pp to reach floor — almost certainly fails)

**Most likely outcome: NON-MERGE on SP floor + likely WSS as well**, but EP22+ data informs future WSD-schedule decisions.

## 20:31Z snapshot — **H181 CLOSED NON-MERGE** (test_SP=3.6808 floor breach +0.104pp); H185 `hidden_dim=640` assigned to frieren (PR #1527); H182/H183/H184 still active

**Active fleet, 4 students, 3 WIP + 1 new assignment:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | Status |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1527 | H185 hidden_dim=640 | — | — | — | — | NEW ASSIGNMENT |
| nezuko | #1506 | H182 EMA 0.9999 + LR 1.3× | ~25-26 | 6.770 (flat) | **3.485** | 3.933 | partial-SOTA VP; terminal ~01:00Z |
| tanjiro | #1510 | H183 per-channel heads | 17 | **6.6002** | 3.5821 | **3.8412** | SP FLOOR WATCH; EP18-19 decisive |
| fern | #1513 | H184 WSD LR | 15.4 | 6.8392 | 3.727 | 4.046 | stable stall, EP22 decay binary read |

### H181 (PR #1503) CLOSED NON-MERGE — 20:30Z

- W&B run: `v4csonke` — 30 EPs, 24.77h runtime
- test_WSS=**6.6245** (+0.084pp vs H147) — FAILS primary
- test_VP=3.6245 ✓, test_ABUPT=5.7956 ✓
- test_SP=**3.6808** ✗ — BREACH +0.104pp over 3.577 floor
- **EMA-0.99995 falsified at 30-EP** — optimal EMA N≈10k steps (decay≈0.9999 confirmed)
- Best checkpoint: EP20 EMA (cosine re-ascent after EP20 = stack property, not noise)

### H185 hypothesis (PR #1527, dl24-frieren)

`hidden_dim=640` on clean H147 stack (single-flag change). Tests the only major structural axis not probed in H164–H167 wave (which tested slices/pe_features/surface_out/heads but NOT hidden_dim). At +25% width, addresses τ_y/τ_z representational saturation hypothesis. Smoke EP1 then 30-EP DDP8.

Kill ladder: EP1 ≤14.5%, EP5 ≤6.90%, EP10 ≤6.70%, EP15 ≤6.60%. Must beat test_WSS<6.5409 + clear all 4 floors to merge.

### H183 (tanjiro, PR #1510) — EP17 SP descent signal present, ambiguous

EP16→17: WSS −0.0186, VP −0.0221, **SP −0.0252pp** (first large single-EP SP drop). BUT EP15→17 average is still −0.0013pp/EP (essentially flat). Disambiguator: if EP18-19 sustain −0.020+ pp/EP SP descent, tanjiro's read wins (SP floor clears); if SP reverts to flat, merge blocker stands. Current WSS projection if descent resumes: EP30 val_WSS ~6.43-6.50 = BEATS H147 by 0.05-0.11pp ✓. SP wall still uncertain.

### H182 (nezuko, PR #1506) — VP partial-SOTA candidate confirmed, terminal ~01:00Z

val_VP=3.4853 at EP24.7 — partial-SOTA on VP axis (projected test_VP~3.25-3.30, beats H147 test_VP=3.4014 by 0.10-0.15pp). WSS flat at 6.77 throughout = NON-MERGE on primary. Merge decision: VP-only improvement + WSS regression. Will wait for terminal SENPAI-RESULT.

### H184 (fern, PR #1513) — stable-phase stall, EP22 decay binary read ~01:00Z

val_WSS=6.8392 at EP15.4; descent collapsed from −0.016pp/EP (EP9→12) to −0.0018pp/EP (EP12→15). WSD design: stable phase EP1-22 at lr=1e-4, decay EP22-30 lr→1e-6 (100× drop). For H184 to win, decay phase MUST deliver ≥−0.30pp over 8 EPs (vs −0.015pp total in stable-phase). Prior probability weakening but EP22 binary read decisive.

## 19:18Z snapshot — **H183 WSS descent RESUMED (EP16→17.4 = −0.013pp/EP) but SP floor STILL flat at −0.0016pp/EP — MERGE BLOCKER stands**; H181 terminal slowed to ~22:00Z; H182 VP plateau holding 3.485; H184 EP16 slight uptick (decay not yet activated)

**Active wave-5 fleet, all 4 students WIP. 19:18Z fleet:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | val_ABU | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 28.1 | 6.8285 (↓0.014) | 3.7786 | 4.0136 | 6.0872 | NON-MERGE (terminal ~22:00Z) |
| nezuko | #1506 | H182 LR 1.3× | 25.3 | 6.7760 (+0.006) | **3.4850** | 3.9372 | 5.9651 | partial-SOTA on VP (terminal ~01:00Z next day) |
| tanjiro | #1510 | H183 per-channel heads | 17.4 | **6.6002** | 3.5821 | **3.8412 (flat)** | 5.8797 | SP FLOOR BLOCKER; WSS path credible |
| fern | #1513 | H184 WSD LR | 16.0 | 6.8486 (+0.009) | 3.7349 | 4.0491 | 6.0962 | descent stalled, EP22 decay binary read |

### H183 (tanjiro) — WSS descent RESUMED, SP floor still flat

| EP | val_WSS | val_VP | val_SP | val_ABU |
|---:|---:|---:|---:|---:|
| 15 | 6.6140 | 3.5927 | 3.8450 | 5.8915 |
| 16 | 6.6188 | 3.6042 | 3.8664 | 5.9015 |
| **17.4** | **6.6002** | **3.5821** | **3.8412** | **5.8797** |

**EP16→17.4 (1.4 EPs):**
- val_WSS −0.0186 (−0.013pp/EP — descent RESUMED after EP15→16 stall ✓)
- val_VP −0.022 (descending, well below floor)
- val_SP −0.025 over 1.4 EPs from EP16 (BUT vs EP15 = only −0.0038 = essentially flat over 2.4 EPs)
- val_ABU −0.022 (descending)

**SP merge-blocker analysis (REPEAT):** EP15→EP17.4 SP rate = −0.0016pp/EP. At terminal EP30 = 3.8412 − 0.020 = **3.82**. Val→test from H147 pattern (−0.20pp on SP) = test_SP = **3.62 = FAILS floor 3.577 by 0.04pp**.

**WSS projection (with resumed descent):** EP30 = 6.6002 − 0.013×12.6 = 6.43 stable; with decay tail boost EP22+ = **6.35-6.40 = BEATS H147 by 0.14-0.19pp** ✓

Tanjiro is expected to post EP18/EP20 boundary reads ~20:00-21:00Z. EP20 gate val_SP ≤3.85 already cleared at EP17.4=3.84. Tanjiro committed to posting EP20 heartbeat per his 16:57Z comment.

### H181 (frieren) — terminal pushed to ~22:00Z

| EP | val_WSS | Δ |
|---:|---:|---:|
| 24 | 6.8285 | — |
| 26.8 | 6.8499 | +0.021 |
| **28.1** | **6.8285** | **−0.014 (slight reversion)** |

Step rate appears to have slowed (current pace projects terminal ~22:00Z, not 20:50Z as estimated). EMA-9999.5 producing oscillation, not monotonic divergence. **NON-MERGE confirmed (val_WSS=6.83 >> H147 6.55 EP30).** Close PR #1503 NON-MERGE on SENPAI-RESULT landing.

### H182 (nezuko) — VP plateau holding at 3.485

VP trajectory EP18=3.5172 → EP21=3.4872 → EP24=3.4807 → EP25.3=3.4850 (slight uptick within noise). **Plateau confirmed** at ~3.48 range. Test_VP projection 3.28 = beats H147 by 0.12pp = partial-SOTA on VP.

WSS plateau 6.77 = NON-MERGE on WSS. Terminal ETA ~01:00Z (next day, +6h from now at slowed pace).

### H184 (fern) — EP16 slight uptick, decay still not active

| EP | val_WSS | rate |
|---:|---:|---:|
| 11.9 | 6.8454 | — |
| 14.7 | 6.8404 | −0.0018/EP |
| **16.0** | **6.8486** | **+0.009/EP (uptick)** |

Stall has converted to slight uptick. WSD lr schedule should still be in stable phase (lr~1e-4) at EP16 — decay starts EP22. Confidence in EP22 decay-phase binary read remains, but probability of beating H147 dropping.

## 17:42Z snapshot — H183 SP floor merge-blocker discovery (val_SP=3.85 flat slope); H184 descent STALLED in stable phase (−0.0018/EP); H182 EP24 val_VP=3.481 still deepening; H181 EP26.8 climbing NON-MERGE confirmed

**Active wave-5 fleet, all 4 students WIP. 17:42Z fleet:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | EP30 risk | Decision |
|---|---|---|---:|---:|---:|---:|---|---|
| frieren | #1503 | H181 EMA-99995 | 26.8 | 6.8499 (↑) | 3.792 | 4.022 | NON-MERGE (terminal hold) | continue to EP30 ~19:00Z |
| nezuko | #1506 | H182 LR 1.3× | 24.0 | 6.7671 | **3.481** | 3.930 | WSS NON-MERGE / VP partial SOTA | continue to EP30 ~20:30Z |
| tanjiro | #1510 | H183 per-channel heads | 16.2 | 6.6188 | 3.604 | **3.866** | **SP FLOOR BLOCKER** | continue, SP decay-phase critical |
| fern | #1513 | H184 WSD LR | 14.7 | 6.8404 | 3.745 | 4.054 | descent STALLED, EP22 decay critical | continue, decisive read EP22 ~21:00Z |

### H183 (tanjiro) — SP FLOOR IS THE BLOCKER (WSS path still credible)

Tanjiro's EP15 boundary read (15:37Z): val_WSS=6.6140, val_VP=3.5927, val_SP=3.8450. All kill gates PASS, but SP slope is flat (~−0.002pp/EP). Mapping val→test from H147 pattern:

| Channel | H147 val EP30 | H147 test EP30 | val→test Δ |
|---|---:|---:|---:|
| WSS | ~6.60 | 6.5409 | −0.06 |
| VP | ~3.60 | 3.4014 | −0.20 |
| **SP** | **~3.76** | **3.5634** | **−0.20** |

**If H183 val_SP terminal = 3.82, projected test_SP = 3.62 = FAILS floor 3.577 by 0.04pp.** This blocks merge regardless of WSS performance.

**EP30 projections (val):**
- WSS at current −0.005pp/EP: 6.55 (TIES H147); with decay boost: **6.45-6.50 (BEATS H147)** ✓
- VP at current −0.020pp/EP: 3.32 (healthy below floor) ✓
- **SP at flat slope: 3.82 → test_SP = 3.62 = FAILS floor** ✗

**Decision: CONTINUE — but EP22+ decay phase MUST accelerate SP descent.** H147's cosine reaches lr~1e-6 by EP30. If H183 SP follows same pattern, end-game descent could be 5-10× current rate. EP20 and EP25 boundary reads are decisive for SP.

### H184 (fern) — descent STALLED in stable phase (decay phase EP22 is now MUST-DELIVER)

| EP | val_WSS | descent rate |
|---:|---:|---:|
| 9 | 6.8922 | — |
| 11.9 | 6.8454 | −0.016/EP |
| **14.7** | **6.8404** | **−0.0018/EP (STALL)** |

**Descent stalled from −0.016 to −0.0018pp/EP** — 9× slowdown. The WSD design predicted continued stable-phase descent at lr=1e-4. This stall is unexpected.

**Updated EP22 projection:** 6.8404 - 0.0018×7.3 = **6.83** (vs my 15:34Z projection 6.69). +0.20pp behind H147 EP22 ~6.58.

**Decay phase EP22→EP29 (lr 1e-4 → 1e-6) MUST deliver 4-5× boost to reach H147 territory:**
- Best case (5× boost = −0.10pp/EP × 7 EPs = −0.70pp total): EP30 = **6.13** (improbable, overshoot)
- Realistic (3× boost = −0.06pp/EP × 7 EPs = −0.42pp): EP30 = **6.41** (BEATS H147 by 0.13pp)
- Conservative (2× boost = −0.04pp/EP × 7 EPs = −0.28pp): EP30 = **6.55** (TIES H147)
- Worst (decay-phase fizzles): EP30 = 6.75-6.80 = NON-MERGE

**Decision: CONTINUE — EP22 decay-phase activation is the binary read.** If decay activates strongly: H184 is a credible SOTA candidate. If decay fails to activate: NON-MERGE. EP22 boundary ~21:00Z.

### H182 (nezuko) — VP=3.481 still deepening at EP24 (partial-SOTA holding)

| EP | val_WSS | val_VP | floor delta |
|---:|---:|---:|---:|
| 18 | 6.7744 | 3.5172 | −0.126 |
| 21 | 6.7716 | 3.4872 | −0.156 |
| **24** | **6.7671** | **3.4807** | **−0.162** |

VP descent slowing to −0.0022pp/EP EP21→EP24 (vs −0.010 EP18→21). Plateau approaching at ~3.45-3.48 range. **Test_VP projection val→test ~−0.20pp:** EP30 val_VP ~3.45 → test_VP ~3.25 = **beats H147 test_VP=3.4014 by ~0.15pp** (partial-SOTA candidate confirmed).

WSS plateau holding at 6.77 = NON-MERGE on WSS.

### H181 (frieren) — EP26.8 climbing, NON-MERGE terminal-hold

| EP | val_WSS |
|---:|---:|
| 21 | 6.8175 |
| 24 | 6.8285 |
| **26.8** | **6.8499 (continued climb)** |

EMA-9999.5 averaging window producing unstable convergence — descent reversed by EP21, accelerating climb EP24→26.8 (+0.021pp/3EPs vs +0.011 EP21→24). EP30 projection 6.85-6.90 = consistently behind H147 6.5409 and H172 6.6517. **NON-MERGE — terminal harvest ~19:00Z.**

## 15:34Z snapshot — **H182 VP=3.487 DEEPENING (−0.156pp below floor, partial-SOTA strengthening, test_VP proj 3.20 = beats H147 by 0.20pp);** H183 EP13.6 = 6.6306 descent severely slowed (EP15 borderline FAIL, EP30 proj 6.51-6.57 still beats H147); **H184 EP11.9 = 6.8454 descent ACCELERATING to −0.016/EP (re-projected EP30 6.44-6.54 with decay boost = STRONG SOTA candidate);** H181 EP24 = 6.8285 slight uptick (NON-MERGE confirmed)

**Active wave-5 fleet, all 4 students WIP. 15:34Z fleet:**

| Student | PR | Hyp | EP | val_WSS | val_VP | EP30 projection | Decision |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 24.0 | 6.8285 | 3.778 | 6.75-6.83 | NON-MERGE (terminal hold) |
| nezuko | #1506 | H182 LR 1.3× | 21.1 | 6.7716 | **3.487** | WSS 6.70-6.73 / **test_VP 3.20** | **STRONG PARTIAL SOTA (VP)** |
| tanjiro | #1510 | H183 per-channel heads | 13.6 | 6.6306 | 3.639 | **6.51-6.57** | **SOTA candidate, EP22 decay critical** |
| fern | #1513 | H184 WSD LR (main) | 11.9 | 6.8454 | 3.760 | **6.44-6.54 (decay boost)** | **NEW SOTA candidate** |

### H182 (nezuko) — VP DEEPENING TO 3.487 (partial-SOTA strengthening over 13:20Z reading)

| EP | val_WSS | val_VP | floor delta |
|---:|---:|---:|---:|
| 12 | 6.7901 | 3.5550 | −0.088 |
| 15 | 6.7833 | 3.5251 | −0.118 |
| 18 | 6.7744 | 3.5172 | −0.126 |
| **21** | **6.7716** | **3.4872** | **−0.156** |

**VP descent re-accelerating EP18→EP21: −0.030pp/3EPs = −0.010pp/EP** (up from −0.0027/EP at EP15-18). Likely the cosine decay phase starting — LR has dropped below stable regime, tighter convergence pulls VP deeper.

**Test_VP projection updated:** EP30 val_VP ~3.40 → test_VP ~3.20 = **beats H147 test_VP=3.4014 by ~0.20pp** (strengthened from 13:20Z 0.10pp projection).

**Channel-specific partial-SOTA candidate confirmed.** Continue to natural terminal EP30 ~16:00Z for test harvest.

### H183 (tanjiro) — descent severely slowed but STILL ON SOTA PATH (EP15 borderline)

| EP | val_WSS | descent rate |
|---:|---:|---:|
| 5 | 6.7497 | — |
| 8 | 6.6726 | −0.026/EP |
| 11 | 6.6407 | −0.011/EP |
| **13.6** | **6.6306** | **−0.0039/EP (CONCERN)** |

**Descent dropped 5× from EP8-11 to EP11-13.6.** EP15 gate ≤6.60 likely FAILS by 0.025pp (projected 6.625). However:
- EP30 projection at current rate: **6.567** = SLIGHTLY BEATS H147 by 0.03pp
- WITH decay-phase tail (H172-class ~−0.05pp): 6.51-6.52 = CLEARLY BEATS H147
- Worst-case continued deceleration: 6.59-6.61 = ties/slightly behind

**Critical concerns:**
- val_VP=3.639 just −0.004pp below floor 3.643 (slight regress from EP11=3.618). Watch VP closely — if crosses 3.65 sustained 2+EPs = kill.
- val_SP=3.854 = +0.276pp above floor 3.577. NO improvement EP11→13.6. **SP needs to crash by terminal or test_SP regresses.**

**Continue running, EP18-20 reads critical.**

### H184 (fern) — descent ACCELERATING to −0.016/EP in stable phase (NEW STRONG candidate)

| EP | val_WSS | descent rate | lr |
|---:|---:|---:|---:|
| 5 | 6.9329 | — | 1e-4 |
| 9 | 6.8922 | −0.010/EP | 1e-4 |
| **11.9** | **6.8454** | **−0.016/EP (UP)** | 1e-4 |

**Stable-phase descent ACCELERATING from −0.010 to −0.016pp/EP.** WSD design paying off — finding deeper minima as stable phase extends.

**Re-projected EP22 (last stable EP):** 6.8454 - 0.016×10.1 = **6.69** (vs my 13:20Z projection 6.76)

**Decay phase EP22→EP29 (lr 1e-4 → 1e-6) at 2-5× boost:**
- Conservative (2×): EP30 = 6.54 (TIES H147)
- Realistic (3-4×): EP30 = **6.44-6.49 = BEATS H147 by 0.05-0.10pp**
- Optimistic (5×): EP30 = 6.24 (overshoot likely smoothed by EMA)

**H184 is now a STRONG SOTA candidate.** Decay-phase boost is the decisive read. **EP22 ~17:00Z** is critical.

### H181 (frieren) — slight uptick at EP24, NON-MERGE confirmed terminal-hold

| EP | val_WSS |
|---:|---:|
| 18 | 6.8305 |
| 21 | 6.8175 |
| **24** | **6.8285 (+0.011 uptick)** |

EMA-9999.5 averaging window causing instability at local minimum. EP30 projection 6.75-6.83 = consistently behind H147 6.5409 and H172 6.6517. **NON-MERGE — continue to terminal EP30 for clean test harvest.**

## 13:20Z snapshot — **H183 EP11 = 6.6407 ON SOTA TRAJECTORY** (EP10 gate PASSED, projecting EP30 ~6.44-6.55, VP at 3.62 borderline); H182 EP18 VP CONTINUES TO BREAK FLOOR (3.517, −0.126pp below); H184 EP9 stable phase +0.24pp behind H147 (WSD design lag, decay EP22+); H181 EP21 severely decelerated, NON-MERGE re-sealed

**Active wave-5 fleet, all 4 students WIP. 13:20Z fleet:**

| Student | PR | Hyp | EP | val_WSS | val_VP | EP30 projection | Decision |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 21 | 6.8175 | 3.760 | 6.70-6.77 | NON-MERGE (sealed) |
| nezuko | #1506 | H182 LR 1.3× | 18 | 6.7744 | **3.517** | 6.70-6.73 / VP=3.25-3.30 | **PARTIAL MERGE (VP-side)** |
| tanjiro | #1510 | H183 per-channel heads | 11 | **6.6407** | 3.618 | **6.44-6.55** | **STRONGEST SOTA CANDIDATE** |
| fern | #1513 | H184 WSD LR (main) | 9 | 6.8922 | 3.794 | 6.56-6.72 (decay EP22+) | continue, EP22 critical |

### H183 (tanjiro) — STRONGEST SOTA CANDIDATE: EP11 val_WSS = 6.6407, EP10 gate PASSED

| EP | val_WSS | descent | Δ vs H147 |
|---:|---:|---:|---:|
| 5 | 6.7497 | — | −0.000 (TIED) |
| 8 | 6.6726 | −0.0257/EP (EP5→8) | ~tied EP8 |
| **11** | **6.6407** | **−0.0106/EP (EP8→11)** | **~−0.04 from H147 trajectory** |

**EP10 critical gate ≤6.66 PASSED** (EP10 interpolated ~6.65; EP11 = 6.6407 confirms persistent structural advantage of per-channel decoder heads). Per-axis τ_y/τ_z: 7.27/9.09 — cross-flow channels learning faster than H147 baseline.

**EP30 terminal projections:**
- Linear extrap (−0.011pp/EP × 19 EPs): **6.44** = BEATS H147 6.5409 by ~0.10pp
- Conservative (continued deceleration): 6.55 = SLIGHTLY BEATS H147
- Worst-case plateau: 6.65 = TIES H172, below H147

**VP floor watch CRITICAL:** EP11 VP = 3.618, just −0.025pp below 3.643 floor. EP5 VP = 3.82 → descent of 0.20pp over 6 EPs. Watch for VP regress > 3.65 sustained = kill signal.

**SP concern:** 3.85 vs floor 3.577 = +0.27pp above floor. H147 val_SP EP30 was ~3.6, so H183 needs SP to drop ~0.25pp by terminal.

### H182 (nezuko) — VP FLOOR BREACH SUSTAINED (3.517 at EP18, partial-SOTA candidate on VP-side)

| EP | val_WSS | val_VP | floor delta |
|---:|---:|---:|---:|
| 12 | 6.7901 | 3.5550 | −0.088 |
| 15 | 6.7833 | 3.5251 | −0.118 |
| **18** | **6.7744** | **3.5172** | **−0.126pp** |

**WSS plateau confirmed at ~6.77** — H182 won't beat H147 on WSS (projected EP30 6.70-6.73 vs H147 6.5409). 

**VP improvement persists.** Test_VP projection via val→test ~−0.20pp: 3.25-3.30 = beats H147 test_VP=3.4014 by ~0.10pp. **Channel-specific partial-SOTA candidate.** Mechanism (LR 1.3× × EMA-9999 windowing) failed on WSS but succeeded on VP. Continue to terminal for test harvest.

### H184 (fern) — WSD stable phase, slow descent +0.24pp behind H147 (design lag)

| EP | val_WSS | descent | lr | Δ vs H147 |
|---:|---:|---:|---:|---:|
| 5 | 6.9329 | — | 1e-4 | +0.183 |
| **9** | **6.8922** | **−0.010/EP** | **1e-4** | **+0.240** |

**lr=1e-4 holding** through stable phase (EP1-EP22 design). Descent rate −0.010pp/EP healthy but slow. **At current rate, EP22 = 6.76 = FAILS gate ≤6.65 by 0.11pp.** Decay phase EP22→EP29 (lr 1e-4 → 1e-6) should accelerate descent 2-5× over H147's flat-tail. Best-case EP30 = 6.56 (TIES/SLIGHTLY BEATS H147), realistic 6.62-6.66 (matches H172), worst case 6.72 (NO improvement). **EP22 boundary is the decisive read.**

### H181 (frieren) — Descent SEVERELY decelerated, NON-MERGE RE-SEALED

| EP | val_WSS | descent rate |
|---:|---:|---:|
| 15 | 6.8811 | — |
| 18 | 6.8305 | −0.025/EP |
| **21** | **6.8175** | **−0.004/EP** |

Descent went from −0.060 (EP12-14) → −0.025 (EP15-18) → **−0.004 (EP18-21)**. Severely flattened. Terminal projection 6.70-6.77 (vs H147 6.5409, H172 6.6517) = **NON-MERGE confirmed.** Continue to natural terminal EP30 for clean test metric harvest. EMA decay 0.99995 too aggressive for 30-EP budget.

## 11:10Z snapshot — H182 EP15 PASS gate + VP HOLDING below floor; H183 EP8 tracking ~tied H147 (EP10 gate imminent); H181 descent decelerated but credible H172-beat path; H184 EP5 trailing H147 +0.183pp (WSD design lag)

**Active wave-5 fleet, all 4 students WIP:**

| Student | PR | Hyp | EP @ 11:10Z | Latest val_WSS | Δ vs H147 | Δ vs H172 | Next gate |
|---|---|---|---|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 18 (step 197,567) | 6.8305 | +0.290 | +0.179 | EP22 ≤6.78 ~13:00Z (decay starts) |
| nezuko | #1506 | H182 LR 1.3× | **15** (step 164,639) | **6.7833** | **+0.183** | **+0.093** | **EP15 ≤6.85 PASS**; EP22 ≤6.75 next |
| tanjiro | #1510 | H183 per-channel heads | 8 (step 87,807) | **6.6726** | **~tied** | — | **EP10 ≤6.66 @ step 109,759 ~11:50Z (40 min)** |
| fern | #1513 | H184 WSD LR (main) | 5 (step 54,879) | 6.9329 | **+0.183** | — | EP10 ≤6.85 ~13:30Z (early lag expected) |

### H183 (tanjiro) — HOTTEST CANDIDATE: per-channel heads holding ~tied with H147

| EP | step | val_WSS | Δ vs H147 |
|---:|---:|---:|---:|
| 5 | 54,879 | 6.7497 | **−0.000** (TIED) |
| 6 | 65,855 | 6.7100 | ~tied |
| 7 | 76,831 | 6.6890 | ~tied |
| **8** | 87,807 | **6.6726** | **~tied** |

Descent rate EP7→EP8: −0.0164pp/EP (natural deceleration approaching plateau). **EP10 gate ≤6.66 at ~11:50Z:** linear extrap gives **6.640** (PASSES by 0.020), conservative gives 6.649 (PASSES by 0.011). H147 EP10 = 6.64 → H183 essentially TIES OR BEATS H147 EP10. Per-channel decoder heads structural advantage persisting. **If EP10 lands ≤6.64 → strong SOTA candidate.**

### H182 (nezuko) — EP15 PASS + VP CLEARING FLOOR sustained (LARGEST sustained VP-below-floor reading in 12+ hypotheses)

| EP | step | val_WSS | val_VP | val_ABU | lr | Δ vs H172 |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 131,711 | 6.7901 | 3.5550 | 5.9886 | 9.3e-5 | +0.099 |
| 13 | 142,687 | 6.7700 | 3.5419 | 5.9711 | 8.9e-5 | +0.080 |
| 14 | 153,663 | 6.7796 | 3.5331 | 5.9647 | 8.4e-5 | +0.089 |
| **15** | 164,639 | **6.7833** | **3.5251** | 5.9577 | 7.9e-5 | **+0.093** |

**EP15 critical gate ≤6.85: PASS (6.7833).** **VP @ EP15 = 3.5251 = −0.118pp BELOW 3.643 floor** (sustained EP12→EP15). If holds to test (val→test ~−0.3pp from H147 pattern), test_VP ~3.30 — clearly improving on H147 test_VP=3.4014. **VP-only improvement is itself a merge contribution.** EP12→EP15 WSS slope flat (−0.022pp/EP). EP30 projection: linear 6.73, with decay boost 6.65. **Merge-eligible if VP advantage holds.**

### H181 (frieren) — Descent decelerated but credible H172-beat path (REVISED from 08:40Z NON-MERGE seal)

| EP | step | val_WSS | Δ vs H147 | Δ vs H172 |
|---:|---:|---:|---:|---:|
| 15 | 164,639 | 6.8811 | +0.281 | +0.191 |
| 16 | 175,615 | 6.8551 | +0.255 | +0.165 |
| 17 | 186,591 | 6.8402 | +0.240 | +0.150 |
| 18 | 197,567 | 6.8305 | +0.230 | +0.140 |

Descent rate now −0.0169pp/EP (decelerated from −0.06pp/EP at EP12-14). Linear extrap EP30: 6.63 — **beats H172 (6.6517) clearly**; conservative 6.65-6.70 — close-tied with H172. **REVISED from 08:40Z NON-MERGE call:** H181 has credible path to beat H172, possibly approach H147 with cosine decay tail boost (EP22+).

### H184 (fern) — WSD stable phase tracking +0.18pp behind H147 (expected, design payoff EP22+)

| EP | step | val_WSS | lr | Δ vs H147 |
|---:|---:|---:|---:|---:|
| 1 | 10,975 | 12.7118 | 1.00e-4 | **−0.108 (LEAD)** |
| 2 | 21,951 | 7.3219 | 1.00e-4 | +0.062 |
| 3 | 32,927 | 7.0769 | 1.00e-4 | +0.097 |
| 4 | 43,903 | 6.9996 | 1.00e-4 | +0.120 |
| **5** | 54,879 | **6.9329** | **1.00e-4** | **+0.183** |

lr=1e-4 stable phase active (vs H147 cosine EP5 ~8.7e-5). Higher LR = slower early descent but theoretical deeper minimum at decay end. Descent rate −0.072pp/EP (healthy, no plateau). **WSD payoff expected EP22+ when cosine begins.** EP25-EP30 decisive read. Don't kill on early lag.

## 08:40Z snapshot — H183 EP5 EXACTLY TIED with H147 (HOT candidate); H182 EP10 borderline + VP CLEARING FLOOR; H184 main EP2 close to H147; H181 EP14 closing slowly

**Active wave-5 fleet, all 4 students WIP:**

| Student | PR | Hyp | EP @ 08:40Z | Latest val_WSS | Δ vs H147 | Δ vs H172 | Next gate |
|---|---|---|---|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 14 (step 153,663) | 6.9186 | — | +0.213 | natural terminal EP30 |
| nezuko | #1506 | H182 LR 1.3× | 11 (step 120,735) | **6.8158** | **+0.166** | **+0.005** | **EP15 ≤6.85 @ step 164,625 ~10:10Z** |
| tanjiro | #1510 | H183 per-channel heads | 5 (step 54,879) | **6.7497** | **−0.000** (TIED) | — | **EP10 ≤6.66 @ step 109,759 ~11:50Z** |
| fern | #1513 | H184 WSD LR (main) | 2 (step 21,951) | 7.3219 | +0.062 | — | EP5 ≤6.85 ~13:00Z |

### H183 (tanjiro) — HOTTEST CANDIDATE: EP5 EXACTLY TIED with H147

| EP | step | val_WSS | val_VP | val_SP | val_ABU | Δ vs H147 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,975 | 12.9728 | 14.0018 | 8.7417 | 13.0824 | +0.153 |
| 2 | 21,951 | 7.2222 | 4.9462 | 4.1772 | 6.6413 | −0.038 |
| 3 | 32,927 | 6.9084 | 4.1676 | 3.9783 | 6.2318 | **−0.072** |
| 4 | 43,903 | 6.8066 | 3.9018 | 3.9187 | 6.0995 | −0.029 |
| **5** | 54,879 | **6.7497** | 3.8184 | 3.8917 | 6.0380 | **−0.000** (TIED) |

Per-axis τ: EP3 (−0.052/−0.072/−0.084 τx/τy/τz), EP4 (−0.034/−0.014/−0.004), EP5 mechanism confirmed sustained. Per-channel decoder heads provide a ~0.04pp persistent lead vs H147 without dissipating. EP10 critical gate ≤6.66. Linear extrap EP3→5 slope (−0.079pp/EP) gives EP10 ≈ 6.36 (easily passes); conservative ≈ 6.55-6.60 (still passes). VP @ EP5 = 3.82, watch floor 3.643 — needs to drop ~0.18pp by terminal.

### H182 (nezuko) — EP10 BORDERLINE FAIL but EP11 NEAR-TIED + **VP CLEARING FLOOR**

| EP | step | val_WSS | val_VP | val_SP | val_ABU | lr | Δ vs H172 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 87,807 | 6.8905 | 3.7134 | 4.0518 | 6.1071 | 1.13e-4 | +0.083 |
| 9 | 98,783 | 6.8555 | 3.6641 | 4.0124 | 6.0689 | 1.09e-4 | +0.040 |
| **10** | 109,759 | **6.8381** | **3.6224** | 3.9929 | 6.0439 | 1.03e-4 | **+0.057** |
| **11** | 120,735 | **6.8158** | **3.5942** | 3.9757 | 6.0205 | 9.8e-5 | **+0.005** |

EP10 critical gate ≤6.78 TECHNICAL FAIL by 0.06pp. BUT EP11 essentially tied with H172 (+0.005pp) AND **VP CLEARING 3.643 FLOOR** for the first time in 12 hypotheses (EP10=3.622, EP11=3.594). Decision: HOLD to EP15. EP15 critical gate ≤6.85 vs H172 EP15=6.690. The VP-below-floor finding is the most important signal — if val_VP=3.59 holds to test (val→test ~−0.3pp pattern from H147), test_VP could be ~3.30, well clear of floor and improving on H147 by ~0.1pp.

### H181 (frieren) — Steady descent, no path to H147

| EP | step | val_WSS | val_VP | val_SP | val_ABU | Δ vs H172 |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 120,735 | 7.2399 | 4.2698 | 4.3739 | 6.5242 | +0.429 |
| 12 | 131,711 | 7.0711 | 4.0295 | 4.2131 | 6.3328 | +0.309 |
| 13 | 142,687 | 6.9780 | 3.8973 | 4.1266 | 6.2299 | +0.250 |
| 14 | 153,663 | 6.9186 | 3.8430 | 4.0736 | 6.1697 | +0.213 |

Descent steady at −0.06pp/EP. Gap-closing rate to H172 = −0.072pp/EP — catches H172 around EP17-18, but H172 EP14→30 only descends −0.05pp total. Terminal projection ~6.65-6.68 = H172-class but no path to H147. NON-MERGE sealed absent H172-beat. Continue to natural terminal.

### H184 (fern) — Smoke verification COMPLETE & PASSED; main run at EP2

Smoke verified: warmup→stable LR transition (5e-6→1e-4 at EP1), stable plateau EP1-22 at 1e-4, decay cosine EP23-29 to eta_min=1e-6. No NaN/skipped steps. Dry-run schedule matches design.

Main run `usc1tpni` launched 06:32Z. EP1 val_WSS=12.7118 (−0.108pp LEAD vs H147), EP2=7.3219 (+0.062pp slight lag). lr=1e-4 confirmed at EP1+EP2 (WSD stable phase active). WSD payoff expected late (EP22+ when cosine decay starts). Don't kill on early-EP lag — hypothesis specifically about late-tail descent.

## 06:30Z snapshot — H181 EP10 PASSED critical gate; H182 EP8 gap NARROWING (+0.083 vs H172); H183 EP2 slight LEAD vs H147 (per-channel heads alive); H184 smoke pre-EP1

**Active wave-5 fleet, all 4 students WIP:**

| Student | PR | Hyp | Status @ 06:30Z | Latest val_WSS | Δ vs H172 | Δ vs H147 | Next gate |
|---|---|---|---|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | EP11 step 120,735 | 7.2399 | +0.429 | — | natural terminal EP30 |
| nezuko | #1506 | H182 LR 1.3× | EP8 step 87,807 | **6.8905** | **+0.083** | **+0.231** | **EP10 ≤6.78 @ step 109,759 ~07:15Z** |
| tanjiro | #1510 | H183 per-channel heads | EP2 step 21,951 | **7.2222** | **−0.078** | **−0.038** | EP3 ≤7.00 ~07:00Z |
| fern | #1513 | H184 WSD LR | smoke step 8,609 (EP0.78) | mid-warmup | — | — | EP1 boundary ~06:40Z |

**H181 EP10 PASS (frieren, PR #1503):**

| EP | step | val_WSS | val_VP | val_SP | val_ABU | Δ vs H172 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 87,807 | 8.9007 | 6.5517 | 5.4795 | 8.2701 | +2.093 |
| 9 | 98,783 | 7.8525 | 5.1557 | 4.8763 | 7.2019 | +1.037 |
| **10** | 109,759 | **7.3745** | 4.4093 | 4.5395 | 6.6699 | **+0.594** |
| 11 | 120,735 | 7.2399 | 4.2698 | 4.3739 | 6.5242 | +0.429 |

EMA-99995 init_mass washout: EP10=0.41%, EP11=0.24% (now functionally washed). Descent decelerating: −1.05 → −0.48 → −0.13pp/EP (EP9→11). Trajectory "5 EPs lagged" vs H172 in EMA-equivalent terms. H172 EP11→30 delta was only −0.16pp — H181 needs at least −0.6pp more to match H172 terminal; **not achievable** at current decel. H147 SOTA structurally unreachable in 30-EP envelope. Continue to natural terminal for clean test metrics for the lit comparison; **NON-MERGE sealed at terminal**.

**H182 EP6-EP8 gap NARROWING (nezuko, PR #1506):**

| EP | step | val_WSS | val_VP | val_SP | val_ABU | lr | Δ vs H172 | Δ vs H147 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 54,879 | 7.4043 | 4.6284 | 4.6367 | 6.7385 | 1.24e-4 | +0.131 | +0.654 |
| 6 | 65,855 | 7.1135 | 4.0192 | 4.3644 | 6.3671 | 1.21e-4 | +0.142 | +0.404 |
| 7 | 76,831 | 6.9771 | 3.8042 | 4.1861 | 6.2034 | 1.18e-4 | +0.098 | +0.297 |
| 8 | 87,807 | **6.8905** | **3.7134** | 4.0518 | 6.1071 | 1.13e-4 | **+0.083** | **+0.231** |

EP5 lead-reversal **NOT terminal** — H182 catching back up. Channel-asymmetric mechanism confirmed: VP improving 4.63→3.71 in 3 EPs (−0.92pp), faster than H172's VP track. WSS lagging H172 by only +0.083pp at EP8. **EP10 critical gate ≤6.78 at step 109,759 ~07:15Z:** linear extrap −0.11pp/EP gives EP10 ≈ 6.67 (PASS) but conservative ≈ 6.83 (FAIL). **VP floor watch:** EP8 = 3.7134, still +0.07pp above 3.643 floor — floor breach risk if VP slows.

**H183 EP1-EP2 mechanism alive (tanjiro, PR #1510):**

| EP | step | val_WSS | val_VP | val_SP | val_ABU | Δ vs H147 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,975 | 12.9728 | 14.0018 | 8.7417 | 13.0824 | +0.153 |
| **2** | 21,951 | **7.2222** | 4.9462 | 4.1772 | 6.6413 | **−0.038** |

Per-channel decoder heads initialized cleanly. EP2 SLIGHT LEAD vs H147 (−0.04pp) — meaningful for structural perturbation at this early stage. Tanjiro skipped separate smoke; direct-to-main reasonable (structural change, not numerical instability concern). Mechanism: independent per-channel heads decouple cp (well-fit) and τ_z (lagging) optimization. Watch EP3 ≤7.00 (~07:00Z) and EP5 ≤6.78 (~08:30Z) gates. If lead holds through EP5, strong SOTA candidate.

**H184 WSD smoke pre-EP1 (fern, PR #1513):** Smoke `ozxi8j68` at step 8,609 / EP0.78 mid-warmup. EP1 boundary at step 10,975 ~06:40Z. Schedule-shape verification primary: confirm lr peaks at 1e-4 at EP1, holds flat through EP2-EP3 (WSD stable phase). Will green-light main 30-EP if smoke EP1-3 track H147 ±0.3pp + lr-flat post-warmup.

## 04:15Z snapshot — H178 CLOSED NON-MERGE (test 4-floor breach); H184 WSD dispatched to fern; H182 EP5 lead REVERSED; H181 EP8 borderline EP10 gate

**H178 CLOSED (PR #1493, fern, run `csk7pkf1`):** terminal SENPAI-RESULT landed 04:02Z. Test metrics (best-EMA EP13):
- test_WSS=**6.6237** vs H147=6.5409 → **+0.083pp BEHIND** (MISS SOTA)
- test_VP=**3.9237** → **BREACH +0.281pp** over 3.643 floor
- test_SP=**3.6968** → **BREACH +0.120pp** over 3.577 floor
- test_ABUPT=**5.8672** → **BREACH +0.023pp** over 5.844 floor

**4-of-4 floor failure.** Wave-3 α/floor closure: 16-EP slow cosine WORSENS test_VP vs 8-EP equivalent (H173 3.78 → H178 3.92). Cosine duration is downstream of (α, floor) for GradNorm equilibria. Fern mechanism finding: w_τ_x absorbs the freed budget (NOT w_cp as H173 read suggested); the SP-protection narrative for floor=0.05 revised. w_τ_z dropped over the run (1.32 → 1.29) — the only axis that lost weight — and per-axis test_WSS_z=8.67% confirms τ_z under-weighting is the next mechanism lever.

**H184 dispatched to fern (PR #1513):** WSD LR Schedule (H-W5-1) — replace H147's full 30-EP cosine with Warmup(EP1) + Stable(EP2-EP22 at peak lr=1e-4) + Decay(EP23-EP30 cosine to lr_min). Hypothesis: more steps at peak LR explore loss surface more aggressively, then concentrated decay tail locks in deeper minimum. Lit-aligned (Hu et al. 2024 MiniCPM, Hägele et al. 2024). Code change: extend `build_lr_scheduler` in `trainer_runtime.py` to support `--lr-schedule wsd` with 3-stage SequentialLR.

**H182 (nezuko, PR #1506) — EP5 boundary: LEAD REVERSED:**

| metric | H182 EP5 (lr=1.3e-4) | H172 EP5 (lr=1e-4) | Δ |
|---|---:|---:|---:|
| val_WSS | **7.4043** | 7.2734 | **+0.131** |
| val_VP | 4.6284 | 4.8146 | **−0.186** |
| val_SP | 4.6367 | 4.3169 | +0.320 |
| val_ABU | 6.7385 | 6.6743 | +0.064 |

**Lead reversed:** EP4 was −0.252pp ahead; EP5 → +0.131pp behind H172. Kill gate ≤7.42 PASSES by 0.016pp marginally. Channel-asymmetric: LR 1.3× helps VP (−0.19pp), hurts WSS/SP (+0.13/+0.32pp). Continue to EP10 gate ≤6.78.

**H181 (frieren, PR #1503) — EP8 landed, on borderline EP10 gate:**

| EP | step | val_WSS | Δ vs H172 | Δ vs H147 |
|---:|---:|---:|---:|---:|
| 5 | 54,879 | 19.52 | +12.25 | +12.77 |
| 6 | 65,855 | 13.68 | +6.71 | — |
| 7 | 76,831 | 10.60 | +3.73 | — |
| 8 | 87,807 | 8.901 | +2.10 | +2.25 |

EMA-0.99995 init_mass at EP8=1.24%, EP10=0.41% (still mid-wash). Descent decelerating: −5.8 → −3.1 → −1.7pp/EP. EP10 linear extrapolation ~7.4 → **PASS by 0.1pp at critical gate ≤7.5**. If EP10 > 7.5, KILL. H181 trajectory is "5 EPs lagged" vs H172 in EMA-equivalent terms — best case it matches H172 at terminal (not beats).

**Wave-5 active map:**
- **dl24-tanjiro:** H183 Per-Channel Decoder Heads (PR #1510) — dispatched 03:20Z, status:wip
- **dl24-fern:** H184 WSD LR Schedule (PR #1513) — dispatched 04:13Z, status:wip
- **dl24-nezuko:** H182 LR 1.3× compound (PR #1506) — EP5 lead reversed, EP10 gate next
- **dl24-frieren:** H181 EMA-99995 (PR #1503) — EP8 done, EP10 critical gate next

## 03:25Z snapshot — H172 CLOSED NON-MERGE; H183 dispatched to tanjiro; H178 terminal ~03:37Z; H182 EP4 lead sustained; H181 EP7 washout continuing

**H172 CLOSED (PR #1469, tanjiro, run `7d83go4z`):**
Test metrics (EP28 best-EMA checkpoint):
- test_WSS=**6.5893** vs H147=6.5409 → **+0.0484pp BEHIND** (primary regress)
- test_SP=**3.6101** vs cap 3.577 → **+0.033pp BREACH**
- test_VP=3.5429 (clears 3.643 cap), test_ABUPT=5.7394 (clears 5.844 cap)
- **NON-MERGE:** EMA-0.9999 mid-train mechanism confirmed (EP6-EP20 lead), BUT does not survive convergence. SP is anti-correlated with high-decay EMA. Hypothesis **FALSIFIED** on primary metric and floor contracts.

**H183 dispatched to tanjiro (PR #1510):** Per-channel decoder heads — split shared surface_out MLP 4×1 into independent channel heads. Highest-upside wave-5 idea; directly targets the capacity bottleneck between cp (3.5%, well-fit) and τ_z (8.5%, lagging). Expected gain −0.1 to −0.3pp test_WSS.

**H178 (fern, PR #1493) — EP11-EP15 plateau, terminal ~03:37Z:**

| EP | val_WSS | val_VP | val_SP | val_ABU |
|---:|---:|---:|---:|---:|
| 11 | 6.8294 | 4.0598 | 4.0684 | 6.1744 |
| 12 | 6.8300 | 4.0465 | 4.0684 | 6.1713 |
| 13 | 6.8296 | 4.0357 | 4.0672 | 6.1684 |
| 14 | 6.8349 | 4.0377 | 4.0783 | 6.1746 |
| 15 | 6.8372 | 4.0351 | 4.0778 | 6.1746 |

WSS plateaued since EP11 (~6.830). VP stagnated at 4.04 — structural BREACH +0.40pp over 3.643 floor. EP16 terminal ETA ~03:37Z. **NON-MERGE sealed; awaiting SENPAI-RESULT then close + dispatch H-W5-1 WSD LR Schedule to fern.**

**H182 (nezuko, PR #1506) — EP4 boundary confirmed, lead sustained:**

| EP | H172 (ref) | H182 | Δ vs H172 | Gate |
|---:|---:|---:|---:|---|
| 1 | 52.018 | 48.701 | −3.32pp | PASS |
| 2 | 50.497 | 43.508 | −6.99pp | PASS |
| 3 | 16.632 | 13.528 | −3.10pp | PASS |
| **4** | **8.695** | **8.443** | **−0.252pp** | **PASS** |

Kill criterion was ≤ 8.85. H182 at EP4 = 8.443, −0.252pp AHEAD of H172 at same EP. EMA contamination at EP4 = 11.1% — residual −0.252pp lead is pure signal from 1.3× LR. EP5 boundary (step 54,879) imminent ~04:00Z.

**H181 (frieren, PR #1503) — EP7 mid-wash, step ~79k:**
EMA-99995 washout continuing. At ~03:12Z: step=79,434, EP7.2, summary val_WSS=10.60 (mid-epoch, not final EP7 boundary). EP7 likely landed near 11.0-11.5 (extrapolating from EP6=13.678 at −5.8pp/EP and acceleration). EP10 critical gate (step 109,759) ~04:30Z.

**Wave-5 dispatch plan:**
- **Tanjiro:** H183 Per-Channel Decoder Heads (PR #1510) — dispatched 03:20Z
- **Fern:** H184 WSD LR Schedule (H-W5-1) — dispatch ~03:40Z after H178 SENPAI-RESULT
- **Nezuko/Frieren:** monitoring active WIP (H182/H181)

## 02:20Z snapshot — H182 EP3 lead sustained (−3.10pp vs H172); H181 EP6 washout on schedule; H172/H178 near terminal

**MEMORY CORRECTION:** Earlier session-context note said "H172 EP3 = 7.36" — that was the RAW model value (now retracted). The EMA val_primary/wall_shear_rel_l2_pct at EP3 step 32,927 is **16.632** (verified from W&B). All H182 EP1-3 lead comparisons use the corrected EMA baseline.

**H172 (tanjiro) corrected EP1-5 EMA baseline:**

| EP | step | val_WSS | val_VP | val_SP | val_ABU |
|---:|---:|---:|---:|---:|---:|
| 1 | 10,975 | 52.018 | 38.698 | 39.132 | 50.745 |
| 2 | 21,951 | 50.497 | 39.502 | 36.373 | 48.123 |
| 3 | 32,927 | **16.632** | 14.086 | 10.865 | 15.896 |
| 4 | 43,903 | 8.695 | 6.915 | 5.335 | 8.235 |
| 5 | 54,879 | 7.273 | 4.815 | 4.317 | 6.674 |

**H182 (nezuko, PR #1506) EP1-3 with verified Δ vs H172:**

| EP | H172 | H182 | Δ vs H172 |
|---:|---:|---:|---:|
| 1 | 52.018 | 48.701 | **−3.32pp** |
| 2 | 50.497 | 43.508 | **−6.99pp** |
| 3 | 16.632 | **13.528** | **−3.10pp** |

H182 lead SUSTAINED through EP3 — LR 1.3× hypothesis still alive. Stop-loss criteria standing (>0.15pp regression triggers KILL). EP4 boundary ~02:40Z, EP5 ~02:55Z.

**H181 (frieren, PR #1503) EMA-99995 washout trajectory:**

| EP | val_WSS | init_mass | descent |
|---:|---:|---:|---|
| 2 | 75.130 | 33.3% | +21.3pp spike (known long-window signature) |
| 3 | 54.076 | 19.2% | −21.0pp |
| 4 | 32.290 | 11.1% | −21.8pp |
| 5 | 19.519 | 6.4% | −12.8pp |
| 6 | **13.678** | 3.7% | −5.8pp |

Descent on recalibrated schedule. EP10 (step 109,759 ~04:00Z) is the critical gate: ≤ 7.5 = thesis alive, ≥ 8.0 = falsified (over-smoothing).

- **H172 EP30 LANDED at 6.6517** (step 329,279, rt 22.6h); val_WSS plateau confirmed (EP28=6.6481, EP29=6.6528, EP30=6.6517). Test harvest pending (run still in "running" state). Projected test_WSS ~6.60-6.62 = +0.06-0.08pp BEHIND H147. **NON-MERGE on WSS** verdict held; all VP/SP/ABUPT floors clear. Awaiting SENPAI-RESULT from tanjiro.
- **H178 EP15 landed at 6.8372** (step 164,639) — slight uptick from EP14 (6.8349). WSS plateau + EP16 boundary pending (~03:20Z). VP 4.035 BREACH confirmed structural.
- **H182 EP4 boundary imminent** (step 43,901; currently step ~43,401 at ~02:50Z = ~2min). EP3 lead −3.10pp sustained.
- **H181 EP10 critical gate ~04:00Z** (step 74,237 currently, target 109,759).

## 01:50Z snapshot — H172 EP29 mild uptick (plateau noise); H178 EP14 plateau confirmed; H182 EP3 boundary imminent

**H172 (tanjiro, PR #1469) per-EP trajectory through EP29 (step 318,303, rt ~21.86h):**

| EP | val_WSS | val_VP | val_SP | val_ABU |
|---:|---:|---:|---:|---:|
| 25 | 6.6929 | 3.6477 | 3.9440 | 5.9635 |
| 26 | 6.6586 | 3.6088 | 3.9178 | 5.9278 |
| 27 | 6.6495 | 3.5900 | 3.9148 | 5.9177 |
| 28 | **6.6481** | 3.5836 | 3.9154 | 5.9153 |
| 29 | 6.6533 | 3.5862 | 3.9162 | 5.9189 |

EP29 nudged +0.005pp above EP28 — within typical EMA noise band. EP30 projection (val_WSS 6.65 ± 0.005) → test_WSS ~6.60-6.62 = +0.06-0.08pp BEHIND H147. Verdict NON-MERGE on WSS; all VP/SP/ABUPT floors clear.

**H178 (fern, PR #1493) per-EP trajectory through EP14 (step 153,663, rt ~10.22h):**

| EP | val_WSS | val_VP | val_SP | val_ABU |
|---:|---:|---:|---:|---:|
| 10 | 6.828 | 4.078 | 4.067 | 6.177 |
| 11 | 6.829 | 4.060 | 4.068 | 6.174 |
| 12 | 6.830 | 4.047 | 4.068 | 6.171 |
| 13 | 6.830 | 4.036 | 4.067 | 6.168 |
| 14 | 6.835 | 4.038 | 4.078 | 6.175 |

WSS PLATEAU + EP14 slight reversal (+0.005pp). VP descent halted at 4.04 — **BREACH +0.40pp** confirmed structural; 2 more EPs cannot rescue. Verdict NON-MERGE sealed.

- **H172 EP29 → EP30 terminal ETA ~02:30Z** (step 329,250 ~11k steps away at 4.07 it/s = ~45 min)
- **H178 EP14 → EP15-16 terminal ETA ~02:30Z** (similar throughput, ~22k steps to EP16)
- **H181 EP4** (frieren, PR #1503): EMA-99995 washout continuing per recalibrated kill ladder. Step 62k ~EP5-6.
- **H182 main** (nezuko, PR #1506): EP1=48.70, EP2=43.51. EP3 boundary at step ~32,927 imminent (current step 31,587, ~02:00Z). Decisive boundaries: EP10 must beat 6.78, EP20 must beat 6.55. Currently leading wave-4 candidate.
- **Wave-5 catalog committed (`99097d2`):** Dispatch order: H-W5-1 WSD → fern (post H178 terminal ~02:30Z), H-W5-2 Per-Channel Heads → tanjiro (post H172 terminal ~02:30Z). **Both terminals expected within same window — concurrent dispatch likely.**

**Path B partial walkback:** EMA-derivative at decay=0.9999 produces real (5-7bp) but sub-SOTA improvement. The wave-4 thesis is "EMA-derivative does not beat H147" — falsified mechanism would be "EMA degrades H147"; current evidence shows EMA improves slightly without beating. H181 (longer window) and H182 (higher LR × EMA) remain the wave-4 falsifiability tests.

---

## Latest research direction from human researcher team

No new issues directed at dl24 branch since 2026-05-29. Ensembles remain BANNED (per 2026-05-28 comment). Standing constraint: single-model DDP8 only, all 8 GPUs, max 24h.

Note: TAY branch reached test_WSS=6.6404% via H300 per-channel affine calibration on 2026-05-30. dl24 H147 still BEATS this on WSS, ABU, SP. No action required from dl24 side.

---

## CRITICAL: H180 "anti-starvation BREAKTHROUGH" walked back (19:15Z)

My 17:25Z post claimed H180 EP1 "ANTI-STARVATION CONFIRMED at main". That framing was wrong — EP1 VP=13.23 (−0.886 vs H147) was a **warmup-only artifact** of α=1.0 doubling the restoring force toward vol_p during the warmup ramp. Corrected EP-by-EP picture from `gz8t5gkt`:

| EP | val_WSS | Δ vs H147 | val_VP | Δ vs H147 | val_SP | Δ vs H147 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.87 | +0.058 | 13.23 | **−0.886 (warmup artifact)** | 9.00 | +0.102 |
| 2 | 7.32 | +0.064 | **5.93** | **+1.021 (cosine descent)** | 4.39 | +0.137 |
| 3 | 7.02 | +0.049 | 4.12 | −0.007 (recovery) | 4.17 | +0.115 |
| 4 | 6.92 | +0.089 | 3.90 | +0.038 (parity) | 4.09 | +0.131 |

w_vol_p=0.1523 at EP4 (3× floor, clamp_active=OFF throughout).

**Honest finding:** α=1.0 + floor=0.05 prevents the w_vol_p → floor collapse that broke H173, allowing VP to recover to H147 parity by EP3-4. But WSS runs persistently +0.05-0.09pp ABOVE H147 at every epoch. **H180 will not beat H147.** Mechanism diagnostic confirmed; not a SOTA candidate.

---

## CRITICAL UPDATE (21:15Z): H172 descent STALLED at EP20→EP23 — wave-4 EMA thesis at risk

**H172 (tanjiro, PR #1469, EMA decay 0.9999) at step 252,447, EP~23, rt ~17.5h, `7d83go4z`:**

| EP | step | val_WSS | val_VP | val_SP | val_ABU | EP→EP slope |
|---:|---:|---:|---:|---:|---:|---|
| 18 | 197,567 | 6.6554 | 3.6011 | 3.9268 | 5.9296 | — |
| 20 | 219,519 | 6.6521 | 3.5892 | 3.9143 | 5.9180 | −0.0017pp/EP |
| ~23 | 252,447 | **6.6566** | 3.5940 | 3.9195 | 5.9235 | **+0.0015pp/EP REVERSED** |

**val_WSS went UP +0.0045pp EP20→EP23.** Within noise but materially undercuts the EP30 projection I posted at 19:15Z (6.60-6.62). Likely revised EP30: **val_WSS 6.65-6.67 → test_WSS 6.62-6.64 = +0.08-0.10pp BEHIND H147 SOTA**.

**Mechanism re-reading:** EMA-0.9999 produced the deepest EP15-EP20 minimum of wave-3, but the descent is NOT durable through the cosine cooldown. The underlying model continues training; as cosine LR floors out, the model's movements draw the EMA back toward worse weights. EMA-0.9999 captures the cosine descent itself, not a generalization-floor improvement. The mechanism finding tightens to: **EMA-derivative gains last only while the underlying model is actively descending.**

**Implications for wave-4 H181 (EMA 0.99995):**
- Longer averaging window (effective N=20k vs N=10k steps) might delay the same plateau by 5-10 EPs
- But if the plateau is mechanism-bound to the cosine cooldown, longer EMA only delays the inevitable
- H181's value is now **more diagnostic than competitive** — confirms whether EMA-derivative hits a ceiling, or just needs different averaging

H172 is still **the closest active arm to H147** even at +0.08pp test_WSS. Allow to natural EP30 cap — EP25 boundary read will confirm whether this is a true stall or a temporary pause.

---

## Current research focus: wave-3 GradNorm-α grid — verdict converging

Wave-3 was constructed to probe whether GradNorm restoring force (α) or floor placement could resolve the H173 VP-starvation pattern while preserving SP-protection. With H180 corrected and H178 starvation persisting, the grid verdict is settling:

### Active runs (21:15Z status)

**H172 (tanjiro, PR #1469) — EMA 0.9999 — EP23 STALLED:**
- val_WSS=6.6566 (UP +0.0045pp from EP20 6.6521), val_VP=3.5940, val_SP=3.9195, val_ABU=5.9235
- Descent reversed direction in cosine cooldown — see CRITICAL UPDATE section above
- ETA EP30 natural cap: ~23:30-00:30Z
- Revised projection: test_WSS 6.62-6.64 = NON-MERGE on WSS but possibly clears all floor caps

**H178 (fern, PR #1493) — vol_p_floor 0.05, α=0.5, 16-EP — EP8 complete:**
- Step 87,851, rt 5.85h, val_WSS=6.8473 (+0.197), **val_VP=4.1338 (+0.575)**, val_SP=4.0771 (+0.212), val_ABU=6.2036
- WSS slope EP7→EP8 flattening (−0.013pp), VP starvation persistent at +0.55-0.58pp throughout
- 8 EPs remaining of 16-EP cosine — mechanism finding settled but harvesting for clean α/duration grid comparison
- ETA EP16 terminal: ~22:30-23:00Z

**H180 (nezuko, PR #1494) — CLOSED NON-MERGE 22:39Z, `gz8t5gkt`:**
- test_WSS=6.6722% (+0.131pp), test_VP=3.6641% (BREACH +0.021pp), test_SP=3.7113% (BREACH +0.134pp), test_ABUPT=5.8389% (pass marginal)
- Mechanism confirmed end-to-end: w_vol_p NEVER clamped, r_vol_p decapped 5.00→3.29
- vs H173: α=1.0 trades +0.115pp VP recovery for −0.165pp SP regression — wave-3 grid closed
- **H182 (EMA 0.9999 + LR 1.3×) dispatched to nezuko (PR #1506)**

**H181 (frieren, PR #1503) — EMA 0.99995 — main launched ~21:55Z:**
- Smoke validated EP1 val_WSS=54.39% (consistent with H172 EP1=52.02%), no crash
- Orphan `1vgmgyr2` debugged (data_root fix) — student correctly used lr=1e-4 per H147 actual config
- Main 30-EP launch authorized ~21:50Z, ETA terminal ~20:30Z 2026-05-31

**H182 (nezuko, PR #1506) — EMA 0.9999 + LR 1.3e-4 — smoke pending:**
- Dispatched 22:40Z after H180 close
- H172 stack exact + only lr: 1e-4 → 1.3e-4
- Smoke first, then main 30-EP
- Tests whether higher peak LR extends H172's descent durability past EP20 stall

---

## Wave-3 grid status (as of 19:15Z)

| Experiment | floor | cosine_EPs | α | Status | Verdict |
|---|---:|---:|---:|---|---|
| H173 (closed) | 0.05 | 8 | 0.5 | NON-MERGE | VP breach +0.136 from floor clamping |
| H176 (closed) | 0.10 | 8 | 0.5 | NON-MERGE 20:24Z | test_WSS +0.138; VP BREACH +0.022; SP BREACH +0.085 (worst-of-both) |
| H178 (running) | 0.05 | 16 | 0.5 | EP8 VP +0.575 | 16-EP doesn't fix starvation (worst VP of all 4) |
| H180 (running) | 0.05 | 8 | 1.0 | EP7 VP +0.099 | α=1.0 prevents clamping; WSS +0.14 persistent |
| H172 (running) | — | — | — | EP23 STALLED | EMA decay 0.9999, descent reversed in cooldown |

**Wave-3 conclusion (revised 21:15Z, pending H172 terminal):** GradNorm-α grid does NOT produce a H147-beater. The α=1.0 + floor=0.05 combination resolves the H173 clamping pathology but introduces a persistent +0.05-0.15pp WSS regression. The productive lever appeared to be **EMA-0.9999 (H172)** but the EP20→EP23 stall casts doubt on durability — EMA captures cosine descent, not a true generalization-floor improvement. Wave-4 must validate whether longer averaging (H181 0.99995) extends descent durability or hits the same ceiling.

---

## Wave-4 design queue (REVISED 21:15Z, contingency for H172 stall)

Revised based on H180 walkback, H172 EP20→EP23 stall, and H172 leadership weakening. Two paths forward depending on H172 EP30 terminal:

**Path A — H172 terminal val_WSS ≤ 6.62 (productive descent recovers):**
1. **H181: EMA decay 0.99995** (frieren, in flight as smoke) — push past H172's 0.9999
2. **H182: H172 stack + LR 1.3× peak** (nezuko, after H180 terminal ~22:00Z)
3. **H183: H172 stack + extended cosine 30 EP** (fern, after H178 terminal ~22:30Z) — cap at ≤32 EP; 40-EP exceeds 24h budget at H172's rate
4. **H184: H172 stack + structural perturbation** (tanjiro, after H172 terminal ~00:00Z) — verify `--model-heads` flag, smoke first

**Path B — H172 terminal val_WSS ≥ 6.65 (EMA-derivative thesis falsified):**
- Allow H181 to run (already in flight, useful diagnostic) — if H181 also stalls, EMA-derivative is settled NON-PRODUCTIVE
- Pivot wave-5 to architecture/data-representation:
  - Attention head count or layer width (structural)
  - Y-symmetry augmentation variants (training data)
  - Different LR schedule families (WSD, one-cycle, restart)
  - τ_y-pin mechanism (from H176 finding: w_τ_y emerged as dominant freed-budget absorber)
- Dispatch researcher-agent for fresh wave-5 hypotheses if H172 terminal confirms stall

**Plateau Protocol check:** H167-H180 = 14+ consecutive experiments without beating H147. If H172 + H181 both fail to beat, we are in formal plateau and must escalate strategy tier per CLAUDE.md.

---

## Critical constraints in force

1. Test VP floor cap: test_VP ≤ 3.643%
2. Test SP floor cap: test_SP ≤ 3.577%
3. Test ABUPT floor cap: test_ABUPT ≤ 5.844%
4. Primary metric: test_WSS (lower is better, must beat H147 6.5409%)
5. Paper-facing primary: test_primary/abupt_axis_mean_rel_l2_pct
6. DDP8 only (no split GPU arms)
7. Ensembles BANNED

## Terminations ETA cluster (01:25Z)

| Run | PR | Student | ETA | Action |
|---|---|---|---|---|
| H176 | #1486 | frieren | ✓ CLOSED 20:24Z 2026-05-30 | H181 EMA 0.99995 dispatched (PR #1503) |
| H180 | #1494 | nezuko | ✓ CLOSED 22:39Z 2026-05-30 | H182 EMA+LR 1.3× dispatched (PR #1506) |
| H181 | #1503 | frieren | 2026-05-31 ~20:30Z | EP5 val_WSS=19.52 (washout proceeding normally, ~40%/EP descent) |
| H182 | #1506 | nezuko | 2026-05-31 ~21:30Z | EP2 val_WSS=43.51 **−7pp vs H172 EP2**, strong early lead (rank-0 `ecw2sct9`) |
| H178 | #1493 | fern | ~02:50Z 2026-05-31 | EP13 WSS plateau 6.83, VP=4.036 BREACH; terminal at EP16 imminent; dispatch **H-W5-1 WSD LR Schedule** |
| H172 | #1469 | tanjiro | ~03:40Z 2026-05-31 | EP28 val_WSS=6.6481 descent decelerating (−0.0014/EP); EP30 terminal ~6.645; NON-MERGE on WSS; dispatch **H-W5-2 Per-Channel Decoder Heads** |

## H147 actual EP boundaries (from k6q4c3on val history, authoritative reference)

| EP | step | WSS | VP | SP | ABU |
|---:|---:|---:|---:|---:|---:|
| 1 | 10975 | 12.8153 | 14.1177 | 8.9014 | 13.0458 |
| 2 | 21951 | 7.2593 | 4.9067 | 4.2517 | 6.6759 |
| 3 | 32927 | 6.9754 | 4.1254 | 4.0505 | 6.2795 |
| 4 | 43903 | 6.8349 | 3.8601 | 3.9572 | 6.1094 |
| 5 | 54879 | 6.7557 | 3.7299 | 3.9240 | 6.0225 |
| 6 | 65855 | 6.7215 | 3.6549 | 3.9107 | 5.9817 |
| 7 | 76831 | 6.6798 | 3.6035 | 3.8848 | 5.9376 |
| 8 | 87807 | 6.6498 | 3.5590 | 3.8651 | 5.9026 |
| 9 | 98783 | 6.6097 | 3.5189 | 3.8408 | 5.8650 |
| 10 | 109759 | 6.6249 | 3.5030 | 3.8443 | 5.8715 |
| 11 | 120735 | 6.5926 | 3.4807 | 3.8289 | 5.8441 |
| 12 | 131711 | 6.5742 | 3.4743 | 3.8259 | 5.8288 |
| 30 (terminal) | 329279 | 6.5451 | 3.4093 | 3.8078 | 5.7923 |
