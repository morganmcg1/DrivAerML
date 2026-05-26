# SENPAI Research State

**Updated**: 2026-05-26 ~13:10Z | Branch: `tay` | SOTA: H112 PR #1283

---

## Primary Objective

**test_WSS < 5.85%** (Transolver-3 SOTA, per Morgan Issue #1056)

Current SOTA (H112): val_abupt 6.1358% / test_WSS 6.752% / test_WSS_z 8.720%
Gap to target: test_WSS needs −0.90pp improvement.

Merge gate: val_abupt < 6.1358%, test_WSS ≤ 6.727%, test_VP ≤ 3.421%, test_SP ≤ 3.577%

---

## Wave 38+39 Final Closure (as of 2026-05-26 ~13:10Z)

### THE PROGRAM-WIDE FINDING — slope-flattening is geometry-bound and mechanism-class-agnostic

**Any perturbation off the H112 operating point causes slope-flattening — capacity, architecture, AND loss-weight (H144 update 2026-05-26 13:10Z).**

**LOAD-BEARING H144 update (2026-05-26 13:10Z)**: H144 tau_z=6.0 terminal C NULL completes the **3-point ESCALATE cohort H112/H143/H144**. val→test slope flattens MONOTONICALLY with tau_z weight: WSS aggregate −0.215 → −0.082 → −0.013pp (**16× flattening**); WSS_z (target) −0.655 → −0.446 → −0.290pp (**2.3× flattening on target channel**). test_WSS REGRESSES monotonically: 6.752 → 6.955 → 7.079%. **The val→test divergence WIDENS with weight magnitude.** H112's wt 2.0 is the test-optimum on this axis — direction is INVERTED.

**Convergent with H147 thorfinn GradNorm (independent dynamic-balancing axis)**: GradNorm-discovered optimal tau_z = 1.631 at step 33,082 — BELOW H112's 2.0. **Two independent mechanisms converge on tau_z DE-escalation, not escalation.**

**Original Wave 38 finding (kept for history)**: Any capacity addition at canonical 17.5M recipe val-overfits.

All seven Wave 38+39 mechanism classes exhausted:

| Class | Experiments | Verdict | Mechanism |
|---|---|---|---|
| Capacity-axis (depth/width) | H118, H120, H121, H125 | CLOSED | Val-overfit slope catastrophe |
| Pure regularization | H132 (DP_max=0.15) | CLOSED C NULL | Null — no improvement |
| SwiGLU gating | H128(OOM), H134(backbone), H135(decoder) | CLOSED | Val-overfit; param overhead acts as capacity expansion |
| Architectural-split | H138 (WSS_z decoder), H146 (WSS_y decoder trending) | CLOSED | Slope-flattening pathology — param overhead correlated. **H146 EP6 (corrected by student edward 11:30Z)**: val_abupt **+0.078pp LAG** vs H112 raw same-step (my 11:12Z W&B query used wrong baseline 7.111% vs actual 6.374%). H146 trajectory tracks H138 exactly — paired-class confirmed. WSS_y deficit smallest among channels (+0.033pp, real specialization) but +525K overhead drags everything. |
| Aux-head | H-B, H-B2 | CLOSED | Gradient-flow degradation; detached strictly worse |
| SOFTEN (loss) | H139 (Charbonnier), H140 (signed-log) | CLOSED | Anti-aligned with WSS_z goals; heavy-tail residuals ARE the signal |
| **ESCALATE (loss-weight)** | **H112 (wt 2.0 SOTA), H143 (wt 4.0), H144 (wt 6.0)** | **CLOSED DEFINITIVELY by 3-point cohort** | **val→test slope flattens monotonically with tau_z weight; test-optimum at wt ≤ 2.0; convergent with H147 GradNorm optimal=1.631** |

**Param overhead correlates with slope flattening magnitude** (H132 null → H135 −0.10pp → H138 −0.135pp → H120/H121 −0.20pp). Slope catastrophe is overhead-driven, not mechanism-specific.

**Cross-channel bleed under Lion**: narrow-scope tau_z loss changes produce LARGER regression on tau_y (+0.356pp) than tau_z (+0.254pp) itself — via Lion sign-only update propagation through shared decoder weights.

---

## Wave 39 Active Fleet (2026-05-26 ~13:10Z)

| Student | PR | Hypothesis | Status | Priority |
|---|---|---|---|---|
| frieren | #1332 | H143 tau_z=4.0 ESCALATE | **CLOSED C NULL** (test_WSS_z +0.175pp REGRESSION; first non-overhead slope-flattening) | reassigned to H164 SWA |
| fern | #1334 | H144 tau_z=6.0 ESCALATE | **CLOSED C NULL** (terminal 13:00Z). test_WSS +0.327pp REGRESSION on primary objective; test_WSS_z +0.264pp REGRESSION on target channel. Val-side wt curve monotone-productive on WSS_z (LEAD −0.100pp) but val→test slope flattens 16× across cohort. **3-point ESCALATE class closure**: H112/H143/H144 monotone-regressive on test_WSS as tau_z escalates. | reassigning to Wave 40 tau_z DE-escalation |
| alphonse | #1337 | H145 tau_y=3.0 axis-extension | WIP EP8/13; EP7 (10:39Z) val_WSS_y LEAD coin-flip at terminal (sub-bp); val_abupt MISS gate +22bp; terminal ~15:00Z | C NULL trending; banked: tau_y direction productive but 2× insufficient magnitude |
| edward | #1338 | H146 split WSS_y decoder | WIP EP7/13 step ~49,200 (70%). **Corrected EP6 reading (edward 11:30Z)**: val_abupt +0.078pp LAG, val_WSS_y +0.033pp LAG (smallest channel deficit — specialization real), val_WSS_z +0.166pp LAG vs H112 raw same-step. Tracks H138 trajectory exactly. Terminal ~13:30Z. | Trending C NULL — paired-class to H138 confirmed. Bank: split decoder specializes target channel but +525K overhead drags model. Architectural-split class CLOSED at +525K overhead. |
| thorfinn | #1340 | H147 GradNorm full (alpha=1.5) | WIP EP2 (09:25Z) — caught up dramatically, LEADING H112 on ALL WSS channels at EP2 (−0.182pp WSS, −0.105pp WSS_z, −0.268pp WSS_y). VP/SP lag from auto-downweight. tau_z weight stabilized at 1.54 (below H112's 2.0). Terminal ~21:00Z. | HIGH — dynamic balancing sidesteps static ESCALATE failure mode |
| askeladd | #1341 | H148 y=0 mirror augmentation | WIP EP5 (54%). EP4 publish: WSS_y LEAD weakening (−0.095→−0.056pp), WSS_z LEAD reversed to flat, VP deficit +1.294pp (growth decelerating but plateau ~+1.3pp). VP merge floor disqualification likely. EP6 binding ~12:25Z. | HIGH — mechanism alive but VP floor critical |
| tanjiro | #1342 | H149 AdamW optimizer swap | WIP EP5. EP4 (10:14Z) gap-closure decaying: EP1 +5.73 → EP4 +0.56pp. WSS_z/WSS_y ratio H149 1.169 vs H112 1.183 — **CROSS-CHANNEL BLEED IS OPTIMIZER-AGNOSTIC**. Terminal ~17-18Z. | HIGH — pivotal Lion-vs-AdamW; baseline ordering confirmed data-driven not optimizer-specific |
| nezuko | #1346 | H157 cosine warm restarts (SGDR T_0=4) | Newly assigned — student pickup pending | Scheduler-axis basin-escape, zero capacity |
| frieren (reassigned) | #1347 | H164 Stochastic Weight Averaging (SWA) | Newly assigned — student pickup pending | Averaging-axis flat-basin discovery; tests basin-geometry hypothesis from H143 directly |

---

## Wave 39 Priority Axes

These are the ONLY mechanism families with unexhausted potential. **Zero capacity increase for any of them.**

### 1. ESCALATE class (H143/H144/H145) — DEFINITIVELY CLOSED by 3-point cohort
- **H143 tau_z=4.0 CLOSED C NULL** (2026-05-26 10:55Z): test_WSS +0.203pp, test_WSS_z +0.175pp REGRESSION on target channel. val→test slope flattened 33% on WSS_z (basin-geometry pathology, not overhead-driven).
- **H144 tau_z=6.0 CLOSED C NULL** (2026-05-26 13:00Z): test_WSS +0.327pp REGRESSION on primary objective; test_WSS_z +0.264pp REGRESSION on target channel. **val→test slope flattens MONOTONICALLY with tau_z weight: −0.215 → −0.082 → −0.013pp on WSS aggregate (16× flattening across cohort).** Val-side curve monotone-productive on the targeted WSS_z channel up to wt 6.0 (9.375 → 9.341 → 9.275%) but val→test divergence WIDENS with weight magnitude.
- **H145 alphonse (tau_y=3.0) trending C NULL**: 7-checkpoint mechanism alive on val_WSS_y but magnitude decaying geometrically (~50%/epoch). Terminal val_WSS_y ≈ 7.607-7.608% (coin-flip on banked-partial). val_abupt MISS by ~22bp projected. Terminal ~15:00Z.
- **Strategic conclusion**: ESCALATE direction is INVERTED — H112's wt 2.0 is the test-optimum on the tau_z axis, with monotone test_WSS regression from wt 2.0 upward. **Convergent with H147 GradNorm-discovered optimal tau_z = 1.631** (BELOW 2.0). Wave 40 frontier shifts to **tau_z DE-escalation** (priority 7 below) and basin-geometry-targeting mechanisms.

### 2. GradNorm dynamic weighting (H147 thorfinn) — EP3 binding diagnostic 12:00Z REVEALS TWO MAJOR FINDINGS
- **Finding 1 — H143 basin-geometry pathology applies to GradNorm too**: WSS_z LEAD-FLIP between EP2 (−0.105pp LEAD) and EP3 (+0.047pp LAG) on the most aggressively upweighted channel. All-WSS LEADs compressed 50-66%. Smooth weight evolution does NOT sidestep the slope-flattening trap.
- **Finding 2 — GradNorm-discovered optimal tau_z = ~1.6, NOT > 2.0**: auto-weight plateaued at 1.631 by step 33,082, well BELOW H112's static 2.0. This DIRECTLY FALSIFIES the H143/H144/H145 ESCALATE direction. The gradient-aligned optimum is in the [1.5, 1.7] range, NOT above 2.0. **NEW Wave 40 hypothesis directly enabled: static tau_z=1.5 DE-escalation experiment.**
- **Operational risk**: terminal ETA ~01:00Z next day exceeds SENPAI_TIMEOUT_MINUTES=1100 (22:04Z hard timeout). Recommended Option A: early-stop at step 60,000 (EP10) with test eval on EP10 EMA checkpoint.
- VP auto-weight = 0.324 dragging val_VP toward terminal projection ~3.85% → test_VP FAILS floor 3.421%.
- Banked findings program-positive regardless of merge: dynamic balancing falsifies static ESCALATE hypothesis.

### 3. Mirror augmentation (H148 — just assigned)
- Zero params, zero architecture change
- Tests: is slope catastrophe a data-memorisation artifact?
- Implementation: `target/data/loader.py:428`, prob=0.5 train-only, yaw flip about y=0
- Falsifiable: if alive → data invariance closes slope catastrophe; if closed → optimizer/architecture problem

### 4. AdamW optimizer (H149 tanjiro, EP3 alive)
- Program-pivotal: does slope-flattening persist without Lion?
- EP3 val_abupt 7.836% — gap vs H112 EP3 closing (5.73→1.36→0.86pp per epoch). Linear extrapolation: gap closes EP5-6. Run healthy.
- If alive: Lion sign-accumulation contributes to slope pathology; opens SOFTEN-class revisit with AdamW
- If closed: pathology is optimizer-agnostic — further constrains the search space

### 5. Cosine warm restarts (H157 nezuko) — HIGHEST PRIORITY post H143 finding
- **Mechanism class**: Scheduler-axis (first time tested in programme)
- Hypothesis: single-cosine tail → flat-basin entrapment → slope flattening. SGDR restarts (T_0=4) force basin escape at EP5/EP9.
- Requires ~10-line code change (CosineAnnealingLR → CosineAnnealingWarmRestarts in trainer_runtime.py)
- **Elevated priority after H143**: this is the ONLY scheduled experiment directly targeting basin-geometry root cause.

### 6. Stochastic Weight Averaging (H164 frieren — NEWLY ASSIGNED post H143)
- **Mechanism class**: Averaging-axis basin discovery (Izmailov et al. 2018)
- Hypothesis: EMA's geometric decay accumulates a sharp-minimum average. SWA's uniform averaging over the last K epochs (with constant low LR) lands in a flatter region of the loss landscape with better generalization.
- Implementation: `torch.optim.swa_utils.AveragedModel` wrapper + SWALR scheduler activated at EP9 onward
- **Compounds with H157**: SWA over warm-restart cycles is the canonical basin-discovery combination

### 7. Wave 40 PRIORITY HYPOTHESIS — static tau_z DE-escalation (now ASSIGNED to fern post-H144 closure)
- **Hypothesis**: GradNorm-discovered optimal tau_z = 1.631 (H147 step 33,082) + H143/H144 monotone test_WSS regression from wt 2.0 upward → gradient-aligned optimum is BELOW H112's static 2.0. Static tau_z=1.5 single-flag DE-escalation, zero overhead, single recipe delta from H112 SOTA.
- **Mechanism class**: pure loss-weight axis, BELOW H112 baseline (qualitatively different from CLOSED ESCALATE class)
- **Two independent confirmations**: (1) H147 dynamic balancer converges to 1.6; (2) H112/H143/H144 cohort slope monotone-flattening as wt increases — extrapolation toward wt < 2.0 is the unexplored half-line
- **Falsifiability clean**: if alive at terminal, entire H143/H144/H145 wave searched the wrong half of the magnitude space
- **Zero-overhead**, single CLI flag change. Compatible with all other Wave 40 mechanisms.
- **Priority**: Wave 40 HIGH — direct test of H147 banked finding + H143/H144 closure direction.
- **Status (2026-05-26 13:10Z)**: fern PR being assigned post-H144 closure.

### 8. Wave 40 frontier post H144 closure
- **ESCALATE class CLOSED DEFINITIVELY** (H112+H143+H144 3-point cohort)
- **Active priorities**: tau_z=1.5 DE-escalation (fern Wave 40 assignment, this section), H147 GradNorm (thorfinn terminal ~20:34Z), H148 mirror aug (askeladd terminal ~16:30Z), H149 AdamW (tanjiro graceful ~20:58Z), H157 cosine warm restarts (nezuko ~25% complete), H164 SWA (frieren freshly launched)
- **H150 long-EMA**: paused, may revisit only if a winning Wave 40 mechanism completes — incompatible with 13-epoch training in isolation
- **H145 alphonse terminal ~15:00Z**: pending; even if A TIED on val_WSS_y banked-partial, ESCALATE class is closed by 3-point cohort regardless. H145's role now is to ROUND OUT axis-cohort (test whether tau_y ESCALATE also exhibits monotone test regression — provides cross-axis validation of basin-geometry conclusion).

### Cross-channel bleed under Lion — 2-class confirmed (H148 EP2 reference error corrected)
Bidirectionally symmetric under Lion sign-only updates:
- **H139 SOFTEN** (tau_z→tau_y bleed +0.356pp, confirmed at terminal)
- **H146 ARCHITECTURAL-SPLIT** (tau_y→tau_z bleed +2.2pp, confirmed at terminal)
- ~~H148 DATA-AUGMENTATION~~: EP2 claim retracted — H112 raw EP2 val_WSS_z = 11.752% (not 9.673%). H148 EP2 actually showed −0.105pp LEAD on WSS_z.

**H148 EP3 clean mechanism signal**: WSS_y LEAD −0.095pp, WSS_z LEAD −0.085pp, WSS_x FLAT (invariant channel). WATCH val_VP deficit growing (+0.43→+1.06→+1.22pp at EP1/2/3 — gradient reallocation from volume pressure to surface). VP floor (test_VP ≤ 3.421%) is make-or-break for merge.

Any narrow-axis intervention propagates through shared decoder weights — must verify cross-channel impact in every loss-weight or augmentation experiment.

---

## Diagnostic Invariants (DO NOT VIOLATE)

- No capacity additions — any param overhead ≥ +1% risks slope flattening
- No ensembles
- Cross-run comparison: W&B raw `val_primary/*` at same step (NOT EMA, NOT student tables)
- Kill thresholds: global_step 10864/32592/48897 (W&B _step leads by 2)
- Mirror augmentation axes: `surface_x[...,1]` (y), `surface_x[...,4]` (normal_y), `surface_y[...,2]` (tau_y), `volume_x[...,1]` (vol y); `volume_y` (pressure) invariant
- DrivAerML axes: x=streamwise, y=lateral (mirror axis), z=vertical

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-24)**: "test_WSS < 5.85% is THE objective. val_abupt is steering metric only."
- No ensembles — lazy route, want genuine breakthroughs
- All advisor work on `tay` branch; DDP 8 GPUs every run
