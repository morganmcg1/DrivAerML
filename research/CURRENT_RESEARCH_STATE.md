# SENPAI Research State

**Updated**: 2026-05-26 ~10:55Z | Branch: `tay` | SOTA: H112 PR #1283

---

## Primary Objective

**test_WSS < 5.85%** (Transolver-3 SOTA, per Morgan Issue #1056)

Current SOTA (H112): val_abupt 6.1358% / test_WSS 6.752% / test_WSS_z 8.720%
Gap to target: test_WSS needs −0.90pp improvement.

Merge gate: val_abupt < 6.1358%, test_WSS ≤ 6.727%, test_VP ≤ 3.421%, test_SP ≤ 3.577%

---

## Wave 38 Final Closure (as of 2026-05-26)

### THE PROGRAM-WIDE FINDING

**Any perturbation off the H112 operating point causes slope-flattening — NOT JUST CAPACITY ADDITIONS (H143 update 2026-05-26 10:55Z).**

**MAJOR UPDATE from H143 terminal C NULL (2026-05-26 10:55Z)**: H143 tau_z=4.0 loss-weight escalation (zero parameter overhead) produces SAME slope-flattening pathology as 3% overhead architecture changes. **The pathology is basin-geometry-driven, not overhead-driven.** Per-channel slope flattening was −0.130pp on WSS aggregate (H138 with 3% overhead: −0.135pp). All channels flattened, including non-escalated ones.

**Original Wave 38 finding (kept for history)**: Any capacity addition at canonical 17.5M recipe val-overfits.

All six Wave 38 mechanism classes exhausted:

| Class | Experiments | Verdict | Mechanism |
|---|---|---|---|
| Capacity-axis (depth/width) | H118, H120, H121, H125 | CLOSED | Val-overfit slope catastrophe |
| Pure regularization | H132 (DP_max=0.15) | CLOSED C NULL | Null — no improvement |
| SwiGLU gating | H128(OOM), H134(backbone), H135(decoder) | CLOSED | Val-overfit; param overhead acts as capacity expansion |
| Architectural-split | H138 (WSS_z decoder), H146 (WSS_y decoder trending) | CLOSED | Slope-flattening pathology — param overhead correlated. **H146 EP6 (corrected by student edward 11:30Z)**: val_abupt **+0.078pp LAG** vs H112 raw same-step (my 11:12Z W&B query used wrong baseline 7.111% vs actual 6.374%). H146 trajectory tracks H138 exactly — paired-class confirmed. WSS_y deficit smallest among channels (+0.033pp, real specialization) but +525K overhead drags everything. |
| Aux-head | H-B, H-B2 | CLOSED | Gradient-flow degradation; detached strictly worse |
| SOFTEN (loss) | H139 (Charbonnier), H140 (signed-log) | CLOSED | Anti-aligned with WSS_z goals; heavy-tail residuals ARE the signal |

**Param overhead correlates with slope flattening magnitude** (H132 null → H135 −0.10pp → H138 −0.135pp → H120/H121 −0.20pp). Slope catastrophe is overhead-driven, not mechanism-specific.

**Cross-channel bleed under Lion**: narrow-scope tau_z loss changes produce LARGER regression on tau_y (+0.356pp) than tau_z (+0.254pp) itself — via Lion sign-only update propagation through shared decoder weights.

---

## Wave 39 Active Fleet (2026-05-26 ~11:00Z)

| Student | PR | Hypothesis | Status | Priority |
|---|---|---|---|---|
| frieren | #1332 | H143 tau_z=4.0 ESCALATE | **CLOSED C NULL** (test_WSS_z +0.175pp REGRESSION; first non-overhead slope-flattening) | reassigned to H164 SWA |
| fern | #1334 | H144 tau_z=6.0 ESCALATE | WIP step 65,851 (93%) terminal ~12:30Z. val_WSS_z LEAD −0.084pp at step 65,212; all other channels regress (val_abupt +0.083pp, val_WSS +0.139pp, val_WSS_x +0.209pp). Identical cross-channel pattern to H143; HIGH RISK of slope flattening on test transfer. | Awaiting terminal; ESCALATE class verdict-pending |
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

### 1. ESCALATE class (H143/H144/H145) — H143 CLOSED, H144 HIGH RISK, H145 trending C NULL
- **H143 tau_z=4.0 CLOSED C NULL** (2026-05-26 10:55Z): test_WSS +0.203pp, test_WSS_z +0.175pp REGRESSION on target channel. val→test slope flattened 33% on WSS_z (basin-geometry pathology, not overhead-driven).
- **H144 fern (tau_z=6.0) HIGH RISK of identical pathology**: mid-train val signal was strong (LEAD −0.114pp at EP9) but val→test slope for H143 invalidated this projection. Terminal ~11:00Z is the verdict — likely C NULL given H143 precedent.
- **H145 alphonse (tau_y=3.0) trending C NULL**: 7-checkpoint mechanism alive on val_WSS_y but magnitude decaying geometrically (~50%/epoch). Terminal val_WSS_y ≈ 7.607-7.608% (coin-flip on banked-partial). val_abupt MISS by ~22bp projected.
- Mechanism: ESCALATE direction was hypothesized to push gradient pressure on hard channels. But static per-axis escalation produces basin-geometry pathology — same root cause as capacity/architecture interventions.
- **If H144 also closes C NULL, the entire ESCALATE class is exhausted.** Wave 40 frontier shifts decisively to basin-geometry-targeting mechanisms.

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

### 7. NEW Wave 40 hypothesis directly enabled by H147 EP3 — static tau_z DE-escalation
- **Hypothesis**: GradNorm-discovered optimal tau_z ≈ 1.6 plateau (H147 step 33,082 auto-weight 1.631) suggests the gradient-aligned optimum is BELOW H112's static 2.0. Try static tau_z=1.5 (single-flag DE-escalation).
- **Mechanism class**: pure loss-weight axis, but BELOW H112 baseline (qualitatively different from H143-H145 ESCALATE)
- **Falsifiability clean**: if alive at terminal, the entire H143/H144/H145 ESCALATE wave searched the wrong half of the magnitude space
- **Zero-overhead**, single CLI flag change. Compatible with all other Wave 40 mechanisms.
- **Priority**: Wave 40 HIGH — direct test of H147's most important banked finding.

### 8. Wave 40 contingency cascade (depends on H144 terminal)
- **H144 A WIN**: H144 × H145 joint escalation; H144 × H147 (escalate + GradNorm); continue magnitude curve to tau_z=8.0
- **H144 A TIED** (val PASS, test marginal MISS): H144 × H148 mirror compound; H144 × H147 GradNorm
- **H144 slope catastrophe** (most likely per H143 precedent): ESCALATE class DEAD. Pivot to: tau_z DE-escalation (priority 7), H149 AdamW (terminal pending), H157 cosine warm restarts, H164 SWA (frieren launched 12:02Z), fresh mechanism families. H150 EMA 0.9999 CLOSED — incompatible with 13-epoch training.

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
