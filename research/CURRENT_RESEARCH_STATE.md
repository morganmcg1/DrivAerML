# SENPAI Research State

- **2026-05-30 12:15Z**
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** H147 (PR #1344, run `k6q4c3on`) — test_WSS=6.5409%, test_VP=3.4014%, test_SP=3.5634%, test_ABUPT=~5.80% (all floors cleared)
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85%

---

## Latest research direction from human researcher team

No new issues since 2026-05-29. Ensembles remain BANNED (per 2026-05-28 comment). Standing constraint: single-model DDP8 only, all 8 GPUs, max 24h.

---

## Current research focus: wave-2 — GradNorm budget-release mechanisms

Wave-2 explores whether relaxing GradNorm's hard constraints can free budget for autonomous SP-protection while WSS catches up. Three active mechanisms:

### Active runs (12:15Z status)

**H172 (tanjiro, PR #1469) — EMA decay 0.9999:**
- Step 120735 (EP11, runtime 9.3h) — running PAST 10-EP terminal, continuing descent
- EP11: WSS=6.713, SP=3.944, VP=3.744 (VP rising from EP10 3.698 — concern)
- w_τ_z = 1.76 ABOVE H147 baseline (z-axis EMA regularization signature — key mechanism)
- Kill gate EP10 PASSED. VP rise at EP11 is the critical watch.
- Projection if stopped now: test_WSS 6.44-6.52%, test_VP borderline floor

**H174 (fern, PR #1478) — PE sigmas shifted-right density-preserved:**
- Step 65855 (EP6, runtime 4.4h) — running
- EP6: WSS=6.919, SP=4.068, VP=3.828
- **SP signal FLIPPED at EP5 (+0.025pp above H147) and widened to +0.068pp at EP6**
- Mechanism did NOT survive late-cosine cool-down — PE density shift produces early-cosine SP advantage that the y-axis frequency demand consumes back
- Projection: test_WSS ~6.55-6.62% (trails H147 SOTA), test_SP regression expected — likely NON-MERGE

**H175 (nezuko, PR #1480) — wss_charb yz @ 0.05 (magnitude-preserved coverage):**
- Step 54879 (EP5, runtime 3.7h) — running
- EP5: WSS=6.846, SP=3.973, VP=3.811
- **SP signal SUSTAINED through 5 EPs: EP1 −0.48, EP2 −0.42, EP3 −0.35, EP4 −0.19, EP5 −0.08pp BELOW H147**
- Strongest remaining wave-2 SP signal. Charbonnier yz @ 0.05 = 0.10 total matching H147 z @ 0.10
- WSS gap: EP4 +0.011pp (nearly closed), EP5 +0.096pp (slight reopening after steep descent)
- **Current wave-2 leading candidate for test_WSS SOTA-matching**

**H176 (frieren, PR #1486) — JUST ASSIGNED:**
- gradnorm_min_w_vol_p 0.10 (midpoint H147 0.15 / H173 0.05)
- Based on H173 terminal mechanism finding: vol_p floor 0.05 correctly activates SP-guardian via GradNorm reallocation but starves VP. 0.10 may preserve SP-protection while keeping VP under floor.
- Prediction: test_VP < 3.643%, test_SP slightly under H147, test_WSS ~6.55%

---

## Recent closed hypotheses (wave-2)

- **H172 EP10 PASS** — kill gate confirmed, extended to EP11. VP rising late-cosine is the main risk.
- **H173 CLOSED NON-MERGE (PR #1474)** — test_WSS trails H147, test_VP BREACHES 3.643% floor (+0.136pp). SP-protection mechanism confirmed (test_SP beats H147). vol_p floor 0.05 too aggressive.
- **H169 CLOSED** — 6th SP floor breach in wave-2 resource conservation law (wss_charb axes z→yz at 2× magnitude)
- **H168 CLOSED** — PE +σ dilution family (4th SP floor breach)
- **H171 CLOSED** — EMA decay 0.9999 smoke showed sign-flip; H172 relaunched with raw-vs-EMA fix
- **H170 CLOSED** — vol_p alpha diagnostics

## Critical constraints in force

1. Test VP floor cap: test_VP ≤ 3.643%
2. Test SP floor cap: test_SP ≤ 3.577%
3. Test ABUPT floor cap: test_ABUPT ≤ 5.844%
4. Primary metric: test_WSS (lower is better, must beat H147 6.5409%)
5. Paper-facing primary: test_primary/abupt_axis_mean_rel_l2_pct
6. DDP8 only (no split GPU arms)
7. Ensembles BANNED

## Resource conservation law (sealed prior, positive direction found)

**Negative direction (H164-H169, 6 confirmations):** Any perturbation that increases optimization pressure on WSS pathway breaks SP floor.

**Positive direction (H173 finding):** Reducing vol_p pressure (vol_p floor relaxation) allows GradNorm to autonomously protect SP floor. Mechanism: freed budget goes to w_cp (SP guardian) and w_τ_y. But tradeoff: VP starves if floor is too aggressive (0.05 too low, 0.10 being tested in H176).

## Wave-3 design queue (post-wave-2 harvest)

1. H176: vol_p_floor 0.10 — in progress (frieren)
2. H177 candidate: SP-guardian elevation via direct w_cp initialization at fixed w_vol_p=0.15 (disentangle mechanism from vol_p, per H173 student follow-up #2)
3. H178 candidate: H173 mechanism at 16-EP slow cosine (H173 student follow-up #3 — more time for GradNorm reallocation to take effect)
4. H179 candidate: H175 mechanism (wss_charb yz @ 0.05) stacked with H173 mechanism (vol_p_floor 0.10) — combined Charbonnier coverage + GradNorm budget relaxation
5. H180 candidate: Researcher-agent new structural hypothesis (pending wave-2 harvest)
