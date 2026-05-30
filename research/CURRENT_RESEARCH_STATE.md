# SENPAI Research State

- **2026-05-30 14:45Z**
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** H147 (PR #1344, run `k6q4c3on`) — test_WSS=6.5409%, test_VP=3.4014%, test_SP=3.5634%, test_ABUPT=~5.80% (all floors cleared)
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85%

---

## Latest research direction from human researcher team

No new issues since 2026-05-29. Ensembles remain BANNED (per 2026-05-28 comment). Standing constraint: single-model DDP8 only, all 8 GPUs, max 24h.

---

## CRITICAL ADVISOR METHODOLOGY CORRECTION (13:00Z)

Pulling H147 (`k6q4c3on`) EP-by-EP trajectory from W&B history this cycle revealed that all wave-2 acks compared against approximate H147 EP-baseline values that were systematically inflated by 0.08-0.30pp on SP (and similar on other metrics). **The wave-2 mechanism narratives ("density-preservation early-cosine advantage", "magnitude-preservation SP sustained 5-EP", etc.) were artifacts of bad H147 reference values, not real findings.** Corrected baselines:

**H147 actual EP boundaries (from k6q4c3on val history):**

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
| (terminal, step 329279) | 30 | 6.5451 | 3.4093 | 3.8078 | 5.7923 |

H147 val terminal at EP30 = 6.5451 WSS. test_WSS=6.5409 (~0.004pp lower). H147 val SP at EP30 = 3.8078; test_SP=3.5634 (test SP much lower than val SP due to dataset split character). H147 val VP at EP30 = 3.4093; test_VP=3.4014.

**Going forward:** all wave-3+ advisor acks MUST pull H147 trajectory from `k6q4c3on` W&B history for EP-deltas. No approximate baselines.

---

## Current research focus: wave-2 closing + wave-3 dispatch

Wave-2 full closure expected by ~20:00Z (H176 main run terminal). Wave-3 dispatched: H178 (fern, PR #1493).

### Active runs (14:45Z status)

**H172 (tanjiro, PR #1469) — EMA decay 0.9999 — OVERRUNNING:**
- Step 152859 (EP14 imminent at 153650, rt 10.47h, state=running)
- EP12 stop authorized 13:32Z, stop-nudge posted 14:05Z — tanjiro unresponsive; run will auto-terminate at configured epoch cap (likely EP15)
- EP12 val (last complete EP): WSS=6.6981 (+0.124pp vs H147 6.5742), VP=3.6633 (+0.189pp), SP=3.9367 (+0.111pp), ABU=5.9702 (+0.141pp)
- Best-checkpoint auto-select active; terminal test eval pending natural completion ~11.5-12h total
- Direction: **NON-MERGE on WSS** — gap won't close; waiting for SENPAI-RESULT
- Note: VP reversed back down at EP12 (3.6633, down from EP11 peak) — floor breach risk reduced

**H175 (nezuko, PR #1480) — wss_charb yz @ 0.05 — TERMINAL:**
- EP8 terminal at step 87807 (rt 6.16h, state=finished, rank0 `o5jmdw3q`)
- EP8 val: WSS=6.7722 (+0.122pp), VP=3.6843 (+0.125pp), SP=3.9442 (+0.079pp)
- **Test results (W&B): test_WSS=6.6370 (+0.096pp TRAILS), test_VP=3.5813 (PASS ✓), test_SP=3.6445 (BREACH +0.068pp ✗), test_ABU=5.7940 (PASS ✓)**
- Direction: **NON-MERGE on WSS + SP floor BREACH** — terminal ack posted 14:45Z, closing PR pending SENPAI-RESULT from nezuko

**H176 (frieren, PR #1486) — vol_p_floor 0.10 midpoint — EP1 IN FLIGHT:**
- Main 8-EP run launched 14:07Z; rank0 `xupvpsxg` at step 7007 (EP1 ~64%, rt 0.46h)
- All 8 DDP ranks running; EP1 boundary ~14:55Z (val only on rank0)
- Terminal harvest ETA ~20:30Z
- **Wave-2's critical remaining mechanism test** — key question: does test_VP stay under floor at terminal with floor=0.10 vs floor=0.05?
- Kill ladder: EP2 ≤7.50, EP3 ≤7.15, EP4 ≤7.00, EP5 ≤6.90, EP6 ≤6.85, EP8 ≤6.75; SP kill val_SP > 4.20

**H178 (fern, PR #1493) — vol_p_floor 0.05 + 16-EP slow cosine — SMOKE PENDING:**
- Dispatched 14:45Z, branch `dl24-fern/h178-vol-p-floor-005-slow-cosine-16ep`
- Hypothesis: H173 mechanism at 16-EP cosine tests whether VP starvation was timing artifact (not fundamental VP-SP trade-off)
- Smoke EP1 first; main 16-EP launch (~12h) after smoke PASS
- Key GradNorm signal: w_vol_p trajectory EP1-8 vs H173 baseline; VP recovery window EP9-14
- Terminal ETA (if smoke+main): ~02:00Z 2026-05-31

---

## Wave-2 corrected conclusions (post H147-baseline-correction)

1. **H173 result is wave-2's only genuine test-level improvement** — test_SP=3.5458% BEAT H147 SOTA 3.5634% by -0.018pp. test_VP=3.7793% BREACHED floor 3.643% by +0.136pp. Mechanism: vol_p floor 0.05 frees ~0.10 budget for GradNorm autonomous SP-guardian elevation (w_cp +0.09 above H147), but VP starves.
2. **H172/H174/H175 trail H147 on val throughout (parallel shift, no advantage)**:
   - H174 (PE-σ density shift): ALL 3 test floor caps BREACHED (test_VP +0.012, test_SP +0.110, test_ABU +0.028) + WSS regress — CLOSED NON-MERGE
   - H175 (Charbonnier yz @ 0.05): test_VP PASS, test_ABU PASS, **test_SP BREACH +0.068pp**, WSS regress — CLOSING NON-MERGE
   - H172 (EMA 0.9999): terminal pending, all EP readings trail H147 by +0.10-0.25pp — NON-MERGE
3. **Resource conservation law:** Any perturbation that adds optimization pressure on WSS pathway costs SP floor. The vol_p floor mechanism (H173 family) is the ONLY wave-2 axis that moves w_cp in the SP-protective direction.
4. **VP starvation timing hypothesis** (to be tested by H178): H173's 8-EP cosine may not give GradNorm enough time to recover VP after initial SP-protection surge. 16-EP cosine may provide VP-recovery window in EP9-14.

---

## Critical constraints in force

1. Test VP floor cap: test_VP ≤ 3.643%
2. Test SP floor cap: test_SP ≤ 3.577%
3. Test ABUPT floor cap: test_ABUPT ≤ 5.844%
4. Primary metric: test_WSS (lower is better, must beat H147 6.5409%)
5. Paper-facing primary: test_primary/abupt_axis_mean_rel_l2_pct
6. DDP8 only (no split GPU arms)
7. Ensembles BANNED

## Wave-3 design queue (post-wave-2 harvest, prioritized by corrected findings)

1. **H178** (dispatched, fern PR #1493) — vol_p_floor 0.05 at 16-EP slow cosine. Tests VP-starvation timing hypothesis from H173.
2. **H176** (in progress, frieren PR #1486) — vol_p_floor 0.10 midpoint at 8-EP screen. Orthogonal to H178 (floor midpoint vs cosine length).
3. **H177 candidate** — SP-guardian elevation via DIRECT w_cp initialization at fixed w_vol_p=0.15. Disentangles SP-protection mechanism from vol_p starvation. Primary if both H176 AND H178 confirm VP breach is unavoidable at floor=0.05.
4. **H179 candidate** — H175 mechanism (wss_charb yz @ 0.05) stacked with H173 mechanism (vol_p_floor 0.10) — combined Charbonnier coverage + GradNorm budget relaxation. WARNING: corrected wave-2 finding shows H175 alone produces no advantage; stacking may not help on test_SP.
5. **H180 candidate** — Researcher-agent structural hypothesis (pending wave-2 full harvest at ~20:00Z).
6. **Nezuko assignment pending** — after H175 close; will dispatch wave-3 next cycle.
