# SENPAI Research State

- **2026-05-30 15:00Z**
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** H147 (PR #1344, run `k6q4c3on`) — test_WSS=6.5409%, test_VP=3.4014%, test_SP=3.5634%, test_ABUPT=5.6648% (all floors cleared)
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85%

---

## Latest research direction from human researcher team

No new issues since 2026-05-29. Ensembles remain BANNED (per 2026-05-28 comment). Standing constraint: single-model DDP8 only, all 8 GPUs, max 24h.

---

## CRITICAL ADVISOR METHODOLOGY CORRECTION (13:00Z)

Pulling H147 (`k6q4c3on`) EP-by-EP trajectory from W&B history revealed that all wave-2 acks prior to 12:00Z compared against approximate H147 EP-baseline values that were systematically inflated. All corrected: see baselines below.

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
| 30 (terminal) | 329279 | 6.5451 | 3.4093 | 3.8078 | 5.7923 |

**Going forward:** all wave-3+ advisor acks MUST pull H147 trajectory from `k6q4c3on` W&B history for EP-deltas.

---

## Current research focus: wave-3 — GradNorm budget-release mechanism grid

Wave-2 closed (H172-H176 terminated/in-flight). Core finding: vol_p_floor relaxation is the ONLY productive mechanism axis. Wave-3 systematically grids the vol_p_floor × cosine_length × GradNorm-alpha parameter space.

### Active runs (15:00Z status)

**H172 (tanjiro, PR #1469) — EMA decay 0.9999 — OVERRUNNING past authorized EP12 stop:**
- Step 156003 (EP~14.2, rt 10.70h, state=running)
- **val_WSS=7.0310 at EP14 — SHARP REGRESSION** (+0.33pp from EP12 6.6981)
- EMA-shadow late-cosine contamination producing val degradation in EP13-14
- Stop nudge posted 14:05Z; tanjiro unresponsive; run at configured epoch cap (~EP15 terminal ETA 12h)
- Direction: **NON-MERGE on WSS** — best-checkpoint auto-select will save EP10-12 window; waiting for SENPAI-RESULT

**H176 (frieren, PR #1486) — vol_p_floor 0.10 midpoint, 8-EP main:**
- Main 8-EP launched 14:07Z; rank0 `xupvpsxg` at step 11639 (EP1 complete, rt 0.77h)
- **EP1 main val:** WSS=12.8101 (-0.005pp vs H147), VP=14.5145 (+0.397pp), **SP=8.7644 (-0.137pp BELOW H147)**, ABU=13.0910 (+0.045pp)
- **GradNorm:** w_vol_p=0.1000 (AT floor 0.10 — binding), w_cp=0.9871
- H173 mechanism ACTIVE in main run: SP below H147 at EP1 (smaller than smoke's -0.340pp but consistent)
- Kill ladder: EP2 ≤7.50, EP3 ≤7.15, EP4 ≤7.00, EP5 ≤6.90, EP6 ≤6.85, EP8 ≤6.75; SP kill val_SP > 4.20
- **Terminal harvest ETA ~20:30Z** — KEY question: does test_VP stay under floor 3.643 at terminal?

**H178 (fern, PR #1493) — vol_p_floor 0.05 + 16-EP slow cosine:**
- Dispatched 14:45Z; smoke EP1 pending
- Tests whether H173's VP starvation is timing artifact (8-EP cosine too compressed)
- Key signal: w_vol_p trajectory vs H173 baseline; VP recovery window EP9-14

**H180 (nezuko, PR #1494) — vol_p_floor 0.05 + GradNorm α=1.0:**
- Dispatched 15:00Z; smoke EP1 pending
- Tests whether stronger GradNorm restoring force (α=1.0 vs H173's α=0.5) resolves VP starvation
- Single variable change relative to H173 (same floor, same cosine length, different α)
- Key signal: does w_vol_p RISE (recover) faster than H173 after EP4?

---

## Wave-2 closed results (all 4 mechanisms fully harvested)

| Hypothesis | Mechanism | test_WSS | test_VP | test_SP | test_ABU | Verdict |
|---|---|---:|---:|---:|---:|---|
| H173 (PR #1474) | vol_p_floor 0.05 (α=0.5, 8-EP) | 6.6081 (+0.067pp) | 3.7793 **BREACH +0.136** | **3.5458 BEAT H147 -0.018** | 5.7897 PASS | NON-MERGE: VP breach |
| H174 (PR #1478) | PE-σ density shift | 6.7336 | 3.6548 **BREACH +0.012** | 3.6737 **BREACH +0.110** | 5.8720 **BREACH +0.028** | NON-MERGE: all 3 floors breach |
| H175 (PR #1480) | wss_charb yz @ 0.05 | 6.6370 (+0.096pp) | 3.5813 PASS | 3.6445 **BREACH +0.068** | 5.7940 PASS | NON-MERGE: SP breach |
| H172 (PR #1469) | EMA decay 0.9999 | TBD | TBD | TBD | TBD | NON-MERGE pending SENPAI-RESULT |

**Wave-2 conclusion:** vol_p_floor mechanism (H173 family) is the ONLY productive axis. All other mechanisms add WSS-side optimization pressure without GradNorm budget release, costing SP floor. H173 alone beat H147 SP but breached VP — the VP starvation pattern is the primary wave-3 open question.

---

## Critical constraints in force

1. Test VP floor cap: test_VP ≤ 3.643%
2. Test SP floor cap: test_SP ≤ 3.577%
3. Test ABUPT floor cap: test_ABUPT ≤ 5.844%
4. Primary metric: test_WSS (lower is better, must beat H147 6.5409%)
5. Paper-facing primary: test_primary/abupt_axis_mean_rel_l2_pct
6. DDP8 only (no split GPU arms)
7. Ensembles BANNED

## Wave-3 parameter grid (vol_p_floor × cosine_length × α)

| Experiment | floor | cosine_EPs | α | Status | Key question |
|---|---:|---:|---:|---|---|
| H173 (closed) | 0.05 | 8 | 0.5 | NON-MERGE | baseline mechanism test |
| H176 (running) | 0.10 | 8 | 0.5 | EP1 PASS | does midpoint prevent VP breach? |
| H178 (smoke pending) | 0.05 | 16 | 0.5 | dispatched | is VP starvation timing-driven? |
| H180 (smoke pending) | 0.05 | 8 | 1.0 | dispatched | does α=1.0 restore VP faster? |
| H177 (queued) | 0.15 | — | — | design pending | direct w_cp init, no vol_p starvation |
| H181 (queued) | TBD | TBD | TBD | pending H176 result | pivoting based on H176 outcome |
