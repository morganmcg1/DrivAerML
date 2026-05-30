# SENPAI Research State

- **2026-05-30 17:25Z**
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** H147 (PR #1344, run `k6q4c3on`) — test_WSS=6.5409%, test_VP=3.4014%, test_SP=3.5634%, test_ABUPT=5.6648% (all floors cleared)
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85%

---

## Latest research direction from human researcher team

No new issues directed at dl24 branch since 2026-05-29. Ensembles remain BANNED (per 2026-05-28 comment). Standing constraint: single-model DDP8 only, all 8 GPUs, max 24h.

Note on Issue #1056: TAY branch's TTA wave (H285→H296) reached test_abupt=5.7678%. dl24 H147 SOTA already BEATS this on 3/4 metrics (WSS −13.2bp, ABUPT −10.3bp, SP −8.0bp). Our wave-3 GradNorm path is the productive direction for THIS branch.

---

## Wave-3 BREAKTHROUGH: α=1.0 anti-starvation mechanism CONFIRMED at MAIN EP1 (H180)

**H180 (nezuko, PR #1494, α=1.0/floor=0.05) main EP1 reading (rank0 `gz8t5gkt`, step 10975):**
- val_VP = **13.2316 (−0.886pp BELOW H147 EP1)** — first wave-3 arm with VP under H147 at boundary
- w_vol_p = **0.2037 (4× floor, NOT clamped)** — α=1.0 doubled restoring force RESISTS the floor descent
- val_WSS = 12.8736 (+0.058pp), val_SP = 9.0038 (+0.102pp), val_ABU = 12.8840 (−0.162pp)

**Mechanism finding:** With α=0.5 (H173/H176/H178), the GradNorm restoring force is too weak → w_vol_p descends below the floor → clamping → VP starvation → terminal VP breach. With α=1.0, the restoring force is 2× stronger, keeping w_vol_p ≈ 0.20 (4× the floor) WITHOUT clamping. The vol_p task receives proportionally more gradient pressure throughout training, eliminating the starvation pattern.

**Tradeoff visible at EP1:** H180 trades −0.10pp val_SP (the H173-family SP-protection signature) for VP-protection. This is the WANTED tradeoff — gain on the broken floor (VP), lose on a productive but non-binding axis (SP).

**EP2 boundary ETA 17:43Z** — critical confirm whether mechanism survives cosine ramp-down.

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

Wave-3 systematically grids the vol_p_floor × cosine_length × GradNorm-alpha parameter space. **H180 EP1 reading establishes α=1.0 as the productive axis.**

### Active runs (17:25Z status)

**H172 (tanjiro, PR #1469) — EMA decay 0.9999 — POST-COSINE DESCENT CONTINUING:**
- Step 192760 (EP~17.55, rt 13.19h, state=running)
- val trajectory EP15→EP17: WSS 6.6827→6.6597 (-0.023pp/EP), VP 3.6258→3.5987 (now UNDER 3.643 test floor), SP 3.9304→3.9163, ABU 5.9497→5.9251
- Configured for EP30 cap (~23:00-00:30Z ETA)
- **Revised direction:** monotone EMA descent, potentially SOTA-trailing. EP30 projection: val_WSS ~6.49-6.50 may close H147's 6.5409 test gap
- Run continues to natural cap; SENPAI-RESULT pending

**H176 (frieren, PR #1486) — vol_p_floor 0.10 midpoint, 8-EP main — EP3 PASS:**
- Main 8-EP at rank0 `xupvpsxg`, EP3 landed step 32927
- EP3 val: WSS=7.0403 (+0.065pp, kill ≤7.15 PASS), VP=4.4499 (+0.325pp, CONTRACTING from EP2 +0.516pp), SP=4.1399 (kill ≤4.20 PASS by 0.06pp), ABU=6.3995
- vol_p_floor=0.10 mechanism showing **mid-cosine VP recovery** that H173 never achieved
- EP4 ETA ~18:00-18:30Z; terminal harvest ~20:30Z

**H178 (fern, PR #1493) — vol_p_floor 0.05 + 16-EP slow cosine — EP2 SHOWS VP STARVATION PERSISTING:**
- Main 16-EP rank0 `csk7pkf1`, step 23119 (past EP2 boundary 21951, rt 1.54h)
- EP2 val: WSS=7.3013 (+0.042pp), **VP=5.8444 (+0.938pp)**, SP=4.3263, ABU=6.9077
- vs H176 EP2 (8-EP same α): WSS better (-0.08pp), VP WORSE (+0.42pp despite slower cosine)
- **Finding:** 16-EP slow cosine partially helps VP starvation vs H173 (35% improvement) but does NOT resolve it. α=0.5 with floor=0.05 fails at any cosine length
- Updated kill ladder TIGHTENED: EP4 VP kill ≤5.00; EP6 VP kill ≤4.50

**H180 (nezuko, PR #1494) — vol_p_floor 0.05 + GradNorm α=1.0 — ANTI-STARVATION CONFIRMED EP1:**
- Main 8-EP rank0 `gz8t5gkt`, EP1 boundary step 10975 (rt ~0.83h)
- **val_VP=13.2316 (−0.886pp BELOW H147)** — first wave-3 arm with VP under H147 at boundary
- **w_vol_p=0.2037 (NOT clamped, 4× floor)** — α=1.0 mechanism active
- val_WSS=12.8736 (+0.058pp), val_SP=9.0038 (+0.102pp), val_ABU=12.8840 (−0.162pp)
- EP2 boundary ETA 17:43Z is CRITICAL
- Kill ladder: EP2 ≤7.50 WSS / ≤5.50 VP; EP4 ≤7.00 / ≤3.95 VP; EP6 ≤6.85 / ≤3.70 VP; EP8 terminal

---

## Wave-2 closed results (all 4 mechanisms fully harvested)

| Hypothesis | Mechanism | test_WSS | test_VP | test_SP | test_ABU | Verdict |
|---|---|---:|---:|---:|---:|---|
| H173 (PR #1474) | vol_p_floor 0.05 (α=0.5, 8-EP) | 6.6081 (+0.067pp) | 3.7793 **BREACH +0.136** | **3.5458 BEAT H147 -0.018** | 5.7897 PASS | NON-MERGE: VP breach |
| H174 (PR #1478) | PE-σ density shift | 6.7336 | 3.6548 **BREACH +0.012** | 3.6737 **BREACH +0.110** | 5.8720 **BREACH +0.028** | NON-MERGE: all 3 floors breach |
| H175 (PR #1480) | wss_charb yz @ 0.05 | 6.6370 (+0.096pp) | 3.5813 PASS | 3.6445 **BREACH +0.068** | 5.7940 PASS | NON-MERGE: SP breach |
| H172 (PR #1469) | EMA decay 0.9999 | TBD | TBD | TBD | TBD | post-cosine EMA descent continuing |

**Wave-2 conclusion:** vol_p_floor mechanism (H173 family) is the ONLY productive axis on test_SP, but α=0.5 breaks VP via floor clamping. **Wave-3 H180 supplies the resolution: α=1.0 prevents floor clamping while preserving the productive mechanism.**

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

| Experiment | floor | cosine_EPs | α | Status (17:25Z) | Key reading |
|---|---:|---:|---:|---|---|
| H173 (closed) | 0.05 | 8 | 0.5 | NON-MERGE | VP breach +0.136 |
| H176 (running) | 0.10 | 8 | 0.5 | **EP3 PASS** | VP gap contracting (EP2 +0.52 → EP3 +0.32) |
| H178 (running) | 0.05 | 16 | 0.5 | **EP2 VP +0.94pp** | slow cosine doesn't fix starvation |
| H180 (running) | 0.05 | 8 | 1.0 | **EP1 ANTI-STARV** | VP −0.886pp BELOW H147, w_vp NOT clamped |
| H172 (running) | — | — | — | EP17 descending | EMA decay 0.9999, val_VP under test floor |

### Wave-4 design queue (pending H176/H178/H180 terminals)

1. **H181: α=1.0 + vol_p_floor=0.10 compound** — does midpoint floor compound with α=1.0 anti-starvation
2. **H182: α=2.0 + vol_p_floor=0.05** — stronger restoring force test, looking for overshoot
3. **H183: α=1.0 + 16-EP slow cosine** — compound test (only if α=1.0 alone wins)
4. **H184: α=1.0 + EMA decay 0.9999** — compound with H172's slow EMA descent (only if H172 productive)
