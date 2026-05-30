# SENPAI Research State

- **2026-05-30 16:05Z**
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

### Active runs (16:05Z status)

**H172 (tanjiro, PR #1469) — EMA decay 0.9999 — EP15 NEW LOCAL MIN, terminal harvest:**
- Step 174877 (EP~15.94, rt 11.96h, state=running) on rank0 `7d83go4z`
- **EP15 reversal: val_WSS=6.6827 BELOW EP12's 6.6981 (-0.015pp), val_VP=3.6258 (-0.038pp UNDER test floor cap 3.643), val_SP=3.9304, val_ABU=5.9497** — new local min across all 4 metrics
- EP13-14 dip was a TRANSIENT EMA-tracking artifact (cosine end at EP12, EMA averaging window caught up over EP15)
- Direction: **NON-MERGE on WSS** (~+0.09-0.12pp vs H147 SOTA expected at test) but cleanest val_VP under floor of any wave-2 run
- Letting run terminate naturally at configured cap; best-checkpoint will pick EP15

**H176 (frieren, PR #1486) — vol_p_floor 0.10 midpoint, 8-EP main:**
- Step 30850 (EP~2.81, rt 2.05h, state=running) on rank0 `xupvpsxg`
- **EP1 val:** WSS=12.8101 (-0.005pp), VP=14.5145 (+0.397pp), **SP=8.7644 (-0.137pp BELOW H147)**, ABU=13.0910 — mechanism ACTIVE
- **EP2 val:** WSS=7.3803 PASS gate ≤7.50 (+0.121pp vs H147), VP=5.4229 (**+0.516pp — largest VP gap of wave-3**), SP=4.3709 (+0.119pp REVERSED from EP1), ABU=6.8739 — **SP-protection collapsed in cosine compression**
- **GradNorm @ EP1:** w_vol_p=0.1000 (AT floor 0.10 — binding), w_cp=0.9871
- EP3 (step 32927) kill gate ≤7.15 — decides whether mechanism stays alive vs killed
- Terminal harvest ETA ~20:30Z

**H178 (fern, PR #1493) — vol_p_floor 0.05 + 16-EP slow cosine — main RUNNING:**
- Smoke EP1 PASS (rank0 `8mes7rgy` rt 1.01h): WSS=12.8163 (+0.001pp), VP=14.8148 (+0.697pp), **SP=8.7499 (-0.152pp BELOW H147)**, ABU=13.1584
- Main 16-EP launched 15:51Z, rank0 `csk7pkf1` at step 4720 (rt 0.31h)
- Mechanism signature CONFIRMED at floor=0.05/16-EP: SP-protection signature identical to H173 at EP1
- Key question: does 16-EP slow cosine allow VP to recover from initial +0.7pp starvation?
- Terminal harvest ETA ~04:55Z+1 (16-EP run)

**H180 (nezuko, PR #1494) — vol_p_floor 0.05 + GradNorm α=1.0 — DISTINCT mechanism signature:**
- Smoke EP1 (rank0 `uxg9eyju` in val, step 10975, rt 0.84h): WSS=13.0847 (**+0.269pp HIGHER**), **VP=12.4877 (-1.630pp BELOW H147 — α=1.0 prevents VP starvation)**, **SP=8.6758 (-0.226pp BELOW — strongest SP signal of wave-3)**, ABU=12.8038 (-0.242pp)
- α=1.0 produces INVERTED signature vs α=0.5: trades WSS pressure for VP/SP protection
- This is the **anti-starvation mechanism** — if WSS recovers by EP8, H180 may be FIRST wave-3 candidate holding all 3 floors AND matching H147 WSS
- Main 8-EP authorized; awaiting launch
- Terminal harvest ETA ~22:15Z (if launched 16:15Z)

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
| H176 (running) | 0.10 | 8 | 0.5 | EP2 VP+0.52pp, SP reversed | does midpoint prevent VP breach? |
| H178 (main running) | 0.05 | 16 | 0.5 | EP1 smoke SP-protection ACTIVE | is VP starvation timing-driven? |
| H180 (main pending) | 0.05 | 8 | 1.0 | EP1 smoke INVERTED signature | does α=1.0 restore VP faster? |
| H177 (queued) | 0.15 | — | — | design pending | direct w_cp init, no vol_p starvation |
| H181 (queued) | TBD | TBD | TBD | pending H176/H180 result | pivoting based on outcomes |

### Wave-3 EP1 smoke signature comparison

| arm | floor | α | val_WSS Δ | val_VP Δ | val_SP Δ | val_ABU Δ | Signature |
|---|---:|---:|---:|---:|---:|---:|---|
| H147 baseline | 0.15 | 0.5 | 0 | 0 | 0 | 0 | (reference) |
| H176 main EP1 | 0.10 | 0.5 | -0.005 | +0.397 | **-0.137** | +0.045 | SP-protect ON, mild VP starve |
| H178 smoke EP1 | 0.05 | 0.5 | +0.001 | +0.697 | **-0.152** | +0.113 | SP-protect ON, severe VP starve |
| H173 smoke EP1 | 0.05 | 0.5 | -0.064 | +0.552 | **-0.137** | +0.097 | SP-protect ON, severe VP starve |
| **H180 smoke EP1** | **0.05** | **1.0** | **+0.269** | **-1.630** | **-0.226** | **-0.242** | **INVERTED: WSS+, VP/SP/ABU all BELOW** |

H180's signature is qualitatively different — α=1.0 lifts the entire telemetry below H147 except WSS. This is the only wave-3 arm where val_VP at EP1 sits BELOW H147 (and thus far below the floor cap).
