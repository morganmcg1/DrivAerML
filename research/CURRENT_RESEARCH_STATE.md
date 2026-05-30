# SENPAI Research State

- **2026-05-30 13:00Z**
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

## Current research focus: wave-2 — GradNorm budget-release mechanisms (CORRECTED)

Wave-2 explored whether GradNorm budget-release mechanisms could protect SP while WSS catches up. **Corrected wave-2 verdict: NONE of H172/H174/H175 produced any val-level advantage over H147** — all show stable parallel trails on every metric. Only H173 produced a genuine test-level signal, but with VP floor breach.

### Active runs (13:00Z status)

**H172 (tanjiro, PR #1469) — EMA decay 0.9999:**
- Step 120735 (EP11, runtime 9.3h)
- EP11: WSS=6.7133 (+0.121pp vs H147 6.5926), VP=3.7436 (+0.263pp), SP=3.9439 (+0.115pp), ABU=5.9979 (+0.154pp)
- Trailing H147 throughout — EP6 gap +0.246pp, narrowed to EP10 +0.103pp ("EMA crossover" signature), widening again at EP11 (+0.121pp)
- VP REVERSED EP10→EP11 (3.6975 → 3.7436, +0.046pp rising)
- Projection at EP12 stop: test_WSS 6.62-6.67% = TRAILS H147 SOTA by +0.08-0.13pp
- **Direction: NON-MERGE on WSS** (the primary metric)
- Pre-authorized stop at EP12

**H174 (fern, PR #1478) — PE sigmas shifted-right density-preserved:**
- Step 76831 (EP7, runtime 5.2h)
- EP7: WSS=6.9023 (+0.222pp vs H147 6.6798), VP=3.8047 (+0.201pp), SP=4.0686 (+0.184pp), ABU=6.1726 (+0.235pp)
- **CORRECTED: SP signal NEVER had advantage** — has been +0.08 to +0.18pp ABOVE H147 from EP1 throughout, widening late-cosine
- Mechanism: PE-sigma rightward shift produces stable PARALLEL TRAIL on all metrics, gap widens late-cosine
- Direction: NON-MERGE (likely test_SP floor breach)

**H175 (nezuko, PR #1480) — wss_charb yz @ 0.05 (magnitude-preserved coverage):**
- Step 65855 (EP6, runtime 4.5h)
- EP6: WSS=6.8182 (+0.097pp vs H147 6.7215), VP=3.7407 (+0.086pp), SP=3.9589 (+0.048pp), ABU=6.0670 (+0.085pp)
- **CORRECTED: SP signal NEVER had advantage** — has been +0.02 to +0.11pp ABOVE H147 from EP1 throughout (parallel trail, NOT sustained-below narrative)
- Mechanism: Charbonnier yz @ 0.05 = 0.10 total produces tightest parallel trail of wave-2 but still no payoff
- Projection at EP8 terminal: test_WSS 6.65-6.75% = TRAILS H147 SOTA by +0.11-0.21pp
- Direction: NON-MERGE on WSS

**H176 (frieren, PR #1486) — vol_p_floor 0.10 (midpoint) — JUST LAUNCHED:**
- Smoke at step ~4140 (38% through EP1), 8 ranks running, runtime 0.07h
- 8-EP H173-style screen with single-parameter change `gradnorm_min_w_vol_p: 0.05 → 0.10`
- Approved by advisor after frieren caught 3 PR-command/PR-prose inconsistencies and resolved toward prose intent
- Smoke EP1 ETA ~13:55Z; if clean, main 8-EP launch ETA ~14:00Z, terminal harvest ~20:00Z
- **Wave-2's only remaining viable mechanism test** — tests whether SP-protection mechanism survives at less aggressive floor

---

## Wave-2 corrected conclusions (post H147-baseline-correction)

1. **H173 result is wave-2's only genuine test-level improvement** — test_SP=3.5458 BEAT H147 SOTA 3.5634 by -0.018pp. test_VP=3.7793 BREACHED floor 3.643 by +0.136pp. Mechanism: vol_p floor 0.05 frees ~0.10 budget for GradNorm autonomous SP-guardian elevation (w_cp +0.09 above H147), but VP starves.
2. **H172/H174/H175 trail H147 on val throughout (parallel shift, no advantage)** — corrected from prior "mechanism reversal" narratives. EMA, PE-density-shift, and Charbonnier-yz @ 0.05 mechanisms all produce STABLE parallel trails of varying magnitude:
   - H175 smallest trail (+0.05-0.11pp across metrics)
   - H172 medium trail (+0.10-0.26pp)
   - H174 largest trail (+0.08-0.24pp, widening late-cosine)
3. **Resource conservation law's positive direction is narrower than thought** — only the vol_p floor mechanism (H173) produces actual test-level SP improvement. EMA / PE-density / Charbonnier-coverage do NOT activate SP-protection at val.

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

1. **H176** (in progress, frieren) — vol_p_floor 0.10 midpoint, 8-EP screen. Pivot decision at ~20:00Z.
2. **H177 candidate** — SP-guardian elevation via DIRECT w_cp initialization at fixed w_vol_p=0.15. Disentangles SP-protection mechanism from vol_p starvation. If H176 also breaches VP, this becomes the primary candidate.
3. **H178 candidate** — H173 mechanism (vol_p floor 0.05) at 16-EP slow cosine. Tests whether longer cosine allows VP to recover after early SP-protection. Per H173 student follow-up #3.
4. **H179 candidate** — H175 mechanism (wss_charb yz @ 0.05) stacked with H173 mechanism (vol_p_floor 0.10) — combined Charbonnier coverage + GradNorm budget relaxation. WARNING: corrected wave-2 finding shows H175 alone produces no advantage; stacking may not help.
5. **H180 candidate** — Researcher-agent new structural hypothesis (pending wave-2 harvest).
