# SENPAI Research State

- **2026-05-30 20:25Z**
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** H147 (PR #1344, run `k6q4c3on`) — test_WSS=6.5409%, test_VP=3.4014%, test_SP=3.5634%, test_ABUPT=5.6648% (all floors cleared)
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85%

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

## H172 EP20 — first run to undercut H147 EP6 reference (most consequential active result)

**H172 (tanjiro, PR #1469, EMA decay 0.9999) at step 220346, EP~20, rt ~15.0h, `7d83go4z`:**

| Metric | H172 EP20 (val) | H147 EP6 (val) | H147 EP12 (val) | H147 EP30 terminal (test) |
|---|---:|---:|---:|---:|
| WSS | **6.6521** | 6.7215 | 6.5742 | 6.5409 |
| VP | **3.5892** | 3.6549 | 3.4743 | 3.4014 |
| SP | 3.9143 | 3.9107 | 3.8259 | 3.5634 |
| ABU | **5.9180** | 5.9817 | 5.8288 | 5.6648 |

**EP15→17 slope on val_WSS was −0.023pp/EP, but EP17→20 slowed to ~−0.0025pp/EP — descent is decelerating.** EP30 val_WSS projection: ~6.60-6.62 → test_WSS likely +0.06-0.08pp above H147 SOTA. **Closest active candidate to H147 but unlikely to beat outright** unless EMA-descent re-accelerates in the cooldown phase.

H172 is the **only mechanism in flight that has materially undercut H147 mid-train references**. Wave-4 should prioritize EMA-derivative experiments.

---

## Current research focus: wave-3 GradNorm-α grid — verdict converging

Wave-3 was constructed to probe whether GradNorm restoring force (α) or floor placement could resolve the H173 VP-starvation pattern while preserving SP-protection. With H180 corrected and H178 starvation persisting, the grid verdict is settling:

### Active runs (19:15Z status)

**H172 (tanjiro, PR #1469) — EMA decay 0.9999 — EP20 BELOW H147 EP6 references:**
- val_WSS=6.6521, val_VP=3.5892, val_SP=3.9143, val_ABU=5.9180
- Monotone descent through EP20 but slope decelerating EP17→EP20
- ETA EP30 natural cap: ~23:00-00:30Z
- Verdict pending terminal — most promising active arm

**H176 (frieren, PR #1486) — CLOSED NON-MERGE 20:24Z — `xupvpsxg`:**
- test_WSS=6.6790 (+0.138pp vs H147), test_VP=3.6646 (+0.022pp BREACH cap), test_SP=3.6616 (+0.085pp BREACH cap), test_ABU=5.8256 PASS
- Worst-of-both-worlds: lost H173's SP-protection (collapsed by EP2) AND failed to fix VP starvation
- Frontier finding: w_τ_y emerged as dominant absorber of freed budget (1.29→1.53 across 8 EPs), suggesting τ_y-pin as a future mechanism axis
- **Wave-4 H181 (EMA decay 0.99995) DISPATCHED to frieren (PR #1503)**

**H178 (fern, PR #1493) — vol_p_floor 0.05, α=0.5, 16-EP — EP9-ish:**
- Step 53625, rt 3.35h, val_WSS=6.96, **val_VP=4.46 (+0.73 vs H147 EP5 ref step 54879)**
- Worst VP performer of all three wave-3 floor-0.05 arms at comparable step
- ETA EP16 terminal: ~21:30-22:00Z

**H180 (nezuko, PR #1494) — vol_p_floor 0.05, α=1.0, 8-EP — EP4 at parity on VP, persistent WSS regression:**
- Step 46703, rt 2.93h, val_WSS=6.9241 (+0.09 vs H147), val_VP=3.8979 (+0.04), val_SP=4.0881 (+0.13)
- w_vol_p=0.1523 (3× floor, NOT clamped) — mechanism prevents floor collapse
- α=1.0 stabilizes VP near parity but WSS regression dominates terminal outcome
- ETA EP8 terminal: ~22:00Z

---

## Wave-3 grid status (as of 19:15Z)

| Experiment | floor | cosine_EPs | α | Status | Verdict |
|---|---:|---:|---:|---|---|
| H173 (closed) | 0.05 | 8 | 0.5 | NON-MERGE | VP breach +0.136 from floor clamping |
| H176 (closed) | 0.10 | 8 | 0.5 | NON-MERGE 20:24Z | test_WSS +0.138; VP BREACH +0.022; SP BREACH +0.085 (worst-of-both) |
| H178 (running) | 0.05 | 16 | 0.5 | EP9 VP +0.73 | 16-EP doesn't fix starvation |
| H180 (running) | 0.05 | 8 | 1.0 | EP4 VP parity | α=1.0 prevents clamping; WSS +0.09 persistent |
| H172 (running) | — | — | — | EP20 below H147 EP6 ref | EMA decay 0.9999, most productive in wave |

**Wave-3 conclusion (provisional, pending H172 terminal):** GradNorm-α grid does NOT produce a H147-beater. The α=1.0 + floor=0.05 combination resolves the H173 clamping pathology but introduces a persistent +0.05-0.09pp WSS regression. The productive lever in this wave is **EMA-0.9999 (H172)**, not GradNorm-α.

---

## Wave-4 design queue (REVISED, EMA-derivative focused)

Revised based on the H180 walkback and H172 leadership. Drop α-grid follow-ups; pivot to EMA + LR + cosine combinations.

1. **H181: EMA decay 0.99995** (frieren, after H176 terminal ~20:00Z) — push past H172's 0.9999 to see if longer averaging window extracts more descent
2. **H182: H172 stack + LR 1.3× peak** (nezuko, after H180 terminal ~22:00Z) — does higher peak LR + EMA 0.9999 give deeper minima
3. **H183: H172 stack + cosine 30→40 EP within 24h budget** (fern, after H178 terminal ~21:30Z) — extended cosine descent with EMA averaging
4. **H184: H172 stack + structural perturbation** (tanjiro, after H172 terminal ~00:00Z) — compound EMA with one architectural change (e.g., attention heads, layer count)

Pre-condition: H172 must finish productive (val_WSS ≤ 6.62 at EP30). If H172 terminal is WSS ≥ 6.65, EMA-derivative thesis weakens and we revert to architecture/data-representation exploration.

---

## Critical constraints in force

1. Test VP floor cap: test_VP ≤ 3.643%
2. Test SP floor cap: test_SP ≤ 3.577%
3. Test ABUPT floor cap: test_ABUPT ≤ 5.844%
4. Primary metric: test_WSS (lower is better, must beat H147 6.5409%)
5. Paper-facing primary: test_primary/abupt_axis_mean_rel_l2_pct
6. DDP8 only (no split GPU arms)
7. Ensembles BANNED

## Terminations ETA cluster (remaining)

| Run | PR | Student | ETA | Action |
|---|---|---|---|---|
| H176 | #1486 | frieren | ✓ CLOSED 20:24Z | H181 EMA 0.99995 dispatched (PR #1503) |
| H178 | #1493 | fern | ~21:30Z | NON-MERGE VP breach expected; dispatch H183 EMA+extended-cosine (cap ≤32 EP) |
| H180 | #1494 | nezuko | ~22:00Z | NON-MERGE WSS regression; dispatch H182 EMA+LR 1.3× |
| H172 | #1469 | tanjiro | ~23:00-00:30Z | Verdict-dependent; if productive → dispatch H184 EMA+heads (verify `--model-heads` flag first) |
| H181 | #1503 | frieren | ~21:00Z + 20h main | smoke first, then main 30-EP |

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
