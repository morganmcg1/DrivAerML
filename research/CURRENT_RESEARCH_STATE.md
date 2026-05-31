# SENPAI Research State

- **2026-05-31 02:20Z**
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** H147 (PR #1344, run `k6q4c3on`) — test_WSS=6.5409%, test_VP=3.4014%, test_SP=3.5634%, test_ABUPT=5.6648% (all floors cleared)
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85%

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

- **H172 EP29 → EP30 terminal ETA ~02:30Z** (currently step ~322k of 329,250; throughput 4.07 it/s)
- **H178 EP14 → EP15-16 terminal ETA ~03:20Z** (step ~155k of 175,600)
- **H182 EP3 → EP4 ~02:40Z** (boundary watch continuing)
- **H181 EP6 → EP10 critical gate ~04:00Z**

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
