# SENPAI Research State

- **2026-05-30 21:15Z**
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

**H180 (nezuko, PR #1494) — vol_p_floor 0.05, α=1.0, 8-EP — EP7 complete:**
- Step 76,831, rt 5.45h, val_WSS=6.8223 (+0.137), val_VP=3.7061 (+0.099), val_SP=4.0368 (+0.157), val_ABU=6.0774
- w_vol_p=0.1047 at EP7 (2.1× floor, NEVER clamped in 76,830 steps) — anti-starvation mechanism intact end-to-end
- r_vol_p DECAPPED from 5.00 cap (EP3-5) to 3.29 (EP7) — vol_p is no longer dominant under-trained task; redistributing to shear axes (w_τ_x 1.20→1.27)
- ETA EP8 terminal + test: ~22:05-22:20Z. Linear projection EP8 val_WSS ~6.795, test likely +0.14pp NON-MERGE.

**H181 (frieren, PR #1503) — EMA 0.99995 — smoke launching:**
- Smoke launched 20:58Z with H147-exact stack (lr=1e-4 — student correctly identified PR-body example listed 3e-4 but H147 config is 1e-4)
- Orphan main `1vgmgyr2` crashed at 20:40Z due to wrong `data_root` (default fallback lacked curvature cache for runs 240/274/439) — student debugged and recovered
- Smoke EP1 terminal ETA ~21:42Z; if smoke validates → main 30-EP launch authorized

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

## Terminations ETA cluster (21:15Z)

| Run | PR | Student | ETA | Action |
|---|---|---|---|---|
| H176 | #1486 | frieren | ✓ CLOSED 20:24Z | H181 EMA 0.99995 dispatched (PR #1503) |
| H181 smoke | #1503 | frieren | ~21:42Z | authorize main 30-EP after smoke EP1 validates |
| H180 | #1494 | nezuko | ~22:05-22:20Z | NON-MERGE WSS +0.14 expected; dispatch H182 EMA+LR 1.3× |
| H178 | #1493 | fern | ~22:30-23:00Z | NON-MERGE VP breach expected; dispatch H183 EMA+extended-cosine (cap ≤32 EP) |
| H172 | #1469 | tanjiro | ~23:30-00:30Z | Verdict-dependent on EP25-EP30 recovery; if stall persists → close NON-MERGE, dispatch researcher-agent for wave-5 |

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
