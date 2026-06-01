# SENPAI Research State

- **2026-06-01 01:00Z**
- **Advisor branch:** drivaerml-long-20260504
- **dl24 SOTA:** H147 (PR #1344, run `k6q4c3on`) — test_WSS=6.5409%, test_VP=3.4014%, test_SP=3.5634%, test_ABUPT=5.6648% (all floors cleared)
- **Paper SOTA to beat:** Transolver-3 test_WSS < 5.85%

## 01:00Z snapshot — H185 EP6 SLIPS BEHIND H147 (+0.07pp); H184 EP27 decay finally activating but still NON-MERGE; H183 EP27 final stretch; H186 EP3 tracking H147

**Active fleet, 4 students, all WIP, terminal cluster ~02:30-03:30Z:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | val_ABU | lr | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| frieren | #1527 | H185 hidden_dim=640 | 6 | **6.8127** (+0.07 vs H147 EP6) | 3.6850 | 3.9736 | 6.0497 | 9.3e-5 | EP1 lead eroded; EP10 ≤6.65 gate sharpened |
| nezuko | #1529 | H186 layers=8 main (rank0=`31pux7bu`) | 2-3 | ~7.27 (tied H147 EP2) | 4.78 | 4.25 | 6.65 | 1.0e-4 | EP5 watch ~02:30Z |
| tanjiro | #1510 | H183 per-channel heads | 27.5 | 6.5883 (plateau) | 3.5816 | **3.8424 (flat)** | 5.8720 | 5e-6 (final tail) | NON-MERGE on SP floor; terminal ~03:30Z |
| fern | #1513 | H184 WSD LR | 27 | 6.8321 (decay NULL) | 3.7123 | 4.0628 | 6.0887 | 3.2e-5 | NON-MERGE projected ALL 4 axes; terminal ~02:30Z |

### Critical 01:00Z finding: capacity-axis (width AND depth) does NOT improve over H147 at hidden_dim=512/layers=6

**H185 trajectory (width=640):** EP1=11.49 (−1.33 vs H147=12.82) → EP6=6.8127 (+0.07 vs H147 EP6 ~6.74). The −1.33pp EP1 lead fully eroded by EP3 and **inverted by EP5-6**. Wider model uses extra capacity for redundancy, not for fitting harder features.

**H186 trajectory (layers=8):** Smoke EP1=11.892 (−0.93 vs H147=12.82), main EP2-3 ~7.27 (tied with H147 EP2=7.26). Depth axis showing the same pattern — strong EP1 lead, convergence to H147 by EP2-3.

**Joint conclusion (provisional, awaiting EP10+ reads):** H147's hidden_dim=512 + layers=6 is at or near the local capacity optimum for this DDP8 stack. Capacity-axis search closing. Next experiments should focus on **non-capacity axes**: loss shaping, schedule shape, attention mechanism, augmentation, distillation, gradient surgery.

### H184 WSD decay finally activating but still NON-MERGE on all 4 floors

EP24→EP27 (3 EPs, lr 8.55e-5 → 3.20e-5):
- val_WSS 6.8200 → 6.8321 (+0.012, slight uptick — decay still doesn't unlock features)
- val_VP 3.7192 → 3.7123 (−0.007, basically flat)
- val_SP 4.0604 → 4.0628 (flat)

Even at lr=32% of peak, no descent. Confirms WSD on this stack requires sharper schedule (100× drop, not 0.32×) AND longer stable phase before decay. Test projection: test_WSS=6.66 (+0.12), test_SP=3.86 (+0.28), test_VP=3.69 (+0.04), test_ABU=5.91 (+0.07). All 4 floors fail.



**Active fleet, 4 students, all WIP, terminal cluster ~02:30-03:30Z:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | val_ABU | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| frieren | #1527 | H185 hidden_dim=640 | 4 | **6.9002** (~tracking H147) | 3.8850 | 4.0246 | 6.1621 | EP5 ~01:40Z; EP1 lead converged |
| nezuko | #1529 | H186 layers=8 main (rank0=`31pux7bu`) | 0.66 | — (smoke EP1=11.892) | — | — | — | EP1 main read ~01:10Z |
| tanjiro | #1510 | H183 per-channel heads | 23+ | 6.5908 (plateau) | 3.5854 | **3.8463 (flat)** | 5.8748 | NON-MERGE on SP floor confirmed; terminal ~03:30Z |
| fern | #1513 | H184 WSD LR | 24+ | 6.8200 (decay NULL) | 3.7192 | 4.0604 | 6.0833 | NON-MERGE projected on WSS + SP; terminal ~02:30Z |

### Cross-cycle findings consolidated (22:43Z + 00:51Z)

**H185 capacity-axis (width=640):** EP1 lead of −1.33pp vs H147 converged to ~+0.05pp behind by EP4. Hidden_dim=640 boosts initial fit speed but H147's hidden_dim=512 catches up by EP2. **Width axis: modest mid-training value, requires EP10-15 decay phase to show whether capacity helps the tail.** EP10 kill gate: ≤6.65 (matches H147 EP10 = 6.64).

**H186 capacity-axis (layers=8):** Smoke EP1=11.892 (−0.928pp vs H147, weaker than H185's −1.33pp width advantage). Depth gives less initial lift than width per unit param. Main 25-EP run launched 00:15Z, EP1 main read ~01:10Z.

**Joint width-vs-depth read at EP10-15** will inform whether structural changes on H147 stack help OR whether H147's specific config is near-optimal at its size.

**H184 WSD decay FAILED to activate** — lr at EP24=8.55e-5 still ~85% of peak (decay schedule too gentle). WSS rate −0.0022pp/EP through decay (same as stable phase). Confirms WSD's literature claim (4-5× boost from sharp 100× lr drop) requires sharper schedule than tested. Future WSD attempts need lr at start-of-decay × 0.01 (not × 0.5).

**H183 SP floor breach confirmed**: val_SP=3.8463 at EP23 — flat slope through EP22-23. Test_SP projected 3.65 (FAILS 3.577 floor +0.07pp). Per-channel-heads ablation extracted some WSS-axis benefit but introduced SP regression — closes the axis-decoupled-heads direction.

## 22:43Z snapshot — **H182 CLOSED NON-MERGE** (test_WSS=6.6180, test_SP=3.6723 BREACH); H186 `layers=8` assigned to nezuko (PR #1529); H183/H184/H185 still active

**Active fleet, 4 students, 3 WIP + 1 new assignment (nezuko just freed by H182 closure):**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | Status |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1527 | H185 hidden_dim=640 | 2 | EP1=11.49 (−1.33 vs H147) | — | — | EP5 watch ~01:30Z |
| nezuko | #1529 | H186 layers=8 (depth) | — | — | — | — | NEW ASSIGNMENT |
| tanjiro | #1510 | H183 per-channel heads | 22 | 6.5894 (plateau) | 3.5802 | **3.8386 (flat)** | NON-MERGE on SP floor confirmed; terminal ~03:30Z |
| fern | #1513 | H184 WSD LR | 16 | 6.8560 (stall) | 3.7216 | 4.0635 | EP22 decay binary read ~01:00Z |

### H182 (PR #1506) CLOSED NON-MERGE — 22:46Z terminal

W&B run `ecw2sct9` — 30 EPs, 22.43h, EMA best EP23.

Test metrics (regress on ALL 4 axes vs H147):
- test_WSS=**6.6180** (+0.0771 vs H147 SOTA) ❌ primary regress
- test_VP=**3.4648** (+0.0634 vs H147, passes 3.643 floor) — partial-SOTA candidacy on VP FAILED
- test_SP=**3.6723** (BREACH 3.577 floor by +0.0953) ❌
- test_ABU=**5.7474** (+0.0826 vs H147, passes 5.844 floor)

**Critical methodological finding**: val→test pattern shifted significantly from H147 baseline. My 15:34Z/19:00Z projections used H147's val→test deltas (VP +0.06pp UP, SP −0.20pp DOWN) but H182 actual deltas were VP −0.018pp (basically flat) and SP −0.27pp DOWN. The val_VP=3.4830 partial-SOTA signal **did not generalize to test_VP**. Future hypotheses claiming val_VP improvements must show test-side validation; val_VP < 3.55 is NOT a reliable test_VP signal.

**Closes the lr-boost direction definitively**: H149 (β1=0.93/β2=0.97 + H147 lr) → H150 (β1=0.97/β2=0.985) → H182 (lr=1.3e-4 + ema=0.9999 compound) all NON-MERGE. Lion + lr=1e-4 + β1=0.95/β2=0.98 is a tight local optimum.

### H186 hypothesis (PR #1529, dl24-nezuko)

`model-layers=8` on H147 stack (single-flag change). Tests the **depth axis**, orthogonal to H185's **width axis** (hidden_dim=640). Together H185 and H186 jointly characterize the capacity-axis space on the H147 architecture for the first time.

- 25 EPs DDP8 (budget-constrained from 30) — loses ~−0.02 to −0.04pp cosine tail
- Smoke 1-EP first to measure throughput (kill if >56 min/EP)
- Kill ladder: EP1 ≤13.5, EP5 ≤6.95, EP10 ≤6.75, EP15 ≤6.65, EP20 ≤6.55, EP25 ≤6.50 + all 4 floors clear
- Companion to H185 (frieren PR #1527); if both win → compose width+depth next cycle

### H185 (frieren, PR #1527) — STRONG EP1 START (−1.33pp vs H147)

EP1 val_WSS=11.49% (H147 EP1=12.82%). EP1 timing ~53 min (~20% slower than H147 as expected with +25% width). Plan: complete 30-EP cosine; truncate at EP27 if budget tight (frieren's call).

Per-axis EP1 WSS: τ_x=10.20, τ_y=12.83, τ_z=14.74 — wider model helps all 3 axes from EP1.

EP5 gate at ≤6.90% — ~3.5h after EP1 = ~01:30Z next day.

### H183 (tanjiro, PR #1510) — SP FLOOR BREACH CONFIRMED, NON-MERGE projected; CONTINUE to EP30 for scientific value

EP18-22 disambiguation read: EP18 val_SP=3.8491 > 3.840 threshold = **CONSERVATIVE READ CONFIRMS** (the EP16→17 −0.025pp drop was noise, not decay-phase mechanism). SP slope EP15→EP22 = −0.0008pp/EP (essentially flat through 8 EPs).

Projected terminal val_SP ~3.83 → test_SP ~3.63 = **FAILS 3.577 floor by ~0.05pp** = NON-MERGE.

WSS plateau ~6.587-6.589 at EP22; terminal projection 6.55-6.59 val → test 6.39-6.43 = TIES or NARROWLY BEATS H147 (best case −0.15pp). Per-channel-heads decoder approach extracted some WSS-axis benefit but introduced SP regression.

Terminal at EP30 expected ~03:30Z (47 min/EP × 8 EPs from EP22 22:22Z). Will close NON-MERGE upon terminal.

**Scientific contribution recovery**: per-axis τ slopes EP15→EP22 (τ_x −0.0028pp/EP, τ_y −0.0070pp/EP, τ_z −0.0014pp/EP) confirm τ_y is the most-improvable WSS axis with axis-decoupled heads, τ_z is the persistent bottleneck (consistent with H154/H155/H156 falsifications).

### H184 (fern, PR #1513) — EP22 decay binary read pending, ~01:00Z

EP15-19 val_WSS stuck at ~6.84 (descent collapsed from −0.016pp/EP to flat). val_SP=4.06 (+0.49 above floor) is severely off-trajectory — even strong WSS decay won't pull SP through 3.577 floor unless decay delivers >−0.50pp SP over 8 EPs (very unlikely).

EP22 decay activation (~01:00Z) tests:
- WSS decay magnitude (target −0.30pp over 8 EPs for SOTA candidacy)
- SP decay magnitude (target ≥−0.50pp to reach floor — almost certainly fails)

**Most likely outcome: NON-MERGE on SP floor + likely WSS as well**, but EP22+ data informs future WSD-schedule decisions.

## 20:31Z snapshot — **H181 CLOSED NON-MERGE** (test_SP=3.6808 floor breach +0.104pp); H185 `hidden_dim=640` assigned to frieren (PR #1527); H182/H183/H184 still active

**Active fleet, 4 students, 3 WIP + 1 new assignment:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | Status |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1527 | H185 hidden_dim=640 | — | — | — | — | NEW ASSIGNMENT |
| nezuko | #1506 | H182 EMA 0.9999 + LR 1.3× | ~25-26 | 6.770 (flat) | **3.485** | 3.933 | partial-SOTA VP; terminal ~01:00Z |
| tanjiro | #1510 | H183 per-channel heads | 17 | **6.6002** | 3.5821 | **3.8412** | SP FLOOR WATCH; EP18-19 decisive |
| fern | #1513 | H184 WSD LR | 15.4 | 6.8392 | 3.727 | 4.046 | stable stall, EP22 decay binary read |

### H181 (PR #1503) CLOSED NON-MERGE — 20:30Z

- W&B run: `v4csonke` — 30 EPs, 24.77h runtime
- test_WSS=**6.6245** (+0.084pp vs H147) — FAILS primary
- test_VP=3.6245 ✓, test_ABUPT=5.7956 ✓
- test_SP=**3.6808** ✗ — BREACH +0.104pp over 3.577 floor
- **EMA-0.99995 falsified at 30-EP** — optimal EMA N≈10k steps (decay≈0.9999 confirmed)
- Best checkpoint: EP20 EMA (cosine re-ascent after EP20 = stack property, not noise)

### H185 hypothesis (PR #1527, dl24-frieren)

`hidden_dim=640` on clean H147 stack (single-flag change). Tests the only major structural axis not probed in H164–H167 wave (which tested slices/pe_features/surface_out/heads but NOT hidden_dim). At +25% width, addresses τ_y/τ_z representational saturation hypothesis. Smoke EP1 then 30-EP DDP8.

Kill ladder: EP1 ≤14.5%, EP5 ≤6.90%, EP10 ≤6.70%, EP15 ≤6.60%. Must beat test_WSS<6.5409 + clear all 4 floors to merge.

### H183 (tanjiro, PR #1510) — EP17 SP descent signal present, ambiguous

EP16→17: WSS −0.0186, VP −0.0221, **SP −0.0252pp** (first large single-EP SP drop). BUT EP15→17 average is still −0.0013pp/EP (essentially flat). Disambiguator: if EP18-19 sustain −0.020+ pp/EP SP descent, tanjiro's read wins (SP floor clears); if SP reverts to flat, merge blocker stands. Current WSS projection if descent resumes: EP30 val_WSS ~6.43-6.50 = BEATS H147 by 0.05-0.11pp ✓. SP wall still uncertain.

### H182 (nezuko, PR #1506) — VP partial-SOTA candidate confirmed, terminal ~01:00Z

val_VP=3.4853 at EP24.7 — partial-SOTA on VP axis (projected test_VP~3.25-3.30, beats H147 test_VP=3.4014 by 0.10-0.15pp). WSS flat at 6.77 throughout = NON-MERGE on primary. Merge decision: VP-only improvement + WSS regression. Will wait for terminal SENPAI-RESULT.

### H184 (fern, PR #1513) — stable-phase stall, EP22 decay binary read ~01:00Z

val_WSS=6.8392 at EP15.4; descent collapsed from −0.016pp/EP (EP9→12) to −0.0018pp/EP (EP12→15). WSD design: stable phase EP1-22 at lr=1e-4, decay EP22-30 lr→1e-6 (100× drop). For H184 to win, decay phase MUST deliver ≥−0.30pp over 8 EPs (vs −0.015pp total in stable-phase). Prior probability weakening but EP22 binary read decisive.

## 19:18Z snapshot — **H183 WSS descent RESUMED (EP16→17.4 = −0.013pp/EP) but SP floor STILL flat at −0.0016pp/EP — MERGE BLOCKER stands**; H181 terminal slowed to ~22:00Z; H182 VP plateau holding 3.485; H184 EP16 slight uptick (decay not yet activated)

**Active wave-5 fleet, all 4 students WIP. 19:18Z fleet:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | val_ABU | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 28.1 | 6.8285 (↓0.014) | 3.7786 | 4.0136 | 6.0872 | NON-MERGE (terminal ~22:00Z) |
| nezuko | #1506 | H182 LR 1.3× | 25.3 | 6.7760 (+0.006) | **3.4850** | 3.9372 | 5.9651 | partial-SOTA on VP (terminal ~01:00Z next day) |
| tanjiro | #1510 | H183 per-channel heads | 17.4 | **6.6002** | 3.5821 | **3.8412 (flat)** | 5.8797 | SP FLOOR BLOCKER; WSS path credible |
| fern | #1513 | H184 WSD LR | 16.0 | 6.8486 (+0.009) | 3.7349 | 4.0491 | 6.0962 | descent stalled, EP22 decay binary read |

### H183 (tanjiro) — WSS descent RESUMED, SP floor still flat

| EP | val_WSS | val_VP | val_SP | val_ABU |
|---:|---:|---:|---:|---:|
| 15 | 6.6140 | 3.5927 | 3.8450 | 5.8915 |
| 16 | 6.6188 | 3.6042 | 3.8664 | 5.9015 |
| **17.4** | **6.6002** | **3.5821** | **3.8412** | **5.8797** |

**EP16→17.4 (1.4 EPs):**
- val_WSS −0.0186 (−0.013pp/EP — descent RESUMED after EP15→16 stall ✓)
- val_VP −0.022 (descending, well below floor)
- val_SP −0.025 over 1.4 EPs from EP16 (BUT vs EP15 = only −0.0038 = essentially flat over 2.4 EPs)
- val_ABU −0.022 (descending)

**SP merge-blocker analysis (REPEAT):** EP15→EP17.4 SP rate = −0.0016pp/EP. At terminal EP30 = 3.8412 − 0.020 = **3.82**. Val→test from H147 pattern (−0.20pp on SP) = test_SP = **3.62 = FAILS floor 3.577 by 0.04pp**.

**WSS projection (with resumed descent):** EP30 = 6.6002 − 0.013×12.6 = 6.43 stable; with decay tail boost EP22+ = **6.35-6.40 = BEATS H147 by 0.14-0.19pp** ✓

Tanjiro is expected to post EP18/EP20 boundary reads ~20:00-21:00Z. EP20 gate val_SP ≤3.85 already cleared at EP17.4=3.84. Tanjiro committed to posting EP20 heartbeat per his 16:57Z comment.

### H181 (frieren) — terminal pushed to ~22:00Z

| EP | val_WSS | Δ |
|---:|---:|---:|
| 24 | 6.8285 | — |
| 26.8 | 6.8499 | +0.021 |
| **28.1** | **6.8285** | **−0.014 (slight reversion)** |

Step rate appears to have slowed (current pace projects terminal ~22:00Z, not 20:50Z as estimated). EMA-9999.5 producing oscillation, not monotonic divergence. **NON-MERGE confirmed (val_WSS=6.83 >> H147 6.55 EP30).** Close PR #1503 NON-MERGE on SENPAI-RESULT landing.

### H182 (nezuko) — VP plateau holding at 3.485

VP trajectory EP18=3.5172 → EP21=3.4872 → EP24=3.4807 → EP25.3=3.4850 (slight uptick within noise). **Plateau confirmed** at ~3.48 range. Test_VP projection 3.28 = beats H147 by 0.12pp = partial-SOTA on VP.

WSS plateau 6.77 = NON-MERGE on WSS. Terminal ETA ~01:00Z (next day, +6h from now at slowed pace).

### H184 (fern) — EP16 slight uptick, decay still not active

| EP | val_WSS | rate |
|---:|---:|---:|
| 11.9 | 6.8454 | — |
| 14.7 | 6.8404 | −0.0018/EP |
| **16.0** | **6.8486** | **+0.009/EP (uptick)** |

Stall has converted to slight uptick. WSD lr schedule should still be in stable phase (lr~1e-4) at EP16 — decay starts EP22. Confidence in EP22 decay-phase binary read remains, but probability of beating H147 dropping.

## 17:42Z snapshot — H183 SP floor merge-blocker discovery (val_SP=3.85 flat slope); H184 descent STALLED in stable phase (−0.0018/EP); H182 EP24 val_VP=3.481 still deepening; H181 EP26.8 climbing NON-MERGE confirmed

**Active wave-5 fleet, all 4 students WIP. 17:42Z fleet:**

| Student | PR | Hyp | EP | val_WSS | val_VP | val_SP | EP30 risk | Decision |
|---|---|---|---:|---:|---:|---:|---|---|
| frieren | #1503 | H181 EMA-99995 | 26.8 | 6.8499 (↑) | 3.792 | 4.022 | NON-MERGE (terminal hold) | continue to EP30 ~19:00Z |
| nezuko | #1506 | H182 LR 1.3× | 24.0 | 6.7671 | **3.481** | 3.930 | WSS NON-MERGE / VP partial SOTA | continue to EP30 ~20:30Z |
| tanjiro | #1510 | H183 per-channel heads | 16.2 | 6.6188 | 3.604 | **3.866** | **SP FLOOR BLOCKER** | continue, SP decay-phase critical |
| fern | #1513 | H184 WSD LR | 14.7 | 6.8404 | 3.745 | 4.054 | descent STALLED, EP22 decay critical | continue, decisive read EP22 ~21:00Z |

### H183 (tanjiro) — SP FLOOR IS THE BLOCKER (WSS path still credible)

Tanjiro's EP15 boundary read (15:37Z): val_WSS=6.6140, val_VP=3.5927, val_SP=3.8450. All kill gates PASS, but SP slope is flat (~−0.002pp/EP). Mapping val→test from H147 pattern:

| Channel | H147 val EP30 | H147 test EP30 | val→test Δ |
|---|---:|---:|---:|
| WSS | ~6.60 | 6.5409 | −0.06 |
| VP | ~3.60 | 3.4014 | −0.20 |
| **SP** | **~3.76** | **3.5634** | **−0.20** |

**If H183 val_SP terminal = 3.82, projected test_SP = 3.62 = FAILS floor 3.577 by 0.04pp.** This blocks merge regardless of WSS performance.

**EP30 projections (val):**
- WSS at current −0.005pp/EP: 6.55 (TIES H147); with decay boost: **6.45-6.50 (BEATS H147)** ✓
- VP at current −0.020pp/EP: 3.32 (healthy below floor) ✓
- **SP at flat slope: 3.82 → test_SP = 3.62 = FAILS floor** ✗

**Decision: CONTINUE — but EP22+ decay phase MUST accelerate SP descent.** H147's cosine reaches lr~1e-6 by EP30. If H183 SP follows same pattern, end-game descent could be 5-10× current rate. EP20 and EP25 boundary reads are decisive for SP.

### H184 (fern) — descent STALLED in stable phase (decay phase EP22 is now MUST-DELIVER)

| EP | val_WSS | descent rate |
|---:|---:|---:|
| 9 | 6.8922 | — |
| 11.9 | 6.8454 | −0.016/EP |
| **14.7** | **6.8404** | **−0.0018/EP (STALL)** |

**Descent stalled from −0.016 to −0.0018pp/EP** — 9× slowdown. The WSD design predicted continued stable-phase descent at lr=1e-4. This stall is unexpected.

**Updated EP22 projection:** 6.8404 - 0.0018×7.3 = **6.83** (vs my 15:34Z projection 6.69). +0.20pp behind H147 EP22 ~6.58.

**Decay phase EP22→EP29 (lr 1e-4 → 1e-6) MUST deliver 4-5× boost to reach H147 territory:**
- Best case (5× boost = −0.10pp/EP × 7 EPs = −0.70pp total): EP30 = **6.13** (improbable, overshoot)
- Realistic (3× boost = −0.06pp/EP × 7 EPs = −0.42pp): EP30 = **6.41** (BEATS H147 by 0.13pp)
- Conservative (2× boost = −0.04pp/EP × 7 EPs = −0.28pp): EP30 = **6.55** (TIES H147)
- Worst (decay-phase fizzles): EP30 = 6.75-6.80 = NON-MERGE

**Decision: CONTINUE — EP22 decay-phase activation is the binary read.** If decay activates strongly: H184 is a credible SOTA candidate. If decay fails to activate: NON-MERGE. EP22 boundary ~21:00Z.

### H182 (nezuko) — VP=3.481 still deepening at EP24 (partial-SOTA holding)

| EP | val_WSS | val_VP | floor delta |
|---:|---:|---:|---:|
| 18 | 6.7744 | 3.5172 | −0.126 |
| 21 | 6.7716 | 3.4872 | −0.156 |
| **24** | **6.7671** | **3.4807** | **−0.162** |

VP descent slowing to −0.0022pp/EP EP21→EP24 (vs −0.010 EP18→21). Plateau approaching at ~3.45-3.48 range. **Test_VP projection val→test ~−0.20pp:** EP30 val_VP ~3.45 → test_VP ~3.25 = **beats H147 test_VP=3.4014 by ~0.15pp** (partial-SOTA candidate confirmed).

WSS plateau holding at 6.77 = NON-MERGE on WSS.

### H181 (frieren) — EP26.8 climbing, NON-MERGE terminal-hold

| EP | val_WSS |
|---:|---:|
| 21 | 6.8175 |
| 24 | 6.8285 |
| **26.8** | **6.8499 (continued climb)** |

EMA-9999.5 averaging window producing unstable convergence — descent reversed by EP21, accelerating climb EP24→26.8 (+0.021pp/3EPs vs +0.011 EP21→24). EP30 projection 6.85-6.90 = consistently behind H147 6.5409 and H172 6.6517. **NON-MERGE — terminal harvest ~19:00Z.**

## 15:34Z snapshot — **H182 VP=3.487 DEEPENING (−0.156pp below floor, partial-SOTA strengthening, test_VP proj 3.20 = beats H147 by 0.20pp);** H183 EP13.6 = 6.6306 descent severely slowed (EP15 borderline FAIL, EP30 proj 6.51-6.57 still beats H147); **H184 EP11.9 = 6.8454 descent ACCELERATING to −0.016/EP (re-projected EP30 6.44-6.54 with decay boost = STRONG SOTA candidate);** H181 EP24 = 6.8285 slight uptick (NON-MERGE confirmed)

**Active wave-5 fleet, all 4 students WIP. 15:34Z fleet:**

| Student | PR | Hyp | EP | val_WSS | val_VP | EP30 projection | Decision |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 24.0 | 6.8285 | 3.778 | 6.75-6.83 | NON-MERGE (terminal hold) |
| nezuko | #1506 | H182 LR 1.3× | 21.1 | 6.7716 | **3.487** | WSS 6.70-6.73 / **test_VP 3.20** | **STRONG PARTIAL SOTA (VP)** |
| tanjiro | #1510 | H183 per-channel heads | 13.6 | 6.6306 | 3.639 | **6.51-6.57** | **SOTA candidate, EP22 decay critical** |
| fern | #1513 | H184 WSD LR (main) | 11.9 | 6.8454 | 3.760 | **6.44-6.54 (decay boost)** | **NEW SOTA candidate** |

### H182 (nezuko) — VP DEEPENING TO 3.487 (partial-SOTA strengthening over 13:20Z reading)

| EP | val_WSS | val_VP | floor delta |
|---:|---:|---:|---:|
| 12 | 6.7901 | 3.5550 | −0.088 |
| 15 | 6.7833 | 3.5251 | −0.118 |
| 18 | 6.7744 | 3.5172 | −0.126 |
| **21** | **6.7716** | **3.4872** | **−0.156** |

**VP descent re-accelerating EP18→EP21: −0.030pp/3EPs = −0.010pp/EP** (up from −0.0027/EP at EP15-18). Likely the cosine decay phase starting — LR has dropped below stable regime, tighter convergence pulls VP deeper.

**Test_VP projection updated:** EP30 val_VP ~3.40 → test_VP ~3.20 = **beats H147 test_VP=3.4014 by ~0.20pp** (strengthened from 13:20Z 0.10pp projection).

**Channel-specific partial-SOTA candidate confirmed.** Continue to natural terminal EP30 ~16:00Z for test harvest.

### H183 (tanjiro) — descent severely slowed but STILL ON SOTA PATH (EP15 borderline)

| EP | val_WSS | descent rate |
|---:|---:|---:|
| 5 | 6.7497 | — |
| 8 | 6.6726 | −0.026/EP |
| 11 | 6.6407 | −0.011/EP |
| **13.6** | **6.6306** | **−0.0039/EP (CONCERN)** |

**Descent dropped 5× from EP8-11 to EP11-13.6.** EP15 gate ≤6.60 likely FAILS by 0.025pp (projected 6.625). However:
- EP30 projection at current rate: **6.567** = SLIGHTLY BEATS H147 by 0.03pp
- WITH decay-phase tail (H172-class ~−0.05pp): 6.51-6.52 = CLEARLY BEATS H147
- Worst-case continued deceleration: 6.59-6.61 = ties/slightly behind

**Critical concerns:**
- val_VP=3.639 just −0.004pp below floor 3.643 (slight regress from EP11=3.618). Watch VP closely — if crosses 3.65 sustained 2+EPs = kill.
- val_SP=3.854 = +0.276pp above floor 3.577. NO improvement EP11→13.6. **SP needs to crash by terminal or test_SP regresses.**

**Continue running, EP18-20 reads critical.**

### H184 (fern) — descent ACCELERATING to −0.016/EP in stable phase (NEW STRONG candidate)

| EP | val_WSS | descent rate | lr |
|---:|---:|---:|---:|
| 5 | 6.9329 | — | 1e-4 |
| 9 | 6.8922 | −0.010/EP | 1e-4 |
| **11.9** | **6.8454** | **−0.016/EP (UP)** | 1e-4 |

**Stable-phase descent ACCELERATING from −0.010 to −0.016pp/EP.** WSD design paying off — finding deeper minima as stable phase extends.

**Re-projected EP22 (last stable EP):** 6.8454 - 0.016×10.1 = **6.69** (vs my 13:20Z projection 6.76)

**Decay phase EP22→EP29 (lr 1e-4 → 1e-6) at 2-5× boost:**
- Conservative (2×): EP30 = 6.54 (TIES H147)
- Realistic (3-4×): EP30 = **6.44-6.49 = BEATS H147 by 0.05-0.10pp**
- Optimistic (5×): EP30 = 6.24 (overshoot likely smoothed by EMA)

**H184 is now a STRONG SOTA candidate.** Decay-phase boost is the decisive read. **EP22 ~17:00Z** is critical.

### H181 (frieren) — slight uptick at EP24, NON-MERGE confirmed terminal-hold

| EP | val_WSS |
|---:|---:|
| 18 | 6.8305 |
| 21 | 6.8175 |
| **24** | **6.8285 (+0.011 uptick)** |

EMA-9999.5 averaging window causing instability at local minimum. EP30 projection 6.75-6.83 = consistently behind H147 6.5409 and H172 6.6517. **NON-MERGE — continue to terminal EP30 for clean test harvest.**

## 13:20Z snapshot — **H183 EP11 = 6.6407 ON SOTA TRAJECTORY** (EP10 gate PASSED, projecting EP30 ~6.44-6.55, VP at 3.62 borderline); H182 EP18 VP CONTINUES TO BREAK FLOOR (3.517, −0.126pp below); H184 EP9 stable phase +0.24pp behind H147 (WSD design lag, decay EP22+); H181 EP21 severely decelerated, NON-MERGE re-sealed

**Active wave-5 fleet, all 4 students WIP. 13:20Z fleet:**

| Student | PR | Hyp | EP | val_WSS | val_VP | EP30 projection | Decision |
|---|---|---|---:|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 21 | 6.8175 | 3.760 | 6.70-6.77 | NON-MERGE (sealed) |
| nezuko | #1506 | H182 LR 1.3× | 18 | 6.7744 | **3.517** | 6.70-6.73 / VP=3.25-3.30 | **PARTIAL MERGE (VP-side)** |
| tanjiro | #1510 | H183 per-channel heads | 11 | **6.6407** | 3.618 | **6.44-6.55** | **STRONGEST SOTA CANDIDATE** |
| fern | #1513 | H184 WSD LR (main) | 9 | 6.8922 | 3.794 | 6.56-6.72 (decay EP22+) | continue, EP22 critical |

### H183 (tanjiro) — STRONGEST SOTA CANDIDATE: EP11 val_WSS = 6.6407, EP10 gate PASSED

| EP | val_WSS | descent | Δ vs H147 |
|---:|---:|---:|---:|
| 5 | 6.7497 | — | −0.000 (TIED) |
| 8 | 6.6726 | −0.0257/EP (EP5→8) | ~tied EP8 |
| **11** | **6.6407** | **−0.0106/EP (EP8→11)** | **~−0.04 from H147 trajectory** |

**EP10 critical gate ≤6.66 PASSED** (EP10 interpolated ~6.65; EP11 = 6.6407 confirms persistent structural advantage of per-channel decoder heads). Per-axis τ_y/τ_z: 7.27/9.09 — cross-flow channels learning faster than H147 baseline.

**EP30 terminal projections:**
- Linear extrap (−0.011pp/EP × 19 EPs): **6.44** = BEATS H147 6.5409 by ~0.10pp
- Conservative (continued deceleration): 6.55 = SLIGHTLY BEATS H147
- Worst-case plateau: 6.65 = TIES H172, below H147

**VP floor watch CRITICAL:** EP11 VP = 3.618, just −0.025pp below 3.643 floor. EP5 VP = 3.82 → descent of 0.20pp over 6 EPs. Watch for VP regress > 3.65 sustained = kill signal.

**SP concern:** 3.85 vs floor 3.577 = +0.27pp above floor. H147 val_SP EP30 was ~3.6, so H183 needs SP to drop ~0.25pp by terminal.

### H182 (nezuko) — VP FLOOR BREACH SUSTAINED (3.517 at EP18, partial-SOTA candidate on VP-side)

| EP | val_WSS | val_VP | floor delta |
|---:|---:|---:|---:|
| 12 | 6.7901 | 3.5550 | −0.088 |
| 15 | 6.7833 | 3.5251 | −0.118 |
| **18** | **6.7744** | **3.5172** | **−0.126pp** |

**WSS plateau confirmed at ~6.77** — H182 won't beat H147 on WSS (projected EP30 6.70-6.73 vs H147 6.5409). 

**VP improvement persists.** Test_VP projection via val→test ~−0.20pp: 3.25-3.30 = beats H147 test_VP=3.4014 by ~0.10pp. **Channel-specific partial-SOTA candidate.** Mechanism (LR 1.3× × EMA-9999 windowing) failed on WSS but succeeded on VP. Continue to terminal for test harvest.

### H184 (fern) — WSD stable phase, slow descent +0.24pp behind H147 (design lag)

| EP | val_WSS | descent | lr | Δ vs H147 |
|---:|---:|---:|---:|---:|
| 5 | 6.9329 | — | 1e-4 | +0.183 |
| **9** | **6.8922** | **−0.010/EP** | **1e-4** | **+0.240** |

**lr=1e-4 holding** through stable phase (EP1-EP22 design). Descent rate −0.010pp/EP healthy but slow. **At current rate, EP22 = 6.76 = FAILS gate ≤6.65 by 0.11pp.** Decay phase EP22→EP29 (lr 1e-4 → 1e-6) should accelerate descent 2-5× over H147's flat-tail. Best-case EP30 = 6.56 (TIES/SLIGHTLY BEATS H147), realistic 6.62-6.66 (matches H172), worst case 6.72 (NO improvement). **EP22 boundary is the decisive read.**

### H181 (frieren) — Descent SEVERELY decelerated, NON-MERGE RE-SEALED

| EP | val_WSS | descent rate |
|---:|---:|---:|
| 15 | 6.8811 | — |
| 18 | 6.8305 | −0.025/EP |
| **21** | **6.8175** | **−0.004/EP** |

Descent went from −0.060 (EP12-14) → −0.025 (EP15-18) → **−0.004 (EP18-21)**. Severely flattened. Terminal projection 6.70-6.77 (vs H147 6.5409, H172 6.6517) = **NON-MERGE confirmed.** Continue to natural terminal EP30 for clean test metric harvest. EMA decay 0.99995 too aggressive for 30-EP budget.

## 11:10Z snapshot — H182 EP15 PASS gate + VP HOLDING below floor; H183 EP8 tracking ~tied H147 (EP10 gate imminent); H181 descent decelerated but credible H172-beat path; H184 EP5 trailing H147 +0.183pp (WSD design lag)

**Active wave-5 fleet, all 4 students WIP:**

| Student | PR | Hyp | EP @ 11:10Z | Latest val_WSS | Δ vs H147 | Δ vs H172 | Next gate |
|---|---|---|---|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 18 (step 197,567) | 6.8305 | +0.290 | +0.179 | EP22 ≤6.78 ~13:00Z (decay starts) |
| nezuko | #1506 | H182 LR 1.3× | **15** (step 164,639) | **6.7833** | **+0.183** | **+0.093** | **EP15 ≤6.85 PASS**; EP22 ≤6.75 next |
| tanjiro | #1510 | H183 per-channel heads | 8 (step 87,807) | **6.6726** | **~tied** | — | **EP10 ≤6.66 @ step 109,759 ~11:50Z (40 min)** |
| fern | #1513 | H184 WSD LR (main) | 5 (step 54,879) | 6.9329 | **+0.183** | — | EP10 ≤6.85 ~13:30Z (early lag expected) |

### H183 (tanjiro) — HOTTEST CANDIDATE: per-channel heads holding ~tied with H147

| EP | step | val_WSS | Δ vs H147 |
|---:|---:|---:|---:|
| 5 | 54,879 | 6.7497 | **−0.000** (TIED) |
| 6 | 65,855 | 6.7100 | ~tied |
| 7 | 76,831 | 6.6890 | ~tied |
| **8** | 87,807 | **6.6726** | **~tied** |

Descent rate EP7→EP8: −0.0164pp/EP (natural deceleration approaching plateau). **EP10 gate ≤6.66 at ~11:50Z:** linear extrap gives **6.640** (PASSES by 0.020), conservative gives 6.649 (PASSES by 0.011). H147 EP10 = 6.64 → H183 essentially TIES OR BEATS H147 EP10. Per-channel decoder heads structural advantage persisting. **If EP10 lands ≤6.64 → strong SOTA candidate.**

### H182 (nezuko) — EP15 PASS + VP CLEARING FLOOR sustained (LARGEST sustained VP-below-floor reading in 12+ hypotheses)

| EP | step | val_WSS | val_VP | val_ABU | lr | Δ vs H172 |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 131,711 | 6.7901 | 3.5550 | 5.9886 | 9.3e-5 | +0.099 |
| 13 | 142,687 | 6.7700 | 3.5419 | 5.9711 | 8.9e-5 | +0.080 |
| 14 | 153,663 | 6.7796 | 3.5331 | 5.9647 | 8.4e-5 | +0.089 |
| **15** | 164,639 | **6.7833** | **3.5251** | 5.9577 | 7.9e-5 | **+0.093** |

**EP15 critical gate ≤6.85: PASS (6.7833).** **VP @ EP15 = 3.5251 = −0.118pp BELOW 3.643 floor** (sustained EP12→EP15). If holds to test (val→test ~−0.3pp from H147 pattern), test_VP ~3.30 — clearly improving on H147 test_VP=3.4014. **VP-only improvement is itself a merge contribution.** EP12→EP15 WSS slope flat (−0.022pp/EP). EP30 projection: linear 6.73, with decay boost 6.65. **Merge-eligible if VP advantage holds.**

### H181 (frieren) — Descent decelerated but credible H172-beat path (REVISED from 08:40Z NON-MERGE seal)

| EP | step | val_WSS | Δ vs H147 | Δ vs H172 |
|---:|---:|---:|---:|---:|
| 15 | 164,639 | 6.8811 | +0.281 | +0.191 |
| 16 | 175,615 | 6.8551 | +0.255 | +0.165 |
| 17 | 186,591 | 6.8402 | +0.240 | +0.150 |
| 18 | 197,567 | 6.8305 | +0.230 | +0.140 |

Descent rate now −0.0169pp/EP (decelerated from −0.06pp/EP at EP12-14). Linear extrap EP30: 6.63 — **beats H172 (6.6517) clearly**; conservative 6.65-6.70 — close-tied with H172. **REVISED from 08:40Z NON-MERGE call:** H181 has credible path to beat H172, possibly approach H147 with cosine decay tail boost (EP22+).

### H184 (fern) — WSD stable phase tracking +0.18pp behind H147 (expected, design payoff EP22+)

| EP | step | val_WSS | lr | Δ vs H147 |
|---:|---:|---:|---:|---:|
| 1 | 10,975 | 12.7118 | 1.00e-4 | **−0.108 (LEAD)** |
| 2 | 21,951 | 7.3219 | 1.00e-4 | +0.062 |
| 3 | 32,927 | 7.0769 | 1.00e-4 | +0.097 |
| 4 | 43,903 | 6.9996 | 1.00e-4 | +0.120 |
| **5** | 54,879 | **6.9329** | **1.00e-4** | **+0.183** |

lr=1e-4 stable phase active (vs H147 cosine EP5 ~8.7e-5). Higher LR = slower early descent but theoretical deeper minimum at decay end. Descent rate −0.072pp/EP (healthy, no plateau). **WSD payoff expected EP22+ when cosine begins.** EP25-EP30 decisive read. Don't kill on early lag.

## 08:40Z snapshot — H183 EP5 EXACTLY TIED with H147 (HOT candidate); H182 EP10 borderline + VP CLEARING FLOOR; H184 main EP2 close to H147; H181 EP14 closing slowly

**Active wave-5 fleet, all 4 students WIP:**

| Student | PR | Hyp | EP @ 08:40Z | Latest val_WSS | Δ vs H147 | Δ vs H172 | Next gate |
|---|---|---|---|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | 14 (step 153,663) | 6.9186 | — | +0.213 | natural terminal EP30 |
| nezuko | #1506 | H182 LR 1.3× | 11 (step 120,735) | **6.8158** | **+0.166** | **+0.005** | **EP15 ≤6.85 @ step 164,625 ~10:10Z** |
| tanjiro | #1510 | H183 per-channel heads | 5 (step 54,879) | **6.7497** | **−0.000** (TIED) | — | **EP10 ≤6.66 @ step 109,759 ~11:50Z** |
| fern | #1513 | H184 WSD LR (main) | 2 (step 21,951) | 7.3219 | +0.062 | — | EP5 ≤6.85 ~13:00Z |

### H183 (tanjiro) — HOTTEST CANDIDATE: EP5 EXACTLY TIED with H147

| EP | step | val_WSS | val_VP | val_SP | val_ABU | Δ vs H147 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,975 | 12.9728 | 14.0018 | 8.7417 | 13.0824 | +0.153 |
| 2 | 21,951 | 7.2222 | 4.9462 | 4.1772 | 6.6413 | −0.038 |
| 3 | 32,927 | 6.9084 | 4.1676 | 3.9783 | 6.2318 | **−0.072** |
| 4 | 43,903 | 6.8066 | 3.9018 | 3.9187 | 6.0995 | −0.029 |
| **5** | 54,879 | **6.7497** | 3.8184 | 3.8917 | 6.0380 | **−0.000** (TIED) |

Per-axis τ: EP3 (−0.052/−0.072/−0.084 τx/τy/τz), EP4 (−0.034/−0.014/−0.004), EP5 mechanism confirmed sustained. Per-channel decoder heads provide a ~0.04pp persistent lead vs H147 without dissipating. EP10 critical gate ≤6.66. Linear extrap EP3→5 slope (−0.079pp/EP) gives EP10 ≈ 6.36 (easily passes); conservative ≈ 6.55-6.60 (still passes). VP @ EP5 = 3.82, watch floor 3.643 — needs to drop ~0.18pp by terminal.

### H182 (nezuko) — EP10 BORDERLINE FAIL but EP11 NEAR-TIED + **VP CLEARING FLOOR**

| EP | step | val_WSS | val_VP | val_SP | val_ABU | lr | Δ vs H172 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 87,807 | 6.8905 | 3.7134 | 4.0518 | 6.1071 | 1.13e-4 | +0.083 |
| 9 | 98,783 | 6.8555 | 3.6641 | 4.0124 | 6.0689 | 1.09e-4 | +0.040 |
| **10** | 109,759 | **6.8381** | **3.6224** | 3.9929 | 6.0439 | 1.03e-4 | **+0.057** |
| **11** | 120,735 | **6.8158** | **3.5942** | 3.9757 | 6.0205 | 9.8e-5 | **+0.005** |

EP10 critical gate ≤6.78 TECHNICAL FAIL by 0.06pp. BUT EP11 essentially tied with H172 (+0.005pp) AND **VP CLEARING 3.643 FLOOR** for the first time in 12 hypotheses (EP10=3.622, EP11=3.594). Decision: HOLD to EP15. EP15 critical gate ≤6.85 vs H172 EP15=6.690. The VP-below-floor finding is the most important signal — if val_VP=3.59 holds to test (val→test ~−0.3pp pattern from H147), test_VP could be ~3.30, well clear of floor and improving on H147 by ~0.1pp.

### H181 (frieren) — Steady descent, no path to H147

| EP | step | val_WSS | val_VP | val_SP | val_ABU | Δ vs H172 |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 120,735 | 7.2399 | 4.2698 | 4.3739 | 6.5242 | +0.429 |
| 12 | 131,711 | 7.0711 | 4.0295 | 4.2131 | 6.3328 | +0.309 |
| 13 | 142,687 | 6.9780 | 3.8973 | 4.1266 | 6.2299 | +0.250 |
| 14 | 153,663 | 6.9186 | 3.8430 | 4.0736 | 6.1697 | +0.213 |

Descent steady at −0.06pp/EP. Gap-closing rate to H172 = −0.072pp/EP — catches H172 around EP17-18, but H172 EP14→30 only descends −0.05pp total. Terminal projection ~6.65-6.68 = H172-class but no path to H147. NON-MERGE sealed absent H172-beat. Continue to natural terminal.

### H184 (fern) — Smoke verification COMPLETE & PASSED; main run at EP2

Smoke verified: warmup→stable LR transition (5e-6→1e-4 at EP1), stable plateau EP1-22 at 1e-4, decay cosine EP23-29 to eta_min=1e-6. No NaN/skipped steps. Dry-run schedule matches design.

Main run `usc1tpni` launched 06:32Z. EP1 val_WSS=12.7118 (−0.108pp LEAD vs H147), EP2=7.3219 (+0.062pp slight lag). lr=1e-4 confirmed at EP1+EP2 (WSD stable phase active). WSD payoff expected late (EP22+ when cosine decay starts). Don't kill on early-EP lag — hypothesis specifically about late-tail descent.

## 06:30Z snapshot — H181 EP10 PASSED critical gate; H182 EP8 gap NARROWING (+0.083 vs H172); H183 EP2 slight LEAD vs H147 (per-channel heads alive); H184 smoke pre-EP1

**Active wave-5 fleet, all 4 students WIP:**

| Student | PR | Hyp | Status @ 06:30Z | Latest val_WSS | Δ vs H172 | Δ vs H147 | Next gate |
|---|---|---|---|---:|---:|---:|---|
| frieren | #1503 | H181 EMA-99995 | EP11 step 120,735 | 7.2399 | +0.429 | — | natural terminal EP30 |
| nezuko | #1506 | H182 LR 1.3× | EP8 step 87,807 | **6.8905** | **+0.083** | **+0.231** | **EP10 ≤6.78 @ step 109,759 ~07:15Z** |
| tanjiro | #1510 | H183 per-channel heads | EP2 step 21,951 | **7.2222** | **−0.078** | **−0.038** | EP3 ≤7.00 ~07:00Z |
| fern | #1513 | H184 WSD LR | smoke step 8,609 (EP0.78) | mid-warmup | — | — | EP1 boundary ~06:40Z |

**H181 EP10 PASS (frieren, PR #1503):**

| EP | step | val_WSS | val_VP | val_SP | val_ABU | Δ vs H172 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 87,807 | 8.9007 | 6.5517 | 5.4795 | 8.2701 | +2.093 |
| 9 | 98,783 | 7.8525 | 5.1557 | 4.8763 | 7.2019 | +1.037 |
| **10** | 109,759 | **7.3745** | 4.4093 | 4.5395 | 6.6699 | **+0.594** |
| 11 | 120,735 | 7.2399 | 4.2698 | 4.3739 | 6.5242 | +0.429 |

EMA-99995 init_mass washout: EP10=0.41%, EP11=0.24% (now functionally washed). Descent decelerating: −1.05 → −0.48 → −0.13pp/EP (EP9→11). Trajectory "5 EPs lagged" vs H172 in EMA-equivalent terms. H172 EP11→30 delta was only −0.16pp — H181 needs at least −0.6pp more to match H172 terminal; **not achievable** at current decel. H147 SOTA structurally unreachable in 30-EP envelope. Continue to natural terminal for clean test metrics for the lit comparison; **NON-MERGE sealed at terminal**.

**H182 EP6-EP8 gap NARROWING (nezuko, PR #1506):**

| EP | step | val_WSS | val_VP | val_SP | val_ABU | lr | Δ vs H172 | Δ vs H147 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 54,879 | 7.4043 | 4.6284 | 4.6367 | 6.7385 | 1.24e-4 | +0.131 | +0.654 |
| 6 | 65,855 | 7.1135 | 4.0192 | 4.3644 | 6.3671 | 1.21e-4 | +0.142 | +0.404 |
| 7 | 76,831 | 6.9771 | 3.8042 | 4.1861 | 6.2034 | 1.18e-4 | +0.098 | +0.297 |
| 8 | 87,807 | **6.8905** | **3.7134** | 4.0518 | 6.1071 | 1.13e-4 | **+0.083** | **+0.231** |

EP5 lead-reversal **NOT terminal** — H182 catching back up. Channel-asymmetric mechanism confirmed: VP improving 4.63→3.71 in 3 EPs (−0.92pp), faster than H172's VP track. WSS lagging H172 by only +0.083pp at EP8. **EP10 critical gate ≤6.78 at step 109,759 ~07:15Z:** linear extrap −0.11pp/EP gives EP10 ≈ 6.67 (PASS) but conservative ≈ 6.83 (FAIL). **VP floor watch:** EP8 = 3.7134, still +0.07pp above 3.643 floor — floor breach risk if VP slows.

**H183 EP1-EP2 mechanism alive (tanjiro, PR #1510):**

| EP | step | val_WSS | val_VP | val_SP | val_ABU | Δ vs H147 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,975 | 12.9728 | 14.0018 | 8.7417 | 13.0824 | +0.153 |
| **2** | 21,951 | **7.2222** | 4.9462 | 4.1772 | 6.6413 | **−0.038** |

Per-channel decoder heads initialized cleanly. EP2 SLIGHT LEAD vs H147 (−0.04pp) — meaningful for structural perturbation at this early stage. Tanjiro skipped separate smoke; direct-to-main reasonable (structural change, not numerical instability concern). Mechanism: independent per-channel heads decouple cp (well-fit) and τ_z (lagging) optimization. Watch EP3 ≤7.00 (~07:00Z) and EP5 ≤6.78 (~08:30Z) gates. If lead holds through EP5, strong SOTA candidate.

**H184 WSD smoke pre-EP1 (fern, PR #1513):** Smoke `ozxi8j68` at step 8,609 / EP0.78 mid-warmup. EP1 boundary at step 10,975 ~06:40Z. Schedule-shape verification primary: confirm lr peaks at 1e-4 at EP1, holds flat through EP2-EP3 (WSD stable phase). Will green-light main 30-EP if smoke EP1-3 track H147 ±0.3pp + lr-flat post-warmup.

## 04:15Z snapshot — H178 CLOSED NON-MERGE (test 4-floor breach); H184 WSD dispatched to fern; H182 EP5 lead REVERSED; H181 EP8 borderline EP10 gate

**H178 CLOSED (PR #1493, fern, run `csk7pkf1`):** terminal SENPAI-RESULT landed 04:02Z. Test metrics (best-EMA EP13):
- test_WSS=**6.6237** vs H147=6.5409 → **+0.083pp BEHIND** (MISS SOTA)
- test_VP=**3.9237** → **BREACH +0.281pp** over 3.643 floor
- test_SP=**3.6968** → **BREACH +0.120pp** over 3.577 floor
- test_ABUPT=**5.8672** → **BREACH +0.023pp** over 5.844 floor

**4-of-4 floor failure.** Wave-3 α/floor closure: 16-EP slow cosine WORSENS test_VP vs 8-EP equivalent (H173 3.78 → H178 3.92). Cosine duration is downstream of (α, floor) for GradNorm equilibria. Fern mechanism finding: w_τ_x absorbs the freed budget (NOT w_cp as H173 read suggested); the SP-protection narrative for floor=0.05 revised. w_τ_z dropped over the run (1.32 → 1.29) — the only axis that lost weight — and per-axis test_WSS_z=8.67% confirms τ_z under-weighting is the next mechanism lever.

**H184 dispatched to fern (PR #1513):** WSD LR Schedule (H-W5-1) — replace H147's full 30-EP cosine with Warmup(EP1) + Stable(EP2-EP22 at peak lr=1e-4) + Decay(EP23-EP30 cosine to lr_min). Hypothesis: more steps at peak LR explore loss surface more aggressively, then concentrated decay tail locks in deeper minimum. Lit-aligned (Hu et al. 2024 MiniCPM, Hägele et al. 2024). Code change: extend `build_lr_scheduler` in `trainer_runtime.py` to support `--lr-schedule wsd` with 3-stage SequentialLR.

**H182 (nezuko, PR #1506) — EP5 boundary: LEAD REVERSED:**

| metric | H182 EP5 (lr=1.3e-4) | H172 EP5 (lr=1e-4) | Δ |
|---|---:|---:|---:|
| val_WSS | **7.4043** | 7.2734 | **+0.131** |
| val_VP | 4.6284 | 4.8146 | **−0.186** |
| val_SP | 4.6367 | 4.3169 | +0.320 |
| val_ABU | 6.7385 | 6.6743 | +0.064 |

**Lead reversed:** EP4 was −0.252pp ahead; EP5 → +0.131pp behind H172. Kill gate ≤7.42 PASSES by 0.016pp marginally. Channel-asymmetric: LR 1.3× helps VP (−0.19pp), hurts WSS/SP (+0.13/+0.32pp). Continue to EP10 gate ≤6.78.

**H181 (frieren, PR #1503) — EP8 landed, on borderline EP10 gate:**

| EP | step | val_WSS | Δ vs H172 | Δ vs H147 |
|---:|---:|---:|---:|---:|
| 5 | 54,879 | 19.52 | +12.25 | +12.77 |
| 6 | 65,855 | 13.68 | +6.71 | — |
| 7 | 76,831 | 10.60 | +3.73 | — |
| 8 | 87,807 | 8.901 | +2.10 | +2.25 |

EMA-0.99995 init_mass at EP8=1.24%, EP10=0.41% (still mid-wash). Descent decelerating: −5.8 → −3.1 → −1.7pp/EP. EP10 linear extrapolation ~7.4 → **PASS by 0.1pp at critical gate ≤7.5**. If EP10 > 7.5, KILL. H181 trajectory is "5 EPs lagged" vs H172 in EMA-equivalent terms — best case it matches H172 at terminal (not beats).

**Wave-5 active map:**
- **dl24-tanjiro:** H183 Per-Channel Decoder Heads (PR #1510) — dispatched 03:20Z, status:wip
- **dl24-fern:** H184 WSD LR Schedule (PR #1513) — dispatched 04:13Z, status:wip
- **dl24-nezuko:** H182 LR 1.3× compound (PR #1506) — EP5 lead reversed, EP10 gate next
- **dl24-frieren:** H181 EMA-99995 (PR #1503) — EP8 done, EP10 critical gate next

## 03:25Z snapshot — H172 CLOSED NON-MERGE; H183 dispatched to tanjiro; H178 terminal ~03:37Z; H182 EP4 lead sustained; H181 EP7 washout continuing

**H172 CLOSED (PR #1469, tanjiro, run `7d83go4z`):**
Test metrics (EP28 best-EMA checkpoint):
- test_WSS=**6.5893** vs H147=6.5409 → **+0.0484pp BEHIND** (primary regress)
- test_SP=**3.6101** vs cap 3.577 → **+0.033pp BREACH**
- test_VP=3.5429 (clears 3.643 cap), test_ABUPT=5.7394 (clears 5.844 cap)
- **NON-MERGE:** EMA-0.9999 mid-train mechanism confirmed (EP6-EP20 lead), BUT does not survive convergence. SP is anti-correlated with high-decay EMA. Hypothesis **FALSIFIED** on primary metric and floor contracts.

**H183 dispatched to tanjiro (PR #1510):** Per-channel decoder heads — split shared surface_out MLP 4×1 into independent channel heads. Highest-upside wave-5 idea; directly targets the capacity bottleneck between cp (3.5%, well-fit) and τ_z (8.5%, lagging). Expected gain −0.1 to −0.3pp test_WSS.

**H178 (fern, PR #1493) — EP11-EP15 plateau, terminal ~03:37Z:**

| EP | val_WSS | val_VP | val_SP | val_ABU |
|---:|---:|---:|---:|---:|
| 11 | 6.8294 | 4.0598 | 4.0684 | 6.1744 |
| 12 | 6.8300 | 4.0465 | 4.0684 | 6.1713 |
| 13 | 6.8296 | 4.0357 | 4.0672 | 6.1684 |
| 14 | 6.8349 | 4.0377 | 4.0783 | 6.1746 |
| 15 | 6.8372 | 4.0351 | 4.0778 | 6.1746 |

WSS plateaued since EP11 (~6.830). VP stagnated at 4.04 — structural BREACH +0.40pp over 3.643 floor. EP16 terminal ETA ~03:37Z. **NON-MERGE sealed; awaiting SENPAI-RESULT then close + dispatch H-W5-1 WSD LR Schedule to fern.**

**H182 (nezuko, PR #1506) — EP4 boundary confirmed, lead sustained:**

| EP | H172 (ref) | H182 | Δ vs H172 | Gate |
|---:|---:|---:|---:|---|
| 1 | 52.018 | 48.701 | −3.32pp | PASS |
| 2 | 50.497 | 43.508 | −6.99pp | PASS |
| 3 | 16.632 | 13.528 | −3.10pp | PASS |
| **4** | **8.695** | **8.443** | **−0.252pp** | **PASS** |

Kill criterion was ≤ 8.85. H182 at EP4 = 8.443, −0.252pp AHEAD of H172 at same EP. EMA contamination at EP4 = 11.1% — residual −0.252pp lead is pure signal from 1.3× LR. EP5 boundary (step 54,879) imminent ~04:00Z.

**H181 (frieren, PR #1503) — EP7 mid-wash, step ~79k:**
EMA-99995 washout continuing. At ~03:12Z: step=79,434, EP7.2, summary val_WSS=10.60 (mid-epoch, not final EP7 boundary). EP7 likely landed near 11.0-11.5 (extrapolating from EP6=13.678 at −5.8pp/EP and acceleration). EP10 critical gate (step 109,759) ~04:30Z.

**Wave-5 dispatch plan:**
- **Tanjiro:** H183 Per-Channel Decoder Heads (PR #1510) — dispatched 03:20Z
- **Fern:** H184 WSD LR Schedule (H-W5-1) — dispatch ~03:40Z after H178 SENPAI-RESULT
- **Nezuko/Frieren:** monitoring active WIP (H182/H181)

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

- **H172 EP30 LANDED at 6.6517** (step 329,279, rt 22.6h); val_WSS plateau confirmed (EP28=6.6481, EP29=6.6528, EP30=6.6517). Test harvest pending (run still in "running" state). Projected test_WSS ~6.60-6.62 = +0.06-0.08pp BEHIND H147. **NON-MERGE on WSS** verdict held; all VP/SP/ABUPT floors clear. Awaiting SENPAI-RESULT from tanjiro.
- **H178 EP15 landed at 6.8372** (step 164,639) — slight uptick from EP14 (6.8349). WSS plateau + EP16 boundary pending (~03:20Z). VP 4.035 BREACH confirmed structural.
- **H182 EP4 boundary imminent** (step 43,901; currently step ~43,401 at ~02:50Z = ~2min). EP3 lead −3.10pp sustained.
- **H181 EP10 critical gate ~04:00Z** (step 74,237 currently, target 109,759).

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
