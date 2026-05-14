# SENPAI Research State
- 2026-05-14 ~12:30 UTC

## Human Research Directive (Issue #882)
**TOP PRIORITY — Volume Pressure Focus:**
- The **TEST volume pressure L2 error** is the only metric that matters for new experiment design
- Do NOT degrade surface error or wall shear stress metrics
- Published SOTA models show significantly better volume pressure test metrics are achievable — large headroom to close
- All new student assignments must be designed with volume pressure improvement as the singular focus

## DATASET ARTIFACT RESOLVED (2026-05-12) — Issue #1053

The persistent +7–8pp val→test volume pressure gap was a **DATASET ARTIFACT** (case-split bug). Under the corrected split (`rawcanon_20260511`) the gap is gone. **All experiments must use corrected dataset:**
`/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`

## INFRASTRUCTURE FIX MERGED: PR #1087 — EMA warm-start shadow re-init

Branch includes fix (`860d08f`): shadow now initialized from live weights at `ema-start-step`. All new EMA runs are clean.

## Wave SOTA (Corrected Split — rawcanon_20260511)

**PR #972** (run `56bcqp3m`, eval `zxnhtagj`)

| Metric | Value |
|--------|-------|
| test_abupt | **5.844%** |
| test_surf_p | 3.577% |
| test_vol_p | **3.643%** |
| test_wss | 6.727% |

## SDF α Sweep — FULLY CLOSED

| α | PR | Run | Best val_abupt | Test result | Status |
|---|-----|-----|---------------:|-------------|--------|
| 0.25 | #1063 | `xfykblf9` | 6.2647% (EP11) | test_abupt=5.955%, test_vol_p=3.990% | **CLOSED** — all metrics regress |
| 0.5 | #1072 | `yp383yq2` | 6.2904% (EP10) | No test (run died EP29.7) | **CLOSED** — dead run |
| 1.0 | #1077 | `m4z2gb65` | 6.3562% (EP11) | terminal pending EP30→EP11-best | In progress, fully plateaued |
| 2.0 | #1054 | — | CLOSED | EP15 FAIL | Over-concentration |
| 3.0 | #1076 | — | 6.5012% (EP6) | No test | Over-concentration, EP10 KILL |

**SDF concentration broadly falsified on corrected split.** Pivot to orthogonal levers confirmed.

## Active Experiments (2026-05-14 ~11:45 UTC)

### Pod Assignments

| Student | PR | Hypothesis | W&B Run | EP / Step | Notes |
|---------|-----|-----------|---------|-----------|-------|
| dl24-tanjiro | #1086 | EMA(0.999) clean re-test + SDF α=0.25 | `fby84xtu` | EP18.7 / 205,163 | 14.6h elapsed / 24h budget. EMA best **6.2647% (EP11)**. Raw trending DOWN 6.51→6.38 — EMA may find new best in EP19-30. |
| dl24-frieren | #1077 | SDF α=1.0 (inverse formula) | `m4z2gb65` | EP20+ / 219,519+ | val_abupt EP20=6.4311% PASS. Plateaued. Best EP11=6.3562%. Terminal protocol: evaluate from EP11 best checkpoint. |
| dl24-fern    | #1098 | WD=0.01 isolated retest | `q4eok915` (long) | EP3 PASS, ~EP5 mid | **val_vol_p=3.674% at EP3** — exceptional early signal (SOTA test_vol_p=3.643%). Long run launched 10:11Z. EP5 gate ~13:20Z. |
| dl24-nezuko  | #1101 | LR=9e-5 isolated control | `5qumfbrs` | **EP5 PASS** | **val_vol_p=3.574% at EP5 — already 0.07pp BELOW SOTA test target.** val_abupt=6.4955%. Trajectory monotonic descent. Tightened gates set for EP10/15/20/25. **WINNER CANDIDATE.** |

### Val Checkpoint Snapshots (latest ~11:45 UTC)

| Student | PR | Run | EP | val_abupt (latest) | val_abupt (best) | val_vol_p (latest) |
|---------|-----|-----|----|-------------------:|-----------------:|-----------------:|
| dl24-tanjiro | #1086 | `fby84xtu` | EP18.7 | 6.291% (EMA) | **6.2647% (EP11)** | 4.346% (EMA) |
| dl24-frieren | #1077 | `m4z2gb65` | EP20+ | 6.4311% | 6.3562% (EP11) | 4.6080% |
| dl24-fern    | #1098 | `q4eok915` | EP3 | 6.6782% | 6.6782% (EP3) | **3.6741%** |
| dl24-nezuko  | #1101 | `5qumfbrs` | **EP5** | **6.4955%** | **6.4955% (EP5)** | **3.5740%** ← BELOW SOTA test |

## Strategic Assessment (~11:45 UTC)

### Promising new signal: nezuko EP5 ALREADY BELOW SOTA test_vol_p
At EP5, nezuko LR=9e-5 has val_vol_p=**3.574%** — already 0.07pp below SOTA `test_vol_p=3.643%`, with 25 epochs remaining. Trajectory is monotonic descent on val_abupt (22.1→7.66→6.76→6.53→6.50), val_vol_p (13.86→4.90→3.71→3.58→3.57), and val_wss (24.1→8.55→7.74→7.49→7.44). Only wss_z showed a 0.02pp uptick EP4→EP5 (within noise).

Fern WD=0.01 at EP3 hit val_vol_p=3.674% (similar magnitude). EP5 gate expected ~13:20Z. If fern also crosses below 3.6% at EP5, both isolated controls are confirmed winner candidates. **Tightened gates are now set on nezuko #1101** to maintain pressure on the descent trajectory.

For context, the prior SOTA training stack typically saw val_vol_p ~ 3.9-4.1% at EP3. This is the strongest early-epoch volume_p signal in recent waves.

### Tanjiro update: raw val trending down past EP11
The "bit-identical to fern through EP15" framing was based on EMA shadow. The **raw** val_abupt is now trending DOWN from EP11 (6.510%) to EP18 (6.381%) — a 0.13pp downward drift. Since EMA tracks raw with ~1000-step delay, the EMA shadow may find a new best in EP19-30. Terminal report should evaluate from EP11 best-val checkpoint AND consider running test from final EMA shadow if EP25+ EMA dips below 6.2647%.

### Frieren #1077: clear close candidate
SDF α=1.0 plateaued for 9+ epochs. Best EP11=6.3562%. Terminal protocol confirmed: evaluate from EP11 checkpoint; expected test_abupt ~5.96-6.05% (will NOT beat SOTA). Will close once SENPAI-RESULT lands.

### Pending hypothesis queue
1. **H1 GradNorm+SDF composition** — deprioritized (SDF concentration falsified).
2. **Long cosine T_max=40/50** — promising if fern/nezuko plateau before EP30.
3. **WD/LR composition (if fern OR nezuko wins individually)** — combine winning regularization with EMA(0.999) or extended cosine.
4. **Surface-point density investigation** — if all OOD test metrics continue to plateau, increase 11k surface train budget.

## Next Key Events

1. **Tanjiro #1086 EP20-21** (~13:00-13:30Z) — gate ≤6.70%
2. **Fern #1098 EP6 gate** (~13:00Z) — gate ≤6.8% PASS, ≤7.2% MARGINAL, >7.2% KILL. Critical to confirm val_vol_p ≈ 3.7% trajectory holds.
3. **Nezuko #1101 EP5-6 gates** (~13:00Z) — confirm val_vol_p ≈ 3.7% trajectory.
4. **Frieren #1077 terminal** (~14:00Z) — likely SENPAI-RESULT for close.
5. **Tanjiro #1086 terminal** (~20:45Z) — KEY result for EMA checkpoint hypothesis.
6. **Fern + Nezuko terminal** (~04-06Z May 15) — clean orthogonal controls. If either beats SOTA, this is a winner candidate.

## Plateau Pattern (across the wave)

All productive arms peak at EP10-13 val_abupt ≈ 6.26–6.35% under the 30-epoch / cosine T_max=30 budget. Val→test gap is now a true generalization gap (~−0.4 to −0.5pp). **If WD=0.01 or LR=9e-5 breaks this pattern by giving stronger late-epoch convergence (val_abupt at EP25-30 < 6.20%), it's the new SOTA direction. If they plateau too, the next pivot must be schedule/architecture (T_max=40, depth=8, or surface density).**
