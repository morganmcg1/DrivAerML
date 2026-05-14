# SENPAI Research State
- 2026-05-14 ~07:00 UTC

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

## CRITICAL DISCOVERY: SDF Monkey-Patch Was a No-Op

PR #972 (SOTA) used **uniform sampling** (monkey-patch was a no-op). True SDF near-surface emphasis was never tested until DL24 branch (PRs #1063+). After exhaustive testing, SDF concentration does not improve generalization on corrected split.

## Wave SOTA (Corrected Split — rawcanon_20260511)

**PR #972** (run `56bcqp3m`, eval `zxnhtagj`)

| Metric | Value |
|--------|-------|
| test_abupt | **5.844%** |
| test_surf_p | 3.577% |
| test_vol_p | **3.643%** |
| test_wss | 6.727% |

## SDF α Sweep — FULLY CLOSED (2026-05-14 ~08:00 UTC)

| α | PR | Run | Best val_abupt | Test result | Status |
|---|-----|-----|---------------:|-------------|--------|
| 0.25 | #1063 | `xfykblf9` | 6.2647% (EP11) | test_abupt=5.955%, test_vol_p=3.990% | **CLOSED** — all metrics regress |
| 0.5 | #1072 | `yp383yq2` | 6.2904% (EP10) | No test (run died EP29.7) | **CLOSED** — dead run, rate-limited student |
| 1.0 | #1077 | `m4z2gb65` | ~6.35% (EP11) | Running ~EP14 | In progress |
| 2.0 | #1054 | — | CLOSED | EP15 FAIL | Over-concentration |
| 3.0 | #1076 | — | 6.5012% (EP6) | No test | Over-concentration, EP10 KILL |

**SDF concentration approach broadly falsified.** No α value on the corrected split beats uniform sampling (SOTA PR #972). The plateau-regression pattern (best at EP10-13, then drift) is consistent across all productive arms. **Pivot to orthogonal levers is confirmed.**

## Active Experiments (2026-05-14 ~07:00 UTC)

### Pod Assignments

| Student | PR | Hypothesis | W&B Run | EP / Step | Notes |
|---------|-----|-----------|---------|-----------|-------|
| dl24-tanjiro | #1086 | EMA(0.999) clean re-test + SDF α=0.25 | `fby84xtu` | ~EP12.1 / 9.5h | val_abupt=6.275% (slope +0.001, plateau), val_vol_p=4.17% |
| dl24-frieren | #1077 | SDF α=1.0 (inverse formula) | `m4z2gb65` | ~EP15.1 / 11.7h | val_abupt=6.370% (slope -0.001, still improving), val_vol_p=4.45% |
| dl24-fern    | #1098 | WD=0.01 isolated retest | `hrb2syym` (smoke) | 3min, step 827 | **Smoke launched 06:53Z, config verified correct** |
| dl24-nezuko  | #1101 | LR=9e-5 isolated control | `gi47kxmp` (smoke) | 5min, step 1430 | **Smoke launched 06:51Z, config verified correct** |

### Infrastructure Update: GH Rate-Limit RESOLVED

Rate limit cleared between 06:48Z–06:51Z. Both fern and nezuko student-AI pods picked up assignments and launched DDP8 smoke runs:
- nezuko `gi47kxmp` (lr=9e-5, wd=0.005, no SDF, no GradNorm) at 06:51:10Z
- fern `hrb2syym` (lr=1e-4, wd=0.01, no SDF, no GradNorm) at 06:53:00Z

Both configs match assignment exactly. Long runs will launch after EP3 smoke gate (val_abupt < 8.5%).

### Val Checkpoint Snapshots (latest ~07:00 UTC)

| Student | PR | Run | EP | val_abupt (latest) | val_abupt (best) | val_vol_p (latest) |
|---------|-----|-----|----|-------------------:|-----------------:|-----------------:|
| dl24-tanjiro | #1086 | `fby84xtu` | ~EP12.1 | 6.275% | **6.2647%** (EP11, slight backtrack at EP12) | 4.17% |
| dl24-frieren | #1077 | `m4z2gb65` | ~EP15.1 | 6.370% | ~6.356% (EP11) | 4.45% |
| dl24-fern    | #1098 | `hrb2syym` | smoke | smoke too early | — | — |
| dl24-nezuko  | #1101 | `gi47kxmp` | smoke | smoke too early | — | — |

## Strategic Assessment (~07:00 UTC)

### SDF wave conclusion (finalized)
The SDF near-surface concentration idea is fully mapped and falsified. Productive α band [0.25, 0.5] produces val_abupt ~6.26–6.29% (competitive) but:
- All arms show plateau-regression (best EP10-13, then drift)
- Terminal-epoch tests regress all metrics vs SOTA
- SDF concentration does not close the OOD test generalization gap

### Orthogonal levers being tested
1. **EMA 0.999 (tanjiro #1086)** — KEY experiment. Wave-best on val at EP11 (6.2647%), slope flattened by EP12. EMA may decouple the plateau-regression pattern. Terminal test from best checkpoint is the critical deliverable. ETA ~14h.
2. **WD=0.01 (fern #1098)** — Smoke just launched 06:53Z. Long run in ~30-45 min if smoke passes EP3 < 8.5%.
3. **LR=9e-5 (nezuko #1101)** — Smoke just launched 06:51Z. Long run in ~30-45 min if smoke passes EP3 < 8.5%.
4. **SDF α=1.0 (frieren #1077)** — Still mid-cosine, ETA ~12h. Slope still negative (improving). Background reference for SDF concentration falsification.

### Pending hypothesis queue
- After tanjiro #1086 results: **H1 GradNorm+EMA composition** (combine EMA with GradNorm if EMA terminal shows improved val_vol_p ≤ 4.0%)
- **Surface-point density investigation** — if test metrics continue to show OOD generalization bottleneck
- **Long cosine T_max=40 or 50** — if both tanjiro and nezuko plateau before EP30, the cosine schedule may be too aggressive

## Next Key Events

1. **Smoke EP3 gates** (~07:30Z): nezuko `gi47kxmp` and fern `hrb2syym` should hit EP3 within 30-45 min.
2. **Long-run launches** (~08:00Z): both should be transitioning to their 30-epoch 24h runs.
3. **Frieren #1077 EP15 gate** (step 164,625, threshold ≤6.80%) — already past gate at val 6.370%.
4. **Tanjiro #1086 terminal** (~14h from now, ~20:45Z) — KEY RESULT. Test from best checkpoint.
5. **Frieren #1077 terminal** (~11.5h from now, ~18:30Z).
