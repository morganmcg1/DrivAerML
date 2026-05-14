# SENPAI Research State
- 2026-05-14 ~09:15 UTC

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
| 1.0 | #1077 | `m4z2gb65` | ~6.356% (EP11) | Running mid-EP19 | In progress, plateaued |
| 2.0 | #1054 | — | CLOSED | EP15 FAIL | Over-concentration |
| 3.0 | #1076 | — | 6.5012% (EP6) | No test | Over-concentration, EP10 KILL |

**SDF concentration approach broadly falsified.** No α value on the corrected split beats uniform sampling (SOTA PR #972). The plateau-regression pattern (best at EP10-13, then drift) is consistent across all productive arms. **Pivot to orthogonal levers is confirmed.**

## Active Experiments (2026-05-14 ~09:15 UTC)

### Pod Assignments

| Student | PR | Hypothesis | W&B Run | EP / Step | Notes |
|---------|-----|-----------|---------|-----------|-------|
| dl24-tanjiro | #1086 | EMA(0.999) clean re-test + SDF α=0.25 | `fby84xtu` | EP15 / 164,625 | **EP11 best EMA val_abupt=6.2647%** — bit-identical to fern through EP15. EP15 EMA=6.2784% PASS. Continuing to terminal. |
| dl24-frieren | #1077 | SDF α=1.0 (inverse formula) | `m4z2gb65` | EP18 done, mid-EP19 | val_abupt **fully plateaued 6.356–6.405%** since EP10. Training-best EP11=6.3562%. ETA terminal ~14:00Z. |
| dl24-fern    | #1098 | WD=0.01 isolated retest | `hrb2syym` (smoke) | EP3 PASS 6.9209% | **Smoke PASSED at 09:08Z, long-run launching imminent** |
| dl24-nezuko  | #1101 | LR=9e-5 isolated control | `5qumfbrs` (long) | Just launched 09:06Z | Long run started; 30 epochs DDP8, ~19h ETA |

### Val Checkpoint Snapshots (latest ~09:15 UTC)

| Student | PR | Run | EP | val_abupt (latest) | val_abupt (best) | val_vol_p (latest) |
|---------|-----|-----|----|-------------------:|-----------------:|-----------------:|
| dl24-tanjiro | #1086 | `fby84xtu` | EP15 | 6.2784% (EMA) | **6.2647% (EP11)** | 4.2428% (EMA) |
| dl24-frieren | #1077 | `m4z2gb65` | EP18 | 6.4049% | 6.3562% (EP11) | 4.5270% |
| dl24-fern    | #1098 | `hrb2syym` | smoke EP3 | 6.9209% (smoke) | — | 7.57% (smoke) |
| dl24-nezuko  | #1101 | `5qumfbrs` | just launched | — | — | — |

## Strategic Assessment (~09:15 UTC)

### Tanjiro #1086 = bit-identical to fern through EP15
Tanjiro confirmed (via config deep-dive at EP6 already) that the trajectory is **bit-identical to fern `xfykblf9`** through at least EP15. The original A/B framing (EMA warm-start fix vs no-fix) is invalidated; both runs use `use_ema=True` and the contamination decayed by step ~7,400. This PR now serves as a **clean re-harvest of fern's run with proper test eval from EP11 best checkpoint**. If terminal test from EP11-best ≠ fern terminal test (5.955%), that proves checkpoint selection matters; if equal, both are CLOSED.

### Orthogonal levers
1. **WD=0.01 (fern #1098)** — Smoke clean. Long-run launching now. ETA ~19h to terminal (~04:00Z May 15).
2. **LR=9e-5 (nezuko #1101)** — Long run launched 09:06Z. ETA ~04:00Z May 15.
3. **EMA(0.999) replication (tanjiro #1086)** — EP15 PASS; ETA terminal ~ in 10h. Need to confirm test-from-best-checkpoint differs from fern terminal.
4. **SDF α=1.0 (frieren #1077)** — Plateaued at EP10-18. Will run to terminal for archival completeness. ETA ~14:00Z.

### Pending hypothesis queue
Three candidate next-wave directions (ranked):
1. **H1 GradNorm+SDF composition** — only if SDF α=0.5 arm or EMA arm produces a new test SOTA (currently both falling short on val). Lower priority now that SDF concentration is broadly falsified.
2. **Long cosine T_max=40/50** — the plateau at EP10-15 across SDF/EMA/lr/wd arms suggests current 30-epoch cosine schedule may be too aggressive. Worth testing with the strongest arm config.
3. **Surface-point density investigation** — if all OOD test metrics continue to plateau, the surface sampling budget may be the next lever (currently 11k surface train points).

## Next Key Events

1. **Fern #1098 long-run launch** (within 30 min of 09:15Z)
2. **Frieren #1077 terminal** (~14:00Z) — likely NOT a winner, will close.
3. **Tanjiro #1086 terminal** (~20:00Z) — KEY RESULT for EMA checkpoint-selection hypothesis.
4. **Fern #1098 / Nezuko #1101 terminal** (~04:00Z May 15) — clean orthogonal controls.

## Plateau-Pattern Observation Across the Wave

The cross-arm plateau is now a research signal:
- All productive arms peak at EP10-13 val_abupt ≈ 6.26–6.35%
- val→test gap is no longer the 7-8pp dataset artifact, but a true generalization gap (~−0.4 to −0.5pp difference)
- Multi-arm convergence to ~6.27% val_abupt **with no SDF dependency** suggests this is the current model capacity / cosine-schedule fixed point.

**If tanjiro/fern/nezuko all terminal with test_abupt ≥ 5.9% from best val checkpoint, the next round should pivot to architecture/loss family** (T_max ↑, surface density ↑, or H1 GradNorm composition).
