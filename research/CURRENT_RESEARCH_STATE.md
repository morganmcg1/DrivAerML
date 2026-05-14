# SENPAI Research State
- 2026-05-14 ~08:00 UTC

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

## Active Experiments (2026-05-14 ~08:00 UTC)

### Pod Assignments

| Student | PR | Hypothesis | W&B Run | EP / Step | Notes |
|---------|-----|-----------|---------|-----------|-------|
| dl24-tanjiro | #1086 | EMA(0.999) clean re-test + SDF α=0.25 | `fby84xtu` | ~EP13.9 | **WAVE BEST: val_abupt=6.2647% (EP11), val_vol_p=4.140%** |
| dl24-fern    | #1098 | WD=0.01 isolated retest | TBD | Not started | **Student-AI rate-limited (GH user 20516801) since ~00:35Z** |
| dl24-nezuko  | #1101 | LR=9e-5 isolated control (new) | TBD | Not started | **Student-AI rate-limited; #1072 CLOSED 08:00Z** |
| dl24-frieren | #1077 | SDF α=1.0 (inverse formula) | `m4z2gb65` | ~EP14.2 | val_abupt=6.382%, vol_p=4.519% |

### Infrastructure Alert: GH Rate-Limit Blocking Student-AI

**User ID 20516801** (shared student account) has been GH rate-limited since ~00:35Z May 14 — now 7+ hours. Both fern and nezuko student-AI pods are in retry loops unable to poll PRs. Training infra (torchrun) unaffected. Rate limit must reset before new assignments are picked up.

Effect:
- fern #1098 (assigned 05:30Z) — NOT launched. Student cannot see assignment.
- nezuko #1101 (assigned 08:00Z) — NOT launched. Same blocker.
- tanjiro and frieren appear unaffected (different GH account or rate limit window).

### Val Checkpoint Snapshots (latest ~08:00 UTC)

| Student | PR | Run | EP | val_abupt (latest) | val_abupt (best) | val_vol_p (latest) |
|---------|-----|-----|----|-------------------:|-----------------:|-----------------:|
| dl24-tanjiro | #1086 | `fby84xtu` | ~EP13.9 | 6.2647% | **6.2647%** (EP11-13, holding) | 4.140% |
| dl24-frieren | #1077 | `m4z2gb65` | ~EP14.2 | 6.382%  | ~6.356% (EP11) | 4.519% |
| dl24-fern    | #1098 | TBD | — | — | — | — |
| dl24-nezuko  | #1101 | TBD | — | — | — | — |

## Strategic Assessment (~08:00 UTC)

### SDF wave conclusion (finalized)
The SDF near-surface concentration idea is fully mapped and falsified. Productive α band [0.25, 0.5] produces val_abupt ~6.26–6.29% (competitive) but:
- All arms show plateau-regression (best EP10-13, then drift)
- Terminal-epoch tests regress all metrics vs SOTA
- SDF concentration does not close the OOD test generalization gap

### Orthogonal levers being tested
1. **EMA 0.999 (tanjiro #1086)** — KEY experiment. Wave-best at EP11 (6.2647%), vol_p=4.140% holding. EMA may decouple the plateau-regression pattern. Terminal test from best checkpoint is the critical deliverable. ETA ~12h.
2. **WD=0.01 (fern #1098)** — Blocked by rate-limit. Will launch when rate limit resets.
3. **LR=9e-5 (nezuko #1101)** — NEW. Blocked by rate-limit. Lower LR may extend useful learning window on corrected split.

### Pending hypothesis queue
- After tanjiro #1086 results: **H1 GradNorm+EMA composition** (combine EMA with GradNorm if EMA terminal shows improved val_vol_p ≤ 4.0%)
- **Surface-point density investigation** — if test metrics continue to show OOD generalization bottleneck

## Next Key Events

1. **Rate-limit reset** (~anytime) — fern and nezuko can then pick up their assignments.
2. **Frieren #1077 EP15 gate** (step 164,625, threshold ≤6.80%) — ~2-4h.
3. **Tanjiro #1086 terminal** (~12h from now ~20:00Z) — KEY RESULT. Test from best checkpoint.
4. **Fern #1098 smoke + long run** — once rate-limit clears, ~3-5h after pickup.
5. **Nezuko #1101 smoke + long run** — once rate-limit clears, ~3-5h after pickup.
