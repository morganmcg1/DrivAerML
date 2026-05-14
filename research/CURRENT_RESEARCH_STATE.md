# SENPAI Research State
- 2026-05-14 ~14:30 UTC

## Human Research Directive (Issue #1056 — 2026-05-14, NEW)

**TOP PRIORITY — Wall Shear Stress (WSS) Focus:**
- The **TEST wall shear stress L2 error** is now the primary metric to drive down
- Target: **test_wss < 5.85%** (Transolver-3 reference, current PR #972 SOTA = 6.727%, gap +0.877pp = 13% relative reduction)
- **Strict floors** (must NOT degrade vs PR #972 SOTA):
  - `test_vol_p ≤ 3.643%`
  - `test_surf_p ≤ 3.577%`
- `test_abupt` may regress slightly if WSS gains are large, but should remain competitive
- **NO ENSEMBLES** — single-model only. Per Morgan 14:17Z: "we want genuine breakthroughs, not incremental improvements based on ensembling".
- All NEW WSS hypotheses must build on PR #972's training stack (multi-sigma STRING + Lion + EMA + cosine + bs=1 + 65k points) — already in advisor branch `drivaerml-long-20260504`.

## Prior Directive (Issue #882 — superseded for new work):
- Volume pressure was the prior priority. nezuko #1101 and fern #1098 in-flight runs continue under this directive (winner candidates if they beat test_vol_p=3.643%).
- All in-flight runs MUST report final WSS metrics for cross-comparison against new wave.

## DATASET ARTIFACT RESOLVED (2026-05-12) — Issue #1053

The persistent +7–8pp val→test volume pressure gap was a **DATASET ARTIFACT** (case-split bug). Under the corrected split (`rawcanon_20260511`) the gap is gone. **All experiments must use corrected dataset:**
`/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`

## INFRASTRUCTURE FIX MERGED: PR #1087 — EMA warm-start shadow re-init

Branch includes fix (`860d08f`/backport `15afb57`): shadow now initialized from live weights at `ema-start-step`. All new EMA runs are clean.

## Wave SOTA (Corrected Split — rawcanon_20260511)

**PR #972** (run `56bcqp3m`, eval `zxnhtagj`) — single-model best on advisor branch

| Metric | Value | Status |
|--------|-------|--------|
| test_abupt | **5.844%** | wave SOTA |
| test_surf_p | 3.577% | **floor for WSS wave** |
| test_vol_p | **3.643%** | **floor for WSS wave** |
| test_wss | 6.727% | **TARGET: < 5.85% (Transolver-3)** |

Note: PR #972 "SDF α=2.0" monkey-patch was a no-op (uniform sampling). The SOTA arises from the underlying stack: multi-sigma STRING + Lion + EMA + cosine T_max=30 + bs=1 + 65k points + corrected dataset.

## SDF α Sweep — FULLY CLOSED (broadly falsified on corrected split)

| α | PR | Run | Best val_abupt | Status |
|---|-----|-----|---------------:|--------|
| 0.25 | #1063 | `xfykblf9` | 6.2647% (EP11) | **CLOSED** — test regresses |
| 0.5 | #1072 | `yp383yq2` | 6.2904% (EP10) | **CLOSED** — dead run |
| 1.0 | #1077 | `m4z2gb65` | 6.3562% (EP11) | In progress, terminal ~14:00Z |
| 2.0 | #1054 | — | EP15 FAIL | Over-concentration |
| 3.0 | #1076 | — | 6.5012% (EP6) | EP10 KILL |

## Active Experiments (2026-05-14 ~14:30 UTC)

### Pod Assignments

| Student | PR | Hypothesis | W&B Run | EP / Step | Notes |
|---------|-----|-----------|---------|-----------|-------|
| dl24-tanjiro | #1086 | EMA(0.999) clean re-test + SDF α=0.25 | `fby84xtu` | EP19+ / 209k+ | Best EP11 6.2647%. Raw trending DOWN. Terminal ~20:45Z. **NOT-a-winner (bit-identical to fern).** |
| dl24-frieren | #1077 | SDF α=1.0 (inverse formula) | `m4z2gb65` | EP21+ / 219,519+ | Plateaued. Best EP11=6.3562%. Terminal protocol: evaluate from EP11 best. Terminal ~14:00Z, likely **close as NOT-a-winner**. |
| dl24-fern    | #1098 | WD=0.01 isolated retest | `q4eok915` (long) | EP6+ | EP3 val_vol_p=3.674% — winner candidate. Terminal ~04-06Z May 15. |
| dl24-nezuko  | #1101 | LR=9e-5 isolated control | `5qumfbrs` | EP5+ PASS | **val_vol_p=3.574% at EP5 — 0.07pp BELOW SOTA test target.** Tightened gates EP10-25. **STRONGEST WINNER CANDIDATE.** Terminal ~04-06Z May 15. |

### Val Checkpoint Snapshots (latest ~11:45 UTC)

| Student | PR | Run | EP | val_abupt (best) | val_vol_p (best) | val_wss (latest) |
|---------|-----|-----|----|-----------------:|-----------------:|-----------------:|
| dl24-tanjiro | #1086 | `fby84xtu` | EP19 | 6.2647% (EP11) | 4.1395% (EP11) | ~6.97% |
| dl24-frieren | #1077 | `m4z2gb65` | EP20 | 6.3562% (EP11) | 4.4500% (EP15) | 7.03% |
| dl24-fern    | #1098 | `q4eok915` | EP3 | 6.6782% (EP3) | **3.6741% (EP3)** | 7.6094% |
| dl24-nezuko  | #1101 | `5qumfbrs` | EP5 | **6.4955% (EP5)** | **3.5740% (EP5)** | 7.4400% |

## Strategic Assessment (~14:30 UTC) — WSS PIVOT

### WSS Headroom Analysis
PR #972 wave SOTA test_wss = 6.727%. Transolver-3 target = 5.85%. Gap = 13% relative reduction. All current in-flight runs have val_wss ~7.0-7.6% — none are competitive on WSS. This means **new hypothesis wave specifically targets WSS mechanism**, not aggregate.

### WSS-relevant signals from historical experiments (background scan)
- **Mild tau weighting (`9mm3sz7x`)**: tau_y=1.2 / tau_z=1.3 + LR=9e-5 → test_wss=7.454% (old split, best WSS in early-wave). LR=9e-5 stack already in nezuko #1101.
- **Surface-loss weight 2.0 (`qqtdnlwq`)**: test_wss=7.634% (old split). Modest.
- **Surface-point density**: 11k surface train budget is current; WSS is a surface-only quantity, so increasing density is a natural lever.

### Researcher-agent WSS hypothesis generation
- Async researcher-agent (ID `a382a78f1c3b32adf`) launched 14:23Z.
- Output target: `target/research/RESEARCH_IDEAS_WSS_2026-05-14_1430.md`
- Constraints: no ensemble, no hard tangent constraints, no extreme weights (<0.5 or >2.0), no backbone replacements, must include floors for vol_p ≤ 3.643% and surf_p ≤ 3.577%.

### In-flight winner candidates (volume_p directive — Issue #882):
- **Nezuko #1101 LR=9e-5**: val_vol_p=3.574% at EP5 is exceptional. If it holds through EP30 and beats test_vol_p=3.643%, this is the new SOTA on the *aggregate* — and unlocks a WSS-floor combinatorial wave (LR=9e-5 + WSS lever).
- **Fern #1098 WD=0.01**: val_vol_p=3.674% at EP3. If both nezuko AND fern beat SOTA, compose into LR=9e-5 + WD=0.01 + WSS lever for next wave.

### Terminal protocol for non-winners
- **Frieren #1077** (terminal ~17:00-18:00Z, revised — was at EP20 at 10:40Z): expected test_abupt ~5.96-6.05% (NOT-a-winner). Will close once SENPAI-RESULT lands.
- **Tanjiro #1086** (terminal ~20:45Z): bit-identical to fern, EMA didn't help. Close.

## WSS Hypothesis Queue — READY (from `RESEARCH_IDEAS_WSS_2026-05-14_1430.md`)

Researcher-agent delivered 10 ranked hypotheses at 14:37Z. Top 4 for assignment:

| Rank | Hypothesis | One-line | Assigned to (when idle) |
|------|-----------|----------|-------------------------|
| **H1** | Wind-exposure geometric proxy (n·u_freestream + \|ny\| as 2 extra input channels) | Direct cross-flow attack-angle signal targets tau_y/z gap | **frieren** (first idle) |
| **H2** | Surface curvature features (kappa_H + kappa_G as 2 extra channels) | Local shape curvature targets WSS spikes at separation edges | **tanjiro** (second idle) |
| **H3** | Near-wall volume cross-attention into surface decoder (SDF<0.05m) | Inject boundary-layer velocity-gradient signal into WSS head | **nezuko** (when terminal closes — needs new assignment if not a winner) |
| **H4** | Per-task GradNorm α sweep (α_wss=0.75 and 1.0, α_surf=α_vol=0.5 fixed) | Allow faster WSS rebalancing without global alpha=0.75 blowup risk | **fern** (when terminal closes — if not a winner) |

**All four build on PR #972 stack** (Lion + 6L STRING 5-sigma + GradNorm α=0.5 + EMA 0.999 + Y-sym 0.5 + bs=1 + 65k surf/vol + SDF α=2.0 + corrected dataset).
**All four explicitly protect vol_p ≤ 3.643% and surf_p ≤ 3.577% with EP6/EP10/EP15 gates.**

Note: If nezuko #1101 or fern #1098 win on test_vol_p, they may continue with combinatorial composition rather than free WSS hypotheses. Reassessment at terminal.

## Next Key Events

1. **Frieren #1077 terminal** (~17:00-18:00Z) — close as NOT-a-winner → assign H1 to frieren.
2. **Tanjiro #1086 EP21+** (~15:00Z+ — non-gate, but EMA may dip).
3. **Fern #1098 EP6 gate** (~13:00Z past) and **EP10 gate** (~16:00Z).
4. **Nezuko #1101 EP6+** (~13:30Z+) and **EP10 tightened gate** (~16:30Z, ≤6.30%).
5. **Tanjiro #1086 terminal** (~20:45Z) — close, assign H2.
6. **Fern + Nezuko terminal** (~04-06Z May 15) — clean orthogonal controls. If winners, compose; else assign H3/H4.
