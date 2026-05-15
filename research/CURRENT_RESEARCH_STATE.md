# SENPAI Research State
- 2026-05-15 ~12:00 UTC (W&B trajectory snapshot)

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
| 1.0 | #1077 | `m4z2gb65` | 6.3562% (EP11) | **CLOSED** — test_abupt=6.042%, test_wss=6.815%, test_vol_p=4.173% — all metrics regress |
| 2.0 | #1054 | — | EP15 FAIL | Over-concentration |
| 3.0 | #1076 | — | 6.5012% (EP6) | EP10 KILL |

## Active Experiments (2026-05-15 ~08:15 UTC)

### Pod Assignments + Live W&B Snapshot (12:00Z)

| Student | PR | Hypothesis | W&B Run (rank0) | Live EP | val_abupt | val_wss | val_vol_p | val_surf_p | Read |
|---------|-----|-----------|---------|---------:|----------:|--------:|----------:|-----------:|------|
| dl24-tanjiro | **#1132** | **H5: curvature additive attention bias** (zero-init, surf_in=7) | `lbi210l2` | EP2.8 (step 31016) | 7.18% (EP2) | 7.58% (EP2) | 6.13% (EP2) | 4.50% (EP2) | **Tracks H2 closely** at EP2 (H2: 7.14/7.51/6.23/4.51) → mechanism preserved. EP3 ~12:45Z. |
| dl24-fern    | #1130 | **H4: per-axis WSS loss weights [1.0, 1.5, 2.5]** | `3i0nnneh` | EP8.3 (step 91310) | **6.21%** | **7.08%** | **3.48%** | 4.11% | **Fleet leader.** val_vol_p UNDER FLOOR (3.643%), val_wss below 7.5% gate. EP10 ~13:00Z. |
| dl24-nezuko  | #1129 | **H3: near-wall volume cross-attn** (SDF<0.05m) | `h75p7dt9` | EP5.7 (step 62460) | **6.38%** | 7.29% | **3.51%** | 4.15% | val_vol_p UNDER FLOOR at EP5. val_wss above gate but trending (-0.10pp/ep). EP6 ~12:25Z. |
| dl24-frieren | #1115 | **H1: wind-exposure proxy** (max(0,-nx) + \|ny\|) | `3rja7gw6` | EP23 (step 252448) | 6.41% | **6.93%** | **4.92%** ⚠️ | 4.11% | **WSS great, vol_p drift UP since EP16 (4.59→4.92) — predicted FLOOR BREACH at terminal.** Same gradnorm task-share pattern as H2. Terminal ~18:50Z. |

**Wave summary**: Fern H4 and nezuko H3 are the live winner candidates (both UNDER vol_p floor). H4 leads on aggregate (val_abupt, val_wss). Frieren H1 is showing the H2 failure pattern (WSS down, vol_p up = gradnorm imbalance from added input channels). Tanjiro H5 is too early to call but step-2 trajectory matches H2 closely.

### Per-Axis WSS Insight (from tanjiro #1086 fby84xtu terminal)

| Component | test value | SOTA-relative | Priority |
|-----------|------------|--------------|---------|
| tau_x | 5.971% | ~1.0× AB-UPT target | secondary |
| **tau_y** | **7.362%** | **~2.0× AB-UPT target** | **HIGH** |
| **tau_z** | **8.747%** | **~2.4× AB-UPT target** | **HIGHEST** |

Cross-flow shear (tau_y, tau_z) dominates. H1/H2 wind-exposure + curvature features directly target this.

### Val Checkpoint Snapshots (latest ~22:55 UTC, RUNNING values from W&B)

| Student | PR | Run | EP / step | val_abupt | val_vol_p | val_surf_p | val_wss |
|---------|-----|-----|----------:|----------:|----------:|-----------:|--------:|
| dl24-tanjiro | #1117 | `1a08e7b`-pid1951208 | **EP3 PASS** | 6.466% | 4.384% | 4.216% | 7.156% |
| dl24-frieren | #1115 | `3rja7gw6` | **EP3 PASS** (no student post yet) | 6.505% | 4.422% | 4.233% | 7.180% |
| dl24-fern    | #1098 | `q4eok915` | 185k (11.2h) | **6.298%** | **3.512%** | 4.113% | 7.200% |
| dl24-nezuko  | #1101 | `5qumfbrs` | 194k (11.5h) | **6.352%** | **3.489%** | 4.125% | 7.295% |

**Headline:** nezuko and fern both val_vol_p well below SOTA test_vol_p=3.643% (margin 0.13-0.15pp at step 185-194k). Terminal projection ~04-06Z May 15 — if held, **new aggregate SOTA candidates**.

### H1 vs H2 EP3 Cross-Flow Comparison (the hypothesis target)

| τ component | baseline (#972 fby84xtu) | frieren H1 EP3 | tanjiro H2 EP3 | H1 delta | H2 delta |
|---|---:|---:|---:|---:|---:|
| τ_x | 5.971% | 6.221% | 6.232% | +0.25pp | +0.26pp |
| τ_y | 7.362% | 7.983% | 7.846% | +0.62pp | +0.48pp |
| τ_z | 8.747% | 9.669% | 9.654% | +0.92pp | +0.91pp |

**H1 ≈ H2 at EP3** — both features encode geometrically related information (wind-exposure and curvature both target separation/edge regions). All τ components are slightly worse than baseline at EP3, which is expected for added features that need epochs to integrate. The downward slopes (especially τ_z at ~-0.038pp/1k-steps for tanjiro) suggest mid-training improvement. **EP10-EP15 will be the real verdict for both H1/H2** — if they diverge here, that reveals which geometric signal is more useful.

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
| **H1** | Wind-exposure geometric proxy (n·u_freestream + \|ny\| as 2 extra input channels) | Direct cross-flow attack-angle signal targets tau_y/z gap | **frieren → PR #1115 ✓ ASSIGNED 18:10Z** |
| **H2** | Surface curvature features (kappa_H + kappa_G as 2 extra channels) | Local shape curvature targets WSS spikes at separation edges | **tanjiro → PR #1117 ✓ CLOSED** — first sub-SOTA WSS (6.668%) but vol_p floor breach +0.340pp. Positive signal: curvature works. |
| **H3** | Near-wall volume cross-attention into surface decoder (SDF<0.05m) | Inject boundary-layer velocity-gradient signal into WSS head | **nezuko → PR #1129 ✓ ASSIGNED (stale, monitoring)** |
| **H4** | Per-axis WSS loss weights [τ_x=1.0, τ_y=1.5, τ_z=2.5] | Direct τ_z/τ_y error-budget attack via loss reweighting (no arch change) | **fern → PR #1130 ✓ ASSIGNED 05:50Z May 15** |
| **H5** | Curvature as additive attention bias (zero-init, no input-dim change) | H2 fix: inject curvature AFTER 7-ch projection, bypasses gradnorm imbalance | **tanjiro → PR #1132 ✓ ASSIGNED 08:10Z May 15** |

**All four build on PR #972 stack** (Lion + 6L STRING 5-sigma + GradNorm α=0.5 + EMA 0.999 + Y-sym 0.5 + bs=1 + 65k surf/vol + SDF α=2.0 + corrected dataset).
**All four explicitly protect vol_p ≤ 3.643% and surf_p ≤ 3.577% with EP6/EP10/EP15 gates.**

Note: If nezuko #1101 or fern #1098 win on test_vol_p, they may continue with combinatorial composition rather than free WSS hypotheses. Reassessment at terminal.

### Cross-advisor findings (tay branch, 00:05Z 2026-05-15 status)

**Tay track late-evening update (no new SOTA tonight on tay):**

1. **Tay structural ceiling at test_WSS ≈ 6.99% without SDF importance sampling.** Two completely different mechanisms (asymmetric eval 131k, slices=256 capacity uplift) converged on 6.99%. Confirms SDF lever in PR #972 is load-bearing for both WSS and test floors. My track already has SDF active (advisor branch contains #972 stack).
2. **Curvature is anti-correlated with |WSS|** (ρ=-0.11) — falsified as a *sampling weight* (tay #1113 killed at val_abupt=8.04% EP3). NOTE: my H2 #1117 uses curvature as **additional input feature** (not sampling weight) — mechanism is different, but the underlying weak correlation is a yellow flag for H2's mid-training trajectory. If H2 EP10 doesn't show τ improvement, this finding becomes strong evidence H2 is also a dead end.
3. **Spatial-position proxy `-x + |z|` has ρ=+0.31 with |WSS|** — much stronger proxy than curvature. Tay's #1120 is testing this as sampling weight. **Queue as candidate H5 for my track if H1/H2 underperform** (could be used as additional input feature OR sampling weight).
4. **GradNorm de-emphasizes τ_z hardcoded prior** — learned weight ~1.0 vs hardcoded 2.0. Suggests current per-channel weighting (1.0/1.5/2.0) may be over-engineered. Could be relevant for compositional wave if fern/nezuko win.

### Cross-advisor Wave 27 Lessons (tay branch, 15:23Z post-mortem on Issue #1056)

The other advisor's WSS-focused Wave 27 failed catastrophically across all 4 arms (EP3 val_abupt 27-32%):

| tay-branch PR | Hypothesis | Failure mode |
|---|---|---|
| #1104 WSS magnitude L1 penalty | L1 magnitude conflicts with L2 reconstruction → gradient instability EP1 |
| #1105 Per-channel rel_L2 loss | Numeric explosion when GT τ near zero (laminar regions) |
| #1106 Normal-frame WSS rotation | Physical coord rotation corrupts Transolver geometry signal |
| #1107 Yaw rotation augmentation | Destroys orientation signal — CFD wind direction is FIXED |

**Hard design constraints from Wave 27:**
- Loss modifications must be **supplementary** (additive to MSE), not replacements that change gradient direction.
- Augmentations must **preserve wind direction** (Y-sym is the only physically valid spatial augmentation; yaw is invalid).
- Loss formulations that risk numerical explosion (relative-L2 with near-zero GT) are unsafe without epsilon-flooring.

**Impact on my H1-H4 queue:**
- **H1 wind-exposure** (additive input channel) — ✅ SAFE: doesn't touch loss, doesn't rotate features.
- **H2 curvature features** (additive input channels) — ✅ SAFE: same as H1.
- **H3 near-wall cross-attention** (architectural, doesn't modify loss) — ✅ SAFE structurally, but adds parameters → overfit risk.
- **H4 per-task GradNorm α** (dynamic loss weighting, still on MSE) — ⚠️ MODERATE: GradNorm preserves gradient direction but α_wss=1.0 arm risky given prior global α=0.75 EP16 blowup. Recommend dropping the α_wss=1.0 arm and keeping only α_wss=0.75.

## Next Key Events (revised 12:00Z based on W&B step rates)

1. **Tanjiro #1132 H5 EP3** (~12:45Z May 15) — first WSS-axis read at val checkpoint.
2. **Nezuko #1129 H3 EP6** (~12:25Z May 15) — student-posted EP3 still pending; W&B side-check shows live data.
3. **Fern #1130 H4 EP10** (~13:00Z May 15) — τ_z/τ_y ratio target ≤1.10; abupt ≤6.60%; wss ≤7.1%.
4. **Frieren #1115 terminal** (~18:50Z May 15, revised) — H1 wind-exposure; vol_p drift suggests close on floor breach.
5. **Tanjiro #1132 / Nezuko #1129 / Fern #1130 terminals** — all ~03-07Z May 16 based on current step rates (~21h per 30-epoch run).

**Note on step rate**: All current runs are at ~4.4-5.6 steps/sec → ~21 hours for 30 epochs (longer than initial 6h estimates). Update advisor cadence accordingly.

## Quiet-state Notes (2026-05-14 ~20:55 UTC)

- All 4 students WIP, all pods 1/1 READY.
- `stale_wip` flag on #1098, #1101, #1115 ignored per operator instruction (08:26Z).
- Frieren #1115 has 0 PR comments despite run live for 1.8h; tolerate until EP3 gate (~22:00Z) — student probably composing the gate-pass writeup. If still silent at EP6, prompt for comment.
- Frieren wave-2 W&B run crashed at 19:38Z; wave-1 healthy. Will surface in EP3 comment.
- Tanjiro #1117 launched ~20:52Z, EP1 at ~17% (1907/10975 iters, 4.5 it/s, no NaN/OOM).

## Follow-up Notes (not blocking current wave)

- **Surface input channel 6 (`area`) is identically zero across batch.** Flagged by tanjiro #1117 EP0 audit. Pre-existing dataset/loader behavior — applies to PR #972 SOTA `56bcqp3m` run too. Effective input dimensionality is 6 (not 7) for the baseline, and 9 (not 10) for tanjiro's run. Not in scope for WSS wave — investigate in a follow-up PR after wave settles. Could be a free WSS gain if `area` channel were actually populated (surface element area is a natural weighting signal for shear).
