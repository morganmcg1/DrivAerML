# SENPAI Research State

**Updated**: 2026-05-26 ~08:50Z | Branch: `tay` | SOTA: H112 PR #1283

---

## Primary Objective

**test_WSS < 5.85%** (Transolver-3 SOTA, per Morgan Issue #1056)

Current SOTA (H112): val_abupt 6.1358% / test_WSS 6.752% / test_WSS_z 8.720%
Gap to target: test_WSS needs −0.90pp improvement.

Merge gate: val_abupt < 6.1358%, test_WSS ≤ 6.727%, test_VP ≤ 3.421%, test_SP ≤ 3.577%

---

## Wave 38 Final Closure (as of 2026-05-26)

### THE PROGRAM-WIDE FINDING

**Any capacity addition at canonical 17.5M recipe val-overfits.**

All six Wave 38 mechanism classes exhausted:

| Class | Experiments | Verdict | Mechanism |
|---|---|---|---|
| Capacity-axis (depth/width) | H118, H120, H121, H125 | CLOSED | Val-overfit slope catastrophe |
| Pure regularization | H132 (DP_max=0.15) | CLOSED C NULL | Null — no improvement |
| SwiGLU gating | H128(OOM), H134(backbone), H135(decoder) | CLOSED | Val-overfit; param overhead acts as capacity expansion |
| Architectural-split | H138 (WSS_z decoder), H146 (WSS_y decoder) | CLOSED | Same slope-flattening pathology — param overhead correlated |
| Aux-head | H-B, H-B2 | CLOSED | Gradient-flow degradation; detached strictly worse |
| SOFTEN (loss) | H139 (Charbonnier), H140 (signed-log) | CLOSED | Anti-aligned with WSS_z goals; heavy-tail residuals ARE the signal |

**Param overhead correlates with slope flattening magnitude** (H132 null → H135 −0.10pp → H138 −0.135pp → H120/H121 −0.20pp). Slope catastrophe is overhead-driven, not mechanism-specific.

**Cross-channel bleed under Lion**: narrow-scope tau_z loss changes produce LARGER regression on tau_y (+0.356pp) than tau_z (+0.254pp) itself — via Lion sign-only update propagation through shared decoder weights.

---

## Wave 39 Active Fleet (2026-05-26 ~08:50Z)

| Student | PR | Hypothesis | Status | Priority |
|---|---|---|---|---|
| frieren | #1332 | H143 tau_z=4.0 ESCALATE | WIP step 62,492 (88%) terminal ~09:50Z | HIGH — late-cosine WSS_z TIED H112 EMA |
| fern | #1334 | H144 tau_z=6.0 ESCALATE | WIP EP9 (70%) terminal ~09:20Z | **PROGRAM-PIVOTAL — first projected A WIN** |
| alphonse | #1337 | H145 tau_y=3.0 axis-extension | WIP | HIGH — axis-extension validation |
| edward | #1338 | H146 split WSS_y decoder | WIP | Monitor — paired-class to H138 |
| thorfinn | #1340 | H147 GradNorm full (alpha=1.5) | WIP EP9+ | HIGH — discovered tau_z>tau_y>tau_x>sp>vp hierarchy |
| askeladd | #1341 | H148 y=0 mirror augmentation | WIP | HIGH — data invariance, EP2 cross-channel bleed observed |
| tanjiro | #1342 | H149 AdamW optimizer swap | WIP | HIGH — program-pivotal Lion vs AdamW ablation |
| nezuko | #1343 | H150 EMA 0.999→0.9999 | WIP | Monitor — late-train averaging

---

## Wave 39 Priority Axes

These are the ONLY mechanism families with unexhausted potential. **Zero capacity increase for any of them.**

### 1. ESCALATE class (H143/H144/H145) — strongest current evidence
- **H144 fern (tau_z=6.0) at EP9: projected val_abupt 6.087% UNDER merge gate, val_WSS_z LEAD −0.281pp, val_WSS_y LEAD −0.204pp — FIRST projected A WIN of Wave 38+39.**
- 3-point monotone-accelerating response curve on val_WSS_z: H112 (2.0)→H143 (4.0) −0.045pp, H143→H144 (6.0) −0.114pp (2.5× larger increment).
- H143 frieren (tau_z=4.0): late-cosine convergence; deficit progressively narrowed from EP3 +0.80pp to step 62,492 ESSENTIALLY TIED H112 EMA (9.370% vs 9.375%).
- Mechanism: increase gradient pressure on hard channels (opposite of SOFTEN, which was anti-aligned).
- **Verdict event: H144 terminal val→test slope at ~09:20Z. Wave 40 cascade depends on this.**

### 2. GradNorm dynamic weighting (H147 thorfinn)
- Task names: `("sp", "tau_x", "tau_y", "tau_z", "vp")`
- Alpha=1.5, tau_y=1.0/tau_z=1.0 balanced starting point
- **Validated**: autonomously discovered tau_z > tau_y > tau_x > sp > vp weight hierarchy. Effective tau_z weight (1.45 × base 2.0 = 2.9) sits BETWEEN H112 (2.0) and H143 (4.0).
- If alive: dynamic loss-weighting sidesteps static ESCALATE search-space problem
- Requires `--no-compile-model` for `full` mode

### 3. Mirror augmentation (H148 — just assigned)
- Zero params, zero architecture change
- Tests: is slope catastrophe a data-memorisation artifact?
- Implementation: `target/data/loader.py:428`, prob=0.5 train-only, yaw flip about y=0
- Falsifiable: if alive → data invariance closes slope catastrophe; if closed → optimizer/architecture problem

### 4. AdamW optimizer (H149 — just assigned)
- Program-pivotal: does slope-flattening persist without Lion?
- `--optimizer adamw --lr 3e-4` vs Lion `--lr 9e-5`
- If alive: Lion sign-accumulation contributes to slope pathology; opens SOFTEN-class revisit with AdamW
- If closed: pathology is optimizer-agnostic — further constrains the search space

### 5. Wave 40 contingency cascade (depends on H144 terminal)
- **H144 A WIN**: H144 × H145 joint escalation (tau_z=6.0 + tau_y=3.0); H144 × H147 (escalate + GradNorm); continue magnitude curve to tau_z=8.0
- **H144 A TIED** (val PASS, test marginal MISS): H144 × H148 mirror compound; H144 × H147 GradNorm
- **H144 slope catastrophe**: pivot harder to H149 AdamW, H150 EMA 0.9999, fresh mechanism families

### Cross-channel bleed under Lion — 3-class paired confirmation
Bidirectionally symmetric under Lion sign-only updates:
- **H139 SOFTEN** (tau_z→tau_y bleed +0.356pp)
- **H146 ARCHITECTURAL-SPLIT** (tau_y→tau_z bleed +2.2pp)
- **H148 DATA-AUGMENTATION** (tau_y mirror→tau_z +1.98pp early EP)

Any narrow-axis intervention propagates through shared decoder weights — must verify cross-channel impact in every loss-weight or augmentation experiment.

---

## Diagnostic Invariants (DO NOT VIOLATE)

- No capacity additions — any param overhead ≥ +1% risks slope flattening
- No ensembles
- Cross-run comparison: W&B raw `val_primary/*` at same step (NOT EMA, NOT student tables)
- Kill thresholds: global_step 10864/32592/48897 (W&B _step leads by 2)
- Mirror augmentation axes: `surface_x[...,1]` (y), `surface_x[...,4]` (normal_y), `surface_y[...,2]` (tau_y), `volume_x[...,1]` (vol y); `volume_y` (pressure) invariant
- DrivAerML axes: x=streamwise, y=lateral (mirror axis), z=vertical

---

## Human Researcher Directives

- **Morgan (Issue #1056, 2026-05-24)**: "test_WSS < 5.85% is THE objective. val_abupt is steering metric only."
- No ensembles — lazy route, want genuine breakthroughs
- All advisor work on `tay` branch; DDP 8 GPUs every run
