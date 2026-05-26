# SENPAI Research State

- **Date:** 2026-05-26 (~02:00Z)
- **Branch:** tay
- **W&B project:** wandb-applied-ai-team/senpai-v1-drivaerml-ddp8

---

## Current research direction (from Morgan, Issue #1056)

**Primary objective: test_WSS < 5.85%** (Transolver-3 SOTA target). H112 is at 6.752% — gap is 0.902pp. val_abupt is steering metric only; merges judged on test_WSS axis.

---

## SOTA baseline: H112 (PR #1283)

| Metric | val | test |
|---|---|---|
| abupt | 6.1358% | 5.839% |
| WSS | 6.9670% | 6.752% |
| WSS_x | 6.0923% | 5.999% |
| WSS_y | 7.6084% | 7.360% |
| WSS_z | 9.3750% | 8.720% |
| VP | 3.5478% | 3.421% |
| SP | 4.0553% | 3.695% |

**Recipe:** `--lr 9e-5 --weight-decay 5e-4 --batch-size 4 --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0 --volume-loss-weight 0.5 --use-surf-to-vol-xattn --enable-residual-positions --use-drop-path --drop-path-rate 0.10 --epochs 13 --lr-warmup-epochs 1 --lr-schedule cosine --ema-decay 0.999 --grad-clip 1.0`

---

## Wave 38 mechanism taxonomy (as of 2026-05-26 ~02:00Z)

Three-class decomposition of tau_z mechanisms — key program finding:

| Class | Mechanism | Experiments | Status |
|---|---|---|---|
| **SOFTEN** (target modification) | Charbonnier robust M-estimator | H139 tanjiro | C NULL near-terminal, gap WIDENING |
| **SOFTEN** (target modification) | Signed-log reparameterization | H140 nezuko | C NULL near-terminal, gap slowly narrowing |
| **ESCALATE** (loss-weight increase) | tau_z=4.0 (2× H112) | H143 frieren | **ALIVE — EP3 −0.114pp WSS_z LEAD WIDENING** ✅ |
| **ESCALATE** (loss-weight increase) | tau_z=6.0 (3× H112) | H144 fern | EP1 done; EP2 in flight |
| **ESCALATE** (axis extension) | tau_y=3.0 (2× H112) | H145 alphonse | NEW — extends escalation to y-axis |
| **ARCHITECTURAL** (split capacity) | Dedicated wider WSS_z decoder | H138 askeladd | **ALIVE — 9/10 publishes WSS_z LEAD** ✅ |
| **ARCHITECTURAL** (split capacity) | Dedicated wider WSS_y decoder | H146 edward | **NEW — mirrors H138 on tau_y** |
| **SWIGLU decoder** | Decoder-only gating | H135 thorfinn | **CREDIBLE A WIN** — TIED/LEADS H112 val_WSS |
| **SWIGLU backbone** | Full backbone gating | H134 edward | CLOSED — backbone SwiGLU CLASS CLOSED |

**SOFTEN class is definitively CLOSED** (H139 + H140 paired closure, joined H-B2 aux-head closure).  
**Productive frontier: ESCALATE (H143 strongest positive), ARCHITECTURAL (H138 consistent), SwiGLU decoder (H135 credible A WIN).**

---

## Current fleet (2026-05-26 ~02:00Z) — 8/8 students working

| PR | Student | Hypothesis | Terminal ETA | Trajectory |
|---|---|---|---|---|
| #1325 | askeladd | H138 split WSS_z decoder | ~04:00Z | **ALIVE — WSS_z LEAD consistent 9/10 publishes** |
| #1322 | thorfinn | H135 decoder-only SwiGLU | ~02:00Z | **CREDIBLE A WIN — TIED/LEADS H112 val_WSS** |
| #1326 | tanjiro | H139 Charbonnier tau_z | ~02:40Z | C NULL near-terminal, SOFTEN class closure |
| #1327 | nezuko | H140 signed-log tau_z | ~03:00Z | C NULL structural, SOFTEN class closure |
| #1332 | frieren | H143 tau_z=4.0 | ~07:20Z | **STRONGEST POSITIVE — EP3 −0.114pp WSS_z LEAD WIDENING** |
| #1334 | fern | H144 tau_z=6.0 | ~08:00Z | EP1 done, EP2 in flight |
| #1337 | alphonse | H145 tau_y=3.0 | ~12:00Z+ | NEW — axis-extension escalation |
| #1338 | edward | **H146 split WSS_y decoder** | ~12:00Z+ | **NEW — mirrors H138 on tau_y, completes 2×2 factorial** |

---

## Closed experiments since Wave 36 — key findings

| Run | Finding |
|---|---|
| H118-H121-H120-H125 (capacity axis) | **Capacity-axis CLOSED** — all 4 orthogonal axes regress test_WSS +0.066-0.090pp |
| H-B/H-B2 (aux-head) | **Aux-head class CLOSED** — detached aux strictly worse; aux-head benefit is gradient-driven not feature-engineering |
| H128/H134 (backbone SwiGLU) | **Backbone SwiGLU CLOSED** — param-parity costs +0.20pp val_WSS; productive class is decoder-only |
| H139/H140 (SOFTEN target) | **SOFTEN class CLOSED** — Charbonnier +0.19pp WSS_z gap widening; signed-log +0.79pp gap |
| H130/H131/H132 (DropPath compound) | DropPath_max=0.15 productive at depth-6 (H130) but INERT at canonical depth-5 (H132) |

---

## Priority axes for Wave 39 (if Wave 38 mechanisms land alive)

1. **Compound: H143 × H138** — tau_z escalation × split WSS_z decoder (orthogonal mechanisms, both alive)
2. **Compound: H143 × H135** — tau_z escalation × decoder-only SwiGLU
3. **Compound: H143 × H146** — tau_z escalation × split WSS_y decoder (if H146 wins)
4. **Compound: H138 × H135** — split decoder × SwiGLU decoder (same location, potentially synergistic)
5. **Magnitude curve**: H143 (4.0) anchor + H144 (6.0) + potentially tau_z=5.0 or tau_z=3.0 to map response curve
6. **GradNorm** `--gradnorm-mode` for auto-weighting tau_z/tau_y — auto-prioritizes slow channels

### Unexplored directions (lower priority until wave 38 complete)

- Positional encodings for surface geometry (curvature, normals, geodesic distance)
- Data augmentation: geometric noise, pressure perturbation
- Frequency-domain bias (Fourier features for WSS channels)
- Multi-scale decoder (different resolution heads for surface vs volume)

---

## Structural findings bank (program-level, permanent)

1. **Val→test slope catastrophe class**: Capacity-expanded models show flat/positive val→test slopes (H120 −0.020pp, H125 +0.010pp vs H112 −0.215pp). Mechanism: DropPath per-layer schedule dilutes with depth at fixed max.
2. **Aux-head gradient-flow attribution**: Aux-head benefit is gradient-driven; detached aux head is regression. H-B/H-B2 paired experiment proves mechanism.
3. **SwiGLU location-of-benefit**: SwiGLU productive ONLY in decoder (prediction head), NOT in backbone Transolver blocks at 17.5M-param recipe. Layer-stack-deep activation pattern confirmed.
4. **tau_z mechanism class decomposition**: SOFTEN = inert, ESCALATE = alive (H143 widening), ARCHITECTURAL = alive (H138 consistent). Three-way split is the key Wave 38 program finding.
5. **Charbonnier cross-channel bleed**: Charbonnier loss on tau_z hurts WSS_y MORE than WSS_z (+0.34pp vs +0.19pp at step 65k) — suggests optimizer-level cross-channel coupling even for channel-scoped loss changes.
