# SENPAI Research State

- **Date:** 2026-05-14 (latest invocation: 2026-05-14 ~00:20Z)
- **Branch:** tay
- **W&B project:** wandb-applied-ai-team/senpai-v1-drivaerml-ddp8

---

## PRIMARY RESEARCH DIRECTIVE (Issue #1056, 2026-05-12)

**Beat Wall Shear Stress SOTA from Transolver-3 (5.85%) while not degrading vol_p or surface pressure.**

- **WSS target:** `test_WSS` < 5.85%
- **Non-negotiable constraints:** `test_vol_p` ≤ 3.643% AND `test_SP` ≤ 3.577% (PR #972 levels)
- **Baseline for all new runs:** PR #972 SDF-stratified stack (see training recipe below)

**WSS Gap:**
- Single-model best: 6.727% (PR #972) → need −0.88pp
- Ensemble best: 6.330% (PR #1059 K=4) → need −0.48pp (but K=4 violates test_vol_p ≤ 3.643%)

---

## CORRECTED DATASET IN EFFECT (since 2026-05-11)

Issue #1053 (deployed 2026-05-11) fixed a case-split/indexing bug that eliminated an artificial ~3× vol_p OOD gap.

**Corrected dataset path:** `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`
**Split:** val=34 cases/7295 views, test=50 cases/11091 views

All new runs MUST use the corrected dataset path and `--data-root` flag (not `--data-path`).

---

## Current Best Results (Corrected Split)

### Single-Model SOTA (PR #972 — SDF-stratified vol importance sampling)
- val_abupt = 6.126% | test_abupt = 5.844%
- val_vol_p = 3.798% | test_vol_p = 3.643%  ← constraint boundary
- test_SP = 3.577%  ← constraint boundary
- test_WSS = 6.727%
- W&B: source=`56bcqp3m`, eval=`zxnhtagj`

### Rank 2 Single-Model (PR #968 — Stochastic vol subsampling)
- val_abupt = 6.278% | test_abupt = 5.986%
- test_WSS = 6.825%
- W&B: source=`a0yoxy85`, eval=`qbg9pkmx`

### Best Ensemble — WSS (PR #1059 — K=4 greedy, ABUPT-optimized)
- val_abupt = 5.758% | test_abupt = 5.594%
- test_WSS = 6.330% ← best WSS
- test_vol_p = 3.889% ⚠ VIOLATES constraint (>3.643%)
- Members: `56bcqp3m` (PR #972), `29nohj67` (PR #958), `a0yoxy85` (PR #968), `ghh0s4ne` (PR #823)

### Best Compliant Ensemble (PR #880 — K=6 greedy, old dataset)
- val_abupt = 6.031% | test_abupt = 6.010%
- test_WSS = 6.708% ← best compliant ensemble WSS
- (built pre-corrected-dataset; needs rebuild with corrected-split models)

---

## Gate Criteria

### Single-Model Gates (new experiments)
- **PASS:** val_abupt ≤ 6.2% AND val_vol_p ≤ 4.5%
- **MARGINAL:** val_abupt ≤ 6.5% AND val_vol_p ≤ 5.0%
- **KILL:** otherwise

### WSS-Targeted Single-Model Win Criteria
- val_WSS ≤ 6.4% AND test_WSS ≤ 6.5% with test_vol_p ≤ 3.643% and test_SP ≤ 3.577%

### Ensemble Win Criteria (true SOTA)
- val_abupt < 5.758% AND test_WSS < 6.330% AND test_vol_p ≤ 3.643%

---

## Current Research Focus and Themes

### Primary: WSS Improvement Campaign

**Root cause of WSS gap:** tau_z (spanwise shear) is the primary bottleneck — consistently worst axis (~9–10% on old dataset, still highest error on corrected split). Key interventions historically:
1. Surf→vol cross-attention (PR #823) — biggest architecture win
2. SDF-stratified sampling (PR #972) — −0.26pp test_WSS
3. Fixed tau loss weights PR #571 (tau_y×1.5, tau_z×2.0) — targeted tau_z upweighting
4. Ensemble diversity (PR #1059) — 6.727% → 6.330% (−0.40pp)

**Research ideas file:** `/research/RESEARCH_IDEAS_2026-05-13_WSS.md` — 10 ranked hypotheses for WSS improvement campaign.

### WSS Hypothesis Priority Queue (for next idle students)
| Priority | Hypothesis | Key Change | Est. Gain |
|----------|-----------|------------|-----------|
| H1 | WSS-targeted greedy ensemble | `--greedy-metric val_WSS` with corrected-split models | −0.3 to −0.5pp |
| H2 | Mild yaw-only aug screen | yaw≤3°, pitch=0, p=0.3, 4-ep screen | −0.1 to −0.2pp |
| H3 | Surface-normal frame prediction | Predict tau in normal/tangent/binormal frame | −0.2 to −0.4pp |
| H4 | Magnitude+direction decomposition | Split WSS into |τ| + unit direction | −0.1 to −0.3pp |
| H5 | Stronger tau_z loss weight (τ_z×3.0) | Isolated from other changes, EP13 run | −0.1 to −0.3pp |
| H6 | Multi-scale surface attention | Additional surface encoder at 0.5× token density | −0.1 to −0.3pp |
| H7 | Test-time ensemble via SDF stochasticity | Average 5 stochastic forward passes at inference | −0.05 to −0.15pp |

---

## Active WIP PRs (all on corrected dataset, Round 25+)

| PR | Student | Hypothesis | W&B Run | Status | WSS Relevance |
|----|---------|------------|---------|--------|--------------|
| #1067 | thorfinn | RFF octave ladder EP30 (sigmas 1.0–16.0) | `0zutsus4` | EP~19/30, val_abupt=6.346% ← projected EP30 ~6.23% | Medium |
| #1075 | tanjiro | EMA decay=0.9995 start-step=2000 | `9i1uu66t` | EP4 screen started | Medium |
| #1078 | alphonse | Asymmetric eval 131k surface (2× resolution at inference) | running | 30-ep run started (EP1 ~95 min) | Medium |
| #1081 | askeladd | SLW=3.0 full 30-ep re-run (Arm B slw=4.0 KILLED EP1) | multi-run | Arm A re-run in progress | Medium |
| #1084 | edward | Surface curvature features H+K for WSS | TBD | Draft cleared, starting | High |
| #1085 | frieren | Adaptive per-point WSS loss via surface curvature | TBD | Draft cleared, starting | High |
| #1089 | nezuko | GradNorm ema_proxy α=2.0 full EP30 | `goh7mght` | EP3 passed (6.666%), continuing to EP5 | High |
| #1090 | fern | GradNorm-partial (sp/vp/tau_x only, freeze tau_y/tau_z) | `g2o80osc` | EP1 ~8% through | High |

**Key WSS-relevant active PRs:**
- **PR #1089 (nezuko GradNorm ema_proxy):** α=2.0, EP3 val_abupt=6.666%, excellent trajectory. EP5 gate ≤7.5%. If this stabilizes to ~6.0% by EP10, it will be the cleanest GradNorm win yet.
- **PR #1090 (fern GradNorm-partial):** Restricts GradNorm balancing to sp/vp/tau_x only, freezes tau_y/tau_z at their explicit weights (1.5×/2.0×). Tests whether letting GradNorm interfere with tau_y/tau_z hurt the #1058 experiment.
- **PR #1084 (edward curvature features):** H+K surface curvature as extra input features for WSS prediction — addresses the root cause (model lacks geometric signal about high-curvature regions where WSS is hardest).
- **PR #1085 (frieren curvature-adaptive WSS loss):** Upweights loss at high-curvature surface points where WSS prediction is most difficult.
- **PR #1067 (thorfinn RFF octave ladder):** Extended 30-ep confirmation run. At EP19 val_abupt=6.346% — likely new best single-model if trajectory holds to EP30 projection of ~6.23%.

---

## Baseline Training Recipe (PR #972 Stack)

All new WSS experiments should use this as the base:
```
--optimizer lion --lr 9e-5 --weight-decay 5e-4
--tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0
--ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1
--pos-encoding-mode string_separable --use-qk-norm
--rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"
--lr-cosine-t-max 13 --epochs 13
--vol-points-schedule "0:16384:3:32768:6:49152:9:65536"
--model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128
--batch-size 4 --validation-every 1
--train-surface-points 65536 --eval-surface-points 65536
--train-volume-points 65536 --eval-volume-points 65536
--use-surf-to-vol-xattn
--sdf-importance-sampling --sdf-alpha 4.0
--data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511
```

---

## Infrastructure Status

### GitHub Token Rate Limiting (ACTIVE ISSUE — escalated 2026-05-13)
The shared student token (`senpai-launch-secrets-drivaerml-ddp8`) is perpetually rate-limited. 8 students × ~2 GraphQL calls per 300s poll = ~9600 points/hour vs 5000/hour limit. This caused tanjiro and frieren to miss PR assignments by ~7 hours.

**Workarounds in use:** Manual draft-clearing and direct pod instruction for affected students.
**Recommended fix:** Per-student dedicated GitHub tokens OR switch `student_poll_for_work` from GraphQL to REST API (60k calls/hour limit).

---

## Key Findings to Date

- **WSS primary bottleneck:** tau_z (spanwise shear) — consistently worst axis; must be targeted via loss weighting, architecture, or data augmentation
- **Corrected dataset (2026-05-11)** eliminated artificial ~3× vol_p OOD gap — biggest insight of research program
- **SDF-stratified vol importance sampling** (PR #972) is single-model SOTA: val_abupt=6.126%, test_WSS=6.727%
- **Ensemble WSS SOTA** (PR #1059 K=4): test_WSS=6.330% but test_vol_p=3.889% violates constraint
- **Training-time vol sampling strategy** matters more than loss weighting or architecture depth for vol_p
- **GradNorm (PR #942 old axis):** vol_p is NOT gradient-starved; GradNorm on tau axes (PR #1058) is different and promising
- **Post-xattn capacity additions** 0-for-3 (PRs #884, #891, #906) — do not add layers after surf→vol xattn
- **Rotation aug** (PR #925): aggressive yaw+pitch degrades WSS most; mild yaw-only (≤3°, p=0.3) untested follow-up
- **K=6 greedy ensemble** (PR #880) needs rebuild with corrected-split models — old dataset ensemble
- **Infra:** `--data-root` (not `--data-path`); mount point `/mnt/new-pvc/` (not `/mnt/pvc/`)
- **EMA lag:** `eval_raw_vs_ema=False` means only EMA weights evaluated — early-epoch EMA metrics appear worse than raw
