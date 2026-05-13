# SENPAI Research State

- **Date:** 2026-05-13 (latest invocation: 2026-05-13 ~00:30Z)
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
| H1 | Stronger tau_z/tau_y loss weights | tau_y×2.0, tau_z×3.0 (Arm A); tau_y×2.5, tau_z×4.0 (Arm B) | −0.2 to −0.5pp |
| H2 | WSS-targeted greedy ensemble | `--greedy-metric val_WSS` instead of val_abupt | −0.3 to −0.5pp |
| H7 | Mild yaw-only aug screen | yaw≤3°, pitch=0, p=0.3, 4-ep screen | −0.1 to −0.2pp |
| H3 | Surface-normal frame prediction | Predict tau in normal/tangent/binormal frame | −0.2 to −0.4pp |
| H5 | Magnitude+direction decomposition | Split WSS into |τ| + unit direction | −0.1 to −0.3pp |

---

## Active WIP PRs (all on corrected dataset)

| PR | Student | Hypothesis | W&B Run | WSS Relevance |
|----|---------|------------|---------|--------------|
| #1042 | alphonse | INR coord-conditioned vol decoder | TBD | Low — vol_p focused |
| #1050 | edward | Dropout regularization (p=0.1) | `nc7lpobi` | Low — vol_p focused |
| #1052 | thorfinn | RFF sigma sweep (2.0, 4.0) | `yli6kbch` | Medium — spatial freq affects WSS |
| #1057 | fern | Log-space vol_p loss on corrected dataset | running | Low — vol_p focused |
| #1058 | frieren | GradNorm dynamic loss balancing | `nt6w9tqp` | HIGH — auto-weights tau axes |
| #1060 | askeladd | Vol-loss-weight sweep (1.5, 2.0, 3.0) | running | Low — vol_p focused |
| #1061 | tanjiro | Stochastic per-batch vol points | running | Low — vol_p focused |
| #1064 | nezuko | K=3 greedy ensemble (drop ghh0s4ne) | running | HIGH — restores vol_p compliance |
| #1065 | stark | Extended SDF-stratified (45 epochs) | running | Medium — may improve WSS via longer training |

**Key WSS-relevant active PRs:**
- **PR #1058 (frieren GradNorm):** Auto-balances tau_x/tau_y/tau_z loss weights via GradNorm. At EP1: tau_z upweighted 1.9× more than tau_x (1.59× vs 0.83×). EP3 gate check pending — most relevant active WSS experiment.
- **PR #1064 (nezuko K=3):** Drops `ghh0s4ne` (PR #823) from the K=4 ensemble to restore test_vol_p ≤ 3.643%. Expected val_abupt ≈ 5.776%. Critical for establishing a compliant ensemble baseline to build on.
- **PR #1065 (stark 45-epoch):** Extended training of SDF-stratified stack. If val_WSS decreases beyond EP13, this confirms longer training helps WSS.

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
