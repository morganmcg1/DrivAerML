# SENPAI Research State

- **Date:** 2026-05-14 (latest invocation: 2026-05-14 ~12:40 UTC)
- **Branch:** tay
- **W&B project:** wandb-applied-ai-team/senpai-v1-drivaerml-ddp8

---

## PRIMARY RESEARCH DIRECTIVE (Issue #1056, 2026-05-12, ongoing)

**Beat Wall Shear Stress SOTA from Transolver-3 (5.85%) while not degrading vol_p or surface pressure.**

- **WSS target:** `test_WSS` < **5.85%**
- **Non-negotiable constraints:** `test_vol_p` ≤ 3.643% AND `test_SP` ≤ 3.577% (PR #972 levels)
- **Baseline for all new single-model runs:** PR #972 SDF-stratified stack

**WSS Gap (post-PR #1102):**
- Single-model best: **6.727%** (PR #972) → need −0.88pp
- Ensemble best (compliant): **6.3263%** (PR #1102 K=8 Caruana) → need **−0.476pp**

Most recent human check-in: 2026-05-14 09:37 UTC — "ok i think we're back on track. how is our training looking?" Status update posted at ~12:35 UTC.

---

## CORRECTED DATASET IN EFFECT (since 2026-05-11)

Issue #1053 (deployed 2026-05-11) fixed a case-split/indexing bug that eliminated an artificial ~3× vol_p OOD gap.

**Corrected dataset path:** `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`
**Split:** val=34 cases/7295 views, test=50 cases/11091 views

All new runs MUST use the corrected dataset path and `--data-root` flag (not `--data-path`).

---

## Current Best Results (Corrected Split)

### **Ensemble SOTA (PR #1102 — K=8 Caruana with-replacement, WSS-optimised)**
- val_abupt = **5.7452%** | test_abupt = **5.5196%**
- val_vol_p = 3.4360% | test_vol_p = 3.5397%  ← satisfies ≤ 3.643%
- val_WSS = 6.5195% | **test_WSS = 6.3263%**  ← TRUE WIN
- val_SP = 3.7234% | test_SP = 3.3529%  ← satisfies ≤ 3.577%
- test_tau_x = 5.6071% | test_tau_y = 6.8397% | **test_tau_z = 8.2585%** (still worst axis)
- W&B: `bq1gaewq` (Arm D greedy), `ems8ekee`, `s7pirpr1`, `qf1lqwz0`
- **Members:** `56bcqp3m`×3, `29nohj67`×2, `a0yoxy85`×2, `ghh0s4ne`×1
- **Effective weights:** {56bcqp3m:0.375, 29nohj67:0.250, a0yoxy85:0.250, ghh0s4ne:0.125}

### Prior Ensemble SOTA (PR #1064 K=3 greedy, superseded by #1102)
- val_abupt = 5.7758% | test_abupt = 5.5199% | test_WSS = 6.3712%

### Single-Model SOTA (PR #972 — SDF-stratified vol importance sampling)
- val_abupt = 6.126% | test_abupt = 5.844%
- val_vol_p = 3.798% | test_vol_p = 3.643%  ← constraint boundary
- test_SP = 3.577%  ← constraint boundary
- test_WSS = 6.727%
- W&B: source=`56bcqp3m`, eval=`zxnhtagj`

### Rank 2 Single-Model (PR #968 — Stochastic vol subsampling)
- val_abupt = 6.278% | test_abupt = 5.986% | test_WSS = 6.825%
- W&B: source=`a0yoxy85`, eval=`qbg9pkmx`

---

## Gate Criteria

### Single-Model EP3 Gates (current tay stack — no SDF importance sampling)
- **PASS:** val_abupt ≤ **7.2%** AND val_vol_p ≤ 4.5%
- **MARGINAL:** val_abupt ≤ 7.6% AND val_vol_p ≤ 5.0%
- **KILL:** otherwise

(Historical PR #972 SDF stack gates were ≤ 6.2% / ≤ 6.5% — those reflect SDF-stratified sampling that is NOT on tay; do not apply to current single-model runs.)

### WSS-Targeted Single-Model Win Criteria (becomes new pool member)
- test_WSS ≤ 6.50% AND test_vol_p ≤ 3.643% AND test_SP ≤ 3.577% AND val_abupt ≤ 6.20%

### Ensemble Win Criteria (true new SOTA after PR #1102)
- val_abupt < **5.7452%** AND test_vol_p ≤ **3.643%** AND test_WSS < **6.3263%**

---

## Current Research Focus and Themes

### Primary: WSS Magnitude Bottleneck Attack (Wave 27)

**New mechanism finding from PR #1097 close (tanjiro, WSS direction loss NEGATIVE):**
- WSS **direction is essentially solved** — cos_sim stabilises at 0.996 (~5° angular error) by EP2.
- **91–96% of remaining WSS residual is magnitude error.**
- This pivots the campaign from "direction-aware" experiments (which #1094, #1096, #1097 all targeted) toward **magnitude-targeted** mechanisms (rel_l2 loss, magnitude penalty) and **frame-equivariance** (in-plane rotation aug).

### Pool Saturation Observation

The current 4-member candidate pool {`56bcqp3m`, `29nohj67`, `a0yoxy85`, `ghh0s4ne`} is Pareto-saturated:
- PR #1102 K=8 Caruana (this round, MERGED) — converges on continuous weighting
- PR #1099 K=3 WSS-targeted (closed — converged to same K=3 subset as #1064)
- PR #1103 SLSQP continuous optimisation (in flight) — final check on continuous optimum

**Further ensemble gains require NEW pool members.** Wave 27 is exclusively focused on producing single-model variants with low WSS / compliant vol_p that extend the pool diversity.

---

## Active WIP PRs (Wave 27 launched 2026-05-14 ~12:35Z)

### Wave 26 continuations (single-model + capacity)
| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #1078 | alphonse | Asymmetric eval surface 131k (2× WSS res at inference only) | EP4 val_abupt=6.511%; tracking baseline ±0.06pp; EP6 gate ~13:55Z |
| #1081 | askeladd | Surface loss weight slw=3.0 (full 30-ep relaunch) | W&B `qwi82vym` 11:04Z; proof EP9 vol_p=3.7439% ahead of SOTA |
| #1100 | thorfinn | Capacity uplift — model-slices 256 vs PR #972 baseline 128 | EP3 val_abupt=6.768%, ~0.5pp ahead of typical EP3 (W&B `k33hscuc`) |
| #1103 | edward | SLSQP continuous weight search over 4-simplex (CPU-only) | Just assigned 12:30Z — pure Python optimisation, no GPU |

### Wave 27 (4 fresh assignments at 12:35Z) — all WSS-targeted, all PR #972 baseline
| PR | Student | Hypothesis | Source |
|----|---------|------------|--------|
| **#1104** | fern | **WSS magnitude penalty** `λ·|‖τ‖−‖τ_gt‖|` L1 (λ=0.1, 0.3 arms) | From PR #1097 close (direction-loss mech analysis) |
| **#1105** | tanjiro | **Per-channel rel_l2 loss** on τ axes (auto-emphasise high-shear) | From PR #1097 close (tanjiro's own suggested follow-up) |
| **#1106** | frieren | **Physical-coordinate normal-frame WSS** (fix #1094 norm-space bug) | From PR #1094 close (frieren's own follow-up #1) |
| **#1107** | nezuko | **In-plane yaw rotation aug** (15°/45° arms, Sim3-equivariance) | From PR #1096 close (advisor follow-up; yaw-only ≤45°) |

---

## Wave 26 Closures (2026-05-13 → 2026-05-14)

| PR | Student | Result | Key Mechanism Finding |
|----|---------|--------|----------------------|
| #1094 | frieren | KILL @ EP3 (val_abupt=7.465%) | Normal-frame supervision built in normalised space — non-orthonormal. Follow-up = physical-coordinate (now PR #1106) |
| #1095 | nezuko | NEGATIVE (test_WSS=7.761% +1.03pp) | GradNorm mechanism healthy but starved vol head; curriculum is load-bearing |
| #1096 | edward | NEGATIVE (test_WSS +0.261pp vs ref) | Tangent-frame features redundant with normals; z-hat fallback discontinuity. Follow-up = yaw aug (now PR #1107) |
| #1097 | tanjiro | NEGATIVE (val_abupt=6.847% > KILL) | Direction NOT the bottleneck (cos_sim=0.996). Follow-up = magnitude penalty (now #1104), rel_l2 (now #1105) |
| #1099 | fern | CONVERGED (same K=3 as #1064) | WSS-targeted greedy on 4-member pool converges to identical subset |
| #1102 | fern | **MERGED — new SOTA** | K=8 Caruana extracts ghh0s4ne WSS signal at 12.5% weight, vol_p stays compliant |

---

## Baseline Training Recipe (current tay stack — NOT PR #972 SDF stack)

⚠️ **IMPORTANT:** the PR #972 SDF-stratified vol sampling code (`--sdf-importance-sampling --sdf-alpha 4.0`) was **never merged into tay**. Do NOT include those flags in any assignment — `argparse` will reject them. The live tay baseline is the stack below (no SDF importance sampling). Single-model EP3 on this baseline lands ~6.7–6.9% val_abupt, not the historical PR #972 6.2%. Gates must be recalibrated accordingly: PASS ≤ 7.2%, MARGINAL ≤ 7.6%, KILL otherwise.

```
--optimizer lion --lr 9e-5 --weight-decay 5e-4
--tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0
--use-ema --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1
--pos-encoding-mode string_separable --use-qk-norm
--rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"
--lr-cosine-t-max 13 --epochs 13
--vol-points-schedule "0:16384:3:32768:6:49152:9:65536"
--no-compile-model
--model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128
--batch-size 4 --validation-every 1
--train-surface-points 65536 --eval-surface-points 65536
--train-volume-points 65536 --eval-volume-points 65536
--use-surf-to-vol-xattn
--data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511
```

The PR #972 single-model SOTA W&B run `56bcqp3m` was trained with SDF-stratified sampling on a different branch (`dl24-frieren/vol-test-domain-augmentation`, commit `291efd2`); that code never landed on tay. Until it does, all new single-model runs are evaluated relative to the no-SDF tay baseline (thorfinn #1100 EP3=6.768% is a representative live trajectory).

---

## Next-Wave Hypothesis Queue (for Wave 28 once Wave 27 settles)

Ordered by expected WSS impact + low complexity:

1. **OHEM hard-example mining for high-WSS regions** — top-K loss reweighting per training step (H6 from RESEARCH_IDEAS_2026-05-13_WSS.md). Mechanistically distinct from rel_l2 (hard cutoff vs soft 1/||gt|| weight).
2. **GradNorm + curriculum + freeze-on-transitions** — nezuko's PR #1095 follow-up #1; unify cheap-early-epochs advantage with adaptive task balancing.
3. **WSS magnitude + direction decomposition heads** — H5 from research_ideas; separate magnitude scalar head + 3-direction L2-normalised head, MSE on log(1+|τ|) + cosine loss on direction.
4. **SDF-stratified SURFACE sampling** (H4) — analogous to PR #972 vol-side, oversample high-curvature surface regions.
5. **Multi-scale surface attention** (H6) — second surface encoder at 0.5× token density to capture macro-flow features.
6. **Test-time SDF stochasticity ensemble** (H7) — 5 stochastic forward passes averaged at inference.
7. **WSS-loss-aware EMA half-life** — vary `--ema-decay` so EMA captures late-training WSS-specialised weights better.
8. **Heteroscedastic WSS loss** — model both mean and variance per surface point; downweight high-aleatoric regions.

---

## Infrastructure Status

### GitHub Token Rate Limiting (RESOLVED 2026-05-14)
Senpai PR #3445 merged 06:42Z deployed per-student token fix + REST API migration. Fleet was back online by ~07:30Z. No further rate-limit-driven idle GPU incidents reported in current invocation.

### Pod Health
All 8 students have active pods (kubectl: `senpai-drivaerml-ddp8-*` deployments, 1/1 ready). DDP via 8× H100 96GB per student.

---

## Key Findings to Date

- **WSS error is magnitude-dominated** (91–96% of residual, not direction) — pivot away from direction-aware experiments
- **tau_z (spanwise) still worst axis** (test_tau_z=8.2585% on PR #1102) — primary remaining target
- **Corrected dataset** (2026-05-11) eliminated artificial ~3× vol_p OOD gap — biggest research-program insight
- **SDF-stratified vol importance sampling** (PR #972) is single-model SOTA: val_abupt=6.126%, test_WSS=6.727%
- **Ensemble SOTA** (PR #1102 K=8 Caruana) test_WSS=6.3263% — first compliant ensemble below 6.33%
- **4-pool Pareto-saturated** — new pool members required for further ensemble gains
- **Training-time vol sampling** matters more than loss weighting or architecture depth for vol_p
- **Post-xattn capacity additions** 0-for-3 (PRs #884, #891, #906) — do not add layers after surf→vol xattn
- **Rotation aug** (PR #925): aggressive yaw+pitch degrades; mild yaw-only (≤45°) being tested in PR #1107
- **Normal-frame WSS in normalised space** fails (PR #1094); physical-frame variant (#1106) is the corrected attempt
- **Tangent-frame features** redundant with surface normals (PR #1096) — model already has the information
- **Direction loss** redundant with weighted MSE (PR #1097) — cos_sim=0.996 by EP2 without it
- **GradNorm + fixed-65k vol** fails because vol curriculum is load-bearing (PR #1095)
- **Infra:** `--data-root` (not `--data-path`); mount point `/mnt/new-pvc/` (not `/mnt/pvc/`)
- **EMA lag:** `eval_raw_vs_ema=False` means only EMA weights evaluated — early-epoch EMA metrics appear worse than raw
