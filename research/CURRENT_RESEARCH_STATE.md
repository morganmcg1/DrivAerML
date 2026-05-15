# SENPAI Research State

- **Date:** 2026-05-15 (latest invocation: 2026-05-15 ~01:25 UTC)
- **Branch:** tay
- **W&B project:** wandb-applied-ai-team/senpai-v1-drivaerml-ddp8

## Latest invocation actions (2026-05-15 01:15–01:30Z)

- **Closed PR #1114** (tanjiro learnable WSS channel weights): terminal SENPAI-RESULT `test_WSS=7.726%, val_abupt=7.066%` at EP3 (budget-truncated). +0.40pp val_abupt over matched 3-EP baseline (mempfubx 7.465%) but driven by EP1 transient drift (weights briefly dropped to ~50% of init, then quadratic-well-regularized back to baseline by EP3 within 3% of init). Mechanism null at convergence. Methodology data preserved.
- **Reassigned tanjiro → PR #1124** (EMA decay 0.9995 + 18h budget): single-flag experiment, slower EMA half-life ≈ 1386 steps vs default 693 steps. Tests whether late-converging τ_z benefits from longer EMA averaging window. Full 13-EP convergence test, comparison to no-SDF tay ceiling 6.99%.
- **Sent PR #1116 back to edward** (per-channel WSS heads, draft state): matched-budget A/B at EP3 truncated showed −0.66pp test_WSS, −0.09pp test_vol_p, −0.23pp test_SP — every metric improved vs single-head baseline `mempfubx`. **First clean positive Wave 28.5 signal.** But test_WSS=7.671% does not beat no-SDF ceiling (6.99%), so requires 18h budget convergence confirmation: if matched-budget delta holds at EP13, test_WSS → 6.33%, would tie/beat ensemble SOTA. Re-running with `SENPAI_TIMEOUT_MINUTES=1100`, no other changes.
- **τ_z/τ_x ratio finding (edward EP3)**: per-channel heads UNIFORMLY uplift capacity, not τ_z-specifically — ratio went 1.44 → 1.52 (wrong direction). Mechanism is "decoupled head capacity", not "τ_z specialization". Implies follow-on work needs deeper/wider τ_z head specifically (overlaps with thorfinn #1123 τ_z subnet).

---

## PRIMARY RESEARCH DIRECTIVE (Issue #1056, 2026-05-12, ongoing)

**Beat Wall Shear Stress SOTA from Transolver-3 (5.85%) while not degrading vol_p or surface pressure.**

- **WSS target:** `test_WSS` < **5.85%**
- **Non-negotiable constraints:** `test_vol_p` ≤ 3.643% AND `test_SP` ≤ 3.577% (PR #972 levels)
- **Baseline for all new single-model runs:** PR #972 SDF-stratified stack

**WSS Gap (post-PR #1102):**
- Single-model best: **6.727%** (PR #972) → need −0.88pp
- Ensemble best (compliant): **6.3263%** (PR #1102 K=8 Caruana) → need **−0.476pp**

Most recent human check-in: 2026-05-14 14:17 UTC — **"NO MORE ENSEMBLES! Its the lazy route to better results, we want genuine breakthroughs, not incremental improvements based on ensembling which we know we can deploy at any point to improve results."** (Issue #1056 comment from morganmcg1). Ensemble experiments are BANNED until explicitly unlocked. Status updates posted at ~12:35 UTC and ~15:00 UTC.

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

### Primary: WSS Magnitude Bottleneck Attack (Wave 28 onwards — single-model only)

**New mechanism finding from PR #1097 close (tanjiro, WSS direction loss NEGATIVE):**
- WSS **direction is essentially solved** — cos_sim stabilises at 0.996 (~5° angular error) by EP2.
- **91–96% of remaining WSS residual is magnitude error.**
- This pivots the campaign from "direction-aware" experiments (which #1094, #1096, #1097 all targeted) toward **magnitude-targeted** mechanisms (rel_l2 loss, magnitude penalty) and **frame-equivariance** (in-plane rotation aug).

### Pool Saturation — CONFIRMED (PR #1103 closed 2026-05-14 13:30Z)

The current 4-member candidate pool {`56bcqp3m`, `29nohj67`, `a0yoxy85`, `ghh0s4ne`} is Pareto-saturated under convex combinations:
- PR #1102 K=8 Caruana (MERGED) — near-globally-optimal at discrete 1/8 grid
- PR #1099 K=3 WSS-targeted (CLOSED) — converged to identical K=3 subset as #1064
- PR #1103 SLSQP continuous optimisation (CLOSED) — confirmed K=8 within ~0.03 L1 of global continuous optimum; best-case val_WSS improvement = 0.0039pp (0.06% relative); val_SP ≤ 3.577% **infeasible** on this pool (simplex floor ~3.72%, every member ≥ 3.98%)

**Active lever for ensemble gains:**
1. **Pool extension via new single-model members** — only remaining path (ensembles BANNED per human directive)

⚠️ **ENSEMBLES BANNED** — Per morganmcg1 Issue #1056 directive 2026-05-14 14:17Z: no new ensemble experiments until explicitly unlocked. PR #1108 (bias-corrected ensemble) was superseded by PR #1109 (τ_z focal loss) before training started; #1108 is effectively dead.

---

## Active WIP PRs (as of 2026-05-15 ~01:25Z)

### Capacity / baseline runs (both CLOSED — no-SDF ceiling confirmed at test_WSS ≈ 6.99%)
| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #1078 | alphonse | ~~Asymmetric eval surface 131k~~ | **CLOSED 22:42Z** — test_WSS=6.996% (+26.8bp vs SOTA), test floors regress; val→test ratio was 1.020 not projected 0.935; hypothesis falsified |
| #1100 | thorfinn | ~~Capacity uplift — model-slices 256~~ | **CLOSED 23:45Z** — test_WSS=6.989% (+26.2bp vs SOTA), test_SP +7.1% above floor; landed EP20 at 18h budget cap; confirms no-SDF tay structural ceiling at test_WSS ≈ 6.99% |

### Wave 28 — ALL 4 CLOSED (Wave 28 closed 19:43–21:33Z, see Wave 28 Closures below)

### Wave 28.5 (current generation) — 1 sent-back, 1 closed, 5 still in flight
| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| ~~#1114~~ | tanjiro | ~~Learnable WSS channel loss weights~~ | **CLOSED 01:13Z** — terminal `test_WSS=7.726%`, mechanism null at convergence (weights drift back to ~baseline by EP3), +0.4pp matched-baseline win from EP1 transient (implicit volume-first curriculum); reassigned → #1124 |
| **#1116** | edward | **Per-channel WSS output heads** — decouple tau_x/tau_y/tau_z heads | **CHANGES REQUESTED 01:25Z** (draft) — terminal `test_WSS=7.671%, val_abupt=6.915%` at EP3; matched-budget A/B: every metric improves vs `mempfubx` baseline (test_WSS −0.66pp). Re-running at 18h budget to confirm at convergence. If delta holds, projected test_WSS ≈ 6.33% (would tie ensemble SOTA). |
| **#1118** | askeladd | **OHEM v2 spike-clipped + reduced λ** — fixes #1110 scale-collapse via spike-clipping; λ=0.25 (was 0.5); 2-ep warmup | Active WIP; OHEM warmup ends end of EP2 ~00:46Z; EP3 gate ~02:21Z |
| **#1119** | fern | **GradNorm short-cycle (t_max=6, ep=6)** — full-convergence test of prior-vs-learned weights at shorter cycle | Active WIP; EP1 learned τ_z weight=1.047 vs prior 2.0 (same de-emphasis pattern as #1111); EP6 terminal ~07:00Z |
| **#1120** | nezuko | **Spatial-prior surface sampling (-x + \|z\|)** — light spatial-prior surface bias; ρ=+0.31 PASS pre-train diagnostic vs |WSS| | Active WIP; **EP2 catastrophe-gate PASS by wide margin** (val_abupt=7.84% vs #1113 13.12% at same EP); EP3 gate ~02:20Z |
| **#1121** | frieren | **WSS magnitude-only decomposition + 18h budget** — `λ_dir=0.0`, full 13-ep cosine via `SENPAI_TIMEOUT_MINUTES=1100`; tests Wave 27 "91-96% magnitude" claim | Active WIP; EP1 val_abupt=28.347% (warmup floor); decomp infra confirmed (λ_dir=0 in train logs); EP3 gate ~02:48Z; EP13 ~14:00Z |
| **#1122** | alphonse | **SDF importance sampling port to tay (alpha=4.0)** — highest-EV untested-on-tay lever; ports PR #972's SDF-stratified vol sampling | Active WIP; launched 22:45Z post-#1078 close; pod healthy (kubectl confirmed 90-100% GPU util × 8 ranks); EP3 gate ~04:00Z |
| **#1123** | thorfinn | **τ_z dedicated subnet** — 2-layer MLP head attacking residual axis test_τ_z ≈ 9.05% | Active WIP; launched 23:50Z post-#1100 close |
| **#1124** | tanjiro | **EMA decay 0.9995 + 18h budget** — single-flag test of slower EMA half-life (~1386 vs 693 steps) for late-converging τ_z | Active WIP; assigned 01:18Z post-#1114 close; full 13-EP convergence test |

---

## Wave 28 Closures (2026-05-14 19:43Z–21:33Z) — methodology data captured, all reassigned

| PR | Student | Result | Key Mechanism Finding | Reassigned As |
|----|---------|--------|----------------------|---------------|
| #1109 | edward | val_WSS=8.766% EP3 (+1.6pp vs no-decomp ref) | Spatial focal α=2.0 amplifies per-point WSS errors at hot-spots faster than they can train down; underweights smooth bulk; baseline isn't smooth-dominated | #1116 per-channel heads |
| #1110 | askeladd | OHEM scale-collapse @ EP3 | Top-20% mining catastrophically scale-collapses without spike-clip; magnitude of L_hard explodes vs base loss | #1118 OHEM v2 spike-clipped |
| #1111 | fern | GradNorm test floors regress (test_vol_p +0.5pp, test_SP +0.4pp) | GradNorm de-emphasizes τ_z prior (hardcoded 2.0 weight); short-cycle test needed to disambiguate prior-vs-learned at convergence | #1119 GradNorm short-cycle |
| #1112 | frieren | Truncated EP3.5 @ 270-min wall-clock; calibration validated (mag head ratio=0.979 at half-cooked) | Mag head infrastructure works; full budget needed for convergence test | #1121 mag-only + 18h budget |
| #1113 | nezuko | val_abupt=8.04% EP3 (KILL) | Curvature is anti-correlated WSS proxy (ρ=-0.11); curvature-weighted sampling steers attention AWAY from high-WSS regions | #1120 spatial-prior (ρ=+0.31) |

## Wave 27 Closures (2026-05-14 ~13:45Z) — CATASTROPHIC FAILURE

All 4 experiments failed at EP3 with val_abupt 27–32% (4× above EP3 KILL gate of 7.6%). Root causes:

| PR | Student | val_abupt@EP3 | Root Cause |
|----|---------|---------------|------------|
| #1104 | fern | ~27% | L1 magnitude penalty `|‖τ‖−‖τ_gt‖|` creates conflicting gradients vs MSE loss; loss scale mismatch blows up training |
| #1105 | tanjiro | ~30% | Relative L2 `(pred-gt)²/‖gt‖²` numerically explodes when GT~0; near-zero WSS regions produce infinite loss |
| #1106 | frieren | ~28% | Physical-coordinate normal-frame rotation corrupts geometry signal — coordinate transformation invalidates learned features |
| #1107 | nezuko | ~32% | Yaw augmentation destroys physical orientation; model cannot learn orientation-dependent aerodynamics |

Common diagnosis: Wave 27 hypotheses all modified the **loss function or input transformation** at a fundamental level without sufficient numerical safeguards. The supplementary-loss OHEM approach (Wave 28) is designed to avoid these failure modes by adding a *supplementary* term (not replacing the base loss) with warmup and floor guards.

## Wave 26 Additional Kill (2026-05-14)

| PR | Student | Result | Key Mechanism Finding |
|----|---------|--------|----------------------|
| #1081 | askeladd | KILL @ EP10 (val_abupt=7.97%) | slw=3.0 surface loss weight — too aggressive; distorts vol_p head; baseline slw=2.0 is optimal |

## Wave 26 Closures (2026-05-13 → 2026-05-14)

| PR | Student | Result | Key Mechanism Finding |
|----|---------|--------|----------------------|
| #1094 | frieren | KILL @ EP3 (val_abupt=7.465%) | Normal-frame supervision built in normalised space — non-orthonormal |
| #1095 | nezuko | NEGATIVE (test_WSS=7.761% +1.03pp) | GradNorm mechanism healthy but starved vol head; curriculum is load-bearing |
| #1096 | edward | NEGATIVE (test_WSS +0.261pp vs ref) | Tangent-frame features redundant with normals; z-hat fallback discontinuity |
| #1097 | tanjiro | NEGATIVE (val_abupt=6.847% > KILL) | Direction NOT the bottleneck (cos_sim=0.996) |
| #1099 | fern | CONVERGED (same K=3 as #1064) | WSS-targeted greedy on 4-member pool converges to identical subset |
| #1102 | fern | **MERGED — new ensemble SOTA** | K=8 Caruana extracts ghh0s4ne WSS signal at 12.5% weight; NOW BANNED FROM EXTENSION per human directive |

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

## Next-Wave Hypothesis Queue

Wave 28.5 is FULLY IN FLIGHT — 6 single-model students busy, 2 capacity students (alphonse #1078, thorfinn #1100) running. Zero idle students.

Queue for Wave 29 (after Wave 28.5 results land ~tomorrow):

1. **Multi-scale surface attention** — second surface encoder at 0.5× token density to capture macro-flow features.
2. **WSS-loss-aware EMA half-life** — vary `--ema-decay` so EMA captures late-training WSS-specialised weights better.
3. **Heteroscedastic WSS loss** — model both mean and variance per surface point; downweight high-aleatoric regions.
4. **τ_z frequency analysis** — Fourier decompose tau_z predictions vs GT to find spatial frequency bands where error is concentrated; use to motivate loss or architecture changes.
5. **Surface point sampling Voronoi tessellation** — sample uniformly over surface area (not raw vertex density) to remove sampling bias from non-uniform mesh refinement.
6. **Spatial focal at α=0.5 (lower bound test)** — if α=2.0 amplified hot-spots too fast (#1109), maybe α=0.5 is a better operating point with smoother modulation.
7. **τ_z-specific subnet** — small dedicated MLP head only for τ_z taking only the surface features that distinguish spanwise-resolved patterns (boundary layer separation vs free-stream).

⚠️ Permanently retired (catastrophic failure): yaw aug (#1107), magnitude penalty (#1104), rel_l2 (#1105), normal-frame rotation (#1094, #1106), curvature-weighted surface sampling (#1113 — wrong-sign proxy).

---

## Infrastructure Status

### GitHub Token Rate Limiting (RESOLVED 2026-05-14)
Senpai PR #3445 merged 06:42Z deployed per-student token fix + REST API migration. Fleet was back online by ~07:30Z. No further rate-limit-driven idle GPU incidents reported in current invocation.

### Pod Health
All 8 students have active pods (kubectl: `senpai-drivaerml-ddp8-*` deployments, 1/1 ready). DDP via 8× H100 96GB per student. Zero idle students as of 23:50Z. PR distribution: tanjiro #1114, edward #1116, askeladd #1118, fern #1119, nezuko #1120, frieren #1121, alphonse #1122 (SDF port), thorfinn #1123 (τ_z subnet).

---

## Key Findings to Date

- **WSS error is magnitude-dominated** (91–96% of residual, not direction) — pivot away from direction-aware experiments
- **tau_z (spanwise) still worst axis** (test_tau_z=8.2585% on PR #1102) — primary remaining target
- **Wave 27 catastrophic lesson**: NEVER replace base MSE loss — always use supplementary/additive formulations; loss scale mismatches and numerical instability (div-by-near-zero) destroy training even at 27–32% val_abupt; Wave 28 OHEM designed as additive supplement with 2-ep warmup to avoid this
- **Relative L2 loss is unstable** (PR #1105) — near-zero GT WSS regions produce unbounded loss; avoid any loss form with GT in denominator without explicit safeguards
- **slw=3.0 surface weight too aggressive** (PR #1081 killed) — baseline slw=2.0 is optimal
- **ENSEMBLES BANNED** (human directive 2026-05-14 14:17Z) — all new work must improve the single-model SOTA
- **Corrected dataset** (2026-05-11) eliminated artificial ~3× vol_p OOD gap — biggest research-program insight
- **SDF-stratified vol importance sampling** (PR #972) is single-model SOTA: val_abupt=6.126%, test_WSS=6.727%
- **Ensemble SOTA** (PR #1102 K=8 Caruana) test_WSS=6.3263% — first compliant ensemble below 6.33%
- **4-pool Pareto-saturated** (PR #1103 CONFIRMED) — K=8 within 0.03 L1 of global continuous optimum; val_SP ≤ 3.577% infeasible on this pool (simplex floor ~3.72%); new pool members are the operative lever
- ~~**Bias-corrected ensemble** (PR #1108)~~ — closed (superseded by τ_z focal loss #1109; ensemble research BANNED)
- **Training-time vol sampling** matters more than loss weighting or architecture depth for vol_p
- **Throughput regression risk on data-pipeline experiments** (nezuko #1113 self-diagnosed 12× slowdown from 20s/case curvature compute serialised through 4 workers; fix = precompute-and-cache; advisor must spec precompute step in any future data-pipeline assignment)
- **Curvature is a bad WSS proxy** (PR #1113 closed) — surface curvature is anti-correlated with |WSS| (ρ=-0.11); using curvature to oversample steers attention AWAY from high-WSS regions. Spatial position (`-x + |z|`) achieved ρ=+0.31 by contrast (PR #1120).
- **270-min wall-clock budget hits Wave 28 recipe at EP3.5** (#1111, #1112, historical #1095 all truncated) — recipe runs 76 min/epoch; full 13-ep cosine needs ~16h. Two responses available: recipe shrink (short t_max, fern #1119) or budget bump (`SENPAI_TIMEOUT_MINUTES=1100`, frieren #1121, matches alphonse #1078 working regime).
- **GradNorm de-emphasizes τ_z hard-coded prior** (#1111 close) — when learned, GradNorm reduces τ_z weight from prior 2.0 toward 1.4, which regresses test_vol_p and test_SP floors. Question: is the 2.0 prior over-tuned, or is the learned weight wrong? Short-cycle test (#1119 fern) measures this at full convergence.
- **OHEM scale-collapse** (#1110 close) — top-k mining catastrophically collapses without spike-clipping; magnitude of L_hard scales superlinearly when targeting top-20% of L distribution.
- **Spatial focal α=2.0 amplifies hot-spot error faster than training rate** (#1109 close) — per-point focal modulation creates concentrated gradients on outliers; baseline isn't bulk-smooth-dominated so amplification destabilizes optimization.
- **val→test ratio is NOT stable across eval configurations** (#1078 close) — asymmetric eval 131k produced val→test ratio of 1.020, not the 0.935 anchored on PR #972. The 0.935 ratio is recipe-specific (SDF stack), not transferable. Advisor SOTA projections must use test results from comparable-recipe runs, not synthetic val × historical ratio. This is a methodology guard for the entire program.
- **18h budget recipe validated end-to-end** (#1078 close): `SENPAI_TIMEOUT_MINUTES=1100` ran 17 epochs cleanly at ~62 min/ep (faster than initially projected). All future Wave 28+ runs can adopt it confidently; frieren #1121 has already.
- **Capacity-uplift ceiling on no-SDF tay is val_abupt ≈ 6.31%** (#1078 EP16 / #1100 EP16 close). Beyond that, the bottleneck is training-time sampling, not parameter count. Justifies #1122 (alphonse SDF port).
- **No-SDF tay structural ceiling at test_WSS ≈ 6.99%** (#1078 + #1100 close, two independent mechanisms). Asymmetric eval 131k and slices=256 capacity uplift both converge here at full convergence. Test floors regress under both. Any "beat SOTA without SDF" claim must beat 6.99% — capacity alone cannot. Direct paper-relevant finding.
- **τ_z is the program-wide residual axis** (test_τ_z ≈ 9.05% across all no-SDF runs). Consistently 30-45% worse than τ_x and ~18bp worse than τ_y. Justifies #1123 (thorfinn τ_z dedicated subnet) attacking representational capacity for τ_z specifically.
- **Initial-state debug crash** (tanjiro #1114 val_abupt=65.34% on 1-ep debug, then 8-rank DDP retry also crashed) — root cause likely learnable-weight unbounded growth; mitigated by lr=1e-3 separate group + L2 reg 1e-4 + 2-ep warmup option
- **Post-xattn capacity additions** 0-for-3 (PRs #884, #891, #906) — do not add layers after surf→vol xattn
- **Rotation aug** (PR #925): aggressive yaw+pitch degrades; mild yaw-only (≤45°) being tested in PR #1107
- **Normal-frame WSS in normalised space** fails (PR #1094); physical-frame variant (#1106) is the corrected attempt
- **Tangent-frame features** redundant with surface normals (PR #1096) — model already has the information
- **Direction loss** redundant with weighted MSE (PR #1097) — cos_sim=0.996 by EP2 without it
- **GradNorm + fixed-65k vol** fails because vol curriculum is load-bearing (PR #1095)
- **Infra:** `--data-root` (not `--data-path`); mount point `/mnt/new-pvc/` (not `/mnt/pvc/`)
- **EMA lag:** `eval_raw_vs_ema=False` means only EMA weights evaluated — early-epoch EMA metrics appear worse than raw
