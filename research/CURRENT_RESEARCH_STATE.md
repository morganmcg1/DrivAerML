# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-04 23:25 UTC (Round 13/14 transition — STRING-RoPE 4-experiment Issue #618 directive in flight; SOTA-breaking long run live on PR #599)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Fleet:** 0 idle students; 1 stuck pod (askeladd, Issue #644 filed for human intervention); 7/8 short-track + 4/4 long-track training healthy
- **Tay-deployed students:** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn

---

## ENSEMBLE SOTA — PR #612 nezuko greedy K=7 (pool 24) — val_abupt **6.1751%** / test_abupt **7.5347%**

W&B run `5veexq8r` (group `nezuko-ensemble-greedy-v3`). Pool expanded 23→24 by adding PR #594 run `d777epep` (askeladd rff32, val=6.7258%). Greedy forward selection from Caruana 2004; `d777epep` selected as seed; `nh2ke150` dropped vs prior K=7. Volume_pressure test-vs-val gap remains chronic (~3×) — primary systematic issue for pool 25.

**Ensemble gate:** any new ensemble must beat val_abupt < **6.1751%**.

## SINGLE-MODEL SOTA — PR #592 alphonse depth-L5 (model-layers=5) — val_abupt **6.5985%** / test_abupt **7.9915%**

W&B run `4k25s25e` (group `model-depth-sweep`, EP4 best-val). Beats prior PR #594 (val=6.7258%) by −0.1273pp (−1.90% rel) by increasing transformer depth from L=4 to L=5 on the PR #594 SOTA stack (Lion lr=9e-5, tau_y×1.5/tau_z×2.0, surface_w=2.0, rff16 multi-sigma, STRING-sep + QK-norm). ~15.9M params, ~52GB VRAM, ~270.8 min training. Surface pressure improves to 4.3322% (was 4.455%); near-wall tau also improves.

**Single-model gate:** any new single-model run must beat val_abupt < **6.5985%**.

### Noise floor calibration
- Identical-config rerun shows ~+0.05pp run-to-run variation
- Genuine win threshold: **val_abupt < 6.5485%** (gate − 5bps)
- Borderline (need replicate): 6.5485–6.5985%
- No signal / regression: > 6.5985%

### Canonical SOTA reproduce command (PR #592 stack — depth-L5)

```bash
cd target/ && torchrun --standalone --nproc_per_node=8 train.py \
  --agent <student> --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --lr-cosine-t-max 13 --epochs 13 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --no-compile-model \
  --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536
```

---

## Latest research direction from human researcher team

- **Issue #618 (active, unconditional 4-experiment STRING-RoPE directive)**: all four experiments assigned and in-flight: #621 nezuko (slice-centroid), #624 alphonse (point-level pre-slice), #625 askeladd (no-slice anchor), #626 frieren (AB-UPT-style geometry branch).
- **Issue #644 (just filed by tay advisor 23:18 UTC, escalation TO human team)**: askeladd PR #625 pod has been silent for 3+ hours despite being 1/1 Ready; iteration logs frozen at iteration 183 (20:06 UTC); 0 GPU utilization. Requesting `kubectl rollout restart deployment/senpai-drivaerml-ddp8-askeladd`. **HUMAN ACTION REQUIRED.**
- Issue #606 directive (assign edward `surface-curvature-features` after his prior PR completes) — **fulfilled** via PR #615 (edward/surface-curvature-v2).

---

## Live SOTA-candidate watch (advisor-monitored W&B)

**PR #599 dl24-frieren STRING multi-sigma long run (`sogus8sx`):**
- Best val_abupt to date: **6.5086%** at step 181,301 (− 0.090pp vs single-model SOTA 6.5985%)
- Run state=running; heartbeat 23:16 UTC; last step 189,845; trajectory still oscillating in 6.51–6.57% band, slope still negative
- Awaiting completion + best-checkpoint test harvest before merge consideration
- All advisor monitoring done via direct W&B query; student pod heartbeat appears to lag but `torchrun` is healthy

**PR #608 dl24-nezuko volume-loss upweight long run (`y301z78k`):** EP24 val_abupt=13.96%, vol_p=6.68% — closing on AB-UPT 6.08% target
**PR #611 dl24-fern tau-weighting AdamW long run (`ug6c3nks`):** EP15 val_abupt=12.37%, descending; EP20 gate ≤15%
**PR #623 dl24-tanjiro stronger tau weights long run (v2 `41wo8cne`):** smoke gate PASSED (EP3=20.11%); v2 launched 23:03 UTC after v1 NCCL hang

---

## Currently in-flight (8 active WIP PRs on tay, ZERO idle)

Each student listed against the most recently-assigned PR; older PRs from the same student are multi-arm sweeps still completing additional arms in the background. Note: PR #592 alphonse depth-L5 was MERGED (now SOTA gate 6.5985%) — alphonse reassigned to #619 (NorMuon). PR #612 nezuko ensemble was MERGED (gate 6.1751%) — nezuko reassigned to #621 (slice-centroid STRING-RoPE).

| PR | Student | Lever | Notes |
|---|---|---|---|
| #593 | frieren | lr-min cosine floor sweep (1e-6 vs 1e-5) | Arm A v2 (lr-min=1e-6, run `axpkjoku`) reached EP8 timeout — awaiting full per-epoch comparison vs SOTA |
| #594 | askeladd | RFF spectral budget sweep (8 vs 32 features) — Arm A (rff32, `d777epep`) finished val=6.7173%; Arm B (rff8) ablation | Arm B att #5 (`6z3qcg6n`) running with full per-rank serialization wandb init fix (multi-attempt wandb-DDP race resolved); Arm A no longer beats new L=5 gate but already part of ensemble pool |
| #603 | tanjiro | EMA decay sweep {0.9993, 0.9997, 0.9999} on PR #592 SOTA stack | Arm A (0.9993) aborted at EP2 (val=8.95%, exceeds 8.5% kill gate); Arm B (0.9997) relaunched after wandb init crash |
| #614 | fern | Lion β2 momentum sweep (0.95 vs 0.99 vs 0.999) | Probing under-smoothing of sign estimates for tau channels |
| #619 | alphonse | NorMuon optimizer (normalized Muon for 2D weight matrices) on L=5 SOTA | Bigger swing — fundamentally different optimizer family |
| #621 | nezuko | Slice-centroid STRING-RoPE for Transolver attention | Architectural change to attention spectral coverage |
| #603 | tanjiro | EMA decay sweep {0.9993, 0.9997, 0.9999} on PR #592 SOTA stack | Arm A (0.9993) aborted at EP2 (val=8.95%); Arm B (0.9997) relaunched |
| #593 | frieren | lr-min cosine floor sweep (1e-6 vs 1e-5) | Arm A v2 (lr-min=1e-6) awaiting full per-epoch comparison vs SOTA |
| #640 | edward | NorMuon optimizer on L=5 SOTA (fresh clean assignment) | Replaces #615 surface-curvature-v2; NorMuon hybrid 2D+AdamW1D, lr=2e-4 |
| #641 | thorfinn | Flow-aligned tau: predict in local surface tangent frame (t̂, b̂) | Replaces prior #613 iteration; physics-motivated attack on tau_y/tau_z 2.3–2.7× gap |

**Round 13/14 focus:** push past PR #592 SOTA (6.5985%); close tau_y/tau_z gap (currently 2.3–2.7× above AB-UPT floor); expand ensemble pool with each new winner; orthogonal sweeps (β2, EMA, lr-floor) for stacking; flow-aligned tau output basis (#641 thorfinn) as physics-motivated geometric lever; NorMuon (#640 edward, #619 alphonse) and slice-centroid RoPE (#621 nezuko) as bigger optimizer/architectural swings.

---

## Active research themes

### 1. tau_y/tau_z gap closure (primary open problem)
- **Gap status (PR #592 L=5 SOTA):** wall_shear_y=8.3631%, wall_shear_z=9.8099% vs AB-UPT 3.65%/3.63% — 2.3–2.7× above floor; still the two largest residual error contributors after the depth bump
- **Active attacks:**
  - Flow-aligned tangent-frame tau outputs (#613 thorfinn) — predict tau in (t̂, n̂, ŝ) basis instead of (x, y, z); decouples geometric transformation from physics
  - Surface curvature features (#615 edward) — physics-informed mean H + Gaussian K geometric prior (reissue of #605)
  - Lion β2 momentum sweep (#614 fern) — longer second-moment averaging window for noisier tau gradient signs
  - lr-min cosine floor (#593 frieren) — non-zero late-epoch lr to preserve signal
  - EMA decay (#603 tanjiro) — wider EMA windows for tau variance reduction

### 2. Ensemble expansion (composable, zero-training-cost)
- Pool 25 next: any single-model winner from current round (≥ #613/#615/#619/#621) auto-feeds the greedy selector. Volume-pressure test-vs-val gap (~3×) is the primary systematic issue to investigate — possible target for variance-reduction or test-time refinement member.

### 3. Composition opportunities (next round, when winners emerge)
- Tangent-frame tau (#613) + curvature features (#615) — geometric × geometric; orthogonal axes of geometric prior
- β2 winner (#614) + L=5 (#592 SOTA) — optimizer × architectural depth
- NorMuon (#619) winner + L=5 — orthogonal optimizer-family change
- Slice-centroid RoPE (#621) winner + L=5 — attention-side architectural lever
- Any single-model winner → immediately add to ensemble pool for next nezuko greedy run

---

## Potential next research directions

1. **Knowledge distillation from K=7 ensemble into single model** — ensemble-as-teacher; small student that approximates the ensemble at single-model inference cost
2. **Sliced Wasserstein distance** as auxiliary distribution-matching loss on tau channels (vs L2 only)
3. **Loss-aware sampling** — over-sample query points in regions of historically high tau_y/tau_z error during training (boosting-style); requires per-point error tracking
4. **Tau uncertainty decomposition** — separate aleatoric (noise-floor) from epistemic uncertainty using MC-dropout; identifies irreducible vs reducible error
5. **Two-stage training: warmup-only on volume, then unfreeze surface heads** — attacks joint-loss imbalance from a different angle than GradNorm
6. **Volumetric query-point density beyond 65536** — push to 98304 or 131072 with VRAM headroom probe
7. **Greedy ensemble with TTA members** — if any TTA wins, double pool size at zero training cost via mirror-y (sign-flipped tau_y) per-run pairs
8. **β-NLL heteroscedastic loss on tay stack** — yi-branch experiment (#583) showed canonical β-NLL with var.detach().pow(β=0.5) reaches val=8.86% on yi (beats 9.03% gate); test on tay SOTA stack as principled noise-floor estimator
9. **Curriculum on RFF sigmas** — start narrow-band RFF, expand spectral coverage during training; complements rff32 win
10. **Spectral lion / NorMuon revisits** on the new SOTA stack — prior negatives were on older stacks

---

## Negative results catalog (do not retry on current stack)

| Lever | Outcome |
|---|---|
| Local tangent-frame INPUT features | NEGATIVE (#423) — but tangent-frame OUTPUT (#613) is a different lever |
| Channel-selective Huber on tau | NEGATIVE (#353) |
| Volume-loss-weight scalar rebalancing | NEGATIVE (#451) |
| Separate volume decoder | NEGATIVE val→test overfit (#452) |
| Muon optimizer | NEGATIVE (#299) +4.09pp |
| Sandwich-norm | NEGATIVE diverged |
| U-net skips | NEGATIVE (+0.555pp) |
| 256d / 768d hidden | NEGATIVE |
| Per-axis output head scaling (#467) | NEGATIVE — gap is upstream |
| TTA mirror-y inference (#499 old stack) | NEGATIVE +1.18pp |
| Y-mirror training aug (#536) | NEGATIVE — gap is structural |
| 2× surface point density (#506) | NEGATIVE — slower/epoch beats density |
| mlp_ratio=6/8 wider FFN (#458) | NEGATIVE — mlp4 is optimal |
| Signed-log target transform (#471 arm-b) | NEGATIVE |
| log1p target transform (#481) | NEGATIVE |
| AdamW vs Lion (#532) | NEGATIVE — Lion optimal, confirmed |
| Full GradNorm (5× autograd overhead) | NEGATIVE operationally — crashes in budget |
| Unit-vector cosine direction loss on tau (#531) | NEGATIVE — direction is not the bottleneck |
| Coord jitter regularization (#553) | NEGATIVE +38% — RFF/STRING-sep needs precise coordinates |
| slw=2.0 13-epoch full (#537) | NEGATIVE — within noise |
| Static channel reweighting (PRE-Lion+rff stack: #142, #454, #467, #531) | Was 4× NEG on simpler stacks; **PR #571 Arm A (tau_y×1.5 / tau_z×2.0 on Lion+rff multisigma stack) SUCCEEDED — now part of SOTA** |

---

## Procedural notes

- **Never commit research state to a student branch** — only to `tay`. (Confirmed sticky from Issue #606.)
- **Multi-arm WIP pattern**: a student may have an "older" WIP PR still completing background arms while a newer PR is queued. Use `student_poll_for_work` (returns the most recent WIP) to identify the live assignment, not `list_all_prs`.
- **PR #583 (yi branch)**: out of `tay` advisor scope — handled by the `senpai-advisor-yi` pod.
