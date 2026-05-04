# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-04 ~14:20 UTC (Round 13 — fleet 8/8 fully utilized)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Fleet:** 0 idle students, 8 WIP PRs (all GPUs in use)
- **Tay-deployed students:** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn

---

## ENSEMBLE SOTA — PR #602 nezuko greedy K=7 (pool 23) — val_abupt **6.2062%** / test_abupt **7.5164%**

W&B run `ydw7rxl2`. Beats prior K=7 (PR #562 val=6.2345%) by adding PR #571 SOTA (`nh96x7m4`, rff16 multi-sigma) to the candidate pool. Greedy forward selection from Caruana 2004 across 23 candidates.

**Ensemble gate:** any new ensemble must beat val_abupt < **6.2062%**.

## SINGLE-MODEL SOTA — PR #594 askeladd RFF=32 on PR #571 SOTA stack — val_abupt **6.7258%**

W&B run `d777epep`. Beats prior PR #571 (val=6.7644%) by spectral-budget upgrade from 16 to 32 RFF features on the rff_multisigma + lion + tau-reweight stack.

**Single-model gate:** any new single-model run must beat val_abupt < **6.7258%**.

### Noise floor calibration
- Identical-config rerun shows ~+0.05pp run-to-run variation
- Genuine win threshold: **val_abupt < 6.6758%** (gate − 5bps)
- Borderline (need replicate): 6.6758–6.7258%
- No signal / regression: > 6.7258%

### Canonical SOTA reproduce command (PR #594 stack)

```bash
torchrun --standalone --nproc_per_node=8 target/train.py \
  --agent <student> --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 \
  --surface-loss-weight 2.0 --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 32 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --no-compile-model --lr-cosine-t-max 13 --epochs 13
```

---

## Latest research direction from human researcher team

No new directives as of 2026-05-04 14:20 UTC (no open `ADVISOR` or `ADVISOR-TAY` issues).
Issue #606 directive (assign edward `surface-curvature-features` after his prior PR completes) — **fulfilled** via PR #615 (edward/surface-curvature-v2).

---

## Currently in-flight (8 active WIP PRs on tay, ZERO idle)

Each student listed against the most recently-assigned PR; older PRs from the same student are multi-arm sweeps still completing additional arms in the background.

| PR | Student | Lever | Notes |
|---|---|---|---|
| #592 | alphonse | Model depth sweep (L=5 vs L=6 on SOTA stack) | Arm A (L=5, run `4k25s25e`) reached EP3 — awaiting full results |
| #593 | frieren | lr-min cosine floor sweep (1e-6 vs 1e-5) | Arm A v2 (lr-min=1e-6, run `axpkjoku`) reached EP8 timeout — awaiting full per-epoch comparison vs SOTA |
| #594 | askeladd | RFF spectral budget sweep (8 vs 32 features) — already merged Arm with rff32 as SOTA; Arm B (rff8) for ablation | Arm B (rff8) hitting NCCL/wandb-DDP init race; relaunched 14:15 UTC |
| #603 | tanjiro | EMA decay sweep {0.9993, 0.9997, 0.9999} | Arm A (0.9993) aborted at EP2 (val=8.95%, exceeds 8.5% kill gate); Arm B (0.9997) relaunched after wandb init crash |
| #612 | nezuko | Greedy ensemble pool expansion 23→24 (add PR #594 rff32 SOTA `d777epep`); re-run K≤15 greedy | Inference-only task — no training |
| #613 | thorfinn | Flow-aligned tau tangent-frame outputs (predict in (t̂, n̂, ŝ) basis) | New approach to tau_y/tau_z gap closure |
| #614 | fern | Lion β2 momentum sweep (0.99 vs 0.999) | Probing under-smoothing of sign estimates for tau channels |
| #615 | edward | Surface curvature features v2 (mean H + Gaussian K via k-NN PCA k=20) | Reissue of #605 which logged no results; on PR #594 SOTA stack |

**Round 13 focus:** push past PR #594 SOTA (6.7258%); continue closing tau_y/tau_z gap; expand ensemble pool with new SOTA candidate; orthogonal sweeps (RFF, EMA, depth, β2) for stacking gains; flow-aligned tau output basis as fundamental reformulation.

---

## Active research themes

### 1. tau_y/tau_z gap closure (primary open problem)
- **Gap status (PR #594 SOTA):** tau_y=8.489%, tau_z=9.997% vs AB-UPT 3.65%/3.63% — 2.3–2.7× above floor; the two largest residual error contributors
- **Active attacks:**
  - Flow-aligned tangent-frame tau outputs (#613 thorfinn) — predict tau in (t̂, n̂, ŝ) basis instead of (x, y, z); decouples geometric transformation from physics
  - Surface curvature features (#615 edward) — physics-informed mean H + Gaussian K geometric prior (reissue of #605)
  - Lion β2 momentum sweep (#614 fern) — longer second-moment averaging window for noisier tau gradient signs
  - Model depth sweep (#592 alphonse) — re-test 5L/6L on SOTA stack
  - RFF budget Arm B (#594 askeladd) — rff8 ablation to confirm rff32 win was real
  - lr-min cosine floor (#593 frieren) — non-zero late-epoch lr to preserve signal
  - EMA decay (#603 tanjiro) — wider EMA windows for tau variance reduction

### 2. Ensemble expansion (composable, zero-training-cost)
- PR #612 nezuko greedy K≤15 over pool 24 (adds PR #594 rff32 SOTA) — should tighten ensemble val below 6.2062%

### 3. Composition opportunities (next round, when winners emerge)
- Tangent-frame tau (#613) + curvature features (#615) — geometric × geometric; orthogonal axes of geometric prior
- β2 winner (#614) + RFF winner (#594) — optimizer × spectral budget
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
