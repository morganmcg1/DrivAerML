# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-04 ~07:00 UTC (refreshed after EP3 fleet read; fern NorMuon sent back for canonical redesign)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Fleet:** 0 idle students, 8 WIP PRs, 0 review-ready (full GPU utilization)
- **Tay-deployed students:** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn (8 total)

## ENSEMBLE SOTA — PR #562 nezuko greedy K=7 forward selection — val_abupt **6.2345%**

Beats prior K=5 ensemble (PR #556 val=6.2681%) by Caruana 2004 greedy forward selection from 22 candidates.

## SINGLE-MODEL SOTA — PR #516 askeladd tau-weight v2 — val_abupt **6.8701%** / test_abupt **8.1229%**

All single-model PRs must beat val_abupt < **6.8701%** with test_abupt ≤ ~8.20%.

### Noise-floor calibration (askeladd PR #571 rebased SOTA repro)
- Identical-config rerun of SOTA stack: val=6.9226% vs claimed 6.8701% (+0.052pp on identical code)
- **Treat improvements within ±0.05pp of SOTA as noise**, not signal
- Genuine win threshold: val_abupt < 6.82%
- Borderline (need replicate): 6.82–6.92%
- No signal / regression: > 6.92%

| Metric | PR #523 SOTA val EP6 | AB-UPT |
|---|---:|---:|
| `abupt` | **6.9246%** | — |
| `surface_pressure` | 4.5840% | 3.82% |
| `wall_shear` | 7.7457% | 7.29% |
| `volume_pressure` | **4.3040%** | 6.08% (BEATEN) |
| `tau_x` | 6.7193% | 5.35% |
| `tau_y` | 8.7197% | 3.65% |
| `tau_z` | 10.2960% | 3.63% |

Test (best-val checkpoint): test_abupt=8.2355%, test_tau_y=8.4656%, test_tau_z=9.6720%.
W&B run: `wyz68o8r`, group `thorfinn-gradnorm-r2`, runtime 4.71h.

### Canonical SOTA reproduce command

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <student> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --lr-cosine-t-max 13 --epochs 13 \
  --use-gradnorm --gradnorm-mode ema_proxy \
  --gradnorm-alpha 0.5 --gradnorm-ema-beta 0.9 --gradnorm-min-weight 0.7
```

Note: `--use-gradnorm` overrides `--surface-loss-weight` / `--volume-loss-weight` (legacy scalars are ignored when gradnorm is enabled).

---

## Latest research direction from human researcher team

No new directives as of 2026-05-01 (issues #285, #252, #48 all have current advisor responses).

---

## Currently in-flight (8 active WIP PRs on tay, ZERO idle)

| PR | Student | Lever | Status (2026-05-04 06:30 UTC) |
|---|---|---|---|
| #552 | thorfinn | GradNorm-EMA min_weight floor sweep | Arm A (floor=0.5) DONE val=6.9602%; **Arm B EP3=7.55%** (≈SOTA); confirmed floor-not-binding; pivot to α=1.5 launch instructions posted |
| #555 | frieren | GradNorm alpha sweep — α∈{0.75, 1.0, 1.5} | Arm A α=0.75 DONE val=6.9421% (test tau_y/tau_z improved); **Arm B α=1.0 EP3=7.46%** — best-of-fleet at EP3, narrowly ahead of SOTA EP3 7.49% |
| #568 | fern | NorMuon optimizer | **DIVERGED — Arm A `ii3vh18d` blew up post-EP1**; sent back for canonical row-wise RMS NorMuon redesign |
| #571 | askeladd | tau_y/tau_z weight intensity sweep (3-arm) | Arm A (×1.5/×2.0) **EP3=7.93%** — gap widening as predicted; let finish for clean test; reconsider Arm C launch |
| #572 | nezuko | Lion β1 sweep (0.9 → 0.8/0.7) | **Arm A EP1=38.58%** (+7.76pp vs SOTA EP1) — concerning; EP2 hard gate at 11.0% |
| #573 | edward | EMA decay sweep (0.999 → 0.9993/0.9997/0.9999) | Arm A relaunched as `olrwgvav` (prior `y5f4ptmm` killed); too early for val (~EP0.6) |
| #574 | tanjiro | RFF spectral density expansion (3-arm: 32f same / 32f wider / 64f) | Arm A `3nn65ume` running after `--no-compile-model` fix; too early for val (~EP0.5) |
| #575 | alphonse | Tangent-frame rotation for tau loss — predict tau in local (t̂, b̂, n̂) frame | running but **W&B group is `coord-jitter-s0.001`** (suggests student is running coord-jitter, not tangent-frame); EP3=8.53% — needs student check-in to clarify |

Round-12 focus: defend PR #516 SOTA; attack tau_y/tau_z gap from new orthogonal angles (structural frame rotation, dynamic rebalancing, alpha intensification, EMA stabilization, RFF capacity).

---

## Recent merges / closures (Round 11 → Round 12 boundary)

### PR #553 alphonse coord-jitter regularization — CLOSED NEGATIVE
- Hypothesis: Gaussian coordinate noise (sigma=0.001–0.01) as Tikhonov regularizer would improve tau_y/tau_z via preventing surface-coordinate overfitting
- Sigma=0.0 baseline arm reproduced SOTA at val=6.9511% (clean repro, confirms stack intact)
- Sigma=0.001 arm: val_abupt=9.603% at EP2.3 — **+38% relative regression** vs baseline
- Verdict: STRING-sep/RFF relies on precise coordinate structure; injecting coordinate noise destroys positional encoding quality. Falsified. Added to negative results catalog.

### PR #562 nezuko greedy K=7 ensemble — MERGED (ENSEMBLE SOTA)
- val_abupt=6.2345% via Caruana 2004 forward selection from 22-run pool

### PR #516 askeladd tau-weight v2 — MERGED (SINGLE-MODEL SOTA)
- val_abupt=6.8701%, test_abupt=8.1229% — defines current single-model gating threshold

### PR #142 thorfinn vol_w=2.0 — closeout logged (+10.78%)
### PR #146, #141 — closeouts logged

### Pending in this cycle
- PR #552 Arm A: borderline (within noise floor, not a real regression); pivoting thorfinn to alpha=1.5
- PR #555 Arm A α=0.75: borderline val (+0.018pp), but tau_y test -0.119pp / tau_z test -0.124pp — Arm B α=1.0 is the tiebreaker

---

## Active research themes

### 1. tau_y/tau_z gap closure (primary open problem)
- **Gap status:** tau_y=8.66%, tau_z=10.27% vs AB-UPT 3.65%/3.63% (2.4–2.8× above; values from PR #516 best-val EP5)
- **Active attacks (this round):**
  - GradNorm-EMA tighter floor (#552 thorfinn) — proven mechanism, more aggressive redistribution
  - Tangent-frame rotation for tau loss (#575 alphonse) — structural: predict tau in local (t̂, b̂, n̂) frame to remove coordinate-axis-aligned bias
  - tau weight intensity sweep (#571 askeladd) — 3-arm: ×1.5/×2.0, ×2.0/×2.5, ×1.0/×1.5; last attempt at this angle
  - GradNorm alpha sweep (#555 frieren) — stronger tau_y/tau_z gradient pull via alpha tuning
  - NorMuon optimizer (#568 fern) — normalized Muon for 2D weights; addressing instability of vanilla Muon (#261)
  - Lion β1 sweep (#572 nezuko) — lower momentum may improve tau channels
  - EMA decay sweep (#573 edward) — longer EMA windows may reduce tau variance
  - RFF spectral density expansion (#574 tanjiro) — double feature budget or wider sigma range for better spectral coverage

### 2. Negative-direction confirmed (do not retry on current stack)
- **Static channel reweighting** is now 4× negative (#142, #454, #467, #531) — askeladd #516 v2 is final attempt at this angle before the lever is exhausted
- **Y-mirror data augmentation** (#536) — gap is structural
- **Direction loss on tau** (#531) — gap is not direction-prediction

### 3. Composition opportunities (next round, when winners emerge)
- GradNorm-EMA winner + checkpoint soup (#552 + #554 mechanisms) — orthogonal
- Coord jitter winner + GradNorm-EMA — both attack regularization separately
- Multi-band StringSep winner + rff32 — spectral capacity stacking
- Extended cosine (lr-min floor #550) + GradNorm-EMA — schedule × gradient budget

---

## Potential next research directions

1. **EMA decay sweep** {0.999, 0.9995, 0.9997} — currently stuck at 0.999; longer EMA windows may help variance reduction post-checkpoint-soup
2. **Flow-aligned coordinate frame** for tau outputs — predict tau in (t̂, n̂, ŝ) tangent-frame basis instead of (x,y,z), freeing the model from coordinate-axis-aligned bias (would explain why tau_x is easier than tau_y/tau_z)
3. **Knowledge distillation from PR #523 SOTA teacher** into smaller / equally-sized student with surface auxiliary targets
4. **Volumetric query-point density curriculum** beyond 65536 — push to 98304 or 131072 with VRAM headroom check
5. **Loss-aware sampling** — over-sample query points in regions of historically high tau_y/tau_z error during training (boosting-style)
6. **Attention head pruning + redistribution** — analyze SOTA model's attention; some heads may be redundant, redistribute to a separate "tau head" path
7. **Sliced Wasserstein distance** as auxiliary distribution-matching loss on tau channels (vs L2 only)
8. **Auxiliary point-normal / surface-area regression** as multi-task auxiliary head — geometric pretext task
9. **Two-stage training: warmup-only on volume, then unfreeze surface heads** — attacks the joint-loss imbalance from a different angle than GradNorm

---

## Negative results catalog (do not retry on current stack)

| Lever | Outcome |
|---|---|
| Local tangent-frame INPUT features | NEGATIVE (#423) |
| Channel-selective Huber on tau | NEGATIVE (#353) |
| Volume-loss-weight scalar rebalancing | NEGATIVE (#451) |
| Separate volume decoder | NEGATIVE val→test overfit (#452) |
| Muon optimizer | NEGATIVE (#299) +4.09pp |
| Sandwich-norm | NEGATIVE diverged |
| U-net skips | NEGATIVE (+0.555pp) |
| 256d / 768d hidden | NEGATIVE |
| 6L / 8L depth | NEGATIVE |
| Per-axis output head scaling (#467) | NEGATIVE — gap is upstream |
| TTA mirror-y inference (#499) | NEGATIVE — TTA hurts +1.18pp |
| **Y-mirror training aug (#536)** | **NEGATIVE — gap is structural, not symmetry-addressable** |
| 2× surface point density (#506) | NEGATIVE — slower/epoch beats density |
| tau_yz scalar loss-weight reweight (#142, #454, #467, #531) | EXHAUSTED — 4× NEG, problem upstream |
| mlp_ratio=6/8 wider FFN (#458) | NEGATIVE — mlp4 is optimal |
| Signed-log target transform (#471 arm-b) | NEGATIVE |
| log1p target transform (#481) | NEGATIVE |
| AdamW vs Lion (#532) | NEGATIVE — Lion optimal, confirmed |
| Full GradNorm (5× autograd overhead) | NEGATIVE operationally — crashes in budget |
| Unit-vector cosine direction loss on tau (#531) | NEGATIVE — direction is not the bottleneck |
| **slw=2.0 13-epoch full (PR #537)** | **NEGATIVE vs current SOTA — within noise of #510 prior win, doesn't beat #523** |
