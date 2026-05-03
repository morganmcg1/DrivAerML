# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-04 (post-context-summary continuation, Round 12 opened)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Fleet:** 0 idle students, 8 WIP PRs, 0 review-ready

## Current SOTA — PR #523 thorfinn EMA-proxy GradNorm alpha=0.5, EP6, val_abupt **6.9246%**

All future PRs must beat val_abupt < **6.9246%** with test_abupt ≤ 8.30%.

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

| PR | Student | Lever | Status |
|---|---|---|---|
| #516 | askeladd | per-channel tau_y/tau_z reweight v2 (1.2x/1.3x) | running |
| #534 | tanjiro | fused 24-feat StringSep multi-band (σ={0.25,1.0,4.0}×8) | running |
| #538 | fern | SGDR cosine warm restarts T0=6 Tmult=2 | running |
| #549 | edward | rff-num-features 32 (double feature budget) | NEEDS REBASE |
| #550 | frieren | lr-min cosine floor — prevent decay-to-zero | NEEDS REBASE |
| #552 | thorfinn | GradNorm-EMA min_weight floor sweep {0.5, 0.6} (NEW) | just assigned |
| #553 | alphonse | input coord jitter regularization sigma sweep {0.002, 0.005, 0.010} (NEW) | just assigned |
| #554 | nezuko | top-K best-val checkpoint averaging "model soup" (NEW) | just assigned |

Round-12 opening focus: defend SOTA, attack tau_y/tau_z gap from new orthogonal angles (regularization, weight averaging, dynamic rebalancing).

---

## This-cycle review actions (just completed)

### PR #537 alphonse slw=2.0 13-epoch — CLOSED NEGATIVE
- val_abupt=6.8994% at EP5 (270min cap), beats old #510 baseline by 0.107pp on val
- Within seed-variance noise (~0.4pp); does NOT beat current SOTA #523 (6.9246%)
- test_abupt=8.2972% essentially flat vs #510's 8.2921%

### PR #536 nezuko y-mirror training augmentation — CLOSED NEGATIVE (hypothesis falsified)
- val_abupt=7.2315% at EP11 (270min timeout), well above SOTA 6.9246%
- Mirror_aug_frac=0.498 (correct firing). Tau_y did NOT improve selectively
- Gap closed uniformly across all channels → tau_y/tau_z gap is structural, not data-symmetry addressable
- Important update to negative results catalog: y-mirror training aug joins TTA mirror-y inference (#499) as both negative

---

## Active research themes

### 1. tau_y/tau_z gap closure (primary open problem)
- **Gap status:** tau_y=8.72%, tau_z=10.30% vs AB-UPT 3.65%/3.63% (2.4–2.8× above)
- **Active attacks (this round):**
  - GradNorm-EMA tighter floor (#552 thorfinn) — proven mechanism, more aggressive redistribution
  - Coord jitter regularization (#553 alphonse) — Tikhonov regularizer, attack overfitting on hard channels
  - Checkpoint averaging / model soup (#554 nezuko) — variance reduction, target slow-converging channels
  - Per-channel reweight v2 (#516 askeladd) — micro-bump 1.2x/1.3x
  - Multi-band fused StringSep (#534 tanjiro) — spectral diversity per-band
  - rff-num-features 32 (#549 edward) — double feature budget
  - lr-min cosine floor (#550 frieren) — prevent decay-to-zero
  - SGDR warm restarts (#538 fern) — multi-cycle convergence

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
