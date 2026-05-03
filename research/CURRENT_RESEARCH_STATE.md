# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 — 7 active WIP PRs, 0 ready for review, 0 idle students
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #511 (edward extended-cosine EP13), val_abupt **7.0134%** (W&B `5o7jc7wi`)

All future PRs must beat val_abupt < **7.0134%**. Test: test_abupt=8.3130%.

| Metric | PR #511 SOTA (val EP13) | AB-UPT |
|---|---:|---:|
| `abupt` | **7.0134%** | — |
| `surface_pressure` | 4.5104% | 3.82% |
| `wall_shear` | 7.9650% | 7.29% |
| `volume_pressure` | **4.2168%** | 6.08% (BEATEN) |
| `tau_x` | 7.0053% | 5.35% |
| `tau_y` | **8.7717%** | 3.65% |
| `tau_z` | **10.5629%** | 3.63% |

### Canonical SOTA stack reproduce command
```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --agent <student> --optimizer lion --lr 1e-4 --weight-decay 5e-4 \
  --no-compile-model --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --lr-cosine-t-max 13 --epochs 13
```
Note: `--train-volume-points 65536` constant (NOT vol-curriculum); `--rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"` is the multi-sigma STRING-sep octave init.

---

## Latest research direction from human researcher team

No new directives as of 2026-05-01. Continuing organic tau_y/tau_z attack programme.

---

## Currently in-flight (7 active WIP PRs on tay, ZERO idle students)

| PR | Student | Hypothesis | Latest val_abupt | Status |
|---|---|---|---:|---|
| #535 | edward | Extended cosine to **EP15** on PR #511 stack | EP1=50.91% | run `nh2ke150` healthy; extended schedule to EP15 |
| #534 | tanjiro | Multi-scale STRING-sep bands σ∈{0.25,1.0,4.0}, 8 feats/band | EP1=28.20% | run `loxzj4xq` EP1 best-ever; awaiting EP3+ |
| #531 | fern | Unit-vector cosine direction loss on tau (Arm B w=0.1) | EP3=8.02% | run `3lurbotq` healthy, on-par trajectory |
| #523 | thorfinn | GradNorm-EMA proxy Run 2 (α=0.5, min_weight=0.7) | (pending) | **Sent back** with Run 2 config; awaiting relaunch |
| #536 | nezuko | Y-mirror training augmentation (p=0.5) for tau_y/tau_z | (pending) | **NEWLY ASSIGNED** — PR #536 |
| #516 | askeladd | Per-channel tau_y/tau_z reweight (Run A: w_y=2.0, w_z=2.5) | (just launched) | run `4uw2c4z2` relaunched clean vs PR #511 |
| #510 | alphonse | Surface-loss-weight=2.0 sweep vs PR #511 | EP3=7.65% | run `qqtdnlwq` — STRONG, leading every channel |
| #501 | frieren | Per-axis multi-sigma STRING priors (frieren-aniso-string-vs511) | EP1 running | run `qawfhlu6` healthy, newly relaunched after rebase |

---

## Recent review results (this cycle)

### PR #532 nezuko AdamW vs Lion — CLOSED NEGATIVE
- Run `3hm5ae1j`, EP4 val_abupt=7.94% vs Lion 7.42% (+0.51pp gap)
- **Lion wins convincingly** at every epoch and every axis including tau_y/tau_z
- AdamW closed the gap from +1.87pp (EP2) to +0.51pp (EP4) — faster per-epoch improvement but starting disadvantage too large
- **Insight:** Adaptive LR helps tau_y/tau_z per-epoch but not net. Scheduled optimizer switch (Lion→AdamW at ~EP3) is the correct experiment if we want adaptive-LR benefits.
- **AdamW vs Lion at this batch size is now closed.** Do not retry.

### PR #523 thorfinn GradNorm-EMA proxy Run 1 — SENT BACK (promising, primary metric miss)
- Run `9477cjoh`, 6 epochs, val_abupt=7.2667% (miss vs SOTA 7.0134%, +0.25pp)
- **First ever tau_z win on val AND test:** val_tau_z=10.481% (−0.08pp below SOTA 10.5629%), test_tau_z=9.704% (−0.22pp below SOTA 9.927%)
- tau_y=8.943% (still +0.17pp above SOTA 8.7717%) — close but didn't close
- Mechanism worked perfectly: balancer r-ordering r_z>r_y>r_x>r_vp≈r_sp; w_tau_z climbed 1.04→1.58; w_sp dropped 0.96→0.50
- **Problem:** α=1.5 too aggressive — sp/vp down-weighted to 0.50, causing +0.79pp vp regression that outweighed the tau gains
- **Run 2 config:** α=0.5 (softer), min_weight=0.7 (floor on sp/vp), epochs=13 with cosine T_max=13 to match SOTA schedule

---

## Active research themes

### 1. tau_y/tau_z gap closure (primary open problem)
- **Gap:** tau_y=8.77%, tau_z=10.56% vs AB-UPT 3.65%/3.63% (2.4–2.9× above target)
- **Active attacks:**
  - GradNorm-EMA Run 2 (thorfinn #523) — proven mechanism, just needs gentler alpha
  - Y-mirror training augmentation (nezuko #536) — fresh orthogonal approach, doubles tau_y sign diversity
  - Per-channel reweighting (askeladd #516) — static weighting variation vs PR #511 stack
  - Surface-loss-weight=2.0 (alphonse #510) — indirect attack via surface up-weighting
  - Unit-vec direction loss (fern #531) — geometric loss formulation
  - Per-axis multi-sigma STRING priors (frieren #501) — spectral frequency priors per-axis
  - Multi-scale STRING bands (tanjiro #534) — spectral diversity across octaves

### 2. Extended schedule / convergence
- edward #535 (EP15 cosine) — descending tail at EP13 suggests EP14-15 headroom
- All students now on EP13 cosine T_max=13 to match SOTA #511

### 3. Composition opportunities (next round, when winners emerge)
- GradNorm-EMA α=0.5 + surface-loss-weight=2.0 — orthogonal mechanisms
- GradNorm-EMA + extended cosine EP15 (if edward #535 advances SOTA)
- Y-mirror aug + winning GradNorm config — both are additive

---

## Potential next research directions

1. **Compose alpha-optimized GradNorm-EMA with slw=2.0** (when alphonse #510 lands and thorfinn Run 2 confirms)
2. **Compose y-mirror aug with EP15 cosine** (if #535 and #536 both win)
3. **Scheduled optimizer switch** (Lion→AdamW at EP3) to capture AdamW's per-epoch tau convergence advantage without the starting disadvantage
4. **Physics-informed no-slip boundary loss** — zero-velocity wall condition as auxiliary loss on tau magnitude at wall proximity
5. **EMA model-soup** — average parameters of top-K best-val checkpoints (variance reduction, zero GPU cost)
6. **Frequency-aware tau loss** — weight loss contributions by spatial frequency (penalize high-freq tau residuals more heavily); targets tau_y/tau_z where the high-freq variation is the AB-UPT gap
7. **Learnable Fourier basis (NTK-style)** — replace fixed Gaussian RFF with learned small matrix B, jointly trained with the main objective

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
| TTA mirror-y inference (#499) | NEGATIVE — TTA hurts +1.18pp (training-aug is different) |
| 2× surface point density (#506) | NEGATIVE — slower/epoch beats density |
| tau_yz scalar loss-weight reweight (#142, #454, #467) | EXHAUSTED — three NEG, problem upstream |
| mlp_ratio=6/8 wider FFN (#458) | NEGATIVE — mlp4 is optimal |
| Signed-log target transform (#471 arm-b) | NEGATIVE |
| log1p target transform (#481) | NEGATIVE |
| AdamW vs Lion (#532) | NEGATIVE — Lion optimal, confirmed |
| Full GradNorm (5× autograd overhead) | NEGATIVE operationally — crashes in budget |
