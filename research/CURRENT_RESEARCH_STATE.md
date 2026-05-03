# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 18:30 UTC — 8 active WIP PRs, 0 ready for review, 0 idle students
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #510 (alphonse slw=2.0, EP5 EMA), val_abupt **7.0063%** (W&B `qqtdnlwq`)

All future PRs must beat val_abupt < **7.0063%**. Test: test_abupt=8.2921%.

| Metric | PR #510 SOTA (val EP5 EMA) | AB-UPT |
|---|---:|---:|
| `abupt` | **7.0063%** | — |
| `surface_pressure` | 4.5994% | 3.82% |
| `wall_shear` | 7.8939% | 7.29% |
| `volume_pressure` | **4.1643%** | 6.08% (BEATEN) |
| `tau_x` | 6.8150% | 5.35% |
| `tau_y` | 8.9516% | 3.65% |
| `tau_z` | 10.5010% | 3.63% |

Note: tau_y/surface_pressure (val) are slightly worse than PR #511 (8.7717%/4.5104% vs 8.9516%/4.5994%), but test_abupt and 5/7 test channels improve.

### Canonical SOTA stack reproduce command
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
  --surface-loss-weight 2.0
```
Note: `--vol-points-schedule "0:16384:3:32768:6:49152:9:65536"` is the vol-curriculum; `--rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"` is the multi-sigma STRING-sep octave init; `--surface-loss-weight 2.0` is the new slw=2.0 lever.

Warning: run `qqtdnlwq` timed out at EP5 with a 50-epoch config (not --epochs 13). A proper 13-epoch run with slw=2.0 should be the canonical reproduce. The BASELINE.md reproduce command uses `--epochs 13 --lr-cosine-t-max 13` for comparison.

---

## Latest research direction from human researcher team

No new directives as of 2026-05-01. Continuing organic tau_y/tau_z attack programme.

---

## Currently in-flight (8 active WIP PRs on tay, ZERO idle students)

| PR | Student | Hypothesis | Latest val_abupt | Status |
|---|---|---|---:|---|
| #538 | fern | Cosine annealing with warm restarts (SGDR T0=6, Tmult=2) | — | **NEW** — assigned 2026-05-03; awaiting launch |
| #537 | alphonse | slw=2.0 full 13-epoch run (complete EP5 timeout SOTA) | — | **NEW** — assigned 2026-05-03; awaiting launch |
| #535 | edward | Extended cosine to **EP15** on SOTA stack | EP2=9.03% | run `nh2ke150` running; EP2 healthy |
| #534 | tanjiro | Multi-scale STRING-sep bands σ∈{0.25,1.0,4.0}, 8 feats/band | EP3=7.606% | run `loxzj4xq` running; strongest trajectory in flight |
| #523 | thorfinn | GradNorm-EMA proxy Run 2 (α=0.5, min_weight=0.7) | Run 1=7.267% | **Sent back** with Run 2 config; awaiting relaunch |
| #536 | nezuko | Y-mirror training augmentation (p=0.5) for tau_y/tau_z | crashed | **BUG: mirror_aug_frac=0** — augmentation never fired; needs debug fix |
| #516 | askeladd | Per-channel tau_y/tau_z reweight (Run A: w_y=2.0, w_z=2.5) | EP3=15.017% | run `4uw2c4z2`; EP3 FAILS gate (≤14.15%), trajectory concerning |
| #501 | frieren | Per-axis multi-sigma STRING priors (frieren-aniso-string-vs511) | EP1 partial | run `qawfhlu6` running; EP1 in progress |

---

## Recent review results (this cycle)

### PR #531 fern unit-vector direction loss (tau Arm B w=0.1) — CLOSED NEGATIVE
- Run `3lurbotq`, best EP6, 4.71h
- val_abupt=7.2105% (MISS vs SOTA 7.0063%, +0.204pp)
- tau_y=9.2710% (+0.319pp WORSE than SOTA), tau_z=10.8471% (+0.346pp WORSE)
- test_abupt=8.5876% (+0.296pp WORSE). Every channel except volume_pressure regresses.
- Conclusion: Geometric direction loss does not help — the tau_y/tau_z problem is not a direction-alignment problem.

### PR #510 alphonse slw=2.0 — MERGED NEW SOTA
- Run `qqtdnlwq`, EP5 EMA, val_abupt=7.0063%, test_abupt=8.2921%
- Beats PR #511 by −0.007pp val, −0.021pp test. 5/7 test channels improve.
- tau_y regresses slightly on val (+0.18pp) but improves on test (−0.063pp). vol_p regresses on test (+0.238pp).
- Run timed out at EP5 (50-epoch config hit 360-min budget). Full 13-epoch run expected to improve further.

### PR #532 nezuko AdamW vs Lion — CLOSED NEGATIVE
- Lion wins convincingly at every epoch and every axis including tau_y/tau_z.

### PR #523 thorfinn GradNorm-EMA proxy Run 1 — SENT BACK (promising)
- Run `9477cjoh`, 6 epochs, val_abupt=7.2667% (miss vs SOTA 7.0134%, +0.25pp — now also behind new SOTA 7.0063%)
- **First ever tau_z win on val AND test.** Run 2 with α=0.5 + min_weight=0.7 + EP13 config pending.

---

## Active research themes

### 1. tau_y/tau_z gap closure (primary open problem)
- **Gap:** tau_y=8.77%, tau_z=10.56% vs AB-UPT 3.65%/3.63% (2.4–2.9× above target)
- **Active attacks:**
  - GradNorm-EMA Run 2 (thorfinn #523) — proven mechanism, just needs gentler alpha
  - Y-mirror training augmentation (nezuko #536) — fresh orthogonal approach, doubles tau_y sign diversity
  - Per-channel reweighting (askeladd #516) — static weighting variation vs PR #511 stack
  - Surface-loss-weight=2.0 (alphonse #510) — indirect attack via surface up-weighting
  - Per-axis multi-sigma STRING priors (frieren #501) — spectral frequency priors per-axis
  - Multi-scale STRING bands (tanjiro #534) — spectral diversity across octaves

### 2. Extended schedule / convergence
- edward #535 (EP15 cosine) — descending tail at EP13 suggests EP14-15 headroom
- fern #538 (SGDR warm restarts T0=6, Tmult=2) — two complete cosine cycles, restart at EP7
- All students now on EP13 cosine T_max=13 or SGDR variant to match SOTA #511

### 3. Composition opportunities (next round, when winners emerge)
- GradNorm-EMA α=0.5 + surface-loss-weight=2.0 — orthogonal mechanisms
- GradNorm-EMA + extended cosine EP15 (if edward #535 advances SOTA)
- Y-mirror aug + winning GradNorm config — both are additive

---

## Potential next research directions

1. **slw=2.0 full 13-epoch run** — current SOTA was from an EP5 timeout. A proper 13-epoch run with slw=2.0 + cosine T_max=13 should improve on 7.0063% meaningfully.
2. **Compose GradNorm-EMA α=0.5 with slw=2.0** (once thorfinn Run 2 confirms) — orthogonal mechanisms, expected to stack.
3. **Per-channel asymmetric loss weight** — don't just scale surface globally (slw=2.0), but specifically upweight tau_y (the slowest axis): `w_tau_y=3.0, w_tau_z=2.0, w_sp=1.5, w_vol=1.0`
4. **Y-mirror aug + slw=2.0** — once nezuko's augmentation bug is fixed, combine p=0.5 mirror aug with slw=2.0 baseline. Both improvements attack the surface representation problem from complementary angles.
5. **Compose y-mirror aug with EP15 cosine** (if #535 and #536 both win)
6. **Scheduled optimizer switch** (Lion→AdamW at EP3) — AdamW shows better per-epoch tau convergence, just slower start.
7. **Physics-informed no-slip boundary loss** — zero-velocity wall condition as auxiliary loss on tau magnitude at wall proximity.
8. **EMA model-soup** — average parameters of top-K best-val checkpoints (variance reduction, zero GPU cost).
9. **Frequency-aware tau loss** — weight loss contributions by spatial frequency (penalize high-freq tau residuals more heavily).

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
| Unit-vector cosine direction loss on tau (Arm B w=0.1) (#531) | NEGATIVE — tau_y/tau_z both regressed vs SOTA; direction loss is not the problem |
