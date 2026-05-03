# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 ~15:25Z — 8 active WIP PRs, 0 ready for review, 0 idle students; askeladd #516 just rebased clean and relaunched; frieren #501 still CONFLICTING (escalated, third request)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #511 (edward extended-cosine EP13), val_abupt **7.0134%** (W&B `5o7jc7wi`)

All future PRs must beat val_abupt < **7.0134%**. Test val: test_abupt=8.3130%.

| Metric | PR #511 SOTA (val EP13) | AB-UPT |
|---|---:|---:|
| `abupt` | **7.0134%** | — |
| `surface_pressure` | 4.5104% | 3.82% |
| `wall_shear` | 7.9650% | 7.29% |
| `volume_pressure` | **4.2168%** | 6.08% (BEATEN) |
| `tau_x` | 7.0053% | 5.35% |
| `tau_y` | **8.7717%** | 3.65% |
| `tau_z` | **10.5629%** | 3.63% |

### Reproduce (canonical SOTA stack)
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
Note: `vol_points_schedule=None` (NOT a curriculum); `train_volume_points=65536` constant.

---

## Latest research direction from human researcher team

No new directives since last cycle. Continuing organic tau_y/tau_z + extended-schedule programme.

---

## Currently in-flight (8 active WIP PRs on tay, ZERO idle students)

| PR | Student | Hypothesis | Latest val_abupt | Status |
|---|---|---|---:|---|
| #535 | edward | Extended cosine to **EP15** on PR #511 stack (no vol-curriculum) | EP1=50.91% | run `nh2ke150` healthy; --vol-points-schedule correctly omitted |
| #534 | tanjiro | Multi-scale STRING-sep bands σ∈{0.25,1.0,4.0} 8feats/band, output_dim=144 | EP1=28.20% | run `loxzj4xq` EP1 strictly better than SOTA EP1; awaiting EP3+ |
| #532 | nezuko | AdamW vs Lion optimizer comparison | EP3=9.35% | run `3hm5ae1j` — AdamW trailing Lion 1.3-1.9pp at every epoch; trending NEG |
| #531 | fern | Unit-vector cosine direction loss on tau (denormalize-first) Arm B w=0.1 | EP3=8.02% | run `3lurbotq` healthy progressing; on par with SOTA arms |
| #523 | thorfinn | GradNorm EMA-proxy dynamic loss balancing | EP3.5=**7.43%** | run `9477cjoh` — TZ already below SOTA terminal (10.66 < 10.56), TY borderline (9.20 vs 8.77); approaching win |
| #516 | askeladd | Per-channel tau_y/tau_z reweight (Run A: w_y=2.0, w_z=2.5) vs PR #511 | (just launched) | run `4uw2c4z2` — REBASED clean, relaunched 15:21Z |
| #510 | alphonse | Surface-loss-weight=2.0 sweep | EP3=**7.65%** | run `qqtdnlwq` — leading SOTA every channel; vol-curriculum 16k→32k entering EP4 |
| #501 | frieren | Anisotropic STRING priors (sigma_x/y/z) | (killed) | run `i5fgc06e` watchdog-killed at step 33648; **CONFLICTING vs tay**, escalated 3rd time |

---

## Top contenders (live W&B data)

### #510 alphonse slw=2.0 — EP3=7.6495% (leading every channel)
Trajectory: EP1=29.21% → EP2=9.03% → EP3=7.65%. Already 0.6pp better than SOTA #511 EP3 (14.15%), within 0.7pp of SOTA terminal. Projected SOTA breach by EP5-EP6 if slope holds. vol_p=4.46% slight regression vs SOTA (+0.25pp) is acceptable cost.

### #523 thorfinn GradNorm-EMA — EP3.5=7.4347% (tau_z below SOTA terminal!)
Trajectory: EP1=30.09% → EP2=9.46% → EP3=8.00% → EP3.5=**7.43%**. EMA-loss-proxy is doing exactly what was hypothesized: tau_z weight elevated to 1.50, tau_z val now 10.655% (below SOTA 10.5629%, **winning on this axis**). tau_y=9.199% (border to SOTA 8.7717%, ~0.43pp away). Win condition (<7.0134% AND tau_y<8.77 AND tau_z<10.56): tau_z cleared, tau_y close, val_abupt 0.42pp away.

### #531 fern unit-vec direction loss w=0.1 — EP3=8.0158% (healthy)
Trajectory: EP1=31.13% → EP2=9.76% → EP3=8.02%. On par with strong arms; cosine loss isn't disrupting primary objective. tau_y=10.51%, tau_z=11.92% — needs ~1.74pp tau_y / 1.36pp tau_z drop by EP8 to win.

### #534 tanjiro multi-scale bands — EP1=28.20% (best EP1 ever)
Strictly below SOTA EP1 on every metric. Awaiting EP3+ to confirm trajectory shape.

---

## Negative trajectories

### #532 nezuko AdamW — trailing Lion ~1.5pp persistently
EP1: +1.1pp; EP2: +2.5pp; EP3: +1.7pp behind. Letting it run to EP8 minimum for clean negative-result documentation.

---

## Recent closeouts

- **PR #511 edward (extended cosine EP13) — MERGED NEW SOTA.** val=7.0134%, test=8.3130%. Replaced PR #489 SOTA.
- **PR #506 nezuko (2× surface points) — CLOSED-NEG.** Slower/epoch beats resolution gains.
- **PR #499 fern (TTA mirror-y) — CLOSED-NEG.** TTA hurts (+1.18pp).
- **PR #489 thorfinn (vol-points curriculum) — was SOTA; superseded by #511.**

---

## Cross-cutting observations

### tau_y/tau_z: STILL primary open problem; FOUR active attacks
- **Per-channel reweighting** (#516 askeladd vs PR #511) — relaunched, awaiting results
- **Dynamic balancing GradNorm-EMA** (#523 thorfinn) — STRONG, tau_z winning
- **Surface-loss-weight=2.0** (#510 alphonse) — STRONG, leading every channel
- **Unit-vector direction loss** (#531 fern) — healthy, on-par trajectory
- **Anisotropic STRING priors** (#501 frieren) — STILL CONFLICTING, idle GPU

### vol_p: holding gains
- SOTA #511 vp=4.22% well below AB-UPT 6.08%
- alphonse slw=2.0 slight regression (+0.25pp) — acceptable cost; watch through vol-curriculum transition

### Optimizer family
- Lion + lr=1e-4 + wd=5e-4 confirmed optimal on this stack
- AdamW (#532) trailing 1.5pp persistently — Lion is the right answer

---

## Current research focus and themes

1. **EXTRACT a new SOTA from #510 (alphonse slw=2.0) or #523 (thorfinn GradNorm)** — both on track to breach #511's 7.0134% by EP6-EP8. Mark for review on first sub-7.0% intermediate val.
2. **Resolve idle frieren GPU (#501)** — 4-5h of GPU time wasted; rebase escalated.
3. **Watch askeladd (#516) Run A vs PR #511** — does per-channel reweighting orthogonally help when extended cosine is already in the stack?
4. **Confirm tanjiro (#534) multi-scale band trajectory** — EP1 was best-ever; need EP3-EP5 to know if specialization theory holds.
5. **Compose winners**: when alphonse slw=2.0 + thorfinn GradNorm both clear SOTA, the natural next round is a composed run (slw=2.0 surface upweight × GradNorm EMA balancer) on the #511 stack — orthogonal mechanisms.

---

## Potential next research directions (Round 28+)

1. **Compose alphonse slw=2.0 + thorfinn GradNorm-EMA** — surface upweight is structural prior, GradNorm is data-driven; orthogonal.
2. **Compose alphonse slw=2.0 + #511 extended-cosine EP15** — if alphonse wins, layer onto edward's #535 trajectory.
3. **Frieren-style anisotropic STRING + thorfinn vol-curriculum** — once frieren rebases.
4. **Wavelet/multi-resolution input encoding** — alternative spectral attack if #501 / #534 plateau.
5. **Learnable Fourier basis (NTK-style)** — learn small Fourier basis matrix instead of fixed sinusoids.
6. **EMA model-soup average across top-3 best-val checkpoints** — variance reduction.
7. **Surface point density 3× (196608 pts)** — only revisit if multi-resolution surface attention is added (so cost doesn't dominate).
8. **Physics-informed boundary condition loss** — enforce no-slip wall condition via auxiliary loss on tau magnitude near wall.

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
| 2× surface point density (#506) | NEGATIVE — slower/epoch beats density |
| tau_yz scalar loss-weight reweight (#142, #454, #467) | EXHAUSTED — three NEG, problem upstream |
| mlp_ratio=6/8 wider FFN (#458) | NEGATIVE — mlp4 is optimal |
| Signed-log target transform (#471 arm-b) | NEGATIVE |
| log1p target transform (#481) | NEGATIVE |
| AdamW vs Lion (#532, in progress) | TRENDING NEG — Lion optimal for stack |
