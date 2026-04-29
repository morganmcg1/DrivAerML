# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-29
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`
- **Most recent direction from human team:** _(none on tay yet — yi advisor
  carried Issue #18: "be bolder; replace the backbone; mine `noam`/`radford`
  branches; trust students to take big leaps")_

## Calibration status

`tay` was just bootstrapped from `main` + DDP8 hardening. None of yi's
proven wins have landed in code yet — the trainer defaults are 3L/192d/3h,
lr=3e-4, bs=2, 40k points, no FiLM, no projection loss, fixed-decay EMA.

Round 1 on tay therefore has two simultaneous purposes:

1. **Calibrate** the tay/DDP8 baseline against the yi numbers (16.64 merged,
   15.82 pending) so we can claim genuine improvements.
2. **Compose orthogonal wins** that yi never finished merging together
   (FiLM × cosine-EMA × 512d × tangential-loss is the obvious composition).

A subset of students push beyond yi with bold architecture or training
changes that yi only got as far as Round-2 assignments for.

## Active research themes

1. **Composition of yi orthogonal wins on DDP8.** Width 512d (chihiro),
   FiLM AdaLN-zero (frieren), cosine EMA 0.99→0.9999 (norman), vol_w=2.0
   (gilbert), tangential wall-shear projection (kohaku). Yi predicted ~12-13
   abupt for the full stack; we run that on DDP8 and verify.
2. **Schedule and feature engineering.** Cosine LR + warmup, Gaussian
   Fourier coordinate features. Both were Round-1 ideas in yi that never
   produced clean comparable results because of stability bugs that have
   since been patched (gradient clipping is now default in the trainer).
3. **Architectural swap.** Replace the Transolver surface MLP head with an
   ANP-style cross-attention decoder. Largest single architectural win
   identified from `noam`'s prior work; never tested on DrivAerML.
4. **Volume-targeted attention.** SDF-gated volume attention bias for
   near-wall p_v emphasis. p_v is one of the hardest targets to move and
   AB-UPT's 6.08 reference is a real gap from yi's 14.21.
5. **Wall-shear axis disparity.** tau_y and tau_z are systematically worse
   than tau_x. Per-axis loss weighting + bilateral-symmetry TTA both
   directly attack this gap with low-risk single deltas.

## Round 1 — assignments (8 students, all DDP8)

| Student | Stream | Hypothesis |
|---|---|---|
| alphonse | calibrate | Reproduce yi PR #4 winning config (4L/512d/8h, lr=5e-5, bs=4) on DDP8 — establishes tay baseline |
| askeladd | exploit | Full yi composition stack: 512d × cosine-EMA × tangential proj × vol_w=2.0 |
| edward | exploit | Cosine LR + 5% warmup on top of 512d composition |
| fern | exploit | Gaussian random Fourier features + 512d composition |
| frieren | exploit | AdaLN-zero per-block FiLM + 512d composition |
| nezuko | explore | A01 — ANP cross-attention surface decoder (replace head) |
| tanjiro | explore | SDF-gated volume attention bias (near-wall p_v emphasis) |
| thorfinn | explore | Per-axis wall-shear loss weighting + bilateral-symmetry TTA |

## Next research directions (Round 2 candidates, queued)

These are the strongest ideas to try after Round 1 lands:

- **Backbone replacement** — Perceiver-IO, Mamba-2 SSM surface decoder, full
  GeoTransolver/Transolver-3 backbone, GINO/FNO operator on the volume
  channel.
- **Pretraining** — denoising on 50 unlabelled test geometries; MAE 75%
  point masking; DPOT transfer.
- **Loss reformulation** — squared rel-L2 (no sqrt, smooth backward);
  deep-evidential-regression head with NIG; uncertainty-aware loss.
- **MoE / routing** — Soft-MoE FFN with 4 experts; dispatcher conditioned
  on geometry token.
- **Data representation** — Morton/Hilbert ordering for SSM-friendly
  sequences; multi-resolution patch tokens; equivariant local frames.
- **Optimization** — µP scaling so width sweeps don't need per-width LR
  retuning; AdamW + Sophia/Lion comparisons.

## Constraints

- `SENPAI_MAX_EPOCHS` and `SENPAI_TIMEOUT_MINUTES` set by harness —
  do not override. Throughput improvements (compile, AMP, point-count,
  attention scaling, DDP8 communication tuning) are the lever for "more
  update budget" inside the budget.
- DDP8 = 8 GPUs per student × 96 GB each. Effective batch size at 512d/bs=4
  per-GPU = 32. Use `--wandb_group` whenever a hypothesis needs multiple
  arms.
