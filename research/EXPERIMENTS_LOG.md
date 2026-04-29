# SENPAI Research Results — DrivAerML (`tay`)

W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`.

Targets to beat (lower is better, AB-UPT public reference):
`surface_pressure 3.82`, `wall_shear 7.29`, `volume_pressure 6.08`,
`tau_x 5.35`, `tau_y 3.65`, `tau_z 3.63`.

## Round 1 — opened 2026-04-29

8 students assigned in parallel on DDP8 (8 GPUs each, 96 GB VRAM, effective
bs scales with `nproc_per_node × per-GPU bs`). Strategy: 5 students compose
yi's confirmed-orthogonal wins (width × FiLM × cosine-EMA × Fourier × LR
warmup); 3 students push beyond yi with architectural / loss / TTA changes
that yi only got as far as Round-2 assignments for.

| PR | Student | Hypothesis |
|---|---|---|
| #30 | alphonse | yi PR #4 reproduce (4L/512d/8h, lr=5e-5, bs=4) — calibration |
| #31 | askeladd | Full composition stack: 512d × cosine-EMA × tangential × vol_w=2.0 |
| #32 | edward | Cosine LR + 5% warmup on top of 512d composition |
| #33 | fern | Gaussian Fourier coord features + 512d composition |
| #34 | frieren | AdaLN-zero per-block FiLM + 512d composition |
| #35 | nezuko | A01 — ANP cross-attention surface decoder |
| #36 | tanjiro | SDF-gated volume attention bias for near-wall p_v |
| #37 | thorfinn | Per-axis wall-shear loss weighting + bilateral-symmetry TTA |
