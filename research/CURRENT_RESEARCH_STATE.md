# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 06:20 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #115 (thorfinn compound lr=1e-4 + EMA=0.999), test_abupt 10.580

**Largest single-PR gain: −5.04% (11.142 → 10.580).** Compound lr=1e-4 + EMA=0.999 stacked additively.

**PR #115 verified config (from W&B `d03oghpp`):**
- lr=1e-4, weight_decay=5e-4, optimizer=lion, lion_beta1=0.9, lion_beta2=0.99
- use_ema=True, ema_decay=0.999
- **lr_cosine_t_max=0** (NO cosine), lr_warmup_epochs=0
- model_layers=4, model_hidden_dim=512, model_heads=8
- **model_slices=128** (not 96 default)
- model_mlp_ratio=4
- **train/eval surface_points=65536, train/eval volume_points=65536** (not 40k)
- volume_loss_weight=1, surface_loss_weight=1, batch_size=4
- compile_model=False

| Metric | tay SOTA (PR #115) | AB-UPT | Gap |
|---|---:|---:|---:|
| `abupt` mean | **10.580** | — | — |
| `surface_pressure` | **5.690** | 3.82 | ×1.5 |
| `wall_shear` | **10.419** | 7.29 | ×1.4 |
| `volume_pressure` | 12.740 | 6.08 | **×2.1** |
| `tau_x` | **8.908** | 5.35 | ×1.7 |
| `tau_y` | **12.491** | 3.65 | **×3.4** |
| `tau_z` | **13.071** | 3.63 | **×3.6** |

W&B SOTA run: `d03oghpp` — best val 9.484 (ep9). val→test ratio ≈ 1.115.
PR #115 SOTA val trajectory: 53.75 / 24.15 / 16.51 / 13.47 / 11.83 / 10.88 / 10.16 / 9.73 / 9.48.

## Round 10 in-flight status

| PR | Student | Hypothesis | Run state | Latest val (vs SOTA ep9 9.48) |
|---|---|---|---|---|
| **#161** | askeladd | lion_beta2=0.999 (optimizer sweep, single delta) | round11 — just assigned | — |
| #162 | edward | model_dropout=0.05 (regularization single delta) | round11 — just assigned | — |
| **#141** | askeladd | sw=2.0 + vw=2.0 + T_max=50 | **CLOSED** | test 10.605 (+0.23%), val won (-0.41%) but tau_y regressed on test |
| #142 | thorfinn | compound + vol_w=2.0 | finalizing test eval | best_val 10.607 (+11.8% NOT a winner) |
| **#146** | edward | 6L/256d depth swap | **CLOSED** | test 12.662 (+19.7%, depth-swap family closed) |
| #147 | frieren | compound + wd=2e-3 | **POD STUCK** (running old PR #134 round9) | no round10 W&B run |
| #149 | tanjiro | per-axis tau_y/z weights W=1.5 | **POD STUCK** (GPU 0%, iter 167 since 02:29) | no round10 W&B run |
| #157 | nezuko | mlp_ratio=6 | running | val ep2 25.28 (SOTA ep2 16.51, +53% behind) |
| #158 | alphonse | vol_pts=96k (vol_p attack) | running | val ep1 77.22 (early, high pts load) |
| #159 | fern | Lion β1=0.95 | running | val ep1 59.07 (underperforming early) |

### Key observations
- **askeladd #141 val→test divergence**: val −0.41% (9.445 vs 9.484) BUT test +0.23% (10.604 vs 10.580). Test breakdown: surface_p −0.08%, vol_p −0.60% (improved!), wall_shear_y +1.25% (regression), wall_shear_z +0.35%. Net: val/test ratio 1.123 vs SOTA 1.115 — slight overfit to val. **CLOSEOUT, not merge.**
- **vol_p test improved -0.60%** with vw=2.0 + sw=2.0 — confirms volume gradient lever has signal but tau-axis regressions cancel it.
- **thorfinn #142 (vw=2.0 alone)** at best_val 10.607 = +11.8% behind SOTA. vw=2.0 alone NOT effective on top of compound base.
- **edward #146 (6L/256d)** test 12.662 = +19.7% behind. Depth swap requires more than 9-epoch budget.
- **thorfinn #142** vol_w=2.0 trajectory looks slow — at 11.15 with ~3 epochs of timeout left.
- **edward #146** 6L/256d at 11.92 — same slow-start pattern as nezuko's failed 5L. Depth swap with same point budget likely won't fit in 9-epoch timeout.
- **frieren and tanjiro pods broken**. Frieren still running PR #134 (round9, GPU 100%); tanjiro idle (GPU 0%) despite branch checked out. Cannot intervene from advisor — pods own their loop.
- **alphonse student auto-corrected** the 4-flag baseline mismatch I introduced — relaunched with vol_pts=96k single-delta from verified SOTA config. Same applies to nezuko — student caught and corrected before launching.

## Round 10/11 closeouts (test_abupt, primary metric)

| PR | Student | Test result | vs SOTA |
|---|---|---:|---:|
| #138 | nezuko (5L depth) | 11.213 | +5.98% (CLOSED) |
| #139 | fern (slices=256) | 12.389 | +17.1% (CLOSED, 98.8% VRAM) |
| #141 | askeladd (sw+vw+T_max 3-way) | 10.605 | +0.23% (CLOSED, val win/test loss) |
| #146 | edward (6L/256d) | 12.662 | +19.7% (CLOSED, depth-swap family closed) |
| #148 | alphonse (lr=1.5e-4) | early-close ep4 | +40% (CLOSED, LR ceiling confirmed) |

## Comparison with yi advisor (parallel branch)

**yi current SOTA: ~7.33 abupt** — tay's 10.580 is **−30% gap**. Yi confirmed: depth >> width (6L/256d −21%), Fourier PE + asinh + sw=2.0 stack works. Tay needs the architecture port.

## Critical lesson learned

**PR template baseline accuracy is now load-bearing.** Three recent PRs had wrong reproduce commands (slice/point/cosine flags missing or wrong). Students caught it themselves. Going forward: copy the exact verified `d03oghpp` config flags for any "single delta from SOTA" instruction.

## Next research directions (priority order)

1. **alphonse #158 vol_pts=96k** — most promising active run, directly attacking dominant binding gap (vol_p ×2.1 vs AB-UPT).
2. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Big architectural move, potential 20%+ gain. Tanjiro/frieren pods stuck and unavailable; assign to working pods.
3. **Per-axis tau loss weights** — tanjiro #149 stuck; may need to reassign to a working pod (tau_y/tau_z are ×3.4-3.6 binding gaps). No train.py flag exists — requires student code addition.
4. **thorfinn #142 closeout** — vol_w=2.0 at best_val +11.8%, not competitive. Clean up when submitted.
5. **Optimizer hyperparam map** — lion_beta2=0.999 (#161) and model_dropout=0.05 (#162) are the two open single-delta experiments. lion_beta1=0.95 (#159) early signal weak.
6. **Frieren/tanjiro pod recovery** — pods stuck; human team intervention needed. Cannot fix from advisor side.

## Optimizer hyperparam map (confirmed on tay)

| Param | Min tested | SOTA | Max tested | Status |
|---|---|---|---|---|
| lr | 5e-5 (PR #50) | **1e-4 (PR #115)** | 1.5e-4 (CLOSED) | Ceiling found |
| ema_decay | 0.9999 (slow) | **0.999** | 0.998 (fast) | Sweet spot confirmed |
| lion_beta1 | — | **0.9** | 0.95 (#159 running) | In flight |
| lion_beta2 | — | **0.99** | 0.999 (#161 assigned) | Untested |
| model_dropout | — | **0.0** | 0.05 (#162 assigned) | Untested |
