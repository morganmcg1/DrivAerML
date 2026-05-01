# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 05:25 UTC
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
| **#141** | askeladd | sw=2.0 + vw=2.0 + (T_max=50 confound) | running rt=252m, near timeout | **9.67 (+2.0% above SOTA)** |
| #142 | thorfinn | compound + vol_w=2.0 | running rt=233m, near timeout | 11.15 (~ep5 SOTA equiv) |
| #146 | edward | 6L/256d depth swap (yi −21% port) | running rt=186m | 11.92 (~ep4-5 SOTA equiv) |
| #147 | frieren | compound + wd=2e-3 | **POD STUCK on PR #134 round9 still** | no round10 W&B run yet |
| #149 | tanjiro | per-axis tau_y/z weights W=1.5 | **POD IDLE 0% GPU** | no round10 W&B run yet |
| #157 | nezuko | mlp_ratio=6 (yi Wave 1 lever) | running rt=31m, attempt 3 (×2 OOM) | val ep0=56.05 |
| #158 | alphonse | vol_pts=96k (attack vol_p ×2.1 gap) | running rt~10m, fresh launch | no val yet |
| #159 | fern | Lion β1=0.95 (momentum sweep) | running rt~5m, retry after fail | no val yet |

### Key observations
- **askeladd #141 4-way** is approaching timeout at val 9.67 — close but unlikely to beat SOTA 9.48. Expected test ~10.78, +1.9% behind SOTA test 10.580. **Will close if final val > 9.48.**
- **thorfinn #142** vol_w=2.0 trajectory looks slow — at 11.15 with ~3 epochs of timeout left.
- **edward #146** 6L/256d at 11.92 — same slow-start pattern as nezuko's failed 5L. Depth swap with same point budget likely won't fit in 9-epoch timeout.
- **frieren and tanjiro pods broken**. Frieren still running PR #134 (round9, GPU 100%); tanjiro idle (GPU 0%) despite branch checked out. Cannot intervene from advisor — pods own their loop.
- **alphonse student auto-corrected** the 4-flag baseline mismatch I introduced — relaunched with vol_pts=96k single-delta from verified SOTA config. Same applies to nezuko — student caught and corrected before launching.

## Round 10 closeouts (test_abupt, primary metric)

| PR | Student | Test result | vs SOTA |
|---|---|---:|---:|
| #138 | nezuko (5L depth) | 11.213 | +5.98% (CLOSED) |
| #139 | fern (slices=256) | 12.389 | +17.1% (CLOSED, 98.8% VRAM) |
| #148 | alphonse (lr=1.5e-4) | early-close ep4 | +40% (CLOSED, LR ceiling 1e-4 to 1.5e-4) |

## Comparison with yi advisor (parallel branch)

**yi current SOTA: ~7.33 abupt** — tay's 10.580 is **−30% gap**. Yi confirmed: depth >> width (6L/256d −21%), Fourier PE + asinh + sw=2.0 stack works. Tay needs the architecture port.

## Critical lesson learned

**PR template baseline accuracy is now load-bearing.** Three recent PRs had wrong reproduce commands (slice/point/cosine flags missing or wrong). Students caught it themselves. Going forward: copy the exact verified `d03oghpp` config flags for any "single delta from SOTA" instruction.

## Next research directions (priority order)

1. **If askeladd #141 closes as non-winner**: vol_w=2.0 isolation (already in flight via thorfinn #142).
2. **If thorfinn #142 wins**: confirm vol_w as new compound lever. Stack with sw=2.0 next.
3. **If alphonse #158 vol_pts=96k wins**: vol_p ×2.1 gap closes via compute, not architecture.
4. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Volume_pressure ×2.1 vs reference is the dominant gap and architecture-level fix.
5. **Tangential wall-shear loss** (yi #11 kohaku) — confirm flag exists in tay's train.py.
6. **Higher LR + cosine warmup** — lr=2e-4 with warmup ramp.
7. **Mixup / curriculum** — fresh data-augmentation lever, untouched on tay.
