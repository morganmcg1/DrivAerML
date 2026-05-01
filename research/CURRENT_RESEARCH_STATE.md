# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 02:50 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #115 (thorfinn compound lr=1e-4 + EMA=0.999), test_abupt 10.580

**Largest single-PR gain in the project: −5.04% (11.142 → 10.580).** Compound lr=1e-4 + EMA=0.999 stacked additively. Note: vol_w=1.0 default → volume_pressure flat (+0.15% regression). Volume_pressure ×2.1 vs AB-UPT is now the leading binding gap.

| Metric | tay SOTA (PR #115) | PR #111 | AB-UPT | Gap |
|---|---:|---:|---:|---:|
| `abupt` mean | **10.580** | 11.142 | — | — |
| `surface_pressure` | **5.690** | 6.209 | 3.82 | ×1.5 |
| `wall_shear` | **10.419** | 11.138 | 7.29 | ×1.4 |
| `volume_pressure` | 12.740 | **12.548** | 6.08 | **×2.1** |
| `tau_x` | **8.908** | 9.436 | 5.35 | ×1.7 |
| `tau_y` | **12.491** | 13.525 | 3.65 | **×3.4** |
| `tau_z` | **13.071** | 13.992 | 3.63 | **×3.6** |

W&B run: `d03oghpp` — best val 9.484 (ep9). Reference val trajectory: 53.75 / 24.15 / 16.51 / 13.47 / 11.83 / 10.88 / 10.16 / 9.73 / 9.48.

## Comparison with yi advisor (parallel branch on different DDP project)

**yi current SOTA: ~7.33 abupt** (Wave 1 mid-training alphonse at step 413k, stack: Fourier PE + 5L + 128 slices + 60k pts + SDF + asinh + sw=2.0). Tay now at 10.580 = **−30% gap to close**.

Yi's confirmed winners (Wave 1):
- **Depth 6L/256d (#14 senku, −21.0%)** ← now in flight on tay (edward #146)
- Per-axis tau_y/z weights (#66 thorfinn, −3.1%) ← now in flight on tay (tanjiro #149)
- LR peak 5e-4 (#99 fern, −16.1%) ← partial port via askeladd #141 (lr=1e-4) + alphonse #148 (lr=1.5e-4)
- Wave 1 stack (Fourier PE, 5L, sw=2.0, asinh): mid-training 7.33 → likely 8.0-9.0 final

**Yi key finding: depth >> width. 6L/256d (4.73M) >> 4L/512d (12.7M) in 9-epoch budget.**

## Active Round 10 assignments (8 students, all DDP8)

| PR | Student | Hypothesis | Status (vs SOTA at same epoch) |
|---|---|---|---|
| **#149** | tanjiro | Per-axis tau_y/z weights W=1.5 (binding gap attack) | rt=15m, just launching |
| **#148** | alphonse | lr=1.5e-4 + EMA=0.999 (push LR ceiling) | rt=27m, just launching |
| **#147** | frieren | compound SOTA + wd=2e-3 | pod-stuck on iter 135, will pick up |
| **#146** | edward | 6L/256d depth swap (yi −21% lever, biggest untested) | rt=49m, ep0 val 56.72 (+5.5%) |
| **#142** | thorfinn | compound SOTA + vol_w=2.0 (recover vol gradient) | rt=98m, ep2 val 18.24 (+10.4%) |
| **#141** | askeladd | Compound lr=1e-4 + sw=2.0 + EMA=0.999 (3-way) | rt=115m, ep2 val 15.99 (**−3.2% LEAD**) |
| **#139** | fern | model-slices=256 (architecture sweep) | rt=192m, ep3 val 14.19 (+5.3%) |
| **#138** | nezuko | 5L depth swap (yi confirmed depth lever) | rt=230m, ep5 val 10.57 (**−2.8% LEAD**) |

## Key learnings to date

1. **Lion uncompiled (4L/512d) is the stable base** — 9 confirmed Lion+compile divergences. Vanilla Lion at lr=5e-5/wd=5e-4 was the original reference; lr=1e-4 + EMA=0.999 is now SOTA.
2. **lr=5e-5 → lr=1e-4 + EMA=0.999 compound = −5% jump** — confirmed in PR #115. Two confirmed levers compound cleanly.
3. **vol_w=1.0 vs 2.0 matters most for volume_pressure** — vol_p the leading binding gap. PR #142 testing vol_w=2.0 onto compound stack.
4. **EMA budget calibration**: 0.9999 too slow, 0.9995 baseline, **0.999 wins**, 0.998 too fast.
5. **T_max=50 sweet spot** — gentle cosine. T_max=100 closed (+4.7% vs PR111), T_max=16/24 closed.
6. **RFF closed-door across sigma** — σ=0.5/1.0/2.0 all regress vs vanilla Lion uncompiled.
7. **Warmup hurts** (frieren #109, +3.5%).
8. **Trajectory compression**: ep1 lead of −30-50% compresses to ~1-5% by ep9. The 9-epoch plateau is the bottleneck.
9. **5L depth shows -2.8% at ep5** — promising mid-training signal (nezuko #138). 5L per-epoch slower (~37 min/ep), may hit timeout at ep7-8.
10. **3-way compound (askeladd #141) showing -3.2% lead at ep2** — adding sw=2.0 onto compound stack is paying off early.

## Round 10 hypothesis themes

- **Compound stacks on PR #115 base:** vol_w=2.0 (#142), wd=2e-3 (#147), lr=1.5e-4 (#148), 3-way+sw=2.0 (#141)
- **Architecture/depth from yi:** 5L (#138), 6L/256d (#146), slices=256 (#139)
- **Binding gap attack:** per-axis tau_y/z weights W=1.5 (#149)

## Next research directions (priority order)

1. **If askeladd #141 wins** (-3.2% LEAD at ep2): confirm 3-way compound is new base, stack further.
2. **If nezuko #138 wins** (-2.8% LEAD at ep5): assign 6L/256d follow-up to verify yi's depth-over-width port.
3. **If edward #146 6L/256d wins**: compound 6L/256d + lr=1e-4 + EMA=0.999 (untried on yi too).
4. **Yi Wave 1 architecture port** — Fourier PE + asinh transform + SDF features. Volume_pressure ×2.1 vs reference is the dominant remaining gap.
5. **Tangential wall-shear loss** (yi #11 kohaku) — confirm flag exists in tay's train.py.
6. **Higher LR + cosine warmup** — lr=2e-4 with warmup ramp; thorfinn val was still descending at ep9.
7. **Mixup / curriculum** — fresh data-augmentation lever, untouched on tay.

## Reference trajectories

PR #50 vanilla val: 80.68 / 46.76 / 24.60 / 17.31 / 14.25 / 12.29 / 11.11 / 10.38 / 10.08
PR #111 EMA=0.999 val: 55.06 / 27.00 / 18.09 / 14.61 / 12.71 / 11.56 / 10.77 / 10.24 / 9.99
**PR #115 compound val: 53.75 / 24.15 / 16.51 / 13.47 / 11.83 / 10.88 / 10.16 / 9.73 / 9.48 (SOTA)**
nezuko #138 (5L) val: 58.80 / 25.99 / 16.69 / 13.35 / 11.57 / 10.57 (mid-flight, **−2.8% LEAD ep5**)
askeladd #141 (3-way) val: 50.49 / 23.03 / 15.99 (mid-flight, **−3.2% LEAD ep2**)
