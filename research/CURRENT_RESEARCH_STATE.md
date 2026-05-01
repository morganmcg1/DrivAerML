# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-01 01:25 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #115 (thorfinn compound lr=1e-4 + EMA=0.999), test_abupt 10.580

**Largest single-PR gain in the project: −5.04% (11.142 → 10.580).** Compound lr=1e-4 + EMA=0.999 stacked additively. Note: vol_w=1.0 default → volume_pressure flat (+0.1%). Next run (PR #142 thorfinn) adds vol_w=2.0 to recover volume gradient — projected ~10.3–10.4.

| Metric | tay SOTA (PR #115) | PR #111 | AB-UPT | Gap |
|---|---:|---:|---:|---:|
| `abupt` mean | **10.580** | 11.142 | — | — |
| `surface_pressure` | **5.690** | 6.209 | 3.82 | ×1.5 |
| `wall_shear` | **10.419** | 11.138 | 7.29 | ×1.4 |
| `volume_pressure` | 12.740 | **12.548** | 6.08 | **×2.1** |
| `tau_x` | **8.908** | 9.436 | 5.35 | ×1.7 |
| `tau_y` | **12.491** | 13.525 | 3.65 | **×3.4** |
| `tau_z` | **13.071** | 13.992 | 3.63 | **×3.6** |

W&B run: `d03oghpp` — best val 9.484 (ep9)

## Comparison with yi advisor (parallel branch on different DDP project)

**yi current SOTA: ~7.33 abupt** (Wave 1 mid-training alphonse at step 413k, stack: Fourier PE + 5L + 128 slices + 60k pts + SDF + asinh + sw=2.0). Tay now at 10.580 = **−30% gap to close**.

Yi's confirmed winners (Wave 1):
- **Depth 6L/256d (#14 senku, −21.0%)** ← biggest single lever, NOT yet ported to tay
- Per-axis tau_y/z weights (#66 thorfinn, −3.1%)
- LR peak 5e-4 (#99 fern, −16.1%)
- Wave 1 stack (Fourier PE, 5L, sw=2.0, asinh): mid-training 7.33 → likely 8.0-9.0 final

**Yi key finding: depth >> width. 6L/256d (4.73M) >> 4L/512d (12.7M) in 9-epoch budget.**

## Active assignments (8 students, all DDP8)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#142** | thorfinn | Compound SOTA + vol_w=2.0 (single delta from #115 SOTA) | Just assigned |
| **#141** | askeladd | Compound lr=1e-4 + sw=2.0 + EMA=0.999 (3-way compound) | rt=4m, launching |
| **#140** | — | — | — |
| **#139** | fern | model-slices=256 (double slice count architecture sweep) | rt=81m, ep1 56.82 |
| **#138** | nezuko | 5L depth swap (yi Wave 1 confirmed lever, single delta) | rt=120m, ep3 16.69 (**-7.7% vs SOTA at ep3**) |
| **#136** | alphonse | Lion + surface_loss_weight=2.0 | rt=212m, ep6 **10.64** (**-8% lead**) |
| **#135** | tanjiro | Lion + T_max=100 + EMA=0.999 | rt=218m, ep6 11.35 (-1.8%) |
| **#134** | frieren | Lion + wd=2e-3 | rt=222m, ep7 **10.50** (**-1.7%**) |
| **#133** | edward | Compound: T_max=50 + EMA=0.999 | rt=239m, ep7 10.77 (-0.3%) |

## Key learnings to date

1. **Lion uncompiled (4L/512d) is the stable base** — 9 confirmed Lion+compile divergences. Vanilla Lion at lr=5e-5/wd=5e-4 is the reference stack.
2. **Lion paper config wrong** — lr=5e-5/wd=5e-4 (AdamW-equivalent) crushes paper's lr=1.7e-5/wd=5e-3 by −27%.
3. **EMA budget calibration matters** — 0.9999 too slow (#90 closed), 0.999 wins (#111 MERGED), 0.9995 baseline.
4. **Cosine schedule has a sweet spot** — T_max=24 closed (+2.8%), T_max=50 wins (#110 MERGED, −0.34%), T_max=100 testing (#135).
5. **RFF closed-door across sigma** — σ=0.5/1.0/2.0 all regress vs vanilla Lion uncompiled.
6. **vol_w=3 closed-door** — both AdamW (#55) and Lion (#68) diverged.
7. **Width 768d budget-limited** — 5ep vs 9ep at 4L/512d.
8. **Warmup hurts** — frieren #109 closed (+3.5%).
9. **lr=1e-4 confounded but ep8-9 plateau overshoot** — Lion overshoots loss minimum at lr>5e-5 in 9-epoch budget. lr=3e-5 also losing (#113 +12.9%).
10. **Trajectory compression** — early lead (ep1 −20-50% better) compresses to ~1-3% by ep9. The 9-epoch plateau is the bottleneck.

## Round 9/10 hypothesis themes

- **Round 9 compound stacks (finishing):** T_max=50+EMA (edward #133), T_max=100+EMA (tanjiro #135), wd=2e-3 (frieren #134), sw=2.0 (alphonse #136)
- **Round 10 architecture/depth:** 5L depth swap (nezuko #138), slices=256 (fern #139)
- **Round 10 compound stack:** lr=1e-4 + sw=2.0 + EMA=0.999 (askeladd #141), compound SOTA + vol_w=2.0 (thorfinn #142)

## Next research directions (priority order)

1. **vol_w=2.0 on compound stack** — PR #142 thorfinn just assigned. Single variable from #115 SOTA. Projected 10.3–10.4.
2. **Port yi 6L/256d depth swap** — biggest unexplored lever (+21% on yi). After nezuko 5L result, follow up with 6L/256d if 5L wins.
3. **Surface_loss_weight=2.0** (alphonse #136 ep6 -8% lead) — if confirmed winner, compound with #115 SOTA.
4. **Per-axis tau_y/tau_z upweighting** — yi #66 got -3.1%; tay PR #54 diverged (too aggressive). Retry with W_y=1.5, W_z=1.5 conservatively.
5. **Yi full Wave 1 stack** — Fourier PE + asinh + SDF features. Volume_pressure ×2.1 vs ref is the target.
6. **Tangential wall-shear loss** (yi #11 kohaku) — confirm flag exists in tay's train.py.
7. **Higher LR + cosine warmup** — lr=2e-4 or 3e-4 with warmup ramp; thorfinn val was still descending at ep9.

## Key learnings to date

1. **Lion uncompiled (4L/512d) is the stable base** — 9 confirmed Lion+compile divergences.
2. **lr=5e-5 → lr=1e-4 + EMA=0.999 compound = −5% jump** — confirmed in PR #115.
3. **vol_w=1.0 vs 2.0 matters most for volume_pressure** — +0.5 regression on vol_p when missing.
4. **EMA budget calibration**: 0.9999 too slow, 0.9995 baseline, **0.999 wins**, 0.998 too fast.
5. **T_max=50 sweet spot** — gentle cosine. T_max=100 and T_max=16/24 closed.
6. **RFF closed-door** — σ=0.5/1.0/2.0 all regress vs vanilla Lion.
7. **Warmup hurts** (frieren #109, +3.5%).
8. **Trajectory compression**: ep1 lead of −30-50% compresses to ~1-5% by ep9. The 9-epoch plateau is the bottleneck.
9. **5L depth shows -7.7% at ep3** — very promising early signal (nezuko #138).

## Reference trajectories

PR #50 vanilla val: 80.68 / 46.76 / 24.60 / 17.31 / 14.25 / 12.29 / 11.11 / 10.38 / 10.08
PR #111 EMA=0.999 val: 55.06 / 27.00 / 18.09 / 14.61 / 12.71 / 11.56 / 10.77 / 10.24 / 9.99
**PR #115 compound val: 53.75 / 24.15 / 16.51 / 13.47 / 11.83 / 10.88 / 10.16 / 9.73 / 9.48 (SOTA)**
