# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-04-30 21:55 UTC
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #111 (tanjiro EMA=0.999), test_abupt 11.142

Two consecutive merges in last 90 min: PR #110 (T_max=50) → PR #111 (EMA=0.999). Total improvement: **−0.59% from 11.208 (PR #50) → 11.142**.

| Metric | tay SOTA (PR #111) | AB-UPT | Gap |
|---|---:|---:|---:|
| `abupt` mean | **11.142** | — | — |
| `surface_pressure` | 6.209 | 3.82 | ×1.6 |
| `wall_shear` | 11.138 | 7.29 | ×1.5 |
| `volume_pressure` | 12.548 | 6.08 | **×2.1** |
| `tau_x` | 9.436 | 5.35 | ×1.8 |
| `tau_y` | 13.525 | 3.65 | **×3.7** |
| `tau_z` | 13.992 | 3.63 | **×3.9** |

W&B run: `ab3y4ej7` — best val 9.989 (ep9, first sub-10 on tay)

## Comparison with yi advisor (parallel branch on different DDP project)

**yi current SOTA: 10.69 abupt** (PR #99 fern, lr=5e-4 base + 6L/256d depth). Tay is +4.2% behind yi.

Yi's compounding wins to date:
- Width 4L/512d (#4 chihiro, −4.3%)
- **Depth 6L/256d (#14 senku, −21.0%)** ← biggest single lever, NOT yet ported to tay
- Per-axis tau_y/z weights (#66 thorfinn, −3.1%)
- LR peak 5e-4 (#99 fern, −16.1%)

**Yi's key finding: depth >> width in epoch-budget regime.** 6L/256d (4.73M params) crushed 4L/512d (12.7M params).

## Active assignments (8 students, all DDP8)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| **#115** | thorfinn | Compound: Lion lr=1e-4 + EMA=0.999 | Running rt=60m, ep1 53.75 |
| **#133** | edward | Compound: Lion T_max=50 + EMA=0.999 (two merged levers) | Running, just started |
| **#134** | frieren | Lion + wd=2e-3 (4× current — Lion paper recommends higher wd) | Running, just started |
| **#135** | tanjiro | Lion + T_max=100 + EMA=0.999 (schedule sweep extension) | Running, just started |
| **#136** | alphonse | Lion + surface_loss_weight=2.0 (binding-gap attack on tau_y/tau_z) | Running, just started |
| **#114** | askeladd | Lion + EMA=0.998 (EMA sweep, faster) | Running v2 rt=84m, ep2 22.93 (−51% vs vanilla, replicating tanjiro) |
| **#113** | nezuko | Lion + lr=3e-5 (LR sweep lower bound) | Running rt=190m, ep6 13.88 (+12.9% vs vanilla, **CONFIRMED LOSER**) |
| **#72** | fern | AdamW+RFF+compile + per-axis tau_y/tau_z | Running rt=156m, ep9 16.48 (way off SOTA) |

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

## Round 9 active hypotheses (hypothesis stacking on EMA=0.999 SOTA)

- **Compound stacks:** lr=1e-4+EMA (thorfinn #115), T_max=50+EMA (edward #133)
- **Regularization:** wd=2e-3 (frieren #134, 4× current)
- **Schedule extension:** T_max=100 (tanjiro #135, 4% decay vs T_max=50's 8%)
- **Loss balance:** surface_w=2.0 (alphonse #136, target tau_y/tau_z gap)
- **EMA sweep continuation:** EMA=0.998 (askeladd #114 v2)

## Next research directions (priority order)

1. **PORT YI'S DEPTH SWAP — 6L/256d on Lion uncompiled** — biggest unexplored lever. Yi got −21% from this single change. Will assign to next student to free up.
2. **Per-axis tau_y/tau_z upweighting** — yi got −3.1%; tay's #54 fern attempt diverged but likely too aggressive. Retry conservatively (e.g., W_y=1.5, W_z=1.5).
3. **LR=5e-4 + warmup + cosine** — yi's high-LR base; tay needs schedule with lr=5e-4 to avoid Lion overshoot.
4. **Tangential wall-shear loss** (yi #11 kohaku) — needs port to tay's train.py if not present.
5. **Perceiver-IO backbone** — yi #18 directive, biggest architecture lever.
6. **Slice count sweep** — model-slices=64/256 (current 128).

## Plateau Protocol status

Current count of close-without-merge since PR #50: 3 (counting #109, #112, #92). Two merges in last 90min broke any plateau forming. **Not yet at plateau threshold (5+ closes required).**

## Reference (vs current SOTA)

| Target | AB-UPT | tay SOTA (PR #111) | yi SOTA | Gap (tay→AB-UPT) |
|---|---:|---:|---:|---:|
| `abupt` mean | — | **11.142** | 10.69 | — |
| `surface_pressure` | 3.82 | 6.209 | 6.97 | ×1.6 |
| `wall_shear` | 7.29 | 11.138 | 11.69 | ×1.5 |
| `volume_pressure` | 6.08 | 12.548 | 7.85 | **×2.1** (yi 1.3×) |
| `tau_x` | 5.35 | 9.436 | 10.17 | ×1.8 |
| `tau_y` | 3.65 | 13.525 | 13.73 | **×3.7** |
| `tau_z` | 3.63 | 13.992 | 14.73 | **×3.9** |

**Volume_pressure (×2.1 vs ref, 1.6× yi's 7.85) is the biggest tay-specific gap.** Tau_y/tau_z (×3.7/×3.9) are the binding gaps both branches share.

## Vanilla SOTA reference per-epoch trajectory (for comparison)

PR #50 vanilla val: 80.68 / 46.76 / 24.60 / 17.31 / 14.25 / 12.29 / 11.11 / 10.38 / 10.08
PR #110 T_max=50 val: 75.57 / 46.58 / 24.22 / 16.70 / 13.52 / 11.84 / 10.90 / 10.29 / 10.06
**PR #111 EMA=0.999 val: 55.06 / 27.00 / 18.09 / 14.61 / 12.71 / 11.56 / 10.77 / 10.24 / 9.99 (current SOTA)**
