# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 00:40 UTC (Round 23 launched)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #387 (alphonse feat16 RFF + QK-norm + STRING-sep), val_abupt **7.3816%** (EP11)

W&B run `wj6mn6ve`, group `alphonse-rff-sweep`. Beats PR #358 by −0.0105pp. All future PRs must beat val_abupt < **7.3816%**.

| Metric | PR #387 SOTA (val EP11 / test) | AB-UPT |
|---|---:|---:|
| `abupt` (val_abupt EP11) | **7.3816%** | — |
| `abupt` (test_abupt) | **8.5936%** | — |
| `surface_pressure` (test) | 4.4377% | 3.82% |
| `wall_shear` (test) | 7.9989% | 7.29% |
| **`volume_pressure` (test)** | **12.1885%** | **6.08% (×2.0 gap — primary laggard)** |

---

## Latest research direction from human researcher team

No new directives in the last cycle. Last actionable directive: Issue #252 (Modded-NanoGPT-derived levers). QK-norm produced the only positive (PR #358), now stacked in SOTA.

---

## Currently in-flight (8 active WIP PRs on tay, zero idle students)

| PR | Student | Hypothesis | EP now | val_abupt (latest) |
|---|---|---|---|---|
| #471 | askeladd | Signed-log transform on vol_p target (c=1.0), 2-arm DDP8 | EP0 | launching |
| #452 | edward | Separate 2-layer MLP volume decoder head | EP~6.7 | **8.934%** (best, still running) |
| #453 | fern | Cosine EMA ramp 0.99→0.9999 (run `2vxs26h2`) | EP4 | **9.677%** (vol_p=6.071%!) |
| #454 | frieren | Per-axis tau_y/tau_z loss weights ×2 on SOTA | EP6 | **8.355%** |
| #458 | nezuko | model-mlp-ratio 6 on SOTA stack | EP1 | 50.937% (early) |
| #459 | thorfinn | Lion β2=0.95 reactive sweep | EP6 | **8.246%** |
| #467 | alphonse | Per-axis output scaling for surface predictions | EP<1 | launching |
| #470 | tanjiro | Sign-preserving log1p on tau channels only | EP<1 | launching |

### Key EP6 metrics summary (2026-05-03 00:40 UTC)

- **PR #459 thorfinn (β2=0.95):** E1:46.27 → E2:26.71 → E3:12.48 → E4:9.78 → E5:8.79 → **E6:8.25** — on-track, EP7 gate ≤8.5% expected PASS
- **PR #454 frieren (tau_yz ×2):** E5:8.93 → **E6:8.36** — EP7 gate ≤8.5% expected PASS
- **PR #452 edward (sep vol decoder):** E3:13.69 → E4:10.09 → E5:9.15 → **best=8.934 EP~6.7** — vol_p collapsed to 8.3% by EP3. Stretched window to EP8.
- **PR #453 fern (cosine EMA):** E1:38.61 → E2:17.81 → E3:11.76 → **E4:9.68**, vol_p=**6.071%** (AT AB-UPT TARGET!) — strongest vol_p signal in fleet history

---

## Recent closeouts

- **PR #451 askeladd (volume-loss-weight=3.0) — CLOSED NEGATIVE.** EP6=9.272%, 1.89pp above SOTA bar. Loss scalar reweighting too blunt; delta per epoch decaying with no inflection.
- **PR #423 thorfinn (local tangent-frame features) — CLOSED NEGATIVE.** Uniform 5% regression across all targets, no tau-specific signal.
- **PR #365 nezuko — CLOSED ABANDONED.** Stale result, merge conflicts.

## Recent merged

- **PR #387 alphonse feat16 RFF — MERGED NEW SOTA** (val EP11 7.3816%, test 8.5936%). feat16 RFF on STRING-sep + QK-norm.
- **PR #358 thorfinn STRING-sep + QK-norm — MERGED** (val 7.3921%).
- **PR #311 edward STRING-sep PE — MERGED** (val 7.546%).

---

## Current research focus and themes

1. **Closing the volume_pressure gap (×2.0 vs AB-UPT: 12.1885% → 6.08%)** — the primary laggard. Three angles now converging: PR #452 (capacity: separate decoder), PR #453 (EMA schedule, incidental vol_p win to 6.07%), PR #471 (representation: signed-log). Any of these could be SOTA-level if vol_p closes.
2. **Closing the tau_y/tau_z gap (×2.53 / ×2.88)** — second laggard. PR #454 (frieren per-axis weights at EP6=8.36%, on-track), PR #470 (tanjiro log1p tau).
3. **Capacity scaling on SOTA stack** — PR #458 (mlp_ratio=6), PR #467 (per-axis output scaling).
4. **Optimizer dynamics** — PR #459 (Lion β2=0.95, EP6=8.25%, likely PASS EP7 gate).
5. **EMA scheduling** — PR #453 (cosine ramp, EP4 best val yet + vol_p at AB-UPT already).

---

## Largest remaining gaps to AB-UPT (test metrics from PR #387)

1. **volume_pressure** ×2.0 (12.1885% vs 6.08%) — PR #452 #453 #471 all targeting
2. **tau_z** ×2.83, **tau_y** ×2.49 — frieren #454, tanjiro #470
3. **wall_shear** ×1.10 (7.9989% vs 7.29%) — partly addressed by tau-axis weights
4. **surface_pressure** ×1.16 (4.4377% vs 3.82%) — PR #467 (per-axis output scaling)

---

## Negative results catalog (key lessons, do not retry on current stack)

| Lever | Outcome |
|---|---|
| Local tangent-frame INPUT features | NEGATIVE (#423) |
| Tangent-frame OUTPUT decomposition | EXHAUSTED (many PRs) |
| Channel-selective Huber on tau | NEGATIVE (#353) |
| Volume-loss-weight scalar rebalancing | NEGATIVE (#451) |
| Muon optimizer | NEGATIVE (#299) +4.09pp |
| Sandwich-norm | NEGATIVE diverged |
| U-net skips | NEGATIVE (+0.555pp) |
| Lion β values >0.99 | All closed on older stacks |
| dropout > 0 | Default 0.0 best |
| 256d / 768d hidden | NEGATIVE on multiple stacks |
| 6L / 8L depth | NEGATIVE |

---

## Potential next research directions (when slots open)

1. **Stacked decoder experiments** — if PR #452 merges, test adding a signed-log vol_p transform (#471 concept) on TOP of the separate-decoder-head stack.
2. **Progressive EMA fixed version** — if PR #453 merges, immediately test with corrected `--ema-cosine-total-epochs` matching actual training duration (cosine bug currently spans 50 epochs not ~12).
3. **Per-axis output normalization statistics recompute** — verify per-target std in loss normalization is fresh and per-axis correct.
4. **Slice-conditioned FFN width** — wider FFN only in middle slices (volumetric tokens).
5. **Curriculum on volume points** — ramp from small → 65k to ease optimizer early focus.
6. **Cross-attention bridge between surface and volume tokens** — explicit cross-attn vs current implicit shared attention.
7. **Test-time augmentation: rotation/reflection symmetries** — easy potential win.
8. **Grad-clip=0.3 confirmatory DDP8 run (yi branch, PR #431)** — askeladd confirmed interior optimum at clip=0.3 vs 0.5; confirmatory DDP8 run assigned.
