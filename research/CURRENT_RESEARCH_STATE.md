# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 00:10 UTC (Round 23 EP11 watch)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #387 (alphonse feat16 RFF + QK-norm + STRING-sep), val_abupt **7.3816%** (EP11)

W&B run `wj6mn6ve`, group `alphonse-rff-sweep`. Beats PR #358 by −0.0105pp. All future PRs must beat val_abupt < **7.3816%**.

### Round 23 EP11 watch — frieren #454 on the SOTA boundary

- **frieren #454 (tau_yz×2.0):** EP11 val=**7.3848%** (+0.0032pp above SOTA — essentially tied). Slope EP10→11=−0.060pp. Projection EP12≈7.32%, EP13≈7.27%. **Strong SOTA candidate**, ~3 epochs of budget remaining.
- **thorfinn #459 (Lion β2=0.95):** EP11 val=**7.4329%** (+0.0513pp above SOTA). Slope decelerated to −0.034pp/ep. Won't beat SOTA. Run 2 (β2=0.97) de-prioritized.
- **edward #452 (separate vol decoder):** **FINISHED.** best_val=**7.9028%** (EP11), test=**8.8681%**. Val-vol_p=4.80% (below AB-UPT) but test-vol_p=12.376% (basically tied with SOTA test 12.189%). **Strong val→test overfitting on vol_p.** Negative result; close pending student post.
- **fern #453 (cosine EMA, buggy schedule):** EP9 val=**7.8099%**, vp=4.87%. Won't beat SOTA but vol_p remains AB-UPT-class. Direction confirmed; rerun with fixed `--ema-cosine-total-epochs 12` is the obvious follow-up.

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
| #471 | askeladd | Signed-log transform on vol_p target (c=1.0), 2-arm DDP8 | EP early | launching |
| #452 | edward | Separate 2-layer MLP volume decoder head | **FINISHED EP11** | **best=7.9028% / test=8.8681%** — NEG |
| #453 | fern | Cosine EMA ramp 0.99→0.9999 (run `2vxs26h2`) | EP9 | **7.8099%** (vp=4.87%) |
| #454 | frieren | Per-axis tau_y/tau_z loss weights ×2 on SOTA | **EP11** | **7.3848%** — SOTA boundary |
| #458 | nezuko | model-mlp-ratio 6 on SOTA stack | early | — |
| #459 | thorfinn | Lion β2=0.95 reactive sweep | EP11 | **7.4329%** — won't beat SOTA |
| #467 | alphonse | Per-axis output scaling for surface predictions | early | — |
| #470 | tanjiro | Sign-preserving log1p on tau channels only | early | — |

### Round-23 EP11 trajectory summary

| PR | EP9 | EP10 | EP11 | Slope (10→11) | Bedget left | SOTA gap |
|---|---:|---:|---:|---:|---|---:|
| #454 frieren | 7.591 | 7.445 | **7.385** | −0.060pp | ~3-4 epochs | **+0.0032pp** |
| #459 thorfinn | 7.591 | 7.467 | 7.433 | −0.034pp | ~3-4 epochs | +0.0513pp |
| #453 fern | — | — | — | — | ~3-4 epochs | +0.428pp (EP9) |
| #452 edward | — | — | best=7.903 (FINISHED) | — | done | +0.521pp |

---

## Recent closeouts

- **PR #452 edward (separate vol decoder) — PENDING CLOSE NEG.** Run finished 2026-05-03 00:01:34Z. best_val=7.903%, test=8.8681%. **Critical finding: val→test gap on vol_p (4.80% → 12.38%) reveals heavy overfitting** — decoder capacity isn't the binding constraint on test-side vol_p. Awaiting student final post; close-with-feedback prepped.
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
