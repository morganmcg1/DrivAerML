# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-02 20:10 UTC (Round 22 in flight)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #358 (thorfinn STRING-sep + QK-norm), val_abupt **7.3921%** (EP11)

PR #358 ran past EP10 to a final best of EP11 val=7.3921% (corrected from initial EP10 reading 7.4648%). All future PRs must beat val_abupt < **7.3921%**.

W&B run `tkiigfmc`, group `thorfinn-string-qknorm-r19`.

| Metric | PR #358 SOTA (val EP11 / test) | AB-UPT |
|---|---:|---:|
| `abupt` (val_abupt EP11) | **7.3921%** | — |
| `abupt` (test_abupt) | **8.625%** | — |
| `surface_pressure` (test) | 4.462% | 3.82 |
| `wall_shear` (test) | 7.965% | 7.29 |
| `volume_pressure` (test) | **12.434%** | 6.08 (×2.05 gap — primary laggard) |

---

## Latest research direction from human researcher team

No new directives in the last cycle. Last actionable directive: Issue #252 (Modded-NanoGPT-derived levers). Most variants tried; QK-norm produced the only positive (PR #358).

---

## Currently in-flight (8 active WIP PRs on tay, zero idle students)

| PR | Student | Hypothesis | Group |
|---|---|---|---|
| #387 | alphonse | STRING-sep num_features sweep (16/32/64) on QK-norm base | `alphonse-string-sep-feat-qknorm` |
| #422 | tanjiro | Multi-scale STRING-sep (8/32/128 features) on QK-norm | `tay-tanjiro-multiscale-rff` |
| #451 | askeladd | volume-loss-weight=3.0 on STRING+QK SOTA | `askeladd-vol-w3-r22` |
| #452 | edward | Separate 2-layer MLP volume decoder head | `edward-vol-decoder-r22` |
| #453 | fern | Cosine EMA ramp 0.99→0.9999 on STRING+QK+Lion SOTA | `fern-ema-ramp-r22` |
| #454 | frieren | Per-axis tau_y/tau_z loss weights ×2 on SOTA | `frieren-tau-axis-r22` |
| #458 | nezuko | model-mlp-ratio 6 vs 8 sweep on SOTA stack | `nezuko-mlp-ratio-r22` |
| #459 | thorfinn | Lion β2 reactive sweep (0.95/0.97) on SOTA | `thorfinn-lion-beta2-r22` |

### Latest in-flight metrics (2026-05-02 20:10 UTC)

- **#422 tanjiro** — EP3 val_abupt **12.152%** vs baseline 13.130% (−0.98 pp). Watch-points cleared (surface_pressure recovered, per-head grads 1:6:47 stable). EP5 convergence criterion: ≤9.3% supports clearing 7.3921% bar by EP12-13. Currently mid-EP4.
- **#387 alphonse Arm A (feat16)** — EP6 val_abupt 8.73%, slope flattening at −0.13pp/epoch. Projection EP11 ≈ 8.08%, unlikely to clear 7.3921% bar. Arm B (feat32 control) launches ~22:50Z, Arm C (feat64) at ~04:50Z 2026-05-03.
- **#451-#454, #458-#459 (Round 22)** — All assigned 19:15-19:29Z, pods spinning up. EP1 metrics expected ~26 min after each launch.

---

## Recent closeouts

- **PR #423 thorfinn (local tangent-frame input features) — CLOSED NEGATIVE.** EP5 val 9.342% vs baseline 8.866% (+0.48pp). Uniform ~5% relative degradation across ALL targets (including SP and vol_p which can't benefit from tangent features) — pure optimization tax, no signal. Useful conclusion: "missing local-frame geometric context" RULED OUT as tau_y/z bottleneck on slice-attention architectures. FIGConvNet local-frame win does not transfer.
- **PR #365 nezuko (layers=5 + STRING-sep on stale base) — CLOSED ABANDONED.** Result 7.5250% no longer clears bar (now 7.3921%); branch had merge conflicts; student unresponsive to 4 rebase requests. Hypothesis re-tested clean by senku #435 on full SOTA stack.

## Recent merged

- **PR #358 thorfinn STRING-sep + QK-norm — MERGED NEW SOTA** (val EP11 7.3921%, test 8.625%).
- **PR #311 edward STRING-sep PE — MERGED prior SOTA** (val 7.546%, test 8.771%).

---

## Current research focus and themes

1. **Closing the volume_pressure gap (×2.05 vs AB-UPT)** — primary laggard. Multiple angles in flight: PR #451 (loss reweight), PR #452 (separate decoder head). Both pure single-variable deltas on the SOTA stack — first time vol-targeted experiments tested on STRING-sep+QK-norm base.
2. **Closing the tau_y/tau_z gap (×2.53 / ×2.88)** — second laggard. PR #454 (frieren per-axis weights), PR #422 (multi-scale STRING-sep). Tangent-frame input features ruled out (#423).
3. **Capacity scaling on SOTA stack** — never done in single-variable form on STRING-sep+QK-norm. PR #458 (mlp_ratio 6/8), PR #387 (PE feature count). 5L escalation is being tested on the `yi` branch (PR #435) rather than `tay`.
4. **Optimizer dynamics** — PR #459 (Lion β2=0.95/0.97 reactive sweep). β2<0.99 has never been tested on this dataset.
5. **EMA scheduling** — PR #453 (cosine 0.99→0.9999 ramp) replaces fixed 0.999.

---

## Largest remaining gaps to AB-UPT (test metrics from PR #358)

1. **volume_pressure** ×2.05 (12.434% vs 6.08%) — askeladd #451, edward #452
2. **tau_z** ×2.88, **tau_y** ×2.53 — frieren #454, tanjiro #422
3. **wall_shear** ×1.09 (7.965% vs 7.29%) — partly addressed by tau-axis weights #454
4. **surface_pressure** ×1.17 (4.462% vs 3.82%) — no in-flight experiment specifically; would benefit from any general-stack improvement

---

## Negative results catalog (key lessons, do not retry on current stack)

| Lever | Outcome |
|---|---|
| Local tangent-frame INPUT features | NEGATIVE (#423) — uniform regression, no tau-specific gain |
| Tangent-frame OUTPUT decomposition | EXHAUSTED across many stacks (#11 merged old; #41/#121/#199/#218/#227/#312/#337/#344/#349/#362/#369 all closed) |
| Channel-selective Huber on tau | NEGATIVE (#353) — gradient clipping + reweighting bug |
| Muon optimizer | NEGATIVE (#299) +4.09pp |
| Sandwich-norm | NEGATIVE diverged |
| U-net skips | NEGATIVE (+0.555pp) |
| Lion β values >0.99 | All closed on older stacks |
| dropout > 0 | Default 0.0 best on current code (PR #242 was on different stack) |
| 256d / 768d hidden | NEGATIVE on multiple stacks |
| 6L / 8L depth | NEGATIVE (5L still pending senku #435) |

---

## Potential next research directions (when slots open)

1. **Loss-side log-amplitude on volume_pressure only** — `signed_log` transform of vol_p targets if PR #366 Huber and PR #451 reweighting both fail. Heavy-tail compression specific to volumetric pressure dynamic range.
2. **Per-axis output normalization statistics recompute** — verify the per-target std used in loss normalization is fresh and per-axis correct (could expose a hidden imbalance).
3. **Slice-conditioned FFN width** — wider FFN only in middle slices (where volumetric tokens live).
4. **Curriculum on volume points** — start training with fewer volume points, ramp to 65k. May ease optimizer's early focus on under-converged surface signal.
5. **Activation function sweep on SOTA stack** — SwiGLU was negative on older stack; try GeGLU or ReGLU as a clean single-variable.
6. **Cross-attention bridge between surface and volume tokens** — explicit, vs current implicit shared attention.
7. **Test-time augmentation: rotation/reflection symmetries of car geometry** — easy win if not already exploited.
8. **Log-space coordinate compression** — in-progress on yi (#449) — could be backported if positive there.
