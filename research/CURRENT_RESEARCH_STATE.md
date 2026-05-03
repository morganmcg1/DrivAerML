# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 04:35 UTC (Round 24/25 — PR #467 closed, PR #488 assigned; 8 students active)
- **Branch:** `tay`
- **Target repo:** `morganmcg1/DrivAerML`
- **W&B project:** `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`

## Current SOTA — PR #387 (alphonse feat16 RFF + QK-norm + STRING-sep), val_abupt **7.3816%** (EP11)

W&B run `wj6mn6ve`, group `alphonse-rff-sweep`. All future PRs must beat val_abupt < **7.3816%**.

| Metric | PR #387 SOTA (val EP11 / test) | AB-UPT |
|---|---:|---:|
| `abupt` (val_abupt EP11) | **7.3816%** | — |
| `abupt` (test_abupt) | **8.5936%** | — |
| `surface_pressure` (test) | 4.4377% | 3.82% |
| `wall_shear` (test) | 7.9989% | 7.29% |
| **`volume_pressure` (test)** | **12.1885%** | **6.08% (×2.0 gap — primary laggard)** |
| `tau_x` / `tau_y` / `tau_z` (test) | 6.96 / 9.11 / 10.27 | 5.35 / 3.65 / 3.63 |

---

## Latest research direction from human researcher team

No new directives in last cycle. Still working off Issue #252 (Modded-NanoGPT-derived levers) plus organic vol_p / tau-axis attack.

---

## Currently in-flight (8 active WIP PRs on tay, ZERO idle students)

| PR | Student | Hypothesis | Run | Status (04:35 UTC) |
|---|---|---|---|---|
| **#488** | **alphonse** | **Multi-sigma log_freq init for STRING-sep** | pending | **JUST ASSIGNED** |
| #471 | askeladd | Signed-log vol target transform | `a2skzz6m` | In flight |
| #458 | nezuko | model-mlp-ratio=8 on SOTA stack | `he54fm6v` | In flight |
| #483 | edward | surface↔volume cross-attention bridge | `ok98szul` | In flight |
| #480 | fern | Cosine EMA ramp (fixed cosine span) | `2u6twuu4` | In flight |
| #481 | tanjiro | log1p tau-norm v2 (corrected stats) | `hnrpuptg` | In flight |
| #482 | thorfinn | TTA mirror-y test-time augmentation | `bq1yef6h` | In flight |
| #454 | frieren | tau_yz loss weight 1.5× | `l8nu1ajz` | In flight |

---

## Recent closeouts / merges

- **PR #467 alphonse (per-axis output scaling) — CLOSED-NEG.** val tie (−0.0022pp << noise), test regression +0.024pp. Learned scales [0.842, 0.888, 0.920, 0.847] — global attenuation ~13%, NOT per-channel recalibration. Key finding: tau_y/tau_z gap is NOT in the output head; it is upstream in spectral representation. Assigns PR #488 to attack at the encoding level.
- **PR #142 frieren (tau_yz weight=2.0) — CLOSED-NEG (#454 follow-up).** Best val EP11=7.3848% (+0.0032pp above SOTA 7.3816%). Test 8.7048% (+0.111pp above SOTA test). Tau_y/z DID improve on test as predicted (-0.290pp / -0.240pp), but vol_p +0.196pp and surface_p +0.118pp regressed → coupled-multitask redistribution effect. Follow-up w=1.5 launched in PR #454.
- **PR #458 nezuko mlp6 (Run 1) — CLOSED-NEG.** Best val EP9=**7.5708%** (+0.189pp above SOTA), test 8.6824%. Vol_p improved (-0.231pp test) but surface fields slightly worse, dragging average up. Per the PR decision rule, mlp8 (Run 2) launched as `he54fm6v`.
- **PR #453 fern (cosine EMA, buggy schedule) — CLOSED.** Run finished EP10 val=7.7247%. **Cosine span bug confirmed** (`epochs=50`, `max_epochs_effective=50` → EMA only ramped to ~0.9908). Ramp hypothesis NOT actually tested. v2 with `--ema-cosine-total-epochs 12` opened as PR #480.
- **PR #470 tanjiro (log1p tau-norm v1) — CLOSED.** Stats bug — z-score stats computed in raw tau space then log1p applied → non-zero-mean / non-unit-std distribution. v2 with corrected stats recompute opened as PR #481.
- **PR #452 edward (separate vol decoder) — CLOSED NEG.** best_val=7.903%, val_vp=4.80%, test_vp=12.38% — heavy val→test overfitting on vol_p. Capacity not the binding constraint on test-side vol_p.
- **PR #451 askeladd (volume-loss-weight=3.0) — CLOSED NEG.** EP6=9.272%, no inflection.
- **PR #387 alphonse feat16 RFF — MERGED NEW SOTA** (val EP11 7.3816%, test 8.5936%).

---

## Current research focus and themes

1. **Closing the volume_pressure gap (×2.0 vs AB-UPT)** — primary laggard. Multi-pronged attack now in flight:
   - **Representation:** PR #471 (askeladd signed-log transform on vol_p targets).
   - **Coupling:** PR #483 (edward surface↔volume cross-attention bridge — directly addresses #452's "capacity isn't enough" finding by attacking from the modality-coupling side).
   - **Curriculum:** PR #484 (frieren volume-points 16k→65k ramp).

2. **Closing the tau_y/tau_z gap (×2.5 / ×2.8 vs AB-UPT)** — second laggard.
   - **Distribution shape:** PR #481 (tanjiro log1p tau-norm v2, fixed stats).
   - **Spectral representation:** PR #488 (alphonse multi-sigma log_freq init — confirmed NOT an output-head issue per PR #467 negative result).

3. **Capacity scaling on SOTA stack** — PR #458 (nezuko mlp_ratio=6) currently strongest in-flight candidate (EP7.1 projection ~7.22%).

4. **EMA dynamics** — PR #480 (fern cosine ramp, fixed cosine span). The earlier #453 buggy run already showed val_vp=4.80% (AB-UPT-class) so the schedule's right-tail high-decay value matters; with the fix in place we now actually test the ramp hypothesis.

5. **Test-time augmentation** — PR #482 (thorfinn TTA mirror-y). Free zero-training-cost lever from Kaggle/segmentation playbook never tried on this stack.

---

## Negative results catalog (do not retry on current stack)

| Lever | Outcome |
|---|---|
| Local tangent-frame INPUT features | NEGATIVE (#423) |
| Tangent-frame OUTPUT decomposition | EXHAUSTED |
| Channel-selective Huber on tau | NEGATIVE (#353) |
| Volume-loss-weight scalar rebalancing | NEGATIVE (#451) |
| Separate volume decoder (capacity) | NEGATIVE val→test overfitting (#452) |
| Muon optimizer | NEGATIVE (#299) +4.09pp |
| Sandwich-norm | NEGATIVE diverged |
| U-net skips | NEGATIVE (+0.555pp) |
| Lion β values >0.99 | All closed on older stacks |
| dropout > 0 | Default 0.0 best |
| 256d / 768d hidden | NEGATIVE on multiple stacks |
| 6L / 8L depth | NEGATIVE |
| Learnable per-axis output head scaling (#467) | NEGATIVE — test +0.024pp, uniform attenuation not recalibration; tau_y/tau_z gap is upstream |

---

## Potential next research directions (when slots open / Round 25)

1. **Volume-points curriculum (16k → 32k → 48k → 65k)** — original plan was frieren #484 but slot was reused for w=1.5 follow-up; resurrect when frieren idle.
2. **Compose tanjiro-#481 log1p (if wins) with vol-curriculum** — orthogonal targets.
3. **Compose cross-attn bridge #483 with multi-sigma STRING init #488** — orthogonal coupling + spectral representation.
4. **GradNorm / uncertainty-weighted multitask loss** — addresses the coupled-multitask redistribution effect (confirmed across #142, #467); principled fix vs hand-tuned weights.
5. **Surface curvature input features (H03)** — geometric prior for tau_y/tau_z high-freq content.
6. **Slice-conditioned FFN width** — wider FFN only in middle (volumetric) slices (informed by mlp6 negative — wider only where it helps).
7. **EMA model-soup average** (frieren #474 yi-track) — port to tay if it wins on yi.
8. **Surface point density 2x** — confirmatory port of yi-track result.
9. **Test-time augmentation: mirror-y + reflect-z + 90-deg rotations** — extends thorfinn-#482 if TTA wins.
9. **Surface point density 2x (haku branch in yi)** — confirmatory port to tay.
10. **Test-time augmentation: mirror-y + reflect-z + 90-deg rotations** — extends thorfinn-#482 if wins.
