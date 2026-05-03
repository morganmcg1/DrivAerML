# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 (post W&B snapshot, late EP9 region — 8 students active, 0 idle)
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

No new directives in last cycle. All 3 open human issues already responded to. Still working off Issue #252 (Modded-NanoGPT-derived levers) plus organic vol_p / tau-axis attack.

---

## Currently in-flight (8 active WIP PRs on tay, ZERO idle students)

| PR | Student | Hypothesis | Run | Step / EP | val_abupt | Status |
|---|---|---|---|---|---:|---|
| **#480** | **fern** | **Cosine EMA ramp (12-ep span)** | `2u6twuu4` | 23171 / EP9.5 | **7.745%** | **PRIMARY SOTA CANDIDATE** — projected EP12 ~7.28% |
| #481 | tanjiro | log1p tau-norm v2 | `hnrpuptg` | 22436 / EP9.2 | 8.064% | Descending, vol_p strong |
| #454 | frieren | tau_yz loss weight 1.5× | `l8nu1ajz` | 20493 / EP8.4 | 8.248% | tau_yz weighting confirmed inert |
| #483 | edward | surface↔vol cross-attn bridge | `ok98szul` | 14537 / EP6.0 | 8.969% (EP5.6) | EP5 gate borderline pass (waiver) |
| #458 | nezuko | model-mlp-ratio=8 | `he54fm6v` | 12815 / EP5.3 | 9.805% (EP4.5) | EP5 slope-waiver granted (slope −1.377 pp/1k → proj 8.05%) |
| #488 | alphonse | Multi-sigma log_freq init (STRING-sep) | `ki2q9ko9` | 9863 / EP4.1 | 13.99% | Pre-gate, slow-warmup expected |
| #489 | thorfinn | Volume-points curriculum 16k→65k | (run pending) | 8617 / EP3.5 | (no val yet) | Pre-gate, first val at EP4 |
| #471 | askeladd | Signed-log vol target (arm-b) | `wlb9zv1v` | 6207 / EP2.6 | 38.36% | Arm-a EP12 done (vol_p 4.62% record); arm-b early |

---

## Latest signals (W&B snapshot — late EP7-9 region)

### Fern #480 — APPROACHING SOTA
- W&B `2u6twuu4` step 23171: **val_abupt=7.745%** (only +0.36pp above SOTA 7.3816%, descending)
- vol_p=**4.688%** (well below AB-UPT 6.08%) — vol_p BREAKTHROUGH confirmed
- tau_y=9.982%, tau_z=11.543% — best tau_y/z of any in-flight run
- Slope: **−0.076 pp/1k steps** at EP9.5 — projected EP12 val ≈ **7.28%** (BELOW SOTA 7.3816%)
- Trajectory log: EP1.1: 39.50% → EP3.4: 11.47% → EP5.6: 8.69% → EP7.8: 7.95% → EP9.0: 7.745%
- **Strongest in-flight run on `tay`. Continue to EP12; needs test metrics from best-val checkpoint for SOTA decision.**

### Tanjiro #481 — vol_p strong, tau-axis flat
- W&B `hnrpuptg` step 22436: val_abupt=8.064% (+0.68pp above SOTA, descending)
- vol_p=4.827% (below AB-UPT 6.08%) — log1p compression confirmed working on vol target
- tau_y=10.71% / tau_z=12.32% — log1p NOT closing tau gap

### Frieren #454 — confirms tau_yz weighting is inert
- W&B `l8nu1ajz` step 20493: val_abupt=8.248% (descending)
- vol_p=5.220% (below AB-UPT 6.08%)
- tau_y=10.39% / tau_z=12.07% — tau_yz weight=1.5x NOT narrowing the gap
- Combined with #142 (w=2.0 NEG), #467 (per-axis scale NEG) → loss-weighting tau is exhausted; problem is upstream

### Edward #483 — EP5 borderline gate pass
- W&B `ok98szul` EP5.6: val_abupt=8.969% (+0.069pp above 8.9% gate — slope-waiver granted)
- vol_p=5.457% (below AB-UPT 6.08%)
- Slope: −0.469 pp/1k steps — healthy descent

### Nezuko #458 — EP5 slope-waiver granted
- W&B `he54fm6v` EP4.5: val_abupt=9.805%, vol_p=6.316%
- Slope: **−1.377 pp/1k steps** (steepest in fleet) → projected EP5 ≈ 8.05% (clean gate pass)
- mlp_ratio=8 is slow-converging — expected given param-count increase

### Alphonse #488 — primary tau_y/z attack, pre-gate
- W&B `ki2q9ko9` EP4.1: val_abupt=13.99% (early, descending)
- Multi-sigma init [0.25, 0.5, 1.0, 2.0, 4.0] — slow-warmup expected; slope-waiver pre-approved
- Most important final metrics: test tau_y, test tau_z

### Thorfinn #489 — vol-curriculum, pre-first-val
- step=8617 EP3.5, no val checkpoint yet — first val at EP4 completion

### Askeladd #471 — Arm-a complete, arm-b launched
- Arm-a (`a2skzz6m`) EP12: val_abupt=7.793% (+0.41pp above SOTA), vol_p=**4.618%** (BEST VOL_P EVER, below AB-UPT)
- Signed-log transform crushes vol_p but doesn't recover headline → composition target with fern (orthogonal vol_p levers)
- Arm-b (`wlb9zv1v`) EP2.6 val_abupt=38.36% (early warmup)

### Cross-cutting observation: vol_p victory broadly, tau_y/z stagnation
- All four late-epoch runs (fern/tanjiro/frieren + askeladd arm-a) now beat AB-UPT vol_p benchmark (4.69% / 4.83% / 5.22% / 4.62% < 6.08%) — **vol_p attack succeeding broadly**
- All four runs have tau_y ~9.98-10.71% and tau_z ~11.54-12.32% — none breaking through to AB-UPT 3.65/3.63%
- Implication: log1p / loss-weight / EMA / signed-log all leave tau_y/z gap intact → confirms tau_y/z is a representational/spectral problem (PR #488 alphonse multi-sigma is the right attack)

---

## Recent closeouts

- **PR #482 thorfinn (TTA mirror-y) — CLOSED (EP5 gate fail).** EP6 val_abupt=10.989%; TTA must be inference-only.
- **PR #467 alphonse (per-axis output scaling) — CLOSED-NEG.** Confirms tau_y/tau_z gap is upstream in spectral representation. Spawned PR #488.
- **PR #142 frieren (tau_yz weight=2.0) — CLOSED-NEG.** w=1.5 follow-up in PR #454.
- **PR #458 nezuko mlp6 (Run 1) — CLOSED-NEG.** mlp8 (Run 2) launched as `he54fm6v`.

---

## Current research focus and themes

1. **Closing the volume_pressure gap (×2.0 vs AB-UPT) — succeeding broadly across the fleet:**
   - **EMA dynamics:** PR #480 fern (cosine ramp) — val vol_p=4.69% at EP9.5; primary SOTA candidate
   - **Target transform:** PR #471 askeladd (signed-log) — vol_p=4.62% best ever on arm-a
   - **Cross-attn coupling:** PR #483 edward (surface↔volume bridge) — borderline gate pass
   - **Data curriculum:** PR #489 thorfinn (16k→65k vol-points ramp)
   - **log1p target:** PR #481 tanjiro — vol_p=4.83%

2. **Closing the tau_y/tau_z gap (×2.5 / ×2.8 vs AB-UPT) — needs upstream attacks:**
   - **Spectral representation:** PR #488 alphonse (multi-sigma STRING-sep init) — most promising line
   - **Distribution shape (likely insufficient):** PR #481 tanjiro (log1p tau-norm v2)
   - **Loss weighting (confirmed inert):** PR #454 frieren — third NEG result for this lever family

3. **Capacity scaling:** PR #458 nezuko (mlp_ratio=8, slow-warmup, slope-waiver granted)

4. **Composition candidates (when winners land):**
   - fern cosine EMA + askeladd signed-log = orthogonal vol_p attacks (data + optimization)
   - thorfinn vol-curriculum + fern cosine EMA = orthogonal data + optimization
   - edward cross-attn bridge + alphonse multi-sigma STRING = orthogonal coupling + spectral
   - DO NOT compose: tanjiro log1p × askeladd signed-log (both target same vol target distribution)

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
| Learnable per-axis output head scaling (#467) | NEGATIVE — tau_y/tau_z gap is upstream |
| TTA mirror-y in training loop (#482) | NEGATIVE — TTA must be inference-only |
| tau_yz loss-weight reweighting (#142, #454, #467) | EXHAUSTED — three NEG results, problem is upstream |

---

## Potential next research directions (when slots open / Round 26)

1. **TTA post-hoc on frozen SOTA checkpoint** — apply mirror-y augmentation strictly at eval/test time on PR #387 checkpoint; near-zero compute. Assign to thorfinn after #489.
2. **Compose fern #480 (cosine EMA) + askeladd signed-log** if both win on vol_p, stack them.
3. **GradNorm / uncertainty-weighted multitask loss** — addresses coupled-multitask redistribution; principled fix vs hand-tuned weights.
4. **Surface curvature input features (H03)** — geometric prior for tau_y/tau_z high-freq content.
5. **Compose cross-attn bridge #483 with multi-sigma STRING init #488** — orthogonal coupling + spectral.
6. **Slice-conditioned FFN width** — wider FFN only in middle (volumetric) slices.
7. **EMA model-soup average** — port from yi-track if it wins there.
8. **Surface point density 2x** — confirmatory port of yi-track result.
9. **Test-time augmentation: mirror-y + reflect-z + 90-deg rotations** — extends TTA if post-hoc baseline wins.
10. **Vol-points curriculum + tanjiro log1p composition** — orthogonal data density + target transform.
11. **Wavelet/multi-resolution input encoding** — alternative tau_y/z spectral attack if multi-sigma STRING fails.
12. **Anisotropic positional encoding** — separate freq sets per spatial axis (tau_y/z gap may reflect anisotropic flow features).
