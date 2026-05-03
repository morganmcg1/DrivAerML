# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 (post-#454 closeout, frieren + edward reassigned — 8 students active, 0 idle)
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

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #499 | fern | TTA mirror-y (post-hoc on frozen #387) | newly assigned, pre-first-val |
| #496 | tanjiro | Uncertainty-weighted multitask loss | newly assigned, pre-first-val |
| **#501** | **frieren** | **Anisotropic per-axis STRING freq (sigma_x/y/z, 3 arms)** | **NEW** — targets tau_y/z gap directly |
| **#500** | **edward** | **Surface curvature (mean H + Gaussian K) input features** | **NEW** — targets tau_y/z gap via geometric prior |
| #458 | nezuko | model-mlp-ratio=8 (Run 2) | EP5 slope-waiver granted; converging slow |
| #488 | alphonse | Multi-sigma log_freq STRING-sep init | tau_y/z spectral attack, pre-EP5 |
| #489 | thorfinn | Volume-points curriculum 16k→65k | pre-gate |
| #471 | askeladd | Signed-log vol target (arm-b) | arm-a vol_p 4.62% recorded; arm-b in early warmup |

(Detailed step/val trajectories last captured at the W&B snapshot earlier 2026-05-03 — see git history of this file.)

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

- **PR #454 frieren (tau_yz w=1.5) — CLOSED-NEG.** Third NEG result in tau_yz loss-weight family. Spawned anisotropic STRING attack #501.
- **PR #483 edward (surf↔vol cross-attn bridge) — CLOSED.** Did not beat SOTA. Edward reassigned to surface curvature features #500.
- **PR #482 thorfinn (TTA mirror-y) — CLOSED (EP5 gate fail).** EP6 val_abupt=10.989%; TTA must be inference-only (now revisited as fern #499).
- **PR #467 alphonse (per-axis output scaling) — CLOSED-NEG.** Confirms tau_y/tau_z gap is upstream in spectral representation. Spawned PR #488.
- **PR #142 frieren (tau_yz weight=2.0) — CLOSED-NEG.** w=1.5 follow-up was #454, also NEG.
- **PR #458 nezuko mlp6 (Run 1) — CLOSED-NEG.** mlp8 (Run 2) currently in flight.

---

## Current research focus and themes

1. **Closing the volume_pressure gap (×2.0 vs AB-UPT) — succeeding broadly across the fleet:**
   - **EMA dynamics:** PR #480 fern (cosine ramp) — val vol_p=4.69% at EP9.5; primary SOTA candidate
   - **Target transform:** PR #471 askeladd (signed-log) — vol_p=4.62% best ever on arm-a
   - **Cross-attn coupling:** PR #483 edward (surface↔volume bridge) — borderline gate pass
   - **Data curriculum:** PR #489 thorfinn (16k→65k vol-points ramp)
   - **log1p target:** PR #481 tanjiro — vol_p=4.83%

2. **Closing the tau_y/tau_z gap (×2.5 / ×2.8 vs AB-UPT) — full-fleet upstream attack:**
   - **Spectral representation, isotropic:** PR #488 alphonse (multi-sigma STRING-sep init)
   - **Spectral representation, anisotropic per-axis:** **PR #501 frieren (sigma_x/y/z 3-arm sweep — NEW)**
   - **Geometric prior (input features):** **PR #500 edward (mean H + Gaussian K curvature — NEW)**
   - **Multitask balance:** PR #496 tanjiro (uncertainty weighting)
   - **Inference-only TTA:** PR #499 fern (mirror-y on frozen SOTA)
   - **Loss weighting (confirmed inert):** PR #142, #454, #467 — three NEG results, family exhausted

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
