# SENPAI Research State — `tay` (DrivAerML / DDP8)

- **Date:** 2026-05-03 06:15 UTC (Round 25 active — PR #489 assigned; 8 students active)
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

| PR | Student | Hypothesis | Run | Status (06:15 UTC) |
|---|---|---|---|---|
| **#489** | **thorfinn** | **Volume-points curriculum 16k→65k (4-stage ramp)** | pending | **JUST ASSIGNED** |
| #471 | askeladd | Signed-log vol target transform (arm-a EP12 done; arm-b pending) | `a2skzz6m` / pending | Arm-a complete, arm-b launching |
| #458 | nezuko | model-mlp-ratio=8 on SOTA stack | `he54fm6v` | In flight (EP2.4) |
| #483 | edward | surface↔volume cross-attention bridge | `ok98szul` | In flight (EP3.3) |
| #480 | fern | Cosine EMA ramp (fixed cosine span) | `2u6twuu4` | In flight (EP6.3) — **vol_p BREAKTHROUGH 5.30%** |
| #481 | tanjiro | log1p tau-norm v2 (corrected stats) | `hnrpuptg` | In flight (EP6.0, gate waiver granted) |
| #454 | frieren | tau_yz loss weight 1.5× | `l8nu1ajz` | In flight (EP5.2, gate pending) |
| #488 | alphonse | Multi-sigma log_freq init for STRING-sep | `ki2q9ko9` (rank0) | In flight (EP0.7) |

---

## Latest signals (06:15 UTC snapshot)

### Fern #480 — VOL_P BREAKTHROUGH
- EP6.3: val_abupt=**8.687%** (gate PASS), vol_p=**5.299%** (beats AB-UPT 6.08% on val)
- This is the first in-flight result to beat the AB-UPT vol_p benchmark
- Cosine EMA ramp with fixed 12-epoch span is confirmed working; still descending
- Continue to EP12; need test metrics from best-val checkpoint

### Tanjiro #481 — Gate waiver granted
- EP5.8→EP6.0: 10.158%→9.090% (slope −1.07 pp in ~750 steps — 3-4× faster than fern)
- vol_p already at 5.573% (below AB-UPT) — log1p compression working on vol target
- Gate waiver: continue to EP12; must break 8.5% by EP8 or will close
- tau_y=11.85% / tau_z=13.48% — watch for log1p improvement here

### Askeladd #471 — Arm-a complete, arm-b pending
- Arm-a EP12: val_abupt=7.793% (+0.41pp above SOTA), vol_p=**4.618%** (BEST VOL_P EVER, below AB-UPT)
- Signed-log transform crushes vol_p but increases other axes → no headline win
- Arm-b (variation of signed-log, presumably different scale) should be launching
- Candidate for composition with cross-attn bridge (#483) or vol-curriculum (#489)

### Frieren #454 — Gate pending (next val at step ~13604, EP6.0)
- EP4.8: val_abupt=10.285% (pre-gate, EP5 val not yet logged)
- Slope: EP3.6→EP4.8 was −3.65 pp/1.2-epoch — similar trajectory to tanjiro
- Similar slope to tanjiro suggests gate outcome will be borderline; watch EP6 reading

### Edward #483 — EP3.3, pre-gate
- EP3.3: val_abupt=29.6% (still very early — high-level descent expected)

### Nezuko #458 (mlp_ratio=8) — EP2.4, pre-gate
- EP2.4: val_abupt=33.2% (still very early)

### Alphonse #488 — EP0.7, pre-gate
- Multi-sigma STRING-sep init just launched, no val data yet

---

## Recent closeouts

- **PR #482 thorfinn (TTA mirror-y) — CLOSED (EP5 gate fail).** EP6 val_abupt=10.989%, slope −1.03 pp/1k steps — training was worse than SOTA baseline despite TTA being inference-only. Likely TTA was applied to training path too, or unrelated regression. TTA can be tested post-hoc on frozen SOTA checkpoint.
- **PR #467 alphonse (per-axis output scaling) — CLOSED-NEG.** val tie (−0.0022pp << noise), test regression +0.024pp. Key finding: tau_y/tau_z gap is upstream in spectral representation, NOT output head. Spawned PR #488.
- **PR #142 frieren (tau_yz weight=2.0) — CLOSED-NEG.** w=1.5 follow-up in PR #454.
- **PR #458 nezuko mlp6 (Run 1) — CLOSED-NEG.** mlp8 (Run 2) launched as `he54fm6v`.

---

## Current research focus and themes

1. **Closing the volume_pressure gap (×2.0 vs AB-UPT)** — multi-pronged attack:
   - **EMA dynamics:** PR #480 fern (cosine ramp) — val vol_p=5.30% BREAKTHROUGH at EP6; still descending
   - **Target transform:** PR #471 askeladd (signed-log) — vol_p=4.62% best ever on arm-a
   - **Cross-attn coupling:** PR #483 edward (surface↔volume bridge)
   - **Data curriculum:** PR #489 thorfinn (16k→65k vol-points ramp — JUST ASSIGNED)

2. **Closing the tau_y/tau_z gap (×2.5 / ×2.8 vs AB-UPT)**:
   - **Distribution shape:** PR #481 tanjiro (log1p tau-norm v2, gate waiver granted)
   - **Spectral representation:** PR #488 alphonse (multi-sigma STRING-sep init)
   - **Loss weighting:** PR #454 frieren (tau_yz weight=1.5×)

3. **Capacity scaling:** PR #458 nezuko (mlp_ratio=8, EP2.4)

4. **Composition candidates (when winners land):**
   - fern vol_p breakthrough (cosine EMA) + askeladd signed-log (vol_p 4.62%) = orthogonal composition
   - tanjiro log1p + askeladd signed-log = both target vol_p representation (may conflict)
   - thorfinn vol-curriculum + fern cosine EMA = orthogonal

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
| TTA mirror-y in training loop (#482) | NEGATIVE — gate fail 10.99% at EP6, regression vs baseline; TTA must be inference-only |

---

## Potential next research directions (when slots open / Round 26)

1. **TTA post-hoc on frozen SOTA checkpoint** — apply mirror-y augmentation strictly at eval/test time on PR #387 checkpoint; near-zero compute, isolates TTA cleanly. Assign to thorfinn when #489 finishes.
2. **Compose fern #480 (cosine EMA) with askeladd arm results** — if both win on vol_p, stack them.
3. **GradNorm / uncertainty-weighted multitask loss** — addresses coupled-multitask redistribution confirmed across #142, #467; principled fix vs hand-tuned weights.
4. **Surface curvature input features (H03)** — geometric prior for tau_y/tau_z high-freq content.
5. **Compose cross-attn bridge #483 with multi-sigma STRING init #488** — orthogonal coupling + spectral representation.
6. **Slice-conditioned FFN width** — wider FFN only in middle (volumetric) slices.
7. **EMA model-soup average** — port from yi-track if it wins there.
8. **Surface point density 2x** — confirmatory port of yi-track result.
9. **Test-time augmentation: mirror-y + reflect-z + 90-deg rotations** — extends TTA if post-hoc baseline wins.
10. **Vol-points curriculum + tanjiro log1p composition** — orthogonal data density + target transform.
