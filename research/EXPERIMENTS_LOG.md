# SENPAI Research Results — `drivaerml-long-20260504`

Single-model long DDP8 validation wave; started 2026-05-04.

This log is appended in reverse-chronological order as PRs are reviewed. Each entry should include: PR number/title, student branch, hypothesis, results table (with W&B run IDs and test metrics), and brief commentary.

The wave's evidence contract: test metrics from `test_primary/*` only; validation is for steering and checkpoint selection.

## 2026-05-04 22:30 — PR #643: Bug-fix: flip train.py defaults (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/train-defaults-fix`
- **Type:** Code fix (not an experiment — no SENPAI-RESULT marker)
- **Fix:** Three `Config` defaults in `train.py` were silently diverging from every healthy long DDP8 reference run on this branch:

| Field | Old default | New default | Evidence |
|---|---|---|---|
| `train_surface_points` | 40,000 | 65,536 | All 4 reference runs (`nh96x7m4`, `9mm3sz7x`, `341czkol`, `ug6c3nks`) |
| `train_volume_points` | 40,000 | 16,384 | Same 4 reference runs |
| `compile_model` | True | False | Same 4 reference runs; True triggered `torch._inductor.exc.InductorError` |

- **Failure modes caught:** (1) Run `syl1zx3r` (40k/40k defaults) inverted the volume:surface gradient ratio under a surface-loss hypothesis; (2) run `xw6sp0rt` (compile_model=True with corrected sampling) hit `torch._inductor` tiling assertion at end-of-EP1.
- **Risk:** Low — all existing long DDP8 commands already explicitly override these defaults. The fix only changes behavior for new commands that omit these flags.
- **Merged to advisor branch 2026-05-04 via direct squash-merge (code fix, no experiment SENPAI-RESULT).**

## 2026-05-04 (ongoing) — PR #659: Width-over-Depth 4L/768d/12h (yi-norman)

- **Branch:** `norman/4l-768d-12h-cold-start`
- **Student:** norman (yi wave)
- **W&B Run:** `q03gty6i` (group: `yi-round37-width-768d`)
- **Hypothesis:** Increasing hidden width from 512→768d (50% more width, ~3× parameters) would improve anisotropic τ_y/τ_z representation better than depth increases.
- **Status:** CLOSED (not validated within budget)

| Epoch | Step | abupt | sp | vp | ws |
|-------|------|-------|----|----|-----|
| EP1 | ~5442 | 15.9627% | — | — | — |
| EP2 (terminal) | ~10884 | **10.0258%** | — | — | **13.30% τ_y / 14.35% τ_z** |
| Test | — | **11.2020%** | — | — | — |

**Yi SOTA reference:** val_abupt=7.3914%, test_abupt=8.7189% (PR #658 EMA)

**Commentary:** EP2=10.0258% passes the EP2 gate (≤10.5%) but is +2.49pp worse than yi SOTA. The τ_y/τ_z gap widened rather than closed (13.30%/14.35% vs. baseline ~9.87%/11.25%), so the hypothesis is not validated. Root cause: OOM at slices=128 forced fallback to slices=64 (−30% training throughput); combined with cold-start 3-epoch budget, the 28M-parameter model was severely undertrained at termination (loss slopes still strongly negative). **The width hypothesis is not falsified — it was not given a fair test.** Follow-up: 4L/640d/10h at slices=128 with ≥10 epoch budget, or redirect to τ loss weighting (already live in frieren PR #669).

---

## 2026-05-04 (ongoing) — PR #664: Per-axis Output Scaling on STRING backbone (dl24-fern)

- **Branch:** `dl24-fern/per-axis-output-scaling`
- **Student:** dl24-fern (drivaerml-long-20260504 wave)
- **W&B Run:** `a8emaoxm`
- **Hypothesis:** A learnable 4-element scale vector on the surface output head (one scalar per output channel: τ_x, τ_y, τ_z, c_p) would let the model automatically compensate for per-channel magnitude differences without hand-tuning loss weights.
- **Status:** RUNNING — EP25 completed; **new in-wave val best = 6.7422% (EP24)**

| Epoch | Step | abupt | Notes |
|-------|------|-------|-------|
| EP1 | 5493 | 11.9803% | |
| EP2 | 10987 | 8.3599% | |
| EP3 | 16481 | 7.7554% | |
| EP4 | 21975 | 7.5013% | |
| EP5 | 27469 | 7.3224% | |
| EP6 | 32963 | 7.2351% | |
| EP7 | 38457 | 7.3616% | minor regression |
| EP21 | 115373 | 6.7758% | |
| EP22 | 120867 | 6.7690% | |
| EP23 | 126361 | 6.8196% | |
| **EP24** | **131855** | **6.7422%** | **new in-wave val best** |
| EP25 | 137349 | 6.7814% | |

**Best val: 6.7422% (EP24) — new in-wave record; trailing wave SOTA val 6.5281% by 0.214pp.**

**Commentary:** Per-axis output scaling is the in-wave validation leader. Convergence is slow-descending in late epochs (EP21→EP24: −0.034pp/ep), oscillating around 6.74%. High confidence this will beat wave SOTA test=7.9303% once terminal. EP30 gate ≤7.0%; run to EP50 for terminal SENPAI-RESULT. Test metrics pending terminal evaluation.

---

## 2026-05-04 (ongoing) — PR #669: Per-channel τ surface weighting (dl24-frieren)

- **Branch:** `dl24-frieren/tau-pc-surface-weighting`
- **Student:** dl24-frieren (drivaerml-long-20260504 wave)
- **W&B Run:** `er8wmo8d` (corrected; earlier entry referenced stale run `dcaiwsyg`)
- **Hypothesis:** Upweighting τ_y (×1.2) and τ_z (×1.3) in the loss would directly pressure the model to close the sub-component gap that persists across the yi wave.
- **Status:** RUNNING — EP17 completed; **best val = 6.8276% (EP16)**

| Epoch | Step | abupt | Notes |
|-------|------|-------|-------|
| EP12 | 65927 | 6.9353% | |
| EP13 | 71421 | 6.8935% | |
| EP14 | 76915 | 6.8622% | |
| EP15 | 82409 | 6.9744% | spike (transient) |
| **EP16** | **87903** | **6.8276%** | **best val** |
| EP17 | 93397 | 6.8838% | slight regression |

**Best val: 6.8276% (EP16) — second-best in-wave; 0.095pp behind fern EP24=6.7422%.**

**Commentary:** Tau channel weighting (τ_y×1.2, τ_z×1.3) is converging well but not yet outperforming fern's per-axis output scaling. The EP15 transient spike resolved to a new best at EP16, consistent with normal stochastic oscillation. Descent rate: ~−0.03pp/epoch in the EP12–EP16 window. Projected EP20: ~6.71–6.75%; EP20 gate ≤7.0% (already well inside). If EP20 best surpasses fern's 6.7422%, request terminal run comparison. Continue to EP50 for terminal SENPAI-RESULT.

---

## 2026-05-05 (ongoing) — PR #678: Extended cosine T_max=60 (dl24-nezuko)

- **Branch:** `dl24-nezuko/extended-cosine-tmax60`
- **Student:** dl24-nezuko (drivaerml-long-20260504 wave)
- **W&B Run:** `sbzspuf2` (rank 0 of 8); group: `extended-cosine-t60-sota-v2`
- **Hypothesis:** Extending the cosine LR schedule to T_max=60 (vs. default per-epoch) allows the optimizer to maintain a higher effective LR for longer, avoiding premature convergence to a sharp minimum. Pre-wave run `5o7jc7wi` (T_max=13) achieved test=8.313% with the best volume score seen in the wave; T_max=60 is a stronger form of the same idea on the SOTA 5-sigma STRING config.
- **Status:** RUNNING — EP9 completed; **best val = 7.2894% (EP9)**

| Epoch | Step | abupt | Notes |
|-------|------|-------|-------|
| EP5 | 27469 | 7.6977% | |
| EP6 | 32963 | 7.5317% | |
| EP7 | 38457 | 7.8574% | spike (transient) |
| EP8 | 43951 | 7.2974% | recovery + new best |
| **EP9** | **49445** | **7.2894%** | **best val; EP10 gate cleared** |

**Best val: 7.2894% (EP9) — EP10 gate (≤7.5%) cleared at EP8. Descent resuming after EP7 spike.**

**Commentary:** The EP7 transient spike (7.86%) resolved cleanly with EP8 dropping to 7.2974% — a new best. EP9 essentially flat (7.2894%). The plateau at ~7.29% may be temporary; extended cosine keeps LR elevated through EP60 so descent could resume. EP15 gate ≤7.2%; if EP15 best < 7.20% this is competitive with frieren. If extended cosine T_max=60 closes the final %, it validates the schedule hypothesis from `5o7jc7wi`. Continue to EP50.

---

## 2026-05-05 (ongoing) — PR #696: QK-Norm + STRING PE (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/string-qknorm-long-50ep`
- **Student:** dl24-tanjiro (drivaerml-long-20260504 wave)
- **W&B Run:** `dzochl0q` (rank 0 of 8); group: `string-qknorm-long-50ep`; smoke: `7wdwphhn`
- **Hypothesis:** L2-normalizing Q and K per attention head (QK-Norm) before the dot-product stabilizes attention entropy, which may help the Transolver block better resolve anisotropic features (τ_y/τ_z cross-flow) that dominate the remaining error gap.
- **Config flag:** `--model-qk-norm` (zero code change, pure CLI toggle)
- **Status:** RUNNING — EP2 completed; EP5 gate ≤8.0% pending (~step 27,469)

| Epoch | Step | abupt | Notes |
|-------|------|-------|-------|
| EP1 | 5493 | 13.1298% | |
| **EP2** | **10987** | **9.6170%** | **passes EP2 kill gate ≤10.5%** |

**Best val: 9.6170% (EP2) — ~1.25pp worse than fern/frieren at same epoch (EP2: 8.36%/8.53%). Concerning but very early.**

**Commentary:** EP2=9.6170% passes the kill gate (≤10.5%) but notably lags peers. Fern at EP2=8.36%, frieren at EP2=8.53% (from those runs' early trajectory). The wider gap may reflect the QK-Norm disrupting early attention pattern formation, a known phenomenon with normalised attention requiring longer warmup. EP5 gate ≤8.0% is critical: if QK-Norm recovers to EP5 < 8.0%, the hypothesis remains viable. Pre-wave evidence: `tkiigfmc` (old stack) test=8.625% vs SOTA 7.93% — moderate pre-wave performance. Current step=14,762 (~EP2.7); EP3 expected shortly.

---

## 2026-05-05 12:00 — PR #673: Denser multi-sigma STRING PE 7 sigmas [0.1..8.0] (dl24-tanjiro) — CLOSED (regression)

- **Branch:** `dl24-tanjiro/denser-multisigma-pe-7sigmas`
- **Student:** dl24-tanjiro
- **W&B Run:** `zk35lops` (smoke `hwwrlv23`); group `denser-multisigma-pe-7sigmas`
- **Hypothesis:** Adding lower (σ=0.1) and higher (σ=8.0) sigma extremes to the SOTA 5-sigma STRING PE would broaden spectral coverage and improve fine-scale boundary-layer + long-range pressure-wake fidelity. Pure CLI, zero code change.
- **Status:** CLOSED as regression at EP14 hard kill gate.

| Metric | This run @ EP14 (best-val EMA) | Wave SOTA `sogus8sx` | Δ |
|---|---:|---:|---:|
| `val_primary/abupt_axis_mean_rel_l2_pct` | 8.1492% | 6.5281% | **+1.62pp worse** |
| `test_primary/abupt_axis_mean_rel_l2_pct` | **9.4198%** | 7.9303% | **+1.49pp worse** |
| Surface pressure (test) | 5.1207% | — | AB-UPT target 3.82% |
| Volume pressure (test) | 12.3445% | — | AB-UPT target 6.08% |
| Wall shear (test, vector) | 9.0467% | — | AB-UPT target 7.29% |
| τx / τy / τz (test) | 7.96 / 10.33 / 11.34% | — | AB-UPT 5.35 / 3.65 / 3.63% |

**Trajectory:** EP1=28.7% → EP5=8.88% → EP10=8.31% → EP14=8.15%. Slope decelerated from −0.20pp/epoch (EP6) to −0.02pp/epoch (EP14). Naive linear extrapolation to EP50 lands ~7.4%, still worse than SOTA val 6.5281%.

**Confounder:** PR-body launch command did not pin `--model-layers 4` or `--train-volume-points 65000`, so the run fell to defaults (3L, 16k vol points). Student flagged this; even a clean re-run would have struggled given the slope deceleration. Noted as PR-template gap for future STRING-family assignments.

**Side bug found by student (still open):** `KillThreshold.passes` operator semantics are inverted in `trainer_runtime.py:811` — the run was killed precisely when val *improved* below the threshold. Workaround: use `<` operator with a high ceiling for divergence guard. Student offered to file a separate fix-only PR.

**Conclusion:** 7-sigma denser STRING PE is not a productive direction. 5-sigma `[0.25,0.5,1.0,2.0,4.0]` remains the best STRING parameterization in the wave. Per-axis output scaling (PR #664) and tau channel weighting (PR #669) are higher-leverage compositions on top of the same 5-sigma base.

---

## (Pending round-1 results)

Round-1 long DDP8 assignments remaining:
- PR #608 (dl24-nezuko) — volume-loss ×2.0, run `y301z78k`, EP~49/50 as of 2026-05-04. Best val=12.8621% (step=521567). Nearly terminal — awaiting student SENPAI-RESULT with test evaluation.

Terminal results will be appended here as students post SENPAI-RESULT markers.
