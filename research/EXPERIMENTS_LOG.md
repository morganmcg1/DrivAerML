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
- **Status:** RUNNING — EP32 completed; **in-wave val best = 6.6970% (EP30)**

| Epoch | Step | abupt | surf | vol | wsh | Notes |
|-------|------|-------|------|-----|-----|-------|
| EP1 | 5493 | 11.9803% | — | — | — | |
| EP2 | 10987 | 8.3599% | — | — | — | |
| EP3 | 16481 | 7.7554% | — | — | — | |
| EP4 | 21975 | 7.5013% | — | — | — | |
| EP5 | 27469 | 7.3224% | — | — | — | |
| EP6 | 32963 | 7.2351% | — | — | — | |
| EP7 | 38457 | 7.3616% | — | — | — | minor regression |
| EP21 | 115373 | 6.7758% | — | — | — | |
| EP22 | 120867 | 6.7690% | — | — | — | |
| EP23 | 126361 | 6.8196% | — | — | — | |
| EP24 | 131855 | 6.7422% | — | — | — | prior best |
| EP25 | 137349 | 6.7814% | — | — | — | |
| EP26 | 142843 | 6.7537% | — | — | — | |
| EP27 | 148337 | 6.7648% | — | — | — | |
| EP28 | 153831 | 6.7380% | — | — | — | new best |
| EP29 | 159325 | 6.7261% | — | — | — | new best |
| **EP30** | **164819** | **6.6970%** | **4.43%** | **3.89%** | **7.57%** | **wave-best val** |
| EP31 | 170313 | 6.7848% | — | — | — | spike |
| EP32 | 175807 | 6.6983% | — | — | — | near-recovery |

**Best val: 6.6970% (EP30) — wave-best; surf=4.43%, vol=3.89%, wsh=7.57%. Trailing SOTA val 6.5281% by 0.167pp.**

**Commentary (updated 2026-05-05):** Per-axis output scaling maintains in-wave validation lead. Trend EP24→EP30 shows −0.006pp/ep net descent with typical Lion odd/even oscillation; EP31 spike (6.7848%) cleanly recovered at EP32 (6.6983%). Volume score (3.89%) is excellent — best in wave. Wall shear (7.57%) remains the bottleneck. EP40 gate ≤6.62% active; advisor also requested EP35 result and `model.surface_out_scale.data` values from student. Strong candidate for test SOTA if current descent trajectory continues. Test metrics pending terminal evaluation at EP50.

---

## 2026-05-04 (ongoing) — PR #669: Per-channel τ surface weighting (dl24-frieren)

- **Branch:** `dl24-frieren/tau-pc-surface-weighting`
- **Student:** dl24-frieren (drivaerml-long-20260504 wave)
- **W&B Run:** `er8wmo8d` (corrected; earlier entry referenced stale run `dcaiwsyg`)
- **Hypothesis:** Upweighting τ_y (×1.2) and τ_z (×1.3) in the loss would directly pressure the model to close the sub-component gap that persists across the yi wave.
- **Status:** RUNNING — EP23 completed; **best val = 6.7823% (EP22)**

| Epoch | Step | abupt | surf | vol | wsh | Notes |
|-------|------|-------|------|-----|-----|-------|
| EP12 | 65927 | 6.9353% | — | — | — | |
| EP13 | 71421 | 6.8935% | — | — | — | |
| EP14 | 76915 | 6.8622% | — | — | — | |
| EP15 | 82409 | 6.9744% | — | — | — | spike (transient) |
| EP16 | 87903 | 6.8276% | — | — | — | prior best |
| EP17 | 93397 | 6.8838% | — | — | — | slight regression |
| EP18 | 98891 | 6.8431% | — | — | — | |
| EP19 | 104385 | 6.8260% | — | — | — | new best |
| EP20 | 109879 | 6.8340% | — | — | — | |
| EP21 | 115373 | 6.7940% | — | — | — | new best |
| **EP22** | **120867** | **6.7823%** | **4.47%** | **3.94%** | **7.69%** | **best val** |
| EP23 | 126361 | 6.8310% | — | — | — | oscillation uptick |

**Best val: 6.7823% (EP22) — second-best in-wave; surf=4.47%, vol=3.94%, wsh=7.69%. 0.085pp behind fern EP30=6.6970%. Trailing SOTA val 6.5281% by 0.254pp.**

**Commentary (updated 2026-05-05):** Tau channel weighting continues descending but lags fern by ~0.085pp at comparable run depth. EP18–EP22 showed gradual improvement with −0.010pp/ep net rate; EP23 uptick to 6.8310% is consistent with Lion oscillation. Plateau pattern from EP18–EP23 is concerning — descent rate has slowed markedly from EP12–EP16 (−0.03pp/ep). EP30 gate ≤6.72% is tight: needs 0.0623pp improvement in 7 epochs from near-plateau. If gate fails, close; if gate passes, continue to EP50 terminal. The per-axis scale vs. channel-weight comparison at similar epoch counts (fern EP32 vs. frieren EP23) favors fern — both mechanisms may ultimately combine well.

---

## 2026-05-05 (ongoing) — PR #678: Extended cosine T_max=60 (dl24-nezuko)

- **Branch:** `dl24-nezuko/extended-cosine-tmax60`
- **Student:** dl24-nezuko (drivaerml-long-20260504 wave)
- **W&B Run:** `sbzspuf2` (rank 0 of 8); group: `extended-cosine-t60-sota-v2`
- **Hypothesis:** Extending the cosine LR schedule to T_max=60 (vs. default per-epoch) allows the optimizer to maintain a higher effective LR for longer, avoiding premature convergence to a sharp minimum. Pre-wave run `5o7jc7wi` (T_max=13) achieved test=8.313% with the best volume score seen in the wave; T_max=60 is a stronger form of the same idea on the SOTA 5-sigma STRING config.
- **Status:** RUNNING — EP16 completed; **best val = 6.9778% (EP16)**

| Epoch | Step | abupt | surf | vol | wsh | Notes |
|-------|------|-------|------|-----|-----|-------|
| EP5 | 27469 | 7.6977% | — | — | — | |
| EP6 | 32963 | 7.5317% | — | — | — | |
| EP7 | 38457 | 7.8574% | — | — | — | spike (transient) |
| EP8 | 43951 | 7.2974% | — | — | — | recovery + new best |
| EP9 | 49445 | 7.2894% | — | — | — | near-flat |
| EP10 | 54939 | 7.1850% | — | — | — | new best |
| EP11 | 60433 | 7.1450% | — | — | — | new best |
| EP12 | 65927 | 7.2085% | — | — | — | slight regression |
| EP13 | 71421 | 7.1019% | — | — | — | new best |
| EP14 | 76915 | 7.1540% | — | — | — | |
| EP15 | 82409 | 7.3457% | — | — | — | spike |
| **EP16** | **87903** | **6.9778%** | **4.52%** | **4.23%** | **7.88%** | **strong recovery + best val** |

**Best val: 6.9778% (EP16) — surf=4.52%, vol=4.23%, wsh=7.88%. Strong recovery from EP15 spike (7.3457%). Trailing SOTA val 6.5281% by 0.450pp.**

**Commentary (updated 2026-05-05):** Extended cosine T_max=60 shows healthy descent with periodic spikes at EP7 and EP15, each cleanly resolved by the next epoch. The EP16 result of 6.9778% is a new best and represents a significant improvement from EP9=7.2894% (+0.312pp in 7 epochs). EP20 gate ≤6.95% requires 0.028pp improvement from EP16 — very achievable given current trajectory. The key question for this run is EP30–50: does the slower LR decay enable continued descent where standard cosine would flatten? The strong EP16 recovery suggests the mechanism is working, but ~0.48pp gap to SOTA val means extended cosine alone may not be sufficient. Volume score at 4.23% is reasonable but above fern (3.89%) and frieren (3.94%). Continue to EP50; EP20 gate is the next checkpoint.

---

## 2026-05-05 (ongoing) — PR #696: QK-Norm + STRING PE (dl24-tanjiro)

- **Branch:** `dl24-tanjiro/string-qknorm-long-50ep`
- **Student:** dl24-tanjiro (drivaerml-long-20260504 wave)
- **W&B Run:** `dzochl0q` (rank 0 of 8); group: `string-qknorm-long-50ep`; smoke: `7wdwphhn`
- **Hypothesis:** L2-normalizing Q and K per attention head (QK-Norm) before the dot-product stabilizes attention entropy, which may help the Transolver block better resolve anisotropic features (τ_y/τ_z cross-flow) that dominate the remaining error gap.
- **Config flag:** `--model-qk-norm` (zero code change, pure CLI toggle)
- **Status:** RUNNING — EP9 completed; **best val = 7.7776% (EP9)**

| Epoch | Step | abupt | surf | vol | wsh | Notes |
|-------|------|-------|------|-----|-----|-------|
| EP1 | 5493 | 13.1298% | — | — | — | |
| EP2 | 10987 | 9.6170% | — | — | — | passes EP2 kill gate ≤10.5% |
| EP3 | 16481 | 9.0533% | — | — | — | |
| EP4 | 21975 | 8.6432% | — | — | — | |
| EP5 | 27469 | 8.3178% | — | — | — | passes EP5 gate ≤8.0%? — marginal |
| EP6 | 32963 | 8.1985% | — | — | — | |
| EP7 | 38457 | 8.2742% | — | — | — | minor spike |
| EP8 | 43951 | 8.0730% | — | — | — | spike |
| **EP9** | **49445** | **7.7776%** | **5.13%** | **4.73%** | **8.74%** | **new best; strong recovery** |

**Best val: 7.7776% (EP9) — surf=5.13%, vol=4.73%, wsh=8.74%. Strong recovery from EP8 spike (8.0730%). Lagging fern/frieren by ~1.08pp at similar epoch range. EP10 gate ≤7.6% active (0.178pp gap).**

**Commentary (updated 2026-05-05):** QK-Norm shows a characteristic slow-start pattern: EP1–EP8 descent rate is −0.43pp/ep vs. fern/frieren at ~−0.55pp/ep. EP8 spike and EP9 recovery (+0.295pp) is the best single-epoch improvement seen in this run. Pre-wave evidence (`tkiigfmc`, test=8.625% on old stack) suggested QK-Norm could work but with a slower warmup; this is consistent. EP10 gate ≤7.6% requires 0.178pp improvement from EP9's 7.7776% — achievable given EP8→EP9 drop. The `tanjiro-heads-sweep` unauthorized W&B group was flagged for explanation by advisor (comment posted 2026-05-05). Key assessment: does EP20–50 show accelerated descent to close the 1.08pp gap vs. fern? If QK-Norm composes with STRING multi-sigma PE and tau weighting, this could be a valuable building block even if it doesn't win alone. Continue to EP10 gate, then reassess.

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
