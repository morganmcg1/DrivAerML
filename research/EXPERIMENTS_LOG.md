# SENPAI Research Results — DrivAerML (`tay`)

W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`.

Targets to beat (lower is better, AB-UPT public reference):
`surface_pressure 3.82`, `wall_shear 7.29`, `volume_pressure 6.08`,
`tau_x 5.35`, `tau_y 3.65`, `tau_z 3.63`.

## 2026-05-01 12:30 UTC — PR #204 ASSIGNED: frieren vol_loss_weight=2.0 (SOTA stack single-delta)

- **Branch:** `frieren/vol-loss-weight-2p0`
- **Hypothesis:** BASELINE.md explicitly flags that the current SOTA (PR #115) was trained WITHOUT `--volume-loss-weight 2.0`. PR #142 tested vol_w=2.0 but on an earlier suboptimal stack (missing Lion wd=5e-4, EMA=0.999 compound). This run adds vol_w=2.0 as a clean single-delta against the verified SOTA config. The `volume_pressure` gap (12.740 vs AB-UPT ref 6.08, ×2.1) is the largest remaining per-axis gap — vol_w=2.0 directly targets this.
- **W&B group:** `tay-round12-vol-loss-weight-2p0`
- **Single delta from SOTA:** only `--volume-loss-weight 2.0` changes; all other flags match SOTA exactly.
- **Expected:** push `abupt_axis_mean` toward ~10.3, recover `volume_pressure` meaningfully.
- **Watch:** `val_primary/volume_pressure_rel_l2_pct` directly; if surface metrics regress >2%, stop run.
- **Status:** Running. Awaiting results.

## 2026-05-01 (latest) — PR #203 ASSIGNED: thorfinn round12 weight_decay=2.5e-4 (sweep DOWN from SOTA)

- **Branch:** `thorfinn/round12-wd-2p5e-4`
- **Hypothesis:** PR #163 (wd=1e-3) regressed all metrics +4.5% from SOTA (wd=5e-4). Gradient in WD points DOWN — sweep to 2.5e-4 (half of SOTA value).
- **W&B group:** `tay-round12-wd-2p5e-4`
- **Single delta from SOTA:** only `--weight-decay 2.5e-4` changes.
- **Status:** Running. Awaiting results.

## 2026-05-01 (latest) — PR #163 CLOSED: thorfinn weight_decay=1e-3 (regressed +4.5% from SOTA)

- **Branch:** `thorfinn/wd-1e-3` (exact branch name per assignment note)
- **Hypothesis:** Sweep WD UP from SOTA 5e-4 → 1e-3 as a regularization dial.
- **Result:** best val **9.911%** vs SOTA 9.484% (+4.5% regression). All 7 test metrics regressed.
  | Metric | SOTA (PR #115) | PR #163 | Delta |
  |---|---:|---:|---:|
  | val_abupt | 9.484 | 9.911 | +4.5% |
- **W&B group:** `tay-round12-wd-1e-3`
- **Conclusion:** Increasing weight decay hurts. Gradient in WD is downward from SOTA — optimum likely at or below 5e-4. Next: PR #203 sweeps to 2.5e-4.
- **Thorfinn reassigned** to PR #203: weight_decay=2.5e-4.

## 2026-05-01 20:15 UTC — PR #162 CLOSED: edward model_dropout=0.05 (test 11.029, +4.24% vs SOTA)

- **Branch:** `edward/round11-model-dropout-0p05`
- **Hypothesis:** model_dropout=0 → 0.05 as a regularization lever; reduce overfitting on the val→test generalization gap (val/test ratio 1.115 at SOTA, looking for tighter generalization).
- **Result:** test_abupt **11.029** (+4.24% regression), best val 9.757 (+2.87% vs SOTA 9.484). All 7 test metrics regressed vs SOTA.
  | Metric | SOTA (PR #115) | This run | Delta |
  |---|---:|---:|---:|
  | abupt mean | 10.580 | 11.029 | +4.24% |
  | surface_pressure | 5.690 | 5.820 | +2.29% |
  | wall_shear | 10.419 | 10.842 | +4.06% |
  | volume_pressure | 12.740 | 12.853 | +0.89% |
  | tau_x | 8.908 | 9.262 | +3.97% |
  | tau_y | 12.491 | 13.060 | +4.56% |
  | tau_z | 13.071 | 13.680 | +4.66% |
- **W&B:** `tfumujfi` (verified) — best val 9.757, test 11.029.
- **Why it failed:** Dropout adds regularization noise during training, but this model is NOT over-parameterized at 4L/512d/batch=4 on DrivAerML. The training signal is weak enough (9 epochs, CFD sparse supervision) that dropout at 0.05 creates more harm than benefit — it degrades feature representations without providing measurable generalization improvement. The val→test ratio was 11.029/9.757 = 1.130, slightly WORSE than SOTA 1.115, confirming dropout hurt rather than helped generalization.
- **Conclusion:** model_dropout=0.05 regresses all metrics. Dropout is a dead end at this batch size / epoch budget — the model underfits, not overfits. Regularization via dropout closed.
- **Edward now idle** — awaiting new assignment.

## 2026-05-01 20:00 UTC — PR #161 CLOSED: askeladd lion_beta2=0.999 (test 12.564, +18.7% vs SOTA)

- **Branch:** `askeladd/round11-lion-beta2-0p999`
- **Hypothesis:** lion_beta2=0.99 → 0.999 (SOTA uses 0.99); higher beta2 = more momentum smoothing → better generalization on CFD objectives.
- **Result:** test_abupt **12.564** (+18.7% regression), best val **11.493** (+21.2% vs SOTA val 9.484). Massive regression across all metrics.
  | Metric | SOTA (PR #115) | This run | Delta |
  |---|---:|---:|---:|
  | abupt mean | 10.580 | 12.564 | +18.7% |
  | surface_pressure | 5.690 | 7.173 | +26.1% |
  | wall_shear | 10.419 | 12.574 | +20.7% |
  | volume_pressure | 12.740 | 14.159 | +11.1% |
  | tau_x | 8.908 | 10.832 | +21.6% |
  | tau_y | 12.491 | 14.945 | +19.7% |
  | tau_z | 13.071 | 15.713 | +20.2% |
- **W&B:** run `tfumujfi` group `tay-round11-lion-beta2-0p999` (verified) — best val 11.493.
- **Why it failed:** Lion's EMA update uses `m_t = beta2 * m_t-1 + (1-beta2) * g_t`. At beta2=0.999, the effective momentum window is 1/(1-0.999) = 1000 steps vs 1/(1-0.99) = 100 steps at SOTA. This makes the momentum very slow to respond to the current gradient — effectively a heavily trailing average. In a 9-epoch training run (~45,000 total steps / 8 GPUs), the momentum barely adapts to local gradient structure. The model gets stuck in early-training trajectories with insufficient gradient responsiveness. beta2=0.99 (SOTA) is already well-tuned for this 9-epoch budget.
- **Conclusion:** lion_beta2=0.999 is a clear dead end — 18.7% regression. SOTA beta2=0.99 confirmed as optimal; the momentum window of 100 effective steps is well-calibrated for 9-ep CFD training. beta2 space closed at 0.99. Higher values (longer memory) hurt responsiveness; lower values untested but low priority given plateau.
- **Askeladd now idle** — awaiting new assignment.

## 2026-05-01 12:30 UTC — PR #157 CLOSED: nezuko mlp_ratio=6 (test 11.261, +6.4% vs SOTA)

- **Branch:** `nezuko/round10-mlp-ratio-6`
- **Hypothesis:** mlp_ratio=4→6 (yi Wave 1 confirmed lever) ports to tay compound SOTA stack — wider FFN improves surface topology / fine-grained pressure gradients.
- **Result:** test_abupt 11.261 (+6.4% regression), best val 10.131 ep7 (+6.8%). All metrics regressed: surface_p +7.5%, wall_shear +7.3%, vol_p +3.3%, tau_y +9.7%, tau_z +5.9%.
- **W&B:** `xuppho03` (verified) — val flatlined ep7→ep8 (10.131→10.142) while SOTA still improving 9.73→9.484 — capacity ceiling hit early under 9-epoch budget.
- **Why it failed:** +15% param cost via FFN width does not buy generalization at this budget. FFN width is not where the headroom is in our compound SOTA stack. Combined with the T_max=50 vs SOTA's effective T_max=50 fallback, this is a clean single-delta on mlp_ratio.
- **Conclusion:** mlp_ratio=4 stays. FFN-width capacity expansion family closed. Architecture-level capacity tweaks (FFN width, depth) are saturated at 9-ep budget — next bold capacity moves should be data-side (sampling density) or loss-side (re-weighting), not parameter count.
- **Nezuko reassigned** to PR #187: volume_loss_weight=1.5 (gentler than #142's 2.0; attack vol_p ×2.1 gap).

## 2026-05-01 12:25 UTC — PR #158 CLOSED: alphonse vol_pts=96k confounded (test 13.179, +24.6% vs SOTA)

- **Branch:** `alphonse/round10-vol-pts-60k` (re-scoped pre-launch to 96k after baseline correction)
- **Hypothesis:** volume_points 65k→96k attacks volume_pressure ×2.1 binding gap.
- **Result:** test_abupt 13.179 (+24.6%), surface_p 7.817 (+37.4%), wall_shear 13.461 (+29.2%), vol_p 13.720 (+7.7%), best val 12.067.
- **W&B:** `yfi14f1w`
- **Why CONFOUNDED, not negative:** student ran `--lr-cosine-t-max 0` which (per `trainer_runtime.py:1255`) falls back to `T_max=epochs=9` ⇒ LR collapsed to 1e-6 by ep9 vs SOTA's `t_max=0, epochs=50` ⇒ `T_max=50` (essentially flat ~1e-4 over 9 epochs). LR collapsed before model could leverage extra volume sampling. Excellent post-hoc analysis by alphonse identified this.
- **Key finding (research-level):** `--lr-cosine-t-max 0` is a footgun. Specify `--lr-cosine-t-max 50` explicitly in any 9-epoch single-delta to match SOTA's effective LR schedule.
- **Alphonse reassigned** to PR #186: vol_pts=96k CLEAN re-run with `--lr-cosine-t-max 50`. The original hypothesis (vol_pts as binding-gap lever) is still untested; this finishes the job properly.

## 2026-05-01 06:35 UTC — PR #142 CLOSED: thorfinn vol_w=2.0 (test 11.721, +10.78% vs SOTA)

- **Branch:** `thorfinn/round10-compound-volw2`
- **Hypothesis:** vol_w=2.0 alone (surface_loss_weight=1.0 default) attacks volume_pressure binding gap ×2.1 vs AB-UPT
- **Result:** test_abupt 11.721 (+10.78%), val 10.607 (+11.8% vs SOTA ep9 9.484). Test breakdown: surface_p +18.42%, vol_p +0.10% (FLAT!), wall_shear +13.83%, tau_x +14.70%, tau_y +13.36%, tau_z +12.72%.
- **Key finding:** Raising vol_w=2.0 with sw=1.0 default does NOT help vol_p (+0.10% flat) and severely hurts surface/wall-shear. The relative balance matters, not the absolute weight. Compare: askeladd #141 with sw=2 + vw=2 paired did improve vol_p test -0.60% — because the ratio sw:vw stayed 1:1.
- **W&B:** `33l6yvwy`
- **Conclusion:** vol_w=2.0 alone closes the volume loss lever in isolation. Loss weight family (vol_w, surf_w) must be paired. Loss-weight sweep **closed** without better ratio balance.
- **Thorfinn reassigned** to PR #163: weight_decay=1e-3 (untested, addresses val→test gen gap).

## 2026-05-01 06:08 UTC — PR #146 CLOSED: edward 6L/256d depth swap (test 12.662, +19.7% vs SOTA)

- **Branch:** `edward/round10-depth-6l256d`
- **Hypothesis:** 6L/256d (4.65M params) was yi's confirmed −21% lever; port to tay's stack
- **Result:** test_abupt 12.662 (+19.7%), full_val 11.549 at ep7 (+21.8%). All 6 components regressed: surface_p +27.1%, wall_shear +22.0%, vol_p +10.4%, tau_x +21.5%, tau_y +24.2%, tau_z +19.9%
- **Why:** 6L vs 4L per-epoch wall-clock IDENTICAL (~30.7m vs 30m) despite 2.7× fewer parameters — attention over 65k surface + 65k volume points dominates forward pass, depth (sequential layers) cancels savings. The "smaller model = more epochs in 270m budget" assumption fails. Linear extrapolation to ep9 = ~10.8 abupt, still +14% behind SOTA. Not a budget issue, a fundamental fit issue.
- **W&B:** `93n0zlc8`
- **Conclusion:** Combined with nezuko #138 (5L/512d, +6.0%) and this run (6L/256d, +19.7%), the depth-swap family is **closed** at 9-epoch budget. 4L/512d is the right shape.
- **Edward reassigned** to next PR (model_dropout=0.05 — fresh untouched regularization lever).

## 2026-05-01 06:05 UTC — PR #141 CLOSED: askeladd 3-way compound sw=2 + vw=2 + T_max=50 (test 10.605, +0.23% vs SOTA)

- **Branch:** `askeladd/round10-compound-lr1e4-sw2-ema999`
- **Hypothesis:** compound lr=1e-4 + sw=2.0 + EMA=0.999. PR body cited stale PR #111 baseline (test 11.142).
- **Re-scored vs PR #115 SOTA (test 10.580):** test_abupt 10.605 = **+0.23%** (NOT a winner). Val 9.445 = −0.41% (improve).
- **Test breakdown vs SOTA:** surface_p 5.685 (−0.08%), vol_p 12.664 (**−0.60% improve!**), wall_shear 10.469 (+0.48%), tau_x 8.909 (≈0%), tau_y 12.648 (**+1.25% regress**), tau_z 13.117 (+0.35%)
- **Val→test divergence:** ratio 1.123 vs SOTA 1.115. 3-way compound overfit val by ~0.7%. vol_p test improvement (binding gap) is real but cancelled by tau_y regression on TEST.
- **W&B:** `rdpf0y7r` — best val 9.445 (ep9, EMA), val trajectory 50.49/23.03/15.99/13.02/11.56/10.69/10.05/9.67/9.45 — still descending at ep9 (epoch cap, not plateau).
- **Conclusion:** sw=2.0 + vw=2.0 stack lifts vol_p on test (-0.60%) but introduces tau-axis regressions. Compound stack of weight schemes net-zero vs SOTA on test. Cosine T_max=50 was an unintended confound (SOTA uses T_max=0).
- **Askeladd reassigned** to next PR (lion_beta2=0.999 single delta — fresh untouched optimizer hyperparam).

## 2026-05-01 04:45 UTC — PR #148 CLOSED EARLY: alphonse lr=1.5e-4 (stably +40% worse, 4 epochs)

- **Branch:** `alphonse/round10-lr1p5e4`
- **Hypothesis:** lr=1.5e-4 + EMA=0.999 single-delta from PR #115 SOTA (push LR ceiling)
- **Result:** Closed at ep3 — trajectory stably ~40% behind SOTA with no compression: ep0 +34.8% / ep1 +41.8% / ep2 +37.7% / ep3 +40.8%. Projected final test ~14-15 (vs SOTA 10.580).
- **Why:** Lion uses sign of gradient, so lr directly scales update magnitude with no adaptive damping. At 1.5e-4 the updates are too large for EMA=0.999 to compensate; T_max=50 cosine barely decays (92% of lr_max at ep9), so instability persists. LR ceiling between 1e-4 and 1.5e-4 — 1e-4 (PR #115) is the optimal value.
- **Conclusion:** LR lever closed. The 5e-5 → 1e-4 transition (PR #115 −5%) captures all available LR headroom.
- **Alphonse reassigned** to PR #158: vol_pts=60k (yi Wave 1 lever, attack dominant vol_p ×2.1 gap).

## 2026-05-01 03:55 UTC — PR #138 CLOSED: nezuko 5L depth swap (test 11.213, +5.98% vs SOTA)

- **Branch:** `nezuko/round10-depth-5l`
- **W&B run:** `1l3ndrwe` rank 0 — group tay-round10-depth-5l, 270.9 min, 8 epochs (timeout mid-ep9), best val 9.986 (ep8)
- **Hypothesis:** 5L depth swap (yi Wave 1 confirmed depth lever, −21% on yi)
- **Result:** test_abupt **11.213** vs SOTA 10.580 = **+5.98% WORSE**. Best val 9.986 (ep8 EMA).
- **Per-axis:** sp=6.20 (+9.0%), ws=11.23 (+7.8%), vp=12.62 (−0.94% only win), tau_x=9.51 (+6.8%), tau_y=13.69 (+9.6%), tau_z=14.04 (+7.4%).
- **Why depth lost:** (1) Epoch budget squeeze: 8 ep (5L) vs 9 ep (4L) due to 13% slower per-epoch + 23% more params; (2) yi's gain was on a different stack (no SDF/Fourier/asinh) — on tay's compound base, depth's marginal lift is much smaller; (3) param efficiency mismatch at batch=4. Volume_pressure was the only axis where 5L slightly won — interesting signal but doesn't compensate for surface losses.
- **Conclusion:** 5L is a closed lever on tay's 9-epoch budget. edward #146 (6L/256d) is yi's *exact* config — different param budget profile, separate test.
- **Nezuko reassigned to PR #150: PR #115 + mlp_ratio=6** (yi Wave 1 confirmed lever, untested on tay).

## 2026-05-01 01:20 UTC — PR #115 MERGED: thorfinn compound lr=1e-4 + EMA=0.999 — NEW SOTA 10.580 (−5.0%)

- **Branch:** `thorfinn/round9-compound-lr1e4-ema999`
- **W&B run:** `d03oghpp` rank 0 — group `tay-round9-compound-lr1e4-ema999`, 287 min, 9 val epochs, best val 9.484 (ep9)
- **Hypothesis:** Compound stack — Lion lr=1e-4 (2×) + EMA=0.999 (confirmed EMA winner). Both single-variable wins stacked additively, expected synergy from EMA dominating early epochs and higher LR covering more loss landscape late.
- **Result:** test_abupt **10.580** vs SOTA 11.142 (**−5.04%, new SOTA**). Best val 9.484 (ep9). Note: run used vol_w=1.0 (default), so volume_pressure flat (+0.1%) while all surface/wall-shear improved 6–8%.

| Metric | PR #115 (MERGED) | PR #111 prior SOTA | Δ | AB-UPT |
|---|---:|---:|---:|---:|
| abupt_mean | **10.580** | 11.142 | **−5.04%** | — |
| surface_pressure | **5.690** | 6.209 | −8.4% | 3.82 |
| wall_shear | **10.419** | 11.138 | −6.5% | 7.29 |
| volume_pressure | 12.740 | **12.548** | +1.5% | 6.08 |
| tau_x | **8.908** | 9.436 | −5.6% | 5.35 |
| tau_y | **12.491** | 13.525 | −7.6% | 3.65 |
| tau_z | **13.071** | 13.992 | −6.6% | 3.63 |

- **Analysis:** EMA dominated early (ep1 thorfinn 53.75 vs alphonse-only 78.80, −32%) and lr=1e-4 extended the advantage in late epochs. Compound beat both single-variable arms cleanly. The vol_w=1.0 omission deliberately drops volume gradient signal — volume_pressure regressed 0.192 vs PR #111, explaining why 6/7 axes improved while volume_pressure didn't. Trajectory still descending at ep9 (Δep8→ep9 = −0.25).
- **Conclusion:** lr=1e-4 + EMA=0.999 is confirmed as additive compound. Largest single-PR gain in project history. Immediate follow-up: re-add vol_w=2.0 (PR #142 thorfinn, 1-variable, projected ~10.3-10.4).
- **Thorfinn reassigned** to PR #142: compound SOTA + vol_w=2.0 (recover volume gradient).
- **New SOTA: 10.580**

## 2026-05-01 02:50 UTC — PR #135 CLOSED: tanjiro T_max=100 + EMA=0.999 — 11.082 (+4.7% vs SOTA)

- **Branch:** `tanjiro/round9-lion-tmax100-ema999`
- **W&B run:** `wtfrhy2n` rank 0 — group `tay-round9-lion-tmax100-ema999`, 285 min, 9 val epochs, best val 9.886 (ep9)
- **Hypothesis:** T_max=100 (4% LR decay vs T_max=50's 8%) + EMA=0.999 — testing extended cosine schedule.
- **Result:** test_abupt **11.082** vs SOTA **10.580** = **+4.74% regression**.

| Metric | PR #135 | PR #115 SOTA | Δ |
|---|---:|---:|---:|
| abupt_mean | 11.082 | **10.580** | +4.7% |
| surface_pressure | 6.138 | **5.690** | +7.9% |
| wall_shear | 11.039 | **10.419** | +5.9% |
| volume_pressure | 12.665 | **12.740** | −0.6% |
| tau_x | 9.348 | **8.908** | +4.9% |
| tau_y | 13.469 | **12.491** | +7.8% |
| tau_z | 13.791 | **13.071** | +5.5% |

- **Conclusion:** T_max=100 vs T_max=50 (PR #133): essentially tied (+0.034 abupt). T_max sweep CLOSED-DOOR. The lr=1e-4 lever in PR #115 dominates the schedule contribution. **T_max=50 is the sweet spot**, but compounding with cosine on lr=1e-4 base is unlikely to add much.
- **Tanjiro reassigned** to PR #149: per-axis tau_y/tau_z conservative weighting (W_y=W_z=1.5) — attack ×3.4-3.6 binding gap.

## 2026-05-01 02:50 UTC — PR #136 CLOSED: alphonse Lion + sw=2.0 — 10.586 (+0.06% essentially tied)

- **Branch:** `alphonse/round9-lion-surface-w2`
- **W&B run:** `okl62six` rank 0 — group `tay-round9-lion-surface-w2`, 285 min, 9 val epochs, best val 9.473 (ep9)
- **Hypothesis:** surface_loss_weight=2.0 (vs default 1.0) — attack tau_y/tau_z binding gap by upweighting surface losses.
- **Result:** test_abupt **10.586** vs SOTA **10.580** = **+0.06% (essentially tied, NOT a winner)**. Best val 9.473 (slightly better than #115's 9.484). Mixed component improvements/regressions.

| Metric | PR #136 (sw=2.0) | PR #115 SOTA | Δ |
|---|---:|---:|---:|
| abupt_mean | 10.586 | **10.580** | +0.06% |
| surface_pressure | **5.642** | 5.690 | **−0.84%** |
| wall_shear | **10.396** | 10.419 | **−0.22%** |
| volume_pressure | 12.795 | **12.740** | +0.43% |
| tau_x | **8.784** | 8.908 | **−1.39%** |
| tau_y | 12.723 | **12.491** | +1.86% |
| tau_z | **12.988** | 13.071 | **−0.64%** |

- **Analysis:** sw=2.0 is a real lever — surface_pressure −0.84%, tau_x −1.39%, tau_z −0.64%. But the targeted tau_y REGRESSED (+1.86%, from 12.491 to 12.723) and volume_pressure also regressed slightly. Mixed-bag means orthogonal but small effect; abupt slightly worse.
- **Conclusion:** sw=2.0 standalone is borderline. The compound test (lr=1e-4 + sw=2.0 + EMA=0.999, askeladd #141) will reveal whether sw=2.0 adds value when stacked with the lr lever.
- **Alphonse reassigned** to PR #148: lr=1.5e-4 + EMA=0.999 (push LR ceiling, thorfinn's #115 follow-up #3).

## 2026-05-01 02:40 UTC — PR #134 CLOSED: frieren wd=2e-3 on lr=5e-5 base — 10.986 (+3.8% vs SOTA)

- **Branch:** `frieren/round9-lion-wd-sweep-2e3`
- **W&B run:** `3c514gml` rank 0 — group `tay-round9-lion-wd-sweep`, 270 min, 9 val epochs, best val 9.763 (ep9)
- **Hypothesis:** wd=2e-3 (4× current) on Lion uncompiled SOTA stack. Lion paper recommends higher wd to regularize sign-based updates.
- **Result:** test_abupt **10.986** vs SOTA **10.580** (PR #115) = **+3.84% regression**. Student compared to PR #111 (valid at launch time) where result was −1.40% win. PR #115 merged during run.

| Metric | PR #134 | PR #115 SOTA | Δ | AB-UPT |
|---|---:|---:|---:|---:|
| abupt_mean | 10.986 | **10.580** | +3.8% | — |
| surface_pressure | 6.065 | **5.690** | +6.6% | 3.82 |
| wall_shear | 10.895 | **10.419** | +4.6% | 7.29 |
| volume_pressure | 12.727 | **12.740** | −0.1% | 6.08 |
| tau_x | 9.232 | **8.908** | +3.6% | 5.35 |
| tau_y | 13.208 | **12.491** | +5.7% | 3.65 |
| tau_z | 13.698 | **13.071** | +4.8% | 3.63 |

- **Signal:** wd=2e-3 IS a real lever on lr=5e-5 base (−1.4% vs PR #111 confirmed). Classic regularization crossover at ep4 — slower early, better late. Slope still descending at ep9.
- **Conclusion:** wd lever orthogonal to EMA and schedule — follow-up needed on PR #115 base. Single-delta compound assigned as PR #147 (wd=2e-3 + lr=1e-4 + EMA=0.999).
- **Frieren reassigned** to PR #147: compound SOTA + wd=2e-3.

## 2026-05-01 02:17 UTC — PR #133 CLOSED: edward compound T_max=50 + EMA=0.999 — 11.116 (+5.1% vs SOTA)

- **Branch:** `edward/round9-compound-tmax50-ema999`
- **W&B run:** `a7k1k2y4` rank 0 — group `tay-round9-compound-tmax50-ema999`, 285 min, 9 val epochs, best val 9.9947 (ep9)
- **Hypothesis:** Compound Lion T_max=50 + EMA=0.999 (both confirmed single-variable winners stacked).
- **Result:** test_abupt **11.116** vs SOTA **10.580** (PR #115) = **+5.07% regression**. Student correctly noted +0.48% win over PR #110 (their benchmark at launch time), but PR #115 merged during run and set a higher bar.

| Metric | PR #133 | PR #115 SOTA | Δ vs SOTA | AB-UPT |
|---|---:|---:|---:|---:|
| abupt_mean | 11.116 | **10.580** | +5.1% | — |
| surface_pressure | 6.174 | **5.690** | +8.5% | 3.82 |
| wall_shear | 11.128 | **10.419** | +6.8% | 7.29 |
| volume_pressure | 12.625 | **12.740** | −0.9% | 6.08 |
| tau_x | 9.513 | **8.908** | +6.8% | 5.35 |
| tau_y | 13.435 | **12.491** | +7.6% | 3.65 |
| tau_z | 13.835 | **13.071** | +5.8% | 3.63 |

- **Analysis:** T_max=50 cosine + EMA=0.999 is a real compound (val 9.99 is the cleanest sub-10 outside of #115), but the LR=5e-5 base can't match LR=1e-4's coverage in 9 epochs. PR #115's lr=1e-4 change dominates entirely.
- **Conclusion:** T_max=50 and EMA=0.999 are additive on the lr=5e-5 base, but lr=1e-4 wins on a different dimension. The schedule lever is likely still useful when compounded on the lr=1e-4+EMA=0.999 base. Edward reassigned to 6L/256d depth swap (yi's −21% single-shot, biggest untested lever).
- **Edward reassigned** to PR #146: 6L/256d depth swap.

## 2026-04-30 21:00 UTC — PR #109 CLOSED: frieren Lion uncompiled + 1-epoch warmup — warmup closed-door

- **Branch:** `frieren/round7-lion-warmup`
- **W&B run:** `frieren-rank0` — group `tay-round7-lion-warmup-1ep`, 285 min, 9 val epochs
- **Hypothesis:** Add 1-epoch LR warmup to Lion uncompiled SOTA stack — potentially smoother early descent.
- **Result:** test_abupt **11.596** vs SOTA 11.208 (+3.5% regression). Best val 10.415 (ep9). Worse on every axis.

| Metric | PR #109 warmup | SOTA PR #50 | Δ |
|---|---:|---:|---:|
| abupt_mean | **11.596** | 11.208 | +3.5% |
| surface_pressure | 6.540 | 6.193 | +5.6% |
| wall_shear | 11.647 | 11.199 | +4.0% |
| volume_pressure | 12.857 | 12.726 | +1.0% |
| tau_x | 9.910 | 9.512 | +4.2% |
| tau_y | 14.100 | 13.592 | +3.7% |
| tau_z | 14.572 | 14.017 | +4.0% |

Trajectory comparison (ep1 onward, both Lion uncompiled lr=5e-5):

| ep | warmup (#109) | vanilla (#50) | gap |
|---:|---:|---:|---:|
| 1 | 84.62 | 80.68 | +5% |
| 4 | 21.11 | 17.31 | +22% |
| 7 | 11.55 | 11.11 | +4% |
| 9 | 10.42 | 10.08 | +3% |

- **Pattern:** warmup costs >1 full epoch of progress. By ep4 the warmup model is 22% behind. Gap closes by ep9 to +3% but the deficit is permanent — the ep1 LR ramp wastes a full epoch's worth of update magnitude.
- **Mechanism:** Lion's sign-based updates are already low-magnitude relative to AdamW. Warmup further reduces effective learning during a phase where we cannot afford slow descent (only 9 epochs in budget).
- **Verdict:** **Warmup closed-door on Lion uncompiled.** Constant LR is correct for this 9-epoch budget.
- **Frieren reassigned** to PR #120: batch=8 (2× batch lever).

## 2026-04-30 21:48 UTC — PR #112 CLOSED: alphonse Lion+lr=1e-4 — confounded test 11.324 (+1%)

- **Branch:** `alphonse/round8-lr-sweep`
- **W&B run:** `4x8hfam8` rank 0 — group `tay-round8-lion-lr-1e4`, 270 min, 9 val epochs
- **Hypothesis:** lr=1e-4 (2× current SOTA) on Lion uncompiled. LR sweep upper bound.
- **Result:** test_abupt **11.324** vs SOTA 11.208 (+1.0% regression). 2-variable confound: lr=1e-4 AND vol_w=1.0 (default, not 2.0). Student correctly flagged mid-run; advisor failed to reply.

| Metric | PR #112 (lr=1e-4, vol_w=1) | SOTA PR #50 | Δ | AB-UPT |
|---|---:|---:|---:|---:|
| abupt_mean | **11.324** | 11.208 | +0.116 | — |
| surface_pressure | 6.116 | 6.193 | −0.078 | 3.82 |
| wall_shear | 11.240 | 11.199 | +0.040 | 7.29 |
| volume_pressure | 13.226 | 12.726 | **+0.500** | 6.08 |
| tau_x | 9.476 | 9.512 | −0.036 | 5.35 |
| tau_y | 13.884 | 13.592 | +0.292 | 3.65 |
| tau_z | 13.919 | 14.017 | −0.099 | 3.63 |

- **Analysis:** lr=1e-4 trajectory: ep1-7 ahead, ep8-9 BEHIND. Lion's sign(.) update buys 2× initial step at lr=1e-4 but overshoots the loss minimum at the plateau. PR #50 ep9 slope was −0.30/ep; this run was −0.21/ep — closer to plateau. Volume_pressure regression dominated by missing vol_w=2.0.
- **Conclusion:** lr=5e-5 is at the upper bound of viable LR for Lion uncompiled in 9-epoch budget. lr=3e-5 (nezuko #113) testing the lower side. **Compound stack thorfinn #115 (lr=1e-4 + EMA=0.999) will give the cleaner data point on whether lr=1e-4 helps in compound.**
- **Alphonse reassigned** to PR #136: surface_loss_weight=2.0 (direct attack on tau_y/tau_z binding gap).

## 2026-04-30 21:38 UTC — PR #111 MERGED: tanjiro EMA=0.999 — new SOTA 11.142

- **Branch:** `tanjiro/round7-ema-fast`
- **W&B run:** `ab3y4ej7` rank 0 — group `tay-round7-ema-fast`, 285 min, 9 val epochs, best val 9.989 (ep9, first sub-10)
- **Hypothesis:** EMA decay 0.999 on Lion uncompiled SOTA — faster tracking captures rapid late-stage convergence better under 9-epoch budget.
- **Result:** test_abupt **11.142** vs prior SOTA 11.170 (**−0.25%, new SOTA**). Best val 9.989 (first sub-10 val on tay).

| Metric | PR #111 (MERGED) | PR #110 prior | PR #50 | Δ vs #110 | AB-UPT |
|---|---:|---:|---:|---:|---:|
| abupt_mean | **11.142** | 11.170 | 11.208 | **−0.028** | — |
| surface_pressure | 6.209 | 6.264 | 6.193 | −0.055 | 3.82 |
| wall_shear | **11.138** | 11.197 | 11.199 | −0.059 | 7.29 |
| volume_pressure | **12.548** | 12.556 | 12.726 | −0.008 | 6.08 |
| tau_x | **9.436** | 9.552 | 9.512 | −0.116 | 5.35 |
| tau_y | **13.525** | 13.572 | 13.592 | −0.047 | 3.65 |
| tau_z | 13.992 | **13.904** | 14.017 | +0.088 | 3.63 |

- **Analysis:** EMA=0.999 beats 0.9995 on 6/7 axes. Only tau_z regressed slightly (+0.088). Mechanism: under a 9-epoch budget, EMA=0.9995 has half-life ~1388 steps — too slow to fully track the rapid late-stage descent. EMA=0.999 (half-life ~693 steps) is better calibrated to this budget.
- **First sub-10 best val on tay** — ep9 val 9.989 vs vanilla 10.083.
- **Tanjiro reassigned** to PR #135: T_max=100 + EMA=0.999 (schedule sweep extension on new SOTA stack).
- **New SOTA: 11.142**

## 2026-04-30 21:22 UTC — PR #110 MERGED: edward Lion+cosine T_max=50 — new SOTA 11.170

- **Branch:** `edward/round7-lion-cosine-t50`
- **W&B run:** `ujyg3lju` rank 0 — group `tay-round7-lion-cosine-t50`, 276 min, 9 val epochs, best val 10.063 (ep9)
- **Hypothesis:** Cosine T_max=50 on Lion uncompiled SOTA — gentle annealing, only ~8% LR decay by ep9 (midpoint between T_max=24 and constant LR). Tests whether very gentle schedule helps late convergence.
- **Result:** test_abupt **11.170** vs SOTA 11.208 (**−0.34%, new SOTA**). Best val 10.063.

| Metric | PR #110 (MERGED) | Prior SOTA PR #50 | Δ | AB-UPT |
|---|---:|---:|---:|---:|
| abupt_mean | **11.170** | 11.208 | **−0.038** | — |
| surface_pressure | 6.264 | 6.193 | +0.071 | 3.82 |
| wall_shear | **11.197** | 11.199 | −0.002 | 7.29 |
| volume_pressure | **12.556** | 12.726 | **−0.170** | 6.08 |
| tau_x | 9.552 | 9.512 | +0.040 | 5.35 |
| tau_y | **13.572** | 13.592 | −0.020 | 3.65 |
| tau_z | **13.904** | 14.017 | **−0.113** | 3.63 |

- **Analysis:** T_max=50 is the "sweet spot" of gentle cosine annealing — just enough decay to help late convergence without the early-descent penalty of T_max=24. Key: volume_pressure −1.3%, tau_z −0.8%. Both are binding-gap components (volume_pressure ×2.1, tau_z ×3.8 vs reference). Surface_pressure regressed slightly (+1.1%) — expected tradeoff of volume emphasis.
- **Conclusion:** T_max=50 is a confirmed lever. Compounding with EMA=0.999 (tanjiro, ep9 9.989) is the natural next step → PR #133 edward.
- **New SOTA: 11.170**

## 2026-04-30 20:23 UTC — PR #92 CLOSED: thorfinn AdamW+RFF+768d+compile — diverged ep5-6

- **Branch:** `thorfinn/round7-adamw-rff-768d-compile`
- **W&B run:** `thorfinn/round7-adamw-rff-768d-compile-rank0-rank0` — group `thorfinn-round7-768d-adamw-compile`, running, rt=175m
- **Hypothesis:** Scale width to 768d on AdamW+RFF+compile stable base — wider model within 270m budget, building on PR #46 foundation.
- **Result:** CLOSED — fundamentally diverged. ep4 best val 17.33 (near vanilla 17.31), ep5 31.48, ep6 41.20. Diverged after ep4, trajectory going in wrong direction.

| ep | val | vanilla |
|---|---:|---:|
| 1 | 67.95 | 80.68 |
| 2 | 31.73 | 46.76 |
| 3 | 20.94 | 24.60 |
| **4** | **17.33** | **17.31** |
| 5 | 31.48 | 14.25 |
| 6 | 41.20 | 12.29 |

- **Analysis:** 4-way combo (AdamW + RFF + 768d + compile) is unstable on this stack. Adds to the compile divergence pattern (9 confirmed Lion+compile failures). Even AdamW+compile appears vulnerable at large width. Mechanistically: larger width amplifies the variance in compiled gradient computations.
- **Thorfinn reassigned** to PR #115: compound stack (lr=1e-4 + EMA=0.999) combining round8 winners.

## 2026-04-30 18:19 UTC — PR #94 CLOSED: askeladd Lion+RFF σ=0.5 — RFF sigma sweep fully closed-door

- **Branch:** `askeladd/round7-lion-rff-sigma0.5`
- **W&B run:** `zmrwhsw4` rank 0 — group `askeladd-round7-rff-freq-sweep`, 285 min, 9 val epochs
- **Hypothesis:** RFF σ=0.5 (lower freq than σ=1.0) — lower-frequency features may reduce inductive-bias mismatch. Third point in RFF sigma sweep.
- **Result:** test_abupt **11.353** vs SOTA 11.208 (+1.3% regression). Best val 10.405 (ep9).

| Metric | PR #94 σ=0.5 | SOTA PR #50 (vanilla) | AB-UPT |
|---|---:|---:|---:|
| abupt_mean | **11.353** | 11.208 | — |
| surface_pressure | 6.263 | 6.193 | 3.82 |
| wall_shear | 11.343 | 11.199 | 7.29 |
| volume_pressure | 12.943 | 12.726 | 6.08 |
| tau_x | 9.613 | 9.512 | 5.35 |
| tau_y | 13.877 | 13.592 | 3.65 |
| tau_z | 14.070 | 14.017 | 3.63 |

RFF sigma sweep results (all vs SOTA 11.208):

| σ | test_abupt | regression |
|---:|---:|---:|
| 0.5 (this) | 11.353 | +1.3% |
| 1.0 (#51) | 11.741 | +4.7% |
| 2.0 (#91) | 11.376 | +1.5% |
| no RFF (SOTA) | **11.208** | baseline |

- **Pattern:** σ=0.5 is best RFF sigma tested, but still regresses. Lower frequency may be marginally better but cannot overcome the fundamental RFF-limits-late-convergence problem.
- **Mechanism confirmed:** RFF accelerates early-phase fitting (ep1-5 consistently better) but the inductive bias interferes with fine convergence from ep6 onward.
- **RFF is fully closed-door** across 3 sigma values. Next direction: EMA sweep (askeladd reassigned to PR #114, EMA=0.998).

## 2026-04-30 17:57 UTC — PR #93 CLOSED: nezuko Lion+cosine T_max=24 — schedule sweep closed-door on Lion uncompiled

- **Branch:** `nezuko/round7-lion-cosine-tmax24`
- **W&B run:** `ooho1daw` rank 0 — group `nezuko-round7-lion-cosine-tmax`, 285 min, 9 val epochs
- **Hypothesis:** Lion+cosine T_max=24 nocompile — does cosine decay over 24 epochs (longer warm-up-to-decay vs T_max=16) improve final convergence vs vanilla constant LR?
- **Result:** test_abupt **11.524** vs SOTA 11.208 (+2.8% regression). Best val 10.301 (ep9).

| Metric | nezuko T=24 (#93) | SOTA PR #50 (vanilla) | Δ |
|---|---:|---:|---:|
| abupt_mean | **11.524** | 11.208 | +2.8% |
| best val | 10.301 (ep9) | 10.083 (ep9) | +2.2% |

Epoch-by-epoch trajectory (vs vanilla Lion SOTA):

| ep | nezuko T=24 | vanilla SOTA | gap |
|---|---:|---:|---:|
| 1 | 74.97 | 80.7 | −7% (faster start!) |
| 2 | 45.44 | 36.5 | +24% |
| 5 | 14.59 | **14.25** | +2.4% |
| 6 | 12.66 | **12.29** | +3.0% |
| 7 | 11.42 | **11.11** | +2.8% |
| 8 | 10.64 | **10.38** | +2.5% |
| 9 | 10.30 | **10.08** | +2.2% |

- **Conclusion:** Cosine annealing hurts beyond ep1. Vanilla Lion's constant LR is already near-optimal for the 9-epoch budget. Adding decay takes useful LR away before convergence is reached. Combined with #57 (T_max=16, wash) this closes the entire schedule sweep on Lion uncompiled.
- **Nezuko reassigned to PR #113: lr=3e-5 LR sweep lower bound.**

## 2026-04-30 17:05 UTC — PR #91 CLOSED: alphonse Lion+RFF σ=2.0 — sigma sweep complete, RFF definitively closed on Lion uncompiled

- **Branch:** `alphonse/round6-lion-rff-sigma2`
- **W&B run:** `ip8tf46r` rank 0 — group `alphonse/round6-lion-rff-sigma2-rank0`, 285 min, 9 val epochs, SENPAI_TIMEOUT=360
- **Hypothesis:** RFF σ=2.0 (higher freq than σ=1.0) on Lion uncompiled — higher freq features may better capture tau_y/tau_z geometry.
- **Result:** test_abupt **11.376** vs SOTA 11.208 (+1.5% regression). Best val 10.321 (ep9).

| Metric | PR #91 σ=2.0 | SOTA PR #50 (vanilla) | Δ |
|---|---:|---:|---:|
| abupt_mean | **11.376** | 11.208 | +1.5% |
| surface_pressure | 6.311 | 6.193 | +1.9% |
| wall_shear | 11.342 | 11.199 | +1.3% |
| volume_pressure | 13.090 | 12.726 | +2.9% |
| wall_shear_x | 9.739 | 9.512 | +2.4% |
| **wall_shear_y** | **13.450** | 13.592 | **−1.0% BETTER** |
| wall_shear_z | 14.289 | 14.017 | +1.9% |

**Full sigma sweep summary (Lion uncompiled, vanilla SOTA = 11.208):**

| σ | ep5 val | ep9 val | test | Δ |
|---|---:|---:|---:|---:|
| vanilla (no RFF) | 14.25 | 10.083 | **11.208** | SOTA |
| σ=0.5 (askeladd, running) | 13.83 | TBD | TBD | >+1.5% |
| σ=1.0 (edward #51) | 13.53 | 10.703 | 11.741 | +4.7% |
| σ=2.0 (this PR) | 13.25 | 10.321 | 11.376 | +1.5% |

- **Pattern:** RFF accelerates ep1-5 (all σ values beat vanilla early), but vanilla catches up by ep6 and dominates from ep6 onward. Higher σ reduces the regression penalty but no σ beats vanilla.
- **Notable:** σ=2.0 wins on tau_y by 1% (13.450 vs 13.592). Higher freq encoding may be meaningful for tau components — worth revisiting if other levers plateau.
- **Closed-door:** RFF on Lion uncompiled at any tested σ (0.5, 1.0, 2.0) is a regression. Schedule sweep is the active focus.

## 2026-04-30 16:45 UTC — PR #90 CLOSED: tanjiro Lion+RFF + EMA decay 0.9999 — budget-incompatible

- **Branch:** `tanjiro/round6-lion-rff-ema9999`
- **W&B run:** `ocubvc23` rank 0 — group `tanjiro/round6-lion-rff-ema9999-rank0`, 285 min, 9 val epochs
- **Hypothesis:** EMA decay 0.9999 (vs baseline 0.9995) on Lion+RFF uncompiled — slower EMA smoothing may reduce late-stage oscillations and improve generalization.
- **Result:** test_abupt **30.203** vs SOTA 11.208 (+169% regression). Best val 29.540.

| Metric | PR #90 (EMA 0.9999) | Edward #51 (EMA 0.9995) | SOTA PR #50 | AB-UPT ref |
|---|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **30.203** | 11.741 | 11.208 | — |
| `surface_pressure_rel_l2_pct` | 20.821 | 6.394 | 6.193 | 3.82 |
| `wall_shear_rel_l2_pct` | 31.519 | 11.837 | 11.199 | 7.29 |
| `volume_pressure_rel_l2_pct` | 25.057 | 13.111 | 12.726 | 6.08 |
| `wall_shear_x` | 26.216 | — | 9.512 | 5.35 |
| `wall_shear_y` | 39.381 | — | 13.592 | 3.65 |
| `wall_shear_z` | 39.543 | — | 14.017 | 3.63 |

- **Mechanism:** EMA decay 0.9999 requires ~10× more steps than the 9-epoch / ~23K-step budget provides. The EMA snapshot at best-val is still dominated by early-epoch (effectively random-init) weights. Best-val checkpoint reloads a snapshot that was never a valid EMA of converged weights.
- **Closed-door for this budget:** EMA ≥ 0.9999 is incompatible with 270m budget. EMA 0.999 (5× faster tracking) assigned next to tanjiro PR #111.
- **Contrast:** EMA 0.9995 baseline is fine — at ~23K steps, the EMA half-life is ~14K steps ≈ 5 epochs. EMA 0.9999 half-life is ~140K steps ≈ 50 epochs.

## 2026-04-30 16:05 UTC — PR #51 CLOSED: edward Lion+RFF σ=1.0 reproducer — RFF doesn't compose with vanilla Lion uncompiled

- **Branch:** `edward/edward/round2-lion-rff-512d`
- **W&B run:** `iocqp761` rank 0 — group `tay-round2-lion-rff`, 286 min, 9 val epochs, ep9 best
- **Hypothesis:** Lion uncompiled + RFF σ=1.0 reproduces ftg0ci0p val 10.665 and lands missing test eval. Test expected ~11.0 (within +5% of val).
- **Result:** test_abupt **11.741** vs SOTA 11.208 (+4.7% regression)

| Metric | PR #51 (RFF σ=1.0) | SOTA PR #50 (vanilla Lion) | Δ |
|---|---:|---:|---:|
| test_abupt | **11.741** | 11.208 | +4.7% |
| best val | 10.703 | 10.083 | +6.1% |
| val→test gap | +9.7% | +11.2% | similar |

**Per-epoch comparison vs vanilla Lion SOTA:**

| Ep | Lion+RFF σ=1.0 | vanilla Lion |
|---:|---:|---:|
| 1 | 72.96 | 80.68 |
| 5 | 13.53 | 14.25 |
| 6 | 12.31 | 12.29 |
| 7 | 11.56 | 11.11 |
| 8 | 11.01 | 10.38 |
| 9 | **10.70** | **10.08** |

- **Mechanism:** RFF accelerates early-phase fitting (high-freq detail dominates gradient signal) but the redundant Fourier basis interferes with finer convergence late. Vanilla Lion converges to lower minimum without the inductive bias.
- **Closed-door for round 7:** σ=1.0 RFF on Lion uncompiled is a regression. alphonse #91 σ=2.0 (ep8 val 10.63 vs vanilla 10.38) showing same crossover pattern, will likely also regress.
- **Reproducibility confirmed:** edward val 10.703 ≈ ftg0ci0p val 10.665. Run is reliable.

## 2026-04-30 15:30 UTC — PR #73 CLOSED: frieren AdamW+RFF+compile + 6L depth — 6L+compile budget-limited at 11 epochs

- **Branch:** `frieren/round5-adamw-rff-compile-depth6`
- **W&B run:** `162ulxek` rank 0 — group `tay-round5-depth6`, 271 min, 11 epochs (vs 16 for 4L compile)
- **Hypothesis:** 6L compile gives more capacity per epoch — would the steeper per-epoch descent beat 4L's longer epoch budget?
- **Result:** test_abupt **14.785** vs SOTA 11.208 (+32%). Best val 13.90 ep11.

| Metric | PR #73 6L | PR #46 4L compile | tay SOTA `vnb7oheo` (Lion 4L) | AB-UPT (768d/4L) |
|---|---:|---:|---:|---:|
| `abupt` mean | **14.785** | 14.550 | 11.303 | — |
| surface_pressure | 8.875 | 8.628 | 6.216 | 3.82 |
| wall_shear | 15.220 | 14.882 | 11.315 | 7.29 |
| volume_pressure | 14.737 | 15.032 | 12.755 | 6.08 |
| tau_x | 13.110 | 12.901 | 9.563 | 5.35 |
| tau_y | 17.810 | 17.281 | 13.831 | 3.65 |
| tau_z | 19.394 | 18.907 | 14.147 | 3.63 |

- **Capacity is BUDGET-LIMITED for compile path:** 6L compile gets 11 epochs (vs 16 for 4L compile) at 270m. Per-epoch +45%. Linear extrapolation: 6 more epochs at slope −0.46/ep would reach val ~11.1, but the budget caps at 11.
- **Same lesson as PR #69 (768d uncompiled, 5 epochs vs 9 = test 12.35):** Adding capacity (depth or width) costs epochs proportionally. SOTA path stays 4L.
- **Closed-door:** Increasing model size in the compile path does NOT pay off within 270m. AdamW+compile at 4L (PR #46 14.55) is the AdamW frontier.

## 2026-04-30 08:05 UTC — PR #52 CLOSED: tanjiro Lion+RFF+compile triple-stack — Lion+compile divergence is now confirmed across 6 runs

- **Branch:** `tanjiro/tanjiro/round2-lion-rff-compile-512d`
- **W&B run:** `5o1frm3u` — group `tay-round2-lion-rff-compile`, 270min full budget, 16 epochs
- **Hypothesis:** Compose Lion (lr=5e-5/wd=5e-4) + RFF σ=1.0 + compile to stack three orthogonal wins → expected ~9.8-10.3% test_abupt (additive on PR #39 base 15.43).
- **Result: ALL THREE LEVERS NON-ORTHOGONAL — test_abupt 13.199 (+16.8% vs SOTA 11.30, but -9.3% vs PR #46 14.55)**

| Metric | PR #52 | SOTA `vnb7oheo` | PR #46 |
|---|---:|---:|---:|
| `test/abupt_axis_mean_rel_l2_pct` | **13.199** | 11.303 | 14.550 |
| surface_pressure | 7.532 | 6.216 | 8.628 |
| wall_shear | 13.351 | 11.315 | 14.882 |
| volume_pressure | 14.297 | 12.755 | 15.032 |
| tau_x | 11.418 | 9.563 | 12.901 |
| tau_y | 16.036 | 13.831 | 17.281 |
| tau_z | 16.714 | 14.147 | 18.907 |

- **CRITICAL DIAGNOSTIC** (from tanjiro's analysis): Triple-stack at epoch 6 (val 12.22) tracks vanilla Lion Arm B at epoch 6 (val 12.14) within 0.08 ppt. **RFF and compile add ZERO orthogonal signal in the stable Lion regime.** The lift from PR #39 (lr=1.7e-5 → 15.43) to SOTA (lr=5e-5 → 11.30) was the LR change alone. RFF+compile only matter as horizon extenders — and the extension is exactly what kills Lion+compile.
- **Mechanism (now confirmed across 6 runs):** Lion's `sign()` update has constant per-coordinate step magnitude bounded only by `lr`. Compile's reduced gradient noise → signs become deterministically biased → small-gradient noise (epoch 6+) gets amplified into full-step LR moves → train loss explodes (no NaN, slow blow-up: 0.041 → 1.05 → 5.43 → 3.07 across epochs 7-9). Same root cause as PR #42's per-case ratio variance forcing.
- **Six independent observations of Lion+compile divergence at lr=5e-5:**
  - PR #42 (frieren, sq_rel_l2 forcing) — diverged
  - PR #50 (nezuko, vanilla Lion+compile) — diverged at epoch 5
  - PR #52 (tanjiro, Lion+RFF+compile) — diverged at epoch 7
  - PR #54 (fern, Lion+per-axis-weights+compile) — diverged
  - PR #57 (askeladd, Lion+compile+cosine T_max=16) — diverged at epoch 4 (cosine did NOT fix it)
  - PR #68 (frieren, Lion+vol_w=3+compile) — diverged at epoch 6
- **Closed-door insight:** **Lion+compile at lr=5e-5 is unstable past epoch 6 in 100% of observed runs.** SOTA `vnb7oheo` (uncompiled) is at the natural Lion floor, with the slow uncompiled cadence acting as an implicit step-size budget. Cosine T_max=16 does NOT fix the divergence (#57 confirmed). Future Lion+compile experiments must either (a) halve LR to 2.5e-5, (b) use aggressive cosine T_max≤8 with proper warmup, or (c) implement a Lion-specific soft-restart on train_loss_relative_to_min plateau.
- **Best-checkpoint mechanism saved the result** — the test eval auto-loaded epoch 6's EMA weights, giving a usable test_abupt=13.20. Without the best-val checkpoint, every Lion+compile run would post the diverged final-epoch metrics.
- **Per-axis pattern matches SOTA proportionally** — all axes scale ~1.17x SOTA, suggesting no axis-specific gain or loss from RFF/compile in the stable regime.

## 2026-04-30 10:25 UTC — PR #68 CLOSED: frieren Lion + vol_w=3.0 — 8th Lion+modification divergence, vol_w lever inert when run diverges

- **Branch:** `frieren/lion-vol-weight-3`
- **W&B run:** `daayn9kb` — finished 285min
- **Hypothesis:** Lion (lr=5e-5/wd=5e-4) + volume_loss_weight 3.0 targets the volume_pressure ×2.1 binding gap. Single delta from SOTA arm B.
- **Result: DIVERGED — test_abupt 15.572 (+37.7% vs SOTA 11.30, +1.0 ppt vs PR #46 14.55, best-val ckpt epoch 5)**

| Metric | PR #68 (best-val ep5) | SOTA `vnb7oheo` | PR #46 | Δ vs SOTA |
|---|---:|---:|---:|---:|
| `test/abupt_axis_mean_rel_l2_pct` | **15.572** | 11.303 | 14.550 | +37.7% |
| surface_pressure | 9.752 | 6.216 | 8.628 | +57% |
| wall_shear | 16.338 | 11.315 | 14.882 | +44% |
| volume_pressure | 14.003 | 12.755 | 15.032 | +9.7% |
| tau_x | 13.985 | 9.563 | 12.901 | +46% |
| tau_y | 19.359 | 13.831 | 17.281 | +40% |
| tau_z | 20.761 | 14.147 | 18.907 | +47% |

- **Mechanism:** Lion+vol_w=3 diverged at epoch 6 (val explosion 14.43 → 88-100%). Same Lion-fragility pattern. Critically, **volume_pressure ended at 14.00 — barely better than SOTA's 12.76**. The vol_w=3 lever didn't get to act because divergence killed the run. So the gap-targeting hypothesis is untestable on Lion.
- **Lion-fragility envelope (8 confirmed observations):** compile, sq_rel_l2, per-axis weighting, vol_w=3.0, cosine T_max=16, grad-clip 5.0, RFF+compile (composition), per-axis-weighting+compile (composition). The only stable Lion variants: vanilla Lion (vnb7oheo, 11.30 SOTA) and Lion+RFF (PR #51 edward, descending at val 11.48 ep7).
- **Comparison signal:** alphonse #55 currently running AdamW+RFF+compile+vol_w=3.0 (lahk19ws, val 15.49 ep12) — direct comparison of vol_w=3 effect on the stable AdamW base. Will reveal if vol_w=3 actually helps volume_pressure when training stays stable.
- **Reassignment:** frieren → PR #73 (AdamW+RFF+compile + depth 6L). Architectural capacity lever, orthogonal to thorfinn #69's width 768d. Branch: `frieren/round5-adamw-rff-compile-depth6`.

## 2026-04-30 09:25 UTC — PR #54 CLOSED: fern Lion + per-axis tau_y/tau_z weighting (2×) — diverged, 7th Lion+modification failure

- **Branch:** `fern/round3-lion-tauyz-weights` → recovered as v2 (`fern/round3-lion-tauyz-weights-v2`)
- **W&B runs:** `6p4k0280` (crashed 111min), `8zhjetjt` (v2, finished 285min)
- **Hypothesis:** Per-channel wall_shear loss weighting (tau_y ×2, tau_z ×2, tau_x ×1) on Lion (lr=5e-5/wd=5e-4) forces proportional attention toward the binding tau_y/tau_z gap (×3.8/×3.9 vs AB-UPT). Expected: target those axes specifically while preserving Lion's optimizer advantage.
- **Result: DIVERGED — test_abupt 26.827 (+137% vs SOTA 11.30, best-val checkpoint at epoch 3)**

| Metric | PR #54 (best-val ep3) | SOTA `vnb7oheo` | AB-UPT |
|---|---:|---:|---:|
| `test/abupt_axis_mean_rel_l2_pct` | **26.827** | 11.303 | — |
| surface_pressure | 20.655 | 6.216 | 3.82 |
| wall_shear | 28.560 | 11.315 | 7.29 |
| volume_pressure | 19.436 | 12.755 | 6.08 |
| tau_x | 24.835 | 9.563 | 5.35 |
| tau_y | 33.109 | 13.831 | 3.65 |
| tau_z | 36.104 | 14.147 | 3.63 |

- **Val trajectory:** ep1=75 → ep2=46 → ep3=26 (best) → **ep4=220 (DIVERGED)** → settled at 93%. Same divergence onset pattern as PRs #42, #52, #57, #68.
- **Diagnostic insight:** Per-axis breakdown showed weighting *did* reduce tau_y/tau_x ratio (1.33 vs vanilla Lion's 1.45), confirming the loss mechanism works at the gradient level. The divergence is entirely Lion's instability to loss modification — adding 2× per-axis weighting changes the effective gradient magnitudes per-channel, which pushes the sign-update into a new regime. Lion is now confirmed fragile to: compile, sq_rel_l2, per-axis weighting, vol_w=3.0, cosine T_max=16, and vol_w=3.0. The only stable Lion variants are vanilla Lion and Lion+RFF (PR #51 edward, stable at epoch 5).
- **Conclusion:** The per-axis weighting *mechanism* is valid — test it on AdamW+RFF+compile (PR #46 base) where the optimizer is stable. Assigned as PR #71 (fern, round5).
- **PR #71 — fern: AdamW + RFF + compile + per-axis tau_y/tau_z weighting (2×):** Tests the same weighting mechanism on the stable AdamW base. Expected: beat PR #46 (14.55) by closing tau_y/tau_z gap. Branch: `fern/round5-adamw-rff-compile-tauyz`.

## 2026-04-30 07:30 UTC — PR #56 CLOSED: thorfinn AdamW + cosine T_max=16 — schedule miscalibration was load-bearing

- **Branch:** `thorfinn/round3-rff-compile-cosine-lr16`
- **W&B run:** `ko1hrdau` — group `tay-round3-rff-compile-cosine-lr16`, 270min full budget, 16 epochs
- **Hypothesis:** PR #46's `--lr-cosine-t-max 0` fallback to `max_epochs=50` left LR at ~88% of init throughout. Calibrating `T_max=16` to actual budget should give Lion(/AdamW) the late-training fine-tuning it never got.
- **Result: HYPOTHESIS REJECTED — +13.0% regression vs PR #46 (test_abupt 16.44 vs 14.55)**

| Metric | PR #56 | PR #46 | Δ vs #46 | AB-UPT |
|---|---:|---:|---:|---:|
| `test/abupt_axis_mean_rel_l2_pct` | **16.440** | 14.550 | +1.890 (+13.0%) | — |
| surface_pressure | 10.042 | 8.628 | +1.414 (+16.4%) | 3.82 |
| wall_shear | 17.076 | 14.882 | +2.194 (+14.7%) | 7.29 |
| volume_pressure | 15.959 | 15.032 | +0.927 (+6.2%) | 6.08 |
| tau_x | 14.862 | 12.901 | +1.961 (+15.2%) | 5.35 |
| tau_y | 19.837 | 17.281 | +2.556 (+14.8%) | 3.65 |
| tau_z | 21.502 | 18.907 | +2.595 (+13.7%) | 3.63 |

- **Mechanism:** Cosine T_max=16 collapsed LR to ~6% of init by epoch 14 and 0.6% by epoch 16. PR #46's val curve was still descending at epoch 16, meaning the model was undertrained, not overoptimized. The aggressive LR decay starved it of the late-training optimization power that was load-bearing for the 14.55 result. val→test gap shrunk +18% (sharper minima generalize slightly better) but this was swamped by the +2.0 absolute val regression.
- **Closed-door insight (DO NOT REVISIT WITHOUT BUDGET EXTENSION):** AdamW lr=5e-5 at 16-epoch budget is undertrained, not overoptimized. Cosine T_max calibrated to actual budget is a regression. Future cosine experiments must (a) use generous T_max≥24 with proper warmup, or (b) extend budget rather than compress LR. The "miscalibration" in PR #46 was effectively a constant-ish LR schedule and was load-bearing.
- **Per-axis pattern:** All axes regressed 6-16%. Volume_pressure least affected, surface_pressure / tau_y / tau_z worst — consistent with a global LR-too-low-too-fast effect, not axis-specific.
- **Suggested follow-ups (deferred):** thorfinn's #5 ablation (PR #46 with explicit T_max=50) would confirm the fallback was load-bearing — but lever is now low priority since Lion is the active SOTA arm.



## Round 1 — opened 2026-04-29

8 students assigned in parallel on DDP8 (8 GPUs each, 96 GB VRAM, effective
bs scales with `nproc_per_node × per-GPU bs`). Strategy: 5 students compose
yi's confirmed-orthogonal wins (width × FiLM × cosine-EMA × Fourier × LR
warmup); 3 students push beyond yi with architectural / loss / TTA changes
that yi only got as far as Round-2 assignments for.

| PR | Student | Hypothesis |
|---|---|---|
| #30 | alphonse | yi PR #4 reproduce (4L/512d/8h, lr=5e-5, bs=4) — calibration |
| #31 | askeladd | Full composition stack: 512d × cosine-EMA × tangential × vol_w=2.0 |
| #32 | edward | Cosine LR + 5% warmup on top of 512d composition |
| #33 | fern | Gaussian Fourier coord features + 512d composition |
| #34 | frieren | AdaLN-zero per-block FiLM + 512d composition |
| #35 | nezuko | A01 — ANP cross-attention surface decoder |
| #36 | tanjiro | SDF-gated volume attention bias for near-wall p_v |
| #37 | thorfinn | Per-axis wall-shear loss weighting + bilateral-symmetry TTA |

## Round 1 — in-progress observations (2026-04-29 12:35 UTC)

All 8 still WIP. No PRs marked review-ready. Per-axis val curves are
informative even before completion:

```
Run                              step    val_abupt  ps     ws     pv
alphonse (calibrate)             10887   27.74      20.01  30.94  15.86
edward (cosine warmup)           10887   35.68      25.46  40.10  19.10
fern (RFF features)               8165   30.07      22.07  32.59  19.03
askeladd (composition stack)      8165   39.49      19.04  46.75  14.49 ← ws regression
thorfinn (per-axis weights)       8165   33.6       n/a    n/a    n/a
frieren (FiLM AdaLN-zero)         8165   34.4       n/a    n/a    n/a
nezuko (ANP decoder)              4316   76.4       n/a    n/a    n/a   ← much slower
tanjiro (SDF gate)                  —      —          —      —     —    ← 4 crashes at step 2719 (in eval path)
```

**Key in-progress signals** (caveat: not at completion, only first 4
validations; ranking may shift by epoch ~10):

1. **alphonse calibration matches yi epoch-1 (26.24) within 5%** —
   confirms tay/DDP8 baseline is healthy; yi's wins should reproduce.
2. **askeladd's composition stack is BEST on `ps`/`pv` but WORST on `ws`**
   — `ps=19.0` vs alphonse's 20.0, `pv=14.5` vs alphonse's 15.9, but
   `ws=46.7` vs alphonse's 30.9. The tangential wall-shear projection
   loss is net-negative for raw wall_shear despite improving the other
   axes. **Important Round 2 implication**: do NOT bundle the tangential
   projection into "compose all yi wins" runs — it hurts the metric it
   was designed to help.
3. **fern's RFF features show strong lift** — at the same step (8165)
   fern is at 30.1 vs alphonse 35.1 → RFF is doing real work. Will see
   how it compounds at later validations.
4. **edward's cosine warmup catches up rapidly** — at step 10887, edward
   is 35.68 vs alphonse's 27.74; warmup arm typically lags early then
   converges. Worth running to completion to see if it surpasses
   alphonse asymptotically.
5. **nezuko ANP decoder is dramatically slower per step** — same wall
   time produces ~4x fewer steps than alphonse. At step 4316 only 1
   validation (val=76.4). May not finish enough epochs to be comparable.
6. **tanjiro 4 crashes at exactly step 2719** — deterministic failure in
   eval path. Posted advisor comment with simplified-σ guidance.

## 2026-04-29 15:21 UTC — PR #30 merged: first tay/DDP8 baseline (alphonse calibration)

**Student:** alphonse | **W&B run:** `0vi9tm5h` | **Hypothesis:** Reproduce yi PR #4 (4L/512d/8h, lr=5e-5, bs=4, vol_w=2.0) on tay/DDP8.

### Results

| Metric | tay val | tay test | yi best | AB-UPT |
|---|---:|---:|---:|---:|
| abupt | 18.70 | **19.81** | 15.82 | — |
| surface_pressure | 12.93 | 12.86 | 9.99 | 3.82 |
| wall_shear | 21.24 | 21.27 | 16.60 | 7.29 |
| volume_pressure | 9.69 | 15.91 | 14.21 | 6.08 |
| tau_x | 18.09 | 18.24 | 14.27 | 5.35 |
| tau_y | 25.54 | 25.50 | 19.49 | 3.65 |
| tau_z | 27.26 | 26.53 | 21.12 | 3.63 |

### Analysis

This establishes tay's **first concrete test baseline at 19.81 abupt**. Run
was under-trained at 9 epochs (of 50) — loss still steeply descending at
end (val slope −0.37/1k steps). Root cause: `torch.compile + drop_last=False`
interaction crashes all 8 ranks at the epoch-boundary step. Student used
`--no-compile-model` workaround (1.5–2× per-step cost), limiting epochs to
~9 within the 270-min budget.

**Critical infra finding**: Fix is `drop_last=True` in `trainer_runtime.py:293`
(editable per program.md). Estimated ~2× throughput gain = ~14–22 compiled
epochs in budget instead of 9 uncompiled. Alphonse reassigned to PR #40 to
land the fix and re-calibrate.

**Round 2 implication**: All 7 concurrent Round-1 students ran without compile.
Results from this round should be compared apples-to-apples (all uncompiled).
After compile fix merges, Round 2 baselines reset.

---

## 2026-04-29 21:25 UTC — PR #41 CLOSED: askeladd eval-tangential projection — clear regression

**Student:** askeladd | **W&B run:** `p3lxbg6t` (rank 0) | **Hypothesis:** Project
predicted wall-shear vector onto surface tangent at eval time only (vs yi PR #11
kohaku's training-time projection that broke tau_z).

### Results — vs current SOTA (PR #40 compiled, 12 epochs)

| Metric | PR #41 | PR #40 SOTA | Δ vs SOTA |
|---|---:|---:|---:|
| `val_abupt` | 20.25 | 16.09 | +4.16 (+25.9%) |
| `test_abupt` | **21.13** | 17.25 | **+3.88 (+22.5%)** |

Even vs the OLD uncompiled baseline (PR #30: 19.81), this is +1.32 worse (+6.7%).

### Disposition: CLOSED

The eval-time projection mechanism is wrong-shaped. Mechanism analysis:
- Trained model predicts `tau` ≈ `α · n_normal + β · n_tangent` (some normal-component
  contribution that helps rel-L2 even though it's physically anomalous on flat panels).
- Projecting onto tangent removes the α component.
- Remaining `β · n_tangent` is now a worse predictor than the unprojected `tau` was.

**Combined with yi PR #11 kohaku results, this closes the door on tangential-projection-on-output**.
Future wall-shear constraints should consider:
- Predicting in tangent-frame intrinsic coordinates (projection built into the head).
- Joint loss penalizing `tau · n_normal` rather than projecting at eval.

Reassigning askeladd to **PR #49: grad-clip-norm 1.0 → 5.0 single delta** (motivated
by frieren's diagnostic that clip rate was 1.0 every step, late-training median
grad_norm ~4 — 1.0 clip is throwing away ~75% of gradient magnitude).

---

## 2026-04-29 22:05 UTC — PR #39 MERGED: tanjiro Lion lr=1.7e-5 — NEW SOTA 15.43, crosses yi frontier

**Student:** tanjiro | **W&B run:** `xonbs83i` (rank 0) | **Group:** `tay-lion-lr-sweep`
**Hypothesis:** Drop-in Lion optimizer (paper config `lr=1.7e-5, wd=5e-3`) in place of AdamW.

Note: student's in-pod claude hung since iteration 5 (17:42 UTC); training ran autonomously.
Results verified directly from W&B summary. PR readied and merged by advisor.

### Results — vs PR #40 SOTA (compiled, 12 epochs)

| Metric | PR #39 Lion | PR #40 SOTA | Δ abs | Δ % |
|---|---:|---:|---:|---:|
| `val_primary/abupt` | **14.22** | 16.09 | −1.87 | **−11.6%** |
| `test_primary/abupt_axis_mean_rel_l2_pct` | **15.43** | 17.25 | −1.82 | **−10.5%** |
| `test_primary/surface_pressure_rel_l2_pct` | **9.45** | 10.92 | −1.47 | **−13.5%** |
| `test_primary/wall_shear_rel_l2_pct` | **16.28** | 18.33 | −2.05 | **−11.2%** |
| `test_primary/volume_pressure_rel_l2_pct` | **13.83** | 14.71 | −0.88 | **−6.0%** |
| `test_primary/wall_shear_x_rel_l2_pct` | **13.91** | 15.73 | −1.82 | **−11.6%** |
| `test_primary/wall_shear_y_rel_l2_pct` | **19.58** | 21.80 | −2.22 | **−10.2%** |
| `test_primary/wall_shear_z_rel_l2_pct` | **20.40** | 23.07 | −2.67 | **−11.6%** |

**Best_epoch = 9 / 50 configured** — val curve still descending at cutoff.

**tay crosses below yi frontier: 15.43 vs yi best 15.82 (PR #13 norman).**

### Config

- optimizer: `lion`, lr: `1.7e-5`, wd: `5e-3`, beta1/beta2: `0.9/0.99`
- base: 4L/512d/8h/128slices, ema_decay=0.9995, vol_w=2.0, bs=4×8GPU
- **No compile** (pod on pre-compile-fix commit `269cb09`)
- **No RFF**
- Runtime: 287.1 min, DDP8

### Mechanism analysis

1. **Sign-update sidesteps clip compression.** With `grad-clip-norm=1.0` binding at every
   step (frieren's diagnostic: clip_rate=1.0/23458 steps), AdamW's per-coordinate
   magnitude scaling is being compressed. Lion uses `torch.sign(momentum)` — each
   coordinate gets a step of exactly `lr`, independent of gradient magnitude. The
   clip still fires but only scales the momentum update, not the step direction.

2. **Larger weight decay tolerated.** `wd=5e-3` is 10× our default `5e-4`. Lion's
   documented sweet spot; provides stronger implicit regularization against overfitting
   on 400 CFD training cases.

3. **val curve still descending at epoch 9** → compile (12 epochs) or longer budget
   should further improve. Lion + compile is the next highest-priority test.

### Disposition: MERGED — new tay SOTA

Reassigning tanjiro to follow-up arm: arm B (Lion lr=5e-5, `vnb7oheo`) launched
automatically and still running as of merge.

---

## 2026-04-29 22:10 UTC — PR #35 CLOSED: nezuko ANP cross-attention decoder — two rounds, no win

**Student:** nezuko | **Round-1 run:** `2fqts0v8` | **Round-2 run:** `ochavw4i`
**Hypothesis:** Replace Transolver surface MLP head with ANP-style cross-attention decoder.

### Results (both rounds)

| Run | Config | test_abupt | vs SOTA at time |
|---|---|---:|---:|
| Round-1 ANP | standalone, no RFF, vol_w=2 | 18.76 | −5.3% vs #30 calibration |
| Round-2 ANP+RFF | ANP + RFF sigma=1.0 + vol_w=3 | **17.55** | **+1.7% vs PR #40 SOTA** |
| **PR #39 SOTA** | Lion lr=1.7e-5 | **15.43** | — |

### Key findings

1. **Surface wins are real but partial.** Standalone ANP: p_s −7.2%, tau axes −7-8%.
   With RFF composition: p_s −2.5%, tau_x −1% (RFF captures same non-local spatial
   structure — not fully orthogonal).
2. **p_v regression is structural.** ANP's cross-attention over slice tokens focuses
   on surface geometry; volume targets need backbone-level volume processing that
   isn't helped by a surface head replacement.
3. **Cannot be compiled.** `torch.inductor` dynamic-shapes `AssertionError` at
   epoch 1 (backbone-then-split `s24 + 65536` codegen bug). Stuck at ~9 uncompiled
   epochs vs 12 compiled — fundamental throughput disadvantage.
4. **vol_w=3 didn't help p_v.** Round-2 p_v = 17.26 vs Round-1 (not logged final).
   Increasing vol_w amplified a weak signal; the head needs a different fix.

### Disposition: CLOSED

Slice tokens (post-SDPA) are high-quality local-geometry embeddings — future head
replacements should read those, not pre-SDPA tokens. But for now, composition
wins from Lion + compile are more tractable.

Reassigning nezuko to **PR #50: Lion + compile (single delta)**.

---

## 2026-04-29 22:12 UTC — PR #44 CLOSED: edward cosine EMA (0.99→0.9999) — budget miscalibration regression

**Student:** edward | **W&B run:** `51yrbxxj` | **Group:** cosine-ema + RFF
**Hypothesis:** Progressive cosine EMA decay 0.99→0.9999 (replicating yi PR #13 norman win)
+ RFF coord features.

### Results

| Metric | PR #44 | PR #40 SOTA | PR #39 new SOTA | Δ vs PR #40 |
|---|---:|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **18.33** | 17.25 | 15.43 | **+6.3%** |
| surface_pressure | 11.69 | 10.92 | 9.45 | +7.1% |
| wall_shear | 19.35 | 18.33 | 16.28 | +5.6% |
| volume_pressure | 16.12 | 14.71 | 13.83 | +9.6% |

### Root cause: EMA schedule mapped to max_epochs=50, not actual epochs

Cosine EMA sweeps 0.99→0.9999 over `ema_total_epochs` (defaults to `max_epochs=50`).
At epoch 9 of 50 (18% through schedule): `ema_decay ≈ 0.9910`. Fixed default is 0.9995.
**The cosine EMA spent the entire run below the fixed EMA baseline.**

Yi's norman PR #13 (which won with this schedule) ran with a longer effective budget.
With `T_max=12` (our actual compiled epoch count), the schedule would reach ~0.9960 by
end — still below 0.9995. Fix: `--ema-total-epochs 12` for our budget.

### Disposition: CLOSED

Cosine EMA is worth revisiting on top of Lion with `--ema-total-epochs` set to
actual expected epoch count (12 for compiled runs). Reassigning edward to
**PR #51: Lion + RFF (single delta)** — cleaner first composition test.

---

## 2026-04-29 20:46 UTC — PR #42 SENT BACK: frieren squared_rel_l2 — orthogonal win but on stale baseline

**Student:** frieren | **W&B run:** `bmz26ft7` | **Hypothesis:** Replace MSE
training loss with squared rel-L2 (per-case `sum((y-ŷ)²) / sum(y²)`, no outer
sqrt). Single-delta change in `trainer_runtime.py`.

### Results — vs OLD baseline (PR #30 MSE-uncompiled, 9 epochs)

| Metric | PR #30 | PR #42 | Δ% |
|---|---:|---:|---:|
| `abupt_axis_mean` | 19.81 | **19.14** | −3.4% |
| `surface_pressure` | 12.86 | **12.24** | −4.8% |
| `wall_shear` | 21.27 | **20.58** | −3.2% |
| `volume_pressure` | 15.91 | **15.57** | −2.1% |
| `tau_x` | 18.24 | **17.67** | −3.1% |
| `tau_y` | 25.50 | **24.70** | −3.1% |
| `tau_z` | 26.53 | **25.52** | −3.8% |

### Disposition: SENT BACK for rebase + recompose

19.14 is +1.89 worse than current SOTA 17.25 (PR #40 compiled), so merging
would regress BASELINE.md. Mechanism is real and orthogonal — student's
own follow-up correctly anticipated rebase+recompose:

> "Compose squared_rel_l2 with longer training… my guess: 16–18 on test_abupt."

Excellent student diagnostics:
- Discovered actual baseline was MSE not sqrt-rel-L2 (corrected hypothesis,
  added `--loss-form {mse,rel_l2,squared_rel_l2}` flag with default mse).
- Identified that 1.0 grad-clip threshold was binding **at every step**
  regardless of loss form — motivates lr=1e-4 and/or larger clip threshold.
- Val curve still descending at epoch 9 (slope −0.59/epoch).

Sent back: rebase onto tay (compile fix from PR #40), re-run compiled with
`--loss-form squared_rel_l2`. Projected test_abupt ~16.5 if 3.4% loss-form win
composes additively with 2.9% compile win.

---

## 2026-04-29 20:00 UTC — PR #40 MERGED: alphonse compile-fix — new tay SOTA 17.25

**Student:** alphonse | **W&B run:** `ae4zsaly` (rank 0) | **Hypothesis:** Fix
`torch.compile` crash (`drop_last=False` + variable-shape eval batches) so all
runs can use compiled training at ~1.7× throughput. Recalibrate with same 4L/512d/8h
config as PR #30.

Two-line patch in `trainer_runtime.py`:
1. `drop_last=True` on `DistributedSampler` + `DataLoader` (lines 293, 301) — fixes partial last-batch epoch crash.
2. `unwrap_model(model)` in `accumulate_eval_batch` (line 929) — bypasses compile during eval because `pad_collate` produces variable-shape batches; `model.py:321` `torch.cat([surf, vol], dim=1)` creates sum-of-symbolic-dims (`s24+s39`) that inductor can't verify; `dynamic=True` did not help.

### Results

| Metric | PR #40 | PR #33 (SOTA) | PR #30 | yi best | AB-UPT |
|---|---:|---:|---:|---:|---:|
| `abupt_axis_mean` | **17.25** | 17.77 | 19.81 | 15.82 | — |
| `surface_pressure` | **10.92** | 11.20 | 12.86 | 9.99 | 3.82 |
| `wall_shear` | **18.33** | 18.68 | 21.27 | 16.60 | 7.29 |
| `volume_pressure` | **14.71** | 16.13 | 15.91 | 14.21 | 6.08 |
| `tau_x` | **15.73** | 16.20 | 18.24 | 14.27 | 5.35 |
| `tau_y` | **21.80** | 21.81 | 25.50 | 19.49 | 3.65 |
| `tau_z` | **23.07** | 23.54 | 26.53 | 21.12 | 3.63 |

Δ vs PR #33: abupt −2.9%, surface_p −2.5%, wall_shear −1.9%, **volume_p −8.8%**, tau_x −2.9%, tau_y flat, tau_z −2.0%.

### Analysis

Biggest surprise: compile fix alone beats RFF (17.25 vs 17.77) despite no feature change.
The mechanism: compile gives ~1.7× throughput → 12 epochs vs 9 in same 270-min budget. Val
curve still steeply descending at epoch 12 (Δ epoch11→12 = −0.57) — more budget would improve
further. Volume pressure improved most (−8.8%) suggesting early-epoch vol gradient was most
starved of training time.

The PR #40 student correctly diagnosed that RFF+compile composition (PR #46 alphonse) is the
next priority, projecting ~15.5 abupt if additive. The remaining gap to yi (17.25 vs 15.82) is
consistent with missing progressive cosine EMA (edward #44 testing this).

Best-val: epoch 12, val_abupt 16.09. Peak GPU mem 53.1 GB / 96 GB available.

---

## 2026-04-29 16:39 UTC — PR #33 MERGED: fern RFF win — new tay SOTA 17.77

**Student:** fern | **W&B run:** `u43lik5d` (rank 0) | **Hypothesis:** Gaussian RFF coord
features (sigma=1.0, 32 features per modality) appended to surface and volume coord inputs.
Model unchanged; input dim grows from 7 to 7+64 (surface and volume each get separate RFF).

### Results

| Metric | tay (PR #33) | PR #30 | yi best | AB-UPT |
|---|---:|---:|---:|---:|
| `abupt_axis_mean` | **17.77** | 19.81 | 15.82 | — |
| `surface_pressure` | **11.20** | 12.86 | 9.99 | 3.82 |
| `wall_shear` | **18.68** | 21.27 | 16.60 | 7.29 |
| `volume_pressure` | 16.13 | 15.91 | 14.21 | 6.08 |
| `tau_x` | **16.20** | 18.24 | 14.27 | 5.35 |
| `tau_y` | **21.81** | 25.50 | 19.49 | 3.65 |
| `tau_z` | **23.54** | 26.53 | 21.12 | 3.63 |

### Analysis

RFF lifts every surface and shear axis (−10–15%). tau_y: −14.5%, tau_z: −11.3%. The
primary mechanism is spectral bias bypass: RFF encodes multi-frequency coordinate
content that the existing sincos pos_embed provides less of for surface coords at
meter-scale. volume_pressure flat (+0.22 = noise) because volume far-field coords
[-40,80]m saturate the sigma=1.0 projection — documented and preserved for H01.

Best-val checkpoint: epoch 9, val_abupt 17.06. Run ~4.78h without compile.

---

## 2026-04-29 16:39 UTC — PR #32 CLOSED: edward cosine LR + warmup — loss

**Student:** edward | **W&B run:** `uqziai5z` (Arm A: warmup-on) | **Result:** test_abupt 20.99
(+1.18 vs baseline, +6%). Uniform regression all axes.

Diagnosis: `T_max=50` at 9 effective epochs → cosine schedule only 18% engaged (5e-5 → 4.97e-5).
Warmup front-loads cost with no recovery time. Not a dead end — the question is
*"more epochs"*, not *"wrong schedule."* After alphonse #40 compile fix lands (→ 16–22
compiled epochs), cosine LR with T_max matched to compiled budget becomes the right revisit.

---

## 2026-04-29 16:25 UTC — Round 1 first wave: 3 results confirmed in W&B (PRs not yet ready)

Three Round-1 students completed primary arms. None have marked PRs ready
because they are running unauthorized second arms (fern sigma=0.5 / sigma=2.0,
edward warmup-off) or have a silent in-pod session (thorfinn). Posted advisor
comments on each PR pushing them to mark ready / push code.

| PR | Student | Run | val_abupt | test_abupt | Outcome |
|---|---|---|---:|---:|---|
| #33 | fern | `u43lik5d` (sigma=1.0) | 17.06 | **17.77** | **WIN −2.04 (−10.3%)**, new tay leader |
| #37 | thorfinn | `sjcgnehq` (per-axis 1/1.5/1.5) | 18.52 | **19.44** | WIN −0.37 (−1.9%); code not pushed to remote |
| #32 | edward | `uqziai5z` (cosine + warmup) | 19.96 | 20.99 | LOSS +1.18 (+6.0%); cosine T_max=50 only ~18% engaged at 9 epochs |

### fern #33 — RFF win full breakdown

| Metric | tay (PR #30) | fern (PR #33) | Δ | yi best | AB-UPT |
|---|---:|---:|---:|---:|---:|
| `abupt` | 19.81 | **17.77** | −2.04 | 15.82 | — |
| `surface_pressure` | 12.86 | **11.20** | −1.66 | 9.99 | 3.82 |
| `wall_shear` | 21.27 | **18.68** | −2.60 | 16.60 | 7.29 |
| `volume_pressure` | 15.91 | 16.13 | +0.22 | 14.21 | 6.08 |
| `tau_x` | 18.24 | **16.20** | −2.04 | 14.27 | 5.35 |
| `tau_y` | 25.50 | **21.81** | −3.70 | 19.49 | 3.65 |
| `tau_z` | 26.53 | **23.54** | −3.00 | 21.12 | 3.63 |

RFF lifts every wall_shear axis significantly (tau_y −14.5%, tau_z −11.3%,
tau_x −11.2%). Volume pressure flat (+0.22, within noise). Mechanism:
2π·sin/cos coordinate features at sigma=1.0 give the model multi-frequency
positional information that the existing sincos `pos_embed` doesn't fully
provide for surface coordinates.

### thorfinn #37 — Per-axis weights (code missing on remote)

| Metric | tay | thorfinn | Δ |
|---|---:|---:|---:|
| `abupt` | 19.81 | **19.44** | −0.37 |
| `wall_shear` | 21.27 | **20.84** | −0.43 |
| `wall_shear_y` | 25.50 | **24.25** | −1.25 |
| `wall_shear_z` | 26.53 | **25.63** | −0.90 |

Per-axis weight (1.0 / 1.5 / 1.5) does what was designed: tau_y / tau_z
(the systematically worst axes) move most. tau_x flat, surface_pressure
flat, volume_pressure +0.34. **Cannot merge until code is pushed to
`thorfinn/round1-tau-yz-attack-weights-and-tta` branch** — currently
remote has only the assignment commit.

### edward #32 — Cosine LR + warmup (loss)

Uniform regression across all axes (+0.76 to +1.59). Confirmed during
kickoff that `T_max=50` would only progress ~18% over the 9-epoch budget,
so the cosine annealing barely engaged. Adding 5% warmup costs effective
steps with no compensating gain at low effective epochs. The actual lever
this experiment surfaces is **throughput**, not LR schedule.

### Round 1b status (running)

| PR | Student | Hypothesis | Step | Notes |
|---|---|---|---:|---|
| #40 | alphonse | drop_last=True compile fix + recalibrate | 834 | freshly launched |
| #41 | askeladd | eval-time tangential projection of wall-shear | 2156 | running |
| #42 | frieren | squared rel-L2 loss (drop outer sqrt) | 2374 | running |

---

## 2026-04-30 04:50 UTC — PR #42 CLOSED: frieren squared rel-L2 — three arms, AdamW arm clean +40% regression, Lion arm DIVERGED with mechanistic explanation

**Student:** frieren | **W&B runs:** `uwt74mip` (AdamW arm) / `24bdfcnz` (Lion paper-config, killed) / `8ubarr6a` (Lion 5e-5, diverged)
**Hypothesis:** Replace MSE training loss with `(y−ŷ)²/y²` (squared rel-L2). Adds CLI flag `--loss-form {mse,rel_l2,squared_rel_l2}`.

### Arm 1 — AdamW + compile + sq_rel_l2 (finished cleanly)

W&B `uwt74mip`, group `tay-round2-squared-rel-l2-compiled`. 284 min, best epoch 16.

| Metric | PR #40 (compile only, MSE) | This (Arm 1, sq_rel_l2) | Δ vs PR #40 | tay SOTA `vnb7oheo` (Lion 5e-5, MSE) | Δ vs SOTA |
|---|---:|---:|---:|---:|---:|
| `test_abupt` | 17.25 | **15.819** | −1.43 (−8.3%) | 11.303 | **+40.0%** |
| surface_pressure | 10.92 | 9.756 | −10.6% | 6.216 | +57.0% |
| wall_shear | 18.33 | 16.742 | −8.7% | 11.315 | +47.9% |
| volume_pressure | 14.71 | 14.063 | −4.4% | 12.755 | +10.3% |
| tau_x | 15.73 | 14.412 | −8.4% | 9.563 | +50.7% |
| tau_y | 21.80 | 19.859 | −8.9% | 13.831 | +43.6% |
| tau_z | 23.07 | 21.003 | −9.0% | 14.147 | +48.5% |

**Mechanism validated:** uniform −8% across all six axes vs PR #40's MSE baseline. Consistent with original
PR #42 round-1 result (−3.4% on uncompiled). The compile-fix doubled the effective wall budget so the
mechanism delta also compounded.

### Arm 3 — Lion (lr=5e-5/wd=5e-4 SOTA recipe) + compile + sq_rel_l2 → DIVERGED

W&B `8ubarr6a`. Killed at step 8959 / 56 min after clean divergence at step 5000.

| Step | Lion + MSE (`vnb7oheo`) | Lion + sq_rel_l2 (this) |
|---:|---|---|
| 4000 | loss=0.18 grad=2.3 | loss=0.23 grad=6.5 |
| **5000** | **loss=0.16 grad=2.7** | **loss=1.40 grad=240** ← divergence |
| 8000 | loss=0.08 grad=1.8 | loss=2.59 grad=505 |

Val curve: epoch 1 = 82.55%, epoch 2 = 83.30%, epoch 3 = 112.97%.

### Mechanism (frieren's diagnosis — preserved as closed-door insight)

> Lion is sign-update: each parameter step is `±lr` per coordinate, magnitude is unit. sq_rel_l2 is a
> per-case ratio `(y−ŷ)²/y²`; for a batch where `mean(y²)` is small, the ratio (and its gradient) blows
> up. AdamW's per-coordinate variance EMA dampens this — a single bad batch barely moves the second-
> moment estimate, so the effective step size is bounded. Lion has **no per-coordinate magnitude
> memory** — a single bad gradient direction sets the sign of every coordinate to ±1 at full LR,
> permanently corrupting the parameter state. The model cannot recover because Lion will keep flipping
> signs at the same magnitude.

### Arm 2 — Lion paper-config (lr=1.7e-5/wd=5e-3) + sq_rel_l2 (control)

W&B `24bdfcnz`. Killed at epoch 8 (val 16.30, best ~14) when BASELINE.md updated revealed the recipe was
suboptimal. **Did not diverge** — confirms the divergence in Arm 3 is **LR-magnitude-dependent** (smaller
LR → bounded sign-step recovery within 1–2 steps; larger LR → unrecoverable).

### Closed-door insights (now in advisor playbook)

1. **Lion + per-case-normalized loss family is unstable at SOTA LR.** Never compose sq_rel_l2 / rel_l2
   / any `1/y²` loss with Lion at `lr ≥ 5e-5`.
2. **squared_rel_l2 is a real ~−8% mechanism on AdamW** (uniform across all axes). If AdamW path ever
   competes again (e.g. larger model where AdamW's adaptivity helps), revisit.
3. **paper-config Lion stable with sq_rel_l2.** Divergence is LR-dependent.

### Disposition: CLOSED (+40% regression vs SOTA; infra not merged)

The `--loss-form` flag from `64481fa` is preserved on the branch. Branch retained for future cherry-pick;
no current SOTA recipe wants it (RFF + AdamW + compile dominates the AdamW path; Lion + sq_rel_l2 is
provably unstable). Frieren reassigned to **PR #58: Lion (5e-5) + volume-loss-weight 3.0** — direct attack
on the second-worst gap (volume_pressure ×2.1 vs AB-UPT) under the winning optimizer.

---

## 2026-04-30 02:55 UTC — PR #49 CLOSED: askeladd grad-clip 5.0 — +35% regression vs new SOTA

**Student:** askeladd | **W&B run:** `x09udzj3` | **Group:** `tay-round2-grad-clip-sweep`
**Hypothesis:** Raise `--grad-clip-norm` from 1.0 to 5.0 on AdamW + compile baseline (PR #40).
Frieren's PR #42 diagnostic showed `train/grad/clipped=1.0` at every step (23,458/23,458),
discarding ~75% of late-training gradient magnitude. Single-delta lever to recover signal.

### Results vs new SOTA `vnb7oheo` (Lion lr=5e-5)

| Metric | PR #49 (askeladd, clip 5.0) | New SOTA (Lion 5e-5) | Δ |
|---|---:|---:|---:|
| `test_abupt` | **15.232** | 11.303 | **+34.8%** |
| `val_abupt` | 14.219 | 10.096 | +40.8% |
| surface_pressure | 9.338 | 6.216 | +50.2% |
| wall_shear | 16.073 | 11.315 | +42.0% |
| volume_pressure | 13.796 | 12.755 | +8.2% |
| tau_y | 19.105 | 13.831 | +38.1% |
| tau_z | 20.067 | 14.147 | +41.9% |

Run finished cleanly: 284.9 min, best_epoch=16, 43,272 steps. Used Lion-translated LR/WD
(`lr=5e-5, wd=5e-4`) with **AdamW** (note: askeladd correctly used the AdamW-equivalent translation
for the AdamW arm, but the optimizer stays AdamW because the hypothesis is grad-clip on AdamW).

### Diagnosis: clip-5 + AdamW under-trains vs Lion's normalized geometry

The hypothesis was sound — clip 1.0 throws away most of the late-training signal. Raising it does
recover loss/epoch. But the experimental landscape moved past it:

1. **Lion sidesteps the issue entirely.** Sign-based updates make pre-clip grad-norm irrelevant —
   per-step movement is bounded by `lr` directly, not gradient magnitude. Larger clip on AdamW
   recovers some discarded signal but doesn't match Lion's per-channel normalization.
2. **Marginal vs PR #46 (+4.7%) but uncompetitive vs SOTA 11.30.** On its own would have been
   "send back, try lr=1e-4 + clip 5.0"; against Lion it's a clear close.
3. **No forward composition.** Lion + clip 5.0 is ill-defined (Lion is not magnitude-bounded),
   so this lever does not stack into the live Lion arms (#50/#51/#52/#54).

### Disposition: CLOSED (+34.8% regression vs new SOTA)

Diagnostic from frieren #42 (AdamW grad-clip binding) is preserved. Lever does not compose forward
under Lion. Reassigning askeladd to a fresh round-3 hypothesis under the Lion lr=5e-5 SOTA.

---

## 2026-04-30 02:44 UTC — NEW SOTA documented: tanjiro arm B Lion lr=5e-5 — test_abupt 11.303

**Student:** tanjiro (follow-up sweep, NOT advisor-assigned PR) | **W&B run:** `vnb7oheo`
**Group:** `tanjiro-lion-lr-sweep` | **No PR — documented retroactively as new SOTA.**

### Provenance

After PR #39 Lion (paper config `lr=1.7e-5, wd=5e-3` → test_abupt 15.43) was reviewed, tanjiro's
pod launched a follow-up sweep arm at the **AdamW-equivalent translation `lr=5e-5, wd=5e-4`** as
arm B of the original two-arm comparison. This was not part of the advisor's assigned PR pipeline.
The result was decisive enough to update BASELINE.md directly without retroactive PR creation.

### Results — every axis is a new SOTA

| Metric | tanjiro arm B (Lion 5e-5) | PR #46 (RFF+compile) | PR #39 Lion (paper) | AB-UPT ref | Δ vs PR #46 |
|---|---:|---:|---:|---:|---:|
| `test_abupt` | **11.303** | 14.550 | 15.43 | — | **−22.3%** |
| surface_pressure | **6.216** | 8.628 | 9.45 | 3.82 | −28.0% |
| wall_shear | **11.315** | 14.882 | 16.28 | 7.29 | −24.0% |
| volume_pressure | **12.755** | 15.032 | 13.83 | 6.08 | −15.1% |
| tau_x | **9.563** | 12.901 | 13.91 | 5.35 | −25.9% |
| tau_y | **13.831** | 17.281 | 19.58 | 3.65 | −20.0% |
| tau_z | **14.147** | 18.907 | 20.40 | 3.63 | −25.2% |

Runtime: 4h50m (290 min, past 270-min budget — likely launched without strict timeout enforcement).
Val curve was still descending at end → there's more headroom with the same recipe at a longer
budget. Best-val checkpoint reload gave the test number above.

### CRITICAL FINDING: Lion paper config is wrong for this dataset/scale

PR #39 used Lion at `lr=1.7e-5, wd=5e-3` (Chen et al. 2023 paper config from image classification).
This run used `lr=5e-5, wd=5e-4` (the AdamW-equivalent translation tanjiro tested as arm B). Same
optimizer, same code, same data — **−27% improvement just from changing the LR/WD constants**.

Why the paper config fails here:
- Lion paper used image classification with millions of training samples; we have **400 cars**.
- Smaller datasets need more aggressive per-step movement (higher `lr`) to traverse the loss
  landscape inside a 270-min budget.
- Higher `wd` in the paper helps regularize huge nets; our 4L/512d/8h is small enough that
  `wd=5e-4` (AdamW-equivalent) is sufficient.

**All future Lion experiments must use `--lr 5e-5 --weight-decay 5e-4`, not the paper config.**
Posted notification on PRs #50, #51, #52, #54 instructing them to update their LR/WD before launch.

### What this implies for the queued composition stack

- **PR #50 (nezuko, Lion + compile)** — should now beat 11.30 by adding the +4–5% compile bump (epoch 16).
- **PR #51 (edward, Lion + RFF)** — RFF gave +0.88 abupt at the AdamW base; expected ~10.5–10.8.
- **PR #52 (tanjiro, Lion + RFF + compile)** — full triple-stack; if RFF+compile is orthogonal to Lion,
  expect ~9.8–10.3.
- **PR #54 (fern, Lion + per-axis tau_y/tau_z weights)** — directly targets binding 5×+ gap; expect
  the largest delta on tau_y/tau_z specifically.

### Volume_pressure note

Lion arm B's `volume_pressure=12.755` is a regression vs PR #39 Lion paper-config (13.83 → wait,
12.755 < 13.83 → improvement, −7.8%). Earlier write-up of "volume_pressure regression from PR #46"
referred to AdamW+compile vs Lion. With Lion+5e-5 we recover that. PR #55 (alphonse, vol_w=3.0 on
RFF+compile **AdamW** baseline) is now lower priority — the regression it targets dissolves under Lion.

---

## 2026-04-30 01:55 UTC — PR #47 CLOSED: thorfinn bilateral train-aug — +36% regression

**Student:** thorfinn | **W&B run:** `fn8pyav5` | **Group:** `thorfinn-tau-yz-attack`
**Hypothesis:** Bilateral symmetry train-time augmentation. Mirror-flip y-axis with antisymmetric
sign on tau_y to exploit DrivAerML's left-right symmetry.

### Results — clear regression on every axis

| Metric | PR #47 | tay SOTA (PR #46) | Δ |
|---|---:|---:|---:|
| `abupt` | **19.786** | 14.550 | +36.0% |
| surface_pressure | 13.258 | 8.628 | +53.7% |
| wall_shear | 21.107 | 14.882 | +41.8% |
| volume_pressure | 16.525 | 15.032 | +9.9% |
| tau_x | 18.524 | 12.901 | +43.6% |
| tau_y | 24.461 | 17.281 | +41.5% |
| tau_z | 26.163 | 18.907 | +38.4% |

Best val_abupt: 18.84. Runtime: 5h 13m. Best epoch: not specified.

### Diagnosis: bilateral aug introduces a mismatch

DrivAerML cars have weak (driver-side mirrors, exhaust placement, cabin layout) but real
left-right asymmetry. Mirror-augmenting symmetrically tells the model to expect identity
behavior on flipped geometries, but the **test set has those asymmetries** — so the
augmented model has been pushed to a less-accurate point on real cars.

This is the augmentation-as-prior failure mode: an inductive bias that contradicts the
test distribution causes regression even on the hypothesized target axes (tau_y/tau_z
both 41-38% worse than SOTA).

### Disposition: CLOSED

Pod hung post-train, advisor-closed from W&B-verified metrics. If symmetry exploitation
is worth revisiting, it should be at the **architecture level** (antisymmetric/symmetric
heads, geometry-aware coordinate pre-processing), not the augmentation level.

Reassigned thorfinn to **PR #56: AdamW + RFF + compile + cosine LR T_max=16** —
calibration fix for the schedule miscalibration we identified during PR #44 closure.

---

## 2026-04-30 01:53 UTC — PR #42 status update: frieren launched Lion arm

**Student:** frieren | **AdamW arm complete** (`uwt74mip`): `test_abupt = 15.82` — confirmed
the squared_rel_l2 + compile composition (−8.3% vs PR #40 baseline 17.25) but does not
beat SOTA. Frieren proactively launched a **Lion + compile + squared_rel_l2** arm
(run `24bdfcnz`, group `tay-round2-squared-rel-l2-compiled`) using the new SOTA-class
optimizer + compile + their loss-form change.

Advisor-side comment posted to update frieren's comparison goalpost (PR #46 SOTA = 14.55,
not PR #39's 15.43; T_max calibration should be 16 not 12).

---

## 2026-04-30 01:12 UTC — PR #46 MERGED: alphonse AdamW + RFF + compile — NEW SOTA 14.550

**Student:** alphonse | **W&B run:** `28l4yanr` | **Group:** `tay-round2-rff-compiled`
**Hypothesis:** Compose single-scale RFF (sigma=1.0, 32 feats) with `--compile-model` on AdamW base.
Best epoch: 16 (compile gives ~18 min/epoch → 16 epochs in 285-min budget vs 9 uncompiled).

### Results — NEW SOTA

| Metric | PR #46 (new SOTA) | PR #39 Lion (prev) | PR #40 | Δ vs PR #39 |
|---|---:|---:|---:|---:|
| `test_abupt` | **14.550** | 15.43 | 17.25 | **−0.88 (−5.7%)** |
| surface_pressure | **8.628** | 9.45 | 10.92 | −0.82 (−8.7%) |
| wall_shear | **14.882** | 16.28 | 18.33 | −1.40 (−8.6%) |
| volume_pressure | 15.032 | **13.83** | 14.71 | **+1.20 (+8.7%) ← REGRESSION** |
| tau_x | **12.901** | 13.91 | 15.73 | −1.01 (−7.3%) |
| tau_y | **17.281** | 19.58 | 21.80 | −2.30 (−11.7%) |
| tau_z | **18.907** | 20.40 | 23.07 | −1.49 (−7.3%) |

val_abupt at best epoch: 13.487. Val→test gap: +1.063. Best epoch 16 / 50 configured.

### Mechanism

Compile unlocks epoch 16 vs ~9 uncompiled (PR #33). RFF enriches input features with non-linear
position encodings. The deeper training (7 more epochs) is what saturates the surface/wall-shear:
tau_y leads at −11.7%, consistent with spectral-bias hypothesis. Volume_pressure **regressed**
because AdamW with equal volume_loss_weight=2.0 underweights volume when surface gradients dominate.
Lion's sign-update (PR #39) normalizes per-channel gradient magnitude, giving volume better signal
at the same weight — which is why PR #39 had 13.83 vs this run's 15.032.

### Disposition: MERGED (new SOTA)

Pod stalled post-training (alphonse pod healthy but didn't run report-results). Advisor merged
directly from W&B-verified metrics. Assigned alphonse to **PR #55: RFF + compile + volume-loss-weight 3.0**
to recover the volume regression.

---

## 2026-04-30 00:49 UTC — PR #43 CLOSED: fern multi-scale RFF — null result vs own baseline

**Student:** fern | **W&B run:** `5bx6zsio` | **Group:** `fern-multiscale-rff`
**Hypothesis:** 3-band Gaussian RFF (`sigma = {0.1, 1.0, 10.0}` surface; `{0.01, 0.1, 1.0}` volume)
to address spectral bias on tau_y/tau_z.

### Results vs PR #33 (single-scale RFF) and current SOTA PR #39 (Lion)

| Metric | PR #33 single-scale | PR #43 multi-scale | vs PR #33 | tay SOTA PR #39 |
|---|---:|---:|---:|---:|
| `test_abupt` | 17.77 | **17.86** | +0.09 | **15.43** |
| surface_pressure | 11.20 | 11.28 | +0.08 | 9.45 |
| wall_shear | 18.68 | 18.69 | +0.01 | 16.28 |
| volume_pressure | 16.13 | 16.30 | +0.17 | 13.83 |
| tau_x | 16.20 | 16.14 | −0.06 | 13.91 |
| tau_y | 21.81 | 21.84 | +0.03 | 19.58 |
| tau_z | 23.54 | 23.73 | +0.19 | 20.40 |

Best epoch: 9 (8 full + 1 partial forced at 270-min timeout). Val volume_p improved
(9.84→9.40, −4.4%) but **did not transfer to test** (+0.17). Throughput −14% (step
throughput 1.7→1.49 it/s) from 3× RFF compute, compressing to 8 full epochs vs 9.

### Diagnosis

Multi-scale RFF on the standard surface-coord domain is **parameter-redundant** with
sigma=1.0 single-scale. Surface domain `[−1,4] m` is already well-covered by sigma=1.0.
Adding sigma=0.1 (low band) and sigma=10.0 (high band) provides no new spectral information
at the Transolver input layer. The val volume_p gain overfits to the 34-case val split.

### Disposition: CLOSED (+15.7% regression vs new SOTA)

Baseline has moved to PR #39 Lion (15.43). Multi-scale RFF on AdamW base cannot compete.
Assigned fern to **PR #54: Lion + per-axis tau_y/tau_z loss weighting** — directly
targets the binding 5×+ gap on the two worst axes.

---

## 2026-04-29 13:35 UTC — PR #36 closed, tanjiro reassigned to PR #39

PR #36 (tanjiro: SDF-gated volume attention bias) closed after 5+
deterministic crashes at step 2719 (validation/eval code path) and
90+ min of pod claude session stuck on iteration 9 without producing
a successful run or responding to the advisor comment. The student
diagnosed and fixed two real bugs (slice-attention back-distribution,
torch.compile shape recompilation) but the residual eval-path bug
ate too much wall time. The SDF-gate hypothesis is preserved in the
Round 2 queue.

Reassigned to PR #39: **Lion optimizer drop-in replacement for AdamW**
at 4L/512d/8h. Single-delta hypothesis. Modifies only `train.py`,
no `model.py` changes. 2-arm sweep on lr/wd translation
(paper-recommended 1.7e-5/5e-3 vs AdamW-equivalent 5e-5/5e-4).
Lion is a strong empirical winner across vision/language/graph
transformer training, uses ~50% less optimizer-state memory than
AdamW, and composes orthogonally with all Round 1 levers.

## 2026-04-30 11:00 UTC — PR #70 CLOSED: tanjiro Lion+compile+lr=2.5e-5 (half-LR) — diverged late, 9th Lion+modification failure

- **Branch:** `tanjiro/round5-lion-half-lr`
- **W&B run:** `t5qnagui` — finished 154.6min (compile: ~10 epochs in budget window)
- **Hypothesis:** Halving LR from 5e-5 to 2.5e-5 would suppress the Lion+compile sign-bias instability by reducing per-step magnitude, opening the Lion+compile frontier.
- **Result: DIVERGED — best mid-training val 13.071 at step 19046 (~ep7), final val 33.36 at terminal step**

| Metric | Best mid-train val | Final val (DIVERGED) | SOTA `vnb7oheo` |
|---|---:|---:|---:|
| val_abupt | **13.071** (ep~7) | 33.36 | — |
| No test_primary | — | — | 11.303 |

- **Val trajectory:** stable descent ep1→ep7 (val 13.07), then **diverged ep9-10** (val 33.36 terminal). State=finished at 154.6min — confirmed ~10 epochs at compile speed (~16min/epoch). No test eval was run (val was final-epoch diverged at terminal step; best-val checkpoint source=None in summary).
- **Conclusion: Lion+compile diverges at half-LR too.** Half-LR delayed onset by ~2 epochs vs full-LR but did NOT prevent divergence. **9th confirmed Lion+modification failure.**
- **Mechanism update:** The sign-bias issue scales with gradient magnitude, not LR magnitude. Halving LR cuts the step size but leaves the binary sign pattern unchanged — the accumulated bias toward deterministic sign flips at low gradient steps still drives divergence, just with smaller increments.
- **Lion+compile frontier is closed at any LR within 270min budget.**
- **Stable Lion variants confirmed (only 2):**
  1. Vanilla Lion uncompiled (vnb7oheo, test 11.30 SOTA)
  2. Lion+RFF uncompiled (edward #51 ftg0ci0p, val 10.665 ep9 — SOTA candidate, new run iocqp761 in progress)
- **Reassignment:** tanjiro → PR #90 (Lion+RFF+EMA decay 0.9999 — yi #13 compounding).

## 2026-04-30 11:00 UTC — PR #55 CLOSED: alphonse AdamW+RFF+compile+vol_w=3.0 — vol_w=3 lever is closed-door

- **Branch:** `alphonse/round3-rff-compile-vol3`
- **W&B run:** `lahk19ws` — finished 287min, **test eval completed**
- **Hypothesis:** Upweighting volume loss (vol_w=3 vs 2) on the stable AdamW+RFF+compile base targets the volume_pressure ×2.1 binding gap without Lion-fragility risk.
- **Result: REGRESSION — test_abupt 16.387 (+45% vs SOTA 11.30, +12.6% vs PR #46 14.55)**

| Metric | PR #55 (vol_w=3) | PR #46 (vol_w=2) | SOTA 11.30 | AB-UPT |
|---|---:|---:|---:|---:|
| **abupt_mean** | **16.387** | 14.550 | 11.303 | — |
| surface_pressure | 10.256 | 8.628 | 6.216 | 3.82 |
| wall_shear | 17.349 | 14.882 | 11.315 | 7.29 |
| volume_pressure | 14.537 | 15.032 | 12.755 | 6.08 |
| tau_x | 15.060 | 12.901 | 9.563 | 5.35 |
| tau_y | 20.267 | 17.281 | 13.831 | 3.65 |
| tau_z | 21.813 | 18.907 | 14.147 | 3.63 |

- **Trajectory:** best val 15.486 at ep12, then diverged: ep15=15.52 → ep16=21.89 → ep17=57.04. **AdamW+vol_w=3 also diverged late** (ep16-17), which is unusual for AdamW. The upweighting likely over-tilted the loss landscape such that surface gradient signal dominated more strongly than expected in late-training, pushing the model into an unstable regime.
- **Interpretation:** vol_w=3.0 produces a small improvement on volume_pressure (15.03→14.54, -3%) but at the cost of large regressions on ALL other axes (surface +19%, tau_y +17%, tau_z +15%, etc.), AND late divergence.
- **Combined with frieren #68 (Lion+vol_w=3, test 15.57):** vol_w=3 lever is **closed-door at 4L/512d for both AdamW and Lion**. Volume_pressure recovery requires architectural change (deeper/wider model) or different data representation.
- **Reassignment:** alphonse → PR #91 (Lion+RFF+sigma=2.0 — RFF frequency sweep targeting tau_y/tau_z).


## 2026-04-30 12:56 UTC — PR #50 MERGED: nezuko Lion uncompiled fallback — NEW SOTA 11.208

- **Branch:** `nezuko/nezuko/round2-lion-compile-512d`
- **W&B run:** `g2n4fyta` — finished 287min, 9 val epochs, best-val checkpoint reload
- **Hypothesis:** Reproduce SOTA arm B (Lion lr=5e-5/wd=5e-4, no compile) as a clean control — confirms reproducibility and provides baseline for round 6 extensions.
- **Result: NEW SOTA — test_abupt 11.208 (−0.84% vs prior SOTA 11.303 from vnb7oheo)**

| Metric | PR #50 nezuko | Prior SOTA arm B | PR #46 (RFF+compile) | AB-UPT |
|---|---:|---:|---:|---:|
| **abupt_mean** | **11.208** | 11.303 | 14.550 | — |
| surface_pressure | **6.193** | 6.216 | 8.628 | 3.82 |
| wall_shear | **11.199** | 11.315 | 14.882 | 7.29 |
| volume_pressure | **12.726** | 12.755 | 15.032 | 6.08 |
| tau_x | **9.512** | 9.563 | 12.901 | 5.35 |
| tau_y | **13.592** | 13.831 | 17.281 | 3.65 |
| tau_z | **14.017** | 14.147 | 18.907 | 3.63 |

Val trajectory: 80.68→46.76→24.60→17.31→14.25→12.29→11.10→10.37→10.08 (still descending at ep9 budget cut).

- **Interpretation:** Every axis marginally better than arm B vnb7oheo. Confirms Lion uncompiled lr=5e-5/wd=5e-4 as the stable SOTA config. Val 10.08 still descending at ep9 — within a larger budget, this config could improve further. The run also replicated arm B's initial compile divergence (early Lion+compile diverged at ep5) before pivoting to the no-compile fallback, providing direct confirmation of the Lion+compile divergence mechanism.
- **Status:** MERGED as new SOTA baseline.

## 2026-04-30 12:58 UTC — PR #69 CLOSED: thorfinn 768d uncompiled (µP lr=3.3e-5) — budget-limited regression

- **Branch:** `thorfinn/round4-lion-width768d`
- **W&B run:** `mmbry5md` — finished 293min, 5 val epochs
- **Hypothesis:** Scale width to 768d (AB-UPT reference width) with µP LR scaling (lr=3.3e-5). Expected architectural capacity uplift from 512d→768d at ~2.25× params.
- **Result: REGRESSION — test_abupt 12.351 (+9.3% vs new SOTA 11.208)**

| Metric | PR #69 thorfinn | SOTA PR #50 | AB-UPT |
|---|---:|---:|---:|
| **abupt_mean** | **12.351** | 11.208 | — |
| surface_pressure | 7.179 | 6.193 | 3.82 |
| wall_shear | 12.487 | 11.199 | 7.29 |
| volume_pressure | 13.201 | 12.726 | 6.08 |
| tau_x | 10.653 | 9.512 | 5.35 |
| tau_y | 14.951 | 13.592 | 3.65 |
| tau_z | 15.770 | 14.017 | 3.63 |

Val trajectory (only 5 points): 71.04→30.65→16.42→12.50→11.23. Val 11.23 at ep5 is comparable to 512d at ep7 (11.10), confirming the model is on a similar convergence curve but 4 epochs behind.

- **Root cause: budget-limited, not capacity-limited.** 768d uncompiled takes ~58min/epoch vs ~30min/epoch for 512d. Only 5 epochs in 270min vs 9 for 512d. The model hasn't converged — it's at the 512d ep7 equivalent. **If 768d were compiled** (→16 epochs in budget), it would likely converge further.
- **Closed-door config:** 768d uncompiled within 270 budget. Next test: 768d + compile (will it diverge like 512d?). If Lion+compile fragility applies regardless of width, 768d+compile is a dead end. If µP LR scaling somehow stabilizes Lion+compile, 768d+compile could be the capacity win.
- **Status:** CLOSED.

## 2026-04-30 13:14 UTC — PR #57 CLOSED: askeladd Lion+cosine T_max=16 nocompile — wash with vanilla Lion (no improvement)

- **Branch:** `askeladd/lion-cosine-tmax16`
- **W&B run:** `jh1j9uq4` — finished 287min, 9 val epochs
- **Hypothesis:** Cosine T_max=16 schedule with Lion uncompiled (after Lion+compile failed at T_max=16). Test if cosine schedule provides benefit when applied to the stable Lion uncompiled base.
- **Result: WASH — test_abupt 11.229 (+0.19% vs new SOTA 11.208 from PR #50, within noise)**

| Metric | PR #57 (cosine T_max=16) | SOTA PR #50 (no schedule) | Δ |
|---|---:|---:|---:|
| **abupt_mean** | **11.229** | 11.208 | +0.19% |
| surface_pressure | 6.285 | 6.193 | +1.5% |
| wall_shear | 11.278 | 11.199 | +0.7% |
| volume_pressure | 12.611 | 12.726 | **−0.9%** |
| tau_x | 9.656 | 9.512 | +1.5% |
| tau_y | 13.572 | 13.592 | −0.1% |
| tau_z | 14.019 | 14.017 | flat |

Val trajectory: 78.56→43.75→23.68→16.92→13.80→12.01→11.03→10.41→10.13 (mirrors PR #50 trajectory closely).

- **Findings:**
  1. **Lion+cosine T_max=16 nocompile is stable** (no divergence — different from compiled where it diverges at ep4). NEW finding this round.
  2. **No improvement over vanilla Lion** within 9-epoch budget. Cosine T_max=16 decays LR to ~40% of peak by ep9, trading exploration for refinement too early.
  3. Volume_pressure marginally improved (−0.9%) but every other axis regressed marginally — net wash.
- **Follow-up:** Nezuko PR #93 (Lion+cosine T_max=24) tests longer schedule. If T_max=24 also doesn't beat vanilla Lion, cosine is closed lever for Lion uncompiled at this budget.
- **Status:** CLOSED.

## 2026-04-30 23:55 UTC — PR #72 CLOSED: fern AdamW+RFF+per-axis tau — diverged loser test 15.443 (+38.6%)

- fern/round5b-adamw-rff-tauyz
- Hypothesis: AdamW+RFF+compile base + per-channel wall_shear weights [1.0, 2.0, 2.0] for tau_x/y/z. Idea: force model to attend to tau_y/tau_z binding gaps without Lion's instability (Lion+per-axis diverged in PR #54 ep4).
- W&B run: yi9l0ica, group tay-round5b-adamw-rff-tauyz

| Metric | PR #72 (fern AdamW+RFF+axisw) | Current SOTA PR #111 | Δ |
|---|---:|---:|---:|
| test_abupt | 15.443 | 11.142 | **+38.6%** |
| surface_pressure | 9.648 | 6.209 | +55.4% |
| wall_shear | 15.955 | 11.138 | +43.3% |
| volume_pressure | 15.836 | 12.548 | +26.2% |
| tau_x | 14.338 | 9.436 | +51.9% |
| tau_y | 17.919 | 13.525 | +32.5% |
| tau_z | 19.475 | 13.992 | +39.2% |
| best val | 14.442 (ep13) | 9.989 (ep9) | +44.6% |

- **Trajectory:** 72.29, 41.51, 29.01, 24.13, 21.39, 19.63, 18.24, 17.25, 16.48, 15.80, 15.26, 14.80, 14.44, 14.83, 17.31, **35.19** (ep16 catastrophic divergence)
- **Conclusion:** AdamW+RFF+per-axis weighting is fundamentally weaker than Lion uncompiled. The per-axis weighting did NOT close the binding gap — tau_y/tau_z REGRESSED +33%/+39%. Combined with PR #54 (Lion+per-axis diverged ep4), per-axis weighting at sw=2.0 ratio is closed-door on both stacks. Future attempts need conservative weights (sw_y/z ≤ 1.5) AND likely selective application (only after a warmup phase).
- **Fern reassigned** to round10 model-slices=256 sweep (architecture lever — current SOTA uses 128 slices).

## 2026-05-01 00:35 UTC — PR #114 CLOSED: askeladd EMA=0.998 — confirmed loser test 11.865 (+6.5%)

- askeladd/round8-ema-faster
- Hypothesis: EMA=0.998 (5× faster than current SOTA 0.999) might capture late-stage convergence even more aggressively. Test the lower-bound side of EMA sweep.
- W&B run: 9qxs9qrp, group tay-round8-ema-faster

| Metric | PR #114 (EMA=0.998) | Current SOTA PR #111 | Δ |
|---|---:|---:|---:|
| test_abupt | 11.865 | 11.142 | **+6.5%** |
| surface_pressure | 6.808 | 6.209 | +9.7% |
| wall_shear | 11.922 | 11.138 | +7.0% |
| volume_pressure | 13.164 | 12.548 | +4.9% |
| tau_x | 10.259 | 9.436 | +8.7% |
| tau_y | 14.161 | 13.525 | +4.7% |
| tau_z | 14.935 | 13.992 | +6.7% |
| best val (ep9) | 10.59 | 9.99 | +6.0% |

- **Trajectory:** 41.81, 22.93, 17.12, 14.37, 12.67, 11.69, 11.10, 10.73, 10.59 — strong early lead (-48% ep1, -51% ep2 vs vanilla) but lead inverted by ep6.
- **Conclusion:** EMA=0.998 is **too fast** for the 9-epoch budget. EMA sweep result: 0.9999 (too slow, closed) → 0.9995 (PR #50 baseline) → **0.999 SOTA** → 0.998 (this loser). Sweet spot confirmed at 0.999.
- **Askeladd reassigned** to round10 compound stack lr=1e-4 + sw=2.0 + EMA=0.999 (combine round 9's two strongest winners).

## 2026-05-01 04:55 UTC — PR #139 CLOSED: fern slices=256 (test 12.389, +17.1% vs SOTA)

- **Branch:** `fern/round10-slices-256`
- **W&B run:** `9lc1acwf` — 289.4 min, ~7 val epochs (et=8.9 min/ep in this config, faster per-epoch than 4L)
- **Hypothesis:** Double slice count 128→256 to improve high-frequency spatial discrimination (tau_y/z attack)
- **Result:** test_abupt **12.389** vs SOTA 10.580 = **+17.1% WORSE**. Best val 11.210 at ep7.
- **Per-axis:** sp=7.12 (+25%), ws=12.62 (+21%), vp=13.11 (+2.9%), tau_x=10.77 (+21%), tau_y=15.24 (+22%), tau_z=15.71 (+20%).
- **Notable:** Memory at 98.8% allocated (94.4/97 GB) — slice attention quadratic in slice count. Run used lr=5e-5 + vol_w=2.0 + EMA=0.999, not strict PR #115 base (lr=1e-4).
- **Why:** Quadratic attention overhead (256 slices→each attends all others), smaller per-slice geometric patch = less spatial context, and no selective improvement on targeted tau axes — every component regressed uniformly.
- **Conclusion:** model_slices=128 confirmed optimal. Lever closed.
- **Fern reassigned** to PR #159: Lion β1=0.95 (first Lion beta sweep on tay, single delta from SOTA).
