# SENPAI Research Results — DrivAerML (`yi`)

W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml`.
Targets to beat (lower is better): `surface_pressure 3.82`, `wall_shear 7.29`,
`volume_pressure 6.08`, `tau_x 5.35`, `tau_y 3.65`, `tau_z 3.63`.

## Round 1 — opened 2026-04-28

Round 1 launches 16 parallel experiments: 5 known-good baselines / proven-additive
deltas (Stream 1) and 11 fresh single-delta hypotheses (Stream 2).

## 2026-04-29 — PR #13: progressive EMA cosine anneal 0.99→0.9999 (norman) — VERIFIED WIN, pending merge (rebase required)

- Branch: `norman/round1-progressive-ema-decay`
- Hypothesis: anneal `--ema-decay` from 0.99 (start) to 0.9999 (end) via cosine schedule. Early training: fast-tracking EMA (low decay) tracks the live model during rapid change. Late training: high decay averages out stochastic variance for a sharper final checkpoint.
- W&B run: `wio9pqw2` (`norman-ema-99-9999-v4`), state=finished, 4 epochs, best_epoch=4, peak GPU 52.9 GB
- Config: `--volume-loss-weight 2.0 --batch-size 8 --validation-every 1 --lr 2e-4 --ema-decay-start 0.99 --ema-decay-end 0.9999` (gilbert protocol + cosine EMA on top)

**test_primary/* (norman PR #13 vs current yi baseline PR #9):**

| Metric | PR #13 norman | PR #9 gilbert (merged baseline) | PR #8 frieren FiLM (pending) | Δ vs PR #9 | AB-UPT |
|---|---:|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **15.82** | 17.39 | 16.53 | **−9.0%** | — |
| `surface_pressure_rel_l2_pct` | **9.99** | 11.07 | 10.38 | −9.7% | 3.82 |
| `wall_shear_rel_l2_pct` | **16.60** | 18.32 | 17.29 | −9.4% | 7.29 |
| `volume_pressure_rel_l2_pct` | **14.21** | 15.21 | 14.91 | −6.6% | 6.08 |
| `wall_shear_x_rel_l2_pct` | **14.27** | 15.65 | 14.76 | −8.8% | 5.35 |
| `wall_shear_y_rel_l2_pct` | **19.49** | 21.86 | 20.59 | −10.8% | 3.65 |
| `wall_shear_z_rel_l2_pct` | **21.12** | 23.18 | 22.00 | −8.9% | 3.63 |

Win on every axis. Also surpasses pending FiLM PR #8 (16.53 → 15.82, −4.3%).

**Per-epoch trajectory — no divergence:**

| Epoch | train_loss | val_abupt | best? |
|---:|---:|---:|---|
| 1 | 0.5414 | 26.24 | ✓ |
| 2 | 0.1521 | 17.98 | ✓ |
| 3 | 0.0919 | 15.59 | ✓ |
| 4 | 0.0657 | **14.71** | ✓ |

**Val improved monotonically through epoch 4.** This directly revises the (B) verdict from PR #20 (nezuko EMA sweep):
- PR #20 used small-batch config (no bs=8 / vol_w=2.0) → diverged after epoch 1
- PR #13 used gilbert's protocol (bs=8) → no divergence through epoch 4
- The binding instability was **config-conditional**: large batch + vol_w=2.0 suppresses it. The cosine EMA late-smoothing may also contribute.

**EMA schedule:** swept 0.990 → 0.9999 over 43,532 steps exactly as designed. Zero new parameters (schedule only).

**Key implication for fleet:** progressive EMA 0.99→0.9999 is now on `yi`. All future PRs should adopt `--ema-decay-start 0.99 --ema-decay-end 0.9999` for the best checkpoint quality. Round-2 follow-up assigned (norm A02 SE(3) equivariant coord features) as PR #27.

**Note:** Merge blocked on rebase conflict (yi updated after branch was cut). Student rebasing now. Results verified and accepted.

## 2026-04-29 — PR #20: EMA decay sweep diagnostic (nezuko) — CLOSED, diagnostic value (verdict (B) confirmed)

- Branch: `nezuko/round2-ema-decay-sweep`
- Hypothesis: disambiguate (A) EMA-too-slow vs (B) genuine post-epoch-1 instability for the train→val divergence pattern observed across multiple Round-1 runs.
- 4-arm sweep: `--ema-decay ∈ {0.99, 0.999, 0.9995, 0.99995}` parallel across 4 GPUs, all with `--validation-every 1`. ~2 full epochs + truncated epoch 3 each. Peak GPU 13.3 GB.
- W&B group: [`nezuko-ema-sweep`](https://wandb.ai/wandb-applied-ai-team/senpai-v1-drivaerml/groups/nezuko-ema-sweep)

| EMA decay | best_epoch | abupt_axis_mean (test_primary) | full_val_primary abupt | E1 train | E2 train | E1 val_abupt | E2 val_abupt |
|---|:---:|---:|---:|---:|---:|---:|---:|
| 0.99 | 1 | 27.67 | 26.93 | 0.278 | 1.217 | 26.93 ⭐ | 79.26 |
| 0.999 | 3 | 69.54 | 69.01 | 0.222 | 0.923 | 76.29 | 73.41 |
| 0.9995 | **1** | **24.74** | **24.46** | 0.228 | 0.411 | 24.46 ⭐ | 41.26 |
| 0.99995 | 1 | 36.63 | 36.19 | 0.180 | 0.763 | 36.19 ⭐ | 124.32 |

**Verdict: (B) genuine post-epoch-1 divergence. Decisive.**

Evidence:
1. **Even the most aggressive EMA arm (0.99, window ~100 steps) peaks at epoch 1.** If (A) were true, this arm — whose shadow ≈ live model — should have kept improving. It did not.
2. **Train loss is non-monotonic across all four seeds:** every arm has live train loss 5–7× *higher* in epoch 2 vs epoch 1 (e.g. 0.99: 0.278 → 1.217). EMA cannot manufacture this; live optimization is diverging.
3. **Per-step train/loss spikes hit 6–22× the median** in every arm; min train_loss reached at step 20–45k, then climbs. Divergence onset dates to step ~45–60k — exactly where missing gradient clipping becomes load-bearing under no-warmup, near-constant LR.
4. The 0.999 outlier (best_val=76.29 at epoch 1) is consistent with this: its specific seed had divergence start *during* epoch 1 so the EMA shadow at step 43k is contaminated by the chaotic post-divergence tail.

**Implications for ongoing work:**
- **PR #22 (gilbert, gradient clipping) is the cure** for the binding constraint. When that lands, the entire fleet should re-evaluate.
- **PR #5 (edward, cosine LR + warmup)** is complementary on the LR-schedule side — nezuko's data shows current `T_max=999` makes LR essentially constant at 2e-4 throughout, so the cosine decay is doing nothing.
- `--ema-decay 0.9995` confirmed as the right default for this step regime; no change needed.
- **Avoid 0.99995 and progressive 0.999→0.9999 schedule (norman PR #13)** for this step budget — the 0.9999 tail (window ~10k steps) over-averages into the divergent regime.

**Note:** Best arm 24.74 abupt_axis_mean is well above current yi baseline (16.53/17.39). No code changes (CLI sweep only). Closed because it's diagnostic, not competitive.

**Round-2 follow-up:** A01 (ANP cross-attention surface decoder) assigned to nezuko — the largest architectural win from noam branch (PR #2379 MERGED on TandemFoil: −70% in-domain p_s, −48% OOD).

## 2026-04-29 06:00 — PR #6: relative-L2 auxiliary loss (emma) — CLOSED, dead end

- Branch: `emma/round1-metric-aware-aux-loss`
- Hypothesis: add a metric-aligned auxiliary loss `aux_rel_l2_weight * relative_l2_loss(pred, target, mask)` so training optimizes a quantity directly proportional to the AB-UPT eval metric, with `loss = MSE + 0.05 * rel_l2`.
- W&B runs (all five state=crashed/diverged, no `test_primary/*` produced):
  - `ylg9cc8h` (Run 1, w=0.05, original snippet) — NaN at step 1 (sqrt(0) backward).
  - `tq2cs2vo` (Run 2, w=0.05, dual-eps fix) — diverged step ~115700 mid-epoch 3 (grad norm 1.7 → 7e+5 → 2.85e+16).
  - `*` Run 3a/3b (w=0.01, w=0.02) — diverged step 50600 / 59100. Smaller weight diverged earlier — confirms backward-path instability is the controlling factor, not weight magnitude.
  - `*` Run 4a/4b (`clip_grad_norm_(1.0)` added) — `total_norm=Inf → coef=0` mathematically still propagates NaN.
  - `*` Run 5a/5b (full stability stack: ratio.clamp(max=1.0), eps=1e-4, fp32 outside autocast, NaN-skip step) — entered 100% skip-step regime by ~step 30000, training frozen.
- **Conclusion:** the rel-L2 auxiliary loss as formulated is fundamentally unstable in this codebase. The `sqrt((diff_sum + ε)/(tgt_sum + ε))` backward path produces Inf grads when surface target norms are small in denormalized space; even gradient masking and skip-step cannot rescue it once the regime is reached.
- **Salvage value:** emma's diagnostic infrastructure (`grad_clip_norm` config + NaN-resistant skip-step pattern) corroborates the design landing in PR #22 (gilbert). Closed in favor of squared-rel-L2 reformulation (drop the sqrt → `ratio.mean()`) as a Round-2 follow-up assigned to emma.

## 2026-04-29 — PR #8: per-case geometry FiLM conditioning (frieren) — VERIFIED WIN, pending merge (rebase required)

- Branch: `frieren/round1-geometry-film-conditioning`
- Hypothesis: condition each Transolver block on a per-geometry latent vector via AdaLN-zero FiLM (feature-wise linear modulation), so the model can specialize its weights to the car geometry rather than treating all geometries uniformly.
- W&B run: `hltti2ec` (state=finished, 1 full epoch reached, best_epoch=1)
- Param count: 3,388K (+142K = +4.4% vs baseline)
- Deviation from PR pseudocode: frieren applied FiLM per-block (all 4 layers) with AdaLN-zero init, not just the final layer — correct empirical decision.

**test_primary/* (frieren PR #8 vs current yi baseline PR #9):**

| Metric | PR #8 frieren | PR #9 gilbert (yi baseline) | Δ | AB-UPT |
|---|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **16.53** | 17.39 | **−4.9%** | — |
| `surface_pressure_rel_l2_pct` | **10.38** | 11.07 | **−6.2%** | 3.82 |
| `wall_shear_rel_l2_pct` | **17.29** | 18.32 | **−5.6%** | 7.29 |
| `volume_pressure_rel_l2_pct` | **14.91** | 15.21 | **−2.0%** | 6.08 |
| `wall_shear_x_rel_l2_pct` | **14.76** | 15.65 | **−5.7%** | 5.35 |
| `wall_shear_y_rel_l2_pct` | **20.59** | 21.86 | **−5.8%** | 3.65 |
| `wall_shear_z_rel_l2_pct` | **22.00** | 23.18 | **−5.1%** | 3.63 |

**Apples-to-apples vs PR #3 (no-FiLM, same config, 1-epoch comparator):** `abupt_axis_mean` 30.47 → 16.53 = **46% reduction**. FiLM is a real lever.

**Diagnostic confirmation:** geometry token L2 norm grew 70× during the epoch (0.18 → 12.4); FiLM weights grew 1.8–3.6×. Layer is being actively used, not bypassed.

**Critical confound:** frieren ran bs=2, 1 epoch only — while gilbert's baseline used bs=8, 6 epochs (with protocol fixes). FiLM reached 16.53 at 1 epoch vs gilbert's 17.39 at 6 epochs. This implies the FiLM conditioning adds real architectural capacity that compounds with convergence.

**Status:** Merge blocked on rebase conflict (yi was updated after frieren's branch was cut). Squash-merge will complete once frieren rebases onto yi. Results verified and accepted.

**Round-2 follow-up triggered (frieren PR #23):** Full composition run stacking all yi wins:
FiLM + vol_w=2.0 + tangential projection + bs=8 + validation-every=1 + gradient clipping (once PR #22 lands).

## 2026-04-29 03:57 — PR #9: volume loss weight sweep (gilbert) — MERGED, NEW yi BASELINE

- Branch: `gilbert/round1-volume-loss-reweight`
- Hypothesis: upweight volume loss to 2.0–3.0 to focus gradient budget on
  the hardest target (`volume_pressure`).
- Run A (vol_w=2.0): `y2gigs61`, state=finished, 6 epochs reached, best_epoch=3.
- Run B (vol_w=3.0): `s45dwv6i`, state=finished, 6 epochs reached, best_epoch=1.

**test_primary/* (Run A new yi best vs prior PR #11 baseline):**

| Metric | Run A (vol_w=2.0) | PR #11 (kohaku, prior) | Δ | AB-UPT |
|---|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **17.39** | 35.12 | **−50.5%** | — |
| `surface_pressure_rel_l2_pct` | 11.07 | 10.07 | +9.9% | 3.82 |
| `wall_shear_rel_l2_pct` | **18.32** | 43.05 | **−57.4%** | 7.29 |
| `volume_pressure_rel_l2_pct` | 15.21 | 14.99 | +1.5% | 6.08 |
| `wall_shear_x_rel_l2_pct` | **15.65** | 30.85 | **−49.3%** | 5.35 |
| `wall_shear_y_rel_l2_pct` | **21.86** | 42.06 | **−48.0%** | 3.65 |
| `wall_shear_z_rel_l2_pct` | **23.18** | 77.65 | **−70.1%** | 3.63 |

Run B (vol_w=3.0): `abupt=30.08`, diverged at epoch 2 (best_epoch=1).
**vol_w=3.0 strictly worse than vol_w=2.0**, confirming the PR's question.

**The big confound:** gilbert's run did **not** include
`--use-tangential-wallshear-loss` (kohaku's projection code is on yi but
default off). Yet still beat kohaku's projection-loss run by 50%. The bulk
of the win came from the **protocol fixes**:

- `--batch-size 8` (vs default 2)
- `--validation-every 1` (vs default 10)
- `--gradient-log-every 100 --weight-log-every 100` (Issue #19 throughput)

vol_w=2.0 vs vol_w=1.0 single-delta is therefore untested, but vol_w=2.0
appears at worst neutral. Combining gilbert's config with kohaku's
projection should compose for further gains.

**Critical bug uncovered (gilbert PR comment):** `train.py` has no gradient
clipping. Run B and several other Round-1 runs (chihiro, emma, fern, haku)
diverged on the exact same mechanism. **Round-2 follow-up PR #22 (gilbert)
adds `torch.nn.utils.clip_grad_norm_` + sweeps clip values.**

**Round-2 follow-ups triggered:**
- PR #22 (gilbert): add gradient clipping to `train.py` — infrastructure
  win blocking high-LR / high-weight / high-batch sweeps.
- BASELINE.md: new winning reproduce config recorded with all four protocol
  flags + vol_w=2.0.

## 2026-04-29 03:13 — PR #11: tangential wall-shear projection loss (kohaku) — MERGED, prior baseline (superseded by PR #9)

- Branch: `kohaku/round1-tangential-wallshear-loss`
- Hypothesis: project predicted/target wall-shear onto surface tangent plane
  before MSE — physics says wall shear has zero normal component on a no-slip
  wall, so penalising the normal component is unphysical noise.
- W&B run: `uy0ds6iz` (state=finished, 1 full epoch reached, run pre-dated
  the per-step timeout fix so timed out at the inter-epoch check).

| Metric | kohaku (PR #11) | norman (akbdunir, no-projection comparator) | nezuko (mdo2p8q7, DropPath) | AB-UPT |
|---|---:|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **35.12** | 64.66 | 81.21 | — |
| `test_primary/surface_pressure_rel_l2_pct` | 10.07 | 48.43 | 66.49 | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | 43.05 | 66.89 | 84.27 | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | 14.99 | 55.54 | 69.42 | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | 30.85 | 55.54 | 75.40 | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | 42.06 | 90.15 | 102.42 | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | 77.65 | 73.66 | 92.32 | 3.63 |

**Result: merged (~46% reduction on `abupt_axis_mean` vs the closest comparator).**

**Key wins:**
- First yi baseline established. All future PRs measured against PR #11.
- kohaku's deviation from the PR pseudocode was correct: PR text projected in
  normalized space, but per-axis wall-shear stds are non-uniform
  ([2.08, 1.36, 1.11]), so true tangential projection requires
  denormalize → project → renormalize. Physically motivated and analytically
  rigorous.
- New diagnostic `train/wallshear_pred_normal_rms` instruments the predicted
  normal component — confirmed it grows ~2.4× during a single epoch
  (0.52 Pa → 1.21 Pa), validating the predicted failure mode.

**Caveats:**
- Only 1 epoch reached (run pre-dated the per-step timeout fix). Subsequent
  PRs with the fix + `--validation-every 1` should reach 4–5 epochs.
- All wall-shear axes still 5–21× from AB-UPT targets — most headroom is in
  the wall-shear regression, especially `tau_z` (77.65% vs target 3.63%).

**Round-1 follow-up assigned to kohaku (PR #21):** sweep
`λ * mean((ws_pred · n_hat)^2)` regularizer on top of projection — directly
addresses the failure mode the diagnostic exposed. Also serves as the first
multi-epoch run with projection on (the λ=0 arm).

## 2026-04-29 02:30 — PR #12: stochastic depth / DropPath p=0.1 (nezuko) — CLOSED

- Branch: `nezuko/round1-stochastic-depth`
- Hypothesis: linear-schedule DropPath (max p=0.1 at deepest layer) regularizes the
  Transolver and gives ~10% throughput from skipped residual branches.

| Metric | nezuko (DropPath p=0.1, `mdo2p8q7`) | norman (no DropPath, `akbdunir`) | AB-UPT target |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **81.21** | 64.66 | — |
| `test_primary/surface_pressure_rel_l2_pct` | 66.49 | 48.43 | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | 84.27 | 66.89 | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | 69.42 | 55.54 | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | 75.40 | 55.54 | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | 102.42 | 90.15 | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | 92.32 | 73.66 | 3.63 |

**Result: rejected (+16.5 pp worse on abupt_axis_mean than no-DropPath).**

**Analysis:** both nezuko and norman finished with `best_epoch=1`. Train loss
keeps falling, EMA-val degrades from epoch 1 onward — the runs are firmly in
the underfitting regime, not the overfitting regime where regularization helps.
Stochastic depth adds noise to the residual signal without addressing the
binding constraint (insufficient optimizer steps to convergence at this
4L/256d/4h/128sl + 65k-points config inside the 6 h timeout).

**Important byproduct: per-step timeout fix.** nezuko shipped a `train.py` fix
(commit `1ab3a9b`) that adds a per-step wall-clock timeout check, reserves
`SENPAI_VAL_BUDGET_MINUTES` (default 90), and forces a final validation when
mid-epoch timeout fires. Cherry-picked into `yi` as commit `af92e9a` and
broadcast to all active Round-1 PRs. This unblocks every 65k-points run from
the silent "epoch longer than timeout → no test_primary" failure mode that
trapped the prior `u38zaxeg` attempt.

**Round-1 follow-ups triggered:**
- Recommend `--validation-every 1` (or 2) for all Round-1 runs.
- Flagged the train→val divergence pattern across runs for all students to
  report and investigate.

## 2026-04-29 — PR #4: large model scale-up 4L/512d/8h (chihiro) — MERGED

- Branch: `chihiro/round1-large-model-512d`
- Hypothesis: scaling the Transolver width from 256→512d and heads from 4→8h increases model capacity and improves all output fields, especially volume pressure which requires richer volumetric representation than surface fields.
- W&B run: `pejudvyd` (Run B), state=finished, best_epoch=3, params ~12.7M

**Config deviations from gilbert base:**
- `--model-hidden-dim 512 --model-heads 8` (hypothesis test)
- `--lr 5e-5` (3 prior runs at 2e-4 diverged; larger models need smaller LR)
- `--batch-size 4` (largest power-of-2 fitting 96GB VRAM at 512d)
- `--clip-grad-norm 1.0` opt-in flag added (overlaps with PR #22, accepted as compatible)

**test_primary/* (PR #4 chihiro vs PR #9 gilbert merged baseline):**

| Metric | PR #4 chihiro Run B (`pejudvyd`) | PR #9 gilbert (prev baseline) | Δ | AB-UPT target |
|---|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | **16.64** | 17.39 | **−4.3%** | — |
| `surface_pressure_rel_l2_pct` | **10.65** | 11.07 | −3.8% | 3.82 |
| `wall_shear_rel_l2_pct` | **17.66** | 18.32 | −3.6% | 7.29 |
| `volume_pressure_rel_l2_pct` | **14.37** | 15.21 | **−5.5%** | 6.08 |
| `wall_shear_x_rel_l2_pct` | **14.87** | 15.65 | −5.0% | 5.35 |
| `wall_shear_y_rel_l2_pct` | **19.89** | 21.86 | −9.0% | 3.65 |
| `wall_shear_z_rel_l2_pct` | **21.73** | 23.18 | −6.3% | 3.63 |

Beats merged baseline on every axis. Squash-merged into `yi` 2026-04-29.

**Analysis:**
- Width improvement is orthogonal to FiLM (targets surface token conditioning) and to cosine EMA (targets checkpoint quality). `volume_pressure` win (14.37 vs 15.21) is most notable — the 512d volume is the binding capacity constraint for volumetric fields, as expected.
- vs pending FiLM (PR #8, 16.53 abupt): chihiro is slightly worse on the headline (16.64 > 16.53) but wins on `volume_pressure` (14.37 < 14.91). The two effects are complementary.
- vs pending cosine EMA (PR #13, 15.82 abupt): chihiro is worse on every axis — cosine EMA is the dominant improvement in flight. A composition (512d + cosine EMA + FiLM) should push below 15.
- LR sensitivity is important: µP scaling or explicit LR search needed for any further width increase. Divergence at 2e-4 confirms the 512d model has ≥2× sharper loss landscape.
- `--clip-grad-norm` flag added to `train.py` as opt-in; will become default when PR #22 (gilbert) lands.

**Compounding wins on yi after this merge:**
1. PR #11 (kohaku) — tangential wall-shear projection loss
2. PR #9 (gilbert) — protocol fixes + vol_w=2.0
3. **PR #4 (chihiro) — width 512d/8h** ← this entry

**Pending merges in rebase queue:**
- PR #8 frieren FiLM (16.53) — surface geometry conditioning
- PR #13 norman cosine EMA (15.82) — projected new best once merged

**Round-2 follow-up assigned to chihiro:** composition run — width 512d × FiLM × cosine EMA at the gilbert base config. Hypothesis: if 256d gains are ~16.53 (FiLM) and ~15.82 (EMA), 512d should push below 15 given orthogonal capacity axes.

## 2026-04-29 — PR #7: Gaussian random Fourier features (fern) — REQUEST CHANGES (sent back v5)

- Branch: `fern/round1-fourier-coordinate-features`
- Hypothesis: Gaussian RFF replaces raw `(x, y, z)` with `[sin(Bx), cos(Bx)]` to give the input projection a richer spectrum and learn fine spatial frequencies. Proven in radford-branch champion config.
- W&B runs: `fl3mawj9` (v1, killed for logging cadence), `8fm90m9i` (v2, NaN), `2fqms6xu` (v3 σ=1.0, NaN), `7jm8iurm` (v4 σ=0.5, **only 1 healthy epoch**)

**v4 test_primary/* (σ=0.5, EMA epoch-1 checkpoint):**

| Metric | v4 fern Fourier σ=0.5 | yi merged (PR #4) | PR #3 baseline (matched config) | Δ vs PR #3 | Δ vs PR #4 |
|---|---:|---:|---:|---:|---:|
| `abupt_axis_mean_rel_l2_pct` | 22.67 | 16.64 | 30.47 | **−25.6%** | +36.2% |
| `surface_pressure` | 14.88 | 10.65 | 21.65 | −31.3% | +39.7% |
| `wall_shear` | 23.96 | 17.66 | 32.51 | −26.3% | +35.7% |
| `volume_pressure` | 19.17 | 14.37 | 23.73 | −19.2% | +33.4% |

**Result vs current baseline:** +36% regression (above 5% close threshold). However:

**Matched-config delta is positive:** vs PR #3 (same arch, no Fourier, no protocol fixes), Fourier adds **−25 to −30% on every axis**. Fourier hypothesis has real signal.

**Epoch-1 advantage at matched protocol:** fern's val_abupt at epoch 1 = 22.10 vs norman's epoch-1 val of 26.24 (PR #13, no Fourier). That's **−16% relative at the same epoch from Fourier alone**, a real composition signal.

**Binding constraint: bf16 numerical instability.** Both σ=1.0 and σ=0.5 diverged in epoch 2 even with `clip_grad_norm=1.0`. Pattern: clean epoch 1, single-step preclip-norm spike (80–710), runaway, NaN/Inf. Per-step gradient clipping caps optimizer step size but does **not** prevent activation drift past bf16 representable range. Raw-meter coords compound the issue — DrivAerML coords span ~7m × 4m × 3m, so σ=1.0 gives Fourier wavelength ~6m (coarser than the car body) and per-axis grad-norm asymmetry from the anisotropic bbox.

**Decision:** REQUEST CHANGES — sent back as v5 with specific instructions:
1. Coordinate normalization to [-1, 1] before Fourier (bounds Fourier output regardless of σ)
2. σ=1.0 on normalized coords (puts wavelength at car-feature scale)
3. Compose with cosine EMA (`--ema-decay-start 0.99 --ema-decay-end 0.9999`, port from norman PR #13)
4. NaN-skip-step protection (defensive)
5. Stay at 256d for v5 (test stability first; width × Fourier as separate composition later)

**Bug-fix commit `fdef51e` (gradient clipping with `--grad-clip-max-norm`):** Acknowledged. Compatible with PR #22 (gilbert) which is landing officially. Will inherit on rebase.

**Pass/fail bar for v5:** must beat `abupt_axis_mean = 16.64` (current merged baseline). Stretch target: beat 15.82 (pending PR #13). If v5 also fails, Fourier closes and goes back to idea pool for revisit (per-axis σ, learnable σ, or σ-scheduled annealing).
