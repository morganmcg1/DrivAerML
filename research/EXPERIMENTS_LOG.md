# SENPAI Research Results

## 2026-04-29 03:00 — PR #11: [kohaku] Tangential wall-shear projection loss
- Branch: `kohaku/round1-tangential-wallshear-loss`
- Hypothesis: Project wall-shear predictions onto the tangential plane at each surface point to enforce the no-slip boundary condition physically.
- Results: W&B run `uy0ds6iz`

| Metric | Value |
|---|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | 35.12 |
| `test_primary/surface_pressure_rel_l2_pct` | 10.07 |
| `test_primary/wall_shear_rel_l2_pct` | 43.05 |
| `test_primary/volume_pressure_rel_l2_pct` | 14.99 |
| `test_primary/wall_shear_x_rel_l2_pct` | 30.85 |
| `test_primary/wall_shear_y_rel_l2_pct` | 42.06 |
| `test_primary/wall_shear_z_rel_l2_pct` | 77.65 |

- Commentary: Established first yi baseline. Only 1 epoch completed due to timeout bug (pre-fix). Projection loss adds physical constraints but alone is insufficient without the protocol fixes.

---

## 2026-04-29 03:57 — PR #9: [gilbert] Protocol fixes (bs=8, vol_w=2.0, validation-every=1)
- Branch: `gilbert/round1-protocol-fixes`
- Hypothesis: Larger batch size and higher volume loss weight stabilize training and improve all metrics.
- Results: W&B run `y2gigs61`

| Metric | Value | Prior (PR #11) |
|---|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **17.39** | 35.12 |
| `test_primary/surface_pressure_rel_l2_pct` | 11.07 | 10.07 |
| `test_primary/wall_shear_rel_l2_pct` | 18.32 | 43.05 |
| `test_primary/volume_pressure_rel_l2_pct` | 15.21 | 14.99 |
| `test_primary/wall_shear_x_rel_l2_pct` | 15.65 | 30.85 |
| `test_primary/wall_shear_y_rel_l2_pct` | 21.86 | 42.06 |
| `test_primary/wall_shear_z_rel_l2_pct` | 23.18 | 77.65 |

- Commentary: 50.5% improvement in headline metric. Flag-only change. Protocol fixes (bs=8, vol_w=2.0) are now the standard base config.

---

## 2026-04-29 — PR #8: [frieren] Per-case geometry FiLM conditioning
- Branch: `frieren/round1-film-geom-conditioning`
- Hypothesis: Geometry-conditioned FiLM layers (GeomEncoder → FiLM per block) allow the model to specialize per car geometry.
- Results: W&B run `hltti2ec` (1 epoch only, timeout)

| Metric | FiLM | No-FiLM (PR #3) | Δ |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **16.53** | 30.47 | −46% |
| `test_primary/surface_pressure_rel_l2_pct` | 10.38 | 21.65 | −52% |
| `test_primary/wall_shear_rel_l2_pct` | 17.29 | 32.51 | −47% |
| `test_primary/volume_pressure_rel_l2_pct` | 14.91 | 23.73 | −37% |
| `test_primary/wall_shear_x_rel_l2_pct` | 14.76 | 28.07 | −47% |
| `test_primary/wall_shear_y_rel_l2_pct` | 20.59 | 39.02 | −47% |
| `test_primary/wall_shear_z_rel_l2_pct` | 22.00 | 39.88 | −45% |

- Commentary: FiLM halves the error at 1 epoch. Mechanistically active (geom_token_norm grew 70× during training). +142k params (+4.4% overhead). Pending rebase to merge.

---

## 2026-04-29 — PR #13: [norman] Progressive EMA decay anneal 0.99→0.9999
- Branch: `norman/round1-progressive-ema-decay`
- Hypothesis: Cosine-annealed EMA from 0.99 (start) to 0.9999 (end) — fast updates early, slow stable averaging late.
- Results: W&B run `wio9pqw2`

| Metric | Value | Δ vs PR #9 |
|---|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **15.82** | −9.0% |
| `test_primary/surface_pressure_rel_l2_pct` | 9.99 | −9.7% |
| `test_primary/wall_shear_rel_l2_pct` | 16.60 | −9.4% |
| `test_primary/volume_pressure_rel_l2_pct` | 14.21 | −6.6% |
| `test_primary/wall_shear_x_rel_l2_pct` | 14.27 | −8.8% |
| `test_primary/wall_shear_y_rel_l2_pct` | 19.49 | −10.8% |
| `test_primary/wall_shear_z_rel_l2_pct` | 21.12 | −8.9% |

- Commentary: Monotonic val improvement through 4 epochs (no divergence). No new parameters. 9% improvement. Code now on yi; all future runs should use `--ema-decay-start 0.99 --ema-decay-end 0.9999`.

---

## 2026-04-29 — PR #3: [askeladd] Codex/optimized-lineage config (4L/256d)
- Branch: `askeladd/round1-codex-lineage`
- Hypothesis: The codex/optimized-lineage config (4L/256d/4h/128sl, lr=2e-4, wd=5e-4, 65k pts, ema=0.9995) is a stronger baseline than stock defaults.
- Results: W&B run `kv586sse`

| Metric | Value | PR #13 pending |
|---|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **15.27** | 15.82 |
| `test_primary/surface_pressure_rel_l2_pct` | 9.45 | 9.99 |
| `test_primary/wall_shear_rel_l2_pct` | 15.97 | 16.60 |
| `test_primary/volume_pressure_rel_l2_pct` | 14.07 | 14.21 |
| `test_primary/wall_shear_x_rel_l2_pct` | 13.67 | 14.27 |
| `test_primary/wall_shear_y_rel_l2_pct` | 19.08 | 19.49 |
| `test_primary/wall_shear_z_rel_l2_pct` | 20.08 | 21.12 |

- Commentary: best_epoch=1 only (gradients diverged epoch 2+, clip guarded checkpoint). NaN checkpoint guard bug discovered here. Win is real but fragile — full improvement will require gradient clipping.

---

## 2026-04-29 — PR #22: [gilbert] Gradient clipping (clip_grad_norm=1.0)
- Branch: `gilbert/round2-gradient-clipping`
- Hypothesis: clip_grad_norm=1.0 between backward and optimizer step stabilizes training.
- Results: 4-arm sweep

| Arm | clip | vol_w | W&B run | abupt |
|---|---:|---:|---|---:|
| 0 | 0.0 | 2.0 | `ujv64aty` | 16.54 |
| **1** | **1.0** | **2.0** | **`9ozwna8l`** | **14.80** |
| 2 | 1.0 | 3.0 | `u1gt9ygf` | NaN |
| 3 | 5.0 | 3.0 | `owuceuvy` | 18.61 |

- Commentary: clip=1.0 with vol_w=2.0 wins (14.80). vol_w=3.0 is not stabilizable. Pre-clip grad norm median 2.17, p99 19.80, max 31.43 — clipping engaged ~50-60% of steps. Code now on yi as default `--clip-grad-norm 1.0`.

---

## 2026-04-29 — PR #24: [emma] Squared rel-L2 aux loss (no sqrt)
- Branch: `emma/round2-squared-rel-l2-aux-loss`
- Hypothesis: Replace relative-L2 with squared relative-L2 to avoid singularity in backward pass near zero targets.
- Results: 2-arm sweep

| Arm | weight | W&B run | abupt |
|---|---:|---|---:|
| 0 | 0.1 | `4lz8rjpy` | diverged |
| **1** | **0.5** | **`zv791js1`** | **14.81** |

- Commentary: w=0.5 won (14.81), trajectory 23.06→17.75→15.13→13.85→14.59 (best epoch 4). Grad norms bounded 1.3–5.8. Pending rebase to merge.

---

## 2026-04-29 — PR #14: [senku] Depth ablation 5L/6L/256d
- Branch: `senku/round1-depth-ablation`
- Hypothesis: Increasing Transolver depth from 4L to 5L/6L while keeping hidden_dim=256 provides compositional capacity that width alone cannot replicate.
- Results: 2 runs

| Config | W&B run | abupt | Δ vs 4L baseline |
|---|---|---:|---:|
| 5L/256d | `t5tv01ch` | **13.52** | −18.7% |
| **6L/256d** | **`et4ajeqj`** | **13.15** | **−21.0%** |

| Metric | 6L/256d | Prior (4L/512d PR #4) | AB-UPT |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **13.15** | 16.64 | — |
| `test_primary/surface_pressure_rel_l2_pct` | 7.64 | 10.65 | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | 13.47 | 17.66 | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | 13.58 | 14.37 | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | 11.53 | 14.87 | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | 16.23 | 19.89 | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | 16.75 | 21.73 | 3.63 |

- Commentary: New yi best. Both runs monotonically improving at timeout — still descending. Depth is more parameter-efficient than width: 6L/256d (4.73M) crushes 4L/512d (12.7M). Volume pressure val→test gap (6.93→13.58) — test generalization issue, not model capacity. Wall_shear axes still 4-5× AB-UPT.

---

## 2026-04-29 — PR #45: [fern] Mamba-2 SSM surface decoder — FAILED
- Branch: `fern/b04-mamba2-surface-decoder`
- Hypothesis: Post-Transolver SSM decoder using Morton Z-order sorted surface tokens.
- Results: W&B run `322xcrnv`, abupt=27.53 (+65% over baseline)
- Commentary: Catastrophic failure. No zero-init residual + no gradient clipping → backbone poisoning. mamba-ssm unavailable (no nvcc), fell back to S4D-Lin. Closed.

---

## 2026-04-29 — PR #38: [violet] NIG evidential regression — FAILED
- Branch: `violet/c02-deep-evidential-regression`
- Hypothesis: NIG-NLL replaces MSE, outputting uncertainty.
- Results: λ=0.01: abupt=33.54 (`vo9ep9fd`); λ=0.1: abupt=42.63 (`trreny49`)
- Commentary: NIG parameter collapse (α→1+ε, ν→0) + max grad norm 4.91×10⁷. Fundamental — not just hyperparameter. Closed.

---

## 2026-04-29 — PR #29: [chihiro] 512d × FiLM × cosine-EMA composition — FAILED
- Branch: `chihiro/b06-width-film-ema-composition`
- Hypothesis: Compose 4L/512d + FiLM + cosine EMA.
- Results: W&B run `kk7wkhkv`, abupt=37.08 (+122%)
- Commentary: EMA schedule bug — denominator set to 50 epochs × steps/epoch but only 3 epochs completed (4% of schedule), so EMA effectively pinned at 0.99. Hypothesis remains valid. Closed.

---

## 2026-04-29 — PR #21: [kohaku] Normal-component suppression — PENDING RERUN
- Branch: `kohaku/round2-normal-suppression`
- Hypothesis: Explicit penalty on (ws_pred · n_hat)² drives predicted normal component to zero.
- Results: Gradient-clip confounded. Best arm λ=1.0: abupt=18.76. λ=0.01 reached 17.89 before diverging.
- Commentary: Mechanism works (wallshear_pred_normal_rms drops with λ). All arms hit gradient-clip bug. Sent back for rerun on 6L base with clip=1.0.

---

## 2026-04-29 — PR #15: [tanjiro] SDF-gated volume attention — PENDING RERUN
- Branch: `tanjiro/round1-sdf-gate-volume`
- Hypothesis: Gaussian gate on |SDF| focuses volume attention on near-wall points.
- Results: W&B run `iiedyq63`, abupt=17.68 (+1.7% regression vs PR #9)
- Commentary: Marginal volume_pressure improvement (−0.5%) but overall regression. sigma=0.05 too wide (gate ≈0.97 at q75). Sent back for rerun with sigma=0.005 on 6L base.

---

## 2026-04-29 — PR #16: [thorfinn] Bilateral xz-plane TTA — CLOSED
- Branch: `thorfinn/round1-tta-bilateral-symmetry`
- Hypothesis: Mirror geometry about xz-plane at inference, average predictions.
- Results: W&B run `xdjsf4ad`, no-TTA=19.28, TTA=19.40 (TTA hurts)
- Commentary: Cars are symmetric but model predictions are not variance-reduced. TTA only helps near-equivariant models. Closed.

---

## 2026-04-29 — PR #2: [alphonse] Stock defaults baseline — CLOSED
- Hypothesis: Reference floor for stock train.py defaults.
- Results: W&B run `a1fikrwe`, abupt=87.30
- Commentary: Confirms massive gap between stock (3L/192d, 40k pts) and optimized protocol. NaN checkpoint guard bug discovered. Closed.

---

## 2026-04-30 14:00 — PR #58: [alphonse] NaN-safe checkpoint guard — MERGED (bugfix)
- Branch: `alphonse/nan-checkpoint-guard-bugfix`
- Hypothesis: Guard `best_checkpoint` overwrite against NaN primary_val to prevent EMA NaN from replacing valid checkpoint.
- Results: Bugfix — no metric change.
- Commentary: Root cause: `_finite_mean([nan, nan])` returns 0.0, and `0.0 < best_val` fires improved=True, overwriting valid checkpoint with NaN model. Fix: `primary_val_is_valid = math.isfinite(primary_val) and primary_val > 0.0`. Validated by smoke run `tcyjp36i`. Merged to yi.

---

## 2026-04-30 14:10 — PR #66: [thorfinn] Per-axis tau_y/z loss upweighting W_y=2, W_z=2 — MERGED (NEW BEST)
- Branch: `thorfinn/surface-loss-weight-and-per-axis-wallshear`
- Hypothesis: Selectively upweight tau_y and tau_z channels in surface MSE loss (W_y=2, W_z=2, W_x=1) to redirect training gradient toward the two hardest wall-shear axes.
- Results: 3-arm sweep on 6L/256d base

| Arm | W_y | W_z | W&B run | abupt | wall_shear_y | wall_shear_z |
|---|---:|---:|---|---:|---:|---:|
| yw1.5-zw1.5 | 1.5 | 1.5 | `vf3y3z7g` | 13.01 | 15.49 | 15.41 |
| **yw2-zw2** | **2.0** | **2.0** | **`gvigs86q`** | **12.74** | **15.15** | **15.05** |
| yw3-zw3 | 3.0 | 3.0 | `w8r0mvf1` | 13.18 | 15.12 | 14.52 |

| Metric | thorfinn yw2-zw2 | PR #14 (6L/256d) | AB-UPT |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **12.74** | 13.15 | — |
| `test_primary/surface_pressure_rel_l2_pct` | 7.86 | 7.64 | 3.82 |
| `test_primary/wall_shear_rel_l2_pct` | 12.86 | 13.47 | 7.29 |
| `test_primary/volume_pressure_rel_l2_pct` | 13.14 | 13.58 | 6.08 |
| `test_primary/wall_shear_x_rel_l2_pct` | 11.29 | 11.53 | 5.35 |
| `test_primary/wall_shear_y_rel_l2_pct` | 15.15 | 16.23 | 3.65 |
| `test_primary/wall_shear_z_rel_l2_pct` | 15.05 | 16.75 | 3.63 |

- Commentary: New yi best (12.74, −3.1% vs 13.15). The W=2 sweet spot outperforms W=1.5 and W=3 — W=3 overfits the tau_y/z directions, slightly hurting abupt. The selective approach (upweight only tau_y/z, not tau_x) avoids the divergence seen in haku's uniform weighting PR #10. tau_y and tau_z are the most challenging axes (4× AB-UPT); explicit gradient emphasis works.

---

## 2026-04-30 14:20 — PR #65: [violet] Volume-loss-weight sweep (1.5/2.0/3.0/4.0) — CLOSED
- Branch: `violet/volume-pressure-loss-weight-sweep`
- Hypothesis: vol_w=4.0 might further reduce volume_pressure by forcing more gradient toward volume prediction.
- Results: 4-arm sweep

| Arm | vol_w | W&B run | abupt | vol_pressure |
|---|---:|---|---:|---:|
| vw-15 | 1.5 | `n3k58pah` | 13.71 | 13.56 |
| vw-20 | 2.0 | `ioq7jh9w` | 13.61 | 13.62 |
| vw-30 | 3.0 | `kj2i4gx3` | 13.72 | 13.45 |
| vw-40 | 4.0 | `v98qrfmd` | 13.71 | 13.30 |

- Commentary: No arm beats baseline 12.74. vol_w=4.0 marginally improves volume_pressure (13.30 vs 13.58) but hurts wall-shear. The kill-threshold bug (>=N syntax) prematurely killed several arms. vol_w=2.0 confirmed as the correct operating point for composite metric. Closed.

---

## 2026-04-30 14:20 — PR #64: [fern] Stochastic depth regularization (3-rate sweep) — CLOSED
- Branch: `fern/stochastic-depth-regularization`
- Hypothesis: Stochastic depth (drop-path) regularization at rates 0.05/0.10/0.20 provides regularization to prevent overfitting.
- Results: 3-arm sweep

| Arm | sdp_rate | W&B run | abupt |
|---|---:|---|---:|
| sdp-005 | 0.05 | — (killed by threshold bug) | — |
| sdp-010 | 0.10 | `q8yv93km` | 13.73 |
| sdp-020 | 0.20 | `w3bt19pk` | 13.92 |

- Commentary: All arms negative vs baseline 12.74 (+7.8% best). Stochastic depth regularization is redundant at 4-epoch budgets. Gradient clip already provides effective implicit regularization. Closed.

---

## 2026-04-30 14:20 — PR #63: [askeladd] Squared rel-L2 aux loss on 6L base — SENT BACK FOR REBASE
- Branch: `askeladd/squared-rel-l2-aux-on-6l`
- Hypothesis: Add squared relative-L2 aux loss on 6L base; weight sweep w=0.1/0.3/0.5/1.0.
- Results: 4-arm sweep

| Arm | weight | W&B run | abupt |
|---|---:|---|---:|
| w=0.1 | 0.1 | `qntz7gzr` | 13.42 |
| w=0.3 | 0.3 | `h5w3vf5y` | 13.11 |
| **w=0.5** | **0.5** | **`dln9trni`** | **12.94** |
| w=1.0 | 1.0 | `n9ckb2qe` | 13.77 |

- Commentary: w=0.5 achieved 12.94 — beats PR #14 baseline (13.15) but not new baseline 12.74 (PR #66). The composition of squared rel-L2 aux loss + thorfinn per-axis weights is untested. Sent back to rebase onto thorfinn base and re-run with both --aux-rel-l2-weight 0.5 and --wallshear-y-weight 2.0 --wallshear-z-weight 2.0.

---

## 2026-04-30 14:20 — PR #61: [gilbert] Tangential wall-shear projection on 6L base — CLOSED
- Branch: `gilbert/tangential-wallshear-on-6l-base`
- Hypothesis: Tangential projection loss on 6L base (no normal penalty).
- Results: abupt=34.07 (W&B `x0pyk2yw`) — catastrophic failure. 2.7× baseline.
- Commentary: Projection without normal penalty allows unbounded normal component growth. Gradient signal is removed in the normal direction but no compensating loss drives it to zero. Tangential projection research line closed.

---

## 2026-04-30 14:20 — PR #60: [chihiro] 6L/512d depth×width composition — CLOSED
- Branch: `chihiro/depth-width-composition-6l-512d`
- Hypothesis: Combining 6L depth with 512d width should outperform either alone.
- Results: abupt=16.00, only 2 epochs completed (data-starved).
- Commentary: 6L/512d (18.1M params) is too large for the 4.5h budget — only 2 epochs vs 4 for 6L/256d. Data-starvation dominates. 6L/256d is the right operating point. Closed.

---

## 2026-04-30 14:20 — PR #59: [senku] Depth 7L/8L sweep — CLOSED
- Branch: `senku/depth-7l-8l-sweep`
- Hypothesis: Pushing depth beyond 6L further improves the composite metric.
- Results: 7L abupt=13.28, 8L abupt=13.57 (both worse than 6L=13.15/12.74 baseline).
- Commentary: 7L/8L hit kill-threshold bug (>=18 means kill when val drops below 18, so some runs killed prematurely). Even corrected, both are worse than 6L at same compute — more depth = fewer epochs = data starvation at this budget. Depth ceiling confirmed at 6L. Closed.

---

## 2026-04-30 14:20 — PR #21: [kohaku] Normal-component suppression on 6L (sweep-v2) — CLOSED
- Branch: `kohaku/round2-normal-component-suppression`
- Hypothesis: Penalty λ*(ws_pred·n_hat)² drives predicted normal component to zero; sweep λ∈{0.01, 0.1, 1.0} on 6L base.
- Results:

| λ | W&B run | abupt |
|---:|---|---:|
| 0.01 | `le10xx7e` | 16.06 |
| 0.1 | `j0gdj2jy` | 17.47 |
| 1.0 | `fsxvmo08` | 15.93 |

- Commentary: All arms 22–33% worse than baseline 12.74. The suppression mechanism works mechanistically but the projection+suppression combination degrades wall-shear badly (tau_y/z ≈20–26%, far worse than baseline 15-17%). Tangential projection research conclusively closed.

---

## 2026-04-30 14:20 — PR #15: [tanjiro] SDF-gated volume attention (v2, sigma sweep) — CLOSED
- Branch: `tanjiro/round1-sdf-gated-volume-attention`
- Hypothesis: Gaussian SDF gate or quantile-rank gate focuses volume attention on near-wall critical points.
- Results: 3-arm sweep on 6L base

| Arm | W&B run | abupt |
|---|---|---:|
| quantile q=0.10 | `l6yfeh31` | 13.26 |
| gaussian σ=0.005 | `gu2v23cs` | 13.87 |
| gaussian σ=0.001 | `r7c8jss2` | 36.48 (diverged) |

- Commentary: Best arm (quantile q=0.10) achieves 13.26 — worse than new baseline 12.74. The val→test gap in volume_pressure (12→13.58) is a distribution shift issue that SDF gating cannot address. σ=0.001 diverges. Closing — SDF gating adds instability and doesn't help test generalization. Closed.

---

## 2026-04-30 14:20 — PR #10: [haku] Per-axis wall-shear loss weights (uniform w2/w3) — CLOSED
- Branch: `haku/round1-per-axis-wallshear-loss-weight`
- Hypothesis: Uniform upweighting of all 3 wall-shear channels (tau_x, tau_y, tau_z) by 2× or 3× should reduce wall-shear error.
- Results (Round-2, 6L base):

| Arm | weights | W&B run | abupt |
|---|---|---|---:|
| Control | (1,1,1,1) | `648ssek0` | 13.15 |
| w2 | (1,2,2,2) | `s3y7sclb` | 18.35 (diverged ep3) |
| w3 | (1,3,3,3) | `inpik7c3` | 14.35 (diverged ep4) |
| w2+tan | (1,2,2,2)+tan | `1cxf7026` | 36.41 |

- Commentary: Uniform upweighting causes divergence — tau_x upweighting destabilizes. Thorfinn's selective approach (W_y=W_z=2, W_x=1) succeeds where uniform fails. Confirmed: tau_x should not be upweighted. Closed.

---

## 2026-04-30 14:20 — PR #24: [emma] Squared rel-L2 aux loss (4L base) — CLOSED
- Branch: `emma/round2-squared-rel-l2-aux-loss`
- Hypothesis: Squared rel-L2 aux loss (no sqrt) is stable where round-1 version diverged.
- Results: w=0.5 → abupt=14.81 (W&B `zv791js1`, 4L base).
- Commentary: Stable (no divergence) and beats 4L baseline, but 4L superseded. Code was incorporated into PR #63 (askeladd, 6L). Closed — hypothesis lives on in PR #63.

---

## 2026-04-30 14:20 — PR #5: [edward] Cosine LR + FiLM composition on 6L — CLOSED
- Branch: `edward/round1-cosine-lr-warmup`
- Hypothesis: Cosine annealing + FiLM conditioning compose orthogonally on 6L base.
- Results: abupt=19.27 (W&B `duv7m45t`) — best val at epoch 1 (18.44), degraded monotonically thereafter.
- Commentary: FiLM + cosine LR + cosine EMA on 6L creates a fragile stack. The interaction between the cosine LR schedule and the cosine EMA ramp produces unstable training. Cosine LR alone on 6L (without FiLM) is being tested in PR #67 kafka. Closed.

---

## 2026-05-01 06:30 — PR #131: [thorfinn] Log-magnitude wall-shear target normalization — CLOSED NEGATIVE
- Branch: `thorfinn/log-magnitude-wallshear-targets`
- Hypothesis: sign(x)*log1p(|x|/eps) normalization compresses heavy-tailed wall-shear distribution, improving y/z gap.
- Results: best arm (eps=1.0, eps_low=0.10) → val abupt=11.03 vs baseline 10.69. wall_shear_y/z both regressed. Smaller eps (0.01, 0.10) caused pre_clip_norm spikes to 2M+ and NaNs (gradient of log1p near 0 is ~1/eps).
- Commentary: NEGATIVE on primary metric and on the targeted sub-metric. The NaN/Inf-skip safeguard added in this PR (commit 2a8f7e4) is structurally valuable and orthogonal — should be cherry-picked into a utility PR. asinh transform (frieren #123, in-flight) is the smoother alternative.

---

## 2026-05-01 06:31 — PR #130: [tanjiro] Curriculum tau_y/z weighting schedule — CLOSED NEGATIVE
- Branch: `tanjiro/curriculum-tau-yz-weighting`
- Hypothesis: Linearly ramping W_y from 1→3 (and W_z parallel) across early training stabilizes Adam moments before reaching final upweight.
- Results: 6/6 arm launches diverged. All hit Adam-second-moment desynchronization band around W_y≈2.7.
- Commentary: NEGATIVE. Curriculum reweighting on top of existing W_y=W_z=2 base (effective surface gradient already ~3× volume) is structurally incompatible with Adam state. Per-component static reweighting at start (so m/v initialize correctly per channel) is the better path.

---

## 2026-05-01 06:32 — PR #124: [gilbert] RANS divergence constraint (Laplacian on pressure) — CLOSED NEGATIVE
- Branch: `gilbert/rans-divergence-constraint`
- Hypothesis: Soft penalty ∇²p ≈ 0 (kNN inverse-distance Laplacian, k=8, sdf_threshold=0.05) acts as physics-informed regularizer.
- Results: λ≥0.01 destabilized training, λ=0.001 was a no-op. No signal at any λ.
- Commentary: NEGATIVE — wrong physics. Δp=0 is incompressible-Stokes (creeping flow); real RANS has advective and turbulent stress terms. The right divergence constraint is ∇·u=0 on velocity targets, which requires loader changes to expose u. Reusable kNN Laplacian implementation is a keeper for future physics work.

---

## 2026-05-01 06:33 — PR #121: [askeladd] Surface-tangent-frame wall-shear head — CLOSED NEGATIVE
- Branch: `askeladd/surface-tangent-frame-wallshear`
- Hypothesis: Reparametrize wall-shear prediction as τ = a·t1 + b·t2 in tangent frame (Duff 2017 ONB) to factor out normal direction and improve y/z accuracy.
- Results: Worse than global Cartesian baseline at every checkpoint. Lr=3e-4, clip=0.3 delayed but didn't prevent gradient explosion.
- Commentary: NEGATIVE. Two structural problems: (1) Duff branchless ONB is discontinuous along t1.x sign-flip — adjacent surface patches with similar normals map to opposite (a,b) targets, and a non-gauge-equivariant Transolver can't learn discontinuous frames. (2) τ_y/τ_z share (a,b) weights → channel coupling in Adam → correlated gradient spikes. Continuous frame (heat-method, PCA-aligned) or soft normal-component penalty are the right alternatives.

---

## 2026-05-01 06:34 — PR #118: [chihiro] MLP ratio sweep 6/8 — CLOSED AMBIGUOUS
- Branch: `chihiro/mlp-ratio-sweep-r4`
- Hypothesis: Wider FFN intermediate dim (mlp_ratio=6 or 8) increases capacity per layer without depth penalty.
- Results: 12/16 runs diverged. 4 valid epoch-1 vals; mlp_ratio=8 trended slightly better than mlp_ratio=4 baseline but no convergent comparison vs 10.69.
- Commentary: AMBIGUOUS — seed-dependent gradient instability (same as alphonse #117). Recommend re-running mlp_ratio=8 under stability-hardened recipe: 1k-step linear LR warmup from 1e-5, --seed flag, lr=3e-4. If stabilized, +3M params with clear epoch-1 trend signal is worth pursuing.

---

## 2026-05-01 06:35 — PR #129: [senku] Surface loss weight sweep on PR #99 base — CLOSED NEGATIVE
- Branch: `senku/surface-loss-upweight-sweep`
- Hypothesis: Uniform surface_loss_weight ∈ {1.5, 2.0, 2.5, 3.0} on PR #99 base improves surface metrics.
- Results: 8 arms total (A/B/C/D + R1/R2/R3/R4 rescue at lr=3e-4, lr=2e-4, varying clips). 7 diverged. R4 (lr=2e-4, clip=0.5, 1k warmup, W&B `jtx73lg0`) was only stable arm but killed by external pod restart at step 27752 mid-ep3. Best partial: R3 ep2 abupt=12.84 vs baseline ep2=12.42 (already behind, then destabilized).
- Commentary: NEGATIVE. Monotone instability with sw confirmed: D@8k → A@18.6k → C@19.7k → B@20.7k. Base config already has W_y=W_z=2 (effective surface gradient ~3× volume); uniform sw≥1.5 amplifies the entire bundle including upweighted y/z, exceeding stability ceiling. Per-component reweighting (--wallshear-y/z-weight only) is the right knob, not uniform --surface-loss-weight.

---

## 2026-05-01 06:36 — PR #117: [alphonse] Width scale-up sweep 384d + 8L depth — CLOSED NEGATIVE (PROMISING SIGNAL)
- Branch: `alphonse/width-384d-sweep`
- Hypothesis: Scale up Transolver width to 384d (4h or 6h) or depth to 8L for capacity gain.
- Results: 8L/256d (W&B `xl92i3f5`, lr=3e-4) reached val abupt=11.33 at partial ep3, slope -0.59 abupt-pct/1k steps; extrapolated baseline crossing at ~step 26,500 but train_timeout=270min fired at step ~25,400. Test: abupt=12.44, volume_pressure=13.84 (only sub-metric beating baseline 14.42). Both 6L/384d arms diverged at all LRs (5e-4, 3e-4, 2e-4); d_head=96 destabilized earlier than d_head=64.
- Commentary: NEGATIVE on merge bar but PROMISING. 8L/256d was time-limited not architecture-limited — depth is the viable scale-up direction. Width 384d needs QK-norm or fp32 attention to be stable in bf16. Round-6 priority: revisit 8L/256d combined with 1cycle LR for super-convergence.

---

## 2026-05-01 07:30 — PR #143: [fern] coordinate normalization sweep — CLOSED NEGATIVE
- Branch: `fern/coord-normalization-sweep`
- Hypothesis: Anisotropic bbox (x~8m, y/z~2-2.5m) makes `ContinuousSincosEmbed` give x-axis ~3-4× more frequency resolution than y/z, contributing to the 4× tau_y/z gap. Adding `--coord-normalize {none, global-scale, per-axis}` should restore isotropy.
- Results: Hypothesis falsified across 9 attempts.

| Mode | First-epoch abupt | vs control | Notes |
|---|---:|---:|---|
| none (control) | ~16.20 | — | matches PR #99 e1, then hits fleet-wide stochastic divergence in e2 |
| global-scale | 24.85 | +8.65 (+53%) | normalizing to unit cube destroys sincos expressiveness — omega bank is calibrated for meter-scale geometry |
| per-axis (4 variants) | diverged | — | volume tokens (~25× wider domain than vehicle bbox) get extreme out-of-range coords → MLP bias → slice attention gradient explosion |

- Commentary: Coordinate normalization is the wrong lever — it breaks the fixed-frequency `omega` bank tuned for meter-scale wavelengths. The 4× tau_y/z gap is **NOT primarily** a sincos-anisotropy problem. Right next attack: tune the omega bank directly in physical-meter coords (denser/per-axis frequency basis on y/z), preserving meter-scale calibration. **Bug-fix side-discovery:** confirmed ~50% fleet-wide divergence at lr=5e-4. Non-finite/large-grad skip guard from haku (commit `6e8b674`) is already on `yi` so future runs are protected. **Decision: closed** — coord-normalize feature not merged. Fern reassigned to PR #183 (omega-bank anisotropic frequency sweep).

---

## 2026-05-01 07:30 — PR #126: [kohaku] FiLM geometry conditioning on PR #99 6L/256d base — CLOSED NEGATIVE (PROMISING SIGNAL)
- Branch: `kohaku/film-conditioning-6l-256d`
- Hypothesis: FiLM (PR #8 frieren code) + lr=5e-4 (PR #99 fern base) is additive — global geometry prior plus fast convergence.
- Results: Hypothesis falsified across all 4 arms.

| Arm | lr | clip | Diverge step | Best partial | W&B |
|---|---:|---:|---:|---|---|
| 1 | 5e-4 | 1.0 | ~15.3k (mid e2) | n/a | h6nlfcdr |
| 2 | 5e-4 | 0.5 | ~12.4k (mid e2) | n/a | 3ddue2xd |
| 3 | 4e-4 | 0.5 | ~30.1k (mid e3) | abupt=11.67, **vp=7.05** (e2) | sudqmuo9 |
| 4 | 3e-4 | 0.5 | ~19.0k (mid e2) | n/a | jd1acg1t |

- Commentary: Best partial (Arm 3 e2 abupt=11.67) is still 0.98pp worse than baseline 10.69. **lr=3e-4 (Arm 4) failed earlier than lr=4e-4 (Arm 3)** — pure LR reduction is not the lever. Root cause from forensics: `train/film/geom_token_norm_mean` was steady (~0.73-0.81) at all 4 divergence points (geom token is fine); layer-0 `to_gamma_beta/bias grad_to_param_norm=0.567` flagged the FiLM linear projections as the gradient amplification path. FiLM × LR ≥ 3e-4 has a fundamental stability ceiling at default-init. **Promising signal:** Arm 3 e2 vp=7.05 vs baseline 7.85 — FiLM helps volume more than surface, exactly the metric closest to AB-UPT (1.3× away). The direction is right; the failure mode is dynamics, not capability. **Decision: closed** — kohaku reassigned to PR #184 (FiLM with identity/zero-init, DiT-style stable conditioning).

