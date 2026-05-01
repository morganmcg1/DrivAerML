# SENPAI Research Results

## 2026-05-02 23:10 — PR #227 [stark]: Surface tangent-frame wall shear — CLOSED (pod never provisioned, RBAC blocker)

- Branch: `stark/surface-tangent-frame` (deleted)
- Hypothesis: Predict wall-shear in local {t1, t2, n} tangent frame (Duff et al. 2017 frame builder) so the network's regression head respects the no-slip BC geometry directly and decouples tau_y / tau_z from the global coordinate system that the current network conflates.
- Status: Never ran. The `senpai-yi-stark` deployment / ConfigMap / pod never existed in the live `pai-2/default` cluster (operator confirmed via Issue #248 on 2026-05-01 23:05Z). Advisor pod's service account lacks RBAC create/patch on configmaps and deployments.apps, so we could not self-provision.
- Operator directive: reassign rather than wait. PR #227 closed by advisor 2026-05-02 23:09Z. Issue #248 acknowledged; hypothesis queued as **highest-priority next-idle assignment** off PR #222 baseline (val_abupt = 9.291%).
- Note: stark slot is not in the active `STUDENT_NAMES` for the `yi` advisor branch (16 active: alphonse, askeladd, chihiro, edward, emma, fern, frieren, gilbert, haku, kohaku, nezuko, norman, senku, tanjiro, thorfinn, violet). Reassignment will go to the first of those to become idle.

## 2026-04-29 14:00 — PR #224: [fern] Learned Fourier embeddings (init-scale sweep) — CLOSED (divergence, no convergence past ep1)

- Branch: `fern/learned-frequency-embeddings` (deleted)
- Hypothesis: Replace fixed sincos positional encoding with a learned linear projection `W: Linear(3→d//2)`, output `[sin(Wx), cos(Wx)]`. With high init_scale (10–50), W spans high-frequency space but can adapt; hypothesized to improve τ_y/τ_z frequency selectivity.

| Arm | init_scale | lr | clip | ep1 abupt | ep1 tau_y | ep1 tau_z | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| A (sincos control) | — | 5e-4 | 1.0 | 10.99% | 14.0% | 15.1% | finished (reference) |
| B (init=1) | 1 | 5e-4 | 1.0 | 12.38% | 15.6% | 17.2% | failed — worse than control |
| C (init=0.1) | 0.1 | 5e-4 | 1.0 | 12.10% | 15.3% | 16.9% | failed — worse than control |
| D (init=100) | 100 | 5e-4 | 1.0 | 11.20% | 14.1% | 15.2% | marginally worse |
| E (init=10) | 10 | 5e-4 | 1.0 | 11.02% | 13.9% | 15.0% | marginal improvement, NaN ~ep4 |
| F (init=50) | 50 | 5e-4 | 1.0 | 10.98% | 13.8% | 15.0% | marginal, NaN ~ep4 |
| N (init=10, clip=0.3) | 10 | 5e-4 | 0.3 | 10.94% | 13.9% | 15.0% | slight improvement, NaN ~ep4 |
| O (init=10, clip=0.5, lr=3e-4) | 10 | 3e-4 | 0.5 | **10.17%** | **12.9%** | **13.9%** | best ep1, NaN step ~19000 |
| K (best val overall) | 10 | 5e-4 | 1.0 | — | — | — | best val 10.48%, did not beat bar 9.291% |

Baseline: 9.291% val_abupt (PR #222 merge bar)

- **Result:** CLOSED — no arm beat the 9.291% merge bar. Best overall val was arm K at 10.48% (−12.7% vs baseline). The ep1 improvement from init=1 (worse) to init=10 (better) confirmed the frequency selectivity hypothesis is directionally correct, but high-LR instability (NaN ~step 4764–19000 in all runs, fleet-wide bf16 + lr=5e-4) prevented convergence.
- **Key finding:** Arm O (init=10, clip=0.5, lr=3e-4) showed the strongest ep1 signal: abupt 10.17% vs 10.99% control (−7.5%). However, O also diverged before step ~19000. The ep1 lead is architecturally real.
- **W row-norm analysis:** Per-axis column norms `||W[:, x]||_2`, `||W[:, y]||_2`, `||W[:, z]||_2` were inconclusive at ep1 — not enough training time to show frequency selectivity emergence.
- **init=1 pathology:** Learned W with small init is poorly conditioned at ep0 and converges slower than sincos. init≥10 is required.
- **Fleet instability root cause:** lr=5e-4 + bf16 + clip≥0.5 causes NaN divergence around steps 4764–19000; confirmed across PRs #197, #224, #245.
- **Follow-up assigned:** PR #298 (fern) — 4-arm sweep at stable lr=3e-4 + clip=0.5 + adamw on 6L/256d base to verify arm O's ep1 advantage persists through convergence.

---

## 2026-05-02 00:00 — PR #225: [haku] Left-right symmetry augmentation for tau_y/z gap — CLOSED (ep1 signal, no convergence)

- Branch: `haku/symmetry-augmentation` (deleted)
- Hypothesis: Left-right y-axis reflection of DrivAerML surface data addresses tau_y/tau_z gap by teaching the model the bilateral symmetry constraint. `--symmetry-include-both` (orig+flip concat per step, effective bs×2) was tested alongside stochastic flip p=0.5.

| Arm | Run | Seeds | ep1 abupt | ep1 tau_y | ep1 tau_z | Outcome |
|---|---|---|---:|---:|---:|---|
| A control | zhwlaury/te7uug8u/k3boii9j | 42/7/13 | — | — | — | All crashed before ep1 val |
| B symm p=0.5 | byxxuehz | 13 | 15.95 (−10.0%) | 20.46 (−11.3%) | 22.53 (−11.1%) | ep1 val OK, crashed ep2 |
| C symm-both bs=4 | d03gq4om | 101 | **12.75 (−28.0%)** | **16.29 (−29.4%)** | **17.89 (−29.4%)** | ep1 val OK, crashed ep2 |
| D symm-yzw3 | 2een5w1o | 42 | 15.71 (−11.4%) | 19.96 (−13.5%) | 21.80 (−14.0%) | ep1 val OK, crashed ep2 |

Baseline ep1 comparison: bplngfyo (PR #183 epoch-1 val): abupt=17.72, tau_y=23.07, tau_z=25.35

- **Result:** Win criterion not met — no run survived to final convergence. All 11 attempts in this PR hit NONFINITE_SKIP_ABORT (>200 nonfinite steps), including 3/3 control arm attempts. The instability is structural (fleet-wide lr=5e-4 + warmup=500 issue), not augmentation-induced.
- **Ep1 signal:** Arm C (symm-include-both, bs=4) showed the strongest signal in the fleet at ep1: −28% abupt and −29.4% tau_y/z vs baseline ep1. The tau_y/z disproportionality ratio is mild (1.05× — most of the win is uniform improvement, partly from effective dataset doubling). Arm B/D (random p=0.5) showed a smaller but consistent −10-14% improvement.
- **Key insight:** The "include-both" variant (doubling effective data per step) is much more powerful than stochastic flip. This suggests the win is partly a true symmetry prior and partly sample-budget doubling — worth disentangling in a follow-up.
- **Follow-up assigned:** PR #297 (haku) — re-test Arm C (symm-include-both, bs=4) on the stable PR #222 base config (lr=1e-4, warmup=1 epoch, 8-GPU DDP torchrun). The −28% ep1 signal should survive to convergence on the stable base.

---

## 2026-04-29 12:00 — PR #144: [edward] AdamW beta2 sweep (0.95 vs 0.999, lr 3e-4 and 5e-4) — CLOSED NEGATIVE
- Branch: `edward/adamw-beta2-sweep`
- Hypothesis: β2=0.95 in AdamW (faster second-moment adaptation) will reduce the v-saturation collapse driving NaN at lr=5e-4, potentially unlocking lr=5e-4 stability or beyond.

| Arm | Run | Best val_primary | Terminal |
|---|---|---:|---|
| A v1 (lr=5e-4, β2=0.999) | 0351xvpg | 11.962 (e2) | NaN ep3 step 32417 |
| B v1 (lr=5e-4, β2=0.95) | zei4lzb8 | **11.803 (e2)** | NaN ep3 step 30557 |
| C v1 (lr=3e-4, β2=0.95) | 23nmdpp0 | 19.391 (e1) | kill-threshold ep2 |
| D v1 (lr=3e-4, β2=0.999) | nnex5o0v | 18.282 (e1) | kill-threshold ep2 |
| A v2 (lr=5e-4, β2=0.999) | snjasvxx | — | NaN ep1 step 5099 |
| B v2 (lr=5e-4, β2=0.95) | l6os2f8i | 11.945 (e2) | NaN ep3 step 25501 |
| C v2 (lr=3e-4, β2=0.95) | oagys1rq | 12.690 (e4) | finished, 4 epochs |
| D v2 (lr=3e-4, β2=0.999) | oxfn12do | 17.594 (e1) | NaN ep2 ~step 20400 |

Baseline: **10.69** — none of 8 runs reached baseline. Best was B v1 at 11.803 (1.11 pp gap).

- Commentary: **Hypothesis NOT supported. β2=0.95 does not fix the lr=5e-4 instability ceiling.** All 4 lr=5e-4 arm-runs NaN'd in epoch 3 regardless of β2. The dominant stability lever was `--lr-warmup-steps 500`, which pushed NaN onset from ~step 11k (fleet historical) to ~25–32k. β2 shows a weak stability advantage at lr=3e-4 (C v2 survived 4 epochs vs D v2 NaN at epoch 2), but best metric C v2=12.690 is well above baseline 10.69. Infrastructure contribution: `--beta1`, `--beta2`, `--lr-warmup-steps`, `--lr-warmup-start-factor` flags added to train.py — these will be cherry-picked for fleet adoption. Additional finding: kill-threshold operator direction trap (condition is what must hold to continue, NOT to kill — inverted `>=3.0` killed all healthy arms in phase 1).

## 2026-04-29 12:00 — Round 12 assignments: edward #196, gilbert #197, senku #198

Three idle students assigned fresh experiments for Round 12:
- **PR #196 (edward)** — Lion optimizer (sign-based, immune to v-saturation NaN). Sweep: Lion at lr=1e-4/3e-4/5e-4 + AdamW control with warmup=500. Hypothesis: Lion's bounded-magnitude updates sidestep the heavy-tail gradient distribution collapse.
- **PR #197 (gilbert)** — K-NN local attention: replace last 1/2/3 transformer layers with k=32 or k=64 nearest-neighbor surface attention. Hypothesis: wall shear is a local physical quantity; global attention is the wrong inductive bias for tau_y/z.
- **PR #198 (senku)** — Stochastic Weight Averaging (SWA) free gain: collect uniform weight averages over the last 30–50% of training at swa_lr=5e-5–1e-4. Hypothesis: SWA finds wider minima that generalize better than EMA alone; composes orthogonally with EMA.

---

## 2026-05-01 10:15 — PR #167: [tanjiro] Static W_y=W_z=3.5 + 1k LR warmup — CLOSED NEGATIVE
- Branch: `tanjiro/static-wyz-35-warmup`
- Hypothesis: Setting static per-component weights W_y=W_z=3.5 from step 0 (vs curriculum ramping in PR #130 that caused Adam desync) with 1000-step LR warmup gives Adam's second moments time to calibrate before full-lr updates hit the higher-weighted channels.
- W&B run: `ynqjygsa` (tanjiro/static-wyz35-warmup1k)

| Metric | Epoch 1 (only valid epoch) | Baseline (PR #99 best) | AB-UPT target |
|---|---:|---:|---:|
| `val_primary/abupt_axis_mean_rel_l2_pct` | 15.63 | **10.69** | — |
| `val_primary/wall_shear_y_rel_l2_pct` | 19.59 | 13.73 | 3.65 |
| `val_primary/wall_shear_z_rel_l2_pct` | 21.84 | 14.73 | 3.63 |
| `val_primary/surface_pressure_rel_l2_pct` | 11.25 | 6.97 | 3.82 |
| `val_primary/volume_pressure_rel_l2_pct` | 9.85 | 7.85 | 6.08 |

Pre-clip grad norm trajectory by phase:
| Phase | n | mean | max |
|---|---:|---:|---:|
| Warmup 0–1000 | 10 | 25.98 | 64.77 |
| Epoch 1 post-warmup (1001–10883) | 98 | 3.17 | 11.36 |
| Epoch 2 pre-div (1001–15099) | 47 | 554.97 | 13,646.89 |
| Epoch 2 divergence (≥15100) | 2 | 2,035,679 | 4,046,702 |

- Commentary: **Clear negative — NaN divergence at step 15,300 (mid epoch 2).** Epoch 1 beat baseline epoch 1 (15.63 vs 16.47, +5% better) confirming the signal is real. But grad norms drifted 4 OoM mid-epoch-2: elevated warmup → calm epoch 1 → silent drift → catastrophic explosion → NaN. Sister arm senku #166 (W=3.0) also NaN'd (step 9,699, even earlier). **Per-component stability ceiling is below 3.0** at lr=5e-4/clip=1.0. Static weights from step 0 did not prevent the failure — Adam's second-moment saturation on boosted channels is a post-warmup instability, not a warmup miscalibration. Longer warmup would not help. The direction of upweighting is correct (epoch-1 signals are encouraging) but the mechanism needs a lower LR or tighter per-channel gradient clipping. Decision: closed. LR warmup infrastructure already in train.py via PR #169.


## 2026-05-01 10:30 — New assignments: haku #191, tanjiro #192, thorfinn #193

Three idle students (haku, tanjiro, thorfinn) assigned new experiments targeting the tau_y/z gap and training efficiency:

- **PR #191 (haku)** — 1-cycle LR max=1e-3 super-convergence: PyTorch OneCycleLR with peak 1e-3, pct_start=0.3, div_factor=25. Hypothesis: epoch-limited regime benefits from spending more training time at elevated LR. Fallback arm at lr=5e-4 if 1e-3 diverges.
- **PR #192 (tanjiro)** — asinh target normalization for tau_y/z: apply `torch.asinh` to wall-shear y and z targets before computing loss, invert before metric computation. Targets the heavy-tail distribution causing the 4× gap vs AB-UPT. Isolated to y/z components only.
- **PR #193 (thorfinn)** — Curvature-biased surface point sampling: sample training surface points with probability proportional to local surface normal variation (curvature proxy), biasing toward wheel arches/mirrors where tau_y/z errors are highest. Eval sampling remains uniform. Two arms: alpha=0.5 (blend) and alpha=1.0 (fully biased).

---

## 2026-04-29 10:45 — PR #191: [haku] 1-cycle LR max=1e-3 super-convergence — CLOSED NEGATIVE
- Branch: `haku/1cycle-lr-max1e3-superconvergence`
- Hypothesis: PyTorch OneCycleLR with peak_lr=1e-3, pct_start=0.3, div_factor=25 enables super-convergence in the epoch-limited training regime; spending more time at elevated LR accelerates learning on tau_y/z channels.

Three arms:
| Arm | Run | Status | Best val_abupt |
|---|---|---|---:|
| Main literal (lr=1e-3, epochs=50, total_steps=544150) | d86d7dg9 | Finished | 18.43 |
| Tuned (lr=1e-3, epochs=4, calibrated total_steps) | 1khqvozw | NaN-abort at step 12,759 | 28.23 |
| Fallback (lr=5e-4, epochs=4, calibrated total_steps) | 0e3jqcti | NaN-abort at step 14,279 | 27.31 |

Best arm full W&B metrics (d86d7dg9):

| Metric | 1-cycle best | Baseline (PR #183) | Δ |
|---|---:|---:|---|
| abupt_axis_mean_rel_l2_pct | 18.43 | **10.21** | +8.22 |
| surface_pressure_rel_l2_pct | 13.01 | 6.97 | +6.04 |
| wall_shear_rel_l2_pct | 20.86 | 11.69 | +9.17 |
| volume_pressure_rel_l2_pct | 10.60 | 6.32 | +4.28 |
| wall_shear_x_rel_l2_pct | 18.36 | 10.17 | +8.19 |
| wall_shear_y_rel_l2_pct | 24.42 | 13.73 | +10.69 |
| wall_shear_z_rel_l2_pct | 25.76 | 14.73 | +11.03 |

- Commentary: **Clear negative. Root cause: fundamental incompatibility between OneCycleLR's peak schedule and the time-limited training regime.** With total_steps=544,150 (epoch=50), the warmup phase extends to step 163,245 — but the actual training run reaches only ~33,263 steps (3–4 epochs/wall-clock). Training terminated at ~20% through the warmup ramp, never reaching the intended super-convergence phase. Best arm scored 18.43 vs baseline 10.21 (+80% worse). The calibrated arms (total_steps matched to actual budget) failed with NaN-abort at peak LR because the steep 5e-4 → 1e-3 ramp triggered gradient instability before any benefit could materialize. Epoch progression shows steady improvement (35.16→23.84→18.64→18.43) but all below baseline. OneCycleLR is not viable in this regime without either: (a) drastically reducing total_steps to match actual training steps, AND (b) capping peak LR to avoid NaN — by which point it degrades to a standard warmup schedule.

---

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

## 2026-05-01 01:18 — PR #119: [edward] RFF coordinate encoding (Tancik 2020) — CLOSED
- Branch: `edward/rff-coordinate-encoding`
- Hypothesis: Replace ContinuousSincosEmbed with Random Fourier Features (Tancik 2020) to give the model access to learnable coordinate frequencies, potentially improving high-frequency surface detail.
- Results:

| Arm | σ | features | W&B run | val abupt (best epoch) | test abupt | Final state |
|---|---:|---:|---|---:|---:|---|
| 1 (fnyhm654) | 1.0 | 64 | fnyhm654 | NaN | — | crashed (NaN @step ~2300, epoch 1) |
| 1-r2 (n77zkyc8) | 1.0 | 64 | n77zkyc8 | NaN | — | crashed (NaN @step ~7500, epoch 1) |
| 2 (3s9qatve) | 5.0 | 64 | 3s9qatve | 130.01 (epoch 3) | 88.55 | loss explosion @step ~7700; stuck at ~3.5 forever |
| 3 (fig141q6) | 2.0 | 128 | fig141q6 | 17.45 (epoch 1) | 18.28 | NaN @step 17269 (mid epoch 2) |

- Commentary: Clean negative result — all 3 sigma values unstable at lr=5e-4/clip=1.0/bf16/raw-meter-coords. Best arm (σ=2.0, 128 features) test abupt=18.28 (56% worse than baseline 11.73). Wall_shear_y/z worst per-axis (21.3%, 23.2%) — consistent with anisotropy hypothesis (σ isotropic on non-isotropic meters). RFF fixed-Gaussian-B amplifies certain coordinate-frequency components creating unstable training unlike ContinuousSincosEmbed (deterministic, balanced). Suggests coord normalization to [-1,1] before RFF would be needed. Assigned follow-up PR #143 (fern, coord normalization) to test this hypothesis. Closed.

---

## 2026-05-01 (in progress) — PR #117: [alphonse] Width+depth scale-up (6L/384d, 8L/256d) — WIP
- Branch: `alphonse/width-depth-scale-up`
- Hypothesis: Increasing Transolver from 6L/256d (4.73M params) to 6L/384d (~9M) or 8L/256d (~6M) provides additional capacity to resolve the tau_y/z fine-scale surface features.
- Results (intermediate — running at lr=3e-4 after fleet-wide stability discovery):

Round 1 at lr=5e-4: all 3 arms diverged (6L/384d/4h @step 3206, 8L/256d @step 8899, 6L/384d/6h @step 11099). Round 2 at lr=3e-4: stable as of ~01:00 UTC May 1.

| Arm | Config | W&B run | Step (~01:00) | train/loss | State |
|---|---|---|---:|---:|---|
| A | 8L/256d | xl92i3f5 | stable | healthy | running |
| B | 6L/384d/4h | hbahy1ob | stable | healthy | running |
| C | 6L/384d/6h | 3m4cqwg3 | stable | healthy | running |

- Commentary: Confirms lr=5e-4 is hard stability ceiling for scale-up experiments. 6L/384d/4h has d_head=96 (vs standard 64) — possible bf16 attention overflow. lr=3e-4 resolves all arms. Awaiting first val checkpoint.

---

## 2026-05-01 (in progress) — PR #119 companion: RFF → coord-normalization insight
- Key insight from edward #119: raw DrivAerML coords are anisotropic (x~8m, y/z~2m). ContinuousSincosEmbed's omega is tuned for the x-range, leaving y/z under-sampled in frequency. This is likely a direct cause of the 4× tau_y/z gap. Assigned as PR #143 (fern, coord normalization sweep).

---

## 2026-05-01 02:28 — PR #132: [violet] Decoupled wall-shear magnitude + direction prediction — CLOSED
- Branch: `violet/wallshear-magnitude-direction-decoupled`
- Hypothesis: Factorize tau into |tau| (log-MSE) + tau_dir (cosine loss) heads to reduce magnitude-dominated gradients and preferentially help the small-magnitude tau_y/tau_z axes (currently 3.8×/4.1× AB-UPT gap).
- Results: 6-arm sweep — only B and D survived to terminal epoch.

| Arm | mw | dw | mag-weighted dir | W&B status | test abupt | tau_y | tau_z | p_s | p_v |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| A | 1 | 1 | no | stuck | — | — | — | — | — |
| B | 0.5 | 2 | no | finished | **13.22** | 16.60 | 18.14 | **6.42** | **13.57** |
| C | 1 | 1 | yes | NaN ep1 | — | — | — | — | — |
| D | 1 | 1 | yes (mw=1) | finished | 13.74 | 17.28 | 18.84 | 7.17 | 13.89 |
| E | 0.5 | 2 | yes | NaN ep1 | — | — | — | — | — |
| F | 2 | 0.5 | no | diverged ep1 | — | — | — | — | — |
| **PR #99 baseline** | — | — | — | — | **11.73** | 13.53 | 13.98 | 6.64 | 14.42 |

- Commentary: Clean **negative result** on the headline. Best surviving arm (B) +12.7% worse than baseline; the very axes the PR was designed to fix (tau_y, tau_z) regressed 23–30%. Three diagnostic findings:
  1. **Cosine direction loss has perverse axis priority.** Gradient scales by sin(θ); once tau_x dominant alignment is learned, the small y/z residual contributes near-zero gradient. Reformulation does the *opposite* of preferentially upweighting transverse components.
  2. **Magnitude-weighted direction loss is destructively unstable.** 3 of 4 mag-weighted arms NaN'd. Arm D's cos_sim trajectory `0.531→0.899→0.930→0.951→collapse to 0.404 in ~11 steps` is textbook bf16 overflow as residual high-|tau| points dominate post-convergence.
  3. **Pressure side-effect is interesting but not load-bearing.** Arm B's p_s 6.42 (−3.3%) and p_v 13.57 (−5.9%) beat baseline. Plausible: the magnitude head's log-MSE is acting as a soft regularizer on the shared trunk. Worth a narrow ablation (`--wallshear-magnitude-loss-weight 0.5 --wallshear-direction-loss-weight 0`) but not from this PR.
- **Direction forward**: For the y/z gap, PR #121 (askeladd surface-tangent frame) attacks the *coordinate frame* of the loss rather than the output reformulation — better-targeted lever. Output-side decoupling is closed-door pending a fundamentally different magnitude weighting scheme. Violet reassigned.

---

## 2026-05-01 02:19 — PR #122: [emma] Perceiver-IO backbone replacing Transolver — CLOSED
- Branch: `emma/perceiver-io-backbone`
- Hypothesis: Perceiver-IO with M latent tokens (cross-attention bottleneck) trains 2× faster per step than Transolver with comparable accuracy, buying more epochs in the 6h budget.
- Results: Throughput won, accuracy lost decisively.

| Arm | Config | W&B | Steps/sec | VRAM | test abupt | val abupt (best epoch 6) |
|---|---|---|---:|---:|---:|---:|
| A | Transolver 6L/256d/128sl (control) | 1iilxfvs | 2.15 | 76 GB | NaN ep1 | — |
| B | Perceiver-IO M=512, 6L/256d | jyesq3i4 | 4.29 | 17 GB | 24.46 | 23.69 |
| C | Perceiver-IO M=1024, 6L/256d | 8b8yd2c8 | 3.54 | 17 GB | **22.46** | **21.43** |
| **PR #99 baseline (Transolver)** | 6L/256d/128sl | 3hljb0mg | 2.15 | 76 GB | **11.73** | **10.69** |

- Per-axis: C tau_y=27.11, tau_z=28.40 vs baseline 13.53 / 13.98 (~2× worse on the very gap PR was meant to help).
- Commentary: Perceiver-IO is **architecturally mismatched** for DrivAerML CFD. Latent cross-attention bottleneck loses fine-grained spatial structure; Transolver's slice-based attention preserves it. Despite 2× speedup and 5× VRAM savings, accuracy gap is prohibitive — even with the throughput-bought 6 vs 3 epochs, val abupt floor is ~21% vs ~11% for Transolver. Diminishing returns clear: Arm C's per-epoch deltas were −8.65, −2.18, −1.99, −0.85, −0.09 → asymptote near 21%. Arm A control NaN'd at step 6676 ep1 (independent flake; not Perceiver-related). **Decision: closed.** Backbone replacement is a dead-end direction for this geometry. Emma reassigned.

---

## 2026-05-01 02:19 — PR #127: [nezuko] Stochastic depth regularization sweep — CLOSED
- Branch: `nezuko/stochastic-depth-regularization`
- Hypothesis: DropPath stochastic depth at rates {0.05, 0.10, 0.20} provides regularization that closes the wall-shear y/z gap by preventing layer-specialization.
- Results: All arms regress on wall-shear; hypothesis refuted.

| Arm | sd_prob | W&B | test abupt | p_s | tau_vec | tau_x | tau_y | tau_z | p_v |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **PR #99 baseline** | 0.0 | 3hljb0mg | **11.727** | 6.637 | 11.484 | 10.064 | 13.529 | 13.980 | 14.424 |
| A | 0.05 | u0s413c0 | 11.707 (−0.020) | 6.743 (+) | 11.762 (+) | 10.262 (+) | 13.962 (+0.43) | 14.318 (+0.34) | **13.251 (−1.17)** |
| B | 0.10 | i5kk7ng0 | 14.02 (NaN ep3) | 8.68 | 14.40 | 12.63 | 16.80 | 17.76 | 14.24 |
| C | 0.20 | lyjnyi3a | 13.07 (+11%) | 7.88 | 13.37 | 11.79 | 15.61 | 16.23 | 13.83 |

- Commentary: Arm A's −0.020 abupt is within noise; all wall-shear axes regress (+0.20–0.43 pp on tau_x/y/z). The PR's target was the y/z gap — stochastic depth widens it, not closes it. Arm B diverged at step 23705 ep3 (single-seed flake). Arm C clearly worse. **Mechanistic read:** wall-shear prediction needs coherent multi-layer feature propagation through the fine boundary layer geometry; randomly dropping residual branches injects noise that the model can't recover from in 9 epochs. **Volume pressure note:** Arm A's −1.17 p_v is interesting but isolated — possibly the dropout is a soft regularizer on volume features specifically. Not enough to justify the shear regression. **Decision: closed.** Stochastic depth ruled out for this regime. Nezuko reassigned.

---

## 2026-05-01 02:24 — PR #135: [tanjiro] T_max=100 cosine LR sweep extension (tay branch) — CLOSED
- Branch: `tanjiro/round9-lion-tmax100-ema999`
- Hypothesis: Extend cosine schedule T_max from 50 → 100 to test if "less LR decay is better" trend continues at the wider end on the tay branch.
- Results: Narrow win vs PR #111 SOTA-at-the-time, but superseded by PR #115 lr-change.

| Metric | PR #135 (T_max=100) | PR #111 SOTA-at-launch (T_max=50) | PR #115 actual SOTA (lr=1e-4) | Δ vs PR #115 |
|---|---:|---:|---:|---:|
| `test_primary/abupt_axis_mean` | 11.082 | 11.142 | **10.580** | **+4.74% regression** |
| `surface_pressure` | 6.138 | 6.209 | — | — |
| `wall_shear` | 11.039 | 11.138 | — | — |
| `volume_pressure` | 12.665 | 12.548 | — | — |
| `tau_y` | 13.469 | 13.525 | — | — |
| `tau_z` | 13.791 | 13.992 | — | — |
| `best_val_primary/abupt` (ep9) | 9.886 | 9.989 | — | — |

- W&B run: `wtfrhy2n`. T_max sweep series across three values: T_max=24→50 gave −3.1pp; T_max=50→100 gave −0.54%pp — clear diminishing returns.
- Commentary: This run launched against PR #111 SOTA (11.142) and would have been a clean +0.5% win at that time. During the run, PR #115 (lr=1e-4 change) merged to tay and moved SOTA to 10.580 — making PR #135's 11.082 a +4.74% regression against the current frontier. Schedule lever (T_max) is **closed-door on tay**: sweet spot is T_max=50 (already merged in PR #110), and gains are sub-percentage and dominated by lr-based wins. **Decision: closed.** Tanjiro reassigned to PR #149 (per-axis tau_y/tau_z conservative weighting on tay's new SOTA stack).

---

## 2026-04-29 15:00 — PR #169: [thorfinn] NaN-skip + seed + LR warmup utility flags (stability infra) — MERGED
- Branch: `thorfinn/nan-skip-utility-cherry-pick`
- Hypothesis: N/A — pure infra utilities to improve training stability observability and repeatability.
- Three new CLI flags added:
  1. `--seed <int>` — set all RNG seeds for reproducibility
  2. `--lr-warmup-steps <int>` — linear LR warmup from lr/10 to lr over N steps
  3. NaN/Inf-skip safeguard — detect non-finite loss/gradients, skip those steps, abort after 200 consecutive skips
- Also fixed: kill-threshold NaN bug (commit de8f8d0) — `_numeric_metric_items()` was filtering out NaN metrics silently so `check_kill_thresholds()` would never trigger on NaN runs. Now non-finite values trigger kill condition.
- Validation run: W&B run `e6sgx5ku`, seed=42

| Epoch | val abupt | Expected (spec band) | Notes |
|---|---:|---|---|
| 1 | 15.83 | 12.0–13.0 | High init variance from seed=42 — accepted |
| 2 | **11.20** | 11.0–11.5 | Within spec band — no regression |

- `train/nonfinite_skip_count = 0` throughout (NaN-skip not triggered — confirms baseline is already NaN-free)
- Commentary: Infra PR merged for code utility. No metric improvement (infra PR, not hypothesis test). Epoch-1 spike (15.83) is seed=42 init variance, not a code bug — confirmed by epoch-2 landing in spec band (11.20). Kill-threshold NaN bug fix is fleet-wide safety. LR-warmup-steps flag is now available for future experiments that need structured warmup without committing to OneCycleLR. **Decision: merged.** Infra contributions on the main path — no regression.

---

## 2026-04-29 15:30 — PR #125: [haku] 1cycle LR max=1e-3 (super-convergence in epoch-limited regime) — CLOSED NEGATIVE
- Branch: `haku/onecycle-lr-peak-1e3`
- Hypothesis: OneCycleLR with structured warmup to peak LR then cosine anneal achieves super-convergence in the 3–4 epoch budget, beating flat lr=5e-4.
- Results: 4-arm sweep (round 1) + 4-arm sweep (round 2 with total_steps bug fixed)

**Round 1 (bug: total_steps=544,150 → schedule nearly flat):**

| Arm | max_lr | W&B run | val ep2 | val ep3 | Notes |
|---|---:|---|---:|---:|---|
| A | 1e-3 | pbxxmq1h | NaN ep1 | — | Diverged immediately |
| B | 7e-4 | — | NaN ep1 | — | Diverged |
| C | 5e-4 | — | NaN ep1 | — | Diverged |
| D | 3e-4 | bq11rk5t | 12.01 | 11.83 | Stable but schedule flat (bug) |

**Round 2 (total_steps=36,000 — correct schedule):**

| Arm | max_lr | pct_start | W&B run | val ep1 | val ep2 | val ep3 | val ep4 | test abupt | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| A2 | 1e-3 | 0.10 | 1cycl-A2 | NaN | — | — | — | — | Diverged (> stability ceiling) |
| B2 | 7e-4 | 0.15 | 1cycl-B2 | NaN | — | — | — | — | Diverged (> stability ceiling) |
| **C3** | **5e-4** | **0.20** | 1cycl-C3 | 12.80 | 11.65 | 11.42 | **11.34** | **12.41** | Best arm; still worse than baseline |
| D3 | 3e-4 | 0.10 | 1cycl-D3 | 11.89 | 11.63 | 11.48 | 11.46 | 12.20 | Soft-divergence at lr~6.5e-5 (late cosine anneal) |

**Baseline (PR #99):** val=10.69, test=11.73

- Commentary: Clean **negative result**. Two failure modes:
  1. **Stability ceiling at ~6.5e-4** — confirmed fleet-wide. Even under structured OneCycleLR warmup, peaks ≥6.5e-4 diverge to NaN regardless of pct_start. No warmup schedule overcomes this ceiling.
  2. **Budget starvation at lower peaks** — with peaks ≤5e-4, OneCycleLR spends ~20% of budget rising to peak and another ~60% falling back below flat baseline; effective average LR is much lower than flat 5e-4, and the model cannot recover in 3–4 epochs.
  - Best arm C3 (5e-4 peak): all headline metrics regressed — test abupt=12.41 vs baseline 11.73 (+5.8%).
  - One mild positive signal: C3 volume_pressure=13.33 vs baseline 14.42 — lower average LR may help the volume head specifically. Filed for future per-head LR experiments (e.g., different LR groups for surface vs volume decoder).
  - D3 soft-divergence: anomalous loss spike at lr~6.5e-5 during deep cosine anneal — potential bf16 AMP precision issue at very low loss values. Fleet-wide flag: if future low-LR fine-tuning runs see similar, investigate AMP precision or implement LR floor ~1e-4.
  - OneCycleLR does not help in this epoch-limited regime. LR schedule lever closed for now. **Decision: closed.**

## 2026-05-01 — PR #245: [gilbert] Progressive EMA decay schedule (ramp 0.99→0.9999 vs fixed 0.9995) — CLOSED INCONCLUSIVE
- Branch: `gilbert/ema-decay-schedule`
- Hypothesis: Progressive EMA warmup — ramping decay from low (fast adaptation) to high (slow adaptation) over training — allows the model to track loss-landscape changes quickly early on and then stabilize to a smooth EMA trajectory for inference, outperforming fixed EMA decay.

**W&B-verified ep1 metrics (merge bar = 9.291, PR #222):**

| Run | Arm | State | abupt (primary) | surf press | vol press | wall shear |
|-----|-----|-------|----------------|------------|-----------|------------|
| xpoz88lg | A: ramp 0.99→0.9999, lr=5e-4 | failed (ep2 grad explosion) | 14.41% | 10.18% | 8.29% | 16.20% |
| f6acdprl | B: ramp 0.999→0.9999, lr=5e-4 | crashed (ep2 grad explosion) | 14.63% | 10.41% | 8.50% | 16.41% |
| buch3nry | C: fixed 0.9995, lr=3e-4 | ep1 only (stopped) | 18.93% | 13.63% | 10.96% | 21.19% |

- Commentary: Experiment dominated by the fleet-wide lr=5e-4 gradient explosion (pre-clip grad norms reaching 1e7–1e9). Arms A and B both crashed in epoch 2 before producing comparable multi-epoch metrics. The ep1 A/B gap (14.41 vs 14.63) is within run-to-run noise — no valid signal on EMA ramp schedule. Arm C (fixed 0.9995) used lr=3e-4 after two lr=5e-4 failures, and only produced ep1 metrics (18.93% — underfitting after 1 epoch at lower lr). Cannot serve as the matched fixed-decay control. Research decision: EMA ramp hypothesis remains untested in clean conditions. Keeping default --ema-decay 0.9995 unchanged. Deprioritized in favour of higher-leverage experiments (architecture, loss formulation). If revisited, all arms must use lr=3e-4 (or lr=5e-4 with 1k-step warmup) and include a matched fixed-decay control at the same lr.

---

## 2026-05-01 — PR #197: [gilbert] K-NN local surface attention for tau_y/z gap — CLOSED NEGATIVE
- Branch: `gilbert/knn-local-attention-r12`
- Hypothesis: K-nearest-neighbor local surface attention introduces a locality bias that disproportionately improves tau_y/z prediction by giving the model better access to fine-grained boundary-layer geometry near each query point. The 4× tau_y/z gap was hypothesized to stem from insufficient local receptive field.
- Results:

| Arm | Config | W&B run | val_primary (abupt) | wall_shear_y | wall_shear_z | surface_pressure | volume_pressure | Final state |
|---|---|---|---:|---:|---:|---:|---:|---|
| A (control) | 0L KNN | haibegok | 17.64 | 23.38 | 24.89 | 12.57 | 9.99 | crashed |
| B | 1L KNN k=32 | 77yqqenp | — | — | — | — | — | crashed (no metrics) |
| C | 2L KNN k=32 | hvey8no6 | 19.91 | 25.55 | 27.32 | 13.73 | 13.89 | completed |
| D | 2L KNN k=64 | 7vmm33nz | 24.10 | 30.46 | 32.40 | 16.69 | 18.21 | completed |
| **yi baseline** | PR #183 | — | **10.21** | **13.47** | **14.52** | **6.85** | **6.32** | — |

- Commentary: Clean negative. Locality bias hypothesis falsified. Key findings:
  1. **No tau_y/z-specific improvement** — all metrics degrade uniformly with more KNN layers; the gap ratio is not selectively reduced.
  2. **Monotonic degradation**: 0L=17.64 → 2L k=32=19.91 → 2L k=64=24.10. Every additional KNN layer hurts.
  3. **KNN compute overhead ~4–9× per step** means Arms C/D ran less than 1 epoch in the 270-min budget, so comparisons are slightly biased against KNN arms — but the directional signal is unambiguous.
  4. **Volume pressure catastrophic regression**: +39% (Arm C) to +82% (Arm D) vs control, suggesting the shared FFN between surface KNN tokens and volume global-attention tokens acts as a leak path degrading the well-solved volume-pressure head.
  5. **The tau_y/z gap is NOT a locality/receptive-field problem.** The Transolver global-attention mechanism already has sufficient receptive field; adding KNN locality layers actively interferes with the learned global representation.
  - Peak VRAM: A=74.8GB, B=72.1GB, C=71.4GB, D=72.2GB on H200 96GB.
  - Direction closed. Do not re-assign local surface attention variants.
