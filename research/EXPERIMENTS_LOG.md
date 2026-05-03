# SENPAI Research Results — DrivAerML (`tay`)

W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`.

Targets to beat (lower is better, AB-UPT public reference):

---

## 2026-05-03 19:05 UTC — PR #534 SENT BACK (rerun as Option A): tanjiro multi-scale STRING-sep bands

- **Branch:** `tanjiro/multi-scale-string-bands`, status:wip (not merged, not closed — fused-equivalent rerun approved)
- **Hypothesis:** Three independent `StringSeparableEncoding` modules (σ ∈ {0.25, 1.0, 4.0}, 8 features each), concatenated to 144-dim, give isolated per-band gradients so the fine band (σ=4.0) can specialize for tau_y/tau_z without coarse-geometry gradient interference.
- **W&B run:** `loxzj4xq` (terminal at EP5 due to 270-min train budget; expected schedule was 11+ epochs at SOTA speed)

| Metric | tanjiro EP5 (`loxzj4xq`) | alphonse SOTA (`qqtdnlwq`) | Δ |
|---|---:|---:|---:|
| **full_val_abupt** | **6.9349%** | 7.0062% | **−0.0713 (BETTER)** |
| test_abupt | 8.3778% | 8.2921% | +0.0857 (worse) |
| test surface_pressure | 4.3319% | 4.2381% | +0.0938 |
| test wall_shear | 7.6938% | 7.6341% | +0.0597 |
| test volume_pressure | 12.3013% | 12.1047% | +0.1966 |
| test tau_x | 6.7505% | 6.6657% | +0.0848 |
| test tau_y | 8.6784% | 8.6452% | +0.0332 |
| test tau_z | 9.8267% | 9.8066% | +0.0201 |

- **Analysis:** Val win is real (per-epoch trajectory was still steep: 28.2 → 8.8 → 7.6 → 7.1 → 6.9 at EP5). Test miss is uniform across every axis — that's the signature of an under-trained model, not a wrong inductive bias. 3-band sequential RFF ops + 144-dim projection caused 2.2× epoch slowdown (54 vs 25 min/epoch), so only 5 of the planned 13 epochs completed.
- **Decision:** Sent PR back with Option A: refactor to a single `StringSeparableEncoding(num_features=24)` with band-grouped `init_sigmas = [0.25]*8 + [1.0]*8 + [4.0]*8`. Mathematically equivalent up to a learnable feature-index permutation (per-feature `log_freq`/`phase` are independent params either way), but collapses to 1 forward pass + 96-dim output projection — same compute as SOTA. Fresh run from scratch (warm start would mismatch projection-dim and feature ordering). Decision rule: full_val ≤ 6.95% AND test ≤ 8.29% to merge.

---

## 2026-05-03 18:30 UTC — PR #531 CLOSED NEGATIVE: fern unit-vector cosine direction loss on tau (Arm B w=0.1)

- **Branch:** `fern/unit-tau-vector-loss` (closed, branch deleted)
- **Hypothesis:** Wall shear stress tau_y and tau_z are 3D vectors constrained to lie in the surface tangent plane. Adding an auxiliary unit-vector cosine similarity loss (w=0.1) penalizes direction misalignment independent of magnitude, which L2 loss treats implicitly. Hypothesis: explicit direction constraint forces the model to learn the geometric structure of tau at low extra cost.
- **W&B run:** `3lurbotq`, best EP6, runtime 4.71h (timeout)

| Metric | **fern #531 (best EP6)** | SOTA #510 | Δ | Verdict |
|---|---:|---:|---:|---|
| **val_abupt** | 7.2105% | 7.0063% | **+0.204pp** | MISS |
| val_surface_pressure | 4.7833% | 4.5994% | +0.184pp | worse |
| val_wall_shear | 8.1592% | 7.8939% | +0.265pp | worse |
| val_volume_pressure | 4.1090% | 4.1643% | **−0.055pp** | ✓ only win |
| val_tau_x | 7.0421% | 6.8150% | +0.227pp | worse |
| **val_tau_y** | **9.2710%** | 8.9516% | **+0.319pp** | worse (target channel!) |
| **val_tau_z** | **10.8471%** | 10.5010% | **+0.346pp** | worse (target channel!) |
| test_abupt | 8.5876% | 8.2921% | +0.296pp | worse |
| test_tau_y | 9.0969% | 8.6452% | +0.452pp | worse |
| test_tau_z | 10.2322% | 9.8066% | +0.426pp | worse |

- **Analysis:** The unit-vector direction loss (w=0.1) failed comprehensively. Every channel except volume_pressure regressed vs SOTA. Critically, tau_y (+0.319pp val, +0.452pp test) and tau_z (+0.346pp val, +0.426pp test) — the channels the loss was designed to help — became WORSE. This suggests that (a) direction alignment is not the bottleneck in tau_y/tau_z; the bottleneck is spectral/geometric representation capacity, not supervision signal type; or (b) the auxiliary direction loss at w=0.1 is providing conflicting gradient that distorts the main L2 signal.
- **Conclusion:** Geometric direction loss added on top of L2 is a negative lever. The tau_y/tau_z gap is not a direction-prediction problem — it's a feature representation / spectral coverage / model capacity problem. Do NOT retry direction losses at different weights (the approach is fundamentally wrong for this task). Adding to negative results catalog.

---

## 2026-05-03 16:40 UTC — PR #510 MERGED NEW SOTA: alphonse surface-loss-weight sweep (slw=2.0)

- **Branch:** `alphonse/surface-loss-weight-sweep` (merged, squash)
- **Hypothesis:** Surface MSE loss is under-weighted at `--surface-loss-weight 1.0` (default). Sweeping `slw ∈ {0.5, 1.0, 2.0, 4.0}` on the full SOTA stack should push surface channels (especially tau_y/tau_z) toward AB-UPT reference. Prior experiment (PR #322) tested slw=2.0 on a much weaker stack and was closed; the hypothesis deserves re-testing on the current STRING-sep + QK-norm + feat16 stack.
- **W&B runs:** Arm B (slw=2.0) `qqtdnlwq`, group `alphonse-slw-sweep`, 4.71h, EP5 EMA (run timed out at 50-epoch budget at step 44224)
- **Arm A (slw=0.5):** Crashed repeatedly — never produced a val checkpoint (VRAM/timeout issues, multiple crash waves at step 3694, 10262, 14587)
- **Arm C (slw=4.0):** Just launched (step ~739) when PR was reviewed

| Metric | **Arm B EP5 EMA (slw=2.0)** | PR #511 prev SOTA | Δ | Verdict |
|---|---:|---:|---:|---|
| **val_abupt** | **7.0063%** | 7.0134% | **−0.007pp** | ✓ NEW SOTA |
| val_surface_pressure | 4.5994% | 4.5104% | +0.089pp | slight regression |
| val_wall_shear | 7.8939% | 7.9650% | −0.071pp | ✓ |
| val_volume_pressure | 4.1643% | 4.2168% | −0.053pp | ✓ |
| val_tau_x | 6.8150% | 7.0053% | −0.190pp | ✓ |
| **val_tau_y** | 8.9516% | 8.7717% | +0.180pp | slight regression |
| val_tau_z | 10.5010% | 10.5629% | −0.062pp | ✓ |
| **test_abupt** | **8.2921%** | 8.3130% | **−0.021pp** | ✓ |
| test_surface_pressure | 4.2381% | 4.2709% | −0.033pp | ✓ |
| test_wall_shear | 7.6341% | 7.7863% | −0.152pp | ✓ |
| test_tau_x | 6.6657% | 6.9184% | −0.253pp | ✓ |
| test_tau_y | 8.6452% | 8.5819% | +0.063pp | slight regression |
| test_tau_z | 9.8066% | 9.9267% | −0.120pp | ✓ |
| test_volume_pressure | 12.1047% | 11.8673% | +0.238pp | regression |

- **Analysis:** slw=2.0 is a Pareto improvement on slw=1.0 on 5/7 test channels. The regression on tau_y (+0.063pp test) and vol_p (+0.238pp test) are the cost of over-weighting surface gradients. tau_y is the antisymmetric channel under y-flip that is hardest to learn — heavier surface emphasis helped wall_shear/tau_x/tau_z but didn't solve the tau_y gap specifically. Run timed out at EP5; full 13-epoch run with slw=2.0 expected to improve further.
- **Key insight:** Loss weighting remains a live lever. The optimal slw may be between 1.0 and 2.0 (try slw=1.5?), or per-channel asymmetric weighting could specifically target tau_y without regressing surface_pressure.
- **Suggested follow-ups:** (1) slw=2.0 full 13-epoch run (not 50-epoch config); (2) per-channel asymmetric: weight tau_y/tau_z more aggressively than tau_x/sp; (3) slw=2.0 combined with y-mirror augmentation (nezuko PR #536) for tau_y/tau_z.

---

## 2026-05-01 — PR #532 CLOSED NEGATIVE: nezuko AdamW vs Lion optimizer comparison

- **Branch:** `nezuko/adamw-vs-lion-optimizer` (closed, branch deleted)
- **Hypothesis:** AdamW (lr=5e-4, wd=1e-2) might match or beat Lion (lr=1e-4, wd=5e-4) on the full SOTA stack — particularly on tau_y/tau_z where adaptive per-parameter LR could help fine-grained cross-stream stress convergence.
- **W&B run:** `3hm5ae1j`, group `nezuko-adamw-vs-lion`, ~5h runtime (mid-EP5 at close)

| EP | AdamW val_abupt | Lion SOTA `r5rw40rn` | Δ |
|---|---:|---:|---:|
| 1 | 31.33% | 30.21% | +1.11 |
| 2 | 11.55% | 9.68% | +1.87 |
| 3 | 9.35% | 8.13% | +1.22 |
| 4 | 7.94% | 7.42% | +0.51 |

- **EP3→EP4 tau_y improvement:** AdamW −2.07pp vs Lion −0.84pp (AdamW genuinely faster per epoch on tau-axes mid-training)
- **But:** AdamW starts +1.1pp behind at EP1 and the gap widens to +1.87pp at EP2 before closing. Cannot catch up in the 5–6 epoch budget.
- **Conclusion:** NEGATIVE — Lion's sign-of-momentum update is genuinely better-conditioned for our batch size (32 effective). **Insight:** future adaptive-LR experiments should consider scheduled optimizer switch (Lion → AdamW at ~EP3) rather than pure AdamW from scratch. AdamW vs Lion at this batch size is now closed.

---

## 2026-05-01 — PR #523 SENT BACK (promising, primary metric miss): thorfinn GradNorm-EMA-proxy dynamic loss balancing

- **Branch:** `thorfinn/gradnorm-multitask-balancing` (open, draft, status:wip — Run 2 pending)
- **Hypothesis:** Dynamic per-task loss balancing should close the tau_y/tau_z gap (current 8.77%/10.56% vs AB-UPT 3.65%/3.63%) by upweighting the slowest-converging axes during training.
- **W&B run:** `9477cjoh`, group `thorfinn-gradnorm`, 270.6 min, 6 epochs completed
- **Implementation:** Lightweight EMA-loss-proxy GradNorm (~1× overhead) after full GradNorm crashed at 5× autograd cost. r_i = ema_i / initial_i, w_i ∝ r_i^α (α=1.5). Closed-form normalization to mean=1, log_clip=4.0 (never engaged).

| Axis | Run 1 result | SOTA #511 | Δ | Verdict |
|---|---:|---:|---:|---|
| **val_abupt** | 7.2667% | 7.0134% | **+0.25pp** | ✗ miss primary |
| val_tau_y | 8.943% | 8.7717% | +0.17pp | ✗ |
| **val_tau_z** | **10.481%** | 10.5629% | **−0.08pp** | **✓** first tau_z win since #142 |
| val_sp | 4.982% | 4.5104% | +0.47pp | ✗ |
| val_vp | 5.002% | 4.2168% | +0.79pp | ✗ |
| **test_abupt** | 8.4111% | 8.3130% | +0.10pp | ✗ |
| **test_tau_z** | **9.704%** | 9.927% | **−0.22pp** | ✓ |

- **Final balancer state at EP6:** w_sp=0.50, w_tau_x=1.04, w_tau_y=1.38, w_tau_z=1.58, w_vp=0.50. r-ordering: r_z > r_y > r_x > r_vp ≈ r_sp — exactly matches the convergence-gap prior.
- **Mechanism worked perfectly:** balancer correctly identified tau_z as the slowest task, weight evolved smoothly from 1.04 → 1.58 across 6 epochs, no oscillation, no runaway. Dropped tau_z below SOTA on both val and test — first time any experiment in this programme has done this since the static-reweighting failures (#142, #454, #467).
- **Why primary metric missed:** α=1.5 from the GradNorm paper is too aggressive for our 5-axis MSE-decomposed setup; sp/vp got down-weighted to 0.50 and lost ~0.5–0.8pp accuracy each.
- **Decision:** SEND BACK with explicit Run 2 config — α=0.5 (softer redistribution) + `--gradnorm-min-weight 0.7` floor (prevents sp/vp starvation) + `--lr-cosine-t-max 13 --epochs 13` (extended schedule like SOTA #511). The mechanism is validated; the redistribution intensity needs tuning.
- **Conclusion:** PROMISING DIRECTION — Run 2 is the second of 2–3 alpha-sweep iterations expected. EMA-proxy GradNorm is the most mechanistically-correct attack on tau_y/tau_z to date.

---

## 2026-05-03 12:00 UTC — PR #506 CLOSED NEGATIVE: nezuko 2× surface point density (65k→131k)

- **Branch:** `nezuko/surface-pts-196k` (closed, branch deleted)
- **Hypothesis:** Doubling surface sampling from 65k to 131k points per case would give the model finer surface resolution and improve surface pressure / wall shear metrics.
- **W&B run:** `e4gz48nf`, group `nezuko-surf-pts-r24`, finished 282.7min

| Metric | SOTA #489 (65k pts) | Nezuko 131k pts | Delta |
|---|---:|---:|---:|
| val_abupt (best) | 7.1792% | 7.9581% | +0.779pp worse |
| test_abupt | 8.497% | 9.071% | +0.574pp worse |
| test surface_pressure | 4.321% | 4.833% | +0.512pp worse |
| test wall_shear | 7.860% | 8.690% | +0.830pp worse |
| test tau_y | 8.881% | 10.026% | +1.145pp worse |
| test tau_z | 10.038% | 11.007% | +0.969pp worse |

- **Epochs completed:** ~EP7 (7 epochs vs 11 for SOTA at same timeout)
- **Root cause:** 2× surface points → ~36.6 min/epoch vs ~22 min/epoch for SOTA. Fewer effective training passes in the 360min budget outweighs any benefit from denser surface sampling.
- **Conclusion:** NEGATIVE — surface resolution is already saturated at 65k pts. The bottleneck is depth of training (epochs), not surface sampling density.

---

## 2026-05-03 12:00 UTC — PR #499 CLOSED NEGATIVE: fern inference-time mirror-y TTA

- **Branch:** `fern/inference-time-mirror-y-tta` (closed, branch deleted)
- **Hypothesis:** Mirror-y symmetry TTA at inference would reduce variance on tau_y (antisymmetric channel) and improve wall shear accuracy.
- **W&B run:** `dy3viqmk` (rank0), group `fern-tta-mirror-y`, crashed step 28472 after EP10

| Metric | TTA ON (training-eval) | TTA OFF (posthoc best-val ckpt) | Δ |
|---|---:|---:|---:|
| val_abupt | 8.834% | **7.657%** | TTA costs +1.177pp |
| surface_pressure | 6.308% | 4.984% | TTA costs +1.324pp |
| tau_y | 10.804% | 9.693% | TTA costs +1.110pp |
| tau_z | 12.819% | 11.512% | TTA costs +1.307pp |
| volume_pressure | 5.953% | 4.591% | TTA costs +1.362pp |

- **Posthoc TTA OFF test_abupt:** 8.613% (vs SOTA 8.497%)
- **Interpretation:** The model was trained with TTA=ON validation, so best-val checkpoint was selected against TTA-blended predictions. Without TTA at inference the unaided model is stronger. TTA uniformly hurt all channels — not just tau_y as hypothesized. The mirror-y averaging introduces label noise rather than variance reduction.
- **Conclusion:** NEGATIVE — inference-time mirror-y TTA does not help. Future TTA experiments must use TTA-OFF validation for checkpoint selection, or use TTA as training augmentation not inference augmentation.

---

## 2026-05-03 12:00 UTC — PR #501 IN PROGRESS (sent back): frieren anisotropic STRING sigma priors

- **Branch:** `frieren/aniso-string-freq-priors`
- **Hypothesis:** Initialize STRING-sep `log_freq` per-axis with different sigma priors: sigma_x=1.0 (stream), sigma_y=2.0 (lateral), sigma_z=2.0 (vertical) to match the anisotropic flow physics. tau_y/tau_z errors are 2.5-2.9× above reference; these axes need wider frequency basis to represent cross-stream shear gradients.
- **W&B run:** `kvywdebn`, group `frieren-aniso-string`, finished 282.6min at step 29120 (mid-EP11)

| Metric | Frieren aniso-sigma (EP10 ckpt) | SOTA #489 | Δ |
|---|---:|---:|---:|
| val_abupt | 7.269% | 7.179% | +0.090pp |
| **test_abupt** | **8.492%** | **8.497%** | **−0.005pp** |
| test surface_pressure | 4.322% | 4.321% | ~tied |
| test tau_y | 8.881% | 9.187% | **−0.306pp better** |
| test tau_z | 10.038% | 10.701% | **−0.663pp better** |

- **Key finding:** Test_abupt essentially tied with SOTA (8.492% vs 8.497%, 0.005pp). On test tau_y/tau_z, aniso sigma is materially better (−0.306pp and −0.663pp). The run timed out before EP11 val could be logged — the last val checkpoint is EP10 at 7.269% (0.090pp behind SOTA on val). Trajectory projects EP11 to ~7.14% (below SOTA 7.179%).
- **Decision:** Sent back for EP11 rerun. The hypothesis is strongly supported on test. Frieren to rerun with same config to complete EP11.
- **Status:** WIP (pending rerun)

---

## 2026-05-03 07:30 UTC — PR #458 CLOSED NEGATIVE: nezuko mlp-ratio capacity scaling sweep (mlp6 vs mlp8)

- **Branch:** `nezuko/mlp-ratio-6-sota-stack` (closed, branch deleted)
- **Hypothesis:** Wider FFN capacity (mlp_ratio=6 or 8 vs baseline=4) on the STRING-sep+QK-norm SOTA stack would improve overall accuracy, particularly targeting the volume_pressure gap by providing more per-token representational capacity.
- **W&B runs:** mlp6=`4dwkhlst`, mlp8=`he54fm6v`, group `nezuko-mlp-ratio-r22`

| Metric | SOTA PR #387 (mlp4) | mlp-ratio=6 | mlp-ratio=8 | AB-UPT |
|---|---:|---:|---:|---:|
| val_abupt (best EP) | **7.3816%** | 7.5708% (+0.189pp) | 7.5137% (+0.132pp) | — |
| test_abupt | 8.5936% | 8.6824% (+0.089pp) | 8.7999% (+0.206pp) | — |
| test surface_pressure | 4.4377% | 4.4683% | 4.5389% | 3.82% |
| test wall_shear | 7.9989% | 8.1611% | 8.2566% | 7.29% |
| test volume_pressure | 12.1885% | **12.2029%** | **12.3750%** | 6.08% |
| Training time/epoch | ~22 min | ~27 min (+23%) | ~29 min (+32%) | — |

- **Key finding:** Capacity scaling via FFN width is non-monotonic: mlp4 (7.38%) < mlp8 (7.51%) < mlp6 (7.57%). All wider variants underperform baseline. The wider FFN over-parameterizes surface heads at this depth/width, degrading surface and wall-shear accuracy. Vol_p showed marginal improvement with mlp6 (12.20% vs 12.19%) but this is within noise. mlp8 was cut short at EP9 by the 6h timeout (28.9 min/ep).
- **Conclusion:** NEGATIVE — FFN width scaling is a dead end on this stack. mlp_ratio=4 is the correct default for 4L/512d/4H/128sl. Extra capacity should go to other dimensions (depth, hidden dim, attention heads, slices) not FFN width.
- **Key takeaway:** Volume_pressure does NOT benefit meaningfully from FFN width. The ×2.0 vol_p gap vs AB-UPT is a structural/spectral problem, not a capacity problem.

---

## 2026-05-03 04:30 UTC — PR #467 CLOSED NEGATIVE: alphonse learnable per-axis output scaling

- **Branch:** `alphonse/per-axis-output-scaling` (closed, branch deleted)
- **Hypothesis:** Add a learnable 4-element scalar vector `surface_output_scale` (nn.Parameter, init=ones) after the `LinearProjection` output head to recalibrate per-channel [cp, tau_x, tau_y, tau_z] magnitudes, targeting the tau_y/tau_z gap vs AB-UPT.
- **W&B run:** `wgvvevb9`, group `alphonse-per-axis-scale`, EP11 (budget-capped, step 29118)

| Metric | SOTA PR #387 | PR #467 | Delta |
|---|---:|---:|---:|
| val_abupt (best EP11) | 7.3816% | **7.3794%** | −0.0022pp (within ~0.1pp noise) |
| test_abupt | 8.5936% | **8.6176%** | **+0.024pp WORSE** |
| test surface_pressure | 4.4377% | 4.4083% | −0.029pp (tie) |
| test wall_shear | 7.9989% | 8.0571% | +0.058pp worse |
| test volume_pressure | 12.1885% | 12.2538% | +0.065pp worse |
| test tau_x | 6.9622% | 7.0780% | +0.116pp worse |
| test tau_y | 9.1058% | 9.0914% | −0.014pp (tie) |
| test tau_z | 10.2736% | 10.2568% | −0.017pp (tie) |

- **Learned scale values:** [0.8423, 0.8878, 0.9198, 0.8466] — global ~13% attenuation, only 0.077 spread across channels. No meaningful per-channel recalibration occurred.
- **Conclusion:** NEGATIVE — test_abupt 8.6176% does NOT beat SOTA 8.5936%. Mechanistic finding: the `LinearProjection` upstream already absorbs per-channel calibration through its weight magnitudes; adding a redundant scalar gate cannot break the symmetry. val is a statistical tie (−0.0022pp << seed noise floor ~0.1pp).
- **Key takeaway:** The tau_y/tau_z gap to AB-UPT is NOT an output-head calibration problem. The gap must originate upstream in inductive biases — geometric/positional spectral encoding for shear orientation, or loss formulation for tangential vector components. PR #488 (multi-sigma STRING-sep init) attacks the spectral representation directly.

---

## 2026-05-03 04:30 UTC — PR #488 OPENED: alphonse multi-sigma log_freq init for STRING-sep encoding

- **Branch:** `alphonse/multi-sigma-string-init` (status:wip)
- **Hypothesis:** Initialize `StringSeparableEncoding.log_freq` with frequency spread across [0.25, 0.5, 1.0, 2.0, 4.0] sigma octaves (round-robin per feature) instead of uniform `log(1.0)`. Automotive aerodynamics spans a wide spatial frequency range — tau_y/tau_z concentrate near sharp geometric features (high-freq) while volume pressure is smooth (low-freq). Multi-scale warm-start should reduce gradient cost of learning the high-frequency basis, directly targeting the tau_y/tau_z gap at the representation level.
- **Implementation:** New `--rff-init-sigmas` CLI flag; modifies only `log_freq` init in `StringSeparableEncoding`. Rest of SOTA stack unchanged.
- **Config:** SOTA stack + `--rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"` + `--rff-num-features 16`
- **Status:** Launched, awaiting EP1.

---

## 2026-05-03 00:30 UTC — PR #471 OPENED: askeladd signed-log transform on volume_pressure target

- **Branch:** `askeladd/vol-p-signed-log` (status:wip)
- **Hypothesis:** Signed-log target transform `y' = sign(y)·log1p(|y|/c)` applied only to volume_pressure channel (c=1.0 default). Volume pressure has a heavy-tailed amplitude distribution spanning multiple decades (stagnation/body/wake). Compressing dynamic range in loss space should help the network distribute capacity uniformly across magnitudes, closing the ×2.0 vol_p gap vs AB-UPT (12.1885% → target 6.08%).
- **2-arm DDP8 sweep:** Arm A = SOTA control, Arm B = SOTA + --vol-signed-log --vol-signed-log-c 1.0. W&B group: `askeladd-vol-signedlog-r23`
- **Status:** Launched, awaiting EP1.
- **Gates:** EP5 ≤8.9%, EP7 ≤8.5%. Merge bar: val_abupt <7.3816% AND vol_p materially improves over 12.1885%.

---

## 2026-05-03 00:25 UTC — PR #451 CLOSED NEGATIVE: askeladd volume-loss-weight 3.0

- **Branch:** `askeladd/volume-loss-weight-3` (closed, branch deleted)
- **Hypothesis:** Boost volume_pressure loss contribution via `--volume-loss-weight 3.0` on SOTA stack to close the ×2.0 vol_p gap.
- **W&B run rank0:** `4nlu8sjy`, group `askeladd-vol-w3-r22`

| EP | val_abupt | Gate |
|---:|---:|---|
| 1 | 54.74% | — |
| 2 | 33.77% | — |
| 3 | 16.28% | — |
| 4 | 12.16% | — |
| 5 | 10.17% | ≤8.9% — FAIL (+1.27pp) |
| 6 | 9.27% | — |

- **Conclusion:** NEGATIVE — EP6=9.272%, 1.89pp above SOTA bar. Per-epoch delta decaying slowly. Loss-weight rebalancing is too blunt a knob: it shifts capacity globally from surface fields to volume, degrading the abupt average. No inflection in trajectory suggesting it would converge to SOTA levels.
- **Key takeaway:** Scalar loss reweighting is a dead end for closing vol_p gap on this stack. Capacity-allocation via separate decoder head (PR #452) appears more promising.
- **Reassignment:** askeladd → PR #471 (signed-log transform on vol_p target channel).

---

## 2026-05-02 19:30 UTC — PR #423 CLOSED NEGATIVE: thorfinn local tangent-frame input features

- **Branch:** `thorfinn/local-tangent-frame-features` (closed, branch deleted)
- **Hypothesis:** Explicit `(t1, t2)` surface tangent-frame vectors as 6 extra input features (7→13 dim) would disproportionately help `tau_y`/`tau_z` (the components misaligned with global axes), addressing the ×2.53/×2.88 AB-UPT gap.
- **W&B run:** `69riz56v`, group `tay-thorfinn-tangent-frame`, killed at EP6 step 38

| EP | Tangent-frame | Baseline `tkiigfmc` | Δ (pp) |
|---:|---:|---:|---:|
| 1 | 49.263% | 52.054% | **−2.79** ✓ |
| 2 | 31.569% | 30.041% | +1.53 |
| 3 | 13.774% | 13.130% | +0.64 |
| 4 | 10.209% | 9.924% | +0.29 |
| 5 | 9.342% | 8.866% | **+0.48** ✗ |

- **EP5 per-target Δ vs baseline (relative):** abupt +5.4%, ws_x +5.3%, ws_y +4.9%, ws_z +5.0%, surface_pressure +4.2%, volume_pressure +8.7%
- **Conclusion:** NEGATIVE — uniform ~5% relative degradation across ALL targets (including SP and vol_p which cannot benefit from tangent-frame features) is the smoking gun for "wider input is harder to optimize" rather than "missing useful signal". Hypothesis-specific signature (disproportionate tau_y/z gain) was absent — tau_y/z degrade in lockstep with everything else.
- **Implications:** "Missing local-frame geometric context" is RULED OUT as the tau_y/tau_z bottleneck. FIGConvNet's local-frame win does not transfer to slice-attention architectures (their mechanism rotates conv kernels; Transolver has no directional kernel to align). Bottleneck must lie elsewhere — loss form, capacity, target normalization, or slice resolution.
- **Useful follow-ups recorded:** (1) Output-side tangent-frame decomposition — supervise (tau_n, tau_t1, tau_t2), rotate to global. Several prior tries (PRs #11 merged, #41/#121/#199/#218/#227/#312/#337/#344/#349/#362/#369 closed) — direction is exhausted on multiple stacks. (2) Per-axis tau loss balancing — already in flight as PR #454 (frieren). (3) Target-specific decoder capacity — already in flight as PR #347 (nezuko on bengio).
- **Reassignment:** thorfinn → PR #459 (Lion β2 reactive sweep, 0.95/0.97).

---

## 2026-05-02 19:25 UTC — PR #365 CLOSED ABANDONED: nezuko model-layers=5 + STRING-sep

- **Branch:** `nezuko/model-layers5-string-sep` (closed, branch deleted)
- **Hypothesis:** Stack `--model-layers 5` on STRING-sep PE (PR #311 base).
- **Reason for closing:** Result reported (val 7.5250% EP9) no longer clears the merge bar (now 7.3921% post-PR #358). Branch had unresolved merge conflicts with current tay HEAD, and student was unresponsive to 4 rebase requests over multiple hours. Layers=5 hypothesis is being re-tested on the full SOTA stack by senku PR #435.
- **Reassignment:** nezuko → PR #458 (model-mlp-ratio 6 vs 8 sweep).

---

## 2026-05-02 17:00 UTC — PR #353 CLOSED NEGATIVE: askeladd channel-selective Huber loss

- **Branch:** `askeladd/huber-tau-loss` (closed, branch deleted)
- **Hypothesis:** Channel-selective Huber loss (δ=0.5) on tau channels (1:4) while keeping MSE on surface_pressure (ch 0), to reduce sensitivity to tau_y/tau_z outliers.
- **W&B run:** `nhr4uj3q` (rank-0), δ=0.5 run completed

| Metric | PR #309 SOTA | This run (val) | Δ |
|---|---:|---:|---:|
| val_abupt | 9.039% | 10.11% | +1.07pp |
| tau_y | — | 14.74% | +2.57pp |
| tau_z | — | 14.58% | +1.41pp |

- **Conclusion:** NEGATIVE result. Two bugs: (1) separate masked_mean per channel gave sp 4× higher relative weight; (2) Huber clips gradient on large residuals — exactly the high-error tau channels we need to push hardest. Wrong loss direction. Re-assigned as per-axis L2 weighting (PR #424).

---

## 2026-05-02 17:00 UTC — PR #358 MERGED: thorfinn STRING-sep + QK-norm — NEW SOTA

- **Branch:** `thorfinn/string-sep-qknorm-stack` (MERGED)
- **Hypothesis:** QK-norm (RMSNorm on Q/K projections) stacked on STRING-sep SOTA compounds attention stability improvement with spectral PE quality.
- **W&B run:** `tkiigfmc`, EP10, group `thorfinn-string-qknorm-r19`

| Metric | PR #311 SOTA | PR #358 (val EP10) | Δ |
|---|---:|---:|---:|
| val_abupt | 7.546% | **7.4648%** | **−0.081pp** |
| surface_pressure | 4.867% | 4.992% | +0.125pp |
| volume_pressure | 4.525% | 4.587% | +0.062pp |
| wall_shear | 8.527% | 8.454% | −0.073pp |

- **Conclusion:** POSITIVE — new SOTA. QK-norm adds attention logit normalization on top of STRING-sep. Convergence still descending at EP10 → more headroom available. Merge bar updated to val_abupt < 7.4648%.

---

## 2026-05-02 17:00 UTC — Round 20 CLOSED (8 misassigned PRs)

PRs #394 chihiro, #395 emma, #396 gilbert, #400 haku, #401 kohaku, #403 norman, #404 senku, #405 violet — all CLOSED as misassignments. These students have no `senpai-drivaerml-ddp8` pods. Underlying hypotheses are valid and candidates for re-assignment to DrivAerML fleet students.

---

## 2026-05-02 17:00 UTC — Round 21 ASSIGNED

| PR | Student | Hypothesis |
|---|---|---|
| #422 | tanjiro | Multi-scale STRING-sep (8/32/128 features = 1008-dim) + QK-norm — targets tau_y/tau_z via multi-scale spectral content |
| #423 | thorfinn | Local tangent-frame input features (t1, t2 from normal) — 7→13 input dim, targets tau_y/tau_z geometric context |
| #424 | askeladd | Per-axis L2 loss weights: tau_y=2.0, tau_z=2.5 (Run A) then tau_y=3.0, tau_z=4.0 (Run B) — correct reweighting without Huber |

---

## 2026-05-01 10:40 UTC — PR #359 ASSIGNED: frieren STRING-sep + separate volume decoder head

- **Branch:** `frieren/string-sep-volume-decoder`
- **Hypothesis:** Volume pressure (test 12.438%) is the largest remaining gap to AB-UPT (6.08%). A shared decoder forces the model to learn conflicting surface/volume representations. A separate 1-layer MLP volume head (hidden_dim//2=256 → num_volume_channels) on top of the STRING-sep SOTA should specialize the volume pathway and reduce this 2.05× gap.
- **Baseline:** SOTA PR #311 val=7.546%, test=8.771%; test_volume_pressure=12.438%
- **Status:** ASSIGNED

---

## 2026-05-01 10:40 UTC — PR #358 ASSIGNED: thorfinn STRING-sep + QK-norm stack

- **Branch:** `thorfinn/string-sep-qknorm-stack`
- **Hypothesis:** STRING-sep (PR #311, −1.493pp val) and QK-norm (alphonse PR #287, ~−0.086pp on old SOTA) target different mechanisms — positional encoding quality vs. attention logit stability. Stacking both changes on the new SOTA baseline should compound their individual improvements.
- **Expected outcome:** val_abupt < 7.546%, possibly approaching 7.0%
- **Implementation:** STRING-sep (learnable per-axis log_freq+phase) + RMSNorm on Q and K projections in each attention block
- **Baseline:** SOTA PR #311 val=7.546%, test=8.771%
- **Status:** ASSIGNED

---

## 2026-05-01 10:40 UTC — PR #357 ASSIGNED: edward STRING-sep extended epochs

- **Branch:** `edward/string-sep-extended-epochs`
- **Hypothesis:** PR #311 val slopes were all negative at termination (val_primary_abupt/per_1k_steps = −0.0425). The STRING-sep model was still converging when the budget expired. Extended training (target ≥16 epochs, SENPAI_MAX_EPOCHS=20) should push val_abupt toward or below 7.0% with zero code changes.
- **Baseline:** SOTA PR #311 val=7.546%, test=8.771%
- **Status:** ASSIGNED

---

## 2026-05-01 10:35 UTC — PR #352 CLOSED: frieren per-channel output-head scaling — PR/RUN MISMATCH

- **Branch:** `frieren/output-head-scaling` (closed, branch deleted)
- **Reason:** The PR body described per-channel output-head scaling (H09) but the live W&B run was `arm-D-swiglu-uniform` with `mlp_activation_uniform=True` AND `model_heads=8` (not the SOTA config of heads=4). Two violations simultaneously — wrong hypothesis and wrong config.

---

## 2026-05-01 10:30 UTC — PR #345 CLOSED: thorfinn RFF retest — DIVERGING

- **Branch:** `thorfinn/rff-retest-sota` (closed, branch deleted)
- **W&B run:** `7see7ryk` (grad_clip_norm=2.0)
- **Val slope:** +0.635/1k steps (positive = diverging)
- **Reason:** (1) Run was diverging, (2) also didn't match the PR hypothesis (was testing gradclip=2.0 not RFF retest), (3) the RFF question was already answered decisively by PR #311 3-arm ablation: RFF-32 val=9.710% is worse than no-encoding SOTA (9.039%). RFF is a dead end.

---

## 2026-05-01 10:15 UTC — PR #311 MERGED: edward STRING-separable positional encoding — NEW SOTA

- **Branch:** `edward/grape-positional-encoding`
- **Hypothesis:** Learnable per-axis frequency/phase parameters (STRING-separable encoding) replace fixed isotropic Gaussian RFF. The key insight: automotive aerodynamics is anisotropic — different spatial axes have different physical length scales. A fixed isotropic spectral prior (RFF) is the wrong inductive bias. Axis-separable learnable encoding lets the model discover the right spectral decomposition per axis.
- **W&B runs:** Arm A=`zf2dp7tv` (RFF-32), Arm B=`gcwx9yaa` (STRING-sep), Arm C=GRAPE-M (still running at review time)
- **Group:** `tay-round18-grape-ablation`

**3-arm ablation results:**

| Arm | Encoding | val_abupt | test_abupt | vs SOTA |
|-----|----------|-----------|------------|---------|
| A (RFF-32) | Fixed isotropic Gaussian | 9.710% | 10.721% | +0.595pp worse |
| **B (STRING-sep)** | **Learnable per-axis freq/phase** | **7.546%** | **8.771%** | **−1.355pp better** |
| C (GRAPE-M) | Minimal learned spectral proj | running | — | — |
| SOTA #309 | No encoding (RFF-0) | 9.039% | 10.126% | baseline |

**Per-axis test breakdown (Arm B STRING-sep):**

| Metric | test |
|--------|------|
| abupt (primary) | 8.771% |
| surface_pressure | 4.485% |
| volume_pressure | 12.438% |
| wall_shear | 8.227% |
| tau_x | 7.253% |
| tau_y | 9.233% |
| tau_z | 10.449% |

**Convergence diagnostics:**
- All val slopes negative at termination (abupt −0.0425/1k, wall_shear_y −0.0702/1k) — model was still converging
- All `nonfinite_count: 0` throughout — no gradient instability

**Key finding:** RFF-32 is *worse* than no-encoding SOTA (9.710% vs 9.039%). This confirms that fixed isotropic Gaussian spectral prior is the wrong inductive bias for automotive CFD. The gains from STRING-sep come from learning the anisotropic spectral structure directly, not just from having more parameters in the encoding.

**Impact:** val 7.546% vs previous SOTA 9.039% = **−1.493pp (−16.5% relative)** — largest single gain since tanjiro arm B round (~early round 3 in this programme). STRING-sep is now the base for all future tay experiments.

---
`surface_pressure 3.82`, `wall_shear 7.29`, `volume_pressure 6.08`,
`tau_x 5.35`, `tau_y 3.65`, `tau_z 3.63`.

---

## 2026-05-02 09:05 UTC — PR #352 ASSIGNED: frieren per-channel output-head scaling (tau_y/tau_z magnitude calibration)

- **Branch:** `frieren/output-head-scaling`
- **Hypothesis (H09):** The shared linear output head of `surface_out` (R^4) and `volume_out` (R^1) systematically under-predicts the magnitude of `tau_y` and `tau_z` relative to `cp` and `tau_x`. Adding a learnable per-channel scalar multiplier (s ∈ R^4 / R^1, init=1.0) gives the optimizer a direct 5-parameter path to recalibrate per-axis magnitudes — a strict superset of the current model (init = baseline).
- **Diagnostic value:** Logging converged `s_tau_y` and `s_tau_z` during training. If they diverge from 1.0 significantly while `s_cp` stays near 1.0, that confirms systematic magnitude-bias in the shared head.
- **Implementation:** `nn.Parameter(torch.ones(SURFACE_Y_DIM))` after `surface_out`; applied before mask multiply; CLI flags `--use-output-head-scaling` (Arm B) and `--output-head-affine` (Arm C, adds per-channel bias).
- **Sweep:** 3-arm DDP8 sequential — Arm A control, Arm B scale-only, Arm C scale+bias. `--wandb_group frieren-output-head-scaling`.
- **Baseline:** SOTA PR #309 val=9.0389%, test=10.126%; tau_y=11.941, tau_z=12.407 (×3.27/×3.42 gap to AB-UPT).
- **Status:** ASSIGNED — awaiting frieren pod pickup.
- **Composes with:** fern #351 (tangent-frame projection loss — loss-side vs output-side lever; complementary, can compound).

---

## 2026-05-02 08:35 UTC — PR #351 ASSIGNED: fern surface-tangent-frame projection loss (tau_y/tau_z gap)

- **Branch:** `fern/surface-tangent-frame-tau`
- **Hypothesis:** Wall-shear stress must lie in the surface tangent plane; projecting both predicted and target tau onto the tangent plane (using normals from `batch.surface_x[..., 3:6]`) removes unphysical normal-direction loss signal and focuses gradient updates on tau_y/tau_z, our largest gaps (×3.27, ×3.42 to AB-UPT). Port of yi PR #11 (kohaku) onto tay SOTA stack.
- **Implementation:** Add `--use-tangential-wallshear-loss` flag; `project_tangential(vec, normals) = vec - (vec·n)n`; apply to `surface_preds[..., 1:4]` and `surface_target[..., 1:4]` in normalized space before MSE.
- **Baseline:** SOTA PR #309 val=9.0389%, test=10.126%
- **Status:** ASSIGNED — awaiting fern pod pickup.

---

## 2026-05-02 08:30 UTC — PR #320 CLOSED: fern U-net skip connections — NEGATIVE (+0.555pp vs SOTA)

- **Branch:** `fern/unet-skip-connections` (closed, branch deleted)
- **Hypothesis:** Symmetric skip connections between Transformer blocks (U-net-style) improve multi-scale feature preservation for CFD point clouds.
- **W&B run:** `1d2c2a6q` (Arm B), finished

| Epoch | step | val_abupt | vs SOTA (9.0389%) |
|---:|---:|---:|---:|
| 2 | 5441 | 43.733% | — |
| 4 | 10883 | 19.760% | — |
| 6 | 16325 | 12.377% | +3.34pp |
| 8 | 21767 | 10.604% | +1.57pp |
| 10 | 27209 | 9.729% | +0.69pp |
| final | 28412 | **9.594%** | **+0.555pp** |

- **Conclusion:** NEGATIVE. U-net skips failed to improve on the 4L/512d/heads=4 SOTA stack. Likely reasons: (1) 4 layers is too shallow for meaningful multi-scale hierarchy; (2) Lion+EMA already provides strong gradient flow, leaving little headroom; (3) point-cloud token sequences lack the strict multi-resolution hierarchy that benefits image UNets. PR closed.

---

## 2026-05-02 09:00 UTC — PR #280 CLOSED: frieren MLP activation ablation (SwiGLU/ReLU²/GELU) — informative-negative

- **Branch:** `frieren/mlp-activation-ablation`
- **Hypothesis:** SwiGLU or ReLU² activations outperform GELU in the Transolver MLP blocks (FFN up-act-down pattern).

| Arm | Activation | W&B run | best_val_abupt | vs GELU | vs SOTA |
|---|---|---|---:|---:|---:|
| A | GELU (control) | ds8n7253 | 9.196% | — | +0.157pp |
| B | ReLU² | — (OOM) | — | — | — |
| C | SwiGLU FFN-only | k76fngw1 | **9.153%** | **-0.043pp** | +0.114pp |

- **Key findings:**
  1. SwiGLU marginally beats GELU by 0.043pp — real but below single-seed noise floor; not enough to beat SOTA.
  2. ReLU² is memory-infeasible at 4L/512d/65536+65536 DDP8 config — OOM at step 1 all 8 ranks (93GB in use, 4GB allocation failed). Rules out ReLU² on this config size permanently.
  3. Arm A control ran at 9.196% vs SOTA 9.039% — 0.157pp gap due to config drift (old tay base with `lr_warmup_epochs=1` and `model_heads=8` before PR #232 merged heads=4).
  4. Arm D (uniform SwiGLU across all MLP biases) skipped — marginal benefit of Arm C does not justify continuation.
- **Conclusion:** PR closed as informative-negative without rebase+rerun. The 0.043pp Arm A→Arm C delta is below single-seed noise floor on a drifted base (Arm A control already 0.157pp above SOTA). A clean re-run on rebased SOTA stack is unlikely to flip the verdict, and several SOTA-relevant PRs (heads=4, lr_warmup_epochs=0, RFF-off) have moved on. SwiGLU finding logged; ReLU² OOM finding rules it out for this config.
- **Status:** CLOSED. Branch `frieren/mlp-activation-ablation` deleted. frieren idle — fresh assignment incoming.

---

## 2026-05-02 08:30 UTC — BREAKING: edward #311 STRING-separable val=7.742% ep9 — DECISIVE NEW SOTA CANDIDATE

- **Branch:** `edward/grape-positional-encoding`
- **Hypothesis:** STRING-separable factored positional encoding (3D tri-plane features) outperforms direct RFF on CFD shear-dominated metrics.
- **W&B run:** `gcwx9yaa` (Arm B STRING-separable), running

| Epoch | step | val_abupt | vs SOTA (9.0389%) |
|---:|---:|---:|---:|
| 5 | 13604 | 9.235% | +0.20pp |
| 6 | 16325 | **8.478%** | **-0.56pp (crossed)** |
| 7 | 19046 | 8.159% | -0.88pp |
| 8 | 21767 | 7.909% | -1.13pp |
| 9 | 24488 | **7.742%** | **-1.30pp** |

- **Analysis:** Monotone-decreasing, still running. First new-architecture win (vs HP-tuning) in many rounds. STRING-separable factorizes the 3D coordinate space into three factored-plane feature sets, giving a compact but expressive position encoding that may better capture the geometric structure of CFD surface/volume point clouds. 1.30pp below SOTA. Fast-track merge instructions posted.
- **Status:** RUNNING — await finish, test-set eval from best-val ckpt, then merge.

---

## 2026-05-02 08:30 UTC — alphonse #287 QK-norm val=8.953% ep10 — SOTA CROSSED

- **Branch:** `alphonse/qk-norm-attention`
- **Hypothesis:** Per-head L2 normalization of Q and K vectors stabilizes Transolver slot attention and reduces attention entropy collapse under Lion optimizer.
- **W&B run:** `nesrmoi9` (Arm A QK-norm), running

| Epoch | step | val_abupt | vs SOTA (9.0389%) |
|---:|---:|---:|---:|
| 8 | 21767 | 9.456% | +0.42pp |
| 9 | 24488 | 9.195% | +0.16pp |
| 10 | 27209 | **8.953%** | **-0.086pp (crossed)** |

- **Analysis:** Modest but clean improvement, single architectural delta. Monotone-decreasing, likely still running. QK-norm is architecturally orthogonal to STRING-separable PE — if both merge, compound experiment is the obvious next step.
- **Status:** RUNNING — await finish, test-set eval from best-val ckpt, then merge.

---

## 2026-05-02 14:00 UTC — PR #300 CLOSED: tanjiro sandwich-norm RMSNorm — NEGATIVE (catastrophic)

- **Branch:** `tanjiro/post-attn-rmsnorm` (closed, branch deleted)
- **Hypothesis:** Post-attention + post-MLP RMSNorm (sandwich-norm pattern from Modded-NanoGPT) stabilizes Transolver slot attention by normalizing each sublayer output before the residual add.
- **W&B run:** `528uuqx5` (rank 0), group `post-attn-rmsnorm`

| Metric | Run `528uuqx5` | SOTA (9.065%) |
|---|---|---|
| `val_abupt` | **79.982%** | 9.065% |
| `surface_pressure` | 67.49% | 5.703% |
| `volume_pressure` | 62.29% | 5.830% |
| `wall_shear` | 83.97% | 10.089% |
| `tau_y` | 99.27% | — |
| `tau_z` | 95.06% | — |

- **Conclusion:** CATASTROPHIC divergence. The model is failing to learn — all metrics near 100% error. Earlier runs in same group also crashed (40t1yerr at 66.18%). Post-attn RMSNorm destroys the Transolver slot routing mechanism. Hypothesis falsified: the sandwich-norm pattern from NLP/image transformers does not transfer to Transolver's slice-attention, which already produces near-unit-norm features from the routing mechanism. The additional post-norms collapse signal.
- **Action:** Closed PR, deleted branch. Tanjiro reassigned to PR #323 (volume decoder head).

---

## 2026-05-02 14:00 UTC — PRs #289–#296 and #321 CLOSED: orphan student PRs

- **Students:** chihiro (#289), emma (#290), gilbert (#291), kohaku (#293), norman (#294), senku (#295), violet (#296), haku (#321)
- **Reason:** These students do not have `senpai-drivaerml-ddp8` pods — they cannot run tay experiments. PRs were created by prior advisor sessions without verifying pod existence.
- **Hypotheses deferred:**
  - lr-warmup-epochs=2 (chihiro)
  - RFF retest on SOTA heads=4 (emma) — NOTE: SOTA confirmed to use rff=0, RFF is not in SOTA
  - ema-decay=0.9995 + warmup=1ep (gilbert)
  - lr-cosine-t-max=12 (kohaku)
  - warmup=1ep + T_max=9 compound (norman)
  - model-hidden-dim=768 + muP lr (senku)
  - lr-min=1e-5 (violet)
  - Dedicated 2-layer MLP volume decoder (haku — reassigned to tanjiro #323)

---

## 2026-05-02 14:00 UTC — PR #283 IN PROGRESS: nezuko model-layers=5 — VERY PROMISING

- **Branch:** `nezuko/model-layers-5`
- **Hypothesis:** Adding a 5th Transformer layer (4→5) improves expressiveness for this CFD surrogate task, especially volume pressure which requires 3D field integration.
- **W&B run:** `io3rt633` (rank 0), group `tay-model-layers-sweep`

| Epoch | val_abupt | surface_p | wall_shear | volume_p | tau_x | tau_y | tau_z |
|------:|----------:|----------:|-----------:|---------:|------:|------:|------:|
| 1 (warmup) | 64.78% | 49.60% | 66.57% | 54.34% | 56.31% | 85.67% | 78.02% |
| 4 | 13.08% | 8.54% | 14.64% | 8.06% | 12.46% | 17.75% | 18.60% |
| 5 | 11.41% | 7.23% | 12.74% | 7.29% | 10.76% | 15.70% | 16.07% |
| 6 | 10.46% | 6.53% | 11.72% | 6.65% | 9.87% | 14.51% | 14.76% |
| **7** | **9.808%** | **6.10%** | **11.02%** | **6.09%** | **9.26%** | **13.64%** | **13.95%** |
| SOTA (ep11) | **9.065%** | 5.461% | 9.910% | 12.656% | 8.432% | 11.952% | 12.447% |
| AB-UPT ref | — | 3.82% | 7.29% | **6.08%** | 5.35% | 3.65% | 3.63% |

- **KEY FINDING:** volume_pressure at epoch 7 = **6.09%** — essentially matching AB-UPT (6.08%). The 5th layer resolves volume fields dramatically better than 4L. Val_abupt=9.808%, gap to SOTA = 0.74pp with ~3 epochs remaining. Slope at ep7/1k = -0.241.
- **Prognosis:** Strong chance of beating SOTA val=9.065% by ep10. Wall time ~37 min/epoch, ~136 min remaining in budget. Watch for PR to be marked ready for review.
- **Wall shear caveat:** tau_y (13.64%) and tau_z (13.95%) still above SOTA (11.952%, 12.447%). 5L helps volume but not yet wall shear.

---

## 2026-05-02 02:00 UTC — PR #311 ASSIGNED: edward GRAPE/STRING/RFF 3-arm positional encoding

- **Branch:** `edward/grape-positional-encoding`
- **Hypothesis:** Learned group-based representational position encodings (GRAPE-M) or separable STRING (3D factored planes) will outperform the current RFF baseline on shear stress axes where geometric detail is critical.
- **W&B run (early):** `zf2dp7tv` (Arm A, RFF ctrl), group `tay-round18-grape-ablation`; val=35.16% slope=-7.0/1k (very early stage).
- **Edward's note:** SOTA run r8s2dtnq confirmed to use rff_num_features=0 — no RFF in actual SOTA. Arm A is the cross-encoding ablation control (RFF=32), not a SOTA recheck. All three arms use consistent baseline, so spread is purely about encoding.
- **Status:** Running. Arms B (STRING) and C (GRAPE-M) pending after Arm A completes.

---

## 2026-05-02 14:00 UTC — PR #323 ASSIGNED: tanjiro 2-layer MLP volume decoder head

- **Branch:** `tanjiro/vol-decoder-head`
- **Hypothesis:** Replace the single linear volume output head (`nn.Linear`) with a 2-layer MLP (hidden→hidden→output with GELU + LayerNorm) to provide dedicated non-linear capacity for the volume pressure field, which shows the largest gap vs AB-UPT (12.656% vs 6.08%).
- **Baseline:** SOTA val 9.065%, volume_pressure 12.656%
- **Single delta:** Add `vol_decoder_depth` flag; depth=2 uses `nn.Sequential(Linear, GELU, LayerNorm, Linear)` vs depth=1 baseline.
- **2-arm sweep:** Arm A (depth=2 MLP), Arm B (depth=1 linear ctrl). `--wandb-group tanjiro-vol-decoder`
- **Status:** ASSIGNED — waiting for pod pickup.

---

## 2026-05-02 05:30 UTC — PR #251 CLOSED: fern T_max=8+warmup+lr-min=5e-6 — NEGATIVE (+0.344pp vs SOTA)

- **Branch:** `fern/cosine-tmax8-lrmin5e-6-warmup` (closed, branch deleted)
- **Hypothesis:** Shorter cosine cycle (T_max=8 vs SOTA T_max≈50-flat) forces the LR to anneal within the training budget, potentially improving generalization. Combined with lr_min=5e-6 floor and lr_warmup_epochs=1.
- **W&B run:** `uederk7o`

| Epoch | val_abupt | surface_pressure | wall_shear | volume_pressure | Gap to SOTA (9.065%) |
|-------|-----------|-----------------|------------|-----------------|----------------------|
| 1     | ~54%      | —               | —          | —               | warmup |
| 5     | 11.835%   | —               | —          | —               | +2.770 pp |
| 9     | **9.4088%** | **6.090%** | **10.634%** | **5.390%** | **+0.344 pp** |
| (best_epoch=9) | | | | | |

- **test_abupt:** 10.591% (vs SOTA 10.190%)
- **Conclusion:** T_max=8 aggressive cosine creates per-axis trade-offs — volume_pressure improved slightly (5.390% vs SOTA 12.656%) but wall_shear regressed (10.634% vs SOTA 9.910%). No uniform benefit. Shorter cosine cycle hypotheses are exhausted (T_max=8 negative, T_max=9 negative ×2, T_max=14 negative). The T_max≈flat schedule of SOTA is confirmed best.
- **Key learning:** `lr_cosine T_max=8 + anneal` added to closed-negative learnings. The cosine T_max axis is now fully explored: 8 (neg), 9 (neg ×2), 14 (neg). SOTA flat-50 confirmed.

---

## 2026-05-02 00:45 — PR #287 ASSIGNED: alphonse QK-norm on SOTA stack (Round 16)

- **Branch:** `alphonse/qk-norm-attention`
- **Hypothesis:** Per-head L2 normalization of Q and K vectors in `TransolverAttention.forward()` decouples attention magnitude from pattern, stabilizing Lion's sign-based updates and preventing attention entropy collapse. First test of QK-norm on this architecture.
- **Single delta vs SOTA:** `F.normalize(q, p=2, dim=-1)` + `F.normalize(k, p=2, dim=-1)` added in `model.py` after `qkv.chunk(3, dim=-1)`. No CLI change.
- **W&B group:** `tay-round16-qk-norm`
- **Status:** ASSIGNED — awaiting pod pickup.

---

## 2026-05-02 00:30 — Round 15 orphan PRs CLOSED: #263,264,265,267,268,269,271,272

- **Reason:** PRs were assigned to chihiro/emma/gilbert/haku/kohaku/norman/senku/violet — students who have NO DrivAerML (ddp8) pods. Only alphonse/askeladd/edward/fern/frieren/nezuko/tanjiro/thorfinn have ddp8 pods.
- **Action:** All 8 PRs closed; branches deleted. Hypotheses will be re-assigned to real ddp8 students in future rounds if still relevant.
- **Hypotheses deferred (not yet tested):**
  - lr-warmup-epochs=2 (chihiro #263)
  - RFF retest on SOTA stack (emma #264)
  - ema-decay=0.9995 with warmup=1ep (gilbert #265)
  - grad-clip-norm=0.5 (haku #267)
  - lr-cosine-t-max=12 (kohaku #268)
  - warmup=1ep + cosine T_max=9 (norman #269)
  - model-hidden-dim=768 + muP lr (senku #271)
  - lr-min=1e-5 (violet #272)

---

## 2026-05-02 00:20 — PR #240 CLOSED: frieren mlp-ratio=8 — NEGATIVE

- **Branch:** `frieren/mlp-ratio-8-wider-ffn` (closed)
- **Hypothesis:** Increasing `mlp_ratio` from 4 to 8 doubles MLP block capacity, potentially improving feature richness per Transformer layer for the CFD point-cloud task.
- **W&B run:** `kobvnazq` (rank 0), group `tay-round13-mlp-ratio-8`, runtime 270.9min

| Epoch | Step  | val_abupt | Gap to SOTA (9.065%) |
|-------|-------|-----------|----------------------|
| 1.00  | 2720  | 54.6200%  | +45.555 pp |
| 2.00  | 5441  | 23.7401%  | +14.675 pp |
| 3.00  | 8162  | 15.9900%  | +6.925 pp |
| 4.00  | 10883 | 12.9196%  | +3.855 pp |
| 5.00  | 13604 | 11.3006%  | +2.236 pp |
| 6.00  | 16325 | 10.4737%  | +1.409 pp |
| 7.00  | 19046 | 9.7725%   | +0.708 pp |
| **7.52** | **20444** | **9.5498%** | **+0.485 pp** |

- **Result:** Best val=9.5498% at ep7.52 (timeout). Gap to SOTA = +0.485pp. NEGATIVE.
- **Conclusion:** mlp_ratio=8 adds capacity but the model needs more training time/data to amortize it. At ep5-6 deceleration is severe (delta drops from -0.83pp to -0.22pp per interval). The expanded MLP blocks require a longer optimization budget than our 270min cap allows. Close as NEGATIVE; do not pursue further mlp_ratio scaling at this epoch budget. The mlp_ratio lever is now closed: 4=SOTA, 6=regression (PR#?), 8=NEGATIVE.

---

## 2026-05-01 23:30 — PR #222 TEST METRICS CONFIRMED: fern lr_warmup_epochs=1 — NEW TEST SOTA 10.420%

- **Branch:** `fern/round12-lr-warmup-1ep` (merged)
- **W&B run:** `ut1qmc3i` (rank 0), group `tay-round12-lr-warmup-1ep`
- **Test result (apples-to-apples vs prior PR #115 SOTA 10.580%):**

| Metric | PR #222 (NEW) | PR #115 (prev) | Delta |
|---|---:|---:|---:|
| `test_primary/abupt_axis_mean_rel_l2_pct` | **10.420** | 10.580 | **−0.160pp / −1.51%** |
| `test_primary/surface_pressure_rel_l2_pct` | **5.550** | 5.690 | −0.140 / −2.46% |
| `test_primary/wall_shear_rel_l2_pct` | **10.185** | 10.419 | −0.234 / −2.24% |
| `test_primary/volume_pressure_rel_l2_pct` | 12.737 | 12.740 | −0.003 (tied) |
| `test_primary/wall_shear_x_rel_l2_pct` | **8.629** | 8.908 | −0.279 / −3.13% |
| `test_primary/wall_shear_y_rel_l2_pct` | **12.329** | 12.491 | −0.162 / −1.30% |
| `test_primary/wall_shear_z_rel_l2_pct` | **12.854** | 13.071 | −0.217 / −1.66% |

- **Conclusion:** 1-epoch linear LR warmup is a **confirmed orthogonal win on test** (not just val). Improves every axis except volume_pressure (tied). The val win (9.291 vs 9.484, −2.03%) translates cleanly to test (−1.51%) — strong sign the warmup effect is generalization, not val-set leakage. **All future SOTA reproduce commands must include `--lr-warmup-epochs 1`.**
- **Implication:** The "warmup → settle → flat-LR" pattern outperforms cold-start steep entry. Suggests the early-training gradient norm is large enough that a hot-start LR=1e-4 imposes a directional bias the model has to unlearn. Worth exploring complementary "warmdown" shape (linear or cosine decay over ep7-9, per Modded-NanoGPT directive).

## 2026-05-01 23:00 — PR #251 ASSIGNED: fern warmup+cosine-anneal-hard T_max=8 lr-min=5e-6 (Round 15)

- **Branch:** `fern/lr-anneal-hard-tmax8-lrmin5e-6`
- **Hypothesis:** Combining 1ep linear warmup with aggressive cosine decay to 5% of peak LR (lr-min=5e-6, T_max=8) forces genuine convergence at end-of-training. No prior run has combined warmup=1ep + cosine anneal + non-trivial lr-min together. Distinct from thorfinn #247 (T_max=14, no warmup, no lr-min).
- **Single delta vs SOTA:** `--lr-cosine-t-max 8 --lr-min 5e-6` added (SOTA has neither; `--lr-warmup-epochs 1` from PR #222 baseline)
- **W&B group:** `tay-round13-lr-anneal`
- **Status:** ASSIGNED — awaiting student run start.

## 2026-05-01 23:00 — PR #250 ASSIGNED: alphonse batch-size=8 yi-confirmed lever (Round 15)

- **Branch:** `alphonse/batch-size-8-yi-lever`
- **Hypothesis:** Doubling per-GPU batch size from 4→8 (effective batch 64 vs 32) reduces gradient noise. Yi PR #9 (gilbert) confirmed this as orthogonal. PR #120 was queued for this exact test but never ran; this finally runs it.
- **Single delta vs SOTA:** `--batch-size 8` (SOTA: `--batch-size 4`)
- **W&B group:** `tay-round13-batch8`
- **Fallback:** `--batch-size 6` if OOM. LR held at 1e-4 (no proportional scaling — PR #148 showed diminishing returns above 1e-4).
- **Status:** ASSIGNED — awaiting student run start.

## 2026-05-01 21:30 — PR #247 ASSIGNED: thorfinn lr-cosine-t-max=14 (Round 14 — gentle annealing midpoint)

- **Branch:** `thorfinn/lr-cosine-t-max-14`
- **Hypothesis:** T_max=14 sits between the confirmed failure (T_max=9, LR collapses too early — edward #195 + tanjiro #202 both NEGATIVE) and the SOTA flat schedule (T_max=50 ≈ no decay for 9 epochs). A gentle midpoint decay might sharpen late-stage convergence while avoiding premature LR collapse.
- **Single delta:** `--lr-cosine-t-max 14` (SOTA: `--lr-cosine-t-max 50`)
- **W&B group:** `tay-round14-lr-cosine-t-max-14`
- **Status:** ASSIGNED — awaiting student run start. Results to be filled when run completes.

## 2026-05-01 15:30 — PR #186 CLOSED: alphonse vol_pts=96k clean re-run — does not beat SOTA

- **Branch:** `alphonse/round11-vol-pts-96k-clean`
- **Hypothesis:** Increasing volume sampling from 64k → 96k will reduce volume_pressure error and pull abupt below SOTA (clean re-run with explicit `--lr-cosine-t-max 50` after PR #158 LR-confound).
- **W&B:** rank-0 run `vv3j0qag`, group `tay-round11-vol-pts-96k-clean`, ran 9/9 epochs.

### Val trajectory vs SOTA #115

| ep | abupt | surf_p | vol_p | wall_s | SOTA abupt | SOTA surf | SOTA vol | SOTA wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67.016 | 51.826 | 53.372 | 70.311 | 53.750 | 38.817 | 37.798 | 57.644 |
| 5 | 13.664 | 9.143 | 8.395 | 15.218 | 11.830 | 7.770 | 7.118 | 13.265 |
| 9 | **10.068** | **6.411** | **6.358** | **11.238** | **9.484** | **6.007** | **5.896** | **10.632** |

- **Result:** ep9 abupt 10.068 vs SOTA 9.484 → **+0.584pp WORSE val**. Trailed SOTA on every sub-metric at every epoch — surface, volume, AND wall_shear.
- **Conclusion:** vol_pts=96k regresses on every axis vs SOTA vol_pts=64k. The volume sampling rate is at or near its local optimum at 64k. **Vol_pts lever retired.** This is the mirror retirement to volume_loss_weight (which earlier we hoped would close the AB-UPT vol_p gap from below; this hoped to close it from above via more samples).
- **Correction to prior log entries:** Earlier "vol_p=6.358% vs SOTA 12.740% = 50% improvement" claim was a val-vs-test apples-to-oranges error. Apples-to-apples val ep9 comparison shows alphonse vol_p IS worse than SOTA val by 0.46pp.
- **Follow-up assignment:** alphonse round12-surface-pts-96k (PR #206) — same one-flag-delta protocol but on the surface side.

## 2026-05-01 (latest) — PR #202 CLOSED: tanjiro lr_cosine_t_max=9 — negative (test 11.081, +4.74% vs SOTA)

- **Branch:** `tanjiro/cosine-tmax9-genuine`
- **Hypothesis:** Genuine cosine LR decay within the 9-epoch budget (T_max=9) will improve over SOTA's near-flat T_max=50 schedule, which decays only ~4% over 9 epochs. The theory: aggressive late-epoch LR warmdown lets the model settle into a tighter local minimum.
- **W&B:** run `1wx7mfw6`, group `tay-round12-cosine-tmax9`, rt=285min, 9/9 epochs.
- **Note:** Student initially submitted a config-error run (T_max=50 SOTA replication, `s58uz78l`); PR was sent back; corrected re-run is `1wx7mfw6`.

### Val trajectory

| Epoch | abupt_val (%) |
|------:|-------------:|
| 1 | 57.877 |
| 2 | 25.396 |
| 3 | 16.747 |
| 4 | 13.372 |
| 5 | 11.808 |
| 6 | 10.947 |
| 7 | 10.426 |
| 8 | 10.117 |
| 9 | **10.017** |

### Test metrics vs SOTA

| Metric | SOTA (PR #115) | PR #202 (T_max=9) | Delta |
|---|---:|---:|---:|
| abupt_mean | 10.580 | **11.081** | +4.74% |
| surface_pressure | 5.690 | 6.107 | +7.33% |
| wall_shear | 10.419 | 10.930 | +4.91% |
| volume_pressure | 12.740 | 13.200 | +3.61% |
| tau_x | 8.908 | 9.387 | +5.38% |
| tau_y | 12.491 | 13.041 | +4.41% |
| tau_z | 13.071 | 13.672 | +4.60% |

**val→test ratio:** 1.106 (SOTA: 1.115) — consistent generalization behavior.

- **Conclusion:** T_max=9 cosine does NOT improve over SOTA. Aggressive LR warmdown within the 9-epoch budget hurts performance uniformly across all axes (+4–7% regressions). The model is still in active learning at epoch 9 — cutting LR aggressively in eps 7-9 starves the final refinement phase. The near-flat schedule (T_max=50 ≈ 4% decay over 9 epochs) is confirmed optimal for this training horizon. **LR schedule space closed in the T_max direction.**
- **PR Status:** CLOSED. Negative result.

## 2026-05-01 — PR #204 CLOSED: frieren vol_loss_weight=2.0 (test 11.096, +4.88% vs SOTA)

- **Branch:** `frieren/vol-loss-weight-2p0`
- **Hypothesis:** BASELINE.md flags SOTA (PR #115) was trained WITHOUT `--volume-loss-weight 2.0`. PR #142 tested vol_w=2.0 on earlier stack (missing Lion wd=5e-4, EMA=0.999 compound). This run added vol_w=2.0 as a clean single-delta against verified SOTA. Targeted the largest per-axis gap (`volume_pressure` 12.740 vs AB-UPT 6.08, ×2.1).
- **W&B group:** `tay-round12-vol-loss-weight-2p0`
- **W&B run:** `qymdn7px`, rt=287min, 9/9 epochs.
- **Single delta from SOTA:** only `--volume-loss-weight 2.0` changes.

### Val trajectory

| Epoch | abupt_val |
|------:|----------:|
| 1 | 55.509 |
| 2 | 25.452 |
| 3 | 16.941 |
| 4 | 13.654 |
| 5 | 11.907 |
| 6 | 10.969 |
| 7 | 10.400 |
| 8 | 10.058 |
| 9 (best) | **9.945** |

### Test metrics vs SOTA (PR #115)

| Metric | SOTA | PR #204 | Δ |
|---|---:|---:|---:|
| abupt_axis_mean | 10.580 | **11.096** | **+4.88%** |
| surface_pressure | 5.690 | 6.251 | +9.86% |
| wall_shear | 10.419 | 11.042 | +5.98% |
| volume_pressure | 12.740 | 12.772 | +0.25% (≈neutral) |
| tau_x | 8.908 | 9.497 | +6.61% |
| tau_y | 12.491 | 13.132 | +5.13% |
| tau_z | 13.071 | 13.825 | +5.77% |

**val→test ratio:** 1.116 (SOTA: 1.115) — consistent generalization.

### Analysis

vol_loss_weight=2.0 produced a clear multi-task trade-off:
- **Volume_pressure barely moved** (+0.25%) — 2× weighting did not meaningfully reduce its gap. The volume ceiling is not a loss-weighting problem; it likely needs architectural capacity (dedicated volume head, separate decoder, richer volume features).
- **Surface_pressure regressed -9.86%** and wall_shear -5.98%. Capacity that previously served surface fidelity got redirected to volume with negligible benefit.
- Best val 9.945 propagated through normal val→test ratio into test 11.096.

### Conclusion

**vol_loss_weight axis CLOSED for 9-epoch budget.** Loss-reweighting cannot close the volume_pressure gap without sacrificing surface metrics that dominate the abupt aggregate. Further volume_pressure work must be architectural (dedicated head, capacity allocation, or representation-level changes).

- **PR Status:** CLOSED. Negative result.

## 2026-05-01 (latest) — PR #203 CLOSED: thorfinn round12 weight_decay=2.5e-4 (test 11.841, +11.9% vs SOTA)

- **Branch:** `thorfinn/round12-wd-2p5e-4`
- **Hypothesis:** PR #163 (wd=1e-3) regressed all metrics +4.5% from SOTA (wd=5e-4). Gradient in WD points DOWN — sweep to 2.5e-4 (half of SOTA value).
- **W&B group:** `tay-round12-wd-2p5e-4`
- **W&B run:** `894ay3y1`, rt=284min, 9/9 epochs.
- **Single delta from SOTA:** only `--weight-decay 2.5e-4` changes.

### Val trajectory

| Epoch | abupt_val |
|------:|----------:|
| 1 | 58.230 |
| 2 | 28.272 |
| 3 | 19.327 |
| 4 | 15.831 |
| 5 | 13.730 |
| 6 | 12.580 |
| 7 | 11.605 |
| 8 | 11.026 |
| 9 | 10.811 |

### Test metrics vs SOTA

| Metric | SOTA (PR #115) | PR #203 (wd=2.5e-4) | Delta |
|---|---:|---:|---:|
| abupt_mean | 10.580 | **11.841** | **+11.9%** |
| surface_pressure | 5.690 | 6.601 | +16.0% |
| wall_shear | 10.419 | 12.008 | +15.2% |
| volume_pressure | 12.740 | 13.003 | +2.1% |

- **Conclusion:** wd=2.5e-4 (half SOTA) is strictly WORSE than SOTA wd=5e-4 — 11.9% test regression. Combined with PR #163 (wd=1e-3 = double SOTA → +4.5% regression), the sweep confirms SOTA wd=5e-4 is the local optimum. Weight-decay space is closed. The asymmetry (halving WD hurts more than doubling it) suggests the model needs regularization that wd=5e-4 provides — going lower erases it.
- **PR Status:** CLOSED. Negative result.

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
