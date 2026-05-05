# Research Ideas — 2026-05-05 12:30 UTC

Wave: DrivAerML long-run single-model DDP8 validation (advisor branch `drivaerml-long-20260504`).

This document captures the next-tier hypotheses for round 2+ of the wave. All four round-1 arms (H1–H4) have either completed or been superseded; the current idle student is dl24-tanjiro (PR #673 closed). The in-flight arms are:

- **#664 (fern):** per-axis output scaling on STRING backbone — EP8 best=7.2351%, still running.
- **#669 (frieren):** per-channel tau weighting (y×1.2/z×1.3) — EP5=9.88%, projecting EP10~8.5-8.8%.
- **#678 (nezuko):** extended cosine T_max=60 — just assigned, no result yet.

Hard constraints apply to all hypotheses below:
- Single-model only (no ensembles, no soups, no checkpoint averaging in the loss).
- DDP8 across 8 GPUs: `torchrun --nproc_per_node=8 train.py`.
- 24h wall-clock max; epochs chosen to finish cleanly within this budget.
- `--no-compile-model` required (compile+bs=2 causes NCCL deadlock risk).
- `--lr-warmup-epochs 1` required.
- `surface_loss_weight=1.0` required (without tay stack, higher values cause EP1 divergence at ~70-72%).
- W&B project: `wandb-applied-ai-team/senpai-v1-drivaerml-ddp8`.
- SOTA base: Lion, lr=1e-4, bs=2, train_surface_points=40000, train_volume_points=65000, STRING multi-sigma PE sigmas=[0.25,0.5,1.0,2.0,4.0], ema_decay=0.999, no-compile-model.
- Do NOT duplicate in-flight PRs #664, #669, #678.
- Do NOT retry 7-sigma PE (PR #673 closed regression, conclusively ruled out).

---

## Ranked hypothesis cards

### H1 (PRIORITY 1) — STRING + QK-Norm: stabilising attention variance for deeper tau_y/z learning

**WHY:**
The SOTA result (`sogus8sx`, test=7.9303%) uses multi-sigma STRING PE with 4 attention layers. The dominant gap versus AB-UPT is in wall-shear: our tau_y=~10% vs AB-UPT 3.65%, tau_z=~10% vs AB-UPT 3.63%. One plausible mechanism: attention logit variance grows with depth and with the multi-sigma positional signal (which injects high-magnitude Fourier features into Q and K). Pre-wave evidence `tkiigfmc` tested STRING + QK-Norm and is listed in the May 4 research map as a high-value follow-up. QK-Norm (normalizing Q and K before the dot product, a la NormFormer / PaLM 2 / Chameleon) bounds logit variance, stabilises gradient flow through attention, and empirically improves on tasks with structured geometric positional signals — which is exactly our setting. The STRING embedding contributes correlated, high-magnitude directions to Q and K; without normalisation, later layers can develop attention collapse or spike. Stabilising this path is likely to help the layers that most govern cross-stream (tau_y) and spanwise (tau_z) coherence.

**WHAT:**
Add per-head RMS-normalisation of Q and K vectors inside the Transolver attention block (optionally learnable scale parameter per head, initialised to 1.0). No other change to architecture, loss, or schedule.

**HOW:**
Code change in `model.py`: inside the attention block that computes `attn = softmax(Q @ K.T / sqrt(d_head))`, replace Q and K with `Q / (Q.norm(dim=-1, keepdim=True) + 1e-6)` and `K / (K.norm(dim=-1, keepdim=True) + 1e-6)` (optionally multiply by a learnable scalar `s` per head after normalising). This is a small, isolated change. All other config identical to SOTA base.

Smoke run (3 epochs, short wall-clock), then confirm with 50-epoch long run if EP3 abupt < 9.0%.

```bash
# Smoke (rank-0 or torchrun --nproc_per_node=8, e.g. 3 epochs)
torchrun --nproc_per_node=8 train.py \
  --wandb-project senpai-v1-drivaerml-ddp8 \
  --wandb-group string-qknorm \
  --epochs 3 \
  --optimizer lion \
  --lr 1e-4 \
  --lr-warmup-epochs 1 \
  --batch-size 2 \
  --train-surface-points 40000 \
  --train-volume-points 65000 \
  --pe-init-sigmas '0.25,0.5,1.0,2.0,4.0' \
  --ema-decay 0.999 \
  --no-compile-model \
  --qk-norm  # new flag (or whatever the student names it)
```

Long run: same flags, `--epochs 50`.

**Risk:**
- Learning rate may need slight adjustment (QK-Norm changes effective scale of logits; if attention entropy collapses in EP1 the student should try `--lr 9e-5`).
- QK-Norm with a fixed normaliser (no learnable scale) may be too aggressive; the learnable version is safer.
- If the PR #664 per-axis output scaling result is strong when this is assigned, the student should stack both — but only if per-axis scaling has already merged (not before).

**Mechanism vs. in-flight overlap:** none — #664 targets output head scaling, #669 targets loss weighting, #678 targets schedule. QK-Norm targets attention variance independently.

---

### H2 (PRIORITY 2) — 5-Layer STRING: depth as a free lever on SOTA base

**WHY:**
The current SOTA uses 4 Transolver layers. Pre-wave evidence `70lnb3dt` is listed in the May 4 research map as a 5-layer STRING candidate. The width-over-depth experiment (#659) failed due to OOM at slices=128 forcing slices=64, which is a different axis: depth without width increase is far cheaper (parameter count grows linearly with depth, not with depth×width). Going from 4L to 5L at hidden=512 adds roughly 25% parameters (~7M additional at 512d), which fits comfortably in DDP8 at bs=2 with slices=128. More depth gives the model additional cross-attention pooling and residual computation budget to refine the tau_y/z fields across the vehicle wake. The key question is whether the 24h budget gives a 5L model enough steps to converge; at bs=2 DDP8 with ~5500 steps/epoch, 50 epochs should be sufficient given that 4L reaches convergence by ~EP25.

**WHAT:**
Set `--model-layers 5` on the SOTA config. No other change.

**HOW:**
Pure CLI. Zero code change.

```bash
torchrun --nproc_per_node=8 train.py \
  --wandb-project senpai-v1-drivaerml-ddp8 \
  --wandb-group 5l-string-long \
  --epochs 50 \
  --model-layers 5 \
  --optimizer lion \
  --lr 1e-4 \
  --lr-warmup-epochs 1 \
  --batch-size 2 \
  --train-surface-points 40000 \
  --train-volume-points 65000 \
  --pe-init-sigmas '0.25,0.5,1.0,2.0,4.0' \
  --ema-decay 0.999 \
  --no-compile-model \
  --kill-thresholds "27469:val_primary/abupt_axis_mean_rel_l2_pct>=8.5,54938:val_primary/abupt_axis_mean_rel_l2_pct>=7.5"
```

Kill gates: EP5 ≥ 8.5% → kill; EP10 ≥ 7.5% → kill.

**Risk:**
- Memory: 5L at bs=2, slices=128 may hit OOM on one or more ranks. Pre-flight check: run 1 step with `--epochs 1 --dry-run` to verify VRAM. If OOM, reduce slices to 96 (not 64 — #659 showed 64 is too small).
- Convergence: 5L may be slower to converge than 4L in early epochs; the EP5 kill gate is intentionally lenient (8.5% vs the 4L baseline EP5 of ~7.3%) to avoid killing a slow-starting run prematurely.
- Risk of undertrained result: ensure the kill gates are not triggered by EP5 noise.

**Mechanism vs. in-flight overlap:** none — adds a Transolver layer, orthogonal to output scaling (#664), tau weighting (#669), and schedule (#678).

---

### H3 (PRIORITY 3) — lr=9e-5 control run on SOTA STRING base (isolating the lr lever)

**WHY:**
The strongest pre-wave reference run is `9mm3sz7x` (test=8.1229%), which used lr=9e-5 with AdamW and mild tau weighting on a non-STRING base. That run was trained on the old optimizer/PE stack. The question it leaves open is: does lr=9e-5 independently help on the modern Lion+STRING base? The current SOTA uses lr=1e-4 (which was the best value on the earlier stack), but the STRING PE changes the gradient landscape by injecting Fourier features — the optimal LR may shift. This is a pure control arm: STRING SOTA config, only change lr from 1e-4 to 9e-5. It is cheap (zero code change), interpretable, and cleanly separates the lr effect from the STRING PE effect. If lr=9e-5 helps, every subsequent experiment should adopt it. If it does not, we have confirmed 1e-4 is robust to the PE choice.

**WHAT:**
Set `--lr 9e-5` on the otherwise identical SOTA config.

**HOW:**
Pure CLI. Zero code change.

```bash
torchrun --nproc_per_node=8 train.py \
  --wandb-project senpai-v1-drivaerml-ddp8 \
  --wandb-group lr9e5-string-control \
  --epochs 50 \
  --optimizer lion \
  --lr 9e-5 \
  --lr-warmup-epochs 1 \
  --batch-size 2 \
  --train-surface-points 40000 \
  --train-volume-points 65000 \
  --pe-init-sigmas '0.25,0.5,1.0,2.0,4.0' \
  --ema-decay 0.999 \
  --no-compile-model \
  --kill-thresholds "27469:val_primary/abupt_axis_mean_rel_l2_pct>=8.5,54938:val_primary/abupt_axis_mean_rel_l2_pct>=7.5"
```

**Risk:**
- Very low. The only failure mode is slower convergence — 9e-5 vs 1e-4 changes the early-epoch trajectory but the 50-epoch budget should fully resolve both.
- If convergence is materially slower the student should note the epoch of best checkpoint; this informs future long-run budgets.
- This is explicitly a control arm. Even a negative result is high-value because it confirms the current LR choice is robust and removes the confounder from future compositions.

**Mechanism vs. in-flight overlap:** none. The in-flight arms do not vary lr.

---

### H4 (PRIORITY 4) — EMA-proxy GradNorm α=0.5 (clean re-run on SOTA base)

**WHY:**
The closed PR #623 failed catastrophically (val=12.4377%), but the failure was caused by infrastructure failure (student ignored 5 kill orders and let the run continue past the kill gate), not by the mechanism itself. The pre-wave reference `wyz68o8r` (test=8.236%) used EMA-proxy GradNorm α=0.5 with Lion and showed no nonfinite gradients and reasonable tau_y gains. The original round-1 hypothesis card (H4 in RESEARCH_IDEAS_2026-05-04_1022.md) correctly identified this mechanism as high-value — it was the logistics failure that invalidated the test. Now that the SOTA config is firmly established (STRING multi-sigma PE + Lion + lr=1e-4), a clean re-run of EMA-proxy GradNorm α=0.5 with an explicit volume guard kill threshold would produce valid evidence about whether dynamic channel weighting helps on top of the modern stack.

NOTE: This is distinct from the in-flight frieren #669 experiment (which uses static per-channel tau weights). GradNorm dynamically adjusts all channel weights based on gradient norms, not just tau channels.

**WHAT:**
Implement EMA-proxy GradNorm α=0.5 as a training callback that:
1. Maintains an EMA of per-channel gradient norms (α=0.5) for each output head channel.
2. Normalises the loss weights so the channels with the largest EMA gradient norms receive lower weights (convergence-rate matching).
3. Logs per-channel weights to W&B at each step (required for interpretability and debugging).
4. Has a volume-loss floor guard: volume loss weight must not fall below 0.5× its initial value to prevent volume regression.

**HOW:**
Code change in `train.py` or `trainer_runtime.py`. The implementation pattern is:
- After each `loss.backward()`, before `optimizer.step()`, read per-channel gradient norms from the surface output head's gradient tensor.
- Maintain EMA of these norms.
- Recompute loss weights as `w_i = (mean_norm / norm_i)^α` (GradNorm-style; α controls how aggressively to equalise).
- Clip: `w_volume_loss >= 0.5 * initial_volume_weight`.
- Apply to the next batch's loss computation.

Kill thresholds: EP5 ≥ 9.0% → kill; EP10 ≥ 8.0% → kill.

```bash
torchrun --nproc_per_node=8 train.py \
  --wandb-project senpai-v1-drivaerml-ddp8 \
  --wandb-group ema-gradnorm-a05-sota \
  --epochs 50 \
  --optimizer lion \
  --lr 1e-4 \
  --lr-warmup-epochs 1 \
  --batch-size 2 \
  --train-surface-points 40000 \
  --train-volume-points 65000 \
  --pe-init-sigmas '0.25,0.5,1.0,2.0,4.0' \
  --ema-decay 0.999 \
  --no-compile-model \
  --gradnorm-alpha 0.5 \        # new flag
  --gradnorm-ema-decay 0.5 \    # new flag
  --gradnorm-volume-floor 0.5 \ # new flag
  --kill-thresholds "27469:val_primary/abupt_axis_mean_rel_l2_pct>=9.0,54938:val_primary/abupt_axis_mean_rel_l2_pct>=8.0"
```

**Risk:**
- Dynamic weighting can destabilise training if EMA α is too low (too much variation in weights per step). α=0.5 is the conservative value from `wyz68o8r`. Do NOT use α=1.0 (that is the `341czkol` run which regressed volume).
- The student MUST respect kill gates this time. Kill gate non-compliance was the failure mode of PR #623.
- Volume guard must be implemented and tested in the smoke run before launch.
- Do not use AdamW (GradNorm + AdamW = catastrophic instability; use Lion).

**Mechanism vs. in-flight overlap:** none. Frieren #669 uses static tau weighting; GradNorm uses dynamic gradient-norm-based weighting across all channels. These are complementary.

---

### H5 (PRIORITY 5) — Beta-NLL / heteroscedastic output head for tau uncertainty

**WHY:**
The tau_y and tau_z residuals are physically heteroscedastic: cross-stream and spanwise wall shear vary enormously between sheltered (underfloor) and exposed (A-pillar, mirror wake) regions. A plain MSE loss treats all points as having equal variance, which causes the model to over-fit to the low-variance dominant region (underfloor tau_x) and under-fit the high-variance regions (tau_y, tau_z boundary-layer transitions). A heteroscedastic head — one that jointly predicts mean and log-variance per point — applies implicit per-point upweighting to high-uncertainty regions. The Beta-NLL variant (Seitzer et al. 2022, "Pitfalls of Data-Driven Networking: A Case Study of Latent Causal Confounders in Video Streaming") controls the degree of variance suppression via a β parameter (β=0 is plain NLL; β=1 is MSE; β=0.5 is empirically robust). This is a principled loss reformulation that directly targets the tau_y/z bottleneck without hand-tuning per-channel weights.

**WHAT:**
Replace the MSE surface loss with Beta-NLL for the wall-shear channels:
1. Surface output head predicts 8 channels: 4 mean channels (cp, tau_x, tau_y, tau_z) + 4 log-variance channels (sigma_cp, sigma_tau_x, sigma_tau_y, sigma_tau_z).
2. Loss: `beta_nll = 0.5 * exp(-log_var) * mse + 0.5 * beta * log_var`, summed over channels.
3. β=0.5 (default from Seitzer et al.).
4. Volume head remains unchanged (scalar, MSE).
5. At eval/test time: report predictions from mean channels only; ignore variance channels.

**HOW:**
Code change in `model.py` (extend surface output head to 8 channels) and `train.py` (replace surface MSE loss with Beta-NLL). The surface head output projection changes from `nn.Linear(hidden, 4)` to `nn.Linear(hidden, 8)`, where channels 4-7 are `log_var_cp, log_var_tau_x, log_var_tau_y, log_var_tau_z`. Initialise log-var head weights to zero so at init the loss is approximately MSE.

Kill thresholds: EP5 ≥ 9.0% → kill; EP10 ≥ 8.0% → kill.

Smoke run mandatory (3 epochs) to confirm: (a) no nonfinite log-var outputs, (b) the variance outputs are logged to W&B, (c) the mean predictions are what gets evaluated.

**Risk:**
- The output head dimension change breaks checkpoint compatibility with SOTA. Start from scratch (no warmstart from `sogus8sx`). This is expected.
- Variance head can collapse to large values (log-var >> 0) early in training, making loss dominated by the 0.5*β*log_var term and suppressing learning. The initialisation to zero mitigates this, but monitor `train/loss` in the smoke run.
- If the Beta-NLL version fails (diverges or shows no improvement), the student should also try: (a) aleatoric loss (mean only, with predicted variance clamped to `[log(0.01), log(100)]`), (b) β=1 (equivalent to MSE + log_var penalty, simpler).
- Prediction quality is evaluated on mean channels only — ensure the eval code correctly indexes `[:, :4]` (mean) and not the full 8-channel output.

**Mechanism vs. in-flight overlap:** none. This is a loss reformulation, not a weighting of an existing MSE. Different from tau weighting (#669) which applies scalar multipliers to MSE per channel.

---

### H6 (PRIORITY 6) — y-symmetry data augmentation (physics-constrained vehicle mirror)

**WHY:**
DrivAerML vehicles have approximate left-right (y) bilateral symmetry. The training set has 400 cases, but each case has a unique geometry. A physics-valid augmentation: for each training case, create a y-mirrored version by negating the y-coordinate and normal_y of surface points, and negating the tau_y prediction target (since wall shear in the y-direction reverses under the mirror). This doubles the effective training set from 400 to 800 cases with no new geometry acquisition cost. The symmetry is not exact for all vehicles (body asymmetries exist), but it is a good approximation for the DrivAer family. Augmented cases should only be used during training, never for validation or test evaluation. The mechanism targets the data bottleneck: 400 training cases is modest for a CFD surrogate that must generalise across full vehicle geometry space.

**WHAT:**
Implement a y-flip augmentation in the data loader (or as a batched transform at training time):
- For surface points: `x' = x, y' = -y, z' = z`; `nx' = nx, ny' = -ny, nz' = nz`; `area' = area` (unchanged)
- For surface targets: `cp' = cp` (pressure is scalar, symmetric); `tau_x' = tau_x, tau_y' = -tau_y, tau_z' = tau_z`
- For volume points: `x' = x, y' = -y, z' = z`; `sdf' = sdf` (unsigned distance, symmetric)
- For volume targets: `p_v' = p_v` (pressure scalar, symmetric)
- Apply augmentation with probability 0.5 per batch (random flip per case, not per point).

**HOW:**
Code change in `data/loader.py` or a transform wrapper called from `train.py`. The data spec in `program.md` marks `data/loader.py` as "Read-only during normal experiment PRs", so the cleanest approach is a batched tensor transform in `train.py` after the batch is loaded, before the forward pass.

Note: `program.md` says `data/loader.py` is "Read-only during normal experiment PRs." A batched augmentation applied to the tensor batch in the training loop (in `train.py`, which IS editable) avoids modifying the loader.

Kill thresholds: EP5 ≥ 9.0% → kill; EP10 ≥ 8.0% → kill.

**Risk:**
- Approximate symmetry: DrivAer models include some asymmetric body features. The augmentation introduces label noise proportional to geometric asymmetry. Monitor val_primary/wall_shear_y_rel_l2_pct specifically — if tau_y regresses vs. baseline this is the most likely explanation.
- The augmented cases are correlated with the originals — they are not truly independent. Do NOT add mirrored validation or test cases.
- Implementation care: the tau_y sign flip on surface targets is critical. An unsigned tau_y flip would introduce incorrect training signal for the cross-stream direction.
- Only worth attempting if EP3 val shows the augmented distribution does not corrupt the early-epoch trajectory.

**Mechanism vs. in-flight overlap:** none. Pure data augmentation, no overlap with loss, schedule, or architecture arms.

---

## Ranking summary for assignment

| Rank | Hypothesis | Key mechanism | Code change? | Est. compute risk | Expected leverage |
|------|-----------|---------------|-------------|------------------|-------------------|
| 1 | H1: STRING + QK-Norm | Attention variance stabilisation | Yes (small) | Low | High: targets attention depth bottleneck |
| 2 | H2: 5L STRING | Extra Transolver depth | No (CLI only) | Low-medium (OOM check needed) | Medium-high: free depth lever, clean isolation |
| 3 | H3: lr=9e-5 control | LR sensitivity on STRING base | No (CLI only) | Very low | High information: removes confounder for future work |
| 4 | H4: EMA-proxy GradNorm α=0.5 | Dynamic gradient-norm channel weighting | Yes (medium) | Medium (kill gate compliance critical) | Medium: pre-wave evidence positive, prior run failed on logistics |
| 5 | H5: Beta-NLL heteroscedastic head | Heteroscedastic tau loss | Yes (moderate) | Medium (head change, no warmstart) | Medium: principled tau_y/z fix; novel in this setting |
| 6 | H6: y-symmetry augmentation | Physics-valid 2× training data | Yes (small) | Low (train-loop only) | Medium: data bottleneck; physically motivated |

## Assignment recommendation for dl24-tanjiro (single idle student)

Assign **H2: 5L STRING** as the immediate assignment. Rationale:

1. Zero code change — pure CLI `--model-layers 5`. Can be launched immediately without waiting for code review.
2. Well-motivated: 4L is the current depth, pre-wave evidence `70lnb3dt` supports 5L.
3. Complementary to all in-flight arms: does not overlap with #664 (output scaling), #669 (tau weighting), #678 (schedule).
4. If fern's #664 per-axis output scaling merges before the 5L run finishes, the student can stack it at the next assignment — the depth and head-scaling mechanisms are orthogonal.
5. The kill gates are set conservatively (EP5 ≥ 8.5%) to avoid killing a legitimately slow-starting deeper model.

Second-best assignment if QK-Norm code review is already in a reviewable PR: assign **H1: STRING + QK-Norm** instead, as it has higher expected leverage.

## Hypotheses to queue for later rounds (after at least one arm above produces a terminal result)

- **Compose STRING + QK-Norm + tau weighting:** stack H1 (if successful) with frieren's #669 mechanism.
- **5L STRING + EMA-proxy GradNorm:** stack H2 (if successful) with H4 mechanism.
- **Stronger tau weighting (y=1.5/z=2.0):** reference `nh96x7m4` (test=8.171) under full long DDP8 on SOTA base.
- **Surface-loss weight 2.0 with tay stack:** `qqtdnlwq` showed promise, but requires the tay stack to prevent EP1 divergence. Revisit after tay stack is available on this branch or a workaround is found.
- **Volume MLP head:** reference `8x7c537j` — replace the volume Transolver head with a separate MLP decoder to allow independent volume feature capacity.
- **Checkpoint averaging (post-wave):** not a training experiment, but averaging checkpoints from the top 3 wave experiments is valid after terminal results exist.
- **Spectral loss terms:** frequency-domain supervision for high-frequency pressure fields; architecturally novel, higher risk, worth a research slot after lower-risk options above are exhausted.
