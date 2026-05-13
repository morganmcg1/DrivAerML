# DrivAerML Hypothesis Cards — 2026-05-13 19:30 UTC
# Wave: drivaerml-long-20260504 | DDP8 DL24 dialect | 24h budget
# SOTA: PR #972 test_vol_p=3.643%, test_abupt=5.844%
# Data root: /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511
# Steps/epoch (bs=1, DDP8, EBS=8): ~10,975
# All cards: --no-compile-model --train-volume-points 65000 --lr-warmup-epochs 1

---

## H1 — GradNorm + SDF α=0.5 Composition (PRIORITY 1)

**One-line claim:** Stacking dynamic per-channel loss weighting (GradNorm) on top of the best SDF arm (α=0.5) should give additive generalization gains on vol_p.

**Mechanism:** GradNorm α=0.5 (PR #1072, val_vol_p=4.2884%, currently WIP) shows the SDF near-surface sampling significantly improves vol_p. GradNorm v4 (wandb `ysycg6xc`) showed test_vol_p=3.6328% (beats SOTA by 0.010pp) but ran without SDF. Both mechanisms are orthogonal: SDF changes the *sampling distribution*, GradNorm changes the *loss weighting*. Combining them should let the model both see more near-surface volume points AND have the optimizer spend gradient budget where the hardest axes are. Must use Lion (AdamW + GradNorm = catastrophic instability per prior experiments).

**Concrete DDP8 CLI:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/gradnorm_sdf05_composition \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-lr 1e-3 \
  --no-compile-model \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.0 \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group gradnorm_sdf_composition \
  --wandb-name gradnorm_sdf_alpha05_ep35 \
  --agent <student-name>
```
**Note:** The SDF monkey-patch must be applied in `data/loader.py` equivalent location using `__class__` reassignment with α=0.5. The student must confirm which file implements SDF sampling from the nezuko PR.

**Smoke test plan:** EP3 (step ~33K). EMA val_abupt should be below 8.0% (current non-SDF baseline was around 7.5% at EP3). If above 9.0% at EP3, something is misconfigured — kill and debug.

**Long-run plan:** Full 35 epochs. Check val_vol_p at EP15 (nezuko best arm reached 4.2884% at EP14.7 without GradNorm). Target: val_vol_p < 4.0% at EP15 as a leading signal.

**Success criterion:** test_vol_p < 3.5% (beats PR #972 by 0.143pp). test_abupt < 5.8%.

**Risk / known failure modes:**
- GradNorm + Lion requires careful gradnorm-lr tuning. 1e-3 was stable in v4; do not change without reason.
- SDF `__class__` monkey-patch must survive the dataloader fork for DDP8. Verify in smoke test that sampling is non-uniform (log SDF bucket counts at EP1).
- GradNorm can suppress vol loss in early training if surface dominates — watch per-channel weights in W&B.

**Suggested student:** Assign to whichever student finishes SDF sweep first (likely nezuko after EP15-20 result).

---

## H2 — EMA Decay=0.999 Re-test on Corrected Split (PRIORITY 2)

**One-line claim:** EMA decay=0.999 was previously on the dead-ends list but all pre-20260511 results are void due to the dataset artifact — this is a clean re-test of a known-good regularization parameter.

**Mechanism:** EMA with decay=0.999 gives ~1,000-step lookback. The current SOTA (PR #972) already uses 0.999. The question is whether the dead-end calls (PR #954) were caused by the split bug masking the real benefit, or whether the mechanism is genuinely neutral. The retracted-from-dead-ends list in CURRENT_RESEARCH_STATE.md explicitly calls this out. This is a pure CLI change, zero code risk.

**Concrete DDP8 CLI:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/ema_decay_retest \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --no-use-y-symmetry-aug \
  --no-use-gradnorm \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group ema_corrected_split_retest \
  --wandb-name ema_0999_corrected_ep35 \
  --agent <student-name>
```

**Smoke test plan:** EP3 (~33K steps). EMA val_abupt < 8.5%.

**Long-run plan:** 35 epochs. Compare directly against PR #972 run on corrected split. This is a clean control-arm to isolate whether PR #972 SOTA can be reproduced cleanly.

**Success criterion:** Reproduces PR #972 test_vol_p ≤ 3.643% on corrected split. Any improvement counts.

**Risk / known failure modes:** This is a near-baseline run — low risk, primarily confirms split-corrected reproducibility. If it fails to reproduce PR #972, there is a configuration inconsistency that needs investigation before further ablations.

**Suggested student:** Ideal for any newly-idle student; zero code change.

---

## H3 — Weight Decay WD=0.01 Re-test on Corrected Split (PRIORITY 3)

**One-line claim:** WD=0.01 (PR #900) was promising on the buggy split; the corrected split may reveal stronger regularization benefit given that test geometries are OOD relative to training.

**Mechanism:** OOD generalization to test (50 held-out geometries) is the primary bottleneck. Higher weight decay penalizes large activations, discouraging geometry-specific overfitting. The dataset artifact (case-split bug) inflated val_vol_p artificially, masking whether WD=0.01 was overfitting to the (leaky) val set or genuinely regularizing. On the corrected split, WD=0.01 may produce a cleaner val→test transfer.

**Concrete DDP8 CLI:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/wd_0p01_corrected \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --no-use-y-symmetry-aug \
  --no-use-gradnorm \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group wd_corrected_split_sweep \
  --wandb-name wd_0p01_corrected_ep35 \
  --agent <student-name>
```

**Smoke test plan:** EP3. val_abupt < 8.5%. If above 9.5%, WD=0.01 is too aggressive for Lion and should be killed.

**Long-run plan:** 35 epochs. Compare val_vol_p trajectory against PR #972 baseline. If val curve is flatter but better at EP35, the OOD hypothesis holds.

**Success criterion:** test_vol_p < 3.5%. val→test gap narrows relative to SOTA.

**Risk / known failure modes:** WD=0.01 + Lion can cause under-training on surface channels (WSS tends to suffer more than vol). Watch per-channel metrics. Also try WD=0.005 as follow-up if this shows promise.

**Suggested student:** Pair with H7 (WD=0.005) on same student in sequential runs.

---

## H4 — Y-Symmetry Augmentation p=0.5 on Corrected Split (PRIORITY 4)

**One-line claim:** Built-in Y-symmetry random flip at p=0.5 is a different mechanism from the dead-end p=1.0 (which halved effective data diversity); p=0.5 adds geometric diversity on OOD test geometries.

**Mechanism:** The DrivAerML vehicles have approximate left-right (Y-axis) symmetry. p=1.0 was a dead end: always flipping collapses the augmentation to a fixed transform. p=0.5 *randomly* flips each sample, doubling geometric coverage for the model without reducing diversity. For OOD generalization (test geometries are held out), seeing both orientations during training may improve the model's ability to handle slight asymmetries in test vehicles. This is the pure CLI `--use-y-symmetry-aug --y-symmetry-aug-prob 0.5` path (PR #979 tested p=1.0 on the buggy split; this is a distinct experiment).

**Concrete DDP8 CLI:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/ysym_p05_corrected \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --no-use-gradnorm \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group ysym_corrected_split \
  --wandb-name ysym_p05_corrected_ep35 \
  --agent <student-name>
```

**Smoke test plan:** EP3. val_abupt < 8.5%. Also check WSS axes specifically — if Y-flip causes sign confusion in wss_y, the model may show wss_y regression while vol_p improves.

**Long-run plan:** 35 epochs. Primary target: test_vol_p. Secondary: check wss_y specifically for sign artifacts from flipping.

**Success criterion:** test_vol_p < 3.5%. wss_y does not regress (test_wss ≤ SOTA 6.727%).

**Risk / known failure modes:** Y-flip changes the sign of wss_y (the lateral shear channel). The data loader must flip wss_y sign when flipping the point cloud, or the model will see inconsistent supervision. Confirm this is handled in the augmentation implementation before running.

**Suggested student:** Any idle student; pure CLI.

---

## H5 — LR=9e-5 Isolated Control on Corrected Split (PRIORITY 5)

**One-line claim:** A lower learning rate (9e-5 vs 3e-4 SOTA) may produce a smoother loss landscape that generalizes better to OOD test geometries, particularly for vol_p.

**Mechanism:** The current SOTA uses lr=3e-4 with Lion. The training curve for vol_p typically continues improving past EP20 while WSS plateaus — a lower LR may extend the useful learning window without overshooting. Prior H3 in RESEARCH_IDEAS_2026-05-05 proposed lr=9e-5 but it was never cleanly tested on the corrected split. This is the minimum-change isolated LR test.

**Concrete DDP8 CLI:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/lr_9e5_corrected \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 9e-5 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 3e-6 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --no-use-y-symmetry-aug \
  --no-use-gradnorm \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=9.0" \
  --wandb-group lr_sweep_corrected \
  --wandb-name lr_9e5_corrected_ep35 \
  --agent <student-name>
```
**Note:** Kill threshold is slightly more lenient (9.0 vs 8.5) at EP5 to account for slower warmup convergence at lower LR. Adjust lr-min proportionally (3e-6 = 9e-5 / 30, matching the 3e-4 / 1e-5 ratio).

**Smoke test plan:** EP5 (~55K steps). EMA val_abupt < 9.0%. Slower convergence is expected; do not kill early.

**Long-run plan:** 35 epochs. If val_vol_p at EP25 is ahead of SOTA trajectory, extend to 45 epochs.

**Success criterion:** test_vol_p < 3.5%.

**Risk / known failure modes:** Slow convergence may mean EP35 is not the right stopping point. Use best-checkpoint logic (already built-in via EMA). Also: very low LR may cause WSS channels to underfit.

**Suggested student:** CLI-only; assign to any idle student.

---

## H6 — Beta-NLL Heteroscedastic Surface Head (PRIORITY 6)

**One-line claim:** Replacing the fixed-weight surface loss with a per-point predictive-variance (Beta-NLL) head turns the loss into an automatic uncertainty-weighted objective that downweights easy points and upweights hard OOD geometry regions.

**Mechanism:** Beta-NLL (Seitzer et al. 2022) augments a prediction head with a learned log-variance σ². The NLL loss is -log N(y; μ, σ²) which simplifies to (y-μ)² / (2σ²) + ½ log σ². The model learns to be uncertain on hard points (high σ² → small loss gradient) and certain on easy points. For surface WSS and pressure, near-edges and geometry-discontinuity regions are systematically harder and OOD-prone. A heteroscedastic head concentrates gradient on the informative residuals. The Beta-NLL variant from Seitzer et al. uses a β parameter to interpolate between homoscedastic (β=0) and pure NLL (β=1) to avoid variance collapse. Start with β=0.5.

**Code change required:** Add a 2x output head to the surface branch: `[4 → 8]` outputs where the second 4 are log-variances. Modify surface loss in `trainer_runtime.py`.

**Concrete DDP8 CLI (after code change):**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/beta_nll_surface \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --no-use-y-symmetry-aug \
  --no-use-gradnorm \
  --beta-nll-surface --beta-nll-beta 0.5 \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group beta_nll_surface \
  --wandb-name beta_nll_beta05_ep35 \
  --agent <student-name>
```
**Note:** The `--beta-nll-surface --beta-nll-beta` flags must be added to `train.py` by the student. Fallback: implement as a modified loss in `trainer_runtime.py` without new CLI flags.

**Smoke test plan:** EP3. Check that log-variance predictions are not collapsing (σ² should vary spatially). Log mean and std of predicted σ² in W&B.

**Long-run plan:** 35 epochs. Primary metric: test_vol_p. Secondary: test_surf_p (surface head changes directly affect this).

**Success criterion:** test_vol_p < 3.5%, test_surf_p < 3.4% (improvement on both).

**Risk / known failure modes:** Variance collapse is the main risk — σ² grows to infinity, zeroing all gradients. Add a soft floor: `log_var = log_var.clamp(-10, 10)`. Also: the vol head is unchanged, so vol_p improvement is indirect (via better surface representation shared in the encoder).

**Suggested student:** Assign to a student comfortable with loss code.

---

## H7 — Weight Decay WD=0.005 Re-test on Corrected Split (PRIORITY 7)

**One-line claim:** WD=0.005 (PR #914, retracted dead end) may show a sweet spot between WD=0.001 (SOTA) and WD=0.01 (H3) for OOD regularization.

**Mechanism:** Same as H3 but with weaker regularization. If H3 (WD=0.01) overfits the training distribution or undershoots surface quality, WD=0.005 is the natural midpoint to try. Run this as H3's follow-up or in parallel if a second student is idle.

**Concrete DDP8 CLI:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/wd_0p005_corrected \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.005 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --no-use-y-symmetry-aug \
  --no-use-gradnorm \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group wd_corrected_split_sweep \
  --wandb-name wd_0p005_corrected_ep35 \
  --agent <student-name>
```

**Smoke test plan:** EP3. val_abupt < 8.5%.

**Long-run plan:** 35 epochs. Compare directly against H3 and SOTA baseline.

**Success criterion:** test_vol_p < 3.5%.

**Risk / known failure modes:** Similar to H3 but lower risk. If both H3 and H7 fail, WD is not the bottleneck for this split.

**Suggested student:** Run as H3 follow-up or pair with H3 on two different students.

---

## H8 — Adaptive SDF α Schedule: Anneal from α=3.0 → α=0.5 (PRIORITY 8)

**One-line claim:** Start with aggressive near-surface focus (α=3.0) in early training to establish geometry-aware representations, then relax to α=0.5 for broader volume coverage in later epochs.

**Mechanism:** The SDF sweep shows a U-shaped response: α=0.5 (nezuko) leads on val_vol_p while α=3.0 (tanjiro) is currently behind but may have better early geometry learning. An adaptive schedule exploits the best of both regimes: aggressive focus early (build sharp near-surface features) then broaden sampling to cover the full volume for final generalization. Implementation: in the dataloader's SDF weight function, linearly interpolate α from 3.0 at epoch 1 to 0.5 at epoch 15, then hold at 0.5 for epochs 15-35.

**Code change required:** Modify the SDF weight function (using `__class__` reassignment pattern) to accept an epoch counter and interpolate α. The trainer must pass the current epoch to the dataloader.

**Implementation sketch:**
```python
# In trainer_runtime.py or equivalent epoch loop:
alpha_start, alpha_end, anneal_epochs = 3.0, 0.5, 15
current_alpha = alpha_start + (alpha_end - alpha_start) * min(epoch / anneal_epochs, 1.0)
dataset.update_sdf_alpha(current_alpha)  # dataset method to update alpha
```

**Concrete DDP8 CLI (after code change):**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/sdf_adaptive_anneal \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --no-use-y-symmetry-aug \
  --no-use-gradnorm \
  --sdf-alpha-start 3.0 --sdf-alpha-end 0.5 --sdf-anneal-epochs 15 \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group sdf_adaptive_anneal \
  --wandb-name sdf_anneal_3p0_to_0p5_ep35 \
  --agent <student-name>
```
**Note:** New CLI flags `--sdf-alpha-start --sdf-alpha-end --sdf-anneal-epochs` must be added by the student. Log current α to W&B at each epoch.

**Smoke test plan:** EP3. Verify α is decreasing in W&B logs. val_abupt < 8.5%.

**Long-run plan:** 35 epochs. Primary signal: compare val_vol_p at EP15 against nezuko (α=0.5 static) — if the adaptive schedule is better, the early-focus hypothesis is confirmed.

**Success criterion:** test_vol_p < 3.5%. val_vol_p at EP15 < 4.0%.

**Risk / known failure modes:** If α decreases too quickly, early geometry learning is incomplete. If too slowly, late-training coverage is insufficient. The 15-epoch anneal window is the main hyperparameter — log α trajectory in W&B.

**Suggested student:** Assign after SDF sweep completes and α=3.0 arm (tanjiro) results are available.

---

## H9 — Volume Coordinate Noise Re-test on Corrected Split (PRIORITY 9)

**One-line claim:** Adding small Gaussian noise to volume point coordinates at training time acts as a volume-space regularizer; the earlier dead-end verdict may have been split-artifact-driven.

**Mechanism:** Volume coordinate noise (PR #990, retracted dead end) adds ε ~ N(0, σ²) to volume x,y,z during training. This prevents the model from memorizing exact grid positions and forces generalization to slightly displaced query points. On the corrected split where test geometries are genuinely OOD, this spatial jitter may improve vol_p by discouraging coordinate-exact overfitting. The effect is analogous to input dropout in DNNs or coordinate noise in NeRF training.

**Implementation:** Small Gaussian noise on volume_x during training only (not eval). σ = 0.01 in normalized coordinate space.

**Concrete DDP8 CLI (after code change or via existing noise flag if present):**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/vol_coord_noise_retest \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --no-use-y-symmetry-aug \
  --no-use-gradnorm \
  --vol-coord-noise-std 0.01 \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group vol_coord_noise_corrected \
  --wandb-name vol_coord_noise_std001_ep35 \
  --agent <student-name>
```
**Note:** `--vol-coord-noise-std` flag must be added to `train.py` if not present. Apply noise in `trainer_runtime.py` only when `model.training` is True.

**Smoke test plan:** EP3. val_abupt < 8.5%. If noise is applied at eval (a common bug), val metrics will degrade — verify in EP1 logs.

**Long-run plan:** 35 epochs.

**Success criterion:** test_vol_p < 3.5%.

**Risk / known failure modes:** Noise applied to eval coordinates is a common bug — must be guarded by `if training`. Also: σ=0.01 is a guess; too large destroys the near-surface structure entirely.

**Suggested student:** Medium-priority; assign after H1-H5.

---

## H10 — GradNorm + Y-Symmetry Composition (PRIORITY 10)

**One-line claim:** Combine dynamic loss weighting (GradNorm) with Y-symmetry augmentation p=0.5 after both are individually validated on the corrected split.

**Mechanism:** If H4 (Y-sym p=0.5) and GradNorm individually improve test_vol_p on the corrected split, stacking them is a natural next step. This is a composition hypothesis — do not run until H4 individual result is available.

**Concrete DDP8 CLI:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/gradnorm_ysym_composition \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-lr 1e-3 \
  --no-compile-model \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group gradnorm_ysym_composition \
  --wandb-name gradnorm_ysym_p05_ep35 \
  --agent <student-name>
```

**Smoke test plan:** Same as H4 — check wss_y sign convention with augmentation + GradNorm.

**Long-run plan:** 35 epochs. Gated on H4 individual result.

**Success criterion:** test_vol_p < 3.4% (improvement over either individual technique).

**Risk / known failure modes:** If H4 (Y-sym) shows wss_y regression due to sign flip, do not run this composition until the sign issue is resolved.

**Suggested student:** Gate on H4 result. Assign after H4 completes.

---

## H11 — Bbox Normalization Re-test on Corrected Split (PRIORITY 11)

**One-line claim:** Per-vehicle bounding-box coordinate normalization (PR #978, retracted dead end) removes scale variation across geometries and may improve OOD generalization on the corrected split.

**Mechanism:** DrivAerML vehicles vary in absolute size. The STRING PE uses absolute coordinates — if test vehicles are slightly larger/smaller than training vehicles, the PE is out-of-distribution. Bbox normalization maps each vehicle into a canonical unit cube before feeding to the model, making the PE scale-invariant. On the buggy split this may have hurt because the leaky val set had similar geometries to training; on the corrected split with genuinely OOD test geometries, normalization may help.

**Concrete DDP8 CLI (requires `--use-bbox-normalization` flag in train.py):**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/bbox_norm_corrected \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --no-use-y-symmetry-aug \
  --no-use-gradnorm \
  --use-bbox-normalization \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group bbox_norm_corrected \
  --wandb-name bbox_norm_corrected_ep35 \
  --agent <student-name>
```

**Smoke test plan:** EP3. val_abupt < 8.5%. Check that coordinate ranges after normalization are within [0,1] or [-0.5, 0.5].

**Long-run plan:** 35 epochs.

**Success criterion:** test_vol_p < 3.5%.

**Risk / known failure modes:** Bbox normalization changes the meaning of the STRING PE — the model may need more warmup epochs to re-learn the new coordinate scale. Try `--lr-warmup-epochs 2` if EP3 smoke fails.

**Suggested student:** Lower priority; assign after H1-H9.

---

## H12 — GradNorm + SDF α=0.5 + Y-Symmetry Triple Stack (PRIORITY 12)

**One-line claim:** The maximal composition: SDF near-surface sampling, dynamic loss weighting, and geometric augmentation all active simultaneously. Run only if H1 and H4 both individually improve on SOTA.

**Mechanism:** This is the "kitchen sink" composition of H1 and H4. It tests whether the three improvements are additive or if they interfere. Gate strictly on individual results.

**Concrete DDP8 CLI:**
```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/gradnorm_sdf05_ysym_triple \
  --epochs 35 \
  --batch-size 1 \
  --train-surface-points 11000 --eval-surface-points 11000 \
  --train-volume-points 65000 --eval-volume-points 65000 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 8 \
  --model-mlp-ratio 2 --model-slices 32 --model-dropout 0.0 \
  --model-pe string_multisigma \
  --pe-num-features 128 \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --optimizer lion \
  --lion-beta1 0.9 --lion-beta2 0.99 \
  --lr 3e-4 \
  --weight-decay 0.001 \
  --lr-warmup-epochs 1 \
  --lr-cosine-t-max 35 \
  --lr-min 1e-5 \
  --grad-clip-norm 1.0 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-lr 1e-3 \
  --no-compile-model \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --kill-thresholds "54875:val/abupt_axis_mean_rel_l2_pct<=8.5" \
  --wandb-group triple_stack_composition \
  --wandb-name gradnorm_sdf05_ysym_ep35 \
  --agent <student-name>
```
**Note:** SDF α=0.5 monkey-patch must also be applied. This is a composition of H1 + H4.

**Smoke test plan:** EP3. val_abupt < 8.0% (stricter target since all three mechanisms should compound).

**Long-run plan:** 35 epochs. Gate on H1 and H4 results.

**Success criterion:** test_vol_p < 3.3%. test_abupt < 5.5%.

**Risk / known failure modes:** Three simultaneous changes make attribution difficult. Only run if both H1 and H4 are individually confirmed winners. Do not run as a discovery experiment.

**Suggested student:** Run as final synthesis after H1 and H4 complete.

---

## Priority Matrix

| Rank | Hypothesis | Code Change | Blocks On | Expected Signal |
|------|-----------|-------------|-----------|-----------------|
| 1 | H1: GradNorm + SDF α=0.5 | Yes (SDF patch) | nezuko result | High (two validated mechanisms) |
| 2 | H2: EMA 0.999 re-test | None | — | Medium (reprodicibility control) |
| 3 | H3: WD=0.01 corrected | None | — | Medium (OOD regularization) |
| 4 | H4: Y-sym p=0.5 corrected | None | — | Medium (geometric diversity) |
| 5 | H5: LR=9e-5 corrected | None | — | Medium (slower convergence) |
| 6 | H6: Beta-NLL surface head | Yes (loss) | — | High potential, medium confidence |
| 7 | H7: WD=0.005 corrected | None | H3 | Low-medium (midpoint sweep) |
| 8 | H8: Adaptive SDF anneal | Yes (schedule) | tanjiro result | Medium (curriculum learning) |
| 9 | H9: Vol coord noise retest | Yes (augment) | — | Low-medium (retracted dead end) |
| 10 | H10: GradNorm + Y-sym | None | H4 result | Medium (composition) |
| 11 | H11: Bbox norm corrected | Yes (flag) | — | Low-medium (retracted dead end) |
| 12 | H12: Triple stack | None | H1 + H4 | High if precursors win |

---

## Config Constraints Checklist (all cards must satisfy)

1. `--no-compile-model` REQUIRED (DDP8 + compile = NCCL deadlock)
2. `--train-volume-points 65000` minimum
3. `--lr-warmup-epochs 1` (epoch-based)
4. GradNorm requires `--optimizer lion`
5. `--model-pe string_multisigma --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0`
6. Kill threshold format: `STEP:metric<=value` (e.g. `54875:val/abupt_axis_mean_rel_l2_pct<=8.5`)
7. EMA decay=0.999 (NOT 0.9999)
8. SDF patch uses `__class__` reassignment
9. Data root: `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`
10. DDP8 launcher: `torchrun --nproc_per_node=8`
11. No `--gradnorm-mode` flag (does not exist in DL24)
