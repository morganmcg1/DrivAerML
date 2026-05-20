# Research Ideas — DrivAerML WSS Wave, 2026-05-20 22:00Z

**Researcher:** Researcher-agent (researcher sub-agent)
**Branch:** drivaerml-long-20260504
**Context:** H19 achieved the wave's first simultaneous SOTA-beat on test_wss (6.6339%) and test_abupt (5.8197%), but both floor constraints were breached: test_vol_p=3.7786% (+0.136pp) and test_surf_p=3.6267% (+0.050pp). Four students are idle. All 8–12 experiments below are ordered by expected information value. All must satisfy the hard contract (Issue #1056):

- test_vol_p ≤ 3.643%
- test_surf_p ≤ 3.577%
- Single-model only
- Build on PR #972 training stack + corrected dataset `/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511`

**H19 baseline CLI** (reference for all diffs below):
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.05 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes z \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-3
```

**Mechanistic map as of this wave:**

| Exp | WSS Δ vs SOTA | vol_p vs floor | Key driver |
|-----|--------------|----------------|------------|
| H19 | −0.093 BEAT | +0.136 breach | Charb-vol_p under GradNorm → w_vol_p collapses to 0.05 |
| H20 | +0.081 miss | +0.204 breach | clamp alone costs +0.143pp wss; MAE_aux slightly reduces wss cost |
| H18 | +0.019 miss | +0.059 breach | truncated EP13; curvature + H9b stack |
| H10b | −0.062 BEAT | +0.517 breach | curvature + Charb_τz; first wss SOTA break |
| H11b | +0.292 miss | +0.213 breach | per-axis τ weights −0.245pp vs H8; additive lever validated |
| H9b | +0.054 miss | ≈0.000 (at floor) | clamp=0.15 + MAE_aux achieves floor; costs wss |

---

## H21 — H19 + clamp=0.15 (HIGHEST PRIORITY)

**Hypothesis:** H19's vol_p breach (+0.136pp) is a gradient-mass problem: GradNorm assigns w_vol_p=0.05 (floor), starving the vol_p head. Raising the clamp to 0.15 triples the guaranteed vol_p gradient mass, replicating H9b's floor-locking mechanism on top of H19's Charb-shaped landscape. The compound should close the floor gap without the +0.14pp wss cost observed in H20 (H10b base without Charb-vol_p), because Charb-vol_p gives vol_p a gentler curvature that allows convergence under lower w_vol_p — but the floor still needs a baseline guaranteed mass.

**Mechanism:** Replace `--gradnorm-min-w-vol-p 0.05` → `0.15`. Every other flag stays identical to H19. GradNorm will see larger vol_p gradient norm at floor (from Charb landscape shaping), and clamp prevents w_vol_p from collapsing below 0.15. Net effect: vol_p gradient mass increases by 3×, pulling test_vol_p toward the H9b-observed floor value (~3.64%). The wss cost is expected to be smaller than in H20 because the Charb-vol_p residuals are already small in H19's trained state — vol_p task difficulty under Charb is lower than under L2, so GradNorm naturally assigns lower w_vol_p. Increasing the clamp from 0.05 to 0.15 partially counteracts this but does not override the wss lead.

**Exact CLI changes from H19:**
```
# Change only this flag:
--gradnorm-min-w-vol-p 0.15   # was 0.05
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes z \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-3 \
  --wandb-group H21-clamp015
```

**Predicted outcome quadrant:**
- Optimistic: test_vol_p ≤ 3.58%, test_wss ≤ 6.70%, test_surf_p ≤ 3.60% → contract win
- Baseline expectation: test_vol_p ≈ 3.58–3.65%, test_wss ≈ 6.68–6.77%, test_surf_p ≈ 3.59–3.63% → contract win if wss stays ≤ SOTA 6.727%
- Pessimistic: clamp forces w_vol_p to floor but asymmetric budget shift kills wss lead → test_wss ≥ 6.73% miss

**Falsification criteria:**
- EP3: val_vol_p < 4.5% and val_wss < 7.4% → both mechanisms engaging; continue
- EP3 kill: val_vol_p > 5.0% OR val_wss > 7.8% → abort
- EP10: val_vol_p < 4.0% and val_wss trending down from EP3 → on track
- EP20: val_vol_p < 3.70% and val_wss < 6.9% → terminal projection feasible
- Terminal: test_vol_p ≤ 3.643% AND test_wss ≤ 6.727% AND test_surf_p ≤ 3.577% → contract win

**Recommended student:** dl24-frieren (primary assignment, wave's first idle post-H19 SOTA)

---

## H22 — H19 + clamp=0.15 + MAE_aux=0.05

**Hypothesis:** MAE_aux carries a unique vol_p benefit (~−0.20pp observed in H9b vs H20 comparison) that clamp alone cannot replicate. H9b's full stack (clamp+MAE_aux) was the only configuration to achieve the floor. Adding MAE_aux on top of H21's stack (H19+clamp) should further reduce vol_p while the H19 Charb landscape still provides the wss gain. The H20 finding (MAE_aux slightly DECREASES wss cost vs H10b base) suggests MAE_aux is complementary, not antagonistic, to the wss pathway.

**Mechanism:** Add auxiliary L1 loss on vol_p predictions in parallel to the Charbonnier reconstruction loss. This creates a stable second signal that prevents the vol_p head from drifting even when GradNorm assigns low w_vol_p. The L1 regularizer has a constant-subgradient property: unlike L2, it maintains a non-decaying training signal even when residuals are small, complementing Charb's sub-quadratic smoothness at the interior. Together, they create a compound loss landscape with both a smooth interior (Charb) and a constant outer gradient (L1).

**Exact CLI changes from H19:**
```
# Two changes:
--gradnorm-min-w-vol-p 0.15         # was 0.05 (H21 change)
--vol-p-aux-mae-weight 0.05         # new flag (H9b-style)
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes z \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-3 \
  --vol-p-aux-mae-weight 0.05 \
  --wandb-group H22-clamp015-mae
```

**Predicted outcome quadrant:**
- Optimistic: test_vol_p ≤ 3.55%, test_wss ≤ 6.69% → contract win with margin
- Baseline: test_vol_p ≈ 3.56–3.64%, test_wss ≈ 6.69–6.75% → contract win
- Pessimistic: MAE_aux + Charb on vol_p + clamp create over-constrained vol_p → surf_p breach from budget shift

**Falsification criteria:**
- EP3: val_vol_p < 4.4% (lower than H21's expected 4.5%) → MAE_aux adding benefit; continue
- EP10: val_vol_p < 3.90% → on track; flag if val_wss > 7.0%
- EP20: val_vol_p < 3.65% and val_wss < 6.85% → terminal good
- Stop if EP10 val_surf_p > 4.2% (surf_p budget compression from dual vol_p signals)

**Recommended student:** dl24-nezuko

---

## H23 — H19 + per-axis τ weights (clamp=0.05 unchanged)

**Hypothesis:** H11b validated that per-axis τ weights (τ_y=1.2, τ_z=1.5) provide an additive −0.245pp on test_wss vs baseline. This mechanism is orthogonal to the Charb-vol_p and curvature mechanisms in H19. Composing them should push test_wss further below SOTA while leaving vol_p and surf_p mechanisms unaffected (H11b showed GradNorm w_vol_p stayed healthy throughout with per-axis weights). The key question is whether the per-axis weighting survives the Charb-vol_p GradNorm interaction.

**Mechanism:** Add per-axis loss weight upweighting for τ_y and τ_z. These are the hardest WSS axes (largest absolute error). The upweighting gives GradNorm stronger gradient signal from the wss task relative to vol_p and surf_p, shifting the task-budget competition. On H11b (H8 base), GradNorm w_vol_p stayed healthy because no Charb was distorting the vol_p residual magnitude — the key risk here is whether H19's Charb-vol_p + per-axis WSS upweighting further compresses the vol_p budget. The student should watch w_vol_p closely at EP2/EP3.

**Exact CLI changes from H19:**
```
# One new flag (assuming --wss-axis-weights is supported):
--wss-axis-weights "1.0,1.2,1.5"   # τ_x, τ_y, τ_z weights
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.05 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes z \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-3 \
  --wss-axis-weights "1.0,1.2,1.5" \
  --wandb-group H23-peraxis-on-H19
```

**Predicted outcome quadrant:**
- Optimistic: test_wss ≤ 6.39% (H19 6.63 − H11b's 0.245pp), floors still breached (vol_p at +0.136pp similar to H19)
- Baseline: test_wss ≈ 6.40–6.50% BEAT, vol_p breach similar to H19 (needs floor recovery from separate experiment)
- Pessimistic: per-axis weights + Charb-vol_p create severe GradNorm starvation → vol_p breach worsens beyond H10b's 0.517pp

**Diagnostic value:** This run tests whether per-axis and Charb-vol_p mechanisms are additive on wss. If vol_p breach is ≈ H19's (+0.136pp), the mechanisms are orthogonal and H21+H23 compound is the logical next step.

**Falsification criteria:**
- EP3: val_wss < 7.2% (lower than H19's expected ~7.35%) → per-axis adding benefit
- EP3 kill: val_vol_p > 5.5% → per-axis catastrophic GradNorm interaction; abort
- EP10: val_wss < 6.9% → strong projection; flag w_vol_p trend
- Terminal: primary question is whether test_wss < 6.60%; floor breach is expected and secondary

**Recommended student:** dl24-tanjiro

---

## H24 — H21 + per-axis τ weights (compound floor+wss attack)

**Hypothesis:** If H21 (H19+clamp=0.15) successfully closes the vol_p floor while maintaining SOTA-beat wss, then H23's per-axis τ weight mechanism is the natural next lever to push wss further below 5.85%. H24 compounds both interventions: clamp=0.15 for floor preservation + per-axis τ weights for additional wss reduction. This experiment is a compound of H21 and H23's mechanisms, making it higher-risk but potentially the path that reaches the 5.85% target.

**Mechanism:** clamp=0.15 ensures vol_p gradient mass ≥ 3× floor; per-axis τ weights upweight τ_y and τ_z in the loss, exploiting the −0.245pp validated additive lever. The key question is whether clamp=0.15 can "protect" vol_p from the additional GradNorm pressure created by per-axis upweighting of wss axes.

**Exact CLI changes from H19:**
```
--gradnorm-min-w-vol-p 0.15         # H21 change (was 0.05)
--wss-axis-weights "1.0,1.2,1.5"   # H23 change (new)
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes z \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-3 \
  --wss-axis-weights "1.0,1.2,1.5" \
  --wandb-group H24-clamp015-peraxis
```

**Predicted outcome quadrant:**
- Optimistic: test_vol_p ≤ 3.60%, test_wss ≤ 6.40% → strong contract win and progress toward 5.85%
- Baseline: test_vol_p ≈ 3.58–3.65%, test_wss ≈ 6.40–6.60% → contract win if floors hold
- Pessimistic: per-axis upweights exceed clamp protection → vol_p breach persists at H19-level

**Risk vs H21:** This bundles two changes. If it fails floor, we don't know if it's per-axis overwhelming the clamp, or clamp insufficient on its own. Recommend running H21 simultaneously so the per-axis effect is isolated by comparison.

**Falsification criteria:**
- EP3: val_vol_p < 4.4% and val_wss < 7.1% → both mechanisms active; continue
- EP3 kill: val_vol_p > 5.0% → per-axis crushing GradNorm clamp; abort
- EP10: val_wss < 6.75% (lower than H19's EP10 benchmark) → per-axis additive on H21 stack
- Terminal: test_vol_p ≤ 3.643% AND test_wss < 6.60% → contract win + wss progress

**Recommended student:** dl24-fern

---

## H25 — H21 + Charbonnier on τ_x axis (multi-axis Charb expansion)

**Hypothesis:** H19 applies Charb on τ_z only (`--wss-charbonnier-axes z`). H11b showed τ_y has the largest absolute error (test_τ_y=7.362% vs τ_z=8.747% but larger relative gain from weighting). Adding Charb on τ_x (the lowest-error WSS axis at test_τ_x=5.971%) may unlock additional wss reduction by reshaping the loss landscape on the relatively converged axis. Alternative: apply to τ_y instead of τ_x. This experiment tests multi-axis Charb expansion on H21's floor-safe stack.

**Mechanism:** Charbonnier loss on τ_x reshapes the loss from L2 to subquadratic, reducing the penalty gradient from small residuals and shifting gradient mass toward the harder τ_y, τ_z axes. Under GradNorm, this changes the relative task difficulty of τ_x, potentially improving wss balance. The clamp=0.15 (from H21) prevents vol_p starvation from any budget rebalancing.

**Exact CLI changes from H19:**
```
--gradnorm-min-w-vol-p 0.15                # H21 change
--wss-charbonnier-axes xz                  # was z; adds τ_x
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes xz \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-3 \
  --wandb-group H25-charb-xz-clamp015
```

**Predicted outcome quadrant:**
- Optimistic: test_wss ≤ 6.60%, floors preserved → contract win with margin; τ_x Charb adds independent wss lever
- Baseline: test_wss ≈ 6.62–6.72%, floors near target → borderline contract result
- Pessimistic: τ_x Charb deactivates an already-converged axis → no wss benefit, neutral

**Falsification criteria:**
- EP3: val_wss lower than H21 equivalent → τ_x Charb adding benefit
- EP10: val_τ_x shows reduced error vs H19 EP10 → mechanism confirmed active
- EP20: val_wss < 6.80% → on track
- Terminal: if test_wss ≥ H21's value, the τ_x Charb axis is not load-bearing; retain H21 only

---

## H26 — H21 + Charbonnier ε = 1e-4 on vol_p (sharper sub-quadratic regime)

**Hypothesis:** H19 uses `--vol-p-charbonnier-eps 1e-3` (the standard ε). A smaller ε = 1e-4 sharpens the transition point between the quadratic and linear regimes of the Charbonnier function. For residuals < ε, Charb behaves like L2; for residuals > ε, it behaves like L1. With ε=1e-4, more of the vol_p residuals fall into the L1-like linear regime, providing stronger gradient signal where H19's current Charb gets weak. This should increase the effective vol_p training signal without changing the loss architecture.

**Mechanism:** Charb(r) = sqrt(r² + ε²) − ε. For r >> ε, gradient ≈ r/|r| (L1-like, constant magnitude). For r << ε, gradient ≈ r/ε (L2-like, proportional). Reducing ε from 1e-3 to 1e-4 moves the transition boundary down tenfold — more residuals are in the L1 regime — meaning the vol_p head receives a stronger average gradient signal even when residuals are small. Under GradNorm, this increases the apparent vol_p task norm, which may allow GradNorm to allocate naturally higher w_vol_p without needing the clamp.

**Exact CLI changes from H19:**
```
--gradnorm-min-w-vol-p 0.15              # H21 change (diagnostic: isolate ε effect with clamp safety)
--vol-p-charbonnier-eps 1e-4             # was 1e-3; 10× sharper transition
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes z \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-4 \
  --wandb-group H26-charb-eps1e4-clamp015
```

**Predicted outcome quadrant:**
- Optimistic: sharper Charb ε naturally increases GradNorm's w_vol_p → clamp less needed → wss stays at H19 level while vol_p converges more
- Baseline: vol_p improved vs H19, wss ≈ H21 level → contract win
- Pessimistic: very small ε causes numerical instability in Charb gradients near zero → loss instability at early epochs

**Falsification criteria:**
- EP1 kill: loss spike > 2× EP0 value → ε too small, Charb gradient blowup
- EP3: w_vol_p in W&B > 0.08 organically (without clamp hitting) → ε change working
- EP10: val_vol_p ≤ H21's EP10 value → sharper ε improving vol_p fit
- Terminal: test_vol_p ≤ 3.643% AND test_wss comparable to H21 → ε tuning is additive

---

## H27 — H21 + cosine T_max=25 (early tail convergence)

**Hypothesis:** The current cosine schedule has T_max=30 (full epochs), meaning LR barely reaches zero at the end. Setting T_max=25 creates a 5-epoch "post-cosine plateau" at near-zero LR, allowing the optimizer to fine-tune in a low-noise regime during the final epochs. For models that are still descending at EP25, this can extract additional gains. The CURRENT_RESEARCH_STATE.md notes this as a suggested follow-up; the base for this test should be H21 (not H19) to ensure the floor test runs on an already-compound-proven stack.

**Mechanism:** Cosine annealing with T_max < epochs creates a warmer-restart-free schedule that decays to minimum LR earlier. Epochs T_max+1 to max_epochs run at LR_min. For convergence-limited scenarios where the model has learned its structure but is still in fine-tuning mode, this extra low-LR plateau can reduce calibration error on held-out samples. The vol_p head may benefit most from fine-tuning at near-zero LR because its Charb loss has small residuals (near the flat interior) where the optimizer needs high-precision updates.

**Exact CLI changes from H19:**
```
--gradnorm-min-w-vol-p 0.15    # H21 change
--lr-cosine-t-max 25           # was 30; creates 5-epoch LR plateau at end
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 25 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes z \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-3 \
  --wandb-group H27-tmax25-clamp015
```

**Predicted outcome quadrant:**
- Optimistic: 0.01–0.03pp additional improvement on both wss and vol_p vs H21
- Baseline: test metrics ≈ H21 with slightly better vol_p convergence at EP28-30 → marginal contract win
- Pessimistic: T_max=25 decays LR too fast, creating premature convergence at EP22-25 plateau → worse than H21

**Falsification criteria:**
- EP10: val metrics should match or slightly lead H21 (same trajectory until EP25 where divergence appears)
- EP25: the two runs (H21 vs H27) should show divergence in val_vol_p if T_max mechanism is active
- EP26-30: H27 val metrics should be flat (confirming plateau) or still descending (LR not at floor yet)
- Terminal: if test results ≈ H21 within ±0.01pp → T_max mechanism neutral; not worth further tuning

---

## H28 — Charb on τ_y + τ_z (highest-error WSS axes, no τ_x) with clamp=0.15

**Hypothesis:** The wave's WSS axes by absolute error: τ_z=8.747% > τ_y=7.362% > τ_x=5.971%. H19 applies Charb only to τ_z. τ_y is the second-hardest axis with 7.362% error. Applying Charb to τ_y in addition to τ_z should give an independent wss-reduction lever by reshaping the τ_y loss landscape. The mechanism is the same as H10b/H19 for τ_z: sub-quadratic loss reduces the gradient norm on the interior, shifting GradNorm budget toward the harder-to-converge regions. Clamp=0.15 prevents vol_p starvation from the enlarged Charb-wss signal.

**Mechanism:** With `--wss-charbonnier-axes yz`, both τ_y and τ_z receive Charbonnier loss treatment. GradNorm sees two reshaped wss axes instead of one, amplifying the wss-side budget advantage that drove H19's SOTA beat. The Charb on τ_y may also reduce the large τ_y error (7.362%) which is currently the second-largest contribution to test_wss. The clamp=0.15 ensures vol_p doesn't lose its floor protection under the increased wss-axis Charb gradient mass.

**Exact CLI changes from H19:**
```
--gradnorm-min-w-vol-p 0.15              # H21 change
--wss-charbonnier-axes yz                # was z; adds τ_y
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.15 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes yz \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-3 \
  --wandb-group H28-charb-yz-clamp015
```

**Predicted outcome quadrant:**
- Optimistic: test_wss ≤ 6.55% (additional −0.08pp from τ_y Charb), floors preserved → strong contract win
- Baseline: test_wss ≈ 6.58–6.68%, floors at boundary → borderline or contract win
- Pessimistic: Charb on τ_y causes w_τ_y to also surge (like w_τ_z surged to 1.85 in H19) → dual axis starvation worsens vol_p despite clamp

**Watch in W&B:** Log w_τ_y weight trajectory alongside w_τ_z. If both surge past 1.5 while w_vol_p stays pinned at 0.15, this indicates the clamp is protecting vol_p but the dual-Charb mechanism is saturating budget on wss. If only w_τ_z surges (τ_y Charb doesn't create same imbalance), the mechanism is benign.

**Falsification criteria:**
- EP3: w_τ_y AND w_τ_z both > 1.3 → dual starvation risk; monitor closely
- EP5 kill: val_vol_p > 5.0% despite clamp=0.15 → mechanism overriding clamp protection
- EP10: val_wss < 6.8% (improvement over H21's expected EP10 level) → τ_y Charb adding value
- Terminal: test_wss < H21's value → τ_y Charb is additive; worth further stacking

---

## H29 — GradNorm replacement with IMTL-G (gradient surgery baseline)

**Hypothesis:** GradNorm uses loss magnitude ratios to set task weights — it is susceptible to distortions when tasks use different loss functions (L2 vs Charb vs L1-aux). The fundamental problem in this wave is that Charb artificially deflates the vol_p loss magnitude, causing GradNorm to underweight vol_p. IMTL-G (Liu et al. ICLR 2021) instead operates on gradient vectors directly: it projects each task gradient to eliminate inter-task conflict and scales them to equal projected magnitudes. This bypasses the loss-magnitude distortion entirely. IMTL-G is closed-form, requires no hyperparameters, and has shown competitive or superior performance to GradNorm on the MTAN benchmark.

**Mechanism:** IMTL-G computes the unit gradient direction for each task and projects conflicting gradients onto the normal plane of the shared gradient. The equal-magnitude scaling ensures no task is systematically underweighted regardless of loss function shape. The key advantage here: when Charb on vol_p makes the vol_p loss small but the gradient itself is meaningful (near the transition zone), IMTL-G sees the correct gradient information instead of a distorted loss ratio. This directly targets the mechanistic flaw in GradNorm's interaction with Charb losses.

**Exact CLI changes from H19:**
```
# Replace GradNorm with IMTL-G:
--use-gradnorm                      # REMOVE this flag
--gradnorm-alpha 0.5                # REMOVE this flag
--gradnorm-min-w-vol-p 0.05         # REMOVE this flag
--use-imtl-g                        # NEW FLAG (requires implementation)
```

**Note for student:** IMTL-G requires implementation in the training loop. The reference implementation (Liu et al. 2021, ICLR) computes the gradient surgery in the backward pass. If `--use-imtl-g` is not yet a supported flag, the student should:
1. Check if `--use-pcgrad` is supported (PCGrad is the simpler alternative)
2. If neither is available, implement IMTL-G as a wrapper over the optimizer step
3. Alternatively, run with `--use-pcgrad` as a feasibility test (PCGrad is easier to implement)

**Predicted outcome quadrant:**
- Optimistic: IMTL-G eliminates the Charb-distortion starvation problem entirely → vol_p naturally converges without any clamp → test_vol_p ≤ 3.64% AND test_wss ≤ 6.65%
- Baseline: IMTL-G provides similar vol_p protection to clamp=0.15 but at lower wss cost → marginal contract win
- Pessimistic: IMTL-G lacks GradNorm's stability in this multi-output mesh setting → training instability at EP2-5

**Falsification criteria:**
- EP1 kill: loss spike > 3× baseline → IMTL-G unstable on this architecture
- EP3: w_vol_p equivalent (if logged) ≥ 0.10 organically → IMTL-G correctly weighting vol_p
- EP10: val_vol_p < H21's equivalent → IMTL-G superior to clamp-based GradNorm
- Terminal: test_vol_p ≤ 3.643% → mechanism validation; floor achieved without clamp

**Implementation reference:** Liu, L. et al. "Towards Impartial Multi-Task Learning" ICLR 2021. The projection formula is: g̃_i = g_i − (g_i · u_s / |u_s|²) u_s where u_s is the shared/mean gradient unit vector. Implementation is ~15 lines of PyTorch.

**Recommended student:** Reserve for the next idle wave slot after H21-H28 results arrive.

---

## H30 — Charbonnier weight sweep on vol_p (0.05 → 0.10)

**Hypothesis:** H19 uses `--vol-p-charbonnier-weight 0.05`, a conservative weight for the Charb-vol_p term. The analogous τ_z Charb weight is `--wss-charbonnier-weight 0.1` (2×). The vol_p Charb contribution to the composite loss is weaker than τ_z's, which may explain why GradNorm's task signal for vol_p is more easily distorted. Doubling to 0.10 increases the relative gradient contribution of Charb-vol_p under GradNorm's composite signal, potentially allowing GradNorm to see a larger vol_p task norm and naturally allocate higher w_vol_p.

**Mechanism:** The Charb loss on vol_p enters GradNorm as part of the vol_p task's gradient magnitude. At `weight=0.05`, the Charb contribution to the total vol_p gradient is small relative to the main MSE term. At `weight=0.10`, Charb contributes twice as much to GradNorm's estimate of vol_p task difficulty, pushing w_vol_p higher organically. This avoids the "clamp hack" entirely — it lets GradNorm naturally see vol_p as harder and allocate appropriate budget. Clamp can be kept at 0.05 to isolate the weight change.

**Exact CLI changes from H19:**
```
--vol-p-charbonnier-weight 0.10     # was 0.05; 2× Charb weight on vol_p
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 --gradnorm-min-w-vol-p 0.05 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes z \
  --vol-p-charbonnier-weight 0.10 --vol-p-charbonnier-eps 1e-3 \
  --wandb-group H30-charb-volp-w010
```

**Predicted outcome quadrant:**
- Optimistic: GradNorm w_vol_p organically rises to ≥ 0.10 without clamp → test_vol_p ≤ 3.64% while test_wss stays near H19 (6.63-6.65%) → contract win
- Baseline: vol_p improves partially vs H19 but still breaches floor → needs clamp layer
- Pessimistic: higher Charb weight on vol_p amplifies the GradNorm starvation (counter-intuitive: more Charb → more apparent "easy" residuals → even lower w_vol_p allocation) → vol_p breach worsens

**Falsification criteria:**
- EP3: w_vol_p (W&B) > 0.07 organically → higher Charb weight working as intended
- EP10: val_vol_p < H19's EP10 value at same clamp floor → mechanism confirmed
- Terminal: if test_vol_p > H21, the weight increase makes things worse → stay on clamp approach

---

## H31 — GradNorm α tuning: α=1.0 (sharper gradient normalization)

**Hypothesis:** GradNorm's α parameter controls the asymmetry of the normalization: α=0 is uniform weighting; α=1.0 applies the strongest normalization toward equal gradient norms across tasks. The current α=0.5 is a middle ground. With the Charb-distorted vol_p task having a systematically lower apparent gradient norm, a sharper α=1.0 normalization would push more weight toward the "harder" task (vol_p in GradNorm's view based on inverse loss ratio). This could organically compensate for the Charb distortion without needing an explicit clamp.

**Mechanism:** GradNorm task weight update: ẇ_i ∝ (L_i / L̄_mean)^α. Higher α amplifies the ratio, making the weight assignment more responsive to relative loss changes. For the Charb-distorted vol_p task (low apparent loss from sub-quadratic shaping), α=1.0 doubles the penalty relative to α=0.5 when vol_p is "easier" than average — but the direction of the distortion matters. If α=1.0 correctly identifies vol_p as "hard" (because normalized loss ratio is high despite Charb), it will upweight vol_p more aggressively. Note: this may interact poorly with Charb if the normalized loss makes vol_p look "easy" regardless.

**Exact CLI changes from H19:**
```
--gradnorm-alpha 1.0     # was 0.5; sharper gradient norm target
```

**Full command:**
```
torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --epochs 30 --batch-size 1 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 1.0 --gradnorm-min-w-vol-p 0.05 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes z \
  --vol-p-charbonnier-weight 0.05 --vol-p-charbonnier-eps 1e-3 \
  --wandb-group H31-gradnorm-alpha10
```

**Predicted outcome quadrant:**
- Optimistic: α=1.0 compensates for Charb distortion → w_vol_p rises organically → vol_p floor achieved without clamp wss cost
- Baseline: partially reduces breach without full floor recovery; diagnostic value even if below floor
- Pessimistic: Charb distortion direction means α=1.0 AMPLIFIES the bias toward lower w_vol_p → breach worsens

**Falsification criteria:**
- EP3: w_vol_p trajectory in W&B — if HIGHER than H19 (H19 went to 0.05 floor) → α=1.0 working
- EP3 kill: val_vol_p > H19's EP3 value (indicating amplified starvation) → abort
- Terminal: if test_vol_p > H19's 3.7786% → α=1.0 is counterproductive; confirms clamp is the right lever

---

## Research State and Decision Tree

### Current best explanation for the bottleneck

The vol_p floor breach in H19 (+0.136pp) is mechanistically a GradNorm gradient-mass problem caused by Charb-vol_p deflating the apparent vol_p task difficulty. The clamp is the most direct fix. The wss floor (test_wss < 5.85%) is a harder challenge requiring either stacking multiple independent levers (per-axis, multi-axis Charb, curvature, etc.) or a fundamentally different gradient surgery approach.

### Experiment priority order for 4 idle students

1. **H21** (dl24-frieren): H19 + clamp=0.15 — highest priority, direct path to contract win
2. **H22** (dl24-nezuko): H19 + clamp=0.15 + MAE_aux=0.05 — compound floor protection
3. **H23** (dl24-tanjiro): H19 + per-axis τ weights — additive wss lever orthogonal to floor
4. **H24** (dl24-fern): H19 + clamp=0.15 + per-axis — compound floor+wss attack

### Decision tree

```
H21 results
├─ test_vol_p ≤ 3.643% AND test_wss ≤ 6.727% (CONTRACT WIN)
│  ├─ test_wss > 6.60% → continue: run H24 (per-axis on top), H28 (τ_y Charb)
│  └─ test_wss ≤ 6.60% → push toward 5.85%: run H24 + H28 + H29 (IMTL-G)
├─ test_vol_p ≤ 3.643% BUT test_wss > 6.727% (floor recovered, wss regressed)
│  └─ clamp is wss-cost → run H22 (MAE_aux complement) or H31 (α=1.0 organic fix)
├─ test_vol_p > 3.643% (floor still breached)
│  ├─ w_vol_p > 0.15 in W&B (clamp not binding) → representational bottleneck
│  │  └─ Run H29 (IMTL-G) or expand vol_p head capacity
│  └─ w_vol_p pinned at 0.15 (clamp binding) → clamp too low → try clamp=0.20
└─ training diverges → abort; check for EP8 spike (H18 pattern)

H23 results (per-axis on H19 base)
├─ test_wss < H19 (6.634%) → per-axis additive → design H24 with H21+per-axis
└─ test_wss ≥ H19 → per-axis absorbed by H19's Charb landscape → different mechanism needed

H22 vs H21 comparison
├─ H22 vol_p better than H21 by > 0.02pp → MAE_aux adds value → standard config going forward
└─ H22 ≈ H21 on vol_p → MAE_aux redundant with clamp+Charb combination
```

### Stop conditions

- **Stop H21 early** if EP5 val_vol_p > H19's EP5 value AND w_vol_p pinned at 0.15 with no downward trend in val_vol_p → clamp mechanism not engaging, abort.
- **Stop H23 early** if EP3 val_vol_p > 5.5% → per-axis causing severe vol_p starvation beyond H19 level.
- **Plateau protocol trigger**: if 3 consecutive experiments fail to beat H19's test_wss while achieving floor → time to attempt IMTL-G (H29) as tier-shift from GradNorm tuning to gradient surgery.

### 5.85% wss target path

The gap from H19 (6.634%) to target (5.85%) is 0.784pp. The validated additive levers are:
- Per-axis τ weights: −0.245pp (H11b vs H8 baseline)
- τ_z Charb (curvature+Charb): −0.062pp to −0.093pp (H10b, H19 vs SOTA)
- τ_y Charb: unknown, likely −0.03 to −0.06pp (proportional to τ_z effect scaled by axis error)
- Stricter floor compliance cost: −0.05 to −0.15pp wss overhead

Sum of validated levers: ~0.34–0.39pp. Gap remaining to 5.85% target: ~0.39–0.44pp. This indicates additional mechanisms beyond the current stack are needed. The IMTL-G experiment (H29) is the highest-potential tier-shift option, as it could eliminate the floor-preservation wss overhead entirely and free up budget for additional wss-specific mechanisms.
