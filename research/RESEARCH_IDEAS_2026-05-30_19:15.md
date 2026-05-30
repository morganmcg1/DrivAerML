# Wave-4 Research Hypotheses — 2026-05-30 19:15Z

**Context:** dl24 H147 SOTA test_WSS=6.5409% vs Transolver-3 paper target <5.85% (0.69pp gap).
Wave-3 verdict sealed: GradNorm-α grid exhausted, EMA decay=0.9999 (H172) is the only productive mechanism.
Wave-4 pivots to EMA-derivative refinement + new mechanistic axes.

**Hard constraints:**
- Single-model DDP8 only (ensembles BANNED)
- All 8 GPUs, max 24h wall-clock
- Floor caps: test_VP ≤ 3.643%, test_SP ≤ 3.577% (EXTREMELY tight — only 0.014pp cushion on SP)
- No new training data
- Do NOT iterate on GradNorm α / vol_p floor (exhausted)
- H147 full CLI baseline: `--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 --ema-decay 0.9999 --epochs 30 --lr 3e-4`

**Exhausted axes (DO NOT REPEAT):** Lookahead (H143), SWA (H141), GradNorm-α variants (H173/H176/H178/H180), vol_p floor variants, single-knob Charbonnier amplification, pe_num_features up, extended cosine standalone, LR standalone, hidden_dim 512→640 (H123), model_layers 6→7+ (H132), IMTL-G (H136), PCGrad (H137), Ada-Temp (H135), GALE (H134), β-grid.

---

## Ranked Hypotheses (by expected impact × tractability)

---

### H181: EMA Longer Window — decay 0.99995

**Priority:** HIGH
**Student assignment:** frieren (after H176 terminal ~20:00Z)

**Hypothesis:**
EMA at decay=0.9999 (H172) produced the only sub-H147-EP6-reference run in wave-3. The mechanism is that EMA with high decay acts as an implicit ensemble over a longer training history, smoothing sharp gradient directions and biasing the averaged weights toward flatter minima. If the optimal averaging window is wider than what decay=0.9999 provides, then decay=0.99995 should extract a slightly smoother trajectory and tighter terminal minima — especially in the cosine tail where learning is slow and EMA memory spans many gradient steps. This is the single most directly motivated follow-up to the H172 finding, testing whether the EMA window optimum is still right of 0.9999.

**Mechanism targeted:** EMA implicit regularization — longer averaging window reduces variance at cost of slower weight tracking. Optimal window should exist where bias-variance trade-off is minimized for this dataset and architecture.

**Concrete CLI changes:**
```
--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
--ema-decay 0.99995 \
--epochs 30 --lr 3e-4
```
All other flags identical to H147 SOTA.

**Expected impact:** val_WSS improvement of 0.01-0.03pp over H172 terminal if window optimum lies above 0.9999. If H172 is already near optimum, expect ≤0.01pp. Moderate probability (35%) of beating H147 outright.

**Risk:** SP floor breach LOW — EMA-only change does not shift gradient pressure between heads. Primary risk is EMA tracking too slowly during early warmup, causing delayed convergence (observable via EP1-3 val metrics being worse than H172 EP1-3). Secondary risk: decay=0.99995 over 30 epochs with ~10k steps/epoch = EMA memory of ~20k steps — may be too smooth, washing out fine-grained minima.

**Wall-clock budget:** 8-EP smoke (~6h), 30-EP main (~20h). Full 30-EP run fits 24h.

**Falsifying result:** If val_WSS at EP20 is above H172 EP20 (6.6521), the longer window is over-smoothing and H172 is near the EMA optimum. Close and move to H182.

---

### H182: EMA 0.9999 + LR Peak Amplification 1.3×

**Priority:** HIGH
**Student assignment:** nezuko (after H180 terminal ~22:00Z)

**Hypothesis:**
H172 (EMA 0.9999) shows decelerating descent at EP17→EP20 (~-0.0025pp/EP vs -0.023pp/EP at EP15→17). This deceleration pattern suggests the optimizer is near a flat plateau in the loss landscape — not a global minimum. Increasing peak LR by 1.3× provides higher escape energy early in training, driving exploration into sharper/deeper basins that then benefit more from EMA averaging during the cosine tail. The combination is mechanistically distinct from either change alone: higher LR explores more aggressively, EMA 0.9999 stabilizes the noisy trajectory, producing a terminal average that neither high-LR-alone nor EMA-alone would find.

**Mechanism targeted:** Coupling between optimizer exploration radius (controlled by peak LR) and EMA bias reduction. Higher LR → sharper descent lines but noisier weights; EMA smooths them, extracting signal from the trajectory.

**Concrete CLI changes:**
```
--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
--ema-decay 0.9999 \
--lr 3.9e-4 \
--epochs 30
```
Note: 3.9e-4 = 3e-4 × 1.3. No warmup change — H147 uses default warmup config.

**Expected impact:** If peak LR is currently sub-optimal (likely given H147 was tuned on AdamW stack then migrated to Lion), the higher LR should give 0.02-0.05pp WSS improvement. Risk of SP floor breach increases slightly due to noisier early optimization.

**Risk:** SP floor breach MEDIUM — higher LR amplifies gradient noise across all heads. Monitor val_SP at EP4-6; if val_SP approaches 3.90 (vs H147 EP6 ref 3.9107), flag as breach risk. Lion's update magnitude is bounded by sign(g), so LR amplification is less explosive than for AdamW — partial mitigation.

**Wall-clock budget:** 8-EP smoke (~6h), 30-EP main (~20h).

**Falsifying result:** val_SP at EP6 breaches 3.95 OR val_WSS at EP10 is above H172 EP10 equivalent. Close if either condition met; the higher LR is hurting generalization.

---

### H183: EMA 0.9999 + Extended Cosine (40 EP within 24h)

**Priority:** HIGH
**Student assignment:** fern (after H178 terminal ~21:30Z)

**Hypothesis:**
H147 uses 30 EP cosine. H172 at EP20 shows val_WSS=6.6521 with descent still active (slope -0.0025pp/EP). If the cosine tail is not yet exhausted at EP30, extending the schedule to 40 EP (but still completing within 24h) allows the optimizer and EMA to continue refining in the low-gradient regime. EMA at decay=0.9999 compounds this: over more training steps at low LR, the averaged weight trajectory has more signal-to-integrate, potentially tightening the final minima further. This is distinct from "train longer" — it is extending the cosine TAIL where EMA smoothing is most valuable.

**Mechanism targeted:** Cosine tail integration time × EMA smoothing. The H147 30-EP cosine reaches LR_min by EP30; 40 EP extends the low-LR regime by ~33%, giving EMA more steps to average over a narrow loss basin.

**Concrete CLI changes:**
```
--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
--ema-decay 0.9999 \
--epochs 40 \
--lr 3e-4
```
**CRITICAL**: Check wall-clock before dispatching. H147 at 30 EP runs ~20h. At 40 EP this would be ~27h — EXCEEDS 24h budget. Must add `--lr-cosine-t-max 30` to keep T_max=30 (cosine completes at EP30, then runs constant LR_min for EP31-40), OR reduce batch/steps per epoch. Alternative: use `SENPAI_TIMEOUT_MINUTES=1380` (23h) with 40 EP and let it terminate naturally. Verify step count per epoch before dispatching — if steps_per_epoch × 40 > 24h, revert to 35 EP.

**Expected impact:** 0.01-0.04pp WSS improvement over H147 30-EP terminal if the descent slope at EP30 is still non-trivial (observed H172 EP20 slope -0.0025pp/EP → EP30 would be ~6.62-6.65 → still ~0.1pp above H147 test). This run is most valuable if H172 shows the slope has NOT bottomed by EP30.

**Risk:** SP floor breach LOW to MEDIUM — extended training at low LR tends to tighten all metrics proportionally. Primary risk is that 40 EP introduces overfitting in low-data regimes (400 training samples — plausible). Monitor val_SP at EP35; if it regresses vs EP30, overfitting has set in.

**Wall-clock budget:** Smoke at 10 EP (~8h), main at 35-40 EP pending step-count check. Set TIMEOUT=23h.

**Falsifying result:** val_WSS at EP35 shows regression vs EP30 (overfitting). If val_WSS stalls or rises at EP32+, 40-EP is too long. Stop early and report EP30 metrics.

---

### H184: EMA 0.9999 + Wider Attention (heads 8→16, dim kept)

**Priority:** HIGH
**Student assignment:** tanjiro (after H172 terminal ~00:30Z)

**Hypothesis:**
H147's architecture (attention heads, layer count, hidden dim) was fixed before EMA was added. With EMA at 0.9999 providing stronger implicit regularization, the model may now tolerate a wider attention configuration that would otherwise overfit — more heads = more diverse attention patterns = potentially better surface-feature decomposition for WSS (the most spatially structured signal). Doubling the head count from 8 to 16 while keeping total dim fixed (dim_per_head halved) shifts the inductive bias from fewer rich channels to more diverse queries. This is the "structural" perturbation in the wave-4 queue.

**Mechanism targeted:** Attention diversity in the surface head — WSS is driven by near-wall flow structure (separation bubbles, attachment lines) that benefits from multiple attention patterns. More heads with narrower per-head dim encourages specialization.

**Concrete CLI changes:**
```
--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
--ema-decay 0.9999 \
--epochs 30 --lr 3e-4 \
--model-heads 16
```
**PREREQUISITE**: Verify `--model-heads` is a valid CLI flag in `train.py` Config. Search Config for `heads`, `num_heads`, or `n_heads` before dispatching. If not in Config, the student must add it (single-line change in model.py constructor call). Check PR #1469 (H172) for model arch args used.

**Expected impact:** Architecture changes have shown diminishing returns past H123 (hidden_dim 512→640 failed). However, EMA as regularizer may unlock headroom that was previously inaccessible. Probability of WSS improvement: 30%. If it works, expect 0.03-0.06pp.

**Risk:** SP floor breach MEDIUM-HIGH — head count changes affect all prediction heads equally. At 30 EP, more diverse attention may overfit surface pressure features (only 400 training samples). Monitor val_SP at EP8-12.

**Wall-clock budget:** 8-EP smoke (required — architecture change), 30-EP main if smoke val_WSS < H172 EP8 val_WSS. If smoke fails (val_WSS > 6.8), close and dispatch H_GSAM instead.

**Falsifying result:** Smoke EP8 val_WSS > 7.0 OR val_SP > 4.1. Architecture divergence from H147 is not adding value.

---

### H_WSD_COOLDOWN: Warmup-Stable-Decay LR with Rapid Final Cooldown

**Priority:** MEDIUM
**Student assignment:** next available after wave-4 EMA cluster dispatched

**Hypothesis:**
The cosine schedule used by H147 has a smooth, slow decay across 30 epochs. WSD/trapezoidal scheduling (Wen et al., "Scaling LLM Training Regularization," ICLR 2025) uses three phases: warmup → stable (constant LR) → rapid linear decay. The key finding from LLM training is that a rapid cooldown from 10× to 100× shorter than the stable phase produces lower final loss than a gradual cosine for the same total compute budget. The mechanism: the stable phase allows the optimizer to explore broadly at high LR; the rapid cooldown then "quenches" the weights to the nearest local minimum, which is often sharper but deeper than the cosine terminal. For EMA models, the quench phase is where EMA averaging is most valuable — it averages over the quench trajectory, producing a smoother minimum.

**Implementation note:** `trainer_runtime.py` `build_lr_scheduler` only supports cosine+warmup. WSD requires adding a `ConstantLR` + `LinearLR` cooldown to the `SequentialLR` chain. Student must add a code path in `build_lr_scheduler`:
```python
if config.lr_wsd_stable_epochs > 0:
    stable = ConstantLR(optimizer, factor=1.0, total_iters=stable_steps)
    cooldown = LinearLR(optimizer, start_factor=1.0, end_factor=config.lr_min/config.lr, total_iters=cooldown_steps)
    return SequentialLR([warmup, stable, cooldown], milestones=[warmup_steps, warmup_steps+stable_steps])
```
New Config fields needed: `lr_wsd_stable_epochs: int = 0` and `lr_wsd_cooldown_epochs: int = 0`.

**Concrete CLI changes (after code addition):**
```
--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
--ema-decay 0.9999 \
--lr 3e-4 --lr-wsd-stable-epochs 22 --lr-wsd-cooldown-epochs 6 \
--epochs 30
```
Warmup 2 EP → stable 22 EP → cooldown 6 EP = 30 EP total.

**Expected impact:** 0.03-0.06pp WSS improvement if the rapid cooldown quenches into a deeper minimum than the gradual cosine. This is one of the few interventions with strong external evidence from a different (but mechanistically adjacent) domain.

**Risk:** SP floor breach MEDIUM — rapid LR drop during cooldown can cause instability if the gradient magnitudes are large. EMA smoothing helps but does not eliminate risk. Monitor val_SP at first cooldown epoch.

**Wall-clock budget:** Requires code change (1-2h student work) + 8-EP smoke + 30-EP main. Total ~24h.

**Falsifying result:** val_WSS at EP28 (end of cooldown) is above cosine-equivalent H147/H172 at EP28. The rapid decay is not quenching into a better basin.

---

### H_GSAM: GSAM Sharpness-Aware Optimization on H147 Stack

**Priority:** MEDIUM
**Student assignment:** next available

**Hypothesis:**
H147 uses Lion optimizer which does not explicitly seek flat minima. GSAM (Gradient-Surrogate Aware Minimization, Zhuang et al. 2022) finds flat minima by perturbing weights in the gradient direction and computing the surrogate gradient on the perturbed model — more efficient than full SAM (avoids double forward pass). Flat minima generalize better to OOD test points, which is directly relevant to the DrivAerML 50-sample test set. With only 400 training samples, sharpness regularization is especially valuable. GSAM can be wrapped around Lion, using Lion for the inner step and GSAM perturbation for the outer gradient correction.

**Implementation note:** Use the `gsam` library (pip install gsam) or implement the perturbation step directly. GSAM adds ~50-70% compute overhead (one additional gradient computation per step). At H147's ~20h runtime, this gives ~30-33h — EXCEEDS 24h budget. Mitigate by: (a) using only 20 EP instead of 30, (b) reducing rho (perturbation radius) to reduce compute, or (c) using periodic GSAM (apply every other step). Recommend starting with 20 EP + rho=0.05.

**Concrete implementation sketch:**
```python
from gsam import GSAM
base_optimizer = lion_optimizer  # build Lion normally
optimizer = GSAM(model.parameters(), base_optimizer, rho=0.05, alpha=0.4)
```

**Concrete CLI changes:** Requires new `--use-gsam` flag + `--gsam-rho` in Config. Student must add code.
```
--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
--use-gsam --gsam-rho 0.05 \
--ema-decay 0.9999 \
--epochs 20 --lr 3e-4
```

**Expected impact:** 0.05-0.10pp WSS improvement if flatness is a key bottleneck. GSAM consistently outperforms SAM in transformer settings (Zhuang et al., 2022). With EMA averaging over a flatter trajectory, the compound effect may be larger.

**Risk:** SP floor breach LOW — GSAM seeks flat minima for ALL heads simultaneously, which should not disproportionately harm SP. Primary risk is compute budget breach (30+h). Verify wall-clock from smoke run before main.

**Wall-clock budget:** 8-EP smoke + 20-EP main. If 20-EP main exceeds 22h, reduce to 16 EP.

**Falsifying result:** 8-EP smoke takes >6h wall-clock (budget exceeded) OR val_WSS at EP8 is above H147 EP8 (6.6498) by >0.05pp. Close if GSAM is not competitive with baseline within the compute budget.

---

### H_SELF_DISTILL: Self-Distillation from H172 EMA Teacher

**Priority:** MEDIUM
**Student assignment:** next available

**Hypothesis:**
H172's EMA model at EP20+ is a higher-quality predictor than the base model at any single step. Using the EMA model as a teacher for soft-label distillation during subsequent training (on the same architecture, same data) is "self-distillation" — the teacher's smoothed predictions provide richer supervision than the raw CFD ground truth alone, particularly in the low-gradient regions of the flow field where the model is most uncertain. This is mechanistically distinct from standard distillation: the teacher and student share architecture, so the teacher's predictions reflect learned physical regularities rather than capacity differences. The mechanism is closest to the SKD-PINNs framework (self-knowledge distillation for PDE surrogates).

**Implementation note:** Load H172 EMA checkpoint as the teacher model (frozen). During training, at each step compute teacher predictions on the same batch and add a soft KL divergence term to the loss: `L_total = L_GT + α * KL(student_pred, teacher_pred.detach())`. Start with α=0.1. The teacher must be the EMA weights, not the base model weights. Student needs to add checkpoint loading code and the distillation loss term.

**Concrete CLI changes:**
```
--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
--ema-decay 0.9999 \
--epochs 30 --lr 3e-4 \
--distill-teacher-path /path/to/H172_ema_checkpoint.pt \
--distill-alpha 0.1
```
Note: H172 checkpoint path must be confirmed from tanjiro's run artifacts before dispatching. This experiment CANNOT start until H172 finishes and its checkpoint is saved.

**Expected impact:** 0.02-0.05pp WSS improvement if the EMA teacher has learned physically consistent regularities that the GT supervision alone does not provide. Risk of degradation if the teacher's errors compound.

**Risk:** SP floor breach LOW to MEDIUM — distillation adds a soft regularizer on all outputs. If teacher is well-calibrated (H172 val_SP is at H147 EP6 reference level), the soft targets should be safe. If teacher has VP imbalance issues, the distillation loss may propagate them.

**Wall-clock budget:** Depends on H172 finishing (~00:30Z tanjiro). Earliest dispatch: EP1 morning. 30-EP main ~20h.

**Falsifying result:** val_WSS at EP10 is above H172 EP10 equivalent (i.e., distillation is hurting). The teacher's soft labels are adding noise rather than signal.

---

### H_SPECTRAL_LOSS: Radially-Binned Spectral Auxiliary Loss on WSS

**Priority:** MEDIUM
**Student assignment:** next available

**Hypothesis:**
The WSS field has strong spatial structure: high-frequency features near separation lines and reattachment zones, low-frequency structure in the attached boundary layer. Standard L2 loss treats all spatial frequencies equally, which biases the model toward fitting low-frequency components (spectral bias). A radially-binned spectral error auxiliary loss (motivated by LOGLO-FNO, Li et al. 2023) penalizes errors across spatial frequency bands, forcing the model to reproduce high-frequency WSS structures that are physically important for drag prediction. For DrivAerML's surface mesh, the "radial bins" correspond to spatial wavenumber bands on the car surface (coarse → fine resolution).

**Implementation note:** Compute FFT (or spectral graph Laplacian eigenmodes if on unstructured mesh) of predicted vs GT WSS on the surface mesh. Bin the spectral errors into K=8 logarithmic wavenumber bands. Add the mean error in each band as auxiliary loss terms with per-band weights, starting with uniform weighting. This requires non-trivial mesh FFT code — on structured parts of the surface (bonnet, roof, rear) a 2D FFT is straightforward; on unstructured parts, use graph Fourier modes or a spatial binning approximation. Recommend starting with a simple 2D FFT approximation on the projected surface.

**Concrete CLI changes:**
```
--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
--ema-decay 0.9999 \
--epochs 30 --lr 3e-4 \
--use-spectral-loss --spectral-loss-weight 0.05 --spectral-bins 8
```
Requires new code in the loss computation (not just a CLI flag).

**Expected impact:** 0.05-0.15pp WSS improvement if spectral bias is a key bottleneck. This is one of the higher-variance hypotheses — the mechanism is well-motivated but implementation quality matters significantly.

**Risk:** SP floor breach MEDIUM — adding explicit WSS spectral loss may increase WSS optimization pressure relative to SP, triggering the resource conservation law (confirmed across H169/H168/H167/H170). Must monitor val_SP at EP4-8. Set `spectral-loss-weight` low (0.05) and do not increase without SP headroom.

**Wall-clock budget:** Code-heavy (2-4h implementation) + 8-EP smoke + 30-EP main (~20h). Total ~26h. Ensure code is ready before GPU is allocated.

**Falsifying result:** val_SP at EP6 approaches 3.92 (within 0.05pp of H147 EP6 val_SP=3.9107 plus 0.01pp floor cushion) → spectral loss is redirecting optimization pressure into SP. Close immediately.

---

### H_LION_BETA2_COOLDOWN: Lion β2 Schedule 0.98→0.995 in Final 10 EP

**Priority:** LOW-MEDIUM
**Student assignment:** next available

**Hypothesis:**
Lion's β2 controls the momentum buffer update rate. Higher β2 = longer memory = more momentum in the gradient direction. During the cosine warmup/peak phase, β2=0.98 is appropriate for rapid descent. In the cosine tail (low LR, refinement phase), higher β2=0.995 makes the optimizer more conservative — smaller effective step sizes relative to the gradient signal, which can help convergence in narrow valleys. This is analogous to the finding in arXiv 2508.01483 that β2 scheduling benefits certain optimizer-model combinations during cooldown. With EMA at 0.9999 providing trajectory smoothing, a more conservative late-training optimizer may compound the smoothing effect.

**Mechanism targeted:** Lion momentum dynamics during cooldown. The hypothesis is that β2=0.98 during the low-LR phase has "excessive memory" — past gradients from higher-LR training contaminate the current update direction, pulling the optimizer away from the local refinement minimum.

**Concrete CLI changes:**
This requires epoch-conditional β2 update in the training loop. Student must add a hook:
```python
if epoch >= epochs - 10:
    for g in optimizer.param_groups:
        g['betas'] = (g['betas'][0], 0.995)
```
No new CLI flag needed if hardcoded at `epochs-10`; or add `--lion-beta2-cooldown` and `--lion-beta2-cooldown-start-epoch`.

```
--optimizer lion --lion-beta1 0.95 --lion-beta2 0.98 \
--ema-decay 0.9999 \
--epochs 30 --lr 3e-4 \
--lion-beta2-cooldown 0.995 --lion-beta2-cooldown-start-epoch 20
```

**Expected impact:** 0.01-0.03pp WSS improvement. This is a fine-grained optimization of an already well-tuned optimizer — low expected impact but also low risk and low cost.

**Risk:** SP floor breach VERY LOW — changing β2 during cooldown is a minor perturbation to an otherwise H147-identical stack.

**Wall-clock budget:** No architecture change. 8-EP smoke (run from EP20 of an existing checkpoint or from scratch) + 30-EP main. If run from scratch: ~20h. If hot-started from H172 EP20 checkpoint with β2 change: ~6h for EP21-30.

**Falsifying result:** val_WSS at EP30 is within 0.005pp of H147 EP30 val (6.5451). If no improvement vs the EMA-only baseline at equivalent compute, the β2 scheduling is not adding value.

---

## Priority Ranking Summary

| Rank | ID | Mechanism | Priority | WSS Improvement Estimate | SP Breach Risk | Budget |
|---:|---|---|---|---:|---|---|
| 1 | H181 | EMA decay 0.99995 | HIGH | 0.01-0.03pp | LOW | 20h |
| 2 | H182 | EMA 0.9999 + LR 1.3× | HIGH | 0.02-0.05pp | MEDIUM | 20h |
| 3 | H183 | EMA 0.9999 + 40-EP cosine | HIGH | 0.01-0.04pp | LOW | 23h |
| 4 | H184 | EMA 0.9999 + heads 8→16 | HIGH | 0.03-0.06pp | MEDIUM-HIGH | 20h (after smoke) |
| 5 | H_WSD_COOLDOWN | WSD trapezoidal LR | MEDIUM | 0.03-0.06pp | MEDIUM | 24h (needs code) |
| 6 | H_GSAM | GSAM sharpness-aware | MEDIUM | 0.05-0.10pp | LOW | 22h (20 EP only) |
| 7 | H_SELF_DISTILL | EMA teacher → student KD | MEDIUM | 0.02-0.05pp | LOW-MEDIUM | 20h (needs H172 done) |
| 8 | H_SPECTRAL_LOSS | Spectral auxiliary WSS loss | MEDIUM | 0.05-0.15pp | MEDIUM | 26h (needs code) |
| 9 | H_LION_BETA2_COOLDOWN | Lion β2 0.98→0.995 at EP20 | LOW-MEDIUM | 0.01-0.03pp | VERY LOW | 20h |

---

## Decision Tree for Wave-4

```
Wave-4 EMA cluster results (H181-H184 terminal):
├── ANY of H181-H184 beats H147 (val_WSS < 6.54 at EP30)
│   ├── Merge winner immediately
│   ├── Dispatch H_WSD_COOLDOWN + H_GSAM on freed GPUs
│   └── H_SELF_DISTILL once H172 checkpoint is available
│
├── H181-H184 all fail (val_WSS ≥ 6.60 at terminal)
│   ├── EMA-derivative thesis WEAKENED
│   ├── Dispatch H_WSD_COOLDOWN + H_GSAM as new mechanism tier
│   ├── Dispatch H_SPECTRAL_LOSS if SP headroom permits
│   └── If ALL of above fail: architecture redesign tier
│       (full Transolver rewrite, different backbone, etc.)
│
└── H181-H184 mixed (some promising, some not)
    ├── Merge best, continue EMA-variant grid
    └── Compound best EMA config with H_WSD_COOLDOWN
```

---

## Stop Conditions

- **EMA-derivative thesis falsified**: All of H181-H184 finish with val_WSS ≥ 6.60 at EP30. → Tier shift to H_WSD_COOLDOWN + H_GSAM + H_SPECTRAL_LOSS.
- **WSD + GSAM + Spectral all fail**: After ~6 more experiments, if no run has val_WSS < 6.60 → Escalate to full architecture redesign (new backbone, attention mechanism redesign, multi-scale mesh processing).
- **SP floor breach becomes uncontrolled**: If 3 consecutive runs breach val_SP > 4.0 → pause and diagnose the gradient pressure mechanism before continuing.

---

## Research State Update (2026-05-30 19:15Z)

**Current best explanation for limiting progress:**
The H147 stack is near a local minimum of the loss landscape for this architecture+dataset+optimizer combination. EMA at 0.9999 (H172) provides incremental improvement (~0.05-0.10pp val_WSS over H147 mid-train) but is decelerating. The fundamental bottleneck is likely a combination of: (1) architecture capacity ceiling for 400 training samples, (2) cosine schedule not optimally shaped for the fine-tuning phase, and (3) optimizer dynamics (Lion β2=0.98) being tuned for exploration rather than exploitation in the terminal phase.

**Ruled-out paths (do not repeat):**
- GradNorm-α variants (H173/H176/H178/H180): all produce VP parity at cost of WSS regression
- SWA (H141): FAILED
- Lookahead (H143): FAILED
- Extended cosine standalone: marginal
- LR standalone: marginal
- Architecture scaling (hidden_dim, layer count): FAILED in isolation

**Open uncertainties (top 3):**
1. Does the EMA optimum lie at 0.9999 (H172) or further right (H181)?
2. Does WSD's rapid cooldown quench into a meaningfully different minimum than cosine for this dataset size?
3. Is GSAM's sharpness pressure compatible with Lion's sign-gradient updates (no prior evidence in this combination)?

**Next most discriminating experiment:** H181 (EMA 0.99995) — cheapest possible test of whether the EMA window is saturated. If H181 val_WSS at EP15 is below H172 EP15 (~6.70), the window optimum is right of 0.9999 and the EMA derivative thesis has headroom. If above, the window is saturated and we must tier-shift.
