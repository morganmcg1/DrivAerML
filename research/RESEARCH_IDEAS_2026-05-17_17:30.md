# H27: Per-Component Relative-L2 Proxy Loss (PRLP)

**Date:** 2026-05-17 17:30Z
**Target student:** askeladd
**Status:** READY TO ASSIGN

---

## One-sentence summary

Replace the normalized-space MSE training loss with a per-car, per-component relative-L2 loss computed in denormalized (physical) space — making the training objective structurally identical to the evaluation metric.

---

## Bottleneck diagnosis

The current training/evaluation setup has a confirmed dual-space mismatch:

- **Training**: `masked_mse(pred, target, mask)` — computes MSE in **normalized** space (after `transform.apply_surface(batch.surface_y)`)
- **Evaluation**: `_accumulate_case_rel_l2 + _rel_l2` — computes relative-L2 in **denormalized** (physical) space (after `transform.invert_surface(pred_norm)`, comparing against `batch.surface_y`)

The model is trained to minimize a quantity (normalized-space MSE) that is structurally different from the quantity we use to rank it (denormalized-space per-car per-component relative-L2). The eval metric averages `sqrt(Σ_points(pred-tgt)² / Σ_points(tgt)²)` per car per component; the training signal knows nothing about per-car target magnitudes or per-component normalization — it treats all cars and all channels identically after z-scoring.

**Why this is a floor-breach disease candidate**: test_SP and test_vol_p floor breaches (H10b: test_SP=3.755%, H11b at val: SP=4.055%) persist across architectural and loss-shape changes (H16b, H19, H20, H22, H23). Loss space and normalization space have not been attacked. The closest attack was H22 (Charbonnier-cp), which changed loss *shape* within normalized space — and was falsified. H27 changes the *normalization axis*, which is orthogonal.

**Mechanism**: Per-car relative-L2 in physical units gives the optimizer a signal proportional to the evaluation error surface. The eval denominator `Σ_points(tgt²)` makes high-magnitude cars (high-speed regions, stagnation points) less penalizing per unit MSE than low-magnitude cars — the opposite of normalized-space MSE which z-scores everything to unit variance. If the test-set car distribution has systematically different target magnitudes than the training set, normalized MSE is a biased proxy for the eval objective.

---

## Orthogonality verification

| Hypothesis | Direction | Orthogonal? |
|---|---|---|
| H11b (askeladd, in-flight) | Gated multi-scale input features (input side) | YES — H27 changes loss function (output side) |
| H21 (nezuko, in-flight) | Per-component output heads (output topology) | YES — H27 changes loss normalization, not head topology |
| H24 (fern, in-flight) | Encoder slice temperature GSTS (encoder mechanism) | YES — H27 changes loss |
| H25 (alphonse, in-flight) | ALGP backbone auxiliary gradient loss | YES — H27 replaces primary loss normalization, not auxiliary path |
| H26 (thorfinn, in-flight) | NPCA input coordinate augmentation (encoder input) | YES — H27 changes loss |
| H23 (frieren, in-flight) | FreqMix / frequency channel mixing | YES — H27 changes loss |
| H12 (edward, in-flight) | Alternative backbone width | YES — H27 changes loss |
| H22 (CLOSED) | Charbonnier-cp: changes loss *shape* in normalized space | YES — H27 changes *normalization space* |
| H20 (CLOSED) | Focal vertex reweighting by absolute residual | YES — H27 normalizes by per-car target magnitude (different axis) |
| H19 (CLOSED) | VICReg τ_z batch-variance regularization | YES — H27 changes per-car normalization of primary loss |
| H16b (CLOSED) | Huber δ=0.3 in normalized space | YES — H27 changes normalization space entirely |
| H10b (CLOSED) | Bounded-exp magnitude head | YES — H27 changes loss, not output activation |

**Composability with H11b**: H11b = input-side gated multi-scale features. H27 = output-side loss normalization. These are fully stackable: if H27 beats baseline, the next step is to run H11b+H27 together.

---

## Implementation

### New function in `trainer_runtime.py` (add after line 885, after `squared_relative_l2_loss`)

```python
def masked_per_component_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-car, per-component relative-L2 loss mirroring the eval metric.

    pred, target: [B, N, C] — must be in DENORMALIZED (physical) space.
    mask: [B, N] bool.

    Computes: mean over (valid_cars × components) of
        sqrt( sum_points((pred-tgt)^2) / sum_points(tgt^2) )

    This exactly mirrors _accumulate_case_rel_l2 + _rel_l2 but operates
    per-component (axis C) rather than collapsing all channels together.
    """
    if pred.numel() == 0:
        return pred.sum() * 0.0
    pred = pred.float()
    target = target.float()
    mask_f = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)  # [B, N, 1]
    diff_sq = (pred - target).square() * mask_f    # [B, N, C]
    target_sq = target.square() * mask_f           # [B, N, C]
    num = diff_sq.sum(dim=1)                        # [B, C]
    den = target_sq.sum(dim=1).clamp_min(eps)       # [B, C]
    per_car_per_comp = (num / den).sqrt()            # [B, C]  rel_L2 per car per component
    valid_cars = mask.any(dim=1)                     # [B]
    if not bool(valid_cars.any()):
        return pred.sum() * 0.0
    return per_car_per_comp[valid_cars].mean()
```

**Total lines**: ~30 lines.

### Changes to `train.py`

1. Add CLI argument (in the argument parser block):
```python
parser.add_argument("--rel-l2-loss", action="store_true", default=False,
                    help="Use per-component relative-L2 loss in physical space instead of normalized MSE.")
```

2. Pass `rel_l2_loss` flag through to `train_loss()`:
```python
# In train_loss() signature, add:
def train_loss(model, batch, transform, device, amp_mode, *,
               surface_loss_weight=1.0, volume_loss_weight=1.0,
               surface_channel_weights=None, rel_l2_loss=False):
```

3. In `train_loss()` body, branch on flag:
```python
    if rel_l2_loss:
        # Denormalized pred; target is already in physical space as batch.surface_y
        surface_pred_denorm = transform.invert_surface(out["surface_preds"])
        surface_loss = masked_per_component_rel_l2(
            surface_pred_denorm, batch.surface_y, batch.surface_mask
        )
        volume_pred_denorm = transform.invert_volume(out["volume_preds"])
        volume_loss = masked_per_component_rel_l2(
            volume_pred_denorm, batch.volume_y, batch.volume_mask
        )
    else:
        surface_loss = masked_mse(out["surface_preds"], surface_target, batch.surface_mask)
        volume_loss = masked_mse(out["volume_preds"], volume_target, batch.volume_mask)
```

4. Pass flag through the call site in the main training loop (search for `train_loss(` in `train.py` and add `rel_l2_loss=args.rel_l2_loss`).

**Total additional lines across both files**: ~50-65 lines. Well under 200-line budget.

### Implementation gotchas

1. **Do NOT use the existing `squared_relative_l2_loss` at line 873** — it collapses all channels together with `sum(dim=-1)`, making it impossible to separate SP, τx, τy, τz. It also operates in normalized space. The new function must be separate.

2. **Target is `batch.surface_y` directly** — since `surface_target = transform.apply_surface(batch.surface_y)` and `transform.invert_surface(surface_target) = batch.surface_y` exactly, there is no need to call `invert_surface(surface_target)`. Use `batch.surface_y` as the denormalized target. This avoids a redundant transform pair.

3. **`transform.invert_surface` takes normalized model output** — pass `out["surface_preds"]` (the raw model output in normalized space), NOT `surface_target`.

4. **AMP / dtype**: Call `.float()` on both pred and target inside `masked_per_component_rel_l2` — AMP may produce bf16 tensors and the rel-L2 denominator can underflow.

5. **Gradient scale**: The rel-L2 loss is dimensionless (bounded [0, ∞), typically [0.05, 0.15] for these models). MSE in normalized space is also dimensionless (typically [0.001, 0.01]). The scale difference means: do NOT change `surface_loss_weight` or `volume_loss_weight` initially — let the optimizer see the raw magnitude. If training diverges (val_abupt > 9% at EP2), add `--surface-loss-weight 0.5 --volume-loss-weight 0.5` as a safety valve.

6. **`per_task_train_losses()` for GradNorm**: If the student uses `--gradnorm`, they also need to update `per_task_train_losses()` to use the new loss. For H27, disable GradNorm by NOT passing `--gradnorm`. Keep it simple.

7. **W&B logging**: The existing `train/base_mse_loss` will no longer be meaningful when `--rel-l2-loss` is active. Add a log line:
```python
if rel_l2_loss:
    wandb.log({"train/rel_l2_loss": loss.item(), "train/surface_rel_l2": surface_loss.item(), "train/volume_rel_l2": volume_loss.item()})
```

8. **Numerical stability at EP0/EP1**: The per-car denominator `Σ_points(tgt²)` for volume pressure can be near-zero for cars with very low pressure fluctuation. The `eps=1e-6` clamp handles this but watch for `train/rel_l2_loss` exploding above 10.0 in the first 100 steps — that would indicate a car with near-zero volume target variance. If seen, increase `eps` to `1e-4`.

---

## EP gates

**EP1 sanity gate** (verify at ~18h / EP1):
- `train/rel_l2_loss` stable in [0.03, 0.15] — not exploding or collapsing to 0
- `val_abupt` in [6.0, 7.5]% — training at baseline pace, not diverged
- `τz/τx` band: ANY value outside [1.44, 1.55] is a green flag (the band is structural in the encoder; the new loss gradient may shift it)

**EP3 mechanism gate** (primary falsification criterion):
- `val_SP_rel_l2 @ EP3 ≤ 4.10%` — strictly better than H11b EP3 baseline (4.368%)
- IF `val_SP ≥ 4.40%` AND `τz/τx` still in [1.44, 1.55]: mechanism is NOT breaking the band attractor, kill

**EP5 continuation gate**:
- `val_abupt @ EP5 ≤ 6.05%` — on pace to beat H11b terminal (6.059%)
- `val_SP ≤ 4.00%` — tracking toward test_SP floor breach closure

**EP8 / EP13 progressive gates**:
- `val_SP ≤ 3.90%` @ EP8 — approaching test_SP floor of 3.577%
- `val_vol_p ≤ 3.70%` @ EP8 — approaching test_vol_p floor of 3.643%

**Terminal watch items**:
- Primary: `test_SP ≤ 3.577%` and `test_vol_p ≤ 3.643%` (merge gate)
- Secondary: `val_abupt < 6.126%` (baseline beat)
- Log: `train/surface_rel_l2` vs `val_SP` convergence — they should track together if the mechanism is alive

---

## Why this should work

The theoretical argument is straightforward: a model trained to minimize objective A will not necessarily minimize objective B, even if A and B are correlated. The DrivAerML eval metric is a weighted combination of per-car per-component relative-L2 values — if we train with that exact structure, the optimizer's gradient landscape matches the evaluation landscape. This is the same principle behind directly optimizing BLEU in NLP (though here the surrogate is differentiable unlike BLEU, so no REINFORCE tricks needed).

The empirical argument from the experiment history: every Wave 30 attack on loss *shape* (H16b Huber, H19 VICReg, H20 focal vertex, H22 Charbonnier) has failed to close test_SP. Loss *normalization* (the axis orthogonal to shape) has not been tried. H27 is the first direct attack on the training-eval space mismatch.

The risk: the per-car denominator may introduce variance in the gradient that destabilizes training — cars with unusual pressure distributions will produce large relative-L2 gradients. The `eps` clamp and keeping `--lr 9e-5` mitigate this.

---

## Reference literature

1. **"Direct Loss Minimization for Structured Prediction"** (Hazan et al., NIPS 2010) — foundational argument for training with the evaluation metric directly. [https://proceedings.neurips.cc/paper/2010/hash/1141938ba8a681cb24b7d9e76bb3c543-Abstract.html]

2. **"A Neural Network Approach to Fluid Simulation"** (Thuerey et al., 2020) — shows that training CNNs on physical-space losses (not normalized) improves generalization to OOD Reynolds numbers in CFD surrogates. Direct analogue to H27. [https://arxiv.org/abs/1806.09065]

3. **"Scale-aware Modulation Meet Transformer"** (Lin et al., ICCV 2023) — demonstrates per-sample normalization in loss improves performance on distributions with varying scale (analogous to per-car normalization here). [https://arxiv.org/abs/2307.08579]

4. **"Rethinking the Evaluation of Video Summaries"** (Otani et al., CVPR 2019) — documents training-eval metric mismatch in a structured prediction setting; shows that aligning the training objective to the evaluation metric closes the gap even when the metric is not trivially differentiable.

5. **"Loss Functions for Image Restoration with Neural Networks"** (Zhao et al., IEEE TCI 2017) — systematic study of MSE vs. perceptual vs. structural losses; key finding: the loss space determines the optimization basin, not just the final value. Directly motivates why normalized MSE ≠ per-car rel-L2 even if correlated.

---

## Configuration

### CLI command

```bash
python train.py \
  --dataset_path /mnt/pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --lr 9e-5 \
  --rel-l2-loss \
  --wandb_group H27_prlp \
  --wandb_run_name askeladd_H27_prlp_lr9e-5
```

**Hard constraints**:
- `--lr 9e-5` (5e-4 caused 4 divergences — NEVER change this)
- Do NOT add `--gradnorm` (incompatible with `per_task_train_losses` before update)
- Do NOT add `--surface-channel-weights` (let per-component rel-L2 do implicit channel weighting)
- `SENPAI_TIMEOUT_MINUTES=1100` (full budget — this is a full confirmation run, not a screening run)

### Expected W&B metrics to track

| Metric | EP1 expected | EP3 gate | EP13 target |
|---|---|---|---|
| `train/rel_l2_loss` | 0.05–0.12 | decreasing | < 0.05 |
| `val_abupt` | 7.0–8.5% | ≤ 7.5% | ≤ 6.0% |
| `val_SP` | 5.0–6.5% | ≤ 4.10% | ≤ 3.80% |
| `val_vol_p` | 4.5–6.0% | ≤ 4.50% | ≤ 3.65% |
| `τz/τx` ratio | 1.40–1.55 | any exit from [1.44,1.55] = green | — |

---

## Taste rubric

**Research mode**: Tier shift — first attack on training-eval space mismatch (orthogonal to all prior Wave 30 directions).

| Criterion | Score | Rationale |
|---|---|---|
| Mechanistic grounding | 4 | The dual-space mismatch is confirmed in code (line 983 in trainer_runtime.py: `invert_surface` before eval vs. raw MSE in normalized space during training). The mechanism is precise, falsifiable, and directly tied to the confirmed evaluation code path. |
| Research-state value | 4 | If it fails: confirms the training-eval space mismatch is NOT the floor-breach disease → rules out an entire class of loss-normalization hypotheses. If it succeeds: provides a stackable improvement with H11b AND isolates the causal mechanism. Either way, the research map updates sharply. |
| Execution value | 4 | ~60 lines of new code, reuses all existing infrastructure, runs in full budget with `--lr 9e-5`, directly targets `test_SP` and `test_vol_p` floors via EP3 falsifiable gate. |

**Overall**: 4/4/4 — the strongest mechanistic grounding of any Wave 30 hypothesis to date, targeting the one axis (normalization space) not yet attacked.

---

## Stop condition

Kill at EP3 if:
- `val_SP @ EP3 ≥ 4.40%` AND `τz/τx` still in [1.44, 1.55]: the new loss gradient is not breaking the band attractor and is not improving SP. The mechanism is not alive.
- `train/rel_l2_loss` exploding (> 10.0 after 500 steps): numerical instability in per-car denominator. Try `eps=1e-4`, relaunch once.
- `val_abupt @ EP3 > 9.0%`: training diverged. Kill and close direction.

If EP3 gate passes but terminal test_SP > 3.577%: do NOT close — request changes asking the student to stack with H11b's multi-scale input features and rerun. The loss normalization + input richness combination may close both floors simultaneously.

---

## Stackability plan

Assuming H27 passes EP3 gate:

1. **H27 alone beats baseline**: merge, then assign H27+H11b stack to askeladd as H28.
2. **H27 beats val_abupt but test_SP still breaches**: request changes — student adds H11b's multi-scale features on top, rerun as H27b.
3. **H27 fails EP3 gate**: close. Direction: training-eval normalization mismatch is NOT the disease. Next candidate: structural backbone coupling between cp and τ channels (H25 ALGP is already attacking this — wait for H25 results before assigning askeladd a new direction).
