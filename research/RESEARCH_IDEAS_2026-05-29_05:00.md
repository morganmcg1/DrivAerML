# Research Ideas — 2026-05-29 05:00Z

**Context**: H185+TTA merged as new SOTA (val_abupt=5.9755%, test_WSS=6.7214%). Gap to Morgan target: −0.872pp on test_WSS. ~6h compute remaining. 5 idle students: alphonse, fern, frieren, nezuko, thorfinn. Active eval sprints: H212 (edward), H213 (tanjiro), H214 (askeladd).

**TTA is now standard**: Any training winner gets TTA applied at eval time (+4-5bp on test_WSS, free).

**Merge gate**: val_abupt < 5.9755% AND test_abupt < 5.8221%. Test floors: VP ≤ 3.421%, SP ≤ 3.577%, WSS ≤ 6.727%.

---

## H215 — Ultra-Low-LR Continuation from H185 EP13

**Student**: alphonse
**Type**: Training-light (~1.5h)
**Rank**: 1 (highest confidence, direct analogue to yi#657 success)

**Rationale**: The yi#657 run (fern's prior reference) demonstrated that continuing from a strong checkpoint at lr=1e-6 for 2-3 epochs improved every metric. H185 EP13 is the strongest checkpoint we have (val_abupt=5.9755%). Ultra-low LR continuation is optimizer-anchored: it refines the basin without escaping it, meaning permutation-symmetry issues (Finding R, Finding O) cannot arise. TTA will be applied at eval time, so even a 5bp raw improvement becomes ~9-10bp after TTA stacking.

**Recipe**:
```
torchrun --nproc_per_node=8 train.py \
  --wandb_run yw2a5dyl \
  --resume_from_checkpoint epoch-13 \
  --lr 1e-6 \
  --epochs 3 \
  --lr_warmup_epochs 0 \
  --lr_schedule constant \
  --optimizer lion \
  --beta1 0.9 --beta2 0.99 \
  --weight_decay 5e-4 \
  --batch_size 4 \
  --mirror_aug_p 0.25 \
  --tau_y 3.0 --tau_z 2.0 \
  --surface_loss_weight 2.0 \
  --volume_loss_weight 0.5 \
  --ema_decay 0.999 \
  --wandb_group H215-continuation
```
Then evaluate best EMA checkpoint with TTA (`eval_tta_h209.py`).

**Expected runtime**: ~1.5h
**Success criterion**: val_abupt < 5.9755% OR test_WSS < 6.7214% with TTA applied

---

## H216 — H185 Recipe + tau_y=2.0 Full Retrain

**Student**: fern
**Type**: Training-heavy (~3.5h)
**Rank**: 2

**Rationale**: Finding J established that the tau_y stacking failure threshold lies in (2.0, 3.0]. H185 uses tau_y=3.0, which is at or above the failure boundary. The WSS_x slope flip (not recovered by TTA per Finding Q) may be caused by tau_y=3.0 disrupting the WSS_x basin during training. Retraining with tau_y=2.0 while keeping all other H185 components (mirror p=0.25, Lion, lr=9e-5, etc.) tests whether de-escalating tau_y restores the WSS_x slope without sacrificing val_abupt. If slope is restored, TTA will provide a larger marginal gain (Finding Q: TTA recovers WSS_y most, but WSS_x recovery requires the slope to be intact at basin entry).

**Recipe**:
```
torchrun --nproc_per_node=8 train.py \
  --lr 9e-5 \
  --optimizer lion \
  --beta1 0.9 --beta2 0.99 \
  --weight_decay 5e-4 \
  --batch_size 4 \
  --epochs 13 \
  --lr_warmup_epochs 1 \
  --lr_schedule cosine \
  --mirror_aug_p 0.25 \
  --tau_y 2.0 --tau_z 2.0 \
  --surface_loss_weight 2.0 \
  --volume_loss_weight 0.5 \
  --ema_decay 0.999 \
  --wandb_group H216-tau_y2
```
Eval with TTA at EP13 best EMA checkpoint.

**Expected runtime**: ~3.5h
**Success criterion**: val_abupt < 5.9755% with healthy WSS_x slope (negative, not flipped); test_WSS < 6.7214% after TTA

---

## H217 — H185 EP15 Extension from Checkpoint

**Student**: frieren
**Type**: Training-light (~2h)
**Rank**: 3

**Rationale**: The H185 val slope was still negative at EP13 (not yet converged). Cosine schedule extension by 2 additional epochs from the EP13 EMA checkpoint — with the cosine cycle extended, not restarted — gives the optimizer more time in the current basin. This is lower-risk than a full retrain and tests whether the slope signal continues beyond EP13. No architecture or recipe changes; single isolated variable is epoch count. TTA applied at new best checkpoint.

**Recipe**:
```
torchrun --nproc_per_node=8 train.py \
  --wandb_run yw2a5dyl \
  --resume_from_checkpoint epoch-13 \
  --lr 9e-5 \
  --optimizer lion \
  --beta1 0.9 --beta2 0.99 \
  --weight_decay 5e-4 \
  --batch_size 4 \
  --epochs 15 \
  --epochs_already_done 13 \
  --lr_warmup_epochs 0 \
  --lr_schedule cosine \
  --mirror_aug_p 0.25 \
  --tau_y 3.0 --tau_z 2.0 \
  --surface_loss_weight 2.0 \
  --volume_loss_weight 0.5 \
  --ema_decay 0.999 \
  --wandb_group H217-ep15
```
Eval with TTA at EP14 and EP15 best EMA checkpoints.

**Expected runtime**: ~2h
**Success criterion**: val_abupt < 5.9755% at EP14 or EP15; test_WSS < 6.7214% after TTA

---

## H218 — Mirror p=0.5 + tau_y=2.0 Full Retrain

**Student**: nezuko
**Type**: Training-heavy (~4h)
**Rank**: 4

**Rationale**: Finding L showed p=0.5 produces the steepest WSS slope (−0.296pp) but worse val. The H185 winner used p=0.25 — which Finding L called "slope collapser" in the raw result, but in context p=0.25 + tau_y=3.0 is what caused the collapse (tau_y=3.0 is near the stacking failure boundary). The untested combination is p=0.5 + tau_y=2.0: p=0.5 for maximum slope signal, tau_y=2.0 below the stacking failure threshold. This recipe may produce a deeper, better-structured basin than H185 while retaining the mirror equivariance that makes TTA effective. If WSS_x slope is intact, TTA gain will be additive rather than partial (Finding Q).

**Recipe**:
```
torchrun --nproc_per_node=8 train.py \
  --lr 9e-5 \
  --optimizer lion \
  --beta1 0.9 --beta2 0.99 \
  --weight_decay 5e-4 \
  --batch_size 4 \
  --epochs 13 \
  --lr_warmup_epochs 1 \
  --lr_schedule cosine \
  --mirror_aug_p 0.5 \
  --tau_y 2.0 --tau_z 2.0 \
  --surface_loss_weight 2.0 \
  --volume_loss_weight 0.5 \
  --ema_decay 0.999 \
  --wandb_group H218-p05-tauy2
```
Eval with TTA. Report WSS_x slope at EP6, EP9, EP13 checkpoints.

**Expected runtime**: ~4h
**Success criterion**: WSS_x slope negative (not flipped) at EP13; val_abupt < 5.9755% after TTA

---

## H219 — H185 Fresh Seed + Within-Recipe LERP with H185 EP13 (Eval-only prep)

**Student**: thorfinn
**Type**: Training-heavy (~4.5h) + eval-only setup (~15 min)
**Rank**: 5

**Rationale**: Finding R showed that cross-recipe LERP (H112↔H183, H112↔H190) is catastrophic due to permutation symmetry. However, within-recipe LERP is qualitatively different: two H185 runs with different RNG seeds share the same loss landscape topology and basin geometry, so permutation alignment is likely trivial or unnecessary. If val_abupt at α=0.5 is better than either endpoint, this is the cleanest possible ensemble-bypass (single effective model, deterministic weight average). First, run H185 recipe with a different seed to EP13. Then evaluate LERP at α=0.1, 0.25, 0.5 between the two EP13 checkpoints.

**Recipe — training arm**:
```
torchrun --nproc_per_node=8 train.py \
  --lr 9e-5 \
  --optimizer lion \
  --seed 42 \
  --beta1 0.9 --beta2 0.99 \
  --weight_decay 5e-4 \
  --batch_size 4 \
  --epochs 13 \
  --lr_warmup_epochs 1 \
  --lr_schedule cosine \
  --mirror_aug_p 0.25 \
  --tau_y 3.0 --tau_z 2.0 \
  --surface_loss_weight 2.0 \
  --volume_loss_weight 0.5 \
  --ema_decay 0.999 \
  --wandb_group H219-fresh-seed
```
**Recipe — eval arm** (after training completes):
```
python eval_weight_interp.py \
  --checkpoint_a yw2a5dyl/epoch-13 \
  --checkpoint_b <new_run>/epoch-13 \
  --alphas 0.1 0.25 0.5 0.75 0.9 \
  --tta
```
**Expected runtime**: ~4.5h (training) + 20 min (eval sweep)
**Success criterion**: val_abupt at any α < 5.9755% (i.e., LERP improves on both endpoints); if LERP collapses (val > 6.5%), confirms basin separation even within-recipe

---

## H220 — W&B Mid-Run Checkpoint Scan (Eval-only diagnostic)

**Student**: alphonse (after H215 completes, or as parallel fast sprint)
**Type**: Eval-only (~15 min)
**Rank**: 6 (diagnostic; 15 min, zero GPU cost)

**Rationale**: Finding M established that mid-EP EMA checkpoints were not saved program-wide. However, this was established for runs prior to H185. The H185 run `yw2a5dyl` may have per-epoch artifact aliases if the training script was updated after Finding M was logged. This 15-minute W&B API scan costs nothing and could unlock EP6–EP12 checkpoints for ultra-low-LR continuation from an earlier, potentially better-shaped basin.

**Recipe**:
```
python - <<'EOF'
import wandb
api = wandb.Api()
run = api.run("yw2a5dyl")
artifacts = run.logged_artifacts()
for a in artifacts:
    print(a.name, a.aliases, a.created_at)
EOF
```
Report all artifact names and aliases. If per-epoch EMA artifacts exist for EP6–EP12, list them and their val_abupt from W&B metrics.

**Expected runtime**: 15 min
**Success criterion**: Returns list of available mid-EP checkpoints; if EP9 or EP10 EMA exists with val_abupt < 6.0%, flag for ultra-low-LR continuation from that earlier basin point

---

## H221 — H185 + AdamW Warmup Switch at EP10 (Loss-landscape escape)

**Student**: fern (secondary, after H216 result)
**Type**: Training-heavy (~3.5h)
**Rank**: 7

**Rationale**: Lion's sign-based momentum is efficient for early convergence but may be overly aggressive in late-epoch refinement, driving the optimizer into a steep local basin from which WSS_x cannot recover (consistent with Finding Q's slope flip observation). Switching to AdamW at EP10 (after Lion establishes the directional structure) with lr=5e-6 could let the optimizer refine within the basin rather than continuing to compress it. This hybrid-optimizer continuation is an unexplored axis in the current program and directly targets the "late-epoch over-compression" hypothesis for WSS_x basin disruption.

**Recipe**:
```
# Phase 1: Lion to EP10 (resume from H185 EP10 if available, else train from scratch to EP10)
torchrun --nproc_per_node=8 train.py \
  --lr 9e-5 --optimizer lion \
  --beta1 0.9 --beta2 0.99 --weight_decay 5e-4 \
  --batch_size 4 --epochs 10 --lr_warmup_epochs 1 --lr_schedule cosine \
  --mirror_aug_p 0.25 --tau_y 3.0 --tau_z 2.0 \
  --surface_loss_weight 2.0 --volume_loss_weight 0.5 --ema_decay 0.999 \
  --wandb_group H221-lion-adamw-switch

# Phase 2: AdamW continuation EP10→EP13
torchrun --nproc_per_node=8 train.py \
  --resume_from_checkpoint <phase1>/epoch-10 \
  --lr 5e-6 --optimizer adamw \
  --beta1 0.9 --beta2 0.999 --weight_decay 1e-4 \
  --batch_size 4 --epochs 13 --epochs_already_done 10 \
  --lr_warmup_epochs 0 --lr_schedule cosine \
  --mirror_aug_p 0.25 --tau_y 3.0 --tau_z 2.0 \
  --surface_loss_weight 2.0 --volume_loss_weight 0.5 --ema_decay 0.999 \
  --wandb_group H221-lion-adamw-switch
```
Eval with TTA at EP13.

**Expected runtime**: ~3.5h
**Success criterion**: val_abupt < 5.9755%; WSS_x slope negative at EP13

---

## H222 — tau_x Explicit Channel Upweighting (WSS_x targeted loss)

**Student**: thorfinn (secondary, after H219 completes)
**Type**: Training-heavy (~3.5h)
**Rank**: 8

**Rationale**: WSS_x is the single channel where TTA provides no slope recovery (Finding Q). The basin disruption is a training-trajectory property. If the training loss underweights WSS_x relative to WSS_y and WSS_z, the optimizer will naturally find basins that sacrifice WSS_x precision. Explicitly upweighting tau_x in the per-channel loss — if a `--tau_x` or `--wss_x_weight` CLI flag exists — directly increases the gradient signal on the axis that fails. If the flag does not exist, student should add a 5-line patch to the loss weighting dict before training. This is the only known mechanism that could address WSS_x basin disruption at training time rather than at inference time.

**Recipe**:
```
# Check if flag exists:
python train.py --help | grep -i "tau_x\|wss_x\|channel_weight"

# If flag exists:
torchrun --nproc_per_node=8 train.py \
  --lr 9e-5 --optimizer lion \
  --beta1 0.9 --beta2 0.99 --weight_decay 5e-4 \
  --batch_size 4 --epochs 13 --lr_warmup_epochs 1 --lr_schedule cosine \
  --mirror_aug_p 0.25 --tau_y 3.0 --tau_z 2.0 \
  --tau_x 4.0 \
  --surface_loss_weight 2.0 --volume_loss_weight 0.5 --ema_decay 0.999 \
  --wandb_group H222-tauxup

# If flag absent, add to loss dict in model/loss.py:
#   channel_weights = {'wss_x': 2.0, 'wss_y': 1.0, 'wss_z': 1.0}
# then run standard H185 recipe with this patch.
```
Report WSS_x, WSS_y, WSS_z slopes at EP6, EP9, EP13. Eval with TTA.

**Expected runtime**: ~3.5h
**Success criterion**: WSS_x slope negative and magnitude larger than H185 baseline; val_abupt < 5.9755% after TTA

---

## Priority Assignment Map

| Student | Primary Assignment | Fallback if primary fast |
|---|---|---|
| alphonse | H215 (ultra-low-LR continuation, 1.5h) | H220 (artifact scan, 15 min, then advise) |
| fern | H216 (tau_y=2.0 retrain, 3.5h) | H221 after result |
| frieren | H217 (EP15 extension, 2h) | H221 or H222 after result |
| nezuko | H218 (p=0.5 + tau_y=2.0, 4h) | — |
| thorfinn | H219 (fresh seed + within-recipe LERP, 4.5h) | H222 after result |

**TTA reminder**: All training experiments must eval with `eval_tta_h209.py` at the best EMA checkpoint. Raw val and TTA val both reported.

**WSS_x slope diagnostic**: All training experiments must report WSS_x slope sign at EP6, EP9, EP13 as a basin health indicator. A positive (flipped) slope at EP13 is a failure signal even if val_abupt looks acceptable.
