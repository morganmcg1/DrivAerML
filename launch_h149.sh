#!/usr/bin/env bash
# H149: AdamW optimizer swap — test if slope-flattening is Lion-specific.
# Mirrors H112 (PR #1283, run u9ue2ryb) exactly except for optimizer+LR.
# Verified H112 config from W&B: volume_loss_weight=1.0, grad_clip_norm=0.5.
set -e
export SENPAI_TIMEOUT_MINUTES=1100
exec torchrun --standalone --nproc-per-node=8 train.py \
    --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
    --manifest data/split_manifest.json \
    --epochs 13 --batch-size 4 \
    --model-hidden-dim 512 --model-layers 5 --model-heads 4 --model-mlp-ratio 4 \
    --model-slices 128 --model-dropout 0.0 \
    --drop-path-max 0.10 \
    --train-surface-points 65536 --eval-surface-points 65536 \
    --train-volume-points 16384 --eval-volume-points 65536 \
    --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
    --surface-loss-weight 2.0 --volume-loss-weight 1.0 \
    --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 \
    --use-qk-norm --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" --pos-encoding-mode string_separable \
    --use-ema --ema-decay 0.999 --ema-start-step 50 \
    --use-surf-to-vol-xattn \
    --lr 3e-4 --lr-warmup-epochs 1 --lr-cosine-t-max 13 --lr-min 1e-6 \
    --weight-decay 5e-4 --grad-clip-norm 0.5 \
    --optimizer adamw \
    --amp-mode bf16 --no-compile-model \
    --kill-thresholds "10864:val_primary/abupt_axis_mean_rel_l2_pct<33.0,32592:val_primary/abupt_axis_mean_rel_l2_pct<10.0,48897:val_primary/abupt_axis_mean_rel_l2_pct<7.5" \
    --wandb-name tanjiro/h149-adamw \
    --wandb-group h149-adamw-optimizer \
    --agent tanjiro
