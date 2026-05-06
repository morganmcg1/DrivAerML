#!/bin/bash
set -u
cd /workspace/senpai/target

COMMON_ARGS="--agent nezuko --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0 \
  --ema-decay 0.999 --grad-clip-norm 0.5 --lr-warmup-epochs 1 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --lr-cosine-t-max 13 --epochs 13 \
  --vol-points-schedule 0:16384:3:32768:6:49152:9:65536 \
  --no-compile-model \
  --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --use-surface-anchor --surface-anchor-chunk-size 4096 \
  --kill-thresholds 21729:val_primary/abupt_axis_mean_rel_l2_pct<32 \
  --wandb-group surf-anchor-learnable-scale-tay"

echo "===== Arm A: shared learnable affine anchor (alpha, beta scalar) ====="
date -u +%FT%TZ
torchrun --standalone --nproc_per_node=8 train.py $COMMON_ARGS \
  --wandb-name nezuko/anchor-arm-a \
  > logs/pr775/arm_a.log 2>&1
echo "Arm A exit: $?"
date -u +%FT%TZ

echo "===== Arm B: same architecture, second seed (verify alpha convergence) ====="
date -u +%FT%TZ
torchrun --standalone --nproc_per_node=8 train.py $COMMON_ARGS \
  --wandb-name nezuko/anchor-arm-b \
  > logs/pr775/arm_b.log 2>&1
echo "Arm B exit: $?"
date -u +%FT%TZ

echo "===== ALL PR775 ARMS DONE ====="
date -u +%FT%TZ
