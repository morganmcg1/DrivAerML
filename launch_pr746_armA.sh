#!/bin/bash
# PR #746 - Y-flip training augmentation - Arm A (--y-flip-prob 0.5)
set -uo pipefail
cd /workspace/senpai/target
mkdir -p logs_haku
LOG=logs_haku/pr746_armA_yflip_p050_$(date +%Y%m%d_%H%M%S).log
echo "Logging to: $LOG"
echo "Started at: $(date -u)" > "$LOG"
nohup torchrun --standalone --nproc_per_node=4 train.py \
  --agent haku \
  --wandb-group yi-round42-haku-yflip-aug \
  --wandb-name haku/yflip-aug-p050 \
  --learnable-pe --optimizer lion --lr 1e-4 --weight-decay 5e-4 --clip-grad-norm 0.5 \
  --grad-ema-alpha 0.5 \
  --surface-curvature-features k1_k2 \
  --beta-nll-beta 0.5 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --batch-size 4 --validation-every 2 --no-compile-model \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --y-flip-prob 0.5 \
  --ema-decay 0.999 --lr-warmup-epochs 1 \
  --kill-thresholds '5442:val_primary/abupt_axis_mean_rel_l2_pct<50,16326:val_primary/abupt_axis_mean_rel_l2_pct<12,54420:val_primary/abupt_axis_mean_rel_l2_pct<9,108840:val_primary/abupt_axis_mean_rel_l2_pct<8.0' \
  --epochs 30 \
  >> "$LOG" 2>&1 &
PID=$!
echo "Launched PID $PID"
echo "PID $PID" >> "$LOG"
echo "$PID" > logs_haku/pr746_armA.pid
echo "$LOG" > logs_haku/pr746_armA.logpath
