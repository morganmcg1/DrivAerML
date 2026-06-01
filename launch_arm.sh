#!/usr/bin/env bash
# Launch a single H348 curvature arm (H, K, or k1k2) using the H336 recipe.
#
# Usage: ./launch_arm.sh <arm_letter> <curvature_mode>
#   e.g. ./launch_arm.sh A H
#        ./launch_arm.sh B K
#        ./launch_arm.sh C k1k2

set -euo pipefail

ARM="${1:?arm letter (A/B/C) required}"
MODE="${2:?curvature mode (H/K/k1k2) required}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/h348_arm_${ARM}_${TS}.log"
PIDFILE="logs/arm_${ARM}.pid"

mkdir -p logs
echo "Arm $ARM ($MODE) log: $LOG"

nohup torchrun --standalone --nproc-per-node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --resume-from-wandb yw2a5dyl --resume-alias epoch-13 \
  --epochs-already-done 13 --epochs 16 --lr-cosine-t-max 16 \
  --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-mlp-ratio 4 --model-slices 128 \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --pos-encoding-mode string_separable \
  --use-qk-norm --use-surf-to-vol-xattn --drop-path-max 0.1 \
  --tau-y-loss-weight 1.3 --tau-z-loss-weight 1.67 \
  --mirror-augmentation \
  --batch-size 4 --train-surface-points 65536 --train-volume-points 65536 \
  --eval-surface-points 65536 --eval-volume-points 65536 \
  --surface-loss-weight 2.0 --volume-loss-weight 0.5 \
  --lr 9e-05 --weight-decay 0.0005 \
  --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --grad-clip-norm 0.5 --save-every-epoch \
  --vol-points-schedule "0:65536" \
  --curvature-mode "$MODE" \
  --no-compile-model \
  --agent fern \
  --wandb-group "h348-fern-curvature-features" \
  --wandb-name "fern/h348-arm-${ARM}-${MODE}" \
  > "$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
echo "PID: $(cat "$PIDFILE")"
echo "Tail: tail -f $LOG"
