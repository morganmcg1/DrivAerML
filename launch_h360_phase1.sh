#!/usr/bin/env bash
# H360: launch Phase 1 LapPE-32 cosine-tail from H342 ep13 (W&B yw2a5dyl).
# Mirrors the H348 launch recipe (Lion lr=9e-5, tau_y=1.3, tau_z=1.67,
# surface_loss=2.0, mirror_aug, drop_path_max=0.1, 65k surface+vol points,
# batch_size=4 per GPU x 8 GPUs) with --lap-pe --lap-pe-channels 32 swapped
# in for --curvature-mode.

set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/h360_phase1_${TS}.log"
PIDFILE="logs/h360_phase1.pid"

mkdir -p logs
echo "Phase 1 log: $LOG"

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
  --no-compile-model \
  --lap-pe --lap-pe-channels 32 \
  --agent fern \
  --wandb-group "h360-fern-lappe-32" \
  --wandb-name "fern/h360-phase1-lappe32" \
  > "$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
echo "PID: $(cat "$PIDFILE")"
echo "Tail: tail -f $LOG"
