#!/bin/bash
# H123 PR #1299: WSS tangent-frame projection (Option A — physical-space).
# H112 DropPath SOTA recipe + --use-wss-tangent-projection.
# Stale recipe flags in the PR body translated to actual train.py CLI:
#   --use-drop-path --drop-path-rate 0.10 -> --drop-path-max 0.10
#   --grad-clip 1.0                       -> --grad-clip-norm 1.0
#   --lr-schedule cosine                  -> implicit via --lr-cosine-t-max 13
#   --save-best-checkpoint                -> default on
#   --wandb-run-name X                    -> --wandb-name X
set -euo pipefail
cd /workspace/senpai/target

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/h123_${TS}.log"
PIDFILE="logs/h123_${TS}.pid"

SENPAI_TIMEOUT_MINUTES=1100 \
nohup torchrun --standalone --nproc_per_node=8 --master_port=29500 train.py \
  --agent fern \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --manifest data/split_manifest.json \
  --epochs 13 --batch-size 4 \
  --model-hidden-dim 512 --model-layers 5 --model-heads 4 --model-mlp-ratio 4 \
  --model-slices 128 --model-dropout 0.0 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 16384 --eval-volume-points 65536 \
  --vol-points-schedule '0:16384:3:32768:6:49152:9:65536' \
  --surface-loss-weight 2.0 --volume-loss-weight 0.5 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 \
  --use-qk-norm \
  --rff-num-features 16 \
  --rff-init-sigmas '0.25,0.5,1.0,2.0,4.0' \
  --pos-encoding-mode string_separable \
  --use-ema --ema-decay 0.999 --ema-start-step 50 \
  --use-surf-to-vol-xattn \
  --drop-path-max 0.10 \
  --use-wss-tangent-projection \
  --lr 9e-5 --lr-warmup-epochs 1 --lr-cosine-t-max 13 --lr-min 1e-6 \
  --weight-decay 5e-4 --grad-clip-norm 1.0 \
  --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99 \
  --amp-mode bf16 --no-compile-model \
  --kill-thresholds '10864:val_primary/abupt_axis_mean_rel_l2_pct<35.0,32594:val_primary/abupt_axis_mean_rel_l2_pct<8.5,65228:val_primary/abupt_axis_mean_rel_l2_pct<6.4' \
  --wandb-name 'fern/h123-wss-tangent-frame-projection' \
  --wandb-group 'wave36_h123_wss_tangent_projection' \
  > "$LOG" 2>&1 &

LAUNCH_PID=$!
echo "$LAUNCH_PID" > "$PIDFILE"
echo "H123 launch PID: $LAUNCH_PID"
echo "Log: $LOG"
echo "PID file: $PIDFILE"
