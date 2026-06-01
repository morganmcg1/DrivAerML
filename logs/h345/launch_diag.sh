#!/bin/bash
# H345 Step 1: 1-epoch diagnostic from H185 ep14 cosine-tail base.
# Measures cos(g_wss, g_pres) per-step without modifying optimization.
# Gate: mean cos > -0.05 across the epoch -> falsify hypothesis, close as
#       `pcgrad-no-conflict-falsifies`. mean cos <= -0.05 -> proceed to Arm A.

set -o pipefail
cd /workspace/senpai/target
LOG_DIR=logs/h345
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/diag_ep14.log"

DIAG_ARGS=(
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511
  --manifest data/split_manifest.json
  --agent frieren
  --epochs 14 --batch-size 4
  --epochs-already-done 13
  --resume-from-wandb yw2a5dyl --resume-alias epoch-13
  --seed 2025
  --model-hidden-dim 512 --model-layers 5 --model-heads 4 --model-mlp-ratio 4
  --model-slices 128 --model-dropout 0.0
  --train-surface-points 65536 --eval-surface-points 65536
  --train-volume-points 65536 --eval-volume-points 65536
  --use-qk-norm --rff-num-features 16
  --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0"
  --pos-encoding-mode string_separable
  --use-ema --ema-decay 0.999 --ema-start-step 50
  --use-surf-to-vol-xattn --drop-path-max 0.10
  --lr 9e-5 --lr-warmup-epochs 0 --lr-cosine-t-max 16 --lr-min 1e-6
  --weight-decay 5e-4 --grad-clip-norm 0.5
  --optimizer lion --lion-beta1 0.9 --lion-beta2 0.99
  --amp-mode bf16 --no-compile-model
  --surface-loss-weight 2.0 --volume-loss-weight 0.5
  --tau-y-loss-weight 1.30 --tau-z-loss-weight 1.67
  --mirror-augmentation
  --diag-grad-conflict
  --wandb-group "h345-frieren-pcgrad"
  --wandb-name "frieren/h345-diag-conflict-ep14"
)

SENPAI_TIMEOUT_MINUTES=180 timeout --signal=TERM --kill-after=120s 150m \
  torchrun --standalone --nproc-per-node=8 train.py "${DIAG_ARGS[@]}" \
  > "$LOG_FILE" 2>&1
echo "diag exit=$?"
pkill -9 -f "train.py.*h345-diag-conflict-ep14" 2>/dev/null || true
