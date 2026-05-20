#!/usr/bin/env bash
set -euo pipefail
cd /workspace/senpai/target
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="runlogs/h23_${TS}.log"
PIDF="runlogs/h23_${TS}.pid"
mkdir -p runlogs
echo "$LOG" > runlogs/CURRENT_LOG_H23
nohup torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/h23_h19_charb_tau_y \
  --epochs 30 --batch-size 1 \
  --train-surface-points 65000 --eval-surface-points 65536 \
  --train-volume-points 65000 --eval-volume-points 65536 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 \
  --gradnorm-min-w-vol-p 0.05 \
  --use-curvature-attention-bias \
  --wss-charbonnier-weight 0.1 --wss-charbonnier-eps 1e-3 --wss-charbonnier-axes y,z \
  --vol-p-charbonnier-weight 0.1 --vol-p-charbonnier-eps 1e-3 \
  --surface-loss-weight 1.0 --volume-loss-weight 1.0 \
  --wandb-group wss_h23_h19_charb_tau_y \
  --wandb-name dl24-tanjiro/h23-h19-charb-tau-y \
  --agent dl24-tanjiro \
  > "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "PID: $(cat $PIDF)"
echo "LOG: $LOG"
