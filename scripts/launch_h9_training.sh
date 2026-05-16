#!/usr/bin/env bash
# Launch WSS H9: H5 curvature attention bias + GradNorm w_vol_p ≥ 0.05 floor.
# Matches PR #1145 command exactly.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f curvature_proxy_stats_k16_v1.json ]]; then
  echo "ERROR: curvature_proxy_stats_k16_v1.json missing" >&2
  exit 2
fi

# Abort if an H9 torchrun is already running on this pod — prevents the
# duplicate-launch GPU contention we saw on 2026-05-16 08:18Z.
if pgrep -f "torchrun.*train.py.*outputs/h9_curvature_bias_gradnorm_clamp" >/dev/null; then
  echo "ERROR: H9 torchrun already running on this pod; aborting." >&2
  pgrep -af "torchrun.*train.py.*outputs/h9_curvature_bias_gradnorm_clamp" >&2 || true
  exit 3
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGFILE="runlogs/h9_clamp_${STAMP}.log"
PIDFILE="runlogs/h9_clamp_${STAMP}.pid"
mkdir -p runlogs

export WANDB_PROJECT="${WANDB_PROJECT:-senpai-v1-drivaerml-ddp8}"

nohup torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --output-dir outputs/h9_curvature_bias_gradnorm_clamp \
  --epochs 30 --batch-size 1 \
  --train-surface-points 65000 --eval-surface-points 65536 \
  --train-volume-points 65000 --eval-volume-points 65536 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --lr 1e-4 --weight-decay 0.005 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 30 --lr-min 1e-6 \
  --use-ema --ema-decay 0.999 --ema-start-step 500 \
  --no-compile-model \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --use-gradnorm --gradnorm-alpha 0.5 \
  --gradnorm-min-w-vol-p 0.05 \
  --use-curvature-attention-bias \
  --surface-loss-weight 1.0 \
  --volume-loss-weight 1.0 \
  --amp-mode bf16 --seed 42 \
  --wandb-group wss_h9_curvature_bias_gradnorm_clamp \
  --wandb-name "dl24-tanjiro/h9-curv-bias-vol-p-clamp-0p05" \
  --agent dl24-tanjiro \
  > "$LOGFILE" 2>&1 &

echo "$!" > "$PIDFILE"
echo "$LOGFILE" > runlogs/CURRENT_LOG_H9
echo "launched torchrun PID=$(cat "${PIDFILE}")"
echo "logfile: ${LOGFILE}"
echo "pidfile: ${PIDFILE}"
