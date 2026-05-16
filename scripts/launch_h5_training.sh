#!/usr/bin/env bash
# Launch WSS H5 curvature attention bias training (PR #1132).
#
# This is the exact run command from the PR body. It expects:
#   - /workspace/senpai/target/curvature_proxy_stats_k16_v1.json present
#   - per-case surface_curvature_proxy_k16_v1.npy in every case dir
#   - 8 GPUs available
#
# Usage:  bash scripts/launch_h5_training.sh
#         (logfile printed at start; PID written to logs/<runname>.pid)
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f curvature_proxy_stats_k16_v1.json ]]; then
  echo "ERROR: curvature_proxy_stats_k16_v1.json missing — wait for precompute" >&2
  exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGFILE="logs/h5_curvature_${STAMP}.log"
PIDFILE="logs/h5_curvature_${STAMP}.pid"
mkdir -p logs

export WANDB_PROJECT="${WANDB_PROJECT:-senpai-v1-drivaerml-ddp8}"

nohup torchrun --standalone --nproc_per_node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --lr 1e-4 --weight-decay 0.005 --batch-size 1 --epochs 30 \
  --train-surface-points 65000 --eval-surface-points 65536 \
  --train-volume-points 65000 --eval-volume-points 65536 \
  --model-layers 6 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --model-pe string_multisigma --pe-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --optimizer lion --use-gradnorm --gradnorm-alpha 0.5 \
  --ema-decay 0.999 --ema-start-step 5000 \
  --use-y-symmetry-aug --y-symmetry-aug-prob 0.5 \
  --lr-cosine-t-max 30 --lr-min 1e-6 --lr-warmup-epochs 1 \
  --amp-mode bf16 --seed 42 \
  --use-curvature-attention-bias \
  --wandb-group h5-curvature-attention-bias \
  --wandb-name "dl24-tanjiro/wss-curvature-attn-bias" \
  --agent dl24-tanjiro \
  > "${LOGFILE}" 2>&1 &

echo "$!" > "${PIDFILE}"
echo "launched torchrun PID=$(cat "${PIDFILE}")"
echo "logfile: ${LOGFILE}"
echo "pidfile: ${PIDFILE}"
