#!/bin/bash
# H242 weight-noise TTA σ sweep over {1e-5, 5e-5, 1e-4, 5e-4} on H185 EP13 EMA.
set -e
cd "$(dirname "$0")"

LOGDIR="logs/h242"
mkdir -p "$LOGDIR"

CKPT="runs/h210/artifacts/h185/checkpoint.pt"

for SIGMA in 1e-5 5e-5 1e-4 5e-4; do
  echo "=== H242 σ=${SIGMA} starting at $(date -Is) ==="
  torchrun --standalone --nproc-per-node=8 eval_tta_h242.py \
    --checkpoint "$CKPT" \
    --weight-noise-sigma "$SIGMA" \
    --weight-noise-passes 5 \
    --wandb-group h242-tanjiro-weight-noise-tta \
    --wandb-name "tanjiro/h242-sigma-${SIGMA}" \
    2>&1 | tee "$LOGDIR/sigma_${SIGMA}.log"
  echo "=== H242 σ=${SIGMA} done at $(date -Is) ==="
done
echo "H242 sweep complete at $(date -Is)"
