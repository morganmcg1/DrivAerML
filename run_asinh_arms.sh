#!/bin/bash
# Sequential 2-arm sweep for PR #595 asinh wall-shear target-norm.
set -u
cd /workspace/senpai/target
mkdir -p train_logs

export SENPAI_TIMEOUT_MINUTES=${SENPAI_TIMEOUT_MINUTES:-360}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

COMMON=(
  --learnable-pe
  --optimizer lion
  --lr 1e-4
  --clip-grad-norm 0.5
  --weight-decay 5e-4
  --lr-warmup-epochs 1
  --ema-decay 0.999
  --model-layers 4
  --model-hidden-dim 512
  --model-heads 8
  --model-slices 128
  --batch-size 4
  --train-surface-points 65536 --eval-surface-points 65536
  --train-volume-points 65536 --eval-volume-points 65536
  --epochs 50
  --validation-every 1
  --no-compile-model
  --agent alphonse
  --wandb-group yi-round34-asinh-ws-target-norm
)

run_arm() {
  local arm_name="$1"
  local scale="$2"
  local logfile="train_logs/${arm_name}.log"
  local donefile="train_logs/${arm_name}.done"
  echo "[$(date -Is)] launching ${arm_name} scale=${scale}"
  torchrun --standalone --nproc_per_node=4 train.py \
    "${COMMON[@]}" \
    --ws-asinh-scale "${scale}" \
    --wandb-name "alphonse/asinh-target-${arm_name}" \
    > "${logfile}" 2>&1
  local rc=$?
  echo "[$(date -Is)] ${arm_name} exited rc=${rc}" | tee -a "${donefile}"
  return $rc
}

run_arm "armA-scale0p1" 0.1
echo "[$(date -Is)] arm A done — Arm B cancelled per advisor decision (saturation kill-gate hit, see PR #595 comment 16:16 UTC)"
# run_arm "armB-scale0p01" 0.01  # cancelled
echo "[$(date -Is)] sweep complete (Arm A only)" | tee -a train_logs/asinh_sweep_done
