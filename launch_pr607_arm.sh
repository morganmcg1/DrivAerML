#!/bin/bash
# PR #607 GradNorm dynamic multi-task balancing — single arm launcher (4-GPU DDP).
# Usage: ./launch_pr607_arm.sh <arm_letter> <gradnorm_alpha>
#   armA: alpha=0.0  (control: gradient norm equalization)
#   armB: alpha=0.5  (mild rebalancing)
#   armC: alpha=1.0  (canonical GradNorm)
#   armD: alpha=1.5  (aggressive upweighting of lagging tasks)
#
# Architecture matches PR #517 / PR #580 winning stack: 4L/512d/8h/128 slices,
# Lion lr=1e-4 wd=5e-4 clip=0.5, 65536 surface+volume points, --learnable-pe,
# --surface-curvature-features k1_k2, EMA 0.999, lr-warmup 1 epoch, 50 epochs.
set -e

ARM="${1:?usage: $0 <arm_letter> <alpha>}"
ALPHA="${2:?usage: $0 <arm_letter> <alpha>}"

cd /workspace/senpai/target
mkdir -p logs_haku
LOG=/workspace/senpai/target/logs_haku/pr607_arm${ARM}_alpha${ALPHA}.log
PIDFILE=/workspace/senpai/target/logs_haku/pr607_arm${ARM}.pid

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun \
  --standalone --nnodes=1 --nproc_per_node=4 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29607 \
  train.py \
  --agent haku \
  --wandb-group haku-gradnorm-r34 \
  --wandb-name "haku/pr607-arm${ARM}-alpha${ALPHA}" \
  --learnable-pe \
  --surface-curvature-features k1_k2 \
  --optimizer lion --lr 1e-4 --clip-grad-norm 0.5 \
  --weight-decay 5e-4 --lr-warmup-epochs 1 --ema-decay 0.999 \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 --model-slices 128 \
  --batch-size 4 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --epochs 50 \
  --no-compile-model \
  --validation-every 1 \
  --gradnorm-alpha ${ALPHA} \
  --gradnorm-lr 1e-3 \
  --gradnorm-init-steps 10 \
  --seed 42 \
  > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
echo "Arm ${ARM} (alpha=${ALPHA}) launched, PID=$(cat ${PIDFILE}), log=${LOG}"
