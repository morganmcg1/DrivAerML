#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SENPAI_TIMEOUT_MINUTES=5
export SENPAI_MAX_EPOCHS=1
export WANDB_ENTITY="${WANDB_ENTITY:-wandb-applied-ai-team}"
export WANDB_PROJECT="${WANDB_PROJECT:-senpai-v1-drivaerml-ddp8}"
export WANDB_MODE="${WANDB_MODE:-online}"

torchrun --standalone --nproc-per-node=8 train.py \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
  --pos-encoding-mode string_separable --use-qk-norm \
  --rff-num-features 16 --rff-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128 \
  --normal-aux-loss-weight 0.1 \
  --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0 \
  --use-surf-to-vol-xattn --use-ema --ema-decay 0.999 --grad-clip-norm 0.5 \
  --lr-warmup-epochs 1 --lr-cosine-t-max 13 --epochs 1 \
  --no-compile-model --batch-size 2 --validation-every 1 \
  --train-surface-points 4096 --eval-surface-points 4096 \
  --train-volume-points 4096 --eval-volume-points 4096 \
  --gradient-log-every 50 --weight-log-every 50 \
  --wandb-group wave30_normal_aux_head_smoke \
  --wandb-name askeladd/smoke-aux-w0.1 --agent askeladd
