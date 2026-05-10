#!/bin/bash
set -euo pipefail
cd /workspace/senpai/target

torchrun --nproc_per_node=8 train.py \
  --model-layers 6 \
  --model-hidden-dim 512 \
  --model-heads 4 \
  --model-slices 128 \
  --model-pe string_multisigma \
  --pe-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
  --no-use-gradnorm \
  --use-y-symmetry-aug \
  --y-symmetry-aug-prob 0.5 \
  --weight-decay 0.005 \
  --volume-loss-weight 3.0 \
  --train-volume-points 65000 \
  --train-surface-points 40000 \
  --lr-warmup-steps 500 \
  --lr 1e-4 \
  --lr-cosine-t-max 50 \
  --optimizer lion \
  --epochs 50 \
  --use-ema \
  --ema-decay 0.999 \
  --no-compile-model \
  --agent dl24-frieren \
  --wandb-group vol-p-targeted-loss-weighting \
  --wandb-name 6l-str5oct-wd0.005-vol3x-nogradnorm \
  --kill-thresholds "27469:val_primary/abupt_axis_mean_rel_l2_pct<=8.0,54938:val_primary/abupt_axis_mean_rel_l2_pct<=7.5,82407:val_primary/abupt_axis_mean_rel_l2_pct<=7.0,109876:val_primary/abupt_axis_mean_rel_l2_pct<=6.80,137345:val_primary/abupt_axis_mean_rel_l2_pct<=6.70,164814:val_primary/abupt_axis_mean_rel_l2_pct<=6.65,192283:val_primary/abupt_axis_mean_rel_l2_pct<=6.60,219752:val_primary/abupt_axis_mean_rel_l2_pct<=6.58"
