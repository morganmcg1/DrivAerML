#!/bin/bash
# H69 CURVATURE-ATTENTION-BIAS — PR #1223
# Wave 31 LR-fix substrate (mirrors H66 askeladd config) + curvature attention bias single-flag delta
set -e
cd /workspace/senpai/target
export SENPAI_TIMEOUT_MINUTES=1080
exec torchrun --standalone --nproc-per-node=8 train.py \
    --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
    --optimizer lion --lr 9e-5 --weight-decay 5e-4 \
    --lion-beta1 0.9 --lion-beta2 0.99 \
    --pos-encoding-mode string_separable --use-qk-norm \
    --rff-num-features 16 --rff-init-sigmas 0.25,0.5,1.0,2.0,4.0 \
    --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-slices 128 --model-mlp-ratio 4 \
    --use-surf-to-vol-xattn --use-ema --ema-decay 0.999 --ema-start-step 50 \
    --grad-clip-norm 0.5 --lr-warmup-epochs 1 --lr-cosine-t-max 25 --epochs 13 \
    --batch-size 4 \
    --train-surface-points 65536 --eval-surface-points 65536 \
    --train-volume-points 65536 --eval-volume-points 65536 \
    --vol-points-schedule "0:16384:3:32768:6:49152:9:65536" \
    --tau-y-loss-weight 1.5 --tau-z-loss-weight 2.0 --surface-loss-weight 2.0 \
    --volume-loss-weight 1.0 \
    --use-curvature-attn-bias --curvature-attn-init 0.0 \
    --no-compile-model --validation-every 1 --amp-mode bf16 \
    --wandb-group wave32_h69_curvature_attention_bias \
    --wandb-name fern/h69-curvature-attention-bias --agent fern \
    --kill-thresholds "32592:val_primary/abupt_axis_mean_rel_l2_pct<7.5,32592:val_primary/surface_pressure_rel_l2_pct<5.5,65184:val_primary/abupt_axis_mean_rel_l2_pct<6.5"
