# thorfinn round7 — 768d AdamW+RFF+compile (width scaling on stable base)

## Hypothesis

Scale width to 768d using the stable AdamW+RFF+compile base.
PR #69 (768d Lion uncompiled) was budget-limited to 5 epochs.
With compile, ~9-10 epochs of 768d should fit in 270 min.
Orthogonal to frieren PR #73 (6L depth).

## Reproduce Command

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --agent thorfinn \
  --wandb-name "thorfinn/round7-adamw-rff-768d-compile-rank0" \
  --wandb-group "thorfinn-round7-768d-adamw-compile" \
  --optimizer adamw \
  --lr 1e-4 --weight-decay 1e-4 \
  --rff-num-features 32 --rff-sigma 1.0 \
  --compile-model \
  --volume-loss-weight 2.0 --batch-size 4 --validation-every 1 \
  --train-surface-points 65536 --eval-surface-points 65536 \
  --train-volume-points 65536 --eval-volume-points 65536 \
  --model-layers 4 --model-hidden-dim 768 --model-heads 12 --model-slices 128 \
  --ema-decay 0.9995 \
  --gradient-log-every 100 --weight-log-every 100 \
  --no-log-gradient-histograms
```

## OOM Fallback

If OOM, reduce points to 32768+32768.
