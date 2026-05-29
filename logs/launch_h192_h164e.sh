#!/bin/bash
set -e
cd /workspace/senpai/target
export WANDB_ENTITY=wandb-applied-ai-team
export WANDB_PROJECT=senpai-v1-drivaerml-ddp8
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
exec python -m torch.distributed.run --nproc-per-node=7 tta_mirror_eval.py \
  --checkpoint outputs/ensemble_cache/zrv3dasr/checkpoint.pt \
  --config-yaml outputs/ensemble_cache/zrv3dasr/config.yaml \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --manifest data/split_manifest.json \
  --batch-size 4 \
  --eval-surface-points 65536 \
  --eval-volume-points 65536 \
  --amp-mode bf16 \
  --splits val,test \
  --eval-modes original,mirrored,tta \
  --wandb-group thorfinn-h192-tta-mirror-aug \
  --wandb-name thorfinn/h192-h164e-tta-mirror
