#!/bin/bash
# H212 mirror-invariance profile across 5 EP13 cohort checkpoints.
# Note: --data-root deviated from PR ("drivaerml_processed") to
#   "drivaerml_processed_rawcanon_20260511" because:
#   1) All 5 models were trained on rawcanon (per their config.yaml data_root).
#   2) H200 control (H189) was run on rawcanon — keeping the comparison
#      apples-to-apples vs the published delta_mirror = +4.532pp.
set -e
DATA_ROOT="/mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511"
LOGDIR="/workspace/senpai/target/logs/h212"
mkdir -p "$LOGDIR"

for model in h112 h148 h183 h185 h190; do
  echo "=== Running $model ==="
  python -m torch.distributed.run --standalone --nproc-per-node=8 tta_mirror_eval.py \
    --checkpoint "outputs/${model}/checkpoint.pt" \
    --config-yaml "outputs/${model}/config.yaml" \
    --data-root "$DATA_ROOT" \
    --manifest data/split_manifest.json \
    --batch-size 4 \
    --eval-surface-points 65536 \
    --eval-volume-points 65536 \
    --amp-mode bf16 \
    --splits test \
    --eval-modes original,mirrored \
    --wandb-group h212-edward-mirror-invariance-profile \
    --wandb-name "edward/h212-${model}-mirror-profile" \
    2>&1 | tee "$LOGDIR/${model}.log"
  echo "=== $model done ==="
done
