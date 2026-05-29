#!/bin/bash
set -e
cd /workspace/senpai/target
H185_CKPT="outputs/h228_eval/_artifacts/yw2a5dyl/epoch-13/checkpoint.pt"
ls -lh "$H185_CKPT"

echo "=== [$(date -u +%H:%M:%SZ)] H272 SMOKE TEST: small debug run ==="
torchrun --standalone --nproc-per-node=8 eval_h272_hutchinson_tta.py \
  --checkpoint "$H185_CKPT" \
  --resolutions "32768,49152" \
  --eval-modes "mirror_res_weight_noise_avg" \
  --weight-noise-sigma 5e-4 \
  --weight-noise-passes 2 \
  --hutchinson-m 2 \
  --hutchinson-beta 1e4 \
  --hutchinson-eval-surface-points 1024 \
  --hutchinson-cache "outputs/h272_diag_hessian_smoke.pt" \
  --batch-size 2 --num-workers 2 \
  --debug \
  --wandb-name "nezuko/h272-smoke-debug-v3" \
  --wandb-group "h272-nezuko-hutchinson-curvature"

echo "=== [$(date -u +%H:%M:%SZ)] DONE ==="
