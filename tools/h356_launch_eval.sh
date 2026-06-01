#!/usr/bin/env bash
# H356: Launch a single TTA eval mirroring the H342 SOTA recipe, upgraded to K=5
# (matching H336's K=5 axis). Usage:
#   tools/h356_launch_eval.sh <ckpt_path> <raw_npz_path> <wandb_name>
# Background launch with logging to logs/h356/<basename>.log and pidfile.
set -euo pipefail
ckpt=${1:?ckpt path}
raw=${2:?raw .npz path}
wname=${3:?wandb name}

mkdir -p outputs/h356 logs/h356
log="logs/h356/$(basename "${raw%.npz}")_eval.log"
pidfile="logs/h356/$(basename "${raw%.npz}").pid"

nohup torchrun --standalone --nproc-per-node=8 eval_tta_h252.py \
  --checkpoint "$ckpt" \
  --resolutions "32768,40960,49152,57344,65536,81920,98304,131072" \
  --eval-modes "weight_noise_mirror_res_avg" \
  --weight-noise-sigma 5e-4 --weight-noise-passes 5 --antithetic-noise \
  --weight-noise-dist student_t --weight-noise-df 4 \
  --test-time-calibration \
  --batch-size 2 --num-workers 4 \
  --save-raw-predictions "$raw" \
  --agent alphonse --wandb-group "h356-alphonse-3cp-k5" \
  --wandb-name "$wname" \
  > "$log" 2>&1 &
pid=$!
echo "$pid" > "$pidfile"
echo "Launched torchrun PID=$pid log=$log pidfile=$pidfile"
