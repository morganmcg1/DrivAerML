#!/usr/bin/env bash
# H360: TTA eval for LapPE-32 cosine-tail (Phase 1 run 9c5d2xwt EP16).
# Mirrors the H342/H348 eval recipe: 8-resolution sweep, weight-noise stacking
# (student-t df=4, sigma=5e-4, 4 passes, antithetic), per-channel OLS calibration.

set -euo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/h360_eval_${TS}.log"
PIDFILE="logs/h360_eval.pid"

mkdir -p logs
echo "Eval log: $LOG"

nohup torchrun --standalone --nproc-per-node=8 eval_tta_h252.py \
  --checkpoint outputs/drivaerml/run-9c5d2xwt/checkpoint_ep16.pt \
  --data-root /mnt/new-pvc/Processed/drivaerml_processed_rawcanon_20260511 \
  --resolutions "32768,40960,49152,57344,65536,81920,98304,131072" \
  --eval-modes "weight_noise_mirror_res_avg" \
  --weight-noise-sigma 5e-4 --weight-noise-passes 4 --antithetic-noise \
  --weight-noise-dist student_t --weight-noise-df 4 \
  --test-time-calibration \
  --lap-pe --lap-pe-channels 32 \
  --lap-pe-root /mnt/new-pvc/Processed/lap_pe_v1 \
  --model-layers 5 --model-hidden-dim 512 --model-heads 4 --model-mlp-ratio 4 --model-slices 128 \
  --rff-num-features 16 --rff-init-sigmas "0.25,0.5,1.0,2.0,4.0" \
  --pos-encoding-mode string_separable \
  --use-qk-norm --use-surf-to-vol-xattn --drop-path-max 0.1 \
  --eval-surface-points 65536 \
  --agent fern \
  --wandb-group "h360-fern-lappe-32-eval" \
  --wandb-name "fern/h360-eval-lappe32" \
  > "$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
echo "PID: $(cat "$PIDFILE")"
echo "Tail: tail -f $LOG"
