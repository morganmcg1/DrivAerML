#!/usr/bin/env bash
# H356: Wait for ep14 K=5 eval to finish, then launch ep13 K=5 eval.
set -euo pipefail
cd "$(dirname "$0")/.."

ep14_pid=$(cat logs/h356/ep14_k5_raw.pid)
echo "[$(date -u +%FT%TZ)] chain ep13 waiting on ep14 PID=$ep14_pid"
while kill -0 "$ep14_pid" 2>/dev/null; do
  sleep 60
done
echo "[$(date -u +%FT%TZ)] ep14 PID=$ep14_pid exited"

if [ ! -f outputs/h356/ep14_k5_raw.npz ]; then
  echo "[$(date -u +%FT%TZ)] ERROR: outputs/h356/ep14_k5_raw.npz missing; not launching ep13"
  exit 1
fi
echo "[$(date -u +%FT%TZ)] ep14 raw .npz present, launching ep13"

exec tools/h356_launch_eval.sh \
  outputs/drivaerml/run-yw2a5dyl/checkpoint_ep13.pt \
  outputs/h356/ep13_k5_raw.npz \
  "alphonse/h356-ep13-k5-tta"
