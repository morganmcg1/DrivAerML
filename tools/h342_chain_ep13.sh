#!/usr/bin/env bash
# H342: Wait for ep14 eval to finish, then launch ep13 eval. Designed to be
# nohup'd so the chain survives shell exits.
set -euo pipefail
cd "$(dirname "$0")/.."

ep14_pid=$(cat logs/h342/ep14_raw.pid)
echo "[$(date -u +%FT%TZ)] chain waiting on ep14 PID=$ep14_pid"
# Poll, then bail out if ep14 raw .npz never gets written (eval failed).
while kill -0 "$ep14_pid" 2>/dev/null; do
  sleep 60
done
echo "[$(date -u +%FT%TZ)] ep14 PID=$ep14_pid exited"

if [ ! -f outputs/h342/ep14_raw.npz ]; then
  echo "[$(date -u +%FT%TZ)] ERROR: outputs/h342/ep14_raw.npz missing; not launching ep13"
  exit 1
fi
echo "[$(date -u +%FT%TZ)] ep14 raw .npz present, launching ep13"

exec tools/h342_launch_eval.sh \
  outputs/drivaerml/run-yw2a5dyl/checkpoint_ep13.pt \
  outputs/h342/ep13_raw.npz \
  "alphonse/h342-ep13-tta"
