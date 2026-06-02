#!/usr/bin/env bash
# H356: Wait for ep15 K=5 eval to finish, then launch ep14 K=5 eval.
# Designed to be nohup'd so the chain survives shell exits.
set -euo pipefail
cd "$(dirname "$0")/.."

ep15_pid=$(cat logs/h356/ep15_k5_raw.pid)
echo "[$(date -u +%FT%TZ)] chain ep14 waiting on ep15 PID=$ep15_pid"
while kill -0 "$ep15_pid" 2>/dev/null; do
  sleep 60
done
echo "[$(date -u +%FT%TZ)] ep15 PID=$ep15_pid exited"

if [ ! -f outputs/h356/ep15_k5_raw.npz ]; then
  echo "[$(date -u +%FT%TZ)] ERROR: outputs/h356/ep15_k5_raw.npz missing; not launching ep14"
  exit 1
fi
echo "[$(date -u +%FT%TZ)] ep15 raw .npz present, launching ep14"

tools/h356_launch_eval.sh \
  outputs/drivaerml/run-0gjfv45i/checkpoint_ep14.pt \
  outputs/h356/ep14_k5_raw.npz \
  "alphonse/h356-ep14-k5-tta"

# After ep14 launch, also chain ep13 to start after ep14 terminates.
nohup tools/h356_chain_ep13.sh > logs/h356/chain_ep13.log 2>&1 &
echo "[$(date -u +%FT%TZ)] launched chain_ep13 PID=$!"
