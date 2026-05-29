#!/usr/bin/env bash
# H235: run eval_tta_h209.py once per mirror-trained EP13 EMA checkpoint.
# Sequential — every job uses all 8 GPUs for DDP-strided eval.
set -euo pipefail

ART=/workspace/senpai/target/artifacts
OUT=/workspace/senpai/target/outputs/h235_eval
LOG=/workspace/senpai/target/outputs/h235_eval/sweep.log
mkdir -p "$OUT"

# label run_id drop_path_max
ROWS=(
  "H185 yw2a5dyl 0.1"
  "H183 5k58uzqc 0.1"
  "H190 9f2jtrg2 0.1"
  "H188 18t5rx2t 0.15"
  "H148 2qr5guel 0.1"
  "H191 5y5a5tgr 0.1"
  "H181b w7w92npw 0.1"
  "H186 d15dm825 0.1"
)

declare -A ARTDIR
ARTDIR[yw2a5dyl]="model-thorfinn-h185-h171-mirror-aug-compound-yw2a5dyl:v0"
ARTDIR[5k58uzqc]="model-fern-h183-mirror-aug-tau-y-3p0-compound-5k58uzqc:v0"
ARTDIR[9f2jtrg2]="model-nezuko-h190-mirror-aug-p025-9f2jtrg2:v0"
ARTDIR[18t5rx2t]="model-alphonse-h188-mirror-droppath-015-18t5rx2t:v0"
ARTDIR[2qr5guel]="model-askeladd-h148-mirror-aug-2qr5guel:v0"
ARTDIR[5y5a5tgr]="model-tanjiro-h191-mirror-tau-y-2p0-5y5a5tgr:v0"
ARTDIR[w7w92npw]="model-askeladd-h181b-mirror-aug-ema-9999-w7w92npw:v0"
ARTDIR[d15dm825]="model-fern-h186-mirror-aug-adamw-cross-axis-d15dm825:v0"

cd /workspace/senpai/target

for ROW in "${ROWS[@]}"; do
  read -r LABEL RUNID DPM <<<"$ROW"
  CKPT="$ART/${ARTDIR[$RUNID]}/checkpoint.pt"
  RUN_OUT="$OUT/$RUNID"
  mkdir -p "$RUN_OUT"
  echo
  echo "=================================================="
  echo "=== ${LABEL} run=${RUNID} dpm=${DPM}"
  echo "  ckpt: ${CKPT}"
  echo "  out:  ${RUN_OUT}"
  echo "=================================================="
  if ! ls "$CKPT" >/dev/null 2>&1; then
    echo "!! missing checkpoint, skipping"
    continue
  fi
  T0=$(date +%s)
  torchrun --standalone --nproc-per-node=8 eval_tta_h209.py \
    --checkpoint "$CKPT" \
    --output-dir "$RUN_OUT" \
    --wandb-group h235-frieren-finding-q-extension \
    --wandb-name "frieren/h235-tta-${LABEL}-${RUNID}" \
    --batch-size 2 \
    --drop-path-max "$DPM" \
    >"$RUN_OUT/eval.log" 2>&1
  T1=$(date +%s)
  echo "  done in $((T1 - T0))s"
done

echo
echo "All sweep runs finished."
