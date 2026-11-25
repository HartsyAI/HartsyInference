#!/usr/bin/env bash
# OVB-4B segmentation over the 40-image COCO sample. One CLI call per image (each loads the model once).
set -u
ROOT="C:/Users/AI Overlord/Desktop/Projects/HartsyInference"
CLI="C:/Users/AI Overlord/Desktop/Worktrees/HartsyInference/open-vision-banana-9722a8/src/HartsyInference.Cli/bin/Debug/net10.0/HartsyInference.Cli.exe"
OUT="$ROOT/Datasets/coco2017/ovb-seg"
LOG="$ROOT/Datasets/coco2017/progress.log"
mkdir -p "$OUT"
PROMPT="person,car,bus,bicycle,dog"
for f in $(python -I -c "import json;print(' '.join(json.load(open('$ROOT/Datasets/coco2017/sample40.json'))['file_names']))"); do
  stem="${f%.jpg}"
  if [ -f "$OUT/$stem/done" ]; then continue; fi
  t0=$(date +%s)
  timeout 900 "$CLI" vision -m open-vision-banana-4b --mode segment -p "$PROMPT" -o "$OUT/$stem" \
    --set "paths.modelsRoot=C:\Users\AI Overlord\Desktop\Projects\HartsyInference\Models" --set paths.sideModelAutofetch=false \
    "$ROOT/Datasets/coco2017/images_png/${stem}.png" < /dev/null > "$OUT/$stem.log" 2>&1
  rc=$?
  n=$(find "$OUT/$stem" -name "*.png" 2>/dev/null | wc -l)
  echo "$(date +%H:%M:%S) $f rc=$rc masks=$n secs=$(( $(date +%s) - t0 ))" >> "$LOG"
  [ $rc -eq 0 ] && touch "$OUT/$stem/done"
done
echo "$(date +%H:%M:%S) ALL DONE" >> "$LOG"
