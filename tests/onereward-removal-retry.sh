#!/usr/bin/env bash
# OneReward object removal, best-of-N with a person check.
#
# Runs the removal recipe from the OneReward README (negative "nsfw", true CFG 4, guidance 1, 50 steps, removal LoRA)
# on consecutive seeds. A candidate passes when tests/person_check.py finds no person inside the mask. The first
# passing candidate is copied to <outdir>/chosen.png. Every verdict is appended to <outdir>/verdicts.txt.
# The gate is a detector, so the chosen image still needs a visual check against the prompt.
#
# Usage: tests/onereward-removal-retry.sh <cli.dll> <outdir> <init.png> <mask.png> <first_seed> <last_seed> [fp8|bf16]
# Environment:
#   CLI_CWD              working directory for the CLI (its Output/ folder receives the renders; default: $PWD)
#   PERSON_CHECK_PYTHON  Python with torch, ultralytics, safetensors, cv2 and numpy (default: python3)
#   PERSON_CHECK_WEIGHTS yolov8n-seg-folded.safetensors (required by person_check.py)
#   PERSON_MIN_COVER     coverage at which a person counts as inside the mask (default: 0.6)
#   NEGATIVE             negative prompt (default: nsfw, the README's recipe)
#   KEEP_GOING=1         run every seed instead of stopping at the first pass
set -uo pipefail

if [ "$#" -lt 6 ]; then
    sed -n '2,14p' "$0" >&2
    exit 2
fi
CLI=$1; OUT=$2; INIT=$3; MASK=$4; FIRST=$5; LAST=$6; PRECISION=${7:-fp8}
case "$PRECISION" in
    fp8) MODEL=onereward-fill-fp8 ;;
    bf16) MODEL=onereward-fill ;;
    *) echo "precision must be fp8 or bf16" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
CLI_CWD=${CLI_CWD:-$PWD}
PY=${PERSON_CHECK_PYTHON:-python3}
MIN_COVER=${PERSON_MIN_COVER:-0.6}

mkdir -p "$OUT"
: > "$OUT/verdicts.txt"
chosen=""
for seed in $(seq "$FIRST" "$LAST"); do
    (cd "$CLI_CWD" && dotnet "$CLI" image "remove" -m "$MODEL" -b cuda:0 \
        --init-image "$INIT" --mask "$MASK" --width 768 --height 1024 --seed "$seed" --steps 50 \
        --negative "${NEGATIVE:-nsfw}" --true-cfg 4 --flux-guidance 1 \
        --lora flux.1-fill-dev-object-removal-lora --lora-weight 1.0 > "$OUT/seed-$seed.log" 2>&1)
    rc=$?
    latest=$(ls -t "$CLI_CWD"/Output/*.png 2>/dev/null | head -1)
    if [ "$rc" -ne 0 ] || [ -z "$latest" ]; then
        echo "seed $seed error exit=$rc" | tee -a "$OUT/verdicts.txt"
        continue
    fi
    mv "$latest" "$OUT/seed-$seed.png"
    verdict=$("$PY" "$HERE/person_check.py" "$OUT/seed-$seed.png" "$MASK" "$MIN_COVER")
    fail=$(printf '%s' "$verdict" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["fail"])')
    echo "seed $seed fail=$fail $verdict" | tee -a "$OUT/verdicts.txt"
    if [ "$fail" = "False" ] && [ -z "$chosen" ]; then
        chosen=$seed
        cp "$OUT/seed-$seed.png" "$OUT/chosen.png"
        echo "chosen seed $chosen" >> "$OUT/verdicts.txt"
        [ "${KEEP_GOING:-0}" = "1" ] || break
    fi
done
if [ -n "$chosen" ]; then
    echo "chosen seed $chosen"
else
    echo "no candidate passed in seeds $FIRST..$LAST"
    exit 1
fi
