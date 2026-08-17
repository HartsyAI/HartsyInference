#!/bin/bash
# LTX-2.5 DISTILLED two-stage in-engine benchmark (CLI, not SwarmUI). Same guards as ltx25_bench.sh (which
# stays the dev-arm baseline): wrong GPU and a squatting CUDA tenant have each invalidated full campaigns.
#
#   ./ltx25_distilled_bench.sh [LABEL] [SEED] [FRAMES] [WIDTH] [HEIGHT]
#
# Sampling flags are DELIBERATELY OMITTED from the invocation: the defaults are the thing under test — the
# run must show the filename remap firing (when driven via the dev id), the 8-step base ladder, the latent
# upsample, and the 3-step refine. MODE=single sets HARTSY_LTX2_TWO_STAGE=0 for the kill-switch arm.
# Geometry defaults to the template's 1280x736x121f; pass smaller values for quick turnarounds.
set -u

LABEL=${1:-run}
SEED=${2:-1}
FRAMES=${3:-121}
WIDTH=${4:-1280}
HEIGHT=${5:-736}

REPO=/home/hartsy/Desktop/HartsyInference
# Same private-copy convention as ltx25_bench.sh — a mid-campaign rebuild silently swaps the binary.
CLI=${LTX25_BENCH_CLI:-$REPO/src/HartsyInference.Cli/bin/Release/net10.0/HartsyInference.Cli.dll}
OUT=${LTX25_BENCH_OUT:-/tmp/ltx25_distilled_bench}
PROMPT=${LTX25_BENCH_PROMPT-"a lone lighthouse keeper walking along a rocky coastline at sunset, waves crashing, cinematic wide shot, volumetric light, highly detailed, 35mm film look"}
# The model id to drive: "ltx-2.5" exercises the filename remap (default); "ltx-2.5-distilled" pins the id.
MODEL_ID=${LTX25_BENCH_MODEL_ID:-ltx-2.5}
MODE=${MODE:-two-stage}

GPU_SMI=${LTX25_BENCH_GPU:-1}
GPU_CUDA=$GPU_SMI
EXPECT_GPU=${LTX25_BENCH_GPU_NAME:-"RTX 4090"}
MIN_FREE_MB=${LTX25_BENCH_MIN_FREE:-22000}

# Stage the split distilled checkpoint the way SwarmUI-shaped runs arrive: transformer-only distilled file
# plus sibling VAEs/Gemma in one directory whose contents (not name) carry the "distilled" marker.
STAGE=${LTX25_BENCH_STAGE:-$OUT/model_stage}
if [ ! -e "$STAGE/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" ]; then
    mkdir -p "$STAGE"
    ln -sf "$REPO/Models/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" "$STAGE/"
    ln -sf "$REPO/Models/Stable-Diffusion/LTX-2.5/gemma4-12b-with-proj-ltx-2.5-int8_lean_convrot.safetensors" "$STAGE/"
    ln -sf "$REPO/Models/Stable-Diffusion/LTX-2.5/ltx-2.5-video-vae-conv-bf16.safetensors" "$STAGE/"
    ln -sf "$REPO/Models/Stable-Diffusion/LTX-2.5/ltx-2.5-audio-vae-bf16.safetensors" "$STAGE/"
fi

mkdir -p "$OUT"
LOG=$OUT/${LABEL}_${MODE}_seed${SEED}.log

[ -f "$CLI" ] || { echo "FATAL: CLI not built: $CLI"; exit 1; }

SWARM_WAS_UP=0
systemctl --user is-active --quiet swarmui.service && SWARM_WAS_UP=1
restore() { [ "$SWARM_WAS_UP" = "1" ] && systemctl --user start swarmui.service >/dev/null 2>&1; }
trap restore EXIT INT TERM
if [ "$SWARM_WAS_UP" = "1" ]; then
    echo "stopping swarmui.service (holds several GB on the 4090)..."
    systemctl --user stop swarmui.service
    sleep 8
fi

NAME=$(nvidia-smi -i "$GPU_SMI" --query-gpu=name --format=csv,noheader)
case "$NAME" in *"$EXPECT_GPU"*) ;; *) echo "FATAL: GPU $GPU_SMI is '$NAME', expected '$EXPECT_GPU'"; exit 1;; esac
FREE=$(nvidia-smi -i "$GPU_SMI" --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "FATAL: only ${FREE} MB free on $NAME (need $MIN_FREE_MB) — another CUDA tenant is resident"; exit 1; }
echo "GPU $GPU_SMI = $NAME, ${FREE} MB free; mode=$MODE id=$MODEL_ID"

EXTRA_ENV=()
[ "$MODE" = "single" ] && EXTRA_ENV=(HARTSY_LTX2_TWO_STAGE=0)

# VRAM peak sampled alongside — the F32 upsampler adds a ~1.9 GB transient the single-pass arm never sees.
nvidia-smi -i "$GPU_SMI" --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/${LABEL}_${MODE}.vram" 2>&1 &
SMI_PID=$!

rm -rf "$OUT/frames_$LABEL"
START=$(date +%s.%N)
env "${EXTRA_ENV[@]}" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_CUDA HARTSY_LOG_LEVEL=Info \
    dotnet "$CLI" video "$PROMPT" -m "$MODEL_ID" --model-path "$STAGE" \
    --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES" --seed "$SEED" \
    -q -o "$OUT/frames_$LABEL" > "$LOG" 2>&1
RC=$?
END=$(date +%s.%N)
kill $SMI_PID 2>/dev/null
WALL=$(echo "$END - $START" | bc)
PEAK=$(sort -rn "$OUT/${LABEL}_${MODE}.vram" | head -1)

grep -E "distilled build — routing|Distilled variant selected|Two-stage enabled|ltx2-phase|ltx2-two-stage" "$LOG"
[ "$RC" -eq 0 ] || { echo "FATAL: generation failed (rc=$RC), see $LOG"; tail -20 "$LOG"; exit 1; }

if [ "$MODE" = "two-stage" ]; then
    grep -q "distilled build — routing" "$LOG" || [ "$MODEL_ID" = "ltx-2.5-distilled" ] \
        || { echo "FATAL: the filename remap did not fire under id $MODEL_ID"; exit 1; }
    grep -q "ltx2-two-stage" "$LOG" || { echo "FATAL: no upsample stage in the log — ran single-pass?"; exit 1; }
fi

awk -v wall="$WALL" -v peak="$PEAK" '
/ltx2-phase\] s1 step / { n1++; if (n1 > 1) { t=$0; gsub(/ms.*/, "", t); sub(/.*: /, "", t); s1 += t; c1++ } }
/ltx2-phase\] s2 step / { t=$0; gsub(/ms.*/, "", t); sub(/.*: /, "", t); s2 += t; c2++ }
/ltx2-two-stage\] latent upsample/ { t=$0; gsub(/ms.*/, "", t); sub(/.*: /, "", t); up = t }
END { if (c1) printf "\n=== s1 steps 2..%d mean: %.1f ms (n=%d) ===\n", n1, s1 / c1, c1;
      if (up) printf "=== latent upsample: %s ms ===\n", up;
      if (c2) printf "=== s2 steps mean: %.1f ms (n=%d) ===\n", s2 / c2, c2;
      printf "=== TOTAL WALL: %.2f s   VRAM PEAK: %s MB ===\n", wall, peak }' "$LOG"
echo "log: $LOG   frames: $OUT/frames_$LABEL"
