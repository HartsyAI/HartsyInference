#!/bin/bash
# LTX-2.5 in-engine phase benchmark (CLI, not SwarmUI). Same guards as minimax_h3/h3_bench.sh: another CUDA
# tenant squatting VRAM and picking the wrong GPU have each invalidated a full round of numbers.
#
#   ./ltx25_bench.sh [STEPS] [LABEL] [SEED] [FRAMES] [WIDTH] [HEIGHT]
#
# Prints the [ltx2-phase] ladder (TE, per-step, VAE decode) plus total wall. Use STEPS=30 for anything
# reported; step 1 runs ~40% high (first-forward int8 row-scale uploads) and is excluded from the step mean.
set -u

STEPS=${1:-30}
LABEL=${2:-run}
SEED=${3:-1}
FRAMES=${4:-97}
WIDTH=${5:-768}
HEIGHT=${6:-512}

REPO=/home/hartsy/Desktop/HartsyInference
# LTX25_BENCH_CLI points this at a PRIVATE COPY of the Release output. The default path is shared build output:
# another session rebuilding the tree mid-campaign silently swaps the binary between reps and every number in
# that campaign becomes uncomparable (happened 2026-08-13 — HartsyInference.Video.dll was rebuilt between rep 1
# and rep 2). Snapshot before a campaign whenever anyone else might be building:
#   cp -r src/HartsyInference.Cli/bin/Release/net10.0 /tmp/benchcli
#   LTX25_BENCH_CLI=/tmp/benchcli/HartsyInference.Cli.dll ./ltx25_ab.sh VAR 0 1 4
CLI=${LTX25_BENCH_CLI:-$REPO/src/HartsyInference.Cli/bin/Release/net10.0/HartsyInference.Cli.dll}
MODEL_DIR=${LTX25_BENCH_MODEL:-$REPO/Models/Stable-Diffusion/LTX-2.5}
OUT=${LTX25_BENCH_OUT:-/tmp/ltx25_bench}
# `-` not `:-` on purpose: an EXPLICITLY EMPTY LTX25_BENCH_PROMPT must stay empty, because the empty prompt is
# the unconditional-prior control for conditioning experiments. With `:-` it silently fell back to the default
# below, so an "empty prompt" run generated a normal prompt and any prior measured from it was meaningless.
PROMPT=${LTX25_BENCH_PROMPT-"a lone lighthouse keeper walking along a rocky coastline at sunset, waves crashing, cinematic wide shot, volumetric light, highly detailed, 35mm film look"}

# The 4090 is nvidia-smi index 1 (PCI 04:00.0); the 3060 is index 0. CUDA_VISIBLE_DEVICES alone defaults to
# fastest-first, so it and nvidia-smi disagree about what "1" means. CUDA_DEVICE_ORDER=PCI_BUS_ID makes them
# identical, which is why GPU_SMI == GPU_CUDA.
GPU_SMI=${LTX25_BENCH_GPU:-1}
GPU_CUDA=$GPU_SMI
EXPECT_GPU=${LTX25_BENCH_GPU_NAME:-"RTX 4090"}
MIN_FREE_MB=${LTX25_BENCH_MIN_FREE:-22000}

mkdir -p "$OUT"
LOG=$OUT/${LABEL}_s${STEPS}_seed${SEED}.log

[ -f "$CLI" ] || { echo "FATAL: CLI not built: $CLI (dotnet build src/HartsyInference.Cli -c Release -f net10.0)"; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "FATAL: model dir not found: $MODEL_DIR"; exit 1; }

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
echo "GPU $GPU_SMI = $NAME, ${FREE} MB free"

rm -rf "$OUT/frames_$LABEL"
START=$(date +%s.%N)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_CUDA HARTSY_LOG_LEVEL=Info \
    dotnet "$CLI" video "$PROMPT" -m ltx-2.5 --model-path "$MODEL_DIR" \
    --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES" --steps "$STEPS" --cfg 3.0 --seed "$SEED" \
    -q -o "$OUT/frames_$LABEL" > "$LOG" 2>&1
RC=$?
END=$(date +%s.%N)
WALL=$(echo "$END - $START" | bc)

grep -E "ltx2-phase|LTX-2 T2V" "$LOG"
[ "$RC" -eq 0 ] || { echo "FATAL: generation failed (rc=$RC), see $LOG"; tail -20 "$LOG"; exit 1; }

# Step mean over steps 2..N — step 1 carries the first-forward int8 row-scale upload storm.
awk -v wall="$WALL" '
/ltx2-phase\] step / { n++; if (n > 1) { gsub(/ms.*/, "", $0); sub(/.*: /, "", $0); s += $0; c++ } }
END { printf "\n=== steps 2..%d mean: %.1f ms  (n=%d) ===\n", n, (c ? s / c : 0), c;
      printf "=== TOTAL WALL: %.2f s ===\n", wall }' "$LOG"
echo "log: $LOG   frames: $OUT/frames_$LABEL"
