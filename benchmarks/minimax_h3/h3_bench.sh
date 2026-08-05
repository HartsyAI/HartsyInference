#!/bin/bash
# MiniMax-H3 step-time benchmark. Guarded against the two landmines that have each invalidated a
# full round of numbers: another CUDA tenant squatting VRAM, and picking the wrong GPU.
#
#   ./h3_bench.sh [STEPS] [LABEL] [SEED] [FRAMES] [WIDTH] [HEIGHT]
#
# STEPS=3 for iteration; use 30 for any number you intend to report — step 1 runs ~2% high and the
# reported figure is the mean of steps 4..N.
set -u

STEPS=${1:-3}
LABEL=${2:-run}
SEED=${3:-1}
FRAMES=${4:-141}
WIDTH=${5:-512}
HEIGHT=${6:-288}

REPO=/home/hartsy/Desktop/HartsyInference
OUT=${H3_BENCH_OUT:-$REPO/benchmarks/results/h3}
CKPT=$REPO/Models/Stable-Diffusion/MiniMaxH3/flat/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
PROMPT_FILE=$REPO/Models/bench-comfy/prompt.txt

# The 4090 is nvidia-smi index 1 (PCI 04:00.0); the 3060 is index 0. CUDA_VISIBLE_DEVICES alone
# defaults to fastest-first, so it and nvidia-smi disagree about what "1" means. Pinning
# CUDA_DEVICE_ORDER=PCI_BUS_ID makes the two indices identical, which is why GPU_SMI == GPU_CUDA.
GPU_SMI=${H3_BENCH_GPU:-1}
GPU_CUDA=$GPU_SMI
EXPECT_GPU=${H3_BENCH_GPU_NAME:-"RTX 4090"}
MIN_FREE_MB=${H3_BENCH_MIN_FREE:-22000}

mkdir -p "$OUT"
LOG=$OUT/${LABEL}_s${STEPS}_seed${SEED}.log

[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 1; }
[ -f "$PROMPT_FILE" ] || { echo "FATAL: prompt not found: $PROMPT_FILE"; exit 1; }
PROMPT=$(cat "$PROMPT_FILE")

SWARM_WAS_UP=0
systemctl --user is-active --quiet swarmui.service && SWARM_WAS_UP=1
restore() { [ "$SWARM_WAS_UP" = "1" ] && systemctl --user start swarmui.service >/dev/null 2>&1; }
trap restore EXIT INT TERM
if [ "$SWARM_WAS_UP" = "1" ]; then
    echo "stopping swarmui.service (holds ~6.7 GB on the 4090)..."
    systemctl --user stop swarmui.service
    sleep 8
fi

NAME=$(nvidia-smi -i "$GPU_SMI" --query-gpu=name --format=csv,noheader)
case "$NAME" in
    *"$EXPECT_GPU"*) ;;
    *) echo "FATAL: GPU $GPU_SMI is '$NAME', expected '$EXPECT_GPU'. Numbers would not be comparable."; exit 1 ;;
esac

echo "--- other CUDA tenants ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
FREE=$(nvidia-smi -i "$GPU_SMI" --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU $GPU_SMI ($NAME): ${FREE} MiB free"
if [ "$FREE" -lt "$MIN_FREE_MB" ]; then
    echo "FATAL: <${MIN_FREE_MB} MB free — the 19.4 GB DiT preload will not fit and numbers will not be comparable."
    exit 1
fi

cd "$REPO"
# `env` rather than a bare assignment prefix: bash does not re-parse KEY=VALUE that arrives via
# variable expansion, so $H3_BENCH_ENV would be taken as the command name.
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_CUDA HARTSY_LOG_LEVEL=Info \
    ${H3_BENCH_ENV:-} \
    dotnet run --project src/HartsyInference.Cli/HartsyInference.Cli.csproj -f net10.0 ${H3_BENCH_NOBUILD:---no-build} -- \
    video -m minimax-h3 --model-path "$CKPT" \
    --frames "$FRAMES" --width "$WIDTH" --height "$HEIGHT" --steps "$STEPS" --seed "$SEED" \
    -o "$OUT/${LABEL}_out" "$PROMPT" > "$LOG" 2>&1
RC=$?

# The pre-run guard cannot catch a tenant that arrives mid-run — which is exactly what silently
# invalidated one 30-step baseline. Re-check after, and say so loudly.
AFTER=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    | grep -v "$$" | grep -viE "rustdesk")
if [ -n "$AFTER" ]; then
    echo "!!! WARNING: another CUDA tenant was present at the END of this run — numbers are suspect:"
    echo "$AFTER"
fi

echo "--- per-step (exit=$RC) ---"
grep -oE "step [0-9]+/$STEPS: [0-9]+ ms" "$LOG"

# Steps 1-3 are warm-up-ish (step 1 carries first-touch costs); report the mean of 4..N.
# Field-split rather than gawk's 3-arg match() — the default awk here is mawk, which lacks it.
grep -oE "step [0-9]+/$STEPS: [0-9]+ ms" "$LOG" | awk -v steps="$STEPS" '
    { split($2, p, "/"); idx = p[1] + 0; ms = $3 + 0;
      if (idx >= 4) { s += ms; n++; if (n == 1 || ms < lo) lo = ms; if (ms > hi) hi = ms } }
    END {
        if (n > 0) printf "MEAN steps 4..%d: %.1f ms  (n=%d, range %d-%d)\n", steps, s / n, n, lo, hi
        else print "MEAN: n/a (need >=4 steps for a reportable figure)"
    }'

[ "$RC" != "0" ] && { echo "RUN FAILED — tail of $LOG:"; tail -25 "$LOG"; }
exit $RC
