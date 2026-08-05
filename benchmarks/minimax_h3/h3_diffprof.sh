#!/bin/bash
# True per-step, per-op GPU cost for MiniMax-H3.
#
# The pipeline resets the op-profile accumulator after step 0 and dumps it at the end of the denoise
# loop, so the table below is denoise-only: no text encode, no VAE decode, no step-0 residency
# warm-up. That replaces an earlier 3-vs-6-step differencing approach, which does NOT work here —
# per-call Linear time depends on how many weights happened to be resident, and a 3-step run measured
# MORE total Linear time than a 6-step run of the same binary, making the difference negative.
#
#   ./h3_diffprof.sh [LABEL] [STEPS]
set -u

LABEL=${1:-prof}
STEPS=${2:-11}
REPO=/home/hartsy/Desktop/HartsyInference
OUT=${H3_BENCH_OUT:-$REPO/benchmarks/results/h3}
CKPT=$REPO/Models/Stable-Diffusion/MiniMaxH3/flat/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
PROMPT_FILE=$REPO/Models/bench-comfy/prompt.txt

GPU_SMI=${H3_BENCH_GPU:-1}
EXPECT_GPU=${H3_BENCH_GPU_NAME:-"RTX 4090"}
MIN_FREE_MB=${H3_BENCH_MIN_FREE:-22000}

mkdir -p "$OUT"
PROMPT=$(cat "$PROMPT_FILE")

SWARM_WAS_UP=0
systemctl --user is-active --quiet swarmui.service && SWARM_WAS_UP=1
restore() { [ "$SWARM_WAS_UP" = "1" ] && systemctl --user start swarmui.service >/dev/null 2>&1; }
trap restore EXIT INT TERM
[ "$SWARM_WAS_UP" = "1" ] && { systemctl --user stop swarmui.service; sleep 8; }

NAME=$(nvidia-smi -i "$GPU_SMI" --query-gpu=name --format=csv,noheader)
case "$NAME" in
    *"$EXPECT_GPU"*) ;;
    *) echo "FATAL: GPU $GPU_SMI is '$NAME', expected '$EXPECT_GPU'."; exit 1 ;;
esac
FREE=$(nvidia-smi -i "$GPU_SMI" --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$FREE" -lt "$MIN_FREE_MB" ] && { echo "FATAL: only ${FREE} MB free on GPU $GPU_SMI."; exit 1; }
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

cd "$REPO"
# PROFILE_SYNC drains the stream at every range close, so each op's time is TRUE GPU time. It
# serializes execution, so these wall times are NOT comparable to an h3_bench.sh run.
# PROFILE_FINE turns on the sub-op ranges (the five Sage attention launches).
# `env` so H3_BENCH_ENV's KEY=VALUE words are parsed as assignments — a bare prefix would take the
# expanded text as the command name. Same passthrough as h3_bench.sh, so a profile can be taken under
# the exact config a benchmark ran (without it this defaulted to Sage attention regardless).
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_SMI HARTSY_LOG_LEVEL=Info \
    ${H3_BENCH_ENV:-} \
    HARTSY_PROFILE=1 HARTSY_PROFILE_SYNC=1 HARTSY_PROFILE_FINE=1 HARTSY_PROFILE_OUT=$OUT/${LABEL}.txt \
    dotnet run --project src/HartsyInference.Cli/HartsyInference.Cli.csproj -f net10.0 --no-build -- \
    video -m minimax-h3 --model-path "$CKPT" --frames 141 --width 512 --height 288 \
    --steps "$STEPS" --seed 1 -o "$OUT/${LABEL}_out" "$PROMPT" > "$OUT/${LABEL}.log" 2>&1
echo "exit=$?  $(grep -oE "step [0-9]+/$STEPS: [0-9]+ ms" "$OUT/${LABEL}.log" | tail -1)"

DENOISE=$OUT/${LABEL}.txt.denoise$((STEPS - 1))
[ -f "$DENOISE" ] || { echo "FATAL: no denoise profile at $DENOISE — did the run reach the loop?"; exit 1; }

H3_PROF=$DENOISE H3_STEPS=$((STEPS - 1)) python3 - <<'PY'
import os

# "Sage.*" ranges nest inside "SDPA", so SDPA's own row double-counts them. Attribute the
# remainder to SDPA.other and drop the parent, or the column would not sum to the step.
CHILDREN = {"SDPA": ("Sage.",)}

steps = int(os.environ["H3_STEPS"])
per = {}
for ln in open(os.environ["H3_PROF"]).read().split("\n")[1:]:
    f = ln.split()
    if len(f) >= 4:
        try:
            per[f[0]] = (float(f[2]) / steps, int(f[1]) // steps)
        except ValueError:
            pass

for parent, prefixes in CHILDREN.items():
    if parent not in per:
        continue
    kids = sum(v[0] for k, v in per.items() if any(k.startswith(p) for p in prefixes))
    pms, pcalls = per.pop(parent)
    if kids > 0:
        per[parent + ".other"] = (pms - kids, pcalls)

rows = sorted(((ms, c, op) for op, (ms, c) in per.items()), reverse=True)
tot = sum(r[0] for r in rows)
print(f"\n{'op':26s} {'calls/step':>10s} {'ms/step':>9s} {'%':>6s}   (denoise-only, PROFILE_SYNC, {steps} steps)")
for ms, c, op in rows:
    if ms > 0.3:
        print(f"{op:26s} {c:>10d} {ms:>9.1f} {100 * ms / tot:>5.1f}%")
print(f"{'TOTAL (sum of ops)':26s} {sum(c for _, c, _ in rows):>10d} {tot:>9.1f}")
print("\nTOTAL is serialized GPU time. Compare against an unsynced h3_bench.sh step to size the launch gap.")
PY
