#!/bin/bash
# Renders the GOLD quality reference for MiniMax-H3: same seed and geometry as a normal run, with the
# precision shortcuts turned off. Every later optimization is scored as SSIM against this.
#
#   ./h3_gold.sh [STEPS] [SEED]
#
# WHY NOT A COMFYUI REFERENCE. The plan originally called for one, but pixel SSIM against ComfyUI
# cannot gate anything: it seeds its own noise (its bench workflow uses seed 42, ours uses 1) and its
# RNG does not match ours, so two equally-correct runs are different videos and SSIM lands near 0.3.
# It also saves lossy VP9 while we save lossless PNG. A gold run of THIS pipeline at the SAME seed
# has bit-identical starting noise, so SSIM is measuring only what the optimization changed.
#
# It also satisfies the reason the plan wanted Comfy rather than "yesterday's output": gold is
# strictly MORE faithful than any optimized variant, so a change that improves fidelity moves SSIM up
# instead of being rejected as a regression.
#
# What is disabled, and why:
#   HARTSY_FP8_NATIVE=0  - no native fp8 GEMM, so activations are never quantized to e4m3. The fp8
#                          WEIGHTS are inherent to the checkpoint and still exact; only the
#                          activation-side approximation goes away.
#   HARTSY_FP8_F32=1     - and the fallback GEMM runs in F32 rather than its default BF16, so the
#                          weight cast is exact too and nothing in the Linear path is approximated.
#                          (Without this the default is BF16, not F16 — ResolveGemmDtype picks BF16
#                          whenever the other operand is F32 precisely because F16 would overflow on
#                          SwiGLU. So the default fallback is already safe; F32 is simply better.)
#
# ATTENTION STAYS AS PRODUCTION (Sage INT8) by default. That is deliberate: Phases 2-5 change the
# Linear path only, so leaving attention identical on both sides makes it cancel and keeps SSIM
# measuring exactly the thing under test. Disabling Sage as well is ~40x slower here (a 3-step run
# did not finish in 10 minutes, because the fallback materializes F32 attention over 6550 tokens x 56
# heads), which is not a viable 30-step reference. Phase 6 needs its own attention-only reference —
# pass GOLD_SAGE=0 for that, and expect it to take hours.
GOLD_SAGE=${GOLD_SAGE:-1}
set -u

STEPS=${1:-30}
SEED=${2:-1}
REPO=/home/hartsy/Desktop/HartsyInference
OUT=${H3_BENCH_OUT:-$REPO/benchmarks/results/h3}
CKPT=$REPO/Models/Stable-Diffusion/MiniMaxH3/flat/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
GPU_SMI=${H3_BENCH_GPU:-1}

mkdir -p "$OUT"
LOG=$OUT/gold_s${STEPS}_seed${SEED}.log

SWARM_WAS_UP=0
systemctl --user is-active --quiet swarmui.service && SWARM_WAS_UP=1
restore() { [ "$SWARM_WAS_UP" = "1" ] && systemctl --user start swarmui.service >/dev/null 2>&1; }
trap restore EXIT INT TERM
[ "$SWARM_WAS_UP" = "1" ] && { systemctl --user stop swarmui.service; sleep 8; }

cd "$REPO"
# The probe forces a D2H sync, so this run's step times are NOT comparable to h3_bench.sh. That is
# fine: this run exists for its pixels, and the probe is how we know they are finite.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_SMI HARTSY_LOG_LEVEL=Info \
    HARTSY_SAGE_ATTN=$GOLD_SAGE HARTSY_FP8_NATIVE=0 HARTSY_FP8_F32=1 HARTSY_H3_PROBE=1 \
    dotnet run --project src/HartsyInference.Cli/HartsyInference.Cli.csproj -f net10.0 --no-build -- \
    video -m minimax-h3 --model-path "$CKPT" --frames 141 --width 512 --height 288 \
    --steps "$STEPS" --seed "$SEED" -o "$OUT/gold_out" "$(cat "$REPO/Models/bench-comfy/prompt.txt")" \
    > "$LOG" 2>&1
echo "exit=$?"

BAD=$(grep -oE "nonfinite=[0-9]+" "$LOG" | grep -v "nonfinite=0" | head -3)
if [ -n "$BAD" ]; then
    echo "!!! GOLD RUN IS NOT CLEAN — non-finite values present, do NOT use as a reference:"
    echo "$BAD"
    exit 1
fi
grep -oE "nonfinite=0" "$LOG" | wc -l | xargs echo "probe points clean:"
find "$OUT/gold_out" -name 'frame_*.png' | wc -l | xargs echo "frames:"
echo "gold reference: $OUT/gold_out/<run>/  — score against it with h3_ssim.sh"
