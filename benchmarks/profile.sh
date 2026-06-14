#!/usr/bin/env bash
# Wraps Nsight Systems around a single HartsyInference end-to-end generation.
#
# Default target: SDXL 1024² 20 steps (the hot path the paper reports). Override via $TEST_FILTER.
#
# Usage:
#   bash benchmarks/profile.sh [--test "Flux_Dev_Generates*"] [--out benchmarks/results/run_<>]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEST_FILTER="FullyQualifiedName~Sdxl_GenerateImage_Gpu"
OUT_DIR="benchmarks/results/run_$(date -u +%Y-%m-%dT%H%M%SZ)_nsys"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TEST_FILTER="$2"; shift 2 ;;
        --out)  OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--test <xunit-filter>] [--out <dir>]"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

command -v nsys >/dev/null || { echo "Error: nsys missing (Nsight Systems). Install via NVIDIA HPC SDK or apt-get install nsight-systems."; exit 1; }
mkdir -p "$OUT_DIR"

REPORT="$OUT_DIR/nsys_report"

echo "[profile] running nsys profile on test=$TEST_FILTER"
echo "[profile] report → $REPORT.qdrep / .nsys-rep"

nsys profile \
    --output "$REPORT" \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=cpu \
    --capture-range=cudaProfilerApi \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -- \
    dotnet test tests/HartsyInference.Diffusion.Tests/HartsyInference.Diffusion.Tests.csproj \
        --filter "$TEST_FILTER" \
        --logger "console;verbosity=detailed" \
        --no-build \
        || { echo "[profile] test run failed; partial trace may still exist at $REPORT"; exit 1; }

echo ""
echo "[profile] done. Generating summary..."

# Stats: top kernels by total time
nsys stats --report cudaapisum,gpukernsum,nvtxsum "$REPORT.nsys-rep" \
    > "$OUT_DIR/nsys_stats.txt" 2>&1 || \
    nsys stats --report cudaapisum,gpukernsum,nvtxsum "$REPORT.qdrep" \
    > "$OUT_DIR/nsys_stats.txt" 2>&1 || true

echo "[profile] summary → $OUT_DIR/nsys_stats.txt"
echo "[profile] open the report in Nsight Systems GUI:"
ls -la "$OUT_DIR"
