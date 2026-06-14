#!/usr/bin/env bash
# HartsyInference end-to-end benchmark harness.
#
# Runs:
#   1. Hardware + software fingerprinting → hardware.txt, software.txt, digests.txt
#   2. C# GPU microbenchmarks (BenchmarkDotNet)        → microbench_csharp.csv
#   3. Python PyTorch baselines                        → microbench_pytorch.csv, e2e_pytorch.csv
#   4. (optional) C# end-to-end generation tests       → e2e_csharp.csv
#   5. Joins both microbench CSVs via analyze.py       → comparison.{md,csv}
#   6. Atomically moves staging dir → benchmarks/results/run_<utc>_<gpu>/
#
# Usage:
#   bash benchmarks/run_benchmarks.sh [--smoke] [--skip-python] [--skip-e2e]
#       [--device cuda:0] [--trials 5] [--out-base benchmarks/results]
#
# Run from the repo root. Requires: dotnet (build only), nvidia-smi, python3 (with venv that has
# requirements.txt installed). The script aborts if any of those is missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ────────────────────────────────────────────────────────────────
SMOKE=0
SKIP_PYTHON=0
SKIP_E2E=0
DEVICE="cuda:0"
TRIALS=5
OUT_BASE="$REPO_ROOT/benchmarks/results"
PY_VENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1; TRIALS=1; shift ;;
        --skip-python) SKIP_PYTHON=1; shift ;;
        --skip-e2e) SKIP_E2E=1; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --out-base) OUT_BASE="$2"; shift 2 ;;
        --py-venv) PY_VENV="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's|^# \?||'
            exit 0
            ;;
        *) echo "Unknown arg: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# ── Sanity checks ───────────────────────────────────────────────────────────
command -v nvidia-smi >/dev/null || { echo "Error: nvidia-smi missing"; exit 1; }
command -v dotnet     >/dev/null || { echo "Error: dotnet missing"; exit 1; }
[[ "$SKIP_PYTHON" -eq 0 ]] && { command -v python3 >/dev/null || { echo "Error: python3 missing"; exit 1; }; }

# Refuse to start if other CUDA processes are pinning VRAM (avoids confounding noise).
OTHER_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)
if [[ -n "$OTHER_PIDS" ]]; then
    echo "Warning: other CUDA processes detected: $OTHER_PIDS"
    echo "  Continue at your own risk — measurements may be noisy."
fi

# ── Identifiers ─────────────────────────────────────────────────────────────
TIMESTAMP_UTC=$(date -u +%Y-%m-%dT%H%M%SZ)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
GPU_SLUG=$(echo "$GPU_NAME" \
    | tr '[:upper:]' '[:lower:]' \
    | tr ' /(),' '-----' \
    | tr -s '-' \
    | sed 's/^-//;s/-$//')
RUN_ID="run_${TIMESTAMP_UTC}_${GPU_SLUG}"

STAGING="$(mktemp -d -t hartsyinference_bench_XXXXXX)"
echo "[run_benchmarks] staging dir: $STAGING"
trap 'rm -rf "$STAGING"' EXIT

# ── 1. Fingerprints ─────────────────────────────────────────────────────────
echo "[1/6] Capturing hardware + software fingerprints..."

{
    echo "## hostname"; hostname
    echo "## uname"; uname -a
    echo "## kernel"; cat /proc/version 2>/dev/null || true
    echo "## CPU"; lscpu | grep -E '^(Architecture|CPU\(s\)|Model name|CPU MHz|Cache)' || true
    echo "## RAM"; free -h
    echo "## nvidia-smi -q (full)"; nvidia-smi -q
    echo "## nvidia-smi --query-gpu (machine readable)"
    nvidia-smi --query-gpu=name,driver_version,vbios_version,compute_cap,memory.total,power.limit,clocks.max.sm,clocks.max.mem,persistence_mode,ecc.mode.current --format=csv
    echo "## PCIe topology"
    nvidia-smi topo --matrix 2>/dev/null || echo "topology unavailable"
    echo "## CPU governor"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a"
} > "$STAGING/hardware.txt"

{
    echo "## dotnet"; dotnet --info 2>/dev/null | head -50
    echo "## CUDA toolkit"; nvcc --version 2>/dev/null || echo "nvcc not on PATH"
    echo "## driver"; nvidia-smi --query-gpu=driver_version --format=csv,noheader
    echo "## python"; python3 --version
    echo "## pip freeze (filtered)"
    if [[ -n "$PY_VENV" ]]; then
        "$PY_VENV/bin/pip" freeze 2>/dev/null | grep -E '^(torch|diffusers|transformers|accelerate|xformers|safetensors|sentencepiece|numpy|scipy|nvidia-)' || true
    else
        pip freeze 2>/dev/null | grep -E '^(torch|diffusers|transformers|accelerate|xformers|safetensors|sentencepiece|numpy|scipy|nvidia-)' || true
    fi
    echo "## git rev"; git rev-parse HEAD 2>/dev/null || echo "no git"
    echo "## git status (uncommitted)"; git status --porcelain 2>/dev/null || true
} > "$STAGING/software.txt"

{
    echo "## PTX SHA-256"
    find src/HartsyInference.Cuda/Ptx -name '*.ptx' 2>/dev/null | sort | xargs -r sha256sum
    echo "## Native CUDA SHA-256"
    find native/cuda -name '*.cu' 2>/dev/null | sort | xargs -r sha256sum
} > "$STAGING/digests.txt"

# ── 2. C# build (Release) ───────────────────────────────────────────────────
echo "[2/6] Building HartsyInference.GpuBenchmarks (Release)..."
dotnet build benchmarks/HartsyInference.GpuBenchmarks/HartsyInference.GpuBenchmarks.csproj -c Release \
    -v minimal > "$STAGING/dotnet_build.log" 2>&1

# ── 3. C# microbenchmarks ───────────────────────────────────────────────────
echo "[3/6] Running C# microbenchmarks ($TRIALS trials each)..."
FILTER='*'
if [[ "$SMOKE" -eq 1 ]]; then
    FILTER='*MatMul*'
fi

dotnet run --no-build -c Release --project benchmarks/HartsyInference.GpuBenchmarks -- \
    --filter "$FILTER" \
    --warmupCount 1 --iterationCount "$TRIALS" \
    --exporters json,markdown,csv \
    --artifacts "$STAGING/csharp_bench" \
    > "$STAGING/csharp_bench.log" 2>&1 || {
        echo "[3/6] FAILED — log saved to $STAGING/csharp_bench.log"
        cat "$STAGING/csharp_bench.log" | tail -30
        exit 1
    }

# Move BDN's combined CSV (if present) to canonical name
COMBINED_CSV=$(find "$STAGING/csharp_bench" -name '*-report.csv' -o -name 'BenchmarkRun-joined.csv' 2>/dev/null | head -n1)
if [[ -n "$COMBINED_CSV" ]]; then
    cp "$COMBINED_CSV" "$STAGING/microbench_csharp.csv"
fi

# ── 4. Python baselines ─────────────────────────────────────────────────────
if [[ "$SKIP_PYTHON" -eq 0 ]]; then
    echo "[4/6] Running PyTorch baselines..."
    PY_BIN=python3
    if [[ -n "$PY_VENV" ]]; then
        PY_BIN="$PY_VENV/bin/python3"
    fi
    PY_TRIALS="$TRIALS"
    if [[ "$SMOKE" -eq 1 ]]; then PY_TRIALS=1; fi
    SKIP_E2E_FLAG=""
    if [[ "$SKIP_E2E" -eq 1 || "$SMOKE" -eq 1 ]]; then SKIP_E2E_FLAG="--skip-e2e"; fi

    # Mimic the baseline runner but redirect output into our staging dir
    PYTHON="$PY_BIN" bash benchmarks/python-baseline/run_all.sh \
        --output "$STAGING/pytorch" \
        --trials "$PY_TRIALS" \
        $SKIP_E2E_FLAG \
        > "$STAGING/pytorch_bench.log" 2>&1 || {
            echo "[4/6] Python baseline failed — see $STAGING/pytorch_bench.log"
            tail -20 "$STAGING/pytorch_bench.log"
        }
    if [[ -f "$STAGING/pytorch/microbench.csv" ]]; then
        cp "$STAGING/pytorch/microbench.csv" "$STAGING/microbench_pytorch.csv"
    fi
    if [[ -f "$STAGING/pytorch/e2e.csv" ]]; then
        cp "$STAGING/pytorch/e2e.csv" "$STAGING/e2e_pytorch.csv"
    fi
else
    echo "[4/6] skipping Python baselines (--skip-python)"
fi

# ── 5. Comparison ───────────────────────────────────────────────────────────
echo "[5/6] Generating comparison report..."
PY_BIN=python3
if [[ -n "$PY_VENV" ]]; then PY_BIN="$PY_VENV/bin/python3"; fi
"$PY_BIN" benchmarks/analyze.py \
    --csharp "$STAGING/microbench_csharp.csv" \
    --pytorch "$STAGING/microbench_pytorch.csv" \
    --output-dir "$STAGING" \
    > "$STAGING/analyze.log" 2>&1 || {
        echo "  (analyze.py is best-effort — error captured to $STAGING/analyze.log)"
    }

# ── 6. Move to results/ atomically ──────────────────────────────────────────
FINAL_DIR="$OUT_BASE/$RUN_ID"
mkdir -p "$OUT_BASE"
mv "$STAGING" "$FINAL_DIR"
trap - EXIT  # don't delete the moved dir

echo ""
echo "[6/6] Done. Results in: $FINAL_DIR"
echo ""
echo "Files written:"
ls -la "$FINAL_DIR" | tail -n +2
echo ""
echo "To view comparison: less '$FINAL_DIR/comparison.md'"
