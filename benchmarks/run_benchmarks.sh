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
TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1; TRIALS=1; shift ;;
        --skip-python) SKIP_PYTHON=1; shift ;;
        --skip-e2e) SKIP_E2E=1; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --out-base) OUT_BASE="$2"; shift 2 ;;
        --py-venv) PY_VENV="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
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

# cuBLAS resolvability. This engine has no system CUDA toolkit; it loads libcublas/libcublasLt by bare
# soname from LD_LIBRARY_PATH (typically a torch venv's nvidia/cublas/lib). BenchmarkDotNet runs each
# benchmark in a CHILD process, so if the path isn't exported the children throw "CUDA is not available"
# and BDN reports every result as `NA` — a silent failure that looks like a finished run. Fail loud, or
# auto-detect a bundle and prepend it. Override the search by exporting LD_LIBRARY_PATH yourself.
cublas_resolvable() {
    ldconfig -p 2>/dev/null | grep -q 'libcublas\.so' && return 0
    local IFS=:; local d
    for d in ${LD_LIBRARY_PATH:-}; do
        [[ -n "$d" ]] && compgen -G "$d/libcublas.so.*" >/dev/null 2>&1 && return 0
    done
    return 1
}
if ! cublas_resolvable; then
    # Bundles live deep (…/venv/lib/python3.x/site-packages/nvidia/cublas/lib, ~depth 13 under $HOME),
    # so keep maxdepth generous. Prefer cu12 over cu13 (the engine's validated cuBLAS), newest path last.
    CAND=$(find "$HOME" -maxdepth 14 -path '*/nvidia/cublas/lib/libcublas.so.*' -not -path '*/Trash/*' 2>/dev/null | sort | head -n1)
    if [[ -n "$CAND" ]]; then
        CUBLAS_DIR=$(dirname "$CAND")
        CUDART_DIR="${CUBLAS_DIR%/cublas/lib}/cuda_runtime/lib"
        export LD_LIBRARY_PATH="${CUBLAS_DIR}$( [[ -d "$CUDART_DIR" ]] && printf ':%s' "$CUDART_DIR" ):/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[run_benchmarks] cuBLAS not on path; auto-detected and prepended: $CUBLAS_DIR"
    else
        echo "Error: libcublas not resolvable and no nvidia/cublas bundle found under \$HOME." >&2
        echo "  Export LD_LIBRARY_PATH to a CUDA 12.x cublas/lib (e.g. a torch venv's nvidia/cublas/lib) and re-run." >&2
        exit 1
    fi
fi

# Refuse to start if other CUDA processes are pinning VRAM (avoids confounding noise).
OTHER_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' || true)
if [[ -n "$OTHER_PIDS" ]]; then
    echo "Warning: other CUDA processes detected: $OTHER_PIDS"
    echo "  Continue at your own risk — measurements may be noisy."
fi

# ── Identifiers ─────────────────────────────────────────────────────────────
TIMESTAMP_UTC=$(date -u +%Y-%m-%dT%H%M%SZ)
# The benchmark fixture binds CUDA device ordinal 0, so when CUDA_VISIBLE_DEVICES masks the GPUs the
# run actually executes on the FIRST visible physical index. Slug that GPU's name (not nvidia-smi's
# default GPU 0) so a 4090 run on a dual-GPU box isn't mislabelled as the 3060. Honors CUDA_DEVICE_ORDER.
PHYS_IDX="${CUDA_VISIBLE_DEVICES:-}"; PHYS_IDX="${PHYS_IDX%%,*}"
[[ -z "$PHYS_IDX" ]] && PHYS_IDX=0
GPU_NAME=$(nvidia-smi -i "$PHYS_IDX" --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1)
[[ -z "$GPU_NAME" ]] && GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
GPU_SLUG=$(echo "$GPU_NAME" \
    | tr '[:upper:]' '[:lower:]' \
    | tr ' /(),' '-----' \
    | tr -s '-' \
    | sed 's/^-//;s/-$//')
# --tag names the A/B run per the perf-plan convention: `run_baseline_…`, `run_post_epilogue_…`, etc.
RUN_PREFIX="run"
[[ -n "$TAG" ]] && RUN_PREFIX="run_${TAG}"
RUN_ID="${RUN_PREFIX}_${TIMESTAMP_UTC}_${GPU_SLUG}"

STAGING="$(mktemp -d -t hartsyinference_bench_XXXXXX)"
echo "[run_benchmarks] staging dir: $STAGING"

# BenchmarkDotNet builds a host project per benchmark by searching the whole repo for the bench
# .csproj BY NAME; a stray duplicate (e.g. a leftover git worktree under .claude/worktrees that
# also contains benchmarks/) makes it abort with "Found more than one matching project file".
# Hide any duplicate outside the canonical path for the duration of the run, restore on exit.
CANONICAL_BENCH_CSPROJ="$REPO_ROOT/benchmarks/HartsyInference.GpuBenchmarks/HartsyInference.GpuBenchmarks.csproj"
HIDDEN_CSPROJS=()
restore_hidden_csprojs() {
    for h in "${HIDDEN_CSPROJS[@]:-}"; do
        [[ -n "$h" && -f "${h}.bdnhidden" ]] && mv "${h}.bdnhidden" "$h"
    done
}
# Restore hidden csprojs on EVERY exit (success OR failure) and clean staging only if it still exists
# (a successful run mv's STAGING into results/, so the rm is then a harmless no-op). Do NOT `trap - EXIT`
# on success — that would skip the csproj restore and strand the worktree's project file as .bdnhidden.
cleanup_on_exit() { restore_hidden_csprojs; [[ -d "$STAGING" ]] && rm -rf "$STAGING"; }
trap cleanup_on_exit EXIT

while IFS= read -r dup; do
    [[ -z "$dup" || "$dup" == "$CANONICAL_BENCH_CSPROJ" ]] && continue
    echo "[run_benchmarks] hiding duplicate bench project to satisfy BenchmarkDotNet: $dup"
    mv "$dup" "${dup}.bdnhidden"
    HIDDEN_CSPROJS+=("$dup")
done < <(find "$REPO_ROOT" -name 'HartsyInference.GpuBenchmarks.csproj' -not -path '*/bin/*' -not -path '*/obj/*' 2>/dev/null)

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
    echo "## perf flags (HARTSY_*) — the A/B config this run executed under"
    echo "run_tag=${TAG:-<none>}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} (phys idx ${PHYS_IDX}, ${GPU_NAME})"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
    for v in HARTSY_EPILOGUE_FUSION HARTSY_TENSORCORE_GEMM HARTSY_FP8_NATIVE \
             HARTSY_HIGH_PRECISION_GEMM HARTSY_LOWVRAM_QUANT HARTSY_PROFILE; do
        echo "${v}=${!v:-0}"
    done
} > "$STAGING/software.txt"

{
    echo "## PTX SHA-256"
    find src/HartsyInference.Cuda/Ptx -name '*.ptx' 2>/dev/null | sort | xargs -r sha256sum
    echo "## Native CUDA SHA-256"
    find native/cuda -name '*.cu' 2>/dev/null | sort | xargs -r sha256sum
} > "$STAGING/digests.txt"

# ── 2. C# build (Release) ───────────────────────────────────────────────────
# The bench project multi-targets (net8.0;net10.0), so dotnet run/build must pin a framework or they
# abort with "Your project targets multiple frameworks". net10.0 is the primary target (see CLAUDE.md).
BENCH_TFM="${BENCH_TFM:-net10.0}"
echo "[2/6] Building HartsyInference.GpuBenchmarks (Release, $BENCH_TFM)..."
dotnet build benchmarks/HartsyInference.GpuBenchmarks/HartsyInference.GpuBenchmarks.csproj -c Release \
    -f "$BENCH_TFM" -v minimal > "$STAGING/dotnet_build.log" 2>&1

# ── 3. C# microbenchmarks ───────────────────────────────────────────────────
echo "[3/6] Running C# microbenchmarks ($TRIALS trials each)..."
FILTER='*'
if [[ "$SMOKE" -eq 1 ]]; then
    FILTER='*MatMul*'
fi

dotnet run --no-build -c Release -f "$BENCH_TFM" --project benchmarks/HartsyInference.GpuBenchmarks -- \
    --filter "$FILTER" \
    --warmupCount 1 --iterationCount "$TRIALS" \
    --exporters json markdown csv \
    --artifacts "$STAGING/csharp_bench" \
    > "$STAGING/csharp_bench.log" 2>&1 || {
        echo "[3/6] FAILED — log saved to $STAGING/csharp_bench.log"
        cat "$STAGING/csharp_bench.log" | tail -30
        exit 1
    }

# BDN writes ONE *-report.csv PER benchmark class (MatMul, Sdpa, Conv2D, …), each with its own param
# columns. Copying just the first (the old `find | head -n1`) silently dropped every other class. Union-
# merge them all into the canonical microbench_csharp.csv (superset header; missing cells blank). The
# per-class CSVs remain under csharp_bench/results/ as the raw record.
python3 - "$STAGING/csharp_bench" "$STAGING/microbench_csharp.csv" <<'PYMERGE' || echo "  (csv union-merge skipped: $?)"
import csv, sys, glob, os
root, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(root, "**", "*-report.csv"), recursive=True))
rows, cols = [], []
for f in files:
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            for k in r:
                if k not in cols: cols.append(k)
            r["_BenchClass"] = os.path.basename(f).replace("-report.csv", "")
            rows.append(r)
if "_BenchClass" not in cols: cols.append("_BenchClass")
if rows:
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"  merged {len(files)} class reports -> {len(rows)} rows in microbench_csharp.csv")
PYMERGE

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
mv "$STAGING" "$FINAL_DIR"  # STAGING no longer exists after this; the EXIT trap's rm becomes a no-op

echo ""
echo "[6/6] Done. Results in: $FINAL_DIR"
echo ""
echo "Files written:"
ls -la "$FINAL_DIR" | tail -n +2
echo ""
echo "To view comparison: less '$FINAL_DIR/comparison.md'"
