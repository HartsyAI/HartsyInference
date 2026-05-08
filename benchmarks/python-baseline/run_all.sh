#!/usr/bin/env bash
# Run every Python baseline benchmark and emit a single result directory.
# Usage:
#   bash run_all.sh --output benchmarks/results/run_<utc>_<gpu>_pytorch_baseline [--trials 5] [--skip-e2e]
#
# This script intentionally does NOT activate a venv — invoke it from inside an activated venv that
# has installed `requirements.txt`. This keeps the path predictable for cloud-runner orchestration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_DIR=""
TRIALS=5
SKIP_E2E=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUT_DIR="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --skip-e2e) SKIP_E2E=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    echo "Error: --output is required" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "[run_all] Output: $OUT_DIR"
echo "[run_all] Trials: $TRIALS"

# Step 1: write fingerprints (mirror of the bash blocks in PROFILING_METHODOLOGY.md)
python3 - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "$SCRIPT_DIR")
from _common import write_hardware_fingerprint, write_software_fingerprint
out = Path("$OUT_DIR")
write_hardware_fingerprint(out)
write_software_fingerprint(out)
print(f"[run_all] hardware.txt + software.txt written")
PY

MICROBENCH_CSV="$OUT_DIR/microbench.csv"
E2E_CSV="$OUT_DIR/e2e.csv"

# Step 2: microbenches (each appends to microbench.csv)
echo "[run_all] matmul..."
python3 "$SCRIPT_DIR/bench_pytorch_matmul.py" --output "$MICROBENCH_CSV" --trials "$TRIALS"

echo "[run_all] conv2d..."
python3 "$SCRIPT_DIR/bench_pytorch_conv2d.py" --output "$MICROBENCH_CSV" --trials "$TRIALS"

echo "[run_all] norms..."
python3 "$SCRIPT_DIR/bench_pytorch_norms.py" --output "$MICROBENCH_CSV" --trials "$TRIALS"

echo "[run_all] sdpa..."
python3 "$SCRIPT_DIR/bench_pytorch_sdpa.py" --output "$MICROBENCH_CSV" --trials "$TRIALS"

echo "[run_all] elementwise..."
python3 "$SCRIPT_DIR/bench_pytorch_elementwise.py" --output "$MICROBENCH_CSV" --trials "$TRIALS"

# Step 3: end-to-end (slower, optional)
if [[ "$SKIP_E2E" -eq 0 ]]; then
    echo "[run_all] e2e (sdxl + flux)..."
    python3 "$SCRIPT_DIR/bench_pytorch_e2e.py" --output "$E2E_CSV" --trials 3 || \
        echo "[run_all] e2e: at least one model skipped (likely missing checkpoint); see logs above"
else
    echo "[run_all] skipping e2e (--skip-e2e set)"
fi

echo "[run_all] DONE → $OUT_DIR"
echo "[run_all] microbench rows: $(($(wc -l < "$MICROBENCH_CSV") - 1))"
if [[ -f "$E2E_CSV" ]]; then
    echo "[run_all] e2e rows: $(($(wc -l < "$E2E_CSV") - 1))"
fi
