#!/usr/bin/env bash
# llama.cpp baseline for Nemotron Dory on the swarm box: builds llama.cpp with Dory support (ggml-org/llama.cpp PR #30306,
# branch dory-support of brianchmielNV's fork) with CUDA, then runs llama-bench on the same pinned GGUF the Swarm run uses,
# with the frozen protocol of the Swarm bench where it maps (greedy decode of 128 tokens after a 512-token prompt, 5 reps),
# and a greedy completion of the probe prompt for the quality comparison. Runs on the box, under a gate lease held by the
# caller (python3 -c 'from gpu_gate import GpuLease' or dory_e2e_run.py --skip-bench --skip-hash hold one; see gpu_gate.py).
#
#   benchmarks/swarm_llm_bench/llamacpp_dory_baseline.sh [model.gguf] [out-dir] [gate gpu index]
#
# Defaults: the pinned Q5_K_M under /mnt/model-storage, benchmarks/results/dory, gpu 0 (the RTX 3060; the gate's index
# equals nvidia-smi's, which CUDA_DEVICE_ORDER=PCI_BUS_ID makes llama.cpp follow too). LLAMACPP_DIR overrides the checkout
# location (default ~/llama.cpp-dory); an existing checkout is reused and only rebuilt.
set -euo pipefail

MODEL="${1:-/mnt/model-storage/Models/llm/nemotron-dory/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf}"
OUT_DIR="${2:-benchmarks/results/dory}"
GPU="${3:-0}"
WORK="${LLAMACPP_DIR:-$HOME/llama.cpp-dory}"
PROMPT="Explain in three sentences why the sky is blue, then list two things that would change its color."

[ -f "$MODEL" ] || { echo "model file missing: $MODEL" >&2; exit 1; }
mkdir -p "$OUT_DIR"

if [ ! -d "$WORK/.git" ]; then
    git clone --depth 1 --branch dory-support https://github.com/brianchmielNV/llama.cpp "$WORK"
fi
git -C "$WORK" rev-parse --short HEAD | tee "$OUT_DIR/llamacpp_dory_commit.txt"
cmake -S "$WORK" -B "$WORK/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$WORK/build" --config Release -j"$(nproc)" --target llama-bench >/dev/null
# The non-interactive completion tool is llama-completion on current masters and llama-cli (with -no-cnv) on older ones.
if cmake --build "$WORK/build" --config Release -j"$(nproc)" --target llama-completion >/dev/null 2>&1; then
    COMPLETE=("$WORK/build/bin/llama-completion")
else
    cmake --build "$WORK/build" --config Release -j"$(nproc)" --target llama-cli >/dev/null
    COMPLETE=("$WORK/build/bin/llama-cli" -no-cnv)
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
# Decode throughput at the checkpoint's own loop count (the same count the Swarm bench measures when --loops is omitted).
"$WORK/build/bin/llama-bench" -m "$MODEL" -p 512 -n 128 -r 5 -ngl 99 -o json | tee "$OUT_DIR/llamacpp_dory_bench.json"
# Greedy text for the quality comparison against dory_probes.json (same prompt, temperature 0, 200 tokens).
"${COMPLETE[@]}" -m "$MODEL" -ngl 99 --temp 0 -n 200 -p "$PROMPT" 2>"$OUT_DIR/llamacpp_dory_completion.log" \
    | tee "$OUT_DIR/llamacpp_dory_completion.txt"
echo "wrote $OUT_DIR/llamacpp_dory_bench.json and $OUT_DIR/llamacpp_dory_completion.txt"
