#!/usr/bin/env bash
# llama.cpp baseline for Nemotron Dory on the swarm box: builds llama.cpp with Dory support (ggml-org/llama.cpp PR #30306,
# branch dory-support of brianchmielNV's fork) with CUDA, then, under a gate lease it holds itself, runs llama-bench on the
# same pinned GGUF the Swarm run uses, with the frozen protocol of the Swarm bench where it maps (greedy decode of 128
# tokens after a 512-token prompt, 5 reps), and a greedy completion of the probe prompt for the quality comparison.
#
#   benchmarks/swarm_llm_bench/llamacpp_dory_baseline.sh [model.gguf] [out-dir] [gpu index]
#
# The file is verified against the catalog pin (size and sha256, through dory_e2e_run.verify_model) before anything runs.
# The build needs no card and runs first; the GPU phase then re-runs this script through gpu_gate.py, which requests the
# lease, heartbeats, stops the phase if the lease is lost, and passes the granted card as GATE_GPU. A caller that already
# holds a lease (GATE_LEASE_ID set) runs the GPU phase directly on that lease's card (GATE_GPU), or on the third argument
# when the lease came without one. The granted card always wins: a third argument that names another card is refused. The
# gate's gpu index equals nvidia-smi's, which CUDA_DEVICE_ORDER=PCI_BUS_ID makes llama.cpp follow too.
# Defaults: the pinned Q5_K_M under /mnt/model-storage, benchmarks/results/dory, the leased card. The lease asks for the
# card the Swarm run uses (GATE_CARD, default gpu1, the RTX 4090 SwarmUI is pinned to) so the comparison is card for card.
# Decode depth: the Swarm bench decodes behind its rendered chat prompt, so llama-bench decodes at the same depth (-d),
# taken from DEPTH or from <out-dir>/swarm_prompt_depth.txt, which dory_e2e_run.py writes (an estimate: system instruction
# plus bench prompt plus template markers). Without either, or with a llama-bench that has no -d, the decode runs from an
# empty context and the output says so.
# LLAMACPP_DIR overrides the checkout location (default ~/llama.cpp-dory); an existing checkout is reused and only rebuilt.
# The box's quiet window is a precondition (tests/swarm-quiet-window.sh --gpu 4090, then QUIET_WINDOW_PASSED=1): the gate's
# lease keeps other gate users off the card, not SwarmUI's own users. The GPU phase spot-checks the window on the leased
# card (queue, card and journal) right before llama-bench, through the same script, and refuses to measure if it is dirty.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-/mnt/model-storage/Models/llm/nemotron-dory/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf}"
OUT_DIR="${2:-benchmarks/results/dory}"
GPU_ARG="${3:-}"
WORK="${LLAMACPP_DIR:-$HOME/llama.cpp-dory}"
GATE_CARD="${GATE_CARD:-gpu1}"
PROMPT="Explain in three sentences why the sky is blue, then list two things that would change its color."
PHASE="${LLAMACPP_BASELINE_PHASE:-all}"
QUIET="$HERE/../../tests/swarm-quiet-window.sh"

# Configuration checks first, before the hash of a 3.6 GB file: the quiet window must have passed, an inherited lease must
# name its card, and an override may not contradict it.
if [ "${QUIET_WINDOW_PASSED:-}" != "1" ]; then
    echo "REFUSED: run tests/swarm-quiet-window.sh --gpu 4090 first, then QUIET_WINDOW_PASSED=1 $0 ...: the gate's lease keeps other gate users off the card, not SwarmUI's own users" >&2
    exit 1
fi
if [ -n "${GATE_LEASE_ID:-}" ] && [ -z "${GATE_GPU:-}" ] && [ -z "$GPU_ARG" ]; then
    echo "GATE_LEASE_ID is set but neither GATE_GPU nor a gpu index argument names the leased card; refusing to guess" >&2
    exit 1
fi
if [ -n "${GATE_GPU:-}" ] && [ -n "$GPU_ARG" ] && [ "$GPU_ARG" != "$GATE_GPU" ]; then
    echo "gpu $GPU_ARG was asked for but the lease is for gpu $GATE_GPU; the leased card wins, re-run without the override" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"
python3 -c "import sys; sys.path.insert(0, sys.argv[1]); from dory_e2e_run import verify_model; verify_model(sys.argv[2], False)" "$HERE" "$MODEL"

if [ "$PHASE" = "all" ]; then
    if [ ! -d "$WORK/.git" ]; then
        git clone --depth 1 --branch dory-support https://github.com/brianchmielNV/llama.cpp "$WORK"
    fi
    # The recorded commit must be what gets built: a checkout with local edits would attribute the numbers to a commit that
    # cannot reproduce them, so it is refused rather than built.
    if [ -n "$(git -C "$WORK" status --porcelain --untracked-files=no)" ]; then
        echo "llama.cpp checkout at $WORK has local changes; commit, stash or discard them so the recorded commit is what is built" >&2
        git -C "$WORK" status --short --untracked-files=no >&2
        exit 1
    fi
    git -C "$WORK" rev-parse --short HEAD | tee "$OUT_DIR/llamacpp_dory_commit.txt"
    cmake -S "$WORK" -B "$WORK/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$WORK/build" --config Release -j"$(nproc)" --target llama-bench >/dev/null
    # The non-interactive completion tool is llama-completion on current masters and llama-cli (with -no-cnv) on older ones.
    if ! cmake --build "$WORK/build" --config Release -j"$(nproc)" --target llama-completion >/dev/null 2>&1; then
        cmake --build "$WORK/build" --config Release -j"$(nproc)" --target llama-cli >/dev/null
    fi
    if [ -z "${GATE_LEASE_ID:-}" ]; then
        # No lease held by the caller: hold one around the GPU phase only; the gate picks the card, so no override is passed.
        # Exclusive: a benchmark never runs alongside another suite, benchmark or generation (docs/Agents/AGENTS.md).
        exec python3 "$HERE/gpu_gate.py" --agent "Dory AI model integration" \
            --task "llama.cpp Dory baseline: llama-bench and a greedy completion on the pinned Q5_K_M" \
            --model-path "$MODEL" --exclusive --gpu "$GATE_CARD" -- env LLAMACPP_BASELINE_PHASE=run "$0" "$MODEL" "$OUT_DIR"
    fi
fi

GPU="${GATE_GPU:-${GPU_ARG:-0}}"
DEPTH="${DEPTH:-}"
if [ -z "$DEPTH" ] && [ -f "$OUT_DIR/swarm_prompt_depth.txt" ]; then
    DEPTH="$(tr -dc '0-9' < "$OUT_DIR/swarm_prompt_depth.txt")"
fi
DEPTH_ARGS=()
if [ -n "$DEPTH" ] && "$WORK/build/bin/llama-bench" --help 2>&1 | grep -q -- '--n-depth'; then
    DEPTH_ARGS=(-d "$DEPTH")
else
    echo "warning: no decode depth (DEPTH unset and no $OUT_DIR/swarm_prompt_depth.txt, or llama-bench lacks -d); decoding from an empty context, not comparable with the Swarm numbers" >&2
fi
if [ -x "$WORK/build/bin/llama-completion" ]; then
    COMPLETE=("$WORK/build/bin/llama-completion")
else
    COMPLETE=("$WORK/build/bin/llama-cli" -no-cnv)
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
# The spot check of the quiet window on the leased card, right before the measurement: the timed window was the operator's
# precondition; this catches anything that started since (a queued gen, a process on the card, a journal line).
GPU_UUID="$(nvidia-smi --query-gpu=uuid -i "$GPU" --format=csv,noheader | tr -d ' ')"
"$QUIET" --minutes 0 --gpu "$GPU_UUID" --host "${SWARM_HOST:-127.0.0.1}" --port "${SWARM_PORT:-7801}" \
    || { echo "gpu $GPU is not quiet; refusing to measure" >&2; exit 1; }
# Decode throughput at the checkpoint's own loop count (the same count the Swarm bench measures when --loops is omitted),
# behind the Swarm requests' depth when known.
"$WORK/build/bin/llama-bench" -m "$MODEL" -p 512 -n 128 -r 5 -ngl 99 ${DEPTH_ARGS[@]+"${DEPTH_ARGS[@]}"} -o json \
    | tee "$OUT_DIR/llamacpp_dory_bench.json"
echo "${DEPTH:-0}" > "$OUT_DIR/llamacpp_dory_depth.txt"
# Greedy text for the quality comparison against dory_probes.json (same prompt, temperature 0, 200 tokens).
"${COMPLETE[@]}" -m "$MODEL" -ngl 99 --temp 0 -n 200 -p "$PROMPT" 2>"$OUT_DIR/llamacpp_dory_completion.log" \
    | tee "$OUT_DIR/llamacpp_dory_completion.txt"
echo "wrote $OUT_DIR/llamacpp_dory_bench.json and $OUT_DIR/llamacpp_dory_completion.txt (gpu $GPU)"
