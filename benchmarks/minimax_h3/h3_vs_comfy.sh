#!/bin/bash
# Interleaved MiniMax-H3 benchmark: ComfyUI and HartsyInference alternating in ONE session.
#
# WHY INTERLEAVED. This 4090 sits at its 450 W cap during a run, so step time drifts with temperature:
# four runs of one identical config measured 1683.7 / 1726.4 / 1808.1 / 1671.4 ms — a 137 ms (8%)
# spread with no other GPU tenant. Comparing our number today against a ComfyUI number from another
# day says nothing at the tens-of-ms scale. Alternating makes both engines eat the same thermal state.
#
# Both engines need ~22 GB, so they cannot be resident together: each ComfyUI round ends with a /free
# to hand the VRAM back. Comfy therefore reloads weights every round, so each round runs TWO Comfy
# generations and keeps the second — the first pays the reload.
#
#   ./h3_vs_comfy.sh [ROUNDS] [PORT]
set -u

ROUNDS=${1:-3}
PORT=${2:-8199}
REPO=/home/hartsy/Desktop/HartsyInference
RIG=$REPO/Models/bench-comfy
OUT=${H3_BENCH_OUT:-$REPO/benchmarks/results/h3}
COMFY_PY="/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv/bin/python"
COMFY_LOG=$OUT/comfy_server.log
GPU_SMI=${H3_BENCH_GPU:-1}
# Matches what h3_bench.sh runs, per the user's full-precision-attention decision.
OURS_ENV=${H3_VS_COMFY_OURS_ENV:-"HARTSY_SAGE_ATTN=0 HARTSY_SDPA_F16=1"}

mkdir -p "$OUT"
[ -f "$COMFY_PY" ] || { echo "FATAL: comfy python not found: $COMFY_PY"; exit 1; }

SWARM_WAS_UP=0
systemctl --user is-active --quiet swarmui.service && SWARM_WAS_UP=1
cleanup() {
    pkill -f "main.py --port $PORT" >/dev/null 2>&1
    [ "$SWARM_WAS_UP" = "1" ] && systemctl --user start swarmui.service >/dev/null 2>&1
}
trap cleanup EXIT INT TERM
[ "$SWARM_WAS_UP" = "1" ] && { systemctl --user stop swarmui.service; sleep 8; }

start_comfy() {
    # Health-check the port rather than pgrep: `pgrep -f "main.py --port N"` also matches this very
    # script's own command line, so it reports a server that was never started.
    curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/system_stats" && return 0
    cd "$RIG"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_SMI PYTHONPATH=$RIG/pylibs \
        nohup "$COMFY_PY" ComfyUI/main.py --port "$PORT" --listen 127.0.0.1 >> "$COMFY_LOG" 2>&1 &
    for _ in $(seq 1 120); do
        curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/system_stats" && return 0
        sleep 2
    done
    echo "FATAL: comfy did not come up on $PORT"; exit 1
}

# One generation; prints the sampler's own s/it. Comfy caches identical workflows and returns in
# 0.00 s, so the seed must change every submission.
comfy_gen() {
    local seed=$1
    local marker
    marker=$(wc -l < "$COMFY_LOG")
    python3 - "$RIG/wf.json" "$seed" "$PORT" <<'PY'
import json, sys, urllib.request, uuid
wf = json.load(open(sys.argv[1]))
prompt = wf.get("prompt", wf)
for node in prompt.values():
    if node.get("class_type") == "KSampler":
        node["inputs"]["seed"] = int(sys.argv[2])
body = json.dumps({"prompt": prompt, "client_id": uuid.uuid4().hex}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[3]}/prompt", body,
                             {"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"])
PY
}

wait_idle() {
    for _ in $(seq 1 900); do
        q=$(curl -s -m 5 "http://127.0.0.1:$PORT/queue" 2>/dev/null)
        case "$q" in *'"queue_running": []'*) return 0 ;; esac
        [ -z "$q" ] && return 1
        sleep 2
    done
    return 1
}

# The sampler's tqdm line is the like-for-like figure: it excludes text encode and VAE decode, which
# is exactly what our own per-step timer measures.
last_sit() { grep -oE "[0-9]+\.[0-9]+s/it" "$COMFY_LOG" | tail -1; }

echo "round,engine,s_per_step"
for r in $(seq 1 "$ROUNDS"); do
    start_comfy
    comfy_gen $((1000 + r * 2)) > /dev/null; wait_idle || { echo "comfy gen failed"; exit 1; }
    comfy_gen $((1001 + r * 2)) > /dev/null; wait_idle || { echo "comfy gen failed"; exit 1; }
    echo "$r,comfy,$(last_sit)"

    curl -s -m 30 -X POST "http://127.0.0.1:$PORT/free" -H "Content-Type: application/json" \
        -d '{"unload_models": true, "free_memory": true}' > /dev/null
    sleep 5

    ms=$(H3_BENCH_ENV="$OURS_ENV" bash "$REPO/benchmarks/minimax_h3/h3_bench.sh" 30 "vs_ours_r$r" 1 2>&1 \
        | grep -oE "MEAN steps 4\.\.30: [0-9.]+" | grep -oE "[0-9.]+$")
    echo "$r,hartsy,$(python3 -c "print(f'{float('${ms:-0}')/1000:.3f}')")"
done
