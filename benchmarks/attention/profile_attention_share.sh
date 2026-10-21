#!/usr/bin/env bash
# Attention share of a denoise step, per regression case. Each case runs three times on one card: a plain run at 2S
# steps for the true step time, then the synced op profiler (diagnostics.profile + profileSync + profileShapes) at S and
# at 2S steps — the difference cancels model load, text encoding, VAE decode and first-step costs, leaving S steps of
# per-label GPU time and call counts. attention_share.py turns that into the share, the attention-adjacent share and the
# shapes the attention table should be frozen from.
#
#   benchmarks/attention/profile_attention_share.sh --gpu-name "RTX 4090" --out DIR --cases krea2,flux1-dev [--steps 4]
#       [--cli path/to/HartsyInference.Cli.dll]
#
# A 4090 run takes /tmp/claude-1000/gpu4090.lock, waits for tests/swarm-quiet-window.sh, never stops SwarmUI, needs
# 4 GB of headroom beyond SwarmUI's resident memory, and checks the window again after every run: Swarm activity voids
# the case and stops the script.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$ROOT/tests/regression-cases.sh"
MODELS="${HARTSY_MODELS_ROOT:-/mnt/model-storage/Models}"
GPU_NAME="" OUT="" CASES="" S=4 CLI=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-name) GPU_NAME="$2"; shift 2 ;;  --out) OUT="$2"; shift 2 ;;  --cases) CASES="$2"; shift 2 ;;
    --steps) S="$2"; shift 2 ;;  --cli) CLI="$2"; shift 2 ;;  *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$GPU_NAME" ] && [ -n "$OUT" ] && [ -n "$CASES" ] || { echo "--gpu-name, --out and --cases are required" >&2; exit 2; }
mkdir -p "$OUT"
IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' -v n="$GPU_NAME" 'index($2,n){print $1; exit}')
[ -n "$IDX" ] || { echo "no GPU named '$GPU_NAME'" >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
if [ -z "$CLI" ]; then
  nice -n 19 dotnet build "$ROOT/src/HartsyInference.Cli" -c Release -f net10.0 -v q -nologo -nodeReuse:false >/dev/null
  CLI="$ROOT/src/HartsyInference.Cli/bin/Release/net10.0/HartsyInference.Cli.dll"
fi
IS4090=0; LOCK=/tmp/claude-1000/vk_3060.lock
case "$GPU_NAME" in *4090*) IS4090=1; LOCK=/tmp/claude-1000/gpu4090.lock ;; esac
exec 9>"$LOCK"; flock 9
quiet() { [ "$IS4090" = 0 ] || until "$ROOT/tests/swarm-quiet-window.sh" --gpu 4090 --minutes 10 --state "$OUT/quiet.state" --max-minutes 9; do sleep 5; done; }
verify() { [ "$IS4090" = 0 ] || "$ROOT/tests/swarm-quiet-window.sh" --verify-since "$1" >/dev/null 2>&1; }
quiet
echo "git=$(git -C "$ROOT" rev-parse HEAD) dirty=$(git -C "$ROOT" status --porcelain | wc -l) gpu=$GPU_NAME S=$S utc=$(date -u +%FT%TZ)" > "$OUT/run.txt"

for CASE in ${CASES//,/ }; do
  LINE="$(regression_cases | awk -F'\t' -v c="$CASE" '$1 == c')"
  [ -n "$LINE" ] || { echo "no case $CASE" >&2; continue; }
  IFS=$'\t' read -r _ CKPT SPEC <<<"$LINE"
  IFS='|' read -r CMD POS ARGS <<<"$SPEC"
  [ "$CMD" = text ] && { echo "$CASE is a text case; skipped (prefill/decode are profiled separately)"; continue; }
  case "$CKPT" in /*) MP="$CKPT" ;; *) MP="$MODELS/$CKPT" ;; esac
  BASEARGS="$(printf '%s' "$ARGS" | sed -E 's/--steps [0-9]+//')"
  D="$OUT/$CASE"; mkdir -p "$D"
  for RUN in plain:$((2 * S)) prof:$S prof:$((2 * S)); do
    KIND="${RUN%%:*}"; N="${RUN##*:}"; TAG="$KIND$N"
    PROF=()
    [ "$KIND" = prof ] && PROF=(--set diagnostics.profile=true --set diagnostics.profileSync=true --set diagnostics.profileShapes=true
                               --set "diagnostics.profileOut=$D/$TAG.profile.txt")
    # SwarmUI may keep a small model resident; a large one means a user is about to generate on this card.
    while [ "$IS4090" = 1 ] && [ "$(nvidia-smi -i "$IDX" --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4096 ]; do
      echo "the 4090 holds more than 4 GB; waiting"; sleep 60; quiet
    done
    START=$(date -u +%FT%TZ)
    # shellcheck disable=SC2086
    timeout 3600 dotnet "$CLI" "$CMD" "$POS" --model-path "$MP" -b "cuda:$IDX" -o "$D/out-$TAG" --seed 42 $BASEARGS --steps "$N" \
      "${PROF[@]}" > "$D/$TAG.log" 2>&1 </dev/null || echo "rc=$? $CASE $TAG" >> "$OUT/failures.txt"
    if ! verify "$START"; then echo "Swarm activity during $CASE/$TAG; stopping" | tee -a "$OUT/failures.txt"; exit 4; fi
  done
  echo "$(date +%T) $CASE done"
done
python3 "$ROOT/benchmarks/attention/attention_share.py" "$OUT" --steps "$S" | tee "$OUT/share.md"
