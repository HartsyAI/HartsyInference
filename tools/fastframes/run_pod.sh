#!/usr/bin/env bash
# FastFrames GPU runs for a rented card. See docs/Checklists/FASTFRAMES_GPU_RUNBOOK.md for the weights, the hardware and what to return.
#
#   run_pod.sh build                  build the CLI (net10.0; the apphost name differs by OS, so the CLI is run as a .dll)
#   run_pod.sh verify                 check every weight's byte count and sha256 under $MODELS
#   run_pod.sh decoder                decoder GPU parity, exact and default lanes
#   run_pod.sh streamed               streamed-vs-resident CUDA test
#   run_pod.sh clip-s                 S, 81 frames, 832x480, seed 1 (24 GB card is enough)
#   run_pod.sh clip-l                 L, 81 frames; refuses unless the card reports at least 79 GB (see the runbook)
#   run_pod.sh all                    build, verify, decoder, streamed, clip-s, clip-l, in order; stops at the first failure
#
# Environment: REPO (checkout, default: the current directory), MODELS (models root, required), OUT (output folder).
# The engine reads the models root from HARTSYINFERENCE_MODELS, so this script exports it from MODELS: without that the CLI looks in
# <repo>/Models and downloads the weights again.
set -u

REPO="${REPO:-$(pwd)}"
MODELS="${MODELS:?set MODELS to the models root}"
OUT="${OUT:-$REPO/Output/fastframes-run}"
CLI_DLL="$REPO/src/HartsyInference.Cli/bin/Release/net10.0/HartsyInference.Cli.dll"
PROMPT="a cat walking through a sunlit garden"
FF="$MODELS/FastFrames"
export HARTSYINFERENCE_MODELS="$MODELS"

mkdir -p "$OUT"
# Stop any nvidia-smi sampler left running when the script exits or is interrupted.
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT TERM

# path|bytes|sha256 (the same values as ModelCatalog.cs and SideModels.cs; keep them in step with those files)
WEIGHTS=(
  "$FF/fastframes-s/framewise-2step.pt|5676220819|8f899a016dffb9539788fac82c6200e43ef6e23b6e61ec6b885bb507a760ce42"
  "$FF/fastframes-s/kv_refiner.safetensors|692418448|b4f36d00effd78a60435526dcc2b1a0e6bbed0f09621ee8ab4019460769a7bbb"
  "$FF/fastframes-s/pixel_decoder.safetensors|255494944|815682581f6376d2cdacc60568facf1bd979112091e4157fb6c00ae946b89b32"
  "$FF/fastframes-l/dit.safetensors|28577096680|cd0ec8980516663d623186fafe4b2a3ca9398e7ed29ccce1c65581640bef15de"
  "$FF/fastframes-l/kv_refiner.safetensors|1780342784|841047b9226cf770ca19f99ef718edcb57db29c29ee58a64f9ae5dc1d6fbdf74"
  "$FF/fastframes-l/pixel_decoder.safetensors|286952688|75e6c99c358a3d8b87476add2c732b0853aef76279bae4c807a6157e95555f55"
  "$MODELS/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors|6735906897|c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68"
)

verify() {
  local failed=0
  for entry in "${WEIGHTS[@]}"; do
    IFS="|" read -r path bytes sha <<< "$entry"
    if [ ! -f "$path" ]; then echo "MISSING $path"; failed=1; continue; fi
    got_bytes=$(stat -c %s "$path")
    if [ "$got_bytes" != "$bytes" ]; then echo "SIZE $path: $got_bytes, expected $bytes"; failed=1; continue; fi
    got_sha=$(sha256sum "$path" | cut -c1-64)
    if [ "$got_sha" != "$sha" ]; then echo "SHA256 $path"; failed=1; continue; fi
    echo "ok $path"
  done
  return $failed
}

# The card's total memory in MiB (the first GPU).
card_mib() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' '; }

run_logged() {
  local name="$1"; shift
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$OUT/$name.nvidia-smi.csv" 2>&1 &
  local start; start=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1
  local status=$?
  kill $! 2>/dev/null
  echo "$name exit $status after $(( $(date +%s) - start )) s (log: $OUT/$name.log)"
  return $status
}

build() {
  (cd "$REPO" && dotnet build src/HartsyInference.Cli/HartsyInference.Cli.csproj -c Release -f net10.0 > "$OUT/build.log" 2>&1)
  local status=$?
  echo "build exit $status (log: $OUT/build.log)"
  return $status
}

decoder() {
  (cd "$REPO" && run_logged decoder env FF_GPU_TESTS=1 \
    dotnet test tests/HartsyInference.Engine.Tests -c Release --filter FullyQualifiedName~FastFramesDecoderGpuParityTests)
}

streamed() {
  (cd "$REPO" && run_logged streamed env FF_GPU_TESTS=1 \
    dotnet test tests/HartsyInference.Diffusion.Tests -c Release --filter FullyQualifiedName~TheStreamedRolloutMatchesTheResidentOneOnCuda)
}

clip() {
  local model="$1"
  if [ "$model" = "l" ]; then
    local mib; mib=$(card_mib)
    if [ -z "$mib" ] || [ "$mib" -lt 79000 ]; then
      echo "clip-l refused: the card reports ${mib:-no} MiB; L needs an 80 GB card (see the runbook)"
      return 3
    fi
  fi
  run_logged "clip-$model" dotnet "$CLI_DLL" video "$PROMPT" -m "fastframes-$model" --frames 81 --seed 1 --gpu 0 -o "$OUT/clip-$model"
}

# In `all`, L runs only on a card that can hold it; on a smaller card it is skipped with a message, and the run is not a failure.
all_l() {
  local mib; mib=$(card_mib)
  if [ -n "$mib" ] && [ "$mib" -ge 79000 ]; then clip l; else echo "clip-l skipped: the card reports ${mib:-no} MiB (L needs 80 GB)"; fi
}

case "${1:-}" in
  build) build ;;
  verify) verify ;;
  decoder) decoder ;;
  streamed) streamed ;;
  clip-s) clip s ;;
  clip-l) clip l ;;
  all)
    build && verify && decoder && streamed && clip s && all_l ;;
  *) echo "usage: $0 {build|verify|decoder|streamed|clip-s|clip-l|all}"; exit 2 ;;
esac
