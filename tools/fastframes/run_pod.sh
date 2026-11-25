#!/usr/bin/env bash
# FastFrames GPU runs for a rented card. See docs/Checklists/FASTFRAMES_GPU_RUNBOOK.md for the weights, the hardware and what to return.
#
#   run_pod.sh verify                 check every weight's byte count and sha256 under $MODELS
#   run_pod.sh decoder                decoder GPU parity, exact and default lanes
#   run_pod.sh streamed               streamed-vs-resident CUDA test
#   run_pod.sh clip-s                 S, 81 frames, 832x480, seed 1
#   run_pod.sh clip-l                 L, 81 frames (80 GB card; the CLI may refuse fastframes-l, see the runbook)
#   run_pod.sh all                    verify, then every run above in order
#
# Environment: REPO (checkout), MODELS (models root), OUT (output folder), CLI (path to HartsyInference.Cli).
# Each run writes its log under $OUT and a nvidia-smi sample every 2 s beside it.
set -u

REPO="${REPO:-$(pwd)}"
MODELS="${MODELS:?set MODELS to the models root}"
OUT="${OUT:-$REPO/Output/fastframes-run}"
CLI="${CLI:-$REPO/src/HartsyInference.Cli/bin/Release/net10.0/HartsyInference.Cli.exe}"
PROMPT="a cat walking through a sunlit garden"
FF="$MODELS/FastFrames"

mkdir -p "$OUT"

# path|bytes|sha256
WEIGHTS=(
  "$FF/fastframes-s/framewise-2step.pt|5676220819|8f899a016dffb9539788fac82c6200e43ef6e23b6e61ec6b885bb507a760ce42"
  "$FF/fastframes-s/kv_refiner.safetensors|692418448|b4f36d00effd78a60435526dcc2b1a0e6bbed0f09621ee8ab4019460769a7bbb"
  "$FF/fastframes-s/pixel_decoder.safetensors|255494944|815682581f6376d2cdacc60568facf1bd979112091e4157fb6c00ae946b89b32"
  "$FF/fastframes-l/dit.safetensors|28577096680|cd0ec8980516663d623186fafe4b2a3ca9398e7ed29ccce1c65581640bef15de"
  "$FF/fastframes-l/kv_refiner.safetensors|1780342784|841047b9226cf770ca19f99ef718edcb57db29c29ee58a64f9ae5dc1d6fbdf74"
  "$FF/fastframes-l/pixel_decoder.safetensors|286952688|75e6c99c358a3d8b87476add2c732b0853aef76279bae4c807a6157e95555f55"
)
UMT5_SHA256=c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68
UMT5="$MODELS/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

verify() {
  local failed=0
  for entry in "${WEIGHTS[@]}"; do
    IFS="|" read -r path bytes sha <<< "$entry"
    if [ ! -f "$path" ]; then echo "MISSING $path"; failed=1; continue; fi
    got_bytes=$(stat -c %s "$path" 2>/dev/null || wc -c < "$path")
    if [ "$got_bytes" != "$bytes" ]; then echo "SIZE $path: $got_bytes, expected $bytes"; failed=1; continue; fi
    got_sha=$(sha256sum "$path" | cut -c1-64)
    if [ "$got_sha" != "$sha" ]; then echo "SHA256 $path"; failed=1; continue; fi
    echo "ok $path"
  done
  if [ ! -f "$UMT5" ]; then echo "MISSING $UMT5"; failed=1
  elif [ "$(sha256sum "$UMT5" | cut -c1-64)" != "$UMT5_SHA256" ]; then echo "SHA256 $UMT5"; failed=1
  else echo "ok $UMT5"; fi
  return $failed
}

sampler() {
  # One nvidia-smi sample every 2 s until the run that started it finishes.
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$1" 2>&1 &
  echo $!
}

run_logged() {
  local name="$1"; shift
  local sample; sample=$(sampler "$OUT/$name.nvidia-smi.csv")
  local start; start=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1
  local status=$?
  kill "$sample" 2>/dev/null
  echo "$name exit $status after $(( $(date +%s) - start )) s (log: $OUT/$name.log)"
  return $status
}

decoder() {
  cd "$REPO" && run_logged decoder env FF_GPU_TESTS=1 \
    dotnet test tests/HartsyInference.Engine.Tests -c Release --filter FullyQualifiedName~FastFramesDecoderGpuParityTests
}

streamed() {
  cd "$REPO" && run_logged streamed env FF_GPU_TESTS=1 \
    dotnet test tests/HartsyInference.Diffusion.Tests -c Release --filter FullyQualifiedName~TheStreamedRolloutMatchesTheResidentOneOnCuda
}

clip() {
  local model="$1"
  run_logged "clip-$model" "$CLI" video "$PROMPT" -m "fastframes-$model" --frames 81 --seed 1 --gpu 0 -o "$OUT/clip-$model"
}

case "${1:-}" in
  verify) verify ;;
  decoder) decoder ;;
  streamed) streamed ;;
  clip-s) clip s ;;
  clip-l) clip l ;;
  all)
    verify || { echo "verify failed; stopping"; exit 1; }
    decoder; streamed; clip s; clip l ;;
  *) echo "usage: $0 {verify|decoder|streamed|clip-s|clip-l|all}"; exit 2 ;;
esac
