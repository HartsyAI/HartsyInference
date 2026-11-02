#!/usr/bin/env bash
# Runs one test project on one card, chosen by name, with both backends pinned to it — CUDA tests construct
# ordinal 0 and Vulkan tests honour HARTSY_TEST_VULKAN_DEVICE, and on this box ordinal 0 of either is the 4090 that
# SwarmUI lives on. Takes the card's lock so GPU suites never overlap, and writes a TRX so failures are kept by name.
#
#   tests/gpu-suite.sh --gpu-name "RTX 3060" --project Cuda [--filter "Category=GpuIntegration"] [--manifest FILE]
#                      [--out DIR] [--no-build]
#
# --project is the middle of tests/HartsyInference.<Project>.Tests. The default filter is the GpuIntegration category,
# or exactly the --manifest tests when one is given. The run goes through tests/require-executed.sh, so a SKIPPED line,
# a backend guard, an empty run or a manifest test that did not execute fails it.
# The 4090 lock is only the turn-taking lock: a 4090 suite still needs SwarmUI quiet (tests/swarm-quiet-window.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_NAME="" PROJECT="" FILTER="" MANIFEST="" OUT="" BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-name) GPU_NAME="$2"; shift 2 ;;  --project) PROJECT="$2"; shift 2 ;;  --filter) FILTER="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;  --out) OUT="$2"; shift 2 ;;  --no-build) BUILD=0; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$FILTER$MANIFEST" ] || FILTER="Category=GpuIntegration"
[ -n "$GPU_NAME" ] && [ -n "$PROJECT" ] || { echo "--gpu-name and --project are required" >&2; exit 2; }
PROJ="$ROOT/tests/HartsyInference.$PROJECT.Tests"
[ -d "$PROJ" ] || { echo "no project $PROJ" >&2; exit 2; }
IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' -v n="$GPU_NAME" 'index($2,n){print $1; exit}')
[ -n "$IDX" ] || { echo "no CUDA device named '$GPU_NAME'" >&2; exit 2; }
VKIDX=$(vulkaninfo --summary 2>/dev/null | awk -v n="$GPU_NAME" '/^GPU[0-9]+:/{g=substr($1,4)+0} /deviceName/ && index($0,n){print g; exit}')
[ -n "$VKIDX" ] || { echo "no Vulkan device named '$GPU_NAME'" >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$IDX" HARTSY_TEST_VULKAN_DEVICE="$VKIDX"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
OUT="${OUT:-${HARTSY_REGRESSION_OUT:-$HOME/Desktop/regression-ab}/gpu-suite/$(date -u +%Y%m%dT%H%M%SZ)-$PROJECT}"
mkdir -p "$OUT"
[ "$BUILD" = 1 ] && nice -n 19 dotnet build "$PROJ" -c Release -v q -nologo -nodeReuse:false > "$OUT/build.log"
LOCK=/tmp/claude-1000/vk_3060.lock
case "$GPU_NAME" in *4090*) LOCK=/tmp/claude-1000/gpu4090.lock ;; esac
exec 9>"$LOCK"; flock 9
set +e
"$ROOT/tests/require-executed.sh" --project "$PROJ" ${FILTER:+--filter "$FILTER"} ${MANIFEST:+--manifest "$MANIFEST"} \
  --out "$OUT" --name "$PROJECT" -- -c Release --no-build --blame-hang-timeout 15m
rc=$?
set -e
echo "trx: $OUT/$PROJECT.trx"
exit $rc
