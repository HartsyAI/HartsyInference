#!/usr/bin/env bash
# cuDNN 9.24 SDPA capability probes, attention plan Phase 0 item 6 (probe plan: cuDNN graph research report §3).
# Each probe is its own `dotnet test` process in its own process group under a watchdog, so a JIT failure, a hang or
# a sticky CUDA error stays inside that probe and a killed probe cannot keep the card. The card is chosen by name and
# pinned the way tests/gpu-suite.sh pins it; the tests themselves fail (not skip) when the device, SM, cuDNN version or
# library directory is not the one asked for.
#
#   tests/cudnn-probes.sh --gpu-name "RTX 3060" [--probes "P0 P15 P1"] [--timeout SECONDS] [--out DIR] [--no-build] [--dry-run]
#
# Default order is the report's run order; P8 (paged KV) is deferred and its test is marked Skip. Output: one JSON line
# per probe in <out>/probes.jsonl, per-mode rows in <out>/probes.jsonl.tsv, one log per probe in <out>/logs/, and
# <out>/summary.txt. The default <out> is under benchmarks/results/, which is gitignored: `git add -f` to keep it.
# A probe is PASS only when its one test ran and passed and its JSON line exists; skipped, timed out or crashed is FAIL.
# Locks: the 3060 or 4090 turn-taking lock, as gpu-suite.sh takes; on the 4090 the run then waits for
# tests/swarm-quiet-window.sh. HARTSY_CUDNN_PROBE_VERSION and HARTSY_CUDNN_PROBE_LIBDIR pass through when set.
# Exit codes: 0 every probe PASS, 1 a probe did not pass, 2 preflight or build failed, 64 bad usage.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="$ROOT/tests/HartsyInference.Cuda.Tests"
CLASS="HartsyInference.Cuda.Tests.CudnnSdpaCapabilityProbeTests"
GPU_NAME="" TIMEOUT=1800 OUT="" BUILD=1 DRY=0
PROBES="P0 P15 P1 P2 P3 P4 P5 P6 P12 P11 P10 P9 P7 P13 P14"
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-name) GPU_NAME="$2"; shift 2 ;;  --probes) PROBES="$2"; shift 2 ;;  --timeout) TIMEOUT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;  --no-build) BUILD=0; shift ;;  --dry-run) DRY=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 64 ;;
  esac
done
[ -n "$GPU_NAME" ] || { echo "--gpu-name is required" >&2; exit 64; }
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || { echo "--timeout takes whole seconds" >&2; exit 64; }
for p in $PROBES; do [[ "$p" =~ ^P([0-9]|1[0-5])$ ]] || { echo "unknown probe $p (P0..P15)" >&2; exit 64; }; done

IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' -v n="$GPU_NAME" 'index($2,n){print $1; exit}')
[ -n "$IDX" ] || { echo "no CUDA device named '$GPU_NAME'" >&2; exit 2; }
SM=$(nvidia-smi -i "$IDX" --query-gpu=compute_cap --format=csv,noheader | tr -d ' .')
CARD=$(grep -oE '[0-9]{4}' <<<"$GPU_NAME" | head -1)
CARD="${CARD:-${GPU_NAME//[^A-Za-z0-9]/_}}"
OUT="${OUT:-$ROOT/benchmarks/results/$(date -u +%Y-%m-%d)_attention_phase0/$CARD/cudnn-probes-$(date -u +%H%M%SZ)}"
JSONL="$OUT/probes.jsonl"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$IDX"
export HARTSY_CUDNN_PROBE_GPU="$GPU_NAME" HARTSY_CUDNN_PROBE_SM="$SM" HARTSY_CUDNN_PROBE_OUT="$JSONL"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1

method_of() { printf 'P%02d_' "$((10#${1#P}))"; }

if [ "$DRY" = 1 ]; then
  echo "card '$GPU_NAME': nvidia-smi index $IDX, SM $SM; output $OUT"
  for p in $PROBES; do
    echo "  dotnet test tests/HartsyInference.Cuda.Tests -c Release --no-build --filter \"FullyQualifiedName~$CLASS.$(method_of "$p")\""
  done
  exit 0
fi

mkdir -p "$OUT/logs"
if [ "$BUILD" = 1 ]; then
  nice -n 19 dotnet build "$PROJ" -c Release -v q -nologo >"$OUT/build.log" 2>&1 || { echo "build failed, see $OUT/build.log" >&2; exit 2; }
fi
LOCK=/tmp/claude-1000/vk_3060.lock
case "$GPU_NAME" in *4090*) LOCK=/tmp/claude-1000/gpu4090.lock ;; esac
exec 9>"$LOCK"; flock 9
if [[ "$GPU_NAME" == *4090* ]]; then
  "$ROOT/tests/swarm-quiet-window.sh" --gpu "$GPU_NAME" || { echo "SwarmUI was not quiet on the 4090; nothing ran" >&2; exit 2; }
fi
FREE=$(nvidia-smi -i "$IDX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[ "${FREE:-0}" -ge 4096 ] || { echo "only ${FREE:-0} MiB free on $GPU_NAME; the probes want 4096" >&2; exit 2; }

SUMMARY="$OUT/summary.txt"
echo "cuDNN SDPA probes on '$GPU_NAME' (nvidia-smi index $IDX, SM $SM), timeout ${TIMEOUT}s per probe" | tee "$SUMMARY"
FAILED=0
record() { printf '%-4s %-8s %s\n' "$1" "$2" "$3" | tee -a "$SUMMARY"; [ "$2" = PASS ] || FAILED=1; }

run_probe() { # probe id
  local id="$1" method log pid rc waited=0 timed_out=0 before after runs total failed skipped
  method="$(method_of "$id")"
  log="$OUT/logs/$id.log"
  before=$(grep -c "\"probe\":\"$id\"" "$JSONL" 2>/dev/null || true)
  setsid dotnet test "$PROJ" -c Release --no-build --filter "FullyQualifiedName~$CLASS.$method" \
    --logger "console;verbosity=detailed" >"$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$TIMEOUT" ]; then
      timed_out=1
      kill -TERM -- "-$pid" 2>/dev/null; sleep 20; kill -KILL -- "-$pid" 2>/dev/null
      break
    fi
    sleep 5; waited=$((waited + 5))
  done
  wait "$pid"; rc=$?
  after=$(grep -c "\"probe\":\"$id\"" "$JSONL" 2>/dev/null || true)
  read -r runs total failed skipped < <(awk -f "$ROOT/tests/dsv41-certification-summary.awk" "$log")
  if [ "$timed_out" = 1 ]; then record "$id" TIMEOUT "killed after ${TIMEOUT}s; see logs/$id.log"
  elif [ "$runs" -eq 0 ]; then record "$id" FAIL "no test summary (exit $rc); see logs/$id.log"
  elif [ "$total" -eq 0 ]; then record "$id" FAIL "no test matched $CLASS.$method"
  elif [ "$skipped" -ne 0 ]; then record "$id" FAIL "skipped: a skip is not a pass ($(grep -m1 -o 'cuDNN capability probe[^\"]*\|deferred[^\"]*' "$log" | cut -c1-120))"
  elif [ "${after:-0}" -le "${before:-0}" ]; then record "$id" FAIL "no JSON line written (exit $rc); see logs/$id.log"
  elif [ "$failed" -ne 0 ] || [ "$rc" -ne 0 ]; then record "$id" FAIL "$(grep -m1 -E "$id failed|environment rejected|Exception" "$log" | sed 's/^ *//' | cut -c1-160)"
  else record "$id" PASS "1 test passed"
  fi
}

for p in $PROBES; do run_probe "$p"; done

{
  echo
  echo "Per spec (outcome: supported = some mode ran correct, unsupported = every mode refused, WRONG = finalized and ran wrong):"
  printf '%-4s %-40s %-10s %s\n' probe spec expect outcome
  [ -f "$JSONL.tsv" ] && awk -F'\t' '!seen[$1 FS $2]++ { printf "%-4s %-40s %-10s %s\n", $1, $2, $3, $4 }' "$JSONL.tsv"
  echo
  echo "Per heuristic mode (relL2 vs the float64 reference; build = heuristic + finalize ms; exec = median of 10, ms):"
  printf '%-4s %-40s %-8s %-8s %-12s %-9s %-9s %-9s %-7s %s\n' probe spec mode scale verdict relL2 buildMs execMs engine error
  [ -f "$JSONL.tsv" ] && awk -F'\t' '{ printf "%-4s %-40s %-8s %-8s %-12s %-9s %-9s %-9s %-7s %.90s\n", $1, $2, $5, $6, $7, $8, $10, $11, $12, $13 }' "$JSONL.tsv"
  echo
  echo "JSON: $JSONL   logs: $OUT/logs/"
} | tee -a "$SUMMARY"

if [ "$FAILED" -eq 0 ]; then echo "GREEN: every probe passed. Summary: $SUMMARY"; exit 0; fi
echo "RED: at least one probe did not pass. Summary: $SUMMARY"
exit 1
