#!/usr/bin/env bash
# Attention shape table on one card: one process per (shape, arm), each sampled for peak VRAM and throttling.
#
#   benchmarks/attention/run_attention_table.sh --gpu-name "RTX 3060" --out DIR [--backend cuda|vulkan] [--shapes FILE]
#       [--only id,id] [--arms a,b] [--warmup 3] [--trials 20] [--cool-to 55] [--no-build]
#
# Rows land in DIR/table.tsv (runner columns + peak_vram_mb, throttle_frac, sm_clock_med). The card is chosen by
# name in PCI order, so the mask names the card nvidia-smi names. The 3060 run takes /tmp/claude-1000/vk_3060.lock;
# a 4090 run takes /tmp/claude-1000/gpu4090.lock and then requires tests/swarm-quiet-window.sh to pass (SwarmUI lives
# there; this script never stops it) and re-verifies the window after every arm, aborting on foreign activity.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHAPES="$ROOT/benchmarks/python-baseline/attention_shapes.json"
GPU_NAME="" OUT="" ONLY="" ARMS="" WARMUP=3 TRIALS=20 COOL=55 BUILD=1 BACKEND=cuda
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-name) GPU_NAME="$2"; shift 2 ;;  --out) OUT="$2"; shift 2 ;;  --shapes) SHAPES="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;  --arms) ARMS="$2"; shift 2 ;;  --warmup) WARMUP="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;  --cool-to) COOL="$2"; shift 2 ;;  --no-build) BUILD=0; shift ;;
    --backend) BACKEND="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$GPU_NAME" ] && [ -n "$OUT" ] || { echo "--gpu-name and --out are required" >&2; exit 2; }
mkdir -p "$OUT/arms"
IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' -v n="$GPU_NAME" 'index($2,n){print $1; exit}')
[ -n "$IDX" ] || { echo "no GPU named '$GPU_NAME'" >&2; exit 2; }
UUID=$(nvidia-smi -i "$IDX" --query-gpu=uuid --format=csv,noheader)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$IDX" DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
if [ "$BACKEND" = vulkan ]; then
  VKIDX=$(vulkaninfo --summary 2>/dev/null | awk -v n="$GPU_NAME" '/^GPU[0-9]+:/{g=substr($1,4)+0} /deviceName/ && index($0,n){print g; exit}')
  [ -n "$VKIDX" ] || { echo "no Vulkan device named '$GPU_NAME'" >&2; exit 2; }
  export HARTSYINFERENCE_BENCH_BACKEND=vulkan HARTSYINFERENCE_BENCH_VULKAN_DEVICE="$VKIDX"
fi
BIN="$ROOT/benchmarks/HartsyInference.GpuBenchmarks/bin/Release/net10.0/HartsyInference.GpuBenchmarks.dll"
if [ "$BUILD" = 1 ]; then nice -n 19 dotnet build "$ROOT/benchmarks/HartsyInference.GpuBenchmarks" -c Release -v q -nologo -nodeReuse:false >/dev/null; fi

IS4090=0; LOCK=/tmp/claude-1000/vk_3060.lock
case "$GPU_NAME" in *4090*) IS4090=1; LOCK=/tmp/claude-1000/gpu4090.lock ;; esac
exec 9>"$LOCK"; flock 9
if [ "$IS4090" = 1 ]; then
  until "$ROOT/tests/swarm-quiet-window.sh" --gpu 4090 --minutes 10 --state "$OUT/quiet.state" --max-minutes 9; do sleep 5; done
fi

{ echo "utc_start=$(date -u +%FT%TZ)"; echo "git=$(git -C "$ROOT" rev-parse HEAD)"; echo "dirty=$(git -C "$ROOT" status --porcelain | wc -l)"
  nvidia-smi -i "$IDX" --query-gpu=name,uuid,driver_version,vbios_version,pci.bus_id,power.limit,clocks.max.sm,memory.total --format=csv
  ls -1 "$HOME/.local/lib/cuda13" | grep -E 'libcudnn\.so\.[0-9.]+$|libcublas\.so\.[0-9.]+$' || true
  dotnet --version; uname -r; } > "$OUT/hardware_software.txt"

PAIRS=$(python3 - "$SHAPES" "$ONLY" "$ARMS" "$BACKEND" <<'PY'
import json, sys
shapes = json.load(open(sys.argv[1]))["shapes"]
only = set(filter(None, sys.argv[2].split(",")))
arms = {"sdpa": ["def-f16","cudnn-f16","sage-f16","flash-f16","mat-f16","def-f32","tiled-f32","cast-f16","bf16"],
        "flash": ["fa-def","fa-f16kv","fa-nocudnn","fa-nofa2","fa-nogqa","fa-splitoff","fa-splitforce"]}
if sys.argv[4] == "vulkan":   # the knob arms steer CUDA dispatch only
    arms = {"sdpa": ["def-f16","def-f32","cast-f16","bf16"], "flash": ["fa-def","fa-f16kv"]}
want = set(filter(None, sys.argv[3].split(",")))
for s in shapes:
    if only and s["id"] not in only: continue
    for a in arms[s["entry"]]:
        if not want or a in want: print(s["id"], a)
PY
)
TABLE="$OUT/table.tsv"
# PIDs of swarmui.service and every descendant (its in-process HartsyInference backend and any ComfyUI child).
swarm_tree() {
  local root; root=$(systemctl --user show -p MainPID --value swarmui.service 2>/dev/null || echo 0)
  [ "${root:-0}" -gt 0 ] || return 0
  ps -eo pid=,ppid= | awk -v r="$root" '{p[$1]=$2} END{for (k in p){x=k; while (x>1){if (x==r){print k; break}; x=p[x]}}}'
}
# The desktop's own GPU clients (display server, remote desktop, browser) hold a context on the display card for good;
# the same list tests/swarm-quiet-window.sh allows.
desktop_pids() { pgrep -x 'Xorg|rustdesk|cinnamon|gnome-shell|chrome' || true; }
foreign_on_card() {
  local allowed; allowed=" $(swarm_tree | tr '\n' ' ') $(desktop_pids | tr '\n' ' ') "
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
    | awk -F', ' -v u="$UUID" -v ok="$allowed" '$1==u && index(ok, " "$2" ")==0 {f=1} END{exit !f}'
}
while read -r SHAPE ARM; do
  [ -n "$SHAPE" ] || continue
  # Cool the card between arms: back-to-back runs on the 3060 lose double-digit SM clock to heat soak.
  for _ in $(seq 1 60); do t=$(nvidia-smi -i "$IDX" --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -le "$COOL" ] && break; sleep 2; done
  # Hard gate: nothing but SwarmUI's own process tree may hold this card.
  while foreign_on_card; do echo "foreign process on $GPU_NAME; waiting"; sleep 30; done
  BASE=$(nvidia-smi -i "$IDX" --query-gpu=memory.used --format=csv,noheader,nounits)
  STEM="$OUT/arms/$SHAPE.$ARM"; START=$(date -u +%FT%TZ)
  nvidia-smi -i "$IDX" --query-gpu=memory.used,clocks.sm,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown \
    --format=csv,noheader,nounits -lms 100 > "$STEM.smi" 2>/dev/null & SMI=$!
  rm -f "$STEM.tsv"
  timeout 1800 dotnet "$BIN" attention-table --shapes "$SHAPES" --shape "$SHAPE" --arm "$ARM" --out "$STEM.tsv" \
    --warmup "$WARMUP" --trials "$TRIALS" > "$STEM.log" 2>&1 || echo "rc=$? $SHAPE $ARM" >> "$OUT/failures.txt"
  kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null || true
  EXTRA=$(python3 - "$STEM.smi" "$BASE" <<'PY'
import sys, statistics
rows = [l.strip().split(", ") for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if len(r) == 6 and r[0].isdigit()]
if not rows: print("NA\tNA\tNA"); sys.exit()
peak = max(int(r[0]) for r in rows) - int(sys.argv[2])
thr = sum(1 for r in rows if "Active" in r[2:]) / len(rows)
print(f"{peak}\t{thr:.3f}\t{statistics.median(int(r[1]) for r in rows)}")
PY
)
  if [ -f "$STEM.tsv" ]; then
    [ -f "$TABLE" ] || { head -1 "$STEM.tsv" | sed 's/$/\tpeak_vram_mb\tthrottle_frac\tsm_clock_med/' > "$TABLE"; }
    tail -n +2 "$STEM.tsv" | sed "s/\$/\t$EXTRA/" >> "$TABLE"
  fi
  if [ "$IS4090" = 1 ] && ! "$ROOT/tests/swarm-quiet-window.sh" --verify-since "$START" >/dev/null 2>&1; then
    echo "Swarm activity during $SHAPE/$ARM; rows from it are void, stopping" | tee -a "$OUT/failures.txt"; exit 4
  fi
  echo "$(date +%T) $SHAPE $ARM: $(tail -n +2 "$STEM.tsv" 2>/dev/null | cut -f5,6,7,21,26 | tr '\n' ' ') [$EXTRA]"
done <<< "$PAIRS"
echo "done: $TABLE"
