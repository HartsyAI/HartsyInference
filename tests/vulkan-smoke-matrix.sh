#!/usr/bin/env bash
# Per-model Vulkan bring-up smoke matrix.
#
# Answers one question per model, which the op-coverage diff cannot: does it CRASH, does it RUN, and if it runs,
# how far off CUDA is it. Everything not run is reported as untested rather than inferred.
#
# Deliberately serial: two GPU workloads on one card produce contention failures that look like real ones.
#
#   tests/vulkan-smoke-matrix.sh                 # every case with a local checkpoint
#   tests/vulkan-smoke-matrix.sh --filter sdxl   # one case
#   tests/vulkan-smoke-matrix.sh --backend cuda  # same cases on CUDA, to produce the comparison column
#
# Output: a TSV on stdout and a human table on stderr. Feed the TSV to docs/Checklists/VULKAN_STATUS.md.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$REPO/src/HartsyInference.Cli/bin/Release/net10.0/HartsyInference.Cli.dll"
MODELS="${HARTSY_MODELS_ROOT:-/mnt/model-storage/Models}"
OUT="${HARTSY_SMOKE_OUT:-/tmp/vulkan-smoke}"
BACKEND="vulkan"
FILTER=""
TIMEOUT="${HARTSY_SMOKE_TIMEOUT:-900}"

while [ $# -gt 0 ]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        --filter)  FILTER="$2";  shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -f "$CLI" ] || { echo "build the CLI first: dotnet build src/HartsyInference.Cli -c Release -f net10.0" >&2; exit 2; }
mkdir -p "$OUT"

# modality | case id | checkpoint (relative to MODELS, "-" for none) | command|positional|arguments
#
# The positional is the command's own first argument: a prompt for image/text/video, an input FILE for the
# commands that read one (vision, transcribe), or "-" for none. A "@repo/" prefix resolves against the repo so a
# fixture that ships with the tests can be named without an absolute path.
#
# Geometry is the smallest thing that still exercises the real path: a batch>1 CFG image, a multi-frame video, a
# decode long enough to leave the prefill. Anything larger belongs in the benchmarks, not here.
CASES=$(cat <<'MATRIX'
image	sd15	bench-cache/e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916/v1-5-pruned-emaonly-fp16.safetensors	image|a red apple on a table|--steps 4 --width 512 --height 512 --seed 42
image	sdxl	Stable-Diffusion/SDXL/sd_xl_base_1.0.safetensors	image|a red apple on a wooden table|--steps 4 --width 768 --height 768 --seed 42
image	krea2	Stable-Diffusion/Krea2	image|a red apple on a wooden table|--steps 4 --width 1024 --height 1024 --seed 42
image	flux	Stable-Diffusion/Flux	image|a red apple on a wooden table|--steps 4 --width 1024 --height 1024 --seed 42
image	qwenimage	Stable-Diffusion/QwenImage	image|a red apple on a wooden table|--steps 4 --width 1024 --height 1024 --seed 42
image	zimage	Stable-Diffusion/ZImage	image|a red apple on a wooden table|--steps 4 --width 1024 --height 1024 --seed 42
llm	llama32-1b	llm/llama32-1b/llama-3.2-1b-instruct-q8_0.gguf	text|Write one sentence about a robot.|--max-tokens 64
llm	gpt2-medium	llm/gpt2/gpt2-medium.Q4_K_M.gguf	text|Once upon a time|--max-tokens 64
video	wan	Stable-Diffusion/Wan/wan2.2_ti2v_5B_fp16.safetensors	video|a cat walking|-m wan --width 256 --height 256 --frames 5 --steps 4
video	ltx-video-2	Stable-Diffusion/LtxVideo2	video|a cat walking|-m ltx-video-2 --width 256 --height 256 --frames 9 --steps 4
vision	rtdetr	rtdetr/rtdetr_r18vd.safetensors	vision|@repo/tests/HartsyInference.Vision.Tests/TestData/bus.png|--mode detect
audio	whisper	-	transcribe|@repo/tests/python-reference/silerovad_reference/jfk.wav|-m whisper:base
MATRIX
)

printf 'modality\tcase\tbackend\tstatus\tseconds\tdetail\n'
printf '%-9s %-14s %-10s %9s  %s\n' "MODALITY" "CASE" "STATUS" "SECONDS" "DETAIL" >&2

while IFS=$'\t' read -r modality id ckpt spec; do
    [ -z "${modality:-}" ] && continue
    [ -n "$FILTER" ] && [[ "$id" != *"$FILTER"* ]] && continue

    cmd="${spec%%|*}"; rest="${spec#*|}"; prompt="${rest%%|*}"; args="${rest#*|}"
    [ "$args" = "-" ] && args=""
    path=""
    if [ "$ckpt" != "-" ]; then
        path="$MODELS/$ckpt"
        if [ ! -e "$path" ]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$modality" "$id" "$BACKEND" "untested" "0" "no local checkpoint at $ckpt"
            printf '%-9s %-14s %-10s %9s  %s\n' "$modality" "$id" "untested" "-" "no local checkpoint" >&2
            continue
        fi
    fi

    case "$prompt" in
        @repo/*)
            prompt="$REPO/${prompt#@repo/}"
            if [ ! -e "$prompt" ]; then
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$modality" "$id" "$BACKEND" "untested" "0" "missing input fixture $prompt"
                printf '%-9s %-14s %-10s %9s  %s\n' "$modality" "$id" "untested" "-" "missing input fixture" >&2
                continue
            fi
            ;;
    esac

    log="$OUT/$id.$BACKEND.log"
    start=$(date +%s.%N)
    # shellcheck disable=SC2086
    if [ "$prompt" = "-" ]; then
        timeout "$TIMEOUT" dotnet "$CLI" "$cmd" ${path:+--model-path "$path"} $args -b "$BACKEND" -o "$OUT/$id.$BACKEND" >"$log" 2>&1
    else
        timeout "$TIMEOUT" dotnet "$CLI" "$cmd" "$prompt" ${path:+--model-path "$path"} $args -b "$BACKEND" -o "$OUT/$id.$BACKEND" >"$log" 2>&1
    fi
    rc=$?
    secs=$(printf '%.1f' "$(echo "$(date +%s.%N) - $start" | bc)")

    if [ $rc -eq 124 ]; then
        status="timeout"; detail="exceeded ${TIMEOUT}s"
    elif [ $rc -ne 0 ]; then
        status="crash"
        # The first exception type is the useful half; the stack is in the log.
        detail=$(grep -oE '[A-Za-z.]*(Exception|Error)[^:]*: [^\n]{0,120}' "$log" | head -1)
        [ -z "$detail" ] && detail="exit $rc (see $log)"
    else
        status="ran"; detail=$(basename "$log")
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$modality" "$id" "$BACKEND" "$status" "$secs" "$detail"
    printf '%-9s %-14s %-10s %9s  %s\n' "$modality" "$id" "$status" "$secs" "${detail:0:80}" >&2
done <<< "$CASES"
