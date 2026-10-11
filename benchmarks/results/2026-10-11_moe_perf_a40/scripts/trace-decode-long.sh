#!/usr/bin/env bash
# Kernel trace of a CLI generation over the 4K benchmark prompt: trace-decode-long.sh <model gguf> <out csv> <variant home> <prompt file> [extra cli args]
export DOTNET_ROOT=/root/dotnet PATH=/root/dotnet:$PATH HOME=/root/baseline/home-$3
M=$1; OUT=$2; PF=$4; shift 4
CUPTI_TRACE_OUT=$OUT LD_PRELOAD=/root/baseline/scripts/libcuptitrace.so \
  dotnet /root/baseline/engine/cli-perf5/HartsyInference.Cli.dll text --model-path $M --device cuda --max-tokens 48 --temperature 0 "$@" "$(cat $PF)" > $OUT.log 2>&1
echo "exit=$?" >> $OUT.log
