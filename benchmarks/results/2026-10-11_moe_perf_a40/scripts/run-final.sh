#!/usr/bin/env bash
# Interleaved llama.cpp vs HartsyInference matrix, same protocol as the baseline runs (3 rounds, temps 0.0 and 0.7).
B=/root/baseline/scripts
R=/root/baseline/results
rm -f $R/final.status
for spec in "granite-3b 4k" "olmoe 4k" "qwen3-30b 4k,32k" "mixtral 4k,32k"; do
  set -- $spec
  m=$1; cells=$2
  bash $B/run-cell.sh $m final $cells 3 final-$m "llama-cpp-gpu,hartsy-cuda,hartsy-cuda-kvf16"
  rc=$?
  echo "$m exit=$rc $(date -u +%T)" >> $R/final.progress
  [ $rc -eq 0 ] || failed="$failed $m"
done
if [ -z "${failed:-}" ]; then echo ALL_DONE > $R/final.status; else echo "FAILED:$failed" > $R/final.status; fi
