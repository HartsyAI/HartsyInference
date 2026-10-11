#!/usr/bin/env bash
# One harness run: run-cell.sh <model> <variant> <cells> <rounds> <outname> [configs]   (configs default: hartsy-cuda)
M=$1; V=$2; CELLS=$3; ROUNDS=$4; OUT=$5; CONFIGS=${6:-hartsy-cuda}
P=/root/baseline/prompts
O=/root/baseline/results/$OUT
case $M in
  granite-3b|olmoe) P4=$P/prompt-4k-short.txt; P32=$P/prompt-32k.txt ;;
  mixtral) P4=$P/prompt-4k.txt; P32=$P/prompt-32k-mixtral.txt ;;
  *) P4=$P/prompt-4k.txt; P32=$P/prompt-32k.txt ;;
esac
rm -rf "$O"; mkdir -p "$O"
cp "$P4" "$O/prompt-4k.txt"; cp "$P32" "$O/prompt-32k.txt"
python3 /root/baseline/Strata/bench/bench_vs_llama.py run --machine /root/baseline/cfg/$M-$V.json --out "$O" --rounds "$ROUNDS" --cells "$CELLS" --configs "$CONFIGS" > "$O/run.log" 2>&1
echo "exit=$? $(date -u +%T)" > "$O/DONE"
