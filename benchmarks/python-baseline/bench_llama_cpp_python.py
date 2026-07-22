#!/usr/bin/env python3
"""Tier-0 python baseline: llama-cpp-python (llama.cpp's CUDA engine, driven from Python) decode throughput,
same protocol as the C# HartsyInference.Cuda.Tests.TextDecodeThroughputBenchmark and the project's frozen
llama-bench params (greedy, maxTokens=128, 5 reps + 1 warmup, batch=1, full GPU offload).
"""
import argparse
import statistics
import sys
import time

from llama_cpp import Llama

MAX_TOKENS = 128
REPS = 5
WARMUP = 1
PROMPT = ("Write a detailed, multi-paragraph technical explanation of how a modern "
          "CPU executes a single machine instruction, covering the fetch, decode, "
          "execute, memory-access, and write-back stages, including pipelining, "
          "hazards, branch prediction, and out-of-order execution.")


def run_once(llm):
    stamps = []
    t0 = time.perf_counter()
    for chunk in llm(
        PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stream=True,
    ):
        stamps.append(time.perf_counter())
    if len(stamps) < 2:
        return None
    ttft = stamps[0] - t0
    decode_window = stamps[-1] - stamps[0]
    tg = (len(stamps) - 1) / decode_window if decode_window > 0 else 0
    return {"ttft": ttft, "tg": tg, "n": len(stamps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--ngpu-layers", type=int, default=99)
    args = ap.parse_args()

    llm = Llama(model_path=args.model_path, n_gpu_layers=args.ngpu_layers, n_ctx=2048, verbose=False)

    for _ in range(WARMUP):
        r = run_once(llm)
        if r is None:
            print("ERROR: warmup produced no tokens", file=sys.stderr)
            sys.exit(1)

    tgs, ttfts = [], []
    for i in range(REPS):
        r = run_once(llm)
        tgs.append(r["tg"])
        ttfts.append(r["ttft"])
        print(f"  rep {i}: tg={r['tg']:.2f} tok/s  ttft={r['ttft']*1000:.0f}ms  n={r['n']}", file=sys.stderr)

    print(f"\nllama-cpp-python {args.model_path}")
    print(f"  tg tok/s: median={statistics.median(tgs):.2f} mean={statistics.mean(tgs):.2f} "
          f"stdev={statistics.pstdev(tgs):.2f}")
    print(f"  ttft ms: median={statistics.median(ttfts)*1000:.0f}")


if __name__ == "__main__":
    main()
