#!/usr/bin/env python3
"""Tier-0 python baseline: llama-cpp-python's VLM path (Llava15ChatHandler, which llama.cpp reuses for
LLaVA-NeXT too) for a same-hardware SPEED comparison against HartsyInference's VlmDecodeThroughputBenchmark.

SPEED ONLY, not a correctness reference: llama.cpp's own LLaVA-NeXT merge step is a documented bug
(base tile placed last instead of first, no unpad step, image_newline loaded but unused in the graph;
ggml-org/llama.cpp#8457) -- its OUTPUT will not match HartsyInference's (deliberately more faithful,
transformers-verified) merge. Do not "fix" HartsyInference's merge to match this baseline's output.
"""
import argparse
import statistics
import sys
import time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

MAX_TOKENS = 64
REPS = 5
WARMUP = 1
QUESTION = "Describe this image in detail."


def run_once(llm, image_path):
    stamps = []
    t0 = time.perf_counter()
    for chunk in llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                {"type": "text", "text": QUESTION},
            ],
        }],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stream=True,
    ):
        delta = chunk["choices"][0].get("delta", {})
        if delta.get("content"):
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
    ap.add_argument("mmproj_path")
    ap.add_argument("image_path")
    ap.add_argument("--ngpu-layers", type=int, default=99)
    args = ap.parse_args()

    chat_handler = Llava15ChatHandler(clip_model_path=args.mmproj_path, verbose=False)
    llm = Llama(model_path=args.model_path, chat_handler=chat_handler,
                n_gpu_layers=args.ngpu_layers, n_ctx=4096, verbose=False)

    for _ in range(WARMUP):
        r = run_once(llm, args.image_path)
        if r is None:
            print("ERROR: warmup produced no tokens", file=sys.stderr)
            sys.exit(1)

    tgs, ttfts = [], []
    for i in range(REPS):
        r = run_once(llm, args.image_path)
        tgs.append(r["tg"])
        ttfts.append(r["ttft"])
        print(f"  rep {i}: tg={r['tg']:.2f} tok/s  ttft={r['ttft']*1000:.0f}ms  n={r['n']}", file=sys.stderr)

    print(f"MEDIAN tg={statistics.median(tgs):.2f} tok/s  ttft={statistics.median(ttfts)*1000:.0f}ms")


if __name__ == "__main__":
    main()
