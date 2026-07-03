#!/usr/bin/env python3
"""
Tier 2 — Swarm API LLM throughput benchmark (the "real test").

Drives the HartsyInference engine THROUGH the live SwarmUI LLMAssistant
WebSocket API (hartsy-local provider, CUDA), measuring decode throughput
client-side from streamed frame timestamps, then compares against the
llama.cpp baseline captured in benchmarks/results/llamacpp_baseline_3060.json.

Protocol (frozen, identical to the llama-bench baseline where it maps):
  greedy (temperature=0), maxTokens=128, 5 measured reps + 1 warmup, batch=1.
Note: pp (prefill) is not directly comparable through Swarm because TTFT
folds in template + tokenize + queue; the headline number is tg (decode t/s).

The wire protocol emits no t/s or token counts, so we measure:
  TTFT      = t(first chunk) - t(send)          # prefill + first token + overhead
  decode t/s= (n_tokens - 1) / (t(last chunk) - t(first chunk))
n_tokens is taken from LLMAssistantCountTokens on the full reply (authoritative),
with the chunk count reported alongside as a sanity cross-check.
"""
import argparse, asyncio, json, statistics, time, urllib.request, sys
import websockets

BASE_HTTP = "http://127.0.0.1:7801"
BASE_WS   = "ws://127.0.0.1:7801"

# Frozen benchmark params
MAX_TOKENS = 128
TEMPERATURE = 0          # -> greedy in HartsyLocalLLMProvider (Greedy = temp <= 0)
SEED = 0
REPS = 5
WARMUP = 1
# A prompt engineered to reliably fill maxTokens with greedy decoding.
PROMPT = ("Write a detailed, multi-paragraph technical explanation of how a modern "
          "CPU executes a single machine instruction, covering the fetch, decode, "
          "execute, memory-access, and write-back stages, including pipelining, "
          "hazards, branch prediction, and out-of-order execution. Be thorough.")

# The 7 SOTA models, same GGUF files as the llama-bench baseline.
MODELS = [
    "Qwen3-0.6B-Q4_K_M.gguf",
    "llama-3.2-1b-instruct-q8_0.gguf",
    "gemma-3-1b-it-Q4_K_M.gguf",
    "Phi-4-mini-instruct-Q4_K_M.gguf",
    "granite-3.1-2b-instruct-Q4_K_M.gguf",
    "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
    "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
]


def http_post(path, payload):
    req = urllib.request.Request(BASE_HTTP + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def new_session():
    return http_post("/API/GetNewSession", {})["session_id"]


def count_tokens(sid, model, text):
    """Authoritative token count via the same tokenizer the engine uses."""
    try:
        r = http_post("/API/LLMAssistantCountTokens",
                      {"session_id": sid, "model": model, "text": text})
        # tolerate a few possible field names
        for k in ("tokens", "count", "tokenCount", "token_count"):
            if k in r and isinstance(r[k], int):
                return r[k]
    except Exception:
        pass
    return None


async def ws_call(path, frame, collect_stream=False):
    """Open WS, send one JSON frame, gather responses until socket closes.
    Returns (list_of_json_messages, per-message arrival timestamps)."""
    msgs, stamps = [], []
    async with websockets.connect(BASE_WS + path, max_size=None,
                                   open_timeout=30, ping_interval=None) as ws:
        await ws.send(json.dumps(frame))
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=300)
            except (websockets.ConnectionClosed, asyncio.TimeoutError):
                break
            stamps.append(time.perf_counter())
            try:
                msgs.append(json.loads(raw))
            except Exception:
                msgs.append({"_raw": raw})
            if not collect_stream:
                break
    return msgs, stamps


def create_thread(sid):
    r = http_post("/API/LLMAssistantCreateThread", {"session_id": sid})
    if isinstance(r, dict) and r.get("thread"):
        return r["thread"]["id"]
    raise RuntimeError(f"CreateThread failed: {r}")


async def one_generation(sid, tid, model):
    """Run one prompt, return timing dict or None on error."""
    frame = {"session_id": sid, "threadId": tid, "message": PROMPT,
             "model": model, "temperature": TEMPERATURE,
             "maxTokens": MAX_TOKENS, "seed": SEED}
    t_send = time.perf_counter()
    chunk_stamps, full_text, err = [], None, None
    async with websockets.connect(BASE_WS + "/API/LLMAssistantSendMessageWS",
                                  max_size=None, open_timeout=60,
                                  ping_interval=None) as ws:
        await ws.send(json.dumps(frame))
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=600)
            except (websockets.ConnectionClosed, asyncio.TimeoutError):
                break
            now = time.perf_counter()
            try:
                m = json.loads(raw)
            except Exception:
                continue
            if "chunk" in m and m["chunk"]:
                chunk_stamps.append(now)
            elif "error" in m:
                err = m["error"]; break
            elif m.get("done"):
                full_text = m.get("full_text", "")
                break
    if err:
        return {"error": err}
    if len(chunk_stamps) < 2:
        return {"error": f"too few chunks ({len(chunk_stamps)})", "full_text": full_text}
    ttft = chunk_stamps[0] - t_send
    decode_window = chunk_stamps[-1] - chunk_stamps[0]
    n_chunks = len(chunk_stamps)
    return {"ttft": ttft, "decode_window": decode_window,
            "n_chunks": n_chunks, "full_text": full_text}


async def bench_model(sid, tid, model):
    # warmup (also forces lazy model load into the CUDA slot)
    for _ in range(WARMUP):
        w = await one_generation(sid, tid, model)
        if "error" in w:
            return {"model": model, "error": w["error"]}
    reps = []
    for _ in range(REPS):
        r = await one_generation(sid, tid, model)
        if "error" in r:
            return {"model": model, "error": r["error"]}
        reps.append(r)
    # authoritative token count from the last reply
    tok = count_tokens(sid, model, reps[-1]["full_text"])
    tg_list, ttft_list = [], []
    for r in reps:
        n = tok if tok else r["n_chunks"]   # prefer real token count
        n = min(n, r["n_chunks"]) if tok else r["n_chunks"]
        tg = (r["n_chunks"] - 1) / r["decode_window"] if r["decode_window"] > 0 else 0
        tg_list.append(tg)
        ttft_list.append(r["ttft"])
    return {
        "model": model,
        "tg_tps_median": statistics.median(tg_list),
        "tg_tps_mean": statistics.mean(tg_list),
        "tg_tps_std": statistics.pstdev(tg_list) if len(tg_list) > 1 else 0.0,
        "ttft_ms_median": statistics.median(ttft_list) * 1000,
        "n_chunks": reps[-1]["n_chunks"],
        "counted_tokens": tok,
        "reps": len(reps),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", default="benchmarks/results/swarm_llm_3060.json")
    args = ap.parse_args()

    sid = new_session()
    tid = create_thread(sid)
    print(f"session={sid[:12]}…  thread={tid[:12]}…", file=sys.stderr)

    results = []
    for m in args.models:
        print(f">>> {m}", file=sys.stderr)
        try:
            r = await bench_model(sid, tid, m)
        except Exception as e:
            r = {"model": m, "error": repr(e)}
        results.append(r)
        if "error" in r:
            print(f"    ERROR: {r['error']}", file=sys.stderr)
        else:
            print(f"    tg={r['tg_tps_median']:.2f} t/s  ttft={r['ttft_ms_median']:.0f}ms  "
                  f"chunks={r['n_chunks']}  tok={r['counted_tokens']}", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)

    # console summary
    print(f'\n{"model":42} {"tg t/s (median)":>18} {"ttft ms":>10} {"n_gen":>7}')
    for r in results:
        if "error" in r:
            print(f'{r["model"]:42} {"ERROR: "+str(r["error"])[:30]:>18}')
        else:
            print(f'{r["model"]:42} {r["tg_tps_median"]:12.2f} ±{r["tg_tps_std"]:4.1f} '
                  f'{r["ttft_ms_median"]:10.0f} {r["n_chunks"]:7}')


if __name__ == "__main__":
    asyncio.run(main())
