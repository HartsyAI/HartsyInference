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

--voice-qwen3 mode (voice-turn latency, Phase 0 of the phone-call plan — bounding context, NOT a gate input):
  Qwen3-4B-Q4_K_M.gguf, a ~500-token prompt (counted with LLMAssistantCountTokens — the engine's own tokenizer —
  and printed), thinking OFF, greedy, 64 max tokens, 2 warm-ups + 5 timed. TTFT and decode tok/s come from the
  streamed chunk timestamps exactly as above. Its constants are separate from the frozen protocol.

  Thinking off: the LLMAssistant WS request has no enable_thinking field (HartsyLocalLLMProvider never sets
  TextRequest.EnableThinking), so the Qwen3 soft switch `/no_think` is appended to the user message. The model
  then emits an empty `<think>\\n\\n</think>` block first; those tokens are streamed verbatim (LLMStreamHelper does
  no think-tag stripping), so the first chunk IS the first generated token and they count inside the 64-token
  budget. The reply is checked for that empty block and the result is recorded.

  Card: the single-reply WS path has no backendId, and Qwen3-4B is registered on both local backends
  (cuda:0 = 4090 and cuda:1 = 3060), so the card that served each request is attributed by sampling
  `nvidia-smi --query-compute-apps` by GPU UUID during the call (used_memory delta / new SwarmUI row). A run
  that lands on the 3060 is bounding context for the wrong card and is flagged as such.

  Gate: refuses to run unless --quiet-window-passed is given (run tests/swarm-quiet-window.sh --gpu 4090
  first) and re-checks GetGlobalStatus (every session's queue) before every call; foreign Swarm activity or a
  foreign GPU process aborts the batch. After the batch, `tests/swarm-quiet-window.sh --verify-since <start>`
  confirms no request landed in Swarm's journal during it.

  Usage: python3 swarm_llm_bench.py --voice-qwen3 --quiet-window-passed [--host 192.168.10.188]
                                    [--voice-out benchmarks/results/swarm_voice_llm.json]
"""
import argparse, asyncio, datetime, json, math, statistics, subprocess, threading, time, urllib.request, sys
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

# ---- --voice-qwen3 protocol (separate from the frozen one above) ---------------------------------------------
VOICE_MODEL = "Qwen3-4B-Q4_K_M.gguf"
VOICE_MAX_TOKENS = 64
VOICE_WARMUP = 2
VOICE_REPS = 5
VOICE_TARGET_PROMPT_TOKENS = 500
VOICE_CONTEXT_PARAGRAPH = (
    "Call notes so far: the caller is phoning about an order placed last week. They want to change the delivery "
    "address to their office, ask whether the courier can call ahead, and confirm the refund on a returned item "
    "has been issued. The account shows one open order, one completed return, and a note that the previous agent "
    "promised a callback that never happened. Company policy: address changes are allowed until the parcel is "
    "scanned at the depot, refunds take three to five business days, and any promise of a callback must be logged. "
)
VOICE_QUESTION = "Summarize what the caller needs in two short sentences. /no_think"
GPU_SAMPLE_INTERVAL_S = 0.15


def http_post(path, payload):
    req = urllib.request.Request(BASE_HTTP + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def new_session():
    return http_post("/API/GetNewSession", {})["session_id"]


def unload_models(sid):
    """Free any resident LLM model so only one is in host+GPU memory at a time."""
    try:
        return http_post("/API/LLMAssistantUnloadModels", {"session_id": sid})
    except Exception:
        return None


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


async def one_generation(sid, tid, model, prompt=PROMPT, max_tokens=MAX_TOKENS):
    """Run one prompt, return timing dict or None on error."""
    frame = {"session_id": sid, "threadId": tid, "message": prompt,
             "model": model, "temperature": TEMPERATURE,
             "maxTokens": max_tokens, "seed": SEED}
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
    # Free the previous model first so peak host+GPU memory stays at one model, not N.
    unload_models(sid)
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


# =============================================================================================================
# --voice-qwen3 helpers
# =============================================================================================================

def percentile(values, p):
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = p * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(values):
    return {"median": statistics.median(values), "p95": percentile(values, 0.95), "min": min(values), "n": len(values)}


def nvidia_gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,index,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append({"uuid": parts[0], "name": parts[1], "index": int(parts[2]), "driver": parts[3]})
    return gpus


def nvidia_compute_apps():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0]:
            try:
                mem = int(parts[3])
            except ValueError:
                mem = 0
            rows.append({"uuid": parts[0], "pid": parts[1], "name": parts[2], "mem": mem})
    return rows


def swarm_pid():
    try:
        pid = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", "swarmui.service"],
                             capture_output=True, text=True).stdout.strip()
        return pid if pid and pid != "0" else None
    except Exception:
        return None


class GpuSampler:
    """Samples compute-apps by UUID while one request is in flight (same rule as swarm_audio_bench.py --voice):
    the card whose SwarmUI row grew the most (>= 32 MiB over baseline) or gained a new SwarmUI row served it;
    any non-SwarmUI pid is foreign and aborts the batch."""

    def __init__(self, spid):
        self.spid = spid
        self.baseline, self.peak, self.foreign = {}, {}, set()
        self._stop = threading.Event()
        self._thread = None

    def _snapshot(self, into):
        for row in nvidia_compute_apps():
            if row["pid"] == self.spid or "swarmui" in row["name"].lower():
                into[row["uuid"]] = max(into.get(row["uuid"], 0), row["mem"])
            else:
                self.foreign.add((row["uuid"], row["pid"], row["name"]))

    def __enter__(self):
        self.baseline, self.peak = {}, {}
        self._snapshot(self.baseline)
        self.peak = dict(self.baseline)
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                self._snapshot(self.peak)
                self._stop.wait(GPU_SAMPLE_INTERVAL_S)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._snapshot(self.peak)

    def attribution(self):
        best_uuid, best_delta = None, 0
        for uuid, peak in self.peak.items():
            delta = peak - self.baseline.get(uuid, 0)
            if uuid not in self.baseline:
                delta = max(delta, 32)
            if delta > best_delta:
                best_uuid, best_delta = uuid, delta
        if best_uuid is None or best_delta < 32:
            return None, 0
        return best_uuid, best_delta


def swarm_status(sid):
    # GetGlobalStatus sums every session; GetCurrentStatus counts only this session's own gens.
    return http_post("/API/GetGlobalStatus", {"session_id": sid}).get("status", {})


def assert_swarm_idle(sid, when):
    s = swarm_status(sid)
    busy = {k: s.get(k) for k in ("live_gens", "waiting_gens", "loading_models") if s.get(k)}
    if busy:
        raise RuntimeError(f"foreign Swarm activity {when}: {busy} — aborting the timed batch")
    return s


def build_voice_prompt(sid):
    """Repeats the context paragraph until the engine's own tokenizer counts >= VOICE_TARGET_PROMPT_TOKENS."""
    copies = 1
    counted, estimated = None, False
    while True:
        text = (VOICE_CONTEXT_PARAGRAPH * copies) + "\n" + VOICE_QUESTION
        n = count_tokens(sid, VOICE_MODEL, text)
        if n is None:
            n = int(len(text.split()) * 1.3)
            estimated = True
        counted = n
        if n >= VOICE_TARGET_PROMPT_TOKENS or copies >= 40:
            return text, counted, estimated
        copies += 1


async def run_voice_qwen3(args):
    if not args.quiet_window_passed:
        print("REFUSED: --voice-qwen3 needs --quiet-window-passed (run tests/swarm-quiet-window.sh --gpu 4090 first).",
              file=sys.stderr)
        sys.exit(2)
    sid = new_session()
    tid = create_thread(sid)
    start_utc = datetime.datetime.now(datetime.timezone.utc)
    gpus = nvidia_gpus()
    spid = swarm_pid()
    status0 = assert_swarm_idle(sid, "at start")
    print(f"session={sid[:12]}…  thread={tid[:12]}…  start {start_utc.isoformat()}", file=sys.stderr)
    print("GPUs: " + "; ".join(f"[{g['index']}] {g['name']} {g['uuid']}" for g in gpus), file=sys.stderr)
    print(f"swarmui.service MainPID: {spid}   Swarm status at start: {json.dumps(status0)}", file=sys.stderr)

    prompt, prompt_tokens, estimated = build_voice_prompt(sid)
    print(f"prompt: {prompt_tokens} tokens ({'ESTIMATED words*1.3 — CountTokens unavailable' if estimated else 'LLMAssistantCountTokens'}), "
          f"max_tokens={VOICE_MAX_TOKENS}, greedy, thinking off via /no_think", file=sys.stderr)

    unload_models(sid)
    reps, cards = [], []
    for i in range(VOICE_WARMUP + VOICE_REPS):
        phase = "warm" if i < VOICE_WARMUP else "timed"
        assert_swarm_idle(sid, f"before call {i + 1}")
        # A fresh thread per call: the WS path re-sends the whole thread history, so reusing one thread
        # would grow the prefill by ~550 tokens + a reply every rep (measured: TTFT 9 -> 35 s over 5 reps).
        tid = create_thread(sid)
        with GpuSampler(spid) as sampler:
            r = await one_generation(sid, tid, VOICE_MODEL, prompt=prompt, max_tokens=VOICE_MAX_TOKENS)
        if sampler.foreign:
            raise RuntimeError(f"foreign GPU process during call {i + 1}: {sorted(sampler.foreign)} — aborting")
        if "error" in r:
            raise RuntimeError(f"call {i + 1} failed: {r['error']}")
        uuid, delta = sampler.attribution()
        tg = (r["n_chunks"] - 1) / r["decode_window"] if r["decode_window"] > 0 else 0.0
        print(f"  {phase} {i + 1}: ttft {r['ttft'] * 1000:7.1f} ms  decode {tg:6.1f} chunk/s  chunks={r['n_chunks']}  "
              f"card={uuid or 'unattributed'}(+{delta} MiB)", file=sys.stderr)
        if phase == "timed":
            r["tg"] = tg
            r["card"] = uuid or "unattributed"
            reps.append(r)
            cards.append(uuid or "unattributed")

    tok = count_tokens(sid, VOICE_MODEL, reps[-1]["full_text"])
    # The extension strips <think>...</think> from full_text (an empty block leaves only its newlines), so
    # "no reasoning" is judged by absence of reasoning text, plus the chunk-vs-max_tokens gap the empty block
    # costs; the raw head is kept so a reader can see it.
    full = reps[-1]["full_text"]
    reasoning = full.split("</think>", 1)[0].replace("<think>", "").strip() if "</think>" in full else ""
    think_empty = reasoning == ""
    result = {
        "mode": "voice-qwen3", "model": VOICE_MODEL, "host": BASE_HTTP, "started_utc": start_utc.isoformat(),
        "gpus": gpus, "swarm_pid": spid, "swarm_status_start": status0, "swarm_status_end": swarm_status(sid),
        "protocol": {"warm": VOICE_WARMUP, "timed": VOICE_REPS, "max_tokens": VOICE_MAX_TOKENS, "greedy": True,
                     "thinking": "off via /no_think soft switch (no WS field)", "prompt_tokens": prompt_tokens,
                     "prompt_tokens_estimated": estimated},
        "ttft_s": summarize([r["ttft"] for r in reps]),
        "decode_chunks_per_s": summarize([r["tg"] for r in reps]),
        "n_chunks_last": reps[-1]["n_chunks"], "counted_reply_tokens": tok, "cards": cards,
        "think_block_empty": think_empty, "reply_head": reps[-1]["full_text"][:200],
    }
    name_of = {g["uuid"]: g["name"] for g in gpus}
    served = sorted(set(cards))
    result["card_names"] = [name_of.get(c, c) for c in served]
    result["served_on_4090"] = all("4090" in name_of.get(c, "") for c in served) and served != ["unattributed"]
    with open(args.voice_out, "w") as f:
        json.dump(result, f, indent=2)
    t, d = result["ttft_s"], result["decode_chunks_per_s"]
    print(f"\nQwen3-4B via Swarm: prompt {prompt_tokens} tok | TTFT median {t['median'] * 1000:.0f} ms p95 {t['p95'] * 1000:.0f} "
          f"min {t['min'] * 1000:.0f} | decode median {d['median']:.1f} chunk/s p95 {d['p95']:.1f} min {d['min']:.1f} "
          f"| chunks {result['n_chunks_last']} counted {tok} | think empty: {think_empty} | card(s): "
          f"{', '.join(result['card_names'])}{'' if result['served_on_4090'] else '  (NOT the 4090 — wrong card for the gate context)'}")
    print(f"wrote {args.voice_out}", file=sys.stderr)


async def main():
    global BASE_HTTP, BASE_WS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", default="benchmarks/results/swarm_llm_3060.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="7801")
    ap.add_argument("--voice-qwen3", action="store_true", help="voice-turn Qwen3-4B TTFT/tok-s mode (see module doc)")
    ap.add_argument("--quiet-window-passed", action="store_true",
                    help="assert that tests/swarm-quiet-window.sh --gpu 4090 exited 0 just before this run")
    ap.add_argument("--voice-out", default="benchmarks/results/swarm_voice_llm.json")
    args = ap.parse_args()
    BASE_HTTP = f"http://{args.host}:{args.port}"
    BASE_WS = f"ws://{args.host}:{args.port}"

    if args.voice_qwen3:
        await run_voice_qwen3(args)
        return

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
