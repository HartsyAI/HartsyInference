#!/usr/bin/env python3
"""
Dory end-to-end run on the swarm box: lease a card from the GPU gate, verify the GGUF, run greedy correctness probes
at each loop count through the SwarmUI LLM Assistant API, then the decode-throughput bench (swarm_llm_bench.py) at
each loop count, and write one summary. Runs on the box itself (the gate and SwarmUI are box-local) and holds its
own heartbeat, so one command covers a run longer than the gate's 10-minute heartbeat timeout.

    python3 benchmarks/swarm_llm_bench/dory_e2e_run.py \
        --model-path /mnt/model-storage/Models/llm/nemotron-dory/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf

Options: --loops 1 2 4 (default), --host/--port for SwarmUI (127.0.0.1:7801), --gate for the lease service, --no-lease
when a lease is already held (under gpu_gate.py's wrapper the inherited GATE_GPU still names the card for the probes and
the bench check) or the gate is not running, --skip-hash to skip the sha256 (the size is always checked), --out-dir for
the result files (benchmarks/results/dory), --device-map for the gate-gpu-to-engine-device mapping.

Outputs in --out-dir: dory_probes.json (every probe reply in full), swarm_llm_dory_loops<N>.json and .log (the bench's own
file and console per loop count) and dory_e2e_summary.json; the summary table is also printed.

The lease covers one card. The probes name it (--device-map); the bench goes through the websocket chat route, which runs on
the LLM Assistant backend's own device, so each bench run is attributed from nvidia-smi: the model is unloaded first and
compute-app memory is sampled throughout the run, so the card whose use grows the most while the bench reloads the model is
the one serving. A run served from any other card, or one that cannot be attributed while nvidia-smi is present, is
rejected. If that happens, point the local backend at the leased card or ask the gate for the card it uses. A probe or
bench in flight when the lease is lost is abandoned and its reply never accepted. A failed probe, a failed bench, a
rejected card or a lost lease fails the run: exit 1 and the lease released as failed.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gpu_gate import GateError, GpuLease, run_under_lease  # noqa: E402

# The pinned release, mirroring the engine catalog's `dory` entry (src/HartsyInference.Engine/Registry/ModelCatalog.cs).
MODEL_FILE = "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
MODEL_SHA256 = "96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0"
MODEL_BYTES = 3_673_288_864
AGENT = "Dory AI model integration"
TASK = "Nemotron Dory 1.0 4B: greedy correctness probes at loops 1/2/4 through the SwarmUI LLM Assistant API, then the decode-throughput bench"
PROBE_PROMPT = "Explain in three sentences why the sky is blue, then list two things that would change its color."
PROBE_MAX_TOKENS = 200
BENCH = pathlib.Path(__file__).resolve().parent / "swarm_llm_bench.py"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def http_post(base, path, payload, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def verify_model(path, skip_hash):
    """The gate does not check the file, so the run does: size always, sha256 unless skipped."""
    if not os.path.isfile(path):
        raise SystemExit(f"model file missing: {path}")
    size = os.path.getsize(path)
    if size != MODEL_BYTES:
        raise SystemExit(f"model size {size} bytes, expected {MODEL_BYTES}: {path}")
    if skip_hash:
        log(f"[model] size ok ({size} bytes); sha256 skipped")
        return {"size_ok": True, "sha256_checked": False}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit(f"model sha256 {digest}, expected {MODEL_SHA256}: {path}")
    log(f"[model] size and sha256 ok: {path}")
    return {"size_ok": True, "sha256_checked": True}


def model_listed(base, sid, name):
    """Whether the LLM Assistant lists the file; the listing shape is the extension's, so this matches on the name."""
    try:
        reply = http_post(base, "/API/LLMAssistantGetModels", {"session_id": sid}, timeout=120)
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        log(f"[swarm] model listing failed: {e}")
        return None
    return name in json.dumps(reply)


def nvidia_smi_apps():
    """{gpu_index: used MiB by compute apps}, or None without nvidia-smi; used to attribute the serving card."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    index_of = {}
    for line in gpus.strip().splitlines():
        index, uuid = [p.strip() for p in line.split(",")]
        index_of[uuid] = int(index)
    used = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        uuid, mib = [p.strip() for p in line.split(",")]
        used[index_of.get(uuid, uuid)] = used.get(index_of.get(uuid, uuid), 0) + int(mib)
    return used


def largest_growth(lows, highs, floor_mib=32):
    """The gpu index whose compute-app memory grew the most between its lowest and highest sample, or None below floor_mib."""
    growth = {gpu: highs.get(gpu, 0) - lows.get(gpu, 0) for gpu in set(lows) | set(highs)}
    gpu, delta = max(growth.items(), key=lambda kv: kv[1], default=(None, 0))
    return gpu if delta >= floor_mib else None


class GpuUseSampler:
    """Samples compute-app memory per gpu index while a step runs, keeping each card's lowest and highest reading; served()
    is the card whose use grew the most during the step. The bench unloads and reloads the model, and the runner unloads it
    before the baseline, so the serving card's growth is the model's size while the others stay flat. available is False
    without nvidia-smi, when no attribution is possible."""

    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self.lows, self.highs = {}, {}
        self.available = False
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        used = nvidia_smi_apps()
        if used is None:
            return
        self.available = True
        for gpu in set(used) | set(self.lows):
            value = used.get(gpu, 0)
            self.lows[gpu] = min(self.lows.get(gpu, value), value)
            self.highs[gpu] = max(self.highs.get(gpu, value), value)

    def __enter__(self):
        self._sample()
        self._stop.clear()

        def loop():
            while not self._stop.wait(self.interval_s):
                self._sample()

        self._thread = threading.Thread(target=loop, name="gpu-use-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()
        return False

    def served(self):
        return largest_growth(self.lows, self.highs)


def post_under_lease(base, path, payload, lease, timeout=600):
    """Issues one SwarmUI request while watching the lease: a lease lost while the request is in flight abandons the wait, and a
    reply that lands after the loss is never accepted; either way the step fails. Returns the reply dict (success false on error)."""
    outcome = {}

    def work():
        try:
            outcome["reply"] = http_post(base, path, payload, timeout=timeout)
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            outcome["error"] = repr(e)

    worker = threading.Thread(target=work, name="swarm-request", daemon=True)
    worker.start()
    while worker.is_alive():
        if lease is not None and lease.lost:
            return {"success": False, "error": "lease lost during the request; abandoned"}
        worker.join(0.5)
    if lease is not None and lease.lost:
        return {"success": False, "error": "lease lost during the request; reply rejected"}
    return outcome.get("reply") or {"success": False, "error": outcome.get("error", "no reply")}


def run_probe(base, sid, loops, device, lease):
    payload = {"session_id": sid, "message": PROBE_PROMPT, "model": MODEL_FILE, "temperature": 0,
               "maxTokens": PROBE_MAX_TOKENS, "noCache": True}
    if loops is not None:
        payload["loops"] = loops
    if device:
        payload["device"] = device
    if lease is not None:
        lease.check_alive()
    started = time.monotonic()
    reply = post_under_lease(base, "/API/LLMAssistantSendMessage", payload, lease)
    elapsed = time.monotonic() - started
    result = {"loops": loops, "device": device, "elapsed_s": round(elapsed, 2), "success": bool(reply.get("success")),
              "response": reply.get("response"), "error": reply.get("error")}
    head = (result["response"] or result["error"] or "")[:120].replace("\n", " ")
    log(f"[probe] loops={loops if loops is not None else 'default'}: {'ok' if result['success'] else 'FAILED'} in {elapsed:.1f}s: {head}")
    return result


def served_card_error(served_gpu, lease_gpu, attributable):
    """Why a bench result must be rejected under the lease, or None: the gate leased one card, so a run served from another,
    or one that cannot be attributed while nvidia-smi is present, is not a result this lease may report."""
    if lease_gpu is None:
        return None
    if not attributable:
        return None
    if served_gpu is None:
        return "the serving card could not be attributed from nvidia-smi; result rejected"
    if served_gpu != lease_gpu:
        return f"served on gpu {served_gpu} but the lease is for gpu {lease_gpu}; result rejected"
    return None


def run_bench(base, sid, host, port, loops, out_file, lease, bound_gpu):
    if lease is not None:
        lease.check_alive()
    cmd = [sys.executable, str(BENCH), "--host", host, "--port", str(port), "--models", MODEL_FILE,
           "--loops", str(loops), "--out", str(out_file)]
    log_file = out_file.with_suffix(".log")
    log(f"[bench] loops={loops}: {' '.join(cmd)}")
    # Unload before the baseline sample, so the bench's own reload shows up as growth on the card that serves it.
    try:
        http_post(base, "/API/LLMAssistantUnloadModels", {"session_id": sid}, timeout=120)
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        pass
    with GpuUseSampler() as sampler:
        returncode, lost = run_under_lease(cmd, lease, log_file=log_file)
    if lost:
        log("[bench] lease lost; the bench was stopped")
    console = log_file.read_text()
    sys.stderr.write(console)
    served = sampler.served()
    record = {"loops": loops, "returncode": returncode, "out": str(out_file), "log": str(log_file), "served_gpu": served,
              "gpu_growth_mib": {str(gpu): sampler.highs.get(gpu, 0) - sampler.lows.get(gpu, 0) for gpu in sampler.highs}}
    if lost:
        record["error"] = "lease lost during the bench; stopped"
    elif returncode == 0 and out_file.is_file():
        rows = json.loads(out_file.read_text())
        row = rows[0] if rows else {}
        record.update({"error": row.get("error"), "tg_tps_median": row.get("tg_tps_median"),
                       "ttft_ms_median": row.get("ttft_ms_median"), "counted_tokens": row.get("counted_tokens")})
    else:
        record["error"] = console.strip().splitlines()[-1] if console.strip() else f"exit {returncode}"
    if not record.get("error"):
        rejected = served_card_error(served, bound_gpu, sampler.available)
        if rejected:
            record.update({"error": rejected, "tg_tps_median": None, "ttft_ms_median": None, "counted_tokens": None})
    if record.get("error"):
        log(f"[bench] loops={loops}: FAILED: {record['error']}")
    return record


def parse_device_map(text):
    """'0=cuda:1,1=cuda:0' -> {0: 'cuda:1', 1: 'cuda:0'}; empty disables the mapping (the backend's own device)."""
    mapping = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        gpu, device = part.split("=", 1)
        mapping[int(gpu)] = device.strip()
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help=f"absolute path of {MODEL_FILE} under /mnt/model-storage/")
    ap.add_argument("--loops", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7801)
    ap.add_argument("--gate", default="http://127.0.0.1:8686")
    ap.add_argument("--no-lease", action="store_true",
                    help="run without requesting a lease; one already held by the caller (GATE_GPU in the environment) still names the card")
    ap.add_argument("--vram-mb", type=int, default=6000)
    ap.add_argument("--minutes", type=int, default=90)
    ap.add_argument("--skip-hash", action="store_true")
    ap.add_argument("--skip-bench", action="store_true", help="probes only")
    ap.add_argument("--out-dir", default="benchmarks/results/dory")
    ap.add_argument("--device-map", default="0=cuda:1,1=cuda:0",
                    help="gate gpu index to engine device for the probes; '' leaves the backend's own device")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "model_path": args.model_path,
               "loops": args.loops, "probes": [], "bench": []}
    summary["model_check"] = verify_model(args.model_path, args.skip_hash)

    try:
        sid = http_post(base, "/API/GetNewSession", {}, timeout=30)["session_id"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
        raise SystemExit(f"SwarmUI at {base} not reachable: {e}")
    summary["model_listed"] = model_listed(base, sid, MODEL_FILE)
    log(f"[swarm] session ok; model listed: {summary['model_listed']}")

    lease = None if args.no_lease else GpuLease(AGENT, TASK, args.model_path, base=args.gate, vram_mb=args.vram_mb,
                                                 minutes=args.minutes, log=log)
    # The card this run is bound to: the lease it holds, or under --no-lease the one the caller's lease (gpu_gate.py's wrapper)
    # exported as GATE_GPU; None only when no lease is involved at all.
    held_gpu = int(os.environ["GATE_GPU"]) if args.no_lease and os.environ.get("GATE_GPU", "").isdigit() else None
    failed = False
    try:
        if lease is not None:
            lease.__enter__()
            summary["lease"] = {"lease_id": lease.lease_id, "gpu": lease.gpu}
        elif held_gpu is not None:
            summary["lease"] = {"lease_id": os.environ.get("GATE_LEASE_ID"), "gpu": held_gpu, "inherited": True}
            log(f"[gate] using the caller's lease on gpu {held_gpu}")
        bound_gpu = lease.gpu if lease is not None else held_gpu
        device = parse_device_map(args.device_map).get(bound_gpu) if bound_gpu is not None else None
        summary["probe_device"] = device

        summary["probes"].append(run_probe(base, sid, None, device, lease))
        for loops in args.loops:
            summary["probes"].append(run_probe(base, sid, loops, device, lease))
        (out_dir / "dory_probes.json").write_text(json.dumps(summary["probes"], indent=2))

        if not args.skip_bench:
            for loops in args.loops:
                summary["bench"].append(run_bench(base, sid, args.host, args.port, loops, out_dir / f"swarm_llm_dory_loops{loops}.json",
                                                  lease, bound_gpu))
        failures = [f"probe loops={p['loops']}: {p['error'] or 'no reply'}" for p in summary["probes"] if not p["success"]]
        failures += [f"bench loops={b['loops']}: {b['error']}" for b in summary["bench"] if b.get("error")]
        if failures:
            failed = True
            summary["error"] = "; ".join(failures)
    except (GateError, KeyboardInterrupt) as e:
        failed = True
        summary["error"] = repr(e)
        log(f"[run] stopped: {e!r}")
    except Exception as e:  # noqa: BLE001 - the lease must be released as failed whatever went wrong
        failed = True
        summary["error"] = repr(e)
        raise
    finally:
        # Swarm cleanup first, the lease last: the card is handed back only once the model is out of it.
        try:
            http_post(base, "/API/LLMAssistantUnloadModels", {"session_id": sid}, timeout=120)
        except (urllib.error.URLError, json.JSONDecodeError):
            pass
        if lease is not None:
            lease.release(failed=failed)
        summary["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out_dir / "dory_e2e_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'loops':>7} {'probe':>7} {'tg t/s':>8} {'ttft ms':>8} {'tokens':>7} {'gpu':>4}  reply head")
    bench_by_loops = {b["loops"]: b for b in summary["bench"]}
    for p in summary["probes"]:
        b = bench_by_loops.get(p["loops"], {})
        tg = b.get("tg_tps_median")
        ttft = b.get("ttft_ms_median")
        tg_text = "" if tg is None else f"{tg:.2f}"
        ttft_text = "" if ttft is None else f"{ttft:.0f}"
        tokens_text = "" if b.get("counted_tokens") is None else str(b["counted_tokens"])
        gpu_text = "" if b.get("served_gpu") is None else str(b["served_gpu"])
        head = (p["response"] or p["error"] or "")[:60].replace("\n", " ")
        print(f"{str(p['loops'] if p['loops'] is not None else 'default'):>7} {'ok' if p['success'] else 'FAIL':>7} "
              f"{tg_text:>8} {ttft_text:>8} {tokens_text:>7} {gpu_text:>4}  {head}")
    print(f"\nwrote {out_dir / 'dory_e2e_summary.json'}")
    return 1 if failed or summary.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
