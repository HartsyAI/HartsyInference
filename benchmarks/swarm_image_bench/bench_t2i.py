#!/usr/bin/env python3
"""
Image T2I Comfy-vs-Hartsy end-to-end benchmark driver (SwarmUI API).

Times full end-to-end wall-clock per generation through POST /API/GenerateText2Image
(blocks until the image file is written), plus peak 4090 VRAM sampled during the run.

Routing is by which backend is ENABLED in SwarmUI (one at a time) — this script does
NOT toggle backends; it just tags output with --backend. Run it once per backend.

Protocol per model: 1 warmup (forces model load) + REPS real gens, RANDOM SEED each
(defeats SwarmUI's identical-params result cache), same params across both backends.
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time, urllib.request

BASE = os.environ.get("BENCH_BASE_URL", "http://192.168.10.188:7801")
REPS = 3
GPU_SMI_INDEX = int(os.environ.get("BENCH_GPU_SMI_INDEX", "1"))  # nvidia-smi index: 1=RTX 4090, 0=RTX 3060
GEN_TIMEOUT_S = int(os.environ.get("BENCH_GEN_TIMEOUT_S", "600"))
PROMPT = ("a highly detailed photograph of an astronaut riding a horse across a "
          "martian desert at golden hour, dramatic lighting, sharp focus")

def http_post(path, payload, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def new_session():
    return http_post("/API/GetNewSession", {})["session_id"]

class VramSampler:
    """Poll 4090 memory.used in a background thread; record the peak."""
    def __init__(self):
        self.peak = 0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._done.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                     "-i", str(GPU_SMI_INDEX)], text=True).strip()
                self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            self._done.wait(0.5)
    def start(self):
        self._thread.start()
    def stop(self):
        self._done.set()
    def join(self):
        self._thread.join()

EXACT_BACKEND_ID = None  # set from --exact-backend-id; pins routing via T2IParamTypes.ExactBackendID

def one_gen(sid, model, params, seed, prompt=PROMPT):
    payload = {"session_id": sid, "images": 1, "model": model, "prompt": prompt,
               "seed": seed, **params}
    if EXACT_BACKEND_ID is not None:
        payload["exactbackendid"] = EXACT_BACKEND_ID
    t0 = time.perf_counter()
    try:
        r = http_post("/API/GenerateText2Image", payload, timeout=GEN_TIMEOUT_S)
    except Exception as e:
        return {"error": f"timeout/exception after {GEN_TIMEOUT_S}s: {e!r}",
                "wall": time.perf_counter() - t0}
    dt = time.perf_counter() - t0
    if "error" in r or "error_id" in r:
        return {"error": r.get("error") or r.get("error_id"), "wall": dt}
    imgs = r.get("images", [])
    if not imgs:
        return {"error": f"no image returned: {str(r)[:200]}", "wall": dt}
    return {"wall": dt, "image": imgs[0]}

def bench_model(sid, name, model, params, prompt=PROMPT, seed=None):
    print(f">>> {name}  ({model})", file=sys.stderr)
    # Deterministic across processes/backends (Python's hash() is per-process
    # randomized for strings) so Hartsy-vs-Comfy runs share seeds for quality
    # comparison. Config may pin an explicit seed; otherwise derive one from
    # the model name via a stable hash (not Python's randomized hash()).
    if seed is None:
        seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
    base_seed = seed
    # warmup / cold (also the model-load number)
    w = one_gen(sid, model, params, base_seed, prompt)
    if "error" in w:
        print(f"    COLD ERROR: {w['error']}", file=sys.stderr)
        return {"name": name, "model": model, "params": params, "error": w["error"],
                "cold_wall": w["wall"]}
    print(f"    cold={w['wall']:.2f}s", file=sys.stderr)
    walls = []
    for i in range(REPS):
        smp = VramSampler(); smp.start()
        r = one_gen(sid, model, params, base_seed + 1 + i, prompt)
        smp.stop(); smp.join()
        if "error" in r:
            print(f"    warm[{i}] ERROR: {r['error']}", file=sys.stderr)
            return {"name": name, "model": model, "params": params, "error": r["error"],
                    "cold_wall": w["wall"], "warm_walls": walls}
        r["peak_vram_mib"] = smp.peak
        walls.append(r)
        print(f"    warm[{i}]={r['wall']:.2f}s  peakVRAM={smp.peak}MiB", file=sys.stderr)
    ws = sorted(x["wall"] for x in walls)
    median = ws[len(ws)//2]
    return {"name": name, "model": model, "params": params,
            "cold_wall": round(w["wall"], 2),
            "warm_walls": [round(x["wall"], 2) for x in walls],
            "warm_median": round(median, 2),
            "warm_min": round(min(ws), 2),
            "peak_vram_mib": max(x["peak_vram_mib"] for x in walls)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="label: comfy | hartsy")
    ap.add_argument("--config", required=True, help="JSON list of {name,model,params}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="subset of model names")
    ap.add_argument("--exact-backend-id", type=int, default=None,
                     help="pin routing via exactbackendid JSON param (T2IParamTypes.ExactBackendID) "
                          "instead of relying on which backend happens to be enabled")
    args = ap.parse_args()
    global EXACT_BACKEND_ID
    EXACT_BACKEND_ID = args.exact_backend_id
    models = json.load(open(args.config))
    if args.only:
        models = [m for m in models if m["name"] in args.only]
    sid = new_session()
    print(f"backend={args.backend}  session={sid[:12]}  models={len(models)}", file=sys.stderr)
    results = []
    for m in models:
        try:
            results.append(bench_model(sid, m["name"], m["model"], m.get("params", {}),
                                       m.get("prompt", PROMPT), m.get("seed")))
        except Exception as e:
            results.append({"name": m["name"], "model": m["model"], "error": repr(e)})
            print(f"    EXC: {e!r}", file=sys.stderr)
        json.dump({"backend": args.backend, "results": results}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
