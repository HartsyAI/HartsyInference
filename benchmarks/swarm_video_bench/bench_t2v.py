#!/usr/bin/env python3
"""
Video T2V end-to-end benchmark driver (SwarmUI API) — companion to ../swarm_image_bench/bench_t2i.py.

Times full end-to-end wall-clock per generation through POST /API/GenerateText2Image with the
Text2Video params (blocks until the video file is written), plus peak 4090 VRAM. Routing is by
which backend is ENABLED in SwarmUI (one at a time); tag runs with --backend.

Protocol per model: 1 cold gen (model load) + REPS warm gens, RANDOM SEED each (defeats the
identical-params result cache). Matches the 2026-07-03 methodology (25f, 512x320, h264-mp4).
"""
import argparse, json, subprocess, sys, threading, time, urllib.request

BASE = "http://192.168.10.188:7801"
REPS = 2
GPU_SMI_INDEX = 1  # nvidia-smi index 1 = RTX 4090
PROMPT = ("a red fox trotting through a sunlit snowy forest, cinematic, shallow depth of field, "
          "golden hour, highly detailed")

MODELS = {
    "LTX-2.3-22B":  {"model": "LtxVideo2/ltx-2.3-22b-dev-fp8",
                     "params": {"width": 512, "height": 320, "steps": 20, "cfgscale": 4,
                                "text2videoframes": 25}},
    "LTX-0.9-2B":   {"model": "LtxVideo/ltx-video-2b-v0.9",
                     "params": {"width": 512, "height": 320, "steps": 20, "cfgscale": 3,
                                "text2videoframes": 25}},
    "Wan21-1.3B":   {"model": "Wan/wan2.1_t2v_1.3B_fp16",
                     "params": {"width": 512, "height": 320, "steps": 20, "cfgscale": 6,
                                "text2videoframes": 25}},
    "Wan22-TI2V-5B": {"model": "Wan/wan2.2_ti2v_5B_fp16",
                     "params": {"width": 512, "height": 320, "steps": 20, "cfgscale": 6,
                                "text2videoframes": 25}},
    "Wan21-14B-fp8": {"model": "Wan/wan2.1_t2v_14B_fp8_scaled",
                     "params": {"width": 512, "height": 320, "steps": 15, "cfgscale": 6,
                                "text2videoframes": 25}},
}
COMMON = {"videoresolution": "Image", "videoformat": "h264-mp4"}


def http_post(path, payload, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def new_session():
    return http_post("/API/GetNewSession", {})["session_id"]


class VramSampler:
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
    def start(self): self._thread.start()
    def stop(self): self._done.set()


def one_gen(sid, model, params, seed):
    payload = {"session_id": sid, "images": 1, "model": model, "prompt": PROMPT,
               "seed": seed, **COMMON, **params}
    t0 = time.perf_counter()
    r = http_post("/API/GenerateText2Image", payload)
    dt = time.perf_counter() - t0
    if "error" in r or "error_id" in r:
        return {"error": r.get("error") or r.get("error_id"), "wall": dt}
    vids = r.get("images", [])
    if not vids:
        return {"error": f"no output returned: {str(r)[:200]}", "wall": dt}
    return {"wall": dt, "video": vids[0]}


def bench_model(sid, name, spec):
    print(f">>> {name}  ({spec['model']})", file=sys.stderr)
    base_seed = (abs(hash(name)) % 100000) + int(time.time()) % 1000
    w = one_gen(sid, spec["model"], spec["params"], base_seed)
    if "error" in w:
        print(f"    COLD ERROR: {w['error']}", file=sys.stderr)
        return {"name": name, **spec, "error": w["error"], "cold_wall": w["wall"]}
    print(f"    cold={w['wall']:.2f}s", file=sys.stderr)
    walls, peaks = [], []
    for i in range(REPS):
        smp = VramSampler(); smp.start()
        r = one_gen(sid, spec["model"], spec["params"], base_seed + 1 + i)
        smp.stop()
        if "error" in r:
            print(f"    WARM ERROR: {r['error']}", file=sys.stderr)
            return {"name": name, **spec, "error": r["error"], "cold_wall": w["wall"]}
        walls.append(r["wall"]); peaks.append(smp.peak)
        print(f"    warm[{i}]={r['wall']:.2f}s  peakVRAM={smp.peak}MiB  ({r['video']})", file=sys.stderr)
    walls.sort()
    return {"name": name, "model": spec["model"], "params": spec["params"],
            "cold_wall": w["wall"], "warm_walls": walls,
            "warm_median": walls[len(walls) // 2], "peak_vram_mib": max(peaks),
            "sample": r["video"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="tag: hartsy | comfy")
    ap.add_argument("--only", default=None, help="comma-separated model names (default: all)")
    ap.add_argument("--out", default="/tmp/bench_t2v.json")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(MODELS)
    sid = new_session()
    results = []
    for n in names:
        if n not in MODELS:
            print(f"unknown model {n}; have {list(MODELS)}", file=sys.stderr); sys.exit(2)
        results.append(bench_model(sid, n, MODELS[n]))
    out = {"backend": args.backend, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
