#!/usr/bin/env python3
"""
LTX-2.5 T2V end-to-end benchmark driver (SwarmUI API) — companion to bench_t2v.py, same protocol
but standalone since LTX-2.5 uses a heavier "decent length/quality" workload than the standard
25f/512x320 smoke workload the other rows in VIDEO.md use.

Times full end-to-end wall-clock per generation through POST /API/GenerateText2Image (blocks until
the video file is written), plus peak 4090 VRAM. Routing is by which backend is ENABLED in SwarmUI
(one at a time) — the caller must toggle backends between runs, this script does not.

Protocol: 1 cold gen (model load) + REPS warm gens, RANDOM SEED each (defeats the identical-params
result cache). Model: LTX-2.5/ltx-2.5-22b-dev-transformer-int8_lean_convrot (dev, non-distilled,
comfy-kitchen int8-convrot on both DiT and Gemma-4 TE).
"""
import argparse, glob, hashlib, json, os, subprocess, sys, threading, time, urllib.request

# The SwarmUI API exercises the DEPLOYED EXTENSION, not this repo's build output. On 2026-08-14 the deployed
# DLLs were 21 hours stale and missing a PTX file entirely, so every SwarmUI-side number taken that day measured
# old code — the second time this has cost a scoreboard row (see `stale-deployed-dll-trap`). A memory note is not
# a control; this is. Same shape as the free-VRAM guard and GPU-name assertion in ltx25_bench.sh, which do work.
EXT_DIR = ("/home/hartsy/Desktop/Swarm/SwarmUI.not too old/src/bin/extensions/"
           "SwarmExtensionSwarmUI-HartsyInference-Backend")
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../src/HartsyInference.Cli/bin/Release/net8.0")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_extension_matches_build():
    """Refuse to benchmark a stale deployment. Skipped only if the local build does not exist at all."""
    if os.environ.get("LTX25_SKIP_DEPLOY_CHECK") == "1":
        print("!!! deploy check SKIPPED by LTX25_SKIP_DEPLOY_CHECK=1", file=sys.stderr)
        return
    # HOLE 1 (was a silent skip): no local build meant the check passed by doing nothing, which is the exact
    # failure it exists to prevent. A missing build is now fatal — you cannot verify a deployment against nothing.
    if not os.path.isdir(BUILD_DIR):
        print(f"FATAL: no local net8.0 build at {BUILD_DIR}, so the deployment cannot be verified.\n"
              f"       Run: dotnet build src/HartsyInference.Cli -c Release -f net8.0", file=sys.stderr)
        sys.exit(2)
    stale = []
    # HOLE 2 (was invisible): this compares deployed-vs-local, so a STALE LOCAL BUILD passed happily. Catch it by
    # requiring the build to be newer than every source file feeding it. Source edited after the last build means
    # the deployed DLLs cannot contain it, no matter how well they match. This is how temporal chunking landed in
    # source at 08:54 while the extension still served 08:17 binaries.
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    newest_build = max((os.path.getmtime(p) for p in glob.glob(os.path.join(BUILD_DIR, "HartsyInference.*.dll"))),
                       default=0)
    newest_src, newest_src_path = 0, None
    for root, dirs, files in os.walk(os.path.join(repo, "src")):
        dirs[:] = [d for d in dirs if d not in ("bin", "obj")]
        for fn in files:
            if fn.endswith((".cs", ".ptx", ".csproj")):
                p = os.path.join(root, fn)
                m = os.path.getmtime(p)
                if m > newest_src:
                    newest_src, newest_src_path = m, p
    if newest_src > newest_build:
        stale.append(f"LOCAL BUILD IS STALE: {os.path.relpath(newest_src_path, repo)} is newer than the net8.0 "
                     f"build output ({time.strftime('%H:%M:%S', time.localtime(newest_src))} vs "
                     f"{time.strftime('%H:%M:%S', time.localtime(newest_build))}) — rebuild, then redeploy")
    for built in sorted(glob.glob(os.path.join(BUILD_DIR, "HartsyInference.*.dll"))):
        deployed = os.path.join(EXT_DIR, os.path.basename(built))
        if not os.path.exists(deployed):
            stale.append(f"{os.path.basename(built)}: MISSING from the deployed extension")
        elif _md5(built) != _md5(deployed):
            stale.append(f"{os.path.basename(built)}: deployed md5 != freshly built md5")
    for ptx in sorted(glob.glob(os.path.join(os.path.dirname(BUILD_DIR.rstrip('/')), "../../../HartsyInference.Cuda/Ptx/*.ptx"))):
        deployed = os.path.join(EXT_DIR, "Ptx", os.path.basename(ptx))
        if not os.path.exists(deployed):
            stale.append(f"Ptx/{os.path.basename(ptx)}: MISSING from the deployed extension")
        elif _md5(ptx) != _md5(deployed):
            stale.append(f"Ptx/{os.path.basename(ptx)}: deployed md5 != repo md5")
    if stale:
        print("FATAL: the deployed extension does not match this repo's build — any number from this run would\n"
              "       describe old code. Deploy first (build net8.0, copy HartsyInference.*.dll AND Ptx/*.ptx\n"
              "       into the extension, restart swarmui.service), or set LTX25_SKIP_DEPLOY_CHECK=1 if you\n"
              "       genuinely mean to benchmark the deployed build.\n", file=sys.stderr)
        for s in stale[:12]:
            print(f"       {s}", file=sys.stderr)
        if len(stale) > 12:
            print(f"       ... and {len(stale) - 12} more", file=sys.stderr)
        sys.exit(2)
    print(">>> deploy check OK: deployed extension matches the local net8.0 build", file=sys.stderr)

BASE = "http://192.168.10.188:7801"
REPS = int(os.environ.get("LTX25_REPS", 5))
GPU_SMI_INDEX = 1  # nvidia-smi index 1 = RTX 4090
MODELS = {"hartsy": "LTX-2.5/ltx-2.5-22b-dev-transformer-int8_lean_convrot",
          "comfy": "ltx-2.5-22b-dev-transformer-int8_lean_convrot"}
PROMPT = ("a lone lighthouse keeper walking along a rocky coastline at sunset, waves crashing, "
          "cinematic wide shot, volumetric light, highly detailed, 35mm film look")
# Env-overridable so a campaign can pin a geometry without editing the file — the workload must be identical
# across arms, so the OVERRIDES ARE PART OF THE ROW and belong in whatever the numbers are written into.
PARAMS = {"width": int(os.environ.get("LTX25_W", 768)), "height": int(os.environ.get("LTX25_H", 512)),
          "steps": int(os.environ.get("LTX25_STEPS", 30)), "cfgscale": 3.0,
          "text2videoframes": int(os.environ.get("LTX25_FRAMES", 97)), "videofps": 24}
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


def one_gen(sid, seed, model):
    payload = {"session_id": sid, "images": 1, "model": model, "prompt": PROMPT,
               "seed": seed, **COMMON, **PARAMS}
    t0 = time.perf_counter()
    r = http_post("/API/GenerateText2Image", payload)
    dt = time.perf_counter() - t0
    if "error" in r or "error_id" in r:
        return {"error": r.get("error") or r.get("error_id"), "wall": dt}
    vids = r.get("images", [])
    if not vids:
        return {"error": f"no output returned: {str(r)[:200]}", "wall": dt}
    return {"wall": dt, "video": vids[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="tag: hartsy | comfy")
    ap.add_argument("--out", default="/tmp/bench_ltx25.json")
    args = ap.parse_args()
    if args.backend == "hartsy":
        assert_extension_matches_build()
    model = MODELS[args.backend]
    sid = new_session()
    base_seed = 900000 + int(time.time()) % 1000
    print(f">>> LTX-2.5 dev  ({model})  backend={args.backend}", file=sys.stderr)
    w = one_gen(sid, base_seed, model)
    if "error" in w:
        print(f"    COLD ERROR: {w['error']}", file=sys.stderr)
        out = {"backend": args.backend, "model": model, "params": PARAMS, "error": w["error"],
               "cold_wall": w["wall"]}
        json.dump(out, open(args.out, "w"), indent=2)
        sys.exit(1)
    print(f"    cold={w['wall']:.2f}s  ({w['video']})", file=sys.stderr)
    walls, peaks = [], []
    for i in range(REPS):
        smp = VramSampler(); smp.start()
        r = one_gen(sid, base_seed + 1 + i, model)
        smp.stop()
        if "error" in r:
            print(f"    WARM ERROR: {r['error']}", file=sys.stderr)
            out = {"backend": args.backend, "model": model, "params": PARAMS, "error": r["error"],
                   "cold_wall": w["wall"], "warm_walls": walls}
            json.dump(out, open(args.out, "w"), indent=2)
            sys.exit(1)
        walls.append(r["wall"]); peaks.append(smp.peak)
        print(f"    warm[{i}]={r['wall']:.2f}s  peakVRAM={smp.peak}MiB  ({r['video']})", file=sys.stderr)
    mean = sum(walls) / len(walls)
    out = {"backend": args.backend, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "model": model, "params": PARAMS, "cold_wall": w["wall"],
           "warm_walls": walls, "warm_mean": mean, "peak_vram_mib": max(peaks)}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
