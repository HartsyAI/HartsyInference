#!/usr/bin/env python3
"""LTX-2.5 Hartsy-vs-ComfyUI matrix benchmark, end-to-end through the SwarmUI API.

Every prior head-to-head here measured ONE length with ONE prompt, and one of them was later found to have
been measuring a stale deployment. This runs a matrix instead, because the interesting question is not a
single ratio: our decode advantage scales with frame count while the denoise gap is roughly fixed, so where
the two curves cross is the actual finding.

Routing is by which backend is ENABLED in SwarmUI — this script does NOT toggle backends. Run it once per
arm and pass --backend to tag the output.

    ./headtohead_matrix.py --backend hartsy --out /tmp/h2h_hartsy.json
    ./headtohead_matrix.py --backend comfy  --out /tmp/h2h_comfy.json --vae LTX-2/ltx-2.5-video-vae-bf16.safetensors

Records for every run: wall clock, the engine's own decode time and CHUNK PLAN (decode time is a function of
the free-VRAM-derived budget, so a timing without its plan is not reproducible), peak VRAM, and the output
path so frames can be compared for quality afterwards. Quality is not inferred from timings.
"""
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request

BASE = "http://192.168.10.188:7801"          # binds the LAN IP only; localhost does NOT answer
GPU_SMI_INDEX = 1                             # nvidia-smi index 1 = RTX 4090 (CUDA ordinal 0 — they disagree)
LOG_DIR = "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Data/Logs/2026-08"

MODELS = {"hartsy": "LTX-2.5/ltx-2.5-22b-dev-transformer-int8_lean_convrot",
          "comfy": "ltx-2.5-22b-dev-transformer-int8_lean_convrot"}

# Chosen to exercise different failure modes, not just to look nice:
#  - talking: a named spoken line, so lip-sync and the audio stream are checkable rather than assumed
#  - dragon:  large fast motion + fine scale detail, where the conv decoder smears
#  - control: the exact prompt every existing scoreboard row used, so this matrix ties back to them
PROMPTS = {
    "talking": ("close-up of a woman with freckles speaking directly to camera in a sunlit kitchen, "
                "she says clearly: \"the lighthouse keeper never came back that winter\", natural lip movement, "
                "shallow depth of field, soft window light, skin texture detail, 35mm film look"),
    "dragon": ("a silver-haired woman in blue riding a huge black dragon through a canyon at golden hour, "
               "wings beating, dust and embers streaming past, sweeping aerial tracking shot, "
               "scales and wing membrane detail, cinematic, highly detailed"),
    "control": ("a lone lighthouse keeper walking along a rocky coastline at sunset, waves crashing, "
                "cinematic wide shot, volumetric light, highly detailed, 35mm film look"),
}

LENGTHS = [25, 49, 97, 193]
WIDTH, HEIGHT, STEPS, CFG = 768, 512, 30, 3.0


def http_post(path, payload, timeout=7200):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class VramSampler:
    def __init__(self):
        self.peak = 0
        self._done = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
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
    def start(self): self._t.start(); return self
    def stop(self): self._done.set(); return self.peak


def newest_log():
    logs = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith(".log")]
    return max(logs, key=os.path.getmtime) if logs else None


def scrape_engine_facts(since_pos):
    """Decode ms, chunk plan and decoder identity, read from the engine's own log rather than inferred."""
    facts = {"decode_ms": None, "chunk_plan": None, "diffusion_confirmed": False, "vae_seen": None}
    log = newest_log()
    if not log:
        return facts, since_pos
    with open(log, "r", errors="ignore") as f:
        f.seek(since_pos)
        tail = f.read()
        pos = f.tell()
    for m in re.finditer(r"video VAE decode: (\d+) ms", tail):
        facts["decode_ms"] = int(m.group(1))
    for m in re.finditer(r"(tokens=\d+.*?frames per chunk[^\n]*)", tail):
        facts["chunk_plan"] = m.group(1).strip()
    if "HARTSY_LTX2_DIFFUSION_VAE set" in tail and "310 tensors" in tail:
        facts["diffusion_confirmed"] = True          # the ~3 s silent conv fall-through is the trap this catches
    for m in re.finditer(r'"vae_name": "([^"]*video[^"]*)"', tail):
        facts["vae_seen"] = m.group(1)
    return facts, pos


def ffprobe(path):
    """Duration, fps and whether an audio stream exists — 'same quality' includes the soundtrack."""
    root = "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Output"
    full = os.path.join(root, path.replace("View/local/raw/", "local/raw/")) if path.startswith("View/") else path
    for cand in (full, full.replace("/local/raw/", "/raw/")):
        if os.path.exists(cand):
            try:
                out = subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "stream=codec_type,codec_name,nb_frames,r_frame_rate:format=duration",
                     "-of", "json", cand], text=True)
                d = json.loads(out)
                streams = [(s.get("codec_type"), s.get("codec_name")) for s in d.get("streams", [])]
                return {"path": cand, "streams": streams,
                        "has_audio": any(t == "audio" for t, _ in streams),
                        "duration": float(d.get("format", {}).get("duration", 0))}
            except Exception as e:
                return {"path": cand, "error": str(e)}
    return {"path": full, "error": "not found"}


def one_gen(sid, model, prompt, frames, seed, vae):
    payload = {"session_id": sid, "images": 1, "model": model, "prompt": prompt, "seed": seed,
               "videoresolution": "Image", "videoformat": "h264-mp4",
               "width": WIDTH, "height": HEIGHT, "steps": STEPS, "cfgscale": CFG,
               "text2videoframes": frames, "videofps": 24}
    if vae:
        payload["vae"] = vae
    log = newest_log()
    pos = os.path.getsize(log) if log else 0
    smp = VramSampler().start()
    t0 = time.perf_counter()
    try:
        r = http_post("/API/GenerateText2Image", payload)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "wall": time.perf_counter() - t0, "peak_vram": smp.stop()}
    wall = time.perf_counter() - t0
    peak = smp.stop()
    if "error" in r or not r.get("images"):
        return {"error": r.get("error") or f"no output: {str(r)[:200]}", "wall": wall, "peak_vram": peak}
    time.sleep(1.0)                                   # the engine flushes its phase lines just after returning
    facts, _ = scrape_engine_facts(pos)
    out = {"wall": round(wall, 2), "peak_vram": peak, "video": r["images"][0], **facts}
    out["media"] = ffprobe(r["images"][0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hartsy", "comfy"])
    ap.add_argument("--vae", default=None, help="per-generation VAE override (comfy arm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS))
    ap.add_argument("--prompts", default="control,talking,dragon")
    ap.add_argument("--seed", type=int, default=90210)
    ap.add_argument("--reps", type=int, default=2, help="timed reps after one warmup at each length")
    args = ap.parse_args()

    model = MODELS[args.backend]
    lengths = [int(x) for x in args.lengths.split(",")]
    prompts = [p.strip() for p in args.prompts.split(",")]
    sid = http_post("/API/GetNewSession", {})["session_id"]
    results = []

    print(f">>> backend={args.backend} model={model} vae={args.vae or '(default)'}", file=sys.stderr)
    # One warmup so model load does not land inside a timed rep.
    print(">>> warmup …", file=sys.stderr)
    w = one_gen(sid, model, PROMPTS["control"], lengths[0], args.seed, args.vae)
    print(f"    warmup: {w.get('wall')}s  {w.get('error','')}", file=sys.stderr)

    for prompt_key in prompts:
        for frames in lengths:
            for rep in range(args.reps):
                seed = args.seed + rep          # vary seed so SwarmUI's result cache cannot serve a repeat
                r = one_gen(sid, model, PROMPTS[prompt_key], frames, seed, args.vae)
                r.update({"backend": args.backend, "prompt": prompt_key, "frames": frames,
                          "rep": rep, "seed": seed})
                results.append(r)
                if "error" in r:
                    print(f"    {prompt_key:8s} {frames:4d}f rep{rep}  ERROR {r['error'][:120]}", file=sys.stderr)
                else:
                    dec = f"{r['decode_ms']}ms" if r["decode_ms"] else "?"
                    aud = "A/V" if r["media"].get("has_audio") else "video-only"
                    print(f"    {prompt_key:8s} {frames:4d}f rep{rep}  wall {r['wall']:7.2f}s  "
                          f"decode {dec:>8s}  peak {r['peak_vram']}MiB  {aud}", file=sys.stderr)
                json.dump(results, open(args.out, "w"), indent=1)   # write as we go; a long matrix can die
    print(f">>> wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
