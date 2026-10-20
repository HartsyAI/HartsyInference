"""End-to-end per-step time on ComfyUI with and without --use-ck-attention, and on this engine, for the same models.

Per-step time is the slope of run time between S and 2S sampler steps: (T(2S) - T(S)) / S. That cancels model load,
text encoding and VAE decode on both engines (ComfyUI also caches loaders and encoders across runs, the engine's CLI
reloads per process; the slope removes both). ComfyUI's run time is execution_start -> execution_success from
/history (server-side, queue wait excluded); the engine's is the CLI's wall clock.

Arms run as passes (ComfyUI cannot switch attention without a restart): ck-off, ck-on, engine, repeated --rounds times
so drift lands on every arm. Each ComfyUI pass starts its own server on --port with its own output/temp/user
directories and stops it afterwards; it never touches SwarmUI's ComfyUI.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python3 bench_comfy_e2e.py --out DIR --models krea2,flux1dev \\
        --cli path/to/HartsyInference.Cli.dll

Cross-engine pixels are not comparable (bf16 activations there, F16/F32 here; different comfy_kitchen pins), so the
outputs are kept for inspection only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

COMFY = Path("/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI")
VENV_PY = COMFY / "venv" / "bin" / "python"
MODELS = Path("/mnt/model-storage/Models")
REPO = Path(__file__).resolve().parents[2]

EXTRA_PATHS = """hartsy_models:
  base_path: /mnt/model-storage/Models/
  checkpoints: |
    Stable-Diffusion/BFL/Flux1/
  diffusion_models: |
    Stable-Diffusion/Krea2/Turbo/
    Stable-Diffusion/Wan/
    diffusion_models/
  vae: |
    VAE/
  text_encoders: |
    text_encoders/
    clip/
  clip: |
    clip/
  loras: |
    Lora/
"""

# API-format graphs; the sampler node id and its steps/seed keys per model, and the engine regression case it matches.
GRAPHS = {
    "krea2": {
        "case": "krea2", "steps": 8, "sampler": ("7", "steps", "seed"),
        "graph": {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_turbo_fp8_scaled.safetensors", "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red apple on a wooden table", "clip": ["2", 0]}},
            "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 42, "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                                                      "scheduler": "simple", "positive": ["4", 0], "negative": ["5", 0],
                                                      "latent_image": ["6", 0], "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "attn_p0/krea2"}},
        },
    },
    "flux1dev": {
        "case": "flux1-dev", "steps": 20, "sampler": ("6", "steps", "seed"),
        "graph": {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red apple on a wooden table", "clip": ["1", 1]}},
            "3": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["2", 0], "guidance": 3.5}},
            "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}},
            "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 42, "steps": 20, "cfg": 1.0, "sampler_name": "euler",
                                                      "scheduler": "simple", "positive": ["3", 0], "negative": ["4", 0],
                                                      "latent_image": ["5", 0], "denoise": 1.0}},
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "attn_p0/flux1dev"}},
        },
    },
    "wan5b": {
        "case": "wan5b", "steps": 20, "sampler": ("8", "steps", "seed"),
        "graph": {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "Wan/wan2.2_vae.safetensors"}},
            "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red fox trotting through a sunlit snowy forest, cinematic, shallow depth of field",
                                                            "clip": ["2", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "7": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": 832, "height": 480, "length": 49, "batch_size": 1}},
            "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": 42, "steps": 20, "cfg": 5.0, "sampler_name": "euler",
                                                      "scheduler": "simple", "positive": ["5", 0], "negative": ["6", 0],
                                                      "latent_image": ["7", 0], "denoise": 1.0}},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            "10": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["9", 0], "filename_prefix": "attn_p0/wan5b", "fps": 24.0,
                                                               "lossless": False, "quality": 80, "method": "default"}},
        },
    },
    "ltx25": {
        "case": "ltx25-dev", "steps": 8, "sampler": ("11", "steps", None), "seed_node": ("14", "noise_seed"),
        "graph": {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "ltx-2.5-22b-dev-transformer-int8_lean_convrot.safetensors", "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "LTX-2/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot-v2.safetensors",
                                                        "type": "ltxv", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "LTX-2/ltx-2.5-video-vae-conv-bf16.safetensors"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": "LTX-2/ltx-2.5-audio-vae-bf16.safetensors"}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red fox trotting through a sunlit snowy forest, cinematic", "clip": ["2", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "pc game, console game, video game, cartoon, childish, ugly", "clip": ["2", 0]}},
            "7": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 24}},
            "8": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 768, "height": 512, "length": 49, "batch_size": 1}},
            "9": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": 49, "frame_rate": 24, "batch_size": 1, "audio_vae": ["4", 0]}},
            "10": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["8", 0], "audio_latent": ["9", 0]}},
            "11": {"class_type": "LTXVScheduler", "inputs": {"steps": 8, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1,
                                                            "latent": ["8", 0]}},
            "12": {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1],
                                                                "video_cfg": 3.0, "audio_cfg": 7.0}},
            "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
            "15": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["12", 0], "sampler": ["13", 0],
                                                                    "sigmas": ["11", 0], "latent_image": ["10", 0]}},
            "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
            "17": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["16", 0], "vae": ["3", 0], "tile_size": 512, "overlap": 64,
                                                             "temporal_size": 64, "temporal_overlap": 16}},
            "18": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["17", 0], "filename_prefix": "attn_p0/ltx25", "fps": 24.0,
                                                               "lossless": False, "quality": 80, "method": "default"}},
        },
    },
}


def http(port, path, body=None, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def start_comfy(port, work, ck, log):
    for d in ("out", "tmp", "user"):
        (work / d).mkdir(parents=True, exist_ok=True)
    (work / "extra_model_paths.yaml").write_text(EXTRA_PATHS)
    cmd = [str(VENV_PY), "-s", "main.py", "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch",
           "--disable-all-custom-nodes", "--output-directory", str(work / "out"), "--temp-directory", str(work / "tmp"),
           "--user-directory", str(work / "user"), "--extra-model-paths-config", str(work / "extra_model_paths.yaml")]
    if ck:
        cmd.append("--use-ck-attention")
    proc = subprocess.Popen(cmd, cwd=COMFY, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early; see {log}")
        try:
            http(port, "/system_stats", timeout=5)
            return proc
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("ComfyUI did not come up")


def run_comfy(port, model, steps, seed, timeout=3600):
    spec = GRAPHS[model]
    g = copy.deepcopy(spec["graph"])
    node, steps_key, seed_key = spec["sampler"]
    g[node]["inputs"][steps_key] = steps
    if seed_key:
        g[node]["inputs"][seed_key] = seed
    if "seed_node" in spec:
        sn, sk = spec["seed_node"]
        g[sn]["inputs"][sk] = seed
    resp = http(port, "/prompt", {"prompt": g, "client_id": "attn-p0"})
    pid = resp["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = http(port, f"/history/{pid}")
        if pid in hist:
            msgs = {m[0]: m[1] for m in hist[pid]["status"]["messages"]}
            if "execution_success" not in msgs:
                raise RuntimeError(f"{model} failed: {hist[pid]['status']}")
            return (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000.0
        time.sleep(0.5)
    raise TimeoutError(model)


def run_engine(cli, idx, model, steps, seed, out):
    case = GRAPHS[model]["case"]
    line = subprocess.run(["bash", "-c", f'. "{REPO}/tests/regression-cases.sh"; regression_cases | awk -F"\\t" -v c="{case}" \'$1==c\''],
                          capture_output=True, text=True).stdout.strip()
    _, ckpt, spec = line.split("\t")
    cmd, pos, args = spec.split("|")
    args = [a for a in args.split() if a]
    if "--steps" in args:
        i = args.index("--steps")
        del args[i:i + 2]
    t0 = time.time()
    r = subprocess.run(["dotnet", cli, cmd, pos, "--model-path", str(MODELS / ckpt), "-b", f"cuda:{idx}", "-o", str(out), "--seed",
                        str(seed), *args, "--steps", str(steps)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"engine {model} failed: {r.stdout[-2000:]}{r.stderr[-2000:]}")
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="krea2,flux1dev,wan5b,ltx25")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--port", type=int, default=8299)
    ap.add_argument("--cli", required=True, help="HartsyInference.Cli.dll of the engine build to time")
    ap.add_argument("--gpu-index", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), help="the engine's -b cuda:<n> under PCI order")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in a.models.split(",") if m]
    rows = []
    seed = 1000
    for rnd in range(a.rounds):
        for arm in ("ck-off", "ck-on", "engine"):
            work = out / f"r{rnd}-{arm}"
            proc = None
            try:
                if arm != "engine":
                    proc = start_comfy(a.port, work, arm == "ck-on", out / f"r{rnd}-{arm}.comfy.log")
                for m in models:
                    s = GRAPHS[m]["steps"]
                    times = {}
                    for n in (s, 2 * s):
                        seed += 1
                        if arm == "engine":
                            times[n] = run_engine(a.cli, a.gpu_index, m, n, seed, work / m)
                        else:
                            if n == s and rnd == 0:
                                run_comfy(a.port, m, s, seed + 500)   # warm-up: loads, encoders, plan caches
                            times[n] = run_comfy(a.port, m, n, seed)
                    per_step = (times[2 * s] - times[s]) / s
                    rows.append({"round": rnd, "arm": arm, "model": m, "steps": s, "t_s": times[s], "t_2s": times[2 * s],
                                 "per_step_ms": per_step * 1000})
                    print(f"r{rnd} {arm:7s} {m:9s} T({s})={times[s]:.2f}s T({2*s})={times[2*s]:.2f}s -> {per_step*1000:.1f} ms/step", flush=True)
            finally:
                if proc is not None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
    (out / "e2e.json").write_text(json.dumps(rows, indent=1))
    print("\n| model | ComfyUI default ms/step | ComfyUI --use-ck-attention ms/step | engine ms/step | engine ÷ ck |")
    print("|---|---|---|---|---|")
    for m in models:
        med = {arm: statistics.median(r["per_step_ms"] for r in rows if r["model"] == m and r["arm"] == arm)
               for arm in ("ck-off", "ck-on", "engine")}
        print(f"| {m} | {med['ck-off']:.1f} | {med['ck-on']:.1f} | {med['engine']:.1f} | {med['engine'] / med['ck-on']:.2f}× |")


if __name__ == "__main__":
    sys.exit(main())
