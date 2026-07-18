#!/usr/bin/env python3
"""
F-Lite Python reference benchmark (diffusers / f_lite pipeline).

ComfyUI (this bundled 0.28.0) has no F-Lite arch, so there is no SwarmUI->Comfy
toggle number for it. This produces the apples-to-apples *Python* baseline the way
diffusers users would run it, matching the image bench protocol:
  1 warmup (model already .to(cuda), so this times a full gen) + 3 timed gens,
  RANDOM seed each, warm = median. Peak VRAM sampled on the active GPU.

Params match benchmarks/swarm_image_bench/models.json F-Lite: 1024^2 / 30 steps / cfg 6.0.
Run on the 4090 via CUDA_VISIBLE_DEVICES=0 (torch fastest-first => cuda:0 = 4090).
"""
import argparse, json, statistics, subprocess, threading, time
import torch
from f_lite import FLitePipeline

PROMPT = ("a highly detailed photograph of an astronaut riding a horse across a "
          "martian desert at golden hour, dramatic lighting, sharp focus")

class VramSampler:
    def __init__(self, smi_index):
        self.peak = 0; self.idx = smi_index
        self._done = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._done.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                     "-i", str(self.idx)], text=True).strip()
                self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            self._done.wait(0.5)
    def start(self): self._t.start()
    def stop(self): self._done.set(); self._t.join()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smi-index", type=int, default=1, help="nvidia-smi index for VRAM sampling")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--offload", choices=["resident", "model", "sequential"], default="sequential")
    args = ap.parse_args()

    t_load = time.perf_counter()
    pipe = FLitePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    # F-Lite is 10B (~20GB bf16) + T5-XXL (~9GB) => ~29GB, over the 4090's 24GB.
    # Try fully-resident first (fastest); fall back to model CPU offload if it OOMs.
    # F-Lite is 10B (~20GB bf16) + T5-XXL (~9GB). Fully-resident and even
    # accelerate model-offload OOM the 4090 (the 20GB DiT + 1024^2 activations
    # exceed 24GB during its forward). Sequential (layer-level) offload is the
    # standard way a 24GB diffusers user runs a 10B model; it fits but is slower.
    mode = "cuda-resident"
    if args.offload == "resident":
        pipe.to("cuda")
    elif args.offload == "model":
        pipe.enable_model_cpu_offload(); mode = "model-cpu-offload"
    else:
        pipe.enable_sequential_cpu_offload(); mode = "sequential-cpu-offload"
    load_s = time.perf_counter() - t_load
    print(f"[load] {load_s:.1f}s  mode={mode}", flush=True)

    def gen(seed):
        g = torch.Generator("cuda").manual_seed(seed)
        t0 = time.perf_counter()
        img = pipe(prompt=PROMPT, height=args.size, width=args.size,
                   num_inference_steps=args.steps, guidance_scale=args.cfg,
                   generator=g).images[0]
        torch.cuda.synchronize()
        return time.perf_counter() - t0, img

    cold, img = gen(1234)
    img.save(args.out.replace(".json", ".png"))
    print(f"[cold] {cold:.2f}s", flush=True)

    walls, peak = [], 0
    for i in range(args.reps):
        smp = VramSampler(args.smi_index); smp.start()
        dt, _ = gen(1235 + i)
        smp.stop(); peak = max(peak, smp.peak)
        walls.append(dt)
        print(f"[warm {i}] {dt:.2f}s peakVRAM={smp.peak}MiB", flush=True)

    res = {"name": "F-Lite", "model": args.model,
           "params": {"width": args.size, "height": args.size, "steps": args.steps, "cfgscale": args.cfg},
           "vram_mode": mode,
           "load_s": round(load_s, 1), "cold_wall": round(cold, 2),
           "warm_walls": [round(w, 2) for w in walls],
           "warm_median": round(statistics.median(walls), 2),
           "warm_min": round(min(walls), 2), "peak_vram_mib": peak}
    json.dump({"backend": "python-diffusers", "results": [res]}, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)

if __name__ == "__main__":
    main()
