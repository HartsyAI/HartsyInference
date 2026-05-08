"""End-to-end PyTorch + diffusers generation baselines. Runs SDXL / Flux Dev / SD3.5 Medium /
Z-Image with matched seed, prompt, scheduler config, and step count to the SharpInference test
suite. Per-step timing emitted to a CSV with the canonical e2e schema (PROFILING_METHODOLOGY.md
§ 14).

Heavy: requires the model checkpoint downloaded locally (HF_HOME or DIFFUSERS_CACHE configured).
The harness skips a model gracefully when its weights aren't present."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from _common import RunInfo, deterministic_seeds, iso8601_utc

# Canonical e2e CSV columns.
E2E_COLUMNS = (
    "run_id",
    "timestamp_utc",
    "gpu_name",
    "backend",
    "model",
    "resolution",
    "steps",
    "seed",
    "trial",
    "step_index",
    "step_ms",
    "total_ms",
    "text_encode_ms",
    "denoise_ms",
    "vae_decode_ms",
    "peak_vram_mb",
)

PROMPT = "A photograph of an astronaut riding a horse"
SEED = 42


def _gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _open_writer(out_path: Path):
    new = not out_path.exists() or out_path.stat().st_size == 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=E2E_COLUMNS)
    if new:
        w.writeheader()
    return f, w


def run_sdxl(out_path: Path, info: RunInfo, *, trials: int, resolution: int, steps: int) -> None:
    try:
        from diffusers import StableDiffusionXLPipeline
    except ImportError:
        print("[sdxl] diffusers missing")
        return
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to("cuda")
    except Exception as e:  # noqa: BLE001
        print(f"[sdxl] skipped — checkpoint unavailable: {e}")
        return

    f, w = _open_writer(out_path)
    try:
        for trial_i in range(trials):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            step_times: list[float] = []
            t_total = time.perf_counter()

            with deterministic_seeds(SEED):
                _start = time.perf_counter()
                # diffusers SDXL uses callback_on_step_end for per-step timing
                last = [time.perf_counter()]

                def cb(pipeline, step, ts, callback_kwargs):
                    now = time.perf_counter()
                    step_times.append((now - last[0]) * 1000.0)
                    last[0] = now
                    return callback_kwargs

                _ = pipe(
                    prompt=PROMPT,
                    width=resolution,
                    height=resolution,
                    num_inference_steps=steps,
                    guidance_scale=5.0,
                    generator=torch.Generator(device="cuda").manual_seed(SEED),
                    callback_on_step_end=cb,
                ).images[0]
            torch.cuda.synchronize()

            total_ms = (time.perf_counter() - t_total) * 1000.0
            peak_mb = _gpu_peak_mb()

            for step_i, step_ms in enumerate(step_times):
                w.writerow(info.base_row(backend="pytorch-diffusers") | {
                    "model": "sdxl-base-1.0",
                    "resolution": f"{resolution}x{resolution}",
                    "steps": steps,
                    "seed": SEED,
                    "trial": trial_i,
                    "step_index": step_i,
                    "step_ms": f"{step_ms:.3f}",
                    "total_ms": f"{total_ms:.3f}" if step_i == 0 else "",
                    "text_encode_ms": "",
                    "denoise_ms": f"{sum(step_times):.3f}" if step_i == 0 else "",
                    "vae_decode_ms": "",
                    "peak_vram_mb": f"{peak_mb:.1f}" if step_i == 0 else "",
                })
    finally:
        f.close()
        del pipe
        torch.cuda.empty_cache()


def run_flux_dev(out_path: Path, info: RunInfo, *, trials: int, resolution: int, steps: int) -> None:
    try:
        from diffusers import FluxPipeline
    except ImportError:
        print("[flux] diffusers missing")
        return
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    except Exception as e:  # noqa: BLE001
        print(f"[flux] skipped — checkpoint unavailable: {e}")
        return

    f, w = _open_writer(out_path)
    try:
        for trial_i in range(trials):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            step_times: list[float] = []
            t_total = time.perf_counter()
            last = [time.perf_counter()]

            def cb(pipeline, step, ts, callback_kwargs):
                now = time.perf_counter()
                step_times.append((now - last[0]) * 1000.0)
                last[0] = now
                return callback_kwargs

            _ = pipe(
                prompt=PROMPT,
                width=resolution,
                height=resolution,
                num_inference_steps=steps,
                guidance_scale=3.5,
                generator=torch.Generator(device="cuda").manual_seed(SEED),
                callback_on_step_end=cb,
            ).images[0]
            torch.cuda.synchronize()

            total_ms = (time.perf_counter() - t_total) * 1000.0
            peak_mb = _gpu_peak_mb()
            for step_i, step_ms in enumerate(step_times):
                w.writerow(info.base_row(backend="pytorch-diffusers") | {
                    "model": "flux-dev",
                    "resolution": f"{resolution}x{resolution}",
                    "steps": steps,
                    "seed": SEED,
                    "trial": trial_i,
                    "step_index": step_i,
                    "step_ms": f"{step_ms:.3f}",
                    "total_ms": f"{total_ms:.3f}" if step_i == 0 else "",
                    "text_encode_ms": "",
                    "denoise_ms": f"{sum(step_times):.3f}" if step_i == 0 else "",
                    "vae_decode_ms": "",
                    "peak_vram_mb": f"{peak_mb:.1f}" if step_i == 0 else "",
                })
    finally:
        f.close()
        del pipe
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="CSV path for end-to-end timings")
    p.add_argument("--trials", type=int, default=3, help="Trials per (model, resolution) combo. E2E is slower than microbench so default lower.")
    p.add_argument("--models", nargs="*", default=["sdxl", "flux"], choices=["sdxl", "flux"])
    args = p.parse_args()

    info = RunInfo.current()
    out_path = Path(args.output)

    if "sdxl" in args.models:
        run_sdxl(out_path, info, trials=args.trials, resolution=1024, steps=20)
    if "flux" in args.models:
        run_flux_dev(out_path, info, trials=args.trials, resolution=512, steps=10)


if __name__ == "__main__":
    main()
