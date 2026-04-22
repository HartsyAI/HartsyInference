"""
Dumps a complete Flux pipeline run: text encoding, scheduler sigmas, per-step
latent stats, final latent, and decoded image. Supports both Schnell and Dev.

Usage: python dump_flux_pipeline.py <model_dir> [output_dir] [--dev]
  model_dir: path to Flux diffusers model (e.g., black-forest-labs/FLUX.1-schnell)
  --dev: use Dev settings (20 steps, guidance=3.5) instead of Schnell (4 steps, no guidance)
Requires: pip install diffusers transformers torch safetensors sentencepiece accelerate
"""
import sys
import os
import json
import math
import torch
import numpy as np
from pathlib import Path


def tensor_stats(name: str, t: torch.Tensor) -> dict:
    with torch.no_grad():
        flat = t.float().flatten()
        return {
            "name": name,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "abs_mean": float(flat.abs().mean()),
            "first_8": [float(x) for x in flat[:8]],
        }


def save_tensor(t: torch.Tensor, path: str):
    t.float().cpu().numpy().tofile(path)


def main():
    args = sys.argv[1:]
    is_dev = "--dev" in args
    args = [a for a in args if a != "--dev"]

    model_dir = args[0] if len(args) > 0 else None
    output_dir = args[1] if len(args) > 1 else "flux_reference_tensors/pipeline"

    if model_dir is None:
        print("Usage: python dump_flux_pipeline.py <model_dir> [output_dir] [--dev]")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    from diffusers import FluxPipeline

    device = "cpu"
    dtype = torch.float32

    # --- Settings ---
    prompt = "A photograph of an astronaut riding a horse"
    width, height = 1024, 1024
    seed = 42

    if is_dev:
        num_steps = 20
        guidance_scale = 3.5
        mode = "dev"
    else:
        num_steps = 4
        guidance_scale = 0.0
        mode = "schnell"

    stats = []
    stats.append({
        "name": "settings",
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "mode": mode,
    })

    print(f"=== Flux {mode} pipeline dump ===")
    print(f"  Resolution: {width}x{height}, Steps: {num_steps}, "
          f"Guidance: {guidance_scale}, Seed: {seed}")

    # --- Load pipeline ---
    print(f"\nLoading Flux pipeline from {model_dir}...")
    pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=dtype).to(device)

    config = pipe.transformer.config
    has_guidance = getattr(config, 'guidance_embeds', False)
    print(f"  Transformer: {config.num_layers} double + {config.num_single_layers} single blocks")
    print(f"  Guidance embed: {has_guidance}")

    # --- Step 1: Tokenize ---
    print(f"\nTokenizing: '{prompt}'")

    clip_tokens = pipe.tokenizer(prompt, max_length=77, padding="max_length",
                                  truncation=True, return_tensors="pt")
    clip_ids = clip_tokens["input_ids"][0].tolist()

    t5_max_len = getattr(pipe.tokenizer_2, 'model_max_length', 512)
    t5_tokens = pipe.tokenizer_2(prompt, max_length=t5_max_len, padding="max_length",
                                  truncation=True, return_tensors="pt")
    t5_ids = t5_tokens["input_ids"][0].tolist()

    print(f"  CLIP IDs (first 15): {clip_ids[:15]}")
    print(f"  T5 IDs (first 15): {t5_ids[:15]}")
    print(f"  T5 max len: {t5_max_len}")

    stats.append({"name": "clip_token_ids", "values": clip_ids})
    stats.append({"name": "t5_token_ids", "values": t5_ids[:50]})  # Save first 50 only

    np.array(clip_ids, dtype=np.int32).tofile(os.path.join(output_dir, "clip_token_ids.bin"))
    np.array(t5_ids, dtype=np.int32).tofile(os.path.join(output_dir, "t5_token_ids.bin"))

    # --- Step 2: Text encoding ---
    print("\nEncoding text...")
    with torch.no_grad():
        # CLIP-L
        clip_out = pipe.text_encoder(clip_tokens["input_ids"].to(device),
                                     output_hidden_states=False)
        clip_pooled = clip_out.pooler_output  # [1, 768]

        # T5-XXL
        t5_out = pipe.text_encoder_2(t5_tokens["input_ids"].to(device))
        t5_hidden = t5_out.last_hidden_state  # [1, seq_len, 4096]

    print(f"  CLIP pooled: {clip_pooled.shape}")
    print(f"  T5 hidden: {t5_hidden.shape}")

    save_tensor(clip_pooled, os.path.join(output_dir, "clip_pooled.bin"))
    save_tensor(t5_hidden, os.path.join(output_dir, "t5_hidden.bin"))
    stats.append(tensor_stats("clip_pooled", clip_pooled))
    stats.append(tensor_stats("t5_hidden", t5_hidden))

    # --- Step 3: Scheduler setup ---
    print("\nSetting up scheduler...")
    scheduler = pipe.scheduler
    h_lat, w_lat = height // 8, width // 8
    image_seq_len = (h_lat // 2) * (w_lat // 2)

    # Compute dynamic shift
    mu = 0.0
    if hasattr(scheduler.config, 'use_dynamic_shifting') and scheduler.config.use_dynamic_shifting:
        base_shift = getattr(scheduler.config, 'base_shift', 0.5)
        max_shift = getattr(scheduler.config, 'max_shift', 1.15)
        base_seq = getattr(scheduler.config, 'base_image_seq_len', 256)
        max_seq = getattr(scheduler.config, 'max_image_seq_len', 4096)
        m = (max_shift - base_shift) / (max_seq - base_seq)
        b = base_shift - m * base_seq
        mu = m * image_seq_len + b
        print(f"  Dynamic shift: mu={mu:.6f}, exp(mu)={math.exp(mu):.6f}")
        print(f"  Image seq len: {image_seq_len}")

    scheduler.set_timesteps(num_steps)
    sigmas = scheduler.sigmas.tolist() if hasattr(scheduler, 'sigmas') else []
    timesteps = scheduler.timesteps.tolist()

    print(f"  Sigmas: {[f'{s:.6f}' for s in sigmas[:6]]}...")
    print(f"  Timesteps: {[f'{t:.1f}' for t in timesteps[:6]]}...")

    stats.append({
        "name": "scheduler",
        "mu": mu,
        "image_seq_len": image_seq_len,
        "sigmas": sigmas,
        "timesteps": timesteps,
    })

    np.array(sigmas, dtype=np.float32).tofile(os.path.join(output_dir, "sigmas.bin"))
    np.array(timesteps, dtype=np.float32).tofile(os.path.join(output_dir, "timesteps.bin"))

    # --- Step 4: Initial noise ---
    print("\nGenerating initial noise...")
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(1, 16, h_lat, w_lat, generator=generator, device=device, dtype=dtype)
    init_sigma = float(scheduler.init_noise_sigma) if hasattr(scheduler, 'init_noise_sigma') else 1.0
    latents = latents * init_sigma

    print(f"  Latent shape: {latents.shape}")
    print(f"  Init sigma: {init_sigma:.6f}")

    save_tensor(latents, os.path.join(output_dir, "initial_latents.bin"))
    stats.append(tensor_stats("initial_latents", latents))

    # --- Step 5: Pack latents ---
    print("\nPacking latents...")
    # [B, C, H, W] -> [B, H/2*W/2, C*4]
    B, C, H, W = latents.shape
    packed = latents.view(B, C, H // 2, 2, W // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    packed = packed.reshape(B, (H // 2) * (W // 2), C * 4)

    print(f"  Packed shape: {packed.shape}")
    save_tensor(packed, os.path.join(output_dir, "packed_latents.bin"))
    stats.append(tensor_stats("packed_latents", packed))

    # --- Step 6: Run full pipeline ---
    print(f"\nRunning full pipeline ({num_steps} steps)...")

    # Hook to capture per-step latents
    step_latents = []

    class StepCallback:
        def __call__(self, pipe, step_index, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents", None)
            if latents is not None:
                step_latents.append(latents.detach().clone())
            return callback_kwargs

    callback = StepCallback()

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="np",
            callback_on_step_end=callback,
        )

    # Save per-step latents
    for i, lat in enumerate(step_latents):
        save_tensor(lat, os.path.join(output_dir, f"step_{i}_latents.bin"))
        s = tensor_stats(f"step_{i}_latents", lat)
        stats.append(s)
        if i < 3 or i == len(step_latents) - 1:
            print(f"  Step {i}: mean={s['mean']:.6f}, std={s['std']:.6f}")

    # --- Step 7: Save output image ---
    if result.images is not None and len(result.images) > 0:
        img_np = result.images[0]  # [H, W, 3] float32 0-1
        np.save(os.path.join(output_dir, "output_image.npy"), img_np)

        # Also save as uint8 for easy viewing
        img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(img_uint8).save(os.path.join(output_dir, "output_image.png"))
            print(f"\n  Output image saved as PNG: {img_uint8.shape}")
        except ImportError:
            print(f"\n  Output image saved as npy: {img_uint8.shape} (install PIL for PNG)")

        stats.append({
            "name": "output_image",
            "shape": list(img_np.shape),
            "pixel_mean": float(img_np.mean()),
            "pixel_std": float(img_np.std()),
            "pixel_min": float(img_np.min()),
            "pixel_max": float(img_np.max()),
        })

    # --- Save stats ---
    with open(os.path.join(output_dir, f"flux_{mode}_pipeline_reference.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll pipeline reference data saved to {output_dir}")
    print(f"  Mode: {mode}, Steps: {num_steps}, Size: {width}x{height}")


if __name__ == "__main__":
    main()
