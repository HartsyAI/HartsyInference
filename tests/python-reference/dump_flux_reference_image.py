"""
Generates Flux reference images for the SSIM gate (Flux Dev + Flux Schnell variants).

Output: tests/python-reference/flux_reference_images/
  meta.json
  init_noise_seed42.bin    # F32 packed-latent shape, see notes
  ref_dev_<idx>.png        # Flux Dev: 10 steps, cfg=3.5
  ref_schnell_<idx>.png    # Flux Schnell: 4 steps, cfg=0.0

The C# tests in tests/HartsyInference.Diffusion.Tests/FluxSsimTests.cs compare against
these references at 256×256 (so generation completes in seconds). Strict SSIM > 0.95
requires noise injection into FluxPipeline; current relaxed threshold is documented in
the test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image


PROMPTS = [
    "A photograph of an astronaut riding a horse on the moon",
    "A red sports car parked in front of a modern house, sunset lighting",
    "A still life of a bowl of fruit on a wooden table, oil painting",
]


def run_variant(args, *, variant: str, num_steps: int, cfg: float):
    from diffusers import FluxPipeline
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    repo = args.dev_repo if variant == "dev" else args.schnell_repo
    pipe = FluxPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Flux latent: [1, 16, H/8, W/8], packed at run time. We dump the unpacked F32 noise
    # so the C# test can pass it through the same PackLatent path.
    latent_h = args.height // 8
    latent_w = args.width // 8
    noise_gen = torch.Generator(device="cpu").manual_seed(args.seed)
    init_noise = torch.randn(1, 16, latent_h, latent_w, generator=noise_gen, dtype=torch.float32)
    init_noise.numpy().astype(np.float32).tofile(out / f"init_noise_seed{args.seed}.bin")

    results = []
    for idx, prompt in enumerate(PROMPTS):
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        img = pipe(
            prompt=prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=num_steps,
            guidance_scale=cfg,
            max_sequence_length=256,
            generator=gen,
        ).images[0]
        png = f"ref_{variant}_{idx:02d}.png"
        rgb = f"ref_{variant}_{idx:02d}.rgb"
        img.save(out / png)
        rgb_array = np.array(img.convert("RGB"), dtype=np.uint8)
        assert rgb_array.shape == (args.height, args.width, 3)
        rgb_array.tofile(out / rgb)
        results.append({"index": idx, "variant": variant, "prompt": prompt, "image": png, "rgb": rgb})
        print(f"[{variant} {idx + 1}/{len(PROMPTS)}] {prompt[:60]}... → {png} (+{rgb})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-repo", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--schnell-repo", default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=["dev", "schnell"])
    args = parser.parse_args()

    all_results = []
    if "dev" in args.variants:
        all_results += run_variant(args, variant="dev", num_steps=10, cfg=3.5)
    if "schnell" in args.variants:
        all_results += run_variant(args, variant="schnell", num_steps=4, cfg=0.0)

    out = Path(args.output)
    meta = {
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "prompts": all_results,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nReference images written to {out}")


if __name__ == "__main__":
    main()
