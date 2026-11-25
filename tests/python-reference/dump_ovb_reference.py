"""Dump a diffusers Flux2KleinPipeline reference for Open Vision Banana (and plain Klein edit), for C# parity.

NOT RUN against real weights yet: that needs the OVB 4B checkpoint (15.5 GB) and a GPU, which have not been available. Its tiny-model
counterpart, dump_flux2_edit_tiny.py, has run (CPU, diffusers 0.43.0) and is what the C# parity tests use. Needs torch, diffusers (record the
version printed in meta.json; OVB's configs say 0.39.0), transformers, numpy and pillow, then compare with the C# Flux2EditPipeline.

Saved inputs, not seeds: the initial latents are drawn once here and saved, so the C# side feeds the same noise
(TextToImageRequest.InitialNoise, shape [1, 128, H/16, W/16]).

    python dump_ovb_reference.py --base black-forest-labs/FLUX.2-klein-base-4B \
        --transformer /models/ovb/4B/zeroshot/checkpoint-87500-95000-w25/transformer \
        --image photo.png --prompt-file prompt.txt --out ovb_ref_4b --steps 28 --guidance 3.5

Writes into --out: meta.json, image_in.png (the resized/cropped reference actually fed), noise.npy ([1,128,h,w] f32),
cond.npy and uncond.npy (the prompt embeddings, [1,512,ctx] f32), ref_tokens.npy ([1,N,128] f32, the packed reference),
latents_step_XX.npy (packed [1,N,128] after each step), out.png and out_latents.npy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.utils import load_image
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="FLUX.2 klein base repo or local snapshot")
    parser.add_argument("--transformer", default=None, help="OVB transformer folder (omit for the plain base)")
    parser.add_argument("--image", default=None, help="reference image (omit for text-to-image)")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").rstrip("\n")

    kwargs = {"torch_dtype": dtype}
    if args.transformer:
        kwargs["transformer"] = Flux2Transformer2DModel.from_pretrained(args.transformer, torch_dtype=dtype)
    pipe = Flux2KleinPipeline.from_pretrained(args.base, **kwargs)
    pipe.enable_model_cpu_offload() if args.offload else pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    image = load_image(args.image) if args.image else None
    call = {"prompt": prompt, "num_inference_steps": args.steps, "guidance_scale": args.guidance}
    if image is not None:
        # Same sizing the pipeline applies: at most 1 MP, floored to a multiple of 16, then fed as-is.
        w, h = image.size
        if w * h > 1024 * 1024:
            image = pipe.image_processor._resize_to_target_area(image, 1024 * 1024)
            w, h = image.size
        w, h = (w // 16) * 16, (h // 16) * 16
        image = image.resize((w, h), Image.BICUBIC) if image.size != (w, h) else image
        image.save(out / "image_in.png")
        call["image"] = image
        call["height"], call["width"] = h, w
    else:
        call["height"], call["width"] = 1024, 1024

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn((1, 128, call["height"] // 16, call["width"] // 16), generator=gen, dtype=torch.float32)
    np.save(out / "noise.npy", noise.numpy())
    call["latents"] = noise

    prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device, num_images_per_prompt=1)
    np.save(out / "cond.npy", prompt_embeds.float().cpu().numpy())
    if args.guidance > 1:
        negative_embeds, _ = pipe.encode_prompt(prompt="", device=pipe._execution_device, num_images_per_prompt=1)
        np.save(out / "uncond.npy", negative_embeds.float().cpu().numpy())

    if image is not None:
        tensor = pipe.image_processor.preprocess(image, height=call["height"], width=call["width"], resize_mode="crop")
        tokens, _ = pipe.prepare_image_latents(
            images=[tensor], batch_size=1, generator=gen, device=pipe._execution_device, dtype=pipe.vae.dtype
        )
        np.save(out / "ref_tokens.npy", tokens.float().cpu().numpy())

    def on_step(_pipe, step, _timestep, callback_kwargs):
        np.save(out / f"latents_step_{step:02d}.npy", callback_kwargs["latents"].float().cpu().numpy())
        return callback_kwargs

    call["callback_on_step_end"] = on_step
    call["callback_on_step_end_tensor_inputs"] = ["latents"]
    result = pipe(**call, generator=torch.Generator(device="cpu").manual_seed(args.seed), output_type="pil")
    result.images[0].save(out / "out.png")

    import diffusers
    import transformers

    (out / "meta.json").write_text(
        json.dumps(
            {
                "diffusers": diffusers.__version__,
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "base": args.base,
                "transformer": args.transformer,
                "dtype": args.dtype,
                "steps": args.steps,
                "guidance": args.guidance,
                "seed": args.seed,
                "size": [call["width"], call["height"]],
                "prompt": prompt,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
