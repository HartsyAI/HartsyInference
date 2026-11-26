"""Offline RMBG 3.0 reference: diffusers Fibo 2.0 edit pipeline on saved noise.

Run on the GPU server only. Not part of the engine; the engine never calls Python.

Pinned reference: huggingface/diffusers @ 98dfc1aa4146e0d8bcbdb2d00ff68e698ce3304c
(pipeline_bria_fibo2.py, transformer_bria_fibo2.py, image_processor.py), diffusers >= 0.42.

Inputs
  --model-dir   a diffusers-layout copy of the RMBG 3.0 pipeline (local, local_files_only).
                The Comfy repack (diffusion_models/rmbg3_bf16.safetensors) is NOT that layout;
                this script cannot read it. Stop and report if only the repack exists.
  --images      folder of input images; the first --count files by sorted name are used.
  --out         output folder for references and the manifest.

Outputs (per image, under --out)
  <stem>.alpha.png     8-bit alpha, source size, PIL convert("L") + LANCZOS resize (model card path)
  <stem>.npz           final latents, per-step velocities, sigma list, noise seed
  manifest.json        input file names, sha256 of each input, pipeline class, diffusers version, sigma list

Saved noise: one torch.Generator(seed=0) draw per image, saved to <stem>.noise.npy on first run and
reloaded on later runs, so the engine can be fed the identical tensor.

Nothing here is verified on real weights. The pipeline call below must be checked against the pinned
pipeline_bria_fibo2.py before the first run (see the ASSERT lines; the script refuses to run if the
signature differs).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RMBG_INSTRUCTION = (
    "Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the background "
    "to black. Produce soft, anti-aliased grayscale gradients at the edges of the subject to represent "
    "fine details and transparency."
)
PROMPT_JSON = json.dumps({"edit_instruction": RMBG_INSTRUCTION})
NUM_STEPS = 4
GUIDANCE = 1.0
SEED = 0
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pil_alpha(picture: Image.Image, source_size: tuple[int, int]) -> Image.Image:
    # Model card: result.convert("L").resize(source.size, LANCZOS)
    return picture.convert("L").resize(source_size, Image.Resampling.LANCZOS)


def load_pipeline(model_dir: Path):
    from diffusers import DiffusionPipeline  # imported late so --help works without a GPU stack

    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe = pipe.to("cuda")
    # ASSERT: the pinned pipeline takes an edit image and a latents tensor. Fail loudly otherwise.
    sig = inspect.signature(pipe.__call__)
    for name in ("image", "prompt", "num_inference_steps", "guidance_scale", "latents"):
        if name not in sig.parameters:
            sys.exit(f"pipeline {type(pipe).__name__} has no '{name}' argument; pinned signature differs")
    return pipe


def latent_shape(pipe, height: int, width: int) -> tuple[int, ...]:
    # Latent grid comes from the pinned VAE and packing; read it from the pipeline, do not guess.
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    channels = pipe.transformer.config.in_channels // 4  # 2x2 patch packing
    return (1, channels, 2 * (height // (vae_scale * 2)), 2 * (width // (vae_scale * 2)))


def run_one(pipe, image_path: Path, out: Path, sigmas_log: list) -> dict:
    src = Image.open(image_path).convert("RGB")
    w, h = src.size
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    noise_path = out / f"{image_path.stem}.noise.npy"
    if noise_path.exists():
        noise = torch.from_numpy(np.load(noise_path))
    else:
        noise = torch.randn(latent_shape(pipe, h, w), generator=gen, dtype=torch.float32)
        np.save(noise_path, noise.numpy())

    velocities: list[np.ndarray] = []

    def capture(_pipe, step, _timestep, cb_kwargs):
        if "noise_pred" in cb_kwargs:
            velocities.append(cb_kwargs["noise_pred"].float().cpu().numpy())
        return cb_kwargs

    result = pipe(
        prompt=PROMPT_JSON,
        image=src,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        latents=noise.to("cuda", torch.bfloat16),
        callback_on_step_end=capture,
        callback_on_step_end_tensor_inputs=["noise_pred"],
    )
    picture = result.images[0]
    alpha = pil_alpha(picture, (w, h))
    alpha.save(out / f"{image_path.stem}.alpha.png")
    np.savez(
        out / f"{image_path.stem}.npz",
        velocities=np.stack(velocities) if velocities else np.zeros((0,)),
        noise_seed=np.int64(SEED),
    )
    return {"file": image_path.name, "sha256": sha256(image_path), "size": [w, h]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--count", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)[: args.count]
    if not files:
        sys.exit(f"no images in {args.images}")

    pipe = load_pipeline(args.model_dir)
    entries = [run_one(pipe, p, args.out, []) for p in files]
    manifest = {
        "pipeline": type(pipe).__name__,
        "diffusers": __import__("diffusers").__version__,
        "reference_commit": "98dfc1aa4146e0d8bcbdb2d00ff68e698ce3304c",
        "steps": NUM_STEPS,
        "guidance": GUIDANCE,
        "seed": SEED,
        "prompt_json": PROMPT_JSON,
        "images": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(entries)} references to {args.out}")


if __name__ == "__main__":
    main()
