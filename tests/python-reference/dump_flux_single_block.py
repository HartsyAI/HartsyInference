"""
Dumps a single Flux SingleStreamBlock's inputs/outputs from a diffusers model.
Hooks the first SingleStreamBlock (single_transformer_blocks.0) and saves all
intermediate tensors for layer-by-layer validation against HartsyInference C#.

Usage: python dump_flux_single_block.py <model_dir> [output_dir]
  model_dir: path to Flux diffusers model (e.g., black-forest-labs/FLUX.1-schnell)
Requires: pip install diffusers transformers torch safetensors sentencepiece accelerate
"""
import sys
import os
import json
import torch
import numpy as np


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
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "flux_reference_tensors/single_block"

    if model_dir is None:
        print("Usage: python dump_flux_single_block.py <model_dir> [output_dir]")
        print("  model_dir: path to Flux diffusers model directory or HF model ID")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    from diffusers import FluxPipeline

    device = "cpu"
    dtype = torch.float32

    prompt = "A photograph of an astronaut riding a horse"
    width, height = 256, 256
    seed = 42
    num_steps = 4

    stats = []
    stats.append({
        "name": "settings",
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "num_steps": num_steps,
    })

    # Load pipeline
    print(f"Loading Flux pipeline from {model_dir}...")
    pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=dtype).to(device)

    transformer = pipe.transformer
    config = transformer.config

    print(f"  Config: num_layers={config.num_layers}, "
          f"num_single_layers={config.num_single_layers}")

    # --- Hook first SingleStreamBlock ---
    print("\nHooking first SingleStreamBlock (single_transformer_blocks.0)...")
    block_data = {}
    hooks = []

    first_single_block = transformer.single_transformer_blocks[0]

    # Pre-hook to capture inputs
    def block_pre_hook(module, args, kwargs):
        if len(args) > 0:
            block_data["input_hidden_states"] = args[0].detach().clone()
        if "hidden_states" in kwargs:
            block_data["input_hidden_states"] = kwargs["hidden_states"].detach().clone()
        if len(args) > 1:
            block_data["input_temb"] = args[1].detach().clone()
        if "temb" in kwargs:
            block_data["input_temb"] = kwargs["temb"].detach().clone()

    hooks.append(first_single_block.register_forward_pre_hook(block_pre_hook, with_kwargs=True))

    # Post-hook for output
    def block_post_hook(module, input, output):
        if isinstance(output, tuple):
            block_data["output_hidden_states"] = output[0].detach().clone()
        else:
            block_data["output_hidden_states"] = output.detach().clone()

    hooks.append(first_single_block.register_forward_hook(block_post_hook))

    # Hook internal norm
    if hasattr(first_single_block, 'norm'):
        def norm_hook(module, input, output):
            if isinstance(output, tuple):
                block_data["norm_output"] = output[0].detach().clone() if output[0] is not None else None
            else:
                block_data["norm_output"] = output.detach().clone()
        hooks.append(first_single_block.norm.register_forward_hook(norm_hook))

    # Hook proj_out
    if hasattr(first_single_block, 'proj_out'):
        def proj_out_hook(module, input, output):
            block_data["proj_out_output"] = output.detach().clone()
        hooks.append(first_single_block.proj_out.register_forward_hook(proj_out_hook))

    # Also hook the last double block to see what goes into the single stream
    last_double_idx = len(transformer.transformer_blocks) - 1
    last_double = transformer.transformer_blocks[last_double_idx]

    def last_double_hook(module, input, output):
        if isinstance(output, tuple):
            block_data["last_double_img"] = output[0].detach().clone()
            if len(output) > 1 and output[1] is not None:
                block_data["last_double_ctx"] = output[1].detach().clone()

    hooks.append(last_double.register_forward_hook(last_double_hook))

    # --- Run pipeline ---
    print("Running pipeline...")
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=0.0,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="np",
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # --- Save captured data ---
    print("\nSaving captured single block data...")

    for key, tensor in block_data.items():
        if tensor is not None and isinstance(tensor, torch.Tensor):
            fname = f"single0_{key}.bin"
            save_tensor(tensor, os.path.join(output_dir, fname))
            s = tensor_stats(f"single0_{key}", tensor)
            stats.append(s)
            print(f"  {key}: shape={list(tensor.shape)}, mean={s['mean']:.6f}, std={s['std']:.6f}")

    # --- Save weight shapes ---
    print("\nSingleStreamBlock 0 weight shapes:")
    weight_shapes = {}
    for name, param in first_single_block.named_parameters():
        weight_shapes[name] = list(param.shape)
        print(f"  {name}: {list(param.shape)}")

    stats.append({"name": "single_block0_weight_shapes", "shapes": weight_shapes})

    # --- Save stats ---
    with open(os.path.join(output_dir, "flux_single_block_reference.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll single block reference data saved to {output_dir}")


if __name__ == "__main__":
    main()
