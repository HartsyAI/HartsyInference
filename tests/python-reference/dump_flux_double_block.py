"""
Dumps a single Flux DoubleStreamBlock's inputs/outputs from a diffusers model.
Hooks the first DoubleStreamBlock (transformer_blocks.0) and saves all intermediate
tensors for layer-by-layer validation against HartsyInference C#.

Usage: python dump_flux_double_block.py <model_dir> [output_dir]
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
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "flux_reference_tensors/double_block"

    if model_dir is None:
        print("Usage: python dump_flux_double_block.py <model_dir> [output_dir]")
        print("  model_dir: path to Flux diffusers model directory or HF model ID")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    from diffusers import FluxPipeline

    device = "cpu"
    dtype = torch.float32

    # Use small resolution for tractable CPU computation
    prompt = "A photograph of an astronaut riding a horse"
    width, height = 256, 256  # Small for fast CPU reference
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
          f"num_single_layers={config.num_single_layers}, "
          f"num_attention_heads={config.num_attention_heads}, "
          f"joint_attention_dim={config.joint_attention_dim}")

    # --- Text encoding ---
    print("\nEncoding text...")
    with torch.no_grad():
        # CLIP-L pooled
        clip_tokens = pipe.tokenizer(prompt, max_length=77, padding="max_length",
                                     truncation=True, return_tensors="pt")
        clip_out = pipe.text_encoder(clip_tokens["input_ids"].to(device),
                                     output_hidden_states=False)
        clip_pooled = clip_out.pooler_output  # [1, 768]

        # T5-XXL per-token
        t5_max_len = 512 if hasattr(config, 'max_seq_len') else 256
        t5_tokens = pipe.tokenizer_2(prompt, max_length=t5_max_len, padding="max_length",
                                     truncation=True, return_tensors="pt")
        t5_out = pipe.text_encoder_2(t5_tokens["input_ids"].to(device))
        t5_hidden = t5_out.last_hidden_state  # [1, seq_len, 4096]

    print(f"  CLIP pooled: {clip_pooled.shape}")
    print(f"  T5 hidden: {t5_hidden.shape}")

    save_tensor(clip_pooled, os.path.join(output_dir, "clip_pooled.bin"))
    save_tensor(t5_hidden, os.path.join(output_dir, "t5_hidden.bin"))
    stats.append(tensor_stats("clip_pooled", clip_pooled))
    stats.append(tensor_stats("t5_hidden", t5_hidden))

    # --- Prepare latents ---
    print("\nPreparing latents...")
    generator = torch.Generator(device=device).manual_seed(seed)
    h_lat, w_lat = height // 8, width // 8
    latents = torch.randn(1, 16, h_lat, w_lat, generator=generator, device=device, dtype=dtype)

    print(f"  Latent shape: {latents.shape}")
    save_tensor(latents, os.path.join(output_dir, "initial_latents.bin"))
    stats.append(tensor_stats("initial_latents", latents))

    # --- Hook the first DoubleStreamBlock ---
    print("\nHooking first DoubleStreamBlock (transformer_blocks.0)...")
    block_data = {}
    hooks = []

    # Hook block input (via pre-hook on the first double block)
    first_block = transformer.transformer_blocks[0]

    def block_pre_hook(module, args, kwargs):
        # FluxTransformerBlock2DModel.forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb)
        if len(args) > 0:
            block_data["input_hidden_states"] = args[0].detach().clone()
        if "hidden_states" in kwargs:
            block_data["input_hidden_states"] = kwargs["hidden_states"].detach().clone()
        if len(args) > 1:
            block_data["input_encoder_hidden_states"] = args[1].detach().clone()
        if "encoder_hidden_states" in kwargs:
            block_data["input_encoder_hidden_states"] = kwargs["encoder_hidden_states"].detach().clone()
        if len(args) > 2:
            block_data["input_temb"] = args[2].detach().clone()
        if "temb" in kwargs:
            block_data["input_temb"] = kwargs["temb"].detach().clone()

    hooks.append(first_block.register_forward_pre_hook(block_pre_hook, with_kwargs=True))

    # Hook block output
    def block_post_hook(module, input, output):
        if isinstance(output, tuple):
            block_data["output_hidden_states"] = output[0].detach().clone()
            if len(output) > 1 and output[1] is not None:
                block_data["output_encoder_hidden_states"] = output[1].detach().clone()
        else:
            block_data["output_hidden_states"] = output.detach().clone()

    hooks.append(first_block.register_forward_hook(block_post_hook))

    # Hook internal components of the first block
    if hasattr(first_block, 'norm1'):
        def norm1_hook(module, input, output):
            if isinstance(output, tuple):
                block_data["norm1_output"] = output[0].detach().clone() if output[0] is not None else None
            else:
                block_data["norm1_output"] = output.detach().clone()
        hooks.append(first_block.norm1.register_forward_hook(norm1_hook))

    if hasattr(first_block, 'norm1_context'):
        def norm1_ctx_hook(module, input, output):
            if isinstance(output, tuple):
                block_data["norm1_context_output"] = output[0].detach().clone() if output[0] is not None else None
            else:
                block_data["norm1_context_output"] = output.detach().clone()
        hooks.append(first_block.norm1_context.register_forward_hook(norm1_ctx_hook))

    # --- Run single forward pass ---
    print("Running pipeline (single step for block capture)...")
    with torch.no_grad():
        # Use the pipeline's __call__ to get proper text encoding and latent preparation
        # But we only need 1 step to capture block data
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=0.0,  # Schnell: no guidance
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="np",
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # --- Save captured data ---
    print("\nSaving captured block data...")

    for key, tensor in block_data.items():
        if tensor is not None and isinstance(tensor, torch.Tensor):
            fname = f"block0_{key}.bin"
            save_tensor(tensor, os.path.join(output_dir, fname))
            s = tensor_stats(f"block0_{key}", tensor)
            stats.append(s)
            print(f"  {key}: shape={list(tensor.shape)}, mean={s['mean']:.6f}, std={s['std']:.6f}")

    # --- Save weight shapes for verification ---
    print("\nBlock 0 weight shapes:")
    weight_shapes = {}
    for name, param in first_block.named_parameters():
        weight_shapes[name] = list(param.shape)
        if "norm" in name or "linear" in name or "proj" in name:
            print(f"  {name}: {list(param.shape)}")

    stats.append({"name": "block0_weight_shapes", "shapes": weight_shapes})

    # --- Save final image ---
    if result.images is not None and len(result.images) > 0:
        img_np = result.images[0]
        np.save(os.path.join(output_dir, "output_image.npy"), img_np)
        print(f"\n  Output image shape: {img_np.shape}, range: [{img_np.min():.3f}, {img_np.max():.3f}]")

    # --- Save stats ---
    with open(os.path.join(output_dir, "flux_double_block_reference.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll double block reference data saved to {output_dir}")


if __name__ == "__main__":
    main()
