"""
Dumps per-JointBlock layer outputs from an SD3 MMDiT forward pass.
Hooks every JointTransformerBlock and saves both image and context stream
outputs for layer-by-layer comparison against HartsyInference C#.

Usage: python dump_sd3_layer_outputs.py <model_dir> [output_dir]
Requires: pip install diffusers transformers torch safetensors sentencepiece accelerate
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path


def tensor_stats(name: str, t: torch.Tensor) -> dict:
    with torch.no_grad():
        flat = t.float().flatten()
        return {
            "name": name,
            "shape": list(t.shape),
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
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sd3_reference_tensors/layers"

    if model_dir is None:
        print("Usage: python dump_sd3_layer_outputs.py <model_dir> [output_dir]")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    from diffusers import StableDiffusion3Pipeline
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

    device = "cpu"
    dtype = torch.float32

    # Load pipeline
    print(f"Loading SD3 pipeline from {model_dir}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=dtype).to(device)

    transformer = pipe.transformer
    config = transformer.config

    print(f"  num_layers={config.num_layers}, num_attention_heads={config.num_attention_heads}")
    print(f"  sample_size={config.sample_size}, patch_size={config.patch_size}")

    # --- Prepare inputs (same as dump_sd3_reference_stats.py) ---
    prompt = "A photograph of an astronaut riding a horse"
    seed = 42
    width, height = 1024, 1024
    num_steps = 28

    # Tokenize all three encoders
    clip_l_tokens = pipe.tokenizer(prompt, max_length=77, padding="max_length",
                                    truncation=True, return_tensors="pt")
    clip_g_tokens = pipe.tokenizer_2(prompt, max_length=77, padding="max_length",
                                      truncation=True, return_tensors="pt")
    t5_tokens = pipe.tokenizer_3(prompt, max_length=77, padding="max_length",
                                  truncation=True, return_tensors="pt")

    # Encode text
    print("Encoding text...")
    with torch.no_grad():
        clip_l_out = pipe.text_encoder(clip_l_tokens["input_ids"].to(device), output_hidden_states=True)
        clip_l_hidden = clip_l_out.hidden_states[-2]
        clip_l_pooled = clip_l_out.text_embeds

        clip_g_out = pipe.text_encoder_2(clip_g_tokens["input_ids"].to(device), output_hidden_states=True)
        clip_g_hidden = clip_g_out.hidden_states[-2]
        clip_g_pooled = clip_g_out.text_embeds

        t5_out = pipe.text_encoder_3(t5_tokens["input_ids"].to(device))
        t5_hidden = t5_out.last_hidden_state

    # Combine conditioning
    lg_hidden = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)
    lg_padded = torch.nn.functional.pad(lg_hidden, (0, 4096 - 2048))
    combined_context = torch.cat([lg_padded, t5_hidden], dim=1)
    pooled = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)

    with torch.no_grad():
        projected_context = pipe.transformer.context_embedder(combined_context)

    # Create initial noise
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_shape = (1, 16, height // 8, width // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

    # Set up scheduler and get first timestep
    pipe.scheduler.set_timesteps(num_steps)
    latents = latents * pipe.scheduler.init_noise_sigma
    first_timestep = pipe.scheduler.timesteps[0]

    print(f"\nRunning single forward pass at t={float(first_timestep):.1f}...")

    # --- Hook every JointTransformerBlock ---
    layer_data = {}
    hooks = []

    # Hook patch embedding
    def patch_embed_hook(module, input, output):
        layer_data["patch_embed_output"] = output.detach().clone()
    hooks.append(transformer.pos_embed.register_forward_hook(patch_embed_hook))

    # Hook timestep embedding
    def time_embed_hook(module, input, output):
        layer_data["time_embed_output"] = output.detach().clone()
    hooks.append(transformer.time_text_embed.register_forward_hook(time_embed_hook))

    # Hook each JointTransformerBlock
    for i, block in enumerate(transformer.transformer_blocks):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # JointTransformerBlock returns (hidden_states, encoder_hidden_states)
                if isinstance(output, tuple):
                    img_out, ctx_out = output[0], output[1] if len(output) > 1 else None
                    layer_data[f"block_{layer_idx}_image"] = img_out.detach().clone()
                    if ctx_out is not None:
                        layer_data[f"block_{layer_idx}_context"] = ctx_out.detach().clone()
                else:
                    layer_data[f"block_{layer_idx}_image"] = output.detach().clone()
            return hook_fn
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Hook final norm + projection
    if hasattr(transformer, 'norm_out'):
        def norm_out_hook(module, input, output):
            layer_data["norm_out"] = output.detach().clone()
        hooks.append(transformer.norm_out.register_forward_hook(norm_out_hook))

    # Run forward pass
    latent_model_input = pipe.scheduler.scale_model_input(latents, first_timestep)

    with torch.no_grad():
        model_output = transformer(
            hidden_states=latent_model_input,
            timestep=first_timestep.unsqueeze(0),
            encoder_hidden_states=projected_context,
            pooled_projections=pooled,
        ).sample

    # Remove hooks
    for h in hooks:
        h.remove()

    # --- Save all layer outputs ---
    print("\nSaving layer outputs...")
    stats = []
    index = []

    # Save model input/output
    save_tensor(latent_model_input, os.path.join(output_dir, "model_input.bin"))
    save_tensor(model_output, os.path.join(output_dir, "model_output.bin"))
    stats.append(tensor_stats("model_input", latent_model_input))
    stats.append(tensor_stats("model_output", model_output))
    index.append({"name": "model_input", "file": "model_input.bin"})
    index.append({"name": "model_output", "file": "model_output.bin"})

    # Save conditioning inputs
    save_tensor(projected_context, os.path.join(output_dir, "context_input.bin"))
    save_tensor(pooled, os.path.join(output_dir, "pooled_input.bin"))
    index.append({"name": "context_input", "file": "context_input.bin"})
    index.append({"name": "pooled_input", "file": "pooled_input.bin"})

    # Save patch embedding output
    if "patch_embed_output" in layer_data:
        t = layer_data["patch_embed_output"]
        save_tensor(t, os.path.join(output_dir, "patch_embed_output.bin"))
        s = tensor_stats("patch_embed_output", t)
        stats.append(s)
        index.append({"name": "patch_embed_output", "file": "patch_embed_output.bin", **s})
        print(f"  patch_embed: shape={list(t.shape)}, mean={s['mean']:.6f}, std={s['std']:.6f}")

    # Save timestep embedding output
    if "time_embed_output" in layer_data:
        t = layer_data["time_embed_output"]
        save_tensor(t, os.path.join(output_dir, "time_embed_output.bin"))
        s = tensor_stats("time_embed_output", t)
        stats.append(s)
        index.append({"name": "time_embed_output", "file": "time_embed_output.bin", **s})
        print(f"  time_embed: shape={list(t.shape)}, mean={s['mean']:.6f}")

    # Save per-block outputs
    num_layers = config.num_layers
    for i in range(num_layers):
        img_key = f"block_{i}_image"
        ctx_key = f"block_{i}_context"

        if img_key in layer_data:
            t = layer_data[img_key]
            fname = f"block_{i}_image.bin"
            save_tensor(t, os.path.join(output_dir, fname))
            s = tensor_stats(img_key, t)
            stats.append(s)
            index.append({"name": img_key, "file": fname, **s})

            if i < 3 or i == num_layers - 1:
                print(f"  block_{i}_image: shape={list(t.shape)}, mean={s['mean']:.6f}, std={s['std']:.6f}")

        if ctx_key in layer_data:
            t = layer_data[ctx_key]
            fname = f"block_{i}_context.bin"
            save_tensor(t, os.path.join(output_dir, fname))
            s = tensor_stats(ctx_key, t)
            stats.append(s)
            index.append({"name": ctx_key, "file": fname, **s})

            if i < 3 or i == num_layers - 1:
                print(f"  block_{i}_context: shape={list(t.shape)}, mean={s['mean']:.6f}")

    # Save norm_out if available
    if "norm_out" in layer_data:
        t = layer_data["norm_out"]
        save_tensor(t, os.path.join(output_dir, "norm_out.bin"))
        s = tensor_stats("norm_out", t)
        stats.append(s)
        index.append({"name": "norm_out", "file": "norm_out.bin", **s})
        print(f"  norm_out: shape={list(t.shape)}, mean={s['mean']:.6f}")

    # --- Save index and stats ---
    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    with open(os.path.join(output_dir, "layer_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved {len(index)} layer outputs to {output_dir}")
    print(f"Use index.json to load specific layers in C# tests")

    # --- Print error progression summary ---
    print("\nLayer output progression (for debugging — look for divergence):")
    for entry in index:
        if entry["name"].startswith("block_"):
            print(f"  {entry['name']:30s}  mean={entry.get('mean', 0):.6f}  std={entry.get('std', 0):.6f}")


if __name__ == "__main__":
    main()
