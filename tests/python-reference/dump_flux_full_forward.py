"""
Dumps a full FluxTransformer2DModel forward pass with per-block outputs.
Hooks every DoubleStreamBlock and SingleStreamBlock, plus embeddings and final layer.
Saves all tensors for comprehensive layer-by-layer validation.

Usage: python dump_flux_full_forward.py <model_dir> [output_dir]
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
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "flux_reference_tensors/full_forward"

    if model_dir is None:
        print("Usage: python dump_flux_full_forward.py <model_dir> [output_dir]")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    from diffusers import FluxTransformer2DModel
    from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings

    device = "cpu"
    dtype = torch.float32

    # --- Load transformer only ---
    print(f"Loading FluxTransformer2DModel from {model_dir}...")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    transformer.eval()

    config = transformer.config
    num_double = config.num_layers
    num_single = config.num_single_layers
    hidden_size = config.num_attention_heads * (config.attention_head_dim if hasattr(config, 'attention_head_dim') else 128)
    has_guidance = getattr(config, 'guidance_embeds', False)

    print(f"  Double blocks: {num_double}, Single blocks: {num_single}")
    print(f"  Hidden size: {hidden_size}, Heads: {config.num_attention_heads}")
    print(f"  Guidance embed: {has_guidance}")

    stats = []
    stats.append({
        "name": "config",
        "num_double_blocks": num_double,
        "num_single_blocks": num_single,
        "hidden_size": hidden_size,
        "num_heads": config.num_attention_heads,
        "has_guidance": has_guidance,
    })

    # --- Prepare synthetic inputs ---
    print("\nPreparing synthetic inputs...")
    torch.manual_seed(42)

    B = 1
    h_lat, w_lat = 32, 32  # 256x256 image -> 32x32 latent
    h_packed, w_packed = h_lat // 2, w_lat // 2  # 16x16
    img_seq_len = h_packed * w_packed  # 256
    txt_seq_len = 128

    # Packed latent input: [B, img_seq_len, in_channels=64]
    hidden_states = torch.randn(B, img_seq_len, 64, device=device, dtype=dtype)

    # T5 text embeddings: [B, txt_seq_len, 4096]
    encoder_hidden_states = torch.randn(B, txt_seq_len, 4096, device=device, dtype=dtype)

    # CLIP pooled: [B, 768]
    pooled_projections = torch.randn(B, 768, device=device, dtype=dtype)

    # Timestep: scalar sigma
    timestep = torch.tensor([0.8], device=device, dtype=dtype)  # High noise

    # Guidance (dev only)
    guidance = torch.tensor([3.5], device=device, dtype=dtype) if has_guidance else None

    # Position IDs
    txt_ids = torch.zeros(txt_seq_len, 3, device=device, dtype=dtype)
    img_ids = torch.zeros(img_seq_len, 3, device=device, dtype=dtype)
    for row in range(h_packed):
        for col in range(w_packed):
            idx = row * w_packed + col
            img_ids[idx, 1] = row
            img_ids[idx, 2] = col

    # Save all inputs
    for name, tensor in [
        ("hidden_states", hidden_states),
        ("encoder_hidden_states", encoder_hidden_states),
        ("pooled_projections", pooled_projections),
        ("timestep", timestep),
        ("txt_ids", txt_ids),
        ("img_ids", img_ids),
    ]:
        save_tensor(tensor, os.path.join(output_dir, f"input_{name}.bin"))
        stats.append(tensor_stats(f"input_{name}", tensor))
    if guidance is not None:
        save_tensor(guidance, os.path.join(output_dir, "input_guidance.bin"))
        stats.append(tensor_stats("input_guidance", guidance))

    # --- Register hooks on all blocks ---
    print("\nRegistering hooks on all blocks...")
    layer_data = {}
    hooks = []

    # Hook x_embedder (img_in)
    if hasattr(transformer, 'x_embedder'):
        def x_embed_hook(module, input, output):
            layer_data["x_embedder"] = output.detach().clone()
        hooks.append(transformer.x_embedder.register_forward_hook(x_embed_hook))

    # Hook context_embedder (txt_in)
    if hasattr(transformer, 'context_embedder'):
        def ctx_embed_hook(module, input, output):
            layer_data["context_embedder"] = output.detach().clone()
        hooks.append(transformer.context_embedder.register_forward_hook(ctx_embed_hook))

    # Hook time_text_embed (timestep + pooled projection)
    if hasattr(transformer, 'time_text_embed'):
        def time_embed_hook(module, input, output):
            layer_data["time_text_embed"] = output.detach().clone()
        hooks.append(transformer.time_text_embed.register_forward_hook(time_embed_hook))

    # Hook every double block
    for i, block in enumerate(transformer.transformer_blocks):
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    layer_data[f"double_{idx}_img"] = output[0].detach().clone()
                    if len(output) > 1 and output[1] is not None:
                        layer_data[f"double_{idx}_ctx"] = output[1].detach().clone()
                else:
                    layer_data[f"double_{idx}_img"] = output.detach().clone()
            return hook_fn
        hooks.append(block.register_forward_hook(make_hook(i)))

    # Hook every single block
    for i, block in enumerate(transformer.single_transformer_blocks):
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    layer_data[f"single_{idx}_out"] = output[0].detach().clone()
                else:
                    layer_data[f"single_{idx}_out"] = output.detach().clone()
            return hook_fn
        hooks.append(block.register_forward_hook(make_hook(i)))

    # Hook norm_out
    if hasattr(transformer, 'norm_out'):
        def norm_out_hook(module, input, output):
            if isinstance(output, tuple):
                layer_data["norm_out"] = output[0].detach().clone()
            else:
                layer_data["norm_out"] = output.detach().clone()
        hooks.append(transformer.norm_out.register_forward_hook(norm_out_hook))

    # --- Run forward pass ---
    print("Running forward pass...")
    with torch.no_grad():
        kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep / 1000.0,  # diffusers expects [0,1] range
            "txt_ids": txt_ids,
            "img_ids": img_ids,
        }
        if guidance is not None:
            kwargs["guidance"] = guidance

        output = transformer(**kwargs)
        model_output = output.sample if hasattr(output, 'sample') else output[0]

    # Remove hooks
    for h in hooks:
        h.remove()

    # --- Save all layer outputs ---
    print("\nSaving layer outputs...")

    # Model output
    save_tensor(model_output, os.path.join(output_dir, "model_output.bin"))
    stats.append(tensor_stats("model_output", model_output))
    print(f"  model_output: shape={list(model_output.shape)}, "
          f"mean={stats[-1]['mean']:.6f}, std={stats[-1]['std']:.6f}")

    # Embedding outputs
    for key in ["x_embedder", "context_embedder", "time_text_embed"]:
        if key in layer_data:
            t = layer_data[key]
            save_tensor(t, os.path.join(output_dir, f"{key}.bin"))
            s = tensor_stats(key, t)
            stats.append(s)
            print(f"  {key}: shape={list(t.shape)}, mean={s['mean']:.6f}")

    # Double block outputs
    for i in range(num_double):
        for suffix in ["img", "ctx"]:
            key = f"double_{i}_{suffix}"
            if key in layer_data:
                t = layer_data[key]
                save_tensor(t, os.path.join(output_dir, f"{key}.bin"))
                s = tensor_stats(key, t)
                stats.append(s)
                if i < 3 or i == num_double - 1:
                    print(f"  {key}: shape={list(t.shape)}, mean={s['mean']:.6f}, std={s['std']:.6f}")

    # Single block outputs
    for i in range(num_single):
        key = f"single_{i}_out"
        if key in layer_data:
            t = layer_data[key]
            save_tensor(t, os.path.join(output_dir, f"{key}.bin"))
            s = tensor_stats(key, t)
            stats.append(s)
            if i < 3 or i == num_single - 1:
                print(f"  {key}: shape={list(t.shape)}, mean={s['mean']:.6f}, std={s['std']:.6f}")

    # Norm out
    if "norm_out" in layer_data:
        t = layer_data["norm_out"]
        save_tensor(t, os.path.join(output_dir, "norm_out.bin"))
        stats.append(tensor_stats("norm_out", t))

    # --- Save stats ---
    with open(os.path.join(output_dir, "flux_full_forward_reference.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll full forward reference data saved to {output_dir}")
    print(f"  Double blocks: {num_double}, Single blocks: {num_single}")
    print(f"  Total tensors saved: {len([k for k in layer_data])}")


if __name__ == "__main__":
    main()
