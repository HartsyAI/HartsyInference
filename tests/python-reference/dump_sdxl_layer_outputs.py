"""
Dumps per-layer output tensors from a single SDXL UNet forward pass.
Hooks every major block and saves each output as a binary tensor for
element-wise comparison against C# implementation.

Matching C# test params:
  - Uses Python step0_scaled_input as UNet input
  - Timestep: 900.0
  - Text embeddings from dump_sdxl_reference_stats.py
  - Same JuggernautXL checkpoint

Output: tests/python-reference/sdxl_reference_tensors/layers/
  - time_embedding.bin
  - add_embedding.bin
  - conv_in.bin
  - down_blocks.{i}.resnets.{j}.bin
  - down_blocks.{i}.attentions.{j}.bin
  - down_blocks.{i}.downsamplers.0.bin
  - mid_block.resnets.0.bin
  - mid_block.attentions.0.bin
  - mid_block.resnets.1.bin
  - up_blocks.{i}.resnets.{j}.bin
  - up_blocks.{i}.attentions.{j}.bin
  - up_blocks.{i}.upsamplers.0.bin
  - conv_norm_out.bin (after GroupNorm + SiLU)
  - conv_out.bin (final output)
  - index.json (metadata for each layer)

Usage: python dump_sdxl_layer_outputs.py [checkpoint_path]
Requires: pip install diffusers transformers torch safetensors accelerate
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path


def tensor_stats(name, t):
    flat = t.float().flatten()
    return {
        "name": name,
        "shape": list(t.shape),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def save_tensor(out_dir, name, t, index):
    """Save tensor as binary and add entry to index."""
    filename = f"{name}.bin"
    filepath = os.path.join(out_dir, filename)
    t.detach().float().numpy().tofile(filepath)
    entry = tensor_stats(name, t)
    entry["file"] = filename
    index.append(entry)
    print(f"  {name}: shape={list(t.shape)}, mean={float(t.float().mean()):.6f}, std={float(t.float().std()):.6f}")


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\SDXL\Juggernaut_XL_-_Ragnarok_by_RunDiffusion.safetensors"
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdxl_reference_tensors")
    out_dir = os.path.join(ref_dir, "layers")
    os.makedirs(out_dir, exist_ok=True)

    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

    device = "cpu"
    dtype = torch.float32

    print(f"Loading SDXL pipeline from: {checkpoint_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=dtype,
    ).to(device)

    unet = pipe.unet
    unet.eval()

    # Load the same inputs used by C# cross-runtime test
    input_path = os.path.join(ref_dir, "step0_scaled_input.bin")
    text_emb_path = os.path.join(ref_dir, "text_embeddings.bin")
    pooled_path = os.path.join(ref_dir, "clip_g_pooled.bin")

    if not os.path.exists(input_path):
        print(f"ERROR: Run dump_sdxl_reference_stats.py first to generate reference tensors")
        print(f"  Missing: {input_path}")
        sys.exit(1)

    print("Loading reference tensors...")
    scaled_input = torch.from_numpy(np.fromfile(input_path, dtype=np.float32).reshape(1, 4, 32, 32))
    text_embeddings = torch.from_numpy(np.fromfile(text_emb_path, dtype=np.float32).reshape(2, 77, 2048))
    pooled = torch.from_numpy(np.fromfile(pooled_path, dtype=np.float32).reshape(2, 1280))

    # Use unconditional (neg) pass — batch element 0
    sample = scaled_input  # [1, 4, 32, 32]
    encoder_hidden_states = text_embeddings[0:1]  # [1, 77, 2048]
    text_embeds = pooled[0:1]  # [1, 1280]

    timestep = torch.tensor([900.0])

    # Size conditioning: [origH, origW, cropTop, cropLeft, targetH, targetW]
    add_time_ids = torch.tensor([[256.0, 256.0, 0.0, 0.0, 256.0, 256.0]])

    added_cond_kwargs = {
        "text_embeds": text_embeds,
        "time_ids": add_time_ids,
    }

    print(f"  sample: {list(sample.shape)}")
    print(f"  encoder_hidden_states: {list(encoder_hidden_states.shape)}")
    print(f"  text_embeds: {list(text_embeds.shape)}")
    print(f"  timestep: {float(timestep[0])}")

    # =========================================================================
    # Hook every major block to capture intermediate outputs
    # =========================================================================
    index = []
    hooks = []
    captured = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                # Some blocks return (hidden, *extras)
                captured[name] = output[0].detach()
            elif isinstance(output, torch.Tensor):
                captured[name] = output.detach()
            else:
                # Some modules return dicts or other types
                if hasattr(output, 'sample'):
                    captured[name] = output.sample.detach()
        return hook_fn

    # Register hooks on all major UNet blocks
    print("\n=== Registering hooks ===")

    # Time embedding
    hooks.append(unet.time_embedding.register_forward_hook(make_hook("time_embedding")))

    # Addition embedding (SDXL ADM)
    if hasattr(unet, 'add_embedding'):
        hooks.append(unet.add_embedding.register_forward_hook(make_hook("add_embedding")))

    # conv_in
    hooks.append(unet.conv_in.register_forward_hook(make_hook("conv_in")))

    # Down blocks
    for i, block in enumerate(unet.down_blocks):
        if hasattr(block, 'resnets'):
            for j, resnet in enumerate(block.resnets):
                hooks.append(resnet.register_forward_hook(make_hook(f"down_blocks.{i}.resnets.{j}")))
        if hasattr(block, 'attentions') and block.attentions is not None:
            for j, attn in enumerate(block.attentions):
                hooks.append(attn.register_forward_hook(make_hook(f"down_blocks.{i}.attentions.{j}")))
        if hasattr(block, 'downsamplers') and block.downsamplers is not None:
            for j, ds in enumerate(block.downsamplers):
                hooks.append(ds.register_forward_hook(make_hook(f"down_blocks.{i}.downsamplers.{j}")))

    # Mid block
    if hasattr(unet.mid_block, 'resnets'):
        for j, resnet in enumerate(unet.mid_block.resnets):
            hooks.append(resnet.register_forward_hook(make_hook(f"mid_block.resnets.{j}")))
    if hasattr(unet.mid_block, 'attentions') and unet.mid_block.attentions is not None:
        for j, attn in enumerate(unet.mid_block.attentions):
            hooks.append(attn.register_forward_hook(make_hook(f"mid_block.attentions.{j}")))

    # Up blocks
    for i, block in enumerate(unet.up_blocks):
        if hasattr(block, 'resnets'):
            for j, resnet in enumerate(block.resnets):
                hooks.append(resnet.register_forward_hook(make_hook(f"up_blocks.{i}.resnets.{j}")))
        if hasattr(block, 'attentions') and block.attentions is not None:
            for j, attn in enumerate(block.attentions):
                hooks.append(attn.register_forward_hook(make_hook(f"up_blocks.{i}.attentions.{j}")))
        if hasattr(block, 'upsamplers') and block.upsamplers is not None:
            for j, us in enumerate(block.upsamplers):
                hooks.append(us.register_forward_hook(make_hook(f"up_blocks.{i}.upsamplers.{j}")))

    # Final norm + conv
    hooks.append(unet.conv_norm_out.register_forward_hook(make_hook("conv_norm_out")))
    hooks.append(unet.conv_out.register_forward_hook(make_hook("conv_out")))

    print(f"  Registered {len(hooks)} hooks")

    # =========================================================================
    # Run forward pass
    # =========================================================================
    print("\n=== Running UNet forward pass (uncond) ===")
    with torch.no_grad():
        output = unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

    # =========================================================================
    # Save all captured tensors
    # =========================================================================
    print(f"\n=== Saving {len(captured)} layer outputs ===")

    # Save in execution order
    layer_order = [
        "time_embedding",
        "add_embedding",
        "conv_in",
    ]

    # Down blocks
    for i in range(len(unet.down_blocks)):
        block = unet.down_blocks[i]
        num_resnets = len(block.resnets) if hasattr(block, 'resnets') else 0
        num_attns = len(block.attentions) if hasattr(block, 'attentions') and block.attentions is not None else 0
        for j in range(max(num_resnets, num_attns)):
            if j < num_resnets:
                layer_order.append(f"down_blocks.{i}.resnets.{j}")
            if j < num_attns:
                layer_order.append(f"down_blocks.{i}.attentions.{j}")
        if hasattr(block, 'downsamplers') and block.downsamplers is not None:
            layer_order.append(f"down_blocks.{i}.downsamplers.0")

    # Mid block
    layer_order.append("mid_block.resnets.0")
    layer_order.append("mid_block.attentions.0")
    layer_order.append("mid_block.resnets.1")

    # Up blocks
    for i in range(len(unet.up_blocks)):
        block = unet.up_blocks[i]
        num_resnets = len(block.resnets) if hasattr(block, 'resnets') else 0
        num_attns = len(block.attentions) if hasattr(block, 'attentions') and block.attentions is not None else 0
        for j in range(max(num_resnets, num_attns)):
            if j < num_resnets:
                layer_order.append(f"up_blocks.{i}.resnets.{j}")
            if j < num_attns:
                layer_order.append(f"up_blocks.{i}.attentions.{j}")
        if hasattr(block, 'upsamplers') and block.upsamplers is not None:
            layer_order.append(f"up_blocks.{i}.upsamplers.0")

    layer_order.extend(["conv_norm_out", "conv_out"])

    for name in layer_order:
        if name in captured:
            save_tensor(out_dir, name, captured[name], index)
        else:
            print(f"  WARNING: {name} not captured")

    # Also save final output
    save_tensor(out_dir, "final_output", output, index)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save index
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Saved {len(index)} layer tensors to: {out_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
