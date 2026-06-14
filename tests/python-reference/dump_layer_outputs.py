"""
Dumps per-layer output tensors from a single UNet forward pass.
Uses the exact same input/text embeddings as the C# tests for element-wise comparison.

Usage: python dump_layer_outputs.py
Requires: pip install diffusers transformers torch safetensors
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file


def main():
    model_dir = r"C:\Users\AI Overlord\Desktop\Projects\HartsyInference\tests\test-models\sd15"
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_tensors")
    out_dir = os.path.join(ref_dir, "layers")
    os.makedirs(out_dir, exist_ok=True)

    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextConfig

    device = "cpu"
    dtype = torch.float32

    print("Loading UNet...")
    unet = UNet2DConditionModel(
        sample_size=32, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768, attention_head_dim=8, norm_num_groups=32,
    ).to(dtype).to(device)
    unet_weights = load_file(os.path.join(model_dir, "unet", "diffusion_pytorch_model.fp16.safetensors"))
    unet_weights = {k: v.float() for k, v in unet_weights.items()}
    unet.load_state_dict(unet_weights, strict=False)
    unet.eval()

    # Load exact input tensors
    input_path = os.path.join(ref_dir, "unet_step0_input.bin")
    text_emb_path = os.path.join(ref_dir, "unet_step0_text_emb.bin")

    model_input = torch.from_numpy(np.fromfile(input_path, dtype=np.float32).reshape(1, 4, 32, 32))
    text_emb = torch.from_numpy(np.fromfile(text_emb_path, dtype=np.float32).reshape(1, 77, 768))
    timestep = 950.0

    print(f"Input: mean={model_input.mean():.6f}, std={model_input.std():.6f}")
    print(f"Text:  mean={text_emb.mean():.6f}, std={text_emb.std():.6f}")

    # Register hooks to save layer outputs
    saved = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            elif isinstance(output, dict):
                out = output.get("sample", output.get("hidden_states", None))
            else:
                out = output
            if out is not None and isinstance(out, torch.Tensor):
                saved[name] = out.detach().clone()
        return hook_fn

    # conv_in
    hooks.append(unet.conv_in.register_forward_hook(make_hook("conv_in")))
    # time_embedding
    hooks.append(unet.time_embedding.register_forward_hook(make_hook("time_embedding")))

    # Down blocks
    for i, db in enumerate(unet.down_blocks):
        for j, resnet in enumerate(db.resnets):
            hooks.append(resnet.register_forward_hook(make_hook(f"down_blocks.{i}.resnets.{j}")))
        if hasattr(db, 'attentions') and db.attentions is not None:
            for j, attn in enumerate(db.attentions):
                hooks.append(attn.register_forward_hook(make_hook(f"down_blocks.{i}.attentions.{j}")))
        if hasattr(db, 'downsamplers') and db.downsamplers is not None:
            for j, ds in enumerate(db.downsamplers):
                hooks.append(ds.register_forward_hook(make_hook(f"down_blocks.{i}.downsamplers.{j}")))

    # Mid block
    for j, resnet in enumerate(unet.mid_block.resnets):
        hooks.append(resnet.register_forward_hook(make_hook(f"mid_block.resnets.{j}")))
    for j, attn in enumerate(unet.mid_block.attentions):
        hooks.append(attn.register_forward_hook(make_hook(f"mid_block.attentions.{j}")))

    # Up blocks
    for i, ub in enumerate(unet.up_blocks):
        for j, resnet in enumerate(ub.resnets):
            hooks.append(resnet.register_forward_hook(make_hook(f"up_blocks.{i}.resnets.{j}")))
        if hasattr(ub, 'attentions') and ub.attentions is not None:
            for j, attn in enumerate(ub.attentions):
                hooks.append(attn.register_forward_hook(make_hook(f"up_blocks.{i}.attentions.{j}")))
        if hasattr(ub, 'upsamplers') and ub.upsamplers is not None:
            for j, us in enumerate(ub.upsamplers):
                hooks.append(us.register_forward_hook(make_hook(f"up_blocks.{i}.upsamplers.{j}")))

    # Final layers
    hooks.append(unet.conv_norm_out.register_forward_hook(make_hook("conv_norm_out")))
    hooks.append(unet.conv_out.register_forward_hook(make_hook("conv_out")))

    # Run forward pass
    print("\nRunning UNet forward pass...")
    with torch.no_grad():
        output = unet(model_input, timestep, encoder_hidden_states=text_emb).sample

    for h in hooks:
        h.remove()

    # Save all layer outputs as binary files
    index = {}
    print(f"\n{'Layer':<45} {'Shape':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 95)

    for name, tensor in saved.items():
        fname = name.replace(".", "_") + ".bin"
        fpath = os.path.join(out_dir, fname)
        tensor.numpy().tofile(fpath)
        index[name] = {"file": fname, "shape": list(tensor.shape), "dtype": "float32"}
        print(f"{name:<45} {str(list(tensor.shape)):<25} {tensor.mean():>10.5f} {tensor.std():>10.5f}")

    # Save output
    output.numpy().tofile(os.path.join(out_dir, "unet_output.bin"))
    index["unet_output"] = {"file": "unet_output.bin", "shape": list(output.shape), "dtype": "float32"}
    print(f"{'unet_output':<45} {str(list(output.shape)):<25} {output.mean():>10.5f} {output.std():>10.5f}")

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved {len(saved)+1} layer outputs to {out_dir}")


if __name__ == "__main__":
    main()
