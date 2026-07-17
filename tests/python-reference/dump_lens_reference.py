"""Dump Microsoft Lens DiT reference stages from the ComfyUI implementation (the parity oracle).

Runs `comfy.ldm.lens.model.LensTransformer2DModel` with REAL checkpoint weights (cast to F32 on CPU)
on small fixed synthetic inputs and writes every stage the C# `LensDebugDump` also emits, as raw F32
little-endian .bin files. Compare with tests/python-reference/diff_lens_layers.py after running
`LensDiffTests` on the C# side.

Usage (must run inside the ComfyUI venv — it has the comfy deps):
  cd "<SwarmUI>/dlbackend/ComfyUI" && ./venv/bin/python \
      <HartsyInference>/tests/python-reference/dump_lens_reference.py \
      --checkpoint "<Models>/Stable-Diffusion/Lens/lens_turbo_bf16.safetensors" \
      --out <HartsyInference>/tests/python-reference/lens_reference_tensors/full_forward
"""

import argparse
import os
import sys

import numpy as np
import torch

# Synthetic input geometry: 16x16 packed grid (256 image tokens), 20 text tokens.
H_PACKED = 16
W_PACKED = 16
S_TXT = 20
IN_CHANNELS = 128
ENC_DIM = 2880
NUM_ENC_LAYERS = 4
TIMESTEP = 0.7
SEED = 42


def save_f32(path: str, t: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.detach().to(torch.float32).contiguous().numpy().tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    import comfy.ops
    from comfy.ldm.lens.model import LensTransformer2DModel
    from safetensors.torch import load_file

    torch.manual_seed(SEED)
    packed = torch.randn(1, H_PACKED * W_PACKED, IN_CHANNELS, dtype=torch.float32)
    enc_layers = [torch.randn(1, S_TXT, ENC_DIM, dtype=torch.float32) for _ in range(NUM_ENC_LAYERS)]

    inputs_dir = os.path.join(args.out, "inputs")
    save_f32(os.path.join(inputs_dir, "packed_latent.bin"), packed)
    for i, layer in enumerate(enc_layers):
        save_f32(os.path.join(inputs_dir, f"enc_layer_{i}.bin"), layer)
    np.array([TIMESTEP], dtype=np.float32).tofile(os.path.join(inputs_dir, "timestep.bin"))

    print("building model (f32, cpu)...")
    model = LensTransformer2DModel(dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    print("loading checkpoint:", args.checkpoint)
    sd = load_file(args.checkpoint)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing]
    if real_missing:
        raise SystemExit(f"missing keys: {real_missing[:10]}")
    if unexpected:
        print("unexpected (ignored):", unexpected[:10])
    model.eval().float()
    del sd

    layers_dir = os.path.join(args.out, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    def dump(name: str, t: torch.Tensor) -> None:
        save_f32(os.path.join(layers_dir, name.replace(".", "_") + ".bin"), t)

    hooks = []
    hooks.append(model.img_in.register_forward_hook(lambda m, i, o: dump("img_in", o)))
    hooks.append(model.txt_in.register_forward_pre_hook(lambda m, i: dump("txt_concat", i[0])))
    hooks.append(model.txt_in.register_forward_hook(lambda m, i, o: dump("txt_in", o)))
    hooks.append(model.time_text_embed.register_forward_hook(lambda m, i, o: dump("time_text_embed", o)))
    for bi, block in enumerate(model.transformer_blocks):
        def block_hook(m, i, o, bi=bi):
            enc, hid = o
            dump(f"block_{bi}_text", enc)
            dump(f"block_{bi}_image", hid)
        hooks.append(block.register_forward_hook(block_hook))
    hooks.append(model.norm_out.register_forward_hook(lambda m, i, o: dump("norm_out", o)))
    hooks.append(model.proj_out.register_forward_hook(lambda m, i, o: dump("proj_out", o)))

    # ComfyUI bridge signature: x [B,C,h,w], context [B,S,L*H] (layer-major per token).
    x = packed.reshape(1, H_PACKED, W_PACKED, IN_CHANNELS).permute(0, 3, 1, 2).contiguous()
    context = torch.cat(enc_layers, dim=-1)
    timestep = torch.tensor([TIMESTEP], dtype=torch.float32)

    print("running reference forward...")
    with torch.no_grad():
        out = model._forward(x, timestep, context, attention_mask=None)

    for h in hooks:
        h.remove()

    # Packed velocity (matches C# transformer return): [B,C,h,w] -> [B, h*w, C]
    out_packed = out.permute(0, 2, 3, 1).reshape(1, H_PACKED * W_PACKED, IN_CHANNELS)
    save_f32(os.path.join(args.out, "output_velocity.bin"), out_packed)
    print("done. wrote", args.out)


if __name__ == "__main__":
    main()
