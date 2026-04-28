"""
Dumps a full ZImageTransformer2DModel forward pass with per-block outputs using
deterministic synthetic inputs (no Qwen3, no real captions).

This is THE source-of-truth for layer-by-layer C# diff. The C# transformer must
produce byte-equivalent output to these tensors when fed the same inputs from
inputs/{latent.bin, caption.bin}.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_zimage_full_forward.py

Output: tests/python-reference/zimage_reference_tensors/full_forward/
  inputs/latent.bin              [1, 16, 32, 32] F32 (deterministic seed 42)
  inputs/caption.bin             [1, 64, 2560]   F32 (deterministic seed 42)
  layers/<safe_name>.bin         per-layer F32 outputs (38 files)
  output_velocity.bin            final transformer output [1, 16, 32, 32] F32
  index.json                     stats per layer + config
"""
import os
import json
import torch
import numpy as np
from diffusers import ZImageTransformer2DModel

MODEL_DIR = "tests/test-models/zimage-turbo"
OUT_DIR = "tests/python-reference/zimage_reference_tensors/full_forward"
SIGMA = 0.5  # Mid-schedule timestep

os.makedirs(f"{OUT_DIR}/inputs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/layers", exist_ok=True)


def save(t: torch.Tensor, path: str):
    t.float().detach().cpu().contiguous().numpy().tofile(path)


def stats(name: str, t: torch.Tensor) -> dict:
    f = t.float().flatten()
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "min": float(f.min()),
        "max": float(f.max()),
        "abs_mean": float(f.abs().mean()),
        "first_8": [float(x) for x in f[:8]],
    }


# ── Deterministic synthetic inputs ──
torch.manual_seed(42)
B, C, H, W = 1, 16, 32, 32  # 64x64 image (latent 32x32 with 8x VAE)
CAP_LEN = 64
CAP_FEAT_DIM = 2560

latent_bchw = torch.randn(B, C, H, W, dtype=torch.float32)
caption_bld = torch.randn(B, CAP_LEN, CAP_FEAT_DIM, dtype=torch.float32)

save(latent_bchw, f"{OUT_DIR}/inputs/latent.bin")
save(caption_bld, f"{OUT_DIR}/inputs/caption.bin")
print(f"Saved synthetic inputs: latent {tuple(latent_bchw.shape)}, caption {tuple(caption_bld.shape)}")

# ── Load transformer in F32 (so reference matches FP8 path's F32 dequantized weights) ──
print(f"Loading ZImageTransformer2DModel from {MODEL_DIR}/transformer ...")
xfm = ZImageTransformer2DModel.from_pretrained(
    MODEL_DIR, subfolder="transformer", torch_dtype=torch.float32
).eval()
print(f"  Loaded. dim={xfm.dim}, n_heads={xfm.n_heads}, layers={len(xfm.layers)}, "
      f"refiners={len(xfm.noise_refiner)}")

# ── Prepare model inputs in the diffusers list-of-tensors format ──
# Image: (C, F, H, W) per batch item — F=1 for 2D image
# Caption: (cap_len, cap_feat_dim) per batch item
x_input = [latent_bchw[0].unsqueeze(1)]  # [16, 1, 32, 32]
cap_input = [caption_bld[0]]  # [64, 2560]

# Timestep: pipeline does (1000 - t) / 1000. With sigma=0.5, scheduler timestep is 500 (sigma*1000),
# so model receives (1000 - 500) / 1000 = 0.5 = (1 - sigma). The transformer multiplies by t_scale=1000 internally.
t_input = torch.tensor([1.0 - SIGMA], dtype=torch.float32)
print(f"  Timestep input: {t_input.item():.6f} (= 1 - sigma; transformer multiplies by t_scale=1000 internally)")

# ── Hook every relevant submodule ──
captures = {}

def make_hook(name: str):
    def _h(module, _in, out):
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, list):
            # final_layer returns list when called via the transformer; capture pre-list
            return
        captures[name] = out.detach().clone()
    return _h

hooks = []
hooks.append(xfm.t_embedder.register_forward_hook(make_hook("t_embedder")))
hooks.append(xfm.cap_embedder.register_forward_hook(make_hook("cap_embedder")))
# Z-Image stores x_embedder under all_x_embedder["{patch}-{f_patch}"]
x_emb_key = list(xfm.all_x_embedder.keys())[0]
hooks.append(xfm.all_x_embedder[x_emb_key].register_forward_hook(make_hook("x_embedder")))

for i, blk in enumerate(xfm.context_refiner):
    hooks.append(blk.register_forward_hook(make_hook(f"context_refiner.{i}")))
for i, blk in enumerate(xfm.noise_refiner):
    hooks.append(blk.register_forward_hook(make_hook(f"noise_refiner.{i}")))
for i, blk in enumerate(xfm.layers):
    hooks.append(blk.register_forward_hook(make_hook(f"layers.{i}")))

final_key = list(xfm.all_final_layer.keys())[0]
hooks.append(xfm.all_final_layer[final_key].register_forward_hook(make_hook("final_layer")))

# ── Forward pass ──
print("Running forward pass ...")
with torch.no_grad():
    output = xfm(x_input, t_input, cap_input, return_dict=False, patch_size=2, f_patch_size=1)[0]
    # output is a list of [C, F, H, W] tensors per batch item
    output_tensor = output[0].unsqueeze(0)  # → [1, 16, 1, 32, 32]
    output_tensor = output_tensor.squeeze(2)  # → [1, 16, 32, 32]

for h in hooks:
    h.remove()

print(f"  Forward done. Output shape: {tuple(output_tensor.shape)}, "
      f"mean={output_tensor.mean().item():.6f}, std={output_tensor.std().item():.6f}")

# ── Save outputs ──
save(output_tensor, f"{OUT_DIR}/output_velocity.bin")

index = {
    "config": {
        "dim": xfm.dim,
        "n_heads": xfm.n_heads,
        "axes_dims": xfm.axes_dims,
        "rope_theta": xfm.rope_theta,
        "t_scale": xfm.t_scale,
        "in_channels": xfm.in_channels,
        "n_layers": len(xfm.layers),
        "n_refiner_layers": len(xfm.noise_refiner),
    },
    "inputs": {
        "latent": stats("latent", latent_bchw),
        "caption": stats("caption", caption_bld),
        "sigma": SIGMA,
        "t_input": float(t_input.item()),
    },
    "output_velocity": stats("output_velocity", output_tensor),
    "layers": [],
}

print(f"Saving {len(captures)} layer captures ...")
for name, t in captures.items():
    safe = name.replace(".", "_")
    save(t, f"{OUT_DIR}/layers/{safe}.bin")
    s = stats(name, t)
    s["file"] = f"layers/{safe}.bin"
    index["layers"].append(s)
    print(f"  {name:<28} shape={list(t.shape)} mean={s['mean']:+.6f} std={s['std']:.6f}")

with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"\nWrote {len(captures)} layer dumps + index.json to {OUT_DIR}")
