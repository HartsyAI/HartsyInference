"""
Dumps an ERNIE-Image transformer forward pass with per-block outputs for a
deterministic synthetic input. Source of truth for the C# layer-by-layer diff.

Output: tests/python-reference/ernie_image_reference_tensors/full_forward/
  inputs/latent.bin            [1, 128, 32, 32]  F32 (Flux2-style 128-ch latent)
  inputs/text_embeds.bin       [1, T, 3072]      F32 (synthetic; per-batch text_lens carried separately)
  inputs/text_lens.bin         [1]               int32
  inputs/timestep.bin          [1]               F32
  layers/<safe_name>.bin       per-block dumps (36 layers + final)
  output_velocity.bin          [1, 128, 32, 32]  F32

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_ernie_image_full_forward.py [model_dir]

`model_dir` defaults to the diffusers folder layout at `Models/Stable-Diffusion/ErnieImage/v1/`.
"""
import os
import sys
import json
import torch
from diffusers.models.transformers.transformer_ernie_image import ErnieImageTransformer2DModel

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/kalebbroo/Desktop/Projects/HartsyInference/Models/Stable-Diffusion/ErnieImage/v1"
OUT_DIR = "tests/python-reference/ernie_image_reference_tensors/full_forward"

os.makedirs(f"{OUT_DIR}/inputs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/layers", exist_ok=True)


def save(t, path):
    t.float().detach().cpu().contiguous().numpy().tofile(path)


def stats(name, t):
    f = t.float().flatten()
    return {
        "name": name, "shape": list(t.shape), "dtype": str(t.dtype),
        "mean": float(f.mean()), "std": float(f.std()),
        "min": float(f.min()), "max": float(f.max()),
        "abs_mean": float(f.abs().mean()), "first_8": [float(x) for x in f[:8]],
    }


if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"ERNIE-Image model directory not found: {MODEL_DIR}\n"
        f"Download from https://huggingface.co/baidu/ERNIE-Image (transformer/ subfolder at minimum)"
    )

print(f"Loading from {MODEL_DIR}/transformer ...")
xfm = ErnieImageTransformer2DModel.from_pretrained(
    MODEL_DIR, subfolder="transformer", torch_dtype=torch.float32
).eval()
print(f"  Loaded. hidden={xfm.config.hidden_size}, layers={xfm.config.num_layers}")

# ── Synthetic inputs ──
torch.manual_seed(42)
B = 1
H = W = 32       # 256x256 image (after VAE 8x downsample)
T = 64           # short text sequence
TEXT_LEN = 50
HIDDEN_TEXT = xfm.config.text_in_dim

latent = torch.randn(B, 128, H, W, dtype=torch.float32)
text_embeds = torch.randn(B, T, HIDDEN_TEXT, dtype=torch.float32)
text_lens = torch.tensor([TEXT_LEN], dtype=torch.long)
timestep = torch.tensor([500.0], dtype=torch.float32)

save(latent, f"{OUT_DIR}/inputs/latent.bin")
save(text_embeds, f"{OUT_DIR}/inputs/text_embeds.bin")
save(text_lens.to(torch.int32), f"{OUT_DIR}/inputs/text_lens.bin")
save(timestep, f"{OUT_DIR}/inputs/timestep.bin")
print(f"Saved inputs: latent {tuple(latent.shape)}, text {tuple(text_embeds.shape)}, t={float(timestep)}")

# ── Hooks ──
layer_data = {}

def hook(name):
    def fn(m, i, o):
        layer_data[name] = (o[0] if isinstance(o, tuple) else o).detach().clone()
    return fn

for i, block in enumerate(xfm.transformer_blocks):
    block.register_forward_hook(hook(f"block_{i}"))
xfm.norm_out.register_forward_hook(hook("norm_out"))

print("Running forward ...")
with torch.no_grad():
    out = xfm(
        hidden_states=latent,
        timestep=timestep,
        encoder_hidden_states=text_embeds,
        text_lengths=text_lens,
    ).sample

save(out, f"{OUT_DIR}/output_velocity.bin")
print(f"Output: shape={tuple(out.shape)} mean={float(out.mean()):.4e} std={float(out.std()):.4e}")

index = []
for name, t in layer_data.items():
    fname = f"{name.replace('.', '_')}.bin"
    save(t, f"{OUT_DIR}/layers/{fname}")
    index.append({"name": name, "file": f"layers/{fname}", **stats(name, t)})
index.append({"name": "output_velocity", "file": "output_velocity.bin", **stats("output_velocity", out)})

with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"\nDone. {len(index)} entries to {OUT_DIR}/")
