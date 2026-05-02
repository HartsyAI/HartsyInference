"""
Dumps an SD3.5 Medium MMDiT-X forward pass with per-block image+context outputs
for a deterministic synthetic input. The output is the source of truth for the
C# layer-by-layer diff (mirrors how Z-Image was debugged in deviation #28).

We use `from_single_file` so the SAME safetensors checkpoint flows through both
the Python reference AND the C# port, eliminating weight-conversion drift.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_sd35_full_forward.py

Output: tests/python-reference/sd35_reference_tensors/full_forward/
  inputs/latent.bin             [1, 16, 64, 64]   F32 (seed 42)
  inputs/context.bin            [1, 154, 4096]    F32 (seed 42, pre-projection)
  inputs/pooled.bin             [1, 2048]         F32 (seed 42)
  inputs/timestep.bin           [1]               F32 (sigma=0.5 → t=500)
  layers/<safe_name>.bin        per-layer F32 (patch_embed, time_embed, block_<i>_image, block_<i>_context, norm_out)
  output_velocity.bin           transformer output [1, 16, 64, 64] F32
  index.json                    stats per layer + config
"""
import os
import json
import torch
from safetensors.torch import load_file
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

CKPT_PATH = os.path.expanduser(
    "/home/kalebbroo/Desktop/Projects/SharpInference/Models/Stable-Diffusion/SD3/sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"
)
OUT_DIR = "tests/python-reference/sd35_reference_tensors/full_forward"
SIGMA = 0.5  # mid-schedule
TIMESTEP_VALUE = SIGMA * 1000.0  # SD3 timestep is sigma*1000

os.makedirs(f"{OUT_DIR}/inputs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/layers", exist_ok=True)


def save(t, path):
    t.float().detach().cpu().contiguous().numpy().tofile(path)


def stats(name, t):
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


# ── 1. Load the Stability single-file safetensors → diffusers SD3Transformer2DModel ──
print(f"Loading single-file safetensors from {CKPT_PATH}...")

# Use diffusers' from_single_file conversion (this handles Stability LDM key naming
# and the swap_scale_shift for the final adaLN).
print("Loading via SD3Transformer2DModel.from_single_file ...")
xfm = SD3Transformer2DModel.from_single_file(CKPT_PATH, torch_dtype=torch.float32)
xfm = xfm.eval()
config = xfm.config
print(
    f"  Loaded. depth={config.num_layers}, "
    f"hidden={config.caption_projection_dim}, heads={config.num_attention_heads}, "
    f"dual={getattr(config, 'dual_attention_layers', None)}, "
    f"qk_norm={getattr(config, 'qk_norm', None)}, "
    f"pos_embed_max_size={config.pos_embed_max_size}"
)

# ── 2. Deterministic synthetic inputs (no real text encoder; we feed random
#       projected context + random pooled. The transformer doesn't care that the
#       caption is "real" — it just needs the right shapes and finite values.) ──
torch.manual_seed(42)
B, C, H, W = 1, 16, 64, 64  # 512×512 latent → 64×64 latent grid → 32×32 patch grid (patch_size=2)
SEQ = 154  # CLIP-L+G+T5 padded to 154 tokens
JOINT_DIM = 4096  # joint_attention_dim
POOLED_DIM = 2048  # pooled CLIP-L (768) + CLIP-G (1280)

latent = torch.randn(B, C, H, W, dtype=torch.float32)
# context is the PRE-projected combined CLIP+T5 (before context_embedder Linear)
context_pre = torch.randn(B, SEQ, JOINT_DIM, dtype=torch.float32)
pooled = torch.randn(B, POOLED_DIM, dtype=torch.float32)
timestep = torch.tensor([TIMESTEP_VALUE], dtype=torch.float32)

# Project context through context_embedder (this is what the C# pipeline does)
with torch.no_grad():
    projected_context = xfm.context_embedder(context_pre)

save(latent, f"{OUT_DIR}/inputs/latent.bin")
save(context_pre, f"{OUT_DIR}/inputs/context_pre.bin")
save(projected_context, f"{OUT_DIR}/inputs/context_projected.bin")
save(pooled, f"{OUT_DIR}/inputs/pooled.bin")
save(timestep, f"{OUT_DIR}/inputs/timestep.bin")
print(f"Saved inputs: latent {tuple(latent.shape)}, context_pre {tuple(context_pre.shape)}, "
      f"context_projected {tuple(projected_context.shape)}, pooled {tuple(pooled.shape)}, "
      f"timestep={float(timestep):.1f}")

# ── 3. Hooks ──
layer_data = {}

def patch_hook(module, inp, out):
    layer_data["patch_embed"] = out.detach().clone()

def time_hook(module, inp, out):
    layer_data["time_text_embed"] = out.detach().clone()

xfm.pos_embed.register_forward_hook(patch_hook)
xfm.time_text_embed.register_forward_hook(time_hook)

for i, block in enumerate(xfm.transformer_blocks):
    def make_hook(idx):
        def fn(module, inp, out):
            # JointTransformerBlock returns (encoder_hidden_states, hidden_states) in newer diffusers,
            # OR (hidden_states,) for the last (pre_only) block.
            if isinstance(out, tuple):
                if len(out) == 2:
                    enc, img = out
                    if enc is not None:
                        layer_data[f"block_{idx}_context"] = enc.detach().clone()
                    layer_data[f"block_{idx}_image"] = img.detach().clone()
                else:
                    layer_data[f"block_{idx}_image"] = out[0].detach().clone()
            else:
                layer_data[f"block_{idx}_image"] = out.detach().clone()
        return fn
    block.register_forward_hook(make_hook(i))

xfm.norm_out.register_forward_hook(lambda m, i, o: layer_data.update({"norm_out": o.detach().clone()}))
xfm.proj_out.register_forward_hook(lambda m, i, o: layer_data.update({"proj_out": o.detach().clone()}))


# ── 4. Forward ──
print("\nRunning forward pass ...")
with torch.no_grad():
    out = xfm(
        hidden_states=latent,
        timestep=timestep,
        encoder_hidden_states=projected_context,
        pooled_projections=pooled,
    ).sample

save(out, f"{OUT_DIR}/output_velocity.bin")
print(f"Output: shape {tuple(out.shape)}, mean {float(out.mean()):.6f}, std {float(out.std()):.6f}")

# ── 5. Save per-layer ──
print("\nSaving per-layer outputs ...")
index = []
for name, t in layer_data.items():
    fname = f"{name}.bin"
    save(t, f"{OUT_DIR}/layers/{fname}")
    s = stats(name, t)
    index.append({"name": name, "file": f"layers/{fname}", **s})

# Also save inputs in index
index.append({"name": "input_latent", "file": "inputs/latent.bin", **stats("input_latent", latent)})
index.append({"name": "input_context_projected", "file": "inputs/context_projected.bin", **stats("input_context_projected", projected_context)})
index.append({"name": "input_pooled", "file": "inputs/pooled.bin", **stats("input_pooled", pooled)})
index.append({"name": "output_velocity", "file": "output_velocity.bin", **stats("output_velocity", out)})

with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump(index, f, indent=2)

# ── 6. Print summary ──
print("\nLayer progression:")
for entry in index:
    name = entry["name"]
    if name.startswith("block_") or name in ("patch_embed", "time_text_embed", "norm_out", "proj_out", "output_velocity"):
        print(f"  {name:35s} shape={entry['shape']}  mean={entry['mean']:+.4e}  std={entry['std']:.4e}  abs_mean={entry['abs_mean']:.4e}")

print(f"\nDone. {len(index)} entries saved to {OUT_DIR}")
