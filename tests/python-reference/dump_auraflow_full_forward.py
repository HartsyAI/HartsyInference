"""
Dumps an AuraFlow-v0.3 transformer forward pass with per-block image+text outputs
for a deterministic synthetic input. Source of truth for the C# layer-by-layer diff
(mirrors the SD3.5 + Z-Image patterns from PHASE_3_DEVIATIONS #28).

The flagship public weight is `fal/AuraFlow-v0.3` — single-file `aura_flow_0.3.safetensors`
(~16.5 GB FP16). Loaded via `from_single_file` → `convert_auraflow_transformer_checkpoint_to_diffusers`
internally, which renames `double_layers.{i}.modX/modC/attn.w[12][qkvo]/mlpX/mlpC` etc. to
diffusers naming.

Output: tests/python-reference/auraflow_reference_tensors/full_forward/
  inputs/latent.bin             [1, 4, 64, 64]    F32 (seed 42; 512x512 image @ patch=2)
  inputs/context_pre.bin        [1, 256, 2048]    F32 (seed 42; pre-context_embedder)
  inputs/timestep.bin           [1]               F32
  layers/<safe_name>.bin        per-layer F32 dumps (4 joint + 32 single + final)
  output_velocity.bin           transformer output [1, 4, 64, 64] F32
  index.json                    stats per layer

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_auraflow_full_forward.py
"""
import os
import json
import torch
from safetensors.torch import load_file
from diffusers.models.transformers.auraflow_transformer_2d import AuraFlowTransformer2DModel

CKPT_PATH = "/home/kalebbroo/Desktop/Projects/SharpInference/Models/Stable-Diffusion/AuraFlow/aura_flow_0.3.safetensors"
OUT_DIR = "tests/python-reference/auraflow_reference_tensors/full_forward"

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


# ── 1. Load + (single_file converter or direct constructor) ──
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(
        f"AuraFlow checkpoint not found: {CKPT_PATH}\n"
        f"Download from https://huggingface.co/fal/AuraFlow-v0.3/blob/main/aura_flow_0.3.safetensors"
    )

print(f"Loading single-file safetensors from {CKPT_PATH}...")
# Use from_single_file which calls convert_auraflow_transformer_checkpoint_to_diffusers internally.
# We avoid the gated repo by passing config kwargs explicitly via `config`.
xfm = AuraFlowTransformer2DModel.from_single_file(
    CKPT_PATH,
    torch_dtype=torch.float32,
    config={
        "sample_size": 64,
        "patch_size": 2,
        "in_channels": 4,
        "num_mmdit_layers": 4,
        "num_single_dit_layers": 32,
        "attention_head_dim": 256,
        "num_attention_heads": 12,
        "joint_attention_dim": 2048,
        "caption_projection_dim": 3072,
        "out_channels": 4,
        "pos_embed_max_size": 1024,
    },
).eval()
print(f"  Loaded. inner_dim={xfm.inner_dim}, num_mmdit={len(xfm.joint_transformer_blocks)}, "
      f"num_single={len(xfm.single_transformer_blocks)}")

# ── 2. Deterministic synthetic inputs ──
torch.manual_seed(42)
B, C, H, W = 1, 4, 64, 64        # 512x512 image latent (after 8x VAE downsample)
SEQ_T = 256                       # AuraFlow Pile-T5-XL max length
JOINT_DIM = 2048                  # Pile-T5-XL output dim

latent = torch.randn(B, C, H, W, dtype=torch.float32)
context_pre = torch.randn(B, SEQ_T, JOINT_DIM, dtype=torch.float32)
timestep = torch.tensor([0.5], dtype=torch.float32)  # mid-schedule

save(latent, f"{OUT_DIR}/inputs/latent.bin")
save(context_pre, f"{OUT_DIR}/inputs/context_pre.bin")
save(timestep, f"{OUT_DIR}/inputs/timestep.bin")
print(f"Saved inputs: latent {tuple(latent.shape)}, context_pre {tuple(context_pre.shape)}, t={float(timestep)}")

# ── 3. Hooks ──
layer_data = {}

def patch_hook(m, i, o):
    layer_data["pos_embed"] = o.detach().clone()

xfm.pos_embed.register_forward_hook(patch_hook)

for i, block in enumerate(xfm.joint_transformer_blocks):
    def make(idx):
        def h(m, i, o):
            # Returns (encoder_hidden_states, hidden_states) per AuraFlow line 275
            if isinstance(o, tuple) and len(o) == 2:
                txt, img = o
                layer_data[f"joint_block_{idx}_text"] = txt.detach().clone()
                layer_data[f"joint_block_{idx}_image"] = img.detach().clone()
        return h
    block.register_forward_hook(make(i))

for i, block in enumerate(xfm.single_transformer_blocks):
    def make(idx):
        def h(m, i, o):
            layer_data[f"single_block_{idx}"] = o.detach().clone()
        return h
    block.register_forward_hook(make(i))

xfm.norm_out.register_forward_hook(lambda m, i, o: layer_data.update({"norm_out": o.detach().clone()}))
xfm.proj_out.register_forward_hook(lambda m, i, o: layer_data.update({"proj_out": o.detach().clone()}))

# ── 4. Forward ──
print("\nRunning forward pass ...")
with torch.no_grad():
    out = xfm(
        hidden_states=latent,
        timestep=timestep,
        encoder_hidden_states=context_pre,
    ).sample

save(out, f"{OUT_DIR}/output_velocity.bin")
print(f"Output velocity shape={tuple(out.shape)} mean={float(out.mean()):.6f} std={float(out.std()):.6f}")

# ── 5. Save per-layer ──
print("\nSaving per-layer outputs ...")
index = []
for name, t in layer_data.items():
    fname = f"{name}.bin"
    save(t, f"{OUT_DIR}/layers/{fname}")
    index.append({"name": name, "file": f"layers/{fname}", **stats(name, t)})

index.append({"name": "input_latent", "file": "inputs/latent.bin", **stats("input_latent", latent)})
index.append({"name": "input_context_pre", "file": "inputs/context_pre.bin", **stats("input_context_pre", context_pre)})
index.append({"name": "output_velocity", "file": "output_velocity.bin", **stats("output_velocity", out)})

with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"\nLayer progression:")
for entry in index:
    n = entry["name"]
    if n.startswith("joint_block_") or n.startswith("single_block_") or n in ("pos_embed", "norm_out", "proj_out", "output_velocity"):
        print(f"  {n:32s} shape={entry['shape']}  mean={entry['mean']:+.4e}  std={entry['std']:.4e}  abs_mean={entry['abs_mean']:.4e}")

print(f"\nDone. {len(index)} entries saved to {OUT_DIR}/")
