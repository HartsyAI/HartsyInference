"""
Dumps a Chroma transformer forward pass with per-block image+text outputs for a
deterministic synthetic input. Source of truth for the C# layer-by-layer diff.

Chroma's distilled-guidance approximator produces a 344-row modulation table; this
script also dumps the table for cross-reference (the C# transformer should produce
the SAME table from the same inputs since it's a pure deterministic MLP).

Output: tests/python-reference/chroma_reference_tensors/full_forward/
  inputs/latent.bin             [1, 16, 64, 64] F32 (seed 42)
  inputs/context_pre.bin        [1, 256, 4096]   F32 (T5-XXL synthetic)
  inputs/timestep.bin           [1]              F32
  layers/mod_table.bin          [1, 344, 3072]   F32 (approximator output)
  layers/<safe_name>.bin        per-block dumps (19 doubles + 38 singles)
  output_velocity.bin           [1, 16, 64, 64]  F32

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_chroma_full_forward.py
"""
import os
import json
import torch
from diffusers.models.transformers.transformer_chroma import ChromaTransformer2DModel

CKPT_PATH = "/home/kalebbroo/Desktop/Projects/SharpInference/Models/Stable-Diffusion/Chroma/Chroma1-HD-fp8mixed-final.safetensors"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "tests/python-reference/chroma_reference_tensors/full_forward")

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


if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Chroma checkpoint not found: {CKPT_PATH}")

print(f"Loading Chroma single-file from {CKPT_PATH}...")
# Memory-saving path: Chroma is 8.9B params; F32 alone is ~36GB which OOMs a 31GB machine. We
# construct on meta device, write params into BF16 storage one at a time, then sync any non-param
# state (RoPE bases, normalization stats) by initializing those buffers on CPU explicitly. This
# uses ~18GB peak.
import gc
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers.loaders.single_file_utils import convert_chroma_transformer_checkpoint_to_diffusers
from safetensors.torch import load_file as load_safetensors

with init_empty_weights():
    xfm = ChromaTransformer2DModel()

print(f"Loading + converting weights (BF16 streamed via set_module_tensor_to_device)...")
raw_state = load_safetensors(CKPT_PATH)
converted = convert_chroma_transformer_checkpoint_to_diffusers(raw_state)
del raw_state
gc.collect()

# Place each tensor onto CPU/BF16 storage via accelerate — this skips the F32 intermediate that
# `.to()` + `load_state_dict(assign=True)` produces internally, AND it correctly initializes any
# non-persistent buffers (RoPE freqs, etc.) by NOT touching them (they stay on meta until forward,
# which lazy-initializes them on the right device).
loaded_keys = set()
for k, v in converted.items():
    set_module_tensor_to_device(xfm, k, device='cpu', dtype=torch.bfloat16, value=v)
    loaded_keys.add(k)
del converted
gc.collect()

# Materialize any remaining meta-device parameters (none expected for a clean state_dict, but the
# init_empty path can leave non-persistent buffers on meta).
remaining_meta = [(n, p) for n, p in xfm.named_parameters() if p.device.type == "meta"]
remaining_meta += [(n, b) for n, b in xfm.named_buffers() if b.device.type == "meta"]
if remaining_meta:
    print(f"  WARNING: {len(remaining_meta)} tensors still on meta after load:")
    for n, _ in remaining_meta[:5]:
        print(f"    {n}")

xfm.eval()
print(f"  Model ready in BF16 (~18GB).")

# ── Synthetic inputs ──
# Memory-saving: 32×32 latent (= 256 image tokens after 2×2 patchify) instead of 64×64.
# Both sides run at this reduced size; the math is identical to full resolution.
torch.manual_seed(42)
B, C, H, W = 1, 16, 32, 32
SEQ_T = 64
JOINT_DIM = 4096

latent = torch.randn(B, C, H, W, dtype=torch.float32)
context_pre = torch.randn(B, SEQ_T, JOINT_DIM, dtype=torch.float32)
timestep = torch.tensor([0.5], dtype=torch.float32)
text_lens = torch.tensor([SEQ_T - 4], dtype=torch.long)  # 4 padded slots at the end
attention_mask = (torch.arange(SEQ_T)[None, :] < text_lens[:, None]).long()

save(latent, f"{OUT_DIR}/inputs/latent.bin")
save(context_pre, f"{OUT_DIR}/inputs/context_pre.bin")
save(timestep, f"{OUT_DIR}/inputs/timestep.bin")
attention_mask.to(torch.int64).detach().cpu().contiguous().numpy().tofile(f"{OUT_DIR}/inputs/attention_mask.bin")

# Extend the prompt attention mask to cover image tokens (diffusers `_prepare_attention_mask`).
# The transformer sees a joint [text|image] sequence; image tokens are always unmasked.
img_seq_len = (H // 2) * (W // 2)
joint_attention_mask = torch.cat([attention_mask,
                                  torch.ones(B, img_seq_len, dtype=torch.long)], dim=1)

# ── Hooks ──
layer_data = {}

def hook_module(name):
    def hook(m, i, o):
        if isinstance(o, tuple):
            for j, t in enumerate(o):
                if t is not None:
                    layer_data[f"{name}_{j}"] = t.detach().clone()
        else:
            layer_data[name] = o.detach().clone()
    return hook

xfm.distilled_guidance_layer.register_forward_hook(hook_module("mod_table"))

for i, block in enumerate(xfm.transformer_blocks):
    block.register_forward_hook(hook_module(f"double_{i}"))
for i, block in enumerate(xfm.single_transformer_blocks):
    block.register_forward_hook(hook_module(f"single_{i}"))
xfm.norm_out.register_forward_hook(hook_module("norm_out"))
xfm.proj_out.register_forward_hook(hook_module("proj_out"))

# Pack latent: [B, C, H, W] → [B, S, C*4] via 2x2 patchify
def pack(x):
    return torch.einsum("bchpwq->bhwcpq", x.reshape(B, C, H//2, 2, W//2, 2)).reshape(B, (H//2)*(W//2), C*4)

packed_latent = pack(latent)

# Position IDs (Flux-style: 3 axes [time=0, y, x] for image, [0, 0, 0] for text)
img_ids = torch.zeros(B, packed_latent.shape[1], 3, dtype=torch.float32)
img_ids[:, :, 1] = torch.arange(H//2).repeat_interleave(W//2)[None, :]
img_ids[:, :, 2] = torch.arange(W//2).tile(H//2)[None, :]
txt_ids = torch.zeros(B, SEQ_T, 3, dtype=torch.float32)

print("Running forward (BF16) ...")
with torch.no_grad():
    out = xfm(
        hidden_states=packed_latent.to(torch.bfloat16),
        encoder_hidden_states=context_pre.to(torch.bfloat16),
        timestep=timestep.to(torch.bfloat16),
        img_ids=img_ids.squeeze(0),                              # 2D per diffusers expectation
        txt_ids=txt_ids.squeeze(0),                              # 2D per diffusers expectation
        attention_mask=joint_attention_mask.to(torch.bfloat16),  # Joint [text|image] mask, SDPA wants float
    ).sample

save(out, f"{OUT_DIR}/output_velocity.bin")
print(f"Output: shape={tuple(out.shape)} mean={float(out.mean()):.4e} std={float(out.std()):.4e}")

print("\nSaving per-layer outputs ...")
index = []
for name, t in layer_data.items():
    fname = f"{name.replace('.', '_')}.bin"
    save(t, f"{OUT_DIR}/layers/{fname}")
    index.append({"name": name, "file": f"layers/{fname}", **stats(name, t)})
index.append({"name": "input_latent", "file": "inputs/latent.bin", **stats("input_latent", latent)})
index.append({"name": "input_context_pre", "file": "inputs/context_pre.bin", **stats("input_context_pre", context_pre)})
index.append({"name": "output_velocity", "file": "output_velocity.bin", **stats("output_velocity", out)})

with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"\nDone. {len(index)} entries saved to {OUT_DIR}/")
