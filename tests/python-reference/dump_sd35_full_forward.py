"""
Dumps an SD3.5 Medium MMDiT-X forward pass with per-block image+context outputs
for a deterministic synthetic input. The output is the source of truth for the
C# layer-by-layer diff (mirrors how Z-Image was debugged in deviation #28).

We construct SD3Transformer2DModel directly with config kwargs (no HF download
required) and convert the Stability single-file weights using diffusers' own
internal converter — so the SAME safetensors flow into both the Python reference
AND the C# port, eliminating weight-conversion drift.

Output: tests/python-reference/sd35_reference_tensors/full_forward/
  inputs/latent.bin             [1, 16, 64, 64]   F32 (seed 42)
  inputs/context_pre.bin        [1, 154, 4096]    F32 (seed 42, pre context_embedder)
  inputs/context_projected.bin  [1, 154, 1536]    F32
  inputs/pooled.bin             [1, 2048]         F32 (seed 42)
  inputs/timestep.bin           [1]               F32 (sigma=0.5 → t=500)
  layers/<safe_name>.bin        F32 per-layer dumps (patch_embed, time_text_embed, block_<i>_image, block_<i>_context, norm_out, proj_out)
  output_velocity.bin           F32 final output [1, 16, 64, 64]
  index.json
"""
import os
import json
import torch
from safetensors.torch import load_file
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.loaders.single_file_utils import convert_sd3_transformer_checkpoint_to_diffusers

CKPT_PATH = "/home/kalebbroo/Desktop/Projects/SharpInference/Models/Stable-Diffusion/SD3/sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"
OUT_DIR = "tests/python-reference/sd35_reference_tensors/full_forward"

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


# ── 1. Load raw safetensors and let diffusers convert to its naming ──
print(f"Loading raw safetensors from {CKPT_PATH} ...")
raw = load_file(CKPT_PATH, device="cpu")

# Filter out the t5/clip/vae sub-checkpoints — keep only model.diffusion_model.* keys
# (the converter expects only transformer keys; T5 and VAE are loaded separately).
print(f"Filtering: {len(raw)} total keys")
trans_keys = [k for k in raw if k.startswith("model.diffusion_model.")]
print(f"  Transformer keys: {len(trans_keys)}")
trans_only = {k: raw[k] for k in trans_keys}

# Cast all to float32 for reference (we want fp32 reference, not fp16)
trans_f32 = {k: v.float() for k, v in trans_only.items()}
del raw, trans_only

# Run diffusers' internal converter (does the swap_scale_shift for final_layer + last block norm1_context)
print("Running diffusers convert_sd3_transformer_checkpoint_to_diffusers ...")
converted = convert_sd3_transformer_checkpoint_to_diffusers(trans_f32)
print(f"  Converted to {len(converted)} diffusers-format keys")

# ── 2. Construct SD3Transformer2DModel with SD3.5 Medium config (no HF download) ──
# These values come from inspecting the stabilityai/stable-diffusion-3.5-medium
# transformer/config.json + the loaded weight shapes.
print("\nConstructing SD3Transformer2DModel(SD3.5 Medium) ...")
xfm = SD3Transformer2DModel(
    sample_size=64,                                   # latent grid (= 512px / VAE 8x = 64)
    patch_size=2,
    in_channels=16,
    num_layers=24,                                    # SD3.5 Medium = 24 joint blocks
    attention_head_dim=64,                            # head_dim, not head_count
    num_attention_heads=24,
    joint_attention_dim=4096,                          # T5 + padded CLIP
    caption_projection_dim=1536,                       # = hidden size
    pooled_projection_dim=2048,                        # CLIP-L 768 + CLIP-G 1280
    out_channels=16,
    pos_embed_max_size=384,                            # SD3.5 Medium — verified from weight shape
    dual_attention_layers=tuple(range(13)),            # 0..12 — SD3.5 Medium MMDiT-X
    qk_norm="rms_norm",                                # SD3.5 = QK-norm
)
xfm = xfm.eval()
print(f"  hidden={xfm.config.caption_projection_dim}, layers={xfm.config.num_layers}, dual={xfm.config.dual_attention_layers}")

# Load converted weights
print("Loading converted weights into model ...")
missing, unexpected = xfm.load_state_dict(converted, strict=False)
print(f"  Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")
if len(missing) > 0:
    print(f"  Missing sample: {missing[:5]}")
if len(unexpected) > 0:
    print(f"  Unexpected sample: {unexpected[:5]}")

# ── 3. Deterministic synthetic inputs ──
torch.manual_seed(42)
B, C, H, W = 1, 16, 64, 64       # 512x512 image latent (after 8x VAE downsample → 64x64 latent)
SEQ = 154                         # CLIP-L+G+T5 tokens
JOINT_DIM = 4096                  # joint_attention_dim
POOLED_DIM = 2048                 # pooled CLIP-L (768) + CLIP-G (1280)
TIMESTEP_VALUE = 500.0            # sigma=0.5 * 1000

latent = torch.randn(B, C, H, W, dtype=torch.float32)
context_pre = torch.randn(B, SEQ, JOINT_DIM, dtype=torch.float32)
pooled = torch.randn(B, POOLED_DIM, dtype=torch.float32)
timestep = torch.tensor([TIMESTEP_VALUE], dtype=torch.float32)

with torch.no_grad():
    projected_context = xfm.context_embedder(context_pre)

save(latent, f"{OUT_DIR}/inputs/latent.bin")
save(context_pre, f"{OUT_DIR}/inputs/context_pre.bin")
save(projected_context, f"{OUT_DIR}/inputs/context_projected.bin")
save(pooled, f"{OUT_DIR}/inputs/pooled.bin")
save(timestep, f"{OUT_DIR}/inputs/timestep.bin")
print(f"\nSaved inputs:")
print(f"  latent              {tuple(latent.shape)}")
print(f"  context_pre         {tuple(context_pre.shape)} → context_projected {tuple(projected_context.shape)}")
print(f"  pooled              {tuple(pooled.shape)}")
print(f"  timestep            {float(timestep):.1f}")

# ── 4. Hooks ──
layer_data = {}

def patch_hook(m, i, o):
    layer_data["patch_embed"] = o.detach().clone()

def time_hook(m, i, o):
    layer_data["time_text_embed"] = o.detach().clone()

xfm.pos_embed.register_forward_hook(patch_hook)
xfm.time_text_embed.register_forward_hook(time_hook)

for i, block in enumerate(xfm.transformer_blocks):
    def make_hook(idx):
        def fn(module, inp, out):
            # JointTransformerBlock returns (encoder_hidden_states, hidden_states)
            # Last block (context_pre_only) returns (None, hidden_states)
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


# ── 5. Forward ──
print("\nRunning forward pass ...")
with torch.no_grad():
    out = xfm(
        hidden_states=latent,
        timestep=timestep,
        encoder_hidden_states=context_pre,           # raw 4096-dim — transformer.context_embedder projects internally
        pooled_projections=pooled,
    ).sample

save(out, f"{OUT_DIR}/output_velocity.bin")
print(f"Output velocity shape={tuple(out.shape)} mean={float(out.mean()):.6f} std={float(out.std()):.6f}")

# ── 6. Save per-layer ──
print("\nSaving per-layer outputs ...")
index = []
for name, t in layer_data.items():
    fname = f"{name}.bin"
    save(t, f"{OUT_DIR}/layers/{fname}")
    s = stats(name, t)
    index.append({"name": name, "file": f"layers/{fname}", **s})

# Also save inputs and final output in index for cross-reference
index.append({"name": "input_latent", "file": "inputs/latent.bin", **stats("input_latent", latent)})
index.append({"name": "input_context_pre", "file": "inputs/context_pre.bin", **stats("input_context_pre", context_pre)})
index.append({"name": "input_context_projected", "file": "inputs/context_projected.bin", **stats("input_context_projected", projected_context)})
index.append({"name": "input_pooled", "file": "inputs/pooled.bin", **stats("input_pooled", pooled)})
index.append({"name": "output_velocity", "file": "output_velocity.bin", **stats("output_velocity", out)})

with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump(index, f, indent=2)

# ── 7. Print summary ──
print("\nLayer progression (look for monotone shift in mean/std after a specific block):")
for entry in index:
    name = entry["name"]
    if name in ("patch_embed", "time_text_embed", "norm_out", "proj_out", "output_velocity") or name.startswith("block_"):
        print(f"  {name:35s} shape={entry['shape']}  mean={entry['mean']:+.4e}  std={entry['std']:.4e}  abs_mean={entry['abs_mean']:.4e}")

print(f"\nDone. {len(index)} entries saved to {OUT_DIR}/")
