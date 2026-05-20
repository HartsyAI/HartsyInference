"""
Dumps a Qwen-Image VAE decode forward pass for layer-by-layer C# comparison.

The Qwen-Image VAE is a 3D causal autoencoder (WAN 2.1 family) that operates in 5D
[B, C, T, H, W] but for image mode runs with T=1. Anima uses this VAE for its decode path.

Tests:
  1. Per-channel latent rescaling: `latents = latents * std + mean` BEFORE VAE.decode
     (the diffusers pipeline does this, my current C# code only does scalar)
  2. Pixel-by-pixel agreement of the decoder output

Output: tests/python-reference/qwen_image_vae_reference/
  inputs/latent_raw.bin          [1, 16, 32, 32]    F32  pre-rescale (post-DiT, post-_unpack)
  inputs/latent_rescaled.bin     [1, 16, 1, 32, 32] F32  post-rescale (= latent * std + mean)
  inputs/latents_mean.bin        [16] F32  (vae.config.latents_mean)
  inputs/latents_std.bin         [16] F32  (vae.config.latents_std)
  output_image.bin               [1, 3, 256, 256]   F32  decoded RGB
  layers/<name>.bin              per-stage F32 dumps
  index.json                     stats per layer + config
"""
import os
import gc
import json
import torch
import numpy as np
from safetensors import safe_open
from diffusers import AutoencoderKLQwenImage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAE_PATH = "/home/kalebbroo/Desktop/Projects/SharpInference/Models/VAE/QwenImage/qwen_image_vae.safetensors"
OUT_DIR = os.path.join(REPO_ROOT, "tests/python-reference/qwen_image_vae_reference")

LATENT_H = 32
LATENT_W = 32
LATENT_CH = 16

os.makedirs(f"{OUT_DIR}/inputs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/layers", exist_ok=True)


def save(t: torch.Tensor, path: str):
    t.float().detach().cpu().contiguous().numpy().tofile(path)


def stats_dict(name: str, t: torch.Tensor) -> dict:
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


# ── 1. Key translator: raw Comfy/WAN naming → diffusers AutoencoderKLQwenImage naming ──
# Encoder uses `down_blocks`, decoder uses `up_blocks`, both with 4 groups of (resnets, optional upsampler/downsampler).
def translate_block_subkey(rest: str) -> str | None:
    """For a resnet stage, map `residual.X` → diffusers conv/norm naming. Pass-through for `resample` / `time_conv`."""
    sub_map = {
        "residual.0.gamma":   "norm1.gamma",
        "residual.2.weight":  "conv1.weight",
        "residual.2.bias":    "conv1.bias",
        "residual.3.gamma":   "norm2.gamma",
        "residual.6.weight":  "conv2.weight",
        "residual.6.bias":    "conv2.bias",
        "shortcut.weight":    "conv_shortcut.weight",
        "shortcut.bias":      "conv_shortcut.bias",
    }
    if rest in sub_map:
        return sub_map[rest]
    if rest.startswith("resample.") or rest.startswith("time_conv."):
        return rest
    return None


# Decoder structure: flat upsamples[0..14] → grouped up_blocks (3 resnets + optional upsampler).
# Stages 0..3: up_blocks.0 (3 res + 1 upsampler)    raw idx 0,1,2,3
# Stages 4..7: up_blocks.1 (3 res + 1 upsampler)    raw idx 4,5,6,7
# Stages 8..11: up_blocks.2 (3 res + 1 upsampler)   raw idx 8,9,10,11
# Stages 12..14: up_blocks.3 (3 res, no upsampler)  raw idx 12,13,14
def decoder_upsample_target(raw_idx: int) -> str:
    if raw_idx <= 3:
        block_idx, sub = 0, raw_idx
    elif raw_idx <= 7:
        block_idx, sub = 1, raw_idx - 4
    elif raw_idx <= 11:
        block_idx, sub = 2, raw_idx - 8
    else:
        block_idx, sub = 3, raw_idx - 12
    # The 4th sub of each group (sub=3) is the upsampler; others are resnets.
    if sub == 3 and block_idx <= 2:
        return f"decoder.up_blocks.{block_idx}.upsamplers.0"
    return f"decoder.up_blocks.{block_idx}.resnets.{sub}"


# Encoder structure: flat downsamples[0..10] → grouped down_blocks. Pattern mirrors decoder but the
# resample is INSIDE the down_block (not a separate "downsampler" group). Probed from the actual
# diffusers state_dict: encoder.down_blocks[0..3] mostly have direct flat structure (no resnets/
# upsamplers nesting). We let strict=False handle any encoder misses since we don't run encoder.
def encoder_downsample_target(raw_idx: int) -> str:
    # The diffusers encoder uses a flat down_blocks.X naming directly matching raw idx in many cases.
    return f"encoder.down_blocks.{raw_idx}"


def translate_qwen_vae(raw: dict) -> dict:
    """Translate raw Comfy safetensors keys → diffusers AutoencoderKLQwenImage state_dict keys."""
    out = {}
    for key, tensor in raw.items():
        # Top-level 1×1×1 conv: raw `conv1` → diffusers `quant_conv`, `conv2` → `post_quant_conv`.
        if key.startswith("conv1."):
            out["quant_conv." + key[len("conv1."):]] = tensor
            continue
        if key.startswith("conv2."):
            out["post_quant_conv." + key[len("conv2."):]] = tensor
            continue

        if key.startswith("decoder."):
            tail = key[len("decoder."):]
            if tail.startswith("conv1."):
                out["decoder.conv_in." + tail[len("conv1."):]] = tensor
                continue
            if tail.startswith("head.0.gamma"):
                out["decoder.norm_out.gamma"] = tensor
                continue
            if tail.startswith("head.2.weight"):
                out["decoder.conv_out.weight"] = tensor
                continue
            if tail.startswith("head.2.bias"):
                out["decoder.conv_out.bias"] = tensor
                continue
            if tail.startswith("middle."):
                # raw: decoder.middle.{0,2}.residual.* (resnets) or middle.1.* (attention)
                parts = tail.split(".")
                mid_idx = int(parts[1])
                rest = ".".join(parts[2:])
                if mid_idx in (0, 2):
                    # resnet → mid_block.resnets.{0 or 1}
                    resnet_idx = 0 if mid_idx == 0 else 1
                    sub = translate_block_subkey(rest)
                    if sub is not None:
                        out[f"decoder.mid_block.resnets.{resnet_idx}.{sub}"] = tensor
                        continue
                elif mid_idx == 1:
                    # attention → mid_block.attentions.0
                    out[f"decoder.mid_block.attentions.0.{rest}"] = tensor
                    continue
                print(f"WARNING: unmapped middle key: {key}")
                continue
            if tail.startswith("upsamples."):
                parts = tail.split(".")
                raw_idx = int(parts[1])
                rest = ".".join(parts[2:])
                target_prefix = decoder_upsample_target(raw_idx)
                sub = translate_block_subkey(rest)
                if sub is not None:
                    out[f"{target_prefix}.{sub}"] = tensor
                    continue
                print(f"WARNING: unmapped upsample key: {key}")
                continue
            print(f"WARNING: unmapped decoder key: {key}")
            continue

        if key.startswith("encoder."):
            tail = key[len("encoder."):]
            if tail.startswith("conv1."):
                out["encoder.conv_in." + tail[len("conv1."):]] = tensor
                continue
            if tail.startswith("head.0.gamma"):
                out["encoder.norm_out.gamma"] = tensor
                continue
            if tail.startswith("head.2.weight"):
                out["encoder.conv_out.weight"] = tensor
                continue
            if tail.startswith("head.2.bias"):
                out["encoder.conv_out.bias"] = tensor
                continue
            if tail.startswith("middle."):
                parts = tail.split(".")
                mid_idx = int(parts[1])
                rest = ".".join(parts[2:])
                if mid_idx in (0, 2):
                    resnet_idx = 0 if mid_idx == 0 else 1
                    sub = translate_block_subkey(rest)
                    if sub is not None:
                        out[f"encoder.mid_block.resnets.{resnet_idx}.{sub}"] = tensor
                        continue
                elif mid_idx == 1:
                    out[f"encoder.mid_block.attentions.0.{rest}"] = tensor
                    continue
            if tail.startswith("downsamples."):
                parts = tail.split(".")
                raw_idx = int(parts[1])
                rest = ".".join(parts[2:])
                target_prefix = encoder_downsample_target(raw_idx)
                # Encoder uses flat naming — try residual/resample mapping
                sub = translate_block_subkey(rest)
                if sub is not None:
                    out[f"{target_prefix}.{sub}"] = tensor
                    continue
                # Some encoder downsamples may be just `resample.1.*` directly
                out[f"{target_prefix}.{rest}"] = tensor
                continue
            print(f"WARNING: unmapped encoder key: {key}")
            continue

        print(f"WARNING: unmapped top-level key: {key}")
    return out


# ── 2. Load the Qwen-Image VAE checkpoint + translate ──
print(f"Loading Qwen-Image VAE: {VAE_PATH}")
raw_state = {}
with safe_open(VAE_PATH, framework="pt", device="cpu") as f:
    for key in f.keys():
        raw_state[key] = f.get_tensor(key).float()
print(f"  Loaded {len(raw_state)} raw tensors")

print("Translating raw → diffusers naming...")
diffusers_state = translate_qwen_vae(raw_state)
print(f"  Translated to {len(diffusers_state)} diffusers keys")
del raw_state
gc.collect()

# ── 3. Construct the diffusers Qwen-Image VAE and load weights ──
print("Constructing AutoencoderKLQwenImage with default config (F32)...")
vae = AutoencoderKLQwenImage().to(torch.float32).eval()

latents_mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32)
latents_std = torch.tensor(vae.config.latents_std, dtype=torch.float32)
print(f"  latents_mean: shape={tuple(latents_mean.shape)} first4={latents_mean[:4].tolist()}")
print(f"  latents_std:  shape={tuple(latents_std.shape)} first4={latents_std[:4].tolist()}")

missing, unexpected = vae.load_state_dict(diffusers_state, strict=False)
decoder_missing = [m for m in missing if "decoder" in m or m.startswith("post_quant_conv")]
decoder_unexpected = [u for u in unexpected if "decoder" in u or u.startswith("post_quant_conv")]
print(f"  Total missing: {len(missing)} (decoder-side missing: {len(decoder_missing)})")
print(f"  Total unexpected: {len(unexpected)} (decoder-side unexpected: {len(decoder_unexpected)})")
for m in decoder_missing[:8]:
    print(f"    decoder missing: {m}")
for u in decoder_unexpected[:8]:
    print(f"    decoder unexpected: {u}")
del diffusers_state
gc.collect()

# ── 4. Deterministic synthetic latent ──
torch.manual_seed(42)
latent_raw = torch.randn(1, LATENT_CH, LATENT_H, LATENT_W, dtype=torch.float32)
save(latent_raw, f"{OUT_DIR}/inputs/latent_raw.bin")
save(latents_mean, f"{OUT_DIR}/inputs/latents_mean.bin")
save(latents_std, f"{OUT_DIR}/inputs/latents_std.bin")

# Apply per-channel rescale: latents = latents * std + mean (broadcast over spatial)
mean_5d = latents_mean.view(1, LATENT_CH, 1, 1, 1)
std_5d = latents_std.view(1, LATENT_CH, 1, 1, 1)
latent_5d = latent_raw.unsqueeze(2)  # [1, 16, 1, 32, 32]
latent_rescaled = latent_5d * std_5d + mean_5d
save(latent_rescaled, f"{OUT_DIR}/inputs/latent_rescaled.bin")

print(f"\nLatent raw: shape={tuple(latent_raw.shape)} mean={latent_raw.mean():.4f} std={latent_raw.std():.4f}")
print(f"Rescaled:   shape={tuple(latent_rescaled.shape)} mean={latent_rescaled.mean():.4f} std={latent_rescaled.std():.4f}")

# ── 5. Hook decoder layers ──
saved_layers = []
def make_hook(name: str):
    def _h(module, _in, out):
        if isinstance(out, tuple):
            out = out[0]
        t = out.detach().to(torch.float32).cpu().contiguous()
        safe = name.replace(".", "_").replace("/", "_")
        file_rel = f"layers/{safe}.bin"
        save(t, f"{OUT_DIR}/{file_rel}")
        s = stats_dict(name, t)
        s["file"] = file_rel
        saved_layers.append(s)
        del t
        gc.collect()
    return _h

decoder = vae.decoder
hooks = []
hooks.append(vae.post_quant_conv.register_forward_hook(make_hook("post_quant_conv")))
hooks.append(decoder.conv_in.register_forward_hook(make_hook("decoder.conv_in")))
hooks.append(decoder.mid_block.register_forward_hook(make_hook("decoder.mid_block")))
for i, ub in enumerate(decoder.up_blocks):
    hooks.append(ub.register_forward_hook(make_hook(f"decoder.up_blocks.{i}")))
hooks.append(decoder.conv_out.register_forward_hook(make_hook("decoder.conv_out")))

# ── 6. Decode ──
print("\nRunning vae.decode(latent_rescaled)...")
with torch.no_grad():
    decoded = vae.decode(latent_rescaled, return_dict=False)[0]
    decoded_image = decoded[:, :, 0]  # drop T dim
print(f"Decoded image shape: {tuple(decoded_image.shape)}")
print(f"  mean={decoded_image.mean():.4f}  std={decoded_image.std():.4f}  min={decoded_image.min():.4f}  max={decoded_image.max():.4f}")
save(decoded_image, f"{OUT_DIR}/output_image.bin")

for h in hooks:
    h.remove()

# ── 7. Save index ──
index = {
    "model": "Qwen-Image VAE (AutoencoderKLQwenImage, F32)",
    "config": {
        "base_dim": 96,
        "z_dim": 16,
        "dim_mult": [1, 2, 4, 4],
        "scale_factor_spatial": 8,
        "scale_factor_temporal": 4,
        "latents_mean": vae.config.latents_mean,
        "latents_std": vae.config.latents_std,
    },
    "inputs": {
        "latent_raw_shape": [1, LATENT_CH, LATENT_H, LATENT_W],
        "latent_rescaled_shape": [1, LATENT_CH, 1, LATENT_H, LATENT_W],
    },
    "output_shape": list(decoded_image.shape),
    "output_stats": stats_dict("output_image", decoded_image),
    "layers": saved_layers,
}
with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"\nDONE. VAE reference saved to: {OUT_DIR}")
print(f"  {len(saved_layers)} per-layer captures")
