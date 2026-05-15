"""
Dumps a full Anima (Cosmos-Predict2 family) DiT forward pass with per-block outputs.

Anima is built on NVIDIA Cosmos-Predict2-2B-Text2Image with an added in-checkpoint
`net.llm_adapter` (6-block self+cross+MLP transformer that refines Qwen-3 0.6B features
before the DiT cross-attention). Diffusers has the DiT trunk via CosmosTransformer3DModel
but does NOT have the llm_adapter — so this script dumps ONLY the DiT trunk reference.
The C# side must dump with ANIMA_BYPASS_LLM_ADAPTER=1 to match (feeding raw Qwen-3 hidden
states directly to the DiT cross-attn, mirroring what diffusers does).

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_anima_full_forward.py

Output: tests/python-reference/anima_reference_tensors/full_forward/
  inputs/latent.bin              [1, 16, 1, 128, 128] F32 (deterministic seed 42; 5-D for Cosmos)
  inputs/encoder_hidden.bin      [1, 6, 1024]   F32 (deterministic seed 42 — stands in for Qwen-3 output)
  inputs/padding_mask.bin        [1, 1, 1024, 1024] F32 (all zeros)
  layers/<safe_name>.bin         per-layer F32 outputs
  output_velocity.bin            final transformer output [1, 16, 1, 128, 128] F32
  index.json                     stats per layer + config

The C# side dumps with ANIMA_DEBUG_DIR=Output/anima_csharp_dump and ANIMA_BYPASS_LLM_ADAPTER=1.
diff_anima_layers.py compares the two trees layer-by-layer.
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from safetensors import safe_open
from diffusers import CosmosTransformer3DModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANIMA_PATH = "/home/kalebbroo/Desktop/Projects/SharpInference/Models/Stable-Diffusion/Anima/anima-preview3-base.safetensors"
OUT_DIR = os.path.join(REPO_ROOT, "tests/python-reference/anima_reference_tensors/full_forward")

# Synthetic input dims (small to keep RAM/time reasonable for CPU torch).
LATENT_H = 128   # 1024px / 8x VAE
LATENT_W = 128
CAP_LEN = 6      # matches the user's actual Qwen-3 output for "a photo of a cat"
TEXT_EMBED_DIM = 1024
SIGMA = 0.5      # mid-trajectory timestep (Cosmos convention: t = sigma/(sigma+1) is what model sees)

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


# ── 1. Translate Anima `net.*` keys → diffusers CosmosTransformer3DModel naming ──
def translate_anima_to_diffusers(anima_state: dict) -> dict:
    """Maps Anima single-file safetensors keys to diffusers CosmosTransformer3DModel state_dict keys.
    Drops `net.llm_adapter.*` (not in diffusers — Anima-specific addition)."""
    out = {}
    for key, tensor in anima_state.items():
        # Strip `net.` prefix
        if key.startswith("net."):
            key = key[len("net."):]

        # Drop llm_adapter entirely — diffusers Cosmos has no equivalent
        if key.startswith("llm_adapter."):
            continue

        # Top-level translations
        if key == "x_embedder.proj.1.weight":
            out["patch_embed.proj.weight"] = tensor
            continue
        if key == "t_embedder.1.linear_1.weight":
            out["time_embed.t_embedder.linear_1.weight"] = tensor
            continue
        if key == "t_embedder.1.linear_2.weight":
            out["time_embed.t_embedder.linear_2.weight"] = tensor
            continue
        if key == "t_embedding_norm.weight":
            out["time_embed.norm.weight"] = tensor
            continue
        if key == "final_layer.adaln_modulation.1.weight":
            out["norm_out.linear_1.weight"] = tensor
            continue
        if key == "final_layer.adaln_modulation.2.weight":
            out["norm_out.linear_2.weight"] = tensor
            continue
        if key == "final_layer.linear.weight":
            out["proj_out.weight"] = tensor
            continue

        # Per-block translations: blocks.{i}.X → transformer_blocks.{i}.X
        if key.startswith("blocks."):
            parts = key.split(".")
            block_idx = parts[1]
            rest = ".".join(parts[2:])

            # AdaLN-LoRA modulators (3 per block, each with .1.weight and .2.weight)
            ada_map = {
                "adaln_modulation_self_attn.1.weight": "norm1.linear_1.weight",
                "adaln_modulation_self_attn.2.weight": "norm1.linear_2.weight",
                "adaln_modulation_cross_attn.1.weight": "norm2.linear_1.weight",
                "adaln_modulation_cross_attn.2.weight": "norm2.linear_2.weight",
                "adaln_modulation_mlp.1.weight": "norm3.linear_1.weight",
                "adaln_modulation_mlp.2.weight": "norm3.linear_2.weight",
            }
            if rest in ada_map:
                out[f"transformer_blocks.{block_idx}.{ada_map[rest]}"] = tensor
                continue

            # Self-attn
            attn_map_self = {
                "self_attn.q_proj.weight": "attn1.to_q.weight",
                "self_attn.k_proj.weight": "attn1.to_k.weight",
                "self_attn.v_proj.weight": "attn1.to_v.weight",
                "self_attn.output_proj.weight": "attn1.to_out.0.weight",
                "self_attn.q_norm.weight": "attn1.norm_q.weight",
                "self_attn.k_norm.weight": "attn1.norm_k.weight",
            }
            if rest in attn_map_self:
                out[f"transformer_blocks.{block_idx}.{attn_map_self[rest]}"] = tensor
                continue

            # Cross-attn
            attn_map_cross = {
                "cross_attn.q_proj.weight": "attn2.to_q.weight",
                "cross_attn.k_proj.weight": "attn2.to_k.weight",
                "cross_attn.v_proj.weight": "attn2.to_v.weight",
                "cross_attn.output_proj.weight": "attn2.to_out.0.weight",
                "cross_attn.q_norm.weight": "attn2.norm_q.weight",
                "cross_attn.k_norm.weight": "attn2.norm_k.weight",
            }
            if rest in attn_map_cross:
                out[f"transformer_blocks.{block_idx}.{attn_map_cross[rest]}"] = tensor
                continue

            # MLP / FeedForward (Cosmos uses GELU FFN: net.0.proj → first linear, net.2 → second)
            if rest == "mlp.layer1.weight":
                out[f"transformer_blocks.{block_idx}.ff.net.0.proj.weight"] = tensor
                continue
            if rest == "mlp.layer2.weight":
                out[f"transformer_blocks.{block_idx}.ff.net.2.weight"] = tensor
                continue

            print(f"WARNING: unmapped block key: {key}")
            continue

        print(f"WARNING: unmapped top-level key: {key}")
    return out


# ── 2. Load Anima safetensors ──
print(f"Loading Anima checkpoint: {ANIMA_PATH}")
anima_state = {}
with safe_open(ANIMA_PATH, framework="pt", device="cpu") as f:
    for key in f.keys():
        anima_state[key] = f.get_tensor(key).float()
print(f"  Loaded {len(anima_state)} tensors")

print("Translating Anima keys → diffusers naming, dropping llm_adapter...")
diffusers_state = translate_anima_to_diffusers(anima_state)
print(f"  Translated to {len(diffusers_state)} diffusers keys")

# ── 3. Construct diffusers Cosmos model with Anima's config ──
# Anima uses extra_pos_embed_type=None (no learnable_pos_embed in checkpoint).
print("Constructing CosmosTransformer3DModel with Anima config...")
xfm = CosmosTransformer3DModel(
    in_channels=16,
    out_channels=16,
    num_attention_heads=16,
    attention_head_dim=128,
    num_layers=28,
    mlp_ratio=4.0,
    text_embed_dim=1024,
    adaln_lora_dim=256,
    max_size=(128, 240, 240),
    patch_size=(1, 2, 2),
    rope_scale=(2.0, 1.0, 1.0),
    concat_padding_mask=True,
    extra_pos_embed_type=None,        # Anima omits this — confirmed by checkpoint
    use_crossattn_projection=False,   # No crossattn_proj key in checkpoint
).eval()

missing, unexpected = xfm.load_state_dict(diffusers_state, strict=False)
print(f"  Missing keys: {len(missing)}")
for m in missing[:10]:
    print(f"    {m}")
if len(missing) > 10:
    print(f"    ... and {len(missing) - 10} more")
print(f"  Unexpected keys: {len(unexpected)}")
for u in unexpected[:10]:
    print(f"    {u}")

# ── 4. Build deterministic inputs ──
torch.manual_seed(42)
# Cosmos expects 5-D latent [B, C, T, H, W]
latent_bcthw = torch.randn(1, 16, 1, LATENT_H, LATENT_W, dtype=torch.float32)
encoder_hidden = torch.randn(1, CAP_LEN, TEXT_EMBED_DIM, dtype=torch.float32)
padding_mask = torch.zeros(1, 1, LATENT_H * 8, LATENT_W * 8, dtype=torch.float32)

save(latent_bcthw, f"{OUT_DIR}/inputs/latent.bin")
save(encoder_hidden, f"{OUT_DIR}/inputs/encoder_hidden.bin")
save(padding_mask, f"{OUT_DIR}/inputs/padding_mask.bin")
print(f"Synthetic inputs:")
print(f"  latent {tuple(latent_bcthw.shape)}, mean={latent_bcthw.mean():.4f}, std={latent_bcthw.std():.4f}")
print(f"  encoder {tuple(encoder_hidden.shape)}, mean={encoder_hidden.mean():.4f}, std={encoder_hidden.std():.4f}")
print(f"  padding_mask {tuple(padding_mask.shape)} (all zeros)")

# Timestep: Cosmos expects t in [0, ~1] range. t = sigma / (sigma + 1).
current_t = SIGMA / (SIGMA + 1.0)
timestep = torch.tensor([current_t], dtype=torch.float32)
print(f"  timestep input: {current_t:.6f} (sigma={SIGMA})")

# ── 5. Hook every relevant submodule for layer-by-layer capture ──
captures = {}

def make_hook(name: str):
    def _h(module, _in, out):
        if isinstance(out, tuple):
            out = out[0]
        captures[name] = out.detach().clone().float()
    return _h

hooks = []
# Top-level submodules
hooks.append(xfm.patch_embed.register_forward_hook(make_hook("patch_embed")))
hooks.append(xfm.time_embed.register_forward_hook(make_hook("time_embed")))
hooks.append(xfm.norm_out.register_forward_hook(make_hook("norm_out")))
hooks.append(xfm.proj_out.register_forward_hook(make_hook("proj_out")))

# Per-block — capture the full block output. The transformer_blocks ModuleList returns the hidden states.
for i, blk in enumerate(xfm.transformer_blocks):
    hooks.append(blk.register_forward_hook(make_hook(f"block_{i:02d}")))

# ── 6. Forward pass ──
print("Running forward pass...")
with torch.no_grad():
    out = xfm(
        hidden_states=latent_bcthw,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden,
        padding_mask=padding_mask,
        return_dict=False,
    )[0]
print(f"  Output shape: {tuple(out.shape)}, mean={out.float().mean():.4f}, std={out.float().std():.4f}")

# Remove hooks
for h in hooks:
    h.remove()

# ── 7. Save all captures + index ──
print(f"Saving {len(captures)} layer captures to {OUT_DIR}/layers/")
layers = []
for name, tensor in captures.items():
    safe = name.replace(".", "_").replace("/", "_")
    file_rel = f"layers/{safe}.bin"
    save(tensor, f"{OUT_DIR}/{file_rel}")
    s = stats(name, tensor)
    s["file"] = file_rel
    layers.append(s)

save(out, f"{OUT_DIR}/output_velocity.bin")

index = {
    "model": "Anima (Cosmos-Predict2 2B, llm_adapter excluded)",
    "config": {
        "in_channels": 16,
        "num_attention_heads": 16,
        "attention_head_dim": 128,
        "num_layers": 28,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "patch_size": [1, 2, 2],
        "extra_pos_embed_type": None,
        "use_crossattn_projection": False,
    },
    "inputs": {
        "latent_shape": list(latent_bcthw.shape),
        "encoder_hidden_shape": list(encoder_hidden.shape),
        "timestep": current_t,
        "sigma": SIGMA,
    },
    "output_shape": list(out.shape),
    "output_stats": stats("output_velocity", out),
    "layers": layers,
}
with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"\nDONE. Reference saved to: {OUT_DIR}")
print(f"  {len(layers)} per-layer dumps")
print(f"  Output velocity: {tuple(out.shape)}")
