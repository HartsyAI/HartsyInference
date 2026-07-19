"""Generates the Stable Audio Open Small DiT parity fixtures (*.bin, gitignored) checked by
StableAudioDitParityTests.cs. Requires: pip install torch einops safetensors numpy huggingface_hub
stable-audio-tools, and the real checkpoint at ~/.cache/hartsyinference/models/stable-audio-open-small
(FastVideo/stable-audio-open-small-Diffusers repack).

The checkpoint's block-level LayerNorms (pre_norm / cross_attend_norm / ff_norm) save real .weight/.bias
keys, but stable_audio_tools' current LayerNorm class names them .gamma/.beta and only allocates .beta as
a real (non-buffer) parameter when bias=True is requested — so this loader renames .weight -> .gamma and
leaves the (always-zero, verified empirically) .bias keys unmapped; the class's default zero .beta buffer
already matches. Do not "fix" this by passing norm_kwargs={"bias": True} unless the checkpoint's norm
biases are also renamed to .beta — see StableAudioDitConfig.cs's class doc for the verified architecture.
"""
import numpy as np
import torch
from safetensors.torch import load_file

from stable_audio_tools.models.dit import DiffusionTransformer

CKPT = "/home/hartsy/.cache/hartsyinference/models/stable-audio-open-small/transformer/diffusion_pytorch_model.safetensors"
OUT_DIR = "/home/hartsy/Desktop/HartsyInference/tests/python-reference/stable_audio_open_small_parity"

torch.manual_seed(0)

model = DiffusionTransformer(
    io_channels=64,
    embed_dim=1024,
    depth=16,
    num_heads=8,
    cond_token_dim=768,
    project_cond_tokens=True,
    global_cond_dim=768,
    project_global_cond=True,
    transformer_type="continuous_transformer",
    global_cond_type="prepend",
    diffusion_objective="rectified_flow",
    attn_kwargs={"qk_norm": "ln"},
)
model.eval()

sd = load_file(CKPT)
renamed = {}
for k, v in sd.items():
    nk = k
    for suffix in (".pre_norm.weight", ".cross_attend_norm.weight", ".ff_norm.weight"):
        if k.endswith(suffix):
            nk = k[: -len(".weight")] + ".gamma"
    renamed[nk] = v
missing, unexpected = model.load_state_dict(renamed, strict=False)
missing_real = [m for m in missing if not m.endswith((".gamma", ".beta"))]
unexpected_real = [u for u in unexpected if not u.endswith(".bias")]
assert not missing_real, f"unexpected missing keys: {missing_real}"
assert not unexpected_real, f"unexpected extra keys: {unexpected_real}"

B, T, Lc = 1, 8, 5
latent = torch.randn(B, 64, T, dtype=torch.float32)
cond = torch.randn(B, Lc, 768, dtype=torch.float32)
global_embed = torch.randn(B, 768, dtype=torch.float32)
timestep = torch.tensor([0.37], dtype=torch.float32)

with torch.no_grad():
    out = model._forward(latent, timestep, cross_attn_cond=cond, global_embed=global_embed)

print("output shape", out.shape)
for name, arr in [("latent", latent), ("cond", cond), ("global_embed", global_embed), ("out_ref", out)]:
    arr.numpy().astype(np.float32).tofile(f"{OUT_DIR}/{name}.bin")
print("saved.")
