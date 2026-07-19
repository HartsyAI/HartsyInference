"""Generates the Oobleck VAE (Stable Audio Open Small) decode parity fixtures (*.bin, gitignored) checked by
OobleckVaeParityTests.cs. Requires: pip install torch diffusers safetensors numpy, and the real checkpoint at
~/.cache/hartsyinference/models/stable-audio-open-small (FastVideo/stable-audio-open-small-Diffusers repack).
Uses diffusers' own AutoencoderOobleck directly — our downloaded vae/ checkpoint IS the diffusers layout
(named submodules), so no key renaming is needed on the Python side (unlike the DiT, which needed the
upstream stable_audio_tools library instead of diffusers' reimplementation).
"""
import numpy as np
import torch
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck

CKPT_DIR = "/home/hartsy/.cache/hartsyinference/models/stable-audio-open-small/vae"
OUT_DIR = "/home/hartsy/Desktop/HartsyInference/tests/python-reference/stable_audio_open_small_parity"

torch.manual_seed(1)

model = AutoencoderOobleck(
    encoder_hidden_size=128,
    downsampling_ratios=[2, 4, 4, 8, 8],
    channel_multiples=[1, 2, 4, 8, 16],
    decoder_channels=128,
    decoder_input_channels=64,
    audio_channels=2,
    sampling_rate=44100,
)
model.eval()

sd = load_file(f"{CKPT_DIR}/diffusion_pytorch_model.safetensors")
missing, unexpected = model.load_state_dict(sd, strict=True)
assert not missing and not unexpected, (missing, unexpected)

B, latentT = 1, 4
latent = torch.randn(B, 64, latentT, dtype=torch.float32)

with torch.no_grad():
    pcm = model.decode(latent).sample

print("decode output shape", pcm.shape)  # expect [1, 2, latentT * 2048]
for name, arr in [("vae_latent", latent), ("vae_pcm_ref", pcm)]:
    arr.numpy().astype(np.float32).tofile(f"{OUT_DIR}/{name}.bin")
print("saved.")
