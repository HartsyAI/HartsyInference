"""Generates the seconds_total NumberConditioner parity fixture checked by
StableAudioNumberEmbedderParityTests.cs, using the real stable_audio_tools NumberConditioner class + the real
conditioner checkpoint. Requires: pip install torch safetensors numpy stable-audio-tools.
"""
import sys
sys.path.insert(0, "/tmp/sat_dl/sat")
import numpy as np
import torch
from safetensors.torch import load_file
from stable_audio_tools.models.conditioners import NumberConditioner

CKPT = "/home/hartsy/.cache/hartsyinference/models/stable-audio-open-small/conditioner/diffusion_pytorch_model.safetensors"
OUT_DIR = "/home/hartsy/Desktop/HartsyInference/tests/python-reference/stable_audio_open_small_parity"

model = NumberConditioner(output_dim=768, min_val=0, max_val=256)
sd = load_file(CKPT)
prefix = "conditioners.seconds_total."
sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
missing, unexpected = model.load_state_dict(sub, strict=True)
assert not missing and not unexpected, (missing, unexpected)

VALUE = 11.89
with torch.no_grad():
    embed, _mask = model.forward([VALUE], device="cpu")

print("embed shape", embed.shape)
embed.numpy().astype(np.float32).tofile(f"{OUT_DIR}/timing_embed_ref.bin")
print("saved.")
