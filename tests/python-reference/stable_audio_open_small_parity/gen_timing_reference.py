"""Generates the seconds_total NumberConditioner parity fixture checked by
StableAudioNumberEmbedderParityTests.cs. Inlines stable_audio_tools' NumberConditioner / NumberEmbedder /
TimePositionalEmbedding / LearnedPositionalEmbedding classes verbatim (copied from
stable_audio_tools/models/{conditioners,adp}.py, pip package stable-audio-tools==0.0.19) rather than importing
the package, to avoid its heavy optional deps (torchaudio, einops_exts, flash_attn) that aren't needed for this
one conditioner. Requires only: pip install torch safetensors numpy, and the real checkpoint at
~/.cache/hartsyinference/models/stable-audio-open-small.
"""
from math import pi
import numpy as np
import torch
from torch import nn, Tensor
from einops import rearrange
from safetensors.torch import load_file

CKPT = "/home/hartsy/.cache/hartsyinference/models/stable-audio-open-small/conditioner/diffusion_pytorch_model.safetensors"
OUT_DIR = "/home/hartsy/Desktop/HartsyInference/tests/python-reference/stable_audio_open_small_parity"


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(LearnedPositionalEmbedding(dim), nn.Linear(in_features=dim + 1, out_features=out_features))


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        self.features = features
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        return embedding.view(*shape, self.features)


class NumberConditioner(nn.Module):
    def __init__(self, output_dim: int, min_val: float = 0, max_val: float = 1):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats):
        floats = torch.tensor([float(x) for x in floats])
        floats = floats.clamp(self.min_val, self.max_val)
        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)
        return self.embedder(normalized_floats).unsqueeze(1)


model = NumberConditioner(output_dim=768, min_val=0, max_val=256)
sd = load_file(CKPT)
prefix = "conditioners.seconds_total."
sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
missing, unexpected = model.load_state_dict(sub, strict=True)
assert not missing and not unexpected, (missing, unexpected)

VALUE = 11.89
with torch.no_grad():
    embed = model.forward([VALUE])

print("embed shape", embed.shape)
embed.numpy().astype(np.float32).tofile(f"{OUT_DIR}/timing_embed_ref.bin")
print("saved.")
