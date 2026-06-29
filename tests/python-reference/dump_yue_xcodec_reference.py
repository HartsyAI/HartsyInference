"""Standalone reference for the YuE x-codec (SoundStream) DECODE path.

Reimplements, directly from the real ckpt_00360000.pth (codec_model state, exported to
xcodec.safetensors), the only path YuE Stage-1 needs:

    code indices [n_q, B, T]  -> EMA-VQ ResidualVectorQuantization.decode  -> [B, 1024, T]
    -> fc_post2 (Linear 1024->256, applied over channels)                 -> [B, 256, T]
    -> decoder_2 (dac2.Decoder: input=256, channels=1024, rates=[8,5,4,2]) -> [B, 1, samples]

We re-derive each submodule from the named tensors so we do NOT need the upstream package
(audiotools / RepCodec / transformers AutoModel). Weight-norm is fused exactly as PyTorch does
(g * v / ||v|| over all dims except dim 0 for Conv1d and WNConvTranspose1d here).

Outputs a safetensors the C# parity test reads:
  - codes_i32        [n_q, B, T]   the fixed code grid
  - rvq_out          [B, 1024, T]  EMA-VQ decode result
  - latent_in        [B, 256, T]   a fixed latent fed straight to decoder_2 (decoupled test)
  - decoder_out      [B, 1, S]     decoder_2(latent_in)
  - full_out         [B, 1, S]     decoder_2(fc_post2(rvq_out))  (end-to-end codec)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

CKPT = "/tmp/xcodec_full/xcodec.safetensors"
OUT = "/home/kalebbroo/Desktop/Projects/SharpInference/tests/python-reference/yue_xcodec_reference/io.safetensors"

torch.manual_seed(0)

# ----- load all codec_model.* tensors as f32 -----
W = {}
with safe_open(CKPT, "pt") as f:
    for k in f.keys():
        assert k.startswith("codec_model.")
        W[k[len("codec_model."):]] = f.get_tensor(k).float()


def wn(prefix):
    """Fuse weight_norm: w = g * v / ||v|| with norm over all dims except dim 0."""
    g = W[f"{prefix}.weight_g"]
    v = W[f"{prefix}.weight_v"]
    norm = v.norm(dim=tuple(range(1, v.dim())), keepdim=True)
    return g * v / norm


def snake(x, alpha):
    # Snake1d: x + (1/alpha) * sin(alpha x)^2 ; alpha shape [1, C, 1]
    a = alpha
    return x + (1.0 / (a + 1e-9)) * torch.sin(a * x) ** 2


def conv1d(x, prefix, k, pad, stride=1, dilation=1):
    w = wn(prefix)
    b = W[f"{prefix}.bias"]
    return F.conv1d(x, w, b, stride=stride, padding=pad, dilation=dilation)


def residual_unit(x, prefix, dilation):
    # block.0 Snake, block.1 WNConv1d k7 dil, block.2 Snake, block.3 WNConv1d k1
    pad = ((7 - 1) * dilation) // 2
    y = snake(x, W[f"{prefix}.block.0.alpha"])
    y = conv1d(y, f"{prefix}.block.1", 7, pad, dilation=dilation)
    y = snake(y, W[f"{prefix}.block.2.alpha"])
    y = conv1d(y, f"{prefix}.block.3", 1, 0)
    p = (x.shape[-1] - y.shape[-1]) // 2
    if p > 0:
        x = x[..., p:-p]
    return x + y


def decoder_block(x, prefix, stride, out_pad):
    # block.0 Snake, block.1 WNConvTranspose1d, block.2/3/4 ResidualUnit(d=1,3,9)
    x = snake(x, W[f"{prefix}.block.0.alpha"])
    w = wn(f"{prefix}.block.1")
    b = W[f"{prefix}.block.1.bias"]
    k = 2 * stride
    pad = math.ceil(stride / 2)
    x = F.conv_transpose1d(x, w, b, stride=stride, padding=pad, output_padding=out_pad)
    x = residual_unit(x, f"{prefix}.block.2", 1)
    x = residual_unit(x, f"{prefix}.block.3", 3)
    x = residual_unit(x, f"{prefix}.block.4", 9)
    return x


def decoder_2(x):
    # model.0 WNConv1d(256->1024,k7,pad3)
    x = conv1d(x, "decoder_2.model.0", 7, 3)
    rates = [8, 5, 4, 2]
    for i, stride in enumerate(rates):
        out_pad = 1 if i == 1 else 0
        x = decoder_block(x, f"decoder_2.model.{i+1}", stride, out_pad)
    # model.5 Snake, model.6 WNConv1d(64->1,k7,pad3)   (no tanh in xcodec)
    x = snake(x, W["decoder_2.model.5.alpha"])
    x = conv1d(x, "decoder_2.model.6", 7, 3)
    return x


def fc_post2(latent_bct):
    # Linear over channels: latent [B, 1024, T] -> [B, 256, T]
    w = W["fc_post2.weight"]  # [256, 1024]
    b = W["fc_post2.bias"]
    y = latent_bct.transpose(1, 2)  # [B, T, 1024]
    y = F.linear(y, w, b)           # [B, T, 256]
    return y.transpose(1, 2)        # [B, 256, T]


def rvq_decode(codes):
    # codes [n_q, B, T] long. EMA EuclideanCodebook.decode = F.embedding(idx, embed); sum over n_q.
    out = None
    for i in range(codes.shape[0]):
        embed = W[f"quantizer.vq.layers.{i}._codebook.embed"]  # [1024, 1024]
        q = F.embedding(codes[i], embed)  # [B, T, D]
        q = q.transpose(1, 2)             # [B, D, T]
        out = q if out is None else out + q
    return out


# ----- fixed inputs -----
T = 16
B = 1
nq = 1
# deterministic code grid in [0, 1024)
codes = (torch.arange(T) * 37 + 11) % 1024
codes = codes.view(1, 1, T).expand(nq, B, T).contiguous().long()

rvq_out = rvq_decode(codes)                 # [B, 1024, T]

# decoupled decoder test: a fixed latent of the decoder's input dim (256)
latent_in = torch.randn(B, 256, T)
decoder_out = decoder_2(latent_in)          # [B, 1, S]

# end-to-end codec decode
full_out = decoder_2(fc_post2(rvq_out))     # [B, 1, S]

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file({
    "codes_i32": codes.to(torch.int32).contiguous(),
    "rvq_out": rvq_out.contiguous(),
    "latent_in": latent_in.contiguous(),
    "decoder_out": decoder_out.contiguous(),
    "full_out": full_out.contiguous(),
}, OUT)

print("T", T, "rvq_out", tuple(rvq_out.shape), "decoder_out", tuple(decoder_out.shape),
      "full_out", tuple(full_out.shape))
print("rvq_out  mean/std", float(rvq_out.mean()), float(rvq_out.std()))
print("dec_out  min/max/rms", float(decoder_out.min()), float(decoder_out.max()),
      float(decoder_out.pow(2).mean().sqrt()))
print("full_out min/max/rms", float(full_out.min()), float(full_out.max()),
      float(full_out.pow(2).mean().sqrt()))
print("wrote", OUT)
