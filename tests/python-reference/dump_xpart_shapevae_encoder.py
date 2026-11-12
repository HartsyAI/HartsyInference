"""Reference values for the X-Part ShapeVAE encoder, produced by upstream's own modules on the CPU.

Runs `FourierEmbedder`, `ResidualCrossAttentionBlock` and `Transformer` from `XPart/partgen/models/autoencoders/attention_blocks.py` of a checkout of
tencent/Hunyuan3D-Part (pinned in docs/Research/HUNYUAN3D_PART_ARCHITECTURE.md) with the released weights, composed the way `PointCrossAttentionEncoder.forward` composes them
(`input_proj`, one cross-attention, the self-attention blocks, `ln_post`). Differences from a normal run, all to keep the dump safe and dependency-free:

* `einops` is not installed here: `rearrange` is replaced by the one pattern upstream uses (`b n h d -> b h n d`). The two relative imports of the file are replaced by their
  contents (`attention_processors.py`) and a plain logger; no other upstream file is loaded.
* The queries are given, not drawn: upstream picks them with `torch_cluster.fps` from a random start (and permutes the input points, which a cross-attention does not see). The
  script picks `--queries` farthest points from index 0 with a plain farthest-point loop and writes the indices, and the C# test feeds the same indices.
* The weights are read from the safetensors files with `safetensors` (BF16 widened to float32, as the engine does).

    python dump_xpart_shapevae_encoder.py --upstream <checkout>/XPart/partgen/models/autoencoders --part <hunyuan3d-part folder> --out <fixture.safetensors>

Fixture keys: `points` [N, 3], `feats` [N, 4], `query_idx` [Q] (float32), and for the encoder of `shapevae/shapevae.safetensors` (`vae.*`) and of the object encoder inside
`conditioner/conditioner.safetensors` (`obj.*`) and of its local encoder (`local.*`): `latents` [Q, 1024]. Times are printed.
"""
import argparse
import logging
import math
import os
import time
import types

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

SIZE = 1024
HEADS = 16
LAYERS = 8
NUM_FREQS = 8
POINT_FEATS = 4


def load_upstream(folder):
    """Executes attention_blocks.py with its two relative imports and einops replaced; returns the module namespace."""
    with open(os.path.join(folder, "attention_processors.py"), encoding="utf-8") as handle:
        processors = types.ModuleType("attention_processors")
        exec(compile(handle.read(), "attention_processors.py", "exec"), processors.__dict__)
    with open(os.path.join(folder, "attention_blocks.py"), encoding="utf-8") as handle:
        source = handle.read()
    source = source.replace("from einops import rearrange", "rearrange = _rearrange")
    source = source.replace("from .attention_processors import CrossAttentionProcessor", "CrossAttentionProcessor = _processors.CrossAttentionProcessor")
    source = source.replace("from ...utils.misc import logger", "logger = _logging.getLogger('xpart')")

    def _rearrange(t, pattern, h=None):
        assert pattern == "b n h d -> b h n d", pattern
        return t.permute(0, 2, 1, 3)

    module = types.ModuleType("attention_blocks")
    module.__dict__.update({"_rearrange": _rearrange, "_processors": processors, "_logging": logging})
    exec(compile(source, "attention_blocks.py", "exec"), module.__dict__)
    return module


def farthest_points(points, count):
    chosen = [0]
    distance = ((points - points[0]) ** 2).sum(1)
    for _ in range(count - 1):
        index = int(distance.argmax())
        chosen.append(index)
        distance = torch.minimum(distance, ((points - points[index]) ** 2).sum(1))
    return torch.tensor(chosen)


def encode(blocks, tensors, prefix, points, feats, query_idx):
    """PointCrossAttentionEncoder.forward with given query indices."""
    fourier = blocks.FourierEmbedder(num_freqs=NUM_FREQS, include_pi=False)
    input_proj = nn.Linear(fourier.out_dim + POINT_FEATS, SIZE)
    cross = blocks.ResidualCrossAttentionBlock(width=SIZE, heads=HEADS, qkv_bias=False, qk_norm=True)
    self_attn = blocks.Transformer(width=SIZE, layers=LAYERS, heads=HEADS, qkv_bias=False, qk_norm=True)
    ln_post = nn.LayerNorm(SIZE)
    p = prefix + "encoder."
    state = lambda m, sub: {k[len(p + sub):]: v for k, v in tensors.items() if k.startswith(p + sub)}
    input_proj.load_state_dict(state(input_proj, "input_proj."))
    cross.load_state_dict(state(cross, "cross_attn."))
    self_attn.load_state_dict(state(self_attn, "self_attn."))
    ln_post.load_state_dict(state(ln_post, "ln_post."))
    with torch.no_grad():
        data = torch.cat([fourier(points), feats], dim=-1)[None]
        query = torch.cat([fourier(points[query_idx]), feats[query_idx]], dim=-1)[None]
        data, query = input_proj(data), input_proj(query)
        x = cross(query, data)
        after_cross = x[0].clone()
        x = ln_post(self_attn(x))
    return after_cross, x[0]


def read(path, keep):
    with safe_open(path, framework="pt") as handle:
        return {k: handle.get_tensor(k).float() for k in handle.keys() if k.startswith(keep)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=256)
    args = parser.parse_args()
    torch.manual_seed(0)
    blocks = load_upstream(args.upstream)

    points = torch.randn(args.points, 3)
    points = points / points.norm(dim=1, keepdim=True) * (0.6 + 0.2 * torch.rand(args.points, 1))
    normals = torch.randn(args.points, 3)
    feats = torch.cat([normals / normals.norm(dim=1, keepdim=True), torch.rand(args.points, 1)], dim=1)
    query_idx = farthest_points(points, args.queries)
    out = {"points": points, "feats": feats, "query_idx": query_idx.float()}

    for name, path, prefix in (
        ("vae", os.path.join(args.part, "shapevae", "shapevae.safetensors"), ""),
        ("obj", os.path.join(args.part, "conditioner", "conditioner.safetensors"), "obj_encoder."),
        ("local", os.path.join(args.part, "conditioner", "conditioner.safetensors"), "geo_encoder.local_encoder."),
    ):
        tensors = read(path, prefix + "encoder.")
        start = time.time()
        _, latents = encode(blocks, tensors, prefix, points, feats, query_idx)
        out[f"{name}.latents"] = latents
        print(f"{name}: {tuple(latents.shape)} in {time.time() - start:.1f} s, latent std {latents.std():.4f}")
    save_file({k: v.contiguous() for k, v in out.items()}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
