#!/usr/bin/env python3
"""Golden reference for the IP-Adapter FaceID-Plus / Plus-v2 image projection (ProjPlusModel).

Loads the `image_proj` half of a real h94/IP-Adapter-FaceID checkpoint
(ip-adapter-faceid-plus_sd15.bin / ip-adapter-faceid-plusv2_sd15.bin /
ip-adapter-faceid-plusv2_sdxl.bin), runs the official module definitions
(copied verbatim from tencent-ailab/IP-Adapter ip_adapter/ip_adapter_faceid.py +
ip_adapter/resampler.py, float32) on seeded inputs, and dumps inputs + outputs
to a safetensors file consumed by IpAdapterFaceIdPlusParityTests.

For each seeded case i the file contains:
  input_id_{i}          [1, 512]        L2-normalized fake ArcFace embedding
  input_clip_{i}        [1, 257, clip]  fake CLIP-Vision penultimate hidden states
  output_plain_{i}      [1, 4, cross]   shortcut=False   (FaceID-Plus v1 semantics)
  output_shortcut10_{i} [1, 4, cross]   shortcut=True, scale=1.0 (v2 default)
  output_shortcut06_{i} [1, 4, cross]   shortcut=True, scale=0.6

Usage: python ipadapter_faceid_plus_ref.py <faceid-plus-checkpoint.bin> <out_ref.safetensors>
"""

import math
import sys

import torch
import torch.nn as nn
from safetensors.torch import save_file


# ── verbatim reference modules (tencent-ailab/IP-Adapter) ──────────────────────

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, l, _ = latents.shape
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)


class FacePerceiverResampler(nn.Module):
    def __init__(self, *, dim=768, depth=4, dim_head=64, heads=16,
                 embedding_dim=1280, output_dim=768, ff_mult=4):
        super().__init__()
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ]))

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class ProjPlusModel(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512,
                 clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim, depth=4, dim_head=64,
            heads=cross_attention_dim // 64, embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim, ff_mult=4)

    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out


# ── driver ──────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ckpt_path, out_path = sys.argv[1], sys.argv[2]

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    image_proj = state["image_proj"] if "image_proj" in state else {
        k[len("image_proj."):]: v for k, v in state.items() if k.startswith("image_proj.")
    }
    image_proj = {k: v.float() for k, v in image_proj.items()}

    cross_dim = image_proj["norm.weight"].shape[0]
    id_dim = image_proj["proj.0.weight"].shape[1]
    clip_dim = image_proj["perceiver_resampler.proj_in.weight"].shape[1]
    num_tokens = image_proj["proj.2.weight"].shape[0] // cross_dim
    print(f"cross={cross_dim} id={id_dim} clip={clip_dim} tokens={num_tokens}")

    model = ProjPlusModel(cross_attention_dim=cross_dim, id_embeddings_dim=id_dim,
                          clip_embeddings_dim=clip_dim, num_tokens=num_tokens).float().eval()
    missing, unexpected = model.load_state_dict(image_proj, strict=True), None
    print(f"load_state_dict: {missing}")

    tensors = {}
    seq_len = 257  # CLIP ViT-H/14 penultimate: 1 CLS + 256 patches
    for i, seed in enumerate([42, 123]):
        g = torch.Generator().manual_seed(seed)
        id_embeds = torch.randn(1, id_dim, generator=g)
        id_embeds = id_embeds / id_embeds.norm(dim=-1, keepdim=True)
        clip_embeds = torch.randn(1, seq_len, clip_dim, generator=g)
        with torch.inference_mode():
            out_plain = model(id_embeds, clip_embeds, shortcut=False)
            out_s10 = model(id_embeds, clip_embeds, shortcut=True, scale=1.0)
            out_s06 = model(id_embeds, clip_embeds, shortcut=True, scale=0.6)
        tensors[f"input_id_{i}"] = id_embeds.contiguous()
        tensors[f"input_clip_{i}"] = clip_embeds.contiguous()
        tensors[f"output_plain_{i}"] = out_plain.contiguous().clone()
        tensors[f"output_shortcut10_{i}"] = out_s10.contiguous().clone()
        tensors[f"output_shortcut06_{i}"] = out_s06.contiguous().clone()
        print(f"case {i}: plain |x|={out_plain.norm():.4f}  s1.0 |x|={out_s10.norm():.4f}  s0.6 |x|={out_s06.norm():.4f}")

    save_file(tensors, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
