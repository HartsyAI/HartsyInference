#!/usr/bin/env python3
"""Wan2.1 I2V-14B CPU parity oracle — a faithful float32 port of ComfyUI's `WanModel.forward_orig`
with the I2V CLIP branch (`WanI2VCrossAttention` k_img/v_img + `MLPProj` img_emb), driven by the C#
WAN_DEBUG_DIR stage dumps so both sides see identical inputs. Adapted from ../s2v_reference/.

Reads from --dump-dir (the C# WAN_DEBUG_DIR):
  layers/{tag}_latent_in.bin      [1,36,T,h,w]   the 36-ch concat model input fed to that CFG branch
  layers/{tag}_in_encoder.bin     [L,4096]       raw umT5 features for that branch
  layers/{tag}_clip_embeds.bin    [257,1280]     CLIP-ViT-H penultimate hidden states (shared)
  layers/{tag}_timesteps.bin      [1]            sampler timestep

Writes {out}/layers/{tag}_{stage}.bin with the same stage names the C# side dumps
(post-patchify ≈ patch_embed, temb≈cond_temb, timestep_proj≈cond_timestepProj, text_proj≈cond_textProj,
img_proj≈cond_imgProj, block_{0,20,39}≈blocks_{0,20,39}, velocity_out) so diff_i2v_layers.py can compare.

Usage:
  ../s2v_reference/venv/bin/python dump_i2v_reference.py --dump-dir /tmp/i2v_dbg --tag cond
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(os.cpu_count() or 8)

DIM, HEADS, FREQ_DIM = 5120, 40, 256
HEAD_DIM = DIM // HEADS                      # 128
AXES_DIM = [HEAD_DIM - 4 * (HEAD_DIM // 6), 2 * (HEAD_DIM // 6), 2 * (HEAD_DIM // 6)]   # [44, 42, 42]
PATCH = (1, 2, 2)
EPS = 1e-6
IMG_TOKENS = 257
DEFAULT_CKPT = ("/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Models/Stable-Diffusion/Wan/"
                "wan2.1_i2v_480p_14B_fp8_scaled.safetensors")


class Ckpt:
    """Lazy fp8-scaled checkpoint access: every tensor is fetched on demand and dequantized to fp32."""

    def __init__(self, path):
        self._f = safe_open(path, framework="pt", device="cpu")
        self.keys = set(self._f.keys())

    def get(self, name):
        t = self._f.get_tensor(name)
        if t.dtype == torch.float8_e4m3fn:
            scale_key = name.rsplit(".", 1)[0] + ".scale_weight"
            assert scale_key in self.keys, f"fp8 tensor {name} has no {scale_key} companion"
            return t.float() * self._f.get_tensor(scale_key).float()
        return t.float()

    def opt(self, name):
        return self.get(name) if name in self.keys else None


def linear(x, ck, prefix):
    return F.linear(x, ck.get(prefix + ".weight"), ck.opt(prefix + ".bias"))


def rms_norm(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w


def read_dump(dump_dir, name):
    shapes = {}
    with open(os.path.join(dump_dir, "shapes.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                shapes[parts[0]] = [int(d) for d in parts[1].split(",")]
    if name not in shapes:
        return None
    raw = np.fromfile(os.path.join(dump_dir, "layers", name + ".bin"), dtype=np.float32)
    return torch.from_numpy(raw.copy()).reshape(shapes[name])


class Dumper:
    def __init__(self, out_dir, tag):
        self.layers = os.path.join(out_dir, "layers")
        os.makedirs(self.layers, exist_ok=True)
        self.shapes_path = os.path.join(out_dir, "shapes.txt")
        self.tag = tag + "_" if tag else ""
        self.written = set()

    def __call__(self, name, t):
        full = (self.tag + name).replace(".", "_")
        data = t.detach().float().contiguous()
        with open(os.path.join(self.layers, full + ".bin"), "wb") as f:
            f.write(data.numpy().tobytes())
        if full not in self.written:
            self.written.add(full)
            with open(self.shapes_path, "a") as f:
                f.write(f"{full} {','.join(str(d) for d in data.shape)}\n")


def rope_axis(pos, dim, theta):
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def rope_encode(t, h, w):
    p0, p1, p2 = PATCH
    t_len = (t + p0 // 2) // p0
    h_len = (h + p1 // 2) // p1
    w_len = (w + p2 // 2) // p2
    ids = torch.zeros((t_len, h_len, w_len, 3), dtype=torch.float32)
    ids[:, :, :, 0] += torch.linspace(0, t_len - 1, steps=t_len).reshape(-1, 1, 1)
    ids[:, :, :, 1] += torch.linspace(0, h_len - 1, steps=h_len).reshape(1, -1, 1)
    ids[:, :, :, 2] += torch.linspace(0, w_len - 1, steps=w_len).reshape(1, 1, -1)
    ids = ids.reshape(1, -1, 3)
    emb = torch.cat([rope_axis(ids[..., i], AXES_DIM[i], 10000.0) for i in range(3)], dim=-3)
    return emb.unsqueeze(1).movedim(1, 2)      # [1, S, 1, head_dim/2, 2, 2]


def apply_rope1(x, freqs):
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape)


def attention(q, k, v, heads):
    b, s, _ = q.shape
    d = q.shape[-1] // heads
    q = q.view(b, s, heads, d).transpose(1, 2)
    k = k.view(b, k.shape[1], heads, d).transpose(1, 2)
    v = v.view(b, v.shape[1], heads, d).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v)
    return o.transpose(1, 2).reshape(b, s, heads * d)


def self_attn(ck, p, x, freqs):
    b, s, _ = x.shape
    q = rms_norm(linear(x, ck, p + ".q"), ck.get(p + ".norm_q.weight")).view(b, s, HEADS, HEAD_DIM)
    q = apply_rope1(q, freqs)
    k = rms_norm(linear(x, ck, p + ".k"), ck.get(p + ".norm_k.weight")).view(b, s, HEADS, HEAD_DIM)
    k = apply_rope1(k, freqs)
    v = linear(x, ck, p + ".v")
    o = attention(q.reshape(b, s, DIM), k.reshape(b, s, DIM), v, HEADS)
    return linear(o, ck, p + ".o")


def i2v_cross_attn(ck, p, x, context, context_img, dump=None, dump_prefix=None):
    """ComfyUI WanI2VCrossAttention: text attention + dedicated k_img/v_img image attention, summed pre-o."""
    q = rms_norm(linear(x, ck, p + ".q"), ck.get(p + ".norm_q.weight"))
    k = rms_norm(linear(context, ck, p + ".k"), ck.get(p + ".norm_k.weight"))
    v = linear(context, ck, p + ".v")
    x_text = attention(q, k, v, HEADS)
    k_img = rms_norm(linear(context_img, ck, p + ".k_img"), ck.get(p + ".norm_k_img.weight"))
    v_img = linear(context_img, ck, p + ".v_img")
    x_img = attention(q, k_img, v_img, HEADS)
    if dump is not None:
        dump(f"{dump_prefix}_xattn_text", x_text)
        dump(f"{dump_prefix}_xattn_img", x_img)
    return linear(x_text + x_img, ck, p + ".o")


def repeat_e(e, x):
    repeats = 1
    if e.size(1) > 1:
        repeats = x.size(1) // e.size(1)
    if repeats == 1:
        return e
    if repeats * e.size(1) == x.size(1):
        return torch.repeat_interleave(e, repeats, dim=1)
    return torch.repeat_interleave(e, repeats + 1, dim=1)[:, :x.size(1)]


def block(ck, p, x, e0, freqs, context, context_img, dump=None, dump_prefix=None):
    e = (ck.get(p + ".modulation").unsqueeze(0) + e0).unbind(2)     # six [1, G, dim]
    y = self_attn(ck, p + ".self_attn",
                  torch.addcmul(repeat_e(e[0], x), F.layer_norm(x, (DIM,), eps=EPS), 1 + repeat_e(e[1], x)),
                  freqs)
    x = torch.addcmul(x, y, repeat_e(e[2], x))
    n3 = F.layer_norm(x, (DIM,), ck.get(p + ".norm3.weight"), ck.get(p + ".norm3.bias"), eps=EPS)
    x = x + i2v_cross_attn(ck, p + ".cross_attn", n3, context, context_img, dump, dump_prefix)
    y = torch.addcmul(repeat_e(e[3], x), F.layer_norm(x, (DIM,), eps=EPS), 1 + repeat_e(e[4], x))
    y = linear(F.gelu(linear(y, ck, p + ".ffn.0"), approximate="tanh"), ck, p + ".ffn.2")
    return torch.addcmul(x, y, repeat_e(e[5], x))


def head(ck, x, e):
    e2 = (ck.get("head.modulation").unsqueeze(0) + e.unsqueeze(2)).unbind(2)    # two [1, G, dim]
    xn = F.layer_norm(x, (DIM,), eps=EPS)
    return linear(torch.addcmul(repeat_e(e2[0], x), xn, 1 + repeat_e(e2[1], x)), ck, "head.head")


def unpatchify(x, grid_sizes, out_dim=16):
    b = x.shape[0]
    u = x[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *PATCH, out_dim)
    u = torch.einsum("bfhwpqrc->bcfphqwr", u)
    return u.reshape(b, out_dim, *[g * p for g, p in zip(grid_sizes, PATCH)])


def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    position = position.float()
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def img_emb_mlp(ck, clip_fea):
    """ComfyUI MLPProj (img_emb.proj = Sequential[LN(1280), Linear(1280,1280), GELU, Linear(1280,5120), LN(5120)])."""
    x = F.layer_norm(clip_fea, (clip_fea.shape[-1],),
                     ck.get("img_emb.proj.0.weight"), ck.get("img_emb.proj.0.bias"), eps=1e-5)
    x = linear(x, ck, "img_emb.proj.1")
    x = F.gelu(x)   # exact gelu — NOT tanh (diffusers WanImageEmbedding.ff uses approximate="none")
    x = linear(x, ck, "img_emb.proj.3")
    return F.layer_norm(x, (x.shape[-1],),
                        ck.get("img_emb.proj.4.weight"), ck.get("img_emb.proj.4.bias"), eps=1e-5)


def forward_orig(ck, x, t_scalar, context_raw, clip_fea, dump, n_layers=40):
    bs, _, t_lat, height, width = x.shape
    dump("latent_in", x)
    pw = ck.get("patch_embedding.weight")
    pb = ck.opt("patch_embedding.bias")
    x = F.conv3d(x, pw, pb, stride=PATCH)
    grid_sizes = tuple(x.shape[2:])
    x = x.flatten(2).transpose(1, 2)
    dump("patch_embed", x)

    freqs = rope_encode(t_lat, height, width)

    t = torch.tensor([t_scalar], dtype=torch.float32)
    e = linear(F.silu(linear(sinusoidal_embedding_1d(FREQ_DIM, t), ck, "time_embedding.0")),
               ck, "time_embedding.2")                                  # [1, dim]
    e = e.reshape(1, 1, DIM)                                            # [1, G=1, dim]
    dump("cond_temb", e)
    e0 = linear(F.silu(e), ck, "time_projection.1").unflatten(2, (6, DIM))   # [1, 1, 6, dim]
    dump("cond_timestepProj", e0)

    context = linear(F.gelu(linear(context_raw, ck, "text_embedding.0"), approximate="tanh"),
                     ck, "text_embedding.2")
    dump("cond_textProj", context)
    context_img = img_emb_mlp(ck, clip_fea)
    dump("cond_imgProj", context_img)

    for i in range(n_layers):
        t0 = time.time()
        want_xattn = i == 0
        x = block(ck, f"blocks.{i}", x, e0, freqs, context, context_img,
                  dump if want_xattn else None, f"b{i}" if want_xattn else None)
        if i in (0, n_layers // 2, n_layers - 1):
            dump(f"blocks_{i}", x)
        print(f"  block {i}/{n_layers - 1} ({time.time() - t0:.1f}s)", flush=True)

    x = head(ck, x, e)   # e stays [1, G=1, dim] — head broadcasts modulation [1,2,dim] + e.unsqueeze(2) [1,1,1,dim]
    dump("pre_unpatchify", x)
    v = unpatchify(x, grid_sizes)
    dump("velocity_out", v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="the C# WAN_DEBUG_DIR")
    ap.add_argument("--tag", default="cond", help="branch tag: cond or uncond")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=None, help="reference output dir (default {dump-dir}/ref)")
    ap.add_argument("--layers", type=int, default=40)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.dump_dir, "ref")
    ck = Ckpt(args.ckpt)
    tag = args.tag

    x = read_dump(args.dump_dir, f"{tag}_latent_in")
    context = read_dump(args.dump_dir, f"{tag}_in_encoder")
    clip_fea = read_dump(args.dump_dir, f"{tag}_clip_embeds")
    tsteps = read_dump(args.dump_dir, f"{tag}_timesteps")
    assert x is not None and context is not None and tsteps is not None, \
        f"missing {tag}_latent_in / {tag}_in_encoder / {tag}_timesteps in {args.dump_dir}"
    assert clip_fea is not None, f"missing {tag}_clip_embeds (add the C# dump)"
    if context.dim() == 2:
        context = context.unsqueeze(0)
    if clip_fea.dim() == 2:
        clip_fea = clip_fea.unsqueeze(0)
    t_scalar = float(tsteps.flatten()[0])

    print(f"[i2v-ref] tag={tag} latent={list(x.shape)} text={list(context.shape)} "
          f"clip={list(clip_fea.shape)} t={t_scalar:.3f}", flush=True)
    dump = Dumper(out_dir, tag)
    with torch.inference_mode():
        v = forward_orig(ck, x, t_scalar, context, clip_fea, dump, n_layers=args.layers)
    print(f"[i2v-ref] velocity {list(v.shape)} mean={v.mean():.6f} std={v.std():.6f} → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
