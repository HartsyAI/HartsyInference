#!/usr/bin/env python3
"""Wan2.2-S2V CPU parity oracle — a faithful float32 port of ComfyUI's `WanModel_S2V.forward_orig`
(+ `CausalAudioEncoder` / `MotionEncoder_tc` / `AudioInjector_WAN` / `WanAttentionBlock` / `Head`) from
`comfy/ldm/wan/model.py`, driven by the C# WAN_DEBUG_DIR stage dumps so both sides see identical inputs.

comfy's `operations.Linear/LayerNorm/RMSNorm` are replaced by plain functional torch ops with weights
streamed lazily from the real fp8 checkpoint (dequant = fp8_tensor.float() * `<name>.scale_weight`
companion) — the 14B model is never materialized in fp32 at once, so this fits a 62 GB RAM box.
`optimized_attention` / `apply_rope1` are replaced with plain SDPA / the `_apply_rope1` math.

Reads from --dump-dir (the C# WAN_DEBUG_DIR), all raw little-endian F32 with a `shapes.txt` sidecar:
  layers/{tag}_latent_in.bin        [1,16,T,H,W]     noisy latent fed to that CFG branch
  layers/{tag}_text_embeds.bin      [L,4096]         umT5 features fed to that branch
  layers/{tag}_audio_features.bin   [4T,25,1024]     stacked Wav2Vec2 features (uncond branch = zeros)
  layers/{tag}_timesteps.bin        [G]              per-group timesteps; element 0 is the sampler timestep
  layers/ref_latent.bin             [1,16,refT,h,w]  VAE-encoded reference image (untagged/shared, optional)

Writes {out}/layers/{tag}_{stage}.bin + {out}/shapes.txt with the SAME stage names the C# side dumps
(post_patchify, post_condmask, ref_tokens, joined_tokens, temb, timestep_proj, text_proj, audio_global,
audio_local, block_{0,20,39}, audio_delta_inj{0,11}, pre_unpatchify, velocity_out) so diff_s2v_layers.py
can walk both dirs. NOTE: C# token tensors are [S,dim] (B=1 dropped) vs [1,S,dim] here — the diff
compares flat, element counts match.

Usage:
  venv/bin/python dump_s2v_reference.py --dump-dir /tmp/s2v_dbg --tag cond
  venv/bin/python dump_s2v_reference.py --dump-dir /tmp/s2v_dbg --tag uncond
Env/flags: --ckpt (default the real S2V fp8 checkpoint), --out (default {dump-dir}/ref),
  --no-control skips the cond_encoder zero-control bias add (ComfyUI's node always feeds a zeros
  control latent — process_in(process_out(0)) == 0 — so the default matches ComfyUI AND the C# port).
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
ADALN_EPS = 1e-5                             # AdaLayerNorm is built without an explicit norm_eps
AUDIO_TOKENS = 4
INJECT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39]   # WanModel_S2V default (12 injectors / 40 blocks)
DEFAULT_CKPT = "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/Wan/wan2.2_s2v_14B_fp8_scaled.safetensors"


# ---------------------------------------------------------------- checkpoint

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


# ---------------------------------------------------------------- dump I/O

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


# ---------------------------------------------------------------- RoPE (flux math, fp64 like comfy)

def rope_axis(pos, dim, theta):
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32), omega)   # promotes to fp64
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def rope_encode(t, h, w, t_start=0):
    p0, p1, p2 = PATCH
    t_len = (t + p0 // 2) // p0
    h_len = (h + p1 // 2) // p1
    w_len = (w + p2 // 2) // p2
    ids = torch.zeros((t_len, h_len, w_len, 3), dtype=torch.float32)
    ids[:, :, :, 0] += torch.linspace(t_start, t_start + (t_len - 1), steps=t_len).reshape(-1, 1, 1)
    ids[:, :, :, 1] += torch.linspace(0, h_len - 1, steps=h_len).reshape(1, -1, 1)
    ids[:, :, :, 2] += torch.linspace(0, w_len - 1, steps=w_len).reshape(1, 1, -1)
    ids = ids.reshape(1, -1, 3)
    emb = torch.cat([rope_axis(ids[..., i], AXES_DIM[i], 10000.0) for i in range(3)], dim=-3)
    return emb.unsqueeze(1).movedim(1, 2)      # [1, S, 1, head_dim/2, 2, 2]


def apply_rope1(x, freqs):
    """x [B,S,N,D]; freqs [1,S,1,D/2,2,2] — comfy _apply_rope1 (interleaved-pair 2x2 rotation)."""
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape)


# ---------------------------------------------------------------- attention / blocks

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


def cross_attn(ck, p, x, context):
    q = rms_norm(linear(x, ck, p + ".q"), ck.get(p + ".norm_q.weight"))
    k = rms_norm(linear(context, ck, p + ".k"), ck.get(p + ".norm_k.weight"))
    v = linear(context, ck, p + ".v")
    return linear(attention(q, k, v, HEADS), ck, p + ".o")


def repeat_e(e, x):
    repeats = 1
    if e.size(1) > 1:
        repeats = x.size(1) // e.size(1)
    if repeats == 1:
        return e
    if repeats * e.size(1) == x.size(1):
        return torch.repeat_interleave(e, repeats, dim=1)
    return torch.repeat_interleave(e, repeats + 1, dim=1)[:, :x.size(1)]


def block(ck, p, x, e0, freqs, context):
    e = (ck.get(p + ".modulation").unsqueeze(0) + e0).unbind(2)     # six [1, G, dim]
    y = self_attn(ck, p + ".self_attn",
                  torch.addcmul(repeat_e(e[0], x), F.layer_norm(x, (DIM,), eps=EPS), 1 + repeat_e(e[1], x)),
                  freqs)
    x = torch.addcmul(x, y, repeat_e(e[2], x))
    n3 = F.layer_norm(x, (DIM,), ck.get(p + ".norm3.weight"), ck.get(p + ".norm3.bias"), eps=EPS)
    x = x + cross_attn(ck, p + ".cross_attn", n3, context)
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


# ---------------------------------------------------------------- audio encoder (CausalAudioEncoder + MotionEncoder_tc)

def causal_conv1d(ck, p, x, stride):
    w = ck.get(p + ".conv.weight")
    x = F.pad(x, (w.shape[2] - 1, 0), mode="replicate")
    return F.conv1d(x, w, ck.opt(p + ".conv.bias"), stride=stride)


def _stream_chain(ck, p, x):
    """The shared LN+SiLU / conv2 / conv3 chain on a [B, C, T] stream → [B, T/4, dim]."""
    x = x.transpose(1, 2)
    x = F.silu(F.layer_norm(x, (DIM // 4,), eps=EPS))
    x = causal_conv1d(ck, p + ".conv2", x.transpose(1, 2), 2).transpose(1, 2)
    x = F.silu(F.layer_norm(x, (DIM // 2,), eps=EPS))
    x = causal_conv1d(ck, p + ".conv3", x.transpose(1, 2), 2).transpose(1, 2)
    return F.silu(F.layer_norm(x, (DIM,), eps=EPS))


def motion_encoder(ck, p, x):
    """x [1, frames, audio_dim] → (global [1, T, 1, dim], local [1, T, tokens+1, dim])."""
    n = AUDIO_TOKENS
    x = x.transpose(1, 2)                                            # b c t
    x_ori = x.clone()
    b, _, t = x.shape
    x = causal_conv1d(ck, p + ".conv1_local", x, 1)                  # [1, (dim//4)*n, t]
    x = x.view(b, n, DIM // 4, t).reshape(b * n, DIM // 4, t)        # 'b (n c) t -> (b n) c t'
    x = _stream_chain(ck, p, x)                                      # [(b n), t4, dim]
    t4 = x.shape[1]
    x = x.view(b, n, t4, DIM).permute(0, 2, 1, 3)                    # '(b n) t c -> b t n c'
    padding = ck.get(p + ".padding_tokens").repeat(b, t4, 1, 1)      # [1,1,1,dim]
    x_local = torch.cat([x, padding], dim=-2)

    g = causal_conv1d(ck, p + ".conv1_global", x_ori, 1)             # [1, dim//4, t]
    g = _stream_chain(ck, p, g)                                      # [1, t4, dim]
    g = linear(g, ck, p + ".final_linear")
    return g.unsqueeze(2), x_local                                   # '(b n) t c -> b t n c', n=1


def causal_audio_encoder(ck, audio):
    """audio [1, num_layers, audio_dim, frames] — SiLU-weighted layer MEAN then the conv chain."""
    w = F.silu(ck.get("casual_audio_encoder.weights"))               # [1, 25, 1, 1]
    feat = ((audio * w) / w.sum(dim=1, keepdims=True)).sum(dim=1)    # [1, audio_dim, frames]
    return motion_encoder(ck, "casual_audio_encoder.encoder", feat.permute(0, 2, 1))


# ---------------------------------------------------------------- audio injector

def audio_injector(ck, idx, x, audio_emb, audio_emb_global, seq_len, dump=None):
    b = x.shape[0]
    t = audio_emb.shape[1]
    n = seq_len // t
    hid = x[:, :seq_len].reshape(b, t, n, DIM).reshape(b * t, n, DIM)          # 'b (t n) c -> (b t) n c'
    g = audio_emb_global.reshape(b * t, -1, DIM)[:, 0]                          # [(b t), dim]
    temb = linear(F.silu(g), ck, f"audio_injector.injector_adain_layers.{idx}.linear")
    shift, scale = temb.chunk(2, dim=1)
    adain = F.layer_norm(hid, (DIM,), eps=ADALN_EPS) * (1 + scale[:, None, :]) + shift[:, None, :]
    attn_audio = audio_emb.reshape(b * t, -1, DIM)                              # 'b t n c -> (b t) n c'
    residual = cross_attn(ck, f"audio_injector.injector.{idx}", adain, attn_audio)
    residual = residual.reshape(b, seq_len, DIM)                                # '(b t) n c -> b (t n) c'
    if dump is not None:
        dump(f"audio_delta_inj{idx}", residual)
    out = x.clone()
    out[:, :seq_len] = out[:, :seq_len] + residual
    return out


# ---------------------------------------------------------------- forward_orig

def forward_orig(ck, x, t_scalar, context, audio_embed, reference_latent, dump, apply_control=True, n_layers=40):
    num_embeds = x.shape[-3] * 4
    audio_emb_global, audio_emb = causal_audio_encoder(ck, audio_embed[:, :, :, :num_embeds])
    dump("audio_global", audio_emb_global)
    dump("audio_local", audio_emb)

    bs, _, t_lat, height, width = x.shape
    dump("latent_in", x)
    pw = ck.get("patch_embedding.weight")
    pb = ck.opt("patch_embedding.bias")
    x = F.conv3d(x, pw, pb, stride=PATCH)
    if apply_control and "cond_encoder.weight" in ck.keys:
        # ComfyUI's node always feeds control_video = process_in(process_out(zeros)) == zeros: conv3d over
        # zeros adds just the cond_encoder bias to every token.
        control = torch.zeros((bs, 16, t_lat, height, width))
        x = x + F.conv3d(control, ck.get("cond_encoder.weight"), ck.opt("cond_encoder.bias"), stride=PATCH)

    t = torch.tensor([t_scalar], dtype=torch.float32).unsqueeze(1).repeat(1, x.shape[2])   # [1, gt]
    grid_sizes = tuple(x.shape[2:])
    x = x.flatten(2).transpose(1, 2)
    dump("post_patchify", x)
    seq_len = x.size(1)

    cond_mask_weight = ck.get("trainable_cond_mask.weight").unsqueeze(1).unsqueeze(1)      # [3,1,1,dim]
    x = x + cond_mask_weight[0]
    dump("post_condmask", x)

    freqs = rope_encode(t_lat, height, width)
    if reference_latent is not None:
        ref = F.conv3d(reference_latent, pw, pb, stride=PATCH).flatten(2).transpose(1, 2)
        freqs_ref = rope_encode(reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1],
                                t_start=max(30, t_lat + 9))
        ref = ref + cond_mask_weight[1]
        dump("ref_tokens", ref)
        x = torch.cat([x, ref], dim=1)
        dump("joined_tokens", x)
        freqs = torch.cat([freqs, freqs_ref], dim=1)
        t = torch.cat([t, torch.zeros((t.shape[0], reference_latent.shape[-3]))], dim=1)

    e = linear(F.silu(linear(sinusoidal_embedding_1d(FREQ_DIM, t.flatten()), ck, "time_embedding.0")),
               ck, "time_embedding.2")
    e = e.reshape(t.shape[0], -1, e.shape[-1])                                             # [1, G, dim]
    dump("temb", e)
    e0 = linear(F.silu(e), ck, "time_projection.1").unflatten(2, (6, DIM))                 # [1, G, 6, dim]
    dump("timestep_proj", e0)

    context = linear(F.gelu(linear(context, ck, "text_embedding.0"), approximate="tanh"), ck, "text_embedding.2")
    dump("text_proj", context)

    inj_map = {layer: i for i, layer in enumerate(INJECT_LAYERS)}
    for i in range(n_layers):
        t0 = time.time()
        x = block(ck, f"blocks.{i}", x, e0, freqs, context)
        if i in (0, n_layers // 2, n_layers - 1):
            dump(f"block_{i}", x)
        if i in inj_map:
            idx = inj_map[i]
            x = audio_injector(ck, idx, x, audio_emb, audio_emb_global, seq_len,
                               dump if idx in (0, len(INJECT_LAYERS) - 1) else None)
        print(f"  block {i}/{n_layers - 1} ({time.time() - t0:.1f}s)", flush=True)

    x = head(ck, x, e)
    dump("pre_unpatchify", x)
    v = unpatchify(x, grid_sizes)
    dump("velocity_out", v)
    return v


def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    position = position.float()
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="the C# WAN_DEBUG_DIR")
    ap.add_argument("--tag", default="cond", help="branch tag: cond or uncond")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=None, help="reference output dir (default {dump-dir}/ref)")
    ap.add_argument("--no-control", action="store_true", help="skip the cond_encoder zero-control bias add")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.dump_dir, "ref")
    ck = Ckpt(args.ckpt)
    tag = args.tag

    x = read_dump(args.dump_dir, f"{tag}_latent_in")
    context = read_dump(args.dump_dir, f"{tag}_text_embeds")
    audio = read_dump(args.dump_dir, f"{tag}_audio_features")
    tsteps = read_dump(args.dump_dir, f"{tag}_timesteps")
    ref = read_dump(args.dump_dir, "ref_latent")
    assert x is not None and context is not None and tsteps is not None, \
        f"missing {tag}_latent_in / {tag}_text_embeds / {tag}_timesteps in {args.dump_dir}"
    if context.dim() == 2:
        context = context.unsqueeze(0)
    assert audio is not None, f"missing {tag}_audio_features (C# dumps it from the pipeline)"
    audio_embed = audio.permute(1, 2, 0).unsqueeze(0)                  # [4T,25,1024] → [1,25,1024,4T]
    t_scalar = float(tsteps.flatten()[0])

    print(f"[s2v-ref] tag={tag} latent={list(x.shape)} text={list(context.shape)} audio={list(audio.shape)} "
          f"t={t_scalar:.3f} ref={'yes' if ref is not None else 'no'} control={not args.no_control}", flush=True)
    dump = Dumper(out_dir, tag)
    with torch.inference_mode():
        v = forward_orig(ck, x, t_scalar, context, audio_embed, ref, dump, apply_control=not args.no_control)
    print(f"[s2v-ref] velocity {list(v.shape)} mean={v.mean():.6f} std={v.std():.6f} → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
