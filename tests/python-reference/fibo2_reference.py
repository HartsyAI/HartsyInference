"""
Fibo 2.0 denoiser, pure-torch reference of the ComfyUI forward (comfy/ldm/fibo2/model.py, upstream snapshot
2026-10-21; see docs/Research/FIBO2_ARCHITECTURE.md). Writes a seeded tiny-config fixture with ComfyUI-native key
names and fused F32 shapes, plus the expected intermediates and velocity, for the C# parity test:

    tests/HartsyInference.Diffusion.Tests/Fixtures/Fibo2/fibo2_tiny.safetensors

Usage: /usr/bin/python3 tests/python-reference/fibo2_reference.py [out_path]

The per-token noisy/clean modulation of Comfy's modulate(timestep_zero_index=...) is written as a torch.where on a
boolean token mask: image tokens take the noisy embedding (t * 1000), gist tokens the clean one (t = 0).

Edit conditioning (ref_latents in Comfy's _forward) is transcribed the same way: the reference tokens follow the image tokens,
are clean (t = 0), sit on RoPE frame max_gist + 2 + k, run through the noise refiners with the image tokens, and the velocity is
read from the image tokens only. This is OUR reading of ComfyUI (87ed5a066d9c), not an independent check of it: the parity test
catches porting slips in the C# denoiser, not a misreading both sides share.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

TINY = dict(
    in_channels=32, patch=2, dim=64, heads=2, head_dim=32, layers=2, refiners=1, injection=(1,),
    cap_feat=96, perceiver_queries=8, perceiver_layers=1, ffn=96, axes=(8, 12, 12), theta=256.0,
    timestep_scale=1000.0, timestep_mlp=64, max_gist=8, min_gist=4, gist_step=2, gist_min_text=16,
    gist_max_text=64, norm_eps=1e-5,
)
LATENT_HW = (8, 12)
TEXT_LEN = 40
SIGMA = 0.6
SEED = 20261021

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO_ROOT, "tests", "HartsyInference.Diffusion.Tests", "Fixtures", "Fibo2", "fibo2_tiny.safetensors")


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def gist_count(text_len, c):
    ratio = min(1.0, max(0.0, (text_len - c["gist_min_text"]) / (c["gist_max_text"] - c["gist_min_text"])))
    raw = c["min_gist"] + ratio * (c["max_gist"] - c["min_gist"])
    return max(c["min_gist"], min(c["max_gist"], math.ceil(raw / c["gist_step"]) * c["gist_step"]))


def rope_pairs(ids, axes, theta):
    # Comfy flux rope() per axis, concatenated along the pair axis in axis order
    cos_parts, sin_parts = [], []
    for i, d in enumerate(axes):
        scale = torch.linspace(0, (d - 2) / d, steps=d // 2, dtype=torch.float64)
        omega = 1.0 / (theta ** scale)
        angle = ids[:, i].double()[:, None] * omega[None, :]
        cos_parts.append(torch.cos(angle))
        sin_parts.append(torch.sin(angle))
    return torch.cat(cos_parts, -1).float(), torch.cat(sin_parts, -1).float()


def apply_rope(x, cos, sin):
    # lumina apply_rope: interleaved pairs (2i, 2i+1) rotated by the same angle
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack([c * x0 - s * x1, s * x0 + c * x1], dim=-1).flatten(-2)


def sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2).flatten(2)


def timestep_embedding(t, dim=256, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = torch.tensor([t], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def patchify(latent):
    # diffusers _patchify / Flux2 pack: feature index = c * 4 + dy * 2 + dx
    b, ch, h, w = latent.shape
    x = latent.view(b, ch, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), ch * 4)


def unpatchify(tokens, gh, gw, ch):
    x = tokens.view(1, gh, gw, ch, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(1, ch, 2 * gh, 2 * gw)


def image_ids(c, gh, gw):
    frame = float(c["max_gist"] + 1)
    rows = torch.arange(gh, dtype=torch.float32).repeat_interleave(gw)
    cols = torch.arange(gw, dtype=torch.float32).repeat(gh)
    return torch.stack([torch.full_like(rows, frame), rows, cols], -1)


def ref_ids(c, k, gh, gw):
    frame = float(c["max_gist"] + 2 + k)
    rows = torch.arange(gh, dtype=torch.float32).repeat_interleave(gw)
    cols = torch.arange(gw, dtype=torch.float32).repeat(gh)
    return torch.stack([torch.full_like(rows, frame), rows, cols], -1)


def gist_ids(count):
    ids = torch.zeros(count, 3, dtype=torch.float32)
    ids[:, 0] = torch.arange(1, count + 1, dtype=torch.float32)
    return ids


def make_weights(c, g):
    dim, hd, ffn, cap = c["dim"], c["head_dim"], c["ffn"], c["cap_feat"]
    w = {}

    def lin(name, out_f, in_f, bias=False):
        w[name + ".weight"] = torch.randn(out_f, in_f, generator=g) / math.sqrt(in_f)
        if bias:
            w[name + ".bias"] = 0.02 * torch.randn(out_f, generator=g)

    def norm(name, n):
        w[name + ".weight"] = 1.0 + 0.1 * torch.randn(n, generator=g)

    def attn_fused(p):
        lin(p + "attention.qkv", 3 * dim, dim)
        lin(p + "attention.out", dim, dim)
        norm(p + "attention.q_norm", hd)
        norm(p + "attention.k_norm", hd)

    def cross(p, kv_dim):
        lin(p + "to_q", dim, dim)
        lin(p + "to_kv", 2 * dim, kv_dim)
        lin(p + "to_out", dim, dim)
        norm(p + "norm_q", hd)
        norm(p + "norm_k", hd)

    def swiglu(p):
        lin(p + "w13", 2 * ffn, dim)
        lin(p + "w2", dim, ffn)

    def block(p, injected):
        attn_fused(p)
        for n in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"):
            norm(p + n, dim)
        swiglu(p + "feed_forward.")
        lin(p + "adaLN_modulation.0", 4 * dim, 256, bias=True)
        if injected:
            norm(p + "cross_attn_norm1", dim)
            norm(p + "cross_attn_norm2", dim)
            w[p + "cross_attn_gate"] = 0.6 + 0.2 * torch.randn(1, generator=g)
            cross(p + "cross_attention.", cap)

    lin("x_embedder", dim, c["in_channels"] * c["patch"] ** 2, bias=True)
    lin("t_embedder.mlp.0", c["timestep_mlp"], 256, bias=True)
    lin("t_embedder.mlp.2", 256, c["timestep_mlp"], bias=True)
    for i in range(c["refiners"]):
        block(f"noise_refiner.{i}.", injected=False)
    for i in range(c["layers"]):
        block(f"layers.{i}.", injected=i in c["injection"])
    w["perceiver.latent_queries"] = 0.5 * torch.randn(c["perceiver_queries"], dim, generator=g)
    for j in range(c["perceiver_layers"]):
        p = f"perceiver.layers.{j}."
        norm(p + "cross_attn_norm", dim)
        cross(p + "cross_attention.", cap)
        norm(p + "self_attn_norm", dim)
        cross(p + "self_attention.", dim)
        norm(p + "ffn_norm", dim)
        swiglu(p + "ffn.")
    norm("perceiver.norm_out", dim)
    lin("final_layer.adaLN_modulation.1", dim, 256, bias=True)
    lin("final_layer.linear", c["in_channels"] * c["patch"] ** 2, dim, bias=True)
    return w


def t_embed(w, t, c):
    emb = timestep_embedding(t)
    hidden = F.silu(F.linear(emb, w["t_embedder.mlp.0.weight"], w["t_embedder.mlp.0.bias"]))
    return F.linear(hidden, w["t_embedder.mlp.2.weight"], w["t_embedder.mlp.2.bias"])


def swiglu(w, p, x):
    gate, up = F.linear(x, w[p + "w13.weight"]).chunk(2, -1)
    return F.linear(F.silu(gate) * up, w[p + "w2.weight"])


def cross_attn(w, p, q_in, kv_in, c):
    heads, hd, dim, eps = c["heads"], c["head_dim"], c["dim"], c["norm_eps"]
    b, sq, skv = q_in.shape[0], q_in.shape[1], kv_in.shape[1]
    q = F.linear(q_in, w[p + "to_q.weight"]).view(b, sq, heads, hd)
    k, v = F.linear(kv_in, w[p + "to_kv.weight"]).split(dim, -1)
    q = rms_norm(q, w[p + "norm_q.weight"], eps)
    k = rms_norm(k.reshape(b, skv, heads, hd), w[p + "norm_k.weight"], eps)
    v = v.reshape(b, skv, heads, hd)
    return F.linear(sdpa(q, k, v), w[p + "to_out.weight"])


def fused_self_attn(w, p, h, cos, sin, c):
    heads, hd, dim, eps = c["heads"], c["head_dim"], c["dim"], c["norm_eps"]
    b, s = h.shape[0], h.shape[1]
    q, k, v = F.linear(h, w[p + "attention.qkv.weight"]).split(dim, -1)
    q = rms_norm(q.reshape(b, s, heads, hd), w[p + "attention.q_norm.weight"], eps)
    k = rms_norm(k.reshape(b, s, heads, hd), w[p + "attention.k_norm.weight"], eps)
    v = v.reshape(b, s, heads, hd)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    return F.linear(sdpa(q, k, v), w[p + "attention.out.weight"])


def block(w, p, x, emb_clean, emb_noisy, is_noisy, cos, sin, text, c):
    eps = c["norm_eps"]

    def modulation(emb):
        return F.linear(emb, w[p + "adaLN_modulation.0.weight"], w[p + "adaLN_modulation.0.bias"]).squeeze(0)

    m = torch.where(is_noisy[:, None], modulation(emb_noisy)[None], modulation(emb_clean)[None])
    s_msa, g_msa, s_mlp, g_mlp = m.chunk(4, -1)
    g_msa, g_mlp = torch.tanh(g_msa)[None], torch.tanh(g_mlp)[None]

    h = rms_norm(x, w[p + "attention_norm1.weight"], eps) * (1 + s_msa)[None]
    a = fused_self_attn(w, p, h, cos, sin, c)
    x = x + g_msa * rms_norm(a, w[p + "attention_norm2.weight"], eps)
    if text is not None:
        cr = cross_attn(w, p + "cross_attention.", rms_norm(x, w[p + "cross_attn_norm1.weight"], eps), text, c)
        x = x + torch.tanh(w[p + "cross_attn_gate"]) * rms_norm(cr, w[p + "cross_attn_norm2.weight"], eps)
    h2 = rms_norm(x, w[p + "ffn_norm1.weight"], eps) * (1 + s_mlp)[None]
    f = swiglu(w, p + "feed_forward.", h2)
    return x + g_mlp * rms_norm(f, w[p + "ffn_norm2.weight"], eps)


def perceiver(w, ctx, count, c):
    eps = c["norm_eps"]
    x = w["perceiver.latent_queries"][:count][None]
    for j in range(c["perceiver_layers"]):
        p = f"perceiver.layers.{j}."
        x = x + cross_attn(w, p + "cross_attention.", rms_norm(x, w[p + "cross_attn_norm.weight"], eps), ctx, c)
        y = rms_norm(x, w[p + "self_attn_norm.weight"], eps)
        x = x + cross_attn(w, p + "self_attention.", y, y, c)
        x = x + swiglu(w, p + "ffn.", rms_norm(x, w[p + "ffn_norm.weight"], eps))
    return rms_norm(x, w["perceiver.norm_out.weight"], eps)


def final_layer(w, img, emb_noisy, c):
    scale = F.linear(F.silu(emb_noisy), w["final_layer.adaLN_modulation.1.weight"], w["final_layer.adaLN_modulation.1.bias"])
    normed = F.layer_norm(img, (c["dim"],), eps=1e-6)
    return F.linear(normed * (1 + scale)[:, None, :], w["final_layer.linear.weight"], w["final_layer.linear.bias"])


def forward(w, c, latent, sigma, context, refs=()):
    gh, gw = latent.shape[2] // 2, latent.shape[3] // 2
    n = gh * gw
    count = gist_count(context.shape[1], c)
    emb_clean = t_embed(w, 0.0, c)
    emb_noisy = t_embed(w, sigma * c["timestep_scale"], c)

    # image tokens, then each reference (Comfy: torch.cat(tokens, dim=1) before x_embedder)
    img = F.linear(torch.cat([patchify(latent)] + [patchify(r) for r in refs], dim=1), w["x_embedder.weight"], w["x_embedder.bias"])
    ids_list = [image_ids(c, gh, gw)] + [ref_ids(c, k, r.shape[2] // 2, r.shape[3] // 2) for k, r in enumerate(refs)]
    ids_img = torch.cat(ids_list)
    cos_i, sin_i = rope_pairs(ids_img, c["axes"], c["theta"])
    n_all = ids_img.shape[0]
    refiner_noisy = torch.zeros(n_all, dtype=torch.bool)
    refiner_noisy[:n] = True
    for i in range(c["refiners"]):
        img = block(w, f"noise_refiner.{i}.", img, emb_clean, emb_noisy, refiner_noisy, cos_i, sin_i, None, c)
    refined = img

    bundles = context.chunk(1 + len(c["injection"]), dim=-1)
    gist = perceiver(w, bundles[0], count, c)
    cos, sin = rope_pairs(torch.cat([gist_ids(count), ids_img]), c["axes"], c["theta"])
    x = torch.cat([gist, img], dim=1)
    is_noisy = torch.cat([torch.zeros(count, dtype=torch.bool), refiner_noisy])
    text = {layer: bundles[j + 1] for j, layer in enumerate(c["injection"])}
    for i in range(c["layers"]):
        x = block(w, f"layers.{i}.", x, emb_clean, emb_noisy, is_noisy, cos, sin, text.get(i), c)

    out = final_layer(w, x[:, count:count + n], emb_noisy, c)
    velocity = unpatchify(out, gh, gw, c["in_channels"])
    return velocity, gist, refined


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    g = torch.Generator().manual_seed(SEED)
    weights = make_weights(TINY, g)
    latent = torch.randn(1, TINY["in_channels"], *LATENT_HW, generator=g)
    context = 0.5 * torch.randn(1, TEXT_LEN, TINY["cap_feat"] * (1 + len(TINY["injection"])), generator=g)

    # Drawn after the text-to-image inputs so those tensors keep their seeded values.
    ref0 = torch.randn(1, TINY["in_channels"], 6, 4, generator=g)
    ref1 = torch.randn(1, TINY["in_channels"], 4, 8, generator=g)

    with torch.no_grad():
        velocity, _, _ = forward(weights, TINY, latent, SIGMA, context)
        velocity_ref1, _, _ = forward(weights, TINY, latent, SIGMA, context, [ref0])
        velocity_ref2, _, _ = forward(weights, TINY, latent, SIGMA, context, [ref0, ref1])

    tensors = {f"weights.{k}": v.contiguous() for k, v in weights.items()}
    tensors["inputs.latent"] = latent.contiguous()
    tensors["inputs.sigma"] = torch.tensor([SIGMA], dtype=torch.float32)
    tensors["inputs.context"] = context.contiguous()
    tensors["expected.velocity"] = velocity.contiguous()
    tensors["inputs.ref0"] = ref0.contiguous()
    tensors["inputs.ref1"] = ref1.contiguous()
    tensors["expected.velocity_ref1"] = velocity_ref1.contiguous()
    tensors["expected.velocity_ref2"] = velocity_ref2.contiguous()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file(tensors, out_path)
    print(f"wrote {out_path}: {len(weights)} weights, velocity {tuple(velocity.shape)}")


if __name__ == "__main__":
    main()
