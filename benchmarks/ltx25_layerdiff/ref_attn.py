"""Reference dump of det_stages[0][0].attn internals, to bisect the divergence inside the attention module.

Reproduces NeighborhoodAttention3D.forward step by step (no chunking: chunk >= t at this geometry) so each
intermediate can be compared against the C# taps of the same names.
"""
import os, sys, json, struct
import numpy as np
import torch

COMFY = "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI"
sys.path.insert(0, COMFY)
from comfy.ldm.lightricks.vae.na_diffusion_decoder import (
    NADiffusionDecoder, rope_inv_freqs, _rope_tables, _rope_matrices_slice)
import comfy_kitchen

OUT = sys.argv[1]
REF = sys.argv[2]          # the main dump, for conv_in (this module's input)
CKPT = "/home/hartsy/Desktop/HartsyInference/Models/VAE/LTX-2/ltx-2.5-video-vae-bf16.safetensors"
os.makedirs(OUT, exist_ok=True)
meta = {}

def save(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    a.tofile(os.path.join(OUT, name + ".f32"))
    meta[name] = list(a.shape)
    print(f"  {name:16s} {str(list(a.shape)):30s} mean {a.mean():+.6f} std {a.std():.6f}")

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); base = 8 + n; out = {}
        for k, v in hdr.items():
            if k == "__metadata__": continue
            s, e = v["data_offsets"]; f.seek(base + s); raw = f.read(e - s)
            out[k] = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(v["shape"]).float()
        return out

state = {k[len("decoder."):]: v for k, v in load_safetensors(CKPT).items() if k.startswith("decoder.")}
dec = NADiffusionDecoder()
missing, unexpected = dec.load_state_dict(state, strict=False)
assert not missing and list(unexpected) == ["type_emb"]
dec = dec.float().eval()
print("load OK")

shapes = json.load(open(os.path.join(REF, "shapes.json")))
ci = np.fromfile(os.path.join(REF, "conv_in.f32"), dtype=np.float32).reshape(shapes["conv_in"])
x = torch.from_numpy(ci)

blk = dec.det_stages[0][0]
attn = blk.attn
with torch.no_grad():
    xin = blk.norm1(x)                       # the block applies norm1 as `pre`
    save("attn_in", xin.numpy())
    b, t, h, w, _ = xin.shape
    qkv = attn.qkv(xin)
    save("attn_qkv", qkv.reshape(t * h * w, -1).numpy())

    shape = (b, t, h, w, attn.num_heads, attn.head_dim)
    q, k, v = [c.reshape(shape).clone() for c in qkv.chunk(3, dim=-1)]
    save("attn_v", v.numpy())

    inv = tuple(rope_inv_freqs(d, attn.rope_base, device=xin.device) for d in attn.rope_split)
    tables = _rope_tables((t, h, w), inv, xin.device)
    freqs = _rope_matrices_slice(tables, 0, t, h, w)
    qw = (attn.q_norm.weight.detach() * attn.scale).float()
    kw = attn.k_norm.weight.detach().float()

    # rms only, to compare against the C# attn_qnorm tap (which is post-RmsNorm, pre-rope)
    from comfy.ldm.lightricks.vae.na_diffusion_decoder import rms_norm
    save("attn_qnorm", rms_norm(q, qw, 1e-6).numpy())

    nt = t * h * w
    comfy_kitchen.rms_rope_(q[0].view(1, nt, attn.num_heads, attn.head_dim),
                            k[0].view(1, nt, attn.num_heads, attn.head_dim), freqs, qw, kw)
    save("attn_qrope", q.numpy())
    save("attn_krope", k.numpy())

    out = comfy_kitchen.na3d(q, k, v, list(attn.kernel_size), None, 1.0)
    save("attn_na3d", out.numpy())
    proj = attn.proj(out.reshape(b, t, h, w, attn.dim))
    save("attn_out", proj.reshape(t * h * w, -1).numpy())
    resid = (x + proj)
    save("blk_resid", resid.reshape(t*h*w,-1).numpy())
    n2 = blk.norm2(resid)
    save("blk_norm2", n2.reshape(t*h*w,-1).numpy())
    save("blk_mlp", blk.mlp(n2).reshape(t*h*w,-1).numpy())
    # full block 0, then all four stage-0 blocks
    y = x.clone()
    y = blk(y)
    save("stage0_block0", y.reshape(-1, y.shape[-1]).numpy())
    for bi in range(1, len(dec.det_stages[0])):
        y = dec.det_stages[0][bi](y)
        save(f"stage0_block{bi}", y.reshape(-1, y.shape[-1]).numpy())

json.dump(meta, open(os.path.join(OUT, "shapes.json"), "w"), indent=1)
print("done ->", OUT)
