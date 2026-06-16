"""
ACE-Step v1 DiT core — Python step-by-step parity reference.

Independent NumPy re-implementation of `AceStepDit.Forward` (latent + precomputed context -> velocity),
mirroring src/HartsyInference.Diffusion/Models/Denoisers/AceStepDit.cs. The C# diff test
(AceStepDitDiffTests) GENERATES the tiny synthetic checkpoint + fixed inputs and runs the C# forward with
ACE_STEP_DEBUG_DIR; this script loads the SAME weights+inputs, runs this reference, and dumps the same tap
points. diff_ace_step_layers.py compares. Validates the LiteLA linear attention, the interleaved-pair /
half-concat RoPE quirk, GLUMBConv, softmax cross-attn, AdaLN-6, patch-embed and final layer.

Run AFTER the C# test (which writes Output/ace_step_dit_parity/):
  python3 tests/python-reference/dump_ace_step_dit.py
"""
import json
import os
import numpy as np
from safetensors.numpy import load_file

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "Output", "ace_step_dit_parity")
INP = os.path.join(ROOT, "inputs")
REF = os.path.join(ROOT, "ref")
os.makedirs(REF, exist_ok=True)

with open(os.path.join(INP, "meta.json")) as f:
    M = json.load(f)
DIM = M["inner_dim"]; HEADS = M["num_heads"]; HD = M["head_dim"]; NL = M["num_layers"]
IN_CH = M["in_channels"]; LH = M["latent_height"]; PH = M["patch_h"]; PW = M["patch_w"]
FREQ = M["freq_dim"]; THETA = M["rope_theta"]; FFH = M["ff_hidden"]
F = M["F"]; L = M["L"]; T = M["timestep"]

W = {k: v.astype(np.float64) for k, v in load_file(os.path.join(ROOT, "transformer.safetensors")).items()}
latent = np.fromfile(os.path.join(INP, "latent.bin"), np.float32).astype(np.float64).reshape(1, IN_CH, LH, F)
context = np.fromfile(os.path.join(INP, "context.bin"), np.float32).astype(np.float64).reshape(L, DIM)

index = []
def tap(name, arr):
    arr.astype(np.float32).ravel().tofile(os.path.join(REF, f"{name.replace('.', '_')}.bin"))
    index.append(dict(name=name, shape=list(arr.shape), file=f"{name.replace('.', '_')}.bin"))

def linear(x, w, b=None):           # F.linear: x @ w.T + b, w is [out,in]
    y = x @ w.T
    return y + b if b is not None else y

def silu(x):
    return x / (1.0 + np.exp(-x))

def rms_no_affine(x):               # per-row 1/sqrt(mean(x^2)+1e-6)
    inv = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6)
    return x * inv

def build_rope(length):             # cos/sin [length, HD], half-concat layout
    half = HD // 2
    cos = np.zeros((length, HD)); sin = np.zeros((length, HD))
    for pos in range(length):
        for i in range(half):
            freq = THETA ** (-2.0 * i / HD)
            c, s = np.cos(pos * freq), np.sin(pos * freq)
            cos[pos, i] = cos[pos, half + i] = c
            sin[pos, i] = sin[pos, half + i] = s
    return cos, sin

def apply_rope(x, cos, sin):        # x [rows, HEADS, HD]; interleaved-pair rotation w/ half-concat tables
    out = x.copy()
    for pos in range(x.shape[0]):
        for h in range(HEADS):
            for d in range(HD // 2):
                a, b = x[pos, h, 2 * d], x[pos, h, 2 * d + 1]
                cA, sA = cos[pos, 2 * d], sin[pos, 2 * d]
                cB, sB = cos[pos, 2 * d + 1], sin[pos, 2 * d + 1]
                out[pos, h, 2 * d] = a * cA - b * sA
                out[pos, h, 2 * d + 1] = b * cB + a * sB
    return out

def patch_embed():
    w1 = W["proj_in.early_conv_layers.0.weight"]; b1 = W["proj_in.early_conv_layers.0.bias"]
    hid = w1.shape[0]
    # height-collapsing conv2d, stride (PH,PW), pad 0  ->  [hid, F]
    c1 = np.zeros((hid, F))
    for o in range(hid):
        for i in range(F):
            acc = b1[o]
            for ic in range(IN_CH):
                for kh in range(PH):
                    acc += latent[0, ic, kh, i] * w1[o, ic, kh, 0]
            c1[o, i] = acc
    # GroupNorm(min(32,hid), eps 1e-6): population var
    gw = W["proj_in.early_conv_layers.1.weight"]; gb = W["proj_in.early_conv_layers.1.bias"]
    groups = min(32, hid); cpg = hid // groups
    n = np.zeros_like(c1)
    for g in range(groups):
        sl = c1[g * cpg:(g + 1) * cpg]
        mean = sl.mean(); var = (sl * sl).mean() - mean * mean
        inv = 1.0 / np.sqrt(var + 1e-6)
        for c in range(g * cpg, (g + 1) * cpg):
            n[c] = (c1[c] - mean) * inv * gw[c] + gb[c]
    # 1x1 conv hid->dim
    w2 = W["proj_in.early_conv_layers.2.weight"][:, :, 0, 0]; b2 = W["proj_in.early_conv_layers.2.bias"]
    c2 = w2 @ n + b2[:, None]        # [dim, F]
    return c2.T                      # tokens [F, dim]

def time_embed():
    half = FREQ // 2
    sinT = np.zeros(FREQ)
    for i in range(half):
        f = np.exp(-np.log(10000.0) * i / half)
        sinT[i] = np.cos(T * f); sinT[half + i] = np.sin(T * f)
    h1 = silu(linear(sinT, W["timestep_embedder.linear_1.weight"], W["timestep_embedder.linear_1.bias"]))
    temb = linear(h1, W["timestep_embedder.linear_2.weight"], W["timestep_embedder.linear_2.bias"])
    tblock = linear(silu(temb), W["t_block.1.weight"], W["t_block.1.bias"])
    return temb, tblock

def litela(n, p, cos, sin):
    q = linear(n, W[f"{p}.attn.to_q.weight"], W[f"{p}.attn.to_q.bias"]).reshape(F, HEADS, HD)
    k = linear(n, W[f"{p}.attn.to_k.weight"], W[f"{p}.attn.to_k.bias"]).reshape(F, HEADS, HD)
    v = linear(n, W[f"{p}.attn.to_v.weight"], W[f"{p}.attn.to_v.bias"]).reshape(F, HEADS, HD)
    q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
    out = np.zeros((F, HEADS, HD))
    for h in range(HEADS):
        rk = np.maximum(k[:, h, :], 0.0)              # [F,HD]
        rq = np.maximum(q[:, h, :], 0.0)
        vk = v[:, h, :].T @ rk                          # [HD(dv), HD(dk)]
        vknorm = rk.sum(axis=0)                         # [HD(dk)]
        num = rq @ vk.T                                 # [F, HD(dv)]
        den = rq @ vknorm + 1e-15                       # [F]
        out[:, h, :] = num / den[:, None]
    o = out.reshape(F, DIM)
    return linear(o, W[f"{p}.attn.to_out.0.weight"], W[f"{p}.attn.to_out.0.bias"])

def cross_attn(x, p, cos, sin, cosC, sinC):
    q = apply_rope(linear(x, W[f"{p}.cross_attn.to_q.weight"], W[f"{p}.cross_attn.to_q.bias"]).reshape(F, HEADS, HD), cos, sin)
    k = apply_rope(linear(context, W[f"{p}.cross_attn.to_k.weight"], W[f"{p}.cross_attn.to_k.bias"]).reshape(L, HEADS, HD), cosC, sinC)
    v = linear(context, W[f"{p}.cross_attn.to_v.weight"], W[f"{p}.cross_attn.to_v.bias"]).reshape(L, HEADS, HD)
    out = np.zeros((F, HEADS, HD))
    scale = 1.0 / np.sqrt(HD)
    for h in range(HEADS):
        s = (q[:, h, :] @ k[:, h, :].T) * scale         # [F,L]
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s); a = e / e.sum(axis=1, keepdims=True)
        out[:, h, :] = a @ v[:, h, :]
    merged = out.reshape(F, DIM)
    return linear(merged, W[f"{p}.cross_attn.to_out.0.weight"], W[f"{p}.cross_attn.to_out.0.bias"])

def conv1d(x, w, b, pad, groups):   # x [Cin, F], w [Cout, Cin/groups, k]
    cout, cing, k = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    out = np.zeros((cout, F))
    gout = cout // groups
    for o in range(cout):
        g = o // gout
        for i in range(F):
            acc = b[o] if b is not None else 0.0
            for ci in range(cing):
                src = g * cing + ci
                for kk in range(k):
                    acc += xp[src, i + kk] * w[o, ci, kk]
            out[o, i] = acc
    return out

def glumb(n, p):
    cf = n.T                                            # [dim, F]
    inv = silu(conv1d(cf, W[f"{p}.ff.inverted_conv.conv.weight"], W[f"{p}.ff.inverted_conv.conv.bias"], 0, 1))
    k = W[f"{p}.ff.depth_conv.conv.weight"].shape[-1]
    depth = conv1d(inv, W[f"{p}.ff.depth_conv.conv.weight"], W[f"{p}.ff.depth_conv.conv.bias"], k // 2, inv.shape[0])
    hidden = depth.shape[0] // 2
    gated = depth[:hidden] * silu(depth[hidden:])
    out = conv1d(gated, W[f"{p}.ff.point_conv.conv.weight"], None, 0, 1)  # no bias
    return out.T                                        # [F, dim]

def block(x, i, cos, sin, cosC, sinC, tblock):
    p = f"transformer_blocks.{i}"
    ss = W[f"{p}.scale_shift_table"]                    # [6, dim]
    cur = x.copy()
    n1 = rms_no_affine(cur) * (1.0 + (ss[1] + tblock[1 * DIM:2 * DIM])) + (ss[0] + tblock[0 * DIM:1 * DIM])
    cur = cur + (ss[2] + tblock[2 * DIM:3 * DIM]) * litela(n1, p, cos, sin)
    cur = cur + cross_attn(cur, p, cos, sin, cosC, sinC)
    n2 = rms_no_affine(cur) * (1.0 + (ss[4] + tblock[4 * DIM:5 * DIM])) + (ss[3] + tblock[3 * DIM:4 * DIM])
    cur = cur + (ss[5] + tblock[5 * DIM:6 * DIM]) * glumb(n2, p)
    return cur

def final_layer(tokens, temb):
    fss = W["final_layer.scale_shift_table"]            # [2, dim]
    normed = rms_no_affine(tokens) * (1.0 + (fss[1] + temb)) + (fss[0] + temb)
    proj = linear(normed, W["final_layer.linear.weight"], W["final_layer.linear.bias"])  # [F, LH*IN_CH]
    out = np.zeros((1, IN_CH, LH, F))
    for i in range(F):
        for pp in range(LH):
            for ci in range(IN_CH):
                out[0, ci, pp, i] = proj[i, pp * IN_CH + ci]
    return out

def main():
    tokens = patch_embed(); tap("patch_embed", tokens)
    temb, tblock = time_embed(); tap("tblock", tblock)
    cos, sin = build_rope(F)
    cosC, sinC = build_rope(L)
    cur = tokens
    for i in range(NL):
        cur = block(cur, i, cos, sin, cosC, sinC, tblock)
        tap(f"layers.{i}", cur)
    vel = final_layer(cur, temb)
    tap("output_velocity", vel)
    with open(os.path.join(REF, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} ACE-Step DiT reference taps to {REF}")
    print(f"  velocity stats: mean={vel.mean():.6f} abs_mean={np.abs(vel).mean():.6f} std={vel.std():.6f}")

if __name__ == "__main__":
    main()
