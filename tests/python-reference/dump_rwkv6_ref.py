"""RWKV-6 reference from the GGUF weights. VALIDATED: matches the official `rwkv` pip package (loading the .pth on
cpu fp32) at cosine = 1.000000, argmax 281, for ids [33155,295,281,4844] on rwkv-6-world-1.6b.

This is the ground truth the C# `RwkvModel` is ported from. The key non-obvious detail: llama.cpp's GGUF
PRE-DIVIDES the deep-layer output weights by 2^(layer//rescale_every_n_layers) (blk.6 → /2, blk.12 → /4, …), so
you must apply the matching runtime `x *= 0.5` every `rescale_every_n_layers` layers (the fp16-stability trick).

Reference setup:  pip install rwkv ; hf download BlinkDL/rwkv-6-world <pth> ; ground truth via rwkv.model.RWKV.
Usage:            python dump_rwkv6_ref.py /path/to/rwkv-6-world-1.6b-F32.gguf
"""
import sys, numpy as np
from gguf import GGUFReader
path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rwkv/rwkv-6-world-1.6b-F32.gguf'
r = GGUFReader(path); W = {t.name: t.data.astype(np.float32) for t in r.tensors}
D, L, H, N = 2048, 24, 32, 64; EPS = 1e-5; RESCALE = 6
ids = [33155, 295, 281, 4844]; T = len(ids)
def ln(x, w, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + EPS) * w + b
def lin(x, name): return x @ W[name].T   # gguf data = [out, in] (reverse-ne row-major) → y = x @ dataᵀ
emb = W['token_embd.weight']             # (vocab, D)
x = np.stack([emb[i] for i in ids])
x = ln(x, W['token_embd_norm.weight'], W['token_embd_norm.bias'])
def gn(o, w, b):
    o = o.reshape(T, H, N); m = o.mean(-1, keepdims=True); v = o.var(-1, keepdims=True)
    return ((o - m) / np.sqrt(v + 64e-5)).reshape(T, D) * w + b
for il in range(L):
    p = f'blk.{il}.'
    # time mix
    xx = ln(x, W[p + 'attn_norm.weight'], W[p + 'attn_norm.bias'])
    sx = np.concatenate([np.zeros((1, D), np.float32), xx[:-1]]) - xx
    xxx = np.tanh((xx + sx * W[p + 'time_mix_lerp_x.weight']) @ W[p + 'time_mix_w1.weight'].T).reshape(T, 5, -1).transpose(1, 0, 2)
    w2 = W[p + 'time_mix_w2.weight'].reshape(5, D, 32).transpose(0, 2, 1)   # [5, lora, D]
    mw, mk, mv, mr, mg = np.matmul(xxx, w2)                                 # [5, T, D]
    wx = xx + sx * (W[p + 'time_mix_lerp_w.weight'] + mw); kx = xx + sx * (W[p + 'time_mix_lerp_k.weight'] + mk)
    vx = xx + sx * (W[p + 'time_mix_lerp_v.weight'] + mv); rx = xx + sx * (W[p + 'time_mix_lerp_r.weight'] + mr)
    gx = xx + sx * (W[p + 'time_mix_lerp_g.weight'] + mg)
    rr = lin(rx, p + 'time_mix_receptance.weight').reshape(T, H, N)
    kk = lin(kx, p + 'time_mix_key.weight').reshape(T, H, N)
    vv = lin(vx, p + 'time_mix_value.weight').reshape(T, H, N)
    g = lin(gx, p + 'time_mix_gate.weight'); g = g / (1 + np.exp(-g))
    wdec = W[p + 'time_mix_decay.weight'].reshape(H, N)[None] + (np.tanh(wx @ W[p + 'time_mix_decay_w1.weight'].T) @ W[p + 'time_mix_decay_w2.weight'].T).reshape(T, H, N)
    wdec = np.exp(-np.exp(wdec))
    u = W[p + 'time_mix_first.weight'].reshape(H, N)
    S = np.zeros((H, N, N), np.float32); out = np.zeros((T, H, N), np.float32)
    for t in range(T):
        at = kk[t][:, :, None] @ vv[t][:, None, :]                          # [H,N,N] outer product
        out[t] = (rr[t][:, None, :] @ (u[:, :, None] * at + S)).squeeze(1)
        S = at + wdec[t][:, :, None] * S
    o = gn(out.reshape(T, D), W[p + 'time_mix_ln.weight'], W[p + 'time_mix_ln.bias']) * g
    x = x + lin(o, p + 'time_mix_output.weight')
    # channel mix
    xx = ln(x, W[p + 'attn_norm_2.weight'], W[p + 'attn_norm_2.bias'])
    sx = np.concatenate([np.zeros((1, D), np.float32), xx[:-1]]) - xx
    k = np.maximum(lin(xx + sx * W[p + 'channel_mix_lerp_k.weight'], p + 'channel_mix_key.weight'), 0) ** 2
    x = x + (1 / (1 + np.exp(-lin(xx + sx * W[p + 'channel_mix_lerp_r.weight'], p + 'channel_mix_receptance.weight')))) * lin(k, p + 'channel_mix_value.weight')
    if (il + 1) % RESCALE == 0: x = x * 0.5      # compensate GGUF's pre-divided deep-layer output weights
x = ln(x, W['output_norm.weight'], W['output_norm.bias'])
logits = lin(x[-1:], 'output.weight')[0]
print('argmax=', int(logits.argmax()), 'logit=', round(float(logits.max()), 4))   # expect 281
